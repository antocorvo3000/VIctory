//! Title memory by address: what a shader finds behind a 64-bit address
//! not read from a descriptor (`victory_shader::emit::SLOT_MEMORY`,
//! `SLOT_FAULTS`).
//!
//! Ray-tracing shaders walk node trees whose pointers lie inside the tree's
//! buffer or cross into other buffers, and builders read and write scene
//! data through bare addresses they build from constants and buffer loads.
//! Nothing in D3D12 lets us hand a shader the title's bare addresses (host
//! copies sit where the host put them, not there), so the translator looks
//! every 64 KiB page of title address space up in a table we give the shader:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; changing a mapped page's owner gives subsequent bindings a new
//!   directory while recorded command lists keep their original directory;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: a buffer where the shader writes the page numbers of
//!   accesses that found no copy, which we read back afterwards to give
//!   those pages their copies for the frames to come.
//!
//! When a shader asks for a page for the first time, it reads zeros and
//! its writes are dropped, and the fault note asks for the page for the
//! next frames.
//!
//! The copies are the host copies [`crate::shadow`] keeps of the title's
//! buffers and 4 MiB windows: the shader sees the same copies that normal
//! draws and dispatches read and write. One page can belong to a copy at a
//! time; pages that hold part of a buffer whose host copy is upload memory
//! (no unordered access view) are left unmapped.
//!
//! What the shader gets a fault note for is mapped at the end of the frame:
//! the directory entry gets the copy's number and offset, the copy gets a
//! raw view in the heap (once per copy, on first use) and is brought up to
//! date with its ranges. Wait-free: a frame that needs a page from the
//! title gets it by the next frame.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! each view heap grows its descriptor run as more copies are needed.
//! Every dispatch of such a shader brings its addressable copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

#[cfg(feature = "diagnostics")]
pub(crate) mod ray_probe;

thread_local! {
    static TEST_RAY_MEMORY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_ray_memory<T>(on: bool, work: impl FnOnce() -> T) -> T {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_RAY_MEMORY.set(self.0);
        }
    }
    let _restore = Reset(TEST_RAY_MEMORY.replace(Some(on)));
    work()
}

/// Whether ray-tracing memory directory tracking, writeback ownership checks,
/// descriptor address tables, and extended retired-object lifetimes are active:
/// `VICTORY_RAY_MEMORY=1` or `VICTORY_RAYS=1` (default `0`).
pub(crate) fn ray_memory() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    if let Some(v) = TEST_RAY_MEMORY.get() {
        return v;
    }
    *ON.get_or_init(|| {
        fn flag(name: &str) -> Option<bool> {
            let v = std::env::var(name).ok()?;
            match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "on" | "yes" => Some(true),
                "0" | "false" | "off" | "no" => Some(false),
                _ => None,
            }
        }
        flag("VICTORY_RAY_MEMORY")
            .or_else(|| flag("VICTORY_RAYS"))
            .unwrap_or(false)
    })
}

/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

// A title may register buffers while another thread is recording GPU work.
// Record only the changed spans. A lagging reader that misses journal history
// rechecks its resident pages. Plain rendering pays no journal cost until a
// shader has used the raw-address directory.
static CHANGES: OnceLock<Mutex<VecDeque<(u64, u64)>>> = OnceLock::new();
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub(crate) fn ownership_changed(start: u64, len: u64) {
    if let Some(changes) = CHANGES.get() {
        let mut changes = changes.lock().unwrap_or_else(|e| e.into_inner());
        if changes.len() == 256 {
            changes.pop_front();
        }
        changes.push_back((start / PAGE, start.saturating_add(len).div_ceil(PAGE)));
        GENERATION.fetch_add(1, Ordering::Release);
    }
}

/// Byte index used by translated shaders in a raw UAV starting at byte zero
/// of the host resource. An in-place resource can begin before the guest
/// allocation, and a directory entry must cover its complete 64 KiB page.
fn page_byte(page_start: u64, first: u64, len: u64, prefix: u64) -> Option<u32> {
    if page_start < first || len < PAGE || page_start - first > len - PAGE {
        return None;
    }
    let byte = prefix.checked_add(page_start - first)?;
    u32::try_from(byte).ok()
}

struct Memory {
    /// Directory buffer, its mapped u32 pointer, and its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    /// Fault buffer, likewise.
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Shadow IDs are unique for the lifetime of a backing, including after
    /// the guest reuses its address.
    numbers: HashMap<u64, usize>,
    pages: BTreeMap<u64, usize>,
    ranges: Vec<Vec<(u64, u64)>>,
    recalculate: BTreeSet<usize>,
    generation: u64,
    coverage_revision: u64,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Typed locations already considered for this heap snapshot this frame.
    acceleration: Option<Arc<[u64]>>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// `MEMORY`'s mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What [`ready`] returns for the draw's constants and root bindings.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
    /// Last numbered copy this draw prepared. Other recording threads can
    /// subsequently add mappings to the same directory.
    pub(crate) count: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        if ray_memory() {
            CHANGES.get_or_init(Default::default);
        }
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        let mut fault_dwords = MEMORY_FAULTS;
        if victory_shader::emit::ray_probe_enabled() {
            fault_dwords += victory_shader::emit::RAY_PROBE_DWORDS;
        }
        Some(Memory {
            directory: Self::table(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: Self::table(u64::from(fault_dwords), "faults")?,
            copies: Vec::new(),
            numbers: HashMap::new(),
            pages: BTreeMap::new(),
            ranges: Vec::new(),
            recalculate: BTreeSet::new(),
            generation: GENERATION.load(Ordering::Acquire),
            coverage_revision: 0,
            frame: u64::MAX,
            refused: Default::default(),
            acceleration: None,
        })
    }

    fn table(dwords: u64, what: &str) -> Option<(ID3D12Resource, *mut u32, u64)> {
        let (buffer, data) = host()?.shared_buffer(dwords * 4)?;
        crate::host::name(&buffer, || format!("title memory by address: {what}"));
        // SAFETY: a fresh mapped buffer of the requested length.
        unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Some((buffer, data.cast(), gpu))
    }

    fn page_owner(start: u64) -> Option<(Arc<Shadow>, u32)> {
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            return None;
        }
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let shadow = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        let (first, len) = shadow.span();
        let offset = page_byte(start, first, len, shadow.offset(first))?;
        Some((shadow, offset))
    }

    /// Assign stable slots. A previously recorded list may still reference any
    /// old slot, even after the corresponding guest address has been reused.
    fn number(&mut self, shadow: &Arc<Shadow>) -> Option<usize> {
        if ray_memory() {
            if let Some(&index) = self.numbers.get(&shadow.id()) {
                return Some(index);
            }
            // The directory stores a one-based u32 slot; zero is the null view.
            u32::try_from(self.copies.len()).ok()?.checked_add(1)?;
            self.numbers.insert(shadow.id(), self.copies.len());
            self.copies.push(shadow.clone());
            self.ranges.push(Vec::new());
            Some(self.copies.len() - 1)
        } else {
            if let Some(index) = self.copies.iter().position(|copy| Arc::ptr_eq(copy, shadow)) {
                return Some(index);
            }
            if self.copies.len() >= 1023 {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        copies = 1023,
                        "title memory by address: no number left for another copy; its pages read zeros"
                    )
                });
                return None;
            }
            self.copies.push(shadow.clone());
            self.ranges.push(Vec::new());
            Some(self.copies.len() - 1)
        }
    }

    fn set_page(&mut self, page: u64, mapping: Option<(usize, u32)>) {
        let (slot, offset) = if let Some((index, offset)) = mapping {
            self.pages.insert(page, index);
            self.recalculate.insert(index);
            (index as u32 + 1, offset)
        } else {
            self.pages.remove(&page);
            (0, 0)
        };
        // SAFETY: page is inside the table. New assignments replace existing
        // ones only in a private directory; first-time assignments publish
        // the offset before the nonzero slot.
        unsafe {
            let entry = self.directory.1.add(page as usize * 2);
            entry.add(1).write_volatile(offset);
            entry.write_volatile(slot);
        }
    }

    fn reconcile(&mut self) -> Option<()> {
        if !ray_memory() || self.generation == GENERATION.load(Ordering::Acquire) {
            return Some(());
        }
        let mut check = BTreeSet::new();
        let generation = {
            let changes = CHANGES.get()?.lock().unwrap_or_else(|e| e.into_inner());
            let now = GENERATION.load(Ordering::Relaxed);
            let count = now - self.generation;
            if count > changes.len() as u64 {
                check.extend(self.pages.keys().copied());
            } else {
                for &(first, end) in changes.iter().rev().take(count as usize) {
                    check.extend(self.pages.range(first..end).map(|(&page, _)| page));
                }
            }
            now
        };
        let mut replaced = false;
        for page in check {
            let old = self.pages[&page];
            let owner = Self::page_owner(page * PAGE);
            if owner
                .as_ref()
                .is_some_and(|(s, _)| Arc::ptr_eq(s, &self.copies[old]))
            {
                continue;
            }
            if !replaced {
                let next = Self::table(u64::from(MEMORY_PAGES) * 2, "directory")?;
                // SAFETY: equal-sized disjoint tables. The CPU records under
                // the memory mutex and all GPU directory accesses are reads.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.directory.1,
                        next.1,
                        MEMORY_PAGES as usize * 2,
                    )
                };
                self.directory = next;
                replaced = true;
                self.coverage_revision += 1;
            }
            let mapping = owner.and_then(|(s, offset)| self.number(&s).map(|n| (n, offset)));
            self.recalculate.insert(old);
            self.set_page(page, mapping);
        }
        self.generation = generation;
        self.refused.clear();
        self.acceleration = None;
        Some(())
    }

    fn update_ranges(&mut self) {
        for index in std::mem::take(&mut self.recalculate) {
            let (start, len) = self.copies[index].span();
            let ranges = &mut self.ranges[index];
            ranges.clear();
            for (&page, &owner) in self
                .pages
                .range(start.div_ceil(PAGE)..(start + len) / PAGE)
            {
                if owner != index {
                    continue;
                }
                let at = page * PAGE;
                if let Some((first, bytes)) = ranges.last_mut()
                    && *first + *bytes == at
                {
                    *bytes += PAGE;
                } else {
                    ranges.push((at, PAGE));
                }
            }
        }
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        let start = page * PAGE;
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        if !ray_memory() {
            // SAFETY: inside the directory's mapping.
            let entry = unsafe { self.directory.1.add(page as usize * 2) };
            // SAFETY: reading one dword from the directory mapping.
            if unsafe { entry.read() } != 0 {
                return;
            }
        } else if self.pages.contains_key(&page) {
            return;
        }
        if self.refused.contains(&page) {
            return;
        }
        let Some((shadow, offset)) = Self::page_owner(start) else {
            self.refused.insert(page);
            return;
        };
        if !ray_memory() {
            let (first, len) = shadow.span();
            let prefix = shadow.offset(first);
            let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
                Some(n) => n,
                None => {
                    let Some(n) = self.number(&shadow) else { return };
                    let mut at = first.next_multiple_of(PAGE);
                    while at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                        let Some(byte) = page_byte(at, first, len, prefix) else { break };
                        let other_page = at / PAGE;
                        if !self.pages.contains_key(&other_page) {
                            self.set_page(other_page, Some((n, byte)));
                        }
                        at += PAGE;
                    }
                    n
                }
            };
            self.set_page(page, Some((n, offset)));
            return;
        }
        let count = self.copies.len();
        let Some(n) = self.number(&shadow) else { return };
        if n < count {
            // An existing copy regained a page that earlier binds did not cover.
            self.coverage_revision += 1;
            self.set_page(page, Some((n, offset)));
            return;
        }
        // A builder can follow an indirect address to any full page in this
        // buffer. Prefer a nested resource already registered at that page.
        let (first, len) = shadow.span();
        let end = (first + len) / PAGE;
        for p in first.div_ceil(PAGE)..end.min(u64::from(MEMORY_PAGES) - 1) {
            if self.pages.contains_key(&p) {
                continue;
            }
            if p == page {
                self.set_page(p, Some((n, offset)));
            } else if let Some((owner, off)) = Self::page_owner(p * PAGE)
                && let Some(number) = self.number(&owner)
            {
                self.set_page(p, Some((number, off)));
            }
        }
    }

    /// Reads the pages shaders found without copies and gives them theirs.
    fn take_faults(&mut self) {
        let faults = self.faults.1;
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside `faults`.
            let page = unsafe { faults.add(i).replace(0) };
            if page != 0 {
                self.map(u64::from(page) * PAGE);
            }
        }
    }

    /// Gives the window holding `address` (and the next window) their
    /// copies: the title puts a view on its root, and the shader walks on
    /// into the next pages.
    fn name(&mut self, address: u64) {
        if address < PAGE {
            return;
        }
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }

    /// A null-resource acceleration-structure SRV supplies a definite guest
    /// address even when no shader root directly names the structure. It
    /// gives no size, so request only the cell containing its header.
    fn map_acceleration_structures(&mut self, heap: &HostHeap) {
        let locations = heap.acceleration_addresses();
        if self
            .acceleration
            .as_ref()
            .is_some_and(|last| Arc::ptr_eq(last, &locations))
        {
            return;
        }
        for &address in locations.iter() {
            self.map(address);
        }
        self.acceleration = Some(locations);
    }

    fn prepare_addresses(&mut self, heap: &HostHeap, addresses: &[u64], constants: &[u32]) {
        if ray_memory() {
            self.map_acceleration_structures(heap);
        }
        for &address in addresses {
            self.name(address);
        }
        // A title address in two constants (low dword, then bits 32 to 39), or
        // in one in 256-byte units. A constant that only looks like one names
        // memory nothing reads; one that is not the title's memory is refused.
        for (i, &c) in constants.iter().enumerate() {
            if let Some(&high) = constants.get(i + 1)
                && (1..=0xFF).contains(&high)
                && c >= 0x1_0000
            {
                self.name(u64::from(c) | u64::from(high) << 32);
            }
            if c >= 0x0100_0000 {
                self.name(u64::from(c) << 8);
            }
        }
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// shader reads of ours (`Ready`), with `heap`'s views and `list`'s copies
/// prepared (`writes` if the shader may write).
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    m.reconcile()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.acceleration = None;
        m.take_faults();
    }
    m.prepare_addresses(heap, addresses, constants);
    if ray_memory() {
        m.update_ranges();
        let (base, count) = heap.raw_page_views(&m.copies)?;
        // A list may submit before the other recording list that numbered a
        // copy. Each must prepare all copies visible at its own bind.
        const READ: usize = usize::MAX - 0x6d;
        let (reads, written) =
            list.raw_memory_ranges(&m.directory.0, m.coverage_revision, count as usize, writes);
        for i in reads {
            for &(start, len) in &m.ranges[i] {
                m.copies[i].refresh(list, start, len);
            }
        }
        for i in written {
            for &(start, len) in &m.ranges[i] {
                m.copies[i].gpu_writes(list, start, len);
            }
        }
        if list.first_use(READ) {
            list.keep(&m.faults.0);
        }
        Some(Ready {
            directory: m.directory.2,
            faults: m.faults.2,
            base,
            count,
        })
    } else {
        m.recalculate.clear();
        let (base, count) = heap.raw_page_views(&m.copies)?;
        const READ: usize = usize::MAX - 0x6d;
        let (reads, written) =
            list.raw_memory_ranges(&m.directory.0, m.coverage_revision, count as usize, writes);
        for shadow in &m.copies[reads] {
            let (start, len) = shadow.span();
            shadow.refresh(list, start, len);
        }
        for shadow in &m.copies[written] {
            let (start, len) = shadow.span();
            shadow.gpu_writes(list, start, len);
        }
        if list.first_use(READ) {
            list.keep(&m.faults.0);
        }
        Some(Ready {
            directory: m.directory.2,
            faults: m.faults.2,
            base,
            count,
        })
    }
}

#[cfg(test)]
#[path = "memory/producer_tests.rs"]
mod producer_tests;

#[cfg(test)]
mod tests {
    use super::{CELL, Memory, PAGE, page_byte};

    #[test]
    fn raw_builder_maps_headers_named_by_the_view_heap_without_a_shader_table() {
        crate::memory::with_ray_memory(true, || {
            use crate::descriptor::{create_srv, host::HostHeap};
            use std::ptr::null;
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            };

            let Some(heap) = HostHeap::new(false, 4) else {
                assert!(
                    std::env::var_os("VICTORY_D3D12_WARP").is_none(),
                    "requested WARP view heap unavailable"
                );
                return;
            };
            let mut memory = Memory::new().unwrap();
            // Reserve three 4 MiB cells in guest address space. Only the first
            // page of each is a title buffer; the middle one has no AS descriptor.
            let start = (0x1300..0x1400)
                .map(|i| i * 4 * CELL)
                .find(|&at| unsafe {
                    !VirtualAlloc(at as _, (3 * CELL) as usize, MEM_RESERVE, PAGE_READWRITE)
                        .is_null()
                })
                .unwrap();
            for at in [start, start + CELL, start + 2 * CELL] {
                assert_eq!(
                    unsafe { VirtualAlloc(at as _, PAGE as usize, MEM_COMMIT, PAGE_READWRITE) }
                        as u64,
                    at
                );
                crate::shadow::register(at as usize, at, PAGE);
            }
            let directory_slot = |m: &Memory, at| unsafe {
                m.directory.1.add((at / PAGE * 2) as usize).read()
            };
            let mut slot = [0u32; 8];
            let handle = slot.as_mut_ptr() as u64;
            let install = |address: u64| {
                let desc = [
                    0,
                    11,
                    5768,
                    0,
                    address as u32,
                    (address >> 32) as u32,
                    0,
                    0,
                    0,
                    0,
                ];
                create_srv(null(), null(), &desc, handle);
                heap.mirror(1, handle, false).unwrap();
            };
            // No table, root address, constants, or previous GPU fault supply
            // these addresses. The two buffers are several MiB apart.
            install(start + 256);
            memory.prepare_addresses(&heap, &[], &[]);
            let first = directory_slot(&memory, start);
            assert_ne!(first, 0);
            assert_eq!(memory.copies[(first - 1) as usize].span(), (start, PAGE));
            assert_eq!(directory_slot(&memory, start + CELL), 0);
            assert_eq!(directory_slot(&memory, start + 2 * CELL), 0);
            install(start + 2 * CELL + 256);
            memory.prepare_addresses(&heap, &[], &[]);
            let second = directory_slot(&memory, start + 2 * CELL);
            assert_ne!(second, 0);
            assert_eq!(
                memory.copies[(second - 1) as usize].span(),
                (start + 2 * CELL, PAGE)
            );
            assert_eq!(directory_slot(&memory, start + CELL), 0);
            assert_eq!(memory.copies.len(), 2);
            assert!(heap.raw_page_views(&memory.copies).is_some());
            drop(memory);
            for at in [start, start + CELL, start + 2 * CELL] {
                crate::shadow::unregister(at as usize, at);
            }
            assert_ne!(unsafe { VirtualFree(start as _, 0, MEM_RELEASE) }, 0);
        });
    }

    #[test]
    fn directory_pages_index_the_same_bytes_as_bound_buffer_views() {
        let heap = 0x1_0000_0000;
        let start = heap + 0x1230;
        let len = 3 * PAGE - 0x1230;
        // The first full guest page is at resource byte 0x10000, even
        // though it lies just 0xedd0 bytes into the guest allocation.
        assert_eq!(page_byte(heap + PAGE, start, len, 0x1230), Some(0x10000));
        assert_eq!(page_byte(heap + 2 * PAGE, start, len, 0x1230), Some(0x20000));
        assert_eq!(page_byte(heap + PAGE, start, len, 0), Some(0xedd0));
        assert_eq!(page_byte(heap, start, len, 0x1230), None);
        assert_eq!(page_byte(heap + 3 * PAGE, start, len, 0x1230), None);
        assert_eq!(page_byte(heap + PAGE, start, len, u32::MAX.into()), None);
    }
}
