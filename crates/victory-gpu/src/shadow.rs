//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
}

/// Disjoint snapshot segments (offset, bytes), in order.
type Segments = Vec<(usize, Vec<u8>)>;

/// A snapshot and the pieces already compared.
type Snapshot = (Segments, std::collections::HashSet<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 2 << 30;

/// `VICTORY_INCREMENTAL_SNAPSHOTS=0` uses the original coalesced snapshots.
fn incremental_snapshots() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_INCREMENTAL_SNAPSHOTS").is_ok_and(|v| v.trim() == "0"))
}

/// Saves each newly covered byte once. Existing bytes must keep their first
/// value, even if the mapped copy has since been refreshed elsewhere.
fn snapshot_gaps<'a>(
    segments: &mut Segments,
    mut from: usize,
    to: usize,
    host: impl Fn(usize, usize) -> &'a [u8],
) -> usize {
    let mut i = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
    let mut copied = 0;
    while from < to {
        let end = match segments.get(i) {
            Some((at, bytes)) if *at <= from => {
                from = at + bytes.len();
                i += 1;
                continue;
            }
            Some((at, _)) => to.min(*at),
            None => to,
        };
        let mut bytes = pooled(end - from);
        bytes.extend_from_slice(host(from, end));
        segments.insert(i, (from, bytes));
        copied += end - from;
        from = end;
        i += 1;
    }
    copied
}

fn snapshot_coalesced<'a>(
    segments: &mut Segments,
    from: usize,
    to: usize,
    host: impl Fn(usize, usize) -> &'a [u8],
) -> usize {
    let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
    let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
    if let [(at, bytes)] = &segments[first..last]
        && *at <= from
        && to <= at + bytes.len()
    {
        return 0;
    }
    let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
    let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
    let mut grown = pooled(end - start);
    let mut at = start;
    for (taken, bytes) in segments.drain(first..last) {
        grown.extend_from_slice(host(at, taken));
        grown.extend_from_slice(&bytes);
        at = taken + bytes.len();
        give_back(bytes);
    }
    grown.extend_from_slice(host(at, end));
    segments.insert(first, (start, grown));
    end - start
}

/// Finds the consecutive snapshot segments covering the entire interval.
fn snapshot_cover(
    segments: &[(usize, Vec<u8>)],
    from: usize,
    to: usize,
) -> Option<&[(usize, Vec<u8>)]> {
    let start = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
    let mut covered = from;
    for (i, (at, bytes)) in segments[start..].iter().enumerate() {
        if *at > covered {
            return None;
        }
        covered = at + bytes.len();
        if covered >= to {
            return Some(&segments[start..=start + i]);
        }
    }
    None
}

/// A covered page usually lies in one segment. Assemble just that page when
/// it crosses a snapshot boundary, so copy-back still merges one page at a time.
fn snapshot_piece<'a>(
    segments: &'a [(usize, Vec<u8>)],
    from: usize,
    n: usize,
    scratch: &'a mut Vec<u8>,
) -> &'a [u8] {
    let (start, bytes) = &segments[0];
    if let Some(bytes) = bytes.get(from - start..from - start + n) {
        return bytes;
    }
    scratch.clear();
    let to = from + n;
    for (at, bytes) in segments {
        let end = (at + bytes.len()).min(to);
        scratch.extend_from_slice(&bytes[from.saturating_sub(*at)..end - at]);
        if end == to {
            break;
        }
    }
    scratch
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) {
            return 0;
        }
        let mut runs = Vec::new();
        changed_runs(host, &title, &mut runs);
        if runs.is_empty() {
            return 0;
        }
        let mut writable = Writable::default();
        let written = write_runs(from, n, host, &runs, &mut writable, back_mode());
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        written
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked: crate::watch::track(start as usize, len as usize, true),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, bytes from only the declared ranges, in
    /// reused memory (fresh snapshots cost ~2.3 million page faults per
    /// world frame). In the world, many overlapping views extend the saved
    /// range of the same 174 MB heap.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        let copied = if incremental_snapshots() {
            snapshot_gaps(&mut before.0, from, to, host)
        } else {
            snapshot_coalesced(&mut before.0, from, to, host)
        };
        #[cfg(feature = "diagnostics")]
        if copied != 0 {
            crate::copystat::add(crate::copystat::C::SnapshotBytes, copied as u64);
            crate::copystat::snapshot(self.start, self.len, copied as u64);
        }
        let _ = copied;
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if back
            && from < to
            && let Some(mut before) = snapshot_cover(segments, from, to)
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut old_piece = Vec::new();
            let mut title = Vec::new();
            let mut runs = Vec::new();
            let mut writable = Writable::default();
            let mode = back_mode();
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                while before[0].0 + before[0].1.len() <= at {
                    before = &before[1..];
                }
                let old = snapshot_piece(before, at, n, &mut old_piece);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old {
                    if title.len() < n {
                        title.resize(n, 0);
                    }
                    let title = &mut title[..n];
                    if crate::host::read_into((start + at) as *const u8, title) {
                        diff_piece_runs(host, old, title, &mut runs);
                        written += write_runs(start + at, n, title, &runs, &mut writable, mode);
                        synced.push((start + at) / PAGE);
                    }
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes: the pages changed since taken, or all of it where the
    /// memory is not tracked.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if let Some(t) = &self.tracked {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let runs = t.take(from, to);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
            // Cut to the range: next to it shaders may be writing the copy.
            let copied: usize = runs.into_iter().map(|(a, b)| self.copy(a, b)).sum();
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::Checked, (to - from) as u64);
                add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
            }
            return copied;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    map.insert((address, len), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64);
        copy.writing(p + 3 * PAGE, 100);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE);
        copy.writing(p + PAGE, PAGE);
        copy.writing(p + 80, 8);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "only the declared union");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 3, "only the changed bytes go back");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), 1, "only the changed byte goes back");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    #[test]
    fn incremental_snapshot_preserves_first_bytes_across_overlaps_and_gaps() {
        let mut host = vec![0; 2 * PAGE + 41];
        let mut incremental = Segments::new();
        let mut coalesced = Segments::new();
        let mut bytes = 0;
        for (from, to, value) in [
            (9, PAGE - 10, 0x11),
            (PAGE + 9, PAGE + 81, 0x22),
            (PAGE - 18, PAGE + 29, 0x33),
            (0, PAGE + 1, 0x44),
            (PAGE + 81, 2 * PAGE + 41, 0x55),
        ] {
            host[from..to].fill(value);
            let read = |a, b| &host[a..b];
            bytes += snapshot_gaps(&mut incremental, from, to, read);
            snapshot_coalesced(&mut coalesced, from, to, read);
            if value == 0x22 {
                assert!(snapshot_cover(&incremental, 9, PAGE + 81).is_none());
            }
        }
        assert_eq!(bytes, host.len());
        assert_eq!(snapshot_gaps(&mut incremental, 0, host.len(), |a, b| &host[a..b]), 0);
        let want = &coalesced[0].1;
        let mut before = snapshot_cover(&incremental, 0, host.len()).unwrap();
        let mut scratch = Vec::new();
        for from in (0..host.len()).step_by(PAGE) {
            while before[0].0 + before[0].1.len() <= from {
                before = &before[1..];
            }
            let n = PAGE.min(host.len() - from);
            assert_eq!(snapshot_piece(before, from, n, &mut scratch), &want[from..from + n]);
        }
    }

    #[test]
    #[ignore = "offline performance benchmark"]
    fn snapshot_overlap_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        const MIB: usize = 1 << 20;
        let host = vec![0x5au8; 174 * MIB];
        let mut runs = [Vec::new(), Vec::new()];
        let mut totals = [0; 2];
        for sample in 0..8 {
            for mode in [sample % 2, 1 - sample % 2] {
                let mut segments = Segments::new();
                let start = Instant::now();
                let read = |a, b| &black_box(&host)[a..b];
                let mut copied = 0;
                for from in (0..12).map(|i| i * 12 * MIB) {
                    let to = from + 32 * MIB;
                    copied += if mode == 0 {
                        snapshot_coalesced(&mut segments, from, to, read)
                    } else {
                        snapshot_gaps(&mut segments, from, to, read)
                    };
                }
                black_box(&segments);
                if sample > 0 {
                    runs[mode].push(start.elapsed());
                }
                totals[mode] = copied;
                for (_, bytes) in segments.drain(..) {
                    give_back(bytes);
                }
            }
        }
        for (mode, samples) in runs.iter_mut().enumerate() {
            samples.sort_unstable();
            let name = ["coalesced", "incremental"][mode];
            let elapsed = samples[samples.len() / 2].as_secs_f64() * 1000.0;
            println!("{name}: {} MiB copied, {elapsed:.2} ms median", totals[mode] / MIB);
        }
    }

    #[test]
    fn copy_back_page_writes_only_changed_runs() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        // Seed title memory and take it into the copy.
        for off in 0..64 {
            // SAFETY: our pool.
            unsafe { ((p + off) as *mut u8).write_volatile(0x11) };
        }
        copy.update(p, len);
        // Shader writes bytes 10..14 and 40..42; byte 20 stays equal to title memory.
        copy.write_host((p + 10) as u64, &[0xaa, 0xbb, 0xcc, 0xdd]);
        copy.write_host((p + 40) as u64, &[0xee, 0xff]);
        assert_eq!(copy.copy_back_page(p), 6);
        // SAFETY: our pool.
        let byte = |off: usize| unsafe { ((p + off) as *const u8).read_volatile() };
        assert_eq!((10..14).map(byte).collect::<Vec<_>>(), [0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!((40..42).map(byte).collect::<Vec<_>>(), [0xee, 0xff]);
        assert_eq!(byte(9), 0x11);
        assert_eq!(byte(14), 0x11);
        assert_eq!(byte(39), 0x11);
        assert_eq!(byte(42), 0x11);
        assert_eq!(copy.copy_back_page(p), 0);
        unregister(p, p as u64);
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn title_memory(len: usize) -> &'static mut [u8] {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        // SAFETY: fresh committed allocation, kept for the test process.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            assert!(!p.is_null());
            std::slice::from_raw_parts_mut(p.cast(), len)
        }
    }

    #[test]
    fn fast_diff_and_changed_runs_match_byte_reference() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..500 {
            let n = 1 + rng.below(300);
            let old: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            let mut host = old.clone();
            let mut title = old.clone();
            for _ in 0..rng.below(1 + round % 20) {
                let i = rng.below(n);
                host[i] = host[i].wrapping_add(1 + rng.below(255) as u8);
            }
            for _ in 0..rng.below(1 + round % 12) {
                let i = rng.below(n);
                title[i] = title[i].wrapping_add(1 + rng.below(255) as u8);
            }

            let mut ref_host = host.clone();
            let mut ref_title = title.clone();
            let mut ref_runs: Vec<(usize, usize)> = Vec::new();
            for (k, ((t, h), o)) in ref_title
                .iter_mut()
                .zip(ref_host.iter_mut())
                .zip(&old)
                .enumerate()
            {
                if *h != *o && *t != *h {
                    *t = *h;
                    push_run(&mut ref_runs, k);
                }
                *h = *t;
            }
            let mut runs = Vec::new();
            diff_piece_runs(&mut host, &old, &mut title, &mut runs);
            assert_eq!(runs, ref_runs);
            assert_eq!(title, ref_title);
            assert_eq!(host, ref_host);

            let ref_changed: Vec<(usize, usize)> = {
                let mut r = Vec::new();
                for k in 0..n {
                    if host[k] != old[k] {
                        push_run(&mut r, k);
                    }
                }
                r
            };
            changed_runs(&host, &old, &mut runs);
            assert_eq!(runs, ref_changed);
        }
    }

    #[test]
    fn write_runs_writes_same_bytes_in_direct_and_kernel_modes() {
        const N: usize = 64 << 10;
        for protect in [false, true] {
            let mut outputs = Vec::new();
            for mode in [BackMode::Kernel, BackMode::Direct] {
                let title = title_memory(N);
                let mut rng = Rng(7);
                for b in title.iter_mut() {
                    *b = rng.next() as u8;
                }
                let old = title.to_vec();
                let mut host = old.clone();
                for _ in 0..400 {
                    let i = rng.below(N);
                    host[i] ^= 0x5a;
                }
                // GPU wrote bytes title already holds: not written.
                host[100..104].copy_from_slice(&[9, 9, 9, 9]);
                title[100..104].copy_from_slice(&[9, 9, 9, 9]);
                // Title concurrently wrote byte 200 (GPU did not).
                title[200] ^= 0xff;
                if protect {
                    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
                    let mut was = 0;
                    // SAFETY: our own allocation.
                    unsafe { VirtualProtect(title.as_ptr().cast(), N, PAGE_READONLY, &mut was) };
                    assert!(!Writable::default().covers(title.as_ptr() as usize, N));
                }
                let mut title_buf = title.to_vec();
                let mut runs = Vec::new();
                diff_piece_runs(&mut host, &old, &mut title_buf, &mut runs);
                // Runs never bridge unchanged byte 200 or title-matching 100..104.
                assert!(!runs.iter().any(|&(a, b)| (a..b).contains(&200)));
                assert!(!runs.iter().any(|&(a, b)| a < 104 && b > 100));
                let mut w = Writable::default();
                let written = write_runs(title.as_ptr() as usize, N, &title_buf, &runs, &mut w, mode);
                let expected_written: usize = runs.iter().map(|&(a, b)| b - a).sum();
                assert_eq!(written, expected_written);
                if !protect {
                    assert_eq!(title, title_buf);
                    assert_eq!(title[200], old[200] ^ 0xff);
                }
                outputs.push((written, title.to_vec()));
            }
            assert_eq!(outputs[0], outputs[1], "protect {protect}");
        }
    }

    #[test]
    fn writable_caches_region_across_pieces() {
        const N: usize = 256 << 10;
        let mem = title_memory(N);
        let at = mem.as_ptr() as usize;
        let mut w = Writable::default();
        assert!(w.covers(at, 64 << 10));
        assert!(w.from <= at && w.to >= at + N);
        assert!(w.covers(at + (64 << 10), 64 << 10));
        assert!(w.covers(at + (128 << 10), 64 << 10));
    }
}

fn push_run(runs: &mut Vec<(usize, usize)>, idx: usize) {
    match runs.last_mut() {
        Some((_, e)) if *e == idx => *e = idx + 1,
        _ => runs.push((idx, idx + 1)),
    }
}

/// Finds the byte runs in `[0..n)` that the GPU changed (`host != old`) and title memory does
/// not already hold (`title != host`), putting those bytes into `title` and `title`'s bytes into
/// `host`. Skips equal 32-byte stretches (`host == old`); runs are never merged across unchanged
/// bytes or bytes already equal to the title's.
fn diff_piece_runs(
    host: &mut [u8],
    old: &[u8],
    title: &mut [u8],
    runs: &mut Vec<(usize, usize)>,
) {
    runs.clear();
    let n = host.len().min(old.len()).min(title.len());
    let mut k = 0;
    while k + 32 <= n {
        let h: &[u8; 32] = host[k..k + 32].try_into().unwrap();
        let o: &[u8; 32] = old[k..k + 32].try_into().unwrap();
        if h == o {
            let t: &[u8; 32] = title[k..k + 32].try_into().unwrap();
            if h != t {
                host[k..k + 32].copy_from_slice(t);
            }
        } else {
            for idx in k..k + 32 {
                if host[idx] != old[idx] && title[idx] != host[idx] {
                    title[idx] = host[idx];
                    push_run(runs, idx);
                }
                host[idx] = title[idx];
            }
        }
        k += 32;
    }
    for idx in k..n {
        if host[idx] != old[idx] && title[idx] != host[idx] {
            title[idx] = host[idx];
            push_run(runs, idx);
        }
        host[idx] = title[idx];
    }
}

/// Finds the byte runs where `host` differs from `title`, skipping equal 32-byte stretches.
fn changed_runs(host: &[u8], title: &[u8], runs: &mut Vec<(usize, usize)>) {
    runs.clear();
    let n = host.len().min(title.len());
    let mut k = 0;
    while k + 32 <= n {
        let h: &[u8; 32] = host[k..k + 32].try_into().unwrap();
        let t: &[u8; 32] = title[k..k + 32].try_into().unwrap();
        if h != t {
            for idx in k..k + 32 {
                if host[idx] != title[idx] {
                    push_run(runs, idx);
                }
            }
        }
        k += 32;
    }
    for idx in k..n {
        if host[idx] != title[idx] {
            push_run(runs, idx);
        }
    }
}

/// How the copy-back writes title memory.
#[derive(Clone, Copy, PartialEq, Debug)]
enum BackMode {
    /// One `WriteProcessMemory` per run of changed bytes.
    Kernel,
    /// Plain stores where the piece is committed read-write (at most one query per region),
    /// the kernel elsewhere (read-only or otherwise protected title memory).
    Direct,
}

/// Local build: `VICTORY_COPY_BACK_KERNEL=1` writes every run through the kernel again.
fn back_mode() -> BackMode {
    static MODE: std::sync::OnceLock<BackMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        if std::env::var("VICTORY_COPY_BACK_KERNEL").is_ok_and(|v| v.trim() == "1") {
            BackMode::Kernel
        } else {
            BackMode::Direct
        }
    })
}

/// Writes each `(r0, r1)` run from `bytes` into title memory at `at`, opening write guards on
/// `[at, at + len)` first and using plain stores when the region is committed read-write.
fn write_runs(
    at: usize,
    len: usize,
    bytes: &[u8],
    runs: &[(usize, usize)],
    writable: &mut Writable,
    mode: BackMode,
) -> usize {
    if runs.is_empty() {
        return 0;
    }
    let _writing = crate::watch::writing(at, len);
    let direct = mode == BackMode::Direct && writable.covers(at, len);
    let mut written = 0;
    for &(r0, r1) in runs {
        let dst = (at + r0) as *mut u8;
        let src = &bytes[r0..r1];
        if direct {
            // SAFETY: committed read-write title memory (`writable.covers` checked `[at, at + len)`),
            // and `bytes` is our own buffer.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        } else {
            crate::host::write_mapped(dst, src.as_ptr(), src.len());
        }
        written += src.len();
    }
    written
}

/// The last title region found committed and read-write (`[from, to)`), kept for the pieces
/// after it: a query costs about as much as walking the region (~80 us for a 64 MB heap).
/// Only read-write answers are kept: a block the write guards (`crate::watch`) opened since
/// is asked again.
#[derive(Default)]
struct Writable {
    from: usize,
    to: usize,
}

impl Writable {
    /// Whether all of `[at, at + n)` is committed and read-write (a plain store cannot fault
    /// there except into the write guards' handler, which opens the block).
    fn covers(&mut self, at: usize, n: usize) -> bool {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD,
            PAGE_READWRITE, VirtualQuery,
        };
        let end = at + n;
        let mut p = at;
        while p < end {
            if (self.from..self.to).contains(&p) {
                p = self.to;
                continue;
            }
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) } == 0 {
                return false;
            }
            let base = info.Protect & 0xff;
            let next = info.BaseAddress as usize + info.RegionSize;
            if info.State != MEM_COMMIT
                || info.Protect & PAGE_GUARD != 0
                || (base != PAGE_READWRITE && base != PAGE_EXECUTE_READWRITE)
                || next <= p
            {
                return false;
            }
            (self.from, self.to) = (info.BaseAddress as usize, next);
            p = next;
        }
        true
    }
}
