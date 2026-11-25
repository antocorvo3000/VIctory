//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Whether a texture's content hash and refresh history are kept when its
/// staging upload buffer is dropped (`VICTORY_TEXTURE_KEEP_HASH=0` clears
/// them on drop, as before).
fn keep_hash_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_KEEP_HASH").is_none_or(|v| v != "0"))
}

/// Whether staging upload buffers are dropped after upload when a texture
/// is not rewritten often (`VICTORY_TEXTURE_DROP_STAGING=0` keeps them in
/// `STAGING` until `let_go_idle`).
fn drop_staging_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_DROP_STAGING").is_none_or(|v| v != "0"))
}

/// Frame window within which a second+ content change keeps a texture's
/// staging upload buffer across frames (`VICTORY_TEXTURE_REWRITE_WINDOW`,
/// default 120 frames).
fn rewrite_window() -> u64 {
    static WIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *WIN.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_REWRITE_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(120)
    })
}

/// Policy: whether a texture is rewritten often enough to retain its staging
/// upload buffer across frames (at least 2 content changes after the initial
/// upload, with the latest change within `window` frames of `frame`).
fn rewritten_often(rewrites: u32, last_change_frame: u64, frame: u64, window: u64) -> bool {
    rewrites >= 2 && frame.saturating_sub(last_change_frame) <= window
}

#[derive(Default)]
struct HistoryInner {
    last: Last,
    host_id: u64,
    uploaded_once: bool,
    dirty: bool,
    rewrites: u32,
    last_change_frame: u64,
    memory: usize,
    size: usize,
    on_demand: bool,
    detiled: u32,
    judged: [u64; 2],
}

/// Persistent per-texture hash and rewrite history kept across staging-buffer
/// drops and residency evictions.
pub(crate) struct History {
    resource: usize,
    inner: Mutex<HistoryInner>,
}

static HISTORY: Mutex<Option<HashMap<usize, Arc<History>>>> = Mutex::new(None);

/// Dropped staging buffers after upload, unchanged submission checks,
/// heap re-uploads skipped by hash.
static HISTORY_STATS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

pub(crate) fn history_stats() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        HISTORY_STATS[0].load(Relaxed),
        HISTORY_STATS[1].load(Relaxed),
        HISTORY_STATS[2].load(Relaxed),
    ]
}

fn history_for(resource: usize) -> Arc<History> {
    let mut map = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    map.entry(resource)
        .or_insert_with(|| {
            Arc::new(History {
                resource,
                inner: Mutex::default(),
            })
        })
        .clone()
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: std::mem::ManuallyDrop<ID3D12Resource>,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Total bytes of the console layout across all planes.
    tiled_size: u64,
    /// Mips per array slice, and subresources per plane (`mips * slices`).
    mips: u32,
    plane_subs: u32,
    /// Second plane (stencil) of a planar depth-stencil texture: console byte
    /// offset from `memory` and its layout.
    stencil: Option<(u64, victory_formats::texture::Layout)>,
    /// The staging buffer's logical bytes (counted in `STAGED` while it lives).
    bytes: u64,
    /// The underlying pooled upload buffer's capacity.
    cap: u64,
    memory: *const u8,
    host_id: u64,
    history: Arc<History>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: 5,411 subresource copies
    /// of 1,120 textures in one frame of run 18.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    let first = list.first_use(placed as usize);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !first && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. It never shows partly read.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there.
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    REFRESH_WAITS[slot].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

/// Whether CPU read scratch buffers are shared across threads in a bounded
/// pool instead of kept per thread (`VICTORY_READ_POOL=0` restores per-thread
/// `RefCell` buffers).
fn read_pool_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_READ_POOL").is_none_or(|v| v != "0"))
}

const READ_POOL_MAX_ITEM: usize = 16 << 20;
const READ_POOL_MAX_BYTES: usize = 64 << 20;
const READ_POOL_MAX_COUNT: usize = 4;

#[derive(Default)]
struct ReadScratchPool {
    kept: Vec<(Vec<u8>, u64)>,
    bytes: usize,
}

impl ReadScratchPool {
    fn take(&mut self, size: usize) -> Vec<u8> {
        let mut best: Option<(usize, usize)> = None;
        for (i, (buf, _)) in self.kept.iter().enumerate() {
            let cap = buf.capacity();
            if cap >= size && best.is_none_or(|(_, best_cap)| cap < best_cap) {
                best = Some((i, cap));
            }
        }
        let mut buf = if let Some((idx, cap)) = best {
            self.bytes -= cap;
            self.kept.swap_remove(idx).0
        } else if let Some((mut buf, _)) = self.kept.pop() {
            self.bytes -= buf.capacity();
            buf.resize(size, 0);
            buf
        } else {
            Vec::new()
        };
        if buf.len() < size {
            buf.resize(size, 0);
        }
        buf
    }

    fn give(&mut self, mut buf: Vec<u8>, now: u64) {
        let cap = buf.capacity();
        if cap == 0 || cap > READ_POOL_MAX_ITEM || self.kept.len() >= READ_POOL_MAX_COUNT || self.bytes + cap > READ_POOL_MAX_BYTES {
            drop(buf);
            return;
        }
        buf.clear();
        self.bytes += cap;
        self.kept.push((buf, now));
    }

    fn trim_idle(&mut self, max_age_ms: u64, now: u64) {
        let mut i = 0;
        while i < self.kept.len() {
            if now.saturating_sub(self.kept[i].1) >= max_age_ms {
                let (buf, _) = self.kept.swap_remove(i);
                self.bytes -= buf.capacity();
            } else {
                i += 1;
            }
        }
    }
}

static READ_POOL: Mutex<ReadScratchPool> = Mutex::new(ReadScratchPool { kept: Vec::new(), bytes: 0 });

pub(crate) fn read_pool_bytes() -> u64 {
    READ_POOL.lock().unwrap_or_else(|e| e.into_inner()).bytes as u64
}

thread_local! {
    /// Fallback per-thread read buffer when `VICTORY_READ_POOL=0`.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of a pooled scratch buffer (their contents
/// are whatever the last user left).
pub(crate) fn with_read_scratch<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    if !read_pool_on() {
        return READ.with(|read| {
            let mut read = read.borrow_mut();
            if read.len() < size {
                read.resize(size, 0);
            }
            f(&mut read[..size])
        });
    }
    let mut buf = READ_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take(size);
    // `take` ensured `buf.len() >= size`.
    let out = f(&mut buf[..size]);
    READ_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .give(buf, crate::host::now_ms());
    out
}

/// Calls `f` with `size` bytes of a read scratch buffer.
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    with_read_scratch(size, f)
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        // SAFETY: only taken here in Drop.
        let buffer = unsafe { std::mem::ManuallyDrop::take(&mut self.buffer) };
        crate::host::give_upload(buffer, self.data, self.cap);
    }
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    READ_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .trim_idle(idle_ms, crate::host::now_ms());
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = map.as_mut() else {
        return;
    };
    if map.is_empty() {
        return;
    }
    // A list that recorded an upload or a check holds the staging until it
    // is reset: one only the table holds is named by no list, recorded or
    // running.
    let (mut went, mut bytes) = (0u64, 0u64);
    map.retain(|_, s| {
        let keep = Arc::strong_count(s) > 1
            || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
        if !keep {
            LET_GO[0].fetch_add(1, Relaxed);
            LET_GO[1].fetch_add(s.bytes, Relaxed);
            went += 1;
            bytes += s.bytes;
        }
        keep
    });
    if went > 0 {
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut hist = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hist) = hist.as_mut() {
        hist.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    use std::sync::atomic::Ordering::Relaxed;
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let current_host_id = crate::resource::host_id(resource).unwrap_or(0);
    // If no staging buffer is currently alive, check whether the texture was
    // already uploaded to this exact host copy (`host_id`) and is still clean:
    // if so, defer only a lightweight hash/watch check instead of allocating
    // a staging upload buffer.
    let staged_now = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref().and_then(|m| m.get(&(resource as usize)).cloned())
    };
    if staged_now.is_none() && keep_hash_on() && current_host_id != 0 {
        let history = history_for(resource as usize);
        let mut inner = history.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.host_id == current_host_id && inner.uploaded_once && !inner.dirty {
            let needs_on_demand = inner.on_demand && judgement_for(inner.memory, inner.size) != inner.judged;
            if needs_on_demand {
                inner.dirty = true;
                inner.last = Last::default();
            } else {
                drop(inner);
                // Look now, while the list is recorded: a texture the title
                // rewrote before this use is uploaded in this list, not one
                // frame late. Writes between now and the submission are still
                // caught by the check at submission (and uploaded at the next use).
                history.check();
                let clean = !history.inner.lock().unwrap_or_else(|e| e.into_inner()).dirty;
                if clean {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::TexSkipped, 1);
                    list.defer_texture_check(history);
                    return;
                }
            }
        }
    }
    let Some(staging) = staging(resource, texture, current_host_id) else {
        return;
    };
    staging.used.store(crate::host::moment(), Relaxed);
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left, and what it finds changed the next list that
    // samples the texture uploads (a frame late where every list of the
    // frame was recorded before that check).
    if staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        list.defer_texture(staging);
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexUploads, 1);
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&*staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // When the list is submitted, its copy takes what the staging buffer
    // holds by then: noted before the check below, which sets `dirty`
    // again if it detiles (the next list then uploads once more).
    {
        let staging = staging.clone();
        list.defer_host(move || {
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);
            let mut inner = staging.history.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.host_id = staging.host_id;
            inner.uploaded_once = true;
            inner.dirty = false;
        });
    }
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level, so each one
    // given here is written whole by it (the rest are copies into no
    // memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    list.defer_texture(staging.clone());
    if let Some(on_demand) = on_demand {
        let staging = staging.clone();
        list.defer_host(move || {
            let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
            staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
            // A texture picked by index has no staging buffer after this
            // upload: its slots are made due again by the texture pass.
            if waiting {
                crate::descriptor::host::levels_waiting(staging.memory as u64);
            }
        });
    }
    // Once the submission's refresh has run, drop this staging buffer from
    // `STAGING` if the texture is not rewritten often; the list still keeps
    // `Arc<Staging>` until reset, and `History` preserves its content hash.
    if drop_staging_on() {
        let staging_drop = staging;
        let key = resource as usize;
        list.defer_host(move || {
            unstage_if_infrequent(key, &staging_drop);
        });
    }
}

fn unstage_if_infrequent(resource: usize, staging: &Arc<Staging>) {
    let keep = {
        let inner = staging.history.inner.lock().unwrap_or_else(|e| e.into_inner());
        let dirty = staging.dirty.load(std::sync::atomic::Ordering::Relaxed);
        let waiting = staging.waiting.load(std::sync::atomic::Ordering::Relaxed) != 0;
        dirty
            || waiting
            || rewritten_often(
                inner.rewrites,
                inner.last_change_frame,
                crate::frame::current(),
                rewrite_window(),
            )
    };
    if keep {
        return;
    }
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut()
        && map.get(&resource).is_some_and(|s| Arc::ptr_eq(s, staging))
    {
        map.remove(&resource);
        HISTORY_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

fn judgement_for(memory: usize, size: usize) -> [u64; 2] {
    let start = memory as u64;
    let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
    let limit = crate::descriptor::host::next_start(start, start.saturating_add(size as u64))
        .map_or(u64::MAX, |at| at - start);
    [first, limit]
}

/// Checks whether `[memory, memory + size)` changed against `last`.
/// When `commit_change` is true (about to detile into a staging buffer),
/// updates `last` on change and invokes `on_change`. When `commit_change` is
/// false, leaves `last.hash` (and `last.asked` on write-watch hits) untouched
/// on change so the next `Staging::refresh` still detiles.
/// Returns `(bytes_read, changed)`.
fn check_memory_changed(
    memory: *const u8,
    size: usize,
    last: &mut Last,
    commit_change: bool,
    on_change: impl FnOnce(&[u8]),
) -> (usize, bool) {
    let start = memory as usize;
    if let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false)) {
        if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
            return (0, false);
        }
        if !commit_change {
            return (0, true);
        }
        last.asked = now;
        // Read through the copy, which stands unmapped pages: memory of
        // big pools is too slow to ask page by page.
        with_read(size, |read| {
            crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
            on_change(read);
        });
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
        return (size, true);
    }
    static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let mut guard_saw_write = false;
    let mut guard_now: Option<u64> = None;
    if *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
        && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, true))
    {
        const HASH_EVERY: u64 = 8;
        let unwritten = last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked);
        if commit_change {
            last.asked = now;
        } else {
            guard_now = Some(now);
        }
        let turn = crate::frame::current().wrapping_add((start >> 16) as u64).is_multiple_of(HASH_EVERY);
        if unwritten && !turn {
            if let Some(n) = guard_now {
                last.asked = n;
            }
            return (0, false);
        }
        guard_saw_write = !unwritten;
    }
    let _mapped = if no_wait() {
        let Some(mapped) = crate::watch::title_memory_reading_now(start, size) else {
            use std::sync::atomic::Ordering::Relaxed;
            let mut changed = false;
            let read = with_read(size, |read| {
                if !crate::host::read_into(memory, read) {
                    return false;
                }
                let hash = blake3::hash(read);
                if last.hash != Some(hash) {
                    changed = true;
                    if commit_change {
                        last.hash = Some(hash);
                        on_change(read);
                    }
                } else if let Some(n) = guard_now {
                    last.asked = n;
                }
                true
            });
            return if read {
                UNLOCKED[0].fetch_add(1, Relaxed);
                (size, changed)
            } else {
                UNLOCKED[1].fetch_add(1, Relaxed);
                last.asked = 0;
                (0, false)
            };
        };
        mapped
    } else {
        waited(2, || crate::watch::title_memory_reading(start, size))
    };
    if !committed(memory, size) {
        let frame = crate::frame::current();
        if should_skip_partial(last.partial, frame, guard_saw_write, partial_guard_immediate()) {
            if let Some(n) = guard_now && !guard_saw_write {
                last.asked = n;
            }
            return (0, false);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        let changed = with_read(size, |read| {
            crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
            let hash = blake3::hash(read);
            if last.hash == Some(hash) {
                last.partial = Some(frame);
                if let Some(n) = guard_now {
                    last.asked = n;
                }
                return false;
            }
            if read.iter().all(|&b| b == 0) {
                last.partial = Some(frame);
                if commit_change {
                    last.hash = Some(hash);
                }
                if let Some(n) = guard_now {
                    last.asked = n;
                }
                return false;
            }
            if commit_change {
                last.partial = Some(frame);
                last.hash = Some(hash);
                on_change(read);
            }
            true
        });
        #[cfg(feature = "diagnostics")]
        if changed && commit_change {
            crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        }
        return (size, changed);
    }
    if through_copy() {
        return with_title(memory, size, |tiled| {
            let hash = blake3::hash(tiled);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            if last.hash == Some(hash) {
                if let Some(n) = guard_now {
                    last.asked = n;
                }
                return (size, false);
            }
            if commit_change {
                last.hash = Some(hash);
                on_change(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            (size, true)
        });
    }
    // SAFETY: committed memory of the texture's layout size, and it
    // stays committed while `_mapped` is held.
    let tiled = unsafe { std::slice::from_raw_parts(memory, size) };
    let hash = blake3::hash(tiled);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
    if last.hash == Some(hash) {
        if let Some(n) = guard_now {
            last.asked = n;
        }
        return (size, false);
    }
    if commit_change {
        last.hash = Some(hash);
        on_change(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
    }
    (size, true)
}

impl History {
    /// Checks at submission whether a previously-uploaded texture whose staging
    /// buffer was dropped changed in title memory. Marks `dirty` and bumps
    /// rewrite tracking if it did.
    pub(crate) fn check(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let whole = std::time::Instant::now();
        let mut inner = waited(0, || self.inner.lock().unwrap_or_else(|e| e.into_inner()));
        if inner.dirty || !inner.uploaded_once || inner.memory == 0 || inner.size == 0 {
            return;
        }
        let memory = inner.memory as *const u8;
        let size = inner.size;
        let (read, changed) = check_memory_changed(memory, size, &mut inner.last, false, |_| {});
        if changed {
            inner.dirty = true;
            inner.rewrites = inner.rewrites.saturating_add(1);
            inner.last_change_frame = crate::frame::current();
        } else {
            HISTORY_STATS[1].fetch_add(1, Relaxed);
        }
        drop(inner);
        let us = whole.elapsed().as_micros() as u64;
        REFRESH_WAITS[3].fetch_add(us, Relaxed);
        crate::host::PREPARING[2].fetch_add(us, Relaxed);
        crate::host::PREPARING[3].fetch_add(read as u64, Relaxed);
    }
}

/// Returns true if `resource` already has its current `host_id` uploaded
/// and its title memory has not changed since that upload (keeping its hash).
pub(crate) fn unchanged_on_card(resource: This) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if !keep_hash_on() {
        return false;
    }
    let Some(host_id) = crate::resource::host_id(resource) else {
        return false;
    };
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map.as_ref().and_then(|m| m.get(&(resource as usize))) {
            if s.dirty.load(Relaxed) || !s.uploaded.load(Relaxed) || s.host_id != host_id {
                return false;
            }
            if s.on_demand && (s.waiting.load(Relaxed) != 0 || s.judgement() != s.judged()) {
                return false;
            }
            let mut inner = s.history.inner.lock().unwrap_or_else(|e| e.into_inner());
            let (_, changed) = check_memory_changed(s.memory, s.tiled_size as usize, &mut inner.last, false, |_| {});
            if changed {
                inner.dirty = true;
                inner.rewrites = inner.rewrites.saturating_add(1);
                inner.last_change_frame = crate::frame::current();
                s.dirty.store(true, Relaxed);
                return false;
            }
            HISTORY_STATS[2].fetch_add(1, Relaxed);
            return true;
        }
    }
    let history = history_for(resource as usize);
    let mut inner = history.inner.lock().unwrap_or_else(|e| e.into_inner());
    if inner.host_id != host_id || !inner.uploaded_once || inner.dirty || inner.memory == 0 || inner.size == 0 {
        return false;
    }
    if inner.on_demand && judgement_for(inner.memory, inner.size) != inner.judged {
        inner.dirty = true;
        return false;
    }
    let memory = inner.memory as *const u8;
    let size = inner.size;
    let (_, changed) = check_memory_changed(memory, size, &mut inner.last, false, |_| {});
    if changed {
        inner.dirty = true;
        inner.rewrites = inner.rewrites.saturating_add(1);
        inner.last_change_frame = crate::frame::current();
        return false;
    }
    HISTORY_STATS[2].fetch_add(1, Relaxed);
    true
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        judgement_for(self.memory as usize, self.tiled_size as usize)
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        let inner = self.history.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.judged
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        let mut inner = self.history.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.last = Last::default();
        inner.dirty = true;
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.layout.size as usize)
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        REFRESH_WAITS[3].fetch_add(whole.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        read
    }

    fn refresh_timed(&self) -> usize {
        let size = self.tiled_size as usize;
        let mut inner_guard = waited(0, || self.history.inner.lock().unwrap_or_else(|e| e.into_inner()));
        let inner = &mut *inner_guard;
        let was_uploaded = inner.uploaded_once;
        let mut detiled_held = None;
        let (read, changed) = check_memory_changed(self.memory, size, &mut inner.last, true, |tiled| {
            detiled_held = Some(self.detile_with_judgement(tiled, &mut inner.judged));
        });
        if let Some(held) = detiled_held {
            inner.detiled = held;
            if was_uploaded {
                inner.rewrites = inner.rewrites.saturating_add(1);
                inner.last_change_frame = crate::frame::current();
            }
        }
        let _ = changed;
        read
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer, storing
    /// the on-demand judgement into `judged_out`.
    fn detile_with_judgement(&self, tiled: &[u8], judged_out: &mut [u64; 2]) -> u32 {
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::Relaxed;
            let judged = self.judgement();
            *judged_out = judged;
            let held = self.held_levels(judged);
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (plane_sub, is_stencil) = if i < self.plane_subs as usize {
                (i, false)
            } else {
                (i - self.plane_subs as usize, true)
            };
            let (mip, slice) = (plane_sub as u32 % self.mips, plane_sub as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let max_len = (self.bytes as usize).saturating_sub(footprint.Offset as usize);
            let len = (pitch * rows as usize * f.Depth as usize).min(max_len);
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            if is_stencil {
                let Some((offset, ref stencil)) = self.stencil else {
                    continue;
                };
                let Some(stencil_tiled) = tiled.get(offset as usize..) else {
                    continue;
                };
                if let Err(e) =
                    stencil.detile(stencil_tiled, mip, slice, linear, pitch, pitch * rows as usize)
                {
                    tracing::warn!(?e, mip, slice, "stencil plane not detiled");
                    SKIPPED[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return held;
                }
                continue;
            }
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                SKIPPED[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return held;
            }
        }
        held
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew. Keeps the texture's content hash unless `VICTORY_TEXTURE_KEEP_HASH=0`.
pub(crate) fn unstage(resource: This) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&(resource as usize));
    }
    if !keep_hash_on() {
        let mut hist = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hist) = hist.as_mut()
            && let Some(h) = hist.get(&(resource as usize))
        {
            let mut inner = h.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.last = Last::default();
            inner.uploaded_once = false;
            inner.dirty = true;
        }
    }
}

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let (at, len) = map.as_ref()?.get(&(resource as usize))?.memory();
    Some((at as u64, len as u64))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end
/// (a texture streams in whole mips; the small ones lie first). Read
/// through the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut hasher = blake3::Hasher::new();
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            hasher.update(&read[..bytes]);
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    let hash = hasher.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first) | 1
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// Whether `resource` currently has a staging buffer whose latest contents
/// were already uploaded on the GPU without further detected changes.
pub(crate) fn staged_uploaded(resource: This) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map.as_ref().and_then(|map| map.get(&(resource as usize))) {
            return s.uploaded.load(Relaxed) && !s.dirty.load(Relaxed);
        }
    }
    let Some(host_id) = crate::resource::host_id(resource) else {
        return false;
    };
    let hist = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
    hist.as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .is_some_and(|h| {
            let inner = h.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.host_id == host_id && inner.uploaded_once && !inner.dirty
        })
}

/// Whether `resource` is staged and has detiled changes waiting to upload
/// (e.g. set by `Staging::refresh` at submission after `upload` skipped
/// copying because the texture appeared clean).
pub(crate) fn is_dirty(resource: This) -> bool {
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map.as_ref().and_then(|map| map.get(&(resource as usize))) {
            return s.dirty.load(std::sync::atomic::Ordering::Relaxed);
        }
    }
    let hist = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
    hist.as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .is_some_and(|h| h.inner.lock().unwrap_or_else(|e| e.into_inner()).dirty)
}

/// Bypass the 30-frame partial-texture throttle when the guard/write-watch
/// saw newly written or mapped pages (`VICTORY_PARTIAL_GUARD_IMMEDIATE=0`
/// restores the unconditional 30-frame wait).
fn partial_guard_immediate() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PARTIAL_GUARD_IMMEDIATE").is_none_or(|v| v != "0"))
}

/// Whether a partly-committed texture refresh at `frame` should skip due to the
/// 30-frame throttle (`partial_at`) given whether the guard just saw writes.
fn should_skip_partial(partial_at: Option<u64>, frame: u64, guard_saw_write: bool, immediate: bool) -> bool {
    !(guard_saw_write && immediate) && partial_at.is_some_and(|at| frame < at + 30)
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
pub(crate) fn staged(resource: This) -> Option<(u64, bool)> {
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    let size = s.layout.size as usize;
    Some((size as u64, committed(s.memory, size)))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again).
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let end = start.saturating_add(len);
    if !committed(at, 1) && !committed((end - 1) as *const u8, 1) {
        return 0;
    }
    if !by_query() {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if readable(lo, 1, true).0 == 1 {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// Whether textures in pipe/bank XOR swizzle modes are uploaded from
/// title memory (`VICTORY_TEXTURE_XOR_UPLOAD=0`: skipped as before).
fn xor_upload() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_XOR_UPLOAD").is_none_or(|v| v != "0"))
}

/// Whether both planes (depth + stencil) of planar depth-stencil title
/// textures are uploaded (`VICTORY_TEXTURE_PLANAR=0`: plane 0 only).
fn planar_upload() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_PLANAR").is_none_or(|v| v != "0"))
}

/// Counts of title textures skipped at upload by reason:
/// `[no_layout, unaddressable_swizzle, msaa, xor_disabled, no_upload_buffer, detile_failed]`.
static SKIPPED: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

pub(crate) fn skipped_counts() -> [u64; 6] {
    SKIPPED.each_ref().map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether a console texture layout can be detiled and uploaded from title
/// memory (`addressable`, and either non-XOR or XOR upload enabled).
pub(crate) fn uploadable(layout: &victory_formats::texture::Layout) -> bool {
    layout.addressable() && (!layout.desc.swizzle.xor() || xor_upload())
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource, host_id: u64) -> Option<Arc<Staging>> {
    use std::sync::atomic::Ordering::Relaxed;
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let history = history_for(resource as usize);
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(tex_layout) = crate::resource::texture_layout(&desc) else {
        SKIPPED[0].fetch_add(1, Relaxed);
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    let layout = tex_layout.planes[0].layout.clone();
    if !uploadable(&layout) {
        if layout.desc.samples > 1 {
            SKIPPED[2].fetch_add(1, Relaxed);
            once!(?desc, "MSAA texture contents not uploaded");
        } else if layout.addressable() {
            SKIPPED[3].fetch_add(1, Relaxed);
            once!(?desc, "texture contents in an XOR mode not uploaded");
        } else {
            SKIPPED[1].fetch_add(1, Relaxed);
            once!(?desc, "texture contents in an unaddressable swizzle mode not uploaded");
        }
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let plane_subs = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let stencil = (planar_upload()
        && tex_layout.planes.len() == 2
        && matches!(d.Format.0, 19 | 20 | 44 | 45))
    .then(|| (tex_layout.planes[1].offset, tex_layout.planes[1].layout.clone()));
    let tiled_size = if stencil.is_some() { tex_layout.size } else { layout.size };
    let count = plane_subs * if stencil.is_some() { 2 } else { 1 };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    if total == 0 || total == u64::MAX {
        SKIPPED[4].fetch_add(1, Relaxed);
        return None;
    }
    let Some((buffer, data, cap)) = host.take_upload(total) else {
        SKIPPED[4].fetch_add(1, Relaxed);
        return None;
    };
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let (detiled, uploaded) = {
        let mut inner = history.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.host_id != host_id || host_id == 0 {
            inner.host_id = host_id;
            inner.uploaded_once = false;
            inner.dirty = true;
            inner.detiled = 0;
            inner.judged = [0, 0];
            if !keep_hash_on() {
                inner.last = Last::default();
            }
        }
        inner.memory = memory as usize;
        inner.size = tiled_size as usize;
        inner.on_demand = on_demand;
        (inner.detiled, inner.uploaded_once && !inner.dirty)
    };
    let s = Arc::new(Staging {
        buffer: std::mem::ManuallyDrop::new(buffer),
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        tiled_size,
        mips: u32::from(d.MipLevels),
        plane_subs,
        stencil,
        bytes: total,
        cap,
        memory,
        host_id,
        history,
        dirty: std::sync::atomic::AtomicBool::new(!uploaded),
        uploaded: std::sync::atomic::AtomicBool::new(uploaded),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(detiled),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }

    #[test]
    fn xor_swizzle_modes_are_uploadable_and_detile_across_mips_and_slices() {
        use victory_formats::texture::{Dimension, Pipes, Swizzle, Texture, TextureDesc};

        let make_layout = |swizzle, slices| {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 64,
                height: 64,
                depth_or_array: slices,
                mips: 4,
                format: 28,
                samples: 1,
                swizzle,
            };
            Texture::for_gpu(&desc, Pipes::SERIES_X, 0)
                .unwrap()
                .planes
                .into_iter()
                .next()
                .unwrap()
                .layout
        };

        assert!(super::uploadable(&make_layout(Swizzle::LINEAR, 1)));
        assert!(super::uploadable(&make_layout(Swizzle::SW_64KB_D, 1)));
        assert!(super::uploadable(&make_layout(Swizzle::SW_64KB_D_X, 1)));
        assert!(super::uploadable(&make_layout(Swizzle::SW_64KB_S_X, 2)));
        assert!(super::uploadable(&make_layout(Swizzle::SW_64KB_R_X, 1)));
        // Unaddressable XOR modes stay skipped even when XOR upload is enabled.
        assert!(!super::uploadable(&make_layout(Swizzle::SW_4KB_R_X, 1)));

        let layout = make_layout(Swizzle::SW_64KB_D_X, 2);
        let mut tiled = vec![0u8; layout.size as usize];
        for slice in 0..2u32 {
            for mip in 0..4u32 {
                let w = (64 >> mip) as usize;
                let h = (64 >> mip) as usize;
                let row_pitch = w * 4;
                let expected: Vec<u8> = (0..h * row_pitch)
                    .map(|i| ((i as u32 + slice * 31 + mip * 7) & 0xff) as u8)
                    .collect();
                layout
                    .tile(&mut tiled, mip, slice, &expected, row_pitch, expected.len())
                    .unwrap();
                let mut got = vec![0u8; expected.len()];
                layout
                    .detile(&tiled, mip, slice, &mut got, row_pitch, expected.len())
                    .unwrap();
                assert_eq!(got, expected, "slice {slice} mip {mip}");
            }
        }
    }

    #[test]
    fn planar_depth_stencil_and_unaligned_bc_detile_into_staging() {
        let Some(_) = crate::host::host() else {
            return;
        };
        // 1. Planar R24G8_TYPELESS (44) and R32G8X24_TYPELESS (19): both depth and stencil planes.
        for format in [44u32, 19u32] {
            let desc_words: [u32; 14] = [3, 0, 0, 0, 8, 0, 8, 0x1_0001, format, 1, 0, 0x100, 0, 0];
            // SAFETY: 56 bytes in ResourceDesc layout.
            let rdesc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(desc_words) };
            let layout = crate::resource::texture_layout(&rdesc).unwrap();
            assert_eq!(layout.planes.len(), 2);
            let mut tiled = vec![0u8; layout.size as usize];
            let depth_linear: Vec<u8> = (0..8 * 8)
                .flat_map(|i| [i as u8, (i * 3) as u8, (i * 5) as u8, 0x7f])
                .collect();
            let stencil_linear: Vec<u8> = (0..8 * 8).map(|i| (i * 11 + 3) as u8).collect();
            layout.planes[0]
                .layout
                .tile(&mut tiled, 0, 0, &depth_linear, 8 * 4, depth_linear.len())
                .unwrap();
            let p1_off = layout.planes[1].offset as usize;
            layout.planes[1]
                .layout
                .tile(&mut tiled[p1_off..], 0, 0, &stencil_linear, 8, stencil_linear.len())
                .unwrap();

            let res = crate::resource::texture_at(tiled.as_ptr() as u64, &desc_words).unwrap();
            let host_tex = crate::resource::host_resource(res).unwrap();
            let host_id = crate::resource::host_id(res).unwrap_or(0);
            let staging = super::staging(res, host_tex, host_id).unwrap();
            assert_eq!(staging.footprints.len(), 2);
            let mut judged = [0, 0];
            staging.detile_with_judgement(&tiled, &mut judged);
            // Verify plane 0 (depth): 4 B/texel in staging buffer for both format 44 and 19.
            let f0 = staging.footprints[0].Footprint;
            // SAFETY: staging buffer is mapped for `staging.bytes` bytes.
            let p0_bytes = unsafe {
                std::slice::from_raw_parts(
                    staging.data.add(staging.footprints[0].Offset as usize),
                    f0.RowPitch as usize * 8,
                )
            };
            for y in 0..8usize {
                for x in 0..8usize {
                    let got = &p0_bytes[y * f0.RowPitch as usize + x * 4..][..4];
                    let want = &depth_linear[(y * 8 + x) * 4..][..4];
                    assert_eq!(got, want, "format {format} depth ({x},{y})");
                }
            }
            // Verify plane 1 (stencil): 1 B/texel.
            let f1 = staging.footprints[1].Footprint;
            // SAFETY: staging buffer is mapped for `staging.bytes` bytes.
            let p1_bytes = unsafe {
                std::slice::from_raw_parts(
                    staging.data.add(staging.footprints[1].Offset as usize),
                    (f1.RowPitch as usize * 7) + 8,
                )
            };
            for y in 0..8usize {
                for x in 0..8usize {
                    assert_eq!(
                        p1_bytes[y * f1.RowPitch as usize + x],
                        stencil_linear[y * 8 + x],
                        "format {format} stencil ({x},{y})"
                    );
                }
            }
            super::unstage(res);
        }

        // 2. Non-multiple-of-4 BC7_UNORM (98) 6x6 texture aligns host copy to 8x8 and detiles.
        let bc_words: [u32; 14] = [3, 0, 0, 0, 6, 0, 6, 0x1_0001, 98, 1, 0, 0x100, 0, 0];
        // SAFETY: 56 bytes in ResourceDesc layout.
        let bc_rdesc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(bc_words) };
        let bc_layout = crate::resource::texture_layout(&bc_rdesc).unwrap();
        let mut bc_tiled = vec![0u8; bc_layout.size as usize];
        let bc_linear: Vec<u8> = (0..4 * 16).map(|i| (i * 7 + 1) as u8).collect();
        bc_layout.planes[0]
            .layout
            .tile(&mut bc_tiled, 0, 0, &bc_linear, 2 * 16, bc_linear.len())
            .unwrap();
        let res = crate::resource::texture_at(bc_tiled.as_ptr() as u64, &bc_words).unwrap();
        let host_tex = crate::resource::host_resource(res).unwrap();
        let host_id = crate::resource::host_id(res).unwrap_or(0);
        let staging = super::staging(res, host_tex, host_id).unwrap();
        let mut judged = [0, 0];
        staging.detile_with_judgement(&bc_tiled, &mut judged);
        let f_bc = staging.footprints[0].Footprint;
        // SAFETY: staging buffer is mapped for `staging.bytes` bytes.
        let bc_bytes = unsafe {
            std::slice::from_raw_parts(
                staging.data.add(staging.footprints[0].Offset as usize),
                f_bc.RowPitch as usize + 2 * 16,
            )
        };
        assert_eq!(&bc_bytes[..32], &bc_linear[..32]);
        assert_eq!(&bc_bytes[f_bc.RowPitch as usize..][..32], &bc_linear[32..64]);
        super::unstage(res);
    }

    fn partial_throttle_bypasses_wait_when_guard_sees_write() {
        assert!(!super::should_skip_partial(None, 100, false, true));
        // Unwritten partial texture waits 30 frames even with immediate enabled.
        assert!(super::should_skip_partial(Some(100), 101, false, true));
        assert!(super::should_skip_partial(Some(100), 129, false, true));
        assert!(!super::should_skip_partial(Some(100), 130, false, true));
        // Guard saw a new write or mapped page: refresh immediately when enabled,
        // wait all 30 frames when disabled (`VICTORY_PARTIAL_GUARD_IMMEDIATE=0`).
        assert!(!super::should_skip_partial(Some(100), 101, true, true));
        assert!(super::should_skip_partial(Some(100), 101, true, false));
    }

    #[test]
    fn read_scratch_pool_reuses_caps_and_trims_idle() {
        use super::{READ_POOL_MAX_ITEM, ReadScratchPool};
        let mut pool = ReadScratchPool::default();
        let buf = pool.take(1024);
        assert!(buf.len() >= 1024);
        let cap = buf.capacity();
        pool.give(buf, 100);
        assert_eq!(pool.bytes, cap);

        // Reuses the pooled buffer rather than allocating a second one.
        let reused = pool.take(512);
        assert_eq!(reused.capacity(), cap);
        assert_eq!(pool.bytes, 0);
        pool.give(reused, 200);

        // Oversized one-off buffer is dropped rather than hoarded in the pool.
        let huge = vec![0u8; READ_POOL_MAX_ITEM + 4096];
        pool.give(huge, 300);
        assert_eq!(pool.kept.len(), 1);

        // Idle trim drops old entries.
        pool.trim_idle(1_000, 1_250);
        assert!(pool.kept.is_empty());
        assert_eq!(pool.bytes, 0);
    }

    #[test]
    fn rewritten_often_policy_keeps_frequent_and_drops_infrequent() {
        use super::rewritten_often;
        // Initial upload (0 rewrites) or single rewrite: not rewritten often -> drop staging.
        assert!(!rewritten_often(0, 0, 10, 120));
        assert!(!rewritten_often(1, 10, 10, 120));
        // Two or more rewrites within the frame window: keep staging.
        assert!(rewritten_often(2, 10, 10, 120));
        assert!(rewritten_often(2, 10, 130, 120));
        // Stale rewrites past the window: drop staging again.
        assert!(!rewritten_often(2, 10, 131, 120));
    }

    #[test]
    fn kept_hash_skips_reupload_when_memory_unchanged_and_detects_edits() {
        use super::{Last, check_memory_changed};
        let mut backing = vec![0x3cu8; 4096];
        let mut last = Last::default();

        // Initial commit pass detiles and records the blake3 hash.
        let mut detiled = 0usize;
        let (_, changed) = check_memory_changed(backing.as_ptr(), backing.len(), &mut last, true, |_| {
            detiled += 1;
        });
        assert!(changed);
        assert_eq!(detiled, 1);
        assert!(last.hash.is_some());

        // After staging is dropped, checking with commit_change=false against
        // unchanged memory finds no change and keeps the hash.
        let saved_hash = last.hash;
        let (_, changed) = check_memory_changed(backing.as_ptr(), backing.len(), &mut last, false, |_| {
            unreachable!("unchanged memory must not detile");
        });
        assert!(!changed);
        assert_eq!(last.hash, saved_hash);

        // Mutate memory: non-committing check detects the change while leaving
        // the old hash intact so the subsequent Staging refresh still detiles.
        backing[123] ^= 0xff;
        let (_, changed) = check_memory_changed(backing.as_ptr(), backing.len(), &mut last, false, |_| {});
        assert!(changed);
        assert_eq!(last.hash, saved_hash);

        let (_, changed) = check_memory_changed(backing.as_ptr(), backing.len(), &mut last, true, |_| {
            detiled += 1;
        });
        assert!(changed);
        assert_eq!(detiled, 2);
        assert_ne!(last.hash, saved_hash);
    }
}
