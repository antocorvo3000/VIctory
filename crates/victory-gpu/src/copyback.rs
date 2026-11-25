//! Lazy (on-demand) GPU->CPU copy-back, per-shadow copy-back telemetry, and
//! redundant upload accounting.
//!
//! All behaviour changes are gated behind `VICTORY_*` environment switches that
//! default off:
//! - `VICTORY_COPYBACK_STATS=1`: logs per-shadow copy-back census (bytes
//!   declared, fetched, merged, CPU reads, preparing reads, GPU overwrites).
//! - `VICTORY_COPYBACK_PROBE=1`: arms `PAGE_NOACCESS` traps on copied-back
//!   pages without skipping the copy-back, to measure which buffers the title
//!   ever reads on the CPU before the GPU writes them again.
//! - `VICTORY_UPLOAD_STATS=1`: logs per-shadow upload byte breakdown (total
//!   updated vs identical bytes already held vs copy-back-induced refreshes vs
//!   64 KiB write-watch page padding).
//! - `VICTORY_DEFER_COPYBACK=1`: defers fetching and merging differing card
//!   pieces until first CPU read/write fault (via `watch.rs` VEH) or internal
//!   `preparing` / OS read/write consumer. GPU-only ranges never touched by
//!   the CPU stay on the GPU.
//! - `VICTORY_DEFER_COPYBACK_PC=1`: also defers non-card (`self.card.is_none()`)
//!   GPU-written shadow ranges by holding their pre-write slice until access.
//! - `VICTORY_DEFER_AT_FENCE=1`: flushes deferred ranges when the title waits
//!   on a GPU fence (`SetEventOnCompletion`).
//! - `VICTORY_COPYBACK_KEEP_COPIED=1`: avoids re-copying & re-uploading
//!   unchanged bytes that were just written into title memory by copy-back.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::shadow::Shadow;

pub(crate) const KERNEL_PAGE: usize = 4 << 10;

fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v == "1")
}

#[cfg(test)]
mod test_overrides {
    use std::cell::Cell;

    std::thread_local! {
        pub(super) static DEFER: Cell<Option<bool>> = const { Cell::new(None) };
        pub(super) static DEFER_PC: Cell<Option<bool>> = const { Cell::new(None) };
        pub(super) static KEEP_COPIED: Cell<Option<bool>> = const { Cell::new(None) };
        pub(super) static STATS: Cell<Option<bool>> = const { Cell::new(None) };
        pub(super) static UPLOAD_STATS: Cell<Option<bool>> = const { Cell::new(None) };
        pub(super) static PROBE: Cell<Option<bool>> = const { Cell::new(None) };
    }
}

/// Whether `VICTORY_COPYBACK_STATS=1` is active (default false).
pub(crate) fn stats_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = test_overrides::STATS.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_COPYBACK_STATS"))
}

/// Whether `VICTORY_COPYBACK_PROBE=1` is active (default false).
pub(crate) fn probe_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = test_overrides::PROBE.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_COPYBACK_PROBE"))
}

/// Whether `VICTORY_UPLOAD_STATS=1` is active (default false).
pub(crate) fn upload_stats_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = test_overrides::UPLOAD_STATS.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_UPLOAD_STATS"))
}

/// Whether `VICTORY_DEFER_COPYBACK=1` is active (default false).
pub(crate) fn defer_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = test_overrides::DEFER.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_DEFER_COPYBACK"))
}

/// Whether `VICTORY_DEFER_COPYBACK_PC=1` is active (default false).
pub(crate) fn defer_pc_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = test_overrides::DEFER_PC.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_DEFER_COPYBACK_PC"))
}

/// Whether `VICTORY_DEFER_AT_FENCE=1` is active (default false).
pub(crate) fn defer_at_fence() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_DEFER_AT_FENCE"))
}

/// Whether `VICTORY_COPYBACK_KEEP_COPIED=1` is active (default false).
pub(crate) fn keep_copied_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = test_overrides::KEEP_COPIED.with(|c| c.get()) {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_COPYBACK_KEEP_COPIED"))
}

/// Fast-path atomic counters so disabled / idle frames pay one relaxed load.
static ACTIVE_DEFERRED: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_PROBED: AtomicUsize = AtomicUsize::new(0);
static SEQ: AtomicU64 = AtomicU64::new(1);

pub(crate) fn has_active_guards() -> bool {
    ACTIVE_DEFERRED.load(Ordering::Relaxed) != 0 || ACTIVE_PROBED.load(Ordering::Relaxed) != 0
}

// ---------------------------------------------------------------------------
// Telemetry: copy-back census & upload waste breakdown
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct CopybackShadowStats {
    start: u64,
    len: u64,
    card: bool,
    window: bool,
    declares: u64,
    pieces_checked: u64,
    pieces_differing: u64,
    fetched_bytes: u64,
    merged_bytes: u64,
    deferred_pieces: u64,
    deferred_bytes: u64,
    overwritten_no_cpu_bytes: u64,
    cpu_reads: u64,
    cpu_read_bytes: u64,
    prep_reads: u64,
    prep_read_bytes: u64,
}

#[derive(Default, Clone)]
struct UploadShadowStats {
    start: u64,
    len: u64,
    card: bool,
    copied_bytes: u64,
    unchanged_bytes: u64,
    copyback_induced_bytes: u64,
    copyback_unchanged_bytes: u64,
    page_padding_bytes: u64,
    card_changed_bytes: u64,
}

#[derive(Default)]
struct Telemetry {
    copyback: HashMap<u64, CopybackShadowStats>,
    last_copyback_log_ms: u64,
    last_copyback_frame: u64,
    upload: HashMap<u64, UploadShadowStats>,
    last_upload_log_ms: u64,
    last_upload_frame: u64,
    /// 4 KiB pages (addr >> 12) written by copy-back since the previous update.
    copyback_pages: HashSet<usize>,
}

static TELEMETRY: Mutex<Option<Telemetry>> = Mutex::new(None);

fn with_telemetry<F: FnOnce(&mut Telemetry)>(f: F) {
    let mut guard = TELEMETRY.lock().unwrap_or_else(|e| e.into_inner());
    let t = guard.get_or_insert_with(Telemetry::default);
    f(t);
}

fn entry_cb<'a>(t: &'a mut Telemetry, s: &Shadow) -> &'a mut CopybackShadowStats {
    t.copyback.entry(s.id).or_insert_with(|| CopybackShadowStats {
        start: s.start,
        len: s.len,
        card: s.carded(),
        window: s.is_window(),
        ..Default::default()
    })
}

fn entry_up<'a>(t: &'a mut Telemetry, s: &Shadow) -> &'a mut UploadShadowStats {
    t.upload.entry(s.id).or_insert_with(|| UploadShadowStats {
        start: s.start,
        len: s.len,
        card: s.carded(),
        ..Default::default()
    })
}

pub(crate) fn stat_declare(s: &Shadow) {
    if !stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        entry_cb(t, s).declares += 1;
    });
}

pub(crate) fn stat_pieces_checked(s: &Shadow, checked: u64, differing: u64) {
    if !stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        let e = entry_cb(t, s);
        e.pieces_checked += checked;
        e.pieces_differing += differing;
    });
}

pub(crate) fn stat_fetched_merged(s: &Shadow, fetched: u64, merged: u64) {
    if !stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        let e = entry_cb(t, s);
        e.fetched_bytes += fetched;
        e.merged_bytes += merged;
    });
}

pub(crate) fn stat_deferred(s: &Shadow, bytes: u64) {
    if !stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        let e = entry_cb(t, s);
        e.deferred_pieces += 1;
        e.deferred_bytes += bytes;
    });
}

pub(crate) fn stat_overwritten_no_cpu(s: &Shadow, bytes: u64) {
    if !stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        entry_cb(t, s).overwritten_no_cpu_bytes += bytes;
    });
}

pub(crate) fn stat_cpu_read(s: &Shadow, bytes: u64) {
    if !stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        let e = entry_cb(t, s);
        e.cpu_reads += 1;
        e.cpu_read_bytes += bytes;
    });
}

pub(crate) fn stat_prep_read(s: &Shadow, bytes: u64) {
    if !stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        let e = entry_cb(t, s);
        e.prep_reads += 1;
        e.prep_read_bytes += bytes;
    });
}

/// Records copy-back writes into title memory for `VICTORY_UPLOAD_STATS=1`.
pub(crate) fn note_copyback_write(addr: usize, len: usize) {
    if len == 0 || !upload_stats_enabled() {
        return;
    }
    let first = addr / KERNEL_PAGE;
    let last = addr.saturating_add(len - 1) / KERNEL_PAGE;
    with_telemetry(|t| {
        for p in first..=last {
            t.copyback_pages.insert(p);
        }
    });
}

/// Compares title memory `[from, to)` against existing shadow bytes before
/// overwriting them, categorizing unchanged bytes and copy-back feedback.
pub(crate) fn stat_upload_slice(s: &Shadow, from: usize, to: usize, staged: *const u8) {
    if from >= to || !upload_stats_enabled() {
        return;
    }
    let len = to - from;
    let mut buf = vec![0u8; len.min(64 << 10)];
    let mut pos = from;
    let mut total_unchanged = 0u64;
    let mut total_cb_bytes = 0u64;
    let mut total_cb_unchanged = 0u64;
    let mut total_padding = 0u64;
    let mut dirty_subpages = Vec::new();
    let base = s.start as usize;

    while pos < to {
        let n = buf.len().min(to - pos);
        let title_slice = &mut buf[..n];
        if !crate::host::read_raw(pos as *const u8, title_slice) {
            pos += n;
            continue;
        }
        // SAFETY: callers of `stat_upload_slice` pass `self.data.add(from - start)`
        // within `self.len`, before modifying those bytes.
        let held = unsafe { std::slice::from_raw_parts(staged.add(pos - base), n) };

        // Inspect 4 KiB blocks for padding waste and unchanged bytes.
        let mut off = 0;
        while off < n {
            let blk = (n - off).min(KERNEL_PAGE);
            let t_blk = &title_slice[off..off + blk];
            let h_blk = &held[off..off + blk];
            if t_blk == h_blk {
                total_unchanged += blk as u64;
                total_padding += blk as u64;
            } else {
                // Count identical 64-byte cache lines inside partially changed 4K blocks.
                let mut l = 0;
                while l < blk {
                    let cn = (blk - l).min(64);
                    if t_blk[l..l + cn] == h_blk[l..l + cn] {
                        total_unchanged += cn as u64;
                    }
                    l += cn;
                }
            }
            dirty_subpages.push(((pos + off) / KERNEL_PAGE, blk as u64, t_blk == h_blk));
            off += blk;
        }
        pos += n;
    }

    with_telemetry(|t| {
        for (p, blk_bytes, same) in dirty_subpages {
            if t.copyback_pages.contains(&p) {
                total_cb_bytes += blk_bytes;
                if same {
                    total_cb_unchanged += blk_bytes;
                }
            }
        }
        let e = entry_up(t, s);
        e.copied_bytes += len as u64;
        e.unchanged_bytes += total_unchanged;
        e.copyback_induced_bytes += total_cb_bytes;
        e.copyback_unchanged_bytes += total_cb_unchanged;
        e.page_padding_bytes += total_padding;
    });
}

pub(crate) fn stat_card_changed(s: &Shadow, bytes: u64) {
    if bytes == 0 || !upload_stats_enabled() {
        return;
    }
    with_telemetry(|t| {
        entry_up(t, s).card_changed_bytes += bytes;
    });
}

/// Periodically logs copy-back and upload statistics (roughly every 2 seconds).
pub(crate) fn maybe_report() {
    let cb = stats_enabled();
    let up = upload_stats_enabled();
    if !cb && !up {
        return;
    }
    let now_ms = crate::host::now_ms();
    let frame = crate::frame::current();
    let mut cb_snap: Option<(u64, Vec<(u64, CopybackShadowStats)>)> = None;
    let mut up_snap: Option<(u64, Vec<(u64, UploadShadowStats)>)> = None;

    with_telemetry(|t| {
        if cb && now_ms.saturating_sub(t.last_copyback_log_ms) >= 2_000 {
            let frames = frame.saturating_sub(t.last_copyback_frame).max(1);
            t.last_copyback_log_ms = now_ms;
            t.last_copyback_frame = frame;
            if !t.copyback.is_empty() {
                cb_snap = Some((frames, t.copyback.drain().collect()));
            }
        }
        if up && now_ms.saturating_sub(t.last_upload_log_ms) >= 2_000 {
            let frames = frame.saturating_sub(t.last_upload_frame).max(1);
            t.last_upload_log_ms = now_ms;
            t.last_upload_frame = frame;
            t.copyback_pages.clear();
            if !t.upload.is_empty() {
                up_snap = Some((frames, t.upload.drain().collect()));
            }
        }
    });

    if let Some((frames, mut rows)) = cb_snap {
        rows.sort_by_key(|(_, r)| {
            std::cmp::Reverse(r.fetched_bytes.saturating_add(r.deferred_bytes))
        });
        let (mut tot_fetch, mut tot_merge, mut tot_def, mut tot_over, mut tot_cpu, mut tot_prep) =
            (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for (_, r) in &rows {
            tot_fetch += r.fetched_bytes;
            tot_merge += r.merged_bytes;
            tot_def += r.deferred_bytes;
            tot_over += r.overwritten_no_cpu_bytes;
            tot_cpu += r.cpu_read_bytes;
            tot_prep += r.prep_read_bytes;
        }
        let mb = |b: u64| b as f64 / (frames as f64 * (1024.0 * 1024.0));
        let top: Vec<String> = rows
            .iter()
            .take(12)
            .filter(|(_, r)| r.fetched_bytes + r.deferred_bytes > 0)
            .map(|(id, r)| {
                format!(
                    "#{} [{:#x}+{:.1}MB {}{}] decl={:.1}/f chk={:.0} diff={:.0} fetch={:.2}MB/f merge={:.2}MB/f def={:.2}MB/f over_no_cpu={:.2}MB/f cpu_rd={}({:.2}MB/f) prep_rd={}({:.2}MB/f)",
                    id,
                    r.start,
                    r.len as f64 / (1024.0 * 1024.0),
                    if r.card { "card" } else { "pc" },
                    if r.window { ",win" } else { "" },
                    r.declares as f64 / frames as f64,
                    r.pieces_checked as f64 / frames as f64,
                    r.pieces_differing as f64 / frames as f64,
                    mb(r.fetched_bytes),
                    mb(r.merged_bytes),
                    mb(r.deferred_bytes),
                    mb(r.overwritten_no_cpu_bytes),
                    r.cpu_reads,
                    mb(r.cpu_read_bytes),
                    r.prep_reads,
                    mb(r.prep_read_bytes),
                )
            })
            .collect();
        tracing::info!(
            frames,
            shadows = rows.len(),
            fetch_mb_f = format_args!("{:.2}", mb(tot_fetch)),
            merge_mb_f = format_args!("{:.2}", mb(tot_merge)),
            defer_mb_f = format_args!("{:.2}", mb(tot_def)),
            overwritten_no_cpu_mb_f = format_args!("{:.2}", mb(tot_over)),
            cpu_read_mb_f = format_args!("{:.2}", mb(tot_cpu)),
            prep_read_mb_f = format_args!("{:.2}", mb(tot_prep)),
            top = ?top,
            "copyback stats (per frame)"
        );
    }

    if let Some((frames, mut rows)) = up_snap {
        rows.sort_by_key(|(_, r)| std::cmp::Reverse(r.copied_bytes));
        let (mut tot_cp, mut tot_unch, mut tot_cb, mut tot_cb_unch, mut tot_pad, mut tot_cc) =
            (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for (_, r) in &rows {
            tot_cp += r.copied_bytes;
            tot_unch += r.unchanged_bytes;
            tot_cb += r.copyback_induced_bytes;
            tot_cb_unch += r.copyback_unchanged_bytes;
            tot_pad += r.page_padding_bytes;
            tot_cc += r.card_changed_bytes;
        }
        let mb = |b: u64| b as f64 / (frames as f64 * (1024.0 * 1024.0));
        let top: Vec<String> = rows
            .iter()
            .take(10)
            .filter(|(_, r)| r.copied_bytes > 0)
            .map(|(id, r)| {
                format!(
                    "#{} [{:#x}+{:.1}MB {}] cp={:.2}MB/f unch={:.2}MB/f cb_ind={:.2}MB/f (unch={:.2}) pad4k={:.2}MB/f card_up={:.2}MB/f",
                    id,
                    r.start,
                    r.len as f64 / (1024.0 * 1024.0),
                    if r.card { "card" } else { "pc" },
                    mb(r.copied_bytes),
                    mb(r.unchanged_bytes),
                    mb(r.copyback_induced_bytes),
                    mb(r.copyback_unchanged_bytes),
                    mb(r.page_padding_bytes),
                    mb(r.card_changed_bytes),
                )
            })
            .collect();
        tracing::info!(
            frames,
            shadows = rows.len(),
            copied_mb_f = format_args!("{:.2}", mb(tot_cp)),
            unchanged_mb_f = format_args!("{:.2}", mb(tot_unch)),
            copyback_induced_mb_f = format_args!("{:.2}", mb(tot_cb)),
            copyback_unchanged_mb_f = format_args!("{:.2}", mb(tot_cb_unch)),
            pad4k_mb_f = format_args!("{:.2}", mb(tot_pad)),
            card_changed_mb_f = format_args!("{:.2}", mb(tot_cc)),
            top = ?top,
            "upload stats (per frame)"
        );
    }
}

// ---------------------------------------------------------------------------
// Deferred / Probed Copy-Back State & Windows Page Guarding
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessReason {
    CpuFault(bool),
    Preparing,
    KernelWrite,
    Remapping,
    Unregister,
    FenceWait,
}

#[derive(Clone)]
pub(crate) struct DeferredPiece {
    pub(crate) seq: u64,
    pub(crate) shadow: Arc<Shadow>,
    pub(crate) lo: usize,
    pub(crate) hi: usize,
    pub(crate) abs_lo: usize,
    pub(crate) abs_hi: usize,
    pub(crate) card: bool,
    pub(crate) pc_before: Option<Vec<u8>>,
}

#[derive(Default)]
struct GuardState {
    /// Keyed by `(shadow.id, lo, hi)`.
    deferred: BTreeMap<(u64, usize, usize), DeferredPiece>,
    /// Count of deferred entries overlapping each 4 KiB page (`addr >> 12`).
    deferred_pages: HashMap<usize, u32>,
    /// Pages protected ONLY for `VICTORY_COPYBACK_PROBE=1` -> `(shadow, byte_len)`.
    probed_pages: HashMap<usize, (Arc<Shadow>, usize)>,
}

/// Lock ordering: ALWAYS `MATERIALIZE` -> `STATE` -> GPU/host -> `Shadow::before` -> `watch::GUARDS`.
/// Never acquire `MATERIALIZE` or `STATE` while holding `Shadow::before` or `watch::GUARDS`.
static STATE: Mutex<Option<GuardState>> = Mutex::new(None);
static MATERIALIZE: Mutex<()> = Mutex::new(());

fn with_state<R>(f: impl FnOnce(&mut GuardState) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(GuardState::default);
    f(s)
}

fn sync_active_counts(s: &GuardState) {
    ACTIVE_DEFERRED.store(s.deferred.len(), Ordering::Relaxed);
    ACTIVE_PROBED.store(s.probed_pages.len(), Ordering::Relaxed);
}

#[cfg(windows)]
fn set_page_protection(page_addr: usize, protect: u32) -> Option<u32> {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let mut old = 0u32;
    // SAFETY: `page_addr` is page-aligned and `VirtualProtect` validates it.
    let ok = unsafe { VirtualProtect(page_addr as *const _, KERNEL_PAGE, protect, &mut old) };
    (ok != 0).then_some(old)
}

#[cfg(windows)]
fn is_page_rw(page_addr: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualQuery,
    };
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let got = unsafe {
        VirtualQuery(
            page_addr as *const _,
            &mut info,
            std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if got < std::mem::size_of::<MEMORY_BASIC_INFORMATION>() || info.State != MEM_COMMIT {
        return false;
    }
    let base = info.Protect & 0xff;
    base == PAGE_READWRITE || base == PAGE_EXECUTE_READWRITE
}

#[cfg(windows)]
fn arm_pages_noaccess(first_page: usize, last_page: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};
    crate::watch::install_veh();
    let mut armed: Vec<(usize, u32)> = Vec::with_capacity(last_page - first_page + 1);
    for p in first_page..=last_page {
        let addr = p * KERNEL_PAGE;
        if !is_page_rw(addr) {
            for (rp, old) in armed.into_iter().rev() {
                let _ = set_page_protection(rp, old);
            }
            return false;
        }
        let Some(old) = set_page_protection(addr, PAGE_NOACCESS) else {
            for (rp, old) in armed.into_iter().rev() {
                let _ = set_page_protection(rp, old);
            }
            return false;
        };
        if (old & 0xff) != PAGE_READWRITE {
            let _ = set_page_protection(addr, old);
            for (rp, pold) in armed.into_iter().rev() {
                let _ = set_page_protection(rp, pold);
            }
            return false;
        }
        armed.push((addr, old));
    }
    true
}

#[cfg(not(windows))]
fn arm_pages_noaccess(_first_page: usize, _last_page: usize) -> bool {
    false
}

#[cfg(windows)]
fn restore_pages_rw(pages: &[usize]) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    for &p in pages {
        let _ = set_page_protection(p * KERNEL_PAGE, PAGE_READWRITE);
    }
}

#[cfg(not(windows))]
fn restore_pages_rw(_pages: &[usize]) {}

/// Returns whether a 4 KiB page (`page_index` = `addr >> 12`) is currently
/// protected by deferred copy-back or copy-back probing.
pub(crate) fn is_page_guarded(page_index: usize) -> bool {
    if !has_active_guards() {
        return false;
    }
    with_state(|s| {
        s.deferred_pages.contains_key(&page_index) || s.probed_pages.contains_key(&page_index)
    })
}

/// Returns whether any 4 KiB page inside a 64 KiB `watch::BLOCK` (`block_index`)
/// is currently protected by deferred copy-back or copy-back probing.
pub(crate) fn block_has_guarded_pages(block_index: usize) -> bool {
    if !has_active_guards() {
        return false;
    }
    let first = block_index * 16;
    with_state(|s| {
        (first..first + 16)
            .any(|p| s.deferred_pages.contains_key(&p) || s.probed_pages.contains_key(&p))
    })
}

/// Checks whether `shadow` has a clean deferred piece completely covering
/// `[from, to)` (`from`, `to` are title addresses), allowing `Shadow::update`
/// to avoid re-reading / re-uploading GPU-only memory.
pub(crate) fn shadow_covers_clean_deferred(shadow_id: u64, from: usize, to: usize) -> bool {
    if ACTIVE_DEFERRED.load(Ordering::Relaxed) == 0 || from >= to {
        return false;
    }
    with_state(|s| {
        s.deferred
            .range((shadow_id, 0, 0)..=(shadow_id, usize::MAX, usize::MAX))
            .any(|(_, p)| p.abs_lo <= from && to <= p.abs_hi)
    })
}

fn too_many_views(addr: usize, len: usize) -> bool {
    const MANY_VIEWS: usize = 64;
    let first = addr / crate::watch::PAGE;
    let last = addr.saturating_add(len).div_ceil(crate::watch::PAGE);
    (first..last).any(|b| crate::inplace::pages_of(b * crate::watch::PAGE).len() > MANY_VIEWS)
}

/// Arms `VICTORY_COPYBACK_PROBE=1` on `[addr, addr + len)` after a synchronous
/// copy-back write without deferring it, to measure CPU reads before next GPU
/// write in-game.
pub(crate) fn probe_copied_back(shadow: &Arc<Shadow>, addr: usize, len: usize) {
    if !probe_enabled() || defer_enabled() || len == 0 {
        return;
    }
    if too_many_views(addr, len) {
        return;
    }
    let first = addr / KERNEL_PAGE;
    let last = addr.saturating_add(len - 1) / KERNEL_PAGE;
    let _mat = MATERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let mut need = Vec::new();
    with_state(|s| {
        for p in first..=last {
            if !s.deferred_pages.contains_key(&p) && !s.probed_pages.contains_key(&p) {
                need.push(p);
            }
        }
    });
    if need.is_empty() {
        return;
    }
    let mut ok_pages = Vec::new();
    for p in need {
        if arm_pages_noaccess(p, p) {
            ok_pages.push(p);
        }
    }
    if ok_pages.is_empty() {
        return;
    }
    with_state(|s| {
        for p in ok_pages {
            s.probed_pages.insert(p, (shadow.clone(), KERNEL_PAGE));
        }
        sync_active_counts(s);
    });
}

/// Clears any `VICTORY_COPYBACK_PROBE=1` marks for `shadow` on `[addr, addr + len)`
/// when a new GPU declaration/copyback runs on it (GPU overwrote before CPU read).
pub(crate) fn note_gpu_reuse(shadow: &Arc<Shadow>, addr: usize, len: usize) {
    if len == 0 {
        return;
    }
    if ACTIVE_PROBED.load(Ordering::Relaxed) != 0 {
        let first = addr / KERNEL_PAGE;
        let last = addr.saturating_add(len - 1) / KERNEL_PAGE;
        let mut to_restore = Vec::new();
        let mut overwritten = 0u64;
        with_state(|s| {
            for p in first..=last {
                if let Some((ps, b)) = s.probed_pages.remove(&p) {
                    if !s.deferred_pages.contains_key(&p) {
                        to_restore.push(p);
                    }
                    if ps.id == shadow.id {
                        overwritten += b as u64;
                    }
                }
            }
            sync_active_counts(s);
        });
        restore_pages_rw(&to_restore);
        if overwritten > 0 {
            stat_overwritten_no_cpu(shadow, overwritten);
        }
    }
}

/// Attempts to defer copy-back of card piece `shadow[lo..hi)` under `MATERIALIZE`
/// lock. Returns `true` if successfully deferred (and protected `PAGE_NOACCESS`),
/// `false` if the caller must fetch & merge immediately.
pub(crate) fn try_defer_card_piece(shadow: &Arc<Shadow>, lo: usize, hi: usize) -> bool {
    if !defer_enabled() || lo >= hi || !shadow.carded() {
        return false;
    }
    let abs_lo = shadow.start as usize + lo;
    let abs_hi = shadow.start as usize + hi;
    let len = hi - lo;
    if too_many_views(abs_lo, len) {
        return false;
    }
    let first_page = abs_lo / KERNEL_PAGE;
    let last_page = (abs_hi - 1) / KERNEL_PAGE;

    let _mat = MATERIALIZE.lock().unwrap_or_else(|e| e.into_inner());

    // Check if CPU already dirtied any watched pages in [abs_lo, abs_hi) since
    // the shadow last synced; if so, fetch & merge immediately to stay safe.
    if !shadow.is_clean_since_update(abs_lo, abs_hi) {
        return false;
    }

    let key = (shadow.id, lo, hi);
    let already_same = with_state(|s| s.deferred.contains_key(&key));
    if already_same {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        with_state(|s| {
            if let Some(entry) = s.deferred.get_mut(&key) {
                entry.seq = seq;
            }
        });
        stat_overwritten_no_cpu(shadow, len as u64);
        stat_deferred(shadow, len as u64);
        return true;
    }

    // Determine pages that are not yet PAGE_NOACCESS from our own guards.
    let new_pages: Vec<usize> = with_state(|s| {
        (first_page..=last_page)
            .filter(|p| !s.deferred_pages.contains_key(p) && !s.probed_pages.contains_key(p))
            .collect()
    });

    let mut newly_armed = Vec::new();
    for &p in &new_pages {
        if arm_pages_noaccess(p, p) {
            newly_armed.push(p);
        } else {
            restore_pages_rw(&newly_armed);
            return false;
        }
    }

    // Re-check after arming PAGE_NOACCESS: if a CPU thread wrote in the race
    // window before VirtualProtect completed, bail cleanly and fetch now.
    if !shadow.is_clean_since_update(abs_lo, abs_hi) {
        restore_pages_rw(&newly_armed);
        return false;
    }

    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    with_state(|s| {
        for p in first_page..=last_page {
            // Convert any probe mark on these pages into real deferred tracking.
            s.probed_pages.remove(&p);
            *s.deferred_pages.entry(p).or_insert(0) += 1;
        }
        s.deferred.insert(
            key,
            DeferredPiece {
                seq,
                shadow: shadow.clone(),
                lo,
                hi,
                abs_lo,
                abs_hi,
                card: true,
                pc_before: None,
            },
        );
        sync_active_counts(s);
    });
    stat_deferred(shadow, len as u64);
    true
}

/// Attempts to defer a non-card (PC-memory) shadow copyback slice (`[lo, hi)`),
/// gated behind `VICTORY_DEFER_COPYBACK_PC=1`.
pub(crate) fn try_defer_pc_piece(shadow: &Arc<Shadow>, lo: usize, hi: usize, before: &[u8]) -> bool {
    if !defer_pc_enabled() || lo >= hi || shadow.carded() || before.len() != hi - lo {
        return false;
    }
    let abs_lo = shadow.start as usize + lo;
    let abs_hi = shadow.start as usize + hi;
    let len = hi - lo;
    if too_many_views(abs_lo, len) {
        return false;
    }
    let first_page = abs_lo / KERNEL_PAGE;
    let last_page = (abs_hi - 1) / KERNEL_PAGE;

    let _mat = MATERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    if !shadow.is_clean_since_update(abs_lo, abs_hi) {
        return false;
    }

    let key = (shadow.id, lo, hi);
    let already_same = with_state(|s| s.deferred.contains_key(&key));
    if already_same {
        // Keep the oldest pre-GPU `pc_before` snapshot! Only advance seq so order
        // reflects the latest GPU write.
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        with_state(|s| {
            if let Some(entry) = s.deferred.get_mut(&key) {
                entry.seq = seq;
            }
        });
        stat_overwritten_no_cpu(shadow, len as u64);
        stat_deferred(shadow, len as u64);
        return true;
    }

    let new_pages: Vec<usize> = with_state(|s| {
        (first_page..=last_page)
            .filter(|p| !s.deferred_pages.contains_key(p) && !s.probed_pages.contains_key(p))
            .collect()
    });
    let mut newly_armed = Vec::new();
    for &p in &new_pages {
        if arm_pages_noaccess(p, p) {
            newly_armed.push(p);
        } else {
            restore_pages_rw(&newly_armed);
            return false;
        }
    }
    if !shadow.is_clean_since_update(abs_lo, abs_hi) {
        restore_pages_rw(&newly_armed);
        return false;
    }

    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    with_state(|s| {
        for p in first_page..=last_page {
            s.probed_pages.remove(&p);
            *s.deferred_pages.entry(p).or_insert(0) += 1;
        }
        s.deferred.insert(
            key,
            DeferredPiece {
                seq,
                shadow: shadow.clone(),
                lo,
                hi,
                abs_lo,
                abs_hi,
                card: false,
                pc_before: Some(before.to_vec()),
            },
        );
        sync_active_counts(s);
    });
    stat_deferred(shadow, len as u64);
    true
}

/// Optional test hook overriding how card deferred pieces are fetched & merged.
#[cfg(test)]
type FetchHook = Box<dyn Fn(&[DeferredPiece]) -> usize + Send + Sync>;
#[cfg(test)]
static TEST_FETCH_HOOK: Mutex<Option<FetchHook>> = Mutex::new(None);
#[cfg(test)]
pub(crate) fn set_test_fetch_hook(hook: Option<FetchHook>) {
    *TEST_FETCH_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = hook;
}

/// Materializes all deferred pieces intersecting the 4 KiB pages covering
/// `[addr, addr + len)` (expanding across connected boundary pages so every
/// restored page has zero remaining deferred references).
pub(crate) fn materialize_range(addr: usize, len: usize, reason: AccessReason) {
    if !has_active_guards() || len == 0 {
        return;
    }
    let end = addr.saturating_add(len);
    let req_first = addr / KERNEL_PAGE;
    let req_last = (end - 1) / KERNEL_PAGE;

    let _mat = MATERIALIZE.lock().unwrap_or_else(|e| e.into_inner());

    // Handle probe-only pages cleanly first if no deferred entries cover them.
    let mut probed_hit: Option<(Arc<Shadow>, usize)> = None;
    let mut probed_to_restore = Vec::new();
    if ACTIVE_PROBED.load(Ordering::Relaxed) != 0 {
        with_state(|s| {
            for p in req_first..=req_last {
                if let Some((sh, b)) = s.probed_pages.remove(&p) {
                    if !s.deferred_pages.contains_key(&p) {
                        probed_to_restore.push(p);
                    }
                    if matches!(reason, AccessReason::CpuFault(_)) {
                        probed_hit = Some((sh, b));
                    }
                }
            }
            sync_active_counts(s);
        });
        restore_pages_rw(&probed_to_restore);
        if let Some((sh, b)) = probed_hit {
            stat_cpu_read(&sh, b as u64);
        }
    }

    if ACTIVE_DEFERRED.load(Ordering::Relaxed) == 0 {
        return;
    }

    // Find connected component of 4 KiB pages and deferred entries intersecting
    // `[req_first, req_last]`, remove them from STATE while STILL holding
    // `MATERIALIZE` (and while pages are STILL `PAGE_NOACCESS` during GPU fetch).
    let mut pages_to_restore = Vec::new();
    let mut entries: Vec<DeferredPiece> = Vec::new();

    with_state(|s| {
        let mut p_min = req_first;
        let mut p_max = req_last;
        loop {
            let mut added = false;
            for (&_k, piece) in &s.deferred {
                let pf = piece.abs_lo / KERNEL_PAGE;
                let pl = (piece.abs_hi - 1) / KERNEL_PAGE;
                if pf <= p_max && pl >= p_min {
                    if pf < p_min {
                        p_min = pf;
                        added = true;
                    }
                    if pl > p_max {
                        p_max = pl;
                        added = true;
                    }
                }
            }
            if !added {
                break;
            }
        }

        let keys: Vec<(u64, usize, usize)> = s
            .deferred
            .iter()
            .filter_map(|(&k, piece)| {
                let pf = piece.abs_lo / KERNEL_PAGE;
                let pl = (piece.abs_hi - 1) / KERNEL_PAGE;
                (pf <= p_max && pl >= p_min).then_some(k)
            })
            .collect();

        if keys.is_empty() {
            return;
        }

        let mut freed_pages = HashSet::new();
        for k in keys {
            if let Some(piece) = s.deferred.remove(&k) {
                let pf = piece.abs_lo / KERNEL_PAGE;
                let pl = (piece.abs_hi - 1) / KERNEL_PAGE;
                for p in pf..=pl {
                    if let Some(cnt) = s.deferred_pages.get_mut(&p) {
                        *cnt -= 1;
                        if *cnt == 0 {
                            s.deferred_pages.remove(&p);
                            freed_pages.insert(p);
                        }
                    }
                }
                entries.push(piece);
            }
        }
        pages_to_restore.extend(freed_pages);
        sync_active_counts(s);
    });

    if entries.is_empty() {
        return;
    }

    entries.sort_by_key(|e| e.seq);

    for e in &entries {
        let b = (e.hi - e.lo) as u64;
        match reason {
            AccessReason::CpuFault(_) => stat_cpu_read(&e.shadow, b),
            AccessReason::Preparing => stat_prep_read(&e.shadow, b),
            _ => {}
        }
    }

    // Separate card vs PC-side deferred pieces.
    let (card_pieces, pc_pieces): (Vec<DeferredPiece>, Vec<DeferredPiece>) =
        entries.into_iter().partition(|e| e.card);

    #[cfg(test)]
    if let Some(hook) = TEST_FETCH_HOOK.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        restore_pages_rw(&pages_to_restore);
        let _ = hook(&card_pieces);
        for p in pc_pieces {
            if let Some(before) = p.pc_before {
                let _ = p.shadow.copy_back_with_snapshot(p.lo, p.hi - p.lo, &before);
            }
        }
        return;
    }

    // Fetch card pieces WHILE pages remain PAGE_NOACCESS so concurrent CPU threads
    // hitting the page wait on MATERIALIZE instead of reading stale RAM.
    if !card_pieces.is_empty() {
        let fetched = crate::host::fetch_deferred_card_pieces(&card_pieces);
        restore_pages_rw(&pages_to_restore);
        if let Some((payloads, after)) = fetched {
            let mut title = Vec::new();
            // Ownership as of now: a deferred piece is fetched after its
            // submission ran (only consulted with ray memory on).
            let owners = crate::writeback::Owners::single(u64::MAX);
            for (piece, gpu) in card_pieces.iter().zip(payloads.iter()) {
                let written = piece
                    .shadow
                    .card_merge_relay(piece.lo, gpu, after, &mut title, &owners);
                stat_fetched_merged(&piece.shadow, (piece.hi - piece.lo) as u64, written as u64);
                note_copyback_write(piece.abs_lo, piece.hi - piece.lo);
            }
        }
    } else {
        restore_pages_rw(&pages_to_restore);
    }

    for p in pc_pieces {
        if let Some(before) = p.pc_before {
            let written = p.shadow.copy_back_with_snapshot(p.lo, p.hi - p.lo, &before);
            stat_fetched_merged(&p.shadow, (p.hi - p.lo) as u64, written as u64);
            note_copyback_write(p.abs_lo, p.hi - p.lo);
        }
    }
}

/// Convenience wrapper for `Shadow::update`: materializes any deferred pieces
/// overlapping `[from, to)` EXCEPT `except_shadow_id` if `keep_self` is true.
pub(crate) fn materialize_for_shadow_update(
    except_shadow_id: u64,
    from: usize,
    to: usize,
    keep_self: bool,
) {
    if ACTIVE_DEFERRED.load(Ordering::Relaxed) == 0 || from >= to {
        return;
    }
    let req_first = from / KERNEL_PAGE;
    let req_last = (to - 1) / KERNEL_PAGE;
    let need = with_state(|s| {
        s.deferred.values().any(|p| {
            (!keep_self || p.shadow.id != except_shadow_id)
                && (p.abs_lo / KERNEL_PAGE) <= req_last
                && ((p.abs_hi - 1) / KERNEL_PAGE) >= req_first
        })
    });
    if need {
        materialize_range(from, to - from, AccessReason::Preparing);
    }
}

/// Flushes every currently deferred copy-back piece (used on `VICTORY_DEFER_AT_FENCE=1`
/// when the title waits on a fence, or on teardown).
pub(crate) fn flush_all(reason: AccessReason) {
    if !has_active_guards() {
        return;
    }
    let spans: Vec<(usize, usize)> = with_state(|s| {
        s.deferred
            .values()
            .map(|p| (p.abs_lo, p.abs_hi - p.abs_lo))
            .collect()
    });
    for (addr, len) in spans {
        materialize_range(addr, len, reason);
    }
}

/// Called from `watch.rs` VEH on any `EXCEPTION_ACCESS_VIOLATION` before `guards()`.
/// Returns `true` if `at` was covered by deferred/probed copy-back guards and
/// the fault has now been resolved so execution can continue.
pub(crate) fn on_page_fault(at: usize, is_write: bool) -> bool {
    if !has_active_guards() {
        return false;
    }
    let page = at / KERNEL_PAGE;
    if !is_page_guarded(page) {
        return false;
    }
    materialize_range(page * KERNEL_PAGE, KERNEL_PAGE, AccessReason::CpuFault(is_write));
    true
}

/// Drops any deferred or probed state owned by a shadow being unregistered
/// (without writing back freed memory if the title already released it,
/// Wait: materialize if still mapped? If shadow is being released while title
/// holds memory, materialize first so title memory keeps GPU writes!).
pub(crate) fn on_unregister_shadow(shadow_id: u64) {
    if !has_active_guards() {
        return;
    }
    let spans: Vec<(usize, usize)> = with_state(|s| {
        s.deferred
            .values()
            .filter(|p| p.shadow.id == shadow_id)
            .map(|p| (p.abs_lo, p.abs_hi - p.abs_lo))
            .collect()
    });
    for (addr, len) in spans {
        materialize_range(addr, len, AccessReason::Unregister);
    }
    if ACTIVE_PROBED.load(Ordering::Relaxed) != 0 {
        let mut to_restore = Vec::new();
        with_state(|s| {
            let pages: Vec<usize> = s
                .probed_pages
                .iter()
                .filter_map(|(&p, (sh, _))| (sh.id == shadow_id).then_some(p))
                .collect();
            for p in pages {
                s.probed_pages.remove(&p);
                if !s.deferred_pages.contains_key(&p) {
                    to_restore.push(p);
                }
            }
            sync_active_counts(s);
        });
        restore_pages_rw(&to_restore);
    }
}

#[cfg(test)]
pub(crate) fn test_reset_guards() {
    let mut to_restore = Vec::new();
    with_state(|s| {
        to_restore.extend(s.deferred_pages.keys().copied());
        to_restore.extend(s.probed_pages.keys().copied());
        s.deferred.clear();
        s.deferred_pages.clear();
        s.probed_pages.clear();
        sync_active_counts(s);
    });
    restore_pages_rw(&to_restore);
    set_test_fetch_hook(None);
}

#[cfg(test)]
pub(crate) fn with_test_flags<R>(
    defer: bool,
    defer_pc: bool,
    keep_copied: bool,
    f: impl FnOnce() -> R,
) -> R {
    test_overrides::DEFER.with(|c| c.set(Some(defer)));
    test_overrides::DEFER_PC.with(|c| c.set(Some(defer_pc)));
    test_overrides::KEEP_COPIED.with(|c| c.set(Some(keep_copied)));
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            test_overrides::DEFER.with(|c| c.set(None));
            test_overrides::DEFER_PC.with(|c| c.set(None));
            test_overrides::KEEP_COPIED.with(|c| c.set(None));
            test_reset_guards();
        }
    }
    let _reset = Reset;
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    #[cfg(windows)]
    struct VmRegion {
        base: *mut u8,
    }

    #[cfg(windows)]
    impl VmRegion {
        unsafe fn new(len: usize, init: impl Fn(usize) -> u8) -> Self {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, ResetWriteWatch, VirtualAlloc,
            };
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            // SAFETY: private write-watched region sized `len`.
            let base = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as *mut u8;
            assert!(!base.is_null());
            // SAFETY: `base` covers `len` writable bytes.
            let slice = unsafe { std::slice::from_raw_parts_mut(base, len) };
            for (i, b) in slice.iter_mut().enumerate() {
                *b = init(i);
            }
            // SAFETY: `base..base + len` is our MEM_WRITE_WATCH region.
            assert_eq!(unsafe { ResetWriteWatch(base.cast(), len) }, 0);
            Self { base }
        }

        fn addr(&self) -> usize {
            self.base as usize
        }
    }

    #[cfg(windows)]
    impl Drop for VmRegion {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            test_reset_guards();
            // SAFETY: allocated by VirtualAlloc above.
            unsafe {
                let _ = VirtualFree(self.base.cast(), 0, MEM_RELEASE);
            }
        }
    }

    /// Verifies on Windows how `MEM_WRITE_WATCH` interacts with `VirtualProtect`
    /// transitions (`PAGE_READWRITE` -> `PAGE_NOACCESS` -> `PAGE_READWRITE`) and
    /// `NtWriteVirtualMemory`.
    #[cfg(windows)]
    #[test]
    fn mem_write_watch_preserves_tracking_across_noaccess_cycle() {
        use windows_sys::Win32::System::Memory::{
            GetWriteWatch, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
            ResetWriteWatch, VirtualAlloc, VirtualFree, VirtualProtect,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const WRITE_WATCH_FLAG_RESET: u32 = 1;
        const N: usize = 64 << 10;
        // SAFETY: fresh private allocation with MEM_WRITE_WATCH.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                N,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!base.is_null());

        unsafe {
            assert_eq!(ResetWriteWatch(base.cast(), N), 0);

            // 1. CPU writes page 2 before we arm page 0..1 with PAGE_NOACCESS.
            base.add(2 * KERNEL_PAGE).write_volatile(0xAB);

            let mut old = 0u32;
            assert_ne!(
                VirtualProtect(base.cast(), KERNEL_PAGE, PAGE_NOACCESS, &mut old),
                0
            );
            assert_eq!(old & 0xff, PAGE_READWRITE);

            // Query GetWriteWatch while page 0 is PAGE_NOACCESS: must work and report page 2.
            let mut addrs = [std::ptr::null_mut(); 16];
            let mut count = addrs.len();
            let mut gran = 0u32;
            assert_eq!(
                GetWriteWatch(0, base.cast(), N, addrs.as_mut_ptr(), &mut count, &mut gran),
                0
            );
            assert_eq!(count, 1);
            assert_eq!(addrs[0] as usize, base as usize + 2 * KERNEL_PAGE);

            // Restore page 0 to PAGE_READWRITE without writing to it.
            assert_ne!(
                VirtualProtect(base.cast(), KERNEL_PAGE, PAGE_READWRITE, &mut old),
                0
            );

            // Verify restoring RW did NOT falsely dirty page 0 and kept page 2 dirty.
            count = addrs.len();
            assert_eq!(
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    base.cast(),
                    N,
                    addrs.as_mut_ptr(),
                    &mut count,
                    &mut gran
                ),
                0
            );
            assert_eq!(count, 1);
            assert_eq!(addrs[0] as usize, base as usize + 2 * KERNEL_PAGE);

            // 2. Verify another write after restoring RW IS tracked by MEM_WRITE_WATCH.
            base.add(0).write_volatile(0xCD);
            count = addrs.len();
            assert_eq!(
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    base.cast(),
                    N,
                    addrs.as_mut_ptr(),
                    &mut count,
                    &mut gran
                ),
                0
            );
            assert_eq!(count, 1);
            assert_eq!(addrs[0] as usize, base as usize);

            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn deferred_card_cpu_read_fault_materializes_gpu_bytes() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const N: usize = 128 << 10;
        const PIECE: usize = 64 << 10;
        let title = unsafe { VmRegion::new(N, |i| ((i * 7 + 3) & 0xff) as u8) };
        let mut side = unsafe { std::slice::from_raw_parts(title.base, N).to_vec() };
        let Some(shadow) = Shadow::test_card(title.addr() as u64, &mut side) else {
            return;
        };

        with_test_flags(true, false, false, || {
            shadow.update(title.addr(), N);
            shadow.test_clear_card_dirty();
            let mut gpu = side.clone();
            gpu[256] = 0x5A;
            gpu[4096 + 17] = 0xA5;
            let fetches = Arc::new(AtomicUsize::new(0));
            let fetches_hook = fetches.clone();
            let gpu_hook = Arc::new(Mutex::new(gpu));
            set_test_fetch_hook(Some(Box::new(move |pieces| {
                fetches_hook.fetch_add(1, Ordering::Relaxed);
                let g = gpu_hook.lock().unwrap();
                let mut written = 0;
                let mut scratch = Vec::new();
                for p in pieces {
                    written += p.shadow.card_merge_relay(p.lo, &g[p.lo..p.hi], 0, &mut scratch, &crate::writeback::Owners::single(u64::MAX));
                }
                written
            })));

            assert!(try_defer_card_piece(&shadow, 0, PIECE));
            assert!(is_page_guarded(title.addr() / KERNEL_PAGE));
            assert_eq!(fetches.load(Ordering::Relaxed), 0);

            // Real CPU read into the deferred PAGE_NOACCESS page faults via VEH and
            // observes the GPU bytes.
            let v1 = unsafe { title.base.add(256).read_volatile() };
            let v2 = unsafe { title.base.add(4096 + 17).read_volatile() };
            assert_eq!(v1, 0x5A);
            assert_eq!(v2, 0xA5);
            assert_eq!(fetches.load(Ordering::Relaxed), 1);
            assert!(!is_page_guarded(title.addr() / KERNEL_PAGE));

            // Untouched piece 1 kept original bytes.
            assert_eq!(
                unsafe { title.base.add(PIECE + 9).read_volatile() },
                (((PIECE + 9) * 7 + 3) & 0xff) as u8
            );
        });
    }

    #[cfg(windows)]
    #[test]
    fn deferred_card_cpu_write_preserves_gpu_bytes_and_updates_later() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const N: usize = 64 << 10;
        let title = unsafe { VmRegion::new(N, |i| (i & 0xff) as u8) };
        let mut side = unsafe { std::slice::from_raw_parts(title.base, N).to_vec() };
        let Some(shadow) = Shadow::test_card(title.addr() as u64, &mut side) else {
            return;
        };

        with_test_flags(true, false, true, || {
            shadow.update(title.addr(), N);
            shadow.test_clear_card_dirty();

            let mut gpu = side.clone();
            gpu[16] = 0x91;
            gpu[KERNEL_PAGE + 8] = 0x77;
            gpu[KERNEL_PAGE + 12] = 0x33;
            set_test_fetch_hook(Some(Box::new(move |pieces| {
                let mut written = 0;
                let mut scratch = Vec::new();
                for p in pieces {
                    written += p.shadow.card_merge_relay(p.lo, &gpu[p.lo..p.hi], 0, &mut scratch, &crate::writeback::Owners::single(u64::MAX));
                }
                written
            })));

            assert!(try_defer_card_piece(&shadow, 0, N));

            // Direct CPU write faults VEH: materializes GPU bytes FIRST, then CPU write
            // overwrites byte KERNEL_PAGE + 12 without losing sibling GPU bytes.
            unsafe {
                title.base.add(KERNEL_PAGE + 12).write_volatile(0xEE);
                assert_eq!(title.base.add(16).read_volatile(), 0x91);
                assert_eq!(title.base.add(KERNEL_PAGE + 8).read_volatile(), 0x77);
                assert_eq!(title.base.add(KERNEL_PAGE + 12).read_volatile(), 0xEE);
            }

            // Next Shadow::update must see CPU write 0xEE and mark only page 1 dirty.
            shadow.test_clear_card_dirty();
            shadow.update(title.addr(), N);
            assert_eq!(side[KERNEL_PAGE + 12], 0xEE);
            assert_eq!(shadow.test_card_dirty(), vec![(KERNEL_PAGE, 2 * KERNEL_PAGE)]);
        });
    }

    #[cfg(windows)]
    #[test]
    fn deferred_gpu_overwrite_without_cpu_read_skips_intermediate_fetch() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const N: usize = 64 << 10;
        let title = unsafe { VmRegion::new(N, |i| ((i + 1) & 0xff) as u8) };
        let mut side = unsafe { std::slice::from_raw_parts(title.base, N).to_vec() };
        let Some(shadow) = Shadow::test_card(title.addr() as u64, &mut side) else {
            return;
        };

        with_test_flags(true, false, false, || {
            shadow.update(title.addr(), N);
            shadow.test_clear_card_dirty();

            let fetches = Arc::new(AtomicUsize::new(0));
            let fetches_hook = fetches.clone();
            let v2 = Arc::new(Mutex::new({
                let mut g = side.clone();
                g[1024] = 0x22;
                g
            }));
            let v2_hook = v2.clone();
            set_test_fetch_hook(Some(Box::new(move |pieces| {
                fetches_hook.fetch_add(1, Ordering::Relaxed);
                let g = v2_hook.lock().unwrap();
                let mut written = 0;
                let mut scratch = Vec::new();
                for p in pieces {
                    written += p.shadow.card_merge_relay(p.lo, &g[p.lo..p.hi], 0, &mut scratch, &crate::writeback::Owners::single(u64::MAX));
                }
                written
            })));

            // GPU write V1 deferred.
            assert!(try_defer_card_piece(&shadow, 0, N));
            // Buffer bound again as shader input: clean deferred piece skips copy & upload.
            let copied = shadow.update(title.addr(), N);
            assert_eq!(copied, 0);
            assert!(shadow.test_card_dirty().is_empty());
            assert_eq!(fetches.load(Ordering::Relaxed), 0);

            // GPU overwrite V2 deferred on same piece: replaces seq cleanly, zero fetch.
            v2.lock().unwrap()[1024] = 0x44;
            assert!(try_defer_card_piece(&shadow, 0, N));
            assert_eq!(fetches.load(Ordering::Relaxed), 0);

            // CPU read sees latest GPU version V2 in one fetch.
            let got = unsafe { title.base.add(1024).read_volatile() };
            assert_eq!(got, 0x44);
            assert_eq!(fetches.load(Ordering::Relaxed), 1);
        });
    }

    #[cfg(windows)]
    #[test]
    fn preparing_readers_materialize_deferred_gpu_bytes() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const N: usize = 64 << 10;
        let title = unsafe { VmRegion::new(N, |i| ((i * 3) & 0xff) as u8) };
        let mut side1 = unsafe { std::slice::from_raw_parts(title.base, N).to_vec() };
        let Some(shadow1) = Shadow::test_card(title.addr() as u64, &mut side1) else {
            return;
        };
        let mut side2 = side1.clone();
        let Some(shadow2) = Shadow::test_card(title.addr() as u64, &mut side2) else {
            return;
        };

        with_test_flags(true, false, false, || {
            shadow1.update(title.addr(), N);
            shadow1.test_clear_card_dirty();
            shadow2.update(title.addr(), N);
            shadow2.test_clear_card_dirty();

            let mut gpu = side1.clone();
            gpu[512] = 0xDE;
            let gpu_arc = Arc::new(Mutex::new(gpu));
            let gpu_hook = gpu_arc.clone();
            set_test_fetch_hook(Some(Box::new(move |pieces| {
                let g = gpu_hook.lock().unwrap();
                let mut written = 0;
                let mut scratch = Vec::new();
                for p in pieces {
                    written += p.shadow.card_merge_relay(p.lo, &g[p.lo..p.hi], 0, &mut scratch, &crate::writeback::Owners::single(u64::MAX));
                }
                written
            })));

            // 1. crate::host::read_into (used by descriptors, ValueWait, textures)
            assert!(try_defer_card_piece(&shadow1, 0, N));
            let mut out = [0u8; 4];
            assert!(crate::host::read_into(
                (title.addr() + 512) as *const u8,
                &mut out
            ));
            assert_eq!(out[0], 0xDE);

            // 2. crate::host::copy_mapped (Step::Fill / Step::Copy)
            shadow1.update(title.addr(), N);
            shadow1.test_clear_card_dirty();
            gpu_arc.lock().unwrap()[512] = 0xAD;
            assert!(try_defer_card_piece(&shadow1, 0, N));
            let mut dst = [0u8; 64];
            crate::host::copy_mapped(
                dst.as_mut_ptr(),
                (title.addr() + 512) as *const u8,
                dst.len(),
                false,
            );
            assert_eq!(dst[0], 0xAD);

            // 3. Overlapping different shadow update materializes shadow1 first
            shadow1.update(title.addr(), N);
            shadow1.test_clear_card_dirty();
            gpu_arc.lock().unwrap()[512] = 0xBE;
            assert!(try_defer_card_piece(&shadow1, 0, N));
            shadow2.update(title.addr(), N);
            assert_eq!(side2[512], 0xBE);
        });
    }

    #[cfg(windows)]
    #[test]
    fn keep_copied_suppresses_copyback_bounce_reupload() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const N: usize = 64 << 10;
        let title = unsafe { VmRegion::new(N, |i| ((i * 5) & 0xff) as u8) };
        let mut side = unsafe { std::slice::from_raw_parts(title.base, N).to_vec() };
        let Some(shadow) = Shadow::test_card(title.addr() as u64, &mut side) else {
            return;
        };

        shadow.update(title.addr(), N);
        shadow.test_clear_card_dirty();

        let mut gpu = side.clone();
        gpu[300] ^= 0x55;
        let merged = shadow.card_merge_relay(0, &gpu, 0, &mut Vec::new(), &crate::writeback::Owners::single(u64::MAX));
        assert!(merged > 0);
        // card_merge_relay dirtied the 64 KiB page via NtWriteVirtualMemory,
        // even though `side` already holds the exact bytes written to `title`.
        with_test_flags(false, false, true, || {
            shadow.test_clear_card_dirty();
            let written = shadow.update(title.addr(), N);
            assert_eq!(written, 0);
            assert!(shadow.test_card_dirty().is_empty());

            // Now a real 1-byte CPU write in block 2 ([8192..12288)) uploads only that 4 KiB block.
            unsafe { title.base.add(8192 + 7).write_volatile(0x11) };
            let written = shadow.update(title.addr(), N);
            assert_eq!(written, KERNEL_PAGE);
            assert_eq!(side[8192 + 7], 0x11);
            assert_eq!(shadow.test_card_dirty(), vec![(8192, 12288)]);
        });
    }

    #[cfg(windows)]
    #[test]
    fn pc_shadow_defer_and_cpu_read_with_snapshot() {
        let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const N: usize = 64 << 10;
        let title = unsafe { VmRegion::new(N, |i| ((i * 13) & 0xff) as u8) };
        let mut pc_data = unsafe { std::slice::from_raw_parts(title.base, N).to_vec() };
        let Some(pc) = Shadow::test_pc(title.addr() as u64, &mut pc_data) else {
            return;
        };

        with_test_flags(false, true, false, || {
            pc.update(title.addr(), N);
            let before = pc_data.clone();
            // Simulate GPU write to pc_data after snapshot was taken.
            pc_data[128] = 0x7E;
            assert!(try_defer_pc_piece(&pc, 0, N, &before));
            assert!(is_page_guarded(title.addr() / KERNEL_PAGE));

            let got = unsafe { title.base.add(128).read_volatile() };
            assert_eq!(got, 0x7E);
            assert!(!is_page_guarded(title.addr() / KERNEL_PAGE));
        });
    }
}
