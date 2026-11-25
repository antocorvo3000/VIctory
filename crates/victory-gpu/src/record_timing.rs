//! Per-stage CPU timing for command list recording and submission preparation.
//!
//! Off by default (`VICTORY_RECORD_TIMING=1` enables it in any build, including
//! release builds without `diagnostics`). When disabled, every measurement
//! compiles down to a single relaxed `OnceLock` check with no allocations or
//! clock reads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Whether per-stage recording CPU timing is enabled (`VICTORY_RECORD_TIMING=1`,
/// off by default).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| enabled_from(std::env::var("VICTORY_RECORD_TIMING").ok().as_deref()))
}

pub(crate) fn enabled_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// Whether parallel command-list recording fast paths (shared RwLock reads,
/// lock-free descriptor mirroring, bitmap dynamic allocation) are enabled
/// (`VICTORY_RECORD_PARALLEL=1`, off by default).
pub(crate) fn record_parallel() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        record_parallel_from(std::env::var("VICTORY_RECORD_PARALLEL").ok().as_deref())
    })
}

pub(crate) fn record_parallel_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Stage {
    DrawCall = 0,
    DispatchCall = 1,
    SetTargetsCall = 2,
    StateSetCall = 3,
    DescriptorWriteCall = 4,
    PipelineLookup = 5,
    BindHeader = 6,
    TablesNonFresh = 7,
    TablesHash = 8,
    TablesHit = 9,
    TablesAllocMake = 10,
    UserData = 11,
    RootViews = 12,
    VertexIndex = 13,
    TargetsState = 14,
    IndirectBound = 15,
    LockWait = 16,
    SubmitAskOnce = 17,
    SubmitCopyFill = 18,
    SubmitRefresh = 19,
    SubmitTexture = 20,
    SubmitSnapshots = 21,
    SubmitCardBefore = 22,
    SubmitPrelude = 23,
}

const STAGE_COUNT: usize = 24;

static CALLS: [AtomicU64; STAGE_COUNT] = [const { AtomicU64::new(0) }; STAGE_COUNT];
static NS: [AtomicU64; STAGE_COUNT] = [const { AtomicU64::new(0) }; STAGE_COUNT];

/// Per-thread recording totals `(thread_id, ns, calls)`.
static THREADS: Mutex<Vec<(u32, u64, u64)>> = Mutex::new(Vec::new());

thread_local! {
    static UNSENT: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

pub(crate) struct Scope {
    stage: Stage,
    started: Option<Instant>,
}

impl Scope {
    #[inline]
    pub(crate) fn begin(stage: Stage) -> Self {
        Self {
            stage,
            started: enabled().then(Instant::now),
        }
    }
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if let Some(started) = self.started {
            add(self.stage, started.elapsed().as_nanos() as u64);
        }
    }
}

#[inline]
pub(crate) fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

#[inline]
pub(crate) fn add(stage: Stage, ns: u64) {
    CALLS[stage as usize].fetch_add(1, Ordering::Relaxed);
    NS[stage as usize].fetch_add(ns, Ordering::Relaxed);
}

#[inline]
pub(crate) fn add_since(stage: Stage, started: Option<Instant>) {
    if let Some(started) = started {
        add(stage, started.elapsed().as_nanos() as u64);
    }
}

/// Records one recording entry-point call on the current OS thread.
#[inline]
pub(crate) fn record_thread_call(started: Option<Instant>) {
    let Some(started) = started else {
        return;
    };
    let ns = started.elapsed().as_nanos() as u64;
    let (unsent_ns, unsent_calls) = UNSENT.with(|u| u.get());
    let (unsent_ns, unsent_calls) = (unsent_ns + ns, unsent_calls + 1);
    if unsent_ns < 2_000_000 {
        UNSENT.with(|u| u.set((unsent_ns, unsent_calls)));
        return;
    }
    UNSENT.with(|u| u.set((0, 0)));
    // SAFETY: a plain thread id query.
    let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let mut threads = THREADS.lock().unwrap_or_else(|e| e.into_inner());
    match threads.iter_mut().find(|t| t.0 == tid) {
        Some(t) => {
            t.1 += unsent_ns;
            t.2 += unsent_calls;
        }
        None => threads.push((tid, unsent_ns, unsent_calls)),
    }
}

/// Acquires `mutex`, recording wait time in [`Stage::LockWait`] when `VICTORY_RECORD_TIMING=1`
/// and the lock is contended.
#[inline]
pub(crate) fn lock_timed<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    if !enabled() {
        return mutex.lock().unwrap_or_else(|e| e.into_inner());
    }
    if let Ok(guard) = mutex.try_lock() {
        return guard;
    }
    let started = Instant::now();
    let guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
    add(Stage::LockWait, started.elapsed().as_nanos() as u64);
    guard
}

/// Acquires a read guard on `rwlock`, recording wait time in [`Stage::LockWait`] when
/// `VICTORY_RECORD_TIMING=1` and the lock is contended.
#[inline]
pub(crate) fn read_timed<T>(rwlock: &std::sync::RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    if !enabled() {
        return rwlock.read().unwrap_or_else(|e| e.into_inner());
    }
    if let Ok(guard) = rwlock.try_read() {
        return guard;
    }
    let started = Instant::now();
    let guard = rwlock.read().unwrap_or_else(|e| e.into_inner());
    add(Stage::LockWait, started.elapsed().as_nanos() as u64);
    guard
}

/// Acquires a write guard on `rwlock`, recording wait time in [`Stage::LockWait`] when
/// `VICTORY_RECORD_TIMING=1` and the lock is contended.
#[inline]
pub(crate) fn write_timed<T>(rwlock: &std::sync::RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    if !enabled() {
        return rwlock.write().unwrap_or_else(|e| e.into_inner());
    }
    if let Ok(guard) = rwlock.try_write() {
        return guard;
    }
    let started = Instant::now();
    let guard = rwlock.write().unwrap_or_else(|e| e.into_inner());
    add(Stage::LockWait, started.elapsed().as_nanos() as u64);
    guard
}

fn take_stage(stage: Stage) -> (u64, u64) {
    let i = stage as usize;
    (
        CALLS[i].swap(0, Ordering::Relaxed),
        NS[i].swap(0, Ordering::Relaxed),
    )
}

/// Logs per-frame CPU recording and preparing breakdowns when `VICTORY_RECORD_TIMING=1`.
pub(crate) fn log(frames: f64) {
    if !enabled() {
        return;
    }
    let frames = frames.max(1.0);
    let ms = |ns: u64| ns as f64 / 1e6 / frames;
    let cnt = |c: u64| c as f64 / frames;

    let (draws, draw_ns) = take_stage(Stage::DrawCall);
    let (dispatches, dispatch_ns) = take_stage(Stage::DispatchCall);
    let (targets, targets_ns) = take_stage(Stage::SetTargetsCall);
    let (states, state_ns) = take_stage(Stage::StateSetCall);
    let (descs, desc_ns) = take_stage(Stage::DescriptorWriteCall);
    let (waits, wait_ns) = take_stage(Stage::LockWait);

    let mut threads = std::mem::take(&mut *THREADS.lock().unwrap_or_else(|e| e.into_inner()));
    threads.sort_by_key(|t| std::cmp::Reverse(t.1));
    let top_threads: Vec<String> = threads
        .iter()
        .take(4)
        .map(|&(tid, ns, calls)| format!("tid {tid}: {:.1} ms ({:.0})", ms(ns), cnt(calls)))
        .collect();

    tracing::info!(
        "record CPU/frame: draws {:.0} ({:.2} ms), dispatches {:.0} ({:.2} ms), set_targets {:.0} ({:.2} ms), \
         state {:.0} ({:.2} ms), descriptors {:.0} ({:.2} ms), contended locks {:.0} ({:.2} ms); threads={}: [{}]",
        cnt(draws),
        ms(draw_ns),
        cnt(dispatches),
        ms(dispatch_ns),
        cnt(targets),
        ms(targets_ns),
        cnt(states),
        ms(state_ns),
        cnt(descs),
        ms(desc_ns),
        cnt(waits),
        ms(wait_ns),
        threads.len(),
        top_threads.join(", "),
    );

    let (_, pipe_ns) = take_stage(Stage::PipelineLookup);
    let (_, bind_hdr_ns) = take_stage(Stage::BindHeader);
    let (nf_cnt, nf_ns) = take_stage(Stage::TablesNonFresh);
    let (hash_cnt, hash_ns) = take_stage(Stage::TablesHash);
    let (hit_cnt, hit_ns) = take_stage(Stage::TablesHit);
    let (make_cnt, make_ns) = take_stage(Stage::TablesAllocMake);
    let (_, ud_ns) = take_stage(Stage::UserData);
    let (_, rv_ns) = take_stage(Stage::RootViews);
    let (_, vi_ns) = take_stage(Stage::VertexIndex);
    let (_, ts_ns) = take_stage(Stage::TargetsState);
    let (ind_cnt, ind_ns) = take_stage(Stage::IndirectBound);

    tracing::info!(
        "record stages CPU/frame: pipeline {:.2} ms, bind_hdr {:.2} ms, \
         tables[nonfresh {:.0}: {:.2} ms, hash {:.0}: {:.2} ms, hit {:.0}: {:.2} ms, alloc+make {:.0}: {:.2} ms], \
         user_data {:.2} ms, root_views {:.2} ms, vb/ib {:.2} ms, targets_state {:.2} ms, indirect {:.0}: {:.2} ms",
        ms(pipe_ns),
        ms(bind_hdr_ns),
        cnt(nf_cnt),
        ms(nf_ns),
        cnt(hash_cnt),
        ms(hash_ns),
        cnt(hit_cnt),
        ms(hit_ns),
        cnt(make_cnt),
        ms(make_ns),
        ms(ud_ns),
        ms(rv_ns),
        ms(vi_ns),
        ms(ts_ns),
        cnt(ind_cnt),
        ms(ind_ns),
    );

    let (_, ask_ns) = take_stage(Stage::SubmitAskOnce);
    let (_, copy_ns) = take_stage(Stage::SubmitCopyFill);
    let (_, ref_ns) = take_stage(Stage::SubmitRefresh);
    let (_, tex_ns) = take_stage(Stage::SubmitTexture);
    let (_, snap_ns) = take_stage(Stage::SubmitSnapshots);
    let (_, card_ns) = take_stage(Stage::SubmitCardBefore);
    let (_, prel_ns) = take_stage(Stage::SubmitPrelude);

    tracing::info!(
        "submit preparing CPU/frame: ask_once {:.2} ms, copy/fill {:.2} ms, refresh {:.2} ms, \
         textures {:.2} ms, snapshots {:.2} ms, card_before {:.2} ms, prelude {:.2} ms",
        ms(ask_ns),
        ms(copy_ns),
        ms(ref_ns),
        ms(tex_ns),
        ms(snap_ns),
        ms(card_ns),
        ms(prel_ns),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_timing_switch_defaults_off() {
        assert!(!enabled_from(None));
        assert!(!enabled_from(Some("")));
        assert!(!enabled_from(Some("0")));
        assert!(enabled_from(Some("1")));
        assert!(enabled_from(Some(" 1 ")));
    }

    #[test]
    fn record_parallel_switch_defaults_off() {
        assert!(!record_parallel_from(None));
        assert!(!record_parallel_from(Some("")));
        assert!(!record_parallel_from(Some("0")));
        assert!(record_parallel_from(Some("1")));
        assert!(record_parallel_from(Some(" 1 ")));
    }

    #[test]
    fn stage_counters_accumulate_and_drain() {
        let _ = take_stage(Stage::PipelineLookup);
        add(Stage::PipelineLookup, 1_500_000);
        add(Stage::PipelineLookup, 500_000);
        let (calls, ns) = take_stage(Stage::PipelineLookup);
        assert!(calls >= 2);
        assert!(ns >= 2_000_000);
        let (calls2, ns2) = take_stage(Stage::PipelineLookup);
        assert_eq!((calls2, ns2), (0, 0));
    }
}
