//! Deadlock-free synchronized start for the parallel search harnesses.
//!
//! # Why this exists
//!
//! Every engine's `search()` warms its workers (connect + one discarded "prime"
//! query) *outside* the measured window and then starts them all at the same
//! instant. That used to be spelled as two fixed-count barriers:
//!
//! ```ignore
//! let ready = Arc::new(Barrier::new(parallel + 1));
//! let go = Arc::new(Barrier::new(parallel + 1));
//! std::thread::scope(|s| {
//!     for _ in 0..parallel { s.spawn(move || { /* warm */ ready.wait(); go.wait(); /* run */ }); }
//!     ready.wait();                      // <-- needs exactly `parallel + 1` arrivals
//!     start_cell.set(Instant::now()).ok();
//!     go.wait();
//! });
//! ```
//!
//! A `Barrier` is sized *before* the workers exist, so the count is a promise
//! the harness cannot keep. Two ordinary failures break it, and both produce a
//! **permanent hang with no output** rather than an error (issue #214):
//!
//! 1. **The OS refuses a thread.** `Scope::spawn` panics on `EAGAIN`
//!    (`ulimit -u`, cgroup `pids.max` — i.e. any CI container with a large
//!    `parallel`). The panic unwinds into `thread::scope`'s drop, which joins
//!    the `k` workers already spawned; they are parked in `ready.wait()`
//!    waiting for `parallel + 1` arrivals that can never happen.
//! 2. **A worker panics before the barrier.** It never arrives either, so the
//!    coordinator parks in `ready.wait()` forever.
//!
//! `--search-timeout` defaults to `0.0` (disabled), so nothing breaks the hang.
//!
//! # What replaces it
//!
//! [`WorkerPool`] pairs a *count-agnostic* gate ([`StartGate`]: `Mutex` +
//! `Condvar`) with a spawn that reports failure instead of panicking
//! (`thread::Builder::spawn_scoped`, which returns [`io::Result`]).
//!
//! Every worker is issued a [`WorkerTicket`] **before** it is spawned — by
//! [`WorkerPool::spawn`] itself, which hands the ticket to the worker closure
//! so it cannot be omitted or mismatched — and the gate only ever waits for
//! *tickets to settle*, never for a number fixed in advance. A ticket settles
//! in exactly one of three ways:
//!
//! | outcome | how | effect |
//! |---|---|---|
//! | ready | [`WorkerTicket::arrive_and_wait`] | worker parks until the start instant is stamped |
//! | failed setup | [`WorkerTicket::fail`] | reason recorded, run aborts |
//! | lost | `Drop` without either of the above | counted as lost, run aborts |
//!
//! The `Drop` arm is what makes a panic (or a never-started thread, whose
//! closure — and therefore ticket — is dropped by the failed `spawn_scoped`)
//! settle its ticket during unwind. The coordinator's wait is satisfied by
//! *any* terminal outcome, so no arrival count is ever left unmet: every way a
//! worker can *finish* — normally, by failing setup, or by panicking — settles
//! its ticket.
//!
//! **What this does not cover.**
//!
//! * A worker that never *finishes* still blocks the coordinator:
//!   [`StartGate::wait_ready`] has no deadline, so a setup step that hangs
//!   forever (a `connect()` against a blackholed endpoint with no timeout, say)
//!   hangs the run exactly as the barrier did. That is not a regression, and
//!   `--search-timeout` is the intended backstop, but it defaults to `0.0`.
//!   A gate-level deadline is tracked separately.
//! * `WorkerPool::spawn` minting the ticket defeats *accidental* omission — you
//!   cannot forget to create one, mint one too few, or pair the wrong ticket
//!   with the wrong worker, and the compiler enforces all three. It is not a
//!   defence against a determined caller: `std::mem::forget(ticket)`, or
//!   sending the ticket out of the closure over a channel, leaves it unsettled
//!   and hangs `wait_ready` forever. Both require going out of your way, and
//!   neither is reachable from the shape every harness uses; treat the ticket
//!   as something the worker must consume before it returns.
//!
//! # Failure semantics: hard error, never "carry on with fewer"
//!
//! A worker that never started, or died before the start gate, means the run
//! executed at a lower concurrency than the `parallel` it reports. That changes
//! the reported number, so per the repo's settled policy it is a hard error —
//! [`WorkerPool::start`] returns `Err` naming what happened. It does not
//! silently proceed at `parallel - k`, and it does not hang.

#[cfg(test)]
use std::io;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{Scope, ScopedJoinHandle};
use std::time::Instant;

/// Lifecycle of the shared start signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Workers are still warming up; nobody may start.
    Waiting,
    /// Every worker warmed successfully; the measured window began at this instant.
    Released(Instant),
    /// The run is being torn down. Parked workers must leave immediately.
    Aborted,
}

#[derive(Debug)]
struct GateState {
    /// Tickets that reached a terminal startup outcome (ready + failed + lost).
    settled: usize,
    /// Tickets parked at the gate, warmed and ready to run.
    ready: usize,
    /// Tickets dropped without arriving or failing — a panic, or a thread the
    /// OS never started.
    lost: usize,
    /// Setup errors reported by workers that did start (connect/client build).
    failures: Vec<String>,
    phase: Phase,
}

/// A start signal whose wait is satisfied by *outcomes*, not by a count fixed
/// before the workers exist. See the module docs.
#[derive(Debug)]
pub struct StartGate {
    state: Mutex<GateState>,
    cv: Condvar,
}

impl Default for StartGate {
    fn default() -> Self {
        Self::new()
    }
}

impl StartGate {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(GateState {
                settled: 0,
                ready: 0,
                lost: 0,
                failures: Vec::new(),
                phase: Phase::Waiting,
            }),
            cv: Condvar::new(),
        }
    }

    /// Lock the state, ignoring poisoning.
    ///
    /// A poisoned mutex here means some thread panicked while holding it. The
    /// state is a handful of counters that are always left consistent, and
    /// refusing the lock would reintroduce exactly the hang this module exists
    /// to remove — so recover the guard and keep making progress. The panic
    /// itself is still reported: the panicking worker's ticket lands in `lost`
    /// (or its join returns `Err`), and both are hard errors.
    fn lock(&self) -> MutexGuard<'_, GateState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Issue a ticket for one worker. Call this **before** spawning, so the
    /// ticket exists even if the spawn itself fails.
    ///
    /// Every worker counted by [`Self::wait_ready`] must own exactly one
    /// ticket: a spawned worker without one never settles and hangs the
    /// coordinator, and a ticket without a worker does the same. Thread-based
    /// harnesses get that pairing for free from [`WorkerPool::spawn`]; a
    /// harness calling this directly owns the invariant itself, and must also
    /// hold an [`AbortGateOnDrop`] across the fan-out.
    pub fn ticket(self: &Arc<Self>) -> WorkerTicket {
        WorkerTicket {
            gate: Arc::clone(self),
            settled: false,
        }
    }

    /// Block until all `expected` tickets have settled.
    ///
    /// `Ok(())` means every worker is warmed and parked at the gate. `Err`
    /// means at least one worker was lost or failed setup; the gate is put into
    /// [`Phase::Aborted`] first, so any parked peers wake and return promptly.
    ///
    /// Public because not every harness fans out over scoped threads: the
    /// Weaviate gRPC path drives `tokio` tasks and coordinates the same gate by
    /// hand. Thread-based harnesses should use [`WorkerPool`] instead.
    pub fn wait_ready(&self, label: &str, expected: usize) -> Result<(), String> {
        let mut st = self.lock();
        while st.settled < expected && st.phase != Phase::Aborted {
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }

        if st.lost == 0 && st.failures.is_empty() && st.phase != Phase::Aborted {
            return Ok(());
        }

        st.phase = Phase::Aborted;
        let mut parts = Vec::new();
        if st.lost > 0 {
            parts.push(format!(
                "{} never reached the start gate (thread not started, or panicked during setup)",
                st.lost
            ));
        }
        if !st.failures.is_empty() {
            parts.push(format!("{} failed setup: {}", st.failures.len(), {
                let mut reasons = st.failures.clone();
                reasons.sort();
                reasons.dedup();
                reasons.join("; ")
            }));
        }
        let ready = st.ready;
        drop(st);
        self.cv.notify_all();

        Err(format!(
            "only {ready} of {expected} {label} workers reached the start gate — {}. \
             Refusing to report a run at parallel={expected} that measured fewer workers",
            parts.join(", ")
        ))
    }

    /// Stamp the measured-window start and release every parked worker.
    pub fn release(&self, at: Instant) {
        let mut st = self.lock();
        if st.phase == Phase::Waiting {
            st.phase = Phase::Released(at);
        }
        drop(st);
        self.cv.notify_all();
    }

    /// Tear the gate down: parked workers wake and return `None`.
    ///
    /// Idempotent, and a no-op once the gate has been released — aborting a
    /// run that already started would be a lie.
    pub fn abort(&self) {
        let mut st = self.lock();
        if st.phase == Phase::Waiting {
            st.phase = Phase::Aborted;
        }
        drop(st);
        self.cv.notify_all();
    }

    fn settle(&self, outcome: Outcome) {
        let mut st = self.lock();
        st.settled += 1;
        match outcome {
            Outcome::Ready => st.ready += 1,
            Outcome::Lost => st.lost += 1,
            Outcome::Failed(reason) => st.failures.push(reason),
        }
        drop(st);
        self.cv.notify_all();
    }
}

enum Outcome {
    Ready,
    Lost,
    Failed(String),
}

/// Aborts a [`StartGate`] if it is still un-released when this guard drops.
///
/// [`WorkerPool`] does this in its own `Drop`. A harness that drives a bare
/// `StartGate` — currently only the Weaviate gRPC path, which fans out `tokio`
/// tasks rather than scoped threads — **must** hold one of these across the
/// whole fan-out, declared *after* whatever owns the workers (the tokio
/// `Runtime`, say) so that drop order aborts the gate before the workers are
/// joined. Without it, a coordinator panic between the first `ticket()` and
/// `wait_ready` leaves the parked workers on a condvar nobody will ever notify.
///
/// Aborting after a successful release is a no-op, so the guard needs no
/// disarm and can simply fall out of scope on the happy path.
pub struct AbortGateOnDrop(Arc<StartGate>);

impl AbortGateOnDrop {
    pub fn new(gate: &Arc<StartGate>) -> Self {
        Self(Arc::clone(gate))
    }
}

impl Drop for AbortGateOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// One worker's claim on the start gate.
///
/// Created before the thread is spawned and moved into it. However the worker
/// ends — arriving, reporting a setup failure, panicking, or never running at
/// all — the ticket settles, so the coordinator's wait always completes.
#[derive(Debug)]
pub struct WorkerTicket {
    gate: Arc<StartGate>,
    settled: bool,
}

impl WorkerTicket {
    /// Report "warmed and ready", then park until the measured window opens.
    ///
    /// Returns the shared start instant, or `None` if the run was aborted (a
    /// peer was lost or failed setup) — in which case the worker must return
    /// immediately without measuring anything.
    #[must_use = "None means the run was aborted; the worker must return without measuring"]
    pub fn arrive_and_wait(mut self) -> Option<Instant> {
        self.settled = true;
        let gate = Arc::clone(&self.gate);
        gate.settle(Outcome::Ready);

        let mut st = gate.lock();
        loop {
            match st.phase {
                Phase::Released(at) => return Some(at),
                Phase::Aborted => return None,
                Phase::Waiting => st = gate.cv.wait(st).unwrap_or_else(|e| e.into_inner()),
            }
        }
    }

    /// Report that this worker could not set itself up (connect, client build,
    /// runtime build). The run is aborted: a search reported at `parallel=N`
    /// must not have been executed by fewer than `N` workers.
    pub fn fail(mut self, reason: impl Into<String>) {
        self.settled = true;
        self.gate.settle(Outcome::Failed(reason.into()));
    }
}

impl Drop for WorkerTicket {
    fn drop(&mut self) {
        if !self.settled {
            // Reached by a panic unwinding out of the worker before it arrived,
            // and by `spawn_scoped` dropping the closure of a thread the OS
            // refused to create. Either way the ticket settles, so the
            // coordinator is never left waiting on an arrival that cannot come.
            self.gate.settle(Outcome::Lost);
        }
    }
}

// Test-only seam: number of further spawns this thread is allowed before
// `WorkerPool::spawn` starts reporting `EAGAIN`-shaped failures.
//
// Thread-local, so tests running in parallel in one binary cannot disturb each
// other. `#[cfg(test)]`, so the shipped binary has neither the counter nor the
// check — this is a test seam, not a runtime backdoor.
#[cfg(test)]
thread_local! {
    static SPAWN_BUDGET: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Allow `n` more spawns on this thread, then fail every subsequent one.
#[cfg(test)]
fn set_spawn_budget(n: usize) {
    SPAWN_BUDGET.with(|b| b.set(Some(n)));
}

#[cfg(test)]
fn clear_spawn_budget() {
    SPAWN_BUDGET.with(|b| b.set(None));
}

/// Consume one unit of spawn budget; `true` means "pretend the OS refused".
#[cfg(test)]
fn injected_spawn_failure() -> bool {
    SPAWN_BUDGET.with(|b| match b.get() {
        None => false,
        Some(0) => true,
        Some(n) => {
            b.set(Some(n - 1));
            false
        }
    })
}

/// A set of scoped worker threads with a synchronized, deadlock-free start.
///
/// Usage inside `std::thread::scope`:
///
/// ```ignore
/// let (results, measured_start) = std::thread::scope(|s| {
///     let mut pool = WorkerPool::new(s, "redis-search", parallel);
///     for _ in 0..parallel {
///         pool.spawn(move |ticket| {
///             // ... connect + prime ...
///             let Some(start) = ticket.arrive_and_wait() else { return Default::default() };
///             // ... measured loop ...
///         })?;
///     }
///     pool.start()
/// })?;
/// ```
///
/// The pool mints each ticket and hands it to the worker closure, so a worker
/// cannot be spawned without one and a ticket cannot be minted without a
/// worker. That pairing is the whole deadlock-freedom argument; making it an
/// API property rather than a convention is why there is no public `ticket()`
/// here — see [`StartGate::ticket`] for the by-hand variant.
pub struct WorkerPool<'scope, 'env: 'scope, T: Send + 'scope> {
    scope: &'scope Scope<'scope, 'env>,
    gate: Arc<StartGate>,
    handles: Vec<ScopedJoinHandle<'scope, T>>,
    label: &'static str,
    /// How many workers the caller intends to run — the `parallel` the result
    /// will be labelled with. Used for honest diagnostics, and checked against
    /// the number actually spawned in [`Self::start_with`].
    planned: usize,
}

impl<'scope, 'env: 'scope, T: Send + 'scope> WorkerPool<'scope, 'env, T> {
    pub fn new(scope: &'scope Scope<'scope, 'env>, label: &'static str, planned: usize) -> Self {
        Self {
            scope,
            gate: Arc::new(StartGate::new()),
            handles: Vec::new(),
            label,
            planned,
        }
    }

    /// Spawn one worker, handing it its own start-gate ticket.
    ///
    /// Unlike `Scope::spawn`, an OS refusal (`EAGAIN` under `ulimit -u` or
    /// cgroup `pids.max`) is returned as `Err`, not a panic. The gate is
    /// aborted first, so the workers already parked at it wake up and return
    /// instead of being stranded when `thread::scope` joins them.
    pub fn spawn<F>(&mut self, f: F) -> Result<(), String>
    where
        F: FnOnce(WorkerTicket) -> T + Send + 'scope,
    {
        let index = self.handles.len();
        let name = format!("{}-{index}", self.label);
        // Minted here, not by the caller: a spawned worker always has exactly
        // one ticket, and a minted ticket always has exactly one worker. If the
        // spawn below fails, the closure — and with it this ticket — is dropped,
        // which settles it as lost.
        let ticket = self.gate.ticket();
        let f = move || f(ticket);

        #[cfg(test)]
        let spawned = if injected_spawn_failure() {
            drop(f); // as `spawn_scoped` does on failure — settles the ticket
            Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "injected spawn failure",
            ))
        } else {
            std::thread::Builder::new()
                .name(name)
                .spawn_scoped(self.scope, f)
        };
        #[cfg(not(test))]
        let spawned = std::thread::Builder::new()
            .name(name)
            .spawn_scoped(self.scope, f);

        match spawned {
            Ok(handle) => {
                self.handles.push(handle);
                Ok(())
            }
            Err(e) => {
                self.gate.abort();
                Err(format!(
                    "could not start {} worker {} of {}: {e}. \
                     The OS refused the thread — lower `parallel`, or raise the thread/process \
                     limit (ulimit -u, cgroup pids.max)",
                    self.label,
                    index + 1,
                    self.planned
                ))
            }
        }
    }

    /// Wait for every worker to warm up, stamp and release the shared start
    /// instant, then join them all.
    ///
    /// Returns each worker's value in spawn order plus the instant the measured
    /// window opened. `Err` if any worker never reached the gate, failed setup,
    /// or panicked mid-run — all of which would make a result labelled
    /// `parallel=N` untrue.
    pub fn start(&mut self) -> Result<(Vec<T>, Instant), String> {
        self.start_with(Instant::now)
    }

    /// [`Self::start`] with a caller-chosen start instant.
    ///
    /// `stamp` runs once, after every worker has warmed, and its result becomes
    /// the shared measurement start. The Vertex open-loop harness uses it to
    /// schedule the window a fixed lead-time in the future.
    pub fn start_with(
        &mut self,
        stamp: impl FnOnce() -> Instant,
    ) -> Result<(Vec<T>, Instant), String> {
        let expected = self.handles.len();
        if expected != self.planned {
            // Belt and braces on the caller's loop bound: reporting `parallel=N`
            // for a run that only ever spawned N-1 workers is the same lie as
            // losing one to the OS.
            self.gate.abort();
            self.drain();
            return Err(format!(
                "{} spawned {expected} workers but the run is labelled parallel={}",
                self.label, self.planned
            ));
        }
        if let Err(e) = self.gate.wait_ready(self.label, expected) {
            // `wait_ready` has already aborted the gate, so the parked workers
            // are on their way out. Join them here rather than leaving it to
            // `thread::scope`, which re-panics on an unjoined panicking thread
            // and would turn our diagnostic into a bare "a scoped thread
            // panicked".
            self.drain();
            return Err(e);
        }

        let at = stamp();
        self.gate.release(at);

        // Join every worker even after one panics, so no thread outlives the
        // scope and the failure is reported once, with a count.
        let mut values = Vec::with_capacity(expected);
        let mut panicked = 0usize;
        for handle in std::mem::take(&mut self.handles) {
            match handle.join() {
                Ok(v) => values.push(v),
                Err(_) => panicked += 1,
            }
        }
        if panicked > 0 {
            return Err(format!(
                "{panicked} of {expected} {} workers panicked mid-run; \
                 discarding the run rather than reporting partial results",
                self.label
            ));
        }
        Ok((values, at))
    }

    /// Join and discard every outstanding worker. Panic payloads are swallowed
    /// deliberately: the caller is already returning a more specific error.
    fn drain(&mut self) {
        for handle in std::mem::take(&mut self.handles) {
            let _ = handle.join();
        }
    }
}

impl<T: Send> Drop for WorkerPool<'_, '_, T> {
    fn drop(&mut self) {
        // Any early exit from the enclosing `thread::scope` closure — a `?` on
        // `spawn`, on `start`, or on anything else in between — drops the pool
        // before `scope` joins the workers. Abort so parked workers leave the
        // gate; without this the `?` would hand control straight to the same
        // deadlock this module removes. No-op once the gate is released.
        self.gate.abort();
        self.drain();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::time::Duration;

    /// How long a call must fail within to count as "prompt" rather than hung.
    const PROMPT: Duration = Duration::from_secs(5);
    /// How long we watch a known-deadlocked shape before declaring it hung.
    const HANG_PROOF: Duration = Duration::from_secs(2);

    // ---------------------------------------------------------------------
    // Part 1 — the pre-fix shape really does hang.
    //
    // These replicate the exact primitive that was in every engine
    // (`Barrier::new(parallel + 1)`) under the two failure modes of #214, and
    // assert the coordinator NEVER returns. They pin down what the rest of this
    // module is buying: without them, "the new code returns Err" would not be
    // evidence that the old code hung.
    //
    // The deadlocked coordinator thread is deliberately never joined — it
    // cannot be, that is the bug. It parks on a condvar and is reclaimed at
    // process exit.
    // ---------------------------------------------------------------------

    /// Run `body` on a detached thread; `true` if it finished within `limit`.
    fn finishes_within(limit: Duration, body: impl FnOnce() + Send + 'static) -> bool {
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&done);
        std::thread::spawn(move || {
            body();
            let (lock, cv) = &*signal;
            *lock.lock().unwrap() = true;
            cv.notify_all();
        });

        let (lock, cv) = &*done;
        let (guard, _) = cv
            .wait_timeout_while(lock.lock().unwrap(), limit, |finished| !*finished)
            .unwrap();
        *guard
    }

    #[test]
    fn legacy_fixed_count_barrier_hangs_when_a_spawn_fails() {
        // Pre-fix shape: the barrier is sized for `parallel + 1` arrivals, but
        // the OS refused worker 3, so only 2 workers + the coordinator ever
        // reach it. `thread::scope` then joins two threads parked forever.
        let hung = !finishes_within(HANG_PROOF, || {
            let parallel = 4;
            let spawned = 2;
            let ready = Arc::new(Barrier::new(parallel + 1));
            std::thread::scope(|s| {
                for _ in 0..spawned {
                    let ready = Arc::clone(&ready);
                    s.spawn(move || {
                        ready.wait();
                    });
                }
                ready.wait(); // never returns: 3 arrivals, needs 5
            });
        });
        assert!(
            hung,
            "the pre-fix Barrier shape was expected to deadlock but completed — \
             this test no longer proves anything"
        );
    }

    #[test]
    fn legacy_fixed_count_barrier_hangs_when_a_worker_panics() {
        // Pre-fix shape: all workers spawn, but one panics before reaching the
        // barrier, so the coordinator waits for an arrival that never comes.
        let hung = !finishes_within(HANG_PROOF, || {
            let parallel = 3;
            let ready = Arc::new(Barrier::new(parallel + 1));
            std::thread::scope(|s| {
                for i in 0..parallel {
                    let ready = Arc::clone(&ready);
                    s.spawn(move || {
                        if i == 1 {
                            // Silence the default hook for this one expected panic.
                            std::panic::panic_any(());
                        }
                        ready.wait();
                    });
                }
                ready.wait(); // never returns: 3 arrivals, needs 4
            });
        });
        assert!(
            hung,
            "the pre-fix Barrier shape was expected to deadlock but completed — \
             this test no longer proves anything"
        );
    }

    // ---------------------------------------------------------------------
    // Part 2 — the same two failure modes through `WorkerPool`, which must fail
    // promptly and informatively instead.
    // ---------------------------------------------------------------------

    /// Drive the pool exactly the way an engine's `search()` does.
    fn run_pool(
        parallel: usize,
        worker: impl Fn(usize, WorkerTicket) -> usize + Sync,
    ) -> Result<(Vec<usize>, Instant), String> {
        std::thread::scope(|s| {
            let mut pool = WorkerPool::new(s, "test", parallel);
            for i in 0..parallel {
                let worker = &worker;
                pool.spawn(move |ticket| worker(i, ticket))?;
            }
            pool.start()
        })
    }

    #[test]
    fn a_pool_that_spawns_fewer_workers_than_it_plans_is_an_error() {
        // The ticket/worker pairing is an API property — `spawn` mints the
        // ticket, so a worker cannot exist without one or vice versa. What the
        // API cannot see is the caller's loop bound: spawning 3 workers for a
        // point that will be published as `parallel=4` is the same lie as losing
        // one to the OS, and used to be invisible.
        let finished = finishes_within(PROMPT, || {
            let err = std::thread::scope(|s| {
                let mut pool: WorkerPool<'_, '_, ()> = WorkerPool::new(s, "test", 4);
                for _ in 0..3 {
                    pool.spawn(|ticket| {
                        let _ = ticket.arrive_and_wait();
                    })?;
                }
                pool.start()
            })
            .expect_err("spawning fewer workers than planned must be an error");
            assert!(
                err.contains("spawned 3 workers but the run is labelled parallel=4"),
                "{err}"
            );
        });
        assert!(finished, "a short-spawned pool hung instead of erroring");
    }

    #[test]
    fn a_worker_that_settles_nothing_is_lost_not_hung() {
        // The residual hazard the old `pool.ticket()` API allowed was a spawned
        // worker with no ticket, which never settles. The closure-minted ticket
        // makes that unrepresentable; the nearest reachable shape is a worker
        // that simply returns, dropping its ticket. That must settle as lost.
        let finished = finishes_within(PROMPT, || {
            let err = run_pool(3, |i, ticket| {
                if i == 1 {
                    drop(ticket);
                    return 0;
                }
                let _ = ticket.arrive_and_wait();
                0
            })
            .expect_err("a worker that never reaches the gate must be an error");
            assert!(
                err.contains("only 2 of 3 test workers reached the start gate"),
                "{err}"
            );
        });
        assert!(finished, "a worker that dropped its ticket hung the pool");
    }

    #[test]
    fn spawn_failure_is_a_prompt_error_not_a_hang() {
        // The first two workers spawn and park at the gate; the third is
        // refused. Pre-fix this deadlocked (see `legacy_..._when_a_spawn_fails`).
        let started = Instant::now();
        let finished = finishes_within(PROMPT, || {
            set_spawn_budget(2);
            let err = run_pool(4, |_, ticket| {
                let _ = ticket.arrive_and_wait();
                0
            })
            .expect_err("a refused spawn must be an error");
            clear_spawn_budget();
            assert!(
                err.contains("could not start test worker 3"),
                "error should name the worker that could not start: {err}"
            );
            assert!(
                err.contains("ulimit -u"),
                "error should tell the operator how to fix it: {err}"
            );
        });
        assert!(
            finished,
            "spawn failure hung for {:?} instead of erroring",
            started.elapsed()
        );
    }

    #[test]
    fn worker_panic_before_the_gate_is_a_prompt_error_not_a_hang() {
        // All four workers spawn; worker 1 panics during "setup", before
        // arriving. Pre-fix the coordinator parked forever (see
        // `legacy_..._when_a_worker_panics`).
        let finished = finishes_within(PROMPT, || {
            let err = run_pool(4, |i, ticket| {
                if i == 1 {
                    std::panic::panic_any(());
                }
                let _ = ticket.arrive_and_wait();
                0
            })
            .expect_err("a worker that panics before the gate must be an error");
            assert!(
                err.contains("only 3 of 4 test workers reached the start gate"),
                "error should quantify the shortfall: {err}"
            );
            assert!(
                err.contains("never reached the start gate"),
                "error should say why: {err}"
            );
        });
        assert!(finished, "a pre-gate worker panic hung instead of erroring");
    }

    #[test]
    fn worker_panic_after_the_gate_is_reported_not_swallowed() {
        // Panicking mid-run cannot deadlock the gate, but it still means the
        // run is short of the concurrency it claims — so it must not be
        // silently folded into the results.
        let finished = finishes_within(PROMPT, || {
            let err = run_pool(3, |i, ticket| {
                let _ = ticket.arrive_and_wait();
                if i == 2 {
                    std::panic::panic_any(());
                }
                0
            })
            .expect_err("a worker that panics mid-run must be an error");
            assert!(
                err.contains("1 of 3 test workers panicked mid-run"),
                "error should count the panicking workers: {err}"
            );
        });
        assert!(finished, "a mid-run worker panic hung instead of erroring");
    }

    #[test]
    fn worker_setup_failure_is_a_hard_error_not_a_quieter_run() {
        // The old code had failing workers cross both barriers and return
        // empty, so the run executed at `parallel - k` while still being
        // labelled `parallel`. That is a changed reported number → hard error.
        let finished = finishes_within(PROMPT, || {
            let err = run_pool(3, |i, ticket| {
                if i == 0 {
                    ticket.fail("connection refused");
                    return 0;
                }
                let _ = ticket.arrive_and_wait();
                0
            })
            .expect_err("a worker that cannot connect must be an error");
            assert!(
                err.contains("connection refused"),
                "error should carry the underlying reason: {err}"
            );
            assert!(
                err.contains("1 failed setup"),
                "error should count the failures: {err}"
            );
        });
        assert!(finished, "a setup failure hung instead of erroring");
    }

    // ---------------------------------------------------------------------
    // Part 3 — the happy path still does what the barriers did.
    // ---------------------------------------------------------------------

    #[test]
    fn all_workers_observe_one_shared_start_after_every_worker_warmed() {
        let warmed = Arc::new(AtomicUsize::new(0));
        let warmed_at_start = Arc::new(Mutex::new(Vec::new()));

        let (values, start) = {
            let warmed = Arc::clone(&warmed);
            let warmed_at_start = Arc::clone(&warmed_at_start);
            run_pool(6, move |i, ticket| {
                // Stagger warm-up so a barrier-less implementation that let the
                // first worker run immediately would be caught.
                std::thread::sleep(Duration::from_millis(5 * i as u64));
                warmed.fetch_add(1, Ordering::SeqCst);
                let observed = ticket.arrive_and_wait().expect("run must not abort");
                warmed_at_start
                    .lock()
                    .unwrap()
                    .push((warmed.load(Ordering::SeqCst), observed));
                i
            })
            .expect("happy path must succeed")
        };

        assert_eq!(
            values,
            vec![0, 1, 2, 3, 4, 5],
            "values come back in spawn order"
        );
        let seen = warmed_at_start.lock().unwrap();
        assert_eq!(seen.len(), 6);
        for (warmed_count, observed_start) in seen.iter() {
            assert_eq!(
                *warmed_count, 6,
                "no worker may start before every worker has warmed up"
            );
            assert_eq!(
                *observed_start, start,
                "every worker must measure from the same start instant"
            );
        }
    }

    #[test]
    fn start_with_hands_every_worker_the_caller_chosen_instant() {
        // The Vertex open-loop harness schedules the window 100ms ahead of the
        // warm-up so no request predates the timer; the gate must hand workers
        // that instant, not `Instant::now()`.
        let lead = Duration::from_millis(120);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (_, start) = {
            let seen = Arc::clone(&seen);
            std::thread::scope(|s| {
                let mut pool = WorkerPool::new(s, "test", 3);
                for _ in 0..3 {
                    let seen = Arc::clone(&seen);
                    pool.spawn(move |ticket| {
                        let at = ticket.arrive_and_wait().expect("run must not abort");
                        seen.lock().unwrap().push(at);
                    })?;
                }
                pool.start_with(|| Instant::now() + lead)
            })
            .expect("happy path must succeed")
        };

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        assert!(seen.iter().all(|at| *at == start));
        assert!(
            start > Instant::now() - lead,
            "the stamp must be the caller's future instant, not `now`"
        );
    }

    #[test]
    fn zero_workers_is_not_a_hang() {
        let finished = finishes_within(PROMPT, || {
            let (values, _) = run_pool(0, |_, ticket| {
                let _ = ticket.arrive_and_wait();
                0
            })
            .expect("an empty pool starts trivially");
            assert!(values.is_empty());
        });
        assert!(finished, "an empty pool hung");
    }

    #[test]
    fn abort_releases_workers_already_parked_at_the_gate() {
        // The ordering that used to strand peers: two workers reach the gate
        // first and park, and only then does the run fail. Both must wake.
        let woke = Arc::new(AtomicUsize::new(0));
        let gate_reached = Arc::new(Barrier::new(3));

        let finished = finishes_within(PROMPT, {
            let woke = Arc::clone(&woke);
            let gate_reached = Arc::clone(&gate_reached);
            move || {
                let err = std::thread::scope(|s| {
                    let mut pool: WorkerPool<'_, '_, ()> = WorkerPool::new(s, "test", 3);
                    for _ in 0..2 {
                        let woke = Arc::clone(&woke);
                        let gate_reached = Arc::clone(&gate_reached);
                        pool.spawn(move |ticket| {
                            gate_reached.wait();
                            assert!(
                                ticket.arrive_and_wait().is_none(),
                                "an aborted run must not hand out a start instant"
                            );
                            woke.fetch_add(1, Ordering::SeqCst);
                        })?;
                    }
                    // The third worker fails only once the first two have parked
                    // — the ordering that used to strand them.
                    let gate_reached = Arc::clone(&gate_reached);
                    pool.spawn(move |ticket| {
                        gate_reached.wait();
                        ticket.fail("simulated late failure");
                    })?;
                    pool.start()
                })
                .expect_err("the run must fail");
                assert!(err.contains("simulated late failure"), "{err}");
            }
        });

        assert!(finished, "aborting the gate stranded the parked workers");
        assert_eq!(
            woke.load(Ordering::SeqCst),
            2,
            "both parked workers must wake"
        );
    }

    #[test]
    fn early_return_from_the_scope_closure_does_not_strand_parked_workers() {
        // Guards the `Drop for WorkerPool` safety net: a `?` on something other
        // than `spawn`/`start` must not leave workers parked while
        // `thread::scope` tries to join them.
        let finished = finishes_within(PROMPT, || {
            let gate_reached = Arc::new(Barrier::new(2));
            let err: Result<(), String> = std::thread::scope(|s| {
                let mut pool: WorkerPool<'_, '_, ()> = WorkerPool::new(s, "test", 1);
                let reached = Arc::clone(&gate_reached);
                pool.spawn(move |ticket| {
                    reached.wait();
                    let _ = ticket.arrive_and_wait();
                })?;
                gate_reached.wait();
                Err("something else went wrong".to_string())
            });
            assert_eq!(err.unwrap_err(), "something else went wrong");
        });
        assert!(
            finished,
            "an unrelated early return stranded a parked worker"
        );
    }
}
