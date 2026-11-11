//! Duration mode for `msmarco-disk-v1` windows (`compound.duration_s`).
//!
//! By default a disk window `[offset, offset + --queries)` is dispatched once,
//! each query exactly once, and the point lasts as long as that takes. With
//! `compound.duration_s` set, the point runs for that many seconds instead:
//! the workers draw the window's timed queries in strict dispatch order and
//! wrap to the start of the window when it is exhausted, until the deadline.
//! Dispatch `k` (0-based, counted across all workers) always sends timed line
//! `first_line + (k mod window)`, on every engine, so two engines that ran the
//! same window for the same time sent the same query sequence (each up to its
//! own dispatched count).
//!
//! The deadline is checked before each dispatch: no worker starts a query
//! once it has passed (one that passed the check just before may start
//! microseconds after). A query in flight at the deadline completes and is
//! counted (`completed_after_deadline` says how many finished after it); the
//! throughput is completed queries over the timed span, from the shared start
//! until every worker has joined (`last_completion_s` is recorded beside it).
//! The completion time of every query (after its reply is checked), on the
//! client's clock, is bucketed into fixed intervals ([`INTERVALS_S`]) with
//! their counts and latency percentiles, so per-interval stability is
//! measured, not reconstructed. The first failed query stops dispatch on
//! every worker: the run is refused anyway, so it is not kept running to the
//! deadline.
//!
//! The window is loaded once and reused on every pass; it does not grow with
//! the duration. What does grow is per completed query: 16 bytes of timing
//! (latency and completion time), and under `--dump-results` one captured
//! record per dispatch (its ids and scores), which at high QPS for minutes is
//! gigabytes.
//!
//! A query repeated on a later pass may be served from a cache the first
//! pass warmed (page cache, engine caches): a window smaller than the number
//! of queries the run dispatches is no longer a cold-read workload. The result
//! records whether any query repeated, and how many full passes ran.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Longest accepted `compound.duration_s` (one week).
pub const MAX_DURATION_S: f64 = 7.0 * 86_400.0;

/// Widths of the recorded completion intervals, in seconds.
pub const INTERVALS_S: [f64; 2] = [10.0, 60.0];

/// The dispatch order duration mode follows, as recorded in the result.
pub const ORDER: &str = "strict dispatch order: dispatch k (0-based, across all workers) sends \
     timed line first_line + (k mod queries); the window wraps to its start when exhausted, \
     until the deadline";

/// How the throughput of a duration-mode point is computed, as recorded.
pub const QPS_BASIS: &str = "the deadline is checked before each dispatch; queries in flight \
     at the deadline complete and are counted. rps = completed / timed_span_s (shared start until \
     every worker has joined; last_completion_s is the last completion); qps_by_deadline = \
     completed_by_deadline / duration_s; intervals cover [0, duration_s) only";

/// Validate a `compound.duration_s` value.
pub fn check_duration(seconds: f64) -> Result<Duration, String> {
    if !seconds.is_finite() || seconds <= 0.0 || seconds > MAX_DURATION_S {
        return Err(format!(
            "compound `duration_s` must be finite, greater than zero and at most \
             {MAX_DURATION_S} seconds, not {seconds}"
        ));
    }
    Ok(Duration::from_secs_f64(seconds))
}

/// Hands out the next query of a timed window to whichever worker asks,
/// shared by all workers of one search config.
///
/// Without a deadline it is the default schedule: each entry of `order` once,
/// in order. With one, it cycles `order` until the deadline.
#[derive(Debug)]
pub struct Dispatcher<'a> {
    order: &'a [usize],
    next: AtomicUsize,
    deadline: Option<Duration>,
    stopped: AtomicBool,
}

impl<'a> Dispatcher<'a> {
    pub fn new(order: &'a [usize], deadline: Option<Duration>) -> Self {
        Self {
            order,
            next: AtomicUsize::new(0),
            deadline,
            stopped: AtomicBool::new(false),
        }
    }

    /// A query failed. In duration mode no further query is dispatched (the
    /// run is refused anyway); without a deadline nothing changes and the
    /// window is dispatched to its end, as before.
    pub fn failed(&self) {
        if self.deadline.is_some() {
            self.stopped.store(true, Ordering::Relaxed);
        }
    }

    /// The next `(seq, index into the query set)` to send, or `None` when the
    /// worker must stop. `start` is the shared start of the timed window.
    /// The deadline is checked before a sequence number is taken, so the
    /// dispatched sequence numbers are exactly `0..dispatched()`, no gaps.
    pub fn next(&self, start: Instant) -> Option<(usize, usize)> {
        match self.deadline {
            None => {
                let seq = self.next.fetch_add(1, Ordering::Relaxed);
                self.order.get(seq).map(|&i| (seq, i))
            }
            Some(deadline) => {
                if self.order.is_empty()
                    || self.stopped.load(Ordering::Relaxed)
                    || start.elapsed() >= deadline
                {
                    return None;
                }
                let seq = self.next.fetch_add(1, Ordering::Relaxed);
                Some((seq, self.order[seq % self.order.len()]))
            }
        }
    }

    /// Queries handed out so far (call after the workers have joined).
    pub fn dispatched(&self) -> usize {
        let n = self.next.load(Ordering::Relaxed);
        match self.deadline {
            None => n.min(self.order.len()),
            Some(_) => n,
        }
    }
}

/// One worker's (or, appended, the whole run's) successful queries: each
/// latency, and in duration mode its completion time in seconds since the
/// shared start, in the same order.
#[derive(Debug, Default)]
pub struct Samples {
    pub latencies: Vec<f64>,
    completed_at: Option<Vec<f64>>,
}

impl Samples {
    /// `timestamps`: record completion times (duration mode only, so a
    /// single-pass run keeps exactly the memory and work it had).
    pub fn new(timestamps: bool) -> Self {
        Self {
            latencies: Vec::new(),
            completed_at: timestamps.then(Vec::new),
        }
    }

    /// Record a successful query that took `latency` seconds and has just
    /// completed; `start` is the shared start of the timed window.
    pub fn push(&mut self, latency: f64, start: Instant) {
        self.latencies.push(latency);
        if let Some(c) = &mut self.completed_at {
            c.push(start.elapsed().as_secs_f64());
        }
    }

    pub fn append(&mut self, other: Samples) {
        self.latencies.extend(other.latencies);
        if let (Some(a), Some(b)) = (&mut self.completed_at, other.completed_at) {
            a.extend(b);
        }
    }

    /// Completion times, empty unless recorded.
    pub fn completed_at(&self) -> &[f64] {
        self.completed_at.as_deref().unwrap_or(&[])
    }
}

/// Linear-interpolation percentile over an ascending slice (numpy's default,
/// as `compute_search_stats` uses).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        n => {
            let rank = q * (n - 1) as f64;
            let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
            sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
        }
    }
}

fn latency_stats(mut latencies: Vec<f64>) -> Value {
    latencies.sort_by(|a, b| a.total_cmp(b));
    if latencies.is_empty() {
        return json!({"p50_time": null, "p95_time": null, "p99_time": null, "max_time": null});
    }
    json!({
        "p50_time": percentile(&latencies, 0.50),
        "p95_time": percentile(&latencies, 0.95),
        "p99_time": percentile(&latencies, 0.99),
        "max_time": latencies[latencies.len() - 1],
    })
}

/// Completions bucketed into `[i * width, (i + 1) * width)` intervals of the
/// timed window, up to the deadline (the last interval is shorter when the
/// duration is not a multiple of `width`). Every interval is listed, empty
/// ones included. Latencies are in seconds, like the result's `p50_time`.
pub fn intervals(completed_at: &[f64], latencies: &[f64], duration_s: f64, width: f64) -> Value {
    let n = (duration_s / width).ceil().max(1.0) as usize;
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (&at, &lat) in completed_at.iter().zip(latencies) {
        if at < duration_s {
            let i = ((at / width) as usize).min(n - 1);
            buckets[i].push(lat);
        }
    }
    let list: Vec<Value> = buckets
        .into_iter()
        .enumerate()
        .map(|(i, lat)| {
            let start = i as f64 * width;
            let end = (start + width).min(duration_s);
            let mut b = json!({
                "start_s": start,
                "end_s": end,
                "completed": lat.len(),
                "qps": lat.len() as f64 / (end - start),
            });
            if let (Value::Object(b), Value::Object(s)) = (&mut b, latency_stats(lat)) {
                b.extend(s);
            }
            b
        })
        .collect();
    json!({"width_s": width, "buckets": list})
}

/// Everything a duration-mode result records about its schedule and timing.
///
/// `window` is the disk window record ([`crate::fulltext::DiskWindow`]'s
/// `record`), `window_len` its number of timed queries, `dispatched` the
/// queries sent, `completed_at` / `latencies` one entry per successful query
/// (seconds since the shared start, at completion; latency in seconds), and
/// `span_s` the timed span the result's `rps` is computed over.
pub fn record(
    duration_s: f64,
    window: &Value,
    window_len: usize,
    dispatched: usize,
    completed_at: &[f64],
    latencies: &[f64],
    span_s: f64,
) -> Value {
    let by_deadline = completed_at.iter().filter(|&&t| t < duration_s).count();
    let (late_lat, last): (Vec<f64>, f64) = (
        completed_at
            .iter()
            .zip(latencies)
            .filter(|(&t, _)| t >= duration_s)
            .map(|(_, &l)| l)
            .collect(),
        completed_at.iter().copied().fold(0.0, f64::max),
    );
    let mut after = json!({"completed": late_lat.len()});
    if let (Value::Object(a), Value::Object(s)) = (&mut after, latency_stats(late_lat)) {
        a.extend(s);
    }
    json!({
        "mode": "cycle-window",
        "duration_s": duration_s,
        "window": {
            "offset": window["offset"],
            "queries": window_len,
            "first_line": window["first_line"],
            "last_line": window["last_line"],
        },
        "order": ORDER,
        "dispatched": dispatched,
        "completed": completed_at.len(),
        "full_passes": dispatched.checked_div(window_len).unwrap_or(0),
        "partial_position": dispatched.checked_rem(window_len).unwrap_or(0),
        "repeated": dispatched > window_len,
        "timed_span_s": span_s,
        "last_completion_s": last,
        "completed_by_deadline": by_deadline,
        "completed_after_deadline": completed_at.len() - by_deadline,
        "qps_by_deadline": by_deadline as f64 / duration_s,
        "qps_basis": QPS_BASIS,
        "after_deadline": after,
        "intervals": INTERVALS_S
            .iter()
            .map(|&w| intervals(completed_at, latencies, duration_s, w))
            .collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_deadline_each_entry_is_dispatched_once_in_order() {
        let order = [7, 8, 9];
        let d = Dispatcher::new(&order, None);
        let start = Instant::now();
        let got: Vec<_> = std::iter::from_fn(|| d.next(start)).collect();
        assert_eq!(got, vec![(0, 7), (1, 8), (2, 9)]);
        assert_eq!(d.next(start), None);
        assert_eq!(d.dispatched(), 3);
    }

    #[test]
    fn with_a_deadline_the_window_wraps_in_order_until_it_passes() {
        let order = [4, 5, 6];
        let d = Dispatcher::new(&order, Some(Duration::from_secs(3600)));
        let start = Instant::now();
        let got: Vec<_> = (0..8).map(|_| d.next(start).unwrap()).collect();
        assert_eq!(
            got,
            vec![
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 4),
                (7, 5)
            ]
        );
        assert_eq!(d.dispatched(), 8);
        // Past the deadline nothing more is handed out, and no sequence
        // number is consumed.
        let past = Dispatcher::new(&order, Some(Duration::from_millis(1)));
        let old = Instant::now() - Duration::from_millis(5);
        assert_eq!(past.next(old), None);
        assert_eq!(past.dispatched(), 0);
    }

    #[test]
    fn a_failure_stops_duration_mode_only() {
        let order = [1, 2];
        let start = Instant::now();
        let timed = Dispatcher::new(&order, Some(Duration::from_secs(3600)));
        assert!(timed.next(start).is_some());
        timed.failed();
        assert_eq!(timed.next(start), None);
        assert_eq!(timed.dispatched(), 1);
        let once = Dispatcher::new(&order, None);
        assert!(once.next(start).is_some());
        once.failed();
        assert_eq!(once.next(start), Some((1, 2)));
    }

    #[test]
    fn concurrent_workers_dispatch_a_gapless_sequence_until_the_deadline() {
        let order: Vec<usize> = (0..5).collect();
        let d = Dispatcher::new(&order, Some(Duration::from_millis(300)));
        let start = Instant::now();
        let mut all: Vec<(usize, usize)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        let mut mine = Vec::new();
                        while let Some(x) = d.next(start) {
                            mine.push(x);
                            std::thread::sleep(Duration::from_micros(200));
                        }
                        mine
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        assert!(start.elapsed() >= Duration::from_millis(300));
        all.sort();
        assert_eq!(all.len(), d.dispatched());
        assert!(all.len() > order.len(), "the window must have wrapped");
        for (k, (seq, i)) in all.iter().enumerate() {
            assert_eq!((*seq, *i), (k, k % order.len()));
        }
    }

    #[test]
    fn samples_keep_completion_times_only_when_asked() {
        let start = Instant::now();
        let mut off = Samples::new(false);
        off.push(0.5, start);
        assert_eq!(off.latencies, vec![0.5]);
        assert!(off.completed_at().is_empty());
        let mut all = Samples::new(true);
        for w in 0..3 {
            let mut s = Samples::new(true);
            s.push(w as f64, start);
            s.push(w as f64 + 0.5, start);
            all.append(s);
        }
        assert_eq!(all.latencies, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(all.completed_at().len(), 6);
    }

    #[test]
    fn duration_is_validated() {
        assert!(check_duration(600.0).is_ok());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, MAX_DURATION_S + 1.0] {
            assert!(check_duration(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn record_counts_passes_deadline_and_intervals() {
        let window = json!({"offset": 10, "first_line": 75, "last_line": 78});
        // Window of 4; 10 dispatched: 2 full passes, then 2 more.
        let at = [0.5, 1.0, 9.9, 10.0, 15.0, 59.0, 60.0, 119.0, 120.0, 120.5];
        let lat = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10];
        let r = record(120.0, &window, 4, 10, &at, &lat, 120.6);
        assert_eq!(r["mode"], "cycle-window");
        assert_eq!(r["full_passes"], 2);
        assert_eq!(r["partial_position"], 2);
        assert_eq!(r["repeated"], true);
        assert_eq!(r["window"]["queries"], 4);
        assert_eq!(r["window"]["offset"], 10);
        assert_eq!(r["completed"], 10);
        assert_eq!(r["completed_by_deadline"], 8);
        assert_eq!(r["completed_after_deadline"], 2);
        assert_eq!(r["after_deadline"]["completed"], 2);
        assert_eq!(r["qps_by_deadline"], 8.0 / 120.0);
        assert_eq!(r["last_completion_s"], 120.5);
        let ten = &r["intervals"][0];
        assert_eq!(ten["width_s"], 10.0);
        let b = ten["buckets"].as_array().unwrap();
        assert_eq!(b.len(), 12);
        assert_eq!(b[0]["completed"], 3);
        assert_eq!(b[0]["qps"], 0.3);
        assert_eq!(b[0]["p50_time"], 0.02);
        assert_eq!(b[1]["completed"], 2);
        assert_eq!(b[2]["completed"], 0);
        assert!(b[2]["p50_time"].is_null());
        assert_eq!(b[11]["completed"], 1);
        let sixty = r["intervals"][1]["buckets"].as_array().unwrap();
        assert_eq!(sixty.len(), 2);
        assert_eq!(sixty[0]["completed"], 6);
        assert_eq!(sixty[1]["completed"], 2);
        assert_eq!(sixty[1]["end_s"], 120.0);
        // Total over the intervals is every completion before the deadline.
        let sum: u64 = b.iter().map(|x| x["completed"].as_u64().unwrap()).sum();
        assert_eq!(sum, 8);
    }

    #[test]
    fn a_window_not_finished_records_no_repeat() {
        let r = record(1.0, &json!({}), 100, 40, &[0.5; 40], &[0.01; 40], 1.0);
        assert_eq!(r["full_passes"], 0);
        assert_eq!(r["partial_position"], 40);
        assert_eq!(r["repeated"], false);
        assert_eq!(r["completed_after_deadline"], 0);
        // Exactly one full pass: nothing ran twice.
        let one = record(1.0, &json!({}), 4, 4, &[0.5; 4], &[0.01; 4], 1.0);
        assert_eq!(one["full_passes"], 1);
        assert_eq!(one["partial_position"], 0);
        assert_eq!(one["repeated"], false);
        let five = record(1.0, &json!({}), 4, 5, &[0.5; 5], &[0.01; 5], 1.0);
        assert_eq!(five["repeated"], true);
    }

    #[test]
    fn a_short_last_interval_uses_its_own_width() {
        let v = intervals(&[14.0], &[0.1], 15.0, 10.0);
        let b = v["buckets"].as_array().unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[1]["end_s"], 15.0);
        assert_eq!(b[1]["qps"], 0.2);
    }
}
