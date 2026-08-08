//! Overhead-invariant guard tests (regression tripwire).
//!
//! These are pure source-scanning tests: they read the engine `.rs` files as
//! TEXT and assert that the measurement-overhead fixes from PRs #108 / #110 /
//! #111 / #113 stay in place. They do NOT need a database — they run in the
//! normal `cargo test` set and fail CI the moment someone reintroduces one of
//! the anti-patterns those PRs removed.
//!
//! Two invariants are locked in:
//!
//!   INV-2  No per-query cross-thread `Mutex<Vec>` push inside a timed worker
//!          loop. Every per-query latency/quality sample must land in a
//!          thread-local buffer that is merged on join — NOT pushed through a
//!          single `Arc<Mutex<Vec<f64>>>` that serializes workers at high
//!          parallelism. (#108 for the Redis-family filter-only/mixed paths,
//!          #110 for turbopuffer.)
//!
//!   INV-3  One stats path. Every search / filter-only / mixed path computes
//!          its latency percentiles through the shared `compute_search_stats`
//!          (linear-interpolation) helper, NOT a hand-rolled nearest-rank
//!          `(len as f64 * q) as usize` index — the biased method that made
//!          `p99 == max` for `N <= 100`. (#108.)
//!
//! Keep the assertions on DISTINCTIVE substrings that will not false-positive on
//! unrelated code (e.g. the legitimate `errors.lock().unwrap().push(e)` in the
//! UPLOAD paths must stay allowed). The percentile-parity behaviour itself is
//! covered by the unit test `engine::tests::filter_mixed_stats_use_linear_percentiles`
//! (p99 == 99.01 on 1..=100); this file guards the SOURCE shape that keeps that
//! true across all engines.

use std::fs;
use std::path::PathBuf;

/// Engine source files that own a timed per-query worker loop whose sample
/// buffers were converted from `Arc<Mutex<Vec>>` to thread-local (#108/#110/#111)
/// and whose percentiles route through `compute_search_stats` (#108).
const ENGINE_FILES: &[&str] = &[
    "redis.rs",
    "valkey.rs",
    "vectorsets.rs",
    "mongodb_engine.rs",
    "turbopuffer.rs",
    "dragonfly.rs",
];

fn engine_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/vector_db_benchmark/engine")
}

fn read_engine(file: &str) -> String {
    let path = engine_dir().join(file);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e))
}

/// INV-2 — no per-query metric buffer is pushed through a cross-thread mutex in
/// a timed loop. We ban the EXACT sample-buffer lock-push idioms that #108/#110
/// removed. Each pattern below named a per-query metric buffer
/// (`search_times`/`precisions`/`recalls`/`mrrs`/`ndcgs`/`update_times`) that
/// used to be an `Arc<Mutex<Vec<f64>>>`; if any reappears, a timed worker loop
/// is once again serializing on a shared lock per query.
///
/// NOTE: this deliberately does NOT ban `.lock().unwrap().push(` in general —
/// the upload/error paths legitimately collect errors under a mutex
/// (`errors.lock().unwrap().push(e)`), which is off the timed hot path.
#[test]
fn inv2_no_per_query_mutex_push_in_timed_loops() {
    // (banned substring, what it used to guard)
    let banned: &[(&str, &str)] = &[
        (
            "search_times.lock().unwrap().push",
            "per-query search latency pushed through a shared Mutex<Vec> (was #108 FIX1)",
        ),
        (
            "update_times.lock().unwrap().push",
            "per-op update latency pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "precisions.lock().unwrap().push",
            "per-query precision pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "recalls.lock().unwrap().push",
            "per-query recall pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "mrrs.lock().unwrap().push",
            "per-query MRR pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "ndcgs.lock().unwrap().push",
            "per-query NDCG pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "times.lock().unwrap().push",
            "per-query latency pushed through a shared Mutex<Vec> in a timed loop",
        ),
    ];

    let mut violations = Vec::new();
    for &file in ENGINE_FILES {
        let src = read_engine(file);
        for &(pat, why) in banned {
            let count = src.matches(pat).count();
            if count != 0 {
                violations.push(format!(
                    "  {} contains `{}` ({}x) — {}",
                    file, pat, count, why
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "INV-2 VIOLATED: a per-query metric buffer is pushed through a cross-thread \
         Mutex<Vec> inside a timed worker loop. Accumulate into a THREAD-LOCAL Vec \
         and merge on join instead (see redis.rs::search). Offenders:\n{}",
        violations.join("\n")
    );
}

/// INV-3a — every timed engine still routes its stats through the single shared
/// `compute_search_stats` helper. If an engine stops referencing it, it has
/// almost certainly grown a private (likely nearest-rank) percentile path again.
#[test]
fn inv3_all_engines_use_shared_compute_search_stats() {
    let mut missing = Vec::new();
    for &file in ENGINE_FILES {
        let src = read_engine(file);
        if !src.contains("compute_search_stats") {
            missing.push(file);
        }
    }
    assert!(
        missing.is_empty(),
        "INV-3 VIOLATED: these engines no longer reference `compute_search_stats`, so \
         their search/filter-only/mixed stats are no longer on the shared \
         linear-percentile footing: {:?}",
        missing
    );
}

/// INV-3b — the hand-rolled nearest-rank percentile idiom `(len as f64 * q) as
/// usize` (which made `p99 == max` for `N <= 100`) must not reappear in any
/// engine file. The three multiplier substrings below are distinctive of that
/// specific indexing pattern and do not occur in correct code (which calls
/// `percentile_linear` / `compute_search_stats`).
#[test]
fn inv3_no_nearest_rank_percentile_indexing() {
    let banned: &[&str] = &[
        "as f64 * 0.50) as usize",
        "as f64 * 0.95) as usize",
        "as f64 * 0.99) as usize",
    ];

    let mut violations = Vec::new();
    // Scan every engine source, not just the five — the biased idiom is wrong
    // anywhere it appears.
    for entry in fs::read_dir(engine_dir()).expect("read engine dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = fs::read_to_string(&path).expect("read engine source");
        for &pat in banned {
            if src.contains(pat) {
                violations.push(format!(
                    "  {} contains `{}`",
                    path.file_name().unwrap().to_string_lossy(),
                    pat
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "INV-3 VIOLATED: a hand-rolled nearest-rank percentile index reappeared \
         (`(len as f64 * q) as usize`). This biases p99 upward and makes p99 == max \
         for N <= 100 — route through `percentile_linear` / `compute_search_stats` \
         instead. Offenders:\n{}",
        violations.join("\n")
    );
}

/// INV-4 — no fixed-count start barrier in a search harness (#214).
///
/// Every engine used to synchronize its measured-window start with
/// `Barrier::new(parallel + 1)`, a participant count fixed *before* the workers
/// exist. Two ordinary failures then hang the run forever with no output: the
/// OS refusing a thread (`Scope::spawn` panics; the workers already parked at
/// the barrier are joined by `thread::scope` and never released), and a worker
/// panicking before it arrives. `--search-timeout` defaults to `0.0`, so
/// nothing breaks the hang.
///
/// The replacement is `vector_db_benchmark::start_gate`, whose wait is
/// satisfied by ticket *outcomes* rather than a count. This guard fails the
/// moment a fixed-count barrier reappears in an engine.
#[test]
fn inv4_no_fixed_count_start_barrier_in_search_harnesses() {
    let mut violations = Vec::new();
    for entry in fs::read_dir(engine_dir()).expect("read engine dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = fs::read_to_string(&path).expect("read engine source");
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        for (lineno, line) in src.lines().enumerate() {
            if line.contains("Barrier::new(") {
                violations.push(format!("  {}:{} {}", name, lineno + 1, line.trim()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "INV-4 VIOLATED: a fixed-count start barrier reappeared in an engine. Sizing the \
         wait before the workers exist deadlocks the run when the OS refuses a thread or a \
         worker panics before arriving (#214) — use `vector_db_benchmark::start_gate::\
         WorkerPool` / `StartGate`, whose wait is satisfied by ticket outcomes. \
         Offenders:\n{}",
        violations.join("\n")
    );
}

/// INV-4b — every engine that fans out a timed search actually routes its start
/// through the gate. Without this, INV-4 could be "satisfied" by an engine that
/// grew some other fixed-count wait, or by one that silently lost its
/// synchronized start altogether.
#[test]
fn inv4_search_harnesses_route_through_the_start_gate() {
    // Engines with a parallel, gate-synchronized measured window. Turbopuffer is
    // absent deliberately: it has no warm-up gate, its workers start on spawn.
    const GATED: &[&str] = &[
        "chroma.rs",
        "dragonfly.rs",
        "elasticsearch.rs",
        "kividb.rs",
        "milvus.rs",
        "mongodb_engine.rs",
        "opensearch.rs",
        "pgvector.rs",
        "qdrant.rs",
        "redis.rs",
        "valkey.rs",
        "vectorsets.rs",
        "vertex.rs",
        "weaviate.rs",
    ];

    let mut missing = Vec::new();
    for &file in GATED {
        let src = read_engine(file);
        let uses_gate = src.contains("start_gate::WorkerPool")
            || src.contains("start_gate::{StartGate, WorkerPool}");
        let arrives = src.contains("ticket.arrive_and_wait()");
        if !uses_gate || !arrives {
            missing.push(format!(
                "  {} (imports gate: {}, parks at gate: {})",
                file, uses_gate, arrives
            ));
        }
    }

    assert!(
        missing.is_empty(),
        "INV-4b VIOLATED: a search harness no longer starts its workers through \
         `start_gate` (#214). Either it regressed to an ad-hoc synchronization \
         primitive, or its warm-up gate was dropped and connection setup is back \
         inside the measured window. Offenders:\n{}",
        missing.join("\n")
    );
}
