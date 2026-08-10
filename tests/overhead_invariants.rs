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
    strip_comments_and_strings(&read_engine_raw(file))
}

/// The raw file, comments and all — for the few checks that need to see them.
fn read_engine_raw(file: &str) -> String {
    let path = engine_dir().join(file);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e))
}

/// Blank out comments and string/char literals, preserving line structure.
///
/// Every guard in this file is a substring search over source text, and every
/// one of them was previously blind to the difference between code and prose.
/// That is not hypothetical: INV-4 bans the `Barrier` type, and all 14 engines
/// used to carry a comment *describing* the barrier it removed — so the guard
/// would have forbidden documenting the very bug it exists to prevent.
/// Newlines are preserved so reported line numbers stay true.
fn strip_comments_and_strings(src: &str) -> String {
    #[derive(PartialEq)]
    enum St {
        Code,
        Line,
        Block,
        Str,
        Chr,
        RawStr,
    }
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut st = St::Code;
    let mut depth = 0usize;
    let mut hashes = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied().unwrap_or('\0');
        match st {
            St::Code => {
                if c == '/' && next == '/' {
                    st = St::Line;
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '/' && next == '*' {
                    st = St::Block;
                    depth = 1;
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == 'r' && (next == '"' || next == '#') {
                    let prev_ident = i > 0 && (b[i - 1].is_alphanumeric() || b[i - 1] == '_');
                    if !prev_ident {
                        let mut j = i + 1;
                        let mut h = 0;
                        while j < b.len() && b[j] == '#' {
                            h += 1;
                            j += 1;
                        }
                        if j < b.len() && b[j] == '"' {
                            st = St::RawStr;
                            hashes = h;
                            out.push_str(&" ".repeat(j - i + 1));
                            i = j + 1;
                            continue;
                        }
                    }
                }
                if c == '"' {
                    st = St::Str;
                    out.push(' ');
                    i += 1;
                    continue;
                }
                if c == '\'' {
                    // lifetime (`'scope`) vs char literal: a char literal closes
                    // within four characters.
                    if let Some(k) = (1..=4).find(|k| b.get(i + k) == Some(&'\'')) {
                        st = St::Chr;
                        out.push_str(&" ".repeat(k));
                        i += k;
                        continue;
                    }
                }
                out.push(c);
                i += 1;
            }
            St::Line => {
                if c == '\n' {
                    st = St::Code;
                    out.push('\n');
                } else {
                    out.push(' ');
                }
                i += 1;
            }
            St::Block => {
                if c == '/' && next == '*' {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '*' && next == '/' {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        st = St::Code;
                    }
                    continue;
                }
                out.push(if c == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            St::Str | St::Chr => {
                if c == '\\' {
                    // An escape consumes two characters — but a `\`-plus-newline
                    // string continuation's second character IS the newline, and
                    // swallowing it shifts every line below it up by one. There
                    // are 219 such continuations under `src/`, so the guards
                    // would quote the wrong source line and, worse, match an
                    // `INV-4-ALLOW:` marker against the wrong line: a marker
                    // three lines above an unrelated barrier would exempt it.
                    out.push(' ');
                    match b.get(i + 1) {
                        Some('\n') => out.push('\n'),
                        Some(_) => out.push(' '),
                        None => {}
                    }
                    i += 2;
                    continue;
                }
                let closing = if st == St::Str { '"' } else { '\'' };
                if c == closing {
                    st = St::Code;
                }
                out.push(if c == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            St::RawStr => {
                if c == '"' {
                    let mut h = 0;
                    while b.get(i + 1 + h) == Some(&'#') {
                        h += 1;
                    }
                    if h >= hashes {
                        st = St::Code;
                        out.push_str(&" ".repeat(1 + hashes));
                        i += 1 + hashes;
                        continue;
                    }
                }
                out.push(if c == '\n' { '\n' } else { ' ' });
                i += 1;
            }
        }
    }
    out
}

/// Every `.rs` file under `src/`, as (repo-relative path, comment-stripped source).
fn all_sources() -> Vec<(String, String)> {
    fn walk(dir: &PathBuf, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("read dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    walk(&root.join("src"), &mut paths);
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let src = fs::read_to_string(&p).expect("read source");
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            (rel, strip_comments_and_strings(&src))
        })
        .collect()
}

/// Engines that own a gate-synchronized parallel search, derived from
/// `engine/mod.rs` rather than hardcoded, so a new engine is opted IN by
/// default and must be explicitly excused.
fn gated_engine_files() -> Vec<String> {
    // Modules with no parallel timed search of their own, or no gate by design.
    // Each needs a reason; adding a name here is the reviewable act.
    const EXCUSED: &[(&str, &str)] = &[
        ("index_naming", "pure helper, no engine"),
        ("redis_utils", "pure helper, no engine"),
        ("vertex_grpc", "transport codegen, no engine"),
        ("weaviate_grpc", "transport codegen, no engine"),
        ("filter_guard", "pure helper, no engine"),
        (
            "geo",
            "pure helper, no engine: great-circle encoding shared by vectorsets and milvus",
        ),
        (
            "turbopuffer",
            "its workers still start on spawn; the ungated harness itself is tracked by \
             UNGATED_FANOUT_DEBT below (#266), which is what will fail when it is converted",
        ),
    ];
    let mod_rs = read_engine_raw("mod.rs");
    let mut gated = Vec::new();
    for line in mod_rs.lines() {
        let line = line.trim();
        let Some(rest) = line
            .strip_prefix("mod ")
            .or_else(|| line.strip_prefix("pub mod "))
        else {
            continue;
        };
        let Some(name) = rest.strip_suffix(';') else {
            continue;
        };
        if EXCUSED.iter().any(|(e, _)| *e == name) {
            continue;
        }
        gated.push(format!("{name}.rs"));
    }
    assert!(
        gated.len() >= 14,
        "expected at least 14 gated engines from engine/mod.rs, found {}: {:?}",
        gated.len(),
        gated
    );
    gated
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

/// Does this raw line carry an `INV-4-ALLOW:` opt-out?
fn line_carries_allow(line: Option<&str>) -> bool {
    line.is_some_and(|l| l.contains("INV-4-ALLOW:"))
}

/// Is this raw line a STANDALONE `INV-4-ALLOW:` comment — one that annotates the
/// line below it rather than the line it trails?
fn standalone_allow(line: Option<&str>) -> bool {
    line.is_some_and(|l| l.trim_start().starts_with("//") && l.contains("INV-4-ALLOW:"))
}

/// INV-4 — no fixed-count start barrier anywhere under `src/` (#214).
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
/// satisfied by ticket *outcomes* rather than a count.
///
/// This bans the TYPE, not one spelling of the call, so a type alias, a UFCS
/// call, `use std::sync::Barrier as B`, and a helper in a different directory
/// are all caught. It scans comment-stripped source, so documenting the removed
/// bug is fine. A future *legitimate* barrier (an upload rendezvous, say) opts
/// out per line with an `INV-4-ALLOW:` marker and a reason — deliberately
/// noisy, so it surfaces in review.
#[test]
fn inv4_no_fixed_count_start_barrier() {
    // `start_gate.rs` owns the `legacy_*` tests that replicate the pre-fix
    // barrier shape and assert it deadlocks. Banning `Barrier` there would
    // delete the proof that the module is worth having.
    const OWNS_THE_PROOF: &str = "src/start_gate.rs";

    let mut violations = Vec::new();
    for (path, src) in all_sources() {
        if path == OWNS_THE_PROOF {
            continue;
        }
        let raw = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&path))
            .expect("read source");
        let raw_lines: Vec<&str> = raw.lines().collect();
        for (lineno, line) in src.lines().enumerate() {
            let mentions_type = line.contains("Barrier::")
                || line.contains("sync::Barrier")
                || line.contains(" Barrier<")
                || line.contains(": Barrier")
                || line.contains("Barrier>")
                || line.contains(", Barrier")
                || line.contains("{Barrier");
            if !mentions_type {
                continue;
            }
            // The opt-out marker lives in a comment, so read the RAW line. It
            // may trail the offending line, or sit on the line immediately
            // above as a STANDALONE comment (a `use` statement reads better
            // with the reason above it). "Standalone" matters: a marker
            // trailing line N must not also exempt line N+1, or one annotated
            // barrier quietly covers its unannotated neighbour.
            if line_carries_allow(raw_lines.get(lineno).copied())
                || (lineno > 0 && standalone_allow(raw_lines.get(lineno - 1).copied()))
            {
                continue;
            }
            violations.push(format!(
                "  {}:{} {}",
                path,
                lineno + 1,
                raw_lines.get(lineno).unwrap_or(&"").trim()
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "INV-4 VIOLATED: `std::sync::Barrier` reappeared under src/. Sizing a wait before the \
         workers exist deadlocks the run when the OS refuses a thread or a worker panics before \
         arriving (#214) — use `vector_db_benchmark::start_gate::WorkerPool` / `StartGate`, whose \
         wait is satisfied by ticket outcomes. If a barrier here genuinely cannot deadlock, add \
         `// INV-4-ALLOW: <reason>` on the line. Offenders:\n{}",
        violations.join("\n")
    );
}

/// One `fn` item's comment-stripped body, located by brace matching.
struct FnSpan {
    name: String,
    /// Byte offset of the body's opening `{`.
    start: usize,
    /// Byte offset of the body's matching `}`.
    end: usize,
}

/// Every `fn` item in `src` (already comment/string-stripped), with the exact
/// extent of its body.
///
/// Brace matching rather than "up to the next `fn`", because the second form
/// bleeds one harness's body into the next one's — which is precisely how a
/// missing gate hides: `search_mixed`'s ungated fan-out would inherit the
/// `WorkerPool::new` that belongs to `search`.
fn fn_spans(src: &str) -> Vec<FnSpan> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = src[cursor..].find("fn ") {
        let at = cursor + rel;
        cursor = at + 3;
        // Token boundary: skip `...fn ` inside a longer identifier. A UTF-8
        // continuation byte is >= 0x80 and so reads as a boundary, which is the
        // safe direction (we consider the item, we do not skip it).
        if at > 0 {
            let prev = b[at - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                continue;
            }
        }
        let name: String = src[at + 3..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        // The first `{` after the signature opens the body; a `;` first means a
        // bodyless declaration (trait method). Neither character can occur in a
        // Rust return type or where-clause between the two.
        let mut j = at + 3 + name.len();
        let body_start = loop {
            match b.get(j) {
                Some(b'{') => break Some(j),
                Some(b';') | None => break None,
                Some(_) => j += 1,
            }
        };
        let Some(bs) = body_start else { continue };
        let mut depth = 0usize;
        let mut k = bs;
        let body_end = loop {
            match b.get(k) {
                Some(b'{') => depth += 1,
                Some(b'}') => {
                    depth -= 1;
                    if depth == 0 {
                        break Some(k);
                    }
                }
                None => break None,
                _ => {}
            }
            k += 1;
        };
        let Some(be) = body_end else { continue };
        out.push(FnSpan {
            name,
            start: bs,
            end: be,
        });
    }
    out
}

/// A `thread::scope` worker fan-out, attributed to the function that owns it.
struct FanOut {
    file: String,
    func: String,
    body: String,
}

/// Every `thread::scope(...)` fan-out under `engine/`, one entry per owning
/// function.
///
/// Attribution is to the SMALLEST enclosing `fn`, so a nested helper is
/// reported as itself rather than as its parent.
fn fan_out_harnesses() -> Vec<FanOut> {
    let mut files: Vec<String> = fs::read_dir(engine_dir())
        .expect("read engine dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs"))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    files.sort();

    let mut out = Vec::new();
    for file in files {
        let src = read_engine(&file);
        let spans = fn_spans(&src);
        let mut owners: Vec<usize> = Vec::new();
        let mut from = 0usize;
        while let Some(rel) = src[from..].find("thread::scope(") {
            let at = from + rel;
            from = at + 1;
            let owner = spans
                .iter()
                .enumerate()
                .filter(|(_, s)| s.start < at && at < s.end)
                .min_by_key(|(_, s)| s.end - s.start)
                .map(|(i, _)| i);
            let Some(owner) = owner else {
                panic!(
                    "{file}: a `thread::scope(` at byte {at} sits outside every `fn` body — \
                     `fn_spans` mis-parsed the file, and every gate check below would silently \
                     skip that fan-out"
                );
            };
            if !owners.contains(&owner) {
                owners.push(owner);
            }
        }
        for owner in owners {
            let s = &spans[owner];
            out.push(FanOut {
                file: file.clone(),
                func: s.name.clone(),
                body: src[s.start..=s.end].to_string(),
            });
        }
    }
    out
}

/// Timed fan-outs that predate the start gate and have NOT been converted yet
/// — the #266 backlog. Each entry is a *claim that the harness is still broken*,
/// not a permanent exemption.
///
/// Read this together with the assertions in
/// `inv4_search_harnesses_route_through_the_start_gate`, which enforce a
/// **bijection** between this list and the set of ungated timed fan-outs found
/// in the source:
///
/// * an ungated harness that is NOT listed fails the build — new debt cannot be
///   added silently;
/// * a listed harness that IS gated fails the build — the only way to claim the
///   debt is to DELETE its entry, so the list can never mark work as paid that
///   was not done (#255);
/// * a listed harness that no longer exists fails the build — a renamed or
///   deleted function cannot leave a stale entry covering nothing.
///
/// Fixing one of these is therefore a two-line change: convert the harness,
/// remove its row. Leaving the row in place is a hard failure, not a warning.
const UNGATED_FANOUT_DEBT: &[(&str, &str, &str)] = &[
    (
        "redis.rs",
        "search_filter_only",
        "#266: --skip-vector-index path never converted to WorkerPool",
    ),
    (
        "redis.rs",
        "search_mixed",
        "#266: --update-search-ratio path never converted to WorkerPool",
    ),
    (
        "valkey.rs",
        "search_filter_only",
        "#266: --skip-vector-index path never converted to WorkerPool",
    ),
    (
        "valkey.rs",
        "search_mixed",
        "#266: --update-search-ratio path never converted to WorkerPool",
    ),
    (
        "vectorsets.rs",
        "search_mixed",
        "#266: --update-search-ratio path never converted to WorkerPool",
    ),
    (
        "turbopuffer.rs",
        "search",
        "#266: workers start measuring on spawn; also EXCUSED from gated_engine_files()",
    ),
];

/// INV-4b — every TIMED worker fan-out routes its start through the gate, once
/// per harness, in the right place.
///
/// The 2026-08 shape of this guard counted `WorkerPool::new` occurrences per
/// FILE and required one park each. That counts *gated* harnesses, so a fan-out
/// that never created a pool was structurally invisible: MongoDB passed at 1
/// gated harness out of 3, while `search_mixed` and `search_filter_only` were
/// timing their workers from spawn and silently dropping any worker that could
/// not connect (#307). A mutation deleting an existing gate was caught; a
/// harness that never had one was not.
///
/// It now enumerates the fan-outs themselves — every `thread::scope(...)` under
/// `engine/`, attributed to its owning `fn` — and classifies each one:
///
/// * publishes search stats, or is named `search*` -> TIMED, must be gated;
/// * named `upload*` and publishes no search stats -> untimed, skipped;
/// * anything else -> unclassified, and fails. Default-in: a new fan-out is
///   covered without anyone remembering to add it here.
///
/// The pre-gate backlog lives in `UNGATED_FANOUT_DEBT` and is checked for exact
/// agreement with reality in both directions.
#[test]
fn inv4_search_harnesses_route_through_the_start_gate() {
    let mut problems = Vec::new();

    // (a) FILE level: every non-excused engine still uses the gate somewhere.
    //     Kept because it also covers an engine that stops fanning out with
    //     `thread::scope` altogether (rayon, a tokio task set) and so would
    //     vanish from the per-harness census below.
    for file in gated_engine_files() {
        let src = read_engine(&file);
        if src.matches("WorkerPool::new(").count() + src.matches("StartGate::new()").count() == 0 {
            problems.push(format!(
                "  {file}: no `WorkerPool::new` / `StartGate::new` — the synchronized start is \
                 gone, so connection setup and the cold first query are back inside the measured \
                 window"
            ));
        }
    }

    // (b) HARNESS level.
    let harnesses = fan_out_harnesses();
    let mut seen: Vec<(String, String)> = Vec::new();
    // (file, func) -> is it gated? Used to audit the debt list afterwards.
    let mut timed: Vec<(String, String, bool)> = Vec::new();

    for h in &harnesses {
        let key = (h.file.clone(), h.func.clone());
        if seen.contains(&key) {
            problems.push(format!(
                "  {}::{}: two fan-out functions share a name, so the debt list below cannot \
                 address them individually — rename one",
                h.file, h.func
            ));
            continue;
        }
        seen.push(key);

        let publishes_search_stats = h.body.contains("compute_search_stats(");
        let is_timed = publishes_search_stats || h.func.starts_with("search");
        if !is_timed {
            if h.func.starts_with("upload") {
                continue; // untimed ingest fan-out: no measured window to protect.
            }
            problems.push(format!(
                "  {}::{}: fans out worker threads but is neither a `search*`/stats-publishing \
                 harness nor an `upload*` one. Classify it: if its workers are timed it must go \
                 through `WorkerPool`; if not, name it `upload*` or excuse it here",
                h.file, h.func
            ));
            continue;
        }

        let pools = h.body.matches("WorkerPool::new(").count();
        let bare_gates = h.body.matches("StartGate::new()").count();
        let gates = pools + bare_gates;
        timed.push((h.file.clone(), h.func.clone(), gates > 0));

        if gates == 0 {
            let excused = UNGATED_FANOUT_DEBT
                .iter()
                .any(|(f, n, _)| *f == h.file && *n == h.func);
            if !excused {
                problems.push(format!(
                    "  {}::{}: times a `thread::scope` worker fan-out with NO \
                     `WorkerPool::new` / `StartGate::new`. Connection setup and the cold first \
                     query are inside the measured window, and a worker that cannot connect \
                     returns empty while the result is still stamped with the requested \
                     `parallel` (#214/#307). Route it through \
                     `vector_db_benchmark::start_gate::WorkerPool`",
                    h.file, h.func
                ));
            }
            continue;
        }

        // (c) exactly one park per gate in THIS harness. Catches un-gating one of
        //     the two gates weaviate's `search` drives (gRPC + GraphQL).
        let parks = h.body.matches("ticket.arrive_and_wait()").count();
        if parks != gates {
            problems.push(format!(
                "  {}::{}: {gates} gate(s) but {parks} `ticket.arrive_and_wait()` call(s) — one \
                 of them no longer parks at the gate",
                h.file, h.func
            ));
        }

        // (d) the abort verdict is honoured. `let _ = ticket.arrive_and_wait();`
        //     compiles, satisfies `#[must_use]`, and silently lets every worker
        //     run on its own clock after an aborted start.
        if h.body.contains("let _ = ticket.arrive_and_wait()") {
            problems.push(format!(
                "  {}::{}: discards the `arrive_and_wait` verdict with `let _ =`. `None` means \
                 the run was aborted; the worker must return, not measure",
                h.file, h.func
            ));
        }

        // (e) the park sits BELOW setup. Every gated harness reports at least one
        //     setup failure through `ticket.fail(...)` before it parks; if the
        //     first park precedes the first `fail`, the park was hoisted above
        //     client construction / the prime query, putting connection setup
        //     back inside the measured window.
        let (Some(first_park), Some(first_fail)) = (
            h.body.find("ticket.arrive_and_wait()"),
            h.body.find("ticket.fail("),
        ) else {
            problems.push(format!(
                "  {}::{}: no `ticket.fail(...)` — a worker that cannot set itself up is \
                 silently reducing the run's real concurrency again",
                h.file, h.func
            ));
            continue;
        };
        if first_park < first_fail {
            problems.push(format!(
                "  {}::{}: parks at the gate before its first setup-failure arm, so client \
                 construction and the prime query happen AFTER the start stamp — connection \
                 setup is back inside the measured window",
                h.file, h.func
            ));
        }
    }

    // (f) the debt list agrees with reality in BOTH directions. A row that no
    //     longer describes a broken harness is a failure, never a silent pass:
    //     deleting the row is the only way to claim the fix (#255).
    for (file, func, why) in UNGATED_FANOUT_DEBT {
        assert!(
            why.contains("#266"),
            "UNGATED_FANOUT_DEBT row {file}::{func} must cite the tracking issue (#266)"
        );
        match timed.iter().find(|(f, n, _)| f == file && n == func) {
            None => problems.push(format!(
                "  {file}::{func}: listed in UNGATED_FANOUT_DEBT but no such timed fan-out \
                 exists (renamed, deleted, or no longer fans out). Remove the row"
            )),
            Some((_, _, true)) => problems.push(format!(
                "  {file}::{func}: listed in UNGATED_FANOUT_DEBT but it IS gated now. The debt \
                 is paid — DELETE the row. Leaving it would let the next un-gating of this \
                 harness pass silently"
            )),
            Some((_, _, false)) => {}
        }
    }

    assert!(
        problems.is_empty(),
        "INV-4b VIOLATED (#214/#266/#307):\n{}",
        problems.join("\n")
    );
}

/// The fan-out census must actually find the harnesses — a parser that returns
/// nothing would make INV-4b vacuously green.
///
/// Pins the two properties INV-4b's correctness rests on:
/// 1. brace-matched bodies do not bleed into the next function (mongodb's
///    `search`, `search_mixed` and `search_filter_only` are three separate
///    harnesses, and each must see only its own gate);
/// 2. upload fan-outs are classified untimed, timed ones are not.
#[test]
fn fan_out_census_separates_each_harness_from_its_neighbours() {
    let harnesses = fan_out_harnesses();
    assert!(
        harnesses.len() >= 30,
        "expected the census to find every engine's upload + search fan-outs, found {}",
        harnesses.len()
    );

    let find = |file: &str, func: &str| {
        harnesses
            .iter()
            .find(|h| h.file == file && h.func == func)
            .unwrap_or_else(|| panic!("census lost {file}::{func}"))
    };

    // mongodb has three search fan-outs in one file; brace matching must keep
    // each one's gate to itself. If bodies bled, all three would look gated.
    for func in ["search", "search_mixed", "search_filter_only"] {
        let h = find("mongodb_engine.rs", func);
        assert!(
            h.body.contains("WorkerPool::new("),
            "mongodb_engine.rs::{func} lost its gate"
        );
        assert_eq!(
            h.body.matches("WorkerPool::new(").count(),
            1,
            "mongodb_engine.rs::{func} sees {} pools — its body bled into a neighbouring \
             harness, which would let an ungated fan-out borrow someone else's gate",
            h.body.matches("WorkerPool::new(").count()
        );
    }

    // An upload fan-out must be visible to the census but classified untimed.
    let upload = find("mongodb_engine.rs", "upload_parallel");
    assert!(
        !upload.body.contains("compute_search_stats("),
        "the untimed classification keys off `compute_search_stats`; if an upload path starts \
         publishing search stats it must be treated as timed instead"
    );
}

/// INV-4c — the bare-`StartGate` harnesses hold an abort guard.
///
/// `WorkerPool` aborts its gate in `Drop`, so any early return from the scope
/// closure releases parked workers. A harness driving `StartGate` by hand has no
/// such cover: a coordinator panic between the first `ticket()` and `wait_ready`
/// leaves the workers on a condvar nobody notifies, and whatever owns them (a
/// tokio `Runtime`) then joins them forever — #214's own shape.
#[test]
fn inv4_bare_start_gate_users_hold_an_abort_guard() {
    let mut problems = Vec::new();
    for file in gated_engine_files() {
        let src = read_engine(&file);
        if src.contains("StartGate::new()") && !src.contains("AbortGateOnDrop::new(") {
            problems.push(format!(
                "  {file}: drives a bare `StartGate` without an `AbortGateOnDrop`"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "INV-4c VIOLATED (#214): a hand-driven start gate can be abandoned without releasing the \
         workers parked at it:\n{}",
        problems.join("\n")
    );
}

/// The comment/string stripper must be line-for-line aligned with its input.
///
/// Every guard above reports `path:line` from the STRIPPED text and reads the
/// `INV-4-ALLOW:` opt-out from the RAW line at that index, so the two must agree
/// exactly. A `\`-plus-newline string continuation used to swallow its newline,
/// which shifted everything below it up: violations quoted the wrong source
/// line, a marker on the real line was ignored, and a marker three lines above
/// an unrelated barrier silently exempted it. There are 219 such continuations
/// under `src/`, so this was not a corner case — it was latent only because no
/// `INV-4-ALLOW:` marker exists yet.
#[test]
fn stripper_is_line_for_line_aligned_with_its_input() {
    // A continuation, an escaped quote, a raw string, a lifetime, a char
    // literal, a block comment and a line comment — each on a known line.
    let src = concat!(
        "fn a() {\n",                                // 1
        "    let s = \"one \\\n",                    // 2  <- `\`-continuation
        "         two\";\n",                         // 3
        "    let q = \"he said \\\"hi\\\"\";\n",     // 4
        "    let r = r#\"raw \" not a close\n",      // 5
        "       still raw\"#;\n",                    // 6
        "    /* block\n",                            // 7
        "       comment */\n",                       // 8
        "    let c = '\\n';\n",                      // 9
        "    let g: &'static str = \"x\";\n",        // 10
        "    // Barrier::new(n + 1) in a comment\n", // 11
        "    let ready = Barrier::new(n + 1);\n",    // 12
        "}\n",                                       // 13
    );
    let stripped = strip_comments_and_strings(src);

    assert_eq!(
        stripped.lines().count(),
        src.lines().count(),
        "stripper changed the line count:\n--- raw ---\n{src}\n--- stripped ---\n{stripped}"
    );

    let raw: Vec<&str> = src.lines().collect();
    let out: Vec<&str> = stripped.lines().collect();
    for (i, (r, o)) in raw.iter().zip(out.iter()).enumerate() {
        assert_eq!(
            r.chars().count(),
            o.chars().count(),
            "line {} changed width: {r:?} -> {o:?}",
            i + 1
        );
    }

    // The only surviving `Barrier` must be the code one on line 12, not the
    // comment on line 11 — and it must be reported as line 12.
    let hits: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains("Barrier::"))
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(hits, vec![12], "stripped text:\n{stripped}");
}

/// The same alignment property, asserted over every real source file rather
/// than a fixture — the fixture cannot anticipate the next construct someone
/// writes.
#[test]
fn stripper_preserves_line_count_of_every_source_file() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut checked = 0usize;
    let mut continuations = 0usize;
    for (path, stripped) in all_sources() {
        let raw = fs::read_to_string(root.join(&path)).expect("read source");
        continuations += raw.matches("\\\n").count();
        assert_eq!(
            stripped.lines().count(),
            raw.lines().count(),
            "{path}: stripping changed the line count, so every guard below the \
             first offset line would quote the wrong source line and match \
             `INV-4-ALLOW:` against the wrong one"
        );
        checked += 1;
    }
    assert!(
        checked > 20,
        "expected to scan the whole tree, saw {checked} files"
    );
    assert!(
        continuations > 100,
        "expected the tree to still contain `\\`-continuations (the case this pins); \
         saw {continuations}"
    );
}

/// The `INV-4-ALLOW:` opt-out actually exempts the line it is written on, and
/// only that line — the property the misalignment silently broke.
#[test]
fn inv4_allow_marker_exempts_only_its_own_line() {
    // Reproduces the shape that failed: a `\`-continuation above the barriers.
    // Line 4 carries a TRAILING marker (exempt, and only itself). Line 5 has
    // none and must be reported even though line 4's marker is directly above
    // it. Line 6 is a STANDALONE marker comment, so line 7 is exempt.
    let src = concat!(
        "fn a() {\n",
        "    let msg = \"a long message \\\n",
        "        continued here\";\n",
        "    let x = Barrier::new(n + 1); // INV-4-ALLOW: pinned by test\n",
        "    let y = Barrier::new(n + 1);\n",
        "    // INV-4-ALLOW: annotates the line below\n",
        "    let z = Barrier::new(n + 1);\n",
        "}\n",
    );
    let stripped = strip_comments_and_strings(src);
    let raw: Vec<&str> = src.lines().collect();

    let mut unexempted = Vec::new();
    for (lineno, line) in stripped.lines().enumerate() {
        if !line.contains("Barrier::") {
            continue;
        }
        if line_carries_allow(raw.get(lineno).copied())
            || (lineno > 0 && standalone_allow(raw.get(lineno - 1).copied()))
        {
            continue;
        }
        unexempted.push(lineno + 1);
    }
    assert_eq!(
        unexempted,
        vec![5],
        "line 4 (trailing marker) and line 7 (standalone marker above) must be exempt; \
         line 5 must NOT be exempted by line 4's trailing marker"
    );
}
