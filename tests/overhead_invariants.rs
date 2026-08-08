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
            "turbopuffer",
            "no warm-up gate by design: its workers start on spawn (tracked separately)",
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

/// INV-4b — every engine that fans out a timed search actually routes its start
/// through the gate, once per harness, in the right place.
///
/// Without this, INV-4 could be "satisfied" by an engine that grew some other
/// fixed-count wait, or by one that dropped its synchronized start altogether.
/// The engine list is derived from `engine/mod.rs`, so a new engine is opted in
/// by default.
#[test]
fn inv4_search_harnesses_route_through_the_start_gate() {
    let mut problems = Vec::new();

    for file in gated_engine_files() {
        let src = read_engine(&file);

        // (a) it uses the gate at all.
        let pools = src.matches("WorkerPool::new(").count();
        let bare_gates = src.matches("StartGate::new()").count();
        let harnesses = pools + bare_gates;
        if harnesses == 0 {
            problems.push(format!(
                "  {file}: no `WorkerPool::new` / `StartGate::new` — the synchronized start is \
                 gone, so connection setup and the cold first query are back inside the measured \
                 window"
            ));
            continue;
        }

        // (b) exactly one park per harness. Catches un-gating ONE of the two
        //     harnesses in weaviate (gRPC/GraphQL) or vertex (search/mixed).
        let parks = src.matches("ticket.arrive_and_wait()").count();
        if parks != harnesses {
            problems.push(format!(
                "  {file}: {harnesses} harness(es) but {parks} `ticket.arrive_and_wait()` call(s) \
                 — one of them no longer parks at the gate"
            ));
        }

        // (c) the abort verdict is honoured. `let _ = ticket.arrive_and_wait();`
        //     compiles, satisfies `#[must_use]`, and silently lets every worker
        //     run on its own clock after an aborted start.
        if src.contains("let _ = ticket.arrive_and_wait()") {
            problems.push(format!(
                "  {file}: discards the `arrive_and_wait` verdict with `let _ =`. `None` means \
                 the run was aborted; the worker must return, not measure"
            ));
        }

        // (d) the park sits BELOW setup. Every gated harness reports at least one
        //     setup failure through `ticket.fail(...)` before it parks; if the
        //     first park precedes the first `fail`, the park was hoisted above
        //     client construction / the prime query, putting connection setup
        //     back inside the measured window — the regression (a) is meant to
        //     catch and cannot.
        let (Some(first_park), Some(first_fail)) = (
            src.find("ticket.arrive_and_wait()"),
            src.find("ticket.fail("),
        ) else {
            problems.push(format!(
                "  {file}: no `ticket.fail(...)` — a worker that cannot set itself up is silently \
                 reducing the run's real concurrency again"
            ));
            continue;
        };
        if first_park < first_fail {
            problems.push(format!(
                "  {file}: parks at the gate before its first setup-failure arm, so client \
                 construction and the prime query happen AFTER the start stamp — connection setup \
                 is back inside the measured window"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "INV-4b VIOLATED (#214):\n{}",
        problems.join("\n")
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
