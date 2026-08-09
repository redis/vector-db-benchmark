//! Test-harness invariants (issue #292) — regression tripwires for the way the
//! integration suites pick, and guard, the server they run against.
//!
//! These are pure tests: they read `tests/*.rs` as TEXT and exercise the pure
//! `common::claim_verdict` decision function. They need no database and run in
//! the container-free CI job, next to `overhead_invariants.rs`.
//!
//! Background. Four suites (`integration_redis`, `integration_valkey`,
//! `integration_dragonfly`, `integration_kividb`) share one `flush_db()` shape:
//! drop every index `FT._LIST` reports, then `FLUSHALL`. Each one's default port
//! is a container from `tests/docker-compose.test.yml` that several sessions
//! share. Two incidents followed from that:
//!
//!   1. A port override spelled as a `sed` against `const TEST_PORT: u16 = 6399;`
//!      matched nothing (the constant had become `fn test_port()`), exited 0, and
//!      the run proceeded against the shared server with no error and no output
//!      difference.
//!   2. A `docker run -p 6399:6379` failed because the shared container already
//!      held the port, so the writes landed in the shared server instead.
//!
//! Three invariants are locked in:
//!
//!   INV-P1  A suite's default port appears exactly ONCE in its executable
//!           source: the `.unwrap_or(<port>)` inside `fn test_port()`. Any other
//!           code-line occurrence — a reintroduced `const TEST_PORT`, a literal
//!           handed to a spawned binary — is a second source of truth that an
//!           env override cannot move. Comments may mention the port freely.
//!
//!   INV-P2  Every suite that issues `FLUSHALL` claims its instance from inside
//!           `fn test_port()`, via `common::claim_resp_instance`. `test_port()`
//!           is the choke point every path to the server goes through, so a
//!           claim there cannot be bypassed by a helper that forgets to call it.
//!
//!   INV-P3  The FLUSHALL scan is not vacuous: the set of suites it discovers is
//!           pinned, so deleting the guard from a suite fails INV-P2 rather than
//!           quietly shrinking the set INV-P2 checks.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

mod common;

use common::{claim_verdict, info_field, Claim, ClaimInputs, ALLOW_DIRTY_ENV};

// ---------------------------------------------------------------------------
// Source scanning helpers (pure — unit-tested against synthetic input below)
// ---------------------------------------------------------------------------

fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// Every `tests/integration_*.rs`, sorted.
fn integration_sources() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(tests_dir())
        .expect("tests/ must be readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("integration_") && n.ends_with(".rs"))
        })
        .collect();
    out.sort();
    out
}

/// Body of `fn test_port() -> u16 { .. }`, from the opening brace to the first
/// closing brace in column 0.
fn test_port_body(src: &str) -> Option<&str> {
    let start = src.find("fn test_port() -> u16 {")?;
    let rest = &src[start..];
    let end = rest.find("\n}")?;
    Some(&rest[..end])
}

/// The default port a suite falls back to, and the 0-based index of the line it
/// is written on. `None` when the file has no `fn test_port()` fallback.
fn default_port(src: &str) -> Option<(usize, u16)> {
    let body = test_port_body(src)?;
    let at = body.find(".unwrap_or(")?;
    let digits: String = body[at + ".unwrap_or(".len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let port: u16 = digits.parse().ok()?;
    // Absolute line index of that `.unwrap_or(` within the whole file.
    let abs = src.find("fn test_port() -> u16 {")? + at;
    let line = src[..abs].lines().count() - 1;
    Some((line, port))
}

/// True when `hay[at..at + needle_len]` is a standalone number (not a digit of a
/// longer one).
fn is_standalone_number(hay: &str, at: usize, len: usize) -> bool {
    let before_ok = hay[..at]
        .chars()
        .next_back()
        .is_none_or(|c| !c.is_ascii_digit());
    let after_ok = hay[at + len..]
        .chars()
        .next()
        .is_none_or(|c| !c.is_ascii_digit());
    before_ok && after_ok
}

/// Code lines (comment lines excluded) that write `port` as a bare literal,
/// other than the single declaration line `allowed_line`.
///
/// Returns `(1-based line number, line text)` pairs.
fn bare_port_hits(src: &str, port: u16, allowed_line: usize) -> Vec<(usize, String)> {
    let needle = port.to_string();
    let mut hits = Vec::new();
    for (idx, line) in src.lines().enumerate() {
        if idx == allowed_line {
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        let mut from = 0;
        while let Some(rel) = line[from..].find(&needle) {
            let at = from + rel;
            if is_standalone_number(line, at, needle.len()) {
                hits.push((idx + 1, line.to_string()));
                break;
            }
            from = at + 1;
        }
    }
    hits
}

/// Suites whose source issues `FLUSHALL`, by test-target name.
fn flushall_suites() -> BTreeSet<String> {
    integration_sources()
        .into_iter()
        .filter(|p| {
            fs::read_to_string(p)
                .unwrap_or_default()
                .contains("cmd(\"FLUSHALL\")")
        })
        .map(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// INV-P1 — one source of truth for each suite's port
// ---------------------------------------------------------------------------

#[test]
fn inv_p1_suite_ports_have_a_single_source_of_truth() {
    let mut checked: BTreeSet<String> = BTreeSet::new();
    for path in integration_sources() {
        let src = fs::read_to_string(&path).unwrap();
        let Some((line, port)) = default_port(&src) else {
            continue;
        };
        checked.insert(path.file_stem().unwrap().to_str().unwrap().to_string());
        let hits = bare_port_hits(&src, port, line);
        assert!(
            hits.is_empty(),
            "{}: port {} must appear in executable code ONLY on the \
             `.unwrap_or({})` line inside `fn test_port()`; a second literal is a \
             source of truth that the env override cannot move. Offending lines:\n{}",
            path.display(),
            port,
            port,
            hits.iter()
                .map(|(n, l)| format!("  {n}: {}", l.trim()))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    // Not vacuous: these five RESP suites must all be reached by the scan. If a
    // suite stops resolving its port through `fn test_port() -> u16`, the scan
    // would silently skip it — so pin the set instead of counting it.
    for required in [
        "integration_dragonfly",
        "integration_kividb",
        "integration_redis",
        "integration_valkey",
        "integration_vectorsets",
    ] {
        assert!(
            checked.contains(required),
            "{required} must resolve its port through a `fn test_port() -> u16` \
             with a `.unwrap_or(<port>)` fallback so this scan can see it; \
             scanned {checked:?}"
        );
    }
}

#[test]
fn inv_p1_scanner_flags_the_historical_const_spelling() {
    // Positive control: the exact shape whose `sed` override silently no-opped.
    let src = "\
//! Requires redis running on port 6399.
const TEST_PORT: u16 = 6399;
fn test_port() -> u16 {
    std::env::var(\"REDIS_TEST_PORT\")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6399)
}
";
    let (line, port) = default_port(src).expect("fallback must be found");
    assert_eq!(port, 6399);
    let hits = bare_port_hits(src, port, line);
    assert_eq!(
        hits.len(),
        1,
        "scanner must flag the `const TEST_PORT` line, got {hits:?}"
    );
    assert!(hits[0].1.contains("const TEST_PORT"));
}

#[test]
fn inv_p1_scanner_flags_a_literal_handed_to_a_spawned_binary() {
    // Positive control #2: the other spelling of a second source of truth.
    let src = "\
fn test_port() -> u16 {
    std::env::var(\"REDIS_TEST_PORT\")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6399)
}
fn run() {
    cmd.env(\"REDIS_PORT\", \"6399\");
}
";
    let (line, port) = default_port(src).unwrap();
    let hits = bare_port_hits(src, port, line);
    assert_eq!(hits.len(), 1, "got {hits:?}");
    assert!(hits[0].1.contains("REDIS_PORT"));
}

#[test]
fn inv_p1_scanner_accepts_comment_mentions_and_longer_numbers() {
    // Negative control: without it, a guard that rejects everything would pass
    // the two positive controls above.
    let src = "\
//! Requires redis running on port 6399.
//! Run with: REDIS_TEST_PORT=6399 cargo test --test integration_redis
fn test_port() -> u16 {
    std::env::var(\"REDIS_TEST_PORT\")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6399)
}
fn unrelated() -> u32 {
    // 6399 in a trailing comment is fine too
    163990
}
";
    let (line, port) = default_port(src).unwrap();
    assert!(
        bare_port_hits(src, port, line).is_empty(),
        "comment mentions and the digits of 163990 must not be flagged"
    );
}

// ---------------------------------------------------------------------------
// INV-P2 / INV-P3 — destructive suites claim the instance they run against
// ---------------------------------------------------------------------------

#[test]
fn inv_p3_flushall_suite_set_is_pinned() {
    let found = flushall_suites();
    let expected: BTreeSet<String> = [
        "integration_dragonfly",
        "integration_kividb",
        "integration_redis",
        "integration_valkey",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(
        found, expected,
        "the set of suites issuing FLUSHALL changed. A NEW one must call \
         `common::claim_resp_instance` from its `fn test_port()` (see issue #292) \
         and be added here; a REMOVED one means its FLUSHALL is gone."
    );
}

#[test]
fn inv_p2_flushall_suites_claim_their_instance_in_test_port() {
    let suites = flushall_suites();
    assert!(!suites.is_empty(), "FLUSHALL scan found nothing");
    for suite in suites {
        let path = tests_dir().join(format!("{suite}.rs"));
        let src = fs::read_to_string(&path).unwrap();
        let body = test_port_body(&src).unwrap_or_else(|| {
            panic!(
                "{}: a FLUSHALL suite must resolve its port through \
                 `fn test_port() -> u16`",
                path.display()
            )
        });
        assert!(
            body.contains("common::claim_resp_instance("),
            "{}: `fn test_port()` must call `common::claim_resp_instance(..)`. \
             It is the choke point every path to the server goes through, so the \
             ownership check belongs there and nowhere else. Body was:\n{body}",
            path.display(),
        );
    }
}

// ---------------------------------------------------------------------------
// The ownership decision itself
// ---------------------------------------------------------------------------

fn inputs<'a>(
    dbsize: i64,
    index_count: usize,
    prior: Option<&'a str>,
    id: &'a str,
) -> ClaimInputs<'a> {
    ClaimInputs {
        target: "integration_redis",
        port_env: "REDIS_TEST_PORT",
        host: "127.0.0.1",
        port: 6399,
        dbsize,
        index_count,
        prior_claim: prior,
        server_id: id,
        forced: false,
    }
}

#[test]
fn claim_verdict_takes_an_empty_server() {
    // Positive control: the guard must not reject everything. A fresh container
    // — which is what CI starts — has to be usable with no env set.
    assert_eq!(claim_verdict(&inputs(0, 0, None, "abc")), Claim::Fresh);
}

#[test]
fn claim_verdict_refuses_foreign_data() {
    let v = claim_verdict(&inputs(400, 1, None, "abc"));
    assert!(
        matches!(v, Claim::Refuse(_)),
        "a server holding keys and indexes this harness never claimed must be \
         refused, got {v:?}"
    );
}

#[test]
fn claim_verdict_refuses_keys_even_with_no_indexes() {
    // The #286 corpus that nearly got wiped was plain keys; FT._LIST was empty.
    let v = claim_verdict(&inputs(400, 0, None, "abc"));
    assert!(matches!(v, Claim::Refuse(_)), "got {v:?}");
}

#[test]
fn claim_verdict_reuses_an_instance_this_target_dir_already_claimed() {
    // Re-running the suite must work: after a full run the server is left
    // non-empty, and its identity still matches the recorded claim.
    assert_eq!(
        claim_verdict(&inputs(400, 1, Some("abc"), "abc")),
        Claim::Reused
    );
}

#[test]
fn claim_verdict_ignores_a_claim_from_a_different_server() {
    // Same port, different server (the container was replaced and repopulated
    // by someone else): the recorded run_id no longer matches.
    let v = claim_verdict(&inputs(400, 1, Some("abc"), "xyz"));
    assert!(matches!(v, Claim::Refuse(_)), "got {v:?}");
}

#[test]
fn claim_verdict_never_matches_an_unidentifiable_server() {
    // A server that reports no `run_id` yields an empty identity. Two unrelated
    // such servers must not look like the same one.
    let v = claim_verdict(&inputs(400, 1, Some(""), ""));
    assert!(matches!(v, Claim::Refuse(_)), "got {v:?}");
}

#[test]
fn claim_verdict_honours_the_waiver() {
    let mut i = inputs(400, 1, None, "abc");
    i.forced = true;
    assert_eq!(claim_verdict(&i), Claim::Reused);
}

#[test]
fn refusal_message_states_the_mechanism_and_the_way_out() {
    let Claim::Refuse(msg) = claim_verdict(&inputs(400, 2, None, "abc")) else {
        panic!("expected a refusal");
    };
    for expected in [
        "integration_redis", // which suite
        "127.0.0.1:6399",    // which server
        "400 key(s)",        // what is at risk
        "2 search index(es)",
        "FLUSHALL",           // the true mechanism of the damage
        "FT._LIST",           // ..and the index half of it
        "REDIS_TEST_PORT",    // the supported override
        ALLOW_DIRTY_ENV,      // the escape hatch
        "harness_invariants", // why editing the source will not work
    ] {
        assert!(
            msg.contains(expected),
            "refusal message must mention {expected:?}; message was:\n{msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// Server identity parsing (what makes a prior claim match, or not)
// ---------------------------------------------------------------------------

#[test]
fn info_field_reads_run_id_and_master_replid() {
    // Captured live from redis:8.8.0 `INFO server` (abridged).
    let server = "# Server\r\nredis_version:8.8.0\r\ntcp_port:6379\r\n\
                  run_id:e47fa0a6160528b56d01a21bfc4146d716ee02e3\r\nuptime_in_seconds:12\r\n";
    assert_eq!(
        info_field(server, "run_id"),
        "e47fa0a6160528b56d01a21bfc4146d716ee02e3"
    );

    // Dragonfly df-v1.40.1 `INFO server` reports NO run_id, which is why
    // `server_identity` falls back to `master_replid` from `INFO replication`.
    let df_server = "# Server\r\nredis_version:7.4.0\r\ndragonfly_version:df-v1.40.1\r\n\
                     tcp_port:6379\r\nuptime_in_seconds:23\r\n";
    assert_eq!(
        info_field(df_server, "run_id"),
        "",
        "absent field must yield the empty identity, which never matches a claim"
    );
    let df_repl = "# Replication\r\nrole:master\r\n\
                   master_replid:4da04ff40e648670f83234caa27706516c67588b\r\n";
    assert_eq!(
        info_field(df_repl, "master_replid"),
        "4da04ff40e648670f83234caa27706516c67588b"
    );
}
