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
//! Four invariants are locked in:
//!
//!   INV-P1  A suite's default port appears exactly ONCE in its executable
//!           source: the `.unwrap_or(<port>)` inside `fn test_port()`. Any other
//!           code-line occurrence — a reintroduced `const TEST_PORT`, a literal
//!           handed to a spawned binary — is a second source of truth that an
//!           env override cannot move. Comments may mention the port freely.
//!
//!   INV-P2  Every suite that wipes a whole server claims its instance from
//!           inside `fn test_port()`, via `common::claim_resp_instance`, passing
//!           ITS OWN target name and ITS OWN port env var. `test_port()` is the
//!           choke point every path to the server goes through, so a claim there
//!           cannot be bypassed by a helper that forgets to call it.
//!
//!   INV-P3  The wipe scan is not vacuous: the set of suites it discovers is
//!           pinned, so deleting the guard from a suite fails INV-P2 rather than
//!           quietly shrinking the set INV-P2 checks.
//!
//!   INV-P4  The ownership decision fails CLOSED. Every way the probe can fail
//!           must produce a refusal, never a `Fresh`. The first version of this
//!           guard coerced an unreachable server, a denied `DBSIZE` and an
//!           unsupported command all to "0 keys" — i.e. to `Fresh` — which fails
//!           open in the one function whose job is to refuse.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

mod common;

use common::{
    claim_verdict, info_field, sum_keyspace_keys, Claim, ClaimInputs, Probe, ALLOW_DIRTY_ENV,
};

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
/// is written on. Recognises `.unwrap_or(<port>)` and `.unwrap_or_else(|| <port>)`.
/// `None` when the file has no such fallback.
fn default_port(src: &str) -> Option<(usize, u16)> {
    let body = test_port_body(src)?;
    let (at, skip) = [".unwrap_or_else(|| ", ".unwrap_or_else(||", ".unwrap_or("]
        .iter()
        .find_map(|pat| body.find(pat).map(|at| (at, pat.len())))?;
    let digits: String = body[at + skip..]
        .chars()
        .skip_while(|c| *c == ' ')
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let port: u16 = digits.parse().ok()?;
    let abs = src.find("fn test_port() -> u16 {")? + at;
    let line = src[..abs].lines().count() - 1;
    Some((line, port))
}

/// True when `hay[at..at + len]` is a standalone number (not a digit of a longer
/// one).
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

/// Remove Rust's digit separators so `6_399` is seen as `6399`. Only strips `_`
/// that sits BETWEEN two digits, so `foo_6399` and `PORT_6` are untouched.
fn strip_digit_separators(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    for (i, c) in chars.iter().enumerate() {
        let is_separator = *c == '_'
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
        if !is_separator {
            out.push(*c);
        }
    }
    out
}

/// Code lines (comment lines excluded) that write `port` as a bare literal,
/// other than the single declaration line `allowed_line`.
fn bare_port_hits(src: &str, port: u16, allowed_line: usize) -> Vec<(usize, String)> {
    let needle = port.to_string();
    let mut hits = Vec::new();
    for (idx, line) in src.lines().enumerate() {
        if idx == allowed_line {
            continue;
        }
        if line.trim_start().starts_with("//") {
            continue;
        }
        let scanned = strip_digit_separators(line);
        let mut from = 0;
        while let Some(rel) = scanned[from..].find(&needle) {
            let at = from + rel;
            if is_standalone_number(&scanned, at, needle.len()) {
                // Report the ORIGINAL line, not the normalised one.
                hits.push((idx + 1, line.to_string()));
                break;
            }
            from = at + 1;
        }
    }
    hits
}

/// True when this source sends a command that wipes a whole server.
///
/// Matches a QUOTED COMMAND NAME anywhere in the file, CASE-INSENSITIVELY, in
/// either quote style. Two rounds of evasion produced that spec, and each round
/// found a synthetic wiping suite with no claim call at all passing every
/// invariant:
///
///   * spelling:  `cmd("FLUSHDB")`, a rustfmt-wrapped `cmd(\n "FLUSHALL",\n)`,
///     `Cmd::new().arg("FLUSHALL")` — beat the original exact-call match.
///   * case:      `cmd("flushall")`, `cmd("FlushAll")`, `cmd("flushdb")` — beat
///     the widened-but-still-case-sensitive match. Redis command names ARE
///     case-insensitive on the wire (`redis-cli flushall` on redis:8.8.0 takes a
///     server from 2 keys to 0), so these are not hypothetical.
///
/// Single quotes are accepted so a Lua payload — `EVAL "redis.call('flushall')"`
/// — is caught too.
///
/// `FLUSHDB` counts: it wipes db 0 of a shared server, and it is the verb an
/// author reaches for precisely because it sounds safer.
fn wipes_whole_server(src: &str) -> bool {
    let upper = src.to_ascii_uppercase();
    for verb in ["FLUSHALL", "FLUSHDB"] {
        let mut from = 0;
        while let Some(rel) = upper[from..].find(verb) {
            let at = from + rel;
            let quoted_before = matches!(upper[..at].chars().next_back(), Some('"') | Some('\''));
            let quoted_after = matches!(
                upper[at + verb.len()..].chars().next(),
                Some('"') | Some('\'')
            );
            if quoted_before && quoted_after {
                return true;
            }
            from = at + 1;
        }
    }
    false
}

/// Test-target names of the suites that wipe a server.
fn wiping_suites() -> BTreeSet<String> {
    integration_sources()
        .into_iter()
        .filter(|p| wipes_whole_server(&fs::read_to_string(p).unwrap_or_default()))
        .map(|p| p.file_stem().unwrap().to_str().unwrap().to_string())
        .collect()
}

/// The env var a `test_port()` body reads, e.g. `REDIS_TEST_PORT`.
fn port_env_var(body: &str) -> Option<String> {
    let at = body.find("std::env::var(\"")? + "std::env::var(\"".len();
    let rest = &body[at..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Collapse runs of whitespace so a match survives rustfmt rewrapping.
fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
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
             with an `.unwrap_or(<port>)` fallback so this scan can see it; \
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
    assert_eq!(hits.len(), 1, "got {hits:?}");
    assert!(hits[0].1.contains("const TEST_PORT"));
}

#[test]
fn inv_p1_scanner_flags_a_literal_handed_to_a_spawned_binary() {
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
fn inv_p1_scanner_reads_the_unwrap_or_else_spelling_too() {
    // The scanner used to require the literal `.unwrap_or(`, so a suite written
    // with `.unwrap_or_else(|| 6399)` was skipped ENTIRELY — no port checked at
    // all, which is the silent-skip shape this whole file exists to prevent.
    let src = "\
const TEST_PORT: u16 = 6399;
fn test_port() -> u16 {
    std::env::var(\"REDIS_TEST_PORT\")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| 6399)
}
";
    let (line, port) = default_port(src).expect("unwrap_or_else fallback must be found");
    assert_eq!(port, 6399);
    assert_eq!(bare_port_hits(src, port, line).len(), 1);
}

#[test]
fn inv_p1_scanner_sees_through_rust_digit_separators() {
    // `6_399` is the same literal to rustc but a different string to a grep.
    let src = "\
const TEST_PORT: u16 = 6_399;
fn test_port() -> u16 {
    std::env::var(\"REDIS_TEST_PORT\")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6399)
}
";
    let (line, port) = default_port(src).unwrap();
    let hits = bare_port_hits(src, port, line);
    assert_eq!(hits.len(), 1, "got {hits:?}");
    assert!(
        hits[0].1.contains("6_399"),
        "the ORIGINAL line must be reported, not the normalised one: {hits:?}"
    );
    // ..and a separator that is not between two digits must not be collapsed,
    // or `PORT_6399` would start matching.
    assert_eq!(
        strip_digit_separators("let foo_6399 = 1_000;"),
        "let foo_6399 = 1000;"
    );
}

#[test]
fn inv_p1_scanner_accepts_comment_mentions_and_longer_numbers() {
    // Negative control: without it, a guard that rejects everything would pass
    // the positive controls above.
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
// INV-P2 / INV-P3 — server-wiping suites claim the instance they run against
// ---------------------------------------------------------------------------

#[test]
fn inv_p3_wiping_suite_set_is_pinned() {
    let found = wiping_suites();
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
        "the set of suites issuing FLUSHALL/FLUSHDB changed. A NEW one must call \
         `common::claim_resp_instance` from its `fn test_port()` (see issue #292) \
         and be added here; a REMOVED one means its wipe is gone."
    );
}

#[test]
fn inv_p3_scanner_catches_every_spelling_that_evaded_it() {
    // Positive controls: each of these wiped a server while passing the old
    // `cmd("FLUSHALL")` substring scan.
    for (name, src) in [
        ("plain", "redis::cmd(\"FLUSHALL\").query(c)"),
        ("flushdb", "redis::cmd(\"FLUSHDB\").query(c)"),
        ("wrapped", "redis::cmd(\n    \"FLUSHALL\",\n)\n.query(c)"),
        ("builder", "redis::Cmd::new().arg(\"FLUSHALL\").query(c)"),
    ] {
        assert!(
            wipes_whole_server(src),
            "spelling {name:?} must be detected as a whole-server wipe"
        );
    }
}

#[test]
fn inv_p3_scanner_is_case_insensitive() {
    // Second round of evasion. Redis command names are case-insensitive on the
    // wire — `redis-cli flushall` against redis:8.8.0 takes a server from 2 keys
    // to 0 — so a lowercase spelling wipes just as hard while the widened but
    // still case-sensitive matcher passed all 23 invariants with no claim call.
    for (name, src) in [
        ("lower", "redis::cmd(\"flushall\").query(c)"),
        ("mixed", "redis::cmd(\"FlushAll\").query(c)"),
        ("lower-db", "redis::cmd(\"flushdb\").query(c)"),
        // Single quotes: a Lua payload smuggles the same wipe.
        (
            "lua",
            "redis::cmd(\"EVAL\").arg(\"redis.call('flushall')\").query(c)",
        ),
    ] {
        assert!(
            wipes_whole_server(src),
            "spelling {name:?} must be detected as a whole-server wipe"
        );
    }
}

#[test]
fn inv_p5_shared_helpers_must_not_wipe_a_server() {
    // `mod common;` is included by ALL 15 integration suites, non-destructive
    // ones included, and `wiping_suites()` only reads `tests/integration_*.rs`.
    // So a `pub fn wipe_server()` moved into `tests/common/` would give every
    // suite a server-wiping path that INV-P2/P3 cannot see — and that refactor is
    // the LIKELY one, since the premise of this whole guard is that four suites
    // share one `flush_db()` shape.
    let dir = tests_dir().join("common");
    let mut scanned = 0usize;
    for entry in fs::read_dir(&dir).expect("tests/common must be readable") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        scanned += 1;
        let src = fs::read_to_string(&path).unwrap();
        assert!(
            !wipes_whole_server(&src),
            "{}: shared test helpers must not issue FLUSHALL/FLUSHDB. Every suite \
             includes `mod common;`, so a wipe here is invisible to the per-suite \
             scan in INV-P2/P3. Put it in the suite that needs it, where the \
             claim guard can see it.",
            path.display(),
        );
    }
    assert!(scanned > 0, "found no .rs files under {}", dir.display());
}

#[test]
fn inv_p3_scanner_ignores_prose_mentions() {
    // Negative control. `integration_vectorsets.rs` and `integration_valkey.rs`
    // both discuss FLUSHALL in comments; only a quoted command name counts, so
    // vectorsets stays OUT of the pinned set and INV-P3 keeps its meaning.
    let src = "\
// FLUSHALL — the suite shares one server with the other tests, so it must not
// FLUSHDB either.
fn noop() {}
";
    assert!(!wipes_whole_server(src));
}

#[test]
fn inv_p2_wiping_suites_claim_their_instance_in_test_port() {
    let suites = wiping_suites();
    assert!(!suites.is_empty(), "wipe scan found nothing");
    for suite in suites {
        let path = tests_dir().join(format!("{suite}.rs"));
        let src = fs::read_to_string(&path).unwrap();
        let body = test_port_body(&src).unwrap_or_else(|| {
            panic!(
                "{}: a server-wiping suite must resolve its port through \
                 `fn test_port() -> u16`",
                path.display()
            )
        });
        let env = port_env_var(body)
            .unwrap_or_else(|| panic!("{}: `fn test_port()` must read an env var", path.display()));
        // Check the ARGUMENTS, not merely that some call exists: a copy-paste
        // passing another suite's target name or env var would otherwise pass,
        // recording the claim under the wrong key and naming the wrong override
        // in the refusal message.
        let expected =
            format!("common::claim_resp_instance(\"{suite}\", \"{env}\", TEST_HOST, port);");
        assert!(
            squeeze(body).contains(&squeeze(&expected)),
            "{}: `fn test_port()` must call exactly\n    {expected}\n\
             (the choke point every path to the server goes through). Body was:\n{body}",
            path.display(),
        );
    }
}

// ---------------------------------------------------------------------------
// INV-P4 — the decision fails closed
// ---------------------------------------------------------------------------

fn reachable(keys: i64, index_count: usize, id: &str) -> Probe {
    Probe::Reachable {
        keys,
        index_count,
        server_id: id.to_string(),
    }
}

fn inputs<'a>(probe: &'a Probe, prior: Option<&'a str>) -> ClaimInputs<'a> {
    ClaimInputs {
        target: "integration_redis",
        port_env: "REDIS_TEST_PORT",
        host: "127.0.0.1",
        port: 6399,
        probe,
        prior_claim: prior,
        claim_path: "/tmp/target/vdbb-test-claims/integration_redis-127.0.0.1-6399.claim",
        forced: false,
    }
}

#[test]
fn inv_p4_every_inconclusive_probe_refuses() {
    // The real ways the old probe failed OPEN. Each used to become "0 keys",
    // i.e. `Fresh`, i.e. FLUSHALL.
    for why in [
        "could not reach 127.0.0.1:6399 within 30s (connection refused)",
        "127.0.0.1:6399 is still loading its dataset (INFO persistence reports loading:1)",
        "`INFO keyspace` failed on 127.0.0.1:6399 (NOPERM)",
        "`FT._LIST` failed on 127.0.0.1:6399 (unknown command)",
    ] {
        let probe = Probe::Inconclusive(why.to_string());
        let v = claim_verdict(&inputs(&probe, None));
        let Claim::Refuse(msg) = v else {
            panic!("an inconclusive probe must refuse, got {v:?} for {why:?}");
        };
        assert!(
            msg.contains(why),
            "the refusal must quote why the probe was inconclusive: {msg}"
        );
    }
}

#[test]
fn inv_p4_an_unreachable_server_is_not_treated_as_empty() {
    // The specific regression: the guard used to `return` early when the server
    // was still starting, so the suite's own 10-30s `wait_for_*()` then ran the
    // whole destructive suite with NO claim ever recorded.
    let probe = Probe::Inconclusive("could not reach 127.0.0.1:6399 within 30s".to_string());
    assert_ne!(
        claim_verdict(&inputs(&probe, None)),
        Claim::Fresh,
        "an unreachable server must never be classified Fresh"
    );
}

#[test]
fn claim_verdict_takes_an_empty_server() {
    // Positive control: the guard must not reject everything. A fresh container
    // — what CI starts — has to be usable with no env set.
    assert_eq!(
        claim_verdict(&inputs(&reachable(0, 0, "abc"), None)),
        Claim::Fresh
    );
}

#[test]
fn claim_verdict_refuses_foreign_data() {
    let p = reachable(400, 1, "abc");
    assert!(matches!(claim_verdict(&inputs(&p, None)), Claim::Refuse(_)));
}

#[test]
fn claim_verdict_refuses_keys_even_with_no_indexes() {
    // The #286 corpus that nearly got wiped was plain keys; FT._LIST was empty.
    let p = reachable(400, 0, "abc");
    assert!(matches!(claim_verdict(&inputs(&p, None)), Claim::Refuse(_)));
}

#[test]
fn claim_verdict_counts_keys_outside_db0() {
    // `DBSIZE` reported db 0 only while `FLUSHALL` destroys every database, so a
    // developer's Redis with an empty db 0 and Sidekiq/Celery data on db 1 read
    // as "empty". The probe now sums `INFO keyspace`, so this must refuse.
    let keys = sum_keyspace_keys("# Keyspace\r\ndb1:keys=3,expires=0,avg_ttl=0\r\n").unwrap();
    assert_eq!(keys, 3);
    let p = reachable(keys, 0, "abc");
    assert!(matches!(claim_verdict(&inputs(&p, None)), Claim::Refuse(_)));
}

#[test]
fn claim_verdict_reuses_an_instance_this_target_dir_already_claimed() {
    assert_eq!(
        claim_verdict(&inputs(&reachable(400, 1, "abc"), Some("abc"))),
        Claim::Reused
    );
}

#[test]
fn claim_verdict_ignores_a_claim_from_a_different_server() {
    let p = reachable(400, 1, "xyz");
    assert!(matches!(
        claim_verdict(&inputs(&p, Some("abc"))),
        Claim::Refuse(_)
    ));
}

#[test]
fn claim_verdict_never_matches_an_unidentifiable_server() {
    let p = reachable(400, 1, "");
    assert!(matches!(
        claim_verdict(&inputs(&p, Some(""))),
        Claim::Refuse(_)
    ));
}

#[test]
fn claim_verdict_honours_the_waiver() {
    let p = reachable(400, 1, "abc");
    let mut i = inputs(&p, None);
    i.forced = true;
    assert_eq!(claim_verdict(&i), Claim::Reused);
    // ..and it overrides an inconclusive probe too, which is the point of an
    // escape hatch: without that, an operator with an unusual server would have
    // no way to run the suite at all.
    let unreachable = Probe::Inconclusive("could not reach it".to_string());
    let mut i = inputs(&unreachable, None);
    i.forced = true;
    assert_eq!(claim_verdict(&i), Claim::Reused);
}

// ---------------------------------------------------------------------------
// The refusal message — the only thing a person ever sees
// ---------------------------------------------------------------------------

#[test]
fn refusal_message_states_the_mechanism_and_the_way_out() {
    let p = reachable(400, 2, "abc");
    let Claim::Refuse(msg) = claim_verdict(&inputs(&p, None)) else {
        panic!("expected a refusal");
    };
    for expected in [
        "integration_redis", // which suite
        "127.0.0.1:6399",    // which server
        "400 key(s) across all databases",
        "2 search index(es)",
        "FLUSHALL",           // the true mechanism of the damage
        "FT._LIST",           // ..and the index half of it
        "REDIS_TEST_PORT",    // the supported override
        ALLOW_DIRTY_ENV,      // the escape hatch
        "harness_invariants", // why editing the source will not work
        "vdbb-test-claims",   // WHERE the claim was looked for
    ] {
        assert!(
            msg.contains(expected),
            "refusal message must mention {expected:?}; message was:\n{msg}"
        );
    }
    // It must not assert as fact something the guard cannot know: when your own
    // container is restarted, the harness DID create the data it is refusing.
    assert!(
        !msg.contains("did not create"),
        "the message must not claim the harness did not create the data — it \
         only knows there is no matching claim. Message was:\n{msg}"
    );
    assert!(msg.contains("no claim for"));
}

#[test]
fn refusal_message_distinguishes_a_missing_claim_from_a_stale_one() {
    // Two opposite actions for the operator, so the message must not blur them.
    let p = reachable(400, 1, "now-id");

    let Claim::Refuse(missing) = claim_verdict(&inputs(&p, None)) else {
        panic!("expected a refusal");
    };
    assert!(missing.contains("NO claim recorded"), "{missing}");
    assert!(
        missing.contains("cargo clean"),
        "a missing claim should name the ways a claim gets lost: {missing}"
    );

    let Claim::Refuse(stale) = claim_verdict(&inputs(&p, Some("old-id"))) else {
        panic!("expected a refusal");
    };
    assert!(stale.contains("A claim IS recorded"), "{stale}");
    assert!(
        stale.contains("old-id") && stale.contains("now-id"),
        "{stale}"
    );
    assert!(
        stale.contains("docker restart"),
        "a stale claim is most often your OWN restarted container, and the \
         message must say so: {stale}"
    );
}

// ---------------------------------------------------------------------------
// Probe parsing (what the guard reads off the wire)
// ---------------------------------------------------------------------------

#[test]
fn keyspace_sum_covers_every_database_on_all_four_engines() {
    // Captured live from `INFO keyspace` on each guarded engine — the field sets
    // and their order differ, which is why the parser reads `keys=` by name.
    for (engine, info, expected) in [
        (
            "redis:8.8.0",
            "# Keyspace\r\ndb0:keys=1,expires=0,avg_ttl=0,subexpiry=0\r\n",
            1,
        ),
        (
            "valkey-bundle",
            "# Keyspace\r\ndb0:keys=1,expires=0,avg_ttl=0,keys_with_volatile_items=0\r\n",
            1,
        ),
        (
            "dragonfly df-v1.40.1",
            "# Keyspace\r\ndb0:keys=1,expires=0,hits=0,misses=0,hit_ratio=0.00,avg_ttl=-1\r\n",
            1,
        ),
        (
            "kividb v1.0.2-full",
            "# Keyspace\r\ndb0:keys=1,expires=0,avg_ttl=0\r\n",
            1,
        ),
        // The blocker this replaced DBSIZE for: keys parked outside db 0.
        (
            "multi-db",
            "# Keyspace\r\ndb0:keys=2,expires=0\r\ndb1:keys=3,expires=0\r\ndb9:keys=5,expires=0\r\n",
            10,
        ),
        // The two EMPTY-server layouts, both measured on a fresh container. They
        // differ, and both must read as zero — this is the path CI takes on every
        // run, so a parser that only handled one of them would refuse CI.
        // redis:8.8.0 / valkey-bundle / kividb omit the db line entirely:
        ("empty (redis/valkey/kividb)", "# Keyspace\r\n", 0),
        // ..while dragonfly df-v1.40.1 still emits db0 with keys=0:
        (
            "empty (dragonfly)",
            "# Keyspace\r\ndb0:keys=0,expires=0,hits=0,misses=0,hit_ratio=0.00,avg_ttl=-1\r\n",
            0,
        ),
    ] {
        assert_eq!(
            sum_keyspace_keys(info),
            Some(expected),
            "keyspace sum wrong for {engine}"
        );
    }
}

#[test]
fn keyspace_sum_says_unknown_rather_than_zero_when_it_cannot_tell() {
    // The parser returns `i64` no more. Everything below used to collapse to
    // `0` — i.e. to `Fresh`, i.e. to FLUSHALL — which is this guard's own bug
    // class one notch in. All of these now yield `None`, which the probe turns
    // into `Probe::Inconclusive` and `claim_verdict` refuses on.
    for (name, info) in [
        ("empty reply", ""),
        ("no # Keyspace header", "db0:keys=3,expires=0\r\n"),
        ("wrong section", "# Server\r\nrun_id:abc\r\n"),
        (
            "db line with no keys=",
            "# Keyspace\r\ndb0:expires=0,avg_ttl=0\r\n",
        ),
        (
            "unparseable keys=",
            "# Keyspace\r\ndb0:keys=lots,expires=0\r\n",
        ),
        ("negative keys=", "# Keyspace\r\ndb0:keys=-1,expires=0\r\n"),
        (
            "overflow",
            "# Keyspace\r\ndb0:keys=9223372036854775807\r\ndb1:keys=1\r\n",
        ),
    ] {
        assert_eq!(
            sum_keyspace_keys(info),
            None,
            "{name} must be reported as unknown, never as zero keys"
        );
    }
    // Positive control for the header check: the guard must not now reject
    // every real reply.
    assert_eq!(
        sum_keyspace_keys("# Keyspace\r\ndb0:keys=7,expires=0\r\n"),
        Some(7)
    );
}

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

    // `loading:1` is what makes the probe inconclusive on a server replaying its
    // RDB/AOF, whose keyspace is still filling and would otherwise read as 0.
    assert_eq!(
        info_field(
            "# Persistence\r\nloading:1\r\nasync_loading:0\r\n",
            "loading"
        ),
        "1"
    );
    assert_eq!(info_field("# Persistence\r\nloading:0\r\n", "loading"), "0");
}
