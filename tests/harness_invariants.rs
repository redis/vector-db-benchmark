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
//! Seven invariants are locked in:
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
//!
//!   INV-P5  No shared helper under `tests/common/` issues a server wipe. Every
//!           suite includes `mod common;`, non-destructive ones included, so a
//!           wipe there would be invisible to the per-suite scan in P2/P3.
//!
//!   INV-P6  Each wiping suite's default port equals the host port
//!           `tests/docker-compose.test.yml` publishes for it, and the container
//!           port it hands the claim equals the mapped container port. P1 rejects
//!           a SECOND literal; only this rejects EDITING the single one — the
//!           bypass incident 1 used, and the one the refusal message promises is
//!           rejected.
//!
//!   INV-P7  The set of test targets under tests/ is pinned, so a wiping suite
//!           named outside the `integration_*.rs` pattern cannot slip past the
//!           INV-P3 scan unclassified.

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

/// True when `hay[at..at + len]` is a standalone NUMBER — not a run of digits
/// inside a longer number (`163990`) and not part of an identifier
/// (`foo_6399`, `PORT_6399`, `x6399`).
///
/// The identifier half matters: with only the digit check, `let foo_6399 = 1;`
/// was reported as a bare port literal even though the line contains no literal
/// at all, because `_` is not a digit and so read as a boundary.
fn is_standalone_number(hay: &str, at: usize, len: usize) -> bool {
    let boundary = |c: char| !c.is_ascii_alphanumeric() && c != '_';
    let before_ok = hay[..at].chars().next_back().is_none_or(boundary);
    let after_ok = hay[at + len..].chars().next().is_none_or(boundary);
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
    // `\` counts as a quote boundary so an ESCAPED literal — `\"FLUSHALL\"`, the
    // natural Rust spelling of a Lua payload or a printed message — is caught.
    // Without it the "either quote style" claim was false.
    let is_quote = |c: char| c == '"' || c == '\'' || c == '\\';
    let upper = src.to_ascii_uppercase();
    for verb in ["FLUSHALL", "FLUSHDB"] {
        let mut from = 0;
        while let Some(rel) = upper[from..].find(verb) {
            let at = from + rel;
            let quoted_before = upper[..at].chars().next_back().is_some_and(is_quote);
            let quoted_after = upper[at + verb.len()..]
                .chars()
                .next()
                .is_some_and(is_quote);
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

/// The compose service each wiping suite is meant to talk to.
const SUITE_COMPOSE_SERVICE: &[(&str, &str)] = &[
    ("integration_dragonfly", "dragonfly"),
    ("integration_kividb", "kividb"),
    ("integration_redis", "redis"),
    ("integration_valkey", "valkey"),
];

/// `(host_port, container_port)` from the first published port of a
/// `tests/docker-compose.test.yml` service, e.g. `- "6386:6380"` -> `(6386, 6380)`.
fn compose_ports(service: &str) -> Option<(u16, u16)> {
    let compose = fs::read_to_string(tests_dir().join("docker-compose.test.yml")).ok()?;
    let mut in_service = false;
    let mut in_ports = false;
    for line in compose.lines() {
        // A service header is exactly two spaces of indent, e.g. `  kividb:`.
        if let Some(name) = line.strip_prefix("  ").and_then(|l| l.strip_suffix(':')) {
            if !name.starts_with(' ') {
                in_service = name == service;
                in_ports = false;
                continue;
            }
        }
        if !in_service {
            continue;
        }
        let trimmed = line.trim();
        if trimmed == "ports:" {
            in_ports = true;
            continue;
        }
        if in_ports {
            if let Some(entry) = trimmed.strip_prefix("- ") {
                let (host, container) = entry.trim().trim_matches('"').split_once(':')?;
                return Some((host.parse().ok()?, container.parse().ok()?));
            }
            if !trimmed.starts_with('#') && !trimmed.is_empty() {
                in_ports = false;
            }
        }
    }
    None
}

/// The env var a `test_port()` body reads, e.g. `REDIS_TEST_PORT`.
fn port_env_var(body: &str) -> Option<String> {
    let at = body.find("std::env::var(\"")? + "std::env::var(\"".len();
    let rest = &body[at..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Normalise a call so the match survives rustfmt: drop all whitespace (it may
/// rewrap the arguments one per line) and the trailing comma it then adds.
fn squeeze(s: &str) -> String {
    let no_ws: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    no_ws.replace(",)", ")")
}

/// Remove `//` and `/* */` comments, leaving string literals intact so
/// `"redis://host"` survives.
///
/// INV-P2 used to match against RAW source, so **commenting the claim call out**
/// — one `/` character — left all invariants green while the suite went on to
/// `FLUSHALL` a server it had not claimed. `bare_port_hits` had skipped comment
/// lines since the first round; INV-P2 was the same file, one dimension short.
fn strip_comments(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            out.push(c);
            i += 1;
            while i < chars.len() {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    out.push(chars[i]);
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                out.push(chars[i]);
                i += 1;
                if chars[i - 1] == '"' {
                    break;
                }
            }
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// The one shape a wiping suite's `fn test_port()` is allowed to have.
///
/// INV-P2 asserts the WHOLE body equals this, not merely that it contains the
/// claim call. Containment could be satisfied while the call was disabled —
/// commented out, wrapped in `if false { .. }`, or parked inside
/// `stringify!(..)`. Equality means any extra or missing token fails, so the
/// choke point cannot be neutered without the build going red. It also makes the
/// rustfmt tolerance in `squeeze` load-bearing: an identity `squeeze` no longer
/// matches.
fn expected_test_port_body(suite: &str, env: &str, default: u16, container: u16) -> String {
    format!(
        "fn test_port() -> u16 {{
            static PORT: std::sync::OnceLock<u16> = std::sync::OnceLock::new();
            *PORT.get_or_init(|| {{
                let port = std::env::var(\"{env}\")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or({default});
                common::claim_resp_instance(\"{suite}\", \"{env}\", TEST_HOST, port, {container});
                port
            }})"
    )
}

/// Every `.rs` file under `tests/common/`, at ANY depth.
///
/// `read_dir` reads one level, so INV-P5 — round 3's headline fix — was evaded in
/// its own idiom by putting the wipe helper in `tests/common/wipe/mod.rs`.
fn common_sources() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(&tests_dir().join("common"), &mut out);
    out.sort();
    out
}

/// Body of a top-level `fn <name>(..)` in `tests/common/mod.rs`, comments kept.
fn common_fn_body(name: &str) -> String {
    let src = fs::read_to_string(tests_dir().join("common").join("mod.rs")).unwrap();
    let marker = format!("fn {name}(");
    let start = src
        .find(&marker)
        .unwrap_or_else(|| panic!("tests/common/mod.rs must define `fn {name}`"));
    let rest = &src[start..];
    let end = rest.find("\n}").unwrap_or(rest.len());
    rest[..end].to_string()
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
    // The stripper only joins digits across a separator that sits BETWEEN two
    // digits, so an identifier keeps its underscore. (Whether `foo_6399` is
    // FLAGGED is a separate question, settled by `is_standalone_number` and
    // covered by the next test.)
    assert_eq!(
        strip_digit_separators("let foo_6399 = 1_000;"),
        "let foo_6399 = 1000;"
    );
}

#[test]
fn inv_p1_scanner_does_not_flag_identifiers_that_merely_end_in_the_port() {
    // Negative control with teeth: none of these lines contains a port literal,
    // and an earlier `is_standalone_number` flagged the first two — it treated
    // `_` as a boundary, so `foo_6399` read as the bare number 6399.
    let src = "\
fn test_port() -> u16 {
    std::env::var(\"REDIS_TEST_PORT\")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6399)
}
fn unrelated() {
    let foo_6399 = 1;
    let port_6399 = 2;
    let x6399y = 3;
}
";
    let (line, port) = default_port(src).unwrap();
    let hits = bare_port_hits(src, port, line);
    assert!(
        hits.is_empty(),
        "identifiers ending in the port number are not port literals: {hits:?}"
    );
    // ..and the real thing on the very same shape of line is still caught, so
    // the precision fix did not cost the scanner its teeth.
    let real = src.replace("let foo_6399 = 1;", "cmd.env(\"REDIS_PORT\", \"6399\");");
    let (line, port) = default_port(&real).unwrap();
    assert_eq!(bare_port_hits(&real, port, line).len(), 1);
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
        // Escaped double quotes — the natural Rust spelling, and the one that
        // beat the "either quote style" matcher.
        (
            "escaped",
            "redis::cmd(\"EVAL\").arg(\"return redis.call(\\\"FLUSHALL\\\")\").query(c)",
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
    let sources = common_sources();
    for path in &sources {
        let src = fs::read_to_string(path).unwrap();
        assert!(
            !wipes_whole_server(&src),
            "{}: shared test helpers must not issue FLUSHALL/FLUSHDB. Every suite \
             includes `mod common;`, so a wipe here is invisible to the per-suite \
             scan in INV-P2/P3. Put it in the suite that needs it, where the \
             claim guard can see it.",
            path.display(),
        );
    }
    // Pin the file set, not just `> 0`. The first version used `read_dir`, which
    // reads ONE level, so a helper at `tests/common/wipe/mod.rs` evaded the check
    // entirely — this invariant re-evaded in its own idiom. `common_sources()`
    // now recurses, and pinning the names means a new shared file has to be
    // acknowledged here rather than silently joining (or dodging) the scan.
    let names: BTreeSet<String> = sources
        .iter()
        .map(|p| {
            p.strip_prefix(tests_dir().join("common"))
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    let expected: BTreeSet<String> = ["mod.rs"].iter().map(|s| s.to_string()).collect();
    assert_eq!(
        names, expected,
        "the set of shared helper files under tests/common/ changed; add it here \
         so the wipe scan provably covers it"
    );
}

#[test]
fn inv_p5_scanner_recurses() {
    // Positive control for the walker: the evasion was depth, not content, so
    // the fix has to be demonstrated on depth.
    let nested = tests_dir().join("common").join("mod.rs");
    assert!(
        common_sources().contains(&nested),
        "the walker must find tests/common/mod.rs: {:?}",
        common_sources()
    );
    // ..and it must be a real walk, not a hardcoded list: point it at a tree it
    // has never seen and it must find the nested file.
    let tmp = std::env::temp_dir().join(format!("vdbb-p5-{}", std::process::id()));
    let deep = tmp.join("a").join("b");
    fs::create_dir_all(&deep).unwrap();
    fs::write(deep.join("wipe.rs"), "redis::cmd(\"FLUSHALL\")").unwrap();
    fs::write(tmp.join("top.rs"), "fn x() {}").unwrap();
    let mut found: Vec<String> = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<String>) {
        for e in fs::read_dir(dir).unwrap().filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                out.push(p.file_name().unwrap().to_string_lossy().into_owned());
            }
        }
    }
    walk(&tmp, &mut found);
    found.sort();
    fs::remove_dir_all(&tmp).ok();
    assert_eq!(found, vec!["top.rs".to_string(), "wipe.rs".to_string()]);
}

#[test]
fn inv_p7_the_scanned_test_target_set_is_pinned() {
    // The wipe scan only sees `tests/integration_*.rs`, so a wiping suite named
    // `tests/zz_wipe_smoke.rs` would never be classified at all. Pinning the
    // target list means a new one must be acknowledged here.
    let mut found: Vec<String> = fs::read_dir(tests_dir())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("rs"))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    found.sort();
    let expected = vec![
        "harness_invariants.rs",
        "integration_chroma.rs",
        "integration_cli.rs",
        "integration_dragonfly.rs",
        "integration_elasticsearch.rs",
        "integration_kividb.rs",
        "integration_milvus.rs",
        "integration_mongodb.rs",
        "integration_opensearch.rs",
        "integration_pgvector.rs",
        "integration_qdrant.rs",
        "integration_redis.rs",
        "integration_valkey.rs",
        "integration_vectorsets.rs",
        "integration_vertex.rs",
        "integration_weaviate.rs",
        "overhead_invariants.rs",
    ];
    assert_eq!(
        found, expected,
        "the set of test targets changed. A new one that wipes a server must be \
         named `integration_*.rs` so the INV-P3 scan sees it, and listed here."
    );
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
        // Compare the WHOLE body, comments stripped, against the one allowed
        // shape. Containment was not enough: commenting the claim line out left
        // every invariant green while the suite went on to wipe an unclaimed
        // server, and `if false { .. }` / `stringify!(..)` do the same. Equality
        // also pins the arguments — a copy-paste carrying another suite's target
        // name or env var would record the claim under the wrong key and name
        // the wrong override in the refusal — and the container port, which the
        // refusal prints in its `docker run -p X:<it>` line (kividb's is 6380).
        let (_line, default) = default_port(&src)
            .unwrap_or_else(|| panic!("{}: no `test_port()` fallback", path.display()));
        let service = SUITE_COMPOSE_SERVICE
            .iter()
            .find(|(s, _)| *s == suite)
            .map(|(_, svc)| *svc)
            .unwrap_or_else(|| panic!("{suite} must be listed in SUITE_COMPOSE_SERVICE"));
        let (_host_port, container_port) = compose_ports(service)
            .unwrap_or_else(|| panic!("no published port for compose service {service}"));
        let expected = expected_test_port_body(&suite, &env, default, container_port);
        assert_eq!(
            squeeze(&strip_comments(body)),
            squeeze(&expected),
            "{}: `fn test_port()` must be EXACTLY this shape (whitespace and \
             comments aside) — it is the choke point every path to the server \
             goes through, so nothing may be added, removed or disabled:\n\
             {expected}\n\nfound:\n{body}",
            path.display(),
        );
    }
}

#[test]
fn inv_p2_rejects_a_commented_out_or_disabled_claim() {
    // The evasions equality closes, each proved to wipe a real server while the
    // containment check stayed green.
    let (suite, env, default, container) =
        ("integration_redis", "REDIS_TEST_PORT", 6399u16, 6379u16);
    let good = expected_test_port_body(suite, env, default, container);
    assert_eq!(
        squeeze(&strip_comments(&good)),
        squeeze(&expected_test_port_body(suite, env, default, container)),
        "positive control: the canonical body must match itself"
    );
    for (name, mutant) in [
        ("commented out", good.replace("common::claim", "// common::claim")),
        (
            "wrapped in if false",
            good.replace(
                "common::claim_resp_instance(",
                "if false { common::claim_resp_instance(",
            ) + " }",
        ),
        (
            "parked in stringify!",
            good.replace("common::claim_resp_instance(", "stringify!(common::claim_resp_instance("),
        ),
        ("deleted", good.replace("common::claim_resp_instance(\"integration_redis\", \"REDIS_TEST_PORT\", TEST_HOST, port, 6379);", "")),
    ] {
        assert_ne!(
            squeeze(&strip_comments(&mutant)),
            squeeze(&good),
            "a {name} claim must not satisfy INV-P2"
        );
    }
}

#[test]
fn comment_stripper_keeps_string_literals() {
    // Negative control: without this the stripper would eat the `//` in every
    // `redis://` URL and INV-P2 would compare mangled text against mangled text.
    let src = "let url = format!(\"redis://{}:{}/\", h, p); // trailing\n/* block */ let x = 1;";
    let out = strip_comments(src);
    assert!(out.contains("\"redis://{}:{}/\""), "{out}");
    assert!(!out.contains("trailing"), "{out}");
    assert!(!out.contains("block"), "{out}");
    assert!(out.contains("let x = 1;"), "{out}");
}

#[test]
fn inv_p6_suite_defaults_match_the_compose_mapping() {
    // INV-P1 rejects a SECOND literal for the port; on its own it does not
    // reject EDITING the single one — `default_port()` re-reads whatever is on
    // the `.unwrap_or(..)` line, so changing 6399 to 7911 left all invariants
    // green. That is precisely the bypass incident 1 in #292 used, and the
    // refusal message tells operators it is rejected. This is what makes that
    // sentence true: the default is pinned to the port
    // `tests/docker-compose.test.yml` publishes, so an edit fails the build, and
    // the suite default and the compose mapping cannot drift apart either way.
    for (suite, service) in SUITE_COMPOSE_SERVICE {
        let src = fs::read_to_string(tests_dir().join(format!("{suite}.rs"))).unwrap();
        let (_line, default) = default_port(&src)
            .unwrap_or_else(|| panic!("{suite}: no `fn test_port()` fallback to check"));
        let (host_port, _container) = compose_ports(service)
            .unwrap_or_else(|| panic!("no published port for compose service {service}"));
        assert_eq!(
            default, host_port,
            "{suite}'s default port ({default}) must equal the host port \
             tests/docker-compose.test.yml publishes for `{service}` ({host_port}). \
             To run against your own server use the suite's env var — editing this \
             literal moves the default for everyone."
        );
    }
}

#[test]
fn inv_p6_compose_parser_reads_a_non_6379_container_port() {
    // Positive control for the parser, and the case that motivated it: kividb
    // does NOT listen on 6379.
    assert_eq!(compose_ports("kividb"), Some((6386, 6380)));
    assert_eq!(compose_ports("redis"), Some((6399, 6379)));
    // Negative control: an absent service must be reported as absent, not as
    // some default that would make INV-P6 pass vacuously.
    assert_eq!(compose_ports("no-such-service"), None);
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
        container_port: 6379,
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
fn inv_p4_probe_server_reports_a_closed_port_as_inconclusive() {
    // INV-P4 only ever exercised `claim_verdict`, the CONSUMER. Everything
    // between it and the wire was untested, so a `probe_server` returning
    // `Reachable { keys: 0, index_count: 0 }` for an unreachable server — the
    // exact regression INV-P4's docstring cites — passed every invariant. A
    // zero wait keeps this a pure, fast test.
    let port = closed_port();
    let probe = common::probe_server("127.0.0.1", port, std::time::Duration::ZERO);
    assert!(
        matches!(probe, Probe::Inconclusive(_)),
        "a closed port must probe as Inconclusive, got {probe:?}"
    );
    // ..and it must flow through to a refusal, not merely be Inconclusive.
    assert!(matches!(
        claim_verdict(&inputs(&probe, None)),
        Claim::Refuse(_)
    ));
}

/// A TCP port with nothing listening: bind one, read the number, drop it.
fn closed_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

#[test]
fn inv_p4_the_producer_side_keeps_its_shape() {
    // Source pins for the three mutations that make the guard INERT and that no
    // integration suite can catch — a permissive mutation makes every suite pass
    // harder, not fail.
    let probe = common_fn_body("probe_server");
    let inconclusive = probe.matches("Probe::Inconclusive(").count();
    let reachable = probe.matches("Probe::Reachable {").count();
    assert!(
        inconclusive >= 5,
        "`probe_server` must refuse on every failure path (unreachable, loading, \
         INFO keyspace error, unparseable keyspace, FT._LIST error); found only \
         {inconclusive} `Probe::Inconclusive(` sites"
    );
    assert_eq!(
        reachable, 1,
        "`Probe::Reachable` must be constructed at exactly one place in \
         `probe_server`, so it cannot be reached before both probes succeeded"
    );
    let last_reachable = probe.rfind("Probe::Reachable {").unwrap();
    let last_inconclusive = probe.rfind("Probe::Inconclusive(").unwrap();
    assert!(
        last_reachable > last_inconclusive,
        "the single `Probe::Reachable` must be the FINAL statement, after every \
         inconclusive early return"
    );

    // The refusal must actually stop the test.
    let claim = common_fn_body("claim_resp_instance");
    assert!(
        claim.contains("panic!(\"{msg}\")"),
        "`claim_resp_instance` must panic on a refusal; swallowing it makes the \
         whole guard inert. Body was:\n{claim}"
    );

    // The waiver must default to OFF: `unwrap_or(true)` would disable the guard
    // for everyone while every suite kept passing.
    let outcome = common_fn_body("claim_outcome");
    assert!(
        outcome.contains("std::env::var(ALLOW_DIRTY_ENV)") && outcome.contains(".unwrap_or(false)"),
        "the waiver must be opt-IN — an absent {ALLOW_DIRTY_ENV} has to mean the \
         guard is ON. Body was:\n{outcome}"
    );
}

#[test]
fn inv_p4_claim_wait_covers_every_suite_wait_for() {
    // A documented relation with no guard until now: if the claim gave up sooner
    // than a suite's own `wait_for_*()`, the startup race that caused BLOCKER 1
    // would reopen — the probe would quit, the suite's wait would succeed, and
    // the `OnceLock` would never retry.
    let common_src = fs::read_to_string(tests_dir().join("common").join("mod.rs")).unwrap();
    let claim_wait: u64 = common_src
        .lines()
        .find_map(|l| {
            l.contains("const CLAIM_REACHABLE_WAIT")
                .then(|| {
                    l.split("from_secs(")
                        .nth(1)?
                        .split(')')
                        .next()?
                        .parse()
                        .ok()
                })
                .flatten()
        })
        .expect("CLAIM_REACHABLE_WAIT must be a literal `from_secs(N)`");
    let mut worst = 0u64;
    for (suite, _) in SUITE_COMPOSE_SERVICE {
        let src = fs::read_to_string(tests_dir().join(format!("{suite}.rs"))).unwrap();
        for line in strip_comments(&src).lines() {
            if !line.contains("Instant::now() + Duration::from_secs(") {
                continue;
            }
            if let Some(n) = line
                .split("from_secs(")
                .nth(1)
                .and_then(|r| r.split(')').next())
                .and_then(|n| n.parse::<u64>().ok())
            {
                worst = worst.max(n);
            }
        }
    }
    assert!(
        worst > 0,
        "found no `wait_for_*()` deadline to compare against"
    );
    assert!(
        claim_wait >= worst,
        "CLAIM_REACHABLE_WAIT ({claim_wait}s) must be >= the longest \
         `wait_for_*()` deadline in the guarded suites ({worst}s), or a slow \
         container start races past the claim"
    );
}

#[test]
fn claim_verdict_refuses_indexes_even_with_no_keys() {
    // The index half of the emptiness test was unpinned: `Fresh` whenever
    // `keys == 0` would have passed everything, even though `flush_db()` drops
    // every index `FT._LIST` reports and the refusal counts them.
    let p = reachable(0, 1, "abc");
    assert!(
        matches!(claim_verdict(&inputs(&p, None)), Claim::Refuse(_)),
        "a server with no keys but a live search index must still be refused"
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
        "FLUSHALL",                // the true mechanism of the damage
        "FT._LIST",                // ..and the index half of it
        "REDIS_TEST_PORT",         // the supported override
        ALLOW_DIRTY_ENV,           // the escape hatch
        "harness_invariants",      // why editing the source will not work
        "docker-compose.test.yml", // ..and the mechanism that makes that true
        "vdbb-test-claims",        // WHERE the claim was looked for
        "-p <your-port>:6379",     // a runnable remedy, with the RIGHT port
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

    // The message must not claim a policing power the invariants do not have.
    // The earlier wording — "Editing the port literal in tests/X.rs is rejected
    // by tests/harness_invariants.rs" — was false: INV-P1 rejects a SECOND
    // literal, and editing the single one left all invariants green. That is now
    // true only because INV-P6 pins the default to the compose mapping, so the
    // message has to cite THAT, and this asserts it does. Naming the file alone
    // is what let the false sentence ship.
    assert!(
        msg.contains("pins that default to the"),
        "the message must name the mechanism that actually rejects an edit \
         (INV-P6's compose pin), not just the file: {msg}"
    );

    // The remedy has to be runnable for the suite it is printed for. kividb
    // listens on 6380, so a hardcoded 6379 would hand the operator a container
    // the suite cannot reach — and then a second refusal from this guard.
    let mut kividb = inputs(&p, None);
    kividb.target = "integration_kividb";
    kividb.port_env = "KIVIDB_PORT";
    kividb.container_port = 6380;
    let Claim::Refuse(kv) = claim_verdict(&kividb) else {
        panic!("expected a refusal");
    };
    assert!(
        kv.contains("-p <your-port>:6380"),
        "kividb's remedy must publish to 6380, not 6379: {kv}"
    );
    assert!(!kv.contains("-p <your-port>:6379"), "{kv}");
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
