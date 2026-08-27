//! Shared utilities for Redis-protocol engines (Redis, Valkey, VectorSets).
//!
//! Provides `INFO COMMANDSTATS` validation to detect silent command failures,
//! and `collect_server_metadata` to snapshot server version / loaded modules /
//! a full `INFO` dump for result reproducibility.

use redis::Connection;
use serde_json::{Map, Value as Json};
use std::collections::HashMap;
use vector_db_benchmark::parsers::parse_info;

/// Collect reproducibility metadata from a Redis-wire server (Redis, Valkey,
/// VectorSets, Dragonfly). Returns
/// `{ "versions": {...}, "modules": {...}|null, "info": { section -> kv } }`.
///
/// This is telemetry only: it never panics and, if a command fails, returns
/// whatever succeeded (e.g. `modules: null` when `MODULE LIST` is denied).
///
/// - `INFO everything` is tried first (richer); when the reply is empty/blank —
///   as on Dragonfly, which returns nothing for `everything` — it falls back to
///   plain `INFO`, which works on every server.
/// - `MODULE LIST` is parsed into `name -> { ver, path, args }`, handling both
///   the RESP2 (flat array of k/v) and RESP3 (map) module shapes.
pub fn collect_server_metadata(conn: &mut Connection) -> Json {
    // INFO everything (rich); fall back to plain INFO when empty (Dragonfly).
    let everything: String = redis::cmd("INFO")
        .arg("everything")
        .query::<String>(conn)
        .unwrap_or_default();
    let info_str = if info_reply_empty(&everything) {
        redis::cmd("INFO").query::<String>(conn).unwrap_or_default()
    } else {
        everything
    };
    let info = parse_info(&info_str);
    let versions = extract_versions(&info);

    let modules = match redis::cmd("MODULE").arg("LIST").query::<redis::Value>(conn) {
        Ok(val) => parse_modules(&val),
        Err(_) => Json::Null,
    };

    // Full server config (includes module config: Redis `search-*`, Dragonfly
    // `search.*`, Valkey's unified search config). Secret-bearing values are
    // redacted. On error, `config: null` (never fails the whole collection).
    let config = match redis::cmd("CONFIG")
        .arg("GET")
        .arg("*")
        .query::<redis::Value>(conn)
    {
        Ok(val) => redact_config(parse_kv_map(&val)),
        Err(_) => Json::Null,
    };

    // SEARCH module's own config. Works on Redis / Dragonfly; Valkey lacks
    // FT.CONFIG (ERR unknown command) → `search_config: null`.
    let search_config = match redis::cmd("FT.CONFIG")
        .arg("GET")
        .arg("*")
        .query::<redis::Value>(conn)
    {
        Ok(val) => parse_kv_map(&val),
        Err(_) => Json::Null,
    };

    serde_json::json!({
        "versions": versions,
        "modules": modules,
        "info": info,
        "config": config,
        "search_config": search_config,
    })
}

/// Parse a config reply (`CONFIG GET *`, `FT.CONFIG GET *`) into a
/// `{ key -> value }` object. Handles all three shapes seen in the wild:
/// - RESP3 map (`{ k: v }`) — e.g. redis-rs' `CONFIG GET`;
/// - RESP2 flat array (`[k, v, k, v, ...]`) — plain `CONFIG GET`;
/// - array of `[k, v]` pair-arrays / maps — e.g. Redis' `FT.CONFIG GET *`.
///
/// The pair-array vs flat shape is detected from the first element (nested ⇒
/// pair-arrays). Values render via `value_to_json`.
fn parse_kv_map(val: &redis::Value) -> Json {
    let mut out = Map::new();
    match val {
        redis::Value::Map(pairs) => {
            for (k, v) in pairs {
                out.insert(value_to_string(k), value_to_json(v));
            }
        }
        redis::Value::Array(items) => {
            let nested = matches!(
                items.first(),
                Some(redis::Value::Array(_)) | Some(redis::Value::Map(_))
            );
            if nested {
                // Array of [key, value] pair-arrays (or per-entry maps).
                for item in items {
                    match item {
                        redis::Value::Array(kv) if kv.len() >= 2 => {
                            out.insert(value_to_string(&kv[0]), value_to_json(&kv[1]));
                        }
                        redis::Value::Map(pairs) => {
                            for (k, v) in pairs {
                                out.insert(value_to_string(k), value_to_json(v));
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                // Flat [k, v, k, v, ...] array.
                for c in items.chunks(2) {
                    if let [k, v] = c {
                        out.insert(value_to_string(k), value_to_json(v));
                    }
                }
            }
        }
        _ => {}
    }
    Json::Object(out)
}

/// Redact secret-bearing config values in place: any key whose lowercased name
/// contains `pass`, `auth`, `secret`, or `token` keeps its key but has its value
/// replaced with `"<redacted>"` (e.g. `requirepass`, `masterauth`,
/// `tls-key-file-pass`). Non-secret keys (paths like `aclfile`) are untouched.
fn redact_config(config: Json) -> Json {
    let Json::Object(mut map) = config else {
        return config;
    };
    for (key, val) in map.iter_mut() {
        let lower = key.to_ascii_lowercase();
        if ["pass", "auth", "secret", "token"]
            .iter()
            .any(|needle| lower.contains(needle))
        {
            *val = Json::String("<redacted>".to_string());
        }
    }
    Json::Object(map)
}

/// True when an `INFO` reply carries no content (empty or whitespace-only),
/// signalling that a fallback command should be issued.
fn info_reply_empty(s: &str) -> bool {
    s.trim().is_empty()
}

/// Pull the server-version keys present in the `Server` section into a compact
/// `versions` object (`redis_version`, `valkey_version`, `dragonfly_version` —
/// whichever the server reports).
fn extract_versions(info: &Json) -> Json {
    let mut out = Map::new();
    if let Some(server) = info.get("Server").and_then(|s| s.as_object()) {
        for key in ["redis_version", "valkey_version", "dragonfly_version"] {
            if let Some(val) = server.get(key) {
                out.insert(key.to_string(), val.clone());
            }
        }
    }
    Json::Object(out)
}

/// Parse a `MODULE LIST` reply into `{ "<name>": { "ver": .., "path": .., "args": .. } }`.
/// Each module entry is a k/v collection in either RESP2 (flat array) or RESP3
/// (map) form. The `name` field becomes the map key; remaining fields become
/// the value object. A non-array reply yields an empty object.
fn parse_modules(val: &redis::Value) -> Json {
    let items = match val {
        redis::Value::Array(items) => items,
        _ => return Json::Object(Map::new()),
    };
    let mut out = Map::new();
    for m in items {
        let mut name: Option<String> = None;
        let mut entry = Map::new();
        for (k, v) in module_fields(m) {
            if k == "name" {
                name = Some(value_to_string(&v));
            } else {
                entry.insert(k, value_to_json(&v));
            }
        }
        if let Some(n) = name {
            out.insert(n, Json::Object(entry));
        }
    }
    Json::Object(out)
}

/// Flatten a single module entry into `(field, value)` pairs, accepting both the
/// RESP3 map shape and the RESP2 flat-array-of-k/v shape.
fn module_fields(m: &redis::Value) -> Vec<(String, redis::Value)> {
    match m {
        redis::Value::Map(pairs) => pairs
            .iter()
            .map(|(k, v)| (value_to_string(k), v.clone()))
            .collect(),
        redis::Value::Array(items) => items
            .chunks(2)
            .filter_map(|c| match c {
                [k, v] => Some((value_to_string(k), v.clone())),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Render a `redis::Value` as a plain string (used for module names / keys).
fn value_to_string(v: &redis::Value) -> String {
    match v {
        redis::Value::SimpleString(s) => s.clone(),
        redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
        redis::Value::Int(n) => n.to_string(),
        // RESP3 returns FT.INFO numeric size fields (`*_mb`) as Double; render the
        // bare number so `ft_info_index_memory_bytes` can parse it (else per-config
        // memory silently degrades to null under `*_PROTOCOL=resp3`).
        redis::Value::Double(d) => d.to_string(),
        other => format!("{:?}", other),
    }
}

/// Convert a `redis::Value` into `serde_json::Value` (module field values —
/// `ver` stays an integer, `path`/`args` become strings). Public so engines
/// without their own converter (e.g. VectorSets) can render `VINFO`/`FT.INFO`.
pub fn value_to_json(v: &redis::Value) -> Json {
    match v {
        redis::Value::Nil => Json::Null,
        redis::Value::Int(n) => serde_json::json!(n),
        redis::Value::Double(f) => serde_json::json!(f),
        redis::Value::Boolean(b) => serde_json::json!(b),
        redis::Value::SimpleString(s) => serde_json::json!(s),
        redis::Value::BulkString(b) => match String::from_utf8(b.clone()) {
            Ok(s) => serde_json::json!(s),
            Err(_) => serde_json::json!(format!("<{} bytes>", b.len())),
        },
        redis::Value::Array(a) => Json::Array(a.iter().map(value_to_json).collect()),
        redis::Value::Map(pairs) => {
            let mut map = Map::new();
            for (k, val) in pairs {
                map.insert(value_to_string(k), value_to_json(val));
            }
            Json::Object(map)
        }
        other => serde_json::json!(format!("{:?}", other)),
    }
}

/// Baseline snapshot of failed_calls per command, used when CONFIG RESETSTAT is unavailable.
pub type CommandStatsBaseline = HashMap<String, u64>;

/// Parse `INFO COMMANDSTATS` output into a map of command → failed_calls.
fn parse_failed_calls(info: &str) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for line in info.lines() {
        let Some((cmd_part, stats_part)) = line.split_once(':') else {
            continue;
        };
        let cmd_name = cmd_part
            .strip_prefix("cmdstat_")
            .unwrap_or(cmd_part)
            .to_ascii_uppercase();

        for field in stats_part.split(',') {
            if let Some(val) = field.strip_prefix("failed_calls=") {
                if let Ok(n) = val.parse::<u64>() {
                    map.insert(cmd_name.clone(), n);
                }
            }
        }
    }
    map
}

/// Reset server command statistics so subsequent checks start from zero.
/// If CONFIG RESETSTAT is not permitted (e.g. Redis Cloud ACLs), captures
/// a baseline snapshot of current failed_calls so check_commandstats can
/// compare against it.
pub fn reset_commandstats(conn: &mut Connection) -> Result<Option<CommandStatsBaseline>, String> {
    match redis::cmd("CONFIG").arg("RESETSTAT").query::<()>(conn) {
        Ok(()) => Ok(None),
        Err(e) => {
            eprintln!(
                "Warning: CONFIG RESETSTAT not available ({}), will use baseline diff for commandstats",
                e
            );
            // Snapshot current stats as baseline
            let info: String = redis::cmd("INFO")
                .arg("commandstats")
                .query(conn)
                .unwrap_or_default();
            Ok(Some(parse_failed_calls(&info)))
        }
    }
}

/// Check `INFO COMMANDSTATS` for failed_calls on the given commands.
///
/// If a baseline is provided (from a failed RESETSTAT), only reports failures
/// that are NEW since the baseline was captured.
///
/// Returns `Err` if any of the specified commands have new `failed_calls`,
/// listing the offending commands and their failure counts.
pub fn check_commandstats(
    conn: &mut Connection,
    commands: &[&str],
    context: &str,
    baseline: Option<&CommandStatsBaseline>,
) -> Result<(), String> {
    let info: String = match redis::cmd("INFO").arg("commandstats").query(conn) {
        Ok(v) => v,
        Err(_) => return Ok(()), // not available, skip validation
    };

    let current = parse_failed_calls(&info);
    evaluate(&current, commands, context, baseline)
}

/// Pure decision logic behind [`check_commandstats`]: given the parsed
/// `current` failed_calls map, decide whether any of `commands` accrued NEW
/// failures since `baseline` (or any failures at all when `baseline` is `None`).
///
/// Returns `Err` listing each offending command and its NEW failure count, or
/// `Ok(())` when every command is clean. Command names are matched
/// case-insensitively (the map is keyed by uppercase); the error text preserves
/// the caller's original spelling.
fn evaluate(
    current: &HashMap<String, u64>,
    commands: &[&str],
    context: &str,
    baseline: Option<&CommandStatsBaseline>,
) -> Result<(), String> {
    let mut failures = Vec::new();

    for cmd in commands {
        let cmd_upper = cmd.to_ascii_uppercase();
        let current_fails = current.get(&cmd_upper).copied().unwrap_or(0);
        let baseline_fails = baseline
            .and_then(|b| b.get(&cmd_upper))
            .copied()
            .unwrap_or(0);
        let new_fails = current_fails.saturating_sub(baseline_fails);
        if new_fails > 0 {
            failures.push(format!("{}: {} failed_calls", cmd, new_fails));
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Command failures detected after {}: {}",
            context,
            failures.join(", ")
        ))
    }
}

/// Confirm a search index exists before a search run (issue #151-4). On the
/// `--skip-upload` path a name/prefix mismatch (old→new upgrade, wrong config,
/// missing prior upload) would otherwise write a silent `recall 0.0` result
/// file. `FT.INFO <index_name>` errors when the index is absent → hard error.
pub fn ensure_index_exists(conn: &mut Connection, index_name: &str) -> Result<(), String> {
    redis::cmd("FT.INFO")
        .arg(index_name)
        .query::<redis::Value>(conn)
        .map(|_| ())
        .map_err(|_| {
            format!(
                "index '{index_name}' not found — did you upload this config? \
                 (--skip-upload requires a prior upload with --keep-data)"
            )
        })
}

/// `num_docs` from an `FT.INFO` reply (RESP2 flat array or RESP3 map).
///
/// This is the count the search actually sees — keys that exist but are not (yet)
/// in the index cannot answer a query, so a raw keyspace count would overstate
/// the corpus. Returns `None` when the reply carries no `num_docs` field.
/// `FT.INFO` fields that carry the live document count, in preference order.
///
/// RediSearch (Redis / Valkey / Dragonfly) reports `num_docs`. **KiviDB does
/// not** — its `FT.INFO` has no `num_docs` at all and reports `hnsw_live_count`
/// instead, so reading only `num_docs` silently degraded the #238 reuse guard to
/// "this engine cannot count" on KiviDB while the docs claimed otherwise.
const FT_INFO_COUNT_FIELDS: [&str; 2] = ["num_docs", "hnsw_live_count"];

pub fn ft_info_num_docs(v: &redis::Value) -> Option<u64> {
    let pairs: Vec<(String, &redis::Value)> = match v {
        redis::Value::Map(m) => m.iter().map(|(k, val)| (value_to_string(k), val)).collect(),
        redis::Value::Array(items) => items
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (value_to_string(&c[0]), &c[1]))
            .collect(),
        _ => return None,
    };
    FT_INFO_COUNT_FIELDS.iter().find_map(|field| {
        pairs
            .iter()
            .find(|(k, _)| k == field)
            .and_then(|(_, val)| match val {
                redis::Value::Int(n) => u64::try_from(*n).ok(),
                other => value_to_string(other).parse::<u64>().ok(),
            })
    })
}

/// Whether a `FT.INFO` error means "that index does not exist" as opposed to
/// "the probe failed".
///
/// The distinction decides whether the #238 reuse guard may report a corpus size
/// of 0. Only a server reply that names the index as unknown is a real zero;
/// `NOPERM`, a dropped connection or a timeout say nothing about the corpus, and
/// reporting them as 0 sends the user to re-upload a corpus that was fine.
///
/// Wordings differ per engine: RediSearch/KiviDB say "no such index", Valkey
/// Search says "Index with name '<n>' not found".
fn is_missing_index_error(e: &redis::RedisError) -> bool {
    let msg = format!("{} {}", e.detail().unwrap_or_default(), e).to_lowercase();
    msg.contains("no such index")
        || msg.contains("unknown index")
        || (msg.contains("index") && msg.contains("not found"))
}

/// Server-side corpus size for a RediSearch-family index, for the `--skip-upload`
/// reuse precondition (issue #238).
///
/// - index missing → `Ok(Some(0))`: "the index is gone" and "the index is empty"
///   are the same fact here — the corpus to reuse is not there.
/// - index present but the reply carries no count field → `Ok(None)`
///   ("cannot tell"), never a fabricated zero.
/// - the probe itself failed (`NOPERM`, connection dropped) → `Err`, so the
///   caller reports a probe failure rather than an imaginary empty corpus.
pub fn ft_index_num_docs(conn: &mut Connection, index_name: &str) -> Result<Option<u64>, String> {
    match redis::cmd("FT.INFO")
        .arg(index_name)
        .query::<redis::Value>(conn)
    {
        Ok(v) => Ok(ft_info_num_docs(&v)),
        Err(e) if is_missing_index_error(&e) => Ok(Some(0)),
        Err(e) => Err(format!("FT.INFO {index_name} failed: {e}")),
    }
}

/// Per-index memory footprint in bytes, summed from every `*_mb` size field of
/// an `FT.INFO` reply (RESP2 flat array or RESP3 map). Under issue #151-4's
/// coexistence mode the server-wide `used_memory` is the SUM of all resident
/// configs, so it cannot attribute memory per graph; this reads the per-index
/// figure instead. Returns `None` when the reply carries no size fields.
pub fn ft_info_index_memory_bytes(v: &redis::Value) -> Option<i64> {
    let pairs: Vec<(String, &redis::Value)> = match v {
        redis::Value::Map(m) => m.iter().map(|(k, val)| (value_to_string(k), val)).collect(),
        redis::Value::Array(items) => items
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (value_to_string(&c[0]), &c[1]))
            .collect(),
        _ => return None,
    };
    let mut mb = 0.0f64;
    let mut found = false;
    for (k, val) in pairs {
        if k.ends_with("_mb") {
            if let Ok(n) = value_to_string(val).parse::<f64>() {
                mb += n;
                found = true;
            }
        }
    }
    found.then_some((mb * 1024.0 * 1024.0) as i64)
}

/// Non-destructive teardown for a single config's index + keys (issue #151-4),
/// for engines with no `DD` flag on `FT.DROPINDEX` (Valkey / Dragonfly). Drops
/// the index, then SCAN+UNLINKs only keys matching `<key_prefix>*`. `key_prefix`
/// is sanitized (see `index_naming::sanitize_token`) so the SCAN `MATCH` glob
/// contains no metacharacters — it is a literal prefix. Runs OUTSIDE every timed
/// window and every `check_commandstats` reset window. UNLINK (async) avoids the
/// blocking spike a large synchronous `DEL` would cause.
pub fn drop_index_and_keys(conn: &mut Connection, index_name: &str, key_prefix: &str) {
    let _ = redis::cmd("FT.DROPINDEX").arg(index_name).query::<()>(conn);
    let pattern = format!("{key_prefix}*"); // key_prefix is sanitized → no glob metachars
    let mut cursor: u64 = 0;
    loop {
        let (next, keys): (u64, Vec<String>) = match redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(&pattern)
            .arg("COUNT")
            .arg(1000)
            .query(conn)
        {
            Ok(v) => v,
            Err(_) => break,
        };
        if !keys.is_empty() {
            let mut unlink = redis::cmd("UNLINK");
            for k in &keys {
                unlink.arg(k);
            }
            let _ = unlink.query::<()>(conn);
        }
        if next == 0 {
            break;
        }
        cursor = next;
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate, parse_failed_calls, CommandStatsBaseline};
    use std::collections::HashMap;

    fn map(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn evaluate_flags_failures_when_no_baseline() {
        // failed_calls > 0 with no baseline → Err listing the command + count.
        let current = map(&[("VADD", 3)]);
        let err = evaluate(&current, &["VADD"], "upload", None).unwrap_err();
        assert!(err.contains("VADD: 3 failed_calls"), "got {err}");
        assert!(
            err.contains("Command failures detected after upload"),
            "got {err}"
        );
    }

    #[test]
    fn evaluate_ok_when_current_equals_baseline() {
        // No NEW failures since the baseline snapshot → clean run.
        let current = map(&[("VADD", 5)]);
        let baseline: CommandStatsBaseline = map(&[("VADD", 5)]);
        assert_eq!(
            evaluate(&current, &["VADD"], "upload", Some(&baseline)),
            Ok(())
        );
    }

    #[test]
    fn evaluate_reports_delta_over_baseline() {
        // 7 current vs 5 baseline → only the 2 NEW failures are reported.
        let current = map(&[("VADD", 7)]);
        let baseline: CommandStatsBaseline = map(&[("VADD", 5)]);
        let err = evaluate(&current, &["VADD"], "search", Some(&baseline)).unwrap_err();
        assert!(err.contains("VADD: 2 failed_calls"), "got {err}");
    }

    #[test]
    fn evaluate_matches_command_case_insensitively() {
        // Map is keyed by uppercase (from parse_failed_calls); a lowercase
        // command spelling still matches, and the error preserves that spelling.
        let current = map(&[("VADD", 4)]);
        let err = evaluate(&current, &["vadd"], "upload", None).unwrap_err();
        assert!(err.contains("vadd: 4 failed_calls"), "got {err}");
    }

    #[test]
    fn evaluate_ok_when_command_absent_or_zero() {
        // A command with no entry (or zero failures) is clean; unrelated
        // failing commands are ignored when not in the checked list.
        let current = map(&[("FT.SEARCH", 9)]);
        assert_eq!(
            evaluate(&current, &["VADD", "VSIM"], "search", None),
            Ok(())
        );
    }

    #[test]
    fn parses_failed_calls_and_uppercases_command() {
        let info = "cmdstat_FT.SEARCH:calls=10,usec=1234,failed_calls=3\r\n\
                    cmdstat_hset:calls=100,usec=50,rejected_calls=0,failed_calls=0\r\n";
        let m = parse_failed_calls(info);
        assert_eq!(m.get("FT.SEARCH"), Some(&3));
        // lowercase `cmdstat_hset` → command name uppercased to HSET
        assert_eq!(m.get("HSET"), Some(&0));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn skips_lines_without_colon_or_failed_calls() {
        // Header lines, blank lines, and stat lines with no failed_calls field
        // must be ignored rather than producing spurious/zero entries.
        let info = "# Commandstats\r\n\r\ncmdstat_ping:calls=1,usec=1\r\n";
        let m = parse_failed_calls(info);
        assert!(m.is_empty(), "got {:?}", m);
    }

    #[test]
    fn handles_missing_cmdstat_prefix_and_bad_number() {
        // A line without the `cmdstat_` prefix keeps its name as-is (uppercased);
        // an unparseable failed_calls value is skipped.
        let info = "FT.CREATE:failed_calls=2\r\ncmdstat_bad:failed_calls=notanumber";
        let m = parse_failed_calls(info);
        assert_eq!(m.get("FT.CREATE"), Some(&2));
        assert!(!m.contains_key("BAD"));
    }

    #[test]
    fn ft_info_memory_sums_mb_fields_from_both_shapes() {
        use redis::Value;
        fn bulk(s: &str) -> Value {
            Value::BulkString(s.as_bytes().to_vec())
        }
        // RESP2 flat array: only `*_mb` fields count; num_docs is ignored.
        let arr = Value::Array(vec![
            bulk("num_docs"),
            bulk("100"),
            bulk("inverted_sz_mb"),
            bulk("1.0"),
            bulk("vector_index_sz_mb"),
            bulk("2.0"),
        ]);
        assert_eq!(
            super::ft_info_index_memory_bytes(&arr),
            Some(3 * 1024 * 1024)
        );
        // RESP3 map: same total.
        let map = Value::Map(vec![
            (bulk("inverted_sz_mb"), bulk("1.0")),
            (bulk("vector_index_sz_mb"), bulk("2.0")),
        ]);
        assert_eq!(
            super::ft_info_index_memory_bytes(&map),
            Some(3 * 1024 * 1024)
        );
        // No size fields → None.
        assert_eq!(
            super::ft_info_index_memory_bytes(&Value::Array(vec![bulk("num_docs"), bulk("5")])),
            None
        );
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::{extract_versions, info_reply_empty, parse_info, parse_modules};
    use redis::Value;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    #[test]
    fn parse_info_builds_nested_sections() {
        // A multi-section INFO dump becomes { section -> { key -> value } }.
        // `# Section` headers open sections; blank and `#`-only lines are skipped;
        // values keep their colons (e.g. db0 keyspace line).
        let info = "# Server\r\n\
                    redis_version:8.8.0\r\n\
                    os:Linux\r\n\
                    \r\n\
                    # Memory\r\n\
                    used_memory:1048576\r\n\
                    used_memory_human:1.00M\r\n\
                    #\r\n\
                    # Keyspace\r\n\
                    db0:keys=3,expires=0,avg_ttl=0\r\n";
        let parsed = parse_info(info);
        assert_eq!(parsed["Server"]["redis_version"], "8.8.0");
        assert_eq!(parsed["Server"]["os"], "Linux");
        assert_eq!(parsed["Memory"]["used_memory"], "1048576");
        // Value with embedded commas/colons preserved verbatim.
        assert_eq!(parsed["Keyspace"]["db0"], "keys=3,expires=0,avg_ttl=0");
        // Exactly the three real sections (no spurious "default").
        assert_eq!(parsed.as_object().unwrap().len(), 3);
    }

    #[test]
    fn parse_info_empty_is_empty_object() {
        // Empty / whitespace INFO yields an empty object and is flagged for
        // fallback (the Dragonfly `INFO everything` case).
        assert!(info_reply_empty(""));
        assert!(info_reply_empty("   \r\n  "));
        assert!(!info_reply_empty("# Server\r\nredis_version:1"));
        assert_eq!(parse_info("").as_object().unwrap().len(), 0);
    }

    #[test]
    fn extract_versions_pulls_present_keys_only() {
        let info = parse_info(
            "# Server\r\nredis_version:7.4.0\r\nvalkey_version:9.1.0\r\narch_bits:64\r\n",
        );
        let versions = extract_versions(&info);
        assert_eq!(versions["redis_version"], "7.4.0");
        assert_eq!(versions["valkey_version"], "9.1.0");
        // dragonfly_version absent → not inserted; arch_bits ignored.
        assert!(versions.get("dragonfly_version").is_none());
        assert_eq!(versions.as_object().unwrap().len(), 2);
    }

    #[test]
    fn parse_modules_resp2_flat_array_shape() {
        // RESP2: array of modules, each a flat [k, v, k, v, ...] array. This is
        // the Redis / Dragonfly shape. `ver` stays an integer; the SEARCH module
        // version is the load-bearing field.
        let reply = Value::Array(vec![
            Value::Array(vec![
                bulk("name"),
                bulk("search"),
                bulk("ver"),
                Value::Int(21015),
                bulk("path"),
                bulk("/usr/lib/redisearch.so"),
                bulk("args"),
                bulk(""),
            ]),
            Value::Array(vec![
                bulk("name"),
                bulk("ReJSON"),
                bulk("ver"),
                Value::Int(20609),
            ]),
        ]);
        let mods = parse_modules(&reply);
        assert_eq!(mods["search"]["ver"], 21015);
        assert_eq!(mods["search"]["path"], "/usr/lib/redisearch.so");
        assert_eq!(mods["search"]["args"], "");
        assert_eq!(mods["ReJSON"]["ver"], 20609);
        // `name` becomes the key, not a nested field.
        assert!(mods["search"].get("name").is_none());
        assert_eq!(mods.as_object().unwrap().len(), 2);
    }

    #[test]
    fn parse_modules_resp3_map_shape() {
        // RESP3: array of Map entries. Same resulting shape as RESP2.
        let reply = Value::Array(vec![Value::Map(vec![
            (bulk("name"), bulk("search")),
            (bulk("ver"), Value::Int(21015)),
            (bulk("path"), bulk("/x.so")),
        ])]);
        let mods = parse_modules(&reply);
        assert_eq!(mods["search"]["ver"], 21015);
        assert_eq!(mods["search"]["path"], "/x.so");
    }

    #[test]
    fn parse_modules_non_array_is_empty() {
        // A malformed / unexpected reply must not panic — yields an empty object.
        assert_eq!(parse_modules(&Value::Nil).as_object().unwrap().len(), 0);
        assert_eq!(parse_modules(&Value::Int(5)).as_object().unwrap().len(), 0);
    }

    #[test]
    fn config_get_parses_and_redacts_secrets() {
        use super::{parse_kv_map, redact_config};
        // RESP2 flat [k, v, k, v, ...] CONFIG GET reply. Secret-bearing keys
        // (pass/auth/secret/token substrings) have their VALUE redacted but the
        // key is kept; normal keys (maxmemory) and non-secret paths (aclfile)
        // stay verbatim.
        let reply = Value::Array(vec![
            bulk("maxmemory"),
            bulk("0"),
            bulk("requirepass"),
            bulk("hunter2"),
            bulk("masterauth"),
            bulk("s3cr3t"),
            bulk("tls-key-file-pass"),
            bulk("filepw"),
            bulk("aclfile"),
            bulk("/etc/redis/users.acl"),
            bulk("search-timeout"),
            bulk("500"),
        ]);
        let cfg = redact_config(parse_kv_map(&reply));
        assert_eq!(cfg["maxmemory"], "0");
        assert_eq!(cfg["search-timeout"], "500");
        assert_eq!(cfg["aclfile"], "/etc/redis/users.acl");
        assert_eq!(cfg["requirepass"], "<redacted>");
        assert_eq!(cfg["masterauth"], "<redacted>");
        assert_eq!(cfg["tls-key-file-pass"], "<redacted>");
    }

    #[test]
    fn ft_config_get_parses_array_of_pairs() {
        use super::parse_kv_map;
        // Redis' FT.CONFIG GET * returns an array of [name, value] pair-arrays
        // (not a flat array). Verify a search param (MAXSEARCHRESULTS) is keyed
        // by name with its scalar value.
        let reply = Value::Array(vec![
            Value::Array(vec![bulk("MAXSEARCHRESULTS"), bulk("1000000")]),
            Value::Array(vec![bulk("MINPREFIX"), bulk("2")]),
            Value::Array(vec![bulk("TIMEOUT"), bulk("500")]),
        ]);
        let cfg = parse_kv_map(&reply);
        assert_eq!(cfg["MAXSEARCHRESULTS"], "1000000");
        assert_eq!(cfg["MINPREFIX"], "2");
        assert_eq!(cfg["TIMEOUT"], "500");
        assert_eq!(cfg.as_object().unwrap().len(), 3);
    }

    #[test]
    fn config_get_flat_array_still_parses() {
        use super::parse_kv_map;
        // Plain CONFIG GET on RESP2 is a flat [k, v, k, v] array — the
        // non-nested branch must chunk it by pairs.
        let reply = Value::Array(vec![
            bulk("maxmemory"),
            bulk("0"),
            bulk("save"),
            bulk("3600 1"),
        ]);
        let cfg = parse_kv_map(&reply);
        assert_eq!(cfg["maxmemory"], "0");
        assert_eq!(cfg["save"], "3600 1");
    }
}

#[cfg(test)]
mod num_docs_tests {
    use super::ft_info_num_docs;
    use redis::Value;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    // RESP2: FT.INFO is a flat alternating array; num_docs arrives as a string.
    #[test]
    fn reads_num_docs_from_resp2_array() {
        let v = Value::Array(vec![
            bulk("index_name"),
            bulk("idx:cfg"),
            bulk("num_docs"),
            bulk("400"),
            bulk("indexing"),
            Value::Int(0),
        ]);
        assert_eq!(ft_info_num_docs(&v), Some(400));
    }

    // RESP3: a map, and num_docs may be a native integer.
    #[test]
    fn reads_num_docs_from_resp3_map() {
        let v = Value::Map(vec![
            (bulk("index_name"), bulk("idx:cfg")),
            (bulk("num_docs"), Value::Int(400)),
        ]);
        assert_eq!(ft_info_num_docs(&v), Some(400));
    }

    // No num_docs field → "cannot tell", never a fabricated 0: a 0 would be read
    // as "the corpus is gone" and abort a legitimate --skip-upload run (#238).
    #[test]
    fn missing_field_is_none_not_zero() {
        let v = Value::Array(vec![bulk("index_name"), bulk("idx:cfg")]);
        assert_eq!(ft_info_num_docs(&v), None);
        assert_eq!(ft_info_num_docs(&Value::Nil), None);
    }

    // An empty index legitimately reports 0.
    #[test]
    fn empty_index_reports_zero() {
        let v = Value::Array(vec![bulk("num_docs"), bulk("0")]);
        assert_eq!(ft_info_num_docs(&v), Some(0));
    }

    // A negative or unparseable value must not wrap into a huge u64.
    #[test]
    fn nonsense_values_are_none() {
        assert_eq!(
            ft_info_num_docs(&Value::Array(vec![bulk("num_docs"), Value::Int(-1)])),
            None
        );
        assert_eq!(
            ft_info_num_docs(&Value::Array(vec![bulk("num_docs"), bulk("many")])),
            None
        );
    }
}
