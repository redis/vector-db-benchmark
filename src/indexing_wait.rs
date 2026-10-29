//! How long an engine's post-upload "wait for the index to catch up" loop may
//! run, and what happens when it runs out (#361).
//!
//! The FT.INFO-polling engines (Redis, Valkey, Dragonfly, KiviDB) used to print
//! `Warning: indexing timeout` and return `Ok(())` when the budget expired, so
//! the run went on to search a partly built index and scored it against the
//! full ground truth. Expiry is now an error: a published recall/QPS number
//! must come from the index the run claims it built.
//!
//! The budget scales with the corpus — a fixed 120 s or 600 s cannot cover
//! background indexing at 100M documents — and `INDEXING_WAIT_SECS` overrides
//! it for every one of those engines.

use std::time::Duration;

/// Env var that replaces the computed budget (whole seconds, > 0).
pub const INDEXING_WAIT_ENV: &str = "INDEXING_WAIT_SECS";

/// What the wait is waiting for, which decides how fast the budget grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitPhase {
    /// After this run's own upload: the index was built as the documents
    /// arrived, so the wait only covers the asynchronous tail.
    UploadTail,
    /// The WHOLE index is built inside the wait: Redis shared-corpus mode
    /// (#188), where configs 2..N recreate their index over hashes that are
    /// already there and RediSearch backfills it from scratch.
    FullBuild,
}

impl WaitPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            WaitPhase::UploadTail => "upload_tail",
            WaitPhase::FullBuild => "full_build",
        }
    }

    /// Documents per extra second of budget beyond the engine floor.
    pub fn docs_per_extra_second(self) -> u64 {
        match self {
            WaitPhase::UploadTail => DOCS_PER_EXTRA_SECOND,
            WaitPhase::FullBuild => FULL_BUILD_DOCS_PER_EXTRA_SECOND,
        }
    }
}

/// [`WaitPhase::UploadTail`]: one second per 10,000 documents beyond the
/// floor (1M -> +100 s, 10M -> +1,000 s, 100M -> +10,000 s ≈ 2.8 h).
pub const DOCS_PER_EXTRA_SECOND: u64 = 10_000;

/// [`WaitPhase::FullBuild`]: one second per 1,000 documents (1M -> +1,000 s,
/// 10M -> +10,000 s ≈ 2.8 h, 100M -> +100,000 s ≈ 27.8 h). An UNMEASURED,
/// deliberately conservative estimate: the only measured rate is the 10M
/// Redis upload including its synchronous index build, 2,277 docs/s (HNSW
/// M=32, EF_CONSTRUCTION=256); a backfill was not timed. `INDEXING_WAIT_SECS`
/// overrides it.
pub const FULL_BUILD_DOCS_PER_EXTRA_SECOND: u64 = 1_000;

/// The wait budget for `expected` documents in `phase`: `floor_secs` plus the
/// phase's per-document slope, unless `override_secs` (from
/// [`INDEXING_WAIT_ENV`]) is set to a positive value.
pub fn budget(
    phase: WaitPhase,
    floor_secs: u64,
    expected: usize,
    override_secs: Option<u64>,
) -> Duration {
    match override_secs {
        Some(s) if s > 0 => Duration::from_secs(s),
        _ => Duration::from_secs(floor_secs + expected as u64 / phase.docs_per_extra_second()),
    }
}

/// Called after a poll found the index incomplete: `Ok(())` keep polling,
/// `Err` the budget is spent. `progress` describes the last observed state for
/// the error message.
pub fn verdict(
    engine: &str,
    elapsed: Duration,
    budget: Duration,
    progress: &str,
) -> Result<(), String> {
    if elapsed > budget {
        Err(format!(
            "{engine}: index still incomplete after {:.0}s, the indexing wait budget \
             ({progress}); refusing to search a partly built index. Raise the budget with \
             {INDEXING_WAIT_ENV}=<seconds>",
            budget.as_secs_f64()
        ))
    } else {
        Ok(())
    }
}

/// What one FT.INFO poll of a RediSearch-dialect engine (Redis, Valkey Search,
/// Dragonfly) reported. `num_docs` is `None` when the field is ABSENT — never
/// read as 0, which would burn the whole budget waiting for a count the server
/// does not report (#361 review; KiviDB FLAT was the observed case).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FtInfoProgress {
    pub num_docs: Option<u64>,
    /// `indexing` (RediSearch / Dragonfly) or `backfill_in_progress` (Valkey
    /// Search) set.
    pub indexing: bool,
    /// `percent_indexed`; absent (Valkey Search) is treated as complete.
    pub percent_indexed: Option<f64>,
    /// Valkey Search's `state` (`ready` once built).
    pub state: Option<String>,
    /// Every top-level field name, for the error when `num_docs` is absent.
    pub fields: Vec<String>,
}

impl FtInfoProgress {
    pub fn describe(&self, expected: u64) -> String {
        format!(
            "num_docs={}/{expected}, indexing={}, percent_indexed={}, state={}",
            self.num_docs.map_or("absent".into(), |n| n.to_string()),
            self.indexing,
            self.percent_indexed
                .map_or("absent".into(), |p| format!("{p:.2}")),
            self.state.as_deref().unwrap_or("absent"),
        )
    }
}

fn value_text(v: &redis::Value) -> Option<String> {
    match v {
        redis::Value::BulkString(b) => Some(String::from_utf8_lossy(b).into_owned()),
        redis::Value::SimpleString(s) => Some(s.clone()),
        redis::Value::Int(n) => Some(n.to_string()),
        redis::Value::Double(f) => Some(f.to_string()),
        redis::Value::Boolean(b) => Some(if *b { "1" } else { "0" }.to_string()),
        _ => None,
    }
}

/// Parse an `FT.INFO` reply (RESP2 flat array or RESP3 map). A field that is
/// present but unparsable is an error naming it; an absent one is `None`.
pub fn parse_ft_info(info: &redis::Value) -> Result<FtInfoProgress, String> {
    let mut p = FtInfoProgress::default();
    for (key, val) in ft_info_pairs(info)? {
        match key.as_str() {
            "num_docs" => p.num_docs = Some(parse_count(&key, val)?),
            "indexing" | "backfill_in_progress" => p.indexing |= parse_flag(&key, val)?,
            "percent_indexed" => {
                let t = field_text(&key, val)?;
                p.percent_indexed = Some(
                    t.trim()
                        .parse::<f64>()
                        .map_err(|_| format!("FT.INFO percent_indexed is not a number: {t:?}"))?,
                );
            }
            "state" => p.state = value_text(val),
            _ => {}
        }
        p.fields.push(key);
    }
    Ok(p)
}

/// Top-level (field, value) pairs of an `FT.INFO` reply, RESP2 or RESP3.
fn ft_info_pairs(info: &redis::Value) -> Result<Vec<(String, &redis::Value)>, String> {
    match info {
        redis::Value::Array(a) => Ok(a
            .chunks(2)
            .filter(|c| c.len() == 2)
            .filter_map(|c| value_text(&c[0]).map(|k| (k, &c[1])))
            .collect()),
        redis::Value::Map(m) => Ok(m
            .iter()
            .filter_map(|(k, v)| value_text(k).map(|k| (k, v)))
            .collect()),
        other => Err(format!("unexpected FT.INFO reply type: {other:?}")),
    }
}

fn field_text(key: &str, val: &redis::Value) -> Result<String, String> {
    value_text(val).ok_or_else(|| format!("FT.INFO {key} is {val:?}"))
}

/// A non-negative document count; anything else is an error naming the field.
fn parse_count(key: &str, val: &redis::Value) -> Result<u64, String> {
    let t = field_text(key, val)?;
    let t = t.trim();
    t.parse::<u64>()
        .ok()
        .or_else(|| {
            t.parse::<f64>()
                .ok()
                .filter(|f| *f >= 0.0)
                .map(|f| f as u64)
        })
        .ok_or_else(|| format!("FT.INFO {key} is not a count: {t:?}"))
}

fn parse_flag(key: &str, val: &redis::Value) -> Result<bool, String> {
    let t = field_text(key, val)?;
    match t.trim() {
        "0" | "false" => Ok(false),
        "1" | "true" => Ok(true),
        other => other
            .parse::<f64>()
            .map(|f| f != 0.0)
            .map_err(|_| format!("FT.INFO {key} is not a flag: {other:?}")),
    }
}

/// Parse a KiviDB `FT.INFO` reply with the same strictness as
/// [`parse_ft_info`]: an absent count is `None`, an unparsable one an error.
pub fn parse_kividb_ft_info(info: &redis::Value) -> Result<KividbProgress, String> {
    let mut p = KividbProgress::default();
    for (key, val) in ft_info_pairs(info)? {
        match key.as_str() {
            "hnsw_live_count" => p.hnsw_live_count = Some(parse_count(&key, val)?),
            "num_docs" => p.num_docs = Some(parse_count(&key, val)?),
            "hnsw_compaction_in_progress" => p.compaction_in_progress |= parse_flag(&key, val)?,
            _ => {}
        }
    }
    Ok(p)
}

/// Is the index complete? `Err` (fail fast, not after the budget) when the
/// reply has no `num_docs`: without it the wait cannot know whether the index
/// holds the corpus. `use_state`: honour Valkey Search's `state` field.
pub fn ft_info_ready(
    engine: &str,
    p: &FtInfoProgress,
    expected: u64,
    use_state: bool,
) -> Result<bool, String> {
    let Some(n) = p.num_docs else {
        return Err(format!(
            "{engine}: FT.INFO reports no num_docs, so the indexing wait cannot tell whether \
             the index holds the {expected} uploaded documents (fields: {})",
            p.fields.join(",")
        ));
    };
    let state_ok = !use_state
        || p.state
            .as_deref()
            .is_none_or(|s| s.is_empty() || s == "ready");
    Ok(n >= expected && !p.indexing && p.percent_indexed.unwrap_or(1.0) >= 1.0 && state_ok)
}

/// What one KiviDB `FT.INFO` poll reported (`None`: field absent).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KividbProgress {
    pub num_docs: Option<u64>,
    pub hnsw_live_count: Option<u64>,
    pub compaction_in_progress: bool,
}

/// Is a KiviDB index holding `expected` documents ready to search? The signal
/// depends on the algorithm (#361 CI: a FLAT index never reports
/// `hnsw_live_count`, so the HNSW rule waited out the whole budget):
///
/// * `HNSW`: `hnsw_live_count` (else `num_docs`) has reached `expected` and no
///   tombstone compaction is running. No count at all is NOT ready.
/// * `FLAT`: `num_docs` (else `hnsw_live_count`) has reached `expected`. A
///   server that reports no count for a FLAT index is ready: KiviDB appends a
///   FLAT vector inside the `HSET` that stores it (there is no graph and no
///   background build), so once the upload's HSETs have returned there is
///   nothing to wait for — and nothing to poll.
///
/// Any other algorithm is an error rather than a guess.
pub fn kividb_ready(algorithm: &str, p: &KividbProgress, expected: u64) -> Result<bool, String> {
    if algorithm.eq_ignore_ascii_case("hnsw") {
        Ok(match p.hnsw_live_count.or(p.num_docs) {
            Some(n) => n >= expected && !p.compaction_in_progress,
            None => false,
        })
    } else if algorithm.eq_ignore_ascii_case("flat") {
        Ok(match p.num_docs.or(p.hnsw_live_count) {
            Some(n) => n >= expected,
            // Verified against KiviDB source (kividbio/kividb src/commands/search.rs,
            // v1.1.0 and main: `hnsw_*` only for HNSW fields) and CI on the pinned
            // quay.io/kividbio/kividb:v1.0.2-full image (#361, d682fb6).
            None => true,
        })
    } else {
        Err(format!(
            "kividb: no indexing-readiness rule for algorithm {algorithm:?} (FLAT or HNSW)"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bulk(s: &str) -> redis::Value {
        redis::Value::BulkString(s.as_bytes().to_vec())
    }

    /// Absent is not 0: an FT.INFO without `num_docs` fails fast, and an
    /// unparsable one is an error naming the field, for both reply shapes.
    #[test]
    fn ft_info_absent_or_unparsable_num_docs_fails_fast() {
        let resp2 = redis::Value::Array(vec![
            bulk("index_name"),
            bulk("idx"),
            bulk("indexing"),
            bulk("0"),
        ]);
        let p = parse_ft_info(&resp2).unwrap();
        assert_eq!(p.num_docs, None);
        let e = ft_info_ready("redis", &p, 10, false).unwrap_err();
        assert!(
            e.contains("no num_docs") && e.contains("index_name,indexing"),
            "{e}"
        );
        let resp3 = redis::Value::Map(vec![(bulk("num_docs"), bulk("many"))]);
        assert!(parse_ft_info(&resp3)
            .unwrap_err()
            .contains("num_docs is not a count"));
        let pct = redis::Value::Array(vec![
            bulk("num_docs"),
            bulk("5"),
            bulk("percent_indexed"),
            bulk("x"),
        ]);
        assert!(parse_ft_info(&pct).unwrap_err().contains("percent_indexed"));
        assert!(parse_ft_info(&redis::Value::Nil).is_err());
    }

    #[test]
    fn ft_info_progress_parses_every_dialect_field() {
        let info = redis::Value::Map(vec![
            (bulk("num_docs"), redis::Value::Int(10)),
            (bulk("indexing"), redis::Value::Int(0)),
            (bulk("backfill_in_progress"), bulk("1")),
            (bulk("percent_indexed"), redis::Value::Double(0.5)),
            (
                bulk("state"),
                redis::Value::SimpleString("backfilling".into()),
            ),
        ]);
        let p = parse_ft_info(&info).unwrap();
        assert_eq!(p.num_docs, Some(10));
        assert!(p.indexing);
        assert_eq!(p.percent_indexed, Some(0.5));
        assert_eq!(p.state.as_deref(), Some("backfilling"));
        assert_eq!(ft_info_ready("valkey", &p, 10, true), Ok(false));
    }

    /// #361 review: KiviDB's counts parse as strictly as the other engines':
    /// absent is `None`, unparsable is an immediate error.
    #[test]
    fn kividb_ft_info_parses_strictly() {
        let ok = redis::Value::Array(vec![
            bulk("hnsw_live_count"),
            bulk("1000"),
            bulk("hnsw_compaction_in_progress"),
            bulk("0"),
        ]);
        let p = parse_kividb_ft_info(&ok).unwrap();
        assert_eq!((p.hnsw_live_count, p.num_docs), (Some(1000), None));
        assert!(!p.compaction_in_progress);
        let flat = redis::Value::Array(vec![bulk("index_name"), bulk("idx")]);
        assert_eq!(
            parse_kividb_ft_info(&flat).unwrap(),
            KividbProgress::default()
        );
        for bad in [
            redis::Value::Array(vec![bulk("hnsw_live_count"), bulk("lots")]),
            redis::Value::Array(vec![bulk("num_docs"), bulk("-3")]),
            redis::Value::Array(vec![bulk("hnsw_compaction_in_progress"), bulk("maybe")]),
        ] {
            assert!(parse_kividb_ft_info(&bad).is_err(), "{bad:?}");
        }
    }

    /// The readiness decision per algorithm, including the CI failure on
    /// #361: a FLAT index whose FT.INFO has no `hnsw_live_count`.
    #[test]
    fn kividb_readiness_follows_the_algorithm() {
        let p = |num_docs, live, compaction| KividbProgress {
            num_docs,
            hnsw_live_count: live,
            compaction_in_progress: compaction,
        };
        // HNSW: the live count decides, compaction blocks, no count waits.
        assert_eq!(
            kividb_ready("HNSW", &p(None, Some(1000), false), 1000),
            Ok(true)
        );
        assert_eq!(
            kividb_ready("hnsw", &p(None, Some(999), false), 1000),
            Ok(false)
        );
        assert_eq!(
            kividb_ready("hnsw", &p(None, Some(1000), true), 1000),
            Ok(false)
        );
        assert_eq!(
            kividb_ready("hnsw", &p(Some(1000), None, false), 1000),
            Ok(true)
        );
        assert_eq!(
            kividb_ready("hnsw", &p(Some(5), Some(1000), false), 1000),
            Ok(true)
        );
        assert_eq!(kividb_ready("hnsw", &p(None, None, false), 1000), Ok(false));
        // FLAT: no hnsw_live_count is normal; num_docs decides when present.
        assert_eq!(kividb_ready("flat", &p(None, None, false), 1000), Ok(true));
        assert_eq!(
            kividb_ready("FLAT", &p(Some(1000), None, false), 1000),
            Ok(true)
        );
        assert_eq!(
            kividb_ready("flat", &p(Some(999), None, false), 1000),
            Ok(false)
        );
        assert_eq!(
            kividb_ready("flat", &p(None, Some(10), false), 1000),
            Ok(false)
        );
        assert!(kividb_ready("ivf", &p(Some(1000), None, false), 1000).is_err());
    }

    #[test]
    fn budget_scales_with_the_corpus_and_the_env_overrides_it() {
        use WaitPhase::*;
        assert_eq!(budget(UploadTail, 120, 0, None), Duration::from_secs(120));
        assert_eq!(
            budget(UploadTail, 120, 99_964, None),
            Duration::from_secs(129)
        );
        assert_eq!(
            budget(UploadTail, 120, 100_000_027, None),
            Duration::from_secs(10_120)
        );
        assert_eq!(
            budget(UploadTail, 600, 10_000_000, None),
            Duration::from_secs(1_600)
        );
        assert_eq!(
            budget(UploadTail, 120, 100_000_027, Some(60)),
            Duration::from_secs(60)
        );
        // 0 means "not set", not "no wait".
        assert_eq!(
            budget(UploadTail, 120, 0, Some(0)),
            Duration::from_secs(120)
        );
    }

    /// #361 review: a shared-corpus backfill builds the whole index inside
    /// the wait, so its slope is 10x the upload tail's. At 1M a full build at
    /// the measured 2,277 docs/s takes ~439 s; the tail budget (220 s) would
    /// hard-fail it, the full-build budget (1,120 s) does not.
    #[test]
    fn a_full_build_gets_a_full_build_budget() {
        use WaitPhase::*;
        let at_measured_rate = |n: usize| Duration::from_secs(n as u64 / 2_277);
        for n in [1_000_044usize, 9_999_959, 100_000_027] {
            assert!(budget(FullBuild, 120, n, None) > at_measured_rate(n), "{n}");
        }
        assert!(budget(UploadTail, 120, 1_000_044, None) < at_measured_rate(1_000_044));
        assert_eq!(
            budget(FullBuild, 120, 1_000_044, None),
            Duration::from_secs(1_120)
        );
        assert_eq!(
            budget(FullBuild, 120, 100_000_027, None),
            Duration::from_secs(100_120)
        );
        assert_eq!(
            budget(FullBuild, 120, 100_000_027, Some(7)),
            Duration::from_secs(7)
        );
        assert_eq!(
            (UploadTail.as_str(), FullBuild.as_str()),
            ("upload_tail", "full_build")
        );
    }

    /// The defect this module exists for: an expired budget must be an error.
    #[test]
    fn an_expired_budget_is_an_error_not_a_pass() {
        let b = Duration::from_secs(10);
        let e = verdict("redis", Duration::from_secs(11), b, "num_docs=5/9").unwrap_err();
        assert!(e.contains("redis") && e.contains("num_docs=5/9"), "{e}");
        assert!(e.contains(INDEXING_WAIT_ENV), "{e}");
        assert_eq!(verdict("redis", Duration::from_secs(10), b, ""), Ok(()));
    }
}
