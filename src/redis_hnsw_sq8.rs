//! Redis HNSW SQ8 options and server read-back for benchmark fidelity.

use redis::Value as RedisValue;
use serde::{Deserialize, Deserializer};

/// The training threshold RediSearch applies when `COMPRESSION SQ8` is given
/// without `TRAINING_THRESHOLD`. This is the server's default, not a value
/// this harness invents. RediSearch `src/vector_index.h` defines
/// `HNSW_QUANT_DEFAULT_TRAINING_THRESHOLD (10 * DEFAULT_BLOCK_SIZE)`, and
/// `src/spec.c` applies it when compression is set and the threshold is not.
/// VectorSimilarity `src/VecSim/vec_sim_common.h` defines `DEFAULT_BLOCK_SIZE`
/// as 1024. RediSearch `tests/pytests/test_vecsim_hnsw_sq8.py` asserts that
/// `FT.INFO` reports 10240 for such an index. `verify_ft_info` checks the
/// value against `FT.INFO` at index creation, so a server whose default
/// differs fails the run instead of being recorded wrongly.
pub const SERVER_DEFAULT_TRAINING_THRESHOLD: u64 = 10 * 1024;

/// Upper bound accepted by RediSearch (`HNSW_QUANT_MAX_TRAINING_THRESHOLD`,
/// `100 * DEFAULT_BLOCK_SIZE`).
pub const MAX_TRAINING_THRESHOLD: u64 = 100 * 1024;

/// Whether the `REDIS_HNSW_*` env vars may apply to a config. They apply only
/// to HNSW configs that build a vector index, so an exported variable never
/// changes a FLAT, SVS-VAMANA or no-vector (filter-only) config.
pub fn env_applies(algorithm: &str, skip_vector_index: bool) -> bool {
    algorithm.eq_ignore_ascii_case("hnsw") && !skip_vector_index
}

/// Refuse `collection_params.hnsw_config.COMPRESSION` / `TRAINING_THRESHOLD`
/// on any engine but `redis`. The keys are typed on the shared `HnswConfig`,
/// so they no longer reach its catch-all and would not be reported as
/// ignored: a copied `*-sq8-*` config on valkey, dragonfly or kividb would
/// build an uncompressed index under an SQ8 name, with nothing in the result
/// to say so. Only the Redis engine reads them, so anywhere else they are an
/// error before anything is built.
pub fn check_engine_reads_sq8_keys(
    engine: &str,
    compression_declared: bool,
    threshold_declared: bool,
) -> Result<(), String> {
    if engine == "redis" || (!compression_declared && !threshold_declared) {
        return Ok(());
    }
    let keys: Vec<&str> = [
        (compression_declared, "COMPRESSION"),
        (threshold_declared, "TRAINING_THRESHOLD"),
    ]
    .into_iter()
    .filter_map(|(declared, key)| declared.then_some(key))
    .collect();
    Err(format!(
        "collection_params.hnsw_config.{} is read only by the redis engine; engine \
         '{engine}' would build an uncompressed index under this config's name",
        keys.join(" / ")
    ))
}

/// The outcome of [`Sq8Options::resolve`]: the options, plus which env vars
/// supplied a value the config did not declare.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sq8Resolution {
    pub options: Sq8Options,
    /// `REDIS_HNSW_*` variables whose value was used, in field order.
    pub from_env: Vec<&'static str>,
}

impl Sq8Resolution {
    /// A one-line stderr notice when any SQ8 setting came from the env rather
    /// than the config. The run still completes and its results carry the
    /// config's name, so a config without SQ8 keys that was built as SQ8 must
    /// at least say so.
    pub fn env_notice(&self, config_name: &str) -> Option<String> {
        if self.from_env.is_empty() {
            return None;
        }
        let mut parts = Vec::new();
        if self.from_env.contains(&"REDIS_HNSW_COMPRESSION") {
            parts.push(format!(
                "COMPRESSION {} from REDIS_HNSW_COMPRESSION",
                self.options.compression.as_deref().unwrap_or("?")
            ));
        }
        if self.from_env.contains(&"REDIS_HNSW_TRAINING_THRESHOLD") {
            parts.push(format!(
                "TRAINING_THRESHOLD {} from REDIS_HNSW_TRAINING_THRESHOLD",
                self.options
                    .training_threshold
                    .map_or_else(|| "?".to_string(), |n| n.to_string())
            ));
        }
        Some(format!(
            "notice: Redis config '{config_name}' builds its HNSW index with {}, not from \
             the config; its results are labeled '{config_name}'",
            parts.join(" and ")
        ))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sq8Options {
    pub compression: Option<String>,
    pub training_threshold: Option<u64>,
}

impl Sq8Options {
    pub fn from_hnsw_fields(
        compression: Option<&str>,
        training_threshold: Option<u64>,
        algorithm: &str,
        data_type: &str,
        skip_vector_index: bool,
    ) -> Result<Self, String> {
        if compression.is_none() && training_threshold.is_none() {
            return Ok(Self::default());
        }
        if !algorithm.eq_ignore_ascii_case("hnsw") || skip_vector_index {
            return Err(
                "collection_params.hnsw_config SQ8 options require an HNSW vector index".into(),
            );
        }
        let compression = match compression {
            Some(s) if s.eq_ignore_ascii_case("SQ8") => Some("SQ8".to_string()),
            Some(_) => return Err("collection_params.hnsw_config.COMPRESSION must be SQ8".into()),
            None => None,
        };
        if let Some(n) = training_threshold {
            if n > MAX_TRAINING_THRESHOLD {
                return Err("collection_params.hnsw_config.TRAINING_THRESHOLD must be an integer from 0 to 102400".into());
            }
            if compression.is_none() {
                return Err(
                    "collection_params.hnsw_config.TRAINING_THRESHOLD requires COMPRESSION SQ8"
                        .into(),
                );
            }
        }
        if compression.is_some()
            && !data_type.eq_ignore_ascii_case("FLOAT32")
            && !data_type.eq_ignore_ascii_case("FLOAT16")
        {
            return Err("HNSW SQ8 COMPRESSION requires FLOAT32 or FLOAT16 vectors".into());
        }
        Ok(Self {
            compression,
            training_threshold,
        })
    }

    /// Merge config and env values (config wins per field), then validate via
    /// `from_hnsw_fields`. The env readers are called only when
    /// [`env_applies`] holds, so FLAT, SVS-VAMANA and no-vector configs never
    /// see `REDIS_HNSW_*`. Env text is trimmed; empty or whitespace-only env is
    /// treated as unset. Invalid env text errors naming the variable.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        config_compression: Option<&str>,
        config_training_threshold: Option<u64>,
        env_compression: impl FnOnce() -> Option<String>,
        env_training_threshold: impl FnOnce() -> Option<String>,
        algorithm: &str,
        data_type: &str,
        skip_vector_index: bool,
    ) -> Result<Sq8Resolution, String> {
        let (env_compression, env_training_threshold) = if env_applies(algorithm, skip_vector_index)
        {
            (env_compression(), env_training_threshold())
        } else {
            (None, None)
        };
        let env_compression = env_compression
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let env_training_threshold = env_training_threshold
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        // A garbage env value is ignored when the config value for that field
        // wins, so only the env text that is actually used is validated.
        let compression_from_env = config_compression.is_none() && env_compression.is_some();
        let compression = match (config_compression, env_compression) {
            (Some(c), _) => Some(c),
            (None, Some(e)) if e.eq_ignore_ascii_case("SQ8") => Some("SQ8"),
            (None, Some(e)) => {
                return Err(format!("REDIS_HNSW_COMPRESSION must be SQ8, got {e:?}"));
            }
            (None, None) => None,
        };
        let threshold_from_env =
            config_training_threshold.is_none() && env_training_threshold.is_some();
        let training_threshold = match (config_training_threshold, env_training_threshold) {
            (Some(n), _) => Some(n),
            (None, Some(e)) => {
                let n = e
                    .parse::<u64>()
                    .ok()
                    .filter(|&n| n <= MAX_TRAINING_THRESHOLD)
                    .ok_or_else(|| {
                    format!(
                        "REDIS_HNSW_TRAINING_THRESHOLD must be an integer from 0 to 102400, got {e:?}"
                    )
                })?;
                if compression.is_none() {
                    return Err("REDIS_HNSW_TRAINING_THRESHOLD requires COMPRESSION SQ8 \
                                (config or REDIS_HNSW_COMPRESSION)"
                        .into());
                }
                Some(n)
            }
            (None, None) => None,
        };
        let options = Self::from_hnsw_fields(
            compression,
            training_threshold,
            algorithm,
            data_type,
            skip_vector_index,
        )
        .map_err(|e| {
            if compression_from_env {
                format!("{e} (from REDIS_HNSW_COMPRESSION)")
            } else {
                e
            }
        })?;
        let mut from_env = Vec::new();
        if compression_from_env {
            from_env.push("REDIS_HNSW_COMPRESSION");
        }
        if threshold_from_env {
            from_env.push("REDIS_HNSW_TRAINING_THRESHOLD");
        }
        Ok(Sq8Resolution { options, from_env })
    }

    /// The `(hnsw_compression, hnsw_training_threshold)` pair recorded in a
    /// result's effective config. It reflects the RESOLVED options, i.e. what
    /// FT.CREATE built (env included), never only what the config declared;
    /// the threshold is the server default when only compression is set. Both
    /// are `null` without compression. The docs tell users to check these
    /// fields before comparing runs, so this is pinned by unit tests.
    pub fn effective_record(&self) -> (serde_json::Value, serde_json::Value) {
        (
            self.compression
                .clone()
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
            self.effective_training_threshold()
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
        )
    }

    /// The threshold the server builds with: the declared one, else the
    /// server default ([`SERVER_DEFAULT_TRAINING_THRESHOLD`]). `None` without
    /// compression, where the server has no training threshold.
    pub fn effective_training_threshold(&self) -> Option<u64> {
        self.compression.as_ref().map(|_| {
            self.training_threshold
                .unwrap_or(SERVER_DEFAULT_TRAINING_THRESHOLD)
        })
    }

    pub fn verify_ft_info(&self, info: &RedisValue) -> Result<(), String> {
        if self.compression.is_none() {
            return Ok(());
        }
        let attributes = field(info, "attributes").ok_or("FT.INFO has no attributes")?;
        let entries = match attributes {
            RedisValue::Array(entries) => entries,
            _ => return Err("FT.INFO attributes has an unexpected shape".into()),
        };
        let vector = entries
            .iter()
            .find(|entry| {
                field(entry, "identifier").and_then(as_string).as_deref() == Some("vector")
            })
            .ok_or("FT.INFO has no vector attribute")?;
        let actual_compression = field(vector, "compression").and_then(as_string);
        if actual_compression.as_deref() != self.compression.as_deref() {
            return Err(format!(
                "FT.INFO vector compression {:?} differs from requested {:?}",
                actual_compression, self.compression
            ));
        }
        let expected_threshold = self
            .training_threshold
            .unwrap_or(SERVER_DEFAULT_TRAINING_THRESHOLD);
        let actual_threshold = field(vector, "training_threshold")
            .and_then(as_string)
            .and_then(|s| s.parse::<u64>().ok());
        if actual_threshold != Some(expected_threshold) {
            return Err(format!(
                "FT.INFO vector training_threshold {:?} differs from requested {}",
                actual_threshold, expected_threshold
            ));
        }
        Ok(())
    }
}

/// A declared value must have the right JSON type, including rejection of null.
pub fn present_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
}

/// A missing threshold is distinct from a declared zero or malformed value.
pub fn present_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    u64::deserialize(deserializer).map(Some)
}

fn field<'a>(value: &'a RedisValue, key: &str) -> Option<&'a RedisValue> {
    match value {
        RedisValue::Map(pairs) => pairs
            .iter()
            .find_map(|(k, v)| (as_string(k).as_deref() == Some(key)).then_some(v)),
        RedisValue::Array(items) => items
            .as_chunks::<2>()
            .0
            .iter()
            .find_map(|pair| (as_string(&pair[0]).as_deref() == Some(key)).then_some(&pair[1])),
        _ => None,
    }
}

fn as_string(value: &RedisValue) -> Option<String> {
    match value {
        RedisValue::SimpleString(s) => Some(s.clone()),
        RedisValue::BulkString(s) => String::from_utf8(s.clone()).ok(),
        RedisValue::Int(n) => Some(n.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_sq8_and_preserves_zero() {
        let options =
            Sq8Options::from_hnsw_fields(Some("sq8"), Some(0), "hnsw", "FLOAT16", false).unwrap();
        assert_eq!(options.compression.as_deref(), Some("SQ8"));
        assert_eq!(options.training_threshold, Some(0));
        assert!(Sq8Options::from_hnsw_fields(Some("SQ8"), None, "hnsw", "FLOAT32", false).is_ok());
        for (compression, threshold, algorithm, data_type, skip) in [
            (Some("SQ4"), None, "hnsw", "FLOAT32", false),
            (Some("SQ8"), Some(102_401), "hnsw", "FLOAT32", false),
            (None, Some(0), "hnsw", "FLOAT32", false),
            (Some("SQ8"), None, "flat", "FLOAT32", false),
            (Some("SQ8"), None, "hnsw", "INT8", false),
            (Some("SQ8"), None, "hnsw", "FLOAT32", true),
        ] {
            assert!(Sq8Options::from_hnsw_fields(
                compression,
                threshold,
                algorithm,
                data_type,
                skip
            )
            .is_err());
        }
    }

    fn resolve_full(
        config: (Option<&str>, Option<u64>),
        env: (Option<&str>, Option<&str>),
    ) -> Result<Sq8Resolution, String> {
        Sq8Options::resolve(
            config.0,
            config.1,
            || env.0.map(String::from),
            || env.1.map(String::from),
            "hnsw",
            "FLOAT32",
            false,
        )
    }

    fn resolve(
        config: (Option<&str>, Option<u64>),
        env: (Option<&str>, Option<&str>),
    ) -> Result<Sq8Options, String> {
        resolve_full(config, env).map(|r| r.options)
    }

    #[test]
    fn sq8_keys_are_refused_on_every_engine_but_redis() {
        assert!(check_engine_reads_sq8_keys("redis", true, true).is_ok());
        for engine in ["valkey", "dragonfly", "kividb", "qdrant", "unknown"] {
            assert!(check_engine_reads_sq8_keys(engine, false, false).is_ok());
            let e = check_engine_reads_sq8_keys(engine, true, false).unwrap_err();
            assert!(e.contains("COMPRESSION") && e.contains(engine), "{e}");
            let e = check_engine_reads_sq8_keys(engine, false, true).unwrap_err();
            assert!(e.contains("TRAINING_THRESHOLD"), "{e}");
            assert!(!e.contains("COMPRESSION"), "{e}");
        }
    }

    #[test]
    fn env_applies_only_to_hnsw_with_a_vector_index() {
        assert!(env_applies("hnsw", false));
        assert!(env_applies("HNSW", false));
        for (algorithm, skip) in [
            ("flat", false),
            ("FLAT", false),
            ("svs-vamana", false),
            ("SVS-VAMANA", false),
            ("hnsw", true),
        ] {
            assert!(!env_applies(algorithm, skip), "{algorithm} skip={skip}");
        }
    }

    /// The `REDIS_HNSW_*` env must never reach a FLAT, SVS-VAMANA or
    /// no-vector config: the readers are not even called, and an env value
    /// that would be invalid there (SQ8 on FLAT) does not error.
    #[test]
    fn resolve_never_reads_env_outside_hnsw_with_a_vector_index() {
        for (algorithm, skip) in [
            ("flat", false),
            ("svs-vamana", false),
            ("SVS-VAMANA", false),
            ("hnsw", true),
        ] {
            let r = Sq8Options::resolve(
                None,
                None,
                || panic!("REDIS_HNSW_COMPRESSION read for {algorithm} skip={skip}"),
                || panic!("REDIS_HNSW_TRAINING_THRESHOLD read for {algorithm} skip={skip}"),
                algorithm,
                "FLOAT32",
                skip,
            )
            .unwrap();
            assert_eq!(r, Sq8Resolution::default(), "{algorithm} skip={skip}");
            let r = Sq8Options::resolve(
                None,
                None,
                || Some("SQ8".into()),
                || Some("4".into()),
                algorithm,
                "FLOAT32",
                skip,
            )
            .unwrap();
            assert_eq!(r, Sq8Resolution::default(), "{algorithm} skip={skip}");
        }
        // The same env does apply to an HNSW vector index.
        let r = resolve_full((None, None), (Some("SQ8"), Some("4"))).unwrap();
        assert_eq!(r.options.compression.as_deref(), Some("SQ8"));
        assert_eq!(r.options.training_threshold, Some(4));
    }

    #[test]
    fn resolve_reports_which_fields_came_from_env() {
        let r = resolve_full((None, None), (Some("SQ8"), Some("4"))).unwrap();
        assert_eq!(
            r.from_env,
            ["REDIS_HNSW_COMPRESSION", "REDIS_HNSW_TRAINING_THRESHOLD"]
        );
        let notice = r.env_notice("redis-m-16-ef-128").unwrap();
        assert!(!notice.contains('\n'), "{notice}");
        assert!(notice.contains("'redis-m-16-ef-128'"), "{notice}");
        assert!(
            notice.contains("COMPRESSION SQ8 from REDIS_HNSW_COMPRESSION"),
            "{notice}"
        );
        assert!(
            notice.contains("TRAINING_THRESHOLD 4 from REDIS_HNSW_TRAINING_THRESHOLD"),
            "{notice}"
        );

        let r = resolve_full((Some("SQ8"), None), (None, Some("4"))).unwrap();
        assert_eq!(r.from_env, ["REDIS_HNSW_TRAINING_THRESHOLD"]);
        let notice = r.env_notice("c").unwrap();
        assert!(!notice.contains("REDIS_HNSW_COMPRESSION"), "{notice}");

        let r = resolve_full((None, Some(7)), (Some("SQ8"), Some("4"))).unwrap();
        assert_eq!(r.from_env, ["REDIS_HNSW_COMPRESSION"]);

        // Config declares both: env is unused, nothing to report.
        let r = resolve_full((Some("SQ8"), Some(0)), (Some("SQ8"), Some("4"))).unwrap();
        assert!(r.from_env.is_empty());
        assert_eq!(r.env_notice("c"), None);
        let r = resolve_full((None, None), (None, None)).unwrap();
        assert_eq!(r.env_notice("c"), None);
    }

    /// What the result metadata says was built, for each way the options can
    /// be resolved. Goes through `resolve`, so an env-built SQ8 index on a
    /// plain config must be recorded as SQ8.
    #[test]
    fn effective_record_reflects_what_was_built() {
        use serde_json::{json, Value};
        let record = |config: (Option<&str>, Option<u64>), env: (Option<&str>, Option<&str>)| {
            resolve(config, env).unwrap().effective_record()
        };
        // Plain config, no env.
        assert_eq!(
            record((None, None), (None, None)),
            (Value::Null, Value::Null)
        );
        // Config SQ8 without a threshold: the server default.
        assert_eq!(
            record((Some("SQ8"), None), (None, None)),
            (json!("SQ8"), json!(SERVER_DEFAULT_TRAINING_THRESHOLD))
        );
        assert_eq!(record((Some("sq8"), None), (None, None)).1, json!(10_240));
        // Config SQ8 with threshold 0: the declared zero, not the default.
        assert_eq!(
            record((Some("SQ8"), Some(0)), (None, None)),
            (json!("SQ8"), json!(0))
        );
        // Env compression on a plain config: what was built, not declared.
        assert_eq!(
            record((None, None), (Some("SQ8"), None)),
            (json!("SQ8"), json!(10_240))
        );
        // Env threshold with config compression.
        assert_eq!(
            record((Some("SQ8"), None), (None, Some("8"))),
            (json!("SQ8"), json!(8))
        );
    }

    #[test]
    fn effective_threshold_is_the_server_default_only_with_compression() {
        assert_eq!(SERVER_DEFAULT_TRAINING_THRESHOLD, 10_240);
        let o = Sq8Options::from_hnsw_fields(Some("SQ8"), None, "hnsw", "FLOAT32", false).unwrap();
        assert_eq!(o.effective_training_threshold(), Some(10_240));
        let o =
            Sq8Options::from_hnsw_fields(Some("SQ8"), Some(0), "hnsw", "FLOAT32", false).unwrap();
        assert_eq!(o.effective_training_threshold(), Some(0));
        assert_eq!(Sq8Options::default().effective_training_threshold(), None);
    }

    #[test]
    fn resolve_config_wins_per_field() {
        let o = resolve((Some("SQ8"), Some(7)), (Some("SQ8"), Some("4"))).unwrap();
        assert_eq!(o.training_threshold, Some(7));
        // Config threshold with env-only compression: fields merge independently.
        let o = resolve((None, Some(7)), (Some("SQ8"), Some("4"))).unwrap();
        assert_eq!(o.compression.as_deref(), Some("SQ8"));
        assert_eq!(o.training_threshold, Some(7));
        // Config compression with env-only threshold.
        let o = resolve((Some("SQ8"), None), (None, Some("4"))).unwrap();
        assert_eq!(o.training_threshold, Some(4));
        // Config zero is a declared value and beats a nonzero env threshold.
        let o = resolve((Some("SQ8"), Some(0)), (Some("SQ8"), Some("4"))).unwrap();
        assert_eq!(o.training_threshold, Some(0));
    }

    #[test]
    fn resolve_env_fills_missing_and_normalizes() {
        let o = resolve((None, None), (Some(" sq8 "), Some(" 4 "))).unwrap();
        assert_eq!(o.compression.as_deref(), Some("SQ8"));
        assert_eq!(o.training_threshold, Some(4));
    }

    #[test]
    fn resolve_blank_env_is_unset() {
        assert_eq!(
            resolve((None, None), (Some(""), Some("  "))).unwrap(),
            Sq8Options::default()
        );
    }

    #[test]
    fn resolve_env_threshold_requires_compression() {
        for env_compression in [None, Some("  ")] {
            let e = resolve((None, None), (env_compression, Some("4"))).unwrap_err();
            assert!(e.starts_with("REDIS_HNSW_TRAINING_THRESHOLD"), "{e}");
        }
    }

    #[test]
    fn resolve_env_threshold_over_range_names_the_variable() {
        let e = resolve((Some("SQ8"), None), (None, Some("999999"))).unwrap_err();
        assert!(e.starts_with("REDIS_HNSW_TRAINING_THRESHOLD"), "{e}");
        assert!(resolve((Some("SQ8"), None), (None, Some("102400"))).is_ok());
    }

    #[test]
    fn resolve_env_compression_dtype_error_names_the_variable() {
        let e = Sq8Options::resolve(
            None,
            None,
            || Some("SQ8".into()),
            || None,
            "hnsw",
            "INT8",
            false,
        )
        .unwrap_err();
        assert!(e.contains("REDIS_HNSW_COMPRESSION"), "{e}");
    }

    #[test]
    fn resolve_bad_env_names_the_variable() {
        let e = resolve((Some("SQ8"), None), (None, Some("abc"))).unwrap_err();
        assert_eq!(
            e,
            "REDIS_HNSW_TRAINING_THRESHOLD must be an integer from 0 to 102400, got \"abc\""
        );
        let e = resolve((None, None), (Some("SQ4"), None)).unwrap_err();
        assert!(e.contains("REDIS_HNSW_COMPRESSION"), "{e}");
    }

    /// A server that ignored or changed COMPRESSION must fail the read-back,
    /// and compression without a declared threshold must expect the server
    /// default.
    #[test]
    fn read_back_rejects_wrong_compression_and_expects_default_threshold() {
        let s = |s: &str| RedisValue::BulkString(s.as_bytes().to_vec());
        let info = |fields: &[&str]| {
            let mut attr = vec![s("identifier"), s("vector")];
            attr.extend(fields.iter().map(|f| s(f)));
            RedisValue::Array(vec![
                s("attributes"),
                RedisValue::Array(vec![RedisValue::Array(attr)]),
            ])
        };
        let sq8 =
            Sq8Options::from_hnsw_fields(Some("SQ8"), Some(0), "hnsw", "FLOAT32", false).unwrap();
        // Compression absent (accepted and ignored) or different.
        let e = sq8
            .verify_ft_info(&info(&["training_threshold", "0"]))
            .unwrap_err();
        assert!(e.contains("compression"), "{e}");
        let e = sq8
            .verify_ft_info(&info(&["compression", "LVQ8", "training_threshold", "0"]))
            .unwrap_err();
        assert!(e.contains("compression"), "{e}");
        // No declared threshold: FT.INFO must report the server default.
        let default =
            Sq8Options::from_hnsw_fields(Some("SQ8"), None, "hnsw", "FLOAT32", false).unwrap();
        assert!(default
            .verify_ft_info(&info(&[
                "compression",
                "SQ8",
                "training_threshold",
                "10240"
            ]))
            .is_ok());
        assert!(default
            .verify_ft_info(&info(&["compression", "SQ8", "training_threshold", "0"]))
            .is_err());
    }

    #[test]
    fn requires_server_read_back() {
        let options =
            Sq8Options::from_hnsw_fields(Some("SQ8"), Some(0), "hnsw", "FLOAT32", false).unwrap();
        let s = |s: &str| RedisValue::BulkString(s.as_bytes().to_vec());
        let info = RedisValue::Array(vec![
            s("attributes"),
            RedisValue::Array(vec![RedisValue::Array(vec![
                s("identifier"),
                s("vector"),
                s("compression"),
                s("SQ8"),
                s("training_threshold"),
                s("0"),
            ])]),
        ]);
        assert!(options.verify_ft_info(&info).is_ok());
        let wrong = RedisValue::Array(vec![
            s("attributes"),
            RedisValue::Array(vec![RedisValue::Array(vec![
                s("identifier"),
                s("vector"),
                s("compression"),
                s("SQ8"),
                s("training_threshold"),
                s("10240"),
            ])]),
        ]);
        assert!(options.verify_ft_info(&wrong).is_err());
    }
}
