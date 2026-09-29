//! Redis HNSW SQ8 options and server read-back for benchmark fidelity.

use redis::Value as RedisValue;
use serde::{Deserialize, Deserializer};

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
            if n > 102_400 {
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
    /// `from_hnsw_fields`. Env text is trimmed; empty or whitespace-only env is
    /// treated as unset. Invalid env text errors naming the variable.
    pub fn resolve(
        config_compression: Option<&str>,
        config_training_threshold: Option<u64>,
        env_compression: Option<String>,
        env_training_threshold: Option<String>,
        algorithm: &str,
        data_type: &str,
        skip_vector_index: bool,
    ) -> Result<Self, String> {
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
        let compression = match (config_compression, env_compression) {
            (Some(c), _) => Some(c),
            (None, Some(e)) if e.eq_ignore_ascii_case("SQ8") => Some("SQ8"),
            (None, Some(e)) => {
                return Err(format!("REDIS_HNSW_COMPRESSION must be SQ8, got {e:?}"));
            }
            (None, None) => None,
        };
        let training_threshold = match (config_training_threshold, env_training_threshold) {
            (Some(n), _) => Some(n),
            (None, Some(e)) => {
                let n = e.parse::<u64>().ok().filter(|&n| n <= 102_400).ok_or_else(|| {
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
        Self::from_hnsw_fields(
            compression,
            training_threshold,
            algorithm,
            data_type,
            skip_vector_index,
        )
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
        let expected_threshold = self.training_threshold.unwrap_or(10_240);
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

    fn resolve(
        config: (Option<&str>, Option<u64>),
        env: (Option<&str>, Option<&str>),
    ) -> Result<Sq8Options, String> {
        Sq8Options::resolve(
            config.0,
            config.1,
            env.0.map(String::from),
            env.1.map(String::from),
            "hnsw",
            "FLOAT32",
            false,
        )
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
    fn resolve_bad_env_names_the_variable() {
        let e = resolve((Some("SQ8"), None), (None, Some("abc"))).unwrap_err();
        assert_eq!(
            e,
            "REDIS_HNSW_TRAINING_THRESHOLD must be an integer from 0 to 102400, got \"abc\""
        );
        let e = resolve((None, None), (Some("SQ4"), None)).unwrap_err();
        assert!(e.contains("REDIS_HNSW_COMPRESSION"), "{e}");
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
