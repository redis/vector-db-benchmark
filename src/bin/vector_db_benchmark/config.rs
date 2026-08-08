//! Configuration loading for datasets and engines.
//!
//! Reads datasets.json and experiments/configurations/*.json

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Dataset configuration from datasets.json
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub dataset_type: Option<String>,
    pub path: serde_json::Value,
    pub distance: Option<String>,
    pub vector_size: Option<i64>,
    pub vector_count: Option<i64>,
    pub link: Option<String>,
    pub schema: Option<serde_json::Value>,
    pub description: Option<String>,
}

/// HNSW configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct HnswConfig {
    #[serde(rename = "M", alias = "m")]
    pub m: Option<i64>,
    #[serde(
        rename = "EF_CONSTRUCTION",
        alias = "ef_construct",
        alias = "ef_construction"
    )]
    pub ef_construction: Option<i64>,
    /// Qdrant: keep the HNSW graph on disk (mmap) instead of in RAM.
    pub on_disk: Option<bool>,
    /// Qdrant: per-payload-value graph links. `m: 0` + `payload_m: k` builds
    /// graphs only per tenant/payload value — the multi-tenancy layout.
    pub payload_m: Option<i64>,
    /// Qdrant: store vectors inline with the graph links (on-disk locality).
    pub inline_storage: Option<bool>,
    /// Qdrant: below this many points a filtered search full-scans instead of
    /// traversing the graph. Decisive for filtered / multi-tenant benchmarks.
    pub full_scan_threshold: Option<i64>,
    /// Qdrant: threads used to build the graph.
    pub max_indexing_threads: Option<i64>,
    /// Catch-all so an unrecognised key inside `hnsw_config` can be REPORTED
    /// rather than silently discarded. Serde ignores undeclared fields by
    /// default, which is how `on_disk` used to vanish here — the very bug this
    /// branch fixes.
    ///
    /// Only the Qdrant engine calls `unsupported_keys()` today. The other five
    /// consumers (redis, valkey, pgvector, dragonfly, kividb) read just
    /// `m`/`ef_construction`, and at least one shipped config deliberately parks
    /// an inert key here (`redis-single-node.json`'s `DISTANCE_METRIC`, which
    /// redis derives from the dataset), so warning for them would be noise.
    #[serde(flatten)]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

impl HnswConfig {
    /// Keys inside `hnsw_config` that no engine reads. Returned so the caller can
    /// warn: a typo'd or unsupported knob must not pass for a configured one.
    pub fn unsupported_keys(&self) -> Vec<&str> {
        self.extra
            .as_ref()
            .map(|e| e.keys().map(|k| k.as_str()).collect())
            .unwrap_or_default()
    }
}

/// Elasticsearch index_options (lowercase keys)
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct IndexOptions {
    pub m: Option<i64>,
    pub ef_construction: Option<i64>,
}

/// Collection parameters
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct CollectionParams {
    pub hnsw_config: Option<HnswConfig>,
    pub index_options: Option<IndexOptions>,
    /// Catch-all for engine-specific collection params (e.g., OpenSearch "method")
    #[serde(flatten)]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

/// Search parameters for a single search configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchParams {
    pub parallel: Option<i64>,
    /// Per-search engine knobs. Upstream (qdrant/vector-db-benchmark) spells this
    /// key `config`; accept BOTH so an upstream configuration file can be used
    /// verbatim. Without the alias, `"config": {...}` was absorbed by the
    /// `extra` catch-all below and silently dropped — the run then used default
    /// `ef`/`hnsw_ef` while reporting as if it had been tuned.
    #[serde(alias = "config")]
    pub search_params: Option<InnerSearchParams>,
    pub top: Option<i64>,
    pub num_candidates: Option<i64>,
    /// Fixed offered rate for an open-loop run. None keeps closed-loop behavior.
    pub target_qps: Option<f64>,
    /// Open-loop measurement duration. Queries recycle when the dataset is shorter.
    pub duration_seconds: Option<f64>,
    /// Requests dispatched later than this are dropped and counted.
    pub max_lateness_ms: Option<f64>,
    /// Calibration: name of the search param to tune (e.g., "ef")
    pub calibration_param: Option<String>,
    /// Calibration: target precision to achieve
    pub calibration_precision: Option<f64>,
    /// Catch-all for engine-specific search params (e.g., OpenSearch "knn.algo_param.ef_search")
    #[serde(flatten)]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InnerSearchParams {
    /// HNSW search-time breadth. Upstream's redis configs spell it `EF`
    /// (uppercase, matching the FT.SEARCH runtime attribute), so accept both.
    #[serde(alias = "EF")]
    pub ef: Option<i64>,
    /// Catch-all for additional search params (e.g., SEARCH_WINDOW_SIZE, data_type)
    #[serde(flatten)]
    pub extra: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl SearchParams {
    /// Resolve an UNTYPED engine-specific search knob by name, accepting it
    /// either nested under `search_params` / `config` or flat at the entry's top
    /// level. Nested wins, being the more specific placement.
    ///
    /// Upstream nests EVERY knob under `config` (`num_candidates`,
    /// `knn.algo_param.ef_search`, `hnsw_ef`, …) while several of our own
    /// configurations put them flat. Engines resolve through here so both shapes
    /// work and neither is silently ignored.
    ///
    /// IMPORTANT — this searches only the two `extra` catch-alls, so it does NOT
    /// see a key that serde captured into a TYPED field. That covers `ef` and
    /// `EF` on the inner struct, and `parallel`, `top`, `num_candidates`,
    /// `target_qps`, `duration_seconds`, `max_lateness_ms`, `calibration_param`
    /// and `calibration_precision` on this one. For those, `knob()` sees only the
    /// placement serde did NOT claim, so a caller must combine both — e.g.
    /// Elasticsearch does `knob("num_candidates").or(params.num_candidates)`, and
    /// engines read `search_params.ef` directly rather than `knob("ef")` (which
    /// would find a flat `ef` while ignoring the nested one that configures the
    /// run — hence the debug assertion below).
    /// Canonicalise a search-knob NAME onto the field it actually sets.
    ///
    /// `calibration_param` names its knob by string. Every alias of the typed
    /// `ef` field must collapse onto `"ef"`, otherwise the calibration loop
    /// writes `search_params.extra["EF"]` — a key NO engine reads (they read
    /// `search_params.ef`) — so the sweep tunes a knob that is never applied and
    /// then reports the calibrated value as if it had been.
    pub fn canonical_knob_name(name: &str) -> &str {
        match name {
            "ef" | "EF" => "ef",
            other => other,
        }
    }

    pub fn knob(&self, key: &str) -> Option<&serde_json::Value> {
        debug_assert!(
            !matches!(
                key,
                "ef" | "EF"
                    | "parallel"
                    | "top"
                    | "target_qps"
                    | "duration_seconds"
                    | "max_lateness_ms"
                    | "calibration_param"
                    | "calibration_precision"
            ),
            "{key:?} is a typed field; knob() cannot see it — read the field directly \
             (num_candidates is exempt: its call sites combine both, see the doc above)"
        );
        self.search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get(key))
            .or_else(|| self.extra.as_ref().and_then(|e| e.get(key)))
    }
}

/// Engine configuration from experiments/configurations/*.json
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineConfig {
    pub name: String,
    pub engine: Option<String>,
    pub algorithm: Option<String>,
    pub connection_params: Option<serde_json::Value>,
    pub collection_params: Option<CollectionParams>,
    pub search_params: Option<Vec<SearchParams>>,
    pub upload_params: Option<serde_json::Value>,
    /// When true, vectors are uploaded but not indexed; search is filter-only.
    #[serde(default)]
    pub skip_vector_index: bool,
}

/// Get the project root directory
pub fn project_root() -> PathBuf {
    let current = std::env::current_dir().unwrap_or_default();

    // Look for datasets/datasets.json to verify we're in the right place
    if current.join("datasets/datasets.json").exists() {
        return current;
    }

    // Try parent directories
    let mut search = current.clone();
    for _ in 0..5 {
        if let Some(parent) = search.parent() {
            search = parent.to_path_buf();
            if search.join("datasets/datasets.json").exists() {
                return search;
            }
        }
    }

    current
}

/// Get datasets directory path
pub fn datasets_dir() -> PathBuf {
    project_root().join("datasets")
}

/// Read all dataset configurations
pub fn read_dataset_configs() -> Result<HashMap<String, DatasetConfig>, String> {
    let datasets_json = project_root().join("datasets/datasets.json");
    let content = fs::read_to_string(&datasets_json)
        .map_err(|e| format!("Failed to read datasets.json at {:?}: {}", datasets_json, e))?;

    let configs: Vec<DatasetConfig> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse datasets.json: {}", e))?;

    let mut map = HashMap::new();
    for config in configs {
        map.insert(config.name.clone(), config);
    }
    Ok(map)
}

/// Read all engine configurations from experiments/configurations/*.json
/// Read engine configs. When `engines_file` is `Some`, ONLY that JSON file is
/// read (the `--engines-file` flag); otherwise every
/// `experiments/configurations/*.json` is globbed. A `--engines-file` that is
/// missing or malformed is a hard error (the previous glob-only behavior
/// silently ignored the flag, so `--engines-file x.json` failed with a
/// confusing "no engines match" — see issue #151).
pub fn read_engine_configs(
    engines_file: Option<&str>,
) -> Result<HashMap<String, EngineConfig>, String> {
    let mut all_configs = HashMap::new();

    if let Some(file) = engines_file {
        let content = fs::read_to_string(file)
            .map_err(|e| format!("failed to read --engines-file {}: {}", file, e))?;
        let configs: Vec<EngineConfig> = serde_json::from_str(&content)
            .map_err(|e| format!("invalid JSON in --engines-file {}: {}", file, e))?;
        for config in configs {
            all_configs.insert(config.name.clone(), config);
        }
        return Ok(all_configs);
    }

    let configs_dir = project_root().join("experiments/configurations");
    let pattern = configs_dir.join("*.json");
    for path in glob::glob(pattern.to_str().unwrap())
        .map_err(|e| e.to_string())?
        .flatten()
    {
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: skipping engine config {:?}: {}", path, e);
                continue;
            }
        };
        // NEVER swallow the parse error. serde rejects the WHOLE file on one bad
        // entry, so a single typo deletes every engine defined in it and the run
        // fails with a baffling "no engines match" — e.g. a typo anywhere in
        // qdrant-on-disk.json removes all four of its configurations. The typed
        // fields and aliases on this branch make that easy to trip:
        //   hnsw_config: {"on_disk": "true"}          -> invalid type: string
        //   {"search_params": {...}, "config": {...}} -> duplicate field
        //   {"config": {"ef": 64, "EF": 512}}         -> duplicate field
        //   {"config": 5}                             -> invalid type
        match serde_json::from_str::<Vec<EngineConfig>>(&content) {
            Ok(configs) => {
                for config in configs {
                    all_configs.insert(config.name.clone(), config);
                }
            }
            Err(e) => eprintln!(
                "Warning: engine config {:?} does not parse, so ALL of its entries were \
                 skipped: {}",
                path, e
            ),
        }
    }
    Ok(all_configs)
}

/// Match a name against a pattern (supports * wildcard)
pub fn matches_pattern(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    glob::Pattern::new(pattern)
        .map(|p| p.matches(name))
        .unwrap_or(false)
}

/// Format a vector count with K/M/B suffixes
fn format_count(count: i64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// Summarize a schema value into a compact string
fn format_schema(schema: &serde_json::Value, max_len: usize) -> String {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return schema.to_string(),
    };
    let field_count = obj.len();
    let base = if field_count == 1 {
        "1 field".to_string()
    } else {
        format!("{} fields", field_count)
    };
    if field_count == 0 {
        return base;
    }
    // Try to add detail
    let detail = if field_count <= 2 {
        let names: Vec<&String> = obj.keys().collect();
        format!(
            "{}: {}",
            base,
            names
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        let mut types: Vec<String> = obj
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        types.sort();
        types.dedup();
        format!("{} ({})", base, types.join(", "))
    };
    if detail.len() <= max_len {
        detail
    } else {
        base
    }
}

/// Describe available datasets
pub fn describe_datasets(verbose: bool) -> Result<(), String> {
    let configs = read_dataset_configs()?;

    // Sort by dimension, then vector count, then name
    let mut sorted: Vec<(&String, &DatasetConfig)> = configs.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| {
        let dim_a = a.vector_size.unwrap_or(0);
        let dim_b = b.vector_size.unwrap_or(0);
        dim_a
            .cmp(&dim_b)
            .then_with(|| {
                a.vector_count
                    .unwrap_or(0)
                    .cmp(&b.vector_count.unwrap_or(0))
            })
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    println!("\nAvailable Datasets ({} found)", configs.len());
    println!("{}", "=".repeat(131));

    if verbose {
        for (name, config) in &sorted {
            println!("\n  {}", name);
            println!(
                "   Vector Size: {}",
                config
                    .vector_size
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "N/A".into())
            );
            println!(
                "   Distance:    {}",
                config.distance.as_deref().unwrap_or("N/A")
            );
            println!(
                "   Type:        {}",
                config.dataset_type.as_deref().unwrap_or("N/A")
            );
            if let serde_json::Value::String(p) = &config.path {
                println!("   Path:        {}", p);
            }
            if let Some(link) = &config.link {
                println!("   Download:    {}", link);
            }
            if let Some(desc) = &config.description {
                println!("   Description: {}", desc);
            }
            if let Some(schema) = &config.schema {
                println!("   Schema:      {}", schema);
            }
        }
    } else {
        // Column widths: Name(35) Dims(6) Distance(10) Count(14) Description(30) Schema(20)
        println!(
            "{:<35}{:<6}{:<10}{:<14}{:<30}{:<20}",
            "Dataset Name", "Dims", "Distance", "Vector Count", "Description", "Schema"
        );
        println!("{}", "-".repeat(115));

        for (name, config) in &sorted {
            let dims = config
                .vector_size
                .map(|v| v.to_string())
                .unwrap_or_else(|| "N/A".into());
            let distance = config.distance.as_deref().unwrap_or("N/A");
            let count_str = config
                .vector_count
                .map(format_count)
                .unwrap_or_else(|| "N/A".into());
            let desc = config.description.as_deref().unwrap_or("");
            let desc_display = if desc.len() > 29 {
                format!("{}...", &desc[..26])
            } else {
                desc.to_string()
            };
            let schema_str = config
                .schema
                .as_ref()
                .map(|s| format_schema(s, 19))
                .unwrap_or_default();

            let display_name = if name.len() > 34 {
                format!("{}...", &name[..31])
            } else {
                name.to_string()
            };

            println!(
                "{:<35}{:<6}{:<10}{:<14}{:<30}{:<20}",
                display_name, dims, distance, count_str, desc_display, schema_str
            );
        }
    }

    println!("\nTotal: {} datasets", configs.len());
    if verbose {
        println!();
    } else {
        println!("Use --verbose for detailed information");
    }
    Ok(())
}

/// Describe available engines
pub fn describe_engines(verbose: bool) -> Result<(), String> {
    let configs = read_engine_configs(None)?;
    println!("Available engines ({}):", configs.len());
    for (name, config) in configs.iter() {
        if verbose {
            println!(
                "  {} - engine: {:?}, algorithm: {:?}",
                name, config.engine, config.algorithm
            );
        } else {
            println!("  {}", name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_pattern_exact_and_star() {
        assert!(matches_pattern("redis", "redis"));
        assert!(matches_pattern("anything-at-all", "*"));
        assert!(!matches_pattern("redis", "qdrant"));
    }

    // #151: `--engines-file` must actually read the given file (it was a silent
    // no-op). A missing/malformed file is a hard error, not a fallback to glob.
    #[test]
    fn engines_file_is_read_and_errors_are_hard() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        write!(
            f,
            r#"[{{"name":"my-cfg","engine":"redis","search_params":[{{"parallel":1}}]}}]"#
        )
        .unwrap();
        f.flush().unwrap();
        let configs = read_engine_configs(Some(f.path().to_str().unwrap())).unwrap();
        assert!(configs.contains_key("my-cfg"));
        assert_eq!(configs["my-cfg"].engine.as_deref(), Some("redis"));

        // Missing file → hard error (not a silent fall-through to the glob).
        assert!(read_engine_configs(Some("/no/such/file.json")).is_err());
    }

    // Upstream spells the per-search knob object `config`. Before the alias it
    // landed in the flattened `extra` catch-all and was silently dropped, so a
    // verbatim upstream file ran with default ef while REPORTING as tuned.
    #[test]
    fn upstream_config_key_is_accepted_as_search_params() {
        let sp: SearchParams =
            serde_json::from_value(json!({"parallel": 8, "config": {"hnsw_ef": 128}})).unwrap();
        assert_eq!(sp.parallel, Some(8));
        assert_eq!(
            sp.knob("hnsw_ef").and_then(|v| v.as_u64()),
            Some(128),
            "`config` must populate search_params, not the extra catch-all"
        );
        // And it must NOT also linger in `extra` under the raw key.
        assert!(sp.extra.is_none_or(|e| !e.contains_key("config")));
    }

    // Our own spelling keeps working, and uppercase `EF` (upstream's redis
    // configs) resolves to the same typed field as lowercase `ef`.
    #[test]
    fn our_search_params_key_and_uppercase_ef_still_work() {
        let ours: SearchParams =
            serde_json::from_value(json!({"parallel": 1, "search_params": {"ef": 64}})).unwrap();
        assert_eq!(
            ours.search_params.as_ref().and_then(|sp| sp.ef),
            Some(64),
            "our historical `search_params` spelling must keep working"
        );

        let upstream_redis: SearchParams =
            serde_json::from_value(json!({"parallel": 100, "config": {"EF": 512}})).unwrap();
        assert_eq!(
            upstream_redis.search_params.as_ref().and_then(|sp| sp.ef),
            Some(512),
            "upstream redis `EF` must map onto the typed ef field"
        );
    }

    // Engines read knobs through SearchParams::knob, which accepts either
    // placement. Nested is more specific and therefore wins.
    #[test]
    fn knob_resolves_nested_and_flat_with_nested_winning() {
        let nested: SearchParams =
            serde_json::from_value(json!({"config": {"num_candidates": 16}})).unwrap();
        assert_eq!(
            nested.knob("num_candidates").and_then(|v| v.as_i64()),
            Some(16)
        );

        let flat: SearchParams =
            serde_json::from_value(json!({"knn.algo_param.ef_search": 256})).unwrap();
        assert_eq!(
            flat.knob("knn.algo_param.ef_search")
                .and_then(|v| v.as_i64()),
            Some(256)
        );

        let both: SearchParams =
            serde_json::from_value(json!({"num_candidates_x": 1, "config": {"k": 2}, "k": 3}))
                .unwrap();
        assert_eq!(both.knob("k").and_then(|v| v.as_i64()), Some(2));

        let neither: SearchParams = serde_json::from_value(json!({"parallel": 1})).unwrap();
        assert!(neither.knob("k").is_none());
    }

    /// `knob()` searches only the `extra` catch-alls, so it is BLIND to a key
    /// serde captured into a typed field. Pinning that contract: a caller must
    /// combine `knob()` with the typed field (as Elasticsearch does) — relying on
    /// `knob()` alone for a declared name silently yields the default.
    #[test]
    fn knob_is_blind_to_typed_fields() {
        // Flat `num_candidates` is a DECLARED field on SearchParams.
        let flat: SearchParams =
            serde_json::from_value(json!({"num_candidates": 16, "parallel": 1})).unwrap();
        assert_eq!(flat.num_candidates, Some(16));
        assert!(
            flat.knob("num_candidates").is_none(),
            "knob() must not be expected to see a typed field"
        );

        // Nested `ef` is declared on InnerSearchParams, so it lands there.
        let nested_ef: SearchParams =
            serde_json::from_value(json!({"config": {"ef": 64}})).unwrap();
        assert_eq!(
            nested_ef.search_params.as_ref().and_then(|sp| sp.ef),
            Some(64)
        );
    }

    // `search_params` and `config` are the SAME field under an alias, so an entry
    // carrying both is a serde duplicate-field error rather than one silently
    // winning. Pin it: if serde ever tolerated it, the losing half would vanish.
    #[test]
    fn both_search_params_and_config_is_a_duplicate_field_error() {
        let err = serde_json::from_value::<SearchParams>(
            json!({"search_params": {"ef": 64}, "config": {"ef": 128}}),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("duplicate field"),
            "expected a duplicate-field error, got: {err}"
        );

        // Same one level down: `ef` and its `EF` alias in one object.
        let err = serde_json::from_value::<SearchParams>(json!({"config": {"ef": 64, "EF": 512}}))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("duplicate field"),
            "expected a duplicate-field error, got: {err}"
        );
    }

    // `"config": null` parses fine and yields NO knobs — an untuned run that
    // looks configured. Documented here so the behaviour is at least known.
    #[test]
    fn null_config_parses_to_no_search_params() {
        let sp: SearchParams =
            serde_json::from_value(json!({"parallel": 4, "config": null})).unwrap();
        assert!(sp.search_params.is_none());
    }

    // Every shipped experiments/configurations/*.json MUST parse. Nothing else in
    // the suite reads the real directory (integration tests all write their own
    // temp config), so a typo — or a new typed field rejecting an existing value
    // — would otherwise only surface at run time as "no engines match", with
    // every entry in the offending file silently gone.
    #[test]
    fn every_shipped_engine_config_file_parses() {
        let dir = project_root().join("experiments/configurations");
        let pattern = dir.join("*.json");
        let mut seen = 0usize;
        for path in glob::glob(pattern.to_str().unwrap()).unwrap().flatten() {
            let content = fs::read_to_string(&path).unwrap();
            let parsed = serde_json::from_str::<Vec<EngineConfig>>(&content);
            assert!(
                parsed.is_ok(),
                "{:?} does not parse — ALL of its engine entries would silently \
                 disappear from every run: {}",
                path,
                parsed.unwrap_err()
            );
            seen += 1;
        }
        assert!(
            seen > 10,
            "expected the real config directory, found {seen} files"
        );
    }

    #[test]
    fn matches_pattern_glob_positions() {
        // prefix* / *suffix / mid*dle
        assert!(matches_pattern("redis-hnsw-m16", "redis*"));
        assert!(matches_pattern("redis-hnsw", "*hnsw"));
        assert!(matches_pattern("redis-m16-hnsw", "redis*hnsw"));
        // no-match cases
        assert!(!matches_pattern("qdrant-hnsw", "redis*"));
        assert!(!matches_pattern("redis-flat", "*hnsw"));
    }

    #[test]
    fn matches_pattern_invalid_pattern_is_false() {
        // An invalid glob (unclosed char class) → Pattern::new errors →
        // unwrap_or(false). Never panics, never matches.
        assert!(!matches_pattern("redis", "redis["));
    }

    #[test]
    fn format_count_magnitude_branches() {
        // < 1000 → raw integer string.
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        // >= 1000 → K.
        assert_eq!(format_count(1000), "1.0K");
        assert_eq!(format_count(1500), "1.5K");
        assert_eq!(format_count(999_999), "1000.0K");
        // >= 1_000_000 → M.
        assert_eq!(format_count(1_000_000), "1.0M");
        assert_eq!(format_count(2_500_000), "2.5M");
        // >= 1_000_000_000 → B.
        assert_eq!(format_count(1_000_000_000), "1.0B");
        assert_eq!(format_count(3_200_000_000), "3.2B");
        // Negative is < 1000 → raw string.
        assert_eq!(format_count(-42), "-42");
    }

    #[test]
    fn format_schema_non_object_uses_to_string() {
        assert_eq!(format_schema(&json!("hello"), 100), "\"hello\"");
        assert_eq!(format_schema(&json!(42), 100), "42");
    }

    #[test]
    fn format_schema_empty_object() {
        assert_eq!(format_schema(&json!({}), 100), "0 fields");
    }

    #[test]
    fn format_schema_one_and_two_fields_list_names() {
        // 1 field: singular "field" + name.
        assert_eq!(
            format_schema(&json!({"category": "keyword"}), 100),
            "1 field: category"
        );
        // 2 fields: names joined in the object's key order (serde_json's Map
        // preserves insertion order here, so "b" then "a" stays "b, a").
        assert_eq!(
            format_schema(&json!({"b": "int", "a": "keyword"}), 100),
            "2 fields: b, a"
        );
    }

    #[test]
    fn format_schema_three_plus_fields_lists_sorted_deduped_types() {
        // 3+ fields → "(types)" where the string-valued types are sorted+deduped.
        assert_eq!(
            format_schema(&json!({"a": "keyword", "b": "int", "c": "keyword"}), 100),
            "3 fields (int, keyword)"
        );
    }

    #[test]
    fn format_schema_falls_back_to_base_when_detail_too_long() {
        // detail "1 field: category" is 17 chars; with max_len 5 it exceeds the
        // budget → returns the bare base "1 field".
        assert_eq!(format_schema(&json!({"category": "keyword"}), 5), "1 field");
    }

    // `--describe datasets|engines` (smoke-tested by the docker-build job) drives
    // these two functions over the REAL registries. Asserting Ok pins that every
    // shipped datasets.json / engine config parses and that the formatting helpers
    // don't panic on any real entry — a regression the docker smoke would catch
    // but only in the non-blocking build job.
    #[test]
    fn describe_datasets_over_real_registry_is_ok() {
        assert!(
            describe_datasets(false).is_ok(),
            "compact --describe datasets"
        );
        assert!(
            describe_datasets(true).is_ok(),
            "verbose --describe datasets"
        );
    }

    #[test]
    fn describe_engines_over_real_registry_is_ok() {
        assert!(
            describe_engines(false).is_ok(),
            "compact --describe engines"
        );
        assert!(describe_engines(true).is_ok(), "verbose --describe engines");
    }
}
