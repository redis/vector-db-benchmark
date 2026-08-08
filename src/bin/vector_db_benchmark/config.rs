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

    /// Corpus size advertised by a dataset's own corpus filename, when the leaf
    /// path segment carries an unambiguous magnitude token — `random_keywords_1m`
    /// => 1_000_000, `..._100k` => 100_000, `...-1G-...` => 1_000_000_000. The
    /// LAST such token wins. `None` when the name says nothing about size.
    ///
    /// Every rejection is a `continue`, never a `?`: an unparseable token must
    /// skip that token, NOT abandon the whole path. A `?` here silently excused
    /// any path with a doubled or trailing separator (`random_keywords__1m`,
    /// `random_keywords_1m_`) from the check entirely — a guard that turns
    /// itself off. Token splitting is char-based for the same reason: a byte
    /// index into a multi-byte char panicked instead of declining.
    fn path_implied_corpus_size(path: &serde_json::Value) -> Option<i64> {
        let leaf = path.as_str()?.split('/').rfind(|s| !s.is_empty())?;
        let mut implied = None;
        for tok in leaf.split(['_', '-', '.']) {
            let mut chars = tok.chars();
            let Some(suffix) = chars.next_back() else {
                continue; // empty token (doubled/trailing separator)
            };
            let digits = chars.as_str();
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let scale: i64 = match suffix.to_ascii_lowercase() {
                'k' => 1_000,
                'm' => 1_000_000,
                'g' | 'b' => 1_000_000_000,
                _ => continue,
            };
            if let Some(size) = digits
                .parse::<i64>()
                .ok()
                .and_then(|n| n.checked_mul(scale))
            {
                implied = Some(size);
            }
        }
        implied
    }

    /// #224: `random-100-match-kw-small-vocab-*` declared `vector_count: 100`
    /// while pointing at `random_keywords_1m_vocab_10`, a 1,000,000-point corpus
    /// (copy-pasted from the genuine `random-100` entry above it). Redis's
    /// shared-corpus gate (#188) treats `vector_count` as "corpus is complete",
    /// so it skipped the upload after 100 keys and the sweep scored recall over
    /// 0.01% of the corpus — silently, with no error.
    ///
    /// This catches that class from `datasets.json` alone, with no corpus on
    /// disk: whenever a corpus path names its own size, the declared count must
    /// agree. (A dataset that is deliberately a subset of a named corpus should
    /// be given a path that reflects its real size.)
    ///
    /// Every mismatch is collected and reported in ONE assertion. Asserting
    /// inside the loop would report the first offender only, so a fix-rerun
    /// cycle is needed to discover its twin — and #224's two entries were
    /// exactly such a pair.
    #[test]
    fn declared_vector_count_agrees_with_the_size_its_path_advertises() {
        let configs = read_dataset_configs().expect("datasets.json must parse");
        let mut checked = 0;
        let mut mismatches = Vec::new();
        for (name, cfg) in &configs {
            let Some(implied) = path_implied_corpus_size(&cfg.path) else {
                continue;
            };
            checked += 1;
            if cfg.vector_count != Some(implied) {
                mismatches.push(format!(
                    "  dataset '{name}' declares vector_count {:?} but its corpus path '{}' \
                     advertises {implied} points",
                    cfg.vector_count,
                    cfg.path.as_str().unwrap_or_default(),
                ));
            }
        }
        mismatches.sort();
        assert!(
            mismatches.is_empty(),
            "{} dataset(s) declare a vector_count their corpus path contradicts (#224):\n{}",
            mismatches.len(),
            mismatches.join("\n"),
        );
        assert!(
            checked >= 30,
            "expected the size-in-path check to cover the bulk of datasets.json, covered {checked}"
        );
    }

    #[test]
    fn path_implied_corpus_size_only_fires_on_real_magnitude_tokens() {
        let s = |v: &str| serde_json::Value::String(v.to_string());
        // The #224 dataset: "1m" in the corpus name, "10" (vocab) must not win.
        assert_eq!(
            path_implied_corpus_size(&s(
                "random-100-match-kw-small-vocab/random_keywords_1m_vocab_10"
            )),
            Some(1_000_000)
        );
        assert_eq!(
            path_implied_corpus_size(&s("yandex-1B-200-angular/yandex_t2i_gt_100k")),
            Some(100_000),
            "only the leaf segment counts, not the '1B' family directory"
        );
        assert_eq!(
            path_implied_corpus_size(&s("laion-img-emb-768/laion-img-emb-768-1G-cosine.hdf5")),
            Some(1_000_000_000)
        );
        // Dimension counts and bare numbers say nothing about corpus size.
        assert_eq!(
            path_implied_corpus_size(&s("arxiv-titles-384-angular/arxiv_no_filters")),
            None
        );
        assert_eq!(path_implied_corpus_size(&s("random-100/")), None);
        assert_eq!(path_implied_corpus_size(&json!({"data": []})), None);
    }

    /// A guard that can be switched off by an odd separator is not a guard. An
    /// unparseable token must skip that TOKEN, not abandon the whole path —
    /// otherwise `random_keywords__1m` / `random_keywords_1m_` (doubled and
    /// trailing separator) drop out of
    /// `declared_vector_count_agrees_with_the_size_its_path_advertises`
    /// entirely and a deliberately wrong `vector_count` sails through CI.
    #[test]
    fn odd_separators_and_unicode_do_not_switch_the_check_off() {
        let s = |v: &str| serde_json::Value::String(v.to_string());
        for leaf in [
            "random-100-match-kw-small-vocab/random_keywords__1m",
            "random-100-match-kw-small-vocab/random_keywords_1m_",
            "random-100-match-kw-small-vocab/random__keywords_1m__vocab__10",
            "random-100-match-kw-small-vocab/-random_keywords_1m.",
        ] {
            assert_eq!(
                path_implied_corpus_size(&s(leaf)),
                Some(1_000_000),
                "empty tokens must be skipped, not abort the scan: {leaf}"
            );
        }
        // Multi-byte characters must decline cleanly, never panic on a byte
        // index that is not a char boundary.
        assert_eq!(
            path_implied_corpus_size(&s("random/random_keywords_1m_café")),
            Some(1_000_000)
        );
        assert_eq!(path_implied_corpus_size(&s("random/café")), None);
        assert_eq!(path_implied_corpus_size(&s("random/日本語")), None);
        // A magnitude token that would overflow i64 must not erase an earlier
        // valid one.
        assert_eq!(
            path_implied_corpus_size(&s("random/corpus_1m_99999999999999999999g")),
            Some(1_000_000)
        );
    }

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

/// Guard against the "declared but never read" config bug class (issue #216).
///
/// `experiments/configurations/mongodb-single-node.json` shipped a 12-config x
/// 8-point grid sweeping `collection_params.hnsw_config.{M,EF_CONSTRUCTION}` and
/// `search_params.ef`. The MongoDB engine read none of them, so all 96 rows were
/// the same measurement of one default index, published as a tradeoff curve.
///
/// Nothing caught it because both knobs *parse* perfectly: `hnsw_config` is
/// absorbed by the typed [`CollectionParams::hnsw_config`] field and `ef` by
/// [`InnerSearchParams::ef`], then silently dropped. `#[serde(deny_unknown_fields)]`
/// cannot catch this — serde applies it through `#[serde(flatten)]` too, which
/// would break the `extra` passthrough every engine depends on.
///
/// So instead of validating *parsing*, this guard checks that each knob a
/// shipped config declares at least *appears* in the production source of the
/// engine that entry targets — comments and `#[cfg(test)]` items stripped first,
/// so a stale doc comment or a test-only mention cannot vouch for a knob.
///
/// SCOPE — this is a useful net, not a durable defence. Read before trusting it.
///
/// What it reliably catches:
/// * a knob no source file of the target engine mentions at all (the #216 shape);
/// * a knob inside a [`TYPED_CONTAINERS`] struct that the struct does not
///   declare, on every engine — a real schema check, not a heuristic;
/// * a misspelled top-level key, which serde drops with everything under it;
/// * a mention that exists only in a comment or a `#[cfg(test)]` item.
///
/// What it does NOT catch, by construction:
/// * **wrong use.** A knob that is read and then overridden, clamped or ignored
///   downstream passes: its token is present. Issue #229's knobs are read and
///   then forced to a constant, and this guard is green on all of them.
/// * **positional mismatch.** It never checks that a knob is read from the block
///   it was declared in, so declaring an `upload_params` knob under
///   `collection_params` passes whenever the token appears anywhere.
/// * **short common leaves.** Token matching cannot distinguish a knob named
///   `ef` from an unrelated local `ef`. [`TYPED_CONTAINERS`] covers the known
///   typed structs; elsewhere this remains a hole.
///
/// Only a read-back assertion against a live server — see
/// `tests/integration_mongodb.rs` — proves a value actually took effect.
///
/// LOAD-BEARING: this walks the RAW JSON keys, never the typed structs and never
/// [`CollectionParams::extra`]. Do not "simplify" it to inspect `extra` — the
/// knobs behind issue #216 are *declared* fields, so serde consumes them into
/// [`CollectionParams`]/[`InnerSearchParams`] and they never reach the flattened
/// catch-all. An `extra`-based check reports MongoDB clean and misses all 96
/// rows. Typed-but-unread is the more dangerous half of this bug class, because
/// the key looks declared and parses without complaint.
#[cfg(test)]
mod shipped_config_knob_guard {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// Engine name (as dispatched in `engine::create_engine`) -> the source
    /// files that make up its read path.
    const ENGINE_SOURCES: &[(&str, &[&str])] = &[
        ("redis", &["redis.rs", "redis_utils.rs"]),
        ("vectorsets", &["vectorsets.rs"]),
        ("elasticsearch", &["elasticsearch.rs"]),
        ("opensearch", &["opensearch.rs"]),
        ("qdrant", &["qdrant.rs"]),
        ("weaviate", &["weaviate.rs", "weaviate_grpc.rs"]),
        ("pgvector", &["pgvector.rs"]),
        ("milvus", &["milvus.rs"]),
        ("mongodb", &["mongodb_engine.rs"]),
        ("valkey", &["valkey.rs"]),
        ("turbopuffer", &["turbopuffer.rs"]),
        ("dragonfly", &["dragonfly.rs"]),
        ("kividb", &["kividb.rs"]),
        ("vertex", &["vertex.rs", "vertex_grpc.rs"]),
        ("chroma", &["chroma.rs"]),
    ];

    /// Top-level `search_params[]` knobs consumed by the runner
    /// (`experiment.rs`) rather than by any engine, so they are legitimately
    /// absent from engine sources.
    ///
    /// Deliberately a fixed list of NAMES, not a grep of `experiment.rs`:
    /// `experiment.rs` contains the literal `"ef"` for calibration, so grepping
    /// it would have re-hidden exactly the `search_params.ef` bug this guard
    /// exists to catch.
    const FRAMEWORK_SEARCH_KNOBS: &[&str] = &[
        "parallel",
        "top",
        "target_qps",
        "duration_seconds",
        "max_lateness_ms",
        "calibration_param",
        "calibration_precision",
        // The inner `search_params` object itself is a container, not a knob.
        // `config` is #215's alias for that same container (upstream's spelling).
        "search_params",
        "config",
    ];

    /// Containers deserialized into a TYPED Rust struct, with the fields that
    /// struct actually declares. A key outside the set is not a typed knob: it
    /// either lands in an untyped `#[serde(flatten)] extra` map or is dropped
    /// outright, and in neither case does an engine read it as a knob.
    ///
    /// This is a schema check, and it is the only thing that reliably stops the
    /// issue #216 bug from being laundered by relocation: moving `ef` to
    /// `collection_params.hnsw_config.ef` defeats token matching on any engine
    /// whose source happens to contain a bare `ef` (qdrant, redis and pgvector
    /// all do), but it cannot defeat this.
    ///
    /// Keep in sync with the structs. [`HnswConfig`] gained `on_disk`,
    /// `payload_m`, `inline_storage`, `full_scan_threshold` and
    /// `max_indexing_threads` in #215; it also gained an `extra` catch-all, so an
    /// unlisted key there is *captured* rather than dropped — but only Qdrant
    /// calls `unsupported_keys()` on it, so for every other engine it is still
    /// silently inert.
    ///
    /// Leaves are compared after [`canonical`], so `M`/`EF_CONSTRUCTION` and
    /// their serde aliases all normalize into the sets below.
    const TYPED_CONTAINERS: &[(&str, &[&str])] = &[
        (
            "collection_params.hnsw_config",
            &[
                "m",
                "ef_construction",
                "on_disk",
                "payload_m",
                "inline_storage",
                "full_scan_threshold",
                "max_indexing_threads",
            ],
        ),
        ("collection_params.index_options", &["m", "ef_construction"]),
    ];

    /// Containers an engine forwards to the server WHOLESALE, as an opaque
    /// sub-object. Their leaves are honoured without ever being named in Rust,
    /// so leaf-token matching cannot see them.
    ///
    /// e.g. `weaviate.rs` pulls `collection_params.vectorIndexConfig` out of
    /// `extra` and merges every key of it into the class body, so
    /// `efConstruction`/`maxConnections` do reach the server.
    ///
    /// The container itself must still be read — that is asserted below, so an
    /// entry here cannot excuse a container the engine ignores entirely.
    const PASSTHROUGH_CONTAINERS: &[(&str, &str)] =
        &[("weaviate", "collection_params.vectorIndexConfig")];

    /// Knobs shipped configs declare that their engine genuinely does NOT read.
    /// Pre-existing debt this guard surfaced, listed rather than silently
    /// excused — and asserted to still be unread, so a fix must delete its entry
    /// instead of leaving a stale exemption behind.
    ///
    /// NOT a place to park new violations. Fix the engine or drop the key.
    const KNOWN_UNREAD: &[(&str, &str, &str)] = &[
        (
            "elasticsearch",
            "connection_params.request_timeout",
            "issue #245: engine takes its timeout from ELASTIC_TIMEOUT (default 300) \
             and never consults the config; the shipped value 10000 has no effect. \
             Units are ambiguous (s vs ms) and the two readings move behaviour in \
             opposite directions, so the fix needs a deliberate decision.",
        ),
        (
            "opensearch",
            "connection_params.request_timeout",
            "issue #245: same as elasticsearch, OPENSEARCH_TIMEOUT only.",
        ),
        // `qdrant collection_params.hnsw_config.on_disk` used to live here for
        // issue #215. #215 landed, HnswConfig gained the field and qdrant.rs
        // reads it, so the entry is gone — removed because the guard demanded it,
        // not because anyone remembered to look.
        (
            "redis",
            "collection_params.hnsw_config.DISTANCE_METRIC",
            "HnswConfig has no DISTANCE_METRIC field; the distance actually comes \
             from the dataset. Harmless but decorative - the key should be dropped \
             from the calibration configs in a follow-up.",
        ),
    ];

    /// Every field [`EngineConfig`] declares. A root key outside this set is a
    /// typo that serde silently discards along with everything under it.
    const ENGINE_CONFIG_ROOT_KEYS: &[&str] = &[
        "name",
        "engine",
        "algorithm",
        "connection_params",
        "collection_params",
        "search_params",
        "upload_params",
        "skip_vector_index",
    ];

    /// Resolve serde aliases to the field name the Rust source actually spells.
    /// A config may say `EF_CONSTRUCTION`, `ef_construct` or `ef_construction`;
    /// engines read `.ef_construction`.
    fn canonical(leaf: &str) -> &str {
        match leaf {
            "M" => "m",
            "EF_CONSTRUCTION" | "ef_construct" => "ef_construction",
            // #215 made `EF` an alias of the typed `ef` field.
            "EF" => "ef",
            other => other,
        }
    }

    /// Whole-token search: `ef` must not be satisfied by `ef_construction`.
    fn contains_token(haystack: &str, token: &str) -> bool {
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let bytes = haystack.as_bytes();
        let mut from = 0usize;
        while let Some(pos) = haystack[from..].find(token) {
            let start = from + pos;
            let end = start + token.len();
            let before_ok = start == 0 || !is_word(bytes[start - 1] as char);
            let after_ok = end >= bytes.len() || !is_word(bytes[end] as char);
            if before_ok && after_ok {
                return true;
            }
            from = start + 1;
        }
        false
    }

    /// Flatten a JSON object into dotted knob paths, recording each leaf.
    fn collect_knobs(value: &serde_json::Value, prefix: &str, out: &mut BTreeSet<String>) {
        if let Some(obj) = value.as_object() {
            for (key, child) in obj {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };
                out.insert(path.clone());
                collect_knobs(child, &path, out);
            }
        }
    }

    /// Remove `//`/`/* */` comments while preserving string literals.
    ///
    /// String contents must survive: engines read many knobs via
    /// `params.get("batch_size")`, so the literal IS the read. Comments must
    /// not: otherwise a refactor that deletes the forwarding code and leaves
    /// `// forwards hnsw_config` behind still satisfies the guard.
    fn strip_comments(src: &str) -> String {
        let c: Vec<char> = src.chars().collect();
        let mut out = String::with_capacity(src.len());
        let (mut i, mut block) = (0usize, 0usize);
        while i < c.len() {
            let (cur, next) = (c[i], c.get(i + 1).copied());
            if block > 0 {
                if cur == '/' && next == Some('*') {
                    block += 1;
                    i += 2;
                } else if cur == '*' && next == Some('/') {
                    block -= 1;
                    i += 2;
                } else {
                    if cur == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
                continue;
            }
            match cur {
                '/' if next == Some('/') => {
                    while i < c.len() && c[i] != '\n' {
                        i += 1;
                    }
                }
                '/' if next == Some('*') => {
                    block = 1;
                    i += 2;
                }
                '"' => {
                    out.push(cur);
                    i += 1;
                    while i < c.len() {
                        out.push(c[i]);
                        if c[i] == '\\' {
                            if let Some(&e) = c.get(i + 1) {
                                out.push(e);
                            }
                            i += 2;
                            continue;
                        }
                        if c[i] == '"' {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
                // Distinguish a char literal ('x', '\n') from a lifetime ('a).
                '\'' if c.get(i + 2) == Some(&'\'')
                    || (next == Some('\\') && c.get(i + 3) == Some(&'\'')) =>
                {
                    let end = if next == Some('\\') { i + 4 } else { i + 3 };
                    for &ch in c.iter().take(end.min(c.len())).skip(i) {
                        out.push(ch);
                    }
                    i = end;
                }
                _ => {
                    out.push(cur);
                    i += 1;
                }
            }
        }
        out
    }

    /// Drop every `#[cfg(test)]` item (both `mod tests { … }` and the
    /// test-only helper `fn`s several engines define), so a knob that is only
    /// ever mentioned inside tests does not count as a production read.
    fn strip_test_items(src: &str) -> String {
        const ATTR: &str = "#[cfg(test)]";
        let mut out = String::with_capacity(src.len());
        let mut rest = src;
        while let Some(pos) = rest.find(ATTR) {
            out.push_str(&rest[..pos]);
            let after: Vec<char> = rest[pos + ATTR.len()..].chars().collect();
            // Walk to the item's opening brace (a `mod`/`fn` body) or its
            // terminating `;` (e.g. a `#[cfg(test)] use …;`), ignoring braces
            // that live inside string literals such as `format!("{}")`.
            let (mut i, mut depth, mut started, mut in_str) = (0usize, 0usize, false, false);
            while i < after.len() {
                let ch = after[i];
                if in_str {
                    if ch == '\\' {
                        i += 2;
                        continue;
                    }
                    if ch == '"' {
                        in_str = false;
                    }
                    i += 1;
                    continue;
                }
                match ch {
                    '"' => in_str = true,
                    ';' if !started => {
                        i += 1;
                        break;
                    }
                    '{' => {
                        depth += 1;
                        started = true;
                    }
                    '}' => {
                        depth -= 1;
                        if started && depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            let consumed: usize = after.iter().take(i).map(|c| c.len_utf8()).sum();
            rest = &rest[pos + ATTR.len() + consumed..];
        }
        out.push_str(rest);
        out
    }

    /// The engine's PRODUCTION read path: sources with comments and
    /// `#[cfg(test)]` items removed.
    fn engine_source(engine: &str) -> Option<String> {
        let files = ENGINE_SOURCES
            .iter()
            .find(|(name, _)| *name == engine)
            .map(|(_, files)| *files)?;
        let dir = repo_root().join("src/bin/vector_db_benchmark/engine");
        let mut blob = String::new();
        for file in files {
            let raw = std::fs::read_to_string(dir.join(file)).unwrap_or_default();
            blob.push_str(&strip_test_items(&strip_comments(&raw)));
            blob.push('\n');
        }
        Some(blob)
    }

    /// If `knob_path` names a direct leaf of a [`TYPED_CONTAINERS`] struct that
    /// the struct does not declare, return that container and its field set.
    ///
    /// This is the engine-independent half of the guard, and the only check that
    /// survives token matching being defeated by a short common identifier.
    fn undeclared_typed_leaf(knob_path: &str) -> Option<(&'static str, &'static [&'static str])> {
        TYPED_CONTAINERS
            .iter()
            .find(|(c, _)| knob_path.starts_with(&format!("{}.", c)))
            .and_then(|(container, allowed)| {
                let rel = &knob_path[container.len() + 1..];
                if !rel.contains('.') && !allowed.contains(&canonical(rel)) {
                    Some((*container, *allowed))
                } else {
                    None
                }
            })
    }

    /// Is `knob_path` (dotted) satisfied by the engine's source?
    ///
    /// The leaf token must ALWAYS be present verbatim. An earlier version let a
    /// distinctive parent stand in for a short leaf (`hnsw_config` vouching for
    /// `M`), which meant the issue #216 bug survived a one-level relocation:
    /// moving `ef` to `collection_params.hnsw_config.ef` passed on engines that
    /// merely mention `hnsw_config`. Parent proxying is gone; field access still
    /// counts, since `h.m` contains the whole token `m`.
    ///
    /// A bare method CALL — `.leaf(..)` — is not evidence. Those are almost
    /// always a client-library builder setter that merely shares a name with the
    /// knob: `opensearch.rs` calls reqwest's `.request_timeout(..)` on the
    /// force-merge and cluster-health requests with a value derived from
    /// OPENSEARCH_FORCE_MERGE_TIMEOUT, and never consults
    /// `connection_params.request_timeout`. Counting that as a read would have
    /// silently retired issue #245's KNOWN_UNREAD entry as paid-off debt.
    ///
    /// The asymmetry is deliberate. Discounting a real read costs a loud, easily
    /// diagnosed failure ("no source file of that engine ever reads it") that a
    /// human resolves in a minute. Accepting a fake read costs silence, which is
    /// the failure mode this whole guard exists to prevent — so when the two are
    /// in tension, err toward loud. An engine that genuinely reads a knob only
    /// through a same-named getter can add a field access, or the knob can be
    /// listed in `KNOWN_UNREAD` with the reason.
    fn knob_is_read(source: &str, knob_path: &str) -> bool {
        let leaf = canonical(knob_path.rsplit('.').next().unwrap_or(knob_path));
        token_occurrences(source, leaf).any(|at| !is_method_call(source, at, leaf))
    }

    /// Byte offsets of every whole-token occurrence of `token` in `haystack`.
    fn token_occurrences<'a>(
        haystack: &'a str,
        token: &'a str,
    ) -> impl Iterator<Item = usize> + 'a {
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let bytes = haystack.as_bytes();
        let mut from = 0usize;
        std::iter::from_fn(move || {
            while let Some(pos) = haystack[from..].find(token) {
                let start = from + pos;
                let end = start + token.len();
                from = start + 1;
                let before_ok = start == 0 || !is_word(bytes[start - 1] as char);
                let after_ok = end >= bytes.len() || !is_word(bytes[end] as char);
                if before_ok && after_ok {
                    return Some(start);
                }
            }
            None
        })
    }

    /// Whether the occurrence at `at` is a method call `.token(` rather than a
    /// field access or a plain identifier.
    ///
    /// Deliberately does NOT skip whitespace before the dot: `foo\n    .bar(` is
    /// a chained builder call and is treated as one, which is exactly the shape
    /// a fluent client library produces.
    fn is_method_call(source: &str, at: usize, token: &str) -> bool {
        let dotted = source[..at].trim_end().ends_with('.');
        let called = source[at + token.len()..]
            .bytes()
            .find(|b| !b.is_ascii_whitespace())
            .map(|b| b == b'(')
            .unwrap_or(false);
        dotted && called
    }

    /// Every knob declared by a shipped config must be read by its engine.
    #[test]
    fn every_shipped_config_knob_is_read_by_its_engine() {
        let dir = repo_root().join("experiments/configurations");
        let mut violations: Vec<String> = Vec::new();
        let mut checked_entries = 0usize;
        let mut used_exemptions: BTreeSet<(String, String)> = BTreeSet::new();

        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("configurations dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no shipped configs found in {:?}", dir);

        for path in files {
            let content = std::fs::read_to_string(&path).expect("read config");
            let entries: Vec<serde_json::Value> = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(_) => continue, // not an engine-config array
            };
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();

            for entry in entries {
                let engine = match entry.get("engine").and_then(|v| v.as_str()) {
                    Some(e) => e,
                    None => continue,
                };
                let name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let source = match engine_source(engine) {
                    Some(s) => s,
                    None => {
                        violations.push(format!(
                            "{} [{}]: unknown engine '{}' - add it to ENGINE_SOURCES",
                            file_name, name, engine
                        ));
                        continue;
                    }
                };
                checked_entries += 1;

                // A misspelled root key (`collection_param`) would silently drop
                // a whole block: serde ignores it and nothing downstream looks
                // for it. Check the roots against EngineConfig's real fields.
                if let Some(obj) = entry.as_object() {
                    for key in obj.keys() {
                        // `_`-prefixed keys are a deliberate JSON documentation
                        // idiom (e.g. qdrant-tenant-on-disk.json's `_comment`
                        // naming the only dataset the config is valid for), not
                        // a typo. serde ignores them by design.
                        if key.starts_with('_') {
                            continue;
                        }
                        if !ENGINE_CONFIG_ROOT_KEYS.contains(&key.as_str()) {
                            violations.push(format!(
                                "{} [{}] has unknown top-level key '{}' - EngineConfig \
                                 has no such field, so the whole block is silently \
                                 ignored (a typo for one of {:?}?)",
                                file_name, name, key, ENGINE_CONFIG_ROOT_KEYS
                            ));
                        }
                    }
                }

                let mut knobs = BTreeSet::new();
                for block in ["collection_params", "connection_params", "upload_params"] {
                    if let Some(v) = entry.get(block) {
                        collect_knobs(v, block, &mut knobs);
                    }
                }
                if let Some(list) = entry.get("search_params").and_then(|v| v.as_array()) {
                    for sp in list {
                        collect_knobs(sp, "search_params", &mut knobs);
                    }
                }

                for knob in knobs {
                    let leaf = knob.rsplit('.').next().unwrap();
                    // Runner-level knobs live directly under a search_params entry.
                    if knob.starts_with("search_params.")
                        && knob.matches('.').count() == 1
                        && FRAMEWORK_SEARCH_KNOBS.contains(&leaf)
                    {
                        continue;
                    }
                    // Leaves inside a wholesale-forwarded container.
                    if PASSTHROUGH_CONTAINERS.iter().any(|(eng, container)| {
                        *eng == engine && knob.starts_with(&format!("{}.", container))
                    }) {
                        continue;
                    }

                    // Decide the violation FIRST, and only then consult the
                    // exemption list. Checking the exemption first would let an
                    // entry stay "used" long after the underlying bug was fixed
                    // — which is exactly what happened when #215 landed and made
                    // `hnsw_config.on_disk` a real, read knob.
                    let schema_violation =
                        undeclared_typed_leaf(&knob).map(|(container, allowed)| {
                            format!(
                                "{} [{}] declares '{}', but '{}' declares no such \
                                 field (only {:?}) - it lands in the untyped \
                                 catch-all that no engine reads as a knob, so it \
                                 can never take effect (see issue #216)",
                                file_name, name, knob, container, allowed
                            )
                        });

                    let reason = schema_violation.or_else(|| {
                        if knob_is_read(&source, &knob) {
                            None
                        } else {
                            Some(format!(
                                "{} [{}] targets engine '{}' but declares '{}', which \
                                 no source file of that engine ever reads - it parses \
                                 and is then silently discarded, so every run of this \
                                 config measures the engine default (see issue #216)",
                                file_name, name, engine, knob
                            ))
                        }
                    });

                    let exempt = KNOWN_UNREAD
                        .iter()
                        .any(|(eng, path, _)| *eng == engine && *path == knob);

                    match (reason, exempt) {
                        (Some(_), true) => {
                            used_exemptions.insert((engine.to_string(), knob.clone()));
                        }
                        (Some(reason), false) => violations.push(reason),
                        (None, true) => violations.push(format!(
                            "{} [{}] {}/{} is listed in KNOWN_UNREAD but is no longer a \
                             violation - the engine reads it now, so delete the entry",
                            file_name, name, engine, knob
                        )),
                        (None, false) => {}
                    }
                }
            }
        }

        // Anti-rot: a pass-through entry must name a container the engine really
        // reads, otherwise it would excuse a block that is dropped entirely.
        for (engine, container) in PASSTHROUGH_CONTAINERS {
            let source = engine_source(engine).expect("passthrough engine exists");
            let leaf = container.rsplit('.').next().unwrap();
            assert!(
                contains_token(&source, leaf),
                "PASSTHROUGH_CONTAINERS claims {} forwards '{}', but its source never \
                 mentions '{}' - the container is not read at all",
                engine,
                container,
                leaf
            );
        }

        // Anti-rot: every exemption must correspond to a violation that is still
        // live. This catches BOTH ways an entry goes stale — the engine learning
        // to read the knob, and the knob being dropped from the configs — so the
        // list cannot quietly accumulate permission slips.
        for (engine, path, _why) in KNOWN_UNREAD {
            assert!(
                used_exemptions.contains(&(engine.to_string(), path.to_string())),
                "KNOWN_UNREAD lists {}/{}, but nothing in experiments/configurations \
                 triggers it any more - the debt is paid, so delete the entry",
                engine,
                path
            );
        }

        assert!(checked_entries > 0, "guard checked no config entries");
        assert!(
            violations.is_empty(),
            "shipped configs declare {} knob(s) their engine never reads:\n  - {}",
            violations.len(),
            violations.join("\n  - ")
        );
    }

    /// The guard must actually fire. Without this, a bug in `contains_token` or
    /// `knob_is_read` would turn the test above into a permanent green light —
    /// the same silence that let issue #216 survive.
    #[test]
    fn guard_detects_a_knob_the_engine_does_not_read() {
        // Exercised against synthetic sources, not a real engine file: a guard
        // whose self-test depends on a 2-letter token being absent from a
        // 1800-line file would break the first time someone names a local `ef`.
        let reads_nothing = "fn search() { let num_candidates = 10; }";
        assert!(
            !knob_is_read(reads_nothing, "search_params.search_params.ef"),
            "a source that never mentions `ef` must not satisfy the ef knob — \
             this is the exact shape of issue #216"
        );
        assert!(knob_is_read(
            "let ef = params.ef.unwrap_or(64);",
            "search_params.search_params.ef"
        ));

        assert!(
            !knob_is_read(reads_nothing, "collection_params.hnsw_config.M"),
            "no hnsw_config in source means M cannot reach the server"
        );
        // Field access is a read; the parent alone is NOT.
        assert!(knob_is_read(
            "let m = cp.hnsw_config.as_ref().and_then(|h| h.m);",
            "collection_params.hnsw_config.M"
        ));
        assert!(
            !knob_is_read("cp.hnsw_config.as_ref()", "collection_params.hnsw_config.M"),
            "mentioning hnsw_config must not vouch for the M leaf"
        );

        // A builder setter is not a read. This is the shape that broke when #246
        // met master: opensearch.rs gained reqwest `.request_timeout(..)` calls
        // whose value comes from OPENSEARCH_FORCE_MERGE_TIMEOUT, and the guard
        // concluded the engine now read `connection_params.request_timeout` —
        // which would have retired issue #245's KNOWN_UNREAD entry as paid debt.
        assert!(
            !knob_is_read(
                ".forcemerge(x)\n.max_num_segments(1)\n.request_timeout(merge_deadline)\n.send()",
                "connection_params.request_timeout"
            ),
            "a chained `.request_timeout(..)` builder call is a transport setter, \
             not evidence that the engine reads the config knob of that name"
        );
        // ...but any non-call mention still counts, so a real read is never lost.
        assert!(
            knob_is_read(
                "let t = cp.request_timeout.unwrap_or(300);",
                "connection_params.request_timeout"
            ),
            "field access must still count as a read"
        );
        assert!(
            knob_is_read(
                ".request_timeout(cfg.request_timeout)",
                "connection_params.request_timeout"
            ),
            "forwarding the config value INTO the setter is a genuine read, and \
             the argument occurrence is not itself a method call"
        );
        // THE relocation defence, tested directly rather than only via the
        // whole-corpus sweep. Token matching cannot stop `hnsw_config.ef`
        // (qdrant, redis and pgvector all contain a bare `ef`); the schema check
        // must, on every engine, because HnswConfig declares no such field.
        assert!(
            undeclared_typed_leaf("collection_params.hnsw_config.ef").is_some(),
            "relocating `ef` under hnsw_config must be caught by the schema check"
        );
        // Fields the struct really declares must NOT be flagged - including the
        // ones #215 added, which are read by qdrant.
        for declared in [
            "M",
            "EF_CONSTRUCTION",
            "ef_construct",
            "on_disk",
            "payload_m",
            "inline_storage",
            "full_scan_threshold",
            "max_indexing_threads",
        ] {
            let path = format!("collection_params.hnsw_config.{}", declared);
            assert!(
                undeclared_typed_leaf(&path).is_none(),
                "{} is a declared HnswConfig field and must not be flagged",
                declared
            );
        }
        // Nested paths are not direct leaves and are left to the token check.
        assert!(undeclared_typed_leaf("collection_params.hnsw_config").is_none());
        assert!(undeclared_typed_leaf("collection_params.vectors_config.on_disk").is_none());

        // The #216 bug must not survive being relocated one level: `ef` under
        // hnsw_config is still unread unless the leaf itself appears.
        assert!(
            !knob_is_read(
                "let h = cp.hnsw_config.as_ref();",
                "collection_params.hnsw_config.ef"
            ),
            "relocating `ef` under a mentioned parent must not launder it"
        );

        // Whole-token matching: `ef` must not be satisfied by `ef_construction`.
        assert!(
            !contains_token("ef_construction", "ef"),
            "token match must respect word boundaries"
        );
        assert!(contains_token("let ef = 1;", "ef"));
        assert!(!contains_token("prefix_m_suffix", "m"));

        // A knob mentioned ONLY in a comment must not count as read: the likely
        // future regression is a refactor that deletes the forwarding code and
        // leaves the explanatory comment behind.
        let comment_only = "/// forwards hnsw_config to hnswOptions\n\
                            // hnsw_config.M -> maxEdges\n\
                            fn create_index() { let x = 1; }";
        assert!(
            !knob_is_read(
                &strip_test_items(&strip_comments(comment_only)),
                "collection_params.hnsw_config.M"
            ),
            "a comment must not vouch for a knob the code no longer forwards"
        );

        // Nor may a mention that exists only inside a #[cfg(test)] item.
        let test_only = "fn create_index() { let x = 1; }\n\
                         #[cfg(test)]\n\
                         mod tests { fn t() { let s = \"hnsw_config\"; } }";
        assert!(
            !knob_is_read(
                &strip_test_items(&strip_comments(test_only)),
                "collection_params.hnsw_config.M"
            ),
            "a #[cfg(test)] mention must not vouch for a production read"
        );

        // Stripping must NOT eat string literals — many engines read knobs via
        // `params.get("batch_size")`, so the literal is the read.
        let via_literal = "let b = p.get(\"batch_size\"); // comment";
        let stripped = strip_test_items(&strip_comments(via_literal));
        assert!(contains_token(&stripped, "batch_size"));
        assert!(!stripped.contains("comment"));

        // Stripping must not swallow code after a `//` inside a string literal.
        let url = "let u = \"mongodb://host\"; let batch_size = 1;";
        let stripped_url = strip_test_items(&strip_comments(url));
        assert!(
            contains_token(&stripped_url, "batch_size"),
            "a `//` inside a string must not start a comment: {:?}",
            stripped_url
        );

        // An engine honouring EF_CONSTRUCTION while silently dropping M must
        // not pass clean.
        assert!(
            !knob_is_read(
                "let e = cp.hnsw_config.ef_construction;",
                "collection_params.hnsw_config.M"
            ),
            "reading hnsw_config must not by itself vouch for M"
        );

        // Finally, tie it to the real engine: the knobs wired for issue #216
        // must be visible in the MongoDB read path.
        let mongodb = engine_source("mongodb").expect("mongodb source");
        assert!(
            knob_is_read(&mongodb, "collection_params.hnsw_config.M"),
            "hnsw_config must reach the MongoDB index definition"
        );
        assert!(
            knob_is_read(&mongodb, "collection_params.hnsw_config.EF_CONSTRUCTION"),
            "EF_CONSTRUCTION must reach the MongoDB index definition"
        );
        assert!(
            contains_token(&mongodb, "hnswOptions"),
            "MongoDB spells the HNSW build knobs `hnswOptions`, not m/efConstruction"
        );
    }
}
