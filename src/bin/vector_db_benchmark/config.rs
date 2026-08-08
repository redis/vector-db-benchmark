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
    pub ef: Option<i64>,
    /// Catch-all for additional search params (e.g., SEARCH_WINDOW_SIZE, data_type)
    #[serde(flatten)]
    pub extra: Option<std::collections::HashMap<String, serde_json::Value>>,
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
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(configs) = serde_json::from_str::<Vec<EngineConfig>>(&content) {
                for config in configs {
                    all_configs.insert(config.name.clone(), config);
                }
            }
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
/// So instead of validating *parsing*, this guard validates *consumption*: for
/// every entry in every shipped config, each declared knob must appear as a real
/// token in the source of the engine that entry targets. It is a source-level
/// check on purpose — it is the only thing that distinguishes "the engine reads
/// this" from "serde accepted this and threw it away".
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
        "search_params",
    ];

    /// Resolve serde aliases to the field name the Rust source actually spells.
    /// A config may say `EF_CONSTRUCTION`, `ef_construct` or `ef_construction`;
    /// engines read `.ef_construction`.
    fn canonical(leaf: &str) -> &str {
        match leaf {
            "M" => "m",
            "EF_CONSTRUCTION" | "ef_construct" => "ef_construction",
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

    fn engine_source(engine: &str) -> Option<String> {
        let files = ENGINE_SOURCES
            .iter()
            .find(|(name, _)| *name == engine)
            .map(|(_, files)| *files)?;
        let dir = repo_root().join("src/bin/vector_db_benchmark/engine");
        let mut blob = String::new();
        for file in files {
            blob.push_str(&std::fs::read_to_string(dir.join(file)).unwrap_or_default());
            blob.push('\n');
        }
        Some(blob)
    }

    /// Is `knob_path` (dotted) satisfied by the engine's source?
    ///
    /// A leaf of one or two characters (`m`, `ef`) is unsearchable on its own —
    /// short identifiers occur everywhere in Rust. For those the *parent*
    /// container token carries the signal: reading `hnsw_config` at all is what
    /// proves `hnsw_config.M` reaches the server. A short knob whose parent is
    /// itself a bare container (`search_params.search_params.ef`) still has to be
    /// found verbatim, which is what catches issue #216.
    fn knob_is_read(source: &str, knob_path: &str) -> bool {
        let parts: Vec<&str> = knob_path.split('.').collect();
        let leaf = canonical(parts[parts.len() - 1]);
        if leaf.len() > 2 {
            return contains_token(source, leaf);
        }
        if parts.len() < 2 {
            return contains_token(source, leaf);
        }
        let parent = parts[parts.len() - 2];
        // `search_params` is the generic container every engine mentions; it
        // carries no evidence that this particular knob is honoured.
        if parent == "search_params" {
            return contains_token(source, leaf);
        }
        contains_token(source, parent)
    }

    /// Every knob declared by a shipped config must be read by its engine.
    #[test]
    fn every_shipped_config_knob_is_read_by_its_engine() {
        let dir = repo_root().join("experiments/configurations");
        let mut violations: Vec<String> = Vec::new();
        let mut checked_entries = 0usize;

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

                let mut knobs = BTreeSet::new();
                if let Some(cp) = entry.get("collection_params") {
                    collect_knobs(cp, "collection_params", &mut knobs);
                }
                if let Some(up) = entry.get("upload_params") {
                    collect_knobs(up, "upload_params", &mut knobs);
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
                    if knob_is_read(&source, &knob) {
                        continue;
                    }
                    violations.push(format!(
                        "{} [{}] targets engine '{}' but declares '{}', which no \
                         source file of that engine ever reads - it parses and is \
                         then silently discarded, so every run of this config \
                         measures the engine default (see issue #216)",
                        file_name, name, engine, knob
                    ));
                }
            }
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

        // A short leaf under a distinctive typed parent is proven by the parent.
        assert!(
            !knob_is_read(reads_nothing, "collection_params.hnsw_config.M"),
            "no hnsw_config in source means M cannot reach the server"
        );
        assert!(knob_is_read(
            "cp.hnsw_config.as_ref()",
            "collection_params.hnsw_config.M"
        ));

        // Whole-token matching: `ef` must not be satisfied by `ef_construction`.
        assert!(
            !contains_token("ef_construction", "ef"),
            "token match must respect word boundaries"
        );
        assert!(contains_token("let ef = 1;", "ef"));
        assert!(!contains_token("prefix_m_suffix", "m"));

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
