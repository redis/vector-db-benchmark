//! Elasticsearch engine implementation.
//!
//! Uses the official `elasticsearch` crate (async, wrapped with tokio block_on).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use elasticsearch::http::request::JsonBody;
use elasticsearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use elasticsearch::indices::{
    IndicesCreateParts, IndicesDeleteParts, IndicesForcemergeParts, IndicesRefreshParts,
};
use elasticsearch::params::WaitForStatus;
use elasticsearch::{BulkParts, CountParts, Elasticsearch, SearchParts};
use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use uuid::Uuid;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

/// Shard count this engine creates its index with. Not configurable: it is
/// pinned so an ES run is reproducible across cluster versions and deployments,
/// whatever their per-index default happens to be.
///
/// It is a named constant rather than a literal because it is a *cross-engine*
/// invariant, not a local choice: the shipped `opensearch-single-node.json` pins
/// the same value so the published ES-vs-OS head-to-head compares two indexes
/// with the same shard count (#211), and
/// `opensearch::tests::shipped_opensearch_configs_pin_their_shard_count`
/// asserts that equality against this constant. Changing this number therefore
/// fails that test until the OpenSearch config is moved with it, instead of
/// silently desynchronising the pairing.
pub(crate) const ES_NUMBER_OF_SHARDS: i64 = 1;

/// Elasticsearch engine configuration parsed from JSON
#[derive(Clone)]
struct ElasticsearchConfig {
    m: i64,
    ef_construction: i64,
    batch_size: usize,
    parallel: usize,
}

pub struct ElasticsearchEngine {
    name: String,
    index_name: String,
    #[allow(dead_code)]
    timeout: u64,
    config: ElasticsearchConfig,
    search_params: Vec<SearchParams>,
    /// Base URL for constructing per-thread clients
    base_url: String,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared Elasticsearch client
    client: Arc<Elasticsearch>,
}

impl ElasticsearchEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("ELASTIC_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9200);

        let index_name = std::env::var("ELASTIC_INDEX").unwrap_or_else(|_| "bench".to_string());
        let timeout: u64 = std::env::var("ELASTIC_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        // Extract HNSW config from collection_params.index_options (ES-specific)
        let (m, ef_construction) = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.index_options.as_ref())
            .map(|io| (io.m.unwrap_or(16), io.ef_construction.unwrap_or(100)))
            .unwrap_or((16, 100));

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(500) as usize;

        let base_url = build_base_url(host, port);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = create_es_client(&base_url, timeout)?;

        Ok(Self {
            name: engine_config.name.clone(),
            index_name,
            timeout,
            config: ElasticsearchConfig {
                m,
                ef_construction,
                batch_size,
                parallel,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            base_url,
            rt,
            client: Arc::new(client),
        })
    }

    fn create_progress_bar(&self, total: usize) -> ProgressBar {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec_int}/s)")
                .unwrap()
                .with_key("per_sec_int", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    write!(w, "{}", HumanCount(state.per_sec() as u64)).unwrap()
                })
                .progress_chars("#>-"),
        );
        pb
    }

    fn delete_index(&self) -> Result<(), String> {
        let resp = self
            .rt
            .block_on(
                self.client
                    .indices()
                    .delete(IndicesDeleteParts::Index(&[&self.index_name]))
                    .send(),
            )
            .map_err(|e| format!("Failed to delete index: {}", e))?;

        let status = resp.status_code().as_u16();
        if status == 200 || status == 404 {
            Ok(())
        } else {
            Err(format!("Failed to delete index: status {}", status))
        }
    }

    fn create_index(&self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let dist_lower = distance.to_lowercase();
        let similarity = resolve_index_similarity(&dist_lower, vector_size)?;

        let mut properties = serde_json::json!({
            "vector": {
                "type": "dense_vector",
                "dims": vector_size,
                "index": true,
                "similarity": similarity,
                "index_options": {
                    "type": "hnsw",
                    "m": self.config.m,
                    "ef_construction": self.config.ef_construction,
                }
            }
        });

        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                let props = properties.as_object_mut().unwrap();
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let es_type = match ft {
                        "int" => "long",
                        "geo" => "geo_point",
                        // Bools/datetimes arrive from the reader as strings
                        // ("true"/"false", ISO-8601); ES coerces those into a
                        // `boolean` field and parses ISO into a `date` field. The
                        // canonical schema names them "bool"/"datetime", which are
                        // NOT valid ES types — forwarding them verbatim made
                        // index creation reject the whole mapping.
                        "bool" => "boolean",
                        "datetime" => "date",
                        // A `uuid` is an exact-match opaque string; "uuid" is not
                        // a valid ES type, so (like bool/datetime above) forwarding
                        // it verbatim made index creation reject the whole mapping,
                        // silently breaking every uuid-equality filter. Map it to
                        // `keyword` (exact term match, no analysis).
                        "uuid" => "keyword",
                        other => other,
                    };
                    props.insert(
                        field_name.clone(),
                        serde_json::json!({
                            "type": es_type,
                            "index": true,
                        }),
                    );
                }
            }
        }

        let body = serde_json::json!({
            "settings": {
                "index": build_index_settings(),
            },
            "mappings": {
                "_source": { "excludes": ["vector"] },
                "properties": properties,
            }
        });

        let resp = self
            .rt
            .block_on(
                self.client
                    .indices()
                    .create(IndicesCreateParts::Index(&self.index_name))
                    .body(body)
                    .send(),
            )
            .map_err(|e| format!("Failed to create index: {}", e))?;

        if !resp.status_code().is_success() {
            let body = self.rt.block_on(resp.text()).unwrap_or_default();
            return Err(format!("Failed to create index: {}", body));
        }

        Ok(())
    }

    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.config.batch_size)
            .map(|start| (start, (start + self.config.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let base_url = base_url.clone();
                let index_name = index_name.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    let client = match create_es_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e);
                            return;
                        }
                    };

                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_batches {
                            break;
                        }
                        if error.lock().unwrap().is_some() {
                            break;
                        }

                        let (batch_start, batch_end) = batches[idx];
                        if let Err(e) = upload_bulk_batch(
                            &rt,
                            &client,
                            &index_name,
                            &ids[batch_start..batch_end],
                            &vectors[batch_start..batch_end],
                            &metadata[batch_start..batch_end],
                        ) {
                            *error.lock().unwrap() = Some(e);
                            break;
                        }
                        pb.inc((batch_end - batch_start) as u64);
                    }
                });
            }
        });

        pb.finish_with_message("Upload complete");

        if let Some(e) = error.lock().unwrap().take() {
            return Err(e);
        }
        Ok(())
    }

    /// Make just-uploaded documents searchable.
    ///
    /// Required because `build_index_settings` sets `refresh_interval: -1`: with
    /// no periodic refresh, documents sitting in the in-memory buffer are not
    /// visible to search. An explicit `_refresh` is also *not* implied by the
    /// force-merge that follows — force-merge rewrites segments, it does not
    /// promise to reopen the searcher — and ES's "search idle" auto-refresh
    /// escape hatch does not apply to an index with an explicit
    /// `refresh_interval`. So this call is the thing that guarantees the search
    /// phase sees the full corpus. Mirrors `opensearch.rs::refresh`.
    ///
    /// Retried on the same budget as force-merge: it runs after the whole
    /// ingest, and failing it un-retried would discard a long upload.
    fn refresh(&self) -> Result<(), String> {
        println!("Refreshing index...");

        let max_retries = 30;
        let mut last_err = String::new();
        for attempt in 0..=max_retries {
            let result = self.rt.block_on(
                self.client
                    .indices()
                    .refresh(IndicesRefreshParts::Index(&[&self.index_name]))
                    .send(),
            );

            match result {
                Ok(resp) if resp.status_code().is_success() => return Ok(()),
                Ok(resp) => {
                    last_err = format!("status {}", resp.status_code());
                }
                Err(e) => {
                    last_err = e.to_string();
                }
            }

            if attempt < max_retries {
                println!("Refresh retry {}/{}: {}", attempt, max_retries, last_err);
            }
        }
        Err(format!(
            "Refresh failed after {} retries: {}",
            max_retries, last_err
        ))
    }

    fn force_merge(&self) -> Result<(), String> {
        println!("Forcing merge into 1 segment...");

        let max_retries = 30;
        for attempt in 0..=max_retries {
            let result = self.rt.block_on(
                self.client
                    .indices()
                    .forcemerge(IndicesForcemergeParts::Index(&[&self.index_name]))
                    .max_num_segments(1)
                    .send(),
            );

            match result {
                Ok(resp) if resp.status_code().is_success() => {
                    self.wait_for_cluster_health()?;
                    return Ok(());
                }
                Ok(resp) => {
                    if attempt < max_retries {
                        println!(
                            "Force merge retry {}/{}: status {}",
                            attempt,
                            max_retries,
                            resp.status_code()
                        );
                        continue;
                    }
                    return Err(format!(
                        "Force merge failed after {} retries: {}",
                        max_retries,
                        resp.status_code()
                    ));
                }
                Err(e) => {
                    if attempt < max_retries {
                        println!("Force merge retry {}/{}: {}", attempt, max_retries, e);
                        continue;
                    }
                    return Err(format!(
                        "Force merge failed after {} retries: {}",
                        max_retries, e
                    ));
                }
            }
        }
        Ok(())
    }

    fn wait_for_cluster_health(&self) -> Result<(), String> {
        println!("Waiting for ES yellow status...");

        for _ in 0..100 {
            let result = self.rt.block_on(
                self.client
                    .cluster()
                    .health(elasticsearch::cluster::ClusterHealthParts::None)
                    .wait_for_status(WaitForStatus::Yellow)
                    .timeout("10s")
                    .send(),
            );

            match result {
                Ok(resp) if resp.status_code().is_success() => return Ok(()),
                _ => std::thread::sleep(std::time::Duration::from_millis(100)),
            }
        }
        Err("Elasticsearch cluster did not reach yellow status in time".to_string())
    }
}

/// Index-level settings applied at create time.
///
/// `refresh_interval: -1` disables periodic refresh for the duration of the
/// benchmark. The whole dataset is indexed in one bulk phase, so a background
/// refresh every N seconds only buys visibility nobody reads. Upstream qdrant's
/// reference tool does the same for Elasticsearch
/// (`engine/clients/elasticsearch/configure.py`: *"no refresh is required
/// because we index all the data at once"*), and `opensearch.rs` already did.
///
/// The reason is **comparability, not throughput** (#240). Measured live, `-1`
/// does ~5x fewer refreshes and cuts ~2.5x fewer segments during ingest, but it
/// does NOT reliably make the upload phase faster: on an HNSW field, frequent
/// refreshes cut many *small* segments, and k small graphs over n/k docs each
/// cost `n·log(n/k)` — cheaper per document than the few large graphs `-1`
/// builds. The setting moves HNSW construction between the ingest and merge
/// phases more than it removes it. What the old `10s` actually broke was the
/// ES-vs-OS head-to-head: the two engines ingested under different index
/// configurations, exactly like the shard-count mismatch `ES_NUMBER_OF_SHARDS`
/// above fixes (#211/#235). Do not "restore" `10s` chasing an upload number.
///
/// Because nothing refreshes on a timer any more, `upload()` MUST issue an
/// explicit `_refresh` before the search phase (see `ElasticsearchEngine::refresh`).
fn build_index_settings() -> serde_json::Value {
    serde_json::json!({
        // Cross-engine invariant, not a local choice — see the constant's docs.
        "number_of_shards": ES_NUMBER_OF_SHARDS,
        "number_of_replicas": 0,
        "refresh_interval": -1,
    })
}

/// Build the base URL with authentication from env vars.
fn build_base_url(host: &str, port: u16) -> String {
    let api_key = std::env::var("ELASTIC_API_KEY").ok();
    let user = std::env::var("ELASTIC_USER").unwrap_or_else(|_| "elastic".to_string());
    let password = std::env::var("ELASTIC_PASSWORD").unwrap_or_else(|_| "passwd".to_string());

    let scheme_host = if host.starts_with("http") {
        host.to_string()
    } else {
        format!("http://{}", host)
    };

    if api_key.is_some() {
        format!("{}:{}", scheme_host, port)
    } else if let Some(rest) = scheme_host.strip_prefix("http://") {
        format!("http://{}:{}@{}:{}", user, password, rest, port)
    } else if let Some(rest) = scheme_host.strip_prefix("https://") {
        format!("https://{}:{}@{}:{}", user, password, rest, port)
    } else {
        format!("http://{}:{}@{}:{}", user, password, scheme_host, port)
    }
}

/// Create an Elasticsearch client from a base URL.
fn create_es_client(base_url: &str, timeout: u64) -> Result<Elasticsearch, String> {
    let url = elasticsearch::http::Url::parse(base_url)
        .map_err(|e| format!("Invalid base URL '{}': {}", base_url, e))?;
    let pool = SingleNodeConnectionPool::new(url);
    let transport = TransportBuilder::new(pool)
        .timeout(std::time::Duration::from_secs(timeout))
        .disable_proxy()
        .cert_validation(elasticsearch::cert::CertificateValidation::None)
        .build()
        .map_err(|e| format!("Failed to build transport: {}", e))?;
    Ok(Elasticsearch::new(transport))
}

/// Convert integer ID to UUID hex string (matches Python uuid.UUID(int=idx).hex)
fn id_to_uuid_hex(id: i64) -> String {
    Uuid::from_u128(id as u128).as_simple().to_string()
}

/// Convert UUID hex string back to integer ID
fn uuid_hex_to_int(hex: &str) -> Result<i64, String> {
    let uuid = Uuid::parse_str(hex).map_err(|e| format!("Invalid UUID hex '{}': {}", hex, e))?;
    Ok(uuid.as_u128() as i64)
}

/// Validate index build parameters and resolve the `dense_vector` similarity.
/// Extracted verbatim from `create_index` so the guard order + error strings are
/// unit-testable without a live Elasticsearch. `dist_lower` must already be
/// lowercased. Order matters: dot/ip is rejected first (with the DOT-specific
/// message), then the dim cap, then the general similarity mapping.
fn resolve_index_similarity(dist_lower: &str, vector_size: i64) -> Result<&'static str, String> {
    if dist_lower == "dot" || dist_lower == "ip" {
        return Err(
            "Elasticsearch does not support DOT product distance for benchmarking".to_string(),
        );
    }
    if vector_size > 2048 {
        return Err(format!(
            "Elasticsearch does not support vector_size > 2048 (got {})",
            vector_size
        ));
    }
    es_similarity(dist_lower)
}

/// Map a dataset distance name to the Elasticsearch `dense_vector` similarity.
/// `dot`/`ip` is unsupported (ES has no raw dot-product similarity) and unknown
/// metrics error. A wrong arm here would silently change ranking, so every arm
/// is unit-tested.
fn es_similarity(distance: &str) -> Result<&'static str, String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok("l2_norm"),
        "cosine" | "angular" => Ok("cosine"),
        "dot" | "ip" => {
            Err("Elasticsearch does not support DOT product distance for benchmarking".to_string())
        }
        other => Err(format!(
            "Unsupported distance metric for Elasticsearch: {}",
            other
        )),
    }
}

// ── Elasticsearch condition parser ─────────────────────────────────────

fn parse_es_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let and_filters = obj
        .get("and")
        .and_then(|v| v.as_array())
        .map(|entries| build_subfilters(entries))
        .filter(|f| !f.is_empty());
    let or_filters = obj
        .get("or")
        .and_then(|v| v.as_array())
        .map(|entries| build_subfilters(entries))
        .filter(|f| !f.is_empty());

    if and_filters.is_none() && or_filters.is_none() {
        return None;
    }

    // Mirror parse_os_conditions: omit empty `must`/`should` keys entirely and
    // set `minimum_should_match: 1` whenever `should` is present. Previously we
    // always emitted both arrays via `unwrap_or_default()` and never set
    // minimum_should_match, so an OR-only filter produced
    // `{bool:{must:[],should:[...]}}` — with an empty `must` present, ES treats
    // `should` as OPTIONAL (scoring only), matching ALL documents and silently
    // running the query UNFILTERED.
    let mut bool_query = serde_json::Map::new();
    if let Some(must) = and_filters {
        bool_query.insert("must".to_string(), serde_json::Value::Array(must));
    }
    if let Some(should) = or_filters {
        bool_query.insert("should".to_string(), serde_json::Value::Array(should));
        bool_query.insert(
            "minimum_should_match".to_string(),
            serde_json::Value::from(1),
        );
    }

    Some(serde_json::json!({ "bool": bool_query }))
}

fn build_subfilters(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            // Nested group: an array element that is itself a `{and:[...]}` /
            // `{or:[...]}` sub-tree (not a field leaf). Recurse via
            // parse_es_conditions to build the sub-group as its OWN nested
            // `{bool:{must|should}}` query and push that as a single clause, so
            // ES nests the group natively instead of flattening its leaves up
            // into the parent's must/should (which would union/intersect the
            // wrong set — e.g. `(a AND b) OR (c AND d)` mis-flattened to
            // `a OR b OR c OR d`).
            if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
                if let Some(sub) = parse_es_conditions(entry) {
                    filters.push(sub);
                }
                continue;
            }
            for (field_name, field_filters) in entry_obj {
                if let Some(filter_obj) = field_filters.as_object() {
                    for (condition_type, criteria) in filter_obj {
                        if let Some(filter) = build_filter(field_name, condition_type, criteria) {
                            filters.push(filter);
                        }
                    }
                }
            }
        }
    }
    filters
}

fn build_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<serde_json::Value> {
    match condition_type {
        "match" => {
            // match_any: field value in a list (keywords or integers). Emit a
            // `terms` query — the exact/case-sensitive OR-of-values semantics of
            // qdrant's Condition::matches(field, Vec). An empty IN-set matches
            // NOTHING, so we emit `terms: []` (a valid match-nothing query)
            // rather than dropping the clause: dropping the sole clause would
            // leave `bool.must:[]`, which ES treats as match-ALL — silently
            // returning unfiltered results, the inverse of the intended filter.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                return Some(serde_json::json!({"terms": {field_name: any}}));
            }
            // Full-text `{"match": {"text": …}}` conditions: emit an ES `match`
            // query against the analyzed `text` field. `match` tokenizes the
            // query the same way the field is analyzed, so it matches docs
            // CONTAINING the term(s) — aligning with the tokenized semantics
            // redis uses via `@field:($tok)` and the ground truth in
            // `write_fulltext_project` (docs whose body CONTAINS "quick"). We use
            // `match` (not `match_phrase`) because the fixture filters on single
            // tokens; dropping this clause would leave `bool.must:[]`, which ES
            // treats as match-ALL — silently running the kNN query UNFILTERED
            // while recall is scored against the filtered ground truth (#120).
            if let Some(text) = criteria.get("text") {
                return Some(serde_json::json!({"match": {field_name: text}}));
            }
            let value = criteria.get("value")?;
            // Guard non-scalar: an array/object/null under `value` is malformed
            // input — the canonical model uses `match.any` for lists. Drop the
            // clause (return None) instead of forwarding
            // `{"match":{field:[1,2]}}` verbatim, matching qdrant/redis/valkey/
            // vectorsets.
            if !(value.is_string() || value.is_number() || value.is_boolean()) {
                return None;
            }
            Some(serde_json::json!({"match": {field_name: value}}))
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            let mut range = serde_json::Map::new();
            for key in &["lt", "gt", "lte", "gte"] {
                if let Some(val) = criteria_obj.get(*key) {
                    if !val.is_null() {
                        range.insert(key.to_string(), val.clone());
                    }
                }
            }
            Some(serde_json::json!({"range": {field_name: range}}))
        }
        "geo" => {
            let lat = criteria.get("lat")?;
            let lon = criteria.get("lon")?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(serde_json::json!({
                "geo_distance": {
                    "distance": format!("{}m", radius),
                    field_name: {"lat": lat, "lon": lon},
                }
            }))
        }
        _ => None,
    }
}

/// Upload a batch using the official Elasticsearch bulk API.
fn upload_bulk_batch(
    rt: &tokio::runtime::Runtime,
    client: &Elasticsearch,
    index_name: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut body: Vec<JsonBody<serde_json::Value>> = Vec::with_capacity(ids.len() * 2);

    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);

        // Action line
        body.push(JsonBody::new(
            serde_json::json!({"index": {"_id": uuid_hex}}),
        ));

        // Document line
        let mut doc = serde_json::Map::new();
        let vec_json: Vec<serde_json::Value> = vectors[i]
            .iter()
            .map(|&f| serde_json::Value::from(f))
            .collect();
        doc.insert("vector".to_string(), serde_json::Value::Array(vec_json));

        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                let val = match v {
                    MetadataValue::String(s) => serde_json::Value::String(s.clone()),
                    MetadataValue::Int(n) => serde_json::Value::from(*n),
                    MetadataValue::Float(f) => serde_json::json!(*f),
                    MetadataValue::Labels(labels) => serde_json::Value::Array(
                        labels
                            .iter()
                            .map(|l| serde_json::Value::String(l.clone()))
                            .collect(),
                    ),
                    MetadataValue::Geo { lon, lat } => {
                        serde_json::json!({ "lon": lon, "lat": lat })
                    }
                };
                doc.insert(k.clone(), val);
            }
        }

        body.push(JsonBody::new(serde_json::Value::Object(doc)));
    }

    let resp = rt
        .block_on(client.bulk(BulkParts::Index(index_name)).body(body).send())
        .map_err(|e| format!("Bulk upload failed: {}", e))?;

    if !resp.status_code().is_success() {
        let text = rt.block_on(resp.text()).unwrap_or_default();
        return Err(format!("Bulk upload error: {}", text));
    }

    let resp_body: serde_json::Value = rt
        .block_on(resp.json())
        .map_err(|e| format!("Failed to parse bulk response: {}", e))?;

    if resp_body
        .get("errors")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        let items = resp_body.get("items").and_then(|v| v.as_array());
        let error_count = items
            .map(|arr| {
                arr.iter()
                    .filter(|item| item.get("index").and_then(|idx| idx.get("error")).is_some())
                    .count()
            })
            .unwrap_or(0);
        return Err(format!(
            "Bulk upload had {} errors out of {} documents",
            error_count,
            ids.len()
        ));
    }

    Ok(())
}

/// Build the KNN search request body for one query and pre-serialize it to a
/// `RawValue`. This is deliberately done OUTSIDE the per-query timed window (the
/// vector-to-JSON serialization — ryu float formatting over every dimension — is
/// client CPU work, not server latency). Pre-serializing to a `RawValue` means
/// the timed send only copies the already-formatted bytes onto the socket, so
/// the measured latency reflects the RPC round-trip like the reference engines
/// (pgvector/qdrant). The produced bytes are byte-identical to what the client
/// would otherwise serialize inline.
fn build_knn_body(
    query_vector: &[f32],
    top: usize,
    num_candidates: i64,
    filter: Option<&serde_json::Value>,
) -> Box<serde_json::value::RawValue> {
    let mut knn = serde_json::json!({
        "field": "vector",
        "query_vector": query_vector,
        "k": top,
        "num_candidates": num_candidates,
    });
    if let Some(f) = filter {
        knn.as_object_mut()
            .unwrap()
            .insert("filter".to_string(), f.clone());
    }

    // Response trimming: the benchmark only needs each hit's id (always returned
    // as `_id` metadata) and score, so don't ship the document `_source`. This
    // trims the response for a fairer QPS/latency measurement — matching the
    // OpenSearch engine's trimming.
    let body = serde_json::json!({
        "knn": knn,
        "size": top,
        "_source": false,
    });

    serde_json::value::to_raw_value(&body).expect("serialize KNN search body")
}

/// Send a pre-built KNN search request and return the DECODED response. The
/// consistent timed boundary (see qdrant/pgvector/redis) is: request-serialize
/// OUT of the window (`raw_body` is already serialized), RPC send + receive +
/// decode-to-structured-response IN the window (this fn: send + status check +
/// wire read + `from_str` into a `serde_json::Value`), and id/score extraction
/// OUT of the window (`extract_knn_hits`). So this whole fn runs inside the
/// timed region; the JSON decode is billed as latency exactly as qdrant's
/// protobuf decode and pgvector's row decode are.
fn knn_send(
    rt: &tokio::runtime::Runtime,
    client: &Elasticsearch,
    index_name: &str,
    raw_body: &serde_json::value::RawValue,
) -> Result<serde_json::Value, String> {
    let resp = rt
        .block_on(
            client
                .search(SearchParts::Index(&[index_name]))
                .body(raw_body)
                .send(),
        )
        .map_err(|e| format!("KNN search failed: {}", e))?;

    if !resp.status_code().is_success() {
        let text = rt.block_on(resp.text()).unwrap_or_default();
        return Err(format!("KNN search error: {}", text));
    }

    let text = rt
        .block_on(resp.text())
        .map_err(|e| format!("Failed to read search response: {}", e))?;
    serde_json::from_str(&text).map_err(|e| format!("Failed to parse search response: {}", e))
}

/// Extract the id/score list from an already-decoded response (done AFTER the
/// timed window — only pulling the final ids out of the decoded struct for
/// recall, mirroring how pgvector/qdrant extract ids after `elapsed`).
fn extract_knn_hits(resp_body: &serde_json::Value) -> Result<Vec<(i64, f64)>, String> {
    let hits = resp_body
        .get("hits")
        .and_then(|h| h.get("hits"))
        .and_then(|h| h.as_array())
        .ok_or_else(|| "Missing hits.hits in search response".to_string())?;

    let mut results = Vec::with_capacity(hits.len());
    for hit in hits {
        let id_hex = hit
            .get("_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing _id in hit".to_string())?;
        let score = hit.get("_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let id = uuid_hex_to_int(id_hex)?;
        results.push((id, score));
    }

    Ok(results)
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for ElasticsearchEngine {
    /// Server-side corpus size, for the `--skip-upload` reuse precondition
    /// (issue #238). `GET /<index>/_count`; a missing index (404) answers 0.
    fn corpus_row_count(&mut self) -> Result<Option<u64>, String> {
        let resp = self
            .rt
            .block_on(
                self.client
                    .count(CountParts::Index(&[&self.index_name]))
                    .send(),
            )
            .map_err(|e| format!("_count on index '{}' failed: {}", self.index_name, e))?;
        if resp.status_code().as_u16() == 404 {
            return Ok(Some(0));
        }
        let body: serde_json::Value = self.rt.block_on(resp.json()).map_err(|e| {
            format!(
                "_count on index '{}' returned no JSON: {}",
                self.index_name, e
            )
        })?;
        Ok(body.get("count").and_then(|v| v.as_u64()))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        println!(
            "Elasticsearch: index_options {{ m: {}, ef_construction: {} }}",
            self.config.m, self.config.ef_construction
        );

        println!("Ensuring index does not exist...");
        self.delete_index()?;

        println!("Creating index '{}'...", self.index_name);
        self.create_index(dataset)?;
        println!("Index '{}' created successfully.", self.index_name);

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();

        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();

        println!(
            "Read {} vectors ({}d) in {:.3}s ({:.0} vectors/sec)",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
            vectors.len() as f64 / read_time
        );

        println!(
            "Starting upload with {} threads, batch size {}...",
            self.config.parallel, self.config.batch_size
        );
        let upload_start = Instant::now();

        self.upload_parallel(&ids, &vectors, &metadata)?;

        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Explicit refresh (refresh_interval is disabled during upload) so the
        // documents are searchable, then merge segments. Include this
        // refresh+merge time in total_time for cross-engine comparability
        // (mirrors opensearch/mongodb; matches v0's post_upload() timing) — the
        // refresh work is not removed from the measurement, only moved out of
        // the concurrent ingest and accounted for once.
        let index_start = Instant::now();
        self.refresh()?;
        self.force_merge()?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: vectors.len(),
            parallel: self.config.parallel,
            batch_size: self.config.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;
        // Upstream nests this under `config`; accept the nested spelling too so an
        // upstream configuration does not silently fall back to 100. Nested is
        // checked FIRST, matching SearchParams::knob's documented precedence —
        // reading the typed flat field first would invert it for this one knob.
        let num_candidates = params
            .knob("num_candidates")
            .and_then(|v| v.as_i64())
            .or(params.num_candidates)
            .unwrap_or(100);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<serde_json::Value>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_es_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` and the fully serialized request bodies
        // BEFORE the parallel region so the timed window wraps only the RPC
        // round-trip (see build_knn_body). `tops[idx]` reproduces the same k the
        // request embeds, so recall is computed against an identical result set.
        let tops: Vec<usize> = (0..num_to_run)
            .map(|idx| {
                explicit_top.unwrap_or_else(|| {
                    let n = neighbors[idx].len();
                    if n > 0 {
                        n
                    } else {
                        10
                    }
                })
            })
            .collect();
        let raw_bodies: Vec<Box<serde_json::value::RawValue>> = (0..num_to_run)
            .map(|idx| {
                build_knn_body(
                    &queries[idx],
                    tops[idx],
                    num_candidates,
                    parsed_filters[idx].as_ref(),
                )
            })
            .collect();

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

        // Barrier-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window (mirrors redis/vertex). Every worker
        // builds its runtime + client and primes ONE discarded query, then blocks
        // on `ready`; the main thread stamps the shared start instant into
        // `start_cell` and releases `go`, so the measurement clock starts only once
        // all workers are warm and poised. A worker that fails to build its runtime
        // or client MUST still pass both barriers before returning, or the run
        // would deadlock.
        let ready = Arc::new(std::sync::Barrier::new(parallel + 1));
        let go = Arc::new(std::sync::Barrier::new(parallel + 1));
        let start_cell = Arc::new(std::sync::OnceLock::<Instant>::new());

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let base_url = base_url.clone();
                let index_name = index_name.clone();
                let neighbors = &neighbors;
                let tops = &tops;
                let raw_bodies = &raw_bodies;
                let query_idx = Arc::clone(&query_idx);
                let ready = Arc::clone(&ready);
                let go = Arc::clone(&go);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(_) => {
                            // Still cross both barriers so peers aren't stranded.
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };
                    let client = match create_es_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(_) => {
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this client with ONE discarded query so the cold first
                    // round-trip is not inside the measured window. Best effort:
                    // errors are ignored and its sample is NOT recorded.
                    if num_to_run > 0 {
                        let _ = knn_send(&rt, &client, &index_name, &raw_bodies[0]);
                    }

                    // Signal "connected + primed", then block until the main thread
                    // stamps the shared measurement start and releases everyone.
                    ready.wait();
                    go.wait();

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = tops[idx];

                        // Timed window: network send + receive + decode of the
                        // response into a structured value. Body is pre-serialized
                        // (out); id/score extraction runs after `elapsed` (out).
                        let query_start = Instant::now();
                        let response = knn_send(&rt, &client, &index_name, &raw_bodies[idx]);
                        let query_time = query_start.elapsed().as_secs_f64();

                        match response.and_then(|resp_body| extract_knn_hits(&resp_body)) {
                            Ok(result_ids) => {
                                let ordered_ids: Vec<i64> =
                                    result_ids.iter().map(|(id, _)| *id).collect();
                                let m = crate::metrics::compute_metrics(
                                    &ordered_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                t.push(query_time);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                            }
                            Err(e) => {
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                }));
            }

            // All workers are spawned; wait until each is connected + primed, then
            // stamp the shared measurement start and release everyone at once.
            ready.wait();
            let st = Instant::now();
            start_cell.set(st).ok();
            go.wait();

            for h in handles {
                let (t, p, r, mr, nd) = h.join().unwrap();
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
        });

        pb.finish_and_clear();
        // total_time excludes connection setup and the cold first query — it is
        // measured from the barrier release stamped into start_cell.
        let total_time = start_cell
            .get()
            .map(|st| st.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        self.delete_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Periodic refresh must stay disabled during ingest. `10s` here made ES
    /// ingest under a different index configuration than OpenSearch
    /// (`refresh_interval: -1`) and than upstream qdrant's reference client, so
    /// the published ES-vs-OS head-to-head was not apples-to-apples. Note this
    /// is a *comparability* guard, not a throughput one — the measurement in
    /// #240 found no reliable upload speedup from `-1` (see
    /// `build_index_settings`), so a future "this makes upload slower, revert
    /// it" is not a reason to restore `10s`. Regression guard for #240.
    #[test]
    fn index_settings_disable_periodic_refresh() {
        let settings = build_index_settings();
        assert_eq!(settings["refresh_interval"], serde_json::json!(-1));
    }

    /// The other half of the apples-to-apples ES-vs-OS pairing (#235): the shard
    /// count actually sent must be the cross-engine constant, not a re-inlined
    /// literal. Asserted against `ES_NUMBER_OF_SHARDS` rather than `1` on
    /// purpose — hoisting the settings into `build_index_settings` is exactly
    /// the kind of edit that could quietly drop the constant and desynchronise
    /// the pairing that `opensearch::tests::shipped_opensearch_configs_pin_their_shard_count`
    /// keys off.
    #[test]
    fn index_settings_pin_single_shard_no_replicas() {
        let settings = build_index_settings();
        assert_eq!(
            settings["number_of_shards"],
            serde_json::json!(ES_NUMBER_OF_SHARDS)
        );
        assert_eq!(settings["number_of_replicas"], serde_json::json!(0));
    }

    /// Load-bearing: the hoisted request body (pre-serialized to a RawValue) must
    /// be byte-identical to what the old inline `.body(value)` path put on the
    /// wire — otherwise this measurement-only change would alter the request.
    #[test]
    fn build_knn_body_bytes_match_old_wire_bytes() {
        let vec = vec![0.1f32, -0.2, 0.3];
        let top = 2usize;
        let num_candidates = 10i64;
        let filter = serde_json::json!({"term": {"color": "red"}});

        // Reconstruct exactly the Value the old inline path serialized via JsonBody.
        let mut knn = serde_json::json!({
            "field": "vector",
            "query_vector": vec,
            "k": top,
            "num_candidates": num_candidates,
        });
        knn.as_object_mut()
            .unwrap()
            .insert("filter".to_string(), filter.clone());
        let expected = serde_json::json!({"knn": knn, "size": top, "_source": false});
        let expected_bytes = serde_json::to_vec(&expected).unwrap();

        let raw = build_knn_body(&vec, top, num_candidates, Some(&filter));
        // The wire bytes: JsonBody(raw).write emits the RawValue verbatim.
        assert_eq!(raw.get().as_bytes(), expected_bytes.as_slice());
        // to_raw_value round-trips to the same bytes as to_vec (guards raw_value).
        let reparsed: serde_json::Value = serde_json::from_str(raw.get()).unwrap();
        assert_eq!(serde_json::to_vec(&reparsed).unwrap(), expected_bytes);

        // Unfiltered variant.
        let expected_nf = serde_json::json!({
            "knn": {"field": "vector", "query_vector": vec, "k": top, "num_candidates": num_candidates},
            "size": top,
            "_source": false,
        });
        let raw_nf = build_knn_body(&vec, top, num_candidates, None);
        assert_eq!(
            raw_nf.get().as_bytes(),
            serde_json::to_vec(&expected_nf).unwrap().as_slice()
        );
    }

    #[test]
    fn test_id_to_uuid_hex_zero() {
        assert_eq!(id_to_uuid_hex(0), "00000000000000000000000000000000");
    }

    #[test]
    fn test_match_any_string_list_emits_terms() {
        let c = build_filter(
            "color",
            "match",
            &serde_json::json!({"any": ["red", "blue"]}),
        )
        .unwrap();
        assert_eq!(c, serde_json::json!({"terms": {"color": ["red", "blue"]}}));
    }

    // #120: a full-text `{"match": {"text": …}}` condition must emit an analyzed
    // `match` query — NOT be dropped. Dropping it leaves `bool.must:[]`, which ES
    // treats as match-ALL, silently running the kNN query UNFILTERED.
    #[test]
    fn test_match_text_emits_match_query() {
        let c = build_filter("body", "match", &serde_json::json!({"text": "quick"})).unwrap();
        assert_eq!(c, serde_json::json!({"match": {"body": "quick"}}));
    }

    #[test]
    fn es_text_only_condition_not_dropped() {
        // `{"and":[{"body":{"match":{"text":"quick"}}}]}` — the exact fixture
        // condition from `write_fulltext_project`. Must yield a non-empty
        // `bool.must` containing the `match` clause (was None → dropped → #120).
        let conditions = serde_json::json!({"and": [{"body": {"match": {"text": "quick"}}}]});
        let parsed =
            parse_es_conditions(&conditions).expect("text-only filter must not be dropped");
        assert_eq!(
            parsed,
            serde_json::json!({"bool": {"must": [{"match": {"body": "quick"}}]}})
        );
    }

    #[test]
    fn test_match_any_int_list_emits_terms() {
        let c = build_filter("size", "match", &serde_json::json!({"any": [1, 2, 3]})).unwrap();
        assert_eq!(c, serde_json::json!({"terms": {"size": [1, 2, 3]}}));
    }

    #[test]
    fn test_match_any_empty_list_matches_nothing() {
        // Empty IN-set must match NOTHING (never invert to match-all): `terms: []`.
        let c = build_filter("color", "match", &serde_json::json!({"any": []})).unwrap();
        assert_eq!(c, serde_json::json!({"terms": {"color": []}}));
    }

    #[test]
    fn test_match_exact_value_still_works() {
        let c = build_filter("color", "match", &serde_json::json!({"value": "red"})).unwrap();
        assert_eq!(c, serde_json::json!({"match": {"color": "red"}}));
    }

    // #121: a non-scalar `value` (a JSON array) is malformed input — the
    // canonical model uses `match.any` for lists. It must be dropped (None),
    // not forwarded verbatim as `{"match":{"n":[1,2]}}`. Matches qdrant/redis/
    // valkey/vectorsets.
    #[test]
    fn test_match_non_scalar_value_dropped() {
        assert_eq!(
            build_filter("n", "match", &serde_json::json!({"value": [1, 2]})),
            None
        );
        assert_eq!(
            build_filter("n", "match", &serde_json::json!({"value": {"x": 1}})),
            None
        );
        assert_eq!(
            build_filter("n", "match", &serde_json::json!({"value": null})),
            None
        );
        // As the sole clause, the whole filter is dropped (no `bool.must`).
        let conditions = serde_json::json!({"and": [{"n": {"match": {"value": [1, 2]}}}]});
        assert_eq!(parse_es_conditions(&conditions), None);
    }

    #[test]
    fn test_id_to_uuid_hex_one() {
        assert_eq!(id_to_uuid_hex(1), "00000000000000000000000000000001");
    }

    #[test]
    fn test_id_to_uuid_hex_large() {
        assert_eq!(id_to_uuid_hex(255), "000000000000000000000000000000ff");
    }

    #[test]
    fn test_id_to_uuid_hex_typical_id() {
        assert_eq!(id_to_uuid_hex(12345), "00000000000000000000000000003039");
    }

    #[test]
    fn test_build_base_url_includes_credentials() {
        std::env::remove_var("ELASTIC_API_KEY");
        let url = build_base_url("myhost", 9200);
        assert!(
            url.contains("@myhost:9200"),
            "URL should contain auth@host:port"
        );
        assert!(url.starts_with("http://"), "URL should start with http://");
    }

    #[test]
    fn test_build_base_url_with_http_scheme() {
        std::env::remove_var("ELASTIC_API_KEY");
        let url = build_base_url("http://myhost", 9200);
        assert!(!url.contains("http://http://"));
        assert!(url.contains("@myhost:9200"));
    }

    #[test]
    fn test_build_base_url_with_https_scheme() {
        std::env::remove_var("ELASTIC_API_KEY");
        let url = build_base_url("https://myhost", 9200);
        assert!(url.starts_with("https://"));
        assert!(url.contains("@myhost:9200"));
    }

    #[test]
    fn test_config_parsing_defaults() {
        let config = EngineConfig {
            name: "test-es".to_string(),
            engine: Some("elasticsearch".to_string()),
            algorithm: None,
            connection_params: None,
            collection_params: None,
            search_params: None,
            upload_params: None,
            skip_vector_index: false,
        };
        let engine = ElasticsearchEngine::new(&config, "localhost").unwrap();
        assert_eq!(engine.name, "test-es");
        assert_eq!(engine.config.m, 16);
        assert_eq!(engine.config.ef_construction, 100);
        assert_eq!(engine.config.batch_size, 500);
        assert_eq!(engine.config.parallel, 16);
    }

    #[test]
    fn test_config_parsing_custom_values() {
        let config = EngineConfig {
            name: "test-es-custom".to_string(),
            engine: Some("elasticsearch".to_string()),
            algorithm: None,
            connection_params: None,
            collection_params: Some(crate::config::CollectionParams {
                hnsw_config: None,
                index_options: Some(crate::config::IndexOptions {
                    m: Some(32),
                    ef_construction: Some(256),
                }),
                extra: None,
            }),
            search_params: None,
            upload_params: Some(serde_json::json!({
                "parallel": 8,
                "batch_size": 1000
            })),
            skip_vector_index: false,
        };
        let engine = ElasticsearchEngine::new(&config, "localhost").unwrap();
        assert_eq!(engine.config.m, 32);
        assert_eq!(engine.config.ef_construction, 256);
        assert_eq!(engine.config.batch_size, 1000);
        assert_eq!(engine.config.parallel, 8);
    }

    // ── parse_es_conditions bool-query shape (mirrors OpenSearch tests) ──

    #[test]
    fn es_or_only_sets_minimum_should_match_no_empty_must() {
        // FAILING-then-fixed: before the fix this emitted `must:[]` and no
        // `minimum_should_match`, so ES matched ALL docs (unfiltered).
        let conditions = serde_json::json!({"or": [{"a": {"match": {"value": 1}}}]});
        let parsed = parse_es_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];

        assert_eq!(bool_query["minimum_should_match"], 1);
        assert_eq!(bool_query["should"].as_array().unwrap().len(), 1);
        // No empty `must` array should be emitted for an OR-only filter.
        assert!(bool_query.get("must").is_none());
    }

    #[test]
    fn es_and_only_omits_should() {
        let conditions = serde_json::json!({"and": [{"a": {"match": {"value": 1}}}]});
        let parsed = parse_es_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];

        assert_eq!(bool_query["must"].as_array().unwrap().len(), 1);
        assert!(bool_query.get("should").is_none());
        assert!(bool_query.get("minimum_should_match").is_none());
    }

    #[test]
    fn es_and_or_combined_keeps_both_and_min_should() {
        let conditions = serde_json::json!({
            "and": [{"a": {"match": {"value": 1}}}],
            "or": [{"b": {"match": {"value": 2}}}],
        });
        let parsed = parse_es_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];

        assert_eq!(bool_query["must"].as_array().unwrap().len(), 1);
        assert_eq!(bool_query["should"].as_array().unwrap().len(), 1);
        // minimum_should_match:1 keeps the OR restrictive even alongside must.
        assert_eq!(bool_query["minimum_should_match"], 1);
    }

    // Nested/grouped boolean: `(color==red AND size>=50) OR (color==blue AND
    // size<10)` must nest each AND-group as its OWN `{bool:{must:[...]}}` inside
    // the parent `should`, NOT flatten the four leaves up into one should
    // (which would union the wrong set → recall collapse).
    #[test]
    fn es_nested_or_of_and_groups_nests_natively() {
        let conditions = serde_json::json!({
            "or": [
                { "and": [
                    {"color": {"match": {"value": "red"}}},
                    {"size": {"range": {"gte": 50}}}
                ]},
                { "and": [
                    {"color": {"match": {"value": "blue"}}},
                    {"size": {"range": {"lt": 10}}}
                ]}
            ]
        });
        let parsed = parse_es_conditions(&conditions).expect("nested filter must parse");
        assert_eq!(
            parsed,
            serde_json::json!({
                "bool": {
                    "should": [
                        {"bool": {"must": [
                            {"match": {"color": "red"}},
                            {"range": {"size": {"gte": 50}}}
                        ]}},
                        {"bool": {"must": [
                            {"match": {"color": "blue"}},
                            {"range": {"size": {"lt": 10}}}
                        ]}}
                    ],
                    "minimum_should_match": 1
                }
            })
        );
    }

    #[test]
    fn es_empty_conditions_return_none() {
        assert!(parse_es_conditions(&serde_json::json!({})).is_none());
        // Present-but-empty sub-arrays should not produce a filter either.
        assert!(parse_es_conditions(&serde_json::json!({"and": [], "or": []})).is_none());
    }

    // ── Range operators ────────────────────────────────────────────────────

    #[test]
    fn range_lt_lte_gt_gte_map_to_es_range() {
        assert_eq!(
            build_filter("n", "range", &serde_json::json!({"lt":5})).unwrap(),
            serde_json::json!({"range":{"n":{"lt":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &serde_json::json!({"lte":5})).unwrap(),
            serde_json::json!({"range":{"n":{"lte":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &serde_json::json!({"gt":5})).unwrap(),
            serde_json::json!({"range":{"n":{"gt":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &serde_json::json!({"gte":5})).unwrap(),
            serde_json::json!({"range":{"n":{"gte":5}}})
        );
    }

    #[test]
    fn range_two_sided_keeps_both_bounds() {
        assert_eq!(
            build_filter("n", "range", &serde_json::json!({"gte":10,"lt":20})).unwrap(),
            serde_json::json!({"range":{"n":{"gte":10,"lt":20}}})
        );
    }

    #[test]
    fn range_unknown_op_yields_empty_range_object() {
        assert_eq!(
            build_filter("n", "range", &serde_json::json!({"foo":5})).unwrap(),
            serde_json::json!({"range":{"n":{}}})
        );
    }

    #[test]
    fn range_null_bound_is_skipped() {
        assert_eq!(
            build_filter(
                "n",
                "range",
                &serde_json::json!({"gte":serde_json::Value::Null})
            )
            .unwrap(),
            serde_json::json!({"range":{"n":{}}})
        );
    }

    // ── Geo filter ─────────────────────────────────────────────────────────

    #[test]
    fn geo_with_radius_emits_geo_distance() {
        assert_eq!(
            build_filter(
                "loc",
                "geo",
                &serde_json::json!({"lat":20.0,"lon":10.0,"radius":500})
            )
            .unwrap(),
            serde_json::json!({"geo_distance":{"distance":"500m","loc":{"lat":20.0,"lon":10.0}}})
        );
    }

    #[test]
    fn geo_without_radius_uses_default_1000m() {
        assert_eq!(
            build_filter("loc", "geo", &serde_json::json!({"lat":20.0,"lon":10.0})).unwrap(),
            serde_json::json!({"geo_distance":{"distance":"1000m","loc":{"lat":20.0,"lon":10.0}}})
        );
    }

    #[test]
    fn geo_missing_lat_or_lon_is_none() {
        assert!(build_filter("loc", "geo", &serde_json::json!({"lon":10.0,"radius":5})).is_none());
        assert!(build_filter("loc", "geo", &serde_json::json!({"lat":20.0,"radius":5})).is_none());
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn es_similarity_covers_all_arms() {
        assert_eq!(es_similarity("l2").unwrap(), "l2_norm");
        assert_eq!(es_similarity("euclidean").unwrap(), "l2_norm");
        assert_eq!(es_similarity("cosine").unwrap(), "cosine");
        assert_eq!(es_similarity("angular").unwrap(), "cosine");
        assert_eq!(es_similarity("COSINE").unwrap(), "cosine");
        // dot/ip unsupported by ES; unknown errors too.
        assert!(es_similarity("dot").is_err());
        assert!(es_similarity("ip").is_err());
        assert!(es_similarity("nope").is_err());
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    #[test]
    fn exact_match_int_float_bool_pass_through_match() {
        assert_eq!(
            build_filter("n", "match", &serde_json::json!({"value":5})).unwrap(),
            serde_json::json!({"match":{"n":5}})
        );
        assert_eq!(
            build_filter("n", "match", &serde_json::json!({"value":1.5})).unwrap(),
            serde_json::json!({"match":{"n":1.5}})
        );
        assert_eq!(
            build_filter("flag", "match", &serde_json::json!({"value":true})).unwrap(),
            serde_json::json!({"match":{"flag":true}})
        );
    }

    #[test]
    fn exact_match_array_value_is_none() {
        // #121: ES build_filter now guards the scalar exact-match arm; a
        // non-scalar value is dropped (None), matching qdrant/redis/valkey/
        // vectorsets (was previously forwarded verbatim as `{"match":{"n":[1,2]}}`).
        assert_eq!(
            build_filter("n", "match", &serde_json::json!({"value":[1,2]})),
            None
        );
    }

    // ── uuid_hex_to_int round-trip + invalid input ─────────────────────────
    #[test]
    fn uuid_hex_to_int_round_trips_with_id_to_uuid_hex() {
        for id in [0i64, 1, 255, 12345, 9_999_999] {
            let hex = id_to_uuid_hex(id);
            assert_eq!(uuid_hex_to_int(&hex).unwrap(), id, "round-trip id={}", id);
        }
    }

    #[test]
    fn uuid_hex_to_int_rejects_invalid_hex() {
        let err = uuid_hex_to_int("not-a-uuid").unwrap_err();
        assert!(
            err.starts_with("Invalid UUID hex 'not-a-uuid':"),
            "err={}",
            err
        );
    }

    // ── extract_knn_hits: happy path + missing-field errors ────────────────
    #[test]
    fn extract_knn_hits_reads_id_and_score() {
        let body = serde_json::json!({
            "hits": {"hits": [
                {"_id": id_to_uuid_hex(7), "_score": 0.9},
                {"_id": id_to_uuid_hex(3), "_score": 0.5},
            ]}
        });
        assert_eq!(extract_knn_hits(&body).unwrap(), vec![(7, 0.9), (3, 0.5)]);
    }

    #[test]
    fn extract_knn_hits_missing_hits_hits_errors() {
        let body = serde_json::json!({"hits": {"total": 0}});
        assert_eq!(
            extract_knn_hits(&body).unwrap_err(),
            "Missing hits.hits in search response"
        );
    }

    #[test]
    fn extract_knn_hits_missing_id_errors() {
        let body = serde_json::json!({"hits": {"hits": [{"_score": 0.9}]}});
        assert_eq!(extract_knn_hits(&body).unwrap_err(), "Missing _id in hit");
    }

    #[test]
    fn extract_knn_hits_missing_score_defaults_zero() {
        // A hit without `_score` is not an error — score defaults to 0.0.
        let body = serde_json::json!({"hits": {"hits": [{"_id": id_to_uuid_hex(4)}]}});
        assert_eq!(extract_knn_hits(&body).unwrap(), vec![(4, 0.0)]);
    }

    // ── resolve_index_similarity: distance mapping + rejections ────────────
    #[test]
    fn resolve_index_similarity_maps_and_rejects() {
        assert_eq!(resolve_index_similarity("cosine", 128).unwrap(), "cosine");
        assert_eq!(resolve_index_similarity("l2", 2048).unwrap(), "l2_norm");
        // dot/ip rejected with the DOT-specific message (before the dim check).
        assert_eq!(
            resolve_index_similarity("dot", 128).unwrap_err(),
            "Elasticsearch does not support DOT product distance for benchmarking"
        );
        assert!(resolve_index_similarity("ip", 128).is_err());
        // dim > 2048 rejected.
        assert_eq!(
            resolve_index_similarity("cosine", 4096).unwrap_err(),
            "Elasticsearch does not support vector_size > 2048 (got 4096)"
        );
        // Unknown distance still errors (via es_similarity).
        assert!(resolve_index_similarity("nope", 128).is_err());
    }
}
