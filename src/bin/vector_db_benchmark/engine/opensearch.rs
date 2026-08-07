//! OpenSearch engine implementation.
//!
//! Uses the official `opensearch` crate (async, wrapped with tokio block_on).
//! Very similar to Elasticsearch but uses knn_vector type and different query format.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use opensearch::http::request::JsonBody;
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use opensearch::indices::{
    IndicesCreateParts, IndicesDeleteParts, IndicesForcemergeParts, IndicesPutSettingsParts,
    IndicesRefreshParts,
};
use opensearch::{BulkParts, OpenSearch, SearchParts};
use uuid::Uuid;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

#[derive(Clone)]
struct OpenSearchConfig {
    m: i64,
    ef_construction: i64,
    batch_size: usize,
    parallel: usize,
    /// None = inherit the cluster default (previous behaviour).
    number_of_shards: Option<i64>,
}

pub struct OpenSearchEngine {
    name: String,
    index_name: String,
    #[allow(dead_code)]
    timeout: u64,
    config: OpenSearchConfig,
    search_params: Vec<SearchParams>,
    /// Base URL for constructing per-thread clients
    base_url: String,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared OpenSearch client
    client: Arc<OpenSearch>,
}

impl OpenSearchEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("OPENSEARCH_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9200);

        let index_name = std::env::var("OPENSEARCH_INDEX").unwrap_or_else(|_| "bench".to_string());
        let timeout: u64 = std::env::var("OPENSEARCH_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        // Extract HNSW config from collection_params.method.parameters (OpenSearch format)
        // or fall back to index_options (ES format) or defaults
        let (m, ef_construction) = extract_hnsw_params(engine_config);

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

        // See create_index: shard count materially changes OpenSearch vector
        // performance and the Service default has differed from open-source
        // OpenSearch, so it must be settable rather than inherited silently.
        let number_of_shards = engine_config
            .collection_params
            .as_ref()
            .and_then(|c| c.extra.as_ref())
            .and_then(|e| e.get("number_of_shards"))
            .and_then(|v| v.as_i64());

        let base_url = build_base_url(host, port);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = create_os_client(&base_url, timeout)?;

        Ok(Self {
            name: engine_config.name.clone(),
            index_name,
            timeout,
            config: OpenSearchConfig {
                m,
                ef_construction,
                batch_size,
                parallel,
                number_of_shards,
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

    /// Delete the benchmark index, tolerating the transient states a *managed*
    /// cluster puts an index into.
    ///
    /// Amazon OpenSearch Service takes automated snapshots on a schedule that
    /// cannot be disabled, and deleting an index caught in one fails with
    /// HTTP 400 `snapshot_in_progress_exception`. Because every config in a grid
    /// starts by dropping the previous index, a single unlucky snapshot window
    /// otherwise fails not just that config but every config after it — observed
    /// as 1/6 completing and 5/6 dying on "Failed to delete index: status 400".
    ///
    /// 503 is the other transient case (cluster-manager busy). Both are retried;
    /// anything else fails immediately.
    ///
    /// The response body is included in the error. Without it "status 400" is
    /// undiagnosable — the cause has to be guessed from the outside.
    fn delete_index(&self) -> Result<(), String> {
        let max_retries: u32 = std::env::var("OPENSEARCH_INDEX_OP_MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let base_delay_ms: u64 = std::env::var("OPENSEARCH_INDEX_OP_RETRY_BASE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2_000);

        let mut attempt: u32 = 0;
        loop {
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
                return Ok(());
            }

            let body = self.rt.block_on(resp.text()).unwrap_or_default();
            let retryable = status == 503
                || (status == 400 && body.contains("snapshot_in_progress"))
                || body.contains("cluster_block_exception");

            if !retryable || attempt >= max_retries {
                return Err(format!(
                    "Failed to delete index: status {} after {} retries: {}",
                    status, attempt, body
                ));
            }
            backoff_sleep(base_delay_ms, attempt);
            attempt += 1;
        }
    }

    fn create_index(&self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let dist_lower = distance.to_lowercase();
        let space_type = resolve_index_space_type(&dist_lower, vector_size)?;

        // Build properties with knn_vector type
        let mut properties = serde_json::json!({
            "vector": {
                "type": "knn_vector",
                "dimension": vector_size,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": space_type,
                    "parameters": {
                        "m": self.config.m,
                        "ef_construction": self.config.ef_construction,
                    }
                }
            }
        });

        // Add schema fields from dataset config
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                let props = properties.as_object_mut().unwrap();
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let os_type = match ft {
                        "int" => "long",
                        "geo" => "geo_point",
                        // "bool"/"datetime" are not valid OS types; map to the real
                        // ones. OS coerces the reader's "true"/"false" string into a
                        // `boolean` field and parses ISO-8601 into a `date` field.
                        // Forwarding them verbatim made index creation reject the
                        // whole mapping.
                        "bool" => "boolean",
                        "datetime" => "date",
                        // A `uuid` is an exact-match opaque string; "uuid" is not a
                        // valid OS type, so (like bool/datetime above) forwarding it
                        // verbatim made index creation reject the whole mapping,
                        // silently breaking every uuid-equality filter. Map it to
                        // `keyword` (exact term match, no analysis).
                        "uuid" => "keyword",
                        other => other,
                    };
                    props.insert(
                        field_name.clone(),
                        serde_json::json!({
                            "type": os_type,
                            "index": true,
                        }),
                    );
                }
            }
        }

        // Shard count is a first-order factor for OpenSearch vector search, not a
        // detail: Amazon OpenSearch Service historically defaulted an index to FIVE
        // primary shards while open-source OpenSearch defaults to one, and an
        // internal 2024 benchmark that tested both found 5 clearly better for
        // precision vs qps/latency/index-time — the 1-shard config "deeply impacts
        // OpenSearch indexing speed and precision". Leaving it unset silently
        // inherits whatever the cluster default happens to be for that version,
        // which makes runs incomparable across versions and can hand OpenSearch the
        // worse of two configurations without anyone choosing it.
        //
        // Set from collection_params.number_of_shards; unset preserves the previous
        // behaviour (cluster default) so existing configs are unaffected.
        let mut index_settings = serde_json::json!({
            "knn": true,
            // Indexing-throughput tuning: no replicas and no periodic
            // refresh, since the benchmark bulk-loads all data up front and
            // force-merges before searching.
            "number_of_replicas": 0,
            "refresh_interval": -1,
        });
        if let Some(n) = self.config.number_of_shards {
            index_settings["number_of_shards"] = serde_json::Value::from(n);
        }

        let body = serde_json::json!({
            "settings": { "index": index_settings },
            "mappings": {
                "properties": properties,
            }
        });

        // Same transient-state problem as delete_index: a cluster whose manager
        // thread pool is busy answers create-index with
        // `process_cluster_event_timeout_exception` (503) rather than creating it.
        // Retrying rides that out instead of failing the config — and every config
        // after it, since each one starts by creating its index.
        let max_retries: u32 = std::env::var("OPENSEARCH_INDEX_OP_MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let base_delay_ms: u64 = std::env::var("OPENSEARCH_INDEX_OP_RETRY_BASE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2_000);

        let mut attempt: u32 = 0;
        loop {
            let resp = self
                .rt
                .block_on(
                    self.client
                        .indices()
                        .create(IndicesCreateParts::Index(&self.index_name))
                        .body(body.clone())
                        .send(),
                )
                .map_err(|e| format!("Failed to create index: {}", e))?;

            if resp.status_code().is_success() {
                return Ok(());
            }

            let status = resp.status_code().as_u16();
            let text = self.rt.block_on(resp.text()).unwrap_or_default();
            let retryable = status == 503
                || status == 429
                || text.contains("process_cluster_event_timeout_exception")
                || text.contains("cluster_block_exception");

            if !retryable || attempt >= max_retries {
                return Err(format!(
                    "Failed to create index (HTTP {}, {} retries): {}",
                    status, attempt, text
                ));
            }
            backoff_sleep(base_delay_ms, attempt);
            attempt += 1;
        }
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

                    let client = match create_os_client(&base_url, timeout) {
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

    /// Refresh the index to make just-uploaded documents searchable. Required
    /// because we set `refresh_interval: -1` (no periodic refresh) during upload.
    fn refresh(&self) -> Result<(), String> {
        let resp = self
            .rt
            .block_on(
                self.client
                    .indices()
                    .refresh(IndicesRefreshParts::Index(&[&self.index_name]))
                    .send(),
            )
            .map_err(|e| format!("Refresh failed: {}", e))?;

        if !resp.status_code().is_success() {
            let text = self.rt.block_on(resp.text()).unwrap_or_default();
            return Err(format!("Refresh error: {}", text));
        }
        Ok(())
    }

    fn force_merge(&self) -> Result<(), String> {
        println!("Forcing merge...");

        let resp = self
            .rt
            .block_on(
                self.client
                    .indices()
                    .forcemerge(IndicesForcemergeParts::Index(&[&self.index_name]))
                    .send(),
            )
            .map_err(|e| format!("Force merge failed: {}", e))?;

        if !resp.status_code().is_success() {
            let text = self.rt.block_on(resp.text()).unwrap_or_default();
            return Err(format!("Force merge error: {}", text));
        }
        Ok(())
    }

    /// Apply search-time settings (e.g., knn.algo_param.ef_search)
    fn setup_search(&self, params: &SearchParams) -> Result<(), String> {
        // Cold-cache graph loading is warmed uniformly by the per-worker prime
        // query in `search` (mirrors redis/vertex), so no engine-specific
        // server-side `_knn/warmup` call is needed here.
        let ef_search = params
            .extra
            .as_ref()
            .and_then(|e| e.get("knn.algo_param.ef_search"))
            .and_then(|v| v.as_i64());

        if let Some(ef) = ef_search {
            let body = serde_json::json!({
                "index": {
                    "knn.algo_param.ef_search": ef,
                }
            });

            let resp = self
                .rt
                .block_on(
                    self.client
                        .indices()
                        .put_settings(IndicesPutSettingsParts::Index(&[&self.index_name]))
                        .body(body)
                        .send(),
                )
                .map_err(|e| format!("Failed to apply search settings: {}", e))?;

            if !resp.status_code().is_success() {
                let text = self.rt.block_on(resp.text()).unwrap_or_default();
                eprintln!("Warning: failed to set ef_search={}: {}", ef, text);
            }
        }
        Ok(())
    }
}

/// Extract m and ef_construction from OpenSearch collection_params.
/// Supports: collection_params.method.parameters.{m, ef_construction}
/// Falls back to: collection_params.index_options.{m, ef_construction}
fn extract_hnsw_params(engine_config: &EngineConfig) -> (i64, i64) {
    if let Some(cp) = &engine_config.collection_params {
        // Try OpenSearch format: method.parameters
        if let Some(extra) = &cp.extra {
            if let Some(method) = extra.get("method") {
                if let Some(params) = method.get("parameters") {
                    let m = params.get("m").and_then(|v| v.as_i64()).unwrap_or(16);
                    let ef = params
                        .get("ef_construction")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(100);
                    return (m, ef);
                }
            }
        }
        // Try ES format: index_options
        if let Some(io) = &cp.index_options {
            return (io.m.unwrap_or(16), io.ef_construction.unwrap_or(100));
        }
    }
    (16, 100)
}

fn build_base_url(host: &str, port: u16) -> String {
    let user = std::env::var("OPENSEARCH_USER").unwrap_or_else(|_| "admin".to_string());
    let password = std::env::var("OPENSEARCH_PASSWORD").unwrap_or_else(|_| "admin".to_string());

    let scheme_host = if host.starts_with("http") {
        host.to_string()
    } else {
        format!("https://{}", host)
    };

    if let Some(rest) = scheme_host.strip_prefix("http://") {
        format!("http://{}:{}@{}:{}", user, password, rest, port)
    } else if let Some(rest) = scheme_host.strip_prefix("https://") {
        format!("https://{}:{}@{}:{}", user, password, rest, port)
    } else {
        format!("https://{}:{}@{}:{}", user, password, scheme_host, port)
    }
}

/// Create an OpenSearch client from a base URL.
fn create_os_client(base_url: &str, timeout: u64) -> Result<OpenSearch, String> {
    let url = opensearch::http::Url::parse(base_url)
        .map_err(|e| format!("Invalid base URL '{}': {}", base_url, e))?;
    let pool = SingleNodeConnectionPool::new(url);
    let transport = TransportBuilder::new(pool)
        .timeout(std::time::Duration::from_secs(timeout))
        .disable_proxy()
        .cert_validation(opensearch::cert::CertificateValidation::None)
        .build()
        .map_err(|e| format!("Failed to build transport: {}", e))?;
    Ok(OpenSearch::new(transport))
}

fn id_to_uuid_hex(id: i64) -> String {
    Uuid::from_u128(id as u128).as_simple().to_string()
}

fn uuid_hex_to_int(hex: &str) -> Result<i64, String> {
    let uuid = Uuid::parse_str(hex).map_err(|e| format!("Invalid UUID hex '{}': {}", hex, e))?;
    Ok(uuid.as_u128() as i64)
}

/// Validate index build parameters and resolve the knn `space_type`. Extracted
/// verbatim from `create_index` so the guard order + error strings are unit-
/// testable without a live OpenSearch. `dist_lower` must already be lowercased.
/// dot/ip is rejected first, then the dim cap, then the general mapping.
fn resolve_index_space_type(dist_lower: &str, vector_size: i64) -> Result<&'static str, String> {
    if dist_lower == "dot" || dist_lower == "ip" {
        return Err("OpenSearch does not support DOT product distance".to_string());
    }
    if vector_size > 2048 {
        return Err(format!(
            "OpenSearch does not support vector_size > 2048 (got {})",
            vector_size
        ));
    }
    os_space_type(dist_lower)
}

/// Map a dataset distance name to the OpenSearch knn `space_type`. `dot`/`ip`
/// is unsupported and unknown metrics error. A wrong arm here would silently
/// change ranking, so every arm is unit-tested.
fn os_space_type(distance: &str) -> Result<&'static str, String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok("l2"),
        "cosine" | "angular" => Ok("cosinesimil"),
        "dot" | "ip" => Err("OpenSearch does not support DOT product distance".to_string()),
        other => Err(format!(
            "Unsupported distance metric for OpenSearch: {}",
            other
        )),
    }
}

/// Parse conditions into OpenSearch bool query (same DSL as Elasticsearch).
fn parse_os_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    build_group(obj)
}

/// Build one boolean GROUP (a `{and:[...], or:[...]}` object) into an OpenSearch
/// `{bool: ...}` query. Recursive: an entry inside `and`/`or` may itself be a
/// nested group, so this and [`build_subfilters`] call each other. A nested group
/// becomes its own `{bool: ...}` object placed inside the parent's `must`/`should`
/// array — OpenSearch's native bool nesting — so `(a AND b) OR (c AND d)` is
/// preserved instead of being silently flattened into one flat clause list.
fn build_group(obj: &serde_json::Map<String, serde_json::Value>) -> Option<serde_json::Value> {
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

    let mut bool_query = serde_json::Map::new();
    if let Some(must) = and_filters {
        bool_query.insert("must".to_string(), serde_json::Value::Array(must));
    }
    if let Some(should) = or_filters {
        bool_query.insert("should".to_string(), serde_json::Value::Array(should));
        // Force OR filters to actually restrict results. `minimum_should_match`
        // defaults to 0 as soon as a `must` clause is also present, which would
        // silently drop the OR condition in a mixed AND+OR filter.
        bool_query.insert(
            "minimum_should_match".to_string(),
            serde_json::Value::from(1),
        );
    }

    Some(serde_json::json!({ "bool": bool_query }))
}

/// Build individual subfilters from an array of condition entries. Each entry is
/// either a nested boolean group (`{and:[...]}` / `{or:[...]}`) — recursed via
/// [`build_group`] into a nested `{bool: ...}` — or a leaf `{field: {op: criteria}}`.
fn build_subfilters(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            // Nested group: an entry carrying an `and`/`or` key is a sub-tree,
            // not a field leaf. Recurse and nest it as its own `{bool: ...}`.
            if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
                if let Some(f) = build_group(entry_obj) {
                    filters.push(f);
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
            // leave `bool.must:[]`, which OpenSearch treats as match-ALL —
            // silently returning unfiltered results, the inverse of the filter.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                return Some(serde_json::json!({"terms": {field_name: any}}));
            }
            // Full-text `{"match": {"text": …}}` conditions: emit an OpenSearch
            // `match` query against the analyzed `text` field. `match` tokenizes
            // the query the same way the field is analyzed, so it matches docs
            // CONTAINING the term(s) — aligning with the tokenized semantics
            // redis uses via `@field:($tok)` and the ground truth in
            // `write_fulltext_project` (docs whose body CONTAINS "quick"). We use
            // `match` (not `match_phrase`) because the fixture filters on single
            // tokens; dropping this clause would leave `bool.must:[]`, which
            // OpenSearch treats as match-ALL — silently running the kNN query
            // UNFILTERED while recall is scored against filtered ground truth
            // (#120).
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

/// Upload a batch using the official OpenSearch bulk API.
fn upload_bulk_batch(
    rt: &tokio::runtime::Runtime,
    client: &OpenSearch,
    index_name: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    // Held as plain Values rather than JsonBody so the batch can be rebuilt and
    // resent on a retry: `.body()` consumes the Vec and JsonBody is not Clone.
    let mut lines: Vec<serde_json::Value> = Vec::with_capacity(ids.len() * 2);

    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);

        // Action line
        lines.push(serde_json::json!({"index": {"_id": uuid_hex}}));

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

        lines.push(serde_json::Value::Object(doc));
    }

    // HTTP 429 (and 503) are back-pressure, not failure: the server is telling us
    // to slow down and come back. Aborting the experiment on the first one makes
    // ingest impossible against a managed service that sheds load — Amazon
    // OpenSearch Service returns 429 readily on a single-node domain, because
    // knn.algo_param.index_thread_qty defaults to 1 and the write path cannot
    // drain as fast as a parallel uploader pushes. Retry with exponential backoff
    // instead, and only give up once the server has had several chances.
    //
    // Rejections also arrive as HTTP 200 with per-item errors (bulk is partial by
    // design), so item-level 429 / es_rejected_execution_exception /
    // circuit_breaking_exception are treated as retryable too.
    let max_retries: u32 = std::env::var("OPENSEARCH_BULK_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let base_delay_ms: u64 = std::env::var("OPENSEARCH_BULK_RETRY_BASE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    let mut attempt: u32 = 0;
    loop {
        let body: Vec<JsonBody<serde_json::Value>> =
            lines.iter().cloned().map(JsonBody::new).collect();

        // A dropped connection or read timeout fails here, BEFORE any HTTP status
        // exists, so status-based retry never sees it. Over a multi-hour ingest a
        // single transient reset would otherwise discard the whole run.
        let resp = match rt.block_on(client.bulk(BulkParts::Index(index_name)).body(body).send()) {
            Ok(r) => r,
            Err(e) => {
                if attempt >= max_retries {
                    return Err(format!(
                        "Bulk upload failed after {} retries: {} \
                         (transport error — the server may be dropping connections \
                         under load; lower upload_params.parallel or batch_size)",
                        max_retries, e
                    ));
                }
                backoff_sleep(base_delay_ms, attempt);
                attempt += 1;
                continue;
            }
        };

        let status = resp.status_code().as_u16();
        // 429 = write queue full. 503 = unavailable. 502/504 = the managed
        // service's front door gave up, which happens both transiently and when a
        // single bulk request is simply too big — retrying distinguishes them,
        // since a size problem survives every attempt.
        let http_retryable = matches!(status, 429 | 502 | 503 | 504);

        if !http_retryable && !resp.status_code().is_success() {
            let text = rt.block_on(resp.text()).unwrap_or_default();
            return Err(format!("Bulk upload error: HTTP {}: {}", status, text));
        }

        if http_retryable {
            if attempt >= max_retries {
                let text = rt.block_on(resp.text()).unwrap_or_default();
                let hint = if matches!(status, 502 | 504 | 413) {
                    "request is likely too large — lower upload_params.batch_size \
                     (bulk bytes scale with vector dimension)"
                } else {
                    "server is shedding load — lower upload_params.parallel or \
                     raise OPENSEARCH_BULK_MAX_RETRIES"
                };
                return Err(format!(
                    "Bulk upload error: HTTP {} still returned after {} retries ({}): {}",
                    status, max_retries, hint, text
                ));
            }
            backoff_sleep(base_delay_ms, attempt);
            attempt += 1;
            continue;
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
            let retryable_count = items
                .map(|arr| arr.iter().filter(|item| item_is_retryable(item)).count())
                .unwrap_or(0);

            // Only retry when every failure is retryable; a genuine mapping or
            // parse error would otherwise be retried pointlessly to exhaustion.
            if retryable_count > 0 && retryable_count == error_count && attempt < max_retries {
                backoff_sleep(base_delay_ms, attempt);
                attempt += 1;
                continue;
            }

            return Err(format!(
                "Bulk upload had {} errors out of {} documents ({} retryable, {} retries used)",
                error_count,
                ids.len(),
                retryable_count,
                attempt
            ));
        }

        return Ok(());
    }
}

/// Exponential backoff, capped so a long stall cannot sleep for hours.
fn backoff_sleep(base_delay_ms: u64, attempt: u32) {
    let delay = base_delay_ms
        .saturating_mul(1u64 << attempt.min(6))
        .min(30_000);
    std::thread::sleep(std::time::Duration::from_millis(delay));
}

/// True when a bulk item failed for a reason worth retrying — the server shedding
/// load rather than rejecting the document itself.
fn item_is_retryable(item: &serde_json::Value) -> bool {
    let Some(idx) = item.get("index") else {
        return false;
    };
    if idx.get("status").and_then(|s| s.as_u64()) == Some(429) {
        return true;
    }
    matches!(
        idx.get("error")
            .and_then(|e| e.get("type"))
            .and_then(|t| t.as_str()),
        Some("es_rejected_execution_exception")
            | Some("circuit_breaking_exception")
            | Some("cluster_block_exception")
    )
}

/// OpenSearch KNN search (different format from Elasticsearch).
/// Uses {"query": {"knn": {"vector": {"vector": [...], "k": top}}}} format.
/// Build the OpenSearch kNN search body.
///
/// Efficient (pre-)filtering: the filter is pushed *inside* the kNN clause so the
/// Lucene engine applies it during graph traversal. Wrapping the kNN query in an
/// outer `bool.must` + `filter` instead performs post-filtering, which collapses
/// recall on filtered datasets (see qdrant/vector-db-benchmark#167). Requires the
/// `lucene` engine, which our index mapping uses (see `configure`).
fn build_knn_body(
    query_vector: &[f32],
    top: usize,
    filter: Option<&serde_json::Value>,
) -> serde_json::Value {
    let mut query = serde_json::json!({
        "knn": {
            "vector": {
                "vector": query_vector,
                "k": top,
            }
        }
    });

    if let Some(f) = filter {
        query["knn"]["vector"]["filter"] = f.clone();
    }

    // Response trimming: the benchmark only needs each hit's id, so skip loading
    // `_source` and stored fields and return `_id` via a doc-value field. This
    // trims the response payload for a fairer QPS/latency measurement.
    serde_json::json!({
        "query": query,
        "size": top,
        "_source": false,
        "docvalue_fields": ["_id"],
        "stored_fields": "_none_",
    })
}

/// Extract the document id from a search hit. With response trimming the id is
/// returned as a doc-value under `fields._id[0]`; fall back to the top-level
/// `_id` for untrimmed responses.
fn hit_id(hit: &serde_json::Value) -> Option<&str> {
    hit.get("fields")
        .and_then(|f| f.get("_id"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .or_else(|| hit.get("_id").and_then(|v| v.as_str()))
}

/// Send a pre-serialized KNN search request and return the DECODED response.
/// The consistent timed boundary (see qdrant/pgvector/redis) is: request body
/// pre-serialized to a `RawValue` OUTSIDE the window (the vector-to-JSON ryu
/// formatting is client CPU work); RPC send + receive + decode-to-structured-
/// response INSIDE the window (this fn: send + status check + wire read +
/// `from_str`); id/score extraction OUTSIDE (`extract_knn_hits`). So the JSON
/// decode is billed as latency exactly like qdrant's protobuf decode.
fn knn_send(
    rt: &tokio::runtime::Runtime,
    client: &OpenSearch,
    index_name: &str,
    raw_body: &serde_json::value::RawValue,
) -> Result<serde_json::Value, String> {
    // Search-path back-pressure is the most dangerous of the lot, because the
    // caller does not fail on it — a failed query is logged and DROPPED from the
    // timing and recall vectors, so mean_precisions ends up computed over only the
    // queries that survived. Since 429s arrive precisely when the node is loaded,
    // the survivors are the cheaper queries and recall is biased UPWARD, silently,
    // with nothing in the summary recording that anything was lost.
    //
    // Measured: a glove-100 run shed 17,584 queries across two of six configs and
    // still reported clean-looking recall for them.
    //
    // Retries are kept short by default — a query is on the latency critical path,
    // and a retried query's own measured time is not used for the latency figure
    // (the caller times the whole call), so a long backoff would distort latency.
    // Better to retry a few times quickly and let the run fail loudly than to drop
    // the query.
    let max_retries: u32 = std::env::var("OPENSEARCH_SEARCH_MAX_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let base_delay_ms: u64 = std::env::var("OPENSEARCH_SEARCH_RETRY_BASE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);

    let mut attempt: u32 = 0;
    let resp = loop {
        match rt.block_on(
            client
                .search(SearchParts::Index(&[index_name]))
                .body(raw_body)
                .send(),
        ) {
            Ok(r) => {
                let status = r.status_code().as_u16();
                if r.status_code().is_success() {
                    break r;
                }
                if !matches!(status, 429 | 502 | 503 | 504) || attempt >= max_retries {
                    let text = rt.block_on(r.text()).unwrap_or_default();
                    return Err(format!(
                        "KNN search error (HTTP {}, {} retries): {}",
                        status, attempt, text
                    ));
                }
            }
            Err(e) => {
                if attempt >= max_retries {
                    return Err(format!(
                        "KNN search failed after {} retries: {}",
                        attempt, e
                    ));
                }
            }
        }
        backoff_sleep(base_delay_ms, attempt);
        attempt += 1;
    };

    let text = rt
        .block_on(resp.text())
        .map_err(|e| format!("Failed to read search response: {}", e))?;
    serde_json::from_str(&text).map_err(|e| format!("Failed to parse search response: {}", e))
}

/// Extract the id/score list from an already-decoded response (done AFTER the
/// timed window — only pulling final ids out of the decoded struct for recall,
/// mirroring pgvector/qdrant).
fn extract_knn_hits(resp_body: &serde_json::Value) -> Result<Vec<(i64, f64)>, String> {
    let hits = resp_body
        .get("hits")
        .and_then(|h| h.get("hits"))
        .and_then(|h| h.as_array())
        .ok_or_else(|| "Missing hits.hits in search response".to_string())?;

    let mut results = Vec::with_capacity(hits.len());
    for hit in hits {
        let id_hex = hit_id(hit).ok_or_else(|| "Missing _id in hit".to_string())?;
        let score = hit.get("_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let id = uuid_hex_to_int(id_hex)?;
        results.push((id, score));
    }

    Ok(results)
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for OpenSearchEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        println!(
            "OpenSearch: HNSW {{ m: {}, ef_construction: {} }}",
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
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
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
        // (mirrors mongodb; matches v0's post_upload() timing).
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

        // Apply search-time settings (ef_search)
        self.setup_search(params)?;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<serde_json::Value>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_os_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` and the fully serialized request bodies
        // BEFORE the parallel region so the timed window wraps only the RPC
        // round-trip. `build_knn_body` builds the DOM and `to_raw_value`
        // performs the vector-to-JSON ryu formatting (client CPU work) once here;
        // the timed send only copies the already-formatted bytes. `tops[idx]`
        // reproduces the same k the request embeds, so recall is unchanged.
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
                let body = build_knn_body(&queries[idx], tops[idx], parsed_filters[idx].as_ref());
                serde_json::value::to_raw_value(&body).expect("serialize KNN search body")
            })
            .collect();

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));
        // A dropped query silently biases recall (see knn_search), so failures are
        // counted across all worker threads and the run is refused below if any
        // survived their retries.
        let failed_queries = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

        // Barrier-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window (mirrors the Redis/Vertex engines).
        // Every worker builds its runtime + client and primes with one discarded
        // query, then blocks on `ready`; the main thread stamps the shared start
        // instant into `start_cell` and releases `go`, so the measurement clock
        // starts only once all workers are warm and poised. A worker that fails to
        // set up MUST still pass both barriers before returning, or the run would
        // deadlock.
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
                let failed_queries = Arc::clone(&failed_queries);
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
                    let client = match create_os_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(_) => {
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this connection with ONE discarded query so the cold
                    // first round-trip is not inside the measured window. Best
                    // effort: errors are ignored and its sample is NOT recorded.
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
                                failed_queries.fetch_add(1, Ordering::Relaxed);
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                }));
            }

            // All workers are connected + primed and blocked on `go`. Stamp the
            // shared measurement start and release them simultaneously. The cold
            // setup is already behind the barrier.
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
        // Measure from the post-barrier start stamp (workers already primed), so
        // total_time excludes connection setup and the cold first query.
        let total_time = start_cell
            .get()
            .map(|st| st.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        // Refuse to report a run that lost queries. Dropped queries never reach
        // `precs`/`recs`, so the recall below would be an average over the survivors
        // only — and because the server sheds load exactly when it is busy, the
        // survivors are the cheaper queries and the figure is biased UPWARD. Nothing
        // in the emitted summary records the loss, so a partial run is
        // indistinguishable from a clean one downstream. Failing here is the only
        // thing that stops a biased number being published by accident.
        //
        // OPENSEARCH_ALLOW_FAILED_QUERIES=1 accepts a deliberately best-effort run.
        let failed = failed_queries.load(Ordering::Relaxed);
        if failed > 0 {
            let allow = std::env::var("OPENSEARCH_ALLOW_FAILED_QUERIES")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let pct = 100.0 * failed as f64 / num_to_run.max(1) as f64;
            if !allow {
                return Err(format!(
                    "{} of {} search queries failed ({:.2}%) after retries — refusing to \
                     report recall averaged over the survivors only, which biases it \
                     upward. Reduce load, raise OPENSEARCH_SEARCH_MAX_RETRIES, or set \
                     OPENSEARCH_ALLOW_FAILED_QUERIES=1 to accept a partial run.",
                    failed, num_to_run, pct
                ));
            }
            eprintln!(
                "⚠  {} of {} search queries failed ({:.2}%) — recall below is averaged over \
                 the survivors and is biased upward (OPENSEARCH_ALLOW_FAILED_QUERIES=1)",
                failed, num_to_run, pct
            );
        }

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
    use serde_json::json;

    /// Load-bearing: the hoisted request body is `to_raw_value(&build_knn_body())`.
    /// Its verbatim bytes (what JsonBody writes on the wire) must equal the bytes
    /// the old inline `.body(build_knn_body())` path serialized via `to_vec`.
    #[test]
    fn build_knn_body_raw_value_roundtrips_to_wire_bytes() {
        let vec = vec![0.1f32, -0.2, 0.3];
        let top = 2usize;
        let filter = json!({"term": {"color": "red"}});

        let body = build_knn_body(&vec, top, Some(&filter));
        let to_vec_bytes = serde_json::to_vec(&body).unwrap();
        let raw = serde_json::value::to_raw_value(&body).unwrap();
        assert_eq!(raw.get().as_bytes(), to_vec_bytes.as_slice());

        // Unfiltered variant.
        let body_nf = build_knn_body(&vec, top, None);
        let raw_nf = serde_json::value::to_raw_value(&body_nf).unwrap();
        assert_eq!(
            raw_nf.get().as_bytes(),
            serde_json::to_vec(&body_nf).unwrap().as_slice()
        );
    }

    #[test]
    fn test_match_any_string_list_emits_terms() {
        let c = build_filter("color", "match", &json!({"any": ["red", "blue"]})).unwrap();
        assert_eq!(c, json!({"terms": {"color": ["red", "blue"]}}));
    }

    #[test]
    fn test_match_any_int_list_emits_terms() {
        let c = build_filter("size", "match", &json!({"any": [1, 2, 3]})).unwrap();
        assert_eq!(c, json!({"terms": {"size": [1, 2, 3]}}));
    }

    #[test]
    fn test_match_any_empty_list_matches_nothing() {
        // Empty IN-set must match NOTHING (never invert to match-all): `terms: []`.
        let c = build_filter("color", "match", &json!({"any": []})).unwrap();
        assert_eq!(c, json!({"terms": {"color": []}}));
    }

    #[test]
    fn test_match_exact_value_still_works() {
        let c = build_filter("color", "match", &json!({"value": "red"})).unwrap();
        assert_eq!(c, json!({"match": {"color": "red"}}));
    }

    // #121: a non-scalar `value` (a JSON array) is malformed input — the
    // canonical model uses `match.any` for lists. It must be dropped (None),
    // not forwarded verbatim as `{"match":{"n":[1,2]}}`. Matches qdrant/redis/
    // valkey/vectorsets.
    #[test]
    fn test_match_non_scalar_value_dropped() {
        assert_eq!(build_filter("n", "match", &json!({"value": [1, 2]})), None);
        assert_eq!(
            build_filter("n", "match", &json!({"value": {"x": 1}})),
            None
        );
        assert_eq!(build_filter("n", "match", &json!({"value": null})), None);
        // As the sole clause, the whole filter is dropped (no `bool.must`).
        let conditions = json!({"and": [{"n": {"match": {"value": [1, 2]}}}]});
        assert_eq!(parse_os_conditions(&conditions), None);
    }

    // #120: a full-text `{"match": {"text": …}}` condition must emit an analyzed
    // `match` query — NOT be dropped. Dropping it leaves `bool.must:[]`, which
    // OpenSearch treats as match-ALL, silently running the kNN query UNFILTERED.
    #[test]
    fn test_match_text_emits_match_query() {
        let c = build_filter("body", "match", &json!({"text": "quick"})).unwrap();
        assert_eq!(c, json!({"match": {"body": "quick"}}));
    }

    #[test]
    fn os_text_only_condition_not_dropped() {
        // `{"and":[{"body":{"match":{"text":"quick"}}}]}` — the exact fixture
        // condition from `write_fulltext_project`. Must yield a non-empty
        // `bool.must` containing the `match` clause (was None → dropped → #120).
        let conditions = json!({"and": [{"body": {"match": {"text": "quick"}}}]});
        let parsed =
            parse_os_conditions(&conditions).expect("text-only filter must not be dropped");
        assert_eq!(
            parsed,
            json!({"bool": {"must": [{"match": {"body": "quick"}}]}})
        );
    }

    // Regression for qdrant/vector-db-benchmark#167: the filter must land inside
    // the kNN clause (efficient filtering), not in an outer bool wrapper
    // (post-filtering), otherwise filtered-search recall collapses.
    #[test]
    fn knn_filter_is_pushed_inside_knn_clause() {
        let filter = json!({"bool": {"must": [{"match": {"a": 1}}]}});
        let body = build_knn_body(&[0.1, 0.2, 0.3], 10, Some(&filter));

        // Filter lives at query.knn.vector.filter ...
        assert_eq!(body["query"]["knn"]["vector"]["filter"], filter);
        // ... and there is no post-filtering bool wrapper around the kNN query.
        assert!(
            body["query"].get("bool").is_none(),
            "kNN query must not be wrapped in an outer bool (post-filtering)"
        );
        assert_eq!(body["query"]["knn"]["vector"]["k"], 10);
        assert_eq!(body["size"], 10);
    }

    #[test]
    fn knn_body_without_filter_has_no_filter_key() {
        let body = build_knn_body(&[0.1, 0.2], 5, None);
        assert!(body["query"]["knn"]["vector"].get("filter").is_none());
    }

    #[test]
    fn knn_body_trims_the_response() {
        // Response trimming: no _source, ids via doc-value, no stored fields.
        let body = build_knn_body(&[0.1, 0.2], 5, None);
        assert_eq!(body["_source"], serde_json::json!(false));
        assert_eq!(body["docvalue_fields"], serde_json::json!(["_id"]));
        assert_eq!(body["stored_fields"], serde_json::json!("_none_"));
    }

    #[test]
    fn hit_id_reads_docvalue_then_falls_back() {
        // Trimmed response: id under fields._id[0].
        let trimmed = serde_json::json!({"fields": {"_id": ["deadbeef"]}, "_score": 1.0});
        assert_eq!(hit_id(&trimmed), Some("deadbeef"));
        // Untrimmed response: top-level _id.
        let plain = serde_json::json!({"_id": "cafef00d", "_score": 1.0});
        assert_eq!(hit_id(&plain), Some("cafef00d"));
        // Missing id.
        assert_eq!(hit_id(&serde_json::json!({"_score": 1.0})), None);
    }

    #[test]
    fn or_conditions_require_minimum_should_match() {
        let conditions = json!({"or": [{"a": {"match": {"value": 1}}}]});
        let parsed = parse_os_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];

        // OR filters must actually restrict results, not just contribute score.
        assert_eq!(bool_query["minimum_should_match"], 1);
        assert!(bool_query["should"].as_array().unwrap().len() == 1);
        // No empty `must` array should be emitted for an OR-only filter.
        assert!(bool_query.get("must").is_none());
    }

    #[test]
    fn and_only_conditions_omit_should() {
        let conditions = json!({"and": [{"a": {"match": {"value": 1}}}]});
        let parsed = parse_os_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];

        assert!(bool_query["must"].as_array().unwrap().len() == 1);
        assert!(bool_query.get("should").is_none());
        assert!(bool_query.get("minimum_should_match").is_none());
    }

    #[test]
    fn empty_conditions_return_none() {
        assert!(parse_os_conditions(&json!({})).is_none());
        // Present-but-empty sub-arrays should not produce a filter either.
        assert!(parse_os_conditions(&json!({"and": [], "or": []})).is_none());
    }

    // Nested/grouped boolean filter: `(color==red AND size>=50) OR (color==blue
    // AND size<10)`. Each OR arm must become its OWN nested `{bool:{must:[...]}}`
    // inside the outer `should`, NOT be flattened into one flat clause list.
    #[test]
    fn nested_group_nests_native_bool() {
        let conditions = json!({ "or": [
            { "and": [
                { "color": { "match": { "value": "red" } } },
                { "size": { "range": { "gte": 50 } } },
            ] },
            { "and": [
                { "color": { "match": { "value": "blue" } } },
                { "size": { "range": { "lt": 10 } } },
            ] },
        ] });
        let parsed = parse_os_conditions(&conditions).expect("nested filter must parse");
        assert_eq!(
            parsed,
            json!({ "bool": {
                "minimum_should_match": 1,
                "should": [
                    { "bool": { "must": [
                        { "match": { "color": "red" } },
                        { "range": { "size": { "gte": 50 } } },
                    ] } },
                    { "bool": { "must": [
                        { "match": { "color": "blue" } },
                        { "range": { "size": { "lt": 10 } } },
                    ] } },
                ],
            } })
        );
    }

    #[test]
    fn and_or_combined_keeps_both_and_min_should() {
        let conditions = json!({
            "and": [{"a": {"match": {"value": 1}}}],
            "or": [{"b": {"match": {"value": 2}}}],
        });
        let parsed = parse_os_conditions(&conditions).expect("should parse");
        let bool_query = &parsed["bool"];
        assert_eq!(bool_query["must"].as_array().unwrap().len(), 1);
        assert_eq!(bool_query["should"].as_array().unwrap().len(), 1);
        assert_eq!(bool_query["minimum_should_match"], 1);
    }

    // ── Range operators ────────────────────────────────────────────────────

    #[test]
    fn range_lt_lte_gt_gte_map_to_os_range() {
        assert_eq!(
            build_filter("n", "range", &json!({"lt":5})).unwrap(),
            json!({"range":{"n":{"lt":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &json!({"lte":5})).unwrap(),
            json!({"range":{"n":{"lte":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &json!({"gt":5})).unwrap(),
            json!({"range":{"n":{"gt":5}}})
        );
        assert_eq!(
            build_filter("n", "range", &json!({"gte":5})).unwrap(),
            json!({"range":{"n":{"gte":5}}})
        );
    }

    #[test]
    fn range_two_sided_keeps_both_bounds() {
        assert_eq!(
            build_filter("n", "range", &json!({"gte":10,"lt":20})).unwrap(),
            json!({"range":{"n":{"gte":10,"lt":20}}})
        );
    }

    #[test]
    fn range_unknown_op_yields_empty_range_object() {
        assert_eq!(
            build_filter("n", "range", &json!({"foo":5})).unwrap(),
            json!({"range":{"n":{}}})
        );
    }

    #[test]
    fn range_null_bound_is_skipped() {
        assert_eq!(
            build_filter("n", "range", &json!({"gte":serde_json::Value::Null})).unwrap(),
            json!({"range":{"n":{}}})
        );
    }

    // ── Geo filter ─────────────────────────────────────────────────────────

    #[test]
    fn geo_with_radius_emits_geo_distance() {
        assert_eq!(
            build_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0,"radius":500})).unwrap(),
            json!({"geo_distance":{"distance":"500m","loc":{"lat":20.0,"lon":10.0}}})
        );
    }

    #[test]
    fn geo_without_radius_uses_default_1000m() {
        assert_eq!(
            build_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0})).unwrap(),
            json!({"geo_distance":{"distance":"1000m","loc":{"lat":20.0,"lon":10.0}}})
        );
    }

    #[test]
    fn geo_missing_lat_or_lon_is_none() {
        assert!(build_filter("loc", "geo", &json!({"lon":10.0,"radius":5})).is_none());
        assert!(build_filter("loc", "geo", &json!({"lat":20.0,"radius":5})).is_none());
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn os_space_type_covers_all_arms() {
        assert_eq!(os_space_type("l2").unwrap(), "l2");
        assert_eq!(os_space_type("euclidean").unwrap(), "l2");
        assert_eq!(os_space_type("cosine").unwrap(), "cosinesimil");
        assert_eq!(os_space_type("angular").unwrap(), "cosinesimil");
        assert_eq!(os_space_type("COSINE").unwrap(), "cosinesimil");
        assert!(os_space_type("dot").is_err());
        assert!(os_space_type("ip").is_err());
        assert!(os_space_type("nope").is_err());
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    #[test]
    fn exact_match_int_float_bool_pass_through_match() {
        assert_eq!(
            build_filter("n", "match", &json!({"value":5})).unwrap(),
            json!({"match":{"n":5}})
        );
        assert_eq!(
            build_filter("n", "match", &json!({"value":1.5})).unwrap(),
            json!({"match":{"n":1.5}})
        );
        assert_eq!(
            build_filter("flag", "match", &json!({"value":true})).unwrap(),
            json!({"match":{"flag":true}})
        );
    }

    #[test]
    fn exact_match_array_value_is_none() {
        // #121: OS build_filter now guards the scalar exact-match arm; a
        // non-scalar value is dropped (None), matching qdrant/redis/valkey/
        // vectorsets (was previously forwarded verbatim as `{"match":{"n":[1,2]}}`).
        assert_eq!(build_filter("n", "match", &json!({"value":[1,2]})), None);
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
        // Trimmed hits carry the id under fields._id[0].
        let body = json!({
            "hits": {"hits": [
                {"fields": {"_id": [id_to_uuid_hex(7)]}, "_score": 0.9},
                {"fields": {"_id": [id_to_uuid_hex(3)]}, "_score": 0.5},
            ]}
        });
        assert_eq!(extract_knn_hits(&body).unwrap(), vec![(7, 0.9), (3, 0.5)]);
    }

    #[test]
    fn extract_knn_hits_missing_hits_hits_errors() {
        let body = json!({"hits": {"total": 0}});
        assert_eq!(
            extract_knn_hits(&body).unwrap_err(),
            "Missing hits.hits in search response"
        );
    }

    #[test]
    fn extract_knn_hits_missing_id_errors() {
        let body = json!({"hits": {"hits": [{"_score": 0.9}]}});
        assert_eq!(extract_knn_hits(&body).unwrap_err(), "Missing _id in hit");
    }

    #[test]
    fn extract_knn_hits_missing_score_defaults_zero() {
        let body = json!({"hits": {"hits": [{"_id": id_to_uuid_hex(4)}]}});
        assert_eq!(extract_knn_hits(&body).unwrap(), vec![(4, 0.0)]);
    }

    // ── resolve_index_space_type: distance mapping + rejections ────────────
    #[test]
    fn resolve_index_space_type_maps_and_rejects() {
        assert_eq!(
            resolve_index_space_type("cosine", 128).unwrap(),
            "cosinesimil"
        );
        assert_eq!(resolve_index_space_type("l2", 2048).unwrap(), "l2");
        assert_eq!(
            resolve_index_space_type("dot", 128).unwrap_err(),
            "OpenSearch does not support DOT product distance"
        );
        assert!(resolve_index_space_type("ip", 128).is_err());
        assert_eq!(
            resolve_index_space_type("cosine", 4096).unwrap_err(),
            "OpenSearch does not support vector_size > 2048 (got 4096)"
        );
        assert!(resolve_index_space_type("nope", 128).is_err());
    }

    #[test]
    fn item_level_rejections_are_retryable_but_real_errors_are_not() {
        // Load shedding shows up two ways: an explicit 429 status on the item...
        assert!(item_is_retryable(&json!({"index": {"status": 429}})));
        // ...or the exception type, which is what OpenSearch actually returns
        // when the write queue is full or the circuit breaker trips.
        for t in [
            "es_rejected_execution_exception",
            "circuit_breaking_exception",
            "cluster_block_exception",
        ] {
            assert!(
                item_is_retryable(&json!({"index": {"status": 503, "error": {"type": t}}})),
                "{t} should be retryable"
            );
        }
        // A bad document is NOT retryable — resending it forever cannot help.
        assert!(!item_is_retryable(
            &json!({"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}})
        ));
        assert!(!item_is_retryable(&json!({"index": {"status": 201}})));
        assert!(!item_is_retryable(&json!({})));
    }

    #[test]
    fn backoff_grows_then_caps() {
        // Guards the shift: attempt is clamped so 1u64 << attempt cannot overflow,
        // and the delay is capped so a long stall cannot sleep for hours.
        let d = |a: u32| 500u64.saturating_mul(1u64 << a.min(6)).min(30_000);
        assert_eq!(d(0), 500);
        assert_eq!(d(3), 4_000);
        assert_eq!(d(6), 30_000);
        assert_eq!(d(50), 30_000, "must not overflow or exceed the cap");
    }
}
