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
        if dist_lower == "dot" || dist_lower == "ip" {
            return Err("OpenSearch does not support DOT product distance".to_string());
        }
        if vector_size > 2048 {
            return Err(format!(
                "OpenSearch does not support vector_size > 2048 (got {})",
                vector_size
            ));
        }

        // Map distance metric (OpenSearch uses different names than ES)
        let space_type = match dist_lower.as_str() {
            "l2" | "euclidean" => "l2",
            "cosine" | "angular" => "cosinesimil",
            other => {
                return Err(format!(
                    "Unsupported distance metric for OpenSearch: {}",
                    other
                ))
            }
        };

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

        let body = serde_json::json!({
            "settings": {
                "index": {
                    "knn": true,
                    // Indexing-throughput tuning: no replicas and no periodic
                    // refresh, since the benchmark bulk-loads all data up front and
                    // force-merges before searching.
                    "number_of_replicas": 0,
                    "refresh_interval": -1,
                }
            },
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

    /// Load the kNN index into memory before searching so the first queries
    /// aren't penalised by cold-cache graph loading. Best-effort: a non-success
    /// response is logged, not fatal.
    fn warmup(&self) -> Result<(), String> {
        use opensearch::http::headers::HeaderMap;
        use opensearch::http::Method;

        let path = format!("/_plugins/_knn/warmup/{}", self.index_name);
        let resp = self
            .rt
            .block_on(self.client.transport().send(
                Method::Get,
                &path,
                HeaderMap::new(),
                Option::<&()>::None,
                Option::<Vec<u8>>::None,
                None,
            ))
            .map_err(|e| format!("kNN warmup request failed: {}", e))?;

        if !resp.status_code().is_success() {
            let text = self.rt.block_on(resp.text()).unwrap_or_default();
            eprintln!("Warning: kNN warmup returned non-success: {}", text);
        }
        Ok(())
    }

    /// Apply search-time settings (e.g., knn.algo_param.ef_search)
    fn setup_search(&self, params: &SearchParams) -> Result<(), String> {
        // Warm the graph into memory before timing any queries.
        self.warmup()?;

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

/// Parse conditions into OpenSearch bool query (same DSL as Elasticsearch).
fn parse_os_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
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

fn build_subfilters(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
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
            let value = criteria.get("value")?;
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

fn knn_search(
    rt: &tokio::runtime::Runtime,
    client: &OpenSearch,
    index_name: &str,
    query_vector: &[f32],
    top: usize,
    filter: Option<&serde_json::Value>,
) -> Result<Vec<(i64, f64)>, String> {
    let body = build_knn_body(query_vector, top, filter);

    let resp = rt
        .block_on(
            client
                .search(SearchParts::Index(&[index_name]))
                .body(body)
                .send(),
        )
        .map_err(|e| format!("KNN search failed: {}", e))?;

    if !resp.status_code().is_success() {
        let text = rt.block_on(resp.text()).unwrap_or_default();
        return Err(format!("KNN search error: {}", text));
    }

    let resp_body: serde_json::Value = rt
        .block_on(resp.json())
        .map_err(|e| format!("Failed to parse search response: {}", e))?;

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

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

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
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let query_idx = Arc::clone(&query_idx);
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
                        Err(_) => return (t, p, r, mr, nd),
                    };
                    let client = match create_os_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(_) => return (t, p, r, mr, nd),
                    };

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let query_start = Instant::now();
                        let results = knn_search(
                            &rt,
                            &client,
                            &index_name,
                            &queries[idx],
                            top,
                            parsed_filters[idx].as_ref(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        match results {
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
        let total_time = start_time.elapsed().as_secs_f64();

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
