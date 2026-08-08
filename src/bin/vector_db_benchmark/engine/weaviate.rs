//! Weaviate engine implementation.
//!
//! Schema/upload/ef-tuning use Weaviate's REST API (v1) via reqwest::blocking.
//! Vector search uses Weaviate's **gRPC** API (port 50051) by default — the
//! high-throughput query path used by the official clients — for BOTH filtered
//! and unfiltered queries: metadata filters are translated into the gRPC
//! `Filters` message from the exact same where-tree the GraphQL path serializes
//! (see `where_json_to_grpc_filters`), so the two transports evaluate identical
//! filter semantics. GraphQL over HTTP remains the fallback for the whole run
//! when `WEAVIATE_USE_GRAPHQL` is set or a filter uses a condition the gRPC
//! proto can't express. (GraphQL with a stringified query vector is markedly
//! slower and caps throughput, which is why gRPC is the default.)

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use uuid::Uuid;

use super::weaviate_grpc::weaviate_v1::{
    filter_target, filters, weaviate_client::WeaviateClient, FilterTarget, Filters,
    GeoCoordinatesFilter, MetadataRequest, NearVector, SearchReply, SearchRequest,
};
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_CLASS_NAME: &str = "Benchmark";

pub struct WeaviateEngine {
    name: String,
    class_name: String,
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    base_url: String,
    /// gRPC endpoint (http://host:50051) for the search RPC.
    grpc_endpoint: String,
    api_key: Option<String>,
    search_params: Vec<SearchParams>,
    /// vectorIndexConfig from collection_params
    vector_index_config: serde_json::Value,
}

impl WeaviateEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("WEAVIATE_HTTP_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8080);

        let class_name =
            std::env::var("WEAVIATE_CLASS_NAME").unwrap_or_else(|_| DEFAULT_CLASS_NAME.to_string());

        let api_key = std::env::var("WEAVIATE_API_KEY").ok();

        let timeout = engine_config
            .connection_params
            .as_ref()
            .and_then(|p| p.get("timeout_config"))
            .and_then(|v| v.as_u64())
            .unwrap_or(90);

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(8) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1024) as usize;

        let base_url = if host.starts_with("http") {
            format!("{}:{}", host, port)
        } else {
            format!("http://{}:{}", host, port)
        };

        // gRPC endpoint: same host, port 50051 (override via WEAVIATE_GRPC_PORT),
        // always plaintext h2c (self-hosted). Strip any scheme/port from `host`.
        let grpc_port: u16 = std::env::var("WEAVIATE_GRPC_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50051);
        let host_only = host
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .split(':')
            .next()
            .unwrap_or(host);
        let grpc_endpoint = format!("http://{}:{}", host_only, grpc_port);

        // Extract vectorIndexConfig from collection_params.extra
        let vector_index_config = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.extra.as_ref())
            .and_then(|e| e.get("vectorIndexConfig"))
            .cloned()
            .unwrap_or(serde_json::json!({}));

        Ok(Self {
            name: engine_config.name.clone(),
            class_name,
            timeout,
            batch_size,
            parallel,
            base_url,
            grpc_endpoint,
            api_key,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            vector_index_config,
        })
    }

    fn create_client(&self) -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))
    }

    fn add_auth(
        &self,
        req: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        if let Some(key) = &self.api_key {
            req.header("Authorization", format!("Bearer {}", key))
        } else {
            req
        }
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

    fn delete_class(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let url = format!("{}/v1/schema/{}", self.base_url, self.class_name);
        let req = client.delete(&url);
        let resp = self.add_auth(req).send().map_err(|e| e.to_string())?;
        if resp.status().is_success() || resp.status().as_u16() == 404 {
            Ok(())
        } else {
            Err(format!(
                "Failed to delete class: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ))
        }
    }

    fn create_class(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let distance = dataset.distance();

        let weaviate_distance = map_weaviate_distance(distance)?;

        // Build properties from schema
        let mut properties = Vec::new();
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    if let Some(prop) = weaviate_property(field_name, ft) {
                        properties.push(prop);
                    }
                }
            }
        }

        // Merge vectorIndexConfig with distance and cache settings
        let mut vic = serde_json::json!({
            "vectorCacheMaxObjects": 1_000_000_000i64,
            "distance": weaviate_distance,
        });
        if let Some(vic_obj) = self.vector_index_config.as_object() {
            let merged = vic.as_object_mut().unwrap();
            for (k, v) in vic_obj {
                merged.insert(k.clone(), v.clone());
            }
        }

        let body = serde_json::json!({
            "class": self.class_name,
            "vectorizer": "none",
            "properties": properties,
            "vectorIndexConfig": vic,
        });

        let url = format!("{}/v1/schema", self.base_url);
        let req = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);
        let resp = self
            .add_auth(req)
            .send()
            .map_err(|e| format!("Failed to create class: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to create class: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
        }

        Ok(())
    }

    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
        schema_types: &HashMap<String, String>,
    ) -> Result<(), String> {
        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let base_url = self.base_url.clone();
                let class_name = self.class_name.clone();
                let api_key = self.api_key.clone();
                let timeout = self.timeout;
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let schema_types = schema_types.clone();
                let pb = &pb;

                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
                        .build()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
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
                        if let Err(e) = upload_batch_objects(
                            &client,
                            &base_url,
                            &class_name,
                            api_key.as_deref(),
                            &ids[batch_start..batch_end],
                            &vectors[batch_start..batch_end],
                            &metadata[batch_start..batch_end],
                            &schema_types,
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

    /// Update the class's vectorIndexConfig `ef` for search-time tuning.
    ///
    /// In Weaviate, query-time `ef` is a class-level schema setting, not a
    /// per-query parameter. Updating it requires `PUT /v1/schema/{class}` with the
    /// *complete* class object: a partial body (just `vectorIndexConfig`) is
    /// rejected with 422 ("class name is immutable: attempted change from
    /// \"<class>\" to \"\""), which would silently leave every query running at the
    /// default dynamic ef (-1) and flatten the recall/QPS sweep. So we fetch the
    /// current class, merge the new `ef` into its `vectorIndexConfig`, and PUT the
    /// whole object back — immutable fields (efConstruction, maxConnections) are
    /// sent back unchanged, so they don't trip the immutability check. A failure
    /// here returns Err (rather than a swallowed warning) so a broken ef sweep
    /// surfaces immediately instead of producing misleading results.
    fn setup_search(
        &self,
        client: &reqwest::blocking::Client,
        params: &SearchParams,
    ) -> Result<(), String> {
        // `vectorIndexConfig.ef`, accepted nested under `search_params`/`config`
        // (upstream's spelling) or flat (ours) — see SearchParams::knob. Reading
        // only the flat map meant an upstream-style `config: {...}` returned Ok
        // WITHOUT patching the class, so every ef in the sweep silently ran at
        // the build-time ef while the results file listed one row per ef.
        // Also accept the plain `search_params.ef` spelling that the other eight
        // ef-bearing engines use. Without it, `{"search_params": {"ef": 128}}`
        // hit the early return below and the whole sweep silently ran at the
        // BUILD-time ef while emitting one results row per ef.
        let ef = params
            .knob("vectorIndexConfig")
            .and_then(|v| v.get("ef"))
            .and_then(|v| v.as_i64())
            .or_else(|| params.search_params.as_ref().and_then(|sp| sp.ef));

        let Some(ef_val) = ef else {
            return Ok(());
        };

        let url = format!("{}/v1/schema/{}", self.base_url, self.class_name);

        // 1. Fetch the current class definition.
        let get_req = client.get(&url);
        let get_resp = self
            .add_auth(get_req)
            .send()
            .map_err(|e| format!("Failed to GET class for ef update: {}", e))?;
        if !get_resp.status().is_success() {
            return Err(format!(
                "Failed to GET class {} for ef update: {} {}",
                self.class_name,
                get_resp.status(),
                get_resp.text().unwrap_or_default()
            ));
        }
        let mut class_obj: serde_json::Value = get_resp
            .json()
            .map_err(|e| format!("Failed to parse class JSON for ef update: {}", e))?;

        // 2. Merge the new ef into vectorIndexConfig (creating it if absent).
        let obj = class_obj
            .as_object_mut()
            .ok_or_else(|| "class definition is not a JSON object".to_string())?;
        let vic = obj
            .entry("vectorIndexConfig")
            .or_insert_with(|| serde_json::json!({}));
        let vic_obj = vic
            .as_object_mut()
            .ok_or_else(|| "vectorIndexConfig is not a JSON object".to_string())?;
        vic_obj.insert("ef".to_string(), serde_json::json!(ef_val));

        // 3. PUT the full class object back.
        let put_req = client
            .put(&url)
            .header("Content-Type", "application/json")
            .json(&class_obj);
        let put_resp = self
            .add_auth(put_req)
            .send()
            .map_err(|e| format!("Failed to PUT class for ef update: {}", e))?;
        if !put_resp.status().is_success() {
            return Err(format!(
                "Failed to update vectorIndexConfig ef={}: {} {}",
                ef_val,
                put_resp.status(),
                put_resp.text().unwrap_or_default()
            ));
        }

        Ok(())
    }
}

/// Build a Weaviate schema property from a dataset schema `field_type`.
///
/// Returns `None` for unsupported field types (skipped). Filtering requires an
/// inverted index: modern Weaviate (>= 1.19, incl. 1.38) replaced the
/// deprecated `indexInverted` flag with `indexFilterable` (roaring-bitmap
/// filter index) and `indexSearchable` (BM25). `Equal` filters read the
/// FILTERABLE index, so it must be enabled explicitly. `indexSearchable` is
/// only valid on text-backed properties, so it is set only for those.
fn weaviate_property(field_name: &str, field_type: &str) -> Option<serde_json::Value> {
    use vector_db_benchmark::readers::metadata::is_multivalued_keyword_field;
    // A multi-valued keyword field (`labels`) is a `text[]` array property so
    // `ContainsAny` matches individual elements; a scalar `text` could only
    // compare the whole value (issue #88). `field` tokenization (below) still
    // applies, giving whole-value equality per array element.
    let multivalued =
        matches!(field_type, "keyword" | "text") && is_multivalued_keyword_field(field_name);
    let wv_type = match field_type {
        "int" => "int",
        "keyword" | "text" if multivalued => "text[]",
        "keyword" | "text" => "text",
        "float" => "number",
        "geo" => "geoCoordinates",
        // Bools become a `boolean` property (upload converts the reader's
        // "true"/"false" string to a native bool). Datetimes become a `date`
        // property; Weaviate stores/filters RFC3339 strings, and its gRPC filter
        // compares date properties via valueText (there is no ValueDate variant).
        "bool" => "boolean",
        "datetime" => "date",
        // A `uuid` is an exact-match opaque string → a `text` property with
        // `field` tokenization (whole-value equality, like keyword). Without this
        // arm it hit `_ => return None`, so no property was created and every
        // uuid filter silently broke.
        "uuid" => "text",
        _ => return None,
    };
    let mut prop = serde_json::json!({
        "name": field_name,
        "dataType": [wv_type],
        "indexFilterable": true,
    });
    let obj = prop.as_object_mut().unwrap();
    // Keyword fields must match on the WHOLE value (exact keyword equality,
    // like qdrant). Weaviate's default `word` tokenization turns `Equal` into
    // token-containment, so `Equal "Blue"` would also match "Dark Blue". Force
    // `field` tokenization for keyword; keep `word` for full-text. Tokenization
    // and `indexSearchable` apply only to text-backed properties.
    if let Some(tok) = match field_type {
        // `uuid` needs whole-value equality just like keyword, so force `field`
        // tokenization (default `word` would turn `Equal` into token-containment).
        "keyword" | "uuid" => Some("field"),
        "text" => Some("word"),
        _ => None,
    } {
        obj.insert("tokenization".to_string(), serde_json::json!(tok));
        obj.insert("indexSearchable".to_string(), serde_json::json!(true));
    }
    Some(prop)
}

/// Convert a dataset metadata value into a JSON value typed for Weaviate's
/// strict schema. Numbers arrive from the reader as strings (see
/// `parse_metadata_from_json`); Weaviate rejects a whole object if e.g. an
/// `int` property receives a string, so numeric fields must be coerced back to
/// JSON numbers using the dataset `schema_type`. Non-numeric fields pass
/// through unchanged.
fn coerce_metadata_value(
    schema_type: Option<&str>,
    value: &vector_db_benchmark::readers::metadata::MetadataValue,
) -> serde_json::Value {
    use vector_db_benchmark::readers::metadata::MetadataValue;
    // A numeric value under a keyword/text-declared property must stay a string,
    // or Weaviate rejects the whole object (the property is declared `text`).
    let value = value.coerce_for_schema(schema_type);
    match value.as_ref() {
        MetadataValue::String(s) => match schema_type {
            Some("int") => s
                .parse::<i64>()
                .map(serde_json::Value::from)
                .unwrap_or_else(|_| serde_json::Value::String(s.clone())),
            Some("float") => s
                .parse::<f64>()
                .map(serde_json::Value::from)
                .unwrap_or_else(|_| serde_json::Value::String(s.clone())),
            // A `boolean` property needs a native JSON bool, not the "true"/
            // "false" string the reader produces (Weaviate is strict-typed).
            Some("bool") => match s.as_str() {
                "true" => serde_json::Value::Bool(true),
                "false" => serde_json::Value::Bool(false),
                _ => serde_json::Value::String(s.clone()),
            },
            // `datetime` -> `date` property: keep the RFC3339 string as-is
            // (Weaviate parses ISO-8601 date strings).
            _ => serde_json::Value::String(s.clone()),
        },
        MetadataValue::Int(n) => serde_json::Value::from(*n),
        MetadataValue::Float(f) => serde_json::json!(*f),
        MetadataValue::Labels(labels) => serde_json::Value::Array(
            labels
                .iter()
                .map(|l| serde_json::Value::String(l.clone()))
                .collect(),
        ),
        MetadataValue::Geo { lon, lat } => {
            serde_json::json!({"latitude": lat, "longitude": lon})
        }
    }
}

/// Convert integer ID to UUID (matches Python uuid.UUID(int=id))
fn id_to_uuid(id: i64) -> String {
    Uuid::from_u128(id as u128).to_string()
}

/// Convert UUID string back to integer
fn uuid_to_int(uuid_str: &str) -> Result<i64, String> {
    let uuid =
        Uuid::parse_str(uuid_str).map_err(|e| format!("Invalid UUID '{}': {}", uuid_str, e))?;
    Ok(uuid.as_u128() as i64)
}

/// Upload a batch of objects via Weaviate's batch API.
#[allow(clippy::too_many_arguments)]
fn upload_batch_objects(
    client: &reqwest::blocking::Client,
    base_url: &str,
    class_name: &str,
    api_key: Option<&str>,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    schema_types: &HashMap<String, String>,
) -> Result<(), String> {
    let mut objects = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let uuid = id_to_uuid(ids[i]);

        let mut properties = serde_json::Map::new();
        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                let val = coerce_metadata_value(schema_types.get(k).map(|s| s.as_str()), v);
                properties.insert(k.clone(), val);
            }
        }

        objects.push(serde_json::json!({
            "class": class_name,
            "id": uuid,
            "properties": properties,
            "vector": vectors[i],
        }));
    }

    let body = serde_json::json!({
        "objects": objects,
    });

    let url = format!("{}/v1/batch/objects", base_url);
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body);

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let resp = req
        .send()
        .map_err(|e| format!("Batch upload failed: {}", e))?;

    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(format!("Batch upload error: {} {}", status, text));
    }

    // The batch API returns HTTP 200 even when individual objects fail schema
    // validation (e.g. a value whose type does not match the property). Those
    // objects are silently NOT stored, which would leave the collection empty
    // and make every filtered search return zero hits. Inspect each object's
    // per-item `result.status` and surface the first failure.
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
        if let Some(arr) = parsed.as_array() {
            for obj in arr {
                let status = obj
                    .get("result")
                    .and_then(|r| r.get("status"))
                    .and_then(|s| s.as_str());
                if let Some(st) = status {
                    if st != "SUCCESS" {
                        let msg = obj
                            .pointer("/result/errors/error/0/message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown error");
                        return Err(format!(
                            "Batch object import failed (status {}): {}",
                            st, msg
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Search Weaviate using GraphQL near_vector query.
/// Build and serialize one GraphQL `nearVector` request body to JSON bytes.
/// Done OUTSIDE the per-query timed window: this is the heaviest client-side
/// step for Weaviate — a per-dimension `f32::to_string` + `format!` to assemble
/// the GraphQL query string, then JSON-wrapping it. That is client CPU work, not
/// server latency, so pre-building means the timed send only copies the finished
/// bytes onto the socket (matching pgvector/qdrant). Bytes are identical to what
/// `.json(&{"query": graphql})` would have sent inline.
fn build_graphql_body(
    class_name: &str,
    query_vector: &[f32],
    top: usize,
    filter: Option<&serde_json::Value>,
) -> Vec<u8> {
    let vector_str: String = query_vector
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let where_clause = if let Some(f) = filter {
        format!(", where: {}", json_to_graphql_literal(f))
    } else {
        String::new()
    };

    let graphql = format!(
        r#"{{
            Get {{
                {class_name}(
                    nearVector: {{ vector: [{vector_str}] }},
                    limit: {top}
                    {where_clause}
                ) {{
                    _additional {{
                        id
                        distance
                    }}
                }}
            }}
        }}"#,
        class_name = class_name,
        vector_str = vector_str,
        top = top,
        where_clause = where_clause,
    );

    let body = serde_json::json!({ "query": graphql });
    serde_json::to_vec(&body).expect("serialize GraphQL search body")
}

/// Send a pre-serialized GraphQL request and return the DECODED response. The
/// consistent timed boundary (see qdrant/pgvector/redis) is: request body
/// pre-serialized OUTSIDE the window; RPC send + receive + decode-to-structured-
/// response INSIDE the window (this fn: post + HTTP-status check + wire read +
/// `from_str`); GraphQL-error check + id/score extraction OUTSIDE
/// (`extract_graphql_hits`). So the JSON decode is billed as latency exactly
/// like qdrant's protobuf decode.
fn send_graphql(
    client: &reqwest::blocking::Client,
    base_url: &str,
    api_key: Option<&str>,
    body: &[u8],
) -> Result<serde_json::Value, String> {
    let url = format!("{}/v1/graphql", base_url);
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec());

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let resp = req
        .send()
        .map_err(|e| format!("GraphQL search failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "GraphQL search error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let text = resp
        .text()
        .map_err(|e| format!("Failed to read GraphQL response: {}", e))?;
    serde_json::from_str(&text).map_err(|e| format!("Failed to parse GraphQL response: {}", e))
}

/// Extract the id/score list from an already-decoded GraphQL response (done
/// AFTER the timed window — the GraphQL-error check and id extraction pull the
/// final ids out of the decoded struct for recall, mirroring pgvector/qdrant).
fn extract_graphql_hits(
    resp_body: &serde_json::Value,
    class_name: &str,
) -> Result<Vec<(i64, f64)>, String> {
    // Check for errors
    if let Some(errors) = resp_body.get("errors").and_then(|e| e.as_array()) {
        if !errors.is_empty() {
            return Err(format!("GraphQL errors: {:?}", errors));
        }
    }

    let results = resp_body
        .get("data")
        .and_then(|d| d.get("Get"))
        .and_then(|g| g.get(class_name))
        .and_then(|c| c.as_array())
        .ok_or_else(|| "Missing results in GraphQL response".to_string())?;

    let mut hits = Vec::with_capacity(results.len());
    for result in results {
        let additional = result.get("_additional").ok_or("Missing _additional")?;
        let id_str = additional
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing id")?;
        let distance = additional
            .get("distance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let id = uuid_to_int(id_str)?;
        hits.push((id, distance));
    }

    Ok(hits)
}

/// Build the gRPC `SearchRequest` for one query — packs the query vector as
/// little-endian f32 bytes (`vector_bytes`) and requests uuid + distance
/// metadata. This is the request-build step and is hoisted OUTSIDE the timed
/// window (precomputed before the parallel region, mirroring `graphql_bodies`
/// and qdrant, which builds its request before `query_start`). The class-level
/// `ef` set via the REST schema update still governs recall.
///
/// `NearVector::vector_bytes` is marked deprecated in the newer weaviate protos
/// (superseded by a multi-vector `vectors` field) but remains the accepted
/// packed-vector input on the 1.29–1.38 servers this targets, so we allow it.
///
/// `filter` is the pre-translated gRPC `Filters` message for this query (built
/// once, outside the timed window, from the same where-tree the GraphQL path
/// serializes), or `None` for an unfiltered query.
#[allow(deprecated)]
fn build_grpc_request(
    class_name: &str,
    query_vector: &[f32],
    top: usize,
    filter: Option<&Filters>,
) -> SearchRequest {
    let vector_bytes: Vec<u8> = query_vector.iter().flat_map(|f| f.to_le_bytes()).collect();

    SearchRequest {
        collection: class_name.to_string(),
        limit: top as u32,
        near_vector: Some(NearVector {
            vector_bytes,
            ..Default::default()
        }),
        metadata: Some(MetadataRequest {
            uuid: true,
            distance: true,
            ..Default::default()
        }),
        filters: filter.cloned(),
        ..Default::default()
    }
}

/// Translate the Weaviate GraphQL where-filter tree (produced by
/// `parse_weaviate_conditions` / `build_weaviate_filter`) into the equivalent
/// gRPC `Filters` message.
///
/// This consumes the SAME tree the GraphQL path serializes, so both transports
/// evaluate byte-for-byte identical filter semantics and score identical recall:
///   * `Equal`/`NotEqual`               → `Operator::Equal`/`NotEqual`
///   * `LessThan[Equal]`/`GreaterThan[Equal]` → the matching range operators
///   * `And`/`Or` (incl. the collapsed `range` two-sided AND and the
///     `match_any` Or-of-`Equal` set) → `Operator::And`/`Or` over nested filters
///   * `WithinGeoRange`                 → `Operator::WithinGeoRange` + `ValueGeo`
///
/// The typed value is placed in the corresponding `TestValue` field
/// (`valueText`→`ValueText`, `valueInt`→`ValueInt`, `valueNumber`→`ValueNumber`,
/// `valueBoolean`→`ValueBoolean`). `match_any` is intentionally NOT lowered to
/// the proto's `ContainsAny`: the shared where-tree already expands it to an
/// Or-of-`Equal`, and reusing that guarantees the gRPC result set matches
/// GraphQL exactly.
///
/// Returns `None` if any node uses an operator or value shape the proto can't
/// express (e.g. a non-scalar `Equal` value), so the caller can keep the run on
/// the GraphQL transport rather than silently dropping the filter.
fn where_json_to_grpc_filters(node: &serde_json::Value) -> Option<Filters> {
    let obj = node.as_object()?;
    let operator = obj.get("operator")?.as_str()?;

    // Logical node: And / Or over `operands`.
    if let Some(operands) = obj.get("operands").and_then(|v| v.as_array()) {
        let op = match operator {
            "And" => filters::Operator::And,
            "Or" => filters::Operator::Or,
            _ => return None,
        };
        let mut children = Vec::with_capacity(operands.len());
        for operand in operands {
            children.push(where_json_to_grpc_filters(operand)?);
        }
        return Some(Filters {
            operator: op as i32,
            filters: children,
            ..Default::default()
        });
    }

    // Leaf node: comparison on a single property path.
    let path = obj.get("path")?.as_array()?;
    let field = path.first()?.as_str()?;
    let target = Some(FilterTarget {
        target: Some(filter_target::Target::Property(field.to_string())),
    });

    // Geo range leaf.
    if operator == "WithinGeoRange" {
        let gr = obj.get("valueGeoRange")?;
        let coords = gr.get("geoCoordinates")?;
        let lat = coords.get("latitude")?.as_f64()? as f32;
        let lon = coords.get("longitude")?.as_f64()? as f32;
        let dist = gr.get("distance")?.get("max")?.as_f64()? as f32;
        return Some(Filters {
            operator: filters::Operator::WithinGeoRange as i32,
            target,
            test_value: Some(filters::TestValue::ValueGeo(GeoCoordinatesFilter {
                latitude: lat,
                longitude: lon,
                distance: dist,
            })),
            ..Default::default()
        });
    }

    let op = match operator {
        "Equal" => filters::Operator::Equal,
        "NotEqual" => filters::Operator::NotEqual,
        "LessThan" => filters::Operator::LessThan,
        "LessThanEqual" => filters::Operator::LessThanEqual,
        "GreaterThan" => filters::Operator::GreaterThan,
        "GreaterThanEqual" => filters::Operator::GreaterThanEqual,
        _ => return None,
    };

    // Typed value field — exactly one is present in a leaf built by
    // `build_weaviate_filter`. A non-scalar (e.g. array under `valueText`) fails
    // the scalar accessor and yields None → GraphQL fallback for the run.
    let test_value = if let Some(v) = obj.get("valueText") {
        filters::TestValue::ValueText(v.as_str()?.to_string())
    } else if let Some(v) = obj.get("valueInt") {
        filters::TestValue::ValueInt(v.as_i64()?)
    } else if let Some(v) = obj.get("valueNumber") {
        filters::TestValue::ValueNumber(v.as_f64()?)
    } else {
        let v = obj.get("valueBoolean")?;
        filters::TestValue::ValueBoolean(v.as_bool()?)
    };

    Some(Filters {
        operator: op as i32,
        target,
        test_value: Some(test_value),
        ..Default::default()
    })
}

/// Issue a prebuilt gRPC search and return the DECODED reply. The consistent
/// timed boundary (see qdrant/pgvector/redis) is: request build OUTSIDE the
/// window (`build_grpc_request`); RPC send + receive + protobuf decode INSIDE
/// the window (this fn: the awaited `client.search` decodes the reply);
/// id/distance extraction OUTSIDE (`extract_grpc_hits`). So the protobuf decode
/// is billed as latency, matching qdrant.
async fn grpc_search(
    client: &mut WeaviateClient<tonic::transport::Channel>,
    search: SearchRequest,
    api_key: Option<&str>,
) -> Result<SearchReply, String> {
    let mut request = tonic::Request::new(search);
    if let Some(key) = api_key {
        let val = format!("Bearer {}", key)
            .parse()
            .map_err(|e| format!("bad api key header: {}", e))?;
        request.metadata_mut().insert("authorization", val);
    }

    Ok(client
        .search(request)
        .await
        .map_err(|e| format!("gRPC search failed: {}", e))?
        .into_inner())
}

/// Extract (int-id, distance) pairs from an already-decoded gRPC reply (done
/// AFTER the timed window), mapping each object UUID back through `uuid_to_int`
/// (inverse of the upload id_to_uuid), mirroring pgvector/qdrant.
fn extract_grpc_hits(reply: &SearchReply) -> Result<Vec<(i64, f64)>, String> {
    let mut hits = Vec::with_capacity(reply.results.len());
    for r in &reply.results {
        if let Some(m) = &r.metadata {
            let id = uuid_to_int(&m.id)?;
            hits.push((id, m.distance as f64));
        }
    }
    Ok(hits)
}

/// Parse conditions into Weaviate where filter format.
/// Map a dataset distance name to the Weaviate `vectorIndexConfig.distance`
/// value. Unknown metrics error. A wrong arm here would silently change ranking,
/// so every arm is unit-tested.
fn map_weaviate_distance(distance: &str) -> Result<&'static str, String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok("l2-squared"),
        "cosine" | "angular" => Ok("cosine"),
        "dot" | "ip" => Ok("dot"),
        other => Err(format!(
            "Unsupported distance metric for Weaviate: {}",
            other
        )),
    }
}

pub(crate) fn parse_weaviate_conditions(
    conditions: &serde_json::Value,
) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut operands = Vec::new();

    if let Some(and_items) = obj.get("and").and_then(|v| v.as_array()) {
        let and_filters: Vec<serde_json::Value> = and_items
            .iter()
            .filter_map(build_weaviate_entry_filter)
            .collect();
        if !and_filters.is_empty() {
            if and_filters.len() == 1 {
                operands.push(and_filters.into_iter().next().unwrap());
            } else {
                operands.push(serde_json::json!({
                    "operator": "And",
                    "operands": and_filters,
                }));
            }
        }
    }

    if let Some(or_items) = obj.get("or").and_then(|v| v.as_array()) {
        let or_filters: Vec<serde_json::Value> = or_items
            .iter()
            .filter_map(build_weaviate_entry_filter)
            .collect();
        if !or_filters.is_empty() {
            if or_filters.len() == 1 {
                operands.push(or_filters.into_iter().next().unwrap());
            } else {
                operands.push(serde_json::json!({
                    "operator": "Or",
                    "operands": or_filters,
                }));
            }
        }
    }

    if operands.is_empty() {
        return None;
    }
    if operands.len() == 1 {
        return Some(operands.into_iter().next().unwrap());
    }

    Some(serde_json::json!({
        "operator": "And",
        "operands": operands,
    }))
}

fn build_weaviate_entry_filter(entry: &serde_json::Value) -> Option<serde_json::Value> {
    let entry_obj = entry.as_object()?;

    // Nested group: an and/or array element that is itself an {and:[...]} /
    // {or:[...]} subtree is NOT a field leaf. Recurse through
    // `parse_weaviate_conditions`, which wraps it in its own native
    // `Filters.And` / `Filters.Or` operand — so the parent combines this group
    // as one nested sub-clause instead of mis-flattening its children into the
    // parent's operand list. Both transports nest natively: the gRPC translator
    // (`where_json_to_grpc_filters`) recurses over operands and the GraphQL
    // serializer (`json_to_graphql_literal`) nests operator objects.
    if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
        return parse_weaviate_conditions(entry);
    }

    let mut filters = Vec::new();

    for (field_name, field_filters) in entry_obj {
        if let Some(filter_obj) = field_filters.as_object() {
            for (cond_type, criteria) in filter_obj {
                if let Some(f) = build_weaviate_filter(field_name, cond_type, criteria) {
                    filters.push(f);
                }
            }
        }
    }

    if filters.is_empty() {
        None
    } else if filters.len() == 1 {
        Some(filters.into_iter().next().unwrap())
    } else {
        Some(serde_json::json!({
            "operator": "And",
            "operands": filters,
        }))
    }
}

/// Serialize a where-filter `Value` as a **GraphQL object literal** (not JSON).
///
/// Weaviate's GraphQL `where` argument requires object keys as bare names and
/// the `operator` value as an unquoted enum (`operator: Equal`), whereas
/// `serde_json::to_string` emits quoted keys/values (`"operator":"Equal"`),
/// which the GraphQL parser rejects with a syntax error. Object keys are
/// emitted unquoted, the `operator` field's value is emitted as an unquoted
/// enum, and every other scalar keeps normal JSON quoting/formatting.
fn json_to_graphql_literal(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let fields: Vec<String> = map
                .iter()
                .map(|(k, val)| {
                    if k == "operator" {
                        // GraphQL enum value: unquoted.
                        format!("{}: {}", k, val.as_str().unwrap_or_default())
                    } else {
                        format!("{}: {}", k, json_to_graphql_literal(val))
                    }
                })
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(json_to_graphql_literal).collect();
            format!("[{}]", items.join(", "))
        }
        serde_json::Value::String(s) => {
            format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

fn build_weaviate_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<serde_json::Value> {
    match condition_type {
        "match" => {
            // Value typing shared by exact match and match_any.
            let value_key = |v: &serde_json::Value| -> &'static str {
                if v.is_string() {
                    "valueText"
                } else if v.is_i64() {
                    "valueInt"
                } else if v.is_number() {
                    "valueNumber"
                } else if v.is_boolean() {
                    "valueBoolean"
                } else {
                    "valueText"
                }
            };

            // match_any: field value in a list -> OR of `Equal` conditions,
            // reusing the engine's proven exact-match path (same OR-of-values
            // semantics as qdrant's Condition::matches(field, Vec)). An empty
            // IN-set matches NOTHING: emit an unsatisfiable `Equal(x) AND
            // NotEqual(x)` rather than dropping the clause, since dropping the
            // sole clause would return every object (the inverse of the filter).
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                let operands: Vec<serde_json::Value> = any
                    .iter()
                    .map(|v| {
                        let vk = value_key(v);
                        serde_json::json!({
                            "path": [field_name],
                            "operator": "Equal",
                            vk: v,
                        })
                    })
                    .collect();
                return Some(match operands.len() {
                    0 => {
                        const NEVER: &str = "__match_any_never_match__";
                        serde_json::json!({
                            "operator": "And",
                            "operands": [
                                {"path": [field_name], "operator": "Equal", "valueText": NEVER},
                                {"path": [field_name], "operator": "NotEqual", "valueText": NEVER},
                            ]
                        })
                    }
                    1 => operands.into_iter().next().unwrap(),
                    _ => serde_json::json!({"operator": "Or", "operands": operands}),
                });
            }

            // Full-text match: `{match:{text}}` over a word-tokenized `text`
            // property. Weaviate `Equal` on a token matches objects whose property
            // CONTAINS that token (verified: Equal "quick" matches "the quick brown
            // fox"), so route it to the same Equal path. Dropping the clause would
            // run the kNN query UNFILTERED while recall is scored against filtered
            // ground truth.
            if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                return Some(serde_json::json!({
                    "path": [field_name],
                    "operator": "Equal",
                    "valueText": text,
                }));
            }

            let value = criteria.get("value")?;
            // Guard non-scalar: an array/object/null under `value` is malformed
            // input — the canonical model uses `match.any` for lists. Without
            // this, `value_key` falls back to "valueText" and forwards the array
            // verbatim. Drop the clause (return None) instead, matching
            // qdrant/redis/valkey/vectorsets.
            if !(value.is_string() || value.is_number() || value.is_boolean()) {
                return None;
            }
            let vk = value_key(value);
            Some(serde_json::json!({
                "path": [field_name],
                "operator": "Equal",
                vk: value,
            }))
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            let mut operands = Vec::new();

            let value_key = |v: &serde_json::Value| -> &str {
                if v.is_i64() {
                    "valueInt"
                } else if v.is_string() {
                    // ISO-8601 datetime bound over a `date` property — Weaviate's
                    // gRPC filter compares dates via valueText (RFC3339).
                    "valueText"
                } else {
                    "valueNumber"
                }
            };

            if let Some(lt) = criteria_obj.get("lt") {
                if !lt.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "LessThan",
                        value_key(lt): lt,
                    }));
                }
            }
            if let Some(gt) = criteria_obj.get("gt") {
                if !gt.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "GreaterThan",
                        value_key(gt): gt,
                    }));
                }
            }
            if let Some(lte) = criteria_obj.get("lte") {
                if !lte.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "LessThanEqual",
                        value_key(lte): lte,
                    }));
                }
            }
            if let Some(gte) = criteria_obj.get("gte") {
                if !gte.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "GreaterThanEqual",
                        value_key(gte): gte,
                    }));
                }
            }

            if operands.is_empty() {
                None
            } else if operands.len() == 1 {
                Some(operands.into_iter().next().unwrap())
            } else {
                Some(serde_json::json!({
                    "operator": "And",
                    "operands": operands,
                }))
            }
        }
        "geo" => {
            let lat = criteria.get("lat")?.as_f64()?;
            let lon = criteria.get("lon")?.as_f64()?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(serde_json::json!({
                "path": [field_name],
                "operator": "WithinGeoRange",
                "valueGeoRange": {
                    "geoCoordinates": {
                        "latitude": lat,
                        "longitude": lon,
                    },
                    "distance": {
                        "max": radius,
                    }
                }
            }))
        }
        _ => None,
    }
}

impl Engine for WeaviateEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let client = self.create_client()?;

        println!("Deleting existing class...");
        self.delete_class(&client)?;

        println!("Creating class '{}'...", self.class_name);
        self.create_class(&client, dataset)?;
        println!("Class '{}' created.", self.class_name);

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
            self.parallel, self.batch_size
        );
        // Map field name -> dataset schema type so uploads can coerce values
        // to the types Weaviate's strict schema expects (e.g. int, not string).
        let schema_types: HashMap<String, String> = dataset
            .config
            .schema
            .as_ref()
            .and_then(|s| s.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        let upload_start = Instant::now();
        self.upload_parallel(&ids, &vectors, &metadata, &schema_types)?;
        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        let total_time = read_time + upload_time;

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: vectors.len(),
            parallel: self.parallel,
            batch_size: self.batch_size,
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

        // Apply search-time vectorIndexConfig
        let client = self.create_client()?;
        self.setup_search(&client, params)?;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<QueryFilter<serde_json::Value>> =
            conditions.resolve_all("Weaviate", parse_weaviate_conditions)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` BEFORE the parallel region. `tops[idx]`
        // reproduces the same k each request embeds, so recall is computed
        // against an identical result set on both transports.
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

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);

        // Vector search runs over gRPC by default (packed vectors — the
        // high-throughput path) for both filtered and unfiltered queries;
        // GraphQL is the fallback when WEAVIATE_USE_GRAPHQL is set or a filter
        // uses a condition the gRPC proto can't express. gRPC concurrency is
        // async tasks on a shared runtime (one connection per task, HTTP/2);
        // GraphQL concurrency is blocking OS threads.
        let grpc_ep: Option<tonic::transport::Endpoint> =
            if std::env::var("WEAVIATE_USE_GRAPHQL").is_ok() {
                None
            } else {
                match tonic::transport::Endpoint::from_shared(self.grpc_endpoint.clone()) {
                    Ok(ep) => Some(
                        ep.timeout(std::time::Duration::from_secs(self.timeout))
                            .connect_timeout(std::time::Duration::from_secs(30)),
                    ),
                    Err(e) => {
                        eprintln!(
                            "Warning: invalid gRPC endpoint {} ({}); falling back to GraphQL",
                            self.grpc_endpoint, e
                        );
                        None
                    }
                }
            };
        // Translate each query's where-filter tree into the gRPC `Filters`
        // message (built once, outside the timed window). A query with no filter
        // stays `None`; a filter that translates becomes `Some(Filters)`. If a
        // filtered query can't be expressed in the proto, its entry is `None`
        // even though the query carries a filter — see `untranslatable` below.
        // Count over the executed slice only (`--num-queries` may cap the run).
        let grpc_filters: Vec<Option<Filters>> = parsed_filters[..num_to_run]
            .iter()
            .map(|f| match f.as_ref() {
                Some(where_json) => where_json_to_grpc_filters(where_json),
                None => None,
            })
            .collect();
        let num_filtered = parsed_filters[..num_to_run]
            .iter()
            .filter(|f| f.is_filtered())
            .count();
        // Filtered queries whose condition the gRPC proto can't express. If any
        // exist we keep the WHOLE run on GraphQL (single transport, never
        // silently unfiltered), so gRPC's Filters and GraphQL's where stay in
        // lockstep on the queries that DO run over gRPC.
        let untranslatable = parsed_filters[..num_to_run]
            .iter()
            .zip(&grpc_filters)
            .filter(|(orig, translated)| orig.is_filtered() && translated.is_none())
            .count();
        let use_grpc = grpc_ep.is_some() && untranslatable == 0;

        // Measurement-fairness note: the query transport is recorded so a reviewer
        // can see which wire carried the run. Filters are translated to the gRPC
        // `Filters` message from the same where-tree the GraphQL path serializes,
        // so filtered and unfiltered runs are now measured on the SAME transport
        // (gRPC) unless a filter forces the GraphQL fallback.
        let transport = if use_grpc { "grpc" } else { "graphql" };
        if grpc_ep.is_some() && !use_grpc && untranslatable > 0 {
            println!(
                "\tWeaviate search transport: graphql (forced — {} of {} filtered queries use a \
                 condition the gRPC proto can't express; the whole set runs on GraphQL so filter \
                 semantics stay identical across queries)",
                untranslatable, num_to_run
            );
        } else if use_grpc && num_filtered > 0 {
            println!(
                "\tWeaviate search transport: grpc ({} of {} queries carry a filter, translated \
                 to the gRPC Filters message)",
                num_filtered, num_to_run
            );
        } else {
            println!("\tWeaviate search transport: {}", transport);
        }

        // Precompute the per-query request payloads BEFORE the parallel region so
        // the timed window wraps only RPC send + receive + decode (request build is
        // client CPU work, hoisted like qdrant builds its request before
        // `query_start`). Only the payloads for the transport actually used are
        // built. GraphQL: the GraphQL-string build + JSON-wrap (the heaviest client
        // step). gRPC: the LE-byte vector packing + `SearchRequest`.
        let graphql_bodies: Vec<Vec<u8>> = if use_grpc {
            Vec::new()
        } else {
            (0..num_to_run)
                .map(|idx| {
                    build_graphql_body(
                        &self.class_name,
                        &queries[idx],
                        tops[idx],
                        parsed_filters[idx].as_ref(),
                    )
                })
                .collect()
        };
        let grpc_requests: Vec<SearchRequest> = if use_grpc {
            (0..num_to_run)
                .map(|idx| {
                    build_grpc_request(
                        &self.class_name,
                        &queries[idx],
                        tops[idx],
                        grpc_filters[idx].as_ref(),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        let neighbors = Arc::new(neighbors);
        let tops = Arc::new(tops);
        let graphql_bodies = Arc::new(graphql_bodies);
        let grpc_requests = Arc::new(grpc_requests);

        // Barrier-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window (mirrors redis.rs / vertex.rs). Every
        // worker connects + primes one discarded query, then blocks on `ready`;
        // the main thread stamps the shared start instant into `start_cell` and
        // releases `go`, so the measurement clock starts only once all workers are
        // warm and poised. A worker that fails to connect MUST still cross both
        // barriers before returning, or the run would deadlock. `total_time` is
        // read from `start_cell` below, so it excludes connection setup and the
        // cold first query. Both transports (gRPC tasks / GraphQL threads) use the
        // same `parallel + 1` barrier counts (main thread is the +1).
        let ready = Arc::new(std::sync::Barrier::new(parallel + 1));
        let go = Arc::new(std::sync::Barrier::new(parallel + 1));
        let start_cell = Arc::new(std::sync::OnceLock::<Instant>::new());

        // Per-thread sample buffers merged after the workers finish — no per-query
        // Mutex<Vec> contention in the timed loop (see redis.rs::search). Both the
        // gRPC async tasks and the GraphQL OS threads accumulate locally and return
        // their buffers; metrics are order-independent so results are unchanged.
        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        type SampleBuffers = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

        if use_grpc {
            // ── gRPC: async task fan-out. `parallel` tasks, each its own
            //     connection, awaiting searches off a shared atomic work queue.
            //     This scales with concurrency (unlike thread-per-block_on). ──
            let endpoint = grpc_ep.unwrap();
            // Size the pool at `parallel + 1` so every worker task can block on the
            // std start-barrier concurrently (the main future is the +1). With the
            // default (num-CPU) pool a `parallel` larger than the CPU count would
            // strand un-scheduled tasks behind barrier-blocked threads and deadlock.
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(parallel + 1)
                .enable_all()
                .build()
                .map_err(|e| format!("failed to build tokio runtime: {}", e))?;
            let collected: Vec<SampleBuffers> = rt.block_on(async {
                let mut tasks = Vec::with_capacity(parallel);
                for _ in 0..parallel {
                    let endpoint = endpoint.clone();
                    let api_key = self.api_key.clone();
                    let grpc_requests = Arc::clone(&grpc_requests);
                    let neighbors = Arc::clone(&neighbors);
                    let tops = Arc::clone(&tops);
                    let query_idx = Arc::clone(&query_idx);
                    let ready = Arc::clone(&ready);
                    let go = Arc::clone(&go);
                    let pb = pb.clone();
                    tasks.push(tokio::spawn(async move {
                        let mut t = Vec::new();
                        let mut p = Vec::new();
                        let mut r = Vec::new();
                        let mut mr = Vec::new();
                        let mut nd = Vec::new();

                        let channel = match endpoint.connect().await {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!("gRPC connect failed: {}", e);
                                // Still cross both barriers so peers + main aren't
                                // stranded (the barrier count includes this task).
                                ready.wait();
                                go.wait();
                                return (t, p, r, mr, nd);
                            }
                        };
                        let mut client = WeaviateClient::new(channel);

                        // Prime this connection with ONE discarded search (query 0)
                        // so the cold first round-trip + server JIT/cache warm-up is
                        // outside the measured window. Best effort: result ignored,
                        // no sample recorded. Reuses the normal per-query RPC path.
                        if !grpc_requests.is_empty() {
                            let prime = grpc_requests[0].clone();
                            let _ = grpc_search(&mut client, prime, api_key.as_deref()).await;
                        }

                        // Signal "connected + primed", then block until the main
                        // thread stamps the shared start and releases everyone.
                        ready.wait();
                        go.wait();
                        loop {
                            let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                            if idx >= num_to_run {
                                break;
                            }
                            let top = tops[idx];
                            // Owned request copy for the RPC, cloned OUT of the
                            // window (memcpy of already-packed bytes, not a rebuild).
                            let request = grpc_requests[idx].clone();
                            // Timed window: send + receive + protobuf decode. Request
                            // build is hoisted (out); id extraction after `elapsed` (out).
                            let query_start = Instant::now();
                            let reply = grpc_search(&mut client, request, api_key.as_deref()).await;
                            let query_time = query_start.elapsed().as_secs_f64();
                            match reply.and_then(|reply| extract_grpc_hits(&reply)) {
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
                                Err(e) => eprintln!("Search query {} failed: {}", idx, e),
                            }
                            pb.inc(1);
                        }
                        (t, p, r, mr, nd)
                    }));
                }
                // All tasks have connected + primed and are poised on `ready`.
                // Stamp the measurement start, then release everyone via `go`; the
                // barrier count includes this main future (the +1).
                ready.wait();
                start_cell.set(Instant::now()).ok();
                go.wait();
                let mut out = Vec::with_capacity(tasks.len());
                for task in tasks {
                    match task.await {
                        Ok(buf) => out.push(buf),
                        Err(e) => eprintln!("gRPC search task failed: {}", e),
                    }
                }
                out
            });
            for (t, p, r, mr, nd) in collected {
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
        } else {
            // ── GraphQL: blocking OS-thread fan-out (each thread its own client). ──
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(parallel);
                for _ in 0..parallel {
                    let base_url = self.base_url.clone();
                    let class_name = self.class_name.clone();
                    let api_key = self.api_key.clone();
                    let timeout = self.timeout;
                    let neighbors = Arc::clone(&neighbors);
                    let tops = Arc::clone(&tops);
                    let graphql_bodies = Arc::clone(&graphql_bodies);
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

                        let client = match reqwest::blocking::Client::builder()
                            .timeout(std::time::Duration::from_secs(timeout))
                            .build()
                        {
                            Ok(c) => c,
                            Err(_) => {
                                // Still cross both barriers so peers + main aren't
                                // stranded (the barrier count includes this worker).
                                ready.wait();
                                go.wait();
                                return (t, p, r, mr, nd);
                            }
                        };

                        // Prime this client with ONE discarded search (query 0) so
                        // the cold first round-trip (TCP/TLS + server warm-up) is
                        // outside the measured window. Best effort: result ignored,
                        // no sample recorded. Reuses the normal per-query HTTP path.
                        if !graphql_bodies.is_empty() {
                            let _ = send_graphql(
                                &client,
                                &base_url,
                                api_key.as_deref(),
                                &graphql_bodies[0],
                            );
                        }

                        // Signal "connected + primed", then block until the main
                        // thread stamps the shared start and releases everyone.
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
                            let response = send_graphql(
                                &client,
                                &base_url,
                                api_key.as_deref(),
                                &graphql_bodies[idx],
                            );
                            let query_time = query_start.elapsed().as_secs_f64();
                            match response
                                .and_then(|resp_body| extract_graphql_hits(&resp_body, &class_name))
                            {
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
                                Err(e) => eprintln!("Search query {} failed: {}", idx, e),
                            }
                            pb.inc(1);
                        }
                        (t, p, r, mr, nd)
                    }));
                }
                // All workers have connected + primed and are poised on `ready`.
                // Stamp the measurement start, then release everyone via `go`; the
                // barrier count includes this main thread (the +1).
                ready.wait();
                start_cell.set(Instant::now()).ok();
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
        }

        pb.finish_and_clear();
        // total_time is measured from the barrier release (start_cell), so it
        // excludes connection setup and the cold first (primed) query.
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
        let client = self.create_client()?;
        self.delete_class(&client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Load-bearing: the hoisted GraphQL request body bytes must equal what the
    /// old inline `.json(&{"query": graphql})` path put on the wire, and the
    /// vector float formatting inside the pinned GraphQL string must not drift.
    #[test]
    fn build_graphql_body_bytes_match_json_serialization() {
        let vec = vec![0.1f32, -0.2, 0.3];
        let top = 2usize;

        // Reconstruct the exact GraphQL string the old inline path built.
        let vector_str = "0.1, -0.2, 0.3";
        let where_clause = String::new();
        let graphql = format!(
            r#"{{
            Get {{
                {class_name}(
                    nearVector: {{ vector: [{vector_str}] }},
                    limit: {top}
                    {where_clause}
                ) {{
                    _additional {{
                        id
                        distance
                    }}
                }}
            }}
        }}"#,
            class_name = "Bench",
            vector_str = vector_str,
            top = top,
            where_clause = where_clause,
        );
        let expected = serde_json::to_vec(&json!({ "query": graphql })).unwrap();

        let body = build_graphql_body("Bench", &vec, top, None);
        assert_eq!(body, expected);

        // Pin the load-bearing float formatting and structure directly.
        let decoded: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let q = decoded["query"].as_str().unwrap();
        assert!(q.contains("vector: [0.1, -0.2, 0.3]"), "q={}", q);
        assert!(q.contains("limit: 2"), "q={}", q);
    }

    #[test]
    fn graphql_literal_uses_bare_keys_and_enum_operator() {
        let f = build_weaviate_filter("color", "match", &json!({"any": ["red", "blue"]})).unwrap();
        let g = json_to_graphql_literal(&f);
        assert!(g.contains("operator: Or"), "g={}", g);
        assert!(g.contains("operator: Equal"), "g={}", g);
        assert!(g.contains("path: [\"color\"]"), "g={}", g);
        assert!(g.contains("valueText: \"red\""), "g={}", g);
        assert!(!g.contains("\"operator\""), "g={}", g);
        assert!(!g.contains("\"path\""), "g={}", g);
    }

    #[test]
    fn graphql_literal_numbers_unquoted() {
        let f = json!({"path": ["size"], "operator": "Equal", "valueInt": 3});
        let g = json_to_graphql_literal(&f);
        assert!(g.contains("valueInt: 3"), "g={}", g);
        assert!(!g.contains("\"valueInt\""), "g={}", g);
    }

    #[test]
    fn match_any_string_list_emits_or_of_equal() {
        let c = build_weaviate_filter("color", "match", &json!({"any": ["red", "blue"]})).unwrap();
        assert_eq!(c["operator"], "Or");
        let ops = c["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0]["operator"], "Equal");
        assert_eq!(ops[0]["path"], json!(["color"]));
        assert_eq!(ops[0]["valueText"], "red");
        assert_eq!(ops[1]["valueText"], "blue");
    }

    #[test]
    fn match_any_int_list_emits_or_of_equal() {
        let c = build_weaviate_filter("size", "match", &json!({"any": [1, 2, 3]})).unwrap();
        assert_eq!(c["operator"], "Or");
        let ops = c["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0]["operator"], "Equal");
        assert_eq!(ops[0]["valueInt"], 1);
    }

    #[test]
    fn match_any_single_element_is_bare_equal() {
        let c = build_weaviate_filter("color", "match", &json!({"any": ["red"]})).unwrap();
        assert_eq!(c["operator"], "Equal");
        assert_eq!(c["valueText"], "red");
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // Empty IN-set -> unsatisfiable And(Equal(x), NotEqual(x)): matches
        // nothing (clause not dropped, never inverted to match-all).
        let c = build_weaviate_filter("color", "match", &json!({"any": []})).unwrap();
        assert_eq!(c["operator"], "And");
        let ops = c["operands"].as_array().unwrap();
        assert_eq!(ops[0]["operator"], "Equal");
        assert_eq!(ops[1]["operator"], "NotEqual");
        assert_eq!(ops[0]["valueText"], ops[1]["valueText"]);
    }

    #[test]
    fn match_exact_value_still_works() {
        let c = build_weaviate_filter("color", "match", &json!({"value": "red"})).unwrap();
        assert_eq!(c["operator"], "Equal");
        assert_eq!(c["valueText"], "red");
    }

    // #121: a non-scalar `value` (object/null) is malformed input — the
    // canonical model uses `match.any` for lists. Without the guard, `value_key`
    // falls back to "valueText" and forwards it verbatim. It must be dropped
    // (None). Matches qdrant/redis/valkey/vectorsets. (Array case covered by
    // exact_match_array_value_is_none.)
    #[test]
    fn match_non_scalar_value_dropped() {
        assert!(build_weaviate_filter("n", "match", &json!({"value": {"x": 1}})).is_none());
        assert!(build_weaviate_filter("n", "match", &json!({"value": null})).is_none());
    }

    #[test]
    fn keyword_property_is_filterable_with_field_tokenization() {
        // Modern Weaviate needs `indexFilterable` (not the deprecated
        // `indexInverted`) for `Equal` filters to match; keyword uses `field`
        // tokenization so `Equal "red"` matches the whole value exactly.
        let p = weaviate_property("color", "keyword").unwrap();
        assert_eq!(p["dataType"], json!(["text"]));
        assert_eq!(p["indexFilterable"], true);
        assert_eq!(p["tokenization"], "field");
        assert_eq!(p["indexSearchable"], true);
        assert!(p.get("indexInverted").is_none(), "p={}", p);
    }

    // #88: the multi-valued keyword field `labels` is a `text[]` array property
    // (with `field` tokenization for whole-value element equality), so
    // `ContainsAny`/`Equal` filters match individual elements. A scalar `text`
    // could only compare the whole value.
    #[test]
    fn labels_property_is_text_array() {
        let p = weaviate_property("labels", "keyword").unwrap();
        assert_eq!(p["dataType"], json!(["text[]"]));
        assert_eq!(p["indexFilterable"], true);
        assert_eq!(p["tokenization"], "field");
    }

    #[test]
    fn int_property_is_filterable_without_searchable() {
        // `indexSearchable` is invalid on non-text properties and would make
        // Weaviate reject the whole schema; it must be omitted for int.
        let p = weaviate_property("size", "int").unwrap();
        assert_eq!(p["dataType"], json!(["int"]));
        assert_eq!(p["indexFilterable"], true);
        assert!(p.get("indexSearchable").is_none(), "p={}", p);
        assert!(p.get("tokenization").is_none(), "p={}", p);
    }

    #[test]
    fn unsupported_property_type_is_skipped() {
        assert!(weaviate_property("x", "bogus").is_none());
    }

    #[test]
    fn coerce_int_string_to_json_number() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        // The reader hands numbers over as strings; an `int` schema field must
        // become a JSON number or Weaviate rejects the whole object.
        let v = MetadataValue::String("1".to_string());
        assert_eq!(coerce_metadata_value(Some("int"), &v), json!(1));
        let f = MetadataValue::String("1.5".to_string());
        assert_eq!(coerce_metadata_value(Some("float"), &f), json!(1.5));
    }

    #[test]
    fn coerce_keyword_stays_string() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let v = MetadataValue::String("red".to_string());
        assert_eq!(coerce_metadata_value(Some("keyword"), &v), json!("red"));
        // Unknown/unmapped fields pass through unchanged.
        assert_eq!(coerce_metadata_value(None, &v), json!("red"));
    }

    // ── OR-branch of the condition parser ──────────────────────────────────

    #[test]
    fn or_only_emits_or_operator_with_two_operands() {
        let cond = json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        let f = parse_weaviate_conditions(&cond).unwrap();
        assert_eq!(f["operator"], "Or");
        assert_eq!(f["operands"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn and_plus_or_wraps_both_groups_in_and() {
        let cond = json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}},{"c":{"match":{"value":"z"}}}],
        });
        let f = parse_weaviate_conditions(&cond).unwrap();
        assert_eq!(f["operator"], "And");
        let ops = f["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 2);
        // One operand is the collapsed AND (a bare Equal), the other the OR group.
        assert!(ops.iter().any(|o| o["operator"] == "Or"));
        assert!(ops.iter().any(|o| o["operator"] == "Equal"));
    }

    // ── Range operators ────────────────────────────────────────────────────

    #[test]
    fn range_single_ops_map_to_weaviate_operators() {
        for (op, wv) in [
            ("lt", "LessThan"),
            ("lte", "LessThanEqual"),
            ("gt", "GreaterThan"),
            ("gte", "GreaterThanEqual"),
        ] {
            let f = build_weaviate_filter("n", "range", &json!({ op: 5 })).unwrap();
            assert_eq!(f["operator"], wv, "op={}", op);
            assert_eq!(f["path"], json!(["n"]));
            assert_eq!(f["valueInt"], 5); // i64 → valueInt
        }
    }

    #[test]
    fn range_float_bound_uses_value_number() {
        let f = build_weaviate_filter("n", "range", &json!({"lt":1.5})).unwrap();
        assert_eq!(f["operator"], "LessThan");
        assert_eq!(f["valueNumber"], 1.5);
    }

    #[test]
    fn range_two_sided_gte_lt_wraps_in_and() {
        let f = build_weaviate_filter("n", "range", &json!({"gte":10,"lt":20})).unwrap();
        assert_eq!(f["operator"], "And");
        let ops = f["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 2);
        // Emitted in fixed order lt, gt, lte, gte.
        assert_eq!(ops[0]["operator"], "LessThan");
        assert_eq!(ops[0]["valueInt"], 20);
        assert_eq!(ops[1]["operator"], "GreaterThanEqual");
        assert_eq!(ops[1]["valueInt"], 10);
    }

    #[test]
    fn range_unknown_op_is_none() {
        assert!(build_weaviate_filter("n", "range", &json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_is_none() {
        assert!(
            build_weaviate_filter("n", "range", &json!({"gte":serde_json::Value::Null})).is_none()
        );
    }

    // ── Geo filter ─────────────────────────────────────────────────────────

    #[test]
    fn geo_with_radius_emits_within_geo_range() {
        let f = build_weaviate_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0,"radius":500}))
            .unwrap();
        assert_eq!(f["operator"], "WithinGeoRange");
        assert_eq!(f["valueGeoRange"]["geoCoordinates"]["latitude"], 20.0);
        assert_eq!(f["valueGeoRange"]["geoCoordinates"]["longitude"], 10.0);
        assert_eq!(f["valueGeoRange"]["distance"]["max"], 500.0);
    }

    #[test]
    fn geo_without_radius_uses_default_1000() {
        let f = build_weaviate_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0})).unwrap();
        assert_eq!(f["valueGeoRange"]["distance"]["max"], 1000.0);
    }

    #[test]
    fn geo_missing_lat_or_lon_is_none() {
        assert!(build_weaviate_filter("loc", "geo", &json!({"lon":10.0,"radius":5})).is_none());
        assert!(build_weaviate_filter("loc", "geo", &json!({"lat":20.0,"radius":5})).is_none());
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn distance_mapping_covers_all_arms() {
        assert_eq!(map_weaviate_distance("l2").unwrap(), "l2-squared");
        assert_eq!(map_weaviate_distance("euclidean").unwrap(), "l2-squared");
        assert_eq!(map_weaviate_distance("cosine").unwrap(), "cosine");
        assert_eq!(map_weaviate_distance("angular").unwrap(), "cosine");
        assert_eq!(map_weaviate_distance("dot").unwrap(), "dot");
        assert_eq!(map_weaviate_distance("ip").unwrap(), "dot");
        assert_eq!(map_weaviate_distance("COSINE").unwrap(), "cosine");
        assert!(map_weaviate_distance("nope").is_err());
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    #[test]
    fn exact_match_int_float_bool_use_correct_value_key() {
        let i = build_weaviate_filter("n", "match", &json!({"value":5})).unwrap();
        assert_eq!(i["operator"], "Equal");
        assert_eq!(i["valueInt"], 5);

        let fl = build_weaviate_filter("n", "match", &json!({"value":1.5})).unwrap();
        assert_eq!(fl["valueNumber"], 1.5);

        let b = build_weaviate_filter("flag", "match", &json!({"value":true})).unwrap();
        assert_eq!(b["valueBoolean"], true);
    }

    #[test]
    fn exact_match_array_value_is_none() {
        // #121: the scalar exact-match arm now guards non-scalars; a JSON array
        // value is dropped (None) instead of falling through value_key's default
        // (valueText) with the array as the value. Matches qdrant/redis/valkey/
        // vectorsets.
        assert!(build_weaviate_filter("n", "match", &json!({"value":[1,2]})).is_none());
    }

    // ── gRPC Filters translation (must mirror the GraphQL where-tree) ───────

    use super::filters::{Operator, TestValue};

    /// Convert a `conditions` blob the same way both transports do: through
    /// `parse_weaviate_conditions` (GraphQL where-tree) then
    /// `where_json_to_grpc_filters` (gRPC Filters). Returns the gRPC message.
    fn grpc_filters_for(conditions: serde_json::Value) -> Filters {
        let where_tree =
            parse_weaviate_conditions(&conditions).expect("conditions must produce a where-tree");
        where_json_to_grpc_filters(&where_tree).expect("where-tree must translate to gRPC")
    }

    fn property_of(f: &Filters) -> &str {
        match f.target.as_ref().unwrap().target.as_ref().unwrap() {
            filter_target::Target::Property(p) => p,
            _ => panic!("expected a Property target"),
        }
    }

    #[test]
    fn grpc_match_keyword_maps_to_equal_value_text() {
        let f = grpc_filters_for(json!({"and":[{"color":{"match":{"value":"red"}}}]}));
        assert_eq!(f.operator, Operator::Equal as i32);
        assert_eq!(property_of(&f), "color");
        assert_eq!(f.test_value, Some(TestValue::ValueText("red".into())));
        assert!(f.filters.is_empty());
    }

    #[test]
    fn grpc_match_int_maps_to_equal_value_int() {
        let f = grpc_filters_for(json!({"and":[{"size":{"match":{"value":7}}}]}));
        assert_eq!(f.operator, Operator::Equal as i32);
        assert_eq!(property_of(&f), "size");
        assert_eq!(f.test_value, Some(TestValue::ValueInt(7)));
    }

    #[test]
    fn grpc_match_bool_maps_to_equal_value_boolean() {
        let f = grpc_filters_for(json!({"and":[{"flag":{"match":{"value":true}}}]}));
        assert_eq!(f.operator, Operator::Equal as i32);
        assert_eq!(f.test_value, Some(TestValue::ValueBoolean(true)));
    }

    #[test]
    fn grpc_match_any_maps_to_or_of_equals() {
        let f = grpc_filters_for(json!({
            "and":[{"color":{"match":{"any":["red","blue"]}}}]
        }));
        assert_eq!(f.operator, Operator::Or as i32);
        assert_eq!(f.filters.len(), 2);
        for (child, want) in f.filters.iter().zip(["red", "blue"]) {
            assert_eq!(child.operator, Operator::Equal as i32);
            assert_eq!(property_of(child), "color");
            assert_eq!(child.test_value, Some(TestValue::ValueText(want.into())));
        }
    }

    #[test]
    fn grpc_match_any_int_maps_to_or_of_equal_value_int() {
        let f = grpc_filters_for(json!({"and":[{"size":{"match":{"any":[1,2]}}}]}));
        assert_eq!(f.operator, Operator::Or as i32);
        assert_eq!(f.filters.len(), 2);
        assert_eq!(f.filters[0].test_value, Some(TestValue::ValueInt(1)));
        assert_eq!(f.filters[1].test_value, Some(TestValue::ValueInt(2)));
    }

    #[test]
    fn grpc_range_two_sided_maps_to_and_of_less_and_greater() {
        let f = grpc_filters_for(json!({"and":[{"n":{"range":{"gte":10,"lt":20}}}]}));
        assert_eq!(f.operator, Operator::And as i32);
        assert_eq!(f.filters.len(), 2);
        // Fixed emission order: lt, gt, lte, gte.
        assert_eq!(f.filters[0].operator, Operator::LessThan as i32);
        assert_eq!(f.filters[0].test_value, Some(TestValue::ValueInt(20)));
        assert_eq!(f.filters[1].operator, Operator::GreaterThanEqual as i32);
        assert_eq!(f.filters[1].test_value, Some(TestValue::ValueInt(10)));
    }

    #[test]
    fn grpc_range_float_bound_uses_value_number() {
        let f = grpc_filters_for(json!({"and":[{"n":{"range":{"lt":1.5}}}]}));
        assert_eq!(f.operator, Operator::LessThan as i32);
        assert_eq!(f.test_value, Some(TestValue::ValueNumber(1.5)));
    }

    #[test]
    fn grpc_or_group_maps_to_or_operator() {
        let f = grpc_filters_for(json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]}));
        assert_eq!(f.operator, Operator::Or as i32);
        assert_eq!(f.filters.len(), 2);
        assert_eq!(property_of(&f.filters[0]), "a");
        assert_eq!(property_of(&f.filters[1]), "b");
    }

    #[test]
    fn grpc_and_plus_or_nests_both_groups_under_and() {
        let f = grpc_filters_for(json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}},{"c":{"match":{"value":"z"}}}],
        }));
        assert_eq!(f.operator, Operator::And as i32);
        assert_eq!(f.filters.len(), 2);
        // One operand is the collapsed AND (a bare Equal), the other the OR group.
        assert!(f.filters.iter().any(|c| c.operator == Operator::Or as i32));
        assert!(f
            .filters
            .iter()
            .any(|c| c.operator == Operator::Equal as i32));
    }

    #[test]
    fn grpc_geo_maps_to_within_geo_range_value_geo() {
        // `geo` isn't produced through parse_weaviate_conditions' and/or entries
        // in these fixtures, so translate the leaf directly.
        let leaf =
            build_weaviate_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0,"radius":500}))
                .unwrap();
        let f = where_json_to_grpc_filters(&leaf).unwrap();
        assert_eq!(f.operator, Operator::WithinGeoRange as i32);
        assert_eq!(property_of(&f), "loc");
        match f.test_value.unwrap() {
            TestValue::ValueGeo(g) => {
                assert_eq!(g.latitude, 20.0);
                assert_eq!(g.longitude, 10.0);
                assert_eq!(g.distance, 500.0);
            }
            other => panic!("expected ValueGeo, got {:?}", other),
        }
    }

    #[test]
    fn grpc_non_scalar_equal_value_is_untranslatable() {
        // A degenerate array-under-valueText leaf (build_weaviate_filter no longer
        // produces one after #121, so it's constructed directly here) can't become
        // a proto scalar, so translation returns None → the run falls back to
        // GraphQL. Guards the gRPC translator's own scalar-accessor check.
        let leaf = json!({"path": ["n"], "operator": "Equal", "valueText": [1, 2]});
        assert!(where_json_to_grpc_filters(&leaf).is_none());
    }

    #[test]
    fn grpc_empty_match_any_maps_to_unsatisfiable_and() {
        // Empty IN-set → And(Equal(x), NotEqual(x)): matches nothing, mirroring
        // the GraphQL path exactly (never inverted to match-all).
        let f = grpc_filters_for(json!({"and":[{"color":{"match":{"any":[]}}}]}));
        assert_eq!(f.operator, Operator::And as i32);
        assert_eq!(f.filters.len(), 2);
        assert_eq!(f.filters[0].operator, Operator::Equal as i32);
        assert_eq!(f.filters[1].operator, Operator::NotEqual as i32);
        assert_eq!(f.filters[0].test_value, f.filters[1].test_value);
    }
}
