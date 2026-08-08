//! Chroma engine implementation.
//!
//! Uses the Chroma v2 REST API via `reqwest::blocking`. Chroma stores one
//! collection of records (`ids` + `embeddings` + scalar `metadatas`) and filters
//! searches with a `where` document that maps directly onto our canonical filter
//! model (`$eq`/`$in`/`$gte`… leaves, native `$and`/`$or` for AND/OR/NESTED).
//!
//! Supported filter datatypes: keyword, int, float, bool, uuid, datetime (stored
//! as epoch-seconds int, like Milvus), `match_any` (`$in`), full-text
//! (`{match:{text}}` → `where_document` `$contains` over the record's uploaded
//! `document`), and AND/OR/nested boolean. NOT supported by Chroma's metadata
//! engine (dropped, like Dragonfly's documented limits): geo-radius and
//! multi-valued `labels` arrays (Chroma metadata values are scalar only).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

use super::geo;
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::parsers::datetime_to_epoch_secs;
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};
use vector_db_benchmark::start_gate::WorkerPool;

const DEFAULT_COLLECTION: &str = "benchmark";

pub struct ChromaEngine {
    name: String,
    collection_name: String,
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    /// `http://{host}:{port}/api/v2/tenants/{tenant}/databases/{db}`
    api_base: String,
    search_params: Vec<SearchParams>,
    /// hnsw space ("l2" | "cosine" | "ip"), set during configure.
    space: String,
    /// Collection id returned by Chroma on create, needed by add/query.
    collection_id: String,
    /// field -> declared schema type (drives datetime->epoch, bool, labels-skip).
    schema_types: HashMap<String, String>,
}

impl ChromaEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("CHROMA_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8000);

        let collection_name =
            std::env::var("CHROMA_COLLECTION").unwrap_or_else(|_| DEFAULT_COLLECTION.to_string());

        let tenant =
            std::env::var("CHROMA_TENANT").unwrap_or_else(|_| "default_tenant".to_string());
        let database =
            std::env::var("CHROMA_DATABASE").unwrap_or_else(|_| "default_database".to_string());

        let timeout: u64 = std::env::var("CHROMA_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

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
            .unwrap_or(1000) as usize;

        let root = if host.starts_with("http") {
            host.to_string()
        } else {
            format!("http://{}:{}", host, port)
        };
        let api_base = format!(
            "{}/api/v2/tenants/{}/databases/{}",
            root.trim_end_matches('/'),
            tenant,
            database
        );

        Ok(Self {
            name: engine_config.name.clone(),
            collection_name,
            timeout,
            batch_size,
            parallel,
            api_base,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            space: String::new(),
            collection_id: String::new(),
            schema_types: HashMap::new(),
        })
    }

    fn create_client(&self) -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))
    }

    fn progress_bar(&self, total: usize) -> ProgressBar {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        pb
    }

    /// Delete the collection by name (ignore "not found").
    fn drop_collection(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let url = format!("{}/collections/{}", self.api_base, self.collection_name);
        let _ = client.delete(&url).send();
        Ok(())
    }

    /// Look the collection's id up by name.
    ///
    /// `collection_id` is the one piece of engine state that is NOT a function of
    /// the dataset — it comes back from the create-collection response — so the
    /// `--skip-upload` path, which never calls configure() (#238), would otherwise
    /// query `.../collections//query` and fail every request after a green reuse
    /// check. Resolving by name recovers it without touching the corpus.
    fn resolve_collection_id(&mut self, client: &reqwest::blocking::Client) -> Result<(), String> {
        if !self.collection_id.is_empty() {
            return Ok(());
        }
        let url = format!("{}/collections/{}", self.api_base, self.collection_name);
        let resp = client
            .get(&url)
            .send()
            .map_err(|e| format!("resolve collection id request failed: {}", e))?;
        if !resp.status().is_success() {
            return Err(format!(
                "collection '{}' not found on the server (status {}) — --skip-upload needs a \
                 collection created by a previous run with --keep-data",
                self.collection_name,
                resp.status()
            ));
        }
        let v: serde_json::Value = resp
            .json()
            .map_err(|e| format!("resolve collection id response parse: {}", e))?;
        self.collection_id = v
            .get("id")
            .and_then(|x| x.as_str())
            .ok_or("resolve collection id: no id in response")?
            .to_string();
        Ok(())
    }

    /// Create the collection and remember its id.
    fn create_collection(&mut self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let url = format!("{}/collections", self.api_base);
        let body = serde_json::json!({
            "name": self.collection_name,
            "configuration": { "hnsw": { "space": self.space } },
            "get_or_create": true,
        });
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| format!("create collection request failed: {}", e))?;
        if !resp.status().is_success() {
            return Err(format!(
                "create collection failed: {}",
                resp.text().unwrap_or_default()
            ));
        }
        let v: serde_json::Value = resp
            .json()
            .map_err(|e| format!("create collection response parse: {}", e))?;
        self.collection_id = v
            .get("id")
            .and_then(|x| x.as_str())
            .ok_or("create collection: no id in response")?
            .to_string();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        let pb = self.progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();
        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let url = format!("{}/collections/{}/add", self.api_base, self.collection_id);
                let timeout = self.timeout;
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let schema_types = &self.schema_types;
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
                        let (bs, be) = batches[idx];
                        if let Err(e) = insert_batch(
                            &client,
                            &url,
                            &ids[bs..be],
                            &vectors[bs..be],
                            &metadata[bs..be],
                            schema_types,
                        ) {
                            *error.lock().unwrap() = Some(e);
                            break;
                        }
                        pb.inc((be - bs) as u64);
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
}

fn schema_type_map(dataset: &Dataset) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Some(obj) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
        for (k, v) in obj {
            if let Some(t) = v.as_str() {
                m.insert(k.clone(), t.to_string());
            }
        }
    }
    m
}

/// Convert one document's metadata into a Chroma scalar-metadata JSON object.
/// datetime -> epoch-seconds int; bool -> JSON bool; numeric-keyword stays a
/// string; multi-valued `labels` and geo are dropped (Chroma metadata is scalar).
fn metadata_to_chroma(
    meta: &MetadataItem,
    schema_types: &HashMap<String, String>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut obj = serde_json::Map::new();
    for (k, v) in &meta.fields {
        let declared = schema_types.get(k).map(|s| s.as_str());
        let v = v.coerce_for_schema(declared);
        let val: Option<serde_json::Value> = match v.as_ref() {
            MetadataValue::String(s) => match declared {
                Some("bool") => match s.as_str() {
                    "true" => Some(serde_json::Value::Bool(true)),
                    "false" => Some(serde_json::Value::Bool(false)),
                    _ => None,
                },
                // datetime stored as epoch-seconds int so Chroma's numeric range
                // operators work (Chroma has no native date type).
                Some("datetime") => datetime_to_epoch_secs(s)
                    .map(|e| serde_json::Value::Number(serde_json::Number::from(e as i64))),
                _ => Some(serde_json::Value::String(s.clone())),
            },
            MetadataValue::Int(i) => Some(serde_json::Value::Number(serde_json::Number::from(*i))),
            MetadataValue::Float(f) => {
                serde_json::Number::from_f64(*f).map(serde_json::Value::Number)
            }
            // Chroma metadata values are scalar; a `labels` array / geo point has
            // no representation and is dropped (documented limitation).
            MetadataValue::Labels(_) | MetadataValue::Geo { .. } => None,
        };
        if let Some(val) = val {
            obj.insert(k.clone(), val);
        }
    }
    obj
}

/// The (deterministic) name of the `text`-typed schema field, if any. Its value
/// is stored as each record's Chroma `document` so `{match:{text}}` filters can
/// run as `where_document` `$contains` full-text search.
fn text_field(schema_types: &HashMap<String, String>) -> Option<String> {
    let mut names: Vec<&String> = schema_types
        .iter()
        .filter(|(_, t)| t.as_str() == "text")
        .map(|(k, _)| k)
        .collect();
    names.sort();
    names.first().map(|s| s.to_string())
}

fn insert_batch(
    client: &reqwest::blocking::Client,
    url: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    schema_types: &HashMap<String, String>,
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let id_strs: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    let metadatas: Vec<serde_json::Value> = (0..ids.len())
        .map(|i| match &metadata[i] {
            Some(m) => serde_json::Value::Object(metadata_to_chroma(m, schema_types)),
            None => serde_json::Value::Object(serde_json::Map::new()),
        })
        .collect();

    let mut body = serde_json::json!({
        "ids": id_strs,
        "embeddings": vectors,
        "metadatas": metadatas,
    });

    // Store the text field's value as each record's `document` so full-text
    // (`where_document` $contains) works. Chroma has one document per record.
    if let Some(tf) = text_field(schema_types) {
        let documents: Vec<serde_json::Value> = (0..ids.len())
            .map(|i| {
                let s = metadata[i].as_ref().and_then(|m| {
                    m.fields.iter().find(|(k, _)| k == &tf).and_then(|(_, v)| {
                        if let MetadataValue::String(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                });
                serde_json::Value::String(s.unwrap_or_default())
            })
            .collect();
        body.as_object_mut()
            .unwrap()
            .insert("documents".to_string(), serde_json::Value::Array(documents));
    }
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .map_err(|e| format!("add request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "add batch failed: {}",
            resp.text().unwrap_or_default()
        ));
    }
    Ok(())
}

fn map_chroma_space(distance: &str) -> Result<&'static str, String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok("l2"),
        "cosine" => Ok("cosine"),
        "ip" | "dot" | "dotproduct" => Ok("ip"),
        other => Err(format!("Unsupported distance metric for Chroma: {}", other)),
    }
}

/// Combine a list of Chroma where-operands under `$and`/`$or`. A single operand
/// is returned bare (Chroma rejects a 1-element `$and`/`$or`); an empty list is
/// `None`.
fn combine(mut ops: Vec<serde_json::Value>, key: &str) -> Option<serde_json::Value> {
    match ops.len() {
        0 => None,
        1 => Some(ops.pop().unwrap()),
        _ => Some(serde_json::json!({ key: ops })),
    }
}

/// Build a Chroma `where` document from the canonical `{and,or}` filter model.
/// Recursive: an entry that is itself an `{and:[...]}`/`{or:[...]}` group nests
/// natively via `$and`/`$or`.
pub(crate) fn build_chroma_where(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }
    // Neither this engine's filter DSL nor its metadata model can express a
    // geo radius (issue #223 — see the geo test below for the evidence), and a
    // per-LEAF refusal is not enough: this builder keeps the leaves it
    // understands, so `and(geo, keyword)` would emit only the keyword clause and
    // run a real-looking filter that constrains less than the ground truth does.
    // Refuse the whole tree, which `resolve_all` turns into a hard error.
    if geo::conditions_mention_geo(conditions) {
        return None;
    }
    let mut clauses = Vec::new();

    if let Some(and_items) = obj.get("and").and_then(|v| v.as_array()) {
        let ops: Vec<serde_json::Value> = and_items.iter().filter_map(build_chroma_entry).collect();
        if let Some(c) = combine(ops, "$and") {
            clauses.push(c);
        }
    }
    if let Some(or_items) = obj.get("or").and_then(|v| v.as_array()) {
        let ops: Vec<serde_json::Value> = or_items.iter().filter_map(build_chroma_entry).collect();
        if let Some(c) = combine(ops, "$or") {
            clauses.push(c);
        }
    }
    // Top-level `and` and `or` groups are themselves AND-combined (mirrors the
    // other engines: an object with both keys means "(and-group) AND (or-group)").
    combine(clauses, "$and")
}

fn build_chroma_entry(entry: &serde_json::Value) -> Option<serde_json::Value> {
    let entry_obj = entry.as_object()?;
    // Nested group: recurse and let $and/$or nest natively.
    if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
        return build_chroma_where(entry);
    }
    // Leaf: {field: {op: criteria}} — a field may carry several ops (AND them).
    let mut ops = Vec::new();
    for (field, filter_obj) in entry_obj {
        if let Some(fo) = filter_obj.as_object() {
            for (cond_type, criteria) in fo {
                if let Some(c) = build_chroma_leaf(field, cond_type, criteria) {
                    ops.push(c);
                }
            }
        }
    }
    combine(ops, "$and")
}

fn build_chroma_leaf(
    field: &str,
    cond_type: &str,
    criteria: &serde_json::Value,
) -> Option<serde_json::Value> {
    match cond_type {
        "match" => {
            // match_any -> $in over scalar values (strings/numbers only).
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                let items: Vec<serde_json::Value> = any
                    .iter()
                    .filter(|v| v.is_string() || v.is_number())
                    .cloned()
                    .collect();
                // An empty IN-set matches nothing; emit a valid never-true clause
                // rather than dropping the filter (which would return every row).
                return Some(serde_json::json!({ field: { "$in": items } }));
            }
            // Full-text metadata match is not a Chroma metadata filter (dropped).
            if criteria.get("text").is_some() {
                return None;
            }
            let value = criteria.get("value")?;
            if value.is_string() || value.is_number() || value.is_boolean() {
                Some(serde_json::json!({ field: { "$eq": value } }))
            } else {
                // Non-scalar exact-match value is malformed (lists use match.any).
                None
            }
        }
        "range" => {
            let obj = criteria.as_object()?;
            // Render a bound: numbers verbatim; an ISO-8601 string over the
            // epoch-int metadata (converted with datetime_to_epoch_secs).
            let bound = |v: &serde_json::Value| -> Option<serde_json::Value> {
                if v.is_number() {
                    Some(v.clone())
                } else if let Some(s) = v.as_str() {
                    datetime_to_epoch_secs(s)
                        .map(|e| serde_json::Value::Number(serde_json::Number::from(e as i64)))
                } else {
                    None
                }
            };
            let mut per_bound = Vec::new();
            for (key, op) in [
                ("lt", "$lt"),
                ("gt", "$gt"),
                ("lte", "$lte"),
                ("gte", "$gte"),
            ] {
                if let Some(b) = obj.get(key) {
                    if !b.is_null() {
                        if let Some(lit) = bound(b) {
                            per_bound.push(serde_json::json!({ field: { op: lit } }));
                        }
                    }
                }
            }
            // Chroma keeps each comparison as its own operand; a two-sided range
            // is their $and (single bound returns bare).
            combine(per_bound, "$and")
        }
        // Chroma has no native geo filter.
        "geo" => None,
        _ => None,
    }
}

/// Build a Chroma `where_document` from the `{match:{text}}` leaves of the
/// filter tree (full-text `$contains`), preserving the tree's `$and`/`$or`
/// structure. Non-text leaves are skipped (they go to the metadata `where`,
/// which Chroma ANDs with `where_document`). Returns `None` when there are no
/// text matches. NOTE: because `where` and `where_document` are separate query
/// params that Chroma ANDs, a filter that ORs a text match with a metadata
/// condition can't be expressed — such queries aren't emitted by our generators.
pub(crate) fn build_chroma_where_document(
    conditions: &serde_json::Value,
) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    // Neither this engine's filter DSL nor its metadata model can express a
    // geo radius (issue #223 — see the geo test below for the evidence), and a
    // per-LEAF refusal is not enough: this builder keeps the leaves it
    // understands, so `and(geo, keyword)` would emit only the keyword clause and
    // run a real-looking filter that constrains less than the ground truth does.
    // Refuse the whole tree, which `resolve_all` turns into a hard error.
    if geo::conditions_mention_geo(conditions) {
        return None;
    }
    let mut clauses = Vec::new();
    if let Some(items) = obj.get("and").and_then(|v| v.as_array()) {
        let ops: Vec<serde_json::Value> = items.iter().filter_map(where_document_entry).collect();
        if let Some(c) = combine(ops, "$and") {
            clauses.push(c);
        }
    }
    if let Some(items) = obj.get("or").and_then(|v| v.as_array()) {
        let ops: Vec<serde_json::Value> = items.iter().filter_map(where_document_entry).collect();
        if let Some(c) = combine(ops, "$or") {
            clauses.push(c);
        }
    }
    combine(clauses, "$and")
}

fn where_document_entry(entry: &serde_json::Value) -> Option<serde_json::Value> {
    let entry_obj = entry.as_object()?;
    if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
        return build_chroma_where_document(entry);
    }
    let mut ops = Vec::new();
    for (_field, filter_obj) in entry_obj {
        if let Some(fo) = filter_obj.as_object() {
            if let Some(text) = fo
                .get("match")
                .and_then(|m| m.get("text"))
                .and_then(|t| t.as_str())
            {
                ops.push(serde_json::json!({ "$contains": text }));
            }
        }
    }
    combine(ops, "$and")
}

/// Parse the returned `{ "ids": [[...]] }` (nested per query; we send one query
/// per request) into ordered i64 ids.
fn extract_query_ids(resp_body: &serde_json::Value) -> Result<Vec<i64>, String> {
    let ids = resp_body
        .get("ids")
        .and_then(|v| v.as_array())
        .and_then(|outer| outer.first())
        .and_then(|inner| inner.as_array())
        .ok_or("query response missing ids")?;
    Ok(ids
        .iter()
        .filter_map(|v| v.as_str().and_then(|s| s.parse::<i64>().ok()))
        .collect())
}

fn build_query_body(
    query: &[f32],
    top: usize,
    where_meta: Option<&serde_json::Value>,
    where_document: Option<&serde_json::Value>,
) -> Vec<u8> {
    let mut body = serde_json::json!({
        "query_embeddings": [query],
        "n_results": top,
        "include": ["distances"],
    });
    let obj = body.as_object_mut().unwrap();
    if let Some(w) = where_meta {
        obj.insert("where".to_string(), w.clone());
    }
    if let Some(wd) = where_document {
        obj.insert("where_document".to_string(), wd.clone());
    }
    serde_json::to_vec(&body).unwrap_or_default()
}

impl Engine for ChromaEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        self.space = map_chroma_space(dataset.distance())?.to_string();
        self.schema_types = schema_type_map(dataset);
        let client = self.create_client()?;
        self.drop_collection(&client)?;
        println!("Creating Chroma collection '{}'...", self.collection_name);
        self.create_collection(&client)?;
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();
        let read_start = Instant::now();
        let (ids, vectors, metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();
        println!("Read {} vectors in {:.3}s", vectors.len(), read_time);

        let upload_start = Instant::now();
        self.upload_parallel(&ids, &vectors, &metadata)?;
        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        Ok(UploadStats {
            upload_time,
            total_time: read_time + upload_time,
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
        // On the --skip-upload path configure() never runs (#238), so the
        // collection id — the only piece of state here that is not derivable from
        // the dataset — has to be recovered from the server by name.
        {
            let client = self.create_client()?;
            self.resolve_collection_id(&client)?;
        }

        let parallel = params.parallel.unwrap_or(1) as usize;
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        // Chroma splits one filter tree across two query params it ANDs server
        // side: metadata leaves go to `where`, full-text leaves to
        // `where_document`. Either builder alone may legitimately produce
        // nothing (a purely-text filter has no `where`), so the #219 guard is
        // applied to the PAIR — only a filter that yields neither is a drop.
        let filters: Vec<QueryFilter<(Option<serde_json::Value>, Option<serde_json::Value>)>> =
            conditions.resolve_all("Chroma", |c| {
                match (build_chroma_where(c), build_chroma_where_document(c)) {
                    (None, None) => None,
                    pair => Some(pair),
                }
            })?;
        let wheres: Vec<Option<serde_json::Value>> = filters
            .iter()
            .map(|f| f.as_ref().and_then(|(w, _)| w.clone()))
            .collect();
        let where_docs: Vec<Option<serde_json::Value>> = filters
            .iter()
            .map(|f| f.as_ref().and_then(|(_, d)| d.clone()))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

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
        let bodies: Vec<Vec<u8>> = (0..num_to_run)
            .map(|idx| {
                build_query_body(
                    &queries[idx],
                    tops[idx],
                    wheres[idx].as_ref(),
                    where_docs[idx].as_ref(),
                )
            })
            .collect();

        let query_idx = Arc::new(AtomicUsize::new(0));
        let pb = self.progress_bar(num_to_run);

        // Gate-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window. Every worker connects + primes, then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant and
        // releases everyone, so the measurement clock starts only once all workers
        // are warm and poised. The gate is count-agnostic: a worker that fails to
        // set up, panics, or is never started by the OS settles its ticket and turns
        // the run into a hard error instead of a hang (#214).

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        let query_url = format!("{}/collections/{}/query", self.api_base, self.collection_id);

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "chroma-search", parallel);
            for _ in 0..parallel {
                let query_url = query_url.clone();
                let timeout = self.timeout;
                let neighbors = &neighbors;
                let tops = &tops;
                let bodies = &bodies;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                pool.spawn(move |ticket| {
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
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("chroma-search worker setup failed: {e}"));
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this client with ONE discarded query so the cold first
                    // round-trip (TCP/TLS setup + server JIT/caches) is not inside the
                    // measured window. Best effort: errors are ignored and its sample
                    // is NOT recorded.
                    if !bodies.is_empty() {
                        let _ = client
                            .post(&query_url)
                            .header("Content-Type", "application/json")
                            .body(bodies[0].clone())
                            .send()
                            .and_then(|resp| resp.error_for_status())
                            .and_then(|resp| resp.text());
                    }

                    // Signal "ready + primed", then block until the coordinator stamps
                    // the shared measurement start and releases everyone.
                    if ticket.arrive_and_wait().is_none() {
                        return (t, p, r, mr, nd);
                    }

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }
                        let top = tops[idx];
                        let query_start = Instant::now();
                        let response = client
                            .post(&query_url)
                            .header("Content-Type", "application/json")
                            .body(bodies[idx].clone())
                            .send()
                            .map_err(|e| e.to_string())
                            .and_then(|resp| {
                                if resp.status().is_success() {
                                    resp.json::<serde_json::Value>().map_err(|e| e.to_string())
                                } else {
                                    Err(resp.text().unwrap_or_default())
                                }
                            });
                        let query_time = query_start.elapsed().as_secs_f64();

                        match response.and_then(|body| extract_query_ids(&body)) {
                            Ok(ordered_ids) => {
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
                                eprintln!("Chroma search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                })?;
            }

            // Every worker is connected + primed and parked at the gate.
            // Stamp the shared measurement start and release them together.
            let (per_worker, measured_start) = pool.start()?;

            for (t, p, r, mr, nd) in per_worker {
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
            Ok(measured_start)
        })?;

        pb.finish_and_clear();
        // total_time excludes connection setup and the cold first query: it is
        // measured from the gate release, not a pre-scope instant.
        let total_time = measured_start.elapsed().as_secs_f64();
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        let client = self.create_client()?;
        self.drop_collection(&client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn space_maps_all_metrics() {
        assert_eq!(map_chroma_space("l2").unwrap(), "l2");
        assert_eq!(map_chroma_space("Cosine").unwrap(), "cosine");
        assert_eq!(map_chroma_space("dot").unwrap(), "ip");
        assert!(map_chroma_space("hamming").is_err());
    }

    #[test]
    fn match_value_emits_eq() {
        let w = build_chroma_where(&json!({"and":[{"color":{"match":{"value":"red"}}}]})).unwrap();
        assert_eq!(w, json!({"color":{"$eq":"red"}}));
    }

    #[test]
    fn match_any_emits_in() {
        let w = build_chroma_where(&json!({"and":[{"size":{"match":{"any":[1,2,3]}}}]})).unwrap();
        assert_eq!(w, json!({"size":{"$in":[1,2,3]}}));
    }

    #[test]
    fn two_sided_range_ands_the_bounds() {
        let w = build_chroma_where(&json!({"and":[{"ts":{"range":{"gte":10,"lt":20}}}]})).unwrap();
        // order: lt, gt, lte, gte -> [lt, gte]
        assert_eq!(w, json!({"$and":[{"ts":{"$lt":20}},{"ts":{"$gte":10}}]}));
    }

    #[test]
    fn datetime_range_bound_becomes_epoch() {
        // 1970-01-01T00:00:10Z -> 10 epoch seconds
        let w =
            build_chroma_where(&json!({"and":[{"ts":{"range":{"gte":"1970-01-01T00:00:10Z"}}}]}))
                .unwrap();
        assert_eq!(w, json!({"ts":{"$gte":10}}));
    }

    #[test]
    fn nested_or_of_and_groups_nests_natively() {
        let cond = json!({"or":[
            {"and":[{"color":{"match":{"value":"red"}}},{"size":{"range":{"gte":50}}}]},
            {"and":[{"color":{"match":{"value":"blue"}}},{"size":{"range":{"lt":10}}}]}
        ]});
        let w = build_chroma_where(&cond).unwrap();
        assert_eq!(
            w,
            json!({"$or":[
                {"$and":[{"color":{"$eq":"red"}},{"size":{"$gte":50}}]},
                {"$and":[{"color":{"$eq":"blue"}},{"size":{"$lt":10}}]}
            ]})
        );
    }

    /// Geo is refused, and that is the FINAL answer for Chroma, not a TODO
    /// (issue #223). Chroma's `where` DSL is a closed enum of `field OP literal`
    /// comparisons — `$eq $ne $gt $gte $lt $lte $in $nin $and $or` — with no
    /// geospatial operator, no attribute-vs-attribute comparison and no
    /// arithmetic, and its metadata values are scalars only. That rules out
    /// every EXACT encoding: a spherical cap is not an axis-aligned box in any
    /// query-independent coordinate system, so no conjunction of ranges can
    /// carve it out, and the `x*qx + y*qy + z*qz >= cos(r/R)` form VectorSets
    /// uses needs cross-field arithmetic Chroma does not have. A bounding box
    /// would admit points up to √2·r away — a widening, i.e. exactly the
    /// silently-wrong recall issue #219 exists to stop — so `None` (→ a hard
    /// error at `resolve_all`) is the honest answer.
    ///
    /// A text match is separately NOT a metadata `where` leaf: it is routed to
    /// `where_document` (see the next test).
    #[test]
    fn geo_is_refused_and_text_leaves_the_metadata_where() {
        assert!(build_chroma_leaf("loc", "geo", &json!({"lat":1,"lon":2,"radius":5})).is_none());
        assert!(build_chroma_leaf("body", "match", &json!({"text":"quick"})).is_none());
    }

    #[test]
    fn fulltext_routes_to_where_document_contains() {
        let cond = json!({"and":[{"body":{"match":{"text":"quick"}}}]});
        // Metadata where is empty (the text leaf is not a metadata condition)...
        assert!(build_chroma_where(&cond).is_none());
        // ...and the full-text clause becomes a where_document $contains.
        assert_eq!(
            build_chroma_where_document(&cond).unwrap(),
            json!({"$contains":"quick"})
        );
    }

    #[test]
    fn where_document_none_without_text() {
        let cond = json!({"and":[{"color":{"match":{"value":"red"}}}]});
        assert!(build_chroma_where_document(&cond).is_none());
    }

    #[test]
    fn empty_conditions_are_none() {
        assert!(build_chroma_where(&json!({})).is_none());
    }
}
