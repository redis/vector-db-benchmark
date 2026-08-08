//! Turbopuffer engine implementation.
//!
//! Uses the `turbopuffer-client` crate (async) with a tokio runtime.
//! Turbopuffer is a cloud-only vector database. Requires an API key
//! set via the TURBOPUFFER_API_KEY environment variable.
//!
//! Supports filtering on metadata attributes using Turbopuffer's
//! filter operators (Eq, In, Gt, Lt, etc.) and logical combinators.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use turbopuffer_client::Client;

use super::geo;
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::MetadataItem;

/// One upload batch: ids, their vectors, and optional per-item metadata.
type UploadBatch = (Vec<i64>, Vec<Vec<f32>>, Vec<Option<MetadataItem>>);

const DEFAULT_NAMESPACE: &str = "benchmark";

pub struct TurbopufferEngine {
    name: String,
    namespace: String,
    batch_size: usize,
    parallel: usize,
    search_params: Vec<SearchParams>,
    distance_metric: String,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared client used by the non-search paths (configure/upload/delete),
    /// which all run on `rt`. The parallel search path deliberately does NOT
    /// use this client: it builds an independent client per worker so each
    /// worker's tokio runtime owns its own reqwest connection pool.
    client: Arc<Client>,
    /// API key kept so each search worker can construct its own client.
    api_key: String,
}

impl TurbopufferEngine {
    pub fn new(engine_config: &EngineConfig, _host: &str) -> Result<Self, String> {
        let api_key = std::env::var("TURBOPUFFER_API_KEY").map_err(|_| {
            "TURBOPUFFER_API_KEY environment variable is required for turbopuffer engine"
                .to_string()
        })?;

        let namespace = std::env::var("TURBOPUFFER_NAMESPACE")
            .unwrap_or_else(|_| DEFAULT_NAMESPACE.to_string());

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
            .unwrap_or(1000) as usize;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = Arc::new(Client::new(&api_key));

        Ok(Self {
            name: engine_config.name.clone(),
            namespace,
            batch_size,
            parallel,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            distance_metric: "cosine_distance".to_string(),
            rt,
            client,
            api_key,
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

    /// Upload a batch of vectors with metadata
    fn upload_batch(
        client: &Client,
        namespace: &str,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
        rt: &tokio::runtime::Runtime,
    ) -> Result<(), String> {
        let ids_json: Vec<serde_json::Value> = ids.iter().map(|id| serde_json::json!(id)).collect();
        let vecs_json: Vec<Vec<f64>> = vectors
            .iter()
            .map(|v| v.iter().map(|f| *f as f64).collect())
            .collect();

        let mut body = serde_json::json!({
            "ids": ids_json,
            "vectors": vecs_json,
        });

        // Build attributes from metadata
        if metadata.iter().any(|m| m.is_some()) {
            let mut attributes: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
            // Collect all unique field names from all metadata items
            let mut field_names: Vec<String> = Vec::new();
            for item in metadata.iter().flatten() {
                for (key, _) in &item.fields {
                    if !field_names.contains(key) {
                        field_names.push(key.clone());
                    }
                }
            }

            for field_name in &field_names {
                let values: Vec<serde_json::Value> = metadata
                    .iter()
                    .map(|m| {
                        m.as_ref()
                            .and_then(|item| {
                                item.fields
                                    .iter()
                                    .find(|(k, _)| k == field_name)
                                    .map(|(_, v)| metadata_value_to_json(v))
                            })
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect();
                attributes.insert(field_name.clone(), serde_json::Value::Array(values));
            }

            body["attributes"] = serde_json::Value::Object(attributes);
        }

        let ns = client.namespace(namespace);
        rt.block_on(async { ns.upsert(&body).await })
            .map_err(|e| format!("Turbopuffer upsert failed: {}", e))?;

        Ok(())
    }
}

/// Convert a MetadataValue to a JSON value for Turbopuffer attributes
fn metadata_value_to_json(val: &MetadataValue) -> serde_json::Value {
    match val {
        MetadataValue::String(s) => {
            // Try to parse as number for numeric attributes
            if let Ok(n) = s.parse::<i64>() {
                serde_json::json!(n)
            } else if let Ok(f) = s.parse::<f64>() {
                serde_json::json!(f)
            } else {
                serde_json::json!(s)
            }
        }
        MetadataValue::Int(n) => serde_json::json!(*n),
        MetadataValue::Float(f) => serde_json::json!(*f),
        MetadataValue::Labels(labels) => serde_json::json!(labels),
        MetadataValue::Geo { lon, lat } => {
            serde_json::json!({"lon": lon, "lat": lat})
        }
    }
}

use vector_db_benchmark::readers::metadata::MetadataValue;

/// Convert benchmark distance metric to Turbopuffer distance_metric name
fn to_turbopuffer_metric(distance: &str) -> &str {
    match distance {
        "cosine" | "angular" => "cosine_distance",
        "l2" | "euclidean" => "euclidean_squared",
        "dot" | "ip" => "cosine_distance", // turbopuffer doesn't have dot product, cosine is closest
        _ => "cosine_distance",
    }
}

/// Parse benchmark condition JSON into Turbopuffer filter format.
///
/// Benchmark conditions look like:
///   { "and": [{"field_name": {"match": {"value": "..."}}}] }
/// or:
///   { "and": [{"field_name": {"range": {"gt": 5}}}] }
///
/// Turbopuffer filters look like:
///   ["And", [["field_name", "Eq", value], ...]]
pub(crate) fn parse_turbopuffer_filter(
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

    // Handle "and" / "or" top-level keys
    for (key, value) in obj {
        match key.as_str() {
            "and" => {
                let clauses = value.as_array()?;
                let mut tpf_clauses = Vec::new();
                for clause in clauses {
                    if let Some(f) = parse_single_condition(clause) {
                        tpf_clauses.push(f);
                    }
                }
                if tpf_clauses.is_empty() {
                    return None;
                }
                return Some(serde_json::json!(["And", tpf_clauses]));
            }
            "or" => {
                let clauses = value.as_array()?;
                let mut tpf_clauses = Vec::new();
                for clause in clauses {
                    if let Some(f) = parse_single_condition(clause) {
                        tpf_clauses.push(f);
                    }
                }
                if tpf_clauses.is_empty() {
                    return None;
                }
                return Some(serde_json::json!(["Or", tpf_clauses]));
            }
            _ => {
                // Single condition at top level
                if let Some(f) = parse_single_condition(conditions) {
                    return Some(f);
                }
            }
        }
    }
    None
}

/// Parse a single condition like {"field_name": {"match": {"value": "..."}} }
fn parse_single_condition(condition: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = condition.as_object()?;
    for (field_name, constraint) in obj {
        let constraint_obj = constraint.as_object()?;
        for (op, operand) in constraint_obj {
            match op.as_str() {
                "match" => {
                    // match_any (IN-list): {"match": {"any": [...]}} =>
                    // ["field_name", "In", [...]]. Checked before {"value": ...}
                    // (the canonical IN-list shape has no "value" key, so without
                    // this arm the clause was silently dropped and the query ran
                    // UNFILTERED). Mirrors qdrant's OR-of-values / redis match_any.
                    if let Some(any) = operand.get("any").and_then(|v| v.as_array()) {
                        // Empty IN-set must match NOTHING. Turbopuffer has no bool
                        // literal, so emit a provably-unsatisfiable contradiction
                        // (Eq AND NotEq the same sentinel) rather than ["In",[]],
                        // whose empty-set semantics on the cloud service are
                        // unverified (could run unfiltered or 400).
                        if any.is_empty() {
                            let sentinel = "__match_any_never_match__";
                            return Some(serde_json::json!([
                                "And",
                                [
                                    [field_name, "Eq", sentinel],
                                    [field_name, "NotEq", sentinel]
                                ]
                            ]));
                        }
                        // follow-up (cloud-only, untestable locally — NOT
                        // implemented): scalar `In` here tests whole-value set
                        // membership and does NOT do array CONTAINS-ANY for a
                        // multi-valued attribute (e.g. `labels` stored as an
                        // array), so it can't match a doc whose attribute is a
                        // list. It also does NOT reconcile ELEMENT TYPES with the
                        // upload-side string→number coercion in
                        // `metadata_value_to_json` (a numeric-looking string is
                        // stored as int/float, so a string `"1"` in the IN-list
                        // would not equal the stored int `1`). The common
                        // keyword/int IN-list is covered; both gaps are
                        // dataset-dependent and require a live cloud namespace to
                        // verify.
                        return Some(serde_json::json!([field_name, "In", any]));
                    }
                    // {"match": {"value": x}} => ["field_name", "Eq", x]
                    if let Some(val) = operand.get("value") {
                        if let Some(arr) = val.as_array() {
                            // Multiple values => In
                            return Some(serde_json::json!([field_name, "In", arr]));
                        }
                        return Some(serde_json::json!([field_name, "Eq", val]));
                    }
                }
                "range" => {
                    let range_obj = operand.as_object()?;
                    let mut clauses = Vec::new();
                    if let Some(gt) = range_obj.get("gt") {
                        clauses.push(serde_json::json!([field_name, "Gt", gt]));
                    }
                    if let Some(gte) = range_obj.get("gte") {
                        clauses.push(serde_json::json!([field_name, "Gte", gte]));
                    }
                    if let Some(lt) = range_obj.get("lt") {
                        clauses.push(serde_json::json!([field_name, "Lt", lt]));
                    }
                    if let Some(lte) = range_obj.get("lte") {
                        clauses.push(serde_json::json!([field_name, "Lte", lte]));
                    }
                    if clauses.len() == 1 {
                        return Some(clauses.into_iter().next().unwrap());
                    } else if clauses.len() > 1 {
                        return Some(serde_json::json!(["And", clauses]));
                    }
                }
                "geo" => {
                    // Turbopuffer doesn't support geo filters natively; skip
                    return None;
                }
                _ => {}
            }
        }
    }
    None
}

impl Engine for TurbopufferEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        // Set distance metric based on dataset
        self.distance_metric =
            to_turbopuffer_metric(dataset.config.distance.as_deref().unwrap_or("cosine"))
                .to_string();

        println!(
            "Turbopuffer namespace: {}, metric: {}",
            self.namespace, self.distance_metric
        );

        // Delete existing namespace if it exists
        let ns = self.client.namespace(&self.namespace);
        let _ = self.rt.block_on(async { ns.delete().await });

        println!("Namespace configured (will be created on first upsert)");
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let dataset_path = dataset.get_path()?;
        let normalize = dataset.needs_normalization();
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
            self.parallel, self.batch_size
        );
        let upload_start = Instant::now();

        let total = vectors.len();
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}"
            )
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-")
        );

        if self.parallel <= 1 {
            // Sequential upload
            for start in (0..total).step_by(self.batch_size) {
                let end = (start + self.batch_size).min(total);
                let batch_ids = &ids[start..end];
                let batch_vecs = &vectors[start..end];
                let batch_meta = &metadata[start..end];

                Self::upload_batch(
                    &self.client,
                    &self.namespace,
                    batch_ids,
                    batch_vecs,
                    batch_meta,
                    &self.rt,
                )?;
                pb.set_position(end as u64);
            }
        } else {
            // Parallel upload
            let client = Arc::clone(&self.client);
            let namespace = self.namespace.clone();
            let batch_size = self.batch_size;
            let counter = Arc::new(AtomicUsize::new(0));
            let errors = Arc::new(Mutex::new(Vec::new()));

            let batches: Vec<UploadBatch> = (0..total)
                .step_by(batch_size)
                .map(|start| {
                    let end = (start + batch_size).min(total);
                    (
                        ids[start..end].to_vec(),
                        vectors[start..end].to_vec(),
                        metadata[start..end].to_vec(),
                    )
                })
                .collect();

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.parallel)
                .build()
                .map_err(|e| format!("Failed to create thread pool: {}", e))?;

            pool.scope(|s| {
                for (batch_ids, batch_vecs, batch_meta) in &batches {
                    let client = Arc::clone(&client);
                    let namespace = namespace.clone();
                    let counter = Arc::clone(&counter);
                    let errors = Arc::clone(&errors);
                    let pb = pb.clone();

                    s.spawn(move |_| {
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .unwrap();
                        if let Err(e) = Self::upload_batch(
                            &client, &namespace, batch_ids, batch_vecs, batch_meta, &rt,
                        ) {
                            errors.lock().unwrap().push(e);
                            return;
                        }
                        let done =
                            counter.fetch_add(batch_ids.len(), Ordering::Relaxed) + batch_ids.len();
                        pb.set_position(done as u64);
                    });
                }
            });

            let errs = errors.lock().unwrap();
            if !errs.is_empty() {
                return Err(format!("Upload errors: {}", errs.join("; ")));
            }
        }

        pb.finish_with_message("done");

        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            total as f64 / upload_time
        );

        let total_time = read_time + upload_time;
        println!("Total time: {:.3}s", total_time);

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: total,
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
        // Re-derive the distance metric here as well as in configure(): on the
        // `--skip-upload` path configure() never runs (#238) and this field would
        // keep its `new()` default of "cosine_distance" — which does not fail, it
        // just measures an l2 or dot dataset under the WRONG metric and writes a
        // plausible file. Pure function of the dataset, so recomputing is
        // idempotent.
        self.distance_metric =
            to_turbopuffer_metric(dataset.config.distance.as_deref().unwrap_or("cosine"))
                .to_string();

        // Clamp to >=1: `parallel` is Option<i64> from config, so 0 would spawn
        // no workers (AtomicUsize never drained → spurious "no searches" error)
        // and a negative value would wrap to usize::MAX via `as usize` and spawn
        // unbounded threads. Both degrade to a single worker.
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<QueryFilter<serde_json::Value>> =
            conditions.resolve_all("Turbopuffer", parse_turbopuffer_filter)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));

        println!(
            "\tRunning {} queries (top={}, parallel={})...",
            HumanCount(num_to_run as u64),
            top,
            parallel
        );

        // Workers index `neighbors[idx]` / `parsed_filters[idx]` for every idx in
        // `0..num_to_run`. Guard the invariant up front (returning Err) rather than
        // letting an out-of-bounds panic inside a worker unwind thread::scope and
        // discard every other worker's collected samples via join().unwrap().
        if neighbors.len() < num_to_run {
            return Err(format!(
                "dataset misaligned: {} neighbor lists for {} queries to run",
                neighbors.len(),
                num_to_run
            ));
        }
        if parsed_filters.len() < num_to_run {
            return Err(format!(
                "dataset misaligned: {} parsed filters for {} queries to run",
                parsed_filters.len(),
                num_to_run
            ));
        }

        // Persistent-worker harness (mirrors qdrant/elasticsearch search()):
        // `workers` persistent workers, each building ONE tokio runtime and its
        // OWN client, pulling query indices from a shared AtomicUsize. This
        // replaces the old O(num_to_run) per-query runtime construction and the
        // per-query global Mutex<Vec> locking. Each worker accumulates thread-local
        // sample buffers (only successful queries contribute, so a failure is never
        // scored as a 0-recall / 0-latency sample) which are merged after join — no
        // per-query mutex in the timed loop. Order is completion order across
        // workers, which is fine for aggregates.
        //
        // Cap the worker count at num_to_run so a `parallel >> num_to_run` misconfig
        // does not build idle runtimes (runtime creation inside the timed window).
        let workers = parallel.min(num_to_run.max(1));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let total_start = Instant::now();

        let api_key = self.api_key.as_str();
        let namespace = self.namespace.as_str();
        let distance_metric = self.distance_metric.as_str();
        let queries = &queries;
        let neighbors = &neighbors;
        let parsed_filters = &parsed_filters;

        let mut latencies: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precisions: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recalls: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrrs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcgs: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    // One runtime per persistent worker, reused across its queries.
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("Turbopuffer worker runtime build failed: {}", e);
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Independent client per worker: a shared (or cloned) reqwest
                    // Client shares one connection pool, which would tie pooled
                    // keep-alive connections to whichever runtime first dialed them
                    // (cross-runtime scheduling jitter in the measured per-query
                    // latency) and cause sporadic "connection closed" failures when
                    // a finished worker drops its runtime mid-flight for another.
                    // Constructing inside block_on binds the pool to this runtime.
                    let client = rt.block_on(async { Client::new(api_key) });

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let query = &queries[idx];
                        let filter = parsed_filters[idx].as_ref();
                        let start = Instant::now();
                        let results = single_query(
                            &client,
                            namespace,
                            query,
                            top,
                            distance_metric,
                            filter,
                            &rt,
                        );
                        let elapsed = start.elapsed().as_secs_f64();

                        match results {
                            Ok(result_ids) => {
                                let m = crate::metrics::compute_metrics(
                                    &result_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                t.push(elapsed);
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
                latencies.extend(t);
                precisions.extend(p);
                recalls.extend(r);
                mrrs.extend(mr);
                ndcgs.extend(nd);
            }
        });

        pb.finish_and_clear();
        let total_time = total_start.elapsed().as_secs_f64();

        // Aggregate over successful queries only. Route through the canonical
        // compute_search_stats so turbopuffer uses the SAME percentile_linear +
        // population-std + rps definition (successes / wall-clock) as every other
        // engine, instead of a hand-rolled copy that could silently diverge if the
        // canonical path ever changes.
        let succeeded = latencies.len();
        if succeeded == 0 {
            return Err("No searches completed (all queries failed)".to_string());
        }
        let failed = num_to_run - succeeded;
        if failed > 0 {
            eprintln!("WARNING: {} of {} queries failed", failed, num_to_run);
        }

        crate::engine::compute_search_stats(
            &latencies,
            &precisions,
            &recalls,
            &mrrs,
            &ndcgs,
            total_time,
            top,
            parallel,
            num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        let ns = self.client.namespace(&self.namespace);
        self.rt
            .block_on(async { ns.delete().await })
            .map_err(|e| format!("Turbopuffer delete failed: {}", e))?;
        println!("Namespace '{}' deleted", self.namespace);
        Ok(())
    }
}

/// Build the Turbopuffer query request body. Pure (no network) so it can be
/// unit-tested. `filters` is only set when a filter is present.
fn build_query_body(
    query_vec: Vec<f64>,
    distance_metric: &str,
    top_k: usize,
    filter: Option<&serde_json::Value>,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "vector": query_vec,
        "distance_metric": distance_metric,
        "top_k": top_k,
    });

    if let Some(f) = filter {
        body["filters"] = f.clone();
    }

    body
}

/// Execute a single query against Turbopuffer
fn single_query(
    client: &Client,
    namespace: &str,
    query: &[f32],
    top_k: usize,
    distance_metric: &str,
    filter: Option<&serde_json::Value>,
    rt: &tokio::runtime::Runtime,
) -> Result<Vec<i64>, String> {
    let query_vec: Vec<f64> = query.iter().map(|f| *f as f64).collect();

    let body = build_query_body(query_vec, distance_metric, top_k, filter);

    let ns = client.namespace(namespace);
    let response = rt
        .block_on(async { ns.query(&body).await })
        .map_err(|e| format!("Turbopuffer query failed: {}", e))?;

    let ids: Vec<i64> = response
        .vectors
        .iter()
        .map(|v| match &v.id {
            turbopuffer_client::response::Id::Int(i) => *i as i64,
            turbopuffer_client::response::Id::String(s) => s.parse::<i64>().unwrap_or(-1),
        })
        .collect();

    Ok(ids)
}

#[cfg(test)]
mod filter_tests {
    use super::{build_query_body, parse_turbopuffer_filter};
    use serde_json::json;

    // NOTE: Turbopuffer is cloud-only (requires TURBOPUFFER_API_KEY and a live
    // service), so these are UNIT tests of the pure filter/body builders only —
    // there is intentionally no integration test.

    // ── match_any (the fixed bug): {"match":{"any":[...]}} → [field,"In",[...]] ──
    // These FAIL before the fix (the "any" shape has no "value" key, so the
    // clause was dropped and the query ran UNFILTERED).

    #[test]
    fn match_any_keyword_list_emits_in() {
        let cond = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        let f = parse_turbopuffer_filter(&cond).expect("match_any must produce a filter");
        assert_eq!(f, json!(["And", [["color", "In", ["red", "blue"]]]]));
    }

    #[test]
    fn match_any_int_list_emits_in() {
        let cond = json!({"and": [{"size": {"match": {"any": [1, 2, 3]}}}]});
        let f = parse_turbopuffer_filter(&cond).expect("match_any must produce a filter");
        assert_eq!(f, json!(["And", [["size", "In", [1, 2, 3]]]]));
    }

    #[test]
    fn match_any_empty_list_emits_never_match_contradiction() {
        // Empty any:[] must match NOTHING, not run unfiltered. Emitted as an
        // Eq-AND-NotEq contradiction wrapped by the top-level "And".
        let cond = json!({"and": [{"color": {"match": {"any": []}}}]});
        let f = parse_turbopuffer_filter(&cond).expect("empty any must not be dropped");
        assert_eq!(
            f,
            json!([
                "And",
                [[
                    "And",
                    [
                        ["color", "Eq", "__match_any_never_match__"],
                        ["color", "NotEq", "__match_any_never_match__"]
                    ]
                ]]
            ])
        );
    }

    #[test]
    fn or_wraps_with_or() {
        let cond = json!({"or": [{"color": {"match": {"value": "red"}}}]});
        let f = parse_turbopuffer_filter(&cond).unwrap();
        assert_eq!(f, json!(["Or", [["color", "Eq", "red"]]]));
    }

    #[test]
    fn and_wraps_with_and() {
        let cond = json!({"and": [{"color": {"match": {"value": "red"}}}]});
        let f = parse_turbopuffer_filter(&cond).unwrap();
        assert_eq!(f, json!(["And", [["color", "Eq", "red"]]]));
    }

    #[test]
    fn single_top_level_condition_passthrough() {
        // A bare `{field:{match:{value}}}` with no and/or wrapper.
        let cond = json!({"color": {"match": {"value": "red"}}});
        let f = parse_turbopuffer_filter(&cond).unwrap();
        assert_eq!(f, json!(["color", "Eq", "red"]));
    }

    #[test]
    fn range_single_and_multi() {
        let single = json!({"and": [{"age": {"range": {"gt": 5}}}]});
        assert_eq!(
            parse_turbopuffer_filter(&single).unwrap(),
            json!(["And", [["age", "Gt", 5]]])
        );

        let multi = json!({"and": [{"age": {"range": {"gte": 5, "lt": 10}}}]});
        // Inner range with >1 bound is itself an ["And", [...]].
        assert_eq!(
            parse_turbopuffer_filter(&multi).unwrap(),
            json!(["And", [["And", [["age", "Gte", 5], ["age", "Lt", 10]]]]])
        );
    }

    /// Geo is refused, and that is the FINAL answer for Turbopuffer, not a TODO
    /// (issue #223). Its filter grammar is a closed set of `[attr, Op, literal]`
    /// triples — `Eq NotEq In NotIn Lt Lte Gt Gte Any* Contains* Glob* Regex
    /// ContainsAllTokens ContainsAnyToken ContainsTokenSequence Fuzzy And Or
    /// Not` — with no geo operator, no attribute on the right-hand side and no
    /// arithmetic; its attribute type list (string/int/uint/float/uuid/datetime/
    /// bool and their array forms) has no geo or struct type either. Turbopuffer
    /// DOES have arithmetic, but only in `rank_by`, which its documentation
    /// states never affects matching — so it cannot exclude a document. That
    /// leaves no exact encoding (see the Chroma note for why a bounding box is
    /// not an acceptable substitute), and `None` → a hard error at `resolve_all`
    /// is the honest answer.
    #[test]
    fn geo_is_refused_because_the_filter_grammar_has_no_geo_and_no_arithmetic() {
        let cond = json!({"and": [{"loc": {"geo": {"lat": 1.0, "lon": 2.0, "radius": 10.0}}}]});
        // geo yields no clause; the only clause dropping leaves an empty And → None.
        assert!(parse_turbopuffer_filter(&cond).is_none());
        // ...and a SIBLING leaf in another entry does NOT die with it on its
        // own. `parse_single_condition`'s `"geo" => return None` exits only that
        // ENTRY, and the `and` loop then SKIPS the `None` — so without the
        // guard above this renders a keyword-only filter that looks real and
        // constrains LESS than the ground truth does (verified by deleting the
        // guard: `Some(["And", [["color", "Eq", "red"]]])`). The `None` below
        // comes from `conditions_mention_geo`, not from the geo arm.
        let mixed = json!({"and": [
            {"loc": {"geo": {"lat": 1.0, "lon": 2.0, "radius": 10.0}}},
            {"color": {"match": {"value": "red"}}}
        ]});
        assert!(parse_turbopuffer_filter(&mixed).is_none());
    }

    #[test]
    fn empty_conditions_none() {
        assert!(parse_turbopuffer_filter(&json!({})).is_none());
        assert!(parse_turbopuffer_filter(&json!({"and": []})).is_none());
        assert!(parse_turbopuffer_filter(&json!({"or": []})).is_none());
    }

    // ── request-body builder ──
    #[test]
    fn body_without_filter_has_no_filters_key() {
        let body = build_query_body(vec![0.1, 0.2], "cosine_distance", 10, None);
        assert_eq!(body["distance_metric"], "cosine_distance");
        assert_eq!(body["top_k"], 10);
        assert!(body.get("filters").is_none());
    }

    #[test]
    fn body_with_filter_sets_filters_key() {
        let filter = json!(["And", [["color", "In", ["red", "blue"]]]]);
        let body = build_query_body(vec![0.1], "euclidean_squared", 5, Some(&filter));
        assert_eq!(body["filters"], filter);
        assert_eq!(body["top_k"], 5);
    }
}
