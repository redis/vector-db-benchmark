//! MongoDB Atlas Vector Search engine implementation.
//!
//! Uses the official `mongodb` crate with sync feature.
//! Supports Atlas Vector Search with HNSW index via `$vectorSearch` aggregation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use mongodb::bson::{doc, Document};
use mongodb::sync::Client;

use rand::{seq::SliceRandom, SeedableRng};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UpdateSearchRatio, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_DB: &str = "bench";
const DEFAULT_COLLECTION: &str = "vectors";
const DEFAULT_INDEX_NAME: &str = "vector_index";

#[derive(Clone)]
struct MongoConfig {
    batch_size: usize,
    parallel: usize,
    num_candidates_factor: i64,
    skip_vector_index: bool,
    /// `collection_params.hnsw_config.M` -> `hnswOptions.maxEdges`.
    hnsw_m: Option<i64>,
    /// `collection_params.hnsw_config.EF_CONSTRUCTION` ->
    /// `hnswOptions.numEdgeCandidates`.
    hnsw_ef_construction: Option<i64>,
}

pub struct MongoDBEngine {
    name: String,
    db_name: String,
    collection_name: String,
    index_name: String,
    config: MongoConfig,
    search_params: Vec<SearchParams>,
    /// MongoDB connection URI
    uri: String,
    /// Shared MongoDB client (connection pool)
    client: Client,
    /// Dataset schema field types (field name -> "int" | "float" | "keyword" |
    /// "text" | "uuid" | "bool" | ...). Drives native-BSON storage of numeric
    /// payload fields at ingest so numeric filters (exact/`$in`/range) match
    /// (mirrors pgvector storing numerics in BIGINT/DOUBLE columns). Populated
    /// from the dataset schema in `configure`/`upload`/`search_mixed`.
    schema_types: HashMap<String, String>,
}

impl MongoDBEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("MONGODB_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(27017);

        let db_name = std::env::var("MONGODB_DB").unwrap_or_else(|_| DEFAULT_DB.to_string());
        let collection_name =
            std::env::var("MONGODB_COLLECTION").unwrap_or_else(|_| DEFAULT_COLLECTION.to_string());
        let index_name =
            std::env::var("MONGODB_INDEX_NAME").unwrap_or_else(|_| DEFAULT_INDEX_NAME.to_string());

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
            .unwrap_or(500) as usize;

        let num_candidates_factor = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("num_candidates_factor"))
            .and_then(|v| v.as_i64())
            .unwrap_or(10);

        // HNSW graph-construction knobs come from the TYPED
        // collection_params.hnsw_config field (serde captures "M"/"m" and
        // "EF_CONSTRUCTION"/"ef_construct"/"ef_construction" there via aliases,
        // so they never land in the flattened `extra` map).
        let typed_hnsw = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref());
        let hnsw_m = typed_hnsw.and_then(|h| h.m);
        let hnsw_ef_construction = typed_hnsw.and_then(|h| h.ef_construction);

        let uri = build_uri(host, port);

        let client = Client::with_uri_str(&uri)
            .map_err(|e| format!("Failed to create MongoDB client: {}", e))?;

        Ok(Self {
            name: engine_config.name.clone(),
            db_name,
            collection_name,
            index_name,
            config: MongoConfig {
                batch_size,
                parallel,
                num_candidates_factor,
                skip_vector_index: engine_config.skip_vector_index,
                hnsw_m,
                hnsw_ef_construction,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            uri,
            client,
            schema_types: HashMap::new(),
        })
    }

    /// Extract the field-type map from the dataset schema (`{field: "int"|...}`).
    /// Stored so ingest can pick a native BSON type per field.
    fn load_schema_types(&mut self, dataset: &Dataset) {
        self.schema_types = dataset
            .config
            .schema
            .as_ref()
            .and_then(|s| s.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(field, ftype)| {
                        ftype.as_str().map(|t| (field.clone(), t.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
    }

    /// Filter-only search: run collection.find(filter).limit(top) with no vector search.
    fn search_filter_only(
        &self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (_queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<Document>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_mongo_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);

        let runnable_indices: Vec<usize> = (0..parsed_filters.len())
            .filter(|&i| parsed_filters[i].is_some())
            .collect();

        if runnable_indices.is_empty() {
            return Err("No queries with filter conditions for filter-only search".to_string());
        }

        // Round-robin: if num_queries > available queries, cycle through them
        let num_to_run = if num_queries > 0 {
            num_queries as usize
        } else {
            runnable_indices.len()
        };

        // Each worker accumulates latencies into a thread-local buffer and returns
        // it on join; the main thread concatenates. This keeps the timed hot loop
        // free of the per-query cross-thread Mutex<Vec> push that serialized
        // workers at high parallelism (matching the main search() path). The work
        // counter uses Relaxed (only its own monotonicity matters). Progress is
        // advanced in batches so the atomic isn't contended once per query.
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let parsed_filters = &parsed_filters;
                let runnable_indices = &runnable_indices;
                let neighbors = &neighbors;
                let errors = Arc::clone(&errors);
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    // Thread-local sample buffer — no cross-thread lock per query.
                    let mut t: Vec<f64> = Vec::new();
                    let mut local_errs: Vec<String> = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(_) => return t,
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    loop {
                        let seq = query_idx.fetch_add(1, Ordering::Relaxed);
                        if seq >= num_to_run {
                            break;
                        }
                        let idx = runnable_indices[seq % runnable_indices.len()];

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let filter = parsed_filters[idx].as_ref().unwrap();

                        let query_start = Instant::now();
                        let result = filter_only_find(&coll, filter, top);
                        let query_time = query_start.elapsed().as_secs_f64();

                        // Record a latency sample only for successful queries, so a
                        // failed $vectorSearch/find is counted as a failure (num_to_run
                        // minus successes) rather than folded into RPS/percentiles.
                        // MongoDB has no check_commandstats backstop, so this is the
                        // only place failures are surfaced.
                        match result {
                            Ok(_) => t.push(query_time),
                            Err(e) => {
                                if local_errs.len() < 3 {
                                    local_errs.push(e);
                                }
                            }
                        }
                        pb_pending += 1;
                        if pb_pending >= 256 {
                            pb.inc(pb_pending);
                            pb_pending = 0;
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
                    }
                    if !local_errs.is_empty() {
                        let mut errs = errors.lock().unwrap();
                        for e in local_errs {
                            if errs.len() < 3 {
                                errs.push(e);
                            }
                        }
                    }
                    t
                }));
            }

            for h in handles {
                times.extend(h.join().unwrap());
            }
        });

        {
            let logged_errors = errors.lock().unwrap();
            if !logged_errors.is_empty() {
                for e in logged_errors.iter() {
                    eprintln!("\tFilter-only search error: {}", e);
                }
            }
        }

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        if times.is_empty() {
            return Err("No filter-only searches completed".to_string());
        }

        // Route latency stats through the shared percentile path (linear
        // interpolation) so filter-only is measured on the same footing as the
        // main search(). Filter-only has no precision/recall: signal that with the
        // mean_precision_at_returned == -1 sentinel, an empty precisions vec,
        // and top == 0.
        let mut results = crate::engine::compute_search_stats(
            &times,
            &[],
            &[],
            &[],
            &[],
            total_time,
            0,
            parallel,
            num_to_run,
        )?;
        results.mean_precision_at_returned = -1.0;
        Ok(results)
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

    fn drop_collection(&self) -> Result<(), String> {
        let db = self.client.database(&self.db_name);

        // 1. Drop the search index explicitly and wait for it to disappear.
        //    On Atlas, stale indexes can prevent clean recreation.
        println!("Dropping search index '{}'...", self.index_name);
        let drop_cmd = doc! {
            "dropSearchIndex": &self.collection_name,
            "name": &self.index_name,
        };
        // Ignore errors (e.g. IndexNotFound, collection doesn't exist)
        let _ = db.run_command(drop_cmd).run();

        let deadline = Instant::now() + std::time::Duration::from_secs(120);
        loop {
            let cmd = doc! { "listSearchIndexes": &self.collection_name };
            let index_exists = db.run_command(cmd).run().ok().is_some_and(|result| {
                result
                    .get_document("cursor")
                    .ok()
                    .and_then(|c| c.get_array("firstBatch").ok())
                    .is_some_and(|batch| {
                        batch.iter().any(|idx| {
                            idx.as_document()
                                .and_then(|d| d.get_str("name").ok())
                                .is_some_and(|n| n == self.index_name)
                        })
                    })
            });

            if !index_exists {
                break;
            }
            if Instant::now() > deadline {
                eprintln!(
                    "Warning: search index '{}' still exists after 120s, proceeding anyway",
                    self.index_name
                );
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }

        // 2. Drop the collection and verify it's gone.
        let coll = db.collection::<Document>(&self.collection_name);
        coll.drop()
            .run()
            .map_err(|e| format!("Failed to drop collection: {}", e))?;

        let deadline = Instant::now() + std::time::Duration::from_secs(60);
        loop {
            let names = db.list_collection_names().run().unwrap_or_default();
            if !names.contains(&self.collection_name.to_string()) {
                break;
            }
            if Instant::now() > deadline {
                eprintln!(
                    "Warning: collection '{}' still exists after 60s, proceeding anyway",
                    self.collection_name
                );
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }

        Ok(())
    }

    fn create_vector_index(&self, dataset: &Dataset) -> Result<(), String> {
        let vector_size = dataset.vector_size();
        let distance = dataset.distance();

        let similarity = match distance.to_lowercase().as_str() {
            "l2" | "euclidean" => "euclidean",
            "cosine" | "angular" => "cosine",
            "dot" | "ip" => "dotProduct",
            other => {
                return Err(format!(
                    "Unsupported distance metric for MongoDB: {}",
                    other
                ))
            }
        };

        // Build vector search index definition
        let mut vector_field = doc! {
            "type": "vector",
            "path": "vector",
            "numDimensions": vector_size as i32,
            "similarity": similarity,
        };

        // Forward the HNSW graph-construction knobs. Atlas spells them
        // `hnswOptions.{maxEdges,numEdgeCandidates}` — NOT `m`/`efConstruction`,
        // which the server rejects outright as unrecognized fields. See
        // `build_hnsw_options` for the config mapping.
        if let Some(hnsw_options) =
            build_hnsw_options(self.config.hnsw_m, self.config.hnsw_ef_construction)?
        {
            vector_field.insert("hnswOptions", hnsw_options);
        }

        let mut fields = vec![vector_field];

        // Add filter fields from dataset schema
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, _field_type) in schema_obj {
                    fields.push(doc! {
                        "type": "filter",
                        "path": field_name,
                    });
                }
            }
        }

        let index_def = doc! {
            "name": &self.index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": fields,
            }
        };

        let db = self.client.database(&self.db_name);
        let cmd = doc! {
            "createSearchIndexes": &self.collection_name,
            "indexes": [index_def],
        };

        db.run_command(cmd)
            .run()
            .map_err(|e| format!("Failed to create vector search index: {}", e))?;

        // Wait for index to become ready
        self.wait_for_index_ready()?;

        Ok(())
    }

    fn wait_for_index_ready(&self) -> Result<(), String> {
        println!("Waiting for vector search index to become ready...");
        let db = self.client.database(&self.db_name);
        let deadline = Instant::now() + std::time::Duration::from_secs(120);

        loop {
            let cmd = doc! {
                "listSearchIndexes": &self.collection_name,
            };

            if let Ok(result) = db.run_command(cmd).run() {
                if let Ok(cursor) = result.get_document("cursor") {
                    if let Ok(batch) = cursor.get_array("firstBatch") {
                        for index in batch {
                            if let Some(index_doc) = index.as_document() {
                                let name = index_doc.get_str("name").unwrap_or("");
                                let status = index_doc.get_str("status").unwrap_or("");
                                let queryable = index_doc.get_bool("queryable").unwrap_or(false);
                                // Atlas uses READY, local uses ACTIVE
                                if name == self.index_name
                                    && (status == "READY" || status == "ACTIVE")
                                    && queryable
                                {
                                    println!(
                                        "Vector search index is ready (status={}, queryable=true).",
                                        status
                                    );
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }

            if Instant::now() > deadline {
                return Err(
                    "Vector search index did not become ready within 120 seconds".to_string(),
                );
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    /// Wait for the vector search index to finish indexing all uploaded documents.
    ///
    /// Polls `listSearchIndexes` until the index reports `queryable=true` and
    /// `status` is READY or ACTIVE, then verifies with a probe search using
    /// the first uploaded vector.
    ///
    /// Atlas Vector Search is EVENTUALLY CONSISTENT: the index flips to
    /// `queryable=true` (and `$vectorSearch` starts returning PARTIAL results)
    /// while it is still ingesting the just-uploaded documents. A probe that
    /// stops at the first non-empty result therefore lets the benchmark search
    /// run against a half-built index, which understates recall (the observed
    /// `recall 0.27` flake). To close that race we run the probe as a
    /// full-precision search (`numCandidates >= expected`) and wait until it
    /// surfaces EVERY uploaded document — i.e. the index has fully caught up —
    /// before returning. The required count is bounded by `PROBE_CAP` so a huge
    /// production dataset doesn't wait on an impractically large single query.
    fn wait_for_index_catchup(
        &self,
        expected_count: usize,
        probe_vector: &[f32],
    ) -> Result<(), String> {
        // Upper bound on how many docs we require the probe to surface. Atlas
        // caps both `limit` and `numCandidates` at 10_000, and for large corpora
        // "index reports ready AND has surfaced this many docs" is a sufficient
        // catch-up signal.
        const PROBE_CAP: usize = 10_000;
        let want = expected_count.min(PROBE_CAP);
        // Request all wanted docs, scanning a candidate pool that covers the
        // whole (bounded) corpus so the search is effectively exact — an indexed
        // doc is guaranteed to be surfaced, so `results.len() < want` means the
        // index has not finished ingesting yet.
        let probe_top = want.max(1);
        let probe_candidates = expected_count.saturating_mul(2).clamp(probe_top, PROBE_CAP) as i64;

        println!(
            "Waiting for vector search index to index all {} documents...",
            expected_count
        );
        let db = self.client.database(&self.db_name);
        let coll = db.collection::<Document>(&self.collection_name);
        let start = Instant::now();
        let deadline = start + std::time::Duration::from_secs(600);
        let mut last_print = Instant::now();
        let mut index_ready = false;

        loop {
            if !index_ready {
                let cmd = doc! { "listSearchIndexes": &self.collection_name };
                if let Ok(result) = db.run_command(cmd).run() {
                    if let Ok(cursor) = result.get_document("cursor") {
                        if let Ok(batch) = cursor.get_array("firstBatch") {
                            for index in batch {
                                if let Some(index_doc) = index.as_document() {
                                    let name = index_doc.get_str("name").unwrap_or("");
                                    if name != self.index_name {
                                        continue;
                                    }
                                    let status = index_doc.get_str("status").unwrap_or("");
                                    let queryable =
                                        index_doc.get_bool("queryable").unwrap_or(false);

                                    if (status == "READY" || status == "ACTIVE") && queryable {
                                        index_ready = true;
                                    } else if last_print.elapsed().as_secs() >= 10 {
                                        println!(
                                            "  index building... status={}, queryable={}",
                                            status, queryable
                                        );
                                        last_print = Instant::now();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Once the index reports ready, probe with a full-precision search and
            // wait until it has surfaced EVERY uploaded document — only then has the
            // eventually-consistent index finished ingesting (partial ingestion
            // returns fewer than `want` and understates recall).
            if index_ready {
                match vector_search(
                    &coll,
                    &self.index_name,
                    probe_vector,
                    probe_top,
                    probe_candidates,
                    None,
                ) {
                    Ok(results) if results.len() >= want => {
                        println!(
                            "Index caught up ({} / {} docs searchable) after {:.1}s.",
                            results.len(),
                            expected_count,
                            start.elapsed().as_secs_f64()
                        );
                        return Ok(());
                    }
                    Ok(results) => {
                        if last_print.elapsed().as_secs() >= 10 {
                            println!(
                                "  index ready but still ingesting: {} / {} docs searchable, waiting...",
                                results.len(),
                                want
                            );
                            last_print = Instant::now();
                        }
                    }
                    Err(_) => {
                        if last_print.elapsed().as_secs() >= 10 {
                            println!("  index ready but probe search errored, waiting...");
                            last_print = Instant::now();
                        }
                    }
                }
            }

            if Instant::now() > deadline {
                return Err(
                    "Vector search index did not finish indexing within 600 seconds".to_string(),
                );
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
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
        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();
        let schema_types = &self.schema_types;

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_batches {
                            break;
                        }
                        if error.lock().unwrap().is_some() {
                            break;
                        }

                        let (batch_start, batch_end) = batches[idx];
                        if let Err(e) = insert_batch(
                            &coll,
                            &ids[batch_start..batch_end],
                            &vectors[batch_start..batch_end],
                            &metadata[batch_start..batch_end],
                            schema_types,
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
}

/// Translate the benchmark's engine-neutral `collection_params.hnsw_config`
/// into the `hnswOptions` sub-document MongoDB Vector Search accepts inside a
/// `vectorSearch` index field.
///
/// Name mapping (verified live against `mongodb/mongodb-atlas-local:8.0.17`):
///
/// | benchmark key     | MongoDB key                    | server-enforced bounds |
/// |-------------------|--------------------------------|------------------------|
/// | `M`               | `hnswOptions.maxEdges`         | `[16..64]`             |
/// | `EF_CONSTRUCTION` | `hnswOptions.numEdgeCandidates`| `[100..3200]`          |
///
/// Values are forwarded verbatim — deliberately NOT clamped. The server
/// validates them and fails index creation loudly with
/// `"...hnswOptions.maxEdges" must be within bounds [16..64]`; clamping here
/// would silently benchmark a different index than the config asked for, which
/// is the exact failure mode this function was added to end (issue #216).
///
/// Returns `Ok(None)` when neither knob is configured, so the index definition
/// stays byte-identical to the pre-existing default-HNSW body.
///
/// Out-of-`i32` values are an error rather than a silent `as i32` truncation:
/// `M: 4294967328` would otherwise wrap to `maxEdges: 32` and build happily,
/// which is the same "benchmarked something other than what the config said"
/// failure this function exists to prevent.
fn build_hnsw_options(
    m: Option<i64>,
    ef_construction: Option<i64>,
) -> Result<Option<Document>, String> {
    if m.is_none() && ef_construction.is_none() {
        return Ok(None);
    }
    let to_i32 = |name: &str, v: i64| {
        i32::try_from(v).map_err(|_| {
            format!(
                "collection_params.hnsw_config.{} = {} does not fit in an i32 and \
                 cannot be sent to MongoDB",
                name, v
            )
        })
    };
    let mut opts = Document::new();
    if let Some(m) = m {
        opts.insert("maxEdges", to_i32("M", m)?);
    }
    if let Some(ef_construction) = ef_construction {
        opts.insert(
            "numEdgeCandidates",
            to_i32("EF_CONSTRUCTION", ef_construction)?,
        );
    }
    Ok(Some(opts))
}

fn build_uri(host: &str, port: u16) -> String {
    let user = std::env::var("MONGODB_USER").ok();
    let password = std::env::var("MONGODB_PASSWORD").ok();

    let host_part = if host.starts_with("mongodb") {
        // Already a full URI
        return host.to_string();
    } else {
        host
    };

    match (user, password) {
        (Some(u), Some(p)) => {
            format!(
                "mongodb://{}:{}@{}:{}/?directConnection=true",
                u, p, host_part, port
            )
        }
        _ => format!("mongodb://{}:{}/?directConnection=true", host_part, port),
    }
}

/// Convert a parsed metadata value into the BSON we store for it, honoring the
/// dataset schema field type. Numeric fields (`int`/`float`) are stored as
/// NATIVE BSON numbers (`Int64`/`Double`) rather than strings so that numeric
/// filters — exact match, `$in` (match_any), and range (`$gt`/`$lt`) — actually
/// match, and range comparisons are numeric (not lexicographic). This mirrors
/// pgvector storing numerics in BIGINT/DOUBLE columns. Everything else
/// (keyword/text/uuid/bool) stays a `String`, exactly as before.
///
/// The metadata reader stringifies every JSON scalar (see
/// `readers::metadata`), so a numeric field arrives here as `String("1")`; we
/// parse it back to a number when the schema says the field is numeric. If the
/// value doesn't parse (defensive), we fall back to storing the string.
fn metadata_value_to_bson(
    field: &str,
    value: &vector_db_benchmark::readers::metadata::MetadataValue,
    schema_types: &HashMap<String, String>,
) -> mongodb::bson::Bson {
    use vector_db_benchmark::readers::metadata::MetadataValue;
    match value {
        MetadataValue::String(s) => match schema_types.get(field).map(|t| t.as_str()) {
            Some("int") => s
                .parse::<i64>()
                .map(mongodb::bson::Bson::Int64)
                .unwrap_or_else(|_| mongodb::bson::Bson::String(s.clone())),
            Some("float") => s
                .parse::<f64>()
                .map(mongodb::bson::Bson::Double)
                .unwrap_or_else(|_| mongodb::bson::Bson::String(s.clone())),
            _ => mongodb::bson::Bson::String(s.clone()),
        },
        MetadataValue::Int(n) => mongodb::bson::Bson::Int64(*n),
        MetadataValue::Float(f) => mongodb::bson::Bson::Double(*f),
        MetadataValue::Labels(labels) => {
            let arr: Vec<mongodb::bson::Bson> = labels
                .iter()
                .map(|l| mongodb::bson::Bson::String(l.clone()))
                .collect();
            mongodb::bson::Bson::Array(arr)
        }
        MetadataValue::Geo { lon, lat } => mongodb::bson::Bson::Document(doc! {
            "type": "Point",
            "coordinates": [*lon, *lat],
        }),
    }
}

/// Insert a batch of documents into MongoDB.
fn insert_batch(
    coll: &mongodb::sync::Collection<Document>,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    schema_types: &HashMap<String, String>,
) -> Result<(), String> {
    let docs: Vec<Document> = ids
        .iter()
        .zip(vectors.iter().zip(metadata.iter()))
        .map(|(&id, (vec, meta))| {
            let bson_vec: Vec<mongodb::bson::Bson> = vec
                .iter()
                .map(|&f| mongodb::bson::Bson::Double(f as f64))
                .collect();

            let mut doc = doc! {
                "_id": id,
                "vector": bson_vec,
            };

            if let Some(meta) = meta {
                for (k, v) in &meta.fields {
                    doc.insert(k.clone(), metadata_value_to_bson(k, v, schema_types));
                }
            }

            doc
        })
        .collect();

    coll.insert_many(docs)
        .run()
        .map_err(|e| format!("Insert batch failed: {}", e))?;

    Ok(())
}

/// Update a single document's vector and metadata.
fn update_one_doc(
    coll: &mongodb::sync::Collection<Document>,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
    schema_types: &HashMap<String, String>,
) -> Result<(), String> {
    let bson_vec: Vec<mongodb::bson::Bson> = vector
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let mut set_doc = doc! { "vector": bson_vec };

    if let Some(meta) = metadata {
        for (k, v) in &meta.fields {
            set_doc.insert(k.clone(), metadata_value_to_bson(k, v, schema_types));
        }
    }

    coll.update_one(doc! { "_id": id }, doc! { "$set": set_doc })
        .run()
        .map_err(|e| format!("Update failed for id {}: {}", id, e))?;

    Ok(())
}

/// Execute a filter-only find (no vector search).
fn filter_only_find(
    coll: &mongodb::sync::Collection<Document>,
    filter: &Document,
    top: usize,
) -> Result<usize, String> {
    let cursor = coll
        .find(filter.clone())
        .limit(top as i64)
        .projection(doc! { "_id": 1 })
        .run()
        .map_err(|e| format!("Filter-only find failed: {}", e))?;

    let mut count = 0usize;
    for result in cursor {
        let _ = result.map_err(|e| format!("Failed to read result: {}", e))?;
        count += 1;
    }
    Ok(count)
}

/// Build the `$vectorSearch` aggregation pipeline for one query.
///
/// Done OUTSIDE the per-query timed window (see the search() precompute): turning
/// the query vector into a full `Vec<Bson::Double>` and assembling the two
/// pipeline docs (`$vectorSearch` + `$project`) is client-side CPU work, not
/// server latency. Precomputing it means the timed window wraps only the
/// aggregate RPC round-trip + cursor decode, matching the reference engines
/// (pgvector/qdrant/redis) and the ES/OS/Milvus/Weaviate boundary from #113.
///
/// The returned pipeline is identical to what the previous inline build produced.
fn build_search_pipeline(
    index_name: &str,
    query_vector: &[f32],
    top: usize,
    num_candidates: i64,
    filter: Option<&Document>,
) -> Vec<Document> {
    let bson_vec: Vec<mongodb::bson::Bson> = query_vector
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let mut vs_stage = doc! {
        "index": index_name,
        "path": "vector",
        "queryVector": bson_vec,
        "numCandidates": num_candidates,
        "limit": top as i64,
    };

    if let Some(f) = filter {
        vs_stage.insert("filter", f.clone());
    }

    vec![
        doc! { "$vectorSearch": vs_stage },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ]
}

/// Send a precomputed aggregation pipeline and return the DECODED documents.
///
/// The consistent timed boundary (see qdrant/pgvector/redis and #113) is:
/// pipeline built OUTSIDE the window; aggregate RPC send + cursor read +
/// decode-to-`Document` INSIDE the window (this fn); id/score extraction OUTSIDE
/// (`extract_search_hits`). So the BSON cursor decode is billed as latency
/// exactly like qdrant's protobuf decode and pgvector's row decode.
fn send_search(
    coll: &mongodb::sync::Collection<Document>,
    pipeline: &[Document],
) -> Result<Vec<Document>, String> {
    let cursor = coll
        .aggregate(pipeline.to_vec())
        .run()
        .map_err(|e| format!("Vector search failed: {}", e))?;

    let mut docs = Vec::new();
    for result in cursor {
        let doc = result.map_err(|e| format!("Failed to read result: {}", e))?;
        docs.push(doc);
    }
    Ok(docs)
}

/// Extract the id/score list from already-decoded documents.
///
/// Done AFTER the timed window (`elapsed`), mirroring pgvector/qdrant pulling the
/// final ids out of the decoded response for recall. This is pure struct field
/// access — no I/O — so it must not be billed as query latency.
fn extract_search_hits(docs: &[Document]) -> Vec<(i64, f64)> {
    docs.iter()
        .map(|doc| {
            let id = doc.get_i64("_id").unwrap_or(0);
            let score = doc.get_f64("score").unwrap_or(0.0);
            (id, score)
        })
        .collect()
}

/// Execute a vector search end-to-end (build + send + extract).
///
/// Convenience wrapper used by the untimed index-catchup probe
/// (`wait_for_index_catchup`), where the split boundary is irrelevant. The timed
/// search()/search_mixed() paths deliberately do NOT use this: they precompute
/// pipelines out of the window and call `send_search`/`extract_search_hits`
/// directly so only the RPC + decode is timed.
fn vector_search(
    coll: &mongodb::sync::Collection<Document>,
    index_name: &str,
    query_vector: &[f32],
    top: usize,
    num_candidates: i64,
    filter: Option<&Document>,
) -> Result<Vec<(i64, f64)>, String> {
    let pipeline = build_search_pipeline(index_name, query_vector, top, num_candidates, filter);
    let docs = send_search(coll, &pipeline)?;
    Ok(extract_search_hits(&docs))
}

/// Parse filter conditions into MongoDB query document.
fn parse_mongo_conditions(conditions: &serde_json::Value) -> Option<Document> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }
    build_mongo_group(obj)
}

/// Build one `{and:[...], or:[...]}` group into a MongoDB filter document, using
/// native `$and`/`$or`. Each array entry is either a field leaf or itself a
/// nested group (`{and:...}`/`{or:...}`); `build_mongo_filter_entry` recurses
/// back here for nested groups, so arbitrarily deep boolean trees nest natively
/// (e.g. `(color==red && size>=50) || (color==blue && size<10)` becomes
/// `{$or:[{$and:[...]},{$and:[...]}]}`), rather than being flattened.
fn build_mongo_group(obj: &serde_json::Map<String, serde_json::Value>) -> Option<Document> {
    let mut filter_clauses: Vec<mongodb::bson::Bson> = Vec::new();

    if let Some(and_entries) = obj.get("and").and_then(|v| v.as_array()) {
        for entry in and_entries {
            if let Some(clause) = build_mongo_filter_entry(entry) {
                filter_clauses.push(mongodb::bson::Bson::Document(clause));
            }
        }
    }

    if let Some(or_entries) = obj.get("or").and_then(|v| v.as_array()) {
        let or_clauses: Vec<mongodb::bson::Bson> = or_entries
            .iter()
            .filter_map(build_mongo_filter_entry)
            .map(mongodb::bson::Bson::Document)
            .collect();
        if !or_clauses.is_empty() {
            filter_clauses.push(mongodb::bson::Bson::Document(doc! {
                "$or": or_clauses,
            }));
        }
    }

    if filter_clauses.is_empty() {
        return None;
    }

    if filter_clauses.len() == 1 {
        if let Some(mongodb::bson::Bson::Document(d)) = filter_clauses.first().cloned() {
            return Some(d);
        }
    }

    Some(doc! { "$and": filter_clauses })
}

fn build_mongo_filter_entry(entry: &serde_json::Value) -> Option<Document> {
    let entry_obj = entry.as_object()?;

    // Nested group: an entry that is itself an `{and:[...]}`/`{or:[...]}` node
    // (not a field leaf) is built as its own grouped sub-clause via native
    // `$and`/`$or`, so nested boolean trees nest instead of mis-flattening.
    if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
        return build_mongo_group(entry_obj);
    }

    let mut clauses = Document::new();

    for (field_name, field_filters) in entry_obj {
        let filter_obj = field_filters.as_object()?;
        for (condition_type, criteria) in filter_obj {
            match condition_type.as_str() {
                "match" => {
                    // match_any: field value in a list -> Mongo `$in`, the
                    // OR-of-values semantics that mirror qdrant's
                    // Condition::matches(field, Vec). An empty IN-set matches
                    // NOTHING: `{$in: []}` is a valid never-match, so we never
                    // drop the clause (which, as the sole condition, would leave
                    // no filter and return every doc — the inverse of intent).
                    if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                        let arr: Vec<mongodb::bson::Bson> = any.iter().map(json_to_bson).collect();
                        clauses.insert(field_name.clone(), doc! { "$in": arr });
                    } else if let Some(value) = criteria.get("value") {
                        clauses.insert(field_name.clone(), json_to_bson(value));
                    }
                }
                "range" => {
                    let mut range_doc = Document::new();
                    if let Some(gt) = criteria.get("gt") {
                        if !gt.is_null() {
                            range_doc.insert("$gt", json_to_bson(gt));
                        }
                    }
                    if let Some(lt) = criteria.get("lt") {
                        if !lt.is_null() {
                            range_doc.insert("$lt", json_to_bson(lt));
                        }
                    }
                    if let Some(gte) = criteria.get("gte") {
                        if !gte.is_null() {
                            range_doc.insert("$gte", json_to_bson(gte));
                        }
                    }
                    if let Some(lte) = criteria.get("lte") {
                        if !lte.is_null() {
                            range_doc.insert("$lte", json_to_bson(lte));
                        }
                    }
                    if !range_doc.is_empty() {
                        clauses.insert(field_name.clone(), range_doc);
                    }
                }
                _ => {}
            }
        }
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses)
    }
}

fn json_to_bson(value: &serde_json::Value) -> mongodb::bson::Bson {
    match value {
        serde_json::Value::Null => mongodb::bson::Bson::Null,
        // Bools are STORED as the string "true"/"false" (metadata_value_to_bson —
        // readers::metadata has no Bool variant), so a filter bool must compare as
        // that string. A native Bson::Boolean never equals the stored String and
        // silently matches zero documents (0 recall). json_to_bson is filter-only.
        serde_json::Value::Bool(b) => {
            mongodb::bson::Bson::String(if *b { "true" } else { "false" }.to_string())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                mongodb::bson::Bson::Int64(i)
            } else if let Some(f) = n.as_f64() {
                mongodb::bson::Bson::Double(f)
            } else {
                mongodb::bson::Bson::Null
            }
        }
        serde_json::Value::String(s) => mongodb::bson::Bson::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let bson_arr: Vec<mongodb::bson::Bson> = arr.iter().map(json_to_bson).collect();
            mongodb::bson::Bson::Array(bson_arr)
        }
        serde_json::Value::Object(obj) => {
            let mut doc = Document::new();
            for (k, v) in obj {
                doc.insert(k.clone(), json_to_bson(v));
            }
            mongodb::bson::Bson::Document(doc)
        }
    }
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for MongoDBEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        // Cache the schema field types so ingest can store numeric payload
        // fields as native BSON numbers (see `metadata_value_to_bson`).
        self.load_schema_types(dataset);
        println!("Dropping existing collection...");
        let _ = self.drop_collection();

        // Create the collection explicitly so we can add the index
        let db = self.client.database(&self.db_name);
        db.create_collection(&self.collection_name)
            .run()
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        println!(
            "Collection '{}.{}' created.",
            self.db_name, self.collection_name
        );

        if self.config.skip_vector_index {
            println!("Skipping vector index (filter-only mode)");
            return Ok(());
        }

        // Insert a dummy document so the index has something to build on
        let coll = db.collection::<Document>(&self.collection_name);
        let dim = dataset.vector_size();
        let dummy_vec: Vec<mongodb::bson::Bson> =
            (0..dim).map(|_| mongodb::bson::Bson::Double(0.0)).collect();
        coll.insert_one(doc! { "_id": -1i64, "vector": dummy_vec })
            .run()
            .map_err(|e| format!("Failed to insert dummy document: {}", e))?;

        println!("Creating vector search index '{}'...", self.index_name);
        self.create_vector_index(dataset)?;

        // Remove dummy document
        coll.delete_one(doc! { "_id": -1i64 })
            .run()
            .map_err(|e| format!("Failed to remove dummy document: {}", e))?;

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        // Ensure schema types are loaded even if upload runs without configure.
        self.load_schema_types(dataset);
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

        let total_time;
        if self.config.skip_vector_index {
            total_time = read_time + upload_time;
            println!(
                "Total time (read+upload): {:.3}s (no vector index)",
                total_time
            );
        } else {
            // Wait for the search index to finish indexing all uploaded documents
            // Use the first vector as a probe query to verify actual search readiness
            let probe_vector = vectors.first().ok_or("No vectors uploaded")?;
            let index_start = Instant::now();
            self.wait_for_index_catchup(vectors.len(), probe_vector)?;
            let index_time = index_start.elapsed().as_secs_f64();

            total_time = read_time + upload_time + index_time;
            println!(
                "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
                index_time, total_time
            );
        }

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
        if self.config.skip_vector_index {
            return self.search_filter_only(dataset, params, num_queries);
        }

        let parallel = params.parallel.unwrap_or(1) as usize;
        // Accept the nested (upstream `config: {...}`) placement as well as the
        // flat typed field, so an upstream-style entry is not silently ignored.
        let num_candidates_factor = params
            .knob("num_candidates")
            .and_then(|v| v.as_i64())
            .or(params.num_candidates)
            .unwrap_or(self.config.num_candidates_factor);
        // Atlas Vector Search has no `ef`: its only breadth knob is
        // numCandidates. A config that sweeps `ef` therefore produces IDENTICAL
        // runs, while the results JSON records the requested ef per row — a flat
        // line that reads as a sweep. Say so rather than let it pass.
        if params.search_params.as_ref().and_then(|sp| sp.ef).is_some() {
            eprintln!(
                "Warning: search_params.ef is ignored for MongoDB — Atlas Vector Search has no \
                 `ef`; use num_candidates to vary search breadth. Rows differing only in ef are \
                 the SAME configuration."
            );
        }

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<Document>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_mongo_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` and the fully built `$vectorSearch`
        // aggregation pipelines BEFORE the parallel region so the timed window
        // wraps only the aggregate RPC round-trip + cursor decode (see
        // build_search_pipeline / send_search). `tops[idx]` reproduces the same k
        // the pipeline embeds, so recall is computed against an identical result
        // set — this is measurement-only, unchanged recall/precision.
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
        let pipelines: Vec<Vec<Document>> = (0..num_to_run)
            .map(|idx| {
                build_search_pipeline(
                    &self.index_name,
                    &queries[idx],
                    tops[idx],
                    (tops[idx] as i64) * num_candidates_factor,
                    parsed_filters[idx].as_ref(),
                )
            })
            .collect();

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);

        // Barrier-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window (mirrors the Redis/Vertex engines).
        // Every worker connects + primes, then blocks on `ready`; the main thread
        // stamps the shared start instant into `start_cell` and releases `go`, so
        // the measurement clock starts only once all workers are warm and poised.
        // A worker that fails to connect MUST still pass both barriers before
        // returning, or the run would deadlock.
        let ready = Arc::new(std::sync::Barrier::new(parallel + 1));
        let go = Arc::new(std::sync::Barrier::new(parallel + 1));
        let start_cell = Arc::new(std::sync::OnceLock::<Instant>::new());

        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let neighbors = &neighbors;
                let tops = &tops;
                let pipelines = &pipelines;
                let query_idx = Arc::clone(&query_idx);
                let ready = Arc::clone(&ready);
                let go = Arc::clone(&go);
                let start_cell = Arc::clone(&start_cell);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(_) => {
                            // Still cross both barriers so peers aren't stranded.
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    // Prime this connection with ONE discarded query so the cold
                    // first round-trip is not inside the measured window. Best
                    // effort: errors are ignored and its sample is NOT recorded.
                    if let Some(prime_pipeline) = pipelines.first() {
                        let _ = send_search(&coll, prime_pipeline);
                    }

                    // Signal "connected + primed", then block until the main thread
                    // stamps the shared measurement start and releases everyone.
                    ready.wait();
                    go.wait();
                    let _start_time = *start_cell.get().unwrap();

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = tops[idx];

                        // Timed window: aggregate RPC send + cursor read + decode
                        // to Documents. Pipeline is prebuilt (out); id/score
                        // extraction runs after `elapsed` (out).
                        let query_start = Instant::now();
                        let response = send_search(&coll, &pipelines[idx]);
                        let query_time = query_start.elapsed().as_secs_f64();

                        match response {
                            Ok(docs) => {
                                let result_ids = extract_search_hits(&docs);
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
                        // Batch progress updates so the highest-QPS runs don't pay a
                        // contended atomic per query.
                        pb_pending += 1;
                        if pb_pending >= 256 {
                            pb.inc(pb_pending);
                            pb_pending = 0;
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
                    }
                    (t, p, r, mr, nd)
                }));
            }

            // All workers are connected + primed and waiting on `ready`. Stamp the
            // shared start instant, then release everyone via `go` so the measured
            // window excludes connection setup and the cold first query.
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
        // total_time excludes connection setup and the cold first query.
        let total_time = start_cell
            .get()
            .map(|st| st.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn search_mixed(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
        ratio: &UpdateSearchRatio,
    ) -> Result<SearchResults, String> {
        // Ensure numeric payloads written during updates use native BSON types.
        self.load_schema_types(dataset);
        let parallel = params.parallel.unwrap_or(1) as usize;
        // Accept the nested (upstream `config: {...}`) placement as well as the
        // flat typed field, so an upstream-style entry is not silently ignored.
        let num_candidates_factor = params
            .knob("num_candidates")
            .and_then(|v| v.as_i64())
            .or(params.num_candidates)
            .unwrap_or(self.config.num_candidates_factor);
        // Atlas Vector Search has no `ef`: its only breadth knob is
        // numCandidates. A config that sweeps `ef` therefore produces IDENTICAL
        // runs, while the results JSON records the requested ef per row — a flat
        // line that reads as a sweep. Say so rather than let it pass.
        if params.search_params.as_ref().and_then(|sp| sp.ef).is_some() {
            eprintln!(
                "Warning: search_params.ef is ignored for MongoDB — Atlas Vector Search has no \
                 `ef`; use num_candidates to vary search breadth. Rows differing only in ef are \
                 the SAME configuration."
            );
        }

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<Document>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_mongo_conditions))
            .collect();

        // Read vectors for updates
        let normalize = dataset.needs_normalization();
        println!("\tReading vectors for updates...");
        let (upd_ids, upd_vectors, upd_metadata) = dataset.read_vectors(normalize)?;

        // Create deterministic shuffled update sequence
        let mut update_seq: Vec<usize> = (0..upd_ids.len()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        update_seq.shuffle(&mut rng);

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` and `$vectorSearch` pipelines BEFORE the
        // parallel region so the timed search window wraps only the aggregate RPC
        // + cursor decode (matching the main search() path). Measurement-only:
        // recall/precision unchanged.
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
        let pipelines: Vec<Vec<Document>> = (0..num_to_run)
            .map(|idx| {
                build_search_pipeline(
                    &self.index_name,
                    &queries[idx],
                    tops[idx],
                    (tops[idx] as i64) * num_candidates_factor,
                    parsed_filters[idx].as_ref(),
                )
            })
            .collect();

        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));

        let ratio_searches = ratio.searches as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();
        let schema_types = &self.schema_types;

        // Each worker accumulates search + update samples into thread-local
        // buffers and returns them on join; the main thread concatenates. This
        // keeps the timed hot loop free of the 5-6 cross-thread Mutex<Vec> pushes
        // per query that serialized workers at high parallelism (matching the main
        // search() path). Dispatch counters use Relaxed (only their own
        // monotonicity matters) and the progress bar is advanced in batches.
        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut u_times: Vec<f64> = Vec::new();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let neighbors = &neighbors;
                let tops = &tops;
                let pipelines = &pipelines;
                let upd_ids = &upd_ids;
                let upd_vectors = &upd_vectors;
                let upd_metadata = &upd_metadata;
                let update_seq = &update_seq;
                let search_idx = Arc::clone(&search_idx);
                let update_idx = Arc::clone(&update_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    // Thread-local sample buffers — no cross-thread lock per query.
                    let mut t: Vec<f64> = Vec::new();
                    let mut p: Vec<f64> = Vec::new();
                    let mut r: Vec<f64> = Vec::new();
                    let mut mr: Vec<f64> = Vec::new();
                    let mut nd: Vec<f64> = Vec::new();
                    let mut ut: Vec<f64> = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(_) => return (t, p, r, mr, nd, ut),
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    'outer: loop {
                        // Search phase: do S searches
                        for _ in 0..ratio_searches {
                            let idx = search_idx.fetch_add(1, Ordering::Relaxed);
                            if idx >= num_to_run {
                                break 'outer;
                            }

                            let top = tops[idx];

                            // Timed window: aggregate RPC + cursor decode only.
                            // Pipeline prebuilt (out); id extraction after elapsed.
                            let query_start = Instant::now();
                            let response = send_search(&coll, &pipelines[idx]);
                            let query_time = query_start.elapsed().as_secs_f64();

                            match response {
                                Ok(docs) => {
                                    let result_ids = extract_search_hits(&docs);
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
                            pb_pending += 1;
                            if pb_pending >= 256 {
                                pb.inc(pb_pending);
                                pb_pending = 0;
                            }
                        }

                        // Update phase: do U updates
                        for _ in 0..ratio_updates {
                            let uidx = update_idx.fetch_add(1, Ordering::Relaxed);
                            let data_idx = update_seq[uidx % update_seq_len];

                            let update_start = Instant::now();
                            let _ = update_one_doc(
                                &coll,
                                upd_ids[data_idx],
                                &upd_vectors[data_idx],
                                upd_metadata[data_idx].as_ref(),
                                schema_types,
                            );
                            let update_time = update_start.elapsed().as_secs_f64();
                            ut.push(update_time);
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
                    }
                    (t, p, r, mr, nd, ut)
                }));
            }

            for h in handles {
                let (t, p, r, mr, nd, ut) = h.join().unwrap();
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
                u_times.extend(ut);
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        // Update latency stats (linear-interpolation percentiles, matching the
        // shared search-stats path).
        let (update_count, update_rps, update_mean_time, update_p50, update_p95, update_p99) =
            if !u_times.is_empty() {
                let u_rps = u_times.len() as f64 / total_time;
                let u_mean = u_times.iter().sum::<f64>() / u_times.len() as f64;
                let mut u_sorted: Vec<f64> = u_times.clone();
                u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (
                    Some(u_times.len()),
                    Some(u_rps),
                    Some(u_mean),
                    Some(crate::engine::percentile_linear(&u_sorted, 0.50)),
                    Some(crate::engine::percentile_linear(&u_sorted, 0.95)),
                    Some(crate::engine::percentile_linear(&u_sorted, 0.99)),
                )
            } else {
                (None, None, None, None, None, None)
            };

        // Search latency + quality stats through the shared percentile path so the
        // mixed harness matches the main search() footing.
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        let mut results = crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )?;
        results.update_count = update_count;
        results.update_rps = update_rps;
        results.update_mean_time = update_mean_time;
        results.update_p50_time = update_p50;
        results.update_p95_time = update_p95;
        results.update_p99_time = update_p99;
        results.update_latencies = Some(u_times);
        results.update_search_ratio = Some(format!("{}:{}", ratio.updates, ratio.searches));
        Ok(results)
    }

    fn delete(&mut self) -> Result<(), String> {
        self.drop_collection()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Issue #216: `hnsw_config` must be translated into the names MongoDB
    /// accepts, not the HNSW-generic `m`/`efConstruction` (which the server
    /// rejects: `unrecognized fields ["m", "efConstruction"]`).
    #[test]
    fn build_hnsw_options_uses_mongodb_field_names() {
        let opts = build_hnsw_options(Some(32), Some(200))
            .expect("no error")
            .expect("options built");
        assert_eq!(opts.get_i32("maxEdges").unwrap(), 32);
        assert_eq!(opts.get_i32("numEdgeCandidates").unwrap(), 200);
        assert!(
            !opts.contains_key("m") && !opts.contains_key("efConstruction"),
            "MongoDB rejects m/efConstruction outright: {:?}",
            opts
        );
    }

    /// Each knob is independently optional, and an unconfigured `hnsw_config`
    /// must leave the index body exactly as it was before this feature.
    #[test]
    fn build_hnsw_options_omits_unset_knobs() {
        assert!(build_hnsw_options(None, None).unwrap().is_none());

        let only_m = build_hnsw_options(Some(64), None).unwrap().unwrap();
        assert_eq!(only_m.get_i32("maxEdges").unwrap(), 64);
        assert!(!only_m.contains_key("numEdgeCandidates"));

        let only_efc = build_hnsw_options(None, Some(512)).unwrap().unwrap();
        assert_eq!(only_efc.get_i32("numEdgeCandidates").unwrap(), 512);
        assert!(!only_efc.contains_key("maxEdges"));
    }

    /// Out-of-range values are forwarded unchanged so the server can reject
    /// them. Clamping would silently benchmark an index the config never asked
    /// for — the failure mode of issue #216.
    #[test]
    fn build_hnsw_options_does_not_clamp_out_of_range_values() {
        let opts = build_hnsw_options(Some(9999), Some(1)).unwrap().unwrap();
        assert_eq!(opts.get_i32("maxEdges").unwrap(), 9999);
        assert_eq!(opts.get_i32("numEdgeCandidates").unwrap(), 1);
    }

    /// "Forwarded verbatim" must not quietly become "forwarded truncated":
    /// `as i32` would wrap 4294967328 to 32 and build a plausible-looking index.
    #[test]
    fn build_hnsw_options_rejects_values_that_do_not_fit_i32() {
        let err = build_hnsw_options(Some(4_294_967_328), None).unwrap_err();
        assert!(err.contains("does not fit in an i32"), "{}", err);
        assert!(
            err.contains('M'),
            "error must name the offending knob: {}",
            err
        );

        let err = build_hnsw_options(None, Some(i64::MAX)).unwrap_err();
        assert!(err.contains("EF_CONSTRUCTION"), "{}", err);
    }

    // Load-bearing: the hoisted (out-of-timed-window) pipeline builder must
    // produce a pipeline byte-identical to the one the previous inline build
    // embedded in the aggregate request. If this drifts, the timed window would
    // send a different request than the recall computation assumes.
    #[test]
    fn build_search_pipeline_matches_inline_build() {
        let query: Vec<f32> = vec![0.1, -0.2, 0.3];
        let top = 2usize;
        let num_candidates = 20i64;
        let filter = doc! { "color": { "$in": ["red", "blue"] } };

        // Reconstruct the exact vector-to-BSON conversion + doc order the old
        // inline path used (f32 -> f64 Double, insertion order preserved).
        let bson_vec: Vec<mongodb::bson::Bson> = query
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();
        let expected = vec![
            doc! { "$vectorSearch": {
                "index": "vidx",
                "path": "vector",
                "queryVector": bson_vec.clone(),
                "numCandidates": num_candidates,
                "limit": top as i64,
                "filter": filter.clone(),
            } },
            doc! { "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            } },
        ];

        let got = build_search_pipeline("vidx", &query, top, num_candidates, Some(&filter));
        assert_eq!(got, expected, "filtered pipeline must be byte-identical");

        // Unfiltered variant: no `filter` key at all (not an empty/null filter).
        let expected_nf = vec![
            doc! { "$vectorSearch": {
                "index": "vidx",
                "path": "vector",
                "queryVector": bson_vec,
                "numCandidates": num_candidates,
                "limit": top as i64,
            } },
            doc! { "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            } },
        ];
        let got_nf = build_search_pipeline("vidx", &query, top, num_candidates, None);
        assert_eq!(got_nf, expected_nf, "unfiltered pipeline must omit filter");
    }

    // extract_search_hits pulls (id, score) pairs out of decoded docs, in order.
    #[test]
    fn extract_search_hits_reads_id_and_score() {
        let docs = vec![
            doc! { "_id": 7i64, "score": 0.9f64 },
            doc! { "_id": 3i64, "score": 0.5f64 },
        ];
        let hits = extract_search_hits(&docs);
        assert_eq!(hits, vec![(7i64, 0.9f64), (3i64, 0.5f64)]);
    }

    // A single AND clause is returned unwrapped: {"color": {"$in": [...]}}.
    #[test]
    fn match_any_string_list_emits_in() {
        let e = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        let doc = parse_mongo_conditions(&e).unwrap();
        let vals = doc.get_document("color").unwrap().get_array("$in").unwrap();
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0].as_str(), Some("red"));
        assert_eq!(vals[1].as_str(), Some("blue"));
    }

    #[test]
    fn match_any_int_list_emits_in() {
        let e = json!({"and": [{"size": {"match": {"any": [1, 2, 3]}}}]});
        let doc = parse_mongo_conditions(&e).unwrap();
        let vals = doc.get_document("size").unwrap().get_array("$in").unwrap();
        assert_eq!(vals.len(), 3);
        // The $in elements must be NATIVE BSON integers (Int64), not strings —
        // MongoDB does no string<->number coercion, so a string "1" would never
        // match a document whose `size` is stored as native Int64(1).
        assert_eq!(vals[0].as_i64(), Some(1));
        assert_eq!(vals[1].as_i64(), Some(2));
        assert_eq!(vals[2].as_i64(), Some(3));
    }

    // Numeric `int` payload fields are stored as native BSON Int64 (mirroring
    // pgvector's BIGINT). The metadata reader stringifies JSON numbers, so we
    // parse them back per the schema type at ingest.
    #[test]
    fn int_schema_field_stored_as_native_i64() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let mut schema = HashMap::new();
        schema.insert("size".to_string(), "int".to_string());
        schema.insert("color".to_string(), "keyword".to_string());

        let size = metadata_value_to_bson("size", &MetadataValue::String("2".to_string()), &schema);
        assert_eq!(size.as_i64(), Some(2), "int field must store as Int64");

        // Keyword fields must stay strings (must NOT be coerced to a number).
        let color =
            metadata_value_to_bson("color", &MetadataValue::String("red".to_string()), &schema);
        assert_eq!(color.as_str(), Some("red"));
    }

    // A `float` schema field is stored as a native BSON Double.
    #[test]
    fn float_schema_field_stored_as_native_f64() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let mut schema = HashMap::new();
        schema.insert("price".to_string(), "float".to_string());
        let price =
            metadata_value_to_bson("price", &MetadataValue::String("3.5".to_string()), &schema);
        assert_eq!(price.as_f64(), Some(3.5));
    }

    // A native numeric MetadataValue (the reader now types JSON numbers, issue
    // #87) must map to native BSON regardless of the schema hint — the value is
    // already unambiguous, so `match_any`/range/exact filters see a real number.
    #[test]
    fn native_numeric_metadata_stored_as_native_bson() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let schema = HashMap::new(); // no schema hint needed for typed values
        let i = metadata_value_to_bson("size", &MetadataValue::Int(2), &schema);
        assert_eq!(i.as_i64(), Some(2), "Int must store as native Int64");
        let f = metadata_value_to_bson("price", &MetadataValue::Float(3.5), &schema);
        assert_eq!(f.as_f64(), Some(3.5), "Float must store as native Double");
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // Empty IN-set -> {$in: []} (matches nothing), clause not dropped.
        let e = json!({"and": [{"color": {"match": {"any": []}}}]});
        let doc = parse_mongo_conditions(&e).unwrap();
        assert!(doc
            .get_document("color")
            .unwrap()
            .get_array("$in")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn match_exact_value_still_works() {
        let e = json!({"and": [{"color": {"match": {"value": "red"}}}]});
        let doc = parse_mongo_conditions(&e).unwrap();
        assert_eq!(doc.get_str("color").unwrap(), "red");
    }

    // ── json_to_bson: non-Int scalar + container arms ──────────────────────
    #[test]
    fn json_to_bson_covers_all_arms() {
        use mongodb::bson::Bson;
        assert_eq!(json_to_bson(&json!(null)), Bson::Null);
        // Bools compare as the stored string form, not native Bson::Boolean.
        assert_eq!(json_to_bson(&json!(true)), Bson::String("true".to_string()));
        assert_eq!(
            json_to_bson(&json!(false)),
            Bson::String("false".to_string())
        );
        // Integer JSON numbers map to Int64, floats to Double.
        assert_eq!(json_to_bson(&json!(7)), Bson::Int64(7));
        assert_eq!(json_to_bson(&json!(1.5)), Bson::Double(1.5));
        assert_eq!(json_to_bson(&json!("hi")), Bson::String("hi".to_string()));
        // Array preserves order and recurses per element.
        assert_eq!(
            json_to_bson(&json!([1, "a"])),
            Bson::Array(vec![Bson::Int64(1), Bson::String("a".to_string())])
        );
        // Object → Document with recursively-converted values.
        match json_to_bson(&json!({"k": 2})) {
            Bson::Document(d) => assert_eq!(d.get_i64("k").unwrap(), 2),
            other => panic!("expected Document, got {:?}", other),
        }
    }

    // ── metadata_value_to_bson: fallback / Labels / Geo ────────────────────
    #[test]
    fn metadata_int_field_unparseable_falls_back_to_string() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let mut schema = HashMap::new();
        schema.insert("size".to_string(), "int".to_string());
        // Schema says int but the value is not a valid i64 → keep the raw string.
        let v = metadata_value_to_bson(
            "size",
            &MetadataValue::String("not-a-number".into()),
            &schema,
        );
        assert_eq!(v.as_str(), Some("not-a-number"));
    }

    #[test]
    fn metadata_labels_become_bson_string_array() {
        use mongodb::bson::Bson;
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let schema = HashMap::new();
        let v = metadata_value_to_bson(
            "tags",
            &MetadataValue::Labels(vec!["a".into(), "b".into()]),
            &schema,
        );
        assert_eq!(
            v,
            Bson::Array(vec![Bson::String("a".into()), Bson::String("b".into()),])
        );
    }

    #[test]
    fn metadata_geo_becomes_geojson_point() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let schema = HashMap::new();
        let v = metadata_value_to_bson(
            "loc",
            &MetadataValue::Geo {
                lon: 10.0,
                lat: 20.0,
            },
            &schema,
        );
        // GeoJSON Point: {type:"Point", coordinates:[lon, lat]} (lon first).
        let expected = mongodb::bson::Bson::Document(doc! {
            "type": "Point",
            "coordinates": [10.0f64, 20.0f64],
        });
        assert_eq!(v, expected);
    }

    // ── build_uri: passthrough / auth / no-auth ────────────────────────────
    // Sequenced in ONE test so the shared MONGODB_USER/PASSWORD env vars are not
    // mutated concurrently by parallel test threads (no serial_test dep here).
    #[test]
    fn build_uri_covers_passthrough_auth_and_noauth() {
        let saved_user = std::env::var("MONGODB_USER").ok();
        let saved_pass = std::env::var("MONGODB_PASSWORD").ok();

        // Full mongodb:// URI is passed through verbatim (env ignored).
        std::env::set_var("MONGODB_USER", "u");
        std::env::set_var("MONGODB_PASSWORD", "p");
        assert_eq!(
            build_uri("mongodb+srv://cluster.example.net/db", 27017),
            "mongodb+srv://cluster.example.net/db"
        );

        // user + password present → credentialled URI.
        assert_eq!(
            build_uri("host1", 27018),
            "mongodb://u:p@host1:27018/?directConnection=true"
        );

        // No credentials → plain URI.
        std::env::remove_var("MONGODB_USER");
        std::env::remove_var("MONGODB_PASSWORD");
        assert_eq!(
            build_uri("host2", 27019),
            "mongodb://host2:27019/?directConnection=true"
        );

        // Restore the environment for any other test that may read it.
        match saved_user {
            Some(v) => std::env::set_var("MONGODB_USER", v),
            None => std::env::remove_var("MONGODB_USER"),
        }
        match saved_pass {
            Some(v) => std::env::set_var("MONGODB_PASSWORD", v),
            None => std::env::remove_var("MONGODB_PASSWORD"),
        }
    }
}
