//! MongoDB Atlas Vector Search engine implementation.
//!
//! Uses the official `mongodb` crate with sync feature.
//! Supports Atlas Vector Search with HNSW index via `$vectorSearch` aggregation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use mongodb::bson::{doc, Document};
use mongodb::sync::Client;

use rand::{seq::SliceRandom, SeedableRng};

use super::geo;
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::index_naming::{derive_index_name, sanitize_token};
use crate::engine::{
    CorpusCount, Engine, IndexCoverage, SearchResults, UpdateSearchRatio, UploadStats,
};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::MetadataItem;
use vector_db_benchmark::start_gate::WorkerPool;

const DEFAULT_DB: &str = "bench";
const DEFAULT_COLLECTION: &str = "vectors";
const DEFAULT_INDEX_NAME: &str = "vector_index";

/// The field the catch-up gate's partitioned completeness probe filters on
/// (#313). It is `_id` — dense, sequential, and already written by
/// `insert_batch` — so no separate field or write path is needed.
const CATCHUP_ID_FILTER_FIELD: &str = "_id";

/// MongoDB refuses a fully-qualified namespace (`<db>.<collection>`) longer
/// than 255 bytes. Verified live against `mongodb/mongodb-atlas-local:8.0.17`:
/// a 200-byte collection under an 8-byte db is accepted, a 250-byte one is
/// rejected with `Fully qualified namespace is too long`. The Redis-wire
/// engines have no analogue — a Redis key name is bounded by 512 MB — so the
/// bounding below is MongoDB's own concern and deliberately does NOT live in
/// the shared `index_naming` module.
///
/// The search index name has no comparable LENGTH ceiling: the same server
/// accepted a 5000-byte `createSearchIndexes` name, so `index_name` is left
/// unbounded. It does, however, have a CHARACTER restriction that the local
/// image does not enforce — see [`derive_search_index_name`].
const MAX_NAMESPACE_BYTES: usize = 255;

/// Per-config search index name, safe for MongoDB **Atlas**.
///
/// [`derive_index_name`] joins base and config suffix with `:`, which is the
/// natural separator for the Redis-wire engines. MongoDB Atlas rejects it:
///
/// ```text
/// Error code 2 (BadValue): invalid index name vector_index:mongodb-m-64-ef-512
/// ```
///
/// This did not show up in the integration tests because they run against
/// `mongodb/mongodb-atlas-local`, which ACCEPTS a colon in a search index name.
/// The local image is more permissive than the hosted service, so "it passed
/// locally" is not evidence a name is valid on Atlas. The failure is total, not
/// partial: every config in a sweep fails at `configure()`, so a real Atlas run
/// produces zero summaries.
///
/// The fix is the separator, not a digest. [`sanitize_token`] already maps the
/// config suffix into `[A-Za-z0-9_-]`, so `:` is the ONLY character the
/// composed name can contain that Atlas refuses. Joining with `_` therefore
/// preserves exactly the distinctness guarantee the `:` form had — two configs
/// collide here only if they already collided before, which the startup
/// collision guard catches.
///
/// The base is sanitized too: it comes straight from `MONGODB_INDEX_NAME` and is
/// otherwise unchecked, so an operator-supplied base could reintroduce a
/// character Atlas rejects — including under the `_EXACT` escape hatch, where
/// the base is used with no suffix at all.
pub(crate) fn derive_search_index_name(engine_name: &str) -> String {
    let raw = derive_index_name("MONGODB_INDEX_NAME", DEFAULT_INDEX_NAME, engine_name);
    sanitize_token(&raw)
}

/// Per-config collection name (#306).
///
/// Before this, every config in a sweep addressed the one literal
/// `bench.vectors`. Because `configure()` drops that collection, config B
/// destroyed config A's corpus; and because `--skip-upload` skips `configure()`
/// entirely, all twelve configs of the shipped `mongodb-single-node.json`
/// M×EF_CONSTRUCTION sweep queried whichever HNSW graph the first config had
/// built — twelve result files, twelve distinct labels, one measurement.
///
/// The suffix is derived by [`derive_index_name`], the same helper the
/// Redis-wire engines use (#151-4), so `MONGODB_COLLECTION` keeps working as an
/// override but is now the *base*: the config suffix is always appended, so a
/// pinned base cannot re-collapse a sweep. `MONGODB_COLLECTION_EXACT=1` is the
/// escape hatch that drops the suffix and uses the base verbatim; combining it
/// with more than one MongoDB config is rejected at startup by the collision
/// guard in `experiment::run`.
///
/// `db_name` is passed in rather than read here because it consumes part of the
/// 255-byte namespace budget the collection name has to fit inside.
pub(crate) fn derive_collection_name(db_name: &str, engine_name: &str) -> String {
    let derived = derive_index_name("MONGODB_COLLECTION", DEFAULT_COLLECTION, engine_name);
    // `<db>` + `.` + `<collection>` must fit in MAX_NAMESPACE_BYTES.
    let budget = MAX_NAMESPACE_BYTES.saturating_sub(db_name.len() + 1);
    bound_to_bytes(&derived, budget)
}

/// The collection this process's MongoDB configs address, reading `MONGODB_DB`
/// for the namespace budget. Used by the startup collision guard, which must
/// answer "do these two configs address the same collection?" without opening a
/// connection.
pub(crate) fn config_collection_name(engine_name: &str) -> String {
    let db_name = crate::effective_config::env_or("MONGODB_DB", DEFAULT_DB);
    derive_collection_name(&db_name, engine_name)
}

/// Deterministically bound `name` to `max_bytes` while keeping distinct inputs
/// distinct.
///
/// Names already inside the bound are returned verbatim, so the common case is
/// the readable `vectors:mongodb-m-16-efc-100`. A longer name becomes
/// `<head>~<16 hex digits>`, where the digest is taken over the WHOLE name — the
/// part that was cut included. A plain truncation would map
/// `vectors:<200 shared bytes>-efc-100` and `…-efc-800` onto one collection and
/// silently reinstate exactly the bug this function exists to prevent.
///
/// FNV-1a rather than `DefaultHasher`: SipHash's keys and output are not
/// guaranteed stable across Rust releases, and this name has to resolve to the
/// same collection next week, from a differently-built binary, or `--skip-upload`
/// cannot find the corpus it is promised.
fn bound_to_bytes(name: &str, max_bytes: usize) -> String {
    if name.len() <= max_bytes {
        return name.to_string();
    }
    let digest = format!("~{:016x}", fnv1a64(name.as_bytes()));
    if max_bytes <= digest.len() {
        // Pathological budget (an absurdly long MONGODB_DB). Keep the tail of
        // the digest: still a function of the whole name, so still distinct.
        return digest[digest.len() - max_bytes..].to_string();
    }
    // `derive_index_name`'s base comes straight from the environment and is not
    // sanitized, so `name` may hold multi-byte UTF-8; walk back to a boundary.
    let mut head_end = max_bytes - digest.len();
    while head_end > 0 && !name.is_char_boundary(head_end) {
        head_end -= 1;
    }
    format!("{}{}", &name[..head_end], digest)
}

/// FNV-1a 64-bit. Fixed constants, no seed, no version dependence.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

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
        let port: u16 = crate::effective_config::env_parsed("MONGODB_PORT", 27017);

        let db_name = crate::effective_config::env_or("MONGODB_DB", DEFAULT_DB);
        // #306: both objects this engine owns are per-config. The collection is
        // the load-bearing one — `configure()` drops it, and `corpus_row_count()`
        // counts it — but the search index carries the config's HNSW knobs, so it
        // is namespaced too: under a pinned `MONGODB_COLLECTION_EXACT` base the
        // collection is shared and the index name is then the only thing keeping
        // two configs' graphs apart.
        let collection_name = derive_collection_name(&db_name, &engine_config.name);
        let index_name = derive_search_index_name(&engine_config.name);

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

        let parsed_filters: Vec<QueryFilter<Document>> =
            conditions.resolve_all("MongoDB", parse_mongo_conditions)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);

        let runnable_indices: Vec<usize> = (0..parsed_filters.len())
            .filter(|&i| parsed_filters[i].is_filtered())
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

        // Per-runnable-query `top`, computed once up front so the prime query and
        // the measured loop resolve `top` identically (and so the worker closure
        // no longer needs to borrow `neighbors`). Aligned with `runnable_indices`.
        let tops: Vec<usize> = runnable_indices
            .iter()
            .map(|&idx| {
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

        // Each worker accumulates latencies into a thread-local buffer and returns
        // it on join; the main thread concatenates. This keeps the timed hot loop
        // free of the per-query cross-thread Mutex<Vec> push that serialized
        // workers at high parallelism (matching the main search() path). The work
        // counter uses Relaxed (only its own monotonicity matters). Progress is
        // advanced in batches so the atomic isn't contended once per query.
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);

        // Gate-synchronized start, exactly as `search()` does (#214/#307). Every
        // worker builds its client, runs ONE discarded prime `find` and only then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant
        // and releases them together. Two things depend on this here:
        //
        //  * `mongodb::sync::Client::with_uri_str` performs NO I/O — topology
        //    discovery, the TCP/TLS handshake and auth are all deferred to the
        //    first operation. Ungated, that entire cost landed inside the FIRST
        //    per-query latency sample of every worker, so at `parallel: 100` a
        //    hundred samples each carried 10-50ms of connect against a ~1ms
        //    steady-state query — enough to own p99 outright.
        //  * A worker that cannot build its client used to `return` its empty
        //    buffer. Survivors still finished all `num_to_run` queries, so
        //    `failed_queries` stayed 0 and the artifact was stamped with the
        //    REQUESTED `parallel`: a `parallel: 100` row from a 60-worker run.
        //    `ticket.fail(...)` makes that a hard error instead.
        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "mongodb-filter-only", parallel);
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let parsed_filters = &parsed_filters;
                let runnable_indices = &runnable_indices;
                let tops = &tops;
                let errors = Arc::clone(&errors);
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                pool.spawn(move |ticket| {
                    // Thread-local sample buffer — no cross-thread lock per query.
                    let mut t: Vec<f64> = Vec::new();
                    let mut local_errs: Vec<String> = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("mongodb-filter-only worker setup failed: {e}"));
                            return t;
                        }
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    // Prime this connection with ONE discarded find so SDAM, the
                    // handshake, auth and the cold first round-trip are not inside
                    // the measured window. Best effort: read-only, errors ignored,
                    // and its sample is NOT recorded. `runnable_indices` is
                    // non-empty (checked above).
                    {
                        let idx = runnable_indices[0];
                        let filter = parsed_filters[idx].as_ref().unwrap();
                        let _ = filter_only_find(&coll, filter, tops[0]);
                    }

                    // Signal "connected + primed", then block until the coordinator
                    // stamps the shared measurement start and releases everyone.
                    let Some(_start_time) = ticket.arrive_and_wait() else {
                        return t;
                    };

                    loop {
                        let seq = query_idx.fetch_add(1, Ordering::Relaxed);
                        if seq >= num_to_run {
                            break;
                        }
                        let slot = seq % runnable_indices.len();
                        let idx = runnable_indices[slot];
                        let top = tops[slot];

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
                })?;
            }

            // Every worker is connected + primed and parked at the gate.
            // Stamp the shared measurement start and release them together.
            let (per_worker, measured_start) = pool.start()?;
            for t in per_worker {
                times.extend(t);
            }
            Ok(measured_start)
        })?;

        {
            let logged_errors = errors.lock().unwrap();
            if !logged_errors.is_empty() {
                for e in logged_errors.iter() {
                    eprintln!("\tFilter-only search error: {}", e);
                }
            }
        }

        pb.finish_and_clear();
        // total_time excludes connection setup and the cold first query.
        let total_time = measured_start.elapsed().as_secs_f64();

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

        // A dataset that declares a `geo` field needs the `search`-type index:
        // the `vectorSearch`-type index has no geo field type at all, and
        // `$vectorSearch`'s `filter` has no geo operator (issue #223). See the
        // `parse_mongo_search_conditions` section header.
        let index_def = if schema_declares_geo(dataset.config.schema.as_ref()) {
            let mut mapped = doc! {
                "vector": {
                    "type": "vector",
                    "numDimensions": vector_size as i32,
                    "similarity": similarity,
                },
            };
            // Same HNSW knobs, same spelling — `hnswOptions` is honoured by the
            // `search`-type index too (verified live: `maxEdges`/
            // `numEdgeCandidates` come back in `latestDefinition`).
            if let Some(hnsw_options) =
                build_hnsw_options(self.config.hnsw_m, self.config.hnsw_ef_construction)?
            {
                if let Ok(v) = mapped.get_document_mut("vector") {
                    v.insert("hnswOptions", hnsw_options);
                }
            }
            if let Some(schema_obj) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
                for (field_name, field_type) in schema_obj {
                    let ty = field_type.as_str().ok_or_else(|| {
                        format!("MongoDB: schema type for `{field_name}` is not a string")
                    })?;
                    mapped.insert(
                        field_name.clone(),
                        search_index_field_mapping(field_name, ty)?,
                    );
                }
            }
            // `_id` as a range-filterable field: the catch-up gate's completeness
            // probe filters on `_id` to stay under mongot's internal wire limit
            // (#313) no matter the corpus size. No shipped schema names a field
            // `_id`, but the check costs nothing and avoids an untested
            // duplicate-`path` shape on Atlas if one ever does.
            if !mapped.contains_key(CATCHUP_ID_FILTER_FIELD) {
                mapped.insert(CATCHUP_ID_FILTER_FIELD, doc! { "type": "number" });
            }
            doc! {
                "name": &self.index_name,
                "type": "search",
                "definition": { "mappings": { "dynamic": false, "fields": mapped } },
            }
        } else {
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

            // See the `search`-type branch above (#313): same field, same reason.
            if !schema_declares_field(dataset.config.schema.as_ref(), CATCHUP_ID_FILTER_FIELD) {
                fields.push(doc! {
                    "type": "filter",
                    "path": CATCHUP_ID_FILTER_FIELD,
                });
            }

            doc! {
                "name": &self.index_name,
                "type": "vectorSearch",
                "definition": {
                    "fields": fields,
                }
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

    /// Wait until EVERY uploaded document is searchable through the index, and
    /// report how many were confirmed.
    ///
    /// Atlas Vector Search is EVENTUALLY CONSISTENT, and the index STATUS
    /// cannot detect that: `configure()` creates the index on a one-document
    /// collection *before* the upload, so `listSearchIndexes` reports
    /// `status: READY, queryable: true` from that moment and keeps reporting it
    /// the whole time mongot is ingesting the real corpus off a change stream.
    /// The searchable-document count is therefore the entire guarantee, and
    /// searching before it is complete measures recall against a fraction of the
    /// corpus (the observed `recall 0.27` flake).
    ///
    /// The count comes from an EXHAUSTIVE (`exact: true`, ENN) probe reduced to
    /// a single number server-side by `$count`, PARTITIONED by `_id` range and
    /// summed (#313) — never one probe over the whole corpus. An approximate
    /// probe cannot answer the question at all: Atlas rejects `numCandidates`
    /// above 10_000 and requires `limit <= numCandidates`, so an ANN probe can
    /// never see past the first 10_000 documents — which is why the previous
    /// capped gate released at 0.85% indexed on glove-100 and let the run
    /// publish recall against a near-empty index (#305). ENN takes no
    /// `numCandidates` and so carries no ANN ceiling; verified against
    /// mongodb-atlas-local 8.0.17 for both dialects, returning 12_000 of 12_000
    /// at `limit` 12_000 and 20_000.
    ///
    /// A SEPARATE ceiling applies to an unpartitioned exhaustive probe: mongot
    /// streams one entry per matched document to mongod before `$count`
    /// collapses it, and that internal hop blows mongod's own message-size
    /// limits above roughly 888k documents — every dataset this suite ships.
    /// Partitioning the `_id` range, not the choice of `exact`, is what removes
    /// THAT ceiling; see `run_partitioned_catchup_count`.
    ///
    /// There is no success path that returns while under-indexed: either the
    /// probe confirms the whole corpus, or this returns `Err` and the run stops
    /// before it can publish an unbacked recall.
    fn wait_for_index_catchup(
        &self,
        expected_count: usize,
        probe_vector: &[f32],
        dialect: SearchDialect,
    ) -> Result<IndexCoverage, String> {
        let plan = catchup_plan(expected_count);

        println!(
            "Waiting for vector search index to index all {} documents (partitioned exhaustive count probe, budget {}s)...",
            expected_count,
            plan.deadline.as_secs()
        );
        let db = self.client.database(&self.db_name);
        let coll = db.collection::<Document>(&self.collection_name);
        let start = Instant::now();
        let deadline = start + plan.deadline;
        let mut last_print = Instant::now();
        let mut last_seen = 0usize;
        // The probe legitimately errors while mongot is still starting up ("is
        // not queryable"), so a single error is not fatal — but the LAST one is
        // kept so a gate that never succeeds can name why instead of reporting a
        // bare timeout. Every arm below that reaches the deadline check assigns
        // it, so it deliberately carries no initial value.
        let mut last_error: Option<String>;

        loop {
            let probe_start = Instant::now();
            let outcome = run_partitioned_catchup_count(
                &coll,
                &self.index_name,
                probe_vector,
                dialect,
                expected_count,
                CATCHUP_PARTITION_WIDTH,
            );
            let probe_elapsed = probe_start.elapsed();

            match outcome {
                Ok(searchable) if searchable >= plan.want => {
                    println!(
                        "Index caught up ({} / {} docs searchable) after {:.1}s.",
                        searchable,
                        expected_count,
                        start.elapsed().as_secs_f64()
                    );
                    return Ok(IndexCoverage {
                        searchable,
                        expected: expected_count,
                    });
                }
                Ok(searchable) => {
                    last_seen = searchable;
                    last_error = None;
                    if last_print.elapsed().as_secs() >= 10 {
                        println!(
                            "  still ingesting: {} / {} docs searchable, waiting...",
                            searchable, plan.want
                        );
                        last_print = Instant::now();
                    }
                }
                Err(e) => {
                    if last_print.elapsed().as_secs() >= 10 {
                        println!("  catch-up probe not answering yet: {}", e);
                        last_print = Instant::now();
                    }
                    last_error = Some(e);
                }
            }

            if Instant::now() > deadline {
                return Err(format!(
                    "MongoDB index catch-up FAILED: only {} of {} documents ({:.2}%) were \
                     searchable after {:.0}s. Refusing to run the search phase against a \
                     partially built index — any recall it produced would be measured against \
                     that fraction of the corpus, not the corpus the ground truth describes. \
                     Last probe error: {}",
                    last_seen,
                    expected_count,
                    IndexCoverage {
                        searchable: last_seen,
                        expected: expected_count
                    }
                    .fraction()
                        * 100.0,
                    start.elapsed().as_secs_f64(),
                    last_error.as_deref().unwrap_or("none"),
                ));
            }
            std::thread::sleep(catchup_poll_interval(probe_elapsed));
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
    let user = crate::effective_config::env_var("MONGODB_USER").ok();
    let password = crate::effective_config::env_var("MONGODB_PASSWORD").ok();

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
///
/// Returns `Ok(true)` when the server reports the update **matched no
/// document** — `update_one` runs without `upsert`, so a `matched_count` of 0
/// means the write changed nothing in this collection. Every mixed update
/// targets an `_id` this same run inserted, so a 0 means the write was not
/// applied to the corpus being searched (#293).
fn update_one_doc(
    coll: &mongodb::sync::Collection<Document>,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
    schema_types: &HashMap<String, String>,
) -> Result<bool, String> {
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

    let result = coll
        .update_one(doc! { "_id": id }, doc! { "$set": set_doc })
        .run()
        .map_err(|e| format!("Update failed for id {}: {}", id, e))?;

    Ok(result.matched_count == 0)
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
///
/// `dialect` selects the stage: [`SearchDialect::VectorSearchStage`] is the
/// long-standing `$vectorSearch` path; [`SearchDialect::SearchStage`] is the
/// `$search` + `vectorSearch`-operator path used only for geo-carrying datasets
/// (issue #223). The vector arguments are identical in both — only the stage
/// name, the nesting and the score meta differ.
fn build_search_pipeline(
    index_name: &str,
    query_vector: &[f32],
    top: usize,
    num_candidates: i64,
    filter: Option<&Document>,
    dialect: SearchDialect,
) -> Vec<Document> {
    let bson_vec: Vec<mongodb::bson::Bson> = query_vector
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let mut vs_stage = doc! {
        "path": "vector",
        "queryVector": bson_vec,
        "numCandidates": num_candidates,
        "limit": top as i64,
    };

    if let Some(f) = filter {
        vs_stage.insert("filter", f.clone());
    }

    match dialect {
        SearchDialect::VectorSearchStage => {
            let mut stage = doc! { "index": index_name };
            stage.extend(vs_stage);
            vec![
                doc! { "$vectorSearch": stage },
                doc! {
                    "$project": {
                        "_id": 1,
                        "score": { "$meta": "vectorSearchScore" },
                    }
                },
            ]
        }
        SearchDialect::SearchStage => vec![
            doc! { "$search": { "index": index_name, "vectorSearch": vs_stage } },
            doc! {
                "$project": {
                    "_id": 1,
                    "score": { "$meta": "searchScore" },
                }
            },
        ],
    }
}

/// Which vector-query stage a run uses. Decided once, from the dataset schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchDialect {
    /// `$vectorSearch` + a `vectorSearch`-type index. Every non-geo dataset.
    VectorSearchStage,
    /// `$search` + the `vectorSearch` operator + a `search`-type index. The only
    /// MongoDB path with a geo pre-filter (issue #223).
    SearchStage,
}

impl SearchDialect {
    fn for_dataset(dataset: &Dataset) -> Self {
        if schema_declares_geo(dataset.config.schema.as_ref()) {
            SearchDialect::SearchStage
        } else {
            SearchDialect::VectorSearchStage
        }
    }

    /// Build one query's filter in the grammar this dialect's stage accepts.
    ///
    /// The two `filter` grammars are NOT interchangeable — MQL on
    /// `$vectorSearch`, a MongoDB Search operator tree on `$search` — so the
    /// pairing lives here rather than at the call sites. `Ok(None)` still means
    /// "produced nothing", which `try_resolve_all` turns into the #219 error.
    fn parse(self, conditions: &serde_json::Value) -> Result<Option<Document>, String> {
        match self {
            SearchDialect::VectorSearchStage => Ok(parse_mongo_conditions(conditions)),
            SearchDialect::SearchStage => parse_mongo_search_conditions(conditions),
        }
    }
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

// ---------------------------------------------------------------------------
// Index catch-up gate (#305, partitioned in #313)
// ---------------------------------------------------------------------------

/// Everything about the index catch-up gate that is a pure function of the
/// corpus size.
///
/// Extracted out of `wait_for_index_catchup` — a `&self` method that needs a
/// live `Client` — precisely because the #305 defect lived in this arithmetic
/// and nothing could reach it: every mongodb integration corpus in this repo is
/// 20-500 vectors, so a cap that only bites above 10_000 documents was
/// unreachable from the whole suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CatchupPlan {
    /// Searchable documents the gate requires before it releases. This is
    /// `expected_count` itself and MUST NOT be capped: capping it is exactly
    /// how the gate came to release at 10_000 of glove-100's 1_183_514.
    want: usize,
    /// Wall-clock budget for the whole gate.
    deadline: Duration,
}

/// Build the catch-up plan for a corpus of `expected_count` documents.
fn catchup_plan(expected_count: usize) -> CatchupPlan {
    // Ten minutes of slack for index creation and mongot startup, plus a
    // millisecond per document — an assumed floor of 1_000 docs/sec of mongot
    // ingest. The previous flat 600s was sized for a gate that gave up after
    // 10_000 documents; a gate that genuinely waits for a million-document
    // corpus would trip it on arrival.
    const BASE_SECS: u64 = 600;
    const ASSUMED_MIN_INGEST_DOCS_PER_SEC: u64 = 1_000;

    CatchupPlan {
        want: expected_count,
        deadline: Duration::from_secs(
            BASE_SECS + expected_count as u64 / ASSUMED_MIN_INGEST_DOCS_PER_SEC,
        ),
    }
}

/// How long to sleep before the next catch-up probe.
///
/// The exhaustive probe costs O(corpus), so a fixed short interval would spend
/// most of a large gate inside probes. Sleeping at least as long as the last
/// probe took bounds probe cost at about half the gate's wall clock; the floor
/// keeps small corpora responsive and the ceiling stops the gate overshooting
/// catch-up by minutes on a huge one.
fn catchup_poll_interval(last_probe: Duration) -> Duration {
    last_probe.clamp(Duration::from_secs(2), Duration::from_secs(30))
}

// --- Wire-limit-safe partitioning (#313) ------------------------------------
//
// A single exhaustive probe over the WHOLE corpus is unusable above roughly
// 888k documents: mongot streams one entry per matched document to mongod
// BEFORE `$count` collapses it, and that internal hop hits mongod's own
// message-size limits — not the aggregation OUTPUT's 16MB BSON cap `$count`
// alone would suggest. Measured live against Atlas 8.0.30 on glove-100-angular
// (100d), the mongot->mongod message for a fully-indexed 1_183_514-document
// corpus plateaued at exactly 63_982_277 bytes = 54.06 bytes/doc, and the
// error surfaced two ways depending on where the message crossed a limit:
//
//   Error code 10334 (BSONObjectTooLarge): BSONObj size: ... is invalid.
//     Size must be between 0 and 16793600(16MB)
//   Error code 17 (ProtocolError): recv(): message msgLen ... is invalid.
//     Min 16 Max: 48000000
//
// The fix is to never let one probe response cover more than a small,
// wire-safe slice of the corpus: partition `0..expected_count` into
// contiguous `_id` ranges (dense and sequential for every dataset reader in
// this repo), probe each range independently, and sum. `CatchupPlan.want`
// stays the full uncapped `expected_count` throughout — only the TRANSPORT of
// the completeness count is chunked, never the bar it is compared against, so
// this does not reopen #305.

/// Single empirical sample (see above): Atlas 8.0.30, glove-100-angular, 100
/// dimensions. Not a documented server contract — it is one measurement, and
/// the safety margin below exists because a higher-dimensional shipped
/// dataset (e.g. dbpedia-openai-1M-1536-angular at 1536d) may cost more bytes
/// per matched document than this.
const OBSERVED_BYTES_PER_DOC: f64 = 54.06;

/// The tighter of the two wire ceilings #313 hit — the exact figure from the
/// `BSONObjectTooLarge` error text, in bytes.
const WIRE_HARD_CEILING_BYTES: f64 = 16_793_600.0;

/// How much of `WIRE_HARD_CEILING_BYTES` a partition's OWN response is allowed
/// to plan for, given `OBSERVED_BYTES_PER_DOC` is one sample, not a bound.
const WIRE_SAFETY_MARGIN: f64 = 0.5;

/// Default documents per catch-up partition — deliberately far below the
/// wire-safe maximum the constants above would allow, since
/// `OBSERVED_BYTES_PER_DOC` is one 100-dimension sample and a higher-dim
/// shipped dataset could cost more per document than that.
const CATCHUP_PARTITION_WIDTH: usize = 50_000;

/// Compile-time guard: fails the BUILD, not just a test, if
/// `CATCHUP_PARTITION_WIDTH` is ever raised past the wire-safety budget
/// without re-deriving the math — silently reproducing #313 is not an
/// available outcome of editing this one line.
const _: () = assert!(
    ((CATCHUP_PARTITION_WIDTH as f64 * OBSERVED_BYTES_PER_DOC) as u64)
        < ((WIRE_HARD_CEILING_BYTES * WIRE_SAFETY_MARGIN) as u64),
    "CATCHUP_PARTITION_WIDTH plans for more bytes than the wire-limit safety budget allows"
);

/// Bisection floor: a partition that still overflows the wire at this width is
/// treated as a hard failure rather than split further.
const CATCHUP_MIN_PARTITION_WIDTH: usize = 500;

/// Tile `0..expected_count` into contiguous, half-open `[lo, hi)` ranges of at
/// most `width` documents each — gap-free, overlap-free, summing their widths
/// to exactly `expected_count`. This is the safety-critical arithmetic behind
/// the catch-up gate's completeness guarantee: a gap or overlap here would
/// silently under- or over-count the corpus.
fn partition_ranges(expected_count: usize, width: usize) -> Vec<(i64, i64)> {
    let width = width.max(1);
    let mut ranges = Vec::new();
    let mut lo = 0usize;
    while lo < expected_count {
        let hi = (lo + width).min(expected_count);
        ranges.push((lo as i64, hi as i64));
        lo = hi;
    }
    ranges
}

/// Whether an aggregation error is the mongot->mongod wire-overflow shape
/// #313 identified, as opposed to some other probe failure (e.g. "index not
/// queryable yet") that bisecting the range cannot help with.
fn is_wire_overflow_error(e: &str) -> bool {
    e.contains("BSONObjectTooLarge") || e.contains("ProtocolError") || e.contains("msgLen")
}

/// One partition's searchable-document count, with adaptive bisection on a
/// wire-overflow error: halve the range and retry each half, down to
/// `CATCHUP_MIN_PARTITION_WIDTH`, before giving up and surfacing the original
/// error. This is what keeps a wrong `CATCHUP_PARTITION_WIDTH` a matter of a
/// few extra round trips rather than a repeat of #313 — the default width is
/// sized off a single empirical sample, and other datasets may need less.
fn run_catchup_count_range(
    coll: &mongodb::sync::Collection<Document>,
    index_name: &str,
    probe_vector: &[f32],
    dialect: SearchDialect,
    lo: i64,
    hi: i64,
) -> Result<usize, String> {
    let pipeline = build_catchup_count_pipeline(index_name, probe_vector, (lo, hi), dialect);
    match run_catchup_count(coll, &pipeline) {
        Ok(n) => Ok(n),
        Err(e)
            if is_wire_overflow_error(&e) && (hi - lo) as usize > CATCHUP_MIN_PARTITION_WIDTH =>
        {
            let mid = lo + (hi - lo) / 2;
            let left = run_catchup_count_range(coll, index_name, probe_vector, dialect, lo, mid)?;
            let right = run_catchup_count_range(coll, index_name, probe_vector, dialect, mid, hi)?;
            Ok(left + right)
        }
        Err(e) => Err(e),
    }
}

/// Run one full sweep of the exhaustive catch-up probe, partitioned by `_id`
/// range so no single probe response can trip mongot's internal wire limit
/// (#313) no matter how large the corpus.
///
/// Ranges are swept in DESCENDING `_id` order: `upload_parallel` claims
/// batches off an ascending atomic counter, so while a run is still catching
/// up, the highest-id partition is the one most likely to still be short —
/// scanning it first lets the early-exit below fire sooner. This is an
/// ordering heuristic only; correctness never depends on it, because the
/// full sum is still required to declare completeness.
///
/// Because `want == expected_count == sum of every partition's own width`, a
/// partition returning fewer documents than its own width already proves this
/// sweep is incomplete — the loop returns the partial sum immediately rather
/// than paying for the remaining partitions on every single poll.
fn run_partitioned_catchup_count(
    coll: &mongodb::sync::Collection<Document>,
    index_name: &str,
    probe_vector: &[f32],
    dialect: SearchDialect,
    expected_count: usize,
    width: usize,
) -> Result<usize, String> {
    let mut ranges = partition_ranges(expected_count, width);
    ranges.reverse();

    let mut total = 0usize;
    for (lo, hi) in ranges {
        let count = run_catchup_count_range(coll, index_name, probe_vector, dialect, lo, hi)?;
        total += count;
        if count < (hi - lo) as usize {
            return Ok(total);
        }
    }
    Ok(total)
}

/// Build the pipeline for one catch-up probe: an exhaustive nearest-neighbour
/// query over one `_id` range of the index, reduced to a single number
/// server-side.
///
/// `exact: true` is what makes the count both trustworthy and UNBOUNDED. The
/// approximate form takes `numCandidates`, which Atlas rejects above 10_000
/// (`"numCandidates" must be within bounds [1..10000]`) while also requiring
/// `limit <= numCandidates` — so no ANN probe can count past 10_000 documents,
/// no matter how it is parameterised. The exhaustive form takes no
/// `numCandidates` at all (passing one alongside `exact` is an error), scans
/// every indexed document in range, and so returns the true indexed count for
/// that range.
///
/// The `[lo, hi)` filter on `_id` is what keeps the probe's OWN response small
/// — `$count` alone only shrinks the reply on the CLIENT wire; mongot still
/// streams one entry per matched document to mongod first, and an
/// unpartitioned probe blows that internal hop above ~888k documents (#313).
/// Partitioning the range, not the choice of `exact`, is what removes the
/// ceiling.
fn build_catchup_count_pipeline(
    index_name: &str,
    query_vector: &[f32],
    range: (i64, i64),
    dialect: SearchDialect,
) -> Vec<Document> {
    let (lo, hi) = range;
    let bson_vec: Vec<mongodb::bson::Bson> = query_vector
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let filter = match dialect {
        SearchDialect::VectorSearchStage => {
            doc! { CATCHUP_ID_FILTER_FIELD: { "$gte": lo, "$lt": hi } }
        }
        SearchDialect::SearchStage => {
            doc! { "range": { "path": CATCHUP_ID_FILTER_FIELD, "gte": lo, "lt": hi } }
        }
    };

    let vs_stage = doc! {
        "path": "vector",
        "queryVector": bson_vec,
        "limit": (hi - lo).max(0),
        "exact": true,
        "filter": filter,
    };

    let search_stage = match dialect {
        SearchDialect::VectorSearchStage => {
            let mut stage = doc! { "index": index_name };
            stage.extend(vs_stage);
            doc! { "$vectorSearch": stage }
        }
        SearchDialect::SearchStage => {
            doc! { "$search": { "index": index_name, "vectorSearch": vs_stage } }
        }
    };

    vec![search_stage, doc! { "$count": "n" }]
}

/// Read the count out of one `$count` output document.
///
/// `$count` emits an int32 on the servers seen so far, but nothing in the
/// aggregation contract pins the width, and reading it as the wrong one would
/// turn a complete index into an unreadable probe and hang the gate to its
/// deadline. A missing `n` is an error rather than a zero: silently reporting
/// "0 searchable" for a malformed reply would look identical to a cold index
/// and burn the whole budget.
fn catchup_count_from_doc(doc: &Document) -> Result<usize, String> {
    let n = doc
        .get_i32("n")
        .map(i64::from)
        .or_else(|_| doc.get_i64("n"))
        .map_err(|_| format!("catch-up count returned no `n` field: {:?}", doc))?;
    Ok(n.max(0) as usize)
}

/// Run a catch-up count pipeline and return the searchable-document count.
///
/// `$count` emits NO document when its input is empty, which is how "nothing
/// indexed yet" arrives — zero, not an error.
fn run_catchup_count(
    coll: &mongodb::sync::Collection<Document>,
    pipeline: &[Document],
) -> Result<usize, String> {
    let cursor = coll
        .aggregate(pipeline.to_vec())
        .run()
        .map_err(|e| format!("catch-up count failed: {}", e))?;

    let mut searchable = 0usize;
    for result in cursor {
        let doc = result.map_err(|e| format!("catch-up count read failed: {}", e))?;
        searchable = catchup_count_from_doc(&doc)?;
    }
    Ok(searchable)
}

/// Parse filter conditions into MongoDB query document.
pub(crate) fn parse_mongo_conditions(conditions: &serde_json::Value) -> Option<Document> {
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
                // Geo-radius, in the MQL dialect (issue #223). `$geoWithin` +
                // `$centerSphere` is an exact spherical-cap test against the
                // GeoJSON Point that `metadata_value_to_bson` already stores, and
                // it needs no `2dsphere` index.
                //
                // `$centerSphere`'s radius is in RADIANS: dividing the dataset's
                // metres by `geo::EARTH_RADIUS_M` makes MongoDB's cap the same set
                // the fixtures' haversine ground truth uses, rather than the
                // 6378.1 km the MongoDB docs suggest (a 0.11 % difference in the
                // boundary).
                //
                // NOTE this arm is reachable only on the `find()` (filter-only)
                // path. `$vectorSearch`'s `filter` accepts a closed list of 10
                // operators — `$eq $ne $gt $gte $lt $lte $in $nin $exists $not` —
                // and rejects `$geoWithin` outright (verified live against
                // `mongodb/mongodb-atlas-local:8.0.17`:
                // `"filter.loc" at least one of [...] must be present`). The
                // vector path therefore uses [`parse_mongo_search_conditions`].
                "geo" => {
                    let center = (
                        criteria.get("lon").and_then(|v| v.as_f64()),
                        criteria.get("lat").and_then(|v| v.as_f64()),
                        criteria.get("radius").and_then(|v| v.as_f64()),
                    );
                    // A missing/invalid component is a DROP, which
                    // `query_filter::resolve` turns into a hard error — never a
                    // default radius nobody asked for (see `geo::query_terms`).
                    let (Some(lon), Some(lat), Some(radius)) = center else {
                        return None;
                    };
                    if !lon.is_finite()
                        || !lat.is_finite()
                        || !(radius.is_finite() && radius >= 0.0)
                    {
                        return None;
                    }
                    clauses.insert(
                        field_name.clone(),
                        doc! { "$geoWithin": { "$centerSphere": [
                            [lon, lat],
                            radius / geo::EARTH_RADIUS_M,
                        ] } },
                    );
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

// ── MongoDB Search operator dialect (the only geo-capable vector path) ───────
//
// `$vectorSearch`'s `filter` is MQL restricted to ten comparison operators, so
// there is NO geo predicate on that stage — see the note in
// `build_mongo_filter_entry`. MongoDB's geo-capable pre-filter for a vector
// query is the `vectorSearch` OPERATOR inside a `$search` stage, whose `filter`
// takes a MongoDB Search operator tree and therefore `geoWithin`. It needs a
// different index (`type: "search"` with a `vector`-typed field and a
// `geo`-typed field) and reports `$meta: "searchScore"`.
//
// This engine switches to that path for — and only for — a dataset whose schema
// declares a `geo` field ([`schema_declares_geo`]). Everything else keeps the
// `$vectorSearch` stage and the `vectorSearch`-type index it has always used, so
// no existing MongoDB number moves.
//
// Verified live on `mongodb/mongodb-atlas-local:8.0.17`: `geoWithin` + `circle`
// selected exactly the in-radius documents, and `equals`/`in`/`range`/`compound`
// behave as the MQL forms do.

/// Whether a dataset's declared schema contains a `geo` field.
pub(crate) fn schema_declares_geo(schema: Option<&serde_json::Value>) -> bool {
    schema
        .and_then(|s| s.as_object())
        .is_some_and(|o| o.values().any(|t| t.as_str() == Some("geo")))
}

/// Whether the dataset schema already declares a field named `field_name` —
/// used to avoid pushing a duplicate `path` into the index definition when
/// adding the catch-up gate's own `_id` filter field (#313).
fn schema_declares_field(schema: Option<&serde_json::Value>, field_name: &str) -> bool {
    schema
        .and_then(|s| s.as_object())
        .is_some_and(|o| o.contains_key(field_name))
}

/// The `mappings.fields` entry for one schema field in a `search`-type index.
///
/// `Err` rather than a silent omission for an unknown type: an unmapped field in
/// a `dynamic: false` index is invisible to every filter that names it, which is
/// the silent-unfiltered failure this whole area exists to stop.
fn search_index_field_mapping(field: &str, schema_type: &str) -> Result<Document, String> {
    Ok(match schema_type {
        "geo" => doc! { "type": "geo" },
        // `token` is the exact-value (non-analyzed) string type — the keyword
        // semantics `equals`/`in` need. `text` fields are also indexed as tokens
        // here because this path only has to serve the filters the harness emits.
        "keyword" | "uuid" | "text" => doc! { "type": "token" },
        "int" | "float" => doc! { "type": "number" },
        // Bools are STORED as the strings "true"/"false" (see `json_to_bson`), so
        // the exact-value token type is what matches them.
        "bool" => doc! { "type": "token" },
        // `datetime` is deliberately absent. `metadata_value_to_bson` stores it
        // as the raw ISO STRING (only `int`/`float` are parsed back to numbers),
        // so neither `number` nor `date` would match it, and a lexicographic
        // `token` range is not something this PR verified against a live server.
        // No shipped dataset carries geo and datetime together, so this arm is
        // unreachable today; it errors rather than guessing.
        other => {
            return Err(format!(
                "MongoDB: dataset schema field `{field}` has type `{other}`, which this engine \
                 cannot map into a `search`-type index. Add a mapping rather than leaving the \
                 field unindexed — an unmapped field in a `dynamic: false` index silently matches \
                 nothing (issue #223)."
            ))
        }
    })
}

/// Parse filter conditions into a MongoDB **Search** operator document, for the
/// `$search` + `vectorSearch` path.
///
/// `Err` — never a silent drop — for a condition this dialect cannot express.
pub(crate) fn parse_mongo_search_conditions(
    conditions: &serde_json::Value,
) -> Result<Option<Document>, String> {
    let Some(obj) = conditions.as_object() else {
        return Ok(None);
    };
    if obj.is_empty() {
        return Ok(None);
    }
    build_search_group(obj)
}

/// One `{and:[…], or:[…]}` group → a `compound` operator. `and` entries become
/// `filter` (non-scoring conjunction), `or` entries become `should` with
/// `minimumShouldMatch: 1`.
fn build_search_group(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<Document>, String> {
    let mut and_clauses: Vec<mongodb::bson::Bson> = Vec::new();

    if let Some(entries) = obj.get("and").and_then(|v| v.as_array()) {
        for entry in entries {
            if let Some(c) = build_search_entry(entry)? {
                and_clauses.push(mongodb::bson::Bson::Document(c));
            }
        }
    }

    if let Some(entries) = obj.get("or").and_then(|v| v.as_array()) {
        let mut should: Vec<mongodb::bson::Bson> = Vec::new();
        for entry in entries {
            if let Some(c) = build_search_entry(entry)? {
                should.push(mongodb::bson::Bson::Document(c));
            }
        }
        if !should.is_empty() {
            and_clauses.push(mongodb::bson::Bson::Document(doc! {
                "compound": { "should": should, "minimumShouldMatch": 1 }
            }));
        }
    }

    match and_clauses.len() {
        0 => Ok(None),
        // A single clause needs no wrapper — and must not get one, or the
        // multi-leaf render comparison in `filter_guard` cannot tell a
        // one-leaf group from a two-leaf group that lost a leaf.
        1 => Ok(match and_clauses.into_iter().next() {
            Some(mongodb::bson::Bson::Document(d)) => Some(d),
            _ => None,
        }),
        _ => Ok(Some(doc! { "compound": { "filter": and_clauses } })),
    }
}

fn build_search_entry(entry: &serde_json::Value) -> Result<Option<Document>, String> {
    let Some(entry_obj) = entry.as_object() else {
        return Ok(None);
    };
    if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
        return build_search_group(entry_obj);
    }

    let mut clauses: Vec<mongodb::bson::Bson> = Vec::new();
    for (field, spec) in entry_obj {
        let Some(spec_obj) = spec.as_object() else {
            return Ok(None);
        };
        for (op, criteria) in spec_obj {
            let clause = match op.as_str() {
                "geo" => {
                    let (Some(lon), Some(lat), Some(radius)) = (
                        criteria.get("lon").and_then(|v| v.as_f64()),
                        criteria.get("lat").and_then(|v| v.as_f64()),
                        criteria.get("radius").and_then(|v| v.as_f64()),
                    ) else {
                        return Ok(None);
                    };
                    if !lon.is_finite()
                        || !lat.is_finite()
                        || !(radius.is_finite() && radius >= 0.0)
                    {
                        return Ok(None);
                    }
                    // `circle.radius` is in METRES, the same unit the dataset
                    // uses, so it goes through unscaled.
                    doc! { "geoWithin": {
                        "path": field.clone(),
                        "circle": {
                            "center": { "type": "Point", "coordinates": [lon, lat] },
                            "radius": radius,
                        },
                    } }
                }
                "match" => {
                    if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                        let arr: Vec<mongodb::bson::Bson> = any.iter().map(json_to_bson).collect();
                        doc! { "in": { "path": field.clone(), "value": arr } }
                    } else if let Some(value) = criteria.get("value") {
                        doc! { "equals": { "path": field.clone(), "value": json_to_bson(value) } }
                    } else {
                        return Ok(None);
                    }
                }
                "range" => {
                    let mut r = doc! { "path": field.clone() };
                    for (key, op) in [("gt", "gt"), ("gte", "gte"), ("lt", "lt"), ("lte", "lte")] {
                        if let Some(v) = criteria.get(key).filter(|v| !v.is_null()) {
                            r.insert(op, json_to_bson(v));
                        }
                    }
                    if r.len() == 1 {
                        return Ok(None);
                    }
                    doc! { "range": r }
                }
                other => {
                    return Err(format!(
                        "MongoDB: the `$search` pre-filter dialect (used for geo-carrying \
                         datasets) has no mapping for condition type `{other}` on field \
                         `{field}`. Refusing rather than dropping the clause (issue #223)."
                    ))
                }
            };
            clauses.push(mongodb::bson::Bson::Document(clause));
        }
    }

    match clauses.len() {
        0 => Ok(None),
        1 => Ok(match clauses.into_iter().next() {
            Some(mongodb::bson::Bson::Document(d)) => Some(d),
            _ => None,
        }),
        _ => Ok(Some(doc! { "compound": { "filter": clauses } })),
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
    /// Server-side corpus size, for the `--skip-upload` reuse precondition
    /// (issue #238). `countDocuments({})` on the benchmark collection — an exact
    /// count, not the metadata estimate, because the estimate can lag a drop and
    /// report a corpus that is no longer there. A missing collection answers 0.
    ///
    /// #306 — why this stays a pure count and gained no identity marker. The
    /// collection it counts is now per-config (see [`derive_collection_name`]),
    /// so the "corpus built by a *different config*" case no longer reaches this
    /// check as a plausible number: config B's collection does not exist, the
    /// count is 0, and `classify_reuse_precondition` fails the run. An
    /// upload-time marker document would answer the same question one layer
    /// later and would itself be a row in the count.
    ///
    /// What remains unverified is the *dataset* dimension — one config reused
    /// across two same-sized datasets still certifies. That is #279, it is
    /// identical on every engine, and it wants one cross-engine mechanism rather
    /// than a MongoDB-only marker that hides the gap on this engine alone.
    fn corpus_row_count(&mut self) -> Result<Option<CorpusCount>, String> {
        let coll = self
            .client
            .database(&self.db_name)
            .collection::<Document>(&self.collection_name);
        // A missing collection is not an error in MongoDB — countDocuments
        // answers 0 — so any Err here is a real probe failure (unreachable node,
        // auth). Reporting that as a corpus of zero would send the user to
        // re-upload a corpus that is still intact.
        coll.count_documents(doc! {})
            .run()
            .map(|n| Some(CorpusCount::exact(n)))
            .map_err(|e| {
                format!(
                    "countDocuments on {}.{} failed: {}",
                    self.db_name, self.collection_name, e
                )
            })
    }

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
        // `None` means "not verified" — the only honest value when no vector
        // index was built. It is NOT the same claim as a verified 1.0 (#305).
        let mut index_coverage = None;
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
            index_coverage = Some(self.wait_for_index_catchup(
                vectors.len(),
                probe_vector,
                SearchDialect::for_dataset(dataset),
            )?);
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
            index_coverage,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        // Defensive, matching what Redis and Valkey already do at the top of
        // their own `search()`: `--skip-upload` never runs configure() (#238),
        // which is where this cache is otherwise primed. It is the identical
        // call with the identical argument, so no divergence is possible.
        //
        // It is NOT load-bearing for a pure search on shipped data — the filter
        // builders (`parse_mongo_conditions` -> `json_to_bson`) type literals off
        // the JSON value itself and never consult the cache; the cache only feeds
        // `metadata_value_to_bson`, which coerces `MetadataValue::String` on the
        // WRITE path (upload, and the update half of `search_mixed`). Kept so the
        // engine's state does not depend on which phases happened to run.
        self.load_schema_types(dataset);
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

        // Which stage this run uses, and therefore which filter grammar. Decided
        // from the dataset schema alone, so it matches the index `configure()`
        // built (issue #223).
        let dialect = SearchDialect::for_dataset(dataset);
        let parsed_filters: Vec<QueryFilter<Document>> =
            conditions.try_resolve_all("MongoDB", |c| dialect.parse(c))?;

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
                    dialect,
                )
            })
            .collect();

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);

        // Gate-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window. Every worker connects + primes, then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant and
        // releases everyone, so the measurement clock starts only once all workers
        // are warm and poised. The gate is count-agnostic: a worker that fails to
        // set up, panics, or is never started by the OS settles its ticket and turns
        // the run into a hard error instead of a hang (#214).

        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "mongodb-search", parallel);
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let neighbors = &neighbors;
                let tops = &tops;
                let pipelines = &pipelines;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                pool.spawn(move |ticket| {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("mongodb-search worker setup failed: {e}"));
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

                    // Signal "connected + primed", then block until the coordinator
                    // stamps the shared measurement start and releases everyone.
                    let Some(_start_time) = ticket.arrive_and_wait() else {
                        return (t, p, r, mr, nd);
                    };

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
        // total_time excludes connection setup and the cold first query.
        let total_time = measured_start.elapsed().as_secs_f64();

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

        // Which stage this run uses, and therefore which filter grammar. Decided
        // from the dataset schema alone, so it matches the index `configure()`
        // built (issue #223).
        let dialect = SearchDialect::for_dataset(dataset);
        let parsed_filters: Vec<QueryFilter<Document>> =
            conditions.try_resolve_all("MongoDB", |c| dialect.parse(c))?;

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
                    dialect,
                )
            })
            .collect();

        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));

        let ratio_searches = ratio.searches as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();

        let pb = self.create_progress_bar(num_to_run);

        // Gate-synchronized start, exactly as `search()` does (#214/#307). Every
        // worker builds its client, runs ONE discarded prime search and only then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant
        // and releases them together. Two things depend on this here:
        //
        //  * `mongodb::sync::Client::with_uri_str` performs NO I/O — topology
        //    discovery, the TCP/TLS handshake and auth are all deferred to the
        //    first operation. Ungated, that entire cost landed inside the FIRST
        //    per-query latency sample of every worker, so at `parallel: 100` a
        //    hundred samples each carried 10-50ms of connect against a ~1ms
        //    steady-state query — enough to own p99 outright.
        //  * A worker that cannot build its client used to `return` its empty
        //    buffers. Survivors still finished all `num_to_run` searches, so
        //    `failed_queries` stayed 0 and the artifact was stamped with the
        //    REQUESTED `parallel`: a `parallel: 100` row from a 60-worker run.
        //    `ticket.fail(...)` makes that a hard error instead.
        //
        // The shared start instant is also what `update_rps` is divided by, so
        // both halves of the mixed window now measure the same interval — one
        // that begins when every worker is warm, not when the first was spawned.
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
        let mut tally = crate::engine::UpdateTally::default();

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "mongodb-mixed", parallel);
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

                pool.spawn(move |ticket| {
                    // Thread-local sample buffers — no cross-thread lock per query.
                    let mut t: Vec<f64> = Vec::new();
                    let mut p: Vec<f64> = Vec::new();
                    let mut r: Vec<f64> = Vec::new();
                    let mut mr: Vec<f64> = Vec::new();
                    let mut nd: Vec<f64> = Vec::new();
                    let mut ut = crate::engine::UpdateTally::default();
                    let mut pb_pending: u64 = 0;

                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("mongodb-mixed worker setup failed: {e}"));
                            return (t, p, r, mr, nd, ut);
                        }
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    // Prime this connection with ONE discarded SEARCH so the cold
                    // first round-trip is outside the measured window. Deliberately
                    // a search and not an update: an update would mutate the corpus
                    // `parallel` times before the run starts, changing what every
                    // later search sees. The update half shares this connection, so
                    // it still gets the warmed pool. Best effort: errors ignored,
                    // sample NOT recorded.
                    if let Some(prime_pipeline) = pipelines.first() {
                        let _ = send_search(&coll, prime_pipeline);
                    }

                    // Signal "connected + primed", then block until the coordinator
                    // stamps the shared measurement start and releases everyone.
                    let Some(_start_time) = ticket.arrive_and_wait() else {
                        return (t, p, r, mr, nd, ut);
                    };

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
                            let outcome = update_one_doc(
                                &coll,
                                upd_ids[data_idx],
                                &upd_vectors[data_idx],
                                upd_metadata[data_idx].as_ref(),
                                schema_types,
                            );
                            let update_time = update_start.elapsed().as_secs_f64();
                            match outcome {
                                // matched_count >= 1: a document in the collection
                                // carried that _id and the $set was applied to it.
                                // Note this is the FILTER matching, not a statement
                                // about what was written — see MatchedRow.
                                Ok(false) => ut.times.push(update_time),
                                // matched_count == 0: nothing carried that _id, so
                                // the write (no upsert) changed nothing at all.
                                Ok(true) => ut.unattributed += 1,
                                Err(e) => {
                                    ut.failed += 1;
                                    eprintln!("Mixed update {} failed: {}", uidx, e);
                                }
                            }
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
                    }
                    (t, p, r, mr, nd, ut)
                })?;
            }

            // Every worker is connected + primed and parked at the gate.
            // Stamp the shared measurement start and release them together.
            let (per_worker, measured_start) = pool.start()?;
            for (t, p, r, mr, nd, ut) in per_worker {
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
                tally.merge(ut);
            }
            Ok(measured_start)
        })?;

        pb.finish_and_clear();
        // total_time excludes connection setup and the cold first query, for both
        // the search half (rps/percentiles) and the update half (update_rps).
        let total_time = measured_start.elapsed().as_secs_f64();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        // Search latency + quality stats through the shared percentile path so the
        // mixed harness matches the main search() footing.
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        let mut results = crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )?;
        crate::engine::finalize_update_stats(
            &mut results,
            tally,
            total_time,
            // NOT CorpusRow: matched_count describes the update's FILTER rather
            // than what was written, and the collection is not the searched
            // index. See UpdateAttribution::MatchedRow.
            crate::engine::UpdateAttribution::MatchedRow,
            ratio,
            "update_one reports matched_count; 0 means no document carried that _id. The \
             count describes the update's FILTER, not the payload, and the collection lags \
             the Atlas vector index by roughly a second, so a matched write is not yet a \
             searchable one",
        );
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

    // ── #306: per-config collection / index namespacing ──────────────────
    //
    // `derive_index_name` resolves through the process-wide effective-config
    // recorder (#212), so every test below takes `test_lock()` to serialize with
    // the other tests that drive it.

    /// The headline of #306, pinned to literal strings.
    ///
    /// Two configs of the shipped M×EF_CONSTRUCTION sweep must address two
    /// different collections AND two different search indexes. Before the fix
    /// both resolved to the bare `vectors` / `vector_index`, so `configure()`
    /// dropped the sibling's corpus and `--skip-upload` measured whichever HNSW
    /// graph happened to survive.
    ///
    /// FAILS ON REVERT: with the constants read straight from `env_or`, both
    /// configs get `"vectors"` and `"vector_index"` — the `assert_ne!`s and the
    /// literal-equality assertions all break.
    #[test]
    fn sweep_configs_derive_distinct_collections_and_indexes() {
        let _recorder_lock = crate::effective_config::test_lock();
        let a = derive_collection_name("bench", "mongodb-m-16-efc-100");
        let b = derive_collection_name("bench", "mongodb-m-64-efc-800");
        assert_eq!(a, "vectors:mongodb-m-16-efc-100");
        assert_eq!(b, "vectors:mongodb-m-64-efc-800");
        assert_ne!(a, b);
        // And neither may be the bare legacy collection every config shared.
        assert_ne!(a, DEFAULT_COLLECTION);
        assert_ne!(b, DEFAULT_COLLECTION);

        // `_`, not `:` — Atlas rejects a colon in a search index name. The
        // distinctness guarantee is unchanged: the suffix is still the whole
        // sanitized config name.
        let ia = derive_search_index_name("mongodb-m-16-efc-100");
        let ib = derive_search_index_name("mongodb-m-64-efc-800");
        assert_eq!(ia, "vector_index_mongodb-m-16-efc-100");
        assert_eq!(ib, "vector_index_mongodb-m-64-efc-800");
        assert_ne!(ia, ib);
        assert_ne!(ia, DEFAULT_INDEX_NAME);
    }

    /// Regression for the total-failure mode: a colon anywhere in the search
    /// index name makes Atlas reject `createSearchIndexes` with
    /// `BadValue: invalid index name`, so EVERY config in a sweep fails at
    /// `configure()` and the run yields zero summaries.
    ///
    /// `mongodb/mongodb-atlas-local` accepts the colon, which is why the
    /// integration tests never caught this — so this has to be asserted on the
    /// name itself rather than left to a live round-trip.
    #[test]
    fn search_index_names_are_atlas_legal() {
        let _recorder_lock = crate::effective_config::test_lock();
        let legal = |n: &str| {
            n.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        };

        for cfg in [
            "mongodb-m-16-ef-128",
            "mongodb-m-64-ef-512",
            "weird name/with:punctuation.and*globs",
            "",
        ] {
            let name = derive_search_index_name(cfg);
            assert!(
                legal(&name),
                "search index name {name:?} (from config {cfg:?}) is not Atlas-legal"
            );
            assert!(!name.contains(':'), "colon survived in {name:?}");
        }
    }

    /// The `_EXACT` escape hatch bypasses the config suffix entirely and uses
    /// the operator-supplied base verbatim, so it is the one path where an
    /// invalid character can reach Atlas without passing through
    /// `sanitize_token` on the suffix. It must be sanitized too.
    #[test]
    fn exact_pinned_index_base_is_still_sanitized() {
        let _recorder_lock = crate::effective_config::test_lock();
        std::env::set_var("MONGODB_INDEX_NAME", "my:pinned:index");
        std::env::set_var("MONGODB_INDEX_NAME_EXACT", "1");
        let name = derive_search_index_name("mongodb-m-16-ef-128");
        std::env::remove_var("MONGODB_INDEX_NAME");
        std::env::remove_var("MONGODB_INDEX_NAME_EXACT");
        assert_eq!(name, "my_pinned_index");
        assert!(!name.contains(':'));
    }

    /// The shipped sweep is the artefact the issue names: 12 configs, 12 result
    /// files, one measurement. Read the real file rather than a hand-copied list
    /// so adding a 13th config that collides is caught here and not in a
    /// published result set.
    ///
    /// FAILS ON REVERT: all 12 configs collapse to `"vectors"`, so the set of
    /// distinct names is 1, not 12.
    #[test]
    fn shipped_hnsw_sweep_gives_every_config_its_own_collection() {
        let _recorder_lock = crate::effective_config::test_lock();
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/experiments/configurations/mongodb-single-node.json"
        );
        let raw = std::fs::read_to_string(path).expect("shipped mongodb sweep config");
        let configs: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("valid JSON array");
        assert!(
            configs.len() >= 12,
            "the sweep this issue is about ships 12 configs, found {}",
            configs.len()
        );

        let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for cfg in &configs {
            let name = cfg["name"].as_str().expect("config name").to_string();
            let coll = derive_collection_name("bench", &name);
            if let Some(prev) = seen.insert(coll.clone(), name.clone()) {
                panic!("configs '{prev}' and '{name}' both address collection '{coll}'");
            }
        }
        assert_eq!(
            seen.len(),
            configs.len(),
            "every config must own its collection"
        );
    }

    /// The namespace ceiling. `<db>.<collection>` over 255 bytes is refused by
    /// the server outright, so a long config name must be bounded — and bounding
    /// must not undo the isolation it is bounding.
    ///
    /// This is the collision path #294 warns about being shipped untested: the
    /// two names below share a 300-byte prefix and differ only in the tail that
    /// truncation throws away.
    ///
    /// FAILS ON REVERT of the bounding: a plain truncate makes the two names
    /// equal. FAILS ON REVERT of #306 entirely: both become `"vectors"`, equal
    /// again.
    #[test]
    fn long_config_names_are_bounded_without_colliding() {
        let _recorder_lock = crate::effective_config::test_lock();
        let shared = "x".repeat(300);
        let a = derive_collection_name("bench", &format!("{shared}-efc-100"));
        let b = derive_collection_name("bench", &format!("{shared}-efc-800"));

        for (label, n) in [("a", &a), ("b", &b)] {
            assert!(
                "bench".len() + 1 + n.len() <= MAX_NAMESPACE_BYTES,
                "{label} namespace is {} bytes, over MongoDB's {MAX_NAMESPACE_BYTES}: {n}",
                "bench".len() + 1 + n.len()
            );
        }
        assert_ne!(
            a, b,
            "two configs differing only past the truncation point must NOT share \
             a collection — that is #306 reintroduced through the bounding"
        );
        assert!(a.contains('~') && b.contains('~'), "{a} / {b}");
    }

    /// A longer database name eats into the collection's budget, and the result
    /// must still be a legal namespace. Pins that the budget is computed from the
    /// db name rather than assumed.
    ///
    /// FAILS ON REVERT: `saturating_sub` removed / db length ignored → the long-db
    /// case exceeds 255 bytes.
    #[test]
    fn the_namespace_budget_accounts_for_the_database_name() {
        let _recorder_lock = crate::effective_config::test_lock();
        let cfg = "y".repeat(400);
        for db in ["b", "bench", &"d".repeat(200)] {
            let coll = derive_collection_name(db, &cfg);
            assert!(
                db.len() + 1 + coll.len() <= MAX_NAMESPACE_BYTES,
                "db '{}' ({} bytes) + collection ({} bytes) exceeds {MAX_NAMESPACE_BYTES}",
                db,
                db.len(),
                coll.len()
            );
            assert!(
                !coll.is_empty(),
                "an empty collection name is not addressable"
            );
        }
    }

    /// The bounding must be reproducible across processes and builds: a
    /// `--skip-upload` run has to resolve to the collection last week's upload
    /// created. FNV-1a's constants are fixed; `DefaultHasher`'s are not
    /// guaranteed to be.
    ///
    /// FAILS ON REVERT to a seeded/unstable hash only if the seed changes within
    /// the process, so it also pins the literal digest — a switch to any other
    /// hash function changes this string.
    #[test]
    fn bounding_is_deterministic() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);

        let long = "z".repeat(400);
        assert_eq!(bound_to_bytes(&long, 40), bound_to_bytes(&long, 40));
        assert_eq!(bound_to_bytes(&long, 40).len(), 40);
        // Short names are returned untouched — readability of the common case.
        assert_eq!(bound_to_bytes("vectors:cfg", 255), "vectors:cfg");
    }

    /// A budget smaller than the digest itself must still produce a distinct,
    /// non-empty, correctly-sized name rather than panicking or slicing out of
    /// bounds. Reachable via an absurd `MONGODB_DB`.
    #[test]
    fn bounding_survives_a_budget_smaller_than_the_digest() {
        // Both inputs are longer than every budget tried, so the bounded form is
        // always exercised (a name shorter than its budget is returned verbatim).
        let alpha = format!("vectors:{}", "a".repeat(60));
        let beta = format!("vectors:{}", "b".repeat(60));
        for budget in [1usize, 4, 8, 16, 17, 18, 32] {
            let a = bound_to_bytes(&alpha, budget);
            let b = bound_to_bytes(&beta, budget);
            assert_eq!(a.len(), budget, "budget {budget}");
            assert_eq!(b.len(), budget, "budget {budget}");
            if budget >= 8 {
                assert_ne!(a, b, "budget {budget} collapsed two distinct names");
            }
        }
    }

    /// Multi-byte UTF-8 in the base (which comes from the environment and is NOT
    /// sanitized) must not be sliced mid-character.
    #[test]
    fn bounding_cuts_on_a_char_boundary() {
        let name = "é".repeat(200); // 400 bytes, 200 chars
        for budget in 17..40 {
            // Slicing at a non-boundary would panic here — that is the assertion.
            let out = bound_to_bytes(&name, budget);
            assert!(out.len() <= budget, "budget {budget}: {out}");
            let head = out.split('~').next().unwrap();
            assert!(
                head.chars().all(|c| c == 'é'),
                "budget {budget} cut mid-character: {head:?}"
            );
        }
    }

    /// `MONGODB_COLLECTION` remains an override, but as a BASE: the config
    /// suffix is still appended, so pinning it cannot re-collapse a sweep.
    /// `MONGODB_COLLECTION_EXACT=1` is the documented escape hatch that drops
    /// the suffix — and the startup guard in `experiment::run` rejects it with
    /// more than one MongoDB config.
    ///
    /// FAILS ON REVERT: the non-exact half asserts the suffix is present, which
    /// the old `env_or` passthrough does not produce.
    #[test]
    fn collection_env_override_is_a_base_not_a_pin() {
        let _recorder_lock = crate::effective_config::test_lock();
        std::env::set_var("MONGODB_COLLECTION", "mycoll");
        assert_eq!(
            derive_collection_name("bench", "mongodb-m-16-efc-100"),
            "mycoll:mongodb-m-16-efc-100"
        );

        std::env::set_var("MONGODB_COLLECTION_EXACT", "1");
        assert_eq!(
            derive_collection_name("bench", "mongodb-m-16-efc-100"),
            "mycoll"
        );

        std::env::remove_var("MONGODB_COLLECTION_EXACT");
        std::env::remove_var("MONGODB_COLLECTION");
    }

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

        let got = build_search_pipeline(
            "vidx",
            &query,
            top,
            num_candidates,
            Some(&filter),
            SearchDialect::VectorSearchStage,
        );
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
        let got_nf = build_search_pipeline(
            "vidx",
            &query,
            top,
            num_candidates,
            None,
            SearchDialect::VectorSearchStage,
        );
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

    // ── Geo (issue #223) ───────────────────────────────────────────────────

    /// The `find()` (filter-only) path keeps MQL, where `$geoWithin` +
    /// `$centerSphere` IS available and needs no `2dsphere` index. The radius is
    /// in RADIANS, so it is divided by the same mean earth radius the fixtures'
    /// haversine ground truth uses — not the 6378.1 km the MongoDB docs suggest.
    #[test]
    fn mql_geo_becomes_center_sphere_in_radians() {
        let doc_ = parse_mongo_conditions(
            &json!({"and":[{"loc":{"geo":{"lat":20.0,"lon":10.0,"radius":6_371_000.0}}}]}),
        )
        .unwrap();
        let within = doc_
            .get_document("loc")
            .unwrap()
            .get_document("$geoWithin")
            .unwrap()
            .get_array("$centerSphere")
            .unwrap();
        assert_eq!(
            within[0],
            mongodb::bson::Bson::Array(vec![
                mongodb::bson::Bson::Double(10.0),
                mongodb::bson::Bson::Double(20.0),
            ]),
            "lon first, matching the stored GeoJSON coordinates"
        );
        // radius == R  =>  exactly one radian.
        assert_eq!(within[1], mongodb::bson::Bson::Double(1.0));
    }

    #[test]
    fn mql_geo_missing_component_is_a_drop() {
        for bad in [
            json!({"lat":20.0,"lon":10.0}),
            json!({"lon":10.0,"radius":500}),
            json!({"lat":20.0,"radius":500}),
            json!({"lat":20.0,"lon":10.0,"radius":-5}),
        ] {
            assert!(
                parse_mongo_conditions(&json!({"and":[{"loc":{"geo": bad}}]})).is_none(),
                "{bad}"
            );
        }
    }

    /// `$vectorSearch`'s `filter` is MQL restricted to ten comparison operators
    /// and REJECTS `$geoWithin` outright (verified live against
    /// `mongodb/mongodb-atlas-local:8.0.17`). The dialect switch is what keeps
    /// the MQL geo form off that stage.
    #[test]
    fn a_geo_schema_selects_the_search_stage_dialect() {
        assert!(schema_declares_geo(Some(&json!({"a":"geo","b":"geo"}))));
        assert!(schema_declares_geo(Some(
            &json!({"kw":"keyword","a":"geo"})
        )));
        assert!(!schema_declares_geo(Some(&json!({"kw":"keyword"}))));
        assert!(!schema_declares_geo(None));
    }

    /// Guards the duplicate-`path` avoidance in `create_vector_index` (#313):
    /// the catch-up gate's own `_id` filter field must not be pushed twice if
    /// a dataset schema ever names a field `_id`.
    #[test]
    fn schema_declares_field_checks_by_name_not_by_type() {
        assert!(schema_declares_field(Some(&json!({"_id": "int"})), "_id"));
        assert!(!schema_declares_field(
            Some(&json!({"category": "keyword"})),
            "_id"
        ));
        assert!(!schema_declares_field(None, "_id"));
    }

    #[test]
    fn search_dialect_geo_emits_geo_within_circle_in_metres() {
        let d = parse_mongo_search_conditions(
            &json!({"and":[{"loc":{"geo":{"lat":20.0,"lon":10.0,"radius":500.0}}}]}),
        )
        .unwrap()
        .unwrap();
        let circle = d
            .get_document("geoWithin")
            .unwrap()
            .get_document("circle")
            .unwrap();
        assert_eq!(
            circle
                .get_document("center")
                .unwrap()
                .get_array("coordinates")
                .unwrap(),
            &vec![
                mongodb::bson::Bson::Double(10.0),
                mongodb::bson::Bson::Double(20.0)
            ]
        );
        // `circle.radius` is metres — the dataset's own unit, unscaled.
        assert_eq!(circle.get_f64("radius").unwrap(), 500.0);
        assert_eq!(
            d.get_document("geoWithin")
                .unwrap()
                .get_str("path")
                .unwrap(),
            "loc"
        );
    }

    #[test]
    fn search_dialect_ands_with_compound_filter_and_ors_with_should() {
        let and = parse_mongo_search_conditions(&json!({"and":[
            {"a":{"geo":{"lat":1.0,"lon":2.0,"radius":10.0}}},
            {"b":{"geo":{"lat":3.0,"lon":4.0,"radius":20.0}}}
        ]}))
        .unwrap()
        .unwrap();
        assert_eq!(
            and.get_document("compound")
                .unwrap()
                .get_array("filter")
                .unwrap()
                .len(),
            2
        );
        let or = parse_mongo_search_conditions(&json!({"or":[
            {"a":{"geo":{"lat":1.0,"lon":2.0,"radius":10.0}}},
            {"a":{"geo":{"lat":3.0,"lon":4.0,"radius":20.0}}}
        ]}))
        .unwrap()
        .unwrap();
        let compound = or.get_document("compound").unwrap();
        assert_eq!(compound.get_array("should").unwrap().len(), 2);
        assert_eq!(compound.get_i32("minimumShouldMatch").unwrap(), 1);
    }

    /// The non-geo leaves of the search dialect, so a mixed geo dataset does not
    /// silently lose them. Verified live on the pinned Atlas image.
    #[test]
    fn search_dialect_maps_match_and_range() {
        let eq = parse_mongo_search_conditions(&json!({"and":[{"kw":{"match":{"value":"red"}}}]}))
            .unwrap()
            .unwrap();
        assert_eq!(
            eq.get_document("equals").unwrap().get_str("value").unwrap(),
            "red"
        );
        let any = parse_mongo_search_conditions(
            &json!({"and":[{"kw":{"match":{"any":["red","blue"]}}}]}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            any.get_document("in")
                .unwrap()
                .get_array("value")
                .unwrap()
                .len(),
            2
        );
        let rng = parse_mongo_search_conditions(&json!({"and":[{"n":{"range":{"gte":1,"lt":9}}}]}))
            .unwrap()
            .unwrap();
        let r = rng.get_document("range").unwrap();
        assert_eq!(r.get_i64("gte").unwrap(), 1);
        assert_eq!(r.get_i64("lt").unwrap(), 9);
    }

    /// An operator the search dialect has no mapping for must be an ERROR, not a
    /// dropped clause — the whole point of #219/#223.
    #[test]
    fn search_dialect_refuses_an_unmappable_condition_type() {
        let err = parse_mongo_search_conditions(&json!({"and":[{"b":{"nonsense":{"x":1}}}]}))
            .unwrap_err();
        assert!(err.contains("nonsense"), "{err}");
        assert!(err.contains("#223"), "{err}");
    }

    /// A `dynamic: false` index leaves an unmapped field invisible to every
    /// filter that names it, so an unknown schema type must fail loudly at index
    /// creation rather than produce a quietly unfiltered run.
    #[test]
    fn search_index_mapping_covers_the_known_types_and_refuses_the_rest() {
        assert_eq!(
            search_index_field_mapping("loc", "geo")
                .unwrap()
                .get_str("type")
                .unwrap(),
            "geo"
        );
        for ty in ["keyword", "uuid", "text", "bool"] {
            assert_eq!(
                search_index_field_mapping("f", ty)
                    .unwrap()
                    .get_str("type")
                    .unwrap(),
                "token",
                "{ty}"
            );
        }
        for ty in ["int", "float"] {
            assert_eq!(
                search_index_field_mapping("f", ty)
                    .unwrap()
                    .get_str("type")
                    .unwrap(),
                "number",
                "{ty}"
            );
        }
        // datetime is stored as an ISO STRING here, so neither `number` nor
        // `date` would match it; refused rather than guessed.
        assert!(search_index_field_mapping("ts", "datetime").is_err());
        assert!(search_index_field_mapping("x", "not-a-type").is_err());
    }

    /// The `$search` pipeline: different stage, different nesting, different
    /// score meta — and the same vector arguments.
    #[test]
    fn search_stage_pipeline_wraps_the_vector_args_and_reads_search_score() {
        let filter = doc! { "geoWithin": { "path": "loc" } };
        let p = build_search_pipeline(
            "idx",
            &[1.0, 0.0],
            7,
            70,
            Some(&filter),
            SearchDialect::SearchStage,
        );
        let inner = p[0]
            .get_document("$search")
            .unwrap()
            .get_document("vectorSearch")
            .unwrap();
        assert_eq!(inner.get_str("path").unwrap(), "vector");
        assert_eq!(inner.get_i64("limit").unwrap(), 7);
        assert_eq!(inner.get_i64("numCandidates").unwrap(), 70);
        assert_eq!(inner.get_document("filter").unwrap(), &filter);
        assert_eq!(
            p[0].get_document("$search")
                .unwrap()
                .get_str("index")
                .unwrap(),
            "idx"
        );
        assert_eq!(
            p[1].get_document("$project")
                .unwrap()
                .get_document("score")
                .unwrap()
                .get_str("$meta")
                .unwrap(),
            "searchScore"
        );
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
        let saved_user = crate::effective_config::env_var("MONGODB_USER").ok();
        let saved_pass = crate::effective_config::env_var("MONGODB_PASSWORD").ok();

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

    // -----------------------------------------------------------------------
    // Index catch-up gate (#305)
    //
    // Every mongodb integration corpus in this repo is 20-500 vectors, so the
    // >10_000 branch these tests pin is reachable from NOTHING else in the
    // suite. That is why #305 survived: the gate's cap only bites above 10_000
    // documents, and no test ever handed it that many.
    // -----------------------------------------------------------------------

    /// The #305 defect itself: `want` was `expected_count.min(10_000)`, so the
    /// gate released once 10_000 documents of a 1_183_514-document corpus were
    /// searchable — 0.85% — and the search phase published recall against it.
    ///
    /// `want` must equal the corpus, at every size, with no ceiling.
    #[test]
    fn catchup_plan_requires_the_whole_corpus_at_every_size() {
        // Straddles the old cap: 9_999 passed before AND after, 10_001 is the
        // smallest corpus the old code got wrong.
        for expected in [1usize, 9_999, 10_000, 10_001, 1_000_000, 1_183_514] {
            let plan = catchup_plan(expected);
            assert_eq!(
                plan.want, expected,
                "catch-up must wait for all {expected} documents, not a capped \
                 fraction of them (#305)"
            );
        }
    }

    /// A gate that genuinely waits for the whole corpus needs a budget that
    /// scales with it. The flat 600s was sized for a gate that gave up after
    /// 10_000 documents; kept flat, a corpus this tool actually ships would hit
    /// the deadline as a matter of course and the fix would trade a silent wrong
    /// answer for a guaranteed hard failure.
    #[test]
    fn catchup_plan_deadline_scales_with_the_corpus() {
        assert_eq!(catchup_plan(0).deadline, Duration::from_secs(600));
        assert_eq!(catchup_plan(9_999).deadline, Duration::from_secs(609));
        assert_eq!(catchup_plan(1_000_000).deadline, Duration::from_secs(1_600));
        assert!(
            catchup_plan(1_183_514).deadline > Duration::from_secs(600),
            "glove-100 must get more than the old flat budget"
        );
    }

    /// Probe cost is O(corpus), so the interval must not be a fixed short sleep
    /// on a large corpus, nor a long one on a small corpus.
    #[test]
    fn catchup_poll_interval_is_bounded_by_the_last_probe() {
        assert_eq!(
            catchup_poll_interval(Duration::from_millis(5)),
            Duration::from_secs(2),
            "floor keeps small corpora responsive"
        );
        assert_eq!(
            catchup_poll_interval(Duration::from_secs(7)),
            Duration::from_secs(7),
            "in between, sleep as long as the probe took"
        );
        assert_eq!(
            catchup_poll_interval(Duration::from_secs(600)),
            Duration::from_secs(30),
            "ceiling stops the gate overshooting catch-up by minutes"
        );
    }

    /// The probe must be EXHAUSTIVE, because the approximate one physically
    /// cannot count past 10_000: Atlas rejects `numCandidates` above 10_000 and
    /// requires `limit <= numCandidates`. A probe carrying `numCandidates` is
    /// therefore a probe that has silently reacquired the #305 ceiling — and it
    /// is also the shape that produced the secondary hazard the issue names
    /// (`limit == numCandidates == 10_000`). It must also carry the `_id` range
    /// filter (#313) — without it, one probe covers the whole corpus again and
    /// reproduces the wire-limit failure the partitioning exists to avoid.
    #[test]
    fn catchup_probe_is_exhaustive_partitioned_and_carries_no_ann_knob() {
        let query = vec![0.1f32, -0.2, 0.3];
        let pipeline = build_catchup_count_pipeline(
            "vidx",
            &query,
            (200_000, 250_000),
            SearchDialect::VectorSearchStage,
        );

        assert_eq!(pipeline.len(), 2, "probe is <search stage> + $count");
        let vs = pipeline[0].get_document("$vectorSearch").unwrap();
        assert!(
            vs.get_bool("exact").unwrap(),
            "the probe must be exhaustive (ENN)"
        );
        assert!(
            !vs.contains_key("numCandidates"),
            "numCandidates caps the probe at 10_000 documents and is invalid \
             alongside `exact`: {vs:?}"
        );
        assert_eq!(
            vs.get_i64("limit").unwrap(),
            50_000,
            "the probe must ask for exactly this partition's width, not the whole corpus"
        );
        assert_eq!(vs.get_str("index").unwrap(), "vidx");

        let filter = vs.get_document("filter").unwrap();
        let id_range = filter.get_document("_id").unwrap();
        assert_eq!(id_range.get_i64("$gte").unwrap(), 200_000);
        assert_eq!(id_range.get_i64("$lt").unwrap(), 250_000);

        // Counted server-side: mongot still scans, but one `{n: ...}` comes back
        // instead of the whole partition's documents.
        assert_eq!(pipeline[1], doc! { "$count": "n" });
    }

    /// The geo datasets run the `$search` dialect (#223), and they are subject
    /// to exactly the same ceiling — so the same exhaustive, partitioned shape
    /// has to reach the other stage too, nested under the `vectorSearch`
    /// operator, with the `_id` range in that dialect's own filter grammar.
    #[test]
    fn catchup_probe_is_exhaustive_partitioned_on_the_search_dialect_too() {
        let query = vec![0.1f32, -0.2, 0.3];
        let pipeline =
            build_catchup_count_pipeline("vidx", &query, (0, 25_000), SearchDialect::SearchStage);

        let vs = pipeline[0]
            .get_document("$search")
            .unwrap()
            .get_document("vectorSearch")
            .unwrap();
        assert!(
            vs.get_bool("exact").unwrap(),
            "the probe must be exhaustive (ENN)"
        );
        assert!(!vs.contains_key("numCandidates"), "{vs:?}");
        assert_eq!(vs.get_i64("limit").unwrap(), 25_000);
        assert_eq!(
            pipeline[0]
                .get_document("$search")
                .unwrap()
                .get_str("index")
                .unwrap(),
            "vidx"
        );

        let range = vs
            .get_document("filter")
            .unwrap()
            .get_document("range")
            .unwrap();
        assert_eq!(range.get_str("path").unwrap(), "_id");
        assert_eq!(range.get_i64("gte").unwrap(), 0);
        assert_eq!(range.get_i64("lt").unwrap(), 25_000);

        assert_eq!(pipeline[1], doc! { "$count": "n" });
    }

    /// The safety-critical arithmetic behind the completeness guarantee: the
    /// tiling must be gap-free, overlap-free, and sum its widths to exactly
    /// `expected_count` — a bug here would silently under- or over-count the
    /// corpus (#313).
    #[test]
    fn partition_ranges_tiles_the_corpus_without_gaps_or_overlaps() {
        for (expected_count, width) in [
            (0usize, 50_000usize),
            (1, 50_000),
            (49_999, 50_000),
            (50_000, 50_000),
            (50_001, 50_000),
            (1_183_514, 50_000),
            (10, 1),
        ] {
            let ranges = partition_ranges(expected_count, width);
            let mut expected_lo = 0i64;
            for &(lo, hi) in &ranges {
                assert_eq!(
                    lo, expected_lo,
                    "range must start exactly where the previous one ended \
                     (expected_count={expected_count}, width={width})"
                );
                assert!(hi > lo, "range must be non-empty: ({lo}, {hi})");
                expected_lo = hi;
            }
            assert_eq!(
                expected_lo, expected_count as i64,
                "the tiling must cover exactly [0, expected_count) with nothing left over \
                 (expected_count={expected_count}, width={width})"
            );
        }
    }

    #[test]
    fn partition_ranges_is_empty_for_an_empty_corpus() {
        assert_eq!(partition_ranges(0, 50_000), Vec::<(i64, i64)>::new());
    }

    /// A corpus smaller than the partition width must produce exactly the one
    /// range the un-partitioned probe used to ask for — partitioning must not
    /// change behaviour below the width it exists to protect against.
    #[test]
    fn partition_ranges_is_a_single_range_below_the_width() {
        assert_eq!(partition_ranges(10_001, 50_000), vec![(0, 10_001)]);
    }

    // `CATCHUP_PARTITION_WIDTH` staying under the wire-safety budget is
    // enforced by a compile-time `const _: () = assert!(...)` right next to
    // its definition — a build-time guard, not just a test-time one.

    #[test]
    fn wire_overflow_error_is_recognised_by_either_observed_shape() {
        assert!(is_wire_overflow_error(
            "catch-up count failed: Error code 10334 (BSONObjectTooLarge): BSONObj size: \
             28234354 is invalid. Size must be between 0 and 16793600(16MB)"
        ));
        assert!(is_wire_overflow_error(
            "catch-up count failed: Error code 17 (ProtocolError): PlanExecutor error during \
             aggregation :: caused by :: recv(): message msgLen 63982277 is invalid. Min 16 \
             Max: 48000000"
        ));
        assert!(
            !is_wire_overflow_error("catch-up count failed: index 'vidx' is not queryable yet"),
            "an unrelated probe error must not trigger bisection — it cannot help"
        );
    }

    /// The count reader must not depend on the integer width `$count` happens
    /// to emit, and must not turn a malformed reply into a plausible-looking
    /// "0 searchable" — which is indistinguishable from a cold index and would
    /// burn the entire budget before failing.
    #[test]
    fn catchup_count_reads_either_integer_width_and_rejects_a_missing_count() {
        assert_eq!(
            catchup_count_from_doc(&doc! { "n": 10_001i32 }).unwrap(),
            10_001
        );
        assert_eq!(
            catchup_count_from_doc(&doc! { "n": 1_183_514i64 }).unwrap(),
            1_183_514
        );
        let err = catchup_count_from_doc(&doc! { "total": 5i32 })
            .expect_err("a reply without `n` is not a zero count");
        assert!(err.contains("no `n`"), "{err}");
    }
}
