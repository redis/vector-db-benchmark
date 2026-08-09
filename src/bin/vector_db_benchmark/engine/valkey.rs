//! Valkey engine implementation.
//!
//! Implements the Engine trait for Valkey Search vector similarity.
//! Valkey is a Redis fork that speaks the same RESP protocol and supports
//! FT.* search commands via the Valkey Search module.
//!
//! # Why `redis` crate instead of Valkey GLIDE?
//!
//! | Option              | Status                                          |
//! |---------------------|-------------------------------------------------|
//! | `valkey-glide` Rust | No published crate. GitHub issue                |
//! |                     | valkey-io/valkey-glide#828 closed NOT_PLANNED.   |
//! |                     | Supported langs: Java, Python, Node.js, Go.     |
//! |                     | Rust is not on the roadmap.                      |
//! | `redis` crate       | Recommended by GLIDE maintainers for Rust.       |
//! |                     | GLIDE team upstreams improvements to redis-rs.   |
//! |                     | Works with Valkey via RESP protocol compat.      |
//!
//! Reference: <https://github.com/valkey-io/valkey-glide/issues/828>

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::geo;
use super::redis_utils;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::index_naming::{derive_index_name, derive_key_prefix};
use crate::engine::{CorpusCount, Engine, SearchResults, UpdateSearchRatio, UploadStats};
use vector_db_benchmark::parsers::{datetime_to_epoch_secs, doc_key_to_id, doc_key_to_id_opt};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};
use vector_db_benchmark::start_gate::WorkerPool;

/// Valkey engine configuration
#[derive(Clone)]
pub struct ValkeyEngineConfig {
    pub m: i64,
    pub ef_construction: i64,
    pub data_type: String,
    pub algorithm: String,
    pub batch_size: usize,
    pub parallel: usize,
    pub skip_vector_index: bool,
    /// Schema fields declared `datetime`: their ISO-8601 payload values are
    /// converted to epoch seconds at HSET time so the NUMERIC index/range
    /// filters match. Populated in `configure()`. `Arc` so per-thread config
    /// clones share one set.
    pub datetime_fields: Arc<HashSet<String>>,
    /// Schema fields declared `text`. Valkey Search has no TEXT field type, so
    /// full-text is DEGRADED to a whitespace-tokenised multi-value TAG: on upload
    /// the text is split on whitespace and stored as a `;`-joined TAG set, and a
    /// `{"match":{"text":"word"}}` query becomes a single-token TAG match. This
    /// supports single-term full-text matching (any doc containing the term); it
    /// does NOT support phrase / multi-term full-text queries.
    pub text_tag_fields: Arc<HashSet<String>>,
    /// Per-config index name (`"<base>:<config>"`, issue #151-4) so a sweep's
    /// configs address disjoint indexes on one server. Resolved once in `new()`.
    pub index_name: String,
    /// Per-config key prefix (`"<config>:"`, issue #151-4). Each config owns a
    /// disjoint keyspace; teardown is a prefix-scoped SCAN+UNLINK (no DD flag).
    pub key_prefix: String,
}

pub struct ValkeyEngine {
    name: String,
    host: String,
    port: u16,
    config: ValkeyEngineConfig,
    search_params: Vec<SearchParams>,
    commandstats_baseline: Option<redis_utils::CommandStatsBaseline>,
    /// Whether `commandstats_baseline` was established by this process.
    ///
    /// `None` is ambiguous on its own: it means BOTH "CONFIG RESETSTAT succeeded,
    /// so the server counters start at zero" and "configure() never ran, so
    /// nothing was reset". On the `--skip-upload` path (#238) configure() is
    /// skipped, and `check_commandstats` would then compare this run's failure
    /// count against zero while the counters still hold every failure since the
    /// server started — failing a search in which nothing actually failed. This
    /// flag disambiguates so `search()` can prime the baseline exactly once.
    commandstats_primed: bool,
}

impl ValkeyEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("VALKEY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6379);

        // Extract HNSW config
        let (m, ef_construction) = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref())
            .map(|h| (h.m.unwrap_or(16), h.ef_construction.unwrap_or(128)))
            .unwrap_or((16, 128));

        let algorithm = engine_config
            .algorithm
            .clone()
            .unwrap_or_else(|| "hnsw".to_string());

        let data_type = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("data_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("FLOAT32")
            .to_string();

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(100) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(64) as usize;

        Ok(Self {
            name: engine_config.name.clone(),
            host: host.to_string(),
            port,
            config: ValkeyEngineConfig {
                m,
                ef_construction,
                data_type,
                algorithm,
                batch_size,
                parallel,
                skip_vector_index: engine_config.skip_vector_index,
                datetime_fields: Arc::new(HashSet::new()),
                text_tag_fields: Arc::new(HashSet::new()),
                index_name: derive_index_name("VALKEY_INDEX_NAME", "idx", &engine_config.name),
                key_prefix: derive_key_prefix(&engine_config.name),
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            commandstats_baseline: None,
            commandstats_primed: false,
        })
    }

    fn get_connection(&self) -> Result<Connection, String> {
        Self::connect(&self.host, self.port)
    }

    fn connect(host: &str, port: u16) -> Result<Connection, String> {
        let auth = std::env::var("VALKEY_AUTH").ok();
        let user = std::env::var("VALKEY_USER").ok();

        let auth_part = match (&user, &auth) {
            (Some(u), Some(p)) => format!("{}:{}@", u, p),
            (None, Some(p)) => format!(":{}@", p),
            _ => String::new(),
        };

        let url = format!(
            "redis://{}{}:{}/{}",
            auth_part,
            host,
            port,
            valkey_url_suffix()
        );
        let client = redis::Client::open(url.as_str()).map_err(|e| e.to_string())?;
        let conn = client.get_connection().map_err(|e| e.to_string())?;
        // Safety timeout: prevents indefinite hangs from pipeline stalls.
        let timeout = std::time::Duration::from_secs(300);
        conn.set_read_timeout(Some(timeout)).ok();
        conn.set_write_timeout(Some(timeout)).ok();
        Ok(conn)
    }

    fn create_index(&self, conn: &mut Connection, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        // Drop this config's index + keys ONLY (Valkey Search has no DD flag, so a
        // prefix-scoped SCAN+UNLINK replaces the old keyspace-wide FLUSHALL). Under
        // #151-4 coexistence a FLUSHALL would wipe sibling configs' data.
        redis_utils::drop_index_and_keys(conn, &self.config.index_name, &self.config.key_prefix);

        // Map distance metric
        let distance_metric = map_distance_metric(distance);

        // Build FT.CREATE command
        let mut cmd = redis::cmd("FT.CREATE");
        cmd.arg(&self.config.index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg(&self.config.key_prefix);

        cmd.arg("SCHEMA");

        // Vector field with HNSW params
        // Skipped when skip_vector_index is set (filter-only benchmark)
        if !self.config.skip_vector_index {
            let num_attrs = 6 + 2 + 2; // TYPE+DIM+DISTANCE_METRIC + M + EF_CONSTRUCTION
            cmd.arg("vector")
                .arg("VECTOR")
                .arg(self.config.algorithm.to_uppercase())
                .arg(num_attrs);
            cmd.arg("TYPE").arg(&self.config.data_type);
            cmd.arg("DIM").arg(vector_size);
            cmd.arg("DISTANCE_METRIC").arg(distance_metric);
            cmd.arg("M").arg(self.config.m);
            cmd.arg("EF_CONSTRUCTION").arg(self.config.ef_construction);
        }

        // Add schema fields from dataset config for filtering.
        // Note: Valkey Search does not support SORTABLE, TEXT, or GEO field types.
        // Only TAG and NUMERIC are supported as filter fields.
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    match ft {
                        // keyword/uuid/bool are exact-match TAG fields.
                        "keyword" | "uuid" | "bool" => {
                            cmd.arg(field_name).arg("TAG").arg("SEPARATOR").arg(";");
                        }
                        "int" | "float" => {
                            cmd.arg(field_name).arg("NUMERIC");
                        }
                        // datetime is stored as epoch seconds (see upload) → NUMERIC.
                        "datetime" => {
                            cmd.arg(field_name).arg("NUMERIC");
                        }
                        // Valkey Search has no TEXT field type: DEGRADE full-text
                        // to a whitespace-tokenised multi-value TAG (see
                        // ValkeyEngineConfig::text_tag_fields). Single-term
                        // full-text matching works; phrase queries do not.
                        "text" => {
                            cmd.arg(field_name).arg("TAG").arg("SEPARATOR").arg(";");
                        }
                        // Valkey Search has no GEO field type, so `geo` is
                        // deliberately not declared here — and, since #223,
                        // `parse_conditions` refuses a geo condition outright
                        // rather than emitting a clause against a field this
                        // loop never created.
                        _ => {}
                    }
                }
            }
        }

        cmd.query::<()>(conn)
            .map_err(|e| format!("Failed to create index: {}", e))?;

        Ok(())
    }

    fn upload_sequential(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let pb = self.create_progress_bar(ids.len());

        for batch_start in (0..ids.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(ids.len());
            self.upload_batch(
                &mut conn,
                &ids[batch_start..batch_end],
                &vectors[batch_start..batch_end],
                &metadata[batch_start..batch_end],
            )?;
            pb.inc((batch_end - batch_start) as u64);
        }

        pb.finish_with_message("Upload complete");
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
        let num_threads = self.config.parallel.min(total_batches);
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let host = self.host.clone();
                let port = self.port;
                let config = self.config.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match ValkeyEngine::connect(&host, port) {
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
                        let (batch_start, batch_end) = batches[idx];
                        if let Err(e) = upload_batch_internal(
                            &mut conn,
                            &config,
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

    fn upload_batch(
        &self,
        conn: &mut Connection,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        upload_batch_internal(conn, &self.config, ids, vectors, metadata)
    }

    /// Wait until FT.INFO reports num_docs >= expected and indexing/backfill is done.
    ///
    /// Checks both Redis Search's `indexing` flag and Valkey Search's
    /// `backfill_in_progress` / `state` fields so this works with either engine.
    fn wait_for_indexing(&self, expected: usize) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let max_wait = 600; // seconds – large HNSW indices can take minutes
        let start = Instant::now();

        loop {
            let info: redis::Value = redis::cmd("FT.INFO")
                .arg(&self.config.index_name)
                .query(&mut conn)
                .map_err(|e| format!("FT.INFO error: {}", e))?;

            let mut num_docs: usize = 0;
            let mut indexing: bool = false;

            fn extract_usize(val: &redis::Value) -> usize {
                match val {
                    redis::Value::BulkString(s) => String::from_utf8_lossy(s).parse().unwrap_or(0),
                    redis::Value::Int(n) => *n as usize,
                    redis::Value::Double(f) => *f as usize,
                    redis::Value::SimpleString(s) => s.parse().unwrap_or(0),
                    _ => 0,
                }
            }

            fn extract_bool_nonzero(val: &redis::Value) -> bool {
                match val {
                    redis::Value::BulkString(s) => s != b"0",
                    redis::Value::Int(n) => *n != 0,
                    redis::Value::Double(f) => *f != 0.0,
                    redis::Value::SimpleString(s) => s != "0",
                    redis::Value::Boolean(b) => *b,
                    _ => false,
                }
            }

            fn extract_string(val: &redis::Value) -> String {
                match val {
                    redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                    redis::Value::SimpleString(s) => s.clone(),
                    _ => String::new(),
                }
            }

            match &info {
                redis::Value::Array(arr) => {
                    for i in (0..arr.len()).step_by(2) {
                        let key_str = match &arr[i] {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if let Some(val) = arr.get(i + 1) {
                            match key_str.as_str() {
                                "num_docs" => num_docs = extract_usize(val),
                                // Redis Search field
                                "indexing" => indexing = indexing || extract_bool_nonzero(val),
                                // Valkey Search fields
                                "backfill_in_progress" => {
                                    indexing = indexing || extract_bool_nonzero(val)
                                }
                                "state" => {
                                    let state = extract_string(val);
                                    if state != "ready" && !state.is_empty() {
                                        indexing = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                redis::Value::Map(map) => {
                    for (k, v) in map {
                        let key_str = match k {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        match key_str.as_str() {
                            "num_docs" => num_docs = extract_usize(v),
                            "indexing" => indexing = indexing || extract_bool_nonzero(v),
                            "backfill_in_progress" => {
                                indexing = indexing || extract_bool_nonzero(v)
                            }
                            "state" => {
                                let state = extract_string(v);
                                if state != "ready" && !state.is_empty() {
                                    indexing = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    eprintln!("Unexpected FT.INFO response type: {:?}", info);
                }
            }

            if num_docs >= expected && !indexing {
                println!(
                    "Indexing complete: {} docs in {:.1}s",
                    num_docs,
                    start.elapsed().as_secs_f64()
                );
                return Ok(());
            }

            if start.elapsed().as_secs() > max_wait {
                println!(
                    "Warning: indexing timeout after {}s (num_docs={}/{}, indexing={})",
                    max_wait, num_docs, expected, indexing
                );
                return Ok(());
            }

            if start.elapsed().as_secs().is_multiple_of(10) && start.elapsed().as_secs() > 0 {
                println!(
                    "Waiting for indexing: {} docs, indexing={} ({:.0}s)",
                    num_docs,
                    indexing,
                    start.elapsed().as_secs_f64()
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    /// Filter-only search: run FT.SEARCH with filter conditions only (no KNN).
    fn search_filter_only(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        // Index-existence guard (#151-4): hard-error on a missing/mismatched index.
        {
            let mut conn = self.get_connection()?;
            redis_utils::ensure_index_exists(&mut conn, &self.config.index_name)?;
        }

        let parallel = params.parallel.unwrap_or(1) as usize;
        let query_timeout: i64 = std::env::var("VALKEY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (_queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<QueryFilter<ParsedFilter>> =
            conditions.resolve_all("Valkey", parse_conditions)?;

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

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);

        // Resolve the per-config index name once (not per query / per worker).
        let index_name = self.config.index_name.clone();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let parsed_filters = &parsed_filters;
                let runnable_indices = &runnable_indices;
                let neighbors = &neighbors;
                let errors = Arc::clone(&errors);
                let query_idx = Arc::clone(&query_idx);
                let index_name = index_name.as_str();
                let pb = &pb;

                handles.push(s.spawn(move || {
                    // Thread-local sample buffer — no cross-thread lock per query.
                    let mut t: Vec<f64> = Vec::new();
                    let mut local_errs: Vec<String> = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let auth = std::env::var("VALKEY_AUTH").ok();
                    let user = std::env::var("VALKEY_USER").ok();
                    let auth_part = match (&user, &auth) {
                        (Some(u), Some(p)) => format!("{}:{}@", u, p),
                        (None, Some(p)) => format!(":{}@", p),
                        _ => String::new(),
                    };
                    let url = format!(
                        "redis://{}{}:{}/{}",
                        auth_part,
                        host,
                        port,
                        valkey_url_suffix()
                    );
                    let client = match redis::Client::open(url.as_str()) {
                        Ok(c) => c,
                        Err(_) => return t,
                    };
                    let mut conn = match client.get_connection() {
                        Ok(c) => c,
                        Err(_) => return t,
                    };

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

                        let query_start = Instant::now();
                        let result = ft_search_filter_only(
                            &mut conn,
                            index_name,
                            top,
                            query_timeout,
                            parsed_filters[idx].as_ref().unwrap(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        // Record a latency sample only for successful queries, so a
                        // failed FT.SEARCH is counted as a failure (num_to_run minus
                        // successes) rather than folded into RPS/percentiles — parity
                        // with the main search() path.
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

        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH"],
            "search",
            self.commandstats_baseline.as_ref(),
        )?;

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
}

/// Maximum RESP wire bytes per pipeline flush.
/// Valkey Search HNSW indexing can stall pipelines when the payload fills
/// the TCP send buffer (default 16 KB on Linux). With concurrent upload
/// threads the server must interleave reads across connections, amplifying
/// the effect. Keeping each sub-batch well below the TCP buffer prevents
/// blocking writes while still amortising round-trip overhead.
const MAX_PIPE_BYTES: usize = 4_096;

/// Internal batch upload function.
///
/// Sends HSET commands in sub-batched pipelines whose total serialised size
/// stays under `MAX_PIPE_BYTES`. This avoids a known interaction between
/// the Rust `redis` crate's synchronous pipeline writer and Valkey Search's
/// HNSW indexing that can stall large single-write pipelines.
fn upload_batch_internal(
    conn: &mut Connection,
    config: &ValkeyEngineConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();
    let mut pipe_bytes: usize = 0;

    for i in 0..ids.len() {
        let key = format!("{}{}", config.key_prefix, ids[i]);
        let vec_bytes: Vec<u8> = match config.data_type.as_str() {
            "FLOAT64" => vectors[i]
                .iter()
                .map(|&f| f as f64)
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            "FLOAT16" => vectors[i]
                .iter()
                .map(|&f| half::f16::from_f32(f).to_bits())
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            "BFLOAT16" => vectors[i]
                .iter()
                .map(|&f| half::bf16::from_f32(f).to_bits())
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            _ => vectors[i].iter().flat_map(|f| f.to_le_bytes()).collect(),
        };

        let mut fields: Vec<(Vec<u8>, Vec<u8>)> = vec![("vector".as_bytes().to_vec(), vec_bytes)];

        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                match v {
                    MetadataValue::String(s) => {
                        fields.push((
                            k.as_bytes().to_vec(),
                            encode_string_field(config, k, s).into_bytes(),
                        ));
                    }
                    MetadataValue::Int(n) => {
                        fields.push((k.as_bytes().to_vec(), n.to_string().into_bytes()));
                    }
                    MetadataValue::Float(f) => {
                        fields.push((k.as_bytes().to_vec(), f.to_string().into_bytes()));
                    }
                    MetadataValue::Labels(labels) => {
                        fields.push((k.as_bytes().to_vec(), labels.join(";").into_bytes()));
                    }
                    MetadataValue::Geo { lon, lat } => {
                        let lat_clamped = lat.clamp(-85.05112878, 85.05112878);
                        let geo_str = format!("{},{}", lon, lat_clamped);
                        fields.push((k.as_bytes().to_vec(), geo_str.into_bytes()));
                    }
                }
            }
        }

        // Estimate RESP wire size: *N\r\n + $4\r\nHSET\r\n + $K\r\nkey\r\n + fields
        let num_args = 1 + 1 + fields.len() * 2; // HSET + key + (field_name, field_value)*
        let mut cmd_bytes = format!("*{}\r\n", num_args).len()
            + 10 // $4\r\nHSET\r\n
            + format!("${}\r\n", key.len()).len() + key.len() + 2;
        for (fk, fv) in &fields {
            cmd_bytes += format!("${}\r\n", fk.len()).len() + fk.len() + 2;
            cmd_bytes += format!("${}\r\n", fv.len()).len() + fv.len() + 2;
        }

        // Flush the current pipeline if adding this command would exceed the limit
        if pipe_bytes > 0 && pipe_bytes + cmd_bytes > MAX_PIPE_BYTES {
            pipe.query::<()>(conn).map_err(|e| e.to_string())?;
            pipe = redis::pipe();
            pipe_bytes = 0;
        }

        let mut hset_cmd = redis::cmd("HSET");
        hset_cmd.arg(key.as_str());
        for (field_key, field_val) in &fields {
            hset_cmd.arg(&field_key[..]).arg(&field_val[..]);
        }
        pipe.add_command(hset_cmd).ignore();
        pipe_bytes += cmd_bytes;
    }

    // Flush remaining commands
    if pipe_bytes > 0 {
        pipe.query::<()>(conn).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Convert a redis::Value to serde_json::Value for serialization.
fn redis_value_to_json(val: &redis::Value) -> serde_json::Value {
    match val {
        redis::Value::Nil => serde_json::Value::Null,
        redis::Value::Int(n) => serde_json::json!(n),
        redis::Value::Double(f) => serde_json::json!(f),
        redis::Value::Boolean(b) => serde_json::json!(b),
        redis::Value::SimpleString(s) => serde_json::json!(s),
        redis::Value::BulkString(bytes) => match String::from_utf8(bytes.clone()) {
            Ok(s) => serde_json::json!(s),
            Err(_) => serde_json::json!(format!("<{} bytes>", bytes.len())),
        },
        redis::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(redis_value_to_json).collect())
        }
        redis::Value::Map(pairs) => {
            let mut map = serde_json::Map::new();
            for (k, v) in pairs {
                let key = match k {
                    redis::Value::SimpleString(s) => s.clone(),
                    redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
                    other => format!("{:?}", other),
                };
                map.insert(key, redis_value_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        other => serde_json::json!(format!("{:?}", other)),
    }
}

/// Parse the `used_memory:` value (bytes) out of an `INFO memory` text block.
/// Returns 0 when the line is absent or unparseable. The `used_memory:` prefix
/// is exact, so it never matches sibling keys like `used_memory_rss:`.
fn parse_used_memory(info: &str) -> i64 {
    info.lines()
        .find(|l| l.starts_with("used_memory:"))
        .and_then(|l| l.strip_prefix("used_memory:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

// ── Condition parser ─────────────────────────────────────────────────────
// Converts JSON filter conditions into Valkey Search query filter syntax.
// Note: Valkey Search does not support $param inside TAG {…} brackets,
// so TAG values are inlined directly (with escaping). Numeric and geo
// filters still use parameterised PARAMS.

#[derive(Debug, Clone)]
pub(crate) enum FilterParamValue {
    Int(i64),
    Float(f64),
}

type ParsedFilter = (String, HashMap<String, FilterParamValue>);

/// Map a dataset distance name to the Valkey Search `DISTANCE_METRIC` value.
/// Unknown metrics default to `COSINE` (matches the historical inline behavior).
/// A typo here (e.g. IP→L2) would silently invert ranking, so it is unit-tested.
fn map_distance_metric(distance: &str) -> &'static str {
    match distance.to_lowercase().as_str() {
        "cosine" | "angular" => "COSINE",
        "euclidean" | "l2" => "L2",
        "dot" | "ip" => "IP",
        _ => "COSINE",
    }
}

pub(crate) fn parse_conditions(conditions: &serde_json::Value) -> Option<ParsedFilter> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }
    // Geo is refused for the WHOLE tree (issue #223), mirroring dragonfly.
    //
    // `configure()` never adds a GEO field to `FT.CREATE` — see the
    // `"geo" => {}` note in the schema loop — but `build_leaf` still emitted
    // `@f:[$lon $lat $r m]`, so a geo dataset sent a clause naming a field that
    // is not in the index. Valkey Search rejects it at query time
    // (`Invalid filter expression: 'location' is not indexed as a numeric
    // field`, verified live on valkey/valkey-bundle), so this was a mid-run
    // failure after a full ingest rather than silent wrong recall — but it was
    // also an ASSERTED-green cell in `engine/filter_guard.rs`, whose dragonfly
    // column had the identical shape.
    //
    // Whole-tree, not per-leaf: this builder keeps the leaves it understands, so
    // `and(geo, keyword)` would otherwise emit the keyword clause alone and
    // under-constrain the query — the partial-drop escape `query_filter.rs`
    // documents. Refusing here makes `query_filter::resolve` raise #219's error
    // up front instead.
    if geo::conditions_mention_geo(conditions) {
        return None;
    }

    let mut counter: usize = 0;
    build_group(obj, &mut counter)
}

/// Build one boolean GROUP (`{and:[...], or:[...]}`) into a parenthesised
/// Valkey-Search clause. Recursive with [`build_subfilters`] so a nested group
/// inside `and`/`or` becomes its own parenthesised sub-clause; `counter` is
/// shared across the tree to keep param placeholders unique.
fn build_group(
    obj: &serde_json::Map<String, serde_json::Value>,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let and_entries = obj.get("and").and_then(|v| v.as_array());
    let or_entries = obj.get("or").and_then(|v| v.as_array());

    let and_subfilters = and_entries.map(|entries| build_subfilters(entries, counter));
    let or_subfilters = or_entries.map(|entries| build_subfilters(entries, counter));

    build_condition(and_subfilters, or_subfilters)
}

fn build_subfilters(entries: &[serde_json::Value], counter: &mut usize) -> Vec<ParsedFilter> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            // Nested group: an entry carrying an `and`/`or` key is a sub-tree,
            // not a field leaf. Recurse and keep it as one parenthesised clause.
            if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
                if let Some(f) = build_group(entry_obj, counter) {
                    filters.push(f);
                }
                continue;
            }
            for (field_name, field_filters) in entry_obj {
                if let Some(filter_obj) = field_filters.as_object() {
                    for (condition_type, criteria) in filter_obj {
                        if let Some(f) = build_filter(field_name, condition_type, criteria, counter)
                        {
                            filters.push(f);
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
    counter: &mut usize,
) -> Option<ParsedFilter> {
    match condition_type {
        "match" => {
            // match_any (IN-list) > exact {value} > full-text {text}.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                Some(build_match_any_filter(field_name, any, counter))
            } else if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                Some(build_text_filter(field_name, text))
            } else {
                build_exact_match_filter(field_name, criteria, counter)
            }
        }
        "range" => build_range_filter(field_name, criteria, counter),
        // Unreachable in production: `parse_conditions` refuses the whole tree
        // before any leaf is built (#223). Kept so `build_geo_filter`'s own unit
        // tests still pin the RediSearch spelling, in case Valkey Search gains a
        // GEO field type and this becomes live again.
        "geo" => build_geo_filter(field_name, criteria, counter),
        _ => None,
    }
}

/// Escape the TAG-structural characters when inlining a value into a `{…}`
/// clause (Valkey Search does not accept `$param` refs inside TAG brackets, so
/// values are inlined). Only the characters that would otherwise break the OR
/// (`|`) or the braces are escaped; other characters are passed raw, matching
/// the exact-match path.
fn escape_tag_value(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('{', "\\{")
        .replace('}', "\\}")
}

/// Build a `match_any` (IN-list) filter, the OR-of-values semantics that mirror
/// qdrant's `Condition::matches(field, Vec)`.
///
/// - All-integer list -> NUMERIC OR `(@f:[$a $a] | @f:[$b $b])` (params).
/// - Otherwise -> TAG OR `@f:{a | b}` over the non-empty string values, inlined
///   (Valkey Search rejects `$param` inside TAG `{…}`). Empty-string tokens are
///   dropped (invalid TAG syntax).
/// - Empty / no representable values -> a never-match `(@f:{s} -@f:{s})`
///   contradiction so an empty IN-set matches NOTHING rather than being dropped
///   (which, as the sole clause, would run kNN over ALL docs). Assumes a TAG
///   field, the realistic case for a keyword IN-list.
///
/// NOTE: Valkey Search TAG matching is case-INSENSITIVE, whereas qdrant keyword
/// match is case-sensitive; all shipped keyword datasets use consistent casing.
fn build_match_any_filter(
    field_name: &str,
    any: &[serde_json::Value],
    counter: &mut usize,
) -> ParsedFilter {
    // All values on the match_any path are inlined (valkey supports neither
    // $param in TAG {…} nor in NUMERIC […]); params stays empty.
    let params = HashMap::new();

    if !any.is_empty() && any.iter().all(|v| v.is_i64()) {
        let clauses: Vec<String> = any
            .iter()
            .filter_map(|v| v.as_i64())
            .map(|i| {
                // Inline the literal — valkey rejects `$param` inside NUMERIC
                // brackets (see build_range_filter).
                *counter += 1;
                format!("@{}:[{} {}]", field_name, i, i)
            })
            .collect();
        return (format!("({})", clauses.join(" | ")), params);
    }

    let tokens: Vec<String> = any
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(escape_tag_value)
        .collect();

    if tokens.is_empty() {
        let never = "__match_any_never_match__";
        return (
            format!("(@{0}:{{{1}}} -@{0}:{{{1}}})", field_name, never),
            params,
        );
    }

    (
        format!("@{}:{{{}}}", field_name, tokens.join(" | ")),
        params,
    )
}

fn build_exact_match_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let value = criteria.get("value")?;
    // Bump the counter for param-name parity with sibling filters — every value
    // here is INLINED: Valkey Search supports neither `$param` inside TAG {…} nor
    // inside NUMERIC […] brackets, so params are never used on this path.
    *counter += 1;
    let params = HashMap::new();

    // bool → inlined TAG match on the literal "true"/"false" token (no escaping
    // needed). Checked before numeric arms (serde bools are neither i64/f64/str).
    if let Some(b) = value.as_bool() {
        let token = if b { "true" } else { "false" };
        return Some((format!("@{}:{{{}}}", field_name, token), params));
    }
    // keyword/uuid string → inlined, ESCAPED TAG match (must escape TAG
    // metacharacters `| { } ( ) space \` exactly like build_match_any_filter /
    // build_text_filter, else a value with those chars yields a malformed query
    // or the wrong document set).
    if let Some(s) = value.as_str() {
        return Some((
            format!("@{}:{{{}}}", field_name, escape_tag_value(s)),
            params,
        ));
    }
    // numeric (int/float) → inlined NUMERIC point range `[v v]` (literals, not
    // `$param`, for the same reason as build_range_filter).
    if let Some(lit) = number_literal(value) {
        return Some((format!("@{}:[{} {}]", field_name, lit, lit), params));
    }
    None
}

/// Build a full-text filter — DEGRADED on Valkey Search (no TEXT field type) to
/// a single-token TAG match. The `text` field is stored as a whitespace-
/// tokenised TAG multi-value on upload (see `tokenize_text_to_tag`), so a
/// single-term query `@field:{term}` matches any doc whose text contains that
/// term. The term is inlined (Valkey rejects `$param` inside TAG `{…}`) and its
/// TAG-structural characters escaped. A blank query degrades to a never-match
/// contradiction rather than an empty clause (which would run kNN over ALL docs).
///
/// LIMITATION: only single-term matching is supported; a multi-word `text` query
/// is treated as one TAG token and will generally not match (phrase / multi-term
/// full-text is not available on Valkey Search).
fn build_text_filter(field_name: &str, text: &str) -> ParsedFilter {
    let params = HashMap::new();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        let never = "__text_never_match__";
        return (
            format!("(@{0}:{{{1}}} -@{0}:{{{1}}})", field_name, never),
            params,
        );
    }
    // Use the first whitespace token; escape TAG-structural chars.
    let term = trimmed.split_whitespace().next().unwrap_or(trimmed);
    (
        format!("@{}:{{{}}}", field_name, escape_tag_value(term)),
        params,
    )
}

fn build_range_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    // Bump the counter for param-name parity with sibling filters even though the
    // bounds are INLINED: Valkey Search does not substitute `$param` inside
    // NUMERIC range brackets (`@f:[$p +inf]` → "Invalid number"); only literal
    // numbers are accepted there (unlike RediSearch). Bounds are numbers /
    // epochs from trusted dataset conditions, so inlining is safe.
    *counter += 1;

    let mut clauses = Vec::new();
    if let Some(v) = criteria.get("lt").and_then(number_literal) {
        clauses.push(format!("@{}:[-inf ({}]", field_name, v));
    }
    if let Some(v) = criteria.get("gt").and_then(number_literal) {
        clauses.push(format!("@{}:[({} +inf]", field_name, v));
    }
    if let Some(v) = criteria.get("lte").and_then(number_literal) {
        clauses.push(format!("@{}:[-inf {}]", field_name, v));
    }
    if let Some(v) = criteria.get("gte").and_then(number_literal) {
        clauses.push(format!("@{}:[{} +inf]", field_name, v));
    }

    if clauses.is_empty() {
        return None;
    }
    Some((clauses.join(" "), HashMap::new()))
}

/// Render a JSON number / ISO-8601 datetime / numeric string as a literal NUMERIC
/// bound (integers verbatim, ISO-8601 → epoch seconds, numeric strings as-is).
fn number_literal(value: &serde_json::Value) -> Option<String> {
    if let Some(i) = value.as_i64() {
        Some(i.to_string())
    } else if let Some(f) = value.as_f64() {
        Some(format!("{}", f))
    } else if let Some(s) = value.as_str() {
        if let Some(epoch) = datetime_to_epoch_secs(s) {
            Some((epoch as i64).to_string())
        } else if s.parse::<f64>().is_ok() {
            Some(s.to_string())
        } else {
            None
        }
    } else {
        None
    }
}

fn build_geo_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let param_prefix = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();

    let lon_name = format!("{}_lon", param_prefix);
    let lat_name = format!("{}_lat", param_prefix);
    let radius_name = format!("{}_radius", param_prefix);

    insert_number_param(&mut params, &lon_name, criteria.get("lon")?);
    insert_number_param(&mut params, &lat_name, criteria.get("lat")?);
    insert_number_param(&mut params, &radius_name, criteria.get("radius")?);

    Some((
        format!(
            "@{}:[${} ${} ${} m]",
            field_name, lon_name, lat_name, radius_name
        ),
        params,
    ))
}

/// Insert a JSON number (or ISO-8601 datetime / numeric string) as a NUMERIC
/// bound: integers/floats verbatim, ISO-8601 strings as epoch **seconds**, and
/// other numeric strings parsed as f64 (so both ISO and raw-epoch datetime
/// bounds work — upstream is ISO-only).
fn insert_number_param(
    params: &mut HashMap<String, FilterParamValue>,
    name: &str,
    value: &serde_json::Value,
) {
    if let Some(i) = value.as_i64() {
        params.insert(name.to_string(), FilterParamValue::Int(i));
    } else if let Some(f) = value.as_f64() {
        params.insert(name.to_string(), FilterParamValue::Float(f));
    } else if let Some(s) = value.as_str() {
        if let Some(epoch) = datetime_to_epoch_secs(s) {
            // Epoch is whole seconds — emit as an integer param. Valkey Search's
            // NUMERIC-range param substitution rejects the float rendering
            // ("Invalid number"), whereas integer params are accepted (as the
            // match_any NUMERIC path already relies on).
            params.insert(name.to_string(), FilterParamValue::Int(epoch as i64));
        } else if let Ok(i) = s.parse::<i64>() {
            params.insert(name.to_string(), FilterParamValue::Int(i));
        } else if let Ok(f) = s.parse::<f64>() {
            params.insert(name.to_string(), FilterParamValue::Float(f));
        }
    }
}

fn build_condition(
    and_subfilters: Option<Vec<ParsedFilter>>,
    or_subfilters: Option<Vec<ParsedFilter>>,
) -> Option<ParsedFilter> {
    let mut clause_parts = Vec::new();
    let mut all_params = HashMap::new();

    if let Some(and_filters) = and_subfilters {
        if !and_filters.is_empty() {
            let and_clauses: Vec<String> = and_filters.iter().map(|(c, _)| c.clone()).collect();
            for (_, p) in &and_filters {
                all_params.extend(p.clone());
            }
            clause_parts.push(format!("({})", and_clauses.join(" ")));
        }
    }

    if let Some(or_filters) = or_subfilters {
        if !or_filters.is_empty() {
            let or_clauses: Vec<String> = or_filters.iter().map(|(c, _)| c.clone()).collect();
            for (_, p) in &or_filters {
                all_params.extend(p.clone());
            }
            clause_parts.push(format!("({})", or_clauses.join(" | ")));
        }
    }

    if clause_parts.is_empty() {
        return None;
    }

    Some((clause_parts.join(" "), all_params))
}

// ── FT.SEARCH ────────────────────────────────────────────────────────────

/// Execute filter-only FT.SEARCH (no KNN vector query).
fn ft_search_filter_only(
    conn: &mut Connection,
    index_name: &str,
    top: usize,
    query_timeout: i64,
    filter: &ParsedFilter,
) -> Result<usize, String> {
    let (filter_expr, params) = filter;

    let mut cmd = redis::cmd("FT.SEARCH");
    cmd.arg(index_name)
        .arg(filter_expr.as_str())
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("DIALECT")
        .arg(2) // Valkey uses DIALECT 2
        .arg("TIMEOUT")
        .arg(query_timeout);

    if !params.is_empty() {
        cmd.arg("PARAMS").arg(params.len() * 2);
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Int(i) => {
                    cmd.arg(*i);
                }
                FilterParamValue::Float(f) => {
                    cmd.arg(*f);
                }
            }
        }
    }

    let response: Vec<redis::Value> = cmd
        .query(conn)
        .map_err(|e| format!("FT.SEARCH filter-only error: {}", e))?;

    if let Some(first) = response.first() {
        match first {
            redis::Value::Int(n) => Ok(*n as usize),
            redis::Value::BulkString(s) => Ok(String::from_utf8_lossy(s).parse().unwrap_or(0)),
            _ => Ok(0),
        }
    } else {
        Ok(0)
    }
}

/// Build the Valkey FT.SEARCH KNN query string for the given filter.
///
/// Pure client-side string formatting, kept OUT of the per-query timed window
/// (precomputed once per query before the parallel region). EF_RUNTIME is a
/// supported per-query HNSW attribute (validated by valkey-search
/// ft_search_parser.cc) — without it, every ef in the search sweep runs at the
/// index default, collapsing the precision/recall curve to a single point.
/// Passed as a `$EF` param. The query vector is bound as `$vec_param`, so this
/// string is identical across queries sharing a filter.
fn build_knn_query_str(filter: Option<&ParsedFilter>) -> String {
    let prefilter = filter.map(|(expr, _)| expr.as_str()).unwrap_or("*");
    format!(
        "{}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]",
        prefilter
    )
}

/// Encode a query vector to the FLOAT32 little-endian blob Valkey expects.
///
/// Kept as a standalone fn so the caller can precompute all query blobs BEFORE
/// the timed window (client work, not server latency).
fn encode_query_vector(query_vector: &[f32]) -> Vec<u8> {
    query_vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Execute a Valkey FT.SEARCH KNN query, return (id, score) pairs.
///
/// `vec_bytes` and `query_str` are precomputed by the caller BEFORE the timed
/// window; this performs only the arg binding, the `cmd.query` RPC round-trip,
/// and the reply parse.
#[allow(clippy::too_many_arguments)]
fn ft_search_knn(
    conn: &mut Connection,
    index_name: &str,
    vec_bytes: &[u8],
    query_str: &str,
    top: usize,
    ef: i64,
    _algorithm: &str,
    _hybrid_policy: &str,
    query_timeout: i64,
    filter: Option<&ParsedFilter>,
) -> Result<Vec<(i64, f64)>, String> {
    // Valkey Search: DIALECT 2 only, no SORTBY on computed fields
    let mut cmd = redis::cmd("FT.SEARCH");
    cmd.arg(index_name)
        .arg(query_str)
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Params: vec_param + K + EF + filter params
    let filter_param_count = filter.as_ref().map(|(_, p)| p.len() * 2).unwrap_or(0);
    let total_param_count = 6 + filter_param_count; // vec_param(2) + K(2) + EF(2) + filter params

    cmd.arg("PARAMS").arg(total_param_count);
    cmd.arg("vec_param").arg(vec_bytes);
    cmd.arg("K").arg(top.to_string());
    cmd.arg("EF").arg(ef.to_string());

    if let Some((_, params)) = filter {
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Int(i) => {
                    cmd.arg(*i);
                }
                FilterParamValue::Float(f) => {
                    cmd.arg(*f);
                }
            }
        }
    }

    // Query the raw Value (not Vec<Value>) so both a RESP2 array and a RESP3 map
    // deserialize; parse_ft_search_response dispatches on the shape.
    let response: redis::Value = cmd
        .query(conn)
        .map_err(|e| format!("FT.SEARCH error: {}", e))?;

    parse_ft_search_response(&response)
}

/// Parse an FT.SEARCH reply under EITHER protocol:
/// - RESP2: a flat array `[count, id, fields, id, fields, ...]`
/// - RESP3: a map `{ results: [ { id, extra_attributes: { vector_score, .. }, .. } ], .. }`
///
/// The engine connects with RESP2 by default, but a caller can negotiate RESP3
/// (e.g. `REDIS_URI=redis://host/?protocol=resp3`), which returns a different
/// shape. Handling both keeps recall correct regardless of protocol.
fn parse_ft_search_response(response: &redis::Value) -> Result<Vec<(i64, f64)>, String> {
    match response {
        redis::Value::Array(items) => Ok(parse_ft_search_resp2(items)),
        redis::Value::Map(pairs) => Ok(parse_ft_search_resp3(pairs)),
        _ => Ok(Vec::new()),
    }
}

/// RESP2 flat array: `[count, id, fields, id, fields, ...]`.
fn parse_ft_search_resp2(response: &[redis::Value]) -> Vec<(i64, f64)> {
    let mut results = Vec::new();
    let mut i = 1;
    while i < response.len() {
        // The reply carries the doc KEY ("<config>:<id>", #151-4); recover the
        // trailing numeric id. Missing string → 0 (positionally present).
        let id = value_as_string(&response[i])
            .map(|s| doc_key_to_id(&s))
            .unwrap_or(0);
        i += 1;

        if i < response.len() {
            let score = match &response[i] {
                redis::Value::Array(fields) => extract_vector_score(fields),
                _ => 0.0,
            };
            results.push((id, score));
            i += 1;
        }
    }
    results
}

/// RESP3 map: top-level map with a `results` array; each result is a map with an
/// `id` and an `extra_attributes` map carrying `vector_score`.
fn parse_ft_search_resp3(pairs: &[(redis::Value, redis::Value)]) -> Vec<(i64, f64)> {
    let docs = match pairs
        .iter()
        .find(|(k, _)| value_as_string(k).as_deref() == Some("results"))
        .map(|(_, v)| v)
    {
        Some(redis::Value::Array(docs)) => docs.as_slice(),
        _ => return Vec::new(),
    };

    let mut out = Vec::with_capacity(docs.len());
    for doc in docs {
        let redis::Value::Map(fields) = doc else {
            continue;
        };
        // A doc whose `id` is missing or cannot be parsed to an integer is
        // skipped (mirrors the RESP2 trailing-id drop) rather than emitted as a
        // phantom id=0 hit.
        let mut id: Option<i64> = None;
        let mut score = 0.0f64;
        for (k, v) in fields {
            match value_as_string(k).as_deref() {
                Some("id") => id = value_as_string(v).and_then(|s| doc_key_to_id_opt(&s)),
                Some("extra_attributes") => {
                    if let redis::Value::Map(attrs) = v {
                        for (ak, av) in attrs {
                            if value_as_string(ak).as_deref() == Some("vector_score") {
                                score = value_as_string(av)
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(0.0);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(id) = id {
            out.push((id, score));
        }
    }
    out
}

/// Best-effort string view of a RESP value (BulkString/SimpleString).
fn value_as_string(v: &redis::Value) -> Option<String> {
    match v {
        redis::Value::BulkString(b) => Some(String::from_utf8_lossy(b).into_owned()),
        redis::Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

/// Parse a RESP value as an i64 doc id (bulk/simple string or integer).
/// Retained for its unit test; the resp2 hot path now goes through
/// `doc_key_to_id` (#151-4) to strip the per-config key prefix, so this is
/// test-only.
#[cfg(test)]
fn value_as_i64(v: &redis::Value) -> i64 {
    match v {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap_or(0),
        redis::Value::Int(n) => *n,
        redis::Value::SimpleString(s) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

/// Optional connection-URL suffix so Valkey can be benchmarked over RESP3
/// (`VALKEY_PROTOCOL=resp3`). Defaults to RESP2 (empty suffix). The FT.SEARCH
/// response parser handles both shapes, so recall is identical either way.
fn valkey_url_suffix() -> &'static str {
    if std::env::var("VALKEY_PROTOCOL")
        .map(|v| v.eq_ignore_ascii_case("resp3"))
        .unwrap_or(false)
    {
        "?protocol=resp3"
    } else {
        ""
    }
}

fn extract_vector_score(fields: &[redis::Value]) -> f64 {
    let mut i = 0;
    while i + 1 < fields.len() {
        if let redis::Value::BulkString(name) = &fields[i] {
            if name == b"vector_score" {
                if let redis::Value::BulkString(val) = &fields[i + 1] {
                    return String::from_utf8_lossy(val).parse::<f64>().unwrap_or(0.0);
                }
            }
        }
        i += 2;
    }
    0.0
}

/// Single-record HSET update (for mixed benchmark).
///
/// Returns `Ok(true)` when the server reports the target document **did not
/// exist**: `HSET` replies with the number of NEW fields, so a reply equal to
/// the number of fields written means nothing was there (verified live). A
/// reply of 0 is a clean overwrite; a reply strictly between the two means the
/// document existed with a different field set (schema drift), which is
/// deliberately NOT treated as a missed write — see the comment at the reply.
///
/// LOAD-BEARING PARITY: `upload_batch_internal` must keep writing the same
/// field NAMES as this function (both are `"vector"` plus `meta.fields`, from
/// the same `dataset.read_vectors()` call). Adding a field to one path only
/// would not break the #293 signal — the all-or-nothing test above tolerates a
/// partial overlap — but it would make the two halves disagree about the
/// document's shape, which is worth knowing before you edit either.
fn hset_single(
    conn: &mut Connection,
    config: &ValkeyEngineConfig,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
) -> Result<bool, String> {
    let key = format!("{}{}", config.key_prefix, id);
    let vec_bytes: Vec<u8> = match config.data_type.as_str() {
        "FLOAT64" => vector
            .iter()
            .map(|&f| f as f64)
            .flat_map(|f| f.to_le_bytes())
            .collect(),
        "FLOAT16" => vector
            .iter()
            .map(|&f| half::f16::from_f32(f).to_bits())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        "BFLOAT16" => vector
            .iter()
            .map(|&f| half::bf16::from_f32(f).to_bits())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        _ => vector.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };

    let mut cmd = redis::cmd("HSET");
    cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);
    let mut written_fields: i64 = 1;

    if let Some(meta) = metadata {
        for (k, v) in &meta.fields {
            written_fields += 1;
            match v {
                MetadataValue::String(s) => {
                    cmd.arg(k.as_str()).arg(encode_string_field(config, k, s));
                }
                MetadataValue::Int(n) => {
                    cmd.arg(k.as_str()).arg(n.to_string());
                }
                MetadataValue::Float(f) => {
                    cmd.arg(k.as_str()).arg(f.to_string());
                }
                MetadataValue::Labels(labels) => {
                    cmd.arg(k.as_str()).arg(labels.join(";"));
                }
                MetadataValue::Geo { lon, lat } => {
                    let lat_clamped = lat.clamp(-85.05112878, 85.05112878);
                    cmd.arg(k.as_str()).arg(format!("{},{}", lon, lat_clamped));
                }
            }
        }
    }

    let new_fields: i64 = cmd
        .query(conn)
        .map_err(|e| format!("HSET update error: {}", e))?;
    // ALL fields new => the key did not exist. Deliberately NOT `new_fields != 0`:
    // a partial count means the document was there but carried a different field
    // set, which is schema drift between the upload and the update half — a real
    // difference, but not "the write missed the corpus", and turning it into a
    // hard error would abort a run whose updates did land.
    Ok(new_fields == written_fields)
}

/// Format a string metadata field for storage on Valkey.
///
/// - `datetime` fields: ISO-8601 → epoch seconds (already-numeric epochs pass
///   through), for the NUMERIC index.
/// - `text` fields: whitespace-tokenised into a `;`-joined TAG multi-value so a
///   single-term full-text query can match via TAG (Valkey has no TEXT type).
/// - everything else: verbatim.
fn encode_string_field(config: &ValkeyEngineConfig, key: &str, value: &str) -> String {
    if config.datetime_fields.contains(key) {
        if let Some(epoch) = datetime_to_epoch_secs(value) {
            return (epoch as i64).to_string();
        }
        return value.to_string();
    }
    if config.text_tag_fields.contains(key) {
        return tokenize_text_to_tag(value);
    }
    value.to_string()
}

/// Split text on whitespace and re-join with `;` (the TAG separator), escaping
/// any literal `;` inside a token, so it can be indexed as a multi-value TAG.
fn tokenize_text_to_tag(text: &str) -> String {
    text.split_whitespace()
        .map(|w| w.replace(';', "\\;"))
        .collect::<Vec<_>>()
        .join(";")
}

/// Collect schema fields typed `datetime` and `text` (in that order).
fn schema_transform_fields(dataset: &Dataset) -> (HashSet<String>, HashSet<String>) {
    prime_field_types(dataset.config.schema.as_ref())
}

/// Pure schema → (datetime-fields, text-fields) priming. Extracted so the maps
/// can be primed both during `configure()` (upload path) and at the start of
/// `search()` (the `--skip-upload` path, where configure() never runs), and
/// unit-tested without a full `Dataset`.
fn prime_field_types(schema: Option<&serde_json::Value>) -> (HashSet<String>, HashSet<String>) {
    let mut datetime_fields = HashSet::new();
    let mut text_fields = HashSet::new();
    if let Some(schema) = schema.and_then(|s| s.as_object()) {
        for (field, ty) in schema {
            match ty.as_str() {
                Some("datetime") => {
                    datetime_fields.insert(field.clone());
                }
                Some("text") => {
                    text_fields.insert(field.clone());
                }
                _ => {}
            }
        }
    }
    (datetime_fields, text_fields)
}

// ── Engine trait implementation ──────────────────────────────────────────

/// Establish the commandstats baseline if configure() did not (issue #238).
impl ValkeyEngine {
    /// Idempotent and a no-op on the normal configure -> upload -> search path,
    /// so the existing accounting is unchanged; it only rescues the
    /// `--skip-upload` path, where nothing had reset the counters.
    fn prime_commandstats_if_needed(&mut self) -> Result<(), String> {
        if self.commandstats_primed {
            return Ok(());
        }
        let mut conn = self.get_connection()?;
        self.commandstats_baseline = redis_utils::reset_commandstats(&mut conn)?;
        self.commandstats_primed = true;
        Ok(())
    }
}

impl Engine for ValkeyEngine {
    /// Server-side corpus size for this config's index, for the `--skip-upload`
    /// reuse precondition (issue #238). `FT.INFO <index>` → `num_docs`; a missing
    /// index counts as 0 (the corpus to reuse is not there).
    fn corpus_row_count(&mut self) -> Result<Option<CorpusCount>, String> {
        let mut conn = self.get_connection()?;
        Ok(
            redis_utils::ft_index_num_docs(&mut conn, &self.config.index_name)?
                .map(CorpusCount::exact),
        )
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let mut conn = self.get_connection()?;

        // Record datetime fields (ISO → epoch seconds) and text fields (degraded
        // to whitespace-tokenised TAG) so upload can transform their values.
        let (dt_fields, text_fields) = schema_transform_fields(dataset);
        self.config.datetime_fields = Arc::new(dt_fields);
        self.config.text_tag_fields = Arc::new(text_fields);

        if self.config.skip_vector_index {
            println!("Skipping vector index (filter-only mode)");
        } else {
            println!(
                "Using algorithm {} with config {{'M': {}, 'EF_CONSTRUCTION': {}}}",
                self.config.algorithm, self.config.m, self.config.ef_construction
            );
        }

        self.create_index(&mut conn, dataset)?;
        self.commandstats_baseline = redis_utils::reset_commandstats(&mut conn)?;
        self.commandstats_primed = true;
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

        if self.config.parallel <= 1 {
            self.upload_sequential(&ids, &vectors, &metadata)?;
        } else {
            self.upload_parallel(&ids, &vectors, &metadata)?;
        }

        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Include the index-build wait in total_time for cross-engine
        // comparability (mirrors mongodb; matches v0's post_upload() timing).
        let expected = vectors.len();
        let index_start = Instant::now();
        self.wait_for_indexing(expected)?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

        // Verify no HSET failures occurred during upload
        let mut conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut conn,
            &["hset"],
            "upload",
            self.commandstats_baseline.as_ref(),
        )?;

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
        // configure() normally resets the server's command counters and
        // establishes the commandstats baseline; on the `--skip-upload` path it
        // never runs (#238), so check_commandstats below would compare this run's
        // failure count against zero while the counters still hold every failure
        // since the server started — failing a run in which nothing failed.
        // Idempotent no-op once primed; outside every timed window.
        self.prime_commandstats_if_needed()?;

        // Prime the datetime/text field-type maps from the schema so range and
        // text/tag filters are built with the correct field type even on the
        // `--skip-upload` path, where configure() (which normally primes them) is
        // never called. Idempotent: on the upload path this reproduces exactly
        // what configure() already set.
        let (dt_fields, text_fields) = schema_transform_fields(dataset);
        self.config.datetime_fields = Arc::new(dt_fields);
        self.config.text_tag_fields = Arc::new(text_fields);

        if self.config.skip_vector_index {
            return self.search_filter_only(dataset, params, num_queries);
        }

        // Index-existence guard (#151-4): fail loudly on a missing/mismatched index.
        {
            let mut conn = self.get_connection()?;
            redis_utils::ensure_index_exists(&mut conn, &self.config.index_name)?;
        }

        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let parallel = params.parallel.unwrap_or(1) as usize;
        let hybrid_policy = std::env::var("VALKEY_HYBRID_POLICY").unwrap_or_default();
        // Valkey Search caps TIMEOUT at 60000ms
        let query_timeout: i64 = std::env::var("VALKEY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;
        if queries.is_empty() {
            return Err("dataset contains no search queries".to_string());
        }

        let parsed_filters: Vec<QueryFilter<ParsedFilter>> =
            conditions.resolve_all("Valkey", parse_conditions)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute client-side request construction BEFORE the timed region so
        // the per-query window wraps ONLY the RPC round-trip + reply parse
        // (matching pgvector/qdrant). Encoding the FLOAT32 blob and formatting
        // the query string are client work, not server latency. Shared read-only
        // across workers.
        let encoded_queries: Vec<Vec<u8>> =
            queries.iter().map(|q| encode_query_vector(q)).collect();
        let query_strs: Vec<String> = parsed_filters
            .iter()
            .map(|f| build_knn_query_str(f.as_ref()))
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

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        // Resolve the per-config index name once (not per query / per worker).
        let index_name = self.config.index_name.clone();

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "valkey-search", parallel);
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let algorithm = self.config.algorithm.clone();
                let hybrid_policy = hybrid_policy.clone();
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let encoded_queries = &encoded_queries;
                let query_strs = &query_strs;
                let query_idx = Arc::clone(&query_idx);
                let index_name = index_name.as_str();
                let pb = &pb;

                pool.spawn(move |ticket| {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let auth = std::env::var("VALKEY_AUTH").ok();
                    let user = std::env::var("VALKEY_USER").ok();
                    let auth_part = match (&user, &auth) {
                        (Some(u), Some(p)) => format!("{}:{}@", u, p),
                        (None, Some(p)) => format!(":{}@", p),
                        _ => String::new(),
                    };
                    let url = format!(
                        "redis://{}{}:{}/{}",
                        auth_part,
                        host,
                        port,
                        valkey_url_suffix()
                    );
                    let client = match redis::Client::open(url.as_str()) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("valkey-search worker setup failed: {e}"));
                            return (t, p, r, mr, nd);
                        }
                    };
                    let mut conn = match client.get_connection() {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("valkey-search worker setup failed: {e}"));
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this connection with ONE discarded query so the cold
                    // first round-trip is not inside the measured window. Best
                    // effort: errors are ignored and its sample is NOT recorded.
                    {
                        let prime_top = explicit_top.unwrap_or(10);
                        let _ = ft_search_knn(
                            &mut conn,
                            index_name,
                            &encoded_queries[0],
                            &query_strs[0],
                            prime_top,
                            ef,
                            &algorithm,
                            &hybrid_policy,
                            query_timeout,
                            parsed_filters[0].as_ref(),
                        );
                    }

                    // Signal "connected + primed", then block until the coordinator
                    // stamps the shared measurement start and releases everyone.
                    if ticket.arrive_and_wait().is_none() {
                        return (t, p, r, mr, nd);
                    }

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

                        // Timed window: precomputed blob + query string are passed
                        // in, so this wraps only the RPC round-trip and reply parse.
                        let query_start = Instant::now();
                        let results = ft_search_knn(
                            &mut conn,
                            index_name,
                            &encoded_queries[idx],
                            &query_strs[idx],
                            top,
                            ef,
                            &algorithm,
                            &hybrid_policy,
                            query_timeout,
                            parsed_filters[idx].as_ref(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        match &results {
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

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        // Verify no FT.SEARCH failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH"],
            "search",
            self.commandstats_baseline.as_ref(),
        )?;

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
        // configure() normally resets the server's command counters and
        // establishes the commandstats baseline; on the `--skip-upload` path it
        // never runs (#238), so check_commandstats below would compare this run's
        // failure count against zero while the counters still hold every failure
        // since the server started — failing a run in which nothing failed.
        // Idempotent no-op once primed; outside every timed window.
        self.prime_commandstats_if_needed()?;

        // Prime the datetime/text field-type maps from the schema so the update
        // half of the mixed workload encodes datetime payloads as epoch seconds
        // and text payloads as tokenised TAGs even on the `--skip-upload` path
        // (configure() does not run there). Idempotent.
        let (dt_fields, text_fields) = schema_transform_fields(dataset);
        self.config.datetime_fields = Arc::new(dt_fields);
        self.config.text_tag_fields = Arc::new(text_fields);

        // Index-existence guard (#151-4): fail loudly on a missing/mismatched index.
        {
            let mut conn = self.get_connection()?;
            redis_utils::ensure_index_exists(&mut conn, &self.config.index_name)?;
        }

        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let parallel = params.parallel.unwrap_or(1) as usize;
        let hybrid_policy = std::env::var("VALKEY_HYBRID_POLICY").unwrap_or_default();
        let query_timeout: i64 = std::env::var("VALKEY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<QueryFilter<ParsedFilter>> =
            conditions.resolve_all("Valkey", parse_conditions)?;

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

        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));

        let ratio_searches = ratio.searches as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

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

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let config = self.config.clone();
                let algorithm = self.config.algorithm.clone();
                let hybrid_policy = hybrid_policy.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
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
                    let mut ut = crate::engine::UpdateTally::default();
                    let mut pb_pending: u64 = 0;

                    let mut conn = match ValkeyEngine::connect(&host, port) {
                        Ok(c) => c,
                        Err(_) => return (t, p, r, mr, nd, ut),
                    };

                    'outer: loop {
                        // Search phase: do S searches
                        for _ in 0..ratio_searches {
                            let idx = search_idx.fetch_add(1, Ordering::Relaxed);
                            if idx >= num_to_run {
                                break 'outer;
                            }

                            let top = explicit_top.unwrap_or_else(|| {
                                let n = neighbors[idx].len();
                                if n > 0 {
                                    n
                                } else {
                                    10
                                }
                            });

                            // NOTE: the mixed (search+update) path is intentionally
                            // left as-is for a later PR — encode + query-string
                            // build stay inside the timed window here to preserve
                            // its current measurement behavior exactly.
                            let query_start = Instant::now();
                            let vec_bytes = encode_query_vector(&queries[idx]);
                            let query_str = build_knn_query_str(parsed_filters[idx].as_ref());
                            let results = ft_search_knn(
                                &mut conn,
                                &config.index_name,
                                &vec_bytes,
                                &query_str,
                                top,
                                ef,
                                &algorithm,
                                &hybrid_policy,
                                query_timeout,
                                parsed_filters[idx].as_ref(),
                            );
                            let query_time = query_start.elapsed().as_secs_f64();

                            match &results {
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
                            let outcome = hset_single(
                                &mut conn,
                                &config,
                                upd_ids[data_idx],
                                &upd_vectors[data_idx],
                                upd_metadata[data_idx].as_ref(),
                            );
                            let update_time = update_start.elapsed().as_secs_f64();
                            match outcome {
                                // HSET replied 0: every field already existed, so
                                // an indexed corpus document was overwritten.
                                Ok(false) => ut.times.push(update_time),
                                // HSET added new fields: the target hash was not
                                // the fully-populated document the search reads.
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
                }));
            }

            for h in handles {
                let (t, p, r, mr, nd, ut) = h.join().unwrap();
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
                tally.merge(ut);
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        // Verify no failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH", "hset"],
            "mixed",
            self.commandstats_baseline.as_ref(),
        )?;

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
            crate::engine::UpdateAttribution::CorpusRow,
            ratio,
            "HSET reported newly-added fields, where overwriting an already-populated \
             corpus document reports 0",
        );
        Ok(results)
    }

    fn delete(&mut self) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        // Valkey Search has no DD flag: drop this config's index + its keys via a
        // prefix-scoped SCAN+UNLINK (not a keyspace-wide FLUSHALL, which under
        // #151-4 coexistence would wipe sibling configs' data).
        redis_utils::drop_index_and_keys(
            &mut conn,
            &self.config.index_name,
            &self.config.key_prefix,
        );
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

        // used_memory is server-wide (SUM of all resident configs under #151-4);
        // kept only as a secondary/global figure. The per-config figure is the
        // FT.INFO index size below.
        let info_str: String = redis::cmd("INFO").arg("memory").query(&mut conn).ok()?;
        let used_memory: i64 = parse_used_memory(&info_str);

        let ft_info_raw: Option<redis::Value> = redis::cmd("FT.INFO")
            .arg(&self.config.index_name)
            .query::<redis::Value>(&mut conn)
            .ok();
        let index_memory_bytes = ft_info_raw
            .as_ref()
            .and_then(redis_utils::ft_info_index_memory_bytes);
        let ft_info = ft_info_raw.as_ref().map(redis_value_to_json);

        Some(serde_json::json!({
            "used_memory": [used_memory],
            "index_memory_bytes": index_memory_bytes,
            "index_info": ft_info,
        }))
    }

    fn server_metadata(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;
        let mut meta = redis_utils::collect_server_metadata(&mut conn);
        // Vector index stats. Errors on the BEFORE snapshot (index not yet
        // created) → index_info: null.
        let ft_info = redis::cmd("FT.INFO")
            .arg(&self.config.index_name)
            .query::<redis::Value>(&mut conn)
            .ok()
            .map(|v| redis_value_to_json(&v));
        if let Some(obj) = meta.as_object_mut() {
            obj.insert(
                "index_info".to_string(),
                ft_info.unwrap_or(serde_json::Value::Null),
            );
        }
        Some(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_knn_query_str, encode_query_vector, parse_conditions, FilterParamValue, ParsedFilter,
    };
    use std::collections::HashMap;

    // ── Timed-window hoisting fidelity ─────────────────────────────────────
    // The perf change moves the FLOAT32 encode + query-string build OUT of the
    // per-query timed window (precomputed once before the parallel region).
    // Prove the precomputed values are byte-identical to the in-window originals.

    #[test]
    fn encode_query_vector_matches_legacy_fp32_le_bytes() {
        let v = vec![1.0f32, -2.5, 3.25];
        // Legacy in-window encode was `iter().flat_map(f.to_le_bytes()).collect()`.
        let legacy: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(encode_query_vector(&v), legacy);
    }

    #[test]
    fn precomputed_blobs_match_per_query_encode() {
        let queries = [vec![1.0f32, -2.5, 3.5], vec![0.0f32, 42.0, -7.25]];
        let precomputed: Vec<Vec<u8>> = queries.iter().map(|q| encode_query_vector(q)).collect();
        for (i, q) in queries.iter().enumerate() {
            assert_eq!(precomputed[i], encode_query_vector(q), "q{i}");
        }
    }

    #[test]
    fn build_knn_query_str_matches_legacy_format() {
        assert_eq!(
            build_knn_query_str(None),
            "*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );
    }

    #[test]
    fn build_knn_query_str_filtered_matches_legacy_format() {
        // query_str varies per query ONLY through the filter prefilter — pin the
        // FILTERED path against the legacy inline `format!` (master):
        //   "{prefilter}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        let params: HashMap<String, FilterParamValue> =
            [("brand_0".to_string(), FilterParamValue::Int(7))]
                .into_iter()
                .collect();
        let filter: ParsedFilter = ("@brand:{apple}".to_string(), params);
        assert_eq!(
            build_knn_query_str(Some(&filter)),
            "@brand:{apple}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );
    }

    #[test]
    fn match_any_string_list_emits_inlined_tag_or() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":["red","blue"]}}}]});
        let (q, _params) = parse_conditions(&cond).unwrap();
        // Values inlined (no $param inside TAG braces on Valkey Search).
        assert!(q.contains("@color:{red | blue}"), "q={}", q);
    }

    #[test]
    fn match_any_int_list_emits_numeric_or() {
        let cond = serde_json::json!({"and":[{"size":{"match":{"any":[1,2]}}}]});
        // Valkey inlines NUMERIC literals (no $param inside […] brackets).
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@size:[1 1]"), "q={}", q);
        assert!(q.contains("@size:[2 2]"), "q={}", q);
        assert!(!params.keys().any(|k| k.starts_with("size_")), "no params");
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":[]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@color:{"), "expected never-match, q={}", q);
    }

    #[test]
    fn match_any_escapes_or_delimiter() {
        // A value containing '|' must not break the OR structure.
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":["a|b"]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("a\\|b"), "q={}", q);
    }

    #[test]
    fn match_exact_value_still_inlined_tag() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"value":"red"}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@color:{red}"), "q={}", q);
    }

    // ── New filter datatypes: bool / uuid / full-text / datetime ───────────
    use super::{
        datetime_to_epoch_secs, encode_string_field, tokenize_text_to_tag, ValkeyEngineConfig,
    };
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn match_bool_emits_inlined_tag_token() {
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": true}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@flag:{true}"), "q={}", q);
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": false}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@flag:{false}"), "q={}", q);
    }

    #[test]
    fn match_uuid_value_inlined_tag() {
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let cond = serde_json::json!({"and":[{"uid":{"match":{"value": uuid}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains(&format!("@uid:{{{}}}", uuid)), "q={}", q);
    }

    #[test]
    fn match_text_degrades_to_single_token_tag() {
        // Full-text on Valkey → single-term TAG match (degraded).
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "quick"}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@body:{quick}"), "q={}", q);
    }

    #[test]
    fn match_text_empty_is_never_match() {
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "  "}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@body:{"), "expected never-match, q={}", q);
    }

    #[test]
    fn range_datetime_iso_bounds_inline_epoch() {
        let cond = serde_json::json!({"and":[{"ts":{"range":{
            "gte": "2021-01-01T00:00:00Z",
            "lt":  "2022-01-01T00:00:00Z"
        }}}]});
        // 2021-01-01T00:00:00Z == 1609459200, 2022-01-01T00:00:00Z == 1640995200.
        // Valkey doesn't substitute $param in NUMERIC brackets → bounds are
        // inlined as integer epochs; no params emitted for the range.
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@ts:[1609459200 +inf]"), "q={}", q);
        assert!(q.contains("@ts:[-inf (1640995200]"), "q={}", q);
        assert!(
            !params.keys().any(|k| k.starts_with("ts_")),
            "no range params"
        );
    }

    #[test]
    fn range_datetime_numeric_epoch_bounds_inline() {
        let cond = serde_json::json!({"and":[{"ts":{"range":{"gte": 1609459200}}}]});
        let (q, _params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@ts:[1609459200 +inf]"), "q={}", q);
    }

    #[test]
    fn range_gt_is_exclusive_gte_inclusive() {
        // gt must be exclusive `[(v +inf]`; gte inclusive `[v +inf]`.
        let gt = parse_conditions(&serde_json::json!({"and":[{"n":{"range":{"gt": 5}}}]}))
            .unwrap()
            .0;
        assert!(gt.contains("@n:[(5 +inf]"), "gt not exclusive: {}", gt);
        let gte = parse_conditions(&serde_json::json!({"and":[{"n":{"range":{"gte": 5}}}]}))
            .unwrap()
            .0;
        assert!(
            gte.contains("@n:[5 +inf]") && !gte.contains("(5"),
            "gte: {}",
            gte
        );
    }

    #[test]
    fn datetime_tolerates_naive_and_date_only() {
        assert_eq!(
            datetime_to_epoch_secs("2021-01-01").map(|f| f as i64),
            Some(1609459200)
        );
        assert_eq!(
            datetime_to_epoch_secs("2021-01-01T00:00:00").map(|f| f as i64),
            Some(1609459200)
        );
        assert_eq!(
            datetime_to_epoch_secs("2021-01-01 00:00:00").map(|f| f as i64),
            Some(1609459200)
        );
    }

    #[test]
    fn two_field_and_combines_both_clauses() {
        let cond = serde_json::json!({"and":[
            {"color":{"match":{"value":"red"}}},
            {"size":{"range":{"gte": 10}}}
        ]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@color:{red}"), "q={}", q);
        assert!(q.contains("@size:[10 +inf]"), "q={}", q);
    }

    #[test]
    fn datetime_to_epoch_secs_parses_rfc3339_and_rejects_plain() {
        assert_eq!(
            datetime_to_epoch_secs("2021-01-01T00:00:00Z").map(|f| f as i64),
            Some(1609459200)
        );
        assert!(datetime_to_epoch_secs("not-a-date").is_none());
        assert!(datetime_to_epoch_secs("1609459200").is_none());
    }

    #[test]
    fn tokenize_text_to_tag_splits_on_whitespace() {
        assert_eq!(tokenize_text_to_tag("the sky is blue"), "the;sky;is;blue");
        assert_eq!(tokenize_text_to_tag("solo"), "solo");
    }

    #[test]
    fn encode_string_field_transforms_datetime_and_text() {
        let mut dt = HashSet::new();
        dt.insert("ts".to_string());
        let mut txt = HashSet::new();
        txt.insert("body".to_string());
        let cfg = ValkeyEngineConfig {
            m: 16,
            ef_construction: 128,
            data_type: "FLOAT32".to_string(),
            algorithm: "hnsw".to_string(),
            batch_size: 1,
            parallel: 1,
            skip_vector_index: false,
            datetime_fields: Arc::new(dt),
            text_tag_fields: Arc::new(txt),
            index_name: "idx:test".to_string(),
            key_prefix: "test:".to_string(),
        };
        assert_eq!(
            encode_string_field(&cfg, "ts", "2021-01-01T00:00:00Z"),
            "1609459200"
        );
        assert_eq!(encode_string_field(&cfg, "ts", "1609459200"), "1609459200");
        assert_eq!(
            encode_string_field(&cfg, "body", "the sky is blue"),
            "the;sky;is;blue"
        );
        assert_eq!(encode_string_field(&cfg, "color", "red"), "red");
    }

    // ── redis::Value response parsing ──────────────────────────────────────
    // These guard the manual RESP-array parsing (the surface most exposed to a
    // redis-crate upgrade / RESP2-vs-RESP3 change).
    use super::{extract_vector_score, parse_ft_search_response, redis_value_to_json};
    use redis::Value;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    #[test]
    fn parse_ft_search_response_resp2_reads_id_score_pairs() {
        // RESP2 FT.SEARCH shape: [count, key1, fields1, key2, fields2, ...]. Doc
        // ids arrive as the string KEY; a per-config-prefixed key ("cfg:42",
        // #151-4) resolves to its trailing numeric id.
        let resp = Value::Array(vec![
            Value::Int(2),
            bulk("7"),
            Value::Array(vec![bulk("vector_score"), bulk("0.25")]),
            bulk("cfg:42"),
            Value::Array(vec![bulk("vector_score"), bulk("1.5")]),
        ]);
        let hits = parse_ft_search_response(&resp).unwrap();
        assert_eq!(hits, vec![(7, 0.25), (42, 1.5)]);
    }

    #[test]
    fn parse_ft_search_response_resp3_map_reads_results() {
        // RESP3 FT.SEARCH shape: map with a `results` array of per-doc maps.
        let doc = |id: &str, score: &str| {
            Value::Map(vec![
                (bulk("id"), bulk(id)),
                (
                    bulk("extra_attributes"),
                    Value::Map(vec![(bulk("vector_score"), bulk(score))]),
                ),
            ])
        };
        let resp = Value::Map(vec![
            (
                bulk("results"),
                Value::Array(vec![doc("7", "0.25"), doc("42", "1.5")]),
            ),
            (bulk("total_results"), Value::Int(2)),
        ]);
        let hits = parse_ft_search_response(&resp).unwrap();
        assert_eq!(hits, vec![(7, 0.25), (42, 1.5)]);
    }

    #[test]
    fn parse_ft_search_response_empty_and_unknown_variants() {
        assert_eq!(
            parse_ft_search_response(&Value::Array(vec![])).unwrap(),
            vec![]
        );
        assert_eq!(parse_ft_search_response(&Value::Nil).unwrap(), vec![]);
        // A non-string/int id falls back to 0; a non-array field block → score 0.
        let resp = Value::Array(vec![Value::Int(1), Value::Nil, Value::Nil]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(0, 0.0)]);
    }

    #[test]
    fn extract_vector_score_finds_field_or_defaults_zero() {
        let fields = vec![
            bulk("__key"),
            bulk("doc:1"),
            bulk("vector_score"),
            bulk("0.75"),
        ];
        assert!((extract_vector_score(&fields) - 0.75).abs() < 1e-9);
        // Missing field → 0.0
        assert_eq!(extract_vector_score(&[bulk("other"), bulk("x")]), 0.0);
    }

    #[test]
    fn redis_value_to_json_covers_scalars_arrays_maps_and_fallthrough() {
        assert_eq!(redis_value_to_json(&Value::Nil), serde_json::Value::Null);
        assert_eq!(redis_value_to_json(&Value::Int(5)), serde_json::json!(5));
        assert_eq!(
            redis_value_to_json(&Value::Boolean(true)),
            serde_json::json!(true)
        );
        assert_eq!(redis_value_to_json(&bulk("hi")), serde_json::json!("hi"));
        assert_eq!(
            redis_value_to_json(&Value::Array(vec![Value::Int(1), bulk("a")])),
            serde_json::json!([1, "a"])
        );
        assert_eq!(
            redis_value_to_json(&Value::Map(vec![(bulk("k"), Value::Int(9))])),
            serde_json::json!({"k": 9})
        );
        // Non-exhaustive/other variants (e.g. Okay) must not panic and must not be
        // silently dropped — they debug-format to a non-empty string rather than
        // vanish. (The exact string is a redis-crate impl detail, so don't pin it.)
        let okay = redis_value_to_json(&Value::Okay);
        assert!(
            okay.as_str().map(|s| !s.is_empty()).unwrap_or(false),
            "Okay should map to a non-empty JSON string, got {:?}",
            okay
        );
    }

    #[test]
    fn redis_value_to_json_covers_double_simplestring_and_bad_utf8() {
        // Arms not exercised by the test above.
        assert_eq!(
            redis_value_to_json(&Value::Double(1.5)),
            serde_json::json!(1.5)
        );
        assert_eq!(
            redis_value_to_json(&Value::SimpleString("PONG".into())),
            serde_json::json!("PONG")
        );
        // Invalid UTF-8 BulkString → "<N bytes>" placeholder (never a panic/drop).
        assert_eq!(
            redis_value_to_json(&Value::BulkString(vec![0xff, 0xfe])),
            serde_json::json!("<2 bytes>")
        );
    }

    #[test]
    fn value_as_i64_reads_simplestring_id() {
        use super::value_as_i64;
        // Redis-8 RESP2 can return the doc id as a SimpleString (e.g. "7").
        assert_eq!(value_as_i64(&Value::SimpleString("7".into())), 7);
        assert_eq!(value_as_i64(&bulk("42")), 42);
        assert_eq!(value_as_i64(&Value::Int(5)), 5);
        assert_eq!(value_as_i64(&Value::SimpleString("x".into())), 0);
        assert_eq!(value_as_i64(&Value::Nil), 0);
    }

    #[test]
    fn parse_ft_search_resp2_drops_trailing_id_without_fields() {
        // Odd trailing element: [count, id] with no field block → dangling id
        // dropped (no doc emitted), NOT surfaced with a zero score.
        let resp = Value::Array(vec![Value::Int(1), bulk("5")]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![]);
    }

    #[test]
    fn parse_ft_search_resp3_skips_non_map_doc_entries() {
        // A non-map entry in `results` is skipped; sibling map docs still parse.
        let doc = Value::Map(vec![
            (bulk("id"), bulk("7")),
            (
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.25"))]),
            ),
        ]);
        let resp = Value::Map(vec![(bulk("results"), Value::Array(vec![Value::Nil, doc]))]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(7, 0.25)]);
    }

    #[test]
    fn parse_ft_search_resp3_id_parse_failure_skips_doc() {
        // A map doc whose `id` is unparseable is SKIPPED (mirrors the RESP2
        // trailing-id drop), not emitted as a phantom id=0 hit. A sibling doc
        // with a valid id still parses and is emitted.
        let bad = Value::Map(vec![
            (bulk("id"), bulk("not-a-number")),
            (
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.5"))]),
            ),
        ]);
        let good = Value::Map(vec![
            (bulk("id"), bulk("42")),
            (
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.9"))]),
            ),
        ]);
        let resp = Value::Map(vec![(bulk("results"), Value::Array(vec![bad, good]))]);
        // Only the valid-id doc survives; the unparseable one is dropped.
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(42, 0.9)]);
    }

    #[test]
    fn parse_ft_search_resp3_missing_id_skips_doc() {
        // A doc with no `id` field at all is also skipped, not emitted as id=0.
        let resp = Value::Map(vec![(
            bulk("results"),
            Value::Array(vec![Value::Map(vec![(
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.5"))]),
            )])]),
        )]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![]);
    }

    // ── field-type map priming (#125) ─────────────────────────────────────
    // The datetime/text maps must be identical whether upload runs (configure())
    // or not (--skip-upload, primed at the start of search()). Cover the pure
    // priming fn directly so both call sites stay correct.
    #[test]
    fn prime_field_types_splits_datetime_and_text_schema_fields() {
        use super::prime_field_types;
        let schema = serde_json::json!({
            "created_at": "datetime",
            "body": "text",
            "title": "text",
            "price": "int",
            "brand": "keyword",
        });
        let (dt, txt) = prime_field_types(Some(&schema));
        let mut expected_dt = std::collections::HashSet::new();
        expected_dt.insert("created_at".to_string());
        let mut expected_txt = std::collections::HashSet::new();
        expected_txt.insert("body".to_string());
        expected_txt.insert("title".to_string());
        assert_eq!(dt, expected_dt);
        assert_eq!(txt, expected_txt);
    }

    #[test]
    fn prime_field_types_empty_for_none_or_no_matching_fields() {
        use super::prime_field_types;
        let (dt, txt) = prime_field_types(None);
        assert!(dt.is_empty() && txt.is_empty());
        let schema = serde_json::json!({ "price": "int", "brand": "keyword" });
        let (dt, txt) = prime_field_types(Some(&schema));
        assert!(dt.is_empty() && txt.is_empty());
    }

    #[test]
    fn parse_ft_search_resp3_missing_results_key_is_empty() {
        // RESP3 map with no `results` key → no hits.
        let resp = Value::Map(vec![(bulk("total_results"), Value::Int(0))]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![]);
    }

    #[test]
    fn parse_used_memory_parses_real_info_block() {
        use super::parse_used_memory;
        let info = "# Memory\r\nused_memory:1048576\r\nused_memory_rss:2097152\r\nused_memory_peak:3000000\r\n";
        // Exact-prefix match: picks used_memory, never used_memory_rss/_peak.
        assert_eq!(parse_used_memory(info), 1_048_576);
        // Missing line → 0.
        assert_eq!(parse_used_memory("# Memory\r\nmaxmemory:0\r\n"), 0);
        // Malformed value → 0.
        assert_eq!(parse_used_memory("used_memory:not_a_number\r\n"), 0);
        // A block with ONLY used_memory_rss must not match the used_memory prefix.
        assert_eq!(parse_used_memory("used_memory_rss:999\r\n"), 0);
    }

    // ── OR-branch of the condition parser ──────────────────────────────────
    use super::{
        build_exact_match_filter, build_geo_filter, build_range_filter, map_distance_metric,
    };

    fn q_of(cond: serde_json::Value) -> Option<String> {
        parse_conditions(&cond).map(|(q, _)| q)
    }

    #[test]
    fn or_only_emits_pipe_joined_group() {
        let cond = serde_json::json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        // Values inlined (no $param inside TAG braces on Valkey Search).
        assert_eq!(q_of(cond).unwrap(), "(@a:{x} | @b:{y})");
    }

    #[test]
    fn and_plus_or_keeps_both_groups() {
        let cond = serde_json::json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}}],
        });
        assert_eq!(q_of(cond).unwrap(), "(@a:{x}) (@b:{y})");
    }

    // ── Range operators (inlined literals on Valkey Search) ────────────────
    // Test the range arm directly (parse_conditions additionally wraps the whole
    // AND group in `(...)`).

    fn range_q(criteria: serde_json::Value) -> Option<String> {
        let mut counter = 0;
        build_range_filter("n", &criteria, &mut counter).map(|(q, _)| q)
    }

    #[test]
    fn range_lt_is_exclusive() {
        assert_eq!(
            range_q(serde_json::json!({"lt":5})).unwrap(),
            "@n:[-inf (5]"
        );
    }

    #[test]
    fn range_lte_is_inclusive() {
        assert_eq!(
            range_q(serde_json::json!({"lte":5})).unwrap(),
            "@n:[-inf 5]"
        );
    }

    #[test]
    fn range_gt_is_exclusive() {
        assert_eq!(
            range_q(serde_json::json!({"gt":5})).unwrap(),
            "@n:[(5 +inf]"
        );
    }

    #[test]
    fn range_gte_is_inclusive() {
        assert_eq!(
            range_q(serde_json::json!({"gte":5})).unwrap(),
            "@n:[5 +inf]"
        );
    }

    #[test]
    fn range_two_sided_gte_lt() {
        // Fixed order lt, gt, lte, gte (space-joined).
        assert_eq!(
            range_q(serde_json::json!({"gte":10,"lt":20})).unwrap(),
            "@n:[-inf (20] @n:[10 +inf]"
        );
    }

    #[test]
    fn range_unknown_op_is_skipped() {
        assert!(range_q(serde_json::json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_is_skipped() {
        assert!(range_q(serde_json::json!({"gte":serde_json::Value::Null})).is_none());
    }

    // ── Geo filter (param-based, like redis) ───────────────────────────────

    fn geo_q(criteria: serde_json::Value) -> Option<(String, HashMap<String, FilterParamValue>)> {
        let mut counter = 0;
        build_geo_filter("loc", &criteria, &mut counter)
    }

    #[test]
    fn geo_with_radius_emits_lon_lat_radius() {
        let (q, params) = geo_q(serde_json::json!({"lon":10.0,"lat":20.0,"radius":500})).unwrap();
        assert_eq!(q, "@loc:[$loc_0_lon $loc_0_lat $loc_0_radius m]", "q={}", q);
        assert!(matches!(
            params.get("loc_0_radius"),
            Some(FilterParamValue::Int(500))
        ));
    }

    #[test]
    fn geo_missing_radius_is_none() {
        // Valkey Search geo has NO default radius; a missing radius drops the clause.
        assert!(geo_q(serde_json::json!({"lon":10.0,"lat":20.0})).is_none());
    }

    #[test]
    fn geo_missing_lat_or_lon_is_none() {
        assert!(geo_q(serde_json::json!({"lon":10.0,"radius":500})).is_none());
        assert!(geo_q(serde_json::json!({"lat":20.0,"radius":500})).is_none());
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn distance_metric_maps_all_arms() {
        assert_eq!(map_distance_metric("cosine"), "COSINE");
        assert_eq!(map_distance_metric("angular"), "COSINE");
        assert_eq!(map_distance_metric("l2"), "L2");
        assert_eq!(map_distance_metric("euclidean"), "L2");
        assert_eq!(map_distance_metric("dot"), "IP");
        assert_eq!(map_distance_metric("ip"), "IP");
        assert_eq!(map_distance_metric("nope"), "COSINE");
    }

    // ── Exact-match numeric / non-scalar arms ──────────────────────────────

    fn exact_q(criteria: serde_json::Value) -> Option<String> {
        let mut counter = 0;
        build_exact_match_filter("n", &criteria, &mut counter).map(|(q, _)| q)
    }

    #[test]
    fn exact_match_int_emits_inlined_numeric_point() {
        assert_eq!(exact_q(serde_json::json!({"value":5})).unwrap(), "@n:[5 5]");
    }

    #[test]
    fn exact_match_float_emits_inlined_numeric_point() {
        assert_eq!(
            exact_q(serde_json::json!({"value":1.5})).unwrap(),
            "@n:[1.5 1.5]"
        );
    }

    #[test]
    fn exact_match_array_value_is_none() {
        assert!(exact_q(serde_json::json!({"value":[1,2]})).is_none());
    }
}
