//! Redis-rs engine implementation.
//!
//! Implements the Engine trait for RediSearch vector similarity.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::seq::SliceRandom;
use rand::SeedableRng;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use super::redis_utils;
use redis::Connection;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::index_naming::{derive_index_name, derive_key_prefix};
use crate::engine::{
    attach_open_loop_metrics, closed_loop_duration, zero_search_results, Engine, OpenLoopPlan,
    SearchResults, UpdateSearchRatio, UploadStats,
};
use crate::metrics::compute_metrics;
use vector_db_benchmark::parsers::{datetime_to_epoch_secs, parse_ft_search_response};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};
use vector_db_benchmark::start_gate::WorkerPool;

/// Redis engine configuration
#[derive(Clone)]
pub struct RedisEngineConfig {
    pub m: i64,
    pub ef_construction: i64,
    pub data_type: String,
    pub algorithm: String,
    pub batch_size: usize,
    pub parallel: usize,
    pub skip_vector_index: bool,
    /// SVS-VAMANA build params, resolved once in `new()` with precedence
    /// `collection_params.svs-vamana_config` > env (`REDIS_SVS_*`) > derived.
    /// `None` graph/window → derive from `m`*2 / `ef_construction` at FT.CREATE.
    /// Ignored for non-SVS algorithms.
    pub svs_graph_max_degree: Option<i64>,
    pub svs_construction_window_size: Option<i64>,
    pub svs_compression: Option<String>,
    pub svs_reduce: Option<i64>,
    /// Per-config RediSearch index name (`"<base>:<config>"`, issue #151-4) so a
    /// sweep's configs address disjoint indexes on one server. Resolved once in
    /// `new()`; never re-read from the environment inside a timed window.
    pub index_name: String,
    /// Per-config key prefix (`"<config>:"`, issue #151-4). Every doc key is
    /// `"<key_prefix><id>"`, and the index is `PREFIX 1 "<key_prefix>"`, so each
    /// config owns a disjoint keyspace and its `FT.DROPINDEX ... DD` teardown
    /// touches only its own docs.
    pub key_prefix: String,
    /// Shared-corpus mode (#188): true when `REDIS_KEY_PREFIX` is set, making ALL
    /// sweep configs share ONE key prefix (one corpus) instead of per-config
    /// disjoint keyspaces. Lets a sweep over one dataset upload the (identical)
    /// corpus ONCE and build many per-config indexes over it (N× ingest -> 1×).
    /// In this mode the index is dropped WITHOUT `DD` (the shared docs survive
    /// across configs) and a re-upload is skipped when the corpus is already
    /// populated. Off by default -> per-config isolation is unchanged.
    pub shared_corpus: bool,
    /// Schema field names declared as `datetime`. Values for these fields are
    /// ISO-8601 strings in the payload; they are converted to epoch seconds at
    /// HSET time so the NUMERIC index/range filters match. Populated in
    /// `configure()` from the dataset schema. Wrapped in `Arc` so the per-thread
    /// config clones share one set.
    pub datetime_fields: Arc<HashSet<String>>,
}

/// The shared corpus key prefix for #188 (upload-once / build-many), read from
/// `REDIS_KEY_PREFIX`. When set, every sweep config shares this one prefix (a
/// single uploaded corpus); when unset, each config keeps its own `<config>:`
/// prefix and disjoint keyspace. A trailing `:` is appended if missing so the
/// value is a clean key namespace. Resolved in `new()`, never in a timed window.
fn shared_key_prefix() -> Option<String> {
    let raw = crate::effective_config::env_var("REDIS_KEY_PREFIX").ok();
    let resolved = raw.as_deref().map(str::trim).and_then(|s| {
        if s.is_empty() {
            None
        } else if s.ends_with(':') {
            Some(s.to_string())
        } else {
            Some(format!("{s}:"))
        }
    });
    // The raw text is not the value used: it is trimmed and gains a `:`, and a
    // whitespace-only setting turns shared-corpus mode OFF while still showing
    // up as set. Record what the run actually keyed on (#212).
    crate::effective_config::record_effective(
        "shared_corpus_key_prefix",
        resolved
            .clone()
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    resolved
}

/// Count keys matching `<prefix>*` via cursor-based SCAN (non-blocking; the
/// prefix is sanitized so `*` is the only glob metachar). Used by shared-corpus
/// mode (#188) to detect whether the corpus is already fully uploaded so a
/// sweep's later configs can skip the re-upload.
fn count_prefix_keys(conn: &mut Connection, prefix: &str) -> usize {
    let pattern = format!("{prefix}*");
    let mut cursor: u64 = 0;
    let mut count = 0usize;
    loop {
        let res: Result<(u64, Vec<String>), redis::RedisError> = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(&pattern)
            .arg("COUNT")
            .arg(1000)
            .query(conn);
        match res {
            Ok((next, keys)) => {
                count += keys.len();
                if next == 0 {
                    break;
                }
                cursor = next;
            }
            Err(_) => break,
        }
    }
    count
}

/// Shared-corpus completeness test (#188): may this config skip its re-upload?
///
/// `existing` is the live key count under the shared prefix, `expected` the
/// corpus size measured from the dataset on disk (see
/// `Dataset::corpus_completeness_target`, #224 — never the raw declared
/// `vector_count`, which can be wrong). Only a corpus that is fully present may
/// be reused; anything short of it must upload.
fn shared_corpus_is_complete(existing: usize, expected: u64) -> bool {
    expected > 0 && existing as u64 >= expected
}

pub struct RedisEngine {
    name: String,
    redis_url: String,
    config: RedisEngineConfig,
    search_params: Vec<SearchParams>,
    commandstats_baseline: Option<redis_utils::CommandStatsBaseline>,
}

impl RedisEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let redis_url = crate::engine::build_redis_url(host);

        // Extract HNSW config
        let (m, ef_construction) = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref())
            .map(|h| (h.m.unwrap_or(16), h.ef_construction.unwrap_or(128)))
            .unwrap_or((16, 128));

        // Untyped catch-all under `collection_params` (serde-flattened). Legacy
        // SVS configs (e.g. dbpedia-calibration.json) nest `algorithm`,
        // `data_type`, and a `svs-vamana_config` block here rather than at the
        // top level, so read them as fallbacks.
        let cp_extra = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.extra.as_ref());
        let cp_str = |key: &str| {
            cp_extra
                .and_then(|e| e.get(key))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        let up_str = |key: &str| {
            engine_config
                .upload_params
                .as_ref()
                .and_then(|p| p.get(key))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };

        // algorithm: top-level > collection_params.algorithm > upload_params.algorithm.
        let algorithm = engine_config
            .algorithm
            .clone()
            .or_else(|| cp_str("algorithm"))
            .or_else(|| up_str("algorithm"))
            .unwrap_or_else(|| "hnsw".to_string());

        // data_type: upload_params.data_type > collection_params.data_type.
        let data_type = up_str("data_type")
            .or_else(|| cp_str("data_type"))
            .unwrap_or_else(|| "FLOAT32".to_string());

        // SVS-VAMANA build params: `collection_params.svs-vamana_config` block
        // wins, else the REDIS_SVS_* env vars, else derived at FT.CREATE. Resolved
        // only for SVS algorithms so a non-SVS config can never pick up a stray
        // REDIS_SVS_* env var (the fields stay None and unused).
        let (svs_graph_max_degree, svs_construction_window_size, svs_compression, svs_reduce) =
            if is_svs(&algorithm) {
                let svs_block = cp_extra
                    .and_then(|e| e.get("svs-vamana_config"))
                    .and_then(|v| v.as_object());
                let svs_i64 =
                    |key: &str| svs_block.and_then(|b| b.get(key)).and_then(|v| v.as_i64());
                let compression = svs_block
                    .and_then(|b| b.get("compression"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .or_else(|| {
                        crate::effective_config::env_var("REDIS_SVS_COMPRESSION")
                            .ok()
                            .map(|s| s.trim().to_string())
                    })
                    .filter(|s| !s.is_empty());
                let reduce = svs_i64("REDUCE")
                    .or_else(|| {
                        crate::effective_config::env_var("REDIS_SVS_REDUCE")
                            .ok()
                            .and_then(|s| s.trim().parse::<i64>().ok())
                    })
                    .filter(|&n| n > 0);
                // `REDIS_SVS_*` reach these only after config precedence, a
                // trim and an emptiness/positivity filter, so the raw text in
                // `env` is not what the index was built with.
                crate::effective_config::record_effective(
                    "svs_compression",
                    compression
                        .clone()
                        .map(serde_json::Value::from)
                        .unwrap_or(serde_json::Value::Null),
                );
                crate::effective_config::record_effective(
                    "svs_reduce",
                    reduce
                        .map(serde_json::Value::from)
                        .unwrap_or(serde_json::Value::Null),
                );
                (
                    svs_i64("GRAPH_MAX_DEGREE"),
                    svs_i64("CONSTRUCTION_WINDOW_SIZE"),
                    compression,
                    reduce,
                )
            } else {
                (None, None, None, None)
            };

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
            redis_url,
            config: RedisEngineConfig {
                m,
                ef_construction,
                data_type,
                algorithm,
                batch_size,
                parallel,
                skip_vector_index: engine_config.skip_vector_index,
                svs_graph_max_degree,
                svs_construction_window_size,
                svs_compression,
                svs_reduce,
                index_name: derive_index_name("REDIS_INDEX_NAME", "idx", &engine_config.name),
                // Shared-corpus mode (#188): REDIS_KEY_PREFIX overrides the
                // per-config prefix with ONE shared prefix across all sweep
                // configs. Default (unset) keeps the per-config `<config>:` prefix
                // and full isolation.
                key_prefix: shared_key_prefix()
                    .unwrap_or_else(|| derive_key_prefix(&engine_config.name)),
                shared_corpus: shared_key_prefix().is_some(),
                datetime_fields: Arc::new(HashSet::new()),
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            commandstats_baseline: None,
        })
    }

    fn get_connection(&self) -> Result<Connection, String> {
        let client = redis::Client::open(self.redis_url.as_str()).map_err(|e| e.to_string())?;
        client.get_connection().map_err(|e| e.to_string())
    }

    fn create_index(&self, conn: &mut Connection, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        // Drop existing index if any. Normally with DD, which deletes the indexed
        // docs too; because the index PREFIX is this config's `<config>:`, DD
        // removes only this config's keys — siblings untouched (#151-4). In
        // shared-corpus mode (#188) the docs are SHARED across configs and
        // uploaded once, so drop the index WITHOUT DD to preserve them for the
        // other configs' indexes (build-many over one corpus).
        {
            let mut cmd = redis::cmd("FT.DROPINDEX");
            cmd.arg(&self.config.index_name);
            if !self.config.shared_corpus {
                cmd.arg("DD");
            }
            let _ = cmd.query::<()>(conn);
        }

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

        // Vector field with HNSW params (matches Python v0 configure.py)
        // Skipped when skip_vector_index is set (filter-only benchmark)
        if !self.config.skip_vector_index {
            // Emit the canonical algorithm token. SVS is matched by substring
            // (is_svs), but RediSearch requires the exact `SVS-VAMANA` keyword —
            // so normalize any SVS spelling to it rather than passing a token
            // like `SVS` that FT.CREATE would reject.
            let algo_token = if is_svs(&self.config.algorithm) {
                "SVS-VAMANA".to_string()
            } else {
                self.config.algorithm.to_uppercase()
            };
            cmd.arg("vector").arg("VECTOR").arg(algo_token);
            if is_svs(&self.config.algorithm) {
                // SVS-VAMANA (Redis 8.2+) uses GRAPH_MAX_DEGREE / CONSTRUCTION_WINDOW_SIZE,
                // NOT HNSW's M / EF_CONSTRUCTION (those are rejected at FT.CREATE:
                // "Bad arguments for algorithm SVS-VAMANA: M"). Values come from the
                // `svs-vamana_config` block if present, else are derived from the
                // swept M / EF_CONSTRUCTION so one grid covers both algorithms.
                // GRAPH_MAX_DEGREE derives as M*2 (Redis docs: it is HNSW's M*2
                // equivalent), so the same M yields a comparable graph degree.
                //
                // Compression/REDUCE come from the config block or REDIS_SVS_*
                // (resolved in new()). REDUCE (LeanVec dim-reduction) is only valid
                // with a LeanVec COMPRESSION. On OSS / non-Intel builds COMPRESSION
                // silently falls back to basic 8-bit scalar quantization.
                let graph_max_degree = self
                    .config
                    .svs_graph_max_degree
                    .unwrap_or(self.config.m * 2);
                let construction_window_size = self
                    .config
                    .svs_construction_window_size
                    .unwrap_or(self.config.ef_construction);
                let compression = self.config.svs_compression.clone();
                let mut reduce = self.config.svs_reduce;
                // REDUCE is only valid with a LeanVec COMPRESSION; emitting it
                // otherwise builds an index RediSearch rejects. Drop it (with a
                // warning) rather than send a doomed FT.CREATE.
                let is_leanvec = compression
                    .as_deref()
                    .map(|c| c.to_uppercase().contains("LEANVEC"))
                    .unwrap_or(false);
                if reduce.is_some() && !is_leanvec {
                    eprintln!(
                        "warning: SVS REDUCE ignored — REDUCE requires a LeanVec COMPRESSION (COMPRESSION={:?})",
                        compression
                    );
                    reduce = None;
                }
                // num_attrs counts the tokens that follow (2 per key/value pair):
                // TYPE,DIM,DISTANCE_METRIC,GRAPH_MAX_DEGREE,CONSTRUCTION_WINDOW_SIZE = 10,
                // + COMPRESSION (2), + REDUCE (2). A miscount is fatal at FT.CREATE.
                let num_attrs = 10
                    + if compression.is_some() { 2 } else { 0 }
                    + if reduce.is_some() { 2 } else { 0 };
                cmd.arg(num_attrs);
                cmd.arg("TYPE").arg(&self.config.data_type);
                cmd.arg("DIM").arg(vector_size);
                cmd.arg("DISTANCE_METRIC").arg(distance_metric);
                cmd.arg("GRAPH_MAX_DEGREE").arg(graph_max_degree);
                cmd.arg("CONSTRUCTION_WINDOW_SIZE")
                    .arg(construction_window_size);
                if let Some(ref c) = compression {
                    cmd.arg("COMPRESSION").arg(c);
                }
                if let Some(r) = reduce {
                    cmd.arg("REDUCE").arg(r);
                }
            } else {
                // HNSW (default): TYPE+DIM+DISTANCE_METRIC + M + EF_CONSTRUCTION.
                let num_attrs = 6 + 2 + 2;
                cmd.arg(num_attrs);
                cmd.arg("TYPE").arg(&self.config.data_type);
                cmd.arg("DIM").arg(vector_size);
                cmd.arg("DISTANCE_METRIC").arg(distance_metric);
                cmd.arg("M").arg(self.config.m);
                cmd.arg("EF_CONSTRUCTION").arg(self.config.ef_construction);
            }
        }

        // Add schema fields from dataset config for filtering
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    match ft {
                        // keyword/uuid/bool are all exact-match TAG fields.
                        // uuid & bool values are single tokens (a UUID string, or
                        // "true"/"false"); SEPARATOR ; + SORTABLE UNF matches the
                        // keyword treatment so exact TAG matching works.
                        "keyword" | "uuid" | "bool" => {
                            cmd.arg(field_name)
                                .arg("TAG")
                                .arg("SEPARATOR")
                                .arg(";")
                                .arg("SORTABLE")
                                .arg("UNF");
                        }
                        "int" | "float" => {
                            cmd.arg(field_name).arg("NUMERIC").arg("SORTABLE");
                        }
                        // datetime is stored as epoch seconds (see upload) and
                        // indexed NUMERIC so range filters work with either ISO or
                        // numeric bounds.
                        "datetime" => {
                            cmd.arg(field_name).arg("NUMERIC").arg("SORTABLE");
                        }
                        "text" => {
                            cmd.arg(field_name).arg("TEXT").arg("SORTABLE");
                        }
                        "geo" => {
                            cmd.arg(field_name).arg("GEO").arg("SORTABLE");
                        }
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
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let redis_url = self.redis_url.clone();
                let config = self.config.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let client = match redis::Client::open(redis_url.as_str()) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };
                    let mut conn = match client.get_connection() {
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

                        // Upload the batch, reconnecting + retrying on a TRANSIENT
                        // connection error (broken pipe / reset / EOF / timeout —
                        // common under heavy index-build load behind a managed-Redis
                        // proxy). Without this a single dropped socket aborted the
                        // whole sweep (#160). Genuine command errors (WRONGTYPE,
                        // syntax, …) are NOT transient and fail fast.
                        let mut attempt = 0usize;
                        let batch_result = loop {
                            match upload_batch_internal(
                                &mut conn,
                                &config,
                                &ids[batch_start..batch_end],
                                &vectors[batch_start..batch_end],
                                &metadata[batch_start..batch_end],
                            ) {
                                Ok(()) => break Ok(()),
                                Err(e) => {
                                    if !is_transient_conn_error(&e) || attempt >= MAX_UPLOAD_RETRIES
                                    {
                                        break Err(format!(
                                            "batch {idx} failed after {attempt} \
                                             reconnect attempt(s): {e}"
                                        ));
                                    }
                                    attempt += 1;
                                    // Bounded exponential backoff: 100·2^(n-1) ms,
                                    // capped at 2s.
                                    let backoff_ms = (100u64 << (attempt - 1)).min(2000);
                                    std::thread::sleep(std::time::Duration::from_millis(
                                        backoff_ms,
                                    ));
                                    // Re-open the connection; if that also fails
                                    // transiently we retry again next iteration.
                                    if let Ok(c) = client.get_connection() {
                                        conn = c;
                                    }
                                }
                            }
                        };
                        if let Err(e) = batch_result {
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

    /// Wait until FT.INFO reports num_docs >= expected.
    fn wait_for_indexing(&self, expected: usize) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let max_wait = 120; // seconds
        let start = Instant::now();

        loop {
            let info: redis::Value = match redis::cmd("FT.INFO")
                .arg(&self.config.index_name)
                .query(&mut conn)
            {
                Ok(v) => v,
                Err(e) => {
                    let msg = e.to_string();
                    // A momentary blip right after a successful upload (a dropped
                    // socket, or a transient `ERRCLUSTER` while the coordinator
                    // re-initializes after a shard restart) must NOT tear down
                    // the whole sweep (#198). While there is still budget, drop
                    // the (possibly dead) connection, reconnect, and retry —
                    // mirroring the #160 upload-retry philosophy on the verify
                    // path. Only a terminal error, or exhausting the budget,
                    // propagates.
                    if is_transient_conn_error(&msg) && start.elapsed().as_secs() < max_wait {
                        eprintln!(
                            "wait_for_indexing: transient FT.INFO error ({msg}); reconnecting and retrying (elapsed {}s/{}s)",
                            start.elapsed().as_secs(),
                            max_wait
                        );
                        // A failed reconnect is itself likely transient; keep the
                        // old handle and let the next iteration try again within
                        // budget rather than aborting here.
                        match self.get_connection() {
                            Ok(c) => conn = c,
                            Err(ce) => eprintln!(
                                "wait_for_indexing: reconnect failed ({ce}); will retry within budget"
                            ),
                        }
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        continue;
                    }
                    return Err(format!("FT.INFO error: {}", msg));
                }
            };

            // Parse num_docs, indexing, and percent_indexed from FT.INFO response.
            // The response can be a flat array (RESP2) or a Map (RESP3).
            let mut num_docs: usize = 0;
            let mut indexing: bool = false;
            let mut percent_indexed: f64 = 1.0;

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

            fn extract_f64(val: &redis::Value) -> f64 {
                match val {
                    redis::Value::BulkString(s) => {
                        String::from_utf8_lossy(s).parse().unwrap_or(1.0)
                    }
                    redis::Value::Int(n) => *n as f64,
                    redis::Value::Double(f) => *f,
                    redis::Value::SimpleString(s) => s.parse().unwrap_or(1.0),
                    _ => 1.0,
                }
            }

            match &info {
                redis::Value::Array(arr) => {
                    // RESP2: alternating key-value pairs (keys can be BulkString or SimpleString)
                    for i in (0..arr.len()).step_by(2) {
                        let key_str = match &arr[i] {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if key_str == "num_docs" {
                            if let Some(val) = arr.get(i + 1) {
                                num_docs = extract_usize(val);
                            }
                        }
                        if key_str == "indexing" {
                            if let Some(val) = arr.get(i + 1) {
                                indexing = extract_bool_nonzero(val);
                            }
                        }
                        if key_str == "percent_indexed" {
                            if let Some(val) = arr.get(i + 1) {
                                percent_indexed = extract_f64(val);
                            }
                        }
                    }
                }
                redis::Value::Map(map) => {
                    // RESP3: key-value map
                    for (k, v) in map {
                        let key_str = match k {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if key_str == "num_docs" {
                            num_docs = extract_usize(v);
                        }
                        if key_str == "indexing" {
                            indexing = extract_bool_nonzero(v);
                        }
                        if key_str == "percent_indexed" {
                            percent_indexed = extract_f64(v);
                        }
                    }
                }
                _ => {
                    eprintln!("Unexpected FT.INFO response type: {:?}", info);
                }
            }

            if num_docs >= expected && !indexing && percent_indexed >= 1.0 {
                println!(
                    "Indexing complete: {} docs in {:.1}s",
                    num_docs,
                    start.elapsed().as_secs_f64()
                );
                return Ok(());
            }

            if start.elapsed().as_secs() > max_wait {
                println!(
                    "Warning: indexing timeout after {}s (num_docs={}/{}, percent_indexed={:.2})",
                    max_wait, num_docs, expected, percent_indexed
                );
                return Ok(());
            }

            if start.elapsed().as_secs() > 0 && start.elapsed().as_secs().is_multiple_of(10) {
                println!(
                    "  indexing... num_docs={}/{}, percent_indexed={:.2}, indexing={}",
                    num_docs, expected, percent_indexed, indexing
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    /// Best-effort FT.INFO for stat collection: reconnect + retry through a few
    /// transient server blips (#198), then give up (`None`) rather than fail —
    /// memory/index telemetry is non-critical, so a persistent error degrades to
    /// a null stat instead of aborting. (`wait_for_indexing` is the strict path
    /// that must actually succeed; it rides the full indexing budget instead.)
    fn ft_info_best_effort(&self) -> Option<redis::Value> {
        const ATTEMPTS: usize = 3;
        let mut conn = self.get_connection().ok()?;
        for attempt in 1..=ATTEMPTS {
            match redis::cmd("FT.INFO")
                .arg(&self.config.index_name)
                .query::<redis::Value>(&mut conn)
            {
                Ok(v) => return Some(v),
                Err(e) => {
                    if is_transient_conn_error(&e.to_string()) && attempt < ATTEMPTS {
                        if let Ok(c) = self.get_connection() {
                            conn = c;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(500 * attempt as u64));
                        continue;
                    }
                    return None;
                }
            }
        }
        None
    }

    /// Filter-only search: run FT.SEARCH with filter conditions only (no KNN).
    /// No precision calculation (no ground truth for filter-only queries).
    fn search_filter_only(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        // Index-existence guard (#151-4): hard-error on a missing/mismatched index
        // instead of writing a silent recall-0.0 file on the --skip-upload path.
        {
            let mut conn = self.get_connection()?;
            redis_utils::ensure_index_exists(&mut conn, &self.config.index_name)?;
        }

        let parallel = params.parallel.unwrap_or(1) as usize;
        let query_timeout: i64 = crate::effective_config::env_parsed("REDIS_QUERY_TIMEOUT", 90_000);

        // Read queries (we only need the filter conditions)
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (_queries, neighbors, conditions) = dataset.read_queries()?;

        // Parse all conditions up front
        let parsed_filters: Vec<QueryFilter<ParsedFilter>> =
            conditions.resolve_all("Redis", parse_conditions)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);

        // Only queries that have filter conditions
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

        // Resolve the index name once (env lookup) instead of per query.
        let index_name = self.config.index_name.clone();

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
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

                    let client = match redis::Client::open(redis_url.as_str()) {
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
                        // Round-robin over available filter queries
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

        // Verify no FT.SEARCH failures occurred
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

/// Encode a vector to the little-endian byte layout RediSearch expects for the
/// given `TYPE`. Integer types (INT8/UINT8) round each component and saturate to
/// the type's range; unknown types fall back to FLOAT32.
fn encode_vector(data_type: &str, vector: &[f32]) -> Vec<u8> {
    match data_type {
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
        // INT8/UINT8: quantized vectors (e.g. Cohere int8 embeddings) arrive as
        // f32; round and clamp to the integer range (float→int casts saturate).
        "INT8" => vector
            .iter()
            .map(|&f| (f.round().clamp(-128.0, 127.0) as i8).to_le_bytes())
            .flat_map(|b| b.into_iter())
            .collect(),
        "UINT8" => vector
            .iter()
            .map(|&f| f.round().clamp(0.0, 255.0) as u8)
            .collect(),
        _ => vector.iter().flat_map(|f| f.to_le_bytes()).collect(),
    }
}

/// Internal batch upload function
/// Max reconnect+retry attempts for a single upload batch that hits a transient
/// connection error before it is treated as fatal (#160).
const MAX_UPLOAD_RETRIES: usize = 5;

/// Whether a stringified error looks TRANSIENT (worth reconnecting + retrying)
/// rather than a genuine command error (worth failing fast). Two families:
///
/// 1. Connection/IO drops. Under heavy index-build load behind a managed-Redis
///    proxy the remote can close the socket ("unexpected end of file", "broken
///    pipe", "connection reset"); those must not abort a whole sweep (#160).
/// 2. Transient server/coordinator errors during a failover or shard restart —
///    notably `ERRCLUSTER: Uninitialized cluster state, could not perform
///    command`, which a managed cluster returns for a few seconds while the
///    coordinator re-initializes (#198). The coordinator recovers on its own, so
///    a reconnect+retry rides straight through it.
///
/// Command errors like WRONGTYPE / wrong number of arguments are NOT matched.
fn is_transient_conn_error(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    const SIGNATURES: &[&str] = &[
        "broken pipe",
        "connection reset",
        "connection aborted",
        "connection refused",
        "unexpected end of file",
        "unexpected eof",
        "timed out",
        "timeout",
        "os error 104", // ECONNRESET
        "os error 32",  // EPIPE
        "os error 111", // ECONNREFUSED
        "os error 110", // ETIMEDOUT
        "connection dropped",
        "not connected",
        // Transient coordinator state during failover / shard restart (#198).
        "errcluster",
        "uninitialized cluster state",
    ];
    SIGNATURES.iter().any(|sig| m.contains(sig))
}

fn upload_batch_internal(
    conn: &mut Connection,
    config: &RedisEngineConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let key = format!("{}{}", config.key_prefix, ids[i]);
        let vec_bytes = encode_vector(&config.data_type, &vectors[i]);

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

        let mut hset_cmd = redis::cmd("HSET");
        hset_cmd.arg(key.as_str());
        for (field_key, field_val) in &fields {
            hset_cmd.arg(&field_key[..]).arg(&field_val[..]);
        }
        pipe.add_command(hset_cmd);
    }

    pipe.query::<()>(conn).map_err(|e| e.to_string())?;
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
// Mirrors Python v0/engine/clients/redis/parser.py RedisConditionParser.
// Converts JSON filter conditions into RediSearch query filter syntax.

/// A filter parameter value to pass to FT.SEARCH PARAMS.
#[derive(Debug, Clone)]
pub(crate) enum FilterParamValue {
    Str(String),
    Int(i64),
    Float(f64),
}

/// Parsed filter: (prefilter_query_string, param_name -> param_value).
pub(crate) type ParsedFilter = (String, HashMap<String, FilterParamValue>);

/// Map a dataset distance name to the RediSearch `DISTANCE_METRIC` value.
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

/// Parse meta_conditions JSON into a RediSearch prefilter string + params.
/// Returns None when no conditions are present.
pub(crate) fn parse_conditions(conditions: &serde_json::Value) -> Option<ParsedFilter> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut counter: usize = 0;
    build_group(obj, &mut counter)
}

/// Build one boolean GROUP (a `{and:[...], or:[...]}` object) into a parenthesised
/// RediSearch clause. Recursive: an entry inside `and`/`or` may itself be a nested
/// group, so this and [`build_subfilters`] call each other. Shares `counter`
/// across the whole tree so param placeholders (`$p0`, `$p1`, …) stay unique.
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

/// Build individual subfilters from an array of condition entries. Each entry is
/// either a nested boolean group (`{and:[...]}` / `{or:[...]}`) — recursed via
/// [`build_group`] — or a leaf `{field: {op: criteria}}`.
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

/// Build a single filter expression from a field condition.
fn build_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    match condition_type {
        "match" => {
            // match_any (IN-list) takes precedence over exact {value}, which
            // takes precedence over full-text {text}.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                Some(build_match_any_filter(field_name, any, counter))
            } else if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                Some(build_text_filter(field_name, text, counter))
            } else {
                build_exact_match_filter(field_name, criteria, counter)
            }
        }
        "range" => build_range_filter(field_name, criteria, counter),
        "geo" => build_geo_filter(field_name, criteria, counter),
        _ => None,
    }
}

/// Build a `match_any` (IN-list) filter, the OR-of-values semantics that mirror
/// qdrant's `Condition::matches(field, Vec)`.
///
/// - All-integer list -> NUMERIC OR of single-value ranges
///   `(@f:[$a $a] | @f:[$b $b])` (field indexed NUMERIC).
/// - Otherwise -> TAG OR `@f:{$a | $b}` over the (non-empty) string values
///   (field indexed TAG). Empty-string tokens are dropped (they are invalid
///   TAG syntax and can never match an exact keyword).
/// - Empty / no representable values -> a never-match `(@f:{$s} -@f:{$s})`
///   (a tag AND not-that-tag contradiction) so an empty IN-set matches NOTHING
///   rather than being dropped — dropping the sole clause would leave no
///   prefilter and run kNN over ALL docs (the inverse of the filter). This
///   assumes a TAG field, the realistic case for a keyword IN-list.
///
/// NOTE: RediSearch TAG matching is case-INSENSITIVE, whereas qdrant keyword
/// match is case-sensitive; for mixed-case data the two engines can select
/// different documents. All shipped keyword datasets use consistent casing.
fn build_match_any_filter(
    field_name: &str,
    any: &[serde_json::Value],
    counter: &mut usize,
) -> ParsedFilter {
    let mut params = HashMap::new();

    // All-integer list -> NUMERIC OR.
    if !any.is_empty() && any.iter().all(|v| v.is_i64()) {
        let clauses: Vec<String> = any
            .iter()
            .filter_map(|v| v.as_i64())
            .map(|i| {
                let p = format!("{}_{}", field_name, counter);
                *counter += 1;
                params.insert(p.clone(), FilterParamValue::Int(i));
                format!("@{}:[${} ${}]", field_name, p, p)
            })
            .collect();
        return (format!("({})", clauses.join(" | ")), params);
    }

    // Otherwise TAG OR over the non-empty string values.
    let tokens: Vec<String> = any
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| {
            let p = format!("{}_{}", field_name, counter);
            *counter += 1;
            params.insert(p.clone(), FilterParamValue::Str(s.to_string()));
            format!("${}", p)
        })
        .collect();

    if tokens.is_empty() {
        // Never-match: `@f:{$s} -@f:{$s}` is an unsatisfiable contradiction.
        let p = format!("{}_{}", field_name, counter);
        *counter += 1;
        params.insert(
            p.clone(),
            FilterParamValue::Str("__match_any_never_match__".to_string()),
        );
        return (
            format!("(@{0}:{{${1}}} -@{0}:{{${1}}})", field_name, p),
            params,
        );
    }

    (
        format!("@{}:{{{}}}", field_name, tokens.join(" | ")),
        params,
    )
}

/// Build exact match filter: string → @field:{$param}, numeric → @field:[$param $param]
fn build_exact_match_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let value = criteria.get("value")?;
    let param_name = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();

    // bool → TAG match on the literal "true"/"false" token. Checked before the
    // numeric arms because serde treats JSON `true`/`false` as neither i64 nor
    // f64 (and never as a string).
    if let Some(b) = value.as_bool() {
        let token = if b { "true" } else { "false" };
        params.insert(param_name.clone(), FilterParamValue::Str(token.to_string()));
        return Some((format!("@{}:{{${}}}", field_name, param_name), params));
    }

    if let Some(s) = value.as_str() {
        params.insert(param_name.clone(), FilterParamValue::Str(s.to_string()));
        Some((format!("@{}:{{${}}}", field_name, param_name), params))
    } else if let Some(i) = value.as_i64() {
        params.insert(param_name.clone(), FilterParamValue::Int(i));
        Some((
            format!("@{}:[${} ${}]", field_name, param_name, param_name),
            params,
        ))
    } else if let Some(f) = value.as_f64() {
        params.insert(param_name.clone(), FilterParamValue::Float(f));
        Some((
            format!("@{}:[${} ${}]", field_name, param_name, param_name),
            params,
        ))
    } else {
        None
    }
}

/// Build a full-text filter over a TEXT field: `@field:($param)`.
///
/// The search terms are passed as a single Str param (DIALECT 2 substitutes it
/// into the text-search context), so RediSearch tokenizes and matches them
/// against the indexed TEXT field. An empty/blank query would be invalid text
/// syntax, so it degrades to a never-match `(@f:($p) -@f:($p))` contradiction
/// rather than an empty clause (which, as the sole prefilter, would run kNN over
/// ALL docs — the inverse of the intended filter).
fn build_text_filter(field_name: &str, text: &str, counter: &mut usize) -> ParsedFilter {
    let param_name = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();
    let trimmed = text.trim();

    if trimmed.is_empty() {
        params.insert(
            param_name.clone(),
            FilterParamValue::Str("__text_never_match__".to_string()),
        );
        return (
            format!("(@{0}:(${1}) -@{0}:(${1}))", field_name, param_name),
            params,
        );
    }

    params.insert(
        param_name.clone(),
        FilterParamValue::Str(trimmed.to_string()),
    );
    (format!("@{}:(${})", field_name, param_name), params)
}

/// Build range filter: @field:[-inf ($param_lt], @field:[($param_gt +inf], etc.
fn build_range_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let param_prefix = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();
    let mut clauses = Vec::new();

    // (suffix, condition key, clause template). `gt` is EXCLUSIVE (`[($p +inf]`),
    // `gte` inclusive; `lt` exclusive, `lte` inclusive.
    let bounds = [
        ("lt", "lt", "@{f}:[-inf (${p}]"),
        ("gt", "gt", "@{f}:[(${p} +inf]"),
        ("lte", "lte", "@{f}:[-inf ${p}]"),
        ("gte", "gte", "@{f}:[${p} +inf]"),
    ];
    for (key, suffix, tmpl) in bounds {
        if let Some(v) = criteria.get(key) {
            let pname = format!("{}_{}", param_prefix, suffix);
            insert_number_param(&mut params, &pname, v);
            // Only emit the clause when the bound actually parsed into a param —
            // otherwise a `$param` with no PARAMS entry makes FT.SEARCH
            // hard-error ("Parameter not found").
            if params.contains_key(&pname) {
                clauses.push(tmpl.replace("{f}", field_name).replace("{p}", &pname));
            }
        }
    }

    if clauses.is_empty() {
        return None;
    }

    Some((clauses.join(" "), params))
}

/// Build geo filter: @field:[$lon $lat $radius m]
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

/// Insert a JSON number (or ISO-8601 datetime / numeric string) into the params
/// map as a NUMERIC bound.
///
/// - integer / float JSON → the numeric value.
/// - ISO-8601 string → epoch **seconds** (datetime range bound). This lets a
///   `datetime` field be filtered with ISO bounds, e.g.
///   `{"range":{"gte":"2021-01-01T00:00:00Z"}}`.
/// - other numeric string → parsed as f64 (accepts a numeric-epoch bound too, so
///   both ISO and raw-epoch datetime bounds work — better than upstream's
///   ISO-only handling).
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
            // Epoch is whole seconds — emit as an integer param (more robust than
            // a float across RediSearch/ValkeySearch NUMERIC param substitution).
            params.insert(name.to_string(), FilterParamValue::Int(epoch as i64));
        } else if let Ok(i) = s.parse::<i64>() {
            params.insert(name.to_string(), FilterParamValue::Int(i));
        } else if let Ok(f) = s.parse::<f64>() {
            params.insert(name.to_string(), FilterParamValue::Float(f));
        }
    }
}

/// Combine AND and OR subfilters into a single prefilter expression.
/// AND clauses are space-joined, OR clauses are pipe-joined.
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
        .arg(4)
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Add filter params
    if !params.is_empty() {
        cmd.arg("PARAMS").arg(params.len() * 2);
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Str(s) => {
                    cmd.arg(s.as_str());
                }
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

    // First element is the total count
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

/// Build the FT.SEARCH KNN query string for the given filter.
///
/// This is pure client-side request construction (string formatting) and is
/// deliberately kept OUT of the per-query timed window: it is precomputed once
/// per query before the parallel search region so the measured latency wraps
/// only the RPC round-trip + reply parse, matching pgvector/qdrant. The query
/// vector is passed as a `$vec_param` PARAM, so this structure is identical
/// across queries that share a filter.
/// True when the configured algorithm is SVS-VAMANA (Intel Scalable Vector
/// Search). Matched by substring so `svs-vamana` / `SVS-VAMANA` / `SVS` all
/// count. SVS-VAMANA takes GRAPH_MAX_DEGREE/CONSTRUCTION_WINDOW_SIZE at build
/// time and SEARCH_WINDOW_SIZE at query time, in place of HNSW's
/// M/EF_CONSTRUCTION and EF_RUNTIME.
fn is_svs(algorithm: &str) -> bool {
    algorithm.to_uppercase().contains("SVS")
}

/// Resolve the runtime search-window value bound as `$EF` (see
/// `build_knn_query_str`). Calibration writes to `search_params.ef` when
/// `calibration_param` is `"ef"` (HNSW), but SVS-VAMANA calibrates
/// `"SEARCH_WINDOW_SIZE"`, which lands in the `extra` catch-all instead —
/// so that key is checked as a fallback.
fn resolve_ef(params: &SearchParams) -> i64 {
    let inner = params.search_params.as_ref();
    inner
        .and_then(|sp| sp.ef)
        .or_else(|| {
            inner
                .and_then(|sp| sp.extra.as_ref())
                .and_then(|e| e.get("SEARCH_WINDOW_SIZE"))
                .and_then(|v| v.as_i64())
        })
        .unwrap_or(64)
}

fn build_knn_query_str(
    algorithm: &str,
    hybrid_policy: &str,
    filter: Option<&ParsedFilter>,
) -> String {
    // Runtime search-window param, bound as `$EF` in build_ft_search_cmd:
    //   HNSW → EF_RUNTIME, SVS-VAMANA → SEARCH_WINDOW_SIZE (its EF_RUNTIME
    //   analogue). ADHOC_BF is brute-force, so no graph-traversal window applies.
    let knn_conditions = if hybrid_policy == "ADHOC_BF" {
        ""
    } else if is_svs(algorithm) {
        "SEARCH_WINDOW_SIZE $EF"
    } else if algorithm.to_uppercase() == "HNSW" {
        "EF_RUNTIME $EF"
    } else {
        ""
    };

    // Build hybrid policy suffix
    let hybrid_suffix = if !hybrid_policy.is_empty() {
        format!("=>{{$HYBRID_POLICY: {} }}", hybrid_policy)
    } else {
        String::new()
    };

    // Prefilter: use filter expression or "*" (match all)
    let prefilter = filter.map(|(expr, _)| expr.as_str()).unwrap_or("*");

    format!(
        "{}=>[KNN $K @vector $vec_param {} AS vector_score]{}",
        prefilter, knn_conditions, hybrid_suffix
    )
}

/// Build the FT.SEARCH KNN `redis::Cmd` — PURE client-side request construction
/// (arg assembly, `top`/`ef` stringification, PARAMS binding), no I/O.
///
/// This is deliberately kept OUT of the per-query timed window: the caller
/// builds the `Cmd` before starting the latency timer and times only the RPC +
/// reply parse (via `exec_ft_search`), so the measured latency wraps only
/// legitimate server latency, matching the Vertex boundary. `index_name` is
/// resolved ONCE by the caller (not re-read from the environment per query);
/// `vec_bytes` (the encoded query blob) and `query_str` are precomputed too.
#[allow(clippy::too_many_arguments)]
fn build_ft_search_cmd(
    index_name: &str,
    vec_bytes: &[u8],
    query_str: &str,
    top: usize,
    ef: i64,
    algorithm: &str,
    hybrid_policy: &str,
    query_timeout: i64,
    filter: Option<&ParsedFilter>,
) -> redis::Cmd {
    let mut cmd = redis::cmd("FT.SEARCH");
    cmd.arg(index_name)
        .arg(query_str)
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Count params: vec_param(2) + K(2) + optional EF(2) + filter params (2 each).
    // The `$EF` param backs HNSW's EF_RUNTIME and SVS-VAMANA's SEARCH_WINDOW_SIZE
    // (see build_knn_query_str), so bind it for either algorithm — and only when
    // that query actually references `$EF` (matches build_knn_query_str exactly,
    // else the bind would be an inert or missing param).
    let binds_ef =
        (algorithm.to_uppercase() == "HNSW" || is_svs(algorithm)) && hybrid_policy != "ADHOC_BF";
    let filter_param_count = filter.as_ref().map(|(_, p)| p.len() * 2).unwrap_or(0);
    let mut base_param_count = 4; // vec_param + K
    if binds_ef {
        base_param_count += 2; // EF
    }
    let total_param_count = base_param_count + filter_param_count;

    cmd.arg("PARAMS").arg(total_param_count);
    cmd.arg("vec_param").arg(vec_bytes);
    cmd.arg("K").arg(top.to_string());

    if binds_ef {
        cmd.arg("EF").arg(ef.to_string());
    }

    // Add filter params
    if let Some((_, params)) = filter {
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Str(s) => {
                    cmd.arg(s.as_str());
                }
                FilterParamValue::Int(i) => {
                    cmd.arg(*i);
                }
                FilterParamValue::Float(f) => {
                    cmd.arg(*f);
                }
            }
        }
    }

    cmd
}

/// Execute a prebuilt FT.SEARCH `Cmd` and parse the reply → (id, score) pairs.
/// This is the ONLY part that belongs in the timed window: the RPC round-trip
/// plus reply parse.
fn exec_ft_search(conn: &mut Connection, cmd: &redis::Cmd) -> Result<Vec<(i64, f64)>, String> {
    // Query the raw Value (not Vec<Value>) so both a RESP2 array and a RESP3 map
    // deserialize; parse_ft_search_response dispatches on the shape.
    let response: redis::Value = cmd
        .query(conn)
        .map_err(|e| format!("FT.SEARCH error: {}", e))?;

    parse_ft_search_response(&response)
}

/// Execute an FT.SEARCH KNN query with optional prefilter (build + exec).
///
/// Convenience wrapper used by the mixed (search+update) path, which
/// intentionally builds inside its timed window. The pure-KNN `search()` path
/// instead calls `build_ft_search_cmd` before the timer and `exec_ft_search`
/// inside it, so construction is not timed.
#[allow(clippy::too_many_arguments)]
fn ft_search_knn(
    conn: &mut Connection,
    index_name: &str,
    vec_bytes: &[u8],
    query_str: &str,
    top: usize,
    ef: i64,
    algorithm: &str,
    hybrid_policy: &str,
    query_timeout: i64,
    filter: Option<&ParsedFilter>,
) -> Result<Vec<(i64, f64)>, String> {
    let cmd = build_ft_search_cmd(
        index_name,
        vec_bytes,
        query_str,
        top,
        ef,
        algorithm,
        hybrid_policy,
        query_timeout,
        filter,
    );
    exec_ft_search(conn, &cmd)
}

/// Single-record HSET update (for mixed benchmark).
fn hset_single(
    conn: &mut Connection,
    config: &RedisEngineConfig,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
) -> Result<(), String> {
    let key = format!("{}{}", config.key_prefix, id);
    let vec_bytes = encode_vector(&config.data_type, vector);

    let mut cmd = redis::cmd("HSET");
    cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);

    if let Some(meta) = metadata {
        for (k, v) in &meta.fields {
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

    cmd.query::<()>(conn)
        .map_err(|e| format!("HSET update error: {}", e))
}

/// Format a string metadata field for storage. `datetime` schema fields whose
/// value is an ISO-8601 string are converted to epoch seconds so the NUMERIC
/// index/range filters match; every other field is stored verbatim. A value
/// that is already numeric (epoch) is left as-is (RFC3339 parse fails → raw).
fn encode_string_field(config: &RedisEngineConfig, key: &str, value: &str) -> String {
    if config.datetime_fields.contains(key) {
        if let Some(epoch) = datetime_to_epoch_secs(value) {
            return (epoch as i64).to_string();
        }
    }
    value.to_string()
}

/// Collect the schema field names typed `datetime`.
fn datetime_fields_from_schema(dataset: &Dataset) -> HashSet<String> {
    prime_datetime_fields(dataset.config.schema.as_ref())
}

/// Pure schema → datetime-field-set priming. Extracted so it can be primed both
/// during `configure()` (upload path) and at the start of `search()` (the
/// `--skip-upload` path, where configure() never runs), and unit-tested without
/// a full `Dataset`.
fn prime_datetime_fields(schema: Option<&serde_json::Value>) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Some(schema) = schema.and_then(|s| s.as_object()) {
        for (field, ty) in schema {
            if ty.as_str() == Some("datetime") {
                set.insert(field.clone());
            }
        }
    }
    set
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for RedisEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let mut conn = self.get_connection()?;

        // Record which schema fields are `datetime` so upload can convert their
        // ISO-8601 payload values to epoch seconds for the NUMERIC index.
        self.config.datetime_fields = Arc::new(datetime_fields_from_schema(dataset));

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
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        // Shared-corpus mode (#188): if the shared prefix is already populated
        // with the full corpus (a prior config in the sweep uploaded it), skip the
        // re-upload entirely and just report it — this is what turns an N-config
        // sweep's ingest from N× into 1×. The index for THIS config was already
        // (re)built over the shared corpus in configure()/create_index.
        //
        // The completeness target comes from `corpus_completeness_target()`, which
        // MEASURES the corpus on disk rather than trusting the declared
        // `vector_count` — a declared count smaller than the real corpus used to
        // satisfy this gate after a fraction of the keys and hand the sweep a
        // silently-truncated corpus (#224). `None` means we cannot confirm a
        // COMPLETE corpus, so we fall through and upload normally.
        if self.config.shared_corpus {
            if let Some(expected) = dataset.corpus_completeness_target()? {
                let mut conn = self.get_connection()?;
                let existing = count_prefix_keys(&mut conn, &self.config.key_prefix);
                if shared_corpus_is_complete(existing, expected) {
                    println!(
                        "Shared corpus already populated ({existing}/{expected} keys under prefix \
                         '{}'); skipping re-upload (#188 upload-once/build-many).",
                        self.config.key_prefix
                    );
                    // Skipping the UPLOAD must not also skip waiting for the
                    // INDEX. This config's index was created in configure() over
                    // a keyspace that was already populated, so RediSearch
                    // backfills it asynchronously from the existing hashes; the
                    // uploading config waits for that below (via
                    // `wait_for_indexing`) but a skipping config used to return
                    // immediately and start searching a half-built index. That
                    // reports low recall with no error — the same silent-wrong
                    // -result failure the completeness gate above guards against,
                    // one step later. It also keeps the sweep honest: index-build
                    // time is part of ingest cost for every other config, so it
                    // is counted in total_time here too (upload_time stays 0 —
                    // the corpus really was uploaded once).
                    let index_start = Instant::now();
                    self.wait_for_indexing(existing)?;
                    let index_time = index_start.elapsed().as_secs_f64();
                    println!(
                        "Index time (backfill over the shared corpus): {:.3}s",
                        index_time
                    );
                    return Ok(UploadStats {
                        upload_time: 0.0,
                        total_time: index_time,
                        upload_count: existing,
                        parallel: self.config.parallel,
                        batch_size: self.config.batch_size,
                        memory_usage: None,
                    });
                }
            }
        }

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

        // Wait for RediSearch indexing to complete. The index-build wait is part
        // of the ingest cost and must be included in total_time for cross-engine
        // comparability (mirrors mongodb; matches v0 which times through
        // post_upload()). Excluding it made every engine but mongodb look
        // artificially fast to index.
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
        // Prime the datetime field-type map from the schema so datetime range
        // filters are built with the correct field type even on the
        // `--skip-upload` path, where configure() (which normally primes it) is
        // never called. Idempotent: on the upload path this reproduces exactly
        // what configure() already set.
        self.config.datetime_fields = Arc::new(datetime_fields_from_schema(dataset));

        if self.config.skip_vector_index {
            return self.search_filter_only(dataset, params, num_queries);
        }

        // Index-existence guard (#151-4): on the --skip-upload path a missing or
        // mismatched index would otherwise write a silent recall-0.0 result file.
        {
            let mut conn = self.get_connection()?;
            redis_utils::ensure_index_exists(&mut conn, &self.config.index_name)?;
        }

        let ef = resolve_ef(params);
        let parallel = params.parallel.unwrap_or(1) as usize;
        let hybrid_policy = crate::effective_config::env_or("REDIS_HYBRID_POLICY", "");
        let query_timeout: i64 = crate::effective_config::env_parsed("REDIS_QUERY_TIMEOUT", 90_000);

        // Read queries, ground truth, and filter conditions
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;
        if queries.is_empty() {
            return Err("dataset contains no search queries".to_string());
        }

        // Parse all conditions up front (before timing begins)
        let parsed_filters: Vec<QueryFilter<ParsedFilter>> =
            conditions.resolve_all("Redis", parse_conditions)?;

        // When top is explicitly set, use it for all queries.
        // When not set, use per-query ground truth count (matches Python v0 behavior
        // where top defaults to len(query.expected_result) per query).
        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let open_loop = OpenLoopPlan::from_params(params)?;
        let closed_loop_duration = closed_loop_duration(params)?;
        let num_to_run = if closed_loop_duration.is_some() {
            usize::MAX
        } else {
            open_loop.map(|p| p.total_requests).unwrap_or_else(|| {
                if num_queries > 0 {
                    (num_queries as usize).min(queries.len())
                } else {
                    queries.len()
                }
            })
        };

        // Precompute all client-side request construction BEFORE the timed
        // region so the per-query timed window wraps ONLY the RPC round-trip +
        // reply parse (matching pgvector/qdrant). `encode_vector` allocates and
        // — for quantized index types — runs a per-element numeric conversion
        // of the query vector; that is client work, not server latency. The
        // FT.SEARCH query string is likewise pure string formatting; the vector
        // is bound as a `$vec_param` PARAM so the string is identical across
        // queries sharing a filter. Both are shared read-only across workers.
        let encoded_queries: Vec<Vec<u8>> = queries
            .iter()
            .map(|q| encode_vector(&self.config.data_type, q))
            .collect();
        let query_strs: Vec<String> = parsed_filters
            .iter()
            .map(|f| build_knn_query_str(&self.config.algorithm, &hybrid_policy, f.as_ref()))
            .collect();

        // Search execution. Each worker accumulates samples into thread-local
        // buffers and returns them on join; the main thread concatenates. This
        // keeps the timed hot loop free of the per-query cross-thread Mutex<Vec>
        // pushes (5 locks/query) that serialized workers at high parallelism.
        // Metrics are order-independent (percentiles sort), so results are
        // unchanged. The work counter stays a single atomic but uses Relaxed
        // (only its own monotonicity matters, not ordering against other memory).
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(if closed_loop_duration.is_some() {
            0
        } else {
            num_to_run
        });
        // Resolve the index name ONCE (env lookup) so it is not re-read from the
        // environment inside the per-query timed window.
        let index_name = self.config.index_name.clone();

        // Gate-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window — for open-loop and closed-loop-duration
        // alike (mirrors the Vertex engine). Every worker connects + primes, then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant
        // and releases everyone, so the measurement clock starts only once all
        // workers are warm and poised. The gate is count-agnostic: a worker that
        // fails to connect, panics, or is never started by the OS settles its
        // ticket and turns the run into an error instead of a hang (#214).

        let sample_capacity = if closed_loop_duration.is_some() {
            queries.len()
        } else {
            num_to_run
        };
        let mut times: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut precs: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut recs: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut mrs: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut nds: Vec<f64> = Vec::with_capacity(sample_capacity);
        let mut schedule_delays: Vec<f64> = Vec::new();
        let mut end_to_end_latencies: Vec<f64> = Vec::new();
        let mut dropped_queries = 0usize;
        let mut late_queries = 0usize;
        let query_count = queries.len();

        // Workers recycle queries with `query_pos = idx % query_count` and index
        // `neighbors[query_pos]` in lock-step; a ground-truth set shorter than the
        // query set would panic a worker (aborting the whole run via
        // `h.join().unwrap()`). Validate once up front for a clean error instead.
        if neighbors.len() < queries.len() {
            return Err("dataset ground-truth shorter than query set".into());
        }

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "redis-search", parallel);
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
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
                    // Thread-local sample buffers — no cross-thread synchronization
                    // in the timed loop.
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut sd = Vec::new();
                    let mut e2e = Vec::new();
                    let mut dropped = 0usize;
                    let mut late = 0usize;
                    let mut pb_pending: u64 = 0;

                    let client = match redis::Client::open(redis_url.as_str()) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the
                            // run at parallel-1 real concurrency while still being
                            // reported as `parallel`. Settle the ticket with the
                            // reason; the coordinator turns it into a hard error.
                            ticket.fail(format!("redis client open failed: {e}"));
                            return (t, p, r, mr, nd, sd, e2e, dropped, late);
                        }
                    };
                    let mut conn = match client.get_connection() {
                        Ok(c) => c,
                        Err(e) => {
                            ticket.fail(format!("redis connect failed: {e}"));
                            return (t, p, r, mr, nd, sd, e2e, dropped, late);
                        }
                    };
                    // Bound a blackholed endpoint to a finite wait instead of the
                    // kernel TCP-retransmit budget (minutes). Kept above the 90s
                    // server-side FT.SEARCH TIMEOUT so it never preempts it.
                    conn.set_read_timeout(Some(Duration::from_secs(120))).ok();
                    conn.set_write_timeout(Some(Duration::from_secs(120))).ok();

                    // Prime this connection with ONE discarded query so the cold
                    // first round-trip is not inside the measured window. Best
                    // effort: errors are ignored and its sample is NOT recorded.
                    {
                        let prime_top = explicit_top.unwrap_or(10);
                        let prime_cmd = build_ft_search_cmd(
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
                        let _ = exec_ft_search(&mut conn, &prime_cmd);
                    }

                    // Signal "connected + primed", then block until the coordinator
                    // stamps the shared measurement start and releases everyone.
                    let Some(start_time) = ticket.arrive_and_wait() else {
                        return (t, p, r, mr, nd, sd, e2e, dropped, late);
                    };

                    loop {
                        if closed_loop_duration
                            .map(|duration| Instant::now() >= start_time + duration)
                            .unwrap_or(false)
                        {
                            break;
                        }
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }
                        let query_pos = idx % query_count;

                        let scheduled_at = open_loop.map(|plan| {
                            let delay = plan.wait_for_slot(start_time, idx);
                            let will_drop = delay > plan.max_lateness;
                            // Count schedule-delay + lateness for DISPATCHED
                            // requests only. A dropped (never-run) request must not
                            // push its delay into schedule_delays nor increment
                            // `late`, or late_queries >= dropped_queries always and
                            // the schedule-delay percentiles are inflated by
                            // requests that never dispatched. It is accounted
                            // solely as `dropped` below.
                            if !will_drop {
                                sd.push(delay.as_secs_f64());
                                if plan.is_late(delay) {
                                    late += 1;
                                }
                            }
                            (plan.scheduled_at(start_time, idx), will_drop)
                        });
                        if scheduled_at.map(|(_, drop)| drop).unwrap_or(false) {
                            dropped += 1;
                            pb_pending += 1;
                            if pb_pending >= 256 {
                                pb.inc(pb_pending);
                                pb_pending = 0;
                            }
                            continue;
                        }

                        // Per-query top: use explicit top if set, otherwise
                        // use ground truth count (matches Python v0 behavior)
                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[query_pos].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        // Build the FT.SEARCH command OUTSIDE the timer — pure
                        // client-side arg assembly on precomputed inputs (blob,
                        // query string, resolved index name). `top` is known here.
                        let cmd = build_ft_search_cmd(
                            index_name,
                            &encoded_queries[query_pos],
                            &query_strs[query_pos],
                            top,
                            ef,
                            &algorithm,
                            &hybrid_policy,
                            query_timeout,
                            parsed_filters[query_pos].as_ref(),
                        );

                        // Timed window: ONLY the RPC round-trip + reply parse,
                        // matching the Vertex engine's boundary.
                        let query_start = Instant::now();
                        let results = exec_ft_search(&mut conn, &cmd);
                        let query_time = query_start.elapsed().as_secs_f64();
                        let completion = Instant::now();

                        // Record latency + quality only for successful queries. A
                        // failed query is surfaced and counted (as num_to_run minus
                        // successes after the loop), never folded into the means as
                        // a 0-recall / fast-latency sample.
                        match results {
                            Ok(result_ids) => {
                                let ordered_ids: Vec<i64> =
                                    result_ids.iter().map(|(id, _)| *id).collect();
                                let m = compute_metrics(&ordered_ids, &neighbors[query_pos], top);
                                t.push(query_time);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                                if let Some((scheduled, _)) = scheduled_at {
                                    e2e.push(
                                        completion
                                            .saturating_duration_since(scheduled)
                                            .as_secs_f64(),
                                    );
                                }
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
                    (t, p, r, mr, nd, sd, e2e, dropped, late)
                })?;
            }

            // Every worker is connected + primed and parked at the gate. Stamp the
            // shared measurement start and release them simultaneously. No arrival
            // offset is needed now — the cold setup is already behind the gate.
            let (per_worker, measured_start) = pool.start()?;

            for (t, p, r, mr, nd, sd, e2e, dropped, late) in per_worker {
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrs.extend(mr);
                nds.extend(nd);
                schedule_delays.extend(sd);
                end_to_end_latencies.extend(e2e);
                dropped_queries += dropped;
                late_queries += late;
            }
            Ok(measured_start)
        })?;

        pb.finish_and_clear();
        // Measure from the post-gate start stamp (workers already primed), so
        // total_time excludes connection setup and the cold first query.
        let total_time = measured_start.elapsed().as_secs_f64();

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        let attempted_queries = query_idx
            .load(Ordering::Relaxed)
            .min(num_to_run)
            .saturating_sub(dropped_queries);

        if times.is_empty() {
            // Genuinely zero attempts (no queries offered/dispatched) is an error.
            // But an overload-shed run that DID dispatch/drop work is a real,
            // informative data point: report rps=0 with the drop accounting rather
            // than collapsing the strongest overload signal into an error.
            if attempted_queries == 0 && dropped_queries == 0 {
                return Err("No searches completed".to_string());
            }
            let mut results = zero_search_results(total_time, top, parallel, attempted_queries);
            if let Some(plan) = open_loop {
                attach_open_loop_metrics(
                    &mut results,
                    plan,
                    &schedule_delays,
                    &end_to_end_latencies,
                    dropped_queries,
                    late_queries,
                );
            }
            return Ok(results);
        }

        // Verify no FT.SEARCH failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH"],
            "search",
            self.commandstats_baseline.as_ref(),
        )?;

        let mut results = crate::engine::compute_search_stats(
            &times,
            &precs,
            &recs,
            &mrs,
            &nds,
            total_time,
            top,
            parallel,
            attempted_queries,
        )?;
        // In open-loop mode, drops are distinct from attempted request failures.
        if let Some(plan) = open_loop {
            attach_open_loop_metrics(
                &mut results,
                plan,
                &schedule_delays,
                &end_to_end_latencies,
                dropped_queries,
                late_queries,
            );
        }
        Ok(results)
    }

    fn search_mixed(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
        ratio: &UpdateSearchRatio,
    ) -> Result<SearchResults, String> {
        // Prime the datetime field-type map from the schema so the update half of
        // the mixed workload encodes datetime payloads as epoch seconds even on
        // the `--skip-upload` path (configure() does not run there). Idempotent.
        self.config.datetime_fields = Arc::new(datetime_fields_from_schema(dataset));

        // Index-existence guard (#151-4): fail loudly on a missing/mismatched index.
        {
            let mut conn = self.get_connection()?;
            redis_utils::ensure_index_exists(&mut conn, &self.config.index_name)?;
        }

        let ef = resolve_ef(params);
        let parallel = params.parallel.unwrap_or(1) as usize;
        let hybrid_policy = crate::effective_config::env_or("REDIS_HYBRID_POLICY", "");
        let query_timeout: i64 = crate::effective_config::env_parsed("REDIS_QUERY_TIMEOUT", 90_000);

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<QueryFilter<ParsedFilter>> =
            conditions.resolve_all("Redis", parse_conditions)?;

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

        // Resolve the index name once (env lookup) rather than per query.
        let index_name = self.config.index_name.clone();

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
        let mut mrs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut nds: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut u_times: Vec<f64> = Vec::new();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
                let config = self.config.clone();
                let algorithm = self.config.algorithm.clone();
                let data_type = self.config.data_type.clone();
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
                let index_name = index_name.as_str();
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

                    let client = match redis::Client::open(redis_url.as_str()) {
                        Ok(c) => c,
                        Err(_) => return (t, p, r, mr, nd, ut),
                    };
                    let mut conn = match client.get_connection() {
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
                            let vec_bytes = encode_vector(&data_type, &queries[idx]);
                            let query_str = build_knn_query_str(
                                &algorithm,
                                &hybrid_policy,
                                parsed_filters[idx].as_ref(),
                            );
                            let results = ft_search_knn(
                                &mut conn,
                                index_name,
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

                            // Record latency + quality only for successful queries;
                            // failures are surfaced, never scored as 0-recall.
                            match results {
                                Ok(result_ids) => {
                                    let ordered_ids: Vec<i64> =
                                        result_ids.iter().map(|(id, _)| *id).collect();
                                    let m = compute_metrics(&ordered_ids, &neighbors[idx], top);
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
                            let _ = hset_single(
                                &mut conn,
                                &config,
                                upd_ids[data_idx],
                                &upd_vectors[data_idx],
                                upd_metadata[data_idx].as_ref(),
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
                mrs.extend(mr);
                nds.extend(nd);
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
            &times, &precs, &recs, &mrs, &nds, total_time, top, parallel, num_to_run,
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
        let mut conn = self.get_connection()?;
        // Normally DD (drop index + this config's docs). In shared-corpus mode
        // (#188) the docs are shared across configs, so drop the index only and
        // leave the corpus for the remaining configs / a later --skip-upload run.
        // The shared corpus is the user's to flush when the sweep is done.
        let mut cmd = redis::cmd("FT.DROPINDEX");
        cmd.arg(&self.config.index_name);
        if !self.config.shared_corpus {
            cmd.arg("DD");
        }
        let _ = cmd.query::<()>(&mut conn);
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

        // Server-wide used_memory is kept only as a SECONDARY/global figure: under
        // #151-4 coexistence it is the SUM of all resident configs, so it cannot
        // attribute memory per graph. The per-config figure is the FT.INFO index
        // size below.
        let info_str: String = redis::cmd("INFO").arg("memory").query(&mut conn).ok()?;
        let used_memory: i64 = parse_used_memory(&info_str);

        // Get FT.INFO for this config's index stats + per-index memory. Reconnect
        // + retry through a transient blip so a momentary hiccup doesn't blank the
        // per-config memory figure (#198); still degrades to None if it persists.
        let ft_info_raw: Option<redis::Value> = self.ft_info_best_effort();
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
        // Vector index stats (HNSW M/EF, num_docs, index memory, ...). Errors on
        // the BEFORE snapshot (index not yet created) → index_info: null. Uses the
        // reconnect+retry best-effort read so a transient blip doesn't blank the
        // stat (#198); a genuinely-absent index still degrades to null.
        let ft_info = self.ft_info_best_effort().map(|v| redis_value_to_json(&v));
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
        encode_vector, is_transient_conn_error, parse_conditions, shared_corpus_is_complete,
        FilterParamValue,
    };
    use crate::config::DatasetConfig;
    use crate::dataset::Dataset;
    use vector_db_benchmark::readers::write_npy_vectors;

    /// A compound ("tar") dataset rooted at an absolute temp dir holding a real
    /// `vectors.npy` of `rows` rows, declaring `declared` in datasets.json.
    /// `datasets_dir().join(<abs>) == <abs>`, so the temp dir resolves directly.
    fn corpus_dataset(dir: &std::path::Path, rows: usize, declared: i64) -> Dataset {
        let vectors: Vec<Vec<f32>> = (0..rows).map(|i| vec![i as f32, 0.0, 0.0]).collect();
        write_npy_vectors(dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
        Dataset::new(DatasetConfig {
            name: "shared-corpus-unit".to_string(),
            dataset_type: Some("tar".to_string()),
            path: serde_json::Value::String(dir.to_str().unwrap().to_string()),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count: Some(declared),
            link: None,
            schema: None,
            description: None,
        })
    }

    /// #188 must keep working: a shared prefix holding the WHOLE corpus is
    /// reused (no re-upload), while anything short of it uploads.
    #[test]
    fn shared_corpus_skips_only_when_the_corpus_is_complete() {
        let dir = tempfile::tempdir().unwrap();
        let ds = corpus_dataset(dir.path(), 1000, 1000);

        let expected = ds
            .corpus_completeness_target()
            .expect("declared count matches the corpus")
            .expect("a measurable local corpus yields a target");
        assert_eq!(expected, 1000);

        // Complete (and over-complete) → skip the re-upload.
        assert!(shared_corpus_is_complete(1000, expected));
        assert!(shared_corpus_is_complete(1001, expected));
        // Incomplete → must upload, however close it is.
        assert!(!shared_corpus_is_complete(999, expected));
        assert!(!shared_corpus_is_complete(100, expected));
        assert!(!shared_corpus_is_complete(0, expected));
    }

    /// Regression for #224: `random-100-match-kw-small-vocab-*` declared
    /// vector_count 100 over a 1,000,000-point corpus, so the gate accepted 100
    /// keys as "already populated" and the sweep scored recall over 0.01% of the
    /// corpus with no error. The declared count must never be able to do that.
    #[test]
    fn declared_count_below_the_real_corpus_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let ds = corpus_dataset(dir.path(), 1000, 100);

        let err = ds
            .corpus_completeness_target()
            .expect_err("under-declared vector_count must be a hard error");
        assert!(err.contains("1000"), "err should name the real size: {err}");
        assert!(
            err.contains("100"),
            "err should name the declared size: {err}"
        );

        // And the same corpus with the count corrected gates on the real size,
        // so the 100 keys that used to satisfy the gate no longer do.
        let fixed = corpus_dataset(dir.path(), 1000, 1000);
        let expected = fixed.corpus_completeness_target().unwrap().unwrap();
        assert!(!shared_corpus_is_complete(100, expected));
    }

    /// A corpus that isn't on disk can't be confirmed complete, so the gate has
    /// no target and the upload proceeds — a stale keyspace is never waved
    /// through on the strength of a declared number alone.
    #[test]
    fn absent_corpus_yields_no_completeness_target() {
        let dir = tempfile::tempdir().unwrap();
        let ds = Dataset::new(DatasetConfig {
            name: "not-downloaded".to_string(),
            dataset_type: Some("tar".to_string()),
            path: serde_json::Value::String(
                dir.path().join("missing").to_str().unwrap().to_string(),
            ),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count: Some(1_000_000),
            link: None,
            schema: None,
            description: None,
        });
        assert_eq!(ds.corpus_completeness_target().unwrap(), None);
    }

    /// The dangerous sibling of the above: the dataset DIRECTORY exists but the
    /// corpus file inside it does not. `extract_tgz` creates the directory and
    /// unpacks in place, and `vectors.npy` is the LAST member of `arxiv.tar.gz`
    /// — so an interrupted download leaves exactly this state, as does deleting
    /// a big `vectors.npy` to reclaim disk while `tests.jsonl` stays behind and
    /// queries keep working. "A directory is here" must never stand in for "the
    /// corpus is complete": that reopens #224 from the corpus side.
    #[test]
    fn present_directory_with_missing_corpus_file_yields_no_target() {
        let dir = tempfile::tempdir().unwrap();
        // Everything an interrupted extract leaves behind EXCEPT vectors.npy.
        std::fs::write(dir.path().join("tests.jsonl"), "{\"query\": [1.0]}\n").unwrap();
        std::fs::write(dir.path().join("payloads.jsonl"), "{}\n").unwrap();

        let ds = Dataset::new(DatasetConfig {
            name: "half-extracted".to_string(),
            dataset_type: Some("tar".to_string()),
            path: serde_json::Value::String(dir.path().to_str().unwrap().to_string()),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count: Some(10),
            link: None,
            schema: None,
            description: None,
        });
        assert_eq!(
            ds.measured_vector_count().unwrap(),
            None,
            "no vectors.npy => nothing measurable"
        );
        assert_eq!(
            ds.corpus_completeness_target().unwrap(),
            None,
            "a measurable layout with its corpus file missing must not fall back \
             to the declared count"
        );

        // Same shape for the jsonl layout (directory present, vectors.jsonl gone).
        let jsonl_dir = tempfile::tempdir().unwrap();
        std::fs::write(jsonl_dir.path().join("queries.jsonl"), "[1.0]\n").unwrap();
        let mut cfg = ds.config.clone();
        cfg.dataset_type = Some("jsonl".to_string());
        cfg.path = serde_json::Value::String(jsonl_dir.path().to_str().unwrap().to_string());
        assert_eq!(
            Dataset::new(cfg).corpus_completeness_target().unwrap(),
            None
        );
    }

    /// The narrow, deliberate exception: layouts with no cheap row count may use
    /// the declared count — but only once their corpus files are actually there.
    #[test]
    fn unmeasurable_layouts_fall_back_only_when_their_files_exist() {
        let dir = tempfile::tempdir().unwrap();
        let sparse = Dataset::new(DatasetConfig {
            name: "sparse-unit".to_string(),
            dataset_type: Some("sparse".to_string()),
            path: serde_json::Value::String(dir.path().to_str().unwrap().to_string()),
            distance: Some("dot".to_string()),
            vector_size: Some(3),
            vector_count: Some(500),
            link: None,
            schema: None,
            description: None,
        });
        assert_eq!(
            sparse.corpus_completeness_target().unwrap(),
            None,
            "empty directory is not a corpus"
        );

        std::fs::write(dir.path().join("data.csr"), b"").unwrap();
        assert_eq!(
            sparse.corpus_completeness_target().unwrap(),
            Some(500),
            "CSR has no cheap row count, so the declared count is all we have"
        );
    }

    #[test]
    fn transient_conn_errors_are_retryable() {
        // Real redis-rs / OS socket-drop signatures → retry (#160).
        for msg in [
            "unexpected end of file",
            "Broken pipe (os error 32)",
            "Connection reset by peer (os error 104)",
            "Connection refused (os error 111)",
            "operation timed out",
            "IoError: Connection aborted",
            "connection dropped",
            // Transient coordinator errors during failover / shard restart (#198).
            "ERRCLUSTER: Uninitialized cluster state, could not perform command",
            "An error was signalled by the server: ERRCLUSTER Uninitialized cluster state",
        ] {
            assert!(is_transient_conn_error(msg), "should retry: {msg}");
        }
    }

    #[test]
    fn command_errors_are_not_retryable() {
        // Genuine server/command errors must fail fast, not spin on retries.
        for msg in [
            "WRONGTYPE Operation against a key holding the wrong kind of value",
            "ERR wrong number of arguments for 'hset' command",
            "OOM command not allowed when used memory > 'maxmemory'",
            "Unknown index name",
        ] {
            assert!(!is_transient_conn_error(msg), "should NOT retry: {msg}");
        }
    }

    #[test]
    fn match_any_string_list_emits_tag_or() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":["red","blue"]}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@color:{$color_0 | $color_1}"), "q={}", q);
        assert!(matches!(params.get("color_0"), Some(FilterParamValue::Str(s)) if s == "red"));
        assert!(matches!(params.get("color_1"), Some(FilterParamValue::Str(s)) if s == "blue"));
    }

    #[test]
    fn match_any_int_list_emits_numeric_or() {
        let cond = serde_json::json!({"and":[{"size":{"match":{"any":[1,2]}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@size:[$size_0 $size_0]"), "q={}", q);
        assert!(q.contains("@size:[$size_1 $size_1]"), "q={}", q);
        assert!(q.contains(" | "), "q={}", q);
        assert!(matches!(
            params.get("size_0"),
            Some(FilterParamValue::Int(1))
        ));
        assert!(matches!(
            params.get("size_1"),
            Some(FilterParamValue::Int(2))
        ));
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // Empty IN-set -> never-match contradiction, not a dropped clause.
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":[]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@color:{"), "expected never-match, q={}", q);
    }

    #[test]
    fn match_any_drops_empty_string_tokens() {
        // Empty-string tokens are invalid TAG syntax; with only such tokens the
        // clause degrades to the never-match rather than emitting `@f:{}`.
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":[""]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(!q.contains("{$color_0 }") && !q.contains("{}"), "q={}", q);
        assert!(q.contains("-@color:{"), "q={}", q);
    }

    #[test]
    fn match_exact_value_still_emits_tag() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"value":"red"}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@color:{$color_0}"), "q={}", q);
        assert!(matches!(params.get("color_0"), Some(FilterParamValue::Str(s)) if s == "red"));
    }

    // ── New filter datatypes: bool / uuid / full-text / datetime ───────────
    use super::{datetime_to_epoch_secs, encode_string_field, ParsedFilter, RedisEngineConfig};
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    #[test]
    fn match_bool_true_emits_tag_true_token() {
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": true}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@flag:{$flag_0}"), "q={}", q);
        assert!(matches!(params.get("flag_0"), Some(FilterParamValue::Str(s)) if s == "true"));
    }

    #[test]
    fn match_bool_false_emits_tag_false_token() {
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": false}}}]});
        let (_q, params) = parse_conditions(&cond).unwrap();
        assert!(matches!(params.get("flag_0"), Some(FilterParamValue::Str(s)) if s == "false"));
    }

    #[test]
    fn match_uuid_value_emits_tag_param() {
        // uuid is a keyword-style string → TAG param match.
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let cond = serde_json::json!({"and":[{"uid":{"match":{"value": uuid}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@uid:{$uid_0}"), "q={}", q);
        assert!(matches!(params.get("uid_0"), Some(FilterParamValue::Str(s)) if s == uuid));
    }

    #[test]
    fn match_text_emits_fulltext_clause() {
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "quick"}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@body:($body_0)"), "q={}", q);
        assert!(matches!(params.get("body_0"), Some(FilterParamValue::Str(s)) if s == "quick"));
    }

    #[test]
    fn match_text_empty_is_never_match() {
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "   "}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@body:("), "expected never-match, q={}", q);
    }

    #[test]
    fn range_datetime_iso_bounds_parse_to_epoch() {
        let cond = serde_json::json!({"and":[{"ts":{"range":{
            "gte": "2021-01-01T00:00:00Z",
            "lt":  "2022-01-01T00:00:00Z"
        }}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@ts:[$ts_0_gte +inf]"), "q={}", q);
        assert!(q.contains("@ts:[-inf ($ts_0_lt]"), "q={}", q);
        // 2021-01-01T00:00:00Z == 1609459200, 2022-01-01T00:00:00Z == 1640995200.
        // Emitted as integer epoch params.
        assert!(matches!(
            params.get("ts_0_gte"),
            Some(FilterParamValue::Int(1609459200))
        ));
        assert!(matches!(
            params.get("ts_0_lt"),
            Some(FilterParamValue::Int(1640995200))
        ));
    }

    #[test]
    fn range_datetime_numeric_epoch_bounds_still_work() {
        // Numeric-epoch bounds (better-than-upstream: upstream is ISO-only).
        let cond = serde_json::json!({"and":[{"ts":{"range":{"gte": 1609459200}}}]});
        let (_q, params) = parse_conditions(&cond).unwrap();
        assert!(matches!(
            params.get("ts_0_gte"),
            Some(FilterParamValue::Int(1609459200))
        ));
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
    fn datetime_tolerates_naive_and_date_only() {
        // Better than upstream (RFC-3339 only): accept naive + date-only bounds.
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
    fn range_gt_is_exclusive_gte_inclusive() {
        // gt must be exclusive `[($p +inf]`; gte inclusive `[$p +inf]`.
        let (gt, _) =
            super::parse_conditions(&serde_json::json!({"and":[{"n":{"range":{"gt": 5}}}]}))
                .unwrap();
        assert!(
            gt.contains("@n:[($n_0_gt +inf]"),
            "gt not exclusive: {}",
            gt
        );
        let (gte, _) =
            super::parse_conditions(&serde_json::json!({"and":[{"n":{"range":{"gte": 5}}}]}))
                .unwrap();
        assert!(
            gte.contains("@n:[$n_0_gte +inf]") && !gte.contains("[("),
            "gte: {}",
            gte
        );
    }

    #[test]
    fn range_unparseable_bound_emits_no_dangling_param() {
        // A bound that can't parse must NOT leave a `$param` with no PARAMS entry.
        let (q, params) =
            super::parse_conditions(&serde_json::json!({"and":[{"n":{"range":{"gte": "nope"}}}]}))
                .unwrap_or_default();
        assert!(
            !q.contains("$n_0_gte") || params.contains_key("n_0_gte"),
            "dangling: {}",
            q
        );
    }

    #[test]
    fn encode_string_field_converts_datetime_and_passes_through_others() {
        let mut dt = HashSet::new();
        dt.insert("ts".to_string());
        let cfg = RedisEngineConfig {
            m: 16,
            ef_construction: 128,
            data_type: "FLOAT32".to_string(),
            algorithm: "hnsw".to_string(),
            batch_size: 1,
            parallel: 1,
            skip_vector_index: false,
            svs_graph_max_degree: None,
            svs_construction_window_size: None,
            svs_compression: None,
            svs_reduce: None,
            index_name: "idx:test".to_string(),
            key_prefix: "test:".to_string(),
            shared_corpus: false,
            datetime_fields: Arc::new(dt),
        };
        // datetime field → epoch seconds string
        assert_eq!(
            encode_string_field(&cfg, "ts", "2021-01-01T00:00:00Z"),
            "1609459200"
        );
        // datetime field already numeric epoch → left as-is
        assert_eq!(encode_string_field(&cfg, "ts", "1609459200"), "1609459200");
        // non-datetime field → verbatim
        assert_eq!(encode_string_field(&cfg, "color", "red"), "red");
    }

    #[test]
    fn encodes_float32_by_default() {
        let v = vec![1.0f32, -2.5];
        let bytes = encode_vector("FLOAT32", &v);
        assert_eq!(bytes.len(), 8); // 2 x f32
        assert_eq!(bytes, encode_vector("SOMETHING_UNKNOWN", &v));
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
    }

    // Legacy SVS configs (dbpedia-calibration.json shape) nest algorithm,
    // data_type, and the svs-vamana_config block under collection_params. new()
    // must resolve all of them so the config actually selects + configures SVS.
    #[test]
    fn resolves_nested_svs_collection_params() {
        use super::{is_svs, RedisEngine};
        let cfg_json = serde_json::json!({
            "name": "dbpedia-cal-svs-LeanVec4x8-div4-float16",
            "engine": "redis",
            "collection_params": {
                "algorithm": "svs-vamana",
                "data_type": "FLOAT16",
                "svs-vamana_config": {
                    "DISTANCE_METRIC": "L2",
                    "GRAPH_MAX_DEGREE": 40,
                    "CONSTRUCTION_WINDOW_SIZE": 250,
                    "compression": "LeanVec4x8",
                    "REDUCE": 384
                }
            },
            "upload_params": { "algorithm": "svs-vamana", "data_type": "FLOAT16" }
        });
        let ec: crate::config::EngineConfig = serde_json::from_value(cfg_json).unwrap();
        let eng = RedisEngine::new(&ec, "localhost").unwrap();
        let c = &eng.config;
        assert_eq!(c.algorithm, "svs-vamana", "nested algorithm resolved");
        assert!(is_svs(&c.algorithm));
        assert_eq!(c.data_type, "FLOAT16", "nested data_type resolved");
        assert_eq!(c.svs_graph_max_degree, Some(40));
        assert_eq!(c.svs_construction_window_size, Some(250));
        assert_eq!(c.svs_compression.as_deref(), Some("LeanVec4x8"));
        assert_eq!(c.svs_reduce, Some(384));
    }

    // Top-level algorithm still wins over a nested one, and a plain HNSW config
    // (no svs block) leaves the SVS fields None + derives nothing.
    #[test]
    fn hnsw_config_leaves_svs_fields_unset() {
        use super::RedisEngine;
        let cfg_json = serde_json::json!({
            "name": "redis-hnsw",
            "engine": "redis",
            "algorithm": "hnsw",
            "collection_params": { "hnsw_config": { "M": 32, "EF_CONSTRUCTION": 200 } }
        });
        let ec: crate::config::EngineConfig = serde_json::from_value(cfg_json).unwrap();
        let c = RedisEngine::new(&ec, "localhost").unwrap().config;
        assert_eq!(c.algorithm, "hnsw");
        assert_eq!(c.m, 32);
        assert_eq!(c.svs_graph_max_degree, None);
        assert_eq!(c.svs_compression, None);
        assert_eq!(c.svs_reduce, None);
    }

    #[test]
    fn encodes_int8_one_byte_per_element_with_rounding() {
        // Values arrive as f32 (e.g. pre-quantized cohere int8 embeddings).
        let v = vec![0.0f32, 1.4, -1.6, 127.0, -128.0];
        let bytes = encode_vector("INT8", &v);
        assert_eq!(bytes.len(), 5); // 1 byte per element
        let decoded: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(decoded, vec![0, 1, -2, 127, -128]);
    }

    #[test]
    fn int8_saturates_out_of_range() {
        let bytes = encode_vector("INT8", &[200.0, -200.0]);
        let decoded: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(decoded, vec![127, -128]);
    }

    #[test]
    fn encodes_uint8_clamped_to_0_255() {
        let bytes = encode_vector("UINT8", &[0.0, 255.0, 300.0, -5.0, 12.6]);
        assert_eq!(bytes, vec![0, 255, 255, 0, 13]);
    }

    // ── Timed-window hoisting fidelity ─────────────────────────────────────
    // The perf change moves encode_vector + query-string construction OUT of
    // the per-query timed window (precomputed once before the parallel region).
    // These tests prove the precomputed values are byte-for-byte identical to
    // what the old in-window code produced, so recall/precision cannot change.

    #[test]
    fn encode_vector_pins_exact_bytes_fp32_and_quantized() {
        // Pin the wire bytes against independently-computed expected vectors so a
        // future change to encode_vector (the hoisted-out client work) that alters
        // what actually reaches the server is caught. Determinism alone is not
        // enough — these assert the CORRECT quantized bytes, not just stability.
        let v = [1.0f32, -1.0, 0.5];

        // FLOAT32: little-endian f32 per element.
        let mut fp32_expected = Vec::new();
        fp32_expected.extend_from_slice(&1.0f32.to_le_bytes());
        fp32_expected.extend_from_slice(&(-1.0f32).to_le_bytes());
        fp32_expected.extend_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(encode_vector("FLOAT32", &v), fp32_expected);

        // INT8: f32::round (half AWAY from zero) then clamp to [-128,127], one
        // signed byte each. 1.0→1, -1.0→-1 (0xFF), 0.5→1. Hard-coded.
        assert_eq!(encode_vector("INT8", &v), vec![0x01u8, 0xFF, 0x01]);

        // UINT8: round then clamp to [0,255], one byte each. 1.0→1, -1.0→0, 0.5→1.
        assert_eq!(encode_vector("UINT8", &v), vec![0x01u8, 0x00, 0x01]);

        // FLOAT16 (IEEE half): 1.0→0x3C00, -1.0→0xBC00, 0.5→0x3800, LE bytes.
        assert_eq!(
            encode_vector("FLOAT16", &v),
            vec![0x00, 0x3C, 0x00, 0xBC, 0x00, 0x38]
        );

        // BFLOAT16: 1.0→0x3F80, -1.0→0xBF80, 0.5→0x3F00, LE bytes.
        assert_eq!(
            encode_vector("BFLOAT16", &v),
            vec![0x80, 0x3F, 0x80, 0xBF, 0x00, 0x3F]
        );

        // FLOAT64: little-endian f64 per element.
        let mut fp64_expected = Vec::new();
        fp64_expected.extend_from_slice(&1.0f64.to_le_bytes());
        fp64_expected.extend_from_slice(&(-1.0f64).to_le_bytes());
        fp64_expected.extend_from_slice(&0.5f64.to_le_bytes());
        assert_eq!(encode_vector("FLOAT64", &v), fp64_expected);
    }

    #[test]
    fn precomputed_blobs_match_per_query_encode() {
        // Emulate the batch precompute (`queries.iter().map(encode_vector)`) and
        // assert each slot equals the old per-query encode call, for FP32 and a
        // quantized type.
        let queries = [
            vec![1.0f32, -2.5, 3.5],
            vec![0.0f32, 127.4, -128.9],
            vec![10.0f32, -10.0, 0.5],
        ];
        for data_type in ["FLOAT32", "INT8"] {
            let precomputed: Vec<Vec<u8>> = queries
                .iter()
                .map(|q| encode_vector(data_type, q))
                .collect();
            for (i, q) in queries.iter().enumerate() {
                assert_eq!(
                    precomputed[i],
                    encode_vector(data_type, q),
                    "precomputed blob differs from per-query encode ({data_type}, q{i})"
                );
            }
        }
    }

    #[test]
    fn build_knn_query_str_matches_legacy_format() {
        use super::build_knn_query_str;
        // No filter, HNSW → EF_RUNTIME present, prefilter "*".
        assert_eq!(
            build_knn_query_str("hnsw", "", None),
            "*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );
        // ADHOC_BF hybrid policy drops EF_RUNTIME and appends the policy suffix.
        assert_eq!(
            build_knn_query_str("hnsw", "ADHOC_BF", None),
            "*=>[KNN $K @vector $vec_param  AS vector_score]=>{$HYBRID_POLICY: ADHOC_BF }"
        );
    }

    // SVS-VAMANA uses SEARCH_WINDOW_SIZE (its EF_RUNTIME analogue), still bound as
    // `$EF`. This MUST stay in lock-step with build_ft_search_cmd's `binds_ef`:
    // the query references `$EF` here, so the search cmd must bind an EF param.
    #[test]
    fn build_knn_query_str_svs_emits_search_window_size() {
        use super::{build_knn_query_str, is_svs};
        assert!(is_svs("svs-vamana") && is_svs("SVS-VAMANA") && is_svs("svs"));
        assert!(!is_svs("hnsw") && !is_svs("flat"));
        // SVS → SEARCH_WINDOW_SIZE $EF (not EF_RUNTIME), prefilter "*".
        assert_eq!(
            build_knn_query_str("svs-vamana", "", None),
            "*=>[KNN $K @vector $vec_param SEARCH_WINDOW_SIZE $EF AS vector_score]"
        );
        // ADHOC_BF is brute-force → no runtime window for SVS either.
        assert_eq!(
            build_knn_query_str("svs-vamana", "ADHOC_BF", None),
            "*=>[KNN $K @vector $vec_param  AS vector_score]=>{$HYBRID_POLICY: ADHOC_BF }"
        );
        // A non-HNSW/non-SVS algorithm (e.g. FLAT) has no runtime window param.
        assert_eq!(
            build_knn_query_str("flat", "", None),
            "*=>[KNN $K @vector $vec_param  AS vector_score]"
        );
    }

    #[test]
    fn build_knn_query_str_filtered_matches_legacy_format() {
        use super::build_knn_query_str;
        // The whole point of hoisting query_str is that it varies per query ONLY
        // through the filter prefilter. Pin the FILTERED path character-for-
        // character against the legacy inline `format!` so the per-query-varying
        // branch cannot silently diverge. Legacy (master) form for a non-empty
        // prefilter, HNSW, empty hybrid_policy:
        //   "{prefilter}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        let params: HashMap<String, FilterParamValue> =
            [("brand_0".to_string(), FilterParamValue::Str("apple".into()))]
                .into_iter()
                .collect();
        let filter: ParsedFilter = ("@brand:{apple}".to_string(), params);
        assert_eq!(
            build_knn_query_str("hnsw", "", Some(&filter)),
            "@brand:{apple}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );

        // Filtered + ADHOC_BF: prefilter kept, EF_RUNTIME dropped, policy suffix.
        assert_eq!(
            build_knn_query_str("hnsw", "ADHOC_BF", Some(&filter)),
            "@brand:{apple}=>[KNN $K @vector $vec_param  AS vector_score]=>{$HYBRID_POLICY: ADHOC_BF }"
        );
    }

    // ── redis::Value (RESP FT.SEARCH) response parsing ─────────────────────
    // Guards the manual RESP-array parsing against a redis-crate / RESP2-vs-RESP3
    // change (the surface most exposed by the redis-rs 0.27 → 1.x upgrade).
    use super::{parse_ft_search_response, redis_value_to_json};
    use redis::Value;
    use vector_db_benchmark::parsers::extract_vector_score;

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
        // RESP3 FT.SEARCH shape: a map whose `results` is an array of per-doc
        // maps with `id` + `extra_attributes` (carrying vector_score).
        let doc = |id: &str, score: &str| {
            Value::Map(vec![
                (bulk("id"), bulk(id)),
                (
                    bulk("extra_attributes"),
                    Value::Map(vec![(bulk("vector_score"), bulk(score))]),
                ),
                (bulk("values"), Value::Array(vec![])),
            ])
        };
        let resp = Value::Map(vec![
            (bulk("attributes"), Value::Array(vec![])),
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
        // RESP3 map with no `results` key → no hits.
        assert_eq!(
            parse_ft_search_response(&Value::Map(vec![(bulk("total_results"), Value::Int(0))]))
                .unwrap(),
            vec![]
        );
        let resp = Value::Array(vec![Value::Int(1), Value::Nil, Value::Nil]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(0, 0.0)]);
    }

    #[test]
    fn extract_vector_score_finds_field_or_defaults_zero() {
        let fields = vec![bulk("vector_score"), bulk("0.75")];
        assert!((extract_vector_score(&fields) - 0.75).abs() < 1e-9);
        assert_eq!(extract_vector_score(&[bulk("other"), bulk("x")]), 0.0);
    }

    #[test]
    fn redis_value_to_json_covers_scalars_and_fallthrough() {
        assert_eq!(redis_value_to_json(&Value::Int(5)), serde_json::json!(5));
        assert_eq!(redis_value_to_json(&bulk("hi")), serde_json::json!("hi"));
        assert_eq!(
            redis_value_to_json(&Value::Array(vec![Value::Int(1), bulk("a")])),
            serde_json::json!([1, "a"])
        );
        // Non-exhaustive/other variant → non-empty JSON string, never dropped.
        let okay = redis_value_to_json(&Value::Okay);
        assert!(okay.as_str().map(|s| !s.is_empty()).unwrap_or(false));
    }

    #[test]
    fn redis_value_to_json_covers_remaining_scalar_arms() {
        // Arms not exercised by the scalar/fallthrough test above.
        assert_eq!(redis_value_to_json(&Value::Nil), serde_json::Value::Null);
        assert_eq!(
            redis_value_to_json(&Value::Double(1.5)),
            serde_json::json!(1.5)
        );
        assert_eq!(
            redis_value_to_json(&Value::Boolean(true)),
            serde_json::json!(true)
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
        // Map with a string key → JSON object keyed by that string.
        assert_eq!(
            redis_value_to_json(&Value::Map(vec![(bulk("k"), Value::Int(9))])),
            serde_json::json!({"k": 9})
        );
    }

    #[test]
    fn value_as_i64_reads_simplestring_id() {
        use vector_db_benchmark::parsers::value_as_i64;
        // Redis-8 RESP2 can return the doc id as a SimpleString (e.g. "7").
        assert_eq!(value_as_i64(&Value::SimpleString("7".into())), 7);
        // Bulk/Int arms already exercised via parse_ft_search_response; pin them too.
        assert_eq!(value_as_i64(&bulk("42")), 42);
        assert_eq!(value_as_i64(&Value::Int(5)), 5);
        // Unparseable / unsupported → 0.
        assert_eq!(value_as_i64(&Value::SimpleString("x".into())), 0);
        assert_eq!(value_as_i64(&Value::Nil), 0);
    }

    #[test]
    fn parse_ft_search_resp2_drops_trailing_id_without_fields() {
        // Odd trailing element: [count, id] with no field block → the dangling id
        // is dropped (no doc emitted), it is NOT surfaced with a zero score.
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
    // These primed maps must be identical whether upload runs (configure())
    // or not (--skip-upload, primed at the start of search()). Cover the pure
    // priming fn directly so both call sites stay correct.
    #[test]
    fn prime_datetime_fields_selects_only_datetime_schema_fields() {
        use super::prime_datetime_fields;
        let schema = serde_json::json!({
            "created_at": "datetime",
            "updated_at": "datetime",
            "price": "int",
            "brand": "keyword",
            "title": "text",
        });
        let got = prime_datetime_fields(Some(&schema));
        let mut expected = std::collections::HashSet::new();
        expected.insert("created_at".to_string());
        expected.insert("updated_at".to_string());
        assert_eq!(got, expected);
    }

    #[test]
    fn prime_datetime_fields_empty_for_none_or_no_datetime() {
        use super::prime_datetime_fields;
        assert!(prime_datetime_fields(None).is_empty());
        let schema = serde_json::json!({ "price": "int", "brand": "keyword" });
        assert!(prime_datetime_fields(Some(&schema)).is_empty());
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

    #[test]
    fn or_only_emits_pipe_joined_group() {
        let cond = serde_json::json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        let (q, _p) = parse_conditions(&cond).unwrap();
        // OR clauses are pipe-joined inside a single parenthesized group; `must`
        // (AND) is absent.
        assert_eq!(q, "(@a:{$a_0} | @b:{$b_1})", "q={}", q);
    }

    #[test]
    fn and_plus_or_keeps_both_groups() {
        let cond = serde_json::json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}}],
        });
        let (q, _p) = parse_conditions(&cond).unwrap();
        // AND group (space-joined) then OR group (pipe-joined), space-separated.
        assert_eq!(q, "(@a:{$a_0}) (@b:{$b_1})", "q={}", q);
    }

    // ── Range operators ────────────────────────────────────────────────────

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
            "@n:[-inf ($n_0_lt]"
        );
    }

    #[test]
    fn range_lte_is_inclusive() {
        assert_eq!(
            range_q(serde_json::json!({"lte":5})).unwrap(),
            "@n:[-inf $n_0_lte]"
        );
    }

    #[test]
    fn range_gt_is_exclusive() {
        assert_eq!(
            range_q(serde_json::json!({"gt":5})).unwrap(),
            "@n:[($n_0_gt +inf]"
        );
    }

    #[test]
    fn range_gte_is_inclusive() {
        assert_eq!(
            range_q(serde_json::json!({"gte":5})).unwrap(),
            "@n:[$n_0_gte +inf]"
        );
    }

    #[test]
    fn range_two_sided_gte_lt() {
        // Bounds are emitted in the fixed order lt, gt, lte, gte (space-joined).
        assert_eq!(
            range_q(serde_json::json!({"gte":10,"lt":20})).unwrap(),
            "@n:[-inf ($n_0_lt] @n:[$n_0_gte +inf]"
        );
    }

    #[test]
    fn range_unknown_op_is_skipped() {
        // No recognized bound → no clause → whole filter is None.
        assert!(range_q(serde_json::json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_is_skipped() {
        // A null bound never parses into a param, so no dangling clause is emitted.
        assert!(range_q(serde_json::json!({"gte":serde_json::Value::Null})).is_none());
    }

    // ── Geo filter ─────────────────────────────────────────────────────────

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
        assert!(matches!(
            params.get("loc_0_lon"),
            Some(FilterParamValue::Float(_))
        ));
        assert!(matches!(
            params.get("loc_0_lat"),
            Some(FilterParamValue::Float(_))
        ));
    }

    #[test]
    fn geo_missing_radius_is_none() {
        // RediSearch geo has NO default radius: a missing radius drops the clause.
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
        assert_eq!(map_distance_metric("COSINE"), "COSINE"); // case-insensitive
                                                             // Unknown → default COSINE (never silently wrong metric type).
        assert_eq!(map_distance_metric("nope"), "COSINE");
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    fn exact_q(criteria: serde_json::Value) -> Option<(String, HashMap<String, FilterParamValue>)> {
        let mut counter = 0;
        build_exact_match_filter("n", &criteria, &mut counter)
    }

    #[test]
    fn exact_match_int_emits_numeric_point() {
        let (q, params) = exact_q(serde_json::json!({"value":5})).unwrap();
        assert_eq!(q, "@n:[$n_0 $n_0]", "q={}", q);
        assert!(matches!(params.get("n_0"), Some(FilterParamValue::Int(5))));
    }

    #[test]
    fn exact_match_float_emits_numeric_point() {
        let (q, params) = exact_q(serde_json::json!({"value":1.5})).unwrap();
        assert_eq!(q, "@n:[$n_0 $n_0]", "q={}", q);
        assert!(
            matches!(params.get("n_0"), Some(FilterParamValue::Float(f)) if (*f - 1.5).abs() < 1e-9)
        );
    }

    #[test]
    fn exact_match_array_value_is_none() {
        // A non-scalar (array) value matches no scalar arm → dropped → None.
        assert!(exact_q(serde_json::json!({"value":[1,2]})).is_none());
    }
}
