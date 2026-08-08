//! Dragonfly engine implementation (Dragonfly Search — Beta).
//!
//! Dragonfly is a Redis-wire-compatible datastore. Dragonfly Search (v1.13+)
//! implements a RediSearch-compatible `FT.*` subset — `FT.CREATE`, `FT.SEARCH`,
//! `FT.INFO`, `FT.DROPINDEX` — with `VECTOR` fields (FLAT / HNSW) and the KNN
//! query syntax `*=>[KNN k @field $blob AS score]`. This engine speaks that
//! subset over the `redis` crate (same RESP protocol).
//!
//! # Scope: vector KNN + metadata filtering
//!
//! Dragonfly Search supports the RediSearch TAG / NUMERIC / TEXT / GEO field
//! types and hybrid filtered KNN (`(prefilter)=>[KNN...]`) — verified live
//! against `dragonfly:df-v1.38.1`. This engine therefore indexes the dataset's
//! metadata schema and applies per-query filter `conditions` exactly like
//! redis.rs / valkey.rs (reusing their RediSearch filter builder): keyword / int
//! / float / bool / datetime / uuid datatypes, `match` / `match_any` / `range`,
//! and AND / OR / nested boolean. A query with no conditions still runs the `*`
//! (match-all) prefilter. GEO is the one unsupported filter type — Dragonfly's
//! geo-query parser rejects the `$param` placeholders the shared builder emits
//! (verified live), so geo fields are not indexed (like Chroma/Milvus). Mixed
//! (search+update) workload and quantization also remain out of scope.
//!
//! # Vector data type: FLOAT32 only
//!
//! Dragonfly Search supports ONLY the `float32` vector type — no
//! INT8/UINT8/FP16/BF16/FP64. Vectors are therefore always encoded as FLOAT32
//! little-endian bytes.
//!
//! # EF_RUNTIME
//!
//! Dragonfly Search **accepts** the per-query `EF_RUNTIME` HNSW attribute
//! (verified live against `dragonfly:df-v1.38.1`: a KNN query with `EF_RUNTIME
//! $EF` returns results, and a non-numeric `$EF` value is rejected with a query
//! syntax error — proving the attribute is actually parsed, not ignored). It is
//! kept, matching redis.rs / valkey.rs, so the search sweep's `ef` values take
//! effect instead of collapsing to the index default.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use super::geo;
use super::redis::ParsedFilter;
use super::redis_utils;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::index_naming::{derive_index_name, derive_key_prefix};
use crate::engine::{CorpusCount, Engine, SearchResults, UploadStats};
use crate::metrics::compute_metrics;
use vector_db_benchmark::parsers::{datetime_to_epoch_secs, doc_key_to_id, doc_key_to_id_opt};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::MetadataItem;
use vector_db_benchmark::start_gate::WorkerPool;

/// Dragonfly engine configuration.
#[derive(Clone)]
pub struct DragonflyEngineConfig {
    pub m: i64,
    pub ef_construction: i64,
    /// Always `FLOAT32` — Dragonfly Search supports no other vector type.
    pub data_type: String,
    pub algorithm: String,
    pub batch_size: usize,
    pub parallel: usize,
    /// Per-config index name (`"<base>:<config>"`, issue #151-4) so a sweep's
    /// configs address disjoint indexes on one server. Resolved once in `new()`.
    pub index_name: String,
    /// Per-config key prefix (`"<config>:"`, issue #151-4). Each config owns a
    /// disjoint keyspace; teardown is a prefix-scoped SCAN+UNLINK (no DD flag).
    pub key_prefix: String,
}

pub struct DragonflyEngine {
    name: String,
    host: String,
    port: u16,
    config: DragonflyEngineConfig,
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

impl DragonflyEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("DRAGONFLY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6385);

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

        // Dragonfly Search only supports float32; ignore any configured override.
        let data_type = "FLOAT32".to_string();

        // Upload concurrency/batch come from the engine config, but each can be
        // overridden at runtime via env (taking precedence over the config). The
        // default 100-thread upload burst resets connections on Dragonfly Cloud
        // for larger-dimensional datasets, so managed-cloud runs set
        // DRAGONFLY_UPLOAD_PARALLEL=16 without having to edit the shared config;
        // search throughput is unaffected by upload concurrency.
        let parallel = std::env::var("DRAGONFLY_UPLOAD_PARALLEL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| {
                engine_config
                    .upload_params
                    .as_ref()
                    .and_then(|p| p.get("parallel"))
                    .and_then(|v| v.as_i64())
                    .map(|v| v as usize)
            })
            .unwrap_or(100);

        let batch_size = std::env::var("DRAGONFLY_UPLOAD_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| {
                engine_config
                    .upload_params
                    .as_ref()
                    .and_then(|p| p.get("batch_size"))
                    .and_then(|v| v.as_i64())
                    .map(|v| v as usize)
            })
            .unwrap_or(64);

        Ok(Self {
            name: engine_config.name.clone(),
            host: host.to_string(),
            port,
            config: DragonflyEngineConfig {
                m,
                ef_construction,
                data_type,
                algorithm,
                batch_size,
                parallel,
                index_name: derive_index_name("DRAGONFLY_INDEX_NAME", "idx", &engine_config.name),
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
        let auth = std::env::var("DRAGONFLY_AUTH").ok();
        let user = std::env::var("DRAGONFLY_USER").ok();

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
            dragonfly_url_suffix()
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

        // Drop this config's index + keys ONLY (Dragonfly Search has no DD flag, so
        // a prefix-scoped SCAN+UNLINK replaces the old keyspace-wide FLUSHALL, which
        // under #151-4 coexistence would wipe sibling configs' data).
        redis_utils::drop_index_and_keys(conn, &self.config.index_name, &self.config.key_prefix);

        let distance_metric = map_distance_metric(distance);

        // Build FT.CREATE: the VECTOR field `vector` plus any filterable metadata
        // fields declared in the dataset schema. Dragonfly Search (df-v1.13+)
        // supports the RediSearch TAG/NUMERIC/TEXT/GEO field types + hybrid
        // filtered KNN (`(prefilter)=>[KNN...]`), verified live, so it is no longer
        // KNN-only.
        let mut cmd = redis::cmd("FT.CREATE");
        cmd.arg(&self.config.index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg(&self.config.key_prefix);

        cmd.arg("SCHEMA");

        // num_attrs = TYPE+DIM+DISTANCE_METRIC (6) + M (2) + EF_CONSTRUCTION (2).
        let num_attrs = 6 + 2 + 2;
        cmd.arg("vector")
            .arg("VECTOR")
            .arg(self.config.algorithm.to_uppercase())
            .arg(num_attrs);
        cmd.arg("TYPE").arg(&self.config.data_type);
        cmd.arg("DIM").arg(vector_size);
        cmd.arg("DISTANCE_METRIC").arg(distance_metric);
        cmd.arg("M").arg(self.config.m);
        cmd.arg("EF_CONSTRUCTION").arg(self.config.ef_construction);

        // Filterable metadata fields (mirrors redis.rs): keyword/uuid/bool exact
        // strings -> TAG (SEPARATOR ; so multi-valued `labels` match per element);
        // int/float/datetime (stored as epoch) -> NUMERIC; full-text -> TEXT;
        // geo point -> nothing: see the NOTE below, geo is NOT declared.
        if let Some(schema) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
            for (field_name, field_type) in schema {
                match field_type.as_str().unwrap_or("") {
                    "keyword" | "uuid" | "bool" => {
                        cmd.arg(field_name).arg("TAG").arg("SEPARATOR").arg(";");
                    }
                    "int" | "float" | "datetime" => {
                        cmd.arg(field_name).arg("NUMERIC");
                    }
                    "text" => {
                        cmd.arg(field_name).arg("TEXT");
                    }
                    // NOTE: geo is intentionally NOT declared. Dragonfly Search
                    // accepts a GEO field but its geo-query parser rejects `$param`
                    // placeholders inside the `[lon lat radius unit]` bracket
                    // (verified live), and the shared RediSearch filter builder
                    // emits geo bounds as params — so geo filtering is unsupported
                    // here (like Chroma/Milvus). Other datatypes work.
                    _ => {}
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
        datetime_fields: &std::collections::HashSet<String>,
    ) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let pb = self.create_progress_bar(ids.len());

        for batch_start in (0..ids.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(ids.len());
            upload_batch_internal(
                &mut conn,
                &ids[batch_start..batch_end],
                &vectors[batch_start..batch_end],
                &metadata[batch_start..batch_end],
                datetime_fields,
                &self.config.key_prefix,
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
        datetime_fields: &std::collections::HashSet<String>,
    ) -> Result<(), String> {
        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.config.batch_size)
            .map(|start| (start, (start + self.config.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let num_threads = self.config.parallel.min(total_batches).max(1);
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let host = self.host.clone();
                let port = self.port;
                let key_prefix = self.config.key_prefix.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match DragonflyEngine::connect(&host, port) {
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
                            &ids[batch_start..batch_end],
                            &vectors[batch_start..batch_end],
                            &metadata[batch_start..batch_end],
                            datetime_fields,
                            &key_prefix,
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

    /// Wait until FT.INFO reports num_docs >= expected and indexing is done.
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
            // Default 1.0 (fully indexed) so an FT.INFO that omits the field does
            // not stall the wait; Dragonfly DOES expose percent_indexed.
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

            let mut handle_pair = |key: &str, val: &redis::Value| match key {
                "num_docs" => num_docs = extract_usize(val),
                "indexing" => indexing = indexing || extract_bool_nonzero(val),
                "backfill_in_progress" => indexing = indexing || extract_bool_nonzero(val),
                "percent_indexed" => percent_indexed = extract_f64(val),
                _ => {}
            };

            match &info {
                redis::Value::Array(arr) => {
                    for i in (0..arr.len()).step_by(2) {
                        let key_str = match &arr[i] {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if let Some(val) = arr.get(i + 1) {
                            handle_pair(&key_str, val);
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
                        handle_pair(&key_str, v);
                    }
                }
                _ => {
                    eprintln!("Unexpected FT.INFO response type: {:?}", info);
                }
            }

            // Require the HNSW graph to be fully built (percent_indexed >= 1.0),
            // not just the doc count, so the search sweep never runs against a
            // partially-backfilled graph (which would depress recall).
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
                    "Warning: indexing timeout after {}s (num_docs={}/{}, indexing={}, percent_indexed={:.2})",
                    max_wait, num_docs, expected, indexing, percent_indexed
                );
                return Ok(());
            }

            if start.elapsed().as_secs().is_multiple_of(10) && start.elapsed().as_secs() > 0 {
                println!(
                    "Waiting for indexing: {} docs, indexing={}, percent_indexed={:.2} ({:.0}s)",
                    num_docs,
                    indexing,
                    percent_indexed,
                    start.elapsed().as_secs_f64()
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
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
}

/// Optional connection-URL suffix so Dragonfly can be benchmarked over RESP3
/// (`DRAGONFLY_PROTOCOL=resp3`). Defaults to RESP2 (empty suffix). The FT.SEARCH
/// response parser handles both shapes, so recall is identical either way.
fn dragonfly_url_suffix() -> &'static str {
    if std::env::var("DRAGONFLY_PROTOCOL")
        .map(|v| v.eq_ignore_ascii_case("resp3"))
        .unwrap_or(false)
    {
        "?protocol=resp3"
    } else {
        ""
    }
}

/// Encode a vector to the FLOAT32 little-endian blob Dragonfly Search expects.
/// Dragonfly Search supports ONLY float32, so this is the single encoding used
/// for both upload and query vectors.
fn encode_query_vector(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Upload one batch of `HSET {id} vector {float32_le_bytes}` via a pipeline.
fn upload_batch_internal(
    conn: &mut Connection,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    datetime_fields: &std::collections::HashSet<String>,
    key_prefix: &str,
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let key = format!("{}{}", key_prefix, ids[i]);
        let vec_bytes = encode_query_vector(&vectors[i]);
        let mut hset_cmd = redis::cmd("HSET");
        hset_cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);

        // Metadata fields for filtering (mirrors redis.rs): bools stay the reader's
        // "true"/"false" string (TAG match); datetime strings become epoch seconds
        // (NUMERIC range); numbers/labels/geo map as redis does.
        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                match v {
                    MetadataValue::String(s) => {
                        let stored = if datetime_fields.contains(k) {
                            match datetime_to_epoch_secs(s) {
                                Some(e) => (e as i64).to_string(),
                                None => s.clone(),
                            }
                        } else {
                            s.clone()
                        };
                        hset_cmd.arg(k.as_str()).arg(stored);
                    }
                    MetadataValue::Int(n) => {
                        hset_cmd.arg(k.as_str()).arg(n.to_string());
                    }
                    MetadataValue::Float(f) => {
                        hset_cmd.arg(k.as_str()).arg(f.to_string());
                    }
                    MetadataValue::Labels(labels) => {
                        hset_cmd.arg(k.as_str()).arg(labels.join(";"));
                    }
                    MetadataValue::Geo { lon, lat } => {
                        hset_cmd.arg(k.as_str()).arg(format!("{},{}", lon, lat));
                    }
                }
            }
        }
        pipe.add_command(hset_cmd);
    }

    pipe.query::<()>(conn).map_err(|e| e.to_string())?;
    Ok(())
}

/// Map a dataset distance name to the Dragonfly Search `DISTANCE_METRIC` value.
/// Unknown metrics default to `COSINE`. A typo here (e.g. IP→L2) would silently
/// invert ranking, so it is unit-tested.
fn map_distance_metric(distance: &str) -> &'static str {
    match distance.to_lowercase().as_str() {
        "cosine" | "angular" => "COSINE",
        "euclidean" | "l2" => "L2",
        "dot" | "ip" => "IP",
        _ => "COSINE",
    }
}

/// Whether `EF_RUNTIME` should be emitted for the given algorithm.
///
/// `EF_RUNTIME` is an HNSW-only per-query attribute — a FLAT index rejects it
/// with a query syntax error. Gating it (query string, the `EF` PARAM, and the
/// PARAMS count) on HNSW keeps a `"algorithm":"flat"` config usable, mirroring
/// redis.rs.
fn uses_ef_runtime(algorithm: &str) -> bool {
    algorithm.eq_ignore_ascii_case("hnsw")
}

/// Build the FT.SEARCH KNN query string (unfiltered `*` prefilter).
///
/// Pure client-side string formatting, kept OUT of the per-query timed window
/// (precomputed once before the parallel region). `EF_RUNTIME $EF` is emitted
/// only for an HNSW index (verified live) — a per-query attribute FLAT rejects;
/// without it every `ef` in the search sweep runs at the index default. The
/// query vector is bound as `$vec_param`, so this string is identical across all
/// queries.
fn build_knn_query_str(algorithm: &str, prefilter: &str) -> String {
    if uses_ef_runtime(algorithm) {
        format!("{prefilter}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
    } else {
        format!("{prefilter}=>[KNN $K @vector $vec_param AS vector_score]")
    }
}

/// Execute a Dragonfly FT.SEARCH KNN query, return (id, score) pairs.
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
    algorithm: &str,
    query_timeout: i64,
    filter: Option<&ParsedFilter>,
) -> Result<Vec<(i64, f64)>, String> {
    use super::redis::FilterParamValue;
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
        .arg(2) // Dragonfly Search uses DIALECT 2
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Params: vec_param(2) + K(2), plus EF(2) only for HNSW (EF_RUNTIME is
    // HNSW-only; binding it on a FLAT index would be a syntax error), plus 2 per
    // filter param (the prefilter references `$name` placeholders).
    let ef_runtime = uses_ef_runtime(algorithm);
    let filter_params = filter.map(|(_, p)| p.len()).unwrap_or(0);
    let n = 4 + if ef_runtime { 2 } else { 0 } + filter_params * 2;
    cmd.arg("PARAMS").arg(n);
    cmd.arg("vec_param").arg(vec_bytes);
    cmd.arg("K").arg(top.to_string());
    if ef_runtime {
        cmd.arg("EF").arg(ef.to_string());
    }
    if let Some((_, params)) = filter {
        for (name, value) in params {
            cmd.arg(name);
            match value {
                FilterParamValue::Str(s) => cmd.arg(s),
                FilterParamValue::Int(i) => cmd.arg(i.to_string()),
                FilterParamValue::Float(f) => cmd.arg(f.to_string()),
            };
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
/// - RESP3: a map `{ results: [ { id, extra_attributes: { vector_score, .. } } ] }`
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

/// Extract the `vector_score` field value from a RESP2 field-values array.
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

/// Convert a redis::Value to serde_json::Value for FT.INFO serialization.
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

// ── Engine trait implementation ──────────────────────────────────────────

/// Establish the commandstats baseline if configure() did not (issue #238).
impl DragonflyEngine {
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

/// Dragonfly's filter builder: RediSearch's, minus geo.
///
/// `configure()` deliberately does NOT declare a GEO field (see the note there:
/// Dragonfly Search accepts a GEO field but its geo-query parser rejects the
/// `$param` placeholders the shared RediSearch builder emits). Calling
/// `redis::parse_conditions` directly therefore produced a real-looking
/// `@field:[$lon $lat $r m]` clause against a field that is not in the schema —
/// a filter that cannot match what it claims to.
///
/// That was invisible in two places at once: nothing in the engine refused it,
/// and `engine/filter_guard.rs`'s dragonfly column reuses this builder, so the
/// matrix scored dragonfly FILTERED on all three shipped geo shapes — an
/// asserted green cell for a field the engine never creates. Refusing here makes
/// `query_filter::resolve` turn a geo dataset into the #219 hard error, which is
/// what every other engine without geo already does.
///
/// Scoped deliberately: only geo is removed. Every other condition type is
/// RediSearch's builder verbatim, which is the whole point of sharing it.
pub(crate) fn parse_conditions(conditions: &serde_json::Value) -> Option<ParsedFilter> {
    if geo::conditions_mention_geo(conditions) {
        return None;
    }
    super::redis::parse_conditions(conditions)
}

impl Engine for DragonflyEngine {
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

        println!(
            "Using algorithm {} with config {{'M': {}, 'EF_CONSTRUCTION': {}}}",
            self.config.algorithm, self.config.m, self.config.ef_construction
        );

        self.create_index(&mut conn, dataset)?;
        // NOTE: Dragonfly's INFO commandstats omits the `failed_calls=` field
        // that redis/valkey expose, so the check_commandstats guards in
        // upload()/search() are best-effort no-ops on Dragonfly — they cannot
        // observe server-side command failures. Primary error propagation is
        // still correct: every HSET/FT.SEARCH goes through `cmd.query`, which
        // surfaces an `Err` on failure. The baseline is still reset for parity
        // (and would activate automatically if Dragonfly ever adds failed_calls).
        self.commandstats_baseline = redis_utils::reset_commandstats(&mut conn)?;
        self.commandstats_primed = true;
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();

        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, metadata): (Vec<i64>, Vec<Vec<f32>>, Vec<Option<MetadataItem>>) =
            dataset.read_vectors(normalize)?;
        // `datetime` schema fields are stored as NUMERIC epoch seconds (like
        // redis/valkey) so range filters over them work.
        let datetime_fields: std::collections::HashSet<String> = dataset
            .config
            .schema
            .as_ref()
            .and_then(|s| s.as_object())
            .map(|o| {
                o.iter()
                    .filter(|(_, t)| t.as_str() == Some("datetime"))
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default();
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
            self.upload_sequential(&ids, &vectors, &metadata, &datetime_fields)?;
        } else {
            self.upload_parallel(&ids, &vectors, &metadata, &datetime_fields)?;
        }

        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Include the index-build wait in total_time for cross-engine
        // comparability (mirrors redis/valkey).
        let expected = vectors.len();
        let index_start = Instant::now();
        self.wait_for_indexing(expected)?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

        // Best-effort HSET failure guard. Inert on Dragonfly (its commandstats
        // has no failed_calls field — see configure()); real HSET errors already
        // propagate as `Err` from the pipelined `cmd.query` above.
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

        // Index-existence guard (#151-4): on the --skip-upload path a missing or
        // mismatched index would otherwise write a silent recall-0.0 result file.
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
        let query_timeout: i64 = std::env::var("DRAGONFLY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        // KNN-only: filter conditions are ignored (never built into the query).
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        // Per-query prefilters, reusing redis's RediSearch filter builder (same
        // FT.SEARCH syntax). Dragonfly Search supports hybrid filtered KNN, so a
        // query with `conditions` runs `(prefilter)=>[KNN...]` instead of `*`.
        let parsed_filters: Vec<QueryFilter<ParsedFilter>> =
            conditions.resolve_all("Dragonfly", parse_conditions)?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute client-side request construction BEFORE the timed region so
        // the per-query window wraps ONLY the RPC round-trip + reply parse
        // (matching redis.rs/valkey.rs). Encoding the FLOAT32 blob is client
        // work, not server latency. Query strings are per-query (each embeds its
        // own prefilter). Shared read-only across workers.
        let encoded_queries: Vec<Vec<u8>> =
            queries.iter().map(|q| encode_query_vector(q)).collect();
        let algorithm = self.config.algorithm.clone();
        let query_strs: Vec<String> = parsed_filters
            .iter()
            .map(|f| {
                let prefilter = f.as_ref().map(|(expr, _)| expr.as_str()).unwrap_or("*");
                build_knn_query_str(&algorithm, prefilter)
            })
            .collect();
        // Resolve the per-config index name once (not per query / per worker).
        let index_name = self.config.index_name.clone();

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

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "dragonfly-search", parallel);
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let neighbors = &neighbors;
                let encoded_queries = &encoded_queries;
                let query_strs = &query_strs;
                let parsed_filters = &parsed_filters;
                let algorithm = algorithm.as_str();
                let index_name = index_name.as_str();
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                pool.spawn(move |ticket| {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let mut conn = match DragonflyEngine::connect(&host, port) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("dragonfly-search worker setup failed: {e}"));
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this connection with ONE discarded query (index 0) so
                    // the cold first round-trip is not inside the measured window.
                    // Best effort: errors ignored and its sample is NOT recorded.
                    {
                        let prime_top = explicit_top.unwrap_or(10);
                        let _ = ft_search_knn(
                            &mut conn,
                            index_name,
                            &encoded_queries[0],
                            &query_strs[0],
                            prime_top,
                            ef,
                            algorithm,
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
                            algorithm,
                            query_timeout,
                            parsed_filters[idx].as_ref(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        match &results {
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

        // Best-effort FT.SEARCH failure guard. Inert on Dragonfly (no
        // failed_calls in commandstats — see configure()); a failing FT.SEARCH
        // already surfaces as `Err` from `ft_search_knn` and is logged +
        // excluded from the stats (num_to_run minus successes).
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

    fn delete(&mut self) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        // Dragonfly Search has no DD flag: drop this config's index + its keys via
        // a prefix-scoped SCAN+UNLINK (not a keyspace-wide FLUSHALL, which under
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
    use super::*;
    use redis::Value;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    #[test]
    fn encode_query_vector_is_float32_le_bytes() {
        let v = vec![1.0f32, -2.5, 3.25];
        let expected: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(encode_query_vector(&v), expected);
        // 3 f32 => 12 bytes.
        assert_eq!(encode_query_vector(&v).len(), 12);
    }

    #[test]
    fn encode_query_vector_pins_exact_bytes() {
        // 1.0f32 = 0x3F800000 little-endian => 00 00 80 3F.
        assert_eq!(encode_query_vector(&[1.0]), vec![0x00, 0x00, 0x80, 0x3F]);
    }

    #[test]
    fn map_distance_metric_covers_all_aliases() {
        assert_eq!(map_distance_metric("cosine"), "COSINE");
        assert_eq!(map_distance_metric("angular"), "COSINE");
        assert_eq!(map_distance_metric("euclidean"), "L2");
        assert_eq!(map_distance_metric("l2"), "L2");
        assert_eq!(map_distance_metric("dot"), "IP");
        assert_eq!(map_distance_metric("ip"), "IP");
        assert_eq!(map_distance_metric("L2"), "L2"); // case-insensitive
        assert_eq!(map_distance_metric("unknown"), "COSINE"); // default
    }

    #[test]
    fn build_knn_query_str_is_unfiltered_with_ef_runtime() {
        // HNSW emits EF_RUNTIME (per-query ef sweep); FLAT must NOT (it would be
        // a syntax error on a FLAT index).
        assert_eq!(
            build_knn_query_str("hnsw", "*"),
            "*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );
        assert_eq!(
            build_knn_query_str("HNSW", "*"),
            "*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );
        assert_eq!(
            build_knn_query_str("flat", "*"),
            "*=>[KNN $K @vector $vec_param AS vector_score]"
        );
        assert!(uses_ef_runtime("hnsw") && !uses_ef_runtime("flat"));
    }

    #[test]
    fn parse_ft_search_response_resp2_reads_id_score_pairs() {
        // [count, key, [vector_score, val], key, [vector_score, val]]. A
        // per-config-prefixed key ("cfg:42", #151-4) resolves to its trailing id.
        let resp = Value::Array(vec![
            Value::Int(2),
            bulk("7"),
            Value::Array(vec![bulk("vector_score"), bulk("0.5")]),
            bulk("cfg:42"),
            Value::Array(vec![bulk("vector_score"), bulk("0.75")]),
        ]);
        let out = parse_ft_search_response(&resp).unwrap();
        assert_eq!(out, vec![(7, 0.5), (42, 0.75)]);
    }

    #[test]
    fn parse_ft_search_response_resp3_map_reads_results() {
        let doc = Value::Map(vec![
            (bulk("id"), bulk("9")),
            (
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.125"))]),
            ),
        ]);
        let resp = Value::Map(vec![(bulk("results"), Value::Array(vec![doc]))]);
        let out = parse_ft_search_response(&resp).unwrap();
        assert_eq!(out, vec![(9, 0.125)]);
    }

    #[test]
    fn parse_ft_search_response_empty_and_unknown_variants() {
        assert!(parse_ft_search_response(&Value::Nil).unwrap().is_empty());
        assert!(parse_ft_search_response(&Value::Int(0)).unwrap().is_empty());
    }

    #[test]
    fn parse_ft_search_resp2_drops_trailing_id_without_fields() {
        // Odd-length body: a trailing id with no field array is dropped.
        let resp = Value::Array(vec![Value::Int(1), bulk("5")]);
        let out = parse_ft_search_response(&resp).unwrap();
        assert!(out.is_empty(), "trailing id without fields must be dropped");
    }

    #[test]
    fn parse_ft_search_resp3_skips_doc_with_unparseable_id() {
        let doc = Value::Map(vec![
            (bulk("id"), bulk("not-a-number")),
            (
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.1"))]),
            ),
        ]);
        let resp = Value::Map(vec![(bulk("results"), Value::Array(vec![doc]))]);
        assert!(parse_ft_search_response(&resp).unwrap().is_empty());
    }

    #[test]
    fn extract_vector_score_finds_field_or_defaults_zero() {
        let fields = vec![bulk("vector_score"), bulk("0.25")];
        assert_eq!(extract_vector_score(&fields), 0.25);
        let none = vec![bulk("other"), bulk("1.0")];
        assert_eq!(extract_vector_score(&none), 0.0);
    }

    #[test]
    fn value_as_i64_reads_variants() {
        assert_eq!(value_as_i64(&bulk("13")), 13);
        assert_eq!(value_as_i64(&Value::Int(-4)), -4);
        assert_eq!(value_as_i64(&Value::SimpleString("8".into())), 8);
        assert_eq!(value_as_i64(&Value::Nil), 0);
    }

    #[test]
    fn parse_used_memory_reads_exact_prefix_only() {
        let info = "# Memory\r\nused_memory:1048576\r\nused_memory_rss:2097152\r\n";
        assert_eq!(parse_used_memory(info), 1048576);
        assert_eq!(parse_used_memory("no memory line here"), 0);
    }
}
