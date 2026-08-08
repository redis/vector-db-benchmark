//! VectorSets-rs engine implementation.
//!
//! Implements the Engine trait for Redis VectorSets (VADD/VSIM commands).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use super::redis_utils;
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UpdateSearchRatio, UploadStats};
use vector_db_benchmark::parsers::datetime_to_epoch_secs;
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};

/// VectorSets engine configuration
#[derive(Clone)]
pub struct VectorSetsConfig {
    pub quant: String,
    pub m: i64,
    pub ef_construction: i64,
    pub cas: bool,
    pub batch_size: usize,
    pub parallel: usize,
}

pub struct VectorSetsEngine {
    name: String,
    redis_url: String,
    config: VectorSetsConfig,
    search_params: Vec<SearchParams>,
    commandstats_baseline: Option<redis_utils::CommandStatsBaseline>,
}

impl VectorSetsEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let redis_url = crate::engine::build_redis_url(host);

        let hnsw_config = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("hnsw_config"))
            .and_then(|v| v.as_object());

        let quant = hnsw_config
            .and_then(|h| h.get("quant"))
            .and_then(|v| v.as_str())
            .unwrap_or("NOQUANT")
            .to_string();

        let m = hnsw_config
            .and_then(|h| h.get("M"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16);

        let ef_construction = hnsw_config
            .and_then(|h| h.get("EF_CONSTRUCTION"))
            .and_then(|v| v.as_i64())
            .unwrap_or(200);

        let cas = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("CAS"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(32) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(64) as usize;

        Ok(Self {
            name: engine_config.name.clone(),
            redis_url,
            config: VectorSetsConfig {
                quant,
                m,
                ef_construction,
                cas,
                batch_size,
                parallel,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            commandstats_baseline: None,
        })
    }

    fn get_connection(&self) -> Result<Connection, String> {
        let client = redis::Client::open(self.redis_url.as_str()).map_err(|e| e.to_string())?;
        client.get_connection().map_err(|e| e.to_string())
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
            vadd_batch(
                &mut conn,
                &self.config,
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
                        if let Err(e) = vadd_batch(
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

/// Execute a batch of VADD commands via pipeline.
/// VADD idx FP32 <vec_bytes> <id> <quant> M <M> EF <EF_CONSTRUCTION> [CAS] [SETATTR '<json>']
fn vadd_batch(
    conn: &mut Connection,
    config: &VectorSetsConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let vec_bytes: Vec<u8> = vectors[i].iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut cmd = redis::cmd("VADD");
        cmd.arg("idx")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(ids[i].to_string())
            .arg(&config.quant)
            .arg("M")
            .arg(config.m)
            .arg("EF")
            .arg(config.ef_construction);

        if config.cas {
            cmd.arg("CAS");
        }

        // Attach metadata as JSON attributes for FILTER support
        if let Some(json_str) = metadata[i].as_ref().and_then(metadata_to_attr_json) {
            cmd.arg("SETATTR").arg(json_str);
        }

        pipe.add_command(cmd);
    }

    pipe.query::<()>(conn)
        .map_err(|e| format!("VADD batch error: {}", e))?;
    Ok(())
}

/// Render one document's metadata as the JSON object attached by
/// `VADD … SETATTR`, or `None` when there is nothing to attach. Shared by the
/// upload batch and the mixed-workload update path so BOTH store the identical
/// representation (notably datetime → epoch seconds, see [`encode_string_attr`]).
fn metadata_to_attr_json(meta: &MetadataItem) -> Option<String> {
    if meta.fields.is_empty() {
        return None;
    }
    let mut map = serde_json::Map::new();
    for (k, v) in &meta.fields {
        let value = match v {
            MetadataValue::String(s) => encode_string_attr(s),
            MetadataValue::Int(n) => serde_json::Value::from(*n),
            MetadataValue::Float(f) => serde_json::json!(*f),
            MetadataValue::Labels(labels) => serde_json::Value::Array(
                labels
                    .iter()
                    .map(|l| serde_json::Value::String(l.clone()))
                    .collect(),
            ),
            MetadataValue::Geo { lon, lat } => serde_json::json!({"lon": lon, "lat": lat}),
        };
        map.insert(k.clone(), value);
    }
    Some(serde_json::Value::Object(map).to_string())
}

/// THE datetime representation decision for this engine, in ONE place.
///
/// VSIM `FILTER` has no date type and its comparison operators are NUMERIC, so a
/// datetime has to live as an epoch-seconds NUMBER. Both halves of the engine —
/// what `VADD … SETATTR` WRITES ([`encode_string_attr`]) and what the query
/// builders COMPARE AGAINST ([`number_literal`], [`build_match_clause`],
/// [`build_match_any_clause`]) — route through this single predicate, so they can
/// never disagree about whether a given string is a datetime.
///
/// It is deliberately **schema-free**. An earlier version gated only the storage
/// half on `dataset.config.schema` while the query half converted any ISO-looking
/// bound unconditionally; the two then disagreed for every datetime-carrying field
/// NOT declared `datetime` (declared `keyword`, or no schema at all) and produced
/// SILENT wrong recall: VSIM coerces a non-numeric string attribute to `0` in a
/// numeric comparison (verified live on redis 8.8), so a two-sided range matched
/// NOTHING while a one-sided `lt`/`lte` matched EVERYTHING — issue #220's exact
/// failure class, reached by another route. Deciding from the VALUE alone makes
/// the two halves agree by construction, with or without a schema.
///
/// Trade-off: a `keyword` field whose values happen to be ISO-8601 timestamps is
/// stored numerically too. Range and equality filters stay correct (both sides
/// convert), but the best-effort full-text form `"<text>" in .field` cannot match
/// such a field. That is strictly better than the silent recall corruption above.
fn epoch_literal(value: &str) -> Option<i64> {
    datetime_to_epoch_secs(value).map(|secs| secs as i64)
}

/// Render a string metadata field as the JSON attribute value stored by
/// `VADD … SETATTR`.
///
/// An ISO-8601 / RFC-3339 value is stored as an epoch-seconds NUMBER, exactly
/// what [`build_range_clause`] emits for an ISO bound — see [`epoch_literal`] for
/// why the decision is made from the value rather than from the schema. Storing
/// the ISO string instead would make every datetime range filter match nothing.
/// Mirrors `redis::encode_string_field` (and milvus/chroma, which likewise store
/// datetimes as epoch ints). Anything not parseable as a datetime (already an
/// epoch, a keyword, a bool token) is stored verbatim.
fn encode_string_attr(value: &str) -> serde_json::Value {
    match epoch_literal(value) {
        Some(epoch) => serde_json::Value::from(epoch),
        None => serde_json::Value::String(value.to_string()),
    }
}

/// Encode a query vector to the FP32 little-endian blob VSIM expects.
///
/// Kept as a standalone fn so the caller can precompute all query blobs BEFORE
/// the per-query timed window (client work, not server latency).
fn encode_query_vector(query_vector: &[f32]) -> Vec<u8> {
    query_vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Execute VSIM query and return (id, score) pairs.
/// VSIM idx FP32 <vec_bytes> WITHSCORES COUNT <top> EF <ef> [FILTER '<expr>' [FILTER-EF <n>]]
/// Response: alternating [id, score, id, score, ...]
/// Score conversion: 1.0 - score (VectorSets: 1=identical, 0=opposite)
///
/// `vec_bytes` is precomputed by the caller BEFORE the timed window; this
/// performs only the arg binding, the `cmd.query` RPC round-trip, and the reply
/// parse (all legitimate server latency).
fn vsim_search(
    conn: &mut Connection,
    vec_bytes: &[u8],
    top: usize,
    ef: i64,
    filter: Option<&str>,
    filter_ef: Option<i64>,
) -> Result<Vec<(i64, f64)>, String> {
    let mut cmd = redis::cmd("VSIM");
    cmd.arg("idx")
        .arg("FP32")
        .arg(vec_bytes)
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(top)
        .arg("EF")
        .arg(ef);

    if let Some(filter_expr) = filter {
        cmd.arg("FILTER").arg(filter_expr);
        // FILTER-EF controls how many candidate nodes the engine inspects
        // for filtered results. Default is COUNT * 100.
        // Docs say FILTER-EF 0 means "scan as many as needed", but
        // Redis 8.6.0 rejects 0 with "invalid FILTER-EF".
        // Workaround: use a very large value (10M) to approximate unlimited.
        // Configurable via search_params.filter_ef in experiment JSON.
        let fe = filter_ef.unwrap_or(10_000_000);
        cmd.arg("FILTER-EF").arg(fe);
    }

    let response: Vec<redis::Value> = cmd.query(conn).map_err(|e| format!("VSIM error: {}", e))?;
    Ok(parse_vsim_response(&response))
}

/// Parse a `VSIM … WITHSCORES` reply — an alternating `[id, score, id, score, …]`
/// array. IDs arrive as bulk strings (RESP2) or integers; scores as bulk strings
/// (RESP2) or doubles (RESP3). VectorSets returns similarity (1 = identical), so
/// each score is converted to a distance via `1.0 - score`. Unrecognized value
/// variants fall back to `0`/`0.0` rather than panicking.
fn parse_vsim_response(response: &[redis::Value]) -> Vec<(i64, f64)> {
    let mut results = Vec::new();
    let mut i = 0;
    while i + 1 < response.len() {
        let id = match &response[i] {
            redis::Value::BulkString(data) => {
                String::from_utf8_lossy(data).parse::<i64>().unwrap_or(0)
            }
            redis::Value::Int(n) => *n,
            _ => 0,
        };

        let score = match &response[i + 1] {
            redis::Value::BulkString(data) => {
                let s = String::from_utf8_lossy(data).parse::<f64>().unwrap_or(0.0);
                1.0 - s // VectorSets: 1=identical, convert to distance
            }
            redis::Value::Double(f) => 1.0 - f,
            _ => 0.0,
        };

        results.push((id, score));
        i += 2;
    }

    results
}

/// Single-record VADD update (for mixed benchmark).
fn vadd_single(
    conn: &mut Connection,
    config: &VectorSetsConfig,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
) -> Result<(), String> {
    let vec_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut cmd = redis::cmd("VADD");
    cmd.arg("idx")
        .arg("FP32")
        .arg(&vec_bytes[..])
        .arg(id.to_string())
        .arg(&config.quant)
        .arg("M")
        .arg(config.m)
        .arg("EF")
        .arg(config.ef_construction);

    if config.cas {
        cmd.arg("CAS");
    }

    if let Some(json_str) = metadata.and_then(metadata_to_attr_json) {
        cmd.arg("SETATTR").arg(json_str);
    }

    cmd.query::<()>(conn)
        .map_err(|e| format!("VADD update error: {}", e))
}

impl Engine for VectorSetsEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, _dataset: &Dataset) -> Result<(), String> {
        // NOTE: nothing schema-derived is cached here. The datetime
        // representation is decided per VALUE by `epoch_literal`, which both the
        // storage and the query half consult, so there is no schema state that
        // could go stale on the `--skip-upload` path.
        let mut conn = self.get_connection()?;

        println!(
            "Using VectorSets with QUANT: {}, M: {}, EF_CONSTRUCTION: {}, CAS: {}",
            self.config.quant, self.config.m, self.config.ef_construction, self.config.cas
        );

        // Delete existing key if any
        let _ = redis::cmd("DEL").arg("idx").query::<()>(&mut conn);

        self.commandstats_baseline = redis_utils::reset_commandstats(&mut conn)?;
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
        let total_time = read_time + upload_time;

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );
        println!("Total time: {:.3}s", total_time);

        // Verify no VADD failures occurred during upload
        let mut conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut conn,
            &["VADD"],
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
        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let filter_ef: Option<i64> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|extra| extra.get("filter_ef"))
            .and_then(|v| v.as_i64());
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        // Pre-build filter expressions for each query
        let filters: Vec<Option<String>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(build_filter_expression))
            .collect();

        // Precompute the encoded query blobs BEFORE the timed region so the
        // per-query window wraps ONLY the RPC round-trip + reply parse (matching
        // pgvector/qdrant). Encoding the FP32 blob is client work, not server
        // latency. Shared read-only across workers.
        let encoded_queries: Vec<Vec<u8>> =
            queries.iter().map(|q| encode_query_vector(q)).collect();

        // When top is explicitly set, use it for all queries.
        // When not set, use per-query ground truth count (matches Python v0 behavior
        // where top defaults to len(query.expected_result) per query).
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

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
                let neighbors = &neighbors;
                let filters = &filters;
                let encoded_queries = &encoded_queries;
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
                    let mut pb_pending: u64 = 0;

                    let client = match redis::Client::open(redis_url.as_str()) {
                        Ok(c) => c,
                        Err(_) => {
                            // Still cross both barriers so peers aren't stranded.
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };
                    let mut conn = match client.get_connection() {
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
                    if !encoded_queries.is_empty() {
                        let prime_top = explicit_top.unwrap_or(10);
                        let _ = vsim_search(
                            &mut conn,
                            &encoded_queries[0],
                            prime_top,
                            ef,
                            filters[0].as_deref(),
                            filter_ef,
                        );
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

                        // Per-query top: use explicit top if set, otherwise
                        // use ground truth count (matches Python v0 behavior)
                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let filter_ref = filters[idx].as_deref();
                        // Timed window: precomputed blob is passed in, so this wraps
                        // only the RPC round-trip and reply parse.
                        let query_start = Instant::now();
                        let results = vsim_search(
                            &mut conn,
                            &encoded_queries[idx],
                            top,
                            ef,
                            filter_ref,
                            filter_ef,
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

            // All workers are spawned; wait until each has connected + primed,
            // stamp the shared measurement start, then release them together.
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
        // total_time excludes connection setup and the cold first query — it is
        // measured from the barrier release stamped into `start_cell`.
        let total_time = start_cell
            .get()
            .map(|st| st.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        // Verify no VSIM failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["VSIM"],
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
        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let filter_ef: Option<i64> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|extra| extra.get("filter_ef"))
            .and_then(|v| v.as_i64());
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let filters: Vec<Option<String>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(build_filter_expression))
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
        let mut u_times: Vec<f64> = Vec::new();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
                let config = self.config.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let filters = &filters;
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

                            let filter_ref = filters[idx].as_deref();
                            // NOTE: the mixed (search+update) path is intentionally
                            // left as-is for a later PR — the encode stays inside
                            // the timed window here to preserve its current
                            // measurement behavior exactly.
                            let query_start = Instant::now();
                            let vec_bytes = encode_query_vector(&queries[idx]);
                            let results =
                                vsim_search(&mut conn, &vec_bytes, top, ef, filter_ref, filter_ef);
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
                            let _ = vadd_single(
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

        // Verify no failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["VSIM", "VADD"],
            "mixed",
            self.commandstats_baseline.as_ref(),
        )?;

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
        let mut conn = self.get_connection()?;
        let _ = redis::cmd("DEL").arg("idx").query::<()>(&mut conn);
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

        let info_str: String = redis::cmd("INFO").arg("memory").query(&mut conn).ok()?;
        let used_memory: i64 = info_str
            .lines()
            .find(|l| l.starts_with("used_memory:"))
            .and_then(|l| l.strip_prefix("used_memory:"))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);

        // Vector-set stats via VINFO (quant-type, hnsw-m, vector-dim, size, ...).
        let index_info: Option<serde_json::Value> = redis::cmd("VINFO")
            .arg("idx")
            .query::<redis::Value>(&mut conn)
            .ok()
            .map(|v| redis_utils::value_to_json(&v));

        Some(serde_json::json!({
            "used_memory": used_memory,
            "shards": 1,
            "index_info": index_info,
        }))
    }

    fn server_metadata(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;
        let mut meta = redis_utils::collect_server_metadata(&mut conn);
        // VectorSets has no FT index; VINFO returns the vector-set metadata.
        // Errors on the BEFORE snapshot (vset not yet created) → index_info: null.
        let vinfo = redis::cmd("VINFO")
            .arg("idx")
            .query::<redis::Value>(&mut conn)
            .ok()
            .map(|v| redis_utils::value_to_json(&v));
        if let Some(obj) = meta.as_object_mut() {
            obj.insert(
                "index_info".to_string(),
                vinfo.unwrap_or(serde_json::Value::Null),
            );
        }
        Some(meta)
    }
}

// ── Filter expression builder ────────────────────────────────────────────
// Converts JSON filter conditions into VectorSets FILTER expression syntax.
// VectorSets uses dot-notation field access with standard comparison operators:
//   .field == "value"   .field > 10   .field in ["a", "b"]

/// Build a VectorSets FILTER expression from JSON conditions.
fn build_filter_expression(conditions: &serde_json::Value) -> Option<String> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }
    build_group(obj)
}

/// Build one boolean GROUP (a `{and:[…], or:[…]}` object) into a VSIM FILTER
/// sub-expression: `and` entries joined by `and`, `or` entries joined by `or`
/// inside parentheses, the two then ANDed. Recursive — an entry inside `and`/`or`
/// may itself be a group, so this and [`build_clauses`] call each other. Mirrors
/// `redis::build_group` / `build_subfilters`.
fn build_group(obj: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let and_entries = obj.get("and").and_then(|v| v.as_array());
    let or_entries = obj.get("or").and_then(|v| v.as_array());

    let mut parts = Vec::new();

    if let Some(entries) = and_entries {
        let clauses = build_clauses(entries);
        if !clauses.is_empty() {
            parts.push(clauses.join(" and "));
        }
    }

    if let Some(entries) = or_entries {
        let clauses = build_clauses(entries);
        if !clauses.is_empty() {
            parts.push(format!("({})", clauses.join(" or ")));
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" and "))
    }
}

fn build_clauses(entries: &[serde_json::Value]) -> Vec<String> {
    let mut clauses = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            // Nested GROUP: an entry carrying an `and`/`or` key whose value is an
            // ARRAY is a sub-tree, not a field leaf. Recurse and keep it as ONE
            // parenthesised clause so `{"or":[{"and":[…]},{"and":[…]}]}` keeps its
            // shape. Without this arm the entry matched no field leaf, every
            // clause was dropped, and a nested condition left VSIM with NO FILTER
            // — a full unfiltered search (same silent-wrong class as issue #220).
            //
            // Two things this arm must NOT do, both of which would re-introduce
            // exactly that failure mode:
            //  * Key off mere key PRESENCE. A field literally named `and`/`or`
            //    carries a condition object (`{"and":{"match":{"value":"x"}}}`),
            //    not an array; treating it as a group yields no clause at all and
            //    the query runs UNFILTERED. `as_array()` distinguishes them.
            //  * `continue` past the field-leaf loop. A mixed entry such as
            //    `{"and":[…], "color":{"match":{"value":"red"}}}` would silently
            //    DROP `color` and under-constrain the filter, so we fall through
            //    and emit the sibling leaves as well. `build_group` itself only
            //    ever consults the `and`/`or` keys, so nothing is double-emitted.
            let is_group_key = |name: &str, value: &serde_json::Value| {
                (name == "and" || name == "or") && value.is_array()
            };
            if entry_obj.iter().any(|(k, v)| is_group_key(k, v)) {
                if let Some(expr) = build_group(entry_obj) {
                    clauses.push(format!("({})", expr));
                }
            }
            for (field_name, field_filters) in entry_obj {
                // Skip the group keys consumed above; every other key is a field.
                if is_group_key(field_name, field_filters) {
                    continue;
                }
                if let Some(filter_obj) = field_filters.as_object() {
                    for (condition_type, criteria) in filter_obj {
                        if let Some(expr) = build_clause(field_name, condition_type, criteria) {
                            clauses.push(expr);
                        }
                    }
                }
            }
        }
    }
    clauses
}

fn build_clause(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<String> {
    match condition_type {
        "match" => build_match_clause(field_name, criteria),
        "range" => build_range_clause(field_name, criteria),
        // STILL OPEN, same silent-wrong class as issue #220: an unhandled
        // condition type is DROPPED, and when it is the only condition VSIM runs
        // with no FILTER at all. `"geo"` is the live instance — the point IS
        // uploaded (see `metadata_to_attr_json`) but has no clause here, so every
        // query of the shipped `random-geo-radius-{100,2048}-angular-filters`
        // runs unfiltered against geo-filtered ground truth (issue #223). The
        // generic guard that would turn "parsed to nothing" into a loud failure
        // instead of an unfiltered search is issue #219.
        _ => None,
    }
}

fn build_match_clause(field_name: &str, criteria: &serde_json::Value) -> Option<String> {
    // match_any (IN-list) takes precedence: {"match": {"any": [...]}}. The
    // canonical IN-list shape has NO "value" key, so without this arm the whole
    // clause returned None; when every clause is None, VSIM omits FILTER entirely
    // and runs UNFILTERED. Emitted as an OR-of-equality expression, mirroring the
    // OR-of-values semantics of redis build_match_any_filter / qdrant matches().
    if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
        return Some(build_match_any_clause(field_name, any));
    }

    // Full-text {"match": {"text": ...}}.
    //
    // IMPORTANT: VectorSets has NO real tokenized full-text index — `contains`
    // and `startswith` BOTH error on a live server (verified on Redis 8.8). The
    // best available approximation is `"<text>" in .field`, whose value-`in`-field
    // form does SUBSTRING/word-membership matching on a scalar string (verified:
    // `"quick" in .body` matches `.body == "the quick brown fox"`). This is a
    // best-effort single-term "contains", NOT true full-text: no stemming, no
    // relevance ranking, no multi-term/phrase logic, and it can over-match a term
    // that appears as a substring of a longer word. So text-filter recall for
    // VectorSets is NOT directly comparable to a tokenized engine (e.g. redis
    // `@field:(token)`). It is emitted (never dropped) so the query is never run
    // UNFILTERED. Blank text → an unsatisfiable never-match.
    if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Some(never_match_clause(field_name));
        }
        return Some(format!("\"{}\" in .{}", escape_str(trimmed), field_name));
    }

    let value = criteria.get("value")?;
    // bool → `.field == "true"`/`"false"`. Booleans are stored as JSON STRINGS
    // ("true"/"false") on upload (see readers::metadata), and a BARE bool literal
    // `.field == true` is a SYNTAX ERROR in VSIM FILTER (verified on Redis 8.8 —
    // it errors the WHOLE query), so it must be quoted to match storage. Mirrors
    // redis build_exact_match_filter's true/false token. Checked before the
    // numeric arms because serde treats JSON true/false as neither i64 nor f64.
    if let Some(b) = value.as_bool() {
        let token = if b { "true" } else { "false" };
        return Some(format!(".{} == \"{}\"", field_name, token));
    }
    if let Some(s) = value.as_str() {
        // A datetime is stored as an epoch NUMBER (see `epoch_literal`), so the
        // equality must be numeric too — a quoted ISO literal would compare a
        // string against a number and match NOTHING.
        Some(equality_clause(field_name, s))
    } else if let Some(n) = value.as_i64() {
        Some(format!(".{} == {}", field_name, n))
    } else {
        value.as_f64().map(|f| format!(".{} == {}", field_name, f))
    }
}

/// Escape a string literal for a VectorSets FILTER expression (backslash first,
/// then double quote).
fn escape_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// `.field == <literal>` for a STRING query value, matching how upload stored it:
/// an ISO-8601 value lives as an epoch NUMBER (see [`epoch_literal`]) and needs a
/// BARE numeric literal, everything else needs a quoted string. Getting this wrong
/// is silent — `.ts == "2021-01-01T00:00:00Z"` against epoch storage returns an
/// empty result set with no error (verified live on redis 8.8).
fn equality_clause(field_name: &str, value: &str) -> String {
    match epoch_literal(value) {
        Some(epoch) => format!(".{} == {}", field_name, epoch),
        None => format!(".{} == \"{}\"", field_name, escape_str(value)),
    }
}

/// Build a `match_any` (IN-list) clause as a per-element OR, covering keyword,
/// int AND float lists. Forms verified against a live VectorSets server
/// (Redis 8.8):
///
/// - string element on a SCALAR keyword field → `.field == "<escaped>"` — EXACT
///   equality. NOTE: the value-`in`-field form (`"v" in .field`) does
///   SUBSTRING/word matching on a scalar string (`"blue" in .color` wrongly
///   matches `.color == "dark blue"`), which violates the exact/whole-value
///   keyword semantics of qdrant/redis match_any, so we use `==` for scalars.
/// - string element on the multi-valued `labels` field → `"<escaped>" in .labels`
///   — ARRAY contains-any. `labels` is the ONLY array field (MetadataValue::Labels
///   is produced solely for key == "labels", see readers::metadata); VSIM stores
///   it as a JSON array and `"v" in .labels` performs true array membership
///   (verified on Redis 8.8), whereas scalar `.labels == "v"` could never match an
///   array. The `in` form is safe here BECAUSE the field is an array — the
///   substring hazard above only applies to scalar strings.
/// - numeric element (i64 OR f64) → `.field == <n>` — matches the native JSON
///   number stored on upload (SETATTR emits a real number for Int/Float fields).
/// - OR all representable elements, parenthesized. Empty-string tokens are dropped
///   (they can never match an exact keyword).
/// - Empty / no representable values → a never-match contradiction so an empty
///   IN-set matches NOTHING rather than being dropped (dropping the sole clause
///   would leave no FILTER and run over ALL vectors — the inverse of the filter).
fn build_match_any_clause(field_name: &str, any: &[serde_json::Value]) -> String {
    // `labels` is the only multi-valued (array) field; it needs `in` for
    // contains-any membership. All other keyword fields are scalar and use `==`.
    let is_array_field = field_name == "labels";
    let clauses: Vec<String> = any
        .iter()
        .filter_map(|v| {
            if let Some(s) = v.as_str() {
                if s.is_empty() {
                    None
                } else if is_array_field {
                    // `labels` is stored as a JSON array of raw strings — never
                    // epoch-converted — so membership stays a string test.
                    Some(format!("\"{}\" in .{}", escape_str(s), field_name))
                } else {
                    // Scalar: epoch-aware, matching the stored representation.
                    Some(equality_clause(field_name, s))
                }
            } else if let Some(i) = v.as_i64() {
                Some(format!(".{} == {}", field_name, i))
            } else {
                v.as_f64().map(|f| format!(".{} == {}", field_name, f))
            }
        })
        .collect();

    match clauses.len() {
        0 => never_match_clause(field_name),
        1 => clauses.into_iter().next().unwrap(),
        _ => format!("({})", clauses.join(" or ")),
    }
}

/// An unsatisfiable clause (`.f == "x" and .f != "x"`) used when a filter can
/// never match, so it restricts to NOTHING instead of being dropped (which would
/// leave the query unfiltered).
fn never_match_clause(field_name: &str) -> String {
    format!(
        "(.{0} == \"__never_match__\" and .{0} != \"__never_match__\")",
        field_name
    )
}

fn build_range_clause(field_name: &str, criteria: &serde_json::Value) -> Option<String> {
    let mut parts = Vec::new();

    // Only representable bounds constrain the range; a null / non-numeric,
    // non-datetime bound carries no constraint and is skipped (matching
    // redis/valkey/qdrant/es/os/weaviate/milvus/pgvector) — otherwise
    // `number_literal` would have to invent a "0" and emit `.n > 0`.
    for (key, op) in [("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<=")] {
        if let Some(lit) = criteria.get(key).and_then(number_literal) {
            parts.push(format!(".{} {} {}", field_name, op, lit));
        }
    }

    match parts.len() {
        0 => None,
        1 => parts.pop(),
        // Parenthesise a multi-bound range. VSIM binds `and` tighter than `or`
        // (verified live on redis 8.8), so a bare `A and B` is correct TODAY even
        // when this clause lands inside an `or` list — but that is an implicit
        // dependency on the server's operator precedence. The parentheses make the
        // grouping explicit and cost nothing.
        _ => Some(format!("({})", parts.join(" and "))),
    }
}

/// Render a range bound as a VSIM numeric literal.
///
/// VSIM `FILTER` comparison operators are NUMERIC — there is no date type — so an
/// ISO-8601 / RFC-3339 bound is converted to epoch seconds, the same
/// representation `encode_string_attr` stores (see [`epoch_literal`]). A numeric
/// string is passed through as a number too (so a raw-epoch bound works). Mirrors
/// `valkey::number_literal` / `redis::insert_number_param`, both of which route
/// through the shared `parsers::datetime_to_epoch_secs`.
///
/// Without the datetime arm BOTH bounds of a `{"gte":"<iso>","lt":"<iso>"}` range
/// were dropped, `build_range_clause` returned `None`, and VSIM ran with NO
/// `FILTER` argument — a full unfiltered search scored against filtered ground
/// truth (issue #220; 1 in 4 queries of the shipped `synthetic-filter-32`).
///
/// CRITICAL: the numeric-string arm emits the PARSED number, never the original
/// string. Rust's parsers accept spellings the VSIM FILTER grammar does not, and
/// re-emitting the raw text produced two distinct failures (both verified live on
/// redis 8.8):
///   * `"inf"` / `"NaN"` / `"+7"` → `ERR syntax error in FILTER expression`. A
///     failed query is only `eprintln!`d and is then DROPPED from the recall and
///     latency vectors, so the reported mean is quietly computed over fewer
///     queries rather than scored 0 — a loud error that lands silently.
///   * `".5"` → NO error and ZERO results: VSIM parses a leading `.` as a FIELD
///     SELECTOR, so `.n >= .5` compares two fields. Wrong recall, no diagnostic.
///
/// Non-finite values have no VSIM literal at all and are rejected outright.
fn number_literal(value: &serde_json::Value) -> Option<String> {
    if let Some(i) = value.as_i64() {
        Some(i.to_string())
    } else if let Some(f) = value.as_f64() {
        finite_literal(f)
    } else if let Some(s) = value.as_str() {
        if let Some(epoch) = epoch_literal(s) {
            // Epoch is whole seconds — emit as an integer literal.
            Some(epoch.to_string())
        } else if let Ok(i) = s.parse::<i64>() {
            // Integer-first, so a 19-digit epoch/ID keeps every digit (an f64
            // round-trip would silently round it).
            Some(i.to_string())
        } else {
            s.parse::<f64>().ok().and_then(finite_literal)
        }
    } else {
        None
    }
}

/// Render an `f64` as a VSIM numeric literal, rejecting the non-finite values
/// (`inf`, `-inf`, `NaN`) that the FILTER grammar has no spelling for.
fn finite_literal(f: f64) -> Option<String> {
    f.is_finite().then(|| f.to_string())
}

#[cfg(test)]
mod vsim_parse_tests {
    use super::{build_filter_expression, encode_query_vector, parse_vsim_response};
    use redis::Value;
    use serde_json::json;

    // ── Timed-window hoisting fidelity ─────────────────────────────────────
    // The perf change moves the FP32 encode OUT of the per-query timed window
    // (precomputed once before the parallel region). Prove the precomputed
    // blobs are byte-identical to the old in-window encode.

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
    fn build_filter_expression_filtered_matches_legacy_string() {
        // VectorSets has no hoisted query_str — the per-query-varying request bit
        // is the VSIM FILTER expression (prebuilt via build_filter_expression
        // before the timed loop, then passed as `cmd.arg("FILTER").arg(expr)`).
        // Pin a NON-trivial compound filter to its exact legacy FILTER string so
        // the filter-string path cannot silently diverge.
        let cond = json!({"and": [
            {"brand": {"match": {"value": "apple"}}},
            {"price": {"range": {"gte": 100}}},
        ]});
        assert_eq!(
            build_filter_expression(&cond),
            Some(".brand == \"apple\" and .price >= 100".to_string())
        );
    }

    #[test]
    fn parses_resp2_bulk_id_and_score_as_distance() {
        // RESP2: id + score both bulk strings; score 0.9 similarity → 0.1 distance.
        let resp = vec![
            Value::BulkString(b"7".to_vec()),
            Value::BulkString(b"0.9".to_vec()),
        ];
        let hits = parse_vsim_response(&resp);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 7);
        assert!((hits[0].1 - 0.1).abs() < 1e-9, "distance={}", hits[0].1);
    }

    #[test]
    fn parses_resp3_int_id_and_double_score() {
        // RESP3: id as Int, score as Double similarity 1.0 → distance 0.0.
        let resp = vec![Value::Int(42), Value::Double(1.0)];
        let hits = parse_vsim_response(&resp);
        assert_eq!(hits, vec![(42, 0.0)]);
    }

    #[test]
    fn multiple_pairs_and_trailing_odd_element_ignored() {
        let resp = vec![
            Value::Int(1),
            Value::Double(0.5),
            Value::Int(2),
            Value::Double(0.25),
            Value::Int(3), // dangling id with no score → dropped
        ];
        let hits = parse_vsim_response(&resp);
        assert_eq!(hits, vec![(1, 0.5), (2, 0.75)]);
    }

    #[test]
    fn unknown_variants_fall_back_without_panicking() {
        let resp = vec![Value::Nil, Value::Nil];
        assert_eq!(parse_vsim_response(&resp), vec![(0, 0.0)]);
        assert_eq!(parse_vsim_response(&[]), vec![]);
    }
}

#[cfg(test)]
mod filter_expr_tests {
    use super::{build_filter_expression, encode_string_attr};
    use serde_json::json;

    // ── scalar match / range (grammar baseline) ──

    #[test]
    fn and_string_match() {
        let cond = json!({"and": [{"color": {"match": {"value": "red"}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some(".color == \"red\"".to_string())
        );
    }

    #[test]
    fn and_numeric_match() {
        let cond = json!({"and": [{"f": {"match": {"value": 10}}}]});
        assert_eq!(build_filter_expression(&cond), Some(".f == 10".to_string()));
    }

    #[test]
    fn string_value_quotes_and_backslashes_escaped() {
        let cond = json!({"and": [{"name": {"match": {"value": "a\"b\\c"}}}]});
        // " → \" and \ → \\ (backslash escaped first).
        assert_eq!(
            build_filter_expression(&cond),
            Some(".name == \"a\\\"b\\\\c\"".to_string())
        );
    }

    #[test]
    fn range_bounds_joined_with_and() {
        let cond = json!({"and": [{"age": {"range": {"gt": 1, "gte": 2, "lt": 9, "lte": 8}}}]});
        // Parenthesised: a multi-bound range is ONE clause and must stay one when
        // it lands inside an `or` list, without leaning on VSIM's precedence.
        assert_eq!(
            build_filter_expression(&cond),
            Some("(.age > 1 and .age >= 2 and .age < 9 and .age <= 8)".to_string())
        );
    }

    #[test]
    fn or_block_parenthesized() {
        let cond = json!({"or": [
            {"a": {"match": {"value": "x"}}},
            {"b": {"match": {"value": "y"}}},
        ]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("(.a == \"x\" or .b == \"y\")".to_string())
        );
    }

    // ── FAILING-then-fixed: previously each returned None → unfiltered ──

    #[test]
    fn match_any_keyword_list_yields_exact_equality_or() {
        // EXACT equality (`==`), NOT `"v" in .field`: on a live Redis 8.8 server
        // `"blue" in .color` substring-matches `.color == "dark blue"`, breaking
        // whole-value keyword semantics — verified in integration_vectorsets.
        let cond = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("(.color == \"red\" or .color == \"blue\")".to_string())
        );
    }

    #[test]
    fn match_any_int_list_yields_numeric_equality_or() {
        // Numbers use `.field == N` (value-in-field does NOT work for numbers).
        let cond = json!({"and": [{"size": {"match": {"any": [1, 2]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("(.size == 1 or .size == 2)".to_string())
        );
    }

    #[test]
    fn match_any_float_list_yields_numeric_equality_or() {
        let cond = json!({"and": [{"score": {"match": {"any": [1.5, 2.5]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("(.score == 1.5 or .score == 2.5)".to_string())
        );
    }

    #[test]
    fn match_any_mixed_list_uses_per_element_form() {
        // Mixed int + string: numbers use `== N`, strings use exact `== "s"`.
        let cond = json!({"and": [{"n": {"match": {"any": [1, "red"]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("(.n == 1 or .n == \"red\")".to_string())
        );
    }

    #[test]
    fn match_any_single_value_unwrapped() {
        let cond = json!({"and": [{"color": {"match": {"any": ["red"]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some(".color == \"red\"".to_string())
        );
    }

    // #121: the multi-valued `labels` field is stored as a JSON ARRAY, so
    // match_any must use `"v" in .labels` (array contains-any). On a live Redis
    // 8.8 server `"x" in .labels` performs array membership; scalar
    // `.labels == "x"` could never match an array.
    #[test]
    fn match_any_labels_uses_array_contains_any_in_form() {
        let cond = json!({"and": [{"labels": {"match": {"any": ["red", "blue"]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("(\"red\" in .labels or \"blue\" in .labels)".to_string())
        );
    }

    #[test]
    fn match_any_labels_single_value_unwrapped_in_form() {
        let cond = json!({"and": [{"labels": {"match": {"any": ["red"]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("\"red\" in .labels".to_string())
        );
    }

    // A SCALAR keyword field (anything but `labels`) keeps EXACT `==`; the `in`
    // form would do substring matching and break whole-value semantics.
    #[test]
    fn match_any_scalar_keyword_unchanged_exact_equality() {
        let cond = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("(.color == \"red\" or .color == \"blue\")".to_string())
        );
    }

    #[test]
    fn match_any_empty_list_is_non_none_never_match() {
        // Must NOT be dropped (dropping the sole clause → unfiltered).
        let cond = json!({"and": [{"color": {"match": {"any": []}}}]});
        let expr = build_filter_expression(&cond).expect("empty match_any must not be dropped");
        assert!(expr.contains("=="), "expr={}", expr);
        assert!(expr.contains("!="), "expr={}", expr);
    }

    #[test]
    fn full_text_match_degrades_to_value_in_field() {
        // VSIM has no tokenized text; best-effort exact-match via value-in-field.
        let cond = json!({"and": [{"body": {"match": {"text": "quick"}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some("\"quick\" in .body".to_string())
        );
    }

    #[test]
    fn bool_value_quotes_true_matching_string_storage() {
        // Bare `.flag == true` is a VSIM syntax error; bools are stored as the
        // JSON strings "true"/"false", so we must quote.
        let cond = json!({"and": [{"flag": {"match": {"value": true}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some(".flag == \"true\"".to_string())
        );
    }

    #[test]
    fn bool_value_quotes_false() {
        let cond = json!({"and": [{"flag": {"match": {"value": false}}}]});
        assert_eq!(
            build_filter_expression(&cond),
            Some(".flag == \"false\"".to_string())
        );
    }

    // ── empty / passthrough ──

    #[test]
    fn empty_conditions_none() {
        assert!(build_filter_expression(&json!({})).is_none());
        assert!(build_filter_expression(&json!({"and": [], "or": []})).is_none());
    }

    // ── OR-branch: combined AND + OR ───────────────────────────────────────

    #[test]
    fn and_plus_or_joins_and_group_with_parenthesized_or_group() {
        let cond = json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}}],
        });
        // AND clause (no parens for a single clause) joined with the parenthesized
        // OR group by ` and `.
        assert_eq!(
            build_filter_expression(&cond),
            Some(".a == \"x\" and (.b == \"y\")".to_string())
        );
    }

    // ── Range operators (individual arms) ──────────────────────────────────

    fn range_expr(criteria: serde_json::Value) -> Option<String> {
        build_filter_expression(&json!({"and":[{"n":{"range":criteria}}]}))
    }

    #[test]
    fn range_lt_is_exclusive() {
        assert_eq!(range_expr(json!({"lt":5})).unwrap(), ".n < 5");
    }

    #[test]
    fn range_lte_is_inclusive() {
        assert_eq!(range_expr(json!({"lte":5})).unwrap(), ".n <= 5");
    }

    #[test]
    fn range_gt_is_exclusive() {
        assert_eq!(range_expr(json!({"gt":5})).unwrap(), ".n > 5");
    }

    #[test]
    fn range_gte_is_inclusive() {
        assert_eq!(range_expr(json!({"gte":5})).unwrap(), ".n >= 5");
    }

    #[test]
    fn range_two_sided_gte_lt() {
        // Fixed order gt, gte, lt, lte joined by ` and `, parenthesised as one
        // clause so the grouping survives being nested inside an `or`.
        assert_eq!(
            range_expr(json!({"gte":10,"lt":20})).unwrap(),
            "(.n >= 10 and .n < 20)"
        );
    }

    #[test]
    fn range_unknown_op_is_none() {
        assert!(range_expr(json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_is_skipped() {
        // A null bound carries no constraint, so the clause is skipped (→ None),
        // matching redis/valkey/qdrant/es/os/weaviate/milvus/pgvector. (Regression
        // guard: build_range_clause used to emit `.n > 0` via format_number(Null).)
        let got = range_expr(json!({"gt": serde_json::Value::Null}));
        assert!(got.is_none(), "null bound should be skipped, got {:?}", got);
    }

    // ── Exact-match float / non-scalar arms ────────────────────────────────

    #[test]
    fn exact_match_float_emits_equality() {
        assert_eq!(
            build_filter_expression(&json!({"and":[{"score":{"match":{"value":1.5}}}]})),
            Some(".score == 1.5".to_string())
        );
    }

    #[test]
    fn exact_match_array_value_is_none() {
        // A non-scalar (array) value matches no scalar arm → clause dropped → None.
        assert!(
            build_filter_expression(&json!({"and":[{"n":{"match":{"value":[1,2]}}}]})).is_none()
        );
    }

    // ── issue #220: datetime range bounds ──────────────────────────────────

    #[test]
    fn range_iso_datetime_bounds_become_epoch_seconds() {
        // Both bounds used to be rejected by `.filter(|v| v.is_number())`, so the
        // whole clause was None and VSIM ran with NO FILTER (full unfiltered
        // search scored against filtered ground truth). Now each ISO bound is
        // converted to epoch seconds, matching how upload stores the attribute.
        assert_eq!(
            range_expr(json!({"gte":"2021-04-11T00:00:00+00:00","lt":"2021-10-28T00:00:00+00:00"}))
                .unwrap(),
            "(.n >= 1618099200 and .n < 1635379200)"
        );
    }

    #[test]
    fn range_epoch_string_bound_passes_through() {
        // A raw-epoch string bound is a valid numeric literal (mirrors
        // valkey::number_literal), so it is NOT dropped either.
        assert_eq!(
            range_expr(json!({"gte":"1609459200"})).unwrap(),
            ".n >= 1609459200"
        );
    }

    #[test]
    fn range_non_datetime_string_bound_is_skipped() {
        // An unparseable string still carries no constraint → clause skipped.
        assert!(range_expr(json!({"gte":"not-a-date"})).is_none());
    }

    #[test]
    fn datetime_range_condition_emits_a_filter() {
        // End-to-end shape of the shipped synthetic-filter-32 datetime query.
        assert_eq!(
            build_filter_expression(&json!({"and":[
                {"ts":{"range":{"gte":"2021-01-01T00:00:00Z","lt":"2021-01-02T00:00:00Z"}}}
            ]})),
            Some("(.ts >= 1609459200 and .ts < 1609545600)".to_string())
        );
    }

    #[test]
    fn datetime_value_is_stored_as_epoch_number() {
        // Upload MUST store the same representation the filter compares against;
        // an ISO string attribute would never satisfy `.ts >= <epoch>`.
        assert_eq!(
            encode_string_attr("2021-01-01T00:00:00Z"),
            json!(1609459200i64)
        );
        // A non-datetime string → verbatim.
        assert_eq!(encode_string_attr("red"), json!("red"));
        // Already an epoch (not ISO) → verbatim string, as before.
        assert_eq!(encode_string_attr("1609459200"), json!("1609459200"));
        // Bool tokens (readers store bools as "true"/"false" strings) → verbatim,
        // so `.flag == "true"` keeps matching.
        assert_eq!(encode_string_attr("true"), json!("true"));
    }

    // ── review M2: the storage half and the filter half must agree WITHOUT a
    // schema. `encode_string_attr` used to be gated on `config.datetime_fields`
    // (primed from `dataset.config.schema`) while the filter half converted any
    // ISO bound unconditionally. For a datetime-carrying field NOT declared
    // `datetime` the two disagreed and recall went silently wrong: VSIM coerces a
    // non-numeric string attribute to `0`, so a two-sided range matched NOTHING
    // and a one-sided `lt`/`lte` matched EVERYTHING (verified live, redis 8.8 —
    // measured 0.800 on the schema-gated build). Both halves now decide from the
    // VALUE alone, so they agree for `keyword`, for `datetime`, and for no schema
    // at all.

    #[test]
    fn storage_and_filter_agree_on_datetime_without_any_schema() {
        // The exact pair the engine emits for one document + one query bound.
        // Before the fix the left side was `"2021-04-11T00:00:00+00:00"` (a
        // string) while the right side was already `1618099200` (a number).
        assert_eq!(
            encode_string_attr("2021-04-11T00:00:00+00:00"),
            json!(1618099200i64)
        );
        assert_eq!(
            range_expr(json!({"gte":"2021-04-11T00:00:00+00:00"})).unwrap(),
            ".n >= 1618099200"
        );
    }

    #[test]
    fn one_sided_datetime_bound_cannot_silently_match_everything() {
        // The inversion that made this HIGH severity: with ISO-string storage a
        // `lt`-only bound returned EVERY document (string coerced to 0 < epoch),
        // exit code 0, no warning. Pinning both halves to the numeric
        // representation is what prevents it, so assert them together.
        assert_eq!(
            range_expr(json!({"lt":"2021-10-28T00:00:00+00:00"})).unwrap(),
            ".n < 1635379200"
        );
        let stored = encode_string_attr("2021-04-11T00:00:00+00:00");
        assert!(
            stored.is_number(),
            "datetime must be stored numerically or `.n < <epoch>` matches ALL docs"
        );
    }

    #[test]
    fn datetime_equality_matches_the_stored_epoch_number() {
        // review S2: the equality builders were schema-blind and emitted a QUOTED
        // ISO literal against epoch-number storage → zero hits, no error.
        assert_eq!(
            build_filter_expression(&json!({"and":[
                {"ts":{"match":{"value":"2021-01-01T00:00:00Z"}}}
            ]})),
            Some(".ts == 1609459200".to_string())
        );
        // match_any on a scalar datetime field, same rule.
        assert_eq!(
            build_filter_expression(&json!({"and":[
                {"ts":{"match":{"any":["2021-01-01T00:00:00Z","2021-01-02T00:00:00Z"]}}}
            ]})),
            Some("(.ts == 1609459200 or .ts == 1609545600)".to_string())
        );
        // `labels` is a JSON array of raw strings — never epoch-converted, so its
        // membership test must stay a STRING test.
        assert_eq!(
            build_filter_expression(&json!({"and":[
                {"labels":{"match":{"any":["2021-01-01T00:00:00Z"]}}}
            ]})),
            Some("\"2021-01-01T00:00:00Z\" in .labels".to_string())
        );
    }

    // ── review M1: a numeric-string bound must be emitted as the PARSED number.
    // Rust's `f64`/`i64` parsers accept spellings the VSIM FILTER grammar does
    // not; re-emitting the raw string produced a hard `ERR syntax error in FILTER
    // expression` (and a failed query is only `eprintln!`d, then DROPPED from the
    // recall/latency vectors — so the mean is quietly taken over fewer queries),
    // or, for `".5"`, ZERO results with no error at all because VSIM reads a
    // leading `.` as a FIELD SELECTOR. All verified live on redis 8.8.

    #[test]
    fn non_finite_string_bounds_are_rejected_not_emitted_raw() {
        for bad in ["inf", "-inf", "infinity", "Infinity", "NaN", "nan"] {
            assert!(
                range_expr(json!({ "gte": bad })).is_none(),
                "{bad:?} has no VSIM literal and must be dropped, not emitted"
            );
        }
    }

    #[test]
    fn rust_only_numeric_spellings_are_normalised_to_vsim_literals() {
        // `+7` and `.5` are valid Rust f64 syntax but NOT valid VSIM literals —
        // `.n >= +7` is a syntax ERROR and `.n >= .5` silently compares against a
        // field named `5`. Emit the parsed value instead.
        assert_eq!(range_expr(json!({"gte":"+7"})).unwrap(), ".n >= 7");
        assert_eq!(range_expr(json!({"gte":".5"})).unwrap(), ".n >= 0.5");
        assert_eq!(range_expr(json!({"gte":"-.5"})).unwrap(), ".n >= -0.5");
        // Forms VSIM DOES accept must not regress (both verified live).
        assert_eq!(range_expr(json!({"gte":"1e5"})).unwrap(), ".n >= 100000");
        assert_eq!(range_expr(json!({"gte":"0100"})).unwrap(), ".n >= 100");
        // A 19-digit id must keep every digit (an f64 round-trip would round it).
        assert_eq!(
            range_expr(json!({"gte":"9007199254740993"})).unwrap(),
            ".n >= 9007199254740993"
        );
    }

    // ── issue #220 follow-on: nested boolean groups ────────────────────────

    #[test]
    fn nested_or_of_and_groups_keeps_its_shape() {
        // `(color == red AND size >= 50) OR (color == blue AND size < 10)`.
        // Each `{"and":[…]}` entry used to match no field leaf → every clause
        // dropped → None → VSIM ran UNFILTERED.
        let cond = json!({"or":[
            {"and":[{"color":{"match":{"value":"red"}}},{"size":{"range":{"gte":50}}}]},
            {"and":[{"color":{"match":{"value":"blue"}}},{"size":{"range":{"lt":10}}}]},
        ]});
        assert_eq!(
            build_filter_expression(&cond),
            Some(
                "((.color == \"red\" and .size >= 50) or (.color == \"blue\" and .size < 10))"
                    .to_string()
            )
        );
    }

    #[test]
    fn flat_condition_string_is_unchanged_by_the_nesting_support() {
        // Regression guard: adding the recursion must not perturb the flat
        // top-level emission (no extra parentheses).
        assert_eq!(
            build_filter_expression(&json!({"and":[
                {"brand":{"match":{"value":"apple"}}},
                {"price":{"range":{"gte":100}}},
            ]})),
            Some(".brand == \"apple\" and .price >= 100".to_string())
        );
        assert_eq!(
            build_filter_expression(&json!({"or":[
                {"brand":{"match":{"value":"apple"}}},
                {"price":{"range":{"gte":100}}},
            ]})),
            Some("(.brand == \"apple\" or .price >= 100)".to_string())
        );
    }

    // ── review M3: the nested-group arm must not swallow the rest of the entry.

    #[test]
    fn nested_group_entry_keeps_its_sibling_field_leaves() {
        // The arm used to `continue` after recursing, so a mixed entry silently
        // DROPPED `color` — an under-constrained filter, the same silent-wrong
        // class this file is being fixed for. The group and the sibling leaf must
        // BOTH be emitted.
        let expr = build_filter_expression(&json!({"and":[
            {
                "and":[{"a":{"match":{"value":"x"}}},{"b":{"match":{"value":"y"}}}],
                "color":{"match":{"value":"red"}},
            }
        ]}))
        .expect("mixed group+leaf entry must produce a filter");
        assert!(
            expr.contains(".color == \"red\""),
            "sibling leaf dropped: {expr}"
        );
        assert!(
            expr.contains(".a == \"x\" and .b == \"y\""),
            "nested group dropped: {expr}"
        );
    }

    #[test]
    fn field_literally_named_and_or_or_is_still_a_field_leaf() {
        // Keying off key PRESENCE mistook a FIELD named `and`/`or` for a boolean
        // group; `build_group` then found no array, returned None, every clause
        // was dropped and VSIM ran UNFILTERED. Array-ness is the discriminator.
        assert_eq!(
            build_filter_expression(&json!({"and":[{"and":{"match":{"value":"x"}}}]})),
            Some(".and == \"x\"".to_string())
        );
        assert_eq!(
            build_filter_expression(&json!({"and":[{"or":{"range":{"gte":5}}}]})),
            Some(".or >= 5".to_string())
        );
    }
}
