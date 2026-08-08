//! KiviDB engine implementation.
//!
//! KiviDB (https://kividb.io) is a Redis-wire-compatible (RESP2) data store
//! that implements a RediSearch-compatible `FT.*` subset — `FT.CREATE`,
//! `FT.SEARCH`, `FT.INFO`, `FT.DROPINDEX` — with `VECTOR` fields (FLAT / HNSW)
//! and the KNN query syntax `*=>[KNN k @field $blob AS score]`. This engine
//! speaks that subset over the `redis` crate (same RESP protocol), and its
//! upload/search/filter code mirrors redis.rs / dragonfly.rs / valkey.rs.
//!
//! # Scope: vector KNN + metadata filtering
//!
//! KiviDB's schema (`src/vector/mod.rs::FieldType`) has exactly four field
//! kinds: `Text`, `Tag`, `Numeric`, `Vector` — no `Geo`. This engine therefore
//! indexes the dataset's metadata schema and applies per-query filter
//! `conditions` for keyword/int/float/bool/datetime/uuid datatypes, but never
//! declares or filters GEO fields — there is no GEO field type to declare,
//! unlike Dragonfly's parser-level `$param` rejection. A query with no
//! conditions still runs the `*` (match-all) prefilter.
//!
//! The filter expression is built by [`kividb_filter`] below — a KiviDB-SPECIFIC
//! builder, NOT redis.rs's shared RediSearch one. See that module's doc comment
//! for the measured list of ways KiviDB's `FT.SEARCH` diverges from RediSearch;
//! the headline one is that KiviDB does not substitute `$param` placeholders
//! inside a hybrid query's prefilter at all, so every filter value must be
//! inlined as a literal.
//!
//! # Vector data type: FLOAT32 only
//!
//! KiviDB's `FT.CREATE` rejects any vector `TYPE` other than `FLOAT32`
//! (`src/commands/vector.rs`) — no INT8/UINT8/FP16/BF16/FP64. Vectors are
//! therefore always encoded as FLOAT32 little-endian bytes.
//!
//! # EF_RUNTIME
//!
//! KiviDB parses the per-query `EF_RUNTIME` HNSW attribute
//! (`src/commands/vector.rs`), so it is kept, matching redis.rs / dragonfly.rs
//! / valkey.rs, so the search sweep's `ef` values take effect instead of
//! collapsing to the index default.
//!
//! # RESP2 only
//!
//! KiviDB does not implement RESP3 — there is no protocol opt-in here (unlike
//! dragonfly.rs's `DRAGONFLY_PROTOCOL=resp3`); every connection stays RESP2.
//!
//! # FT.INFO dialect: no `num_docs` / `percent_indexed`
//!
//! This is the detail that actually motivated this engine. KiviDB's `FT.INFO`
//! does not expose RediSearch's `num_docs` / `indexing` / `percent_indexed`
//! fields at all — a benchmark client that polls for those (as the generic
//! wait-for-indexing loops in this repo, and in the legacy `v0/` Python tool,
//! do) sees `num_docs` default to 0 forever and stalls until its own timeout,
//! even though the index is already fully built. Instead, KiviDB reports HNSW
//! graph state directly: `hnsw_live_count` (documents actually in the graph)
//! and `hnsw_compaction_in_progress` (`"0"`/`"1"`, set during background
//! tombstone compaction). `wait_for_indexing` below polls those fields
//! instead. Because KiviDB builds each vector's HNSW entry synchronously
//! inside the `HSET` that stores it, this in practice returns immediately —
//! there is no async backfill phase to wait out.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use super::redis_utils;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::index_naming::{derive_index_name, derive_key_prefix};
use crate::engine::{CorpusCount, Engine, SearchResults, UploadStats};
use crate::metrics::compute_metrics;
use vector_db_benchmark::parsers::{datetime_to_epoch_secs, doc_key_to_id, doc_key_to_id_opt};
use vector_db_benchmark::readers::metadata::{is_multivalued_keyword_field, MetadataItem};
use vector_db_benchmark::start_gate::WorkerPool;

/// KiviDB engine configuration.
#[derive(Clone)]
pub struct KividbEngineConfig {
    pub m: i64,
    pub ef_construction: i64,
    /// Always `FLOAT32` — KiviDB's `FT.CREATE` supports no other vector type.
    pub data_type: String,
    pub algorithm: String,
    pub batch_size: usize,
    pub parallel: usize,
    /// Per-config index name (`"<base>:<config>"`, mirrors redis.rs/dragonfly.rs)
    /// so a sweep's configs address disjoint indexes on one server. Resolved
    /// once in `new()`.
    pub index_name: String,
    /// Per-config key prefix (`"<config>:"`). Each config owns a disjoint
    /// keyspace; teardown is a prefix-scoped SCAN+UNLINK (no DD flag).
    pub key_prefix: String,
}

pub struct KividbEngine {
    name: String,
    host: String,
    port: u16,
    config: KividbEngineConfig,
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

impl KividbEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        // KiviDB's own default listen port is 6380, NOT Redis's 6379 (its
        // banner reads `Listening on 0.0.0.0:6380`, and the official
        // `quay.io/kividbio/kividb` image exposes 6380). Defaulting to 6379
        // meant an out-of-the-box run could not connect at all, or — worse —
        // silently benchmarked a real Redis that happened to be on 6379.
        let port: u16 = std::env::var("KIVIDB_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6380);

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

        // KiviDB's FT.CREATE only supports float32; ignore any configured override.
        let data_type = "FLOAT32".to_string();

        // Upload concurrency/batch come from the engine config, but each can be
        // overridden at runtime via env (taking precedence over the config),
        // mirroring dragonfly.rs / valkey.rs.
        let parallel = std::env::var("KIVIDB_UPLOAD_PARALLEL")
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

        let batch_size = std::env::var("KIVIDB_UPLOAD_BATCH_SIZE")
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
            config: KividbEngineConfig {
                m,
                ef_construction,
                data_type,
                algorithm,
                batch_size,
                parallel,
                index_name: derive_index_name("KIVIDB_INDEX_NAME", "idx", &engine_config.name),
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
        let auth = std::env::var("KIVIDB_AUTH").ok();
        let user = std::env::var("KIVIDB_USER").ok();

        let auth_part = match (&user, &auth) {
            (Some(u), Some(p)) => format!("{}:{}@", u, p),
            (None, Some(p)) => format!(":{}@", p),
            _ => String::new(),
        };

        // KiviDB is RESP2-only (no protocol opt-in, unlike dragonfly.rs).
        let url = format!("redis://{}{}:{}/0", auth_part, host, port);
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

        // Drop this config's index + keys ONLY (KiviDB has no DD flag on
        // FT.DROPINDEX either, so a prefix-scoped SCAN+UNLINK replaces a
        // keyspace-wide FLUSHALL, which would wipe sibling configs' data).
        redis_utils::drop_index_and_keys(conn, &self.config.index_name, &self.config.key_prefix);

        let distance_metric = map_distance_metric(distance);

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

        // Filterable metadata fields (mirrors redis.rs/dragonfly.rs):
        // keyword/uuid/bool exact strings -> TAG; int/float/datetime (stored as
        // epoch) -> NUMERIC; full-text -> TEXT.
        //
        // Two KiviDB-specific departures from the RediSearch schema (both
        // verified live against the FT.CREATE parser):
        //   * NO `SEPARATOR` modifier. KiviDB's FT.CREATE rejects it outright
        //     ("unknown field type"), and a KiviDB TAG value is atomic — it is
        //     never split on any separator (a query `@f:{b}` does not match a
        //     stored `a;b;c`). So we declare a bare TAG (exact whole-string
        //     match). Scalar keyword/uuid/bool filtering is unaffected; a
        //     genuinely multi-valued `labels` array is caught and rejected at
        //     upload time rather than indexed into a filter that silently
        //     matches nothing (see `upload_batch_internal`).
        //   * GEO is never declared: KiviDB's schema (`FieldType`) has no Geo
        //     variant at all, unlike Dragonfly's parser-level rejection of
        //     `$param` geo bounds — there is simply nothing to declare here.
        //     Unlike Chroma/Milvus, a geo CONDITION is then a hard error rather
        //     than a silent drop (see `kividb_filter`).
        if let Some(schema) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
            for (field_name, field_type) in schema {
                match field_type.as_str().unwrap_or("") {
                    "keyword" | "uuid" | "bool" => {
                        cmd.arg(field_name).arg("TAG");
                    }
                    "int" | "float" | "datetime" => {
                        cmd.arg(field_name).arg("NUMERIC");
                    }
                    "text" => {
                        cmd.arg(field_name).arg("TEXT");
                    }
                    // `"geo"` never reaches here: `validate_dataset_support`
                    // rejects the dataset in `configure()` before this runs, so
                    // no geo field is ever silently left undeclared. Anything
                    // else is a schema type this repo does not filter on.
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
                    let mut conn = match KividbEngine::connect(&host, port) {
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

    /// Wait until KiviDB's FT.INFO reports the HNSW graph as fully built.
    ///
    /// KiviDB has no `num_docs` / `indexing` / `percent_indexed` in FT.INFO —
    /// see the module doc comment. This polls `hnsw_live_count` (documents
    /// actually in the graph) and `hnsw_compaction_in_progress` instead. In
    /// practice this returns on the first poll: KiviDB builds each vector's
    /// HNSW entry synchronously inside the `HSET` that stores it, so by the
    /// time upload's HSETs have all returned, `hnsw_live_count` is already at
    /// `expected` and no compaction is running.
    fn wait_for_indexing(&self, expected: usize) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let max_wait = 600; // seconds – generous, even though KiviDB shouldn't need it
        let start = Instant::now();

        loop {
            let info: redis::Value = redis::cmd("FT.INFO")
                .arg(&self.config.index_name)
                .query(&mut conn)
                .map_err(|e| format!("FT.INFO error: {}", e))?;

            let mut live_count: usize = 0;
            let mut compaction_in_progress: bool = false;

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

            let mut handle_pair = |key: &str, val: &redis::Value| match key {
                "hnsw_live_count" => live_count = extract_usize(val),
                "hnsw_compaction_in_progress" => {
                    compaction_in_progress = compaction_in_progress || extract_bool_nonzero(val)
                }
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

            if live_count >= expected && !compaction_in_progress {
                println!(
                    "Indexing complete: {} docs in {:.1}s",
                    live_count,
                    start.elapsed().as_secs_f64()
                );
                return Ok(());
            }

            if start.elapsed().as_secs() > max_wait {
                println!(
                    "Warning: indexing timeout after {}s (hnsw_live_count={}/{}, compaction_in_progress={})",
                    max_wait, live_count, expected, compaction_in_progress
                );
                return Ok(());
            }

            if start.elapsed().as_secs().is_multiple_of(10) && start.elapsed().as_secs() > 0 {
                println!(
                    "Waiting for indexing: {} docs, compaction_in_progress={} ({:.0}s)",
                    live_count,
                    compaction_in_progress,
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

// ── KiviDB filter builder ────────────────────────────────────────────────

/// A KiviDB-specific `FT.SEARCH` prefilter builder.
///
/// This deliberately does NOT reuse redis.rs's `parse_conditions`. That builder
/// is correct for RediSearch/ValkeySearch/Dragonfly and is left untouched; the
/// list below is what was measured against a real `quay.io/kividbio/kividb:v1.0.2-full`
/// server and is why KiviDB needs its own emitter. Every item was reproduced by
/// hand with `FT.SEARCH` against a seeded index before being encoded here.
///
/// 1. **`$param` placeholders are NOT substituted in a filter expression, in
///    ANY query form.** This is the root cause of issue #205. The failure is
///    NOT hybrid-specific: a plain `FT.SEARCH idx "(@color:{$a})" PARAMS 2 a
///    red` returns 0 documents exactly as the hybrid
///    `<prefilter>=>[KNN $K @vector $vec_param ...]` does, and `@n:[$lo +inf]`
///    returns the whole corpus in both forms. (In the hybrid form KiviDB *does*
///    substitute the KNN params `$vec_param`, `$K`, `$EF` — only the filter
///    expression is left as literal text, which is what made this look
///    prefilter-specific at first.) A TAG or TEXT clause therefore matches ZERO
///    documents (`@kw:{$p}` looks for the literal tag `$p`) and a NUMERIC clause
///    silently degrades to match-ALL. That accounts for the `uuid` / `bool` /
///    `fulltext` / `and` / `nested` recall ≈ 0.0 and the `datetime` recall ≈ 0.5
///    measured on `master`. Note it does NOT by itself account for the `or` and
///    `match_any` rows (≈ 0.46): those are dominated by divergences 4 and 9
///    below — a spaced intra-brace TAG-OR reads as match-all, and a redundantly
///    wrapped OR arm degrades the same way — which is why fixing this one alone
///    would not have fixed them. So this builder inlines every value as a
///    LITERAL and binds no filter params at all.
/// 2. **No escaping.** KiviDB TAG values are matched raw. RediSearch-style
///    backslash escaping matches nothing (`@uid:{3f2a\-9c}` → 0 docs, while
///    `@uid:{3f2a-9c}` → the document). Spaces, `-`, `$`, `{`, `}`, `:`, `;`,
///    `.`, `/`, quotes and backslashes all round-trip verbatim, even inside a
///    compound clause.
/// 3. **Four characters are inexpressible in a TAG value** — see
///    [`tag_value_offender`], which carries the measurement. `|` is always read
///    as the TAG-OR separator (under-match), `@` degrades the whole query to
///    match-all (over-match), and `(`/`)` are parsed as grouping the moment the
///    clause sits inside a group (`@t:{zz(zz} @u:{blue}` → 0;
///    `(@t:{zz)zz})` → the whole corpus), which every non-trivial filter does.
///    All four are hard errors here rather than a silent wrong recall.
/// 4. **`|` spacing is inverted between the two positions it appears in.**
///    INSIDE braces it must NOT be spaced: `@f:{a | b}` matches EVERY document
///    (a parser divergence) while `@f:{a|b}` is correct. BETWEEN clauses it
///    MUST be spaced: `@t:{a} | @t:{c}` is correct while the unspaced
///    `@t:{a}|@t:{c}` returns 0. This builder sidesteps the intra-brace form
///    entirely by emitting separate clauses, and [`Node::render`] always joins
///    disjuncts with `" | "`.
/// 5. **Exclusive numeric bounds `(x` are not parsed** — `@f:[-inf (5]`
///    degrades to match-all, so `gt`/`lt` are emulated exactly with
///    `f64::next_up`/`next_down`. That emulation is only sound if KiviDB
///    compares NUMERIC values as `f64`, which was established by two probes
///    that an integer representation could not pass:
///    `@val:[9007199254740993 9007199254740993]` returns **2** — 2^53+1
///    collides with 2^53, which no `i64` comparison would do — and
///    `@val:[1.5 1.5]` returns **1**, so non-integers are stored and compared
///    rather than truncated. (An earlier note cited 16777217 and 1609459201;
///    those only rule out `f32`, since both are equally distinguishable under
///    `i64`, and `next_up`/`next_down` is exact only for `f64`.)
/// 6. **Negation is broken.** A leading `-@f:{v}` is ignored (matches all) and a
///    `-` clause next to a positive one annihilates the result. This builder
///    never emits `-`; a never-match degenerates to a sentinel value instead.
/// 7. **TEXT accepts one alphanumeric term per clause.** A multi-term
///    `@body:(quick brown)` returns ZERO documents, and any non-alphanumeric
///    character in the term makes it miss (`)` makes it match-all). Anything
///    else is a hard error.
/// 8. **No GEO field type**, so a geo condition is a hard error (unlike
///    Chroma/Milvus, which drop it: a dropped sole condition leaves an empty
///    prefilter, and KNN over the whole corpus scored against geo-filtered
///    ground truth is precisely the silent-wrong-result failure this module
///    exists to prevent). This follows vertex.rs, which hard-errors on every
///    filter it cannot express.
/// 9. **Redundant parentheses corrupt the query**, which is why this builder
///    renders an expression TREE with MINIMAL parenthesisation ([`Node`]) rather
///    than wrapping every group the way redis.rs does. Measured:
///    * `((X))` — any doubly-wrapped group — degrades to match-ALL. So
///      `((@c:{red}))`, `((@c:{red} @n:[50 +inf]))` and `((@c:{red} | @c:{blue}))`
///      all match the whole corpus.
///    * In a conjunction, a parenthesised operand may not FOLLOW a bare one:
///      `@c:{red} (@n:[50 +inf])` returns 0 documents, while the same conjuncts
///      as `(@n:[50 +inf]) @c:{red}`, `(@c:{red}) (@n:[50 +inf])` and
///      `@c:{red} @n:[50 +inf]` are all correct. Hence [`Node::render`] emits
///      every parenthesised (OR) conjunct BEFORE the bare ones — sound because
///      conjunction is commutative.
///    * Disjunction is unaffected (`A|B`, `(A)|B`, `A|(B)`, `(A B)|(C D)` all
///      agree), and AND binds tighter than `|`, as in RediSearch.
///
/// One thing that is NOT a divergence: KiviDB genuinely PRE-filters. Measured
/// recall against brute-force filtered ground truth over 2000 docs is 1.000 for
/// keyword / range / AND / OR / nested filters all the way down to 2%
/// selectivity, so the filtered numbers this engine reports are meaningful.
pub(crate) mod kividb_filter {
    use serde_json::{Map, Value};
    use vector_db_benchmark::parsers::datetime_to_epoch_secs;

    /// Sentinel TAG/TEXT value used to express "matches nothing".
    ///
    /// RediSearch's usual trick — the `(@f:{$s} -@f:{$s})` contradiction — is
    /// unusable on KiviDB (divergence 6: for a TEXT field that form matches
    /// ALL documents). A plain lookup for a value no corpus contains was
    /// verified to return 0 documents for both TAG and TEXT.
    const NEVER_MATCH: &str = "__kividb_never_match__";

    /// The prefilter that selects the whole corpus, i.e. an unfiltered kNN.
    /// Returned explicitly by [`parse_conditions`] for conditions that carry no
    /// clause, so callers never have to interpret an `Option`.
    pub const MATCH_ALL: &str = "*";

    /// Largest magnitude an integer bound may have and still be converted to
    /// `f64` losslessly (needed only for the exclusive-bound emulation).
    const MAX_EXACT_INT_F64: i64 = 1 << 53;

    /// A filter expression tree.
    ///
    /// Kept as a tree (rather than eagerly concatenated strings, as redis.rs
    /// does) purely so [`Node::render`] can apply KiviDB's parenthesisation
    /// constraints — divergence 9 in this module's doc comment — which make
    /// redis.rs's "wrap every group" approach silently wrong here.
    #[derive(Debug, Clone, PartialEq)]
    enum Node {
        /// A single `@field:...` clause, already fully rendered.
        Leaf(String),
        /// Conjunction (space-joined).
        And(Vec<Node>),
        /// Disjunction (`|`-joined).
        Or(Vec<Node>),
    }

    /// Build a conjunction, collapsing a one-element group to its only child.
    ///
    /// The collapse matters beyond tidiness: a parent decides whether to
    /// parenthesise a child from the child's KIND, so a single-child `And`
    /// wrapping an `Or` (or vice versa) would earn a paren pair its rendering
    /// does not need — and on KiviDB a redundant wrap is not cosmetic, it
    /// degrades the group to match-all.
    fn node_and(mut children: Vec<Node>) -> Node {
        if children.len() == 1 {
            return children.pop().expect("len checked");
        }
        Node::And(children)
    }

    /// Build a disjunction, collapsing a one-element group. See [`node_and`].
    fn node_or(mut children: Vec<Node>) -> Node {
        if children.len() == 1 {
            return children.pop().expect("len checked");
        }
        Node::Or(children)
    }

    impl Node {
        /// Render WITHOUT an enclosing paren pair, using the minimum
        /// parenthesisation KiviDB parses correctly.
        ///
        /// * Associative children are flattened (`And[And[a,b],c]` → `a b c`),
        ///   and a one-child group collapses to that child — both avoid the
        ///   `((X))` double-wrap that degrades to match-all.
        /// * Only a child of the OPPOSITE kind needs parens, and in a
        ///   conjunction those parenthesised children are emitted FIRST,
        ///   because a parenthesised conjunct following a bare one returns zero
        ///   documents. Conjunction is commutative, so the reorder is sound.
        fn render(&self) -> String {
            match self {
                Node::Leaf(s) => s.clone(),
                Node::And(children) => {
                    let mut flat = Vec::new();
                    flatten(children, true, &mut flat);
                    if flat.len() == 1 {
                        return flat[0].render();
                    }
                    // Parenthesised (disjunction) conjuncts first, bare after.
                    let mut parts: Vec<String> = flat
                        .iter()
                        .filter(|c| matches!(c, Node::Or(_)))
                        .map(|c| format!("({})", c.render()))
                        .collect();
                    parts.extend(
                        flat.iter()
                            .filter(|c| !matches!(c, Node::Or(_)))
                            .map(|c| c.render()),
                    );
                    parts.join(" ")
                }
                Node::Or(children) => {
                    let mut flat = Vec::new();
                    flatten(children, false, &mut flat);
                    if flat.len() == 1 {
                        return flat[0].render();
                    }
                    flat.iter()
                        .map(|c| match c {
                            Node::And(_) => format!("({})", c.render()),
                            other => other.render(),
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                }
            }
        }
    }

    /// Flatten same-kind children one level at a time (`and` into `and`, `or`
    /// into `or`), exploiting associativity so no redundant group survives.
    fn flatten<'a>(nodes: &'a [Node], want_and: bool, out: &mut Vec<&'a Node>) {
        for n in nodes {
            match n {
                Node::And(inner) if want_and => flatten(inner, want_and, out),
                Node::Or(inner) if !want_and => flatten(inner, want_and, out),
                other => out.push(other),
            }
        }
    }

    /// Parse `meta_conditions` JSON into a KiviDB prefilter string.
    ///
    /// Returns the prefilter to splice into the hybrid query. `Err` means the
    /// conditions cannot be expressed FAITHFULLY on KiviDB — the caller must
    /// abort the run rather than report a recall computed against a filter that
    /// was never applied.
    ///
    /// Deliberately `Result<String, _>` and NOT `Result<Option<String>, _>`
    /// (issue #219): the `Option` would keep alive the ambiguity between "there
    /// was no filter" and "there was a filter and it parsed to nothing", which
    /// is exactly the class of bug this module exists to remove. `vertex.rs`'s
    /// `parse_vertex_filter` is the same shape — the no-conditions case is split
    /// structurally instead, here by returning the explicit [`MATCH_ALL`]
    /// prefilter (the caller does the same for an absent condition). "Parsed to
    /// nothing" is NOT unreachable here — `{"and": []}`, `{"or": []}`, a bare
    /// leaf and a non-object all render [`MATCH_ALL`] — so the CALLER maps a
    /// match-all render back to "produced nothing" and lets
    /// `QueryConditions::try_resolve_all` reject it. Keeping that check at the
    /// call site rather than here is what stops the sentinel laundering a
    /// dropped filter into an unfiltered run.
    pub fn parse_conditions(conditions: &Value) -> Result<String, String> {
        let Some(obj) = conditions.as_object() else {
            return Ok(MATCH_ALL.to_string());
        };
        if obj.is_empty() {
            return Ok(MATCH_ALL.to_string());
        }
        Ok(build_group(obj)?
            .map(|node| node.render())
            .unwrap_or_else(|| MATCH_ALL.to_string()))
    }

    /// Build one boolean group (`{and:[...], or:[...]}`). Recursive with
    /// [`build_subfilters`], mirroring the shape of redis.rs's builder so the
    /// same dataset conditions parse identically.
    fn build_group(obj: &Map<String, Value>) -> Result<Option<Node>, String> {
        let and_children = match obj.get("and").and_then(|v| v.as_array()) {
            Some(entries) => build_subfilters(entries)?,
            None => Vec::new(),
        };
        let or_children = match obj.get("or").and_then(|v| v.as_array()) {
            Some(entries) => build_subfilters(entries)?,
            None => Vec::new(),
        };

        let mut parts = Vec::new();
        if !and_children.is_empty() {
            parts.push(node_and(and_children));
        }
        if !or_children.is_empty() {
            parts.push(node_or(or_children));
        }
        if parts.is_empty() {
            return Ok(None);
        }
        // A group carrying BOTH `and` and `or` intersects the two (same as
        // redis.rs, which space-joins them).
        Ok(Some(node_and(parts)))
    }

    /// Build the children of one `and`/`or` array. An entry carrying an
    /// `and`/`or` key is a nested group (recursed); anything else is a
    /// `{field: {op: criteria}}` leaf.
    ///
    /// A malformed entry is an ERROR, not a skip. redis.rs `continue`s past
    /// these, which is the same silent-drop this module exists to eliminate:
    /// skipping one entry of an `and` array widens the prefilter and publishes a
    /// recall for a filter that was only partly applied. No shipped dataset
    /// produces either shape, so this changes no current run.
    fn build_subfilters(entries: &[Value]) -> Result<Vec<Node>, String> {
        let mut children = Vec::new();
        for entry in entries {
            let Some(entry_obj) = entry.as_object() else {
                return Err(format!(
                    "KiviDB filter: filter entry must be an object, got {entry}. Refusing to skip \
                     it — dropping one conjunct would widen the prefilter and report a recall for \
                     a filter that was never fully applied."
                ));
            };
            if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
                if let Some(group) = build_group(entry_obj)? {
                    children.push(group);
                }
                continue;
            }
            for (field_name, field_filters) in entry_obj {
                let Some(filter_obj) = field_filters.as_object() else {
                    return Err(format!(
                        "KiviDB filter: filter for `{field_name}` must be an object, got \
                         {field_filters}. Refusing to skip it — see above."
                    ));
                };
                for (condition_type, criteria) in filter_obj {
                    children.push(build_filter(field_name, condition_type, criteria)?);
                }
            }
        }
        Ok(children)
    }

    /// Build a single leaf clause. Unlike redis.rs's builder — which returns
    /// `None` for anything it does not recognise — every unhandled shape here is
    /// an error, so a condition can never be dropped on the floor and leave a
    /// weaker (or empty) prefilter behind.
    fn build_filter(field: &str, condition_type: &str, criteria: &Value) -> Result<Node, String> {
        match condition_type {
            "match" => {
                // match_any (IN-list) takes precedence over exact {value}, which
                // takes precedence over full-text {text} — same order as redis.rs.
                if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                    build_match_any(field, any)
                } else if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                    build_text(field, text)
                } else {
                    build_exact_match(field, criteria)
                }
            }
            "range" => build_range(field, criteria),
            "geo" => Err(format!(
                "KiviDB filter: cannot filter geo field `{field}` — KiviDB's index schema has no \
                 GEO field type, so the field is never indexed and the clause would match nothing \
                 (verified: a clause over an undeclared field returns 0 documents). Refusing to \
                 report a recall for a geo filter KiviDB cannot apply."
            )),
            other => Err(format!(
                "KiviDB filter: unsupported condition `{other}` on field `{field}`"
            )),
        }
    }

    /// Exact match: string/bool → `@f:{value}` (TAG), number → `@f:[v v]`.
    fn build_exact_match(field: &str, criteria: &Value) -> Result<Node, String> {
        let Some(value) = criteria.get("value") else {
            return Err(format!("KiviDB filter: empty match filter for `{field}`"));
        };

        // Checked before the numeric arms: serde treats JSON `true`/`false` as
        // neither i64 nor f64. Bools are stored as the literal "true"/"false".
        if let Some(b) = value.as_bool() {
            let token = if b { "true" } else { "false" };
            return Ok(Node::Leaf(format!("@{field}:{{{token}}}")));
        }
        if let Some(s) = value.as_str() {
            return tag_clause(field, s);
        }
        if let Some(i) = value.as_i64() {
            return Ok(Node::Leaf(format!("@{field}:[{i} {i}]")));
        }
        if let Some(f) = value.as_f64() {
            let n = fmt_f64(f);
            return Ok(Node::Leaf(format!("@{field}:[{n} {n}]")));
        }
        Err(format!(
            "KiviDB filter: unsupported match value for `{field}`: {value}"
        ))
    }

    /// `match_any` (IN-list), the OR-of-values semantics mirroring qdrant's
    /// `Condition::matches(field, Vec)`.
    ///
    /// Emitted as SEPARATE OR'd clauses — `(@f:{a} | @f:{b})` — rather than
    /// RediSearch's single `@f:{a | b}` clause. That is divergence 4: KiviDB
    /// reads a *spaced* intra-brace `|` as "match everything", so the shared
    /// builder's form silently disables the filter. The separate-clause form was
    /// verified to return exactly the union.
    fn build_match_any(field: &str, any: &[Value]) -> Result<Node, String> {
        // An all-integer list filters a NUMERIC field, so OR single-value ranges.
        if !any.is_empty() && any.iter().all(|v| v.is_i64()) {
            let clauses: Vec<Node> = any
                .iter()
                .filter_map(|v| v.as_i64())
                .map(|i| Node::Leaf(format!("@{field}:[{i} {i}]")))
                .collect();
            return Ok(node_or(clauses));
        }

        let mut clauses = Vec::new();
        for v in any {
            let Some(s) = v.as_str() else {
                return Err(format!(
                    "KiviDB filter: unsupported `match_any` element for `{field}`: {v} \
                     (expected all-integer or all-string)"
                ));
            };
            // Empty strings are invalid TAG syntax and can never match an exact
            // keyword, so they contribute nothing to the union (same as redis.rs).
            if s.is_empty() {
                continue;
            }
            check_tag_value(field, s)?;
            clauses.push(Node::Leaf(format!("@{field}:{{{s}}}")));
        }

        if clauses.is_empty() {
            // An empty IN-set must match NOTHING. Dropping the sole clause would
            // leave no prefilter and run KNN over ALL docs — the inverse filter.
            return Ok(never_match_tag(field));
        }
        Ok(node_or(clauses))
    }

    /// Full-text clause over a TEXT field: `@f:(term)`.
    ///
    /// Divergence 7: KiviDB matches ONE alphanumeric term per clause. A
    /// multi-term query returns zero documents, and a term containing any
    /// non-alphanumeric character misses (a `)` even flips the whole query to
    /// match-all). Both are hard errors — a silent zero-match would be reported
    /// as recall 0.0 as though the corpus genuinely had no hit.
    fn build_text(field: &str, text: &str) -> Result<Node, String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(Node::Leaf(format!("@{field}:({NEVER_MATCH})")));
        }
        if trimmed.split_whitespace().count() > 1 {
            return Err(format!(
                "KiviDB filter: multi-term full-text query {trimmed:?} on `{field}` is not \
                 supported — KiviDB's TEXT matcher handles a single term per `@field:(...)` \
                 clause and returns ZERO documents for a multi-term query (verified live on \
                 v1.0.2), which would be reported as recall 0.0."
            ));
        }
        if !trimmed.chars().all(|c| c.is_alphanumeric()) {
            return Err(format!(
                "KiviDB filter: full-text term {trimmed:?} on `{field}` contains a \
                 non-alphanumeric character. KiviDB's TEXT tokenizer splits on those and the \
                 clause then either misses entirely or (for `)`) degrades to match-all; there is \
                 no escaping mechanism. Refusing to emit a filter that would report a wrong recall."
            ));
        }
        Ok(Node::Leaf(format!("@{field}:({trimmed})")))
    }

    /// Numeric/datetime range. `gt`/`lt` are EXCLUSIVE, `gte`/`lte` inclusive.
    ///
    /// Divergence 5: KiviDB does not parse RediSearch's `(` exclusive-bound
    /// marker — `@f:[-inf (5]` degrades to match-all — so an exclusive bound is
    /// emulated by nudging the inclusive bound to the adjacent `f64`. KiviDB
    /// NUMERIC comparisons are `f64` (proved by the 2^53+1 collision and the
    /// `1.5` probe in divergence 5 above), so this is exact, not an
    /// approximation.
    fn build_range(field: &str, criteria: &Value) -> Result<Node, String> {
        // (key, is_upper_bound, is_exclusive) in a fixed order so the emitted
        // string is deterministic.
        let bounds = [
            ("lt", true, true),
            ("gt", false, true),
            ("lte", true, false),
            ("gte", false, false),
        ];

        let mut clauses = Vec::new();
        for (key, is_upper, exclusive) in bounds {
            let Some(raw) = criteria.get(key) else {
                continue;
            };
            let bound = number_bound(field, key, raw)?;
            let rendered = if exclusive {
                // `< x` becomes `<= prev(x)`; `> x` becomes `>= next(x)`.
                let as_f64 = match bound {
                    // `unsigned_abs`, not `abs`: `i64::MIN.abs()` panics in a
                    // debug build (and is UB-adjacent in release).
                    Bound::Int(i) if i.unsigned_abs() > MAX_EXACT_INT_F64 as u64 => {
                        return Err(format!(
                            "KiviDB filter: exclusive `{key}` bound {i} on `{field}` exceeds \
                             2^53 and cannot be nudged to the adjacent f64 exactly. KiviDB does \
                             not support RediSearch's `(` exclusive-bound syntax (the clause \
                             silently degrades to match-all), so this bound is inexpressible."
                        ));
                    }
                    Bound::Int(i) => i as f64,
                    Bound::Float(f) => f,
                };
                fmt_f64(if is_upper {
                    as_f64.next_down()
                } else {
                    as_f64.next_up()
                })
            } else {
                match bound {
                    Bound::Int(i) => i.to_string(),
                    Bound::Float(f) => fmt_f64(f),
                }
            };
            clauses.push(Node::Leaf(if is_upper {
                format!("@{field}:[-inf {rendered}]")
            } else {
                format!("@{field}:[{rendered} +inf]")
            }));
        }

        if clauses.is_empty() {
            return Err(format!(
                "KiviDB filter: range filter on `{field}` has no usable bound \
                 (expected any of gt/gte/lt/lte)"
            ));
        }
        // A two-bound range is a conjunction of its bounds; the renderer
        // flattens it into an enclosing AND and parenthesises it only when it
        // sits inside an OR.
        Ok(node_and(clauses))
    }

    /// A numeric range bound, keeping integers exact instead of routing every
    /// bound through `f64`.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Bound {
        Int(i64),
        Float(f64),
    }

    /// Parse a JSON range bound: a number, an ISO-8601 datetime (→ epoch
    /// seconds, matching how `upload` stores `datetime` fields), or a numeric
    /// string.
    fn number_bound(field: &str, key: &str, value: &Value) -> Result<Bound, String> {
        if let Some(i) = value.as_i64() {
            return Ok(Bound::Int(i));
        }
        if let Some(f) = value.as_f64() {
            return Ok(Bound::Float(f));
        }
        if let Some(s) = value.as_str() {
            if let Some(epoch) = datetime_to_epoch_secs(s) {
                return Ok(Bound::Int(epoch as i64));
            }
            if let Ok(i) = s.parse::<i64>() {
                return Ok(Bound::Int(i));
            }
            if let Ok(f) = s.parse::<f64>() {
                return Ok(Bound::Float(f));
            }
        }
        Err(format!(
            "KiviDB filter: non-numeric `{key}` range bound on `{field}`: {value}"
        ))
    }

    /// A TAG clause for a literal string value.
    fn tag_clause(field: &str, value: &str) -> Result<Node, String> {
        if value.is_empty() {
            return Ok(never_match_tag(field));
        }
        check_tag_value(field, value)?;
        Ok(Node::Leaf(format!("@{field}:{{{value}}}")))
    }

    /// The first character in `value` that a KiviDB TAG value cannot represent,
    /// if any.
    ///
    /// Measured on v1.0.2 by storing `zz<c>zz` and querying it back for 32
    /// candidate characters. FOUR fail to round-trip, and the probe has to be
    /// run in every clause POSITION this builder emits, not just as a lone bare
    /// clause — the first version of this check tested bare clauses only and
    /// consequently missed the parens:
    /// * `|` — always parsed as the TAG-OR separator, so a clause carrying it
    ///   silently UNDER-matches (0 documents);
    /// * `@` — starts a new field reference, degrading the whole query to
    ///   match-all, so the clause silently OVER-matches (every document);
    /// * `(` and `)` — correct ONLY in a lone bare clause. Measured on a
    ///   200-doc index (35 matching documents):
    ///
    ///   | value        | bare clause | inside a group |
    ///   | ------------ | ----------- | -------------- |
    ///   | `@t:{zz(zz}` | 1 (correct) | 0 (UNDER-match) |
    ///   | `@t:{zz)zz}` | 1 (correct) | 35 (MATCH-ALL) |
    ///
    ///   Every non-trivial rendering puts a TAG clause inside a group — a sole
    ///   `match_any`, a two-clause AND, a `match_any` AND'd with a leaf, any
    ///   nested boolean — so a parenthesised TAG value is silently wrong almost
    ///   everywhere. `build_text` already rejects `)` for the same reason.
    ///
    /// There is no escaping mechanism to fall back on: a backslash escape (the
    /// RediSearch answer) matches nothing on KiviDB.
    ///
    /// # Why this is checked on the QUERY side only
    ///
    /// The obvious worry is the corpus: if a stored value carrying one of these
    /// characters were mis-indexed, no query-side check could see it. Measured
    /// on v1.0.2, it is not — the whole failure lives in the query parser and a
    /// stored occurrence is INERT:
    ///
    /// * A stored `a|b` is **not** split on the separator. With documents
    ///   `a|b`, `a` and `b` in one TAG field, `@t:{a}` returns exactly 1 (the
    ///   `a` document), not 2. TAG values really are atomic on storage, so the
    ///   feared over-match — a stored `a|b` answering an unrelated `@t:{a}` —
    ///   does not happen. The same holds for a stored `@` (`@t:{c}` returns 1
    ///   with `c` and `c@d` both indexed).
    /// * Two identical corpora, one seeded with `Bag (Small)` / `x|y` / `p@q`
    ///   and one entirely clean, return IDENTICAL counts for every clause shape
    ///   this builder emits (bare leaf, ` | ` disjunction, parenthesised
    ///   disjunct AND'd with a leaf, two parenthesised disjuncts).
    ///
    /// So a corpus value carrying one of these is simply unaddressable — the
    /// only query that could match it is one this function rejects — and it
    /// never perturbs any other query's result. Adding an upload-side check
    /// would therefore buy no correctness and would hard-fail a dataset that
    /// benchmarks correctly today. Scanned over the shipped h-and-m corpus and
    /// its shipped queries (13 TAG-typed schema fields):
    ///
    /// | | `(` | `)` | `@` | `|` |
    /// | --- | --- | --- | --- | --- |
    /// | corpus field-values (105,100 docs) | 3,900 | 3,878 | 0 | 0 |
    /// | distinct query filter values (586) | 0 | 0 | 0 | 0 |
    ///
    /// An upload-side check would abort h-and-m outright; the query-side check
    /// it actually needs fires on nothing shipped.
    pub fn tag_value_offender(value: &str) -> Option<char> {
        value.chars().find(|c| matches!(c, '|' | '@' | '(' | ')'))
    }

    /// Reject a QUERY-side TAG value KiviDB cannot represent. See
    /// [`tag_value_offender`] for the measurement and for the corpus-side twin.
    fn check_tag_value(field: &str, value: &str) -> Result<(), String> {
        if let Some(c) = tag_value_offender(value) {
            return Err(format!(
                "KiviDB filter: TAG value {value:?} for field `{field}` contains {c:?}, which \
                 KiviDB's FT.SEARCH cannot express — `|` is always read as the TAG-OR separator \
                 (silent under-match), `@` degrades the whole query to match-all (silent \
                 over-match), and `(`/`)` are parsed as grouping the moment the clause sits \
                 inside a group, which every non-trivial filter does (`(` under-matches, `)` \
                 degrades to match-all). Backslash escaping matches nothing on KiviDB. Refusing \
                 to emit a filter that would report a wrong recall."
            ));
        }
        Ok(())
    }

    /// TAG clause that matches nothing. See [`NEVER_MATCH`].
    fn never_match_tag(field: &str) -> Node {
        Node::Leaf(format!("@{field}:{{{NEVER_MATCH}}}"))
    }

    /// Render an `f64` bound so KiviDB parses back the identical value.
    /// Rust's `Display` for `f64` emits the shortest round-tripping decimal, and
    /// KiviDB parses bounds as `f64`, so the nudged exclusive bounds survive
    /// exactly (verified at epoch scale: `1800000000.0000002`).
    fn fmt_f64(f: f64) -> String {
        format!("{f}")
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        fn q(v: Value) -> String {
            parse_conditions(&v).unwrap()
        }

        fn err(v: Value) -> String {
            parse_conditions(&v).unwrap_err()
        }

        #[test]
        /// The BUILDER renders match-all for these. That is not the same as the
        /// engine accepting them: the search path maps a match-all render back
        /// to "produced nothing", so `{"and": []}` and friends fail the run.
        /// See `kividb_match_all_render_is_rejected_at_the_call_site`.
        fn no_conditions_yields_the_explicit_match_all_prefilter() {
            // Not `Ok(None)` — see `parse_conditions`' doc comment on issue #219.
            // The absence of a filter is expressed as the `*` prefilter, which is
            // exactly what it means on the wire, so no caller can confuse it with
            // a filter that silently parsed away.
            assert_eq!(parse_conditions(&json!({})).unwrap(), MATCH_ALL);
            assert_eq!(parse_conditions(&json!(null)).unwrap(), MATCH_ALL);
            assert_eq!(parse_conditions(&json!({"and": []})).unwrap(), MATCH_ALL);
        }

        #[test]
        fn the_tag_offender_set_is_exactly_pipe_at_and_the_two_parens() {
            // Pins the shared predicate both the query side and the corpus side
            // consult, so the two can never drift apart (the corpus side would
            // otherwise store a value the query side would have rejected).
            for (v, expected) in [
                ("a|b", Some('|')),
                ("a@b", Some('@')),
                ("a(b", Some('(')),
                ("a)b", Some(')')),
                ("plain-value_1.2:3/4", None),
            ] {
                assert_eq!(tag_value_offender(v), expected, "{v}");
            }
            // Everything else was verified live to round-trip raw.
            for c in "-_.:/\\+*?~^$#%&[]{}<>=!,;'\"` ".chars() {
                assert_eq!(
                    tag_value_offender(&format!("zz{c}zz")),
                    None,
                    "{c:?} must round-trip through a KiviDB TAG value"
                );
            }
        }

        #[test]
        fn values_are_inlined_as_literals_never_as_params() {
            // The whole point of this builder: KiviDB does not substitute
            // `$param` inside a hybrid query's prefilter, so no `$` may appear.
            let out = q(json!({"and": [{"color": {"match": {"value": "red"}}}]}));
            assert_eq!(out, "@color:{red}");
            assert!(!out.contains('$'));
        }

        #[test]
        fn uuid_hyphens_are_not_escaped() {
            // Regression guard for #205: RediSearch would escape the hyphens,
            // and the escaped form matches NOTHING on KiviDB.
            let uuid = "550e8400-e29b-41d4-a716-446655440000";
            let out = q(json!({"and": [{"uid": {"match": {"value": uuid}}}]}));
            assert_eq!(out, format!("@uid:{{{uuid}}}"));
            assert!(!out.contains('\\'));
        }

        #[test]
        fn bool_matches_the_literal_token() {
            assert_eq!(
                q(json!({"and": [{"flag": {"match": {"value": true}}}]})),
                "@flag:{true}"
            );
            assert_eq!(
                q(json!({"and": [{"flag": {"match": {"value": false}}}]})),
                "@flag:{false}"
            );
        }

        #[test]
        fn match_any_uses_separate_or_clauses_not_a_spaced_intra_brace_or() {
            // `@c:{red | blue}` matches EVERY document on KiviDB, so the shared
            // builder's single-clause form must never be emitted here.
            let out = q(json!({"and": [{"c": {"match": {"any": ["red", "blue"]}}}]}));
            assert_eq!(out, "@c:{red} | @c:{blue}");
            assert!(!out.contains("{red | blue}"));
        }

        #[test]
        fn match_any_all_ints_uses_numeric_ors() {
            assert_eq!(
                q(json!({"and": [{"n": {"match": {"any": [1, 2]}}}]})),
                "@n:[1 1] | @n:[2 2]"
            );
        }

        #[test]
        fn empty_match_any_matches_nothing_rather_than_everything() {
            let out = q(json!({"and": [{"c": {"match": {"any": []}}}]}));
            assert_eq!(out, "@c:{__kividb_never_match__}");
            // Must NOT collapse to an empty prefilter (which would run KNN over
            // the whole corpus — the inverse of an empty IN-set).
            assert!(out.contains("never_match"));
        }

        #[test]
        fn inclusive_range_bounds_are_verbatim() {
            assert_eq!(
                q(json!({"and": [{"n": {"range": {"gte": 50}}}]})),
                "@n:[50 +inf]"
            );
            assert_eq!(
                q(json!({"and": [{"n": {"range": {"lte": 50}}}]})),
                "@n:[-inf 50]"
            );
        }

        #[test]
        fn exclusive_range_bounds_are_nudged_never_parenthesised() {
            // KiviDB does not parse `(` exclusive bounds — the clause degrades
            // to match-all — so `lt`/`gt` must be emulated on the adjacent f64.
            let lt = q(json!({"and": [{"n": {"range": {"lt": 200}}}]}));
            assert_eq!(lt, "@n:[-inf 199.99999999999997]");
            assert!(!lt.contains("[-inf (200"));
            let gt = q(json!({"and": [{"n": {"range": {"gt": 200}}}]}));
            assert_eq!(gt, "@n:[200.00000000000003 +inf]");
            assert!(!gt.contains("[(200"));
        }

        #[test]
        fn exclusive_nudge_is_exact_at_epoch_scale() {
            // f64 has the headroom for epoch seconds, so `lt 1800000000` must
            // still exclude exactly 1800000000 and keep 1799999999.
            let out = q(json!({"and": [{"ts": {"range": {"lt": 1_800_000_000}}}]}));
            assert_eq!(out, "@ts:[-inf 1799999999.9999998]");
            let bound: f64 = "1799999999.9999998".parse().unwrap();
            assert!(1_799_999_999.0_f64 <= bound);
            assert!(1_800_000_000.0_f64 > bound);
        }

        #[test]
        fn datetime_range_bounds_become_epoch_seconds() {
            let out = q(json!({"and": [{"ts": {"range": {
                "gte": "2021-01-01T00:00:00Z",
                "lt": "2021-01-02T00:00:00Z",
            }}}]}));
            // `lt` is emitted before `gte` (fixed bound order), flattened into
            // the enclosing conjunction with no redundant parens.
            assert_eq!(out, "@ts:[-inf 1609545599.9999998] @ts:[1609459200 +inf]");
        }

        #[test]
        fn two_bound_range_is_parenthesised_so_an_enclosing_or_cannot_split_it() {
            let out = q(json!({"or": [
                {"c": {"match": {"value": "red"}}},
                {"n": {"range": {"gte": 1, "lte": 9}}},
            ]}));
            assert_eq!(out, "@c:{red} | (@n:[-inf 9] @n:[1 +inf])");
        }

        #[test]
        fn and_or_and_nested_groups_compose() {
            assert_eq!(
                q(json!({"and": [
                    {"c": {"match": {"value": "red"}}},
                    {"n": {"range": {"gte": 50}}},
                ]})),
                "@c:{red} @n:[50 +inf]"
            );
            assert_eq!(
                q(json!({"or": [
                    {"c": {"match": {"value": "red"}}},
                    {"n": {"range": {"gte": 90}}},
                ]})),
                "@c:{red} | @n:[90 +inf]"
            );
            assert_eq!(
                q(json!({"or": [
                    {"and": [
                        {"c": {"match": {"value": "red"}}},
                        {"n": {"range": {"gte": 50}}},
                    ]},
                    {"and": [
                        {"c": {"match": {"value": "blue"}}},
                        {"n": {"range": {"lt": 10}}},
                    ]},
                ]})),
                "(@c:{red} @n:[50 +inf]) | (@c:{blue} @n:[-inf 9.999999999999998])"
            );
        }

        #[test]
        fn malformed_entries_are_errors_never_silently_skipped() {
            // redis.rs `continue`s past both of these. Skipping one conjunct of
            // an `and` widens the prefilter, which is the silent drop this
            // module exists to eliminate.
            let e = err(json!({"and": [{"c": {"match": {"value": "red"}}}, "not-an-object"]}));
            assert!(e.contains("must be an object"), "{e}");
            let e = err(json!({"and": [{"c": "not-an-object"}]}));
            assert!(e.contains("must be an object"), "{e}");
        }

        #[test]
        fn extreme_integer_bounds_do_not_panic() {
            // `i64::MIN.abs()` panics in a debug build, so the 2^53 guard uses
            // `unsigned_abs`. Both extremes must surface as the inexpressible
            // -bound error, not as an abort.
            for v in [i64::MIN, i64::MAX] {
                let e = err(json!({"and": [{"n": {"range": {"gt": v}}}]}));
                assert!(e.contains("2^53"), "{v}: {e}");
            }
            // An in-range bound still works.
            assert_eq!(
                q(json!({"and": [{"n": {"range": {"gt": 5}}}]})),
                "@n:[5.000000000000001 +inf]"
            );
        }

        #[test]
        fn fulltext_single_term_is_emitted_bare() {
            assert_eq!(
                q(json!({"and": [{"body": {"match": {"text": "quick"}}}]})),
                "@body:(quick)"
            );
        }

        #[test]
        fn fulltext_multi_term_is_a_hard_error_not_a_silent_zero_match() {
            let e = err(json!({"and": [{"body": {"match": {"text": "quick brown"}}}]}));
            assert!(e.contains("multi-term"), "{e}");
        }

        #[test]
        fn fulltext_non_alphanumeric_term_is_a_hard_error() {
            let e = err(json!({"and": [{"body": {"match": {"text": "co-op"}}}]}));
            assert!(e.contains("non-alphanumeric"), "{e}");
        }

        #[test]
        fn tag_values_with_pipe_at_or_parens_are_hard_errors() {
            // `|` under-matches, `@` over-matches, and `(`/`)` are read as
            // grouping the moment the clause sits inside a group (which every
            // non-trivial rendering does) — all silently, so none may be
            // emitted. The parens were missed by the first probe because it
            // only tested a LONE BARE clause, the one position where they work.
            for (v, needle) in [
                ("a|b", "'|'"),
                ("a@b", "'@'"),
                ("a(b", "'('"),
                ("a)b", "')'"),
                ("Bag (Small)", "'('"),
            ] {
                let e = err(json!({"and": [{"c": {"match": {"value": v}}}]}));
                assert!(e.contains(needle), "{v}: {e}");
            }
            let e = err(json!({"and": [{"c": {"match": {"any": ["ok", "a|b"]}}}]}));
            assert!(e.contains("'|'"), "{e}");
            let e = err(json!({"and": [{"c": {"match": {"any": ["ok", "a)b"]}}}]}));
            assert!(e.contains("')'"), "{e}");
        }

        #[test]
        fn tag_values_with_other_specials_round_trip_unescaped() {
            // Verified live: spaces, hyphens, braces, `$`, `:`, `;`, `.` and `/`
            // all match raw, so they must NOT be rejected or escaped.
            for v in [
                "new york", "co-op", "x{y}z", "$a", "a:b", "a;b", "a.b", "a/b",
            ] {
                let out = q(json!({"and": [{"c": {"match": {"value": v}}}]}));
                assert_eq!(out, format!("@c:{{{v}}}"));
            }
        }

        #[test]
        fn geo_is_a_hard_error_not_a_silent_drop() {
            let e = err(json!({"and": [{"loc": {"geo": {"lat": 1, "lon": 2, "radius": 5}}}]}));
            assert!(e.contains("geo"), "{e}");
            assert!(e.contains("GEO field type"), "{e}");
        }

        #[test]
        fn unknown_condition_and_bad_values_are_hard_errors() {
            assert!(err(json!({"and": [{"c": {"wat": {"value": 1}}}]})).contains("unsupported"));
            assert!(err(json!({"and": [{"c": {"match": {}}}]})).contains("empty match"));
            assert!(err(json!({"and": [{"c": {"range": {}}}]})).contains("no usable bound"));
            assert!(err(json!({"and": [{"n": {"range": {"gte": "abc"}}}]})).contains("non-numeric"));
            assert!(
                err(json!({"and": [{"c": {"match": {"any": ["a", 1]}}}]})).contains("match_any")
            );
        }

        /// True when `s` contains a group whose entire body is itself one group
        /// — i.e. the `((X))` shape that degrades to match-all on KiviDB.
        /// Brace-bearing TAG values (`{x{y}z}`) are irrelevant here: only round
        /// brackets are scanned.
        fn has_redundant_wrap(s: &str) -> bool {
            let b = s.as_bytes();
            for (i, _) in b.iter().enumerate().filter(|(_, c)| **c == b'(') {
                // Find this group's matching close paren.
                let mut depth = 0usize;
                let mut close = None;
                for (j, c) in b.iter().enumerate().skip(i) {
                    match c {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                close = Some(j);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let Some(close) = close else { continue };
                let inner = s[i + 1..close].trim();
                // Redundant when the body is exactly one balanced group.
                if inner.starts_with('(') && inner.ends_with(')') {
                    let mut d = 0usize;
                    let mut spans_all = true;
                    for (k, c) in inner.bytes().enumerate() {
                        match c {
                            b'(' => d += 1,
                            b')' => {
                                d -= 1;
                                if d == 0 && k + 1 != inner.len() {
                                    spans_all = false;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    if spans_all {
                        return true;
                    }
                }
            }
            false
        }

        #[test]
        fn no_redundant_parentheses_are_ever_emitted() {
            // Divergence 9: `((X))` degrades to match-ALL on KiviDB, so no
            // rendering may double-wrap a group.
            for cond in [
                json!({"and": [{"c": {"match": {"value": "red"}}}]}),
                json!({"and": [{"c": {"match": {"any": ["red", "blue"]}}}]}),
                json!({"and": [{"n": {"range": {"gte": 1, "lte": 9}}}]}),
                json!({"or": [{"and": [{"c": {"match": {"value": "red"}}}]}]}),
                json!({"and": [{"and": [{"and": [{"c": {"match": {"value": "red"}}}]}]}]}),
                json!({"or": [{"and": [{"n": {"range": {"gte": 1, "lte": 9}}}]}]}),
                json!({"or": [
                    {"and": [{"c": {"match": {"value": "red"}}}, {"n": {"range": {"gte": 50}}}]},
                    {"and": [{"c": {"match": {"value": "blue"}}}, {"n": {"range": {"lt": 10}}}]},
                ]}),
            ] {
                let out = q(cond);
                assert!(!has_redundant_wrap(&out), "redundant parens: {out}");
            }
            // Sanity-check the detector itself.
            assert!(has_redundant_wrap("((@c:{red}))"));
            assert!(has_redundant_wrap("((@a:{x} @b:{y}))"));
            assert!(!has_redundant_wrap("((@a:{x} | @b:{y}) @c:{z}) | @d:{w}"));
            assert!(!has_redundant_wrap("(@a:{x} @b:{y}) | (@c:{z} @d:{w})"));
        }

        #[test]
        fn parenthesised_conjuncts_are_emitted_before_bare_ones() {
            // Divergence 9: `A (B)` returns ZERO documents on KiviDB, while
            // `(B) A` is correct. A `match_any` (an OR, so parenthesised) AND'ed
            // with a plain leaf must therefore put the OR first, whichever order
            // the condition JSON lists them in.
            assert_eq!(
                q(json!({"and": [
                    {"c": {"match": {"value": "red"}}},
                    {"k": {"match": {"any": ["x", "y"]}}},
                ]})),
                "(@k:{x} | @k:{y}) @c:{red}"
            );
            assert_eq!(
                q(json!({"and": [
                    {"k": {"match": {"any": ["x", "y"]}}},
                    {"c": {"match": {"value": "red"}}},
                ]})),
                "(@k:{x} | @k:{y}) @c:{red}"
            );
        }

        #[test]
        fn nested_group_inside_a_conjunction_still_leads() {
            // Three-level: `or[ and[ leaf, or[..] ], leaf ]`. The inner OR must
            // lead its conjunction, and the conjunction gets exactly one paren
            // pair as an OR arm.
            let out = q(json!({"or": [
                {"and": [
                    {"c": {"match": {"value": "red"}}},
                    {"n": {"match": {"any": [1, 2]}}},
                ]},
                {"c": {"match": {"value": "blue"}}},
            ]}));
            assert_eq!(out, "((@n:[1 1] | @n:[2 2]) @c:{red}) | @c:{blue}");
        }

        #[test]
        fn no_emitted_clause_ever_uses_negation() {
            // Divergence 6: `-` is broken on KiviDB (a leading `-` is ignored,
            // and a `-` beside a positive clause annihilates the result), so no
            // code path may emit one — including the never-match degenerates.
            for cond in [
                json!({"and": [{"c": {"match": {"any": []}}}]}),
                json!({"and": [{"c": {"match": {"value": ""}}}]}),
                json!({"and": [{"body": {"match": {"text": "  "}}}]}),
            ] {
                let out = q(cond);
                assert!(!out.contains(" -@"), "must not emit negation: {out}");
            }
        }
    }
}

/// Encode a vector to the FLOAT32 little-endian blob KiviDB expects.
/// KiviDB's FT.CREATE supports ONLY float32, so this is the single encoding
/// used for both upload and query vectors.
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

        // Metadata fields for filtering (mirrors redis.rs/dragonfly.rs): bools
        // stay the reader's "true"/"false" string (TAG match); datetime
        // strings become epoch seconds (NUMERIC range); numbers/labels map as
        // redis does. No geo case: KiviDB has no Geo field type.
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
                        // Stored verbatim, with NO corpus-side counterpart to
                        // `kividb_filter::check_tag_value`. That asymmetry is
                        // deliberate and measured — see `tag_value_offender`:
                        // the `|`/`@`/`(`/`)` failure lives entirely in the
                        // QUERY parser, and a stored occurrence is inert.
                        hset_cmd.arg(k.as_str()).arg(stored);
                    }
                    MetadataValue::Int(n) => {
                        hset_cmd.arg(k.as_str()).arg(n.to_string());
                    }
                    MetadataValue::Float(f) => {
                        hset_cmd.arg(k.as_str()).arg(f.to_string());
                    }
                    MetadataValue::Labels(labels) => {
                        // A DECLARED multi-valued field never reaches this
                        // point: `validate_dataset_support` rejects the dataset
                        // in `configure()`, before the payload is even read,
                        // because a KiviDB TAG value is atomic and could only
                        // match the whole joined string.
                        //
                        // An UNDECLARED labels field was never filterable on any
                        // engine, so it is still stored joined — the HSET
                        // succeeds and a pure-KNN run over such a dataset keeps
                        // working, exactly as before.
                        hset_cmd.arg(k.as_str()).arg(labels.join(";"));
                    }
                    MetadataValue::Geo { lon, lat } => {
                        // No Geo field type exists to index this against; stored
                        // as a plain string so the HSET at least succeeds, but it
                        // is never declared in the schema and never filterable.
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

/// Reject — BEFORE anything is created, read or uploaded — the dataset schema
/// shapes KiviDB cannot index faithfully.
///
/// Both rejections are decidable from `dataset.config.schema` alone, so they
/// belong here and not deeper in the run. Doing them late is not merely slower,
/// it is actively harmful:
///
/// * A geo condition can only be seen in `search()`, i.e. AFTER the whole corpus
///   is uploaded (1M vectors for `random-geo-radius-100-angular-filters`) AND
///   after `experiment.rs` has already written the upload result file. That
///   orphan file then satisfies `--skip-if-exists`, so a later `--skip-upload`
///   re-run skips the pair silently instead of reporting it. `engine.delete()`
///   is also never reached on the error path, leaving the keyspace populated.
/// * A multi-valued `labels` array can only be seen mid-`upload()`, i.e. after
///   parsing a 3.6 GB payload file for `arxiv-titles-384-angular-filters`.
///
/// Failing in `configure()` makes both aborts instant and leaves nothing behind.
///
/// Exactly three shipped datasets are affected: the two
/// `random-geo-radius-*-angular-filters` (geo) and
/// `arxiv-titles-384-angular-filters` (multi-valued `labels`).
fn validate_dataset_support(dataset: &Dataset) -> Result<(), String> {
    let Some(schema) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) else {
        return Ok(());
    };
    let dataset_name = &dataset.config.name;
    for (field_name, field_type) in schema {
        match field_type.as_str().unwrap_or("") {
            "geo" => {
                return Err(format!(
                    "KiviDB cannot benchmark dataset `{dataset_name}`: its schema declares geo \
                     field `{field_name}`, and KiviDB's index schema has no GEO field type at all \
                     (its `FieldType` has no `Geo` variant, unlike Dragonfly, which has the type \
                     but rejects this tool's `$param` geo bounds at the query parser). The field \
                     can never be indexed, so a geo condition matches zero documents and the run \
                     would publish recall 0.0 for a filter that was never applied. Per-run \
                     remedies: drop this dataset from `--datasets` (the geo datasets are \
                     `random-geo-radius-100-angular-filters` and \
                     `random-geo-radius-2048-angular-filters`), or keep the sweep going past it \
                     with `--exit-on-error false`. Do NOT edit the field out of \
                     `datasets/datasets.json` — that file is shared, and qdrant, milvus, weaviate, \
                     pgvector and elasticsearch all filter this field natively."
                ));
            }
            // Only `keyword` reaches `create_index`'s TAG arm, which is where the
            // atomic-TAG limitation below bites.
            "keyword" if is_multivalued_keyword_field(field_name) => {
                return Err(format!(
                    "KiviDB cannot benchmark dataset `{dataset_name}`: its schema declares the \
                     multi-valued keyword field `{field_name}`, and a KiviDB TAG value is ATOMIC \
                     — `FT.CREATE` rejects the `SEPARATOR` modifier outright and `@{field_name}:\
                     {{b}}` does not match a stored `a;b;c`. A `match_any` over `{field_name}` \
                     could therefore only ever match the whole joined string, i.e. zero documents, \
                     and the run would publish recall 0.0 for a filter that was never applied. \
                     Per-run remedies: drop this dataset from `--datasets` and use its unfiltered \
                     twin `arxiv-titles-384-angular-no-filters` for a pure-KNN KiviDB number, or \
                     keep the sweep going past it with `--exit-on-error false`. Do NOT edit the \
                     field out of `datasets/datasets.json` — that file is shared, and qdrant, \
                     milvus, weaviate, pgvector and elasticsearch all filter this field natively."
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Map a dataset distance name to KiviDB's `DISTANCE_METRIC` value.
/// Unknown metrics default to `COSINE`. A typo here (e.g. IP->L2) would
/// silently invert ranking, so it is unit-tested.
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
/// redis.rs/dragonfly.rs.
fn uses_ef_runtime(algorithm: &str) -> bool {
    algorithm.eq_ignore_ascii_case("hnsw")
}

/// Build the FT.SEARCH KNN query string (unfiltered `*` prefilter).
///
/// Pure client-side string formatting, kept OUT of the per-query timed window
/// (precomputed once before the parallel region). `EF_RUNTIME $EF` is emitted
/// only for an HNSW index — a per-query attribute FLAT rejects; without it
/// every `ef` in the search sweep runs at the index default. The query vector
/// is bound as `$vec_param`, so this string is identical across all queries.
fn build_knn_query_str(algorithm: &str, prefilter: &str) -> String {
    if uses_ef_runtime(algorithm) {
        format!("{prefilter}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
    } else {
        format!("{prefilter}=>[KNN $K @vector $vec_param AS vector_score]")
    }
}

/// Execute a KiviDB FT.SEARCH KNN query, return (id, score) pairs.
///
/// `vec_bytes` and `query_str` are precomputed by the caller BEFORE the timed
/// window; this performs only the arg binding, the `cmd.query` RPC round-trip,
/// and the reply parse. Any metadata prefilter is already inlined as literals
/// into `query_str` — KiviDB does not substitute `$param` placeholders inside a
/// hybrid query's prefilter, so no filter params are bound here (see
/// [`kividb_filter`]).
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
) -> Result<Vec<(i64, f64)>, String> {
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
        .arg(2)
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Params: vec_param(2) + K(2), plus EF(2) only for HNSW (EF_RUNTIME is
    // HNSW-only; binding it on a FLAT index would be a syntax error). NO filter
    // params — the prefilter carries literals, not `$name` placeholders.
    let ef_runtime = uses_ef_runtime(algorithm);
    let n = 4 + if ef_runtime { 2 } else { 0 };
    cmd.arg("PARAMS").arg(n);
    cmd.arg("vec_param").arg(vec_bytes);
    cmd.arg("K").arg(top.to_string());
    if ef_runtime {
        cmd.arg("EF").arg(ef.to_string());
    }

    let response: redis::Value = cmd
        .query(conn)
        .map_err(|e| format!("FT.SEARCH error: {}", e))?;

    parse_ft_search_response(&response)
}

/// Parse an FT.SEARCH reply under EITHER protocol shape (KiviDB is RESP2-only
/// in practice, but the parser mirrors dragonfly.rs/redis.rs for consistency):
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
        // The reply carries the doc KEY ("<config>:<id>"); recover the
        // trailing numeric id. Missing string -> 0 (positionally present).
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
/// `doc_key_to_id` to strip the per-config key prefix, so this is test-only.
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
impl KividbEngine {
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

impl Engine for KividbEngine {
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
        // FIRST, before a connection is opened or an index dropped/created: a
        // dataset whose schema KiviDB cannot index faithfully aborts here rather
        // than after the corpus has been uploaded (geo) or parsed (labels). See
        // `validate_dataset_support`.
        validate_dataset_support(dataset)?;

        let mut conn = self.get_connection()?;

        println!(
            "Using algorithm {} with config {{'M': {}, 'EF_CONSTRUCTION': {}}}",
            self.config.algorithm, self.config.m, self.config.ef_construction
        );

        self.create_index(&mut conn, dataset)?;
        // Best-effort; see check_commandstats note in upload()/search() below.
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
        // redis/dragonfly/valkey) so range filters over them work.
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
        // comparability (mirrors redis/dragonfly/valkey) — in KiviDB's case
        // this wait should be near-zero; see wait_for_indexing's doc comment.
        let expected = vectors.len();
        let index_start = Instant::now();
        self.wait_for_indexing(expected)?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

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

        // Repeated for the `--skip-upload` path, which reaches `search()`
        // without ever calling `configure()`. Schema-only, so it costs nothing.
        validate_dataset_support(dataset)?;

        // Index-existence guard: on the --skip-upload path a missing or
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
        let query_timeout: i64 = std::env::var("KIVIDB_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        // Per-query prefilters, built by the KiviDB-specific builder (NOT
        // redis.rs's): values are inlined as literals because KiviDB does not
        // substitute `$param` placeholders inside a hybrid query's prefilter.
        // A condition KiviDB cannot express faithfully aborts the run here — a
        // dropped clause would publish a recall for a filter never applied.
        //
        // The absent-condition case is split out HERE rather than folded into an
        // `Option` return (issue #219) — see `kividb_filter::parse_conditions`.
        // A declared filter that renders the match-all prefilter IS the #219
        // drop, so it is mapped to `None` and `try_resolve_all` rejects it;
        // only a genuinely absent condition reaches `MATCH_ALL` here.
        let prefilters: Vec<String> = conditions
            .try_resolve_all("KiviDB", |v| {
                let rendered = kividb_filter::parse_conditions(v)?;
                Ok((rendered != kividb_filter::MATCH_ALL).then_some(rendered))
            })?
            .into_iter()
            .map(|f| {
                f.into_inner()
                    .unwrap_or_else(|| kividb_filter::MATCH_ALL.to_string())
            })
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute client-side request construction BEFORE the timed region so
        // the per-query window wraps ONLY the RPC round-trip + reply parse
        // (matching redis.rs/dragonfly.rs/valkey.rs).
        let encoded_queries: Vec<Vec<u8>> =
            queries.iter().map(|q| encode_query_vector(q)).collect();
        let algorithm = self.config.algorithm.clone();
        let query_strs: Vec<String> = prefilters
            .iter()
            .map(|prefilter| build_knn_query_str(&algorithm, prefilter))
            .collect();
        let index_name = self.config.index_name.clone();

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
            let mut pool = WorkerPool::new(s, "kividb-search", parallel);
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let neighbors = &neighbors;
                let encoded_queries = &encoded_queries;
                let query_strs = &query_strs;
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

                    let mut conn = match KividbEngine::connect(&host, port) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("kividb-search worker setup failed: {e}"));
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this connection with ONE discarded query (index 0) so
                    // the cold first round-trip is not inside the measured window.
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
                        );
                    }

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
        let total_time = measured_start.elapsed().as_secs_f64();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

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
        redis_utils::drop_index_and_keys(
            &mut conn,
            &self.config.index_name,
            &self.config.key_prefix,
        );
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

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

    #[test]
    fn wait_for_indexing_reads_kividb_specific_fields_not_num_docs() {
        // Regression guard for the bug this engine exists to fix: a generic
        // wait-for-indexing loop that only understands `num_docs` would stall
        // forever against KiviDB's FT.INFO, which never sets that field.
        // hnsw_live_count/hnsw_compaction_in_progress must be the fields read.
        let info = Value::Array(vec![
            bulk("hnsw_live_count"),
            bulk("1000"),
            bulk("hnsw_compaction_in_progress"),
            bulk("0"),
        ]);
        let mut live_count: usize = 0;
        let mut compaction_in_progress = false;
        if let Value::Array(arr) = &info {
            for i in (0..arr.len()).step_by(2) {
                if let Value::BulkString(k) = &arr[i] {
                    let key = String::from_utf8_lossy(k);
                    if let Some(v) = arr.get(i + 1) {
                        if key == "hnsw_live_count" {
                            live_count = value_as_i64(v) as usize;
                        } else if key == "hnsw_compaction_in_progress" {
                            compaction_in_progress = value_as_i64(v) != 0;
                        }
                    }
                }
            }
        }
        assert_eq!(live_count, 1000);
        assert!(!compaction_in_progress);
    }
}
