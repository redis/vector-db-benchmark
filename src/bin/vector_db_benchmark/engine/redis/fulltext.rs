//! Redis arm of the ranked full-text workload (#339). I/O only: the query,
//! the reply contract, the index checks and the result metadata all come from
//! `vector_db_benchmark::fulltext`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use super::{redis_value_to_json, RedisEngine};
use crate::config::SearchParams;
use crate::dataset::Dataset;
use crate::engine::{redis_utils, SearchResults};
use vector_db_benchmark::duration_mode::{Dispatcher, Samples};
use vector_db_benchmark::fulltext;
use vector_db_benchmark::results_dump::{RawHit, RawQueryResult};
use vector_db_benchmark::start_gate::WorkerPool;

pub(super) fn resp3_connection(
    url: &str,
    cluster_api: bool,
) -> redis::RedisResult<vector_db_benchmark::redis_transport::Connection> {
    vector_db_benchmark::redis_transport::Client::open(url, cluster_api)?
        .with_protocol(redis::ProtocolVersion::RESP3)
        .get_connection()
}

type WorkerOut = (Samples, Vec<String>, Vec<RawQueryResult>, Option<String>);

impl RedisEngine {
    /// Whether this engine config runs the fulltext workload, i.e. whether
    /// `FT.CREATE` must pin the analyzer settings (`fulltext::redis_stopword_args`).
    /// The index profile this engine config's fulltext search configs need
    /// (`None` when it has none). Same resolution as the startup check.
    pub(super) fn fulltext_profile(&self) -> Result<Option<fulltext::IndexProfile>, String> {
        let profile = fulltext::index_profile(self.search_params.iter().map(|sp| {
            (
                sp.workload
                    .unwrap_or(vector_db_benchmark::workload::Workload::Vector),
                sp.knob(fulltext::SPEC_KEY),
            )
        }))?;
        fulltext::check_redis_shared_prefix(
            "redis",
            profile.as_ref(),
            self.config
                .shared_corpus
                .then_some(self.config.key_prefix.as_str()),
        )?;
        Ok(profile)
    }

    /// Record the profile next to the index (clearing a stale one), so a
    /// `--skip-upload` reuse can verify how the corpus was uploaded. Nothing
    /// at all for a config without a fulltext profile.
    pub(super) fn write_fulltext_marker(
        &self,
        conn: &mut vector_db_benchmark::redis_transport::Connection,
        profile: Option<&fulltext::IndexProfile>,
    ) -> Result<(), String> {
        for args in fulltext::redis_marker_create_commands(&self.config.index_name, profile) {
            let mut cmd = redis::cmd(&args[0]);
            cmd.arg(&args[1..]);
            if let Err(e) = cmd.query::<()>(conn) {
                // English: non-fatal (the result records index_marker: null);
                // parity: fatal, naming the key pattern it needs.
                let p = profile.expect("marker commands exist only with a profile");
                eprintln!(
                    "{}",
                    fulltext::redis_marker_failure(p, "write", &e.to_string())?
                );
                break;
            }
        }
        Ok(())
    }

    /// Parity mode: newlines → spaces in the indexed text fields.
    pub(super) fn normalize_fulltext_upload(
        &self,
        metadata: &mut [Option<vector_db_benchmark::readers::metadata::MetadataItem>],
    ) -> Result<(), String> {
        if let Some(p) = self.fulltext_profile()? {
            if p.normalizes_whitespace() {
                let n = fulltext::normalize_metadata(metadata, &p.text_fields)?;
                println!(
                    "\tfulltext parity: normalized whitespace in {n} values of {:?}",
                    p.text_fields
                );
            }
        }
        Ok(())
    }

    /// The live index serves the ranked BM25 query as pinned: every searched
    /// field indexed as `profile` says (FT.INFO), the recorded marker matches,
    /// and the server's minimum stem length is the default. Refused before
    /// any timed window. Returns `(redis_min_stem_len record, marker read)`.
    /// Shared by the fulltext and hybrid-rrf arms.
    pub(super) fn validate_fulltext_live_index(
        &self,
        dataset: &Dataset,
        spec: &fulltext::FulltextSpec,
        fields: &[fulltext::FieldBoost],
        profile: &fulltext::IndexProfile,
    ) -> Result<(serde_json::Value, serde_json::Value), String> {
        let mut conn = resp3_connection(&self.redis_url, self.config.cluster_api)
            .map_err(|e| format!("fulltext: Redis RESP3 connection failed: {e}"))?;
        let info: redis::Value = redis::cmd("FT.INFO")
            .arg(&self.config.index_name)
            .query(&mut conn)
            .map_err(|e| format!("fulltext: FT.INFO failed: {e}"))?;
        let marker: Option<String> = match redis::cmd("GET")
            .arg(fulltext::redis_marker_key(&self.config.index_name))
            .query(&mut conn)
        {
            Ok(m) => m,
            Err(e) => {
                // English: read as "no marker"; parity: refused.
                eprintln!(
                    "{}",
                    fulltext::redis_marker_failure(profile, "read", &e.to_string())?
                );
                None
            }
        };
        let marker = match marker {
            None => serde_json::Value::Null,
            Some(m) => serde_json::from_str(&m)
                .map_err(|e| format!("fulltext: unreadable index marker {m:?}: {e}"))?,
        };
        if self.config.skip_vector_index || spec.compound.is_some() {
            let validate = if self.config.skip_vector_index {
                vector_db_benchmark::filter_only::validate_redis_schema
            } else {
                vector_db_benchmark::filter_only::validate_redis_full_schema
            };
            let schema = dataset
                .config
                .schema
                .as_ref()
                .map(|s| profile.scalar_indexing.project(s));
            validate(&redis_value_to_json(&info), schema.as_ref())?;
            if profile.scalar_indexing.is_reduced() {
                vector_db_benchmark::text_tag_profile::validate_redis(&redis_value_to_json(&info))?;
            }
        }
        fulltext::validate_redis_index_for(&redis_value_to_json(&info), fields, profile, &marker)?;
        // Server-wide, not in FT.INFO: which tokens get stemmed at all.
        let mut read = |cmd: &mut redis::Cmd, name: &str| {
            cmd.query::<redis::Value>(&mut conn)
                .ok()
                .and_then(|v| fulltext::config_get_value(&redis_value_to_json(&v), name))
        };
        let value = read(
            redis::cmd("CONFIG").arg("GET").arg("search-min-stem-len"),
            "search-min-stem-len",
        )
        .or_else(|| {
            read(
                redis::cmd("FT.CONFIG").arg("GET").arg("MINSTEMLEN"),
                "MINSTEMLEN",
            )
        });
        Ok((fulltext::check_redis_min_stem_len(value)?, marker))
    }

    pub(super) fn search_fulltext_harness(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        self.prepare_search(dataset)?;
        let spec = fulltext::parse_spec(params.knob(fulltext::SPEC_KEY))?;
        let total_hits = fulltext::total_hits_plan("redis", &spec)?;
        if !total_hits.symmetric {
            eprintln!("⚠  {}", fulltext::ASYMMETRIC_TOTAL_HITS_WARNING);
        }
        let fields = fulltext::resolve_fields(&spec, dataset.config.schema.as_ref())?;
        let fetch = fulltext::resolve_fetch(&spec, dataset.config.schema.as_ref())?;
        let fetch = fetch.as_deref();

        let profile = self
            .fulltext_profile()?
            .ok_or("fulltext: no fulltext search config resolved an index profile")?;
        // The live index must have every searched field, as pinned. Refused
        // before the timed window, never searched anyway.
        let (min_stem_len, recorded_marker) =
            self.validate_fulltext_live_index(dataset, &spec, &fields, &profile)?;

        let top = params
            .top
            .map(|t| t as usize)
            .unwrap_or(vector_db_benchmark::filter_only::DEFAULT_LIMIT);
        let set = fulltext::load_query_set_for(
            &dataset.get_path()?,
            &spec,
            dataset.config.schema.as_ref(),
            top,
            num_queries,
        )?;
        if !set.skipped.is_empty() {
            eprintln!(
                "\tfulltext: {} of {} queries have no searchable term and are skipped",
                set.skipped.len(),
                set.total
            );
        }
        let query_timeout: i64 = crate::effective_config::env_parsed("REDIS_QUERY_TIMEOUT", 90_000);
        let commands: Vec<redis::Cmd> = set
            .queries
            .iter()
            .map(|q| {
                let mut cmd = redis::cmd("FT.SEARCH");
                for a in fulltext::redis_args_fetch(
                    &self.config.index_name,
                    &fulltext::query_redis(&fields, q),
                    top,
                    query_timeout,
                    profile.analysis,
                    total_hits.mode,
                    fetch,
                ) {
                    cmd.arg(a);
                }
                cmd
            })
            .collect();
        // `none`: WITHOUTCOUNT keeps the BM25 ranking only on some RediSearch
        // versions, so it is checked on this server before the timed window.
        // A disk window probes only its reserved lines (never a timed query,
        // which the probe would warm), or not at all when no timed query can
        // be cut by the page.
        let withoutcount_probe = if total_hits.mode == fulltext::TotalHits::None {
            match set.withoutcount_probe_plan(top) {
                fulltext::ProbePlan::NotNeeded(reason) => Some(Err(reason)),
                fulltext::ProbePlan::Run(ids) => {
                    let mut conn = resp3_connection(&self.redis_url, self.config.cluster_api)
                        .map_err(|e| format!("fulltext: Redis RESP3 connection failed: {e}"))?;
                    let probe = fulltext::probe_withoutcount(ids.len(), top, |i, mode| {
                        let mut cmd = redis::cmd("FT.SEARCH");
                        for a in fulltext::redis_args_with(
                            &self.config.index_name,
                            &fulltext::query_redis(&fields, &set.queries[ids[i]]),
                            top,
                            query_timeout,
                            profile.analysis,
                            mode,
                        ) {
                            cmd.arg(a);
                        }
                        cmd.query::<redis::Value>(&mut conn)
                            .map_err(|e| e.to_string())
                    })?;
                    Some(Ok(probe))
                }
            }
        } else {
            None
        };
        let rank_violations = AtomicUsize::new(0);
        let (fetched_hits, fetched_bytes) = (AtomicU64::new(0), AtomicU64::new(0));
        let order = set.dispatch_order(num_queries)?;
        let prime = set.prime_index();
        let num_to_run = order.len();
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;
        let capture = params.capture_query_results;
        let key_prefix = self.config.key_prefix.as_str();
        // Each timed query once, or (duration mode) the window cycled in
        // strict dispatch order until the deadline.
        let deadline = set.deadline();
        let dispatcher = Dispatcher::new(&order, deadline);
        let pb = self.create_progress_bar(num_to_run);

        let mut times = Samples::new(deadline.is_some());
        let mut errors = Vec::new();
        let mut captured: Option<Vec<RawQueryResult>> = capture.then(Vec::new);
        let mut capture_failure: Option<String> = None;
        let measured_start = std::thread::scope(|scope| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(scope, "redis-fulltext", parallel);
            for _ in 0..parallel {
                let (url, commands, queries) = (&self.redis_url, &commands, &set.queries);
                let (dispatcher, pb, rank_violations) = (&dispatcher, &pb, &rank_violations);
                let (fetched_hits, fetched_bytes) = (&fetched_hits, &fetched_bytes);
                let cluster_api = self.config.cluster_api;
                pool.spawn(move |ticket| -> WorkerOut {
                    let (mut errs, mut cap) = (Vec::new(), Vec::new());
                    let mut samples = Samples::new(deadline.is_some());
                    let mut capture_error = None;
                    let mut conn = match resp3_connection(url, cluster_api) {
                        Ok(c) => c,
                        Err(e) => {
                            ticket.fail(format!("Redis fulltext connection failed: {e}"));
                            return (samples, errs, cap, capture_error);
                        }
                    };
                    let deadline =
                        std::time::Duration::from_millis(query_timeout.max(0) as u64 + 30_000);
                    if let Err(e) = conn
                        .set_read_timeout(Some(deadline))
                        .and_then(|_| conn.set_write_timeout(Some(deadline)))
                    {
                        ticket.fail(format!("Redis fulltext socket deadline setup failed: {e}"));
                        return (samples, errs, cap, capture_error);
                    }
                    // Prime: the cold first round trip stays out of the window.
                    let _ = commands[prime].query::<redis::Value>(&mut conn);
                    let Some(t0) = ticket.arrive_and_wait() else {
                        return (samples, errs, cap, capture_error);
                    };
                    while let Some((seq, i)) = dispatcher.next(t0) {
                        let start = Instant::now();
                        let reply = commands[i].query::<redis::Value>(&mut conn);
                        let elapsed = start.elapsed().as_secs_f64();
                        let query_id = queries[i].query_id;
                        match reply.map_err(|e| e.to_string()).and_then(|v| {
                            let p = fulltext::redis_ranked_page_fetch(
                                &v,
                                top,
                                capture,
                                total_hits.mode,
                                fetch,
                            )?;
                            fulltext::check_query_page(&queries[i], p.total, p.returned, top)?;
                            if fetch.is_some() {
                                fetched_hits.fetch_add(p.returned as u64, Ordering::Relaxed);
                                fetched_bytes.fetch_add(p.fetched_bytes, Ordering::Relaxed);
                            }
                            Ok(p)
                        }) {
                            Ok(page) => {
                                samples.push(elapsed, t0);
                                // `Some` only under capture: the plain timed
                                // loop builds no id strings.
                                if let Some(hits) = page.hits {
                                    // A key outside this config's prefix is a
                                    // harness bug: fail the run, never guess.
                                    let rows =
                                        hits.iter()
                                            .map(|(key, score)| {
                                                vector_db_benchmark::filter_only::redis_key_row(
                                                    key, key_prefix,
                                                )
                                                .map(|row| RawHit {
                                                    row,
                                                    score: Some(*score),
                                                })
                                            })
                                            .collect::<Result<Vec<_>, _>>();
                                    match rows {
                                        Ok(rows) => cap.push(RawQueryResult::ok(
                                            query_id,
                                            seq,
                                            rows,
                                            queries[i].compound.as_ref().and(page.total),
                                            elapsed,
                                        )),
                                        Err(e) => {
                                            capture_error = Some(e);
                                            dispatcher.failed();
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                dispatcher.failed();
                                if fulltext::is_rank_violation(&e) {
                                    rank_violations.fetch_add(1, Ordering::Relaxed);
                                }
                                if capture {
                                    cap.push(RawQueryResult::failed(query_id, seq, elapsed, &e));
                                }
                                if errs.len() < 3 {
                                    errs.push(format!("fulltext query {query_id} failed: {e}"));
                                }
                            }
                        }
                        pb.inc(1);
                    }
                    (samples, errs, cap, capture_error)
                })?;
            }
            let (workers, start) = pool.start()?;
            for (samples, errs, cap, capture_error) in workers {
                times.append(samples);
                errors.extend(errs);
                if let Some(records) = captured.as_mut() {
                    records.extend(cap);
                }
                if let Some(e) = capture_error {
                    capture_failure.get_or_insert(e);
                }
            }
            Ok(start)
        })?;
        let total_time = measured_start.elapsed().as_secs_f64();
        if spec.compound.is_some() && !errors.is_empty() {
            return Err(format!(
                "compound suite failed ({} errors): {}",
                errors.len(),
                errors
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        pb.finish_and_clear();
        for e in errors.iter().take(3) {
            eprintln!("{e}");
        }
        fulltext::check_rank_violations(total_hits.mode, rank_violations.into_inner())?;
        if let Some(e) = capture_failure {
            return Err(format!("--dump-results: {e}"));
        }
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH"],
            "search",
            self.commandstats_baseline.as_ref(),
        )?;
        // Duration mode: the queries actually dispatched; otherwise the window.
        let num_to_run = if deadline.is_some() {
            dispatcher.dispatched()
        } else {
            num_to_run
        };
        let mut result = crate::engine::compute_search_stats(
            &times.latencies,
            &[],
            &[],
            &[],
            &[],
            total_time,
            top,
            parallel,
            num_to_run,
        )?;
        // No ground truth: the -1 sentinel keeps it out of precision tables.
        result.mean_precision_at_returned = -1.0;
        result.query_records = captured;
        let mut meta =
            fulltext::result_metadata_for("redis", &fields, &set, top, &profile, &recorded_marker)?;
        meta["redis_min_stem_len"] = min_stem_len;
        set.record_duration(
            &mut meta,
            num_to_run,
            times.completed_at(),
            &times.latencies,
            total_time,
        );
        fulltext::record_total_hits(&mut meta, "redis", &total_hits)?;
        match &withoutcount_probe {
            Some(Ok(probe)) => fulltext::record_withoutcount_probe(&mut meta, probe),
            Some(Err(reason)) => fulltext::record_withoutcount_probe_skipped(&mut meta, reason),
            None => {}
        }
        fulltext::record_fetch(
            &mut meta,
            "redis",
            fetch,
            fetched_hits.into_inner(),
            fetched_bytes.into_inner(),
        )?;
        result.fulltext = Some(meta);
        Ok(result)
    }
}
