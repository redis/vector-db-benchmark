//! Redis arm of the ranked full-text workload (#339). I/O only: the query,
//! the reply contract, the index checks and the result metadata all come from
//! `vector_db_benchmark::fulltext`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use super::{redis_value_to_json, RedisEngine};
use crate::config::SearchParams;
use crate::dataset::Dataset;
use crate::engine::{redis_utils, SearchResults};
use vector_db_benchmark::fulltext;
use vector_db_benchmark::results_dump::{RawHit, RawQueryResult};
use vector_db_benchmark::start_gate::WorkerPool;

fn resp3_connection(url: &str) -> redis::RedisResult<redis::Connection> {
    let info = redis::Client::open(url)?.get_connection_info().clone();
    let settings = info
        .redis_settings()
        .clone()
        .set_protocol(redis::ProtocolVersion::RESP3);
    redis::Client::open(info.set_redis_settings(settings))?.get_connection()
}

type WorkerOut = (Vec<f64>, Vec<String>, Vec<RawQueryResult>, Option<String>);

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
        conn: &mut redis::Connection,
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

    pub(super) fn search_fulltext_harness(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        if self.config.skip_vector_index {
            return Err(
                "fulltext search requested on a no-vector (filter-only) index; fulltext needs \
                 the vector layout's TEXT fields"
                    .into(),
            );
        }
        self.prepare_search(dataset)?;
        let spec = fulltext::parse_spec(params.knob(fulltext::SPEC_KEY))?;
        let fields = fulltext::resolve_fields(&spec, dataset.config.schema.as_ref())?;

        let profile = self
            .fulltext_profile()?
            .ok_or("fulltext: no fulltext search config resolved an index profile")?;
        // The live index must have every searched field, as pinned. Refused
        // before the timed window, never searched anyway.
        let (min_stem_len, recorded_marker) = {
            let mut conn = resp3_connection(&self.redis_url)
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
                        fulltext::redis_marker_failure(&profile, "read", &e.to_string())?
                    );
                    None
                }
            };
            let marker = match marker {
                None => serde_json::Value::Null,
                Some(m) => serde_json::from_str(&m)
                    .map_err(|e| format!("fulltext: unreadable index marker {m:?}: {e}"))?,
            };
            fulltext::validate_redis_index_for(
                &redis_value_to_json(&info),
                &fields,
                &profile,
                &marker,
            )?;
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
            (fulltext::check_redis_min_stem_len(value)?, marker)
        };

        let texts = fulltext::read_query_texts(&dataset.get_path()?)?;
        let set = fulltext::build_query_set(&texts)?;
        if !set.skipped.is_empty() {
            eprintln!(
                "\tfulltext: {} of {} queries have no searchable term and are skipped",
                set.skipped.len(),
                set.total
            );
        }
        let top = params
            .top
            .map(|t| t as usize)
            .unwrap_or(vector_db_benchmark::filter_only::DEFAULT_LIMIT);
        let query_timeout: i64 = crate::effective_config::env_parsed("REDIS_QUERY_TIMEOUT", 90_000);
        let commands: Vec<redis::Cmd> = set
            .queries
            .iter()
            .map(|q| {
                let mut cmd = redis::cmd("FT.SEARCH");
                for a in fulltext::redis_args_for(
                    &self.config.index_name,
                    &fulltext::redis_query(&fields, &q.terms),
                    top,
                    query_timeout,
                    profile.analysis,
                ) {
                    cmd.arg(a);
                }
                cmd
            })
            .collect();
        let order = fulltext::schedule(commands.len(), num_queries);
        let num_to_run = order.len();
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;
        let capture = params.capture_query_results;
        let key_prefix = self.config.key_prefix.as_str();
        let next = AtomicUsize::new(0);
        let pb = self.create_progress_bar(num_to_run);

        let mut times = Vec::with_capacity(num_to_run);
        let mut errors = Vec::new();
        let mut captured: Option<Vec<RawQueryResult>> = capture.then(Vec::new);
        let mut capture_failure: Option<String> = None;
        let measured_start = std::thread::scope(|scope| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(scope, "redis-fulltext", parallel);
            for _ in 0..parallel {
                let (url, commands, order, queries) =
                    (&self.redis_url, &commands, &order, &set.queries);
                let (next, pb) = (&next, &pb);
                pool.spawn(move |ticket| -> WorkerOut {
                    let (mut samples, mut errs, mut cap) = (Vec::new(), Vec::new(), Vec::new());
                    let mut capture_error = None;
                    let mut conn = match resp3_connection(url) {
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
                    let _ = commands[0].query::<redis::Value>(&mut conn);
                    if ticket.arrive_and_wait().is_none() {
                        return (samples, errs, cap, capture_error);
                    }
                    loop {
                        let seq = next.fetch_add(1, Ordering::Relaxed);
                        if seq >= order.len() {
                            break;
                        }
                        let i = order[seq];
                        let start = Instant::now();
                        let reply = commands[i].query::<redis::Value>(&mut conn);
                        let elapsed = start.elapsed().as_secs_f64();
                        let query_id = queries[i].query_id;
                        match reply
                            .map_err(|e| e.to_string())
                            .and_then(|v| fulltext::redis_ranked_page(&v, top, capture))
                        {
                            Ok(page) => {
                                samples.push(elapsed);
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
                                            query_id, seq, rows, None, elapsed,
                                        )),
                                        Err(e) => {
                                            capture_error = Some(e);
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
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
                times.extend(samples);
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
        pb.finish_and_clear();
        for e in errors.iter().take(3) {
            eprintln!("{e}");
        }
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
        let mut result = crate::engine::compute_search_stats(
            &times,
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
        result.fulltext = Some(meta);
        Ok(result)
    }
}
