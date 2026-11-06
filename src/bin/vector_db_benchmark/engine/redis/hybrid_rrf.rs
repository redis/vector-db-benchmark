//! Redis arm of the hybrid-rrf workload: `FT.HYBRID … COMBINE RRF` (Redis
//! 8.4+). I/O only: the semantics, the command, the reply contract, the
//! reference and the score all come from `vector_db_benchmark::hybrid_rrf`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use super::fulltext::resp3_connection;
use super::{encode_vector, RedisEngine};
use crate::config::SearchParams;
use crate::dataset::Dataset;
use crate::engine::{redis_utils, SearchResults};
use vector_db_benchmark::filter_only::redis_key_row;
use vector_db_benchmark::fulltext;
use vector_db_benchmark::hybrid_rrf as hr;
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::start_gate::WorkerPool;

use crate::engine::HybridRrfOutcome as Outcome;

fn ft_hybrid_calls(conn: &mut vector_db_benchmark::redis_transport::Connection) -> Option<u64> {
    // Read after the untimed probe, so a missing line means the server hides
    // the stat (managed services), not "no call yet": recorded unverified.
    let info: String = redis::cmd("INFO").arg("commandstats").query(conn).ok()?;
    hr::commandstats_calls(&info, "FT.HYBRID")
}

impl RedisEngine {
    pub(super) fn search_hybrid_rrf_harness(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        if self.config.skip_vector_index {
            return Err("hybrid-rrf requested on an index built without a vector field".into());
        }
        // FT.HYBRID's KNN clause takes K and EF_RUNTIME only (RediSearch 8.6
        // `parseKNNClause`): FLAT has no breadth to set and SVS's window has
        // no spelling there, so `ef` would be silently meaningless.
        if self.vector_algorithm_token() != "HNSW" {
            return Err(format!(
                "hybrid-rrf on Redis needs an HNSW vector index (FT.HYBRID KNN takes K and \
                 EF_RUNTIME only); this config builds {}",
                self.vector_algorithm_token()
            ));
        }
        self.prepare_search(dataset)?;
        let spec = hr::parse_spec(params.knob(hr::SPEC_KEY))?;
        let text_spec = hr::parse_text_spec(params.knob(fulltext::SPEC_KEY))?;
        let ef = params.search_params.as_ref().and_then(|p| p.ef);
        let top = hr::check_params(
            params.top,
            ef,
            &spec,
            params.target_qps,
            params.duration_seconds,
        )?;
        let ef = ef.expect("check_params requires ef");
        let fields = fulltext::resolve_fields(&text_spec, dataset.config.schema.as_ref())?;
        let profile = self
            .fulltext_profile()?
            .ok_or("hybrid-rrf: no text search config resolved an index profile")?;
        let (min_stem_len, recorded_marker) =
            self.validate_fulltext_live_index(dataset, &text_spec, &fields, &profile)?;

        // The vector field must be this config's (algorithm, compression).
        {
            let mut conn = resp3_connection(&self.redis_url, self.config.cluster_api)
                .map_err(|e| format!("hybrid-rrf: Redis RESP3 connection failed: {e}"))?;
            let info: redis::Value = redis::cmd("FT.INFO")
                .arg(&self.config.index_name)
                .query(&mut conn)
                .map_err(|e| format!("hybrid-rrf: FT.INFO failed: {e}"))?;
            self.verify_vector_field(
                &info,
                vector_db_benchmark::redis_hnsw_sq8::ReadBack::Created,
            )?;
        }

        // Queries: text from tests.jsonl, vectors + exact KNN truth from the
        // dataset. A declared condition is refused through the choke point.
        let dir = dataset.get_path()?;
        let texts = fulltext::read_query_texts(&dir)?;
        let (vectors, neighbours, conditions) = dataset.read_queries()?;
        let _unfiltered: Vec<QueryFilter<()>> =
            conditions.try_resolve_all("Redis hybrid-rrf", hr::refuse_conditions)?;
        let set = hr::build_query_set(&texts, &neighbours, &spec)?;
        if set.queries.is_empty() {
            return Err("hybrid-rrf: no query has a searchable term".into());
        }
        if !set.skipped.is_empty() {
            eprintln!(
                "\thybrid-rrf: {} of {} queries have no searchable term and are skipped",
                set.skipped.len(),
                set.total
            );
        }
        let query_timeout: i64 = crate::effective_config::env_parsed("REDIS_QUERY_TIMEOUT", 90_000);
        let index = self.config.index_name.clone();
        let key_prefix = self.config.key_prefix.clone();
        let search_queries: Vec<String> = set
            .queries
            .iter()
            .map(|q| hr::redis_search_query(&fields, &q.terms, profile.analysis))
            .collect();

        // Reference (untimed): the engine's exact BM25 top W per query.
        let mut conn = resp3_connection(&self.redis_url, self.config.cluster_api)
            .map_err(|e| format!("hybrid-rrf: Redis RESP3 connection failed: {e}"))?;
        println!(
            "\thybrid-rrf: computing the reference (exact BM25 top {} + exact KNN top {}) for {} \
             queries...",
            spec.window,
            spec.window,
            set.queries.len()
        );
        let mut references = Vec::with_capacity(set.queries.len());
        for (q, query) in set.queries.iter().zip(&search_queries) {
            let mut cmd = redis::cmd("FT.SEARCH");
            for a in hr::redis_reference_args(&index, query, &spec, query_timeout, profile.analysis)
            {
                cmd.arg(a);
            }
            let reply: redis::Value = cmd
                .query(&mut conn)
                .map_err(|e| format!("hybrid-rrf reference, query {}: {e}", q.query_id))?;
            let page = fulltext::redis_ranked_page(&reply, spec.window, true)
                .map_err(|e| format!("hybrid-rrf reference, query {}: {e}", q.query_id))?;
            let bm25 = page
                .hits
                .unwrap_or_default()
                .iter()
                .map(|(key, _)| redis_key_row(key, &key_prefix))
                .collect::<Result<Vec<_>, _>>()?;
            references.push(hr::reference(&bm25, &q.knn_truth, &spec)?);
        }

        // Commands, built outside the timed window.
        let commands: Vec<redis::Cmd> = set
            .queries
            .iter()
            .zip(&search_queries)
            .map(|(q, query)| {
                let mut cmd = redis::cmd("FT.HYBRID");
                for a in hr::redis_hybrid_args(&index, query, ef, &spec, top, query_timeout) {
                    cmd.arg(a);
                }
                cmd.arg(encode_vector(&self.config.data_type, &vectors[q.query_id]));
                cmd
            })
            .collect();

        // Probe (untimed): a server without FT.HYBRID, or one that rejects
        // the command, is refused here with its own message.
        let server_version = redis::cmd("INFO")
            .arg("server")
            .query::<String>(&mut conn)
            .ok()
            .and_then(|i| {
                i.lines().find_map(|l| {
                    l.strip_prefix("redis_version:")
                        .map(|v| v.trim().to_string())
                })
            });
        commands[0]
            .query::<redis::Value>(&mut conn)
            .map_err(|e| {
                format!(
                    "hybrid-rrf: FT.HYBRID failed on the probe query (Redis {}; server-side RRF \
                     needs Redis 8.4+, there is no client-side fallback): {e}",
                    server_version.as_deref().unwrap_or("version unknown")
                )
            })
            .and_then(|v| {
                // Same checks as every timed reply, scored against its
                // reference: a server that accepted COMBINE RRF but fused
                // differently is refused before anything is timed.
                let page = hr::parse_redis_hybrid_reply(&v, top)?
                    .into_iter()
                    .map(|(key, score)| redis_key_row(&key, &key_prefix).map(|row| (row, score)))
                    .collect::<Result<Vec<_>, _>>()?;
                hr::score_page(&page, &references[0], &spec, top)
                    .map(|_| ())
                    .map_err(|e| format!("hybrid-rrf: probe query: {e}"))
            })?;
        // One node's INFO cannot count a cluster's calls: verified standalone only.
        let cluster_api = self.config.cluster_api;
        let calls_before = (!cluster_api).then(|| ft_hybrid_calls(&mut conn)).flatten();

        let order = fulltext::schedule(commands.len(), num_queries);
        let num_to_run = order.len();
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;
        let next = AtomicUsize::new(0);
        let pb = self.create_progress_bar(num_to_run);
        let mut outcomes: Vec<Outcome> = Vec::with_capacity(num_to_run);
        let measured_start = std::thread::scope(|scope| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(scope, "redis-hybrid-rrf", parallel);
            for _ in 0..parallel {
                let (url, commands, order, next, pb) =
                    (&self.redis_url, &commands, &order, &next, &pb);
                let (cluster_api, key_prefix) = (self.config.cluster_api, key_prefix.as_str());
                pool.spawn(move |ticket| -> Vec<Outcome> {
                    let mut out = Vec::new();
                    let mut conn = match resp3_connection(url, cluster_api) {
                        Ok(c) => c,
                        Err(e) => {
                            ticket.fail(format!("Redis hybrid-rrf connection failed: {e}"));
                            return out;
                        }
                    };
                    let deadline =
                        std::time::Duration::from_millis(query_timeout.max(0) as u64 + 30_000);
                    if let Err(e) = conn
                        .set_read_timeout(Some(deadline))
                        .and_then(|_| conn.set_write_timeout(Some(deadline)))
                    {
                        ticket.fail(format!(
                            "Redis hybrid-rrf socket deadline setup failed: {e}"
                        ));
                        return out;
                    }
                    // Prime: the cold first round trip stays out of the window.
                    let _ = commands[0].query::<redis::Value>(&mut conn);
                    if ticket.arrive_and_wait().is_none() {
                        return out;
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
                        let page = reply.map_err(|e| e.to_string()).and_then(|v| {
                            hr::parse_redis_hybrid_reply(&v, top)?
                                .into_iter()
                                .map(|(key, score)| {
                                    redis_key_row(&key, key_prefix).map(|row| (row, score))
                                })
                                .collect::<Result<Vec<_>, _>>()
                        });
                        out.push((i, seq, elapsed, page));
                        pb.inc(1);
                    }
                    out
                })?;
            }
            let (workers, start) = pool.start()?;
            for w in workers {
                outcomes.extend(w);
            }
            Ok(start)
        })?;
        let total_time = measured_start.elapsed().as_secs_f64();
        pb.finish_and_clear();
        let calls_after = (!cluster_api).then(|| ft_hybrid_calls(&mut conn)).flatten();

        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.HYBRID", "FT.SEARCH"],
            "search",
            self.commandstats_baseline.as_ref(),
        )?;
        let calls = hr::check_redis_calls(calls_before, calls_after, num_to_run)?;

        let (mut result, mean_strict) = crate::engine::hybrid_rrf_results(
            &set,
            &references,
            &spec,
            outcomes,
            total_time,
            top,
            parallel,
            params.capture_query_results,
        )?;
        let mut text_meta = fulltext::result_metadata_for(
            "redis",
            &fields,
            &set.text_view(),
            spec.window,
            &profile,
            &recorded_marker,
        )?;
        text_meta["redis_min_stem_len"] = min_stem_len;
        let engine = serde_json::json!({
            "engine": "redis",
            "redis_version": server_version,
            "command": "FT.HYBRID <idx> SEARCH <bm25 query> SCORER BM25STD VSIM @vector $BLOB \
                        KNN 4 K <window> EF_RUNTIME <ef> COMBINE RRF 4 CONSTANT <rank_constant> \
                        WINDOW <window> LIMIT 0 <top> TIMEOUT <ms> PARAMS 2 BLOB <vector>",
            "search_subquery": if profile.analysis == fulltext::Analysis::Parity {
                "the fulltext query with every term quoted (verbatim, no stem expansion: \
                 FT.HYBRID's SEARCH takes no VERBATIM)"
            } else {
                "the fulltext workload's query string"
            },
            "reference_command": "FT.SEARCH <idx> <same query> NOCONTENT WITHSCORES SCORER \
                                  BM25STD LIMIT 0 <window> DIALECT 2 TIMEOUT <ms> (+ VERBATIM \
                                  in parity)",
            "vector_algorithm": self.vector_algorithm_token(),
            "vector_field": "verified via FT.INFO (algorithm, compression)",
            "commandstats": calls,
            "protocol": "RESP3; a reply with any warning fails its query",
        });
        result.hybrid_rrf = Some(hr::result_metadata(
            &spec,
            top,
            ef,
            text_meta,
            engine,
            &hr::ReferenceStats::of(&references, spec.window),
            mean_strict,
        ));
        Ok(result)
    }
}
