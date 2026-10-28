//! OpenSearch arm of the ranked full-text workload (#339). I/O only: the
//! analyzer, the request, the reply contract, the index checks and the result
//! metadata all come from `vector_db_benchmark::fulltext`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use opensearch::indices::{IndicesGetMappingParts, IndicesGetSettingsParts};

use super::{create_os_client, knn_send, uuid_hex_to_int, OpenSearchEngine, SearchRetryPolicy};
use crate::config::SearchParams;
use crate::dataset::Dataset;
use crate::engine::SearchResults;
use vector_db_benchmark::fulltext;
use vector_db_benchmark::results_dump::{RawHit, RawQueryResult};
use vector_db_benchmark::start_gate::WorkerPool;

type WorkerOut = (
    Vec<f64>,
    Vec<String>,
    u64,
    Vec<RawQueryResult>,
    Option<String>,
);

impl OpenSearchEngine {
    /// Whether this engine config runs the fulltext workload, i.e. whether the
    /// index must be created with the pinned analyzer.
    pub(super) fn serves_fulltext(&self) -> bool {
        self.search_params
            .iter()
            .any(|sp| sp.workload == Some(vector_db_benchmark::workload::Workload::Fulltext))
    }

    /// Analyze every schema `text` field with the pinned analyzer and declare
    /// it in the index settings.
    pub(super) fn pin_fulltext_analyzer(
        &self,
        dataset: &Dataset,
        properties: &mut serde_json::Value,
        index_settings: &mut serde_json::Value,
    ) {
        for field in fulltext::schema_text_fields(dataset.config.schema.as_ref()) {
            properties[field.as_str()] = fulltext::opensearch_text_mapping();
        }
        index_settings["analysis"] = fulltext::opensearch_analysis();
    }

    fn validate_fulltext_index(
        &self,
        fields: &[fulltext::FieldBoost],
        allow_multi_shard: bool,
    ) -> Result<u64, String> {
        let (mapping, settings) = self
            .rt
            .block_on(async {
                let mapping = self
                    .client
                    .indices()
                    .get_mapping(IndicesGetMappingParts::Index(&[&self.index_name]))
                    .send()
                    .await?
                    .error_for_status_code()?
                    .json::<serde_json::Value>()
                    .await?;
                let settings = self
                    .client
                    .indices()
                    .get_settings(IndicesGetSettingsParts::Index(&[&self.index_name]))
                    .send()
                    .await?
                    .error_for_status_code()?
                    .json::<serde_json::Value>()
                    .await?;
                Ok::<_, opensearch::Error>((mapping, settings))
            })
            .map_err(|e| format!("fulltext: index check failed: {e}"))?;
        fulltext::validate_opensearch_index(
            &mapping[&self.index_name]["mappings"]["properties"],
            &settings[&self.index_name]["settings"]["index"],
            fields,
            allow_multi_shard,
        )
    }

    pub(super) fn search_fulltext_impl(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        if self.config.skip_vector_index {
            return Err(
                "fulltext search requested on a no-vector (filter-only) index; fulltext needs \
                 the vector layout's text fields"
                    .into(),
            );
        }
        let spec = fulltext::parse_spec(params.knob(fulltext::SPEC_KEY))?;
        let fields = fulltext::resolve_fields(&spec, dataset.config.schema.as_ref())?;
        // The live index must have every searched field, as pinned. Refused
        // before the timed window, never searched anyway.
        let shards = self.validate_fulltext_index(&fields, spec.allow_multi_shard)?;

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
        // Serialized outside the timed window, as on the KNN path.
        let bodies: Vec<Box<serde_json::value::RawValue>> = set
            .queries
            .iter()
            .map(|q| {
                serde_json::value::to_raw_value(&fulltext::opensearch_body(&fields, &q.terms, top))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        let order = fulltext::schedule(bodies.len(), num_queries);
        let num_to_run = order.len();
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;
        let capture = params.capture_query_results;
        let policy = SearchRetryPolicy::from_env();
        let next = AtomicUsize::new(0);
        let pb = self.create_progress_bar(num_to_run);
        let (base_url, timeout, index_name) =
            (self.base_url.clone(), self.timeout, self.index_name.clone());

        let mut times = Vec::with_capacity(num_to_run);
        let mut errors = Vec::new();
        let mut retried_queries = 0u64;
        let mut captured: Option<Vec<RawQueryResult>> = capture.then(Vec::new);
        let mut capture_failure: Option<String> = None;
        let measured_start = std::thread::scope(|scope| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(scope, "opensearch-fulltext", parallel);
            for _ in 0..parallel {
                let (base_url, index_name) = (&base_url, &index_name);
                let (bodies, order, queries, next, pb) =
                    (&bodies, &order, &set.queries, &next, &pb);
                pool.spawn(move |ticket| -> WorkerOut {
                    let (mut samples, mut errs, mut cap) = (Vec::new(), Vec::new(), Vec::new());
                    let (mut retried, mut capture_error) = (0u64, None);
                    let setup = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())
                        .and_then(|rt| create_os_client(base_url, timeout).map(|c| (rt, c)));
                    let (rt, client) = match setup {
                        Ok(s) => s,
                        Err(e) => {
                            ticket.fail(format!("opensearch-fulltext worker setup failed: {e}"));
                            return (samples, errs, retried, cap, capture_error);
                        }
                    };
                    // Prime: the cold first round trip stays out of the window.
                    let _ = knn_send(&rt, &client, index_name, &bodies[0], policy);
                    if ticket.arrive_and_wait().is_none() {
                        return (samples, errs, retried, cap, capture_error);
                    }
                    loop {
                        let seq = next.fetch_add(1, Ordering::Relaxed);
                        if seq >= order.len() {
                            break;
                        }
                        let i = order[seq];
                        let query_id = queries[i].query_id;
                        // Timed: send + receive + decode, as on the KNN path.
                        let start = Instant::now();
                        let reply = knn_send(&rt, &client, index_name, &bodies[i], policy);
                        let elapsed = start.elapsed();
                        match reply.and_then(|(body, cost)| {
                            fulltext::opensearch_ranked_page(&body, top, capture).map(|p| (p, cost))
                        }) {
                            Ok((page, cost)) => {
                                // Bill only the attempt that produced the result
                                // (the KNN path's contract).
                                let t = elapsed.saturating_sub(cost.overhead).as_secs_f64();
                                retried += u64::from(cost.retried);
                                samples.push(t);
                                // `Some` only under capture.
                                if let Some(hits) = page.hits {
                                    let rows = hits
                                        .iter()
                                        .map(|(id, score)| {
                                            uuid_hex_to_int(id).map(|row| RawHit {
                                                row,
                                                score: Some(*score),
                                            })
                                        })
                                        .collect::<Result<Vec<_>, _>>();
                                    match rows {
                                        Ok(rows) => cap
                                            .push(RawQueryResult::ok(query_id, seq, rows, None, t)),
                                        Err(e) => {
                                            capture_error = Some(e);
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                if capture {
                                    cap.push(RawQueryResult::failed(
                                        query_id,
                                        seq,
                                        elapsed.as_secs_f64(),
                                        &e,
                                    ));
                                }
                                if errs.len() < 3 {
                                    errs.push(format!("fulltext query {query_id} failed: {e}"));
                                }
                            }
                        }
                        pb.inc(1);
                    }
                    (samples, errs, retried, cap, capture_error)
                })?;
            }
            let (workers, start) = pool.start()?;
            for (samples, errs, retried, cap, capture_error) in workers {
                times.extend(samples);
                errors.extend(errs);
                retried_queries += retried;
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
        if retried_queries > 0 {
            eprintln!(
                "⚠  {retried_queries} of {num_to_run} fulltext queries succeeded only after a \
                 retry — the server was shedding load; backoff is excluded from the latency"
            );
        }
        if let Some(e) = capture_failure {
            return Err(format!("--dump-results: {e}"));
        }
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
        result.mean_precision_at_returned = -1.0;
        result.query_records = captured;
        let mut meta = fulltext::result_metadata("opensearch", &fields, &set, top)?;
        meta["retried_queries"] = serde_json::json!(retried_queries);
        // Read back from the live index (`_settings`), not from the config.
        meta["number_of_shards"] = serde_json::json!(shards);
        meta["allow_multi_shard"] = serde_json::json!(spec.allow_multi_shard);
        result.fulltext = Some(meta);
        Ok(result)
    }
}
