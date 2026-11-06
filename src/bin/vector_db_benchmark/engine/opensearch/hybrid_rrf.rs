//! OpenSearch arm of the hybrid-rrf workload: a `hybrid` query fused by a
//! per-config search pipeline whose `score-ranker-processor` combines with
//! `rrf` (OpenSearch 2.19+). I/O only: the semantics, the request, the
//! pipeline and its read-back check, the reference and the score all come
//! from `vector_db_benchmark::hybrid_rrf`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use opensearch::http::headers::HeaderMap;
use opensearch::http::request::JsonBody;
use opensearch::http::Method;

use super::{
    create_os_client, knn_send, search_send, uuid_hex_to_int, OpenSearchEngine, SearchRetryPolicy,
};
use crate::config::SearchParams;
use crate::dataset::Dataset;
use crate::engine::{HybridRrfOutcome as Outcome, SearchResults};
use vector_db_benchmark::fulltext;
use vector_db_benchmark::hybrid_rrf as hr;
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::start_gate::WorkerPool;

impl OpenSearchEngine {
    /// `method` on `path` with an optional JSON body; the decoded reply.
    fn rest(
        &self,
        method: Method,
        path: &str,
        body: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        self.rt
            .block_on(async {
                self.client
                    .send(
                        method,
                        path,
                        HeaderMap::new(),
                        None::<&()>,
                        body.map(|b| JsonBody::new(b.clone())),
                        None,
                    )
                    .await?
                    .error_for_status_code()?
                    .json::<serde_json::Value>()
                    .await
            })
            .map_err(|e| format!("hybrid-rrf: {path}: {e}"))
    }

    /// Create (or replace) this config's search pipeline and verify it as the
    /// server reports it. Returns the read-back.
    fn put_rrf_pipeline(
        &self,
        name: &str,
        spec: &hr::RrfSpec,
    ) -> Result<serde_json::Value, String> {
        let path = format!("/_search/pipeline/{name}");
        self.rest(
            Method::Put,
            &path,
            Some(&hr::opensearch_pipeline_body(spec)),
        )?;
        let read_back = self.rest(Method::Get, &path, None)?;
        hr::check_opensearch_pipeline(&read_back, name, spec)?;
        Ok(read_back[name].clone())
    }

    pub(super) fn search_hybrid_rrf_impl(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        if self.config.skip_vector_index {
            return Err("hybrid-rrf requested on an index built without a vector field".into());
        }
        let spec = hr::parse_spec(params.knob(hr::SPEC_KEY))?;
        let text_spec = hr::parse_text_spec(params.knob(fulltext::SPEC_KEY))?;
        let total_hits = hr::total_hits_plan("opensearch", &text_spec)?;
        let ef = params.search_params.as_ref().and_then(|p| p.ef);
        let top = hr::check_params(
            params.top,
            ef,
            &spec,
            params.target_qps,
            params.duration_seconds,
        )?;
        let ef = ef.expect("check_params requires ef");

        // Server-side RRF exists from 2.19: refuse anything older here, with
        // the version in the message, never fall back to client-side fusion.
        let info = self.rest(Method::Get, "/", None)?;
        let version = info
            .pointer("/version/number")
            .and_then(|v| v.as_str())
            .ok_or("hybrid-rrf: OpenSearch version response is missing version.number")?
            .to_string();
        hr::check_opensearch_version(&version)?;

        let fields = fulltext::resolve_fields(&text_spec, dataset.config.schema.as_ref())?;
        let profile = self
            .fulltext_profile()?
            .ok_or("hybrid-rrf: no text search config resolved an index profile")?;
        let (shards, recorded_marker) = self.validate_fulltext_index(
            &fields,
            text_spec.allow_multi_shard,
            &profile,
            &fulltext::schema_text_fields(dataset.config.schema.as_ref()),
        )?;
        let (properties, _) = self.realized_schema("hybrid-rrf")?;
        if properties["vector"]["type"] != "knn_vector" {
            return Err(format!(
                "hybrid-rrf: the index has no knn_vector field `vector` (mapping: {})",
                properties["vector"]
            ));
        }

        let dir = dataset.get_path()?;
        let texts = fulltext::read_query_texts(&dir)?;
        let (vectors, neighbours, conditions) = dataset.read_queries()?;
        let _unfiltered: Vec<QueryFilter<()>> =
            conditions.try_resolve_all("OpenSearch hybrid-rrf", hr::refuse_conditions)?;
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

        let pipeline = hr::opensearch_pipeline_name(&self.name, &spec);
        let pipeline_read_back = self.put_rrf_pipeline(&pipeline, &spec)?;
        let outcome = self.run_hybrid_rrf(
            &set,
            &vectors,
            &fields,
            &spec,
            ef,
            top,
            &pipeline,
            params,
            num_queries,
            total_hits.mode,
        );
        // Drop the pipeline whatever happened; a failure to drop is reported
        // but does not hide the run's own error.
        if let Err(e) = self.rest(
            Method::Delete,
            &format!("/_search/pipeline/{pipeline}"),
            None,
        ) {
            eprintln!("⚠  hybrid-rrf: could not delete search pipeline {pipeline}: {e}");
        }
        let (mut result, references, mean_strict, retried) = outcome?;

        let mut text_meta = fulltext::result_metadata_for(
            "opensearch",
            &fields,
            &set.text_view(),
            spec.window,
            &profile,
            &recorded_marker,
        )?;
        text_meta["number_of_shards"] = serde_json::json!(shards);
        text_meta["allow_multi_shard"] = serde_json::json!(text_spec.allow_multi_shard);
        let engine = serde_json::json!({
            "engine": "opensearch",
            "opensearch_version": version,
            "request": "POST /<index>/_search?search_pipeline=<pipeline> {size: top, query: \
                        {hybrid: {pagination_depth: window, queries: [<fulltext multi_match>, \
                        {knn: {vector: {k: window, method_parameters: {ef_search: ef}}}}]}}}",
            "search_pipeline": pipeline,
            "search_pipeline_read_back": pipeline_read_back,
            "reference_request": "the fulltext workload's multi_match at size window, no pipeline",
            "vector_mapping": properties["vector"]["method"],
            "number_of_shards": shards,
            "retried_queries": retried,
        });
        let mut meta = hr::result_metadata(
            &spec,
            top,
            ef,
            text_meta,
            engine,
            &hr::ReferenceStats::of(&references, spec.window),
            mean_strict,
        );
        hr::record_total_hits(&mut meta, "opensearch", &total_hits)?;
        result.hybrid_rrf = Some(meta);
        Ok(result)
    }

    /// Reference pass, then the timed loop. Separate so the caller drops the
    /// pipeline on every path.
    #[allow(clippy::too_many_arguments)]
    fn run_hybrid_rrf(
        &self,
        set: &hr::QuerySet,
        vectors: &[Vec<f32>],
        fields: &[fulltext::FieldBoost],
        spec: &hr::RrfSpec,
        ef: i64,
        top: usize,
        pipeline: &str,
        params: &SearchParams,
        num_queries: i64,
        total: fulltext::TotalHits,
    ) -> Result<(SearchResults, Vec<hr::Reference>, f64, u64), String> {
        let policy = SearchRetryPolicy::from_env();
        let to_raw =
            |v: &serde_json::Value| serde_json::value::to_raw_value(v).map_err(|e| e.to_string());

        // Reference (untimed): the engine's exact BM25 top W per query.
        println!(
            "\thybrid-rrf: computing the reference (exact BM25 top {} + exact KNN top {}) for {} \
             queries...",
            spec.window,
            spec.window,
            set.queries.len()
        );
        let mut references = Vec::with_capacity(set.queries.len());
        for q in &set.queries {
            let body = to_raw(&hr::opensearch_reference_body(fields, &q.terms, spec))?;
            let (reply, _) = knn_send(&self.rt, &self.client, &self.index_name, &body, policy)
                .map_err(|e| format!("hybrid-rrf reference, query {}: {e}", q.query_id))?;
            let page = fulltext::opensearch_ranked_page(&reply, spec.window, true)
                .map_err(|e| format!("hybrid-rrf reference, query {}: {e}", q.query_id))?;
            let bm25 = page
                .hits
                .unwrap_or_default()
                .iter()
                .map(|(id, _)| uuid_hex_to_int(id))
                .collect::<Result<Vec<_>, _>>()?;
            references.push(hr::reference(&bm25, &q.knn_truth, spec)?);
        }

        // Serialized outside the timed window, as on the KNN path.
        let bodies: Vec<Box<serde_json::value::RawValue>> = set
            .queries
            .iter()
            .map(|q| {
                to_raw(&hr::opensearch_hybrid_body(
                    fields,
                    &q.terms,
                    &vectors[q.query_id],
                    ef,
                    spec,
                    top,
                    total,
                ))
            })
            .collect::<Result<_, _>>()?;

        // Probe (untimed): the request must be accepted and its reply pass the
        // checks before anything is timed.
        let (probe, _) = search_send(
            &self.rt,
            &self.client,
            &self.index_name,
            &bodies[0],
            policy,
            Some(pipeline),
        )
        .map_err(|e| format!("hybrid-rrf: probe query failed (OpenSearch hybrid + RRF): {e}"))?;
        parse_page(&probe, top, total)
            .and_then(|p| hr::score_page(&p, &references[0], spec, top))
            .map_err(|e| format!("hybrid-rrf: probe query: {e}"))?;

        let order = fulltext::schedule(bodies.len(), num_queries);
        let num_to_run = order.len();
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;
        let next = AtomicUsize::new(0);
        let pb = self.create_progress_bar(num_to_run);
        let (base_url, timeout, index_name) =
            (self.base_url.clone(), self.timeout, self.index_name.clone());
        let mut outcomes: Vec<Outcome> = Vec::with_capacity(num_to_run);
        let mut retried_queries = 0u64;
        let measured_start = std::thread::scope(|scope| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(scope, "opensearch-hybrid-rrf", parallel);
            for _ in 0..parallel {
                let (base_url, index_name) = (&base_url, &index_name);
                let (bodies, order, next, pb) = (&bodies, &order, &next, &pb);
                pool.spawn(move |ticket| -> (Vec<Outcome>, u64) {
                    let (mut out, mut retried) = (Vec::new(), 0u64);
                    let setup = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())
                        .and_then(|rt| create_os_client(base_url, timeout).map(|c| (rt, c)));
                    let (rt, client) = match setup {
                        Ok(s) => s,
                        Err(e) => {
                            ticket.fail(format!("opensearch-hybrid-rrf worker setup failed: {e}"));
                            return (out, retried);
                        }
                    };
                    // Prime: the cold first round trip stays out of the window.
                    let _ =
                        search_send(&rt, &client, index_name, &bodies[0], policy, Some(pipeline));
                    if ticket.arrive_and_wait().is_none() {
                        return (out, retried);
                    }
                    loop {
                        let seq = next.fetch_add(1, Ordering::Relaxed);
                        if seq >= order.len() {
                            break;
                        }
                        let i = order[seq];
                        // Timed: send + receive + decode, as on the KNN path.
                        let start = Instant::now();
                        let reply = search_send(
                            &rt,
                            &client,
                            index_name,
                            &bodies[i],
                            policy,
                            Some(pipeline),
                        );
                        let elapsed = start.elapsed();
                        let (latency, page) = match reply {
                            Ok((body, cost)) => {
                                // Bill only the attempt that produced the result.
                                retried += u64::from(cost.retried);
                                (
                                    elapsed.saturating_sub(cost.overhead).as_secs_f64(),
                                    parse_page(&body, top, total),
                                )
                            }
                            Err(e) => (elapsed.as_secs_f64(), Err(e)),
                        };
                        out.push((i, seq, latency, page));
                        pb.inc(1);
                    }
                    (out, retried)
                })?;
            }
            let (workers, start) = pool.start()?;
            for (o, r) in workers {
                outcomes.extend(o);
                retried_queries += r;
            }
            Ok(start)
        })?;
        let total_time = measured_start.elapsed().as_secs_f64();
        pb.finish_and_clear();
        if retried_queries > 0 {
            eprintln!(
                "⚠  {retried_queries} of {num_to_run} hybrid-rrf queries succeeded only after a \
                 retry — the server was shedding load; backoff is excluded from the latency"
            );
        }
        let (result, mean_strict) = crate::engine::hybrid_rrf_results(
            set,
            &references,
            spec,
            outcomes,
            total_time,
            top,
            parallel,
            params.capture_query_results,
        )?;
        Ok((result, references, mean_strict, retried_queries))
    }
}

/// A hybrid reply's page as `(row, fused score)`: the fulltext workload's
/// completeness checks (no timeout, no failed shard, ranked, no duplicate,
/// at most `top`, `hits.total` as `total` asked for), ids mapped to rows.
fn parse_page(
    body: &serde_json::Value,
    top: usize,
    total: fulltext::TotalHits,
) -> Result<Vec<(i64, f64)>, String> {
    fulltext::opensearch_ranked_page_with(body, top, true, total)?
        .hits
        .unwrap_or_default()
        .iter()
        .map(|(id, score)| uuid_hex_to_int(id).map(|row| (row, *score)))
        .collect()
}
