//! Azure Cosmos DB for NoSQL adapter using the official Rust SDK.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use azure_data_cosmos::feed::FeedRange;
use azure_data_cosmos::models::{
    ContainerProperties, ThroughputProperties, VectorDistanceFunction, VectorIndexType,
};
use azure_data_cosmos::options::{
    ContentResponseOnWrite, CreateContainerOptions, MaxItemCountHint, OperationOptionsBuilder,
    QueryOptions, ReadFeedRangesOptions,
};
use azure_data_cosmos::{
    AccountReference, ContainerClient, CosmosClient, CosmosRuntime, FeedScope, Query,
    RoutingStrategy,
};
use futures::{stream, StreamExt, TryStreamExt};
use serde::Serialize;
use tokio::sync::Semaphore;
use vector_db_benchmark::cosmosdb::{
    await_index_use, container_name, container_policy, count_stable_ranges, fan_out_hits,
    index_probe_deadline, index_type, knn_query, manual_throughput, parity_sample_indices,
    positive_u32, reject_unknown_keys, retry_throttled, sample_parity, scan_query, upload_parallel,
    validate_corpus_size, validate_hit_count, validate_query_vector, validate_top,
    vector_distance_function, verify_policy, DiskAnnBuildOptions, IndexProbe, KnnQueryText,
    ParityTally, PartitionConcurrency, PartitionFanout, ProbeQuery, QueryHit, SearchDiagnostics,
};
use vector_db_benchmark::start_gate::WorkerPool;
use vector_db_benchmark::workload::IndexLayout;

use super::{compute_search_stats, CorpusCount, Engine, SearchResults, UploadStats};
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;

#[derive(Clone, Copy, Serialize)]
struct Item<'a> {
    id: &'a str,
    embedding: &'a [f32],
}

const SEARCH_LIST_SIZE: &str = "search_list_size_multiplier";
const QUANTIZED_VECTOR_LIST: &str = "quantized_vector_list_multiplier";
const PARTITION_CONCURRENCY: &str = "partition_concurrency";
const SEARCH_KNOBS: [&str; 3] = [
    SEARCH_LIST_SIZE,
    QUANTIZED_VECTOR_LIST,
    PARTITION_CONCURRENCY,
];
const COLLECTION_KEYS: [&str; 3] = [
    "ru_per_second",
    "indexing_search_list_size",
    "quantization_byte_size",
];
const UNSUPPORTED_SEARCH_KNOB: &str =
    "Cosmos DB supports only DiskANN search multipliers and partition_concurrency as tuning knobs";
const UNSUPPORTED_COLLECTION_KEY: &str =
    "Cosmos DB collection_params supports only ru_per_second, indexing_search_list_size and quantization_byte_size";
/// Sampled queries compared across routing modes before a fan-out search.
const PARITY_SAMPLES: usize = 8;
/// Feed ranges whose IDs are counted at once.
const ROW_COUNT_CONCURRENCY: usize = 16;
/// TOP K of the index-use probe queries.
const INDEX_PROBE_TOP: usize = 10;
/// Feed ranges probed at once.
const INDEX_PROBE_CONCURRENCY: usize = 16;
/// Sampled parity queries in flight at once.
const PARITY_CONCURRENCY: usize = 4;

fn positive_knob(params: &SearchParams, key: &str) -> Result<Option<NonZeroU32>, String> {
    params
        .knob(key)
        .map(|value| positive_u32(value, key))
        .transpose()
}

fn diskann_multiplier(
    params: &SearchParams,
    kind: &VectorIndexType,
    key: &str,
) -> Result<Option<NonZeroU32>, String> {
    let value = positive_knob(params, key)?;
    if value.is_some() && *kind != VectorIndexType::DiskANN {
        return Err(format!("Cosmos DB {key} requires DiskANN"));
    }
    Ok(value)
}

pub struct CosmosDbEngine {
    name: String,
    database: String,
    container_name: String,
    index_type: VectorIndexType,
    build_options: DiskAnnBuildOptions,
    upload_parallel: usize,
    search_params: Vec<SearchParams>,
    throughput: Option<u64>,
    client: CosmosClient,
    rt: tokio::runtime::Runtime,
    container: Option<ContainerClient>,
    created_container: bool,
    verified_policy: Option<ContainerProperties>,
    /// `None` skips the DiskANN index-use probe.
    index_probe_deadline: Option<Duration>,
    /// Per-range probe results once every range served from its index.
    index_probes: Option<Vec<IndexProbe>>,
    index_wait_seconds: Option<f64>,
}

impl CosmosDbEngine {
    pub fn new(config: &EngineConfig, dataset: &str) -> Result<Self, String> {
        let kind = index_type(config.algorithm.as_deref())?;
        let search_params = config.search_params.clone().unwrap_or_default();
        for params in &search_params {
            if let Some(top) = params.top {
                validate_top(usize::try_from(top).map_err(|_| "Cosmos DB top must be positive")?)?;
            }
            if params.calibration_param.is_some()
                || params.num_candidates.is_some()
                || params
                    .search_params
                    .as_ref()
                    .is_some_and(|p| p.ef.is_some())
            {
                return Err(UNSUPPORTED_SEARCH_KNOB.into());
            }
            let nested = params.search_params.as_ref().and_then(|p| p.extra.as_ref());
            for extra in [params.extra.as_ref(), nested].into_iter().flatten() {
                reject_unknown_keys(extra.keys(), &SEARCH_KNOBS, UNSUPPORTED_SEARCH_KNOB)?;
            }
            diskann_multiplier(params, &kind, SEARCH_LIST_SIZE)?;
            diskann_multiplier(params, &kind, QUANTIZED_VECTOR_LIST)?;
            PartitionConcurrency::from_config(params.knob(PARTITION_CONCURRENCY))?;
        }
        let collection = config.collection_params.as_ref();
        if collection.is_some_and(|p| p.hnsw_config.is_some() || p.index_options.is_some()) {
            return Err(UNSUPPORTED_COLLECTION_KEY.into());
        }
        if let Some(extra) = collection.and_then(|p| p.extra.as_ref()) {
            reject_unknown_keys(extra.keys(), &COLLECTION_KEYS, UNSUPPORTED_COLLECTION_KEY)?;
        }
        let collection_option = |key| {
            collection
                .and_then(|p| p.extra.as_ref())
                .and_then(|p| p.get(key))
        };
        let build_options = DiskAnnBuildOptions::from_config(
            &kind,
            collection_option("indexing_search_list_size"),
            collection_option("quantization_byte_size"),
        )?;
        let upload_parallel = upload_parallel(config.upload_params.as_ref())?;
        let index_probe_deadline = index_probe_deadline(
            crate::effective_config::env_opt("AZURE_COSMOS_INDEX_PROBE_SECONDS").as_deref(),
        )?;

        let endpoint = crate::effective_config::env_opt("AZURE_COSMOS_ENDPOINT")
            .ok_or("Set AZURE_COSMOS_ENDPOINT for the Cosmos DB engine")?
            .parse()
            .map_err(|_| "AZURE_COSMOS_ENDPOINT must be a Cosmos DB account URL")?;
        // The configuration recorder redacts this variable.
        let key = crate::effective_config::env_opt("AZURE_COSMOS_KEY")
            .filter(|key| !key.is_empty())
            .ok_or("Set AZURE_COSMOS_KEY for the Cosmos DB engine")?;
        let database = crate::effective_config::env_or("AZURE_COSMOS_DATABASE", "vdb-benchmark");
        let regions = crate::effective_config::env_opt("AZURE_COSMOS_REGION")
            .into_iter()
            .map(Into::into)
            .collect();
        // The SDK reads its own retry, transport and pool settings from the
        // environment when the runtime starts. Record all of them.
        crate::effective_config::observe_env_prefix("AZURE_COSMOS_");
        let throughput = manual_throughput(
            collection_option("ru_per_second"),
            &crate::effective_config::env_or("AZURE_COSMOS_RU_PER_SECOND", "400"),
        )?;
        let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
        let account = AccountReference::with_authentication_key(endpoint, key);
        let client = rt
            .block_on(async {
                let runtime = CosmosRuntime::builder().build().await?;
                CosmosClient::builder()
                    .with_runtime(runtime)
                    .with_default_operation_options(
                        OperationOptionsBuilder::new()
                            .with_content_response_on_write(ContentResponseOnWrite::Disabled)
                            .build(),
                    )
                    .build(account, RoutingStrategy::PreferredRegions(regions))
                    .await
            })
            .map_err(|e| format!("Cosmos DB connection failed: {e}"))?;

        Ok(Self {
            name: config.name.clone(),
            database,
            container_name: container_name(&config.name, dataset),
            index_type: kind,
            build_options,
            upload_parallel,
            search_params,
            throughput,
            client,
            rt,
            container: None,
            created_container: false,
            verified_policy: None,
            index_probe_deadline,
            index_probes: None,
            index_wait_seconds: None,
        })
    }

    fn open_container(&mut self) -> Result<&ContainerClient, String> {
        if self.container.is_none() {
            self.container = Some(
                self.rt
                    .block_on(
                        self.client
                            .database_client(&self.database)
                            .container_client(&self.container_name, None),
                    )
                    .map_err(|e| {
                        format!(
                            "Could not open Cosmos DB container '{}': {e}",
                            self.container_name
                        )
                    })?,
            );
        }
        self.container
            .as_ref()
            .ok_or_else(|| "Cosmos DB container is unavailable".into())
    }

    fn policy(&self, dataset: &Dataset) -> Result<ContainerProperties, String> {
        container_policy(
            self.container_name.clone(),
            self.index_type.clone(),
            dataset.distance(),
            dataset.vector_size(),
            self.build_options,
        )
    }

    /// Write every row and check the stored row count. Returns the write
    /// time and the row count.
    fn upload_corpus(&mut self, dataset: &Dataset) -> Result<(f64, usize), String> {
        let (ids, vectors, _) = dataset.read_vectors(dataset.needs_normalization())?;
        validate_corpus_size(&self.index_type, ids.len() as u64)?;
        let dimensions = dataset.vector_size() as usize;
        if vectors
            .iter()
            .any(|vector| vector.len() != dimensions || vector.iter().any(|v| !v.is_finite()))
        {
            return Err(
                "Cosmos DB corpus contains a vector with invalid dimensions or values".into(),
            );
        }
        let container = self
            .container
            .as_ref()
            .ok_or("Cosmos DB container is unavailable")?;
        let throttle_retries = AtomicUsize::new(0);
        let upload_start = Instant::now();
        self.rt.block_on(async {
            let throttle_retries = &throttle_retries;
            stream::iter(ids.iter().zip(&vectors))
                .map(|(id, vector)| async move {
                    let id = id.to_string();
                    let item = Item {
                        id: &id,
                        embedding: vector,
                    };
                    // The SDK gives up on a throttled write after its own
                    // retries. Back off and send it again.
                    retry_throttled(
                        || container.upsert_item(id.clone(), &id, item, None),
                        |error| u16::from(error.status().status_code()) == 429,
                        tokio::time::sleep,
                        throttle_retries,
                    )
                    .await
                    .map_err(|e| format!("Cosmos DB upload failed for row {id}: {e}"))
                })
                .buffer_unordered(self.upload_parallel)
                .try_for_each(|_| async { Ok(()) })
                .await
        })?;
        let upload_time = upload_start.elapsed().as_secs_f64();
        let upload_count = ids.len();
        drop((ids, vectors));
        let throttle_retries = throttle_retries.into_inner();
        if throttle_retries > 0 {
            eprintln!(
                "Cosmos DB upload at parallel={} resent {throttle_retries} throttled writes after the SDK's own HTTP 429 retries. Upload time includes that backoff. Raise the container's RU/s or lower upload_params.parallel to avoid it.",
                self.upload_parallel
            );
        }
        // The row-count scan is verification, so it stays out of both times.
        let count = self
            .corpus_row_count()?
            .ok_or("Cosmos DB did not return a row count")?;
        if count.rows != upload_count as u64 {
            return Err(format!(
                "Cosmos DB stored {} rows of {upload_count} uploaded",
                count.rows
            ));
        }
        Ok((upload_time, upload_count))
    }

    /// Wait until every feed range answers an index-requested query for
    /// clearly less than a forced scan of that range costs. Azure scans while
    /// a DiskANN index builds or a range holds too few vectors, so a matching
    /// row count does not show that searches will use the index.
    fn verify_index_use(&mut self, dataset: &Dataset) -> Result<(), String> {
        if self.index_type != VectorIndexType::DiskANN || self.index_probes.is_some() {
            return Ok(());
        }
        let Some(deadline) = self.index_probe_deadline else {
            eprintln!("Cosmos DB DiskANN index-use probe skipped (AZURE_COSMOS_INDEX_PROBE_SECONDS=0). Azure may answer from a scan.");
            return Ok(());
        };
        let (vectors, _, conditions) = dataset.read_queries()?;
        conditions.try_resolve_all("Cosmos DB", refuse_filter)?;
        let vector = vectors
            .first()
            .ok_or("Cosmos DB needs a query vector for the index-use probe")?;
        let scan = scan_query(vector, INDEX_PROBE_TOP)?;
        let indexed = knn_query(vector, INDEX_PROBE_TOP, None, None, false)?;
        let container = self.open_container()?.clone();
        let rt = &self.rt;
        let ranges = rt.block_on(read_search_ranges(&container))?;
        let started = Instant::now();
        // One forced scan per range is the costly part, so the ranges run
        // concurrently.
        let probes = await_index_use(
            ranges.len(),
            deadline,
            |query, pending| {
                let query = match query {
                    ProbeQuery::ForcedScan => &scan,
                    ProbeQuery::IndexRequested => &indexed,
                };
                rt.block_on(probe_charges(&container, query, &ranges, pending))
            },
            || started.elapsed(),
            std::thread::sleep,
        )?;
        if rt.block_on(read_search_ranges(&container))? != ranges {
            return Err("Cosmos DB feed ranges changed during the index-use probe".into());
        }
        let waited = started.elapsed().as_secs_f64();
        println!(
            "Cosmos DB DiskANN index use verified on {} feed ranges after {waited:.1}s",
            probes.len()
        );
        self.index_wait_seconds = Some(waited);
        self.index_probes = Some(probes);
        Ok(())
    }
}

impl Engine for CosmosDbEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let policy = self.policy(dataset)?;
        if let Some(rows) = dataset
            .config
            .vector_count
            .and_then(|n| u64::try_from(n).ok())
        {
            validate_corpus_size(&self.index_type, rows)?;
        }
        // Reject filter conditions before provisioning or uploading a corpus we
        // cannot search correctly. Use the same guard in search-only runs.
        let (_, _, conditions) = dataset.read_queries()?;
        conditions.try_resolve_all("Cosmos DB", refuse_filter)?;
        let database = self.client.database_client(&self.database);
        self.rt.block_on(async {
            match database.read(None).await {
                Ok(_) => {}
                Err(e) if u16::from(e.status().status_code()) == 404 => {
                    match self.client.create_database(&self.database, None).await {
                        Ok(_) => {}
                        Err(e) if u16::from(e.status().status_code()) == 409 => {}
                        Err(e) => return Err(format!("Could not create Cosmos DB database: {e}")),
                    }
                }
                Err(e) => return Err(format!("Could not read Cosmos DB database: {e}")),
            }
            let options = self.throughput.map(|rus| {
                CreateContainerOptions::default().with_throughput(ThroughputProperties::manual(rus))
            });
            match database.create_container(policy, options).await {
                Ok(_) => Ok(()),
                Err(e) if u16::from(e.status().status_code()) == 409 => Err(format!(
                    "Cosmos DB container '{}' already exists. An earlier failed run may have left it provisioned. Use --skip-upload for a complete corpus, or remove the container explicitly before uploading.",
                    self.container_name
                )),
                Err(e) => Err(format!("Could not create Cosmos DB container: {e}")),
            }
        })?;
        self.created_container = true;
        println!("Created Cosmos DB container '{}'", self.container_name);
        self.open_container()?;
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        // A partial corpus is of no use to a later run, and the container
        // bills until it is removed. A complete corpus is kept below even if
        // the index probe then fails, because `--skip-upload` can reuse it.
        let (upload_time, upload_count) = match self.upload_corpus(dataset) {
            Ok(uploaded) => uploaded,
            Err(error) if self.created_container => {
                let error = error.trim_end_matches('.');
                return Err(match self.delete() {
                    Ok(()) => format!(
                        "{error}. Removed container '{}', which this run created.",
                        self.container_name
                    ),
                    Err(delete) => format!(
                        "{error}. Container '{}' is still provisioned and billing: {delete}",
                        self.container_name
                    ),
                });
            }
            Err(error) => return Err(error),
        };
        // For DiskANN, total time adds the wait until the index serves queries.
        self.verify_index_use(dataset).map_err(|error| {
            format!("{error} The uploaded corpus is complete and was kept. Rerun with --skip-upload to search it.")
        })?;
        Ok(UploadStats {
            upload_time,
            total_time: upload_time + self.index_wait_seconds.unwrap_or(0.0),
            upload_count,
            parallel: self.upload_parallel,
            batch_size: 1,
            ..UploadStats::default()
        })
    }

    fn corpus_row_count(&mut self) -> Result<Option<CorpusCount>, String> {
        let container = self.open_container()?.clone();
        // SDK 1.0 rejects aggregates in every feed-range and full-container
        // query, so count the ID projection page by page. That keeps memory
        // bounded and never reads the stored vectors. The SDK drains a
        // full-container query one range at a time, so count the disjoint
        // feed ranges concurrently instead.
        let count = self.rt.block_on(count_stable_ranges(
            || read_search_ranges(&container),
            ROW_COUNT_CONCURRENCY,
            |range| count_ids(&container, FeedScope::range(range)),
        ))?;
        validate_corpus_size(&self.index_type, count)?;
        Ok(Some(CorpusCount::exact(count)))
    }

    fn check_index_layout(&mut self, dataset: &Dataset, layout: IndexLayout) -> Result<(), String> {
        if layout != IndexLayout::Vector {
            return Err("Cosmos DB requires the vector index layout".into());
        }
        let expected = self.policy(dataset)?;
        let container = self.open_container()?.clone();
        let actual = self
            .rt
            .block_on(container.read(None))
            .and_then(|r| r.into_model())
            .map_err(|e| format!("Could not read Cosmos DB container policy: {e}"))?;
        verify_policy(&actual, &expected)?;
        if let Some(rus) = self.throughput {
            let offer = self
                .rt
                .block_on(container.read_throughput(None))
                .map_err(|e| format!("Could not read Cosmos DB container throughput: {e}"))?;
            if !offer
                .is_some_and(|t| t.throughput() == Some(rus) && t.autoscale_maximum().is_none())
            {
                return Err(format!(
                    "Cosmos DB container does not have {rus} manual RU/s"
                ));
            }
        }
        self.verified_policy = Some(actual);
        self.verify_index_use(dataset)
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let (vectors, neighbors, conditions) = dataset.read_queries()?;
        conditions.try_resolve_all("Cosmos DB", refuse_filter)?;
        let limit = if num_queries > 0 {
            (num_queries as usize).min(vectors.len())
        } else {
            vectors.len()
        };
        if limit == 0 || vectors.len() != neighbors.len() {
            return Err("Cosmos DB needs a nonempty query set with aligned ground truth".into());
        }
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;
        let top = match params.top {
            Some(t) => usize::try_from(t).map_err(|_| "Cosmos DB top must be positive")?,
            None => neighbors[0].len(),
        };
        let search_list_size = diskann_multiplier(params, &self.index_type, SEARCH_LIST_SIZE)?;
        let quantized_vector_list =
            diskann_multiplier(params, &self.index_type, QUANTIZED_VECTOR_LIST)?;
        let partition_concurrency =
            PartitionConcurrency::from_config(params.knob(PARTITION_CONCURRENCY))?;

        // Validate every vector before timing. Workers bind each one to the
        // shared query text as they send it.
        let dimensions = dataset.vector_size() as usize;
        for vector in &vectors[..limit] {
            validate_query_vector(vector, dimensions)?;
        }
        let stock_text = KnnQueryText::new(top, search_list_size, quantized_vector_list, false)?;
        let text = KnnQueryText::new(
            top,
            search_list_size,
            quantized_vector_list,
            partition_concurrency.is_some(),
        )?;
        let text = &text;
        let container = self
            .container
            .as_ref()
            .ok_or("Cosmos DB container is unavailable")?;
        let rt = &self.rt;
        let partition_search = partition_concurrency.map(|concurrency| {
            let ranges = rt.block_on(read_search_ranges(container))?;
            let per_query_concurrency = concurrency.resolve(ranges.len());
            let shared_request_limit = parallel.max(per_query_concurrency);
            let summary = PartitionFanout {
                feed_ranges: ranges.len(),
                per_query_concurrency,
                shared_request_limit,
                parity: ParityTally::default(),
            };
            let mut search = PartitionSearch {
                ranges,
                summary,
                metric: vector_distance_function(dataset.distance())?,
                permits: Semaphore::new(shared_request_limit),
            };
            let parity = rt.block_on(sample_parity(
                parity_sample_indices(limit, PARITY_SAMPLES),
                PARITY_CONCURRENCY,
                |index| parity_sample(container, &stock_text, text, &vectors[index], top, &search),
            ))?;
            search.summary.parity = parity;
            println!(
                "Cosmos DB feed-range search verified against SDK on {} sampled queries: {} ranges, {} per query, {} in flight; fan-out missed {} stock IDs, repeated stock requests missed {}",
                parity.samples, summary.feed_ranges, summary.per_query_concurrency, summary.shared_request_limit,
                parity.cross_mode_missing_ids, parity.stock_repeat_missing_ids
            );
            Ok::<_, String>(search)
        }).transpose()?;
        let partition_search = partition_search.as_ref();
        let next = AtomicUsize::new(0);
        let (samples, elapsed) = std::thread::scope(|scope| {
            let mut pool = WorkerPool::new(scope, "cosmosdb-search", parallel);
            for worker in 0..parallel {
                let vectors = &vectors[..limit];
                let neighbors = &neighbors;
                let next = &next;
                pool.spawn(move |ticket| {
                    let mut samples = Vec::new();
                    // Warm the worker's first request outside the measured window.
                    if let Err(error) = rt.block_on(search_hits(
                        container,
                        text,
                        &vectors[worker % limit],
                        top,
                        partition_search,
                    )) {
                        ticket.fail(format!("Cosmos DB worker {worker} warm-up failed: {error}"));
                        return Ok(samples);
                    }
                    if ticket.arrive_and_wait().is_none() {
                        return Ok(samples);
                    }
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        let Some(vector) = vectors.get(idx) else {
                            break;
                        };
                        let started = Instant::now();
                        let (hits, diagnostics) = rt
                            .block_on(search_hits(container, text, vector, top, partition_search))
                            .map_err(|e| format!("Cosmos DB query {idx} failed: {e}"))?;
                        let latency = started.elapsed().as_secs_f64();
                        let ids = hits
                            .iter()
                            .map(|hit| hit.id.parse::<i64>())
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| {
                                format!("Cosmos DB query {idx} returned a non-integer id: {e}")
                            })?;
                        let metrics = vector_db_benchmark::metrics::compute_metrics(
                            &ids,
                            &neighbors[idx],
                            top,
                        );
                        samples.push((latency, metrics, diagnostics));
                    }
                    Ok::<_, String>(samples)
                })?;
            }
            let (batches, start) = pool.start()?;
            Ok::<_, String>((batches, start.elapsed().as_secs_f64()))
        })?;
        if let Some(search) = partition_search {
            if rt.block_on(read_search_ranges(container))? != search.ranges {
                return Err("Cosmos DB feed ranges changed during the measured search".into());
            }
        }
        let (mut times, mut precision, mut recall, mut mrr, mut ndcg) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut diagnostics = SearchDiagnostics::default();
        for batch in samples {
            for (latency, m, query_diagnostics) in batch? {
                times.push(latency);
                precision.push(m.precision);
                recall.push(m.recall);
                mrr.push(m.mrr);
                ndcg.push(m.ndcg);
                diagnostics.add_query(query_diagnostics);
            }
        }
        diagnostics.partition_fanout = partition_search.map(|search| search.summary);
        if diagnostics.http_429_attempts > 0 {
            eprintln!(
                "Cosmos DB search at parallel={parallel} observed {} HTTP 429 responses. Latency includes SDK backoff and retries.",
                diagnostics.http_429_attempts
            );
        }
        let mut results = compute_search_stats(
            &times, &precision, &recall, &mrr, &ndcg, elapsed, top, parallel, limit,
        )?;
        results.cosmosdb_search_diagnostics = Some(diagnostics);
        Ok(results)
    }

    fn server_metadata(&mut self) -> Option<serde_json::Value> {
        let policy = self.verified_policy.as_ref()?;
        Some(serde_json::json!({
            "cosmosdb": {
                "database": self.database,
                "container": self.container_name,
                "partition_key": policy.partition_key,
                "vector_embedding_policy": policy.vector_embedding_policy,
                "vector_indexing_policy": policy.indexing_policy,
                "manual_ru_per_second": self.throughput,
                "vector_index_use_verified": self.index_probes.is_some(),
                "vector_index_use_probes": self.index_probes,
                "vector_index_wait_seconds": self.index_wait_seconds,
            }
        }))
    }

    fn delete(&mut self) -> Result<(), String> {
        if self.created_container {
            let container = self
                .container
                .as_ref()
                .ok_or("Cosmos DB container is unavailable")?;
            self.rt
                .block_on(container.delete(None))
                .map_err(|e| format!("Could not delete Cosmos DB container: {e}"))?;
            self.created_container = false;
            self.container = None;
        }
        Ok(())
    }
}

struct PartitionSearch {
    ranges: Vec<FeedRange>,
    summary: PartitionFanout,
    metric: VectorDistanceFunction,
    permits: Semaphore,
}

/// One sampled query under both routing modes. The stock query runs twice
/// because repeated DiskANN requests can differ at the cutoff, and that
/// variation is the bar the fan-out result is held to.
async fn parity_sample(
    container: &ContainerClient,
    stock_text: &KnnQueryText,
    partition_text: &KnnQueryText,
    vector: &[f32],
    top: usize,
    search: &PartitionSearch,
) -> Result<[Vec<QueryHit>; 3], String> {
    let ((stock, _), (stock_repeat, _), (partitioned, _)) = futures::try_join!(
        search_hits(container, stock_text, vector, top, None),
        search_hits(container, stock_text, vector, top, None),
        search_hits(container, partition_text, vector, top, Some(search)),
    )?;
    Ok([stock, stock_repeat, partitioned])
}

/// Request charge of `query` on each listed feed range, as (range index, RU).
async fn probe_charges(
    container: &ContainerClient,
    query: &Query,
    ranges: &[FeedRange],
    indices: &[usize],
) -> Result<Vec<(usize, f64)>, String> {
    stream::iter(indices.iter().copied())
        .map(|feed_range| async move {
            let scope = FeedScope::range(ranges[feed_range].clone());
            query_hits(container, query.clone(), INDEX_PROBE_TOP, scope)
                .await
                .map(|(_, diagnostics)| (feed_range, diagnostics.request_charge_ru))
                .map_err(|e| format!("Cosmos DB index-use probe failed: {e}"))
        })
        .buffer_unordered(INDEX_PROBE_CONCURRENCY)
        .try_collect()
        .await
}

async fn count_ids(container: &ContainerClient, scope: FeedScope) -> Result<u64, String> {
    let options = QueryOptions::default().with_max_item_count(MaxItemCountHint::ServerDecides);
    container
        .query_items::<String>("SELECT VALUE c.id FROM c", scope, Some(options))
        .await
        .map_err(|e| format!("Cosmos DB row count failed: {e}"))?
        .into_pages()
        .try_fold(0_u64, |count, page| async move {
            Ok(count + page.items().len() as u64)
        })
        .await
        .map_err(|e| format!("Cosmos DB row count failed: {e}"))
}

async fn read_search_ranges(container: &ContainerClient) -> Result<Vec<FeedRange>, String> {
    let ranges = container
        .read_feed_ranges(Some(
            ReadFeedRangesOptions::default().with_force_refresh(true),
        ))
        .await
        .map_err(|e| format!("Could not read Cosmos DB feed ranges: {e}"))?;
    if ranges.is_empty() {
        return Err("Cosmos DB returned no feed ranges".into());
    }
    Ok(ranges)
}

async fn search_hits(
    container: &ContainerClient,
    text: &KnnQueryText,
    vector: &[f32],
    top: usize,
    partition_search: Option<&PartitionSearch>,
) -> Result<(Vec<QueryHit>, SearchDiagnostics), String> {
    let Some(search) = partition_search else {
        return query_hits(
            container,
            text.bind(vector)?,
            top,
            FeedScope::full_container(),
        )
        .await;
    };
    fan_out_hits(
        &search.ranges,
        search.summary.per_query_concurrency,
        &search.permits,
        top,
        &search.metric,
        |range| async move {
            query_hits(container, text.bind(vector)?, top, FeedScope::range(range)).await
        },
    )
    .await
}

async fn query_hits(
    container: &ContainerClient,
    query: Query,
    top: usize,
    scope: FeedScope,
) -> Result<(Vec<QueryHit>, SearchDiagnostics), String> {
    let mut pages = container
        .query_items::<QueryHit>(query, scope, None)
        .await
        .map_err(|error| error.to_string())?
        .into_pages();
    let mut hits = Vec::new();
    let mut combined = SearchDiagnostics::default();
    // Enforce TOP K within the given scope across all SDK pages.
    while let Some(page) = pages.try_next().await.map_err(|error| error.to_string())? {
        validate_hit_count(hits.len() + page.items().len(), top)?;
        let diagnostics = page.diagnostics();
        let charge = diagnostics.total_request_charge().value();
        let attempts = diagnostics.request_count();
        if let Some(compacted) = diagnostics.compaction() {
            combined.observe_page(
                charge,
                attempts,
                compacted
                    .runs
                    .iter()
                    .map(|run| (u16::from(run.status.status_code()), run.count)),
            );
        } else {
            combined.observe_page(
                charge,
                attempts,
                diagnostics
                    .requests()
                    .iter()
                    .map(|request| (u16::from(request.status().status_code()), 1)),
            );
        }
        hits.extend(page.into_items());
    }
    Ok((hits, combined))
}

fn refuse_filter(_: &serde_json::Value) -> Result<Option<()>, String> {
    Err("filtered vector search is unsupported".into())
}
