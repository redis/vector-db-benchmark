//! Azure Cosmos DB for NoSQL adapter using the official Rust SDK.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use azure_data_cosmos::models::{ContainerProperties, ThroughputProperties, VectorIndexType};
use azure_data_cosmos::options::{
    ContentResponseOnWrite, CreateContainerOptions, MaxItemCountHint, OperationOptionsBuilder,
    QueryOptions,
};
use azure_data_cosmos::{
    AccountReference, ContainerClient, CosmosClient, CosmosRuntime, FeedScope, Query,
    RoutingStrategy,
};
use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use vector_db_benchmark::cosmosdb::{
    container_name, container_policy, index_type, knn_query, validate_corpus_size,
    validate_hit_count, validate_top, verify_policy, SearchDiagnostics,
};
use vector_db_benchmark::start_gate::WorkerPool;
use vector_db_benchmark::workload::IndexLayout;

use super::{compute_search_stats, CorpusCount, Engine, SearchResults, UploadStats};
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;

#[derive(Serialize)]
struct Item<'a> {
    id: &'a str,
    embedding: &'a [f32],
}

#[derive(Deserialize)]
struct Hit {
    id: String,
}

const SEARCH_LIST_SIZE: &str = "search_list_size_multiplier";
const QUANTIZED_VECTOR_LIST: &str = "quantized_vector_list_multiplier";

fn diskann_multiplier(
    params: &SearchParams,
    kind: &VectorIndexType,
    key: &str,
) -> Result<Option<NonZeroU32>, String> {
    params
        .knob(key)
        .map(|value| {
            if *kind != VectorIndexType::DiskANN {
                return Err(format!("Cosmos DB {key} requires DiskANN"));
            }
            value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .and_then(NonZeroU32::new)
                .ok_or_else(|| format!("Cosmos DB {key} must be a positive 32-bit integer"))
        })
        .transpose()
}

pub struct CosmosDbEngine {
    name: String,
    database: String,
    container_name: String,
    index_type: VectorIndexType,
    upload_parallel: usize,
    search_params: Vec<SearchParams>,
    throughput: Option<u64>,
    client: CosmosClient,
    rt: tokio::runtime::Runtime,
    container: Option<ContainerClient>,
    created_container: bool,
    verified_policy: Option<ContainerProperties>,
}

impl CosmosDbEngine {
    pub fn new(config: &EngineConfig, dataset: &str) -> Result<Self, String> {
        let kind = index_type(config.algorithm.as_deref())?;
        let search_params = config.search_params.clone().unwrap_or_default();
        for params in &search_params {
            if let Some(top) = params.top {
                validate_top(usize::try_from(top).map_err(|_| "Cosmos DB top must be positive")?)?;
            }
            let unsupported =
                |key: &String| !matches!(key.as_str(), SEARCH_LIST_SIZE | QUANTIZED_VECTOR_LIST);
            if params.calibration_param.is_some()
                || params.num_candidates.is_some()
                || params
                    .extra
                    .as_ref()
                    .is_some_and(|p| p.keys().any(unsupported))
                || params.search_params.as_ref().is_some_and(|p| {
                    p.ef.is_some()
                        || p.extra
                            .as_ref()
                            .is_some_and(|extra| extra.keys().any(unsupported))
                })
            {
                return Err(
                    "Cosmos DB supports only DiskANN search multipliers as tuning knobs".into(),
                );
            }
            diskann_multiplier(params, &kind, SEARCH_LIST_SIZE)?;
            diskann_multiplier(params, &kind, QUANTIZED_VECTOR_LIST)?;
        }
        let collection = config.collection_params.as_ref();
        if collection.is_some_and(|p| {
            p.hnsw_config.is_some()
                || p.index_options.is_some()
                || p.extra
                    .as_ref()
                    .is_some_and(|extra| extra.keys().any(|k| k != "ru_per_second"))
        }) {
            return Err("Cosmos DB collection_params supports only ru_per_second".into());
        }
        let configured_ru = collection
            .and_then(|p| p.extra.as_ref())
            .and_then(|options| options.get("ru_per_second"))
            .map(|value| {
                value.as_u64().filter(|ru| *ru >= 400).ok_or(
                    "Cosmos DB collection_params.ru_per_second must be an integer of at least 400",
                )
            })
            .transpose()?;
        let upload_parallel = match config.upload_params.as_ref() {
            None => 8,
            Some(params) => {
                let options = params
                    .as_object()
                    .ok_or("Cosmos DB upload_params must be an object")?;
                if options.keys().any(|k| k != "parallel") {
                    return Err(
                        "Cosmos DB upload_params supports only parallel (one item per request)"
                            .into(),
                    );
                }
                match options.get("parallel") {
                    None => 8,
                    Some(p) => p
                        .as_u64()
                        .and_then(|p| usize::try_from(p).ok())
                        .filter(|p| *p > 0)
                        .ok_or("Cosmos DB upload parallel must be positive")?,
                }
            }
        };

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
        // The SDK consumes this environment setting when the runtime starts.
        let _ = crate::effective_config::env_opt("AZURE_COSMOS_MAX_THROTTLE_RETRY_COUNT");
        let throughput = if let Some(ru) = configured_ru {
            Some(ru)
        } else {
            match crate::effective_config::env_or("AZURE_COSMOS_RU_PER_SECOND", "400").as_str() {
                "serverless" => None,
                value => {
                    Some(
                        value.parse::<u64>().ok().filter(|ru| *ru >= 400).ok_or(
                            "AZURE_COSMOS_RU_PER_SECOND must be at least 400, or serverless",
                        )?,
                    )
                }
            }
        };
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
            upload_parallel,
            search_params,
            throughput,
            client,
            rt,
            container: None,
            created_container: false,
            verified_policy: None,
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
        )
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
        let total_start = Instant::now();
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
        let upload_start = Instant::now();
        self.rt.block_on(async {
            stream::iter(ids.iter().zip(&vectors))
                .map(|(id, vector)| async move {
                    let id = id.to_string();
                    let item = Item {
                        id: &id,
                        embedding: vector,
                    };
                    container
                        .upsert_item(id.clone(), &id, item, None)
                        .await
                        .map_err(|e| format!("Cosmos DB upload failed for row {id}: {e}"))
                })
                .buffer_unordered(self.upload_parallel)
                .try_for_each(|_| async { Ok(()) })
                .await
        })?;
        let upload_time = upload_start.elapsed().as_secs_f64();
        let count = self
            .corpus_row_count()?
            .ok_or("Cosmos DB did not return a row count")?;
        if count.rows != ids.len() as u64 {
            return Err(format!(
                "Cosmos DB stored {} rows of {} uploaded",
                count.rows,
                ids.len()
            ));
        }
        Ok(UploadStats {
            upload_time,
            total_time: total_start.elapsed().as_secs_f64(),
            upload_count: ids.len(),
            parallel: self.upload_parallel,
            batch_size: 1,
            ..UploadStats::default()
        })
    }

    fn corpus_row_count(&mut self) -> Result<Option<CorpusCount>, String> {
        let container = self.open_container()?.clone();
        // SDK 1.0 does not implement cross-partition aggregate queries on Azure.
        // Count the ID projection page by page so reuse has an exact count with
        // bounded memory, without reading the stored vectors.
        let count = self
            .rt
            .block_on(async {
                let options =
                    QueryOptions::default().with_max_item_count(MaxItemCountHint::ServerDecides);
                container
                    .query_items::<String>(
                        "SELECT VALUE c.id FROM c",
                        FeedScope::full_container(),
                        Some(options),
                    )
                    .await?
                    .into_pages()
                    .try_fold(0_u64, |count, page| async move {
                        Ok(count + page.items().len() as u64)
                    })
                    .await
            })
            .map_err(|e| format!("Cosmos DB row count failed: {e}"))?;
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
        if self.index_type == VectorIndexType::DiskANN {
            eprintln!("Cosmos DB DiskANN policy verified. Index use is unverified; Azure may fall back to a scan.");
        }
        Ok(())
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

        let queries = vectors[..limit]
            .iter()
            .map(|vector| {
                if vector.len() != dataset.vector_size() as usize {
                    return Err("Cosmos DB query vector has the wrong number of dimensions".into());
                }
                knn_query(vector, top, search_list_size, quantized_vector_list)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let container = self
            .container
            .as_ref()
            .ok_or("Cosmos DB container is unavailable")?;
        let rt = &self.rt;
        let next = AtomicUsize::new(0);
        let (samples, elapsed) = std::thread::scope(|scope| {
            let mut pool = WorkerPool::new(scope, "cosmosdb-search", parallel);
            for worker in 0..parallel {
                let queries = &queries;
                let neighbors = &neighbors;
                let next = &next;
                pool.spawn(move |ticket| {
                    let mut samples = Vec::new();
                    // Warm the worker's first request outside the measured window.
                    if let Err(error) =
                        rt.block_on(query_hits(container, &queries[worker % limit], top))
                    {
                        ticket.fail(format!("Cosmos DB worker {worker} warm-up failed: {error}"));
                        return Ok(samples);
                    }
                    if ticket.arrive_and_wait().is_none() {
                        return Ok(samples);
                    }
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        let Some(query) = queries.get(idx) else { break };
                        let started = Instant::now();
                        let (hits, diagnostics) = rt
                            .block_on(query_hits(container, query, top))
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
                "vector_index_use_verified": false,
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

async fn query_hits(
    container: &ContainerClient,
    query: &Query,
    top: usize,
) -> Result<(Vec<Hit>, SearchDiagnostics), String> {
    let mut pages = container
        .query_items::<Hit>(query.clone(), FeedScope::full_container(), None)
        .await
        .map_err(|error| error.to_string())?
        .into_pages();
    let mut hits = Vec::new();
    let mut combined = SearchDiagnostics::default();
    // The SDK merges ORDER BY across partitions. Check the resulting global
    // limit anyway, including pages where the server sends no hits.
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
