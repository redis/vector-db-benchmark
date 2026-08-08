//! Qdrant engine implementation.
//!
//! Uses the official `qdrant-client` crate with gRPC transport.
//! Wraps async calls with a tokio runtime (block_on) since the
//! benchmark Engine trait is synchronous.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    BinaryQuantization, CompressionRatio, Condition, CreateCollectionBuilder, DatetimeRange,
    DeleteCollectionBuilder, Distance, FieldType, Filter, Fusion, HnswConfigDiff,
    MaxOptimizationThreads, NamedVectors, OptimizersConfigDiff, PointStruct, PrefetchQueryBuilder,
    ProductQuantization, QuantizationSearchParams, QuantizationType, Query, QueryPointsBuilder,
    ScalarQuantization, SearchParams as QdrantSearchParams, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, Timestamp, Vector, VectorInput, VectorParamsBuilder, VectorsConfig,
    VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_COLLECTION: &str = "benchmark";

pub struct QdrantEngine {
    name: String,
    collection_name: String,
    #[allow(dead_code)]
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    #[allow(dead_code)]
    grpc_url: String,
    /// REST base URL (e.g. http://host:6333) for the /metrics and /telemetry endpoints
    rest_url: String,
    api_key: Option<String>,
    search_params: Vec<SearchParams>,
    /// Raw collection_params JSON to pass through to Qdrant
    collection_params_extra: serde_json::Value,
    hnsw_m: Option<u64>,
    hnsw_ef_construct: Option<u64>,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared Qdrant client (wrapped in Arc for thread-safe sharing)
    client: Arc<Qdrant>,
}

impl QdrantEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let grpc_port: u16 = std::env::var("QDRANT_GRPC_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6334);

        let collection_name = std::env::var("QDRANT_COLLECTION_NAME")
            .unwrap_or_else(|_| DEFAULT_COLLECTION.to_string());

        let api_key = std::env::var("QDRANT_API_KEY").ok();

        let timeout = engine_config
            .connection_params
            .as_ref()
            .and_then(|p| p.get("timeout"))
            .and_then(|v| v.as_u64())
            .unwrap_or(300);

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1024) as usize;

        // Determine the host without scheme
        let clean_host = host
            .trim_start_matches("http://")
            .trim_start_matches("https://");

        let grpc_url = if let Ok(url) = std::env::var("QDRANT_URL") {
            url
        } else {
            format!("http://{}:{}", clean_host, grpc_port)
        };

        // REST endpoint (default port 6333) for /metrics and /telemetry. Overridable
        // via QDRANT_REST_URL, or QDRANT_REST_PORT for just the port.
        let rest_url = if let Ok(url) = std::env::var("QDRANT_REST_URL") {
            url.trim_end_matches('/').to_string()
        } else {
            let rest_port: u16 = std::env::var("QDRANT_REST_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6333);
            format!("http://{}:{}", clean_host, rest_port)
        };

        let collection_params_extra = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.extra.as_ref())
            .map(|e| serde_json::to_value(e).unwrap_or_default())
            .unwrap_or(serde_json::json!({}));

        let typed_hnsw = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref());
        let hnsw_m = typed_hnsw.and_then(|h| h.m).map(|v| v as u64);
        let hnsw_ef_construct = typed_hnsw.and_then(|h| h.ef_construction).map(|v| v as u64);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = rt
            .block_on(async {
                let mut builder =
                    Qdrant::from_url(&grpc_url).timeout(std::time::Duration::from_secs(timeout));
                if let Some(key) = &api_key {
                    builder = builder.api_key(key.clone());
                }
                builder.build()
            })
            .map_err(|e| format!("Failed to create Qdrant client: {}", e))?;

        Ok(Self {
            name: engine_config.name.clone(),
            collection_name,
            timeout,
            batch_size,
            parallel,
            grpc_url,
            rest_url,
            api_key,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            collection_params_extra,
            hnsw_m,
            hnsw_ef_construct,
            rt,
            client: Arc::new(client),
        })
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

    fn delete_collection(&self) -> Result<(), String> {
        let _ = self.rt.block_on(
            self.client
                .delete_collection(DeleteCollectionBuilder::new(&self.collection_name)),
        );
        Ok(())
    }

    fn create_collection(&self, dataset: &Dataset) -> Result<(), String> {
        if dataset.is_hybrid() {
            return self.create_hybrid_collection(dataset);
        }
        if dataset.is_sparse() {
            return self.create_sparse_collection(dataset);
        }

        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let qdrant_distance = map_qdrant_distance(distance)?;

        // HNSW params come from the TYPED collection_params.hnsw_config field
        // (serde captures "m"/"ef_construct" there via aliases; the flattened
        // `extra` map never contains hnsw_config since it is a declared field).
        let hnsw_m = self.hnsw_m;
        let hnsw_ef = self.hnsw_ef_construct;

        // Optionally store vectors on disk (mmap) — collection_params.vectors_config.on_disk.
        let vectors_on_disk = self
            .collection_params_extra
            .get("vectors_config")
            .and_then(|v| v.get("on_disk"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let vector_params =
            VectorParamsBuilder::new(vector_size as u64, qdrant_distance).on_disk(vectors_on_disk);

        let mut create_builder = CreateCollectionBuilder::new(&self.collection_name)
            .vectors_config(VectorsConfig {
                config: Some(Config::Params(vector_params.build())),
            });

        // Apply HNSW config if specified
        if hnsw_m.is_some() || hnsw_ef.is_some() {
            let mut hnsw_config = HnswConfigDiff::default();
            if let Some(m) = hnsw_m {
                hnsw_config.m = Some(m);
            }
            if let Some(ef) = hnsw_ef {
                hnsw_config.ef_construct = Some(ef);
            }
            create_builder = create_builder.hnsw_config(hnsw_config);
        }

        // Pass through optimizers_config + quantization_config (shared with the
        // hybrid create path).
        create_builder = self.apply_optimizers_and_quantization(create_builder)?;

        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        // Disable optimization during indexing.
        self.disable_indexing_optimizers();

        self.create_payload_indexes(dataset);

        Ok(())
    }

    /// Apply `collection_params.optimizers_config` (rps-tuned segment / memmap
    /// knobs) and `collection_params.quantization_config` (scalar/binary) to a
    /// `CreateCollectionBuilder`. Shared verbatim by the dense-only and hybrid
    /// create paths so the hybrid collection honours the SAME tuning (e.g.
    /// qdrant-hybrid.json's `memmap_threshold`).
    fn apply_optimizers_and_quantization(
        &self,
        mut create_builder: CreateCollectionBuilder,
    ) -> Result<CreateCollectionBuilder, String> {
        if let Some(opt) = self.collection_params_extra.get("optimizers_config") {
            let mut diff = OptimizersConfigDiff::default();
            if let Some(v) = opt.get("default_segment_number").and_then(|v| v.as_u64()) {
                diff.default_segment_number = Some(v);
            }
            if let Some(v) = opt.get("max_segment_size").and_then(|v| v.as_u64()) {
                diff.max_segment_size = Some(v);
            }
            if let Some(v) = opt.get("memmap_threshold").and_then(|v| v.as_u64()) {
                diff.memmap_threshold = Some(v);
            }
            create_builder = create_builder.optimizers_config(diff);
        }

        if let Some(q) = self.collection_params_extra.get("quantization_config") {
            if let Some(quantization) = build_quantization(q)? {
                create_builder = create_builder.quantization_config(quantization);
            }
        }
        Ok(create_builder)
    }

    /// Throttle optimizer threads to 0 during bulk indexing (re-enabled to auto
    /// by `wait_collection_green`). Shared by the dense and hybrid create paths.
    fn disable_indexing_optimizers(&self) {
        let _ = self.rt.block_on(
            self.client.update_collection(
                qdrant_client::qdrant::UpdateCollectionBuilder::new(&self.collection_name)
                    .optimizers_config(OptimizersConfigDiff {
                        max_optimization_threads: Some(MaxOptimizationThreads {
                            variant: Some(
                                qdrant_client::qdrant::max_optimization_threads::Variant::Value(0),
                            ),
                        }),
                        ..Default::default()
                    }),
            ),
        );
    }

    /// Create Qdrant payload indexes for the dataset's schema fields.
    fn create_payload_indexes(&self, dataset: &Dataset) {
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let qdrant_type = match ft {
                        "int" => FieldType::Integer,
                        "keyword" => FieldType::Keyword,
                        "text" => FieldType::Text,
                        "float" => FieldType::Float,
                        "geo" => FieldType::Geo,
                        "uuid" => FieldType::Uuid,
                        // Bools are stored as the STRING "true"/"false" (readers::metadata
                        // has no Bool variant), so index them as Keyword — a Bool index
                        // would index nothing for a string payload and the filter would
                        // silently match zero points.
                        "bool" => FieldType::Keyword,
                        "datetime" => FieldType::Datetime,
                        _ => continue,
                    };
                    let _ = self.rt.block_on(self.client.create_field_index(
                        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
                            &self.collection_name,
                            field_name.clone(),
                            qdrant_type,
                        ),
                    ));
                }
            }
        }
    }

    /// Create a sparse-vector collection with a single named "sparse" vector.
    fn create_sparse_collection(&self, dataset: &Dataset) -> Result<(), String> {
        let mut sparse_cfg = SparseVectorsConfigBuilder::default();
        sparse_cfg.add_named_vector_params("sparse", SparseVectorParamsBuilder::default());
        let create_builder =
            CreateCollectionBuilder::new(&self.collection_name).sparse_vectors_config(sparse_cfg);
        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create sparse collection: {}", e))?;
        self.create_payload_indexes(dataset);
        Ok(())
    }

    /// Create a HYBRID collection with a named dense vector ("dense") AND a named
    /// sparse vector ("sparse"), so searches can fuse a dense-vector prefetch and
    /// a sparse-vector prefetch server-side (RRF). The dense vector carries the
    /// dataset's distance metric (and HNSW config, if configured); the sparse
    /// vector uses Qdrant's default sparse index.
    fn create_hybrid_collection(&self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();
        let qdrant_distance = map_qdrant_distance(distance)?;

        // Named dense vector "dense" (per-vector HNSW config if requested).
        let mut dense_params = VectorParamsBuilder::new(vector_size as u64, qdrant_distance);
        if self.hnsw_m.is_some() || self.hnsw_ef_construct.is_some() {
            let mut hnsw_config = HnswConfigDiff::default();
            if let Some(m) = self.hnsw_m {
                hnsw_config.m = Some(m);
            }
            if let Some(ef) = self.hnsw_ef_construct {
                hnsw_config.ef_construct = Some(ef);
            }
            dense_params = dense_params.hnsw_config(hnsw_config);
        }
        let mut dense_cfg = VectorsConfigBuilder::default();
        dense_cfg.add_named_vector_params("dense", dense_params);

        // Named sparse vector "sparse" (mirrors create_sparse_collection).
        let mut sparse_cfg = SparseVectorsConfigBuilder::default();
        sparse_cfg.add_named_vector_params("sparse", SparseVectorParamsBuilder::default());

        let mut create_builder = CreateCollectionBuilder::new(&self.collection_name)
            .vectors_config(dense_cfg)
            .sparse_vectors_config(sparse_cfg);
        // Honour the SAME optimizers_config / quantization_config tuning as the
        // dense-only path (finding: hybrid previously dropped e.g. memmap_threshold).
        create_builder = self.apply_optimizers_and_quantization(create_builder)?;

        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create hybrid collection: {}", e))?;

        // Throttle optimizer threads during indexing, same as the dense path.
        self.disable_indexing_optimizers();

        self.create_payload_indexes(dataset);
        Ok(())
    }

    /// Upload points carrying BOTH named vectors ("dense" + "sparse"), batched.
    fn upload_hybrid(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();
        let dataset_path = dataset.get_path()?;
        println!("Reading hybrid dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, dense, sparse) = dataset.read_hybrid_data(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();
        println!("Read {} hybrid vectors in {:.3}s", ids.len(), read_time);

        println!("Starting hybrid upload, batch size {}...", self.batch_size);
        let upload_start = Instant::now();
        let pb = self.create_progress_bar(ids.len());
        for start in (0..ids.len()).step_by(self.batch_size) {
            let end = (start + self.batch_size).min(ids.len());
            let points: Vec<PointStruct> = (start..end)
                .map(|i| {
                    PointStruct::new(
                        ids[i] as u64,
                        NamedVectors::default()
                            .add_vector("dense", dense[i].clone())
                            .add_vector(
                                "sparse",
                                Vector::new_sparse(
                                    sparse[i].indices.clone(),
                                    sparse[i].values.clone(),
                                ),
                            ),
                        Payload::new(),
                    )
                })
                .collect();
            self.rt
                .block_on(
                    self.client.upsert_points(
                        qdrant_client::qdrant::UpsertPointsBuilder::new(
                            &self.collection_name,
                            points,
                        )
                        .wait(true),
                    ),
                )
                .map_err(|e| format!("Hybrid upsert failed: {}", e))?;
            pb.inc((end - start) as u64);
        }
        pb.finish_with_message("Upload complete");
        let upload_time = upload_start.elapsed().as_secs_f64();
        let index_start = Instant::now();
        self.wait_collection_green()?;
        let index_time = index_start.elapsed().as_secs_f64();

        Ok(UploadStats {
            upload_time,
            total_time: read_time + upload_time + index_time,
            upload_count: ids.len(),
            parallel: 1,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    /// Upload sparse vectors under the named "sparse" vector, batched.
    fn upload_sparse(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let dataset_path = dataset.get_path()?;
        println!("Reading sparse dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors) = dataset.read_sparse_data()?;
        let read_time = read_start.elapsed().as_secs_f64();
        println!("Read {} sparse vectors in {:.3}s", vectors.len(), read_time);

        println!("Starting sparse upload, batch size {}...", self.batch_size);
        let upload_start = Instant::now();
        let pb = self.create_progress_bar(ids.len());
        for start in (0..ids.len()).step_by(self.batch_size) {
            let end = (start + self.batch_size).min(ids.len());
            let points: Vec<PointStruct> = (start..end)
                .map(|i| {
                    PointStruct::new(
                        ids[i] as u64,
                        NamedVectors::default().add_vector(
                            "sparse",
                            Vector::new_sparse(
                                vectors[i].indices.clone(),
                                vectors[i].values.clone(),
                            ),
                        ),
                        Payload::new(),
                    )
                })
                .collect();
            self.rt
                .block_on(
                    self.client.upsert_points(
                        qdrant_client::qdrant::UpsertPointsBuilder::new(
                            &self.collection_name,
                            points,
                        )
                        .wait(true),
                    ),
                )
                .map_err(|e| format!("Sparse upsert failed: {}", e))?;
            pb.inc((end - start) as u64);
        }
        pb.finish_with_message("Upload complete");
        let upload_time = upload_start.elapsed().as_secs_f64();
        // Include the index-build wait in total_time (see upload()).
        let index_start = Instant::now();
        self.wait_collection_green()?;
        let index_time = index_start.elapsed().as_secs_f64();

        Ok(UploadStats {
            upload_time,
            total_time: read_time + upload_time + index_time,
            upload_count: vectors.len(),
            parallel: 1,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
        schema_types: &HashMap<String, String>,
    ) -> Result<(), String> {
        use vector_db_benchmark::readers::metadata::MetadataValue;

        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let client = Arc::clone(&self.client);
        let collection_name = self.collection_name.clone();

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let client = Arc::clone(&client);
                let collection_name = collection_name.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                        Ok(rt) => rt,
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
                        if error.lock().unwrap().is_some() {
                            break;
                        }

                        let (batch_start, batch_end) = batches[idx];
                        let mut points = Vec::with_capacity(batch_end - batch_start);

                        for i in batch_start..batch_end {
                            let mut payload = Payload::new();
                            if let Some(meta) = &metadata[i] {
                                for (k, v) in &meta.fields {
                                    // A numeric value under a keyword-declared field
                                    // must stay a string, or the keyword index won't
                                    // match it (integer payload vs string condition).
                                    let v = v.coerce_for_schema(
                                        schema_types.get(k).map(|s| s.as_str()),
                                    );
                                    match v.as_ref() {
                                        MetadataValue::String(s) => {
                                            payload.insert(k.clone(), s.clone());
                                        }
                                        MetadataValue::Int(n) => {
                                            payload.insert(k.clone(), *n);
                                        }
                                        MetadataValue::Float(f) => {
                                            payload.insert(k.clone(), *f);
                                        }
                                        MetadataValue::Labels(labels) => {
                                            let arr: Vec<qdrant_client::qdrant::Value> =
                                                labels.iter().map(|l| l.clone().into()).collect();
                                            payload.insert(
                                                k.clone(),
                                                qdrant_client::qdrant::Value {
                                                    kind: Some(
                                                        qdrant_client::qdrant::value::Kind::ListValue(
                                                            qdrant_client::qdrant::ListValue {
                                                                values: arr,
                                                            },
                                                        ),
                                                    ),
                                                },
                                            );
                                        }
                                        MetadataValue::Geo { lon, lat } => {
                                            let mut geo_payload = Payload::new();
                                            geo_payload.insert("lon", *lon);
                                            geo_payload.insert("lat", *lat);
                                            payload.insert(
                                                k.clone(),
                                                qdrant_client::qdrant::Value::from(
                                                    serde_json::json!({"lon": lon, "lat": lat}),
                                                ),
                                            );
                                        }
                                    };
                                }
                            }

                            points.push(PointStruct::new(
                                ids[i] as u64,
                                vectors[i].clone(),
                                payload,
                            ));
                        }

                        let result = rt.block_on(client.upsert_points(
                            qdrant_client::qdrant::UpsertPointsBuilder::new(
                                &collection_name,
                                points,
                            )
                            .wait(false),
                        ));

                        if let Err(e) = result {
                            *error.lock().unwrap() = Some(format!("Upsert failed: {}", e));
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

    fn wait_collection_green(&self) -> Result<(), String> {
        println!("Waiting for collection to be GREEN...");

        // Re-enable optimization (auto mode)
        let _ = self.rt.block_on(
            self.client.update_collection(
                qdrant_client::qdrant::UpdateCollectionBuilder::new(&self.collection_name)
                    .optimizers_config(OptimizersConfigDiff {
                        max_optimization_threads: Some(MaxOptimizationThreads {
                            variant: Some(
                                qdrant_client::qdrant::max_optimization_threads::Variant::Setting(
                                    qdrant_client::qdrant::max_optimization_threads::Setting::Auto
                                        as i32,
                                ),
                            ),
                        }),
                        ..Default::default()
                    }),
            ),
        );

        // Index build can take hours on large datasets (e.g. deep-image-96 has
        // ~10M vectors); the old fixed 50-min cap (600 * 5s) was too short for
        // high-M / high-ef_construct configs and silently aborted the whole run.
        // Make the budget configurable (QDRANT_GREEN_WAIT_SECS) with a generous
        // default, and log indexing progress so a slow build is visible and
        // distinguishable from a genuinely stuck one.
        let green_wait_secs: u64 = std::env::var("QDRANT_GREEN_WAIT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14400); // 4h
        let iterations = (green_wait_secs / 5).max(1);
        for i in 0..iterations {
            std::thread::sleep(std::time::Duration::from_secs(5));

            if let Ok(info) = self
                .rt
                .block_on(self.client.collection_info(&self.collection_name))
            {
                // status: 1 = Green, 2 = Yellow, 3 = Red (from protobuf enum)
                if let Some(result) = info.result {
                    // Progress heartbeat every ~60s so long builds aren't opaque.
                    if i % 12 == 11 {
                        println!(
                            "  ...waiting for GREEN: status={} indexed={}/{} ({}s / {}s budget)",
                            result.status,
                            result.indexed_vectors_count.unwrap_or(0),
                            result.points_count.unwrap_or(0),
                            (i + 1) * 5,
                            green_wait_secs
                        );
                    }
                    if result.status == 1 {
                        // Double-check
                        std::thread::sleep(std::time::Duration::from_secs(5));
                        if let Ok(info2) = self
                            .rt
                            .block_on(self.client.collection_info(&self.collection_name))
                        {
                            if let Some(result2) = info2.result {
                                if result2.status == 1 {
                                    println!("Collection is GREEN.");
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(format!(
            "Timed out waiting for collection to reach GREEN status after {}s \
             (override with QDRANT_GREEN_WAIT_SECS)",
            green_wait_secs
        ))
    }
}

/// Parse conditions into Qdrant filter format.
/// Map a dataset distance name to the Qdrant `Distance` enum. Unknown metrics
/// return a clear `Err` (never silently default). A wrong arm here (e.g. IP→L2)
/// would silently invert ranking, so every arm is unit-tested.
fn map_qdrant_distance(distance: &str) -> Result<Distance, String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(Distance::Euclid),
        "cosine" | "angular" => Ok(Distance::Cosine),
        "dot" | "ip" => Ok(Distance::Dot),
        other => Err(format!("Unsupported distance metric for Qdrant: {}", other)),
    }
}

// ── House rule for "the config asks for something Qdrant cannot express" ─────
//
// This file deliberately holds TWO opposite responses, and which one applies is
// decided by ONE question: can the unsupported thing change the number we
// publish?
//
//   * It cannot (it only affects indexing/search *speed* or resource layout) →
//     WARN and continue. See the collection-params builders, e.g. an unknown
//     `payload_index_params` field name: the run is still measuring the right
//     query, just possibly slower.
//   * It can (it changes which points are eligible, hence recall/precision) →
//     ERROR and fail the run, via `unrepresentable` below. A filter is always
//     in this category: its recall is scored against ground truth built WITH
//     the filter, so a request that lost a constraint reports a wrong number
//     that looks entirely plausible.
//
// Do not copy one policy into the other's situation without re-asking that
// question (#219).

/// Single phrasing for every filter shape Qdrant cannot express (#219/#222).
///
/// The alternative to failing is always the same and always worse: the request
/// goes out carrying FEWER constraints than the config asked for, while the
/// recall it produces is scored against ground truth built WITH them. That is a
/// plausible-looking wrong number, not a crash, so nothing downstream can catch
/// it — `Option::is_some()` cannot tell a filter that constrains from one that
/// does not.
fn unrepresentable(what: &str, detail: &str) -> String {
    format!(
        "Qdrant cannot express {what}: {detail}. Running it would search with fewer \
         constraints than the config asked for while the recall is scored against \
         filtered ground truth (#219) — fix the dataset/config instead."
    )
}

/// Build the `Filter` for one query's `conditions` object.
///
/// `Ok(None)` means **there was nothing to filter on** — the input was not an
/// object, was `{}`, or carried `and`/`or` arrays that were literally empty.
/// Anything that was asked for but could not be built is an `Err`, never a
/// silent `None`: see [`unrepresentable`].
fn parse_qdrant_conditions(conditions: &serde_json::Value) -> Result<Option<Filter>, String> {
    let Some(obj) = conditions.as_object() else {
        return Ok(None);
    };
    if obj.is_empty() {
        return Ok(None);
    }

    // EMPTINESS GUARD (mirrors elasticsearch.rs `.filter(|f| !f.is_empty())`):
    // an `and`/`or` array that yields no conditions must collapse the whole arm
    // back to `None`, not to `Some(vec![])`. Without this, `must.is_none()` is
    // false, the function returns `Some(Filter{must:[], should:[]})`, and the
    // search path attaches an EMPTY filter — which Qdrant evaluates as
    // match-all. The query then runs effectively UNFILTERED while every check
    // downstream (`Option::is_some()`, "a filter was built") says it filtered.
    // PER-ARM, not just overall: `{"and":[<real>], "or":[]}` produces a perfectly
    // valid-looking `Filter{must:[1], should:[]}` with a WHOLE BOOLEAN ARM
    // missing, and neither an overall `must.is_none() && should.is_none()` check
    // nor the call-site guard can see it — the filter is `Some` and constrains
    // something, just not what was asked. So an arm that is PRESENT must produce
    // conditions.
    let build_arm = |key: &str| -> Result<Option<Vec<Condition>>, String> {
        let Some(v) = obj.get(key) else {
            return Ok(None);
        };
        let Some(entries) = v.as_array() else {
            return Err(unrepresentable(
                &format!("`{key}` group {v}"),
                "expected an array of clauses",
            ));
        };
        if entries.is_empty() {
            return Err(unrepresentable(
                &format!("`{key}` group"),
                "it is empty, so the arm constrains nothing",
            ));
        }
        let conds = build_qdrant_subfilters(entries)?;
        if conds.is_empty() {
            // Unreachable while `build_qdrant_subfilters` errors on every drop;
            // kept so a future edit there cannot resurrect the empty-`Filter`
            // match-all this whole change exists to prevent.
            return Err(unrepresentable(
                &format!("`{key}` group"),
                "it produced no condition",
            ));
        }
        Ok(Some(conds))
    };
    let must = build_arm("and")?;
    let should = build_arm("or")?;

    if must.is_none() && should.is_none() {
        return Ok(None);
    }

    let mut filter = Filter::default();
    if let Some(m) = must {
        filter.must = m;
    }
    if let Some(s) = should {
        filter.should = s;
    }

    Ok(Some(filter))
}

/// Build the conditions for one `and`/`or` array.
///
/// EVERY entry must contribute at least one condition. Dropping one is the
/// dangerous half of #219: in a single-leaf group it collapses the filter to
/// nothing and the query runs fully unfiltered, and in a multi-leaf group it
/// emits a real-looking `Filter` that constrains LESS than the config asked for
/// — which is worse, because the run then publishes a plausible recall instead
/// of an obviously-broken one. So an entry that cannot be built is an error.
fn build_qdrant_subfilters(entries: &[serde_json::Value]) -> Result<Vec<Condition>, String> {
    let mut filters = Vec::new();
    for entry in entries {
        let Some(entry_obj) = entry.as_object() else {
            return Err(unrepresentable(
                &format!("filter clause {entry}"),
                "expected an object like {\"field\": {\"op\": criteria}}",
            ));
        };
        // NESTED GROUP: an entry that is itself an `{and:[...]}` / `{or:[...]}`
        // sub-tree must be built as its OWN sub-Filter and nested via a Filter
        // condition, so grouping is preserved natively — e.g.
        // `(color==red AND size>=50) OR (color==blue AND size<10)` becomes a
        // top-level `should` of two nested Filters, each with its own `must`.
        // Flattening the sub-tree's leaves into the parent must/should would
        // change the boolean meaning and collapse recall.
        if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
            let Some(sub) = parse_qdrant_conditions(entry)? else {
                return Err(unrepresentable(
                    &format!("nested filter group {entry}"),
                    "it produced no condition",
                ));
            };
            filters.push(Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Filter(
                    sub,
                )),
            });
            continue;
        }
        if entry_obj.is_empty() {
            return Err(unrepresentable(
                "an empty filter clause {}",
                "it names no field",
            ));
        }
        // LEAF: `{ field: { op: criteria } }`. A shorthand leaf such as
        // `{"color": "red"}` is NOT that shape; it used to be skipped silently,
        // which is exactly the drop this function now refuses to perform.
        for (field_name, field_filters) in entry_obj {
            let Some(filter_obj) = field_filters.as_object() else {
                return Err(unrepresentable(
                    &format!("filter on field `{field_name}`"),
                    &format!(
                        "expected {{op: criteria}} (e.g. {{\"match\": {{\"value\": …}}}}), \
                         got {field_filters}"
                    ),
                ));
            };
            if filter_obj.is_empty() {
                return Err(unrepresentable(
                    &format!("filter on field `{field_name}`"),
                    "it names no operator",
                ));
            }
            for (cond_type, criteria) in filter_obj {
                filters.push(build_qdrant_filter(field_name, cond_type, criteria)?);
            }
        }
    }
    Ok(filters)
}

/// Turn one query's `conditions` value into the filter attached to its request.
///
/// This is the call site #219 named: `.map(|c| c.as_ref().and_then(parse))`
/// collapsed two different states into the same `None`, and `None` downstream
/// means "omit the filter". The two are kept apart here:
///
/// * genuinely **no** conditions (`null`, or a literally empty `{}`) → `Ok(None)`;
///   the query is unfiltered and so is its ground truth, which is correct.
/// * conditions that are **present** but produced no filter → `Err`; running
///   those unfiltered against filtered ground truth is the silent wrong number.
///
/// `parse_qdrant_conditions` already returns `Ok(None)` for an empty object, so
/// any non-empty input arriving here by definition dropped something.
///
/// A JSON `null` counts as "no conditions". This is load-bearing, not defensive:
/// `compound_reader.rs` builds the vector with `row.get("conditions").cloned()`,
/// so a row spelling `"conditions": null` — which is EVERY row of the shipped
/// `random_keywords_1m_vocab_10_no_filters` dataset — arrives as
/// `Some(Value::Null)`, not `None`. Treating that as a dropped filter would fail
/// a dataset whose queries are genuinely, intentionally unfiltered.
fn parse_query_filter(conditions: Option<&serde_json::Value>) -> Result<Option<Filter>, String> {
    let Some(cond) = conditions.filter(|c| !c.is_null()) else {
        return Ok(None);
    };
    match parse_qdrant_conditions(cond)? {
        Some(f) => Ok(Some(f)),
        None if cond.as_object().is_none_or(|o| !o.is_empty()) => Err(unrepresentable(
            &format!("query filter {cond}"),
            "it produced no condition at all",
        )),
        None => Ok(None),
    }
}

/// Parse a datetime bound string into a protobuf Timestamp.
///
/// Accepts exactly the forms the Redis/Valkey range path accepts, in the same
/// order, because an unparseable bound is now a hard error (see
/// `build_qdrant_filter`) and a config that benchmarks fine on five engines must
/// not kill the Qdrant run:
///
/// 1. RFC-3339 (`…Z`/offset) — sub-second precision preserved;
/// 2. the wider naive/date-only forms via `parsers::datetime_to_epoch_secs`
///    (second granularity by construction: `%S` consumes no fractional part);
/// 3. a bare **epoch-seconds string** such as `"1609459200"` or
///    `"1609459200.5"`, which `datetime_to_epoch_secs` rejects on purpose
///    (`parsers.rs`) and which `redis.rs`/`valkey.rs` handle by falling through
///    to `parse::<i64>`/`parse::<f64>`.
///
/// Note this is not a type substitution: `DatetimeRange`'s bounds are
/// `prost_types::Timestamp` (seconds + nanos) whatever the source spelling, so
/// every form above targets the same Qdrant datetime index.
fn parse_rfc3339_timestamp(s: &str) -> Option<Timestamp> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(Timestamp {
            seconds: dt.timestamp(),
            nanos: dt.timestamp_subsec_nanos() as i32,
        });
    }
    if let Some(secs) = vector_db_benchmark::parsers::datetime_to_epoch_secs(s) {
        return Some(Timestamp {
            seconds: secs as i64,
            nanos: 0,
        });
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(Timestamp {
            seconds: n,
            nanos: 0,
        });
    }
    // `parse::<f64>` also accepts "inf"/"nan", which have no Timestamp — reject.
    match s.parse::<f64>() {
        Ok(n) if n.is_finite() => {
            // Floor + non-negative nanos, so a negative (pre-1970) fractional
            // epoch stays a valid protobuf Timestamp instead of wrapping.
            let secs = n.floor();
            Some(Timestamp {
                seconds: secs as i64,
                nanos: (((n - secs) * 1e9).round() as i64).clamp(0, 999_999_999) as i32,
            })
        }
        _ => None,
    }
}

/// Map a `quantization_config.product.compression` JSON string to Qdrant's
/// `CompressionRatio` enum. Accepts the lowercase ProtoBuf names `"x4".."x64"`
/// (matching the config-file style); returns a clear `Err` on anything else.
fn parse_compression_ratio(s: &str) -> Result<CompressionRatio, String> {
    match s {
        "x4" => Ok(CompressionRatio::X4),
        "x8" => Ok(CompressionRatio::X8),
        "x16" => Ok(CompressionRatio::X16),
        "x32" => Ok(CompressionRatio::X32),
        "x64" => Ok(CompressionRatio::X64),
        other => Err(format!(
            "Unsupported product quantization compression: {} (expected one of x4, x8, x16, x32, x64)",
            other
        )),
    }
}

/// Translate a `quantization_config` JSON object into a Qdrant `Quantization`.
/// Recognizes `scalar` (int8 default, rejects other types), `product` (requires
/// a valid `compression`), and `binary`. Returns `Ok(None)` when none of those
/// keys are present. Extracted verbatim from `apply_optimizers_and_quantization`
/// so the branch logic is unit-testable without a live Qdrant.
fn build_quantization(q: &serde_json::Value) -> Result<Option<Quantization>, String> {
    if let Some(s) = q.get("scalar") {
        let qtype = match s.get("type").and_then(|v| v.as_str()) {
            Some("int8") | None => QuantizationType::Int8,
            Some(other) => return Err(format!("Unsupported scalar quantization type: {}", other)),
        };
        Ok(Some(Quantization::Scalar(ScalarQuantization {
            r#type: qtype.into(),
            quantile: s.get("quantile").and_then(|v| v.as_f64()).map(|v| v as f32),
            always_ram: s.get("always_ram").and_then(|v| v.as_bool()),
        })))
    } else if let Some(p) = q.get("product") {
        let compression = match p.get("compression").and_then(|v| v.as_str()) {
            Some(s) => parse_compression_ratio(s)?,
            None => return Err("Product quantization requires a `compression` value".to_string()),
        };
        Ok(Some(Quantization::Product(ProductQuantization {
            compression: compression.into(),
            always_ram: p.get("always_ram").and_then(|v| v.as_bool()),
        })))
    } else {
        Ok(q.get("binary").map(|b| {
            Quantization::Binary(BinaryQuantization {
                always_ram: b.get("always_ram").and_then(|v| v.as_bool()),
                ..Default::default()
            })
        }))
    }
}

/// Extract `field -> declared-type` from the dataset schema, so uploads keep a
/// numeric-valued keyword field as a string (matching its keyword index).
fn schema_type_map(dataset: &Dataset) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Some(obj) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
        for (k, v) in obj {
            if let Some(t) = v.as_str() {
                m.insert(k.clone(), t.to_string());
            }
        }
    }
    m
}

/// Build ONE leaf condition (`{field: {op: criteria}}`).
///
/// Total by construction: a leaf either becomes a `Condition` or fails the run.
/// There is deliberately no `Ok(None)` "just skip it" arm — every skip this
/// function used to perform (unknown operator, unrepresentable value, vacuous
/// range) removed a constraint the config asked for while leaving the recall
/// scored against ground truth that still had it (#219).
fn build_qdrant_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Result<Condition, String> {
    // What this leaf is, for error messages.
    let what = || format!("`{condition_type}` on field `{field_name}`");
    match condition_type {
        "match" => {
            let Some(criteria_obj) = criteria.as_object() else {
                return Err(unrepresentable(
                    &what(),
                    &format!(
                        "expected an object ({{\"value\": …}} / {{\"any\": [...]}} / \
                         {{\"text\": …}}), got {criteria}"
                    ),
                ));
            };
            // Unrecognized keys would be silently ignored, i.e. a constraint the
            // config asked for would vanish from the request.
            for k in criteria_obj.keys() {
                if !["any", "text", "value"].contains(&k.as_str()) {
                    return Err(unrepresentable(
                        &what(),
                        &format!("unknown match key `{k}` (expected any / text / value)"),
                    ));
                }
            }
            // match_any: value in a list (keywords or integers).
            if let Some(any) = criteria_obj.get("any") {
                let Some(any) = any.as_array() else {
                    return Err(unrepresentable(
                        &what(),
                        &format!("`any` must be a list, got {any}"),
                    ));
                };
                if !any.is_empty() && any.iter().all(|v| v.is_i64()) {
                    let vals: Vec<i64> = any.iter().filter_map(|v| v.as_i64()).collect();
                    return Ok(Condition::matches(field_name.to_string(), vals));
                }
                // BOOLEAN list → the keyword tokens "true"/"false". This engine
                // stores and indexes bools as those STRINGS (see the `value` arm
                // below and `matches_bool_as_string_keyword`), so that is the
                // faithful translation, not a substitution. Erroring here would
                // kill an entire Qdrant run for a filter Elasticsearch executes
                // fine (`elasticsearch.rs` forwards booleans verbatim).
                if !any.is_empty() && any.iter().all(|v| v.is_boolean()) {
                    let vals: Vec<String> = any
                        .iter()
                        .map(|v| match v.as_bool() {
                            Some(true) => "true".to_string(),
                            _ => "false".to_string(),
                        })
                        .collect();
                    return Ok(Condition::matches(field_name.to_string(), vals));
                }
                // DECISION (#222): a member Qdrant cannot express is a HARD
                // ERROR, never a silent drop. Qdrant's `MatchValue` protobuf has
                // exactly Keyword / Integer / Boolean / Text / Keywords /
                // Integers / Except* / Phrase / TextAny. Two consequences:
                //   * there is NO float variant, so a float `any` list (which
                //     pgvector supports — see pgvector.rs
                //     `match_any_float_list_binds_double_array_any`) cannot be
                //     sent as a MatchAny at all;
                //   * the list variants are HOMOGENEOUS, so a mixed list cannot
                //     be one MatchAny either.
                // The previous code ran `filter_map(as_str)` over the list, which
                // deleted every non-string member: a float list became an EMPTY
                // MatchAny (matches nothing — recall 0 reported as an engine
                // result) and a mixed list like `["a", 1]` silently NARROWED to
                // `Keywords(["a"])`. Erroring makes the unsupported combination
                // impossible to mistake for a benchmark number. (Emulating float
                // equality with `Range{gte:v, lte:v}` was rejected: it silently
                // switches to a different index/comparison than the `match` the
                // config asked for, and is exactly the kind of quiet
                // substitution this issue is about.)
                if !any.iter().all(|v| v.is_string()) {
                    return Err(unrepresentable(
                        &what(),
                        &format!(
                            "`any` must be a homogeneous list of strings, integers or \
                             booleans; got {criteria}. Qdrant's MatchValue has no float \
                             variant and its list variants are homogeneous, so this list \
                             cannot be sent as one MatchAny"
                        ),
                    ));
                }
                // All-string list, or an EMPTY list.
                //
                // The empty list is deliberately NOT an error, even though the
                // float case above is and even though it yields zero hits.
                // The distinction this whole change rests on is *faithfulness*,
                // not hit count: `value ∈ ∅` is exactly what the config asked
                // for and exactly what Qdrant evaluates, so nothing is dropped
                // and nothing is substituted — the ground truth for that same
                // condition is empty too. A float `any` is the opposite: the
                // request that goes out is NOT the one the config expressed.
                // pgvector makes the same call in
                // `match_any_empty_list_matches_nothing`, so the two engines
                // stay comparable.
                let vals: Vec<String> = any
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                return Ok(Condition::matches(field_name.to_string(), vals));
            }
            // match_text: full-text match.
            if let Some(text) = criteria_obj.get("text") {
                let Some(text) = text.as_str() else {
                    return Err(unrepresentable(
                        &what(),
                        &format!("`text` must be a string, got {text}"),
                    ));
                };
                return Ok(Condition::matches_text(
                    field_name.to_string(),
                    text.to_string(),
                ));
            }
            // exact match on keyword / integer / bool.
            let Some(value) = criteria_obj.get("value") else {
                return Err(unrepresentable(
                    &what(),
                    &format!("no any / text / value key in {criteria}"),
                ));
            };
            if let Some(s) = value.as_str() {
                Ok(Condition::matches(field_name.to_string(), s.to_string()))
            } else if let Some(b) = value.as_bool() {
                // Bools are stored+indexed as the STRING "true"/"false", so match
                // the string form. A native boolean Match never matches the string
                // payload and silently returns zero points (0 recall).
                let token = if b { "true" } else { "false" };
                Ok(Condition::matches(
                    field_name.to_string(),
                    token.to_string(),
                ))
            } else if let Some(n) = value.as_i64() {
                Ok(Condition::matches(field_name.to_string(), n))
            } else {
                // Same field, same unrepresentable type, same server response
                // (400) as the float `any` above — so the same verdict. A float
                // exact value used to be dropped here, which silently widened
                // the query; use a `range` with equal bounds if that is what the
                // dataset means.
                Err(unrepresentable(
                    &what(),
                    &format!(
                        "`value` must be a string, integer or bool — Qdrant's MatchValue \
                         has no float or array variant; got {value}"
                    ),
                ))
            }
        }
        "range" => {
            let Some(criteria_obj) = criteria.as_object() else {
                return Err(unrepresentable(
                    &what(),
                    &format!("expected an object of lt/gt/lte/gte bounds, got {criteria}"),
                ));
            };
            for k in criteria_obj.keys() {
                if !["lt", "gt", "lte", "gte"].contains(&k.as_str()) {
                    return Err(unrepresentable(
                        &what(),
                        &format!("unknown range operator `{k}` (expected lt / gt / lte / gte)"),
                    ));
                }
            }
            // A string bound means an ISO-8601 datetime range rather than numeric.
            let is_datetime = ["lt", "gt", "lte", "gte"]
                .iter()
                .any(|k| criteria_obj.get(*k).map(|v| v.is_string()).unwrap_or(false));
            if is_datetime {
                // A datetime bound that is PRESENT but does not parse used to
                // yield `None` for that bound. With every bound unparseable the
                // result was `DatetimeRange{lt:None,gt:None,gte:None,lte:None}`
                // — a present but VACUOUS condition matching every point, the
                // exact shape the numeric arm below already guards against
                // (#115). Worse, a PARTIALLY parsed range silently WIDENS,
                // which dropping-when-all-bounds-are-None would not catch. So
                // every present, non-null bound must parse or the clause is an
                // error; this branch is only entered when at least one bound is
                // a string, so a vacuous DatetimeRange is now unconstructible.
                let ts = |k: &str| -> Result<Option<Timestamp>, String> {
                    match criteria_obj.get(k) {
                        None | Some(serde_json::Value::Null) => Ok(None),
                        Some(v) => v
                            .as_str()
                            .and_then(parse_rfc3339_timestamp)
                            .map(Some)
                            .ok_or_else(|| {
                                format!(
                                    "Qdrant datetime range on field `{}` has an \
                                     unparseable `{}` bound: {}. Expected an RFC-3339 \
                                     timestamp (e.g. 2023-01-01T00:00:00Z); emitting the \
                                     range without it would match everything.",
                                    field_name, k, v
                                )
                            }),
                    }
                };
                let dt_range = DatetimeRange {
                    lt: ts("lt")?,
                    gt: ts("gt")?,
                    gte: ts("gte")?,
                    lte: ts("lte")?,
                };
                return Ok(Condition::datetime_range(field_name.to_string(), dt_range));
            }
            // NUMERIC BOUNDS — same rule as the datetime arm above, which the
            // two used to disagree on: a bound that is PRESENT but not a number
            // was dropped, so `{"gte":100,"lte":true}` went out as `gte:100`
            // alone — the exact silent widening #222 is about, just spelled
            // numerically. A `null` bound still means "no bound" (it is how
            // configs express an open side) and a range with no bound at all is
            // vacuous, i.e. match-all, so it is an error rather than a drop
            // (the drop would collapse a single-leaf group to no filter at all).
            let bound = |k: &str| -> Result<Option<f64>, String> {
                match criteria_obj.get(k) {
                    None | Some(serde_json::Value::Null) => Ok(None),
                    Some(v) => v.as_f64().map(Some).ok_or_else(|| {
                        unrepresentable(&what(), &format!("bound `{k}` is not a number: {v}"))
                    }),
                }
            };
            let range = qdrant_client::qdrant::Range {
                lt: bound("lt")?,
                gt: bound("gt")?,
                lte: bound("lte")?,
                gte: bound("gte")?,
            };
            if range.lt.is_none()
                && range.gt.is_none()
                && range.lte.is_none()
                && range.gte.is_none()
            {
                return Err(unrepresentable(
                    &what(),
                    &format!("no bound at all in {criteria} — an empty Range matches everything"),
                ));
            }
            Ok(Condition::range(field_name.to_string(), range))
        }
        "geo" => {
            let (Some(lat), Some(lon)) = (
                criteria.get("lat").and_then(|v| v.as_f64()),
                criteria.get("lon").and_then(|v| v.as_f64()),
            ) else {
                return Err(unrepresentable(
                    &what(),
                    &format!("requires numeric `lat` and `lon`, got {criteria}"),
                ));
            };
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Ok(Condition::geo_radius(
                field_name.to_string(),
                qdrant_client::qdrant::GeoRadius {
                    center: Some(qdrant_client::qdrant::GeoPoint { lon, lat }),
                    radius: radius as f32,
                },
            ))
        }
        _ => Err(unrepresentable(
            &what(),
            "unsupported operator (Qdrant supports match / range / geo)",
        )),
    }
}

impl Engine for QdrantEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        println!("Deleting existing collection...");
        self.delete_collection()?;

        println!("Creating collection '{}'...", self.collection_name);
        self.create_collection(dataset)?;
        println!("Collection '{}' created.", self.collection_name);

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        if dataset.is_hybrid() {
            return self.upload_hybrid(dataset);
        }
        if dataset.is_sparse() {
            return self.upload_sparse(dataset);
        }

        let normalize = dataset.needs_normalization();
        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();

        println!(
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
        );

        println!(
            "Starting upload with {} threads, batch size {}...",
            self.parallel, self.batch_size
        );
        let upload_start = Instant::now();
        let schema_types = schema_type_map(dataset);
        self.upload_parallel(&ids, &vectors, &metadata, &schema_types)?;
        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Wait for indexing to complete. Include this wait in total_time for
        // cross-engine comparability (mirrors mongodb; matches v0's post_upload()
        // timing) — the HNSW build is the dominant part of Qdrant ingest cost.
        let index_start = Instant::now();
        self.wait_collection_green()?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: vectors.len(),
            parallel: self.parallel,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Build Qdrant search params
        let hnsw_ef: Option<u64> = params.search_params.as_ref().and_then(|sp| {
            sp.ef.map(|e| e as u64).or_else(|| {
                sp.extra
                    .as_ref()
                    .and_then(|e| e.get("hnsw_ef"))
                    .and_then(|v| v.as_u64())
            })
        });

        // Search-time quantization params (rescore/oversampling) from the config's
        // search_params.quantization object — mirrors python rest.SearchParams(**params).
        let quantization_params: Option<QuantizationSearchParams> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get("quantization"))
            .map(|q| QuantizationSearchParams {
                rescore: q.get("rescore").and_then(|v| v.as_bool()),
                oversampling: q.get("oversampling").and_then(|v| v.as_f64()),
                ..Default::default()
            });

        // Prefetch (two-stage retrieval / rescoring): search_params.prefetch =
        // { "limit": N, "params": { "hnsw_ef": .., "quantization": {..} } }.
        // Mirrors python `models.Prefetch(**prefetch, query=query_vector)`.
        let prefetch = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get("prefetch"));
        let prefetch_enabled = prefetch.is_some();
        let prefetch_limit = prefetch
            .and_then(|p| p.get("limit"))
            .and_then(|v| v.as_u64());
        let prefetch_params = prefetch.and_then(|p| p.get("params"));
        let prefetch_hnsw_ef = prefetch_params
            .and_then(|p| p.get("hnsw_ef"))
            .and_then(|v| v.as_u64());
        let prefetch_quant: Option<QuantizationSearchParams> = prefetch_params
            .and_then(|p| p.get("quantization"))
            .map(|q| QuantizationSearchParams {
                rescore: q.get("rescore").and_then(|v| v.as_bool()),
                oversampling: q.get("oversampling").and_then(|v| v.as_f64()),
                ..Default::default()
            });

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());

        // Dense and sparse queries are read into separate vectors. For dense-only
        // and sparse-only runs exactly one is populated; for HYBRID runs BOTH are
        // populated (row-aligned) and fused server-side. Filters/prefetch/
        // quantization apply to the dense-only path.
        let is_sparse = dataset.is_sparse();
        let is_hybrid = dataset.is_hybrid();
        // Per-prefetch candidate depth for hybrid fusion: reuse the configured
        // search_params.prefetch.limit if present, else a generous default (>= any
        // sensible top-k). Larger = better fusion recall, more work.
        let hybrid_prefetch_limit = prefetch_limit.unwrap_or(50);
        let (queries, sparse_queries, neighbors, parsed_filters) = if is_hybrid {
            let (dq, sq, nb) = dataset.read_hybrid_queries()?;
            (dq, sq, nb, Vec::<Option<Filter>>::new())
        } else if is_sparse {
            let (sq, nb) = dataset.read_sparse_queries()?;
            (Vec::<Vec<f32>>::new(), sq, nb, Vec::<Option<Filter>>::new())
        } else {
            let (q, nb, conditions) = dataset.read_queries()?;
            // "no conditions" and "conditions that produced no filter" are NOT
            // the same thing — `parse_query_filter` keeps them apart, so an
            // unrepresentable filter fails the run instead of quietly running
            // the query unfiltered against filtered ground truth (#219/#222).
            let pf: Vec<Option<Filter>> = conditions
                .iter()
                .map(|c| parse_query_filter(c.as_ref()))
                .collect::<Result<Vec<_>, String>>()?;
            (
                q,
                Vec::<vector_db_benchmark::readers::SparseVector>::new(),
                nb,
                pf,
            )
        };

        let query_count = if is_sparse {
            sparse_queries.len()
        } else {
            queries.len()
        };
        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(query_count)
        } else {
            query_count
        };

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);

        // Barrier-synchronized start so per-worker client/runtime setup AND the cold
        // first query fall OUTSIDE the measured window (mirrors redis/vertex). Every
        // worker builds its runtime + client and primes with one discarded query,
        // then blocks on `ready`; the main thread stamps the shared start instant
        // into `start_cell` and releases `go`, so the measurement clock starts only
        // once all workers are warm and poised. A worker that fails to build its
        // runtime/client MUST still pass both barriers before returning, or the run
        // would deadlock.
        let ready = Arc::new(std::sync::Barrier::new(parallel + 1));
        let go = Arc::new(std::sync::Barrier::new(parallel + 1));
        let start_cell = Arc::new(std::sync::OnceLock::<Instant>::new());

        // Each worker builds its own client/connection (like ES/OpenSearch) rather
        // than sharing one gRPC connection, which would serialize parallel queries.
        let grpc_url = self.grpc_url.clone();
        let api_key = self.api_key.clone();
        let timeout = self.timeout;
        let collection_name = self.collection_name.clone();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let grpc_url = grpc_url.clone();
                let api_key = api_key.clone();
                let collection_name = collection_name.clone();
                let queries = &queries;
                let sparse_queries = &sparse_queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
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

                    // Build the fully-configured query for a given query index. Shared
                    // by the prime (query 0, discarded) and the timed loop so both use
                    // the exact same per-query request path.
                    let build_query = |idx: usize| -> QueryPointsBuilder {
                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let mut query_builder = if is_hybrid {
                            // Hybrid: two prefetches (dense NN + sparse NN) fused
                            // server-side with reciprocal-rank fusion (RRF). Each
                            // prefetch pulls `pf_limit` candidates from its own named
                            // vector; RRF ranks by combined rank. Floor the depth at
                            // `top` so the fusion pool is never smaller than the
                            // requested result count (which would understate recall).
                            let pf_limit = hybrid_prefetch_limit.max(top as u64);
                            let dense_pf = PrefetchQueryBuilder::default()
                                .using("dense")
                                .query(Query::new_nearest(queries[idx].clone()))
                                .limit(pf_limit)
                                .build();
                            let sv = &sparse_queries[idx];
                            let sparse_pf = PrefetchQueryBuilder::default()
                                .using("sparse")
                                .query(VectorInput::new_sparse(
                                    sv.indices.clone(),
                                    sv.values.clone(),
                                ))
                                .limit(pf_limit)
                                .build();
                            QueryPointsBuilder::new(collection_name.clone())
                                .query(Query::new_fusion(Fusion::Rrf))
                                .prefetch(vec![dense_pf, sparse_pf])
                                .limit(top as u64)
                                .with_payload(false)
                        } else if is_sparse {
                            let sv = &sparse_queries[idx];
                            QueryPointsBuilder::new(collection_name.clone())
                                .query(VectorInput::new_sparse(
                                    sv.indices.clone(),
                                    sv.values.clone(),
                                ))
                                .using("sparse")
                                .limit(top as u64)
                                .with_payload(false)
                        } else {
                            let mut qb = QueryPointsBuilder::new(collection_name.clone())
                                .query(queries[idx].clone())
                                .limit(top as u64)
                                .with_payload(false);
                            if hnsw_ef.is_some() || quantization_params.is_some() {
                                qb = qb.params(QdrantSearchParams {
                                    hnsw_ef,
                                    quantization: quantization_params,
                                    ..Default::default()
                                });
                            }
                            if let Some(filter) = &parsed_filters[idx] {
                                qb = qb.filter(filter.clone());
                            }
                            qb
                        };

                        if !is_sparse && !is_hybrid && prefetch_enabled {
                            let mut pf =
                                PrefetchQueryBuilder::default().query(queries[idx].clone());
                            if let Some(l) = prefetch_limit {
                                pf = pf.limit(l);
                            }
                            if prefetch_hnsw_ef.is_some() || prefetch_quant.is_some() {
                                pf = pf.params(QdrantSearchParams {
                                    hnsw_ef: prefetch_hnsw_ef,
                                    quantization: prefetch_quant,
                                    ..Default::default()
                                });
                            }
                            query_builder = query_builder.prefetch(vec![pf.build()]);
                        }

                        query_builder
                    };

                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(_) => {
                            // Still cross both barriers so peers aren't stranded.
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Per-worker client so each thread has an independent connection.
                    let client = match rt.block_on(async {
                        let mut b = Qdrant::from_url(&grpc_url)
                            .timeout(std::time::Duration::from_secs(timeout));
                        if let Some(k) = &api_key {
                            b = b.api_key(k.clone());
                        }
                        b.build()
                    }) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Qdrant worker client build failed: {}", e);
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Prime this worker with ONE discarded query (query 0) so the cold
                    // first round-trip (client warm-up + server caches/JIT) is not
                    // inside the measured window. Best effort: errors are ignored and
                    // its sample is NOT recorded.
                    if num_to_run > 0 {
                        let _ = rt.block_on(client.query(build_query(0)));
                    }

                    // Signal "runtime + client built + primed", then block until the
                    // main thread stamps the shared measurement start and releases all.
                    ready.wait();
                    go.wait();

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

                        let query_builder = build_query(idx);

                        let query_start = Instant::now();
                        let result = rt.block_on(client.query(query_builder));
                        let query_time = query_start.elapsed().as_secs_f64();

                        match result {
                            Ok(response) => {
                                let ordered_ids: Vec<i64> = response
                                    .result
                                    .iter()
                                    .filter_map(|p| {
                                        if let Some(
                                            qdrant_client::qdrant::point_id::PointIdOptions::Num(n),
                                        ) = &p
                                            .id
                                            .as_ref()
                                            .and_then(|id| id.point_id_options.as_ref())
                                        {
                                            Some(*n as i64)
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
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
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                }));
            }

            // All workers are built + primed and blocked on `go`. Stamp the shared
            // measurement start and release them simultaneously, so connection setup
            // and the cold first query are already behind the barrier.
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
        // Measure from the post-barrier start stamp (workers already primed), so
        // total_time excludes connection setup and the cold first query.
        let total_time = start_cell
            .get()
            .map(|st| st.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        self.delete_collection()
    }

    /// Collect memory usage from Qdrant's REST observability endpoints, mirroring
    /// the Redis wrapper's `{used_memory, index_info}` shape.
    ///
    /// - `/metrics` (Prometheus): jemalloc `memory_*_bytes` gauges + collection counts.
    ///   `memory_resident_bytes` (RSS) is used as `used_memory`, the analog of
    ///   Redis' `used_memory`.
    /// - `/telemetry` (JSON): collection/cluster/segment state, the analog of FT.INFO.
    ///
    /// See https://qdrant.tech/documentation/cloud/cluster-monitoring/.
    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let http = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok()?;

        let get = |path: &str| -> Option<reqwest::blocking::Response> {
            let mut req = http.get(format!("{}{}", self.rest_url, path));
            if let Some(key) = &self.api_key {
                req = req.header("api-key", key);
            }
            req.send().ok().filter(|r| r.status().is_success())
        };

        // Prometheus /metrics → curated gauge map.
        let metrics = get("/metrics")
            .and_then(|r| r.text().ok())
            .map(|t| parse_qdrant_metrics(&t))
            .unwrap_or_default();
        let resident = metrics
            .get("memory_resident_bytes")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as i64;

        // Telemetry JSON → collection/cluster state.
        let telemetry: Option<serde_json::Value> = get("/telemetry?anonymize=true")
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|mut v| v.get_mut("result").map(|r| r.take()));

        // Per-collection info from the gRPC client (segments, vector counts, status).
        let collection_info = self
            .rt
            .block_on(self.client.collection_info(&self.collection_name))
            .ok()
            .map(|info| format!("{:?}", info.result));

        Some(serde_json::json!({
            "used_memory": [resident],
            "index_info": telemetry,
            "qdrant_metrics": metrics,
            "collection_info": collection_info,
        }))
    }
}

/// Parse the curated set of Qdrant `/metrics` (Prometheus text) gauges into a JSON
/// map. Only memory and collection-count gauges are kept; labeled/histogram lines
/// and comments are ignored.
fn parse_qdrant_metrics(text: &str) -> serde_json::Map<String, serde_json::Value> {
    const WANTED: &[&str] = &[
        "memory_active_bytes",
        "memory_allocated_bytes",
        "memory_metadata_bytes",
        "memory_resident_bytes",
        "memory_retained_bytes",
        "collections_total",
        "collections_vector_total",
    ];
    let mut out = serde_json::Map::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Split off the trailing value: "metric_name{labels} 123" or "metric_name 123".
        let mut it = line.rsplitn(2, char::is_whitespace);
        let value_str = match it.next() {
            Some(v) => v,
            None => continue,
        };
        let name_part = it.next().unwrap_or("").trim();
        let name = name_part.split('{').next().unwrap_or(name_part).trim();
        if WANTED.contains(&name) {
            if let Ok(v) = value_str.parse::<f64>() {
                out.insert(name.to_string(), serde_json::json!(v));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        build_qdrant_filter, parse_compression_ratio, parse_qdrant_metrics, parse_rfc3339_timestamp,
    };
    use qdrant_client::qdrant::{
        condition::ConditionOneOf, CompressionRatio, FieldCondition, Filter, QueryPointsBuilder,
    };
    use serde_json::json;

    fn field_condition(c: &qdrant_client::qdrant::Condition) -> FieldCondition {
        match c.condition_one_of.clone().unwrap() {
            ConditionOneOf::Field(fc) => fc,
            other => panic!("expected FieldCondition, got {:?}", other),
        }
    }

    #[test]
    fn parses_product_compression_ratio() {
        assert_eq!(parse_compression_ratio("x4").unwrap(), CompressionRatio::X4);
        assert_eq!(parse_compression_ratio("x8").unwrap(), CompressionRatio::X8);
        assert_eq!(
            parse_compression_ratio("x16").unwrap(),
            CompressionRatio::X16
        );
        assert_eq!(
            parse_compression_ratio("x32").unwrap(),
            CompressionRatio::X32
        );
        assert_eq!(
            parse_compression_ratio("x64").unwrap(),
            CompressionRatio::X64
        );
        // Unknown / wrongly-cased values must error, not silently default.
        assert!(parse_compression_ratio("x128").is_err());
        assert!(parse_compression_ratio("X16").is_err());
        assert!(parse_compression_ratio("").is_err());
    }

    #[test]
    fn parses_rfc3339_timestamp() {
        let ts = parse_rfc3339_timestamp("1970-01-01T00:00:01Z").unwrap();
        assert_eq!(ts.seconds, 1);
        assert!(parse_rfc3339_timestamp("not-a-date").is_none());
        // `parse::<f64>` accepts these; a Timestamp does not.
        assert!(parse_rfc3339_timestamp("inf").is_none());
        assert!(parse_rfc3339_timestamp("NaN").is_none());
    }

    #[test]
    fn epoch_seconds_string_bound_parses_like_redis_and_valkey() {
        // `parsers::datetime_to_epoch_secs` rejects a plain numeric string BY
        // DESIGN; redis.rs/valkey.rs then fall through to `parse::<i64>/<f64>`.
        // Qdrant must too, or a config that benchmarks fine on the Redis family
        // hard-errors here (the bound is now fatal, not dropped).
        assert_eq!(
            parse_rfc3339_timestamp("1609459200").unwrap().seconds,
            1_609_459_200,
            "2021-01-01T00:00:00Z as epoch seconds"
        );
        let frac = parse_rfc3339_timestamp("1609459200.25").unwrap();
        assert_eq!(frac.seconds, 1_609_459_200);
        assert_eq!(frac.nanos, 250_000_000);
        // Pre-1970 fractional epoch: floor the seconds, keep nanos non-negative,
        // so the protobuf Timestamp stays valid instead of wrapping.
        let neg = parse_rfc3339_timestamp("-0.5").unwrap();
        assert_eq!(neg.seconds, -1);
        assert_eq!(neg.nanos, 500_000_000);
    }

    #[test]
    fn builds_match_any_integers() {
        let c = build_qdrant_filter("cat", "match", &json!({"any": [1, 2, 3]})).unwrap();
        let fc = field_condition(&c);
        assert_eq!(fc.key, "cat");
        assert!(fc.r#match.is_some(), "match_any should set a Match");
    }

    #[test]
    fn builds_match_text() {
        let c = build_qdrant_filter("body", "match", &json!({"text": "hello"})).unwrap();
        let fc = field_condition(&c);
        assert_eq!(fc.key, "body");
        assert!(fc.r#match.is_some());
    }

    #[test]
    fn builds_bool_exact_match() {
        use qdrant_client::qdrant::r#match::MatchValue;
        // Bools are stored as the string "true"/"false", so the filter must match
        // the STRING keyword, NOT a native boolean (which matches 0 points).
        let c = build_qdrant_filter("flag", "match", &json!({"value": true})).unwrap();
        let m = field_condition(&c).r#match.expect("match present");
        assert_eq!(
            m.match_value,
            Some(MatchValue::Keyword("true".to_string())),
            "bool true must match keyword \"true\", not a native boolean"
        );
        let c = build_qdrant_filter("flag", "match", &json!({"value": false})).unwrap();
        let m = field_condition(&c).r#match.expect("match present");
        assert_eq!(
            m.match_value,
            Some(MatchValue::Keyword("false".to_string()))
        );
    }

    #[test]
    fn range_with_string_bound_becomes_datetime_range() {
        let dt =
            build_qdrant_filter("ts", "range", &json!({"gte": "2023-01-01T00:00:00Z"})).unwrap();
        let fc = field_condition(&dt);
        assert!(fc.datetime_range.is_some(), "string bound → datetime_range");
        assert!(fc.range.is_none());

        let num = build_qdrant_filter("n", "range", &json!({"gte": 5, "lt": 10})).unwrap();
        let fc = field_condition(&num);
        assert!(fc.range.is_some(), "numeric bounds → numeric range");
        assert!(fc.datetime_range.is_none());
    }

    #[test]
    fn parses_qdrant_memory_and_collection_gauges() {
        let sample = "\
# HELP memory_resident_bytes Resident memory
# TYPE memory_resident_bytes gauge
app_info{name=\"qdrant\",version=\"1.13.4\"} 1
collections_total 3
collections_vector_total 1500000
cluster_enabled 0
memory_active_bytes 57212928
memory_allocated_bytes 48281048
memory_resident_bytes 74133504
rest_responses_total{method=\"GET\"} 42
";
        let m = parse_qdrant_metrics(sample);
        assert_eq!(
            m.get("memory_resident_bytes").unwrap().as_f64(),
            Some(74133504.0)
        );
        assert_eq!(
            m.get("collections_vector_total").unwrap().as_f64(),
            Some(1500000.0)
        );
        assert_eq!(m.get("collections_total").unwrap().as_f64(), Some(3.0));
        // Non-curated / labeled / comment lines are ignored.
        assert!(m.get("app_info").is_none());
        assert!(m.get("rest_responses_total").is_none());
        assert!(m.get("cluster_enabled").is_none());
        // 5 curated gauges present in the sample.
        assert_eq!(m.len(), 5);
    }

    // ── OR-branch of the condition parser ──────────────────────────────────
    use super::{map_qdrant_distance, parse_qdrant_conditions};
    use qdrant_client::qdrant::Distance;

    #[test]
    fn or_only_populates_should_not_must() {
        let cond = json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        let filter = parse_qdrant_conditions(&cond).unwrap().unwrap();
        assert_eq!(filter.should.len(), 2, "or → 2 should entries");
        assert!(filter.must.is_empty(), "or-only leaves must empty");
    }

    #[test]
    fn and_plus_or_populates_both() {
        let cond = json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}},{"c":{"match":{"value":"z"}}}],
        });
        let filter = parse_qdrant_conditions(&cond).unwrap().unwrap();
        assert_eq!(filter.must.len(), 1);
        assert_eq!(filter.should.len(), 2);
    }

    // ── Range operators ────────────────────────────────────────────────────

    fn numeric_range(criteria: serde_json::Value) -> qdrant_client::qdrant::Range {
        let c = build_qdrant_filter("n", "range", &criteria).unwrap();
        field_condition(&c).range.expect("numeric range")
    }

    #[test]
    fn range_lt_sets_only_lt() {
        let r = numeric_range(json!({"lt":5}));
        assert_eq!(r.lt, Some(5.0));
        assert!(r.gt.is_none() && r.lte.is_none() && r.gte.is_none());
    }

    #[test]
    fn range_lte_sets_only_lte() {
        let r = numeric_range(json!({"lte":5}));
        assert_eq!(r.lte, Some(5.0));
        assert!(r.lt.is_none() && r.gt.is_none() && r.gte.is_none());
    }

    #[test]
    fn range_gt_sets_only_gt() {
        let r = numeric_range(json!({"gt":5}));
        assert_eq!(r.gt, Some(5.0));
        assert!(r.lt.is_none() && r.lte.is_none() && r.gte.is_none());
    }

    #[test]
    fn range_gte_sets_only_gte() {
        let r = numeric_range(json!({"gte":5}));
        assert_eq!(r.gte, Some(5.0));
        assert!(r.lt.is_none() && r.gt.is_none() && r.lte.is_none());
    }

    #[test]
    fn range_two_sided_gte_lt() {
        let r = numeric_range(json!({"gte":10,"lt":20}));
        assert_eq!(r.gte, Some(10.0));
        assert_eq!(r.lt, Some(20.0));
    }

    // #121/#222: a range that produces NO valid bound is an empty, entirely
    // unconstraining Range — i.e. match-all. It used to be dropped, which is no
    // better: dropping the only leaf of a group leaves the query unfiltered.
    // Both spellings of "this range constrains nothing" are now errors.
    #[test]
    fn range_unknown_op_is_an_error() {
        let err = build_qdrant_filter("n", "range", &json!({"foo": 5})).unwrap_err();
        assert!(
            err.contains("foo"),
            "error should name the bad operator: {err}"
        );
    }

    #[test]
    fn range_null_bound_only_is_an_error() {
        assert!(
            build_qdrant_filter("n", "range", &json!({"gte": serde_json::Value::Null})).is_err(),
            "a range with no bound at all matches everything — must not be built"
        );
    }

    #[test]
    fn range_valid_bound_survives_null_sibling() {
        // `null` is how a config spells an OPEN side, so it is not an error —
        // a present valid bound still yields a Range carrying only that bound.
        let r = numeric_range(json!({"gte": 5, "lt": serde_json::Value::Null}));
        assert_eq!(r.gte, Some(5.0));
        assert!(r.lt.is_none() && r.gt.is_none() && r.lte.is_none());
    }

    #[test]
    fn numeric_range_bound_that_is_present_but_not_a_number_is_an_error() {
        // D3: the numeric arm used to disagree with the datetime arm 30 lines
        // above — `{"gte":100,"lte":true}` silently emitted `Range{gte:100}`,
        // dropping the upper bound and matching far more than asked, which is
        // the same silent widening #222 hardened the datetime arm against.
        let err = build_qdrant_filter("n", "range", &json!({"gte": 100, "lte": true})).unwrap_err();
        assert!(
            err.contains("lte"),
            "error should name the bad bound: {err}"
        );
        assert!(build_qdrant_filter("n", "range", &json!({"gte": [1]})).is_err());
    }

    // ── Geo filter ─────────────────────────────────────────────────────────

    #[test]
    fn geo_with_radius_sets_center_and_radius() {
        let c = build_qdrant_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0,"radius":500}))
            .unwrap();
        let gr = field_condition(&c).geo_radius.expect("geo_radius");
        let center = gr.center.expect("center");
        assert_eq!(center.lat, 20.0);
        assert_eq!(center.lon, 10.0);
        assert_eq!(gr.radius, 500.0);
    }

    #[test]
    fn geo_without_radius_uses_default_1000() {
        let c = build_qdrant_filter("loc", "geo", &json!({"lat":20.0,"lon":10.0})).unwrap();
        let gr = field_condition(&c).geo_radius.expect("geo_radius");
        assert_eq!(gr.radius, 1000.0);
    }

    #[test]
    fn geo_missing_lat_or_lon_is_an_error() {
        // Dropping a geo clause leaves the query searching the whole globe.
        assert!(build_qdrant_filter("loc", "geo", &json!({"lon":10.0,"radius":500})).is_err());
        assert!(build_qdrant_filter("loc", "geo", &json!({"lat":20.0,"radius":500})).is_err());
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn distance_mapping_covers_all_arms() {
        assert_eq!(map_qdrant_distance("cosine").unwrap(), Distance::Cosine);
        assert_eq!(map_qdrant_distance("angular").unwrap(), Distance::Cosine);
        assert_eq!(map_qdrant_distance("l2").unwrap(), Distance::Euclid);
        assert_eq!(map_qdrant_distance("euclidean").unwrap(), Distance::Euclid);
        assert_eq!(map_qdrant_distance("dot").unwrap(), Distance::Dot);
        assert_eq!(map_qdrant_distance("ip").unwrap(), Distance::Dot);
        assert_eq!(map_qdrant_distance("COSINE").unwrap(), Distance::Cosine);
        assert!(map_qdrant_distance("nope").is_err());
    }

    // ── Exact-match numeric / non-scalar arms ──────────────────────────────

    #[test]
    fn exact_match_int_sets_match() {
        let c = build_qdrant_filter("n", "match", &json!({"value":5})).unwrap();
        assert!(field_condition(&c).r#match.is_some());
    }

    #[test]
    fn exact_match_float_is_an_error_like_float_match_any() {
        // Qdrant `Match` supports keyword/integer/bool only. A float `{"any":[1.5]}`
        // is a hard error (Qdrant's MatchValue has no float variant); `{"value":1.5}`
        // is the SAME field, the SAME unrepresentable type and the SAME server
        // response, so it cannot be a silent drop — that drop widened the query.
        let err = build_qdrant_filter("n", "match", &json!({"value":1.5})).unwrap_err();
        assert!(
            err.contains("field `n`"),
            "error should name the field: {err}"
        );
        assert!(build_qdrant_filter("n", "match", &json!({"any":[1.5]})).is_err());
    }

    #[test]
    fn exact_match_array_value_is_an_error() {
        assert!(build_qdrant_filter("n", "match", &json!({"value":[1,2]})).is_err());
        assert!(
            build_qdrant_filter("n", "match", &json!({"value": serde_json::Value::Null})).is_err()
        );
    }

    #[test]
    fn unknown_match_key_is_an_error_not_a_silently_ignored_constraint() {
        // e.g. an `except` list: honouring only the keys we know would send a
        // LESS constrained query than the config expressed.
        assert!(build_qdrant_filter("n", "match", &json!({"value":"x","except":["y"]})).is_err());
        assert!(build_qdrant_filter("n", "match", &json!({"any": "not-a-list"})).is_err());
        assert!(build_qdrant_filter("n", "match", &json!({"text": 5})).is_err());
        assert!(build_qdrant_filter("n", "match", &json!({})).is_err());
        assert!(build_qdrant_filter("n", "match", &json!("red")).is_err());
    }

    #[test]
    fn unknown_condition_operator_is_an_error() {
        // The catch-all used to be `_ => None`: any operator the parser did not
        // know simply vanished from the request.
        let err = build_qdrant_filter("a", "nosuchop", &json!({"value":"x"})).unwrap_err();
        assert!(
            err.contains("nosuchop"),
            "error should name the operator: {err}"
        );
    }

    // ── parse_qdrant_conditions edge cases + subfilter builder ──────────────
    use super::{build_qdrant_subfilters, parse_query_filter};

    #[test]
    fn empty_conditions_object_is_none() {
        assert!(parse_qdrant_conditions(&json!({})).unwrap().is_none());
    }

    #[test]
    fn and_only_populates_must_not_should() {
        let cond = json!({"and":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        let filter = parse_qdrant_conditions(&cond).unwrap().unwrap();
        assert_eq!(filter.must.len(), 2, "and → 2 must entries");
        assert!(filter.should.is_empty(), "and-only leaves should empty");
    }

    #[test]
    fn subfilters_build_match_any_keyword_list() {
        // A keyword match_any list → one Condition carrying a keyword Match.
        let entries = vec![json!({"cat":{"match":{"any":["a","b"]}}})];
        let conds = build_qdrant_subfilters(&entries).unwrap();
        assert_eq!(conds.len(), 1);
        let fc = field_condition(&conds[0]);
        assert_eq!(fc.key, "cat");
        assert!(
            fc.r#match.is_some(),
            "match_any keyword list should set a Match"
        );
    }

    // ── #222: no filter may ever be sent that silently matches everything ───

    #[test]
    fn and_group_whose_every_leaf_drops_is_an_error() {
        // Every leaf here is un-buildable: an unknown condition type, an
        // unknown range operator, and a null bound. Pre-fix this returned
        // `Some(Filter{must:[], should:[]})` — an EMPTY filter, which Qdrant
        // evaluates as MATCH-ALL, so the query ran unfiltered while the code
        // (and any `is_some()` check) believed it was filtered. Returning
        // `None` instead is not a fix: `None` ALSO means "send no filter", so
        // the query still runs unfiltered against filtered ground truth. Only
        // failing the run distinguishes it from "this query has no filter".
        let cond = json!({"and":[
            {"a":{"nosuchop":{"value":"x"}}},
            {"n":{"range":{"foo":5}}},
            {"m":{"range":{"gte": serde_json::Value::Null}}},
        ]});
        assert!(parse_qdrant_conditions(&cond).is_err());
        assert!(parse_query_filter(Some(&cond)).is_err());
    }

    #[test]
    fn or_group_whose_every_leaf_drops_is_an_error() {
        let cond = json!({"or":[{"a":{"nosuchop":{"value":"x"}}}]});
        assert!(parse_query_filter(Some(&cond)).is_err());
    }

    #[test]
    fn and_with_one_dropping_leaf_is_an_error_not_a_widened_filter() {
        // The MOST dangerous shape of #219, and the one an "all leaves dropped"
        // guard cannot see: one leaf survives, so a real-looking `Filter` goes
        // out — constraining LESS than the config asked for. Live, this returned
        // 10 rows where the config asked for 4. A partially-constrained query
        // publishes a plausible recall, which is worse than an obviously-broken
        // one, so it must fail rather than filter approximately.
        let cond = json!({"and":[
            {"a":{"nosuchop":{"value":"x"}}},
            {"color":{"match":{"value":"red"}}},
        ]});
        assert!(
            parse_query_filter(Some(&cond)).is_err(),
            "a group that lost a leaf must fail, not filter on what is left"
        );
    }

    #[test]
    fn nested_group_that_collapses_is_an_error_not_a_dropped_clause() {
        // A nested `{and:[...]}` that collapses must not be pushed as an empty
        // sub-Filter condition (match-all inside a `must`) — nor silently
        // omitted, which removes one side of the boolean the config wrote.
        let cond = json!({"and":[
            {"and":[{"a":{"nosuchop":{"value":"x"}}}]},
            {"color":{"match":{"value":"red"}}},
        ]});
        assert!(parse_query_filter(Some(&cond)).is_err());
    }

    // ── #219: the call site must not collapse "no filter" into "filter that
    //    parsed to nothing" ────────────────────────────────────────────────

    #[test]
    fn parse_query_filter_separates_no_conditions_from_dropped_conditions() {
        // Genuinely unfiltered: ground truth is unfiltered too, so `None` is right.
        assert!(parse_query_filter(None).unwrap().is_none());
        assert!(parse_query_filter(Some(&json!({}))).unwrap().is_none());
        // `compound_reader.rs` uses `row.get("conditions").cloned()`, so a row
        // spelling `"conditions": null` — every row of the shipped
        // `random_keywords_1m_vocab_10_no_filters` dataset — arrives as
        // `Some(Value::Null)`, NOT `None`. Rejecting that would fail a dataset
        // whose queries are intentionally unfiltered.
        assert!(parse_query_filter(Some(&serde_json::Value::Null))
            .unwrap()
            .is_none());
        // Present and buildable → a real filter.
        let ok = json!({"and":[{"color":{"match":{"value":"red"}}}]});
        assert!(parse_query_filter(Some(&ok)).unwrap().is_some());
    }

    #[test]
    fn conditions_without_an_and_or_wrapper_are_an_error_not_an_unfiltered_run() {
        // `parse_qdrant_conditions` only reads "and"/"or", so a bare top-level
        // field map — and the shorthand leaf spelling — used to reach the search
        // path as `None` and run COMPLETELY unfiltered while the ground truth
        // was filtered. These are the highest-frequency shapes in that class.
        for cond in [
            json!({"color":{"match":{"value":"red"}}}),
            json!({"color":"red"}),
            json!({"and":[{"color":"red"}]}),
            json!({"and":[]}),
            json!({"and":[{}]}),
            json!({"and":["not-an-object"]}),
            json!({"and":[{"color":{}}]}),
            json!([{"color":{"match":{"value":"red"}}}]),
            json!("red"),
        ] {
            assert!(
                parse_query_filter(Some(&cond)).is_err(),
                "{cond} must fail the run, not run unfiltered"
            );
        }
    }

    #[test]
    fn empty_and_array_never_becomes_an_empty_match_all_filter() {
        // A literally empty `and`/`or` was the SIMPLEST trigger for defect 1 —
        // simpler than "every leaf drops" — because the builder handed back
        // `Some(Filter{must:[], should:[]})`, the object Qdrant reads as
        // match-all. It must never build, in either arm or both.
        assert!(parse_qdrant_conditions(&json!({"and":[], "or":[]})).is_err());
        assert!(parse_qdrant_conditions(&json!({"and":[]})).is_err());
        assert!(parse_qdrant_conditions(&json!({"or":[]})).is_err());
    }

    #[test]
    fn every_shipped_dataset_condition_shape_still_builds() {
        // Guard against the new errors firing on data we actually ship. These
        // are the exact shapes emitted by `generate_dataset.rs` (synthetic-
        // filter-32, synthetic-selectivity-32) and present in the downloadable
        // h-and-m-2048-angular-filters / random-100-match-kw-small-vocab-filters
        // `tests.jsonl` files.
        for cond in [
            json!({"and":[{"color":{"match":{"any":["red","blue"]}}}]}),
            json!({"and":[{"size":{"match":{"any":[1,2,3]}}}]}),
            json!({"and":[{"flag":{"match":{"value":true}}}]}),
            json!({"and":[{"ts":{"range":{"gte":"2023-04-11T00:00:00+00:00",
                                          "lt":"2023-10-28T00:00:00+00:00"}}}]}),
            json!({"and":[{"rank":{"range":{"lt":1000}}}]}),
            json!({"and":[{"section_name":{"match":{"value":"Womens Everyday Basics"}}}]}),
            json!({"and":[{"a":{"match":{"value":"x"}}},{"b":{"match":{"value":"y"}}}]}),
            json!({"or":[{"a":{"match":{"value":"x"}}}]}),
        ] {
            parse_query_filter(Some(&cond))
                .unwrap_or_else(|e| panic!("shipped shape {cond} must still build: {e}"))
                .unwrap_or_else(|| panic!("shipped shape {cond} must produce a filter"));
        }
    }

    #[test]
    fn unparseable_datetime_bound_is_an_error_not_a_vacuous_range() {
        // Pre-fix: every bound failed to parse, yielding
        // `DatetimeRange{lt:None,gt:None,gte:None,lte:None}` — a condition
        // present in the request that matches EVERY point.
        let err = build_qdrant_filter("ts", "range", &json!({"gte":"not-a-date"})).unwrap_err();
        assert!(
            err.contains("gte"),
            "error should name the bad bound: {}",
            err
        );
    }

    #[test]
    fn partially_unparseable_datetime_bound_is_an_error_not_a_widened_range() {
        // The dangerous half-case: `gte` parses, `lt` does not, so the range
        // silently loses its upper bound and matches far more than asked.
        assert!(build_qdrant_filter(
            "ts",
            "range",
            &json!({"gte":"2023-01-01T00:00:00Z","lt":"whenever"}),
        )
        .is_err());
    }

    #[test]
    fn datetime_bounds_accept_the_same_forms_as_every_other_engine() {
        // The bound is now FATAL when it does not parse, so Qdrant must accept
        // every spelling the other engines do or a config that benchmarks fine
        // on five engines would kill the Qdrant run. That means both the wider
        // forms `parsers::datetime_to_epoch_secs` handles (naive, date-only)
        // AND the bare epoch-seconds string it rejects by design, which
        // redis.rs / valkey.rs pick up via `parse::<i64>`/`parse::<f64>`.
        for s in [
            "2023-01-01T00:00:00Z",
            "2023-01-01T00:00:00+00:00",
            "2023-01-01T00:00:00",
            "2023-01-01 00:00:00",
            "2023-01-01",
            "1672531200",
        ] {
            let c = build_qdrant_filter("ts", "range", &json!({ "gte": s }))
                .unwrap_or_else(|e| panic!("{} should parse: {}", s, e));
            let dt = field_condition(&c).datetime_range.expect("datetime range");
            assert_eq!(
                dt.gte.expect("gte bound").seconds,
                1_672_531_200,
                "{} should be 2023-01-01T00:00:00Z",
                s
            );
        }
    }

    #[test]
    fn numeric_match_any_is_never_silently_emptied() {
        // Pre-fix: `filter_map(as_str)` deleted every non-string member, so a
        // float list became an EMPTY MatchAny — matching nothing, i.e. recall 0
        // reported as an engine result. Qdrant's MatchValue has no float
        // variant, so this combination is a hard error (see build_qdrant_filter).
        let err = build_qdrant_filter("score", "match", &json!({"any":[1.5, 2.5]})).unwrap_err();
        assert!(
            err.contains("score"),
            "error should name the field: {}",
            err
        );

        // Mixed int/string is equally unrepresentable (MatchAny is homogeneous).
        // Pre-fix this did NOT empty the list — it silently NARROWED it to
        // `Keywords(["a"])`, dropping the integer member and under-matching.
        assert!(build_qdrant_filter("cat", "match", &json!({"any":["a", 1]})).is_err());
        // Mixed int/float too: not all-i64, so it would have been emptied.
        assert!(build_qdrant_filter("cat", "match", &json!({"any":[1, 2.5]})).is_err());
        // A null member is not a keyword either.
        assert!(build_qdrant_filter("cat", "match", &json!({"any":["a", null]})).is_err());
        // …and the error must propagate out of the whole condition tree rather
        // than degrading into "no filter" (which would run unfiltered).
        assert!(
            parse_qdrant_conditions(&json!({"and":[{"score":{"match":{"any":[1.5]}}}]})).is_err()
        );

        // Homogeneous lists still build.
        assert!(build_qdrant_filter("cat", "match", &json!({"any":[1, 2]})).is_ok());
        assert!(build_qdrant_filter("cat", "match", &json!({"any":["a"]})).is_ok());
    }

    #[test]
    fn boolean_match_any_builds_the_same_keyword_tokens_as_a_bool_value() {
        use qdrant_client::qdrant::r#match::MatchValue;
        // Bools are stored+indexed as the STRINGS "true"/"false" (see
        // `builds_bool_exact_match`), so a boolean `any` is a Keywords list of
        // those tokens — a faithful translation, not a substitution. Erroring
        // instead would kill the whole Qdrant run for a filter Elasticsearch
        // executes fine.
        let c = build_qdrant_filter("flag", "match", &json!({"any":[true, false]})).unwrap();
        let m = field_condition(&c).r#match.expect("match present");
        assert_eq!(
            m.match_value,
            Some(MatchValue::Keywords(
                qdrant_client::qdrant::RepeatedStrings {
                    strings: vec!["true".to_string(), "false".to_string()],
                }
            )),
            "boolean any must become the keyword tokens the payload actually holds"
        );
        assert!(build_qdrant_filter("flag", "match", &json!({"any":[true]})).is_ok());
        // …but a bool mixed with anything else is still unrepresentable.
        assert!(build_qdrant_filter("flag", "match", &json!({"any":[true, "x"]})).is_err());
    }

    #[test]
    fn malformed_condition_shapes_all_fail_rather_than_widen() {
        for criteria in [
            json!({"gte":"2023-01-01T00:00:00Z","lt":5}), // datetime range, numeric sibling
            json!({"lt": "not-a-date"}),
            json!("just-a-string"),
        ] {
            assert!(
                build_qdrant_filter("ts", "range", &criteria).is_err(),
                "{criteria} must fail"
            );
        }
        for cond in [
            json!({"and": {"color": {"match": {"value": "red"}}}}), // and is not an array
            json!({"or": 5}),
            json!({"and": [42, "str"]}),
            json!({"and":[{"color":{"match":{"value":"red"}}}], "or": []}),
            json!({"and":[], "or":[{"color":{"match":{"value":"red"}}}]}),
        ] {
            assert!(
                parse_query_filter(Some(&cond)).is_err(),
                "{cond} must fail the run, not filter approximately"
            );
        }
    }

    #[test]
    fn a_present_arm_that_constrains_nothing_is_an_error_even_if_the_other_arm_is_real() {
        // `{"and":[real], "or":[]}` builds `Filter{must:[1], should:[]}` — a
        // filter that IS `Some` and DOES constrain, just with a whole boolean
        // arm missing. Neither `must.is_none() && should.is_none()` nor the
        // call-site guard can see that, so the per-arm check has to.
        let cond = json!({"and":[{"color":{"match":{"value":"red"}}}], "or":[]});
        let err = parse_qdrant_conditions(&cond).unwrap_err();
        assert!(
            err.contains("`or`"),
            "error should name the empty arm: {err}"
        );
        // Mirror.
        let cond = json!({"or":[{"color":{"match":{"value":"red"}}}], "and":[]});
        assert!(parse_qdrant_conditions(&cond)
            .unwrap_err()
            .contains("`and`"));
    }

    #[test]
    fn a_none_parsed_filter_omits_the_filter_field_on_the_wire() {
        // The wire behaviour #222 is actually about, which is otherwise only
        // visible by reading the search loop: `None` means the request carries
        // NO filter at all — which Qdrant treats as match-all.
        let built =
            parse_query_filter(Some(&json!({"and":[{"c":{"match":{"value":"red"}}}]}))).unwrap();
        let parsed: Vec<Option<Filter>> = vec![None, built];
        for (idx, expect_filter) in [(0usize, false), (1usize, true)] {
            let mut qb = QueryPointsBuilder::new("c")
                .query(vec![0.0f32, 1.0])
                .limit(1);
            if let Some(f) = &parsed[idx] {
                qb = qb.filter(f.clone());
            }
            assert_eq!(
                qb.build().filter.is_some(),
                expect_filter,
                "parsed_filters[{idx}] = {:?} should {}attach a filter",
                parsed[idx],
                if expect_filter { "" } else { "NOT " }
            );
        }
    }

    #[test]
    fn empty_match_any_list_is_built_faithfully_not_rejected() {
        // Deliberate asymmetry with the float case above, and the line is
        // FAITHFULNESS, not hit count. `value ∈ ∅` is precisely what the config
        // wrote, precisely what Qdrant evaluates (live: 0 hits, no error), and
        // the ground truth for that same condition is empty too — nothing is
        // dropped and nothing is substituted. A float `any` fails not because
        // it returns zero rows but because the request that would go out is a
        // DIFFERENT query from the one the config expressed. pgvector agrees
        // (`match_any_empty_list_matches_nothing`), so the two stay comparable.
        assert!(build_qdrant_filter("cat", "match", &json!({"any":[]})).is_ok());
    }

    // ── Quantization config builder ─────────────────────────────────────────
    use super::build_quantization;
    use qdrant_client::qdrant::quantization_config::Quantization;
    use qdrant_client::qdrant::QuantizationType;

    #[test]
    fn quantization_scalar_int8_happy_path() {
        let q = build_quantization(
            &json!({"scalar":{"type":"int8","quantile":0.99,"always_ram":true}}),
        )
        .unwrap()
        .unwrap();
        match q {
            Quantization::Scalar(sq) => {
                assert_eq!(sq.r#type, i32::from(QuantizationType::Int8));
                assert_eq!(sq.quantile, Some(0.99));
                assert_eq!(sq.always_ram, Some(true));
            }
            other => panic!("expected Scalar, got {:?}", other),
        }
    }

    #[test]
    fn quantization_scalar_defaults_type_to_int8() {
        // Missing `type` defaults to Int8 (not an error).
        let q = build_quantization(&json!({"scalar":{}})).unwrap().unwrap();
        assert!(
            matches!(q, Quantization::Scalar(sq) if sq.r#type == i32::from(QuantizationType::Int8))
        );
    }

    #[test]
    fn quantization_scalar_bogus_type_errors() {
        let err = build_quantization(&json!({"scalar":{"type":"int4"}})).unwrap_err();
        assert_eq!(err, "Unsupported scalar quantization type: int4");
    }

    #[test]
    fn quantization_product_happy_path() {
        let q = build_quantization(&json!({"product":{"compression":"x8","always_ram":false}}))
            .unwrap()
            .unwrap();
        match q {
            Quantization::Product(pq) => {
                assert_eq!(pq.compression, i32::from(CompressionRatio::X8));
                assert_eq!(pq.always_ram, Some(false));
            }
            other => panic!("expected Product, got {:?}", other),
        }
    }

    #[test]
    fn quantization_product_without_compression_errors() {
        let err = build_quantization(&json!({"product":{}})).unwrap_err();
        assert_eq!(err, "Product quantization requires a `compression` value");
    }

    #[test]
    fn quantization_binary_happy_path() {
        let q = build_quantization(&json!({"binary":{"always_ram":true}}))
            .unwrap()
            .unwrap();
        assert!(matches!(q, Quantization::Binary(bq) if bq.always_ram == Some(true)));
    }

    #[test]
    fn quantization_none_when_no_known_key() {
        assert!(build_quantization(&json!({})).unwrap().is_none());
        assert!(build_quantization(&json!({"unknown":1})).unwrap().is_none());
    }
}
