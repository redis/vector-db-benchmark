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
    payload_index_params::IndexParams, BinaryQuantization, CompressionRatio, Condition,
    CreateCollectionBuilder, Datatype, DatetimeIndexParams, DatetimeRange, DeleteCollectionBuilder,
    Distance, FieldType, Filter, FloatIndexParams, Fusion, GeoIndexParams, HnswConfigDiff,
    IntegerIndexParams, KeywordIndexParams, MaxOptimizationThreads, NamedVectors,
    OptimizersConfigDiff, PointStruct, PrefetchQueryBuilder, ProductQuantization,
    QuantizationSearchParams, QuantizationType, Query, QueryPointsBuilder, ScalarQuantization,
    SearchParams as QdrantSearchParams, SparseIndexConfigBuilder, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, Timestamp, UuidIndexParams, Vector, VectorInput,
    VectorParamsBuilder, VectorsConfig, VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant};

use crate::config::{EngineConfig, HnswConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_COLLECTION: &str = "benchmark";

pub struct QdrantEngine {
    name: String,
    collection_name: String,
    /// Config knobs this run could NOT honour, accumulated during `configure`.
    /// Surfaced through `server_metadata()` so they land in the saved result JSON:
    /// a stderr line alone leaves the artifact indistinguishable from a run where
    /// the knob DID apply, and the artifact is what someone reads months later.
    ignored_config: Vec<String>,
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
    /// Typed `collection_params.hnsw_config`: m / ef_construct plus the on-disk
    /// knobs (`on_disk`, `payload_m`, `inline_storage`).
    hnsw: Option<HnswConfig>,
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

        // HNSW params come from the TYPED collection_params.hnsw_config field
        // (serde captures "m"/"ef_construct" there via aliases; the flattened
        // `extra` map never contains hnsw_config since it is a declared field).
        let hnsw = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.clone());

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
            ignored_config: Vec::new(),
            timeout,
            batch_size,
            parallel,
            grpc_url,
            rest_url,
            api_key,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            collection_params_extra,
            hnsw,
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

    fn create_collection(&mut self, dataset: &Dataset) -> Result<(), String> {
        if dataset.is_hybrid() {
            return self.create_hybrid_collection(dataset);
        }
        if dataset.is_sparse() {
            return self.create_sparse_collection(dataset);
        }

        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let qdrant_distance = map_qdrant_distance(distance)?;

        let vector_params = self.dense_vector_params(vector_size, qdrant_distance)?;

        let mut create_builder = CreateCollectionBuilder::new(&self.collection_name)
            .vectors_config(VectorsConfig {
                config: Some(Config::Params(vector_params.build())),
            });

        if let Some(hnsw_config) = self.hnsw_config_diff() {
            create_builder = create_builder.hnsw_config(hnsw_config);
        }

        // Pass through optimizers_config + quantization_config + on_disk_payload
        // (shared with the hybrid create path).
        create_builder = self.apply_optimizers_and_quantization(create_builder)?;

        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        // Disable optimization during indexing.
        self.disable_indexing_optimizers();

        self.create_payload_indexes(dataset)?;

        Ok(())
    }

    /// Build the dense `VectorParams` from `collection_params.vectors_config`,
    /// honouring `on_disk` (mmap the vectors) and `datatype`
    /// (`float32`/`float16`/`uint8` — half/byte storage roughly halves or
    /// quarters the vector footprint, so leaving it unread silently benchmarked
    /// full-precision storage instead of what the config asked for).
    ///
    /// `on_disk` is forwarded ONLY when the config actually declares it. An
    /// explicit `false` is NOT equivalent to omitting the field: verified on
    /// qdrant v1.18.2 with `optimizers_config.memmap_threshold: 1` by inspecting
    /// the resulting segment layout —
    ///
    /// * omitted        -> `vector_storage/matrix.dat`            (mmap'd, on disk)
    /// * explicit false -> `vector_storage/vectors/chunk_0.mmap`  (kept in RAM)
    /// * explicit true  -> `vector_storage/matrix.dat`
    ///
    /// so an explicit `false` OVERRIDES `memmap_threshold`. Defaulting to
    /// `false` therefore silently disabled mmap for every experiment that drives
    /// on-disk storage through the threshold alone: `qdrant-on-disk-default` and
    /// all six `qdrant-mmap-*` configurations ran their vectors in RAM while
    /// their names, configs and results all said otherwise.
    fn dense_vector_params(
        &self,
        vector_size: i64,
        qdrant_distance: Distance,
    ) -> Result<VectorParamsBuilder, String> {
        let vectors_config = self.collection_params_extra.get("vectors_config");

        let mut params = VectorParamsBuilder::new(vector_size as u64, qdrant_distance);
        if let Some(on_disk) = vectors_config
            .and_then(|v| v.get("on_disk"))
            .and_then(|v| v.as_bool())
        {
            params = params.on_disk(on_disk);
        }

        // An unrecognised datatype is a hard error, not a silent fallback to
        // float32: the whole point of setting it is to measure that storage.
        if let Some(dt) = vectors_config
            .and_then(|v| v.get("datatype"))
            .and_then(|v| v.as_str())
        {
            params = params.datatype(parse_datatype(dt)?);
        }
        Ok(params)
    }

    /// Build the named "sparse" vector params, mapping `vectors_config`'s
    /// `on_disk` / `datatype` onto the SPARSE inverted index (which has its own
    /// equivalents). Shared by the sparse-only and hybrid create paths — the
    /// hybrid path previously built a bare default here, so an "all-on-disk"
    /// hybrid run put the dense half on disk and silently left the sparse half in
    /// RAM at full precision.
    fn sparse_vector_params(&self) -> Result<SparseVectorParamsBuilder, String> {
        let vectors_config = self.collection_params_extra.get("vectors_config");
        let mut index = SparseIndexConfigBuilder::default();
        if let Some(on_disk) = vectors_config
            .and_then(|v| v.get("on_disk"))
            .and_then(|v| v.as_bool())
        {
            index = index.on_disk(on_disk);
        }
        if let Some(dt) = vectors_config
            .and_then(|v| v.get("datatype"))
            .and_then(|v| v.as_str())
        {
            index = index.datatype(parse_datatype(dt)?);
        }
        Ok(SparseVectorParamsBuilder::default().index(index.build()))
    }

    /// Build the Qdrant `HnswConfigDiff` from the typed
    /// `collection_params.hnsw_config`. Returns `None` when nothing was
    /// configured, leaving Qdrant's own defaults in place rather than sending an
    /// empty diff. Shared by the dense and hybrid create paths.
    ///
    /// `m: 0` combined with `payload_m` is meaningful (build per-payload-value
    /// graphs only — the multi-tenancy layout), so a zero is passed through
    /// rather than treated as unset.
    fn hnsw_config_diff(&self) -> Option<HnswConfigDiff> {
        let h = self.hnsw.as_ref()?;
        // Any key inside hnsw_config that no engine reads is reported, not
        // dropped — silently discarding one is the bug this branch exists to fix.
        let unsupported = h.unsupported_keys();
        if !unsupported.is_empty() {
            eprintln!(
                "Warning: unsupported collection_params.hnsw_config keys ignored: {} \
                 (supported: m, ef_construct, on_disk, payload_m, inline_storage, \
                 full_scan_threshold, max_indexing_threads)",
                unsupported.join(", ")
            );
        }
        if h.m.is_none()
            && h.ef_construction.is_none()
            && h.on_disk.is_none()
            && h.payload_m.is_none()
            && h.inline_storage.is_none()
            && h.full_scan_threshold.is_none()
            && h.max_indexing_threads.is_none()
        {
            return None;
        }
        Some(HnswConfigDiff {
            m: h.m.map(|v| v as u64),
            ef_construct: h.ef_construction.map(|v| v as u64),
            on_disk: h.on_disk,
            payload_m: h.payload_m.map(|v| v as u64),
            inline_storage: h.inline_storage,
            full_scan_threshold: h.full_scan_threshold.map(|v| v as u64),
            max_indexing_threads: h.max_indexing_threads.map(|v| v as u64),
        })
    }

    /// Apply the collection tuning that is valid for ANY collection shape:
    /// `collection_params.optimizers_config` (rps-tuned segment / memmap knobs)
    /// and `collection_params.on_disk_payload`. Shared by the dense, hybrid and
    /// sparse create paths — quantization is separate because it is dense-only.
    fn apply_optimizers_and_payload_storage(
        &self,
        mut create_builder: CreateCollectionBuilder,
    ) -> CreateCollectionBuilder {
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
            // `max_optimization_threads` cannot be honoured: this harness forces
            // it to 0 for the ingest window (disable_indexing_optimizers) and to
            // Auto afterwards (wait_collection_green) so that index-build time is
            // measured the same way for every engine. Upstream's configs do set
            // it, so say what we do with it instead of dropping it silently.
            if opt.get("max_optimization_threads").is_some() {
                eprintln!(
                    "Warning: collection_params.optimizers_config.max_optimization_threads is \
                     ignored — this harness pins it to 0 during upload and Auto during the index \
                     wait, so index-build time stays comparable across engines"
                );
            }
            // Report anything else rather than dropping it: this allow-list is
            // narrower than Qdrant's optimizer config (indexing_threshold,
            // flush_interval_sec, ...), so an unread knob here looks honoured.
            if let Some(obj) = opt.as_object() {
                let unknown: Vec<&str> = obj
                    .keys()
                    .map(|k| k.as_str())
                    .filter(|k| {
                        !matches!(
                            *k,
                            "default_segment_number"
                                | "max_segment_size"
                                | "memmap_threshold"
                                | "max_optimization_threads"
                        )
                    })
                    .collect();
                if !unknown.is_empty() {
                    eprintln!(
                        "Warning: unsupported collection_params.optimizers_config keys ignored: {} \
                         (supported: default_segment_number, max_segment_size, memmap_threshold)",
                        unknown.join(", ")
                    );
                }
            }
            create_builder = create_builder.optimizers_config(diff);
        }

        // Keep payloads on disk instead of in RAM — the on-disk experiments pair
        // this with vectors_config.on_disk and hnsw_config.on_disk.
        if let Some(v) = self
            .collection_params_extra
            .get("on_disk_payload")
            .and_then(|v| v.as_bool())
        {
            create_builder = create_builder.on_disk_payload(v);
        }
        create_builder
    }

    /// Apply `collection_params.quantization_config` (scalar / product / binary).
    /// DENSE-only: Qdrant quantizes dense vectors, so the sparse path warns
    /// rather than calling this.
    fn apply_quantization(
        &self,
        mut create_builder: CreateCollectionBuilder,
    ) -> Result<CreateCollectionBuilder, String> {
        if let Some(q) = self.collection_params_extra.get("quantization_config") {
            if let Some(quantization) = build_quantization(q)? {
                create_builder = create_builder.quantization_config(quantization);
            }
        }
        Ok(create_builder)
    }

    /// Both of the above, for the dense and hybrid create paths.
    fn apply_optimizers_and_quantization(
        &self,
        create_builder: CreateCollectionBuilder,
    ) -> Result<CreateCollectionBuilder, String> {
        let create_builder = self.apply_optimizers_and_payload_storage(create_builder);
        self.apply_quantization(create_builder)
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
    ///
    /// `collection_params.payload_index_params` refines individual keyword/uuid
    /// indexes with `is_tenant` (group points by that value on disk — the
    /// multi-tenancy layout) and `on_disk` (keep the index off the heap).
    ///
    /// An entry that cannot take effect — a field absent from the dataset schema,
    /// or one whose type has no `is_tenant`/`on_disk` (int, float, geo, text) —
    /// WARNS loudly and continues. It is deliberately not fatal: upstream's own
    /// `qdrant-on-disk.json` names fields (`a`, `d`) that our dataset schemas do
    /// not all declare, and rejecting those would make the upstream file
    /// unrunnable verbatim, which is the opposite of this branch's goal. The
    /// warning is what stops the run from quietly passing as tenant-optimised.
    fn create_payload_indexes(&mut self, dataset: &Dataset) -> Result<(), String> {
        let schema = dataset.config.schema.as_ref().and_then(|s| s.as_object());
        let index_params = self
            .collection_params_extra
            .get("payload_index_params")
            .and_then(|v| v.as_object());

        for warning in payload_index_warnings(schema, index_params, &dataset.config.name) {
            eprintln!("Warning: {}", warning);
            self.ignored_config.push(warning);
        }

        let Some(schema_obj) = schema else {
            return Ok(());
        };
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

            let mut builder = qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
                &self.collection_name,
                field_name.clone(),
                qdrant_type,
            );
            let field_params = index_params.and_then(|p| p.get(field_name));
            if let Some(params) = build_payload_index_params(ft, field_params) {
                builder = builder.field_index_params(params);
            }

            // FATAL, deliberately: `configure` always deletes and recreates the
            // collection first, so "index already exists" is unreachable here and
            // every error means the index is genuinely missing. Qdrant still
            // filters correctly without one, so recall looks healthy while
            // latency/QPS are garbage — the hardest kind of wrong number to spot.
            self.rt
                .block_on(self.client.create_field_index(builder))
                .map_err(|e| {
                    format!(
                        "failed to create the {} payload index on {:?}: {} (a missing payload \
                         index silently turns every filtered query into a full scan)",
                        ft, field_name, e
                    )
                })?;
        }
        Ok(())
    }

    /// Create a sparse-vector collection with a single named "sparse" vector.
    ///
    /// Honours the SAME collection tuning as the dense path — `optimizers_config`,
    /// `on_disk_payload`, and `vectors_config`'s `on_disk` / `datatype`, the last
    /// two mapped onto the sparse index (which has its own equivalents). Without
    /// this, running a sparse dataset under an on-disk config reported under an
    /// "on-disk" run name while being entirely in memory.
    ///
    /// `hnsw_config` and `quantization_config` have no sparse equivalent in
    /// Qdrant. They are NOT silently ignored: each warns, so a config asking for
    /// something the sparse index cannot do says so instead of producing a
    /// mislabelled result.
    fn create_sparse_collection(&mut self, dataset: &Dataset) -> Result<(), String> {
        let mut sparse_cfg = SparseVectorsConfigBuilder::default();
        sparse_cfg.add_named_vector_params("sparse", self.sparse_vector_params()?);

        let mut create_builder =
            CreateCollectionBuilder::new(&self.collection_name).sparse_vectors_config(sparse_cfg);
        create_builder = self.apply_optimizers_and_payload_storage(create_builder);

        for warning in sparse_ignored_warnings(
            self.hnsw.is_some(),
            self.collection_params_extra
                .get("quantization_config")
                .is_some(),
            &dataset.config.name,
        ) {
            eprintln!("Warning: {}", warning);
            self.ignored_config.push(warning);
        }

        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create sparse collection: {}", e))?;

        // Same optimizer regime as the dense and hybrid paths (0 threads during
        // ingest, Auto once green), so sparse index-build time is measured the
        // same way — and so the max_optimization_threads warning is true here too.
        self.disable_indexing_optimizers();

        self.create_payload_indexes(dataset)?;
        Ok(())
    }

    /// Create a HYBRID collection with a named dense vector ("dense") AND a named
    /// sparse vector ("sparse"), so searches can fuse a dense-vector prefetch and
    /// a sparse-vector prefetch server-side (RRF). The dense vector carries the
    /// dataset's distance metric (and HNSW config, if configured); the sparse
    /// vector uses Qdrant's default sparse index.
    fn create_hybrid_collection(&mut self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();
        let qdrant_distance = map_qdrant_distance(distance)?;

        // Named dense vector "dense" (same vectors_config + HNSW handling as the
        // dense-only path, so on_disk/datatype/payload_m are not silently lost
        // just because the dataset is hybrid).
        let mut dense_params = self.dense_vector_params(vector_size, qdrant_distance)?;
        if let Some(hnsw_config) = self.hnsw_config_diff() {
            dense_params = dense_params.hnsw_config(hnsw_config);
        }
        let mut dense_cfg = VectorsConfigBuilder::default();
        dense_cfg.add_named_vector_params("dense", dense_params);

        // Named sparse vector "sparse" — shares sparse_vector_params() with the
        // sparse-only path, so vectors_config's on_disk/datatype reaches BOTH
        // halves of a hybrid collection.
        let mut sparse_cfg = SparseVectorsConfigBuilder::default();
        sparse_cfg.add_named_vector_params("sparse", self.sparse_vector_params()?);

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

        self.create_payload_indexes(dataset)?;
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

fn parse_qdrant_conditions(conditions: &serde_json::Value) -> Option<Filter> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let must = obj
        .get("and")
        .and_then(|v| v.as_array())
        .map(|entries| build_qdrant_subfilters(entries));
    let should = obj
        .get("or")
        .and_then(|v| v.as_array())
        .map(|entries| build_qdrant_subfilters(entries));

    if must.is_none() && should.is_none() {
        return None;
    }

    let mut filter = Filter::default();
    if let Some(m) = must {
        filter.must = m;
    }
    if let Some(s) = should {
        filter.should = s;
    }

    Some(filter)
}

fn build_qdrant_subfilters(entries: &[serde_json::Value]) -> Vec<Condition> {
    let mut filters = Vec::new();
    for entry in entries {
        let Some(entry_obj) = entry.as_object() else {
            continue;
        };
        // NESTED GROUP: an entry that is itself an `{and:[...]}` / `{or:[...]}`
        // sub-tree must be built as its OWN sub-Filter and nested via a Filter
        // condition, so grouping is preserved natively — e.g.
        // `(color==red AND size>=50) OR (color==blue AND size<10)` becomes a
        // top-level `should` of two nested Filters, each with its own `must`.
        // Flattening the sub-tree's leaves into the parent must/should would
        // change the boolean meaning and collapse recall.
        if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
            if let Some(sub) = parse_qdrant_conditions(entry) {
                filters.push(Condition {
                    condition_one_of: Some(
                        qdrant_client::qdrant::condition::ConditionOneOf::Filter(sub),
                    ),
                });
            }
            continue;
        }
        // LEAF: `{ field: { op: criteria } }`.
        for (field_name, field_filters) in entry_obj {
            if let Some(filter_obj) = field_filters.as_object() {
                for (cond_type, criteria) in filter_obj {
                    if let Some(f) = build_qdrant_filter(field_name, cond_type, criteria) {
                        filters.push(f);
                    }
                }
            }
        }
    }
    filters
}

/// Parse an ISO-8601 / RFC 3339 datetime string into a protobuf Timestamp.
fn parse_rfc3339_timestamp(s: &str) -> Option<Timestamp> {
    let dt = chrono::DateTime::parse_from_rfc3339(s).ok()?;
    Some(Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    })
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

/// Every `payload_index_params` entry that cannot take effect, as a list of
/// warning strings.
///
/// Pure so it is testable: `create_payload_indexes` has no error return left
/// (an unusable key warns, and a failed index creation warns), so a test that
/// only asserted `is_ok()` would pass even if this logic were deleted — and
/// deleting it restores the "run quietly passes as tenant-optimised" failure
/// this branch exists to remove.
fn payload_index_warnings(
    schema: Option<&serde_json::Map<String, serde_json::Value>>,
    index_params: Option<&serde_json::Map<String, serde_json::Value>>,
    dataset_name: &str,
) -> Vec<String> {
    let Some(params) = index_params else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (field, spec) in params {
        match schema.and_then(|s| s.get(field)).and_then(|v| v.as_str()) {
            None => out.push(format!(
                "collection_params.payload_index_params names {:?}, which is NOT in the schema of \
                 dataset {} — no index is created for it, so {} has no effect on this run",
                field, dataset_name, spec
            )),
            // `text` cannot carry either parameter here — see
            // build_payload_index_params for why (required tokenizer field).
            Some("text") => out.push(format!(
                "payload_index_params.{} is ignored — a Qdrant text index needs an explicit \
                 tokenizer, so this harness sends no index params for text fields",
                field
            )),
            // `on_disk` works on every other index type; `is_tenant` only on
            // keyword/uuid (and bool, which we index as a keyword).
            Some(ft)
                if spec.get("is_tenant").is_some()
                    && !matches!(ft, "keyword" | "uuid" | "bool") =>
            {
                out.push(format!(
                    "payload_index_params.{}.is_tenant is ignored — a Qdrant {} index has no \
                     is_tenant (only keyword and uuid do; this harness also accepts it for bool, \
                     which it indexes AS a keyword); its on_disk setting, if any, IS applied",
                    field, ft
                ))
            }
            Some(_) => {}
        }
    }
    out
}

/// The collection params a SPARSE collection cannot honour, as warning strings.
/// Pure for the same reason as `payload_index_warnings`: these warnings are the
/// only signal that a run named `*-bq` is not actually quantized.
fn sparse_ignored_warnings(
    hnsw_configured: bool,
    quantization_configured: bool,
    dataset_name: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    if hnsw_configured {
        out.push(format!(
            "collection_params.hnsw_config is ignored for the SPARSE dataset {} (Qdrant's sparse \
             index has no HNSW graph)",
            dataset_name
        ));
    }
    if quantization_configured {
        out.push(format!(
            "collection_params.quantization_config is ignored for the SPARSE dataset {} (Qdrant \
             quantization applies to dense vectors only) — this run is NOT quantized",
            dataset_name
        ));
    }
    out
}

/// Map a `vectors_config.datatype` string onto Qdrant's `Datatype`.
///
/// `"default"` maps to `Datatype::Default` (let Qdrant choose), NOT to
/// `Float32` — conflating them would misreport which storage was measured.
/// An unrecognised value is an error rather than a silent fallback.
fn parse_datatype(dt: &str) -> Result<Datatype, String> {
    match dt.to_lowercase().as_str() {
        "default" => Ok(Datatype::Default),
        "float32" | "f32" => Ok(Datatype::Float32),
        "float16" | "f16" => Ok(Datatype::Float16),
        "uint8" | "u8" => Ok(Datatype::Uint8),
        other => Err(format!(
            "unknown vectors_config.datatype {:?} (expected default, float32, float16 or uint8)",
            other
        )),
    }
}

/// Build the per-field `IndexParams` for one schema field from its
/// `payload_index_params` entry (`is_tenant` / `on_disk`).
///
/// The two parameters have DIFFERENT support in Qdrant, and conflating them
/// silently drops config:
/// * `on_disk` exists on every index-params message — keyword, uuid, integer,
///   float, geo, text, bool and datetime.
/// * `is_tenant` (group points by that value on disk) exists ONLY on keyword and
///   uuid. `bool` is mapped onto the keyword index here for the same reason it is
///   above: bools are stored as the strings "true"/"false".
///
/// So `{"some_int": {"on_disk": true}}` IS honoured, while `is_tenant` on a
/// non-keyword field cannot be and is reported by the caller. Returns `None`
/// when there is nothing to refine, leaving the index on plain defaults.
fn build_payload_index_params(
    field_type: &str,
    field_params: Option<&serde_json::Value>,
) -> Option<IndexParams> {
    let params = field_params?;
    let is_tenant = params.get("is_tenant").and_then(|v| v.as_bool());
    let on_disk = params.get("on_disk").and_then(|v| v.as_bool());
    if is_tenant.is_none() && on_disk.is_none() {
        return None;
    }

    Some(match field_type {
        "keyword" | "bool" => IndexParams::KeywordIndexParams(KeywordIndexParams {
            is_tenant,
            on_disk,
            ..Default::default()
        }),
        "uuid" => IndexParams::UuidIndexParams(UuidIndexParams {
            is_tenant,
            on_disk,
            ..Default::default()
        }),
        // `text` is deliberately absent: TextIndexParams.tokenizer is the one
        // REQUIRED (non-optional) proto field across the index-params messages,
        // so emitting the message to carry on_disk would pin the tokenizer to
        // Unknown(0) — either rejected by the server (leaving NO text index, so
        // every full-text filter degrades silently) or a tokenizer the config
        // never asked for. The caller warns instead.
        "text" => return None,
        // is_tenant does not exist on the rest, so there is nothing to send
        // unless on_disk was actually asked for.
        _ if on_disk.is_none() => return None,
        "int" => IndexParams::IntegerIndexParams(IntegerIndexParams {
            on_disk,
            ..Default::default()
        }),
        "float" => IndexParams::FloatIndexParams(FloatIndexParams {
            on_disk,
            ..Default::default()
        }),
        "geo" => IndexParams::GeoIndexParams(GeoIndexParams {
            on_disk,
            ..Default::default()
        }),
        "datetime" => IndexParams::DatetimeIndexParams(DatetimeIndexParams {
            on_disk,
            ..Default::default()
        }),
        _ => return None,
    })
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
    } else if q.get("binary").is_none() {
        // Neither scalar, product nor binary: report it. Returning Ok(None) here
        // silently un-quantizes a run whose config name says otherwise — the same
        // silent-drop class this branch exists to remove.
        if let Some(obj) = q.as_object() {
            eprintln!(
                "Warning: unrecognised collection_params.quantization_config key(s) {} — this run \
                 is NOT quantized (expected one of: scalar, product, binary)",
                obj.keys().cloned().collect::<Vec<_>>().join(", ")
            );
        }
        Ok(None)
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

fn build_qdrant_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<Condition> {
    match condition_type {
        "match" => {
            let criteria_obj = criteria.as_object()?;
            // match_any: value in a list (keywords or integers).
            if let Some(any) = criteria_obj.get("any").and_then(|v| v.as_array()) {
                if !any.is_empty() && any.iter().all(|v| v.is_i64()) {
                    let vals: Vec<i64> = any.iter().filter_map(|v| v.as_i64()).collect();
                    return Some(Condition::matches(field_name.to_string(), vals));
                }
                let vals: Vec<String> = any
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                return Some(Condition::matches(field_name.to_string(), vals));
            }
            // match_text: full-text match.
            if let Some(text) = criteria_obj.get("text").and_then(|v| v.as_str()) {
                return Some(Condition::matches_text(
                    field_name.to_string(),
                    text.to_string(),
                ));
            }
            // exact match on keyword / integer / bool.
            let value = criteria_obj.get("value")?;
            if let Some(s) = value.as_str() {
                Some(Condition::matches(field_name.to_string(), s.to_string()))
            } else if let Some(b) = value.as_bool() {
                // Bools are stored+indexed as the STRING "true"/"false", so match
                // the string form. A native boolean Match never matches the string
                // payload and silently returns zero points (0 recall).
                let token = if b { "true" } else { "false" };
                Some(Condition::matches(
                    field_name.to_string(),
                    token.to_string(),
                ))
            } else {
                value
                    .as_i64()
                    .map(|n| Condition::matches(field_name.to_string(), n))
            }
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            // A string bound means an ISO-8601 datetime range rather than numeric.
            let is_datetime = ["lt", "gt", "lte", "gte"]
                .iter()
                .any(|k| criteria_obj.get(*k).map(|v| v.is_string()).unwrap_or(false));
            if is_datetime {
                let ts = |k: &str| {
                    criteria_obj
                        .get(k)
                        .and_then(|v| v.as_str())
                        .and_then(parse_rfc3339_timestamp)
                };
                return Some(Condition::datetime_range(
                    field_name.to_string(),
                    DatetimeRange {
                        lt: ts("lt"),
                        gt: ts("gt"),
                        gte: ts("gte"),
                        lte: ts("lte"),
                    },
                ));
            }
            let mut range = qdrant_client::qdrant::Range::default();
            if let Some(lt) = criteria_obj.get("lt").and_then(|v| v.as_f64()) {
                range.lt = Some(lt);
            }
            if let Some(gt) = criteria_obj.get("gt").and_then(|v| v.as_f64()) {
                range.gt = Some(gt);
            }
            if let Some(lte) = criteria_obj.get("lte").and_then(|v| v.as_f64()) {
                range.lte = Some(lte);
            }
            if let Some(gte) = criteria_obj.get("gte").and_then(|v| v.as_f64()) {
                range.gte = Some(gte);
            }
            // If NO valid numeric bound was produced (all bounds unknown-op /
            // null / non-numeric), SKIP the clause (return None) rather than
            // emitting an empty, unconstraining Range — a present-but-vacuous
            // condition that other engines drop. Mirrors the vectorsets
            // null-bound fix (#115). A range with SOME valid bounds still
            // produces a Range carrying just those.
            if range.lt.is_none()
                && range.gt.is_none()
                && range.lte.is_none()
                && range.gte.is_none()
            {
                return None;
            }
            Some(Condition::range(field_name.to_string(), range))
        }
        "geo" => {
            let lat = criteria.get("lat")?.as_f64()?;
            let lon = criteria.get("lon")?.as_f64()?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(Condition::geo_radius(
                field_name.to_string(),
                qdrant_client::qdrant::GeoRadius {
                    center: Some(qdrant_client::qdrant::GeoPoint { lon, lat }),
                    radius: radius as f32,
                },
            ))
        }
        _ => None,
    }
}

impl Engine for QdrantEngine {
    fn name(&self) -> &str {
        &self.name
    }

    /// Qdrant is the only engine with a sparse / hybrid path.
    fn supports_sparse(&self) -> bool {
        true
    }

    /// Report the config knobs this run could not honour, so they land in the
    /// saved result JSON. Without this the artifact is identical to a run where
    /// the knob DID apply — and the artifact, not a scrolled-past stderr line, is
    /// what gets read later.
    fn server_metadata(&mut self) -> Option<serde_json::Value> {
        if self.ignored_config.is_empty() {
            return None;
        }
        Some(serde_json::json!({ "ignored_collection_params": self.ignored_config }))
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

        // Every search knob in THIS engine resolves through `knob()` (nested
        // under `search_params`/`config`, or flat), so an entry cannot be
        // half-applied. (Other engines vary; see SearchParams::knob.)
        // The asymmetry this replaces was worse than a plain drop: with only
        // `with_payload` accepting the flat spelling, `{"with_payload": true,
        // "hnsw_ef": 128}` returned payloads while silently searching at DEFAULT
        // ef, i.e. the config looked honoured and was not.
        // `ef` is exempt: serde captures it into the typed `search_params.ef`
        // field, which `knob()` cannot see (see SearchParams::knob).
        let hnsw_ef: Option<u64> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .map(|e| e as u64)
            .or_else(|| params.knob("hnsw_ef").and_then(|v| v.as_u64()));

        // Whether to return payloads with each hit. Default false — recall only
        // needs ids, and shipping payloads back would tax the wire for every
        // engine unevenly. Upstream's on-disk experiments set it true on purpose,
        // to price payload retrieval, so honour it when asked.
        let with_payload = params
            .knob("with_payload")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Search only the already-indexed portion of the collection. Upstream's
        // qdrant-single-node.json sets this and it materially changes RPS, so
        // dropping it while honouring its neighbours in the same object would
        // misreport a "faithfully tuned" run.
        let indexed_only = params
            .knob("indexed_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Search-time quantization params (rescore/oversampling) from the config's
        // search_params.quantization object — mirrors python rest.SearchParams(**params).
        let quantization_params: Option<QuantizationSearchParams> = params
            .knob("quantization")
            .map(|q| QuantizationSearchParams {
                rescore: q.get("rescore").and_then(|v| v.as_bool()),
                oversampling: q.get("oversampling").and_then(|v| v.as_f64()),
                ..Default::default()
            });

        // Prefetch (two-stage retrieval / rescoring): search_params.prefetch =
        // { "limit": N, "params": { "hnsw_ef": .., "quantization": {..} } }.
        // Mirrors python `models.Prefetch(**prefetch, query=query_vector)`.
        let prefetch = params.knob("prefetch");
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
            let pf: Vec<Option<Filter>> = conditions
                .iter()
                .map(|c| c.as_ref().and_then(parse_qdrant_conditions))
                .collect();
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
                            let mut qb = QueryPointsBuilder::new(collection_name.clone())
                                .query(Query::new_fusion(Fusion::Rrf))
                                .prefetch(vec![dense_pf, sparse_pf])
                                .limit(top as u64)
                                .with_payload(with_payload);
                            // indexed_only is collection-level, not vector-kind
                            // specific, so it must reach the fused query too —
                            // otherwise it is honoured for dense runs and
                            // silently dropped for hybrid ones.
                            if indexed_only {
                                qb = qb.params(QdrantSearchParams {
                                    indexed_only: Some(true),
                                    ..Default::default()
                                });
                            }
                            qb
                        } else if is_sparse {
                            let sv = &sparse_queries[idx];
                            let mut qb = QueryPointsBuilder::new(collection_name.clone())
                                .query(VectorInput::new_sparse(
                                    sv.indices.clone(),
                                    sv.values.clone(),
                                ))
                                .using("sparse")
                                .limit(top as u64)
                                .with_payload(with_payload);
                            if indexed_only {
                                qb = qb.params(QdrantSearchParams {
                                    indexed_only: Some(true),
                                    ..Default::default()
                                });
                            }
                            qb
                        } else {
                            let mut qb = QueryPointsBuilder::new(collection_name.clone())
                                .query(queries[idx].clone())
                                .limit(top as u64)
                                .with_payload(with_payload);
                            // `indexed_only` must widen this guard: gating on
                            // hnsw_ef/quantization alone would drop a config that
                            // sets ONLY indexed_only.
                            if hnsw_ef.is_some() || quantization_params.is_some() || indexed_only {
                                qb = qb.params(QdrantSearchParams {
                                    hnsw_ef,
                                    quantization: quantization_params,
                                    indexed_only: Some(indexed_only),
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
        build_payload_index_params, build_qdrant_filter, parse_compression_ratio,
        parse_qdrant_metrics, parse_rfc3339_timestamp, IndexParams, QdrantEngine,
    };
    use crate::config::EngineConfig;
    use qdrant_client::qdrant::{condition::ConditionOneOf, CompressionRatio, FieldCondition};
    use serde_json::json;

    fn field_condition(c: &qdrant_client::qdrant::Condition) -> FieldCondition {
        match c.condition_one_of.clone().unwrap() {
            ConditionOneOf::Field(fc) => fc,
            other => panic!("expected FieldCondition, got {:?}", other),
        }
    }

    /// Build an engine from a `collection_params` object. The gRPC client is lazy,
    /// so this needs no live Qdrant — it exercises config → request translation.
    fn engine_with_collection_params(collection_params: serde_json::Value) -> QdrantEngine {
        let cfg: EngineConfig = serde_json::from_value(json!({
            "name": "qdrant-unit",
            "engine": "qdrant",
            "collection_params": collection_params,
        }))
        .expect("engine config should parse");
        QdrantEngine::new(&cfg, "localhost").expect("client construction is lazy")
    }

    // ── collection_params.hnsw_config: the on-disk knobs ───────────────────

    /// `m: 0` + `payload_m` is the multi-tenancy layout (no global graph, one
    /// graph per payload value). Zero must survive as Some(0), not be dropped.
    #[test]
    fn hnsw_diff_forwards_on_disk_payload_m_and_inline_storage() {
        let e = engine_with_collection_params(json!({
            "hnsw_config": { "m": 0, "ef_construct": 256, "on_disk": true,
                             "payload_m": 16, "inline_storage": true }
        }));
        let diff = e
            .hnsw_config_diff()
            .expect("configured hnsw must yield a diff");
        assert_eq!(
            diff.m,
            Some(0),
            "m: 0 must be forwarded, not treated as unset"
        );
        assert_eq!(diff.ef_construct, Some(256));
        assert_eq!(diff.on_disk, Some(true));
        assert_eq!(diff.payload_m, Some(16));
        assert_eq!(diff.inline_storage, Some(true));
    }

    /// No hnsw_config, or one carrying nothing we understand, leaves Qdrant's
    /// defaults alone rather than sending an empty diff.
    #[test]
    fn hnsw_diff_is_none_when_unconfigured() {
        assert!(engine_with_collection_params(json!({}))
            .hnsw_config_diff()
            .is_none());
        assert!(engine_with_collection_params(json!({"hnsw_config": {}}))
            .hnsw_config_diff()
            .is_none());
    }

    // ── collection_params.vectors_config.datatype ──────────────────────────

    #[test]
    fn dense_vector_params_forwards_datatype_and_on_disk() {
        for (spelling, expected) in [
            ("float16", super::Datatype::Float16),
            ("f16", super::Datatype::Float16),
            ("uint8", super::Datatype::Uint8),
            ("float32", super::Datatype::Float32),
        ] {
            let e = engine_with_collection_params(
                json!({"vectors_config": {"on_disk": true, "datatype": spelling}}),
            );
            let params = e
                .dense_vector_params(128, super::Distance::Cosine)
                .unwrap()
                .build();
            assert_eq!(
                params.datatype,
                Some(expected as i32),
                "datatype {:?} should map to {:?}",
                spelling,
                expected
            );
            assert_eq!(params.on_disk, Some(true));
        }
    }

    /// Silently falling back to float32 would benchmark different storage than
    /// the config asked for, so an unknown datatype must fail the run.
    #[test]
    fn dense_vector_params_rejects_unknown_datatype() {
        let e = engine_with_collection_params(json!({"vectors_config": {"datatype": "bfloat16"}}));
        let err = match e.dense_vector_params(128, super::Distance::Cosine) {
            Err(e) => e,
            Ok(_) => panic!("an unknown datatype must not be accepted"),
        };
        assert!(err.contains("bfloat16"), "got: {}", err);
    }

    /// REGRESSION: an OMITTED `vectors_config.on_disk` must stay omitted on the
    /// wire — it must NOT be sent as an explicit `false`.
    ///
    /// Verified on qdrant v1.18.2 with `memmap_threshold: 1` by inspecting the
    /// segment layout: omitted -> `vector_storage/matrix.dat` (mmap'd);
    /// explicit `false` -> `vector_storage/vectors/chunk_0.mmap` (RAM);
    /// `true` -> `matrix.dat`. An explicit `false` OVERRIDES `memmap_threshold`,
    /// so `unwrap_or(false)` kept the vectors in RAM for `qdrant-on-disk-default`
    /// and all six `qdrant-mmap-*` configurations.
    #[test]
    fn omitted_vectors_on_disk_is_not_sent_as_explicit_false() {
        let e = engine_with_collection_params(json!({}));
        let params = e
            .dense_vector_params(64, super::Distance::Cosine)
            .unwrap()
            .build();
        assert_eq!(params.datatype, None);
        assert_eq!(
            params.on_disk, None,
            "an omitted on_disk must not be sent at all — an explicit false \
             overrides optimizers_config.memmap_threshold and pins the vectors in RAM"
        );

        // A vectors_config that exists but says nothing about on_disk is still
        // "omitted".
        let e = engine_with_collection_params(json!({"vectors_config": {"datatype": "uint8"}}));
        assert_eq!(
            e.dense_vector_params(64, super::Distance::Cosine)
                .unwrap()
                .build()
                .on_disk,
            None
        );

        // Both explicit values are forwarded verbatim.
        for want in [true, false] {
            let e = engine_with_collection_params(json!({"vectors_config": {"on_disk": want}}));
            assert_eq!(
                e.dense_vector_params(64, super::Distance::Cosine)
                    .unwrap()
                    .build()
                    .on_disk,
                Some(want)
            );
        }
    }

    // ── collection_params.payload_index_params ─────────────────────────────

    #[test]
    fn payload_index_params_refine_keyword_and_uuid_only() {
        let p = json!({"is_tenant": true, "on_disk": true});
        match build_payload_index_params("keyword", Some(&p)) {
            Some(IndexParams::KeywordIndexParams(k)) => {
                assert_eq!(k.is_tenant, Some(true));
                assert_eq!(k.on_disk, Some(true));
            }
            other => panic!("expected keyword params, got {:?}", other),
        }
        // bool is stored as the string "true"/"false", hence a keyword index.
        assert!(matches!(
            build_payload_index_params("bool", Some(&p)),
            Some(IndexParams::KeywordIndexParams(_))
        ));
        match build_payload_index_params("uuid", Some(&p)) {
            Some(IndexParams::UuidIndexParams(u)) => assert_eq!(u.is_tenant, Some(true)),
            other => panic!("expected uuid params, got {:?}", other),
        }
        // Nothing to refine → plain default index.
        assert!(build_payload_index_params("keyword", None).is_none());
        assert!(build_payload_index_params("keyword", Some(&json!({}))).is_none());
    }

    /// `on_disk` exists on EVERY Qdrant index-params message, not just
    /// keyword/uuid — only `is_tenant` is keyword/uuid-only. Dropping `on_disk`
    /// for an int/float/geo/text/datetime field would be a config-says-on-disk /
    /// server-is-in-RAM run, the exact failure this branch exists to remove.
    #[test]
    fn payload_index_params_forward_on_disk_for_every_indexable_type() {
        let on_disk_only = json!({"on_disk": true});
        for ft in ["int", "float", "geo", "datetime"] {
            let params = build_payload_index_params(ft, Some(&on_disk_only))
                .unwrap_or_else(|| panic!("{ft} index must forward on_disk"));
            let forwarded = match params {
                IndexParams::IntegerIndexParams(p) => p.on_disk,
                IndexParams::FloatIndexParams(p) => p.on_disk,
                IndexParams::GeoIndexParams(p) => p.on_disk,
                IndexParams::DatetimeIndexParams(p) => p.on_disk,
                other => panic!("unexpected params for {ft}: {other:?}"),
            };
            assert_eq!(forwarded, Some(true), "{ft} dropped on_disk");
        }
        assert!(build_payload_index_params("nonsense", Some(&on_disk_only)).is_none());
        // `text` is excluded on purpose: TextIndexParams.tokenizer is a REQUIRED
        // proto field, so emitting the message to carry on_disk would pin the
        // tokenizer to Unknown — either rejected (leaving no text index at all)
        // or silently not the tokenizer the config asked for. The caller warns.
        assert!(
            build_payload_index_params("text", Some(&on_disk_only)).is_none(),
            "text must not be sent with a defaulted tokenizer"
        );
    }

    /// `"default"` must map to Qdrant's `Datatype::Default` (let the server
    /// choose), NOT to Float32 — conflating them misreports which storage the run
    /// actually measured.
    #[test]
    fn parse_datatype_covers_every_spelling() {
        use super::parse_datatype;
        for (s, want) in [
            ("default", super::Datatype::Default),
            ("float32", super::Datatype::Float32),
            ("f32", super::Datatype::Float32),
            ("float16", super::Datatype::Float16),
            ("f16", super::Datatype::Float16),
            ("uint8", super::Datatype::Uint8),
            ("u8", super::Datatype::Uint8),
            ("FLOAT16", super::Datatype::Float16),
        ] {
            assert_eq!(parse_datatype(s).unwrap(), want, "spelling {s:?}");
        }
        assert!(parse_datatype("bfloat16").is_err());
    }

    /// Every key inside `hnsw_config` that no engine reads must be reported, not
    /// dropped — the silent drop of `on_disk` there is the bug this branch fixes,
    /// so the next unrecognised key must not repeat it.
    #[test]
    fn hnsw_config_surfaces_unsupported_keys_and_forwards_the_new_ones() {
        let e = engine_with_collection_params(json!({
            "hnsw_config": { "m": 16, "full_scan_threshold": 20000,
                             "max_indexing_threads": 4, "on_disc": true }
        }));
        assert_eq!(
            e.hnsw.as_ref().expect("hnsw parsed").unsupported_keys(),
            vec!["on_disc"],
            "a typo'd hnsw_config key must be reported, not silently dropped"
        );

        let diff = e.hnsw_config_diff().expect("configured hnsw yields a diff");
        assert_eq!(diff.full_scan_threshold, Some(20000));
        assert_eq!(diff.max_indexing_threads, Some(4));
    }

    /// A `payload_index_params` key that cannot take effect WARNS and continues
    /// rather than failing the run.
    ///
    /// It is deliberately not fatal: upstream's own `qdrant-on-disk.json` names
    /// fields (`a`, `d`) that our dataset schemas do not all declare, so
    /// rejecting them would make the upstream file unrunnable verbatim — the
    /// opposite of this branch's goal. The warning is what keeps the run from
    /// quietly passing as tenant-optimised. (There is no live server in a unit
    /// test, so index creation itself warns too; what is asserted here is that
    /// neither case is escalated to an error.)
    #[test]
    fn payload_index_params_naming_unknown_field_warns_but_does_not_fail() {
        use super::payload_index_warnings;

        let schema = json!({"a": "keyword", "price": "int"});
        let schema = schema.as_object().unwrap();

        // A key absent from the schema gets no index at all, so the spec is inert.
        let params = json!({"tenant_id": {"is_tenant": true}});
        let warnings = payload_index_warnings(Some(schema), params.as_object(), "ds");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("tenant_id") && warnings[0].contains("NOT in the schema"));

        // `is_tenant` on a type that cannot carry it (int) — its on_disk still applies.
        let params = json!({"price": {"is_tenant": true, "on_disk": true}});
        let warnings = payload_index_warnings(Some(schema), params.as_object(), "ds");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("price") && warnings[0].contains("is_tenant is ignored"));

        // Usable specs are silent, and so is no config at all.
        let params = json!({"a": {"is_tenant": true, "on_disk": true}, "price": {"on_disk": true}});
        assert!(payload_index_warnings(Some(schema), params.as_object(), "ds").is_empty());
        assert!(payload_index_warnings(Some(schema), None, "ds").is_empty());
        // No schema means nothing can be satisfied.
        let params = json!({"a": {"on_disk": true}});
        assert_eq!(
            payload_index_warnings(None, params.as_object(), "ds").len(),
            1
        );
    }

    /// A sparse collection cannot honour hnsw_config or quantization_config, and
    /// the warning is the ONLY signal that a run named `*-bq` is not quantized —
    /// so assert the strings, not merely that configure() returned Ok.
    #[test]
    fn sparse_collection_warns_about_knobs_it_cannot_honour() {
        use super::sparse_ignored_warnings;

        assert!(sparse_ignored_warnings(false, false, "ds").is_empty());

        let both = sparse_ignored_warnings(true, true, "msmarco-sparse-1M");
        assert_eq!(both.len(), 2, "{both:?}");
        assert!(both[0].contains("hnsw_config") && both[0].contains("msmarco-sparse-1M"));
        assert!(
            both[1].contains("NOT quantized"),
            "the quantization warning must say the run is not quantized: {}",
            both[1]
        );
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
        let filter = parse_qdrant_conditions(&cond).unwrap();
        assert_eq!(filter.should.len(), 2, "or → 2 should entries");
        assert!(filter.must.is_empty(), "or-only leaves must empty");
    }

    #[test]
    fn and_plus_or_populates_both() {
        let cond = json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}},{"c":{"match":{"value":"z"}}}],
        });
        let filter = parse_qdrant_conditions(&cond).unwrap();
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

    // #121: a range that produces NO valid numeric bound (unrecognized key,
    // null/non-numeric bound) must be SKIPPED (None), not emitted as an empty,
    // unconstraining Range (a present-but-vacuous condition). Mirrors the
    // vectorsets null-bound fix (#115) and the other engines, which drop it.
    #[test]
    fn range_unknown_op_only_is_none() {
        assert!(build_qdrant_filter("n", "range", &json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_only_is_none() {
        assert!(
            build_qdrant_filter("n", "range", &json!({"gte": serde_json::Value::Null})).is_none()
        );
    }

    #[test]
    fn range_valid_bound_survives_null_sibling() {
        // A null/unknown bound is dropped, but a present valid bound still yields
        // a Range carrying only that bound.
        let r = numeric_range(json!({"gte": 5, "lt": serde_json::Value::Null, "foo": 9}));
        assert_eq!(r.gte, Some(5.0));
        assert!(r.lt.is_none() && r.gt.is_none() && r.lte.is_none());
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
    fn geo_missing_lat_or_lon_is_none() {
        assert!(build_qdrant_filter("loc", "geo", &json!({"lon":10.0,"radius":500})).is_none());
        assert!(build_qdrant_filter("loc", "geo", &json!({"lat":20.0,"radius":500})).is_none());
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
    fn exact_match_float_is_none() {
        // Qdrant `Match` supports keyword/integer/bool only — a float exact value
        // matches no arm and is dropped (float filtering uses range instead).
        assert!(build_qdrant_filter("n", "match", &json!({"value":1.5})).is_none());
    }

    #[test]
    fn exact_match_array_value_is_none() {
        assert!(build_qdrant_filter("n", "match", &json!({"value":[1,2]})).is_none());
    }

    // ── parse_qdrant_conditions edge cases + subfilter builder ──────────────
    use super::build_qdrant_subfilters;

    #[test]
    fn empty_conditions_object_is_none() {
        assert!(parse_qdrant_conditions(&json!({})).is_none());
    }

    #[test]
    fn and_only_populates_must_not_should() {
        let cond = json!({"and":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        let filter = parse_qdrant_conditions(&cond).unwrap();
        assert_eq!(filter.must.len(), 2, "and → 2 must entries");
        assert!(filter.should.is_empty(), "and-only leaves should empty");
    }

    #[test]
    fn subfilters_build_match_any_keyword_list() {
        // A keyword match_any list → one Condition carrying a keyword Match.
        let entries = vec![json!({"cat":{"match":{"any":["a","b"]}}})];
        let conds = build_qdrant_subfilters(&entries);
        assert_eq!(conds.len(), 1);
        let fc = field_condition(&conds[0]);
        assert_eq!(fc.key, "cat");
        assert!(
            fc.r#match.is_some(),
            "match_any keyword list should set a Match"
        );
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
