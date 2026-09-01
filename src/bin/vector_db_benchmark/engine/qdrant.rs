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
    IntegerIndexParams, KeywordIndexParams, MaxOptimizationThreads, MultiVectorComparator,
    MultiVectorConfigBuilder, NamedVectors, OptimizersConfigDiff, PointStruct,
    PrefetchQueryBuilder, ProductQuantization, QuantizationSearchParams, QuantizationType, Query,
    QueryPointsBuilder, ScalarQuantization, SearchParams as QdrantSearchParams,
    SparseIndexConfigBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder, Timestamp,
    UuidIndexParams, Vector, VectorInput, VectorParamsBuilder, VectorsConfig, VectorsConfigBuilder,
};
use qdrant_client::{Payload, Qdrant};
use vector_db_benchmark::start_gate::WorkerPool;

use crate::config::{EngineConfig, HnswConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{CorpusCount, Engine, SearchResults, UploadStats};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::MetadataItem;
use vector_db_benchmark::readers::MultiVector;

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
        let grpc_port: u16 = crate::effective_config::env_parsed("QDRANT_GRPC_PORT", 6334);

        let collection_name =
            crate::effective_config::env_or("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION);

        let api_key = crate::effective_config::env_var("QDRANT_API_KEY").ok();

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

        let grpc_url = if let Ok(url) = crate::effective_config::env_var("QDRANT_URL") {
            url
        } else {
            format!("http://{}:{}", clean_host, grpc_port)
        };

        // REST endpoint (default port 6333) for /metrics and /telemetry. Overridable
        // via QDRANT_REST_URL, or QDRANT_REST_PORT for just the port.
        let rest_url = if let Ok(url) = crate::effective_config::env_var("QDRANT_REST_URL") {
            url.trim_end_matches('/').to_string()
        } else {
            let rest_port: u16 = crate::effective_config::env_parsed("QDRANT_REST_PORT", 6333);
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
        if dataset.is_multivector() {
            return self.create_multivector_collection(dataset);
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

    /// Create a MULTIVECTOR collection with a single named vector ("colbert")
    /// configured for late-interaction (MaxSim) scoring. Unlike hybrid, there is
    /// no fusion: MaxSim scoring happens entirely server-side once
    /// `multivector_config` is set, so a plain nearest-neighbour query against
    /// this named vector is all a search needs (see `search`'s multivector
    /// branch). Reuses `dense_vector_params` for on_disk/datatype handling;
    /// HNSW is applied separately below via `hnsw_config_diff()`, same as the
    /// dense and hybrid paths.
    fn create_multivector_collection(&mut self, dataset: &Dataset) -> Result<(), String> {
        // `read_multivector_data`/`read_multivector_queries` have no `normalize`
        // parameter and `generate_multivector`'s brute-force ground truth scores
        // raw (unnormalized) dot products unconditionally. Silently proceeding
        // on a cosine (or omitted-distance, which defaults to cosine) dataset
        // would score normalized-in-Qdrant vectors against un-normalized ground
        // truth — a silent-wrong-result bug, not a missing feature. Refuse
        // loudly until per-token normalization is actually threaded through.
        if dataset.needs_normalization() {
            return Err(format!(
                "multivector dataset '{}' declares distance '{}', which requires \
                 per-token normalization that read_multivector_data/read_multivector_queries \
                 do not yet apply — refusing rather than silently scoring against \
                 un-normalized ground truth",
                dataset.config.name,
                dataset.distance()
            ));
        }

        let distance = dataset.distance();
        let vector_size = dataset.vector_size();
        let qdrant_distance = map_qdrant_distance(distance)?;

        let mut params = self
            .dense_vector_params(vector_size, qdrant_distance)?
            .multivector_config(MultiVectorConfigBuilder::new(MultiVectorComparator::MaxSim));
        if let Some(hnsw_config) = self.hnsw_config_diff() {
            params = params.hnsw_config(hnsw_config);
        }
        let mut vectors_cfg = VectorsConfigBuilder::default();
        vectors_cfg.add_named_vector_params("colbert", params);

        let mut create_builder =
            CreateCollectionBuilder::new(&self.collection_name).vectors_config(vectors_cfg);
        create_builder = self.apply_optimizers_and_quantization(create_builder)?;

        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create multivector collection: {}", e))?;

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
            index_coverage: None,
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
            index_coverage: None,
        })
    }

    /// Upload points carrying a single named multivector ("colbert"), batched.
    fn upload_multivector(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let dataset_path = dataset.get_path()?;
        println!(
            "Reading multivector dataset from {}...",
            dataset_path.display()
        );
        let read_start = Instant::now();
        let (ids, vectors) = dataset.read_multivector_data()?;
        let read_time = read_start.elapsed().as_secs_f64();
        println!(
            "Read {} multivector documents in {:.3}s",
            vectors.len(),
            read_time
        );

        println!(
            "Starting multivector upload, batch size {}...",
            self.batch_size
        );
        let upload_start = Instant::now();
        let pb = self.create_progress_bar(ids.len());
        for start in (0..ids.len()).step_by(self.batch_size) {
            let end = (start + self.batch_size).min(ids.len());
            let points: Vec<PointStruct> = (start..end)
                .map(|i| {
                    PointStruct::new(
                        ids[i] as u64,
                        NamedVectors::default()
                            .add_vector("colbert", Vector::new_multi(vectors[i].vectors.clone())),
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
                .map_err(|e| format!("Multivector upsert failed: {}", e))?;
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
            index_coverage: None,
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
        let green_wait_secs: u64 =
            crate::effective_config::env_parsed("QDRANT_GREEN_WAIT_SECS", 14400); // 4h
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
pub(crate) fn parse_qdrant_conditions(
    conditions: &serde_json::Value,
) -> Result<Option<Filter>, String> {
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

/// Whether a Qdrant client error means "no such collection" rather than "the
/// probe failed" (issue #238 — a probe failure must never be reported as a
/// corpus size of zero).
fn collection_missing(e: &qdrant_client::QdrantError) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("doesn't exist") || msg.contains("does not exist") || msg.contains("not found")
}

impl Engine for QdrantEngine {
    /// Server-side corpus size, for the `--skip-upload` reuse precondition
    /// (issue #238). `points_count` from the collection info — the same figure
    /// `GET /collections/<name>` reports. A missing collection answers 0.
    fn corpus_row_count(&mut self) -> Result<Option<CorpusCount>, String> {
        match self
            .rt
            .block_on(self.client.collection_info(&self.collection_name))
        {
            // Pass the Option THROUGH: a reply with no `points_count` means
            // "cannot tell", and `unwrap_or(0)` would turn that into "the corpus
            // is gone" — the exact fabrication `ft_info_num_docs` is careful to
            // avoid (see its `missing_field_is_none_not_zero` test). None here
            // lands in `NoServerCount`, whose note covers this case explicitly
            // ("or the engine replied without one") — the probe IS implemented
            // for Qdrant, so a note claiming otherwise would be wrong.
            Ok(info) => Ok(info
                .result
                .and_then(|r| r.points_count)
                .map(CorpusCount::exact)),
            // "collection doesn't exist" is a normal error reply, and for this
            // check it means the same thing as an empty collection. Anything else
            // (unreachable node, refused gRPC port) says nothing about the corpus
            // and must NOT be reported as a corpus of zero — that sends the user
            // to re-upload data that is still there.
            Err(e) if collection_missing(&e) => Ok(Some(CorpusCount::exact(0))),
            Err(e) => Err(format!(
                "collection_info('{}') failed: {}",
                self.collection_name, e
            )),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Qdrant is the only engine with a sparse / hybrid path.
    fn supports_sparse(&self) -> bool {
        true
    }

    /// Qdrant is the only engine with a multivector (ColBERT/MaxSim) path.
    fn supports_multivector(&self) -> bool {
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
        if dataset.is_multivector() {
            return self.upload_multivector(dataset);
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
            index_coverage: None,
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
        let is_multivector = dataset.is_multivector();
        // Per-prefetch candidate depth for hybrid fusion: reuse the configured
        // search_params.prefetch.limit if present, else a generous default (>= any
        // sensible top-k). Larger = better fusion recall, more work.
        let hybrid_prefetch_limit = prefetch_limit.unwrap_or(50);
        let (queries, sparse_queries, multivector_queries, neighbors, parsed_filters) = if is_hybrid
        {
            let (dq, sq, nb) = dataset.read_hybrid_queries()?;
            (
                dq,
                sq,
                Vec::<MultiVector>::new(),
                nb,
                Vec::<QueryFilter<Filter>>::new(),
            )
        } else if is_sparse {
            let (sq, nb) = dataset.read_sparse_queries()?;
            (
                Vec::<Vec<f32>>::new(),
                sq,
                Vec::<MultiVector>::new(),
                nb,
                Vec::<QueryFilter<Filter>>::new(),
            )
        } else if is_multivector {
            let (mq, nb) = dataset.read_multivector_queries()?;
            (
                Vec::<Vec<f32>>::new(),
                Vec::<vector_db_benchmark::readers::SparseVector>::new(),
                mq,
                nb,
                Vec::<QueryFilter<Filter>>::new(),
            )
        } else {
            let (q, nb, conditions) = dataset.read_queries()?;
            // "no conditions" and "conditions that produced no filter" are NOT
            // the same thing — `try_resolve_all` keeps them apart, so an
            // unrepresentable filter fails the run instead of quietly running
            // the query unfiltered against filtered ground truth (#219/#222).
            let pf: Vec<QueryFilter<Filter>> =
                conditions.try_resolve_all("Qdrant", parse_qdrant_conditions)?;
            (
                q,
                Vec::<vector_db_benchmark::readers::SparseVector>::new(),
                Vec::<MultiVector>::new(),
                nb,
                pf,
            )
        };

        let query_count = if is_sparse {
            sparse_queries.len()
        } else if is_multivector {
            multivector_queries.len()
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

        // Gate-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window. Every worker connects + primes, then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant and
        // releases everyone, so the measurement clock starts only once all workers
        // are warm and poised. The gate is count-agnostic: a worker that fails to
        // set up, panics, or is never started by the OS settles its ticket and turns
        // the run into a hard error instead of a hang (#214).

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

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "qdrant-search", parallel);
            for _ in 0..parallel {
                let grpc_url = grpc_url.clone();
                let api_key = api_key.clone();
                let collection_name = collection_name.clone();
                let queries = &queries;
                let sparse_queries = &sparse_queries;
                let multivector_queries = &multivector_queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                pool.spawn(move |ticket| {
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
                        } else if is_multivector {
                            // No prefetch/fusion needed: MaxSim scoring happens
                            // entirely server-side once the "colbert" named
                            // vector's multivector_config is set (see
                            // create_multivector_collection). Unlike the sparse
                            // arm, this collection IS HNSW-backed (same
                            // hnsw_config_diff() as the dense path), so hnsw_ef
                            // and quantization search-time overrides are
                            // meaningful here and must be forwarded too.
                            let mv = &multivector_queries[idx];
                            let mut qb = QueryPointsBuilder::new(collection_name.clone())
                                .query(Query::new_nearest(VectorInput::new_multi(
                                    mv.vectors.clone(),
                                )))
                                .using("colbert")
                                .limit(top as u64)
                                .with_payload(with_payload);
                            if hnsw_ef.is_some() || quantization_params.is_some() || indexed_only {
                                qb = qb.params(QdrantSearchParams {
                                    hnsw_ef,
                                    quantization: quantization_params,
                                    indexed_only: Some(indexed_only),
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
                            if let Some(filter) = parsed_filters[idx].as_ref() {
                                qb = qb.filter(filter.clone());
                            }
                            qb
                        };

                        if !is_sparse && !is_hybrid && !is_multivector && prefetch_enabled {
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
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("qdrant-search worker setup failed: {e}"));
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
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!("qdrant-search worker setup failed: {e}"));
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
                    // coordinator stamps the shared measurement start and releases all.
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
        // Measure from the post-gate start stamp (workers already primed), so
        // total_time excludes connection setup and the cold first query.
        let total_time = measured_start.elapsed().as_secs_f64();

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
    use super::build_qdrant_subfilters;

    /// The production path resolves conditions through
    /// `QueryConditions::try_resolve_all`, which is where the "declared but
    /// dropped" rule lives (`query_filter.rs`). These tests exercise that exact
    /// rule for one query, so they go through the same function.
    fn parse_query_filter(
        conditions: Option<&serde_json::Value>,
    ) -> Result<Option<Filter>, String> {
        vector_db_benchmark::query_filter::resolve("Qdrant", 0, conditions, parse_qdrant_conditions)
            .map(vector_db_benchmark::query_filter::QueryFilter::into_inner)
    }

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
