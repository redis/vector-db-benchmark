//! Dataset handling and loading.
//!
//! Provides Dataset struct that wraps config and provides data access.

use std::path::PathBuf;

use crate::config::{datasets_dir, DatasetConfig};
use crate::download;
use vector_db_benchmark::readers::metadata::MetadataItem;
use vector_db_benchmark::readers::{
    hdf5_train_row_count, npy_row_count, read_compound_data, read_compound_queries,
    read_gt_neighbours, read_hdf5_vectors, read_jsonl_queries, read_jsonl_vectors,
    read_npy_vectors, read_sparse_matrix, SparseVector,
};

/// Dataset wrapper that provides access to vectors and metadata
pub struct Dataset {
    pub config: DatasetConfig,
}

impl Dataset {
    pub fn new(config: DatasetConfig) -> Self {
        Self { config }
    }

    /// Get the file path for this dataset
    pub fn get_path(&self) -> Result<PathBuf, String> {
        if let Some(path_str) = self.config.path.as_str() {
            // Check in datasets/ directory
            let datasets_path = datasets_dir().join(path_str);
            if datasets_path.exists() {
                return Ok(datasets_path);
            }
            // Not found locally — try downloading if link is available
            if let Some(link) = &self.config.link {
                let target_path = datasets_dir().join(path_str);
                download::download_dataset(link, &target_path)?;
                // Re-check after download
                if target_path.exists() {
                    return Ok(target_path);
                }
                Err(format!(
                    "Downloaded from {} but path still not found: {}",
                    link, path_str
                ))
            } else {
                Err(format!(
                    "Dataset path not found and no download link: {} (tried {:?})",
                    path_str, datasets_path
                ))
            }
        } else if let Some(path_obj) = self.config.path.as_object() {
            // For dict-style paths (like h5-multi)
            if let Some(data_files) = path_obj.get("data").and_then(|d| d.as_array()) {
                if let Some(first) = data_files.first() {
                    if let Some(p) = first.get("path").and_then(|p| p.as_str()) {
                        let datasets_path = datasets_dir().join(p);
                        if datasets_path.exists() {
                            return Ok(datasets_path);
                        }
                    }
                }
            }
            Err("Could not resolve dict-style path".to_string())
        } else {
            Err("Dataset path is not a string or object".to_string())
        }
    }

    /// Get the distance metric for this dataset
    pub fn distance(&self) -> &str {
        self.config.distance.as_deref().unwrap_or("cosine")
    }

    /// Get vector dimensions
    pub fn vector_size(&self) -> i64 {
        self.config.vector_size.unwrap_or(128)
    }

    /// Check if normalization is needed (for angular/cosine distance)
    pub fn needs_normalization(&self) -> bool {
        matches!(
            self.distance().to_lowercase().as_str(),
            "angular" | "cosine"
        )
    }

    /// The dataset's corpus path on local disk, WITHOUT triggering a download.
    /// `None` when the corpus isn't present yet, or the path is dict-style
    /// (`h5-multi`), which has no single corpus file.
    fn local_path(&self) -> Option<PathBuf> {
        let p = datasets_dir().join(self.config.path.as_str()?);
        p.exists().then_some(p)
    }

    /// How many vectors the LOCAL corpus *actually* holds, measured from the
    /// files themselves — NPY/HDF5 shape headers (a ~128-byte read, never the
    /// multi-GB payload) or a JSONL line count.
    ///
    /// `Ok(None)` means "not measurable here": the corpus isn't downloaded yet,
    /// or the layout has no cheap row count (`sparse` CSR, `h5-multi` parts).
    /// Callers must treat `None` as "unknown", never as zero.
    pub fn measured_vector_count(&self) -> Result<Option<u64>, String> {
        let Some(path) = self.local_path() else {
            return Ok(None);
        };
        let dataset_type = self.config.dataset_type.as_deref().unwrap_or("");
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Resolve to the concrete file whose header/lines we measure.
        let effective = match dataset_type {
            // Compound and hybrid layouts both keep the dense corpus in
            // <dir>/vectors.npy (hybrid is a superset of the sparse layout).
            "tar" | "hybrid" => "npy",
            "hdf5" | "h5" => "hdf5",
            "jsonl" => "jsonl",
            // No declared type: fall back to the file extension, exactly as
            // read_vectors() does.
            "" => match ext.as_str() {
                "npy" => "npy",
                "hdf5" | "h5" => "hdf5",
                "jsonl" => "jsonl",
                _ => return Ok(None),
            },
            // "sparse" (CSR) and "h5-multi" (many part files) are not measured
            // here, so they fall back to the declared count via
            // `unmeasurable_corpus_is_present`.
            //
            // This matters now: `msmarco-sparse-100K` / `-1M` are `sparse` AND
            // declare a vector_count, so their gate target IS the declared
            // number. What keeps that honest is the path-size CI check in
            // config.rs — their leaf segments (`100K`, `1M`) advertise their
            // size, so a wrong count fails CI from datasets.json alone, with no
            // corpus on disk. The residual it does NOT cover is a correct count
            // paired with the wrong `data.csr` (the two sizes share one query
            // set), which would let the gate skip early.
            //
            // Closing that properly is cheap and worth doing: the CSR header's
            // FIRST i64 is n_row, so a 24-byte read yields an exact row count —
            // the same trick `npy_row_count` uses. Deliberately left out of the
            // merge that introduced this collision so it lands with its own
            // tests rather than riding in as a conflict resolution.
            _ => return Ok(None),
        };

        match effective {
            "npy" => {
                let npy = if path.is_dir() {
                    path.join("vectors.npy")
                } else {
                    path
                };
                if !npy.exists() {
                    return Ok(None);
                }
                let s = npy.to_str().ok_or("Invalid vectors.npy path encoding")?;
                npy_row_count(s).map(Some)
            }
            "hdf5" => {
                let s = path.to_str().ok_or("Invalid HDF5 path encoding")?;
                hdf5_train_row_count(s).map(Some)
            }
            _ => {
                let file = if path.is_dir() {
                    path.join("vectors.jsonl")
                } else {
                    path
                };
                if !file.exists() {
                    return Ok(None);
                }
                count_nonempty_lines(&file).map(Some)
            }
        }
    }

    /// Cross-check the declared `vector_count` in `datasets.json` against the
    /// corpus actually on disk (#224).
    ///
    /// `vector_count` is not decorative — Redis's shared-corpus mode (#188) uses
    /// it as the "is the corpus fully uploaded?" gate, so an UNDER-declared count
    /// lets a partially-populated keyspace pass as complete and the sweep then
    /// measures recall over a fraction of the corpus, silently, with no error.
    /// That direction is therefore a hard error: there is no benign reading of
    /// "the corpus has more vectors than we claim".
    ///
    /// The opposite direction (declared > actual) only makes every consumer more
    /// conservative — the completeness gate can never be satisfied early, so the
    /// worst case is a redundant re-upload. It is still a metadata bug (it is
    /// what `--list-datasets` prints), so it warns loudly but does not abort a
    /// run over a corpus that is merely smaller than advertised.
    pub fn validate_vector_count(&self) -> Result<(), String> {
        let declared = self
            .config
            .vector_count
            .filter(|&n| n > 0)
            .map(|n| n as u64);
        let measured = self.measured_vector_count()?;
        if let (Some(d), Some(m)) = (declared, measured) {
            self.compare_counts(d, m)?;
        }
        Ok(())
    }

    fn compare_counts(&self, declared: u64, measured: u64) -> Result<(), String> {
        if declared == measured {
            return Ok(());
        }
        if measured > declared {
            return Err(format!(
                "dataset '{}' declares vector_count {} in datasets.json but its corpus at {} \
                 actually holds {} vectors. A declared count SMALLER than the corpus makes the \
                 shared-corpus completeness gate (#188) accept a partially-uploaded keyspace as \
                 complete, which silently collapses recall (#224). Fix vector_count in \
                 datasets/datasets.json (or point the dataset at the corpus it describes).",
                self.config.name,
                declared,
                self.local_path()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "<unknown>".into()),
                measured,
            ));
        }
        eprintln!(
            "WARNING: dataset '{}' declares vector_count {} but its corpus holds only {} vectors. \
             Benchmarks will run over the {} vectors that exist; fix vector_count in \
             datasets/datasets.json (it is what --list-datasets reports).",
            self.config.name, declared, measured, measured,
        );
        Ok(())
    }

    /// The number of uploaded points that means "this corpus is fully present",
    /// for engine-side completeness gates such as Redis shared-corpus mode
    /// (#188 upload-once / build-many).
    ///
    /// Prefers the MEASURED corpus size over the declared `vector_count`, so a
    /// wrong number in `datasets.json` can no longer authorise an early skip
    /// (#224). It falls back to the declared count ONLY for the layouts that
    /// genuinely have no cheap row count (`sparse`, `h5-multi`) and whose corpus
    /// files are all present; everything else must be measured or is reported as
    /// `Ok(None)` — which callers must read as "cannot confirm completeness, do
    /// NOT skip".
    ///
    /// The fallback is keyed on `dataset_type`, never on "some directory
    /// exists". A measurable layout whose corpus file is missing — a `tar`
    /// dataset whose directory was created by an interrupted `extract_tgz`
    /// (`vectors.npy` is the LAST member of `arxiv.tar.gz`, so this is the
    /// normal shape of a truncated download), or one whose big `vectors.npy`
    /// was deleted to reclaim disk while `tests.jsonl` stayed behind so queries
    /// still work — must NOT be waved through on the declared number. That is
    /// #224 all over again, just reached from the corpus side.
    pub fn corpus_completeness_target(&self) -> Result<Option<u64>, String> {
        let declared = self
            .config
            .vector_count
            .filter(|&n| n > 0)
            .map(|n| n as u64);
        let measured = self.measured_vector_count()?;
        if let (Some(d), Some(m)) = (declared, measured) {
            self.compare_counts(d, m)?;
        }
        if measured.is_some() {
            return Ok(measured);
        }
        if self.unmeasurable_corpus_is_present() {
            Ok(declared)
        } else {
            Ok(None)
        }
    }

    /// Whether this is one of the layouts with no cheap row count AND all of its
    /// corpus files are on disk (no download attempted).
    ///
    /// Measurable layouts always answer `false`: they have a row count, so they
    /// must be measured rather than trusted. Only `sparse` (CSR) and `h5-multi`
    /// (many part files) may fall back to the declared count, and only when the
    /// files they need actually exist.
    fn unmeasurable_corpus_is_present(&self) -> bool {
        match self.config.dataset_type.as_deref().unwrap_or("") {
            "sparse" => self
                .local_path()
                .is_some_and(|p| p.join("data.csr").exists()),
            "h5-multi" => self
                .config
                .path
                .as_object()
                .and_then(|o| o.get("data"))
                .and_then(|d| d.as_array())
                .is_some_and(|parts| {
                    !parts.is_empty()
                        && parts.iter().all(|p| {
                            p.get("path")
                                .and_then(|p| p.as_str())
                                .is_some_and(|s| datasets_dir().join(s).exists())
                        })
                }),
            _ => false,
        }
    }

    /// Read all vectors and metadata from the dataset
    #[allow(clippy::type_complexity)]
    pub fn read_vectors(
        &self,
        normalize: bool,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>, Vec<Option<MetadataItem>>), String> {
        let path = self.get_path()?;
        // Cross-check declared vs. actual corpus size on every read — this is the
        // first point at which the corpus is guaranteed present (get_path may
        // have just downloaded it), so it is where a bad vector_count surfaces
        // as an error instead of as silently-wrong recall (#224).
        self.validate_vector_count()?;
        let path_str = path.to_str().ok_or("Invalid path encoding")?;
        let dataset_type = self.config.dataset_type.as_deref().unwrap_or("");

        match dataset_type {
            "tar" => {
                // Compound format (vectors.npy + payloads.jsonl)
                read_compound_data(path_str, normalize)
            }
            "hdf5" | "h5" => {
                // Explicit HDF5 type — trust it regardless of file extension
                let (ids, vectors) = read_hdf5_vectors(path_str, normalize)?;
                let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                Ok((ids, vectors, metadata))
            }
            "" => {
                // No type specified — infer from file extension
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                match ext.to_lowercase().as_str() {
                    "hdf5" | "h5" => {
                        let (ids, vectors) = read_hdf5_vectors(path_str, normalize)?;
                        let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                        Ok((ids, vectors, metadata))
                    }
                    "jsonl" => {
                        let (ids, vectors) = read_jsonl_vectors(path_str, normalize)?;
                        let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                        Ok((ids, vectors, metadata))
                    }
                    "npy" => {
                        let (ids, vectors) = read_npy_vectors(path_str, normalize)?;
                        let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                        Ok((ids, vectors, metadata))
                    }
                    _ => Err(format!("Unsupported file extension: {}", ext)),
                }
            }
            "jsonl" => {
                // JSONL path is a directory containing vectors.jsonl
                let vectors_file = if path.is_dir() {
                    path.join("vectors.jsonl")
                } else {
                    path.clone()
                };
                let vectors_str = vectors_file.to_str().ok_or("Invalid vectors.jsonl path")?;
                let (ids, vectors) = read_jsonl_vectors(vectors_str, normalize)?;
                let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                Ok((ids, vectors, metadata))
            }
            other => Err(format!("Unsupported dataset type: {}", other)),
        }
    }

    /// Read query vectors, ground truth neighbors, and filter conditions from the dataset.
    /// Returns (queries, neighbors, conditions) where conditions is per-query filter JSON.
    #[allow(clippy::type_complexity)]
    pub fn read_queries(
        &self,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<i64>>, Vec<Option<serde_json::Value>>), String> {
        let path = self.get_path()?;
        let path_str = path.to_str().ok_or("Invalid path encoding")?;
        let dataset_type = self.config.dataset_type.as_deref().unwrap_or("");
        let normalize = self.needs_normalization();

        match dataset_type {
            "tar" => {
                // Compound format: tests.jsonl includes conditions
                read_compound_queries(path_str, normalize)
            }
            "hdf5" | "h5" => {
                // Explicit HDF5 type — trust it regardless of file extension
                let (queries, neighbors) = self.read_hdf5_queries(path_str)?;
                let conditions = vec![None; queries.len()];
                Ok((queries, neighbors, conditions))
            }
            "jsonl" => {
                let dir = if path.is_dir() {
                    path.clone()
                } else {
                    path.parent()
                        .ok_or_else(|| "Cannot get parent dir of JSONL path".to_string())?
                        .to_path_buf()
                };
                let (queries, neighbors) = read_jsonl_queries(
                    dir.to_str().ok_or("Invalid dir path encoding")?,
                    normalize,
                )?;
                let conditions = vec![None; queries.len()];
                Ok((queries, neighbors, conditions))
            }
            _ => {
                if path_str.ends_with(".hdf5") || path_str.ends_with(".h5") {
                    let (queries, neighbors) = self.read_hdf5_queries(path_str)?;
                    let conditions = vec![None; queries.len()];
                    Ok((queries, neighbors, conditions))
                } else if path.is_dir() {
                    let tests_path = path.join("tests.jsonl");
                    let queries_path = path.join("queries.jsonl");
                    if tests_path.exists() {
                        read_compound_queries(path_str, normalize)
                    } else if queries_path.exists() {
                        let (queries, neighbors) = read_jsonl_queries(path_str, normalize)?;
                        let conditions = vec![None; queries.len()];
                        Ok((queries, neighbors, conditions))
                    } else {
                        Err(format!("No query files found in directory: {}", path_str))
                    }
                } else {
                    Err(format!(
                        "Query reading not supported for dataset type '{}' at path: {}",
                        dataset_type, path_str
                    ))
                }
            }
        }
    }

    /// Whether this is a sparse-vector dataset (`dataset_type: "sparse"`).
    pub fn is_sparse(&self) -> bool {
        self.config.dataset_type.as_deref() == Some("sparse")
    }

    /// Read sparse data vectors from `<dir>/data.csr`. Ids are the row indices.
    pub fn read_sparse_data(&self) -> Result<(Vec<i64>, Vec<SparseVector>), String> {
        let dir = self.get_path()?;
        let data = dir.join("data.csr");
        let vectors = read_sparse_matrix(data.to_str().ok_or("Invalid data.csr path")?)?;
        let ids: Vec<i64> = (0..vectors.len() as i64).collect();
        Ok((ids, vectors))
    }

    /// Read sparse queries from `<dir>/queries.csr` plus ground-truth neighbours.
    ///
    /// Two ground-truth layouts are accepted, because the public sparse datasets
    /// and our generated fixtures differ:
    ///
    /// * `neighbours.jsonl` — one JSON array of ids per line (our generator, and
    ///   the layout the dense/compound readers already use).
    /// * `results.gt` — the binary `n × d` ids+scores block shipped by the
    ///   `msmarco-sparse-*` datasets.
    ///
    /// `neighbours.jsonl` wins when both exist, so a locally regenerated fixture
    /// overrides a downloaded one. Neither present is an error naming both, since
    /// searching without ground truth would report a meaningless recall.
    ///
    /// The jsonl branch goes through `read_neighbours_strict` (blank lines are an
    /// error, not skipped: skipping one shifts every later row up and scores each
    /// query against its neighbour's truth), and the row count MUST equal the
    /// query count — the same two guards the hybrid path already applies.
    pub fn read_sparse_queries(&self) -> Result<(Vec<SparseVector>, Vec<Vec<i64>>), String> {
        let dir = self.get_path()?;
        let queries = read_sparse_matrix(
            dir.join("queries.csr")
                .to_str()
                .ok_or("Invalid queries.csr path")?,
        )?;

        let jsonl_path = dir.join("neighbours.jsonl");
        let gt_path = dir.join("results.gt");
        let neighbours: Vec<Vec<i64>> = if jsonl_path.exists() {
            read_neighbours_strict(&jsonl_path)?
        } else if gt_path.exists() {
            read_gt_neighbours(gt_path.to_str().ok_or("Invalid results.gt path")?)?
        } else {
            return Err(format!(
                "no ground truth for sparse dataset {}: expected {} or {}",
                self.config.name,
                jsonl_path.display(),
                gt_path.display()
            ));
        };

        // Ground truth must be row-aligned with the queries. Without this, a
        // short file makes the search loop index past the end of `neighbours`
        // (a panic in every worker), and a `results.gt` header declaring a
        // transposed shape — e.g. (n=2, d=4) for 4 queries of 2 neighbours,
        // which has the identical byte length and so passes that reader's own
        // length check — would score every query against the wrong truth.
        if neighbours.len() != queries.len() {
            return Err(format!(
                "sparse ground-truth row mismatch in {}: {} queries vs {} neighbour rows",
                dir.display(),
                queries.len(),
                neighbours.len()
            ));
        }

        // Guard against pairing one corpus size's ground truth with another's.
        // msmarco-sparse-100K and msmarco-sparse-1M ship the IDENTICAL 6980-query
        // set, so a 100K `results.gt` sitting next to the 1M corpus passes the
        // row-count check above and the reader's own length check — recall then
        // silently collapses instead of failing. Ids outside the declared corpus
        // are the only available signal.
        if let Some(vector_count) = self.config.vector_count {
            if let Some(&max_id) = neighbours.iter().flatten().max() {
                if max_id >= vector_count {
                    return Err(format!(
                        "sparse ground truth for {} references point id {} but the dataset \
                         declares only {} vectors — this ground truth does not belong to \
                         this corpus (the msmarco-sparse sizes share one query set, so the \
                         row counts match even when the corpora do not)",
                        self.config.name, max_id, vector_count
                    ));
                }
            }
        }

        Ok((queries, neighbours))
    }

    /// Whether this is a hybrid (dense + sparse) dataset (`dataset_type: "hybrid"`).
    ///
    /// A hybrid dataset directory carries BOTH dense npy files (`vectors.npy` /
    /// `queries.npy`) and sparse CSR files (`data.csr` / `queries.csr`), sharing
    /// a single `neighbours.jsonl` ground truth. This lets an engine fuse a dense
    /// prefetch and a sparse prefetch server-side (e.g. Qdrant RRF). It is a
    /// SUPERSET of the sparse layout: same CSR files, plus the dense npy files.
    pub fn is_hybrid(&self) -> bool {
        self.config.dataset_type.as_deref() == Some("hybrid")
    }

    /// Read hybrid upload data: dense vectors from `<dir>/vectors.npy` (reusing
    /// the npy reader) and the row-aligned sparse vectors from `<dir>/data.csr`
    /// (reusing the sparse CSR reader). Ids are the row indices. The dense and
    /// sparse matrices MUST have the same row count — one dense AND one sparse
    /// vector per point.
    #[allow(clippy::type_complexity)]
    pub fn read_hybrid_data(
        &self,
        normalize: bool,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>, Vec<SparseVector>), String> {
        let dir = self.get_path()?;
        // Same declared-vs-actual cross-check as read_vectors (#224).
        self.validate_vector_count()?;
        let (_ids, dense_vectors) = read_npy_vectors(
            dir.join("vectors.npy")
                .to_str()
                .ok_or("Invalid vectors.npy path")?,
            normalize,
        )?;
        let sparse = read_sparse_matrix(
            dir.join("data.csr")
                .to_str()
                .ok_or("Invalid data.csr path")?,
        )?;
        if dense_vectors.len() != sparse.len() {
            return Err(format!(
                "hybrid data row mismatch: {} dense rows vs {} sparse rows",
                dense_vectors.len(),
                sparse.len()
            ));
        }
        let ids: Vec<i64> = (0..dense_vectors.len() as i64).collect();
        Ok((ids, dense_vectors, sparse))
    }

    /// Read hybrid queries: dense queries from `<dir>/queries.npy`, the
    /// row-aligned sparse queries from `<dir>/queries.csr`, and shared
    /// ground-truth neighbours from `<dir>/neighbours.jsonl` (one JSON array of
    /// ids per line). The ground truth is shared because it describes the FUSED
    /// result, not either modality alone.
    #[allow(clippy::type_complexity)]
    pub fn read_hybrid_queries(
        &self,
    ) -> Result<(Vec<Vec<f32>>, Vec<SparseVector>, Vec<Vec<i64>>), String> {
        let dir = self.get_path()?;
        let normalize = self.needs_normalization();
        let (_ids, dense_queries) = read_npy_vectors(
            dir.join("queries.npy")
                .to_str()
                .ok_or("Invalid queries.npy path")?,
            normalize,
        )?;
        let sparse_queries = read_sparse_matrix(
            dir.join("queries.csr")
                .to_str()
                .ok_or("Invalid queries.csr path")?,
        )?;
        if dense_queries.len() != sparse_queries.len() {
            return Err(format!(
                "hybrid query row mismatch: {} dense vs {} sparse",
                dense_queries.len(),
                sparse_queries.len()
            ));
        }

        let gt_path = dir.join("neighbours.jsonl");
        let neighbours = read_neighbours_strict(&gt_path)?;

        // Ground truth must be row-aligned with the queries: a short (or long)
        // neighbours.jsonl would otherwise index out of bounds mid-run, or
        // silently score the wrong rows.
        if neighbours.len() != dense_queries.len() {
            return Err(format!(
                "hybrid ground-truth row mismatch: {} queries vs {} neighbour rows",
                dense_queries.len(),
                neighbours.len()
            ));
        }
        Ok((dense_queries, sparse_queries, neighbours))
    }

    /// Read queries from HDF5 file (test + neighbors datasets).
    #[allow(clippy::type_complexity)]
    fn read_hdf5_queries(&self, path_str: &str) -> Result<(Vec<Vec<f32>>, Vec<Vec<i64>>), String> {
        let file =
            hdf5::File::open(path_str).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;

        // Read test vectors
        let test_ds = file
            .dataset("test")
            .map_err(|e| format!("Failed to read 'test' dataset: {}", e))?;
        let shape = test_ds.shape();
        if shape.len() != 2 {
            return Err("Expected 2D test dataset".to_string());
        }
        let num_queries = shape[0];
        let dims = shape[1];
        let flat_data: Vec<f32> = test_ds
            .read_raw()
            .map_err(|e| format!("Failed to read test data: {}", e))?;
        let queries: Vec<Vec<f32>> = flat_data
            .chunks(dims)
            .take(num_queries)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Read neighbors (ground truth)
        let neighbors_ds = file
            .dataset("neighbors")
            .map_err(|e| format!("Failed to read 'neighbors' dataset: {}", e))?;
        let shape = neighbors_ds.shape();
        if shape.len() != 2 {
            return Err("Expected 2D neighbors dataset".to_string());
        }
        let num_neighbors = shape[0];
        let k = shape[1];
        let flat_neighbors: Vec<i64> = neighbors_ds
            .read_raw()
            .map_err(|e| format!("Failed to read neighbors data: {}", e))?;
        let neighbors: Vec<Vec<i64>> = flat_neighbors
            .chunks(k)
            .take(num_neighbors)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok((queries, neighbors))
    }
}

/// Count the non-blank lines of a JSONL corpus — one line per vector, matching
/// what `read_jsonl_vectors` yields (it skips blank lines). Streamed, so a large
/// `vectors.jsonl` is never held in memory.
fn count_nonempty_lines(path: &std::path::Path) -> Result<u64, String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut count = 0u64;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(|e| format!("read {}: {}", path.display(), e))?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

/// Parse a `neighbours.jsonl` ground-truth file (one JSON id-array per line)
/// with STRICT line alignment: every line must be a valid id-array so that row
/// `i` is unambiguously query `i`'s ground truth. A blank OR unparseable line in
/// the interior is rejected (a blanket "skip empty lines" would shift every
/// subsequent row up by one and silently corrupt recall). Exactly one trailing
/// newline is tolerated.
fn read_neighbours_strict(path: &std::path::Path) -> Result<Vec<Vec<i64>>, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut lines: Vec<&str> = raw.split('\n').collect();
    // Tolerate a single trailing newline (final split element is "").
    if lines.last() == Some(&"") {
        lines.pop();
    }
    let mut out = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(format!(
                "{}: blank line at row {} (ground-truth rows must be contiguous)",
                path.display(),
                i + 1
            ));
        }
        let row: Vec<i64> = serde_json::from_str(trimmed)
            .map_err(|e| format!("{}: parse row {}: {}", path.display(), i + 1, e))?;
        out.push(row);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatasetConfig;
    use vector_db_benchmark::readers::{
        write_gt_neighbours, write_npy_vectors, write_sparse_matrix,
    };

    /// A dataset of the given `dataset_type` rooted at an absolute temp path.
    fn dataset_at(
        path: &std::path::Path,
        dataset_type: &str,
        vector_count: Option<i64>,
    ) -> Dataset {
        Dataset::new(DatasetConfig {
            name: "count-unit".to_string(),
            dataset_type: Some(dataset_type.to_string()),
            path: serde_json::Value::String(path.to_str().unwrap().to_string()),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count,
            link: None,
            schema: None,
            description: None,
        })
    }

    /// The corpus, not `datasets.json`, is the authority on how many vectors
    /// exist (#224) — for both the compound layout and plain JSONL.
    #[test]
    fn measures_the_corpus_from_the_files_themselves() {
        let dir = tempfile::tempdir().unwrap();
        let vectors: Vec<Vec<f32>> = (0..42).map(|i| vec![i as f32, 0.0, 0.0]).collect();
        write_npy_vectors(dir.path().join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
        let ds = dataset_at(dir.path(), "tar", Some(42));
        assert_eq!(ds.measured_vector_count().unwrap(), Some(42));
        ds.validate_vector_count().expect("declared matches corpus");

        let jsonl_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            jsonl_dir.path().join("vectors.jsonl"),
            "[1.0,0.0,0.0]\n\n[0.0,1.0,0.0]\n[0.0,0.0,1.0]\n",
        )
        .unwrap();
        let ds = dataset_at(jsonl_dir.path(), "jsonl", Some(3));
        assert_eq!(
            ds.measured_vector_count().unwrap(),
            Some(3),
            "blank lines are not vectors (read_jsonl_vectors skips them)"
        );
    }

    /// An UNDER-declared count is the #224 failure and must abort the read; an
    /// OVER-declared one only makes consumers more conservative, so it warns and
    /// lets the run proceed over the vectors that actually exist.
    #[test]
    fn count_mismatch_is_fatal_only_when_the_corpus_is_bigger_than_declared() {
        let dir = tempfile::tempdir().unwrap();
        let vectors: Vec<Vec<f32>> = (0..500).map(|i| vec![i as f32, 0.0, 0.0]).collect();
        write_npy_vectors(dir.path().join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

        let under = dataset_at(dir.path(), "tar", Some(100));
        let err = under
            .validate_vector_count()
            .expect_err("corpus larger than declared must be fatal");
        assert!(err.contains("500"), "{err}");
        assert!(
            under.read_vectors(false).is_err(),
            "read_vectors must surface the mismatch rather than benchmark a lie"
        );

        let over = dataset_at(dir.path(), "tar", Some(100_000));
        over.validate_vector_count()
            .expect("corpus smaller than declared only warns");
        assert_eq!(over.read_vectors(false).unwrap().1.len(), 500);

        // A dataset with no declared count is unconstrained.
        let silent = dataset_at(dir.path(), "tar", None);
        silent.validate_vector_count().unwrap();
        assert_eq!(silent.corpus_completeness_target().unwrap(), Some(500));
    }

    /// Build a `Dataset` whose `path` is an absolute temp dir (so `get_path`
    /// resolves to it directly — `datasets_dir().join(abs)` == `abs`).
    fn hybrid_dataset(dir: &std::path::Path) -> Dataset {
        Dataset::new(DatasetConfig {
            name: "hybrid-unit".to_string(),
            dataset_type: Some("hybrid".to_string()),
            path: serde_json::Value::String(dir.to_str().unwrap().to_string()),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count: Some(2),
            link: None,
            schema: None,
            description: None,
        })
    }

    #[test]
    fn is_hybrid_only_for_hybrid_type() {
        let dir = tempfile::tempdir().unwrap();
        let ds = hybrid_dataset(dir.path());
        assert!(ds.is_hybrid());
        assert!(!ds.is_sparse());

        let mut cfg = ds.config.clone();
        cfg.dataset_type = Some("sparse".to_string());
        let sparse = Dataset::new(cfg);
        assert!(!sparse.is_hybrid());
        assert!(sparse.is_sparse());
    }

    #[test]
    fn reads_hybrid_data_and_queries() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        let dense = vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 2.0, 0.0]];
        let sparse = vec![
            SparseVector {
                indices: vec![0, 2],
                values: vec![1.0, 3.0],
            },
            SparseVector {
                indices: vec![1],
                values: vec![5.0],
            },
        ];
        write_npy_vectors(p.join("vectors.npy").to_str().unwrap(), &dense).unwrap();
        write_sparse_matrix(p.join("data.csr").to_str().unwrap(), &sparse).unwrap();

        let dense_q = vec![vec![1.0f32, 1.0, 1.0]];
        let sparse_q = vec![SparseVector {
            indices: vec![0],
            values: vec![2.0],
        }];
        write_npy_vectors(p.join("queries.npy").to_str().unwrap(), &dense_q).unwrap();
        write_sparse_matrix(p.join("queries.csr").to_str().unwrap(), &sparse_q).unwrap();
        std::fs::write(p.join("neighbours.jsonl"), "[0, 1]\n").unwrap();

        let ds = hybrid_dataset(p);
        let (ids, d, s) = ds.read_hybrid_data(false).unwrap();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(d, dense);
        assert_eq!(s, sparse);

        let (dq, sq, nb) = ds.read_hybrid_queries().unwrap();
        assert_eq!(dq, dense_q);
        assert_eq!(sq, sparse_q);
        assert_eq!(nb, vec![vec![0i64, 1]]);
    }

    /// Build a sparse `Dataset` rooted at an absolute temp dir.
    fn sparse_dataset(dir: &std::path::Path) -> Dataset {
        let mut cfg = hybrid_dataset(dir).config;
        cfg.name = "sparse-unit".to_string();
        cfg.dataset_type = Some("sparse".to_string());
        // Large enough that the ground-truth ids used by the other tests are in
        // range; `sparse_ground_truth_from_the_wrong_corpus_errors` exercises the
        // id-range guard deliberately.
        cfg.vector_count = Some(100);
        Dataset::new(cfg)
    }

    /// msmarco-sparse-100K and msmarco-sparse-1M ship the IDENTICAL 6980 queries,
    /// so pairing the 100K ground truth with the 1M corpus (or the reverse)
    /// passes every row-count and file-length check and merely collapses recall.
    /// Ids outside the declared corpus are the only signal, so they must Err.
    #[test]
    fn sparse_ground_truth_from_the_wrong_corpus_errors() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sparse_matrix(
            p.join("queries.csr").to_str().unwrap(),
            &[SparseVector {
                indices: vec![0],
                values: vec![1.0],
            }],
        )
        .unwrap();
        // vector_count is 100 → id 100 is one past the end of the corpus.
        write_gt_neighbours(p.join("results.gt").to_str().unwrap(), &[vec![100i64]]).unwrap();

        let err = sparse_dataset(p).read_sparse_queries().unwrap_err();
        assert!(
            err.contains("does not belong to"),
            "unexpected error: {}",
            err
        );
    }

    /// The public `msmarco-sparse-*` datasets ship binary `results.gt` rather
    /// than `neighbours.jsonl`; both layouts must read.
    #[test]
    fn reads_sparse_ground_truth_from_results_gt() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        let queries = vec![SparseVector {
            indices: vec![0, 4],
            values: vec![1.0, 2.0],
        }];
        write_sparse_matrix(p.join("queries.csr").to_str().unwrap(), &queries).unwrap();
        write_gt_neighbours(p.join("results.gt").to_str().unwrap(), &[vec![9i64, 4, 1]]).unwrap();

        let (q, nb) = sparse_dataset(p).read_sparse_queries().unwrap();
        assert_eq!(q, queries);
        assert_eq!(nb, vec![vec![9i64, 4, 1]]);
    }

    /// A regenerated local fixture must win over a downloaded binary one.
    #[test]
    fn sparse_neighbours_jsonl_takes_precedence_over_results_gt() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sparse_matrix(
            p.join("queries.csr").to_str().unwrap(),
            &[SparseVector {
                indices: vec![0],
                values: vec![1.0],
            }],
        )
        .unwrap();
        write_gt_neighbours(p.join("results.gt").to_str().unwrap(), &[vec![7i64]]).unwrap();
        std::fs::write(p.join("neighbours.jsonl"), "[42]\n").unwrap();

        let (_, nb) = sparse_dataset(p).read_sparse_queries().unwrap();
        assert_eq!(nb, vec![vec![42i64]]);
    }

    /// Ground truth with fewer rows than there are queries must be REJECTED, not
    /// returned short: the search loop indexes `neighbors[idx]` per query, so a
    /// short file panicked every worker thread. Mirrors the hybrid path's guard.
    #[test]
    fn sparse_rejects_ground_truth_row_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sparse_matrix(
            p.join("queries.csr").to_str().unwrap(),
            &[
                SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
                SparseVector {
                    indices: vec![1],
                    values: vec![1.0],
                },
                SparseVector {
                    indices: vec![2],
                    values: vec![1.0],
                },
            ],
        )
        .unwrap();
        // 3 queries but only 2 ground-truth rows.
        write_gt_neighbours(
            p.join("results.gt").to_str().unwrap(),
            &[vec![1i64], vec![2i64]],
        )
        .unwrap();

        let err = sparse_dataset(p).read_sparse_queries().unwrap_err();
        assert!(
            err.contains("row mismatch") && err.contains("3 queries"),
            "got: {err}"
        );
    }

    /// A blank line must be an error, never skipped: skipping shifts every later
    /// row up one, scoring each query against its neighbour's truth — a silently
    /// wrong recall. The hybrid path already rejects this via
    /// `read_neighbours_strict`; the sparse path now shares it.
    #[test]
    fn sparse_rejects_blank_line_in_neighbours_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sparse_matrix(
            p.join("queries.csr").to_str().unwrap(),
            &[
                SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
                SparseVector {
                    indices: vec![1],
                    values: vec![1.0],
                },
            ],
        )
        .unwrap();
        std::fs::write(p.join("neighbours.jsonl"), "[1]\n\n[2]\n").unwrap();

        let err = sparse_dataset(p).read_sparse_queries().unwrap_err();
        assert!(err.contains("blank line"), "got: {err}");
    }

    /// No ground truth at all must fail loudly and name both candidates — a run
    /// without ground truth would report a meaningless recall.
    #[test]
    fn sparse_without_any_ground_truth_errors() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sparse_matrix(
            p.join("queries.csr").to_str().unwrap(),
            &[SparseVector {
                indices: vec![0],
                values: vec![1.0],
            }],
        )
        .unwrap();

        let err = sparse_dataset(p).read_sparse_queries().unwrap_err();
        assert!(err.contains("neighbours.jsonl"), "got: {}", err);
        assert!(err.contains("results.gt"), "got: {}", err);
    }

    /// Helper: write the four hybrid data files (2 docs / 1 query) into `p`,
    /// leaving `neighbours.jsonl` for the caller to control.
    fn write_hybrid_files_without_neighbours(p: &std::path::Path) {
        write_npy_vectors(
            p.join("vectors.npy").to_str().unwrap(),
            &[vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        )
        .unwrap();
        write_sparse_matrix(
            p.join("data.csr").to_str().unwrap(),
            &[
                SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
                SparseVector {
                    indices: vec![1],
                    values: vec![1.0],
                },
            ],
        )
        .unwrap();
        write_npy_vectors(
            p.join("queries.npy").to_str().unwrap(),
            &[vec![1.0f32, 1.0, 1.0]],
        )
        .unwrap();
        write_sparse_matrix(
            p.join("queries.csr").to_str().unwrap(),
            &[SparseVector {
                indices: vec![0],
                values: vec![1.0],
            }],
        )
        .unwrap();
    }

    #[test]
    fn read_hybrid_queries_rejects_neighbour_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_hybrid_files_without_neighbours(p);
        // 1 query but 2 ground-truth rows → must error (finding 3).
        std::fs::write(p.join("neighbours.jsonl"), "[0]\n[1]\n").unwrap();
        let ds = hybrid_dataset(p);
        let err = ds.read_hybrid_queries().unwrap_err();
        assert!(err.contains("ground-truth row mismatch"), "got: {err}");
    }

    #[test]
    fn read_hybrid_queries_rejects_interior_blank_line() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_hybrid_files_without_neighbours(p);
        // Interior blank line would silently shift rows → must error (finding 5).
        std::fs::write(p.join("neighbours.jsonl"), "\n[0]\n").unwrap();
        let ds = hybrid_dataset(p);
        let err = ds.read_hybrid_queries().unwrap_err();
        assert!(err.contains("blank line"), "got: {err}");
    }

    #[test]
    fn read_neighbours_strict_tolerates_single_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("neighbours.jsonl");
        std::fs::write(&p, "[1, 2]\n[3]\n").unwrap();
        let nb = read_neighbours_strict(&p).unwrap();
        assert_eq!(nb, vec![vec![1i64, 2], vec![3]]);

        // No trailing newline is also fine.
        std::fs::write(&p, "[1, 2]\n[3]").unwrap();
        assert_eq!(
            read_neighbours_strict(&p).unwrap(),
            vec![vec![1i64, 2], vec![3]]
        );

        // A doubled trailing newline leaves an interior blank → rejected.
        std::fs::write(&p, "[1]\n\n").unwrap();
        assert!(read_neighbours_strict(&p).is_err());
    }

    /// Build a `Dataset` with just the accessor-relevant fields set. `path` is a
    /// dummy string (these tests never touch the filesystem).
    fn accessor_dataset(
        dataset_type: Option<&str>,
        distance: Option<&str>,
        vector_size: Option<i64>,
    ) -> Dataset {
        Dataset::new(DatasetConfig {
            name: "acc".to_string(),
            dataset_type: dataset_type.map(String::from),
            path: serde_json::Value::String("dummy".to_string()),
            distance: distance.map(String::from),
            vector_size,
            vector_count: None,
            link: None,
            schema: None,
            description: None,
        })
    }

    #[test]
    fn distance_defaults_to_cosine() {
        assert_eq!(accessor_dataset(None, None, None).distance(), "cosine");
        assert_eq!(accessor_dataset(None, Some("l2"), None).distance(), "l2");
    }

    #[test]
    fn vector_size_defaults_to_128() {
        assert_eq!(accessor_dataset(None, None, None).vector_size(), 128);
        assert_eq!(accessor_dataset(None, None, Some(768)).vector_size(), 768);
    }

    #[test]
    fn needs_normalization_true_for_cosine_and_angular_case_insensitive() {
        for d in [
            "cosine", "COSINE", "Cosine", "angular", "ANGULAR", "Angular",
        ] {
            assert!(
                accessor_dataset(None, Some(d), None).needs_normalization(),
                "distance={d}"
            );
        }
        // Default (None) resolves to "cosine" → needs normalization.
        assert!(accessor_dataset(None, None, None).needs_normalization());
    }

    #[test]
    fn needs_normalization_false_for_l2_and_dot() {
        for d in ["l2", "L2", "dot", "Dot", "euclidean", "ip"] {
            assert!(
                !accessor_dataset(None, Some(d), None).needs_normalization(),
                "distance={d}"
            );
        }
    }

    #[test]
    fn type_detection_is_sparse_and_is_hybrid() {
        assert!(accessor_dataset(Some("sparse"), None, None).is_sparse());
        assert!(!accessor_dataset(Some("sparse"), None, None).is_hybrid());

        assert!(accessor_dataset(Some("hybrid"), None, None).is_hybrid());
        assert!(!accessor_dataset(Some("hybrid"), None, None).is_sparse());

        // Neither for other / missing types.
        let tar = accessor_dataset(Some("tar"), None, None);
        assert!(!tar.is_sparse());
        assert!(!tar.is_hybrid());
        let none = accessor_dataset(None, None, None);
        assert!(!none.is_sparse());
        assert!(!none.is_hybrid());
    }

    #[test]
    fn read_hybrid_data_rejects_row_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        // 2 dense rows but only 1 sparse row → must error.
        write_npy_vectors(
            p.join("vectors.npy").to_str().unwrap(),
            &[vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        )
        .unwrap();
        write_sparse_matrix(
            p.join("data.csr").to_str().unwrap(),
            &[SparseVector {
                indices: vec![0],
                values: vec![1.0],
            }],
        )
        .unwrap();
        let ds = hybrid_dataset(p);
        assert!(ds.read_hybrid_data(false).is_err());
    }
}
