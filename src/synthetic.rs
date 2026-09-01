//! Deterministic synthetic dataset generation.
//!
//! Shared, fixed-seed generators for the small synthetic datasets used to
//! exercise the sparse / hybrid / filter code paths end-to-end. The SAME
//! generators back both:
//!   * the `generate-dataset` binary (writes runnable datasets under `datasets/`
//!     that are registered in `datasets/datasets.json`), and
//!   * the `tests/common` integration fixtures (temp-project scaffolding + engine
//!     configs around the identical data),
//!
//! so there is a single source of truth for the planted data and its ground
//! truth. Only the on-disk serialization primitives (`write_sparse_matrix`,
//! `write_npy_vectors`) live in [`crate::readers`]; this module adds the
//! remaining jsonl writers and the pure data-generation logic.

use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::readers::{MultiVector, SparseVector};

/// Write `neighbours.jsonl` (one JSON id-array per line) at `path`. This is the
/// ground-truth layout the sparse/hybrid readers expect (see
/// `dataset.rs::read_neighbours_strict`).
pub fn write_neighbours_jsonl(path: &Path, neighbours: &[Vec<i64>]) -> Result<(), String> {
    let body = neighbours
        .iter()
        .map(|nn| serde_json::to_string(nn).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    std::fs::write(path, body).map_err(|e| format!("write {}: {}", path.display(), e))
}

/// Write `vectors` as a `.jsonl` file (one JSON float-array per line), the
/// layout the `type:"jsonl"` reader expects.
pub fn write_jsonl_vectors(path: &Path, vectors: &[Vec<f32>]) -> Result<(), String> {
    let body = vectors
        .iter()
        .map(|v| {
            serde_json::to_string(&v.iter().map(|x| *x as f64).collect::<Vec<_>>())
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    std::fs::write(path, body).map_err(|e| format!("write {}: {}", path.display(), e))
}

/// A generated sparse dataset: `data` (corpus) + `queries`, with `neighbours`
/// the top-`top` brute-force dot-product (descending / MIPS) ground truth.
pub struct SparseData {
    pub data: Vec<SparseVector>,
    pub queries: Vec<SparseVector>,
    pub neighbours: Vec<Vec<i64>>,
}

/// Corpus size of the shipped `synthetic-sparse-300` dataset.
///
/// Single source of truth, shared by `generate-dataset` (which writes the
/// corpus) and the `datasets.json` guard in `config.rs` (which checks the
/// declared `vector_count` against it). CI has no corpus on disk, so without
/// this constant nothing could police that number from `datasets.json` alone —
/// and the dataset's NAME says 300, which is its DIMENSION, so the wrong value
/// is the easy mistake to make. It was in fact proposed during review.
pub const SYNTHETIC_SPARSE_ROWS: usize = 150;

/// Sparse dimensionality of `synthetic-sparse-300` — the `300` in its name.
pub const SYNTHETIC_SPARSE_DIM: usize = 300;

/// Generate a deterministic random sparse dataset and its dot-product
/// (descending) ground truth. Sparse similarity is MIPS — larger dot = more
/// similar — so the neighbours are sorted DESCENDING by dot product.
///
/// `seed` fixes the RNG; `dim` is the sparse dimensionality, `nnz` the non-zeros
/// per vector, `n` the corpus size, `q` the query count and `top` the number of
/// ground-truth neighbours per query.
pub fn generate_sparse(
    seed: u64,
    dim: usize,
    nnz: usize,
    n: usize,
    q: usize,
    top: usize,
) -> SparseData {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut make = |count: usize| -> Vec<SparseVector> {
        (0..count)
            .map(|_| {
                let mut idx: Vec<u32> = Vec::with_capacity(nnz);
                while idx.len() < nnz {
                    let c = rng.gen_range(0..dim as u32);
                    if !idx.contains(&c) {
                        idx.push(c);
                    }
                }
                idx.sort_unstable();
                let values: Vec<f32> = (0..nnz).map(|_| rng.gen_range(0.1..1.0)).collect();
                SparseVector {
                    indices: idx,
                    values,
                }
            })
            .collect()
    };
    let data = make(n);
    let queries = make(q);

    // Brute-force sparse dot product; sort DESCENDING (MIPS).
    let dot = |a: &SparseVector, b: &SparseVector| -> f64 {
        let mut s = 0.0f64;
        for (i, &ai) in a.indices.iter().enumerate() {
            if let Some(j) = b.indices.iter().position(|&bi| bi == ai) {
                s += a.values[i] as f64 * b.values[j] as f64;
            }
        }
        s
    };
    let neighbours: Vec<Vec<i64>> = queries
        .iter()
        .map(|qv| {
            let mut scored: Vec<(i64, f64)> = data
                .iter()
                .enumerate()
                .map(|(i, d)| (i as i64, dot(qv, d)))
                .collect();
            // DESCENDING by dot product (b vs a). Do NOT flip this.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.iter().take(top).map(|(id, _)| *id).collect()
        })
        .collect();

    SparseData {
        data,
        queries,
        neighbours,
    }
}

/// A generated hybrid (dense + sparse) dataset. The fused (RRF) top-`top` ground
/// truth is recoverable ONLY by combining both modalities — neither dense nor
/// sparse alone reaches it (see the detailed planting rationale below).
pub struct HybridData {
    pub dense: Vec<Vec<f32>>,
    pub dense_queries: Vec<Vec<f32>>,
    pub sparse: Vec<SparseVector>,
    pub sparse_queries: Vec<SparseVector>,
    pub neighbours: Vec<Vec<i64>>,
    /// Dense dimensionality.
    pub dim: usize,
    /// Ground-truth / top-k set size.
    pub top: usize,
}

/// Generate a deterministic PLANTED hybrid dataset whose fused (RRF) top-K
/// ground truth is recoverable ONLY by combining a dense prefetch and a sparse
/// prefetch. Per query, K ground-truth docs are split into two halves and two
/// rings of single-modality distractors so that:
///   * dense-only top-K  → recall(R) ≈ 0.5
///   * sparse-only top-K → recall(R) ≈ 0.5
///   * fused (RRF) top-K → recall(R) ≈ 1.0
///
/// This is the exact construction validated by the qdrant hybrid integration
/// test (and its dense-only negative control).
pub fn generate_hybrid(seed: u64) -> HybridData {
    const K: usize = 8; // top-k / ground-truth-set size (must be even)
    const HALF: usize = K / 2; // per-half / per-distractor-ring size
    const Q: usize = 6; // queries (and dense centre axes)
    const DENSE_DIM: usize = 16; // >= Q centre dims + HALF distractor bump dims
    const PER_Q: usize = 4 * HALF; // R_dense + R_sparse + D_d + D_s per query
    const FILLER: usize = 24;
    const N: usize = Q * PER_Q + FILLER; // = 120
    const BIG: f32 = 100.0; // centre magnitude → regions ~141 apart, origin ~100

    // Sparse layout: query q owns index block F_q = [q*HALF .. q*HALF+HALF);
    // dense-only distractors / filler use a disjoint "junk" block J (dot 0).
    const F_TOTAL: usize = Q * HALF;

    let mut rng = StdRng::seed_from_u64(seed);
    let tiny = |rng: &mut StdRng| -> f32 { rng.gen_range(-0.01f32..0.01) };

    // Doc-id block layout for query q: base = q*PER_Q, then four HALF-sized rings.
    let base = |q: usize| q * PER_Q;
    let r_dense_id = |q: usize, j: usize| base(q) + j; //            [base,       base+HALF)
    let r_sparse_id = |q: usize, j: usize| base(q) + HALF + j; //    [base+HALF,  base+2HALF)
    let d_d_id = |q: usize, j: usize| base(q) + 2 * HALF + j; //     [base+2HALF, base+3HALF)
    let d_s_id = |q: usize, j: usize| base(q) + 3 * HALF + j; //     [base+3HALF, base+4HALF)
    let filler_start = Q * PER_Q;

    // Dense centre for query q: BIG on axis q, else 0.
    let centre = |q: usize| -> Vec<f32> {
        let mut v = vec![0.0f32; DENSE_DIM];
        v[q] = BIG;
        v
    };
    // A dense doc = centre + `mag` along a distractor axis (dims Q..DENSE_DIM),
    // so its L2 distance from the query (= centre) is exactly `mag`.
    let offset_from = |c: &[f32], mag: f32, j: usize| -> Vec<f32> {
        let mut v = c.to_vec();
        v[Q + (j % (DENSE_DIM - Q))] += mag;
        v
    };

    let junk: Vec<u32> = (F_TOTAL..F_TOTAL + HALF).map(|i| i as u32).collect();

    let mut dense: Vec<Vec<f32>> = vec![vec![0.0f32; DENSE_DIM]; N];
    let mut sparse: Vec<SparseVector> = vec![
        SparseVector {
            indices: vec![],
            values: vec![]
        };
        N
    ];

    for q in 0..Q {
        let c = centre(q);
        let f_q: Vec<u32> = (q * HALF..q * HALF + HALF).map(|i| i as u32).collect();
        // Per-doc tiny increments break ties so tiers stay crisply ordered.
        let sp = |indices: &[u32], val: f32, j: usize| SparseVector {
            indices: indices.to_vec(),
            values: indices.iter().map(|_| val + 0.001 * j as f32).collect(),
        };
        for j in 0..HALF {
            // R_dense: dense dist 1.0 (ranks 0..HALF); sparse dot ~ HALF*1 (low).
            dense[r_dense_id(q, j)] = offset_from(&c, 1.0, j);
            sparse[r_dense_id(q, j)] = sp(&f_q, 1.0, j);

            // D_d: dense dist 2.0 (ranks HALF..K); sparse = junk → dot 0.
            dense[d_d_id(q, j)] = offset_from(&c, 2.0, HALF + j);
            sparse[d_d_id(q, j)] = sp(&junk, 1.0, j);

            // R_sparse: dense dist 3.0 (ranks K..3K/2); sparse dot ~ HALF*3 (top).
            dense[r_sparse_id(q, j)] = offset_from(&c, 3.0, 2 * HALF + j);
            sparse[r_sparse_id(q, j)] = sp(&f_q, 3.0, j);

            // D_s: dense ≈ origin (dist ~BIG, absent from dense top); sparse dot ~
            // HALF*2 (ranks HALF..K, between R_sparse and R_dense).
            let mut ds_v = vec![0.0f32; DENSE_DIM];
            for x in ds_v.iter_mut() {
                *x += tiny(&mut rng);
            }
            dense[d_s_id(q, j)] = ds_v;
            sparse[d_s_id(q, j)] = sp(&f_q, 2.0, j);
        }
    }
    // Filler: dense ≈ origin, sparse in junk block (dot 0 with every query).
    for id in filler_start..N {
        let mut fv = vec![0.0f32; DENSE_DIM];
        for x in fv.iter_mut() {
            *x += tiny(&mut rng);
        }
        dense[id] = fv;
        sparse[id] = SparseVector {
            indices: junk.clone(),
            values: vec![1.0f32; HALF],
        };
    }

    // Queries: dense = centre_q, sparse = ones on F_q. Ground truth R_q =
    // R_dense ∪ R_sparse (the full K planted docs).
    let mut dense_q: Vec<Vec<f32>> = Vec::with_capacity(Q);
    let mut sparse_q: Vec<SparseVector> = Vec::with_capacity(Q);
    let mut neighbours: Vec<Vec<i64>> = Vec::with_capacity(Q);
    for q in 0..Q {
        dense_q.push(centre(q));
        sparse_q.push(SparseVector {
            indices: (q * HALF..q * HALF + HALF).map(|i| i as u32).collect(),
            values: vec![1.0f32; HALF],
        });
        let mut gt: Vec<i64> = (0..HALF).map(|j| r_dense_id(q, j) as i64).collect();
        gt.extend((0..HALF).map(|j| r_sparse_id(q, j) as i64));
        neighbours.push(gt);
    }

    HybridData {
        dense,
        dense_queries: dense_q,
        sparse,
        sparse_queries: sparse_q,
        neighbours,
        dim: DENSE_DIM,
        top: K,
    }
}

/// A generated multi-vector (ColBERT-style) dataset: `data` (corpus) +
/// `queries`, with `neighbours` the top-`top` brute-force MaxSim
/// (late-interaction, descending) ground truth.
pub struct MultiVectorGenData {
    pub data: Vec<MultiVector>,
    pub queries: Vec<MultiVector>,
    pub neighbours: Vec<Vec<i64>>,
}

/// Corpus size of the shipped `synthetic-multivector-16` dataset. Single
/// source of truth, mirroring [`SYNTHETIC_SPARSE_ROWS`].
pub const SYNTHETIC_MULTIVECTOR_ROWS: usize = 150;

/// Per-token dimensionality of `synthetic-multivector-16` — the `16` in its name.
pub const SYNTHETIC_MULTIVECTOR_DIM: usize = 16;

/// Generate a deterministic random multi-vector dataset and its MaxSim
/// (descending) ground truth, genuinely brute-forced — NOT planted like
/// [`generate_hybrid`]'s ring construction, which only works because an
/// RRF-fused rank can be built from two independent single-scalar rankings.
/// MaxSim (`sum over query tokens of max over doc tokens of dot(q, d)`) has no
/// such decomposition, so this mirrors [`generate_sparse`]'s real-brute-force
/// pattern instead: every doc and query gets a random number of random token
/// vectors, and ground truth is the actual top-`top` MaxSim ranking over the
/// full corpus.
///
/// `seed` fixes the RNG; `dim` is the per-token dimensionality, `min_tokens`/
/// `max_tokens` bound the (inclusive) random token count per row, `n` the
/// corpus size, `q` the query count and `top` the number of ground-truth
/// neighbours per query.
///
/// Ground truth is always scored with raw dot-product MaxSim, regardless of
/// whatever `distance` the caller later registers the dataset under in
/// `datasets.json` — correct for a `dot`-declared dataset (what this repo
/// ships), but silently wrong for an `l2`-declared one: Qdrant would rank by
/// Euclid-MaxSim while this ground truth is dot-MaxSim. An `l2` multivector
/// dataset needs its own Euclid-MaxSim ground truth, not this function's.
pub fn generate_multivector(
    seed: u64,
    dim: usize,
    min_tokens: usize,
    max_tokens: usize,
    n: usize,
    q: usize,
    top: usize,
) -> MultiVectorGenData {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut make = |count: usize| -> Vec<MultiVector> {
        (0..count)
            .map(|_| {
                let num_tokens = rng.gen_range(min_tokens..=max_tokens);
                let vectors: Vec<Vec<f32>> = (0..num_tokens)
                    .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
                    .collect();
                MultiVector { vectors }
            })
            .collect()
    };
    let data = make(n);
    let queries = make(q);

    let dot =
        |a: &[f32], b: &[f32]| -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum() };
    // MaxSim: for each query token, the best-matching doc token; sum across
    // query tokens. Larger = more similar, so ground truth sorts DESCENDING.
    let maxsim = |query: &MultiVector, doc: &MultiVector| -> f64 {
        query
            .vectors
            .iter()
            .map(|qv| {
                doc.vectors
                    .iter()
                    .map(|dv| dot(qv, dv))
                    .fold(f64::MIN, f64::max)
            })
            .sum()
    };

    let neighbours: Vec<Vec<i64>> = queries
        .iter()
        .map(|qv| {
            let mut scored: Vec<(i64, f64)> = data
                .iter()
                .enumerate()
                .map(|(i, d)| (i as i64, maxsim(qv, d)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.iter().take(top).map(|(id, _)| *id).collect()
        })
        .collect();

    MultiVectorGenData {
        data,
        queries,
        neighbours,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_generation_is_deterministic_and_shaped() {
        let a = generate_sparse(0x5A5A_5EED, 300, 10, 150, 10, 10);
        let b = generate_sparse(0x5A5A_5EED, 300, 10, 150, 10, 10);
        assert_eq!(a.data.len(), 150);
        assert_eq!(a.queries.len(), 10);
        assert_eq!(a.neighbours.len(), 10);
        assert!(a.neighbours.iter().all(|n| n.len() == 10));
        // Every vector has exactly nnz non-zeros with distinct, sorted indices.
        for v in a.data.iter().chain(a.queries.iter()) {
            assert_eq!(v.indices.len(), 10);
            assert_eq!(v.values.len(), 10);
            assert!(v.indices.windows(2).all(|w| w[0] < w[1]));
        }
        // Deterministic across calls.
        assert_eq!(a.data, b.data);
        assert_eq!(a.neighbours, b.neighbours);
    }

    #[test]
    fn hybrid_generation_is_deterministic_and_row_aligned() {
        let h = generate_hybrid(0xB19_1DEA);
        assert_eq!(h.dim, 16);
        assert_eq!(h.top, 8);
        assert_eq!(h.dense.len(), h.sparse.len());
        assert_eq!(h.dense_queries.len(), h.sparse_queries.len());
        assert_eq!(h.neighbours.len(), h.dense_queries.len());
        assert!(h.neighbours.iter().all(|n| n.len() == h.top));
        assert!(h.dense.iter().all(|v| v.len() == h.dim));
        let h2 = generate_hybrid(0xB19_1DEA);
        assert_eq!(h.dense, h2.dense);
        assert_eq!(h.neighbours, h2.neighbours);
    }

    #[test]
    fn multivector_generation_is_deterministic_and_shaped() {
        let a = generate_multivector(0xC0FFEE, 16, 4, 8, 150, 10, 10);
        let b = generate_multivector(0xC0FFEE, 16, 4, 8, 150, 10, 10);
        assert_eq!(a.data.len(), 150);
        assert_eq!(a.queries.len(), 10);
        assert_eq!(a.neighbours.len(), 10);
        assert!(a.neighbours.iter().all(|n| n.len() == 10));
        for row in a.data.iter().chain(a.queries.iter()) {
            assert!((4..=8).contains(&row.vectors.len()));
            assert!(row.vectors.iter().all(|v| v.len() == 16));
        }
        // Deterministic across calls.
        assert_eq!(a.data.len(), b.data.len());
        assert_eq!(a.neighbours, b.neighbours);
    }

    /// Ground truth must be a REAL brute-force MaxSim ranking, not planted:
    /// recompute MaxSim by hand for every corpus row against query 0 and check
    /// the generator's top-K matches the hand-computed top-K exactly.
    #[test]
    fn multivector_ground_truth_is_genuinely_brute_forced() {
        let g = generate_multivector(0xC0FFEE, 8, 3, 5, 60, 3, 5);
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
        };
        let maxsim = |q: &MultiVector, d: &MultiVector| -> f64 {
            q.vectors
                .iter()
                .map(|qv| {
                    d.vectors
                        .iter()
                        .map(|dv| dot(qv, dv))
                        .fold(f64::MIN, f64::max)
                })
                .sum()
        };
        let query = &g.queries[0];
        let mut scored: Vec<(i64, f64)> = g
            .data
            .iter()
            .enumerate()
            .map(|(i, d)| (i as i64, maxsim(query, d)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let expected: Vec<i64> = scored.iter().take(5).map(|(id, _)| *id).collect();
        assert_eq!(g.neighbours[0], expected);
    }
}
