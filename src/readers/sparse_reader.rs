//! Sparse vector reader.
//!
//! Reads sparse vectors stored as a binary CSR (compressed sparse row) matrix,
//! matching the format used by qdrant/vector-db-benchmark's `sparse_reader.py`:
//!
//! ```text
//! [ n_row: i64 ][ n_col: i64 ][ n_non_zero: i64 ]
//! [ index_pointer: i64 × (n_row + 1) ]
//! [ columns: i32 × n_non_zero ]
//! [ values:  f32 × n_non_zero ]
//! ```
//!
//! Row `i` is the sparse vector with `indices = columns[ip[i]..ip[i+1]]` and the
//! parallel `values` slice.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// A single sparse vector: parallel `indices` (dimension ids) and `values`.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Read `n` little-endian `i64` values, bounding the up-front allocation by
/// `max_bytes` (the file size). A corrupt/hostile header can claim an absurd
/// count; capping the allocation to what the file could actually contain turns
/// an OOM into a clean `Err`. Size math is checked to reject integer overflow.
fn read_i64_le(r: &mut impl Read, n: usize, max_bytes: u64) -> Result<Vec<i64>, String> {
    let byte_len = n
        .checked_mul(8)
        .ok_or_else(|| "CSR size overflow (i64 count too large)".to_string())?;
    if byte_len as u64 > max_bytes {
        return Err(format!(
            "CSR claims {} bytes but file is only {} bytes",
            byte_len, max_bytes
        ));
    }
    let mut buf = vec![0u8; byte_len];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    // `as_chunks::<8>()` yields `&[u8; 8]` directly, so the fallible
    // `try_into().unwrap()` that `chunks_exact` forced is gone: the length is
    // now proven by the type rather than re-checked at runtime.
    Ok(buf
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect())
}

/// Read a `n_non_zero`-long array of little-endian 4-byte values, mapping each
/// via `f`. Allocation is bounded by `max_bytes` and the byte length is checked
/// for overflow, so a hostile `n_non_zero` cannot OOM or wrap.
fn read_u32_array<T>(
    r: &mut impl Read,
    n_non_zero: usize,
    max_bytes: u64,
    f: impl Fn([u8; 4]) -> T,
) -> Result<Vec<T>, String> {
    let byte_len = n_non_zero
        .checked_mul(4)
        .ok_or_else(|| "CSR nnz size overflow".to_string())?;
    if byte_len as u64 > max_bytes {
        return Err(format!(
            "CSR claims {} bytes but file is only {} bytes",
            byte_len, max_bytes
        ));
    }
    let mut buf = vec![0u8; byte_len];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf.as_chunks::<4>().0.iter().map(|c| f(*c)).collect())
}

/// Number of rows a CSR file declares, read from its 24-byte header WITHOUT
/// parsing the matrix.
///
/// The sibling of [`crate::readers::npy_row_count`], and for the same reason: it
/// makes the corpus — not `datasets.json` — the authority on how many vectors
/// exist (#224). Until this existed, `sparse` was the one shipped layout with no
/// cheap row count, so the `--skip-upload` reuse check had to TRUST the declared
/// `vector_count`; a wrong declaration then classified a correct corpus as
/// `Short` (false abort) or a genuinely short one as `Surplus` (warn, and publish
/// the wrong number) — the exact failure class #290 exists to close, reached
/// through the datasets.json door.
///
/// Cost is a 24-byte read, so this is safe to call on a multi-GB `data.csr`.
/// Validation is deliberately minimal and matches what the header alone can
/// support: the first `i64` must be non-negative, and the file must be at least
/// as long as the three-`i64` header. Everything structural (index_pointer
/// monotonicity, nnz agreement) is [`read_sparse_matrix`]'s job — a count read
/// must not have to parse the matrix to answer.
pub fn csr_row_count(path: &str) -> Result<u64, String> {
    let file = File::open(Path::new(path)).map_err(|e| format!("open {}: {}", path, e))?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if file_len < 24 {
        return Err(format!(
            "{}: not a readable CSR file ({} bytes, header needs 24)",
            path, file_len
        ));
    }
    let mut r = BufReader::new(file);
    let sizes = read_i64_le(&mut r, 3, file_len)?;
    let n_row = sizes[0];
    if n_row < 0 {
        return Err(format!(
            "{}: invalid CSR header (n_row {} is negative)",
            path, n_row
        ));
    }
    Ok(n_row as u64)
}

/// Parse a CSR file into a list of sparse vectors.
pub fn read_sparse_matrix(path: &str) -> Result<Vec<SparseVector>, String> {
    let file = File::open(Path::new(path)).map_err(|e| format!("open {}: {}", path, e))?;
    // File size is an upper bound on any array the header can legitimately
    // describe; use it to cap every up-front allocation so a corrupt header
    // cannot OOM the process before `read_exact` would fail.
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut r = BufReader::new(file);

    let sizes = read_i64_le(&mut r, 3, file_len)?;
    let (n_row, n_col, n_non_zero) = (sizes[0], sizes[1], sizes[2]);
    if n_row < 0 || n_col < 0 || n_non_zero < 0 {
        return Err(format!("invalid CSR header in {}", path));
    }
    let n_row = n_row as usize;
    let n_non_zero = n_non_zero as usize;

    let ip_count = n_row
        .checked_add(1)
        .ok_or_else(|| "CSR n_row overflow".to_string())?;
    let index_pointer = read_i64_le(&mut r, ip_count, file_len)?;
    if index_pointer.last().copied().unwrap_or(0) as usize != n_non_zero {
        return Err(format!("CSR index_pointer/nnz mismatch in {}", path));
    }
    // Validate the whole index_pointer array BEFORE using any entry to slice:
    // every value must be non-negative, `<= n_non_zero`, and monotonically
    // non-decreasing. Otherwise `columns[start..end]` could panic on an
    // out-of-bounds or `start > end` range.
    for pair in index_pointer.windows(2) {
        let (prev, next) = (pair[0], pair[1]);
        if prev < 0 || next < 0 {
            return Err(format!("CSR index_pointer has negative offset in {}", path));
        }
        if prev > next {
            return Err(format!("CSR index_pointer not monotonic in {}", path));
        }
        if next as usize > n_non_zero {
            return Err(format!(
                "CSR index_pointer offset {} exceeds nnz {} in {}",
                next, n_non_zero, path
            ));
        }
    }

    let columns: Vec<u32> = read_u32_array(&mut r, n_non_zero, file_len, |c| {
        i32::from_le_bytes(c) as u32
    })?;
    let values: Vec<f32> = read_u32_array(&mut r, n_non_zero, file_len, f32::from_le_bytes)?;

    let mut out = Vec::with_capacity(n_row);
    for i in 0..n_row {
        let start = index_pointer[i] as usize;
        let end = index_pointer[i + 1] as usize;
        // `.get` instead of `[..]`: validated above, but stay panic-free even if
        // an invariant is ever missed.
        let indices = columns
            .get(start..end)
            .ok_or_else(|| format!("CSR columns range {}..{} out of bounds", start, end))?;
        let vals = values
            .get(start..end)
            .ok_or_else(|| format!("CSR values range {}..{} out of bounds", start, end))?;
        out.push(SparseVector {
            indices: indices.to_vec(),
            values: vals.to_vec(),
        });
    }
    Ok(out)
}

/// Write a list of sparse vectors as a CSR file (used by tests / fixtures).
pub fn write_sparse_matrix(path: &str, rows: &[SparseVector]) -> Result<(), String> {
    use std::io::Write;
    let n_row = rows.len() as i64;
    let n_col = rows
        .iter()
        .flat_map(|r| r.indices.iter())
        .map(|&i| i as i64 + 1)
        .max()
        .unwrap_or(0);
    let n_non_zero: i64 = rows.iter().map(|r| r.indices.len() as i64).sum();

    let mut buf = Vec::new();
    for v in [n_row, n_col, n_non_zero] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut ip: i64 = 0;
    buf.extend_from_slice(&ip.to_le_bytes());
    for r in rows {
        ip += r.indices.len() as i64;
        buf.extend_from_slice(&ip.to_le_bytes());
    }
    for r in rows {
        for &c in &r.indices {
            buf.extend_from_slice(&(c as i32).to_le_bytes());
        }
    }
    for r in rows {
        for &val in &r.values {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    let mut f = File::create(Path::new(path)).map_err(|e| e.to_string())?;
    f.write_all(&buf).map_err(|e| e.to_string())?;
    Ok(())
}

/// Read a binary k-NN ground-truth file (`results.gt`), the format shipped by
/// the public `msmarco-sparse-*` datasets and produced by upstream's
/// `knn_result_read`:
///
/// ```text
/// [ n: u32 ][ d: u32 ]
/// [ ids:    i32 × (n × d) ]
/// [ scores: f32 × (n × d) ]
/// ```
///
/// `n` is the query count and `d` the neighbours per query. Only the ids are
/// returned — recall is computed from ids alone, and the scores block is
/// engine-independent.
///
/// The file length MUST be exactly `8 + n*d*8`. A truncated or mis-declared file
/// is rejected rather than read short: short ground truth silently DEFLATES
/// measured recall, which is worse than a failed run.
///
/// `d == 0` alongside `n > 0` is rejected for the same reason. It would otherwise
/// pass the length check (an 8-byte file can legitimately claim any `n` when
/// `n*d == 0`) and yield ZERO rows for a non-empty query set — a silent empty
/// ground truth, exactly what the length check exists to prevent.
///
/// Note the length check constrains only the PRODUCT `n*d`, so it cannot by
/// itself detect a transposed header — (n=2, d=4) and (n=4, d=2) have identical
/// byte lengths. `Dataset::read_sparse_queries` pins `n` against the real query
/// count, which then pins `d` too; this reader alone is not sufficient.
pub fn read_gt_neighbours(path: &str) -> Result<Vec<Vec<i64>>, String> {
    let file = File::open(Path::new(path)).map_err(|e| format!("open {}: {}", path, e))?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut r = BufReader::new(file);

    let mut header = [0u8; 8];
    r.read_exact(&mut header)
        .map_err(|e| format!("read {} header: {}", path, e))?;
    let n = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    // Zero neighbours per query for a non-empty query set is a mis-declared
    // header, not an empty result — reject before the length check, which cannot
    // distinguish it (n*d == 0 for ANY n).
    if d == 0 && n > 0 {
        return Err(format!(
            "ground-truth {} declares {} queries but 0 neighbours each",
            path, n
        ));
    }

    let count = n
        .checked_mul(d)
        .ok_or_else(|| format!("ground-truth n*d overflow in {}", path))?;
    // 4 bytes per id + 4 per score, plus the 8-byte header.
    let expected = count
        .checked_mul(8)
        .and_then(|b| b.checked_add(8))
        .ok_or_else(|| format!("ground-truth size overflow in {}", path))?;
    if expected as u64 != file_len {
        return Err(format!(
            "ground-truth {} declares {} queries × {} neighbours ({} bytes) but the file is {} bytes",
            path, n, d, expected, file_len
        ));
    }

    // `read_u32_array` read_exact's a `count * 4` buffer, so it returns exactly
    // `count` ids or errors — no short-read check needed here.
    let ids = read_u32_array(&mut r, count, file_len, i32::from_le_bytes)?;
    // `d > 0` is guaranteed above whenever there is anything to chunk, so this
    // yields exactly `n` rows of `d` ids.
    Ok(ids
        .chunks(d.max(1))
        .map(|c| c.iter().map(|&i| i as i64).collect())
        .collect())
}

/// Write a `results.gt` ground-truth file (used by tests / fixtures). Every row
/// must have the same neighbour count, as the format stores a single `d`.
///
/// Ids must fit in `i32` — the on-disk format is int32. An out-of-range id is
/// REJECTED rather than truncated: `id as i32` would silently write a different
/// (often negative) id, so `write` then `read` would not round-trip and a fixture
/// would carry ground truth nobody wrote.
pub fn write_gt_neighbours(path: &str, rows: &[Vec<i64>]) -> Result<(), String> {
    use std::io::Write;
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    if rows.iter().any(|r| r.len() != d) {
        return Err("ground-truth rows must all have the same length".to_string());
    }
    // Refuse to write a shape our own reader rejects. `n > 0, d == 0` produces an
    // 8-byte file that satisfies the length check while carrying no ground truth,
    // so `read_gt_neighbours` errors on it — a writer that can emit input its
    // reader calls corrupt is the defect, not an accepted asymmetry.
    if d == 0 && !rows.is_empty() {
        return Err(format!(
            "cannot write ground truth for {} queries with 0 neighbours each: the format \
             cannot represent it (and read_gt_neighbours rejects it)",
            rows.len()
        ));
    }
    if let Some(bad) = rows
        .iter()
        .flatten()
        .find(|&&id| i32::try_from(id).is_err())
    {
        return Err(format!(
            "ground-truth id {} does not fit in the format's int32 id field",
            bad
        ));
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(d as u32).to_le_bytes());
    for row in rows {
        for &id in row {
            buf.extend_from_slice(&(id as i32).to_le_bytes());
        }
    }
    // Scores are not read back, but the block must exist for the length check.
    for row in rows {
        for _ in row {
            buf.extend_from_slice(&0f32.to_le_bytes());
        }
    }
    let mut f = File::create(Path::new(path)).map_err(|e| e.to_string())?;
    f.write_all(&buf).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_csr_sparse_matrix() {
        let rows = vec![
            SparseVector {
                indices: vec![0, 5, 9],
                values: vec![1.0, 2.5, -3.0],
            },
            SparseVector {
                indices: vec![],
                values: vec![],
            },
            SparseVector {
                indices: vec![3],
                values: vec![0.75],
            },
        ];
        let dir = std::env::temp_dir();
        let path = dir
            .join(format!("vdb_sparse_test_{}.csr", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        write_sparse_matrix(&path, &rows).unwrap();
        let read = read_sparse_matrix(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, rows);
    }

    // ---- Regression tests for fuzzer-found crashes ----
    // Each writes exactly-crafted malformed CSR bytes and asserts the reader
    // returns Err instead of panicking / overflowing / OOMing.

    fn write_tmp(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    /// The count must come out of LITERAL little-endian bytes, not out of
    /// whatever `write_sparse_matrix` happens to emit (#290 review).
    ///
    /// Without this, `csr_row_count`, `read_sparse_matrix` and the fixture
    /// writer all agree by construction: a COHERENT endianness flip
    /// (`from_le_bytes` + `to_le_bytes` → `_be_`) left the whole suite green,
    /// because every party to the comparison flipped together. The CSR format is
    /// little-endian by definition — it is what qdrant/vector-db-benchmark's
    /// `sparse_reader.py` writes — so a byte fixture is the only thing that pins
    /// it. `results.gt` already has this treatment in
    /// `gt_decodes_a_literal_little_endian_fixture`; with both, that flip now
    /// fails two tests instead of none.
    ///
    /// `n_row = 150` is the real `synthetic-sparse-300` count, and decodes
    /// big-endian to 0x9600000000000000 — wildly wrong rather than subtly so.
    #[test]
    fn csr_row_count_decodes_a_literal_little_endian_fixture() {
        #[rustfmt::skip]
        let bytes: &[u8] = &[
            // n_row = 150
            0x96, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // n_col = 300 — two non-zero bytes, so a swap shows here too if the
            // header fields are ever read in a different order
            0x2c, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // n_non_zero = 1500
            0xdc, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let f = write_tmp(bytes);
        assert_eq!(
            csr_row_count(f.path().to_str().unwrap()).unwrap(),
            150,
            "n_row must decode little-endian"
        );
    }

    /// `csr_row_count` must agree with what `read_sparse_matrix` yields on a
    /// WELL-FORMED file — it is a cheap substitute for parsing, not a second
    /// opinion (#290 review).
    ///
    /// Scope, so the name does not overstate: they agree on well-formed input
    /// only. A valid header over a truncated body gives `Ok(n)` here and `Err`
    /// there, by design — a count read must not have to parse the matrix to
    /// answer. The header is itself a declaration, just one stored inside the
    /// corpus rather than in `datasets.json`; what the change buys is that the
    /// declaration now travels WITH the file it describes.
    #[test]
    fn csr_row_count_agrees_with_a_full_parse_on_well_formed_input() {
        for n in [0usize, 1, 150] {
            let rows: Vec<SparseVector> = (0..n)
                .map(|i| SparseVector {
                    indices: vec![i as u32 % 5],
                    values: vec![0.5],
                })
                .collect();
            let f = tempfile::NamedTempFile::new().unwrap();
            let path = f.path().to_str().unwrap().to_string();
            write_sparse_matrix(&path, &rows).unwrap();
            assert_eq!(csr_row_count(&path).unwrap(), n as u64, "n = {n}");
            assert_eq!(read_sparse_matrix(&path).unwrap().len(), n, "n = {n}");
        }
    }

    /// A count read must not be fooled by a file too short to hold a header, and
    /// must not report a negative n_row as an enormous unsigned one. Both would
    /// feed a bogus "expected rows" straight into the --skip-upload verdict.
    #[test]
    fn csr_row_count_rejects_a_truncated_or_negative_header() {
        let short = write_tmp(&[0u8; 23]);
        let err = csr_row_count(short.path().to_str().unwrap()).unwrap_err();
        assert!(err.contains("not a readable CSR file"), "{err}");

        let mut b = Vec::new();
        b.extend_from_slice(&(-5i64).to_le_bytes()); // n_row
        b.extend_from_slice(&1i64.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        let neg = write_tmp(&b);
        let err = csr_row_count(neg.path().to_str().unwrap()).unwrap_err();
        assert!(err.contains("negative"), "{err}");
    }

    /// Header n_row so large that `(n_row + 1) * 8` overflows usize.
    /// Previously panicked "attempt to multiply with overflow" at read_i64_le.
    #[test]
    fn rejects_index_pointer_count_overflow() {
        let mut b = Vec::new();
        b.extend_from_slice(&i64::MAX.to_le_bytes()); // n_row
        b.extend_from_slice(&1i64.to_le_bytes()); // n_col
        b.extend_from_slice(&0i64.to_le_bytes()); // nnz
        let f = write_tmp(&b);
        assert!(read_sparse_matrix(f.path().to_str().unwrap()).is_err());
    }

    /// nnz so large that `nnz * 4` overflows usize.
    /// Previously panicked "attempt to multiply with overflow" at columns alloc.
    #[test]
    fn rejects_nnz_byte_overflow() {
        let nnz: i64 = 1 << 62; // *4 overflows u64/usize
        let mut b = Vec::new();
        b.extend_from_slice(&0i64.to_le_bytes()); // n_row = 0 -> ip has 1 elem
        b.extend_from_slice(&1i64.to_le_bytes()); // n_col
        b.extend_from_slice(&nnz.to_le_bytes()); // nnz
        b.extend_from_slice(&nnz.to_le_bytes()); // index_pointer[0] == nnz
        let f = write_tmp(&b);
        assert!(read_sparse_matrix(f.path().to_str().unwrap()).is_err());
    }

    /// index_pointer with start > end (non-monotonic) for a row.
    /// Previously panicked "slice index starts at 5 but ends at 1".
    #[test]
    fn rejects_non_monotonic_index_pointer() {
        let mut b = Vec::new();
        b.extend_from_slice(&1i64.to_le_bytes()); // n_row = 1
        b.extend_from_slice(&1i64.to_le_bytes()); // n_col
        b.extend_from_slice(&1i64.to_le_bytes()); // nnz = 1
        b.extend_from_slice(&5i64.to_le_bytes()); // ip[0] = 5
        b.extend_from_slice(&1i64.to_le_bytes()); // ip[1] = 1 (last == nnz)
        b.extend_from_slice(&0i32.to_le_bytes()); // columns[0]
        b.extend_from_slice(&0f32.to_le_bytes()); // values[0]
        let f = write_tmp(&b);
        assert!(read_sparse_matrix(f.path().to_str().unwrap()).is_err());
    }

    /// index_pointer offset exceeding nnz (out of bounds).
    #[test]
    fn rejects_out_of_bounds_index_pointer() {
        let mut b = Vec::new();
        b.extend_from_slice(&2i64.to_le_bytes()); // n_row = 2
        b.extend_from_slice(&1i64.to_le_bytes()); // n_col
        b.extend_from_slice(&1i64.to_le_bytes()); // nnz = 1
        b.extend_from_slice(&0i64.to_le_bytes()); // ip[0] = 0
        b.extend_from_slice(&9i64.to_le_bytes()); // ip[1] = 9 > nnz
        b.extend_from_slice(&1i64.to_le_bytes()); // ip[2] = 1 (last == nnz)
        b.extend_from_slice(&0i32.to_le_bytes()); // columns[0]
        b.extend_from_slice(&0f32.to_le_bytes()); // values[0]
        let f = write_tmp(&b);
        assert!(read_sparse_matrix(f.path().to_str().unwrap()).is_err());
    }

    /// nnz large enough that `nnz * 4` fits in usize but implies a multi-TB
    /// allocation far exceeding the tiny file. Must Err (alloc cap), not OOM.
    #[test]
    fn rejects_absurd_allocation_from_small_file() {
        let nnz: i64 = 1 << 40; // *4 = 4 TiB
        let mut b = Vec::new();
        b.extend_from_slice(&0i64.to_le_bytes()); // n_row = 0
        b.extend_from_slice(&1i64.to_le_bytes()); // n_col
        b.extend_from_slice(&nnz.to_le_bytes()); // nnz
        b.extend_from_slice(&nnz.to_le_bytes()); // index_pointer[0] == nnz
        let f = write_tmp(&b);
        assert!(read_sparse_matrix(f.path().to_str().unwrap()).is_err());
    }

    // ---- results.gt (binary ground truth) ----

    /// GOLDEN test: hand-written bytes, no writer involved.
    ///
    /// `round_trips_gt_neighbours` below CANNOT catch a wrong on-disk format —
    /// `write_gt_neighbours` is a test-only encoder written against the same
    /// assumptions as the decoder, so flipping BOTH to big-endian ids (or
    /// swapping the ids and scores blocks) leaves the round-trip green. These
    /// bytes pin the layout the real `msmacro-sparse-*.tar.gz` files use:
    /// `[n: u32 LE][d: u32 LE][ids: i32 LE × n·d][scores: f32 × n·d]`, verified
    /// byte-for-byte against the real 100K download (n=6980, d=10, file length
    /// 558408 == 8 + 6980*10*8).
    #[test]
    fn gt_decodes_a_literal_little_endian_fixture() {
        #[rustfmt::skip]
        let bytes: &[u8] = &[
            // header: n = 2, d = 3
            0x02, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00,
            // ids row 0: 1, 256, 65536  (each distinguishes LE from BE)
            0x01, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x00,
            // ids row 1: 0, 16777216, 2147483647 (i32::MAX)
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01,
            0xff, 0xff, 0xff, 0x7f,
            // scores block: 6 f32. Never read, but it must FOLLOW the ids — if
            // the two blocks were swapped these zero bytes would decode as ids.
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let f = write_tmp(bytes);
        assert_eq!(
            read_gt_neighbours(f.path().to_str().unwrap()).unwrap(),
            vec![vec![1i64, 256, 65536], vec![0, 16_777_216, 2_147_483_647]],
            "results.gt ids must be parsed as little-endian i32, ids block first"
        );
    }

    #[test]
    fn round_trips_gt_neighbours() {
        let rows = vec![vec![7i64, 3, 11], vec![0, 1, 2]];
        let path = std::env::temp_dir()
            .join(format!("vdb_gt_test_{}.gt", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        write_gt_neighbours(&path, &rows).unwrap();
        let read = read_gt_neighbours(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, rows);
    }

    #[test]
    fn gt_empty_file_reads_as_no_queries() {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_le_bytes()); // n = 0
        b.extend_from_slice(&0u32.to_le_bytes()); // d = 0
        let f = write_tmp(&b);
        assert_eq!(
            read_gt_neighbours(f.path().to_str().unwrap()).unwrap(),
            Vec::<Vec<i64>>::new()
        );
    }

    /// `n > 0, d = 0` is an 8-byte file that SATISFIES the length check (n*d == 0
    /// for any n) and would otherwise return zero rows for a non-empty query set
    /// — a silent empty ground truth, which is precisely what the length check is
    /// there to prevent.
    #[test]
    fn gt_rejects_zero_neighbours_for_nonempty_query_set() {
        let mut b = Vec::new();
        b.extend_from_slice(&1000u32.to_le_bytes()); // n = 1000
        b.extend_from_slice(&0u32.to_le_bytes()); // d = 0
        let f = write_tmp(&b);
        let err = read_gt_neighbours(f.path().to_str().unwrap()).unwrap_err();
        assert!(
            err.contains("0 neighbours"),
            "expected a 0-neighbours rejection, got: {}",
            err
        );
    }

    /// The length check constrains only the product n*d, so a transposed header
    /// is byte-identical and parses "successfully" here. Documenting that limit
    /// in a test: the real defence is the query-count check in
    /// `Dataset::read_sparse_queries`, not this reader.
    #[test]
    fn gt_cannot_detect_a_transposed_header_alone() {
        let rows = vec![vec![0i64, 1, 2, 3], vec![4, 5, 6, 7]];
        let path = std::env::temp_dir()
            .join(format!("vdb_gt_shape_{}.gt", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        write_gt_neighbours(&path, &rows).unwrap(); // (n=2, d=4)
        let read = read_gt_neighbours(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        // Same 8 ids could equally be 4 rows of 2; this reader takes the header's
        // word for it.
        assert_eq!(read, rows);
        assert_eq!(read.len(), 2, "header shape is trusted at this layer");
    }

    /// A header promising more neighbours than the file holds must Err. Reading
    /// it short would silently deflate recall for every engine.
    #[test]
    fn gt_rejects_truncated_file() {
        let mut b = Vec::new();
        b.extend_from_slice(&2u32.to_le_bytes()); // n = 2
        b.extend_from_slice(&10u32.to_le_bytes()); // d = 10 -> needs 8 + 160 bytes
        b.extend_from_slice(&1i32.to_le_bytes()); // only one id present
        let f = write_tmp(&b);
        let err = read_gt_neighbours(f.path().to_str().unwrap()).unwrap_err();
        assert!(err.contains("but the file is"), "unexpected error: {}", err);
    }

    /// A header whose n*d overflows must Err, not wrap or OOM.
    #[test]
    fn gt_rejects_size_overflow() {
        let mut b = Vec::new();
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        let f = write_tmp(&b);
        assert!(read_gt_neighbours(f.path().to_str().unwrap()).is_err());
    }

    /// The on-disk id field is int32. An out-of-range id must be REFUSED, not
    /// truncated by `as i32` into a different (often negative) id, which would
    /// make write→read non-identity and bake wrong ground truth into a fixture.
    #[test]
    fn gt_writer_rejects_ids_outside_i32() {
        let path = std::env::temp_dir()
            .join(format!("vdb_gt_range_{}.gt", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        for bad in [i64::from(i32::MAX) + 1, i64::from(i32::MIN) - 1, i64::MAX] {
            let err = write_gt_neighbours(&path, &[vec![1i64, bad]]).unwrap_err();
            assert!(err.contains("int32"), "id {bad}: {err}");
        }
        // In-range values, including the -1 padding sentinel, are fine.
        assert!(write_gt_neighbours(&path, &[vec![-1i64, i64::from(i32::MAX)]]).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    /// The writer must not emit a shape its own reader rejects: `n > 0, d == 0`
    /// is an 8-byte file that passes the length check while carrying no ground
    /// truth, so refusing it at the writer keeps the codec self-consistent.
    #[test]
    fn gt_writer_rejects_zero_width_rows() {
        let path = std::env::temp_dir()
            .join(format!("vdb_gt_zerow_{}.gt", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        let err = write_gt_neighbours(&path, &[vec![], vec![]]).unwrap_err();
        assert!(err.contains("0 neighbours"), "got: {err}");
        // No rows at all is legitimately empty and still writes.
        assert!(write_gt_neighbours(&path, &[]).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gt_writer_rejects_ragged_rows() {
        let path = std::env::temp_dir()
            .join(format!("vdb_gt_ragged_{}.gt", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        assert!(write_gt_neighbours(&path, &[vec![1i64, 2], vec![3]]).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
