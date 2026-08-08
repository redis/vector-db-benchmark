//! NPY file reader for vector datasets.

use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read};

/// Parse the `shape` tuple's dimensions out of an NPY header dict string, e.g.
/// `{'descr': '<f4', 'fortran_order': False, 'shape': (100, 25), }` -> `[100, 25]`.
/// Returns `None` if the tuple can't be located or holds a non-integer.
fn parse_npy_shape_dims(header: &str) -> Option<Vec<u64>> {
    let shape_idx = header.find("'shape'")?;
    let rest = &header[shape_idx..];
    let open = rest.find('(')?;
    let close = rest[open..].find(')')? + open;
    let inner = &rest[open + 1..close];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        dims.push(t.parse::<u64>().ok()?);
    }
    Some(dims)
}

/// Parse the element count and item size (bytes) from an NPY header dict string,
/// e.g. `{'descr': '<f4', 'fortran_order': False, 'shape': (100, 25), }`.
/// Returns `None` if the fields can't be located (caller then skips the
/// data-size bound and lets the real NPY parser report the error).
fn parse_npy_shape_itemsize(header: &str) -> Option<(u64, u64)> {
    // 'descr' -> trailing digits give the item size in bytes ('<f4' => 4).
    let descr_idx = header.find("'descr'")?;
    let after_descr = &header[descr_idx + "'descr'".len()..];
    let q_start = after_descr.find(['\'', '"'])?;
    let after_q = &after_descr[q_start + 1..];
    let q_end = after_q.find(['\'', '"'])?;
    let descr = &after_q[..q_end];
    let digits: String = descr.chars().filter(|c| c.is_ascii_digit()).collect();
    let itemsize: u64 = digits.parse().ok()?;

    // 'shape' -> product of the integers inside the (...) tuple.
    let mut count: u64 = 1;
    for dim in parse_npy_shape_dims(header)? {
        count = count.checked_mul(dim)?;
    }
    Some((count, itemsize))
}

/// An NPY file's header, read without touching the (potentially multi-GB) data
/// payload: the raw dict text plus the byte offset where the data begins and the
/// file's real length.
struct NpyHeader {
    dict: String,
    data_start: u64,
    file_len: u64,
}

/// Read ONLY the NPY header (magic + length prefix + dict) — never the data.
/// Returns `Ok(None)` when the file is too short or lacks the NPY magic, so the
/// caller can fall through and let the real parser report the error.
fn read_npy_header(path: &str) -> Result<Option<NpyHeader>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open NPY file: {}", e))?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

    let mut prefix = [0u8; 12];
    if file.read_exact(&mut prefix[..10]).is_err() {
        return Ok(None); // too short to be NPY; let the real parser report it
    }
    if &prefix[0..6] != b"\x93NUMPY" {
        return Ok(None);
    }
    let major = prefix[6];
    let (header_len, header_start) = if major >= 2 {
        if file.read_exact(&mut prefix[10..12]).is_err() {
            return Ok(None);
        }
        let hl = u32::from_le_bytes([prefix[8], prefix[9], prefix[10], prefix[11]]) as u64;
        (hl, 12u64)
    } else {
        let hl = u16::from_le_bytes([prefix[8], prefix[9]]) as u64;
        (hl, 10u64)
    };

    // The header dict itself must fit in the file.
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| "NPY header length overflow".to_string())?;
    if header_end > file_len {
        return Err(format!(
            "NPY header length {} exceeds file size {}",
            header_len, file_len
        ));
    }

    let mut header = vec![0u8; header_len as usize];
    if file.read_exact(&mut header).is_err() {
        return Ok(None);
    }
    let header = String::from_utf8_lossy(&header).into_owned();

    // Guard against pathological headers that make the downstream Python-dict
    // parser (`ndarray-npy`) blow up. A valid NPY header is a flat dict
    // `{'descr':.., 'fortran_order':.., 'shape':(..)}` that nests at most 2
    // levels deep (the outer `{}` then the `()` shape tuple). A fuzzed header
    // like `{{{{{{{{{...` sends the recursive-descent parser into a multi-minute
    // spin (found by the nightly fuzzer as a 1728s timeout). Reject absurd
    // nesting cheaply, before handing off. Valid headers (depth <= 2) are
    // unaffected.
    let mut depth: u32 = 0;
    let mut max_depth: u32 = 0;
    for b in header.bytes() {
        match b {
            b'{' | b'[' | b'(' => {
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            b'}' | b']' | b')' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    if max_depth > 4 {
        return Err(format!(
            "malformed NPY header: bracket nesting depth {} exceeds limit (4)",
            max_depth
        ));
    }

    Ok(Some(NpyHeader {
        dict: header,
        data_start: header_end,
        file_len,
    }))
}

/// Reject NPY files whose header declares more bytes than the file can possibly
/// contain, BEFORE handing them to `ndarray-npy` (which otherwise allocates the
/// declared header/data up front and can OOM on a corrupt/hostile header).
/// A mismatched magic or an unparseable header falls through to the real parser.
fn validate_npy_size_bound(path: &str) -> Result<(), String> {
    let Some(header) = read_npy_header(path)? else {
        return Ok(());
    };

    // The declared data payload must also fit in the file.
    if let Some((count, itemsize)) = parse_npy_shape_itemsize(&header.dict) {
        let data_bytes = count
            .checked_mul(itemsize)
            .ok_or_else(|| "NPY shape * itemsize overflow".to_string())?;
        let total = header
            .data_start
            .checked_add(data_bytes)
            .ok_or_else(|| "NPY total size overflow".to_string())?;
        if total > header.file_len {
            return Err(format!(
                "NPY declares {} data bytes but file has only {}",
                data_bytes, header.file_len
            ));
        }
    }
    Ok(())
}

/// Number of rows (the first `shape` dimension) declared by an NPY file, read
/// from its ~128-byte header WITHOUT loading the data payload.
///
/// This is the ground truth for "how many vectors does this corpus actually
/// hold?" and is cheap enough to call on a multi-GB `vectors.npy` in a hot path
/// (#224: a declared `vector_count` that disagrees with the corpus silently
/// corrupted recall, so the corpus itself must be the authority).
pub fn npy_row_count(path: &str) -> Result<u64, String> {
    let header =
        read_npy_header(path)?.ok_or_else(|| format!("{}: not a readable NPY file", path))?;
    let dims = parse_npy_shape_dims(&header.dict)
        .ok_or_else(|| format!("{}: could not parse NPY 'shape' from header", path))?;
    dims.first()
        .copied()
        .ok_or_else(|| format!("{}: NPY 'shape' tuple is empty", path))
}

/// Read vectors from NPY file, returning (ids, vectors).
/// If normalize is true, each vector is divided by its L2 norm.
pub fn read_npy_vectors(path: &str, normalize: bool) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
    validate_npy_size_bound(path)?;
    let file = File::open(path).map_err(|e| format!("Failed to open NPY file: {}", e))?;
    let reader = BufReader::new(file);

    let arr: Array2<f32> =
        Array2::read_npy(reader).map_err(|e| format!("Failed to read NPY file: {}", e))?;

    let (rows, _cols) = arr.dim();

    // Convert to Vec<Vec<f32>>
    let mut vectors: Vec<Vec<f32>> = arr.rows().into_iter().map(|row| row.to_vec()).collect();

    if normalize {
        for vec in vectors.iter_mut() {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    let ids: Vec<i64> = (0..rows as i64).collect();

    Ok((ids, vectors))
}

/// Write vectors to an NPY file as a 2-D `f32` array (row-major), the same
/// layout [`read_npy_vectors`] expects. All rows must have identical length.
/// IDs are implicit row indices, matching the compound-format convention.
pub fn write_npy_vectors(path: &str, vectors: &[Vec<f32>]) -> Result<(), String> {
    let rows = vectors.len();
    let cols = vectors.first().map(|v| v.len()).unwrap_or(0);
    if vectors.iter().any(|v| v.len() != cols) {
        return Err("All vectors must have the same dimension".to_string());
    }

    let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
    let arr = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| format!("Failed to build array: {}", e))?;

    let file = File::create(path).map_err(|e| format!("Failed to create NPY file: {}", e))?;
    let writer = BufWriter::new(file);
    arr.write_npy(writer)
        .map_err(|e| format!("Failed to write NPY file: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    /// Regression: fuzzer-found OOM. A v3.0 NPY header declaring a ~4 GiB header
    /// length caused `ndarray-npy` to allocate the header up front and OOM.
    /// Must now return Err (header length exceeds file size).
    #[test]
    fn rejects_absurd_npy_header_length() {
        let mut b = Vec::new();
        b.extend_from_slice(b"\x93NUMPY"); // magic
        b.push(3); // major version
        b.push(0); // minor version
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // 4 GiB header length
        b.extend_from_slice(&[0u8; 64]); // some padding bytes
        let f = write_tmp(&b);
        let r = read_npy_vectors(f.path().to_str().unwrap(), false);
        assert!(
            r.is_err(),
            "expected Err for absurd header length, got {:?}",
            r
        );
    }

    /// Regression: a small file declaring a huge `shape` must Err (declared data
    /// exceeds file size), not allocate the array and OOM.
    #[test]
    fn rejects_npy_shape_exceeding_file() {
        // Valid v1.0 header dict claiming shape (1_000_000_000, 1_000_000_000).
        let dict = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1000000000, 1000000000), }";
        let mut b = Vec::new();
        b.extend_from_slice(b"\x93NUMPY");
        b.push(1); // major
        b.push(0); // minor
        b.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        b.extend_from_slice(dict);
        let f = write_tmp(&b);
        let r = read_npy_vectors(f.path().to_str().unwrap(), false);
        assert!(r.is_err(), "expected Err for oversized shape, got {:?}", r);
    }

    /// Regression: a fuzzer-found header full of nested `{{{{{...` sent
    /// `ndarray-npy`'s recursive Python-dict parser into a 1728s spin (nightly
    /// timeout). The nesting-depth guard must reject it quickly (a real valid
    /// header nests at most 2 levels), so this test also completes near-instantly.
    #[test]
    fn rejects_deeply_nested_npy_header() {
        // Header length is honest (fits the file), so it passes the size bound;
        // the excessive brace nesting is what must trip the guard.
        let mut dict = Vec::new();
        dict.extend_from_slice(b"{'descr': '<f4', 'shape': (3, 3), ");
        dict.extend(std::iter::repeat_n(b'{', 40)); // pathological nesting
        let mut b = Vec::new();
        b.extend_from_slice(b"\x93NUMPY");
        b.push(1); // major
        b.push(0); // minor
        b.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        b.extend_from_slice(&dict);
        let f = write_tmp(&b);
        let r = read_npy_vectors(f.path().to_str().unwrap(), false);
        assert!(
            r.is_err(),
            "expected Err for deeply nested header, got {:?}",
            r
        );
        assert!(
            r.unwrap_err().contains("nesting depth"),
            "expected nesting-depth rejection message"
        );
    }

    /// Valid NPY files must still round-trip identically (hardening must not
    /// change behavior for legitimate input).
    #[test]
    fn valid_npy_still_round_trips() {
        let vectors = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let f = tempfile::NamedTempFile::new().unwrap();
        let path = f.path().to_str().unwrap();
        write_npy_vectors(path, &vectors).unwrap();
        let (ids, read) = read_npy_vectors(path, false).unwrap();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(read, vectors);
    }

    /// `npy_row_count` must report the ROW count (first shape dim), not the
    /// element count — it is what the shared-corpus completeness gate compares
    /// against (#224) — and must not read the data payload.
    #[test]
    fn row_count_reads_rows_from_the_header_only() {
        let vectors: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32; 5]).collect();
        let f = tempfile::NamedTempFile::new().unwrap();
        let path = f.path().to_str().unwrap();
        write_npy_vectors(path, &vectors).unwrap();
        assert_eq!(npy_row_count(path).unwrap(), 7);
    }

    #[test]
    fn row_count_rejects_a_non_npy_file() {
        let f = write_tmp(b"this is not an NPY file at all, not even close");
        assert!(npy_row_count(f.path().to_str().unwrap()).is_err());
    }
}
