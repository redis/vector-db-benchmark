//! Multi-vector (ColBERT-style / late-interaction) reader.
//!
//! Reads a document or query set where each row is a *variable-length* list
//! of token embeddings (e.g. one embedding per token of a ColBERT-style
//! model, scored via MaxSim), stored as:
//!
//! ```text
//! [ num_rows: u32 ][ dim: u32 ]
//! [ token_count: u32 × num_rows ]
//! [ values: f32 × (sum(token_count) × dim), row-major: row 0's tokens (each
//!   `dim` floats, one token after another), then row 1's, ... ]
//! ```
//!
//! Unlike CSR (`sparse_reader.rs`), there is no upstream reference format to
//! stay byte-compatible with — this layout is new, invented for this repo's
//! multi-vector support, since neither NPY (rectangular) nor CSR (2D sparse)
//! can represent a ragged list of per-document token vectors.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// One document's (or query's) multi-vector representation: a list of token
/// embeddings, each of the dataset's declared `dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiVector {
    pub vectors: Vec<Vec<f32>>,
}

fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(buf))
}

/// Read `n` little-endian `u32` values, bounding the up-front allocation by
/// `max_bytes` (the file size). A corrupt/hostile header can claim an absurd
/// count; capping the allocation to what the file could actually contain
/// turns an OOM into a clean `Err` (mirrors `sparse_reader::read_u32_array`).
fn read_u32_array(r: &mut impl Read, n: usize, max_bytes: u64) -> Result<Vec<u32>, String> {
    let byte_len = n
        .checked_mul(4)
        .ok_or_else(|| "multivector token-count size overflow".to_string())?;
    if byte_len as u64 > max_bytes {
        return Err(format!(
            "multivector file claims {} token-count bytes but file is only {} bytes",
            byte_len, max_bytes
        ));
    }
    let mut buf = vec![0u8; byte_len];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// Read `n` little-endian `f32` values, bounded the same way as
/// [`read_u32_array`].
fn read_f32_array(r: &mut impl Read, n: usize, max_bytes: u64) -> Result<Vec<f32>, String> {
    let byte_len = n
        .checked_mul(4)
        .ok_or_else(|| "multivector value size overflow".to_string())?;
    if byte_len as u64 > max_bytes {
        return Err(format!(
            "multivector file claims {} value bytes but file is only {} bytes",
            byte_len, max_bytes
        ));
    }
    let mut buf = vec![0u8; byte_len];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Parse a multivector file into a list of `MultiVector` rows.
pub fn read_multivector_matrix(path: &str) -> Result<Vec<MultiVector>, String> {
    let file = File::open(Path::new(path)).map_err(|e| format!("open {}: {}", path, e))?;
    // File size is an upper bound on any array the header can legitimately
    // describe; use it to cap every up-front allocation so a corrupt header
    // cannot OOM the process before `read_exact` would fail.
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if file_len < 8 {
        return Err(format!(
            "{}: not a readable multivector file ({} bytes, header needs 8)",
            path, file_len
        ));
    }
    let mut r = BufReader::new(file);

    let num_rows = read_u32(&mut r)? as usize;
    let dim = read_u32(&mut r)? as usize;

    let token_counts = read_u32_array(&mut r, num_rows, file_len)?;

    let total_tokens: u64 = token_counts.iter().map(|&c| c as u64).sum();
    let total_floats = total_tokens
        .checked_mul(dim as u64)
        .ok_or_else(|| format!("multivector total value count overflow in {}", path))?;
    if total_floats > usize::MAX as u64 {
        return Err(format!(
            "multivector total value count too large in {}",
            path
        ));
    }
    let values = read_f32_array(&mut r, total_floats as usize, file_len)?;

    let mut out = Vec::with_capacity(num_rows);
    let mut offset = 0usize;
    for &count in &token_counts {
        let n_floats = (count as usize)
            .checked_mul(dim)
            .ok_or_else(|| format!("multivector row size overflow in {}", path))?;
        let end = offset
            .checked_add(n_floats)
            .ok_or_else(|| format!("multivector row offset overflow in {}", path))?;
        let row_values = values
            .get(offset..end)
            .ok_or_else(|| format!("multivector row range {}..{} out of bounds", offset, end))?;
        let vectors: Vec<Vec<f32>> = if dim == 0 {
            Vec::new()
        } else {
            row_values.chunks(dim).map(|c| c.to_vec()).collect()
        };
        out.push(MultiVector { vectors });
        offset = end;
    }
    Ok(out)
}

/// Write a list of `MultiVector` rows to a multivector file (used by the
/// synthetic dataset generator and tests). Every token vector, across every
/// row, must share the same dimension.
pub fn write_multivector_matrix(path: &str, rows: &[MultiVector]) -> Result<(), String> {
    use std::io::Write;
    let dim = rows
        .iter()
        .flat_map(|r| r.vectors.iter())
        .map(|v| v.len())
        .next()
        .unwrap_or(0);
    if let Some(bad) = rows
        .iter()
        .flat_map(|r| r.vectors.iter())
        .find(|v| v.len() != dim)
    {
        return Err(format!(
            "multivector token dim {} does not match the dataset's dim {}",
            bad.len(),
            dim
        ));
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    for row in rows {
        buf.extend_from_slice(&(row.vectors.len() as u32).to_le_bytes());
    }
    for row in rows {
        for v in &row.vectors {
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    let mut f = File::create(Path::new(path)).map_err(|e| e.to_string())?;
    f.write_all(&buf).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn round_trips_multivector_matrix() {
        let rows = vec![
            MultiVector {
                vectors: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            },
            MultiVector { vectors: vec![] },
            MultiVector {
                vectors: vec![vec![-1.0, 0.5]],
            },
        ];
        let path = std::env::temp_dir()
            .join(format!("vdb_mvec_test_{}.mvec", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        write_multivector_matrix(&path, &rows).unwrap();
        let read = read_multivector_matrix(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, rows);
    }

    #[test]
    fn rejects_truncated_header() {
        let short = write_tmp(&[0u8; 7]);
        let err = read_multivector_matrix(short.path().to_str().unwrap()).unwrap_err();
        assert!(err.contains("not a readable multivector file"), "{err}");
    }

    #[test]
    fn rejects_token_count_size_overflow() {
        let mut b = Vec::new();
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // num_rows
        b.extend_from_slice(&16u32.to_le_bytes()); // dim
        let f = write_tmp(&b);
        assert!(read_multivector_matrix(f.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn rejects_total_value_count_overflow() {
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_le_bytes()); // num_rows = 1
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // dim
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // token_count[0]
        let f = write_tmp(&b);
        assert!(read_multivector_matrix(f.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn rejects_truncated_value_block() {
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_le_bytes()); // num_rows = 1
        b.extend_from_slice(&2u32.to_le_bytes()); // dim = 2
        b.extend_from_slice(&3u32.to_le_bytes()); // token_count[0] = 3 -> needs 24 bytes
        b.extend_from_slice(&1.0f32.to_le_bytes()); // only one float present
        let f = write_tmp(&b);
        assert!(read_multivector_matrix(f.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn writer_rejects_ragged_token_dim() {
        let rows = vec![MultiVector {
            vectors: vec![vec![1.0, 2.0], vec![3.0]],
        }];
        let path = std::env::temp_dir()
            .join(format!("vdb_mvec_ragged_{}.mvec", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        assert!(write_multivector_matrix(&path, &rows).is_err());
    }

    #[test]
    fn empty_dataset_round_trips() {
        let path = std::env::temp_dir()
            .join(format!("vdb_mvec_empty_{}.mvec", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        write_multivector_matrix(&path, &[]).unwrap();
        let read = read_multivector_matrix(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, Vec::<MultiVector>::new());
    }
}
