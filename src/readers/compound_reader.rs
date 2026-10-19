//! Compound format reader for datasets with vectors.npy and payloads.jsonl.

use std::fs::File;
use std::io::{BufRead, BufReader};

use super::metadata::{parse_metadata_from_json, MetadataItem};
use super::npy_reader::{npy_row_count, read_npy_vectors};

/// Read payloads from payloads.jsonl file, returning Vec<Option<MetadataItem>>.
/// Each line is a JSON object with key-value pairs.
pub fn read_payloads_jsonl(path: &str) -> Result<Vec<Option<MetadataItem>>, String> {
    let mut metadata = Vec::new();
    visit_payloads_jsonl(std::path::Path::new(path), |item| {
        metadata.push(item);
        Ok(())
    })?;
    Ok(metadata)
}

/// Visit one metadata row at a time, preserving blank rows as missing payloads.
/// The callback may reject a row without reading the remainder of the file.
fn visit_payloads_jsonl(
    path: &std::path::Path,
    mut visit: impl FnMut(Option<MetadataItem>) -> Result<(), String>,
) -> Result<u64, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open payloads file: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut rows = 0;
    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read payload row: {e}"))?
            == 0
        {
            return Ok(rows);
        }
        let item = if line.trim().is_empty() {
            None
        } else {
            let value = serde_json::from_str(&line)
                .map_err(|e| format!("Failed to parse payload row {}: {e}", rows + 1))?;
            parse_metadata_from_json(value)
        };
        visit(item)?;
        rows += 1;
    }
}

/// Visit compound metadata without reading or allocating the vector payload.
/// The NPY header supplies the expected row count; absent payloads mean missing
/// metadata, exactly as in `read_compound_data`. Memory is bounded by one row.
pub fn visit_compound_metadata(
    dir: &std::path::Path,
    visit: impl FnMut(Option<MetadataItem>) -> Result<(), String>,
) -> Result<(), String> {
    let vectors = dir.join("vectors.npy");
    let expected = npy_row_count(vectors.to_str().ok_or("Invalid vectors path")?)?;
    let payloads = dir.join("payloads.jsonl");
    if payloads.exists() {
        let actual = visit_payloads_jsonl(&payloads, visit)?;
        if actual != expected {
            return Err(format!(
                "Metadata count ({actual}) doesn't match vector count ({expected})"
            ));
        }
    }
    Ok(())
}

/// Read compound data from directory containing vectors.npy and optionally payloads.jsonl.
/// Returns (ids, vectors, metadata).
#[allow(clippy::type_complexity)]
pub fn read_compound_data(
    dir_path: &str,
    normalize: bool,
) -> Result<(Vec<i64>, Vec<Vec<f32>>, Vec<Option<MetadataItem>>), String> {
    let dir = std::path::Path::new(dir_path);

    let vectors_path = dir.join("vectors.npy");
    let payloads_path = dir.join("payloads.jsonl");

    // Read vectors from NPY
    let vectors_str = vectors_path
        .to_str()
        .ok_or_else(|| "Invalid vectors path".to_string())?;
    let (ids, vectors) = read_npy_vectors(vectors_str, normalize)?;

    // Read metadata from payloads.jsonl if it exists
    let all_metadata = if payloads_path.exists() {
        let payloads_str = payloads_path
            .to_str()
            .ok_or_else(|| "Invalid payloads path".to_string())?;
        let metadata = read_payloads_jsonl(payloads_str)?;

        // Ensure metadata count matches vector count
        if metadata.len() != vectors.len() {
            return Err(format!(
                "Metadata count ({}) doesn't match vector count ({})",
                metadata.len(),
                vectors.len()
            ));
        }
        metadata
    } else {
        // No payloads file - use empty metadata
        vec![None; vectors.len()]
    };

    Ok((ids, vectors, all_metadata))
}

/// Read only vectors from compound format (ignoring metadata).
/// Useful for engines that don't support metadata.
pub fn read_compound_vectors_only(
    dir_path: &str,
    normalize: bool,
) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
    let dir = std::path::Path::new(dir_path);
    let vectors_path = dir.join("vectors.npy");

    let vectors_str = vectors_path
        .to_str()
        .ok_or_else(|| "Invalid vectors path".to_string())?;

    read_npy_vectors(vectors_str, normalize)
}

/// Read queries from compound format tests.jsonl.
///
/// Each line is a JSON object:
/// ```json
/// {"query": [...], "conditions": {...}, "closest_ids": [...], "closest_scores": [...]}
/// ```
///
/// Returns (query_vectors, neighbors, conditions).
#[allow(clippy::type_complexity)]
/// Read `tests.jsonl`: query vectors, ground-truth ids, and per-query filter
/// conditions.
///
/// The conditions come back as a [`QueryConditions`], never as raw JSON. That is
/// the #219 guard: an engine cannot get at the per-query `conditions` value
/// except through `query_filter`'s resolvers, which refuse to turn a declared
/// filter into an unfiltered query. [`read_compound_queries_raw`] keeps the raw
/// vector for this crate's own tests.
pub fn read_compound_queries(
    dir_path: &str,
    normalize: bool,
) -> Result<
    (
        Vec<Vec<f32>>,
        Vec<Vec<i64>>,
        crate::query_filter::QueryConditions,
    ),
    String,
> {
    let (queries, neighbors, conditions) = read_compound_queries_raw(dir_path, normalize)?;
    Ok((
        queries,
        neighbors,
        crate::query_filter::QueryConditions::new(conditions),
    ))
}

#[allow(clippy::type_complexity)]
pub(crate) fn read_compound_queries_raw(
    dir_path: &str,
    normalize: bool,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<i64>>, Vec<Option<serde_json::Value>>), String> {
    let dir = std::path::Path::new(dir_path);
    let tests_path = dir.join("tests.jsonl");

    let file = File::open(&tests_path).map_err(|e| format!("Failed to open tests.jsonl: {}", e))?;
    let reader = BufReader::new(file);

    let mut queries: Vec<Vec<f32>> = Vec::new();
    let mut neighbors: Vec<Vec<i64>> = Vec::new();
    let mut conditions: Vec<Option<serde_json::Value>> = Vec::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Failed to read test line: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }

        let row: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("Failed to parse tests.jsonl: {}", e))?;

        // Parse query vector
        let query_arr = row
            .get("query")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "Missing 'query' field in tests.jsonl".to_string())?;
        let mut query: Vec<f32> = query_arr
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if normalize {
            let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in query.iter_mut() {
                    *x /= norm;
                }
            }
        }

        queries.push(query);

        // Parse ground truth neighbor IDs
        let ids = row
            .get("closest_ids")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
            .unwrap_or_default();

        neighbors.push(ids);

        // Parse filter conditions (may be absent)
        let cond = row.get("conditions").cloned();
        conditions.push(cond);
    }

    Ok((queries, neighbors, conditions))
}

#[cfg(test)]
mod metadata_stream_tests {
    use super::*;

    fn metadata_only_fixture(rows: usize) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.npy");
        super::super::npy_reader::write_npy_vectors(
            path.to_str().unwrap(),
            &vec![vec![0.0; 3]; rows],
        )
        .unwrap();
        // Leave the valid row-count header but remove every vector value. A
        // vector decoder cannot read this fixture; metadata validation can.
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        let header_len = file.metadata().unwrap().len() - (rows * 3 * 4) as u64;
        file.set_len(header_len).unwrap();
        dir
    }

    #[test]
    fn metadata_visit_reads_no_vector_values_and_preserves_blank_rows() {
        let dir = metadata_only_fixture(3);
        std::fs::write(
            dir.path().join("payloads.jsonl"),
            "{\"tag\":\"red\"}\n\n{}\n",
        )
        .unwrap();
        let mut rows = Vec::new();
        visit_compound_metadata(dir.path(), |item| {
            rows.push(item.is_some());
            Ok(())
        })
        .unwrap();
        assert_eq!(rows, vec![true, false, true]);
        assert!(read_compound_data(dir.path().to_str().unwrap(), false).is_err());
    }

    #[test]
    fn metadata_visit_stops_on_rejection_before_parsing_next_row() {
        let dir = metadata_only_fixture(2);
        std::fs::write(dir.path().join("payloads.jsonl"), "{}\nnot JSON\n").unwrap();
        let error =
            visit_compound_metadata(dir.path(), |_| Err("rejected first row".into())).unwrap_err();
        assert_eq!(error, "rejected first row");
    }

    #[test]
    fn metadata_visit_rejects_count_mismatch_and_bad_json() {
        let dir = metadata_only_fixture(2);
        let path = dir.path().join("payloads.jsonl");
        for rows in ["{}\n", "{}\n{}\n{}\n"] {
            std::fs::write(&path, rows).unwrap();
            let error = visit_compound_metadata(dir.path(), |_| Ok(())).unwrap_err();
            assert!(error.contains("doesn't match vector count"), "{error}");
        }
        std::fs::write(&path, "{}\nnot JSON\n").unwrap();
        assert!(visit_compound_metadata(dir.path(), |_| Ok(()))
            .unwrap_err()
            .contains("Failed to parse payload row 2"));
    }

    #[test]
    fn absent_payloads_preserve_missing_metadata_semantics() {
        let dir = metadata_only_fixture(2);
        visit_compound_metadata(dir.path(), |_| panic!("absent payloads have no rows")).unwrap();
    }
}
