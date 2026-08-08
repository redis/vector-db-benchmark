//! Dataset readers for various file formats.
//! Provides readers for HDF5, JSONL, NPY, and compound (tar) formats.

mod compound_reader;
mod hdf5_reader;
mod jsonl_reader;
pub mod metadata;
mod npy_reader;
mod sparse_reader;

pub use compound_reader::{
    read_compound_data, read_compound_queries, read_compound_vectors_only, read_payloads_jsonl,
};
pub use hdf5_reader::{hdf5_train_row_count, read_hdf5_vectors};
pub use jsonl_reader::{read_jsonl_queries, read_jsonl_vectors};
pub use metadata::{parse_metadata_from_json, MetadataItem, MetadataValue};
pub use npy_reader::{npy_row_count, read_npy_vectors, write_npy_vectors};
pub use sparse_reader::{
    csr_row_count, read_gt_neighbours, read_sparse_matrix, write_gt_neighbours,
    write_sparse_matrix, SparseVector,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;
    use std::path::PathBuf;

    /// Get the project root directory (assumes tests run from project root)
    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf()
    }

    /// Get the datasets directory (user's site-packages or local datasets/)
    fn datasets_dir() -> PathBuf {
        let home = env::var("HOME").unwrap_or_default();
        let site_packages =
            PathBuf::from(&home).join(".local/lib/python3.12/site-packages/datasets");
        if site_packages.exists() {
            return site_packages;
        }
        project_root().join("datasets")
    }

    // ===========================================
    // JSONL Reader Unit Tests (no external files)
    // ===========================================

    #[test]
    fn test_jsonl_reader_basic() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmpfile, "[1.0, 2.0, 3.0]").unwrap();
        writeln!(tmpfile, "[4.0, 5.0, 6.0]").unwrap();
        writeln!(tmpfile, "[7.0, 8.0, 9.0]").unwrap();

        let (ids, vectors) = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), false).unwrap();

        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0], vec![1.0f32, 2.0, 3.0]);
        assert_eq!(vectors[1], vec![4.0f32, 5.0, 6.0]);
        assert_eq!(vectors[2], vec![7.0f32, 8.0, 9.0]);
    }

    #[test]
    fn test_jsonl_reader_single_vector() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmpfile, "[42.0]").unwrap();

        let (ids, vectors) = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), false).unwrap();

        assert_eq!(ids, vec![0]);
        assert_eq!(vectors, vec![vec![42.0f32]]);
    }

    #[test]
    fn test_jsonl_reader_skips_empty_lines() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmpfile, "[1.0, 2.0]").unwrap();
        writeln!(tmpfile).unwrap();
        writeln!(tmpfile, "  ").unwrap();
        writeln!(tmpfile, "[3.0, 4.0]").unwrap();

        let (ids, vectors) = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), false).unwrap();

        assert_eq!(ids, vec![0, 1]);
        assert_eq!(vectors.len(), 2);
    }

    #[test]
    fn test_jsonl_reader_with_normalization() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        // Vector [3.0, 4.0] has L2 norm = 5.0
        writeln!(tmpfile, "[3.0, 4.0]").unwrap();

        let (_, vectors) = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), true).unwrap();

        let norm: f32 = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Normalized vector should have L2 norm = 1.0, got {}",
            norm
        );
        assert!((vectors[0][0] - 0.6).abs() < 1e-6); // 3/5
        assert!((vectors[0][1] - 0.8).abs() < 1e-6); // 4/5
    }

    #[test]
    fn test_jsonl_reader_normalization_zero_vector() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmpfile, "[0.0, 0.0, 0.0]").unwrap();

        let (_, vectors) = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), true).unwrap();

        // Zero vector should remain zero (no divide-by-zero)
        assert_eq!(vectors[0], vec![0.0f32, 0.0, 0.0]);
    }

    #[test]
    fn test_jsonl_reader_normalization_already_unit() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        // Unit vector along x-axis
        writeln!(tmpfile, "[1.0, 0.0, 0.0]").unwrap();

        let (_, vectors) = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), true).unwrap();

        let norm: f32 = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((vectors[0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jsonl_reader_f64_to_f32_conversion() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        // JSON numbers are parsed as f64, then converted to f32
        writeln!(tmpfile, "[1.5, 2.5, 3.5]").unwrap();

        let (_, vectors) = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), false).unwrap();

        assert_eq!(vectors[0], vec![1.5f32, 2.5, 3.5]);
    }

    #[test]
    fn test_jsonl_reader_invalid_json() {
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmpfile, "not json").unwrap();

        let result = read_jsonl_vectors(tmpfile.path().to_str().unwrap(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_jsonl_reader_nonexistent_file() {
        let result = read_jsonl_vectors("/nonexistent/path/vectors.jsonl", false);
        assert!(result.is_err());
    }

    // ===========================================
    // Metadata Parsing Unit Tests
    // ===========================================

    #[test]
    fn test_metadata_parse_string_field() {
        let json: serde_json::Value = serde_json::json!({"name": "test_item"});
        let meta = parse_metadata_from_json(json).unwrap();

        assert_eq!(meta.fields.len(), 1);
        assert_eq!(meta.fields[0].0, "name");
        match &meta.fields[0].1 {
            MetadataValue::String(s) => assert_eq!(s, "test_item"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_metadata_parse_number_field() {
        let json: serde_json::Value = serde_json::json!({"price": 42});
        let meta = parse_metadata_from_json(json).unwrap();

        assert_eq!(meta.fields.len(), 1);
        match &meta.fields[0].1 {
            MetadataValue::Int(n) => assert_eq!(*n, 42),
            other => panic!("Expected Int variant for integer, got {:?}", other),
        }
    }

    #[test]
    fn test_metadata_parse_float_field() {
        let json: serde_json::Value = serde_json::json!({"price": 3.5});
        let meta = parse_metadata_from_json(json).unwrap();

        assert_eq!(meta.fields.len(), 1);
        match &meta.fields[0].1 {
            MetadataValue::Float(f) => assert_eq!(*f, 3.5),
            other => panic!("Expected Float variant for non-integer, got {:?}", other),
        }
    }

    #[test]
    fn test_metadata_parse_bool_field() {
        let json: serde_json::Value = serde_json::json!({"active": true});
        let meta = parse_metadata_from_json(json).unwrap();

        assert_eq!(meta.fields.len(), 1);
        match &meta.fields[0].1 {
            MetadataValue::String(s) => assert_eq!(s, "true"),
            _ => panic!("Expected String variant for bool"),
        }
    }

    #[test]
    fn test_metadata_parse_labels_field() {
        let json: serde_json::Value = serde_json::json!({"labels": ["red", "blue", "green"]});
        let meta = parse_metadata_from_json(json).unwrap();

        assert_eq!(meta.fields.len(), 1);
        match &meta.fields[0].1 {
            MetadataValue::Labels(labels) => {
                assert_eq!(labels, &vec!["red", "blue", "green"]);
            }
            _ => panic!("Expected Labels variant"),
        }
    }

    #[test]
    fn test_metadata_parse_geo_field() {
        let json: serde_json::Value =
            serde_json::json!({"location": {"lon": -73.9857, "lat": 40.7484}});
        let meta = parse_metadata_from_json(json).unwrap();

        assert_eq!(meta.fields.len(), 1);
        match &meta.fields[0].1 {
            MetadataValue::Geo { lon, lat } => {
                assert!((*lon - (-73.9857)).abs() < 1e-4);
                assert!((*lat - 40.7484).abs() < 1e-4);
            }
            _ => panic!("Expected Geo variant"),
        }
    }

    #[test]
    fn test_metadata_parse_null_field_ignored() {
        let json: serde_json::Value = serde_json::json!({"name": "test", "empty": null});
        let meta = parse_metadata_from_json(json).unwrap();

        // null fields are skipped
        assert_eq!(meta.fields.len(), 1);
        assert_eq!(meta.fields[0].0, "name");
    }

    #[test]
    fn test_metadata_parse_empty_object() {
        let json: serde_json::Value = serde_json::json!({});
        let meta = parse_metadata_from_json(json).unwrap();
        assert_eq!(meta.fields.len(), 0);
    }

    #[test]
    fn test_metadata_parse_non_object_returns_none() {
        let json: serde_json::Value = serde_json::json!([1, 2, 3]);
        assert!(parse_metadata_from_json(json).is_none());

        let json: serde_json::Value = serde_json::json!("a string");
        assert!(parse_metadata_from_json(json).is_none());

        let json: serde_json::Value = serde_json::json!(42);
        assert!(parse_metadata_from_json(json).is_none());
    }

    #[test]
    fn test_metadata_parse_multiple_field_types() {
        let json: serde_json::Value = serde_json::json!({
            "title": "Widget",
            "price": 9.99,
            "labels": ["sale", "new"],
            "location": {"lon": 1.0, "lat": 2.0},
            "active": false,
            "notes": null
        });
        let meta = parse_metadata_from_json(json).unwrap();

        // 5 fields (null is skipped)
        assert_eq!(meta.fields.len(), 5);

        let field_names: Vec<&str> = meta.fields.iter().map(|(k, _)| k.as_str()).collect();
        assert!(field_names.contains(&"title"));
        assert!(field_names.contains(&"price"));
        assert!(field_names.contains(&"labels"));
        assert!(field_names.contains(&"location"));
        assert!(field_names.contains(&"active"));
    }

    // ===========================================
    // Compound Reader Unit Tests (with temp files)
    // ===========================================

    #[test]
    fn test_compound_reader_payloads_parsing() {
        let dir = tempfile::tempdir().unwrap();
        let payloads_path = dir.path().join("payloads.jsonl");

        let mut f = std::fs::File::create(&payloads_path).unwrap();
        writeln!(f, r#"{{"name": "item1", "price": 10}}"#).unwrap();
        writeln!(f, r#"{{"name": "item2", "labels": ["a", "b"]}}"#).unwrap();
        writeln!(f, r#"{{}}"#).unwrap();

        let metadata =
            compound_reader::read_payloads_jsonl(payloads_path.to_str().unwrap()).unwrap();

        assert_eq!(metadata.len(), 3);
        assert!(metadata[0].is_some());
        assert!(metadata[1].is_some());
        assert!(metadata[2].is_some()); // empty object still produces Some with 0 fields

        let m0 = metadata[0].as_ref().unwrap();
        assert_eq!(m0.fields.len(), 2);

        let m1 = metadata[1].as_ref().unwrap();
        let labels_field = m1.fields.iter().find(|(k, _)| k == "labels").unwrap();
        match &labels_field.1 {
            MetadataValue::Labels(l) => assert_eq!(l, &vec!["a", "b"]),
            _ => panic!("Expected Labels"),
        }

        let m2 = metadata[2].as_ref().unwrap();
        assert_eq!(m2.fields.len(), 0);
    }

    #[test]
    fn test_compound_reader_payloads_empty_lines() {
        let dir = tempfile::tempdir().unwrap();
        let payloads_path = dir.path().join("payloads.jsonl");

        let mut f = std::fs::File::create(&payloads_path).unwrap();
        writeln!(f, r#"{{"name": "item1"}}"#).unwrap();
        writeln!(f).unwrap(); // empty line => None
        writeln!(f, r#"{{"name": "item3"}}"#).unwrap();

        let metadata =
            compound_reader::read_payloads_jsonl(payloads_path.to_str().unwrap()).unwrap();

        assert_eq!(metadata.len(), 3);
        assert!(metadata[0].is_some());
        assert!(metadata[1].is_none()); // empty line
        assert!(metadata[2].is_some());
    }

    // ===========================================
    // JSONL Query Reader Unit Tests
    // ===========================================

    #[test]
    fn test_jsonl_queries_basic() {
        let dir = tempfile::tempdir().unwrap();

        // Create queries.jsonl
        let mut qf = std::fs::File::create(dir.path().join("queries.jsonl")).unwrap();
        writeln!(qf, "[1.0, 2.0, 3.0]").unwrap();
        writeln!(qf, "[4.0, 5.0, 6.0]").unwrap();

        // Create neighbours.jsonl
        let mut nf = std::fs::File::create(dir.path().join("neighbours.jsonl")).unwrap();
        writeln!(nf, "[10, 20, 30]").unwrap();
        writeln!(nf, "[40, 50, 60]").unwrap();

        let (queries, neighbors) = read_jsonl_queries(dir.path().to_str().unwrap(), false).unwrap();

        assert_eq!(queries.len(), 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(queries[0], vec![1.0f32, 2.0, 3.0]);
        assert_eq!(queries[1], vec![4.0f32, 5.0, 6.0]);
        assert_eq!(neighbors[0], vec![10i64, 20, 30]);
        assert_eq!(neighbors[1], vec![40i64, 50, 60]);
    }

    #[test]
    fn test_jsonl_queries_without_neighbours() {
        let dir = tempfile::tempdir().unwrap();

        // Only queries, no neighbours file
        let mut qf = std::fs::File::create(dir.path().join("queries.jsonl")).unwrap();
        writeln!(qf, "[1.0, 0.0]").unwrap();
        writeln!(qf, "[0.0, 1.0]").unwrap();

        let (queries, neighbors) = read_jsonl_queries(dir.path().to_str().unwrap(), false).unwrap();

        assert_eq!(queries.len(), 2);
        assert_eq!(neighbors.len(), 2);
        // Without neighbours file, should get empty vecs
        assert!(neighbors[0].is_empty());
        assert!(neighbors[1].is_empty());
    }

    #[test]
    fn test_jsonl_queries_with_normalization() {
        let dir = tempfile::tempdir().unwrap();

        let mut qf = std::fs::File::create(dir.path().join("queries.jsonl")).unwrap();
        writeln!(qf, "[3.0, 4.0]").unwrap(); // norm = 5.0

        let mut nf = std::fs::File::create(dir.path().join("neighbours.jsonl")).unwrap();
        writeln!(nf, "[0]").unwrap();

        let (queries, _) = read_jsonl_queries(dir.path().to_str().unwrap(), true).unwrap();

        let norm: f32 = queries[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((queries[0][0] - 0.6).abs() < 1e-6);
        assert!((queries[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_jsonl_queries_skips_empty_lines() {
        let dir = tempfile::tempdir().unwrap();

        let mut qf = std::fs::File::create(dir.path().join("queries.jsonl")).unwrap();
        writeln!(qf, "[1.0, 2.0]").unwrap();
        writeln!(qf).unwrap();
        writeln!(qf, "[3.0, 4.0]").unwrap();

        let mut nf = std::fs::File::create(dir.path().join("neighbours.jsonl")).unwrap();
        writeln!(nf, "[0, 1]").unwrap();
        writeln!(nf).unwrap();
        writeln!(nf, "[2, 3]").unwrap();

        let (queries, neighbors) = read_jsonl_queries(dir.path().to_str().unwrap(), false).unwrap();

        assert_eq!(queries.len(), 2);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_jsonl_queries_missing_dir() {
        let result = read_jsonl_queries("/nonexistent/dir", false);
        assert!(result.is_err());
    }

    // ===========================================
    // Compound Query Reader Unit Tests
    // ===========================================

    #[test]
    fn test_compound_queries_basic() {
        let dir = tempfile::tempdir().unwrap();

        let mut f = std::fs::File::create(dir.path().join("tests.jsonl")).unwrap();
        writeln!(
            f,
            r#"{{"query": [1.0, 2.0, 3.0], "closest_ids": [10, 20], "closest_scores": [0.9, 0.8]}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"query": [4.0, 5.0, 6.0], "closest_ids": [30, 40], "closest_scores": [0.7, 0.6]}}"#
        )
        .unwrap();

        let (queries, neighbors, _conditions) =
            compound_reader::read_compound_queries_raw(dir.path().to_str().unwrap(), false)
                .unwrap();

        assert_eq!(queries.len(), 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(queries[0], vec![1.0f32, 2.0, 3.0]);
        assert_eq!(queries[1], vec![4.0f32, 5.0, 6.0]);
        assert_eq!(neighbors[0], vec![10i64, 20]);
        assert_eq!(neighbors[1], vec![30i64, 40]);
    }

    /// The `Some(Value::Null)` trap, on the checked-in fixture, at the layer
    /// that can still see raw JSON.
    ///
    /// `compound_reader` builds the vector with `row.get("conditions").cloned()`,
    /// so a row spelling `"conditions": null` arrives as `Some(Value::Null)` —
    /// NOT `None`, which is what a guard keyed on `is_some()` would need it to
    /// be, and which is why every query of the shipped `*_no_filters` tarballs
    /// would have failed. `"conditions": {}` arrives as a third, distinct value.
    /// All three mean "no filter declared" to `query_filter`, but they are not
    /// the same value, and this is the only place that can prove it.
    #[test]
    fn fixture_null_and_empty_conditions_are_distinct_values() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/shipped_conditions");
        let (_q, _n, raw) =
            compound_reader::read_compound_queries_raw(dir.to_str().unwrap(), false).unwrap();
        assert_eq!(raw.len(), 21);

        let nulls = raw
            .iter()
            .filter(|c| **c == Some(serde_json::Value::Null))
            .count();
        let empties = raw
            .iter()
            .filter(|c| **c == Some(serde_json::json!({})))
            .count();
        let absent = raw.iter().filter(|c| c.is_none()).count();
        assert_eq!(
            nulls, 3,
            "`\"conditions\": null` must survive as Some(Null)"
        );
        assert_eq!(
            empties, 1,
            "`\"conditions\": {{}}` must survive as Some(Object)"
        );
        assert_eq!(
            absent, 0,
            "no fixture row omits the key, so none may read back as None"
        );
        assert_ne!(
            Some(serde_json::Value::Null),
            Some(serde_json::json!({})),
            "null and {{}} are different values even though both mean 'no filter'"
        );

        // ...and all four resolve to an unfiltered query rather than an error.
        let (_q, _n, guarded) = read_compound_queries(dir.to_str().unwrap(), false).unwrap();
        assert_eq!(guarded.declared_count(), 17);
    }

    #[test]
    fn test_compound_queries_with_conditions() {
        let dir = tempfile::tempdir().unwrap();

        let mut f = std::fs::File::create(dir.path().join("tests.jsonl")).unwrap();
        // Line with conditions field
        writeln!(
            f,
            r#"{{"query": [1.0, 0.0], "conditions": {{"and": [{{"color": {{"match": {{"value": "red"}}}}}}]}}, "closest_ids": [5]}}"#
        )
        .unwrap();

        let (queries, neighbors, conditions) =
            read_compound_queries(dir.path().to_str().unwrap(), false).unwrap();

        assert_eq!(queries.len(), 1);
        assert_eq!(queries[0], vec![1.0f32, 0.0]);
        assert_eq!(neighbors[0], vec![5i64]);
        assert_eq!(
            conditions.declared_count(),
            1,
            "the row declares a filter, so it must survive the guarded reader"
        );
    }

    #[test]
    fn test_compound_queries_with_normalization() {
        let dir = tempfile::tempdir().unwrap();

        let mut f = std::fs::File::create(dir.path().join("tests.jsonl")).unwrap();
        writeln!(f, r#"{{"query": [3.0, 4.0], "closest_ids": [0]}}"#).unwrap();

        let (queries, _, _) =
            compound_reader::read_compound_queries_raw(dir.path().to_str().unwrap(), true).unwrap();

        let norm: f32 = queries[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compound_queries_missing_closest_ids() {
        let dir = tempfile::tempdir().unwrap();

        let mut f = std::fs::File::create(dir.path().join("tests.jsonl")).unwrap();
        // No closest_ids field - should default to empty
        writeln!(f, r#"{{"query": [1.0, 2.0]}}"#).unwrap();

        let (queries, neighbors, _conditions) =
            compound_reader::read_compound_queries_raw(dir.path().to_str().unwrap(), false)
                .unwrap();

        assert_eq!(queries.len(), 1);
        assert!(neighbors[0].is_empty());
    }

    #[test]
    fn test_compound_queries_skips_empty_lines() {
        let dir = tempfile::tempdir().unwrap();

        let mut f = std::fs::File::create(dir.path().join("tests.jsonl")).unwrap();
        writeln!(f, r#"{{"query": [1.0], "closest_ids": [0]}}"#).unwrap();
        writeln!(f).unwrap();
        writeln!(f, r#"{{"query": [2.0], "closest_ids": [1]}}"#).unwrap();

        let (queries, neighbors, _conditions) =
            compound_reader::read_compound_queries_raw(dir.path().to_str().unwrap(), false)
                .unwrap();

        assert_eq!(queries.len(), 2);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_compound_queries_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        // No tests.jsonl created
        let result =
            compound_reader::read_compound_queries_raw(dir.path().to_str().unwrap(), false);
        assert!(result.is_err());
    }

    // ===========================================
    // Dataset File Tests (require real datasets on disk)
    // ===========================================

    #[test]
    fn test_hdf5_reader_glove25() {
        let path = project_root().join("datasets/glove-25-angular/glove-25-angular.hdf5");
        if !path.exists() {
            eprintln!(
                "Skipping test_hdf5_reader_glove25: dataset not found at {:?}",
                path
            );
            return;
        }

        let (ids, vectors) =
            read_hdf5_vectors(path.to_str().unwrap(), false).expect("Failed to read HDF5 file");

        assert_eq!(vectors.len(), 1183514, "Expected 1,183,514 vectors");
        assert_eq!(ids.len(), vectors.len());
        assert_eq!(vectors[0].len(), 25, "Expected 25 dimensions per vector");
        assert_eq!(ids[0], 0);
        assert_eq!(ids[ids.len() - 1], (ids.len() - 1) as i64);
    }

    #[test]
    fn test_hdf5_reader_with_normalization_dataset() {
        let path = project_root().join("datasets/glove-25-angular/glove-25-angular.hdf5");
        if !path.exists() {
            eprintln!("Skipping: dataset not found");
            return;
        }

        let (_, vectors) = read_hdf5_vectors(path.to_str().unwrap(), true).unwrap();

        let norm: f32 = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jsonl_reader_random100k() {
        let path = project_root().join("datasets/random-100k/vectors.jsonl");
        if !path.exists() {
            eprintln!("Skipping: dataset not found at {:?}", path);
            return;
        }

        let (ids, vectors) = read_jsonl_vectors(path.to_str().unwrap(), false).unwrap();

        assert_eq!(vectors.len(), 100000);
        assert_eq!(ids.len(), vectors.len());
        assert_eq!(vectors[0].len(), 100);
        assert_eq!(ids[0], 0);
    }

    #[test]
    fn test_compound_reader_hnm() {
        let path = datasets_dir().join("h-and-m-2048-angular/hnm");
        if !path.exists() {
            eprintln!("Skipping: dataset not found at {:?}", path);
            return;
        }

        let (ids, vectors, metadata) = read_compound_data(path.to_str().unwrap(), false).unwrap();

        assert!(vectors.len() > 100000 && vectors.len() <= 110000);
        assert_eq!(ids.len(), vectors.len());
        assert_eq!(metadata.len(), vectors.len());
        assert_eq!(vectors[0].len(), 2048);

        let metadata_count = metadata.iter().filter(|m| m.is_some()).count();
        assert!(metadata_count > 0);
    }

    #[test]
    fn test_compound_reader_vectors_only() {
        let path = datasets_dir().join("h-and-m-2048-angular/hnm");
        if !path.exists() {
            eprintln!("Skipping: dataset not found");
            return;
        }

        let (ids, vectors) = read_compound_vectors_only(path.to_str().unwrap(), false).unwrap();

        assert!(vectors.len() > 100000);
        assert_eq!(ids.len(), vectors.len());
        assert_eq!(vectors[0].len(), 2048);
    }

    #[test]
    fn test_compound_reader_queries_hnm() {
        let path = datasets_dir().join("h-and-m-2048-angular/hnm");
        if !path.exists() {
            eprintln!("Skipping: dataset not found at {:?}", path);
            return;
        }

        let (queries, neighbors, _conditions) =
            compound_reader::read_compound_queries_raw(path.to_str().unwrap(), false).unwrap();

        assert!(
            !queries.is_empty(),
            "Expected at least one query in tests.jsonl"
        );
        assert_eq!(queries.len(), neighbors.len());
        assert_eq!(
            queries[0].len(),
            2048,
            "Query dimension should match dataset"
        );
        assert!(
            !neighbors[0].is_empty(),
            "Expected ground truth neighbor IDs"
        );
    }

    #[test]
    fn test_jsonl_reader_queries_random100() {
        let path = project_root().join("datasets/random-100");
        if !path.exists() {
            eprintln!("Skipping: dataset not found at {:?}", path);
            return;
        }

        let queries_file = path.join("queries.jsonl");
        if !queries_file.exists() {
            eprintln!("Skipping: queries.jsonl not found at {:?}", queries_file);
            return;
        }

        let (queries, neighbors) = read_jsonl_queries(path.to_str().unwrap(), false).unwrap();

        assert!(!queries.is_empty(), "Expected at least one query");
        assert_eq!(queries.len(), neighbors.len());
    }
}
