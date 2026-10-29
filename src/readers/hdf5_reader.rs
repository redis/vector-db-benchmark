//! HDF5 file reader for vector datasets.

use hdf5::File as Hdf5File;

/// Number of rows in the HDF5 file's `train` dataset, read from the dataspace
/// metadata WITHOUT loading any vector data. The corpus's own answer to "how
/// many vectors are in here?" (#224).
pub fn hdf5_train_row_count(path: &str) -> Result<u64, String> {
    let file = Hdf5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;
    let train = file
        .dataset("train")
        .map_err(|e| format!("Failed to open 'train' dataset: {}", e))?;
    train
        .shape()
        .first()
        .map(|&n| n as u64)
        .ok_or_else(|| format!("{}: HDF5 'train' dataset has no dimensions", path))
}

/// Read vectors from HDF5 file, returning (ids, vectors).
/// If normalize is true, each vector is divided by its L2 norm.
pub fn read_hdf5_vectors(path: &str, normalize: bool) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
    let file = Hdf5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;
    let train = file
        .dataset("train")
        .map_err(|e| format!("Failed to open 'train' dataset: {}", e))?;

    let shape = train.shape();
    let count = shape[0];
    let dim = shape[1];

    // Read as flat array
    let flat: Vec<f32> = train
        .read_raw()
        .map_err(|e| format!("Failed to read dataset: {}", e))?;

    // Convert to Vec<Vec<f32>> and optionally normalize
    let mut vectors: Vec<Vec<f32>> = flat.chunks(dim).map(|chunk| chunk.to_vec()).collect();

    if normalize {
        for vec in &mut vectors {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    // Generate sequential IDs (matching Python behavior)
    let ids: Vec<i64> = (0..count as i64).collect();

    Ok((ids, vectors))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips an ann-benchmarks-shaped file through the HDF5 library this
    /// build links — including the DEFLATE filter. The prebuilt release archives
    /// (#366, `--features release-static`) compile HDF5 and zlib from source
    /// instead of using the system library, and the release workflow runs this
    /// test against that build: a static HDF5 missing zlib would open plain
    /// files and fail only on a compressed corpus, in the middle of a run.
    #[test]
    fn reads_a_deflate_compressed_train_dataset() {
        assert!(
            hdf5::filters::deflate_available(),
            "the linked HDF5 has no DEFLATE (zlib) filter"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.hdf5");
        // Row-major 3 x 2.
        let train = [3.0f32, 4.0, 1.0, 0.0, 0.0, 2.0];
        {
            let f = Hdf5File::create(&path).unwrap();
            let ds = f
                .new_dataset::<f32>()
                .deflate(6)
                .shape((3, 2))
                .create("train")
                .unwrap();
            ds.write_raw(&train).unwrap();
            assert!(
                ds.filters()
                    .iter()
                    .any(|f| matches!(f, hdf5::filters::Filter::Deflate(_))),
                "dataset was written without the DEFLATE filter: {:?}",
                ds.filters()
            );
        }
        let p = path.to_str().unwrap();
        assert_eq!(hdf5_train_row_count(p).unwrap(), 3);
        let (ids, vectors) = read_hdf5_vectors(p, true).unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(
            vectors,
            vec![vec![0.6, 0.8], vec![1.0, 0.0], vec![0.0, 1.0]]
        );
    }
}
