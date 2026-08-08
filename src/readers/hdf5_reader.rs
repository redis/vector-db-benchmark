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
