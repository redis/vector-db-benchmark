#![no_main]
//! Fuzz the binary `results.gt` ground-truth reader.
//!
//! The reader ingests bytes straight from a downloaded dataset archive and sizes
//! its allocation from the header (`n`, `d`). We only care that it never
//! panics/overflows/OOMs: any malformed input must return `Err`. Note `chunks(d)`
//! panics on `d == 0`, so the `n > 0, d == 0` header must be rejected before it.

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use vector_db_benchmark::readers::read_gt_neighbours;

fuzz_target!(|data: &[u8]| {
    let mut tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    if tmp.write_all(data).is_err() {
        return;
    }
    if tmp.flush().is_err() {
        return;
    }
    let path = match tmp.path().to_str() {
        Some(p) => p,
        None => return,
    };
    // We only assert it doesn't crash; Err is fine.
    let _ = read_gt_neighbours(path);
});
