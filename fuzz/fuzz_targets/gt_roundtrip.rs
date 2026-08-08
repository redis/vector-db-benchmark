#![no_main]
//! Structured / differential round-trip fuzzing of the `results.gt` codec:
//! `read_gt_neighbours(write_gt_neighbours(x)) == x`.
//!
//! This catches asymmetries a byte-fuzzer cannot: the writer stores ids as int32
//! while the reader sign-extends to i64, so an id outside i32 range must be
//! REJECTED by the writer rather than silently truncated into a different id.
//! Every row carries the same neighbour count because the format stores a single
//! `d`.

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use vector_db_benchmark::readers::{read_gt_neighbours, write_gt_neighbours};

/// Bounded generator: 0..=32 rows of a single shared width 0..=16. Ids span the
/// full i64 range on purpose, so out-of-i32 values are exercised.
fn gen_rows(u: &mut Unstructured) -> arbitrary::Result<Vec<Vec<i64>>> {
    let n_rows = u8::arbitrary(u)? as usize % 33;
    let d = u8::arbitrary(u)? as usize % 17;
    let mut rows = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let mut row = Vec::with_capacity(d);
        for _ in 0..d {
            row.push(i64::arbitrary(u)?);
        }
        rows.push(row);
    }
    Ok(rows)
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let rows = match gen_rows(&mut u) {
        Ok(r) => r,
        Err(_) => return,
    };

    let tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    let path = match tmp.path().to_str() {
        Some(p) => p,
        None => return,
    };

    // The writer refuses anything the format cannot represent faithfully: an id
    // outside i32 (which `as i32` would truncate into a DIFFERENT id) and the
    // `n > 0, d == 0` shape (an 8-byte file the reader rejects). Those are the
    // asymmetries this target exists to pin, so a refusal is a pass, and
    // everything the writer DOES accept must read back byte-for-byte.
    if write_gt_neighbours(path, &rows).is_err() {
        return;
    }

    let read = read_gt_neighbours(path).expect("valid results.gt written by us must read back");
    assert_eq!(read, rows, "results.gt round-trip mismatch");
});
