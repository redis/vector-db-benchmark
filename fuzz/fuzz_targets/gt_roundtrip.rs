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

    // An id that does not fit the int32 on-disk field must be refused, not
    // written lossily — that is the asymmetry this target exists to pin.
    if write_gt_neighbours(path, &rows).is_err() {
        return;
    }

    let read = read_gt_neighbours(path).expect("valid results.gt written by us must read back");
    // A zero-width row set writes n rows of 0 ids, which the reader reports as
    // "no queries" (and rejects outright when n > 0) — so only compare when
    // there is something to compare.
    if rows.first().is_some_and(|r| !r.is_empty()) {
        assert_eq!(read, rows, "results.gt round-trip mismatch");
    }
});
