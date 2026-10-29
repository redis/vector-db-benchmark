//! Peak client memory of the #342 streaming paths, measured with a counting
//! global allocator. No container; runs in `make agent-check-test`.
//!
//! The claim under test is "memory does not scale with the corpus": the same
//! code is run over a small and a 10x larger corpus and the peak allocation
//! above the baseline must not grow with it. A control loads the large corpus
//! with the in-memory reader and must see the whole corpus, so a meter that
//! silently stopped counting cannot pass the real assertions.
//!
//! One allocator is shared by every test in this binary, so each test holds
//! `SERIAL` while it measures.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Mutex;

use vector_db_benchmark::msmarco::hybrid;
use vector_db_benchmark::readers::{read_compound_data, CompoundStream};
use vector_db_benchmark::upload_stream::{memory_bound_bytes, upload_stream};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = CURRENT.fetch_add(by, SeqCst) + by;
    PEAK.fetch_max(now, SeqCst);
}

// SAFETY: every call forwards to the system allocator unchanged; the
// counters only observe sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), SeqCst);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, SeqCst);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static SERIAL: Mutex<()> = Mutex::new(());

/// Run `f` and return its result with the peak bytes allocated above the
/// level at entry.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = CURRENT.load(SeqCst);
    PEAK.store(base, SeqCst);
    let out = f();
    (out, PEAK.load(SeqCst).saturating_sub(base))
}

const DIM: usize = 256;

/// A compound corpus of `rows` x `DIM` float32 vectors and ~300-byte payloads,
/// written row by row so building it never holds it in memory either.
fn write_corpus(dir: &Path, rows: usize) -> u64 {
    let header = vector_db_benchmark::msmarco::npy_f32_header(rows as u64, DIM);
    let mut v = std::io::BufWriter::new(std::fs::File::create(dir.join("vectors.npy")).unwrap());
    v.write_all(&header).unwrap();
    let mut p = std::io::BufWriter::new(std::fs::File::create(dir.join("payloads.jsonl")).unwrap());
    for r in 0..rows {
        for c in 0..DIM {
            v.write_all(&(((r * 7 + c) % 101) as f32 - 50.0).to_le_bytes())
                .unwrap();
        }
        writeln!(
            p,
            "{{\"docid\":\"doc-{r}\",\"title\":\"{}\",\"n\":{r},\"tag\":\"t{}\"}}",
            "lorem ipsum dolor sit amet ".repeat(10),
            r % 13
        )
        .unwrap();
    }
    v.flush().unwrap();
    p.flush().unwrap();
    (rows * DIM * 4) as u64
}

/// Stream the corpus through the upload driver with an "engine" that encodes
/// every batch as JSON (the most expensive wire format of the two engines) and
/// drops it. Returns the peak bytes.
fn stream_peak(dir: &Path, parallel: usize, batch: usize) -> usize {
    let stream = CompoundStream::open(dir, true).unwrap();
    let rows = stream.rows();
    let (stats, peak) = peak_during(|| {
        upload_stream(
            stream,
            parallel,
            batch,
            || Ok(()),
            |_, b| {
                let body = serde_json::to_vec(&b.vectors).map_err(|e| e.to_string())?;
                std::hint::black_box(&body);
                Ok(())
            },
            &|_| {},
        )
        .unwrap()
    });
    assert_eq!(stats.rows, rows);
    peak
}

/// #342: streaming a corpus 10x larger must not need more memory. The bound is
/// the documented `parallel x batch_size x per_row` (`memory_bound_bytes`),
/// and both corpora stay under it; the 20K one stays under 1/8 of its own
/// vector bytes.
#[test]
fn streaming_upload_peak_does_not_grow_with_the_corpus() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let (small_dir, large_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    write_corpus(small_dir.path(), 2_000);
    let large_bytes = write_corpus(large_dir.path(), 20_000);
    let (parallel, batch) = (4, 64);

    let small = stream_peak(small_dir.path(), parallel, batch);
    let large = stream_peak(large_dir.path(), parallel, batch);
    let payload = std::fs::metadata(large_dir.path().join("payloads.jsonl"))
        .unwrap()
        .len()
        / 20_000;
    let bound = memory_bound_bytes(parallel, batch, DIM, payload) as usize;
    println!(
        "streaming peak: 2K rows {small} B, 20K rows {large} B; bound {bound} B; 20K corpus \
         {large_bytes} B of vectors"
    );
    assert!(
        large <= small + small / 4 + (256 << 10),
        "streaming peak grew with the corpus: {small} B at 2K rows, {large} B at 20K"
    );
    assert!(
        large <= bound,
        "peak {large} B above the documented bound {bound} B"
    );
    assert!(
        (large as u64) < large_bytes / 8,
        "peak {large} B is not small next to the {large_bytes}-byte corpus"
    );

    // Control: the in-memory reader on the same corpus must register at least
    // the vectors themselves, or this meter is not measuring anything.
    let path = large_dir.path().to_str().unwrap().to_string();
    let (_, loaded) = peak_during(|| read_compound_data(&path, true).unwrap());
    assert!(
        loaded as u64 >= large_bytes,
        "the in-memory control registered {loaded} B for {large_bytes} B of vectors"
    );
}

/// #342: the hybrid query set's preparation streams the payloads and keeps 12
/// bits per row. Both corpora are larger than one scan chunk
/// (`hybrid::SCAN_CHUNK_ROWS`), so the fixed working set is the same for both
/// and only per-row state can differ: growing the corpus by 3x may cost at
/// most 8 bytes per added row. Holding every title and a byte per row per
/// predicate, as the in-memory corpus does, costs ~80.
#[test]
fn hybrid_preparation_memory_is_bits_per_row_not_the_corpus() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let peak = |n: usize| {
        let dir = tempfile::tempdir().unwrap();
        hybrid::write_fixture_corpus(dir.path(), n, 8, 48).unwrap();
        let inputs = hybrid::HybridInputs::in_dir(dir.path());
        let (prepared, peak) = peak_during(|| hybrid::prepare(&inputs, Some(n as u64)).unwrap());
        assert_eq!(prepared.rows.len(), 48);
        peak
    };
    let (n_small, n_large) = (hybrid::SCAN_CHUNK_ROWS + 4_000, 3 * hybrid::SCAN_CHUNK_ROWS);
    let (small, large) = (peak(n_small), peak(n_large));
    let per_row = large.saturating_sub(small) as f64 / (n_large - n_small) as f64;
    println!(
        "hybrid prepare peak: {n_small} rows {small} B, {n_large} rows {large} B \
         ({per_row:.2} B/row)"
    );
    assert!(
        per_row <= 8.0,
        "hybrid preparation grows by {per_row:.1} B per corpus row ({small} B at {n_small}, \
         {large} B at {n_large}); it must stream the payloads, not hold them"
    );
}
