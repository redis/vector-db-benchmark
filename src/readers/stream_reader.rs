//! Streaming reader for the compound (`type: "tar"`) layout: `vectors.npy` +
//! `payloads.jsonl`, read a block of rows at a time (#342).
//!
//! [`read_compound_data`](super::read_compound_data) materialises the whole
//! corpus as `Vec<Vec<f32>>` plus one parsed payload per row. At 100M x 1024
//! that is ~410 GB of vectors on the CLIENT before an engine sees a byte. This
//! reader holds at most one block: the NPY rows are read sequentially from the
//! data offset and zipped with a line iterator over `payloads.jsonl`, so its
//! memory is `block rows x (row bytes + payload line)` plus two fixed read
//! buffers, whatever the corpus size.
//!
//! It yields EXACTLY what [`read_compound_data`](super::read_compound_data)
//! does, row for row (ids are row indices, the same normalization, blank
//! payload lines are missing metadata, a payload count that differs from the
//! vector count is an error). `compound_stream_equals_the_in_memory_reader`
//! pins that, so an engine can switch readers without changing what it
//! uploads.
//!
//! Reading is split in two so an uploader can parallelise the expensive half:
//! [`CompoundStream::next_raw`] does the sequential I/O (raw little-endian
//! bytes and the raw payload lines) and must be called under whatever lock
//! serialises the stream; [`RawBatch::decode`] converts and parses, needs no
//! shared state, and runs on the worker that will upload the batch.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use super::metadata::{parse_metadata_from_json, MetadataItem};

/// Read buffer for each of the two files. Fixed, so it is a constant term of
/// the memory bound, not a per-row one.
pub const STREAM_READ_BUFFER: usize = 1 << 20;

/// A C-order little-endian `float32` 2-D NPY, positioned at its first row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32NpyLayout {
    pub rows: u64,
    pub cols: usize,
    /// Byte offset of row 0.
    pub data_start: u64,
}

impl F32NpyLayout {
    /// Bytes of the data payload (`rows x cols x 4`).
    pub fn data_bytes(&self) -> u64 {
        self.rows * self.cols as u64 * 4
    }
}

/// Parse a `vectors.npy` header for streaming: it must be a 2-D, C-order,
/// little-endian `float32` array whose file holds exactly the declared data
/// (no shortfall, no trailing bytes).
///
/// The in-memory reader goes through `ndarray-npy`, which also accepts
/// big-endian or Fortran-order files; this one does not, and says so rather
/// than reading such a file with the wrong stride. [`streamable_layout`] lets a
/// caller fall back to the in-memory reader instead.
pub fn f32_npy_layout(path: &Path) -> Result<F32NpyLayout, String> {
    let shown = path.display();
    let path_str = path.to_str().ok_or("Invalid vectors path encoding")?;
    let header = super::npy_reader::read_header_for_streaming(path_str)?
        .ok_or_else(|| format!("{shown}: not a readable NPY file"))?;
    let dict = &header.dict;
    let descr_ok = dict.contains("'descr': '<f4'") || dict.contains("\"descr\": \"<f4\"");
    let c_order = dict.contains("'fortran_order': False");
    if !descr_ok || !c_order {
        return Err(format!(
            "{shown}: the streaming reader needs a C-order little-endian float32 array \
             ('<f4', fortran_order False); header is {}",
            dict.trim()
        ));
    }
    let dims = super::npy_reader::shape_dims(dict)
        .ok_or_else(|| format!("{shown}: could not parse NPY 'shape' from header"))?;
    let (rows, cols) = match dims.as_slice() {
        [rows, cols] if *cols > 0 => (*rows, *cols as usize),
        _ => return Err(format!("{shown}: expected a 2-D array, shape is {dims:?}")),
    };
    let layout = F32NpyLayout {
        rows,
        cols,
        data_start: header.data_start,
    };
    let want = layout
        .data_bytes()
        .checked_add(layout.data_start)
        .ok_or_else(|| format!("{shown}: NPY size overflow"))?;
    if header.file_len != want {
        return Err(format!(
            "{shown}: header declares {rows} x {cols} float32 ({} data bytes from offset {}) \
             but the file is {} bytes; it is truncated or carries trailing data",
            layout.data_bytes(),
            layout.data_start,
            header.file_len
        ));
    }
    Ok(layout)
}

/// [`f32_npy_layout`] as an option: `None` when the file is not in the layout
/// the streaming reader handles, so a caller can keep the in-memory reader
/// (which reports the real error if the file is unreadable).
pub fn streamable_layout(path: &Path) -> Option<F32NpyLayout> {
    f32_npy_layout(path).ok()
}

/// Divide a vector by its L2 norm, exactly as every in-memory reader does
/// (same summation order, same `norm > 0` rule), so streamed and loaded
/// vectors are bit-identical.
pub fn normalize_row(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// One block of rows as read from disk, not yet converted.
#[derive(Debug)]
pub struct RawBatch {
    /// Corpus id (row index) of the first row.
    pub first_id: u64,
    rows: usize,
    cols: usize,
    normalize: bool,
    vector_bytes: Vec<u8>,
    /// One raw line per row, or `None` when the corpus has no payload file.
    lines: Option<Vec<String>>,
}

impl RawBatch {
    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Bytes this batch holds before decoding (vector bytes + payload lines).
    pub fn raw_bytes(&self) -> usize {
        self.vector_bytes.len()
            + self
                .lines
                .as_ref()
                .map_or(0, |l| l.iter().map(String::len).sum())
    }

    /// Convert the vectors and parse the payloads. Consumes the raw buffers,
    /// so the raw and decoded forms of a batch never coexist for longer than
    /// the conversion itself.
    pub fn decode(self) -> Result<Batch, String> {
        let mut vectors = Vec::with_capacity(self.rows);
        for row in self.vector_bytes.chunks_exact(self.cols * 4) {
            let mut v: Vec<f32> = row
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            if self.normalize {
                normalize_row(&mut v);
            }
            vectors.push(v);
        }
        drop(self.vector_bytes);
        let metadata = match self.lines {
            None => vec![None; self.rows],
            Some(lines) => {
                let mut out = Vec::with_capacity(lines.len());
                for (i, line) in lines.into_iter().enumerate() {
                    out.push(if line.trim().is_empty() {
                        None
                    } else {
                        let value = serde_json::from_str(&line).map_err(|e| {
                            format!(
                                "Failed to parse payload row {}: {e}",
                                self.first_id + i as u64 + 1
                            )
                        })?;
                        parse_metadata_from_json(value)
                    });
                }
                out
            }
        };
        let ids = (0..self.rows)
            .map(|i| (self.first_id + i as u64) as i64)
            .collect();
        Ok(Batch {
            ids,
            vectors,
            metadata,
        })
    }
}

/// A decoded block: the same three parallel arrays the in-memory reader
/// returns, for a contiguous id range.
#[derive(Debug, Clone)]
pub struct Batch {
    pub ids: Vec<i64>,
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<Option<MetadataItem>>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Sequential block reader over a compound directory. See the module docs.
pub struct CompoundStream {
    vectors: BufReader<File>,
    vectors_path: PathBuf,
    payloads: Option<(BufReader<File>, PathBuf)>,
    layout: F32NpyLayout,
    next_row: u64,
    normalize: bool,
    finished: bool,
}

impl CompoundStream {
    /// Open `<dir>/vectors.npy` (and `<dir>/payloads.jsonl` when present).
    pub fn open(dir: &Path, normalize: bool) -> Result<Self, String> {
        let vectors_path = dir.join("vectors.npy");
        let layout = f32_npy_layout(&vectors_path)?;
        let mut vectors = BufReader::with_capacity(
            STREAM_READ_BUFFER,
            File::open(&vectors_path).map_err(|e| format!("Failed to open NPY file: {e}"))?,
        );
        std::io::copy(
            &mut (&mut vectors).take(layout.data_start),
            &mut std::io::sink(),
        )
        .map_err(|e| format!("{}: skip header: {e}", vectors_path.display()))?;
        let payloads_path = dir.join("payloads.jsonl");
        let payloads = if payloads_path.exists() {
            let f = File::open(&payloads_path)
                .map_err(|e| format!("Failed to open payloads file: {e}"))?;
            Some((
                BufReader::with_capacity(STREAM_READ_BUFFER, f),
                payloads_path,
            ))
        } else {
            None
        };
        Ok(Self {
            vectors,
            vectors_path,
            payloads,
            layout,
            next_row: 0,
            normalize,
            finished: false,
        })
    }

    /// Rows the corpus declares (the NPY header's first dimension).
    pub fn rows(&self) -> u64 {
        self.layout.rows
    }

    pub fn dim(&self) -> usize {
        self.layout.cols
    }

    /// Rows handed out so far.
    pub fn rows_read(&self) -> u64 {
        self.next_row
    }

    /// The next block of at most `max_rows` rows, or `None` once every row has
    /// been handed out. The end-of-corpus call also checks that
    /// `payloads.jsonl` has no rows beyond the vectors', with the same message
    /// the in-memory reader gives.
    pub fn next_raw(&mut self, max_rows: usize) -> Result<Option<RawBatch>, String> {
        if max_rows == 0 {
            return Err("a stream batch must hold at least one row".into());
        }
        if self.finished {
            return Ok(None);
        }
        let remaining = self.layout.rows - self.next_row;
        if remaining == 0 {
            self.finish()?;
            return Ok(None);
        }
        let n = (remaining as usize).min(max_rows);
        let mut vector_bytes = vec![0u8; n * self.layout.cols * 4];
        self.vectors.read_exact(&mut vector_bytes).map_err(|e| {
            format!(
                "{}: read rows {}..{}: {e}",
                self.vectors_path.display(),
                self.next_row,
                self.next_row + n as u64
            )
        })?;
        let lines = match &mut self.payloads {
            None => None,
            Some((reader, _)) => {
                let mut lines = Vec::with_capacity(n);
                for i in 0..n {
                    let mut line = String::new();
                    let got = reader
                        .read_line(&mut line)
                        .map_err(|e| format!("Failed to read payload row: {e}"))?;
                    if got == 0 {
                        // Same message as the in-memory reader, which counts
                        // the whole file first: here the file ended early, so
                        // the count is where it ended.
                        return Err(format!(
                            "Metadata count ({}) doesn't match vector count ({})",
                            self.next_row + i as u64,
                            self.layout.rows
                        ));
                    }
                    lines.push(line);
                }
                Some(lines)
            }
        };
        let batch = RawBatch {
            first_id: self.next_row,
            rows: n,
            cols: self.layout.cols,
            normalize: self.normalize,
            vector_bytes,
            lines,
        };
        self.next_row += n as u64;
        Ok(Some(batch))
    }

    /// Every row has been handed out: the payload file must be exhausted too.
    fn finish(&mut self) -> Result<(), String> {
        self.finished = true;
        if let Some((reader, _)) = &mut self.payloads {
            let mut extra = 0u64;
            let mut line = String::new();
            loop {
                line.clear();
                if reader
                    .read_line(&mut line)
                    .map_err(|e| format!("Failed to read payload row: {e}"))?
                    == 0
                {
                    break;
                }
                extra += 1;
            }
            if extra > 0 {
                return Err(format!(
                    "Metadata count ({}) doesn't match vector count ({})",
                    self.layout.rows + extra,
                    self.layout.rows
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readers::{read_compound_data, write_npy_vectors, MetadataValue};

    fn corpus(rows: usize, dim: usize, payloads: Option<&str>) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let vectors: Vec<Vec<f32>> = (0..rows)
            .map(|r| {
                (0..dim)
                    .map(|c| (r * dim + c) as f32 * 0.37 - 5.0)
                    .collect()
            })
            .collect();
        write_npy_vectors(dir.path().join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
        if let Some(p) = payloads {
            std::fs::write(dir.path().join("payloads.jsonl"), p).unwrap();
        }
        dir
    }

    fn payload_lines(rows: usize) -> String {
        (0..rows)
            .map(|i| {
                if i % 7 == 3 {
                    "\n".to_string()
                } else {
                    format!("{{\"tag\":\"t{}\",\"n\":{i},\"x\":{}.5}}\n", i % 3, i)
                }
            })
            .collect()
    }

    fn drain(dir: &Path, normalize: bool, block: usize) -> Result<Batch, String> {
        let mut s = CompoundStream::open(dir, normalize)?;
        let mut all = Batch {
            ids: vec![],
            vectors: vec![],
            metadata: vec![],
        };
        while let Some(raw) = s.next_raw(block)? {
            assert!(raw.len() <= block);
            let b = raw.decode()?;
            all.ids.extend(b.ids);
            all.vectors.extend(b.vectors);
            all.metadata.extend(b.metadata);
        }
        Ok(all)
    }

    fn same_metadata(a: &Option<MetadataItem>, b: &Option<MetadataItem>) -> bool {
        format!("{a:?}") == format!("{b:?}")
    }

    /// The contract that lets an engine switch readers: every block size, with
    /// and without normalization, with and without payloads, gives the same
    /// ids, the same vector BITS and the same metadata as the in-memory reader.
    #[test]
    fn compound_stream_equals_the_in_memory_reader() {
        for payloads in [Some(payload_lines(53)), None] {
            let dir = corpus(53, 5, payloads.as_deref());
            for normalize in [false, true] {
                let (ids, vectors, metadata) =
                    read_compound_data(dir.path().to_str().unwrap(), normalize).unwrap();
                for block in [1, 2, 7, 53, 64, 1000] {
                    let got = drain(dir.path(), normalize, block).unwrap();
                    assert_eq!(got.ids, ids, "block {block}");
                    let bits = |v: &Vec<Vec<f32>>| -> Vec<Vec<u32>> {
                        v.iter()
                            .map(|r| r.iter().map(|x| x.to_bits()).collect())
                            .collect()
                    };
                    assert_eq!(bits(&got.vectors), bits(&vectors), "block {block}");
                    assert_eq!(got.metadata.len(), metadata.len());
                    for (a, b) in got.metadata.iter().zip(&metadata) {
                        assert!(same_metadata(a, b), "{a:?} vs {b:?}");
                    }
                }
            }
        }
        // Spot-check that metadata really is parsed, not just matched as None.
        let dir = corpus(4, 2, Some(&payload_lines(4)));
        let b = drain(dir.path(), false, 3).unwrap();
        assert!(b.metadata[3].is_none(), "blank line is missing metadata");
        let first = b.metadata[0].as_ref().unwrap();
        assert!(first
            .fields
            .iter()
            .any(|(k, v)| k == "n" && matches!(v, MetadataValue::Int(0))));
    }

    #[test]
    fn a_payload_count_mismatch_is_the_same_error_as_in_memory() {
        for (rows, lines) in [(5, 4), (5, 6), (5, 0)] {
            let text: String = (0..lines).map(|i| format!("{{\"n\":{i}}}\n")).collect();
            let dir = corpus(rows, 3, Some(&text));
            let e = drain(dir.path(), false, 2).unwrap_err();
            assert!(e.contains("doesn't match vector count (5)"), "{e}");
            let mem = read_compound_data(dir.path().to_str().unwrap(), false).unwrap_err();
            assert!(mem.contains("doesn't match vector count (5)"), "{mem}");
            assert!(e.contains(&format!("Metadata count ({lines})")), "{e}");
        }
        let dir = corpus(3, 3, Some("{}\nnot json\n{}\n"));
        let e = drain(dir.path(), false, 2).unwrap_err();
        assert!(e.contains("Failed to parse payload row 2"), "{e}");
    }

    #[test]
    fn a_truncated_or_padded_npy_is_refused_before_any_row() {
        let dir = corpus(4, 3, None);
        let npy = dir.path().join("vectors.npy");
        let bytes = std::fs::read(&npy).unwrap();
        std::fs::write(&npy, &bytes[..bytes.len() - 1]).unwrap();
        assert!(CompoundStream::open(dir.path(), false)
            .err()
            .unwrap()
            .contains("truncated or carries trailing data"));
        let mut longer = bytes.clone();
        longer.extend_from_slice(&[0; 4]);
        std::fs::write(&npy, &longer).unwrap();
        assert!(CompoundStream::open(dir.path(), false).is_err());
        std::fs::write(&npy, &bytes).unwrap();
        assert_eq!(CompoundStream::open(dir.path(), false).unwrap().rows(), 4);
    }

    #[test]
    fn a_non_f4_or_fortran_npy_is_not_streamable() {
        let dir = tempfile::tempdir().unwrap();
        let npy = dir.path().join("vectors.npy");
        let arr = ndarray::Array2::<f64>::zeros((2, 3));
        ndarray_npy::WriteNpyExt::write_npy(&arr, File::create(&npy).unwrap()).unwrap();
        assert!(streamable_layout(&npy).is_none());
        assert!(f32_npy_layout(&npy).unwrap_err().contains("'<f4'"));
        let arr = ndarray::Array2::<f32>::zeros((2, 3)).reversed_axes();
        ndarray_npy::WriteNpyExt::write_npy(&arr, File::create(&npy).unwrap()).unwrap();
        assert!(streamable_layout(&npy).is_none());
        let arr = ndarray::Array2::<f32>::zeros((2, 3));
        ndarray_npy::WriteNpyExt::write_npy(&arr, File::create(&npy).unwrap()).unwrap();
        assert_eq!(
            streamable_layout(&npy).map(|l| (l.rows, l.cols)),
            Some((2, 3))
        );
    }

    /// The reader never holds more than one block: whatever the corpus size,
    /// a batch's buffers are sized by the block, and asking for a block larger
    /// than what remains allocates only what remains.
    #[test]
    fn a_block_never_holds_more_than_it_was_asked_for() {
        let dir = corpus(1000, 16, Some(&payload_lines(1000)));
        let mut s = CompoundStream::open(dir.path(), false).unwrap();
        let mut seen = 0;
        while let Some(raw) = s.next_raw(64).unwrap() {
            assert_eq!(raw.first_id, seen);
            assert!(raw.len() <= 64);
            assert_eq!(raw.vector_bytes.len(), raw.len() * 16 * 4);
            assert_eq!(raw.lines.as_ref().unwrap().len(), raw.len());
            seen += raw.len() as u64;
            assert_eq!(s.rows_read(), seen);
        }
        assert_eq!(seen, 1000);
        assert!(s.next_raw(64).unwrap().is_none());
        assert!(s.next_raw(0).is_err());
    }
}
