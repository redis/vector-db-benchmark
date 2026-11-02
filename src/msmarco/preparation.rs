//! Preparation capabilities and integrity contract (#388). Legacy full artifacts
//! remain readable; newly prepared artifacts are fail-closed and checksummed.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::workload::Workload;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PreparationMode {
    Full,
    TextScalar,
}

impl PreparationMode {
    /// Both expensive passes share this gate, including verification. Callbacks
    /// are lazy: text/scalar cannot accidentally compute and discard truth.
    pub fn truth<T, H>(
        self,
        unfiltered: impl FnOnce() -> Result<T, String>,
        filtered: impl FnOnce(&T) -> Result<H, String>,
    ) -> Result<Option<(T, H)>, String> {
        match self {
            Self::TextScalar => Ok(None),
            Self::Full => {
                let truth = unfiltered()?;
                let hybrid = filtered(&truth)?;
                Ok(Some((truth, hybrid)))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub version: u32,
    pub vectors: bool,
    pub text: bool,
    pub scalar: bool,
    pub vector_truth: bool,
    pub filtered_truth: bool,
    pub files: BTreeMap<String, FileIntegrity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIntegrity {
    pub bytes: u64,
    pub sha256: String,
}

pub fn digest(path: &Path) -> Result<FileIntegrity, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut sha = Sha256::new();
    let mut buf = vec![0; 4 << 20];
    let mut bytes = 0;
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
        bytes += n as u64;
    }
    Ok(FileIntegrity {
        bytes,
        sha256: sha.finalize().iter().map(|b| format!("{b:02x}")).collect(),
    })
}

/// Record exactly the staged/current files that will be published. Never use
/// an arbitrary stale .part file: the caller supplies the transaction's names.
pub fn seal(
    dir: &Path,
    manifest: &mut Value,
    staged: &[&str],
    vector: bool,
    hybrid: bool,
) -> Result<(), String> {
    let mut files = BTreeMap::new();
    let mut names = vec![
        "vectors.npy",
        "payloads.jsonl",
        "tests.jsonl",
        "filter_queries.jsonl",
    ];
    if hybrid {
        names.push("hybrid_queries.jsonl");
    }
    for name in names {
        let path = if staged.contains(&name) {
            dir.join(format!("{name}.part"))
        } else {
            dir.join(name)
        };
        files.insert(name.to_string(), digest(&path)?);
    }
    manifest["source_revision"] = json!(super::HF_REVISION);
    manifest["preparation"] = serde_json::to_value(Capabilities {
        version: 1,
        vectors: true,
        text: true,
        scalar: true,
        vector_truth: vector,
        filtered_truth: hybrid,
        files,
    })
    .map_err(|e| e.to_string())?;
    if !vector {
        manifest
            .as_object_mut()
            .ok_or("manifest must be an object")?
            .remove("ground_truth");
        manifest.as_object_mut().unwrap().remove("neighbours");
    }
    Ok(())
}

pub fn read_manifest(dir: &Path) -> Result<Value, String> {
    let path = dir.join("PREPARED.json");
    serde_json::from_reader(File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Cheap loader gate (sizes/NPY header/query schema); --verify and truth
/// upgrades additionally hash every file with bounded buffers.
pub fn check(
    dir: &Path,
    variant: &super::Variant,
    hashes: bool,
) -> Result<Option<Capabilities>, String> {
    inspect(dir, variant, hashes, false)
}

fn artifact_path(dir: &Path, name: &str, staged: bool) -> std::path::PathBuf {
    let part = dir.join(format!("{name}.part"));
    if staged && part.exists() {
        part
    } else {
        dir.join(name)
    }
}

/// Validate the exact post-publication view without copying any corpus bytes.
/// Recovery may read staged files, but ordinary loaders never may.
fn inspect(
    dir: &Path,
    variant: &super::Variant,
    hashes: bool,
    staged: bool,
) -> Result<Option<Capabilities>, String> {
    let manifest_path = artifact_path(dir, "PREPARED.json", staged);
    if !manifest_path.exists() {
        // Legacy tarballs may omit the manifest, but cannot masquerade as
        // complete text/scalar preparation with the new ranking-free rows.
        check_legacy_truth(dir)?;
        return Ok(None);
    }
    let manifest: Value =
        serde_json::from_reader(File::open(manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let Some(value) = manifest.get("preparation") else {
        check_legacy_truth(dir)?;
        return Ok(None);
    };
    let caps: Capabilities = serde_json::from_value(value.clone())
        .map_err(|e| format!("invalid preparation capabilities: {e}"))?;
    if caps.version != 1
        || !caps.vectors
        || !caps.text
        || !caps.scalar
        || (caps.filtered_truth && !caps.vector_truth)
    {
        return Err("unsupported/incomplete preparation capabilities".into());
    }
    if !staged && !dir.join("COMPLETE").is_file() {
        return Err("incomplete preparation: missing COMPLETE marker; finish/rebuild the interrupted preparation".into());
    }
    if manifest["dataset"] != variant.dataset_name
        || manifest["passages"] != variant.limit
        || manifest["dimensions"] != super::DIM
        || manifest["distance"] != "cosine"
        || manifest["sampling"]["threshold"] != variant.threshold
        || manifest["sampling"]["buckets"] != super::CRC32_BUCKETS
        || manifest["source_revision"] != super::HF_REVISION
    {
        return Err(
            "prepared corpus identity does not match the registered dataset/sampling/source".into(),
        );
    }
    let mut required = vec![
        "vectors.npy",
        "payloads.jsonl",
        "tests.jsonl",
        "filter_queries.jsonl",
    ];
    if caps.filtered_truth {
        required.push("hybrid_queries.jsonl");
    }
    if caps.files.len() != required.len() {
        return Err("incomplete preparation file inventory".into());
    }
    for name in required {
        let expected = caps
            .files
            .get(name)
            .ok_or_else(|| format!("no integrity record for {name}"))?;
        let path = artifact_path(dir, name, staged);
        let actual = std::fs::metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if actual.len() != expected.bytes
            || expected.sha256.len() != 64
            || !expected.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(format!("{name}: invalid size/integrity record"));
        }
        if hashes && digest(&path)?.sha256 != expected.sha256 {
            return Err(format!("{name}: SHA-256 mismatch"));
        }
    }
    let layout =
        crate::readers::stream_reader::f32_npy_layout(&artifact_path(dir, "vectors.npy", staged))?;
    if layout.rows != variant.limit || layout.cols != super::DIM {
        return Err("vectors.npy: corpus shape mismatch".into());
    }
    let count = manifest["queries"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("manifest has no query count")?;
    check_query_rows(
        &artifact_path(dir, "tests.jsonl", staged),
        super::DIM,
        Some(count),
        caps.vector_truth,
    )?;
    Ok(Some(caps))
}

/// Finish an interrupted sealed publication, including upgrades, without
/// rebuilding unchanged corpus files. Validate ALL candidate hashes, identity,
/// shape and query inputs before the first rename. A missing/truncated/mixed
/// candidate remains incomplete. Call only from preparation commands, never a
/// benchmark loader; there must be exactly one writer for the directory.
pub fn recover_publication(dir: &Path, variant: &super::Variant) -> Result<bool, String> {
    if dir.join("COMPLETE").exists() || !artifact_path(dir, "PREPARED.json", true).exists() {
        return Ok(false);
    }
    let Some(caps) = inspect(dir, variant, true, true)? else {
        return Ok(false);
    };
    // Drop invalidated filtered truth before publishing a replacement payload.
    if !caps.filtered_truth {
        match std::fs::remove_file(dir.join("hybrid_queries.jsonl")) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    // Preserve the older derived-field publication invariant too: predicates
    // are published only after the payloads they count, and are renamed last.
    for name in [
        "vectors.npy",
        "payloads.jsonl",
        "tests.jsonl",
        "hybrid_queries.jsonl",
        "PREPARED.json",
        "filter_queries.jsonl",
    ] {
        if name != "PREPARED.json" && !caps.files.contains_key(name) {
            continue;
        }
        let part = dir.join(format!("{name}.part"));
        if part.exists() {
            std::fs::rename(&part, dir.join(name)).map_err(|e| format!("recover {name}: {e}"))?;
        }
    }
    std::fs::write(dir.join("COMPLETE"), "ok\n").map_err(|e| e.to_string())?;
    Ok(true)
}

/// Old upload-only directories can omit queries. When queries are present,
/// require the legacy ranking field so deleting a new manifest cannot silently
/// turn a ranking-free artifact into a legacy full artifact.
fn check_legacy_truth(dir: &Path) -> Result<(), String> {
    let path = dir.join("tests.jsonl");
    if !path.exists() {
        return Ok(());
    }
    let f = File::open(&path).map_err(|e| e.to_string())?;
    let mut rows = 0;
    for line in BufReader::new(f).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        rows += 1;
        if row["closest_ids"].as_array().is_none_or(Vec::is_empty) {
            return Err("KNN truth is absent; restore PREPARED.json and run prepare-msmarco --dataset <dataset> --add-knn-truth --out-dir <datasets dir>".into());
        }
    }
    if rows == 0 {
        return Err("tests.jsonl: no queries or KNN truth".into());
    }
    Ok(())
}

fn check_query_rows(
    path: &Path,
    dim: usize,
    count: Option<u64>,
    truth: bool,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("tests.jsonl: {e}"))?;
    let mut rows = 0;
    for line in BufReader::new(file).lines() {
        let row: Value = serde_json::from_str(&line.map_err(|e| e.to_string())?)
            .map_err(|e| format!("tests.jsonl: {e}"))?;
        if row["query"].as_array().is_none_or(|q| {
            q.len() != dim || q.iter().any(|v| v.as_f64().is_none_or(|x| !x.is_finite()))
        }) || row["query_text"]
            .as_str()
            .is_none_or(|s| s.trim().is_empty())
            || row["query_id"].as_str().is_none_or(str::is_empty)
        {
            return Err("tests.jsonl: invalid query input".into());
        }
        if truth
            && (row["closest_ids"].as_array().is_none_or(Vec::is_empty)
                || row["closest_scores"].as_array().is_none_or(Vec::is_empty))
        {
            return Err("KNN truth is absent; restore PREPARED.json and run prepare-msmarco --dataset <dataset> --add-knn-truth --out-dir <datasets dir>".into());
        }
        if !truth && (row.get("closest_ids").is_some() || row.get("closest_scores").is_some()) {
            return Err("text/scalar-only query rows must omit vector rankings".into());
        }
        rows += 1;
    }
    if rows == 0 || count.is_some_and(|n| n != rows) {
        return Err("tests.jsonl: query count mismatch".into());
    }
    Ok(())
}

pub fn require_workloads(
    caps: Option<&Capabilities>,
    dataset: &str,
    workloads: &[Workload],
) -> Result<(), String> {
    if let Some(caps) = caps {
        for workload in workloads {
            if (*workload == Workload::Vector && !caps.vector_truth)
                || (*workload == Workload::Hybrid && !caps.filtered_truth)
            {
                return Err(format!("{dataset}: {workload} requires missing KNN truth. Upgrade the same corpus with `prepare-msmarco --dataset {dataset} --add-knn-truth --out-dir <datasets dir>`"));
            }
        }
    }
    Ok(())
}

pub fn input_row(query: &[f32], id: &str, text: &str, year: Option<i64>) -> Value {
    json!({"query": query, "query_id": id, "query_text": text, "trec_year": year})
}

/// The source query inputs must be identical before adding any rankings.
pub fn check_query_inputs(expected: &[Value], stored: &[Value]) -> Result<Vec<Vec<f32>>, String> {
    if expected.len() != stored.len() {
        return Err("stored query count differs from the pinned upstream source".into());
    }
    expected
        .iter()
        .zip(stored)
        .enumerate()
        .map(|(i, (a, b))| {
            for key in ["query_id", "query_text", "trec_year"] {
                if a[key] != b[key] {
                    return Err(format!(
                        "query {i}: {key} differs from the pinned upstream source"
                    ));
                }
            }
            let parse = |v: &Value| -> Result<Vec<f32>, String> {
                v.as_array()
                    .ok_or("query is not an array")?
                    .iter()
                    .map(|x| {
                        x.as_f64()
                            .filter(|x| x.is_finite())
                            .map(|x| x as f32)
                            .ok_or("non-finite/non-numeric query".into())
                    })
                    .collect()
            };
            let reference = parse(&a["query"])?;
            let stored = parse(&b["query"])?;
            // Normalizing float32 on another build/architecture can differ by a
            // handful of ulps. Bound that difference, then ALWAYS score/preserve
            // the stored vectors, never replace them with the re-normalized copy.
            if reference.len() != stored.len()
                || reference.is_empty()
                || reference
                    .iter()
                    .zip(&stored)
                    .any(|(x, y)| (*x - *y).abs() > 2e-6)
                || (stored.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() > 1e-3
            {
                return Err(format!(
                    "query {i}: vector differs from the pinned upstream input"
                ));
            }
            Ok(stored)
        })
        .collect()
}

/// The published v2 tarballs predate removal of the `-crc32-` alias.
/// Accept that alias only with the exact registered count/hash selection.
pub fn source_variant(manifest: &Value) -> Result<&'static super::Variant, String> {
    let name = manifest["dataset"]
        .as_str()
        .ok_or("source manifest has no dataset")?;
    let variant = super::VARIANTS
        .iter()
        .find(|v| {
            name == v.dataset_name
                || (manifest.get("preparation").is_none()
                    && name == v.dataset_name.replace("-cosine", "-crc32-cosine"))
        })
        .ok_or("source is not a registered MS MARCO corpus")?;
    if manifest["passages"] != variant.limit
        || manifest["dimensions"] != super::DIM
        || manifest["sampling"]["threshold"] != variant.threshold
        || manifest["sampling"]["buckets"] != super::CRC32_BUCKETS
        || manifest["sampling"]["mode"] != "crc32"
    {
        return Err("source corpus identity/sampling does not match its registered name".into());
    }
    Ok(variant)
}

/// Stream a nested sample from an existing prepared corpus. Selection is the
/// SAME docid hash predicate as the upstream scan, and order is preserved.
/// The caller validates the source capability/integrity contract first.
pub fn sample_corpus(
    source: &Path,
    destination: &Path,
    threshold: u32,
    expected_rows: u64,
) -> Result<std::collections::HashMap<String, usize>, String> {
    use std::io::Write;
    let (mut source_vectors, rows, dim) = super::hybrid::open_f32_npy(&source.join("vectors.npy"))?;
    if dim != super::DIM {
        return Err("nested source has wrong dimensions".into());
    }
    let mut vectors =
        super::DeferredNpyF32Writer::create(&destination.join("vectors.npy.part"), super::DIM)?;
    let mut payloads = std::io::BufWriter::with_capacity(
        4 << 20,
        File::create(destination.join("payloads.jsonl.part")).map_err(|e| e.to_string())?,
    );
    let source_payloads = File::open(source.join("payloads.jsonl")).map_err(|e| e.to_string())?;
    let mut lines = BufReader::with_capacity(4 << 20, source_payloads).lines();
    let mut max_bytes = std::collections::HashMap::new();
    let mut raw = vec![0; dim * 4];
    for _ in 0..rows {
        source_vectors
            .read_exact(&mut raw)
            .map_err(|e| e.to_string())?;
        let line = lines
            .next()
            .ok_or("payloads ended before vectors")?
            .map_err(|e| e.to_string())?;
        // Parse only docid on rejected rows: allocating all the text fields
        // for 100M passages would dominate a 1M nested sample.
        #[derive(Deserialize)]
        struct SampleId {
            docid: String,
        }
        let id: SampleId = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if !super::keeps(&id.docid, threshold) {
            continue;
        }
        let row: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        for (name, value) in row.as_object().ok_or("non-object payload")? {
            if let Some(value) = value.as_str() {
                let max = max_bytes.entry(name.clone()).or_insert(0);
                *max = (*max).max(value.len());
            }
        }
        let vector: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        super::ensure_finite(&vector, 0, dim)?;
        vectors.write_row(&vector)?;
        writeln!(payloads, "{line}").map_err(|e| e.to_string())?;
    }
    if lines.next().is_some() {
        return Err("payloads outlast vectors".into());
    }
    let rows = vectors.finish()?;
    payloads.flush().map_err(|e| e.to_string())?;
    if rows != expected_rows {
        return Err(format!(
            "nested sample selected {rows}, expected {expected_rows}"
        ));
    }
    Ok(max_bytes)
}

/// Selection metadata recovered from a shard's exact pinned upstream metadata.
#[derive(Debug)]
pub struct RecoveredShard {
    pub kept: u64,
    pub wanted: Vec<(u64, i64)>,
    pub docids: std::collections::HashMap<u64, String>,
    pub max_field_bytes: std::collections::HashMap<String, usize>,
}

/// Recover a prior attempt's selected shard only after re-reading ALL source
/// metadata, comparing every selected payload in order, checking exact vector
/// length and scanning all stored floats for non-finite values. No row-count
/// estimate or filename alone can make an interrupted shard reusable.
#[allow(clippy::too_many_arguments)]
pub fn recover_part(
    source: impl BufRead,
    payloads: &Path,
    vectors: &Path,
    source_rows: u64,
    first_id: u64,
    threshold: u32,
    dimensions: usize,
    wanted_offsets: &std::collections::HashSet<u64>,
) -> Result<RecoveredShard, String> {
    let mut stored =
        BufReader::with_capacity(4 << 20, File::open(payloads).map_err(|e| e.to_string())?).lines();
    let mut out = RecoveredShard {
        kept: 0,
        wanted: Vec::new(),
        docids: Default::default(),
        max_field_bytes: Default::default(),
    };
    let mut seen = 0;
    for line in source.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let source: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        let payload = super::payload_from_passage(&source)?;
        let docid = payload["docid"].as_str().ok_or("missing source docid")?;
        let offset = first_id + seen;
        seen += 1;
        if !super::keeps(docid, threshold) {
            continue;
        }
        let actual = stored
            .next()
            .ok_or("cached payloads are truncated")?
            .map_err(|e| e.to_string())?;
        let actual: Value = serde_json::from_str(&actual).map_err(|e| e.to_string())?;
        if actual != payload {
            return Err("cached payload differs from the pinned source/order".into());
        }
        for (name, value) in payload.as_object().ok_or("payload is not an object")? {
            if let Some(text) = value.as_str() {
                let size = out.max_field_bytes.entry(name.clone()).or_insert(0);
                *size = (*size).max(text.len());
            }
        }
        if wanted_offsets.contains(&offset) {
            out.wanted.push((offset, out.kept as i64));
            out.docids.insert(offset, docid.to_string());
        }
        out.kept += 1;
    }
    if seen != source_rows || stored.next().is_some() {
        return Err("cached/source shard row counts differ".into());
    }
    let mut file = File::open(vectors).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() != out.kept * dimensions as u64 * 4 {
        return Err("cached vectors are truncated or carry extra rows".into());
    }
    let mut bytes = vec![0; 4 << 20];
    let mut remaining = out.kept * dimensions as u64 * 4;
    while remaining > 0 {
        let size = remaining.min(bytes.len() as u64) as usize;
        file.read_exact(&mut bytes[..size])
            .map_err(|e| e.to_string())?;
        if bytes[..size]
            .as_chunks::<4>()
            .0
            .iter()
            .any(|b| !f32::from_le_bytes(*b).is_finite())
        {
            return Err("cached vectors contain non-finite values".into());
        }
        remaining -= size as u64;
    }
    Ok(out)
}

/// Bounded HTTP retries for transient upstream failures. Honor Retry-After
/// seconds; other statuses (including auth and range errors) fail immediately.
pub fn retry_delay(
    status: u16,
    retry_after: Option<&str>,
    attempt: u32,
) -> Option<std::time::Duration> {
    if attempt >= 6 || !matches!(status, 429 | 500 | 502 | 503 | 504) {
        return None;
    }
    Some(std::time::Duration::from_secs(
        retry_after
            .and_then(|s| s.parse().ok())
            .unwrap_or(2_u64.pow(attempt + 1)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_scalar_never_invokes_either_oracle() {
        let result: Option<((), ())> = PreparationMode::TextScalar
            .truth(
                || panic!("unfiltered oracle invoked"),
                |_| panic!("filtered oracle invoked"),
            )
            .unwrap();
        assert!(result.is_none());
        assert_eq!(
            PreparationMode::Full
                .truth(|| Ok(7), |v| Ok(v + 1))
                .unwrap(),
            Some((7, 8))
        );
    }
    #[test]
    fn truth_is_required_for_vector_and_hybrid_only() {
        let caps = Capabilities {
            version: 1,
            vectors: true,
            text: true,
            scalar: true,
            vector_truth: false,
            filtered_truth: false,
            files: BTreeMap::new(),
        };
        require_workloads(
            Some(&caps),
            "fixture",
            &[Workload::Fulltext, Workload::FilterOnly],
        )
        .unwrap();
        for w in [Workload::Vector, Workload::Hybrid] {
            assert!(require_workloads(Some(&caps), "fixture", &[w])
                .unwrap_err()
                .contains("--add-knn-truth"));
        }
    }
    fn fixture() -> (tempfile::TempDir, super::super::Variant) {
        let dir = tempfile::tempdir().unwrap();
        let variant = super::super::Variant {
            dataset_name: "fixture",
            dir: "fixture",
            limit: 3,
            threshold: 886,
        };
        let mut query = vec![0.0; super::super::DIM];
        query[0] = 1.0;
        crate::readers::write_npy_vectors(
            dir.path().join("vectors.npy").to_str().unwrap(),
            &vec![query.clone(); 3],
        )
        .unwrap();
        std::fs::write(dir.path().join("payloads.jsonl"), "{}\n{}\n{}\n").unwrap();
        std::fs::write(dir.path().join("filter_queries.jsonl"), "{}\n").unwrap();
        std::fs::write(
            dir.path().join("tests.jsonl"),
            format!("{}\n", input_row(&query, "q1", "example search", None)),
        )
        .unwrap();
        let mut manifest = json!({"dataset": "fixture", "passages": 3, "dimensions": super::super::DIM,
            "distance": "cosine", "queries": 1, "sampling": {"threshold": 886, "buckets": super::super::CRC32_BUCKETS}});
        seal(dir.path(), &mut manifest, &[], false, false).unwrap();
        std::fs::write(dir.path().join("PREPARED.json"), manifest.to_string()).unwrap();
        std::fs::write(dir.path().join("COMPLETE"), "ok\n").unwrap();
        (dir, variant)
    }

    #[test]
    fn interrupted_publication_recovers_every_rename_boundary_without_rebuilding_corpus() {
        for done in 0..=3 {
            let (dir, variant) = fixture();
            let d = dir.path();
            let before_vectors = digest(&d.join("vectors.npy")).unwrap().sha256;
            let before_payloads = digest(&d.join("payloads.jsonl")).unwrap().sha256;
            let mut query: Value = serde_json::from_str(
                std::fs::read_to_string(d.join("tests.jsonl"))
                    .unwrap()
                    .trim(),
            )
            .unwrap();
            query["closest_ids"] = json!([0]);
            query["closest_scores"] = json!([1.0]);
            std::fs::write(d.join("tests.jsonl.part"), format!("{query}\n")).unwrap();
            std::fs::write(d.join("hybrid_queries.jsonl.part"), "{}\n").unwrap();
            let mut manifest = read_manifest(d).unwrap();
            seal(
                d,
                &mut manifest,
                &["tests.jsonl", "hybrid_queries.jsonl"],
                true,
                true,
            )
            .unwrap();
            std::fs::write(d.join("PREPARED.json.part"), manifest.to_string()).unwrap();
            std::fs::remove_file(d.join("COMPLETE")).unwrap();
            for name in ["PREPARED.json", "hybrid_queries.jsonl", "tests.jsonl"]
                .iter()
                .take(done)
            {
                std::fs::rename(d.join(format!("{name}.part")), d.join(name)).unwrap();
            }
            assert!(check(d, &variant, false).is_err());
            assert!(recover_publication(d, &variant).unwrap(), "boundary {done}");
            let caps = check(d, &variant, true).unwrap().unwrap();
            assert!(caps.vector_truth && caps.filtered_truth);
            assert_eq!(
                digest(&d.join("vectors.npy")).unwrap().sha256,
                before_vectors
            );
            assert_eq!(
                digest(&d.join("payloads.jsonl")).unwrap().sha256,
                before_payloads
            );
            assert!(!recover_publication(d, &variant).unwrap());
        }
    }

    #[test]
    fn publication_recovery_rejects_corruption_before_any_rename() {
        let (dir, variant) = fixture();
        let d = dir.path();
        let mut manifest = read_manifest(d).unwrap();
        std::fs::write(
            d.join("payloads.jsonl.part"),
            "{\"new\":1}\n{\"new\":2}\n{\"new\":3}\n",
        )
        .unwrap();
        seal(d, &mut manifest, &["payloads.jsonl"], false, false).unwrap();
        std::fs::write(d.join("PREPARED.json.part"), manifest.to_string()).unwrap();
        std::fs::remove_file(d.join("COMPLETE")).unwrap();
        let original_vectors = std::fs::read(d.join("vectors.npy")).unwrap();
        let mut corrupt = original_vectors.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        std::fs::write(d.join("vectors.npy"), corrupt).unwrap();
        assert!(recover_publication(d, &variant)
            .unwrap_err()
            .contains("SHA-256"));
        assert!(d.join("payloads.jsonl.part").exists());
        assert_eq!(
            std::fs::read_to_string(d.join("payloads.jsonl")).unwrap(),
            "{}\n{}\n{}\n"
        );
        assert!(!d.join("COMPLETE").exists());
        std::fs::write(d.join("vectors.npy"), original_vectors).unwrap();
        // Derived-field upgrades invalidate the previously published hybrid set.
        std::fs::write(d.join("hybrid_queries.jsonl"), "stale\n").unwrap();
        assert!(recover_publication(d, &variant).unwrap());
        assert!(!d.join("hybrid_queries.jsonl").exists());
        check(d, &variant, true).unwrap();
    }

    #[test]
    fn capabilities_reject_incomplete_corrupt_and_wrong_corpora() {
        let (dir, variant) = fixture();
        let caps = check(dir.path(), &variant, true).unwrap().unwrap();
        assert!(!caps.vector_truth && !caps.filtered_truth);
        assert_eq!(
            crate::fulltext::read_query_texts(dir.path()).unwrap(),
            ["example search"]
        );
        let manifest = read_manifest(dir.path()).unwrap();
        for key in [
            "dataset",
            "passages",
            "dimensions",
            "distance",
            "sampling",
            "source_revision",
            "queries",
            "preparation",
        ] {
            let mut broken = manifest.clone();
            broken[key] = Value::Null;
            std::fs::write(dir.path().join("PREPARED.json"), broken.to_string()).unwrap();
            assert!(
                check(dir.path(), &variant, true).is_err(),
                "accepted corrupt {key}"
            );
        }
        std::fs::write(dir.path().join("PREPARED.json"), manifest.to_string()).unwrap();
        std::fs::remove_file(dir.path().join("COMPLETE")).unwrap();
        assert!(check(dir.path(), &variant, false)
            .unwrap_err()
            .contains("COMPLETE"));
        std::fs::write(dir.path().join("COMPLETE"), "ok\n").unwrap();
        // Same-length corruption must be caught by verification/upgrade hashing.
        std::fs::write(dir.path().join("payloads.jsonl"), "[]\n{}\n{}\n").unwrap();
        assert!(check(dir.path(), &variant, true)
            .unwrap_err()
            .contains("SHA-256"));
        std::fs::write(dir.path().join("payloads.jsonl"), "{}\n").unwrap();
        assert!(check(dir.path(), &variant, false)
            .unwrap_err()
            .contains("size"));
        std::fs::remove_file(dir.path().join("PREPARED.json")).unwrap();
        assert!(check(dir.path(), &variant, false)
            .unwrap_err()
            .contains("truth is absent"));
    }

    #[test]
    fn upgrade_rejects_changed_query_identity_and_preserves_inputs() {
        let inputs = vec![input_row(&[1.0, 0.0], "q", "term", None)];
        let ranked = vec![super::super::test_row(
            &[1.0, 0.0],
            "q",
            "term",
            None,
            &[(1.0, 2), (0.0, 1)],
        )];
        check_query_inputs(&inputs, &ranked).unwrap();
        for key in ["query", "query_id", "query_text", "trec_year"] {
            let mut broken = ranked.clone();
            broken[0][key] = json!("wrong");
            assert!(check_query_inputs(&inputs, &broken).is_err());
        }
        assert!(check_query_inputs(&inputs, &[]).is_err());
    }

    #[test]
    fn nested_sample_preserves_selection_vectors_and_document_order() {
        let source = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        let payloads: Vec<_> = (0..24)
            .map(|i| json!({"docid": format!("sample-{i}"), "title": format!("title-{i}")}))
            .collect();
        let vectors: Vec<_> = (0..24).map(|i| vec![i as f32; super::super::DIM]).collect();
        crate::readers::write_npy_vectors(
            source.path().join("vectors.npy").to_str().unwrap(),
            &vectors,
        )
        .unwrap();
        std::fs::write(
            source.path().join("payloads.jsonl"),
            payloads
                .iter()
                .map(|p| format!("{p}\n"))
                .collect::<String>(),
        )
        .unwrap();
        let chosen: Vec<usize> = payloads
            .iter()
            .enumerate()
            .filter(|(_, p)| super::super::keeps(p["docid"].as_str().unwrap(), 500_000))
            .map(|(i, _)| i)
            .collect();
        assert!(!chosen.is_empty() && chosen.len() < vectors.len());
        sample_corpus(source.path(), target.path(), 500_000, chosen.len() as u64).unwrap();
        let (_, sampled) = crate::readers::read_npy_vectors(
            target.path().join("vectors.npy.part").to_str().unwrap(),
            false,
        )
        .unwrap();
        assert_eq!(
            sampled,
            chosen
                .iter()
                .map(|i| vectors[*i].clone())
                .collect::<Vec<_>>()
        );
        let sampled = std::fs::read_to_string(target.path().join("payloads.jsonl.part")).unwrap();
        assert_eq!(
            sampled,
            chosen
                .iter()
                .map(|i| format!("{}\n", payloads[*i]))
                .collect::<String>()
        );
        assert!(sample_corpus(source.path(), target.path(), 500_000, 999)
            .unwrap_err()
            .contains("expected 999"));
    }
    #[test]
    fn legacy_source_alias_requires_the_exact_registered_corpus() {
        let mut manifest = json!({"dataset": "msmarco-cohere-1024-100K-crc32-cosine", "passages": 99964, "dimensions": 1024,
            "sampling": {"mode": "crc32", "threshold": 886, "buckets": 1000000}});
        assert_eq!(
            source_variant(&manifest).unwrap(),
            &super::super::VARIANTS[0]
        );
        manifest["passages"] = json!(100000);
        assert!(source_variant(&manifest).is_err());
        manifest["passages"] = json!(99964);
        manifest["sampling"]["threshold"] = json!(887);
        assert!(source_variant(&manifest).is_err());
    }
    #[test]
    fn transient_retry_is_bounded_and_honors_server_delay() {
        assert_eq!(retry_delay(429, Some("17"), 0).unwrap().as_secs(), 17);
        assert_eq!(retry_delay(503, None, 2).unwrap().as_secs(), 8);
        assert!(retry_delay(429, None, 6).is_none());
        for status in [200, 206, 400, 401, 403, 404, 416] {
            assert!(retry_delay(status, None, 0).is_none());
        }
    }
    #[test]
    fn recovery_rechecks_all_source_rows_and_rejects_partial_or_reordered_parts() {
        let dir = tempfile::tempdir().unwrap();
        let payloads = dir.path().join("payloads");
        let vectors = dir.path().join("vectors");
        let source: Vec<_> = (0..4)
            .map(|i| json!({"docid": format!("source-{i}"), "title": "test"}))
            .collect();
        let input = source.iter().map(|p| format!("{p}\n")).collect::<String>();
        let expected = source
            .iter()
            .map(|p| format!("{}\n", super::super::payload_from_passage(p).unwrap()))
            .collect::<String>();
        std::fs::write(&payloads, &expected).unwrap();
        let bytes: Vec<u8> = (0..8).flat_map(|n| (n as f32).to_le_bytes()).collect();
        std::fs::write(&vectors, &bytes).unwrap();
        let wanted = std::collections::HashSet::from([11]);
        let run = || {
            recover_part(
                std::io::Cursor::new(input.as_bytes()),
                &payloads,
                &vectors,
                4,
                10,
                1_000_000,
                2,
                &wanted,
            )
        };
        let result = run().unwrap();
        assert_eq!(result.kept, 4);
        assert_eq!(result.wanted, [(11, 1)]);
        assert_eq!(result.docids[&11], "source-1");
        std::fs::write(&vectors, &bytes[..28]).unwrap();
        assert!(run().unwrap_err().contains("truncated"));
        std::fs::write(&vectors, &bytes).unwrap();
        let reversed = expected
            .lines()
            .rev()
            .map(|l| format!("{l}\n"))
            .collect::<String>();
        std::fs::write(&payloads, reversed).unwrap();
        assert!(run().unwrap_err().contains("order"));
        std::fs::write(
            &payloads,
            expected
                .lines()
                .take(3)
                .map(|l| format!("{l}\n"))
                .collect::<String>(),
        )
        .unwrap();
        assert!(run().unwrap_err().contains("truncated"));
        std::fs::write(&payloads, &expected).unwrap();
        assert!(recover_part(
            std::io::Cursor::new(input.as_bytes()),
            &payloads,
            &vectors,
            5,
            10,
            1_000_000,
            2,
            &wanted
        )
        .unwrap_err()
        .contains("counts"));
    }
}
