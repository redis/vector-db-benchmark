//! Per-query result records and the `--dump-results` file format (#343).
//!
//! # Contract
//!
//! Engines capture a [`RawQueryResult`] per timed query, in engine rank order,
//! naming documents by their **corpus row id** (the id space the dataset's
//! ground truth uses — the integer the engine stored the row under, recovered
//! from its key / `_id`). Capture is opt-in (`SearchParams::capture_query_results`)
//! and happens after the latency sample is taken: outside each per-query
//! latency, but inside the throughput window (`total_time`, hence `rps`) when
//! on. Without it, engines must not build per-hit ids at all.
//!
//! After the search phase the runner turns those into [`QueryRecord`]s with
//! [`resolve_records`], replacing row ids by EXTERNAL ids through
//! [`ExternalIds`] — the dataset's `docid` field when its schema declares one
//! (MS MARCO), else the decimal row id. Never an engine-internal key
//! (`<prefix>:<row>`), Lucene doc id or offset.
//!
//! One JSONL file per (config, dataset, workload, parallel, search config):
//! a [`DumpLine::Header`] first, then one [`DumpLine::Query`] per timed query in
//! dispatch order (`seq`). Failed queries are kept, with `error` set, so a
//! comparison can tell "answered nothing" from "never answered".
//!
//! ```text
//! {"kind":"header","schema_version":1,"engine":"redis","config":"redis-m-16-ef-128",...}
//! {"kind":"query","query_id":0,"seq":0,"ids":["msmarco_v2.1_doc_00_0#1_2"],"scores":[0.12],"total":null,"latency":0.0011,...}
//! ```

use crate::workload::Workload;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};

pub const DUMP_SCHEMA_VERSION: u32 = 1;

/// One returned document, as captured inside an engine's search loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawHit {
    /// Corpus row id (the dataset's id space, NOT an engine key).
    pub row: i64,
    /// Engine-native score (distance for Redis KNN, `_score` for OpenSearch).
    /// Not comparable across engines; `None` where the engine returns none
    /// (filter-only).
    pub score: Option<f64>,
}

/// What an engine records for one timed query.
#[derive(Debug, Clone, PartialEq)]
pub struct RawQueryResult {
    /// Index of the query in the dataset's query set (`tests.jsonl` line / test
    /// row) — NOT the position in a filtered or cycled subset.
    pub query_id: usize,
    /// Dispatch ordinal within the timed window.
    pub seq: usize,
    /// In engine rank order. Empty on failure.
    pub hits: Vec<RawHit>,
    /// Exact total match count, when the workload requests one (filter-only).
    pub total: Option<u64>,
    /// Seconds, exactly the sample the latency statistics received (or would
    /// have, for a failed query).
    pub latency: f64,
    /// `Some` when the query failed; it is then excluded from the statistics.
    pub error: Option<String>,
}

impl RawQueryResult {
    pub fn ok(
        query_id: usize,
        seq: usize,
        hits: Vec<RawHit>,
        total: Option<u64>,
        latency: f64,
    ) -> Self {
        Self {
            query_id,
            seq,
            hits,
            total,
            latency,
            error: None,
        }
    }

    pub fn failed(query_id: usize, seq: usize, latency: f64, error: impl Into<String>) -> Self {
        Self {
            query_id,
            seq,
            hits: Vec::new(),
            total: None,
            latency,
            error: Some(error.into()),
        }
    }
}

/// Which id space a dump's `ids` are in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IdSpace {
    /// The dataset's row id, in decimal — for datasets with no external id.
    Row,
    /// A payload field (e.g. MS MARCO `docid`).
    Field { field: String },
}

/// Row id -> external id.
#[derive(Debug, Clone)]
pub enum ExternalIds {
    Row,
    Field { field: String, values: Vec<String> },
}

/// Payload field used as the external id when a dataset's schema declares it.
pub const EXTERNAL_ID_FIELD: &str = "docid";

/// Row id -> external id for one dataset: the [`EXTERNAL_ID_FIELD`] payload
/// field when `schema` declares it (MS MARCO), else [`ExternalIds::Row`].
///
/// Only compound (`tar`) datasets carry payloads, so declaring the field on any
/// other layout is an error. `dir` is only resolved when the payloads are read.
/// Payload line `i` is corpus row `i` — the same order the uploaders assign row
/// ids in (`read_compound_data`), and `visit_compound_metadata` refuses a
/// payload file whose line count differs from `vectors.npy`'s row count.
pub fn external_ids_for_dataset(
    dataset_name: &str,
    dataset_type: Option<&str>,
    schema: Option<&serde_json::Value>,
    dir: impl FnOnce() -> Result<std::path::PathBuf, String>,
) -> Result<ExternalIds, String> {
    if schema.and_then(|s| s.get(EXTERNAL_ID_FIELD)).is_none() {
        return Ok(ExternalIds::Row);
    }
    if dataset_type != Some("tar") {
        return Err(format!(
            "dataset '{dataset_name}' declares '{EXTERNAL_ID_FIELD}' but is not a compound (tar) \
             dataset, the only layout whose payloads are read for external ids"
        ));
    }
    external_ids_from_compound(&dir()?)
}

/// Read [`EXTERNAL_ID_FIELD`] for every corpus row of a compound dataset
/// directory (`vectors.npy` + `payloads.jsonl`), in row order.
pub fn external_ids_from_compound(dir: &std::path::Path) -> Result<ExternalIds, String> {
    use crate::readers::metadata::MetadataValue;
    let mut values = Vec::new();
    crate::readers::visit_compound_metadata(dir, |item| {
        values.push(item.and_then(|m| {
            m.fields
                .into_iter()
                .find(|(k, _)| k == EXTERNAL_ID_FIELD)
                .and_then(|(_, v)| match v {
                    MetadataValue::String(s) => Some(s),
                    MetadataValue::Int(i) => Some(i.to_string()),
                    _ => None,
                })
        }));
        Ok(())
    })?;
    if values.is_empty() {
        return Err(format!(
            "{}: no payload rows to read '{EXTERNAL_ID_FIELD}' from",
            dir.display()
        ));
    }
    ExternalIds::from_field(EXTERNAL_ID_FIELD, values)
}

impl ExternalIds {
    /// Build a field map from per-row values. A row without the field is an
    /// error: silently falling back to the row id for some rows would mix two
    /// id spaces in one dump.
    pub fn from_field(field: &str, values: Vec<Option<String>>) -> Result<Self, String> {
        let mut out = Vec::with_capacity(values.len());
        for (row, v) in values.into_iter().enumerate() {
            out.push(v.ok_or_else(|| {
                format!("corpus row {row} has no '{field}' to use as its external id")
            })?);
        }
        Ok(ExternalIds::Field {
            field: field.to_string(),
            values: out,
        })
    }

    pub fn space(&self) -> IdSpace {
        match self {
            ExternalIds::Row => IdSpace::Row,
            ExternalIds::Field { field, .. } => IdSpace::Field {
                field: field.clone(),
            },
        }
    }

    pub fn resolve(&self, row: i64) -> Result<String, String> {
        if row < 0 {
            return Err(format!("negative corpus row id {row}"));
        }
        match self {
            ExternalIds::Row => Ok(row.to_string()),
            ExternalIds::Field { field, values } => {
                values.get(row as usize).cloned().ok_or_else(|| {
                    format!(
                        "corpus row id {row} is outside the {} rows that carry '{field}'",
                        values.len()
                    )
                })
            }
        }
    }
}

/// First line of every dump file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DumpHeader {
    pub schema_version: u32,
    /// Engine type (`redis`, `opensearch`, ...).
    pub engine: String,
    /// Engine configuration name (the result-file name prefix).
    pub config: String,
    pub dataset: String,
    pub workload: Workload,
    pub parallel: usize,
    /// Result limit / k the queries asked for.
    pub top: usize,
    /// Index of the search config within the engine config's `search_params`.
    pub search_id: usize,
    pub id_space: IdSpace,
}

/// One timed query, as written to the dump.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryRecord {
    pub query_id: usize,
    pub seq: usize,
    /// Query shape for per-shape reporting (e.g. a selectivity bucket). `None`
    /// groups the query under its workload. Filled by the workload that knows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    /// External ids, engine rank order.
    pub ids: Vec<String>,
    /// Engine-native scores aligned with `ids`; absent when the engine returns
    /// none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scores: Option<Vec<f64>>,
    /// Exact total, when requested.
    #[serde(default)]
    pub total: Option<u64>,
    /// Seconds.
    pub latency: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Ground-truth external ids, truncated to `top`, for workloads that have
    /// ground truth. Carried in the dump so `compare-results` scores recall
    /// against the exact neighbours the run used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Vec<String>>,
}

impl QueryRecord {
    pub fn answered(&self) -> bool {
        self.error.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DumpLine {
    Header(DumpHeader),
    Query(QueryRecord),
}

/// A parsed dump.
#[derive(Debug, Clone, PartialEq)]
pub struct Dump {
    pub header: DumpHeader,
    pub records: Vec<QueryRecord>,
}

/// Prepare `--dump-results DIR` at startup, before any download or connection:
/// create it if absent, refuse it if it holds anything. Dump files are opened
/// with `create_new`, so a reused directory would otherwise only fail after the
/// whole timed run (#343 review); and mixing two runs' dumps in one directory
/// would let `compare-results` pair records from different runs.
pub fn prepare_dump_dir(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("--dump-results: cannot create {}: {e}", dir.display()))?;
    let mut entries = std::fs::read_dir(dir)
        .map_err(|e| format!("--dump-results: cannot read {}: {e}", dir.display()))?;
    if let Some(entry) = entries.next() {
        let name = entry
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .unwrap_or_else(|e| e.to_string());
        return Err(format!(
            "--dump-results: {} is not empty (contains {name:?}); use an empty or new \
             directory per run",
            dir.display()
        ));
    }
    Ok(())
}

/// The runner's check that a capture covers every timed query: one record per
/// attempted query, failures included. A shortfall means queries vanished from
/// the dump, which is exactly what `compare-results` must be able to see.
pub fn validate_capture(records: usize, requested_queries: usize) -> Result<(), String> {
    if records != requested_queries {
        return Err(format!(
            "per-query capture recorded {records} queries but the run attempted \
             {requested_queries}; a dump that silently drops queries cannot be compared"
        ));
    }
    Ok(())
}

/// Turn raw captures into dump records: row ids -> external ids, ground truth
/// attached (`expected[query_id]`, row ids, truncated to `top`), sorted by
/// `seq`. `shape` names a query's shape, if the workload has one.
pub fn resolve_records(
    raw: &[RawQueryResult],
    ids: &ExternalIds,
    expected: Option<&[Vec<i64>]>,
    top: usize,
    shape: &dyn Fn(usize) -> Option<String>,
) -> Result<Vec<QueryRecord>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let resolved = r
            .hits
            .iter()
            .map(|h| ids.resolve(h.row))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("query {}: {e}", r.query_id))?;
        let scores = if !r.hits.is_empty() && r.hits.iter().all(|h| h.score.is_some()) {
            Some(r.hits.iter().map(|h| h.score.unwrap_or_default()).collect())
        } else {
            None
        };
        let expected = match expected {
            None => None,
            Some(gt) => {
                let row = gt.get(r.query_id).ok_or_else(|| {
                    format!(
                        "query {} has no ground-truth row ({} rows)",
                        r.query_id,
                        gt.len()
                    )
                })?;
                Some(
                    row.iter()
                        .filter(|&&id| id >= 0)
                        .take(top)
                        .map(|&id| ids.resolve(id))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| format!("ground truth of query {}: {e}", r.query_id))?,
                )
            }
        };
        out.push(QueryRecord {
            query_id: r.query_id,
            seq: r.seq,
            shape: shape(r.query_id),
            ids: resolved,
            scores,
            total: r.total,
            latency: r.latency,
            error: r.error.clone(),
            expected,
        });
    }
    out.sort_by_key(|r| r.seq);
    Ok(out)
}

/// Dump file name: one per (config, dataset, workload, parallel, search config).
pub fn dump_file_name(h: &DumpHeader) -> String {
    format!(
        "{}--{}--{}--p{}--s{}.jsonl",
        sanitize(&h.config),
        sanitize(&h.dataset),
        h.workload.as_str(),
        h.parallel,
        h.search_id
    )
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub fn write_dump(
    mut w: impl Write,
    header: &DumpHeader,
    records: &[QueryRecord],
) -> Result<(), String> {
    let mut line = |v: &DumpLine| -> Result<(), String> {
        serde_json::to_writer(&mut w, v).map_err(|e| e.to_string())?;
        w.write_all(b"\n").map_err(|e| e.to_string())
    };
    line(&DumpLine::Header(header.clone()))?;
    for r in records {
        line(&DumpLine::Query(r.clone()))?;
    }
    w.flush().map_err(|e| e.to_string())
}

pub fn read_dump(r: impl BufRead) -> Result<Dump, String> {
    let mut header = None;
    let mut records = Vec::new();
    for (n, line) in r.lines().enumerate() {
        let line = line.map_err(|e| format!("line {}: {e}", n + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: DumpLine =
            serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", n + 1))?;
        match (parsed, header.is_some()) {
            (DumpLine::Header(h), false) => {
                if h.schema_version != DUMP_SCHEMA_VERSION {
                    return Err(format!(
                        "dump schema_version {} is not the supported {DUMP_SCHEMA_VERSION}",
                        h.schema_version
                    ));
                }
                header = Some(h);
            }
            (DumpLine::Header(_), true) => {
                return Err(format!("line {}: second header", n + 1));
            }
            (DumpLine::Query(_), false) => {
                return Err(format!("line {}: query before header", n + 1));
            }
            (DumpLine::Query(q), true) => records.push(q),
        }
    }
    Ok(Dump {
        header: header.ok_or("dump has no header line")?,
        records,
    })
}

/// The first ANSWERED record per `query_id` (by `seq`), plus the ids of queries
/// that were attempted but never answered. A query cycled several times is
/// compared on its first answer.
pub fn first_answers(records: &[QueryRecord]) -> (HashMap<usize, &QueryRecord>, Vec<usize>) {
    let mut sorted: Vec<&QueryRecord> = records.iter().collect();
    sorted.sort_by_key(|r| r.seq);
    let mut answered: HashMap<usize, &QueryRecord> = HashMap::new();
    let mut attempted = std::collections::BTreeSet::new();
    for r in sorted {
        attempted.insert(r.query_id);
        if r.answered() {
            answered.entry(r.query_id).or_insert(r);
        }
    }
    let unanswered = attempted
        .into_iter()
        .filter(|q| !answered.contains_key(q))
        .collect();
    (answered, unanswered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> DumpHeader {
        DumpHeader {
            schema_version: DUMP_SCHEMA_VERSION,
            engine: "redis".into(),
            config: "redis-m-16".into(),
            dataset: "d".into(),
            workload: Workload::Vector,
            parallel: 4,
            top: 2,
            search_id: 0,
            id_space: IdSpace::Row,
        }
    }

    #[test]
    fn round_trips_through_jsonl() {
        let ids =
            ExternalIds::from_field("docid", vec![Some("a".into()), Some("b".into())]).unwrap();
        let raw = vec![
            RawQueryResult::ok(
                1,
                1,
                vec![RawHit {
                    row: 1,
                    score: Some(0.5),
                }],
                None,
                0.002,
            ),
            RawQueryResult::failed(0, 0, 0.1, "timeout"),
        ];
        let gt = vec![vec![0, 1, 1], vec![1, -1, 0]];
        let recs = resolve_records(&raw, &ids, Some(&gt), 2, &|_| None).unwrap();
        // Sorted by seq; external ids; ground truth truncated to top and -1 dropped.
        assert_eq!(recs[0].seq, 0);
        assert!(!recs[0].answered());
        assert_eq!(recs[1].ids, vec!["b"]);
        assert_eq!(recs[1].scores, Some(vec![0.5]));
        assert_eq!(recs[1].expected, Some(vec!["b".into(), "a".into()]));
        let mut buf = Vec::new();
        let mut h = header();
        h.id_space = ids.space();
        write_dump(&mut buf, &h, &recs).unwrap();
        let back = read_dump(&buf[..]).unwrap();
        assert_eq!(back.header, h);
        assert_eq!(back.records, recs);
        // Header is the first line, tagged.
        let first = String::from_utf8(buf).unwrap();
        assert!(first.starts_with("{\"kind\":\"header\""), "{first}");
    }

    #[test]
    fn external_ids_never_fall_back_to_rows() {
        assert!(ExternalIds::from_field("docid", vec![Some("a".into()), None]).is_err());
        let ids = ExternalIds::from_field("docid", vec![Some("a".into())]).unwrap();
        assert!(ids.resolve(1).is_err(), "out-of-range row must not resolve");
        assert!(ids.resolve(-1).is_err());
        assert_eq!(ExternalIds::Row.resolve(7).unwrap(), "7");
        let raw = vec![RawQueryResult::ok(
            0,
            0,
            vec![RawHit {
                row: 5,
                score: None,
            }],
            Some(1),
            0.0,
        )];
        assert!(resolve_records(&raw, &ids, None, 10, &|_| None).is_err());
    }

    #[test]
    fn filter_only_records_have_totals_and_no_scores() {
        let raw = vec![RawQueryResult::ok(
            3,
            0,
            vec![RawHit {
                row: 9,
                score: None,
            }],
            Some(42),
            0.001,
        )];
        let recs =
            resolve_records(&raw, &ExternalIds::Row, None, 10, &|_| Some("tag".into())).unwrap();
        assert_eq!(recs[0].total, Some(42));
        assert_eq!(recs[0].scores, None);
        assert_eq!(recs[0].expected, None);
        assert_eq!(recs[0].shape.as_deref(), Some("tag"));
    }

    /// A tiny compound fixture whose `docid`s are NOT the row ids, so an
    /// off-by-one in the row -> docid mapping cannot pass by coincidence.
    fn compound_fixture(docids: &[Option<&str>]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let vectors: Vec<Vec<f32>> = (0..docids.len()).map(|i| vec![i as f32, 1.0]).collect();
        crate::readers::write_npy_vectors(
            dir.path().join("vectors.npy").to_str().unwrap(),
            &vectors,
        )
        .unwrap();
        let payloads: Vec<String> = docids
            .iter()
            .map(|d| match d {
                Some(d) => serde_json::json!({"docid": d, "tag": "x"}).to_string(),
                None => serde_json::json!({"tag": "x"}).to_string(),
            })
            .collect();
        std::fs::write(dir.path().join("payloads.jsonl"), payloads.join("\n")).unwrap();
        dir
    }

    #[test]
    fn docid_external_ids_follow_corpus_row_order() {
        let fx = compound_fixture(&[Some("doc-c"), Some("doc-a"), Some("doc-b")]);
        let schema = serde_json::json!({"docid": "keyword", "tag": "keyword"});
        let ids = external_ids_for_dataset("fx", Some("tar"), Some(&schema), || {
            Ok(fx.path().to_path_buf())
        })
        .unwrap();
        assert_eq!(
            ids.space(),
            IdSpace::Field {
                field: "docid".into()
            }
        );
        // Row i is payload line i — exactly, not shifted by one either way.
        assert_eq!(ids.resolve(0).unwrap(), "doc-c");
        assert_eq!(ids.resolve(1).unwrap(), "doc-a");
        assert_eq!(ids.resolve(2).unwrap(), "doc-b");
        assert!(ids.resolve(3).is_err());
        // End to end through resolve_records: ids and ground truth both map.
        let raw = vec![RawQueryResult::ok(
            0,
            0,
            vec![
                RawHit {
                    row: 2,
                    score: Some(0.1),
                },
                RawHit {
                    row: 0,
                    score: Some(0.2),
                },
            ],
            None,
            0.001,
        )];
        let gt = vec![vec![1, 2]];
        let recs = resolve_records(&raw, &ids, Some(&gt), 2, &|_| None).unwrap();
        assert_eq!(recs[0].ids, vec!["doc-b", "doc-c"]);
        assert_eq!(recs[0].expected, Some(vec!["doc-a".into(), "doc-b".into()]));
    }

    #[test]
    fn docid_external_ids_fail_closed() {
        let schema = serde_json::json!({"docid": "keyword"});
        // A row without a docid is an error, never a fallback to its row id.
        let fx = compound_fixture(&[Some("d0"), None]);
        assert!(
            external_ids_for_dataset("fx", Some("tar"), Some(&schema), || {
                Ok(fx.path().to_path_buf())
            })
            .is_err()
        );
        // Declared on a layout without payloads.
        assert!(
            external_ids_for_dataset("fx", Some("hdf5"), Some(&schema), || unreachable!()).is_err()
        );
        // Not declared: row ids, and the corpus is never read.
        let ids = external_ids_for_dataset(
            "fx",
            Some("tar"),
            Some(&serde_json::json!({"tag": "keyword"})),
            || unreachable!(),
        )
        .unwrap();
        assert_eq!(ids.space(), IdSpace::Row);
        // Payload count must match the vector rows (no silent shift).
        let fx = compound_fixture(&[Some("d0"), Some("d1")]);
        std::fs::write(fx.path().join("payloads.jsonl"), "{\"docid\":\"d0\"}").unwrap();
        assert!(external_ids_from_compound(fx.path()).is_err());
    }

    #[test]
    fn dump_dir_must_be_new_or_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let fresh = tmp.path().join("a/b");
        assert!(prepare_dump_dir(&fresh).is_ok(), "absent dir is created");
        assert!(fresh.is_dir());
        assert!(prepare_dump_dir(&fresh).is_ok(), "empty dir is accepted");
        std::fs::write(fresh.join("old.jsonl"), "x").unwrap();
        let e = prepare_dump_dir(&fresh).unwrap_err();
        assert!(e.contains("not empty") && e.contains("old.jsonl"), "{e}");
        // Any entry counts, not just *.jsonl: a subdirectory is refused too.
        let other = tmp.path().join("c");
        std::fs::create_dir_all(other.join("sub")).unwrap();
        assert!(prepare_dump_dir(&other).is_err());
    }

    #[test]
    fn capture_must_cover_every_attempted_query() {
        assert!(validate_capture(5, 5).is_ok());
        assert!(validate_capture(4, 5).is_err());
        assert!(validate_capture(6, 5).is_err());
    }

    #[test]
    fn malformed_dumps_are_rejected() {
        let q = r#"{"kind":"query","query_id":0,"seq":0,"ids":[],"latency":0.1}"#;
        assert!(read_dump(q.as_bytes()).is_err(), "query before header");
        assert!(read_dump("".as_bytes()).is_err(), "no header");
        let h = serde_json::to_string(&DumpLine::Header(header())).unwrap();
        assert!(read_dump(format!("{h}\n{h}\n").as_bytes()).is_err());
        let mut old = header();
        old.schema_version = 0;
        let h0 = serde_json::to_string(&DumpLine::Header(old)).unwrap();
        assert!(read_dump(h0.as_bytes()).is_err());
        assert_eq!(
            read_dump(format!("{h}\n{q}\n").as_bytes())
                .unwrap()
                .records
                .len(),
            1
        );
    }

    #[test]
    fn first_answer_wins_and_unanswered_are_listed() {
        let rec = |q, seq, ok: bool| QueryRecord {
            query_id: q,
            seq,
            shape: None,
            ids: vec![format!("{seq}")],
            scores: None,
            total: None,
            latency: 0.0,
            error: (!ok).then(|| "x".to_string()),
            expected: None,
        };
        let recs = vec![
            rec(0, 3, true),
            rec(0, 1, false),
            rec(0, 2, true),
            rec(1, 0, false),
        ];
        let (ans, un) = first_answers(&recs);
        assert_eq!(ans[&0].seq, 2);
        assert_eq!(un, vec![1]);
    }

    #[test]
    fn file_names_are_distinct_per_parallel_and_search_config() {
        let a = dump_file_name(&header());
        let mut h = header();
        h.parallel = 8;
        assert_ne!(a, dump_file_name(&h));
        h.parallel = 4;
        h.search_id = 1;
        assert_ne!(a, dump_file_name(&h));
        h.config = "a/b".into();
        assert!(!dump_file_name(&h).contains('/'));
    }
}
