//! One ingest, many workloads (#344): a fixture whose every test row serves
//! the vector, filter-only and fulltext workloads, and the assertions both
//! engine suites make about a run of all three against one shared index.
//!
//! The fixture is built so the shared `full` layout is observable from the
//! results, not just from the schema:
//!
//! - `color` holds `red` and `Red`: an exact (case-sensitive) tag matches one
//!   or the other, a case-insensitive one both — so filter-only totals AND the
//!   filtered vector ground truth are only right on a `CASESENSITIVE` TAG.
//! - `price` holds 1.0, 1.00000001 and 1.00000002, which a 32-bit `float`
//!   mapping collapses into one value; the range predicate on it only counts
//!   correctly on a `double` mapping.
//! - `title` is a text field the fulltext configs search, which the
//!   filter-only contract stores but never queries.

use std::fs;
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use vector_db_benchmark::readers::write_npy_vectors;

pub const DATASET: &str = "shared-index-test";
pub const DOCS: usize = 400;
const DIM: usize = 4;
const QUERIES: usize = 12;
pub const VECTOR_TOP: usize = 10;
pub const FILTER_TOP: usize = 5;
const WORDS: [&str; 4] = ["alpha", "bravo", "charlie", "delta"];
const PRICES: [f64; 3] = [1.0, 1.000_000_01, 1.000_000_02];

fn color(id: usize) -> &'static str {
    match id % 7 {
        0 | 1 => "red",
        2 => "Red",
        _ => "blue",
    }
}

fn size(id: usize) -> i64 {
    (id % 11) as i64 + 1
}

/// The predicates, each with an independent oracle over document ids.
#[allow(clippy::type_complexity)]
fn predicates() -> Vec<(Value, fn(usize) -> bool)> {
    vec![
        (json!({"and":[{"color":{"match":{"value":"red"}}}]}), |id| {
            color(id) == "red"
        }),
        (json!({"and":[{"color":{"match":{"value":"Red"}}}]}), |id| {
            color(id) == "Red"
        }),
        (
            json!({"and":[{"size":{"range":{"gte":2,"lte":5}}}]}),
            |id| (2..=5).contains(&size(id)),
        ),
        (
            json!({"and":[{"color":{"match":{"value":"red"}}},{"size":{"range":{"gte":2}}}]}),
            |id| color(id) == "red" && size(id) >= 2,
        ),
        (
            json!({"or":[{"color":{"match":{"value":"Red"}}},{"size":{"match":{"value":2}}}]}),
            |id| color(id) == "Red" || size(id) == 2,
        ),
        (
            json!({"and":[{"price":{"range":{"gt":1.0,"lte":1.000000015}}}]}),
            |id| id % 3 == 1,
        ),
    ]
}

pub struct SharedIndexProject {
    pub root: PathBuf,
    /// Sum over the test rows of each predicate's exact match count.
    pub matched_sum: u64,
    /// Same, capped at `FILTER_TOP` per query.
    pub returned_sum: u64,
}

/// The three search configs one engine config runs against one index.
pub fn search_params() -> Value {
    json!([
        {"parallel": 1, "top": VECTOR_TOP},
        {"parallel": 2, "top": FILTER_TOP, "workload": "filter-only"},
        {"parallel": 1, "top": 5, "workload": "fulltext"}
    ])
}

pub fn write_project(engine_configs_json: &str) -> SharedIndexProject {
    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);
    let ds = root.join("datasets").join(DATASET);
    fs::create_dir_all(&ds).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    let mut rng = StdRng::seed_from_u64(0x0344);
    let mut vec4 = || -> Vec<f32> { (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..DOCS).map(|_| vec4()).collect();
    let queries: Vec<Vec<f32>> = (0..QUERIES).map(|_| vec4()).collect();
    write_npy_vectors(ds.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    let payloads = (0..DOCS)
        .map(|id| {
            json!({
                "docid": format!("d{id}"),
                "color": color(id),
                "size": size(id),
                "price": PRICES[id % 3],
                "title": format!("{} story", WORDS[id % WORDS.len()]),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds.join("payloads.jsonl"), payloads).unwrap();

    let preds = predicates();
    let l2 = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum()
    };
    let (mut matched_sum, mut returned_sum) = (0u64, 0u64);
    let rows = queries
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let (conditions, oracle) = &preds[i % preds.len()];
            let mut hits: Vec<(usize, f64)> = (0..DOCS)
                .filter(|id| oracle(*id))
                .map(|id| (id, l2(q, &vectors[id])))
                .collect();
            matched_sum += hits.len() as u64;
            returned_sum += hits.len().min(FILTER_TOP) as u64;
            hits.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: Vec<usize> = hits.iter().take(VECTOR_TOP).map(|(id, _)| *id).collect();
            json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": conditions,
                "closest_ids": gt,
                "query_text": WORDS[i % WORDS.len()],
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds.join("tests.jsonl"), rows).unwrap();

    let registry = json!([{
        "name": DATASET, "type": "tar", "path": format!("{DATASET}/"),
        "distance": "l2", "vector_size": DIM, "vector_count": DOCS,
        "schema": {"docid": "keyword", "color": "keyword", "size": "int",
                   "price": "float", "title": "text"},
    }]);
    fs::write(root.join("datasets/datasets.json"), registry.to_string()).unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();
    SharedIndexProject {
        root,
        matched_sum,
        returned_sum,
    }
}

fn result_docs(root: &Path, config: &str, kind: &str) -> Vec<Value> {
    fs::read_dir(root.join("results"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            n.starts_with(&format!("{config}-{DATASET}-{kind}-"))
        })
        .map(|p| serde_json::from_str(&fs::read_to_string(p).unwrap()).unwrap())
        .collect()
}

fn f(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("expected a number, got {v}"))
}

/// Every claim a run of [`search_params`] makes, on either engine:
///
/// - ONE upload file, naming the `full` layout and the three workloads it
///   serves, with ingest phases that sum to the wall time between their
///   timestamped boundaries and agree with the legacy keys;
/// - three search files, one per workload, each on the `full` layout;
/// - vector: filtered recall against exact (case-sensitive, binary64) truth;
/// - filter-only: exact totals, the `full` layout in its stats;
/// - fulltext: its fingerprint;
/// - the summary lists the mix and keeps vector quality on the vector row.
pub fn assert_run(proj: &SharedIndexProject, config: &str, index_wait: &str) {
    let uploads = result_docs(&proj.root, config, "upload");
    assert_eq!(uploads.len(), 1, "one ingest for all three workloads");
    let up = &uploads[0];
    assert_eq!(up["params"]["index_layout"], "full", "{up}");
    assert_eq!(
        up["params"]["index_workloads"],
        json!(["vector", "filter-only", "fulltext"])
    );
    let r = &up["results"];
    assert_eq!(r["upload_count"], DOCS);
    let (upload, wait, total) = (
        f(&r["upload_time"]),
        f(&r["index_wait_time"]),
        f(&r["total_ingest_time"]),
    );
    assert!(upload > 0.0 && wait >= 0.0, "{r}");
    assert!((upload + wait - total).abs() < 1e-3, "{r}");
    let b = &r["ingest_phases"];
    assert_eq!(b["index_wait"], index_wait);
    assert_eq!(b["documents"], DOCS);
    let wall = f(&b["index_ready_at"]) - f(&b["upload_started_at"]);
    assert!((wall - total).abs() < 1e-3, "{b}");
    assert!(
        (f(&b["upload_finished_at"]) - f(&b["upload_started_at"]) - upload).abs() < 1e-3,
        "{b}"
    );
    assert!(
        (f(&r["total_time"]) - f(&r["read_time"]) - total).abs() < 1e-3,
        "legacy total_time = read + upload + index wait: {r}"
    );
    assert!(f(&r["ingest_docs_per_sec"]) > 0.0);

    let searches = result_docs(&proj.root, config, "search");
    assert_eq!(searches.len(), 3, "one result file per search config");
    let by = |w: &str| {
        let found: Vec<&Value> = searches
            .iter()
            .filter(|d| d["params"]["workload"] == w)
            .collect();
        assert_eq!(found.len(), 1, "{w}");
        found[0]
    };
    for doc in &searches {
        assert_eq!(doc["params"]["index_layout"], "full", "{}", doc["params"]);
        assert_eq!(doc["results"]["failed_queries"], 0);
    }
    let vector = &by("vector")["results"];
    assert!(
        f(&vector["mean_recall"]) >= 0.9,
        "filtered recall against exact ground truth on the full index: {vector}"
    );
    let fo = &by("filter-only")["results"];
    assert_eq!(fo["filter_only"]["index_layout"], "full");
    assert_eq!(fo["filter_only"]["result_limit"], FILTER_TOP);
    assert_eq!(fo["filter_only"]["matched_documents_sum"], proj.matched_sum);
    assert_eq!(
        fo["filter_only"]["returned_documents_sum"],
        proj.returned_sum
    );
    assert!(fo.get("mean_recall").is_none());
    let ft = by("fulltext");
    assert!(ft["params"]["fulltext"].is_object(), "{}", ft["params"]);
    assert!(ft["results"].get("mean_recall").is_none());

    let summary: Value = serde_json::from_str(
        &fs::read_to_string(
            proj.root
                .join("results")
                .join(format!("{config}-{DATASET}-summary.json")),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        summary["workloads"],
        json!(["vector", "filter-only", "fulltext"])
    );
    assert!(summary.get("workload").is_none());
    let rows = summary["search_results"].as_array().unwrap();
    let vrow = rows.iter().find(|r| r["workload"] == "vector").unwrap();
    assert!(vrow.get("mean_recall").is_some());
}

/// #344 review: filter-only's metadata domain is checked before the upload on
/// the shared `full` index too. One corpus row gets a `;` in its keyword tag
/// (outside the domain: it would split into two RediSearch tags); the mixed
/// config, vector search config first, must fail with the domain error before
/// any upload or search result is written.
pub fn assert_bad_metadata_refused_before_upload(
    proj: &SharedIndexProject,
    config: &str,
    host: &str,
    envs: &[(&str, &str)],
) {
    poison_keyword_token(proj);
    assert_refused_on_bad_metadata(proj, config, host, envs);
}

/// Give one corpus row a `;` in its keyword tag: outside filter-only's
/// metadata domain (it would split into two RediSearch tags), but uploadable
/// by a `vector`-layout config, which does not check that domain.
pub fn poison_keyword_token(proj: &SharedIndexProject) {
    let payloads = proj
        .root
        .join("datasets")
        .join(DATASET)
        .join("payloads.jsonl");
    let text = fs::read_to_string(&payloads).unwrap();
    let mut rows: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    rows[0]["color"] = json!("re;d");
    let text: Vec<String> = rows.iter().map(Value::to_string).collect();
    fs::write(&payloads, text.join("\n")).unwrap();
}

/// Run `config` over the (poisoned) corpus: it must fail with the domain error
/// before any upload or search result file is written.
pub fn assert_refused_on_bad_metadata(
    proj: &SharedIndexProject,
    config: &str,
    host: &str,
    envs: &[(&str, &str)],
) {
    let out = super::run_binary_output(
        &proj.root,
        config,
        DATASET,
        host,
        envs,
        &["--fail-on-dropped-queries", "--keep-data"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(
        stderr.contains("filter-only field \"color\": keyword tokens must be nonempty"),
        "{stderr}"
    );
    let written: Vec<_> = fs::read_dir(proj.root.join("results"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.to_string_lossy();
            name.contains("-upload-") || name.contains("-search-")
        })
        .collect();
    assert!(
        written.is_empty(),
        "refused before the upload, so no result file: {written:?}"
    );
}

/// Vector-only search configs: a config with only these plans the `vector`
/// layout.
pub fn vector_only() -> Value {
    json!([{"parallel": 1, "top": VECTOR_TOP}])
}

/// #344, `--skip-upload` reuse: build `engine_config`'s index with the `build`
/// search configs (`--skip-search --keep-data`), then rerun the SAME config
/// with the `reuse` search configs over that index. The rerun must fail with
/// `expected` before ANY search result is written — so the realized layout is
/// checked once, up front, not by whichever search config happens to run
/// first.
#[allow(clippy::too_many_arguments)]
pub fn assert_reuse_refused(
    proj: &SharedIndexProject,
    engine_config: &Value,
    host: &str,
    envs: &[(&str, &str)],
    build: Value,
    reuse: Value,
    expected: &str,
) {
    let config = engine_config["name"].as_str().unwrap();
    let write = |search_params: Value| {
        let mut c = engine_config.clone();
        c["search_params"] = search_params;
        fs::write(
            proj.root.join("experiments/configurations/test.json"),
            json!([c]).to_string(),
        )
        .unwrap();
    };
    write(build);
    assert!(super::run_binary_extra(
        &proj.root,
        config,
        DATASET,
        host,
        envs,
        &["--skip-search", "--keep-data"],
    ));
    super::clear_result_files(&proj.root);
    write(reuse);
    let out = super::run_binary_output(
        &proj.root,
        config,
        DATASET,
        host,
        envs,
        &["--skip-upload", "--keep-data", "--fail-on-dropped-queries"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains(expected), "expected {expected:?}: {stderr}");
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("Experiment stage: Search"),
        "refused before the search stage"
    );
    let written: Vec<_> = fs::read_dir(proj.root.join("results"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().contains("-search-"))
        .collect();
    assert!(written.is_empty(), "no search result written: {written:?}");
}
