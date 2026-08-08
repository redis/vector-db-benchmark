use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;
use std::time::Instant;

use crate::config::RedisConfig;
use crate::redis_client::create_connection;

static SEARCH_HOST: Mutex<Option<String>> = Mutex::new(None);
static SEARCH_PARAMS: Mutex<Option<SearchConfig>> = Mutex::new(None);

#[derive(Clone)]
struct SearchConfig {
    ef: i64,
    parallel: i64,
    top: Option<usize>,
}

/// A query with its vector bytes and expected result for precision calculation.
struct RustQuery {
    vector: Vec<u8>,
    expected_result: Option<Vec<i64>>,
    top: usize,
}

/// Per-query result: (precision, recall, mrr, ndcg, latency)
type SearchResult = (f64, f64, f64, f64, f64);

#[pyclass]
pub struct RustVsetSearcher {
    host: String,
    #[pyo3(get)]
    connection_params: PyObject,
    #[pyo3(get)]
    search_params: PyObject,
}

#[pymethods]
impl RustVsetSearcher {
    #[new]
    fn new(host: String, connection_params: PyObject, search_params: PyObject) -> Self {
        Self {
            host,
            connection_params,
            search_params,
        }
    }

    #[classmethod]
    fn init_client(
        _cls: &Bound<'_, pyo3::types::PyType>,
        host: String,
        _distance: &Bound<'_, PyAny>,
        _connection_params: &Bound<'_, PyAny>,
        search_params: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let parallel: i64 = search_params
            .call_method1("get", ("parallel", 1i64))?
            .extract()?;
        let top: Option<usize> = search_params
            .call_method1("get", ("top", Python::with_gil(|py| py.None())))?
            .extract()?;

        let inner_params = search_params
            .call_method1("get", ("search_params", Python::with_gil(|py| py.None())))?;
        let ef: i64 = if !inner_params.is_none() {
            inner_params.call_method1("get", ("ef", 10i64))?.extract()?
        } else {
            10
        };

        *SEARCH_HOST.lock().unwrap() = Some(host);
        *SEARCH_PARAMS.lock().unwrap() = Some(SearchConfig { ef, parallel, top });
        Ok(())
    }

    #[classmethod]
    fn search_one(
        _cls: &Bound<'_, pyo3::types::PyType>,
        vector: &Bound<'_, PyAny>,
        _meta_conditions: &Bound<'_, PyAny>,
        top: i64,
    ) -> PyResult<Vec<(i64, f64)>> {
        let vec_bytes: Vec<u8> = vector.extract()?;
        let host_guard = SEARCH_HOST.lock().unwrap();
        let host = host_guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("init_client not called"))?;
        let params_guard = SEARCH_PARAMS.lock().unwrap();
        let params = params_guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("init_client not called"))?;

        let config = RedisConfig::from_env();
        let mut conn = create_connection(host, &config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
        })?;

        let response: Vec<redis::Value> = redis::cmd("VSIM")
            .arg("idx")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg("WITHSCORES")
            .arg("COUNT")
            .arg(top)
            .arg("EF")
            .arg(params.ef)
            .query(&mut conn)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("VSIM error: {}", e)))?;

        Ok(parse_vsim_response(&response))
    }

    /// Run the full search loop in Rust, bypassing the GIL for maximum concurrency.
    /// This replaces BaseSearcher.search_all() for the vectorsets engine.
    fn search_all(
        &self,
        _distance: &Bound<'_, PyAny>,
        queries: &Bound<'_, PyAny>,
        num_queries: i64,
    ) -> PyResult<PyObject> {
        // Extract search params from self
        let search_params = Python::with_gil(|py| -> PyResult<SearchConfig> {
            let sp = self.search_params.bind(py);
            let parallel: i64 = sp.call_method1("get", ("parallel", 1i64))?.extract()?;
            let top: Option<usize> = sp.call_method1("get", ("top", py.None()))?.extract()?;
            let inner_params = sp.call_method1("get", ("search_params", py.None()))?;
            let ef: i64 = if !inner_params.is_none() {
                inner_params.call_method1("get", ("ef", 10i64))?.extract()?
            } else {
                10
            };
            Ok(SearchConfig { ef, parallel, top })
        })?;

        // Convert Python queries to Rust structs
        let default_top: usize = 10;
        let rust_queries: Vec<RustQuery> = Python::with_gil(|_py| -> PyResult<Vec<RustQuery>> {
            let queries_iter = queries.call_method0("__iter__")?;
            let mut qs = Vec::new();
            loop {
                match queries_iter.call_method0("__next__") {
                    Ok(query) => {
                        // Vector may be List[float] or bytes — convert to f32 bytes
                        let vector_obj = query.getattr("vector")?;
                        let vector: Vec<u8> = if let Ok(bytes) = vector_obj.extract::<Vec<u8>>() {
                            bytes
                        } else {
                            let floats: Vec<f32> = vector_obj.extract()?;
                            floats.iter().flat_map(|f| f.to_le_bytes()).collect()
                        };

                        let expected_obj = query.getattr("expected_result")?;
                        let expected_result: Option<Vec<i64>> = if expected_obj.is_none() {
                            None
                        } else {
                            // Handle numpy int32/int64 arrays by extracting as Python ints
                            let py_list = expected_obj
                                .call_method0("tolist")
                                .unwrap_or_else(|_| expected_obj.clone());
                            Some(py_list.extract()?)
                        };

                        let top = search_params.top.unwrap_or_else(|| {
                            expected_result
                                .as_ref()
                                .map(|e| if e.is_empty() { default_top } else { e.len() })
                                .unwrap_or(default_top)
                        });

                        qs.push(RustQuery {
                            vector,
                            expected_result,
                            top,
                        });
                    }
                    Err(_) => break,
                }
            }
            Ok(qs)
        })?;

        // Handle num_queries
        let used_queries: Vec<&RustQuery> = if num_queries > 0 {
            let n = num_queries as usize;
            if n > rust_queries.len() && !rust_queries.is_empty() {
                (0..n)
                    .map(|i| &rust_queries[i % rust_queries.len()])
                    .collect()
            } else {
                rust_queries.iter().take(n).collect()
            }
        } else {
            rust_queries.iter().collect()
        };

        let host = self.host.clone();
        let parallel = search_params.parallel;

        // Measure wall-clock time for the entire batch
        let start = Instant::now();
        // Release GIL and run the search loop in Rust
        let results: Vec<SearchResult> = Python::with_gil(|py| -> PyResult<Vec<SearchResult>> {
            py.allow_threads(|| {
                if parallel <= 1 {
                    run_sequential(&host, &search_params, &used_queries)
                } else {
                    run_parallel(&host, &search_params, &used_queries, parallel as usize)
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })?;
        let total_time = start.elapsed().as_secs_f64();

        // Compute stats and return as Python dict
        Python::with_gil(|py| -> PyResult<PyObject> {
            let precisions: Vec<f64> = results.iter().map(|r| r.0).collect();
            let recalls: Vec<f64> = results.iter().map(|r| r.1).collect();
            let mrrs: Vec<f64> = results.iter().map(|r| r.2).collect();
            let ndcgs: Vec<f64> = results.iter().map(|r| r.3).collect();
            let latencies: Vec<f64> = results.iter().map(|r| r.4).collect();

            let n = precisions.len() as f64;
            let mean_precision = if n > 0.0 {
                precisions.iter().sum::<f64>() / n
            } else {
                0.0
            };
            let mean_recall = if n > 0.0 {
                recalls.iter().sum::<f64>() / n
            } else {
                0.0
            };
            let mean_mrr = if n > 0.0 {
                mrrs.iter().sum::<f64>() / n
            } else {
                0.0
            };
            let mean_ndcg = if n > 0.0 {
                ndcgs.iter().sum::<f64>() / n
            } else {
                0.0
            };
            let mean_time = if latencies.is_empty() {
                0.0
            } else {
                latencies.iter().sum::<f64>() / latencies.len() as f64
            };

            let std_time = if latencies.len() > 1 {
                let variance = latencies
                    .iter()
                    .map(|l| (l - mean_time).powi(2))
                    .sum::<f64>()
                    / latencies.len() as f64;
                variance.sqrt()
            } else {
                0.0
            };

            let min_time = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_time = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let rps = if total_time > 0.0 {
                latencies.len() as f64 / total_time
            } else {
                0.0
            };

            let p50 = percentile(&latencies, 50.0);
            let p95 = percentile(&latencies, 95.0);
            let p99 = percentile(&latencies, 99.0);

            let dict = PyDict::new_bound(py);
            dict.set_item("total_time", total_time)?;
            dict.set_item("mean_time", mean_time)?;
            // NOTE (#217): this legacy PyO3 accelerator (not part of the Rust
            // binary and not compiled today) fills Python v0's `mean_precisions`
            // key with OUR precision (hits / results returned), while pure-Python
            // v0 fills it with recall@top (hits / top). If this path is ever
            // revived, feed it `mean_recall` — or rename the key — before any of
            // its output is compared with upstream numbers.
            dict.set_item("mean_precisions", mean_precision)?;
            dict.set_item("mean_recall", mean_recall)?;
            dict.set_item("mean_mrr", mean_mrr)?;
            dict.set_item("mean_ndcg", mean_ndcg)?;
            dict.set_item("std_time", std_time)?;
            dict.set_item("min_time", min_time)?;
            dict.set_item("max_time", max_time)?;
            dict.set_item("rps", rps)?;
            dict.set_item("p50_time", p50)?;
            dict.set_item("p95_time", p95)?;
            dict.set_item("p99_time", p99)?;
            dict.set_item("precisions", precisions)?;
            dict.set_item("recalls", recalls)?;
            dict.set_item("mrrs", mrrs)?;
            dict.set_item("ndcgs", ndcgs)?;
            dict.set_item("latencies", latencies)?;

            Ok(dict.into())
        })
    }

    fn setup_search(&self) -> PyResult<()> {
        Ok(())
    }

    fn post_search(&self) -> PyResult<()> {
        Ok(())
    }

    #[classmethod]
    fn delete_client(_cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<()> {
        *SEARCH_HOST.lock().unwrap() = None;
        *SEARCH_PARAMS.lock().unwrap() = None;
        Ok(())
    }
}

fn parse_vsim_response(response: &[redis::Value]) -> Vec<(i64, f64)> {
    let mut results = Vec::new();
    let mut i = 0;
    while i + 1 < response.len() {
        let id = match &response[i] {
            redis::Value::BulkString(data) => {
                String::from_utf8_lossy(data).parse::<i64>().unwrap_or(0)
            }
            redis::Value::Int(n) => *n,
            _ => 0,
        };
        let score = match &response[i + 1] {
            redis::Value::BulkString(data) => {
                let s = String::from_utf8_lossy(data);
                1.0 - s.parse::<f64>().unwrap_or(0.0)
            }
            redis::Value::Double(f) => 1.0 - f,
            _ => 1.0,
        };
        results.push((id, score));
        i += 2;
    }
    results
}

fn search_one_rust(conn: &mut redis::Connection, query: &RustQuery, ef: i64) -> SearchResult {
    let start = Instant::now();

    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("idx")
        .arg("FP32")
        .arg(&query.vector[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(query.top as i64)
        .arg("EF")
        .arg(ef)
        .query(conn)
        .unwrap_or_default();

    let elapsed = start.elapsed().as_secs_f64();

    let search_results = parse_vsim_response(&response);

    let m = if let Some(expected) = &query.expected_result {
        let ordered_ids: Vec<i64> = search_results.iter().map(|(id, _)| *id).collect();
        crate::metrics::compute_metrics(&ordered_ids, expected, query.top)
    } else {
        crate::metrics::QueryMetrics {
            recall: 1.0,
            precision: 1.0,
            recall_at_top: 1.0,
            mrr: 1.0,
            ndcg: 1.0,
        }
    };

    (m.precision, m.recall, m.mrr, m.ndcg, elapsed)
}

fn run_sequential(
    host: &str,
    params: &SearchConfig,
    queries: &[&RustQuery],
) -> Result<Vec<SearchResult>, String> {
    let config = RedisConfig::from_env();
    let mut conn = create_connection(host, &config).map_err(|e| format!("Redis error: {}", e))?;

    let results: Vec<SearchResult> = queries
        .iter()
        .map(|q| search_one_rust(&mut conn, q, params.ef))
        .collect();

    Ok(results)
}

fn run_parallel(
    host: &str,
    params: &SearchConfig,
    queries: &[&RustQuery],
    parallel: usize,
) -> Result<Vec<SearchResult>, String> {
    let chunk_size = std::cmp::max(1, queries.len() / parallel);
    let chunks: Vec<&[&RustQuery]> = queries.chunks(chunk_size).collect();

    // Use scoped threads so we can borrow queries
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let host = host.to_string();
                let ef = params.ef;
                s.spawn(move || -> Result<Vec<SearchResult>, String> {
                    let config = RedisConfig::from_env();
                    let mut conn = create_connection(&host, &config)
                        .map_err(|e| format!("Redis error: {}", e))?;
                    Ok(chunk
                        .iter()
                        .map(|q| search_one_rust(&mut conn, q, ef))
                        .collect())
                })
            })
            .collect();

        let mut all_results = Vec::with_capacity(queries.len());
        for handle in handles {
            match handle.join() {
                Ok(Ok(chunk_results)) => all_results.extend(chunk_results),
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err("Thread panicked".to_string()),
            }
        }
        Ok(all_results)
    })
}

fn percentile(data: &[f64], pct: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
