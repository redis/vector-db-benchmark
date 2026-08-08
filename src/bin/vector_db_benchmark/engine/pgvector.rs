//! PgVector engine implementation.
//!
//! Uses the `postgres` crate for PostgreSQL connectivity with the pgvector extension.
//! Supports HNSW index with configurable m/ef_construction, COPY bulk upload,
//! and distance operators <-> (L2) and <=> (cosine).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use postgres::types::ToSql;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{CorpusCount, Engine, SearchResults, UploadStats};
use vector_db_benchmark::query_filter::QueryFilter;
use vector_db_benchmark::readers::metadata::{
    is_multivalued_keyword_field, MetadataItem, MetadataValue,
};
use vector_db_benchmark::start_gate::WorkerPool;

/// Warn when `parallel` cannot fit in the server's connection budget.
///
/// Best effort: any failure to read the budget is silently ignored — this must
/// never be the thing that stops a run.
fn warn_if_over_connection_budget(conn_str: &str, parallel: usize) {
    let Ok(mut client) = postgres::Client::connect(conn_str, postgres::NoTls) else {
        return;
    };
    let scalar = |client: &mut postgres::Client, sql: &str| -> Option<i64> {
        let row = client.query_one(sql, &[]).ok()?;
        row.try_get::<_, String>(0).ok()?.parse().ok()
    };
    let Some(max_conns) = scalar(&mut client, "SHOW max_connections") else {
        return;
    };
    let reserved = scalar(&mut client, "SHOW superuser_reserved_connections").unwrap_or(0);
    let in_use: i64 = client
        .query_one("SELECT count(*) FROM pg_stat_activity", &[])
        .ok()
        .and_then(|r| r.try_get(0).ok())
        .unwrap_or(0);
    // `in_use` counts this advisory connection, which is closed on return.
    let available = (max_conns - reserved - in_use + 1).max(0);
    if (parallel as i64) > available {
        eprintln!(
            "\t⚠ WARNING: parallel={parallel} exceeds this server's connection budget \
             (max_connections={max_conns}, superuser_reserved={reserved}, {in_use} in use \
             → {available} available). Each search worker holds one connection for the whole \
             run, so the run will FAIL rather than quietly measure fewer workers. Raise \
             max_connections or lower parallel."
        );
    }
}

/// Render a `postgres::Error` usefully.
///
/// `postgres::Error`'s own `Display` is the literal string `"db error"`; the
/// server's message — `FATAL: sorry, too many clients already`, the single most
/// likely reason a worker cannot connect at high `parallel` — lives in the
/// `DbError` behind it. Without this, the whole diagnostic for a `parallel` that
/// exceeds `max_connections` reads `36 failed setup: ... db error`.
fn describe_pg_error(e: &postgres::Error) -> String {
    if let Some(db) = e.as_db_error() {
        let mut msg = format!("{}: {}", db.severity(), db.message());
        if let Some(hint) = db.hint() {
            msg.push_str(&format!(" (hint: {hint})"));
        }
        return msg;
    }
    // Not a server-side error (TLS, DNS, refused socket): walk the source chain,
    // whose leaf carries the real cause.
    let mut msg = e.to_string();
    let mut src: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(e);
    while let Some(cause) = src {
        msg.push_str(&format!(": {cause}"));
        src = cause.source();
    }
    msg
}

/// Map a dataset schema field type to a Postgres column type. Returns None for
/// types pgvector can't filter on with a plain scalar column (e.g. geo).
fn pg_column_type(field_type: &str) -> Option<&'static str> {
    match field_type {
        // A `uuid` is an exact-match opaque string; store it in a TEXT column
        // (exact `=` match, like keyword). Without this arm it hit `_ => None`,
        // so no column was created and every uuid filter silently broke.
        "keyword" | "text" | "uuid" => Some("TEXT"),
        "int" => Some("BIGINT"),
        "float" => Some("DOUBLE PRECISION"),
        // Bools/datetimes arrive from the reader as strings ("true"/"false",
        // ISO-8601). COPY-text coerces those into a BOOLEAN column and parses
        // ISO into a TIMESTAMPTZ column, so a plain scalar column filters fine.
        "bool" => Some("BOOLEAN"),
        "datetime" => Some("TIMESTAMPTZ"),
        // Geo point stored as a "lat,lon" TEXT scalar; filtered with the
        // earthdistance extension (great-circle metres). A dedicated earth/cube
        // column would need binary COPY, so the text form keeps the COPY path.
        "geo" => Some("TEXT"),
        _ => None,
    }
}

/// Escape a string for a COPY ... FORMAT text field.
fn escape_copy_text(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Format the value of `column` from a row's metadata as a COPY text field,
/// returning `\N` (NULL) when the field is absent or unsupported.
fn copy_field(meta: Option<&MetadataItem>, column: &str) -> String {
    let value = meta.and_then(|m| m.fields.iter().find(|(k, _)| k == column).map(|(_, v)| v));
    match value {
        Some(MetadataValue::String(s)) => escape_copy_text(s),
        // Numeric columns receive the literal digits; COPY coerces them into the
        // declared BIGINT/DOUBLE column. No escaping needed (digits/./-/e only).
        Some(MetadataValue::Int(n)) => n.to_string(),
        Some(MetadataValue::Float(f)) => f.to_string(),
        Some(MetadataValue::Labels(labels)) => escape_copy_text(&labels.join(";")),
        // Geo point -> "lat,lon" TEXT (earthdistance filter splits it back out).
        Some(MetadataValue::Geo { lon, lat }) => escape_copy_text(&format!("{},{}", lat, lon)),
        // missing → NULL
        _ => "\\N".to_string(),
    }
}

pub struct PgVectorEngine {
    name: String,
    host: String,
    port: u16,
    dbname: String,
    user: String,
    password: String,
    m: i64,
    ef_construction: i64,
    batch_size: usize,
    parallel: usize,
    search_params: Vec<SearchParams>,
    distance_op: String,
    hnsw_ops_class: String,
    /// Payload columns (name, pg type) derived from the dataset schema, set in
    /// `configure` and written by `upload` so filtered queries have real columns.
    schema_columns: Vec<(String, String)>,
}

impl PgVectorEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("PGVECTOR_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5432);

        let dbname = std::env::var("PGVECTOR_DB").unwrap_or_else(|_| "postgres".to_string());
        let user = std::env::var("PGVECTOR_USER").unwrap_or_else(|_| "postgres".to_string());
        let password = std::env::var("PGVECTOR_PASSWORD").unwrap_or_else(|_| "passwd".to_string());

        // Extract HNSW config
        let (m, ef_construction) = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref())
            .map(|h| (h.m.unwrap_or(16), h.ef_construction.unwrap_or(128)))
            .unwrap_or((16, 128));

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1024) as usize;

        Ok(Self {
            name: engine_config.name.clone(),
            host: host.to_string(),
            port,
            dbname,
            user,
            password,
            m,
            ef_construction,
            batch_size,
            parallel,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            // Default: will be set during configure based on dataset distance
            distance_op: String::new(),
            hnsw_ops_class: String::new(),
            schema_columns: Vec::new(),
        })
    }

    fn connection_string(&self) -> String {
        format!(
            "host={} port={} dbname={} user={} password={}",
            self.host, self.port, self.dbname, self.user, self.password
        )
    }

    fn connect(&self) -> Result<postgres::Client, String> {
        postgres::Client::connect(&self.connection_string(), postgres::NoTls)
            .map_err(|e| format!("PostgreSQL connection failed: {}", e))
    }

    fn create_progress_bar(&self, total: usize) -> ProgressBar {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec_int}/s)")
                .unwrap()
                .with_key("per_sec_int", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    write!(w, "{}", HumanCount(state.per_sec() as u64)).unwrap()
                })
                .progress_chars("#>-"),
        );
        pb
    }
}

impl Engine for PgVectorEngine {
    /// Server-side corpus size, for the `--skip-upload` reuse precondition
    /// (issue #238), as an **estimate**.
    ///
    /// It is deliberately not `SELECT count(*) FROM items`. This engine never
    /// VACUUMs and forces `STORAGE PLAIN`, so an exact count plans a Seq Scan
    /// over the whole heap — 2.6s and ~1.1GB of I/O on a 1M-row table, ~6GB at
    /// 1536 dims — pulling the entire corpus into the OS page cache and shared
    /// buffers **immediately before the search phase**. That silently converts a
    /// cold-cache pgvector run into a warm one and changes the published number,
    /// which is a worse failure than the one the guard prevents.
    ///
    /// So: `ANALYZE` (a bounded sample) then read the planner's `reltuples`.
    /// Being approximate, it can only ever warn — never abort a run.
    fn corpus_row_count(&mut self) -> Result<Option<CorpusCount>, String> {
        let mut conn = self.connect()?;
        // A missing table is "nothing to reuse", not a probe failure.
        let exists: bool = conn
            .query_one("SELECT to_regclass('public.items') IS NOT NULL", &[])
            .map_err(|e| format!("probing for the items table failed: {e}"))?
            .get(0);
        if !exists {
            return Ok(Some(CorpusCount::exact(0)));
        }
        conn.execute("ANALYZE items", &[])
            .map_err(|e| format!("ANALYZE items failed: {e}"))?;
        let reltuples: f32 = conn
            .query_one(
                "SELECT reltuples FROM pg_class WHERE oid = 'public.items'::regclass",
                &[],
            )
            .map_err(|e| format!("reading reltuples for items failed: {e}"))?
            .get(0);
        // -1 is "never analyzed"; treat it as "cannot tell" rather than as zero.
        if reltuples < 0.0 {
            return Ok(None);
        }
        Ok(Some(CorpusCount::estimated(reltuples as u64)))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let dist_lower = distance.to_lowercase();

        // Set distance operator and HNSW ops class
        let (distance_op, hnsw_ops_class) = map_pg_distance_ops(&dist_lower)?;
        self.distance_op = distance_op.to_string();
        self.hnsw_ops_class = hnsw_ops_class.to_string();

        let mut conn = self.connect()?;

        // Ensure vector extension exists
        println!("Creating vector extension...");
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .map_err(|e| format!("Failed to create vector extension: {}", e))?;
        // cube + earthdistance back the geo-radius filter (great-circle metres).
        // Best-effort: only needed when a dataset has a geo field.
        let _ = conn.execute("CREATE EXTENSION IF NOT EXISTS cube", &[]);
        let _ = conn.execute("CREATE EXTENSION IF NOT EXISTS earthdistance", &[]);

        // Drop existing table
        println!("Dropping existing items table...");
        conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
            .map_err(|e| format!("Failed to drop table: {}", e))?;

        // Derive payload columns from the dataset schema so filtered queries have
        // real columns to reference (otherwise every filter is a SQL error).
        self.schema_columns = dataset
            .config
            .schema
            .as_ref()
            .and_then(|s| s.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(field, ftype)| {
                        pg_column_type(ftype.as_str().unwrap_or(""))
                            .map(|t| (field.clone(), t.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Create table (id, embedding, + one column per schema field)
        println!("Creating items table (vector dimension {})...", vector_size);
        let mut columns = vec![
            "id SERIAL PRIMARY KEY".to_string(),
            format!("embedding vector({}) NOT NULL", vector_size),
        ];
        for (name, pg_type) in &self.schema_columns {
            columns.push(format!("\"{}\" {}", name, pg_type));
        }
        let create_sql = format!("CREATE TABLE items ({})", columns.join(", "));
        conn.execute(&create_sql, &[])
            .map_err(|e| format!("Failed to create table: {}", e))?;

        // Set storage to PLAIN for better performance
        conn.execute(
            "ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN",
            &[],
        )
        .map_err(|e| format!("Failed to alter storage: {}", e))?;

        // Create HNSW index
        println!(
            "Creating HNSW index (m={}, ef_construction={}, ops={})...",
            self.m, self.ef_construction, self.hnsw_ops_class
        );
        let index_sql = format!(
            "CREATE INDEX ON items USING hnsw(embedding {}) WITH (m = {}, ef_construction = {})",
            self.hnsw_ops_class, self.m, self.ef_construction
        );
        conn.execute(&index_sql, &[])
            .map_err(|e| format!("Failed to create HNSW index: {}", e))?;

        println!("PgVector configured successfully.");

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();
        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();

        println!(
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
        );

        // PgVector uses COPY for bulk upload
        // Use batched INSERT for parallel upload since COPY requires exclusive connection
        let pb = self.create_progress_bar(ids.len());
        let upload_start = Instant::now();

        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let conn_str = self.connection_string();
        let schema_columns = self.schema_columns.clone();

        // COPY target column list: id, embedding, then one per schema field.
        let copy_sql = {
            let mut cols = vec!["id".to_string(), "embedding".to_string()];
            for (name, _) in &schema_columns {
                cols.push(format!("\"{}\"", name));
            }
            format!(
                "COPY items ({}) FROM STDIN WITH (FORMAT text)",
                cols.join(", ")
            )
        };

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let conn_str = conn_str.clone();
                let copy_sql = copy_sql.clone();
                let schema_columns = &schema_columns;
                let batches = &batches;
                let ids = &ids;
                let vectors = &vectors;
                let metadata = &metadata;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_batches {
                            break;
                        }
                        if error.lock().unwrap().is_some() {
                            break;
                        }

                        let (batch_start, batch_end) = batches[idx];

                        // Use COPY for bulk insert
                        let copy_result = (|| -> Result<(), String> {
                            let mut writer = conn
                                .copy_in(copy_sql.as_str())
                                .map_err(|e| format!("COPY start failed: {}", e))?;

                            use std::io::Write;
                            for i in batch_start..batch_end {
                                let vec_str: String = vectors[i]
                                    .iter()
                                    .map(|v| v.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",");
                                let mut line = format!("{}\t[{}]", ids[i], vec_str);
                                for (name, _) in schema_columns {
                                    line.push('\t');
                                    line.push_str(&copy_field(metadata[i].as_ref(), name));
                                }
                                writeln!(writer, "{}", line)
                                    .map_err(|e| format!("COPY write failed: {}", e))?;
                            }

                            writer
                                .finish()
                                .map_err(|e| format!("COPY finish failed: {}", e))?;
                            Ok(())
                        })();

                        if let Err(e) = copy_result {
                            *error.lock().unwrap() = Some(e);
                            break;
                        }

                        pb.inc((batch_end - batch_start) as u64);
                    }
                });
            }
        });

        pb.finish_with_message("Upload complete");

        if let Some(e) = error.lock().unwrap().take() {
            return Err(e);
        }

        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        let total_time = read_time + upload_time;

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: vectors.len(),
            parallel: self.parallel,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        // Re-derive the distance operator here as well as in configure(): on the
        // `--skip-upload` path configure() never runs (#238), and `distance_op`
        // would still be its `new()` default of "" — which splices into the query
        // as `embedding  $1` and fails EVERY search with a bare SQL syntax error
        // after a green reuse check. It is a pure function of the dataset, so
        // recomputing is idempotent; it also restores the unsupported-metric hard
        // error that the skip-upload path otherwise loses.
        let (distance_op, hnsw_ops_class) =
            map_pg_distance_ops(&dataset.distance().to_lowercase())?;
        self.distance_op = distance_op.to_string();
        self.hnsw_ops_class = hnsw_ops_class.to_string();

        let parallel = params.parallel.unwrap_or(1) as usize;

        // Extract hnsw_ef from search params: the typed `ef` field first, then
        // `hnsw_ef` in either placement (nested under `search_params`/`config`,
        // or flat) via knob().
        let hnsw_ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .or_else(|| params.knob("hnsw_ef").and_then(|v| v.as_i64()))
            .unwrap_or(128);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        // Parse each query's filter ONCE (outside the timed region) into a
        // parameterized template + a list of typed values to bind. The template
        // SQL text is stable per filter shape (values are `$N` placeholders, not
        // inlined), so the server reuses one prepared plan across queries in a
        // run. Filter placeholders start at $3 ($1 = query vector, $2 = LIMIT).
        let parsed_filters: Vec<QueryFilter<(String, Vec<PgValue>)>> =
            conditions.resolve_all("pgvector", |v| parse_pg_conditions(v, 3))?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let conn_str = self.connection_string();
        let distance_op = self.distance_op.clone();

        // Every worker holds its own connection for the whole run, so `parallel`
        // is a claim on the server's connection budget. Exceeding it used to
        // publish a run labelled `parallel=N` that N-k workers actually ran; it
        // is now a hard error, which is correct but opaque at the point of
        // failure. Warn here, with the arithmetic, rather than leaving the
        // operator to infer it from k identical "too many clients" strings.
        //
        // Advisory only: this is racy by construction, and a pooler in front of
        // Postgres makes `max_connections` meaningless. The start gate remains
        // the authority.
        warn_if_over_connection_budget(&conn_str, parallel);

        // Gate-synchronized start so connection setup AND the cold first query
        // fall OUTSIDE the measured window. Every worker connects + primes, then
        // parks at the gate; `WorkerPool::start` stamps the shared start instant and
        // releases everyone, so the measurement clock starts only once all workers
        // are warm and poised. The gate is count-agnostic: a worker that fails to
        // set up, panics, or is never started by the OS settles its ticket and turns
        // the run into a hard error instead of a hang (#214).

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        let measured_start = std::thread::scope(|s| -> Result<Instant, String> {
            let mut pool = WorkerPool::new(s, "pgvector-search", parallel);
            for _ in 0..parallel {
                let conn_str = conn_str.clone();
                let distance_op = distance_op.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                pool.spawn(move |ticket| {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(e) => {
                            // A worker that cannot set itself up would leave the run at a
                            // lower real concurrency than the `parallel` it reports. Settle
                            // the ticket with the reason; the coordinator makes it an error.
                            ticket.fail(format!(
                                "pgvector-search worker connect failed: {}",
                                describe_pg_error(&e)
                            ));
                            return (t, p, r, mr, nd);
                        }
                    };

                    // Set ef_search for this connection
                    let _ = conn.execute(
                        &format!("SET hnsw.ef_search = {}", hnsw_ef),
                        &[],
                    );

                    // Enable iterative index scans (pgvector >= 0.8) so that
                    // non-sargable filters (e.g. array contains-any) keep pulling
                    // from the HNSW index until LIMIT is satisfied in exact order,
                    // instead of post-filtering a bounded candidate set.
                    let _ = conn.execute("SET hnsw.iterative_scan = strict_order", &[]);

                    // Prime this connection with ONE discarded query (index 0) so
                    // the cold first round-trip + plan build is not inside the
                    // measured window. Best effort: errors ignored, no sample
                    // recorded. Reuses the exact per-query request path below.
                    if !queries.is_empty() {
                        let prime_top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[0].len();
                            if n > 0 { n } else { 10 }
                        });
                        let prime_vec = pgvector::Vector::from(queries[0].clone());
                        let prime_top_param = prime_top as i64;
                        let prime_filter = parsed_filters[0].as_ref();
                        let prime_where = prime_filter
                            .map(|(tmpl, _)| format!(" WHERE {}", tmpl))
                            .unwrap_or_default();
                        let prime_sql = format!(
                            "SELECT id, embedding {} $1 AS _score FROM items{} ORDER BY _score LIMIT $2::bigint",
                            distance_op, prime_where
                        );
                        let mut prime_params: Vec<&(dyn ToSql + Sync)> =
                            Vec::with_capacity(2 + prime_filter.map(|(_, v)| v.len()).unwrap_or(0));
                        prime_params.push(&prime_vec);
                        prime_params.push(&prime_top_param);
                        if let Some((_, values)) = prime_filter {
                            for v in values {
                                prime_params.push(v.as_sql());
                            }
                        }
                        let _ = conn.query(&prime_sql, &prime_params);
                    }

                    // Signal "connected + primed", then block until the coordinator
                    // stamps the shared measurement start and releases everyone.
                    if ticket.arrive_and_wait().is_none() {
                        return (t, p, r, mr, nd);
                    }

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 { n } else { 10 }
                        });

                        let query_vec = pgvector::Vector::from(queries[idx].clone());
                        // `top` (LIMIT) and every filter value are BOUND params,
                        // not inlined, so the statement text is stable per filter
                        // shape and the plan is reused across queries.
                        let top_param = top as i64;

                        let filter = parsed_filters[idx].as_ref();
                        let where_clause = filter
                            .map(|(tmpl, _)| format!(" WHERE {}", tmpl))
                            .unwrap_or_default();

                        // $1 = query vector, $2 = LIMIT (bigint), $3.. = filter values.
                        let query_sql = format!(
                            "SELECT id, embedding {} $1 AS _score FROM items{} ORDER BY _score LIMIT $2::bigint",
                            distance_op, where_clause
                        );

                        let mut query_params: Vec<&(dyn ToSql + Sync)> =
                            Vec::with_capacity(2 + filter.map(|(_, v)| v.len()).unwrap_or(0));
                        query_params.push(&query_vec);
                        query_params.push(&top_param);
                        if let Some((_, values)) = filter {
                            for v in values {
                                query_params.push(v.as_sql());
                            }
                        }

                        let query_start = Instant::now();
                        let results = conn.query(&query_sql, &query_params);
                        let query_time = query_start.elapsed().as_secs_f64();

                        match results {
                            Ok(rows) => {
                                let ordered_ids: Vec<i64> = rows
                                    .iter()
                                    .map(|row| {
                                        let id: i32 = row.get(0);
                                        id as i64
                                    })
                                    .collect();
                                let m = crate::metrics::compute_metrics(
                                    &ordered_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                t.push(query_time);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                            }
                            Err(e) => {
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                })?;
            }

            // Every worker is connected + primed and parked at the gate.
            // Stamp the shared measurement start and release them together.
            let (per_worker, measured_start) = pool.start()?;

            for (t, p, r, mr, nd) in per_worker {
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
            Ok(measured_start)
        })?;

        pb.finish_and_clear();
        // total_time excludes connection setup and the cold first query: it is
        // measured from the gate release stamp.
        let total_time = measured_start.elapsed().as_secs_f64();

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        let mut conn = self.connect()?;
        conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
            .map_err(|e| format!("Failed to drop table: {}", e))?;
        Ok(())
    }
}

// ── PgVector condition parser ─────────────────────────────────────

/// Map a dataset distance name to the pgvector `(operator, hnsw_ops_class)` pair.
/// `dot`/`ip` is unsupported and unknown metrics error. A wrong arm here (e.g.
/// L2 op with a cosine ops class) would silently change ranking, so every arm is
/// unit-tested.
fn map_pg_distance_ops(distance: &str) -> Result<(&'static str, &'static str), String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(("<->", "vector_l2_ops")),
        "cosine" | "angular" => Ok(("<=>", "vector_cosine_ops")),
        other => Err(format!(
            "PgVector does not support distance metric: {}",
            other
        )),
    }
}

/// A filter value bound as a query parameter (instead of inlined into the SQL
/// text) so the prepared-statement text stays stable across queries in a run and
/// the server reuses one plan. Each variant owns its data and lends a
/// `&(dyn ToSql + Sync)` for binding via `as_sql`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PgValue {
    Text(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    TextArray(Vec<String>),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

impl PgValue {
    fn as_sql(&self) -> &(dyn ToSql + Sync) {
        match self {
            PgValue::Text(v) => v,
            PgValue::Int(v) => v,
            PgValue::Float(v) => v,
            PgValue::Bool(v) => v,
            PgValue::TextArray(v) => v,
            PgValue::IntArray(v) => v,
            PgValue::FloatArray(v) => v,
        }
    }
}

/// Accumulates the ordered list of bound values and hands out sequential `$N`
/// placeholders so the emitted template and the values line up positionally.
struct FilterBuilder {
    next_param: usize,
    values: Vec<PgValue>,
}

impl FilterBuilder {
    /// Record `v` as the next bound value and return its `$N` placeholder text.
    fn bind(&mut self, v: PgValue) -> String {
        let n = self.next_param;
        self.next_param += 1;
        self.values.push(v);
        format!("${}", n)
    }
}

/// Parse a dataset filter into a parameterized SQL template + the ordered typed
/// values to bind. `first_param` is the placeholder number the filter's first
/// value takes (callers pass 3: $1 = query vector, $2 = LIMIT). The template
/// text depends only on the filter *shape* (fields/operators), never on the
/// values, so distinct queries in a run share one prepared statement.
pub(crate) fn parse_pg_conditions(
    conditions: &serde_json::Value,
    first_param: usize,
) -> Option<(String, Vec<PgValue>)> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut builder = FilterBuilder {
        next_param: first_param,
        values: Vec::new(),
    };

    let template = build_pg_group(obj, &mut builder)?;
    Some((template, builder.values))
}

/// Build the SQL for a `{and:[...], or:[...]}` group. Each `and`/`or` array
/// becomes a parenthesised sub-expression joined by SQL `AND`/`OR`; the two
/// groups (when both present) are combined with `AND`. Entries that are
/// themselves nested groups recurse through `build_pg_entry`, so an arbitrarily
/// deep tree like `(a AND b) OR (c AND d)` maps to correctly parenthesised SQL.
fn build_pg_group(
    obj: &serde_json::Map<String, serde_json::Value>,
    builder: &mut FilterBuilder,
) -> Option<String> {
    let mut clauses = Vec::new();

    if let Some(and_entries) = obj.get("and").and_then(|v| v.as_array()) {
        let sub: Vec<String> = and_entries
            .iter()
            .filter_map(|e| build_pg_entry(e, builder))
            .collect();
        if !sub.is_empty() {
            clauses.push(format!("({})", sub.join(" AND ")));
        }
    }

    if let Some(or_entries) = obj.get("or").and_then(|v| v.as_array()) {
        let sub: Vec<String> = or_entries
            .iter()
            .filter_map(|e| build_pg_entry(e, builder))
            .collect();
        if !sub.is_empty() {
            clauses.push(format!("({})", sub.join(" OR ")));
        }
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

/// Build one entry of an `and`/`or` array. An entry is EITHER a nested group
/// (it carries an `and`/`or` key) — built as its own parenthesised sub-clause so
/// the engine's native SQL grouping preserves precedence — OR a field leaf,
/// dispatched to `build_pg_clause`. This is the recursion that keeps a
/// mis-flattening builder from collapsing `(a AND b) OR (c AND d)` into a flat
/// disjunction.
fn build_pg_entry(entry: &serde_json::Value, builder: &mut FilterBuilder) -> Option<String> {
    if let Some(entry_obj) = entry.as_object() {
        if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
            return build_pg_group(entry_obj, builder).map(|s| format!("({})", s));
        }
    }
    build_pg_clause(entry, builder)
}

fn build_pg_clause(entry: &serde_json::Value, builder: &mut FilterBuilder) -> Option<String> {
    let entry_obj = entry.as_object()?;
    let mut parts = Vec::new();
    for (field_name, field_filters) in entry_obj {
        let filter_obj = field_filters.as_object()?;
        for (condition_type, criteria) in filter_obj {
            match condition_type.as_str() {
                "match" => {
                    // match_any: OR-of-values, mirroring qdrant's
                    // Condition::matches(field, Vec).
                    if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                        if !any.is_empty() && any.iter().all(|v| v.is_number()) {
                            // Numeric contains-any on a scalar numeric column. The
                            // WHOLE list is bound as ONE array param and tested with
                            // `= ANY(...)` (equivalent to `IN`), so the SQL text is
                            // independent of the list length — a stable statement.
                            if any.iter().all(|v| v.is_i64() || v.is_u64()) {
                                let nums: Vec<i64> = any.iter().map(json_as_i64).collect();
                                let ph = builder.bind(PgValue::IntArray(nums));
                                parts.push(format!("\"{}\" = ANY({}::bigint[])", field_name, ph));
                            } else {
                                let nums: Vec<f64> =
                                    any.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                                let ph = builder.bind(PgValue::FloatArray(nums));
                                parts.push(format!(
                                    "\"{}\" = ANY({}::double precision[])",
                                    field_name, ph
                                ));
                            }
                        } else {
                            // Keyword contains-any. Multi-valued keyword payloads
                            // are stored as a single ';'-joined TEXT scalar (see
                            // copy_field), so a scalar `IN` can never match an
                            // array-valued doc. Split the column on ';' and test
                            // set intersection with the query values — correct for
                            // both single-valued ('red' -> {red}) and multi-valued
                            // ('red;green' -> {red,green}) keyword fields, mirroring
                            // qdrant's array-contains-any semantics. The value set is
                            // a single bound text[] param (no manual escaping, stable
                            // statement text regardless of list length).
                            let elems: Vec<String> = any
                                .iter()
                                .filter_map(|v| v.as_str())
                                .map(|s| s.to_string())
                                .collect();
                            if elems.is_empty() {
                                // Empty (or no representable) IN-set matches
                                // NOTHING; a bare `false` keeps that in both AND
                                // and OR contexts instead of dropping the clause
                                // (which as the sole condition would return every
                                // row — the inverse of the filter).
                                parts.push("false".to_string());
                            } else {
                                let ph = builder.bind(PgValue::TextArray(elems));
                                parts.push(format!(
                                    "string_to_array(\"{}\", ';') && {}::text[]",
                                    field_name, ph
                                ));
                            }
                        }
                    } else if let Some(value) = criteria.get("value") {
                        if let Some(s) = value.as_str() {
                            let ph = builder.bind(PgValue::Text(s.to_string()));
                            if is_multivalued_keyword_field(field_name) {
                                // Multi-valued keyword: the ';'-joined column holds
                                // a set, so exact-match means "contains this value".
                                // Test set membership, mirroring the match_any arm
                                // above (issue #88) — a scalar `=` compares the whole
                                // joined string and never matches one element.
                                parts.push(format!(
                                    "{} = ANY(string_to_array(\"{}\", ';'))",
                                    ph, field_name
                                ));
                            } else {
                                parts.push(format!("\"{}\" = {}", field_name, ph));
                            }
                        } else if let Some(b) = value.as_bool() {
                            let ph = builder.bind(PgValue::Bool(b));
                            parts.push(format!("\"{}\" = {}", field_name, ph));
                        } else if value.is_i64() || value.is_u64() {
                            let ph = builder.bind(PgValue::Int(json_as_i64(value)));
                            parts.push(format!("\"{}\" = {}", field_name, ph));
                        } else if let Some(f) = value.as_f64() {
                            let ph = builder.bind(PgValue::Float(f));
                            parts.push(format!("\"{}\" = {}", field_name, ph));
                        } else {
                            // Non-scalar match value (a JSON array/object/null)
                            // is malformed input — the canonical model uses
                            // `match.any` for lists. Drop the clause (emit
                            // nothing) instead of inlining `"f" = [1,2]`
                            // verbatim, matching qdrant/redis/valkey/vectorsets.
                            // As the sole condition this leaves `parts` empty →
                            // the builder returns None.
                        }
                    } else if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                        // Full-text match: a `{match:{text}}` clause over a TEXT
                        // column. Postgres FTS token containment via
                        // `to_tsvector(col) @@ plainto_tsquery(term)`. Dropping the
                        // clause would run the kNN query UNFILTERED while recall is
                        // scored against the filtered ground truth.
                        let ph = builder.bind(PgValue::Text(text.to_string()));
                        parts.push(format!(
                            "to_tsvector('english', \"{}\") @@ plainto_tsquery('english', {})",
                            field_name, ph
                        ));
                    }
                }
                "range" => {
                    if let Some(co) = criteria.as_object() {
                        for (op, val) in co {
                            let sql_op = match op.as_str() {
                                "lt" => "<",
                                "gt" => ">",
                                "lte" => "<=",
                                "gte" => ">=",
                                _ => continue,
                            };
                            if val.is_null() {
                                continue;
                            }
                            if val.is_i64() || val.is_u64() {
                                let ph = builder.bind(PgValue::Int(json_as_i64(val)));
                                parts.push(format!("\"{}\" {} {}", field_name, sql_op, ph));
                            } else if let Some(f) = val.as_f64() {
                                let ph = builder.bind(PgValue::Float(f));
                                parts.push(format!("\"{}\" {} {}", field_name, sql_op, ph));
                            } else if let Some(s) = val.as_str() {
                                // A string bound is an ISO-8601 datetime range over a
                                // TIMESTAMPTZ column. Inline as a single-quoted SQL
                                // literal + cast: binding `$n::timestamptz` makes
                                // Postgres infer the param type as timestamptz and
                                // reject a text value ("error serializing parameter").
                                // Single-quote-escape the benchmark-controlled ISO
                                // string (was previously inlined bare, i.e. rendered
                                // with double quotes → a broken SQL identifier).
                                let esc = s.replace('\'', "''");
                                parts.push(format!(
                                    "\"{}\" {} '{}'::timestamptz",
                                    field_name, sql_op, esc
                                ));
                            } else {
                                // Non-scalar range bound (degenerate) — inline.
                                parts.push(format!("\"{}\" {} {}", field_name, sql_op, val));
                            }
                        }
                    }
                }
                "geo" => {
                    // Geo-radius over the "lat,lon" TEXT column via earthdistance
                    // (great-circle metres). lat/lon/radius are dataset numbers
                    // (not injectable), so inline them; the stored point is split
                    // back into (lat, lon) for ll_to_earth.
                    if let (Some(lat), Some(lon), Some(radius)) = (
                        criteria.get("lat").and_then(|v| v.as_f64()),
                        criteria.get("lon").and_then(|v| v.as_f64()),
                        criteria.get("radius").and_then(|v| v.as_f64()),
                    ) {
                        parts.push(format!(
                            "earth_distance(ll_to_earth(split_part(\"{f}\", ',', 1)::float8, \
                             split_part(\"{f}\", ',', 2)::float8), ll_to_earth({lat}, {lon})) <= {radius}",
                            f = field_name,
                            lat = lat,
                            lon = lon,
                            radius = radius
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

/// Coerce a JSON number to i64, tolerating u64 values that fit (falling back to
/// a saturating cast for the rare > i64::MAX case).
fn json_as_i64(v: &serde_json::Value) -> i64 {
    v.as_i64()
        .or_else(|| v.as_u64().map(|u| u as i64))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};

    #[test]
    fn maps_schema_types_to_pg_columns() {
        assert_eq!(pg_column_type("keyword"), Some("TEXT"));
        assert_eq!(pg_column_type("text"), Some("TEXT"));
        assert_eq!(pg_column_type("int"), Some("BIGINT"));
        assert_eq!(pg_column_type("float"), Some("DOUBLE PRECISION"));
        assert_eq!(pg_column_type("bool"), Some("BOOLEAN"));
        assert_eq!(pg_column_type("datetime"), Some("TIMESTAMPTZ"));
        // Geo is now stored as a "lat,lon" TEXT scalar (earthdistance filter).
        assert_eq!(pg_column_type("geo"), Some("TEXT"));
        assert_eq!(pg_column_type("uuid"), Some("TEXT"));
        assert_eq!(pg_column_type("unknown"), None);
    }

    #[test]
    fn escapes_copy_text_specials() {
        assert_eq!(escape_copy_text("a\tb\nc\\d"), "a\\tb\\nc\\\\d");
    }

    #[test]
    fn copy_field_reads_value_or_null() {
        let meta = MetadataItem {
            fields: vec![
                (
                    "category".to_string(),
                    MetadataValue::String("shoes".to_string()),
                ),
                (
                    "labels".to_string(),
                    MetadataValue::Labels(vec!["a".to_string(), "b".to_string()]),
                ),
            ],
        };
        assert_eq!(copy_field(Some(&meta), "category"), "shoes");
        assert_eq!(copy_field(Some(&meta), "labels"), "a;b");
        assert_eq!(copy_field(Some(&meta), "missing"), "\\N");
        assert_eq!(copy_field(None, "category"), "\\N");
    }

    // Native numeric fields (issue #87) must COPY the literal digits into the
    // BIGINT/DOUBLE column — NOT fall through to the `_ => NULL` arm.
    #[test]
    fn copy_field_emits_numeric_literals() {
        let meta = MetadataItem {
            fields: vec![
                ("size".to_string(), MetadataValue::Int(42)),
                ("price".to_string(), MetadataValue::Float(3.5)),
            ],
        };
        assert_eq!(copy_field(Some(&meta), "size"), "42");
        assert_eq!(copy_field(Some(&meta), "price"), "3.5");
    }

    // Parse with the production base offset ($1 = vector, $2 = LIMIT, filter
    // values start at $3) and return the (template, values) pair.
    fn parse(cond: &serde_json::Value) -> Option<(String, Vec<PgValue>)> {
        parse_pg_conditions(cond, 3)
    }

    #[test]
    fn filter_clause_quotes_columns_and_binds_values() {
        // AND of a keyword match and a numeric range → quoted columns, values as
        // sequential placeholders (NOT inlined), values captured in bind order.
        let cond = json!({
            "and": [
                {"category": {"match": {"value": "shoes"}}},
                {"year": {"range": {"gte": 2020}}}
            ]
        });
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"category\" = $3 AND \"year\" >= $4)");
        assert_eq!(
            vals,
            vec![PgValue::Text("shoes".to_string()), PgValue::Int(2020)]
        );
    }

    #[test]
    fn match_value_binds_string_without_inline_escaping() {
        // The value is a bound param — no SQL text escaping, no value in the text.
        let cond = json!({"and": [{"name": {"match": {"value": "O'Brien"}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"name\" = $3)");
        assert_eq!(vals, vec![PgValue::Text("O'Brien".to_string())]);
    }

    #[test]
    fn statement_text_is_stable_across_differing_values() {
        // Same filter SHAPE, different values (tenant_7 vs tenant_3) → identical
        // statement text so the server reuses one prepared plan. Only the bound
        // values differ.
        let (sql7, v7) =
            parse(&json!({"and":[{"tenant":{"match":{"value":"tenant_7"}}}]})).unwrap();
        let (sql3, v3) =
            parse(&json!({"and":[{"tenant":{"match":{"value":"tenant_3"}}}]})).unwrap();
        assert_eq!(
            sql7, sql3,
            "statement text must be identical across queries"
        );
        assert_eq!(sql7, "(\"tenant\" = $3)");
        assert_eq!(v7, vec![PgValue::Text("tenant_7".to_string())]);
        assert_eq!(v3, vec![PgValue::Text("tenant_3".to_string())]);
    }

    #[test]
    fn first_param_offset_is_honored() {
        // Placeholder numbering starts at the caller-supplied base.
        let (sql, _) =
            parse_pg_conditions(&json!({"and":[{"a":{"match":{"value":"x"}}}]}), 5).unwrap();
        assert_eq!(sql, "(\"a\" = $5)");
    }

    #[test]
    fn match_any_string_list_binds_text_array_overlap() {
        // Contains-any via set intersection, so it matches both single-valued and
        // ';'-joined multi-valued keyword columns. The whole set is one text[]
        // param, so the statement text is independent of the list length.
        let cond = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(string_to_array(\"color\", ';') && $3::text[])");
        assert_eq!(
            vals,
            vec![PgValue::TextArray(vec![
                "red".to_string(),
                "blue".to_string()
            ])]
        );
    }

    #[test]
    fn labels_exact_match_uses_set_membership() {
        // #88: exact-match on the multi-valued `labels` column means "contains
        // this value" — test set membership, not whole-string equality. A
        // single-valued keyword (`name`) keeps scalar `=`.
        let cond = json!({"and": [{"labels": {"match": {"value": "red"}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "($3 = ANY(string_to_array(\"labels\", ';')))");
        assert_eq!(vals, vec![PgValue::Text("red".to_string())]);

        let (sql_name, _) =
            parse(&json!({"and": [{"name": {"match": {"value": "red"}}}]})).unwrap();
        assert_eq!(sql_name, "(\"name\" = $3)");
    }

    #[test]
    fn match_any_int_list_binds_bigint_array_any() {
        // `= ANY($N::bigint[])` is equivalent to `IN`, with the list bound as one
        // array param (stable statement text regardless of length).
        let cond = json!({"and": [{"size": {"match": {"any": [1, 2, 3]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"size\" = ANY($3::bigint[]))");
        assert_eq!(vals, vec![PgValue::IntArray(vec![1, 2, 3])]);
    }

    #[test]
    fn match_any_float_list_binds_double_array_any() {
        let cond = json!({"and": [{"score": {"match": {"any": [1.5, 2.5]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"score\" = ANY($3::double precision[]))");
        assert_eq!(vals, vec![PgValue::FloatArray(vec![1.5, 2.5])]);
    }

    #[test]
    fn match_any_string_values_are_bound_not_escaped() {
        let cond = json!({"and": [{"name": {"match": {"any": ["O'Brien"]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(string_to_array(\"name\", ';') && $3::text[])");
        assert_eq!(vals, vec![PgValue::TextArray(vec!["O'Brien".to_string()])]);
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // An empty IN-set must match NOTHING (never invert to returning all
        // rows), so the clause is a bare `false` (no bound value) rather than
        // being dropped.
        let cond = json!({"and": [{"color": {"match": {"any": []}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(false)");
        assert!(vals.is_empty());
    }

    // #121: a non-scalar `value` (a JSON array) is malformed input — the
    // canonical model uses `match.any` for lists. The clause is dropped (no SQL,
    // no bound value) rather than inlining `"n" = [1,2]` verbatim. As the sole
    // condition the whole filter is None. Matches qdrant/redis/valkey/vectorsets.
    #[test]
    fn match_non_scalar_value_dropped() {
        assert!(parse(&json!({"and": [{"n": {"match": {"value": [1, 2]}}}]})).is_none());
        assert!(parse(&json!({"and": [{"n": {"match": {"value": {"x": 1}}}}]})).is_none());
        assert!(parse(&json!({"and": [{"n": {"match": {"value": null}}}]})).is_none());
        // A non-scalar clause is dropped but a sibling scalar clause survives.
        let cond = json!({"and": [
            {"n": {"match": {"value": [1, 2]}}},
            {"c": {"match": {"value": "red"}}},
        ]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"c\" = $3)");
        assert_eq!(vals, vec![PgValue::Text("red".to_string())]);
    }

    // ── OR-branch of the condition parser ──────────────────────────────────

    #[test]
    fn or_only_emits_or_joined_group() {
        let cond = json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"a\" = $3 OR \"b\" = $4)");
        assert_eq!(
            vals,
            vec![
                PgValue::Text("x".to_string()),
                PgValue::Text("y".to_string())
            ]
        );
    }

    #[test]
    fn and_plus_or_keeps_both_groups() {
        // AND group binds first ($3), then OR group ($4).
        let cond = json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}}],
        });
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"a\" = $3) AND (\"b\" = $4)");
        assert_eq!(
            vals,
            vec![
                PgValue::Text("x".to_string()),
                PgValue::Text("y".to_string())
            ]
        );
    }

    // ── Nested / grouped boolean tree ──────────────────────────────────────

    #[test]
    fn nested_or_of_and_groups_parenthesises_each_subgroup() {
        // (color==red AND size>=50) OR (color==blue AND size<10). Each nested
        // AND-group must become its OWN parenthesised sub-expression so the top
        // OR does not mis-flatten into `red AND size>=50 OR color==blue AND ...`.
        let cond = json!({
            "or": [
                {"and": [
                    {"color": {"match": {"value": "red"}}},
                    {"size": {"range": {"gte": 50}}}
                ]},
                {"and": [
                    {"color": {"match": {"value": "blue"}}},
                    {"size": {"range": {"lt": 10}}}
                ]}
            ]
        });
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(
            sql,
            "(((\"color\" = $3 AND \"size\" >= $4)) OR ((\"color\" = $5 AND \"size\" < $6)))"
        );
        assert_eq!(
            vals,
            vec![
                PgValue::Text("red".to_string()),
                PgValue::Int(50),
                PgValue::Text("blue".to_string()),
                PgValue::Int(10),
            ]
        );
    }

    // ── Range operators ────────────────────────────────────────────────────

    fn range_sql(criteria: serde_json::Value) -> Option<(String, Vec<PgValue>)> {
        parse(&json!({"and":[{"n":{"range":criteria}}]}))
    }

    #[test]
    fn range_lt_lte_gt_gte() {
        assert_eq!(
            range_sql(json!({"lt":5})).unwrap(),
            ("(\"n\" < $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            range_sql(json!({"lte":5})).unwrap(),
            ("(\"n\" <= $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            range_sql(json!({"gt":5})).unwrap(),
            ("(\"n\" > $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            range_sql(json!({"gte":5})).unwrap(),
            ("(\"n\" >= $3)".to_string(), vec![PgValue::Int(5)])
        );
    }

    #[test]
    fn range_two_sided_gte_lt() {
        // Criteria object keys iterate in sorted order (BTreeMap): gte, then lt.
        let (sql, vals) = range_sql(json!({"gte":10,"lt":20})).unwrap();
        assert_eq!(sql, "(\"n\" >= $3 AND \"n\" < $4)");
        assert_eq!(vals, vec![PgValue::Int(10), PgValue::Int(20)]);
    }

    #[test]
    fn range_unknown_op_is_none() {
        assert!(range_sql(json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_is_none() {
        assert!(range_sql(json!({"gte":serde_json::Value::Null})).is_none());
    }

    // ── Geo filter (unsupported → None) ────────────────────────────────────

    #[test]
    fn geo_builds_earthdistance_clause() {
        // Geo-radius is filtered via earthdistance over the "lat,lon" TEXT column:
        // earth_distance(ll_to_earth(lat, lon), ll_to_earth(qlat, qlon)) <= radius.
        let cond = json!({"and":[{"loc":{"geo":{"lat":20.0,"lon":10.0,"radius":5}}}]});
        let (sql, binds) = parse(&cond).expect("geo clause should be built");
        assert!(sql.contains("earth_distance"), "sql: {sql}");
        assert!(sql.contains("ll_to_earth(20"), "query point inlined: {sql}");
        assert!(sql.contains("<= 5"), "radius inlined: {sql}");
        assert!(binds.is_empty(), "geo inlines its numeric args, no binds");
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn distance_ops_mapping_covers_all_arms() {
        assert_eq!(map_pg_distance_ops("l2").unwrap(), ("<->", "vector_l2_ops"));
        assert_eq!(
            map_pg_distance_ops("euclidean").unwrap(),
            ("<->", "vector_l2_ops")
        );
        assert_eq!(
            map_pg_distance_ops("cosine").unwrap(),
            ("<=>", "vector_cosine_ops")
        );
        assert_eq!(
            map_pg_distance_ops("angular").unwrap(),
            ("<=>", "vector_cosine_ops")
        );
        assert_eq!(
            map_pg_distance_ops("COSINE").unwrap(),
            ("<=>", "vector_cosine_ops")
        );
        // dot/ip unsupported; unknown errors too.
        assert!(map_pg_distance_ops("dot").is_err());
        assert!(map_pg_distance_ops("ip").is_err());
        assert!(map_pg_distance_ops("nope").is_err());
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    #[test]
    fn exact_match_int_float_bool() {
        // Scalar match values become typed bound params (stable statement text).
        assert_eq!(
            parse(&json!({"and":[{"n":{"match":{"value":5}}}]})).unwrap(),
            ("(\"n\" = $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            parse(&json!({"and":[{"n":{"match":{"value":1.5}}}]})).unwrap(),
            ("(\"n\" = $3)".to_string(), vec![PgValue::Float(1.5)])
        );
        assert_eq!(
            parse(&json!({"and":[{"flag":{"match":{"value":true}}}]})).unwrap(),
            ("(\"flag\" = $3)".to_string(), vec![PgValue::Bool(true)])
        );
    }

    #[test]
    fn exact_match_array_value_is_dropped() {
        // #121: a non-scalar match value (a JSON array) is malformed input — the
        // canonical model uses `match.any` for lists. The clause is dropped (as
        // the sole condition → None), matching qdrant/redis/valkey/vectorsets
        // (was previously inlined verbatim as `"n" = [1,2]`).
        assert!(parse(&json!({"and":[{"n":{"match":{"value":[1,2]}}}]})).is_none());
    }
}
