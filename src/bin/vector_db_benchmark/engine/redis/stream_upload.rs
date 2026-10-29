//! Streaming upload for Redis (#342): `vectors.npy` + `payloads.jsonl` read a
//! batch at a time by `upload_stream`, so client memory is bounded by
//! `parallel x batch_size` instead of the corpus. The in-memory upload in
//! `redis.rs` is untouched; both send every batch through
//! `upload_batch_with_retry`, so keys, fields, vector encoding and the
//! reconnect policy are identical.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::{redis_utils, upload_batch_with_retry, RedisEngine};
use crate::dataset::Dataset;
use crate::engine::UploadStats;
use vector_db_benchmark::fulltext;
use vector_db_benchmark::indexing_wait::WaitPhase;
use vector_db_benchmark::ingest::IndexWait;
use vector_db_benchmark::readers::CompoundStream;
use vector_db_benchmark::upload_stream;
use vector_db_benchmark::workload::Workload;

impl RedisEngine {
    pub(super) fn upload_streamed(
        &mut self,
        dataset: &Dataset,
        stream: CompoundStream,
    ) -> Result<UploadStats, String> {
        let rows = stream.rows();
        let dim = stream.dim();
        let read_start = Instant::now();
        // Pre-flight, BEFORE anything is written (#361 review): one
        // payload-only pass parses every `payloads.jsonl` row and counts them
        // against the vectors, so a truncated or malformed file fails here,
        // as on the in-memory path, not hours into the upload with a partial
        // corpus on the server. When the index serves filter-only (`no-vector`
        // or the shared `full` layout, #344) the same pass checks its metadata
        // domain. This pass is this path's `read_time`.
        let check_domain =
            self.config.layout.serves(Workload::FilterOnly) && !self.scalar_metadata_validated;
        dataset.preflight_streamed_payloads(check_domain)?;
        if check_domain {
            self.scalar_metadata_validated = true;
        }
        // #339 parity: the same per-row whitespace normalization the
        // in-memory path applies to the whole corpus, applied per batch.
        let parity_fields = match self.fulltext_profile()? {
            Some(p) if p.normalizes_whitespace() => Some(p.text_fields),
            _ => None,
        };
        let parallel = self.config.parallel.max(1);
        let batch_size = self.config.batch_size;
        let bound = upload_stream::memory_bound_bytes(
            parallel,
            batch_size,
            dim,
            upload_stream::mean_payload_bytes(&dataset.get_path()?, rows),
        );
        println!(
            "Streaming {rows} vectors ({dim}d) from disk with {parallel} threads, batch size \
             {batch_size}; client memory bound ~{:.0} MiB, independent of corpus size (#342)",
            bound as f64 / (1 << 20) as f64
        );

        let pb = self.create_progress_bar(rows as usize);
        let normalized = AtomicU64::new(0);
        let redis_url = self.redis_url.clone();
        let config = self.config.clone();
        let read_time = read_start.elapsed().as_secs_f64();
        // One clock for both ingest phases (#344): the streaming upload is the
        // upload phase (disk read + decode included), the FT.INFO wait the
        // index phase.
        let (stats, phases) = upload_stream::ingest_streamed(
            read_time,
            IndexWait::RedisPercentIndexed,
            || {
                let stats = upload_stream::upload_stream(
                    stream,
                    parallel,
                    batch_size,
                    || {
                        let client =
                            redis::Client::open(redis_url.as_str()).map_err(|e| e.to_string())?;
                        let conn = client.get_connection().map_err(|e| e.to_string())?;
                        Ok((client, conn))
                    },
                    |(client, conn), batch| {
                        let first = batch.ids.first().copied().unwrap_or_default();
                        if let Some(fields) = &parity_fields {
                            let n = fulltext::normalize_metadata(&mut batch.metadata, fields)
                                .map_err(|e| format!("batch at row {first}: {e}"))?;
                            normalized.fetch_add(n as u64, Ordering::Relaxed);
                        }
                        upload_batch_with_retry(
                            client,
                            conn,
                            &config,
                            &format!("batch at row {first}"),
                            &batch.ids,
                            &batch.vectors,
                            &batch.metadata,
                        )
                    },
                    &|n| pb.inc(n),
                )?;
                pb.finish_with_message("Upload complete");
                Ok(stats)
            },
            |expected| self.wait_for_indexing(expected, WaitPhase::UploadTail),
        )?;
        if let Some(fields) = &parity_fields {
            println!(
                "\tfulltext parity: normalized whitespace in {} values of {fields:?}",
                normalized.load(Ordering::Relaxed)
            );
        }
        let upload_time = phases.upload_time;
        let total_time = read_time + phases.total_ingest_time;
        println!(
            "Upload time: {:.3}s ({:.0} records/sec, {} batches, at most {} in flight); index \
             time: {:.3}s, total time (read+upload+index): {:.3}s",
            upload_time,
            stats.rows as f64 / upload_time,
            stats.batches,
            stats.max_in_flight,
            phases.index_wait_time,
            total_time
        );
        let expected = stats.rows as usize;

        let mut conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut conn,
            &["hset"],
            "upload",
            self.commandstats_baseline.as_ref(),
        )?;
        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: expected,
            parallel: self.config.parallel,
            batch_size: self.config.batch_size,
            memory_usage: None,
            index_coverage: None,
            ingest_phases: Some(phases),
        })
    }
}
