//! Ingestion reported as phases (#344).
//!
//! The ingestion KPI is "time to a searchable, fully indexed corpus". An
//! engine's `upload()` splits it into two phases measured on ONE clock:
//!
//! | phase             | starts                   | ends                                   |
//! |-------------------|--------------------------|----------------------------------------|
//! | `upload_time`     | first batch sent         | last batch acknowledged                |
//! | `index_wait_time` | last batch acknowledged  | index reports the corpus searchable    |
//!
//! `total_ingest_time = upload_time + index_wait_time`, i.e. the wall time from
//! the first upload to index-ready. Reading the dataset from disk happens
//! before the first upload and is reported separately as `read_time`; the
//! upload file's legacy `total_time` stays `read_time + total_ingest_time`, so
//! existing consumers keep their meaning.
//!
//! What "index-ready" means is engine-specific and named in `index_wait`
//! ([`IndexWait`]). Because both phases come from offsets of one
//! [`IngestClock`], the phases sum to the wall time between the timestamped
//! boundaries by construction, and [`IngestPhases::validate`] refuses a report
//! where they do not.

use serde_json::{json, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Seconds of disagreement tolerated between derived quantities. Every
/// duration comes from one monotonic clock, so only f64 rounding of epoch
/// timestamps (~2e-7 s at 2026 epochs) separates them.
pub const TOLERANCE_SECS: f64 = 1e-3;

/// What the index-wait phase waited for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexWait {
    /// RediSearch `FT.INFO` until `percent_indexed` reaches 1 and the doc
    /// count reaches the corpus size.
    RedisPercentIndexed,
    /// OpenSearch force-merge to one segment, the refresh that makes it
    /// searchable, and the cluster-health wait.
    OpenSearchForceMergeRefresh,
}

impl IndexWait {
    pub fn as_str(self) -> &'static str {
        match self {
            IndexWait::RedisPercentIndexed => "redis_ft_info_percent_indexed",
            IndexWait::OpenSearchForceMergeRefresh => "opensearch_force_merge_refresh",
        }
    }
}

/// Measures the two phases on one monotonic clock, anchored to wall-clock
/// time once so the boundaries can be published as timestamps.
#[derive(Debug, Clone)]
pub struct IngestClock {
    start: Instant,
    start_epoch: f64,
    upload_done: Option<Instant>,
}

impl IngestClock {
    /// Call immediately before the first batch is sent.
    pub fn start() -> Self {
        let start_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        IngestClock {
            start: Instant::now(),
            start_epoch,
            upload_done: None,
        }
    }

    /// For an ingest that uploads nothing (a config that reuses a corpus
    /// another config uploaded and only waits for its own index to backfill):
    /// the upload phase is zero-length by construction, so `upload_time` is
    /// exactly `0.0` — not the nanoseconds between two clock reads, which
    /// would publish an `upload_docs_per_sec` of ~10^11 — and
    /// `upload_docs_per_sec` is `null`.
    pub fn without_upload() -> Self {
        let mut clock = Self::start();
        clock.upload_done = Some(clock.start);
        clock
    }

    /// Call when the last batch is acknowledged. Returns `upload_time`.
    pub fn upload_done(&mut self) -> f64 {
        let now = Instant::now();
        self.upload_done = Some(now);
        now.duration_since(self.start).as_secs_f64()
    }

    /// Call when the index reports the corpus searchable.
    pub fn index_ready(
        &self,
        documents: usize,
        read_time: f64,
        index_wait: IndexWait,
    ) -> Result<IngestPhases, String> {
        let ready = Instant::now();
        let upload_done = self
            .upload_done
            .ok_or("ingest clock: index_ready() before upload_done()")?;
        let phases = IngestPhases::from_offsets(
            self.start_epoch,
            upload_done.duration_since(self.start).as_secs_f64(),
            ready.duration_since(self.start).as_secs_f64(),
            documents,
            read_time,
            index_wait,
        );
        phases.validate()?;
        Ok(phases)
    }
}

/// One ingest, as phases with timestamped boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestPhases {
    /// Client-side dataset read, before the first upload — including, when the
    /// index serves filter-only (`full` / `no-vector`), the metadata-domain
    /// scan of the corpus. Not part of `total_ingest_time`.
    pub read_time: f64,
    pub upload_time: f64,
    pub index_wait_time: f64,
    pub total_ingest_time: f64,
    /// Unix epoch seconds.
    pub upload_started_at: f64,
    pub upload_finished_at: f64,
    pub index_ready_at: f64,
    /// Documents the ingest made searchable (the upload count).
    pub documents: usize,
    pub index_wait: IndexWait,
}

impl IngestPhases {
    /// Build from offsets (seconds after `start_epoch`) on one clock.
    pub fn from_offsets(
        start_epoch: f64,
        upload_done: f64,
        index_ready: f64,
        documents: usize,
        read_time: f64,
        index_wait: IndexWait,
    ) -> Self {
        IngestPhases {
            read_time,
            upload_time: upload_done,
            index_wait_time: index_ready - upload_done,
            total_ingest_time: index_ready,
            upload_started_at: start_epoch,
            upload_finished_at: start_epoch + upload_done,
            index_ready_at: start_epoch + index_ready,
            documents,
            index_wait,
        }
    }

    /// Documents per second of the upload phase alone.
    pub fn upload_docs_per_sec(&self) -> Option<f64> {
        rate(self.documents, self.upload_time)
    }

    /// Documents per second to a searchable index (upload + index wait).
    pub fn ingest_docs_per_sec(&self) -> Option<f64> {
        rate(self.documents, self.total_ingest_time)
    }

    /// The phase invariants: every quantity finite and non-negative, the
    /// phases sum to the total, and the total equals the wall time between
    /// the first-upload and index-ready timestamps.
    pub fn validate(&self) -> Result<(), String> {
        let named = [
            ("read_time", self.read_time),
            ("upload_time", self.upload_time),
            ("index_wait_time", self.index_wait_time),
            ("total_ingest_time", self.total_ingest_time),
            ("upload_started_at", self.upload_started_at),
            ("upload_finished_at", self.upload_finished_at),
            ("index_ready_at", self.index_ready_at),
        ];
        if let Some((name, v)) = named.iter().find(|(_, v)| !v.is_finite() || *v < 0.0) {
            return Err(format!(
                "ingest phases: {name} = {v} is not a finite, non-negative time"
            ));
        }
        let sum = self.upload_time + self.index_wait_time;
        if (sum - self.total_ingest_time).abs() > TOLERANCE_SECS {
            return Err(format!(
                "ingest phases: upload_time + index_wait_time = {sum:.6}s but \
                 total_ingest_time = {:.6}s",
                self.total_ingest_time
            ));
        }
        let wall = self.index_ready_at - self.upload_started_at;
        if (wall - self.total_ingest_time).abs() > TOLERANCE_SECS {
            return Err(format!(
                "ingest phases: index_ready_at - upload_started_at = {wall:.6}s but \
                 total_ingest_time = {:.6}s",
                self.total_ingest_time
            ));
        }
        let upload_wall = self.upload_finished_at - self.upload_started_at;
        if (upload_wall - self.upload_time).abs() > TOLERANCE_SECS {
            return Err(format!(
                "ingest phases: upload_finished_at - upload_started_at = {upload_wall:.6}s but \
                 upload_time = {:.6}s",
                self.upload_time
            ));
        }
        Ok(())
    }

    /// Cross-check against the legacy upload-file fields the engine reported:
    /// `upload_time` must be the same measurement, and the legacy `total_time`
    /// (read + upload + index) must equal `read_time + total_ingest_time`.
    pub fn check_legacy_fields(&self, upload_time: f64, total_time: f64) -> Result<(), String> {
        self.validate()?;
        if (upload_time - self.upload_time).abs() > TOLERANCE_SECS {
            return Err(format!(
                "ingest phases: upload_time {:.6}s disagrees with the reported upload_time \
                 {upload_time:.6}s",
                self.upload_time
            ));
        }
        let expected_total = self.read_time + self.total_ingest_time;
        if (total_time - expected_total).abs() > TOLERANCE_SECS {
            return Err(format!(
                "ingest phases: total_time {total_time:.6}s is not read_time + \
                 total_ingest_time = {expected_total:.6}s"
            ));
        }
        Ok(())
    }

    /// Add the phase keys to an upload file's `results` object. Every legacy
    /// key (`upload_time`, `total_time`, `upload_count`, …) is left untouched.
    pub fn write_into(&self, results: &mut Value) {
        results["read_time"] = json!(self.read_time);
        results["index_wait_time"] = json!(self.index_wait_time);
        results["total_ingest_time"] = json!(self.total_ingest_time);
        results["upload_docs_per_sec"] = json!(self.upload_docs_per_sec());
        results["ingest_docs_per_sec"] = json!(self.ingest_docs_per_sec());
        results["ingest_phases"] = json!({
            "clock": "unix_epoch_seconds",
            "upload_started_at": self.upload_started_at,
            "upload_finished_at": self.upload_finished_at,
            "index_ready_at": self.index_ready_at,
            "index_wait": self.index_wait.as_str(),
            "documents": self.documents,
        });
    }
}

fn rate(documents: usize, secs: f64) -> Option<f64> {
    (secs > 0.0).then(|| documents as f64 / secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ingest_without_upload_has_an_exactly_empty_upload_phase() {
        let clock = IngestClock::without_upload();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let p = clock
            .index_ready(1_000_000, 0.0, IndexWait::RedisPercentIndexed)
            .unwrap();
        assert_eq!(p.upload_time, 0.0);
        assert_eq!(p.upload_finished_at, p.upload_started_at);
        assert!(p.index_wait_time > 0.0);
        assert_eq!(p.index_wait_time, p.total_ingest_time);
        assert_eq!(p.upload_docs_per_sec(), None);
        let mut results = serde_json::json!({});
        p.write_into(&mut results);
        assert!(results["upload_docs_per_sec"].is_null(), "{results}");
        assert!(results["ingest_docs_per_sec"].as_f64().unwrap() > 0.0);
        p.check_legacy_fields(0.0, p.total_ingest_time).unwrap();
    }

    fn phases() -> IngestPhases {
        IngestPhases::from_offsets(
            1_800_000_000.25,
            12.5,
            20.0,
            1000,
            3.0,
            IndexWait::RedisPercentIndexed,
        )
    }

    #[test]
    fn phases_sum_to_the_wall_time_between_the_boundaries() {
        let p = phases();
        p.validate().unwrap();
        assert_eq!(p.upload_time, 12.5);
        assert_eq!(p.index_wait_time, 7.5);
        assert_eq!(p.total_ingest_time, 20.0);
        assert!((p.index_ready_at - p.upload_started_at - 20.0).abs() < 1e-6);
        assert!((p.upload_finished_at - p.upload_started_at - 12.5).abs() < 1e-6);
        assert_eq!(p.upload_docs_per_sec(), Some(80.0));
        assert_eq!(p.ingest_docs_per_sec(), Some(50.0));
    }

    #[test]
    fn validate_refuses_each_broken_invariant() {
        let mut p = phases();
        p.index_wait_time += 0.5;
        assert!(p
            .validate()
            .unwrap_err()
            .contains("upload_time + index_wait_time"));
        let mut p = phases();
        p.index_ready_at += 0.5;
        assert!(p.validate().unwrap_err().contains("index_ready_at"));
        let mut p = phases();
        p.upload_finished_at -= 0.5;
        assert!(p.validate().unwrap_err().contains("upload_finished_at"));
        let mut p = phases();
        p.read_time = f64::NAN;
        assert!(p.validate().unwrap_err().contains("read_time"));
        let mut p = phases();
        p.index_wait_time = -0.0001;
        p.total_ingest_time -= 0.0001;
        assert!(p.validate().is_err());
    }

    #[test]
    fn legacy_fields_must_be_the_same_measurement() {
        let p = phases();
        p.check_legacy_fields(12.5, 23.0).unwrap();
        assert!(p
            .check_legacy_fields(12.6, 23.0)
            .unwrap_err()
            .contains("upload_time"));
        // total_time without the read phase, or with the index wait twice.
        for bad in [20.0, 30.5] {
            assert!(p
                .check_legacy_fields(12.5, bad)
                .unwrap_err()
                .contains("total_time"));
        }
    }

    #[test]
    fn written_keys_extend_without_touching_legacy_ones() {
        let mut results = json!({"upload_time": 12.5, "total_time": 23.0, "upload_count": 1000});
        phases().write_into(&mut results);
        assert_eq!(results["upload_time"], 12.5);
        assert_eq!(results["total_time"], 23.0);
        assert_eq!(results["upload_count"], 1000);
        assert_eq!(results["read_time"], 3.0);
        assert_eq!(results["index_wait_time"], 7.5);
        assert_eq!(results["total_ingest_time"], 20.0);
        assert_eq!(results["upload_docs_per_sec"], 80.0);
        assert_eq!(results["ingest_docs_per_sec"], 50.0);
        let b = &results["ingest_phases"];
        assert_eq!(b["clock"], "unix_epoch_seconds");
        assert_eq!(b["index_wait"], "redis_ft_info_percent_indexed");
        assert_eq!(b["documents"], 1000);
        assert_eq!(b["upload_started_at"], 1_800_000_000.25);
        // A zero-length phase has no rate rather than an infinite one.
        let mut results = json!({});
        IngestPhases::from_offsets(0.0, 0.0, 0.0, 5, 0.0, IndexWait::RedisPercentIndexed)
            .write_into(&mut results);
        assert!(results["upload_docs_per_sec"].is_null());
    }

    #[test]
    fn the_clock_produces_valid_phases_and_needs_the_upload_mark() {
        let mut clock = IngestClock::start();
        assert!(clock
            .index_ready(1, 0.0, IndexWait::OpenSearchForceMergeRefresh)
            .unwrap_err()
            .contains("before upload_done"));
        let upload = clock.upload_done();
        let p = clock
            .index_ready(7, 0.5, IndexWait::OpenSearchForceMergeRefresh)
            .unwrap();
        assert_eq!(p.upload_time, upload);
        assert!(p.index_wait_time >= 0.0);
        assert_eq!(p.documents, 7);
        assert_eq!(p.index_wait.as_str(), "opensearch_force_merge_refresh");
        p.validate().unwrap();
    }
}
