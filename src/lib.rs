//! Vector DB Benchmark - Pure Rust implementation
//!
//! This library provides tools for benchmarking vector databases.
//! Modular design mirroring the Python v0/ structure:
//! - `readers` - Dataset readers for HDF5, JSONL, NPY, compound formats
//! - `msmarco` - MS MARCO v2.1 + Cohere embed-v3 corpus preparation (vectors + metadata)
//! - `query_filter` - the one door from a dataset's `conditions` JSON to a filter
//! - `workload` - per-search-config query family and the engine/workload table
//! - `results_dump` - per-query result records and the `--dump-results` format
//! - `compare_results` - scores one engine's dumps against a declared reference
//! - `config` - Redis configuration from environment
//! - `redis_client` - Redis connection management

pub mod compare_results;
pub mod config;
pub mod filter_only;
pub mod fulltext;
pub mod metrics;
pub mod msmarco;
pub mod opensearch_maintenance;
pub mod opensearch_query;
pub mod parsers;
pub mod query_filter;
pub mod readers;
pub mod redis_client;
pub mod redis_numeric_range;
pub mod results_dump;
pub mod start_gate;
pub mod synthetic;
pub mod workload;

// Re-export commonly used types
pub use config::RedisConfig;
pub use readers::metadata::{MetadataItem, MetadataValue};
pub use readers::{read_compound_data, read_hdf5_vectors, read_jsonl_vectors, read_npy_vectors};
