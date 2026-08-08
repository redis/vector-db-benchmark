//! Vector DB Benchmark - Pure Rust implementation
//!
//! This library provides tools for benchmarking vector databases.
//! Modular design mirroring the Python v0/ structure:
//! - `readers` - Dataset readers for HDF5, JSONL, NPY, compound formats
//! - `config` - Redis configuration from environment
//! - `redis_client` - Redis connection management

pub mod config;
pub mod metrics;
pub mod parsers;
pub mod readers;
pub mod redis_client;
pub mod start_gate;
pub mod synthetic;

// Re-export commonly used types
pub use config::RedisConfig;
pub use readers::metadata::{MetadataItem, MetadataValue};
pub use readers::{read_compound_data, read_hdf5_vectors, read_jsonl_vectors, read_npy_vectors};
