//! Vector DB Benchmark - Pure Rust CLI
//!
//! This is the main entry point for the Rust implementation of vector-db-benchmark.
//! Supports redis and vectorsets engines.

mod cli;
mod compare;
mod config;
mod dataset;
mod download;
mod effective_config;
mod engine;
mod experiment;
mod ground_truth;
mod latency_digest;
mod metrics;
mod plot;
mod proc_cpu;
mod summary;

use clap::Parser;
use cli::Args;

fn main() {
    let args = Args::parse();

    if let Some(cli::Command::CompareResults(c)) = &args.command {
        if let Err(e) = compare::run(c) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Handle --describe option
    if let Some(describe) = &args.describe {
        match describe.as_str() {
            "datasets" => {
                if let Err(e) = config::describe_datasets(args.verbose) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                return;
            }
            "engines" => {
                if let Err(e) = config::describe_engines(args.verbose) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                return;
            }
            "build" => {
                // The `params.build` block every result file gets: commit,
                // target, features, profile, rustc and the OpenSSL / HDF5
                // actually loaded. Release packaging copies it into
                // BUILD-INFO.txt.
                println!(
                    "{}",
                    serde_json::to_string_pretty(&vector_db_benchmark::build_info::provenance())
                        .expect("a JSON value serializes")
                );
                return;
            }
            other => {
                eprintln!(
                    "Unknown describe option: '{}'. Use 'datasets', 'engines' or 'build'.",
                    other
                );
                std::process::exit(1);
            }
        }
    }

    // Handle --plot: render a chart from existing results instead of benchmarking.
    if args.plot.is_some() {
        if let Err(e) = plot::export_chart(&args) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Run experiments
    if let Err(e) = experiment::run(&args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
