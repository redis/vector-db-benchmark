//! `compare-results` subcommand: file I/O only. The comparison itself is
//! `vector_db_benchmark::compare_results` (unit-tested in the library).

use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::cli::CompareResultsArgs;
use vector_db_benchmark::compare_results::{compare, render_markdown};
use vector_db_benchmark::results_dump::{read_dump, Dump};

/// A dump file, or every `*.jsonl` directly inside a directory (sorted).
fn load(path: &Path) -> Result<Vec<(String, Dump)>, String> {
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut v: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
            .collect();
        v.sort();
        v
    } else {
        vec![path.to_path_buf()]
    };
    if files.is_empty() {
        return Err(format!("{}: no *.jsonl dumps", path.display()));
    }
    files
        .into_iter()
        .map(|f| {
            let file = fs::File::open(&f).map_err(|e| format!("{}: {e}", f.display()))?;
            let dump =
                read_dump(BufReader::new(file)).map_err(|e| format!("{}: {e}", f.display()))?;
            Ok((f.display().to_string(), dump))
        })
        .collect()
}

pub fn run(args: &CompareResultsArgs) -> Result<(), String> {
    let first = load(&args.first)?;
    let second = load(&args.second)?;
    let report = compare(&first, &second, &args.reference)?;
    let md = render_markdown(&report);
    print!("{md}");
    if let Some(p) = &args.markdown {
        fs::write(p, &md).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if let Some(p) = &args.json {
        let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        fs::write(p, json).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if !report.passed_allowing_unpaired(args.allow_unpaired) {
        return Err(format!(
            "comparison failed: {} quer{} answered by one engine and not the other, {} \
             failed query attempt(s) across both dumps, {} unpaired dump(s){} (see the report)",
            report.one_sided_total,
            if report.one_sided_total == 1 {
                "y was"
            } else {
                "ies were"
            },
            report.failed_attempts_total,
            report.unpaired.len(),
            if args.allow_unpaired {
                " (allowed)"
            } else {
                ""
            }
        ));
    }
    Ok(())
}
