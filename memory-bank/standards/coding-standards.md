# Coding Standards

## Overview
Idiomatic Rust conventions with pragmatic strictness. Prioritize clarity and correctness over cleverness.

## Code Formatting

**Tool**: rustfmt (default configuration)
**Enforcement**: Before commit / CI

## Linting

**Tool**: clippy
**Strictness**: Balanced — use `#[allow(...)]` with justification when needed (e.g., `#[allow(dead_code)]` for future-use fields)

**Key Rules**:
- Fix all clippy warnings unless explicitly suppressed with rationale
- No unused imports in committed code

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Variables | snake_case | `connection_url`, `auth_part` |
| Functions | snake_case | `from_env`, `read_hdf5_vectors` |
| Structs / Enums | PascalCase | `RedisConfig`, `MetadataValue` |
| Traits | PascalCase | `VectorReader` |
| Constants | UPPER_SNAKE | `TEST_PORT`, `TEST_HOST` |
| Modules | snake_case | `redis_client`, `hdf5_reader` |
| Crate name | snake_case | `vector_db_benchmark` |

**File Naming**:
- Modules: `snake_case.rs`
- Module directories: `mod.rs` pattern

## File Organization

**Pattern**: Module-based

```text
src/
  lib.rs                          # Library root, public re-exports
  config.rs                       # Shared configuration (RedisConfig)
  redis_client.rs                 # Redis connection management
  readers/                        # Dataset reader modules
    mod.rs
    hdf5_reader.rs
    jsonl_reader.rs
    npy_reader.rs
    compound_reader.rs
    metadata.rs
  redisearch/                     # RediSearch engine module
    mod.rs
    configure.rs
    parser.rs
    search.rs
    upload.rs
  bin/
    bench_jsonl.rs                # Standalone benchmark binaries
    bench_npy.rs
    bench_hdf5.rs
    vector_db_benchmark/          # Main CLI binary
      main.rs
      cli.rs
      config.rs
      dataset.rs
      download.rs
      experiment.rs
      engine/
        mod.rs
        redis.rs
        vectorsets.rs
tests/
  integration_redis.rs            # Integration tests (require Docker)
  docker-compose.test.yml         # Test infrastructure
```

**Conventions**:
- Engine modules follow a consistent structure: `configure`, `search`, `upload`, `parser`
- Integration tests live in `tests/` (separate from src)
- Test infrastructure (Docker Compose) co-located with tests

## Testing Strategy

**Framework**: cargo test (built-in)
**Dev Dependencies**: tempfile, rand

**Test Types**:

| Type | Location | When to Use |
|------|----------|-------------|
| Unit tests | `#[cfg(test)] mod tests` inline | Pure logic, parsers, config |
| Integration tests | `tests/*.rs` | Tests requiring Redis/Docker |

**Conventions**:
- Integration tests require Docker: `docker compose -f tests/docker-compose.test.yml up -d`
- Run integration tests single-threaded: `cargo test --test integration_{engine} -- --test-threads=1`
- Each engine has its own Makefile target: `make integration-test-{engine}`
- Engines with integration tests: redis, pgvector, qdrant, elasticsearch, opensearch, weaviate, milvus, mongodb
- Use descriptive `expect()` messages for test setup failures
- Helper functions at top of test files (e.g., `get_test_connection`, `flush_db`)

## Error Handling

**Pattern**: Idiomatic Rust `Result<T, E>`

- Library code: Return `Result` types, propagate with `?`
- Binary/CLI code: `expect()` with descriptive messages for unrecoverable setup errors
- `unwrap()` acceptable in tests and one-off scripts
- No custom error types yet — use `Box<dyn Error>` or specific error types from dependencies

## Logging

**Tool**: `println!` / `eprintln!` (no logging framework)
**Format**: Human-readable text output

Appropriate for a CLI benchmarking tool. Progress feedback via `indicatif` progress bars rather than log lines.
