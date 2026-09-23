# Makefile for vector-db-benchmark Rust implementation
#
# Targets:
#   make setup               - Install system deps and Rust toolchain
#   make install             - Build and install binary to PREFIX (/usr/local)
#   make vector-db-benchmark - Build the main CLI binary
#   make test                - Run Rust unit tests (no docker needed)
#   make integration-test    - Run integration tests (requires redis:8.6.0 docker)
#   make benchmark           - Run Rust microbenchmarks
#   make check               - Run linting (clippy) and formatting (rustfmt) checks
#   make build               - Build Rust code in release mode
#   make docker-build        - Build Docker image
#   make docker-integration  - Run benchmark in Docker against Redis
#   make clean               - Clean build artifacts

# Environment variables
ARCH := $(shell dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)
export HDF5_DIR ?= /usr/lib/$(ARCH)/hdf5/serial

# Directories
DATASETS_DIR := datasets

# Docker
IMAGE_TAG ?= latest

# Default target
.PHONY: all
all: check build test

# ============================================================
# SETUP - Install system dependencies and Rust toolchain
# ============================================================

.PHONY: setup
setup:
	@echo "=== Setting up development environment ==="
	@OS=$$(uname -s); \
	if [ "$$OS" = "Linux" ]; then \
		echo "Detected Linux — installing libhdf5-dev and pkg-config via apt..."; \
		sudo apt-get update && sudo apt-get install -y libhdf5-dev pkg-config; \
	elif [ "$$OS" = "Darwin" ]; then \
		echo "Detected macOS — installing hdf5 and pkg-config via brew..."; \
		brew install hdf5 pkg-config; \
	else \
		echo "Unsupported OS: $$OS (expected Linux or Darwin)"; \
		exit 1; \
	fi
	@if ! command -v cargo >/dev/null 2>&1; then \
		echo "Rust not found — installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo "NOTE: Run 'source $$HOME/.cargo/env' to add cargo to your current shell."; \
	else \
		echo "Rust already installed — running rustup update..."; \
		rustup update; \
	fi
	@echo ""
	@echo "=== Setup complete! Next step: make build ==="

# ============================================================
# BUILD
# ============================================================

.PHONY: build
build:
	@echo "=== Building Rust code (release) ==="
	cargo build --release

.PHONY: build-debug
build-debug:
	@echo "=== Building Rust code (debug) ==="
	cargo build

# Main CLI binary target
.PHONY: vector-db-benchmark
vector-db-benchmark:
	@echo "=== Building vector-db-benchmark CLI ==="
	cargo build --release --bin vector-db-benchmark
	@echo ""
	@echo "Binary built at: target/release/vector-db-benchmark"
	@echo ""
	@echo "Usage:"
	@echo "  ./target/release/vector-db-benchmark --help"
	@echo "  ./target/release/vector-db-benchmark --describe datasets"
	@echo "  ./target/release/vector-db-benchmark --engines 'redis-rs*' --datasets 'glove*' --skip-search"

# ============================================================
# INSTALL - Install binary to system path
# ============================================================

PREFIX ?= /usr/local

.PHONY: install
install: build
	@echo "=== Installing vector-db-benchmark to $(PREFIX)/bin ==="
	install -d $(PREFIX)/bin
	install -m 755 target/release/vector-db-benchmark $(PREFIX)/bin/vector-db-benchmark
	@echo "Installed: $(PREFIX)/bin/vector-db-benchmark"

# ============================================================
# TEST - Run Rust unit tests with dataset coverage
# ============================================================

.PHONY: test
test:
	@echo "=== Running Rust unit tests ==="
	cargo test --lib --release -- --nocapture

.PHONY: test-verbose
test-verbose:
	@echo "=== Running Rust unit tests (verbose) ==="
	cargo test --lib --release -- --nocapture --test-threads=1

# ============================================================
# INTEGRATION TEST - Requires redis:8.6.0 on port 6399
# ============================================================

.PHONY: integration-test
integration-test:
	@echo "=== Starting redis:8.6.0 for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d --wait
	@echo "=== Running integration tests ==="
	cargo test --test integration_redis --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping redis ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-pgvector
integration-test-pgvector:
	@echo "=== Starting PgVector (pgvector/pgvector:pg16) for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d pgvector --wait
	@echo "=== Running PgVector integration tests ==="
	PGVECTOR_PORT=5433 cargo test --test integration_pgvector --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping PgVector ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-qdrant
integration-test-qdrant:
	@echo "=== Starting Qdrant v1.13.4 for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d qdrant --wait
	@echo "=== Running Qdrant integration tests ==="
	QDRANT_GRPC_PORT=6335 cargo test --test integration_qdrant --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping Qdrant ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-elasticsearch
integration-test-elasticsearch:
	@echo "=== Starting Elasticsearch 8.10.2 for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d elasticsearch --wait
	@echo "=== Running Elasticsearch integration tests ==="
	ELASTIC_PORT=9201 cargo test --test integration_elasticsearch --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping Elasticsearch ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-milvus
integration-test-milvus:
	@echo "=== Starting Milvus 2.5.6 (+ etcd + minio) for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d milvus --wait
	@echo "=== Running Milvus integration tests ==="
	MILVUS_PORT=19531 cargo test --test integration_milvus --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping Milvus ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-weaviate
integration-test-weaviate:
	@echo "=== Starting Weaviate 1.28.9 for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d weaviate --wait
	@echo "=== Running Weaviate integration tests ==="
	WEAVIATE_HTTP_PORT=8081 cargo test --test integration_weaviate --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping Weaviate ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-opensearch
integration-test-opensearch:
	@echo "=== Starting OpenSearch 2.19.2 for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d opensearch --wait
	@echo "=== Running OpenSearch integration tests ==="
	OPENSEARCH_PORT=9202 cargo test --test integration_opensearch --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping OpenSearch ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-mongodb
integration-test-mongodb:
	@echo "=== Starting MongoDB Atlas Local 8.0.4 for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d mongodb-search --wait
	@echo "=== Running MongoDB integration tests ==="
	MONGODB_PORT=27018 cargo test --test integration_mongodb --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping MongoDB ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-valkey
integration-test-valkey:
	@echo "=== Starting Valkey Bundle for integration tests ==="
	docker compose -f tests/docker-compose.test.yml up -d valkey --wait
	@echo "=== Running Valkey integration tests ==="
	VALKEY_PORT=6380 cargo test --test integration_valkey --release -- --nocapture --test-threads=1; \
	EXIT_CODE=$$?; \
	echo "=== Stopping Valkey ===" ; \
	docker compose -f tests/docker-compose.test.yml down ; \
	exit $$EXIT_CODE

.PHONY: integration-test-no-docker
integration-test-no-docker:
	@echo "=== Running integration tests (assumes redis on port 6399) ==="
	cargo test --test integration_redis --release -- --nocapture --test-threads=1

# ============================================================
# BENCHMARK - Run Rust microbenchmarks
# ============================================================

.PHONY: benchmark
benchmark: build
	@echo "=== Running Rust microbenchmarks ==="
	@echo ""
	@echo "--- HDF5 Benchmark (glove-25-angular: 1.18M × 25d) ---"
	@if [ -f $(DATASETS_DIR)/glove-25-angular/glove-25-angular.hdf5 ]; then \
		./target/release/bench_hdf5 $(DATASETS_DIR)/glove-25-angular/glove-25-angular.hdf5 3; \
	else \
		echo "Skipping: $(DATASETS_DIR)/glove-25-angular/glove-25-angular.hdf5 not found"; \
	fi
	@echo ""
	@echo "--- NPY Benchmark (h-and-m: 105K × 2048d) ---"
	@if [ -f $(DATASETS_DIR)/h-and-m-2048-angular/hnm/vectors.npy ]; then \
		./target/release/bench_npy $(DATASETS_DIR)/h-and-m-2048-angular/hnm/vectors.npy 3; \
	else \
		echo "Skipping: h-and-m vectors.npy not found"; \
	fi
	@echo ""
	@echo "--- JSONL Benchmark (random-100k: 100K × 100d) ---"
	@if [ -f $(DATASETS_DIR)/random-100k/vectors.jsonl ]; then \
		./target/release/bench_jsonl $(DATASETS_DIR)/random-100k/vectors.jsonl 3; \
	else \
		echo "Skipping: $(DATASETS_DIR)/random-100k/vectors.jsonl not found"; \
	fi

# ============================================================
# CHECK - Linting and formatting
# ============================================================

.PHONY: check
check: fmt-check lint

.PHONY: lint
lint:
	@echo "=== Running Clippy (Rust linter) ==="
	cargo clippy

.PHONY: check-strict
check-strict: fmt-check lint-strict

.PHONY: lint-strict
lint-strict:
	@echo "=== Running Clippy (strict mode - warnings as errors) ==="
	cargo clippy -- -D warnings

.PHONY: fmt-check
fmt-check:
	@echo "=== Checking Rust formatting ==="
	cargo fmt --check

.PHONY: fmt
fmt:
	@echo "=== Formatting Rust code ==="
	cargo fmt

# ============================================================
# DATASETS - Prepare the MS MARCO v2.1 + Cohere embed-v3 corpus
# ============================================================

# Size is chosen by name, not by a flag: msmarco-cohere-1024-{100K,1M,10M}-cosine.
MSMARCO_DATASET ?= msmarco-cohere-1024-100K-cosine

.PHONY: prepare-msmarco
prepare-msmarco:
	@echo "=== Preparing $(MSMARCO_DATASET) (vectors + metadata) ==="
	cargo run --release --bin prepare-msmarco -- --dataset $(MSMARCO_DATASET)

# ============================================================
# V0-CHECK - Compare Rust vs Python v0 precision/QPS/latency
# ============================================================

.PHONY: v0-check
v0-check: vector-db-benchmark
	@echo "=== Running v0-check (Rust vs Python comparison) ==="
	./scripts/v0_check.sh $(ENGINE) $(DATASET)

.PHONY: v0-check-all
v0-check-all: vector-db-benchmark
	@echo "=== Running v0-check (all combinations) ==="
	./scripts/v0_check.sh --all

# ============================================================
# DOCKER - Build image and run Docker-based integration tests
# ============================================================

.PHONY: docker-build
docker-build:
	@echo "=== Building Docker image vector-db-benchmark:$(IMAGE_TAG) ==="
	docker build -t vector-db-benchmark:$(IMAGE_TAG) .

.PHONY: docker-integration
docker-integration: docker-build
	@echo "=== Running Docker integration test (h-and-m-2048-angular-filters + Redis) ==="
	docker compose -f tests/docker-compose.docker-test.yml up -d redis --wait
	@echo "=== Configuring Redis search-workers ==="
	docker compose -f tests/docker-compose.docker-test.yml exec redis redis-cli CONFIG SET search-workers 8
	@echo "=== Starting benchmark container ==="
	docker compose -f tests/docker-compose.docker-test.yml run --rm benchmark; \
	EXIT_CODE=$$?; \
	echo "=== Stopping services ===" ; \
	docker compose -f tests/docker-compose.docker-test.yml down ; \
	exit $$EXIT_CODE

.PHONY: docker-integration-fast
docker-integration-fast: docker-build
	@echo "=== Running fast Docker integration test (random-100 + Redis) ==="
	docker compose -f tests/docker-compose.ci-test.yml up -d redis --wait
	@echo "=== Configuring Redis search-workers ==="
	docker compose -f tests/docker-compose.ci-test.yml exec redis redis-cli CONFIG SET search-workers 8
	@echo "=== Starting benchmark container ==="
	docker compose -f tests/docker-compose.ci-test.yml run --rm benchmark; \
	EXIT_CODE=$$?; \
	echo "=== Stopping services ===" ; \
	docker compose -f tests/docker-compose.ci-test.yml down ; \
	exit $$EXIT_CODE

# ============================================================
# CLEAN
# ============================================================

.PHONY: clean
clean:
	@echo "=== Cleaning Rust build artifacts ==="
	cargo clean

# ============================================================
# HELP
# ============================================================

.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "  make setup             - Install system deps (libhdf5, pkg-config) and Rust toolchain"
	@echo ""
	@echo "  Build & Test:"
	@echo "  make build             - Build Rust code in release mode"
	@echo "  make install           - Build and install binary to PREFIX (default: /usr/local)"
	@echo "  make test              - Run Rust unit tests (no docker needed)"
	@echo "  make check             - Run linting (clippy) and formatting checks"
	@echo "  make check-strict      - Run linting with warnings as errors"
	@echo "  make fmt               - Auto-format Rust code"
	@echo "  make benchmark         - Run Rust microbenchmarks (HDF5, NPY, JSONL)"
	@echo ""
	@echo "  Integration Tests (each starts/stops Docker containers):"
	@echo "  make integration-test                - Redis"
	@echo "  make integration-test-elasticsearch   - Elasticsearch 8.10.2"
	@echo "  make integration-test-opensearch      - OpenSearch 2.19.2"
	@echo "  make integration-test-pgvector        - PgVector (PostgreSQL 16)"
	@echo "  make integration-test-qdrant          - Qdrant v1.13.4"
	@echo "  make integration-test-weaviate        - Weaviate 1.28.9"
	@echo "  make integration-test-milvus          - Milvus 2.5.6"
	@echo "  make integration-test-mongodb         - MongoDB Atlas Local 8.0.4"
	@echo "  make integration-test-valkey          - Valkey Bundle (latest)"
	@echo ""
	@echo "  Docker:"
	@echo "  make docker-build                - Build Docker image (IMAGE_TAG=latest)"
	@echo "  make docker-integration          - Run benchmark in Docker (h-and-m dataset)"
	@echo "  make docker-integration-fast     - Run fast benchmark in Docker (random-100 dataset)"
	@echo ""
	@echo "  Datasets:"
	@echo "  make prepare-msmarco   - Download+build the MS MARCO v2.1 / Cohere embed-v3 corpus"
	@echo "                           (vectors + real document metadata). Size is picked by name:"
	@echo "                           MSMARCO_DATASET=msmarco-cohere-1024-{100K,1M,10M}-cosine"
	@echo ""
	@echo "  Validation:"
	@echo "  make v0-check          - Compare Rust vs Python v0 (precision, QPS, latency)"
	@echo ""
	@echo "  make clean             - Clean build artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  HDF5_DIR          - Path to HDF5 library (default: /usr/lib/<arch>/hdf5/serial)"

