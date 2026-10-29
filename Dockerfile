# Multi-stage Dockerfile for vector-db-benchmark (Rust)
#
# Build: docker build -t vector-db-benchmark .
# Run:   docker run --rm vector-db-benchmark --help

# ============================================================
# Stage 1: Build the Rust binary
# ============================================================
FROM rust:bookworm AS builder

# Install HDF5 development libraries, pkg-config, and dpkg-dev (for dpkg-architecture)
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    dpkg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# --- Dependency caching layer ---
# Copy only Cargo manifests and lock file first, then build dependencies
# with dummy source files. This layer is cached unless Cargo.toml/lock change.
COPY Cargo.toml Cargo.lock ./
# build.rs embeds the git commit (#366) and compiles src/build_info/resolve.rs;
# both must exist for the dependency layer to build.
COPY build.rs ./

# Create dummy source structure matching all targets
RUN mkdir -p src/bin/vector_db_benchmark/engine src/readers src/redisearch src/vectorsets \
    && echo "fn main() {}" > src/bin/bench_hdf5.rs \
    && echo "fn main() {}" > src/bin/bench_jsonl.rs \
    && echo "fn main() {}" > src/bin/bench_npy.rs \
    && echo "fn main() {}" > src/bin/vector_db_benchmark/main.rs \
    && touch src/lib.rs \
    && touch src/bin/vector_db_benchmark/engine/mod.rs
COPY src/build_info/resolve.rs ./src/build_info/resolve.rs

# Build dependencies only (this layer is cached)
# HDF5_DIR is set dynamically to support both amd64 and arm64
RUN HDF5_DIR=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/hdf5/serial \
    cargo build --release --bin vector-db-benchmark 2>/dev/null || true
# Remove dummy source and binary fingerprints, keep compiled dependencies
RUN rm -rf src target/release/vector-db-benchmark target/release/.fingerprint/vector_db_benchmark-*

# The build context has no .git, so the binary reports "git commit unknown"
# unless the commit is passed in: --build-arg VDB_BUILD_GIT_SHA=$(git rev-parse HEAD)
ARG VDB_BUILD_GIT_SHA=""
ENV VDB_BUILD_GIT_SHA=${VDB_BUILD_GIT_SHA}

# --- Source build layer ---
# Copy real source code
COPY src ./src

# Build the actual binary (dependencies are cached, only project code recompiles)
RUN HDF5_DIR=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/hdf5/serial \
    cargo build --release --bin vector-db-benchmark

# ============================================================
# Stage 2: Slim runtime image
# ============================================================
FROM debian:bookworm-slim

# Install HDF5 runtime library
RUN apt-get update && apt-get install -y \
    libhdf5-103-1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

WORKDIR /code

# Copy the binary from builder
COPY --from=builder /build/target/release/vector-db-benchmark /usr/local/bin/vector-db-benchmark

# Copy dataset definitions and experiment configurations into the image.
# project_root() searches for datasets/datasets.json as its marker.
COPY datasets/datasets.json /code/datasets/datasets.json
COPY datasets/random-100 /code/datasets/random-100
COPY experiments/configurations /code/experiments/configurations

# Create mount point directories.
# Downloaded datasets go to project_root/datasets/ (checked first by get_path).
RUN mkdir -p /code/datasets /code/results

ENTRYPOINT ["vector-db-benchmark"]
CMD ["--help"]
