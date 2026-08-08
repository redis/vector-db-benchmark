#!/usr/bin/env bash
#
# v0_check.sh — Compare Rust vs Python v0 benchmark results.
#
# Spins up redis:8.6.0 on port 6399, runs both Python and Rust benchmarks
# with the same engine config + dataset, compares precision/QPS/latency,
# then tears down docker.
#
# Usage:
#   ./scripts/v0_check.sh                     Run all combinations
#   ./scripts/v0_check.sh --all               Run all combinations (explicit)
#   ./scripts/v0_check.sh ENGINE DATASET      Run a single combination
#
# Exit code 0 = all checks pass, 1 = precision mismatch or failure.

set -euo pipefail

PORT=6399

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUST_BIN="$ROOT/target/release/vector-db-benchmark"
COMPOSE="$ROOT/tests/docker-compose.test.yml"
PY_RESULTS="$ROOT/v0/results"
RS_RESULTS="$ROOT/results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

# Track results across combinations
PASSED=()
FAILED=()

cleanup() {
    echo ""
    echo "=== Stopping redis ==="
    docker compose -f "$COMPOSE" down 2>/dev/null || true
}
trap cleanup EXIT

# ── Compare result files ──────────────────────────────────────
#
# compare_results PY_FILE RS_FILE
# Prints comparison table. Returns 0 if pass, 1 if fail.
compare_results() {
    local py_file="$1" rs_file="$2"

    python3 -c "
import json, sys

py = json.load(open('$py_file'))
rs = json.load(open('$rs_file'))

py_r = py['results']
rs_r = rs['results']

failed = False

print(f\"{'Metric':<20} {'Python v0':>15} {'Rust':>15} {'Status':>20}\")
print('=' * 72)

# Python v0 (like upstream qdrant/vector-db-benchmark) computes
# len(ids & expected[:top]) / top and stores it under 'mean_precisions'. That is
# recall@top, i.e. the Rust 'mean_recall' field — NOT the Rust
# 'mean_precision_at_returned', whose denominator is the number of results the
# engine returned (#217). The two agree only on full-width ground truth, which
# every dataset this script runs has; comparing the wrong pair here is precisely
# the mistake the rename was meant to make impossible.
KEY_MAP = {
    'mean_precisions': 'mean_recall',
}

for key in ['mean_precisions', 'rps', 'mean_time', 'p50_time', 'p95_time', 'p99_time']:
    rs_key = KEY_MAP.get(key, key)
    pv = py_r.get(key, 0.0)
    rv = rs_r.get(rs_key, 0.0)

    if key == 'mean_precisions':
        if abs(pv - rv) < 0.001:
            status = 'PASS'
        else:
            status = 'FAIL'
            failed = True
    elif key == 'rps':
        # Allow 30% tolerance for RPS — microbenchmarks with tiny datasets
        # have high variance due to async runtime overhead
        status = 'PASS (Rust >= Py)' if rv >= pv * 0.7 else 'FAIL (Rust < Py)'
        if rv < pv * 0.7:
            failed = True
    else:
        status = 'PASS (Rust <= Py)' if rv <= pv * 1.5 else 'WARN (Rust > Py)'

    label = key if rs_key == key else f'{key}→{rs_key}'
    print(f'{label:<20} {pv:>15.6f} {rv:>15.6f} {status:>20}')

print()
if failed:
    print('RESULT: FAIL')
    sys.exit(1)
else:
    print('RESULT: PASS')
    sys.exit(0)
"
}

# ── Run a single comparison ───────────────────────────────────
#
# run_comparison ENGINE DATASET
# Returns 0 if pass, 1 if fail.
run_comparison() {
    local engine="$1" dataset="$2"
    local label="${engine} / ${dataset}"

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  v0-check: ${label}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Flush Redis between combinations
    redis-cli -p "$PORT" FLUSHALL >/dev/null 2>&1 || true

    # Clean old results
    rm -f "$PY_RESULTS/${engine}-${dataset}-search-"*.json
    rm -f "$RS_RESULTS/${engine}-${dataset}-search-"*.json

    # ── Run Python v0 ──
    echo "=== Running Python v0: ${label} (parallel=100) ==="
    if ! (cd "$ROOT/v0" && REDIS_PORT="$PORT" REPETITIONS=1 poetry run python3 run.py \
        --engines "$engine" \
        --datasets "$dataset" \
        --parallels 100 \
        --no-skip-if-exists 2>&1) | grep -E "→|Experiment stage|Running"; then
        echo -e "${RED}WARNING: Python v0 output filtered, check logs${NC}"
    fi
    echo ""

    # Find Python result file
    local py_file
    py_file=$(ls -t "$PY_RESULTS/${engine}-${dataset}-search-"*.json 2>/dev/null | head -1)
    if [ -z "$py_file" ]; then
        echo -e "${RED}ERROR: No Python search result found for ${label}${NC}"
        FAILED+=("$label")
        return 1
    fi

    # ── Run Rust ──
    echo "=== Running Rust: ${label} (parallel=100) ==="
    if ! REDIS_PORT="$PORT" "$RUST_BIN" \
        --engines "$engine" \
        --datasets "$dataset" \
        --parallels 100 2>&1 | grep -E "→|Experiment stage|Running|Results saved"; then
        echo -e "${RED}WARNING: Rust output filtered, check logs${NC}"
    fi
    echo ""

    # Find Rust result file
    local rs_file
    rs_file=$(ls -t "$RS_RESULTS/${engine}-${dataset}-search-"*.json 2>/dev/null | head -1)
    if [ -z "$rs_file" ]; then
        echo -e "${RED}ERROR: No Rust search result found for ${label}${NC}"
        FAILED+=("$label")
        return 1
    fi

    # ── Compare ──
    echo "=== Comparing results ==="
    echo "  Python: $py_file"
    echo "  Rust:   $rs_file"
    echo ""

    local result
    result=$(compare_results "$py_file" "$rs_file" || true)
    echo "$result"
    echo ""

    if echo "$result" | grep -q "RESULT: PASS"; then
        echo -e "${GREEN}${BOLD}  PASS: ${label}${NC}"
        PASSED+=("$label")
        return 0
    else
        echo -e "${RED}${BOLD}  FAIL: ${label}${NC}"
        FAILED+=("$label")
        return 1
    fi
}

# ── Print summary ─────────────────────────────────────────────

print_summary() {
    local total=$(( ${#PASSED[@]} + ${#FAILED[@]} ))

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                      v0-check Summary                       ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════╣${NC}"

    for label in "${PASSED[@]}"; do
        printf "  ${GREEN}PASS${NC}  %s\n" "$label"
    done
    for label in "${FAILED[@]}"; do
        printf "  ${RED}FAIL${NC}  %s\n" "$label"
    done

    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Total: ${total}  |  Passed: ${#PASSED[@]}  |  Failed: ${#FAILED[@]}"
    echo ""

    if [ ${#FAILED[@]} -eq 0 ]; then
        echo -e "${GREEN}${BOLD}v0-check: ALL PASSED${NC}"
        return 0
    else
        echo -e "${RED}${BOLD}v0-check: ${#FAILED[@]} FAILED${NC}"
        return 1
    fi
}

# ── Pre-flight ────────────────────────────────────────────────

# Build Rust binary
echo "=== Building Rust binary ==="
cargo build --release --bin vector-db-benchmark --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -3
echo ""

# Check Python v0 environment
if ! (cd "$ROOT/v0" && poetry run python3 -c "from benchmark import DATASETS_DIR" 2>/dev/null); then
    echo -e "${RED}ERROR: Python v0 environment not set up. Run: cd v0 && poetry install${NC}"
    exit 1
fi

# ── Start Redis ───────────────────────────────────────────────

echo "=== Starting redis:8.6.0 on port $PORT ==="
docker compose -f "$COMPOSE" up -d --wait 2>/dev/null
# Wait for ping
for i in $(seq 1 10); do
    if redis-cli -p "$PORT" ping 2>/dev/null | grep -q PONG; then
        break
    fi
    sleep 0.5
done

if ! redis-cli -p "$PORT" ping 2>/dev/null | grep -q PONG; then
    echo -e "${RED}ERROR: Redis not responding on port $PORT${NC}"
    exit 1
fi

# Configure search workers to match core count
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
redis-cli -p "$PORT" CONFIG SET search-workers "$CORES" >/dev/null 2>&1 || true
echo "Redis ready (search-workers=$CORES)."
echo ""

# ── Dispatch ──────────────────────────────────────────────────

if [ $# -eq 0 ] || [ "${1:-}" = "--all" ]; then
    # ── Run all combinations ──
    echo -e "${BOLD}=== v0-check: running all combinations ===${NC}"

    COMBOS=(
        "redis-m-16-ef-128:h-and-m-2048-angular-filters"
        "redis-m-16-ef-128:glove-25-angular"
        "redis-m-16-ef-128:random-100"
    )

    for combo in "${COMBOS[@]}"; do
        engine="${combo%%:*}"
        dataset="${combo#*:}"
        run_comparison "$engine" "$dataset" || true
    done

    print_summary
    exit $?

else
    # ── Single combination mode ──
    ENGINE="${1}"
    DATASET="${2:-h-and-m-2048-angular-filters}"

    run_comparison "$ENGINE" "$DATASET" || true

    if [ ${#FAILED[@]} -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
fi
