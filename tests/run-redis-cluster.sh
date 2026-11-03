#!/usr/bin/env bash
# Disposable loopback-only transport cluster; no Docker or host service install.
set -euo pipefail
exec 9>/tmp/vector-db-benchmark-cluster-test.lock
flock -w 300 9
source_sha=88422181efb0c9c0abba332e3e391d409e1e13714b838931669235e5796f704b
build_dir="$PWD/target/redis-cluster-test-$source_sha"
mkdir -p "$build_dir"
if ! (cd "$build_dir" && sha256sum -c binaries.sha256 >/dev/null 2>&1); then
  curl --fail --location --retry 3 --max-time 120 https://download.redis.io/releases/redis-8.8.0.tar.gz -o "$build_dir/redis.tar.gz"
  printf '%s  %s\n' "$source_sha" "$build_dir/redis.tar.gz" | sha256sum -c -
  tar -xzf "$build_dir/redis.tar.gz" -C "$build_dir"
  make -C "$build_dir/redis-8.8.0/src" -j2 MALLOC=libc BUILD_TLS=no USE_SYSTEMD=no redis-server redis-cli
  (cd "$build_dir" && sha256sum redis-8.8.0/src/redis-server redis-8.8.0/src/redis-cli > binaries.sha256)
fi
server="$build_dir/redis-8.8.0/src/redis-server"
cli="$build_dir/redis-8.8.0/src/redis-cli"
"$server" --version
cat "$build_dir/binaries.sha256"
run_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/vdb-cluster-test.XXXXXX")
chmod 700 "$run_dir"
export REDIS_CLUSTER_TEST_PASSWORD="fixture-$(od -An -N12 -tx1 /dev/urandom | tr -d ' \n')"
export REDISCLI_AUTH="$REDIS_CLUSTER_TEST_PASSWORD"
export REDIS_CLUSTER_TEST_PORT=17379
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  rm -rf "$run_dir"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
for offset in 0 1 2; do
  port=$((REDIS_CLUSTER_TEST_PORT+offset))
  mkdir "$run_dir/$offset"
  "$server" --port "$port" --bind 127.0.0.1 --protected-mode yes --daemonize no \
    --dir "$run_dir/$offset" --logfile "$run_dir/$offset/server.log" \
    --requirepass "$REDIS_CLUSTER_TEST_PASSWORD" --masterauth "$REDIS_CLUSTER_TEST_PASSWORD" \
    --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 1000 \
    --cluster-announce-ip 127.0.0.1 --cluster-announce-port "$port" \
    --cluster-announce-bus-port "$((port+10000))" --save '' --appendonly no &
  pids+=("$!")
  for attempt in $(seq 1 30); do
    kill -0 "${pids[$offset]}" || { cat "$run_dir/$offset/server.log"; exit 1; }
    if "$cli" -p "$port" --no-auth-warning PING 2>/dev/null | grep -qx PONG; then break; fi
    if [ "$attempt" = 30 ]; then cat "$run_dir/$offset/server.log"; exit 1; fi
    sleep 1
  done
done
"$cli" --no-auth-warning --cluster create 127.0.0.1:17379 127.0.0.1:17380 127.0.0.1:17381 --cluster-replicas 0 --cluster-yes >/dev/null
for attempt in $(seq 1 30); do
  if "$cli" -p 17379 --no-auth-warning CLUSTER INFO | tr -d '\r' | grep -qx cluster_state:ok; then break; fi
  if [ "$attempt" = 30 ]; then cat "$run_dir/0/server.log"; exit 1; fi
  sleep 1
done
read -r -a cargo_flags <<< "${CLUSTER_CARGO_FLAGS:-}"
# Make exports a system HDF5_DIR by default. Static builds must clear it here,
# after entering the recipe, just like scripts/release/package.sh does.
if [[ "${CLUSTER_CARGO_FLAGS:-}" == *release-static* ]]; then
  unset HDF5_DIR HDF5_VERSION OPENSSL_NO_VENDOR
fi
cargo test --test integration_redis_cluster --release "${cargo_flags[@]}" -- --nocapture --test-threads=1
