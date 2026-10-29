#!/usr/bin/env bash
# One real HTTPS request through an extracted release archive's static OpenSSL (#366).
#
#   scripts/release/tls-probe.sh <extracted archive dir> expect-ok|expect-fail [url]
#
# The archive links a vendored OpenSSL, which finds the host's CA store through
# openssl-probe rather than at a compiled-in path. This drives the shipped
# `vector-db-benchmark` binary through a certificate-VALIDATING client (the
# Chroma engine's reqwest/native-tls) against a public HTTPS endpoint that is
# not a Chroma server:
#
#  * expect-ok:   the TLS handshake and certificate check succeed, and the
#                 failure is the server's HTTP answer ("create collection failed").
#  * expect-fail: on a host with no CA bundle, the request must NOT get that far
#                 ("create collection request failed") — the negative control
#                 that shows the probe can tell the two apart. The binary's
#                 error does not carry the cause, so the probe first proves, in
#                 the same container, that the name resolves and port 443
#                 accepts a TCP connection: what is left to fail is TLS.
#
# Side effect, intentional: the Chroma client sends an unauthenticated DELETE
# and then a POST for a collection path to the URL (api.github.com by
# default). GitHub answers 404 and nothing is created; the probe needs a
# request that has to pass certificate validation to get any HTTP answer.
#
# Runs inside a bare ubuntu:22.04 container in the release workflow.
set -euo pipefail

dir="${1:-}" mode="${2:-}" url="${3:-https://api.github.com}"
[[ -x "$dir/vector-db-benchmark" && ( "$mode" == expect-ok || "$mode" == expect-fail ) ]] || {
  echo "usage: tls-probe.sh <extracted dir> expect-ok|expect-fail [url]" >&2; exit 2; }
bin="$(cd "$dir" && pwd)/vector-db-benchmark"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/datasets"
# A dataset that is never downloaded (engine setup fails first) and an engine
# config that is not shipped: nothing here depends on the archive's configs.
printf '%s' '[{"name":"tls-probe","vector_size":4,"distance":"cosine","type":"h5","path":"tls-probe/p.hdf5","link":"https://127.0.0.1:9/never","description":"tls probe"}]' >"$work/datasets/datasets.json"
printf '%s' '[{"name":"tls-probe","engine":"chroma","connection_params":{},"collection_params":{},"search_params":[{"parallel":1,"search_params":{}}],"upload_params":{"parallel":1}}]' >"$work/engines.json"

if [[ "$mode" == expect-fail ]]; then
  # DNS + TCP to the same host and port, without TLS (bash's /dev/tcp: the
  # bare image has no curl), so a failure below is not "no network".
  hostport="${url#https://}"; hostport="${hostport%%/*}"
  host="${hostport%%:*}"; port=443; [[ "$hostport" == *:* ]] && port="${hostport##*:}"
  if ! timeout 20 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ "$host" "$port" 2>/dev/null; then
    echo "tls-probe: FAIL: cannot reach $host:$port over TCP here, so a failed request would prove nothing about TLS" >&2
    exit 1
  fi
  echo "probe: $host:$port reachable over TCP"
fi

set +e
out="$(cd "$work" && timeout 120 "$bin" --engines-file engines.json --engines tls-probe \
  --datasets tls-probe --host "$url" 2>&1)"
set -e
err="$(grep -m1 '^Error' <<<"$out" || true)"
echo "probe: ${err:0:300}"

case "$mode" in
  expect-ok)
    [[ "$err" == *"create collection failed"* ]] || {
      echo "tls-probe: FAIL: HTTPS to $url did not complete with a verified certificate" >&2
      echo "$out" | tail -n 20 >&2; exit 1; } ;;
  expect-fail)
    [[ "$err" == *"create collection request failed"* ]] || {
      echo "tls-probe: FAIL: with no CA bundle the request should fail before any HTTP answer" >&2
      echo "$out" | tail -n 20 >&2; exit 1; } ;;
esac
echo "tls-probe: $mode OK"
