#!/usr/bin/env bash
# Runs every runtime on the identical corpus + seed-42 init (bench/shared) and prints best-of-N.
# Needs: cargo, mojo, gforth, dyalogscript.   Usage: bench/run.sh [N=3]
set -e
N=${1:-3}; ROOT=$(cd "$(dirname "$0")/.." && pwd)
(cd "$ROOT/src/rust" && cargo build --release -q)
(cd "$ROOT/mojo" && mojo build -O3 morphology.mojo -o /tmp/morph_mojo 2>/dev/null)
best() { for _ in $(seq "$N"); do "$@" 2>&1 | tr "\r" "\n" | tail -1; done | sort -t= -k5 -n | head -1; }
best_by_time() { for _ in $(seq "$N"); do "$@" 2>&1 | tr "\r" "\n" | tail -1; done | awk '{for(i=1;i<=NF;i++) if($i ~ /^time=/){t=$i; sub("time=","",t); sub("s","",t); print t, $0}}' | sort -n | head -1 | cut -d' ' -f2-; }
cd "$ROOT/src/rust"; best_by_time ./target/release/export /tmp   # writes to /tmp, runs the original engine
cp /tmp/corpus.txt /tmp/init.txt "$ROOT/bench/shared/" 2>/dev/null || true
best_by_time ./target/release/lean ../../bench/shared mean   # mean-pool: the model every other runtime trains (lean defaults to attention pooling)
cd "$ROOT/mojo";        best_by_time /tmp/morph_mojo
cd "$ROOT/dyalog-apl";  best_by_time dyalogscript morphology.apls
cd "$ROOT/forth";       best_by_time gforth morphology.fs -e bye
cd "$ROOT/bqn";          best_by_time ${BQN:-BQN} morphology.bqn   # CBQN, built from github.com/dzaima/CBQN
