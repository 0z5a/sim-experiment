#!/usr/bin/env bash
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
prefix=$1
export MODEL_DIR=$2
bash tools/run_baseline.sh --runs 3 --out "artifacts/$prefix-final-A0.json" > "artifacts/$prefix-final-A0.log" 2>&1
for arm in P0 P1; do
    SIM_ENABLE_LOADER=1 SIM_LOADER_TRACE="$PWD/artifacts/$prefix-$arm-loader.jsonl" \
        bash tools/run_baseline.sh --runs 3 --out "artifacts/$prefix-$arm.json" > "artifacts/$prefix-$arm.log" 2>&1
done
bash tools/run_baseline.sh --runs 3 --out "artifacts/$prefix-final-A1.json" > "artifacts/$prefix-final-A1.log" 2>&1
bash tools/run_baseline.sh --runs 1 --profile --out "artifacts/$prefix-profile.json" > "artifacts/$prefix-profile.log" 2>&1
/home/gongji/0z5a/bin/python tools/analyze.py --artifacts artifacts --prefix "$prefix" --out "artifacts/$prefix-summary.json"
