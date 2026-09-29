"""Launch two deterministic external ranks and wait for their natural exit."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--loader", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    children = []
    for rank in range(2):
        prefix = root / "artifacts" / f"{args.prefix}-rank{rank}"
        env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), MODEL_DIR=args.model,
                   SIM_GPUS="GPU-739fcff8-0a90-fd63-8d91-65847ad665d2,GPU-c1e2d922-6fb4-92a2-c9e3-9f4ef9d9c079")
        if args.loader:
            env.update(SIM_ENABLE_LOADER="1", SIM_LOADER_TRACE=str(prefix.with_suffix(".loader.jsonl")))
        with prefix.with_suffix(".log").open("w") as log:
            children.append(subprocess.Popen([
                "bash", str(root / "tools/run_baseline.sh"), "--tp", "2", "--runs", "3",
                "--out", str(prefix.with_suffix(".json")),
            ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [child.wait() for child in children]
    print(json.dumps({"exit_codes": codes}), flush=True)
    if codes != [0, 0]:
        raise SystemExit(1)
    results = [json.loads((root / "artifacts" / f"{args.prefix}-rank{rank}.json").read_text()) for rank in range(2)]
    for left, right in zip(results[0]["runs"], results[1]["runs"], strict=True):
        assert left["tokens"] == right["tokens"], "TP ranks disagree"
    print("TP2 rank outputs agree", flush=True)


if __name__ == "__main__":
    main()
