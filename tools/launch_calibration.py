"""Use external TP ranks; never signal or terminate processes."""

import os
from pathlib import Path
import socket
import subprocess


root = Path(__file__).resolve().parents[1]
output = root / "artifacts" / "crossrank-v2"
output.mkdir(exist_ok=True)
inventory = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv",
], text=True)
(output / "gpu-before.csv").write_text(inventory)
processes = subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv",
], text=True)
(output / "processes-before.csv").write_text(processes)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
children = []
for rank in range(2):
    env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
               CUDA_VISIBLE_DEVICES="GPU-739fcff8-0a90-fd63-8d91-65847ad665d2,GPU-c1e2d922-6fb4-92a2-c9e3-9f4ef9d9c079",
               VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", PYTHONPATH=str(root),
               VLLM_CACHE_ROOT=str(root / "cache/vllm"), TRITON_CACHE_DIR=str(root / "cache/triton"))
    with (output / f"rank{rank}.log").open("w") as log:
        children.append(subprocess.Popen([
            "/home/gongji/0z5a/bin/python", "-u", "tools/calibrate_serving.py",
            "--model", str(root / "models/Qwen2.5-0.5B"), "--root", str(output),
        ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [child.wait() for child in children]
print("Rank exit codes:", codes, flush=True)
raise SystemExit(0 if codes == [0, 0] else 1)
