import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
checkout = Path(__file__).resolve().parents[1]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(checkout), OMP_NUM_THREADS="1",
           VLLM_ENABLE_V1_MULTIPROCESSING="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
started = time.perf_counter_ns()
subprocess.run([sys.executable, "tools/simulate_serving.py", "--root", str(args.root)],
               cwd=checkout, env=env, check=True)
elapsed = time.perf_counter_ns() - started
path = args.root / "simulated.json"
result = json.loads(path.read_text())
result["external_wall_ns"] = elapsed
path.write_text(json.dumps(result))
print("Complete simulator child process wall seconds:", elapsed / 1e9)
