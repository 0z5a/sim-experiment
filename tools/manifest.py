"""Hash the installed stack and experiment-owned model assets without changing them."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for model in sorted((args.root / "models").iterdir()):
        models[model.name] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(model.iterdir()) if p.is_file()
        }
    packages = {}
    for name in ("vllm", "torch", "triton", "transformers", "flashinfer-python"):
        dist = importlib.metadata.distribution(name)
        checked = 0
        mismatches = []
        sources = {}
        for entry in dist.files:
            filename = str(entry)
            if name == "vllm" and filename.endswith((".py", ".so")) and entry.hash:
                path = Path(dist.locate_file(entry))
                digest = sha256(path)
                actual = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode()
                checked += 1
                if entry.hash.mode != "sha256" or actual != entry.hash.value:
                    mismatches.append(filename)
                sources[filename] = digest
        packages[name] = {"version": dist.version, "record_checked_files": checked,
                          "record_mismatches": mismatches, "sources": sources}
    result = {
        "python": sys.executable, "models": models, "packages": packages,
        "gpu_inventory": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv"], text=True),
        "gpu_topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "toolkit": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
        "sources": {str(p.relative_to(args.root)): sha256(p) for sub in ("tools", "tests", "build") for p in sorted((args.root / sub).glob("*")) if p.is_file()},
    }
    (args.root / "artifacts" / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"models": list(models), "vllm_record_mismatches": packages["vllm"]["record_mismatches"]}))


if __name__ == "__main__":
    main()
