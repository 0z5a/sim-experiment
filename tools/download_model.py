"""Download one pinned dense model into this experiment's private directory."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = "Qwen/Qwen2.5-0.5B"
    info = HfApi(token=False).model_info(repo, timeout=15)
    destination = args.root / "models" / "Qwen2.5-0.5B"
    snapshot_download(
        repo, revision=info.sha, local_dir=destination, token=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        max_workers=4,
    )
    manifest = {"repo": repo, "revision": info.sha, "path": str(destination)}
    (args.root / "artifacts" / "model.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
