"""Compare a fresh-process quartet and inspect the independent pause probe."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def keyed_runs(document: dict) -> dict:
    grouped = defaultdict(list)
    for run in document["runs"]:
        key = (run["batch"], run["prompt_tokens"], run["output_tokens"])
        if run["wall_ns"] <= 0 or len(run["tokens"]) != key[0]:
            raise ValueError("Invalid duration or request count")
        if any(len(tokens) != key[2] for tokens in run["tokens"].values()):
            raise ValueError("Invalid output length")
        grouped[key].append(run)
    return grouped


def compare(documents: list[dict]) -> list[dict]:
    if len(documents) != 4 or len({d["pid"] for d in documents}) != 4:
        raise ValueError("Four distinct fresh processes are required")
    reference = documents[0]
    for document in documents[1:]:
        for field in ("versions", "visible_devices", "sources", "resolved_config", "harness_sha256"):
            if document[field] != reference[field]:
                raise ValueError(f"Different experiment contract: {field}")
    arms = [keyed_runs(d) for d in documents]
    if any(set(arm) != set(arms[0]) for arm in arms):
        raise ValueError("Different workload matrices")
    rows = []
    for key in sorted(arms[0]):
        def ordered_tokens(run: dict) -> list[list[int]]:
            return [tokens for request_id, tokens in sorted(
                run["tokens"].items(), key=lambda item: int(item[0].rsplit("-", 1)[1]))]

        expected = ordered_tokens(arms[0][key][0])
        for arm in arms:
            if len(arm[key]) != len(arms[0][key]):
                raise ValueError("Different repeat counts")
            for run in arm[key]:
                if ordered_tokens(run) != expected:
                    raise ValueError("Token identity differs")
        medians = [statistics.median(r["wall_ns"] for r in arm[key]) / 1e6 for arm in arms]
        baseline = statistics.geometric_mean([medians[0], medians[3]])
        instrumented = statistics.geometric_mean(medians[1:3])
        rows.append({
            "batch": key[0], "prompt": key[1], "output": key[2],
            "baseline_ms": baseline, "instrumented_ms": instrumented,
            "speed_ratio": baseline / instrumented,
            "throughput_change_pct": 100 * (baseline / instrumented - 1),
            "latency_change_pct": 100 * (instrumented / baseline - 1),
            "arm_medians_ms": medians, "token_identity": True,
        })
    return rows


def check_pause(trace: dict) -> dict:
    events = trace["traceEvents"]
    pauses = [e for e in events if e.get("name") == "harness_paused_with_pending_request" and e.get("ph") == "X"]
    if len(pauses) != 1:
        raise ValueError("Expected one pending-request pause")
    pause = pauses[0]
    start, end = pause["ts"], pause["ts"] + pause["dur"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels or not any(e["ts"] < start for e in kernels) or not any(e["ts"] >= end for e in kernels):
        raise ValueError("Missing GPU activity before or after pause")
    overlapping = [e for e in kernels if e["ts"] < end and e["ts"] + e["dur"] > start]
    if overlapping:
        raise ValueError("GPU work continued while the harness was paused")
    return {"pause_ms": pause["dur"] / 1000, "kernels_during_pause": 0, "total_kernels": len(kernels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    documents = [json.loads((args.artifacts / f"{args.prefix}-{arm}.json").read_text()) for arm in ("final-A0", "P0", "P1", "final-A1")]
    rows = compare(documents)
    pause = check_pause(json.loads((args.artifacts / f"{args.prefix}-profile.trace.json").read_text()))
    result = {"comparison": "native vs loader audit, real GPU", "evidence": "one fresh quartet", "rows": rows, "pause_probe": pause}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
