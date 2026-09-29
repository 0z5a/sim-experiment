"""Keep simulator speed separate from modeled-system throughput and prediction error."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import re


def external_ids(steps: list[dict], count: int) -> list[list[str]]:
    mapping = {}
    for step in steps:
        for internal in step["ids"]:
            match = re.fullmatch(r"(\d+)-[0-9a-f]{8}", internal)
            if match is None:
                raise ValueError(f"Unexpected engine request ID: {internal}")
            mapping[internal] = match[1]
    if len(mapping) != count or set(mapping.values()) != {str(i) for i in range(count)}:
        raise ValueError("Engine request IDs are not a one-to-one workload mapping")
    return [[mapping[rid] for rid in step["ids"]] for step in steps]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    real = [json.loads((args.root / f"real-rank{rank}.json").read_text()) for rank in range(2)]
    simulated = json.loads((args.root / "simulated.json").read_text())
    assert not simulated["cuda_initialized"]
    grouped = defaultdict(list)
    for left, right, sim in zip(real[0]["validation"], real[1]["validation"], simulated["results"], strict=True):
        assert left["workload"] == right["workload"] == sim["workload"]
        count = left["workload"]["requests"]
        left_ids = external_ids(left["steps"], count)
        assert left_ids == external_ids(right["steps"], count)
        membership_matches = left_ids == [s["ids"] for s in sim["steps"]]
        for metric in ("completed_requests", "generated_tokens"):
            assert left["metrics"][metric] == right["metrics"][metric] == sim["metrics"][metric]
        duration = max(left["metrics"]["finish_ns"], right["metrics"]["finish_ns"])
        estimated = sim["metrics"]["finish_ns"]
        grouped[left["workload"]["name"]].append({
            "real_ns": duration, "sim_wall_ns": sim["simulator_wall_ns"],
            "estimated_ns": estimated, "speedup": duration / sim["simulator_wall_ns"],
            "duration_error_pct": 100 * (estimated / duration - 1),
            "throughput_error_pct": 100 * (duration / estimated - 1),
            "ttft_error_pct": 100 * (sim["metrics"]["ttft_median_ms"] / left["metrics"]["ttft_median_ms"] - 1),
            "itl_error_pct": 100 * (sim["metrics"]["itl_median_ms"] / left["metrics"]["itl_median_ms"] - 1),
            "batch_membership_matches": membership_matches,
        })
    rows = []
    for name, repeats in grouped.items():
        row = {"workload": name}
        for key in repeats[0]:
            row[key] = all(x[key] for x in repeats) if key == "batch_membership_matches" else statistics.median(x[key] for x in repeats)
        row["repeats"] = repeats
        row["accuracy_goal_met"] = all(abs(row[key]) <= 10 for key in (
            "duration_error_pct", "throughput_error_pct", "ttft_error_pct", "itl_error_pct"))
        rows.append(row)
    slow = simulated["feedback_reference"]
    fast = simulated["feedback_faster_device"]
    feedback = {"normal_max_waiting": max(s["waiting"] for s in slow["steps"]),
                "faster_max_waiting": max(s["waiting"] for s in fast["steps"]),
                "normal_finish_ns": slow["metrics"]["finish_ns"],
                "faster_finish_ns": fast["metrics"]["finish_ns"]}
    if feedback["normal_max_waiting"] == feedback["faster_max_waiting"]:
        raise ValueError("Feedback fixture did not change the original scheduler queue")
    real_total = sum(max(a["metrics"]["finish_ns"], b["metrics"]["finish_ns"]) for a, b in zip(real[0]["validation"], real[1]["validation"]))
    setup = max(r["initialization_and_calibration_ns"] for r in real)
    report = {"rows": rows, "feedback": feedback,
              "all_accuracy_goals_met": all(row["accuracy_goal_met"] for row in rows),
              "calibration_and_warmup_ns": setup, "real_validation_workload_ns": real_total,
              "simulation_process_wall_ns": simulated["external_wall_ns"],
              "simulation_setup_ns": simulated["setup_ns"],
              "campaign_speedup_excluding_calibration": real_total / simulated["external_wall_ns"],
              "campaign_speedup_including_calibration": real_total / (setup + simulated["external_wall_ns"])}
    (args.root / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Cross-rank high-load E2E results", "", "CPU scheduler-boundary adapter; this is not U0 CUDA/NCCL ABI simulation. Qwen2.5-0.5B, TP2, GPUs 4/7, max 32 active requests, P=64/O=64. Three workload repeats in one resident real-engine session; two independent scheduler instances in the CPU simulation.", "", "| Workload | Real workload s | Simulator s | Workload speedup | Duration error | Throughput error | TTFT median error | ITL median error |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['workload']} | {r['real_ns']/1e9:.3f} | {r['sim_wall_ns']/1e9:.3f} | {r['speedup']:.1f}× | {r['duration_error_pct']:+.2f}% | {r['throughput_error_pct']:+.2f}% | {r['ttft_error_pct']:+.2f}% | {r['itl_error_pct']:+.2f}% |")
    lines += ["", "Initial accuracy target (absolute median errors ≤10% for all four metrics in every workload): " + ("PASS" if report["all_accuracy_goals_met"] else "NOT MET") + "."]
    lines += ["", f"One-time real initialization, warmup and calibration: {setup/1e9:.2f} s. Complete CPU process measurement (imports, fit, 12 validation workloads and two feedback scenarios): {simulated['external_wall_ns']/1e9:.2f} s. Real validation workload total: {real_total/1e9:.2f} s.", "", f"Campaign speedup with an existing calibration: {report['campaign_speedup_excluding_calibration']:.2f}×. Charging this campaign the entire one-time calibration cost: {report['campaign_speedup_including_calibration']:.2f}×. Downloads are excluded. Workload rows exclude process startup and calibration; campaign figures include simulator startup.", "", f"Feedback check: reducing modeled step durations to 20% changes the original scheduler's maximum waiting queue from {feedback['normal_max_waiting']} to {feedback['faster_max_waiting']}. This is a causal sensitivity fixture, not a measured device optimization.", "", "The cost model is fit only from calibration files (batches 1/8/16/32, prompts 32/64/128, O=96). Held-out requests/arrivals/output lengths are not used for fitting. Mixed prefill/decode steps and features outside the calibration domain are rejected. Rank costs include communication and host overhead; NCCL is not added again.", "", "Fixed token counts and request lifetimes are validated. Synthetic token values do not reproduce model text. No CUDA context is initialized by the simulation. GPU 4/7 were shared during calibration/validation; see the saved GPU/process snapshots and failed-run notes. These results are conditional on that measured stack and contention, and are not isolated-hardware estimates or statistical confidence intervals.", ""]
    (args.root / "TEST_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
