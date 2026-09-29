import copy
import unittest

from tools.analyze import check_pause, compare


def quartet() -> list[dict]:
    documents = []
    for pid in range(4):
        documents.append({
            "pid": pid, "versions": {"vllm": "test"}, "visible_devices": "test",
            "sources": {}, "resolved_config": "test", "harness_sha256": "test",
            "runs": [{"batch": 2, "prompt_tokens": 2, "output_tokens": 2,
                      "wall_ns": 100 if pid in (0, 3) else 50,
                      "tokens": {"r0-0": [1, 2], "r0-1": [3, 4]}}],
        })
    return documents


class AnalysisTest(unittest.TestCase):
    def test_ratio_direction(self) -> None:
        row = compare(quartet())[0]
        self.assertEqual(row["speed_ratio"], 2)
        self.assertEqual(row["throughput_change_pct"], 100)
        self.assertEqual(row["latency_change_pct"], -50)

    def test_request_swap_rejected(self) -> None:
        docs = quartet()
        docs[1]["runs"][0]["tokens"] = {"r0-0": [3, 4], "r0-1": [1, 2]}
        with self.assertRaisesRegex(ValueError, "Token identity"):
            compare(docs)

    def test_short_output_rejected(self) -> None:
        docs = quartet()
        docs[2]["runs"][0]["tokens"]["r0-0"].pop()
        with self.assertRaisesRegex(ValueError, "output length"):
            compare(docs)

    def test_same_process_rejected(self) -> None:
        docs = quartet()
        docs[2]["pid"] = docs[0]["pid"]
        with self.assertRaisesRegex(ValueError, "fresh processes"):
            compare(docs)

    def test_source_change_rejected(self) -> None:
        docs = quartet()
        docs[2]["harness_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "contract"):
            compare(docs)

    def test_pause_requires_no_overlapping_gpu_work(self) -> None:
        trace = {"traceEvents": [
            {"name": "harness_paused_with_pending_request", "ph": "X", "ts": 10, "dur": 10},
            {"cat": "kernel", "ts": 0, "dur": 5},
            {"cat": "kernel", "ts": 21, "dur": 5},
        ]}
        self.assertEqual(check_pause(trace)["kernels_during_pause"], 0)
        overlap = copy.deepcopy(trace)
        overlap["traceEvents"].append({"cat": "kernel", "ts": 8, "dur": 5})
        with self.assertRaisesRegex(ValueError, "GPU work continued"):
            check_pause(overlap)

    def test_empty_gpu_trace_rejected(self) -> None:
        trace = {"traceEvents": [{"name": "harness_paused_with_pending_request", "ph": "X", "ts": 10, "dur": 10}]}
        with self.assertRaisesRegex(ValueError, "Missing GPU activity"):
            check_pause(trace)


if __name__ == "__main__":
    unittest.main()
