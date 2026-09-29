from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class Shape:
    phase: str
    batch: int
    context: float


class RankClock:
    """Synchronous TP step rendezvous; calibrated costs already include NCCL."""

    def __init__(self, ranks: int):
        self.tails = [0] * ranks
        self.now = 0

    def complete(self, release: int, costs: list[int]) -> int:
        if len(costs) != len(self.tails) or any(c <= 0 for c in costs):
            raise ValueError("Every rank must submit one positive step cost")
        if release < self.now:
            raise ValueError("Host release precedes the committed frontier")
        self.tails = [max(release, tail) + cost for tail, cost in zip(self.tails, costs)]
        self.now = max(self.tails)
        return self.now


def interpolate(points: dict[float, float], x: float) -> float:
    keys = sorted(points)
    if x < keys[0] or x > keys[-1]:
        raise ValueError(f"Outside calibration domain: {x} not in [{keys[0]}, {keys[-1]}]")
    i = bisect_left(keys, x)
    if keys[i] == x:
        return points[x]
    left, right = keys[i - 1], keys[i]
    weight = (x - left) / (right - left)
    return points[left] * (1 - weight) + points[right] * weight


class StepModel:
    """Bilinear interpolation of median rank costs, with explicit domain rejection."""

    def __init__(self, samples: list[dict]):
        grouped = defaultdict(list)
        for sample in samples:
            shape = sample["shape"]
            # Decode contexts are binned only during fitting; lookup interpolates.
            context = shape["context"] if shape["phase"] == "prefill" else round(shape["context"] / 16) * 16
            grouped[(shape["phase"], shape["batch"], context)].append(sample["duration_ns"])
        self.table = defaultdict(dict)
        for (phase, batch, context), durations in grouped.items():
            self.table[phase].setdefault(batch, {})[context] = statistics.median(durations)

    def predict(self, shape: Shape) -> int:
        if shape.phase not in self.table:
            raise ValueError(f"Uncalibrated step phase: {shape.phase}")
        batches = self.table[shape.phase]
        keys = sorted(batches)
        i = bisect_left(keys, shape.batch)
        if i == len(keys) or shape.batch < keys[0]:
            raise ValueError("Batch outside calibration domain")
        neighbors = [keys[i]] if keys[i] == shape.batch else keys[i - 1:i + 1]
        costs = {batch: interpolate(batches[batch], shape.context) for batch in neighbors}
        return round(interpolate(costs, shape.batch))
