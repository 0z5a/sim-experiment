from dataclasses import dataclass
import statistics

from simulator.runtime import Shape


@dataclass(frozen=True)
class Workload:
    name: str
    requests: int
    prompt: int
    output: int
    second_wave_ns: int = 0

    def arrivals(self) -> list[int]:
        first = self.requests if self.second_wave_ns == 0 else self.requests // 2
        return [0] * first + [self.second_wave_ns] * (self.requests - first)


def calibration_workloads() -> list[Workload]:
    return [Workload(f"cal-b{batch}-p{prompt}-r{repeat}", batch, prompt, 96)
            for repeat in range(2) for batch in (1, 8, 16, 32) for prompt in (32, 64, 128)]


def validation_workloads() -> list[Workload]:
    return [Workload(f"burst-{requests}", requests, 64, 64) for requests in (128, 256, 512)] + [
        Workload("two-waves-192", 192, 64, 64, 500_000_000)]


def scheduled_shape(scheduler, output) -> Shape:
    items = output.num_scheduled_tokens
    if not items:
        raise ValueError("A timed step must schedule tokens")
    prefill = []
    contexts = []
    for rid, count in items.items():
        request = scheduler.requests[rid]
        previous = request.num_computed_tokens - count
        prefill.append(previous < request.num_prompt_tokens)
        contexts.append(request.num_computed_tokens)
    if any(prefill) and not all(prefill):
        phase = "mixed"
    else:
        phase = "prefill" if all(prefill) else "decode"
    return Shape(phase, len(items), statistics.mean(contexts))


def metrics(workload: Workload, stamps: dict[str, list[int]], finish_ns: int) -> dict:
    arrivals = workload.arrivals()
    if len(stamps) != workload.requests or any(len(v) != workload.output for v in stamps.values()):
        raise ValueError("Incomplete workload")
    ttft = [stamps[str(i)][0] - arrival for i, arrival in enumerate(arrivals)]
    latency = [stamps[str(i)][-1] - arrival for i, arrival in enumerate(arrivals)]
    itl = [b - a for times in stamps.values() for a, b in zip(times, times[1:])]
    if min(ttft) < 0 or min(itl) <= 0:
        raise ValueError("Invalid token visibility timeline")
    return {
        "finish_ns": finish_ns, "tokens_per_second": workload.requests * workload.output * 1e9 / finish_ns,
        "ttft_median_ms": statistics.median(ttft) / 1e6,
        "request_median_ms": statistics.median(latency) / 1e6,
        "itl_median_ms": statistics.median(itl) / 1e6,
        "completed_requests": len(stamps), "generated_tokens": sum(map(len, stamps.values())),
    }
