"""Find how many cells and synapses this computer can simulate.

Builds the app's real network (brain_model/simulator.py) at each requested size,
runs it briefly and reports synapses, build time, time per 1 ms step, the estimated
time for a full 6 s run, peak memory and resting firing. Uses the NVIDIA GPU when
CUDA is available, otherwise the CPU.

Examples (Windows, from this folder):
  .venv\\Scripts\\python.exe benchmark.py
  .venv\\Scripts\\python.exe benchmark.py 2097152 4194304 8388608 --steps 1000
  .venv\\Scripts\\python.exe benchmark.py 65536 --cpu

Sizes run in order and the benchmark stops at the first one that runs out of memory.
"""

import argparse
import os
import sys
import time
import warnings

DEFAULT_SIZES = [16_384, 65_536, 262_144, 1_048_576, 2_097_152, 4_194_304]


def peak_memory_gib(device):
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if sys.platform != "win32":
        import resource

        # macOS reports bytes; Linux reports KiB. Windows has no resource module.
        divisor = 2**30 if sys.platform == "darwin" else 2**20
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sizes", nargs="*", type=int, default=DEFAULT_SIZES, help="cell counts to try"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="1 ms steps per run (6000 = the app's full 6 s)"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="run on the CPU even if CUDA is available"
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    if any(size <= 0 for size in args.sizes):
        parser.error("cell counts must be positive")
    if args.cpu:
        os.environ["HUMAN_EMOTION_DEVICE"] = "cpu"
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
    warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

    import torch

    from brain_model import simulator
    from brain_model.model import DEFAULT_INPUTS, NETWORK

    device = simulator.pick_device()
    step_ms = NETWORK["runtime"]["time_step_ms"]
    full_steps = NETWORK["runtime"]["duration_seconds"] * 1000 / step_ms
    print(f"Device: {simulator.device_info(device)['name']} · {args.steps} steps per run\n")
    print(
        f"{'cells':>11} {'synapses':>13} {'build s':>8} {'ms/step':>8} {'full run':>9} {'peak GiB':>9} {'cell Hz':>8}"
    )
    for size in args.sizes:
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            topology = simulator.build_topology(NETWORK, size, device)
            started = time.perf_counter()
            result = simulator.run(
                NETWORK,
                topology,
                DEFAULT_INPUTS,
                seed=1,
                duration_s=args.steps * step_ms / 1000,
                record=False,
            )
            run_seconds = time.perf_counter() - started
        except (RuntimeError, MemoryError, ValueError) as exc:
            print(f"{size:>11,}  stopped: {str(exc).splitlines()[0][:90]}")
            return 1
        ms_per_step = 1000 * run_seconds / args.steps
        rate = result["spike_count"] / size / result["duration_seconds"]
        peak = peak_memory_gib(device)
        memory = f"{peak:.2f}" if peak is not None else "n/a"
        print(
            f"{size:>11,} {topology['synapse_count']:>13,} {topology['build_seconds']:>8.1f} {ms_per_step:>8.1f} "
            f"{ms_per_step * full_steps / 1000:>7.0f} s {memory:>9} {rate:>8.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
