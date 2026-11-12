"""Run a set of situations through the full model and check the leading emotion readout.

This is a sanity check that each input reaches the expected circuit and produces the
intended readout; it is not a validation against human data.

  python tests/check_scenarios.py                 # default size for this device
  python tests/check_scenarios.py --neurons 65536 --seeds 1 2 3
"""

import argparse
import pathlib
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

from brain_model import model

SCENARIOS = [
    ("nothing happens", {}, {"Little change"}),
    ("threat cue", {"threat_cue": 1.0}, {"Fear-like"}),
    ("threat cue, full control", {"threat_cue": 0.8, "control": 1.0}, {"Anger-like", "Fear-like"}),
    ("uncertain and stressed", {"uncertainty": 1.0, "stress": 0.8}, {"Anxiety-like"}),
    ("reward cue", {"reward_cue": 1.0}, {"Joy-like"}),
    ("loss or rejection", {"loss_cue": 1.0}, {"Sadness-like"}),
    ("pain", {"pain": 1.0}, {"Distress-like", "Fear-like"}),
    ("warm social contact", {"social_touch": 1.0}, {"Social warmth-like"}),
    ("something new", {"novelty": 1.0}, {"Curiosity-like"}),
    (
        "safe and in control",
        {"control": 0.9, "stress": 0.0, "uncertainty": 0.05, "body_arousal": 0.1},
        {"Calm-like", "Little change"},
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--neurons", type=int, help="network size (default: this device's default)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--only", nargs="+", help="run only scenarios whose name contains one of these words"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print the evidence behind each readout"
    )
    args = parser.parse_args()
    selected = [
        scenario
        for scenario in SCENARIOS
        if not args.only or any(word in scenario[0] for word in args.only)
    ]
    if not selected:
        parser.error("No scenarios matched --only")
    failures = 0
    for name, inputs, expected in selected:
        for seed in args.seeds:
            payload = {"inputs": inputs, "seed": seed}
            if args.neurons:
                payload["neurons"] = args.neurons
            started = time.perf_counter()
            result = model.simulate(payload)
            top = result["emotions"][0]["label"]
            ok = top in expected
            failures += not ok
            ranked = ", ".join(f"{row['label']} {row['score']}%" for row in result["emotions"])
            print(
                f"{'ok  ' if ok else 'FAIL'} {name:26} seed {seed}: {ranked}  ({time.perf_counter() - started:.0f}s)"
            )
            if args.verbose or not ok:
                for key in ("response", "mood"):
                    print(
                        f"       {key:8}",
                        {k: round(v, 2) for k, v in result["evidence"][key].items()},
                    )
                print("       logits  ", result["evidence"]["emotion_logits"])
    print("all scenarios passed" if not failures else f"{failures} scenario run(s) did not match")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
