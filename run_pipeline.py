"""Run the full analysis pipeline end to end.

Stages run in order and each writes its outputs to data/processed and outputs/.
The raw BTS zip files must already be in place (see the README). The causal
forest stage is the slow one, taking roughly half an hour on a laptop.

Usage:
    python run_pipeline.py            # run every stage
    python run_pipeline.py --from forest   # resume from a given stage
"""
import argparse

from src.data import parse_db1b, parse_t100, build_panel
from src import matching, event_study, causal_forest, robustness, diagnostics, descriptives

STAGES = [
    ("parse_db1b", parse_db1b.parse_db1b),
    ("parse_t100", parse_t100.parse_t100),
    ("panel", build_panel.build_panel),
    ("matching", matching.run_matching),
    ("event_study", event_study.run_event_study),
    ("forest", causal_forest.run_causal_forest),
    ("robustness", robustness.run_robustness),
    ("diagnostics", diagnostics.run_diagnostics),
    ("descriptives", descriptives.run_descriptives),
]


def main() -> None:
    names = [name for name, _ in STAGES]
    parser = argparse.ArgumentParser(description="Run the LCC entry analysis pipeline.")
    parser.add_argument("--from", dest="start", choices=names, default=names[0],
                        help="stage to start from (default: the first)")
    args = parser.parse_args()

    start = names.index(args.start)
    for name, func in STAGES[start:]:
        print(f"\n{'=' * 10} {name} {'=' * 10}")
        func()


if __name__ == "__main__":
    main()
