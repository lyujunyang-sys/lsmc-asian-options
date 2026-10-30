"""Run one or more experiment configs and write results/<name>/*.csv and summary.json.

    python scripts/run_experiment.py configs/appendix_baseline.json
    python scripts/run_experiment.py configs/*.json
"""

import sys

import pandas as pd

from lsmc_asian.experiments import load_config, run


def main(paths):
    if not paths:
        sys.exit(__doc__)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        for path in paths:
            cfg = load_config(path)
            print(f"== {cfg['name']}: {cfg['description']}")
            for name, table in run(cfg).items():
                print(f"-- {name}")
                print(table.drop(columns=[c for c in ("runtime_s",) if c in table]).to_string(index=False))
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
