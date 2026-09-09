"""Small real native SFNO training/forecast and column-closure qualification."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--artifact-directory", type=Path)
    args = parser.parse_args()
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "examples"
            / "geophysical_operator_forecast.py"
        )
    )
    report = example["run_example"](
        steps=args.steps, artifact_directory=args.artifact_directory
    )
    if (
        report["sfno_completed_steps"] != args.steps
        or report["column_completed_steps"] != args.steps
    ):
        raise AssertionError("Native training did not complete the requested updates.")
    if not report["sfno_final_loss"] < report["sfno_initial_loss"]:
        raise AssertionError(
            "Native SFNO training did not reduce the supervised objective."
        )
    if not report["column_final_loss"] < report["column_initial_loss"]:
        raise AssertionError(
            "Native column operator training did not reduce the supervised objective."
        )
    if report["restart_max_error"] > 1e-6:
        raise AssertionError(
            "Native recurrent continuation differs from uninterrupted forecasting."
        )
    if not report["column_admitted"] or report["column_budget_max_error"] > 0.05:
        raise AssertionError(
            "Column deployment violates physical admission or its mass-weighted budget."
        )
    if not np.all(np.isfinite(report["rmse_kelvin"])) or not np.all(
        np.isfinite(report["crps_kelvin"])
    ):
        raise AssertionError("Physical forecast verification is not finite.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
