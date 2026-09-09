"""Qualify real native conservative-flux learning and explicitly withheld supports."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--artifact-directory", type=Path)
    args = parser.parse_args()
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "examples"
            / "geophysical_flux_closure.py"
        )
    )
    report = example["run_example"](
        steps=args.steps, artifact_directory=args.artifact_directory
    )
    if report["completed_steps"] != [args.steps, args.steps]:
        raise AssertionError(
            "Both physical flux operators must complete real native training."
        )
    if not np.all(np.asarray(report["final_loss"]) < np.asarray(report["initial_loss"])):
        raise AssertionError(
            "Native flux training failed to reduce its supervised objectives."
        )
    for field, artifact in report["native_artifact_roundtrip"].items():
        if (
            not artifact["exact_task_weights_normalization_identity"]
            or artifact["same_eager_max_error"] != 0.0
        ):
            raise AssertionError(
                f"Native serialization changed same-mode {field} behavior: {artifact}"
            )
        if (
            not np.isfinite(artifact["max_tolerance_ratio"])
            or artifact["max_tolerance_ratio"] > 1.0
        ):
            raise AssertionError(
                f"Compiled/eager {field} difference exceeds native artifact tolerance: {artifact}"
            )
    if report["native_column_restart_max_error"] != 0.0:
        raise AssertionError(
            "Native physical column restart changed continued trained-flux evolution."
        )
    training_cases = set(report["training_case_ids"])
    evaluation_cases = set()
    for name, evaluation in report["evaluation"].items():
        cases = set(evaluation["case_ids"])
        if cases & (training_cases | evaluation_cases):
            raise AssertionError(
                "Independent transfer evaluation leaked simulation cases."
            )
        evaluation_cases.update(cases)
        trained, untrained = evaluation["trained"], evaluation["untrained"]
        for result in (trained, untrained):
            metrics = (
                result["relative_rmse_vs_no_closure"] + result["inventory_residual_max"]
            )
            if not np.all(np.isfinite(metrics)) or result["correction_rate"] != 0.0:
                raise AssertionError(
                    "Flux evaluation is nonfinite or applies hidden corrections."
                )
            if (
                result["inventory_residual_max"][0] > 1e-10
                or result["inventory_residual_max"][1] > 1e-5
            ):
                raise AssertionError("Paired closed interfaces leaked inventory.")
        if trained["rejection_rate"] != 0.0:
            raise AssertionError(f"Trained closure rejected physical {name} cases.")
        if not np.all(np.asarray(trained["relative_rmse_vs_no_closure"]) < 1.0):
            raise AssertionError(
                f"Trained closure does not beat no closure on {name}: {trained}"
            )
        if not np.all(
            np.asarray(trained["relative_rmse_vs_no_closure"])
            < np.asarray(untrained["relative_rmse_vs_no_closure"])
        ):
            raise AssertionError(
                f"Training does not beat its untrained baseline on {name}."
            )
    resolution = report["evaluation"]["withheld_resolution"]["trained"]["resolution_m"]
    interval = report["evaluation"]["withheld_interval"]["trained"]["closure_interval_s"]
    forcing = report["evaluation"]["withheld_forcing"]["trained"]["forcing_id"]
    if (
        resolution in report["training_resolution_m"]
        or interval in report["training_closure_interval_s"]
        or forcing in report["training_forcing_ids"]
    ):
        raise AssertionError("Claimed transfer supports were not actually withheld.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
