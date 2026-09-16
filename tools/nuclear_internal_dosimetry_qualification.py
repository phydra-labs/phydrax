#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic numerical qualification for research-only internal dosimetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.nuclear import dosimetry


def qualify() -> dict[str, object]:
    """Exercise time integration, regional arithmetic, and spatial convolution."""

    axis = phx.measurement.SampleTimeAxis(
        "qualification-time", np.asarray((0.0, 1.0, 3.0)), phx.units.SECOND
    )
    integration = dosimetry.TimeActivityIntegrationPlan(
        axis, 0.0, 3.0, phx.units.SECOND
    ).prepare(0)
    activity = jnp.asarray((4.0, 2.0, 1.0))
    integral = integration.evaluate(activity, jnp.ones_like(activity, dtype=bool))
    integrated_value = float(integral.values)

    regional = dosimetry.PreparedRegionalSValuePlan(
        jnp.asarray(((2.0,), (0.5,))),
        jnp.asarray(((True,), (True,))),
        1,
        2,
        "analytic-regional-s-value",
    ).evaluate(jnp.asarray((integrated_value,)), jnp.asarray((True,)))
    regional_values = np.asarray(regional.dose_gy)

    concentration = np.zeros((3, 3, 3), dtype=float)
    concentration[1, 1, 1] = integrated_value
    spatial = dosimetry.PreparedSpatialSValueConvolution(
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1), dtype=bool),
        jnp.asarray(1.0),
        (3, 3, 3),
        "analytic-spatial-s-value",
    ).evaluate(jnp.asarray(concentration), jnp.ones((3, 3, 3), dtype=bool))
    spatial_values = np.asarray(spatial.dose_gy)

    integration_error = abs(integrated_value - 6.0)
    regional_error = float(np.max(np.abs(regional_values - np.asarray((12.0, 3.0)))))
    expected_spatial = np.zeros((3, 3, 3), dtype=float)
    expected_spatial[1, 1, 1] = 6.0
    spatial_error = float(np.max(np.abs(spatial_values - expected_spatial)))
    successful = bool(
        integration_error < 1.0e-12
        and regional_error < 1.0e-12
        and spatial_error < 1.0e-12
        and np.all(np.asarray(integral.valid))
        and np.all(np.asarray(regional.valid))
        and np.all(np.asarray(spatial.valid))
    )
    return {
        "successful": successful,
        "scope": "analytic-numerical-research-only",
        "scientifically_qualified": False,
        "clinical_use_permitted": False,
        "time_integral_bq_s": integrated_value,
        "regional_dose_gy": regional_values.tolist(),
        "spatial_nonzero_dose_gy": float(spatial_values[1, 1, 1]),
        "errors": {
            "time_integration": integration_error,
            "regional": regional_error,
            "spatial": spatial_error,
        },
        "missing_gates": [
            "source-admission",
            "locked-reference",
            "external-transfer",
            "clinical-validation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(qualify(), indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
