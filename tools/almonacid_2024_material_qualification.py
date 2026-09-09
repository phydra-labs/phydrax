#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compare native stresses with frozen, independently executed upstream points."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.continuum import (
    Almonacid2024MaterialParameters,
    almonacid_2024_material_response,
)


_DEFAULT_ORACLE = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/flexodeal_0698e3d/reference_outputs/material-points.tar.gz"
)


def qualify(oracle: Path = _DEFAULT_ORACLE) -> dict:
    with tarfile.open(oracle, "r:gz") as archive:
        with archive.extractfile("run-record.json") as stream:
            record = json.load(stream)
        content_id = record.pop("content_id")
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != content_id:
            raise ValueError("Material oracle record content identity does not match.")
        with archive.extractfile("samples.jsonl") as stream:
            raw_samples = stream.read()
        if (
            hashlib.sha256(raw_samples).hexdigest()
            != record["artifacts"]["samples.jsonl"]["sha256"]
        ):
            raise ValueError("Material oracle samples do not match their recorded bytes.")
    samples = [json.loads(line) for line in raw_samples.splitlines()]
    evaluate = eqx.filter_jit(almonacid_2024_material_response)
    maximum_errors: dict[str, float] = {}
    for sample in samples:
        values = sample["parameter_values"]
        parameters = Almonacid2024MaterialParameters(
            **dict(zip(record["parameter_names"][:-3], values[:-3], strict=True)),
            base_coefficients=values[-3:],
        )
        result = evaluate(
            parameters,
            jnp.asarray(sample["F"], dtype=jnp.float64),
            jnp.asarray(sample["F_previous"], dtype=jnp.float64),
            jnp.asarray(sample["pressure_Pa"], dtype=jnp.float64),
            jnp.asarray(sample["dilation"], dtype=jnp.float64),
            jnp.asarray(sample["activation"], dtype=jnp.float64),
            jnp.asarray(sample["dt_s"], dtype=jnp.float64),
            jnp.asarray(sample["reference_direction"], dtype=jnp.float64),
            jnp.asarray(sample["tissue_id"], dtype=jnp.int32),
            dynamic=sample["dynamic"],
        )
        if not bool(result.admissible):
            raise ValueError(
                f"Native material rejected valid source point {sample['id']}."
            )
        comparisons = (
            ("first_piola_Pa", result.first_piola_Pa, sample["first_piola_Pa"], 5e-8),
            ("kirchhoff_Pa", result.kirchhoff_Pa, sample["kirchhoff_Pa"], 5e-8),
            ("volume_ratio", result.volume_ratio, sample["volume_ratio"], 1e-12),
            ("fiber_stretch", result.fiber_stretch, sample["fiber_stretch"], 1e-12),
            (
                "isochoric_fiber_stretch",
                result.isochoric_fiber_stretch,
                sample["isochoric_fiber_stretch"],
                1e-12,
            ),
            (
                "normalized_strain_rate",
                result.normalized_strain_rate,
                sample["normalized_strain_rate"],
                1e-12,
            ),
            (
                "volume_constraint",
                result.volume_constraint,
                sample["volume_ratio"] - sample["dilation"],
                1e-12,
            ),
            (
                "dilation_residual_Pa",
                result.dilation_residual_Pa,
                sample["dPsi_vol_dJ_Pa"] - sample["pressure_Pa"],
                5e-8,
            ),
        )
        for name, actual, expected, absolute_tolerance in comparisons:
            actual_array, expected_array = np.asarray(actual), np.asarray(expected)
            np.testing.assert_allclose(
                actual_array,
                expected_array,
                rtol=2e-11,
                atol=absolute_tolerance,
                err_msg=f"Source point {sample['id']}: {name}",
            )
            error = float(np.max(np.abs(actual_array - expected_array)))
            maximum_errors[name] = max(maximum_errors.get(name, 0.0), error)
    return {
        "passed": True,
        "sample_count": len(samples),
        "oracle_content_id": content_id,
        "maximum_absolute_errors": maximum_errors,
        "scope": (
            "Pinned upstream material-point stresses and mixed residuals; not "
            "biological validation or equality to its approximate hand-coded "
            "Newton tangent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=_DEFAULT_ORACLE)
    args = parser.parse_args()
    with jax.enable_x64(True):
        print(json.dumps(qualify(args.oracle), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
