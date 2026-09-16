from __future__ import annotations

import json

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import capture_environment, logical_array_bytes, measure_repeated
from phydrax.applications import polymer_field_theory as pft


def main() -> int:
    architecture = pft.PolymerContourArchitecturePlan(
        "AB",
        (
            pft.ContourBlockPlan("A", "left", "middle", 0, 0.5, 8),
            pft.ContourBlockPlan("B", "middle", "right", 1, 0.5, 8),
        ),
        root_node="left",
    )
    model = pft.IncompressibleGaussianMixturePlan(
        ("A", "B"),
        [1.0, 1.0],
        [[0.0, 0.0], [0.0, 0.0]],
        (pft.PolymerComponentPlan("AB", architecture, 1.0, 20.0),),
    )
    spectral = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(32),), axis_names=("x",)
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 8.0),))
    prepared = pft.SCFTPlan(model).prepare(spectral)
    fields = jnp.zeros(prepared.field_shape)
    operation = eqx.filter_jit(prepared.evaluate)
    result, timing = measure_repeated(lambda: operation(fields), warmup=1, repeats=3)
    payload = {
        "benchmark": "polymer-scft-evaluation",
        "grid_shape": list(spectral.physical_shape),
        "contour_steps": 16,
        "timing": timing.to_milliseconds_dict(),
        "logical_bytes": logical_array_bytes(result),
        "successful": bool(result.successful),
        "environment": capture_environment().to_dict(),
        "release_claim": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
