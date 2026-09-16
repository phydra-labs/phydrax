from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx
from phydrax.applications import polymer_field_theory as pft


def main() -> int:
    architecture = pft.PolymerContourArchitecturePlan(
        "AB",
        (
            pft.ContourBlockPlan("A", "left", "middle", 0, 0.5, 2),
            pft.ContourBlockPlan("B", "middle", "right", 1, 0.5, 2),
        ),
        root_node="left",
    )
    model = pft.IncompressibleGaussianMixturePlan(
        ("A", "B"),
        [1.0, 1.0],
        [[0.0, 0.0], [0.0, 0.0]],
        (pft.PolymerComponentPlan("AB", architecture, 1.0, 10.0),),
    )
    spectral = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(4),), axis_names=("x",)
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 4.0),))
    prepared = pft.SCFTPlan(model, maximum_iterations=8).prepare(spectral)
    result = pft.solve_scft(prepared, jnp.zeros(prepared.field_shape))
    successful = bool(result.successful)
    print(
        json.dumps(
            {
                "successful": successful,
                "free_energy": float(result.evaluation.free_energy),
                "residual": float(jnp.max(jnp.abs(result.evaluation.residual))),
            }
        )
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
