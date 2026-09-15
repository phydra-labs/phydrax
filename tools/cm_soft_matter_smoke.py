#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small thermodynamic soft-matter transaction and path-reversal smoke."""

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx
from phydrax.stochastic.path_sampling import (
    DiscretePathThermodynamicsPlan,
    normalized_discrete_path_thermodynamics,
    PathBuffer,
)


def main() -> None:
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    phase_model = phx.applications.phase_field.BinaryPhaseFieldModel(
        phx.equations.BinaryThermodynamicParameters(1.0, 1.0)
    )
    phase_method = phx.applications.phase_field.AllenCahnFEMPlan(
        phase_model,
        1.0,
    ).prepare(discretization, "eta")
    phase = phase_method.step_detailed(
        jnp.asarray(0),
        jnp.asarray(0.0),
        phase_method.initialize(jnp.asarray([-0.2, 0.1, 0.3, -0.1, 0.0])),
        jnp.asarray(0.01),
    )

    paths = tuple(
        PathBuffer.from_trajectory(
            jnp.asarray([[0.0], [offset], [1.0 + offset]]),
            jnp.asarray([0.0, 0.5, 1.0]),
            capacity=4,
        )
        for offset in (0.1, 0.3, 0.6)
    )
    forward = jnp.asarray([0.6, 0.3, 0.1])
    reverse = jnp.asarray([0.3, 0.3, 0.4])
    entropy = jnp.log(forward) - jnp.log(reverse)
    heat = -entropy
    path_result = normalized_discrete_path_thermodynamics(
        DiscretePathThermodynamicsPlan(
            1.0,
            maximum_paths=3,
            maximum_steps=3,
            autocorrelation_lag=1,
        ),
        paths,
        jnp.log(forward),
        jnp.log(reverse),
        jnp.zeros(3),
        -heat,
        heat,
        jnp.zeros(3),
    )
    if not bool(phase.successful & path_result.successful):
        raise RuntimeError("Soft-matter smoke failed thermodynamic acceptance.")
    print(
        json.dumps(
            {
                "phase_energy_before": float(phase.evidence.energy_before),
                "phase_energy_after": float(phase.evidence.energy_after),
                "phase_energy_stable": bool(phase.evidence.energy_stable),
                "phase_energy_balance_defect": float(
                    phase.evidence.energy_balance_defect
                ),
                "path_integral_fluctuation_average": float(
                    path_result.integral_fluctuation_average
                ),
                "path_reversal_residual": float(path_result.reversal_residual),
                "successful": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
