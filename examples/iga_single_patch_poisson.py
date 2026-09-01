#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx


p = 2
grid = phx.discretization.iga.BSplineGrid.open_uniform(
    p,
    1,
    interval=(0.0, 1.0),
)
coordinates = grid.greville_abscissae
xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
geometry = phx.discretization.iga.NURBSGeometryState(
    jnp.stack((xx, yy), axis=-1),
    jnp.ones((grid.coefficient_count, grid.coefficient_count)),
)
plan = phx.discretization.iga.IsogeometricPlan.isoparametric(
    (grid, grid),
    geometry,
    field_name="u",
    axis_names=("xi", "eta"),
    quadrature_policy=phx.discretization.iga.IsogeometricQuadraturePolicy(p + 1),
    qualification_policy=phx.discretization.iga.IsogeometricH1QualificationPolicy(),
)
discretization = plan.prepare(numeric_version="single-patch-poisson")
constraint = discretization.homogeneous_trace_constraint("u")
source = phx.equations.coefficient(
    lambda points, args: (
        2.0
        * (
            points[..., 0] * (1.0 - points[..., 0])
            + points[..., 1] * (1.0 - points[..., 1])
        )
    ),
    coefficient_id="iga-single-patch-poisson-source",
)
form = phx.equations.FiniteElementForm(
    "iga-single-patch-poisson",
    "u",
    (
        phx.equations.DiffusionAction("u", 1.0),
        phx.equations.SourceAction("u", source),
    ),
)
compiled = phx.equations.compile_finite_element_problem(
    form,
    discretization,
    constraint=constraint,
    execution_policy=phx.equations.FiniteElementExecutionPolicy(
        realization="matrix_free",
        local_kernel="sum_factorized",
    ),
)
system, right_hand_side = compiled.linear_system()
result = phx.linalg.solve(system, right_hand_side)
solution = compiled.expand(result.value)
line_coefficients = jnp.asarray((0.0, 0.5, 0.0), dtype=solution.dtype)
expected = jnp.outer(line_coefficients, line_coefficients).reshape(solution.shape)
coefficient_error = jnp.max(jnp.abs(solution - expected))
free_residual = compiled.residual(result.value)
residual_norm = jnp.sqrt(jnp.real(jnp.vdot(free_residual, free_residual)))
right_hand_side_norm = jnp.sqrt(jnp.real(jnp.vdot(right_hand_side, right_hand_side)))
normalized_residual = residual_norm / jnp.maximum(right_hand_side_norm, 1.0)
tolerance = 4096.0 * jnp.finfo(solution.dtype).eps
successful = bool(jnp.all(result.successful))
passed = (
    successful
    and float(coefficient_error) <= float(tolerance)
    and float(normalized_residual) <= float(tolerance)
)
if not passed:
    raise RuntimeError("The isogeometric Poisson example failed exactness checks.")

print(
    json.dumps(
        {
            "coefficient_error": float(coefficient_error),
            "compilation_id": compiled.compilation_id,
            "normalized_free_residual": float(normalized_residual),
            "prepared_id": discretization.prepared_id,
            "successful": successful,
        },
        indent=2,
        sort_keys=True,
    )
)
