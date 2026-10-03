# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Meshfree solid mechanics: small strain, finite strain, and Herrmann mixed form.

All workflows run on a jittered unit plate whose volumes are the tensor
trapezoid rule and whose boundary points carry the trapezoid boundary measure
(corners: the normalized diagonal normal with measure ``h/sqrt 2``).

* Manufactured refinement: ``u = (0.1 sin(2x) e^y, 0.05 cos(x + 2y))`` with a
  traction face at ``x = 1`` and prescribed displacement elsewhere; the body
  force is the autodiff divergence of the independent Hooke stress.
* Rigid motion: an infinitesimal rotation plus translation has zero strain,
  stress, and energy.
* Plate tension: rollers on ``x = 0`` and ``y = 0`` and a traction ``(t, 0)`` on
  ``x = 1`` give homogeneous plane-strain uniaxial stress; displacement,
  energy ``U = t eps_xx / 2``, work ``W = 2 U``, and the closed boundary
  resultant are compared with closed forms.
* Finite strain: the same plate with the logarithmic neo-Hookean law; the
  homogeneous stretches solve ``P_xx = t, P_yy = 0`` independently, and the
  stored energy and load-path work are compared with ``W(a, b)``. A small
  load reproduces the linear response.
* Clamped Herrmann form: ``u = u_s - kappa grad psi`` and ``p = lap psi`` (with a
  nonzero mean) satisfy ``div u + kappa p = 0`` exactly for every ``kappa``;
  the all-Dirichlet body recovers displacement and the absolute pressure,
  whose mean is fixed by the boundary volume flux, as ``kappa = 1/lambda -> 0``
  without locking.
* Refusal: a Newton budget of one iteration refuses the first increment and
  rolls back to the last accepted (zero) load.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from phydrax.discretization import (
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MechanicsStatus,
    MeshfreeElasticityPlan,
    MeshfreeGeneralizedStokesPlan,
    MeshfreeHyperelasticPlan,
)
from phydrax.nonlinear import NonlinearTermination
from phydrax.operators.mechanics import (
    LinearElasticityTensor,
    NeoHookeanLaw,
    NeoHookeanParameters,
)


WorkflowMetric = float | int | bool | str | list[float] | list[int] | list[str]

LAMBDA, MU = 1.0, 0.5


def plate(side: int) -> PreparedPointCloudDiscretization:
    axis = np.linspace(0.0, 1.0, side)
    grid = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([value.reshape(-1) for value in grid], axis=1)
    faces = np.count_nonzero(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    boundary = faces > 0
    spacing = 1.0 / (side - 1)
    points[~boundary] += (
        np.random.default_rng(7).uniform(-0.15, 0.15, (np.count_nonzero(~boundary), 2))
        * spacing
    )
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    measure = np.where(faces == 2, spacing / np.sqrt(2.0), spacing)
    return PointCloudPlan(
        points,
        spacing**2 * 0.5**faces,
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=np.where(boundary, measure, 0.0),
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()


def tensor() -> LinearElasticityTensor:
    return LinearElasticityTensor.isotropic(2, lame_lambda=LAMBDA, shear_modulus=MU)


def hooke(displacement: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Independent plane-strain Hooke stress of an analytic displacement."""

    def stress(x: Array) -> Array:
        gradient = jax.jacfwd(displacement)(x)
        strain = 0.5 * (gradient + gradient.T)
        return LAMBDA * jnp.trace(strain) * jnp.eye(2) + 2.0 * MU * strain

    return stress


def smooth(x: Array) -> Array:
    return jnp.stack(
        (0.1 * jnp.sin(2.0 * x[0]) * jnp.exp(x[1]), 0.05 * jnp.cos(x[0] + 2.0 * x[1]))
    )


def traction_face(
    cloud: PreparedPointCloudDiscretization, displacement: Callable[[Array], Array]
) -> PointBoundaryPlan:
    """Exact traction on ``x = 1``; exact displacement on the other faces."""
    x = cloud.points
    exact = jax.vmap(displacement)(x)
    right = np.isclose(np.asarray(x[:, 0]), 1.0)
    rows = np.flatnonzero(right)
    rest = np.flatnonzero(np.asarray(cloud.plan.boundary_mask) & ~right)
    traction = jax.vmap(hooke(displacement))(x[rows])[:, :, 0]
    normals = np.tile([[1.0, 0.0]], (rows.size, 1))
    conditions = []
    for a in range(2):
        conditions.append(
            PointBoundaryCondition(
                "neumann",
                rows,
                traction[:, a],
                label=f"traction-{a}",
                component=a,
                normals=normals,
            )
        )
        conditions.append(
            PointBoundaryCondition(
                "dirichlet", rest, exact[rest, a], label=f"clamp-{a}", component=a
            )
        )
    return PointBoundaryPlan(conditions, row_count=x.shape[0], components=2)


def rollers(cloud: PreparedPointCloudDiscretization, pull: float) -> PointBoundaryPlan:
    """Rollers on ``x = 0`` (``u_x``) and ``y = 0`` (``u_y``); traction ``(pull, 0)`` on ``x = 1``."""
    x = np.asarray(cloud.points)
    left, right = np.isclose(x[:, 0], 0.0), np.isclose(x[:, 0], 1.0)
    bottom, top = np.isclose(x[:, 1], 0.0), np.isclose(x[:, 1], 1.0)

    def traction(
        label: str,
        mask: NDArray[np.bool_],
        normal: tuple[float, float],
        component: int,
        value: float,
    ) -> PointBoundaryCondition:
        rows = np.flatnonzero(mask)
        return PointBoundaryCondition(
            "neumann",
            rows,
            value,
            label=label,
            component=component,
            normals=np.tile(np.asarray(normal), (rows.size, 1)),
        )

    sides = ~left & ~right
    return PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", np.flatnonzero(left), label="roller-x"),
            traction("pull", right, (1.0, 0.0), 0, pull),
            traction("free-x-bottom", bottom & sides, (0.0, -1.0), 0, 0.0),
            traction("free-x-top", top & sides, (0.0, 1.0), 0, 0.0),
            PointBoundaryCondition(
                "dirichlet", np.flatnonzero(bottom), label="roller-y", component=1
            ),
            traction("free-y-top", top, (0.0, 1.0), 1, 0.0),
            traction("free-y-left", left & ~bottom & ~top, (-1.0, 0.0), 1, 0.0),
            traction("free-y-right", right & ~bottom & ~top, (1.0, 0.0), 1, 0.0),
        ),
        row_count=x.shape[0],
        components=2,
    )


def uniaxial_small_strain(pull: float) -> tuple[float, float]:
    stiff = LAMBDA + 2.0 * MU
    axial = pull * stiff / (4.0 * MU * (LAMBDA + MU))
    return axial, -LAMBDA * axial / stiff


def uniaxial_neo_hookean(pull: float) -> tuple[float, float, float]:
    """Stretches and energy of ``P = mu (F - F^-T) + lambda ln J F^-T``, ``P_yy = 0``."""
    stretch = np.ones(2)
    for _ in range(50):
        a, b = stretch
        log_j = np.log(a * b)
        residual = np.asarray(
            [
                MU * (a - 1.0 / a) + LAMBDA * log_j / a - pull,
                MU * (b - 1.0 / b) + LAMBDA * log_j / b,
            ]
        )
        cross = LAMBDA / (a * b)
        jacobian = np.asarray(
            [
                [MU * (1.0 + a**-2) + LAMBDA * (1.0 - log_j) / a**2, cross],
                [cross, MU * (1.0 + b**-2) + LAMBDA * (1.0 - log_j) / b**2],
            ]
        )
        stretch = stretch - np.linalg.solve(jacobian, residual)
    a, b = stretch
    log_j = np.log(a * b)
    energy = 0.5 * MU * (a**2 + b**2 - 2.0) - MU * log_j + 0.5 * LAMBDA * log_j**2
    return float(a), float(b), float(energy)


def run_refinement(sides: tuple[int, ...]) -> dict[str, WorkflowMetric]:
    errors, balances, traction_errors, statuses = [], [], [], []
    for side in sides:
        cloud = plate(side)
        x = cloud.points
        force = jax.vmap(
            lambda y: -jnp.trace(jax.jacfwd(hooke(smooth))(y), axis1=1, axis2=2)
        )(x)
        result = (
            MeshfreeElasticityPlan(cloud, traction_face(cloud, smooth), tensor())
            .prepare()
            .solve(force)
        )
        exact = jax.vmap(smooth)(x)
        # Face rows only: corner points carry the diagonal normal, where the
        # computed traction is sigma n_corner rather than sigma e_x.
        y = np.asarray(x[:, 1])
        right = (
            np.isclose(np.asarray(x[:, 0]), 1.0)
            & ~np.isclose(y, 0.0)
            & ~np.isclose(y, 1.0)
        )
        expected = jax.vmap(hooke(smooth))(x)[:, :, 0]
        statuses.append(MechanicsStatus(int(result.status)).name)
        errors.append(float(jnp.max(jnp.abs(result.candidate_displacement - exact))))
        balances.append(float(result.force_balance_defect))
        traction_errors.append(
            float(jnp.max(jnp.abs((result.boundary_traction - expected)[right])))
        )
    spacing = 1.0 / (np.asarray(sides, dtype=np.float64) - 1.0)
    rates = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(
        spacing[:-1] / spacing[1:]
    )
    return {
        "sides": list(sides),
        "status": statuses,
        "max_displacement_error": errors,
        "observed_rate": rates.tolist(),
        "force_balance_defect": balances,
        "traction_face_error": traction_errors,
    }


def run_rigid(side: int) -> dict[str, WorkflowMetric]:
    cloud = plate(side)
    prepared = MeshfreeElasticityPlan(
        cloud, traction_face(cloud, smooth), tensor()
    ).prepare()
    x = cloud.points
    rigid = jnp.stack((0.3 - 0.7 * x[:, 1], -0.2 + 0.7 * x[:, 0]), axis=1)
    return {
        "max_strain": float(jnp.max(jnp.abs(prepared.strain(rigid)))),
        "max_stress": float(jnp.max(jnp.abs(prepared.stress(rigid)))),
        "strain_energy": float(prepared.strain_energy(rigid)),
    }


def run_tension(
    side: int, *, pull: float = 0.05, large: float = 0.25
) -> dict[str, WorkflowMetric]:
    cloud = plate(side)
    x = cloud.points
    zero = jnp.zeros((x.shape[0], 2))
    boundary = rollers(cloud, pull)
    linear = MeshfreeElasticityPlan(cloud, boundary, tensor()).prepare()
    result = linear.solve(zero)
    axial, lateral = uniaxial_small_strain(pull)
    exact = jnp.stack((axial * x[:, 0], lateral * x[:, 1]), axis=1)

    law = NeoHookeanLaw(NeoHookeanParameters(jnp.asarray(MU), jnp.asarray(LAMBDA)))
    finite = MeshfreeHyperelasticPlan(cloud, boundary, law, load_steps=4).prepare()
    small = 1e-3
    gentle = finite.solve(zero, boundary_values={"pull": small})
    reference = linear.solve(zero, boundary_values={"pull": small})
    stretched = finite.solve(zero, boundary_values={"pull": large})
    a, b, energy = uniaxial_neo_hookean(large)
    homogeneous = jnp.stack(((a - 1.0) * x[:, 0], (b - 1.0) * x[:, 1]), axis=1)
    linearized = linear.solve(zero, boundary_values={"pull": large})

    # Refusal: one Newton iteration cannot meet the residual tolerance; the
    # first increment is refused and the committed state stays at zero load.
    strict = MeshfreeHyperelasticPlan(
        cloud,
        boundary,
        law,
        load_steps=2,
        termination=NonlinearTermination(
            absolute_residual=1e-14, relative_residual=1e-15, maximum_steps=1
        ),
    ).prepare()
    refused = strict.solve(zero, boundary_values={"pull": large})
    return {
        "linear_status": MechanicsStatus(int(result.status)).name,
        "linear_displacement_error": float(
            jnp.max(jnp.abs(result.candidate_displacement - exact))
        ),
        "linear_strain_energy": float(result.strain_energy),
        "analytic_strain_energy": 0.5 * pull * axial,
        "linear_external_work": float(result.external_work),
        "analytic_external_work": pull * axial,
        "clapeyron_defect": float(result.clapeyron_defect),
        "force_balance_defect": float(result.force_balance_defect),
        "linear_iterations": int(result.block.linear_result.diagnostics.iterations),
        "small_load_status": MechanicsStatus(int(gentle.status)).name,
        "small_load_relative_gap": float(
            jnp.max(jnp.abs(gentle.displacement - reference.displacement))
            / jnp.max(jnp.abs(reference.displacement))
        ),
        "finite_status": MechanicsStatus(int(stretched.status)).name,
        "finite_step_iterations": np.asarray(stretched.step_iterations).tolist(),
        "finite_displacement_error": float(
            jnp.max(jnp.abs(stretched.displacement - homogeneous))
        ),
        "analytic_stretches": [a, b],
        "min_jacobian": float(jnp.min(stretched.jacobian)),
        "stored_energy": float(stretched.stored_energy),
        "analytic_stored_energy": energy,
        "load_path_work": float(stretched.external_work),
        "energy_work_defect": float(stretched.energy_work_defect),
        "finite_vs_linear_relative_gap": float(
            jnp.max(jnp.abs(stretched.displacement - linearized.displacement))
            / jnp.max(jnp.abs(linearized.displacement))
        ),
        "refused_status": MechanicsStatus(int(refused.status)).name,
        "refused_step_status": np.asarray(refused.step_status).tolist(),
        "refused_accepted_load_factor": float(refused.accepted_load_factor),
        "refused_committed_max_displacement": float(
            jnp.max(jnp.abs(refused.displacement))
        ),
        "refused_candidate_max_displacement": float(
            jnp.max(jnp.abs(refused.candidate_displacement))
        ),
    }


def solenoidal(x: Array) -> Array:
    return 0.05 * jnp.stack(
        (
            jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
    )


MIXED_MEAN_PRESSURE = 0.2


def mixed_potential(x: Array) -> Array:
    """``psi`` with ``lap psi = p``."""
    return (
        -0.1 * jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]) / (2.0 * jnp.pi**2)
        + MIXED_MEAN_PRESSURE * (x[0] ** 2 + x[1] ** 2) / 4.0
    )


def mixed_pressure(x: Array) -> Array:
    return 0.1 * jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]) + MIXED_MEAN_PRESSURE


def mixed_displacement(x: Array, kappa: float) -> Array:
    """``u_s - kappa grad psi``: ``div u + kappa p = 0`` holds exactly."""
    return solenoidal(x) - kappa * jax.grad(mixed_potential)(x)


def mixed_force(x: Array, kappa: float) -> Array:
    """``-div(2 mu eps(u) - p I)`` of the manufactured pair by autodiff."""

    def cauchy(y: Array) -> Array:
        gradient = jax.jacfwd(mixed_displacement)(y, kappa)
        return MU * (gradient + gradient.T) - mixed_pressure(y) * jnp.eye(2)

    return -jnp.trace(jax.jacfwd(cauchy)(x), axis1=1, axis2=2)


def run_mixed(
    side: int, compressibilities: tuple[float, ...]
) -> dict[str, WorkflowMetric]:
    cloud = plate(side)
    x = cloud.points
    pressure = jax.vmap(mixed_pressure)(x)
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    weights = cloud.quadrature_weights
    statuses: list[str] = []
    iterations: list[int] = []
    field_errors: list[float] = []
    pressure_errors: list[float] = []
    mean_pressures: list[float] = []
    compatibility: list[float] = []
    volumetric: list[float] = []
    oscillation: list[float] = []
    for kappa in compressibilities:
        exact = jax.vmap(mixed_displacement, in_axes=(0, None))(x, kappa)
        boundary = PointBoundaryPlan(
            tuple(
                PointBoundaryCondition(
                    "dirichlet", rows, exact[rows, a], label=f"clamp-{a}", component=a
                )
                for a in range(2)
            ),
            row_count=x.shape[0],
            components=2,
        )
        result = (
            MeshfreeGeneralizedStokesPlan(
                cloud, boundary, shear_modulus=MU, compressibility=kappa
            )
            .prepare()
            .solve(jax.vmap(mixed_force, in_axes=(0, None))(x, kappa))
        )
        statuses.append(MechanicsStatus(int(result.status)).name)
        iterations.append(int(result.linear.diagnostics.iterations))
        field_errors.append(float(jnp.max(jnp.abs(result.field - exact))))
        pressure_errors.append(
            float(
                jnp.sqrt(
                    jnp.sum(weights * (result.pressure - pressure) ** 2)
                    / jnp.sum(weights)
                )
            )
        )
        mean_pressures.append(
            float(jnp.sum(weights * result.pressure) / jnp.sum(weights))
        )
        compatibility.append(float(result.compatibility_residual))
        volumetric.append(float(result.volumetric_defect))
        oscillation.append(float(result.pressure_oscillation))
    return {
        "compressibility": list(compressibilities),
        "status": statuses,
        "linear_iterations": iterations,
        "max_displacement_error": field_errors,
        "pressure_rms_error": pressure_errors,
        "mean_pressure": mean_pressures,
        "exact_mean_pressure": MIXED_MEAN_PRESSURE,
        "compatibility_residual": compatibility,
        "volumetric_defect": volumetric,
        "pressure_oscillation": oscillation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", type=int, nargs="+", default=[9, 13, 17])
    parser.add_argument("--side", type=int, default=9)
    args = parser.parse_args()
    report = {
        "manufactured_refinement": run_refinement(tuple(args.sides)),
        "rigid_body": run_rigid(args.side),
        "plate_tension": run_tension(args.side),
        "near_incompressible_mixed": run_mixed(args.side, (1.0, 1e-2, 1e-8)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
