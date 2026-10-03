#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Meshfree solid-mechanics workflows on public APIs.

References are closed-form: plane-strain uniaxial tension of a roller-supported
plate (small strain and homogeneous logarithmic neo-Hookean), and a
manufactured solenoidal displacement for the Herrmann mixed form.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
from phydrax.operators.mechanics import (
    LinearElasticityTensor,
    NeoHookeanLaw,
    NeoHookeanParameters,
)


_LAMBDA, _MU = 1.0, 0.5
_SIDE = 9


@pytest.fixture(scope="module")
def plate() -> PreparedPointCloudDiscretization:
    """Jittered unit plate with tensor-trapezoid volume and boundary measures.

    Corner points carry the normalized diagonal normal and measure ``h/sqrt 2``,
    so ``w n = (h/2)(n_1 + n_2)`` is exactly the trapezoid share of both faces.
    """
    axis = np.linspace(0.0, 1.0, _SIDE)
    grid = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([value.reshape(-1) for value in grid], axis=1)
    on_face = np.isclose(points, 0.0) | np.isclose(points, 1.0)
    faces = np.count_nonzero(on_face, axis=1)
    boundary = faces > 0
    spacing = 1.0 / (_SIDE - 1)
    interior = ~boundary
    points[interior] += (
        np.random.default_rng(7).uniform(-0.15, 0.15, (np.count_nonzero(interior), 2))
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


def _rollers(cloud: PreparedPointCloudDiscretization, pull: float) -> PointBoundaryPlan:
    """Rollers ``u_x = 0`` on x = 0 and ``u_y = 0`` on y = 0; traction ``(pull, 0)`` on x = 1.

    Every other boundary (row, component) is traction free, so the exact
    response is homogeneous uniaxial stress for both small and finite strain.
    """
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


def _plane_strain_uniaxial(pull: float) -> tuple[float, float]:
    """Small-strain ``(eps_xx, eps_yy)`` with ``sigma_yy = sigma_xy = 0``."""
    stiff = _LAMBDA + 2.0 * _MU
    axial = pull * stiff / (4.0 * _MU * (_LAMBDA + _MU))
    return axial, -_LAMBDA * axial / stiff


def _neo_hookean_uniaxial(pull: float) -> tuple[float, float, float]:
    """Homogeneous stretches ``(a, b)`` and energy of ``P_xx = pull, P_yy = 0``.

    ``W = mu/2 (a^2 + b^2 - 2) - mu ln J + lambda/2 ln^2 J`` (plane strain,
    ``J = a b``) and ``P = mu (F - F^-T) + lambda ln J F^-T``, solved by a
    host Newton iteration on the two stretch equations.
    """
    stretch = np.ones(2)
    for _ in range(50):
        a, b = stretch
        log_j = np.log(a * b)
        residual = np.asarray(
            [
                _MU * (a - 1.0 / a) + _LAMBDA * log_j / a - pull,
                _MU * (b - 1.0 / b) + _LAMBDA * log_j / b,
            ]
        )
        cross = _LAMBDA / (a * b)
        jacobian = np.asarray(
            [
                [_MU * (1.0 + a**-2) + _LAMBDA * (1.0 - log_j) / a**2, cross],
                [cross, _MU * (1.0 + b**-2) + _LAMBDA * (1.0 - log_j) / b**2],
            ]
        )
        stretch = stretch - np.linalg.solve(jacobian, residual)
    a, b = stretch
    log_j = np.log(a * b)
    energy = 0.5 * _MU * (a**2 + b**2 - 2.0) - _MU * log_j + 0.5 * _LAMBDA * log_j**2
    return float(a), float(b), float(energy)


def test_plate_tension_from_small_to_finite_strain_with_energy_ledger(
    plate: PreparedPointCloudDiscretization,
) -> None:
    pull = 0.05
    boundary = _rollers(plate, pull)
    x = plate.points
    count = x.shape[0]
    zero = jnp.zeros((count, 2))
    tensor = LinearElasticityTensor.isotropic(2, lame_lambda=_LAMBDA, shear_modulus=_MU)
    linear = MeshfreeElasticityPlan(plate, boundary, tensor).prepare()

    result = linear.solve(zero)

    assert int(result.status) == MechanicsStatus.ACCEPTED
    axial, lateral = _plane_strain_uniaxial(pull)
    exact = jnp.stack((axial * x[:, 0], lateral * x[:, 1]), axis=1)
    np.testing.assert_allclose(result.displacement, exact, atol=1e-9)
    np.testing.assert_allclose(
        result.strain,
        jnp.broadcast_to(jnp.diag(jnp.asarray([axial, lateral])), (count, 2, 2)),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.stress,
        jnp.broadcast_to(jnp.diag(jnp.asarray([pull, 0.0])), (count, 2, 2)),
        atol=1e-8,
    )
    loaded = (
        np.isclose(np.asarray(x[:, 0]), 1.0)
        & ~np.isclose(np.asarray(x[:, 1]), 0.0)
        & ~np.isclose(np.asarray(x[:, 1]), 1.0)
    )
    np.testing.assert_allclose(
        result.boundary_traction[loaded] - jnp.asarray([pull, 0.0]), 0.0, atol=1e-8
    )
    # Unit plate: U = sigma_xx eps_xx / 2 and W = pull * u_x(1) = 2 U.
    np.testing.assert_allclose(result.strain_energy, 0.5 * pull * axial, rtol=1e-8)
    np.testing.assert_allclose(result.external_work, pull * axial, rtol=1e-8)
    assert float(result.clapeyron_defect) < 1e-8
    # Support reactions balance the pull: the closed boundary resultant vanishes.
    assert float(result.force_balance_defect) < 1e-8

    law = NeoHookeanLaw(NeoHookeanParameters(jnp.asarray(_MU), jnp.asarray(_LAMBDA)))
    finite = MeshfreeHyperelasticPlan(plate, boundary, law, load_steps=4).prepare()

    # Small load: the finite-strain response reduces to the linear one, with
    # a relative difference of the order of the strain itself.
    small = 1e-3
    gentle = finite.solve(zero, boundary_values={"pull": small})
    reference = linear.solve(zero, boundary_values={"pull": small})
    assert int(gentle.status) == MechanicsStatus.ACCEPTED
    assert int(reference.status) == MechanicsStatus.ACCEPTED
    scale = float(jnp.max(jnp.abs(reference.displacement)))
    gap = float(jnp.max(jnp.abs(gentle.displacement - reference.displacement)))
    assert gap < 5.0 * small * scale / _MU

    # Finite load: homogeneous neo-Hookean stretch, stored energy equals the
    # closed-form W(a, b) and the load-path work matches it.
    strong = 0.25
    stretched = finite.solve(zero, boundary_values={"pull": strong})
    assert int(stretched.status) == MechanicsStatus.ACCEPTED
    np.testing.assert_array_equal(stretched.step_status, 0)
    assert float(stretched.accepted_load_factor) == 1.0
    a, b, energy = _neo_hookean_uniaxial(strong)
    expected = jnp.stack(((a - 1.0) * x[:, 0], (b - 1.0) * x[:, 1]), axis=1)
    np.testing.assert_allclose(stretched.displacement, expected, atol=1e-8)
    np.testing.assert_allclose(stretched.jacobian, a * b, rtol=1e-8)
    np.testing.assert_allclose(stretched.stored_energy, energy, rtol=1e-8)
    np.testing.assert_allclose(stretched.external_work, energy, rtol=2e-2)
    assert float(stretched.energy_work_defect) < 2e-2
    linearized = linear.solve(zero, boundary_values={"pull": strong})
    # At 25 % of the shear modulus geometric and material nonlinearity are
    # visible against the small-strain prediction.
    assert float(jnp.max(jnp.abs(stretched.displacement - linearized.displacement))) > (
        0.02 * float(jnp.max(jnp.abs(linearized.displacement)))
    )


def _solenoidal(x: Array) -> Array:
    return 0.05 * jnp.stack(
        (
            jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
    )


def _pressure(x: Array) -> Array:
    return 0.1 * jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1])


def test_near_incompressible_herrmann_solid_recovers_a_solenoidal_displacement(
    plate: PreparedPointCloudDiscretization,
) -> None:
    # div u = 0, so -div(2 mu eps(u)) + grad p = -mu lap u + grad p; the
    # residual kappa p of the constraint is the O(kappa) consistency error.
    def force(x: Array) -> Array:
        laplacian = jnp.trace(jax.jacfwd(jax.jacfwd(_solenoidal))(x), axis1=1, axis2=2)
        return -_MU * laplacian + jax.grad(_pressure)(x)

    x = plate.points
    exact = jax.vmap(_solenoidal)(x)
    pressure = jax.vmap(_pressure)(x)
    rows = np.flatnonzero(np.asarray(plate.plan.boundary_mask))
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
    compressibility = 1e-6  # lambda = 1e6: Poisson ratio 0.4999998
    mixed = MeshfreeGeneralizedStokesPlan(
        plate, boundary, shear_modulus=_MU, compressibility=compressibility
    ).prepare()

    result = mixed.solve(jax.vmap(force)(x))

    assert int(result.status) == MechanicsStatus.ACCEPTED
    assert bool(result.linear.successful)
    weights = plate.quadrature_weights
    amplitude = float(jnp.max(jnp.abs(exact)))
    # No volumetric locking: the displacement error stays a small fraction of
    # the response at Poisson ratio ~1/2.
    assert float(jnp.max(jnp.abs(result.field - exact))) < 0.1 * amplitude
    # The clamped data carry no volume flux, so the boundary volume balance
    # fixes the mean pressure at zero: the absolute pressure is compared.
    pressure_error = jnp.sqrt(jnp.sum(weights * (result.pressure - pressure) ** 2))
    assert float(pressure_error) < 0.25 * float(jnp.sqrt(jnp.sum(weights * pressure**2)))
    # The volumetric constraint div u + kappa p = 0 is resolved and the
    # stabilized equal-order pressure is free of checkerboard modes.
    assert float(result.volumetric_defect) < 0.1 * amplitude
    assert float(result.pressure_oscillation) < 1.0


def test_finite_strain_plate_tension_stays_quadratic_on_a_refined_cloud() -> None:
    # 17 x 17 points (578 unknowns): GMRES preconditioned by a frozen ILU of the
    # reference tangent stagnated here, so the first load increment was refused
    # by residual stagnation. The closed forms are those of the side-9 workflow.
    side = 17
    axis = np.linspace(0.0, 1.0, side)
    grid = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([value.reshape(-1) for value in grid], axis=1)
    faces = np.count_nonzero(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    spacing = 1.0 / (side - 1)
    interior = faces == 0
    points[interior] += (
        np.random.default_rng(7).uniform(-0.15, 0.15, (np.count_nonzero(interior), 2))
        * spacing
    )
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    measure = np.where(faces == 2, spacing / np.sqrt(2.0), spacing)
    cloud = PointCloudPlan(
        points,
        spacing**2 * 0.5**faces,
        boundary_mask=faces > 0,
        boundary_normals=normals,
        boundary_quadrature_weights=np.where(faces > 0, measure, 0.0),
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()
    x = cloud.points
    zero = jnp.zeros((x.shape[0], 2))
    law = NeoHookeanLaw(NeoHookeanParameters(jnp.asarray(_MU), jnp.asarray(_LAMBDA)))
    finite = MeshfreeHyperelasticPlan(
        cloud, _rollers(cloud, 0.05), law, load_steps=4
    ).prepare()

    for pull in (1e-3, 0.25):
        result = finite.solve(zero, boundary_values={"pull": pull})

        assert int(result.status) == MechanicsStatus.ACCEPTED
        np.testing.assert_array_equal(result.step_status, 0)
        # A consistent tangent with a resolved inner solve converges
        # quadratically (two or three steps here; the stagnating inner solve
        # took 17-22 when it converged at all).
        assert int(jnp.max(result.step_iterations)) <= 4
        a, b, energy = _neo_hookean_uniaxial(pull)
        expected = jnp.stack(((a - 1.0) * x[:, 0], (b - 1.0) * x[:, 1]), axis=1)
        np.testing.assert_allclose(result.displacement, expected, atol=1e-9)
        np.testing.assert_allclose(result.stored_energy, energy, rtol=1e-8)
