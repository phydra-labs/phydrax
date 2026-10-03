#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCollocationStabilityRefusal,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MechanicsStatus,
    MeshfreeElasticityPlan,
    MeshfreeGeneralizedStokesPlan,
    MeshfreeHyperelasticPlan,
    PointGhostLayerPlan,
    PreparedMeshfreeHyperelastic,
)
from phydrax.nonlinear import NonlinearTermination
from phydrax.operators.mechanics import (
    LinearElasticityTensor,
    NeoHookeanLaw,
    NeoHookeanParameters,
)


_LAMBDA, _MU = 1.0, 0.5


def _square(side: int) -> PreparedPointCloudDiscretization:
    """Jittered unit square with tensor-trapezoid volume and boundary weights."""
    grid = np.meshgrid(*(np.linspace(0.0, 1.0, side),) * 2, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    on_face = np.isclose(points, 0.0) | np.isclose(points, 1.0)
    boundary = np.any(on_face, axis=1)
    rng = np.random.default_rng(7)
    points[~boundary] += rng.uniform(-0.15, 0.15, (np.count_nonzero(~boundary), 2)) / (
        side - 1
    )
    spacing = 1.0 / (side - 1)
    faces = np.count_nonzero(on_face, axis=1)
    volumes = spacing**2 * 0.5**faces
    # Face trapezoid weights. A corner joins two half-segments of length h/2
    # with normals n1, n2; the cloud stores the unit bisector (n1 + n2)/sqrt 2,
    # so the equivalent corner weight is h/sqrt 2.
    boundary_weights = np.where(
        boundary, spacing * np.where(faces == 2, 1.0 / np.sqrt(2.0), 1.0), 0.0
    )
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    return PointCloudPlan(
        points,
        volumes,
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=boundary_weights,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()


def _tensor() -> LinearElasticityTensor:
    return LinearElasticityTensor.isotropic(2, lame_lambda=_LAMBDA, shear_modulus=_MU)


def _stress(displacement: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Independent Hooke stress of an analytic displacement by autodiff."""

    def stress(x: Array) -> Array:
        gradient = jax.jacfwd(displacement)(x)
        strain = 0.5 * (gradient + gradient.T)
        return _LAMBDA * jnp.trace(strain) * jnp.eye(2) + 2.0 * _MU * strain

    return stress


def _traction_problem(
    cloud: PreparedPointCloudDiscretization, displacement: Callable[[Array], Array]
) -> PointBoundaryPlan:
    """Traction on x = 1, prescribed displacement on the other faces."""
    points = cloud.points
    exact = jax.vmap(displacement)(points)
    right = np.flatnonzero(np.isclose(np.asarray(points[:, 0]), 1.0))
    rest = np.flatnonzero(
        np.asarray(cloud.plan.boundary_mask) & ~np.isclose(np.asarray(points[:, 0]), 1.0)
    )
    traction = jax.vmap(_stress(displacement))(points[right])[:, :, 0]
    normals = np.tile([[1.0, 0.0]], (right.size, 1))
    conditions = []
    for a in range(2):
        conditions.append(
            PointBoundaryCondition(
                "neumann",
                right,
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
    return PointBoundaryPlan(conditions, row_count=points.shape[0], components=2)


def _body_force(displacement: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return lambda x: -jnp.trace(jax.jacfwd(_stress(displacement))(x), axis1=1, axis2=2)


def _affine(x: Array) -> Array:
    return jnp.asarray([[0.1, 0.2], [-0.05, 0.3]]) @ x + jnp.asarray([0.01, -0.02])


def _smooth(x: Array) -> Array:
    return jnp.stack(
        (0.1 * jnp.sin(2.0 * x[0]) * jnp.exp(x[1]), 0.05 * jnp.cos(x[0] + 2.0 * x[1]))
    )


@pytest.fixture(scope="module")
def coarse() -> PreparedPointCloudDiscretization:
    return _square(9)


def test_affine_patch_is_exact_with_energy_and_force_balance(
    coarse: PreparedPointCloudDiscretization,
) -> None:
    plan = MeshfreeElasticityPlan(coarse, _traction_problem(coarse, _affine), _tensor())

    prepared = plan.prepare()
    result = prepared.solve(jnp.zeros((coarse.state_shape[0], 2)))
    # The default ghost route keeps the PDE at traction points: admitted.
    assert prepared.block.stability.outcome == "admitted"

    assert int(result.status) == MechanicsStatus.ACCEPTED
    exact = jax.vmap(_affine)(coarse.points)
    np.testing.assert_allclose(result.displacement, exact, atol=1e-8)
    expected = _stress(_affine)(jnp.zeros(2))
    np.testing.assert_allclose(
        result.stress, jnp.broadcast_to(expected, result.stress.shape), atol=1e-8
    )
    # Constant stress: oint sigma n = 0 and Clapeyron 2U = W hold exactly under
    # the trapezoid boundary/volume rules for an affine displacement.
    assert float(result.force_balance_defect) < 1e-8
    assert float(result.clapeyron_defect) < 1e-8


def test_rigid_body_motion_has_zero_strain_and_stress(
    coarse: PreparedPointCloudDiscretization,
) -> None:
    plan = MeshfreeElasticityPlan(coarse, _traction_problem(coarse, _affine), _tensor())
    prepared = plan.prepare()
    x = coarse.points
    rigid = jnp.stack((0.3 - 0.7 * x[:, 1], -0.2 + 0.7 * x[:, 0]), axis=1)

    assert float(jnp.max(jnp.abs(prepared.strain(rigid)))) < 1e-10
    assert float(jnp.max(jnp.abs(prepared.stress(rigid)))) < 1e-10
    assert abs(float(prepared.strain_energy(rigid))) < 1e-20


def test_nonpolynomial_displacement_converges_under_refinement() -> None:
    errors, balances, tractions = [], [], []
    for side in (9, 13):
        cloud = _square(side)
        plan = MeshfreeElasticityPlan(cloud, _traction_problem(cloud, _smooth), _tensor())
        prepared = plan.prepare()
        result = prepared.solve(jax.vmap(_body_force(_smooth))(cloud.points))
        assert int(result.status) == MechanicsStatus.ACCEPTED
        exact = jax.vmap(_smooth)(cloud.points)
        errors.append(float(jnp.max(jnp.abs(result.displacement - exact))))
        balances.append(float(result.force_balance_defect))
        # Ghost route: the traction condition rows (one per ghost) are solved
        # to the solve tolerance, and the cloud's own one-sided traction at the
        # face (also at its corner rows) converges to the prescribed traction.
        assert float(result.block.boundary_residual_norm) <= float(
            result.block.residual_tolerance
        )
        right = np.isclose(np.asarray(cloud.points[:, 0]), 1.0)
        normal = jnp.zeros_like(cloud.points).at[:, 0].set(1.0)
        computed = prepared.traction(result.displacement, normal)
        expected = jax.vmap(_stress(_smooth))(cloud.points)[:, :, 0]
        tractions.append(float(jnp.max(jnp.abs((computed - expected)[right]))))

    assert errors[1] < 0.6 * errors[0]
    assert tractions[1] < tractions[0]
    assert balances[1] < balances[0] < 0.1


def _roller_problem(
    cloud: PreparedPointCloudDiscretization, displacement: Callable[[Array], Array]
) -> PointBoundaryPlan:
    """Rollers on x = 0 (u_x) and y = 0 (u_y); every other (row, component) is
    traction-loaded with that face's normal, so at the corner (1, 1) component
    x owns the right face's traction and component y the top face's."""
    points = cloud.points
    x = np.asarray(points)
    exact = jax.vmap(displacement)(points)
    stress = jax.vmap(_stress(displacement))(points)
    left, right = np.isclose(x[:, 0], 0.0), np.isclose(x[:, 0], 1.0)
    bottom, top = np.isclose(x[:, 1], 0.0), np.isclose(x[:, 1], 1.0)

    def traction(
        mask: np.ndarray, normal: tuple[float, float], a: int
    ) -> PointBoundaryCondition:
        rows = np.flatnonzero(mask)
        n = np.tile(np.asarray(normal), (rows.size, 1))
        values = jnp.einsum("nj,nj->n", stress[rows, a], jnp.asarray(n))
        return PointBoundaryCondition(
            "neumann", rows, values, label=f"t{a}-{normal}", component=a, normals=n
        )

    sides, ends = ~left & ~right, ~bottom & ~top
    return PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet", np.flatnonzero(left), exact[left, 0], label="rx"
            ),
            traction(right, (1.0, 0.0), 0),
            traction(bottom & sides, (0.0, -1.0), 0),
            traction(top & sides, (0.0, 1.0), 0),
            PointBoundaryCondition(
                "dirichlet",
                np.flatnonzero(bottom),
                exact[bottom, 1],
                label="ry",
                component=1,
            ),
            traction(top, (0.0, 1.0), 1),
            traction(left & ends, (-1.0, 0.0), 1),
            traction(right & ends, (1.0, 0.0), 1),
        ),
        row_count=x.shape[0],
        components=2,
    )


def test_traction_traction_corner_ghost_converges_and_is_admitted() -> None:
    errors = []
    for side in (9, 17):
        cloud = _square(side)
        boundary = _roller_problem(cloud, _smooth)
        prepared = MeshfreeElasticityPlan(cloud, boundary, _tensor()).prepare()
        assert prepared.block.stability.outcome == "admitted"
        # The corner (1, 1) ghost lies on the bisector of the two declared
        # normals, while each component keeps its own normal there.
        assert prepared.plan.system.ghosts is not None
        evidence = prepared.plan.system.ghosts.evidence
        corner = int(
            np.flatnonzero(np.all(np.isclose(np.asarray(cloud.points), 1.0), axis=1))[0]
        )
        g = int(
            np.flatnonzero(np.asarray(prepared.plan.system.ghosts.plan.rows) == corner)[0]
        )
        np.testing.assert_allclose(evidence.direction[g], np.full(2, 2.0**-0.5))
        np.testing.assert_allclose(evidence.component_normals[g], np.eye(2))
        result = prepared.solve(jax.vmap(_body_force(_smooth))(cloud.points))
        assert int(result.status) == MechanicsStatus.ACCEPTED
        errors.append(
            float(jnp.max(jnp.abs(result.displacement - jax.vmap(_smooth)(cloud.points))))
        )
    # Design order (PHS degree 3): about second order under halving h.
    assert errors[1] < 0.35 * errors[0]


def test_opposite_flux_normals_at_one_row_are_refused(
    coarse: PreparedPointCloudDiscretization,
) -> None:
    row = np.flatnonzero(np.isclose(np.asarray(coarse.points[:, 0]), 1.0))[:1]
    rest = np.setdiff1d(np.flatnonzero(np.asarray(coarse.plan.boundary_mask)), row)
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "neumann", row, 0.0, label="a", component=0, normals=[[1.0, 0.0]]
            ),
            PointBoundaryCondition(
                "neumann", row, 0.0, label="b", component=1, normals=[[-1.0, 0.0]]
            ),
            PointBoundaryCondition("dirichlet", rest, 0.0, label="c0"),
            PointBoundaryCondition("dirichlet", rest, 0.0, label="c1", component=1),
        ),
        row_count=coarse.state_shape[0],
        components=2,
    )
    with pytest.raises(ValueError, match="cancel"):
        PointGhostLayerPlan(boundary)


def test_floating_and_mismatched_elasticity_is_refused(
    coarse: PreparedPointCloudDiscretization,
) -> None:
    rows = np.flatnonzero(np.asarray(coarse.plan.boundary_mask))
    normals = np.asarray(coarse.plan.boundary_normals)[rows]
    traction_only = PointBoundaryPlan(
        tuple(
            PointBoundaryCondition(
                "neumann", rows, 0.0, label=f"free-{a}", component=a, normals=normals
            )
            for a in range(2)
        ),
        row_count=coarse.state_shape[0],
        components=2,
    )
    with pytest.raises(ValueError, match="floats"):
        MeshfreeElasticityPlan(coarse, traction_only, _tensor())
    with pytest.raises(ValueError, match="dimension"):
        MeshfreeElasticityPlan(
            coarse,
            _traction_problem(coarse, _affine),
            LinearElasticityTensor.isotropic(3, lame_lambda=1.0, shear_modulus=1.0),
        )


def _pulled_plate(
    cloud: PreparedPointCloudDiscretization,
) -> PreparedMeshfreeHyperelastic:
    """Neo-Hookean plate pulled on x = 1 (dead traction), clamped elsewhere."""
    rows = np.flatnonzero(np.isclose(np.asarray(cloud.points[:, 0]), 1.0))
    rest = np.flatnonzero(
        np.asarray(cloud.plan.boundary_mask)
        & ~np.isclose(np.asarray(cloud.points[:, 0]), 1.0)
    )
    normals = np.tile([[1.0, 0.0]], (rows.size, 1))
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "neumann", rows, 0.08, label="pull", component=0, normals=normals
            ),
            PointBoundaryCondition(
                "neumann", rows, 0.0, label="slide", component=1, normals=normals
            ),
            PointBoundaryCondition("dirichlet", rest, 0.0, label="clamp-x"),
            PointBoundaryCondition("dirichlet", rest, 0.0, label="clamp-y", component=1),
        ),
        row_count=cloud.state_shape[0],
        components=2,
    )
    law = NeoHookeanLaw(NeoHookeanParameters(jnp.asarray(_MU), jnp.asarray(_LAMBDA)))
    return MeshfreeHyperelasticPlan(cloud, boundary, law, load_steps=4).prepare()


@pytest.fixture(scope="module")
def hyperelastic(
    coarse: PreparedPointCloudDiscretization,
) -> PreparedMeshfreeHyperelastic:
    return _pulled_plate(coarse)


def test_finite_strain_tangent_matches_finite_differences_and_its_adjoint(
    hyperelastic: PreparedMeshfreeHyperelastic,
) -> None:
    # Equation unknowns: cloud displacements, then the traction ghosts.
    count = hyperelastic.unknown_count
    assert count > hyperelastic.discretization.state_shape[0]
    rng = np.random.default_rng(3)
    state = jnp.asarray(0.02 * rng.standard_normal((count, 2)))
    force = jnp.asarray(
        0.1 * rng.standard_normal((hyperelastic.discretization.state_shape[0], 2))
    )
    linearization = hyperelastic.linearize(state, force)
    direction = jnp.asarray(rng.standard_normal(2 * count))
    cotangent = jnp.asarray(rng.standard_normal(2 * count))

    def flat(value: Array) -> Array:
        return hyperelastic.residual(value.reshape((2, count)).T, force).T.reshape(-1)

    base = state.T.reshape(-1)
    step = 1e-6
    finite_difference = (
        flat(base + step * direction) - flat(base - step * direction)
    ) / (2.0 * step)
    tangent = linearization.jvp(direction)
    np.testing.assert_allclose(
        tangent,
        finite_difference,
        rtol=1e-6,
        atol=1e-6 * float(jnp.max(jnp.abs(tangent))),
    )
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        jnp.vdot(linearization.vjp(cotangent), direction),
        rtol=1e-12,
    )


def test_finite_strain_equilibrium_balances_energy_and_work(
    coarse: PreparedPointCloudDiscretization,
    hyperelastic: PreparedMeshfreeHyperelastic,
) -> None:
    result = hyperelastic.solve(jnp.zeros((coarse.state_shape[0], 2)))

    assert int(result.status) == MechanicsStatus.ACCEPTED
    np.testing.assert_array_equal(result.step_status, 0)
    assert float(result.accepted_load_factor) == 1.0
    assert float(jnp.min(result.jacobian)) > 0.0
    # Path work equals stored energy up to spatial consistency. The
    # clamp/traction corners carry a stress singularity, so the defect decays
    # slowly: measured 0.0181 and 0.0197 at sides 9 and 17 (its linear
    # Clapeyron analogue on the ghost route: 0.0235, 0.0236, 0.0114 at sides
    # 9, 17, 33, while square traction collocation grows 0.0167, 0.0263,
    # 0.0386). Kinematics from the cloud's own one-sided stencils instead of
    # the solved ghost-extended ones gave 0.0999 at side 9.
    finer = _square(17)
    refined = _pulled_plate(finer).solve(jnp.zeros((finer.state_shape[0], 2)))
    assert int(refined.status) == MechanicsStatus.ACCEPTED
    assert float(result.energy_work_defect) < 0.025
    assert float(refined.energy_work_defect) < 0.025
    # The pulled face elongates on average (its clamped corners contract by
    # the Poisson effect, so the sign is not pointwise).
    right = np.isclose(np.asarray(coarse.points[:, 0]), 1.0)
    assert coarse.plan.boundary_quadrature_weights is not None
    weights = coarse.plan.boundary_quadrature_weights[right]
    assert float(jnp.sum(weights * result.displacement[right, 0])) > 0.0
    assert result.ghost_displacement is not None
    unknowns = jnp.concatenate((result.displacement, result.ghost_displacement))
    residual = hyperelastic.residual(unknowns, jnp.zeros_like(result.displacement))
    assert float(jnp.max(jnp.abs(residual))) < 1e-8


def test_failed_increment_rolls_back_to_the_last_accepted_load(
    coarse: PreparedPointCloudDiscretization,
    hyperelastic: PreparedMeshfreeHyperelastic,
) -> None:
    plan = hyperelastic.plan
    strict = MeshfreeHyperelasticPlan(
        coarse,
        plan.system.boundary,
        plan.law,
        load_steps=2,
        termination=NonlinearTermination(
            absolute_residual=1e-14, relative_residual=1e-15, maximum_steps=1
        ),
    ).prepare()

    result = strict.solve(jnp.zeros((coarse.state_shape[0], 2)))

    assert int(result.status) != MechanicsStatus.ACCEPTED
    assert float(result.accepted_load_factor) == 0.0
    np.testing.assert_array_equal(result.displacement, 0.0)
    assert float(jnp.max(jnp.abs(result.candidate_displacement))) > 0.0


def test_near_incompressible_mixed_response(
    coarse: PreparedPointCloudDiscretization,
) -> None:
    def displacement(x: Array) -> Array:
        # Solenoidal: the incompressible limit admits it exactly.
        return 0.05 * jnp.stack(
            (
                jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
                -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
            )
        )

    def pressure(x: Array) -> Array:
        return 0.1 * jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1])

    def force(x: Array) -> Array:
        laplacian = jnp.trace(jax.jacfwd(jax.jacfwd(displacement))(x), axis1=1, axis2=2)
        return -_MU * laplacian + jax.grad(pressure)(x)

    points = coarse.points
    exact = jax.vmap(displacement)(points)
    right = np.flatnonzero(np.isclose(np.asarray(points[:, 0]), 1.0))
    rest = np.flatnonzero(
        np.asarray(coarse.plan.boundary_mask) & ~np.isclose(np.asarray(points[:, 0]), 1.0)
    )

    def cauchy(x: Array) -> Array:
        gradient = jax.jacfwd(displacement)(x)
        return _MU * (gradient + gradient.T) - pressure(x) * jnp.eye(2)

    traction = jax.vmap(cauchy)(points[right])[:, :, 0]
    normals = np.tile([[1.0, 0.0]], (right.size, 1))
    boundary = PointBoundaryPlan(
        tuple(
            condition
            for a in range(2)
            for condition in (
                PointBoundaryCondition(
                    "neumann",
                    right,
                    traction[:, a],
                    label=f"t{a}",
                    component=a,
                    normals=normals,
                ),
                PointBoundaryCondition(
                    "dirichlet", rest, exact[rest, a], label=f"d{a}", component=a
                ),
            )
        ),
        row_count=points.shape[0],
        components=2,
    )
    body = jax.vmap(force)(points)
    results = {}
    for compressibility in (0.0, 1e-8):
        prepared = MeshfreeGeneralizedStokesPlan(
            coarse, boundary, shear_modulus=_MU, compressibility=compressibility
        ).prepare()
        assert prepared.plan.gauge_row is None  # a traction face fixes the pressure
        results[compressibility] = prepared.solve(body)
        assert int(results[compressibility].status) == MechanicsStatus.ACCEPTED

    nearly = results[1e-8]
    mixed_error = float(jnp.max(jnp.abs(nearly.field - exact)))
    assert mixed_error < 5e-3
    np.testing.assert_allclose(nearly.field, results[0.0].field, atol=1e-6)
    # The stabilized constraint holds up to its consistent O(h^2) residual.
    assert float(nearly.volumetric_defect) < 0.02 * 0.05 * np.pi
    # The displacement-only form with lambda = 1 / kappa locks: its error is far
    # larger (or its solve is refused) on the same cloud and data. Its spectral
    # assessment is unresolved at lambda = 1e8 (no estimate converges), so the
    # default preparation refuses it; the evidence is recorded to compare.
    locked_plan = MeshfreeElasticityPlan(
        coarse,
        boundary,
        LinearElasticityTensor.isotropic(2, lame_lambda=1e8, shear_modulus=_MU),
        stability="diagnostic",
    ).prepare()
    assert locked_plan.block.stability.outcome != "admitted"
    locked = locked_plan.solve(body)
    assert (
        int(locked.status) != MechanicsStatus.ACCEPTED
        or float(jnp.max(jnp.abs(locked.candidate_displacement - exact)))
        > 5.0 * mixed_error
    )


_MEAN_PRESSURE = 0.2


def _herrmann_pressure(x: Array) -> Array:
    return 0.1 * jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]) + _MEAN_PRESSURE


def _herrmann_displacement(x: Array, kappa: float) -> Array:
    """``u_s - kappa grad psi`` with ``lap psi = p``: ``div u + kappa p = 0`` exactly."""

    def potential(y: Array) -> Array:
        return (
            -0.1 * jnp.cos(jnp.pi * y[0]) * jnp.cos(jnp.pi * y[1]) / (2.0 * jnp.pi**2)
            + _MEAN_PRESSURE * (y[0] ** 2 + y[1] ** 2) / 4.0
        )

    solenoidal = 0.05 * jnp.stack(
        (
            jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
    )
    return solenoidal - kappa * jax.grad(potential)(x)


def _herrmann_force(x: Array, kappa: float) -> Array:
    def cauchy(y: Array) -> Array:
        gradient = jax.jacfwd(_herrmann_displacement)(y, kappa)
        return _MU * (gradient + gradient.T) - _herrmann_pressure(y) * jnp.eye(2)

    return -jnp.trace(jax.jacfwd(cauchy)(x), axis1=1, axis2=2)


def _clamped(cloud: PreparedPointCloudDiscretization, exact: Array) -> PointBoundaryPlan:
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    return PointBoundaryPlan(
        tuple(
            PointBoundaryCondition(
                "dirichlet", rows, exact[rows, a], label=f"d{a}", component=a
            )
            for a in range(2)
        ),
        row_count=cloud.points.shape[0],
        components=2,
    )


def test_clamped_compressible_mixed_body_converges_without_spurious_pressure() -> None:
    # Clamped kappa = 1e-2 (lambda = 100) against an independent manufactured
    # pair with a nonzero mean pressure, which only the boundary volume flux
    # determines. The collocated-Laplacian stabilization was singular near this
    # kappa (corner pressure modes) and closed the mean by c / kappa.
    kappa = 1e-2
    displacement_errors, pressure_errors, compatibility = [], [], []
    for side in (9, 13):
        cloud = _square(side)
        x = cloud.points
        exact = jax.vmap(_herrmann_displacement, in_axes=(0, None))(x, kappa)
        boundary = _clamped(cloud, exact)
        body = jax.vmap(_herrmann_force, in_axes=(0, None))(x, kappa)
        prepared = MeshfreeGeneralizedStokesPlan(
            cloud, boundary, shear_modulus=_MU, compressibility=kappa
        ).prepare()
        assert prepared.plan.gauge_row is not None
        result = prepared.solve(body)
        assert int(result.status) == MechanicsStatus.ACCEPTED
        weights = cloud.quadrature_weights
        pressure_error = result.pressure - jax.vmap(_herrmann_pressure)(x)
        displacement_errors.append(float(jnp.max(jnp.abs(result.field - exact))))
        pressure_errors.append(
            float(jnp.sqrt(jnp.sum(weights * pressure_error**2) / jnp.sum(weights)))
        )
        compatibility.append(float(result.compatibility_residual))
        if side == 9:
            displacement_form = (
                MeshfreeElasticityPlan(
                    cloud,
                    boundary,
                    LinearElasticityTensor.isotropic(
                        2, lame_lambda=1.0 / kappa, shear_modulus=_MU
                    ),
                )
                .prepare()
                .solve(body)
            )
            assert int(displacement_form.status) == MechanicsStatus.ACCEPTED
            assert displacement_errors[0] < 0.25 * float(
                jnp.max(jnp.abs(displacement_form.displacement - exact))
            )

    # h ratio 12/8: at least second order in displacement and pressure
    # (observed 3.5 and 4.0), with the absolute pressure (mean included) accurate.
    assert displacement_errors[1] < displacement_errors[0] * (8.0 / 12.0) ** 2
    assert pressure_errors[1] < pressure_errors[0] * (8.0 / 12.0) ** 2
    assert pressure_errors[0] < 0.05 * _MEAN_PRESSURE
    assert np.isfinite(compatibility[0]) and compatibility[1] < compatibility[0]


def test_clamped_compressible_mixed_body_requires_boundary_quadrature() -> None:
    side = 5
    grid = np.meshgrid(*(np.linspace(0.0, 1.0, side),) * 2, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    boundary = np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    cloud = PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=2),
    ).prepare()
    clamp = _clamped(cloud, jnp.zeros_like(cloud.points))
    with pytest.raises(ValueError, match="boundary quadrature"):
        MeshfreeGeneralizedStokesPlan(
            cloud, clamp, shear_modulus=_MU, compressibility=1e-2
        )
    # Incompressible plans gauge the mean pressure and need no boundary flux.
    incompressible = MeshfreeGeneralizedStokesPlan(cloud, clamp, shear_modulus=_MU)
    assert incompressible.gauge_row is not None


@pytest.mark.parametrize(
    ("component", "anchored"), ((0, False), (1, True)), ids=("tangential", "normal")
)
def test_component_traction_only_anchors_owned_normal_pressure(
    coarse: PreparedPointCloudDiscretization, component: int, anchored: bool
) -> None:
    points = np.asarray(coarse.points)
    boundary_rows = np.flatnonzero(np.asarray(coarse.plan.boundary_mask))
    top = np.flatnonzero(
        np.isclose(points[:, 1], 1.0) & (points[:, 0] > 0.0) & (points[:, 0] < 1.0)
    )
    conditions = [
        PointBoundaryCondition(
            "neumann",
            top,
            0.0,
            label="traction",
            component=component,
            normals=np.tile([[0.0, 1.0]], (top.size, 1)),
        )
    ]
    for axis in range(2):
        rows = np.setdiff1d(boundary_rows, top) if axis == component else boundary_rows
        conditions.append(
            PointBoundaryCondition(
                "dirichlet", rows, 0.0, label=f"clamp-{axis}", component=axis
            )
        )
    boundary = PointBoundaryPlan(conditions, row_count=points.shape[0], components=2)
    prepared = MeshfreeGeneralizedStokesPlan(
        coarse, boundary, shear_modulus=_MU
    ).prepare()
    assert (prepared.plan.gauge_row is None) == anchored
    momentum, _ = prepared.physical_operator.mv(
        (
            jnp.zeros((points.shape[0] * 2,), dtype=jnp.float64),
            jnp.ones((points.shape[0],), dtype=jnp.float64),
        )
    )
    expected = np.zeros((2, points.shape[0]))
    expected[component, top] = -1.0 if anchored else 0.0
    np.testing.assert_allclose(momentum, expected.reshape(-1), atol=1e-10)
    result = prepared.solve(jnp.zeros_like(coarse.points))
    assert int(result.status) == MechanicsStatus.ACCEPTED
    np.testing.assert_allclose(result.pressure, 0.0, atol=1e-10)


@pytest.fixture(scope="module")
def traction_refinement() -> tuple[tuple[float, float, float, float, int], ...]:
    """``(h, relative L2 error, bulk tau, traction tau, status)`` at sides 33 and 49.

    ``tau`` is the largest physical residual of the exact field (truncation).
    The traction-face error is ``~0.98 h^2 - 9.6 h^3``: the traction rows
    (O(h^3) truncation) and the one-sided near-boundary bulk rows contribute
    O(h^3) errors anti-aligned with the interior O(h^2) error, so sides below
    ~33 sit in a cancellation regime (seed-0 Q11 plate local orders: -0.1,
    1.2, 1.6 from side 13 to 33, then 1.6, 1.8, 1.9 to 97; this seed: 1.0,
    1.5 from 17 to 33).
    The multigrid route keeps the check independent of the meshcore provider.
    """
    rows = []
    for side in (33, 49):
        cloud = _square(side)
        prepared = MeshfreeElasticityPlan(
            cloud,
            _traction_problem(cloud, _smooth),
            _tensor(),
            preconditioner="multigrid",
        ).prepare()
        force = jax.vmap(_body_force(_smooth))(cloud.points)
        result = prepared.solve(force)
        exact = jax.vmap(_smooth)(cloud.points)
        weights = cloud.quadrature_weights
        error = jnp.sqrt(
            jnp.sum(
                weights * jnp.sum((result.candidate_displacement - exact) ** 2, axis=1)
            )
            / jnp.sum(weights * jnp.sum(exact**2, axis=1))
        )
        truncation = np.abs(np.asarray(prepared.residual(exact, force)))
        right = np.isclose(np.asarray(cloud.points[:, 0]), 1.0)
        bulk = ~np.asarray(cloud.plan.boundary_mask)
        rows.append(
            (
                1.0 / (side - 1),
                float(error),
                float(truncation[bulk].max()),
                float(truncation[right].max()),
                int(result.status),
            )
        )
    return tuple(rows)


def test_traction_face_displacement_converges_at_second_order_asymptotically(
    traction_refinement: tuple[tuple[float, float, float, float, int], ...],
) -> None:
    (coarse_h, coarse_error, *_, coarse_status), (fine_h, fine_error, *_, fine_status) = (
        traction_refinement
    )
    assert coarse_status == fine_status == MechanicsStatus.ACCEPTED
    # Design order 2 (PHS-RBF-FD degree 3); observed 1.71 between sides 33 and 49.
    assert np.log(coarse_error / fine_error) / np.log(coarse_h / fine_h) >= 1.5


def test_traction_face_rows_and_bulk_rows_are_consistent_at_design_order(
    traction_refinement: tuple[tuple[float, float, float, float, int], ...],
) -> None:
    (
        (coarse_h, _, coarse_bulk, coarse_traction, _),
        (fine_h, _, fine_bulk, fine_traction, _),
    ) = traction_refinement
    ratio = np.log(coarse_h / fine_h)
    # Second derivatives lose two of the four reproduced orders: O(h^2)
    # (observed 2.07). Traction rows use first derivatives, O(h^3) (observed
    # 2.51); a boundary stencil losing a polynomial degree would give 2.
    assert np.log(coarse_bulk / fine_bulk) / ratio >= 1.8
    assert np.log(coarse_traction / fine_traction) / ratio >= 2.2


def test_square_traction_collocation_is_refused_by_its_spectral_assessment(
    coarse: PreparedPointCloudDiscretization,
) -> None:
    problem = _traction_problem(coarse, _affine)
    with pytest.raises(PointCollocationStabilityRefusal, match="nonpositive real part"):
        MeshfreeElasticityPlan(
            coarse, problem, _tensor(), traction_route="square"
        ).prepare()
    # Square traction collocation under recorded evidence: the same modes.
    diagnostic = MeshfreeElasticityPlan(
        coarse, problem, _tensor(), traction_route="square", stability="diagnostic"
    ).prepare()
    assert diagnostic.block.stability.outcome == "nonpositive-real-part"
