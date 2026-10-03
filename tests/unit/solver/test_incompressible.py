#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    MortonAddressPlan,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MechanicsStatus,
    MeshfreeEdgeRelationPlan,
    MeshfreeExteriorCalculusPlan,
    MeshfreeGeneralizedStokesPlan,
    MeshfreeMetricPolicy,
    minimum_image_edge_charts,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.linalg import FailurePolicy, LinearSolvePolicy, ProjectedPCG
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver import (
    CompatibleIncompressibleProjection,
    FixedStepProblem,
    IncompressibleDensityModel,
    MeshfreeFlowEvidence,
    MeshfreeFlowStatus,
    MeshfreeIncompressibleFlowPlan,
    PreparedMeshfreeIncompressibleFlow,
    solve_fixed_step,
)


_LENGTH = 2.0 * np.pi
_VISCOSITY = 0.05


@pytest.fixture(scope="module")
def periodic_box() -> tuple[PreparedMeshfreeExteriorCalculus, PointCloudPlan]:
    # 12 x 12 periodic lattice: the axial radius graph is a closed torus graph
    # (E - V + C = 145 cycles; the domain has first Betti number 2).
    count = 12
    spacing = _LENGTH / count
    lattice = np.stack(
        np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    points = (lattice + 0.5) * spacing
    box = HyperRectangle(np.zeros(2), np.full(2, _LENGTH))
    address = MortonAddressPlan.from_periodic_identifications(
        tuple(PeriodicIdentification(box, "x", component=axis) for axis in range(2)),
        maximum_depth=12,
    )
    capacity = 4 * count * count
    relation = MeshfreeEdgeRelationPlan(
        points, 1.1 * spacing, capacity, address=address
    ).prepare()
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        1.1 * spacing,
        capacity,
        node_volumes=np.full(count * count, spacing**2),
        intrinsic_displacements=minimum_image_edge_charts(relation, points),
        metric_policy=MeshfreeMetricPolicy(),
    ).prepare(edge_relation=relation)
    assert bool(exterior.metric_result.hilbert_admitted)
    cloud = PointCloudPlan(
        np.asarray(exterior.points),
        np.asarray(exterior.node_volumes),
        address=address,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    )
    return exterior, cloud


@pytest.fixture(scope="module")
def flow(
    periodic_box: tuple[PreparedMeshfreeExteriorCalculus, PointCloudPlan],
) -> PreparedMeshfreeIncompressibleFlow:
    exterior, cloud = periodic_box
    return MeshfreeIncompressibleFlowPlan(
        exterior,
        cloud.prepare(),
        viscosity=_VISCOSITY,
        domain_betti_number=2,
        transport_scheme="reconstructed",
        cycle_tolerance=0.15,
    ).prepare()


def _taylor_green(points: Array, time: float) -> Array:
    x, y = points[:, 0], points[:, 1]
    decay = np.exp(-2.0 * _VISCOSITY * time)
    return decay * jnp.stack((jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)), axis=1)


def test_projection_removes_a_gradient_and_keeps_the_solenoidal_field(
    flow: PreparedMeshfreeIncompressibleFlow,
) -> None:
    points = flow.points
    exact = _taylor_green(points, 0.0)
    # grad(0.1 sin(x + y)) is a pure gradient perturbation.
    gradient = 0.1 * jnp.cos(points[:, 0] + points[:, 1])[:, None] * jnp.ones((1, 2))

    result = flow.initialize(exact + gradient)

    assert int(result.evidence.status) == MeshfreeFlowStatus.ACCEPTED
    assert float(result.evidence.graph_divergence_norm) < 1e-12
    assert float(result.evidence.nodal_divergence_after) < 0.1 * float(
        result.evidence.nodal_divergence_before
    )
    # The removed part is the gradient, up to the O(h^2) reconstruction error.
    assert float(jnp.max(jnp.abs(result.state.velocity - exact))) < 1e-2
    assert result.evidence.cycles.cycle_space_dimension == 145
    assert result.evidence.cycles.artificial_cycle_dimension == 143


def test_taylor_green_decays_with_conserved_momentum_through_native_rollout(
    flow: PreparedMeshfreeIncompressibleFlow,
) -> None:
    initial = flow.initialize(_taylor_green(flow.points, 0.0)).state
    problem = FixedStepProblem(
        flow,
        initial,
        t0=0.0,
        t1=1.0,
        step_size=0.1,
        state_geometry=EuclideanStateGeometry(),
    )

    solution = solve_fixed_step(problem, evidence_retention="steps")

    assert bool(jnp.all(solution.successful))
    assert solution.evidence is not None
    evidence = solution.evidence.steps
    assert isinstance(evidence, MeshfreeFlowEvidence)
    assert float(jnp.max(evidence.graph_divergence_norm)) < 1e-12
    energy = np.asarray(evidence.kinetic_energy_after) / float(
        evidence.kinetic_energy_before[0]
    )
    exact = np.exp(-4.0 * _VISCOSITY * np.arange(1, 11) * 0.1)
    assert np.all(np.diff(energy) < 0.0)
    # Coarse lattice (h = pi/6): transport dissipation stays within 10%.
    np.testing.assert_allclose(energy, exact, rtol=0.1)
    momentum = np.asarray(evidence.momentum_after)
    np.testing.assert_allclose(momentum, 0.0, atol=1e-12)
    final = jax.tree.map(lambda leaf: leaf[-1], solution.states)
    assert (
        float(jnp.max(jnp.abs(final.velocity - _taylor_green(flow.points, 1.0)))) < 0.05
    )


def test_taylor_green_temporal_error_has_the_tableau_order(
    flow: PreparedMeshfreeIncompressibleFlow,
) -> None:
    # Fixed lattice, halved steps: successive differences isolate the temporal
    # error of ARS(2,2,2). A flux lagged over the step, one projection per step
    # or a projection of the whole pressure each step makes it first order.
    advance = eqx.filter_jit(flow.advance)
    initial = flow.initialize(_taylor_green(flow.points, 0.0)).state
    finals = []
    for steps in (8, 16, 32, 64):
        state = initial
        for index in range(steps):
            result = advance(state, jnp.asarray(index / steps), jnp.asarray(1.0 / steps))
            assert int(result.evidence.status) == MeshfreeFlowStatus.ACCEPTED
            state = result.state
        finals.append(state.velocity)

    differences = np.asarray(
        [
            float(jnp.max(jnp.abs(coarse - fine)))
            for coarse, fine in zip(finals, finals[1:], strict=False)
        ]
    )
    orders = np.log2(differences[:-1] / differences[1:])
    assert np.all(orders > 1.8), orders


def test_variable_density_conserves_mass_and_momentum(
    periodic_box: tuple[PreparedMeshfreeExteriorCalculus, PointCloudPlan],
) -> None:
    exterior, cloud = periodic_box
    flow = MeshfreeIncompressibleFlowPlan(
        exterior,
        cloud.prepare(),
        viscosity=_VISCOSITY,
        domain_betti_number=2,
        density_model="variable",
        cycle_tolerance=0.15,
    ).prepare()
    points = flow.points
    density = 1.0 + 0.5 * jnp.exp(
        -((points[:, 0] - np.pi) ** 2) - (points[:, 1] - np.pi) ** 2
    )
    state = flow.initialize(_taylor_green(points, 0.0), density=density).state
    step = eqx.filter_jit(flow.advance)
    mass = float(jnp.sum(exterior.node_volumes * density))

    for index in range(4):
        result = step(state, jnp.asarray(0.1 * index), jnp.asarray(0.1))
        assert int(result.evidence.status) == MeshfreeFlowStatus.ACCEPTED
        state = result.state
        assert float(result.evidence.graph_divergence_norm) < 1e-11
        np.testing.assert_allclose(result.evidence.mass_after, mass, rtol=1e-13)
        np.testing.assert_allclose(result.evidence.momentum_after, 0.0, atol=1e-10)

    # The limited transport keeps density inside its initial bounds.
    assert float(jnp.min(state.density)) >= 1.0 - 1e-12
    assert float(jnp.max(state.density)) <= float(jnp.max(density)) + 1e-12


def test_artificial_cycle_content_is_classified_and_refused(
    flow: PreparedMeshfreeIncompressibleFlow,
) -> None:
    plan = flow.plan
    assert isinstance(plan.projection, CompatibleIncompressibleProjection)
    smooth = plan.velocity_reconstruction.interpolate(_taylor_green(flow.points, 0.0))
    noise = jnp.asarray(np.random.default_rng(1).standard_normal(smooth.shape))
    solenoidal_noise = plan.projection.project(noise).candidate_velocity

    smooth_evidence = flow.classify_edge_velocity(smooth)
    noise_evidence = flow.classify_edge_velocity(solenoidal_noise)

    assert bool(smooth_evidence.physical)
    assert not bool(noise_evidence.physical)
    assert float(noise_evidence.nonphysical_fraction) > 3.0 * float(
        smooth_evidence.nonphysical_fraction
    )
    # A rough nodal field projects to graph-solenoidal cycles the nodes cannot
    # represent; the step is refused and the input is retained.
    rough = jnp.asarray(np.random.default_rng(2).standard_normal(flow.points.shape))
    refused = flow.initialize(rough)
    assert int(refused.evidence.status) == MeshfreeFlowStatus.SPURIOUS_CYCLES
    np.testing.assert_array_equal(refused.state.velocity, rough)


@pytest.mark.parametrize(
    ("step_size", "status"),
    [(50.0, MeshfreeFlowStatus.CFL_REFUSED), (-0.1, MeshfreeFlowStatus.INVALID_STEP)],
    ids=["cfl", "negative-step"],
)
def test_refused_step_holds_the_state(
    flow: PreparedMeshfreeIncompressibleFlow, step_size: float, status: int
) -> None:
    state = flow.initialize(_taylor_green(flow.points, 0.0)).state

    result = flow.advance(state, 0.0, step_size)

    assert int(result.evidence.status) == status
    assert not bool(result.successful)
    np.testing.assert_array_equal(result.state.velocity, state.velocity)
    np.testing.assert_array_equal(result.state.volume_flux, state.volume_flux)


def test_inadmissible_flow_declarations_are_refused(
    periodic_box: tuple[PreparedMeshfreeExteriorCalculus, PointCloudPlan],
) -> None:
    exterior, cloud = periodic_box
    prepared = cloud.prepare()
    with pytest.raises(ValueError, match="Betti"):
        MeshfreeIncompressibleFlowPlan(
            exterior, prepared, viscosity=0.1, domain_betti_number=1000
        )
    with pytest.raises(ValueError, match="viscosity"):
        MeshfreeIncompressibleFlowPlan(
            exterior, prepared, viscosity=-1.0, domain_betti_number=2
        )


@pytest.mark.parametrize("density_model", ("constant", "variable"))
def test_pressure_error_policy_is_refused_before_transactional_flow(
    periodic_box: tuple[PreparedMeshfreeExteriorCalculus, PointCloudPlan],
    density_model: IncompressibleDensityModel,
) -> None:
    exterior, cloud = periodic_box
    policy = LinearSolvePolicy(ProjectedPCG(), failure=FailurePolicy("error"))
    with pytest.raises(ValueError, match="pressure_policy.*status-returning"):
        MeshfreeIncompressibleFlowPlan(
            exterior,
            cloud.prepare(),
            viscosity=_VISCOSITY,
            domain_betti_number=2,
            density_model=density_model,
            pressure_policy=policy,
        )


@pytest.mark.parametrize("density_model", ("constant", "variable"))
def test_pressure_status_policy_preserves_native_projection_consumer(
    periodic_box: tuple[PreparedMeshfreeExteriorCalculus, PointCloudPlan],
    density_model: IncompressibleDensityModel,
) -> None:
    exterior, cloud = periodic_box
    policy = LinearSolvePolicy(ProjectedPCG(), failure=FailurePolicy("status"))
    prepared = MeshfreeIncompressibleFlowPlan(
        exterior,
        cloud.prepare(),
        viscosity=_VISCOSITY,
        domain_betti_number=2,
        density_model=density_model,
        pressure_policy=policy,
        transport_scheme="reconstructed",
        cycle_tolerance=0.15,
    ).prepare()
    velocity = _taylor_green(prepared.points, 0.0)
    density = (
        jnp.ones(prepared.points.shape[:1], dtype=jnp.float64)
        if density_model == "variable"
        else None
    )
    result = prepared.initialize(velocity, density=density)
    assert int(result.evidence.status) == MeshfreeFlowStatus.ACCEPTED
    np.testing.assert_allclose(result.state.velocity, velocity, atol=1e-10)


def _square(side: int) -> PreparedPointCloudDiscretization:
    grid = np.meshgrid(*(np.linspace(0.0, 1.0, side),) * 2, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    boundary = np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    rng = np.random.default_rng(7)
    points[~boundary] += rng.uniform(-0.15, 0.15, (np.count_nonzero(~boundary), 2)) / (
        side - 1
    )
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    return PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()


def _stokes_velocity(x: Array) -> Array:
    return jnp.stack(
        (
            jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
    )


def _stokes_pressure(x: Array) -> Array:
    return jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1])


def _stokes_errors(side: int) -> tuple[float, float, float, float]:
    # Independent manufactured solution with div u = 0 and mu = 1, so
    # -div(2 mu eps(u)) = -lap u and f = -lap u + grad p.
    cloud = _square(side)
    points = cloud.points
    exact = jax.vmap(_stokes_velocity)(points)
    pressure = jax.vmap(_stokes_pressure)(points)

    def force(x: Array) -> Array:
        laplacian = jnp.trace(
            jax.jacfwd(jax.jacfwd(_stokes_velocity))(x), axis1=1, axis2=2
        )
        return -laplacian + jax.grad(_stokes_pressure)(x)

    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    boundary = PointBoundaryPlan(
        tuple(
            PointBoundaryCondition(
                "dirichlet", rows, exact[rows, a], label=f"wall-{a}", component=a
            )
            for a in range(2)
        ),
        row_count=points.shape[0],
        components=2,
    )
    prepared = MeshfreeGeneralizedStokesPlan(cloud, boundary, shear_modulus=1.0).prepare()

    result = prepared.solve(jax.vmap(force)(points))

    assert int(result.status) == MechanicsStatus.ACCEPTED
    assert prepared.plan.gauge_row is not None
    weights = cloud.quadrature_weights
    # The plan reports the mean-zero pressure representative.
    shifted = pressure - jnp.sum(weights * pressure) / jnp.sum(weights)
    gradient = jax.vmap(jax.jacfwd(_stokes_velocity))(points)
    stress = (gradient + jnp.swapaxes(gradient, 1, 2)) - shifted[:, None, None] * jnp.eye(
        2
    )
    traction = jnp.sum(stress * cloud.plan.boundary_normals[:, None, :], axis=2)
    return (
        float(jnp.sqrt(jnp.sum(weights[:, None] * (result.field - exact) ** 2))),
        float(jnp.sqrt(jnp.sum(weights * (result.pressure - shifted) ** 2))),
        float(jnp.max(jnp.abs((result.boundary_traction - traction)[rows]))),
        float(result.pressure_oscillation),
    )


def test_bounded_stokes_velocity_pressure_and_traction_refine() -> None:
    coarse = _stokes_errors(9)
    fine = _stokes_errors(13)

    velocity, pressure, traction, oscillation = zip(coarse, fine, strict=True)
    assert velocity[1] < 0.5 * velocity[0] and velocity[1] < 5e-3
    assert pressure[1] < 0.5 * pressure[0]
    assert traction[1] < traction[0]
    # The consistent stabilization suppresses checkerboard pressure modes.
    assert oscillation[1] < oscillation[0] < 1.0
