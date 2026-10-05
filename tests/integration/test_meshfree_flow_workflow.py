#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Meshfree physical flow workflows on public APIs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from phydrax import discretization as d
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MaterialParticleMeasure,
    MeshfreeEdgeRelationPlan,
    MeshfreeExteriorCalculusPlan,
    MeshfreeMetricPolicy,
    minimum_image_edge_charts,
    PointTransferStatus,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver import (
    FixedStepProblem,
    FixedStepSolution,
    MeshfreeFlowEvidence,
    MeshfreeFlowStatus,
    MeshfreeIncompressibleFlowPlan,
    MeshfreeIncompressibleState,
    MeshfreeLagrangianFlowPlan,
    MeshfreeLagrangianStatus,
    MeshfreeMeasureTransferPlan,
    MeshfreeSPHReconstruction,
    PreparedMeshfreeIncompressibleFlow,
    solve_fixed_step,
)


_WAVE = 2.0 * np.pi


def _torus_address(length: float, maximum_depth: int) -> MortonAddressPlan:
    """The derived address of the square ``[0, length]^2`` with both seams identified."""
    box = HyperRectangle(np.zeros(2), np.full(2, length))
    return MortonAddressPlan.from_periodic_identifications(
        tuple(PeriodicIdentification(box, "x", component=axis) for axis in range(2)),
        maximum_depth=maximum_depth,
    )


_PERIODIC = _torus_address(1.0, 10)


def _lattice(side: int) -> NDArray[np.float64]:
    axis = (np.arange(side, dtype=np.float64) + 0.5) / side
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((x.ravel(), y.ravel()), axis=1)


def _taylor_green(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = points[:, 0], points[:, 1]
    return np.stack(
        (
            np.sin(_WAVE * x) * np.cos(_WAVE * y),
            -np.cos(_WAVE * x) * np.sin(_WAVE * y),
        ),
        axis=1,
    )


def test_lagrangian_taylor_green_particles_with_cloud_round_trip() -> None:
    side = 12
    count = side * side
    spacing = 1.0 / side
    points = _lattice(side)
    particles = d.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), 1.0 / count), ambient_dimension=2
    ).prepare()
    box = d.ParticleBox.from_address(_PERIODIC)
    verlet = d.VerletParticleNeighborhoodPlan(
        d.DenseParticleNeighborhoodPlan(count * (count - 1) // 2, box=box),
        2.6 * spacing,
        0.4 * spacing,
    ).prepare(particles)
    sph = MeshfreeSPHReconstruction(
        particles, verlet, d.WendlandC2SPHKernel(2), 1.3 * spacing
    )
    material = MaterialParticleMeasure(
        particles,
        d.ParticlePopulationPlan(particles).initialize(),
        1.0,
        neighbors=verlet,
        reference=verlet.initialize(points),
    )
    flow = MeshfreeLagrangianFlowPlan(
        "material-mass",
        reference_density=1.0,
        address=_PERIODIC,
        neighbors=21,  # complete lattice distance shells
        material=material,
        sph=sph,
    ).prepare(points)
    state = flow.initialize(_taylor_green(points))
    energies = [
        0.5 * np.sum(np.asarray(state.masses)[:, None] * _taylor_green(points) ** 2)
    ]
    for _ in range(3):
        result = flow.step(state, 0.01)
        evidence = result.evidence
        assert int(evidence.status) == MeshfreeLagrangianStatus.ACCEPTED
        assert float(evidence.mass_after) == float(evidence.mass_before)
        assert float(evidence.divergence_after) <= 1e-8 * max(
            float(evidence.divergence_before), 1.0
        )
        assert float(evidence.strong_divergence_after) < 0.1
        assert float(evidence.kinetic_energy_after) <= float(
            evidence.kinetic_energy_predicted
        )
        energies.append(float(evidence.kinetic_energy_after))
        flow, state = result.flow, result.state
    # Inviscid Taylor-Green keeps its energy; the M-orthogonal projection may
    # only dissipate, and slowly at this resolution.
    assert np.all(np.diff(energies) <= 1e-14)
    assert energies[-1] > 0.99 * energies[0]

    grid = _lattice(8)
    grid_volumes = np.full(64, 1.0 / 64)
    forward = MeshfreeMeasureTransferPlan(
        state.positions,
        state.volumes,
        grid,
        grid_volumes,
        source_measure="material-mass",
        target_measure="quadrature-volume",
        address=_PERIODIC,
    ).prepare()
    assert forward.status is PointTransferStatus.ADMITTED
    on_grid = forward.apply(state.masses, state.velocity)
    masses = np.asarray(state.masses)
    momentum = np.sum(masses[:, None] * np.asarray(state.velocity), axis=0)
    assert abs(np.sum(np.asarray(on_grid.masses)) - np.sum(masses)) < 1e-13
    np.testing.assert_allclose(
        np.sum(
            np.asarray(on_grid.masses)[:, None] * np.asarray(on_grid.velocity), axis=0
        ),
        momentum,
        atol=1e-13,
    )
    # The Eulerian Taylor-Green field is steady: the grid reconstruction is
    # compared with the analytic field at the grid points.
    reference = _taylor_green(grid)
    error = np.linalg.norm(np.asarray(on_grid.velocity) - reference)
    assert error < 0.15 * np.linalg.norm(reference)
    assert float(on_grid.density_mismatch) < 0.1

    backward = MeshfreeMeasureTransferPlan(
        grid,
        grid_volumes,
        state.positions,
        state.volumes,
        source_measure="quadrature-volume",
        target_measure="material-mass",
        address=_PERIODIC,
    ).prepare()
    assert backward.status is PointTransferStatus.ADMITTED
    returned = backward.apply(on_grid.masses, on_grid.velocity)
    assert abs(np.sum(np.asarray(returned.masses)) - np.sum(masses)) < 1e-13
    np.testing.assert_allclose(
        np.sum(
            np.asarray(returned.masses)[:, None] * np.asarray(returned.velocity), axis=0
        ),
        momentum,
        atol=1e-13,
    )


_TORUS = 2.0 * np.pi
_VISCOSITY = 0.05


@pytest.fixture(scope="module")
def torus() -> tuple[
    PreparedMeshfreeExteriorCalculus, d.PreparedPointCloudDiscretization
]:
    """12 x 12 periodic lattice: a closed radius graph over a box with b_1 = 2."""
    count = 12
    spacing = _TORUS / count
    points = _TORUS * _lattice(count)
    address = _torus_address(_TORUS, 12)
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
    cloud = d.PointCloudPlan(
        np.asarray(exterior.points),
        np.asarray(exterior.node_volumes),
        address=address,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()
    return exterior, cloud


def _decaying_vortex(points: Array, time: float) -> Array:
    """Viscous Taylor-Green on [0, 2 pi]^2: ``u = e^{-2 nu t} (sin x cos y, -cos x sin y)``."""
    x, y = points[:, 0], points[:, 1]
    decay = np.exp(-2.0 * _VISCOSITY * time)
    return decay * jnp.stack((jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)), axis=1)


def _rollout(
    flow: PreparedMeshfreeIncompressibleFlow,
    state: MeshfreeIncompressibleState,
    t0: float,
    t1: float,
    step_size: float,
) -> FixedStepSolution:
    problem = FixedStepProblem(
        flow,
        state,
        t0=t0,
        t1=t1,
        step_size=step_size,
        state_geometry=EuclideanStateGeometry(),
    )
    return solve_fixed_step(problem, evidence_retention="steps")


def _final(solution: FixedStepSolution) -> MeshfreeIncompressibleState:
    return jax.tree.map(lambda leaf: leaf[-1], solution.states)


def _ledger(
    solution: FixedStepSolution,
) -> tuple[MeshfreeFlowEvidence, Array, MeshfreeFlowEvidence, Array]:
    """Per-step evidence, committed mask, refused-attempt record, and its step."""
    record = solution.evidence
    assert record is not None and record.step_committed is not None
    assert isinstance(record.steps, MeshfreeFlowEvidence)
    assert isinstance(record.refused, MeshfreeFlowEvidence)
    return record.steps, record.step_committed, record.refused, record.refused_step


def test_periodic_taylor_green_ledger_survives_a_refused_step(
    torus: tuple[PreparedMeshfreeExteriorCalculus, d.PreparedPointCloudDiscretization],
) -> None:
    exterior, cloud = torus
    flow = MeshfreeIncompressibleFlowPlan(
        exterior,
        cloud,
        viscosity=_VISCOSITY,
        domain_betti_number=2,
        transport_scheme="reconstructed",
        cycle_tolerance=0.15,
    ).prepare()
    points = flow.points
    exact = _decaying_vortex(points, 0.0)
    # 0.1 grad sin(x + y) is a pure gradient the projection must remove.
    gradient = 0.1 * jnp.cos(points[:, 0] + points[:, 1])[:, None] * jnp.ones((1, 2))
    initial = flow.initialize(exact + gradient)
    assert int(initial.evidence.status) == MeshfreeFlowStatus.ACCEPTED
    assert float(initial.evidence.nodal_divergence_after) < 0.1 * float(
        initial.evidence.nodal_divergence_before
    )
    assert float(jnp.max(jnp.abs(initial.state.velocity - exact))) < 1e-2

    first = _rollout(flow, initial.state, 0.0, 0.5, 0.1)
    before, committed, _, _ = _ledger(first)
    assert bool(jnp.all(committed))
    held = _final(first)

    # A step 500 times the stable size violates the CFL bound: the native
    # rollout refuses it, holds the committed state, and publishes why.
    refused = _rollout(flow, held, 0.5, 50.5, 50.0)
    _, committed, refusal, refused_step = _ledger(refused)
    assert not bool(jnp.any(committed))
    assert int(refused_step) == 0
    assert int(refusal.status) == MeshfreeFlowStatus.CFL_REFUSED
    assert float(refusal.cfl) > flow.plan.cfl_limit
    np.testing.assert_array_equal(_final(refused).velocity, held.velocity)
    np.testing.assert_array_equal(_final(refused).volume_flux, held.volume_flux)

    resumed = _rollout(flow, _final(refused), 0.5, 1.0, 0.1)
    after, committed, _, _ = _ledger(resumed)
    assert bool(jnp.all(committed))
    ledgers = (before, after)
    # The ledger is continuous across the refusal: nothing was committed.
    np.testing.assert_array_equal(
        ledgers[1].kinetic_energy_before[0], ledgers[0].kinetic_energy_after[-1]
    )
    energy = np.concatenate(
        [np.asarray(ledger.kinetic_energy_after) for ledger in ledgers]
    )
    energy = energy / float(ledgers[0].kinetic_energy_before[0])
    assert np.all(np.diff(energy) < 0.0)
    # Exact viscous decay e^{-4 nu t}; the coarse lattice (h = pi/6) adds
    # transport dissipation within 10 %.
    np.testing.assert_allclose(
        energy, np.exp(-4.0 * _VISCOSITY * 0.1 * np.arange(1, 11)), rtol=0.1
    )
    for ledger in ledgers:
        assert float(jnp.max(ledger.graph_divergence_norm)) < 1e-12
        assert float(jnp.max(ledger.nodal_divergence_after)) < 0.05
        assert bool(jnp.all(ledger.cycles.physical))
        np.testing.assert_allclose(ledger.momentum_after, 0.0, atol=1e-12)
    error = jnp.max(jnp.abs(_final(resumed).velocity - _decaying_vortex(points, 1.0)))
    assert float(error) < 0.05


def test_variable_density_vortex_conserves_mass_and_momentum(
    torus: tuple[PreparedMeshfreeExteriorCalculus, d.PreparedPointCloudDiscretization],
) -> None:
    exterior, cloud = torus
    flow = MeshfreeIncompressibleFlowPlan(
        exterior,
        cloud,
        viscosity=_VISCOSITY,
        domain_betti_number=2,
        density_model="variable",
        cycle_tolerance=0.15,
    ).prepare()
    points = flow.points
    # Heavy core at the box center: the lattice, density, and vortex are
    # point-symmetric about (pi, pi), so the initial momentum is exactly zero.
    density = 1.0 + 0.5 * jnp.exp(
        -((points[:, 0] - np.pi) ** 2) - (points[:, 1] - np.pi) ** 2
    )
    initial = flow.initialize(_decaying_vortex(points, 0.0), density=density)
    assert int(initial.evidence.status) == MeshfreeFlowStatus.ACCEPTED
    mass = float(jnp.sum(exterior.node_volumes * density))

    solution = _rollout(flow, initial.state, 0.0, 0.4, 0.1)

    ledger, committed, _, _ = _ledger(solution)
    assert bool(jnp.all(committed))
    np.testing.assert_allclose(ledger.mass_before, mass, rtol=1e-13)
    np.testing.assert_allclose(ledger.mass_after, mass, rtol=1e-13)
    np.testing.assert_allclose(ledger.momentum_after, 0.0, atol=1e-10)
    assert float(jnp.max(ledger.graph_divergence_norm)) < 1e-11
    final = _final(solution)
    # Limited transport keeps the density inside its initial bounds.
    assert float(jnp.min(final.density)) >= 1.0 - 1e-12
    assert float(jnp.max(final.density)) <= float(jnp.max(density)) + 1e-12
    assert float(jnp.max(jnp.abs(final.density - density))) > 0.0
    # Viscosity dissipates; no step creates kinetic energy overall.
    energy = np.asarray(ledger.kinetic_energy_after)
    assert 0.8 * float(ledger.kinetic_energy_before[0]) < energy[-1]
    assert energy[-1] < float(ledger.kinetic_energy_before[0])
