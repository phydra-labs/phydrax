#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Lagrangian GMLS particle flow, measure transfer and SPH interoperability.

References are independent of the implementation: analytic Taylor–Green and
gradient fields, NumPy ledgers, and a NumPy all-pairs Wendland C2 summation.
Symmetric lattices use 21 neighbors (complete distance shells) so GMLS
stencils keep the lattice symmetry.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax import discretization as d
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MaterialParticleMeasure,
    PointTransferStatus,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.solver import (
    MeshfreeLagrangianFlowPlan,
    MeshfreeLagrangianStatus,
    MeshfreeMeasureTransferPlan,
    MeshfreeSPHReconstruction,
    PreparedMeshfreeLagrangianFlow,
)


_SIDE = 12
_COUNT = _SIDE * _SIDE
_WAVE = 2.0 * np.pi
_TORUS = HyperRectangle(np.zeros(2), np.ones(2))
_PERIODIC = MortonAddressPlan.from_periodic_identifications(
    tuple(PeriodicIdentification(_TORUS, "x", component=axis) for axis in range(2)),
    maximum_depth=10,
)
_SHELLS = 21


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


def _gradient_perturbation(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gradient of ``phi = 0.1 cos(2 pi x) cos(2 pi y)``."""
    x, y = points[:, 0], points[:, 1]
    return (
        -0.1
        * _WAVE
        * np.stack(
            (
                np.sin(_WAVE * x) * np.cos(_WAVE * y),
                np.cos(_WAVE * x) * np.sin(_WAVE * y),
            ),
            axis=1,
        )
    )


def _volume_flow(
    points: NDArray[np.float64],
    *,
    neighbors: int | None = _SHELLS,
    viscosity: float = 0.0,
    body_acceleration: NDArray[np.float64] | None = None,
) -> PreparedMeshfreeLagrangianFlow:
    plan = MeshfreeLagrangianFlowPlan(
        "quadrature-volume",
        reference_density=1.0,
        address=_PERIODIC,
        neighbors=neighbors,
        viscosity=viscosity,
        body_acceleration=body_acceleration,
    )
    return plan.prepare(points, volumes=np.full(points.shape[0], 1.0 / points.shape[0]))


def _sph(
    points: NDArray[np.float64], box: d.ParticleBox | None = None
) -> MeshfreeSPHReconstruction:
    count = points.shape[0]
    spacing = 1.0 / _SIDE
    particles = d.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), 1.0 / count), ambient_dimension=2
    ).prepare()
    verlet = d.VerletParticleNeighborhoodPlan(
        d.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2,
            box=d.ParticleBox.from_address(_PERIODIC) if box is None else box,
        ),
        2.6 * spacing,
        0.4 * spacing,
    ).prepare(particles)
    return MeshfreeSPHReconstruction(
        particles, verlet, d.WendlandC2SPHKernel(2), 1.3 * spacing
    )


def _material_flow(points: NDArray[np.float64]) -> PreparedMeshfreeLagrangianFlow:
    sph = _sph(points)
    population = d.ParticlePopulationPlan(sph.particles).initialize()
    material = MaterialParticleMeasure(
        sph.particles,
        population,
        1.0,
        neighbors=sph.neighbors,
        reference=sph.neighbors.initialize(points),
    )
    plan = MeshfreeLagrangianFlowPlan(
        "material-mass",
        reference_density=1.0,
        address=_PERIODIC,
        neighbors=_SHELLS,
        material=material,
        sph=sph,
    )
    return plan.prepare(points)


def _wendland_density(
    points: NDArray[np.float64], masses: NDArray[np.float64], h: float
) -> NDArray[np.float64]:
    """All-pairs periodic Wendland C2 summation density (2-D)."""
    offset = points[:, None, :] - points[None, :, :]
    offset -= np.round(offset)
    q = np.linalg.norm(offset, axis=-1) / h
    profile = np.where(q < 2.0, (1.0 - 0.5 * q) ** 4 * (2.0 * q + 1.0), 0.0)
    return (7.0 / (4.0 * np.pi * h * h)) * profile @ masses


def test_material_mass_steps_conserve_mass_exactly() -> None:
    points = _lattice(_SIDE)
    flow = _material_flow(points)
    state = flow.initialize(_taylor_green(points))
    masses = np.asarray(state.masses)
    for _ in range(2):
        result = flow.step(state, 0.01)
        evidence = result.evidence
        assert int(evidence.status) == MeshfreeLagrangianStatus.ACCEPTED
        assert float(evidence.mass_after) == float(evidence.mass_before)
        assert float(evidence.mass_after) == pytest.approx(1.0, abs=1e-14)
        np.testing.assert_array_equal(np.asarray(result.state.masses), masses)
        # Material-mass volumes follow the SPH summation density exactly.
        assert evidence.sph_density_mismatch is not None
        assert float(evidence.sph_density_mismatch) < 1e-12
        flow, state = result.flow, result.state
    assert float(state.time) == pytest.approx(0.02, abs=1e-15)


def test_measure_transfer_conserves_mass_and_momentum_and_refuses_uncovered() -> None:
    rng = np.random.default_rng(7)
    source = _lattice(8) + rng.uniform(-0.01, 0.01, (64, 2))
    source_volumes = np.full(64, 1.0 / 64)
    density = 1.0 + 0.2 * np.sin(_WAVE * source[:, 0])
    masses = density * source_volumes
    velocity = _taylor_green(source)
    target = _lattice(6)
    target_volumes = np.full(36, 1.0 / 36)
    prepared = MeshfreeMeasureTransferPlan(
        source,
        source_volumes,
        target,
        target_volumes,
        source_measure="material-mass",
        target_measure="quadrature-volume",
        address=_PERIODIC,
    ).prepare()
    assert prepared.status is PointTransferStatus.ADMITTED
    result = prepared.apply(masses, velocity)
    moved = np.asarray(result.masses)
    momentum = moved[:, None] * np.asarray(result.velocity)
    scale = np.sum(masses)
    assert bool(result.positive)
    assert abs(np.sum(moved) - np.sum(masses)) <= 1e-13 * scale
    np.testing.assert_allclose(
        np.sum(momentum, axis=0), np.sum(masses[:, None] * velocity, axis=0), atol=1e-13
    )
    np.testing.assert_allclose(np.asarray(result.mass_defect), 0.0, atol=1e-13)
    reference_density = np.sum(masses) / np.sum(source_volumes)
    mismatch = np.max(np.abs(moved / target_volumes / reference_density - 1.0))
    assert float(result.density_mismatch) == pytest.approx(mismatch, rel=1e-12)
    assert mismatch > 1e-2  # the varying source density is reported, not hidden

    left_half = _lattice(6) * np.asarray([0.5, 1.0])
    refused = MeshfreeMeasureTransferPlan(
        source,
        source_volumes,
        left_half,
        np.full(36, 0.5 / 36),
        source_measure="material-mass",
        target_measure="quadrature-volume",
        address=_PERIODIC,
    ).prepare()
    assert refused.status is PointTransferStatus.UNCOVERED_SOURCE
    assert not refused.admitted
    with pytest.raises(ValueError, match="UNCOVERED_SOURCE"):
        refused.apply(masses, velocity)


def test_projection_removes_gradient_perturbation() -> None:
    points = _lattice(_SIDE)
    flow = _volume_flow(points)
    perturbation = _gradient_perturbation(points)
    initial = _taylor_green(points) + perturbation
    state = flow.initialize(initial)
    result = flow.step(state, 0.01)
    evidence = result.evidence
    assert int(evidence.status) == MeshfreeLagrangianStatus.ACCEPTED
    assert float(evidence.divergence_after) < 1e-8 * float(evidence.divergence_before)
    assert float(evidence.strong_divergence_after) < 1e-8 * float(
        evidence.strong_divergence_before
    )
    removed = initial - np.asarray(result.candidate.velocity)
    error = np.linalg.norm(removed - perturbation) / np.linalg.norm(perturbation)
    assert error < 1e-6
    # The projection is M-orthogonal: it removes exactly the gradient energy.
    energy = 0.5 * np.sum(np.asarray(state.masses)[:, None] * _taylor_green(points) ** 2)
    assert float(evidence.kinetic_energy_after) == pytest.approx(energy, rel=1e-8)
    assert float(evidence.kinetic_energy_after) < float(evidence.kinetic_energy_predicted)


def test_projection_uses_current_volumes_without_reanchoring_support() -> None:
    points = _lattice(_SIDE)
    flow = _volume_flow(points)
    initial = _taylor_green(points) + _gradient_perturbation(points)
    state = flow.initialize(initial)
    volumes = state.volumes * jnp.asarray(
        1.0 + 0.4 * np.cos(_WAVE * points[:, 0]), dtype=jnp.float64
    )
    state = eqx.tree_at(
        lambda item: (item.volumes, item.masses), state, (volumes, volumes)
    )
    result = flow.step(state, 0.001)
    assert int(result.evidence.status) == MeshfreeLagrangianStatus.ACCEPTED
    assert not bool(result.evidence.reprepared)
    cloud = result.flow.cloud
    np.testing.assert_array_equal(cloud.quadrature_weights, flow.cloud.quadrature_weights)
    before = jnp.zeros((_COUNT,), dtype=jnp.float64)
    after = jnp.zeros((_COUNT,), dtype=jnp.float64)
    for axis in range(2):
        before = before + cloud.transpose_partial_derivative(
            volumes * state.velocity[:, axis], axis=axis
        )
        after = after + cloud.transpose_partial_derivative(
            volumes * result.candidate.velocity[:, axis], axis=axis
        )
    before_norm = jnp.sqrt(jnp.sum(before**2 / volumes) / jnp.sum(volumes))
    after_norm = jnp.sqrt(jnp.sum(after**2 / volumes) / jnp.sum(volumes))
    assert float(after_norm) < 1e-8 * float(before_norm)
    np.testing.assert_allclose(result.evidence.divergence_before, before_norm, rtol=1e-10)
    np.testing.assert_allclose(result.evidence.divergence_after, after_norm, atol=1e-10)


@pytest.mark.parametrize("side", [15, 16, 24], ids=["odd-15", "even-16", "even-24"])
def test_viscous_steps_are_projected_whatever_the_gradient_kernel(side: int) -> None:
    # Even periodic lattices add odd-even modes to ker G and odd ones near-null
    # modes; a gauged square pressure solve refused these steps.
    points = _lattice(side)
    flow = _volume_flow(points, viscosity=0.01)
    # The gradient perturbation gives the first step a divergence to remove.
    state = flow.initialize(_taylor_green(points) + _gradient_perturbation(points))
    mass = float(jnp.sum(state.masses))
    for _ in range(3):
        result = flow.step(state, 0.12 / side)
        evidence = result.evidence
        assert int(evidence.status) == MeshfreeLagrangianStatus.ACCEPTED
        assert float(evidence.divergence_after) < 1e-7 * float(evidence.divergence_before)
        assert float(evidence.mass_after) == pytest.approx(mass, rel=1e-14)
        assert float(evidence.kinetic_energy_after) <= float(
            evidence.kinetic_energy_predicted
        )
        state, flow = result.state, result.flow


def test_sph_and_gmls_reconstructions_agree_on_lattice() -> None:
    points = _lattice(_SIDE)
    sph = _sph(points)
    cache = sph.initialize(points)
    cloud = d.PointCloudPlan(
        points,
        np.full(_COUNT, 1.0 / _COUNT),
        stencil=LocalStencilPolicy(polynomial_degree=3),
        neighbors=_SHELLS,
        address=_PERIODIC,
    ).prepare()
    masses = np.full(_COUNT, 1.0 / _COUNT)
    x, y = points[:, 0], points[:, 1]
    velocity = 0.1 * np.stack((np.sin(_WAVE * x), np.sin(_WAVE * y)), axis=1)
    divergence = 0.1 * _WAVE * (np.cos(_WAVE * x) + np.cos(_WAVE * y))
    comparison = sph.compare(cloud, points, velocity, masses, masses, cache)
    assert bool(comparison.neighbors_successful)
    np.testing.assert_allclose(
        np.asarray(comparison.sph_density),
        _wendland_density(points, masses, sph.smoothing_length),
        rtol=1e-12,
    )
    # Declared lattice tolerances: Wendland C2 at h = 1.3 dx carries a ~1%
    # partition-of-unity defect; SPH divergence is first-order accurate.
    assert float(comparison.density_mismatch) < 2e-2
    norm = np.sqrt(np.mean(divergence**2))
    gmls_error = np.sqrt(
        np.mean((np.asarray(comparison.gmls_divergence) - divergence) ** 2)
    )
    sph_error = np.sqrt(
        np.mean((np.asarray(comparison.sph_divergence) - divergence) ** 2)
    )
    assert gmls_error < 1e-2 * norm
    assert sph_error < 0.15 * norm
    assert float(comparison.divergence_mismatch) < 0.15 * norm

    conversion = sph.convert(points, masses, masses, cache, target="material-mass")
    np.testing.assert_array_equal(np.asarray(conversion.masses), masses)
    assert abs(float(conversion.mass_defect)) < 1e-14
    rho = _wendland_density(points, masses, sph.smoothing_length)
    assert float(conversion.density_mismatch) == pytest.approx(
        np.max(np.abs(rho - 1.0)), rel=1e-10
    )


@pytest.mark.parametrize(
    ("step_size", "status"),
    [
        pytest.param(0.3, MeshfreeLagrangianStatus.COURANT_REFUSED, id="courant"),
        pytest.param(-0.01, MeshfreeLagrangianStatus.INVALID_STEP, id="negative-step"),
    ],
)
def test_refused_step_keeps_accepted_state(
    step_size: float, status: MeshfreeLagrangianStatus
) -> None:
    points = _lattice(_SIDE)
    flow = _volume_flow(points)
    state = flow.initialize(_taylor_green(points))
    refused = flow.step(state, step_size)
    assert int(refused.evidence.status) == status
    assert not bool(refused.successful)
    kept, source = refused.state, state
    for kept_field, source_field in (
        (kept.positions, source.positions),
        (kept.velocity, source.velocity),
        (kept.masses, source.masses),
        (kept.volumes, source.volumes),
        (kept.pressure, source.pressure),
        (kept.time, source.time),
    ):
        np.testing.assert_array_equal(np.asarray(kept_field), np.asarray(source_field))
    assert float(refused.candidate.time) == pytest.approx(step_size)
    # The native fixed-step record publishes the same refusal.
    assert not bool(refused.fixed_step.successful)
    continued = refused.flow.step(refused.state, 0.01)
    assert int(continued.evidence.status) == MeshfreeLagrangianStatus.ACCEPTED


def test_pressure_correction_momentum_change_is_reported() -> None:
    rng = np.random.default_rng(3)
    points = np.mod(_lattice(_SIDE) + rng.uniform(-0.1, 0.1, (_COUNT, 2)) / _SIDE, 1.0)
    gravity = np.asarray([0.5, 0.0])
    flow = _volume_flow(points, neighbors=None, viscosity=0.01, body_acceleration=gravity)
    state = flow.initialize(_taylor_green(points))
    result = flow.step(state, 0.01)
    evidence = result.evidence
    assert int(evidence.status) == MeshfreeLagrangianStatus.ACCEPTED
    masses = np.asarray(state.masses)
    change = np.sum(
        masses[:, None]
        * (np.asarray(result.candidate.velocity) - np.asarray(state.velocity)),
        axis=0,
    )
    np.testing.assert_allclose(
        np.asarray(evidence.momentum_after) - np.asarray(evidence.momentum_before),
        change,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(evidence.body_impulse), 0.01 * np.sum(masses) * gravity, rtol=1e-12
    )
    ledger = (
        np.asarray(evidence.body_impulse)
        + np.asarray(evidence.viscous_impulse)
        + np.asarray(evidence.pressure_impulse)
    )
    np.testing.assert_allclose(ledger, change, atol=1e-14)
    # An irregular collocated cloud does not conserve momentum under the
    # pressure correction; the impulse is published rather than claimed zero.
    assert np.max(np.abs(np.asarray(evidence.pressure_impulse))) > 1e-8


@pytest.mark.parametrize(
    "box",
    [
        pytest.param(
            d.ParticleBox(np.zeros(2), np.full(2, 2.0)), id="other-periodic-cell"
        ),
        pytest.param(
            d.ParticleBox(np.zeros(2), np.ones(2), periodic_axes=(True, False)),
            id="other-periodic-mask",
        ),
    ],
)
def test_sph_box_must_be_the_flow_address_cell(box: d.ParticleBox) -> None:
    sph = _sph(_lattice(_SIDE), box)
    with pytest.raises(ValueError, match="ParticleBox.from_address"):
        MeshfreeLagrangianFlowPlan(
            "quadrature-volume", reference_density=1.0, address=_PERIODIC, sph=sph
        )
    with pytest.raises(ValueError, match="matching periodic cloud address"):
        MeshfreeLagrangianFlowPlan("quadrature-volume", reference_density=1.0, sph=sph)
