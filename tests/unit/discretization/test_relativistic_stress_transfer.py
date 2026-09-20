from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax._physical import RelativityScaleContract
from phydrax.applications.cosmology._scales import CODE_COSMOLOGY_SCALE
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticParticleState,
    RelativisticStressDepositPlan,
)
from phydrax.discretization.splatting import ParticleGridSplatPlan
from phydrax.metrix import (
    ADMGridGeometry,
    RelativityConvention,
    StressEnergyProjection,
)


def _case(*, count=4, particle_count=1, momenta=None, weights=None):
    convention = RelativityConvention.canonical()
    scale = RelativityScaleContract(CODE_COSMOLOGY_SCALE, 1, 1, 1, 1)
    units = RelativisticUnitContract(scale, convention)
    spectral = TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="stress",
    ).prepare(tuple(AxisDomain.periodic(0.0, 1.0) for _ in range(3)))
    positions = (
        jnp.asarray([[0.125, 0.25, 0.375]])
        if particle_count == 1
        else jnp.stack(
            (
                jnp.linspace(0.11, 0.79, particle_count),
                jnp.linspace(0.19, 0.71, particle_count),
                jnp.linspace(0.23, 0.83, particle_count),
            ),
            axis=-1,
        )
    )
    particles = ParticleSetPlan(
        jnp.arange(100, 100 + particle_count),
        jnp.ones((particle_count,)),
        ambient_dimension=3,
    ).prepare()
    transfer = ParticleGridSplatPlan(spectral.grid).prepare(particles)
    plan = RelativisticStressDepositPlan(
        transfer,
        units,
        jnp.asarray([7], dtype=jnp.int32),
        jnp.asarray([2.0]),
    )
    shape = spectral.physical_shape
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    geometry = ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype="bool"),
        jnp.ones(shape, dtype="bool"),
        snapshot_token=jnp.asarray(11, dtype=jnp.int32),
        chart_id="periodic-cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id=spectral.grid.topology.topology_id,
        geometry_lineage_id="weak-field-test-lineage",
    )
    xyz = spectral.grid.points.reshape(shape + (3,))
    coordinates = jnp.concatenate((jnp.zeros(shape + (1,)), xyz), axis=-1)
    frame = LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        coordinates,
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="eulerian-observer",
        orientation_id="cartesian-right-handed",
    )
    covariant = (
        jnp.asarray([[3.0, 4.0, 0.0]]) if momenta is None else jnp.asarray(momenta)
    )
    multiplicity = (
        jnp.ones((particle_count,)) if weights is None else jnp.asarray(weights)
    )
    state = plan.initialize(
        multiplicity,
        positions,
        covariant,
        jnp.full((particle_count,), 7, dtype=jnp.int32),
        frame,
    )
    return plan, frame, state, spectral


def test_boosted_single_particle_has_analytic_stress_energy_and_units():
    plan, frame, state, _ = _case(weights=jnp.asarray([2.5]))
    result = plan.deposit(state, frame)

    energy = np.sqrt(2.0**2 + 3.0**2 + 4.0**2)
    expected_momentum = 2.5 * np.asarray([3.0, 4.0, 0.0])
    expected_stress = (
        2.5 * np.outer(np.asarray([3.0, 4.0, 0.0]), np.asarray([3.0, 4.0, 0.0])) / energy
    )
    assert isinstance(result.projection, StressEnergyProjection)
    assert bool(result.successful)
    assert bool(result.projection.compatible_with(frame.geometry))
    np.testing.assert_allclose(result.source_integrals.energy, 2.5 * energy, rtol=2e-6)
    np.testing.assert_allclose(
        result.source_integrals.momentum_covector, expected_momentum, rtol=2e-6
    )
    np.testing.assert_allclose(
        result.source_integrals.stress_covariant, expected_stress, rtol=2e-6
    )
    np.testing.assert_allclose(result.mass_shell_defect, 0.0, atol=2e-5)
    np.testing.assert_allclose(
        result.projection.stress_covariant,
        jnp.swapaxes(result.projection.stress_covariant, -1, -2),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        jnp.trace(result.anisotropic_stress, axis1=-2, axis2=-1), 0.0, atol=2e-6
    )
    assert result.energy_density_unit_id == result.momentum_density_unit_id
    assert result.volume_measure_id == plan.volume_measure_id


def test_proper_measure_and_anisotropic_trace_use_the_spatial_metric():
    plan, _, original, spectral = _case(momenta=jnp.asarray([[2.0, 0.0, 0.0]]))
    shape = spectral.physical_shape
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    geometry = ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        4.0 * identity,
        0.25 * identity,
        jnp.full(shape, 8.0),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype="bool"),
        jnp.ones(shape, dtype="bool"),
        snapshot_token=jnp.asarray(12, dtype=jnp.int32),
        chart_id="periodic-cartesian",
        convention_id=plan.units.convention.convention_id,
        scale_id=plan.units.scale.scale_id,
        topology_id=plan.topology_id,
        geometry_lineage_id="weak-field-test-lineage",
    )
    coordinates = jnp.concatenate(
        (
            jnp.zeros(shape + (1,)),
            spectral.grid.points.reshape(shape + (3,)),
        ),
        axis=-1,
    )
    frame = LocalRelativisticFramePlan.from_adm(
        geometry,
        plan.units,
        coordinates,
        jnp.asarray(0.0),
        jnp.asarray(2.0),
        observer_id="eulerian-observer",
        orientation_id="cartesian-right-handed",
    )
    state = plan.initialize(
        jnp.asarray([1.0]),
        original.positions,
        jnp.asarray([[2.0, 0.0, 0.0]]),
        jnp.asarray([7], dtype=jnp.int32),
        frame,
    )
    result = plan.deposit(state, frame)
    anisotropic_trace = jnp.sum(
        geometry.inverse_spatial_metric * result.anisotropic_stress,
        axis=(-2, -1),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(state.local_momenta, [[1.0, 0.0, 0.0]], atol=2e-6)
    np.testing.assert_allclose(
        result.source_integrals.stress_covariant[0, 0],
        4.0 / np.sqrt(5.0),
        rtol=2e-6,
    )
    np.testing.assert_allclose(anisotropic_trace, 0.0, atol=2e-6)


def test_zero_energy_massless_slot_is_rejected_without_nonfinite_stress():
    base, frame, _, _ = _case(momenta=jnp.zeros((1, 3)))
    plan = RelativisticStressDepositPlan(
        base.transfer,
        base.units,
        jnp.asarray([7], dtype=jnp.int32),
        jnp.asarray([0.0]),
    )
    state = plan.initialize(
        jnp.asarray([1.0]),
        jnp.asarray([[0.125, 0.25, 0.375]]),
        jnp.zeros((1, 3)),
        jnp.asarray([7], dtype=jnp.int32),
        frame,
    )
    result = plan.deposit(state, frame)

    assert not bool(result.mass_shell_valid)
    assert not bool(result.successful)
    assert bool(result.finite)
    np.testing.assert_allclose(result.projection.energy_density, 0.0, atol=0.0)


def test_matched_deposit_gather_is_adjoint_and_state_identity_is_stable():
    momenta = jnp.asarray([[0.3, -0.2, 0.1], [0.5, 0.4, -0.1], [-0.2, 0.3, 0.6]])
    plan, frame, state, spectral = _case(
        particle_count=3,
        momenta=momenta,
        weights=jnp.asarray([1.0, 2.0, 0.5]),
    )
    particle_values = jnp.asarray([[1.0, -2.0], [0.5, 0.75], [-1.25, 0.2]])
    x, y, z = spectral.grid.primary_entity_layout.coordinates_by_axis
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    grid_values = jnp.stack((xx + 2.0 * yy, yy - zz), axis=-1)
    evidence = plan.adjoint_evidence(state, frame, particle_values, grid_values)
    gathered = plan.gather(state, frame, grid_values)

    assert bool(evidence.successful)
    assert bool(gathered.successful)
    np.testing.assert_allclose(
        evidence.particle_pairing, evidence.grid_pairing, rtol=2e-6, atol=2e-6
    )
    assert evidence.relative_defect < 2e-6
    assert np.array_equal(np.asarray(state.particle_ids), np.asarray([100, 101, 102]))
    assert np.array_equal(np.asarray(state.incarnations), np.zeros(3, dtype=np.int32))
    assert np.array_equal(np.asarray(state.lineage_ids), np.asarray([100, 101, 102]))
    assert state.frame_token.shape == ()
    assert state.frame_id == frame.frame_id


def test_inactive_capacity_is_masked_without_changing_stable_slots():
    plan, frame, state, _ = _case(particle_count=2, momenta=jnp.zeros((2, 3)))
    masked = RelativisticParticleState(
        state.particle_ids,
        jnp.asarray([1.0, 0.0]),
        state.positions,
        state.local_momenta,
        state.covariant_momenta,
        state.species_ids,
        jnp.asarray([True, False]),
        state.incarnations,
        state.lineage_ids,
        state.time,
        state.scale_factor,
        state.frame_token,
        frame_id=state.frame_id,
        topology_id=state.topology_id,
        frame_lineage_id=state.frame_lineage_id,
    )
    result = plan.deposit(masked, frame)

    assert bool(result.successful)
    np.testing.assert_allclose(result.source_integrals.energy, 2.0, rtol=2e-6)
    assert np.array_equal(np.asarray(masked.particle_ids), np.asarray([100, 101]))
