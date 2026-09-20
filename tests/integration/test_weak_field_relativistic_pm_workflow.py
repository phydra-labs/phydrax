from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._physical import RelativityScaleContract
from phydrax.applications.cosmology._scales import CODE_COSMOLOGY_SCALE
from phydrax.applications.cosmology._weak_field_relativistic_pm import (
    WeakFieldRelativisticPMPlan,
    WeakFieldRelativisticPMPolicy,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticStressDepositPlan,
)
from phydrax.discretization.splatting import ParticleGridSplatPlan
from phydrax.metrix import (
    ADMGridGeometry,
    RelativityConvention,
    StressEnergyProjection,
)


def _frame(spectral, units, time, token):
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
        snapshot_token=jnp.asarray(token, dtype=jnp.int32),
        chart_id="periodic-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id=spectral.grid.topology.topology_id,
        geometry_lineage_id="compiled-weak-field-workflow",
    )
    xyz = spectral.grid.points.reshape(shape + (3,))
    coordinates = jnp.concatenate((jnp.full(shape + (1,), time), xyz), axis=-1)
    return LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        coordinates,
        jnp.asarray(time),
        jnp.asarray(1.0),
        observer_id="eulerian-observer",
        orientation_id="cartesian-right-handed",
    )


def test_compiled_relativistic_stress_metric_geodesic_endpoint_workflow():
    count = 3
    convention = RelativityConvention.canonical()
    scale = RelativityScaleContract(CODE_COSMOLOGY_SCALE, 1, 1, 1, 1)
    units = RelativisticUnitContract(scale, convention)
    spectral = TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="workflow-metric",
    ).prepare(tuple(AxisDomain.periodic(0.0, 1.0) for _ in range(3)))
    capacity = count**3
    particles = ParticleSetPlan(
        jnp.arange(capacity),
        jnp.ones((capacity,)),
        ambient_dimension=3,
    ).prepare()
    transfer = ParticleGridSplatPlan(spectral.grid).prepare(particles)
    stress = RelativisticStressDepositPlan(
        transfer,
        units,
        jnp.asarray([13], dtype=jnp.int32),
        jnp.asarray([1.0]),
        mass_shell_relative_tolerance=1e-5,
        conservation_tolerance=1e-5,
        frame_momentum_relative_tolerance=1e-5,
    )
    policy = WeakFieldRelativisticPMPolicy(
        maximum_scalar_metric_fraction=0.1,
        maximum_vector_metric_fraction=0.1,
        maximum_tensor_metric_norm=0.1,
        maximum_cell_crossing=0.75,
        constraint_relative_tolerance=5e-4,
        gauge_absolute_tolerance=5e-4,
        force_relative_tolerance=5e-4,
        conservation_relative_tolerance=5e-3,
    )
    plan = WeakFieldRelativisticPMPlan(
        stress,
        spectral,
        units,
        gravitational_constant=1e-10,
        policy=policy,
    )
    start = _frame(spectral, units, 0.0, 100)
    end = _frame(spectral, units, 0.01, 101)
    momentum = jnp.broadcast_to(jnp.asarray([0.4, 0.1, 0.0]), (capacity, 3))
    state = stress.initialize(
        jnp.ones((capacity,)),
        spectral.grid.points,
        momentum,
        jnp.full((capacity,), 13, dtype=jnp.int32),
        start,
    )

    compiled = jax.jit(lambda current: plan.step(current, start, end))
    result = compiled(state)
    jax.block_until_ready(result.state.positions)

    assert bool(result.successful)
    assert isinstance(result.initial.stress, StressEnergyProjection)
    assert result.initial.stress.projection_id == result.endpoint.stress.projection_id
    assert int(result.initial.stress.snapshot_token) == 100
    assert int(result.endpoint.stress.snapshot_token) == 101
    assert bool(result.endpoint.support_complete)
    assert bool(result.diagnostics.source_conserved)
    assert result.diagnostics.maximum_cell_crossing < policy.maximum_cell_crossing
    assert result.initial.source_integrals.energy > 0.0
    np.testing.assert_allclose(result.initial.net_force, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.endpoint.net_force, 0.0, atol=2e-7)
    assert not np.array_equal(
        np.asarray(result.state.frame_token), np.asarray(state.frame_token)
    )
    assert np.all((np.asarray(result.state.positions) >= 0.0))
    assert np.all((np.asarray(result.state.positions) < 1.0))
