#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import (
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
    SmoothCompressibleD2VStepStatus,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport


def _method():
    quadrature = d2v17_quadrature()
    return SmoothCompressibleD2VKineticMethod(
        quadrature,
        IdealGasMaterial(1.4, 1.0),
        ConstantTransport(0.03, 0.04),
    )


def _runtime():
    method = _method()
    energy_plan = PositiveEnergyEquilibriumPlan(method.quadrature)
    transport = D2V17PeriodicTransportPlan(
        method.quadrature,
        (5, 6),
        (0.01, 0.01),
        0.01,
    )
    return PreparedSmoothCompressibleD2V17SpatialDynamics(method, energy_plan, transport)


def _uniform_state(runtime):
    conserved = jnp.asarray((1.0, 0.03, -0.02, 1.251))
    density = conserved[0]
    momentum = conserved[1:3]
    velocity = momentum / density
    kinetic = 0.5 * jnp.sum(momentum * velocity)
    pressure = (1.4 - 1.0) * (conserved[-1] - kinetic)
    target_flux = (conserved[-1] + pressure) * velocity
    oracle = runtime.energy_plan.solve(conserved[-1], target_flux)
    equilibrium, evidence = runtime.method.equilibrium_from_energy_dual_with_evidence(
        conserved, oracle.dual, runtime.energy_plan
    )
    assert bool(evidence.successful)
    return SmoothCompressibleKineticState(
        jnp.broadcast_to(
            equilibrium.particle_populations,
            runtime.transport.spatial_shape + (17,),
        ),
        jnp.broadcast_to(
            equilibrium.total_energy_populations,
            runtime.transport.spatial_shape + (17,),
        ),
    )


def test_d2v17_periodic_pull_routes_every_integer_velocity():
    quadrature = d2v17_quadrature()
    transport = D2V17PeriodicTransportPlan(quadrature, (5, 6), (0.01, 0.01), 0.01)
    populations = jnp.arange(5 * 6 * 17, dtype=jnp.float64).reshape((5, 6, 17))

    routed = transport.transport(populations)

    for index, offset in enumerate(transport.pull_offsets):
        np.testing.assert_array_equal(
            routed[..., index],
            jnp.roll(populations[..., index], shift=offset, axis=(0, 1)),
        )
    assert transport.maximum_reach == (2, 2)


def test_periodic_spatial_oracle_step_preserves_uniform_state_and_content():
    runtime = _runtime()
    state = _uniform_state(runtime)

    result, oracle = runtime.step_oracle(state, jnp.asarray(0.01))

    assert bool(oracle.successful.all())
    assert bool(result.successful)
    assert int(result.evidence.status) == int(SmoothCompressibleD2VStepStatus.SUCCESS)
    np.testing.assert_allclose(
        result.accepted_state.particle_populations,
        state.particle_populations,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.accepted_state.total_energy_populations,
        state.total_energy_populations,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.evidence.conservation.total_residual,
        0.0,
        atol=2e-12,
    )


def test_periodic_spatial_failure_rolls_back_both_population_fields():
    runtime = _runtime()
    state = _uniform_state(runtime)
    invalid_particles = state.particle_populations.at[2, 3, 0].set(-1.0)
    invalid = SmoothCompressibleKineticState(
        invalid_particles, state.total_energy_populations
    )

    result, _ = runtime.step_oracle(invalid, jnp.asarray(0.01))

    assert not bool(result.successful)
    assert bool(result.rollback_applied)
    assert int(result.evidence.status) == int(
        SmoothCompressibleD2VStepStatus.INVALID_INPUT_STATE
    )
    np.testing.assert_array_equal(
        result.accepted_state.particle_populations, invalid.particle_populations
    )
    np.testing.assert_array_equal(
        result.accepted_state.total_energy_populations,
        invalid.total_energy_populations,
    )


def test_periodic_spatial_refuses_wrong_step_and_non_d2v17_quadrature():
    runtime = _runtime()
    state = _uniform_state(runtime)

    result, _ = runtime.step_oracle(state, jnp.asarray(0.005))

    assert not bool(result.successful)
    assert int(result.evidence.status) == int(
        SmoothCompressibleD2VStepStatus.INVALID_INPUT_STATE
    )
    with pytest.raises(ValueError, match="certified D2V17"):
        D2V17PeriodicTransportPlan(
            d2v37_off_lattice_quadrature(),
            (7, 7),
            (0.01, 0.01),
            0.01,
        )
    with pytest.raises(ValueError, match="exceed twice"):
        D2V17PeriodicTransportPlan(
            d2v17_quadrature(),
            (4, 5),
            (0.01, 0.01),
            0.01,
        )
