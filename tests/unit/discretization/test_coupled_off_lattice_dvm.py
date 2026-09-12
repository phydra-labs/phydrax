#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.discrete_velocity._quadrature import (
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
)
from phydrax.discretization.discrete_velocity._semi_lagrangian import (
    CoupledD2V37TransportStatus,
    PeriodicUniformGridDepartureTransfer,
    PreparedCoupledD2V37OffLatticeTransport,
    SemiLagrangianTransferRequirements,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport


def _method(quadrature):
    return SmoothCompressibleD2VKineticMethod(
        quadrature,
        IdealGasMaterial(1.4, 1.0),
        ConstantTransport(0.03, 0.04),
    )


def _prepared(shape=(5, 6), spacing=(0.7, 1.1), time_step=0.13):
    method = _method(d2v37_off_lattice_quadrature())
    return PreparedCoupledD2V37OffLatticeTransport.prepare(
        method, shape, spacing, time_step
    )


def test_periodic_uniform_departure_transfer_preserves_constant_integral_and_direction():
    transfer = PeriodicUniformGridDepartureTransfer((4, 3), (0.5, 2.0), (0.25, -0.5))
    constant = jnp.full((4, 3), 2.75, dtype=jnp.float64)
    transported_constant = transfer.primal_operator.mv(constant)
    np.testing.assert_allclose(transported_constant, constant, atol=1.0e-14)

    population = jnp.zeros((4, 3), dtype=jnp.float64).at[1, 1].set(1.0)
    transported = transfer.primal_operator.mv(population)
    assert float(jnp.min(transported)) >= 0.0
    np.testing.assert_allclose(jnp.sum(transported), jnp.sum(population), atol=1.0e-14)

    expected = jnp.zeros_like(population)
    expected = expected.at[1, 0].set(0.375)
    expected = expected.at[1, 1].set(0.375)
    expected = expected.at[2, 0].set(0.125)
    expected = expected.at[2, 1].set(0.125)
    np.testing.assert_allclose(transported, expected, atol=1.0e-14)


def test_coupled_d2v37_uses_one_transfer_tuple_and_audits_declared_moments():
    prepared = _prepared()
    quadrature = prepared.quadrature
    f_constants = jnp.linspace(
        0.1, 0.9, quadrature.population_count, dtype=quadrature.velocities.dtype
    )
    g_constants = jnp.linspace(
        0.2, 1.1, quadrature.population_count, dtype=quadrature.velocities.dtype
    )
    shape = prepared.population_transport.source_shape
    state = SmoothCompressibleKineticState(
        jnp.broadcast_to(f_constants, shape + (quadrature.population_count,)),
        jnp.broadcast_to(g_constants, shape + (quadrature.population_count,)),
    )

    result = prepared.transport_with_evidence(state, prepared.required_step_size)

    np.testing.assert_allclose(
        result.candidate_state.particle_populations,
        state.particle_populations,
    )
    np.testing.assert_allclose(
        result.candidate_state.total_energy_populations,
        state.total_energy_populations,
    )
    np.testing.assert_allclose(
        result.evidence.f.population_conservation_residual, 0.0, atol=2.0e-11
    )
    np.testing.assert_allclose(
        result.evidence.g.population_conservation_residual, 0.0, atol=2.0e-11
    )
    np.testing.assert_allclose(result.evidence.f.conservation_residual, 0.0, atol=2.0e-11)
    np.testing.assert_allclose(result.evidence.g.conservation_residual, 0.0, atol=2.0e-11)
    assert result.evidence.f.declared_moment_names == (
        "mass",
        "momentum_x",
        "momentum_y",
    )
    assert result.evidence.g.declared_moment_names == ("total_energy",)
    assert bool(result.evidence.positivity_preserved)
    assert bool(result.successful)
    assert int(result.status) == int(CoupledD2V37TransportStatus.SUCCESS)
    assert result.transport_id == result.evidence.transport_id
    assert result.prepared_id == result.evidence.prepared_id
    shared = prepared.transport_with_evidence(
        SmoothCompressibleKineticState(
            state.particle_populations, state.particle_populations
        ),
        prepared.required_step_size,
    )
    np.testing.assert_allclose(
        shared.candidate_state.particle_populations,
        shared.candidate_state.total_energy_populations,
    )


def test_coupled_d2v37_preserves_positive_population_integrals_and_gradients():
    prepared = _prepared(shape=(4, 5), spacing=(0.8, 0.6), time_step=0.09)
    q = prepared.quadrature.population_count
    dtype = prepared.quadrature.velocities.dtype
    f = jnp.linspace(0.01, 1.4, 4 * 5 * q, dtype=dtype).reshape((4, 5, q))
    g = jnp.linspace(0.02, 1.8, 4 * 5 * q, dtype=dtype).reshape((4, 5, q))
    state = SmoothCompressibleKineticState(f, g)

    result = prepared.transport_with_evidence(state, prepared.required_step_size)

    assert float(jnp.min(result.candidate_state.particle_populations)) >= 0.0
    assert float(jnp.min(result.candidate_state.total_energy_populations)) >= 0.0
    np.testing.assert_allclose(
        result.evidence.f.source_population_integrals,
        result.evidence.f.target_population_integrals,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        result.evidence.g.source_population_integrals,
        result.evidence.g.target_population_integrals,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        result.evidence.f.source_moments,
        result.evidence.f.target_moments,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        result.evidence.g.source_moments,
        result.evidence.g.target_moments,
        atol=2.0e-11,
    )

    def objective(particle_populations):
        candidate = prepared.transport_with_evidence(
            SmoothCompressibleKineticState(particle_populations, g),
            prepared.required_step_size,
        ).candidate_state
        return jnp.sum(candidate.particle_populations**2)

    gradient = jax.grad(objective)(f)
    assert gradient.shape == f.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_coupled_d2v37_refuses_variable_step_nonperiodic_geometry_and_bad_shapes():
    prepared = _prepared()
    q = prepared.quadrature.population_count
    shape = prepared.population_transport.source_shape
    dtype = prepared.quadrature.velocities.dtype
    state = SmoothCompressibleKineticState(
        jnp.ones(shape + (q,), dtype=dtype),
        jnp.ones(shape + (q,), dtype=dtype),
    )

    refused = prepared.transport_with_evidence(state, 0.5 * prepared.required_step_size)
    assert not bool(refused.successful)
    assert int(refused.status) == int(CoupledD2V37TransportStatus.FIXED_STEP_MISMATCH)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="refuses a time step"
    ):
        invalid = prepared.transport(state, 0.5 * prepared.required_step_size)
        jax.block_until_ready(invalid.particle_populations)

    with pytest.raises(ValueError, match="periodic 2-D geometry"):
        PreparedCoupledD2V37OffLatticeTransport.prepare(
            _method(d2v37_off_lattice_quadrature()),
            (4, 5),
            (1.0, 1.0),
            0.1,
            periodic_axes=(True, False),
        )
    with pytest.raises(ValueError, match="at least two cells"):
        PeriodicUniformGridDepartureTransfer((1, 5), (1.0, 1.0), (0.25, 0.25))
    bad_state = SmoothCompressibleKineticState(
        jnp.ones((shape[0], shape[1] + 1, q), dtype=dtype),
        jnp.ones((shape[0], shape[1] + 1, q), dtype=dtype),
    )
    with pytest.raises(ValueError, match="must have shape"):
        prepared.transport_with_evidence(bad_state, prepared.required_step_size)


def test_coupled_d2v37_rejects_integer_roll_and_d2v17_identity_claims():
    transfer = PeriodicUniformGridDepartureTransfer((4, 5), (1.0, 1.0), (0.25, 0.5))
    with pytest.raises(ValueError, match="integer_roll"):
        SemiLagrangianTransferRequirements(exact_on=("integer_roll",)).validate(transfer)

    d2v17_method = _method(d2v17_quadrature())
    with pytest.raises(ValueError, match="D2V37 quadrature identity"):
        PreparedCoupledD2V37OffLatticeTransport.prepare(
            d2v17_method, (4, 5), (1.0, 1.0), 0.1
        )

    d2v37_transport = _prepared().population_transport
    with pytest.raises(ValueError, match="D2V37 quadrature identity"):
        PreparedCoupledD2V37OffLatticeTransport(d2v17_method, d2v37_transport, (0.7, 1.1))
