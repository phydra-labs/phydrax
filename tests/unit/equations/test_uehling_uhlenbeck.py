#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.equations._uehling_uhlenbeck import (
    QuantumStatistics,
    UehlingUhlenbeckPlan,
    UUCollisionStatus,
)


def _plan(statistics, *, kernel=1.0, maximum_substeps=64):
    four_momenta = jnp.asarray(
        (
            ((2.0, 1.0, 0.0, 0.0),),
            ((2.0, -1.0, 0.0, 0.0),),
            ((2.0, 0.0, 1.0, 0.0),),
            ((2.0, 0.0, -1.0, 0.0),),
        )
    )
    return UehlingUhlenbeckPlan(
        jnp.asarray(statistics, dtype=jnp.int8),
        jnp.ones((4, 1)),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.asarray((kernel,)),
        four_momenta,
        jnp.asarray(((1.0,), (-1.0,), (1.0,), (-1.0,))),
        time_unit_id="dimensionless-equation-test-time",
        maximum_substeps=maximum_substeps,
        invariant_tolerance=1.0e-6,
        entropy_tolerance=1.0e-6,
    )


def _occupancy(values):
    return jnp.asarray(values, dtype=float).reshape((4, 1, 1))


def test_shared_quantum_event_has_classical_limit_fermi_blocking_and_bose_enhancement():
    dilute = _occupancy((1.0e-6, 2.0e-6, 3.0e-7, 4.0e-7))
    classical = _plan((0, 0, 0, 0)).event_flux(dilute)
    quantum = _plan((1, 1, 1, 1)).event_flux(dilute)
    np.testing.assert_allclose(quantum, classical, rtol=1.0e-6, atol=1.0e-18)

    saturated = _occupancy((0.4, 0.5, 1.0, 0.0))
    np.testing.assert_allclose(
        _plan((-1, -1, -1, -1)).event_flux(saturated), 0.0, atol=0.0
    )

    populated_outgoing = _occupancy((0.2, 0.3, 0.4, 0.0))
    classical_flux = _plan((0, 0, 0, 0)).event_flux(populated_outgoing)
    bose_flux = _plan((1, 1, 1, 1)).event_flux(populated_outgoing)
    assert float(bose_flux[0, 0]) > float(classical_flux[0, 0])


def test_fd_and_be_equilibria_satisfy_pointwise_detailed_balance():
    energies = jnp.asarray((1.0, 2.0, 1.5, 1.5))
    chemical = jnp.asarray((0.1, 0.2, 0.15, 0.15))
    exponent = energies - chemical
    fermi = 1.0 / (jnp.exp(exponent) + 1.0)
    bose = 1.0 / jnp.expm1(exponent)

    np.testing.assert_allclose(
        _plan((-1, -1, -1, -1)).detailed_balance_residual(_occupancy(fermi)),
        0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        _plan((1, 1, 1, 1)).detailed_balance_residual(_occupancy(bose)),
        0.0,
        atol=2.0e-7,
    )


def test_subcycled_scatter_preserves_bounds_invariants_and_increases_entropy():
    plan = _plan(
        (
            QuantumStatistics.FERMI,
            QuantumStatistics.FERMI,
            QuantumStatistics.FERMI,
            QuantumStatistics.FERMI,
        ),
        kernel=3.0,
    )
    initial = _occupancy((0.8, 0.7, 0.1, 0.2))
    result = plan.advance(initial, 0.1)

    assert bool(result.successful)
    assert int(result.evidence.status) == int(UUCollisionStatus.SUCCESS)
    assert float(result.evidence.minimum_occupancy) >= 0.0
    assert float(result.evidence.maximum_fermi_occupancy) <= 1.0
    np.testing.assert_allclose(result.evidence.charge_defect, 0.0, atol=2.0e-7)
    np.testing.assert_allclose(result.evidence.four_momentum_defect, 0.0, atol=2.0e-7)
    assert float(jnp.min(result.evidence.entropy_production)) >= -1.0e-7


def test_invariant_and_h_theorem_evidence_is_pointwise_in_space():
    plan = _plan((-1, -1, -1, -1), kernel=2.0)
    initial = jnp.asarray(
        (
            ((0.8,), (0.6,)),
            ((0.7,), (0.5,)),
            ((0.1,), (0.2,)),
            ((0.2,), (0.3,)),
        )
    )
    result = plan.advance(initial, 0.05)

    assert bool(result.successful)
    assert result.evidence.charge_defect.shape == (2, 1)
    assert result.evidence.four_momentum_defect.shape == (2, 4)
    assert result.evidence.entropy_production.shape == (2,)
    assert bool(jnp.all(result.evidence.entropy_production >= -1.0e-7))


def test_substep_capacity_exhaustion_rolls_back_atomically():
    plan = _plan((-1, -1, -1, -1), kernel=100.0, maximum_substeps=1)
    initial = _occupancy((0.9, 0.9, 0.01, 0.01))
    result = plan.advance(initial, 1.0)

    assert not bool(result.successful)
    assert int(result.evidence.status) == int(
        UUCollisionStatus.SUBSTEP_CAPACITY_EXHAUSTED
    )
    np.testing.assert_array_equal(result.accepted_occupancy, initial)
