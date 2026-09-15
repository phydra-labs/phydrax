#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.optics.wave._coupled_mode import (
    BidirectionalCoupledModePlan,
    BidirectionalCoupledModeStatus,
    prepare_bidirectional_coupled_mode,
    solve_bidirectional_coupled_mode,
)


jax.config.update("jax_enable_x64", True)


def test_uniform_no_grating_has_exact_phase_loss_and_power_closure():
    grid = jnp.linspace(0.0, 2.0, 9)
    plan = BidirectionalCoupledModePlan(
        grid,
        2.0e15,
        jnp.full((8,), 0.4),
        jnp.zeros((8,), dtype=complex),
        jnp.full((8,), 0.3),
        reference_propagation_constant=1.1,
        power_tolerance=1.0e-12,
    )
    result = solve_bidirectional_coupled_mode(
        prepare_bidirectional_coupled_mode(plan), left_incoming=2.0 - 0.5j
    )
    expected = (2.0 - 0.5j) * np.exp((-0.15 + 1.5j) * 2.0)

    np.testing.assert_allclose(result.boundary.left_outgoing, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.boundary.right_outgoing, expected, rtol=1.0e-12)
    np.testing.assert_allclose(
        result.evidence.output_power,
        result.evidence.input_power * np.exp(-0.3 * 2.0),
        rtol=1.0e-12,
    )
    assert result.evidence.power_balance_residual < 1.0e-12
    assert result.status == int(BidirectionalCoupledModeStatus.SUCCESS)


def test_exact_scattering_stays_finite_deep_inside_uniform_stop_band():
    strength_length = 800.0
    plan = BidirectionalCoupledModePlan(
        jnp.asarray((0.0, 1.0)),
        2.0e15,
        jnp.asarray((0.0,)),
        jnp.asarray((strength_length + 0.0j,)),
        jnp.asarray((0.0,)),
    )
    result = plan.prepare().execute(left_incoming=1.0 + 0.0j)

    assert np.isfinite(result.scattering_matrix).all()
    np.testing.assert_allclose(abs(result.boundary.left_outgoing), 1.0, atol=1.0e-12)
    assert abs(result.boundary.right_outgoing) < 1.0e-300
    assert result.evidence.passivity_excess < 1.0e-12
    assert result.evidence.reciprocity_error < 1.0e-14
    assert result.successful


def test_stable_section_composition_is_reciprocal_for_two_sided_batches():
    sections = 128
    plan = BidirectionalCoupledModePlan(
        jnp.linspace(0.0, 0.08, sections + 1),
        1.2e15,
        jnp.linspace(-60.0, 70.0, sections),
        120.0 * jnp.exp(1j * jnp.linspace(-0.4, 0.7, sections)),
        jnp.linspace(0.0, 4.0, sections),
        reciprocity_tolerance=1.0e-11,
        passivity_tolerance=1.0e-11,
        power_tolerance=1.0e-10,
    )
    prepared = plan.prepare()
    left = jnp.asarray((1.0 + 0.2j, -0.3j))
    right = jnp.asarray((0.1 - 0.4j, 0.7 + 0.0j))
    result = prepared.execute(left_incoming=left, right_incoming=right)

    assert result.forward_amplitude.shape == (sections + 1, 2)
    assert result.backward_amplitude.shape == (sections + 1, 2)
    assert result.evidence.section_absorbed_power.shape == (sections, 2)
    assert np.all(result.evidence.section_absorbed_power >= -1.0e-10)
    assert result.evidence.reciprocity_error < 1.0e-11
    assert np.all(result.evidence.power_balance_residual < 1.0e-10)
    assert result.successful


def test_passive_plan_rejects_gain_instead_of_reinterpreting_its_sign():
    with pytest.raises(ValueError, match="rejects gain"):
        BidirectionalCoupledModePlan(
            (0.0, 1.0),
            1.0,
            (0.0,),
            (0.0,),
            (-0.01,),
        )


def test_fixed_branch_exact_map_has_finite_coupling_gradient():
    base = BidirectionalCoupledModePlan(
        jnp.asarray((0.0, 0.2, 0.5)),
        4.0,
        jnp.asarray((0.2, -0.1)),
        jnp.asarray((0.7, 0.7)),
        jnp.asarray((0.01, 0.02)),
    )

    def transmitted_power(coupling):
        plan = eqx.tree_at(
            lambda value: value.coupling,
            base,
            jnp.broadcast_to(coupling, base.coupling.shape),
        )
        output = plan.prepare().execute(left_incoming=1.0)
        return jnp.real(
            output.boundary.right_outgoing * jnp.conj(output.boundary.right_outgoing)
        )

    derivative = jax.grad(transmitted_power)(jnp.asarray(0.7))
    assert jnp.isfinite(derivative)
    assert derivative != 0.0
