#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _cochain(shape=(3, 4), *, periodic=True):
    dimension = len(shape)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for count in shape
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    return phx.discretization.StructuredCochainBridge(grid).cochain


def test_phi4_action_matches_explicit_cochain_energy_and_gradient():
    cochain = _cochain()
    action = phx.operators.path_integral.Phi4LatticeAction(
        cochain,
        kinetic_scale=1.3,
        mass_squared=-0.7,
        quartic_coupling=0.9,
    )
    field = jnp.linspace(-0.8, 1.1, cochain.cell_counts[0])
    derivative = cochain.exterior_derivative(0, field)
    expected = 0.5 * 1.3 * jnp.vdot(
        derivative, cochain.apply_hodge(1, derivative)
    ) + jnp.sum(cochain.dual_measures[0] * (-0.35 * field**2 + 0.225 * field**4))

    direction = jnp.cos(jnp.arange(field.size, dtype=field.dtype))
    step = 1e-5
    finite_difference = (
        action.action(field + step * direction) - action.action(field - step * direction)
    ) / (2.0 * step)
    gradient = phx.operators.path_integral.lattice_action_local_gradient(action, field)

    assert action.evidence.normalizable
    assert jnp.allclose(action.action(field), expected)
    assert jnp.allclose(jnp.vdot(gradient, direction), finite_difference, rtol=2e-4)


def test_phi4_incremental_cache_matches_full_action_and_rejection():
    action = phx.operators.path_integral.Phi4LatticeAction(
        _cochain((4, 3)),
        kinetic_scale=0.8,
        mass_squared=0.4,
        quartic_coupling=0.6,
    )
    local = phx.operators.path_integral.prepare_local_phi4_action(action)
    field = jnp.linspace(-0.3, 0.7, action.configuration_shape[0])
    move = phx.sampling.SingleCoordinateProposalPayload(
        index=jnp.asarray(5, dtype=jnp.int32),
        displacement=jnp.asarray(0.25),
    )
    proposed = field.at[5].add(0.25)
    initial_value, cache = local.initialize_incremental(field)
    delta, proposed_cache, valid = local.propose_incremental(field, cache, proposed, move)
    accepted_cache = local.select_incremental(cache, proposed_cache, jnp.asarray(True))
    rejected_cache = local.select_incremental(cache, proposed_cache, jnp.asarray(False))
    refreshed_value, refreshed_cache = local.refresh_incremental(proposed)

    assert valid
    assert jnp.allclose(initial_value, action.action(field))
    assert jnp.allclose(delta, action.action(proposed) - action.action(field))
    assert jnp.allclose(accepted_cache.action, refreshed_value)
    assert jnp.allclose(accepted_cache.edge_differences, refreshed_cache.edge_differences)
    assert jnp.array_equal(rejected_cache.edge_differences, cache.edge_differences)
    assert jnp.array_equal(rejected_cache.site_contributions, cache.site_contributions)
    assert jnp.array_equal(rejected_cache.action, cache.action)


def test_phi4_target_and_observable_contracts_are_composable():
    action = phx.operators.path_integral.Phi4LatticeAction(
        _cochain((3, 3)),
        mass_squared=1.0,
        quartic_coupling=0.0,
    )
    local = phx.operators.path_integral.prepare_local_phi4_action(action)
    full_target = phx.operators.path_integral.full_target_from_lattice_action(action)
    incremental = phx.operators.path_integral.incremental_target_from_lattice_action(
        local,
        refresh_cadence=4,
    )
    field = jnp.linspace(-1.0, 1.0, action.configuration_shape[0])

    assert jnp.allclose(full_target.initialize(field).log_target, -action.action(field))
    assert jnp.allclose(incremental.initialize(field).log_target, -action.action(field))
    plans = phx.operators.path_integral.phi4_observable_plans(action)
    values = [
        phx.operators.path_integral.evaluate_lattice_observable(plan, field)
        for plan in plans
    ]
    assert all(bool(value.valid) for value in values)
    assert all(value.value.shape == () for value in values)

    pair = phx.operators.path_integral.phi4_pair_correlation_plan(
        action,
        jnp.asarray([0, 1, 2]),
        jnp.asarray([1, 2, 0]),
    )
    expected = jnp.mean(field[jnp.asarray([0, 1, 2])] * field[jnp.asarray([1, 2, 0])])
    assert jnp.allclose(
        phx.operators.path_integral.evaluate_lattice_observable(pair, field).value,
        expected,
    )


def test_phi4_normalizability_and_locality_are_fail_closed():
    massless = phx.operators.path_integral.Phi4LatticeAction(
        _cochain((4,)),
        mass_squared=0.0,
        quartic_coupling=0.0,
    )
    assert not massless.evidence.normalizable
    with pytest.raises(ValueError, match="normalizability"):
        phx.operators.path_integral.full_target_from_lattice_action(massless)
    with pytest.raises(ValueError, match="non-negative"):
        phx.operators.path_integral.Phi4LatticeAction(
            _cochain((4,)), quartic_coupling=-1.0
        )

    cochain = _cochain((4,))
    dense_hodge = phx.discretization.CochainDiscretization(
        cochain.topology,
        cochain.hodge_stars,
        hodge_matrices=(
            None,
            jnp.diag(cochain.hodge_stars[1]),
        ),
        primal_measures=cochain.primal_measures,
        dual_measures=cochain.dual_measures,
        coordinates=cochain.coordinates,
    )
    dense_action = phx.operators.path_integral.Phi4LatticeAction(dense_hodge)
    with pytest.raises(ValueError, match="diagonal"):
        phx.operators.path_integral.prepare_local_phi4_action(dense_action)


def test_phi4_action_and_local_proposal_are_jittable():
    action = phx.operators.path_integral.Phi4LatticeAction(_cochain((4, 4)))
    field = jnp.zeros(action.configuration_shape)
    compiled_action = eqx.filter_jit(action.action)(field)
    proposal = phx.sampling.SingleCoordinateGaussianProposal(0.2)
    move = jax.jit(proposal.propose)(jax.random.key(4), field)

    assert compiled_action == 0.0
    assert move.valid
    assert jnp.sum(move.position != field) <= 1
