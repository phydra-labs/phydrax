#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


def _level(index, *, shape=(5,), parent=None, transfer=None, coupling="shared"):
    return phx.stochastic.StochasticLevelSpec(
        f"level-{index}",
        index,
        refinement_axes=("time",),
        resolutions=(0.25 / 2**index,),
        state_shape=shape,
        problem_id="ou-problem",
        observable_id="terminal-square",
        solver_id="euler-maruyama",
        approximation_id=f"em-{index}",
        parent_level_id=parent,
        state_transfer_id=transfer,
        noise_coupling=coupling,
    )


def test_hierarchy_validates_order_parent_and_transfer_identity():
    levels = (
        _level(0),
        _level(1, parent="level-0"),
        _level(2, shape=(9,), parent="level-1", transfer="nested-grid-transfer"),
    )
    hierarchy = phx.stochastic.StochasticCouplingPlan(
        levels,
        hierarchy_id="ou-time-hierarchy",
    )

    assert hierarchy.num_levels == 3
    assert hierarchy.level(1) is levels[1]
    assert hierarchy.level("level-2") is levels[2]
    assert hierarchy.coupled
    assert isinstance(
        hierarchy.discretization_hierarchy,
        phx.discretization.DiscretizationHierarchy,
    )
    assert tuple(
        level.level_id for level in hierarchy.discretization_hierarchy.levels
    ) == ("level-0", "level-1", "level-2")
    assert all(level.discretization_bundle.records for level in levels)
    assert (
        hierarchy.fingerprint
        == phx.stochastic.StochasticCouplingPlan(
            levels,
            hierarchy_id="ou-time-hierarchy",
        ).fingerprint
    )

    with pytest.raises(ValueError, match="state_transfer_id"):
        phx.stochastic.StochasticCouplingPlan(
            (
                _level(0),
                _level(1, shape=(9,), parent="level-0"),
            ),
            hierarchy_id="missing-transfer",
        )
    with pytest.raises(ValueError, match="name 'level-0'"):
        phx.stochastic.StochasticCouplingPlan(
            (_level(0), _level(1, parent="wrong-parent")),
            hierarchy_id="bad-parent",
        )


def test_multi_axis_and_independent_noise_are_explicit():
    level = phx.stochastic.StochasticLevelSpec(
        "base",
        0,
        refinement_axes=("time", "space"),
        resolutions=(0.1, 0.25),
        state_shape=(4,),
        problem_id="problem",
        observable_id="observable",
        solver_id="solver",
        approximation_id="base",
        noise_coupling="independent",
    )
    with pytest.raises(ValueError, match="allow_multi_axis"):
        phx.stochastic.StochasticCouplingPlan((level,), hierarchy_id="multi")
    hierarchy = phx.stochastic.StochasticCouplingPlan(
        (level,), hierarchy_id="multi", allow_multi_axis=True
    )
    assert hierarchy.refinement_axes == ("time", "space")


def test_tensor_grid_transfer_preserves_nested_nodes_constants_and_jit():
    transfer = phx.discretization.TensorGridStateTransfer(
        (5, 5),
        (3, 3),
        restriction="injection",
    )
    coarse = jnp.arange(9.0).reshape((3, 3))
    fine = eqx.filter_jit(transfer.prolong)(coarse)
    round_trip = eqx.filter_jit(transfer.restrict)(fine)

    assert fine.shape == (5, 5)
    assert jnp.allclose(round_trip, coarse)
    assert jnp.allclose(transfer.prolong(jnp.ones((3, 3))), 1.0)

    weighted = phx.discretization.TensorGridStateTransfer(
        (6,),
        (3,),
        boundary="periodic",
        restriction="weighted",
    )
    assert jnp.allclose(weighted.restrict(jnp.ones((6,))), 1.0)
    assert jnp.allclose(weighted.prolong(jnp.ones((3,))), 1.0)


def test_spectral_and_identity_transfers_preserve_trailing_channels():
    identity = phx.discretization.IdentityStateTransfer((3,))
    state = jnp.arange(6.0).reshape((3, 2))
    assert identity.restrict(state) is state

    spectral = phx.discretization.SpectralCoefficientStateTransfer(
        (5,),
        (3,),
    )
    coarse = spectral.restrict(jnp.arange(10.0).reshape((5, 2)))
    restored = spectral.prolong(coarse)

    assert coarse.shape == (3, 2)
    assert restored.shape == (5, 2)
    assert jnp.array_equal(restored[:3], coarse)
    assert jnp.all(restored[3:] == 0.0)
