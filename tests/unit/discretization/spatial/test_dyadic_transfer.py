from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spatial import (
    AdaptiveDyadicGridPlan,
    DyadicCellTransferPlan,
    MortonAddressPlan,
)


def _refined_pair():
    plan = AdaptiveDyadicGridPlan(
        MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 3),
        cell_capacity=32,
    )
    coarse = plan.prepare()
    refine = jnp.zeros((plan.cell_capacity,), dtype=bool).at[0].set(True)
    fine = plan.adapt(coarse, refine_mask=refine).accepted
    return plan, coarse, fine


def test_dyadic_average_prolongation_and_restriction_are_conservative() -> None:
    _, coarse, fine = _refined_pair()
    coarse_values = jnp.where(coarse.leaf_active, 3.5, 0.0)
    prolongation = DyadicCellTransferPlan(coarse, fine)
    fine_result = eqx.filter_jit(prolongation.apply_cell_averages)(coarse_values)
    assert bool(fine_result.successful)
    np.testing.assert_allclose(fine_result.values[fine.leaf_active], 3.5)
    np.testing.assert_allclose(fine_result.conservation_residual, 0.0, atol=1e-14)

    restriction = DyadicCellTransferPlan(fine, coarse)
    coarse_result = eqx.filter_jit(restriction.apply_cell_averages)(fine_result.values)
    assert bool(coarse_result.successful)
    np.testing.assert_allclose(coarse_result.values[coarse.leaf_active], 3.5)
    np.testing.assert_allclose(coarse_result.conservation_residual, 0.0, atol=1e-14)


def test_dyadic_content_transfer_preserves_total_content() -> None:
    _, coarse, fine = _refined_pair()
    coarse_content = jnp.where(coarse.leaf_active, 8.0, 0.0)
    prolongation = DyadicCellTransferPlan(coarse, fine)
    fine_result = prolongation.apply_cell_contents(coarse_content)
    assert bool(fine_result.successful)
    np.testing.assert_allclose(jnp.sum(fine_result.values), 8.0)
    np.testing.assert_allclose(fine_result.values[fine.leaf_active], 2.0)
    restriction = DyadicCellTransferPlan(fine, coarse)
    coarse_result = restriction.apply_cell_contents(fine_result.values)
    np.testing.assert_allclose(jnp.sum(coarse_result.values), 8.0)
    np.testing.assert_allclose(coarse_result.conservation_residual, 0.0, atol=1e-14)
