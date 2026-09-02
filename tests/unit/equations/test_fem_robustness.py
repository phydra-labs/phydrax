#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization._conservation_ledger import (
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from phydrax.discretization.fem._high_order import SimplexNodalFamily
from phydrax.equations.fem._robustness import (
    ConservationCorrectionLadderPlan,
    ConservativeSubcellPlan,
    RobustnessSensorState,
)
from phydrax.integration import GaussLegendreRule, ReferenceTriangleRule


def test_conservative_subcell_projection_preserves_contents_and_constants():
    element = SimplexNodalFamily("triangle", 2).finite_element()
    plan = ConservativeSubcellPlan(element, ReferenceTriangleRule(GaussLegendreRule(6)))
    state = jnp.linspace(-0.2, 0.8, element.local_dof_count * 2).reshape(
        (element.local_dof_count, 2)
    )
    contents = plan.contents(state)
    reconstructed = plan.reconstruct(contents)
    np.testing.assert_allclose(reconstructed, state, rtol=3.0e-10, atol=3.0e-10)
    constant = jnp.ones((element.local_dof_count, 1))
    np.testing.assert_allclose(plan.averages(constant), 1.0, atol=3.0e-11)
    np.testing.assert_allclose(
        jnp.sum(contents, axis=0),
        jnp.sum(plan.dg_to_subcell @ state, axis=0),
        atol=3.0e-12,
    )
    assert plan.evidence.positive_volumes


def _ledger(block, *, high=None, low=None):
    return ConservationStageLedger(
        (block,),
        jnp.zeros((2, 1)),
        jnp.ones((2,), dtype=bool),
        geometry_family_id="geometry-family",
        geometry_layout_id="geometry-layout",
        geometry_version=0,
        evidence_policy_id="evidence-policy",
        evidence_version=0,
        topology_epoch_id="topology-epoch",
        high_order_blocks=(block,) if high is None else (high,),
        low_order_blocks=(block,) if low is None else (low,),
    )


def test_correction_ladder_selects_shared_face_rate_and_conserves_content():
    high = ConservationStageFluxRateBlock(
        jnp.asarray(((10.0,),)),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((True,)),
        "shared-face",
        "interior",
    )
    low = high.with_flux_rate(jnp.asarray(((2.0,),)))
    high_ledger = _ledger(high, high=high, low=low)
    low_ledger = _ledger(low, high=high, low=low)
    sensor = RobustnessSensorState(
        jnp.asarray((0.0, 1.0)),
        jnp.asarray((False, True)),
        jnp.asarray((0, 2), dtype=jnp.int32),
    )
    result = ConservationCorrectionLadderPlan().apply(high_ledger, low_ledger, sensor)
    np.testing.assert_allclose(result.selected_ledger.blocks[0].flux_rate, 2.0)
    np.testing.assert_allclose(result.stage_content_rate[:, 0], (-2.0, 2.0))
    np.testing.assert_allclose(jnp.sum(result.stage_content_rate, axis=0), 0.0)
    np.testing.assert_array_equal(result.correction_level, (0, 2))
    assert result.successful
