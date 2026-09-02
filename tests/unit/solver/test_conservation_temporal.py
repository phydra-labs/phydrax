#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.solver._balance_law_composition import AdditiveIMEXTableau
from phydrax.solver._conservation_temporal import (
    ConservationIMEXMethod,
    ImplicitConservationStageResult,
    prepare_element_block_preconditioner,
)
from phydrax.solver._fem_multirate import (
    ConservativeLocalTimeStepPlan,
    DGMultirateTracePlan,
    TimeSlabFluxLedger,
)


def test_conservation_imex_commits_converged_implicit_stage():
    tableau = AdditiveIMEXTableau(
        jnp.asarray(((0.0,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
    )

    def implicit_solver(provisional, time, coefficient, args):
        del time, args
        state = provisional / (1.0 + 10.0 * coefficient)
        return ImplicitConservationStageResult(
            state,
            jnp.asarray(True),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.0),
        )

    method = ConservationIMEXMethod(
        tableau,
        lambda time, state, args: -state,
        lambda time, state, args: -10.0 * state,
        implicit_solver,
        method_id="linear-imex",
    )
    result = method.step(0.0, jnp.asarray((1.0,)), 0.1)
    assert result.successful
    np.testing.assert_allclose(result.accepted_state, (0.45,), atol=2.0e-12)
    assert result.implicit_iterations == 1


def test_element_block_preconditioner_uses_local_implicit_jacobians():
    state = jnp.asarray(((1.0,), (2.0,)))
    preconditioner = prepare_element_block_preconditioner(
        state,
        (jnp.asarray((0,)), jnp.asarray((1,))),
        lambda time, value, args: -2.0 * value,
        time=0.0,
        step_coefficient=0.1,
    )
    residual = jnp.asarray(((1.2,), (2.4,)))
    np.testing.assert_allclose(preconditioner.apply(residual), ((1.0,), (2.0,)))


def test_local_time_slab_accumulates_one_equal_opposite_flux():
    trace_plan = DGMultirateTracePlan(jnp.asarray(((0, 1),)), history_depth=2)
    plan = ConservativeLocalTimeStepPlan(
        jnp.asarray((0, 1), dtype=jnp.int32), 0.2, trace_plan
    )
    np.testing.assert_allclose(plan.cell_step_sizes(), (0.2, 0.1))
    np.testing.assert_array_equal(plan.active_cells(0), (True, True))
    np.testing.assert_array_equal(plan.active_cells(1), (False, True))
    ledger = TimeSlabFluxLedger.zeros(1, (1,), 0.2, ledger_id="shared-interface")
    ledger = ledger.add_substep(jnp.asarray(((3.0,),)), 0.1)
    ledger = ledger.add_substep(jnp.asarray(((5.0,),)), 0.1)
    assert ledger.complete
    contribution = ledger.equal_opposite_contributions()
    np.testing.assert_allclose(contribution.plus, ((0.8,),))
    np.testing.assert_allclose(contribution.minus, ((-0.8,),))
    np.testing.assert_allclose(contribution.conservation_defect, 0.0)
