#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _program():
    return phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([-1.0, 0.0]),
            equality_matrix=jnp.asarray([[0.0, 1.0]]),
            equality_rhs=jnp.asarray([0.25]),
            bounds=phx.optim.Bounds(
                jnp.asarray([0.0, 0.0]),
                jnp.asarray([1.0, 1.0]),
            ),
        ),
        binary_indices=(0,),
    )


def test_learned_proposal_is_audited_before_becoming_solver_input():
    program = _program()
    manifest = phx.ml.optimization.MixedIntegerProposalManifest(
        program.structure_id,
        "frozen-proposal-model",
        support_id="training-support",
    )
    proposal = phx.ml.optimization.materialize_mixed_integer_proposal(
        program,
        jnp.asarray([0.8, 0.25]),
        jnp.asarray([0.5]),
        manifest,
        support_accepted=True,
    )

    assert proposal.accepted_as_start
    np.testing.assert_array_equal(proposal.materialized, [1.0, 0.25])
    result = phx.optim.solve_mixed_integer_program(
        program,
        candidates=(proposal.candidate,),
    )
    assert result.successful
    np.testing.assert_allclose(result.primal, [1.0, 0.25], atol=1e-7)


def test_support_refusal_and_infeasible_continuous_value_block_admission():
    program = _program()
    manifest = phx.ml.optimization.MixedIntegerProposalManifest(
        program.structure_id,
        "frozen-proposal-model",
    )
    unsupported = phx.ml.optimization.materialize_mixed_integer_proposal(
        program,
        jnp.asarray([0.8, 0.25]),
        jnp.asarray([0.5]),
        manifest,
        support_accepted=False,
    )
    infeasible = phx.ml.optimization.materialize_mixed_integer_proposal(
        program,
        jnp.asarray([0.8, 0.75]),
        jnp.asarray([0.5]),
        manifest,
        support_accepted=True,
    )

    assert unsupported.audit.valid
    assert not unsupported.accepted_as_start
    assert not infeasible.audit.valid
    assert not infeasible.accepted_as_start
