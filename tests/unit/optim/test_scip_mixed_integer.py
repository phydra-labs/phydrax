#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_scip_method_rejects_quadratic_program_before_provider_import():
    program = phx.optim.MixedIntegerProgram(
        phx.optim.QuadraticProgram(
            jnp.eye(1),
            jnp.zeros((1,)),
            bounds=phx.optim.Bounds(0.0, 1.0),
            convexity_evidence="construction",
        ),
        binary_indices=(0,),
    )

    with pytest.raises(ValueError, match="does not support"):
        phx.optim.plan_mixed_integer_program(
            program,
            phx.optim.MixedIntegerSolvePolicy(phx.optim.SCIPMixedInteger()),
        )


def test_live_scip_solution_is_primal_audited_but_provider_qualified():
    pytest.importorskip("pyscipopt")
    program = phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([-1.0]),
            bounds=phx.optim.Bounds(0.0, 1.0),
        ),
        binary_indices=(0,),
    )

    result = phx.optim.solve_mixed_integer_program(
        program,
        phx.optim.MixedIntegerSolvePolicy(phx.optim.SCIPMixedInteger()),
    )

    assert result.status == phx.optim.MixedIntegerStatus.OPTIMAL
    assert result.feasible
    assert result.certificate.provider_solved
    assert result.certificate.proof_kind == "provider-reported"
    assert not result.certified
    assert not result.successful
    assert result.primal[0] == 1.0
