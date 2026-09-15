#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax.numpy as jnp

import phydrax as phx
from phydrax.optim._programming import _mixed_integer_native as mip_native


def _binary_implications(power_limit):
    return phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([0.0, 0.0, 1.0]),
            inequality_matrix=jnp.asarray(
                [
                    [1.0, -1.0, 0.0],
                    [0.0, 0.5, -1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            inequality_rhs=jnp.asarray([0.0, 0.0, power_limit]),
            bounds=phx.optim.Bounds(
                jnp.asarray([1.0, 0.0, 0.0]),
                jnp.asarray([1.0, 1.0, jnp.inf]),
            ),
        ),
        binary_indices=(0, 1),
    )


def test_binary_implication_contradiction_has_original_coordinate_farkas_proof():
    program = _binary_implications(0.0)
    certificate = mip_native._linear_bound_certificate(
        program.relaxation.as_quadratic_program(),
        1e-8,
    )
    assert certificate is not None
    assert certificate.valid
    assert certificate.objective < -1e-8
    assert certificate.residual_norm <= 1e-8
    result = phx.optim.solve_mixed_integer_program(program)
    assert result.status == phx.optim.MixedIntegerStatus.INFEASIBLE
    assert not result.successful


def test_near_feasible_binary_implications_are_not_pruned_without_a_proof():
    program = _binary_implications(0.5 + 1e-6)
    assert (
        mip_native._linear_bound_certificate(
            program.relaxation.as_quadratic_program(),
            1e-8,
        )
        is None
    )
    result = phx.optim.solve_mixed_integer_program(program)
    assert result.successful
    assert jnp.allclose(result.primal, jnp.asarray([1.0, 1.0, 0.5]), atol=2e-6)


def test_mixed_integer_preserves_linear_identity_and_refreshes_numeric_data():
    first = phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([-1.0]),
            bounds=phx.optim.Bounds(0.0, 1.0),
            problem_id="refresh-linear",
        ),
        binary_indices=(0,),
        program_id="refresh-mixed",
    )
    second = phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([1.0]),
            bounds=phx.optim.Bounds(0.0, 1.0),
            problem_id="refresh-linear",
        ),
        binary_indices=(0,),
        program_id="refresh-mixed",
    )

    assert isinstance(first.relaxation, phx.optim.LinearProgram)
    prepared = phx.optim.prepare_mixed_integer_program(first)
    refreshed = phx.optim.refresh_mixed_integer_program(prepared, second)
    result = phx.optim.solve_prepared_mixed_integer_program(refreshed).result

    assert refreshed.numeric_version == 1
    assert result.successful
    assert jnp.array_equal(result.primal, jnp.asarray([0.0]))


def test_invalid_supplied_candidate_cannot_become_an_incumbent():
    program = phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([-1.0]),
            bounds=phx.optim.Bounds(0.0, 1.0),
        ),
        binary_indices=(0,),
    )
    candidate = phx.optim.MixedIntegerCandidate(
        jnp.asarray([0.5]),
        source_id="fractional-proposal",
    )

    result = phx.optim.solve_mixed_integer_program(
        program,
        candidates=(candidate,),
    )

    assert result.successful
    assert jnp.allclose(result.primal, jnp.asarray([1.0]), atol=1e-7)
    assert result.work.candidates_audited >= 2
    assert result.work.candidates_accepted == 1
