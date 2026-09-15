#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _soc_program():
    conic = phx.optim.ConicProgram(
        None,
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0, -1.0], [-2.0, 0.0]]),
        jnp.asarray([0.0, -1.0]),
        phx.optim.SecondOrderCone(2),
        bounds=phx.optim.Bounds(
            jnp.asarray([0.0, 0.0]),
            jnp.asarray([1.0, 2.0]),
        ),
        problem_id="binary-soc",
    )
    return phx.optim.MixedIntegerProgram(
        conic,
        binary_indices=(0,),
        program_id="binary-soc-mixed",
    )


def _clarabel_outer(*, maximum_rounds=100):
    pytest.importorskip("clarabel")
    return phx.optim.ConicOuterApproximation(
        conic=phx.optim.ConvexSolvePolicy(phx.optim.ClarabelInteriorPoint()),
        maximum_rounds=maximum_rounds,
    )


def test_projection_separator_is_globally_valid_and_excludes_source():
    program = _soc_program().relaxation
    source = jnp.asarray([0.0, 0.0])

    separation = phx.optim.separate_conic_point(
        program,
        source,
        tolerance=1e-8,
        binding_id="separator-test",
    )

    assert separation.valid
    assert separation.violated_blocks == 1
    assert len(separation.cuts) == 1
    cut, audit = separation.cuts[0], separation.audits[0]
    assert audit.global_valid
    assert audit.source_violated
    feasible = (
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([0.5, 0.0]),
    )
    for candidate in feasible:
        assert cut.row @ candidate <= cut.rhs + 1e-8


def test_conic_outer_approximation_matches_fixed_integer_reference():
    program = _soc_program()
    policy = phx.optim.MixedIntegerSolvePolicy(_clarabel_outer())

    result = phx.optim.solve_mixed_integer_program(program, policy)

    assert result.successful
    assert result.certificate.global_bound_certified
    assert jnp.isclose(result.objective, 1.0, atol=1e-7)
    assert jnp.isclose(result.global_lower_bound, 1.0, atol=1e-7)
    assert jnp.isclose(result.primal[1], 1.0, atol=1e-7)
    assert result.work.master_solves >= 1
    assert result.work.fixed_discrete_solves >= 1
    assert result.work.cuts_accepted >= 1


def test_outer_round_limit_never_reports_certified_success():
    program = _soc_program()
    policy = phx.optim.MixedIntegerSolvePolicy(_clarabel_outer(maximum_rounds=1))

    result = phx.optim.solve_mixed_integer_program(program, policy)

    assert result.status == phx.optim.MixedIntegerStatus.WORK_LIMIT
    assert not result.certificate.search_complete
    assert not result.successful
