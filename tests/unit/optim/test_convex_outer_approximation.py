#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_ecp_matches_binary_enumeration_for_convex_objective():
    program = phx.optim.ConvexMixedIntegerNonlinearProgram(
        lambda value, args: (value[0] - args) ** 2,
        lambda value, args: jnp.empty((0,)),
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        jnp.empty((0,)),
        jnp.empty((0,)),
        constraint_evidence=(),
        binary_indices=(0,),
        args=0.7,
        objective_evidence_id="squared-distance-construction",
    )

    result = phx.optim.solve_convex_minlp(program)

    assert result.successful
    assert result.primal[0] == pytest.approx(1.0, abs=1e-6)
    assert result.objective == pytest.approx(0.09, abs=1e-6)
    assert result.global_lower_bound == pytest.approx(0.09, abs=1e-6)
    assert result.work.master_solves >= 1
    assert result.work.cuts_accepted >= 2


def test_ecp_convex_constraint_excludes_nonlinear_infeasible_binary():
    program = phx.optim.ConvexMixedIntegerNonlinearProgram(
        lambda value, args: (value[0] - 1.0) ** 2,
        lambda value, args: jnp.asarray([value[0] ** 2]),
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        jnp.asarray([-jnp.inf]),
        jnp.asarray([0.25]),
        constraint_evidence=(
            phx.optim.ConvexConstraintEvidence(
                "convex-upper",
                "squared-coordinate-construction",
            ),
        ),
        binary_indices=(0,),
        objective_evidence_id="squared-distance-construction",
    )

    result = phx.optim.solve_convex_minlp(program)

    assert result.successful
    assert result.primal[0] == pytest.approx(0.0, abs=1e-7)
    assert result.certificate.candidate_audit.constraint_violation <= 1e-7


def test_convex_minlp_rejects_curvature_orientation_mismatch():
    with pytest.raises(ValueError, match="orientation"):
        phx.optim.ConvexMixedIntegerNonlinearProgram(
            lambda value, args: value[0] ** 2,
            lambda value, args: jnp.asarray([value[0]]),
            jnp.asarray([0.0]),
            jnp.asarray([1.0]),
            jnp.asarray([0.0]),
            jnp.asarray([0.0]),
            constraint_evidence=(
                phx.optim.ConvexConstraintEvidence(
                    "convex-upper",
                    "invalid-equality-evidence",
                ),
            ),
            binary_indices=(0,),
            objective_evidence_id="quadratic",
        )


def test_ecp_round_limit_never_reports_success():
    program = phx.optim.ConvexMixedIntegerNonlinearProgram(
        lambda value, args: (value[0] - 0.7) ** 2,
        lambda value, args: jnp.empty((0,)),
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        jnp.empty((0,)),
        jnp.empty((0,)),
        constraint_evidence=(),
        binary_indices=(0,),
        objective_evidence_id="quadratic",
    )

    result = phx.optim.solve_convex_minlp(
        program,
        phx.optim.ConvexMINLPOuterApproximation(maximum_rounds=1),
    )

    assert result.status == phx.optim.ConvexMINLPStatus.WORK_LIMIT
    assert not result.successful
