from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _problem(coefficients, volume, *, extra=None):
    coefficients = jnp.asarray(coefficients)
    constraints = [
        phx.optim.NonlinearConstraint(
            lambda value, _: jnp.sum(value),
            upper=volume,
            constraint_id="volume",
        )
    ]
    if extra is not None:
        weights, budget = extra
        constraints.append(
            phx.optim.NonlinearConstraint(
                lambda value, _, weights=jnp.asarray(weights): jnp.sum(weights * value),
                upper=budget,
                constraint_id="weighted-volume",
            )
        )
    return phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(coefficients / value),
        bounds=phx.optim.Bounds(0.05, 5.0),
        constraints=tuple(constraints),
        problem_id="analytic-mma",
    )


def _solve(problem, initial, *, steps=180):
    return phx.optim.minimize(
        problem,
        initial,
        method=phx.optim.MethodOfMovingAsymptotes(),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=2.0e-6,
            relative_optimality=0.0,
            absolute_step=1.0e-12,
            relative_step=0.0,
            maximum_steps=steps,
        ),
    )


def test_mma_recovers_analytic_reciprocal_optimum_and_certificate():
    coefficients = jnp.asarray((0.5, 1.0, 2.0, 3.0, 4.0, 1.5))
    volume = 3.0
    expected = jnp.sqrt(coefficients)
    expected = expected / jnp.sum(expected) * volume
    result = _solve(_problem(coefficients, volume), jnp.full((6,), volume / 6))

    assert result.successful
    np.testing.assert_allclose(result.parameters, expected, rtol=2.0e-4, atol=2.0e-5)
    assert float(jnp.sum(result.parameters)) <= volume + 2.0e-6
    assert result.certificate is not None
    assert result.optimality_certificate is not None
    assert result.optimality_certificate.certified
    assert result.diagnostics.primal_feasibility <= 2.0e-6
    assert isinstance(result.method_evidence, phx.optim.MMAEvidence)


def test_mma_handles_pytree_design_and_two_binding_constraints():
    coefficients = jnp.asarray((1.0, 1.5, 2.0, 2.5))
    initial = {"density": jnp.asarray((0.3, 0.3, 0.7, 0.7))}
    weights = jnp.asarray((1.0, 1.0, 0.0, 0.0))
    problem = phx.optim.MinimizationProblem(
        lambda value, _: jnp.sum(coefficients / value["density"]),
        bounds=phx.optim.Bounds(
            {"density": jnp.full((4,), 0.05)}, {"density": jnp.ones((4,))}
        ),
        constraints=(
            phx.optim.NonlinearConstraint(
                lambda value, _: jnp.sum(value["density"]),
                upper=2.0,
                constraint_id="volume",
            ),
            phx.optim.NonlinearConstraint(
                lambda value, _: jnp.sum(weights * value["density"]),
                upper=0.7,
                constraint_id="partial-volume",
            ),
        ),
        problem_id="pytree-mma",
    )
    result = _solve(problem, initial, steps=240)

    assert result.successful
    assert float(jnp.sum(result.parameters["density"])) <= 2.0 + 3.0e-6
    assert float(jnp.sum(weights * result.parameters["density"])) <= 0.7 + 3.0e-6
    assert result.certificate is not None
    assert result.certificate.inequality_sources == (
        "constraint:0:0:upper",
        "constraint:1:0:upper",
    )


def test_mma_is_jittable_for_fixed_problem_structure():
    coefficients = jnp.asarray((1.0, 2.0, 3.0))
    problem = _problem(coefficients, 1.5)
    solve = eqx.filter_jit(lambda initial: _solve(problem, initial, steps=80).parameters)
    result = solve(jnp.full((3,), 0.5))
    assert bool(jnp.all(jnp.isfinite(result)))
    assert float(jnp.sum(result)) <= 1.5 + 5.0e-5


def test_mma_rejects_missing_finite_bounds_equalities_and_infeasible_start():
    constraint = phx.optim.NonlinearConstraint(
        lambda value, _: jnp.sum(value),
        upper=1.0,
        constraint_id="volume",
    )
    method = phx.optim.MethodOfMovingAsymptotes()
    termination = phx.optim.OptimizationTermination(maximum_steps=2)

    with pytest.raises(ValueError, match="explicit finite"):
        method.solve(
            phx.optim.MinimizationProblem(
                lambda value, _: jnp.sum(value**2), constraints=(constraint,)
            ),
            jnp.asarray((0.2, 0.2)),
            termination=termination,
            args=None,
        )
    with pytest.raises(ValueError, match="finite parameter bounds"):
        method.solve(
            phx.optim.MinimizationProblem(
                lambda value, _: jnp.sum(value**2),
                bounds=phx.optim.Bounds(0.0, jnp.inf),
                constraints=(constraint,),
            ),
            jnp.asarray((0.2, 0.2)),
            termination=termination,
            args=None,
        )
    with pytest.raises(ValueError, match="not equalities"):
        method.solve(
            phx.optim.MinimizationProblem(
                lambda value, _: jnp.sum(value**2),
                bounds=phx.optim.Bounds(0.0, 1.0),
                constraints=(
                    phx.optim.NonlinearConstraint(
                        lambda value, _: jnp.sum(value),
                        lower=1.0,
                        upper=1.0,
                    ),
                ),
            ),
            jnp.asarray((0.5, 0.5)),
            termination=termination,
            args=None,
        )

    result = method.solve(
        phx.optim.MinimizationProblem(
            lambda value, _: jnp.sum(value**2),
            bounds=phx.optim.Bounds(0.0, 1.0),
            constraints=(constraint,),
        ),
        jnp.asarray((0.8, 0.8)),
        termination=termination,
        args=None,
    )
    assert int(result.status) == int(phx.optim.OptimizationStatus.INFEASIBLE)
