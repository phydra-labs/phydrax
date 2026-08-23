#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest
from jaxtyping import Array

import phydrax as phx


la = phx.linalg
opt = phx.optim


class _NoMaterializeSelfAdjoint(la.AbstractLinearOperator):
    diagonal: Array

    def __init__(self, diagonal):
        diagonal_ = jnp.asarray(diagonal)
        self.diagonal = diagonal_
        self.source = la.ArraySpace(diagonal_.shape, dtype=diagonal_.dtype)
        self.target = self.source
        self.properties = la.OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        )
        self.capabilities = la.OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = "no-materialize-self-adjoint"

    def mv(self, vector):
        return self.diagonal * self.source.validate(vector)

    def transpose_mv(self, vector):
        return self.mv(vector)

    def adjoint_mv(self, vector):
        return self.mv(vector)

    def _materialize(self):
        raise AssertionError("Trust-region solve must remain matrix-free.")


def test_steihaug_toint_recovers_interior_newton_step_without_materialization():
    operator = _NoMaterializeSelfAdjoint(jnp.asarray([2.0, 4.0]))
    result = opt.solve_trust_region_subproblem(
        opt.TrustRegionQuadraticProblem(
            operator,
            jnp.asarray([-2.0, -4.0]),
            10.0,
        )
    )

    assert bool(result.successful)
    assert result.status == int(opt.TrustRegionSubproblemStatus.CONVERGED)
    assert jnp.allclose(result.step, jnp.ones((2,)), atol=1e-8)
    assert float(result.diagnostics.predicted_reduction) == pytest.approx(3.0)
    assert not bool(result.diagnostics.boundary_hit)


def test_steihaug_toint_reports_boundary_and_negative_curvature():
    positive = _NoMaterializeSelfAdjoint(jnp.asarray([1.0, 1.0]))
    boundary = opt.solve_trust_region_subproblem(
        opt.TrustRegionQuadraticProblem(
            positive,
            jnp.asarray([-2.0, 0.0]),
            0.5,
        )
    )
    assert boundary.status == int(opt.TrustRegionSubproblemStatus.BOUNDARY_REACHED)
    assert float(jnp.linalg.norm(boundary.step)) == pytest.approx(0.5)

    indefinite = _NoMaterializeSelfAdjoint(jnp.asarray([-1.0, 2.0]))
    negative = opt.solve_trust_region_subproblem(
        opt.TrustRegionQuadraticProblem(
            indefinite,
            jnp.asarray([1.0, 0.0]),
            1.0,
        )
    )
    assert negative.status == int(opt.TrustRegionSubproblemStatus.NEGATIVE_CURVATURE)
    assert bool(negative.diagnostics.negative_curvature)
    assert float(jnp.linalg.norm(negative.step)) == pytest.approx(1.0)


def _optimization_termination(maximum_steps=30):
    return opt.OptimizationTermination(
        absolute_optimality=1e-9,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
    )


def test_matrix_free_newton_trust_region_has_no_dense_dimension_cap():
    size = 1024
    target = jnp.ones((size,))
    problem = opt.MinimizationProblem(
        lambda parameters, expected: 0.5 * jnp.sum((parameters - expected) ** 2),
        problem_id="large-matrix-free-trust-region",
    )
    result = opt.minimize(
        problem,
        jnp.zeros((size,)),
        args=target,
        method=opt.NewtonTrustRegion(initial_radius=64.0),
        termination=_optimization_termination(maximum_steps=10),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.parameters, target)
    assert result.provenance.method == "newton-trust-region/steihaug-toint"
    assert int(result.diagnostics.hvp_evaluations) > 0


def test_dense_dogleg_remains_an_explicit_small_system_method():
    problem = opt.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - 1.0) ** 2)
    )
    result = opt.minimize(
        problem,
        jnp.zeros((2,)),
        method=opt.DenseNewtonDogleg(),
        termination=_optimization_termination(),
    )
    assert bool(result.successful)
    assert result.provenance.method == "dense-newton-dogleg"


def test_bounded_newton_trust_region_preserves_active_bounds():
    problem = opt.MinimizationProblem(
        lambda parameters, target: 0.5 * jnp.sum((parameters - target) ** 2),
        bounds=opt.Bounds(0.0, 1.0),
        problem_id="bounded-trust-region",
    )
    result = opt.minimize(
        problem,
        jnp.asarray([0.2, 0.2]),
        args=jnp.asarray([2.0, -0.5]),
        method=opt.BoundedNewtonTrustRegion(initial_radius=2.0),
        termination=_optimization_termination(),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.parameters, jnp.asarray([1.0, 0.0]))
    assert float(result.diagnostics.primal_feasibility) == 0.0
    assert result.provenance.globalization == "projected-trust-region"
    assert int(result.diagnostics.active_constraints) == 2


@pytest.mark.parametrize(
    "method",
    [opt.BoundedGaussNewton(), opt.BoundedLevenbergMarquardt()],
)
def test_bounded_least_squares_uses_residual_model_and_never_leaves_box(method):
    def residual(parameters, target):
        return jnp.where(
            (parameters < 0.0) | (parameters > 1.0),
            jnp.nan,
            parameters - target,
        )

    problem = opt.NonlinearLeastSquaresProblem(
        residual,
        bounds=opt.Bounds(0.0, 1.0),
        problem_id="bounded-residual",
    )
    result = opt.least_squares(
        problem,
        jnp.asarray([-1.0, 2.0]),
        args=jnp.asarray([2.0, -0.5]),
        method=method,
        termination=_optimization_termination(),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.parameters, jnp.asarray([1.0, 0.0]), atol=1e-7)
    assert float(result.diagnostics.primal_feasibility) == 0.0
    assert result.provenance.globalization == "projected-residual-trust-region"


def test_unbounded_least_squares_method_rejects_declared_bounds():
    problem = opt.NonlinearLeastSquaresProblem(
        lambda parameters, target: parameters - target,
        bounds=opt.Bounds(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="do not silently ignore bounds"):
        opt.least_squares(
            problem,
            jnp.asarray([0.5]),
            args=jnp.asarray([1.0]),
            method=opt.GaussNewton(),
        )
