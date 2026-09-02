#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.domain import Boundary
from phydrax.optim._kfac._blocks import (
    initialize_block_state,
    update_block_state,
    update_block_state_from_observations,
)
from phydrax.solver._functional_residual import (
    prepare_functional_residual,
    prepared_residual_jacobians,
    prepared_residual_loss_and_flat_gradient,
    prepared_term_residual_vector,
)
from phydrax.solver._kfac_layout import discover_parameter_layout
from phydrax.solver._kfac_problem import (
    term_block_curvature_observations,
    validate_derivative_coverage,
)


def _zero_model(model):
    replacements = tuple(
        (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias))
        for layer in model.layers
    )
    for index, (weight, bias) in enumerate(replacements):
        model = eqx.tree_at(lambda item: item.layers[index].weight, model, weight)
        model = eqx.tree_at(lambda item: item.layers[index].bias, model, bias)
    return model


def _residual_term(domain, fields, operator, *, samples, density=None):
    condition = phx.conditions.Residual(fields, domain.component(), operator)
    return phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(samples),
        ),
        density=density,
    )


def _prepare_residual(solver, params, non_trainable, *, key):
    prepared = solver.objective.prepare_training(
        range(len(solver.terms)),
        scale=1.0,
        evaluation_key=key,
        sampling_key=key,
        iteration=1,
    )
    return prepare_functional_residual(
        prepared,
        params,
        non_trainable,
        solver.enforcement,
        require_all=True,
    )


def test_type_two_residual_curvature_is_nonzero_at_zero_residual():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = _zero_model(
        phx.nn.models.MLP(
            in_size=1,
            out_size="scalar",
            hidden_sizes=(),
            rwf=False,
            key=jr.key(4),
        )
    )
    u = domain.Model("x")(model)
    term = _residual_term(domain, "u", lambda field: field, samples=7)
    solver = phx.solver.FunctionalSolver(functions={"u": u}, terms=term)
    params, non_trainable = solver.partition_functions()
    residual_map = _prepare_residual(solver, params, non_trainable, key=jr.key(5))
    flat, jacobians, _ = prepared_residual_jacobians(residual_map, params)
    residual = prepared_term_residual_vector(
        params,
        non_trainable,
        solver.enforcement,
        residual_map.terms[0],
        iteration=1,
    )

    assert flat.size > 0
    assert jnp.allclose(residual, 0.0, atol=1e-12)
    assert jnp.sum(jnp.square(jacobians[0])) > 0.0


def test_frozen_loss_uses_nonnegative_quadratic_coefficients():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(33),
    )

    @domain.Function("x")
    def zero_density(x):
        return jnp.zeros_like(x[0])

    term = _residual_term(
        domain,
        "u",
        lambda field: field,
        samples=5,
        density=zero_density,
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": domain.Model("x")(model)},
        terms=term,
    )
    params, non_trainable = solver.partition_functions()
    residual_map = _prepare_residual(solver, params, non_trainable, key=jr.key(34))
    loss, gradient, _ = prepared_residual_loss_and_flat_gradient(
        params,
        non_trainable,
        solver.enforcement,
        residual_map.terms,
        iteration=jnp.asarray(1.0),
    )
    residual = prepared_term_residual_vector(
        params,
        non_trainable,
        solver.enforcement,
        residual_map.terms[0],
        iteration=jnp.asarray(1.0),
    )

    assert jnp.allclose(loss, 0.0)
    assert jnp.allclose(gradient, 0.0)
    assert jnp.allclose(residual, 0.0)


def test_hard_enforced_ansatz_has_finite_residual_curvature():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(3,),
        rwf=False,
        key=jr.key(6),
    )
    raw = domain.Model("x")(model)
    boundary = domain.component({"x": Boundary()})
    spec = phx.enforcement.EnforcementSpec(
        phx.conditions.Dirichlet("u", boundary, target=0.0)
    )
    term = _residual_term(domain, "u", lambda field: field, samples=6)
    enforcement = phx.enforcement.compile(
        {"u": raw},
        (spec,),
        options=phx.enforcement.EnforcementOptions(num_reference=64),
        key=jr.key(6),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": raw},
        terms=term,
        enforcement=enforcement,
    )
    params, non_trainable = solver.partition_functions()
    validate_derivative_coverage(
        solver.terms,
        solver.enforcement.apply(solver.functions),
    )
    residual_map = _prepare_residual(solver, params, non_trainable, key=jr.key(7))
    _, jacobians, _ = prepared_residual_jacobians(residual_map, params)

    assert jnp.all(jnp.isfinite(jacobians[0]))
    assert jnp.linalg.norm(jacobians[0]) > 0.0


@pytest.mark.parametrize("approximation", ("expand", "reduce"))
def test_streamed_block_observations_match_dense_jacobian_oracle(approximation):
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(3,),
        rwf=False,
        key=jr.key(8),
    )
    functions = {
        "u": domain.Model("x")(model),
        "coefficient": domain.Parameter(0.7),
    }
    term = _residual_term(
        domain,
        ("u", "coefficient"),
        lambda field, coefficient: coefficient * field,
        samples=5,
    )
    solver = phx.solver.FunctionalSolver(functions=functions, terms=term)
    params, non_trainable = solver.partition_functions()
    layout = discover_parameter_layout(
        functions,
        params,
        exact_block_max_size=64,
        uncovered="error",
    )
    residual_map = _prepare_residual(solver, params, non_trainable, key=jr.key(9))
    flat, jacobians, _ = prepared_residual_jacobians(residual_map, params)
    streamed_flat, observations = term_block_curvature_observations(
        params,
        non_trainable,
        solver,
        residual_map.terms,
        layout,
        approximation=approximation,
        chunk_size=2,
        iter_=1,
    )
    initial = initialize_block_state(
        layout,
        num_terms=1,
        dtype=flat.dtype,
    )
    dense_state = update_block_state(
        initial,
        layout,
        jacobians,
        approximation=approximation,
        factor_decay=0.0,
    )
    streamed_state = update_block_state_from_observations(
        initial,
        observations,
        factor_decay=0.0,
    )

    assert jnp.array_equal(streamed_flat, flat)
    for streamed, dense in zip(
        jax.tree_util.tree_leaves(streamed_state),
        jax.tree_util.tree_leaves(dense_state),
        strict=True,
    ):
        assert jnp.allclose(streamed, dense, rtol=1e-9, atol=1e-10)


def test_kfac_derivative_coverage_rejects_orders_above_two():
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(1.0)
    term = _residual_term(
        domain,
        "u",
        lambda value: phx.operators.partial_n(value, var="x", order=3),
        samples=4,
    )

    with pytest.raises(ValueError, match="through order two"):
        validate_derivative_coverage((term,), {"u": field})
