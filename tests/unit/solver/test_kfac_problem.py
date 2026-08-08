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
from phydrax.enforcement import enforce_dirichlet
from phydrax.optim._kfac._blocks import (
    initialize_block_state,
    update_block_state,
    update_block_state_from_observations,
)
from phydrax.solver._kfac_layout import discover_parameter_layout
from phydrax.solver._kfac_problem import (
    frozen_loss_and_flat_gradient,
    frozen_term_residual_vector,
    materialize_frozen_terms,
    term_block_curvature_observations,
    term_residual_jacobians,
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


def test_type_two_residual_curvature_is_nonzero_at_zero_residual():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = _zero_model(
        phx.nn.MLP(
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
    terms = materialize_frozen_terms(
        solver.terms,
        solver.collocation,
        key=jr.key(5),
    )
    flat, jacobians, _ = term_residual_jacobians(
        params,
        non_trainable,
        solver,
        terms,
        iter_=1,
    )
    residual = frozen_term_residual_vector(
        params,
        non_trainable,
        solver,
        terms[0],
        iter_=1,
    )

    assert flat.size > 0
    assert jnp.allclose(residual, 0.0, atol=1e-12)
    assert jnp.sum(jnp.square(jacobians[0])) > 0.0


def test_frozen_loss_uses_nonnegative_quadratic_coefficients():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
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
    terms = materialize_frozen_terms(
        solver.terms,
        solver.collocation,
        key=jr.key(34),
    )
    loss, gradient, _ = frozen_loss_and_flat_gradient(
        params,
        non_trainable,
        solver,
        terms,
        iter_=jnp.asarray(1.0),
    )
    residual = frozen_term_residual_vector(
        params,
        non_trainable,
        solver,
        terms[0],
        iter_=jnp.asarray(1.0),
    )

    assert jnp.allclose(loss, 0.0)
    assert jnp.allclose(gradient, 0.0)
    assert jnp.allclose(residual, 0.0)


def test_hard_enforced_ansatz_has_finite_residual_curvature():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(3,),
        rwf=False,
        key=jr.key(6),
    )
    raw = domain.Model("x")(model)
    boundary = domain.component({"x": Boundary()})
    spec = phx.enforcement.EnforcementSpec(
        phx.conditions.Dirichlet("u", boundary, target=0.0),
        kind="custom",
        transform=lambda field, _get_field: enforce_dirichlet(
            field,
            boundary,
            var="x",
            target=0.0,
        ),
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
    terms = materialize_frozen_terms(
        solver.terms,
        solver.collocation,
        key=jr.key(7),
    )
    _, jacobians, _ = term_residual_jacobians(
        params,
        non_trainable,
        solver,
        terms,
        iter_=1,
    )

    assert jnp.all(jnp.isfinite(jacobians[0]))
    assert jnp.linalg.norm(jacobians[0]) > 0.0


@pytest.mark.parametrize("approximation", ("expand", "reduce"))
def test_streamed_block_observations_match_dense_jacobian_oracle(approximation):
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.MLP(
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
    terms = materialize_frozen_terms(
        solver.terms,
        solver.collocation,
        key=jr.key(9),
    )
    flat, jacobians, _ = term_residual_jacobians(
        params,
        non_trainable,
        solver,
        terms,
        iter_=1,
    )
    streamed_flat, observations = term_block_curvature_observations(
        params,
        non_trainable,
        solver,
        terms,
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
