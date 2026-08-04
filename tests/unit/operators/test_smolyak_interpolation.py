#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._numerics import smolyak_axis_data
from phydrax._trainable import partition_trainable


def _square_domain():
    x = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(-1.0, 1.0, label="y")
    return phx.domain.ProductDomain(x, y)


def test_polynomial_is_exact_for_scalar_vector_matrix_and_complex_outputs():
    domain = _square_domain()

    def target(x, y):
        polynomial = x**2 + x * y + y**3
        return jnp.asarray(
            [
                [polynomial + 1j * x, 2.0 * polynomial - 1j * y],
                [x - y, polynomial**2],
            ]
        )

    function = domain.Function("x", "y")(target)
    approximation = phx.operators.interpolate_smolyak(
        function,
        phx.operators.SmolyakInterpolationPlan(2, 7),
    )
    query = {"x": jnp.asarray(0.23), "y": jnp.asarray(-0.41)}

    assert isinstance(approximation.func, phx.operators.SmolyakInterpolant)
    assert approximation.func.output_shape == (2, 2)
    assert jnp.allclose(approximation(query).data, function(query).data, atol=1e-11)


def test_named_batched_evaluation_jit_and_dependency_order_are_preserved():
    domain = _square_domain()
    function = domain.Function("y", "x")(lambda y, x: y**3 + x * y + 2.0 * x)
    approximation = phx.operators.interpolate_smolyak(
        function,
        phx.operators.SmolyakInterpolationPlan(
            2,
            5,
            anisotropy=(0.75, 1.25),
        ),
    )
    batch = domain.component().sample(
        9,
        structure=phx.domain.ProductStructure((("x", "y"),)),
        key=jr.key(2),
    )
    evaluated = eqx.filter_jit(approximation)(batch)
    expected = function(batch)

    assert approximation.deps == ("y", "x")
    assert approximation.func.axis_labels == ("y", "x")
    assert evaluated.dims == expected.dims
    assert jnp.allclose(evaluated.data, expected.data, atol=1e-11)


def test_first_and_second_derivatives_are_exact_at_and_near_nodes():
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    function = interval.Function("x")(lambda x: x**4 - 2.0 * x**2 + x)
    approximation = phx.operators.interpolate_smolyak(
        function,
        phx.operators.SmolyakInterpolationPlan(1, 5, axis_rules="leja"),
    )

    def evaluated(x):
        return approximation({"x": x}).data

    nodes = jnp.asarray(smolyak_axis_data("leja", 4).nodes)
    first = jax.vmap(jax.grad(evaluated))(nodes)
    second = jax.vmap(jax.grad(jax.grad(evaluated)))(nodes)
    near_nodes = nodes + 1e-13

    assert jnp.all(jnp.isfinite(first))
    assert jnp.all(jnp.isfinite(second))
    assert jnp.allclose(first, 4.0 * nodes**3 - 4.0 * nodes + 1.0, atol=1e-10)
    assert jnp.allclose(second, 12.0 * nodes**2 - 4.0, atol=1e-9)
    assert jnp.allclose(
        jax.vmap(evaluated)(near_nodes),
        near_nodes**4 - 2.0 * near_nodes**2 + near_nodes,
        atol=1e-11,
    )


def test_auto_rules_support_uniform_normal_and_lognormal_reference_coordinates():
    uniform = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="u")
    normal = phx.domain.ProbabilityDomain(phx.uq.Normal(2.0, 3.0), label="z")
    lognormal = phx.domain.ProbabilityDomain(phx.uq.LogNormal(0.1, 0.2), label="l")
    domain = phx.domain.ProductDomain(uniform, normal, lognormal)

    def target(u, z, l):
        zr = (z - 2.0) / 3.0
        lr = (jnp.log(l) - 0.1) / 0.2
        return u**2 + u * zr + lr**3

    function = domain.Function("u", "z", "l")(target)
    approximation = phx.operators.interpolate_smolyak(
        function,
        phx.operators.SmolyakInterpolationPlan(3, 5),
    )
    query = {
        "u": jnp.asarray(0.3),
        "z": jnp.asarray(0.8),
        "l": jnp.exp(jnp.asarray(0.1 + 0.2 * 0.7)),
    }

    assert approximation.func.axis_rules == (
        "leja",
        "gauss-hermite",
        "gauss-hermite",
    )
    assert jnp.allclose(approximation(query).data, function(query).data, atol=1e-11)


def test_interpolant_is_fixed_state_and_does_not_retain_source_callable():
    class CountingCallable:
        def __init__(self):
            self.count = 0

        def __call__(self, x, *, key, iter_=None):
            del key, iter_
            self.count += 1
            return x**2 + 1.0

    source = CountingCallable()
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    function = interval.Function("x")(source)
    approximation = phx.operators.interpolate_smolyak(
        function,
        phx.operators.SmolyakInterpolationPlan(1, 3),
    )
    fit_count = source.count
    trainable, _ = partition_trainable(approximation)
    leaves = jax.tree_util.tree_leaves(trainable)

    assert fit_count == 1
    assert not any(eqx.is_inexact_array(leaf) for leaf in leaves)
    assert not any(leaf is source for leaf in jax.tree_util.tree_leaves(approximation))
    assert approximation({"x": jnp.asarray(0.4)}).data == pytest.approx(1.16)
    assert source.count == fit_count


def test_interpolation_preserves_unused_domain_factors():
    x = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    y = phx.domain.ScalarInterval(0.0, 2.0, label="y")
    domain = phx.domain.ProductDomain(x, y)
    function = domain.Function("x")(lambda x: x**3)
    approximation = phx.operators.interpolate_smolyak(
        function,
        phx.operators.SmolyakInterpolationPlan(1, 4),
    )

    assert approximation.domain is domain
    assert approximation.deps == ("x",)
    assert approximation(
        {"x": jnp.asarray(0.3), "y": jnp.asarray(1.7)}
    ).data == pytest.approx(
        0.3**3,
        abs=1e-12,
    )


def test_stochastic_fitting_is_reproducible_for_the_same_key():
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    function = interval.Function("x")(lambda x, *, key: x**2 + 0.1 * jr.normal(key))
    plan = phx.operators.SmolyakInterpolationPlan(1, 4)
    first = phx.operators.interpolate_smolyak(function, plan, key=jr.key(7))
    second = phx.operators.interpolate_smolyak(function, plan, key=jr.key(7))
    third = phx.operators.interpolate_smolyak(function, plan, key=jr.key(8))
    query = {"x": jnp.asarray(0.2)}

    assert first(query).data == second(query).data
    assert first(query).data != third(query).data


def test_interpolation_rejects_invalid_domains_rules_and_source_values():
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    function = interval.Function("x")(lambda x: x)

    with pytest.raises(ValueError, match="dimension=2"):
        phx.operators.interpolate_smolyak(
            function,
            phx.operators.SmolyakInterpolationPlan(2, 2),
        )
    with pytest.raises(TypeError, match="requires a probability factor"):
        phx.operators.interpolate_smolyak(
            function,
            phx.operators.SmolyakInterpolationPlan(
                1,
                2,
                axis_rules="gauss-hermite",
            ),
        )
    with pytest.raises(ValueError, match="non-finite"):
        phx.operators.interpolate_smolyak(
            interval.Function("x")(lambda x: jnp.inf * jnp.ones_like(x)),
            phx.operators.SmolyakInterpolationPlan(1, 2),
        )

    empirical = phx.domain.ProbabilityDomain(
        phx.uq.EmpiricalDistribution(jnp.asarray([0.0, 1.0])),
        label="e",
    )
    with pytest.raises(ValueError, match="no canonical reference transform"):
        phx.operators.interpolate_smolyak(
            empirical.Function("e")(lambda e: e),
            phx.operators.SmolyakInterpolationPlan(1, 2),
        )


def test_plan_validation_and_fitted_diagnostics_are_explicit():
    with pytest.raises(ValueError, match="one rule per dimension"):
        phx.operators.SmolyakInterpolationPlan(2, 3, axis_rules=("leja",))
    with pytest.raises(ValueError, match="Unsupported interpolation axis rule"):
        phx.operators.SmolyakInterpolationPlan(1, 3, axis_rules="unknown")

    domain = _square_domain()
    approximation = phx.operators.interpolate_smolyak(
        domain.Function("x", "y")(lambda x, y: x + y),
        phx.operators.SmolyakInterpolationPlan(2, 4),
    )

    assert approximation.func.num_terms > 0
    assert approximation.func.num_evaluations == approximation.func.num_unique_nodes
    assert approximation.func.maximum_active_dimension <= 2
    assert approximation.func.num_blocks > 0
