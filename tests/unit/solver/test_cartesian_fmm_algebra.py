from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.operators.integral.multipole._cartesian_radial import (
    plummer_scaled_cartesian_derivatives,
)
from phydrax.solver import CartesianExpansionSpace, CartesianFMMOperators


def test_cartesian_expansion_layout_reaches_order_seven() -> None:
    expected = {1: 4, 2: 10, 3: 20, 5: 56, 7: 120}
    for order, count in expected.items():
        space = CartesianExpansionSpace(order)
        assert space.coefficient_count == count
        assert space.degrees == tuple(sum(exponent) for exponent in space.exponents)
        assert len(set(space.exponents)) == count
        for exponent, degree in zip(space.exponents, space.degrees, strict=True):
            assert degree <= order
            assert all(component >= 0 for component in exponent)


def test_plummer_cartesian_derivatives_match_jax() -> None:
    space = CartesianExpansionSpace(4)
    displacement = jnp.asarray([0.7, -0.4, 0.2], dtype=jnp.float64)
    softening = 0.13
    gravity = 1.7
    scale = jnp.asarray(2.0, dtype=displacement.dtype)
    derivatives = plummer_scaled_cartesian_derivatives(
        space.exponents,
        displacement,
        softening,
        gravity,
        scale,
    )

    def potential(value):
        return -gravity / jnp.sqrt(jnp.sum(value * value) + (softening / scale) ** 2)

    scaled = displacement / scale
    functions = {0: potential}
    tensors = {0: potential(scaled)}
    for degree in range(1, 5):
        functions[degree] = jax.jacfwd(functions[degree - 1])
        tensors[degree] = functions[degree](scaled)
    expected = []
    for exponent in space.exponents:
        axes = tuple(axis for axis, count in enumerate(exponent) for _ in range(count))
        expected.append(tensors[sum(exponent)][axes] if axes else tensors[0])
    np.testing.assert_allclose(derivatives, jnp.stack(expected), rtol=2e-11, atol=2e-11)


def test_scaled_m2m_matches_direct_parent_moments() -> None:
    operators = CartesianFMMOperators(CartesianExpansionSpace(5), 1.0, 0.03)
    positions = jnp.asarray(
        [[0.03, -0.02, 0.01], [0.11, 0.04, -0.03], [-0.07, 0.02, 0.05]]
    )
    masses = jnp.asarray([1.0, 2.0, 0.5])
    child_center = jnp.asarray([0.02, 0.01, -0.01])
    parent_center = jnp.asarray([0.5, -0.25, 0.125])
    child_scale = jnp.asarray(0.25)
    parent_scale = jnp.asarray(1.0)
    child = operators.p2m(
        positions,
        masses,
        child_center,
        scale=child_scale,
    )
    translated = operators.m2m(
        child,
        child_center - parent_center,
        source_scale=child_scale,
        target_scale=parent_scale,
    )
    direct = operators.p2m(
        positions,
        masses,
        parent_center,
        scale=parent_scale,
    )
    np.testing.assert_allclose(translated, direct, rtol=2e-11, atol=2e-11)


def test_scaled_l2l_preserves_polynomial_value_and_gradient() -> None:
    space = CartesianExpansionSpace(5)
    operators = CartesianFMMOperators(space, 1.0, 0.03)
    parent_scale = jnp.asarray(2.0)
    child_scale = jnp.asarray(0.5)
    shift = jnp.asarray([0.3, -0.2, 0.1])
    point_from_child = jnp.asarray([0.04, -0.02, 0.03])
    local = jnp.arange(space.coefficient_count, dtype=jnp.float64) / 50
    child = operators.l2l(
        local,
        shift,
        source_scale=parent_scale,
        target_scale=child_scale,
    )
    parent_value, parent_force = operators.l2p(
        local,
        shift + point_from_child,
        scale=parent_scale,
    )
    child_value, child_force = operators.l2p(
        child,
        point_from_child,
        scale=child_scale,
    )
    np.testing.assert_allclose(child_value, parent_value, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(child_force, parent_force, rtol=2e-11, atol=2e-11)


def test_m2l_force_converges_to_direct_plummer_force() -> None:
    source = jnp.asarray([[-0.08, 0.02, 0.01], [0.05, -0.04, 0.03], [0.02, 0.07, -0.02]])
    masses = jnp.asarray([1.0, 1.5, 0.75])
    source_center = jnp.zeros((3,))
    target_center = jnp.asarray([2.0, 1.5, -1.0])
    target = target_center + jnp.asarray([0.03, -0.02, 0.01])
    errors = []
    for order in (1, 3, 5):
        operators = CartesianFMMOperators(CartesianExpansionSpace(order), 1.0, 0.05)
        multipole = operators.p2m(
            source,
            masses,
            source_center,
            scale=jnp.asarray(0.125),
        )
        local = operators.m2l(
            multipole,
            source_center,
            target_center,
            source_scale=jnp.asarray(0.125),
            target_scale=jnp.asarray(0.125),
        )
        _, force = operators.l2p(
            local,
            target - target_center,
            scale=jnp.asarray(0.125),
        )
        direct = operators.p2p(target, source, masses)
        errors.append(float(jnp.sqrt(jnp.sum((force - direct) ** 2))))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1.0e-7
