from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.extend import core

from phydrax._identity import execution_metadata_payload
from phydrax.backends._result_storage import _publish_p, _reserve_p
from phydrax.discretization.particle._articulated_kinematics import (
    _motion_p,
    _poses_p,
    _velocity_p,
    _velocity_transpose_p,
)
from phydrax.discretization.particle._articulated_sweeps import (
    _factor_p,
    _forces_p,
    _solve_p,
    factor_articulated_inertia,
    reduce_articulated_forces,
    solve_articulated_factor,
)
from phydrax.discretization.particle._reduced_dynamics import reduced_forward_dynamics
from phydrax.sparse._structural_trace import trace_sparse_pattern
from tests._support.native_articulated import slider_model


pytestmark = pytest.mark.strict_jax


def _tree(chain: bool) -> tuple[Array, Array, Array, Array]:
    transforms = jnp.broadcast_to(jnp.eye(6, dtype=jnp.float64), (2, 6, 6))
    subspaces = jnp.zeros((2, 6), dtype=jnp.float64).at[:, 0].set(1.0)
    parents = jnp.asarray([0, 1] if chain else [0, 0], dtype=jnp.int32)
    children = jnp.asarray([1, 2], dtype=jnp.int32)
    return transforms, subspaces, parents, children


def _response(values: Array, chain: bool) -> Array:
    transforms, subspaces, parents, children = _tree(chain)
    spatial = jnp.broadcast_to(jnp.eye(6, dtype=jnp.float64), (3, 6, 6))
    articulated, projected, _, _, reciprocal = factor_articulated_inertia(
        spatial, transforms, subspaces, jnp.ones((2,), dtype=jnp.bool_), parents, children
    )
    acceleration = solve_articulated_factor(
        articulated,
        transforms,
        subspaces,
        projected,
        reciprocal,
        parents,
        children,
        values[:, None],
        None,
        None,
    )
    force = jnp.zeros((3, 6, 1), dtype=jnp.float64).at[1:, 0, 0].set(values)
    effort = reduce_articulated_forces(transforms, subspaces, parents, children, force)
    return jnp.concatenate((acceleration[:, 0], effort[:, 0]))


def test_sweep_execution_identity_binds_topology_not_dynamic_inputs() -> None:
    values = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    branch = lambda value: _response(value, False)
    chain = lambda value: _response(value, True)
    identity = execution_metadata_payload(jax.make_jaxpr(branch)(values))
    assert identity == execution_metadata_payload(jax.make_jaxpr(branch)(values * 2))
    assert identity != execution_metadata_payload(jax.make_jaxpr(chain)(values))
    np.testing.assert_allclose(jax.jit(branch)(values), [2.0, 3.0, 2.0, 3.0])
    np.testing.assert_allclose(jax.jit(chain)(values), [-1.0, 4.0, 5.0, 3.0])


@pytest.mark.parametrize(
    "owned",
    (
        _factor_p,
        _solve_p,
        _forces_p,
        _poses_p,
        _velocity_p,
        _velocity_transpose_p,
        _motion_p,
        _reserve_p,
        _publish_p,
    ),
    ids=(
        "factor",
        "solve",
        "forces",
        "poses",
        "velocity",
        "velocity-transpose",
        "motion",
        "result-reservation",
        "result-publication",
    ),
)
def test_foreign_execution_primitive_name_requires_owned_object(
    owned: core.Primitive,
) -> None:
    foreign = core.Primitive(owned.name)
    foreign.def_impl(lambda value: value)
    foreign.def_abstract_eval(lambda value: value)
    with pytest.raises(TypeError, match="Foreign compiler primitive"):
        execution_metadata_payload(
            jax.make_jaxpr(foreign.bind)(jnp.ones((2,), dtype=jnp.float64))
        )


@pytest.mark.parametrize("chain", (False, True), ids=("branches", "chain"))
def test_force_sweep_sparse_jacobian_covers_tree_ancestry(chain: bool) -> None:
    transforms, subspaces, parents, children = _tree(chain)

    def function(values: Array, args: None) -> Array:
        del args
        force = jnp.zeros((3, 6, 1), dtype=jnp.float64).at[1:, 0, 0].set(values)
        return reduce_articulated_forces(transforms, subspaces, parents, children, force)[
            :, 0
        ]

    values = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    pattern = trace_sparse_pattern(
        function, values, None, source_size=2, target_size=2, derivative_kind="jacobian"
    )
    actual = set(
        zip(
            np.asarray(pattern.rows).tolist(),
            np.asarray(pattern.cols).tolist(),
            strict=True,
        )
    )
    expected = {(0, 0), (1, 1)} | ({(0, 1)} if chain else set())
    # Loop-index alternatives may overapproximate support, never omit it.
    assert expected <= actual
    np.testing.assert_array_equal(
        jax.jacfwd(function)(values, None), [[1.0, 1.0 if chain else 0.0], [0.0, 1.0]]
    )


def test_factor_solve_hessian_preserves_independent_branches() -> None:
    transforms, subspaces, parents, children = _tree(False)

    def function(values: Array, args: None) -> Array:
        del args
        spatial = (
            jnp.broadcast_to(jnp.eye(6, dtype=jnp.float64), (3, 6, 6))
            .at[1:, 0, 0]
            .set(values)
        )
        articulated, projected, _, _, reciprocal = factor_articulated_inertia(
            spatial,
            transforms,
            subspaces,
            jnp.ones((2,), dtype=jnp.bool_),
            parents,
            children,
        )
        acceleration = solve_articulated_factor(
            articulated,
            transforms,
            subspaces,
            projected,
            reciprocal,
            parents,
            children,
            jnp.ones((2, 1), dtype=jnp.float64),
            None,
            None,
        )
        return jnp.sum(acceleration)

    values = jnp.asarray([1.5, 2.0], dtype=jnp.float64)
    pattern = trace_sparse_pattern(
        function, values, None, source_size=2, target_size=2, derivative_kind="hessian"
    )
    actual = set(
        zip(
            np.asarray(pattern.rows).tolist(),
            np.asarray(pattern.cols).tolist(),
            strict=True,
        )
    )
    assert {(0, 0), (1, 1)} <= actual
    np.testing.assert_allclose(
        jax.hessian(function)(values, None),
        np.diag(2.0 / np.asarray(values) ** 3),
        rtol=1e-11,
        atol=1e-11,
    )


def _slider_query(mass: float) -> Callable[[Array, None], Array]:
    articulation = slider_model(mass=mass, floor=False).prepare().articulation
    body = articulation.dof_body_indices[0]

    def query(values: Array, args: None) -> Array:
        del args
        point, force = values[:1], values[1:]
        position = articulation.body_transforms(point)[body, 2, 3]
        result = reduced_forward_dynamics(
            articulation,
            point,
            jnp.zeros((1,), dtype=jnp.float64),
            force,
            jnp.zeros((3,), dtype=jnp.float64),
        )
        return jnp.stack((position, result.acceleration[0]))

    return query


def test_certified_kinematic_identity_preserves_dynamic_and_captured_roles() -> None:
    query = _slider_query(2.0)
    heavier = _slider_query(3.0)
    values = jnp.asarray([0.2, 4.0], dtype=jnp.float64)
    identity = execution_metadata_payload(jax.make_jaxpr(query)(values, None))
    assert identity == execution_metadata_payload(
        jax.make_jaxpr(query)(values + 0.1, None)
    )
    assert identity != execution_metadata_payload(jax.make_jaxpr(heavier)(values, None))
    np.testing.assert_allclose(jax.jit(query)(values, None), [0.3, 2.0], atol=1.0e-12)
    np.testing.assert_allclose(
        jax.jit(heavier)(values, None), [0.3, 4.0 / 3.0], atol=1.0e-12
    )


def test_certified_kinematic_sparse_support_covers_analytic_slider_actions() -> None:
    query = _slider_query(2.0)
    values = jnp.asarray([0.2, 4.0], dtype=jnp.float64)
    pattern = trace_sparse_pattern(
        query, values, None, source_size=2, target_size=2, derivative_kind="jacobian"
    )
    support = set(
        zip(
            np.asarray(pattern.rows).tolist(),
            np.asarray(pattern.cols).tolist(),
            strict=True,
        )
    )
    assert {(0, 0), (1, 1)} <= support
    np.testing.assert_allclose(
        jax.jacfwd(query)(values, None), [[1.0, 0.0], [0.0, 0.5]], atol=1.0e-12
    )
