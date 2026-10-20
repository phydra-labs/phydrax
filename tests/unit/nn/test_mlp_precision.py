#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit `Linear`/`MLP` parameter dtype with x64 enabled.

The float32 contract is checked where it can actually break: every allocated
parameter role, the traced construction/derivative/update programs (no float64
array may appear anywhere, which is what an FP32-only device admits), and
higher derivatives against a float64 evaluation of the *same* float32
parameters.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.core import ShapedArray
from jax.extend.core import Jaxpr, jaxprs_in_params

from phydrax._trainable import (
    combine_parameters,
    partition_parameters,
    require_parameter_roles,
)
from phydrax.nn.layers import Linear
from phydrax.nn.models import MLP
from phydrax.operators.differential import laplacian
from phydrax.optim import adam
from tests._support.metal_pde import (
    build_model,
    optimizer,
    point_mapping,
    poisson_loss,
    poisson_points,
    train_step,
    unit_square,
)


# Hidden/final RWF and bias, no bias, residual projection, zero hidden layers,
# homogeneous scan, and heterogeneous (non-scan) widths.
_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "rwf-bias": {"width_size": 8, "depth": 2, "rwf": True},
    "no-bias": {"width_size": 8, "depth": 2, "use_bias": False, "use_final_bias": False},
    "projection": {"width_size": 8, "depth": 2, "skip_connection": True},
    "zero-hidden": {"width_size": 8, "depth": 0, "skip_connection": True, "rwf": True},
    "scan": {"width_size": 8, "depth": 4, "scan": True, "rwf": True},
    "heterogeneous": {"hidden_sizes": (5, 7), "skip_connection": True, "rwf": True},
}
_IN_SIZE = 2
# float32 unit roundoff is 6e-8; 1e-4 allows a 1e3 amplification through the
# depth-four second-derivative/gradient chains while an O(1) derivative defect
# (wrong role dtype, dropped term, detached parameter) remains far outside it.
_FLOAT32_RELATIVE_BOUND = 1e-4
_UNUSABLE_KEY: Any = None


def _mlp(name: str, /, **kwargs: Any) -> MLP:
    return MLP(
        in_size=_IN_SIZE,
        out_size="scalar",
        key=jr.key(17),
        **_CONFIGURATIONS[name],
        **kwargs,
    )


def _inexact_leaves(tree: Any, /) -> list[jax.Array]:
    return jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))


def _upcast(tree: Any, /) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.float64) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def _laplacian(model: Any, points: jax.Array, /) -> jax.Array:
    return jax.vmap(lambda point: jnp.trace(jax.hessian(model)(point)))(points)


def _pde_loss(model: Any, points: jax.Array, /) -> jax.Array:
    # Poisson residual plus a value penalty: the value term keeps the affine
    # zero-hidden network (identically zero Laplacian) a real gradient consumer.
    residual = _laplacian(model, points) - 1.0
    values = jax.vmap(model)(points)
    return jnp.mean(residual**2) + jnp.mean(values**2)


def _points(dtype: Any, /) -> jax.Array:
    return jr.uniform(jr.key(3), (5, _IN_SIZE), dtype=dtype, minval=-1.0, maxval=1.0)


def _relative_error(actual: Any, reference: Any, /) -> float:
    actual_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in _inexact_leaves(actual)])
    reference_flat = jnp.concatenate(
        [jnp.ravel(leaf) for leaf in _inexact_leaves(reference)]
    )
    difference = actual_flat.astype(jnp.float64) - reference_flat
    return float(jnp.linalg.norm(difference) / jnp.linalg.norm(reference_flat))


def _float64_avals(jaxpr: Jaxpr, /) -> list[str]:
    """Every float64/complex128 value in a traced program, including sub-programs."""
    found: list[str] = []

    def check(where: str, dtype: Any) -> None:
        if str(dtype) in ("float64", "complex128"):
            found.append(f"{where}: {dtype}")

    def visit(inner: Jaxpr) -> None:
        for const in inner.consts:
            check("const", getattr(const, "dtype", None))
        for var in (*inner.all_invars, *inner.outvars):
            if isinstance(var.aval, ShapedArray):
                check("jaxpr variable", var.aval.dtype)
        for equation in inner.eqns:
            name = equation.primitive.name
            for atom in (*equation.invars, *equation.outvars):
                if isinstance(atom.aval, ShapedArray):
                    check(name, atom.aval.dtype)
            for nested in jaxprs_in_params(equation.params):
                visit(nested)

    visit(jaxpr)
    return found


def _assert_no_float64(jaxpr: Any, /) -> None:
    assert _float64_avals(jaxpr) == []


def _train_step(parameters: Any, rest: tuple[Any, Any], points: jax.Array) -> Any:
    state, fixed = rest

    def objective(candidate: Any) -> jax.Array:
        return _pde_loss(combine_parameters(candidate, state, fixed), points)

    loss, gradient = eqx.filter_value_and_grad(objective)(parameters)
    optimizer = adam(1e-3)
    updates, _ = optimizer.update(gradient, optimizer.init(parameters), parameters)
    return loss, gradient, eqx.apply_updates(parameters, updates)


@pytest.mark.parametrize("name", tuple(_CONFIGURATIONS))
def test_explicit_dtype_reaches_every_parameter_role(name: str) -> None:
    default = _mlp(name)
    explicit32 = _mlp(name, dtype=jnp.float32)
    explicit64 = _mlp(name, dtype="float64")

    # Default construction is unchanged: same keys, values and x64 dtype.
    assert all(leaf.dtype == jnp.float64 for leaf in _inexact_leaves(default))
    for leaf, reference in zip(
        _inexact_leaves(explicit64), _inexact_leaves(default), strict=True
    ):
        assert leaf.dtype == reference.dtype
        assert jnp.array_equal(leaf, reference)

    # Same PyTree, roles and shapes; every role (weights, RWF scales, biases,
    # projection) is allocated in float32.
    assert jax.tree.structure(explicit32) == jax.tree.structure(default)
    roles32 = require_parameter_roles(explicit32, context="float32 MLP")
    roles = require_parameter_roles(default, context="default MLP")
    assert roles32.paths == roles.paths
    assert roles32.roles == roles.roles
    leaves32 = _inexact_leaves(explicit32)
    assert leaves32
    for leaf, reference in zip(leaves32, _inexact_leaves(default), strict=True):
        assert leaf.dtype == jnp.float32
        assert leaf.shape == reference.shape
        assert bool(jnp.all(jnp.isfinite(leaf)))
    if name == "scan":
        assert explicit32._scan_enabled

    value = explicit32(_points(jnp.float32)[0])
    assert value.dtype == jnp.float32
    assert value.shape == ()


def test_linear_dtype_spellings_are_one_canonical_dtype() -> None:
    def leaves(dtype: Any) -> list[jax.Array]:
        return _inexact_leaves(
            Linear(in_size=3, out_size=4, rwf=True, dtype=dtype, key=jr.key(5))
        )

    canonical = leaves("float32")
    assert len(canonical) == 3  # weight, RWF log-scales, bias
    for spelling in (jnp.float32, np.float32, np.dtype("float32")):
        for leaf, reference in zip(leaves(spelling), canonical, strict=True):
            assert leaf.dtype == jnp.float32
            assert jnp.array_equal(leaf, reference)


@pytest.mark.parametrize(
    ("dtype", "message"),
    (
        (jnp.int32, "'int32'"),
        (jnp.uint64, "'uint64'"),
        ("bool", "'bool'"),
        (jnp.complex64, "not real floating-point"),
    ),
)
def test_non_real_floating_dtype_refuses_before_key_use(dtype: Any, message: str) -> None:
    # An unusable key proves refusal happens before any key split/allocation.
    with pytest.raises(ValueError, match=message):
        Linear(in_size=2, out_size=3, dtype=dtype, key=_UNUSABLE_KEY)
    with pytest.raises(ValueError, match=message):
        MLP(
            in_size=2,
            out_size="scalar",
            width_size=4,
            depth=1,
            skip_connection=True,
            dtype=dtype,
            key=_UNUSABLE_KEY,
        )


@pytest.mark.parametrize("name", tuple(_CONFIGURATIONS))
def test_float32_construction_derivatives_and_update_trace_without_float64(
    name: str,
) -> None:
    construction, _, _ = eqx.filter_make_jaxpr(
        lambda key: eqx.filter(
            MLP(
                in_size=_IN_SIZE,
                out_size="scalar",
                dtype=jnp.float32,
                key=key,
                **_CONFIGURATIONS[name],
            ),
            eqx.is_array,
        )
    )(jr.key(0))
    _assert_no_float64(construction)

    model = _mlp(name, dtype=jnp.float32)
    parameters, state, fixed = partition_parameters(model)
    step, _, _ = eqx.filter_make_jaxpr(_train_step)(
        parameters, (state, fixed), _points(jnp.float32)
    )
    _assert_no_float64(step)


@pytest.mark.parametrize("name", tuple(_CONFIGURATIONS))
def test_float32_second_derivative_gradient_and_update_match_float64_evaluation(
    name: str,
) -> None:
    model = _mlp(name, dtype=jnp.float32)
    points = _points(jnp.float32)
    # The reference evaluates the same float32 parameters and points in float64.
    reference_model = _upcast(model)
    reference_points = points.astype(jnp.float64)

    laplacian = _laplacian(model, points)
    reference_laplacian = _laplacian(reference_model, reference_points)
    assert laplacian.dtype == jnp.float32
    assert reference_laplacian.dtype == jnp.float64
    if name == "zero-hidden":
        # An affine network has an exactly zero coordinate Laplacian.
        assert bool(jnp.all(laplacian == 0.0))
    else:
        assert bool(jnp.any(laplacian != 0.0))
        assert _relative_error(laplacian, reference_laplacian) <= _FLOAT32_RELATIVE_BOUND

    parameters, state, fixed = partition_parameters(model)
    loss, gradient, updated = eqx.filter_jit(_train_step)(
        parameters, (state, fixed), points
    )
    reference_parameters, reference_state, reference_fixed = partition_parameters(
        reference_model
    )
    reference_loss, reference_gradient, _ = _train_step(
        reference_parameters, (reference_state, reference_fixed), reference_points
    )
    assert loss.dtype == jnp.float32
    assert abs(float(loss) - float(reference_loss)) <= _FLOAT32_RELATIVE_BOUND * abs(
        float(reference_loss)
    )
    assert all(leaf.dtype == jnp.float32 for leaf in _inexact_leaves(gradient))
    assert _relative_error(gradient, reference_gradient) <= _FLOAT32_RELATIVE_BOUND

    # The optimizer step keeps every parameter float32 and actually moves it.
    moved = False
    for new, old in zip(
        _inexact_leaves(updated), _inexact_leaves(parameters), strict=True
    ):
        assert new.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(new)))
        moved = moved or bool(jnp.any(new != old))
    assert moved
    trained = combine_parameters(updated, state, fixed)
    assert _pde_loss(trained, points).dtype == jnp.float32


def test_active_dropout_keeps_float32_values() -> None:
    model = MLP(
        in_size=_IN_SIZE,
        out_size=3,
        width_size=8,
        depth=2,
        dropout=0.5,
        dtype=jnp.float32,
        key=jr.key(9),
    )
    value = model(_points(jnp.float32)[0], key=jr.key(10))
    assert value.dtype == jnp.float32
    assert value.shape == (3,)


def test_float32_poisson_domain_step_is_float32_and_matches_float64() -> None:
    # The Metal Poisson workload on CPU: no float64 value anywhere in the compiled
    # step, and the same float32 parameters evaluated in float64 agree with it.
    host_interior, host_boundary = poisson_points(np.float32)
    interior, boundary = jnp.asarray(host_interior), jnp.asarray(host_boundary)
    box = unit_square(jnp.float32)
    model = build_model()
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    step = train_step(box, static, interior, boundary)
    state = optimizer().init(parameters)

    program, _, _ = eqx.filter_make_jaxpr(step)(parameters, state)
    _assert_no_float64(program)

    # The contracted coordinate Laplacian is the trace of the Hessian.
    field = box.Model("x")(model)
    contracted = laplacian(field, var="x")(point_mapping(interior)).data
    assert contracted.dtype == jnp.float32
    assert _relative_error(contracted, _laplacian(model, interior)) <= 1e-5

    loss, gradient, updated, _ = step(parameters, state)
    reference_loss, reference_gradient = eqx.filter_value_and_grad(
        lambda candidate: poisson_loss(
            eqx.combine(candidate, static),
            unit_square(jnp.float64),
            interior.astype(jnp.float64),
            boundary.astype(jnp.float64),
        )
    )(_upcast(parameters))
    assert loss.dtype == jnp.float32
    assert abs(float(loss) - float(reference_loss)) <= _FLOAT32_RELATIVE_BOUND * float(
        reference_loss
    )
    assert _relative_error(gradient, reference_gradient) <= _FLOAT32_RELATIVE_BOUND
    assert all(leaf.dtype == jnp.float32 for leaf in _inexact_leaves(updated))
    next_loss = poisson_loss(eqx.combine(updated, static), box, interior, boundary)
    assert float(next_loss) < float(loss)
