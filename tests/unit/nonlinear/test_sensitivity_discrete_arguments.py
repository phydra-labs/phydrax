# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._strict import StrictModule
from phydrax.linalg import ArraySpace
from phydrax.nonlinear import NonlinearSystemProblem, root_solution_vjp


@final
class _Arguments(StrictModule):
    target: Array
    enabled: Array
    category: Array
    label: str = eqx.field(static=True)


def _residual(state: Array, arguments: _Arguments, /) -> Array:
    return state - arguments.target


def _valid(
    state: Array, residual: Array, auxiliary: None, arguments: _Arguments, /
) -> Array:
    del state, residual, auxiliary
    return arguments.enabled & (arguments.category >= 0)


@pytest.mark.strict_jax
@pytest.mark.parametrize("primal_valid", (True, False), ids=("root", "invalid-primal"))
def test_root_vjp_preserves_discrete_argument_float0_roles(primal_valid: bool) -> None:
    target = jnp.asarray((2.0, 3.0), dtype=jnp.float64)
    arguments = _Arguments(
        target,
        jnp.asarray(True, dtype=jnp.bool_),
        jnp.asarray(7, dtype=jnp.int32),
        "discrete-root-metadata",
    )
    space = ArraySpace((2,), dtype=jnp.float64)
    problem = NonlinearSystemProblem(
        _residual,
        state_space=space,
        residual_space=space,
        validity=_valid,
        problem_id="root-with-discrete-arguments",
    )
    state = target if primal_valid else target + 1.0
    result = root_solution_vjp(problem, state, arguments, jnp.ones_like(target))
    assert bool(result.evidence.successful) == primal_valid
    gradient = result.value
    if not isinstance(gradient, _Arguments):
        raise TypeError("The argument cotangent structure changed.")
    if primal_valid:
        np.testing.assert_allclose(
            gradient.target, np.ones(2, dtype=np.float64), atol=1e-14
        )
    else:
        assert jnp.all(jnp.isnan(gradient.target))
    assert gradient.enabled.dtype == jax.dtypes.float0
    assert gradient.category.dtype == jax.dtypes.float0
    assert gradient.enabled.shape == arguments.enabled.shape
    assert gradient.category.shape == arguments.category.shape
    assert gradient.label == arguments.label
