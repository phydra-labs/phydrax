# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._strict import StrictModule
from phydrax.linalg import ArraySpace, LinearizationPolicy, prepare_linearization
from phydrax.linalg._linearizations import RematerializationPolicy
from phydrax.nonlinear import NewtonKrylov, NonlinearSystemProblem, NonlinearTermination


@final
class _Auxiliary(StrictModule):
    admissible: Array
    constant_vector: Array
    counter: Array
    literal_metadata: tuple[bool, int] = eqx.field(static=True)


@final
class _Equation(StrictModule):
    scale: Array
    target: Array

    def __call__(self, state: Array, /) -> tuple[Array, _Auxiliary]:
        return self.scale * (state**2 - self.target), _Auxiliary(
            jnp.asarray(True, dtype=jnp.bool_),
            jnp.asarray((2.0, 3.0), dtype=state.dtype),
            jnp.asarray(7, dtype=jnp.int32),
            (True, 7),
        )


@final
class _RootEquation(StrictModule):
    equation: _Equation

    def __call__(self, state: Array, args: None, /) -> tuple[Array, _Auxiliary]:
        del args
        return self.equation(state)


def _valid(state: Array, residual: Array, auxiliary: _Auxiliary, args: None, /) -> Array:
    del state, residual, args
    return auxiliary.admissible


def _trial_valid(state: Array, args: None, /) -> Array:
    del args
    return jnp.all(state > 0.0)


@pytest.mark.strict_jax
@pytest.mark.parametrize("storage", ("store", "rematerialize"))
def test_constant_auxiliary_arrays_preserve_schema_and_metadata(
    storage: RematerializationPolicy,
) -> None:
    point = jnp.asarray((1.0, 2.0), dtype=jnp.float64)
    equation = _Equation(jnp.asarray(2.0, dtype=jnp.float64), point**2)
    space = ArraySpace((2,), dtype=jnp.float64)
    prepared = prepare_linearization(
        equation,
        point,
        source=space,
        target=space,
        has_aux=True,
        policy=LinearizationPolicy(storage),
    )
    auxiliary = prepared.auxiliary
    if not isinstance(auxiliary, _Auxiliary):
        raise TypeError("The retained auxiliary contract changed.")
    assert isinstance(auxiliary.admissible, Array)
    assert isinstance(auxiliary.constant_vector, Array)
    assert isinstance(auxiliary.counter, Array)
    assert auxiliary.admissible.dtype == jnp.bool_
    assert auxiliary.counter.dtype == jnp.int32
    assert auxiliary.literal_metadata == (True, 7)
    assert isinstance(auxiliary.literal_metadata[0], bool)
    np.testing.assert_array_equal(auxiliary.constant_vector, (2.0, 3.0))
    direction = jnp.asarray((0.5, -0.25), dtype=jnp.float64)
    np.testing.assert_allclose(
        prepared.jvp(direction), 4.0 * point * direction, atol=1e-14
    )
    np.testing.assert_allclose(
        prepared.vjp(direction), 4.0 * point * direction, atol=1e-14
    )


@pytest.mark.strict_jax
def test_guarded_newton_preserves_constant_auxiliary_array_roles() -> None:
    space = ArraySpace((2,), dtype=jnp.float64)
    equation = _Equation(
        jnp.asarray(2.0, dtype=jnp.float64),
        jnp.asarray((1.0, 4.0), dtype=jnp.float64),
    )
    problem = NonlinearSystemProblem(
        _RootEquation(equation),
        state_space=space,
        residual_space=space,
        has_aux=True,
        validity=_valid,
        trial_validity=_trial_valid,
        trial_validity_id="positive-coordinate-domain",
        problem_id="constant-auxiliary-schema",
    )
    method = NewtonKrylov()
    termination = NonlinearTermination(absolute_residual=1e-12, relative_residual=1e-12)
    solve = eqx.filter_jit(method.solve)
    result = solve(problem, jnp.ones((2,), dtype=jnp.float64), termination=termination)
    assert bool(result.successful)
    np.testing.assert_allclose(result.state, (1.0, 2.0), atol=1e-12)
    auxiliary = result.auxiliary
    if not isinstance(auxiliary, _Auxiliary):
        raise TypeError("The guarded root changed the auxiliary contract.")
    assert isinstance(auxiliary.admissible, Array)
    assert isinstance(auxiliary.counter, Array)
    assert bool(auxiliary.admissible)
    assert auxiliary.literal_metadata == (True, 7)
