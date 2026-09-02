#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import isfinite
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array


PreconditionerMatrix = Array | None


@jax.tree_util.register_pytree_node_class
class SOAPPreconditioner:
    """Per-axis SOAP covariance or orthogonal-basis matrices for one parameter."""

    __slots__ = ("matrices",)

    matrices: tuple[PreconditionerMatrix, ...]

    def __init__(self, matrices: Iterable[PreconditionerMatrix], /):
        self.matrices = tuple(matrices)

    def tree_flatten(self):
        return self.matrices, None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        return cls(children)

    def map(
        self,
        function: Callable[[PreconditionerMatrix], PreconditionerMatrix],
        /,
    ) -> SOAPPreconditioner:
        return SOAPPreconditioner(function(matrix) for matrix in self.matrices)


class SOAPState(NamedTuple):
    """SOAP moments, axis covariances, and orthogonal preconditioner bases."""

    count: Array
    first_moment: Any
    second_moment: Any
    covariance: Any
    basis: Any


def _is_preconditioner(value: object, /) -> bool:
    return isinstance(value, SOAPPreconditioner)


def _real_floating_dtype(value: Any, /):
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError("SOAP requires real floating-point parameter leaves.")
    return array.dtype


def _state_dtype(parameter_dtype: Any, requested_dtype: Any, /):
    dtype = jnp.dtype(parameter_dtype if requested_dtype is None else requested_dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError("SOAP state dtypes must be real floating-point dtypes.")
    return dtype


def _initialize_preconditioner(
    parameter: Any,
    /,
    *,
    maximum_size: int,
    precondition_1d: bool,
    dtype: Any,
) -> SOAPPreconditioner:
    array = jnp.asarray(parameter)
    _real_floating_dtype(array)
    default_dtype = jnp.promote_types(array.dtype, jnp.float32)
    matrix_dtype = _state_dtype(default_dtype, dtype)
    if array.ndim == 1 and not precondition_1d:
        return SOAPPreconditioner((None,))
    return SOAPPreconditioner(
        jnp.zeros((size, size), dtype=matrix_dtype)
        if 0 < size <= maximum_size
        else None
        for size in array.shape
    )


def _update_covariance(
    gradient: Array,
    covariance: SOAPPreconditioner,
    /,
    *,
    decay: float,
    precision: jax.lax.PrecisionLike,
) -> SOAPPreconditioner:
    updated: list[PreconditionerMatrix] = []
    for axis, matrix in enumerate(covariance.matrices):
        if matrix is None:
            updated.append(None)
            continue
        gradient_ = jnp.asarray(gradient, dtype=matrix.dtype)
        contracted = tuple(index for index in range(gradient_.ndim) if index != axis)
        gram = jnp.tensordot(
            gradient_,
            gradient_,
            axes=(contracted, contracted),
            precision=precision,
        )
        next_matrix = decay * matrix + (1.0 - decay) * gram
        updated.append(0.5 * (next_matrix + next_matrix.T))
    return SOAPPreconditioner(updated)


def _project(
    value: Array,
    basis: SOAPPreconditioner,
    /,
    *,
    precision: jax.lax.PrecisionLike,
) -> Array:
    projected = value
    for matrix in basis.matrices:
        if matrix is None:
            projected = jnp.moveaxis(projected, 0, -1)
        else:
            projected = jnp.tensordot(
                projected,
                matrix,
                axes=((0,), (0,)),
                precision=precision,
            )
    return projected


def _project_back(
    value: Array,
    basis: SOAPPreconditioner,
    /,
    *,
    precision: jax.lax.PrecisionLike,
) -> Array:
    projected = value
    for matrix in basis.matrices:
        if matrix is None:
            projected = jnp.moveaxis(projected, 0, -1)
        else:
            projected = jnp.tensordot(
                projected,
                matrix,
                axes=((0,), (1,)),
                precision=precision,
            )
    return projected


def _eigenbasis(matrix: PreconditionerMatrix, /) -> PreconditionerMatrix:
    if matrix is None:
        return None
    symmetric = 0.5 * (matrix + matrix.T)
    _, vectors = jnp.linalg.eigh(symmetric, symmetrize_input=False)
    return jnp.flip(vectors, axis=1)


def _refresh_parameter_basis(
    covariance: SOAPPreconditioner,
    basis: SOAPPreconditioner,
    second_moment: Array,
    /,
    *,
    precision: jax.lax.PrecisionLike,
) -> tuple[SOAPPreconditioner, Array]:
    refreshed: list[PreconditionerMatrix] = []
    reordered_moment = second_moment
    for axis, (matrix, vectors) in enumerate(
        zip(covariance.matrices, basis.matrices, strict=True)
    ):
        if matrix is None or vectors is None:
            refreshed.append(None)
            continue
        rotated = jnp.matmul(
            jnp.matmul(vectors.T, matrix, precision=precision),
            vectors,
            precision=precision,
        )
        order = jnp.argsort(jnp.diag(rotated), descending=True)
        ordered_vectors = vectors[:, order]
        reordered_moment = jnp.take(reordered_moment, order, axis=axis)
        power_iteration = jnp.matmul(
            matrix,
            ordered_vectors,
            precision=precision,
        )
        next_vectors, _ = jnp.linalg.qr(power_iteration)
        refreshed.append(next_vectors.astype(vectors.dtype))
    return SOAPPreconditioner(refreshed), reordered_moment


def _validate_configuration(
    *,
    b1: float,
    b2: float,
    preconditioner_decay: float,
    eps: float,
    precondition_frequency: int,
    max_preconditioner_size: int,
) -> None:
    if not 0.0 <= b1 < 1.0:
        raise ValueError("b1 must lie in [0, 1).")
    if not 0.0 <= b2 < 1.0:
        raise ValueError("b2 must lie in [0, 1).")
    if not 0.0 <= preconditioner_decay < 1.0:
        raise ValueError("preconditioner_decay must lie in [0, 1).")
    if not isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    if int(precondition_frequency) < 1:
        raise ValueError("precondition_frequency must be positive.")
    if int(max_preconditioner_size) < 1:
        raise ValueError("max_preconditioner_size must be positive.")


def scale_by_soap(
    *,
    b1: float = 0.95,
    b2: float = 0.95,
    preconditioner_decay: float | None = None,
    eps: float = 1e-8,
    bias_correction: bool = True,
    precondition_frequency: int = 10,
    max_preconditioner_size: int = 10_000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    moment_dtype: Any = None,
    preconditioner_dtype: Any = None,
) -> optax.GradientTransformation:
    """Scale gradients with SOAP's Adam moments in adaptive eigenbases.

    The first call accumulates axis covariances and initializes their orthogonal
    bases; it intentionally returns a zero update. Subsequent calls update Adam
    moments in the current bases, project the normalized direction back, and
    refresh each bounded basis at the declared cadence.
    """

    b1_ = float(b1)
    b2_ = float(b2)
    covariance_decay = b2_ if preconditioner_decay is None else float(
        preconditioner_decay
    )
    epsilon = float(eps)
    frequency = int(precondition_frequency)
    maximum_size = int(max_preconditioner_size)
    _validate_configuration(
        b1=b1_,
        b2=b2_,
        preconditioner_decay=covariance_decay,
        eps=epsilon,
        precondition_frequency=frequency,
        max_preconditioner_size=maximum_size,
    )
    if moment_dtype is not None:
        _state_dtype(jnp.float32, moment_dtype)
    if preconditioner_dtype is not None:
        dtype = _state_dtype(jnp.float32, preconditioner_dtype)
        if jnp.finfo(dtype).bits < 32:
            raise ValueError(
                "preconditioner_dtype must provide at least 32-bit precision."
            )

    def initialize(parameters):
        def zero_moment(parameter):
            dtype = _state_dtype(
                _real_floating_dtype(parameter),
                moment_dtype,
            )
            return jnp.zeros_like(jnp.asarray(parameter), dtype=dtype)

        first_moment = jax.tree.map(zero_moment, parameters)
        second_moment = jax.tree.map(zero_moment, parameters)
        covariance = jax.tree.map(
            lambda parameter: _initialize_preconditioner(
                parameter,
                maximum_size=maximum_size,
                precondition_1d=bool(precondition_1d),
                dtype=preconditioner_dtype,
            ),
            parameters,
        )
        basis = jax.tree.map(
            lambda conditioner: conditioner.map(
                lambda matrix: None if matrix is None else jnp.zeros_like(matrix)
            ),
            covariance,
            is_leaf=_is_preconditioner,
        )
        return SOAPState(
            count=jnp.zeros((), dtype=jnp.int32),
            first_moment=first_moment,
            second_moment=second_moment,
            covariance=covariance,
            basis=basis,
        )

    def initialize_basis(gradients, state):
        covariance = jax.tree.map(
            lambda gradient, conditioner: _update_covariance(
                gradient,
                conditioner,
                decay=covariance_decay,
                precision=precision,
            ),
            gradients,
            state.covariance,
            is_leaf=_is_preconditioner,
        )
        basis = jax.tree.map(
            lambda conditioner: conditioner.map(_eigenbasis),
            covariance,
            is_leaf=_is_preconditioner,
        )
        return jax.tree.map(jnp.zeros_like, gradients), SOAPState(
            state.count,
            state.first_moment,
            state.second_moment,
            covariance,
            basis,
        )

    def optimizer_step(gradients, state):
        projected = jax.tree.map(
            lambda gradient, basis: _project(
                gradient,
                basis,
                precision=precision,
            ),
            gradients,
            state.basis,
            is_leaf=_is_preconditioner,
        )
        first_moment = jax.tree.map(
            lambda gradient, moment: (
                b1_ * moment
                + (1.0 - b1_) * jnp.asarray(gradient, dtype=moment.dtype)
            ),
            projected,
            state.first_moment,
        )
        second_moment = jax.tree.map(
            lambda gradient, moment: (
                b2_ * moment
                + (1.0 - b2_)
                * jnp.square(jnp.asarray(gradient, dtype=moment.dtype))
            ),
            projected,
            state.second_moment,
        )
        effective_step = jnp.maximum(state.count - 1, 1)
        if bias_correction:
            first_scale = 1.0 - b1_**effective_step
            second_scale = 1.0 - b2_**effective_step
        else:
            first_scale = jnp.asarray(1.0)
            second_scale = jnp.asarray(1.0)
        normalized = jax.tree.map(
            lambda first, second: (first / first_scale)
            / (jnp.sqrt(second / second_scale) + epsilon),
            first_moment,
            second_moment,
        )
        directions = jax.tree.map(
            lambda direction, basis, gradient: _project_back(
                direction,
                basis,
                precision=precision,
            ).astype(jnp.asarray(gradient).dtype),
            normalized,
            state.basis,
            gradients,
            is_leaf=_is_preconditioner,
        )
        covariance = jax.tree.map(
            lambda gradient, conditioner: _update_covariance(
                gradient,
                conditioner,
                decay=covariance_decay,
                precision=precision,
            ),
            gradients,
            state.covariance,
            is_leaf=_is_preconditioner,
        )

        def refresh():
            basis_and_second = jax.tree.map(
                lambda conditioner, basis, second: _refresh_parameter_basis(
                    conditioner,
                    basis,
                    second,
                    precision=precision,
                ),
                covariance,
                state.basis,
                second_moment,
                is_leaf=_is_preconditioner,
            )
            basis = jax.tree.map(
                lambda _, pair: pair[0],
                gradients,
                basis_and_second,
            )
            reordered_second = jax.tree.map(
                lambda _, pair: pair[1],
                gradients,
                basis_and_second,
            )
            rotated_first = jax.tree.map(
                lambda first, old_basis, new_basis: _project(
                    _project_back(first, old_basis, precision=precision),
                    new_basis,
                    precision=precision,
                ),
                first_moment,
                state.basis,
                basis,
                is_leaf=_is_preconditioner,
            )
            return basis, reordered_second, rotated_first

        def retain():
            return state.basis, second_moment, first_moment

        basis, second_moment, first_moment = jax.lax.cond(
            (state.count - 1) % frequency == 0,
            refresh,
            retain,
        )
        return directions, SOAPState(
            state.count,
            first_moment,
            second_moment,
            covariance,
            basis,
        )

    def update(gradients, state, parameters=None):
        del parameters
        count = optax.safe_int32_increment(state.count)
        counted = state._replace(count=count)
        return jax.lax.cond(
            count == 1,
            lambda: initialize_basis(gradients, counted),
            lambda: optimizer_step(gradients, counted),
        )

    return optax.GradientTransformation(initialize, update)


def _resolve_learning_rate(
    learning_rate: optax.ScalarOrSchedule,
    count: Array,
    /,
) -> Array:
    return (
        jnp.asarray(learning_rate(count))
        if callable(learning_rate)
        else jnp.asarray(learning_rate)
    )


def soap(
    learning_rate: optax.ScalarOrSchedule = 3e-3,
    *,
    b1: float = 0.95,
    b2: float = 0.95,
    preconditioner_decay: float | None = None,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    bias_correction: bool = True,
    precondition_frequency: int = 10,
    max_preconditioner_size: int = 10_000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    moment_dtype: Any = None,
    preconditioner_dtype: Any = None,
) -> optax.GradientTransformation:
    """Construct the Phydrax-native SOAP optimizer.

    SOAP combines per-axis Shampoo covariance eigenbases with Adam moments in
    the resulting coordinates. Oversized axes are skipped independently, so the
    same optimizer remains bounded on embeddings and high-rank scientific
    parameter arrays.
    """

    if not callable(learning_rate):
        rate = jnp.asarray(learning_rate)
        if rate.shape != ():
            raise ValueError("learning_rate must be a scalar or scalar schedule.")
        learning_rate_ = float(rate)
        if not isfinite(learning_rate_) or learning_rate_ <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
    decay = float(weight_decay)
    if not isfinite(decay) or decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    core = scale_by_soap(
        b1=b1,
        b2=b2,
        preconditioner_decay=preconditioner_decay,
        eps=eps,
        bias_correction=bias_correction,
        precondition_frequency=precondition_frequency,
        max_preconditioner_size=max_preconditioner_size,
        precondition_1d=precondition_1d,
        precision=precision,
        moment_dtype=moment_dtype,
        preconditioner_dtype=preconditioner_dtype,
    )

    def initialize(parameters):
        return core.init(parameters)

    def update(gradients, state, parameters=None):
        directions, next_state = core.update(gradients, state, parameters)
        schedule_step = jnp.maximum(next_state.count - 2, 0)
        rate = _resolve_learning_rate(learning_rate, schedule_step)
        active = next_state.count > 1
        if decay == 0.0:
            updates = jax.tree.map(
                lambda direction: jnp.where(
                    active,
                    -jnp.asarray(rate, dtype=direction.dtype) * direction,
                    jnp.zeros_like(direction),
                ),
                directions,
            )
        else:
            if parameters is None:
                raise ValueError("SOAP weight decay requires current parameters.")
            updates = jax.tree.map(
                lambda direction, parameter: jnp.where(
                    active,
                    -jnp.asarray(rate, dtype=direction.dtype)
                    * (direction + decay * parameter),
                    jnp.zeros_like(direction),
                ),
                directions,
                parameters,
            )
        return updates, next_state

    return optax.GradientTransformation(initialize, update)


__all__ = ["SOAPPreconditioner", "SOAPState", "scale_by_soap", "soap"]
