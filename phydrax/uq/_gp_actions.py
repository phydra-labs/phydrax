#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from numbers import Integral, Real
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    AbstractSparseLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    OperatorCapabilities,
    OperatorProperties,
    SparseStorage,
)
from ._gp_likelihood import GaussianProcessLikelihoodState


GaussianProcessActionKind: TypeAlias = Literal[
    "fixed",
    "block-sparse",
    "pseudo-input",
    "lanczos",
    "conjugate-gradient",
    "gauss-seidel",
]


class _ResolvedGaussianProcessActions(StrictModule):
    """Validated fixed-capacity actions and iteration/breakdown evidence."""

    operator: AbstractLinearOperator
    active_mask: Array
    residual_history: Array
    breakdown_mask: Array
    convergence_mask: Array
    selected_indices: Array
    kind: GaussianProcessActionKind = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    storage_elements: int = eqx.field(static=True)
    structurally_sparse: bool = eqx.field(static=True)
    requires_residual: bool = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        kind: GaussianProcessActionKind,
        active_mask: ArrayLike | None = None,
        residual_history: ArrayLike | None = None,
        breakdown_mask: ArrayLike | None = None,
        convergence_mask: ArrayLike | None = None,
        selected_indices: ArrayLike | None = None,
        requires_residual: bool = False,
    ):
        kinds = (
            "fixed",
            "block-sparse",
            "pseudo-input",
            "lanczos",
            "conjugate-gradient",
            "gauss-seidel",
        )
        if kind not in kinds:
            raise ValueError("Unknown Gaussian-process action kind.")
        _validate_action_operator(operator)
        action_count = operator.source.size
        mask = (
            jnp.ones((action_count,), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if mask.shape != (action_count,):
            raise ValueError("active_mask must align with action capacity.")
        history = (
            jnp.empty((0,), dtype=operator.source.dtype)
            if residual_history is None
            else jnp.asarray(residual_history, dtype=operator.source.dtype)
        )
        if history.ndim != 1:
            raise ValueError("residual_history must be a vector.")
        breakdown = (
            ~mask if breakdown_mask is None else jnp.asarray(breakdown_mask, dtype=bool)
        )
        convergence = (
            ~mask
            if convergence_mask is None
            else jnp.asarray(convergence_mask, dtype=bool)
        )
        indices = (
            jnp.full((action_count,), -1, dtype=jnp.int32)
            if selected_indices is None
            else jnp.asarray(selected_indices, dtype=jnp.int32)
        )
        if (
            breakdown.shape != (action_count,)
            or convergence.shape != (action_count,)
            or indices.shape != (action_count,)
        ):
            raise ValueError("Action iteration diagnostics must align with capacity.")
        storage_elements, structurally_sparse = _action_storage(operator)
        self.operator = operator
        self.active_mask = mask
        self.residual_history = history
        self.breakdown_mask = breakdown
        self.convergence_mask = convergence
        self.selected_indices = indices
        self.kind = kind
        self.action_id = f"gp-actions:{kind}:{operator.operator_id}"
        self.storage_elements = storage_elements
        self.structurally_sparse = structurally_sparse
        self.requires_residual = bool(requires_residual)

    @property
    def num_observations(self) -> int:
        return self.operator.target.size

    @property
    def num_actions(self) -> int:
        return self.operator.source.size


class _BlockSparseGaussianProcessOperator(AbstractSparseLinearOperator):
    """One normalized sparse coefficient per observation row."""

    values: Array
    source_indices: Array

    def __init__(self, values: Array, num_actions: int, /):
        observation_count = int(values.shape[0])
        action_count = int(num_actions)
        source = ArraySpace((action_count,), dtype=values.dtype)
        target = ArraySpace((observation_count,), dtype=values.dtype)
        self.values = values
        self.source_indices = _balanced_block_indices(
            observation_count,
            action_count,
        )
        self.source = source
        self.target = target
        self.properties = OperatorProperties(
            rank=action_count,
            evidence={"rank": "construction"},
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = f"gp-block-sparse:{observation_count}:{action_count}"

    def mv(self, vector, /):
        value = self.source.validate(vector)
        return self.target.validate(self.values * value[self.source_indices])

    def transpose_mv(self, vector, /):
        value = self.target.validate(vector)
        return (
            jnp.zeros(
                (self.source.size,),
                dtype=self.values.dtype,
            )
            .at[self.source_indices]
            .add(self.values * value)
        )

    def adjoint_mv(self, vector, /):
        return self.transpose_mv(vector)

    def _materialize(self, /) -> Array:
        rows = jnp.arange(self.target.size, dtype=jnp.int32)
        return (
            jnp.zeros(
                (self.target.size, self.source.size),
                dtype=self.values.dtype,
            )
            .at[rows, self.source_indices]
            .set(self.values)
        )

    def _assemble_diagonal(self, /) -> Array:
        size = min(self.source.size, self.target.size)
        rows = jnp.arange(size, dtype=jnp.int32)
        return jnp.where(
            self.source_indices[:size] == rows,
            self.values[:size],
            jnp.zeros((size,), dtype=self.values.dtype),
        )

    def sparse_storage(self, /) -> SparseStorage:
        return SparseStorage(
            self.values,
            self.source_indices,
            jnp.arange(self.target.size + 1, dtype=jnp.int32),
            shape=(self.target.size, self.source.size),
        )


class AbstractGaussianProcessActionPolicy(StrictModule):
    """Construct a fixed-capacity linear observation-action subspace."""

    @property
    def requires_residual(self) -> bool:
        return False

    @abstractmethod
    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        residual: Array | None = None,
    ) -> _ResolvedGaussianProcessActions:
        raise NotImplementedError


class FixedGaussianProcessActionPolicy(AbstractGaussianProcessActionPolicy):
    """Use a caller-supplied dense or native sparse action operator."""

    operator: AbstractLinearOperator

    def __init__(self, actions: ArrayLike | AbstractLinearOperator, /):
        if isinstance(actions, AbstractLinearOperator):
            operator = actions
        else:
            raw_matrix = jnp.asarray(actions)
            if jnp.issubdtype(raw_matrix.dtype, jnp.complexfloating):
                raise TypeError("GP actions must be real-valued.")
            matrix = raw_matrix.astype(float)
            if matrix.ndim != 2:
                raise ValueError(
                    "Dense GP actions must have shape (observations, actions)."
                )
            operator = DenseLinearOperator(matrix)
        _validate_action_operator(operator)
        if not isinstance(operator, (DenseLinearOperator, AbstractSparseLinearOperator)):
            raise TypeError(
                "Fixed GP actions must use a DenseLinearOperator or "
                "AbstractSparseLinearOperator."
            )
        self.operator = operator

    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        residual: Array | None = None,
    ) -> _ResolvedGaussianProcessActions:
        del residual
        _require_state(state)
        points = jnp.asarray(observation_points)
        if self.operator.target.size != int(points.shape[0]):
            raise ValueError("Fixed GP actions must align with the observation design.")
        if self.operator.target.structure().dtype != points.dtype:
            raise TypeError("Fixed GP action dtype must match observation-point dtype.")
        return _ResolvedGaussianProcessActions(self.operator, kind="fixed")


class BlockSparseGaussianProcessActionPolicy(AbstractGaussianProcessActionPolicy):
    """Contiguous, balanced, column-normalized sparse actions with one value per row."""

    values: Array
    num_actions: int = eqx.field(static=True)

    def __init__(self, values: ArrayLike, num_actions: int, /):
        raw_values = jnp.asarray(values)
        if jnp.issubdtype(raw_values.dtype, jnp.complexfloating):
            raise TypeError("GP actions must be real-valued.")
        array = raw_values.astype(float)
        count = int(num_actions)
        if array.ndim != 1 or int(array.shape[0]) <= 0:
            raise ValueError("Block-sparse GP action values must be a nonempty vector.")
        if count < 1 or count > int(array.shape[0]):
            raise ValueError(
                "num_actions must lie between one and the observation count."
            )
        self.values = array
        self.num_actions = count

    @classmethod
    def from_random(
        cls,
        key: Key[Array, ""],
        num_observations: int,
        num_actions: int,
        /,
        *,
        dtype=None,
    ) -> BlockSparseGaussianProcessActionPolicy:
        observation_count = int(num_observations)
        action_count = int(num_actions)
        if observation_count < 1:
            raise ValueError("num_observations must be positive.")
        if action_count < 1 or action_count > observation_count:
            raise ValueError("num_actions must lie between one and num_observations.")
        values = jr.normal(key, (observation_count,), dtype=dtype)
        return cls(values, action_count)

    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        residual: Array | None = None,
    ) -> _ResolvedGaussianProcessActions:
        del residual
        _require_state(state)
        points = jnp.asarray(observation_points)
        observation_count = int(points.shape[0])
        if observation_count != int(self.values.shape[0]):
            raise ValueError(
                "Block-sparse GP action values must align with observations."
            )
        values = self.values.astype(points.dtype)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Block-sparse GP action values must be finite.",
        )
        source_indices = _balanced_block_indices(observation_count, self.num_actions)
        squared_norms = (
            jnp.zeros((self.num_actions,), dtype=values.dtype)
            .at[source_indices]
            .add(values * values)
        )
        norms = jnp.sqrt(squared_norms)
        minimum_norm = jnp.sqrt(jnp.asarray(jnp.finfo(values.dtype).tiny, values.dtype))
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(norms)) | jnp.any(norms <= minimum_norm),
            "Every block-sparse GP action block must have nonzero finite norm.",
        )
        normalized = values / norms[source_indices]
        operator = _BlockSparseGaussianProcessOperator(
            normalized,
            self.num_actions,
        )
        return _ResolvedGaussianProcessActions(operator, kind="block-sparse")


class PseudoInputGaussianProcessActionPolicy(AbstractGaussianProcessActionPolicy):
    """Dense kernel-section actions constructed from trainable pseudo-inputs."""

    pseudo_inputs: Array
    orthogonalize: bool = eqx.field(static=True)

    def __init__(self, pseudo_inputs: ArrayLike, /, *, orthogonalize: bool = True):
        raw_points = jnp.asarray(pseudo_inputs)
        if jnp.issubdtype(raw_points.dtype, jnp.complexfloating):
            raise TypeError("GP pseudo-inputs must be real-valued.")
        points = raw_points.astype(float)
        if points.ndim < 2 or int(points.shape[0]) <= 0:
            raise ValueError(
                "Pseudo-input GP actions need one action axis and kernel input axes."
            )
        self.pseudo_inputs = points
        self.orthogonalize = bool(orthogonalize)

    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        residual: Array | None = None,
    ) -> _ResolvedGaussianProcessActions:
        del residual
        _require_state(state)
        observations = jnp.asarray(observation_points)
        pseudo_inputs = self.pseudo_inputs.astype(observations.dtype)
        expected_rank = state.kernel.input_ndim + 1
        if observations.ndim != expected_rank or pseudo_inputs.ndim != expected_rank:
            raise ValueError(
                "Observation and pseudo-input designs must follow the kernel input rank."
            )
        if observations.shape[1:] != pseudo_inputs.shape[1:]:
            raise ValueError(
                "Observation and pseudo-input trailing dimensions must match."
            )
        observation_count = int(observations.shape[0])
        action_count = int(pseudo_inputs.shape[0])
        if action_count > observation_count:
            raise ValueError("Pseudo-input action count cannot exceed observations.")
        matrix = state.kernel.matrix(observations, pseudo_inputs)
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix)),
            "Pseudo-input GP actions must be finite.",
        )
        if self.orthogonalize:
            matrix, triangular = jnp.linalg.qr(matrix, mode="reduced")
            diagonal = jnp.abs(jnp.diag(triangular))
            scale = jnp.maximum(jnp.max(diagonal), jnp.asarray(1.0, diagonal.dtype))
            tolerance = (
                jnp.finfo(diagonal.dtype).eps
                * max(observation_count, action_count)
                * scale
            )
            matrix = eqx.error_if(
                matrix,
                jnp.any(~jnp.isfinite(diagonal)) | jnp.any(diagonal <= tolerance),
                "Pseudo-input GP actions must be linearly independent.",
            )
        operator = DenseLinearOperator(
            matrix,
            properties=OperatorProperties(
                rank=action_count,
                evidence={"rank": "construction"},
            ),
        )
        return _ResolvedGaussianProcessActions(operator, kind="pseudo-input")


class LanczosGaussianProcessActionPolicy(AbstractGaussianProcessActionPolicy):
    """Fixed-scan orthogonal Krylov actions from one explicit start vector."""

    start_vector: Array
    max_actions: int = eqx.field(static=True)
    breakdown_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        start_vector: ArrayLike,
        /,
        *,
        max_actions: int,
        breakdown_tolerance: float = 1e-8,
    ):
        vector = _action_vector(start_vector, name="start_vector")
        count = _action_count(max_actions, observation_count=int(vector.shape[0]))
        self.start_vector = vector
        self.max_actions = count
        self.breakdown_tolerance = _positive_tolerance(
            breakdown_tolerance, name="breakdown_tolerance"
        )

    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        residual: Array | None = None,
    ) -> _ResolvedGaussianProcessActions:
        del residual
        points, covariance = _iterative_covariance(observation_points, state)
        if self.start_vector.shape != (points.shape[0],):
            raise ValueError("start_vector must align with observations.")
        vector = self.start_vector.astype(points.dtype)
        norm = jnp.linalg.vector_norm(vector)
        active = jnp.isfinite(norm) & (norm > self.breakdown_tolerance)
        vector = jnp.where(active, vector / jnp.where(active, norm, 1.0), 0.0)
        previous = jnp.zeros_like(vector)
        beta = jnp.zeros((), dtype=points.dtype)
        basis = []
        masks = []
        history = [norm]
        for _ in range(self.max_actions):
            basis.append(jnp.where(active, vector, 0.0))
            masks.append(active)
            candidate = covariance @ vector - beta * previous
            for column in basis:
                candidate = candidate - jnp.vdot(column, candidate).real * column
            next_norm = jnp.linalg.vector_norm(candidate)
            next_active = (
                active & jnp.isfinite(next_norm) & (next_norm > self.breakdown_tolerance)
            )
            next_vector = jnp.where(
                next_active,
                candidate / jnp.where(next_active, next_norm, 1.0),
                0.0,
            )
            previous, vector, beta, active = (
                vector,
                next_vector,
                next_norm,
                next_active,
            )
            history.append(next_norm)
        matrix = jnp.stack(tuple(basis), axis=1)
        active_mask = jnp.stack(tuple(masks))
        operator = DenseLinearOperator(matrix)
        return _ResolvedGaussianProcessActions(
            operator,
            kind="lanczos",
            active_mask=active_mask,
            residual_history=jnp.stack(tuple(history)),
            breakdown_mask=~active_mask,
            convergence_mask=~active_mask,
        )


class ConjugateGradientGaussianProcessActionPolicy(AbstractGaussianProcessActionPolicy):
    """Residual-dependent fixed-scan covariance-conjugate search directions."""

    max_actions: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        max_actions: int,
        /,
        *,
        residual_tolerance: float = 1e-8,
    ):
        if not isinstance(max_actions, Integral) or isinstance(max_actions, bool):
            raise TypeError("max_actions must be an integer.")
        if int(max_actions) <= 0:
            raise ValueError("max_actions must be positive.")
        self.max_actions = int(max_actions)
        self.residual_tolerance = _positive_tolerance(
            residual_tolerance, name="residual_tolerance"
        )

    @property
    def requires_residual(self) -> bool:
        return True

    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        residual: Array | None = None,
    ) -> _ResolvedGaussianProcessActions:
        points, covariance = _iterative_covariance(observation_points, state)
        count = _action_count(self.max_actions, observation_count=int(points.shape[0]))
        values = _required_residual(residual, observation_count=int(points.shape[0]))
        remainder = values.astype(points.dtype)
        direction = remainder
        squared = jnp.vdot(remainder, remainder).real
        columns = []
        masks = []
        history = [jnp.sqrt(squared)]
        for _ in range(count):
            product = covariance @ direction
            denominator = jnp.vdot(direction, product).real
            direction_norm = jnp.linalg.vector_norm(direction)
            active = (
                jnp.isfinite(denominator)
                & (denominator > self.residual_tolerance)
                & jnp.isfinite(direction_norm)
                & (direction_norm > self.residual_tolerance)
                & (jnp.sqrt(squared) > self.residual_tolerance)
            )
            columns.append(
                jnp.where(
                    active,
                    direction / jnp.where(active, direction_norm, 1.0),
                    0.0,
                )
            )
            masks.append(active)
            step = jnp.where(active, squared / denominator, 0.0)
            next_remainder = remainder - step * product
            next_squared = jnp.vdot(next_remainder, next_remainder).real
            ratio = jnp.where(active & (squared > 0.0), next_squared / squared, 0.0)
            direction = next_remainder + ratio * direction
            remainder = next_remainder
            squared = next_squared
            history.append(jnp.sqrt(next_squared))
        matrix = jnp.stack(tuple(columns), axis=1)
        active_mask = jnp.stack(tuple(masks))
        return _ResolvedGaussianProcessActions(
            DenseLinearOperator(matrix),
            kind="conjugate-gradient",
            active_mask=active_mask,
            residual_history=jnp.stack(tuple(history)),
            breakdown_mask=~active_mask,
            convergence_mask=jnp.asarray(history[1:]) <= self.residual_tolerance,
            requires_residual=True,
        )


class GaussSeidelGaussianProcessActionPolicy(AbstractGaussianProcessActionPolicy):
    """Residual-dependent coordinate actions under an explicit fixed ordering."""

    fixed_order: Array
    max_actions: int = eqx.field(static=True)
    ordering: Literal["cyclic", "fixed", "largest-residual"] = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        max_actions: int,
        /,
        *,
        ordering: Literal["cyclic", "fixed", "largest-residual"] = "cyclic",
        fixed_order: ArrayLike | None = None,
        residual_tolerance: float = 1e-8,
    ):
        if not isinstance(max_actions, Integral) or isinstance(max_actions, bool):
            raise TypeError("max_actions must be an integer.")
        count = int(max_actions)
        if count <= 0:
            raise ValueError("max_actions must be positive.")
        if ordering not in ("cyclic", "fixed", "largest-residual"):
            raise ValueError("Unknown Gauss-Seidel ordering.")
        if ordering == "fixed":
            if fixed_order is None:
                raise ValueError("fixed ordering requires fixed_order.")
            order = jnp.asarray(fixed_order)
            if order.ndim != 1 or not jnp.issubdtype(order.dtype, jnp.integer):
                raise ValueError("fixed_order must be an integer vector.")
            if int(order.shape[0]) < count:
                raise ValueError("fixed_order must cover max_actions entries.")
        elif fixed_order is not None:
            raise ValueError("fixed_order is valid only with ordering='fixed'.")
        else:
            order = jnp.empty((0,), dtype=jnp.int32)
        self.fixed_order = order.astype(jnp.int32)
        self.max_actions = count
        self.ordering = ordering
        self.residual_tolerance = _positive_tolerance(
            residual_tolerance, name="residual_tolerance"
        )

    @property
    def requires_residual(self) -> bool:
        return True

    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        residual: Array | None = None,
    ) -> _ResolvedGaussianProcessActions:
        points, covariance = _iterative_covariance(observation_points, state)
        observation_count = int(points.shape[0])
        count = _action_count(self.max_actions, observation_count=observation_count)
        values = _required_residual(residual, observation_count=observation_count)
        if self.ordering == "fixed":
            host = jax.device_get(self.fixed_order[:count])
            if (
                bool(jnp.any(host < 0))
                or bool(jnp.any(host >= observation_count))
                or len(set(int(index) for index in host)) != count
            ):
                raise ValueError(
                    "fixed_order must select distinct in-range observation indices."
                )
        remainder = values.astype(points.dtype)
        used = jnp.zeros((observation_count,), dtype=bool)
        columns = []
        masks = []
        indices = []
        history = [jnp.linalg.vector_norm(remainder)]
        for iteration in range(count):
            if self.ordering == "cyclic":
                index = jnp.asarray(iteration, dtype=jnp.int32)
            elif self.ordering == "fixed":
                index = self.fixed_order[iteration]
            else:
                scores = jnp.where(used, -jnp.inf, jnp.abs(remainder))
                index = jax.lax.stop_gradient(jnp.argmax(scores).astype(jnp.int32))
            diagonal = covariance[index, index]
            active = (
                ~used[index]
                & jnp.isfinite(diagonal)
                & (diagonal > self.residual_tolerance)
                & (jnp.abs(remainder[index]) > self.residual_tolerance)
            )
            column = jnp.zeros((observation_count,), dtype=points.dtype)
            column = column.at[index].set(active.astype(points.dtype))
            columns.append(column)
            masks.append(active)
            indices.append(index)
            correction = jnp.where(active, remainder[index] / diagonal, 0.0)
            remainder = remainder - correction * covariance[:, index]
            used = used.at[index].set(True)
            history.append(jnp.linalg.vector_norm(remainder))
        active_mask = jnp.stack(tuple(masks))
        return _ResolvedGaussianProcessActions(
            DenseLinearOperator(jnp.stack(tuple(columns), axis=1)),
            kind="gauss-seidel",
            active_mask=active_mask,
            residual_history=jnp.stack(tuple(history)),
            breakdown_mask=~active_mask,
            convergence_mask=jnp.asarray(history[1:]) <= self.residual_tolerance,
            selected_indices=jnp.stack(tuple(indices)),
            requires_residual=True,
        )


def _iterative_covariance(
    observation_points: Array,
    state: GaussianProcessLikelihoodState,
    /,
) -> tuple[Array, Array]:
    _require_state(state)
    points = jnp.asarray(observation_points)
    expected_rank = state.kernel.input_ndim + 1
    if points.ndim != expected_rank or int(points.shape[0]) <= 0:
        raise ValueError("Observation design does not match the kernel input rank.")
    observation_count = int(points.shape[0])
    noise = jnp.broadcast_to(state.noise_scale, (observation_count,))
    covariance = state.kernel.matrix(points, points) + jnp.diag(
        noise * noise + state.jitter
    )
    covariance = eqx.error_if(
        covariance,
        jnp.any(~jnp.isfinite(covariance)),
        "Iterative GP actions require finite covariance entries.",
    )
    return points, covariance


def _required_residual(
    residual: Array | None,
    /,
    *,
    observation_count: int,
) -> Array:
    if residual is None:
        raise ValueError(
            "This GP action policy is residual-dependent; reusable factor() without "
            "a residual is not defined."
        )
    values = jnp.asarray(residual)
    if values.shape != (observation_count,):
        raise ValueError("Residual-dependent actions must align with observations.")
    return eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        "Residual-dependent actions require finite residuals.",
    )


def _action_vector(value: ArrayLike, /, *, name: str) -> Array:
    raw = jnp.asarray(value)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    vector = raw.astype(float)
    if vector.ndim != 1 or int(vector.shape[0]) <= 0:
        raise ValueError(f"{name} must be a nonempty vector.")
    return eqx.error_if(
        vector,
        jnp.any(~jnp.isfinite(vector)),
        f"{name} must be finite.",
    )


def _action_count(value: int, /, *, observation_count: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError("max_actions must be an integer.")
    count = int(value)
    if count <= 0 or count > observation_count:
        raise ValueError("max_actions must lie between one and observation count.")
    return count


def _positive_tolerance(value: Real, /, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    tolerance = float(value)
    if not jnp.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return tolerance


def _balanced_block_indices(num_observations: int, num_actions: int, /) -> Array:
    base = num_observations // num_actions
    remainder = num_observations % num_actions
    row = jnp.arange(num_observations, dtype=jnp.int32)
    leading = remainder * (base + 1)
    return jnp.where(
        row < leading,
        row // (base + 1),
        remainder + (row - leading) // base,
    ).astype(jnp.int32)


def _validate_action_operator(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("GP actions must be an AbstractLinearOperator.")
    if operator.batch_shape:
        raise ValueError("GP action operators must be unbatched.")
    if not isinstance(operator.source, ArraySpace) or not isinstance(
        operator.target, ArraySpace
    ):
        raise TypeError("GP actions require array-valued source and target spaces.")
    if operator.source.shape != (operator.source.size,) or operator.target.shape != (
        operator.target.size,
    ):
        raise ValueError("GP action spaces must be one-dimensional vectors.")
    if operator.source.size < 1 or operator.source.size > operator.target.size:
        raise ValueError("GP action count must lie between one and observation count.")
    if jnp.issubdtype(operator.source.dtype, jnp.complexfloating) or jnp.issubdtype(
        operator.target.dtype, jnp.complexfloating
    ):
        raise TypeError("GP action operators must be real-valued.")
    if operator.source.dtype != operator.target.dtype:
        raise TypeError("GP action source and target dtypes must match.")


def _action_storage(operator: AbstractLinearOperator, /) -> tuple[int, bool]:
    if isinstance(operator, DenseLinearOperator):
        return int(operator.matrix.size), False
    if isinstance(operator, AbstractSparseLinearOperator):
        return int(operator.sparse_storage().values.size), True
    raise TypeError("GP action storage is known only for dense and sparse operators.")


def _require_state(state: GaussianProcessLikelihoodState, /) -> None:
    if not isinstance(state, GaussianProcessLikelihoodState):
        raise TypeError("state must be a GaussianProcessLikelihoodState.")


__all__ = [
    "AbstractGaussianProcessActionPolicy",
    "BlockSparseGaussianProcessActionPolicy",
    "ConjugateGradientGaussianProcessActionPolicy",
    "FixedGaussianProcessActionPolicy",
    "GaussSeidelGaussianProcessActionPolicy",
    "GaussianProcessActionKind",
    "LanczosGaussianProcessActionPolicy",
    "PseudoInputGaussianProcessActionPolicy",
]
