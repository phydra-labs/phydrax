#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypeAlias

import equinox as eqx
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


GaussianProcessActionKind: TypeAlias = Literal["fixed", "block-sparse", "pseudo-input"]


class _ResolvedGaussianProcessActions(StrictModule):
    """Validated native action operator bound to one observation design."""

    operator: AbstractLinearOperator
    kind: GaussianProcessActionKind = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    storage_elements: int = eqx.field(static=True)
    structurally_sparse: bool = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        kind: GaussianProcessActionKind,
    ):
        if kind not in ("fixed", "block-sparse", "pseudo-input"):
            raise ValueError("Unknown Gaussian-process action kind.")
        _validate_action_operator(operator)
        storage_elements, structurally_sparse = _action_storage(operator)
        self.operator = operator
        self.kind = kind
        self.action_id = f"gp-actions:{kind}:{operator.operator_id}"
        self.storage_elements = storage_elements
        self.structurally_sparse = structurally_sparse

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
    """Construct a linear observation-action subspace for one scalar GP design."""

    @abstractmethod
    def resolve(
        self,
        observation_points: Array,
        /,
        *,
        state: GaussianProcessLikelihoodState,
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
    ) -> _ResolvedGaussianProcessActions:
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
    ) -> _ResolvedGaussianProcessActions:
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
    ) -> _ResolvedGaussianProcessActions:
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
    "FixedGaussianProcessActionPolicy",
    "GaussianProcessActionKind",
    "PseudoInputGaussianProcessActionPolicy",
]
