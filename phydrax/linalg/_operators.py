#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, cast, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._costs import _array_tree_storage_bytes, OperatorActionCostEstimate
from ._linearizations import PreparedLinearization
from ._pairings import DiagonalPairing, EuclideanPairing
from ._properties import (
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
    PropertyEvidence,
)
from ._spaces import (
    _coordinate_dtype,
    _coordinate_pairing_weights,
    _has_diagonal_pairing,
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
)


def _id(value: str | None, payload: dict[str, Any], /) -> str:
    if value is None:
        return canonical_fingerprint(payload)
    identifier = str(value)
    if not identifier:
        raise ValueError("operator_id must be non-empty.")
    return identifier


def _batch_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size < 0 for size in shape):
        raise ValueError("Operator batch dimensions must be nonnegative.")
    return shape


def _same_space(left: AbstractVectorSpace, right: AbstractVectorSpace, /) -> None:
    if not left.compatible(right):
        raise ValueError(
            f"Incompatible vector spaces {left.space_id!r} and {right.space_id!r}."
        )


def _validate_properties(
    properties: OperatorProperties,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    /,
) -> None:
    if properties.rank is not None and properties.rank > min(source.size, target.size):
        raise ValueError("Certified operator rank exceeds the operator dimensions.")
    square_required = (
        properties.diagonal
        or properties.triangular
        or properties.self_adjoint
        or properties.positive_definite
        or properties.positive_semidefinite
    )
    if square_required and not source.compatible(target):
        raise ValueError(
            "Diagonal, triangular, self-adjoint, and definite properties "
            "require identical source and target spaces."
        )


def _transformed_evidence(
    claims: Mapping[str, bool | int | None],
    *operators: "AbstractLinearOperator",
) -> dict[str, PropertyEvidence]:
    return {
        name: "transformed"
        for name, claimed in claims.items()
        if claimed is not False
        and claimed is not None
        and all(
            operator.properties.evidence_for(name) != "unknown" for operator in operators
        )
    }


def _validate_action_dtype(
    coefficient_dtype: Any,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    name: str,
    /,
) -> None:
    result_dtype = np.dtype(
        jax.dtypes.canonicalize_dtype(
            jnp.result_type(np.dtype(coefficient_dtype), _coordinate_dtype(source))
        )
    )
    if result_dtype != _coordinate_dtype(target):
        raise TypeError(
            f"{name} acting on source coordinates produces dtype {result_dtype}, "
            f"but the target requires {_coordinate_dtype(target)}."
        )


def _supports_batched_adjoint(
    batch_shape: tuple[int, ...],
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    /,
) -> bool:
    return not batch_shape or (
        isinstance(source, ArraySpace)
        and isinstance(target, ArraySpace)
        and _has_diagonal_pairing(source)
        and _has_diagonal_pairing(target)
    )


def _tree_add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_scale(value: PyTree[Array], scalar: Array, /) -> PyTree[Array]:
    return jax.tree.map(lambda leaf: scalar * leaf, value)


def _tree_conj(value: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(jnp.conj, value)


def _array_value(
    space: ArraySpace,
    batch_shape: tuple[int, ...],
    vector: ArrayLike,
    name: str,
    /,
) -> tuple[Array, tuple[int, ...]]:
    value = jnp.asarray(vector)
    prefix = batch_shape + space.shape
    if value.shape[: len(prefix)] == prefix:
        rhs_shape = value.shape[len(prefix) :]
    elif batch_shape and value.shape[: len(space.shape)] == space.shape:
        rhs_shape = value.shape[len(space.shape) :]
        value = jnp.broadcast_to(value, batch_shape + value.shape)
    else:
        raise ValueError(
            f"{name} must begin with shape {prefix} or shared event shape "
            f"{space.shape}; got {value.shape}."
        )
    if np.dtype(value.dtype) != space.dtype:
        raise TypeError(f"{name} must have dtype {space.dtype}; got {value.dtype}.")
    return value, tuple(int(size) for size in rhs_shape)


def _array_pairing_action(
    space: ArraySpace,
    batch_shape: tuple[int, ...],
    vector: ArrayLike,
    /,
    *,
    inverse: bool,
) -> Array:
    value, rhs_shape = _array_value(space, batch_shape, vector, "vector")
    if isinstance(space.pairing, EuclideanPairing):
        return value
    if isinstance(space.pairing, DiagonalPairing):
        weights = jnp.asarray(space.pairing.weights)
        if weights.shape != space.shape:
            raise ValueError("Diagonal pairing weights must match the array-space shape.")
        reshaped = weights.reshape(
            (1,) * len(batch_shape) + space.shape + (1,) * len(rhs_shape)
        )
        return value / reshaped if inverse else reshaped * value
    if batch_shape:
        raise TypeError(
            "Operator-batched adjoints require Euclidean or diagonal array pairings."
        )
    rhs_size = prod(rhs_shape) if rhs_shape else 1
    columns = value.reshape(space.shape + (rhs_size,))
    action = space.inverse_riesz if inverse else space.riesz
    transformed = jax.vmap(action, in_axes=-1, out_axes=-1)(columns)
    return transformed.reshape(space.shape + rhs_shape)


def _dense_array_action(
    matrix: Array,
    source: ArraySpace,
    target: ArraySpace,
    batch_shape: tuple[int, ...],
    vector: ArrayLike,
    /,
) -> Array:
    value, rhs_shape = _array_value(source, batch_shape, vector, "vector")
    rhs_size = prod(rhs_shape) if rhs_shape else 1
    flattened = value.reshape(batch_shape + (source.size, rhs_size))
    image = jnp.matmul(matrix, flattened)
    return image.reshape(batch_shape + target.shape + rhs_shape)


def _dense_action(
    matrix: Array,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    batch_shape: tuple[int, ...],
    vector: PyTree[Any],
    /,
) -> PyTree[Array]:
    if isinstance(source, ArraySpace) and isinstance(target, ArraySpace):
        return _dense_array_action(matrix, source, target, batch_shape, vector)
    if batch_shape:
        raise ValueError("Batched dense operators currently require ArraySpace values.")
    coordinates = source.flatten(vector)
    return target.unflatten(matrix @ coordinates)


def _generic_adjoint(
    operator: "AbstractLinearOperator",
    vector: PyTree[Any],
    /,
) -> PyTree[Array]:
    if isinstance(operator.source, ArraySpace) and isinstance(
        operator.target, ArraySpace
    ):
        target_covector = _array_pairing_action(
            operator.target,
            operator.batch_shape,
            vector,
            inverse=False,
        )
        source_covector = _tree_conj(operator.transpose_mv(_tree_conj(target_covector)))
        return _array_pairing_action(
            operator.source,
            operator.batch_shape,
            source_covector,
            inverse=True,
        )
    if operator.batch_shape:
        raise ValueError("Batched adjoints currently require ArraySpace values.")
    target_vector = operator.target.validate(vector)
    target_covector = operator.target.riesz(target_vector)
    conjugated = _tree_conj(target_covector)
    source_covector = _tree_conj(operator.transpose_mv(conjugated))
    return operator.source.inverse_riesz(source_covector)


def _materialize_by_basis(operator: "AbstractLinearOperator", /) -> Array:
    if operator.batch_shape:
        raise ValueError("Basis materialization does not support operator batches.")
    structure = operator.source.structure()
    dtype = jnp.result_type(*[spec.dtype for spec in jax.tree.leaves(structure)])

    def column(index):
        coordinates = jax.nn.one_hot(index, operator.source.size, dtype=dtype)
        vector = operator.source.unflatten(coordinates)
        return operator.target.flatten(operator.mv(vector))

    columns = jax.lax.map(column, jnp.arange(operator.source.size))
    return jnp.swapaxes(columns, -1, -2)


class AbstractLinearOperator(StrictModule):
    """Rectangular linear map between explicitly declared vector spaces."""

    source: AbstractVectorSpace
    target: AbstractVectorSpace
    properties: OperatorProperties
    capabilities: OperatorCapabilities
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def _materialize(self, /) -> Array:
        raise NotImplementedError

    def __call__(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.mv(vector)

    def __add__(self, other: object) -> "AbstractLinearOperator":
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return SumLinearOperator(self, other)

    def __sub__(self, other: object) -> "AbstractLinearOperator":
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return SumLinearOperator(self, ScaledLinearOperator(other, -1.0))

    def __mul__(self, scalar: Any) -> "AbstractLinearOperator":
        return ScaledLinearOperator(self, scalar)

    def __rmul__(self, scalar: Any) -> "AbstractLinearOperator":
        return ScaledLinearOperator(self, scalar)

    def __matmul__(self, other: object) -> "AbstractLinearOperator":
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return ComposedLinearOperator(self, other)

    def __neg__(self) -> "AbstractLinearOperator":
        return ScaledLinearOperator(self, -1.0)


class _AbstractCostedLinearOperator(AbstractLinearOperator):
    """Linear operator with exact per-right-hand-side action scratch."""

    @abc.abstractmethod
    def _action_workspace_cost(self, /) -> tuple[int, str]:
        raise NotImplementedError


class _DiagonalAssembly(Protocol):
    def _assemble_diagonal(self, /) -> Array: ...


def _assemble_operator_diagonal(
    operator: AbstractLinearOperator,
    /,
) -> Array:
    return cast(_DiagonalAssembly, operator)._assemble_diagonal()


class DenseLinearOperator(AbstractLinearOperator):
    """Explicit dense matrix over flattened source and target event coordinates."""

    matrix: Array

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        source: AbstractVectorSpace | None = None,
        target: AbstractVectorSpace | None = None,
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
    ):
        matrix_ = jnp.asarray(matrix)
        if matrix_.ndim < 2:
            raise ValueError("matrix must have at least two dimensions.")
        if not jnp.issubdtype(matrix_.dtype, jnp.inexact):
            matrix_ = matrix_.astype(float)
        target_size, source_size = (int(size) for size in matrix_.shape[-2:])
        source_ = (
            ArraySpace((source_size,), dtype=matrix_.dtype) if source is None else source
        )
        target_ = (
            ArraySpace((target_size,), dtype=matrix_.dtype) if target is None else target
        )
        if not isinstance(source_, AbstractVectorSpace) or not isinstance(
            target_, AbstractVectorSpace
        ):
            raise TypeError("source and target must be AbstractVectorSpace values.")
        if source_.size != source_size or target_.size != target_size:
            raise ValueError("Matrix dimensions must match source and target sizes.")
        batch = _batch_shape(matrix_.shape[:-2])
        if batch and (
            not isinstance(source_, ArraySpace) or not isinstance(target_, ArraySpace)
        ):
            raise ValueError("Batched dense operators require ArraySpace values.")
        _validate_action_dtype(matrix_.dtype, source_, target_, "matrix")
        _validate_action_dtype(matrix_.dtype, target_, source_, "transposed matrix")

        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        _validate_properties(properties_, source_, target_)
        self.source = source_
        self.target = target_
        self.matrix = matrix_
        self.properties = properties_
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=_supports_batched_adjoint(batch, source_, target_),
            materialize=True,
            diagonal_assembly=True,
        )
        self.batch_shape = batch
        self.operator_id = _id(
            operator_id,
            {
                "kind": "dense",
                "source": source_.space_id,
                "target": target_.space_id,
                "batch_shape": list(batch),
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _dense_action(
            self.matrix, self.source, self.target, self.batch_shape, vector
        )

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _dense_action(
            jnp.swapaxes(self.matrix, -1, -2),
            self.target,
            self.source,
            self.batch_shape,
            vector,
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if isinstance(self.source, ArraySpace) and isinstance(self.target, ArraySpace):
            matrix = jnp.conj(jnp.swapaxes(self.matrix, -1, -2))
            target_pairing = self.target.pairing
            source_pairing = self.source.pairing
            if isinstance(target_pairing, DiagonalPairing):
                weights = jnp.asarray(target_pairing.weights).reshape((self.target.size,))
                matrix = matrix * weights
            elif not isinstance(target_pairing, EuclideanPairing):
                if self.batch_shape:
                    raise ValueError("Custom pairings do not support batched adjoints.")
                return _generic_adjoint(self, vector)
            if isinstance(source_pairing, DiagonalPairing):
                weights = jnp.asarray(source_pairing.weights).reshape((self.source.size,))
                matrix = matrix / weights[..., None]
            elif not isinstance(source_pairing, EuclideanPairing):
                if self.batch_shape:
                    raise ValueError("Custom pairings do not support batched adjoints.")
                return _generic_adjoint(self, vector)
            return _dense_action(
                matrix,
                self.target,
                self.source,
                self.batch_shape,
                vector,
            )
        if self.batch_shape:
            raise ValueError("Batched PyTree adjoints are not supported.")
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return self.matrix

    def _assemble_diagonal(self, /) -> Array:
        return jnp.diagonal(self.matrix, axis1=-2, axis2=-1)


class DiagonalLinearOperator(AbstractLinearOperator):
    """Diagonal endomorphism in canonical flattened coordinates."""

    diagonal: Array

    def __init__(
        self,
        diagonal: ArrayLike,
        /,
        *,
        space: AbstractVectorSpace | None = None,
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
    ):
        diagonal_ = jnp.asarray(diagonal)
        if diagonal_.ndim < 1:
            raise ValueError("diagonal must have at least one dimension.")
        if not jnp.issubdtype(diagonal_.dtype, jnp.inexact):
            diagonal_ = diagonal_.astype(float)
        size = int(diagonal_.shape[-1])
        space_ = ArraySpace((size,), dtype=diagonal_.dtype) if space is None else space
        if not isinstance(space_, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        if space_.size != size:
            raise ValueError("Diagonal length must match the space size.")
        batch = _batch_shape(diagonal_.shape[:-1])
        if batch and not isinstance(space_, ArraySpace):
            raise ValueError("Batched diagonal operators require an ArraySpace.")
        _validate_action_dtype(diagonal_.dtype, space_, space_, "diagonal")

        if properties is None:
            self_adjoint = not jnp.issubdtype(
                diagonal_.dtype, jnp.complexfloating
            ) and _has_diagonal_pairing(space_)
            properties_ = OperatorProperties(
                diagonal=True,
                self_adjoint=self_adjoint,
                evidence={
                    name: "construction"
                    for name, claimed in {
                        "diagonal": True,
                        "self_adjoint": self_adjoint,
                    }.items()
                    if claimed
                },
            )
        else:
            properties_ = properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        _validate_properties(properties_, space_, space_)
        self.source = space_
        self.target = space_
        self.diagonal = diagonal_
        self.properties = properties_
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=_supports_batched_adjoint(batch, space_, space_),
            materialize=True,
            diagonal_assembly=True,
        )
        self.batch_shape = batch
        self.operator_id = _id(
            operator_id,
            {
                "kind": "diagonal",
                "space": space_.space_id,
                "batch_shape": list(batch),
            },
        )

    def _action(self, diagonal: Array, vector: PyTree[Any], /) -> PyTree[Array]:
        if isinstance(self.source, ArraySpace):
            value, rhs_shape = _array_value(
                self.source, self.batch_shape, vector, "vector"
            )
            reshaped = diagonal.reshape(
                self.batch_shape + self.source.shape + (1,) * len(rhs_shape)
            )
            return reshaped * value
        if self.batch_shape:
            raise ValueError("Batched diagonal operators require ArraySpace values.")
        coordinates = self.source.flatten(vector)
        return self.source.unflatten(diagonal * coordinates)

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self._action(self.diagonal, vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self._action(self.diagonal, vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if isinstance(self.source, ArraySpace) and isinstance(
            self.source.pairing, (EuclideanPairing, DiagonalPairing)
        ):
            return self._action(jnp.conj(self.diagonal), vector)
        if self.batch_shape:
            raise ValueError("Custom pairings do not support batched adjoints.")
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        identity = jnp.eye(self.source.size, dtype=self.diagonal.dtype)
        return self.diagonal[..., :, None] * identity

    def _assemble_diagonal(self, /) -> Array:
        return self.diagonal


class IdentityLinearOperator(AbstractLinearOperator):
    """Identity map on one declared vector space."""

    def __init__(self, space: AbstractVectorSpace, /, *, operator_id: str | None = None):
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        self.source = space
        self.target = space
        self.properties = OperatorProperties(
            diagonal=True,
            self_adjoint=True,
            positive_definite=True,
            rank=space.size,
            evidence={
                "diagonal": "construction",
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
                "rank": "construction",
            },
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=True,
        )
        self.batch_shape = ()
        self.operator_id = _id(operator_id, {"kind": "identity", "space": space.space_id})

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if isinstance(self.source, ArraySpace):
            return _array_value(self.source, (), vector, "vector")[0]
        return self.source.validate(vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.mv(vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        dtype = jnp.result_type(
            *[spec.dtype for spec in jax.tree.leaves(self.source.structure())]
        )
        return jnp.eye(self.source.size, dtype=dtype)

    def _assemble_diagonal(self, /) -> Array:
        dtype = jnp.result_type(
            *[spec.dtype for spec in jax.tree.leaves(self.source.structure())]
        )
        return jnp.ones((self.source.size,), dtype=dtype)


class FunctionLinearOperator(AbstractLinearOperator):
    """Linear callable with declared source and target spaces."""

    function: Callable[[PyTree[Array]], PyTree[Array]]
    transpose_action: Callable[[PyTree[Array]], PyTree[Array]] | None

    def __init__(
        self,
        function: Callable[[PyTree[Array]], PyTree[Array]],
        /,
        *,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        transpose_action: Callable[[PyTree[Array]], PyTree[Array]] | None = None,
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
        closure_convert: bool = True,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if transpose_action is not None and not callable(transpose_action):
            raise TypeError("transpose_action must be callable or None.")
        if not isinstance(source, AbstractVectorSpace) or not isinstance(
            target, AbstractVectorSpace
        ):
            raise TypeError("source and target must be AbstractVectorSpace values.")
        function_ = (
            eqx.filter_closure_convert(function, source.structure())
            if closure_convert
            else function
        )
        transpose_ = (
            eqx.filter_closure_convert(transpose_action, target.structure())
            if closure_convert and transpose_action is not None
            else transpose_action
        )
        output = jax.eval_shape(function_, source.structure())
        if eqx.tree_equal(output, target.structure()) is not True:
            raise ValueError("function output structure must match target space.")
        if transpose_ is not None:
            transposed = jax.eval_shape(transpose_, target.structure())
            if eqx.tree_equal(transposed, source.structure()) is not True:
                raise ValueError("transpose_action output must match source space.")
        self.source = source
        self.target = target
        self.function = function_
        self.transpose_action = transpose_
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        _validate_properties(properties_, source, target)
        self.properties = properties_
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "function",
                "source": source.space_id,
                "target": target.space_id,
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        value = self.source.validate(vector)
        return self.target.validate(self.function(value))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        value = self.target.validate(vector)
        if self.transpose_action is not None:
            return self.source.validate(self.transpose_action(value))
        transpose_action = jax.linear_transpose(self.function, self.source.structure())
        return self.source.validate(transpose_action(value)[0])

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return _materialize_by_basis(self)


class JacobianLinearOperator(AbstractLinearOperator):
    """Matrix-free Jacobian backed by one reusable prepared linearization."""

    linearization: PreparedLinearization

    def __init__(
        self,
        linearization: PreparedLinearization,
        /,
        *,
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(linearization, PreparedLinearization):
            raise TypeError("linearization must be a PreparedLinearization.")
        self.source = linearization.source
        self.target = linearization.target
        self.linearization = linearization
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        _validate_properties(properties_, self.source, self.target)
        self.properties = properties_
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "jacobian",
                "linearization": linearization.linearization_id,
                "source": self.source.space_id,
                "target": self.target.space_id,
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.linearization.jvp(vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.linearization.vjp(vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return _materialize_by_basis(self)


class ScaledLinearOperator(AbstractLinearOperator):
    """Scalar multiple of one operator."""

    operator: AbstractLinearOperator
    scalar: Array

    def __init__(self, operator: AbstractLinearOperator, scalar: Any, /):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        scalar_ = jnp.asarray(scalar)
        if scalar_.shape != () or not jnp.issubdtype(scalar_.dtype, jnp.inexact):
            raise TypeError("scalar must be one real or complex inexact scalar.")
        _validate_action_dtype(scalar_.dtype, operator.target, operator.target, "scalar")
        _validate_action_dtype(scalar_.dtype, operator.source, operator.source, "scalar")
        self.source = operator.source
        self.target = operator.target
        self.operator = operator
        self.scalar = scalar_
        claims = {
            "diagonal": operator.properties.diagonal,
            "triangular": operator.properties.triangular,
            "block_diagonal": operator.properties.block_diagonal,
        }
        self.properties = OperatorProperties(
            **claims,
            evidence=_transformed_evidence(claims, operator),
        )
        self.capabilities = operator.capabilities
        self.batch_shape = operator.batch_shape
        self.operator_id = _id(
            None,
            {"kind": "scale", "operator": operator.operator_id},
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _tree_scale(self.operator.mv(vector), self.scalar)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _tree_scale(self.operator.transpose_mv(vector), self.scalar)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _tree_scale(self.operator.adjoint_mv(vector), jnp.conj(self.scalar))

    def _materialize(self, /) -> Array:
        return self.scalar * self.operator._materialize()

    def _assemble_diagonal(self, /) -> Array:
        return self.scalar * _assemble_operator_diagonal(self.operator)


class SumLinearOperator(AbstractLinearOperator):
    """Sum of operators with identical spaces and batch layout."""

    left: AbstractLinearOperator
    right: AbstractLinearOperator

    def __init__(
        self,
        left: AbstractLinearOperator,
        right: AbstractLinearOperator,
        /,
    ):
        if not isinstance(left, AbstractLinearOperator) or not isinstance(
            right, AbstractLinearOperator
        ):
            raise TypeError("left and right must be AbstractLinearOperator values.")
        _same_space(left.source, right.source)
        _same_space(left.target, right.target)
        if left.batch_shape != right.batch_shape:
            raise ValueError("Summed operators must have identical batch shapes.")
        self.source = left.source
        self.target = left.target
        self.left = left
        self.right = right
        claims = {
            "diagonal": left.properties.diagonal and right.properties.diagonal,
            "self_adjoint": (
                left.properties.self_adjoint and right.properties.self_adjoint
            ),
            "block_diagonal": (
                left.properties.block_diagonal and right.properties.block_diagonal
            ),
        }
        self.properties = OperatorProperties(
            **claims,
            evidence=_transformed_evidence(claims, left, right),
        )
        self.capabilities = OperatorCapabilities(
            transpose=left.capabilities.transpose and right.capabilities.transpose,
            adjoint=left.capabilities.adjoint and right.capabilities.adjoint,
            materialize=left.capabilities.materialize and right.capabilities.materialize,
            diagonal_assembly=(
                left.capabilities.diagonal_assembly
                and right.capabilities.diagonal_assembly
            ),
        )
        self.batch_shape = left.batch_shape
        self.operator_id = _id(
            None,
            {
                "kind": "sum",
                "left": left.operator_id,
                "right": right.operator_id,
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _tree_add(self.left.mv(vector), self.right.mv(vector))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _tree_add(self.left.transpose_mv(vector), self.right.transpose_mv(vector))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _tree_add(self.left.adjoint_mv(vector), self.right.adjoint_mv(vector))

    def _materialize(self, /) -> Array:
        return self.left._materialize() + self.right._materialize()

    def _assemble_diagonal(self, /) -> Array:
        return _assemble_operator_diagonal(self.left) + _assemble_operator_diagonal(
            self.right
        )


class ComposedLinearOperator(AbstractLinearOperator):
    """Composition ``left(right(x))``."""

    left: AbstractLinearOperator
    right: AbstractLinearOperator

    def __init__(
        self,
        left: AbstractLinearOperator,
        right: AbstractLinearOperator,
        /,
    ):
        if not isinstance(left, AbstractLinearOperator) or not isinstance(
            right, AbstractLinearOperator
        ):
            raise TypeError("left and right must be AbstractLinearOperator values.")
        _same_space(right.target, left.source)
        if left.batch_shape != right.batch_shape:
            raise ValueError("Composed operators must have identical batch shapes.")
        self.source = right.source
        self.target = left.target
        self.left = left
        self.right = right
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=left.capabilities.transpose and right.capabilities.transpose,
            adjoint=left.capabilities.adjoint and right.capabilities.adjoint,
            materialize=left.capabilities.materialize and right.capabilities.materialize,
        )
        self.batch_shape = left.batch_shape
        self.operator_id = _id(
            None,
            {
                "kind": "composition",
                "left": left.operator_id,
                "right": right.operator_id,
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.left.mv(self.right.mv(vector))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.right.transpose_mv(self.left.transpose_mv(vector))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.right.adjoint_mv(self.left.adjoint_mv(vector))

    def _materialize(self, /) -> Array:
        return self.left._materialize() @ self.right._materialize()


class TransposeLinearOperator(AbstractLinearOperator):
    """Algebraic transpose view of one operator."""

    operator: AbstractLinearOperator

    def __init__(self, operator: AbstractLinearOperator, /):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        self.source = operator.target
        self.target = operator.source
        self.operator = operator
        diagonal = operator.properties.diagonal
        triangular = operator.properties.triangular
        rank = operator.properties.rank
        claimed = {"diagonal": diagonal, "triangular": triangular, "rank": rank}
        self.properties = OperatorProperties(
            diagonal=diagonal,
            triangular=triangular,
            rank=rank,
            evidence=_transformed_evidence(claimed, operator),
        )
        self.capabilities = operator.capabilities
        self.batch_shape = operator.batch_shape
        self.operator_id = _id(
            None, {"kind": "transpose", "operator": operator.operator_id}
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.transpose_mv(vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.mv(vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return jnp.swapaxes(self.operator._materialize(), -1, -2)

    def _assemble_diagonal(self, /) -> Array:
        return _assemble_operator_diagonal(self.operator)


class AdjointLinearOperator(AbstractLinearOperator):
    """Hilbert-adjoint view relative to the source and target pairings."""

    operator: AbstractLinearOperator

    def __init__(self, operator: AbstractLinearOperator, /):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        self.source = operator.target
        self.target = operator.source
        self.operator = operator
        diagonal = operator.properties.diagonal and (
            operator.properties.certifies("self_adjoint")
            or (
                _has_diagonal_pairing(operator.source)
                and _has_diagonal_pairing(operator.target)
            )
        )
        self_adjoint = operator.properties.self_adjoint
        rank = operator.properties.rank
        claimed = {
            "diagonal": diagonal,
            "self_adjoint": self_adjoint,
            "rank": rank,
        }
        self.properties = OperatorProperties(
            diagonal=diagonal,
            self_adjoint=self_adjoint,
            rank=rank,
            evidence=_transformed_evidence(claimed, operator),
        )
        self.capabilities = OperatorCapabilities(
            transpose=operator.capabilities.transpose,
            adjoint=operator.capabilities.adjoint,
            materialize=operator.capabilities.materialize,
            diagonal_assembly=(
                operator.capabilities.diagonal_assembly
                and operator.source.compatible(operator.target)
                and _has_diagonal_pairing(operator.source)
                and _has_diagonal_pairing(operator.target)
            ),
        )
        self.batch_shape = operator.batch_shape
        self.operator_id = _id(
            None, {"kind": "adjoint", "operator": operator.operator_id}
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.adjoint_mv(vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if isinstance(self.operator.source, ArraySpace) and isinstance(
            self.operator.target, ArraySpace
        ):
            source_primal = _array_pairing_action(
                self.operator.source,
                self.batch_shape,
                vector,
                inverse=True,
            )
            image = _tree_conj(self.operator.mv(_tree_conj(source_primal)))
            return _array_pairing_action(
                self.operator.target,
                self.batch_shape,
                image,
                inverse=False,
            )
        if self.batch_shape:
            raise ValueError(
                "Batched adjoint transposes currently require ArraySpace values."
            )
        transpose_action = jax.linear_transpose(self.mv, self.source.structure())
        return self.source.validate(transpose_action(self.target.validate(vector))[0])

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.mv(vector)

    def _materialize(self, /) -> Array:
        if self.batch_shape:
            if not isinstance(self.operator.source, ArraySpace) or not isinstance(
                self.operator.target, ArraySpace
            ):
                raise ValueError(
                    "Batched adjoint materialization currently requires ArraySpace values."
                )
            matrix = jnp.conj(jnp.swapaxes(self.operator._materialize(), -1, -2))
            target_weights = _array_pairing_action(
                self.operator.target,
                (),
                jnp.ones(self.operator.target.shape, dtype=matrix.dtype),
                inverse=False,
            ).reshape((self.operator.target.size,))
            source_inverse_weights = _array_pairing_action(
                self.operator.source,
                (),
                jnp.ones(self.operator.source.shape, dtype=matrix.dtype),
                inverse=True,
            ).reshape((self.operator.source.size,))
            return (
                source_inverse_weights[..., :, None]
                * matrix
                * target_weights[..., None, :]
            )
        return _materialize_by_basis(self)

    def _assemble_diagonal(self, /) -> Array:
        diagonal = jnp.conj(_assemble_operator_diagonal(self.operator))
        target_weights = _coordinate_pairing_weights(self.operator.target)
        source_weights = _coordinate_pairing_weights(self.operator.source)
        return diagonal * target_weights / source_weights


class BlockLinearOperator(AbstractLinearOperator):
    """Explicit block operator without flattening block vectors during application."""

    blocks: tuple[tuple[AbstractLinearOperator | None, ...], ...]
    source: BlockSpace
    target: BlockSpace

    def __init__(
        self,
        blocks: Sequence[Sequence[AbstractLinearOperator | None]],
        /,
        *,
        source: BlockSpace,
        target: BlockSpace,
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(source, BlockSpace) or not isinstance(target, BlockSpace):
            raise TypeError("source and target must be BlockSpace values.")
        blocks_ = tuple(tuple(row) for row in blocks)
        if len(blocks_) != len(target.spaces) or any(
            len(row) != len(source.spaces) for row in blocks_
        ):
            raise ValueError("Block grid must match target rows and source columns.")
        for row, target_space in zip(blocks_, target.spaces, strict=True):
            for block, source_space in zip(row, source.spaces, strict=True):
                if block is None:
                    continue
                if not isinstance(block, AbstractLinearOperator):
                    raise TypeError("Blocks must be linear operators or None.")
                _same_space(block.source, source_space)
                _same_space(block.target, target_space)
                if block.batch_shape:
                    raise ValueError("Block operators do not yet support batch axes.")
        self.source = source
        self.target = target
        self.blocks = blocks_
        block_diagonal = all(
            row_index == column_index or block is None
            for row_index, row in enumerate(blocks_)
            for column_index, block in enumerate(row)
        )
        properties_ = (
            OperatorProperties(
                block_diagonal=block_diagonal,
                evidence={"block_diagonal": "construction"} if block_diagonal else {},
            )
            if properties is None
            else properties
        )
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties or None.")
        _validate_properties(properties_, source, target)
        if properties_.block_diagonal and not block_diagonal:
            raise ValueError(
                "A block-diagonal claim contradicts the supplied block grid."
            )
        self.properties = properties_
        capabilities = [block.capabilities for row in blocks_ for block in row if block]
        self.capabilities = OperatorCapabilities(
            transpose=all(value.transpose for value in capabilities),
            adjoint=all(value.adjoint for value in capabilities),
            materialize=all(value.materialize for value in capabilities),
            diagonal_assembly=(
                source.compatible(target)
                and all(
                    block is None or block.capabilities.diagonal_assembly
                    for index, row in enumerate(blocks_)
                    for column, block in enumerate(row)
                    if index == column
                )
            ),
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "block",
                "source": source.space_id,
                "target": target.space_id,
                "blocks": [
                    [None if block is None else block.operator_id for block in row]
                    for row in blocks_
                ],
            },
        )

    def mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        values = self.source.validate(vector)
        output: list[PyTree[Array]] = []
        for row, target_space in zip(self.blocks, self.target.spaces, strict=True):
            accumulated = None
            for block, value in zip(row, values, strict=True):
                if block is None:
                    continue
                image = block.mv(value)
                accumulated = (
                    image if accumulated is None else _tree_add(accumulated, image)
                )
            output.append(target_space.zeros() if accumulated is None else accumulated)
        return tuple(output)

    def transpose_mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        values = self.target.validate(vector)
        output: list[PyTree[Array]] = []
        for column, source_space in enumerate(self.source.spaces):
            accumulated = None
            for row, value in zip(self.blocks, values, strict=True):
                block = row[column]
                if block is None:
                    continue
                image = block.transpose_mv(value)
                accumulated = (
                    image if accumulated is None else _tree_add(accumulated, image)
                )
            output.append(source_space.zeros() if accumulated is None else accumulated)
        return tuple(output)

    def adjoint_mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        values = self.target.validate(vector)
        output: list[PyTree[Array]] = []
        for column, source_space in enumerate(self.source.spaces):
            accumulated = None
            for row, value in zip(self.blocks, values, strict=True):
                block = row[column]
                if block is None:
                    continue
                image = block.adjoint_mv(value)
                accumulated = (
                    image if accumulated is None else _tree_add(accumulated, image)
                )
            output.append(source_space.zeros() if accumulated is None else accumulated)
        return tuple(output)

    def _materialize(self, /) -> Array:
        rows: list[Array] = []
        dtype = jnp.result_type(
            *[
                spec.dtype
                for space in self.source.spaces + self.target.spaces
                for spec in jax.tree.leaves(space.structure())
            ]
        )
        for row, target_space in zip(self.blocks, self.target.spaces, strict=True):
            values = [
                jnp.zeros((target_space.size, source_space.size), dtype=dtype)
                if block is None
                else block._materialize()
                for block, source_space in zip(row, self.source.spaces, strict=True)
            ]
            rows.append(jnp.concatenate(values, axis=1))
        return jnp.concatenate(rows, axis=0)

    def _assemble_diagonal(self, /) -> Array:
        values = []
        for index, source_space in enumerate(self.source.spaces):
            block = self.blocks[index][index]
            values.append(
                jnp.zeros((source_space.size,), dtype=_coordinate_dtype(source_space))
                if block is None
                else _assemble_operator_diagonal(block)
            )
        return jnp.concatenate(tuple(values))


def _space_storage_bytes(space: AbstractVectorSpace, /) -> int:
    return int(space.size * np.dtype(_coordinate_dtype(space)).itemsize)


def _operator_action_workspace(
    operator: AbstractLinearOperator,
    /,
) -> tuple[int, bool, str]:
    custom = (
        operator._action_workspace_cost()
        if isinstance(operator, _AbstractCostedLinearOperator)
        else None
    )
    if custom is not None:
        workspace, operation_class = custom
        workspace = int(workspace)
        operation_class = str(operation_class)
        if workspace < 0 or not operation_class:
            raise ValueError(
                "Custom operator action costs must be non-negative and named."
            )
        return workspace, True, operation_class

    from ._sparse_contract import AbstractSparseLinearOperator
    from ._structured_operators import (
        BandedLinearOperator,
        BasePlusLowRankLinearOperator,
        BlockDiagonalLinearOperator,
        DiagonalPlusLowRankLinearOperator,
        KroneckerLinearOperator,
        KroneckerSumLinearOperator,
        LocalBlockDiagonalLinearOperator,
        LowRankLinearOperator,
        PermutationLinearOperator,
        SchurComplementLinearOperator,
        StackedLinearOperator,
        SymmetricLowRankLinearOperator,
        TriangularLinearOperator,
        TridiagonalLinearOperator,
        TwoSidedScaledLinearOperator,
    )
    from ._transform_operators import TransformDiagonalLinearOperator

    batch_count = prod(operator.batch_shape) if operator.batch_shape else 1
    source_bytes = batch_count * _space_storage_bytes(operator.source)
    target_bytes = batch_count * _space_storage_bytes(operator.target)
    vector_bytes = max(source_bytes, target_bytes)

    if isinstance(
        operator,
        (
            DenseLinearOperator,
            DiagonalLinearOperator,
            IdentityLinearOperator,
            AbstractSparseLinearOperator,
            PermutationLinearOperator,
            TriangularLinearOperator,
            LocalBlockDiagonalLinearOperator,
        ),
    ):
        kind = (
            "sparse-action"
            if isinstance(operator, AbstractSparseLinearOperator)
            else "explicit-action"
        )
        return 0, True, kind
    if isinstance(operator, TridiagonalLinearOperator):
        return 2 * vector_bytes, True, "banded-action"
    if isinstance(operator, BandedLinearOperator):
        return 2 * vector_bytes, True, "banded-action"
    if isinstance(operator, TransformDiagonalLinearOperator):
        return 3 * vector_bytes, True, "transform-diagonal-action"
    if isinstance(operator, LowRankLinearOperator):
        rank_bytes = int(
            operator.left_factor.shape[-1] * operator.left_factor.dtype.itemsize
        )
        return rank_bytes, True, "low-rank-action"
    if isinstance(operator, SymmetricLowRankLinearOperator):
        rank_bytes = int(operator.factor.shape[-1] * operator.factor.dtype.itemsize)
        return rank_bytes, True, "low-rank-action"
    if isinstance(operator, DiagonalPlusLowRankLinearOperator):
        rank_bytes = int(
            operator.left_factor.shape[-1] * operator.left_factor.dtype.itemsize
        )
        return target_bytes + rank_bytes, True, "low-rank-action"
    if isinstance(operator, BasePlusLowRankLinearOperator):
        base_workspace, base_exact, _ = _operator_action_workspace(operator.base)
        rank_bytes = int(operator.rank * operator.left_factor.dtype.itemsize)
        return (
            target_bytes + 2 * rank_bytes + base_workspace,
            base_exact,
            "base-plus-low-rank-action",
        )
    if isinstance(operator, BlockDiagonalLinearOperator):
        estimates = tuple(_operator_action_workspace(block) for block in operator.blocks)
        return (
            max((estimate[0] for estimate in estimates), default=0),
            all(estimate[1] for estimate in estimates),
            "block-action",
        )
    if isinstance(operator, KroneckerLinearOperator):
        estimates = tuple(
            _operator_action_workspace(factor) for factor in operator.factors
        )
        return (
            vector_bytes + max((estimate[0] for estimate in estimates), default=0),
            all(estimate[1] for estimate in estimates),
            "tensor-product-action",
        )
    if isinstance(operator, KroneckerSumLinearOperator):
        estimates = tuple(
            _operator_action_workspace(factor) for factor in operator.factors
        )
        return (
            2 * vector_bytes + max((estimate[0] for estimate in estimates), default=0),
            all(estimate[1] for estimate in estimates),
            "tensor-sum-action",
        )
    if isinstance(operator, StackedLinearOperator):
        estimates = tuple(
            _operator_action_workspace(child) for child in operator.operators
        )
        accumulation = target_bytes if operator.axis == "horizontal" else 0
        return (
            accumulation + max((estimate[0] for estimate in estimates), default=0),
            all(estimate[1] for estimate in estimates),
            "stacked-action",
        )
    if isinstance(operator, SchurComplementLinearOperator):
        estimates = tuple(
            _operator_action_workspace(child)
            for child in (
                operator.diagonal_block,
                operator.upper_block,
                operator.lower_block,
            )
        )
        return (
            3 * vector_bytes + max((estimate[0] for estimate in estimates), default=0),
            False,
            "schur-action",
        )
    if isinstance(operator, TwoSidedScaledLinearOperator):
        workspace, exact, _ = _operator_action_workspace(operator.operator)
        return 2 * vector_bytes + workspace, exact, "two-sided-scaled-action"
    if isinstance(operator, ScaledLinearOperator):
        workspace, exact, _ = _operator_action_workspace(operator.operator)
        return target_bytes + workspace, exact, "scaled-action"
    if isinstance(operator, SumLinearOperator):
        left = _operator_action_workspace(operator.left)
        right = _operator_action_workspace(operator.right)
        return (
            target_bytes + max(left[0], right[0]),
            left[1] and right[1],
            "sum-action",
        )
    if isinstance(operator, ComposedLinearOperator):
        left = _operator_action_workspace(operator.left)
        right = _operator_action_workspace(operator.right)
        intermediate_bytes = batch_count * _space_storage_bytes(operator.right.target)
        return (
            max(right[0], intermediate_bytes + left[0]),
            left[1] and right[1],
            "composition-action",
        )
    if isinstance(operator, TransposeLinearOperator):
        workspace, exact, kind = _operator_action_workspace(operator.operator)
        return workspace, exact, kind
    if isinstance(operator, AdjointLinearOperator):
        workspace, exact, _ = _operator_action_workspace(operator.operator)
        return vector_bytes + workspace, exact, "adjoint-action"
    if isinstance(operator, BlockLinearOperator):
        workspaces = []
        exact = True
        for row, target in zip(operator.blocks, operator.target.spaces, strict=True):
            children = tuple(
                _operator_action_workspace(child) for child in row if child is not None
            )
            if children:
                workspaces.append(
                    _space_storage_bytes(target) + max(child[0] for child in children)
                )
                exact = exact and all(child[1] for child in children)
        return max(workspaces, default=0), exact, "block-action"
    return vector_bytes, False, "opaque-action"


def estimate_operator_action_cost(
    operator: AbstractLinearOperator,
    /,
) -> OperatorActionCostEstimate:
    """Estimate resident operator state and structural action scratch."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    from ._sparse_contract import AbstractSparseLinearOperator

    workspace, exact, operation_class = _operator_action_workspace(operator)
    resident = (
        _array_tree_storage_bytes((operator, operator.sparse_storage()))
        if isinstance(operator, AbstractSparseLinearOperator)
        else _array_tree_storage_bytes(operator)
    )
    reason = (
        "exact structural action accounting"
        if exact
        else "opaque kernel or nested inverse action may require additional scratch"
    )
    return OperatorActionCostEstimate(
        operator_id=operator.operator_id,
        storage_bytes=resident,
        apply_workspace_bytes_per_rhs=workspace,
        operation_class=operation_class,
        exact=exact,
        reason=reason,
    )


def transpose(operator: AbstractLinearOperator, /) -> AbstractLinearOperator:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not operator.capabilities.transpose:
        raise LinearCapabilityError("Operator does not declare transpose capability.")
    if isinstance(operator, TransposeLinearOperator):
        return operator.operator
    return TransposeLinearOperator(operator)


def adjoint(operator: AbstractLinearOperator, /) -> AbstractLinearOperator:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not operator.capabilities.adjoint:
        raise LinearCapabilityError("Operator does not declare adjoint capability.")
    if isinstance(operator, AdjointLinearOperator):
        return operator.operator
    return AdjointLinearOperator(operator)


__all__ = [
    "AbstractLinearOperator",
    "AdjointLinearOperator",
    "BlockLinearOperator",
    "ComposedLinearOperator",
    "DenseLinearOperator",
    "DiagonalLinearOperator",
    "estimate_operator_action_cost",
    "FunctionLinearOperator",
    "IdentityLinearOperator",
    "JacobianLinearOperator",
    "ScaledLinearOperator",
    "SumLinearOperator",
    "TransposeLinearOperator",
    "adjoint",
    "transpose",
]
