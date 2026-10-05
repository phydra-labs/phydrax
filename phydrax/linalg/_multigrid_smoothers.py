#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array, core as jax_core
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._trainable import NonTrainableState
from ..typing import checked, parse
from ._costs import _array_tree_storage_bytes, PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator, DenseLinearOperator
from ._preconditioner_properties import PreconditionerProperties
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import _lowered_space, AbstractPreconditionerBuilder
from ._properties import LinearCapabilityError, PropertyEvidence
from ._spaces import _coordinate_dtype
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._sparse_triangular import (
    _analysis_storage_bytes,
    analyze_sparse_triangular,
    SparseTriangularFactor,
    SparseTriangularStatus,
)


GaussSeidelDirection: TypeAlias = Literal["forward", "backward", "symmetric"]


def _explicit_csr(operator: AbstractLinearOperator, /) -> sp.csr_matrix:
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        if storage.batch_shape:
            raise ValueError("Gauss-Seidel requires unbatched CSR values.")
        if not storage.canonical or not storage.sorted_indices:
            raise ValueError("Gauss-Seidel requires canonical sorted CSR storage.")
        matrix = sp.csr_matrix(
            (
                np.asarray(storage.values),
                np.asarray(storage.indices),
                np.asarray(storage.indptr),
            ),
            shape=storage.shape,
        )
    elif isinstance(operator, DenseLinearOperator):
        matrix = sp.csr_matrix(np.asarray(operator.matrix))
    else:
        raise TypeError("Gauss-Seidel requires an explicit dense or sparse operator.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gauss-Seidel requires a square operator.")
    diagonal = matrix.diagonal()
    if np.any(~np.isfinite(matrix.data)) or np.any(~np.isfinite(diagonal)):
        raise ValueError("Gauss-Seidel requires finite operator entries.")
    if np.any(diagonal == 0):
        raise ValueError("Gauss-Seidel requires a nonzero diagonal.")
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _triangular_storage(
    matrix: sp.csr_matrix,
    triangle: Literal["lower", "upper"],
    relaxation: float,
    dtype: np.dtype,
    /,
) -> SparseStorage:
    triangular = (
        sp.tril(matrix, format="csr")
        if triangle == "lower"
        else sp.triu(matrix, format="csr")
    )
    triangular.setdiag(matrix.diagonal() / relaxation)
    triangular.sum_duplicates()
    triangular.sort_indices()
    index_dtype = np.int32 if matrix.shape[0] <= np.iinfo(np.int32).max else np.int64
    # Values are rounded once from the exact coordinate-precision splitting.
    return SparseStorage(
        jnp.asarray(triangular.data.astype(dtype)),
        jnp.asarray(triangular.indices, dtype=index_dtype),
        jnp.asarray(triangular.indptr, dtype=index_dtype),
        shape=triangular.shape,
    )


def _sweep_ordering(ordering: ArrayLike | None, size: int | None, /) -> np.ndarray | None:
    """Validated sweep permutation; ``size=None`` takes the vector length."""
    if ordering is None:
        return None
    permutation = np.asarray(jax.device_get(ordering))
    expected = permutation.size if size is None else size
    if (
        permutation.shape != (expected,)
        or not np.issubdtype(permutation.dtype, np.integer)
        or not np.array_equal(np.sort(permutation), np.arange(expected))
    ):
        raise ValueError("ordering must be a permutation of the coordinate indices.")
    index_dtype = np.int32 if expected <= np.iinfo(np.int32).max else np.int64
    return permutation.astype(index_dtype)


def _factor(
    storage: SparseStorage,
    triangle: Literal["lower", "upper"],
    /,
    *,
    previous: SparseTriangularFactor | None = None,
) -> SparseTriangularFactor:
    analysis = analyze_sparse_triangular(storage, triangle=triangle)
    if previous is not None:
        if previous.analysis.pattern_id != analysis.pattern_id:
            raise ValueError(
                "Gauss-Seidel numeric refresh requires an unchanged triangular pattern."
            )
        analysis = previous.analysis
    return SparseTriangularFactor(analysis, storage.values)


def _sparse_pattern_id(storage: SparseStorage, /) -> str | None:
    """Concrete canonical-pattern identity; ``None`` for a traced pattern."""
    if isinstance(storage.indices, jax_core.Tracer) or isinstance(
        storage.indptr, jax_core.Tracer
    ):
        return None
    return canonical_fingerprint(
        {
            "kind": "gauss-seidel-pattern",
            "shape": list(storage.shape),
            "indices": array_tree_fingerprint(storage.indices),
            "indptr": array_tree_fingerprint(storage.indptr),
        }
    )


def _triangular_route(
    storage: SparseStorage,
    permutation: np.ndarray | None,
    triangle: Literal["lower", "upper"],
    /,
) -> tuple[Array, Array]:
    """Canonical value position and diagonal flag of each stored factor entry.

    The route mirrors ``_triangular_storage`` on positions instead of values,
    so numeric refresh gathers the factor from new values without host reads.
    """
    count = storage.indices.shape[0]
    positions = sp.csr_matrix(
        (
            np.arange(1, count + 1, dtype=np.int64),
            np.asarray(storage.indices),
            np.asarray(storage.indptr),
        ),
        shape=storage.shape,
    )
    if permutation is not None:
        positions = positions[permutation][:, permutation].tocsr()
    part = (
        sp.tril(positions, format="csr")
        if triangle == "lower"
        else sp.triu(positions, format="csr")
    )
    part.sort_indices()
    rows = np.repeat(np.arange(part.shape[0], dtype=np.int64), np.diff(part.indptr))
    return jnp.asarray(part.data - 1), jnp.asarray(rows == part.indices)


def _routed_factor(
    previous: SparseTriangularFactor,
    route: tuple[Array, Array],
    values: Array,
    relaxation: float,
    dtype: np.dtype,
    /,
) -> SparseTriangularFactor:
    sources, diagonal = route
    gathered = values[sources]
    factor_values = jnp.where(diagonal, gathered / relaxation, gathered).astype(dtype)
    return SparseTriangularFactor(previous.analysis, factor_values)


def _validated_refresh_values(storage: SparseStorage, /) -> Array:
    """Refreshed canonical values under the finite, nonzero-diagonal contract."""
    if storage.batch_shape:
        raise ValueError("Gauss-Seidel requires unbatched CSR values.")
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError("Gauss-Seidel requires canonical sorted CSR storage.")
    values = storage.values
    rows = jnp.repeat(
        jnp.arange(storage.shape[0], dtype=storage.indices.dtype),
        jnp.diff(storage.indptr),
        total_repeat_length=values.shape[0],
    )
    on_diagonal = rows == storage.indices
    diagonal_count = (
        jnp.zeros((storage.shape[0],), dtype=jnp.int32)
        .at[rows]
        .add(jnp.where(on_diagonal & (values != 0), 1, 0))
    )
    values = eqx.error_if(
        values,
        ~jnp.all(jnp.isfinite(values)),
        "Gauss-Seidel requires finite operator entries.",
    )
    return eqx.error_if(
        values,
        jnp.any(diagonal_count == 0),
        "Gauss-Seidel requires a nonzero diagonal.",
    )


def _checked_route(
    storage: SparseStorage,
    permutation: np.ndarray | None,
    triangle: Literal["lower", "upper"],
    factor: SparseTriangularFactor,
    /,
) -> tuple[Array, Array]:
    route = _triangular_route(storage, permutation, triangle)
    if route[0].shape != factor.values.shape:
        raise RuntimeError("Gauss-Seidel refresh route does not match its factor.")
    return route


class GaussSeidelPreconditioner(AbstractPreconditioner, NonTrainableState):
    """Prepared forward, backward, or multiplicative symmetric sweep.

    ``ordering`` sweeps coordinates in the given permutation: the triangular
    splitting is taken of ``A[ordering][:, ordering]``. A single-direction
    sweep may be stored and applied in a lower ``compute_dtype``; the symmetric
    sweep forms an inner residual and therefore stays in coordinate precision.
    """

    operator: AbstractLinearOperator
    forward_factor: SparseTriangularFactor | None
    backward_factor: SparseTriangularFactor | None
    ordering: Array | None
    inverse_ordering: Array | None
    forward_route: tuple[Array, Array] | None
    backward_route: tuple[Array, Array] | None
    direction: GaussSeidelDirection = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    pattern_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        direction: GaussSeidelDirection = "symmetric",
        relaxation: float = 1.0,
        ordering: ArrayLike | None = None,
        compute_dtype: str | None = None,
        previous: "GaussSeidelPreconditioner | None" = None,
    ) -> None:
        direction = parse(direction, GaussSeidelDirection, "direction")
        omega = float(relaxation)
        if not isfinite(omega) or omega <= 0.0 or omega >= 2.0:
            raise ValueError("Gauss-Seidel relaxation must lie strictly between 0 and 2.")
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError("Gauss-Seidel requires an unbatched endomorphism.")
        if compute_dtype is not None and direction == "symmetric":
            raise LinearCapabilityError(
                "Symmetric Gauss-Seidel forms an inner residual and has no lower-precision contract."
            )
        space = (
            operator.source
            if compute_dtype is None
            else _lowered_space(operator.source, compute_dtype)
        )
        value_dtype = np.dtype(_coordinate_dtype(space))
        storage = (
            operator.sparse_storage()
            if isinstance(operator, AbstractSparseLinearOperator)
            else None
        )
        if (
            storage is not None
            and previous is not None
            and previous.pattern_id is not None
            and previous.direction == direction
            and previous.relaxation == omega
            and previous.space.compatible(space)
        ):
            # Numeric refresh over the unchanged canonical pattern: factor values
            # are gathered on device, so the refresh is traceable.
            pattern_id = _sparse_pattern_id(storage)
            if pattern_id is not None and pattern_id != previous.pattern_id:
                raise ValueError(
                    "Gauss-Seidel numeric refresh requires an unchanged triangular pattern."
                )
            values = _validated_refresh_values(storage)
            permutation = (
                None if previous.ordering is None else np.asarray(previous.ordering)
            )
            forward_route, backward_route = (
                previous.forward_route,
                previous.backward_route,
            )
            forward = (
                None
                if previous.forward_factor is None or forward_route is None
                else _routed_factor(
                    previous.forward_factor, forward_route, values, omega, value_dtype
                )
            )
            backward = (
                None
                if previous.backward_factor is None or backward_route is None
                else _routed_factor(
                    previous.backward_factor, backward_route, values, omega, value_dtype
                )
            )
            pattern_id = previous.pattern_id
        else:
            matrix = _explicit_csr(operator)
            permutation = _sweep_ordering(ordering, matrix.shape[0])
            if permutation is not None:
                matrix = matrix[permutation][:, permutation].tocsr()
                matrix.sort_indices()
            old_forward = None if previous is None else previous.forward_factor
            old_backward = None if previous is None else previous.backward_factor
            forward = (
                _factor(
                    _triangular_storage(matrix, "lower", omega, value_dtype),
                    "lower",
                    previous=old_forward,
                )
                if direction in ("forward", "symmetric")
                else None
            )
            backward = (
                _factor(
                    _triangular_storage(matrix, "upper", omega, value_dtype),
                    "upper",
                    previous=old_backward,
                )
                if direction in ("backward", "symmetric")
                else None
            )
            pattern_id = None if storage is None else _sparse_pattern_id(storage)
            forward_route = (
                None
                if forward is None or storage is None
                else _checked_route(storage, permutation, "lower", forward)
            )
            backward_route = (
                None
                if backward is None or storage is None
                else _checked_route(storage, permutation, "upper", backward)
            )
        symmetric = direction == "symmetric" and operator.properties.certifies(
            "self_adjoint"
        )
        positive = symmetric and operator.properties.certifies("positive_definite")
        evidence: dict[str, PropertyEvidence] = {
            "linear": "construction",
            "stationary": "construction",
            **({"self_adjoint": "transformed"} if symmetric else {}),
            **({"positive_definite": "transformed"} if positive else {}),
        }
        self.operator = operator
        self.forward_factor = forward
        self.backward_factor = backward
        with jax.ensure_compile_time_eval():
            self.ordering = None if permutation is None else jnp.asarray(permutation)
            self.inverse_ordering = (
                None
                if permutation is None
                else jnp.asarray(np.argsort(permutation).astype(permutation.dtype))
            )
        self.forward_route = forward_route
        self.backward_route = backward_route
        self.pattern_id = pattern_id
        self.direction = direction
        self.relaxation = omega
        self.space = space
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=symmetric,
            positive_definite=positive,
            evidence=evidence,
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "gauss-seidel",
                "operator": operator.operator_id,
                "direction": direction,
                "relaxation": omega,
                "ordering": (
                    None if permutation is None else array_tree_fingerprint(permutation)
                ),
                "compute_dtype": value_dtype.name,
                "forward_pattern": (
                    None if forward is None else forward.analysis.pattern_id
                ),
                "backward_pattern": (
                    None if backward is None else backward.analysis.pattern_id
                ),
            }
        )

    def _solve(
        self,
        factor: SparseTriangularFactor,
        residual: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        coordinates = self.space.flatten(self.space.validate(residual))
        if self.ordering is not None:
            coordinates = coordinates[self.ordering]
        result = factor.solve(coordinates)
        value = eqx.error_if(
            result.value,
            result.status != int(SparseTriangularStatus.SUCCESS),
            "Gauss-Seidel triangular sweep failed.",
        )
        if self.inverse_ordering is not None:
            value = value[self.inverse_ordering]
        return self.space.unflatten(value)

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        del iteration
        residual_ = self.space.validate(residual)
        if self.direction == "forward":
            if self.forward_factor is None:
                raise RuntimeError("Prepared forward factor is missing.")
            return self._solve(self.forward_factor, residual_)
        if self.direction == "backward":
            if self.backward_factor is None:
                raise RuntimeError("Prepared backward factor is missing.")
            return self._solve(self.backward_factor, residual_)
        if self.forward_factor is None or self.backward_factor is None:
            raise RuntimeError("Prepared symmetric factors are missing.")
        first = self._solve(self.forward_factor, residual_)
        defect = jax.tree.map(
            lambda rhs, image: rhs - image,
            residual_,
            self.operator.mv(first),
        )
        second = self._solve(self.backward_factor, defect)
        return jax.tree.map(lambda left, right: left + right, first, second)

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        if not self.space.compatible(setup_operator.source):
            raise ValueError("Gauss-Seidel and setup operator spaces must match.")
        # Values plus every resident schedule array of both orientations,
        # including the padded level blocks, and the refresh routes.
        storage = _array_tree_storage_bytes(
            (
                self.forward_factor,
                self.backward_factor,
                self.forward_route,
                self.backward_route,
            )
        )
        itemsize = self.space.flatten(
            self.space.unflatten(jnp.zeros(self.space.size))
        ).dtype.itemsize
        multiplier = 4 if self.direction == "symmetric" else 2
        return PreconditionerCostEstimate(
            component=self.preconditioner_id,
            storage_bytes=storage,
            apply_workspace_bytes_per_rhs=multiplier * self.space.size * itemsize,
            reason="prepared level-scheduled Gauss-Seidel sweep",
        )


class GaussSeidelPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Build a reusable Gauss-Seidel smoother from explicit operator storage.

    ``ordering`` fixes the sweep permutation (for example a stable-ID order),
    making the action independent of coordinate numbering. The symmetric sweep
    is self-adjoint/positive only when the operator certifies those properties.
    """

    ordering: Array | None
    direction: GaussSeidelDirection = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        direction: GaussSeidelDirection = "symmetric",
        relaxation: float = 1.0,
        ordering: ArrayLike | None = None,
    ) -> None:
        direction = parse(direction, GaussSeidelDirection, "direction")
        omega = float(relaxation)
        if not isfinite(omega) or omega <= 0.0 or omega >= 2.0:
            raise ValueError("Gauss-Seidel relaxation must lie strictly between 0 and 2.")
        permutation = _sweep_ordering(ordering, None)
        self.ordering = None if permutation is None else jnp.asarray(permutation)
        self.direction = direction
        self.relaxation = omega
        self._builder_id = canonical_fingerprint(
            {
                "kind": "gauss-seidel-builder",
                "direction": direction,
                "relaxation": omega,
                "ordering": (
                    None if permutation is None else array_tree_fingerprint(permutation)
                ),
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    @checked
    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        if setup_operator.batch_shape or not setup_operator.source.compatible(
            setup_operator.target
        ):
            raise ValueError("Gauss-Seidel requires an unbatched endomorphism.")
        symmetric = self.direction == "symmetric" and setup_operator.properties.certifies(
            "self_adjoint"
        )
        positive = symmetric and setup_operator.properties.certifies("positive_definite")
        evidence: dict[str, PropertyEvidence] = {
            "linear": "construction",
            "stationary": "construction",
            **({"self_adjoint": "transformed"} if symmetric else {}),
            **({"positive_definite": "transformed"} if positive else {}),
        }
        return PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=symmetric,
            positive_definite=positive,
            evidence=evidence,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        if not isinstance(
            setup_operator,
            (DenseLinearOperator, AbstractSparseLinearOperator),
        ):
            return PreconditionerCostEstimate(
                component=self.builder_id,
                accepted=False,
                reason="Gauss-Seidel requires explicit dense or sparse operator storage",
            )
        matrix = _explicit_csr(setup_operator)
        return self._sweep_cost(
            matrix, matrix.data.dtype.itemsize, "explicit triangular sweep preparation"
        )

    def _sweep_cost(
        self, matrix: sp.csr_matrix, itemsize: int, reason: str, /
    ) -> PreconditionerCostEstimate:
        factors = 2 if self.direction == "symmetric" else 1
        index_size = matrix.indices.dtype.itemsize
        ordering_bytes = 0 if self.ordering is None else 2 * self.ordering.nbytes
        # Each factor also keeps its numeric-refresh route: one value position
        # and one diagonal flag per stored entry.
        route_bytes = np.dtype(np.int64).itemsize + np.dtype(np.bool_).itemsize
        storage = ordering_bytes + factors * (
            matrix.nnz * (itemsize + route_bytes)
            + _analysis_storage_bytes(matrix.shape[0], matrix.nnz, index_size)
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=int(storage),
            preparation_workspace_bytes=int(storage),
            apply_workspace_bytes_per_rhs=int(
                (4 if self.direction == "symmetric" else 2) * matrix.shape[0] * itemsize
            ),
            accepted=True,
            reason=reason,
        )

    def lowered_cost(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        if self.direction == "symmetric":
            raise LinearCapabilityError(
                "Symmetric Gauss-Seidel forms an inner residual and has no lower-precision contract."
            )
        cost = self.cost_for(setup_operator, materialization=materialization)
        if not cost.accepted:
            return cost
        return self._sweep_cost(
            _explicit_csr(setup_operator),
            jnp.dtype(compute_dtype).itemsize,
            f"triangular sweep stored/applied in {compute_dtype} with explicit coordinate casts",
        )

    def prepare_lowered(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy,
        previous: AbstractPreconditioner | None = None,
    ) -> GaussSeidelPreconditioner:
        del materialization
        if previous is not None and not isinstance(previous, GaussSeidelPreconditioner):
            raise TypeError("Gauss-Seidel refresh requires GaussSeidelPreconditioner.")
        return GaussSeidelPreconditioner(
            setup_operator,
            direction=self.direction,
            relaxation=self.relaxation,
            ordering=self.ordering,
            compute_dtype=compute_dtype,
            previous=previous,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> GaussSeidelPreconditioner:
        del materialization
        return GaussSeidelPreconditioner(
            setup_operator,
            direction=self.direction,
            relaxation=self.relaxation,
            ordering=self.ordering,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> GaussSeidelPreconditioner:
        del materialization
        if not isinstance(preconditioner, GaussSeidelPreconditioner):
            raise TypeError("Gauss-Seidel refresh requires GaussSeidelPreconditioner.")
        if preconditioner.direction != self.direction:
            raise ValueError("Gauss-Seidel refresh cannot change sweep direction.")
        return GaussSeidelPreconditioner(
            setup_operator,
            direction=self.direction,
            relaxation=self.relaxation,
            ordering=self.ordering,
            previous=preconditioner,
        )


__all__ = [
    "GaussSeidelDirection",
    "GaussSeidelPreconditioner",
    "GaussSeidelPreconditionerBuilder",
]
