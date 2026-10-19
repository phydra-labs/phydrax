#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState
from ._costs import _array_tree_storage_bytes, PreconditionerCostEstimate
from ._local_blocks import (
    LocalBlockFactorization,
    prepare_local_block_factorization,
    solve_local_blocks_detailed,
)
from ._materialization import MaterializationPolicy
from ._named_blocks import BlockSelection, CoordinateBlock
from ._operators import AbstractLinearOperator, estimate_operator_action_cost
from ._preconditioner_properties import PreconditionerProperties
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import AbstractPreconditionerBuilder, PreconditionerRefreshPolicy
from ._properties import LinearCapabilityError
from ._spaces import ArraySpace, coordinate_dtype


def _positive_integer(value: int, name: str, /) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an exact integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_operator(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("setup_operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Block probing requires an unbatched endomorphism.")
    if operator.source.size < 1:
        raise ValueError("Block probing requires nonempty coordinates.")
    if not jnp.issubdtype(coordinate_dtype(operator.source), jnp.inexact):
        raise TypeError("Block probing requires an inexact coordinate dtype.")


def _validate_selection(
    selection: BlockSelection, local_size: int, operator: AbstractLinearOperator, /
) -> None:
    _validate_operator(operator)
    if not selection.source.compatible(operator.source):
        raise ValueError("Local selection must identify the native operator source.")
    if not selection.covers_source:
        raise ValueError("Local selection must be a complete coordinate partition.")
    if any(member.size != local_size for member in selection.target.spaces):
        raise ValueError("Every declared local member must have local_size coordinates.")


def _fixed_properties() -> PreconditionerProperties:
    return PreconditionerProperties(
        linear=True,
        stationary=True,
        evidence={"linear": "construction", "stationary": "construction"},
    )


def _probe_coordinate_action(
    operator: AbstractLinearOperator, native_column: Array, /
) -> Array:
    """Return one exact native coordinate column without materialization.

    Internal callers own operator validation and scalar coordinate bounds through
    a validated BlockSelection. Batch only a resource-admitted chunk with
    lax.map/vmap; retain or inspect rows before advancing to the next chunk.
    """
    coordinates = jnp.arange(operator.source.size, dtype=jnp.int32)
    basis = (coordinates == native_column).astype(coordinate_dtype(operator.source))
    return operator.target.flatten(operator.mv(operator.source.unflatten(basis)))


def _probe_local_blocks(
    operator: AbstractLinearOperator,
    selection: BlockSelection,
    local_size: int,
    probe_chunk_size: int,
    /,
) -> Array:
    """Probe exact principal blocks, retaining only local rows of each action.

    lax.map generates at most one bounded batch of basis vectors. There is no
    global identity, global Jacobian, or simultaneous excitation of distinct
    blocks: offdiagonal dependencies cannot contaminate a probed diagonal block.
    """
    dimension = operator.source.size
    positions = jnp.concatenate(
        tuple(
            jnp.arange(start, stop, dtype=jnp.int32)
            for ranges in selection.member_ranges
            for start, stop in ranges
        )
    )
    native_indices = jnp.arange(dimension, dtype=jnp.int32)

    def probe(column: Array) -> Array:
        value = _probe_coordinate_action(operator, positions[column])
        local_rows = jax.lax.dynamic_slice_in_dim(
            positions, (column // local_size) * local_size, local_size
        )
        return value[local_rows]

    columns = jax.lax.map(
        probe, native_indices, batch_size=min(probe_chunk_size, dimension)
    )
    return columns.reshape((-1, local_size, local_size)).swapaxes(-1, -2)


def _probing_cost(
    identifier: str,
    operator: AbstractLinearOperator,
    local_size: int,
    probe_chunk_size: int,
    max_workspace_bytes: int,
    max_output_bytes: int,
    /,
) -> PreconditionerCostEstimate:
    _validate_operator(operator)
    dimension = operator.source.size
    dtype = coordinate_dtype(operator.source)
    real_itemsize = jnp.empty((), dtype=dtype).real.dtype.itemsize
    block_bytes = dimension * local_size * dtype.itemsize
    storage = (
        block_bytes
        + dimension * (jnp.dtype(jnp.int32).itemsize + real_itemsize)
        + (dimension // local_size) * jnp.dtype(jnp.bool_).itemsize
    )
    action_cost = estimate_operator_action_cost(operator)
    chunk_size = min(probe_chunk_size, dimension)
    workspace = (
        3 * block_bytes
        + 2 * dimension * jnp.dtype(jnp.int32).itemsize
        + chunk_size
        * (2 * dimension * dtype.itemsize + action_cost.apply_workspace_bytes_per_rhs)
    )
    # The operator and its stored linearization tape are shared by every probe.
    # Admit the resident state once, separately from batched action scratch.
    retained = action_cost.storage_bytes
    peak = retained + workspace
    accepted = storage <= max_output_bytes and peak <= max_workspace_bytes
    accounting = (
        f"local LU output={storage} bytes; retained operator={retained} bytes; "
        f"additional probing/factor scratch={workspace} bytes; peak={peak} bytes; "
        f"output/peak limits={max_output_bytes}/{max_workspace_bytes} bytes"
    )
    reason = (
        (
            "exact selected local coordinate probing and native LU; "
            if accepted
            else "local LU resource refusal; "
        )
        + accounting
        + "; "
        + action_cost.reason
    )
    return PreconditionerCostEstimate(
        component=identifier,
        storage_bytes=storage,
        preparation_workspace_bytes=workspace,
        apply_workspace_bytes_per_rhs=3 * dimension * dtype.itemsize,
        setup_matvec_count=dimension,
        accepted=accepted,
        reason=reason,
    )


@final
class ProbedBlockSolveResult(StrictModule):
    """Native-space solution and independent factor/application failures."""

    value: PyTree[Array]
    failed_blocks: Array


@final
class ProbedBlockPreconditioner(AbstractPreconditioner, NonTrainableState):
    """Selected local LU actions; failed components are never successful zeros."""

    selection: BlockSelection = fixed_field()
    factorization: LocalBlockFactorization
    builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        selection: BlockSelection,
        factorization: LocalBlockFactorization,
        /,
        *,
        builder_id: str,
        setup_operator_id: str,
    ) -> None:
        if not isinstance(selection, BlockSelection):
            raise TypeError("selection must be a BlockSelection.")
        if not isinstance(factorization, LocalBlockFactorization):
            raise TypeError("factorization must be a LocalBlockFactorization.")
        if not selection.covers_source:
            raise ValueError("Probed block action requires a complete selection.")
        if len(selection.target.spaces) != factorization.num_blocks or any(
            member.size != factorization.block_size for member in selection.target.spaces
        ):
            raise ValueError("Selection members must match the factored local blocks.")
        if factorization.factors.dtype != coordinate_dtype(selection.source):
            raise TypeError("Local factors must use the native coordinate dtype.")
        if not builder_id or not setup_operator_id:
            raise ValueError("Builder and setup operator identifiers must be nonempty.")
        self.selection = selection
        self.factorization = factorization
        self.builder_id = builder_id
        self.space = selection.source
        self.properties = _fixed_properties()
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "probed-local-lu-preconditioner",
                "builder": builder_id,
                "selection": selection.selection_id,
                "setup_operator": setup_operator_id,
            }
        )

    @property
    def rank(self) -> None:
        """Rank is not inferred from LU failure flags."""
        return None

    @property
    def condition_estimate(self) -> None:
        """No condition estimator is run during local preparation."""
        return None

    @property
    def rank_available(self) -> bool:
        return False

    @property
    def condition_available(self) -> bool:
        return False

    @property
    def preparation_failed(self) -> Array:
        return jnp.any(self.factorization.failed_blocks)

    def apply_detailed(self, residual: PyTree[ArrayLike], /) -> ProbedBlockSolveResult:
        selected = self.selection.restrict(residual)
        coordinates = self.selection.target.flatten(selected)
        grouped = coordinates.reshape(
            (self.factorization.num_blocks, self.factorization.block_size)
        )
        result = solve_local_blocks_detailed(self.factorization, grouped)
        # The canonical local solver masks failures. Restore explicit invalid
        # values at this consumer boundary; detailed status retains each failure.
        values = jnp.where(
            result.failed_blocks[:, None],
            jnp.full((), jnp.nan, dtype=result.value.dtype),
            result.value,
        )
        value = self.selection.prolong(
            self.selection.target.unflatten(values.reshape((-1,)))
        )
        return ProbedBlockSolveResult(value=value, failed_blocks=result.failed_blocks)

    def apply(
        self,
        residual: PyTree[ArrayLike],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        del iteration
        result = self.apply_detailed(residual)
        return result.value

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_selection(self.selection, self.factorization.block_size, setup_operator)
        return PreconditionerCostEstimate(
            component=self.preconditioner_id,
            storage_bytes=_array_tree_storage_bytes(self),
            apply_workspace_bytes_per_rhs=3
            * self.space.size
            * coordinate_dtype(self.space).itemsize,
            reason="native local LU factors, pivots, failure flags, and mapped solve workspace",
        )


@final
class MatrixFreeLocalBlockPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Bounded exact local probing of an explicitly declared complete partition.

    Workspace admission includes retained native operator/tape state once plus
    owned probing/factor buffers and the native operator action scratch estimate.
    Opaque scratch remains explicitly estimated. No sparsity/coloring is assumed.
    """

    selection: BlockSelection = fixed_field()
    local_size: int = eqx.field(static=True)
    probe_chunk_size: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)
    max_output_bytes: int = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        selection: BlockSelection,
        /,
        *,
        local_size: int,
        probe_chunk_size: int,
        max_workspace_bytes: int,
        max_output_bytes: int,
    ) -> None:
        if not isinstance(selection, BlockSelection):
            raise TypeError("selection must be a BlockSelection.")
        size = _positive_integer(local_size, "local_size")
        chunk = _positive_integer(probe_chunk_size, "probe_chunk_size")
        workspace = _positive_integer(max_workspace_bytes, "max_workspace_bytes")
        output = _positive_integer(max_output_bytes, "max_output_bytes")
        if not selection.covers_source or any(
            member.size != size for member in selection.target.spaces
        ):
            raise ValueError(
                "Selection must completely partition the source into equal local_size members."
            )
        self.selection = selection
        self.local_size = size
        self.probe_chunk_size = chunk
        self.max_workspace_bytes = workspace
        self.max_output_bytes = output
        self._builder_id = canonical_fingerprint(
            {
                "kind": "matrix-free-local-block-preconditioner-builder",
                "selection": selection.selection_id,
                "local_size": size,
                "probe_chunk_size": chunk,
                "max_workspace_bytes": workspace,
                "max_output_bytes": output,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self, setup_operator: AbstractLinearOperator, /
    ) -> PreconditionerProperties:
        _validate_selection(self.selection, self.local_size, setup_operator)
        return _fixed_properties()

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_selection(self.selection, self.local_size, setup_operator)
        return _probing_cost(
            self.builder_id,
            setup_operator,
            self.local_size,
            self.probe_chunk_size,
            self.max_workspace_bytes,
            self.max_output_bytes,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> ProbedBlockPreconditioner:
        cost = self.cost_for(setup_operator, materialization=materialization)
        if not cost.accepted:
            raise LinearCapabilityError(cost.reason)
        blocks = _probe_local_blocks(
            setup_operator, self.selection, self.local_size, self.probe_chunk_size
        )
        return ProbedBlockPreconditioner(
            self.selection,
            prepare_local_block_factorization(blocks),
            builder_id=self.builder_id,
            setup_operator_id=setup_operator.operator_id,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> ProbedBlockPreconditioner:
        if not isinstance(preconditioner, ProbedBlockPreconditioner):
            raise TypeError("Local numeric refresh requires a ProbedBlockPreconditioner.")
        if preconditioner.builder_id != self.builder_id:
            raise ValueError(
                "Local refresh must preserve the builder and coordinate partition."
            )
        return self.prepare(setup_operator, materialization=materialization)


@final
class BoundedLUPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Bounded coarse inverse action using native LU, never an explicit inverse."""

    max_size: int = eqx.field(static=True)
    probe_chunk_size: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)
    max_output_bytes: int = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_size: int,
        probe_chunk_size: int,
        max_workspace_bytes: int,
        max_output_bytes: int,
    ) -> None:
        size = _positive_integer(max_size, "max_size")
        chunk = _positive_integer(probe_chunk_size, "probe_chunk_size")
        workspace = _positive_integer(max_workspace_bytes, "max_workspace_bytes")
        output = _positive_integer(max_output_bytes, "max_output_bytes")
        self.max_size = size
        self.probe_chunk_size = chunk
        self.max_workspace_bytes = workspace
        self.max_output_bytes = output
        self._builder_id = canonical_fingerprint(
            {
                "kind": "bounded-lu-preconditioner-builder",
                "max_size": size,
                "probe_chunk_size": chunk,
                "max_workspace_bytes": workspace,
                "max_output_bytes": output,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self, setup_operator: AbstractLinearOperator, /
    ) -> PreconditionerProperties:
        _validate_operator(setup_operator)
        return _fixed_properties()

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_operator(setup_operator)
        dimension = setup_operator.source.size
        cost = _probing_cost(
            self.builder_id,
            setup_operator,
            dimension,
            self.probe_chunk_size,
            self.max_workspace_bytes,
            self.max_output_bytes,
        )
        entries = dimension * dimension
        matrix_bytes = entries * coordinate_dtype(setup_operator.source).itemsize
        admitted = dimension <= self.max_size and (
            materialization is None
            or (
                entries <= materialization.max_entries
                and matrix_bytes <= materialization.max_bytes
            )
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=cost.storage_bytes,
            preparation_workspace_bytes=cost.preparation_workspace_bytes,
            apply_workspace_bytes_per_rhs=cost.apply_workspace_bytes_per_rhs,
            setup_matvec_count=cost.setup_matvec_count,
            accepted=admitted and cost.accepted,
            reason=(
                cost.reason
                if admitted
                else "Coarse LU dimension or dense output exceeds the explicit policy; "
                + cost.reason
            ),
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> ProbedBlockPreconditioner:
        cost = self.cost_for(setup_operator, materialization=materialization)
        if not cost.accepted:
            raise LinearCapabilityError(cost.reason)
        source = setup_operator.source
        selection = BlockSelection(
            source,
            (
                (
                    "coarse",
                    CoordinateBlock(
                        ArraySpace((source.size,), dtype=coordinate_dtype(source)),
                        ((0, source.size),),
                    ),
                ),
            ),
        )
        blocks = _probe_local_blocks(
            setup_operator, selection, source.size, self.probe_chunk_size
        )
        return ProbedBlockPreconditioner(
            selection,
            prepare_local_block_factorization(blocks),
            builder_id=self.builder_id,
            setup_operator_id=setup_operator.operator_id,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> ProbedBlockPreconditioner:
        if not isinstance(preconditioner, ProbedBlockPreconditioner):
            raise TypeError(
                "Coarse LU numeric refresh requires a ProbedBlockPreconditioner."
            )
        if preconditioner.builder_id != self.builder_id:
            raise ValueError("Coarse LU refresh must preserve the builder.")
        if not preconditioner.space.compatible(setup_operator.source):
            raise ValueError("Coarse LU refresh must preserve the source space.")
        return self.prepare(setup_operator, materialization=materialization)


__all__ = [
    "BoundedLUPreconditionerBuilder",
    "MatrixFreeLocalBlockPreconditionerBuilder",
    "ProbedBlockPreconditioner",
    "ProbedBlockSolveResult",
]
