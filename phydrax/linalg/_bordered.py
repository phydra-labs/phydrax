#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import finite_real_scalar
from ._costs import LinearCostEstimate
from ._materialization import MaterializationPolicy
from ._named_blocks import BlockSelection
from ._operators import (
    _array_value,
    _id,
    _validate_action_dtype,
    AbstractLinearOperator,
    AdjointLinearOperator,
    estimate_operator_action_cost,
    TransposeLinearOperator,
)
from ._properties import LinearCapabilityError, OperatorCapabilities, OperatorProperties
from ._spaces import _coordinate_dtype, ArraySpace
from ._structured_operators import BandedLinearOperator


@final
class BorderedAssemblyEvidence(StrictModule):
    """Measured couplings omitted by one declared bordered coordinate route."""

    maximum_absolute_omitted_coupling: Array
    maximum_relative_omitted_coupling: Array
    relative_roundoff_limit: Array
    accepted: Array

    @property
    def exact(self) -> Array:
        return self.maximum_absolute_omitted_coupling == 0


@final
class BorderedBlockLinearOperator(AbstractLinearOperator):
    """Independent local pivots coupled to one dense border in canonical packing.

    Coordinates are all local blocks, in their declared order, followed by the
    border. Off-diagonal local-to-local couplings are not represented. Dense
    local LU requires an explicit bounded policy; declared bandwidth instead
    selects native pivoted banded LU and is checked against the supplied blocks.
    An optional complete coordinate map identifies native physical coordinates
    with ordered members: one local block per layer, then the border.
    """

    _fused_block_action_kind: ClassVar[str] = "fused"

    source: ArraySpace
    target: ArraySpace
    local_blocks: Array | BandedLinearOperator
    upper: Array
    lower: Array
    border: Array
    num_blocks: int = eqx.field(static=True)
    local_size: int = eqx.field(static=True)
    border_size: int = eqx.field(static=True)
    local_bandwidth: tuple[int, int] | None = eqx.field(static=True)
    dense_local_policy: MaterializationPolicy | None
    coordinate_map: BlockSelection | None
    assembly_evidence: BorderedAssemblyEvidence

    def __init__(
        self,
        local_blocks: ArrayLike | BandedLinearOperator,
        upper: ArrayLike,
        lower: ArrayLike,
        border: ArrayLike,
        /,
        *,
        space: ArraySpace | None = None,
        coordinate_map: BlockSelection | None = None,
        dense_local_policy: MaterializationPolicy | None = None,
        operator_id: str | None = None,
        assembly_evidence: BorderedAssemblyEvidence | None = None,
    ) -> None:
        if isinstance(local_blocks, BandedLinearOperator):
            blocks: Array | BandedLinearOperator = local_blocks
            if len(blocks.batch_shape) != 1 or blocks.batch_shape[0] < 1:
                raise ValueError(
                    "Banded locals require exactly one nonempty layer batch axis."
                )
            if not isinstance(blocks.source, ArraySpace) or blocks.source.shape != (
                blocks.source.size,
            ):
                raise ValueError("Banded locals require a flat local ArraySpace.")
            layers = blocks.batch_shape[0]
            local_size = blocks.source.size
            local_bandwidth = (blocks.lower_bandwidth, blocks.upper_bandwidth)
            local_dtype = blocks.bands.dtype
            if local_size < 1 or max(local_bandwidth) >= local_size:
                raise ValueError(
                    "Local bandwidths must be smaller than the positive local size."
                )
        else:
            blocks = jnp.asarray(local_blocks)
            if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
                raise ValueError("local_blocks must have shape (layers, local, local).")
            if any(size < 1 for size in blocks.shape):
                raise ValueError("Local block dimensions must be positive.")
            layers, local_size, _ = blocks.shape
            local_bandwidth = None
            local_dtype = blocks.dtype
        upper_ = jnp.asarray(upper)
        lower_ = jnp.asarray(lower)
        border_ = jnp.asarray(border)
        if border_.ndim != 2 or border_.shape[0] != border_.shape[1]:
            raise ValueError("border must be one square matrix.")
        border_size = border_.shape[0]
        if border_size < 1:
            raise ValueError("The border dimension must be positive.")
        if upper_.shape != (layers, local_size, border_size):
            raise ValueError("upper must have shape (layers, local, border).")
        if lower_.shape != (layers, border_size, local_size):
            raise ValueError("lower must have shape (layers, border, local).")
        dtype = jnp.result_type(local_dtype, upper_.dtype, lower_.dtype, border_.dtype)
        if not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.dtype(jnp.float64)
        if not isinstance(blocks, BandedLinearOperator):
            blocks = blocks.astype(dtype, copy=False)
        upper_ = upper_.astype(dtype, copy=False)
        lower_ = lower_.astype(dtype, copy=False)
        border_ = border_.astype(dtype, copy=False)
        if dense_local_policy is not None and not isinstance(
            dense_local_policy, MaterializationPolicy
        ):
            raise TypeError("dense_local_policy must be a MaterializationPolicy.")
        if isinstance(blocks, BandedLinearOperator):
            if dense_local_policy is not None:
                raise ValueError("Banded local LU does not use a dense local policy.")
        else:
            if dense_local_policy is None:
                raise ValueError("Dense local LU requires dense_local_policy.")
            dense_local_policy.require(blocks.size, dtype)
        size = layers * local_size + border_size
        if coordinate_map is not None:
            _validate_bordered_selection(coordinate_map, layers, local_size, border_size)
        space_ = (
            (
                ArraySpace((size,), dtype=dtype)
                if coordinate_map is None
                else coordinate_map.source
            )
            if space is None
            else space
        )
        if not isinstance(space_, ArraySpace):
            raise TypeError("space must be an ArraySpace.")
        if space_.shape != (size,):
            raise ValueError("space must have the canonical packed flat shape.")
        if coordinate_map is not None and not coordinate_map.source.compatible(space_):
            raise ValueError(
                "coordinate_map must preserve the supplied native space identity."
            )
        _validate_action_dtype(dtype, space_, space_, "bordered coefficients")
        self.source = space_
        self.target = space_
        self.local_blocks = blocks
        self.upper = upper_
        self.lower = lower_
        self.border = border_
        self.num_blocks = layers
        self.local_size = local_size
        self.border_size = border_size
        self.local_bandwidth = local_bandwidth
        self.dense_local_policy = dense_local_policy
        self.coordinate_map = coordinate_map
        if assembly_evidence is None:
            zero = jnp.asarray(0, dtype=dtype)
            assembly_evidence = BorderedAssemblyEvidence(
                zero, zero, zero, jnp.asarray(True, dtype=jnp.bool_)
            )
        elif not isinstance(assembly_evidence, BorderedAssemblyEvidence):
            raise TypeError(
                "assembly_evidence must be a BorderedAssemblyEvidence or None."
            )
        self.assembly_evidence = assembly_evidence
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False, diagonal_assembly=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "bordered-block",
                "space": space_.space_id,
                "layers": layers,
                "local_size": local_size,
                "border_size": border_size,
                "local_bandwidth": local_bandwidth,
                "coordinate_map": None
                if coordinate_map is None
                else coordinate_map.selection_id,
                "dense_local_limits": None
                if dense_local_policy is None
                else (dense_local_policy.max_entries, dense_local_policy.max_bytes),
            },
        )

    def _pack_columns(self, columns: Array, /) -> Array:
        selection = self.coordinate_map
        if selection is None:
            return columns

        def pack(column: Array) -> Array:
            return selection.target.flatten(
                selection.restrict(self.source.unflatten(column))
            )

        return jax.vmap(pack, in_axes=1, out_axes=1)(columns)

    def _unpack_columns(self, columns: Array, /) -> Array:
        selection = self.coordinate_map
        if selection is None:
            return columns

        def unpack(column: Array) -> Array:
            return self.source.flatten(
                selection.prolong(selection.target.unflatten(column))
            )

        return jax.vmap(unpack, in_axes=1, out_axes=1)(columns)

    def _apply(self, columns: Array, /, *, transpose: bool) -> Array:
        columns = self._pack_columns(columns)
        split = self.num_blocks * self.local_size
        local = columns[:split].reshape((self.num_blocks, self.local_size, -1))
        border = columns[split:]
        if isinstance(self.local_blocks, BandedLinearOperator):
            local_image = self.local_blocks._apply_bands(
                local, transpose=transpose, conjugate=False
            )
        else:
            local_matrix = (
                jnp.swapaxes(self.local_blocks, -1, -2)
                if transpose
                else self.local_blocks
            )
            local_image = local_matrix @ local
        if transpose:
            local_image = local_image + jnp.swapaxes(self.lower, -1, -2) @ border
            border_image = ein.contract("lib,lik->bk", self.upper, local)
            border_image = border_image + self.border.T @ border
        else:
            local_image = local_image + self.upper @ border
            border_image = (
                ein.contract("lbi,lik->bk", self.lower, local) + self.border @ border
            )
        return self._unpack_columns(
            jnp.concatenate((local_image.reshape((split, -1)), border_image), axis=0)
        )

    def _array_action(self, vector: ArrayLike, /, *, transpose: bool) -> Array:
        value, rhs_shape = _array_value(self.source, (), vector, "vector")
        columns = value.reshape((self.source.size, -1))
        return self._apply(columns, transpose=transpose).reshape(
            self.source.shape + rhs_shape
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        return self._array_action(vector, transpose=False)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        return self._array_action(vector, transpose=True)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value, rhs_shape = _array_value(self.source, (), vector, "vector")
        columns = value.reshape((self.source.size, -1))
        covectors = jax.vmap(self.target.riesz, in_axes=1, out_axes=1)(columns)
        image = jnp.conj(self._apply(jnp.conj(covectors), transpose=True))
        result = jax.vmap(self.source.inverse_riesz, in_axes=1, out_axes=1)(image)
        return result.reshape(self.source.shape + rhs_shape)

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Bordered operators do not materialize a global matrix."
        )

    def _assemble_diagonal(self, /) -> Array:
        packed = jnp.concatenate(
            (
                (
                    self.local_blocks.bands[:, self.local_blocks.upper_bandwidth, :]
                    if isinstance(self.local_blocks, BandedLinearOperator)
                    else jnp.diagonal(self.local_blocks, axis1=-2, axis2=-1)
                ).reshape((-1,)),
                jnp.diagonal(self.border),
            )
        )
        return self._unpack_columns(packed[:, None])[:, 0]


def _validate_bordered_selection(
    selection: BlockSelection,
    layers: int,
    local_size: int,
    border_size: int,
    /,
) -> ArraySpace:
    if not isinstance(selection, BlockSelection):
        raise TypeError("coordinate_map must be a BlockSelection.")
    if not isinstance(selection.source, ArraySpace):
        raise TypeError("coordinate_map must act on a native flat ArraySpace.")
    if selection.source.shape != (layers * local_size + border_size,):
        raise ValueError("coordinate_map source must have the native packed dimension.")
    if not selection.covers_source:
        raise ValueError("coordinate_map must cover every source coordinate.")
    expected_sizes = (local_size,) * layers + (border_size,)
    if tuple(member.size for member in selection.target.spaces) != expected_sizes:
        raise ValueError(
            "coordinate_map must order local layer members before the border."
        )
    return selection.source


def _bordered_base(
    operator: AbstractLinearOperator, /
) -> BorderedBlockLinearOperator | None:
    if isinstance(operator, BorderedBlockLinearOperator):
        return operator
    if isinstance(
        operator, (TransposeLinearOperator, AdjointLinearOperator)
    ) and isinstance(operator.operator, BorderedBlockLinearOperator):
        return operator.operator
    return None


def _bordered_structure(operator: AbstractLinearOperator, /) -> str | None:
    bordered = _bordered_base(operator)
    if bordered is None:
        return None
    locals_ = bordered.local_blocks
    return canonical_fingerprint(
        {
            "kind": (
                "transpose-bordered"
                if isinstance(operator, TransposeLinearOperator)
                else "adjoint-bordered"
                if isinstance(operator, AdjointLinearOperator)
                else "bordered"
            ),
            "space": bordered.source.space_id,
            "layers": bordered.num_blocks,
            "local_size": bordered.local_size,
            "border_size": bordered.border_size,
            "local_bandwidth": bordered.local_bandwidth,
            "local_dtype": jnp.dtype(
                locals_.bands.dtype
                if isinstance(locals_, BandedLinearOperator)
                else locals_.dtype
            ).str,
            "local_space": locals_.source.space_id
            if isinstance(locals_, BandedLinearOperator)
            else None,
            "coordinate_map": None
            if bordered.coordinate_map is None
            else bordered.coordinate_map.selection_id,
            "dense_local_limits": None
            if bordered.dense_local_policy is None
            else (
                bordered.dense_local_policy.max_entries,
                bordered.dense_local_policy.max_bytes,
            ),
        }
    )


@final
class BorderedBlockAssemblyPlan(StrictModule):
    """Resource-bounded probing with explicit omitted-coupling roundoff evidence."""

    coordinate_map: BlockSelection
    num_blocks: int = eqx.field(static=True)
    local_size: int = eqx.field(static=True)
    border_size: int = eqx.field(static=True)
    local_bandwidth: tuple[int, int] = eqx.field(static=True)
    probe_chunk_size: int = eqx.field(static=True)
    max_local_size: int = eqx.field(static=True)
    max_border_size: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)
    max_output_bytes: int = eqx.field(static=True)
    leakage_roundoff_multiplier: float = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    setup_matvec_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_map: BlockSelection,
        /,
        *,
        local_bandwidth: tuple[int, int],
        verify_local_independence: bool,
        probe_chunk_size: int,
        max_local_size: int,
        max_border_size: int,
        max_workspace_bytes: int,
        max_output_bytes: int,
        leakage_roundoff_multiplier: float = 0.0,
    ) -> None:
        if not isinstance(coordinate_map, BlockSelection):
            raise TypeError("coordinate_map must be a BlockSelection.")
        if type(verify_local_independence) is not bool:
            raise TypeError("verify_local_independence must be a boolean.")
        if not verify_local_independence:
            raise ValueError("Bordered assembly requires verified local independence.")
        limits = (
            probe_chunk_size,
            max_local_size,
            max_border_size,
            max_workspace_bytes,
            max_output_bytes,
        )
        if any(type(value) is not int for value in limits):
            raise TypeError("Bordered assembly resource limits must be exact integers.")
        if any(value < 1 for value in limits):
            raise ValueError("Bordered assembly resource limits must be positive.")
        members = coordinate_map.target.spaces
        if len(members) < 2:
            raise ValueError("Bordered assembly requires local members and one border.")
        layers, local, border = len(members) - 1, members[0].size, members[-1].size
        if local < 1 or border < 1:
            raise ValueError("Bordered assembly dimensions must be positive.")
        source = _validate_bordered_selection(coordinate_map, layers, local, border)
        if local > max_local_size or border > max_border_size:
            raise LinearCapabilityError(
                "Bordered assembly exceeds declared block-size limits."
            )
        if not isinstance(local_bandwidth, tuple) or len(local_bandwidth) != 2:
            raise TypeError("local_bandwidth must be a (lower, upper) tuple.")
        if any(type(value) is not int for value in local_bandwidth):
            raise TypeError("Local bandwidths must be exact integers.")
        lower, upper = local_bandwidth
        if not (0 <= lower < local and 0 <= upper < local):
            raise ValueError("Local bandwidths must lie in [0, local_size).")
        dtype = _coordinate_dtype(source)
        if not jnp.issubdtype(dtype, jnp.inexact):
            raise TypeError(
                "Bordered assembly requires an inexact native coordinate dtype."
            )
        output_bytes = (
            layers * local * (lower + upper + 1)
            + 2 * layers * local * border
            + border * border
        ) * dtype.itemsize
        if output_bytes > max_output_bytes:
            raise LinearCapabilityError(
                f"Bordered assembly requires {output_bytes} output bytes; "
                f"limit is {max_output_bytes}."
            )
        self.coordinate_map = coordinate_map
        self.num_blocks = layers
        self.local_size = local
        self.border_size = border
        self.local_bandwidth = local_bandwidth
        self.probe_chunk_size = probe_chunk_size
        self.max_local_size = max_local_size
        self.max_border_size = max_border_size
        self.max_workspace_bytes = max_workspace_bytes
        self.max_output_bytes = max_output_bytes
        roundoff_multiplier = finite_real_scalar(
            leakage_roundoff_multiplier, "leakage_roundoff_multiplier"
        )
        if roundoff_multiplier < 0:
            raise ValueError("leakage_roundoff_multiplier must be nonnegative.")
        self.leakage_roundoff_multiplier = roundoff_multiplier
        self.output_bytes = output_bytes
        self.setup_matvec_count = source.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bordered-block-assembly",
                "selection": coordinate_map.selection_id,
                "local_bandwidth": local_bandwidth,
                "verify_local_independence": True,
                "resource_limits": limits,
                "leakage_roundoff_multiplier": roundoff_multiplier,
            }
        )

    def cost_for(self, operator: AbstractLinearOperator, /) -> LinearCostEstimate:
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        source = self.coordinate_map.source
        if (
            operator.batch_shape
            or not operator.source.compatible(source)
            or not operator.target.compatible(source)
        ):
            raise ValueError("Bordered assembly requires the mapped native endomorphism.")
        action_cost = estimate_operator_action_cost(operator)
        dimension = source.size
        itemsize = _coordinate_dtype(source).itemsize
        chunk = min(self.probe_chunk_size, dimension)
        workspace = (
            3 * self.output_bytes
            + 4 * dimension * jnp.dtype(jnp.int32).itemsize
            + 2 * dimension * itemsize
            + chunk
            * (
                5 * dimension * itemsize
                + (sum(self.local_bandwidth) + 1) * jnp.dtype(jnp.int32).itemsize
                + action_cost.apply_workspace_bytes_per_rhs
            )
        )
        # Every chunk reuses the same retained operator/tape. The admission
        # limit covers total live assembly storage, not only additional scratch.
        retained_bytes = action_cost.storage_bytes
        peak_bytes = retained_bytes + workspace
        accepted = (
            self.output_bytes <= self.max_output_bytes
            and peak_bytes <= self.max_workspace_bytes
        )
        return LinearCostEstimate(
            provider="native-coordinate-probing",
            method="bordered-block-assembly",
            existing_storage_bytes=retained_bytes,
            additional_matrix_bytes=self.output_bytes,
            preparation_workspace_bytes=workspace,
            operation_class="bounded-coordinate-probing",
            accepted=accepted,
            reason=(
                f"{self.setup_matvec_count} coordinate actions with relative roundoff "
                f"multiplier {self.leakage_roundoff_multiplier}; "
                f"{self.output_bytes} output bytes; {retained_bytes} retained source/tape bytes, "
                f"{workspace} estimated additional scratch bytes, "
                f"{peak_bytes} estimated peak live bytes; "
                f"output/peak limits {self.max_output_bytes}, {self.max_workspace_bytes}; "
                + action_cost.reason
            ),
        )

    def assemble(
        self, operator: AbstractLinearOperator, /
    ) -> BorderedBlockLinearOperator:
        from ._block_probing import _probe_coordinate_action

        cost = self.cost_for(operator)
        if not cost.accepted:
            raise LinearCapabilityError(cost.reason)
        selection = self.coordinate_map
        positions = jnp.concatenate(
            tuple(
                jnp.arange(start, stop, dtype=jnp.int32)
                for ranges in selection.member_ranges
                for start, stop in ranges
            )
        )
        layers, local, border = self.num_blocks, self.local_size, self.border_size
        split = layers * local
        lower, upper = self.local_bandwidth
        offsets = jnp.arange(-upper, lower + 1, dtype=jnp.int32)
        local_rows = jnp.arange(split, dtype=jnp.int32)

        def probe_local(column: Array) -> tuple[Array, Array, Array, Array]:
            image = _probe_coordinate_action(operator, positions[column])[positions]
            block = column // local
            within = column % local
            difference = local_rows % local - within
            allowed = (
                (local_rows // local == block)
                & (difference >= -upper)
                & (difference <= lower)
            )
            omitted = jnp.where(allowed, 0, jnp.abs(image[:split]))
            maximum_omitted = jnp.max(omitted)
            column_scale = jnp.max(jnp.abs(image))
            relative_omitted = jnp.where(
                column_scale > 0,
                maximum_omitted / column_scale,
                jnp.where(maximum_omitted == 0, 0, jnp.inf),
            )
            rows = within + offsets
            valid = (rows >= 0) & (rows < local)
            band_values = image[block * local + jnp.clip(rows, 0, local - 1)]
            # Only nonexistent edge padding is zeroed. Omitted coordinates retain
            # measured evidence and must satisfy the explicit roundoff policy.
            band_values = jnp.where(valid, band_values, 0)
            return band_values, image[split:], maximum_omitted, relative_omitted

        chunk = min(self.probe_chunk_size, split)
        dtype = _coordinate_dtype(selection.source)
        initial = (
            jnp.zeros((split, lower + upper + 1), dtype=dtype),
            jnp.zeros((split, border), dtype=dtype),
            jnp.asarray(0, dtype=dtype),
            jnp.asarray(0, dtype=dtype),
        )

        def add_chunk(
            start: Array,
            count: int,
            state: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            band_values, border_values, omitted, relative = jax.lax.map(
                probe_local,
                start + jnp.arange(count, dtype=jnp.int32),
                batch_size=count,
            )
            current_bands, current_lower, maximum_omitted, maximum_relative = state
            return (
                jax.lax.dynamic_update_slice(
                    current_bands, band_values, (start, jnp.asarray(0, dtype=jnp.int32))
                ),
                jax.lax.dynamic_update_slice(
                    current_lower, border_values, (start, jnp.asarray(0, dtype=jnp.int32))
                ),
                jnp.maximum(maximum_omitted, jnp.max(omitted)),
                jnp.maximum(maximum_relative, jnp.max(relative)),
            )

        def advance(
            index: Array, state: tuple[Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array]:
            return add_chunk((index * chunk).astype(jnp.int32), chunk, state)

        accumulated = jax.lax.fori_loop(0, split // chunk, advance, initial)
        remainder = split % chunk
        if remainder:
            accumulated = add_chunk(
                jnp.asarray(split - remainder, dtype=jnp.int32),
                remainder,
                accumulated,
            )
        band_columns, lower_columns, maximum_omitted, maximum_relative = accumulated

        def probe_border(column: Array) -> Array:
            return _probe_coordinate_action(operator, positions[split + column])[
                positions
            ]

        border_columns = jax.lax.map(
            probe_border,
            jnp.arange(border, dtype=jnp.int32),
            batch_size=min(self.probe_chunk_size, border),
        ).T
        relative_roundoff_limit = jnp.asarray(
            self.leakage_roundoff_multiplier
            * selection.source.size
            * jnp.finfo(dtype).eps,
            dtype=dtype,
        )
        accepted = maximum_relative <= relative_roundoff_limit
        bands = band_columns.reshape((layers, local, lower + upper + 1)).swapaxes(-1, -2)
        bands = eqx.error_if(
            bands,
            ~accepted,
            "Operator has resolved off-local or outside-band couplings; the "
            "declared bordered structure is not admissible under its roundoff policy.",
        )
        local_operator = BandedLinearOperator(
            bands, lower_bandwidth=lower, upper_bandwidth=upper
        )
        return BorderedBlockLinearOperator(
            local_operator,
            border_columns[:split].reshape((layers, local, border)),
            lower_columns.reshape((layers, local, border)).swapaxes(-1, -2),
            border_columns[split:],
            assembly_evidence=BorderedAssemblyEvidence(
                maximum_omitted,
                maximum_relative,
                relative_roundoff_limit,
                accepted,
            ),
            coordinate_map=selection,
            operator_id=canonical_fingerprint(
                {
                    "kind": "assembled-bordered-block",
                    "operator": operator.operator_id,
                    "assembly": self.plan_id,
                }
            ),
        )


__all__ = [
    "BorderedAssemblyEvidence",
    "BorderedBlockAssemblyPlan",
    "BorderedBlockLinearOperator",
]
