#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._differentiation import BranchDifferentiationPolicy
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics._compensated import compensated_sum, compensated_sum_chunks
from .._strict import StrictModule
from .._trainable import fixed_field
from .._validation import is_canonical_identifier
from ..ein import contract
from ..typing import checked


_STAGE_RATE_UNITS = "content/time"
_ACCEPTED_INTEGRAL_UNITS = "content"
_SSPRK33_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)


def _flux_identity(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not is_canonical_identifier(value):
        raise ValueError(
            f"{name} must be a nonempty identifier without surrounding whitespace; got an invalid string"
        )
    return value


def _route_array(value: ArrayLike, name: str, /) -> tuple[Array, np.ndarray]:
    host = np.asarray(value)
    if host.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got another rank")
    if not np.issubdtype(host.dtype, np.signedinteger):
        raise TypeError(f"{name} must have a signed integer dtype; got another dtype")
    limits = np.iinfo(np.int32)
    if np.any(host < limits.min) or np.any(host > limits.max):
        raise ValueError(
            f"{name} must contain indices in the supported range; got an out-of-range index"
        )
    normalized = np.asarray(host, dtype=np.int32)
    return jnp.asarray(normalized), normalized


def _active_array(value: ArrayLike, face_count: int, /) -> tuple[Array, np.ndarray]:
    host = np.asarray(value)
    if host.dtype != np.dtype(np.bool_):
        raise TypeError(
            "conservation_ledger.active_mask must have boolean dtype; got another dtype"
        )
    if host.shape != (face_count,):
        raise ValueError(
            "conservation_ledger.active_mask must contain one value per routed face; got another mask shape"
        )
    normalized = np.asarray(host, dtype=np.bool_)
    return jnp.asarray(normalized), normalized


def _active_cell_array(value: ArrayLike, cell_count: int, /) -> Array:
    array = jnp.asarray(value)
    if array.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(
            "conservation_ledger.active_cell_mask must have boolean dtype; got another dtype"
        )
    if array.shape != (cell_count,):
        raise ValueError(
            "conservation_ledger.active_cell_mask must contain one value per cell; got another mask shape"
        )
    return array


def _integer_scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar; got another rank")
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype; got another dtype")
    return array


def _validate_inactive_values(
    values: Array,
    active_cell_mask: Array,
    name: str,
    /,
) -> Array:
    mask = active_cell_mask.reshape(active_cell_mask.shape + (1,) * (values.ndim - 1))
    return eqx.error_if(
        values,
        jnp.any((~mask) & (values != 0)),
        f"{name} must be zero on inactive cells; got nonzero inactive content",
    )


def _validate_route_values(
    owner_cells: np.ndarray,
    neighbor_cells: np.ndarray,
    active_mask: np.ndarray,
    /,
    allow_self: bool = False,
) -> None:
    if np.any(owner_cells < 0):
        raise ValueError(
            "conservation_ledger.owner_cells must contain nonnegative cell indices; got a negative index"
        )
    if np.any(neighbor_cells < -1):
        raise ValueError(
            "conservation_ledger.neighbor_cells must use only -1 as a boundary sentinel; got another negative index"
        )
    if not allow_self and np.any(active_mask & (neighbor_cells == owner_cells)):
        raise ValueError(
            "conservation_ledger.neighbor_cells must differ from owner_cells on active untransformed routes; got a self-route"
        )


def _finite_values(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.integer)
        or jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise TypeError(f"{name} must have a real numeric dtype; got another dtype")
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)),
        f"{name} must contain only finite values; got a nonfinite entry",
    )


def _finite_scalar(value: ArrayLike, name: str, /) -> Array:
    array = _finite_values(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar; got another rank")
    return array


def _time_values_close(first: Array, second: Array, /) -> Array:
    dtype = jnp.result_type(first, second, jnp.asarray(1.0))
    first_ = first.astype(dtype)
    second_ = second.astype(dtype)
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
        jnp.maximum(jnp.abs(first_), jnp.abs(second_)),
    )
    tolerance = jnp.asarray(16.0, dtype=dtype) * jnp.finfo(dtype).eps * scale
    return jnp.abs(first_ - second_) <= tolerance


def _masked_face_values(values: Array, active_mask: Array, /) -> Array:
    mask = active_mask.reshape(active_mask.shape + (1,) * (values.ndim - 1))
    return jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))


def _route_fingerprint(
    owner_cells: np.ndarray,
    neighbor_cells: np.ndarray,
    active_mask: np.ndarray,
    component_shape: tuple[int, ...],
    block_kind: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "conservation-owner-neighbor-flux-route",
            "owner_cells": array_tree_fingerprint(owner_cells),
            "neighbor_cells": array_tree_fingerprint(neighbor_cells),
            "active_mask": array_tree_fingerprint(active_mask),
            "component_shape": list(component_shape),
            "block_kind": block_kind,
        }
    )


def _capacity_route_arrays(
    owner_cells: ArrayLike,
    neighbor_cells: ArrayLike,
    active_mask: ArrayLike,
    /,
    allow_self: bool = False,
) -> tuple[Array, Array, Array]:
    owner = jnp.asarray(owner_cells)
    neighbor = jnp.asarray(neighbor_cells)
    active = jnp.asarray(active_mask)
    if owner.dtype != jnp.int32 or neighbor.dtype != jnp.int32:
        raise TypeError(
            "conservation_ledger.owner_cells must have int32 dtype with neighbor_cells; got another slot dtype"
        )
    if owner.ndim != 1 or neighbor.shape != owner.shape:
        raise ValueError(
            "conservation_ledger.neighbor_cells must match the one-dimensional owner_cells shape; got incompatible route shapes"
        )
    if active.dtype != jnp.bool_:
        raise TypeError(
            "conservation_ledger.active_mask must have boolean dtype; got another dtype"
        )
    if active.shape != owner.shape:
        raise ValueError(
            "conservation_ledger.active_mask must contain one value per routed face; got another mask shape"
        )
    owner = eqx.error_if(
        owner,
        jnp.any(owner < 0),
        "conservation_ledger.owner_cells must contain nonnegative cell indices; got a negative index",
    )
    neighbor = eqx.error_if(
        neighbor,
        jnp.any(neighbor < -1),
        "conservation_ledger.neighbor_cells must use only -1 as a boundary sentinel; got another negative index",
    )
    active = eqx.error_if(
        active,
        jnp.any(active & (neighbor == owner)) & (not allow_self),
        "conservation_ledger.neighbor_cells must differ from owner_cells on active untransformed routes; got a self-route",
    )
    return owner, neighbor, active


def _capacity_route_fingerprint(
    route_signature_id: str,
    component_shape: tuple[int, ...],
    block_kind: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "conservation-capacity-flux-route",
            "route_signature_id": route_signature_id,
            "component_shape": list(component_shape),
            "block_kind": block_kind,
        }
    )


def _validate_route_bounds(
    validation_token: Array,
    owner_cells: Array,
    neighbor_cells: Array,
    active_mask: Array,
    active_cell_mask: Array,
    cell_count: int,
    block_id: str,
    /,
    allow_self: bool = False,
) -> Array:
    validation_token = eqx.error_if(
        validation_token,
        jnp.any((owner_cells < 0) | (owner_cells >= cell_count)),
        f"conservation_ledger.owner_cells must index cells in the ledger range; got out-of-range block {block_id!r}",
    )
    validation_token = eqx.error_if(
        validation_token,
        jnp.any((neighbor_cells < -1) | (neighbor_cells >= cell_count)),
        f"conservation_ledger.neighbor_cells must index cells in the ledger range or use -1; got out-of-range block {block_id!r}",
    )
    validation_token = eqx.error_if(
        validation_token,
        jnp.any(active_mask & (neighbor_cells == owner_cells)) & (not allow_self),
        f"conservation_ledger.neighbor_cells must differ from owners on active untransformed routes; got self-route block {block_id!r}",
    )
    safe_neighbor = jnp.maximum(neighbor_cells, 0)
    inactive_endpoint = (~active_cell_mask[owner_cells]) | (
        (neighbor_cells >= 0) & (~active_cell_mask[safe_neighbor])
    )
    return eqx.error_if(
        validation_token,
        jnp.any(active_mask & inactive_endpoint),
        f"conservation_ledger.active_cell_mask must enable every active routed cell; got an inactive cell in block {block_id!r}",
    )


def _validate_blocks[
    Block: ConservationStageFluxRateBlock | AcceptedConservationFluxIntegralBlock
](
    blocks: Sequence[Block],
    block_type: type[Block],
    cell_count: int,
    component_shape: tuple[int, ...],
    active_cell_mask: Array,
    validation_token: Array,
    /,
) -> tuple[tuple[Block, ...], Array]:
    normalized = tuple(blocks)
    if any(not isinstance(block, block_type) for block in normalized):
        raise TypeError(
            f"conservation_ledger.blocks must contain only {block_type.__name__} values; got another block kind"
        )
    block_ids = tuple(block.block_id for block in normalized)
    route_ids = tuple(block.route_id for block in normalized)
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(
            f"conservation_ledger.block_ids must be unique within a ledger; got {block_ids!r}"
        )
    if len(set(route_ids)) != len(route_ids):
        raise ValueError(
            f"conservation_ledger.route_ids must be unique within a ledger; got {route_ids!r}"
        )
    for block in normalized:
        if block.component_shape != component_shape:
            raise ValueError(
                f"conservation_ledger.component_shape must agree across all blocks; got incompatible block {block.block_id!r}"
            )
        validation_token = _validate_route_bounds(
            validation_token,
            block.owner_cells,
            block.neighbor_cells,
            block.active_mask,
            active_cell_mask,
            cell_count,
            block.block_id,
            block.neighbor_state_action is not None,
        )
    return normalized, validation_token


def _state_action(values: Array, action: Array | None) -> Array:
    if action is None:
        return values
    return contract("fij,fj->fi", action, values.reshape((values.shape[0], -1))).reshape(
        values.shape
    )


def _prepare_state_action(values: Array, action: ArrayLike | None) -> Array | None:
    if action is None:
        return None
    matrix = _finite_values(
        jnp.asarray(action, dtype=jnp.result_type(values, 1.0)), "neighbor_state_action"
    )
    count = math.prod(values.shape[1:])
    if matrix.shape != (values.shape[0], count, count):
        raise ValueError(
            "conservation_ledger.neighbor_state_action must match the routed component space; got another matrix shape"
        )
    tolerance = 256 * jnp.finfo(matrix.dtype).eps
    return eqx.error_if(
        matrix,
        jnp.any(
            jnp.abs(jnp.swapaxes(matrix, -1, -2) @ matrix - jnp.eye(count)) > tolerance
        ),
        "conservation_ledger.neighbor_state_action must be orthogonal within dtype-scaled tolerance; got a nonorthogonal action",
    )


def _frame_exchange(
    blocks: (
        tuple[ConservationStageFluxRateBlock, ...]
        | tuple[AcceptedConservationFluxIntegralBlock, ...]
    ),
    values: tuple[Array, ...],
    component_shape: tuple[int, ...],
    dtype: DTypeLike,
) -> Array:
    chunks = tuple(
        jnp.where(
            (block.neighbor_cells >= 0).reshape(
                (block.neighbor_cells.size,) + (1,) * len(component_shape)
            ),
            _state_action(value, block.neighbor_state_action) - value,
            0,
        )
        for block, value in zip(blocks, values, strict=True)
    )
    return (
        compensated_sum_chunks(chunks, output_ndim=len(component_shape))
        if chunks
        else jnp.zeros(component_shape, dtype=dtype)
    )


def _scatter_block(
    scattered: Array,
    values: Array,
    owner_cells: Array,
    neighbor_cells: Array,
    neighbor_state_action: Array | None,
    /,
) -> Array:
    safe_neighbor = jnp.maximum(neighbor_cells, 0)
    scattered = scattered.at[owner_cells].add(-values)
    neighbor_mask = (neighbor_cells >= 0).reshape(
        neighbor_cells.shape + (1,) * (values.ndim - 1)
    )
    return scattered.at[safe_neighbor].add(
        jnp.where(
            neighbor_mask,
            _state_action(values, neighbor_state_action),
            jnp.zeros((), dtype=values.dtype),
        )
    )


def _conservation_sums(
    source: Array,
    routed_values: tuple[tuple[Array, Array], ...],
    scattered: Array,
    component_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array, Array]:
    """Compensated source, boundary-outward, and net-cell sums of one ledger.

    ``routed_values`` pairs each block's owner-outward values with its neighbor
    cells; boundary routes are the ``-1`` neighbors. The physical-frame balance
    is ``net_cell == source - boundary_outward + frame_exchange_sum``.
    """
    source_sum = compensated_sum(source, axis=0)
    boundary_chunks = tuple(
        jnp.where(
            (neighbor_cells < 0).reshape(neighbor_cells.shape + (1,) * (values.ndim - 1)),
            values,
            jnp.zeros((), dtype=values.dtype),
        )
        for values, neighbor_cells in routed_values
    )
    boundary_sum = (
        compensated_sum_chunks(boundary_chunks, output_ndim=len(component_shape))
        if boundary_chunks
        else jnp.zeros(component_shape, dtype=source.dtype)
    )
    net_cell_sum = compensated_sum(scattered, axis=0)
    return source_sum, boundary_sum, net_cell_sum


class ConservationStageFluxRateBlock(StrictModule):
    """One immutable owner-oriented block of area-integrated stage flux rates."""

    flux_rate: Array
    owner_cells: Array
    neighbor_cells: Array
    active_mask: Array
    neighbor_state_action: Array | None = fixed_field()
    block_id: str = eqx.field(static=True)
    block_kind: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    units: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    rate_block_id: str = eqx.field(static=True)

    def __init__(
        self,
        flux_rate: ArrayLike,
        owner_cells: ArrayLike,
        neighbor_cells: ArrayLike,
        active_mask: ArrayLike,
        block_id: str,
        block_kind: str,
        /,
        *,
        route_signature_id: str | None = None,
        neighbor_state_action: ArrayLike | None = None,
        state_action_id: str = "identity",
    ) -> None:
        """Validate one owner-oriented route block.

        Without ``route_signature_id`` the routes are host topology: they are
        validated with NumPy and fingerprinted by value. With it, the routes are
        device-resident fixed-capacity data (for example masked simplex
        half-facets) that may be traced: they are validated in-graph with
        ``eqx.error_if`` and the route identity is the capacity signature, never
        the route values, so one compiled consumer serves every topology of the
        capacity bucket.
        """
        block_id_ = _flux_identity(block_id, "block_id")
        block_kind_ = _flux_identity(block_kind, "block_kind")
        if route_signature_id is None:
            owner, owner_host = _route_array(owner_cells, "owner_cells")
            neighbor, neighbor_host = _route_array(neighbor_cells, "neighbor_cells")
            if neighbor.shape != owner.shape:
                raise ValueError(
                    "conservation_ledger.neighbor_cells must match owner_cells shape; got incompatible route shapes"
                )
            active, active_host = _active_array(active_mask, owner.shape[0])
            _validate_route_values(
                owner_host, neighbor_host, active_host, neighbor_state_action is not None
            )
        else:
            signature = _flux_identity(route_signature_id, "route_signature_id")
            owner, neighbor, active = _capacity_route_arrays(
                owner_cells,
                neighbor_cells,
                active_mask,
                neighbor_state_action is not None,
            )
        rate = _finite_values(flux_rate, "flux_rate")
        if rate.ndim == 0 or rate.shape[0] != owner.shape[0]:
            raise ValueError(
                "ConservationStageFluxRateBlock.flux_rate must begin with the routed face count; got another leading shape"
            )
        component_shape = tuple(rate.shape[1:])
        rate = _masked_face_values(rate, active)
        route_id = (
            _route_fingerprint(
                owner_host,
                neighbor_host,
                active_host,
                component_shape,
                block_kind_,
            )
            if route_signature_id is None
            else _capacity_route_fingerprint(signature, component_shape, block_kind_)
        )
        action = _prepare_state_action(rate, neighbor_state_action)
        if action is not None:
            if state_action_id == "identity":
                raise ValueError(
                    "conservation_ledger.state_action_id must identify an explicit neighbor state action; got identity"
                )
            route_id = canonical_fingerprint(
                {
                    "route": route_id,
                    "state_action": _flux_identity(state_action_id, "state_action_id"),
                }
            )
        self.flux_rate = rate
        self.owner_cells = owner
        self.neighbor_cells = neighbor
        self.active_mask = active
        self.neighbor_state_action = action
        self.block_id = block_id_
        self.block_kind = block_kind_
        self.component_shape = component_shape
        self.units = _STAGE_RATE_UNITS
        self.route_id = route_id
        self.rate_block_id = canonical_fingerprint(
            {
                "kind": "conservation-stage-flux-rate-block",
                "block_id": block_id_,
                "block_kind": block_kind_,
                "route": route_id,
                "units": _STAGE_RATE_UNITS,
            }
        )

    def with_flux_rate(self, flux_rate: ArrayLike, /) -> ConservationStageFluxRateBlock:
        """Replace rates while reusing this block's validated route identity."""
        rate = _finite_values(flux_rate, "flux_rate")
        if rate.shape != self.flux_rate.shape:
            raise ValueError(
                "ConservationStageFluxRateBlock.flux_rate must match the route-template shape; got another rate shape"
            )
        rate = _masked_face_values(rate, self.active_mask)
        return eqx.tree_at(lambda block: block.flux_rate, self, rate)


class ConservationStageLedger(StrictModule):
    """Immutable content-rate ledger for one SSPRK flux-evaluation stage."""

    blocks: tuple[ConservationStageFluxRateBlock, ...]
    source_rate: Array
    high_order_blocks: tuple[ConservationStageFluxRateBlock, ...]
    low_order_blocks: tuple[ConservationStageFluxRateBlock, ...]
    blend_factors: tuple[Array, ...]
    entropy_production: Array
    troubled_cell_mask: Array
    correction_level: Array
    accepted: Array
    active_cell_mask: Array
    geometry_version: Array
    evidence_version: Array
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    evidence_policy_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    differentiability_policy_id: str = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    units: str = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[ConservationStageFluxRateBlock],
        source_rate: ArrayLike,
        active_cell_mask: ArrayLike,
        /,
        *,
        geometry_family_id: str,
        geometry_layout_id: str,
        geometry_version: ArrayLike,
        evidence_policy_id: str,
        evidence_version: ArrayLike,
        topology_epoch_id: str,
        high_order_blocks: Sequence[ConservationStageFluxRateBlock] | None = None,
        low_order_blocks: Sequence[ConservationStageFluxRateBlock] | None = None,
        blend_factors: Sequence[ArrayLike] | None = None,
        entropy_production: ArrayLike | None = None,
        troubled_cell_mask: ArrayLike | None = None,
        correction_level: ArrayLike | None = None,
        accepted: ArrayLike = True,
        differentiability_policy_id: str = BranchDifferentiationPolicy.SMOOTH.value,
    ) -> None:
        geometry_family = _flux_identity(geometry_family_id, "geometry_family_id")
        geometry_layout = _flux_identity(geometry_layout_id, "geometry_layout_id")
        version = _integer_scalar(geometry_version, "geometry_version")
        evidence_policy = _flux_identity(evidence_policy_id, "evidence_policy_id")
        evidence = _integer_scalar(evidence_version, "evidence_version")
        topology_id = _flux_identity(topology_epoch_id, "topology_epoch_id")
        source = _finite_values(source_rate, "source_rate")
        if source.ndim == 0 or source.shape[0] == 0:
            raise ValueError(
                "ConservationStageLedger.source_rate must begin with a nonempty cell axis; got an empty or scalar source"
            )
        cell_count = source.shape[0]
        component_shape = tuple(source.shape[1:])
        active_cells = _active_cell_array(active_cell_mask, cell_count)
        source = _validate_inactive_values(source, active_cells, "source_rate")
        blocks_, source = _validate_blocks(
            blocks,
            ConservationStageFluxRateBlock,
            cell_count,
            component_shape,
            active_cells,
            source,
        )
        high = blocks_ if high_order_blocks is None else tuple(high_order_blocks)
        low = blocks_ if low_order_blocks is None else tuple(low_order_blocks)
        if (
            any(not isinstance(block, ConservationStageFluxRateBlock) for block in high)
            or any(not isinstance(block, ConservationStageFluxRateBlock) for block in low)
            or tuple(block.route_id for block in high)
            != tuple(block.route_id for block in blocks_)
            or tuple(block.route_id for block in low)
            != tuple(block.route_id for block in blocks_)
        ):
            raise ValueError(
                "ConservationStageLedger.blocks must match high_order_blocks and low_order_blocks route identities; got incompatible rate blocks"
            )
        raw_factors = (
            tuple(
                jnp.ones(block.owner_cells.shape, dtype=source.dtype) for block in blocks_
            )
            if blend_factors is None
            else tuple(jnp.asarray(value, dtype=source.dtype) for value in blend_factors)
        )
        if len(raw_factors) != len(blocks_) or any(
            factor.shape != block.owner_cells.shape
            for factor, block in zip(raw_factors, blocks_, strict=True)
        ):
            raise ValueError(
                "ConservationStageLedger.blend_factors must contain one scalar per routed face; got another factor shape"
            )
        factors = tuple(
            eqx.error_if(
                factor,
                jnp.any((factor < 0.0) | (factor > 1.0) | ~jnp.isfinite(factor)),
                "ConservationStageLedger.blend_factors must be finite and lie in [0, 1]; got an invalid factor",
            )
            for factor in raw_factors
        )
        entropy = (
            jnp.zeros((cell_count,), dtype=source.dtype)
            if entropy_production is None
            else jnp.asarray(entropy_production, dtype=source.dtype)
        )
        troubled = (
            jnp.zeros((cell_count,), dtype=jnp.bool_)
            if troubled_cell_mask is None
            else jnp.asarray(troubled_cell_mask, dtype=jnp.bool_)
        )
        levels = (
            jnp.zeros((cell_count,), dtype=jnp.int32)
            if correction_level is None
            else jnp.asarray(correction_level, dtype=jnp.int32)
        )
        accepted_ = jnp.asarray(accepted, dtype=jnp.bool_)
        if (
            entropy.shape != (cell_count,)
            or troubled.shape != (cell_count,)
            or levels.shape != (cell_count,)
            or accepted_.shape != ()
        ):
            raise ValueError(
                "ConservationStageLedger.robustness_evidence must align with routed cells and faces; got an incompatible evidence shape"
            )
        entropy = eqx.error_if(
            entropy,
            jnp.any(~jnp.isfinite(entropy)),
            "ConservationStageLedger.entropy_production must be finite; got a nonfinite entry",
        )
        levels = eqx.error_if(
            levels,
            jnp.any(levels < 0),
            "ConservationStageLedger.correction_level must be nonnegative; got a negative level",
        )
        differentiability = _flux_identity(
            differentiability_policy_id, "differentiability_policy_id"
        )
        self.blocks = blocks_
        self.source_rate = source
        self.high_order_blocks = high
        self.low_order_blocks = low
        self.blend_factors = factors
        self.entropy_production = entropy
        self.troubled_cell_mask = troubled
        self.correction_level = levels
        self.accepted = accepted_
        self.active_cell_mask = active_cells
        self.geometry_version = version
        self.evidence_version = evidence
        self.geometry_family_id = geometry_family
        self.geometry_layout_id = geometry_layout
        self.evidence_policy_id = evidence_policy
        self.topology_epoch_id = topology_id
        self.differentiability_policy_id = differentiability
        self.cell_count = cell_count
        self.component_shape = component_shape
        self.units = _STAGE_RATE_UNITS
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "conservation-stage-ledger",
                "geometry_family_id": geometry_family,
                "geometry_layout_id": geometry_layout,
                "topology_epoch_id": topology_id,
                "evidence_policy_id": evidence_policy,
                "block_rate_ids": [block.rate_block_id for block in blocks_],
                "high_order_rate_ids": [block.rate_block_id for block in high],
                "low_order_rate_ids": [block.rate_block_id for block in low],
                "blend_factor_shapes": [list(value.shape) for value in factors],
                "differentiability_policy_id": differentiability,
                "cell_count": cell_count,
                "component_shape": list(component_shape),
                "active_cell_policy": "exact-boolean-source-zero-active-routes",
                "block_units": _STAGE_RATE_UNITS,
                "source_units": _STAGE_RATE_UNITS,
                "units": _STAGE_RATE_UNITS,
            }
        )

    def with_selected_rates(
        self,
        blocks: Sequence[ConservationStageFluxRateBlock],
        source_rate: ArrayLike,
        /,
        *,
        blend_factors: Sequence[ArrayLike],
        troubled_cell_mask: ArrayLike,
        correction_level: ArrayLike,
        accepted: ArrayLike,
        differentiability_policy_id: str,
    ) -> "ConservationStageLedger":
        selected = tuple(blocks)
        if len(selected) != len(self.blocks) or tuple(
            block.route_id for block in selected
        ) != tuple(block.route_id for block in self.blocks):
            raise ValueError(
                "ConservationStageLedger.blocks must reuse retained route identities for rate selection; got changed routes"
            )
        if jnp.shape(source_rate) != self.source_rate.shape:
            raise ValueError(
                "ConservationStageLedger.source_rate must retain the ledger source shape; got another selected source shape"
            )
        factors = tuple(blend_factors)
        if len(factors) != len(selected):
            raise ValueError(
                "ConservationStageLedger.blend_factors must provide one bank per route block; got another bank count"
            )
        # Identity and differentiability ids are static fields, so the selected ledger
        # is rebuilt through the validating constructor rather than `eqx.tree_at`.
        return type(self)(
            selected,
            source_rate,
            self.active_cell_mask,
            geometry_family_id=self.geometry_family_id,
            geometry_layout_id=self.geometry_layout_id,
            geometry_version=self.geometry_version,
            evidence_policy_id=self.evidence_policy_id,
            evidence_version=self.evidence_version,
            topology_epoch_id=self.topology_epoch_id,
            high_order_blocks=self.high_order_blocks,
            low_order_blocks=self.low_order_blocks,
            blend_factors=factors,
            entropy_production=self.entropy_production,
            troubled_cell_mask=troubled_cell_mask,
            correction_level=correction_level,
            accepted=accepted,
            differentiability_policy_id=differentiability_policy_id,
        )

    def scatter_content_rate(self) -> Array:
        """Scatter owner-outward rates and sources without multiplying by a step."""
        scattered = self.source_rate
        for block in self.blocks:
            scattered = _scatter_block(
                scattered,
                block.flux_rate,
                block.owner_cells,
                block.neighbor_cells,
                block.neighbor_state_action,
            )
        return scattered

    def frame_exchange_sum(self) -> Array:
        """Internal reciprocal rates expressed in different physical cell frames."""
        return _frame_exchange(
            self.blocks,
            tuple(block.flux_rate for block in self.blocks),
            self.component_shape,
            self.source_rate.dtype,
        )

    def conservation_sums(self) -> tuple[Array, Array, Array]:
        """Return source, boundary-outward, and net-cell content-rate sums."""
        return _conservation_sums(
            self.source_rate,
            tuple((block.flux_rate, block.neighbor_cells) for block in self.blocks),
            self.scatter_content_rate(),
            self.component_shape,
        )


class AcceptedConservationFluxIntegralBlock(StrictModule):
    """One immutable owner-oriented block of accepted content integrals."""

    flux_integral: Array
    owner_cells: Array
    neighbor_cells: Array
    active_mask: Array
    neighbor_state_action: Array | None = fixed_field()
    block_id: str = eqx.field(static=True)
    block_kind: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    units: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    integral_block_id: str = eqx.field(static=True)

    def __init__(
        self,
        flux_integral: ArrayLike,
        owner_cells: ArrayLike,
        neighbor_cells: ArrayLike,
        active_mask: ArrayLike,
        block_id: str,
        block_kind: str,
        /,
        *,
        _validated_route_id: str | None = None,
        neighbor_state_action: ArrayLike | None = None,
        state_action_id: str = "identity",
    ) -> None:
        block_id_ = _flux_identity(block_id, "block_id")
        block_kind_ = _flux_identity(block_kind, "block_kind")
        owner = jnp.asarray(owner_cells)
        neighbor = jnp.asarray(neighbor_cells)
        active = jnp.asarray(active_mask)
        if _validated_route_id is None:
            owner, owner_host = _route_array(owner_cells, "owner_cells")
            neighbor, neighbor_host = _route_array(
                neighbor_cells,
                "neighbor_cells",
            )
            if neighbor.shape != owner.shape:
                raise ValueError(
                    "conservation_ledger.neighbor_cells must match owner_cells shape; got incompatible route shapes"
                )
            active, active_host = _active_array(active_mask, owner.shape[0])
            _validate_route_values(
                owner_host, neighbor_host, active_host, neighbor_state_action is not None
            )
        elif (
            owner.ndim != 1
            or neighbor.shape != owner.shape
            or active.dtype != jnp.bool_
            or active.shape != owner.shape
        ):
            raise ValueError(
                "AcceptedConservationFluxIntegralBlock.owner_cells must retain the validated route shape with neighbor_cells and active_mask; got changed route shapes"
            )
        integral = _finite_values(flux_integral, "flux_integral")
        if integral.ndim == 0 or integral.shape[0] != owner.shape[0]:
            raise ValueError(
                "AcceptedConservationFluxIntegralBlock.flux_integral must begin with the routed face count; got another leading shape"
            )
        component_shape = tuple(integral.shape[1:])
        integral = _masked_face_values(integral, active)
        route_id = (
            _route_fingerprint(
                owner_host,
                neighbor_host,
                active_host,
                component_shape,
                block_kind_,
            )
            if _validated_route_id is None
            else _validated_route_id
        )
        action = _prepare_state_action(integral, neighbor_state_action)
        if _validated_route_id is None and action is not None:
            if state_action_id == "identity":
                raise ValueError(
                    "conservation_ledger.state_action_id must identify an explicit neighbor state action; got identity"
                )
            route_id = canonical_fingerprint(
                {
                    "route": route_id,
                    "state_action": _flux_identity(state_action_id, "state_action_id"),
                }
            )
        self.flux_integral = integral
        self.owner_cells = owner
        self.neighbor_cells = neighbor
        self.active_mask = active
        self.neighbor_state_action = action
        self.block_id = block_id_
        self.block_kind = block_kind_
        self.component_shape = component_shape
        self.units = _ACCEPTED_INTEGRAL_UNITS
        self.route_id = route_id
        self.integral_block_id = canonical_fingerprint(
            {
                "kind": "accepted-conservation-flux-integral-block",
                "block_id": block_id_,
                "block_kind": block_kind_,
                "route": route_id,
                "units": _ACCEPTED_INTEGRAL_UNITS,
            }
        )

    @classmethod
    @checked
    def _from_stage_rate_block(
        cls,
        flux_integral: ArrayLike,
        stage_block: ConservationStageFluxRateBlock,
        /,
    ) -> AcceptedConservationFluxIntegralBlock:
        return cls(
            flux_integral,
            stage_block.owner_cells,
            stage_block.neighbor_cells,
            stage_block.active_mask,
            stage_block.block_id,
            stage_block.block_kind,
            _validated_route_id=stage_block.route_id,
            neighbor_state_action=stage_block.neighbor_state_action,
        )


class AcceptedConservationIntegralLedger(StrictModule):
    """Immutable accepted SSPRK content integrals and their conservation sums."""

    blocks: tuple[AcceptedConservationFluxIntegralBlock, ...]
    source_integral: Array
    active_cell_mask: Array
    start_time: Array
    end_time: Array
    accepted_step: Array
    stage_geometry_versions: tuple[Array, Array, Array]
    start_geometry_version: Array
    end_geometry_version: Array
    stage_evidence_versions: tuple[Array, Array, Array]
    start_evidence_version: Array
    end_evidence_version: Array
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    evidence_policy_id: str = eqx.field(static=True)
    start_topology_epoch_id: str = eqx.field(static=True)
    end_topology_epoch_id: str = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    units: str = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[AcceptedConservationFluxIntegralBlock],
        source_integral: ArrayLike,
        active_cell_mask: ArrayLike,
        /,
        *,
        geometry_family_id: str,
        geometry_layout_id: str,
        stage_geometry_versions: tuple[ArrayLike, ArrayLike, ArrayLike],
        start_geometry_version: ArrayLike,
        end_geometry_version: ArrayLike,
        evidence_policy_id: str,
        stage_evidence_versions: tuple[ArrayLike, ArrayLike, ArrayLike],
        start_evidence_version: ArrayLike,
        end_evidence_version: ArrayLike,
        start_topology_epoch_id: str,
        end_topology_epoch_id: str,
        start_time: ArrayLike,
        end_time: ArrayLike,
        accepted_step: ArrayLike,
    ) -> None:
        start_time_ = _finite_scalar(start_time, "start_time")
        end_time_ = _finite_scalar(end_time, "end_time")
        end_time_ = eqx.error_if(
            end_time_,
            end_time_ <= start_time_,
            "AcceptedConservationIntegralLedger.end_time must be greater than start_time; got an empty or reversed interval",
        )
        accepted_step_ = _integer_scalar(accepted_step, "accepted_step")
        accepted_step_ = eqx.error_if(
            accepted_step_,
            accepted_step_ < 0,
            "AcceptedConservationIntegralLedger.accepted_step must be nonnegative; got a negative step",
        )
        geometry_family = _flux_identity(geometry_family_id, "geometry_family_id")
        geometry_layout = _flux_identity(geometry_layout_id, "geometry_layout_id")
        if (
            not isinstance(stage_geometry_versions, tuple)
            or len(stage_geometry_versions) != 3
        ):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stage_geometry_versions must identify all three SSPRK stages; got another version bank"
            )
        stage_versions = (
            _integer_scalar(stage_geometry_versions[0], "stage_geometry_versions[0]"),
            _integer_scalar(stage_geometry_versions[1], "stage_geometry_versions[1]"),
            _integer_scalar(stage_geometry_versions[2], "stage_geometry_versions[2]"),
        )
        start_version = _integer_scalar(start_geometry_version, "start_geometry_version")
        end_version = _integer_scalar(end_geometry_version, "end_geometry_version")
        evidence_policy = _flux_identity(evidence_policy_id, "evidence_policy_id")
        if (
            not isinstance(stage_evidence_versions, tuple)
            or len(stage_evidence_versions) != 3
        ):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stage_evidence_versions must identify all three SSPRK stages; got another version bank"
            )
        stage_evidence = (
            _integer_scalar(stage_evidence_versions[0], "stage_evidence_versions[0]"),
            _integer_scalar(stage_evidence_versions[1], "stage_evidence_versions[1]"),
            _integer_scalar(stage_evidence_versions[2], "stage_evidence_versions[2]"),
        )
        start_evidence = _integer_scalar(start_evidence_version, "start_evidence_version")
        end_evidence = _integer_scalar(end_evidence_version, "end_evidence_version")
        start_topology = _flux_identity(
            start_topology_epoch_id, "start_topology_epoch_id"
        )
        end_topology = _flux_identity(end_topology_epoch_id, "end_topology_epoch_id")
        if end_topology != start_topology:
            raise ValueError(
                "AcceptedConservationIntegralLedger.end_topology_epoch_id must match start_topology_epoch_id; got a topology epoch change"
            )
        source = _finite_values(source_integral, "source_integral")
        if source.ndim == 0 or source.shape[0] == 0:
            raise ValueError(
                "AcceptedConservationIntegralLedger.source_integral must begin with a nonempty cell axis; got an empty or scalar source"
            )
        cell_count = source.shape[0]
        component_shape = tuple(source.shape[1:])
        active_cells = _active_cell_array(active_cell_mask, cell_count)
        source = _validate_inactive_values(source, active_cells, "source_integral")
        source = eqx.error_if(
            source,
            start_version != stage_versions[0],
            "AcceptedConservationIntegralLedger.start_geometry_version must match the first stage geometry version; got another version",
        )
        source = eqx.error_if(
            source,
            start_evidence != stage_evidence[0],
            "AcceptedConservationIntegralLedger.start_evidence_version must match the first stage evidence version; got another version",
        )
        blocks_, source = _validate_blocks(
            blocks,
            AcceptedConservationFluxIntegralBlock,
            cell_count,
            component_shape,
            active_cells,
            source,
        )
        self.blocks = blocks_
        self.source_integral = source
        self.active_cell_mask = active_cells
        self.start_time = start_time_
        self.end_time = end_time_
        self.accepted_step = accepted_step_
        self.stage_geometry_versions = stage_versions
        self.start_geometry_version = start_version
        self.end_geometry_version = end_version
        self.stage_evidence_versions = stage_evidence
        self.start_evidence_version = start_evidence
        self.end_evidence_version = end_evidence
        self.geometry_family_id = geometry_family
        self.geometry_layout_id = geometry_layout
        self.evidence_policy_id = evidence_policy
        self.start_topology_epoch_id = start_topology
        self.end_topology_epoch_id = end_topology
        self.cell_count = cell_count
        self.component_shape = component_shape
        self.units = _ACCEPTED_INTEGRAL_UNITS
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "accepted-conservation-integral-ledger",
                "geometry_family_id": geometry_family,
                "geometry_layout_id": geometry_layout,
                "evidence_policy_id": evidence_policy,
                "integration": "ssprk33",
                "ssprk_weights": list(_SSPRK33_WEIGHTS),
                "start_topology_epoch_id": start_topology,
                "end_topology_epoch_id": end_topology,
                "block_integral_ids": [block.integral_block_id for block in blocks_],
                "cell_count": cell_count,
                "component_shape": list(component_shape),
                "active_cell_policy": "exact-boolean-source-zero-active-routes",
                "block_units": _ACCEPTED_INTEGRAL_UNITS,
                "source_units": _ACCEPTED_INTEGRAL_UNITS,
                "units": _ACCEPTED_INTEGRAL_UNITS,
            }
        )

    @classmethod
    @checked
    def integrate_ssprk33(
        cls,
        stage1: ConservationStageLedger,
        stage2: ConservationStageLedger,
        stage3: ConservationStageLedger,
        dt: ArrayLike,
        /,
        *,
        start_geometry_version: ArrayLike,
        end_geometry_version: ArrayLike,
        start_evidence_version: ArrayLike,
        end_evidence_version: ArrayLike,
        start_topology_epoch_id: str,
        end_topology_epoch_id: str,
        start_time: ArrayLike,
        end_time: ArrayLike,
        accepted_step: ArrayLike,
    ) -> AcceptedConservationIntegralLedger:
        """Form accepted SSPRK(3,3) content integrals with one multiplication by dt."""
        stages = (stage1, stage2, stage3)
        start_version = _integer_scalar(start_geometry_version, "start_geometry_version")
        end_version = _integer_scalar(end_geometry_version, "end_geometry_version")
        start_evidence = _integer_scalar(start_evidence_version, "start_evidence_version")
        end_evidence = _integer_scalar(end_evidence_version, "end_evidence_version")
        start_topology = _flux_identity(
            start_topology_epoch_id, "start_topology_epoch_id"
        )
        end_topology = _flux_identity(end_topology_epoch_id, "end_topology_epoch_id")
        start_time_ = _finite_scalar(start_time, "start_time")
        end_time_ = _finite_scalar(end_time, "end_time")
        end_time_ = eqx.error_if(
            end_time_,
            end_time_ <= start_time_,
            "AcceptedConservationIntegralLedger.end_time must be greater than start_time; got an empty or reversed interval",
        )
        accepted_step_ = _integer_scalar(accepted_step, "accepted_step")
        if any(stage.cell_count != stage1.cell_count for stage in stages[1:]):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stages must have identical cell counts; got incompatible cell axes"
            )
        if any(stage.component_shape != stage1.component_shape for stage in stages[1:]):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stages must have identical component shapes; got incompatible component axes"
            )
        if any(
            stage.geometry_family_id != stage1.geometry_family_id for stage in stages[1:]
        ):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stages must share one geometry_family_id; got different geometry families"
            )
        if any(
            stage.geometry_layout_id != stage1.geometry_layout_id for stage in stages[1:]
        ):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stages must share one geometry_layout_id; got different layouts"
            )
        if any(
            stage.evidence_policy_id != stage1.evidence_policy_id for stage in stages[1:]
        ):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stages must share one evidence_policy_id; got different policies"
            )
        if any(
            stage.topology_epoch_id != stage1.topology_epoch_id for stage in stages[1:]
        ):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stages must share one topology_epoch_id; got different topology epochs"
            )
        if start_topology != stage1.topology_epoch_id:
            raise ValueError(
                "AcceptedConservationIntegralLedger.start_topology_epoch_id must match the stage topology; got another epoch"
            )
        if end_topology != stage1.topology_epoch_id:
            raise ValueError(
                "AcceptedConservationIntegralLedger.end_topology_epoch_id must match the stage topology; got another epoch"
            )
        if any(len(stage.blocks) != len(stage1.blocks) for stage in stages[1:]):
            raise ValueError(
                "AcceptedConservationIntegralLedger.stages must have identical block layouts; got different block counts"
            )
        for block_index, corresponding in enumerate(
            zip(*(stage.blocks for stage in stages))
        ):
            expected_identity = (
                corresponding[0].block_id,
                corresponding[0].block_kind,
                corresponding[0].route_id,
                corresponding[0].rate_block_id,
            )
            if any(
                (
                    block.block_id,
                    block.block_kind,
                    block.route_id,
                    block.rate_block_id,
                )
                != expected_identity
                for block in corresponding[1:]
            ):
                raise ValueError(
                    f"AcceptedConservationIntegralLedger.stages must have identical block IDs, kinds, and routes; got differing block {block_index}"
                )
        validated_first_source = stage1.source_rate
        for stage in stages[1:]:
            validated_first_source = eqx.error_if(
                validated_first_source,
                jnp.any(stage.active_cell_mask != stage1.active_cell_mask),
                "AcceptedConservationIntegralLedger.stages must have identical active-cell masks; got changed cell activity",
            )
        step = _finite_values(dt, "dt")
        if step.ndim != 0:
            raise ValueError(
                "AcceptedConservationIntegralLedger.dt must be scalar; got another rank"
            )
        step = eqx.error_if(
            step,
            step <= 0,
            "AcceptedConservationIntegralLedger.dt must be positive; got a nonpositive increment",
        )
        step = eqx.error_if(
            step,
            accepted_step_ < 0,
            "AcceptedConservationIntegralLedger.accepted_step must be nonnegative; got a negative step",
        )
        interval = end_time_ - start_time_
        step = eqx.error_if(
            step,
            ~_time_values_close(step, interval),
            "AcceptedConservationIntegralLedger.dt must match end_time - start_time within dtype-scaled tolerance; got an incompatible increment",
        )
        w1, w2, w3 = _SSPRK33_WEIGHTS

        def integrate_values(first: Array, second: Array, third: Array, /) -> Array:
            weighted_rate = w1 * first + w2 * second + w3 * third
            return step * weighted_rate

        accepted_blocks = tuple(
            AcceptedConservationFluxIntegralBlock._from_stage_rate_block(
                integrate_values(
                    first.flux_rate,
                    second.flux_rate,
                    third.flux_rate,
                ),
                first,
            )
            for first, second, third in zip(*(stage.blocks for stage in stages))
        )
        source_integral = integrate_values(
            validated_first_source,
            stage2.source_rate,
            stage3.source_rate,
        )
        return cls(
            accepted_blocks,
            source_integral,
            stage1.active_cell_mask,
            geometry_family_id=stage1.geometry_family_id,
            geometry_layout_id=stage1.geometry_layout_id,
            # ty: ignore[invalid-argument-type]
            stage_geometry_versions=tuple(stage.geometry_version for stage in stages),
            start_geometry_version=start_version,
            end_geometry_version=end_version,
            evidence_policy_id=stage1.evidence_policy_id,
            # ty: ignore[invalid-argument-type]
            stage_evidence_versions=tuple(stage.evidence_version for stage in stages),
            start_evidence_version=start_evidence,
            end_evidence_version=end_evidence,
            start_topology_epoch_id=start_topology,
            end_topology_epoch_id=end_topology,
            start_time=start_time_,
            end_time=end_time_,
            accepted_step=accepted_step_,
        )

    def scatter_content_integral(self) -> Array:
        """Scatter accepted face and source content without dividing by measures."""
        scattered = self.source_integral
        for block in self.blocks:
            scattered = _scatter_block(
                scattered,
                block.flux_integral,
                block.owner_cells,
                block.neighbor_cells,
                block.neighbor_state_action,
            )
        return scattered

    def frame_exchange_sum(self) -> Array:
        """Accepted reciprocal frame exchange, not a physical source or boundary."""
        return _frame_exchange(
            self.blocks,
            tuple(block.flux_integral for block in self.blocks),
            self.component_shape,
            self.source_integral.dtype,
        )

    def conservation_sums(self) -> tuple[Array, Array, Array]:
        """Return source, boundary-outward, and net-cell accepted content sums."""
        return _conservation_sums(
            self.source_integral,
            tuple((block.flux_integral, block.neighbor_cells) for block in self.blocks),
            self.scatter_content_integral(),
            self.component_shape,
        )


__all__ = [
    "AcceptedConservationFluxIntegralBlock",
    "AcceptedConservationIntegralLedger",
    "ConservationStageFluxRateBlock",
    "ConservationStageLedger",
]
