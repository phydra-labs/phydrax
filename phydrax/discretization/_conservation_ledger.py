#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics._compensated import compensated_sum, compensated_sum_chunks
from .._strict import StrictModule


_STAGE_RATE_UNITS = "content/time"
_ACCEPTED_INTEGRAL_UNITS = "content"
_SSPRK33_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)


def _flux_identity(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string.")
    return value


def _route_array(value: ArrayLike, name: str, /) -> tuple[Array, np.ndarray]:
    host = np.asarray(value)
    if host.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.issubdtype(host.dtype, np.signedinteger):
        raise TypeError(f"{name} must have a signed integer dtype.")
    limits = np.iinfo(np.int32)
    if np.any(host < limits.min) or np.any(host > limits.max):
        raise ValueError(f"{name} contains an index outside the supported range.")
    normalized = np.asarray(host, dtype=np.int32)
    return jnp.asarray(normalized), normalized


def _active_array(value: ArrayLike, face_count: int, /) -> tuple[Array, np.ndarray]:
    host = np.asarray(value)
    if host.dtype != np.dtype(bool):
        raise TypeError("active_mask must have boolean dtype.")
    if host.shape != (face_count,):
        raise ValueError("active_mask must contain one value per routed face.")
    normalized = np.asarray(host, dtype=bool)
    return jnp.asarray(normalized), normalized


def _active_cell_array(value: ArrayLike, cell_count: int, /) -> Array:
    array = jnp.asarray(value)
    if array.dtype != jnp.dtype(bool):
        raise TypeError("active_cell_mask must have boolean dtype.")
    if array.shape != (cell_count,):
        raise ValueError("active_cell_mask must contain one value per cell.")
    return array


def _integer_scalar(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype.")
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
        f"{name} must be exactly zero on inactive cells.",
    )


def _validate_route_values(
    owner_cells: np.ndarray,
    neighbour_cells: np.ndarray,
    active_mask: np.ndarray,
    /,
) -> None:
    if np.any(owner_cells < 0):
        raise ValueError("owner_cells cannot contain negative cell indices.")
    if np.any(neighbour_cells < -1):
        raise ValueError("neighbour_cells can use only -1 as a boundary sentinel.")
    if np.any(active_mask & (neighbour_cells == owner_cells)):
        raise ValueError("An active flux route cannot connect a cell to itself.")


def _finite_values(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.integer)
        or jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise TypeError(f"{name} must have a real numeric dtype.")
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)),
        f"{name} must contain only finite values.",
    )


def _finite_scalar(value: ArrayLike, name: str, /) -> Array:
    array = _finite_values(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
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
    neighbour_cells: np.ndarray,
    active_mask: np.ndarray,
    component_shape: tuple[int, ...],
    block_kind: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "conservation-owner-neighbour-flux-route",
            "owner_cells": array_tree_fingerprint(owner_cells),
            "neighbour_cells": array_tree_fingerprint(neighbour_cells),
            "active_mask": array_tree_fingerprint(active_mask),
            "component_shape": list(component_shape),
            "block_kind": block_kind,
        }
    )


def _validate_route_bounds(
    validation_token: Array,
    owner_cells: Array,
    neighbour_cells: Array,
    active_mask: Array,
    active_cell_mask: Array,
    cell_count: int,
    block_id: str,
    /,
) -> Array:
    validation_token = eqx.error_if(
        validation_token,
        jnp.any((owner_cells < 0) | (owner_cells >= cell_count)),
        f"Flux block {block_id!r} has an owner outside the cell range.",
    )
    validation_token = eqx.error_if(
        validation_token,
        jnp.any((neighbour_cells < -1) | (neighbour_cells >= cell_count)),
        f"Flux block {block_id!r} has a neighbour outside the cell range.",
    )
    validation_token = eqx.error_if(
        validation_token,
        jnp.any(active_mask & (neighbour_cells == owner_cells)),
        f"Flux block {block_id!r} has an active self-neighbour route.",
    )
    safe_neighbour = jnp.maximum(neighbour_cells, 0)
    inactive_endpoint = (~active_cell_mask[owner_cells]) | (
        (neighbour_cells >= 0) & (~active_cell_mask[safe_neighbour])
    )
    return eqx.error_if(
        validation_token,
        jnp.any(active_mask & inactive_endpoint),
        f"Flux block {block_id!r} has an active route through an inactive cell.",
    )


def _validate_blocks(
    blocks: Sequence[
        ConservationStageFluxRateBlock | AcceptedConservationFluxIntegralBlock
    ],
    block_type: type,
    cell_count: int,
    component_shape: tuple[int, ...],
    active_cell_mask: Array,
    validation_token: Array,
    /,
) -> tuple[tuple, Array]:
    normalized = tuple(blocks)
    if any(not isinstance(block, block_type) for block in normalized):
        raise TypeError(f"blocks must contain only {block_type.__name__} values.")
    block_ids = tuple(block.block_id for block in normalized)
    route_ids = tuple(block.route_id for block in normalized)
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(
            f"Conservation flux block IDs must be unique within a ledger: {block_ids!r}."
        )
    if len(set(route_ids)) != len(route_ids):
        raise ValueError(
            f"Conservation flux routes must be unique within a ledger: {route_ids!r}."
        )
    for block in normalized:
        if block.component_shape != component_shape:
            raise ValueError(
                f"Flux block {block.block_id!r} has a different component shape."
            )
        validation_token = _validate_route_bounds(
            validation_token,
            block.owner_cells,
            block.neighbour_cells,
            block.active_mask,
            active_cell_mask,
            cell_count,
            block.block_id,
        )
    return normalized, validation_token


def _scatter_block(
    scattered: Array,
    values: Array,
    owner_cells: Array,
    neighbour_cells: Array,
    /,
) -> Array:
    safe_neighbour = jnp.maximum(neighbour_cells, 0)
    scattered = scattered.at[owner_cells].add(-values)
    neighbour_mask = (neighbour_cells >= 0).reshape(
        neighbour_cells.shape + (1,) * (values.ndim - 1)
    )
    return scattered.at[safe_neighbour].add(
        jnp.where(neighbour_mask, values, jnp.zeros((), dtype=values.dtype))
    )


class ConservationStageFluxRateBlock(StrictModule):
    """One immutable owner-oriented block of area-integrated stage flux rates."""

    flux_rate: Array
    owner_cells: Array
    neighbour_cells: Array
    active_mask: Array
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
        neighbour_cells: ArrayLike,
        active_mask: ArrayLike,
        block_id: str,
        block_kind: str,
        /,
    ):
        block_id_ = _flux_identity(block_id, "block_id")
        block_kind_ = _flux_identity(block_kind, "block_kind")
        owner, owner_host = _route_array(owner_cells, "owner_cells")
        neighbour, neighbour_host = _route_array(neighbour_cells, "neighbour_cells")
        if neighbour.shape != owner.shape:
            raise ValueError(
                "owner_cells and neighbour_cells must have identical shapes."
            )
        active, active_host = _active_array(active_mask, owner.shape[0])
        _validate_route_values(owner_host, neighbour_host, active_host)
        rate = _finite_values(flux_rate, "flux_rate")
        if rate.ndim == 0 or rate.shape[0] != owner.shape[0]:
            raise ValueError("flux_rate must begin with the routed face count.")
        component_shape = tuple(rate.shape[1:])
        rate = _masked_face_values(rate, active)
        route_id = _route_fingerprint(
            owner_host,
            neighbour_host,
            active_host,
            component_shape,
            block_kind_,
        )
        self.flux_rate = rate
        self.owner_cells = owner
        self.neighbour_cells = neighbour
        self.active_mask = active
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
        """Replace rates while reusing this block's host-validated route identity."""
        rate = _finite_values(flux_rate, "flux_rate")
        if rate.shape != self.flux_rate.shape:
            raise ValueError("flux_rate must have the exact route-template shape.")
        rate = _masked_face_values(rate, self.active_mask)
        instance = object.__new__(type(self))
        object.__setattr__(instance, "flux_rate", rate)
        object.__setattr__(instance, "owner_cells", self.owner_cells)
        object.__setattr__(instance, "neighbour_cells", self.neighbour_cells)
        object.__setattr__(instance, "active_mask", self.active_mask)
        object.__setattr__(instance, "block_id", self.block_id)
        object.__setattr__(instance, "block_kind", self.block_kind)
        object.__setattr__(instance, "component_shape", self.component_shape)
        object.__setattr__(instance, "units", self.units)
        object.__setattr__(instance, "route_id", self.route_id)
        object.__setattr__(instance, "rate_block_id", self.rate_block_id)
        return instance


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
        differentiability_policy_id: str = "smooth-discrete",
    ):
        geometry_family = _flux_identity(geometry_family_id, "geometry_family_id")
        geometry_layout = _flux_identity(geometry_layout_id, "geometry_layout_id")
        version = _integer_scalar(geometry_version, "geometry_version")
        evidence_policy = _flux_identity(evidence_policy_id, "evidence_policy_id")
        evidence = _integer_scalar(evidence_version, "evidence_version")
        topology_id = _flux_identity(topology_epoch_id, "topology_epoch_id")
        source = _finite_values(source_rate, "source_rate")
        if source.ndim == 0 or source.shape[0] == 0:
            raise ValueError("source_rate must begin with a nonempty cell axis.")
        cell_count = int(source.shape[0])
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
                "High-, low-, and selected-order ledger routes must match exactly."
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
            raise ValueError("Blend factors must contain one scalar per routed face.")
        factors = tuple(
            eqx.error_if(
                factor,
                jnp.any((factor < 0.0) | (factor > 1.0) | ~jnp.isfinite(factor)),
                "Blend factors must be finite and lie in [0, 1].",
            )
            for factor in raw_factors
        )
        entropy = (
            jnp.zeros((cell_count,), dtype=source.dtype)
            if entropy_production is None
            else jnp.asarray(entropy_production, dtype=source.dtype)
        )
        troubled = (
            jnp.zeros((cell_count,), dtype=bool)
            if troubled_cell_mask is None
            else jnp.asarray(troubled_cell_mask, dtype=bool)
        )
        levels = (
            jnp.zeros((cell_count,), dtype=jnp.int32)
            if correction_level is None
            else jnp.asarray(correction_level, dtype=jnp.int32)
        )
        accepted_ = jnp.asarray(accepted, dtype=bool)
        if (
            entropy.shape != (cell_count,)
            or troubled.shape != (cell_count,)
            or levels.shape != (cell_count,)
            or accepted_.shape != ()
        ):
            raise ValueError("Stage robustness evidence has incompatible shape.")
        entropy = eqx.error_if(
            entropy,
            jnp.any(~jnp.isfinite(entropy)),
            "Stage entropy production must be finite.",
        )
        levels = eqx.error_if(
            levels,
            jnp.any(levels < 0),
            "Stage correction levels must be nonnegative.",
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
            raise ValueError("Selected stage blocks must reuse exact ledger routes.")
        source = jnp.asarray(source_rate)
        if source.shape != self.source_rate.shape:
            raise ValueError("Selected source rate changed shape.")
        factors = tuple(jnp.asarray(value) for value in blend_factors)
        if len(factors) != len(selected):
            raise ValueError("Selected ledger needs one factor block per route block.")
        instance = object.__new__(type(self))
        object.__setattr__(instance, "blocks", selected)
        object.__setattr__(instance, "high_order_blocks", self.high_order_blocks)
        object.__setattr__(instance, "low_order_blocks", self.low_order_blocks)
        object.__setattr__(instance, "blend_factors", factors)
        object.__setattr__(instance, "entropy_production", self.entropy_production)
        object.__setattr__(
            instance, "troubled_cell_mask", jnp.asarray(troubled_cell_mask)
        )
        object.__setattr__(instance, "correction_level", jnp.asarray(correction_level))
        object.__setattr__(instance, "accepted", jnp.asarray(accepted))
        object.__setattr__(instance, "source_rate", source)
        object.__setattr__(instance, "active_cell_mask", self.active_cell_mask)
        object.__setattr__(instance, "geometry_family_id", self.geometry_family_id)
        object.__setattr__(instance, "geometry_layout_id", self.geometry_layout_id)
        object.__setattr__(instance, "topology_epoch_id", self.topology_epoch_id)
        object.__setattr__(instance, "evidence_policy_id", self.evidence_policy_id)
        object.__setattr__(
            instance,
            "differentiability_policy_id",
            str(differentiability_policy_id),
        )
        object.__setattr__(instance, "cell_count", self.cell_count)
        object.__setattr__(instance, "component_shape", self.component_shape)
        object.__setattr__(instance, "units", self.units)
        object.__setattr__(instance, "ledger_id", self.ledger_id)
        return instance

    def scatter_content_rate(self) -> Array:
        """Scatter owner-outward rates and sources without multiplying by a step."""
        scattered = self.source_rate
        for block in self.blocks:
            scattered = _scatter_block(
                scattered,
                block.flux_rate,
                block.owner_cells,
                block.neighbour_cells,
            )
        return scattered


class AcceptedConservationFluxIntegralBlock(StrictModule):
    """One immutable owner-oriented block of accepted content integrals."""

    flux_integral: Array
    owner_cells: Array
    neighbour_cells: Array
    active_mask: Array
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
        neighbour_cells: ArrayLike,
        active_mask: ArrayLike,
        block_id: str,
        block_kind: str,
        /,
    ):
        block_id_ = _flux_identity(block_id, "block_id")
        block_kind_ = _flux_identity(block_kind, "block_kind")
        owner, owner_host = _route_array(owner_cells, "owner_cells")
        neighbour, neighbour_host = _route_array(neighbour_cells, "neighbour_cells")
        if neighbour.shape != owner.shape:
            raise ValueError(
                "owner_cells and neighbour_cells must have identical shapes."
            )
        active, active_host = _active_array(active_mask, owner.shape[0])
        _validate_route_values(owner_host, neighbour_host, active_host)
        integral = _finite_values(flux_integral, "flux_integral")
        if integral.ndim == 0 or integral.shape[0] != owner.shape[0]:
            raise ValueError("flux_integral must begin with the routed face count.")
        component_shape = tuple(integral.shape[1:])
        integral = _masked_face_values(integral, active)
        route_id = _route_fingerprint(
            owner_host,
            neighbour_host,
            active_host,
            component_shape,
            block_kind_,
        )
        self.flux_integral = integral
        self.owner_cells = owner
        self.neighbour_cells = neighbour
        self.active_mask = active
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
    def _from_stage_rate_block(
        cls,
        flux_integral: ArrayLike,
        stage_block: ConservationStageFluxRateBlock,
        /,
    ) -> AcceptedConservationFluxIntegralBlock:
        integral = _finite_values(flux_integral, "flux_integral")
        if integral.shape != stage_block.flux_rate.shape:
            raise ValueError(
                "flux_integral must have the exact stage flux-rate block shape."
            )
        integral = _masked_face_values(integral, stage_block.active_mask)
        instance = object.__new__(cls)
        object.__setattr__(instance, "flux_integral", integral)
        object.__setattr__(instance, "owner_cells", stage_block.owner_cells)
        object.__setattr__(instance, "neighbour_cells", stage_block.neighbour_cells)
        object.__setattr__(instance, "active_mask", stage_block.active_mask)
        object.__setattr__(instance, "block_id", stage_block.block_id)
        object.__setattr__(instance, "block_kind", stage_block.block_kind)
        object.__setattr__(instance, "component_shape", stage_block.component_shape)
        object.__setattr__(instance, "units", _ACCEPTED_INTEGRAL_UNITS)
        object.__setattr__(instance, "route_id", stage_block.route_id)
        object.__setattr__(
            instance,
            "integral_block_id",
            canonical_fingerprint(
                {
                    "kind": "accepted-conservation-flux-integral-block",
                    "block_id": stage_block.block_id,
                    "block_kind": stage_block.block_kind,
                    "route": stage_block.route_id,
                    "units": _ACCEPTED_INTEGRAL_UNITS,
                }
            ),
        )
        return instance


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
    ):
        start_time_ = _finite_scalar(start_time, "start_time")
        end_time_ = _finite_scalar(end_time, "end_time")
        end_time_ = eqx.error_if(
            end_time_,
            end_time_ <= start_time_,
            "end_time must be greater than start_time.",
        )
        accepted_step_ = _integer_scalar(accepted_step, "accepted_step")
        accepted_step_ = eqx.error_if(
            accepted_step_,
            accepted_step_ < 0,
            "accepted_step must be nonnegative.",
        )
        geometry_family = _flux_identity(geometry_family_id, "geometry_family_id")
        geometry_layout = _flux_identity(geometry_layout_id, "geometry_layout_id")
        if (
            not isinstance(stage_geometry_versions, tuple)
            or len(stage_geometry_versions) != 3
        ):
            raise ValueError(
                "stage_geometry_versions must identify all three SSPRK stages."
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
                "stage_evidence_versions must identify all three SSPRK stages."
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
                "An accepted flux-integral ledger cannot span a topology epoch change."
            )
        source = _finite_values(source_integral, "source_integral")
        if source.ndim == 0 or source.shape[0] == 0:
            raise ValueError("source_integral must begin with a nonempty cell axis.")
        cell_count = int(source.shape[0])
        component_shape = tuple(source.shape[1:])
        active_cells = _active_cell_array(active_cell_mask, cell_count)
        source = _validate_inactive_values(source, active_cells, "source_integral")
        source = eqx.error_if(
            source,
            start_version != stage_versions[0],
            "start_geometry_version must match the first stage geometry version.",
        )
        source = eqx.error_if(
            source,
            start_evidence != stage_evidence[0],
            "start_evidence_version must match the first stage evidence version.",
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
        if any(not isinstance(stage, ConservationStageLedger) for stage in stages):
            raise TypeError("SSPRK integration requires three stage flux-rate ledgers.")
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
            "end_time must be greater than start_time.",
        )
        accepted_step_ = _integer_scalar(accepted_step, "accepted_step")
        if any(stage.cell_count != stage1.cell_count for stage in stages[1:]):
            raise ValueError("SSPRK stage ledgers must have identical cell counts.")
        if any(stage.component_shape != stage1.component_shape for stage in stages[1:]):
            raise ValueError("SSPRK stage ledgers must have identical component shapes.")
        if any(
            stage.geometry_family_id != stage1.geometry_family_id for stage in stages[1:]
        ):
            raise ValueError("SSPRK stage ledgers must share one geometry family.")
        if any(
            stage.geometry_layout_id != stage1.geometry_layout_id for stage in stages[1:]
        ):
            raise ValueError("SSPRK stage ledgers must share one geometry layout.")
        if any(
            stage.evidence_policy_id != stage1.evidence_policy_id for stage in stages[1:]
        ):
            raise ValueError("SSPRK stage ledgers must share one evidence policy.")
        if any(
            stage.topology_epoch_id != stage1.topology_epoch_id for stage in stages[1:]
        ):
            raise ValueError("SSPRK stage ledgers must share one topology epoch.")
        if start_topology != stage1.topology_epoch_id:
            raise ValueError("start_topology_epoch_id must match the stage topology.")
        if end_topology != stage1.topology_epoch_id:
            raise ValueError("end_topology_epoch_id must match the stage topology.")
        if any(len(stage.blocks) != len(stage1.blocks) for stage in stages[1:]):
            raise ValueError("SSPRK stage ledgers must have identical block layouts.")
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
                    "SSPRK stage ledgers must have identical block IDs, block kinds, "
                    f"and routes; block {block_index} differs."
                )
        validated_first_source = stage1.source_rate
        for stage in stages[1:]:
            validated_first_source = eqx.error_if(
                validated_first_source,
                jnp.any(stage.active_cell_mask != stage1.active_cell_mask),
                "SSPRK stage ledgers must have identical active-cell masks.",
            )
        step = _finite_values(dt, "dt")
        if step.ndim != 0:
            raise ValueError("dt must be scalar.")
        step = eqx.error_if(step, step <= 0, "dt must be positive.")
        step = eqx.error_if(
            step,
            accepted_step_ < 0,
            "accepted_step must be nonnegative.",
        )
        interval = end_time_ - start_time_
        step = eqx.error_if(
            step,
            ~_time_values_close(step, interval),
            "dt must match end_time - start_time within dtype-scaled tolerance.",
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
            stage_geometry_versions=tuple(stage.geometry_version for stage in stages),
            start_geometry_version=start_version,
            end_geometry_version=end_version,
            evidence_policy_id=stage1.evidence_policy_id,
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
                block.neighbour_cells,
            )
        return scattered

    def conservation_sums(self) -> tuple[Array, Array, Array]:
        """Return source, boundary-outward, and net-cell accepted content sums."""
        source_sum = compensated_sum(self.source_integral, axis=0)
        boundary_chunks_list: list[Array] = []
        for block in self.blocks:
            boundary = (block.neighbour_cells < 0).reshape(
                block.neighbour_cells.shape + (1,) * (block.flux_integral.ndim - 1)
            )
            boundary_chunks_list.append(
                jnp.where(
                    boundary,
                    block.flux_integral,
                    jnp.zeros((), dtype=block.flux_integral.dtype),
                )
            )
        boundary_chunks = tuple(boundary_chunks_list)
        boundary_sum = (
            compensated_sum_chunks(
                boundary_chunks,
                output_ndim=len(self.component_shape),
            )
            if boundary_chunks
            else jnp.zeros(self.component_shape, dtype=self.source_integral.dtype)
        )
        net_cell_sum = compensated_sum(self.scatter_content_integral(), axis=0)
        return source_sum, boundary_sum, net_cell_sum


__all__ = [
    "AcceptedConservationFluxIntegralBlock",
    "AcceptedConservationIntegralLedger",
    "ConservationStageFluxRateBlock",
    "ConservationStageLedger",
]
