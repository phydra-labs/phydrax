#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


AMRAxisEntity: TypeAlias = Literal["point", "interval"]


def _minmod(left: Array, right: Array, /) -> Array:
    same = jnp.sign(left) == jnp.sign(right)
    return jnp.where(
        same, jnp.sign(left) * jnp.minimum(jnp.abs(left), jnp.abs(right)), 0.0
    )


class AMREntityTransferReport(StrictModule, NonTrainableState):
    constant_residual: float = eqx.field(static=True)
    conservation_residual: float | None = eqx.field(static=True)
    declared_order: int = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        constant_residual: float,
        conservation_residual: float | None,
        declared_order: int,
        transfer_id: str,
    ):
        constant = float(constant_residual)
        conservation = (
            None if conservation_residual is None else float(conservation_residual)
        )
        self.constant_residual = constant
        self.conservation_residual = conservation
        self.declared_order = int(declared_order)
        self.passed = constant <= 1e-12 and (
            conservation is None or conservation <= 1e-12
        )
        self.report_id = canonical_fingerprint(
            {
                "kind": "amr-entity-transfer-report",
                "transfer": transfer_id,
                "constant_residual": constant,
                "conservation_residual": conservation,
                "declared_order": int(declared_order),
            }
        )


class AMREntityTransferPlan(StrictModule, NonTrainableState):
    """Second-order interval and linear point transfer for cells, nodes, faces, and edges."""

    axis_entities: tuple[AMRAxisEntity, ...] = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    report: AMREntityTransferReport

    def __init__(
        self,
        axis_entities: Sequence[AMRAxisEntity],
        refinement_ratio: int = 2,
        /,
    ):
        entities = tuple(axis_entities)
        ratio = int(refinement_ratio)
        if (
            not entities
            or any(value not in ("point", "interval") for value in entities)
            or ratio <= 1
        ):
            raise ValueError("AMR entity axes and refinement ratio are invalid.")
        identifier = canonical_fingerprint(
            {
                "kind": "amr-entity-transfer",
                "axis_entities": list(entities),
                "refinement_ratio": ratio,
            }
        )
        probe_shape = tuple(4 if value == "interval" else 5 for value in entities)
        probe = jnp.ones(probe_shape)
        prolonged = _prolong(probe, entities, ratio)
        restricted = _restrict(prolonged, entities, ratio)
        constant_residual = float(np.max(np.abs(np.asarray(restricted - probe))))
        conservation = (
            float(abs(np.mean(np.asarray(prolonged)) - np.mean(np.asarray(probe))))
            if all(value == "interval" for value in entities)
            else None
        )
        report = AMREntityTransferReport(
            constant_residual=constant_residual,
            conservation_residual=conservation,
            declared_order=2,
            transfer_id=identifier,
        )
        if not report.passed:
            raise RuntimeError(
                "AMR entity transfer failed constant/conservation evidence."
            )
        self.axis_entities = entities
        self.refinement_ratio = ratio
        self.transfer_id = identifier
        self.report = report

    @classmethod
    def cells(cls, dimensions: int, refinement_ratio: int = 2, /):
        return cls(("interval",) * int(dimensions), refinement_ratio)

    @classmethod
    def nodes(cls, dimensions: int, refinement_ratio: int = 2, /):
        return cls(("point",) * int(dimensions), refinement_ratio)

    @classmethod
    def faces(
        cls,
        dimensions: int,
        normal_axis: int,
        refinement_ratio: int = 2,
        /,
    ):
        entities: list[AMRAxisEntity] = ["interval"] * int(dimensions)
        entities[int(normal_axis)] = "point"
        return cls(entities, refinement_ratio)

    @classmethod
    def edges(
        cls,
        dimensions: int,
        tangent_axis: int,
        refinement_ratio: int = 2,
        /,
    ):
        entities: list[AMRAxisEntity] = ["point"] * int(dimensions)
        entities[int(tangent_axis)] = "interval"
        return cls(entities, refinement_ratio)

    def fine_shape(self, coarse_shape: Sequence[int], /) -> tuple[int, ...]:
        shape = tuple(int(value) for value in coarse_shape)
        if len(shape) != len(self.axis_entities):
            raise ValueError("Coarse shape rank must match AMR entity axes.")
        return tuple(
            size * self.refinement_ratio
            if entity == "interval"
            else (size - 1) * self.refinement_ratio + 1
            for size, entity in zip(shape, self.axis_entities, strict=True)
        )

    def prolong(self, coarse: ArrayLike, /) -> Array:
        value = jnp.asarray(coarse)
        if value.ndim < len(self.axis_entities):
            raise ValueError("Coarse AMR values lack declared spatial axes.")
        return _prolong(value, self.axis_entities, self.refinement_ratio)

    def restrict(self, fine: ArrayLike, /) -> Array:
        value = jnp.asarray(fine)
        if value.ndim < len(self.axis_entities):
            raise ValueError("Fine AMR values lack declared spatial axes.")
        return _restrict(value, self.axis_entities, self.refinement_ratio)


def _prolong(
    value: Array,
    entities: tuple[AMRAxisEntity, ...],
    ratio: int,
    /,
) -> Array:
    result = value
    for axis, entity in enumerate(entities):
        if entity == "interval":
            previous = jnp.roll(result, 1, axis=axis)
            following = jnp.roll(result, -1, axis=axis)
            lower_index: list[slice | int] = [slice(None)] * result.ndim
            upper_index: list[slice | int] = [slice(None)] * result.ndim
            lower_index[axis] = 0
            upper_index[axis] = result.shape[axis] - 1
            previous = previous.at[tuple(lower_index)].set(result[tuple(lower_index)])
            following = following.at[tuple(upper_index)].set(result[tuple(upper_index)])
            slope = _minmod(result - previous, following - result)
            pieces = []
            for child in range(ratio):
                offset = (child + 0.5) / ratio - 0.5
                pieces.append(result + offset * slope)
            stacked = jnp.stack(pieces, axis=axis + 1)
            shape = stacked.shape
            result = stacked.reshape(
                shape[:axis] + (shape[axis] * shape[axis + 1],) + shape[axis + 2 :]
            )
        else:
            old_size = result.shape[axis]
            new_size = (old_size - 1) * ratio + 1
            coordinate = jnp.arange(new_size) / ratio
            lower = jnp.floor(coordinate).astype(jnp.int32)
            upper = jnp.minimum(lower + 1, old_size - 1)
            fraction_shape = [1] * result.ndim
            fraction_shape[axis] = new_size
            fraction = (coordinate - lower).reshape(fraction_shape)
            lower_value = jnp.take(result, lower, axis=axis)
            upper_value = jnp.take(result, upper, axis=axis)
            result = (1.0 - fraction) * lower_value + fraction * upper_value
    return result


def _restrict(
    value: Array,
    entities: tuple[AMRAxisEntity, ...],
    ratio: int,
    /,
) -> Array:
    result = value
    for axis in reversed(range(len(entities))):
        if entities[axis] == "interval":
            if result.shape[axis] % ratio:
                raise ValueError("Fine interval axis must divide by refinement ratio.")
            shape = result.shape
            reshaped = shape[:axis] + (shape[axis] // ratio, ratio) + shape[axis + 1 :]
            result = result.reshape(reshaped).mean(axis=axis + 1)
        else:
            if (result.shape[axis] - 1) % ratio:
                raise ValueError("Fine point axis must be nested by refinement ratio.")
            result = jnp.take(
                result,
                jnp.arange(0, result.shape[axis], ratio),
                axis=axis,
            )
    return result


__all__ = [
    "AMRAxisEntity",
    "AMREntityTransferPlan",
    "AMREntityTransferReport",
]
