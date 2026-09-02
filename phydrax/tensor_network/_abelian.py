#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._precision import TensorNetworkPrecisionPolicy


AbelianCharge: TypeAlias = tuple[int, ...]


class AbelianGroup(StrictModule):
    """Finite direct product of U(1) and cyclic Abelian charge components."""

    components: tuple[int | None, ...] = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(self, components: Sequence[int | None], /):
        values = tuple(components)
        if not values:
            raise ValueError("An Abelian group requires at least one component.")
        normalized = []
        for component in values:
            if component is None:
                normalized.append(None)
            else:
                if isinstance(component, bool) or int(component) < 2:
                    raise ValueError(
                        "Cyclic Abelian moduli must be integers at least two."
                    )
                normalized.append(int(component))
        self.components = tuple(normalized)
        self.group_id = canonical_fingerprint(
            {"kind": "abelian-group", "components": self.components}
        )

    @property
    def zero(self) -> AbelianCharge:
        return (0,) * len(self.components)

    def normalize(self, charge: Sequence[int], /) -> AbelianCharge:
        values = tuple(charge)
        if len(values) != len(self.components):
            raise ValueError("Charge component count does not match the Abelian group.")
        output = []
        for value, modulus in zip(values, self.components, strict=True):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Abelian charges must contain integers.")
            output.append(value if modulus is None else value % modulus)
        return tuple(output)

    def add(self, *charges: Sequence[int]) -> AbelianCharge:
        total = [0] * len(self.components)
        for charge in charges:
            values = self.normalize(charge)
            total = [left + right for left, right in zip(total, values, strict=True)]
        return self.normalize(total)

    def negate(self, charge: Sequence[int], /) -> AbelianCharge:
        return self.normalize(tuple(-value for value in self.normalize(charge)))


class AbelianLeg(StrictModule):
    group: AbelianGroup = eqx.field(static=True)
    charges: tuple[AbelianCharge, ...] = eqx.field(static=True)
    capacities: tuple[int, ...] = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    active_degeneracies: Array
    leg_id: str = eqx.field(static=True)

    def __init__(
        self,
        group: AbelianGroup,
        charges: Sequence[Sequence[int]],
        capacities: Sequence[int],
        /,
        *,
        orientation: int,
        active_degeneracies: ArrayLike | None = None,
    ):
        if not isinstance(group, AbelianGroup):
            raise TypeError("group must be AbelianGroup.")
        charges_ = tuple(group.normalize(charge) for charge in charges)
        capacities_ = tuple(int(value) for value in capacities)
        if not charges_ or len(charges_) != len(capacities_):
            raise ValueError(
                "Abelian leg charges and capacities must be nonempty and aligned."
            )
        if len(set(charges_)) != len(charges_):
            raise ValueError("Abelian leg charges must be unique in declared order.")
        if any(value < 1 for value in capacities_):
            raise ValueError("Abelian leg capacities must be positive.")
        orientation_ = int(orientation)
        if orientation_ not in (-1, 1):
            raise ValueError("Abelian leg orientation must be +1 or -1.")
        active = jnp.asarray(
            capacities_ if active_degeneracies is None else active_degeneracies,
            dtype=jnp.int32,
        )
        if active.shape != (len(charges_),):
            raise ValueError("active_degeneracies must have one entry per charge.")
        bounds = jnp.asarray(capacities_, dtype=jnp.int32)
        active = eqx.error_if(
            active,
            jnp.any((active < 0) | (active > bounds)),
            "Active Abelian degeneracies must lie within static capacities.",
        )
        self.group = group
        self.charges = charges_
        self.capacities = capacities_
        self.orientation = orientation_
        self.active_degeneracies = active
        self.leg_id = canonical_fingerprint(
            {
                "kind": "abelian-leg",
                "group": group.group_id,
                "charges": charges_,
                "capacities": capacities_,
                "orientation": orientation_,
            }
        )

    @property
    def size(self) -> int:
        return sum(self.capacities)

    def with_active(self, active_degeneracies: ArrayLike, /) -> AbelianLeg:
        return AbelianLeg(
            self.group,
            self.charges,
            self.capacities,
            orientation=self.orientation,
            active_degeneracies=active_degeneracies,
        )

    def dual(self) -> AbelianLeg:
        return AbelianLeg(
            self.group,
            self.charges,
            self.capacities,
            orientation=-self.orientation,
            active_degeneracies=self.active_degeneracies,
        )

    def dual_compatible(self, other: AbelianLeg, /) -> bool:
        return (
            isinstance(other, AbelianLeg)
            and self.group.group_id == other.group.group_id
            and self.charges == other.charges
            and self.capacities == other.capacities
            and self.orientation == -other.orientation
        )


class AbelianTensorLayout(StrictModule):
    legs: tuple[AbelianLeg, ...]
    total_charge: AbelianCharge = eqx.field(static=True)
    sectors: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    block_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        legs: Sequence[AbelianLeg],
        /,
        *,
        total_charge: Sequence[int] | None = None,
    ):
        values = tuple(legs)
        if not values or any(not isinstance(leg, AbelianLeg) for leg in values):
            raise TypeError("legs must be a nonempty sequence of AbelianLeg values.")
        group = values[0].group
        if any(leg.group.group_id != group.group_id for leg in values):
            raise ValueError("Every Abelian tensor leg must use the same group.")
        total = group.zero if total_charge is None else group.normalize(total_charge)
        sectors = []
        shapes = []
        for sector in product(*(range(len(leg.charges)) for leg in values)):
            oriented = tuple(
                leg.charges[ordinal]
                if leg.orientation > 0
                else group.negate(leg.charges[ordinal])
                for leg, ordinal in zip(values, sector, strict=True)
            )
            if group.add(*oriented) == total:
                sectors.append(tuple(sector))
                shapes.append(
                    tuple(
                        leg.capacities[ordinal]
                        for leg, ordinal in zip(values, sector, strict=True)
                    )
                )
        if not sectors:
            raise ValueError("Abelian tensor layout has no charge-conserving sectors.")
        self.legs = values
        self.total_charge = total
        self.sectors = tuple(sectors)
        self.block_shapes = tuple(shapes)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "abelian-tensor-layout",
                "legs": tuple(leg.leg_id for leg in values),
                "total_charge": total,
                "sectors": self.sectors,
            }
        )


class AbelianTensor(StrictModule):
    layout: AbelianTensorLayout
    blocks: tuple[Array, ...]
    precision: TensorNetworkPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    tensor_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: AbelianTensorLayout,
        blocks: Sequence[ArrayLike],
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
    ):
        if not isinstance(layout, AbelianTensorLayout):
            raise TypeError("layout must be AbelianTensorLayout.")
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        raw_values = tuple(precision_.storage(jnp.asarray(block)) for block in blocks)
        if len(raw_values) != len(layout.sectors):
            raise ValueError("Abelian tensor requires one block per allowed sector.")
        checked = []
        for block, shape, sector in zip(
            raw_values, layout.block_shapes, layout.sectors, strict=True
        ):
            if block.shape != shape:
                raise ValueError("Abelian tensor block shape does not match its layout.")
            mask = jnp.asarray(True)
            for axis, (leg, ordinal) in enumerate(zip(layout.legs, sector, strict=True)):
                axis_mask = jnp.arange(shape[axis]) < leg.active_degeneracies[ordinal]
                reshape = [1] * len(shape)
                reshape[axis] = shape[axis]
                mask = mask & axis_mask.reshape(tuple(reshape))
            checked.append(
                eqx.error_if(
                    block,
                    jnp.any((~mask) & (block != 0)),
                    "Inactive Abelian block padding must be exactly zero.",
                )
            )
        values = tuple(checked)
        precision_.validate_storage(values)
        self.layout = layout
        self.blocks = values
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(values)
        self.tensor_id = canonical_fingerprint(
            {
                "kind": "abelian-tensor",
                "layout": layout.layout_id,
                "dtype": str(values[0].dtype) if values else "empty",
                "precision": precision_.policy_id,
            }
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(leg.size for leg in self.layout.legs)

    def to_dense(self) -> Array:
        dtype = self.blocks[0].dtype if self.blocks else jnp.float64
        output = jnp.zeros(self.shape, dtype=dtype)
        offsets = []
        for leg in self.layout.legs:
            starts = []
            start = 0
            for capacity in leg.capacities:
                starts.append(start)
                start += capacity
            offsets.append(tuple(starts))
        for sector, block in zip(self.layout.sectors, self.blocks, strict=True):
            slices = tuple(
                slice(offsets[axis][ordinal], offsets[axis][ordinal] + block.shape[axis])
                for axis, ordinal in enumerate(sector)
            )
            output = output.at[slices].set(block)
        return self.precision.output(output)

    @classmethod
    def from_dense(
        cls,
        layout: AbelianTensorLayout,
        value: ArrayLike,
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
        forbidden_tolerance: float = 1e-10,
    ) -> AbelianTensor:
        array = jnp.asarray(value)
        expected = tuple(leg.size for leg in layout.legs)
        if array.shape != expected:
            raise ValueError("Dense tensor shape does not match the Abelian layout.")
        offsets = []
        for leg in layout.legs:
            starts = []
            start = 0
            for capacity in leg.capacities:
                starts.append(start)
                start += capacity
            offsets.append(tuple(starts))
        blocks = []
        reconstruction = jnp.zeros_like(array)
        for sector, shape in zip(layout.sectors, layout.block_shapes, strict=True):
            slices = tuple(
                slice(offsets[axis][ordinal], offsets[axis][ordinal] + shape[axis])
                for axis, ordinal in enumerate(sector)
            )
            block = array[slices]
            blocks.append(block)
            reconstruction = reconstruction.at[slices].set(block)
        violation = jnp.max(jnp.abs(array - reconstruction)) > float(forbidden_tolerance)
        blocks[0] = eqx.error_if(
            blocks[0],
            violation,
            "Dense tensor contains charge-forbidden entries.",
        )
        return cls(layout, tuple(blocks), precision=precision)


__all__ = [
    "AbelianCharge",
    "AbelianGroup",
    "AbelianLeg",
    "AbelianTensor",
    "AbelianTensorLayout",
]
