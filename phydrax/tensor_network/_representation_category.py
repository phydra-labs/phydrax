#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


class Irrep(StrictModule):
    label: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    dual_label: str = eqx.field(static=True)
    irrep_id: str = eqx.field(static=True)

    def __init__(self, label: str, dimension: int, /, *, dual_label: str):
        label_ = str(label)
        dual = str(dual_label)
        dimension_ = int(dimension)
        if not label_ or not dual or dimension_ < 1:
            raise ValueError("Irrep labels must be nonempty and dimension positive.")
        self.label = label_
        self.dimension = dimension_
        self.dual_label = dual
        self.irrep_id = canonical_fingerprint(
            {
                "kind": "representation-irrep",
                "label": label_,
                "dimension": dimension_,
                "dual": dual,
            }
        )


class RepresentationCategory(StrictModule):
    """Finite explicit fusion category data, independent of Abelian charges."""

    irreps: tuple[Irrep, ...] = eqx.field(static=True)
    unit_label: str = eqx.field(static=True)
    fusion_rules: tuple[tuple[str, str, tuple[tuple[str, int], ...]], ...] = eqx.field(
        static=True
    )
    category_id: str = eqx.field(static=True)

    def __init__(
        self,
        irreps: Sequence[Irrep],
        fusion_rules: Sequence[tuple[str, str, Sequence[tuple[str, int]]]],
        /,
        *,
        unit_label: str,
    ):
        values = tuple(irreps)
        if not values or any(not isinstance(value, Irrep) for value in values):
            raise TypeError("irreps must be a nonempty sequence of Irrep values.")
        labels = tuple(value.label for value in values)
        if len(set(labels)) != len(labels):
            raise ValueError("Representation irrep labels must be unique.")
        unit = str(unit_label)
        if unit not in labels:
            raise ValueError("Representation category unit is not an irrep.")
        if any(value.dual_label not in labels for value in values):
            raise ValueError("Every representation dual must be present.")
        rules = tuple(
            (
                str(left),
                str(right),
                tuple(
                    (str(output), int(multiplicity)) for output, multiplicity in outputs
                ),
            )
            for left, right, outputs in fusion_rules
        )
        pairs = tuple((left, right) for left, right, _ in rules)
        expected_pairs = tuple(product(labels, labels))
        if len(set(pairs)) != len(pairs) or set(pairs) != set(expected_pairs):
            raise ValueError("Fusion rules must specify every ordered irrep pair once.")
        for left, right, outputs in rules:
            if not outputs or len({label for label, _ in outputs}) != len(outputs):
                raise ValueError("Fusion outputs must be nonempty and unique.")
            if any(
                label not in labels or multiplicity < 1 for label, multiplicity in outputs
            ):
                raise ValueError("Fusion outputs or multiplicities are invalid.")
        self.irreps = values
        self.unit_label = unit
        self.fusion_rules = rules
        for label in labels:
            if self.fusion(label, unit) != ((label, 1),) or self.fusion(unit, label) != (
                (label, 1),
            ):
                raise ValueError("Fusion rules violate the unit law.")
        for value in values:
            dual = self.irrep(value.dual_label)
            if dual.dual_label != value.label:
                raise ValueError("Representation duality must be involutive.")
            if unit not in tuple(
                label for label, _ in self.fusion(value.label, value.dual_label)
            ):
                raise ValueError("An irrep does not fuse with its dual to the unit.")
        for left, right, outputs in rules:
            expected_dimension = self.irrep(left).dimension * self.irrep(right).dimension
            fused_dimension = sum(
                multiplicity * self.irrep(output).dimension
                for output, multiplicity in outputs
            )
            if fused_dimension != expected_dimension:
                raise ValueError(
                    "Fusion multiplicities do not preserve representation dimension."
                )
        for left in labels:
            for middle in labels:
                for right in labels:
                    left_counts = self._triple_counts(
                        left, middle, right, left_associated=True
                    )
                    right_counts = self._triple_counts(
                        left, middle, right, left_associated=False
                    )
                    if left_counts != right_counts:
                        raise ValueError("Fusion multiplicities violate associativity.")
        self.category_id = canonical_fingerprint(
            {
                "kind": "representation-category",
                "irreps": tuple(value.irrep_id for value in values),
                "unit": unit,
                "fusion_rules": rules,
            }
        )

    def fusion(self, left: str, right: str, /) -> tuple[tuple[str, int], ...]:
        pair = (str(left), str(right))
        for first, second, outputs in self.fusion_rules:
            if (first, second) == pair:
                return outputs
        raise ValueError("Fusion pair is outside the representation category.")

    def multiplicity(self, left: str, right: str, output: str, /) -> int:
        target = str(output)
        for label, multiplicity in self.fusion(left, right):
            if label == target:
                return multiplicity
        return 0

    def irrep(self, label: str, /) -> Irrep:
        target = str(label)
        for value in self.irreps:
            if value.label == target:
                return value
        raise ValueError("Irrep label is outside the representation category.")

    def _triple_counts(
        self, left: str, middle: str, right: str, /, *, left_associated: bool
    ) -> tuple[int, ...]:
        counts = []
        for target in (value.label for value in self.irreps):
            total = 0
            if left_associated:
                for intermediate, first_multiplicity in self.fusion(left, middle):
                    total += first_multiplicity * self.multiplicity(
                        intermediate, right, target
                    )
            else:
                for intermediate, first_multiplicity in self.fusion(middle, right):
                    total += first_multiplicity * self.multiplicity(
                        left, intermediate, target
                    )
            counts.append(total)
        return tuple(counts)


class FusionChannel(StrictModule):
    category: RepresentationCategory = eqx.field(static=True)
    left: str = eqx.field(static=True)
    right: str = eqx.field(static=True)
    output: str = eqx.field(static=True)
    multiplicity_ordinal: int = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        category: RepresentationCategory,
        left: str,
        right: str,
        output: str,
        /,
        *,
        multiplicity_ordinal: int = 0,
    ):
        if not isinstance(category, RepresentationCategory):
            raise TypeError("category must be RepresentationCategory.")
        left_, right_, output_ = str(left), str(right), str(output)
        ordinal = int(multiplicity_ordinal)
        multiplicity = category.multiplicity(left_, right_, output_)
        if not 0 <= ordinal < multiplicity:
            raise ValueError("Fusion channel multiplicity ordinal is invalid.")
        self.category = category
        self.left = left_
        self.right = right_
        self.output = output_
        self.multiplicity_ordinal = ordinal
        self.channel_id = canonical_fingerprint(
            {
                "kind": "fusion-channel",
                "category": category.category_id,
                "left": left_,
                "right": right_,
                "output": output_,
                "multiplicity_ordinal": ordinal,
            }
        )


class FusionTree(StrictModule):
    """Left-associated finite fusion tree with explicit multiplicity channels."""

    category: RepresentationCategory = eqx.field(static=True)
    leaves: tuple[str, ...] = eqx.field(static=True)
    channels: tuple[FusionChannel, ...] = eqx.field(static=True)
    output: str = eqx.field(static=True)
    tree_id: str = eqx.field(static=True)

    def __init__(
        self,
        category: RepresentationCategory,
        leaves: Sequence[str],
        channels: Sequence[FusionChannel],
        /,
    ):
        if not isinstance(category, RepresentationCategory):
            raise TypeError("category must be RepresentationCategory.")
        leaves_ = tuple(str(label) for label in leaves)
        channels_ = tuple(channels)
        if len(leaves_) < 2 or len(channels_) != len(leaves_) - 1:
            raise ValueError("A fusion tree requires n-1 channels for n leaves.")
        if any(category.irrep(label).label != label for label in leaves_):
            raise ValueError("Fusion tree leaf is outside the category.")
        current = leaves_[0]
        for index, channel in enumerate(channels_):
            if (
                not isinstance(channel, FusionChannel)
                or channel.category.category_id != category.category_id
            ):
                raise ValueError("Fusion tree channel category differs.")
            if channel.left != current or channel.right != leaves_[index + 1]:
                raise ValueError("Fusion tree channels are not left-associated.")
            current = channel.output
        self.category = category
        self.leaves = leaves_
        self.channels = channels_
        self.output = current
        self.tree_id = canonical_fingerprint(
            {
                "kind": "fusion-tree",
                "category": category.category_id,
                "leaves": leaves_,
                "channels": tuple(channel.channel_id for channel in channels_),
            }
        )


def enumerate_fusion_trees(
    category: RepresentationCategory,
    leaves: Sequence[str],
    /,
    *,
    output: str,
) -> tuple[FusionTree, ...]:
    """Enumerate all left-associated paths including multiplicity copies."""

    leaves_ = tuple(str(label) for label in leaves)
    if len(leaves_) < 2:
        raise ValueError("Fusion-tree enumeration requires at least two leaves.")
    partial: list[tuple[str, tuple[FusionChannel, ...]]] = [(leaves_[0], ())]
    for leaf in leaves_[1:]:
        following = []
        for current, channels in partial:
            for result, multiplicity in category.fusion(current, leaf):
                for ordinal in range(multiplicity):
                    channel = FusionChannel(
                        category,
                        current,
                        leaf,
                        result,
                        multiplicity_ordinal=ordinal,
                    )
                    following.append((result, channels + (channel,)))
        partial = following
    return tuple(
        FusionTree(category, leaves_, channels)
        for result, channels in partial
        if result == str(output)
    )


class ReducedLeg(StrictModule):
    category: RepresentationCategory = eqx.field(static=True)
    irreps: tuple[str, ...] = eqx.field(static=True)
    capacities: tuple[int, ...] = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    active_multiplicities: Array
    basis_id: str = eqx.field(static=True)
    allocation_id: str = eqx.field(static=True)

    def __init__(
        self,
        category: RepresentationCategory,
        irreps: Sequence[str],
        capacities: Sequence[int],
        /,
        *,
        orientation: int,
        active_multiplicities: ArrayLike | None = None,
    ):
        if not isinstance(category, RepresentationCategory):
            raise TypeError("category must be RepresentationCategory.")
        labels = tuple(str(label) for label in irreps)
        capacities_ = tuple(int(value) for value in capacities)
        if (
            not labels
            or len(labels) != len(capacities_)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                "Reduced-leg irreps and capacities must be aligned and unique."
            )
        for label in labels:
            category.irrep(label)
        if any(value < 1 for value in capacities_):
            raise ValueError("Reduced-leg capacities must be positive.")
        orientation_ = int(orientation)
        if orientation_ not in (-1, 1):
            raise ValueError("Reduced-leg orientation must be +1 or -1.")
        active = jnp.asarray(
            capacities_ if active_multiplicities is None else active_multiplicities,
            dtype=jnp.int32,
        )
        if active.shape != (len(labels),):
            raise ValueError("One active multiplicity is required per reduced irrep.")
        active = eqx.error_if(
            active,
            jnp.any((active < 0) | (active > jnp.asarray(capacities_))),
            "Active reduced multiplicities exceed allocation.",
        )
        self.category = category
        self.irreps = labels
        self.capacities = capacities_
        self.orientation = orientation_
        self.active_multiplicities = active
        self.basis_id = canonical_fingerprint(
            {
                "kind": "reduced-leg-basis",
                "category": category.category_id,
                "irreps": labels,
                "orientation": orientation_,
            }
        )
        self.allocation_id = canonical_fingerprint(
            {
                "kind": "reduced-leg-allocation",
                "basis": self.basis_id,
                "capacities": capacities_,
            }
        )

    def dual(self) -> ReducedLeg:
        labels = tuple(self.category.irrep(label).dual_label for label in self.irreps)
        return ReducedLeg(
            self.category,
            labels,
            self.capacities,
            orientation=-self.orientation,
            active_multiplicities=self.active_multiplicities,
        )


__all__ = [
    "FusionChannel",
    "FusionTree",
    "Irrep",
    "ReducedLeg",
    "RepresentationCategory",
    "enumerate_fusion_trees",
]
