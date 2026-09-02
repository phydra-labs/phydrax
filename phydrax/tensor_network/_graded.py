#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._abelian import (
    AbelianGroup,
    AbelianLeg,
    AbelianTensor,
    AbelianTensorLayout,
)
from ._abelian_plan import AbelianContractionPlan, contract_abelian_tensors


class FermionGrading(StrictModule):
    """Explicit homomorphism from a charge group to Z2; never inferred from charge."""

    group: AbelianGroup = eqx.field(static=True)
    generator_parities: tuple[int, ...] = eqx.field(static=True)
    grading_id: str = eqx.field(static=True)

    def __init__(self, group: AbelianGroup, generator_parities: Sequence[int], /):
        if not isinstance(group, AbelianGroup):
            raise TypeError("group must be AbelianGroup.")
        parities = tuple(int(value) for value in generator_parities)
        if len(parities) != len(group.components) or any(
            value not in (0, 1) for value in parities
        ):
            raise ValueError("One binary generator parity is required per component.")
        for modulus, parity in zip(group.components, parities, strict=True):
            if modulus is not None and modulus % 2 == 1 and parity:
                raise ValueError(
                    "An odd-order cyclic component has no nontrivial Z2 grading."
                )
        self.group = group
        self.generator_parities = parities
        self.grading_id = canonical_fingerprint(
            {
                "kind": "fermion-grading-homomorphism",
                "group": group.group_id,
                "generator_parities": parities,
            }
        )

    def parity(self, charge: Sequence[int], /) -> int:
        values = self.group.normalize(charge)
        return (
            sum(
                value * parity
                for value, parity in zip(values, self.generator_parities, strict=True)
            )
            % 2
        )

    def verify_homomorphism(self, charges: Sequence[Sequence[int]], /) -> bool:
        values = tuple(self.group.normalize(charge) for charge in charges)
        return all(
            self.parity(self.group.add(left, right))
            == (self.parity(left) + self.parity(right)) % 2
            for left in values
            for right in values
        )


class GradedLeg(StrictModule):
    """One Abelian leg plus an explicit grading and global mode label."""

    leg: AbelianLeg
    grading: FermionGrading = eqx.field(static=True)
    mode_label: str = eqx.field(static=True)
    sector_parities: tuple[int, ...] = eqx.field(static=True)
    graded_leg_id: str = eqx.field(static=True)

    def __init__(self, leg: AbelianLeg, grading: FermionGrading, /, *, mode_label: str):
        if not isinstance(leg, AbelianLeg) or not isinstance(grading, FermionGrading):
            raise TypeError("leg and grading must be AbelianLeg and FermionGrading.")
        if leg.group.group_id != grading.group.group_id:
            raise ValueError("Graded leg and grading groups must match.")
        label = str(mode_label)
        if not label:
            raise ValueError("Graded leg mode_label must be nonempty.")
        parities = tuple(grading.parity(charge) for charge in leg.charges)
        self.leg = leg
        self.grading = grading
        self.mode_label = label
        self.sector_parities = parities
        self.graded_leg_id = canonical_fingerprint(
            {
                "kind": "graded-leg",
                "leg": leg.allocation_id,
                "grading": grading.grading_id,
                "mode_label": label,
            }
        )

    def dual(self) -> GradedLeg:
        return GradedLeg(self.leg.dual(), self.grading, mode_label=self.mode_label)


class GradedTensor(StrictModule):
    """Block-sparse tensor whose Koszul convention is part of its value type."""

    tensor: AbelianTensor
    legs: tuple[GradedLeg, ...]
    grading_id: str = eqx.field(static=True)
    graded_tensor_id: str = eqx.field(static=True)

    def __init__(self, tensor: AbelianTensor, legs: Sequence[GradedLeg], /):
        if not isinstance(tensor, AbelianTensor):
            raise TypeError("tensor must be AbelianTensor.")
        values = tuple(legs)
        if len(values) != len(tensor.layout.legs) or any(
            not isinstance(leg, GradedLeg) for leg in values
        ):
            raise ValueError("One GradedLeg is required per tensor axis.")
        grading_id = values[0].grading.grading_id
        if any(leg.grading.grading_id != grading_id for leg in values):
            raise ValueError("Every graded tensor leg must share one grading.")
        if len({leg.mode_label for leg in values}) != len(values):
            raise ValueError("Mode labels within a graded tensor must be unique.")
        if any(
            leg.leg.allocation_id != tensor_leg.allocation_id
            for leg, tensor_leg in zip(values, tensor.layout.legs, strict=True)
        ):
            raise ValueError("Graded legs must match Abelian tensor allocations.")
        self.tensor = tensor
        self.legs = values
        self.grading_id = grading_id
        self.graded_tensor_id = canonical_fingerprint(
            {
                "kind": "graded-tensor",
                "tensor": tensor.allocation_id,
                "legs": tuple(leg.graded_leg_id for leg in values),
            }
        )


def _permutation_sign(permutation: tuple[int, ...], parities: tuple[int, ...], /) -> int:
    positions = tuple(permutation.index(axis) for axis in range(len(permutation)))
    exponent = sum(
        parities[first] * parities[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
        if positions[first] > positions[second]
    )
    return -1 if exponent % 2 else 1


def graded_permute(tensor: GradedTensor, permutation: Sequence[int], /) -> GradedTensor:
    """Permute tensor axes with the sector-resolved Koszul sign."""

    if not isinstance(tensor, GradedTensor):
        raise TypeError("tensor must be GradedTensor.")
    rank = len(tensor.legs)
    order = tuple(int(axis) % rank for axis in permutation)
    if len(order) != rank or len(set(order)) != rank:
        raise ValueError("Graded permutation must contain every axis exactly once.")
    output_legs = tuple(tensor.legs[axis] for axis in order)
    layout = AbelianTensorLayout(
        tuple(leg.leg for leg in output_legs),
        total_charge=tensor.tensor.layout.total_charge,
    )
    blocks = []
    for output_sector in layout.sectors:
        source_sector_values = [0] * rank
        for output_axis, source_axis in enumerate(order):
            source_sector_values[source_axis] = output_sector[output_axis]
        source_sector = tuple(source_sector_values)
        source_index = tensor.tensor.layout.sectors.index(source_sector)
        parities = tuple(
            leg.sector_parities[ordinal]
            for leg, ordinal in zip(tensor.legs, source_sector, strict=True)
        )
        sign = _permutation_sign(order, parities)
        blocks.append(sign * jnp.transpose(tensor.tensor.blocks[source_index], order))
    value = AbelianTensor(layout, tuple(blocks), precision=tensor.tensor.precision)
    return GradedTensor(value, output_legs)


class GradedContractionPlan(StrictModule):
    """Canonical mode-order reduction of one binary graded contraction."""

    left_tensor_id: str = eqx.field(static=True)
    right_tensor_id: str = eqx.field(static=True)
    left_permutation: tuple[int, ...] = eqx.field(static=True)
    right_permutation: tuple[int, ...] = eqx.field(static=True)
    output_permutation: tuple[int, ...] = eqx.field(static=True)
    abelian_plan: AbelianContractionPlan
    result_mode_labels: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left: GradedTensor,
        right: GradedTensor,
        contracted_modes: Sequence[str],
        mode_order: Sequence[str],
        /,
    ):
        if not isinstance(left, GradedTensor) or not isinstance(right, GradedTensor):
            raise TypeError("left and right must be GradedTensor values.")
        if left.grading_id != right.grading_id:
            raise ValueError("Graded contractions require the same grading.")
        contracted = tuple(str(label) for label in contracted_modes)
        global_order = tuple(str(label) for label in mode_order)
        if len(set(global_order)) != len(global_order):
            raise ValueError("Explicit global mode order must be unique.")
        left_labels = tuple(leg.mode_label for leg in left.legs)
        right_labels = tuple(leg.mode_label for leg in right.legs)
        if any(
            label not in left_labels or label not in right_labels for label in contracted
        ):
            raise ValueError("Each contracted mode must occur on both tensors.")
        all_labels = set(left_labels) | set(right_labels)
        if set(global_order) != all_labels:
            raise ValueError(
                "Global mode order must list exactly all participating modes."
            )
        contracted_order = tuple(label for label in global_order if label in contracted)
        left_free = tuple(
            label
            for label in global_order
            if label in left_labels and label not in contracted
        )
        right_free = tuple(
            label
            for label in global_order
            if label in right_labels and label not in contracted
        )
        left_perm = tuple(
            left_labels.index(label) for label in left_free + contracted_order
        )
        right_perm = tuple(
            right_labels.index(label) for label in contracted_order + right_free
        )
        left_normal = graded_permute(left, left_perm)
        right_normal = graded_permute(right, right_perm)
        contraction_count = len(contracted_order)
        abelian_plan = AbelianContractionPlan(
            left_normal.tensor.layout,
            right_normal.tensor.layout,
            tuple(range(len(left_free), len(left_free) + contraction_count)),
            tuple(range(contraction_count)),
        )
        intermediate_labels = left_free + right_free
        result_labels = tuple(label for label in global_order if label not in contracted)
        output_perm = tuple(intermediate_labels.index(label) for label in result_labels)
        self.left_tensor_id = left.graded_tensor_id
        self.right_tensor_id = right.graded_tensor_id
        self.left_permutation = left_perm
        self.right_permutation = right_perm
        self.output_permutation = output_perm
        self.abelian_plan = abelian_plan
        self.result_mode_labels = result_labels
        self.plan_id = canonical_fingerprint(
            {
                "kind": "graded-contraction-plan",
                "left": left.graded_tensor_id,
                "right": right.graded_tensor_id,
                "left_permutation": left_perm,
                "right_permutation": right_perm,
                "output_permutation": output_perm,
                "abelian_plan": abelian_plan.plan_id,
            }
        )


def prepare_graded_contraction(
    left: GradedTensor,
    right: GradedTensor,
    contracted_modes: Sequence[str],
    mode_order: Sequence[str],
    /,
) -> GradedContractionPlan:
    return GradedContractionPlan(left, right, contracted_modes, mode_order)


def contract_graded_tensors(
    plan: GradedContractionPlan,
    left: GradedTensor,
    right: GradedTensor,
    /,
) -> GradedTensor | Array:
    """Contract in canonical mode order, making closed contractions path invariant."""

    if not isinstance(plan, GradedContractionPlan):
        raise TypeError("plan must be GradedContractionPlan.")
    if (
        left.graded_tensor_id != plan.left_tensor_id
        or right.graded_tensor_id != plan.right_tensor_id
    ):
        raise ValueError("Graded tensors do not match the prepared plan.")
    left_normal = graded_permute(left, plan.left_permutation)
    right_normal = graded_permute(right, plan.right_permutation)
    result = contract_abelian_tensors(
        plan.abelian_plan, left_normal.tensor, right_normal.tensor
    )
    if not isinstance(result, AbelianTensor):
        return result
    intermediate_legs = tuple(
        leg for leg in left_normal.legs if leg.mode_label in plan.result_mode_labels
    ) + tuple(
        leg for leg in right_normal.legs if leg.mode_label in plan.result_mode_labels
    )
    graded = GradedTensor(result, intermediate_legs)
    return graded_permute(graded, plan.output_permutation)


def contract_graded_closed_network(
    tensors: Sequence[GradedTensor], mode_order: Sequence[str], /
) -> Array:
    """Evaluate a closed graded network by one canonical n-ary contraction."""

    values = tuple(tensors)
    if not values or any(not isinstance(value, GradedTensor) for value in values):
        raise TypeError("tensors must contain GradedTensor values.")
    if any(value.grading_id != values[0].grading_id for value in values):
        raise ValueError("Every closed-network tensor must share one grading.")
    ordered_modes = tuple(str(label) for label in mode_order)
    occurrences = tuple(
        (tensor_index, axis, leg)
        for tensor_index, tensor in enumerate(values)
        for axis, leg in enumerate(tensor.legs)
    )
    if set(ordered_modes) != {leg.mode_label for _, _, leg in occurrences}:
        raise ValueError("Closed-network mode order does not match tensor legs.")
    for mode in ordered_modes:
        legs = tuple(leg for _, _, leg in occurrences if leg.mode_label == mode)
        if len(legs) != 2 or not legs[0].leg.dual_compatible(legs[1].leg):
            raise ValueError("Each closed graded mode must occur on one dual leg pair.")
    tensor_order = tuple(
        sorted(
            range(len(values)),
            key=lambda index: (
                tuple(leg.mode_label for leg in values[index].legs),
                values[index].graded_tensor_id,
            ),
        )
    )
    ordered_values = tuple(values[index] for index in tensor_order)
    dtype = jnp.result_type(
        *(block for tensor in ordered_values for block in tensor.tensor.blocks)
    )
    result = jnp.zeros((), dtype=dtype)
    desired_occurrences = tuple(
        (tensor_index, axis)
        for mode in ordered_modes
        for tensor_index, tensor in enumerate(ordered_values)
        for axis, leg in enumerate(tensor.legs)
        if leg.mode_label == mode
    )
    concatenated_occurrences = tuple(
        (tensor_index, axis)
        for tensor_index, tensor in enumerate(ordered_values)
        for axis in range(len(tensor.legs))
    )
    permutation = tuple(
        concatenated_occurrences.index(item) for item in desired_occurrences
    )
    for block_indices in product(
        *(range(len(tensor.tensor.blocks)) for tensor in ordered_values)
    ):
        sectors = tuple(
            tensor.tensor.layout.sectors[index]
            for tensor, index in zip(ordered_values, block_indices, strict=True)
        )
        compatible = all(
            len(
                {
                    sectors[tensor_index][axis]
                    for tensor_index, tensor in enumerate(ordered_values)
                    for axis, leg in enumerate(tensor.legs)
                    if leg.mode_label == mode
                }
            )
            == 1
            for mode in ordered_modes
        )
        if not compatible:
            continue
        parities = tuple(
            leg.sector_parities[sectors[tensor_index][axis]]
            for tensor_index, tensor in enumerate(ordered_values)
            for axis, leg in enumerate(tensor.legs)
        )
        sign = _permutation_sign(permutation, parities)
        operands = []
        for tensor, index in zip(ordered_values, block_indices, strict=True):
            operands.extend(
                (
                    tensor.tensor.blocks[index],
                    [ordered_modes.index(leg.mode_label) for leg in tensor.legs],
                )
            )
        result = result + sign * oe.contract(*operands, [])
    return result


__all__ = [
    "FermionGrading",
    "GradedContractionPlan",
    "GradedLeg",
    "GradedTensor",
    "contract_graded_tensors",
    "contract_graded_closed_network",
    "graded_permute",
    "prepare_graded_contraction",
]
