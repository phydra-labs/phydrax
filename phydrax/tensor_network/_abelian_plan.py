#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._array_archive import array_payload_digest
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._abelian import AbelianTensor, AbelianTensorLayout
from ._abelian_core import AbelianMatrixProductState


class AbelianContractionPlan(StrictModule):
    """Static block routes for one charge-preserving tensor contraction."""

    left_layout_id: str = eqx.field(static=True)
    right_layout_id: str = eqx.field(static=True)
    left_axes: tuple[int, ...] = eqx.field(static=True)
    right_axes: tuple[int, ...] = eqx.field(static=True)
    left_free_axes: tuple[int, ...] = eqx.field(static=True)
    right_free_axes: tuple[int, ...] = eqx.field(static=True)
    routes: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    result_layout: AbelianTensorLayout | None
    scalar_result: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_layout: AbelianTensorLayout,
        right_layout: AbelianTensorLayout,
        left_axes: Sequence[int],
        right_axes: Sequence[int],
        /,
    ):
        if not isinstance(left_layout, AbelianTensorLayout) or not isinstance(
            right_layout, AbelianTensorLayout
        ):
            raise TypeError("Contraction layouts must be AbelianTensorLayout values.")
        if left_layout.legs[0].group.group_id != right_layout.legs[0].group.group_id:
            raise ValueError("Abelian contraction groups must match.")
        left_rank = len(left_layout.legs)
        right_rank = len(right_layout.legs)
        left_axes_ = tuple(int(axis) % left_rank for axis in left_axes)
        right_axes_ = tuple(int(axis) % right_rank for axis in right_axes)
        if (
            not left_axes_
            or len(left_axes_) != len(right_axes_)
            or len(set(left_axes_)) != len(left_axes_)
            or len(set(right_axes_)) != len(right_axes_)
        ):
            raise ValueError("Contraction axes must be nonempty, unique, and aligned.")
        for left_axis, right_axis in zip(left_axes_, right_axes_, strict=True):
            if not left_layout.legs[left_axis].dual_compatible(
                right_layout.legs[right_axis]
            ):
                raise ValueError("Contracted Abelian legs must be dual compatible.")
        left_free = tuple(axis for axis in range(left_rank) if axis not in left_axes_)
        right_free = tuple(axis for axis in range(right_rank) if axis not in right_axes_)
        output_legs = tuple(left_layout.legs[axis] for axis in left_free) + tuple(
            right_layout.legs[axis] for axis in right_free
        )
        group = left_layout.legs[0].group
        total_charge = group.add(left_layout.total_charge, right_layout.total_charge)
        result_layout = (
            AbelianTensorLayout(output_legs, total_charge=total_charge)
            if output_legs
            else None
        )
        routes = []
        for left_index, left_sector in enumerate(left_layout.sectors):
            for right_index, right_sector in enumerate(right_layout.sectors):
                if any(
                    left_sector[left_axis] != right_sector[right_axis]
                    for left_axis, right_axis in zip(left_axes_, right_axes_, strict=True)
                ):
                    continue
                output_sector = tuple(left_sector[axis] for axis in left_free) + tuple(
                    right_sector[axis] for axis in right_free
                )
                if result_layout is None:
                    if total_charge != group.zero:
                        continue
                    result_index = 0
                else:
                    if output_sector not in result_layout.sectors:
                        continue
                    result_index = result_layout.sectors.index(output_sector)
                routes.append((left_index, right_index, result_index))
        self.left_layout_id = left_layout.layout_id
        self.right_layout_id = right_layout.layout_id
        self.left_axes = left_axes_
        self.right_axes = right_axes_
        self.left_free_axes = left_free
        self.right_free_axes = right_free
        self.routes = tuple(routes)
        self.result_layout = result_layout
        self.scalar_result = not output_legs
        self.plan_id = canonical_fingerprint(
            {
                "kind": "abelian-contraction-plan",
                "left": left_layout.layout_id,
                "right": right_layout.layout_id,
                "left_axes": left_axes_,
                "right_axes": right_axes_,
                "routes": self.routes,
                "result": None if result_layout is None else result_layout.layout_id,
            }
        )


def prepare_abelian_contraction(
    left_layout: AbelianTensorLayout,
    right_layout: AbelianTensorLayout,
    left_axes: Sequence[int],
    right_axes: Sequence[int],
    /,
) -> AbelianContractionPlan:
    return AbelianContractionPlan(left_layout, right_layout, left_axes, right_axes)


def contract_abelian_tensors(
    plan: AbelianContractionPlan,
    left: AbelianTensor,
    right: AbelianTensor,
    /,
) -> AbelianTensor | Array:
    """Execute only statically prepared block routes; no dense local tensor exists."""

    if not isinstance(plan, AbelianContractionPlan):
        raise TypeError("plan must be AbelianContractionPlan.")
    if not isinstance(left, AbelianTensor) or not isinstance(right, AbelianTensor):
        raise TypeError("left and right must be AbelianTensor values.")
    if (
        left.layout.layout_id != plan.left_layout_id
        or right.layout.layout_id != plan.right_layout_id
    ):
        raise ValueError("Abelian tensors do not match the prepared contraction plan.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("Abelian contraction precision policies must match.")
    precision = left.precision
    left_labels = list(range(len(left.layout.legs)))
    right_labels = list(
        range(len(left.layout.legs), len(left.layout.legs) + len(right.layout.legs))
    )
    for left_axis, right_axis in zip(plan.left_axes, plan.right_axes, strict=True):
        right_labels[right_axis] = left_labels[left_axis]
    output_labels = [left_labels[axis] for axis in plan.left_free_axes] + [
        right_labels[axis] for axis in plan.right_free_axes
    ]
    contributions = []
    if plan.scalar_result:
        for left_index, right_index, _ in plan.routes:
            contributions.append(
                ein.contract(
                    precision.contraction(left.blocks[left_index]),
                    left_labels,
                    precision.contraction(right.blocks[right_index]),
                    right_labels,
                    output_labels,
                )
            )
        dtype = (
            jnp.result_type(left.blocks[0], right.blocks[0])
            if left.blocks and right.blocks
            else jnp.complex128
        )
        result = jnp.zeros((), dtype=dtype)
        for contribution in contributions:
            result = result + contribution
        return precision.output(result)
    if plan.result_layout is None:
        raise RuntimeError("Nonscalar Abelian contraction plan has no result layout.")
    dtype = (
        jnp.result_type(left.blocks[0], right.blocks[0])
        if left.blocks and right.blocks
        else jnp.complex128
    )
    blocks = [jnp.zeros(shape, dtype=dtype) for shape in plan.result_layout.block_shapes]
    for left_index, right_index, result_index in plan.routes:
        contribution = ein.contract(
            precision.contraction(left.blocks[left_index]),
            left_labels,
            precision.contraction(right.blocks[right_index]),
            right_labels,
            output_labels,
        )
        blocks[result_index] = blocks[result_index] + contribution
    return AbelianTensor(plan.result_layout, tuple(blocks), precision=precision)


class AbelianCanonicalizationPlan(StrictModule):
    """Static charge routes used by repeated canonical sweeps of one allocation."""

    structure_id: str = eqx.field(static=True)
    left_routes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    right_routes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, state: AbelianMatrixProductState, /):
        if not isinstance(state, AbelianMatrixProductState):
            raise TypeError("state must be AbelianMatrixProductState.")
        left_routes = []
        right_routes = []
        for tensor in state.tensors:
            left_leg = tensor.layout.legs[0]
            right_leg = tensor.layout.legs[2]
            left_routes.append(
                tuple(
                    tuple(
                        index
                        for index, sector in enumerate(tensor.layout.sectors)
                        if sector[0] == charge
                    )
                    for charge in range(len(left_leg.charges))
                )
            )
            right_routes.append(
                tuple(
                    tuple(
                        index
                        for index, sector in enumerate(tensor.layout.sectors)
                        if sector[2] == charge
                    )
                    for charge in range(len(right_leg.charges))
                )
            )
        self.structure_id = state.structure_id
        self.left_routes = tuple(left_routes)
        self.right_routes = tuple(right_routes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "abelian-canonicalization-plan",
                "structure": state.structure_id,
                "left_routes": self.left_routes,
                "right_routes": self.right_routes,
            }
        )


def prepare_abelian_canonicalization(
    state: AbelianMatrixProductState, /
) -> AbelianCanonicalizationPlan:
    return AbelianCanonicalizationPlan(state)


def execute_abelian_canonicalization(
    plan: AbelianCanonicalizationPlan,
    state: AbelianMatrixProductState,
    /,
    *,
    center: int,
    normalize: bool = True,
) -> AbelianMatrixProductState:
    if not isinstance(plan, AbelianCanonicalizationPlan):
        raise TypeError("plan must be AbelianCanonicalizationPlan.")
    if state.structure_id != plan.structure_id:
        raise ValueError("State allocation does not match canonicalization plan.")
    from ._abelian_core import canonicalize_abelian_mps

    return canonicalize_abelian_mps(
        state,
        center=center,
        normalize=normalize,
        prepared_routes=(plan.left_routes, plan.right_routes),
    )


class AbelianTwoSiteGatePlan(StrictModule):
    """Prepared gate value and static charge routes for one fixed MPS bond."""

    structure_id: str = eqx.field(static=True)
    left_site: int = eqx.field(static=True)
    gate: Array
    routes: tuple[
        tuple[
            tuple[tuple[int, int, int], ...],
            tuple[tuple[int, int, int], ...],
        ],
        ...,
    ] = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    protected_charges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: AbelianMatrixProductState,
        left_site: int,
        gate: ArrayLike,
        /,
        *,
        maximum_bond_dimension: int,
        normalize: bool = True,
        protected_charges: Sequence[Sequence[int]] = (),
    ):
        if not isinstance(state, AbelianMatrixProductState):
            raise TypeError("state must be AbelianMatrixProductState.")
        site = int(left_site)
        if not 0 <= site < state.site_count - 1:
            raise ValueError("Prepared gate site is outside the Abelian MPS.")
        capacity = int(maximum_bond_dimension)
        if capacity < 1:
            raise ValueError("maximum_bond_dimension must be positive.")
        value = jnp.asarray(gate)
        left_tensor, right_tensor = state.tensors[site : site + 2]
        left_physical = left_tensor.layout.legs[1]
        right_physical = right_tensor.layout.legs[1]
        expected = (
            left_physical.size,
            right_physical.size,
            left_physical.size,
            right_physical.size,
        )
        if value.shape != expected:
            raise ValueError("Prepared Abelian gate shape is invalid.")
        middle = left_tensor.layout.legs[2]
        left_virtual = left_tensor.layout.legs[0]
        right_virtual = right_tensor.layout.legs[2]
        group = middle.group
        routes = []
        for middle_charge in middle.charges:
            row_routes = tuple(
                (
                    left_sector,
                    physical_sector,
                    left_virtual.capacities[left_sector]
                    * left_physical.capacities[physical_sector],
                )
                for left_sector, left_charge in enumerate(left_virtual.charges)
                for physical_sector, physical_charge in enumerate(left_physical.charges)
                if group.add(left_charge, physical_charge) == middle_charge
            )
            column_routes = tuple(
                (
                    physical_sector,
                    right_sector,
                    right_physical.capacities[physical_sector]
                    * right_virtual.capacities[right_sector],
                )
                for physical_sector, physical_charge in enumerate(right_physical.charges)
                for right_sector, right_charge in enumerate(right_virtual.charges)
                if group.add(middle_charge, physical_charge) == right_charge
            )
            routes.append((row_routes, column_routes))
        protected = tuple(group.normalize(charge) for charge in protected_charges)
        self.structure_id = state.structure_id
        self.left_site = site
        self.gate = value
        self.routes = tuple(routes)
        self.maximum_bond_dimension = capacity
        self.normalize = bool(normalize)
        self.protected_charges = protected
        self.plan_id = canonical_fingerprint(
            {
                "kind": "abelian-two-site-gate-plan",
                "structure": state.structure_id,
                "left_site": site,
                "gate": array_payload_digest(value),
                "routes": self.routes,
                "maximum_bond_dimension": capacity,
                "normalize": self.normalize,
                "protected_charges": protected,
            }
        )


def prepare_abelian_two_site_gate(
    state: AbelianMatrixProductState,
    left_site: int,
    gate: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
    normalize: bool = True,
    protected_charges: Sequence[Sequence[int]] = (),
) -> AbelianTwoSiteGatePlan:
    return AbelianTwoSiteGatePlan(
        state,
        left_site,
        gate,
        maximum_bond_dimension=maximum_bond_dimension,
        normalize=normalize,
        protected_charges=protected_charges,
    )


def execute_abelian_two_site_gate(
    plan: AbelianTwoSiteGatePlan,
    state: AbelianMatrixProductState,
    /,
):
    if not isinstance(plan, AbelianTwoSiteGatePlan):
        raise TypeError("plan must be AbelianTwoSiteGatePlan.")
    if state.structure_id != plan.structure_id:
        raise ValueError("State allocation does not match the prepared gate plan.")
    from ._abelian_evolution import apply_abelian_two_site_gate

    return apply_abelian_two_site_gate(
        state,
        plan.left_site,
        plan.gate,
        maximum_bond_dimension=plan.maximum_bond_dimension,
        normalize=plan.normalize,
        protected_charges=plan.protected_charges,
        prepared_routes=plan.routes,
    )


class AbelianProgramInstruction(StrictModule):
    """One bounded nearest-neighbor gate instruction."""

    left_site: int = eqx.field(static=True)
    gate: Array
    maximum_bond_dimension: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    protected_charges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    instruction_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_site: int,
        gate: ArrayLike,
        /,
        *,
        maximum_bond_dimension: int,
        normalize: bool = False,
        protected_charges: Sequence[Sequence[int]] = (),
    ):
        capacity = int(maximum_bond_dimension)
        if capacity < 1:
            raise ValueError("maximum_bond_dimension must be positive.")
        value = jnp.asarray(gate)
        if value.ndim != 4:
            raise ValueError("Abelian program gates must have four tensor axes.")
        protected = tuple(
            tuple(int(item) for item in charge) for charge in protected_charges
        )
        self.left_site = int(left_site)
        self.gate = value
        self.maximum_bond_dimension = capacity
        self.normalize = bool(normalize)
        self.protected_charges = protected
        self.instruction_id = canonical_fingerprint(
            {
                "kind": "abelian-program-instruction",
                "left_site": self.left_site,
                "shape": value.shape,
                "dtype": str(value.dtype),
                "gate": array_payload_digest(value),
                "maximum_bond_dimension": capacity,
                "normalize": self.normalize,
                "protected_charges": protected,
            }
        )


class AbelianProgram(StrictModule):
    """Finite, statically sized sequence of Abelian gate instructions."""

    instructions: tuple[AbelianProgramInstruction, ...]
    program_id: str = eqx.field(static=True)

    def __init__(self, instructions: Sequence[AbelianProgramInstruction], /):
        values = tuple(instructions)
        if any(not isinstance(item, AbelianProgramInstruction) for item in values):
            raise TypeError("instructions must contain AbelianProgramInstruction values.")
        self.instructions = values
        self.program_id = canonical_fingerprint(
            {
                "kind": "abelian-program",
                "instructions": tuple(item.instruction_id for item in values),
            }
        )


class AbelianProgramEvidence(StrictModule):
    discarded_weights: Array
    charge_drifts: Array
    protected_sector_failures: Array
    valid: Array
    program_id: str = eqx.field(static=True)


def execute_abelian_program(
    program: AbelianProgram,
    state: AbelianMatrixProductState,
    /,
) -> tuple[AbelianMatrixProductState, AbelianProgramEvidence]:
    """Execute every instruction and return explicit truncation/protection evidence."""

    if not isinstance(program, AbelianProgram):
        raise TypeError("program must be AbelianProgram.")
    if not isinstance(state, AbelianMatrixProductState):
        raise TypeError("state must be AbelianMatrixProductState.")
    from ._abelian_evolution import apply_abelian_two_site_gate

    current = state
    discarded = []
    drift = []
    failures = []
    for instruction in program.instructions:
        before_charge = current.total_charge
        current, evidence = apply_abelian_two_site_gate(
            current,
            instruction.left_site,
            instruction.gate,
            maximum_bond_dimension=instruction.maximum_bond_dimension,
            normalize=instruction.normalize,
            protected_charges=instruction.protected_charges,
        )
        discarded.append(evidence.discarded_weight)
        drift.append(
            jnp.asarray(
                0.0 if current.total_charge == before_charge else 1.0,
                dtype=evidence.discarded_weight.dtype,
            )
        )
        failures.append(~evidence.protected_sectors_satisfied)
    real_dtype = current.norm().dtype
    discarded_values = (
        jnp.stack(discarded) if discarded else jnp.zeros((0,), dtype=real_dtype)
    )
    drift_values = jnp.stack(drift) if drift else jnp.zeros((0,), dtype=real_dtype)
    failure_values = jnp.stack(failures) if failures else jnp.zeros((0,), dtype=bool)
    valid = (
        jnp.all(jnp.isfinite(discarded_values))
        & jnp.all(drift_values == 0)
        & ~jnp.any(failure_values)
    )
    return current, AbelianProgramEvidence(
        discarded_values,
        drift_values,
        failure_values,
        valid,
        program.program_id,
    )


__all__ = [
    "AbelianCanonicalizationPlan",
    "AbelianTwoSiteGatePlan",
    "AbelianContractionPlan",
    "AbelianProgram",
    "AbelianProgramEvidence",
    "AbelianProgramInstruction",
    "contract_abelian_tensors",
    "execute_abelian_canonicalization",
    "execute_abelian_program",
    "execute_abelian_two_site_gate",
    "prepare_abelian_canonicalization",
    "prepare_abelian_contraction",
    "prepare_abelian_two_site_gate",
]
