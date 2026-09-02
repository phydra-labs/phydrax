#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import prod

import equinox as eqx
import opt_einsum as oe

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule
from ._topology import ContractionStructure


class ContractionScheduleStep(StrictModule):
    """One immutable SSA contraction instruction."""

    ordinal: int = eqx.field(static=True)
    input_value_ids: tuple[str, ...] = eqx.field(static=True)
    output_value_id: str = eqx.field(static=True)
    equation: str = eqx.field(static=True)
    output_labels: tuple[str, ...] = eqx.field(static=True)
    contracted_labels: tuple[str, ...] = eqx.field(static=True)
    live_value_ids_before: tuple[str, ...] = eqx.field(static=True)
    live_value_ids_after: tuple[str, ...] = eqx.field(static=True)
    output_elements: int = eqx.field(static=True)
    estimated_flops: int = eqx.field(static=True)
    live_elements_before: int = eqx.field(static=True)
    allocation_peak_elements: int = eqx.field(static=True)
    live_elements_after: int = eqx.field(static=True)
    step_id: str = eqx.field(static=True)


class ContractionSchedule(StrictModule):
    """Inspectable deterministic SSA schedule with complete live-set accounting."""

    structure_id: str = eqx.field(static=True)
    initial_value_ids: tuple[str, ...] = eqx.field(static=True)
    steps: tuple[ContractionScheduleStep, ...] = eqx.field(static=True)
    output_value_id: str = eqx.field(static=True)
    path: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    initial_live_elements: int = eqx.field(static=True)
    peak_live_elements: int = eqx.field(static=True)
    peak_live_bytes: int = eqx.field(static=True)
    total_estimated_flops: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)


class ReverseScheduleEvidence(StrictModule):
    forward_schedule_id: str = eqx.field(static=True)
    reverse_schedule_id: str = eqx.field(static=True)
    rematerialization: str = eqx.field(static=True)
    retained_forward_bytes: int = eqx.field(static=True)
    reverse_peak_bytes: int = eqx.field(static=True)
    exact_derivative: bool = eqx.field(static=True)


def planner_search_state_bound(operand_count: int, optimizer: str, /) -> int:
    """Return a finite upper bound on pair candidates inspected by a planner."""

    count = int(operand_count)
    if count < 1:
        raise ValueError("operand_count must be positive.")
    if optimizer == "greedy":
        return sum(size * (size - 1) // 2 for size in range(2, count + 1))
    if optimizer == "optimal":
        states = 1
        for size in range(count, 1, -1):
            states *= size * (size - 1) // 2
        return states
    raise ValueError("optimizer must be 'greedy' or 'optimal'.")


def _symbols(structure: ContractionStructure) -> dict[str, str]:
    return {label: oe.get_symbol(index) for index, label in enumerate(structure.labels)}


def _elements(labels: Sequence[str], dimensions: dict[str, int], /) -> int:
    return prod(dimensions[label] for label in labels)


def build_contraction_schedule(
    structure: ContractionStructure,
    path: Sequence[Sequence[int]],
    /,
    *,
    dtype: str,
) -> ContractionSchedule:
    """Compile a contraction path into stable SSA values and liveness evidence."""

    if not isinstance(structure, ContractionStructure):
        raise TypeError("structure must be ContractionStructure.")
    path_ = tuple(tuple(int(index) for index in step) for step in path)
    if any(not step or len(set(step)) != len(step) or min(step) < 0 for step in path_):
        raise ValueError(
            "Every contraction path step needs distinct non-negative indices."
        )
    dimensions = dict(zip(structure.labels, structure.dimensions, strict=True))
    symbols = _symbols(structure)
    live: list[tuple[str, tuple[str, ...]]] = [
        (operand.operand_id, tuple(leg.label for leg in operand.legs))
        for operand in structure.operands
    ]
    live_sizes: dict[str, int] = {
        value_id: _elements(labels, dimensions) for value_id, labels in live
    }
    initial_ids = tuple(value_id for value_id, _ in live)
    initial_live = sum(live_sizes.values())
    current_live = initial_live
    peak_live = initial_live
    total_flops = 0
    steps: list[ContractionScheduleStep] = []

    for ordinal, positions in enumerate(path_):
        if max(positions) >= len(live):
            raise ValueError("A contraction path step indexes a retired SSA value.")
        selected = tuple(live[position] for position in positions)
        selected_positions = set(positions)
        remaining = tuple(
            value
            for position, value in enumerate(live)
            if position not in selected_positions
        )
        remaining_counts: Counter[str] = Counter(
            label for _, labels in remaining for label in labels
        )
        ordered_selected: list[str] = []
        for _, labels in selected:
            for label in labels:
                if label not in ordered_selected:
                    ordered_selected.append(label)
        if not remaining:
            output_labels = structure.outputs
        else:
            output_labels = tuple(
                label
                for label in ordered_selected
                if label in structure.outputs or remaining_counts[label] > 0
            )
        selected_counts: Counter[str] = Counter(
            label for _, labels in selected for label in labels
        )
        contracted = tuple(
            label
            for label in ordered_selected
            if label not in output_labels and selected_counts[label] > 0
        )
        inputs = ",".join(
            "".join(symbols[label] for label in labels) for _, labels in selected
        )
        output = "".join(symbols[label] for label in output_labels)
        equation = f"{inputs}->{output}"
        output_id = canonical_fingerprint(
            {
                "kind": "contraction-ssa-value",
                "structure": structure.structure_id,
                "ordinal": ordinal,
                "inputs": tuple(value_id for value_id, _ in selected),
                "equation": equation,
            }
        )
        output_elements = _elements(output_labels, dimensions)
        union_labels = tuple(dict.fromkeys(ordered_selected))
        flops = _elements(union_labels, dimensions) * max(1, len(selected) - 1)
        allocation_peak = current_live + output_elements
        after = current_live - sum(live_sizes[value_id] for value_id, _ in selected)
        after += output_elements
        step_payload = {
            "ordinal": ordinal,
            "inputs": tuple(value_id for value_id, _ in selected),
            "output": output_id,
            "equation": equation,
            "output_labels": output_labels,
        }
        step = ContractionScheduleStep(
            ordinal,
            tuple(value_id for value_id, _ in selected),
            output_id,
            equation,
            output_labels,
            contracted,
            tuple(value_id for value_id, _ in live),
            tuple(value_id for value_id, _ in remaining) + (output_id,),
            output_elements,
            flops,
            current_live,
            allocation_peak,
            after,
            canonical_fingerprint({"kind": "contraction-ssa-step", **step_payload}),
        )
        steps.append(step)
        total_flops += flops
        peak_live = max(peak_live, allocation_peak)
        for position in sorted(positions, reverse=True):
            live.pop(position)
        live.append((output_id, output_labels))
        for value_id, _ in selected:
            live_sizes.pop(value_id)
        live_sizes[output_id] = output_elements
        current_live = after

    if len(live) != 1:
        raise ValueError("The contraction path did not reduce to one SSA value.")
    final_id, final_labels = live[0]
    if final_labels != structure.outputs:
        ordinal = len(steps)
        equation = (
            "".join(symbols[label] for label in final_labels)
            + "->"
            + "".join(symbols[label] for label in structure.outputs)
        )
        output_id = canonical_fingerprint(
            {
                "kind": "contraction-ssa-value",
                "structure": structure.structure_id,
                "ordinal": ordinal,
                "inputs": (final_id,),
                "equation": equation,
            }
        )
        output_elements = _elements(structure.outputs, dimensions)
        union_labels = tuple(dict.fromkeys(final_labels))
        flops = _elements(union_labels, dimensions)
        allocation_peak = current_live + output_elements
        step = ContractionScheduleStep(
            ordinal,
            (final_id,),
            output_id,
            equation,
            structure.outputs,
            tuple(label for label in union_labels if label not in structure.outputs),
            (final_id,),
            (output_id,),
            output_elements,
            flops,
            current_live,
            allocation_peak,
            output_elements,
            canonical_fingerprint(
                {
                    "kind": "contraction-ssa-step",
                    "ordinal": ordinal,
                    "inputs": (final_id,),
                    "output": output_id,
                    "equation": equation,
                }
            ),
        )
        steps.append(step)
        total_flops += flops
        peak_live = max(peak_live, allocation_peak)
        final_id = output_id

    dtype_ = str(dtype)
    schedule_id = canonical_fingerprint(
        {
            "kind": "contraction-ssa-schedule",
            "structure": structure.structure_id,
            "path": path_,
            "dtype": dtype_,
            "steps": tuple(step.step_id for step in steps),
        }
    )
    return ContractionSchedule(
        structure.structure_id,
        initial_ids,
        tuple(steps),
        final_id,
        path_,
        dtype_,
        initial_live,
        peak_live,
        peak_live * precision_itemsize(dtype_),
        total_flops,
        True,
        "exact ordinary product-sum contraction",
        schedule_id,
    )


def reverse_schedule_evidence(
    schedule: ContractionSchedule,
    /,
    *,
    rematerialization: str,
    operand_bytes: int,
    output_bytes: int,
) -> ReverseScheduleEvidence:
    if rematerialization not in ("store", "rematerialize"):
        raise ValueError("rematerialization must be 'store' or 'rematerialize'.")
    retained = (
        schedule.peak_live_bytes
        if rematerialization == "store"
        else int(operand_bytes) + int(output_bytes)
    )
    reverse_peak = retained + schedule.peak_live_bytes
    reverse_id = canonical_fingerprint(
        {
            "kind": "reverse-contraction-schedule",
            "forward": schedule.schedule_id,
            "rematerialization": rematerialization,
            "retained": retained,
        }
    )
    return ReverseScheduleEvidence(
        schedule.schedule_id,
        reverse_id,
        rematerialization,
        retained,
        reverse_peak,
        True,
    )


__all__ = [
    "ContractionSchedule",
    "ContractionScheduleStep",
    "ReverseScheduleEvidence",
    "build_contraction_schedule",
    "planner_search_state_bound",
    "reverse_schedule_evidence",
]
