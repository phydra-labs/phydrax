#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stable compartment, branch, and cell morphology planning."""

from __future__ import annotations

from math import isfinite, pi
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _positive(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


class CompartmentSpec(StrictModule, NonTrainableState):
    """Host plan for one cylindrical isopotential compartment."""

    compartment_id: str = eqx.field(static=True)
    parent_id: str | None = eqx.field(static=True)
    length_um: float = eqx.field(static=True)
    diameter_um: float = eqx.field(static=True)
    capacitance_density_uF_cm2: float = eqx.field(static=True)
    axial_resistivity_ohm_cm: float = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        compartment_id: str,
        parent_id: str | None,
        length_um: float,
        diameter_um: float,
        /,
        *,
        capacitance_density_uF_cm2: float = 1.0,
        axial_resistivity_ohm_cm: float = 100.0,
    ):
        compartment = _identifier(compartment_id, "compartment_id")
        if parent_id is not None:
            parent = _identifier(parent_id, "parent_id")
            if parent == compartment:
                raise ValueError("A compartment cannot be its own parent.")
        else:
            parent = None
        length = _positive(length_um, "length_um")
        diameter = _positive(diameter_um, "diameter_um")
        capacitance = _positive(capacitance_density_uF_cm2, "capacitance_density_uF_cm2")
        resistivity = _positive(axial_resistivity_ohm_cm, "axial_resistivity_ohm_cm")
        self.compartment_id = compartment
        self.parent_id = parent
        self.length_um = length
        self.diameter_um = diameter
        self.capacitance_density_uF_cm2 = capacitance
        self.axial_resistivity_ohm_cm = resistivity
        self.spec_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-compartment-v1",
                "compartment_id": compartment,
                "parent_id": parent,
                "length_um": length,
                "diameter_um": diameter,
                "capacitance_density_uF_cm2": capacitance,
                "axial_resistivity_ohm_cm": resistivity,
            }
        )


class BranchSpec(StrictModule, NonTrainableState):
    """Stable ordered path through one or more morphology compartments."""

    branch_id: str = eqx.field(static=True)
    compartment_ids: tuple[str, ...] = eqx.field(static=True)
    branch_spec_id: str = eqx.field(static=True)

    def __init__(self, branch_id: str, compartment_ids: Sequence[str], /):
        identifier = _identifier(branch_id, "branch_id")
        values = tuple(compartment_ids)
        if not values:
            raise ValueError("compartment_ids must contain at least one identifier.")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("Every compartment identifier must be a non-empty string.")
        if len(set(values)) != len(values):
            raise ValueError("A branch cannot repeat a compartment identifier.")
        self.branch_id = identifier
        self.compartment_ids = values
        self.branch_spec_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-branch-v1",
                "branch_id": identifier,
                "compartments": list(values),
            }
        )


class CellMorphologyPlan(StrictModule, NonTrainableState):
    """Validated stable-ID tree plan with optional named branch paths."""

    cell_id: str = eqx.field(static=True)
    compartments: tuple[CompartmentSpec, ...]
    branches: tuple[BranchSpec, ...]
    root_id: str = eqx.field(static=True)
    compartment_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_id: str,
        compartments: Sequence[CompartmentSpec],
        /,
        *,
        branches: Sequence[BranchSpec] = (),
    ):
        cell = _identifier(cell_id, "cell_id")
        specs = tuple(compartments)
        if not specs:
            raise ValueError("A cell morphology requires at least one compartment.")
        if any(not isinstance(spec, CompartmentSpec) for spec in specs):
            raise TypeError("compartments must contain only CompartmentSpec values.")
        identifiers = tuple(spec.compartment_id for spec in specs)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Compartment identifiers must be unique within a cell.")
        identifier_set = set(identifiers)
        roots = tuple(spec.compartment_id for spec in specs if spec.parent_id is None)
        if len(roots) != 1:
            raise ValueError("A cell morphology must contain exactly one root.")
        for spec in specs:
            if spec.parent_id is not None and spec.parent_id not in identifier_set:
                raise ValueError(
                    f"Compartment {spec.compartment_id!r} references missing parent "
                    f"{spec.parent_id!r}."
                )
        by_id = {spec.compartment_id: spec for spec in specs}
        for identifier in identifiers:
            visited: set[str] = set()
            current: str | None = identifier
            while current is not None:
                if current in visited:
                    raise ValueError("Compartment parent relations must be acyclic.")
                visited.add(current)
                current = by_id[current].parent_id
            if roots[0] not in visited:
                raise ValueError("Every compartment must be connected to the root.")
        branch_values = tuple(branches)
        if any(not isinstance(branch, BranchSpec) for branch in branch_values):
            raise TypeError("branches must contain only BranchSpec values.")
        branch_ids = tuple(branch.branch_id for branch in branch_values)
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("Branch identifiers must be unique within a cell.")
        for branch in branch_values:
            if any(value not in identifier_set for value in branch.compartment_ids):
                raise ValueError(
                    f"Branch {branch.branch_id!r} references an unknown compartment."
                )
            for left, right in zip(
                branch.compartment_ids[:-1], branch.compartment_ids[1:], strict=True
            ):
                adjacent = (
                    by_id[right].parent_id == left or by_id[left].parent_id == right
                )
                if not adjacent:
                    raise ValueError(
                        f"Branch {branch.branch_id!r} is not a contiguous tree path."
                    )
        self.cell_id = cell
        self.compartments = specs
        self.branches = branch_values
        self.root_id = roots[0]
        self.compartment_ids = identifiers
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-cell-morphology-v1",
                "cell_id": cell,
                "compartments": [spec.spec_id for spec in specs],
                "branches": [branch.branch_spec_id for branch in branch_values],
            }
        )

    @property
    def compartment_count(self) -> int:
        return len(self.compartments)

    def compartment_index(self, compartment_id: str, /) -> int:
        """Resolve a stable identifier at host planning time."""
        identifier = _identifier(compartment_id, "compartment_id")
        if identifier not in self.compartment_ids:
            raise ValueError(f"Unknown compartment identifier {identifier!r}.")
        return self.compartment_ids.index(identifier)

    def prepare(self) -> PreparedCellMorphology:
        """Materialize geometry, axial conductance, and elimination schedules."""
        return prepare_cell_morphology(self)


class PreparedCellMorphology(StrictModule, NonTrainableState):
    """Fixed-shape device morphology and reusable tree solve structure."""

    plan: CellMorphologyPlan
    parent_index: Array
    membrane_area_um2: Array
    capacitance_nF: Array
    edge_conductance_uS: Array
    axial_laplacian_uS: Array
    elimination_order: Array
    back_substitution_order: Array
    root_index: int = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CellMorphologyPlan,
        parent_index: Array,
        membrane_area_um2: Array,
        capacitance_nF: Array,
        edge_conductance_uS: Array,
        axial_laplacian_uS: Array,
        elimination_order: Array,
        back_substitution_order: Array,
        /,
        *,
        root_index: int,
        runtime_id: str,
    ):
        self.plan = plan
        self.parent_index = parent_index
        self.membrane_area_um2 = membrane_area_um2
        self.capacitance_nF = capacitance_nF
        self.edge_conductance_uS = edge_conductance_uS
        self.axial_laplacian_uS = axial_laplacian_uS
        self.elimination_order = elimination_order
        self.back_substitution_order = back_substitution_order
        self.root_index = root_index
        self.runtime_id = runtime_id


def _postorder(parent: np.ndarray, root: int, /) -> tuple[int, ...]:
    children = tuple(
        tuple(np.flatnonzero(parent == index).tolist()) for index in range(parent.size)
    )
    result: list[int] = []
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        index, expanded = stack.pop()
        if expanded:
            result.append(index)
            continue
        stack.append((index, True))
        stack.extend((child, False) for child in reversed(children[index]))
    return tuple(result)


def prepare_cell_morphology(plan: CellMorphologyPlan, /) -> PreparedCellMorphology:
    """Prepare one morphology without embedding host mutation in compiled paths."""
    if not isinstance(plan, CellMorphologyPlan):
        raise TypeError("plan must be a CellMorphologyPlan.")
    count = plan.compartment_count
    by_id = {value: index for index, value in enumerate(plan.compartment_ids)}
    parent = np.asarray(
        [
            -1 if spec.parent_id is None else by_id[spec.parent_id]
            for spec in plan.compartments
        ],
        dtype=np.int32,
    )
    root = int(np.flatnonzero(parent < 0)[0])
    length = np.asarray([spec.length_um for spec in plan.compartments], dtype=float)
    diameter = np.asarray([spec.diameter_um for spec in plan.compartments], dtype=float)
    area = pi * diameter * length
    capacitance_density = np.asarray(
        [spec.capacitance_density_uF_cm2 for spec in plan.compartments], dtype=float
    )
    capacitance = capacitance_density * area * 1.0e-5
    resistivity = np.asarray(
        [spec.axial_resistivity_ohm_cm for spec in plan.compartments], dtype=float
    )
    half_resistance = (
        resistivity * (0.5 * length * 1.0e-4) / (pi * (0.5 * diameter * 1.0e-4) ** 2)
    )
    edge_conductance = np.zeros((count,), dtype=float)
    laplacian = np.zeros((count, count), dtype=float)
    for child, parent_index in enumerate(parent.tolist()):
        if parent_index < 0:
            continue
        resistance = half_resistance[child] + half_resistance[parent_index]
        conductance = 1.0e6 / resistance
        edge_conductance[child] = conductance
        laplacian[child, child] += conductance
        laplacian[parent_index, parent_index] += conductance
        laplacian[child, parent_index] -= conductance
        laplacian[parent_index, child] -= conductance
    order = tuple(index for index in _postorder(parent, root) if index != root)
    runtime_id = canonical_fingerprint(
        {
            "kind": "prepared-electrophysiology-morphology-v1",
            "plan": plan.plan_id,
            "parent_index": parent.tolist(),
            "elimination_order": list(order),
        }
    )
    dtype = jnp.asarray(0.0).dtype
    return PreparedCellMorphology(
        plan,
        jnp.asarray(parent),
        jnp.asarray(area, dtype=dtype),
        jnp.asarray(capacitance, dtype=dtype),
        jnp.asarray(edge_conductance, dtype=dtype),
        jnp.asarray(laplacian, dtype=dtype),
        jnp.asarray(order, dtype=jnp.int32),
        jnp.asarray(order[::-1], dtype=jnp.int32),
        root_index=root,
        runtime_id=runtime_id,
    )


__all__ = [
    "BranchSpec",
    "CellMorphologyPlan",
    "CompartmentSpec",
    "PreparedCellMorphology",
    "prepare_cell_morphology",
]
