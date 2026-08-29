#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization import PolygonalConnectivity
from ...discretization.fem import FiniteElementDiscretization
from ._ir import FieldSlot, FiniteElementActionIR, LocalActionIR, RegionIR
from ._kernels import KernelBinding, KernelTable
from ._worksets import CompiledWorkset, WorksetProgram, WorksetSignature


def _domain_for_action(action, discretization: FiniteElementDiscretization):
    if action.domain is not None:
        return action.domain
    action_kind = type(action).__name__
    if action_kind in ("BoundaryLoadAction", "ExteriorFacetAction"):
        return discretization.exterior_facet_domain
    if action_kind == "InteriorFacetAction":
        return discretization.interior_facet_domain
    return discretization.cell_domain


def _action_kind(action) -> str:
    action_type = type(action).__name__
    if action_type == "CellEnergyAction":
        return "energy"
    if action_type == "CellBilinearAction":
        return "bilinear"
    if action_type in ("SourceAction", "BoundaryLoadAction"):
        return "linear"
    return "residual"


def _input_fields(action) -> tuple[str, ...]:
    if type(action).__name__ == "CellResidualAction":
        return action.input_fields
    return (action.field_name,)


def _operators(action) -> tuple[tuple[str, str], ...]:
    inputs = _input_fields(action)
    action_type = type(action).__name__
    if action_type == "DiffusionAction":
        return ((action.field_name, "grad"),)
    if action_type == "InteriorFacetAction":
        return ((action.field_name, "jump"), (action.field_name, "average"))
    if action_type == "SIPGFacetAction":
        return (
            (action.field_name, "jump"),
            (action.field_name, "average"),
            (action.field_name, "grad"),
            (action.field_name, "normal-trace"),
        )
    if action_type in (
        "CellResidualAction",
        "CellEnergyAction",
        "CellBilinearAction",
    ):
        return tuple(
            item for field in inputs for item in ((field, "value"), (field, "grad"))
        )
    return ((action.field_name, "value"),)


def _region_kind(kind: str) -> str:
    return {
        "cell": "cell",
        "exterior_facet": "exterior-facet",
        "interior_facet": "interior-facet",
    }[kind]


def _rule_ids(action, discretization: FiniteElementDiscretization):
    explicit = dict(action.rules)
    return tuple(
        (
            block.name,
            canonical_fingerprint(
                {
                    "kind": "finite-element-rule-binding",
                    "block": block.name,
                    "rule_type": (
                        type(explicit[block.name]).__name__
                        if block.name in explicit
                        else "default"
                    ),
                }
            ),
        )
        for block in discretization.mesh.blocks
    )


def kernel_table_from_form(form: object, /) -> KernelTable:
    if type(form).__name__ != "FiniteElementForm":
        raise TypeError("Expected FiniteElementForm.")
    bindings = []
    for action in form.actions:
        action_type = type(action).__name__
        if action_type in (
            "CellResidualAction",
            "InteriorFacetAction",
            "ExteriorFacetAction",
            "CellBilinearAction",
        ):
            evaluator = action.kernel
        elif action_type == "CellEnergyAction":
            evaluator = action.density
        else:
            evaluator = None
        bindings.append(KernelBinding(action.action_id, action_type, evaluator))
    return KernelTable(bindings)


def lower_finite_element_form(
    form: object,
    discretization: FiniteElementDiscretization,
    /,
) -> LocalActionIR:
    if type(form).__name__ != "FiniteElementForm" or not isinstance(
        discretization, FiniteElementDiscretization
    ):
        raise TypeError("Expected FiniteElementForm and FiniteElementDiscretization.")
    slots = []
    for field_name in form.field_names:
        field_index = discretization._field_index(field_name)
        field_space = discretization.field_spaces[field_index]
        element = discretization.elements[field_index][0]
        value_shape = element.value_shape + tuple(
            discretization.dof_maps[field_index].component_shape
        )
        slots.append(
            FieldSlot(
                field_name,
                "unknown",
                field_space.vector_space.space_id,
                value_shape=value_shape,
            )
        )
    actions = []
    for action in form.actions:
        domain = _domain_for_action(action, discretization)
        region = RegionIR(
            _region_kind(domain.kind),
            domain.domain_id,
            _rule_ids(action, discretization),
        )
        actions.append(
            FiniteElementActionIR(
                _action_kind(action),
                action.field_name,
                _input_fields(action),
                _operators(action),
                region,
                action.action_id,
            )
        )
    return LocalActionIR(slots, actions)


def compile_workset_program(
    ir: LocalActionIR,
    form: object,
    discretization: FiniteElementDiscretization,
    /,
) -> WorksetProgram:
    worksets = []
    mesh = discretization.mesh
    cell_blocks = np.concatenate(
        tuple(
            np.full((block.cell_count,), index, dtype=np.int32)
            for index, block in enumerate(mesh.blocks)
        )
    )
    cell_locals = np.concatenate(
        tuple(np.arange(block.cell_count, dtype=np.int32) for block in mesh.blocks)
    )
    cell_offset = 0
    for block_index, block in enumerate(mesh.blocks):
        block_cells = np.arange(
            cell_offset,
            cell_offset + block.cell_count,
            dtype=np.int32,
        )
        cell_offset += block.cell_count
        for action_index, action in enumerate(form.actions):
            domain = _domain_for_action(action, discretization)
            if domain.kind == "cell":
                selected = np.flatnonzero(
                    np.isin(block_cells, np.asarray(domain.entity_indices))
                )
                if selected.size == 0:
                    continue
                entity_indices = block_cells[selected]
                owner_cells = entity_indices
                neighbour_cells = np.full_like(entity_indices, -1)
                owner_local_entities = np.full_like(entity_indices, -1)
                neighbour_local_entities = np.full_like(entity_indices, -1)
                owner_permutations = np.ones_like(entity_indices, dtype=np.int8)
                neighbour_permutations = np.ones_like(entity_indices, dtype=np.int8)
            else:
                domain_owners = np.asarray(domain.owner_cells, dtype=np.int32)
                selected = np.flatnonzero(
                    (domain_owners >= block_cells[0]) & (domain_owners <= block_cells[-1])
                )
                if selected.size == 0:
                    continue
                entity_indices = np.asarray(domain.entity_indices)[selected]
                owner_cells = domain_owners[selected]
                neighbour_cells = np.asarray(domain.neighbour_cells)[selected]
                owner_local_entities = np.asarray(
                    domain.owner_local_entities, dtype=np.int32
                )[selected]
                neighbour_local_entities = np.asarray(
                    domain.neighbour_local_entities, dtype=np.int32
                )[selected]
                owner_permutations = np.ones(entity_indices.shape, dtype=np.int8)
                neighbour_permutations = np.ones(entity_indices.shape, dtype=np.int8)
                if isinstance(mesh.connectivity, PolygonalConnectivity):
                    signs = np.asarray(mesh.connectivity.cell_edge_signs)
                    owner_permutations = signs[owner_cells, owner_local_entities].astype(
                        np.int8
                    )
                    safe_neighbours = np.maximum(neighbour_cells, 0)
                    safe_local = np.maximum(neighbour_local_entities, 0)
                    neighbour_permutations = np.where(
                        neighbour_cells >= 0,
                        signs[safe_neighbours, safe_local],
                        1,
                    ).astype(np.int8)
            fields = tuple(dict.fromkeys((action.field_name,) + _input_fields(action)))
            gathers = {}
            neighbour_gathers = {}
            widths = {}
            for field in fields:
                field_index = discretization._field_index(field)
                dof_map = discretization.dof_maps[field_index]
                if domain.kind == "cell":
                    route = np.asarray(dof_map.cell_dofs[block_index])[selected]
                    neighbour_route = np.full_like(route, -1)
                else:
                    local_owners = owner_cells - block_cells[0]
                    route = np.asarray(dof_map.cell_dofs[block_index])[local_owners]
                    neighbour_rows = []
                    for neighbour in neighbour_cells:
                        if neighbour < 0:
                            neighbour_rows.append(
                                np.full((route.shape[1],), -1, dtype=np.int32)
                            )
                            continue
                        neighbour_block = int(cell_blocks[neighbour])
                        neighbour_local = int(cell_locals[neighbour])
                        neighbour_dofs = np.asarray(
                            dof_map.cell_dofs[neighbour_block][neighbour_local],
                            dtype=np.int32,
                        )
                        if neighbour_dofs.shape != (route.shape[1],):
                            raise ValueError(
                                "Facet neighbours require compatible local widths."
                            )
                        neighbour_rows.append(neighbour_dofs)
                    neighbour_route = np.asarray(neighbour_rows, dtype=np.int32)
                gathers[field] = route
                neighbour_gathers[field] = neighbour_route
                widths[field] = route.shape[1]
            signature = WorksetSignature(
                _region_kind(domain.kind),
                block.name,
                block.cell_kind,
                dict(ir.actions[action_index].region.rule_ids)[block.name],
                widths,
            )
            worksets.append(
                CompiledWorkset(
                    signature,
                    jnp.asarray([action_index], dtype=jnp.int32),
                    entity_indices,
                    owner_cells,
                    neighbour_cells,
                    gathers,
                    neighbour_gathers=neighbour_gathers,
                    owner_local_entities=owner_local_entities,
                    neighbour_local_entities=neighbour_local_entities,
                    owner_permutations=owner_permutations,
                    neighbour_permutations=neighbour_permutations,
                )
            )
    groups = {}
    for workset in worksets:
        key = (
            workset.signature.signature_id,
            np.asarray(workset.entity_indices).tobytes(),
            np.asarray(workset.owner_cells).tobytes(),
            np.asarray(workset.neighbour_cells).tobytes(),
            np.asarray(workset.owner_local_entities).tobytes(),
            np.asarray(workset.neighbour_local_entities).tobytes(),
            np.asarray(workset.owner_permutations).tobytes(),
            np.asarray(workset.neighbour_permutations).tobytes(),
            tuple((name, np.asarray(route).tobytes()) for name, route in workset.gathers),
            tuple(
                (name, np.asarray(route).tobytes())
                for name, route in workset.neighbour_gathers
            ),
        )
        groups.setdefault(key, []).append(workset)
    fused = []
    for group in groups.values():
        first = group[0]
        action_indices = np.concatenate(
            tuple(np.asarray(value.action_indices) for value in group)
        )
        fused.append(
            CompiledWorkset(
                first.signature,
                action_indices,
                first.entity_indices,
                first.owner_cells,
                first.neighbour_cells,
                dict(first.gathers),
                neighbour_gathers=dict(first.neighbour_gathers),
                owner_local_entities=first.owner_local_entities,
                neighbour_local_entities=first.neighbour_local_entities,
                owner_permutations=first.owner_permutations,
                neighbour_permutations=first.neighbour_permutations,
                valid=first.valid,
            )
        )
    return WorksetProgram(ir, fused)


__all__ = [
    "compile_workset_program",
    "kernel_table_from_form",
    "lower_finite_element_form",
]
