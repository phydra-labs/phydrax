#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization.fem import FiniteElementDiscretization
from .._finite_element_variational import (
    BoundaryLoadTerm,
    CellBilinearTerm,
    CellEnergyTerm,
    CellResidualTerm,
    DiffusionTerm,
    InteriorFacetTerm,
    SourceTerm,
    WeakForm,
)
from ._ir import FieldSlot, LocalActionIR, LocalActionTermIR, RegionIR
from ._worksets import CompiledWorkset, WorksetProgram, WorksetSignature


def _domain_for_term(term, discretization: FiniteElementDiscretization):
    if term.domain is not None:
        return term.domain
    if isinstance(term, BoundaryLoadTerm):
        return discretization.exterior_facet_domain
    if isinstance(term, InteriorFacetTerm):
        return discretization.interior_facet_domain
    return discretization.cell_domain


def _action_kind(term) -> str:
    if isinstance(term, CellEnergyTerm):
        return "energy"
    if isinstance(term, CellBilinearTerm):
        return "bilinear"
    if isinstance(term, (SourceTerm, BoundaryLoadTerm)):
        return "linear"
    return "residual"


def _input_fields(term) -> tuple[str, ...]:
    if isinstance(term, CellResidualTerm):
        return term.input_fields
    return (term.field_name,)


def _operators(term) -> tuple[tuple[str, str], ...]:
    inputs = _input_fields(term)
    if isinstance(term, DiffusionTerm):
        return ((term.field_name, "grad"),)
    if isinstance(term, InteriorFacetTerm):
        return ((term.field_name, "jump"), (term.field_name, "average"))
    if isinstance(term, (CellResidualTerm, CellEnergyTerm, CellBilinearTerm)):
        return tuple(
            item for field in inputs for item in ((field, "value"), (field, "grad"))
        )
    return ((term.field_name, "value"),)


def _region_kind(kind: str) -> str:
    return {
        "cell": "cell",
        "exterior_facet": "exterior-facet",
        "interior_facet": "interior-facet",
    }[kind]


def _rule_ids(term, discretization: FiniteElementDiscretization):
    explicit = dict(term.rules)
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


def lower_weak_form(
    form: WeakForm,
    discretization: FiniteElementDiscretization,
    /,
) -> LocalActionIR:
    if not isinstance(form, WeakForm) or not isinstance(
        discretization, FiniteElementDiscretization
    ):
        raise TypeError("Expected WeakForm and FiniteElementDiscretization.")
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
    terms = []
    for term in form.terms:
        domain = _domain_for_term(term, discretization)
        region = RegionIR(
            _region_kind(domain.kind),
            domain.domain_id,
            _rule_ids(term, discretization),
        )
        terms.append(
            LocalActionTermIR(
                _action_kind(term),
                term.field_name,
                _input_fields(term),
                _operators(term),
                region,
                term.term_id,
            )
        )
    return LocalActionIR(slots, terms)


def compile_workset_program(
    ir: LocalActionIR,
    form: WeakForm,
    discretization: FiniteElementDiscretization,
    /,
) -> WorksetProgram:
    worksets = []
    cell_offset = 0
    for block_index, block in enumerate(discretization.mesh.blocks):
        block_cells = np.arange(
            cell_offset,
            cell_offset + block.cell_count,
            dtype=np.int32,
        )
        cell_offset += block.cell_count
        for term_index, term in enumerate(form.terms):
            domain = _domain_for_term(term, discretization)
            if domain.kind == "cell":
                selected = np.flatnonzero(
                    np.isin(block_cells, np.asarray(domain.entity_indices))
                )
                if selected.size == 0:
                    continue
                entity_indices = block_cells[selected]
                owner_cells = entity_indices
                neighbour_cells = np.full_like(entity_indices, -1)
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
            fields = tuple(dict.fromkeys((term.field_name,) + _input_fields(term)))
            gathers = {}
            widths = {}
            for field in fields:
                field_index = discretization._field_index(field)
                dof_map = discretization.dof_maps[field_index]
                if domain.kind == "cell":
                    route = np.asarray(dof_map.cell_dofs[block_index])[selected]
                else:
                    local_owners = owner_cells - block_cells[0]
                    route = np.asarray(dof_map.cell_dofs[block_index])[local_owners]
                gathers[field] = route
                widths[field] = route.shape[1]
            signature = WorksetSignature(
                _region_kind(domain.kind),
                block.name,
                block.cell_kind,
                dict(ir.terms[term_index].region.rule_ids)[block.name],
                widths,
            )
            worksets.append(
                CompiledWorkset(
                    signature,
                    jnp.asarray([term_index], dtype=jnp.int32),
                    entity_indices,
                    owner_cells,
                    neighbour_cells,
                    gathers,
                )
            )
    return WorksetProgram(ir, worksets)


__all__ = ["compile_workset_program", "lower_weak_form"]
