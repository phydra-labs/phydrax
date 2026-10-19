"""Prepare and assert scalar, form, and material continuation on mapped cells."""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import (
    FiniteElementDiscretization,
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.meshing import CellMeshingResult, MeshAdaptationResult
from phydrax.meshing.qualification import (
    layer_core_cell_regions,
    layer_core_divergence_content,
    layer_core_edge_circulation_defect,
    layer_core_field_elements,
    LAYER_CORE_FIELD_KINDS,
    layer_core_initial_values,
    layer_core_scalar_content,
    prepare_layer_core_remap,
)


# Preserve the allfield qualification limits from
# tools/meshing_qualification/scenarios/family.py:_family_allfield_transfer.
# They are fixed empirical limits for the mapped Q2/form continuation fixture.
TRANSFER_TOLERANCE = 2.0e-11
PHYSICAL_CONTENT_TOLERANCE = 3.0e-11
MATERIAL_INVENTORY_TOLERANCE = 1.0e-13


class FieldTransferEvidence(TypedDict):
    field_name: str
    transfer_id: str
    defects: dict[str, float]
    semantics: str
    jvp_vjp_defect: float
    adjoint_defect: float
    source_field_space_id: str
    target_field_space_id: str


class FormTransferEvidence(TypedDict):
    fields: list[FieldTransferEvidence]
    physical_defect: float | None
    components: int


class MaterialTransferEvidence(TypedDict):
    region_inventory_defect: float
    source: list[list[float]]
    target: list[list[float]]


class FormInventory(TypedDict):
    dofs: int
    components: int
    field_space_ids: list[str]
    physical_content: np.ndarray | None


class MaterialInventory(TypedDict):
    region_inventories: list[list[float]]


type TransferEvidence = dict[str, FormTransferEvidence | MaterialTransferEvidence]
type FieldInventory = dict[str, FormInventory | MaterialInventory]


def _euler_fv(result: CellMeshingResult, /) -> UnstructuredFiniteVolumeDiscretization:
    system = phx.equations.EulerSystem(result.mesh.ambient_dimension)
    return phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh, field_name="state", component_names=system.component_names
    ).prepare(cell_geometry=result.geometry)


def _field_spaces(result: CellMeshingResult, /) -> dict[str, FiniteElementDiscretization]:
    spaces: dict[str, FiniteElementDiscretization] = {}
    for kind in LAYER_CORE_FIELD_KINDS:
        elements = layer_core_field_elements(result, kind)
        spaces[kind] = phx.discretization.FiniteElementPlan(
            result.mesh,
            (
                phx.discretization.FiniteElementFieldSpec("current", elements),
                phx.discretization.FiniteElementFieldSpec("history", elements),
            ),
            coordinate_spec=result.geometry,
        ).prepare()
    return spaces


def _current_gradient(points: Array, args: object, /) -> Array:
    del args
    return jnp.broadcast_to(jnp.asarray((0.0, 1.0, 1.0)), points.shape)


def _history_gradient(points: Array, args: object, /) -> Array:
    del args
    return jnp.stack(
        (jnp.zeros_like(points[:, 0]), 0.25 * points[:, 1], 0.2 * points[:, 2]),
        axis=-1,
    )


def allfield_state(result: CellMeshingResult, /) -> dict[str, Array]:
    fields: dict[str, Array] = {}
    for kind, space in _field_spaces(result).items():
        if kind in ("H1", "DG"):
            fields[kind] = jnp.asarray(layer_core_initial_values(space, kind))
        else:
            first = space.project("current", _current_gradient)
            second = space.project("history", _history_gradient)
            fields[kind] = jnp.stack((first, second), axis=-1)
    fv = _euler_fv(result)
    system = phx.equations.EulerSystem(result.mesh.ambient_dimension)
    primitive = jnp.zeros(
        (fv.cell_count, result.mesh.ambient_dimension + 2),
        dtype=result.mesh.coordinates.dtype,
    )
    primitive = primitive.at[:, 0].set(1.0).at[:, -1].set(1.0)
    fields["material_content"] = fv.cell_volumes[:, None] * system.primitive_to_conserved(
        primitive
    )
    return fields


def _material_inventory(
    result: CellMeshingResult, content: Array, /
) -> list[list[float]]:
    fv = _euler_fv(result)
    regions = layer_core_cell_regions(result)
    labels = np.asarray([regions[int(gid)] for gid in np.asarray(fv.cell_global_ids)])
    values = np.asarray(content)
    assert values.shape == (labels.size, result.mesh.ambient_dimension + 2)
    assert np.all(np.isfinite(values))
    assert result.certification is not None
    assert result.certification.request.domain is not None
    return [
        [
            float(np.sum(values[labels == region, component]))
            for component in range(values.shape[1])
        ]
        for region in range(len(result.certification.request.domain.region_ids))
    ]


def _transfer_form_fields(
    adaptation: MeshAdaptationResult,
    kind: str,
    values: Array,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    /,
    *,
    refine: bool,
) -> tuple[Array, FormTransferEvidence]:
    parts: list[Array] = []
    field_evidence: list[FieldTransferEvidence] = []
    for index, name in enumerate(("current", "history")):
        transfer = phx.discretization.prepare_nested_field_transfer(
            source,
            target,
            adaptation.parent_cells if refine else None,
            field_name=name,
            parent_reference_vertices=adaptation.parent_reference_vertices
            if refine
            else None,
            source_geometry=adaptation.source.geometry,
            target_geometry=adaptation.target.geometry,
            geometry_transition=adaptation.geometry_transition,
            coarsening_witnesses=None if refine else adaptation.coarsening_witnesses,
        )
        assert transfer.evidence.passed, (kind, name, transfer.evidence)
        state = values[:, index]
        mapped = jax.block_until_ready(transfer.transfer.apply(state))
        tangent = jnp.ones_like(state)
        _, jvp = jax.jvp(transfer.transfer.apply, (state,), (tangent,))
        dual = jnp.ones_like(mapped)
        _, vjp = jax.vjp(transfer.transfer.apply, state)
        expected_jvp = transfer.transfer.apply(tangent)
        expected_vjp = transfer.transfer.pullback(dual)
        differential_defect = max(
            float(jnp.max(jnp.abs(jvp - expected_jvp))),
            float(jnp.max(jnp.abs(vjp(dual)[0] - expected_vjp))),
        )
        adjoint_defect = float(
            jnp.abs(jnp.vdot(mapped, dual) - jnp.vdot(state, expected_vjp))
        )
        assert differential_defect <= TRANSFER_TOLERANCE, (kind, name)
        assert adjoint_defect <= TRANSFER_TOLERANCE * max(
            1.0, float(jnp.sum(jnp.abs(mapped)))
        ), (kind, name)
        parts.append(mapped)
        field_evidence.append(
            {
                "field_name": name,
                "transfer_id": transfer.transfer_id,
                "defects": dict(transfer.evidence.defects),
                "semantics": transfer.semantics,
                "jvp_vjp_defect": differential_defect,
                "adjoint_defect": adjoint_defect,
                "source_field_space_id": source.field_spaces[index].field_space_id,
                "target_field_space_id": target.field_spaces[index].field_space_id,
            }
        )
    carried = jnp.stack(parts, axis=-1)
    physical = None
    match kind:
        case "DG":
            physical = float(
                np.max(
                    np.abs(
                        layer_core_scalar_content(target, np.asarray(carried))
                        - layer_core_scalar_content(source, np.asarray(values))
                    )
                )
            )
        case "Hdiv":
            physical = float(
                np.max(
                    np.abs(
                        layer_core_divergence_content(target, np.asarray(carried))
                        - layer_core_divergence_content(source, np.asarray(values))
                    )
                )
            )
        case "Hcurl" if not refine:
            physical = layer_core_edge_circulation_defect(
                source, target, np.asarray(values), np.asarray(carried), adaptation
            )
        case "H1" | "Hcurl":
            pass
        case _:
            raise ValueError(f"hex_fields.kind must be a declared field kind; got {kind}")
    if physical is not None:
        assert physical <= PHYSICAL_CONTENT_TOLERANCE, kind
    return carried, {
        "fields": field_evidence,
        "physical_defect": physical,
        "components": values.shape[1],
    }


def allfield_transfer(
    adaptation: MeshAdaptationResult,
    fields: dict[str, Array],
    /,
    *,
    refine: bool,
) -> tuple[dict[str, Array], TransferEvidence]:
    source = _field_spaces(adaptation.source)
    target = _field_spaces(adaptation.target)
    carried: dict[str, Array] = {}
    evidence: TransferEvidence = {}
    for kind in source:
        carried[kind], evidence[kind] = _transfer_form_fields(
            adaptation, kind, fields[kind], source[kind], target[kind], refine=refine
        )
    first, last = _euler_fv(adaptation.source), _euler_fv(adaptation.target)
    remap = prepare_layer_core_remap(
        first, last, adaptation, "family-refinement" if refine else "family-coarsening"
    )
    carried["material_content"] = jax.block_until_ready(
        remap.apply_content(fields["material_content"])
    )
    old_inventory = _material_inventory(adaptation.source, fields["material_content"])
    new_inventory = _material_inventory(adaptation.target, carried["material_content"])
    defect = float(np.max(np.abs(np.asarray(new_inventory) - np.asarray(old_inventory))))
    assert defect <= MATERIAL_INVENTORY_TOLERANCE * max(
        1.0, float(np.sum(np.abs(np.asarray(fields["material_content"]))))
    )
    evidence["material_content"] = {
        "region_inventory_defect": defect,
        "source": old_inventory,
        "target": new_inventory,
    }
    return carried, evidence


def allfield_inventory(
    result: CellMeshingResult, fields: dict[str, Array], /
) -> FieldInventory:
    spaces = _field_spaces(result)
    assert set(fields) == {*spaces, "material_content"}
    inventory: FieldInventory = {}
    for kind, space in spaces.items():
        values = np.asarray(fields[kind])
        assert values.shape == (space.dof_maps[0].global_dof_count, 2), kind
        assert np.all(np.isfinite(values)), kind
        for index, owner in enumerate(space.field_spaces):
            owner.vector_space.validate(fields[kind][:, index])
        content = (
            layer_core_scalar_content(space, values)
            if kind in ("H1", "DG")
            else layer_core_divergence_content(space, values)
            if kind == "Hdiv"
            else None
        )
        inventory[kind] = {
            "dofs": values.shape[0],
            "components": 2,
            "field_space_ids": [owner.field_space_id for owner in space.field_spaces],
            "physical_content": content,
        }
    inventory["material_content"] = {
        "region_inventories": _material_inventory(result, fields["material_content"])
    }
    return inventory


def quadratic_value(points: Array, /) -> Array:
    return jnp.sum(points * points, axis=-1)


def quadratic_source(points: Array, parameters: object, /) -> Array:
    del parameters
    return jnp.full(points.shape[:-1], -2.0 * points.shape[-1], dtype=points.dtype)
