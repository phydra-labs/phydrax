#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Durable layer/core lifecycle continuation: fields, finite volumes, history, PDE.

The consumer reads one accepted native layer/core source together with the
refinement and the inverse coarsening that were archived on that same source
lineage. It never regenerates the source or invents a refined cell count:

* H1 Q2, DG1, H(curl) degree 2 and H(div) degree 2 fields cross the refinement
  and its inverse through the adaptation's own parent, geometry-transition and
  coarsening witnesses, with bitwise restoration, adjoint and physical-content
  evidence;
* finite-volume cell averages of a two-material density move through the
  mapped nested conservative remap, preserving every material inventory;
* an irreversible damage history, the material density and a harmonic
  diffusion solution are carried by the atomic finite-element topology
  transaction, whose changed-PDE certificate must roll back unchanged.

The durable meshing-source closure must retain ``accepted_data`` with the
``(refinement, coarsening)`` adaptation results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import perf_counter
from types import MappingProxyType
from typing import final, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax import NonTrainableState, StrictModule
from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._meshcore import NativeBatchTicket, NativeExecutionScope
from phydrax.discretization import (
    CellMesh,
    PolynomialComposedCellGeometryElement,
    reference_cell_topology,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization.fem import (
    dirichlet_constraint,
    discontinuous_element,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementFieldTransfer,
    FiniteElementPlan,
    FiniteElementSpec,
    FiniteElementTransferDiscretization,
    form_element,
    lagrange_element,
    prepare_nested_compatible_field_transfers,
    prepare_nested_field_transfer,
)
from phydrax.discretization.fem._reference_rules import degree_aware_reference_rule
from phydrax.discretization.finite_volume import (
    prepare_unstructured_conservative_remap,
    UnstructuredConservativeRemapPlan,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.equations import (
    compile_finite_element_problem,
    DiffusionAction,
    FiniteElementForm,
    MaterialSiteId,
    MaterialState,
    MaterialTransaction,
)
from phydrax.geometry._embedding_state import coordinate_certificate_request
from phydrax.lifecycle import read_meshing_source_execution_controls
from phydrax.lifecycle._meshing_sources import (
    _read_meshing_source_archive,
    _restore_meshing_source_archive,
)
from phydrax.linalg import DenseLU, LinearSolvePolicy, solve
from phydrax.meshing import (
    CellMeshingResult,
    MeshAdaptationResult,
    MeshAdaptationStatus,
    MeshLineage,
    MeshPart,
    NativeExecutionRecord,
)
from phydrax.meshing._volume_generation import native_volume_source_execution_budget
from phydrax.meshing.providers import NativeLayerCoreSource
from phydrax.solver import FiniteElementAcceptedState, FiniteElementTopologyTransaction
from phydrax.typing import checked


class FiniteVolumeFieldTemplate(TypedDict):
    name: str
    space_family: str
    degree: int
    continuity: str
    component_shape: tuple[int, ...]
    component_names: tuple[str, ...]
    role: str


def layer_core_finite_volume_field_templates() -> tuple[FiniteVolumeFieldTemplate, ...]:
    """Return the cell-average material-density declaration."""
    return (
        {
            "name": "state",
            "space_family": "finite-volume",
            "degree": 0,
            "continuity": "cell-local",
            "component_shape": (1,),
            "component_names": ("value",),
            "role": "conservative-density",
        },
    )


class LayerCoreFieldContinuation(TypedDict):
    field: str
    dofs: list[int]
    roundtrip_error: float
    physical_error: float
    forward_defects: dict[str, float]
    reverse_defects: dict[str, float]


class LayerCoreVolumeContinuation(TypedDict):
    cells: list[int]
    region_inventories: list[float]
    defects: dict[str, float]


class LayerCoreHistoryStage(TypedDict):
    phase: str
    cells: int
    changed_pde_rolled_back: bool
    pde_defect: float
    damage_defect: float
    inventory_defect: float


class LayerCoreHistoryContinuation(TypedDict):
    initial_cells: int
    region_inventories: list[float]
    stages: list[LayerCoreHistoryStage]


LAYER_CORE_FIELD_ROLES = MappingProxyType(
    {
        "H1": "intensive",
        "DG": "conservative-density",
        "Hcurl": "compatible-cochain",
        "Hdiv": "compatible-cochain",
    }
)
LAYER_CORE_FIELD_KINDS = tuple(LAYER_CORE_FIELD_ROLES)
LAYER_CORE_REGION_DENSITIES = (2.0, 3.0)


@final
class LayerCoreLifecyclePlans(StrictModule):
    """Authenticated immutable preparations shared by one cold continuation."""

    transfer_spaces: dict[
        str, tuple[CellMeshingResult, FiniteElementTransferDiscretization]
    ] = eqx.field(default_factory=dict)
    history_spaces: dict[str, tuple[CellMeshingResult, FiniteElementDiscretization]] = (
        eqx.field(default_factory=dict)
    )
    field_transfers: dict[tuple[str, str], FiniteElementFieldTransfer] = eqx.field(
        default_factory=dict
    )
    finite_volumes: dict[
        str, tuple[CellMeshingResult, UnstructuredFiniteVolumeDiscretization]
    ] = eqx.field(default_factory=dict)
    remaps: dict[tuple[str, str], UnstructuredConservativeRemapPlan] = eqx.field(
        default_factory=dict
    )
    diffusion_solutions: dict[
        tuple[str, float], tuple[FiniteElementDiscretization, Array, str]
    ] = eqx.field(default_factory=dict)
    _readiness: dict[str, Event] = eqx.field(static=True, default_factory=dict)

    def __init__(self) -> None:
        self.transfer_spaces = {}
        self.history_spaces = {}
        self.field_transfers = {}
        self.finite_volumes = {}
        self.remaps = {}
        self.diffusion_solutions = {}
        self._readiness = {}


class LayerCoreMaterialDeclaration(TypedDict):
    source_id: str
    source_revision: str
    domain_id: str
    region_ids: tuple[str, ...]
    densities: tuple[float, ...]
    site_id: str


@checked
def layer_core_material_declaration(
    source: CellMeshingResult,
    /,
    *,
    densities: tuple[float, ...] = LAYER_CORE_REGION_DENSITIES,
    site_id: str = "two-material-density",
) -> LayerCoreMaterialDeclaration:
    report = source.certification
    if report is None or report.coverage is None or report.request.domain is None:
        raise ValueError(
            f"material.source_certification must include coverage and a domain; got {report!r}"
        )
    declaration: LayerCoreMaterialDeclaration = {
        "source_id": report.request.domain.source_id,
        "source_revision": report.request.domain.source_revision,
        "domain_id": report.request.domain.domain_id,
        "region_ids": report.coverage.region_ids,
        "densities": densities,
        "site_id": site_id,
    }
    layer_core_material_values(source, declaration)
    return declaration


@checked
def layer_core_material_values(
    result: CellMeshingResult, declaration: LayerCoreMaterialDeclaration, /
) -> np.ndarray:
    report = result.certification
    if (
        report is None
        or not report.passed
        or report.coverage is None
        or report.coverage.status != "certified"
        or report.request.domain is None
    ):
        raise ValueError(
            f"material.result_certification must pass with certified coverage and a domain; got {report!r}"
        )
    if (
        report.request.domain.source_id,
        report.request.domain.source_revision,
        report.coverage.domain_id,
        report.coverage.region_ids,
    ) != (
        declaration["source_id"],
        declaration["source_revision"],
        declaration["domain_id"],
        declaration["region_ids"],
    ):
        raise ValueError(
            f"material.declaration must bind the result source and region namespace; got {declaration!r} for {(report.request.domain.source_id, report.request.domain.source_revision, report.coverage.domain_id, report.coverage.region_ids)!r}"
        )
    values = np.asarray(declaration["densities"], dtype=np.float64)
    if (
        values.shape != (len(declaration["region_ids"]),)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            f"material.densities must be finite and positive with shape {(len(declaration['region_ids']),)!r}; got {values!r}"
        )
    MaterialSiteId(declaration["site_id"])
    labels = layer_core_region_labels(result)
    if np.any(labels < 0) or np.any(labels >= values.size):
        raise ValueError(
            f"material.cell_regions must index the density bank of size {values.size}; got {labels!r}"
        )
    return values


def accepted_layer_core_lineage(
    records: object, /
) -> tuple[CellMeshingResult, MeshAdaptationResult, MeshAdaptationResult]:
    """The archived source, refinement and inverse coarsening of one lineage."""
    if not isinstance(records, Mapping):
        raise TypeError(
            f"closure.records must be a named record mapping; got {type(records).__name__}"
        )
    part = records["generation_part"]
    accepted = records["accepted_data"]
    if not isinstance(part, MeshPart) or not isinstance(part.carrier, CellMeshingResult):
        raise TypeError(
            f"closure.generation_part must be a MeshPart with a CellMeshingResult carrier; got {part!r}"
        )
    if not isinstance(accepted, Mapping):
        raise TypeError(
            f"closure.accepted_data must be a mapping; got {type(accepted).__name__}"
        )
    refinement, coarsening = accepted["adaptation_results"]
    source = part.carrier
    if not isinstance(refinement, MeshAdaptationResult) or not isinstance(
        coarsening, MeshAdaptationResult
    ):
        raise TypeError(
            f"closure.adaptation_results must contain MeshAdaptationResult objects; got {(type(refinement).__name__, type(coarsening).__name__)!r}"
        )
    if (
        refinement.source.result_id != source.result_id
        or coarsening.source.result_id != refinement.target.result_id
    ):
        raise ValueError(
            f"closure.adaptation_results must continue source lineage {source.result_id!r}; got {(refinement.source.result_id, refinement.target.result_id, coarsening.source.result_id)!r}"
        )
    for stage in (refinement, coarsening):
        if stage.status is not MeshAdaptationStatus.COMPLETE:
            raise ValueError(
                f"adaptation.status must be MeshAdaptationStatus.COMPLETE; got {stage.status!r}"
            )
    for result in (source, refinement.target, coarsening.target):
        if result.certification is None or not result.certification.passed:
            raise ValueError(
                f"continuation.certification must pass; got {result.certification!r}"
            )
    return source, refinement, coarsening


def _cell_ids(result: CellMeshingResult, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )


@checked
def layer_core_cell_regions(result: CellMeshingResult, /) -> dict[int, int]:
    report = result.certification
    if report is None or report.request.cell_regions is None:
        raise ValueError(
            f"continuation.cell_regions must be present in certification; got {report!r}"
        )
    labels = np.asarray(report.request.cell_regions, dtype=np.int64)
    return dict(zip(_cell_ids(result).tolist(), labels.tolist(), strict=True))


def _field_template(kind: str, cell_kind: str, /) -> tuple[str, FiniteElementFieldSpec]:
    """Build caller-owned element records; only their owners cache host tables.

    FiniteElementSpec can carry device arrays, so continuation must not retain
    these records in a process-global cache.
    """
    match kind:
        case "H1":
            element = lagrange_element(cell_kind, 2)
        case "DG":
            element = discontinuous_element(cell_kind, 1)
        case "Hcurl":
            element = form_element(
                cell_kind,
                1,
                2,
                family=(
                    "tensor-trimmed"
                    if cell_kind in ("quadrilateral", "hexahedron")
                    else "trimmed"
                ),
                twist="untwisted",
                proxy="circulation",
            )
        case "Hdiv":
            element = form_element(
                cell_kind,
                2,
                2,
                family=(
                    "tensor-trimmed"
                    if cell_kind in ("quadrilateral", "hexahedron")
                    else "trimmed"
                ),
                twist="twisted",
                proxy="flux",
            )
        case _:
            raise ValueError(
                f"continuation.field_kind must be one of {LAYER_CORE_FIELD_KINDS!r}; got {kind!r}"
            )
    return LAYER_CORE_FIELD_ROLES[kind], FiniteElementFieldSpec(kind, element)


def layer_core_field_templates(
    cell_kind: str, /, *, kinds: tuple[str, ...] = LAYER_CORE_FIELD_KINDS
) -> tuple[tuple[str, FiniteElementFieldSpec], ...]:
    """Return mesh-free H1, DG, H(curl), and H(div) transfer declarations."""
    return tuple(_field_template(kind, cell_kind) for kind in kinds)


@checked
def layer_core_field_elements(
    result: CellMeshingResult, kind: str, /
) -> dict[str, FiniteElementSpec]:
    if kind not in LAYER_CORE_FIELD_KINDS:
        raise ValueError(
            f"continuation.field_kind must be one of {LAYER_CORE_FIELD_KINDS!r}; got {kind!r}"
        )
    templates = {
        cell_kind: {
            field.name: field
            for _, field in layer_core_field_templates(cell_kind, kinds=(kind,))
        }
        for cell_kind in {block.cell_kind for block in result.mesh.blocks}
    }
    return {
        block.name: templates[block.cell_kind][kind].elements[0]
        for block in result.mesh.blocks
    }


type _FieldSpace = FiniteElementDiscretization | FiniteElementTransferDiscretization


def _transfer_space(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    reuse_prepared: FiniteElementTransferDiscretization | None = None,
    reuse_source: CellMeshingResult | None = None,
) -> FiniteElementTransferDiscretization:
    if (reuse_prepared is None) != (reuse_source is None):
        raise ValueError(
            f"transfer.reuse_preparation must be supplied together with reuse_source; got {(reuse_prepared is not None, reuse_source is not None)!r}"
        )
    templates = {
        cell_kind: {
            field.name: field for _, field in layer_core_field_templates(cell_kind)
        }
        for cell_kind in {block.cell_kind for block in result.mesh.blocks}
    }
    fields = tuple(
        FiniteElementFieldSpec(
            field.name,
            {
                block.name: templates[block.cell_kind][field.name].elements[0]
                for block in result.mesh.blocks
            },
            component_shape=field.component_shape,
        )
        for field in next(iter(templates.values())).values()
    )
    plan = FiniteElementPlan(
        result.mesh,
        fields,
        coordinate_spec=result.geometry,
    )
    retained = plans.transfer_spaces.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError(
                f"transfer.cached_result_id must equal {result.result_id!r}; got {retained_result.result_id!r}"
            )
        retained_space.require_exact_plan(plan)
        return retained_space
    if reuse_source is not None:
        require_restored_layer_core_result_identity(reuse_source, result)
    prepared = plan.prepare_transfer(reuse_prepared=reuse_prepared)
    plans.transfer_spaces[result.result_id] = (result, prepared)
    return prepared


type _CellRoute = tuple[int, FiniteElementSpec, np.ndarray, np.ndarray]


def _cells(space: _FieldSpace, /, *, field_name: str | None = None) -> list[_CellRoute]:
    index = 0 if field_name is None else space.field_index(field_name)
    rows: list[_CellRoute] = []
    for block, element, routes, transforms in zip(
        space.mesh.blocks,
        space.elements[index],
        space.dof_maps[index].cell_dofs,
        space.dof_maps[index].cell_transforms,
        strict=True,
    ):
        for identifier, route, transform in zip(
            np.asarray(block.global_ids).tolist(),
            np.asarray(routes),
            np.asarray(transforms),
            strict=True,
        ):
            rows.append((identifier, element, route, transform))
    return rows


@checked
def layer_core_scalar_content(
    space: _FieldSpace,
    values: np.ndarray,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    """Return the physical integral through mapped coordinate Jacobians."""
    index = 0 if field_name is None else space.field_index(field_name)
    total = np.zeros(values.shape[1:], dtype=np.float64)
    runtime = np.asarray(space.default_runtime.coordinates)
    for element, coordinate, routes, geometry_routes in zip(
        space.elements[index],
        space.coordinate_elements,
        space.dof_maps[index].cell_dofs,
        space.coordinate_dofs,
        strict=True,
    ):
        # Refined and restored blocks keep the source basis restricted or
        # composed onto each child; they are tabulated, not reinterpolated.
        if not isinstance(
            coordinate,
            (
                FiniteElementSpec,
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                SplineCellGeometryElement,
            ),
        ):
            raise TypeError(
                f"continuation.coordinate_element must support coordinate tabulation; got {type(coordinate).__name__}"
            )
        points, weights = degree_aware_reference_rule(element.cell_kind, 8)
        basis = np.asarray(element.tabulate(points)[0])
        gradients = np.asarray(coordinate.tabulate(points)[1])
        for route, geometry_route in zip(
            np.asarray(routes), np.asarray(geometry_routes), strict=True
        ):
            determinant = np.linalg.det(
                np.einsum("qna,nd->qda", gradients, runtime[geometry_route])
            )
            if np.any(determinant <= 0.0):
                raise ValueError(
                    f"continuation.jacobian_determinants must be positive; got {determinant!r}"
                )
            total += np.einsum(
                "q,qn,np->p", np.asarray(weights) * determinant, basis, values[route]
            )
    return total


@checked
def layer_core_divergence_content(
    space: _FieldSpace,
    values: np.ndarray,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    """Return total flux content using reference-space contravariant Piola integration."""
    total = np.zeros(values.shape[1:], dtype=np.float64)
    for _, element, route, transform in _cells(space, field_name=field_name):
        points, weights = degree_aware_reference_rule(element.cell_kind, 8)
        divergence = np.trace(np.asarray(element.tabulate(points)[1]), axis1=-2, axis2=-1)
        total += np.einsum(
            "q,qn,np->p", np.asarray(weights), divergence, transform @ values[route]
        )
    return total


def _line_content(
    element: FiniteElementSpec,
    coefficients: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(8)
    points = first + ((nodes + 1.0) / 2.0)[:, None] * (second - first)
    basis = np.asarray(element.tabulate(points)[0])
    return np.einsum("q,qnc,np,c->p", weights / 2.0, basis, coefficients, second - first)


@checked
def layer_core_edge_circulation_defect(
    fine: _FieldSpace,
    coarse: _FieldSpace,
    fine_values: np.ndarray,
    coarse_values: np.ndarray,
    coarsening: MeshAdaptationResult,
    /,
    *,
    field_name: str | None = None,
) -> float:
    """Every coarse reference edge circulation equals its fine sub-edge sum."""
    witnesses = coarsening.coarsening_witnesses
    if witnesses is None:
        raise ValueError(
            "coarsening.coarsening_witnesses must retain nested reference data; got None"
        )
    joins = {
        identifier: (parent, np.asarray(reference))
        for identifier, parent, reference in zip(
            np.asarray(witnesses.fine_cell_ids).tolist(),
            np.asarray(witnesses.coarse_cell_ids).tolist(),
            np.asarray(witnesses.fine_reference_vertices),
            strict=True,
        )
    }
    fine_cells = _cells(fine, field_name=field_name)
    defect = 0.0
    for identifier, element, route, transform in _cells(coarse, field_name=field_name):
        topology = reference_cell_topology(element.cell_kind)
        vertices = np.asarray(topology.vertices)
        children = []
        for child, fine_element, fine_route, fine_transform in fine_cells:
            child_topology = reference_cell_topology(fine_element.cell_kind)
            parent, image = joins.get(child, (child, np.asarray(child_topology.vertices)))
            if parent == identifier:
                children.append((fine_element, fine_route, fine_transform, image))
        if not children:
            raise ValueError(
                f"coarsening.fine_children must be nonempty for coarse cell {identifier}; got {children!r}"
            )
        for first, second in topology.entities[1]:
            start, tangent = vertices[first], vertices[second] - vertices[first]
            expected = np.zeros(fine_values.shape[1:], dtype=np.float64)
            supports: set[tuple[tuple[float, ...], ...]] = set()
            for fine_element, fine_route, fine_transform, image in children:
                local = reference_cell_topology(fine_element.cell_kind)
                local_vertices = np.asarray(local.vertices)
                for a, b in local.entities[1]:
                    endpoints = image[[a, b]]
                    parameters = (endpoints - start) @ tangent / (tangent @ tangent)
                    if (
                        np.max(
                            np.abs(endpoints - (start + parameters[:, None] * tangent))
                        )
                        > 1.0e-12
                        or np.min(parameters) < -1.0e-12
                        or np.max(parameters) > 1.0 + 1.0e-12
                    ):
                        continue
                    support = tuple(sorted(tuple(point.tolist()) for point in endpoints))
                    if support in supports:
                        continue
                    supports.add(support)
                    sign = 1.0 if parameters[1] > parameters[0] else -1.0
                    expected += sign * _line_content(
                        fine_element,
                        fine_transform @ fine_values[fine_route],
                        local_vertices[a],
                        local_vertices[b],
                    )
            actual = _line_content(
                element,
                transform @ coarse_values[route],
                vertices[first],
                vertices[second],
            )
            defect = max(defect, float(np.max(np.abs(actual - expected))))
    return defect


@checked
def layer_core_initial_values(
    space: _FieldSpace,
    kind: str,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    index = 0 if field_name is None else space.field_index(field_name)
    points = np.asarray(space.dof_maps[index].dof_coordinates)
    if kind not in ("H1", "DG"):
        return np.random.default_rng(19).normal(
            size=(space.dof_maps[index].global_dof_count, 2)
        )
    # A physical affine field pulls back through the quadratic fiber graph into
    # H1 Q2 exactly. A physical quadratic would compose to degree four and is
    # not an admissible polynomial-reproduction oracle for this curved geometry.
    return np.column_stack(
        (
            1.0 + points[:, 1] + points[:, 2],
            0.5 + 0.125 * points[:, 1] + 0.1 * points[:, 2],
        )
    )


@checked
def continue_layer_core_fields(
    source: CellMeshingResult,
    refinement: MeshAdaptationResult,
    coarsening: MeshAdaptationResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans | None = None,
) -> list[LayerCoreFieldContinuation]:
    """Refine and restore H1/DG/H(curl)/H(div) fields through the archived witnesses."""
    plans_ = LayerCoreLifecyclePlans() if plans is None else plans
    spaces_ready = plans_._readiness.get("spaces")
    h1_ready = plans_._readiness.get("h1")
    first = _transfer_space(source, plans=plans_)
    fine = _transfer_space(refinement.target, plans=plans_)
    last = _transfer_space(
        coarsening.target,
        plans=plans_,
        reuse_prepared=first,
        reuse_source=source,
    )
    forward: dict[str, FiniteElementFieldTransfer] = {}
    reverse: dict[str, FiniteElementFieldTransfer] = {}
    for kind, role in LAYER_CORE_FIELD_ROLES.items():
        if role not in ("intensive", "conservative-density"):
            continue
        forward[kind] = prepare_nested_field_transfer(
            first,
            fine,
            refinement.parent_cells,
            field_name=kind,
            parent_reference_vertices=refinement.parent_reference_vertices,
            source_geometry=source.geometry,
            target_geometry=refinement.target.geometry,
            geometry_transition=refinement.geometry_transition,
        )
        reverse[kind] = prepare_nested_field_transfer(
            fine,
            last,
            None,
            field_name=kind,
            source_geometry=refinement.target.geometry,
            target_geometry=coarsening.target.geometry,
            geometry_transition=coarsening.geometry_transition,
            coarsening_witnesses=coarsening.coarsening_witnesses,
        )
        plans_.field_transfers[(refinement.result_id, kind)] = forward[kind]
        plans_.field_transfers[(coarsening.result_id, kind)] = reverse[kind]
        if kind == "H1" and h1_ready is not None:
            h1_ready.set()
    compatible = tuple(
        kind
        for kind, role in LAYER_CORE_FIELD_ROLES.items()
        if role == "compatible-cochain"
    )
    compatible_forward = prepare_nested_compatible_field_transfers(
        first,
        fine,
        refinement.parent_cells,
        field_names=compatible,
        parent_reference_vertices=refinement.parent_reference_vertices,
        source_geometry=source.geometry,
        target_geometry=refinement.target.geometry,
        geometry_transition=refinement.geometry_transition,
    )
    compatible_reverse = prepare_nested_compatible_field_transfers(
        fine,
        last,
        None,
        field_names=compatible,
        source_geometry=refinement.target.geometry,
        target_geometry=coarsening.target.geometry,
        geometry_transition=coarsening.geometry_transition,
        coarsening_witnesses=coarsening.coarsening_witnesses,
    )
    forward.update(zip(compatible, compatible_forward, strict=True))
    reverse.update(zip(compatible, compatible_reverse, strict=True))
    for kind in compatible:
        plans_.field_transfers[(refinement.result_id, kind)] = forward[kind]
        plans_.field_transfers[(coarsening.result_id, kind)] = reverse[kind]

    if spaces_ready is not None:
        spaces_ready.set()
    summaries: list[LayerCoreFieldContinuation] = []
    for kind in LAYER_CORE_FIELD_KINDS:
        forward_transfer = forward[kind]
        reverse_transfer = reverse[kind]
        if not forward_transfer.evidence.passed or not reverse_transfer.evidence.passed:
            raise ValueError(
                f"continuation.transfer_evidence must pass for {kind}; got {(forward_transfer.evidence.passed, reverse_transfer.evidence.passed)!r}"
            )
        values = layer_core_initial_values(first, kind, field_name=kind)
        transferred = np.asarray(forward_transfer.transfer.apply(values))
        restored = np.asarray(reverse_transfer.transfer.apply(transferred))
        np.testing.assert_allclose(restored, values, rtol=2.0e-11, atol=2.0e-11)
        dual = np.random.default_rng(29).normal(size=transferred.shape)
        np.testing.assert_allclose(
            np.vdot(transferred, dual),
            np.vdot(values, forward_transfer.transfer.pullback(dual)),
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        compiled = np.asarray(
            eqx.filter_jit(forward_transfer.transfer.apply)(jnp.asarray(values))
        )
        np.testing.assert_allclose(compiled, transferred, rtol=2.0e-13, atol=2.0e-13)
        arbitrary = np.random.default_rng(47).normal(size=transferred.shape)
        projected = np.asarray(reverse_transfer.transfer.apply(arbitrary))
        match kind:
            case "H1":
                expected = layer_core_initial_values(fine, kind, field_name=kind)
                physical = float(np.max(np.abs(transferred - expected)))
            case "DG":
                physical = float(
                    np.max(
                        np.abs(
                            layer_core_scalar_content(last, projected, field_name=kind)
                            - layer_core_scalar_content(fine, arbitrary, field_name=kind)
                        )
                    )
                )
            case "Hdiv":
                physical = float(
                    np.max(
                        np.abs(
                            layer_core_divergence_content(
                                last, projected, field_name=kind
                            )
                            - layer_core_divergence_content(
                                fine, arbitrary, field_name=kind
                            )
                        )
                    )
                )
            case _:
                physical = layer_core_edge_circulation_defect(
                    fine,
                    last,
                    arbitrary,
                    projected,
                    coarsening,
                    field_name=kind,
                )
        if physical > 3.0e-11:
            raise ValueError(
                f"continuation.physical_defect must be at most 3.0e-11 for {kind}; got {physical:.3e}"
            )
        index = first.field_index(kind)
        fine_index = fine.field_index(kind)
        last_index = last.field_index(kind)
        summaries.append(
            {
                "field": kind,
                "dofs": [
                    first.dof_maps[index].global_dof_count,
                    fine.dof_maps[fine_index].global_dof_count,
                    last.dof_maps[last_index].global_dof_count,
                ],
                "roundtrip_error": float(np.max(np.abs(restored - values))),
                "physical_error": physical,
                "forward_defects": dict(forward_transfer.evidence.defects),
                "reverse_defects": dict(reverse_transfer.evidence.defects),
            }
        )
    return summaries


def _finite_volume(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> UnstructuredFiniteVolumeDiscretization:
    retained = plans.finite_volumes.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError(
                f"finite_volume.cached_result_id must equal {result.result_id!r}; got {retained_result.result_id!r}"
            )
        return retained_space
    report = result.certification
    if report is None or report.embedding is None:
        raise ValueError(
            f"finite_volume.embedding must be present in certification; got {report!r}"
        )
    periodic = result.mesh.periodic_topology is not None
    periodic_geometry = result.geometry if periodic else None
    periodic_embedding = report.embedding if periodic else None
    template = layer_core_finite_volume_field_templates()[0]
    space = UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        field_name=template["name"],
        component_names=template["component_names"],
        periodic_geometry=periodic_geometry,
        periodic_embedding=periodic_embedding,
    ).prepare(
        cell_geometry=result.geometry,
        periodic_embedding=periodic_embedding,
    )
    plans.finite_volumes[result.result_id] = (result, space)
    return space


@checked
def prepare_layer_core_remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    stage: MeshAdaptationResult,
    provenance: str,
    /,
    *,
    plans: LayerCoreLifecyclePlans | None = None,
) -> UnstructuredConservativeRemapPlan:
    plans_ = LayerCoreLifecyclePlans() if plans is None else plans
    key = (stage.result_id, provenance)
    if key in plans_.remaps:
        plan = plans_.remaps[key]
        if (
            plan.source_topology_id != source.topology_id
            or plan.source_geometry_id != source.geometry_id
            or plan.target_topology_id != target.topology_id
            or plan.target_geometry_id != target.geometry_id
        ):
            raise ValueError(
                f"remap.cached_epoch must match source and target topology and geometry; got {(plan.source_topology_id, plan.source_geometry_id, plan.target_topology_id, plan.target_geometry_id)!r} for {(source.topology_id, source.geometry_id, target.topology_id, target.geometry_id)!r}"
            )
        return plan
    remap = prepare_unstructured_conservative_remap(
        source,
        target,
        provenance=provenance,
        source_geometry=stage.source.geometry,
        target_geometry=stage.target.geometry,
        geometry_transition=stage.geometry_transition,
    )
    if remap.plan is None:
        raise ValueError(
            f"remap.plan must be accepted for {provenance!r}; got {remap.status!r}: {remap.reason}"
        )
    plans_.remaps[key] = remap.plan
    return remap.plan


def _inventories(
    volumes: np.ndarray,
    densities: np.ndarray,
    gids: np.ndarray,
    regions: Mapping[int, int],
    region_count: int,
    /,
) -> np.ndarray:
    labels = np.asarray([regions[int(identifier)] for identifier in gids], dtype=np.int64)
    return np.bincount(labels, weights=volumes * densities, minlength=region_count)


@checked
def continue_layer_core_finite_volume(
    source: CellMeshingResult,
    refinement: MeshAdaptationResult,
    coarsening: MeshAdaptationResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans | None = None,
) -> LayerCoreVolumeContinuation:
    """Conservative declared source-material density through refinement and its inverse."""
    plans_ = LayerCoreLifecyclePlans() if plans is None else plans
    spaces_ready = plans_._readiness.get("volume")
    region_densities = layer_core_material_values(source, material)
    layer_core_material_values(refinement.target, material)
    layer_core_material_values(coarsening.target, material)
    first = _finite_volume(source, plans=plans_)
    fine = _finite_volume(refinement.target, plans=plans_)
    last = _finite_volume(coarsening.target, plans=plans_)
    if spaces_ready is not None:
        spaces_ready.set()
    forward = prepare_layer_core_remap(
        first, fine, refinement, "layer-core-refinement", plans=plans_
    )
    reverse = prepare_layer_core_remap(
        fine,
        last,
        coarsening,
        "layer-core-inverse-coarsening",
        plans=plans_,
    )
    gids = [
        np.asarray(space.cell_global_ids, dtype=np.int64) for space in (first, fine, last)
    ]
    regions = [
        layer_core_cell_regions(result)
        for result in (source, refinement.target, coarsening.target)
    ]
    volumes = [
        np.asarray(space.cell_volumes, dtype=np.float64) for space in (first, fine, last)
    ]
    densities = np.asarray(
        [region_densities[regions[0][int(identifier)]] for identifier in gids[0]]
    )
    content = (volumes[0] * densities)[:, None]
    inventories = _inventories(
        volumes[0], densities, gids[0], regions[0], region_densities.size
    )
    fine_content = np.asarray(forward.apply_content(jnp.asarray(content)))
    fine_densities = fine_content[:, 0] / volumes[1]
    restored_content = np.asarray(reverse.apply_content(jnp.asarray(fine_content)))
    rows = {identifier: row for row, identifier in enumerate(gids[0].tolist())}
    order = np.asarray(
        [rows[identifier] for identifier in gids[2].tolist()], dtype=np.int64
    )
    defects = {
        "refined_region_inventory": float(
            np.max(
                np.abs(
                    _inventories(
                        volumes[1],
                        fine_densities,
                        gids[1],
                        regions[1],
                        region_densities.size,
                    )
                    - inventories
                )
            )
        ),
        "restored_cell_content": float(np.max(np.abs(restored_content - content[order]))),
        "restored_cell_volume": float(np.max(np.abs(volumes[2] - volumes[0][order]))),
    }
    arbitrary = (
        np.random.default_rng(53).uniform(1.0, 4.0, size=(volumes[1].size, 1))
        * volumes[1][:, None]
    )
    defects["arbitrary_inverse_inventory"] = float(
        abs(
            np.sum(np.asarray(reverse.apply_content(jnp.asarray(arbitrary))))
            - np.sum(arbitrary)
        )
    )
    scale = float(np.sum(np.abs(content)))
    if max(defects.values()) > 1.0e-13 * max(1.0, scale):
        raise ValueError(
            f"finite_volume.conservation_defects must be at most {1.0e-13 * max(1.0, scale)!r}; got {defects!r}"
        )
    return {
        "cells": [value.size for value in gids],
        "region_inventories": inventories.tolist(),
        "defects": defects,
    }


@final
class _HarmonicBoundary(StrictModule, NonTrainableState):
    offset: Array

    def __init__(self, offset: Array) -> None:
        self.offset = offset

    def __call__(self, points: Array) -> Array:
        return points[..., 1] + points[..., 2] + self.offset


def _damage(points: Array, args: object) -> Array:
    """A bounded physical affine history represented exactly by mapped Q2."""
    del args
    return 0.5 + 0.125 * points[..., 1] + 0.1 * points[..., 2]


def _solve_diffusion(
    space: FiniteElementDiscretization,
    offset: float,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> tuple[Array, str]:
    key = (space.prepared_id, float(offset))
    retained = plans.diffusion_solutions.get(key)
    if retained is not None:
        retained_space, values, compilation_id = retained
        retained_space.require_exact_plan(
            space.construction_plan, numeric_version=space.numeric_version
        )
        return values, compilation_id
    problem = compile_finite_element_problem(
        FiniteElementForm(
            "layer-core-harmonic-diffusion",
            "u",
            (DiffusionAction("u", 1.0),),
        ),
        space,
        constraint=dirichlet_constraint(space, "u"),
        dirichlet_values=_HarmonicBoundary(jnp.asarray(offset, dtype=jnp.float64)),
    )
    operator, rhs = problem.linear_system()
    result = solve(operator, rhs, policy=LinearSolvePolicy(DenseLU()))
    if not bool(jnp.all(result.successful)):
        raise ValueError(
            f"diffusion.solve_successful must be true for every solve; got {result.successful!r}"
        )
    values = problem.expand(result.value)
    error = float(
        jnp.max(
            jnp.abs(
                values
                - _HarmonicBoundary(jnp.asarray(offset, dtype=jnp.float64))(
                    space.dof_maps[0].dof_coordinates
                )
            )
        )
    )
    if error > 1.0e-9:
        raise ValueError(
            f"diffusion.harmonic_error must be at most 1.0e-9; got {error:.3e}"
        )
    plans.diffusion_solutions[key] = (space, values, problem.compilation_id)
    return values, problem.compilation_id


@final
class _DensityTransport(StrictModule, NonTrainableState):
    plan: UnstructuredConservativeRemapPlan
    source_volumes: Array
    target_volumes: Array
    source_gather: Array
    target_gather: Array
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: UnstructuredConservativeRemapPlan,
        source_volumes: Array,
        target_volumes: Array,
        source_gather: Array,
        target_gather: Array,
        site_id: str,
    ) -> None:
        self.plan = plan
        self.source_volumes = source_volumes
        self.target_volumes = target_volumes
        self.source_gather = source_gather
        self.target_gather = target_gather
        self.site_id = site_id

    def __call__(
        self, before: MaterialTransaction, lineage: MeshLineage, args: object
    ) -> MaterialTransaction:
        del lineage, args
        state = before.state(self.site_id)
        content = self.source_volumes * state.committed[self.source_gather]
        transferred = (
            self.plan.apply_content(content[:, None])[:, 0] / self.target_volumes
        )
        return MaterialTransaction(
            (
                MaterialState(
                    state.site_id,
                    state.model_id,
                    transferred[self.target_gather],
                    state_version=state.state_version + 1,
                ),
            )
        )


@final
class _PhysicalCertification(StrictModule, NonTrainableState):
    expected_solution: Array
    expected_damage: Array
    measures: Array
    region_labels: Array
    inventories: Array
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        solution: Array,
        damage: Array,
        measures: Array,
        regions: Array,
        inventories: Array,
        site_id: str,
    ) -> None:
        self.expected_solution = solution
        self.expected_damage = damage
        self.measures = measures
        self.region_labels = regions
        self.inventories = inventories
        self.site_id = site_id

    def __call__(
        self,
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        lineage: MeshLineage,
        args: object,
    ) -> bool:
        del mesh, lineage, args
        if materials is None:
            return False
        inventory = layer_core_region_inventory(
            self.measures * materials.state(self.site_id).committed,
            self.region_labels,
            self.inventories.shape[0],
        )
        solution_defect = jnp.max(jnp.abs(fields[0] - self.expected_solution))
        damage_defect = jnp.max(jnp.abs(fields[1] - self.expected_damage))
        inventory_defect = jnp.max(jnp.abs(inventory - self.inventories))
        return bool(
            jnp.all(jnp.isfinite(fields[0]))
            & jnp.all(jnp.isfinite(fields[1]))
            & (solution_defect <= 1.0e-9)
            & (damage_defect <= 1.0e-9)
            & jnp.all((fields[1] >= 0.0) & (fields[1] <= 1.0))
            & (inventory_defect <= 1.0e-12)
        )


@checked
def layer_core_cell_measures(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> np.ndarray:
    """Certified mapped cell measures in mesh block order."""
    space = _finite_volume(result, plans=plans)
    rows = dict(
        zip(
            np.asarray(space.cell_global_ids).tolist(),
            np.asarray(space.cell_volumes).tolist(),
            strict=True,
        )
    )
    return np.asarray(
        [rows[identifier] for identifier in _cell_ids(result).tolist()], dtype=np.float64
    )


@checked
def layer_core_region_labels(result: CellMeshingResult, /) -> np.ndarray:
    report = result.certification
    if report is None or report.request.cell_regions is None:
        raise ValueError(
            f"continuation.cell_regions must be present in certification; got {report!r}"
        )
    if report.request.cell_global_ids != tuple(_cell_ids(result).tolist()):
        raise ValueError(
            f"material.cell_global_ids must equal mesh block cell order {tuple(_cell_ids(result).tolist())!r}; got {report.request.cell_global_ids!r}"
        )
    return np.asarray(report.request.cell_regions, dtype=np.int64)


def layer_core_region_inventory(
    content: Array, labels: Array | np.ndarray, region_count: int, /
) -> Array:
    return (
        jnp.zeros((region_count,), dtype=jnp.float64)
        .at[jnp.asarray(labels, dtype=jnp.int32)]
        .add(content)
    )


def _density_transport(
    adaptation: MeshAdaptationResult,
    site_id: str,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
) -> _DensityTransport:
    first = _finite_volume(adaptation.source, plans=plans)
    second = _finite_volume(adaptation.target, plans=plans)
    provenance = (
        "layer-core-inverse-coarsening"
        if adaptation.coarsening_witnesses is not None
        else "layer-core-refinement"
    )
    plan = prepare_layer_core_remap(
        first,
        second,
        adaptation,
        provenance,
        plans=plans,
    )
    source_rows = {
        identifier: row
        for row, identifier in enumerate(_cell_ids(adaptation.source).tolist())
    }
    target_rows = {
        identifier: row
        for row, identifier in enumerate(np.asarray(second.cell_global_ids).tolist())
    }
    source_gather = np.asarray(
        [
            source_rows[int(identifier)]
            for identifier in np.asarray(first.cell_global_ids)
        ],
        dtype=np.int64,
    )
    target_gather = np.asarray(
        [target_rows[int(identifier)] for identifier in _cell_ids(adaptation.target)],
        dtype=np.int64,
    )
    constant = plan.apply_content(first.cell_volumes[:, None])[:, 0] / second.cell_volumes
    if not np.allclose(constant, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"material.remapped_constant must equal one within 1.0e-12; got {constant!r}"
        )
    return _DensityTransport(
        plan,
        first.cell_volumes,
        second.cell_volumes,
        jnp.asarray(source_gather, dtype=jnp.int32),
        jnp.asarray(target_gather, dtype=jnp.int32),
        site_id,
    )


def _same_coordinate_map(
    source: CellMeshingResult, restored: CellMeshingResult, /
) -> bool:
    """Bitwise-equal elements, routes, coefficients and periodic source map.

    The restored geometry additionally records the coarsening restriction that
    produced it; that provenance is not part of the coordinate map.
    """
    certification = source.certification
    if certification is None:
        raise ValueError(
            "qualification.coordinate_map_certificate must retain source request limits; got None"
        )
    source_geometry, restored_geometry = (source.geometry, restored.geometry)
    with coordinate_certificate_request(certification.request.limits) as budget:
        return (
            source_geometry.block_names == restored_geometry.block_names
            and all(
                (
                    first.element_id == second.element_id
                    for first, second in zip(
                        source_geometry.elements, restored_geometry.elements, strict=True
                    )
                )
            )
            and all(
                (
                    np.array_equal(np.asarray(first), np.asarray(second))
                    for first, second in zip(
                        source_geometry.geometry_dofs,
                        restored_geometry.geometry_dofs,
                        strict=True,
                    )
                )
            )
            and np.array_equal(
                np.asarray(source_geometry.coordinates).view(np.uint64),
                np.asarray(restored_geometry.coordinates).view(np.uint64),
            )
            and (
                source_geometry.source_coordinates(budget=budget)
                == restored_geometry.source_coordinates(budget=budget)
            )
            and bool(
                eqx.tree_equal(
                    source_geometry.periodic_source, restored_geometry.periodic_source
                )
            )
        )


@checked
def require_restored_layer_core_result_identity(
    source: CellMeshingResult, restored: CellMeshingResult, /
) -> None:
    source_report = source.certification
    restored_report = restored.certification
    if source_report is None or restored_report is None:
        raise ValueError(
            f"restoration.certification must be present for source and restored results; got {(source_report, restored_report)!r}"
        )
    source_request = source_report.request
    restored_request = restored_report.request
    if (
        source.mesh.topology_id != restored.mesh.topology_id
        or tuple(block.block_id for block in source.mesh.blocks)
        != tuple(block.block_id for block in restored.mesh.blocks)
        or not np.array_equal(
            np.asarray(source.mesh.vertex_global_ids),
            np.asarray(restored.mesh.vertex_global_ids),
        )
        or (
            source_request.source_id,
            source_request.source_revision,
            source_request.domain_id,
        )
        != (
            restored_request.source_id,
            restored_request.source_revision,
            restored_request.domain_id,
        )
        or not _same_coordinate_map(source, restored)
    ):
        raise ValueError(
            f"restoration.result_identity must preserve mesh, coordinate map, and source revision; got restored result {restored.result_id!r} for source {source.result_id!r}"
        )


def layer_core_history_field_templates(
    cell_kind: str, /
) -> tuple[tuple[str, FiniteElementFieldSpec], ...]:
    """Return the quadratic diffusion and irreversible damage-history declarations."""
    element = lagrange_element(cell_kind, 2)
    return (
        ("history:intensive", FiniteElementFieldSpec("u", element)),
        (
            "history:material-history",
            FiniteElementFieldSpec("damage-history", element),
        ),
    )


def _history_fields(result: CellMeshingResult, /) -> tuple[FiniteElementFieldSpec, ...]:
    templates = {
        cell_kind: {
            field.name: field
            for _, field in layer_core_history_field_templates(cell_kind)
        }
        for cell_kind in {block.cell_kind for block in result.mesh.blocks}
    }
    return tuple(
        FiniteElementFieldSpec(
            field.name,
            {
                block.name: templates[block.cell_kind][field.name].elements[0]
                for block in result.mesh.blocks
            },
            component_shape=field.component_shape,
        )
        for field in next(iter(templates.values())).values()
    )


def _history_space(
    result: CellMeshingResult,
    /,
    *,
    plans: LayerCoreLifecyclePlans,
    reuse_source: CellMeshingResult | None = None,
) -> FiniteElementDiscretization:
    plan = FiniteElementPlan(
        result.mesh, _history_fields(result), coordinate_spec=result.geometry
    )
    retained = plans.history_spaces.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError(
                f"history.cached_result_id must equal {result.result_id!r}; got {retained_result.result_id!r}"
            )
        retained_space.require_exact_plan(plan)
        return retained_space
    if reuse_source is not None:
        require_restored_layer_core_result_identity(reuse_source, result)
        source_entry = plans.history_spaces.get(reuse_source.result_id)
        if source_entry is None:
            raise ValueError(
                f"history.source_preparation must be cached for {reuse_source.result_id!r}; got None"
            )
        source_space = source_entry[1]
        source_space.require_exact_plan(plan)
        plans.history_spaces[result.result_id] = (result, source_space)
        return source_space
    transfer_entry = plans.transfer_spaces.get(result.result_id)
    if transfer_entry is None:
        prepared = plan.prepare()
    else:
        transfer_space = transfer_entry[1]
        h1 = transfer_space.field_index("H1")
        dof_map = transfer_space.dof_maps[h1]
        prepared = FiniteElementDiscretization(plan, dof_maps=(dof_map, dof_map))
    plans.history_spaces[result.result_id] = (result, prepared)
    return prepared


def _history_field_transfers(
    adaptation: MeshAdaptationResult,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    plans: LayerCoreLifecyclePlans,
    /,
) -> tuple[FiniteElementFieldTransfer, ...] | None:
    retained = plans.field_transfers.get((adaptation.result_id, "H1"))
    if retained is None:
        return None
    return tuple(
        FiniteElementFieldTransfer(
            retained.transfer,
            source.field_spaces[index],
            target.field_spaces[index],
            retained.geometry,
            retained.evidence,
            source_measures=retained.source_measures,
            target_measures=retained.target_measures,
        )
        for index in range(len(source.field_spaces))
    )


@checked
def continue_layer_core_history(
    source: CellMeshingResult,
    refinement: MeshAdaptationResult,
    coarsening: MeshAdaptationResult,
    /,
    *,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans | None = None,
) -> LayerCoreHistoryContinuation:
    """Atomic PDE, damage-history and material continuation with changed-PDE rollback."""
    plans_ = LayerCoreLifecyclePlans() if plans is None else plans
    h1_ready = plans_._readiness.get("h1")
    finite_volume_ready = plans_._readiness.get("volume")
    region_densities = layer_core_material_values(source, material)
    layer_core_material_values(refinement.target, material)
    layer_core_material_values(coarsening.target, material)
    site_id = material["site_id"]
    fields = _history_fields(source)
    space = _history_space(source, plans=plans_)
    solution, compilation = _solve_diffusion(space, 0.0, plans=plans_)
    damage = space.project("damage-history", _damage)
    if finite_volume_ready is not None:
        finite_volume_ready.wait()
    measures = layer_core_cell_measures(source, plans=plans_)
    regions = layer_core_region_labels(source)
    densities = jnp.asarray(region_densities[regions])
    inventories = layer_core_region_inventory(
        jnp.asarray(measures) * densities, regions, region_densities.size
    )
    materials = MaterialTransaction(
        (
            MaterialState(
                MaterialSiteId(site_id),
                "physical-cell-average-density",
                densities,
            ),
        )
    )
    accepted = FiniteElementAcceptedState(
        (solution, damage),
        0.0,
        0,
        source.mesh.topology_id,
        space.prepared_id,
        compilation,
        materials=materials,
    )
    stages: list[LayerCoreHistoryStage] = []
    current_space = space
    for phase, adaptation in (("refine", refinement), ("coarsen", coarsening)):
        target = adaptation.target
        target_space = _history_space(
            target,
            plans=plans_,
            reuse_source=source if phase == "coarsen" else None,
        )
        independent, target_compilation = _solve_diffusion(
            target_space, 0.0, plans=plans_
        )
        shifted, _ = _solve_diffusion(target_space, 0.1, plans=plans_)
        expected_damage = target_space.project("damage-history", _damage)
        target_measures = layer_core_cell_measures(target, plans=plans_)
        target_regions = layer_core_region_labels(target)
        transport = _density_transport(adaptation, site_id, plans=plans_)

        def transaction(solution_: Array, /) -> FiniteElementTopologyTransaction:
            return FiniteElementTopologyTransaction(
                _PhysicalCertification(
                    solution_,
                    expected_damage,
                    jnp.asarray(target_measures, dtype=jnp.float64),
                    jnp.asarray(target_regions, dtype=jnp.int32),
                    inventories,
                    site_id,
                ),
                fields=fields,
                material_transfer=transport,
            )

        if h1_ready is not None:
            h1_ready.wait()
        committing = transaction(independent)
        preparation = committing.prepare_transition(
            accepted,
            adaptation.source.mesh,
            adaptation,
            source=current_space,
            target=target_space,
            field_transfers=_history_field_transfers(
                adaptation, current_space, target_space, plans_
            ),
        )
        refused = transaction(shifted).execute(
            accepted,
            adaptation.source.mesh,
            adaptation,
            preparation=preparation,
        )
        if (
            bool(refused.committed)
            or refused.state is not accepted
            or refused.mesh is not adaptation.source.mesh
        ):
            raise ValueError(
                f"history.changed_pde_rollback must retain accepted state and mesh without committing; got {(bool(refused.committed), refused.state is accepted, refused.mesh is adaptation.source.mesh)!r}"
            )
        committed = committing.execute(
            accepted,
            adaptation.source.mesh,
            adaptation,
            compiled_layout_id=target_compilation,
            preparation=preparation,
        )
        if (
            not bool(committed.committed)
            or committed.receipt is None
            or not committed.receipt.published
        ):
            raise ValueError(
                f"history.transaction must commit with a published transaction record; got {(bool(committed.committed), committed.receipt)!r}: {committed.diagnostics}"
            )
        if not all(transfer.evidence.passed for transfer in committed.transfers):
            raise ValueError(
                f"history.transfer_evidence must pass for every transfer; got {tuple(transfer.evidence.passed for transfer in committed.transfers)!r}"
            )
        accepted = committed.state
        current_space = target_space
        if accepted.materials is None:
            raise ValueError(
                "history.committed_materials must retain material state; got None"
            )
        inventory = layer_core_region_inventory(
            jnp.asarray(target_measures) * accepted.materials.state(site_id).committed,
            target_regions,
            region_densities.size,
        )
        stages.append(
            {
                "phase": phase,
                "cells": int(_cell_ids(target).size),
                "changed_pde_rolled_back": True,
                "pde_defect": float(jnp.max(jnp.abs(accepted.fields[0] - independent))),
                "damage_defect": float(
                    jnp.max(jnp.abs(accepted.fields[1] - expected_damage))
                ),
                "inventory_defect": float(jnp.max(jnp.abs(inventory - inventories))),
            }
        )
    if coarsening.target.mesh.topology_id != source.mesh.topology_id:
        raise ValueError(
            f"coarsening.topology_id must equal source topology {source.mesh.topology_id!r}; got {coarsening.target.mesh.topology_id!r}"
        )
    if not _same_coordinate_map(source, coarsening.target):
        raise ValueError(
            f"coarsening.coordinate_map must equal the source map; got differing map on result {coarsening.target.result_id!r}"
        )
    if not np.allclose(accepted.fields[1], damage, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"history.restored_damage must equal initial damage within 1.0e-9; got {accepted.fields[1]!r} for {damage!r}"
        )
    return {
        "initial_cells": int(_cell_ids(source).size),
        "region_inventories": np.asarray(inventories).tolist(),
        "stages": stages,
    }


@final
class LayerCoreLifecycleReport(StrictModule):
    """Source-bound conservation, restoration, rollback, and ended resource evidence."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    material_declaration: LayerCoreMaterialDeclaration = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    coarsened_geometry_id: str = eqx.field(static=True)
    restored_coordinate_map_bitwise: bool = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)
    coarsening_id: str = eqx.field(static=True)
    cells: tuple[int, ...] = eqx.field(static=True)
    stages_seconds: tuple[tuple[str, float], ...] = eqx.field(static=True)
    fields: tuple[LayerCoreFieldContinuation, ...] = eqx.field(static=True)
    finite_volume: LayerCoreVolumeContinuation = eqx.field(static=True)
    history: LayerCoreHistoryContinuation = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord
    limits_id: str = eqx.field(static=True)

    def to_records(self) -> tuple[dict[str, object], ...]:
        """Return the five diagnostic JSON records in continuation publication order."""
        return (
            {
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "material_declaration": self.material_declaration,
                "source_mesh_id": self.source_mesh_id,
                "source_geometry_id": self.source_geometry_id,
                "coarsened_geometry_id": self.coarsened_geometry_id,
                "restored_coordinate_map_bitwise": self.restored_coordinate_map_bitwise,
                "refinement_id": self.refinement_id,
                "coarsening_id": self.coarsening_id,
                "cells": list(self.cells),
                "stages_seconds": dict(self.stages_seconds),
            },
            {"fields": list(self.fields)},
            {"finite_volume": self.finite_volume},
            {"history": self.history},
            {
                "execution_evidence": self.execution_evidence.to_record(),
                "limits_id": self.limits_id,
            },
        )


def _run_continuation_workers(
    source: CellMeshingResult,
    refinement: MeshAdaptationResult,
    coarsening: MeshAdaptationResult,
    material: LayerCoreMaterialDeclaration,
    plans: LayerCoreLifecyclePlans,
    stages_seconds: dict[str, float],
    /,
    *,
    execution: NativeExecutionScope,
) -> tuple[
    list[LayerCoreFieldContinuation],
    LayerCoreVolumeContinuation,
    LayerCoreHistoryContinuation,
]:
    """Share authenticated preparations under equal reservations of the remaining allowance."""
    spaces_ready, h1_ready, finite_volume_ready = Event(), Event(), Event()
    plans._readiness.update(spaces=spaces_ready, h1=h1_ready, volume=finite_volume_ready)

    def run_fields() -> list[LayerCoreFieldContinuation]:
        started = perf_counter()
        try:
            return continue_layer_core_fields(
                source,
                refinement,
                coarsening,
                plans=plans,
            )
        finally:
            spaces_ready.set()
            h1_ready.set()
            stages_seconds["compatible_field_continuation"] = perf_counter() - started

    def run_volume() -> LayerCoreVolumeContinuation:
        started = perf_counter()
        try:
            return continue_layer_core_finite_volume(
                source,
                refinement,
                coarsening,
                material=material,
                plans=plans,
            )
        finally:
            finite_volume_ready.set()
            stages_seconds["finite_volume_continuation"] = perf_counter() - started

    def run_history() -> LayerCoreHistoryContinuation:
        started = perf_counter()
        try:
            return continue_layer_core_history(
                source,
                refinement,
                coarsening,
                material=material,
                plans=plans,
            )
        finally:
            stages_seconds["history_and_pde_continuation"] = perf_counter() - started

    def run_deferred[T](ticket: NativeBatchTicket, call: Callable[[], T], /) -> T:
        with ticket, ticket.native_target():
            return call()

    worker_count = 3
    remaining = execution.remaining()
    work_upper = remaining.remaining_work_units // worker_count
    queries_upper = remaining.remaining_geometry_queries // worker_count
    bytes_upper = remaining.remaining_managed_bytes // worker_count
    cavity_upper = remaining.remaining_cavity_cells // worker_count
    # No aggregate worker bounds are declared by these continuations. Partition
    # only the remaining request allowance; keep division remainders at the root.
    with execution.batch(
        worker_count * work_upper,
        queries_upper=worker_count * queries_upper,
        bytes_upper=worker_count * bytes_upper,
        cavity_upper=worker_count * cavity_upper,
    ) as fanout:
        tickets: list[NativeBatchTicket] = []
        try:
            for _ in range(worker_count):
                tickets.append(
                    fanout.batch(
                        work_upper,
                        queries_upper=queries_upper,
                        bytes_upper=bytes_upper,
                        cavity_upper=cavity_upper,
                    )
                )
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                fields_future = pool.submit(lambda: run_deferred(tickets[0], run_fields))
                spaces_ready.wait()
                volume_future = pool.submit(lambda: run_deferred(tickets[1], run_volume))
                history_future = pool.submit(
                    lambda: run_deferred(tickets[2], run_history)
                )
                fields = fields_future.result()
                volume = volume_future.result()
                history = history_future.result()
        finally:
            for ticket in tickets:
                if not ticket.settled:
                    ticket.settle()
    return fields, volume, history


def continue_layer_core_lifecycle(
    archive_path: str | Path,
    content_id: str,
    /,
    *,
    region_densities: tuple[float, ...] = LAYER_CORE_REGION_DENSITIES,
    material_site: str = "two-material-density",
) -> LayerCoreLifecycleReport:
    """Continue an authenticated layer/core archive under its source domain limits.

    Field, volume, and history workers share immutable preparations and equally
    reserved remaining resources. Refused transfers and changed-PDE rollback
    retain their scientific diagnostics and the ended execution record.
    """
    stages_seconds: dict[str, float] = {}
    phase_started = perf_counter()
    archive = _read_meshing_source_archive(archive_path, expected_content_id=content_id)
    stages_seconds["archive_member_authentication"] = perf_counter() - phase_started
    phase_started = perf_counter()
    controls = read_meshing_source_execution_controls(
        archive_path,
        expected_content_id=content_id,
        _authenticated_archive=archive,
    )
    stages_seconds["execution_control_restoration"] = perf_counter() - phase_started
    previous = controls.execution_evidence
    if previous.owner_id is None:
        raise ValueError(
            "continuation.execution_owner_id must retain the archived execution owner; got None"
        )
    with (
        NativeExecutionScope(controls.limits.native_limits()) as execution,
        native_volume_source_execution_budget(
            controls.limits, previous, previous.owner_id, scope=execution
        ),
    ):
        phase_started = perf_counter()
        records = _restore_meshing_source_archive(
            archive, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        )
        stages_seconds["scientific_owner_restoration"] = perf_counter() - phase_started
        source, refinement, coarsening = accepted_layer_core_lineage(records)
        authored = records["generation_source"]
        specification = records["generation_specification"]
        if not isinstance(authored, NativeLayerCoreSource):
            raise TypeError(
                f"closure.generation_source must be a NativeLayerCoreSource; got {type(authored).__name__}"
            )
        if (
            authored.source_id,
            authored.source_revision,
            authored.binding_id,
            specification.specification_id,
        ) != (
            controls.source_id,
            controls.source_revision,
            controls.source_binding_id,
            controls.specification_id,
        ):
            raise ValueError(
                f"continuation.source_controls must match authenticated controls; got {(authored.source_id, authored.source_revision, authored.binding_id, specification.specification_id)!r} for {(controls.source_id, controls.source_revision, controls.source_binding_id, controls.specification_id)!r}"
            )
        material = layer_core_material_declaration(
            source, densities=region_densities, site_id=material_site
        )
        plans = LayerCoreLifecyclePlans()
        fields, volume, history = _run_continuation_workers(
            source,
            refinement,
            coarsening,
            material,
            plans,
            stages_seconds,
            execution=execution,
        )
        source_geometry_id = cell_geometry_id(source.geometry)
        coarsened_geometry_id = cell_geometry_id(coarsening.target.geometry)
        restored_coordinate_map_bitwise = _same_coordinate_map(source, coarsening.target)
        cells = tuple(
            _cell_ids(result).size
            for result in (source, refinement.target, coarsening.target)
        )
    if execution.evidence is None:
        raise RuntimeError(
            "continuation.execution_evidence must retain ended scope evidence; got None"
        )
    latest = NativeExecutionRecord(
        execution.evidence, preparation_evidence=previous, owner_id=previous.owner_id
    )
    return LayerCoreLifecycleReport(
        source_id=authored.source_id,
        source_revision=authored.source_revision,
        material_declaration=material,
        source_mesh_id=source.mesh.mesh_id,
        source_geometry_id=source_geometry_id,
        coarsened_geometry_id=coarsened_geometry_id,
        restored_coordinate_map_bitwise=restored_coordinate_map_bitwise,
        refinement_id=refinement.result_id,
        coarsening_id=coarsening.result_id,
        cells=cells,
        stages_seconds=tuple(sorted(stages_seconds.items())),
        fields=tuple(fields),
        finite_volume=volume,
        history=history,
        execution_evidence=latest,
        limits_id=controls.limits.limits_id,
    )
