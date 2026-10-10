#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Durable layer/core lifecycle continuation: fields, finite volumes, history, PDE.

The consumer reads one accepted native layer/core source together with the
refinement and the inverse coarsening that were archived on that same source
lineage. It never regenerates the source or invents a refined cell count:

* H1 Q2, DG1, H(curl) degree 2 and H(div) degree 2 fields cross the refinement
  and its inverse through the adaptation's own parent, geometry-transition and
  coarsening witnesses, with exact restoration, adjoint and physical-content
  evidence;
* finite-volume cell averages of a two-material density move through the
  mapped nested conservative remap, preserving every material inventory;
* an irreversible damage history, the material density and a harmonic
  diffusion solution are carried by the atomic finite-element topology
  transaction, whose changed-PDE certificate must roll back unchanged.

Run ``python -m tools.layer_core_lifecycle_continuation ARCHIVE CONTENT_ID``
on a durable meshing-source closure whose ``accepted_data`` holds the
``(refinement, coarsening)`` adaptation results.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from threading import Event
from time import perf_counter
from typing import Any, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


jax.config.update("jax_enable_x64", True)
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._constraints import dirichlet_constraint
from phydrax.discretization.fem._form_elements import form_element
from phydrax.discretization.fem._generic import (
    _degree_aware_reference_rule,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementTransferDiscretization,
)
from phydrax.discretization.fem._reference import (
    discontinuous_element,
    FiniteElementSpec,
    lagrange_element,
)
from phydrax.discretization.fem._topology_transfer import (
    FiniteElementFieldTransfer,
    prepare_nested_compatible_field_transfers,
    prepare_nested_field_transfer,
)
from phydrax.discretization.finite_volume._automatic_remap import (
    prepare_unstructured_conservative_remap,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from phydrax.equations._finite_element_material import (
    MaterialSiteId,
    MaterialState,
    MaterialTransaction,
)
from phydrax.equations._finite_element_variational import (
    compile_finite_element_problem,
    FiniteElementForm,
)
from phydrax.equations._variational import DiffusionAction
from phydrax.linalg._policies import DenseLU, LinearSolvePolicy
from phydrax.linalg._runtime import solve
from phydrax.meshing._adaptation import (
    MeshAdaptationResult,
    MeshAdaptationStatus,
)
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._lineage import MeshLineage
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers._native_sources import NativeLayerCoreSource
from phydrax.solver._finite_element_adaptivity import (
    FiniteElementAcceptedState,
    FiniteElementTopologyTransaction,
)


class FieldContinuation(TypedDict):
    field: str
    dofs: list[int]
    roundtrip_error: float
    physical_error: float
    forward_defects: dict[str, float]
    reverse_defects: dict[str, float]


class VolumeContinuation(TypedDict):
    cells: list[int]
    region_inventories: list[float]
    defects: dict[str, float]


class HistoryStage(TypedDict):
    phase: str
    cells: int
    changed_pde_rolled_back: bool
    pde_defect: float
    damage_defect: float
    inventory_defect: float


class HistoryContinuation(TypedDict):
    initial_cells: int
    region_inventories: list[float]
    stages: list[HistoryStage]


FIELD_KINDS = ("H1", "DG", "Hcurl", "Hdiv")
REGION_DENSITIES = (2.0, 3.0)


@dataclass
class _LifecyclePlanCache:
    """Authenticated immutable preparations shared by one cold continuation."""

    transfer_spaces: dict[
        str, tuple[CellMeshingResult, FiniteElementTransferDiscretization]
    ] = field(default_factory=dict)
    history_spaces: dict[str, tuple[CellMeshingResult, FiniteElementDiscretization]] = (
        field(default_factory=dict)
    )
    field_transfers: dict[tuple[str, str], FiniteElementFieldTransfer] = field(
        default_factory=dict
    )
    finite_volumes: dict[
        str, tuple[CellMeshingResult, UnstructuredFiniteVolumeDiscretization]
    ] = field(default_factory=dict)
    remaps: dict[tuple[str, str], UnstructuredConservativeRemapPlan] = field(
        default_factory=dict
    )
    diffusion_solutions: dict[
        tuple[str, float], tuple[FiniteElementDiscretization, Array, str]
    ] = field(default_factory=dict)


class MaterialDeclaration(TypedDict):
    source_id: str
    source_revision: str
    domain_id: str
    region_ids: tuple[str, ...]
    densities: tuple[float, ...]
    site_id: str


def material_declaration(
    source: CellMeshingResult,
    /,
    *,
    densities: tuple[float, ...] = REGION_DENSITIES,
    site_id: str = "two-material-density",
) -> MaterialDeclaration:
    report = source.certification
    if report is None or report.coverage is None or report.request.domain is None:
        raise ValueError(
            "Material continuation requires its certified original source and region namespace."
        )
    declaration: MaterialDeclaration = {
        "source_id": report.request.domain.source_id,
        "source_revision": report.request.domain.source_revision,
        "domain_id": report.request.domain.domain_id,
        "region_ids": report.coverage.region_ids,
        "densities": densities,
        "site_id": site_id,
    }
    _material_values(source, declaration)
    return declaration


def _material_values(
    result: CellMeshingResult, declaration: MaterialDeclaration, /
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
            "Material continuation requires current source-bound region certification."
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
            "Material declaration binds another original source or region namespace."
        )
    values = np.asarray(declaration["densities"], dtype=np.float64)
    if (
        values.shape != (len(declaration["region_ids"]),)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            "Material densities must be finite, positive and cover exactly the original regions."
        )
    MaterialSiteId(declaration["site_id"])
    labels = _region_labels(result)
    if np.any(labels < 0) or np.any(labels >= values.size):
        raise ValueError("Material cell ownership leaves its original region namespace.")
    return values


def accepted_lineage(
    records: object, /
) -> tuple[CellMeshingResult, MeshAdaptationResult, MeshAdaptationResult]:
    """The archived source, refinement and inverse coarsening of one lineage."""
    if not isinstance(records, Mapping):
        raise TypeError("A durable meshing-source closure is a named record mapping.")
    part = records["generation_part"]
    accepted = records["accepted_data"]
    if not isinstance(part, MeshPart) or not isinstance(part.carrier, CellMeshingResult):
        raise TypeError("The closure must retain its accepted native generation part.")
    if not isinstance(accepted, Mapping):
        raise TypeError("The closure must retain its accepted adaptation records.")
    refinement, coarsening = accepted["adaptation_results"]
    source = part.carrier
    if not isinstance(refinement, MeshAdaptationResult) or not isinstance(
        coarsening, MeshAdaptationResult
    ):
        raise TypeError("Accepted adaptation records must be actual adaptation results.")
    if (
        refinement.source.result_id != source.result_id
        or coarsening.source.result_id != refinement.target.result_id
    ):
        raise ValueError(
            "The archived adaptations do not continue the archived source lineage."
        )
    for stage in (refinement, coarsening):
        if stage.status is not MeshAdaptationStatus.COMPLETE:
            raise ValueError("Refused or partial adaptations cannot be continued.")
    for result in (source, refinement.target, coarsening.target):
        if result.certification is None or not result.certification.passed:
            raise ValueError("Every continued mesh must carry its passed certification.")
    return source, refinement, coarsening


def _cell_ids(result: CellMeshingResult, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )


def _cell_regions(result: CellMeshingResult, /) -> dict[int, int]:
    report = result.certification
    if report is None or report.request.cell_regions is None:
        raise ValueError("Continued material inventories require certified cell regions.")
    labels = np.asarray(report.request.cell_regions, dtype=np.int64)
    return dict(zip(_cell_ids(result).tolist(), labels.tolist(), strict=True))


@cache
def _field_element(kind: str, cell_kind: str, /) -> FiniteElementSpec:
    match kind:
        case "H1":
            return lagrange_element(cell_kind, 2)
        case "DG":
            return discontinuous_element(cell_kind, 1)
        case "Hcurl":
            return form_element(
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
            return form_element(
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
            raise ValueError(f"Unknown continued field kind {kind!r}.")


def _field_elements(
    result: CellMeshingResult, kind: str, /
) -> dict[str, FiniteElementSpec]:
    return {
        block.name: _field_element(kind, block.cell_kind) for block in result.mesh.blocks
    }


type _FieldSpace = FiniteElementDiscretization | FiniteElementTransferDiscretization


def _transfer_space(
    result: CellMeshingResult,
    /,
    *,
    plans: _LifecyclePlanCache,
    reuse_prepared: FiniteElementTransferDiscretization | None = None,
    reuse_source: CellMeshingResult | None = None,
) -> FiniteElementTransferDiscretization:
    if (reuse_prepared is None) != (reuse_source is None):
        raise ValueError(
            "Transfer-space reuse requires both its preparation and source result."
        )
    fields = tuple(
        FiniteElementFieldSpec(kind, _field_elements(result, kind))
        for kind in FIELD_KINDS
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
            raise ValueError("Transfer-space cache changed its exact result identity.")
        retained_space.require_exact_plan(plan)
        return retained_space
    if reuse_source is not None:
        _require_restored_result_identity(reuse_source, result)
    prepared = plan.prepare_transfer(reuse_prepared=reuse_prepared)
    plans.transfer_spaces[result.result_id] = (result, prepared)
    return prepared


type _CellRoute = tuple[int, FiniteElementSpec, np.ndarray, np.ndarray]


def _cells(space: _FieldSpace, /, *, field_name: str | None = None) -> list[_CellRoute]:
    index = 0 if field_name is None else space._field_index(field_name)
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


def _scalar_content(
    space: _FieldSpace,
    values: np.ndarray,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    """Physical integral through the actual mapped coordinate Jacobians."""
    index = 0 if field_name is None else space._field_index(field_name)
    total = np.zeros(values.shape[1:], dtype=np.float64)
    runtime = np.asarray(space.default_runtime.coordinates)
    for element, coordinate, routes, geometry_routes in zip(
        space.elements[index],
        space.coordinate_elements,
        space.dof_maps[index].cell_dofs,
        space.coordinate_dofs,
        strict=True,
    ):
        # Refined and restored blocks keep the exact source basis restricted or
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
            raise TypeError("Continued content integrates tabulated coordinate maps.")
        points, weights = _degree_aware_reference_rule(element.cell_kind, 8)
        basis = np.asarray(element.tabulate(points)[0])
        gradients = np.asarray(coordinate.tabulate(points)[1])
        for route, geometry_route in zip(
            np.asarray(routes), np.asarray(geometry_routes), strict=True
        ):
            determinant = np.linalg.det(
                np.einsum("qna,nd->qda", gradients, runtime[geometry_route])
            )
            if np.any(determinant <= 0.0):
                raise ValueError("Continued content requires positive mapped Jacobians.")
            total += np.einsum(
                "q,qn,np->p", np.asarray(weights) * determinant, basis, values[route]
            )
    return total


def _divergence_content(
    space: _FieldSpace,
    values: np.ndarray,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    """Total flux content; the contravariant Piola map makes it reference exact."""
    total = np.zeros(values.shape[1:], dtype=np.float64)
    for _, element, route, transform in _cells(space, field_name=field_name):
        points, weights = _degree_aware_reference_rule(element.cell_kind, 8)
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


def _edge_circulation_defect(
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
        raise ValueError("Inverse coarsening must retain its nested reference witnesses.")
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
            raise ValueError("A restored coarse cell has no witnessed fine children.")
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


def _initial_values(
    space: _FieldSpace,
    kind: str,
    /,
    *,
    field_name: str | None = None,
) -> np.ndarray:
    index = 0 if field_name is None else space._field_index(field_name)
    points = np.asarray(space.dof_maps[index].dof_coordinates)
    if kind not in ("H1", "DG"):
        return np.random.default_rng(19).normal(
            size=(space.dof_maps[index].global_dof_count, 2)
        )
    # A physical affine field pulls back through the quadratic fiber graph into
    # H1 Q2 exactly. A physical quadratic would compose to degree four and is
    # not an admissible exact-reproduction oracle for this curved geometry.
    return np.column_stack(
        (
            1.0 + points[:, 1] + points[:, 2],
            0.5 + 0.125 * points[:, 1] + 0.1 * points[:, 2],
        )
    )


def continue_fields(
    source: CellMeshingResult,
    refinement: MeshAdaptationResult,
    coarsening: MeshAdaptationResult,
    /,
    *,
    plans: _LifecyclePlanCache | None = None,
    spaces_ready: Event | None = None,
    h1_ready: Event | None = None,
) -> list[FieldContinuation]:
    """Refine and restore H1/DG/H(curl)/H(div) fields through the archived witnesses."""
    plans_ = _LifecyclePlanCache() if plans is None else plans
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
    for kind in ("H1", "DG"):
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
    compatible = ("Hcurl", "Hdiv")
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
    summaries: list[FieldContinuation] = []
    for kind in FIELD_KINDS:
        forward_transfer = forward[kind]
        reverse_transfer = reverse[kind]
        if not forward_transfer.evidence.passed or not reverse_transfer.evidence.passed:
            raise ValueError(f"{kind} transfer evidence did not pass.")
        values = _initial_values(first, kind, field_name=kind)
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
                expected = _initial_values(fine, kind, field_name=kind)
                physical = float(np.max(np.abs(transferred - expected)))
            case "DG":
                physical = float(
                    np.max(
                        np.abs(
                            _scalar_content(last, projected, field_name=kind)
                            - _scalar_content(fine, arbitrary, field_name=kind)
                        )
                    )
                )
            case "Hdiv":
                physical = float(
                    np.max(
                        np.abs(
                            _divergence_content(last, projected, field_name=kind)
                            - _divergence_content(fine, arbitrary, field_name=kind)
                        )
                    )
                )
            case _:
                physical = _edge_circulation_defect(
                    fine,
                    last,
                    arbitrary,
                    projected,
                    coarsening,
                    field_name=kind,
                )
        if physical > 3.0e-11:
            raise ValueError(f"{kind} physical continuation defect {physical:.3e}.")
        index = first._field_index(kind)
        fine_index = fine._field_index(kind)
        last_index = last._field_index(kind)
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
    plans: _LifecyclePlanCache,
) -> UnstructuredFiniteVolumeDiscretization:
    retained = plans.finite_volumes.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError("Finite-volume cache changed its exact result identity.")
        return retained_space
    report = result.certification
    if report is None or report.embedding is None:
        raise ValueError("Finite-volume continuation requires global embedding evidence.")
    periodic = result.mesh.periodic_topology is not None
    periodic_geometry = result.geometry if periodic else None
    periodic_embedding = report.embedding if periodic else None
    space = UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        periodic_geometry=periodic_geometry,
        periodic_embedding=periodic_embedding,
    ).prepare(
        cell_geometry=result.geometry,
        periodic_embedding=periodic_embedding,
    )
    plans.finite_volumes[result.result_id] = (result, space)
    return space


def _remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    stage: MeshAdaptationResult,
    provenance: str,
    /,
    *,
    plans: _LifecyclePlanCache | None = None,
) -> UnstructuredConservativeRemapPlan:
    plans_ = _LifecyclePlanCache() if plans is None else plans
    key = (stage.result_id, provenance)
    if key in plans_.remaps:
        plan = plans_.remaps[key]
        if (
            plan.source_topology_id != source.topology_id
            or plan.source_geometry_id != source.geometry_id
            or plan.target_topology_id != target.topology_id
            or plan.target_geometry_id != target.geometry_id
        ):
            raise ValueError("Retained conservative remap belongs to another epoch.")
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
            f"{provenance} conservative remap refused: {remap.status} {remap.reason}."
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


def continue_finite_volume(
    source: CellMeshingResult,
    refinement: MeshAdaptationResult,
    coarsening: MeshAdaptationResult,
    /,
    *,
    material: MaterialDeclaration,
    plans: _LifecyclePlanCache | None = None,
    spaces_ready: Event | None = None,
) -> VolumeContinuation:
    """Conservative declared source-material density through refinement and its inverse."""
    plans_ = _LifecyclePlanCache() if plans is None else plans
    region_densities = _material_values(source, material)
    _material_values(refinement.target, material)
    _material_values(coarsening.target, material)
    first = _finite_volume(source, plans=plans_)
    fine = _finite_volume(refinement.target, plans=plans_)
    last = _finite_volume(coarsening.target, plans=plans_)
    if spaces_ready is not None:
        spaces_ready.set()
    forward = _remap(first, fine, refinement, "layer-core-refinement", plans=plans_)
    reverse = _remap(
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
        _cell_regions(result) for result in (source, refinement.target, coarsening.target)
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
        raise ValueError(f"Finite-volume continuation lost conservation: {defects}.")
    return {
        "cells": [value.size for value in gids],
        "region_inventories": inventories.tolist(),
        "defects": defects,
    }


class _HarmonicBoundary(StrictModule, NonTrainableState):
    offset: Array

    def __init__(self, offset: float) -> None:
        self.offset = jnp.asarray(offset, dtype=jnp.float64)

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
    plans: _LifecyclePlanCache,
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
        dirichlet_values=_HarmonicBoundary(offset),
    )
    operator, rhs = problem.linear_system()
    result = solve(operator, rhs, policy=LinearSolvePolicy(DenseLU()))
    if not bool(jnp.all(result.successful)):
        raise ValueError("The continued harmonic diffusion solve was refused.")
    values = problem.expand(result.value)
    error = float(
        jnp.max(
            jnp.abs(values - _HarmonicBoundary(offset)(space.dof_maps[0].dof_coordinates))
        )
    )
    if error > 1.0e-9:
        raise ValueError(f"Continued harmonic diffusion error {error:.3e}.")
    plans.diffusion_solutions[key] = (space, values, problem.compilation_id)
    return values, problem.compilation_id


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
        source_gather: np.ndarray,
        target_gather: np.ndarray,
        site_id: str,
    ) -> None:
        self.plan = plan
        self.source_volumes = source_volumes
        self.target_volumes = target_volumes
        self.source_gather = jnp.asarray(source_gather, dtype=jnp.int32)
        self.target_gather = jnp.asarray(target_gather, dtype=jnp.int32)
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
        measures: np.ndarray,
        regions: np.ndarray,
        inventories: Array,
        site_id: str,
    ) -> None:
        self.expected_solution = solution
        self.expected_damage = damage
        self.measures = jnp.asarray(measures, dtype=jnp.float64)
        self.region_labels = jnp.asarray(regions, dtype=jnp.int32)
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
        inventory = _region_inventory(
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


def _measures(
    result: CellMeshingResult,
    /,
    *,
    plans: _LifecyclePlanCache,
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


def _region_labels(result: CellMeshingResult, /) -> np.ndarray:
    report = result.certification
    if report is None or report.request.cell_regions is None:
        raise ValueError("Continued material inventories require certified cell regions.")
    if report.request.cell_global_ids != tuple(_cell_ids(result).tolist()):
        raise ValueError(
            "Material region rows do not bind the actual source SCI cell order."
        )
    return np.asarray(report.request.cell_regions, dtype=np.int64)


def _region_inventory(
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
    plans: _LifecyclePlanCache,
) -> _DensityTransport:
    first = _finite_volume(adaptation.source, plans=plans)
    second = _finite_volume(adaptation.target, plans=plans)
    provenance = (
        "layer-core-inverse-coarsening"
        if adaptation.coarsening_witnesses is not None
        else "layer-core-refinement"
    )
    plan = _remap(
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
        raise ValueError("Continued material density has incomplete physical coverage.")
    return _DensityTransport(
        plan,
        first.cell_volumes,
        second.cell_volumes,
        source_gather,
        target_gather,
        site_id,
    )


def _same_coordinate_map(source: CellGeometrySpec, restored: CellGeometrySpec, /) -> bool:
    """Bitwise-equal elements, routes, coefficients and periodic source map.

    The restored geometry additionally records the coarsening restriction that
    produced it; that provenance is not part of the coordinate map.
    """
    return (
        source.block_names == restored.block_names
        and all(
            first.element_id == second.element_id
            for first, second in zip(source.elements, restored.elements, strict=True)
        )
        and all(
            np.array_equal(np.asarray(first), np.asarray(second))
            for first, second in zip(
                source.geometry_dofs, restored.geometry_dofs, strict=True
            )
        )
        and np.array_equal(
            np.asarray(source.coordinates).view(np.uint64),
            np.asarray(restored.coordinates).view(np.uint64),
        )
        and source.source_coordinates() == restored.source_coordinates()
        and bool(eqx.tree_equal(source.periodic_source, restored.periodic_source))
    )


def _require_restored_result_identity(
    source: CellMeshingResult, restored: CellMeshingResult, /
) -> None:
    source_report = source.certification
    restored_report = restored.certification
    if source_report is None or restored_report is None:
        raise ValueError("Restored space reuse requires both certification owners.")
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
        or not _same_coordinate_map(source.geometry, restored.geometry)
    ):
        raise ValueError(
            "Restored preparation changed its mesh, coordinate map, or source revision."
        )


def _history_fields(result: CellMeshingResult, /) -> tuple[FiniteElementFieldSpec, ...]:
    return tuple(
        FiniteElementFieldSpec(
            name,
            {
                block.name: lagrange_element(block.cell_kind, 2)
                for block in result.mesh.blocks
            },
        )
        for name in ("u", "damage-history")
    )


def _history_space(
    result: CellMeshingResult,
    /,
    *,
    plans: _LifecyclePlanCache,
    reuse_source: CellMeshingResult | None = None,
) -> FiniteElementDiscretization:
    plan = FiniteElementPlan(
        result.mesh, _history_fields(result), coordinate_spec=result.geometry
    )
    retained = plans.history_spaces.get(result.result_id)
    if retained is not None:
        retained_result, retained_space = retained
        if retained_result.result_id != result.result_id:
            raise ValueError("History-space cache changed its exact result identity.")
        retained_space.require_exact_plan(plan)
        return retained_space
    if reuse_source is not None:
        _require_restored_result_identity(reuse_source, result)
        source_entry = plans.history_spaces.get(reuse_source.result_id)
        if source_entry is None:
            raise ValueError("Restored history-space reuse lacks its source preparation.")
        source_space = source_entry[1]
        source_space.require_exact_plan(plan)
        plans.history_spaces[result.result_id] = (result, source_space)
        return source_space
    transfer_entry = plans.transfer_spaces.get(result.result_id)
    if transfer_entry is None:
        prepared = plan.prepare()
    else:
        transfer_space = transfer_entry[1]
        h1 = transfer_space._field_index("H1")
        dof_map = transfer_space.dof_maps[h1]
        prepared = FiniteElementDiscretization(plan, dof_maps=(dof_map, dof_map))
    plans.history_spaces[result.result_id] = (result, prepared)
    return prepared


def _history_field_transfers(
    adaptation: MeshAdaptationResult,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    plans: _LifecyclePlanCache,
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


def continue_history(
    source: CellMeshingResult,
    refinement: MeshAdaptationResult,
    coarsening: MeshAdaptationResult,
    /,
    *,
    material: MaterialDeclaration,
    plans: _LifecyclePlanCache | None = None,
    h1_ready: Event | None = None,
    finite_volume_ready: Event | None = None,
) -> HistoryContinuation:
    """Atomic PDE, damage-history and material continuation with changed-PDE rollback."""
    plans_ = _LifecyclePlanCache() if plans is None else plans
    region_densities = _material_values(source, material)
    _material_values(refinement.target, material)
    _material_values(coarsening.target, material)
    site_id = material["site_id"]
    fields = _history_fields(source)
    space = _history_space(source, plans=plans_)
    solution, compilation = _solve_diffusion(space, 0.0, plans=plans_)
    damage = space.project("damage-history", _damage)
    if finite_volume_ready is not None:
        finite_volume_ready.wait()
    measures = _measures(source, plans=plans_)
    regions = _region_labels(source)
    densities = jnp.asarray(region_densities[regions])
    inventories = _region_inventory(
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
    stages: list[HistoryStage] = []
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
        target_measures = _measures(target, plans=plans_)
        target_regions = _region_labels(target)
        transport = _density_transport(adaptation, site_id, plans=plans_)

        def transaction(solution_: Array, /) -> FiniteElementTopologyTransaction:
            return FiniteElementTopologyTransaction(
                _PhysicalCertification(
                    solution_,
                    expected_damage,
                    target_measures,
                    target_regions,
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
                "A changed PDE certificate did not retain every accepted owner."
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
                f"Continued physical state was refused: {committed.diagnostics}."
            )
        if not all(transfer.evidence.passed for transfer in committed.transfers):
            raise ValueError("Continued field transfer evidence did not pass.")
        accepted = committed.state
        current_space = target_space
        if accepted.materials is None:
            raise ValueError("The committed state lost its material owner.")
        inventory = _region_inventory(
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
            "Inverse coarsening did not restore the source scientific topology."
        )
    if not _same_coordinate_map(source.geometry, coarsening.target.geometry):
        raise ValueError(
            "Inverse coarsening did not restore the complete source coordinate map."
        )
    if not np.allclose(accepted.fields[1], damage, rtol=0.0, atol=1.0e-9):
        raise ValueError("The irreversible damage history was not restored.")
    return {
        "initial_cells": int(_cell_ids(source).size),
        "region_inventories": np.asarray(inventories).tolist(),
        "stages": stages,
    }


def main() -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_sources import (
        _read_meshing_source_archive,
        _restore_meshing_source_archive,
        read_meshing_source_execution_controls,
    )
    from phydrax.meshing._measurements import NativeExecutionRecord
    from phydrax.meshing._volume_generation import native_volume_source_execution_budget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("content_id")
    parser.add_argument(
        "--region-densities", type=float, nargs="+", default=REGION_DENSITIES
    )
    parser.add_argument("--material-site", default="two-material-density")
    args = parser.parse_args()
    stages_seconds: dict[str, float] = {}
    phase_started = perf_counter()
    archive = _read_meshing_source_archive(
        args.archive, expected_content_id=args.content_id
    )
    stages_seconds["archive_member_authentication"] = perf_counter() - phase_started
    phase_started = perf_counter()
    controls = read_meshing_source_execution_controls(
        args.archive,
        expected_content_id=args.content_id,
        _authenticated_archive=archive,
    )
    stages_seconds["execution_control_restoration"] = perf_counter() - phase_started
    previous = controls.execution_evidence
    if previous.owner_id is None:
        raise ValueError(
            "Cold layer continuation requires its actual original execution owner."
        )
    with native_volume_source_execution_budget(
        controls.limits, previous, previous.owner_id
    ) as execution:
        phase_started = perf_counter()
        records = _restore_meshing_source_archive(
            archive, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        )
        stages_seconds["scientific_owner_restoration"] = perf_counter() - phase_started
        source, refinement, coarsening = accepted_lineage(records)
        authored = records["generation_source"]
        specification = records["generation_specification"]
        if not isinstance(authored, NativeLayerCoreSource):
            raise TypeError(
                "The closure must retain its authored native layer/core source."
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
                "Cold continuation changed its authenticated original source or controls."
            )
        material = material_declaration(
            source, densities=tuple(args.region_densities), site_id=args.material_site
        )
        plans = _LifecyclePlanCache()
        spaces_ready, h1_ready, finite_volume_ready = Event(), Event(), Event()

        def run_fields() -> list[FieldContinuation]:
            started = perf_counter()
            try:
                return continue_fields(
                    source,
                    refinement,
                    coarsening,
                    plans=plans,
                    spaces_ready=spaces_ready,
                    h1_ready=h1_ready,
                )
            finally:
                spaces_ready.set()
                h1_ready.set()
                stages_seconds["compatible_field_continuation"] = perf_counter() - started

        def run_volume() -> VolumeContinuation:
            started = perf_counter()
            try:
                return continue_finite_volume(
                    source,
                    refinement,
                    coarsening,
                    material=material,
                    plans=plans,
                    spaces_ready=finite_volume_ready,
                )
            finally:
                finite_volume_ready.set()
                stages_seconds["finite_volume_continuation"] = perf_counter() - started

        def run_history() -> HistoryContinuation:
            started = perf_counter()
            try:
                return continue_history(
                    source,
                    refinement,
                    coarsening,
                    material=material,
                    plans=plans,
                    h1_ready=h1_ready,
                    finite_volume_ready=finite_volume_ready,
                )
            finally:
                stages_seconds["history_and_pde_continuation"] = perf_counter() - started

        def run_deferred(call: Callable[[], Any], /) -> Any:
            with execution.deferred_worker():
                return call()

        with execution.deferred_workers():
            with ThreadPoolExecutor(max_workers=3) as pool:
                fields_future = pool.submit(run_deferred, run_fields)
                spaces_ready.wait()
                volume_future = pool.submit(run_deferred, run_volume)
                history_future = pool.submit(run_deferred, run_history)
                fields = fields_future.result()
                volume = volume_future.result()
                history = history_future.result()
        print(
            json.dumps(
                {
                    "source_id": authored.source_id,
                    "source_revision": authored.source_revision,
                    "material_declaration": material,
                    "source_mesh_id": source.mesh.mesh_id,
                    "source_geometry_id": cell_geometry_id(source.geometry),
                    "coarsened_geometry_id": cell_geometry_id(coarsening.target.geometry),
                    "restored_coordinate_map_bitwise": _same_coordinate_map(
                        source.geometry, coarsening.target.geometry
                    ),
                    "refinement_id": refinement.result_id,
                    "coarsening_id": coarsening.result_id,
                    "cells": [
                        int(_cell_ids(result).size)
                        for result in (source, refinement.target, coarsening.target)
                    ],
                    "stages_seconds": stages_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            json.dumps(
                {"fields": fields},
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            json.dumps(
                {"finite_volume": volume},
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            json.dumps(
                {"history": history},
                sort_keys=True,
            ),
            flush=True,
        )
    if execution.evidence is None:
        raise RuntimeError(
            "Cold continuation did not produce its actual ended original-scope evidence."
        )
    latest = NativeExecutionRecord(
        execution.evidence, preparation_evidence=previous, owner_id=previous.owner_id
    )
    print(
        json.dumps(
            {
                "execution_evidence": latest.to_record(),
                "limits_id": controls.limits.limits_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
