#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed mesh-adaptation requests, declared routes, and certified transitions.

`prepare_mesh_adaptation` binds a request to a certified source result under
`MeshAdaptationRoute` and resolves protection and organization constraints.
`execute_mesh_adaptation` executes that route and returns the certified target,
lineage, available nodal transfer or geometric common refinement, route evidence,
and compliance.
"""

from __future__ import annotations

import dataclasses
import time
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, ExitStack, nullcontext
from enum import StrEnum
from fractions import Fraction
from typing import Any, assert_never, final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray

import phydrax._geometry_predicates as _dependency_phydrax__geometry_predicates
import phydrax.discretization._adaptive_simplex as _dependency_phydrax_discretization__adaptive_simplex
import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_predecessor as cell_predecessor_owner
import phydrax.discretization._cell_geometry_transfer as _dependency_phydrax_discretization__cell_geometry_transfer
import phydrax.discretization._cell_geometry_transfer as _owner_phydrax_discretization__cell_geometry_transfer
import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._cell_geometry_validity as _owner_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _owner_phydrax_discretization__coordinate_enclosure
import phydrax.discretization._exact_plc_geometry as _dependency_phydrax_discretization__exact_plc_geometry
import phydrax.discretization._exact_power_geometry as _dependency_phydrax_discretization__exact_power_geometry
import phydrax.discretization._exact_rational_geometry as exact_rational_geometry_owner
import phydrax.discretization._reference_cell as _owner_phydrax_discretization__reference_cell
import phydrax.discretization._source_geometry_realization as source_realization_owner
import phydrax.discretization._sphere_chart_records as _owner_phydrax_discretization__sphere_chart_deformation
import phydrax.discretization._surface_chart_records as _dependency_phydrax_discretization__surface_chart_deformation
import phydrax.discretization._surface_chart_records as _owner_phydrax_discretization__surface_chart_deformation
import phydrax.discretization.fem._reference_types as _owner_phydrax_discretization_fem__reference_types
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._mapped_reference_domain as mapped_reference_domain_owner
import phydrax.geometry._mesh_certificates as _dependency_phydrax_geometry__mesh_certificates
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry._sphere_material_atlas as _owner_phydrax_geometry__sphere_material_atlas
import phydrax.geometry._supermesh as _dependency_phydrax_geometry__supermesh
import phydrax.geometry._surface_source_support as _owner_phydrax_geometry__surface_source_support
import phydrax.geometry.brep._patches as _owner_phydrax_geometry_brep__patches
import phydrax.geometry.brep._placed as _owner_phydrax_geometry_brep__placed
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._association_composition as _dependency_phydrax_meshing__association_composition
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._audit as _dependency_phydrax_meshing__audit
import phydrax.meshing._bisection as _dependency_phydrax_meshing__bisection
import phydrax.meshing._bisection_kernel_host as bisection_kernel_owner
import phydrax.meshing._canonical as _dependency_phydrax_meshing__canonical
import phydrax.meshing._certification as _owner_phydrax_meshing__certification
import phydrax.meshing._certification_inputs as _dependency_phydrax_meshing__certification_inputs
import phydrax.meshing._compartments as _owner_phydrax_meshing__compartments
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._curving as _dependency_phydrax_meshing__curving
import phydrax.meshing._device_adaptation as _owner_phydrax_meshing__device_adaptation
import phydrax.meshing._device_metric as _owner_phydrax_meshing__device_metric
import phydrax.meshing._distribution as _dependency_phydrax_meshing__distribution
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._geometry_realization as _owner_phydrax_meshing__geometry_realization
import phydrax.meshing._implicit_association_transfer as _dependency_phydrax_meshing__implicit_association_transfer
import phydrax.meshing._layer_core as _owner_phydrax_meshing__layer_core
import phydrax.meshing._level_set as _owner_phydrax_meshing__level_set
import phydrax.meshing._lineage as _dependency_phydrax_meshing__lineage
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage_records
import phydrax.meshing._local_metric as _dependency_phydrax_meshing__local_metric
import phydrax.meshing._mesh_part as mesh_part_owner
import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._mixed_adaptation as _dependency_phydrax_meshing__mixed_adaptation
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._organization_records as _owner_phydrax_meshing__organization
import phydrax.meshing._periodic as _owner_phydrax_meshing__periodic
import phydrax.meshing._plc_mapped_support as plc_mapped_support_owner
import phydrax.meshing._plc_planar_support as plc_planar_support_owner
import phydrax.meshing._polyhedral_adaptation as _dependency_phydrax_meshing__polyhedral_adaptation
import phydrax.meshing._polyhedral_generation as _dependency_phydrax_meshing__polyhedral_generation
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope
import phydrax.meshing._surface_association_transfer as _owner_phydrax_meshing__surface_association_transfer
import phydrax.meshing._surface_boundary_sources as surface_boundary_sources_owner
import phydrax.meshing._surface_metric as _dependency_phydrax_meshing__surface_metric
import phydrax.meshing._surface_metric as _owner_phydrax_meshing__surface_metric
import phydrax.meshing._surface_metric_controller as _owner_phydrax_meshing__surface_metric_controller
import phydrax.meshing._tetra_metric as _dependency_phydrax_meshing__tetra_metric
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit
import phydrax.meshing._topology_edit as _owner_phydrax_meshing__topology_edit
import phydrax.meshing.providers._mmg as _dependency_phydrax_meshing_providers__mmg
import phydrax.meshing.providers._native_periodic as _dependency_phydrax_meshing_providers__native_periodic
import phydrax.meshing.providers._native_sources as _dependency_phydrax_meshing_providers__native_sources
import phydrax.meshing.providers._omega_h as _dependency_phydrax_meshing_providers__omega_h
from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.meshing._host_banks import (
    check_marked_source,
    closure_rows,
    edge_classes,
    membership,
    mesh_cell_classes,
    mesh_cell_ids,
    organization_scopes,
    row_classes,
    selected_rows,
)

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._host_storage import (
    retained_host_storage_fanout_upper,
    retained_host_storage_profile,
)
from .._meshcore import (
    is_native_host_buffer,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from .._model._structure import model_array_bytes
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._exact_algebraic_geometry import ExactAlgebraicCellGeometrySource
from ..discretization.fem import (
    balanced_hp_refinement_ids,
    coarsen_tensor_hp_cells,
    finite_element_hp_interface_plan,
    FiniteElementHPEpoch,
    FiniteElementHPLineage,
    FiniteElementHPRefinementResult,
    FiniteElementTopologyTransfer,
    hp_active_cell_mesh,
    refine_tensor_hp_cells,
)
from ..geometry.surface import SurfaceModel
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..linalg._exact import solve_exact
from ..typing import checked, parse
from . import _measurements as measurement_owner
from ._bisection import BisectionCompatibility, BisectionEvidence, BisectionHierarchy
from ._curving import CurvedGeometryEvidence
from ._device_metric import DeviceMetricEvidence
from ._local_metric import LocalMetricEvidence
from ._measurements import (
    _NATIVE_EXECUTION_RECORD_ARRAY_BYTES,
    NativeExecutionImports,
    NativeExecutionRecord,
)
from ._native_preparation import (
    _NativeReceiptSnapshot,
    _prepare_native_receipt_graph,
    debit_request_preparation,
    native_preparation_ledger,
    NativeCompletedSourceIssue,
    stage_native_completed_source_issue,
    stage_native_source_publication,
)
from ._tetra_metric import MetricRemeshingEvidence
from ._volume_generation import require_native_preparation_allowance
from .providers._mmg import MmgAdaptationResult
from .providers._omega_h import OmegaHAdaptationResult


if TYPE_CHECKING:
    from phydrax.discretization._surface_chart_records import SurfaceChartWitness

    from ..discretization._coordinate_enclosure import CoordinateSourceBank
    from ._surface_association_transfer import PreparedSurfaceCurveWitness


type FullP1Element = (
    _owner_phydrax_discretization_fem__reference_types.FiniteElementSpec
    | _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement
    | _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement
)
type _FullP1OriginalCell = tuple[
    str | None,
    FullP1Element,
    NDArray[np.int32],
    NDArray[np.int64],
    int,
]
type _FullP1SourceCell = tuple[
    FullP1Element,
    NDArray[np.int64],
    NDArray[np.int32],
    int,
    NDArray[np.int64],
    str | None,
    int,
    FullP1Element,
]
type _UniformRationalActions = dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]]


class MeshAdaptationStatus(StrEnum):
    """Outcome of one executed adaptation.

    ``COMPLETE``: every requested operation was realized (for metric routes the
    unit-mesh criterion holds). ``PARTIAL``: some requested marks were rejected
    (protected entities, ineligible coarsening families); the evidence lists them.
    ``PASS_LIMIT``: a metric route stopped at ``maximum_passes`` before the
    unit-mesh criterion. ``STALLED``: no admissible operation remains while the
    criterion is unmet. ``RESOURCE_LIMIT``: native execution exhausted a hard
    resource budget. ``UNCHANGED``: nothing was applied; the target is the
    source result itself.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    PASS_LIMIT = "pass_limit"
    STALLED = "stalled"
    RESOURCE_LIMIT = "resource_limit"
    UNCHANGED = "unchanged"

    @property
    def converged(self) -> bool:
        """Whether the request is fully realized (COMPLETE or UNCHANGED)."""
        return self in (MeshAdaptationStatus.COMPLETE, MeshAdaptationStatus.UNCHANGED)


class MeshAdaptationRoute(StrEnum):
    """Explicit execution route; each request kind admits only its own routes.

    ``NATIVE_BISECTION``: marked Maubach/newest-vertex bisection and coarsening of
    simplex meshes. ``NATIVE_METRIC_2D``: planar triangle metric adaptation (or
    relocation-only r-adaptation). ``NATIVE_METRIC_3D``: affine tetrahedral metric
    remeshing. ``NATIVE_MIXED``: family-preserving template refinement and sibling
    coarsening. ``NATIVE_LEVEL_SET``: existing-tetrahedron P1 zero-set insertion.
    ``NATIVE_POLYHEDRAL``: planar packed-cell subdivision, connected agglomeration,
    or explicit site regeneration with certified geometric common refinement.
    ``DEVICE_BISECTION`` and ``DEVICE_METRIC_2D``:
    the same requests executed as compiled fixed-capacity device passes (see
    `prepare_adaptive_simplex`); device bisection commits the meshes of the host
    bisection route byte for byte. ``MMG`` and ``OMEGA_H``: metric adaptation by
    the configured provider. ``HP``: marked tensor-product h-refinement or
    coarsening of one prepared `FiniteElementHPEpoch`.
    """

    NATIVE_BISECTION = "native_bisection"
    NATIVE_METRIC_2D = "native_metric_2d"
    NATIVE_METRIC_3D = "native_metric_3d"
    NATIVE_SURFACE_METRIC = "native_surface_metric"
    NATIVE_LEVEL_SET = "native_level_set"
    NATIVE_MIXED = "native_mixed"
    NATIVE_POLYHEDRAL = "native_polyhedral"
    NATIVE_GEOMETRY_REALIZATION = "native_geometry_realization"
    DEVICE_BISECTION = "device_bisection"
    DEVICE_METRIC_2D = "device_metric_2d"
    MMG = "mmg"
    OMEGA_H = "omega_h"
    HP = "hp"


_DEVICE_ROUTES = frozenset(
    (MeshAdaptationRoute.DEVICE_BISECTION, MeshAdaptationRoute.DEVICE_METRIC_2D)
)

type MeshAdaptationHierarchy = (
    BisectionHierarchy
    | FiniteElementHPEpoch
    | _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationHierarchy
    | _owner_phydrax_meshing__periodic.PeriodicRefinement
    | None
)


def _identifier_vector(values: ArrayLike, name: str, /) -> np.ndarray:
    identifiers = np.asarray(values)
    if identifiers.size == 0:
        return np.empty((0,), dtype=np.int64)
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError(
            f"MeshAdaptation.cell_ids must be one integer vector; got {f'name={name}, shape={identifiers.shape}, dtype={identifiers.dtype}'!r}"
        )
    identifiers = identifiers.astype(np.int64, copy=False)
    if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
        raise ValueError(
            f"MeshAdaptation.cell_ids must be unique and non-negative; got {f'name={name}, count={identifiers.size}'!r}"
        )
    return np.sort(identifiers)


def _flag(value: bool, name: str, /) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"MeshAdaptationPolicy.relocation must be a bool; got {f'name={name}, kind={type(value).__name__}'!r}"
        )
    return bool(value)


@final
class MarkedMeshAdaptation(StrictModule, NonTrainableState):
    """Source cells marked for refinement and for coarsening, by global ID.

    ``hierarchy`` carries the state of earlier adaptations: the
    `BisectionHierarchy` returned by a previous NATIVE_BISECTION result (required
    for coarsening, and to continue its refinement labels), the source
    `FiniteElementHPEpoch` of the HP route, or `MixedAdaptationHierarchy` for
    NATIVE_MIXED template siblings. ``layer_columns`` carries explicit layer
    ancestry and axial-refinement permissions only for NATIVE_MIXED.
    ``refinement_generations`` requests repeated whole-edge-star descendant
    generations in one native request; other subdivision routes require one.
    With ``refinement_target_cells`` it is a maximum: stop only after a complete
    generation reaches that provisioning goal. The goal guarantees neither a
    minimum cell count nor physical accuracy; None requests every generation.
    """

    refine_cell_ids: Array
    coarsen_cell_ids: Array
    hierarchy: MeshAdaptationHierarchy
    layer_columns: _dependency_phydrax_meshing__mixed_adaptation.MixedLayerColumns | None
    refinement_generations: int = eqx.field(static=True)
    refinement_target_cells: int | None = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        refine_cell_ids: ArrayLike = (),
        coarsen_cell_ids: ArrayLike = (),
        /,
        *,
        hierarchy: MeshAdaptationHierarchy = None,
        layer_columns: _dependency_phydrax_meshing__mixed_adaptation.MixedLayerColumns
        | None = None,
        refinement_generations: int = 1,
        refinement_target_cells: int | None = None,
    ) -> None:
        refine = _identifier_vector(refine_cell_ids, "refine_cell_ids")
        coarsen = _identifier_vector(coarsen_cell_ids, "coarsen_cell_ids")
        generations = positive_integer(refinement_generations, "refinement_generations")
        target_cells = (
            None
            if refinement_target_cells is None
            else positive_integer(refinement_target_cells, "refinement_target_cells")
        )
        if np.intersect1d(refine, coarsen).size:
            raise ValueError(
                f"MarkedMeshAdaptation.cell_ids must not mark a cell for both refinement and coarsening; got {f'refine_count={refine.size}, coarsen_count={coarsen.size}'!r}"
            )
        _host_refine_cell_ids = np.asarray(refine)
        _host_coarsen_cell_ids = np.asarray(coarsen)
        (
            self.refine_cell_ids,
            self.coarsen_cell_ids,
        ) = jax.device_put(
            (
                _host_refine_cell_ids,
                _host_coarsen_cell_ids,
            )
        )
        self.hierarchy = hierarchy
        self.layer_columns = layer_columns
        self.refinement_generations = generations
        self.refinement_target_cells = target_cells
        self.request_id = canonical_fingerprint(
            {
                "kind": "marked-mesh-adaptation",
                "refine": array_tree_fingerprint(refine),
                "coarsen": array_tree_fingerprint(coarsen),
                "refinement_generations": generations,
                "refinement_target_cells": target_cells,
                "hierarchy": _hierarchy_id(hierarchy),
                "layer_columns": None
                if layer_columns is None
                else layer_columns.columns_id,
            }
        )


@final
class MetricMeshAdaptation(StrictModule, NonTrainableState):
    """Adapt toward a unit mesh of one vertex metric bound to the source vertices."""

    metric: _dependency_phydrax_meshing__metric.MeshMetricField
    coordinate_contract: SpatialCoordinateContract | None
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        metric: _dependency_phydrax_meshing__metric.MeshMetricField,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> None:
        self.metric = metric
        self.coordinate_contract = coordinate_contract
        self.request_id = canonical_fingerprint(
            {
                "kind": "metric-mesh-adaptation",
                "metric": metric.metric_id,
                "coordinate_contract": None
                if coordinate_contract is None
                else coordinate_contract.spatial_id,
            }
        )


@final
class RelocationMeshAdaptation(StrictModule, NonTrainableState):
    """Move vertices toward one vertex metric without changing the topology."""

    metric: _dependency_phydrax_meshing__metric.MeshMetricField
    coordinate_contract: SpatialCoordinateContract | None
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        metric: _dependency_phydrax_meshing__metric.MeshMetricField,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> None:
        self.metric = metric
        self.coordinate_contract = coordinate_contract
        self.request_id = canonical_fingerprint(
            {
                "kind": "relocation-mesh-adaptation",
                "metric": metric.metric_id,
                "coordinate_contract": None
                if coordinate_contract is None
                else coordinate_contract.spatial_id,
            }
        )


@final
class GeometryRealizationMeshAdaptation(StrictModule, NonTrainableState):
    """Source-realized coordinate-order candidate on a material complex.

    Retained topology does not imply retained physical support. Curving
    certificates and quantitative old/new measure correspondence are required
    separately from the prior source-fidelity certificate.
    """

    source: _dependency_phydrax_meshing__result.CellMeshingResult
    realization: _dependency_phydrax_meshing__curving.HighOrderCurvingResult
    source_fidelity: source_fidelity_owner.SourceFidelityCertificate
    geometry_realization: source_realization_owner.SourceGeometryRealization
    certificate_limits: (
        _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    )
    source_result_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        realization: _dependency_phydrax_meshing__curving.HighOrderCurvingResult,
        source_fidelity: source_fidelity_owner.SourceFidelityCertificate,
        geometry_realization: source_realization_owner.SourceGeometryRealization,
        certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
        /,
    ) -> None:
        geometry_realization.require_current()
        if (
            realization.status
            is not _dependency_phydrax_meshing__curving.HighOrderCurvingStatus.CURVED
            or not realization.evidence.accepted
        ):
            raise ValueError(
                f"GeometryRealizationMeshAdaptation.realization must contain an accepted curving product; got {(realization.status.name, realization.evidence.accepted)!r}"
            )
        target_fidelity = realization.evidence.fidelity
        if (
            target_fidelity is None
            or target_fidelity.status != "certified"
            or source_fidelity.status != "certified"
        ):
            raise ValueError(
                f"GeometryRealizationMeshAdaptation.certification must retain continuous fidelity for both coordinate epochs; got {(source_fidelity.status, target_fidelity.status if target_fidelity is not None else None)!r}"
            )
        source_fidelity.binding.require(source.mesh, source.geometry)
        target_fidelity.binding.require(source.mesh, realization.geometry)
        if (
            source_fidelity.binding.source_id,
            source_fidelity.binding.source_revision,
        ) != (target_fidelity.binding.source_id, target_fidelity.binding.source_revision):
            raise ValueError(
                f"GeometryRealizationMeshAdaptation.source must retain scientific source provenance across coordinate order; got {source.result_id!r}"
            )
        for certificate in (
            source_fidelity,
            target_fidelity,
            geometry_realization.source_embedding,
            geometry_realization.target_embedding,
        ):
            if certificate.binding.limits_id != certificate_limits.limits_id:
                raise ValueError(
                    f"GeometryRealizationMeshAdaptation.limits must retain every source-certificate budget; got {source.result_id!r}"
                )
        geometry_realization.source_embedding.binding.require(
            source.mesh, source.geometry
        )
        geometry_realization.target_embedding.binding.require(
            source.mesh, realization.geometry
        )
        transition = geometry_realization.transition
        if (
            transition.evidence.kind != "source_realization"
            or transition.source_topology_id != source.mesh.topology_id
            or transition.target_topology_id != source.mesh.topology_id
            or transition.source_geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source.geometry
            )
            or transition.target_geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                realization.geometry
            )
        ):
            raise ValueError(
                f"GeometryRealizationMeshAdaptation.geometry_realization must bind the old and realized coordinate maps; got {(transition.source_geometry_id, transition.target_geometry_id)!r}"
            )
        old_elements = source.geometry.resolve(source.mesh)[0]
        new_elements = realization.geometry.resolve(source.mesh)[0]
        old_orders = tuple(
            _dependency_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
                value, "Source realization"
            ).degree
            for value in old_elements
        )
        new_orders = tuple(
            _dependency_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
                value, "Source realization"
            ).degree
            for value in new_elements
        )
        if any(
            new < old for old, new in zip(old_orders, new_orders, strict=True)
        ) or not any(new > old for old, new in zip(old_orders, new_orders, strict=True)):
            raise ValueError(
                f"GeometryRealizationMeshAdaptation.realization must increase coordinate order; got {(old_orders, new_orders)!r}"
            )
        self.source = source
        self.realization = realization
        self.source_fidelity = source_fidelity
        self.geometry_realization = geometry_realization
        self.certificate_limits = certificate_limits
        self.source_result_id = source.result_id
        self.request_id = canonical_fingerprint(
            {
                "kind": "geometry-realization-mesh-adaptation",
                "source": source.result_id,
                "realization": realization.result_id,
                "old_fidelity": source_fidelity.certificate_id,
                "correspondence": geometry_realization.realization_id,
            }
        )


type MeshAdaptationRequest = (
    MarkedMeshAdaptation
    | MetricMeshAdaptation
    | RelocationMeshAdaptation
    | _owner_phydrax_meshing__level_set.LevelSetMeshAdaptation
    | _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralMeshAdaptation
    | GeometryRealizationMeshAdaptation
)


def _hierarchy_id(
    hierarchy: MeshAdaptationHierarchy,
    /,
) -> str | None:
    match hierarchy:
        case None:
            return None
        case (
            BisectionHierarchy()
            | _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationHierarchy()
        ):
            return hierarchy.hierarchy_id
        case FiniteElementHPEpoch():
            return hierarchy.epoch_id
        case _owner_phydrax_meshing__periodic.PeriodicRefinement():
            return hierarchy.refinement_id
        case invalid:
            assert_never(invalid)


def _provider_options_record(
    options: _dependency_phydrax_meshing_providers__mmg.MmgOptions
    | _dependency_phydrax_meshing_providers__omega_h.OmegaHOptions
    | None,
    /,
) -> Any:
    match options:
        case None:
            return None
        case _dependency_phydrax_meshing_providers__mmg.MmgOptions():
            return {"mmg": dataclasses.asdict(options)}
        case _dependency_phydrax_meshing_providers__omega_h.OmegaHOptions():
            return {"omega_h": options.worker_record()}
        case _:
            raise TypeError(
                f"MeshAdaptationPolicy.provider_options must be MmgOptions or OmegaHOptions; got {type(options).__name__!r}"
            )


def _check_provider(
    route: MeshAdaptationRoute,
    provider: _dependency_phydrax_meshing_providers__mmg.MmgProvider
    | _dependency_phydrax_meshing_providers__omega_h.OmegaHProvider
    | None,
    options: _dependency_phydrax_meshing_providers__mmg.MmgOptions
    | _dependency_phydrax_meshing_providers__omega_h.OmegaHOptions
    | None,
    protected: tuple[_dependency_phydrax_meshing__scope.MeshingScope, ...],
    distribution: _dependency_phydrax_meshing__distribution.MeshDistribution | None,
    /,
) -> None:
    match route:
        case MeshAdaptationRoute.MMG:
            match (provider, options):
                case (
                    _dependency_phydrax_meshing_providers__mmg.MmgProvider(),
                    _dependency_phydrax_meshing_providers__mmg.MmgOptions() | None,
                ):
                    pass
                case _:
                    raise TypeError(
                        f"MeshAdaptationPolicy.provider must provide an MmgProvider with MmgOptions for MMG; got {(type(provider).__name__, type(options).__name__)!r}"
                    )
        case MeshAdaptationRoute.OMEGA_H:
            match (provider, options):
                case (
                    _dependency_phydrax_meshing_providers__omega_h.OmegaHProvider(),
                    _dependency_phydrax_meshing_providers__omega_h.OmegaHOptions() | None,
                ):
                    pass
                case _:
                    raise TypeError(
                        f"MeshAdaptationPolicy.provider must provide an OmegaHProvider with OmegaHOptions for OMEGA_H; got {(type(provider).__name__, type(options).__name__)!r}"
                    )
            if protected:
                raise ValueError(
                    f"MeshAdaptationPolicy.protected_scopes must be empty for Omega_h adaptation; got {len(protected)!r}"
                )
        case (
            MeshAdaptationRoute.NATIVE_BISECTION
            | MeshAdaptationRoute.NATIVE_METRIC_2D
            | MeshAdaptationRoute.NATIVE_METRIC_3D
            | MeshAdaptationRoute.NATIVE_SURFACE_METRIC
            | MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION
            | MeshAdaptationRoute.NATIVE_LEVEL_SET
            | MeshAdaptationRoute.NATIVE_MIXED
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.NATIVE_POLYHEDRAL
            | MeshAdaptationRoute.DEVICE_METRIC_2D
            | MeshAdaptationRoute.HP
        ):
            if provider is not None or options is not None:
                raise ValueError(
                    f"MeshAdaptationPolicy.provider must be absent along with provider options on native routes; got {(type(provider).__name__, type(options).__name__)!r}"
                )
        case invalid:
            assert_never(invalid)
    if distribution is not None and route in (
        MeshAdaptationRoute.MMG,
        MeshAdaptationRoute.OMEGA_H,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.distribution must be absent for provider remeshing with unknown lineage; got {type(distribution).__name__!r}"
        )


def _resolved_predicate_mode(
    route: MeshAdaptationRoute,
    mode: _dependency_phydrax__geometry_predicates.PredicateMode | None,
    /,
) -> _dependency_phydrax__geometry_predicates.PredicateMode:
    """EXACT for host routes and FILTERED_DEVICE for device routes unless given."""

    device = route in _DEVICE_ROUTES
    if mode is None:
        return (
            _dependency_phydrax__geometry_predicates.PredicateMode.FILTERED_DEVICE
            if device
            else _dependency_phydrax__geometry_predicates.PredicateMode.EXACT
        )
    mode = parse(
        mode, _dependency_phydrax__geometry_predicates.PredicateMode, "predicate_mode"
    )
    if (
        route is MeshAdaptationRoute.NATIVE_METRIC_3D
        and mode is not _dependency_phydrax__geometry_predicates.PredicateMode.EXACT
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.predicate_mode must select exact predicates for NATIVE_METRIC_3D; got {mode.name!r}"
        )
    if device != (
        mode is _dependency_phydrax__geometry_predicates.PredicateMode.FILTERED_DEVICE
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.predicate_mode must select PredicateMode.FILTERED_DEVICE on device routes and PredicateMode.FILTERED or PredicateMode.EXACT on host routes; got {mode.name!r}"
        )
    return mode


# Routes whose changed targets are published by ``finalize_native``, which
# carries envelope repair history from the authored anchor. Geometry
# realization, HP and provider routes certify targets directly; partitioned
# device bisection is refused separately in ``prepare_mesh_adaptation``.
_ENVELOPE_CARRY_ROUTES = frozenset(
    (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.NATIVE_METRIC_2D,
        MeshAdaptationRoute.NATIVE_METRIC_3D,
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        MeshAdaptationRoute.NATIVE_LEVEL_SET,
        MeshAdaptationRoute.NATIVE_MIXED,
        MeshAdaptationRoute.NATIVE_POLYHEDRAL,
        MeshAdaptationRoute.DEVICE_BISECTION,
        MeshAdaptationRoute.DEVICE_METRIC_2D,
    )
)


@final
class MeshAdaptationPolicy(StrictModule, NonTrainableState):
    """Route, protection, and execution controls of one adaptation.

    ``route`` is explicit and never replaced by another route. ``protected_scopes``
    (bound like result organization: source mesh ID and numeric version) are never
    split, removed, flipped, or moved; declare periodic boundaries here as well.
    ``compatibility`` selects matching-label initialization or the explicit host
    triangle CONFORMING_CLOSURE / tetrahedral WHOLE_EDGE_STAR subdivision laws.
    Device and periodic orbit bisection refuse these host-only laws. ``predicate_mode`` certifies
    geometric decisions of the native metric route (EXACT uses meshcore when
    available, otherwise the filtered host predicates whose unresolved signs
    reject the operation); device routes evaluate FILTERED_DEVICE predicates and
    escalate unresolved signs. ``maximum_passes`` bounds metric passes and
    ``maximum_closure_iterations`` the bisection closure, whose batches are also
    admitted against the cell, vertex, and work-unit ``limits`` before they are
    bisected. ``relocation`` enables vertex relocation in metric passes.
    ``device_policy`` (required by, and only by, device routes) fixes the capacity
    bucket of the device state. ``distribution`` with ``partition_policy`` carries
    cell ownership through the lineage. ``association_transfer`` carries the
    source's B-Rep associations: B-Rep corner vertices are fixed, ambiguously
    classified edges are protected, CAD classes separate cell, facet, and edge
    classes, and the target associations are propagated through the native
    lineage or re-derived after provider remeshing. Nonconforming HP active meshes
    require an ``audit_policy`` that records rather than rejects nonmanifold
    (hanging) vertices. ``geometry_transition`` governs how nested native routes
    carry a non-affine source coordinate map: refinement restricts it exactly, and
    coarsening follows its declared approximation policy or refuses.
    ``envelope_source`` is the independently authored surface envelope of a source
    that retains envelope repair history; its native wrap must reproduce it.
    Preparation refuses such a source without it, and repair evidence is carried
    only from this owner, so it is admitted only on routes that publish through
    native carry (not geometry realization, HP, provider or partitioned device
    routes). Sources without envelope history require it absent.
    """

    route: MeshAdaptationRoute = eqx.field(static=True)
    protected_scopes: tuple[_dependency_phydrax_meshing__scope.MeshingScope, ...]
    compatibility: BisectionCompatibility = eqx.field(static=True)
    predicate_mode: _dependency_phydrax__geometry_predicates.PredicateMode = eqx.field(
        static=True
    )
    maximum_passes: int = eqx.field(static=True)
    maximum_closure_iterations: int = eqx.field(static=True)
    relocation: bool = eqx.field(static=True)
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits
    audit_policy: _dependency_phydrax_meshing__audit.CellMeshAuditPolicy
    device_policy: (
        _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexPolicy | None
    )
    provider: (
        _dependency_phydrax_meshing_providers__mmg.MmgProvider
        | _dependency_phydrax_meshing_providers__omega_h.OmegaHProvider
        | None
    ) = eqx.field(static=True)
    provider_options: (
        _dependency_phydrax_meshing_providers__mmg.MmgOptions
        | _dependency_phydrax_meshing_providers__omega_h.OmegaHOptions
        | None
    )
    distribution: _dependency_phydrax_meshing__distribution.MeshDistribution | None
    partition_policy: _dependency_phydrax_meshing__distribution.MeshPartitionPolicy | None
    association_transfer: (
        _dependency_phydrax_meshing__association.BRepAssociationTransfer
        | _dependency_phydrax_meshing__association.PlcAssociationTransfer
        | _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer
        | _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
        | _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer
        | _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer
        | _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer
        | None
    )
    geometry_transition: transition_types_owner.CellGeometryTransitionPolicy
    envelope_source: (
        _dependency_phydrax_meshing_providers__native_sources.NativeSurfaceEnvelopeSource
        | None
    )
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        route: MeshAdaptationRoute,
        /,
        *,
        protected_scopes: tuple[
            _dependency_phydrax_meshing__scope.MeshingScope, ...
        ] = (),
        compatibility: BisectionCompatibility = BisectionCompatibility.REJECT,
        predicate_mode: _dependency_phydrax__geometry_predicates.PredicateMode
        | None = None,
        maximum_passes: int = 16,
        maximum_closure_iterations: int = 256,
        relocation: bool = True,
        limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None = None,
        audit_policy: _dependency_phydrax_meshing__audit.CellMeshAuditPolicy
        | None = None,
        device_policy: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexPolicy
        | None = None,
        provider: _dependency_phydrax_meshing_providers__mmg.MmgProvider
        | _dependency_phydrax_meshing_providers__omega_h.OmegaHProvider
        | None = None,
        provider_options: _dependency_phydrax_meshing_providers__mmg.MmgOptions
        | _dependency_phydrax_meshing_providers__omega_h.OmegaHOptions
        | None = None,
        distribution: _dependency_phydrax_meshing__distribution.MeshDistribution
        | None = None,
        partition_policy: _dependency_phydrax_meshing__distribution.MeshPartitionPolicy
        | None = None,
        association_transfer: _dependency_phydrax_meshing__association.BRepAssociationTransfer
        | _dependency_phydrax_meshing__association.PlcAssociationTransfer
        | _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer
        | _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
        | _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer
        | _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer
        | _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer
        | None = None,
        geometry_transition: transition_types_owner.CellGeometryTransitionPolicy
        | None = None,
        envelope_source: _dependency_phydrax_meshing_providers__native_sources.NativeSurfaceEnvelopeSource
        | None = None,
    ) -> None:
        route = parse(route, MeshAdaptationRoute, "route")
        protected = tuple(protected_scopes)
        if not all(
            isinstance(scope, _dependency_phydrax_meshing__scope.MeshingScope)
            for scope in protected
        ):
            raise TypeError(
                f"MeshAdaptationPolicy.protected_scopes must contain MeshingScope values; got {tuple((type(item).__name__ for item in protected))!r}"
            )
        compatibility = parse(compatibility, BisectionCompatibility, "compatibility")
        mode = _resolved_predicate_mode(route, predicate_mode)
        if (route in _DEVICE_ROUTES) != (device_policy is not None):
            raise ValueError(
                f"MeshAdaptationPolicy.device_policy must be supplied on device routes and absent on host routes; got {(route.name, device_policy is not None)!r}"
            )
        if route is MeshAdaptationRoute.DEVICE_BISECTION and compatibility in (
            BisectionCompatibility.CONFORMING_CLOSURE,
            BisectionCompatibility.WHOLE_EDGE_STAR,
        ):
            raise ValueError(
                f"MeshAdaptationPolicy.compatibility must select matching-condition closure on DEVICE_BISECTION; host closure is executed by NATIVE_BISECTION; got {compatibility.name!r}"
            )
        passes = positive_integer(maximum_passes, "maximum_passes")
        closure = positive_integer(
            maximum_closure_iterations, "maximum_closure_iterations"
        )
        relocate = _flag(relocation, "relocation")
        limits_ = (
            _dependency_phydrax_meshing__contracts.MeshingLimits()
            if limits is None
            else limits
        )
        audit = (
            _dependency_phydrax_meshing__audit.CellMeshAuditPolicy()
            if audit_policy is None
            else audit_policy
        )
        if (distribution is None) != (partition_policy is None):
            raise ValueError(
                f"MeshAdaptationPolicy.distribution must be supplied together with partition_policy; got {(distribution is not None, partition_policy is not None)!r}"
            )
        if association_transfer is not None:
            if route is MeshAdaptationRoute.HP:
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must be absent for HP adaptation; got {type(association_transfer).__name__!r}"
                )
            if isinstance(
                association_transfer,
                _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer,
            ) and (
                route is not MeshAdaptationRoute.NATIVE_MIXED or distribution is not None
            ):
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must use the host nested mixed-family route for mapped or composed sources; got {(route.name, distribution is not None)!r}"
                )
            if isinstance(
                association_transfer,
                _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer,
            ) and (
                route is not MeshAdaptationRoute.NATIVE_MIXED or distribution is not None
            ):
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must use the host nested mixed-family route for mapped or composed sources; got {(route.name, distribution is not None)!r}"
                )
            if isinstance(
                association_transfer,
                _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer,
            ) and (
                route
                not in (
                    MeshAdaptationRoute.NATIVE_BISECTION,
                    MeshAdaptationRoute.DEVICE_BISECTION,
                    MeshAdaptationRoute.NATIVE_METRIC_3D,
                )
                or distribution is not None
            ):
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must use a native nested or tetrahedral metric route for implicit sources; got {(route.name, distribution is not None)!r}"
                )
            if isinstance(
                association_transfer,
                _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer,
            ) and (route is not MeshAdaptationRoute.NATIVE_BISECTION):
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must use the host nested quotient route for periodic sources; got {route.name!r}"
                )
        transition = (
            transition_types_owner.CellGeometryTransitionPolicy()
            if geometry_transition is None
            else geometry_transition
        )
        if envelope_source is not None:
            if route not in _ENVELOPE_CARRY_ROUTES:
                raise ValueError(
                    f"MeshAdaptationPolicy.envelope_source must use a native route that carries surface-envelope repair history; got {route.name!r}"
                )
            # Lazy: the envelope owner also uses native volume generation.
            from ._surface_envelope import _require_surface_envelope_anchor

            # The request's own allowance governs the authenticating rewrap.
            _require_surface_envelope_anchor(envelope_source, limits=limits_)
            match association_transfer:
                case None:
                    pass
                case (
                    _dependency_phydrax_meshing__association.PlcAssociationTransfer() as plc
                ) if (
                    plc.domain.source_id,
                    plc.source_revision,
                ) == (envelope_source.source_id, envelope_source.source_revision):
                    pass
                case _:
                    raise ValueError(
                        f"MeshAdaptationPolicy.association_transfer must transfer the authored envelope's repaired PLC source revision; got {type(association_transfer).__name__!r}"
                    )
        _check_provider(route, provider, provider_options, protected, distribution)
        self.route = route
        self.protected_scopes = protected
        self.compatibility = compatibility
        self.predicate_mode = mode
        self.maximum_passes = passes
        self.maximum_closure_iterations = closure
        self.relocation = relocate
        self.limits = limits_
        self.audit_policy = audit
        self.device_policy = device_policy
        self.provider = provider
        self.provider_options = provider_options
        self.distribution = distribution
        self.partition_policy = partition_policy
        self.association_transfer = association_transfer
        self.geometry_transition = transition
        self.envelope_source = envelope_source
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-policy",
                "route": route.value,
                "protected": [scope.scope_id for scope in protected],
                "compatibility": compatibility.value,
                "predicate_mode": mode.value,
                "maximum_passes": passes,
                "maximum_closure_iterations": closure,
                "relocation": relocate,
                "limits": limits_.limits_id,
                "audit": audit.policy_id,
                "device_policy": None
                if device_policy is None
                else device_policy.policy_id,
                "provider": None if provider is None else type(provider).__name__,
                "provider_options": _provider_options_record(provider_options),
                "distribution": None
                if distribution is None
                else distribution.distribution_id,
                "partition_policy": None
                if partition_policy is None
                else partition_policy.policy_id,
                "association_transfer": None
                if association_transfer is None
                else association_transfer.transfer_id,
                "geometry_transition": transition.policy_id,
                **(
                    {}
                    if envelope_source is None
                    else {
                        "envelope_source": envelope_source.binding_id,
                        "envelope_evidence": envelope_source.envelope.evidence.evidence_id,
                    }
                ),
            }
        )


class _AdaptationConstraints(NamedTuple):
    """Host constraint arrays resolved once from protection and organization."""

    protected_edge_keys: np.ndarray
    protected_edge_mask: np.ndarray
    protected_vertex_ids: np.ndarray
    fixed_vertex_mask: np.ndarray
    cell_classes: np.ndarray
    facet_classes: np.ndarray
    edge_classes: np.ndarray
    metric_values: np.ndarray


def _metric_rows(
    mesh: CellMesh, metric: _dependency_phydrax_meshing__metric.MeshMetricField, /
) -> np.ndarray:
    scope = metric.scope
    vertices = mesh.entity_set(0)
    if (
        scope.entity_kind is not _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH
        or scope.entity_dimension != 0
        or scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_set_id != vertices.entity_set_id
    ):
        raise ValueError(
            f"MeshMetricField.scope must be bound to the source vertices; got {scope.scope_id!r}"
        )
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    rows = entity_keys_owner.key_rows(
        identifiers[:, None],
        np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:, None],
    )
    if identifiers.size != rows.size or np.any(rows < 0):
        raise ValueError(
            f"MeshMetricField.scope must cover every source vertex; got {(identifiers.size, rows.size)!r}"
        )
    values = np.asarray(metric.values, dtype=np.float64)[rows]
    if values.shape[1:] != (mesh.ambient_dimension, mesh.ambient_dimension):
        raise ValueError(
            f"MeshMetricField.values must match the ambient dimension; got {f'shape={values.shape}, ambient_dimension={mesh.ambient_dimension}'!r}"
        )
    return values


@contextmanager
@checked
def adaptation_phase(
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    /,
) -> Iterator[
    _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
]:
    if not budget.ticket.settled:
        yield budget
        return
    remaining = scope.remaining()
    ticket = scope.batch(
        min(budget.maximum_work_units, remaining.remaining_work_units),
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=min(budget.maximum_memory_bytes, remaining.remaining_managed_bytes),
        cavity_upper=remaining.remaining_cavity_cells,
    )
    with budget.bind_ticket(ticket), ticket.native_target():
        yield budget


def _adaptation_charge(
    work: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:

    ticket = budget.ticket
    if work > ticket.remaining_work_units:
        # Refusal names the next increment, never consumed totals.
        ticket.refuse(work)
    ticket.consume(work)


def _adaptation_cavity(
    count: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    ticket = budget.ticket
    if count > ticket.cavity_upper:
        # A cavity is a replacement capacity; no work or query increment is requested.
        ticket.refuse(0, cavity_upper=count)
    ticket.consume(cavity_cells=count)


def _uniform_scientific_block_rows(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    hierarchy: BisectionHierarchy,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:

    lineage = hierarchy.uniform_refinement
    if lineage is None:
        raise ValueError(
            f"BisectionHierarchy.uniform_refinement must retain its scientific source owner; got {hierarchy.hierarchy_id!r}"
        )
    original = lineage.source
    origin = source.geometry.restriction_source
    if (
        origin is None
        or origin.source_geometry_id
        != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            original.geometry
        )
        or origin.source_topology_id != original.mesh.topology_id
    ):
        raise ValueError(
            f"BisectionUniformRefinement.source_geometry must match its captured geometry owner; got {source.geometry.geometry_layout_id!r}"
        )
    count = sum(block.cell_count for block in source.mesh.blocks)
    original_count = sum(block.cell_count for block in original.mesh.blocks)
    storage = (count + original_count) * 256
    if storage > limits.maximum_scratch_bytes:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"MeshAdaptationPolicy.limits must cover scientific block preparation storage; got {f'required_bytes={storage}, maximum_scratch_bytes={limits.maximum_scratch_bytes}'!r}",
            stage="scientific-class-preparation",
        )
    _adaptation_charge(count + original_count, budget=budget)
    budget.reserve(0, storage)
    root_blocks = {
        int(identifier): block_index
        for block_index, block in enumerate(original.mesh.blocks)
        for identifier in block.global_ids
    }
    rows = []
    for block in source.mesh.blocks:
        parents = np.asarray(origin.block_parent_cell_ids[block.name], dtype=np.int64)
        if parents.shape != (block.cell_count,) or any(
            int(parent) not in root_blocks for parent in parents
        ):
            raise ValueError(
                f"BisectionUniformRefinement.block_routes must declare a scientific cell route for each presentation block; got {block.name!r}"
            )
        rows.append(
            np.asarray([root_blocks[int(parent)] for parent in parents], dtype=np.int64)
        )
    return np.concatenate(rows)


def _uniform_class_declarations_match(
    original: _dependency_phydrax_meshing__result.CellMeshingResult,
    current: _dependency_phydrax_meshing__result.CellMeshingResult,
    degree: int,
    /,
) -> bool:

    if len(original.zones) != len(current.zones):
        return False
    zone_ids = {
        before.zone_id: after.zone_id
        for before, after in zip(original.zones, current.zones, strict=True)
    }
    for previous, present in (
        (original.patches, current.patches),
        (original.zones, current.zones),
        (original.labels, current.labels),
    ):
        expected = tuple(
            value for value in previous if value.scope.entity_dimension == degree
        )
        actual = tuple(
            value for value in present if value.scope.entity_dimension == degree
        )
        if len(expected) != len(actual):
            return False
        for before, after in zip(expected, actual, strict=True):
            match before:
                case _dependency_phydrax_meshing__organization.MeshPatch():
                    if not isinstance(
                        after, _dependency_phydrax_meshing__organization.MeshPatch
                    ) or any(value not in zone_ids for value in before.adjacent_zone_ids):
                        return False
                    witness = _dependency_phydrax_meshing__organization.MeshPatch(
                        before.name,
                        after.scope,
                        connected=before.connected,
                        adjacent_zone_ids=tuple(
                            zone_ids[value] for value in before.adjacent_zone_ids
                        ),
                        source_adjacent_region_ids=before.source_adjacent_region_ids,
                    )
                case _dependency_phydrax_meshing__organization.MeshZone():
                    if not isinstance(
                        after, _dependency_phydrax_meshing__organization.MeshZone
                    ):
                        return False
                    witness = _dependency_phydrax_meshing__organization.MeshZone(
                        before.name,
                        before.role,
                        after.scope,
                        material_id=before.material_id,
                        region_role=before.region_role,
                    )
                case _dependency_phydrax_meshing__organization.MeshLabel():
                    if not isinstance(
                        after, _dependency_phydrax_meshing__organization.MeshLabel
                    ):
                        return False
                    witness = _dependency_phydrax_meshing__organization.MeshLabel(
                        before.name, after.scope
                    )
                case _:
                    raise TypeError(
                        f"MeshAdaptation.source.organization must retain canonical ordering of scientific class declarations; got {before!r}"
                    )
            if _owner_phydrax_meshing__organization.organization_definition_id(
                witness
            ) != _owner_phydrax_meshing__organization.organization_definition_id(after):
                return False
    return True


def _uniform_cell_scientific_classes(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    uniform: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    current_classes: np.ndarray,
    /,
) -> np.ndarray:
    """Keep captured class identities on unchanged, explicitly rooted occurrences."""
    if source.geometry.restriction_source is None:
        raise ValueError(
            f"BisectionUniformRefinement.source must retain scientific cell routes; got {source.result_id!r}"
        )
    original = uniform.source
    parent_classes = dict(
        zip(
            np.asarray(uniform.parent_ids).tolist(),
            np.asarray(uniform.parent_classes).tolist(),
            strict=True,
        )
    )
    origin = source.geometry.restriction_source
    root_by_cell = {
        int(identifier): int(parent)
        for block in source.mesh.blocks
        for identifier, parent in zip(
            block.global_ids,
            np.asarray(origin.block_parent_cell_ids[block.name]),
            strict=True,
        )
    }
    cell_ids = np.asarray(
        source.mesh.entity_set(source.mesh.topological_dimension).entity_ids
    )
    roots = np.asarray(
        [root_by_cell[int(identifier)] for identifier in cell_ids], dtype=np.int64
    )
    original_ids = np.asarray(
        original.mesh.entity_set(original.mesh.topological_dimension).entity_ids
    )
    original_rows = entity_keys_owner.key_rows(original_ids[:, None], roots[:, None])
    if np.any(original_rows < 0) or any(
        int(root) not in parent_classes for root in roots
    ):
        raise ValueError(
            f"BisectionUniformRefinement.parent_cell_ids must declare every scientific class parent; got roots={roots.tolist()!r}"
        )
    before = membership(
        original.mesh, organization_scopes(original), original.mesh.topological_dimension
    )
    after = membership(
        source.mesh, organization_scopes(source), source.mesh.topological_dimension
    )
    changed = (
        np.ones(roots.shape, dtype=np.bool_)
        if before.shape[1] != after.shape[1]
        else np.any(after != before[original_rows], axis=1)
    )
    if not _uniform_class_declarations_match(
        original, source, source.mesh.topological_dimension
    ):
        changed[:] = True
    for root in np.unique(roots):
        selected = roots == root
        if np.unique(current_classes[selected]).size != 1:
            changed[selected] = True
    captured = np.asarray([parent_classes[int(root)] for root in roots], dtype=np.int64)
    return np.where(
        changed,
        current_classes + int(np.max(np.asarray(uniform.parent_classes))) + 1,
        captured,
    )


@final
class _ScientificConstraints(StrictModule):
    """Scientific block rows and the admitted source uniform lineage."""

    block_rows: np.ndarray | None
    uniform: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement | None


def _scientific_constraints(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _ScientificConstraints:
    mesh = source.mesh
    uniform_hierarchy = (
        request.hierarchy
        if policy.route
        in (MeshAdaptationRoute.NATIVE_BISECTION, MeshAdaptationRoute.DEVICE_BISECTION)
        and isinstance(request, MarkedMeshAdaptation)
        and isinstance(request.hierarchy, BisectionHierarchy)
        and request.hierarchy.uniform_refinement is not None
        else None
    )
    scientific_blocks = (
        None
        if uniform_hierarchy is None
        else _uniform_scientific_block_rows(
            source, uniform_hierarchy, policy.limits, budget=budget
        )
    )
    if (
        uniform_hierarchy is None
        and policy.route
        in (MeshAdaptationRoute.NATIVE_BISECTION, MeshAdaptationRoute.DEVICE_BISECTION)
        and isinstance(request, MarkedMeshAdaptation)
        and isinstance(request.hierarchy, BisectionHierarchy)
    ):
        authority_ids = np.asarray(request.hierarchy.scientific_cell_ids, dtype=np.int64)
        active_ids = mesh_cell_ids(mesh)
        rows = np.searchsorted(authority_ids, active_ids)
        if np.any(rows >= authority_ids.size) or not np.array_equal(
            authority_ids[rows], active_ids
        ):
            raise ValueError(
                f"BisectionHierarchy.scientific_cell_ids must include every active scientific cell; got {active_ids.size!r}"
            )
        scientific_blocks = np.asarray(
            request.hierarchy.scientific_block_ids, dtype=np.int32
        )[rows]
    if (
        policy.route
        in (MeshAdaptationRoute.NATIVE_BISECTION, MeshAdaptationRoute.DEVICE_BISECTION)
        and isinstance(request, MarkedMeshAdaptation)
        and isinstance(
            request.hierarchy, _owner_phydrax_meshing__periodic.PeriodicRefinement
        )
        and source.geometry.restriction_source is not None
    ):
        scientific_blocks = (
            _owner_phydrax_meshing__periodic.periodic_scientific_block_rows(
                source,
                request.hierarchy,
                limits=policy.limits,
                scope=scope,
                budget=budget,
            )
        )
    return _ScientificConstraints(
        scientific_blocks,
        None if uniform_hierarchy is None else uniform_hierarchy.uniform_refinement,
    )


@final
class _AssociationConstraints(StrictModule):
    """Source-stratum classes resolved with protected host masks."""

    cell_classes: np.ndarray
    facet_classes: np.ndarray
    edge_codes: np.ndarray


@final
class _RouteSourcePreparation(StrictModule):
    """Source proofs earned once within the route's live preparation budget."""

    plc_proofs: list[plc_mapped_support_owner.MappedPlcSupport]
    periodic_proofs: list[
        _dependency_phydrax_meshing_providers__native_periodic._PeriodicSourceAssociationPreparation
    ]
    planar_proofs: list[plc_planar_support_owner.PlanarPlcSourceProof]


def _association_constraints(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    protected_edges: np.ndarray,
    fixed: np.ndarray,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _AssociationConstraints:
    mesh = source.mesh
    dimension = mesh.topological_dimension
    edge_count = mesh.entity_set(1).count
    edge_codes = np.zeros((edge_count,), dtype=np.int64)
    # Edges of source strata below this dimension are feature edges.
    feature_dimension = dimension
    transfer = policy.association_transfer
    midpoint_required = policy.route in (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.DEVICE_BISECTION,
        MeshAdaptationRoute.NATIVE_MIXED,
    )
    if policy.route is MeshAdaptationRoute.NATIVE_MIXED:
        match request:
            case MarkedMeshAdaptation():
                pass
            case _:
                raise TypeError(
                    f"MeshAdaptation.request must provide marked mixed-template adaptation with its mixed hierarchy; got {type(request).__name__!r}"
                )
        # Inverse full-family restoration uses existing parent vertices;
        # a future split midpoint is not a hard constraint on discarded children.
        midpoint_required = request.refine_cell_ids.size > 0
    if transfer is not None and source.associations:
        match transfer:
            case _dependency_phydrax_meshing__association.BRepAssociationTransfer():
                classes = transfer.classes(source, budget=budget)
                codes = tuple(
                    np.where(
                        level.dimensions >= 0,
                        transfer.projection.entity_codes(
                            np.maximum(level.dimensions, 0), np.maximum(level.indices, 0)
                        ),
                        -1,
                    )
                    for level in classes
                )
            case _dependency_phydrax_meshing__association.PlcAssociationTransfer():
                classes = transfer.classes(
                    source, _prepared_source=source_preparation.plc_proofs, budget=budget
                )
                codes = tuple(level.codes for level in classes)
                exact_midpoints = policy.route is MeshAdaptationRoute.NATIVE_MIXED
                if (
                    policy.route is MeshAdaptationRoute.NATIVE_BISECTION
                    and source.mesh.topological_dimension == 2
                    and source.mesh.ambient_dimension == 2
                ):
                    source_elements, _, _ = source.geometry.resolve(
                        source.mesh, budget=budget
                    )
                    exact_midpoints = _full_p1_action_sources(source_elements)
                # Exact nested and admitted full-P1 producers retain chart
                # midpoints; rounded nodal producers keep the old point guard.
                protected_edges |= transfer.protected_edges(
                    source,
                    midpoint_required=midpoint_required,
                    exact_chart_midpoints=exact_midpoints,
                    _classes=classes,
                    budget=budget,
                )
                if transfer.domain.ambient_dimension == 2:
                    # Execution of this request reuses the planar source maps
                    # and classes earned here instead of reproving them.
                    planar = plc_planar_support_owner.issue_planar_source_proof(
                        transfer, source, budget=budget
                    )
                    if planar is not None:
                        source_preparation.planar_proofs.append(planar)
            case _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer():
                composed_contexts: list[
                    _dependency_phydrax_meshing__association.PlcBankContext
                ] = []
                classes = transfer.classes(
                    source,
                    budget=budget,
                    _prepared_contexts=composed_contexts,
                    _prepared_source=source_preparation.plc_proofs,
                )
                codes = tuple(level.codes for level in classes)
                protected_edges |= transfer.protected_edges(
                    source,
                    midpoint_required=midpoint_required,
                    _classes=classes,
                    _prepared_contexts=composed_contexts,
                    budget=budget,
                )
            case _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer():
                classes = transfer.classes(source, budget=budget)
                codes = tuple(level.codes for level in classes)
            case _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer():
                classes = transfer.classes(source, budget=budget)
                codes = tuple(level.codes for level in classes)
                if policy.route is MeshAdaptationRoute.NATIVE_METRIC_3D:
                    # The tetra route proves every curved facet from the analytic
                    # source: edges on the smooth zero set are not features;
                    # real feature curves and corners (dimension < 2) remain.
                    feature_dimension = 2
            case _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer():
                classes = transfer.classes(
                    source,
                    scope=scope,
                    budget=budget,
                    _prepared_source=source_preparation.periodic_proofs,
                )
                codes = tuple(level.codes for level in classes)
            case _dependency_phydrax_meshing__association.SurfaceAssociationTransfer():
                classes = transfer.classes(source, budget=budget)
                codes = tuple(level.codes for level in classes)
                protected_edges |= transfer.protected_edges(
                    source,
                    midpoint_required=midpoint_required,
                    budget=budget,
                )
            case invalid:
                assert_never(invalid)
        fixed |= classes[0].dimensions == 0
        protected_edges |= ~classes[1].resolved
        cell_classes = row_classes(np.column_stack((cell_classes, codes[dimension])))
        facet_classes = row_classes(
            np.column_stack((facet_classes, codes[dimension - 1]))
        )
        edge_codes = np.where(classes[1].dimensions < feature_dimension, 1 + codes[1], 0)
    return _AssociationConstraints(cell_classes, facet_classes, edge_codes)


def _resolve_constraints(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _AdaptationConstraints:
    if policy.route is not MeshAdaptationRoute.NATIVE_POLYHEDRAL:
        return _resolve_admitted_constraints(
            source, request, policy, source_preparation=source_preparation,
            budget=budget, scope=scope,
        )
    required, work = _polyhedral_classification_bounds(source, policy)
    # The association/scientific owners retain their own proof graphs. This
    # lease covers only the host masks, keys, membership and class banks.
    with budget.temporary_scope(), budget.live_storage() as storage:
        storage.set_bound(required, work=work)
        result = _resolve_admitted_constraints(
            source, request, policy, source_preparation=source_preparation,
            budget=budget, scope=scope,
        )
        budget.retain_owner(result, temporary_storage_credit=storage)
        return result


def _resolve_admitted_constraints(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _AdaptationConstraints:
    mesh = source.mesh
    dimension = mesh.topological_dimension
    edge_count = mesh.entity_set(1).count
    protected_edges = np.zeros((edge_count,), dtype=np.bool_)
    fixed = np.zeros((mesh.coordinates.shape[0],), dtype=np.bool_)
    for protected_scope in policy.protected_scopes:
        if (
            policy.route is MeshAdaptationRoute.NATIVE_SURFACE_METRIC
            and protected_scope.entity_dimension == dimension
        ):
            # A protected surface owns continuous fidelity, not frozen in-surface
            # topology. Its declared curves/corners remain protected below.
            continue
        rows = selected_rows(mesh, protected_scope)
        if protected_scope.entity_dimension >= 1:
            protected_edges[
                closure_rows(mesh, protected_scope.entity_dimension, rows, 1)
            ] = True
        fixed[closure_rows(mesh, protected_scope.entity_dimension, rows, 0)] = True
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    organization = organization_scopes(source)
    science = _scientific_constraints(source, request, policy, budget=budget, scope=scope)
    scientific_blocks = science.block_rows
    cell_classes = mesh_cell_classes(
        mesh,
        organization,
        family_identity=policy.route
        in (
            MeshAdaptationRoute.NATIVE_MIXED,
            MeshAdaptationRoute.NATIVE_POLYHEDRAL,
            MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        ),
        scientific_block_rows=scientific_blocks,
    )
    if source.attributes:
        block_rows = entity_keys_owner.key_rows(
            mesh_cell_ids(mesh)[:, None],
            entity_keys_owner.entity_keys(mesh, dimension),
        )
        cell_classes = row_classes(
            np.column_stack(
                (
                    cell_classes,
                    _owner_phydrax_meshing__layer_core.layer_interval_classes(source)[
                        block_rows
                    ],
                )
            )
        )
    facet_classes = row_classes(
        np.column_stack(
            (
                np.zeros((mesh.entity_set(dimension - 1).count,), dtype=np.int64),
                membership(mesh, organization, dimension - 1).astype(np.int64),
            )
        )
    )
    association = _association_constraints(
        source,
        request,
        policy,
        cell_classes,
        facet_classes,
        protected_edges,
        fixed,
        budget=budget,
        scope=scope,
        source_preparation=source_preparation,
    )
    cell_classes, facet_classes, edge_codes = (
        association.cell_classes,
        association.facet_classes,
        association.edge_codes,
    )
    protected_vertex_ids = vertex_ids[fixed]
    fixed |= np.any(membership(mesh, organization, 0), axis=1)
    surface = mesh.topological_dimension == 2
    metric = (
        _metric_rows(mesh, request.metric)
        if isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation))
        else np.zeros(
            (0, mesh.ambient_dimension, mesh.ambient_dimension), dtype=np.float64
        )
    )
    local_uniform = science.uniform
    if local_uniform is not None:
        uniform = local_uniform
        collective = source.collective_evidence
        if (
            isinstance(
                collective, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
            )
            and collective.uniform_refinement is not None
        ):
            collective_uniform = collective.uniform_refinement
            if (
                collective_uniform.dimension != local_uniform.dimension
                or collective_uniform.source.result_id != local_uniform.source.result_id
            ):
                raise ValueError(
                    f"MeshAdaptation.collective_uniform_refinement must retain the same source and dimension as local uniform lineage; got {(collective_uniform.dimension, collective_uniform.source.result_id)!r}"
                )
            uniform = collective_uniform
        cell_classes = _uniform_cell_scientific_classes(source, uniform, cell_classes)
        if not _uniform_class_declarations_match(uniform.source, source, dimension - 1):
            facet_classes += int(np.max(np.asarray(uniform.parent_facet_classes))) + 1
    return _AdaptationConstraints(
        protected_edge_keys=entity_keys_owner.entity_keys(mesh, 1)[protected_edges],
        protected_edge_mask=protected_edges,
        protected_vertex_ids=protected_vertex_ids,
        fixed_vertex_mask=fixed,
        cell_classes=cell_classes,
        facet_classes=facet_classes,
        edge_classes=(
            _feature_classes(edge_classes(mesh, organization, cell_classes), edge_codes)
            if surface
            else _feature_classes(
                np.where(
                    np.any(membership(mesh, organization, 1), axis=1),
                    1 + row_classes(membership(mesh, organization, 1)),
                    0,
                ),
                edge_codes,
            )
        ),
        metric_values=metric,
    )


def _feature_classes(classes: np.ndarray, cad_codes: np.ndarray, /) -> np.ndarray:
    """Split feature edge classes by B-Rep edge (``cad_codes`` 0 off B-Rep edges)."""
    feature = (classes > 0) | (cad_codes > 0)
    result = np.zeros(classes.shape, dtype=np.int64)
    if np.any(feature):
        result[feature] = 1 + row_classes(np.column_stack((classes, cad_codes))[feature])
    return result


@checked
def require_affine_geometry(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Refuse sources whose coordinate map is not the mesh's affine carrier.

    Routes without a geometry transition publish affine successors; a curved or
    displaced source map would collapse to its corners, so it is refused before
    any work.
    """

    if not _dependency_phydrax_discretization__cell_geometry_transfer.is_affine_cell_geometry(
        source.mesh, source.geometry, budget=budget
    ):
        raise ValueError(
            f"MeshAdaptation.source_geometry must be affine when the route has no geometry transition for non-affine sources; got {source.geometry.geometry_layout_id!r}"
        )


def _require_native_source(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    transfer: _dependency_phydrax_meshing__association.BRepAssociationTransfer
    | _dependency_phydrax_meshing__association.PlcAssociationTransfer
    | _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer
    | _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
    | _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer
    | _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer
    | _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer
    | None,
    /,
    *,
    nested: bool,
    plc_ancestry: bool = False,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> None:
    """Admit a native source; nested routes carry their supported coordinate maps."""

    if nested:
        _dependency_phydrax_discretization__cell_geometry_transfer.nested_geometry_degree(
            source.mesh, source.geometry, budget=budget
        )
    else:
        match source.geometry.exact_source:
            case (
                _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource()
                | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource()
                | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource() as power
            ):
                power.prepare(source.geometry.coordinates, budget=budget)
            case (
                _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource()
                | _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource() as plc
            ):
                if not plc_ancestry:
                    raise ValueError(
                        f"MeshAdaptation.geometry_transition must carry PLC source ancestry on this route; got {source.result_id!r}"
                    )
                plc.prepare(source.geometry.coordinates, budget=budget)
            case ExactAlgebraicCellGeometrySource():
                raise ValueError(
                    "MeshAdaptation.geometry_transition must carry algebraic source ancestry through a nested transition; got a nonnested route"
                )
            case exact_rational_geometry_owner.ExactRationalCellGeometrySource():
                raise ValueError(
                    "MeshAdaptation.geometry_transition must carry exact rational source ancestry through a nested transition; got a nonnested route"
                )
            case None:
                require_affine_geometry(source, budget=budget)
            case invalid:
                assert_never(invalid)
    if source.boundary is not None and not isinstance(
        transfer,
        _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must provide a boundary-model remap; got {type(transfer).__name__!r}"
        )
    if source.attributes:
        _owner_phydrax_meshing__layer_core.validate_layer_index_attributes(source)
    if source.associations or isinstance(
        transfer,
        _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer,
    ):
        if transfer is None:
            raise ValueError(
                f"MeshAdaptationPolicy.association_transfer must be supplied for geometry associations; got {type(transfer).__name__!r}"
            )
        match transfer:
            case _dependency_phydrax_meshing__association.BRepAssociationTransfer():
                transfer.source_associations(source)
            case _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer():
                transfer.source_associations(source, budget=budget)
            case _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer():
                transfer.source_associations(
                    source,
                    scope=scope,
                    budget=budget,
                    _prepared_source=source_preparation.periodic_proofs,
                )
            case _dependency_phydrax_meshing__association.PlcAssociationTransfer():
                transfer.source_associations(
                    source, _prepared_source=source_preparation.plc_proofs, budget=budget
                )
            case _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer():
                transfer.source_associations(
                    source, budget=budget, _prepared_source=source_preparation.plc_proofs
                )
            case (
                _dependency_phydrax_meshing__association.SurfaceAssociationTransfer()
                | _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer()
            ):
                transfer.source_associations(source, budget=budget)
            case invalid:
                assert_never(invalid)
    if not source.coordinate_contract.is_orthonormal_cartesian:
        raise ValueError(
            f"CellMeshingResult.coordinate_contract must be an orthonormal Cartesian frame; got {source.coordinate_contract.spatial_id!r}"
        )


def _validate_polyhedral_route(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> None:
    mesh = source.mesh
    match request:
        case _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralMeshAdaptation():
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide PolyhedralMeshAdaptation for NATIVE_POLYHEDRAL; got {type(request).__name__!r}"
            )
    if {block.cell_kind for block in mesh.blocks} != {
        "polyhedron"
    } or mesh.ambient_dimension != 3:
        raise ValueError(
            f"MeshAdaptation.source_mesh must contain packed 3D polyhedral cells for NATIVE_POLYHEDRAL; got {f'kinds={sorted((block.cell_kind for block in mesh.blocks))}, ambient_dimension={mesh.ambient_dimension}'!r}"
        )
    regenerated = (
        request.operation
        is _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationOperation.REGENERATE
    )
    if (
        regenerated
        and any(item.entity_dimension < 3 for item in organization_scopes(source))
        and not source.associations
    ):
        raise ValueError(
            f"MeshAdaptation.source.associations must cover organized regeneration source strata; got {len(source.associations)!r}"
        )
    if source.attributes or policy.distribution is not None:
        raise ValueError(
            f"MeshAdaptation.source must omit polyhedral attributes and distribution without a common-refinement transition; got {f'attributes={len(source.attributes)}, distribution={policy.distribution is not None}'!r}"
        )
    if regenerated and (source.associations or source.geometry.exact_source is not None):
        _validate_polyhedral_regeneration_source(source, policy, budget=budget)
    else:
        _require_native_source(
            source,
            policy.association_transfer,
            nested=False,
            budget=budget,
            scope=scope,
            source_preparation=source_preparation,
        )


def _validate_polyhedral_regeneration_source(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    policy: MeshAdaptationPolicy,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    if source.geometry.exact_source is None:
        require_affine_geometry(source, budget=budget)
    else:
        source.geometry.exact_source.prepare(source.geometry.coordinates, budget=budget)
    if (
        source.boundary is not None
        or not source.coordinate_contract.is_orthonormal_cartesian
    ):
        raise ValueError(
            f"MeshAdaptation.source must provide a Cartesian PLC source for regeneration; got {source.result_id!r}"
        )
    if policy.association_transfer is not None:
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent when regeneration derives source-carrier associations; got {type(policy.association_transfer).__name__!r}"
        )


def _validate_surface_route(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> None:
    match request:
        case MetricMeshAdaptation() | RelocationMeshAdaptation():
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide a metric or relocation request for NATIVE_SURFACE_METRIC; got {type(request).__name__!r}"
            )
    mesh = source.mesh
    if {block.cell_kind for block in mesh.blocks} != {
        "triangle"
    } or mesh.ambient_dimension != 3:
        raise ValueError(
            f"MeshAdaptation.source_mesh must contain embedded triangle blocks for NATIVE_SURFACE_METRIC; got {f'ambient_dimension={mesh.ambient_dimension}'!r}"
        )
    if (
        request.coordinate_contract is None
        or request.coordinate_contract.spatial_id != source.coordinate_contract.spatial_id
    ):
        raise ValueError(
            f"MeshAdaptation.request.coordinate_contract must declare the source physical coordinate frame; got {(request.coordinate_contract.spatial_id if request.coordinate_contract is not None else None)!r}"
        )
    match policy.association_transfer:
        case (
            _dependency_phydrax_meshing__association.SurfaceAssociationTransfer() as transfer
        ):
            pass
        case _:
            raise ValueError(
                f"MeshAdaptationPolicy.association_transfer must provide parametric source support for NATIVE_SURFACE_METRIC; got {type(policy.association_transfer).__name__!r}"
            )
    if policy.geometry_transition.reconstruction != "bounded_chart_deformation":
        raise ValueError(
            f"MeshAdaptationPolicy.geometry_transition.reconstruction must select bounded_chart_deformation for NATIVE_SURFACE_METRIC; got {policy.geometry_transition.reconstruction!r}"
        )
    if (
        source.certification is None
        or not source.certification.passed
        or source.certification.request.fidelity_tolerance is None
    ):
        raise ValueError(
            f"MeshAdaptation.source.certification must accept continuous source fidelity before surface adaptation; got {source.certification is not None!r}"
        )
    _require_native_source(
        source,
        transfer,
        nested=True,
        budget=budget,
        scope=scope,
        source_preparation=source_preparation,
    )
    _owner_phydrax_meshing__surface_association_transfer.source_associations(
        transfer.support, source, budget=budget
    )


def _validate_bisection_route(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> None:
    mesh = source.mesh
    kinds = {block.cell_kind for block in mesh.blocks}
    match request:
        case MarkedMeshAdaptation():
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide marked bisection adaptation; got {type(request).__name__!r}"
            )
    if kinds != {"triangle"} and kinds != {"tetrahedron"}:
        raise ValueError(
            f"MeshAdaptation.source_mesh must contain triangle or tetrahedron blocks for bisection; got {sorted(kinds)!r}"
        )
    if mesh.periodic_topology is None:
        if not isinstance(request.hierarchy, (BisectionHierarchy, type(None))):
            raise TypeError(
                f"MeshAdaptation.request.hierarchy must provide a bisection hierarchy for a nonperiodic source; got {type(request.hierarchy).__name__!r}"
            )
    elif policy.route is MeshAdaptationRoute.DEVICE_BISECTION:
        raise ValueError(
            f"MeshAdaptationPolicy.route must select host bisection for periodic orbit transactions; got {policy.route.name!r}"
        )
    elif policy.compatibility in (
        BisectionCompatibility.CONFORMING_CLOSURE,
        BisectionCompatibility.WHOLE_EDGE_STAR,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.compatibility must avoid host-only closure for periodic orbit bisection; got {policy.compatibility.name!r}"
        )
    _require_native_source(
        source,
        policy.association_transfer,
        nested=True,
        budget=budget,
        scope=scope,
        source_preparation=source_preparation,
    )
    check_marked_source(source, request)


def _validate_mixed_route(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> None:
    mesh = source.mesh
    kinds = {block.cell_kind for block in mesh.blocks}
    match request:
        case MarkedMeshAdaptation(
            hierarchy=_dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationHierarchy()
            | None
        ):
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide marked mixed-template adaptation with its mixed hierarchy; got {type(request).__name__!r}"
            )
    match (mesh.topological_dimension, mesh.ambient_dimension):
        case (2, 2):
            if kinds != {"quadrilateral"}:
                raise ValueError(
                    f"MeshAdaptation.source_mesh must contain quadrilateral blocks for planar NATIVE_MIXED; got {sorted(kinds)!r}"
                )
            if request.layer_columns is not None:
                raise ValueError(
                    f"MeshAdaptation.request.layer_columns must be absent for planar quad adaptation; got {request.layer_columns is not None!r}"
                )
        case (3, 3):
            if not kinds <= {"tetrahedron", "hexahedron", "prism", "pyramid"}:
                raise ValueError(
                    f"MeshAdaptation.source_mesh must contain tetrahedron, hexahedron, prism, or pyramid blocks for 3D NATIVE_MIXED; got {sorted(kinds)!r}"
                )
        case _:
            raise ValueError(
                f"MeshAdaptation.source_mesh must contain certified planar quads or supported 3D cells for NATIVE_MIXED; got {(mesh.topological_dimension, mesh.ambient_dimension)!r}"
            )
    _require_native_source(
        source,
        policy.association_transfer,
        nested=True,
        budget=budget,
        scope=scope,
        source_preparation=source_preparation,
    )
    check_marked_source(source, request)


def _validate_hp_route(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> None:
    mesh = source.mesh
    match request:
        case MarkedMeshAdaptation(hierarchy=FiniteElementHPEpoch() as epoch):
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide a marked request with its hp epoch; got {type(request).__name__!r}"
            )
    if epoch.mesh.mesh_id != mesh.mesh_id:
        raise ValueError(
            f"MeshAdaptation.request.hierarchy must describe the source mesh revision; got {epoch.mesh.mesh_id!r}"
        )
    if request.refine_cell_ids.size and request.coarsen_cell_ids.size:
        raise ValueError(
            f"MeshAdaptation.request must either refine or coarsen in one HP adaptation; got {f'refine_count={request.refine_cell_ids.size}, coarsen_count={request.coarsen_cell_ids.size}'!r}"
        )
    if source.patches or source.zones or source.labels:
        raise ValueError(
            f"MeshAdaptation.source must omit organization for HP adaptation; got {f'patches={len(source.patches)}, zones={len(source.zones)}, labels={len(source.labels)}'!r}"
        )
    _require_native_source(
        source,
        None,
        nested=False,
        budget=budget,
        scope=scope,
        source_preparation=source_preparation,
    )
    check_marked_source(source, request)


def _check_route_request(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    source_preparation: _RouteSourcePreparation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> None:
    mesh = source.mesh
    kinds = {block.cell_kind for block in mesh.blocks}
    if (
        isinstance(request, MarkedMeshAdaptation)
        and request.layer_columns is not None
        and policy.route is not MeshAdaptationRoute.NATIVE_MIXED
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.route must select NATIVE_MIXED when layer columns are supplied; got {policy.route.name!r}"
        )
    if isinstance(request, MarkedMeshAdaptation) and (
        request.refinement_generations != 1 or request.refinement_target_cells is not None
    ):
        if (
            policy.route is not MeshAdaptationRoute.NATIVE_BISECTION
            or policy.compatibility is not BisectionCompatibility.WHOLE_EDGE_STAR
        ):
            raise ValueError(
                "MeshAdaptation.refinement_generations/target_cells require native whole-edge-star refinement"
            )
        if request.refinement_generations > policy.maximum_passes:
            raise ValueError(
                "MeshAdaptation.refinement_generations must fit maximum_passes"
            )
    match policy.route:
        case MeshAdaptationRoute.NATIVE_BISECTION | MeshAdaptationRoute.DEVICE_BISECTION:
            _validate_bisection_route(
                source,
                request,
                policy,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case MeshAdaptationRoute.NATIVE_MIXED:
            _validate_mixed_route(
                source,
                request,
                policy,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION:
            match request:
                case GeometryRealizationMeshAdaptation():
                    pass
                case _:
                    raise TypeError(
                        f"MeshAdaptation.request must provide GeometryRealizationMeshAdaptation for NATIVE_GEOMETRY_REALIZATION; got {type(request).__name__!r}"
                    )

            _owner_phydrax_meshing__geometry_realization.validate_geometry_realization_adaptation(
                source, request, policy
            )
        case MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            _validate_polyhedral_route(
                source,
                request,
                policy,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case MeshAdaptationRoute.NATIVE_METRIC_2D | MeshAdaptationRoute.DEVICE_METRIC_2D:
            match request:
                case MetricMeshAdaptation() | RelocationMeshAdaptation():
                    pass
                case _:
                    raise TypeError(
                        f"MeshAdaptation.request must provide a metric or relocation request; got {type(request).__name__!r}"
                    )
            if kinds != {"triangle"} or mesh.ambient_dimension != 2:
                raise ValueError(
                    f"MeshAdaptation.source_mesh must contain planar triangle blocks for the requested metric route; got {(sorted(kinds), mesh.ambient_dimension)!r}"
                )
            _require_native_source(
                source,
                policy.association_transfer,
                nested=False,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case MeshAdaptationRoute.NATIVE_METRIC_3D:
            match request:
                case MetricMeshAdaptation() | RelocationMeshAdaptation():
                    pass
                case _:
                    raise TypeError(
                        f"MeshAdaptation.request must provide a metric or relocation request; got {type(request).__name__!r}"
                    )
            if kinds != {"tetrahedron"} or mesh.ambient_dimension != 3:
                raise ValueError(
                    f"MeshAdaptation.source_mesh must contain tetrahedron blocks in 3D for NATIVE_METRIC_3D; got {(sorted(kinds), mesh.ambient_dimension)!r}"
                )
            if mesh.periodic_topology is not None:
                raise ValueError(
                    f"MeshAdaptation.source_mesh.periodic_topology must be absent without an orbit-synchronized 3D metric transaction; got {mesh.periodic_topology is not None!r}"
                )
            _require_native_source(
                source,
                policy.association_transfer,
                nested=False,
                plc_ancestry=True,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case MeshAdaptationRoute.NATIVE_SURFACE_METRIC:
            _validate_surface_route(
                source,
                request,
                policy,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case MeshAdaptationRoute.NATIVE_LEVEL_SET:
            match request:
                case _owner_phydrax_meshing__level_set.LevelSetMeshAdaptation():
                    pass
                case _:
                    raise TypeError(
                        f"MeshAdaptation.request must provide LevelSetMeshAdaptation for NATIVE_LEVEL_SET; got {type(request).__name__!r}"
                    )
            if kinds != {"tetrahedron"} or mesh.ambient_dimension != 3:
                raise ValueError(
                    f"MeshAdaptation.source_mesh must contain tetrahedron blocks in 3D for NATIVE_LEVEL_SET; got {(sorted(kinds), mesh.ambient_dimension)!r}"
                )
            # Zero insertion is a nested refinement: declared exact rational and
            # algebraic sources travel through the nested geometry transition,
            # and the level-set owner admits only affine source maps. PLC
            # certificate carriers publish roots of the certifying polytope, not
            # the correctly rounded exact roots, so PLC ancestry stays refused.
            if isinstance(
                source.geometry.exact_source,
                (
                    _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource,
                    _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource,
                ),
            ):
                raise ValueError(
                    f"MeshAdaptation.geometry_transition must carry PLC source ancestry on this route; got {source.result_id!r}"
                )
            _require_native_source(
                source,
                policy.association_transfer,
                nested=True,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case MeshAdaptationRoute.MMG | MeshAdaptationRoute.OMEGA_H:
            match request:
                case MetricMeshAdaptation():
                    pass
                case _:
                    raise TypeError(
                        f"MeshAdaptation.request must provide a metric request for provider remeshing; got {type(request).__name__!r}"
                    )
            require_affine_geometry(source, budget=budget)
            if source.region_evidence is not None:
                raise ValueError(
                    f"MeshAdaptation.source.region_evidence must be absent when provider remeshing has unknown lineage; got {source.region_evidence is not None!r}"
                )
            if isinstance(
                policy.association_transfer,
                (
                    _dependency_phydrax_meshing__association.PlcAssociationTransfer,
                    _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer,
                ),
            ):
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must use topology lineage rather than provider rederivation for PLC or surface sources; got {type(policy.association_transfer).__name__!r}"
                )
            if isinstance(
                policy.association_transfer,
                _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
            ):
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must use topology lineage rather than provider rederivation for PLC or surface sources; got {type(policy.association_transfer).__name__!r}"
                )
        case MeshAdaptationRoute.HP:
            _validate_hp_route(
                source,
                request,
                policy,
                budget=budget,
                scope=scope,
                source_preparation=source_preparation,
            )
        case invalid:
            assert_never(invalid)


def _prepared_adaptation_identity(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
) -> str:
    payload: dict[str, object] = {
        "kind": "prepared-mesh-adaptation",
        "source": source.result_id,
        "request": request.request_id,
        "policy": policy.policy_id,
    }
    if source.mesh.storage is None and policy.route in (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.NATIVE_MIXED,
    ):
        payload["source_certificates"] = (
            cell_predecessor_owner.cell_certificate_source_content_id(
                source.mesh,
                source.geometry,
                source.audit.validity,
                None if source.certification is None else source.certification.embedding,
            )
        )
    return canonical_fingerprint(payload)


@final
class _PlcSourceProofs:
    """Live PLC source-support proofs earned by one preparation.

    This is an execution resource, not preparation content: it stays outside
    ``prepared_id``, every attachment compares content-equal, and its stored
    form is detached (no proofs), so a restored preparation re-prepares and
    owns that cost. Its issued attachment binds the exact ended preparation
    receipt; without that receipt, mapped proofs import their own charges.
    """

    __slots__ = ("owner_id", "proofs", "__weakref__")

    owner_id: str
    proofs: tuple[
        plc_mapped_support_owner.MappedPlcSupportProof
        | plc_planar_support_owner.PlanarPlcSourceProof,
        ...,
    ]

    def __init__(
        self,
        owner_id: str,
        proofs: tuple[
            plc_mapped_support_owner.MappedPlcSupportProof
            | plc_planar_support_owner.PlanarPlcSourceProof,
            ...,
        ],
        /,
    ) -> None:
        object.__setattr__(self, "owner_id", owner_id)
        object.__setattr__(self, "proofs", proofs)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"_PlcSourceProofs.{name} must remain the earned attachment; got assignment"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _PlcSourceProofs)

    def __hash__(self) -> int:
        return hash(_PlcSourceProofs)

    def to_record(self) -> dict[str, object]:
        """Store only the detached form; live proofs never enter an archive."""
        return {}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> _PlcSourceProofs:
        if record:
            raise ValueError(
                f"_PlcSourceProofs.record must be the detached empty form; got keys {sorted(record)!r}"
            )
        return cls("", ())


# Identity-keyed because all attachments deliberately compare content-equal.
# Values retain only the earning receipt and sealed fields, never the attachment.
_ISSUED_PLC_SOURCE_PROOFS: dict[
    int,
    tuple[
        weakref.ReferenceType[_PlcSourceProofs],
        str,
        tuple[
            plc_mapped_support_owner.MappedPlcSupportProof
            | plc_planar_support_owner.PlanarPlcSourceProof,
            ...,
        ],
        NativeExecutionRecord,
    ],
] = {}


def _issued_plc_source_proofs(
    owner_id: str,
    proofs: tuple[
        plc_mapped_support_owner.MappedPlcSupportProof
        | plc_planar_support_owner.PlanarPlcSourceProof,
        ...,
    ],
    preparation: NativeExecutionRecord,
    /,
) -> _PlcSourceProofs:
    """Seal live support to the exact preparation phase that earned it."""
    attached = _PlcSourceProofs(owner_id, proofs)
    identity = id(attached)
    reference = weakref.ref(
        attached, lambda _: _ISSUED_PLC_SOURCE_PROOFS.pop(identity, None)
    )
    _ISSUED_PLC_SOURCE_PROOFS[identity] = (
        reference,
        owner_id,
        proofs,
        preparation,
    )
    return attached


@final
class PreparedMeshAdaptation(StrictModule, NonTrainableState):
    """Bind a request to a source revision under one policy.

    ``preparation_evidence`` is the ended preparation phase of this request.
    Execution charges it once against the same original allowance, so work
    done before ``execute_mesh_adaptation`` never escapes the request limits.
    """

    source: _dependency_phydrax_meshing__result.CellMeshingResult
    request: MeshAdaptationRequest
    policy: MeshAdaptationPolicy
    constraints: _AdaptationConstraints
    prepared_id: str = eqx.field(static=True)
    periodic_source_proofs: tuple[
        _dependency_phydrax_meshing_providers__native_periodic._PeriodicSourceAssociationPreparation,
        ...,
    ]
    periodic_source_receipt: (
        _dependency_phydrax_meshing_providers__native_periodic._PeriodicSourceAssociationReceipt
        | None
    )
    plc_source_proofs: _PlcSourceProofs | None
    preparation_evidence: NativeExecutionRecord | None

    @checked
    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        request: MeshAdaptationRequest,
        policy: MeshAdaptationPolicy,
        constraints: _AdaptationConstraints,
        /,
        *,
        periodic_source_proofs: tuple[
            _dependency_phydrax_meshing_providers__native_periodic._PeriodicSourceAssociationPreparation,
            ...,
        ] = (),
        periodic_source_receipt: _dependency_phydrax_meshing_providers__native_periodic._PeriodicSourceAssociationReceipt
        | None = None,
        plc_source_proofs: _PlcSourceProofs | None = None,
        preparation_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        self.source = source
        self.request = request
        self.policy = policy
        self.constraints = constraints
        self.prepared_id = _prepared_adaptation_identity(source, request, policy)
        if periodic_source_receipt is not None:
            # The periodic receipt's charges are this same preparation phase.
            if preparation_evidence is None:
                preparation_evidence = periodic_source_receipt.evidence
            elif (
                preparation_evidence.receipt_hex()
                != periodic_source_receipt.evidence.receipt_hex()
            ):
                raise ValueError(
                    "PreparedMeshAdaptation.preparation_evidence must be the periodic "
                    "receipt's preparation phase; got another ended record"
                )
        if preparation_evidence is not None:
            preparation_evidence.require_valid()
            if preparation_evidence.owner_id != self.prepared_id:
                raise ValueError(
                    "PreparedMeshAdaptation.preparation_evidence.owner_id must bind this "
                    f"preparation; got {preparation_evidence.owner_id!r}"
                )
        self.periodic_source_proofs = periodic_source_proofs
        self.periodic_source_receipt = periodic_source_receipt
        self.plc_source_proofs = plc_source_proofs
        self.preparation_evidence = preparation_evidence

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PreparedMeshAdaptation],
        value: PreparedMeshAdaptation,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> PreparedMeshAdaptation:
        """Authenticate retained preparation without repeating charged native work."""
        _validate_prepared_constraint_bindings(value)
        # Lazy: the envelope owner also uses native volume generation.
        from ._surface_envelope import _require_surface_envelope_result_anchor

        # Retained envelope history belongs to the policy's authored anchor.
        _require_surface_envelope_result_anchor(
            value.policy.envelope_source,
            value.source,
            limits=value.policy.limits,
            budget=budget,
        )
        rebuilt = cls(
            value.source,
            value.request,
            value.policy,
            value.constraints,
            periodic_source_proofs=value.periodic_source_proofs,
            periodic_source_receipt=value.periodic_source_receipt,
            plc_source_proofs=value.plc_source_proofs,
            preparation_evidence=value.preparation_evidence,
        )
        for member in dataclasses.fields(value):
            if not tree_content_equal(
                getattr(rebuilt, member.name), getattr(value, member.name)
            ):
                raise ValueError(
                    f"PreparedMeshAdaptation.{member.name} must match retained preparation; got changed stored content"
                )
        return rebuilt


def _validate_prepared_constraint_bindings(value: PreparedMeshAdaptation, /) -> None:
    """Validate stored host constraints against retained topology and metric inputs."""
    mesh = value.source.mesh
    constraints = value.constraints
    match value.policy.route:
        case (
            MeshAdaptationRoute.MMG
            | MeshAdaptationRoute.OMEGA_H
            | MeshAdaptationRoute.HP
            | MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION
        ):
            if not tree_content_equal(constraints, _empty_constraints(mesh)):
                raise ValueError(
                    "PreparedMeshAdaptation.constraints must match the empty route binding; got different stored arrays"
                )
            return
        case (
            MeshAdaptationRoute.NATIVE_BISECTION
            | MeshAdaptationRoute.NATIVE_METRIC_2D
            | MeshAdaptationRoute.NATIVE_METRIC_3D
            | MeshAdaptationRoute.NATIVE_SURFACE_METRIC
            | MeshAdaptationRoute.NATIVE_LEVEL_SET
            | MeshAdaptationRoute.NATIVE_MIXED
            | MeshAdaptationRoute.NATIVE_POLYHEDRAL
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.DEVICE_METRIC_2D
        ):
            pass
        case invalid:
            assert_never(invalid)
    dimension = mesh.topological_dimension
    edge_count = mesh.entity_set(1).count
    vertex_count = mesh.coordinates.shape[0]
    for name, shape, dtype in (
        ("protected_edge_mask", (edge_count,), np.dtype(np.bool_)),
        ("fixed_vertex_mask", (vertex_count,), np.dtype(np.bool_)),
        ("cell_classes", (mesh.entity_set(dimension).count,), np.dtype(np.int64)),
        ("facet_classes", (mesh.entity_set(dimension - 1).count,), np.dtype(np.int64)),
        ("edge_classes", (edge_count,), np.dtype(np.int64)),
    ):
        array = getattr(constraints, name)
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != dtype
        ):
            raise ValueError(
                f"PreparedMeshAdaptation.constraints.{name} must match source shape {shape} and dtype {dtype}; got incompatible stored array"
            )
        if dtype == np.dtype(np.int64) and np.any(array < 0):
            raise ValueError(
                f"PreparedMeshAdaptation.constraints.{name} must contain nonnegative class identities; got negative values"
            )
    expected_edges = entity_keys_owner.entity_keys(mesh, 1)[
        constraints.protected_edge_mask
    ]
    if not tree_content_equal(constraints.protected_edge_keys, expected_edges):
        raise ValueError(
            "PreparedMeshAdaptation.constraints.protected_edge_keys must match the source edge mask; got changed keys"
        )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    selected = np.isin(vertex_ids, constraints.protected_vertex_ids)
    if not tree_content_equal(
        constraints.protected_vertex_ids, vertex_ids[selected]
    ) or np.any(selected & ~constraints.fixed_vertex_mask):
        raise ValueError(
            "PreparedMeshAdaptation.constraints.protected_vertex_ids must select fixed source vertices in source order; got incompatible IDs or mask"
        )
    metric = (
        _metric_rows(mesh, value.request.metric)
        if isinstance(value.request, (MetricMeshAdaptation, RelocationMeshAdaptation))
        else np.zeros(
            (0, mesh.ambient_dimension, mesh.ambient_dimension), dtype=np.float64
        )
    )
    if not tree_content_equal(constraints.metric_values, metric):
        raise ValueError(
            "PreparedMeshAdaptation.constraints.metric_values must match the retained metric request; got changed values"
        )


def _periodic_polyhedral_policy(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MeshAdaptationPolicy:
    """Derive the effective PLC transfer of a periodic power edit from its retained source.

    An absent transfer means the caller supplied none. A periodic exact-power
    source retains its authored PLC in its geometry root; that PLC, the single
    association namespace and the certified coverage domain are the only
    authorities. Only the authored root is read here: execution prepares the
    source coordinates once for the edit itself. The transfer is prepared
    inside this request's scope and the returned policy differs only by it, so
    its identity names the effective transfer. Any other source keeps the
    explicit-transfer contract.
    """
    if (
        policy.route is not MeshAdaptationRoute.NATIVE_POLYHEDRAL
        or policy.association_transfer is not None
        or not source.associations
        or source.mesh.periodic_topology is None
        or not isinstance(
            request,
            _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralMeshAdaptation,
        )
        or request.operation
        is _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationOperation.REGENERATE
    ):
        return policy
    root = _owner_phydrax_meshing__topology_edit.power_domain_source(source.geometry)
    complex_ = root.domain_source
    if complex_ is None:
        raise ValueError(
            "MeshAdaptation.source.geometry must retain its authored PLC; got no domain source"
        )
    namespaces = {
        (association.source_id, association.source_revision)
        for association in source.associations
    }
    certificate = source.certification
    if (
        len(namespaces) != 1
        or certificate is None
        or not certificate.passed
        or certificate.request.domain is None
    ):
        raise ValueError(
            f"MeshAdaptation.source must bind one PLC namespace and certified coverage; got namespaces {sorted(namespaces)!r}"
        )
    ((source_id, source_revision),) = namespaces
    # Use the same preadmitted host tables and mapped native phase as source
    # generation. A bare PLC call here loses the native admission observation
    # and allocates source-transfer tables outside the coordinate ledger.
    from .providers._native_polyhedral import _polyhedral_support_source

    transfer = _polyhedral_support_source(
        complex_, source_id, source_revision, source.coordinate_contract,
        policy.limits, None, scope=scope, budget=budget,
    ).association_transfer
    if transfer.domain.domain_id != certificate.request.domain.domain_id:
        raise ValueError(
            "MeshAdaptation.source.geometry must retain the certified PLC domain; got a different domain"
        )
    return MeshAdaptationPolicy(
        policy.route,
        protected_scopes=policy.protected_scopes,
        compatibility=policy.compatibility,
        predicate_mode=policy.predicate_mode,
        maximum_passes=policy.maximum_passes,
        maximum_closure_iterations=policy.maximum_closure_iterations,
        relocation=policy.relocation,
        limits=policy.limits,
        audit_policy=policy.audit_policy,
        device_policy=policy.device_policy,
        provider=policy.provider,
        provider_options=policy.provider_options,
        distribution=policy.distribution,
        partition_policy=policy.partition_policy,
        association_transfer=transfer,
        geometry_transition=policy.geometry_transition,
        envelope_source=policy.envelope_source,
    )


@checked
def prepare_mesh_adaptation(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: MeshAdaptationRequest,
    /,
    *,
    policy: MeshAdaptationPolicy,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
    observe_managed_admission: bool = False,
) -> PreparedMeshAdaptation:
    """Validate the route/request pairing and resolve protection and organization.

    Managed admission is optional observation of this preparation's actual scope;
    it does not change its limits, deadline, or source-proof authority.
    """
    # Lazy: wrapping uses native volume generation.
    from ._surface_envelope import (
        _require_surface_envelope_adaptation_source,
        _surface_envelope_repair_payload,
    )

    limits = policy.limits
    periodic_binding: tuple[tuple[int, ...], str] | None = None
    with NativeExecutionScope(limits.native_limits()) as scope, ExitStack() as stack:
        if observe_managed_admission or policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            scope.observe_managed_admission()
        if budget is None:
            initial = scope.batch(0)
            budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
                initial,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            initial.settle()
        elif budget.ticket.settled:
            stack.enter_context(budget.bind_ticket(scope.ticket_view()))
        with (
            budget.retained_storage(
                lambda amount: (
                    scope.ticket_view()
                    if policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL
                    else scope.batch(0, bytes_upper=amount)
                )
            )
            if budget.storage_ticket is None and budget.ticket.settled
            else nullcontext()
        ):
            with adaptation_phase(scope, budget):
                with _surface_envelope_repair_payload(source, budget=budget) as payload:
                    source.audit.require_passed()
                    _require_surface_envelope_adaptation_source(
                        source, payload, policy.envelope_source
                    )
                if policy.envelope_source is not None and source.mesh.storage is not None:
                    raise ValueError(
                        "MeshAdaptationPolicy.envelope_source must adapt a serial source; got partitioned storage, whose collective publication carries no envelope history"
                    )
            # Source preparation opens its own native phase within this scope;
            # an open adaptation phase would reserve the whole allowance first.
            policy = _periodic_polyhedral_policy(
                source, request, policy, scope=scope, budget=budget
            )
            with adaptation_phase(scope, budget):
                source_preparation = _RouteSourcePreparation([], [], [])
                _check_route_request(
                    source,
                    request,
                    policy,
                    budget=budget,
                    scope=scope,
                    source_preparation=source_preparation,
                )
                if policy.distribution is not None:
                    policy.distribution.require_current(
                        mesh_part_owner.MeshPart(policy.distribution.part.name, source)
                    )
                match policy.route:
                    case (
                        MeshAdaptationRoute.NATIVE_BISECTION
                        | MeshAdaptationRoute.NATIVE_METRIC_2D
                        | MeshAdaptationRoute.NATIVE_METRIC_3D
                        | MeshAdaptationRoute.NATIVE_SURFACE_METRIC
                        | MeshAdaptationRoute.NATIVE_LEVEL_SET
                        | MeshAdaptationRoute.NATIVE_MIXED
                        | MeshAdaptationRoute.NATIVE_POLYHEDRAL
                        | MeshAdaptationRoute.DEVICE_BISECTION
                        | MeshAdaptationRoute.DEVICE_METRIC_2D
                    ):
                        constraints = _resolve_constraints(
                            source,
                            request,
                            policy,
                            budget=budget,
                            scope=scope,
                            source_preparation=source_preparation,
                        )
                    case (
                        MeshAdaptationRoute.MMG
                        | MeshAdaptationRoute.OMEGA_H
                        | MeshAdaptationRoute.HP
                        | MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION
                    ):
                        for protected_scope in policy.protected_scopes:
                            _dependency_phydrax_meshing__scope.resolve_mesh_scope(
                                source.mesh, protected_scope
                            )
                        constraints = _empty_constraints(source.mesh)
                    case invalid:
                        assert_never(invalid)
                prepared = PreparedMeshAdaptation(
                    source,
                    request,
                    policy,
                    constraints,
                    periodic_source_proofs=tuple(source_preparation.periodic_proofs),
                )
                if policy.route is MeshAdaptationRoute.NATIVE_LEVEL_SET:
                    _owner_phydrax_meshing__level_set.validate_level_set_adaptation(
                        prepared, budget=budget
                    )
                if (
                    policy.route is MeshAdaptationRoute.NATIVE_BISECTION
                    and source.mesh.periodic_topology is not None
                ):
                    _owner_phydrax_meshing__periodic.validate_periodic_bisection(
                        prepared, budget=budget
                    )
                    periodic_binding = (
                        _owner_phydrax_meshing__periodic._periodic_preparation_binding(
                            prepared, budget=budget
                        )
                    )
                power_witnesses = tuple(budget.power_witness_cache.values())
        checked_at = time.monotonic()
        expires_at = checked_at + scope.remaining().remaining_seconds
    plc_proofs = (
        *(view.proof for view in source_preparation.plc_proofs),
        *source_preparation.planar_proofs,
    )
    if policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL:
        from .providers._native_publication import _require_retained_bytes

        preparation_bytes = _NATIVE_EXECUTION_RECORD_ARRAY_BYTES - (
            8
            if scope.completed_evidence().managed_admission_peak_bytes_upper is None
            else 0
        )
        _require_retained_bytes(
            model_array_bytes((prepared, plc_proofs)) + preparation_bytes,
            limits,
        )
    preparation = NativeExecutionRecord(
        scope.completed_evidence(), owner_id=prepared.prepared_id
    )
    parent = scope.parent
    if parent is not None:
        # Nested preparation work already counts in the enclosing request.
        native_preparation_ledger(parent).remember_receipt(parent, preparation)
    if plc_proofs:
        # The ledger-free proofs outlive this scope; their preparation charges
        # are inside this receipt, which the binding execution charges once.
        # The constructor keeps the source objects the proofs were earned for.
        return _remember_power_witnesses(
            PreparedMeshAdaptation(
                source,
                request,
                policy,
                constraints,
                plc_source_proofs=_issued_plc_source_proofs(
                    prepared.prepared_id, plc_proofs, preparation
                ),
                preparation_evidence=preparation,
            ),
            power_witnesses,
            publication_expires_at=(
                expires_at
                if policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL
                else None
            ),
        )
    if not prepared.periodic_source_proofs and periodic_binding is None:
        return _remember_power_witnesses(
            PreparedMeshAdaptation(
                source, request, policy, constraints, preparation_evidence=preparation
            ),
            power_witnesses,
            publication_expires_at=(
                expires_at
                if policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL
                else None
            ),
        )
    return _remember_power_witnesses(
        _close_periodic_preparation_receipt(
            prepared, preparation, expires_at, periodic_binding
        ),
        power_witnesses,
        publication_expires_at=(
            expires_at if policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL else None
        ),
    )


# Live-only exact-power source constructions earned by one preparation call,
# keyed by the returned object. Their work is inside its preparation receipt,
# which the binding execution charges once; they are never archived, so a
# restored preparation prepares its sources again inside its execution.
_LIVE_POWER_WITNESSES: dict[
    int,
    tuple[
        weakref.ReferenceType[PreparedMeshAdaptation],
        tuple[
            _dependency_phydrax_discretization__exact_power_geometry.PreparedExactPowerGeometry,
            ...,
        ],
        float | None,
    ],
] = {}


def _remember_power_witnesses(
    prepared: PreparedMeshAdaptation,
    witnesses: tuple[
        _dependency_phydrax_discretization__exact_power_geometry.PreparedExactPowerGeometry,
        ...,
    ],
    /,
    *,
    publication_expires_at: float | None = None,
) -> PreparedMeshAdaptation:
    _require_adaptation_publication_deadline(
        publication_expires_at, prepared.policy.limits.maximum_wall_seconds
    )
    if witnesses or publication_expires_at is not None:
        identity = id(prepared)
        reference = weakref.ref(
            prepared, lambda _: _LIVE_POWER_WITNESSES.pop(identity, None)
        )
        _LIVE_POWER_WITNESSES[identity] = (
            reference, witnesses, publication_expires_at
        )
        try:
            _require_adaptation_publication_deadline(
                publication_expires_at, prepared.policy.limits.maximum_wall_seconds
            )
        except BaseException:
            _LIVE_POWER_WITNESSES.pop(identity, None)
            raise
    return prepared


def _import_power_witnesses(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Retain this preparation's earned power constructions in its execution."""
    entry = _LIVE_POWER_WITNESSES.get(id(prepared))
    if entry is None or entry[0]() is not prepared:
        return
    for witness in entry[1]:
        _dependency_phydrax_discretization__exact_power_geometry._retain_power_witness(
            witness, budget=budget
        )


def _close_periodic_preparation_receipt(
    prepared: PreparedMeshAdaptation,
    evidence: NativeExecutionRecord,
    expires_at: float,
    binding: tuple[tuple[int, ...], str] | None,
) -> PreparedMeshAdaptation:
    """Publish ended charges and live-only mark-rebinding authority together."""
    receipt = _dependency_phydrax_meshing_providers__native_periodic._PeriodicSourceAssociationReceipt(
        prepared.periodic_source_proofs,
        evidence,
        expires_at,
    )
    result = PreparedMeshAdaptation(
        prepared.source,
        prepared.request,
        prepared.policy,
        prepared.constraints,
        periodic_source_proofs=prepared.periodic_source_proofs,
        periodic_source_receipt=receipt,
    )
    if binding is not None:
        _owner_phydrax_meshing__periodic._remember_periodic_preparation(result, binding)
    return result


def _empty_constraints(mesh: CellMesh, /) -> _AdaptationConstraints:
    empty = np.zeros((0,), dtype=np.int64)
    return _AdaptationConstraints(
        protected_edge_keys=np.zeros((0, 2), dtype=np.int64),
        protected_edge_mask=np.zeros((0,), dtype=np.bool_),
        protected_vertex_ids=empty,
        fixed_vertex_mask=np.zeros((0,), dtype=np.bool_),
        cell_classes=empty,
        facet_classes=empty,
        edge_classes=empty,
        metric_values=np.zeros((0, mesh.ambient_dimension, mesh.ambient_dimension)),
    )


type MeshAdaptationEvidence = (
    BisectionEvidence
    | LocalMetricEvidence
    | DeviceMetricEvidence
    | MetricRemeshingEvidence
    | _owner_phydrax_meshing__level_set.LevelSetEvidence
    | _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationEvidence
    | _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationEvidence
    | CurvedGeometryEvidence
    | MmgAdaptationResult
    | OmegaHAdaptationResult
    | FiniteElementHPRefinementResult
)


def _evidence_id(evidence: MeshAdaptationEvidence | None, /) -> str | None:
    match evidence:
        case None:
            return None
        case (
            BisectionEvidence()
            | LocalMetricEvidence()
            | DeviceMetricEvidence()
            | MetricRemeshingEvidence()
            | _owner_phydrax_meshing__level_set.LevelSetEvidence()
            | _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationEvidence()
            | _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationEvidence()
            | CurvedGeometryEvidence()
        ):
            return evidence.evidence_id
        case (
            MmgAdaptationResult()
            | OmegaHAdaptationResult()
            | FiniteElementHPRefinementResult()
        ):
            return evidence.result_id
        case invalid:
            assert_never(invalid)


@final
class MeshAdaptationResult(StrictModule, NonTrainableState):
    """Certified adaptation target with lineage and execution evidence.

    A source-preserving result omits transition, lineage, stencil, and transfer.
    Its status is UNCHANGED only if work was not refused and all convergence
    criteria are met; otherwise PARTIAL, STALLED, PASS_LIMIT, or RESOURCE_LIMIT
    report the unmet contract. Provider routes and nonnested polyhedral vertices
    without interpolation ancestry have no nodal stencil or transfer.
    The transfer maps source vertex values to target vertices when ancestry
    supports that map. The common refinement carries certified geometric
    transfer of polyhedral cells. The request records the executed transaction,
    the metric is target-bound, and hierarchy continues marked adaptation.
    Elapsed wall time is evidence outside result_id.
    """

    status: MeshAdaptationStatus = eqx.field(static=True)
    route: MeshAdaptationRoute = eqx.field(static=True)
    request: MeshAdaptationRequest
    policy: MeshAdaptationPolicy
    source: _dependency_phydrax_meshing__result.CellMeshingResult
    target: _dependency_phydrax_meshing__result.CellMeshingResult
    transition: _dependency_phydrax_meshing__lineage.CellMeshTransition | None
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage | None
    stencil: (
        _dependency_phydrax_meshing__lineage_records.VertexInterpolationStencil | None
    )
    transfer: FiniteElementTopologyTransfer | None
    metric: _dependency_phydrax_meshing__metric.MeshMetricField | None
    compliance: _dependency_phydrax_meshing__result.MeshingComplianceReport
    evidence: MeshAdaptationEvidence | None
    common_refinement: (
        _dependency_phydrax_geometry__supermesh.PreparedCommonRefinement | None
    )
    hierarchy: MeshAdaptationHierarchy
    distribution: (
        _dependency_phydrax_meshing__distribution.MeshDistributionTransition | None
    )
    elapsed_seconds: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        prepared: PreparedMeshAdaptation,
        outcome: RouteOutcome,
        compliance: _dependency_phydrax_meshing__result.MeshingComplianceReport,
        distribution: _dependency_phydrax_meshing__distribution.MeshDistributionTransition
        | None,
        elapsed_seconds: float,
        /,
    ) -> None:
        source_preserved = outcome.target.result_id == prepared.source.result_id
        absent = (
            outcome.transition is None
            and outcome.lineage is None
            and outcome.stencil is None
            and outcome.transfer is None
        )
        if source_preserved != absent:
            raise ValueError(
                f"MeshAdaptationResult.transition must omit transition and transfer evidence when the source is preserved; got {(source_preserved, absent)!r}"
            )
        if outcome.status is MeshAdaptationStatus.UNCHANGED and not source_preserved:
            raise ValueError(
                f"MeshAdaptationResult.status must preserve the source result for UNCHANGED; got {outcome.target.result_id!r}"
            )
        if outcome.transition is not None:
            if outcome.lineage is None:
                raise ValueError(
                    f"MeshAdaptationResult.transition must retain its lineage; got {outcome.lineage!r}"
                )
            if (
                outcome.transition.target.result_id != outcome.target.result_id
                or outcome.transition.lineage.lineage_id != outcome.lineage.lineage_id
            ):
                raise ValueError(
                    f"MeshAdaptationResult.transition must describe the certified target and its lineage; got {outcome.transition.target.result_id!r}"
                )
        common = outcome.common_refinement
        if common is not None and (
            common.source_mesh_id != prepared.source.mesh.mesh_id
            or common.target_mesh_id != outcome.target.mesh.mesh_id
            or common.source_topology_id != prepared.source.mesh.topology_id
            or common.target_topology_id != outcome.target.mesh.topology_id
        ):
            raise ValueError(
                f"MeshAdaptationResult.common_refinement must bind the source and accepted target; got {(common.source_mesh_id, common.target_mesh_id)!r}"
            )
        self.status = outcome.status
        self.route = prepared.policy.route
        self.request = prepared.request
        self.policy = prepared.policy
        self.source = prepared.source
        self.target = outcome.target
        self.transition = outcome.transition
        self.lineage = outcome.lineage
        self.stencil = outcome.stencil
        self.transfer = outcome.transfer
        self.metric = outcome.metric
        self.compliance = compliance
        self.evidence = outcome.evidence
        self.common_refinement = outcome.common_refinement
        self.hierarchy = outcome.hierarchy
        self.distribution = distribution
        self.elapsed_seconds = float(elapsed_seconds)
        self.prepared_id = prepared.prepared_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-result",
                "prepared": prepared.prepared_id,
                "status": outcome.status.value,
                "target": outcome.target.result_id,
                "transition": None
                if outcome.transition is None
                else outcome.transition.transition_id,
                "transfer": None
                if outcome.transfer is None
                else outcome.transfer.transfer_id,
                "metric": None if outcome.metric is None else outcome.metric.metric_id,
                "compliance": compliance.report_id,
                "evidence": _evidence_id(outcome.evidence),
                "common_refinement": None
                if outcome.common_refinement is None
                else outcome.common_refinement.refinement_id,
                "hierarchy": _hierarchy_id(outcome.hierarchy),
                "distribution": None
                if distribution is None
                else distribution.transition_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MeshAdaptationResult],
        value: MeshAdaptationResult,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> MeshAdaptationResult:
        """Bind retained accepted transitions without executing another adaptation."""
        collective = value.target.collective_evidence
        match collective:
            case _dependency_phydrax_meshing__result.CollectiveMeshEvidence(
                preparation=prepared
            ):
                pass
            case _:
                prepared = prepare_mesh_adaptation(
                    value.source, value.request, policy=value.policy, budget=budget
                )
        if prepared.prepared_id != value.prepared_id:
            raise ValueError(
                f"MeshAdaptationResult.prepared_id must match retained preparation; got {value.prepared_id!r}"
            )
        if value.route is not prepared.policy.route:
            raise ValueError(
                f"MeshAdaptationResult.route must match retained policy; got {value.route!r}"
            )
        if value.target.result_id != value.source.result_id:
            # Lazy: the envelope owner also uses native volume generation.
            from ._surface_envelope import _require_anchored_surface_envelope_target

            _require_anchored_surface_envelope_target(
                prepared.policy.envelope_source,
                value.target,
                limits=prepared.policy.limits,
                budget=budget,
            )
        outcome = RouteOutcome(
            value.status,
            value.target,
            value.transition,
            value.lineage,
            value.stencil,
            value.transfer,
            value.metric,
            value.evidence,
            value.hierarchy,
            value.common_refinement,
        )
        rebuilt = cls(
            prepared, outcome, value.compliance, value.distribution, value.elapsed_seconds
        )
        if rebuilt.result_id != value.result_id:
            raise ValueError(
                f"MeshAdaptationResult.result_id must match retained transition; got {value.result_id!r}"
            )
        return rebuilt

    @property
    def geometry_transition(
        self,
    ) -> transition_types_owner.CellGeometryTransition | None:
        """How a non-affine source coordinate map reached the target, if any."""

        return None if self.transition is None else self.transition.geometry_transition

    def _parent_ids(self) -> np.ndarray:
        transition = self.transition
        if transition is None or transition.parent_cell_ids is None:
            raise ValueError(
                f"MeshAdaptationResult.transition must provide nested parent maps; got {self.route.name!r}"
            )
        parents = np.asarray(transition.parent_cell_ids, dtype=np.int64)
        if np.any(parents < 0):
            raise ValueError(
                f"MeshAdaptationResult.transition must provide a single parent for each target cell; got {self.route.name!r}"
            )
        return parents

    @property
    def parent_cells(self) -> np.ndarray:
        """Return the source row containing each target cell, in concatenated block order.

        These are the parent_cells input to prepare_nested_field_transfer.
        Refuse when refinement lineage is absent or a target cell was coarsened.
        """

        parents = self._parent_ids()
        ids = np.concatenate(
            [np.asarray(block.global_ids, np.int64) for block in self.source.mesh.blocks]
        )
        order = np.argsort(ids, kind="stable")
        return order[np.searchsorted(ids[order], parents)]

    @property
    def parent_reference_vertices(self) -> np.ndarray:
        """Target reference corners in their source parents, with family-dependent extent."""

        self._parent_ids()
        transition = self.transition
        if transition is None or transition.parent_reference_vertices is None:
            raise ValueError(
                f"MeshAdaptationResult.transition must provide nested parent maps; got {self.route.name!r}"
            )
        return np.asarray(transition.parent_reference_vertices, dtype=np.float64)

    @property
    def parent_reference_homogeneous(self) -> np.ndarray | None:
        """Rational source-owned corners of ``parent_reference_vertices``, if retained."""

        self._parent_ids()
        transition = self.transition
        if transition is None or transition.parent_reference_vertices is None:
            raise ValueError(
                f"MeshAdaptationResult.transition must provide nested parent maps; got {self.route.name!r}"
            )
        bank = transition.parent_reference_homogeneous
        return None if bank is None else np.asarray(bank, dtype=np.float64)

    @property
    def coarsening_witnesses(
        self,
    ) -> transition_types_owner.NestedReferenceWitnesses | None:
        """Return retained fine-to-coarse charts from the coarsening transition."""
        transition = self.transition
        if transition is None:
            return None
        if (
            self.route is MeshAdaptationRoute.NATIVE_BISECTION
            and self.source.mesh.periodic_topology is not None
            and transition.transition_kind
            is _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.COARSEN
        ):
            return _owner_phydrax_meshing__periodic.periodic_coarsening_witnesses(self)
        if transition.coarsened_cell_ids is not None:
            bank = transition.coarsened_reference_homogeneous
            return transition_types_owner.NestedReferenceWitnesses(
                np.asarray(transition.coarsened_cell_ids, dtype=np.int64),
                np.asarray(transition.coarsened_into_ids, dtype=np.int64),
                np.asarray(transition.coarsened_reference_vertices, dtype=np.float64),
                None if bank is None else np.asarray(bank, dtype=np.float64),
            )
        geometry = transition.geometry_transition
        if geometry is None or geometry.coarsened_cell_ids.shape[0] == 0:
            return None
        return geometry.coarsening_witnesses()


class RouteOutcome(NamedTuple):
    status: MeshAdaptationStatus
    target: _dependency_phydrax_meshing__result.CellMeshingResult
    transition: _dependency_phydrax_meshing__lineage.CellMeshTransition | None
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage | None
    stencil: (
        _dependency_phydrax_meshing__lineage_records.VertexInterpolationStencil | None
    )
    transfer: FiniteElementTopologyTransfer | None
    metric: _dependency_phydrax_meshing__metric.MeshMetricField | None
    evidence: MeshAdaptationEvidence | None
    hierarchy: MeshAdaptationHierarchy
    common_refinement: (
        _dependency_phydrax_geometry__supermesh.PreparedCommonRefinement | None
    ) = None


@checked
def unchanged(
    prepared: PreparedMeshAdaptation,
    evidence: MeshAdaptationEvidence | None,
    hierarchy: MeshAdaptationHierarchy,
    /,
    *,
    status: MeshAdaptationStatus = MeshAdaptationStatus.UNCHANGED,
) -> RouteOutcome:
    return RouteOutcome(
        status, prepared.source, None, None, None, None, None, evidence, hierarchy
    )


def _p1_vertex_measures(mesh: CellMesh, /) -> np.ndarray:
    """Integral of every P1 hat function: each simplex measure over its vertices."""

    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    measures = np.zeros((coordinates.shape[0],), dtype=np.float64)
    dimension = mesh.topological_dimension
    plan = SmallLinearSolvePlan(dimension)
    factorial = float(np.prod(np.arange(1, dimension + 1)))
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        edges = coordinates[cells[:, 1:]] - coordinates[cells[:, :1]]
        if dimension == mesh.ambient_dimension:
            volume = np.abs(np.asarray(determinant_small_linear(plan, edges))) / factorial
        else:
            gram = edges @ np.swapaxes(edges, 1, 2)
            volume = (
                np.sqrt(np.maximum(np.asarray(determinant_small_linear(plan, gram)), 0.0))
                / factorial
            )
        np.add.at(
            measures,
            cells.reshape((-1,)),
            np.repeat(volume / cells.shape[1], cells.shape[1]),
        )
    return measures


class _NativeTarget(NamedTuple):
    target: _dependency_phydrax_meshing__result.CellMeshingResult
    transition: _dependency_phydrax_meshing__lineage.CellMeshTransition
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage
    stencil: (
        _dependency_phydrax_meshing__lineage_records.VertexInterpolationStencil | None
    )
    transfer: FiniteElementTopologyTransfer | None


def _geometry_transition(
    prepared: PreparedMeshAdaptation,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    mesh: CellMesh,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> transition_types_owner.CellGeometryTransition | None:
    """Target coordinate map of a non-affine source through a nested edit.

    Simplex corner carriers keep publishing their affine successor. Mixed corner
    maps are restricted exactly too; degree-one prism/pyramid/hex maps need not be
    affine in reference coordinates. Nonnested curved routes require their owner.
    """

    source = prepared.source
    mixed = prepared.policy.route is MeshAdaptationRoute.NATIVE_MIXED
    if (
        not mixed
        and source.geometry.exact_source is None
        and source.geometry.periodic_source is None
        and source.geometry.restriction_source is None
        and source.geometry.storage is None
        and source.geometry.geometry_layout_id
        == CellGeometrySpec.affine(source.mesh).geometry_layout_id
        and np.array_equal(
            np.asarray(source.geometry.coordinates), np.asarray(source.mesh.coordinates)
        )
    ):
        return None
    match edit.operation:
        case "nested_refinement" | "nested_coarsening" | "nested_adaptation":
            pass
        case "local_reconnection" | "relocation":
            raise ValueError(
                f"MeshAdaptation.geometry_transition must have nested ancestry for a non-affine source coordinate map; got {prepared.policy.route.name!r}"
            )
        case operation:
            assert_never(operation)
    degree = (
        _dependency_phydrax_discretization__cell_geometry_transfer.nested_geometry_degree(
            source.mesh, source.geometry, budget=budget
        )
    )
    layout = (
        CellGeometrySpec.affine(mesh)
        if mixed or degree == 1
        else _dependency_phydrax_meshing__curving.straight_geometry(mesh, degree)
    )
    return _dependency_phydrax_discretization__cell_geometry_transfer.transition_nested_cell_geometry(
        source.mesh,
        source.geometry,
        mesh,
        layout,
        refinement=edit.refinement,
        coarsening=edit.coarsening,
        policy=prepared.policy.geometry_transition,
        budget=budget,
    )


@checked
def region_boundary_transition(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    mesh: CellMesh,
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage,
    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...],
    zones: tuple[_dependency_phydrax_meshing__organization.MeshZone, ...],
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...],
    /,
) -> tuple[_dependency_phydrax_meshing__organization.RegionBoundaryEvidence, ...]:
    """Rebind source-volume controls by declaration and lineage IDs."""
    if not source.region_boundary_evidence:
        return ()
    for evidence in source.region_boundary_evidence:
        evidence.require_current(source.mesh, source.labels, source.patches)
    target_zone_ids = {zone.zone_id for zone in zones}
    zone_ids: dict[str, str] = {}
    for zone in source.zones:
        expected_zone = _dependency_phydrax_meshing__organization.MeshZone(
            zone.name,
            zone.role,
            _dependency_phydrax_meshing__lineage.inherit_scope(
                zone.scope, lineage, mesh, zone.name
            ),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        if expected_zone.zone_id not in target_zone_ids:
            raise ValueError(
                f"RegionBoundaryEvidence.zone must have a lineage successor; got {zone.zone_id!r}"
            )
        zone_ids[zone.zone_id] = expected_zone.zone_id
    target_labels = {label.label_id: label for label in labels}
    inherited_labels: dict[str, _dependency_phydrax_meshing__organization.MeshLabel] = {}
    for evidence in source.region_boundary_evidence:
        original_label = evidence.boundary_label
        if original_label.label_id in inherited_labels:
            continue
        expected_label = _dependency_phydrax_meshing__organization.MeshLabel(
            original_label.name,
            _dependency_phydrax_meshing__lineage.inherit_scope(
                original_label.scope, lineage, mesh, original_label.name
            ),
        )
        if expected_label.label_id not in target_labels:
            raise ValueError(
                f"RegionBoundaryEvidence.label must have a lineage successor; got {original_label.label_id!r}"
            )
        inherited_labels[original_label.label_id] = target_labels[expected_label.label_id]
    source_patches = {patch.patch_id: patch for patch in source.patches}
    target_patch_ids = {patch.patch_id for patch in patches}
    patch_ids: dict[str, str] = {}
    for evidence in source.region_boundary_evidence:
        for identifier, _ in evidence.patch_sides:
            if identifier in patch_ids:
                continue
            original_patch = source_patches[identifier]
            expected_patch = _dependency_phydrax_meshing__organization.MeshPatch(
                original_patch.name,
                _dependency_phydrax_meshing__lineage.inherit_scope(
                    original_patch.scope, lineage, mesh, original_patch.name
                ),
                connected=original_patch.connected,
                adjacent_zone_ids=tuple(
                    zone_ids[value] for value in original_patch.adjacent_zone_ids
                ),
                source_adjacent_region_ids=original_patch.source_adjacent_region_ids,
            )
            if expected_patch.patch_id not in target_patch_ids:
                raise ValueError(
                    f"RegionBoundaryEvidence.patch must have a lineage successor; got {identifier!r}"
                )
            patch_ids[identifier] = expected_patch.patch_id
    renewed = tuple(
        _dependency_phydrax_meshing__organization.RegionBoundaryEvidence(
            mesh,
            evidence.source_scope,
            evidence.source_region_id,
            evidence.control_id,
            evidence.material_id,
            evidence.role,
            inherited_labels[evidence.boundary_label.label_id],
            tuple(
                (patch_ids[identifier], side) for identifier, side in evidence.patch_sides
            ),
        )
        for evidence in source.region_boundary_evidence
    )
    for evidence in renewed:
        evidence.require_current(mesh, labels, patches)
    return renewed


class _BinaryGeometryForest(NamedTuple):
    hierarchy: BisectionHierarchy
    owners: tuple[Array, ...]
    records: dict[int, int]
    predecessors: dict[int, int]
    parent_rows: np.ndarray
    tuples: np.ndarray
    tags: np.ndarray
    vertices: np.ndarray


def _binary_parent_geometry_element(
    element: CellGeometryElement,
    corners: np.ndarray,
    source_id: int,
    target_id: int,
    hierarchy: BisectionHierarchy,
    /,
    *,
    charts: _RationalP1Charts,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellGeometryElement:
    """Recover an ancestor through authenticated, possibly packed binary actions.

    One retained barycentric layer may encode several same-request splits.
    Bind its complete exact action to the corresponding forest path before
    removing it, or before factoring out an intermediate requested parent.
    """
    forest = charts.binary_forest(hierarchy, budget=budget)
    width = corners.size
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(width))
        for row in range(width)
    )
    while source_id != target_id:
        if not isinstance(
            element,
            _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement,
        ):
            raise ValueError(
                f"MeshAdaptation.geometry.parent_action must retain a binary parent action for coarsening; got {source_id!r}"
            )
        budget.reserve(width * width, 128 * width * width)
        declared = _owner_phydrax_discretization__cell_geometry.exact_barycentric_action(
            element
        )
        accumulated = identity
        ancestor, ancestor_corners = source_id, corners
        target_action: tuple[tuple[Fraction, ...], ...] | None = None
        visited: set[int] = set()
        while accumulated != declared:
            if ancestor in visited or ancestor not in forest.predecessors:
                raise ValueError(
                    f"MeshAdaptation.geometry.parent_action must preserve the authored binary coefficient action; got {source_id!r}"
                )
            visited.add(ancestor)
            parent = forest.predecessors[ancestor]
            record = forest.records[parent]
            parent_corners = forest.parent_rows[record]
            ordered = forest.tuples[record]
            tag, midpoint = int(forest.tags[record]), int(forest.vertices[record])
            positions = {
                int(vertex): index for index, vertex in enumerate(parent_corners)
            }
            step = [[Fraction(0)] * width for _ in range(width)]
            budget.reserve(width**3 + width * width, 128 * width * width)
            for row, vertex in enumerate(ancestor_corners):
                support = (ordered[0], ordered[tag]) if vertex == midpoint else (vertex,)
                for identifier in support:
                    column = positions.get(int(identifier))
                    if column is None:
                        raise ValueError(
                            f"MeshAdaptation.geometry.parent_support must use the declared binary coefficient parent support; got {ancestor!r}"
                        )
                    step[row][column] = Fraction(1, len(support))
            accumulated = tuple(
                tuple(
                    sum(
                        (
                            weight * step[column][axis]
                            for column, weight in enumerate(row)
                        ),
                        Fraction(0),
                    )
                    for axis in range(width)
                )
                for row in accumulated
            )
            ancestor, ancestor_corners = parent, parent_corners
            if ancestor == target_id:
                target_action = accumulated
        if target_action is not None:
            if ancestor == target_id:
                return element.source_element
            # The retained layer reaches farther than the requested parent.
            # target_action * parent_action = declared, exactly.
            budget.reserve(4 * width**3, 256 * width * width)
            parent_action = solve_exact(
                [list(row) for row in target_action],
                [list(row) for row in declared],
                singular_message="Binary parent coefficient action must be nonsingular.",
            )
            return charts.chart(
                element.source_element,
                tuple(tuple(row) for row in parent_action),
                budget=budget,
            )
        element, corners, source_id = element.source_element, ancestor_corners, ancestor
    return element


def _original_full_p1_cells(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    authority_mesh: CellMesh,
    authority_geometry: CellGeometrySpec,
    original_elements: tuple[CellGeometryElement, ...],
    original_routes: tuple[Array, ...],
    /,
    *,
    direct_authority: bool,
) -> dict[int, _FullP1OriginalCell]:
    origin = source.geometry.restriction_source
    original_cells: dict[int, _FullP1OriginalCell] = {}
    restriction = authority_geometry.restriction_source
    retained_owners = None if restriction is None else restriction.block_source_blocks
    for original_block, original_element, original_route in zip(
        authority_mesh.blocks,
        original_elements,
        original_routes,
        strict=True,
    ):
        root_name: str | None = (
            original_block.name if direct_authority or origin is None else None
        )
        if retained_owners is None:
            root = _owner_phydrax_discretization__cell_geometry.require_full_p1_source(
                original_element
            )
        else:
            # A retained owner's presentation may be a composed child chart
            # (for example a nondyadic zero insertion into a bisection target);
            # its scientific root is the P1 owner beneath the action chain.
            root_name = retained_owners[original_block.name]
            root = original_element
            while isinstance(
                root,
                (
                    _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement,
                    _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
                ),
            ):
                root = root.source_element
            root = _owner_phydrax_discretization__cell_geometry.require_full_p1_source(
                root
            )
        for original_row, identifier in enumerate(original_block.global_ids):
            root_identifier = (
                int(identifier)
                if direct_authority or origin is None
                else int(
                    np.asarray(origin.block_parent_cell_ids[original_block.name])[
                        original_row
                    ]
                )
            )
            root_corners = (
                np.asarray(authority_mesh.vertex_global_ids)[
                    np.asarray(original_block.vertices)[original_row]
                ]
                if direct_authority or origin is None
                else np.asarray(origin.block_parent_vertex_ids[original_block.name])[
                    original_row
                ]
            )
            original_cells[root_identifier] = (
                root_name,
                root,
                np.asarray(np.asarray(original_route)[original_row], dtype=np.int32),
                np.asarray(root_corners, dtype=np.int64),
                original_row,
            )
    return original_cells


def _full_p1_source_cells(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    authority_mesh: CellMesh,
    authority_geometry: CellGeometrySpec,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    /,
    *,
    direct_authority: bool,
    restored_authority: bool = False,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    dict[int, _FullP1SourceCell],
    dict[int, _FullP1OriginalCell],
    Array,
    CoordinateSourceBank,
]:
    """Authenticate coefficient columns and retained scientific cell owners.

    A root presentation belongs to its retained authored source block: the
    supplied block for a direct source, or the restriction record's
    block_source_blocks entry. None denotes a block owner that was not retained.
    """

    origin = source.geometry.restriction_source
    original_elements, original_routes, bank = authority_geometry.resolve(
        authority_mesh, budget=budget
    )
    original_bank = _owner_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
        authority_geometry, budget=budget
    )
    current_bank = (
        original_bank
        if source.geometry.coordinates is authority_geometry.coordinates
        and source.geometry.exact_source is authority_geometry.exact_source
        else _owner_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
            source.geometry, budget=budget
        )
    )
    original_cells = _original_full_p1_cells(
        source,
        authority_mesh,
        authority_geometry,
        original_elements,
        original_routes,
        direct_authority=direct_authority,
    )
    if (
        direct_authority
        and not restored_authority
        and origin is not None
        and (
            origin.source_geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                authority_geometry
            )
            or origin.source_topology_id != authority_mesh.topology_id
        )
    ):
        raise ValueError(
            f"MeshAdaptation.source_geometry.restriction_source must retain the declared scientific source owner for full P1 actions; got {source.geometry.geometry_layout_id!r}"
        )
    source_cells: dict[int, _FullP1SourceCell] = {}
    for block, element, route in zip(source.mesh.blocks, elements, routes, strict=True):
        if isinstance(
            element,
            _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
        ):
            root: CellGeometryElement = element
            while isinstance(
                root,
                _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
            ):
                root = root.source_element
            _owner_phydrax_discretization__cell_geometry.require_full_p1_source(root)
            if element.degree != 1 or element.cell_kind not in (
                "triangle",
                "tetrahedron",
            ):
                raise ValueError(
                    f"MeshAdaptation.geometry.chart must remain an affine simplex map for full P1 actions; got {(element.cell_kind, element.degree)!r}"
                )
        else:
            element = _owner_phydrax_discretization__cell_geometry.require_full_p1_source(
                element
            )
        for cell_index, (cell, vertices, coefficients) in enumerate(
            zip(block.global_ids, block.vertices, route, strict=True)
        ):
            ancestor = (
                int(cell)
                if origin is None
                else int(np.asarray(origin.block_parent_cell_ids[block.name])[cell_index])
            )
            corners = (
                np.asarray(source.mesh.vertex_global_ids)[vertices]
                if origin is None
                else np.asarray(origin.block_parent_vertex_ids[block.name])[cell_index]
            )
            if ancestor not in original_cells:
                raise ValueError(
                    f"MeshAdaptation.geometry.parent_cell must declare the scientific source cell; got {ancestor!r}"
                )
            root_name, original_root, original_route, original_corners, original_row = (
                original_cells[ancestor]
            )
            if not np.array_equal(corners, original_corners):
                raise ValueError(
                    f"MeshAdaptation.geometry.parent_corners must preserve the scientific corner columns; got {ancestor!r}"
                )
            if tuple(current_bank[int(index)] for index in coefficients) != tuple(
                original_bank[int(index)] for index in original_route
            ):
                raise ValueError(
                    f"MeshAdaptation.geometry.coefficients must preserve the scientific coefficient bank and route columns; got {ancestor!r}"
                )
            source_cells[int(cell)] = (
                element,
                np.asarray(source.mesh.vertex_global_ids)[vertices],
                original_route,
                ancestor,
                corners,
                root_name,
                original_row,
                original_root,
            )
    return source_cells, original_cells, bank, original_bank


@final
class _RationalP1Charts:
    """Compose each distinct full-P1 source element and exact action once per edit.

    A rational chart is a pure function of its source element and exact
    barycentric rows. Target cells repeating one pair share that element and
    its charged preparation instead of rebuilding and discarding equal copies.
    Entries retain their source element, so an object identity is never reused.
    """

    __slots__ = ("_charts", "_binary_forests")

    def __init__(self) -> None:
        self._charts: dict[
            tuple[int, tuple[tuple[Fraction, ...], ...]],
            tuple[FullP1Element, FullP1Element],
        ] = {}
        self._binary_forests: dict[int, _BinaryGeometryForest] = {}

    def binary_forest(
        self,
        hierarchy: BisectionHierarchy,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> _BinaryGeometryForest:
        key = id(hierarchy)
        owners = (
            hierarchy.record_parent_ids,
            hierarchy.record_child_ids,
            hierarchy.record_parent_rows,
            hierarchy.record_parent_vertices,
            hierarchy.record_parent_tags,
            hierarchy.record_vertex_ids,
        )
        known = self._binary_forests.get(key)
        # The index is reusable only for the same immutable array owners.
        # Replaced fields or mutable NumPy inputs require a fresh derivation.
        if (
            known is not None
            and known.hierarchy is hierarchy
            and all(isinstance(owner, Array) for owner in owners)
            and all(
                owner is prior for owner, prior in zip(owners, known.owners, strict=True)
            )
        ):
            return known
        size = hierarchy.record_parent_ids.size
        budget.reserve(3 * size, 512 * size)
        parents = np.asarray(hierarchy.record_parent_ids)
        children = np.asarray(hierarchy.record_child_ids)
        forest = _BinaryGeometryForest(
            hierarchy,
            owners,
            {int(parent): row for row, parent in enumerate(parents)},
            {
                int(child): int(parent)
                for parent, row in zip(parents, children, strict=True)
                for child in row
            },
            np.asarray(hierarchy.record_parent_rows),
            np.asarray(hierarchy.record_parent_vertices),
            np.asarray(hierarchy.record_parent_tags),
            np.asarray(hierarchy.record_vertex_ids),
        )
        budget.retain_owner(forest)
        self._binary_forests[key] = forest
        return forest

    def chart(
        self,
        source_element: FullP1Element,
        barycentric_weights: tuple[tuple[Fraction, ...], ...],
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> FullP1Element:
        key = (id(source_element), barycentric_weights)
        known = self._charts.get(key)
        if known is not None and known[0] is source_element:
            return known[1]
        chart = uniform_rational_p1_chart(
            source_element, barycentric_weights, budget=budget
        )
        # The edit retains one mapping slot and key tuple per distinct chart.
        budget.reserve(0, 128)
        self._charts[key] = (source_element, chart)
        return chart


@checked
def uniform_rational_p1_chart(
    source_element: FullP1Element,
    barycentric_weights: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> FullP1Element:
    """Encode one uniform-plus-binary action as an exact rational chart."""

    width = source_element.local_dof_count
    dimension = source_element.topological_dimension
    if (
        len(barycentric_weights) != width
        or any(len(row) != width for row in barycentric_weights)
        or width != dimension + 1
        or any(value < 0 for row in barycentric_weights for value in row)
        or any(sum(row, Fraction(0)) != 1 for row in barycentric_weights)
    ):
        raise ValueError(
            f"MeshAdaptation.rational_chart must contain simplex barycentric rows; got {barycentric_weights!r}"
        )
    if isinstance(
        source_element,
        (
            _owner_phydrax_discretization_fem__reference_types.FiniteElementSpec,
            _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement,
        ),
    ) and _full_p1_action_sources((source_element,)):
        # These are the actual reference-action rows, including column zero.
        # Select the existing cardinal-action representation only when its
        # binary64 execution bank represents every declared rational exactly.
        budget.reserve(2 * width * width)
        with budget.live_storage() as storage:
            storage.set_bound(1024 + 512 * width * width)
            execution = np.asarray(barycentric_weights, dtype=np.float64)
            if all(
                Fraction(float(execution[row, column]))
                == barycentric_weights[row][column]
                for row in range(width)
                for column in range(width)
            ):
                chart = _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement(
                    source_element, execution
                )
                budget.retain_owner(chart, temporary_storage_credit=storage)
                return chart
            del execution
    numerators = np.asarray(
        [
            [row[source].numerator for source in range(1, width)]
            for row in barycentric_weights
        ],
        dtype=object,
    )
    denominators = np.asarray(
        [
            [row[source].denominator for source in range(1, width)]
            for row in barycentric_weights
        ],
        dtype=object,
    )
    return _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement(
        source_element,
        _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
            source_element.cell_kind, 1
        ),
        numerators,
        denominators,
        budget=budget,
    )


@checked
def apply_full_p1_action(
    source_element: FullP1Element,
    action: ArrayLike,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    charts: _RationalP1Charts | None = None,
) -> FullP1Element:
    """Compose an affine simplex action in exact arithmetic with a supported P1 owner."""

    if isinstance(
        source_element,
        _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
    ):
        values = np.asarray(action)
        exact = tuple(
            tuple(
                value if isinstance(value, Fraction) else Fraction(float(value))
                for value in row
            )
            for row in values
        )
        if charts is not None:
            return charts.chart(source_element, exact, budget=budget)
        return uniform_rational_p1_chart(source_element, exact, budget=budget)
    return _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement(
        source_element, np.asarray(action)
    )


@final
class _UniformSourceActions(StrictModule):
    """Rational vertex actions and scientific roots authenticated from uniform rows."""

    vertex_actions: dict[tuple[int, int], tuple[Fraction, ...]]
    cell_roots: dict[int, int]


def _uniform_source_actions(
    uniform: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement, /
) -> _UniformSourceActions:
    arrays = uniform.host_arrays()
    width = uniform.dimension + 1
    vertex_actions: dict[tuple[int, int], tuple[Fraction, ...]] = {}
    cell_roots: dict[int, int] = {}

    for parent, children, vertices, actions in zip(
        np.asarray(arrays["parent_ids"], dtype=np.int64).tolist(),
        np.asarray(arrays["child_ids"], dtype=np.int64).tolist(),
        np.asarray(arrays["child_vertices"], dtype=np.int64).tolist(),
        np.asarray(arrays["barycentric_weights"], dtype=np.float64).tolist(),
        strict=True,
    ):
        root = int(parent)
        cell_roots[root] = root
        for child, child_vertices, child_actions in zip(
            children,
            vertices,
            actions,
            strict=True,
        ):
            cell_roots[int(child)] = root
            for vertex, raw in zip(child_vertices, child_actions, strict=True):
                raw_ = np.asarray(raw, dtype=np.float64)
                support = np.flatnonzero(raw_)
                if support.size == 0:
                    raise ValueError(
                        f"MeshAdaptation.geometry.vertex_action must provide barycentric support for uniform source vertices; got {int(vertex)!r}"
                    )
                expected = 1.0 / support.size
                if not np.all(raw_[support] == expected) or np.any(
                    np.delete(raw_, support) != 0.0
                ):
                    raise ValueError(
                        f"MeshAdaptation.geometry.vertex_action must preserve the declared subset action of uniform source vertices; got {int(vertex)!r}"
                    )
                exact = tuple(
                    Fraction(1, int(support.size)) if source in support else Fraction(0)
                    for source in range(width)
                )
                key = root, int(vertex)
                previous = vertex_actions.setdefault(key, exact)
                if previous != exact:
                    raise ValueError(
                        f"MeshAdaptation.geometry.vertex_action must agree across uniform siblings; got {int(vertex)!r}"
                    )
    return _UniformSourceActions(vertex_actions, cell_roots)


def _uniform_exact_target_actions(
    uniform: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    hierarchy: BisectionHierarchy,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    parents: dict[int, int],
    /,
) -> _UniformRationalActions:
    """Compose subset rows and binary midpoint records once per source action."""

    source_actions = _uniform_source_actions(uniform)
    vertex_actions, cell_roots = source_actions.vertex_actions, source_actions.cell_roots

    records = sorted(
        zip(
            np.asarray(hierarchy.record_parent_ids, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_parent_vertices, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_parent_tags, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_child_ids, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_vertex_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    for parent, ordered, tag, children, midpoint in records:
        root = cell_roots.get(parent)
        if root is None:
            raise ValueError(
                f"MeshAdaptation.geometry.uniform_root must declare a scientific root for every binary refinement record; got {parent!r}"
            )
        first = vertex_actions.get((root, int(ordered[0])))
        second = vertex_actions.get((root, int(ordered[tag])))
        if first is None or second is None:
            raise ValueError(
                f"MeshAdaptation.geometry.midpoint_action must retain consistent endpoint actions across binary refinement records; got {midpoint!r}"
            )
        exact = tuple(
            (left + right) / 2 for left, right in zip(first, second, strict=True)
        )
        key = root, midpoint
        previous = vertex_actions.setdefault(key, exact)
        if previous != exact:
            raise ValueError(
                f"MeshAdaptation.geometry.midpoint_action must retain consistent endpoint actions across binary refinement records; got {midpoint!r}"
            )
        for child in children:
            known = cell_roots.setdefault(int(child), root)
            if known != root:
                raise ValueError(
                    f"MeshAdaptation.geometry.uniform_root must agree across binary refinement descendants; got {int(child)!r}"
                )

    result: _UniformRationalActions = {}
    for block in edit.blocks:
        if not isinstance(block, _owner_phydrax_meshing__topology_edit.TopologyEditBlock):
            raise ValueError(
                f"MeshAdaptation.edit.blocks must contain fixed-family simplex blocks for uniform rational actions; got {type(block).__name__!r}"
            )
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            target = int(identifier)
            predecessor = parents.get(target, target)
            root = cell_roots.get(target, cell_roots.get(predecessor, predecessor))
            known = cell_roots.get(target, root)
            if known != root:
                raise ValueError(
                    f"MeshAdaptation.geometry.parent_cell must preserve the uniform scientific parent; got {int(identifier)!r}"
                )
            vertices = edit.vertex_global_ids[np.asarray(row, dtype=np.int64)]
            action = tuple(vertex_actions[(root, int(vertex))] for vertex in vertices)
            result[target] = root, action
    return result


def _periodic_full_p1_parent_elements(
    hierarchy: MeshAdaptationHierarchy,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    /,
    *,
    carrier: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> dict[int, CellGeometryElement]:
    """Retain immediate-parent charts independently of scientific root columns."""
    if (
        not isinstance(hierarchy, _owner_phydrax_meshing__periodic.PeriodicRefinement)
        or edit.coarsening is None
    ):
        return {}
    geometry = hierarchy.source_geometry
    if carrier:
        retained = geometry.restriction_source
        geometry = (
            CellGeometrySpec.affine(hierarchy.source, budget=budget)
            if retained is None or retained.carrier_geometry is None
            else retained.carrier_geometry
        )
    elements, _, _ = geometry.resolve(hierarchy.source, budget=budget)
    selected = set(int(identifier) for identifier in edit.coarsening.coarse_cell_ids)
    return {
        int(identifier): element
        for block, element in zip(hierarchy.source.blocks, elements, strict=True)
        for identifier in block.global_ids
        if int(identifier) in selected
    }


def _full_p1_target_element(
    source_cell: _FullP1SourceCell,
    original_cells: dict[int, _FullP1OriginalCell],
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    identifier: int,
    row: NDArray[np.int32],
    parent: dict[int, int],
    direct_authority: bool,
    uniform: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement | None,
    hierarchy: BisectionHierarchy | None,
    uniform_actions: _UniformRationalActions | None,
    periodic_parents: dict[int, CellGeometryElement],
    /,
    *,
    charts: _RationalP1Charts,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> CellGeometryElement:
    """Recover a retained parent action or author the incoming coefficient action."""

    current_element, current_corners = source_cell[:2]
    if (
        edit.coarsening is not None
        and parent.get(identifier, identifier) != identifier
        and identifier in np.asarray(edit.coarsening.coarse_cell_ids)
    ):
        if identifier in periodic_parents:
            return periodic_parents[identifier]
        if direct_authority and identifier in original_cells:
            return original_cells[identifier][1]
        if uniform_actions is not None and identifier in uniform_actions:
            root, action = uniform_actions[identifier]
            if source_cell[3] != root or root not in original_cells:
                raise ValueError(
                    f"MeshAdaptation.geometry.rational_chart must preserve the scientific parent cell; got {identifier!r}"
                )
            return charts.chart(original_cells[root][1], action, budget=budget)
        if uniform is not None and hierarchy is not None:
            root_index, child_index, steps = bisection_kernel_owner.uniform_binary_chart(
                identifier, hierarchy, scope=scope, budget=budget
            )
            root_identifier = int(np.asarray(uniform.parent_ids)[root_index])
            element = original_cells[root_identifier][1]
            if child_index is not None:
                element = apply_full_p1_action(
                    element,
                    uniform.barycentric_weights[root_index, child_index],
                    budget=budget,
                    charts=charts,
                )
            for step in steps:
                element = apply_full_p1_action(
                    element, step, budget=budget, charts=charts
                )
            return element
        if hierarchy is not None:
            return _binary_parent_geometry_element(
                current_element,
                current_corners,
                parent[identifier],
                identifier,
                hierarchy,
                charts=charts,
                budget=budget,
            )
        raise ValueError(
            f"MeshAdaptation.geometry.parent_chart must retain the scientific parent chart of a coarsened coefficient action; got {identifier!r}"
        )
    if uniform_actions is not None and identifier in uniform_actions:
        root, action = uniform_actions[identifier]
        if source_cell[3] != root or root not in original_cells:
            raise ValueError(
                f"MeshAdaptation.geometry.rational_chart must preserve the scientific parent cell; got {identifier!r}"
            )
        return charts.chart(original_cells[root][1], action, budget=budget)
    homogeneous = _exact_full_p1_stencil_bank(
        edit, row, current_corners, identifier, budget=budget
    )
    if homogeneous is not None:
        # Exact rows h / sum(h) own the chart; rounded stencil weights are views.
        if np.array_equal(
            homogeneous != 0.0, np.eye(current_element.local_dof_count, dtype=np.bool_)
        ):
            return current_element
        if isinstance(
            current_element,
            _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
        ):
            return charts.chart(
                current_element,
                _owner_phydrax_discretization__cell_geometry.exact_homogeneous_action(
                    homogeneous, budget=budget
                ),
                budget=budget,
            )
        return (
            _owner_phydrax_discretization__cell_geometry.homogeneous_barycentric_element(
                current_element, homogeneous, budget=budget
            )
        )
    action = np.zeros(
        (current_element.local_dof_count, current_element.local_dof_count),
        dtype=np.float64,
    )
    for corner, vertex in enumerate(row):
        for support, weight, valid in zip(
            edit.stencil_sources[vertex],
            edit.stencil_weights[vertex],
            edit.stencil_valid[vertex],
            strict=True,
        ):
            if valid:
                action[
                    corner, _full_p1_support_corner(current_corners, support, identifier)
                ] += weight
    return (
        current_element
        if np.array_equal(
            action, np.eye(current_element.local_dof_count, dtype=np.float64)
        )
        else apply_full_p1_action(current_element, action, budget=budget, charts=charts)
    )


def _full_p1_support_corner(
    corners: NDArray[np.int64], support: int, identifier: int, /
) -> int:
    matches = np.flatnonzero(corners == support)
    if matches.size != 1:
        raise ValueError(
            f"MeshAdaptation.geometry.parent_support must use the declared full P1 parent corner support; got {identifier!r}"
        )
    return int(matches[0])


def _exact_full_p1_stencil_bank(
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    row: NDArray[np.int32],
    corners: NDArray[np.int64],
    identifier: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray | None:
    """Aggregate exact homogeneous stencils onto parent corners, if retained.

    ``H[corner, match]`` sums the homogeneous weights naming one parent corner;
    the deterministic aggregation meters its visits and refuses inexact binary64
    sums, so ``H / sum(H)`` is the unrounded rational action.
    """
    bank = edit.stencil_homogeneous
    if bank is None:
        return None
    valid = np.asarray(edit.stencil_valid, dtype=np.bool_)[row]
    sources = np.where(valid, np.asarray(edit.stencil_sources, dtype=np.int64)[row], -1)
    for support in sources[sources >= 0].tolist():
        _full_p1_support_corner(corners, support, identifier)
    return _owner_phydrax_meshing__topology_edit.exact_nested_reference_vertices(
        sources[None],
        np.asarray(bank, dtype=np.float64)[row][None],
        np.asarray(corners, dtype=np.int64)[None],
        budget=budget,
    )[1][0]


def _whole_source_inverse_edit(
    original: CellMesh,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    /,
) -> _dependency_phydrax_meshing__topology_edit.CellTopologyEdit | None:
    """Restore full source scientific parent incidence."""

    original_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in original.blocks)
    )
    target_ids = np.concatenate(tuple(block.cell_ids for block in edit.blocks))
    if not np.array_equal(np.sort(target_ids), np.sort(original_ids)):
        return None
    actual_cells = {}
    for block in edit.blocks:
        if not isinstance(block, _owner_phydrax_meshing__topology_edit.TopologyEditBlock):
            raise ValueError(
                f"MeshAdaptation.edit.blocks must contain source simplex cells for a whole-source inverse; got {type(block).__name__!r}"
            )
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            actual_cells[int(identifier)] = (
                block.cell_kind,
                tuple(edit.vertex_global_ids[row].tolist()),
            )
    original_vertices = np.asarray(original.vertex_global_ids)
    expected_cells = {
        int(identifier): (block.cell_kind, tuple(original_vertices[row].tolist()))
        for block in original.blocks
        for identifier, row in zip(
            block.global_ids, np.asarray(block.vertices), strict=True
        )
    }
    if actual_cells != expected_cells:
        raise ValueError(
            f"MeshAdaptation.geometry.parent_incidence must preserve scientific parent incidence in a source inverse; got {original.mesh_id!r}"
        )
    positions = {
        int(identifier): index for index, identifier in enumerate(edit.vertex_global_ids)
    }
    blocks = tuple(
        _owner_phydrax_meshing__topology_edit.TopologyEditBlock(
            block.name,
            block.cell_kind,
            None,
            np.asarray(
                [
                    [positions[int(identifier)] for identifier in row]
                    for row in original_vertices[block.vertices]
                ],
                dtype=np.int32,
            ),
            np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in original.blocks
    )
    return edit._replace(blocks=blocks)


@final
class _FullP1EditSource(StrictModule):
    """Authenticated source coefficient owners and retained refinement provenance."""

    source_cells: dict[int, _FullP1SourceCell]
    original_cells: dict[int, _FullP1OriginalCell]
    bank: Array
    original_bank: (
        _owner_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank
    )
    authority_geometry: CellGeometrySpec
    direct_authority: bool
    uniform: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement | None
    hierarchy: MeshAdaptationHierarchy


@final
class _FullP1EditTarget(StrictModule):
    """Coefficient grouping and the exact images of its actual coordinate owner."""

    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    geometry: CellGeometrySpec | None
    image_geometry: CellGeometrySpec | None = None
    images: (
        tuple[
            tuple[_owner_phydrax_discretization__coordinate_enclosure.ExactScalar, ...]
            | None,
            ...,
        ]
        | None
    ) = None


def _full_p1_target_coordinates(
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    source_mesh: CellMesh,
    groups: dict[
        str,
        tuple[
            _owner_phydrax_meshing__topology_edit.TopologyEditBlock,
            list[np.ndarray],
            list[np.int64],
        ],
    ],
    geometry_elements: dict[str, CellGeometryElement],
    geometry_routes: dict[str, list[np.ndarray]],
    original_bank: _owner_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    np.ndarray,
    tuple[
        tuple[_owner_phydrax_discretization__coordinate_enclosure.ExactScalar, ...]
        | None,
        ...,
    ],
]:
    coordinates = np.array(edit.coordinates, dtype=np.float64, copy=True)
    source_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    retained = np.isin(edit.vertex_global_ids, source_ids)
    order = np.argsort(source_ids, kind="stable")
    positions = order[
        np.searchsorted(source_ids[order], edit.vertex_global_ids[retained])
    ]
    coordinates[retained] = np.asarray(source_mesh.coordinates)[positions]
    # Each distinct coefficient operation is evaluated once for all cells that
    # share it; every vertex reached through several operations must agree.
    names = [name for name, (_, rows, _) in groups.items() if rows]
    images = _owner_phydrax_discretization__coordinate_enclosure.coordinate_vertex_images(
        tuple(
            (
                geometry_elements[name],
                np.asarray(geometry_routes[name], dtype=np.int64),
                np.asarray(groups[name][1], dtype=np.int64),
            )
            for name in names
        ),
        original_bank,
        coordinates.shape[0],
        mismatch="MeshAdaptation.geometry.corner_images must agree on each shared vertex across incident full P1 actions; got unequal rational images",
        budget=budget,
    )
    if images is None:
        raise ValueError(
            f"MeshAdaptation.geometry.corner_images must provide physical corner images for full P1 coefficient actions; got {names!r}"
        )
    created = [
        (vertex, image)
        for vertex, image in enumerate(images)
        if image is not None and not retained[vertex]
    ]
    if created:
        coordinates[[vertex for vertex, _ in created]] = np.asarray(
            [
                _owner_phydrax_discretization__coordinate_enclosure.rounded_point(image)
                for _, image in created
            ],
            dtype=np.float64,
        )
    return coordinates, images


@final
class _FullP1Groups(StrictModule):
    """Target rows grouped by coefficient action with aligned source provenance."""

    groups: dict[
        str,
        tuple[
            _owner_phydrax_meshing__topology_edit.TopologyEditBlock,
            list[np.ndarray],
            list[np.int64],
        ],
    ]
    geometry_elements: dict[str, CellGeometryElement]
    geometry_routes: dict[str, list[np.ndarray]]
    source_parent_ids: dict[str, list[int]]
    source_parent_vertices: dict[str, list[np.ndarray]]
    owner_blocks: dict[str, str | None]


def _group_full_p1_actions(
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    phase: _FullP1EditSource,
    parent: dict[int, int],
    uniform_actions: _UniformRationalActions | None,
    /,
    *,
    charts: _RationalP1Charts,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _FullP1Groups:
    source_cells, original_cells = phase.source_cells, phase.original_cells
    direct_authority, uniform, hierarchy = (
        phase.direct_authority,
        phase.uniform,
        phase.hierarchy,
    )
    periodic_parents = _periodic_full_p1_parent_elements(
        hierarchy, edit, carrier=False, budget=budget
    )
    groups = {}
    geometry_elements = {}
    geometry_routes = {}
    source_parent_ids = {}
    source_parent_vertices = {}
    owner_blocks: dict[str, str | None] = {}
    for block in edit.blocks:
        if not isinstance(block, _owner_phydrax_meshing__topology_edit.TopologyEditBlock):
            raise ValueError(
                f"MeshAdaptation.edit.blocks must contain fixed-family simplex edits for full P1 actions; got {type(block).__name__!r}"
            )
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            source_cell = source_cells[parent.get(int(identifier), int(identifier))]
            _, _, coefficients, ancestor, corners, root_name, _, original_root = (
                source_cell
            )
            element = _full_p1_target_element(
                source_cell,
                original_cells,
                edit,
                int(identifier),
                np.asarray(row, dtype=np.int32),
                parent,
                direct_authority,
                uniform,
                hierarchy if isinstance(hierarchy, BisectionHierarchy) else None,
                uniform_actions,
                periodic_parents,
                charts=charts,
                budget=budget,
                scope=scope,
            )
            presentation_root = (
                f"source-cell/{ancestor}" if root_name is None else root_name
            )
            name = (
                presentation_root
                if element.element_id == original_root.element_id
                else f"{presentation_root}/coefficient-action/{element.element_id}"
            )
            owner_blocks[name] = root_name
            groups.setdefault(name, (block, [], []))[1].append(row)
            groups[name][2].append(identifier)
            geometry_elements[name] = element
            geometry_routes.setdefault(name, []).append(coefficients)
            source_parent_ids.setdefault(name, []).append(ancestor)
            source_parent_vertices.setdefault(name, []).append(corners)
    for name, (original, rows, ids) in groups.items():
        order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
        groups[name] = (
            original,
            [rows[index] for index in order],
            [ids[index] for index in order],
        )
        geometry_routes[name] = [geometry_routes[name][index] for index in order]
        source_parent_ids[name] = [source_parent_ids[name][index] for index in order]
        source_parent_vertices[name] = [
            source_parent_vertices[name][index] for index in order
        ]
    return _FullP1Groups(
        groups,
        geometry_elements,
        geometry_routes,
        source_parent_ids,
        source_parent_vertices,
        owner_blocks,
    )


@final
class _FullP1CarrierSource(StrictModule):
    geometry: CellGeometrySpec
    cells: dict[int, _FullP1SourceCell]
    roots: dict[int, _FullP1OriginalCell]
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)


def _carrier_p1_element(element: CellGeometryElement, /) -> FullP1Element:
    match element:
        case (
            _owner_phydrax_discretization_fem__reference_types.FiniteElementSpec()
            | _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement()
            | _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement() as result
        ):
            return result
        case _:
            raise ValueError(
                "MeshAdaptation.carrier_geometry.elements must describe affine P1 cells; got another source family"
            )


def _prepare_full_p1_carrier_source(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _FullP1CarrierSource:
    retained = source.geometry.restriction_source
    geometry = (
        CellGeometrySpec.affine(source.mesh, budget=budget)
        if retained is None or retained.carrier_geometry is None
        else retained.carrier_geometry
    )
    elements, routes, _ = geometry.resolve(source.mesh, budget=budget)
    origin = geometry.restriction_source
    source_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    cells: dict[int, _FullP1SourceCell] = {}
    roots: dict[int, _FullP1OriginalCell] = {}
    for block, raw_element, route in zip(
        source.mesh.blocks, elements, routes, strict=True
    ):
        element = _carrier_p1_element(raw_element)
        root = element
        while isinstance(
            root,
            (
                _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement,
                _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
            ),
        ):
            root = _carrier_p1_element(root.source_element)
        for row, identifier in enumerate(np.asarray(block.global_ids)):
            ancestor = (
                int(identifier)
                if origin is None
                else int(np.asarray(origin.block_parent_cell_ids[block.name])[row])
            )
            corners = (
                source_ids[np.asarray(block.vertices)[row]]
                if origin is None
                else np.asarray(origin.block_parent_vertex_ids[block.name])[row]
            )
            owner = (
                block.name
                if origin is None
                else (
                    None
                    if origin.block_source_blocks is None
                    else origin.block_source_blocks[block.name]
                )
            )
            coefficients = np.asarray(np.asarray(route)[row], dtype=np.int32)
            corners = np.asarray(corners, dtype=np.int64)
            roots[ancestor] = (owner, root, coefficients, corners, row)
            cells[int(identifier)] = (
                element,
                source_ids[np.asarray(block.vertices)[row]],
                coefficients,
                ancestor,
                corners,
                owner,
                row,
                root,
            )
    return _FullP1CarrierSource(
        geometry,
        cells,
        roots,
        _owner_phydrax_discretization__cell_geometry_validity.cell_geometry_id(geometry)
        if origin is None
        else origin.source_geometry_id,
        source.mesh.topology_id if origin is None else origin.source_topology_id,
    )


def _full_p1_carrier_geometry(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    phase: _FullP1EditSource,
    grouped: _FullP1Groups,
    parent: dict[int, int],
    uniform_actions: _UniformRationalActions | None,
    /,
    *,
    charts: _RationalP1Charts,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> CellGeometrySpec:
    carrier = _prepare_full_p1_carrier_source(source, budget=budget)
    periodic_parents = _periodic_full_p1_parent_elements(
        phase.hierarchy, edit, carrier=True, budget=budget
    )
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, list[np.ndarray]] = {}
    ancestors: dict[str, list[int]] = {}
    corners: dict[str, list[np.ndarray]] = {}
    owners: dict[str, str] = {}
    for name, (_, rows, identifiers) in grouped.groups.items():
        routes[name], ancestors[name], corners[name] = [], [], []
        for row, identifier in zip(rows, identifiers, strict=True):
            current = carrier.cells[parent.get(int(identifier), int(identifier))]
            element = _full_p1_target_element(
                current,
                carrier.roots,
                edit,
                int(identifier),
                row,
                parent,
                phase.direct_authority,
                phase.uniform,
                phase.hierarchy
                if isinstance(phase.hierarchy, BisectionHierarchy)
                else None,
                uniform_actions,
                periodic_parents,
                charts=charts,
                budget=budget,
                scope=scope,
            )
            prior = elements.setdefault(name, element)
            if not tree_content_equal(prior, element):
                raise ValueError(
                    "MeshAdaptation.carrier_geometry.elements must agree within each source-action block; got inconsistent affine actions"
                )
            routes[name].append(current[2])
            ancestors[name].append(current[3])
            corners[name].append(current[4])
            if current[5] is not None:
                owners[name] = current[5]
    return CellGeometrySpec(
        elements,
        {name: np.asarray(values, dtype=np.int32) for name, values in routes.items()},
        carrier.geometry.coordinates,
        restriction_source=_owner_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
            carrier.geometry_id,
            carrier.topology_id,
            {
                name: np.asarray(values, dtype=np.int64)
                for name, values in ancestors.items()
            },
            {
                name: np.asarray(values, dtype=np.int64)
                for name, values in corners.items()
            },
            block_source_blocks=owners if len(owners) == len(elements) else None,
        ),
    )


def _assemble_full_p1_target(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    phase: _FullP1EditSource,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _FullP1EditTarget:
    bank = phase.bank
    authority_geometry, uniform, hierarchy = (
        phase.authority_geometry,
        phase.uniform,
        phase.hierarchy,
    )
    origin = source.geometry.restriction_source
    parent = (
        {}
        if edit.refinement is None
        else dict(
            zip(
                np.asarray(edit.refinement.fine_cell_ids).tolist(),
                np.asarray(edit.refinement.coarse_cell_ids).tolist(),
                strict=True,
            )
        )
    )
    if edit.coarsening is not None:
        for child, target in zip(
            edit.coarsening.fine_cell_ids, edit.coarsening.coarse_cell_ids, strict=True
        ):
            parent.setdefault(int(target), int(child))
    uniform_actions = (
        _uniform_exact_target_actions(uniform, hierarchy, edit, parent)
        if uniform is not None and isinstance(hierarchy, BisectionHierarchy)
        else None
    )
    # Scientific and carrier passes share one memo of distinct exact charts.
    charts = _RationalP1Charts()
    grouped = _group_full_p1_actions(
        edit,
        phase,
        parent,
        uniform_actions,
        charts=charts,
        budget=budget,
        scope=scope,
    )
    groups, geometry_elements, geometry_routes = (
        grouped.groups,
        grouped.geometry_elements,
        grouped.geometry_routes,
    )
    source_parent_ids, source_parent_vertices, owner_blocks = (
        grouped.source_parent_ids,
        grouped.source_parent_vertices,
        grouped.owner_blocks,
    )
    blocks = tuple(
        _owner_phydrax_meshing__topology_edit.TopologyEditBlock(
            name,
            original.cell_kind,
            None,
            np.asarray(rows, dtype=np.int32),
            np.asarray(ids, dtype=np.int64),
        )
        for name, (original, rows, ids) in sorted(groups.items())
    )
    target_routes = {
        name: np.asarray(values, dtype=np.int32)
        for name, values in geometry_routes.items()
    }
    exact_source = authority_geometry.exact_source
    if isinstance(
        exact_source,
        (
            exact_rational_geometry_owner.ExactRationalCellGeometrySource,
            ExactAlgebraicCellGeometrySource,
        ),
    ):
        used = np.arange(bank.shape[0], dtype=np.int64)
        exact_source = exact_source.reindexed(
            used, geometry_elements, target_routes, budget=budget
        )
    declared_source = isinstance(
        exact_source,
        (
            exact_rational_geometry_owner.ExactRationalCellGeometrySource,
            ExactAlgebraicCellGeometrySource,
        ),
    )
    carrier_geometry = (
        None
        if declared_source
        else _full_p1_carrier_geometry(
            source,
            edit,
            phase,
            grouped,
            parent,
            uniform_actions,
            charts=charts,
            budget=budget,
            scope=scope,
        )
    )
    geometry = CellGeometrySpec(
        geometry_elements,
        target_routes,
        bank,
        restriction_source=_owner_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
            _owner_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source.geometry
            )
            if origin is None
            else origin.source_geometry_id,
            source.mesh.topology_id if origin is None else origin.source_topology_id,
            {
                name: np.asarray(values, dtype=np.int64)
                for name, values in source_parent_ids.items()
            },
            {
                name: np.asarray(values, dtype=np.int64)
                for name, values in source_parent_vertices.items()
            },
            block_source_blocks=None
            if any(owner is None for owner in owner_blocks.values())
            else {name: owner for name, owner in owner_blocks.items() if owner},
            carrier_publication="source_image"
            if declared_source
            else "published_affine_roots",
            carrier_geometry=carrier_geometry,
        ),
        exact_source=exact_source,
        periodic_source=authority_geometry.periodic_source,
        budget=budget,
    )
    coordinate_geometry = geometry if carrier_geometry is None else carrier_geometry
    coordinate_bank = (
        phase.original_bank
        if carrier_geometry is None
        else _owner_phydrax_discretization__cell_geometry.prepared_published_affine_root_bank(
            carrier_geometry, budget=budget
        )
    )
    coordinates, images = _full_p1_target_coordinates(
        edit,
        source.mesh,
        groups,
        dict(
            zip(
                coordinate_geometry.block_names,
                coordinate_geometry.elements,
                strict=True,
            )
        ),
        {
            name: list(np.asarray(route))
            for name, route in zip(
                coordinate_geometry.block_names,
                coordinate_geometry.geometry_dofs,
                strict=True,
            )
        },
        coordinate_bank,
        budget=budget,
    )
    return _FullP1EditTarget(
        edit._replace(blocks=blocks, coordinates=coordinates),
        geometry,
        coordinate_geometry,
        images,
    )


def _coefficient_action_root(element: CellGeometryElement, /) -> CellGeometryElement:
    """Owner beneath a chain of rational barycentric or composed P1 actions."""

    root = element
    while isinstance(
        root,
        (
            _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement,
            _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
        ),
    ):
        root = root.source_element
    return root


def _full_p1_action_sources(elements: tuple[CellGeometryElement, ...], /) -> bool:
    """The exact owner premise shared by protection and the actual producer."""
    return all(
        isinstance(
            root, _owner_phydrax_discretization_fem__reference_types.FiniteElementSpec
        )
        and root.cell_kind in ("triangle", "tetrahedron")
        and _owner_phydrax_discretization__coordinate_enclosure.coordinate_source_signature(
            root
        )
        == _owner_phydrax_discretization__coordinate_enclosure.coordinate_source_signature(
            _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                root.cell_kind, 1
            )
        )
        for root in map(_coefficient_action_root, elements)
    )


def _composes_nested_child_maps(
    prepared: PreparedMeshAdaptation,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    /,
) -> bool:
    """Whether `_geometry_transition` composes one coordinate map per child.

    Mirrors `transition_nested_cell_geometry`: restricted or composed source
    elements, and rational non-PLC sources, take the mixed nested transition,
    whose target blocks each own one child map. Nodal simplex maps and PLC
    sources keep their own owners, and NATIVE_MIXED authors its own blocks.
    """

    geometry = prepared.source.geometry
    return (
        prepared.policy.route is not MeshAdaptationRoute.NATIVE_MIXED
        and edit.refinement is not None
        and not isinstance(
            geometry.exact_source,
            (
                _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource,
                _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource,
            ),
        )
        and (
            geometry.exact_source is not None
            or any(
                not isinstance(
                    element,
                    _owner_phydrax_discretization_fem__reference_types.FiniteElementSpec,
                )
                for element in geometry.elements
            )
        )
    )


def _nested_action_blocks(
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit, /
) -> _dependency_phydrax_meshing__topology_edit.CellTopologyEdit:
    """Give every distinct nested child reference action its own target block.

    The nested geometry transition owns one coordinate element per target block.
    Children of restricted or composed parents differ by their rational reference
    corners and homogeneous rows, so rows are grouped by that action key: identity
    children keep the source block name, every other action gets the
    deterministic ``{block}/reference-action/{key}`` block. Cell IDs, rows,
    lineage relations and certificates are unchanged.
    """

    witnesses = edit.refinement
    if witnesses is None:
        return edit
    reference = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    homogeneous = (
        None
        if witnesses.fine_reference_homogeneous is None
        else np.asarray(witnesses.fine_reference_homogeneous, dtype=np.float64)
    )
    rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(witnesses.fine_cell_ids, dtype=np.int64).tolist()
        )
    }
    blocks: list[
        _dependency_phydrax_meshing__topology_edit.TopologyEditBlock
        | _dependency_phydrax_meshing__topology_edit.PolyhedralTopologyEditBlock
    ] = []
    for block in edit.blocks:
        if not isinstance(
            block, _owner_phydrax_meshing__topology_edit.TopologyEditBlock
        ) or block.cell_kind not in ("triangle", "tetrahedron"):
            blocks.append(block)
            continue
        identity = np.asarray(
            _owner_phydrax_discretization__reference_cell.reference_cell_topology(
                block.cell_kind
            ).vertices,
            dtype=np.float64,
        )
        arity = identity.shape[0]
        groups: dict[str, list[int]] = {}
        for index, identifier in enumerate(np.asarray(block.cell_ids).tolist()):
            row = rows.get(int(identifier))
            corners = None if row is None else reference[row, :arity]
            name = block.name
            if corners is not None and not np.array_equal(corners, identity):
                witness = canonical_fingerprint(
                    {
                        "reference": array_tree_fingerprint(corners),
                        "homogeneous": None
                        if homogeneous is None
                        else array_tree_fingerprint(homogeneous[row, :arity]),
                    }
                )
                name = f"{block.name}/reference-action/{witness}"
            groups.setdefault(name, []).append(index)
        for name in sorted(groups):
            selected = np.asarray(groups[name], dtype=np.int64)
            blocks.append(
                _owner_phydrax_meshing__topology_edit.TopologyEditBlock(
                    name,
                    block.cell_kind,
                    block.source_kind if name == block.name else None,
                    np.asarray(block.cells)[selected],
                    np.asarray(block.cell_ids)[selected],
                )
            )
    return edit._replace(blocks=tuple(blocks))


def _full_p1_edit_geometry(
    prepared: PreparedMeshAdaptation,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    /,
    *,
    bisection_hierarchy: BisectionHierarchy | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _FullP1EditTarget:
    """Group coefficient actions before topology and lineage binding."""

    # Level-set cuts carry rational homogeneous stencils; their children must
    # publish exact source actions (rational), not a chart of rounded vertices.
    if prepared.policy.route not in (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.DEVICE_BISECTION,
        MeshAdaptationRoute.NATIVE_LEVEL_SET,
    ):
        return _FullP1EditTarget(edit, None)
    source = prepared.source
    request = prepared.request
    request_hierarchy = (
        request.hierarchy if isinstance(request, MarkedMeshAdaptation) else None
    )
    hierarchy = (
        bisection_hierarchy
        if bisection_hierarchy is not None
        and (
            bisection_hierarchy.uniform_refinement is not None
            or request_hierarchy is None
        )
        else request_hierarchy
    )
    uniform = (
        hierarchy.uniform_refinement
        if isinstance(hierarchy, BisectionHierarchy)
        else None
    )
    periodic = (
        hierarchy
        if isinstance(hierarchy, _owner_phydrax_meshing__periodic.PeriodicRefinement)
        else None
    )
    elements, routes, bank = source.geometry.resolve(source.mesh, budget=budget)
    origin = source.geometry.restriction_source

    # Coefficient actions qualify only over a full P1 owner; composed charts of
    # restricted or curved parents keep their responsible nested transition.
    if not _full_p1_action_sources(elements):
        return _FullP1EditTarget(edit, None)
    # Restriction columns bind the root source; coarsening charts bind the
    # immediate parent retained separately by the incoming periodic history.
    authority_periodic = periodic
    while (
        authority_periodic is not None
        and authority_periodic.previous is not None
        and origin is not None
        and (
            origin.source_geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                authority_periodic.source_geometry
            )
            or origin.source_topology_id != authority_periodic.source.topology_id
        )
    ):
        authority_periodic = authority_periodic.previous
    authority_mesh = (
        uniform.source.mesh
        if uniform is not None
        else authority_periodic.source
        if authority_periodic is not None
        else source.mesh
    )
    authority_geometry = (
        uniform.source.geometry
        if uniform is not None
        else authority_periodic.source_geometry
        if authority_periodic is not None
        else source.geometry
    )
    restored_authority = (
        origin is not None
        and origin.source_topology_id == source.mesh.topology_id
        and all(
            isinstance(
                element,
                _owner_phydrax_discretization_fem__reference_types.FiniteElementSpec,
            )
            for element in elements
        )
    )
    direct_authority = uniform is not None or periodic is not None or restored_authority
    source_cells, original_cells, bank, original_bank = _full_p1_source_cells(
        source,
        authority_mesh,
        authority_geometry,
        elements,
        routes,
        direct_authority=direct_authority,
        restored_authority=restored_authority,
        budget=budget,
    )
    if direct_authority:
        restored = _whole_source_inverse_edit(authority_mesh, edit)
        if restored is not None:
            return _FullP1EditTarget(restored, authority_geometry)
    phase = _FullP1EditSource(
        source_cells,
        original_cells,
        bank,
        original_bank,
        authority_geometry,
        direct_authority,
        uniform,
        hierarchy,
    )
    target = _assemble_full_p1_target(source, edit, phase, budget=budget, scope=scope)
    return target


def _complete_coarsening_witnesses(
    source: CellMesh,
    target: CellMesh,
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage,
    witnesses: transition_types_owner.NestedReferenceWitnesses | None,
    /,
) -> transition_types_owner.NestedReferenceWitnesses | None:
    """Fill a pure coarsening chart bank with preserved-cell identity maps."""

    if witnesses is None or np.asarray(witnesses.fine_cell_ids).size == 0:
        return None
    dimension = target.topological_dimension
    cells = lineage.entity_lineage(dimension)
    relation_sources = np.asarray(cells.source_global_ids, dtype=np.int64)
    relation_targets = np.asarray(cells.target_global_ids, dtype=np.int64)
    relation_kinds = np.asarray(cells.relation_kinds, dtype=np.int32)
    preserved = relation_kinds == int(
        _dependency_phydrax_meshing__lineage_records.EntityLineageKind.PRESERVED
    )
    coarsened = relation_kinds == int(
        _dependency_phydrax_meshing__lineage_records.EntityLineageKind.COARSENED_INTO
    )
    if (
        not np.all(preserved | coarsened)
        or np.asarray(cells.created_target_ids).size
        or np.asarray(cells.deleted_source_ids).size
    ):
        return None
    source_cells = {
        int(identifier): (
            block.cell_kind,
            tuple(np.asarray(source.vertex_global_ids)[np.asarray(vertices)].tolist()),
        )
        for block in source.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    target_cells = {
        int(identifier): (
            block.cell_kind,
            tuple(np.asarray(target.vertex_global_ids)[np.asarray(vertices)].tolist()),
        )
        for block in target.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }

    fine = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    reference = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    represented = {int(identifier): row for row, identifier in enumerate(fine)}
    identity_sources: list[int] = []
    identity_targets: list[int] = []
    identity_vertices: list[np.ndarray] = []
    for source_id, target_id in zip(
        relation_sources[preserved], relation_targets[preserved], strict=True
    ):
        source_cell = source_cells.get(int(source_id))
        target_cell = target_cells.get(int(target_id))
        if source_cell is None or source_cell != target_cell:
            return None
        vertices = np.asarray(
            _owner_phydrax_discretization__reference_cell.reference_cell_topology(
                source_cell[0]
            ).vertices,
            dtype=np.float64,
        )
        if int(source_id) in represented:
            row = represented[int(source_id)]
            if (
                coarse[row] != target_id
                or not np.array_equal(reference[row, : vertices.shape[0]], vertices)
                or np.any(reference[row, vertices.shape[0] :] != 0.0)
            ):
                raise ValueError(
                    f"MeshAdaptation.coarsening.preserved_cell must retain its identity reference chart; got {int(source_id)!r}"
                )
            continue
        identity_sources.append(int(source_id))
        identity_targets.append(int(target_id))
        identity_vertices.append(vertices)
    width = max(
        reference.shape[1],
        max((vertices.shape[0] for vertices in identity_vertices), default=0),
    )
    bank = (
        None
        if witnesses.fine_reference_homogeneous is None
        else np.asarray(witnesses.fine_reference_homogeneous, dtype=np.float64)
    )
    if reference.shape[1] < width:
        reference = np.pad(reference, ((0, 0), (0, width - reference.shape[1]), (0, 0)))
        if bank is not None:
            bank = np.pad(bank, ((0, 0), (0, width - bank.shape[1]), (0, 0)))
    if identity_vertices:
        identities = np.zeros(
            (len(identity_vertices), width, dimension), dtype=np.float64
        )
        for row, vertices in enumerate(identity_vertices):
            identities[row, : vertices.shape[0]] = vertices
        fine = np.concatenate((fine, np.asarray(identity_sources, dtype=np.int64)))
        coarse = np.concatenate((coarse, np.asarray(identity_targets, dtype=np.int64)))
        reference = np.concatenate((reference, identities))
        if bank is not None:
            # Zero rows declare the dyadic identity corners exactly representable.
            bank = np.concatenate(
                (bank, np.zeros((len(identity_vertices), width, dimension + 1)))
            )
    if set(fine.tolist()) != set(source_cells) or set(coarse.tolist()) != set(
        target_cells
    ):
        return None
    return transition_types_owner.NestedReferenceWitnesses(fine, coarse, reference, bank)


def _adaptation_certification_inputs(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs | None:
    """Rebind target certification work/storage to the adaptation policy ledger."""

    if source.certification is None:
        return None

    retained = source.certification.request
    original = retained.limits
    certificate_limits = (
        _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits(
            maximum_candidate_pairs=original.maximum_candidate_pairs,
            maximum_ray_tests=original.maximum_ray_tests,
            maximum_source_samples=original.maximum_source_samples,
            maximum_distance_evaluations=original.maximum_distance_evaluations,
            maximum_subdivision_depth=original.maximum_subdivision_depth,
            maximum_subdivision_pieces=original.maximum_subdivision_pieces,
            maximum_bernstein_nodes=original.maximum_bernstein_nodes,
            maximum_periodic_images=original.maximum_periodic_images,
            maximum_work_units=limits.maximum_work_units,
            maximum_scratch_bytes=limits.maximum_scratch_bytes,
        )
    )
    return _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs(
        source.mesh,
        source.geometry,
        retained.schedule,
        domain=retained.domain,
        cell_regions=retained.cell_regions,
        source=retained.source,
        fidelity_tolerance=retained.fidelity_tolerance,
        fidelity_sample_order=retained.fidelity_sample_order,
        limits=certificate_limits,
        junction_vertices=retained.junction_vertices,
        scoped_fidelity=retained.scoped_fidelity,
        mapped_source_evidence=retained.mapped_source_evidence
        if tree_content_equal(original, certificate_limits)
        else None,
        budget=budget,
    )


@final
class _NativeGeometry(StrictModule):
    """Topology and coordinate proof at the native publication boundary."""

    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    mesh: CellMesh
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage
    stencil: (
        _dependency_phydrax_meshing__lineage_records.VertexInterpolationStencil | None
    )
    geometry: CellGeometrySpec
    geometry_transition: transition_types_owner.CellGeometryTransition | None
    audit_prepared_validity: (
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
        | None
    )
    metric_prepared_embedding: global_embedding_owner.GlobalEmbeddingCertificate | None
    metric_prepared_fidelity: source_fidelity_owner.SourceFidelityCertificate | None


@final
class _RetainedSurfaceFidelity(StrictModule):
    """Authenticated restriction fidelity with the limits that admitted it."""

    fidelity: source_fidelity_owner.SourceFidelityCertificate
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits


def _retained_surface_fidelity(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> _RetainedSurfaceFidelity | None:
    if source.certification is not None and isinstance(
        source.certification.request.source,
        _owner_phydrax_geometry__surface_source_support.SurfaceNativeRestrictionBoundarySource,
    ):
        retained_source = source.certification.request.source
        retained_source.require_current()
        restriction = source.geometry.restriction_source
        if (
            restriction is None
            or retained_source.mesh.mesh_id != source.mesh.mesh_id
            or _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                retained_source.geometry
            )
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source.geometry
            )
            or retained_source.support.root_geometry_id != restriction.source_geometry_id
            or retained_source.support.root_topology_id != restriction.source_topology_id
        ):
            raise ValueError(
                f"MeshAdaptation.source_geometry.restriction_source must match retained source-fidelity provenance; got {source.geometry.geometry_layout_id!r}"
            )
        prepared_source_fidelity = source.certification.fidelity
        if prepared_source_fidelity is None:
            raise ValueError(
                f"MeshAdaptation.source.certification.fidelity must contain restriction fidelity evidence; got {source.result_id!r}"
            )
        return _RetainedSurfaceFidelity(
            prepared_source_fidelity, source.certification.request.limits
        )
    return None


@final
class _SurfaceGeometry(StrictModule):
    """Accepted chart transition with its target lineage and retained certificates."""

    mesh: CellMesh
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage
    geometry_transition: transition_types_owner.CellGeometryTransition
    audit_prepared_validity: (
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
        | None
    )
    metric_prepared_embedding: global_embedding_owner.GlobalEmbeddingCertificate | None
    metric_prepared_fidelity: source_fidelity_owner.SourceFidelityCertificate | None


def _prepare_uv_geometry(
    prepared: PreparedMeshAdaptation,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    mesh: CellMesh,
    surface_metric_outcome: _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome,
    surface_transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    surface_curve_witness: PreparedSurfaceCurveWitness | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _SurfaceGeometry:
    source = prepared.source
    metric_prepared_fidelity: source_fidelity_owner.SourceFidelityCertificate | None = (
        None
    )
    witness = _owner_phydrax_meshing__surface_metric.prepare_surface_metric_source(
        source, surface_transfer, budget=budget
    )
    _, material_root_witness = _surface_material_epoch(source, witness)
    if surface_curve_witness is None:
        raise TypeError(
            f"MeshAdaptation.surface_curve_witness must bind source curves for UV reconstruction; got {surface_curve_witness!r}"
        )
    retained_fidelity = _retained_surface_fidelity(source)
    prepared_source_fidelity = (
        None if retained_fidelity is None else retained_fidelity.fidelity
    )
    prepared_source_certificates = None
    if source.certification is not None and source.certification.embedding is not None:
        prepared_source_certificates = (
            source.audit.validity,
            source.certification.embedding,
        )
    chart_transition = (
        _owner_phydrax_meshing__surface_metric.prepare_policy_surface_metric_transition(
            source,
            mesh,
            surface_metric_outcome,
            surface_transfer,
            prepared.policy,
            witness,
            maximum_fidelity=_surface_fidelity_limit(source),
            curve_witness=surface_curve_witness,
            material_root_witness=material_root_witness,
            prepared_source_fidelity=prepared_source_fidelity,
            prepared_source_certificates=prepared_source_certificates,
            scope=scope,
            budget=budget,
        )
    )
    audit_prepared_validity = chart_transition.deformation.target_validity
    metric_prepared_embedding = chart_transition.deformation.target_embedding
    geometry_transition = chart_transition.transition
    mesh = chart_transition.target_mesh
    target_fidelity_bound = float(
        np.max(
            np.asarray(chart_transition.deformation.target_fidelity_bounds),
            initial=0.0,
        )
    )
    if (
        retained_fidelity is not None
        and chart_transition.deformation.target_domain_coverage == "certified"
    ):
        prepared_source_fidelity = retained_fidelity.fidelity
        if target_fidelity_bound > prepared_source_fidelity.tolerance:
            raise ValueError(
                f"MeshAdaptation.target.fidelity_bound must not exceed the source tolerance; got {target_fidelity_bound!r}"
            )
        fidelity_binding = (
            _dependency_phydrax_geometry__mesh_certificates.MeshCertificateBinding(
                mesh,
                geometry_transition.geometry,
                retained_fidelity.fidelity.binding.coordinate_scope,
                retained_fidelity.limits,
                source_id=surface_transfer.support.domain.source_id,
                source_revision=surface_transfer.support.domain.source_revision,
            )
        )
        metric_prepared_fidelity = source_fidelity_owner.SourceFidelityCertificate(
            fidelity_binding,
            (),
            tolerance=prepared_source_fidelity.tolerance,
            sample_order=prepared_source_fidelity.sample_order,
            bounds=source_fidelity_owner.SourceFidelityBounds(
                semantics=("certified", "certified"),
                mesh_to_source=(target_fidelity_bound, 0.0),
                source_to_mesh=(target_fidelity_bound, 0.0),
                sample_counts=(
                    mesh.entity_set(mesh.topological_dimension).count,
                    surface_transfer.support.root_cell_ids.shape[0],
                ),
            ),
        )
    lineage = _owner_phydrax_meshing__periodic.metric_target_lineage(
        source.mesh, edit, mesh, scope=scope, budget=budget
    )
    return _SurfaceGeometry(
        mesh,
        lineage,
        geometry_transition,
        audit_prepared_validity,
        metric_prepared_embedding,
        metric_prepared_fidelity,
    )


def _polyhedral_publication_bounds(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    /,
    *,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None = None,
) -> tuple[int, int, int]:
    """Concrete host banks of polyhedral assembly and publication.

    This supplements (not replaces) the coordinate/proof owners. Counts are
    actual face occurrences, corner occurrences, padded entity keys, relation
    rows and stencil entries. No request limit is used as an allocation size.
    Device/compiler memory and certificate-owner storage are not claimed here.
    """
    vertices = edit.coordinates.shape[0]
    cells = loops = corners = 0
    target_key_width = 2
    squared_arities = 0
    for block in edit.blocks:
        if not isinstance(
            block, _dependency_phydrax_meshing__topology_edit.PolyhedralTopologyEditBlock
        ):
            raise TypeError(
                "polyhedral.publication.blocks must retain packed polyhedral connectivity"
            )
        connectivity = block.connectivity
        cells += connectivity.cell_count
        loops += connectivity.cell_face_values.size
        offsets = np.asarray(connectivity.face_vertex_offsets)
        for face in np.asarray(connectivity.cell_face_values):
            width = int(offsets[int(face) + 1]) - int(offsets[int(face)])
            corners += width
            target_key_width = max(target_key_width, width)
            squared_arities += width * width
        cell_offsets = np.asarray(connectivity.cell_vertex_offsets)
        for start, end in zip(cell_offsets[:-1], cell_offsets[1:], strict=True):
            squared_arities += (int(end) - int(start)) ** 2
    if limits is not None:
        _dependency_phydrax_meshing__polyhedral_generation._admit_polyhedral_mesh_capacity(
            vertices, cells, loops, corners, limits
        )
    required, retained = (
        _dependency_phydrax_meshing__polyhedral_generation._polyhedral_mesh_storage_bound(
            vertices, cells, loops, corners
        )
    )
    # prepare_topology_edit_target keeps its lifted ID probe live while it
    # constructs the published carrier. Both real builder graphs need room.
    required *= 2
    periodic = (
        edit.target_periodic_topology
        if edit.target_periodic_topology is not None
        else source.mesh.periodic_topology
    )
    if periodic is not None:
        periodic_required, periodic_retained = (
            _dependency_phydrax_meshing__polyhedral_generation._periodic_topology_storage_bound(
                vertices,
                cells,
                loops,
                corners,
                squared_arities + 4 * corners,
                periodic.cell.rank,
            )
        )
        required += periodic_required
        retained += periodic_retained
    source_entities = sum(source.mesh.entity_set(d).entity_ids.size for d in range(4))
    target_entities = vertices + corners + loops + cells
    relation_rows = sum(value.kinds.size for value in edit.relations)
    key_slots = sum(
        value.source_keys.size + value.target_keys.size for value in edit.relations
    )
    source_key_width = max(
        (block.vertices.shape[1] for block in source.mesh.blocks), default=2
    )
    key_slots += source_entities * source_key_width + target_entities * target_key_width
    # entity_lineage: key sorting/matching, implicit preserved rows, explicit
    # source/target IDs and kinds. Sixteen banks cover simultaneously live
    # lookup, permutation, validity and relation-validation operands.
    lineage_slots = 16 * (key_slots + relation_rows + source_entities + target_entities)
    stencil_slots = edit.stencil_sources.size
    # Stencil construction and as_transfer: source-ID sorting/lookup, support
    # validation, CSR indices/data/row pointers and linear reproduction checks.
    transfer_slots = 24 * (stencil_slots + vertices + source.mesh.coordinates.shape[0])
    scopes = len(source.patches) + len(source.zones) + len(source.labels)
    scopes += len(source.region_boundary_evidence)
    scope_slots = sum(
        value.scope.global_entity_ids.size
        for value in (*source.patches, *source.zones, *source.labels)
    )
    # Each scope's deciding-source remap sorts at most the lineage rows plus
    # implicit preserved rows. Region-boundary renewal repeats those remaps;
    # it never constructs a Cartesian entity incidence.
    organization_slots = 32 * (
        scope_slots + scopes * (relation_rows + source_entities + target_entities)
    )
    attribute_slots = 0
    for attribute in source.attributes:
        width = int(np.prod(attribute.component_shape))
        attribute_slots += (
            (target_entities + relation_rows + source_entities)
            * max(8, attribute.global_values.dtype.itemsize)
            * max(1, width)
        )
    headers = 256 * (128 + 32 * (scopes + len(source.attributes)))
    text_units = sum(
        len(value.name)
        for value in (*source.patches, *source.zones, *source.labels, *source.attributes)
    )
    text_units += sum(
        len(zone.material_id) if zone.material_id is not None else 0
        for zone in source.zones
    )
    # Eight Unicode/UTF-8/fingerprint staging copies, at four bytes per code
    # point. Names and material IDs have no fixed maximum length.
    text_storage = 32 * text_units
    extra = 8 * (lineage_slots + transfer_slots + organization_slots)
    extra += attribute_slots + headers + text_storage
    records = (
        512
        + 64 * cells
        + 16 * target_entities
        + 32 * (scopes + len(source.attributes))
        + 64
    )
    # All these logical output banks remain covered while any published
    # topology/lineage/organization/transfer alias survives.
    retained += extra
    required += extra + 1024 + 512 * records
    return required, retained, records


@contextmanager
def _polyhedral_publication_storage(
    prepared: PreparedMeshAdaptation,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    scope: NativeExecutionScope,
    /,
) -> Iterator[
    tuple[
        _dependency_phydrax_meshing__polyhedral_generation._PolyhedralTopologyStorage,
        int,
        int,
    ]
    | None
]:
    if prepared.policy.route is not MeshAdaptationRoute.NATIVE_POLYHEDRAL:
        yield None
        return
    with _dependency_phydrax_meshing__polyhedral_generation._PolyhedralTopologyStorage(
        scope
    ) as storage:
        # The bound profiler itself copies offset/index banks from JAX. Admit
        # those simultaneous views and their traversal before computing bounds.
        profile_bytes = profile_work = 0
        for block in edit.blocks:
            if not isinstance(
                block, _dependency_phydrax_meshing__topology_edit.PolyhedralTopologyEditBlock
            ):
                raise TypeError(
                    "polyhedral.publication.blocks must retain packed polyhedral connectivity"
                )
            connectivity = block.connectivity
            profile_bytes += 3 * 256 + 8 * (
                connectivity.face_vertex_offsets.size
                + connectivity.cell_face_values.size
                + connectivity.cell_vertex_offsets.size
            )
            profile_work += connectivity.cell_face_values.size + connectivity.cell_count
        storage.workspace.set_bound(profile_bytes)
        scope.charge(work=profile_work)
        required, retained, records = _polyhedral_publication_bounds(
            prepared.source, edit, limits=prepared.policy.limits
        )
        # Create in the active publication target, not as a root token that
        # would later be grown under a foreign native_target.
        storage.workspace.set_bound(required)
        # Publication walks explicit lineage/stencil/organization banks in
        # addition to the topology builder. These are host scalar visits only;
        # source-coordinate and native construction work keep their own ledger.
        scope.charge(
            work=records + edit.stencil_sources.size
            + sum(
                value.source_keys.size + value.target_keys.size + value.kinds.size
                for value in edit.relations
            )
        )
        yield storage, retained, records


def _prepare_native_geometry(
    prepared: PreparedMeshAdaptation,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    /,
    *,
    numeric_version: str,
    surface_metric_outcome: _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome
    | _dependency_phydrax_meshing__surface_metric.SphereMetricOutcome
    | None,
    plc_source: _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource
    | _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource
    | None,
    polyhedral_geometry: CellGeometrySpec | None,
    surface_curve_witness: PreparedSurfaceCurveWitness | None,
    bisection_hierarchy: BisectionHierarchy | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _NativeGeometry:
    source = prepared.source
    periodic_surface = (
        surface_metric_outcome.periodic_stage
        if isinstance(
            surface_metric_outcome,
            _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome,
        )
        else None
    )
    coefficient_target = _full_p1_edit_geometry(
        prepared,
        edit,
        bisection_hierarchy=bisection_hierarchy,
        budget=budget,
        scope=scope,
    )
    edit, coefficient_geometry = coefficient_target.edit, coefficient_target.geometry
    if (
        coefficient_geometry is None
        and surface_metric_outcome is None
        and plc_source is None
        and polyhedral_geometry is None
        and _composes_nested_child_maps(prepared, edit)
    ):
        # Restricted, composed or rational-source parents: the nested transition
        # publishes one rational child map per target block.
        edit = _nested_action_blocks(edit)
    mesh, lineage, stencil = (
        _dependency_phydrax_meshing__topology_edit.assemble_topology_edit(
            source.mesh, edit, numeric_version=numeric_version, budget=budget
        )
    )
    audit_prepared_validity: (
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
        | None
    ) = None
    metric_prepared_embedding: (
        global_embedding_owner.GlobalEmbeddingCertificate | None
    ) = None
    metric_prepared_fidelity: source_fidelity_owner.SourceFidelityCertificate | None = (
        None
    )
    if surface_metric_outcome is not None:
        surface_transfer = prepared.policy.association_transfer
        match surface_transfer:
            case _dependency_phydrax_meshing__association.SurfaceAssociationTransfer():
                pass
            case _:
                raise TypeError(
                    f"MeshAdaptationPolicy.association_transfer must provide source support for surface reconstruction; got {type(surface_transfer).__name__!r}"
                )
        match surface_metric_outcome:
            case (
                _dependency_phydrax_meshing__surface_metric.SphereMetricOutcome() as actual
            ):
                mesh = actual.target_mesh
                lineage = _owner_phydrax_meshing__periodic.metric_target_lineage(
                    source.mesh, edit, mesh, scope=scope, budget=budget
                )
                geometry_transition = _owner_phydrax_discretization__cell_geometry_transfer.transition_chart_deformed_cell_geometry(
                    source.mesh,
                    source.geometry,
                    mesh,
                    actual.reconstruction,
                    actual.deformation,
                    policy=prepared.policy.geometry_transition,
                    budget=budget,
                )
                audit_prepared_validity = actual.reconstruction.target_validity
                metric_prepared_embedding = actual.reconstruction.target_embedding
            case _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome():
                if periodic_surface is not None:
                    mesh = periodic_surface.target_mesh
                    lineage = _owner_phydrax_meshing__periodic.metric_target_lineage(
                        source.mesh, edit, mesh, scope=scope, budget=budget
                    )
                    geometry_transition = (
                        periodic_surface.geometry_stage.geometry_transition
                    )
                else:
                    uv = _prepare_uv_geometry(
                        prepared,
                        edit,
                        mesh,
                        surface_metric_outcome,
                        surface_transfer,
                        surface_curve_witness=surface_curve_witness,
                        budget=budget,
                        scope=scope,
                    )
                    mesh, lineage, geometry_transition = (
                        uv.mesh,
                        uv.lineage,
                        uv.geometry_transition,
                    )
                    audit_prepared_validity = uv.audit_prepared_validity
                    metric_prepared_embedding = uv.metric_prepared_embedding
                    metric_prepared_fidelity = uv.metric_prepared_fidelity
            case invalid:
                assert_never(invalid)
        geometry = geometry_transition.geometry
    elif plc_source is not None:
        geometry_transition = None
        geometry = CellGeometrySpec.plc(mesh, plc_source)
    elif polyhedral_geometry is not None:
        geometry_transition = None
        match polyhedral_geometry.exact_source:
            case None:
                geometry = CellGeometrySpec.affine(mesh)
            case (
                _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource()
                | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource()
                | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource() as power_source
            ):
                geometry = CellGeometrySpec.power(mesh, power_source, budget=budget)
            case ExactAlgebraicCellGeometrySource():
                raise ValueError(
                    "MeshAdaptation.polyhedral_geometry.exact_source must use a power-polyhedron source; got an algebraic nodal source"
                )
            case (
                _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource()
                | _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource()
                | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
            ):
                raise ValueError(
                    f"MeshAdaptation.polyhedral_geometry.exact_source must not republish PLC geometry on polyhedra; got {type(polyhedral_geometry.exact_source).__name__!r}"
                )
            case invalid:
                assert_never(invalid)
    else:
        geometry_transition = (
            None
            if coefficient_geometry is not None
            else _geometry_transition(prepared, edit, mesh, budget=budget)
        )
        geometry = (
            coefficient_geometry
            if coefficient_geometry is not None
            else CellGeometrySpec.affine(mesh)
            if geometry_transition is None
            else geometry_transition.geometry
        )
    if geometry_transition is not None:
        mesh = mesh.with_coordinates(
            geometry_transition.vertex_coordinates, numeric_version=numeric_version
        )
        geometry = geometry_transition.geometry
    if (
        geometry is coefficient_target.geometry
        and coefficient_target.image_geometry is not None
        and coefficient_target.images is not None
    ):
        from ..discretization._coordinate_identity import coordinate_scope_key

        # Retain only the exact bank actually evaluated for this geometry. A
        # published-affine carrier and the coefficient source have DIFFERENT
        # keys whenever their root fields differ.
        key = coordinate_scope_key(mesh, coefficient_target.image_geometry)
        images = coefficient_target.images
        if len(images) != mesh.coordinates.shape[0]:
            raise ValueError(
                "Full P1 coordinate images must bind every actual target vertex."
            )
        budget.retain_basis((key, images), operand=True, _kernel_operands=True)
        budget.coordinate_vertex_image_cache[key] = images
    return _NativeGeometry(
        edit,
        mesh,
        lineage,
        stencil,
        geometry,
        geometry_transition,
        audit_prepared_validity,
        metric_prepared_embedding,
        metric_prepared_fidelity,
    )


@final
class _NativePublication(StrictModule):
    """Inherited scientific declarations and source proofs for target certification."""

    associations: tuple[association_state_owner.GeometryAssociation, ...]
    attributes: tuple[_dependency_phydrax_meshing__organization.MeshAttribute, ...]
    boundary: SurfaceModel | None
    certification_inputs: (
        _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs | None
    )
    prepared_volume: (
        _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence | None
    )


@final
class _PreparedAssociations(StrictModule):
    """Propagated source associations and their optional volume proof."""

    associations: tuple[association_state_owner.GeometryAssociation, ...]
    volume: _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence | None


def _bind_plc_source_proofs(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    planar_only: bool = False,
) -> list[plc_mapped_support_owner.MappedPlcSupport] | None:
    """Bind this preparation's earned source proofs to the execution ledger.

    Their work and queries are debited before the execution scope opens.
    Only the issued attachment's exact earning receipt permits binding with
    no replayed charges; a preparation without it imports mapped proof charges.
    """
    attached = prepared.plc_source_proofs
    if attached is None or not attached.proofs:
        return None
    transfer = prepared.policy.association_transfer
    if attached.owner_id != prepared.prepared_id or not isinstance(
        transfer,
        (
            _dependency_phydrax_meshing__association.PlcAssociationTransfer,
            _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer,
        ),
    ):
        raise ValueError(
            f"PreparedMeshAdaptation.plc_source_proofs must bind this preparation and its PLC or composed PLC transfer; got owner {attached.owner_id!r}, expected {prepared.prepared_id!r}, transfer {type(transfer).__name__!r}"
        )
    preparation = prepared.preparation_evidence
    charged = preparation is not None
    if preparation is not None:
        preparation.require_valid()
        issued = _ISSUED_PLC_SOURCE_PROOFS.get(id(attached))
        if (
            preparation.owner_id != prepared.prepared_id
            or issued is None
            or issued[0]() is not attached
            or issued[1] is not attached.owner_id
            or issued[2] is not attached.proofs
            or issued[3] is not preparation
        ):
            raise ValueError(
                "PreparedMeshAdaptation.plc_source_proofs must retain their issued "
                "attachment and exact earning preparation receipt; got a stale or foreign receipt binding"
            )
    views: list[plc_mapped_support_owner.MappedPlcSupport] = []
    for proof in attached.proofs:
        if isinstance(proof, plc_planar_support_owner.PlanarPlcSourceProof):
            # A planar proof carries no separable charges: only the request
            # whose charged receipt earned it may consume it.
            if charged and isinstance(
                transfer, _dependency_phydrax_meshing__association.PlcAssociationTransfer
            ):
                plc_planar_support_owner.bind_planar_source_proof(
                    transfer, prepared.source, proof, budget=budget
                )
            continue
        if planar_only:
            continue
        views.append(
            transfer.bind_source_proof(
                prepared.source, proof, budget=budget, import_charges=not charged
            )
        )
    return views or None


def _propagate_plc_associations(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    association_transfer: _dependency_phydrax_meshing__association.PlcAssociationTransfer
    | _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer,
    phase: _NativeGeometry,
    certification_inputs: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs
    | None,
    certification_cell_regions: np.ndarray | None,
    /,
    *,
    source_views: list[plc_mapped_support_owner.MappedPlcSupport] | None,
    planar_embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    planar_prepared: _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PreparedAssociations:
    mesh, lineage, geometry = phase.mesh, phase.lineage, phase.geometry
    certification_prepared = planar_prepared
    planar_coverage = None
    if planar_prepared is not None:
        if not isinstance(
            planar_prepared.coverage, domain_coverage_owner.DomainCoverageCertificate
        ):
            raise ValueError(
                "MeshAdaptation.planar_coverage must certify the exact represented PLC domain; got approximate source coverage"
            )
        planar_coverage = planar_prepared.coverage
    target_evidence: (
        list[_owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence]
        | None
    ) = None
    target_request = None
    if (
        isinstance(
            association_transfer,
            _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer,
        )
        or association_transfer.domain.ambient_dimension == 3
    ) and certification_inputs is not None:
        target_request = certification_inputs.transition_request(
            mesh,
            geometry,
            lineage=lineage,
            cell_regions=certification_cell_regions,
            budget=budget,
        )
        target_evidence = []
    match association_transfer:
        case _dependency_phydrax_meshing__association.PlcAssociationTransfer():
            associations = association_transfer.propagate(
                source,
                lineage,
                mesh,
                geometry=geometry,
                embedding=planar_embedding,
                coverage=planar_coverage,
                _certification_request=target_request,
                _geometry_transition=phase.geometry_transition,
                _prepared_target=target_evidence,
                _prepared_source=source_views,
                budget=budget,
            )
        case _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer():
            associations = association_transfer.propagate(
                source,
                lineage,
                mesh,
                geometry=geometry,
                _certification_request=target_request,
                _geometry_transition=phase.geometry_transition,
                _prepared_target=target_evidence,
                _prepared_source=source_views,
                budget=budget,
            )
        case invalid:
            assert_never(invalid)
    if target_evidence is not None:
        if len(target_evidence) != 1:
            raise ValueError(
                f"MeshAdaptation.associations.prepared_target must publish one coherent target proof; got {len(target_evidence)!r}"
            )
        certification_prepared = target_evidence[0]
    return _PreparedAssociations(associations, certification_prepared)


def _prepare_native_publication(
    prepared: PreparedMeshAdaptation,
    phase: _NativeGeometry,
    certification_inputs: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs
    | None,
    certification_cell_regions: np.ndarray | None,
    /,
    *,
    polyhedral_construction: _dependency_phydrax_meshing__polyhedral_generation.PolyhedralConstruction
    | None,
    planar_embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    planar_prepared: _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _NativePublication:
    source = prepared.source
    mesh, lineage, geometry, geometry_transition = (
        phase.mesh,
        phase.lineage,
        phase.geometry,
        phase.geometry_transition,
    )
    association_transfer = prepared.policy.association_transfer
    associations: tuple[association_state_owner.GeometryAssociation, ...] = ()
    certification_prepared = None
    if polyhedral_construction is not None and source.associations:
        if not isinstance(
            prepared.request,
            _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralMeshAdaptation,
        ):
            raise TypeError(
                f"MeshAdaptation.request must provide a polyhedral request for construction-derived associations; got {type(prepared.request).__name__!r}"
            )
        associations = _dependency_phydrax_meshing__polyhedral_adaptation.regenerated_polyhedral_associations(
            source,
            prepared.request,
            polyhedral_construction,
            mesh,
            budget=budget,
        )
    elif source.associations:
        match association_transfer:
            case _dependency_phydrax_meshing__association.BRepAssociationTransfer():
                associations = association_transfer.propagate(
                    source, lineage, mesh, budget=budget
                )
            case (
                _dependency_phydrax_meshing__association.PlcAssociationTransfer()
                | _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer()
            ):
                propagated = _propagate_plc_associations(
                    source,
                    association_transfer,
                    phase,
                    certification_inputs,
                    certification_cell_regions,
                    source_views=_bind_plc_source_proofs(prepared, budget=budget),
                    planar_embedding=planar_embedding,
                    planar_prepared=planar_prepared,
                    budget=budget,
                )
                associations, certification_prepared = (
                    propagated.associations,
                    propagated.volume,
                )
            case _dependency_phydrax_meshing__association.SurfaceAssociationTransfer():
                associations = association_transfer.propagate(
                    source,
                    lineage,
                    mesh,
                    geometry=geometry,
                    deformation=None
                    if geometry_transition is None
                    else geometry_transition.chart_deformation,
                    budget=budget,
                )
            case _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer():
                associations = association_transfer.propagate(
                    source, lineage, mesh, geometry=geometry, budget=budget
                )
            case _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer():
                associations = association_transfer.propagate(
                    source, lineage, mesh, geometry=geometry, budget=budget
                )
            case _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer():
                associations = association_transfer.propagate(
                    source,
                    lineage,
                    mesh,
                    geometry=geometry,
                    scope=scope,
                    budget=budget,
                    _prepared_source=prepared.periodic_source_proofs,
                )
            case None:
                raise ValueError(
                    f"MeshAdaptationPolicy.association_transfer must be supplied for associated native publication; got {association_transfer!r}"
                )
            case invalid:
                assert_never(invalid)
    attributes: tuple[_dependency_phydrax_meshing__organization.MeshAttribute, ...] = ()
    if source.attributes:
        attributes = _owner_phydrax_meshing__layer_core.remap_layer_index_attribute(
            source, lineage, mesh
        )
    if (
        isinstance(
            association_transfer,
            (
                _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
                _dependency_phydrax_meshing__association.BRepAssociationTransfer,
            ),
        )
        and certification_inputs is not None
    ):
        certification_inputs = _owner_phydrax_meshing__surface_association_transfer.successor_certification_inputs(
            association_transfer.support
            if isinstance(
                association_transfer,
                _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
            )
            else None,
            source,
            certification_inputs,
            lineage,
            mesh,
            geometry,
            deformation=None
            if geometry_transition is None
            else geometry_transition.chart_deformation,
            budget=budget,
        )
    boundary = None
    if source.boundary is not None:
        if not isinstance(
            association_transfer,
            _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer,
        ):
            raise ValueError(
                f"MeshAdaptationPolicy.association_transfer must remap the boundary source model for publication; got {type(association_transfer).__name__!r}"
            )
        boundary = association_transfer.remap_boundary(
            source, lineage, mesh, geometry=geometry, budget=budget
        )
    return _NativePublication(
        associations, attributes, boundary, certification_inputs, certification_prepared
    )


def _bind_native_transfer(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    target: _dependency_phydrax_meshing__result.CellMeshingResult,
    phase: _NativeGeometry,
    /,
    *,
    conservative: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> FiniteElementTopologyTransfer | None:
    stencil, geometry_transition = phase.stencil, phase.geometry_transition
    affine = geometry_transition is None
    source_measures = None
    target_measures = None
    if conservative:
        if stencil is None:
            raise ValueError(
                f"MeshAdaptation.transfer.stencil must be supplied for conservative vertex transfer; got {stencil!r}"
            )
        if affine:
            source_measures = _p1_vertex_measures(source.mesh)
            target_measures = _p1_vertex_measures(target.mesh)
        else:
            source_measures = _dependency_phydrax_discretization__cell_geometry_transfer.cell_geometry_vertex_measures(
                source.mesh, source.geometry, budget=budget
            )
            target_measures = _dependency_phydrax_discretization__cell_geometry_transfer.cell_geometry_vertex_measures(
                target.mesh, target.geometry, budget=budget
            )
    transfer = (
        None
        if stencil is None
        else stencil.as_transfer(
            source.mesh.vertex_global_ids,
            source_topology_id=source.mesh.topology_id,
            target_topology_id=target.mesh.topology_id,
            preserves_linear=affine,
            conservative=conservative,
            source_coordinates=source.mesh.coordinates,
            target_coordinates=target.mesh.coordinates,
            source_measures=source_measures,
            target_measures=target_measures,
        )
    )
    return transfer


def _source_fidelity_predecessor(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
) -> source_fidelity_owner.SourceFidelityPredecessor | None:
    report = source.certification
    if report is None or not report.passed:
        return None
    request = report.request
    if (
        request is None
        or request.source is None
        or not isinstance(
            request.domain,
            (
                domain_coverage_owner.PiecewiseLinearDomain,
                mapped_reference_domain_owner.MappedReferenceDomain,
            ),
        )
        or request.cell_regions is None
        or report.embedding is None
        or not isinstance(
            report.coverage, domain_coverage_owner.DomainCoverageCertificate
        )
        or report.fidelity is None
        or report.fidelity.target_facet_ids is not None
        or report.fidelity.source_scope_id is not None
        or source.mesh.storage is not None
        or source.mesh.periodic_topology is not None
        or source.mesh.topological_dimension != source.mesh.ambient_dimension
    ):
        return None
    return source_fidelity_owner.SourceFidelityPredecessor(
        source.mesh,
        source.geometry,
        request.source,
        request.domain,
        np.asarray(request.cell_regions, dtype=np.int64),
        report.embedding,
        report.coverage,
        report.fidelity,
        request.limits,
    )


def _inherited_native_certificates(
    prepared: PreparedMeshAdaptation,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    certification_inputs: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs
    | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> tuple[
    _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    | None,
    global_embedding_owner.GlobalEmbeddingCertificate | None,
]:
    source = prepared.source
    receipt = source.execution_evidence
    validity_policy = prepared.policy.audit_policy.validity_policy
    if (
        prepared.policy.route
        not in (
            MeshAdaptationRoute.NATIVE_BISECTION,
            MeshAdaptationRoute.NATIVE_MIXED,
        )
        or source.mesh.storage is not None
        or mesh.storage is not None
        or receipt is None
        or not source.audit.validity.all_certified
        or source.audit.validity.policy_id != validity_policy.policy_id
    ):
        return None, None
    # Restricted children may import exact parent validity without any
    # unchanged cell; other targets need shared cells to reuse anything.
    restricted = geometry.restriction_source is not None
    old_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in source.mesh.blocks]
    )
    new_ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    if not restricted and not np.intersect1d(old_ids, new_ids).size:
        return None, None
    predecessor = cell_predecessor_owner.prepare_cell_geometry_predecessor(
        source.mesh,
        source.geometry,
        source.audit.validity,
        None if source.certification is None else source.certification.embedding,
        mesh,
        geometry,
        source_limits=None
        if source.certification is None or source.certification.embedding is None
        else source.certification.request.limits,
        budget=budget,
    )
    if not restricted and not any(row >= 0 for row in predecessor.source_rows):
        return None, None
    # The reused certificates were earned and charged by the completed source
    # request; this request retains that receipt as history, never as a charge.
    native_preparation_ledger(scope).record_history(scope, receipt)
    validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=validity_policy,
        predecessor=predecessor,
        budget=budget,
    )
    embedding = (
        None
        if certification_inputs is None or predecessor.source_embedding is None
        else global_embedding_owner.certify_global_embedding(
            mesh,
            geometry,
            validity,
            limits=certification_inputs.limits,
            junction_vertices=certification_inputs.junction_vertices,
            predecessor=predecessor,
            budget=budget,
        )
    )
    return validity, embedding


def _planar_target_certificates(
    prepared: PreparedMeshAdaptation,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    certification_inputs: _dependency_phydrax_meshing__certification_inputs.MeshCertificationInputs,
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage,
    cell_regions: np.ndarray | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence:
    """Certify a serial planar target's publication validity and embedding once.

    Exact restricted children and unchanged cells consume the source's
    authenticated predecessor/root evidence through the existing owner.
    Targets lacking those premises take the full proofs below.
    """
    validity, embedding = _inherited_native_certificates(
        prepared, mesh, geometry, certification_inputs, budget=budget, scope=scope
    )
    if validity is None:
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            geometry,
            mesh=mesh,
            policy=prepared.policy.audit_policy.validity_policy,
            budget=budget,
        )
    if embedding is None:
        embedding = global_embedding_owner.certify_global_embedding(
            mesh,
            geometry,
            validity,
            limits=certification_inputs.limits,
            junction_vertices=certification_inputs.junction_vertices,
            budget=budget,
        )
    request = certification_inputs.transition_request(
        mesh, geometry, lineage=lineage, cell_regions=cell_regions, budget=budget
    )
    if request.domain is None or request.cell_regions is None:
        raise ValueError(
            "MeshAdaptation.planar_certification must retain its domain and cell regions; got missing coverage inputs"
        )
    return _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence(
        mesh,
        geometry,
        validity,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
        embedding=embedding,
        limits=request.limits,
        source=request.source,
        fidelity_tolerance=request.fidelity_tolerance,
        fidelity_sample_order=request.fidelity_sample_order,
        scoped_fidelity=request.scoped_fidelity,
        budget=budget,
    )


@checked
def finalize_native(
    prepared: PreparedMeshAdaptation,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    kind: _dependency_phydrax_meshing__lineage_records.MeshTransitionKind,
    /,
    *,
    conservative: bool,
    common_refinement: _dependency_phydrax_geometry__supermesh.PreparedCommonRefinement
    | None = None,
    polyhedral_construction: _dependency_phydrax_meshing__polyhedral_generation.PolyhedralConstruction
    | None = None,
    polyhedral_geometry: CellGeometrySpec | None = None,
    surface_metric_outcome: _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome
    | _dependency_phydrax_meshing__surface_metric.SphereMetricOutcome
    | None = None,
    plc_source: _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource
    | _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource
    | None = None,
    surface_curve_witness: _owner_phydrax_meshing__surface_association_transfer.PreparedSurfaceCurveWitness
    | None = None,
    bisection_hierarchy: BisectionHierarchy | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _NativeTarget:
    """Publish the native target, geometry transition, lineage, and nodal transfer.

    Non-affine source maps are carried by the accepted geometry transition;
    target corners become mesh coordinates and the successor CellGeometrySpec
    is certified. Conservative simplex transfers use mapped hat measures.
    Nonnested polyhedral cell transfer uses certified geometric common refinement.
    Source restoration, organization renewal, certification, and transfer binding
    retain their required failure order. No intermediate target escapes.
    """

    with (
        adaptation_phase(scope, budget),
        _polyhedral_publication_storage(prepared, edit, scope) as polyhedral_storage,
    ):
        source = prepared.source
        periodic_surface = (
            surface_metric_outcome.periodic_stage
            if isinstance(
                surface_metric_outcome,
                _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome,
            )
            else None
        )
        numeric_version = (
            surface_metric_outcome.target_mesh.numeric_version
            if isinstance(
                surface_metric_outcome,
                _dependency_phydrax_meshing__surface_metric.SphereMetricOutcome,
            )
            else periodic_surface.target_mesh.numeric_version
            if periodic_surface is not None
            else f"adaptation:{prepared.prepared_id}"
        )
        phase = _prepare_native_geometry(
            prepared,
            edit,
            numeric_version=numeric_version,
            surface_metric_outcome=surface_metric_outcome,
            plc_source=plc_source,
            polyhedral_geometry=polyhedral_geometry,
            surface_curve_witness=surface_curve_witness,
            bisection_hierarchy=bisection_hierarchy,
            budget=budget,
            scope=scope,
        )
        edit, mesh, lineage, stencil = (
            phase.edit,
            phase.mesh,
            phase.lineage,
            phase.stencil,
        )
        geometry, geometry_transition = phase.geometry, phase.geometry_transition
        audit_prepared_validity = phase.audit_prepared_validity
        metric_prepared_embedding = phase.metric_prepared_embedding
        metric_prepared_fidelity = phase.metric_prepared_fidelity
        patches, zones, labels = (
            _dependency_phydrax_meshing__lineage.inherit_mesh_organization(
                source, mesh, lineage
            )
        )
        if prepared.policy.route is MeshAdaptationRoute.NATIVE_LEVEL_SET:
            patches, zones, labels = (
                _owner_phydrax_meshing__level_set.level_set_organization(
                    prepared, mesh, patches, zones, labels, edit
                )
            )
        region_boundary_evidence = region_boundary_transition(
            source, mesh, lineage, patches, zones, labels
        )
        region_evidence = None
        if source.region_evidence is not None:
            renewal = _owner_phydrax_meshing__compartments.revalidate_region_evidence(
                source,
                mesh,
                geometry,
                zones,
                patches,
                lineage=lineage,
                common_refinement=common_refinement,
                limits=prepared.policy.limits,
                budget=budget,
            )
            zones, patches, region_evidence = (
                renewal.zones,
                renewal.patches,
                renewal.region_evidence,
            )
        certification_inputs = _adaptation_certification_inputs(
            source, prepared.policy.limits, budget=budget
        )
        certification_cell_regions: np.ndarray | None = None
        if (
            certification_inputs is not None
            and certification_inputs.cell_regions is not None
            and region_evidence is not None
        ):
            domain = certification_inputs.domain
            if domain is None or domain.domain_id != region_evidence.domain.domain_id:
                raise ValueError(
                    f"MeshAdaptation.target.certification.domain must match the region-renewal domain; got {(domain.domain_id if domain is not None else None)!r}"
                )
            region_rows = {region: row for row, region in enumerate(domain.region_ids)}
            certification_cell_regions = np.asarray(
                [region_rows[region] for region in region_evidence.cell_region_ids],
                dtype=np.int64,
            )
        if polyhedral_construction is not None:
            certification_cell_regions = polyhedral_construction.cell_regions
        planar_embedding = None
        planar_prepared = None
        plc_transfer = prepared.policy.association_transfer
        if (
            isinstance(
                plc_transfer,
                _dependency_phydrax_meshing__association.PlcAssociationTransfer,
            )
            and plc_transfer.domain.ambient_dimension == 2
            and source.associations
            and polyhedral_construction is None
            and certification_inputs is not None
            and audit_prepared_validity is None
            and metric_prepared_embedding is None
            and mesh.storage is None
        ):
            # Planar support consumes a certified target embedding. Certify
            # the publication's validity and embedding once, under its own
            # policy and limits, for both the support proof and acceptance.
            planar_prepared = _planar_target_certificates(
                prepared,
                mesh,
                geometry,
                certification_inputs,
                lineage,
                certification_cell_regions,
                budget=budget,
                scope=scope,
            )
            audit_prepared_validity = planar_prepared.validity
            metric_prepared_embedding = planar_prepared.embedding
            planar_embedding = metric_prepared_embedding
        publication = _prepare_native_publication(
            prepared,
            phase,
            certification_inputs,
            certification_cell_regions,
            polyhedral_construction=polyhedral_construction,
            planar_embedding=planar_embedding,
            planar_prepared=planar_prepared,
            budget=budget,
            scope=scope,
        )
        certification_inputs = publication.certification_inputs
        if publication.prepared_volume is None and audit_prepared_validity is None:
            inherited_validity, inherited_embedding = _inherited_native_certificates(
                prepared,
                mesh,
                geometry,
                certification_inputs,
                budget=budget,
                scope=scope,
            )
            audit_prepared_validity = inherited_validity
            if metric_prepared_embedding is None:
                metric_prepared_embedding = inherited_embedding
        fidelity_predecessor = _source_fidelity_predecessor(source)
        certification_prepared = (
            _owner_phydrax_meshing__certification.PreparedVolumeAcceptance(
                publication.prepared_volume,
                metric_prepared_fidelity,
                fidelity_predecessor,
            )
            if publication.prepared_volume is not None
            else _owner_phydrax_meshing__certification.PreparedBoundaryAcceptance(
                metric_prepared_embedding,
                metric_prepared_fidelity,
                fidelity_predecessor,
            )
            if metric_prepared_embedding is not None
            or metric_prepared_fidelity is not None
            or fidelity_predecessor is not None
            else None
        )
        # Supplying the geometry also refuses a successor canonicalization would reorder.
        target = _dependency_phydrax_meshing__canonical.certify_cell_mesh(
            mesh,
            source.coordinate_contract,
            geometry=geometry,
            audit_policy=prepared.policy.audit_policy,
            boundary=publication.boundary,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=publication.attributes,
            associations=publication.associations,
            region_evidence=region_evidence,
            region_boundary_evidence=region_boundary_evidence,
            certification_inputs=certification_inputs,
            lineage=None if certification_inputs is None else lineage,
            certification_cell_regions=certification_cell_regions,
            certification_prepared=certification_prepared,
            audit_prepared_validity=audit_prepared_validity,
            surface_source=source.surface_source,
            budget=budget,
        )
        # Route evidence enters the result identity before carry and transition.
        target = _publish_adapted_runtime(prepared, target, scope=scope, budget=budget)
        # Lazy: the envelope owner also uses native volume generation.
        from ._surface_envelope import _carry_surface_envelope_repair

        target = _carry_surface_envelope_repair(
            source, target, prepared.policy.envelope_source, budget=budget
        )
        transition = _dependency_phydrax_meshing__lineage.CellMeshTransition(
            source.mesh.mesh_id,
            source.mesh.topology_id,
            target,
            lineage,
            kind,
            vertex_stencil=stencil,
            geometry_transition=geometry_transition,
            parents=edit.refinement,
            coarsening=_complete_coarsening_witnesses(
                source.mesh, target.mesh, lineage, edit.coarsening
            ),
            budget=budget,
        )
        transfer = _bind_native_transfer(
            source, target, phase, conservative=conservative, budget=budget
        )
        if polyhedral_storage is not None:
            storage, retained, records = polyhedral_storage
            storage.retain(
                retained,
                mesh,
                lineage,
                *((stencil,) if stencil is not None else ()),
                *patches,
                *zones,
                *labels,
                *publication.attributes,
                *((transfer,) if transfer is not None else ()),
                record_count_upper=records,
            )
        return _NativeTarget(target, transition, lineage, stencil, transfer)


# Work and scratch are refused by the root native allowance and the host
# tickets/ledger of every nested route; the publication below refuses entity,
# connectivity, retained-data and deadline excess of the target. Cavity cells
# are admitted per local edit before it builds children: bisection closure and
# uniform batches (``_bisection_kernel_host._uniform_cavity``), the mixed
# template closure (``_mixed_adaptation._execute_mixed_closure``) and periodic
# orbit refinement/coarsening (``_periodic._periodic_charge``). Every
# source-facing query reaches ``CoordinateEnclosureBudget.reserve_geometry_queries``
# on the request ticket: rational PLC/power source preparation, PLC, composed,
# surface, mapped, periodic and B-Rep association queries, implicit face
# evaluations and certification fidelity sampling. Affine carriers and retained
# rational, algebraic and periodic coordinate banks query no source.
_NESTED_ADAPTATION_LIMITS = (
    "vertices",
    "edges",
    "faces",
    "cells",
    "connectivity_entries",
    "data_bytes",
    "work_units",
    "cavity_cells",
    "geometry_queries",
    "scratch_bytes",
    "wall_seconds",
)


def _adapted_runtime_limits(
    prepared: PreparedMeshAdaptation, /, *, scope: NativeExecutionScope
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Declare the enforcement the nested native routes prove, else ``None``."""
    match prepared.policy.route:
        case MeshAdaptationRoute.NATIVE_BISECTION | MeshAdaptationRoute.NATIVE_MIXED:
            return _NESTED_ADAPTATION_LIMITS, ()
        case MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            if prepared.source.attributes or prepared.policy.distribution is not None:
                # These variants are rejected during preparation; reconstructed
                # records must not turn that rejection into an enforcement claim.
                return (), _NESTED_ADAPTATION_LIMITS
            # Cavity admission belongs to each current candidate: affine and
            # exact plane cuts, disjoint agglomerations and full regeneration.
            # Whole-result data includes the actual owner union and ended
            # receipt (adaptation_result and the final publication preflight).
            # Work/scratch remain unproved at prepare_mesh_adaptation's ended
            # receipt/rebinding, execute_mesh_adaptation's pre-scope identity
            # and post-scope hashes, and _AdaptationPublicationStorage's schema
            # and handover passes. Regeneration also reaches unmetered host
            # normalization/optimization in _polyhedral_generation and the
            # restricted wrapper. Native predicate telemetry is not a source
            # query receipt; zero-query clipping cannot prove the surrounding
            # source-proof/certification paths. Keep all three axes partial.
            live = _LIVE_POWER_WITNESSES.get(id(prepared))
            wall_proved = (
                scope.parent is None
                and prepared.policy.distribution is None
                and prepared.source.mesh.storage is None
                and live is not None
                and live[0]() is prepared
                and live[2] is not None
            )
            return (
                (
                    "vertices",
                    "edges",
                    "faces",
                    "cells",
                    "connectivity_entries",
                    "data_bytes",
                    "cavity_cells",
                    *(("wall_seconds",) if wall_proved else ()),
                    "work_units:native_and_coordinate_batches",
                    "geometry_queries:native_and_source_batches",
                    "scratch_bytes:managed_native_and_bounded_host",
                ),
                (
                    "work_units",
                    "geometry_queries",
                    "scratch_bytes",
                    *(() if wall_proved else ("wall_seconds",)),
                ),
            )
        case _:
            return None


def _publish_adapted_runtime(
    prepared: PreparedMeshAdaptation,
    target: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__result.CellMeshingResult:
    """Refuse an over-limit target and publish its route's resource guarantees.

    Routes without a declaration keep the certification runtime, which claims
    no execution limits.
    """
    declared = _adapted_runtime_limits(prepared, scope=scope)
    if declared is None:
        return target
    # Lazy: provider publication owns the shared entity and retained-data checks.
    from .providers._native_publication import _require_retained_data
    from .providers._native_route_checks import published_entity_limits

    limits = prepared.policy.limits
    published_entity_limits(target.mesh, limits)
    _require_retained_data(target, limits)
    remaining = scope.remaining().remaining_seconds
    if remaining <= 0.0:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.TIMED_OUT,
            "MeshAdaptation.execution.elapsed_seconds must respect limits.maximum_wall_seconds; got an exhausted deadline before publication",
            stage="adaptation-publication",
            requested=(("maximum_wall_seconds", limits.maximum_wall_seconds),),
            achieved=(("remaining_seconds", remaining),),
        )
    enforced, unenforced = declared
    runtime = target.runtime
    published = _dependency_phydrax_meshing__result.MeshingRuntimeInfo(
        runtime.provider_id,
        runtime.actual_version,
        runtime.execution_mode,
        deterministic=runtime.deterministic,
        enforced_limits=(*runtime.enforced_limits, *enforced),
        unenforced_limits=(*runtime.unenforced_limits, *unenforced),
    )
    return _dependency_phydrax_meshing__result.CellMeshingResult(
        target.mesh,
        target.geometry,
        target.coordinate_contract,
        target.audit,
        target.quality,
        target.compliance,
        target.trace,
        target.provider,
        published,
        target.derivative_mode,
        target.provenance,
        boundary=target.boundary,
        patches=target.patches,
        zones=target.zones,
        labels=target.labels,
        attributes=target.attributes,
        associations=target.associations,
        adapter_reports=target.adapter_reports,
        certification=target.certification,
        region_evidence=target.region_evidence,
        region_boundary_evidence=target.region_boundary_evidence,
        surface_source=target.surface_source,
        collective_evidence=target.collective_evidence,
        storage_binding=target.storage_binding,
        collective_certificates=target.collective_certificates,
        execution_evidence=target.execution_evidence,
        budget=budget,
    )


def _run_bisection_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    if prepared.source.mesh.periodic_topology is not None:
        return _owner_phydrax_meshing__periodic.execute_periodic_bisection_route(
            prepared, scope=scope, budget=budget
        )
    request = prepared.request
    match request:
        case MarkedMeshAdaptation(
            hierarchy=(BisectionHierarchy() | None) as bisection_hierarchy
        ):
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide marked bisection adaptation; got {type(request).__name__!r}"
            )
    policy = prepared.policy
    constraints = prepared.constraints
    outcome = _dependency_phydrax_meshing__bisection.execute_bisection(
        prepared.source.mesh,
        np.asarray(request.refine_cell_ids, dtype=np.int64),
        np.asarray(request.coarsen_cell_ids, dtype=np.int64),
        source_result=prepared.source,
        hierarchy=bisection_hierarchy,
        compatibility=policy.compatibility,
        refinement_generations=request.refinement_generations,
        refinement_target_cells=request.refinement_target_cells,
        protected_edges=constraints.protected_edge_keys,
        protected_vertices=constraints.protected_vertex_ids,
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        maximum_closure_iterations=policy.maximum_closure_iterations,
        maximum_cavity_cells=policy.limits.maximum_cavity_cells,
        maximum_cells=policy.limits.maximum_cells,
        maximum_vertices=policy.limits.maximum_vertices,
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_scratch_bytes=policy.limits.maximum_scratch_bytes,
        maximum_wall_seconds=policy.limits.maximum_wall_seconds,
        maximum_geometry_queries=policy.limits.maximum_geometry_queries,
        scope=scope,
        budget=budget,
    )
    evidence = outcome.evidence
    refined = evidence.bisections > 0 or evidence.uniform_refinement_applied
    coarsened = evidence.coarsened_vertices > 0
    partial = (
        np.asarray(evidence.rejected_refinement_ids).size > 0
        or np.asarray(evidence.rejected_coarsening_ids).size > 0
    )
    if not refined and not coarsened:
        return unchanged(
            prepared,
            evidence,
            outcome.hierarchy,
            status=MeshAdaptationStatus.PARTIAL
            if partial
            else MeshAdaptationStatus.UNCHANGED,
        )
    match (refined, coarsened):
        case (True, False):
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REFINE
        case (False, True):
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.COARSEN
        case _:
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH
    native = finalize_native(
        prepared,
        outcome.edit,
        kind,
        conservative=not coarsened,
        bisection_hierarchy=outcome.hierarchy,
        scope=scope,
        budget=budget,
    )

    hierarchy = bisection_kernel_owner.rebind_bisection_presentation_blocks(
        outcome.hierarchy, native.target, scope=scope, budget=budget
    )
    return RouteOutcome(
        MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        evidence,
        hierarchy,
    )


@checked
def target_metric(
    metric: _dependency_phydrax_meshing__metric.MeshMetricField,
    target: CellMesh,
    values: np.ndarray,
    /,
) -> _dependency_phydrax_meshing__metric.MeshMetricField:
    vertices = target.entity_set(0)
    identifiers = np.asarray(target.vertex_global_ids, dtype=np.int64)
    order = np.argsort(identifiers, kind="stable")
    return _dependency_phydrax_meshing__metric.MeshMetricField(
        _dependency_phydrax_meshing__scope.MeshingScope(
            target.mesh_id,
            target.numeric_version,
            _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            identifiers,
        ),
        np.asarray(values, dtype=np.float64)[order],
        minimum_size=metric.minimum_size,
        maximum_size=metric.maximum_size,
        maximum_anisotropy=metric.maximum_anisotropy,
    )


@checked
def metric_status(
    evidence: LocalMetricEvidence | DeviceMetricEvidence, topology: bool, /
) -> MeshAdaptationStatus:
    if not topology:
        # Relocation reaches its fixed point when a pass moves nothing.
        return (
            MeshAdaptationStatus.COMPLETE
            if evidence.converged or evidence.stalled
            else MeshAdaptationStatus.PASS_LIMIT
        )
    if evidence.converged:
        return MeshAdaptationStatus.COMPLETE
    if evidence.stalled:
        return MeshAdaptationStatus.STALLED
    return MeshAdaptationStatus.PASS_LIMIT


def _execute_metric_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    with adaptation_phase(scope, budget):
        request = prepared.request
        match request:
            case MetricMeshAdaptation() | RelocationMeshAdaptation():
                pass
            case _:
                raise TypeError(
                    f"MeshAdaptation.request must provide a metric or relocation request; got {type(request).__name__!r}"
                )
        policy = prepared.policy
        constraints = prepared.constraints
        topology = isinstance(request, MetricMeshAdaptation)
        outcome = _dependency_phydrax_meshing__local_metric.execute_local_metric_adaptation(
            prepared.source.mesh,
            constraints.metric_values,
            cell_classes=constraints.cell_classes,
            edge_classes=constraints.edge_classes,
            protected_edges=constraints.protected_edge_mask,
            fixed_vertices=constraints.fixed_vertex_mask,
            predicate_mode=_dependency_phydrax__geometry_predicates.resolve_host_predicate_mode(
                policy.predicate_mode
            ),
            maximum_passes=policy.maximum_passes,
            topology_operations=topology,
            relocation=policy.relocation,
            budget=budget,
        )
        evidence = outcome.evidence
        status = metric_status(evidence, topology)
        applied = (
            evidence.splits + evidence.collapses + evidence.flips + evidence.relocations
        )
        if applied == 0:
            # Preserve the source result while retaining whether the request was
            # already satisfied, stalled, or exhausted its pass budget.
            return unchanged(prepared, evidence, None, status=status)
        native = finalize_native(
            prepared,
            outcome.edit,
            _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH,
            conservative=False,
            scope=scope,
            budget=budget,
        )
        return RouteOutcome(
            status,
            native.target,
            native.transition,
            native.lineage,
            native.stencil,
            native.transfer,
            target_metric(request.metric, native.target.mesh, outcome.metric),
            evidence,
            None,
        )


def _implicit_curved_boundary(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    policy: MeshAdaptationPolicy,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__tetra_metric.CurvedBoundarySource | None:
    """The certified analytic zero set as the explicit smooth tetra boundary source.

    Only the validated implicit association declares smooth faces; its
    original profile, fidelity tolerance and certificate limits bind every
    whole-facet proof.
    """
    transfer = policy.association_transfer
    if not isinstance(
        transfer,
        _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer,
    ):
        return None
    association, _ = transfer.source_associations(source, budget=budget)
    certification = source.certification
    if certification is None or certification.request.fidelity_tolerance is None:
        raise ValueError(
            f"MeshAdaptation.source.certification must retain implicit source fidelity; got {source.result_id!r}"
        )
    mesh = source.mesh
    facets = np.isin(
        np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64),
        np.asarray(association.target_global_ids, dtype=np.int64),
    )
    return _dependency_phydrax_meshing__tetra_metric.CurvedBoundarySource(
        transfer.profile,
        certification.request.fidelity_tolerance,
        certification.request.limits,
        facets,
        budget=budget,
    )


def _execute_tetra_metric_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    request = prepared.request
    match request:
        case MetricMeshAdaptation() | RelocationMeshAdaptation():
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide a metric or relocation request; got {type(request).__name__!r}"
            )
    policy = prepared.policy
    constraints = prepared.constraints
    with adaptation_phase(scope, budget):
        curved_boundary = _implicit_curved_boundary(
            prepared.source, policy, budget=budget
        )
    outcome = _dependency_phydrax_meshing__tetra_metric.execute_tetra_metric_adaptation(
        prepared.source.mesh,
        constraints.metric_values,
        source_geometry=prepared.source.geometry,
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        edge_classes=constraints.edge_classes,
        protected_edges=constraints.protected_edge_mask,
        fixed_vertices=constraints.fixed_vertex_mask,
        maximum_passes=policy.maximum_passes,
        topology_operations=isinstance(request, MetricMeshAdaptation),
        relocation=policy.relocation,
        maximum_vertices=policy.limits.maximum_vertices,
        maximum_cells=policy.limits.maximum_cells,
        maximum_operations=policy.limits.maximum_work_units,
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_location_pairs=policy.limits.maximum_geometry_queries,
        maximum_cavity_cells=policy.limits.maximum_cavity_cells,
        maximum_scratch_bytes=policy.limits.maximum_scratch_bytes,
        curved_boundary=curved_boundary,
        scope=scope,
        budget=budget,
    )
    evidence = outcome.evidence
    match evidence.status:
        case _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE:
            status = MeshAdaptationStatus.COMPLETE
        case _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.STALLED:
            status = MeshAdaptationStatus.STALLED
        case _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.PASS_LIMIT:
            status = MeshAdaptationStatus.PASS_LIMIT
        case (
            _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        ):
            status = MeshAdaptationStatus.RESOURCE_LIMIT
        case invalid:
            assert_never(invalid)
    applied = evidence.splits + evidence.collapses + evidence.flips + evidence.relocations
    if applied == 0:
        return unchanged(prepared, evidence, None, status=status)
    native = finalize_native(
        prepared,
        outcome.edit,
        _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH,
        conservative=False,
        plc_source=outcome.exact_source,
        scope=scope,
        budget=budget,
    )
    return RouteOutcome(
        status,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        target_metric(request.metric, native.target.mesh, outcome.metric),
        evidence,
        None,
    )


def _surface_fidelity_limit(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> float:
    certificate = source.certification
    if certificate is None or certificate.request.fidelity_tolerance is None:
        raise ValueError(
            f"MeshAdaptation.source.certification must retain continuous source fidelity; got {source.result_id!r}"
        )
    return certificate.request.fidelity_tolerance


def _surface_material_epoch(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    witness: SurfaceChartWitness,
    /,
) -> tuple[NDArray[np.object_] | None, SurfaceChartWitness]:
    """Retain material anchors and the admitted source chart provenance."""

    retained = (
        source.certification.request.source if source.certification is not None else None
    )
    if isinstance(
        retained,
        surface_boundary_sources_owner.SurfaceChartBoundarySource,
    ) and isinstance(
        retained.deformation,
        _owner_phydrax_discretization__surface_chart_deformation.PreparedSurfaceChartDeformation,
    ):
        retained.require_current()
        retained.deformation.require_bound(
            retained.deformation.source_mesh,
            retained.deformation.source_geometry,
            source.mesh,
            source.geometry,
        )
        return (
            retained.deformation.material_charts(target=True),
            retained.deformation.material_root_witness,
        )
    return None, witness


@final
class _SurfaceCandidate(StrictModule):
    """Metric candidate and the curve evidence needed for target reconstruction."""

    outcome: (
        _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome
        | _dependency_phydrax_meshing__surface_metric.SphereMetricOutcome
    )
    curve_witness: (
        _owner_phydrax_meshing__surface_association_transfer.PreparedSurfaceCurveWitness
        | None
    )


def _run_sphere_metric_candidate(
    prepared: PreparedMeshAdaptation,
    request: MetricMeshAdaptation | RelocationMeshAdaptation,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _SurfaceCandidate:
    source, policy, constraints = prepared.source, prepared.policy, prepared.constraints
    domain = transfer.support.domain
    retained = (
        source.certification.request.source if source.certification is not None else None
    )
    if isinstance(
        retained,
        surface_boundary_sources_owner.SurfaceChartBoundarySource,
    ) and isinstance(
        retained.deformation,
        _owner_phydrax_discretization__sphere_chart_deformation.PreparedSphereChartDeformation,
    ):
        retained.require_current()
        atlas = retained.deformation.target_atlas
    else:
        atlas = _owner_phydrax_geometry__sphere_material_atlas.sphere_material_atlas_from_result(
            domain,
            source,
            maximum_fidelity=_surface_fidelity_limit(source),
            maximum_cells=policy.limits.maximum_cells,
            maximum_work_units=policy.limits.maximum_work_units,
            maximum_coordinate_memory_bytes=policy.limits.maximum_scratch_bytes,
            budget=budget,
        )
    produced = _owner_phydrax_meshing__surface_metric.execute_sphere_metric_adaptation(
        source,
        constraints.metric_values,
        domain,
        atlas,
        transfer,
        policy=policy,
        coordinate_contract=source.coordinate_contract,
        maximum_fidelity=_surface_fidelity_limit(source),
        topology_operations=isinstance(request, MetricMeshAdaptation),
        relocation=policy.relocation,
        budget=budget,
        scope=scope,
    )
    return _SurfaceCandidate(produced, None)


def _run_surface_metric_candidate(
    prepared: PreparedMeshAdaptation,
    request: MetricMeshAdaptation | RelocationMeshAdaptation,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _SurfaceCandidate:
    source, policy, constraints = prepared.source, prepared.policy, prepared.constraints
    domain = transfer.support.domain
    witness = _owner_phydrax_meshing__surface_metric.prepare_surface_metric_source(
        source, transfer, budget=budget
    )
    material_charts, material_root_witness = _surface_material_epoch(source, witness)
    curve_witness = _owner_phydrax_meshing__surface_association_transfer.prepare_surface_curve_witness(
        transfer.support, source, budget=budget
    )
    controller = (
        _owner_phydrax_meshing__surface_metric_controller.SurfaceMetricController(
            source,
            transfer,
            policy,
            budget,
            witness,
            _surface_fidelity_limit(source),
            material_root_witness,
            scope=scope,
        )
        if source.mesh.periodic_topology is not None
        else None
    )
    produced = _owner_phydrax_meshing__surface_metric.execute_surface_metric_adaptation(
        source.mesh,
        constraints.metric_values,
        domain,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        cell_patches=witness.patches,
        cell_charts=witness.charts,
        cell_classes=constraints.cell_classes,
        edge_classes=constraints.edge_classes,
        protected_edges=constraints.protected_edge_mask,
        fixed_vertices=constraints.fixed_vertex_mask,
        maximum_fidelity=_surface_fidelity_limit(source),
        maximum_passes=policy.maximum_passes,
        topology_operations=isinstance(request, MetricMeshAdaptation),
        relocation=policy.relocation,
        maximum_vertices=policy.limits.maximum_vertices,
        maximum_cells=policy.limits.maximum_cells,
        maximum_operations=policy.limits.maximum_work_units,
        maximum_arc_work_units=policy.limits.maximum_work_units,
        maximum_location_pairs=policy.limits.maximum_geometry_queries,
        controller=controller,
        source_material_charts=material_charts,
        curve_witness=curve_witness,
        scope=scope,
        budget=budget,
    )
    return _SurfaceCandidate(produced, curve_witness)


def _execute_surface_metric_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    with adaptation_phase(scope, budget):
        request, policy = prepared.request, prepared.policy
        transfer = policy.association_transfer
        match (request, transfer):
            case (
                MetricMeshAdaptation() | RelocationMeshAdaptation(),
                _dependency_phydrax_meshing__association.SurfaceAssociationTransfer(),
            ):
                pass
            case _:
                raise TypeError(
                    f"MeshAdaptation.request must provide an admitted metric request and surface source support; got {(type(request).__name__, type(transfer).__name__)!r}"
                )
        source = prepared.source
        domain = transfer.support.domain
        patches = np.unique(transfer.classes(source, budget=budget)[2].indices)
        sphere = True
        for patch in patches:
            surface = domain.patches[int(patch)].surface
            operation = (
                surface.definition
                if isinstance(surface, _owner_phydrax_geometry_brep__placed.PlacedSurface)
                else surface
            )
            if (
                _owner_phydrax_geometry_brep__patches.sphere_source_radius_terms(
                    operation, budget=budget
                )
                is None
            ):
                sphere = False
                break
        try:
            with budget.measurement(), budget.activate():
                if sphere:
                    candidate = _run_sphere_metric_candidate(
                        prepared, request, transfer, budget=budget, scope=scope
                    )
                else:
                    candidate = _run_surface_metric_candidate(
                        prepared, request, transfer, budget=budget, scope=scope
                    )
                produced, curve_witness = candidate.outcome, candidate.curve_witness
                evidence = produced.evidence
                status = MeshAdaptationStatus(evidence.status.value)
                if (
                    evidence.status
                    is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
                    or evidence.splits
                    + evidence.collapses
                    + evidence.flips
                    + evidence.relocations
                    == 0
                ):
                    outcome = unchanged(prepared, evidence, None, status=status)
                else:
                    match produced:
                        case _dependency_phydrax_meshing__surface_metric.SurfaceMetricOutcome():
                            target_edit = produced.geometry.edit
                        case _dependency_phydrax_meshing__surface_metric.SphereMetricOutcome():
                            target_edit = produced.edit
                        case invalid:
                            assert_never(invalid)
                    native = finalize_native(
                        prepared,
                        target_edit,
                        _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH,
                        conservative=False,
                        surface_metric_outcome=produced,
                        surface_curve_witness=curve_witness,
                        scope=scope,
                        budget=budget,
                    )
                    outcome = RouteOutcome(
                        status,
                        native.target,
                        native.transition,
                        native.lineage,
                        native.stencil,
                        native.transfer,
                        target_metric(
                            request.metric, native.target.mesh, produced.metric
                        ),
                        evidence,
                        None,
                    )
            return outcome
        except (
            _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
            _dependency_phydrax_discretization__surface_chart_deformation.SurfaceChartResourceError,
        ) as error:
            return unchanged(
                prepared,
                _surface_resource_refusal(prepared, budget, error),
                None,
                status=MeshAdaptationStatus.RESOURCE_LIMIT,
            )


_NATIVE_RESOURCE_STATUSES = (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT)


def _surface_resource_refusal(
    prepared: PreparedMeshAdaptation,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    error: MeshcoreError
    | _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError
    | _dependency_phydrax_discretization__surface_chart_deformation.SurfaceChartResourceError,
    /,
) -> MetricRemeshingEvidence:
    """Bounded RESOURCE_LIMIT evidence; the accepted source remains the target."""
    certification = prepared.source.certification
    if certification is None or certification.fidelity is None:
        raise ValueError(
            f"MeshAdaptation.source.certification must retain continuous fidelity for the accepted source; got {prepared.source.result_id!r}"
        )
    limits = prepared.policy.limits
    requested = (
        ("maximum_work_units", float(limits.maximum_work_units)),
        ("maximum_geometry_queries", float(limits.maximum_geometry_queries)),
        ("maximum_scratch_bytes", float(limits.maximum_scratch_bytes)),
    )
    achieved = (
        ("coordinate_work_units", float(budget.work_units)),
        ("native_charged_coordinate_work_units", float(budget.native_charged_work_units)),
        ("coordinate_peak_bytes_upper", float(budget.peak_bytes_upper)),
    )
    if isinstance(error, MeshcoreError):
        if error.batch_requested_upper is not None:
            requested += tuple(
                (f"native_batch:requested_{name}", float(value))
                for name, value in zip(
                    ("work_units", "geometry_queries", "cavity_cells", "managed_bytes"),
                    error.batch_requested_upper,
                    strict=True,
                )
            )
        if error.batch_refusal_evidence is not None:
            achieved += tuple(
                (f"native_batch:refusal:{index}", float(value))
                for index, value in enumerate(error.batch_refusal_evidence)
            )
    if isinstance(
        error,
        _dependency_phydrax_discretization__surface_chart_deformation.SurfaceChartResourceError,
    ):
        requested += tuple(
            (name, float(limit)) for name, _, limit in error.resource_counts
        )
        achieved += tuple(
            (name, float(completed)) for name, completed, _ in error.resource_counts
        )
    return MetricRemeshingEvidence(
        _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT,
        passes=0,
        counts=(0, 0, 0, 0),
        rejected_operations=0,
        work_units=budget.work_units,
        lengths=np.zeros((0,), dtype=np.float64),
        quality=np.zeros((0,), dtype=np.float64),
        maximum_fidelity_bound=certification.fidelity.mesh_to_source_upper,
        criterion=(
            _dependency_phydrax_meshing__tetra_metric.MetricRemeshingCriterion.UNIT_MESH
            if isinstance(prepared.request, MetricMeshAdaptation)
            else _dependency_phydrax_meshing__tetra_metric.MetricRemeshingCriterion.RELOCATION_FIXED_POINT
        ),
        native_work_units=0,
        resource_message=str(error),
        resource_requested=requested,
        resource_achieved=achieved,
    )


def _prepare_mixed_route(
    prepared: PreparedMeshAdaptation, /, *, scope: NativeExecutionScope
) -> _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationOutcome:
    request = prepared.request
    match request:
        case MarkedMeshAdaptation(
            hierarchy=(
                _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationHierarchy()
                | None
            ) as mixed_hierarchy
        ):
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide marked mixed-template adaptation with its mixed hierarchy; got {type(request).__name__!r}"
            )
    source = prepared.source.mesh
    rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(source, source.topological_dimension),
        mesh_cell_ids(source)[:, None],
    )
    cell_ids = np.asarray(
        source.entity_set(source.topological_dimension).entity_ids, dtype=np.int64
    )
    protected_cells = [
        cell_ids[selected_rows(source, scope)]
        for scope in prepared.policy.protected_scopes
        if scope.entity_dimension == source.topological_dimension
    ]
    protected_cell_ids = (
        np.unique(np.concatenate(protected_cells))
        if protected_cells
        else np.zeros((0,), dtype=np.int64)
    )
    protected_face_keys = np.zeros((0, 0), dtype=np.int64)
    if source.topological_dimension == 3:
        face_keys = entity_keys_owner.entity_keys(source, 2)
        protected_faces = np.zeros((face_keys.shape[0],), dtype=np.bool_)
        for protected_scope in prepared.policy.protected_scopes:
            if protected_scope.entity_dimension >= 2:
                face_rows = closure_rows(
                    source,
                    protected_scope.entity_dimension,
                    selected_rows(source, protected_scope),
                    2,
                )
                protected_faces[face_rows] = True
        protected_face_keys = face_keys[protected_faces]
    outcome = _dependency_phydrax_meshing__mixed_adaptation.adapt_mixed_mesh(
        source,
        refine_cell_ids=request.refine_cell_ids,
        coarsen_cell_ids=request.coarsen_cell_ids,
        hierarchy=mixed_hierarchy,
        cell_classes=prepared.constraints.cell_classes[rows],
        protected_vertex_ids=prepared.constraints.protected_vertex_ids,
        protected_cell_ids=protected_cell_ids,
        protected_edge_keys=prepared.constraints.protected_edge_keys,
        protected_face_keys=protected_face_keys,
        layer_columns=request.layer_columns,
        maximum_cells=prepared.policy.limits.maximum_cells,
        maximum_closure_steps=prepared.policy.maximum_closure_iterations,
        maximum_work_units=prepared.policy.limits.maximum_work_units,
        maximum_scratch_bytes=prepared.policy.limits.maximum_scratch_bytes,
        maximum_cavity_cells=prepared.policy.limits.maximum_cavity_cells,
        maximum_geometry_queries=prepared.policy.limits.maximum_geometry_queries,
        maximum_wall_seconds=prepared.policy.limits.maximum_wall_seconds,
    )
    return outcome


def _finalize_mixed_route(
    prepared: PreparedMeshAdaptation,
    outcome: _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationOutcome,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    evidence = outcome.evidence
    partial = evidence.rejected_coarsening_ids.size > 0
    refined = evidence.refined_cell_ids.size > 0
    coarsened = evidence.coarsened_cell_ids.size > 0
    if not refined and not coarsened:
        return unchanged(
            prepared,
            evidence,
            outcome.hierarchy,
            status=MeshAdaptationStatus.PARTIAL
            if partial
            else MeshAdaptationStatus.UNCHANGED,
        )
    match (refined, coarsened):
        case (True, False):
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REFINE
        case (False, True):
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.COARSEN
        case _:
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH
    native = finalize_native(
        prepared, outcome.edit, kind, conservative=False, scope=scope, budget=budget
    )
    return RouteOutcome(
        MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        evidence,
        outcome.hierarchy,
    )




def _polyhedral_classification_bounds(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    policy: MeshAdaptationPolicy,
    /,
) -> tuple[int, int]:
    """Host key, closure and classification banks, excluding source proofs.

    Work units count scalar visits in the declared host passes and sorting
    levels. Native predicates and association-source proofs charge separately.
    Incidence banks, not padded cell vertices, bound closure traversal.
    """
    mesh = source.mesh
    vertices = mesh.coordinates.shape[0]
    entities = sum(mesh.entity_set(d).entity_ids.size for d in range(4))
    corners = sum(block.vertices.size for block in mesh.blocks)
    width = max((block.vertices.shape[1] for block in mesh.blocks), default=2)
    scopes = len(policy.protected_scopes)
    scopes += len(source.patches) + len(source.zones) + len(source.labels)
    incidence = sum(
        value.relation.source_indices.size for value in mesh.topology.incidences
    )
    identifiers = sum(value.global_entity_ids.size for value in policy.protected_scopes)
    identifiers += sum(
        value.global_entity_ids.size for value in source.patches.scopes
    )
    identifiers += sum(value.scope.global_entity_ids.size for value in source.zones)
    identifiers += sum(value.scope.global_entity_ids.size for value in source.labels)
    # key_rows/entity_keys sorts, mesh_cell_ids concatenation, boolean isin
    # lookup/sorts and membership/closure rows. At most sixteen scalar banks
    # of each concrete padded key/closure, plus dictionaries/sets per row.
    slots = (
        entities * width + corners + vertices + entities * (scopes + 1)
        + incidence * (scopes + 1) + identifiers
    )
    required = 4096 + 8 * 16 * slots + 256 * (entities + scopes + 16)
    work = slots * max(1, (entities + identifiers).bit_length())
    return required, work


def _execute_polyhedral_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    with _dependency_phydrax_meshing__polyhedral_generation._PolyhedralTopologyStorage(
        scope
    ) as storage:
        # This root-created token is never resized within a child target.
        required, work = _polyhedral_classification_bounds(
            prepared.source, prepared.policy
        )
        storage.workspace.set_bound(required)
        scope.charge(work=work)
        return _execute_admitted_polyhedral_route(prepared, budget=budget, scope=scope)


def _execute_admitted_polyhedral_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    request = prepared.request
    match request:
        case _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralMeshAdaptation():
            pass
        case _:
            raise TypeError(
                f"MeshAdaptation.request must provide PolyhedralMeshAdaptation for NATIVE_POLYHEDRAL; got {type(request).__name__!r}"
            )
    source = prepared.source.mesh
    rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(source, 3),
        mesh_cell_ids(source)[:, None],
    )
    face_keys = entity_keys_owner.entity_keys(source, 2)
    if (
        request.operation
        is _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationOperation.PLANE_SPLIT
    ):
        # A plane cut keeps every existing vertex in place and its new rows are
        # re-earned by the association transfer, so only declared protection
        # forbids subdividing a face; source-fixed corners stay fixed.
        declared = np.zeros((source.coordinates.shape[0],), dtype=np.bool_)
        for protected_scope in prepared.policy.protected_scopes:
            declared[
                closure_rows(
                    source,
                    protected_scope.entity_dimension,
                    selected_rows(source, protected_scope),
                    0,
                )
            ] = True
        protected_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)[
            declared
        ]
    else:
        protected_vertex_ids = prepared.constraints.protected_vertex_ids
    protected_faces = np.any(np.isin(face_keys, protected_vertex_ids), axis=1)
    if (
        request.operation
        is _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationOperation.AGGLOMERATE
    ):
        protected_faces |= np.any(
            membership(source, organization_scopes(prepared.source), 2), axis=1
        )
    limits = prepared.policy.limits
    outcome = _dependency_phydrax_meshing__polyhedral_adaptation.adapt_polyhedral_mesh(
        source,
        request,
        cell_classes=prepared.constraints.cell_classes[rows],
        source_geometry=prepared.source.geometry,
        protected_face_ids=np.asarray(source.entity_set(2).entity_ids)[protected_faces],
        maximum_cells=limits.maximum_cells,
        maximum_vertices=limits.maximum_vertices,
        maximum_work_units=limits.maximum_work_units,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
        limits=limits,
        source_id=(
            prepared.source.associations[0].source_id
            if prepared.source.associations
            else "native-polyhedral"
        ),
        numeric_version=f"adaptation:{prepared.prepared_id}",
        common_refinement_policy=_dependency_phydrax_geometry__supermesh.CommonRefinementPolicy(
            predicate_mode=_dependency_phydrax__geometry_predicates.resolve_host_predicate_mode(
                prepared.policy.predicate_mode
            ),
            maximum_candidate_pairs=limits.maximum_geometry_queries,
            maximum_accepted_pairs=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
            maximum_exact_work=limits.maximum_work_units,
            overlap_simplices=True,
        ),
        scope=scope,
        budget=budget,
    )
    edit = outcome.edit
    if outcome.construction is not None and prepared.source.associations:
        with (
            adaptation_phase(scope, budget),
            _polyhedral_publication_storage(prepared, edit, scope) as candidate_storage,
        ):
            candidate, _, _ = (
                _dependency_phydrax_meshing__topology_edit.assemble_topology_edit(
                    source,
                    edit,
                    numeric_version=f"adaptation:{prepared.prepared_id}",
                    budget=budget,
                )
            )
            associations = _dependency_phydrax_meshing__polyhedral_adaptation.regenerated_polyhedral_associations(
                prepared.source,
                request,
                outcome.construction,
                candidate,
                budget=budget,
            )
            edit = _dependency_phydrax_meshing__polyhedral_adaptation.regenerated_polyhedral_relations(
                prepared.source,
                candidate,
                edit,
                associations,
                limits.maximum_work_units,
                maximum_geometry_queries=limits.maximum_geometry_queries,
                target_geometry=outcome.geometry,
                scope=scope,
                budget=budget,
            )
            if candidate_storage is not None:
                storage, retained, records = candidate_storage
                storage.retain(retained, candidate, record_count_upper=records)
    match request.operation:
        case _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationOperation.PLANE_SPLIT:
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REFINE
        case _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationOperation.AGGLOMERATE:
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.COARSEN
        case _dependency_phydrax_meshing__polyhedral_adaptation.PolyhedralAdaptationOperation.REGENERATE:
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH
        case invalid:
            assert_never(invalid)
    native = finalize_native(
        prepared,
        edit,
        kind,
        conservative=False,
        common_refinement=outcome.common_refinement,
        polyhedral_construction=outcome.construction,
        polyhedral_geometry=outcome.geometry,
        scope=scope,
        budget=budget,
    )
    return RouteOutcome(
        MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        outcome.evidence,
        None,
        outcome.common_refinement,
    )


def _unknown_lineage(
    source: CellMesh, target: CellMesh, /
) -> _dependency_phydrax_meshing__lineage_records.MeshLineage:
    """Provider remeshing: every target entity created, every source entity deleted."""

    records = []
    for dimension in range(source.topological_dimension + 1):
        source_entities = source.entity_set(dimension)
        target_entities = target.entity_set(dimension)
        empty = np.zeros((0,), dtype=np.int64)
        records.append(
            _dependency_phydrax_meshing__lineage_records.EntityLineage(
                dimension,
                source_entities.entity_set_id,
                target_entities.entity_set_id,
                empty,
                empty,
                np.zeros((0,), dtype=np.int32),
                created_target_ids=np.sort(np.asarray(target_entities.entity_ids)),
                deleted_source_ids=np.sort(np.asarray(source_entities.entity_ids)),
            )
        )
    return _dependency_phydrax_meshing__lineage_records.MeshLineage(
        source.topology_id, target.topology_id, tuple(records)
    )


def _provider_outcome(
    prepared: PreparedMeshAdaptation,
    target: _dependency_phydrax_meshing__result.CellMeshingResult,
    metric: _dependency_phydrax_meshing__metric.MeshMetricField | None,
    evidence: MmgAdaptationResult | OmegaHAdaptationResult,
    /,
) -> RouteOutcome:
    source = prepared.source
    lineage = _unknown_lineage(source.mesh, target.mesh)
    transition = _dependency_phydrax_meshing__lineage.CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH,
    )
    return RouteOutcome(
        MeshAdaptationStatus.COMPLETE,
        target,
        transition,
        lineage,
        None,
        None,
        metric,
        evidence,
        None,
    )


def _execute_mmg_route(prepared: PreparedMeshAdaptation, /) -> RouteOutcome:
    policy = prepared.policy
    transfer = policy.association_transfer
    if isinstance(
        transfer,
        (
            _dependency_phydrax_meshing__association.PlcAssociationTransfer,
            _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer,
        ),
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider rederivation of represented PLC associations; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer, _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider rederivation of parametric surface associations; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer,
        _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider rederivation of mapped-root associations; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer,
        _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider replacement of implicit source provenance; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer,
        _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider replacement of periodic source provenance; got {type(transfer).__name__!r}"
        )
    provider = policy.provider
    # ty: ignore[unresolved-attribute]
    plan = provider.plan(
        prepared.source,
        # ty: ignore[unresolved-attribute]
        metric=prepared.request.metric,
        required=policy.protected_scopes,
        # ty: ignore[invalid-argument-type]
        options=policy.provider_options,
        limits=policy.limits,
        audit_policy=policy.audit_policy,
        association_transfer=transfer,
    )
    # ty: ignore[invalid-argument-type, missing-argument, unresolved-attribute]
    result = provider.execute(plan)
    # ty: ignore[unresolved-attribute]
    return _provider_outcome(prepared, result.mesh, result.metric, result)


def _execute_omega_h_route(prepared: PreparedMeshAdaptation, /) -> RouteOutcome:
    policy = prepared.policy
    transfer = policy.association_transfer
    if isinstance(
        transfer,
        (
            _dependency_phydrax_meshing__association.PlcAssociationTransfer,
            _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer,
        ),
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider rederivation of represented PLC associations; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer, _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider rederivation of parametric surface associations; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer,
        _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider rederivation of mapped-root associations; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer,
        _dependency_phydrax_meshing__implicit_association_transfer.ImplicitAssociationTransfer,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider replacement of implicit source provenance; got {type(transfer).__name__!r}"
        )
    if isinstance(
        transfer,
        _dependency_phydrax_meshing_providers__native_periodic.PeriodicAssociationTransfer,
    ):
        raise ValueError(
            f"MeshAdaptationPolicy.association_transfer must be absent for provider replacement of periodic source provenance; got {type(transfer).__name__!r}"
        )
    # ty: ignore[unresolved-attribute]
    result = policy.provider.execute(
        # ty: ignore[invalid-argument-type]
        prepared.source,
        # ty: ignore[too-many-positional-arguments, unresolved-attribute]
        prepared.request.metric,
        # ty: ignore[invalid-argument-type, unknown-argument]
        options=policy.provider_options,
        # ty: ignore[unknown-argument]
        limits=policy.limits,
        # ty: ignore[unknown-argument]
        association_transfer=transfer,
    )
    # ty: ignore[unresolved-attribute]
    if result.target is None:
        raise ValueError(
            f"OmegaHAdaptationResult.target must contain the serial adapted mesh; got {type(result).__name__!r}"
        )
    # ty: ignore[unresolved-attribute]
    return _provider_outcome(prepared, result.target, result.metric, result)


@checked
def project_hp_lineage(
    source: FiniteElementHPEpoch,
    target: FiniteElementHPEpoch,
    lineage: FiniteElementHPLineage,
    /,
) -> _dependency_phydrax_meshing__lineage_records.MeshLineage:
    """Project fixed-capacity hp slot lineage onto the epochs' active cell meshes."""

    if (
        lineage.source_topology_id != source.topology.topology_id
        or lineage.target_topology_id != target.topology.topology_id
    ):
        raise ValueError(
            f"FiniteElementHPLineage.endpoints must match the supplied epochs; got {(lineage.source_topology_id, lineage.target_topology_id)!r}"
        )
    valid = np.asarray(lineage.valid, dtype=np.bool_)
    source_slots = np.asarray(lineage.source_slots, dtype=np.int32)[valid]
    target_slots = np.asarray(lineage.target_slots, dtype=np.int32)[valid]
    source_ids = np.asarray(source.topology.cell_global_ids, dtype=np.int64)[source_slots]
    target_ids = np.asarray(target.topology.cell_global_ids, dtype=np.int64)[target_slots]
    kinds = np.full(
        source_ids.shape,
        int(_dependency_phydrax_meshing__lineage_records.EntityLineageKind.PRESERVED),
        dtype=np.int32,
    )
    refinement = np.asarray(lineage.relation_mask("refinement"), dtype=np.bool_)[valid]
    coarsening = np.asarray(lineage.relation_mask("coarsening"), dtype=np.bool_)[valid]
    kinds[refinement] = int(
        _dependency_phydrax_meshing__lineage_records.EntityLineageKind.REFINED_FROM
    )
    kinds[coarsening] = int(
        _dependency_phydrax_meshing__lineage_records.EntityLineageKind.COARSENED_INTO
    )
    dimension = source.mesh.topological_dimension
    entities = _dependency_phydrax_meshing__lineage_records.EntityLineage(
        dimension,
        source.mesh.entity_set(dimension).entity_set_id,
        target.mesh.entity_set(dimension).entity_set_id,
        source_ids,
        target_ids,
        kinds,
    )
    return _dependency_phydrax_meshing__lineage_records.MeshLineage(
        source.mesh.topology_id, target.mesh.topology_id, (entities,)
    )


def _complete_hp_parents(epoch: FiniteElementHPEpoch, marked: np.ndarray, /) -> Any:
    """Return parents whose full active leaf family is marked for coarsening."""

    topology = epoch.topology
    identifiers = np.asarray(topology.cell_global_ids, dtype=np.int64)
    active = np.asarray(topology.active, dtype=np.bool_)
    children = np.asarray(topology.child_slots, dtype=np.int64)
    child_valid = np.asarray(topology.child_valid, dtype=np.bool_)
    safe = np.where(child_valid, children, 0)
    leaf = ~np.any(child_valid[safe], axis=2)
    family = (
        np.asarray(topology.allocated, dtype=np.bool_)
        & ~active
        & (np.count_nonzero(child_valid, axis=1) == topology.child_capacity)
        & np.all(~child_valid | (active[safe] & leaf), axis=1)
    )
    chosen = family & np.all(~child_valid | np.isin(identifiers[safe], marked), axis=1)
    return identifiers[chosen]


def _execute_hp_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> RouteOutcome:
    with adaptation_phase(scope, budget):
        request = prepared.request
        # ty: ignore[unresolved-attribute]
        epoch = request.hierarchy
        source = prepared.source
        # ty: ignore[unresolved-attribute]
        refine = np.asarray(request.refine_cell_ids, dtype=np.int64)
        # ty: ignore[unresolved-attribute]
        coarsen = np.asarray(request.coarsen_cell_ids, dtype=np.int64)
        if refine.size:
            # ty: ignore[unresolved-attribute]
            closed, _ = balanced_hp_refinement_ids(
                epoch.topology, epoch.interfaces, refine
            )
            result = refine_tensor_hp_cells(
                # ty: ignore[unresolved-attribute]
                epoch.topology,
                # ty: ignore[unresolved-attribute]
                epoch.geometry,
                np.asarray(closed, dtype=np.int64),
            )
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REFINE
            partial = False
        else:
            # ty: ignore[invalid-argument-type]
            parents = _complete_hp_parents(epoch, coarsen)
            if parents.size == 0:
                return unchanged(prepared, None, epoch)
            # ty: ignore[unresolved-attribute]
            result = coarsen_tensor_hp_cells(epoch.topology, epoch.geometry, parents)
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.COARSEN
            covered = np.asarray(result.lineage.source_slots)[
                np.asarray(result.lineage.relation_mask("coarsening"), dtype=np.bool_)
            ]
            partial = covered.size != coarsen.size
        mesh = _dependency_phydrax_meshing__canonical.canonicalize_cell_mesh(
            hp_active_cell_mesh(
                result.topology,
                result.geometry,
                numeric_version=f"adaptation:{prepared.prepared_id}",
            )[0],
            budget=budget,
        )
        target = _dependency_phydrax_meshing__canonical.certify_cell_mesh(
            mesh,
            source.coordinate_contract,
            geometry=CellGeometrySpec.affine(mesh),
            audit_policy=prepared.policy.audit_policy,
            budget=budget,
        )
        target_epoch = FiniteElementHPEpoch(
            target.mesh,
            result.topology,
            result.geometry,
            finite_element_hp_interface_plan(result.topology, result.geometry),
        )
        # ty: ignore[invalid-argument-type]
        lineage = project_hp_lineage(epoch, target_epoch, result.lineage)
        transition = _dependency_phydrax_meshing__lineage.CellMeshTransition(
            source.mesh.mesh_id,
            source.mesh.topology_id,
            target,
            lineage,
            kind,
            budget=budget,
        )
        return RouteOutcome(
            MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE,
            target,
            transition,
            lineage,
            None,
            None,
            None,
            result,
            target_epoch,
        )


@checked
def compliance(
    prepared: PreparedMeshAdaptation, outcome: RouteOutcome, elapsed: float, /
) -> _dependency_phydrax_meshing__result.MeshingComplianceReport:
    limits = prepared.policy.limits
    audit = outcome.target.audit
    observations = (
        ("vertices", audit.vertex_count, limits.maximum_vertices),
        ("cells", audit.entity_counts[-1], limits.maximum_cells),
        (
            "connectivity_entries",
            audit.connectivity_entries,
            limits.maximum_connectivity_entries,
        ),
    )
    issues = tuple(
        f"maximum_{name}" for name, actual, maximum in observations if actual > maximum
    )
    if elapsed > limits.maximum_wall_seconds:
        issues += ("maximum_wall_seconds",)
    requested = tuple(
        (f"maximum_{name}", float(maximum)) for name, _, maximum in observations
    )
    achieved = tuple((name, float(actual)) for name, actual, _ in observations)
    evidence = outcome.evidence
    if isinstance(evidence, MetricRemeshingEvidence):
        quality_floor = (
            0.05
            if evidence.metric_quality_floor is None
            else evidence.metric_quality_floor
        )
        requested += (("minimum_metric_quality", quality_floor),)
        achieved += (
            ("minimum_metric_quality", evidence.minimum_metric_quality),
            ("out_of_range_edges", float(evidence.out_of_range_edges)),
            ("unmet_metric_edges", float(evidence.unmet_metric_edges)),
            ("source_lower_exempt_edges", float(len(evidence.source_lower_exemptions))),
            ("work_units", float(evidence.work_units)),
        )
        if evidence.execution_evidence is not None:
            record = evidence.execution_evidence
            work, queries = (
                float(np.asarray(record.work_counters.construction_work)),
                float(np.asarray(record.externally_charged_geometry_queries)),
            )
            requested += (
                ("maximum_work_units", float(limits.maximum_work_units)),
                ("maximum_geometry_queries", float(limits.maximum_geometry_queries)),
                ("maximum_scratch_bytes", float(limits.maximum_scratch_bytes)),
            )
            achieved += (
                ("native_scope_work_units", work),
                ("source_geometry_queries", queries),
                (
                    "native_primitive_queries",
                    float(np.asarray(record.native_primitive_queries)),
                ),
            )
            if (
                work > limits.maximum_work_units
                or queries > limits.maximum_geometry_queries
            ):
                issues += ("native_surface_cumulative_resource_limit",)
        if evidence.metric_arc_bounds is not None and isinstance(
            prepared.request, MetricMeshAdaptation
        ):
            arcs = np.asarray(evidence.metric_arc_bounds)
            requested += (("certified_metric_arcs_out_of_range", 0.0),)
            arc_failures = float(
                np.count_nonzero(
                    (arcs[:, 0] < evidence.lower_metric_length)
                    | (arcs[:, 1] > evidence.upper_metric_length)
                )
            )
            achieved += (("certified_metric_arcs_out_of_range", arc_failures),)
            if arc_failures:
                issues += ("source_metric_arc_goal",)
        if evidence.minimum_metric_quality < quality_floor:
            issues += ("minimum_metric_quality",)
        if isinstance(prepared.request, MetricMeshAdaptation):
            requested += (("unmet_metric_edges", 0.0),)
            if evidence.unmet_metric_edges:
                issues += ("metric_unit_mesh",)
        issues += evidence.size_issues
        if (
            evidence.status
            is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        ):
            issues += ("native_metric_resource_limit",)
    return _dependency_phydrax_meshing__result.MeshingComplianceReport(
        f"mesh-adaptation:{prepared.prepared_id}",
        issues=issues,
        requested=requested,
        achieved=achieved,
    )


@checked
def adaptation_result(
    prepared: PreparedMeshAdaptation, outcome: RouteOutcome, started: float, /
) -> MeshAdaptationResult:
    """Carry the distribution through the lineage, measure compliance, and bind."""

    policy = prepared.policy
    distribution = None
    if policy.distribution is not None and outcome.transition is not None:
        distribution = (
            _dependency_phydrax_meshing__distribution.prepare_distribution_transition(
                policy.distribution,
                mesh_part_owner.MeshPart(policy.distribution.part.name, outcome.target),
                # ty: ignore[invalid-argument-type]
                outcome.lineage,
                # ty: ignore[invalid-argument-type]
                policy=policy.partition_policy,
            )
        )
    if policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL:
        from .providers._native_publication import _require_retained_data

        # These are exactly the numerical owners retained by the root below;
        # compliance and the result identity contain only static metadata.
        _require_retained_data(
            (prepared.request, policy, prepared.source, outcome, distribution),
            policy.limits,
        )
    elapsed = time.monotonic() - started
    return MeshAdaptationResult(
        prepared,
        outcome,
        compliance(prepared, outcome, elapsed),
        distribution,
        elapsed,
    )


def _execute_mesh_routes(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> RouteOutcome | MeshAdaptationResult:
    """Execute exactly the prepared route and certify its target."""

    match prepared.policy.route:
        case MeshAdaptationRoute.NATIVE_BISECTION:
            outcome = _run_bisection_route(prepared, budget=budget, scope=scope)
        case MeshAdaptationRoute.NATIVE_METRIC_2D:
            outcome = _execute_metric_route(prepared, budget=budget, scope=scope)
        case MeshAdaptationRoute.NATIVE_METRIC_3D:
            outcome = _execute_tetra_metric_route(prepared, budget=budget, scope=scope)
        case MeshAdaptationRoute.NATIVE_SURFACE_METRIC:
            outcome = _execute_surface_metric_route(prepared, scope=scope, budget=budget)
        case MeshAdaptationRoute.NATIVE_LEVEL_SET:
            outcome = _owner_phydrax_meshing__level_set.execute_level_set_route(
                prepared, scope=scope, budget=budget
            )
        case MeshAdaptationRoute.NATIVE_MIXED:
            produced = _prepare_mixed_route(prepared, scope=scope)
            outcome = _finalize_mixed_route(
                prepared, produced, scope=scope, budget=budget
            )
        case MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            outcome = _execute_polyhedral_route(prepared, budget=budget, scope=scope)
        case MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION:
            outcome = _owner_phydrax_meshing__geometry_realization.execute_geometry_realization_route(
                prepared
            )
        case MeshAdaptationRoute.DEVICE_BISECTION:
            # Lazy: the device epoch module builds on this module's result types.

            storage = prepared.source.mesh.storage
            if storage is not None:
                results = _owner_phydrax_meshing__device_adaptation.execute_partitioned_mesh_adaptation(
                    _owner_phydrax_meshing__device_adaptation.prepare_partitioned_mesh_adaptation(
                        prepared
                    ),
                )
                for result in results.local_results:
                    target_storage = result.target.mesh.storage
                    if target_storage is not None and (
                        target_storage.partition_index == storage.partition_index
                    ):
                        return result
                raise RuntimeError(
                    f"MeshAdaptation.target.storage must publish the source addressable partition; got {storage.partition_index!r}"
                )
            outcome = (
                _owner_phydrax_meshing__device_adaptation.execute_device_bisection_route(
                    prepared, scope=scope, budget=budget
                )
            )
        case MeshAdaptationRoute.DEVICE_METRIC_2D:
            outcome = _owner_phydrax_meshing__device_metric.execute_device_metric_route(
                prepared, scope=scope, budget=budget
            )
        case MeshAdaptationRoute.MMG:
            outcome = _execute_mmg_route(prepared)
        case MeshAdaptationRoute.OMEGA_H:
            outcome = _execute_omega_h_route(prepared)
        case MeshAdaptationRoute.HP:
            outcome = _execute_hp_route(prepared, budget=budget, scope=scope)
        case invalid:
            assert_never(invalid)
    return outcome


def _preparation_execution_limits(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    receipt_snapshots: list[_NativeReceiptSnapshot] | None = None,
) -> tuple[NativeExecutionLimits, NativeExecutionRecord | None]:
    """Return execution limits after this request's own ended preparation.

    Preparation and execution share one original allowance: the ended phase's
    work, queries and elapsed time are debited once here, never reset. A live
    caller deadline (an enclosing scope or workflow) still intersects the
    result when the execution scope opens.
    """
    limits = prepared.policy.limits
    preparation = prepared.preparation_evidence
    native_limits = limits.native_limits()
    if preparation is None:
        return native_limits, None
    preparation.require_valid()
    if preparation.owner_id != prepared.prepared_id:
        raise ValueError(
            "PreparedMeshAdaptation.preparation_evidence must bind its preparation identity; got evidence for another preparation"
        )
    require_native_preparation_allowance(limits, preparation)
    # Debit the policy's ended phase before the live deadline is intersected
    # with the caller's wall deadline. Importing elapsed into that shorter
    # deadline would charge the same preparation interval twice.
    native_limits = debit_request_preparation(native_limits, preparation)
    live = _LIVE_POWER_WITNESSES.get(id(prepared))
    if live is not None and live[0]() is prepared and live[2] is not None:
        # A live polyhedral request retains its absolute preparation deadline,
        # including caller time between prepare and execute. This live clock
        # is not scientific identity or portable archive authority.
        native_limits = dataclasses.replace(
            native_limits,
            maximum_seconds=max(
                0.0, min(native_limits.maximum_seconds, live[2] - time.monotonic())
            ),
        )
    source_receipt = prepared.periodic_source_receipt
    if source_receipt is not None:
        # Live periodic mark-rebinding authority also expires with its proofs.
        native_limits = dataclasses.replace(
            native_limits,
            maximum_seconds=max(
                0.0,
                min(
                    native_limits.maximum_seconds,
                    source_receipt.expires_at - time.monotonic(),
                ),
            ),
        )
    if receipt_snapshots is not None:
        _prepare_native_receipt_graph(preparation, snapshots=receipt_snapshots)
    return native_limits, preparation


def _admit_periodic_source_proofs(
    prepared: PreparedMeshAdaptation, scope: NativeExecutionScope, /
) -> None:
    source_receipt = prepared.periodic_source_receipt
    if source_receipt is None or len(source_receipt.proofs) != len(
        prepared.periodic_source_proofs
    ):
        return
    for proof, earned in zip(
        prepared.periodic_source_proofs, source_receipt.proofs, strict=True
    ):
        if proof is earned:
            native_preparation_ledger(scope).remember_live(scope, proof)


def _publication_record_arrays(record: NativeExecutionRecord, /) -> tuple[Array, ...]:
    """Own only this new phase's leaves, never its retained predecessor graphs."""
    return tuple(
        jax.tree_util.tree_leaves(
            tuple(
                getattr(record, field.name)
                for field in dataclasses.fields(measurement_owner.NativeExecutionRecord)
                if field.name
                not in (
                    "preparation_evidence",
                    "consumer_evidence",
                    "history_evidence",
                    "owner_id",
                )
            )
        )
    )


class _AdaptationRootBinding:
    """Stage direct root slots while live; completion only installs references."""

    def __init__(
        self, owner: Any, names: tuple[str, ...], scope: NativeExecutionScope, /
    ) -> None:
        # The caller admits these root/path banks before entering this helper.
        # Charge the field traversal before flattening, including absent slots.
        scope.charge(work=len(dataclasses.fields(owner)))
        children, self.structure = jax.tree_util.tree_flatten_with_path(
            owner, is_leaf=lambda value: value is not owner
        )
        self.children = [value for _, value in children]
        self.positions: list[int] = []
        for name in names:
            scope.charge(work=len(children))
            slot = (jax.tree_util.GetAttrKey(name),)
            positions = [index for index, (path, _) in enumerate(children) if path == slot]
            if len(positions) != 1:
                raise RuntimeError(f"publication.{name} must have one root PyTree slot")
            self.positions.append(positions[0])
        # The ended operation performs only these already admitted slot writes
        # and one root reconstruction, without rediscovering paths or hashing.
        scope.charge(work=len(self.children) + len(self.positions))

    def bind(self, *values: object) -> Any:
        children, self.children = self.children, []
        for position, value in zip(self.positions, values, strict=True):
            children[position] = value
        return jax.tree_util.tree_unflatten(self.structure, children)


class _AdaptationPublicationStorage:
    """One pre-ended token for completion scratch and the new published owners.

    Native external-host tokens retain their allocator independently of the
    execution handle. Only admission runs while that handle is live; completion
    never resizes a detached workspace. Success transfers this exact token to
    the new roots and numerical leaves, and every failing path closes it.
    """

    def __init__(self, scope: NativeExecutionScope, /) -> None:
        self.scope = scope
        self.workspace = NativeHostStorageWorkspace(scope).open()
        self._finalizers: list[weakref.finalize] = []
        self._remaining = 0
        self._record_count_upper = 0
        self._publication_binding: _AdaptationRootBinding | None = None
        self._target_binding: _AdaptationRootBinding | None = None
        self._transition_binding: _AdaptationRootBinding | None = None

    def admit(
        self,
        publication: MeshAdaptationResult,
        preparation: NativeExecutionRecord | None,
        history: NativeExecutionRecord | None,
        imports: NativeExecutionImports,
        coordinate_operands: CoordinateOperandCollector | None,
        /,
    ) -> None:
        """Bound the concrete union's traversals, fixed leaves and root copies.

        Existing scientific payload is NOT charged a second time. The host
        profile is used only to count traversal/index storage and phase schemas;
        its returned owner map is temporary, not an ownership snapshot.
        """
        from .._host_storage import declared_object_storage_bytes
        from .providers._native_publication import _publication_provenance_bytes

        traversal = 1024
        self.workspace.set_bound(traversal)
        owner_indices = pending_slots = largest_fanout = 0
        # The profiler enqueues (owner, leaving, device_bank, tree_aux) tuples.
        # Reuse the canonical tuple header/reference bound and one list slot.
        pending_row = declared_object_storage_bytes((None, False, False, False)) + 16

        def admit_owner(owner: object) -> None:
            nonlocal traversal, owner_indices, pending_slots, largest_fanout
            # This is a new traversal, not the earlier topology retention
            # profile. Its canonical callback runs once per fresh identity.
            self.scope.charge(work=1)
            children = retained_host_storage_fanout_upper(owner)
            owner_indices += 128 + declared_object_storage_bytes(id(owner))
            pending_slots += children + int(children != 0)
            largest_fanout = max(largest_fanout, children)
            # One retained-dict slot (64), visiting-set slot (64), and the
            # canonical bit-length bound for its identity integer per owner.
            # All duplicate edges/leaving rows bound every pending frontier.
            # Only the largest current expansion needs a children-tuple/list
            # staging envelope, not eight such slots for every scalar leaf.
            traversal = (
                1024 + owner_indices + pending_row * pending_slots + 512 * largest_fanout
            )
            self.workspace.set_bound(traversal)

        owners, _, _, _ = retained_host_storage_profile(
            (publication, preparation, history),
            {},
            is_native_host_buffer,
            admit_owner=admit_owner,
        )
        work_fields = dataclasses.fields(
            measurement_owner.NativeExecutionDeviceWorkCounters
        )
        memory_fields = dataclasses.fields(
            measurement_owner.NativeExecutionDeviceMemoryCounters
        )
        record_fields = dataclasses.fields(measurement_owner.NativeExecutionRecord)
        # One scalar per work/memory field, plus each direct numerical field.
        arrays = (
            len(work_fields)
            + len(memory_fields)
            + sum(
                field.name
                not in (
                    "work_counters",
                    "memory_counters",
                    "preparation_evidence",
                    "consumer_evidence",
                    "history_evidence",
                    "owner_id",
                )
                for field in record_fields
            )
        )
        roots = (
            publication,
            publication.target,
            publication.transition,
            *(
                (
                    publication.distribution,
                    publication.distribution.target,
                    publication.distribution.target.part,
                )
                if publication.distribution is not None
                else ()
            ),
        )
        root_slots = sum(
            len(dataclasses.fields(root)) for root in roots if root is not None
        )
        # A copied root, GetAttrKey/path/pair, flatten/unflatten lists, static
        # auxiliary slots and tree descriptors: eight 256-byte headers/field.
        root_copies = 256 * (8 * root_slots + 8 * len(roots))
        if coordinate_operands is not None:
            from ..discretization._coordinate_operands import PreparedCoordinateOperands

            # complete() publishes one issued root, its field-reference tuple
            # and weak-key index entry; all referenced banks already exist.
            root_copies += 256 * (
                8 * len(dataclasses.fields(PreparedCoordinateOperands)) + 8
            )
        self._record_count_upper = arrays + 3 + len(roots)
        finalizers = 1024 + 512 * self._record_count_upper
        # Finalizer installation/deduplication is bounded by this pre-ended
        # schema; it must not disappear from the ended work receipt.
        self.scope.charge(work=2 * self._record_count_upper)
        # New JAX leaves, NumPy inputs and validation's host reads coexist.
        # The optional observation is conservatively reserved here; the data
        # preflight below counts it only if the actual ended evidence has it.
        numerical = 3 * (_NATIVE_EXECUTION_RECORD_ARRAY_BYTES + 256 * arrays)
        # Scope-end's 9+8 counters and two clocks, and the two host counter
        # records/evidence wrapper, precede construction of the numerical record.
        numerical += 19 * (256 + 8) + 3 * 256
        fixed = root_copies + finalizers + numerical
        digest = "f" * 64
        integer = (1 << 65) - 1  # Includes sums of two uint64 counters.
        phases = (
            (
                str(publication.prepared_id),
                len(imports.receipt_ids),
                int(history is not None),
            ),
            *(
                (
                    None if record.owner_id is None else str(record.owner_id),
                    int(np.asarray(record.imported_preparation_count)),
                    len(record.history_evidence),
                )
                for record in owners.values()
                if isinstance(record, measurement_owner.NativeExecutionRecord)
            ),
        )
        hash_peak = 0
        for owner_id, import_count, history_count in phases:
            # Admit the shape-only sizing tree before constructing it. It is
            # never passed off as completed evidence or hashed as a receipt.
            slots = (
                len(measurement_owner.NativeExecutionPhaseRecord.__annotations__)
                + len(work_fields)
                + len(memory_fields)
                + import_count
                + history_count
                + 16
            )
            schema = (
                1024
                + 512 * slots
                + 4 * (len(owner_id or "") + 64 * (import_count + history_count))
            )
            self.workspace.set_bound(traversal + fixed + max(hash_peak, schema))
            self.scope.charge(work=slots)
            phase: dict[str, object] = dict.fromkeys(
                measurement_owner.NativeExecutionPhaseRecord.__annotations__, integer
            )
            phase.update(
                owner_id=owner_id,
                receipt_id="",
                imported_preparation_ids=[digest] * import_count,
                work_counters={field.name: integer for field in work_fields},
                memory_counters={field.name: integer for field in memory_fields},
                status_name=max(MeshcoreStatus.__members__, key=len),
                elapsed_seconds=0.0,
                prior_elapsed_seconds=0.0,
                preparation_seconds=0.0,
                total_elapsed_seconds=0.0,
                preparation_evidence=None,
                consumer_evidence=None,
                history_evidence=[],
            )
            bound = _publication_provenance_bytes(
                {
                    "kind": "native-execution-receipt",
                    "phase": phase,
                    "preparation": digest,
                    "consumer": digest,
                    "history": [digest] * history_count,
                }
            )
            if bound is None:
                raise RuntimeError("publication receipt sizing must be primitive")
            hash_peak = max(hash_peak, schema + bound)
        self.workspace.set_bound(max(self.workspace.bound, traversal + fixed + hash_peak))
        # The result's identity and invariants were established by
        # adaptation_result while the request was live. Only receipt-bearing
        # root references change afterward; do not rerun its constructor/hash.
        self._publication_binding = _AdaptationRootBinding(
            publication, ("target", "transition", "distribution"), self.scope
        )
        self._target_binding = _AdaptationRootBinding(
            publication.target, ("execution_evidence",), self.scope
        )
        if publication.transition is not None:
            self._transition_binding = _AdaptationRootBinding(
                publication.transition, ("target",), self.scope
            )

    def bind_receipt(
        self, outcome: RouteOutcome, record: NativeExecutionRecord, /
    ) -> RouteOutcome:
        """Bind the constructor-validated record without repeating its graph audit."""
        if self._target_binding is None:
            raise RuntimeError("target binding must be admitted before scope end")
        target = self._target_binding.bind(record)
        transition = outcome.transition
        if self._transition_binding is not None:
            transition = self._transition_binding.bind(target)
        return outcome._replace(target=target, transition=transition)

    def bind(
        self,
        outcome: RouteOutcome,
        distribution: _dependency_phydrax_meshing__distribution.MeshDistributionTransition
        | None,
        /,
    ) -> MeshAdaptationResult:
        """Bind the ended receipt using the already validated immutable root."""
        if self._publication_binding is None:
            raise RuntimeError("publication binding must be admitted before scope end")
        return self._publication_binding.bind(
            outcome.target, outcome.transition, distribution
        )

    def handover(
        self, result: MeshAdaptationResult, record: NativeExecutionRecord, /
    ) -> None:
        """Transfer the admitted token without a release, resize or readmission."""
        roots = (
            result,
            result.target,
            record,
            record.work_counters,
            record.memory_counters,
            *((result.transition,) if result.transition is not None else ()),
            *(
                (
                    result.distribution,
                    result.distribution.target,
                    result.distribution.target.part,
                )
                if result.distribution is not None
                else ()
            ),
        )
        aliases = (*roots, *_publication_record_arrays(record))
        if len(aliases) > self._record_count_upper:
            raise RuntimeError("publication retention must fit its admitted owner count")
        seen: set[int] = set()
        for alias in aliases:
            if id(alias) in seen:
                continue
            seen.add(id(alias))
            self._finalizers.append(weakref.finalize(alias, self._release))
            self._remaining += 1

    def _release(self) -> None:
        self._remaining -= 1
        if not self._remaining:
            self.close()

    def close(self) -> None:
        for finalizer in self._finalizers:
            finalizer.detach()
        self._finalizers.clear()
        self._remaining = 0
        self._publication_binding = None
        self._target_binding = None
        self._transition_binding = None
        self.workspace.close()


def _bind_target_receipt(
    outcome: RouteOutcome, record: NativeExecutionRecord, /
) -> RouteOutcome:
    """Attach the ended settlement record to the published target and transition view.

    Execution evidence is outside result and transition identities, so both
    consumer-visible targets are the same bound result object without new IDs.
    Only the transition root slot is replaced; ``eqx.tree_at`` would rebuild
    the target's containers and publish a second, content-equal result graph.
    """
    target = outcome.target.with_execution_evidence(record)
    transition = outcome.transition
    if transition is not None:
        slot = (jax.tree_util.GetAttrKey("target"),)
        children, structure = jax.tree_util.tree_flatten_with_path(
            transition, is_leaf=lambda value: value is not transition
        )
        if sum(path == slot for path, _ in children) != 1:
            raise RuntimeError(
                "transition.target must have one root PyTree slot; got missing or duplicate slots"
            )
        transition = jax.tree_util.tree_unflatten(
            structure,
            [target if path == slot else value for path, value in children],
        )
    return outcome._replace(target=target, transition=transition)


def _require_adaptation_publication_deadline(
    expires_at: float | None, maximum_seconds: float, /
) -> None:
    """Check the same absolute request/parent clock after ended-record binding."""
    if expires_at is not None and time.monotonic() >= expires_at:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.TIMED_OUT,
            "MeshAdaptation.execution must finish receipt publication before its "
            "original request or earlier parent deadline",
            stage="adaptation-publication",
            requested=(("maximum_wall_seconds", maximum_seconds),),
            achieved=(("remaining_seconds", expires_at - time.monotonic()),),
        )


@checked
def execute_mesh_adaptation(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    coordinate_operands: CoordinateOperandCollector | None = None,
    observe_managed_admission: bool = False,
) -> MeshAdaptationResult:
    """Execute the prepared route under its declared native request limits.

    Explicit managed observation covers this execution scope, independently of
    preparation evidence. Missing preparation observation remains incomplete.
    """
    # Lazy: the envelope owner also uses native volume generation.
    from ._surface_envelope import (
        _require_surface_envelope_adaptation_source,
        _surface_envelope_repair_payload,
    )

    started = time.monotonic()
    limits = prepared.policy.limits
    receipt_snapshots: list[_NativeReceiptSnapshot] = []
    native_limits, preparation = _preparation_execution_limits(
        prepared, receipt_snapshots=receipt_snapshots
    )
    history = prepared.source.execution_evidence
    scope = NativeExecutionScope(native_limits)
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None
    ) = None
    outcome: RouteOutcome | MeshAdaptationResult | None = None
    imports: NativeExecutionImports | None = None
    operand_root: NativeExecutionScope | None = None
    publication: MeshAdaptationResult | None = None
    publication_expires_at: float | None = None
    publication_storage: _AdaptationPublicationStorage | None = None
    source_issue: NativeCompletedSourceIssue | None = None
    try:
        with scope:
            if observe_managed_admission or prepared.policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL:
                scope.observe_managed_admission()
            for snapshot in receipt_snapshots:
                native_preparation_ledger(scope).retain_receipt_snapshot(
                    scope, snapshot, preparation_outside_scope=True
                )
            receipt_snapshots.clear()
            if prepared.prepared_id != _prepared_adaptation_identity(
                prepared.source, prepared.request, prepared.policy
            ):
                raise ValueError(
                    "PreparedMeshAdaptation.prepared_id must authenticate its source certificate payload; got changed source evidence"
                )
            if history is not None:
                # The source's completed request is provenance, never a charge.
                native_preparation_ledger(scope).record_history(scope, history)
            initial = scope.batch(0)
            budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
                initial,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            initial.settle()
            budget.capture_coordinate_operands = coordinate_operands is not None
            budget.capture_polynomial_operands = (
                coordinate_operands is not None
                and coordinate_operands.capture_polynomials
            )
            budget.capture_signed_determinant_operands = (
                coordinate_operands is not None
                and coordinate_operands.capture_signed_determinants
            )
            budget.capture_measure_operands = (
                coordinate_operands is not None and coordinate_operands.capture_measures
            )
            _admit_periodic_source_proofs(prepared, scope)
            with (
                budget.measurement(),
                budget.retained_storage(
                    lambda amount: (
                        scope.ticket_view()
                        if prepared.policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL
                        else scope.batch(0, bytes_upper=amount)
                    )
                ),
            ):
                # A publicly constructed preparation may pair history with
                # another anchor; no-op outcomes would never reach carry.
                with (
                    adaptation_phase(scope, budget),
                    _surface_envelope_repair_payload(
                        prepared.source, budget=budget
                    ) as payload,
                ):
                    _import_power_witnesses(prepared, budget=budget)
                    # Planar source maps are consumed during route construction,
                    # before target publication reaches the mapped-proof binder.
                    _bind_plc_source_proofs(prepared, budget=budget, planar_only=True)
                    _require_surface_envelope_adaptation_source(
                        prepared.source, payload, prepared.policy.envelope_source
                    )
                outcome = _execute_mesh_routes(prepared, scope=scope, budget=budget)
                imports = NativeExecutionImports(scope, preparation=preparation)
                if (
                    coordinate_operands is not None
                    and outcome.status is MeshAdaptationStatus.COMPLETE
                ):
                    operand_root = scope
                    while operand_root.parent is not None:
                        operand_root = operand_root.parent
                    coordinate_operands.begin(
                        operand_root,
                        prepared.prepared_id if operand_root is scope else None,
                        native_preparation_ledger(operand_root).overlap(
                            operand_root, preparation if operand_root is scope else None
                        ),
                    )
                    with adaptation_phase(scope, budget):
                        coordinate_operands.capture(
                            outcome.target.mesh,
                            outcome.target.geometry,
                            budget,
                            scope,
                            coordinate_contract=outcome.target.coordinate_contract,
                            input_owners=(
                                prepared.source,
                                prepared.request,
                                prepared.policy,
                                prepared.constraints,
                            )
                            if prepared.policy.route
                            is MeshAdaptationRoute.NATIVE_POLYHEDRAL
                            else (),
                        )
                if (
                    prepared.policy.route is MeshAdaptationRoute.NATIVE_POLYHEDRAL
                    and outcome.status is MeshAdaptationStatus.COMPLETE
                ):
                    if not isinstance(outcome, RouteOutcome):
                        raise RuntimeError(
                            "Native polyhedral execution must retain its actual "
                            "route outcome until the ended receipt is bound."
                        )
                    # Distribution and compliance are part of this request, not
                    # a new phase with a renewed wall allowance.
                    publication = adaptation_result(prepared, outcome, started)
                    publication_storage = _AdaptationPublicationStorage(scope)
                    publication_storage.admit(
                        publication, preparation, history, imports, coordinate_operands
                    )
                    from .providers._native_publication import _require_retained_bytes

                    # The observation is enabled above, so the ended receipt's
                    # scalar bank has its full declared size. Receipt rebinding
                    # replaces references only and adds no other numerical data.
                    _require_retained_bytes(
                        model_array_bytes((publication, preparation, history))
                        + _NATIVE_EXECUTION_RECORD_ARRAY_BYTES,
                        limits,
                    )
                    publication_expires_at = (
                        time.monotonic() + scope.remaining().remaining_seconds
                    )
                    live = _LIVE_POWER_WITNESSES.get(id(prepared))
                    if live is not None and live[0]() is prepared and live[2] is not None:
                        publication_expires_at = min(publication_expires_at, live[2])
                if (
                    isinstance(outcome, RouteOutcome)
                    and outcome.status is MeshAdaptationStatus.COMPLETE
                ):
                    if scope.parent is None:
                        source_issue = stage_native_completed_source_issue(
                            scope,
                            outcome.target,
                            owner_id=prepared.prepared_id,
                            imports=imports,
                        )
                    else:
                        stage_native_source_publication(scope, outcome.target)
    except MeshcoreError as error:
        if source_issue is not None:
            source_issue.close()
            source_issue = None
        if publication_storage is not None:
            publication_storage.close()
        if coordinate_operands is not None:
            coordinate_operands.abort()
            coordinate_operands = None
        if (
            prepared.policy.route is not MeshAdaptationRoute.NATIVE_SURFACE_METRIC
            or error.status not in _NATIVE_RESOURCE_STATUSES
            or budget is None
        ):
            raise
        if outcome is None:
            outcome = unchanged(
                prepared,
                _surface_resource_refusal(prepared, budget, error),
                None,
                status=MeshAdaptationStatus.RESOURCE_LIMIT,
            )
        else:
            match outcome:
                case RouteOutcome(status=MeshAdaptationStatus.RESOURCE_LIMIT):
                    pass
                case _:
                    raise
    except BaseException:
        if source_issue is not None:
            source_issue.close()
        if publication_storage is not None:
            publication_storage.close()
        if coordinate_operands is not None:
            coordinate_operands.abort()
        raise
    if outcome is None:
        raise RuntimeError(
            f"MeshAdaptation.execution must retain an ended execution outcome and resource evidence; got {outcome is not None!r}"
        )
    match outcome:
        case MeshAdaptationResult():
            if coordinate_operands is not None:
                coordinate_operands.abort()
            return outcome
        case RouteOutcome():
            pass
    if publication is not None:
        try:
            _require_adaptation_publication_deadline(
                publication_expires_at, limits.maximum_wall_seconds
            )
            ended = scope.completed_evidence() if imports is None else imports.evidence()
            record = NativeExecutionRecord(
                ended,
                owner_id=prepared.prepared_id,
                imports=imports,
                preparation_evidence=preparation,
                history_evidence=() if history is None else (history,),
            )
            if publication_storage is None:
                raise RuntimeError(
                    "polyhedral publication must retain its pre-ended storage token"
                )
            outcome = publication_storage.bind_receipt(outcome, record)
            distribution = publication.distribution
            if distribution is not None:
                # Rebind only the three root slots; traversing the whole tree
                # would duplicate the scientific target's container graph.
                replacement = outcome.target
                for owner, name in (
                    (distribution.target.part, "carrier"),
                    (distribution.target, "part"),
                    (distribution, "target"),
                ):
                    slot = (jax.tree_util.GetAttrKey(name),)
                    children, structure = jax.tree_util.tree_flatten_with_path(
                        owner, is_leaf=lambda value: value is not owner
                    )
                    if sum(path == slot for path, _ in children) != 1:
                        raise RuntimeError(
                            f"distribution.{name} must have exactly one root PyTree slot"
                        )
                    replacement = jax.tree_util.tree_unflatten(
                        structure,
                        [
                            replacement if path == slot else value
                            for path, value in children
                        ],
                    )
                distribution = replacement
            result = publication_storage.bind(outcome, distribution)
            _require_adaptation_publication_deadline(
                publication_expires_at, limits.maximum_wall_seconds
            )
            if coordinate_operands is not None and operand_root is scope:
                coordinate_operands.complete(record)
            _require_adaptation_publication_deadline(
                publication_expires_at, limits.maximum_wall_seconds
            )
            parent = scope.parent
            if parent is not None:
                native_preparation_ledger(parent).remember_receipt(parent, record)
            publication_storage.handover(result, record)
            _require_adaptation_publication_deadline(
                publication_expires_at, limits.maximum_wall_seconds
            )
            if source_issue is not None:
                source_issue.complete(result.target)
            elif parent is not None:
                stage_native_source_publication(parent, result.target)
            _require_adaptation_publication_deadline(
                publication_expires_at, limits.maximum_wall_seconds
            )
            return result
        except BaseException:
            if source_issue is not None:
                source_issue.close()
            if publication_storage is not None:
                publication_storage.close()
            if coordinate_operands is not None:
                coordinate_operands.abort()
            raise
    record = NativeExecutionRecord(
        scope.completed_evidence() if imports is None else imports.evidence(),
        owner_id=prepared.prepared_id,
        imports=imports,
        preparation_evidence=preparation,
        history_evidence=() if history is None else (history,),
    )
    parent = scope.parent
    if parent is not None and outcome.status is MeshAdaptationStatus.COMPLETE:
        # Only successful preparation is reusable. Refused child work already
        # counts in the parent; retain its record solely as failure evidence.
        native_preparation_ledger(parent).remember_receipt(parent, record)
    match outcome.evidence:
        case MetricRemeshingEvidence() as metric_evidence:
            evidence = eqx.tree_at(
                lambda value: value.execution_evidence,
                metric_evidence,
                record,
                is_leaf=lambda value: value is None,
            )
            outcome = outcome._replace(evidence=evidence)
        case (
            _dependency_phydrax_meshing__mixed_adaptation.MixedAdaptationEvidence() as mixed_evidence
        ):
            outcome = outcome._replace(
                evidence=mixed_evidence._replace(execution_evidence=record)
            )
        case _:
            pass
    outcome = _bind_target_receipt(outcome, record)
    if outcome.status is MeshAdaptationStatus.COMPLETE:
        if source_issue is not None:
            source_issue.complete(outcome.target)
        elif parent is not None:
            stage_native_source_publication(parent, outcome.target)
        if coordinate_operands is not None and operand_root is scope:
            coordinate_operands.complete(record)
    else:
        if source_issue is not None:
            source_issue.close()
        if coordinate_operands is not None:
            coordinate_operands.abort()
    return adaptation_result(prepared, outcome, started)


__all__ = [
    "GeometryRealizationMeshAdaptation",
    "MarkedMeshAdaptation",
    "MeshAdaptationPolicy",
    "MeshAdaptationResult",
    "MeshAdaptationRoute",
    "MeshAdaptationStatus",
    "MetricMeshAdaptation",
    "PreparedMeshAdaptation",
    "RelocationMeshAdaptation",
    "execute_mesh_adaptation",
    "prepare_mesh_adaptation",
    "project_hp_lineage",
]
