#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Public Gmsh provider, plans, and the process-global Gmsh session."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from importlib import import_module, util
from typing import Any

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar
from ...discretization import CellMesh
from ...geometry.brep import BRepEntityId, BRepModel
from ...geometry.surface import SurfaceModel
from ...logging import emit
from ...typing import checked
from .._boundary_layer import BoundaryLayerMesh
from .._contracts import (
    MeshingCapability,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceDescriptor,
    MeshingSourceKind,
    ProviderSupportReport,
    SurfaceMeshingSpec,
    SurfaceRemeshingSpec,
    VolumeMeshingSpec,
)
from .._controls import BackgroundMetricControl, SurfaceReconstructionControl
from .._planar_bands import PlanarBandResult
from .._result import CellMeshingResult
from .._scope import MeshingEntityKind, MeshingScope
from .._session import AbstractMeshingSession, MeshingExecutionPolicy
from ._gmsh_boundary_layer import (
    _execute_layered_volume,
    _fill_boundary_layer_core,
    _layered_volume,
)
from ._gmsh_execute import _execute_brep
from ._gmsh_import import _brep_model, _BRepMeshingSource, _CadImportCache
from ._gmsh_inventory import _cad_occurrence_inventory, _cad_scope_set
from ._gmsh_options import GmshOptions
from ._gmsh_preflight import _brep_support_issues, _remeshing_support_issues
from ._gmsh_remesh import _execute_remesh, _surface_source


# Gmsh owns one process-global model state; one session may hold it at a time.
_GMSH_LOCK = threading.Lock()


def _optional_background(value: Any, /) -> BackgroundMetricControl | None:
    if value is not None and not isinstance(value, BackgroundMetricControl):
        raise TypeError("background_metric must be BackgroundMetricControl or None.")
    return value


def _background_for_specification(
    specification: SurfaceMeshingSpec | SurfaceRemeshingSpec | VolumeMeshingSpec,
    provider_metric: BackgroundMetricControl | None,
    /,
) -> BackgroundMetricControl | None:
    match specification:
        case SurfaceMeshingSpec():
            if provider_metric is not None:
                raise TypeError(
                    "Surface background metrics belong to SurfaceMeshingSpec.background_metric, "
                    "not the provider argument."
                )
            return specification.background_metric
        case SurfaceRemeshingSpec() | VolumeMeshingSpec():
            return _optional_background(provider_metric)
        case _:
            raise TypeError("specification must be a Gmsh meshing specification.")


class GmshMeshingPlan(StrictModule, NonTrainableState):
    source: BRepModel
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    options: GmshOptions
    support: ProviderSupportReport
    planar_bands: PlanarBandResult | None
    background_metric: BackgroundMetricControl | None
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: _BRepMeshingSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        options: GmshOptions,
        support: ProviderSupportReport,
        /,
        *,
        background_metric: BackgroundMetricControl | None = None,
    ) -> None:
        model = _brep_model(source)
        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError("specification must be surface or volume meshing.")
        background = _background_for_specification(specification, background_metric)
        support.require_supported()
        planar_bands = source if isinstance(source, PlanarBandResult) else None
        self.source = model
        self.specification = specification
        self.options = options
        self.support = support
        self.planar_bands = planar_bands
        self.background_metric = background
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gmsh-meshing-plan",
                "source_revision": model.report.source_revision,
                "specification": specification.specification_id,
                "options": options.options_id,
                "support": support.report_id,
                "planar_bands": (
                    None if planar_bands is None else planar_bands.result_id
                ),
                "background_metric": None
                if background is None
                else background.control_id,
            }
        )

    def execute(self, /) -> CellMeshingResult:
        return GmshProvider(self.options).execute(self)


class GmshRemeshingPlan(StrictModule, NonTrainableState):
    """Classify, reparametrize, and remesh one discrete source surface."""

    source: SurfaceModel
    specification: SurfaceRemeshingSpec
    options: GmshOptions
    support: ProviderSupportReport
    reconstruction: SurfaceReconstructionControl
    background_metric: BackgroundMetricControl | None
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: SurfaceModel,
        specification: SurfaceRemeshingSpec,
        options: GmshOptions,
        support: ProviderSupportReport,
        reconstruction: SurfaceReconstructionControl | None,
        /,
        *,
        background_metric: BackgroundMetricControl | None = None,
    ) -> None:
        background = _optional_background(background_metric)
        support.require_supported()
        if not isinstance(reconstruction, SurfaceReconstructionControl):
            raise TypeError("reconstruction must be SurfaceReconstructionControl.")
        self.source = source
        self.specification = specification
        self.options = options
        self.support = support
        self.reconstruction = reconstruction
        self.background_metric = background
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gmsh-remeshing-plan",
                "source_mesh": source.mesh.mesh_id,
                "source_metadata": source.metadata.metadata_id,
                "specification": specification.specification_id,
                "options": options.options_id,
                "support": support.report_id,
                "reconstruction": reconstruction.control_id,
                "background_metric": None
                if background is None
                else background.control_id,
            }
        )

    def execute(self, /) -> CellMeshingResult:
        return GmshProvider(self.options).execute(self)


class GmshSession(AbstractMeshingSession):
    """Exclusive owner of the process-global Gmsh state and its CAD import cache.

    CAD imports are cached by source revision and coordinate contract; every
    execution re-digests the persisted source bytes, so replaced bytes evict the
    cached import and fail as an invalid source instead of reusing stale geometry.
    """

    def __init__(
        self,
        provider: GmshProvider,
        policy: MeshingExecutionPolicy,
        /,
        *,
        import_cache_capacity: int = 4,
    ) -> None:
        capacity = int(import_cache_capacity)
        if capacity <= 0:
            raise ValueError("import_cache_capacity must be positive.")
        if policy.execution_mode is not MeshingExecutionMode.IN_PROCESS:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Gmsh currently supports in-process execution only.",
            )
        if util.find_spec("gmsh") is None:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "The optional gmsh Python package is unavailable.",
            )
        if not _GMSH_LOCK.acquire(blocking=False):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Another in-process Gmsh session owns the global provider state.",
            )
        try:
            gmsh = import_module("gmsh")
            if gmsh.isInitialized():
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    "An external owner already initialized the global Gmsh session.",
                )
            gmsh.initialize()
        except BaseException:
            _GMSH_LOCK.release()
            raise
        self._provider = provider
        self._policy = policy
        self._gmsh = gmsh
        self._imports = _CadImportCache(capacity)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def version(self) -> str:
        return str(self._gmsh.__version__)

    @property
    def import_cache_hits(self) -> int:
        return self._imports.hits

    @property
    def import_cache_misses(self) -> int:
        return self._imports.misses

    @property
    def cached_source_revisions(self) -> tuple[str, ...]:
        return self._imports.revisions

    def execute(self, plan: GmshMeshingPlan | GmshRemeshingPlan, /) -> CellMeshingResult:
        if self.closed:
            raise RuntimeError("Cannot execute with a closed Gmsh session.")
        if not isinstance(plan, (GmshMeshingPlan, GmshRemeshingPlan)):
            raise TypeError("plan must be GmshMeshingPlan or GmshRemeshingPlan.")
        if plan.options.num_threads > self._policy.parallelism:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Gmsh options request more threads than the session execution policy permits.",
            )
        info = self._provider.info
        match plan:
            case GmshMeshingPlan() if _layered_volume(plan):
                return _execute_layered_volume(
                    self._gmsh,
                    plan,
                    self.version,
                    info,
                    self._imports,
                    plan.background_metric,
                )
            case GmshMeshingPlan():
                return _execute_brep(
                    self._gmsh,
                    plan,
                    self.version,
                    info,
                    self._imports,
                    plan.background_metric,
                )
            case GmshRemeshingPlan():
                return _execute_remesh(self._gmsh, plan, self.version, info)
            case _:
                raise TypeError("plan must be GmshMeshingPlan or GmshRemeshingPlan.")

    @checked
    def fill_boundary_layer_core(
        self,
        layers: BoundaryLayerMesh,
        boundary: SurfaceModel,
        /,
        *,
        maximum_size: float | None = None,
    ) -> CellMeshingResult:
        """Tetrahedralize between a closed layer cap and the remaining boundary.

        The cap and ``boundary`` triangles stay fixed: the result fails unless
        every fixed node is bitwise unchanged and the core faces conform exactly.
        """
        if self.closed:
            raise RuntimeError("Cannot execute with a closed Gmsh session.")
        size = (
            None
            if maximum_size is None
            else finite_real_scalar(maximum_size, "maximum_size")
        )
        if size is not None and size <= 0.0:
            raise ValueError("maximum_size must be positive.")
        return _fill_boundary_layer_core(
            self._gmsh,
            layers,
            boundary,
            self._provider.options,
            size,
            self.version,
            self._provider.info,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._gmsh.finalize()
        finally:
            self._closed = True
            self._imports.close()
            _GMSH_LOCK.release()


class GmshProvider:
    def __init__(self, options: GmshOptions | None = None, /) -> None:
        self.options = GmshOptions() if options is None else options
        if not isinstance(self.options, GmshOptions):
            raise TypeError("options must be GmshOptions or None.")

    @property
    def info(self) -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "gmsh",
            "runtime",
            "GPL-2.0-or-later",
            operations=(
                MeshingOperation.MESH_SURFACE,
                MeshingOperation.MESH_VOLUME,
                MeshingOperation.REMESH_SURFACE,
            ),
            source_kinds=(
                MeshingSourceKind.BREP,
                MeshingSourceKind.SURFACE,
                MeshingSourceKind.CELL_MESH,
            ),
            capabilities=(
                MeshingCapability.DETERMINISTIC,
                MeshingCapability.CAD_CONFORMING,
                MeshingCapability.HIGH_ORDER_GEOMETRY,
                MeshingCapability.PERIODIC,
                MeshingCapability.MIXED_CELLS,
                MeshingCapability.BOUNDARY_LAYERS,
                MeshingCapability.MULTI_MATERIAL,
                MeshingCapability.ANISOTROPIC_METRIC,
                MeshingCapability.PARALLEL,
            ),
            cell_kinds=(
                "triangle",
                "quadrilateral",
                "tetrahedron",
                "prism",
                "hexahedron",
            ),
            dimensions=(2, 3),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    def entity_scope(
        self,
        source: _BRepMeshingSource,
        entity_ids: Sequence[BRepEntityId] | BRepEntityId,
        /,
    ) -> MeshingScope:
        """Select exact authored occurrence rows, or all members of a definition.

        An empty ``occurrence_path`` explicitly selects every authored occurrence
        of that definition. A nonempty path selects only that exact occurrence.
        """
        model = _brep_model(source)
        entities = (
            (entity_ids,) if isinstance(entity_ids, BRepEntityId) else tuple(entity_ids)
        )
        if not entities or not all(
            isinstance(entity, BRepEntityId) for entity in entities
        ):
            raise TypeError("entity_ids must contain at least one BRepEntityId.")
        revisions = {entity.source_revision for entity in entities}
        kinds = {entity.kind for entity in entities}
        if revisions != {model.report.source_revision} or len(kinds) != 1:
            raise ValueError(
                "BRep entity scopes require one kind from the supplied source revision."
            )
        kind = kinds.pop()
        dimensions = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
        inventory = _cad_occurrence_inventory(model)
        if kind not in dimensions:
            raise ValueError(
                "Gmsh BRep scopes support vertices, edges, faces, and solids."
            )
        dimension = dimensions[kind]
        identifiers = np.asarray(inventory.select(entities, dimension), dtype=np.int64)
        return MeshingScope(
            model.report.source_id,
            model.report.source_revision,
            MeshingEntityKind.GEOMETRY,
            dimension,
            _cad_scope_set(model, dimension),
            identifiers,
        )

    def whole_scope(
        self,
        source: _BRepMeshingSource | SurfaceModel | CellMesh,
        dimension: int,
        /,
    ) -> MeshingScope:
        """Select every physical CAD occurrence, or discrete surface mesh cells."""
        target = int(dimension)
        if isinstance(source, (SurfaceModel, CellMesh)):
            mesh = source.mesh if isinstance(source, SurfaceModel) else source
            if target != 2 or mesh.topological_dimension != 2:
                raise ValueError("Discrete surface scopes select dimension-two cells.")
            cells = mesh.entity_set(2)
            return MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                2,
                cells.entity_set_id,
                cells.entity_ids,
            )
        model = _brep_model(source)
        if target not in (0, 1, 2, 3):
            raise ValueError(
                "Gmsh BRep scope dimension must be zero, one, two, or three."
            )
        inventory = _cad_occurrence_inventory(model)
        count = len(inventory.entities[target])
        if not count:
            raise ValueError(
                f"The BRep source contains no dimension-{target} entities to scope."
            )
        return MeshingScope(
            model.report.source_id,
            model.report.source_revision,
            MeshingEntityKind.GEOMETRY,
            target,
            _cad_scope_set(model, target),
            np.arange(count, dtype=np.int64),
        )

    def inspect_source(
        self, source: _BRepMeshingSource | SurfaceModel | CellMesh, /
    ) -> MeshingSourceDescriptor:
        if isinstance(source, (SurfaceModel, CellMesh)):
            mesh = source.mesh if isinstance(source, SurfaceModel) else source
            return MeshingSourceDescriptor(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingSourceKind.SURFACE
                if isinstance(source, SurfaceModel)
                else MeshingSourceKind.CELL_MESH,
                mesh.topological_dimension,
                mesh.ambient_dimension,
                # ty: ignore[unresolved-attribute]
                closed=not bool(np.any(np.asarray(mesh.connectivity.boundary_edges))),
            )
        model = _brep_model(source)
        closed = bool(_cad_occurrence_inventory(model).entities[3])
        return MeshingSourceDescriptor(
            model.report.source_id,
            model.report.source_revision,
            MeshingSourceKind.BREP,
            3 if closed else 2,
            3,
            closed=closed,
        )

    def validate(
        self,
        source: _BRepMeshingSource | SurfaceModel | CellMesh,
        specification: SurfaceMeshingSpec | SurfaceRemeshingSpec | VolumeMeshingSpec,
        /,
        *,
        background_metric: BackgroundMetricControl | None = None,
        reconstruction: SurfaceReconstructionControl | None = None,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> ProviderSupportReport:
        background = _background_for_specification(specification, background_metric)
        if reconstruction is not None and not isinstance(
            reconstruction, SurfaceReconstructionControl
        ):
            raise TypeError(
                "reconstruction must be SurfaceReconstructionControl or None."
            )
        match specification:
            case SurfaceRemeshingSpec():
                # ty: ignore[invalid-argument-type]
                surface = _surface_source(source, coordinate_contract)
                descriptor = self.inspect_source(source)
                unsupported = _remeshing_support_issues(
                    self.options, surface, specification, background, reconstruction
                )
            case SurfaceMeshingSpec() | VolumeMeshingSpec():
                if reconstruction is not None or coordinate_contract is not None:
                    raise ValueError(
                        "BRep meshing sources own their coordinates and take no reconstruction control."
                    )
                descriptor = self.inspect_source(source)
                unsupported = _brep_support_issues(
                    self.options,
                    source,
                    # ty: ignore[invalid-argument-type]
                    _brep_model(source),
                    descriptor,
                    specification,
                    background,
                )
            case _:
                raise TypeError("specification must be a Gmsh meshing specification.")
        return ProviderSupportReport(
            self.info,
            descriptor,
            specification,
            unsupported=tuple(dict.fromkeys(unsupported)),
        )

    def plan(
        self,
        source: _BRepMeshingSource | SurfaceModel | CellMesh,
        specification: SurfaceMeshingSpec | SurfaceRemeshingSpec | VolumeMeshingSpec,
        /,
        *,
        background_metric: BackgroundMetricControl | None = None,
        reconstruction: SurfaceReconstructionControl | None = None,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> GmshMeshingPlan | GmshRemeshingPlan:
        support = self.validate(
            source,
            specification,
            background_metric=background_metric,
            reconstruction=reconstruction,
            coordinate_contract=coordinate_contract,
        )
        if isinstance(specification, SurfaceRemeshingSpec):
            return GmshRemeshingPlan(
                # ty: ignore[invalid-argument-type]
                _surface_source(source, coordinate_contract),
                specification,
                self.options,
                support,
                reconstruction,
                background_metric=background_metric,
            )
        return GmshMeshingPlan(
            # ty: ignore[invalid-argument-type]
            source,
            specification,
            self.options,
            support,
            background_metric=background_metric,
        )

    def open_session(
        self,
        policy: MeshingExecutionPolicy | None = None,
        /,
        *,
        import_cache_capacity: int = 4,
    ) -> GmshSession:
        execution = MeshingExecutionPolicy() if policy is None else policy
        if not isinstance(execution, MeshingExecutionPolicy):
            raise TypeError("policy must be MeshingExecutionPolicy or None.")
        return GmshSession(self, execution, import_cache_capacity=import_cache_capacity)

    def fill_boundary_layer_core(
        self,
        layers: BoundaryLayerMesh,
        boundary: SurfaceModel,
        /,
        *,
        maximum_size: float | None = None,
        # ty: ignore[invalid-return-type]
    ) -> CellMeshingResult:
        """Fill the core between native boundary layers and a fixed outer boundary."""
        threads = self.options.num_threads
        policy = MeshingExecutionPolicy(parallelism=threads, deterministic=threads == 1)
        with self.open_session(policy) as session:
            return session.fill_boundary_layer_core(
                layers, boundary, maximum_size=maximum_size
            )

    def execute(self, plan: GmshMeshingPlan | GmshRemeshingPlan, /) -> CellMeshingResult:
        started = time.perf_counter()
        emit(
            "DEBUG",
            "provider.execution.started",
            "Gmsh execution started",
            plan_id=plan.plan_id,
            provider="gmsh",
        )
        threads = plan.options.num_threads
        policy = MeshingExecutionPolicy(parallelism=threads, deterministic=threads == 1)
        try:
            with self.open_session(policy) as session:
                result = session.execute(plan)
        except MeshingFailure as error:
            emit(
                "ERROR",
                "provider.execution.failed",
                "Gmsh execution failed",
                elapsed_seconds=time.perf_counter() - started,
                failure_category=error.category.value,
                plan_id=plan.plan_id,
                provider="gmsh",
            )
            raise
        emit(
            "INFO",
            "provider.execution.completed",
            "Gmsh execution completed",
            elapsed_seconds=time.perf_counter() - started,
            mesh_id=result.mesh.mesh_id,
            plan_id=plan.plan_id,
            provider="gmsh",
            trace_id=result.trace.trace_id,
        )
        return result


__all__ = [
    "GmshMeshingPlan",
    "GmshProvider",
    "GmshRemeshingPlan",
    "GmshSession",
]
