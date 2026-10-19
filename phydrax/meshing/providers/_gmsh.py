#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Public Gmsh provider, plans, and the process-global Gmsh session."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from importlib import import_module, util

import equinox as eqx
import numpy as np

import phydrax.discretization as cell_mesh_owner
from phydrax.meshing._contracts import (
    MeshingCapability,
    MeshingExecutionMode,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceDescriptor,
    MeshingSourceKind,
    ProviderSupportReport,
    SurfaceMeshingSpec,
    SurfaceRemeshingSpec,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar
from ...geometry.brep import BRepEntityId, BRepModel
from ...geometry.surface import SurfaceModel
from ...logging import emit
from ...typing import checked
from .._boundary_layer import BoundaryLayerMesh
from .._controls import BackgroundMetricControl, SurfaceReconstructionControl
from .._planar_bands import PlanarBandResult
from .._result import CellMeshingResult
from .._scope import MeshingEntityKind, MeshingScope
from .._session import AbstractMeshingSession, MeshingExecutionPolicy
from ._gmsh_boundary_layer import (
    execute_layered_volume,
    fill_boundary_layer_core,
    layered_volume,
)
from ._gmsh_execute import execute_brep
from ._gmsh_import import brep_model, BRepMeshingSource, CadImportCache
from ._gmsh_inventory import cad_occurrence_inventory, cad_scope_set
from ._gmsh_options import GmshOptions
from ._gmsh_preflight import brep_support_issues, remeshing_support_issues
from ._gmsh_remesh import execute_remesh, surface_source


# Gmsh owns one process-global model state; one session may hold it at a time.
_GMSH_LOCK = threading.Lock()


def _background_for_specification(
    specification: SurfaceMeshingSpec | SurfaceRemeshingSpec | VolumeMeshingSpec,
    provider_metric: BackgroundMetricControl | None,
    /,
) -> BackgroundMetricControl | None:
    match specification:
        case SurfaceMeshingSpec():
            if provider_metric is not None:
                raise TypeError(
                    (
                        "gmsh.background_for_specification must bind a supported specification and its declared metric owner; got "
                        + "Surface background metrics belong to SurfaceMeshingSpec.background_metric, not the provider argument."
                    )
                )
            return specification.background_metric
        case SurfaceRemeshingSpec() | VolumeMeshingSpec():
            return provider_metric
        case _:
            raise TypeError(
                (
                    "gmsh.background_for_specification must bind a supported specification and its declared metric owner; got "
                    + "specification must be a Gmsh meshing specification."
                )
            )


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
        source: BRepMeshingSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        options: GmshOptions,
        support: ProviderSupportReport,
        /,
        *,
        background_metric: BackgroundMetricControl | None = None,
    ) -> None:
        model = brep_model(source)
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
        background = background_metric
        support.require_supported()
        if reconstruction is None:
            raise TypeError(
                "GmshRemeshingPlan.reconstruction must be SurfaceReconstructionControl; got None"
            )
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
            raise ValueError(
                f"GmshSession.import_cache_capacity must be positive; got {import_cache_capacity!r}"
            )
        if policy.execution_mode is not MeshingExecutionMode.IN_PROCESS:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                (
                    "gmsh.init__ must open an available exclusive in-process session with positive cache capacity; got "
                    + "Gmsh currently supports in-process execution only."
                ),
            )
        if policy.deterministic and policy.parallelism != 1:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "GmshSession.policy.parallelism must be 1 when deterministic "
                f"execution is required; got {policy.parallelism}",
                provider_code="deterministic_parallelism_unavailable",
                requested=(("parallelism", policy.parallelism),),
                achieved=(("deterministic_parallelism", 0),),
            )
        if util.find_spec("gmsh") is None:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                (
                    "gmsh.init__ must open an available exclusive in-process session with positive cache capacity; got "
                    + "The optional gmsh Python package is unavailable."
                ),
            )
        if not _GMSH_LOCK.acquire(blocking=False):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                (
                    "gmsh.init__ must open an available exclusive in-process session with positive cache capacity; got "
                    + "Another in-process Gmsh session owns the global provider state."
                ),
            )
        try:
            gmsh = import_module("gmsh")
            if gmsh.isInitialized():
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    (
                        "gmsh.init__ must open an available exclusive in-process session with positive cache capacity; got "
                        + "An external owner already initialized the global Gmsh session."
                    ),
                )
            gmsh.initialize()
        except BaseException:
            _GMSH_LOCK.release()
            raise
        self._provider = provider
        self._policy = policy
        self._gmsh = gmsh
        self._imports = CadImportCache(capacity)
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

    @checked
    def execute(self, plan: GmshMeshingPlan | GmshRemeshingPlan, /) -> CellMeshingResult:
        if self.closed:
            raise RuntimeError(
                (
                    "gmsh.execute must bind a live session and fit its thread policy; got "
                    + "Cannot execute with a closed Gmsh session."
                )
            )
        if plan.options.num_threads > self._policy.parallelism:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                (
                    "gmsh.execute must bind a live session and fit its thread policy; got "
                    + "Gmsh options request more threads than the session execution policy permits."
                ),
            )
        info = self._provider.info
        match plan:
            case GmshMeshingPlan() if layered_volume(plan):
                return execute_layered_volume(
                    self._gmsh,
                    plan,
                    self.version,
                    info,
                    self._imports,
                    plan.background_metric,
                )
            case GmshMeshingPlan():
                return execute_brep(
                    self._gmsh,
                    plan,
                    self.version,
                    info,
                    self._imports,
                    plan.background_metric,
                )
            case GmshRemeshingPlan():
                return execute_remesh(self._gmsh, plan, self.version, info)
            case _:
                raise TypeError(
                    (
                        "gmsh.execute must bind a live session and fit its thread policy; got "
                        + "plan must be GmshMeshingPlan or GmshRemeshingPlan."
                    )
                )

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
            raise RuntimeError(
                (
                    "gmsh.execute must bind a live session and fit its thread policy; got "
                    + "Cannot execute with a closed Gmsh session."
                )
            )
        size = (
            None
            if maximum_size is None
            else finite_real_scalar(maximum_size, "maximum_size")
        )
        if size is not None and size <= 0.0:
            raise ValueError(
                f"GmshSession.maximum_size must be positive; got {maximum_size!r}"
            )
        return fill_boundary_layer_core(
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
    @checked
    def __init__(self, options: GmshOptions | None = None, /) -> None:
        self.options = GmshOptions() if options is None else options

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

    @checked
    def entity_scope(
        self,
        source: BRepMeshingSource,
        entity_ids: Sequence[BRepEntityId] | BRepEntityId,
        /,
    ) -> MeshingScope:
        """Select authored occurrence rows, or all members of a definition.

        An empty ``occurrence_path`` selects every authored occurrence of that
        definition. A nonempty path selects only that occurrence.
        """
        model = brep_model(source)
        entities = (
            (entity_ids,) if isinstance(entity_ids, BRepEntityId) else tuple(entity_ids)
        )
        if not entities or not all(
            isinstance(entity, BRepEntityId) for entity in entities
        ):
            raise TypeError(
                (
                    "gmsh.entity_scope must select nonempty IDs of one source revision and entity kind; got "
                    + "entity_ids must contain at least one BRepEntityId."
                )
            )
        revisions = {entity.source_revision for entity in entities}
        kinds = {entity.kind for entity in entities}
        if revisions != {model.report.source_revision} or len(kinds) != 1:
            raise ValueError(
                (
                    "gmsh.entity_scope must select nonempty IDs of one source revision and entity kind; got "
                    + "BRep entity scopes require one kind from the supplied source revision."
                )
            )
        kind = kinds.pop()
        dimensions = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
        inventory = cad_occurrence_inventory(model)
        if kind not in dimensions:
            raise ValueError(
                (
                    "gmsh.entity_scope must select nonempty IDs of one source revision and entity kind; got "
                    + "Gmsh BRep scopes support vertices, edges, faces, and solids."
                )
            )
        dimension = dimensions[kind]
        identifiers = np.asarray(inventory.select(entities, dimension), dtype=np.int64)
        return MeshingScope(
            model.report.source_id,
            model.report.source_revision,
            MeshingEntityKind.GEOMETRY,
            dimension,
            cad_scope_set(model, dimension),
            identifiers,
        )

    @checked
    def whole_scope(
        self,
        source: BRepMeshingSource | SurfaceModel | cell_mesh_owner.CellMesh,
        dimension: int,
        /,
    ) -> MeshingScope:
        """Select every physical CAD occurrence, or discrete surface mesh cells."""
        target = int(dimension)
        if isinstance(source, (SurfaceModel, cell_mesh_owner.CellMesh)):
            mesh = source.mesh if isinstance(source, SurfaceModel) else source
            if target != 2 or mesh.topological_dimension != 2:
                raise ValueError(
                    (
                        "gmsh.whole_scope must select a populated source stratum of the requested dimension; got "
                        + "Discrete surface scopes select dimension-two cells."
                    )
                )
            cells = mesh.entity_set(2)
            return MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                2,
                cells.entity_set_id,
                cells.entity_ids,
            )
        model = brep_model(source)
        if target not in (0, 1, 2, 3):
            raise ValueError(
                (
                    "gmsh.whole_scope must select a populated source stratum of the requested dimension; got "
                    + "Gmsh BRep scope dimension must be zero, one, two, or three."
                )
            )
        inventory = cad_occurrence_inventory(model)
        count = len(inventory.entities[target])
        if not count:
            raise ValueError(
                (
                    "gmsh.whole_scope must select a populated source stratum of the requested dimension; got "
                    + f"The BRep source contains no dimension-{target} entities to scope."
                )
            )
        return MeshingScope(
            model.report.source_id,
            model.report.source_revision,
            MeshingEntityKind.GEOMETRY,
            target,
            cad_scope_set(model, target),
            np.arange(count, dtype=np.int64),
        )

    @checked
    def inspect_source(
        self, source: BRepMeshingSource | SurfaceModel | cell_mesh_owner.CellMesh, /
    ) -> MeshingSourceDescriptor:
        if isinstance(source, (SurfaceModel, cell_mesh_owner.CellMesh)):
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
        model = brep_model(source)
        closed = bool(cad_occurrence_inventory(model).entities[3])
        return MeshingSourceDescriptor(
            model.report.source_id,
            model.report.source_revision,
            MeshingSourceKind.BREP,
            3 if closed else 2,
            3,
            closed=closed,
        )

    @checked
    def validate(
        self,
        source: BRepMeshingSource | SurfaceModel | cell_mesh_owner.CellMesh,
        specification: SurfaceMeshingSpec | SurfaceRemeshingSpec | VolumeMeshingSpec,
        /,
        *,
        background_metric: BackgroundMetricControl | None = None,
        reconstruction: SurfaceReconstructionControl | None = None,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> ProviderSupportReport:
        background = _background_for_specification(specification, background_metric)
        match specification:
            case SurfaceRemeshingSpec():
                # ty: ignore[invalid-argument-type]
                surface = surface_source(source, coordinate_contract)
                descriptor = self.inspect_source(source)
                unsupported = remeshing_support_issues(
                    self.options, surface, specification, background, reconstruction
                )
            case SurfaceMeshingSpec() | VolumeMeshingSpec():
                if reconstruction is not None or coordinate_contract is not None:
                    raise ValueError(
                        (
                            "gmsh.validate must bind supported source/specification semantics; got "
                            + "BRep meshing sources own their coordinates and take no reconstruction control."
                        )
                    )
                descriptor = self.inspect_source(source)
                unsupported = brep_support_issues(
                    self.options,
                    source,
                    # ty: ignore[invalid-argument-type]
                    brep_model(source),
                    descriptor,
                    specification,
                    background,
                )
            case _:
                raise TypeError(
                    (
                        "gmsh.background_for_specification must bind a supported specification and its declared metric owner; got "
                        + "specification must be a Gmsh meshing specification."
                    )
                )
        return ProviderSupportReport(
            self.info,
            descriptor,
            specification,
            unsupported=tuple(dict.fromkeys(unsupported)),
        )

    @checked
    def plan(
        self,
        source: BRepMeshingSource | SurfaceModel | cell_mesh_owner.CellMesh,
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
                surface_source(source, coordinate_contract),
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

    @checked
    def open_session(
        self,
        policy: MeshingExecutionPolicy | None = None,
        /,
        *,
        import_cache_capacity: int = 4,
    ) -> GmshSession:
        execution = MeshingExecutionPolicy() if policy is None else policy
        return GmshSession(self, execution, import_cache_capacity=import_cache_capacity)

    @checked
    def fill_boundary_layer_core(
        self,
        layers: BoundaryLayerMesh,
        boundary: SurfaceModel,
        /,
        *,
        maximum_size: float | None = None,
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
