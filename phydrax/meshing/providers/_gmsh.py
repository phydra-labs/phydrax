#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import threading
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Shape

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellBlock, CellGeometrySpec, CellMesh, lagrange_element
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...discretization._hexahedral import HexahedralConnectivity
from ...geometry.brep import BRepModel, BRepSource
from ...geometry.brep._occt import read_occt_shape
from ...geometry.simplicial import TriangleMesh
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import audit_cell_mesh
from .._canonical import canonicalize_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceDescriptor,
    MeshingSourceKind,
    ProviderSupportReport,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import (
    LayerTerminationPolicy,
    PrismLayerControl,
    ShellLayerControl,
    ThinRegionLayerControl,
)
from .._organization import MeshAttribute, MeshAttributeRole, MeshZone, MeshZoneRole
from .._quality import evaluate_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._session import AbstractMeshingSession, MeshingExecutionPolicy
from .._sizing import CurvatureSizeControl, UniformSizeControl
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


_GMSH_LOCK = threading.Lock()


def _brep_model(source: BRepModel | BRepSource, /) -> BRepModel:
    if isinstance(source, BRepSource):
        return source.model
    if isinstance(source, BRepModel):
        return source
    raise TypeError("source must be a BRepModel or solid BRepSource.")


class GmshOptions(StrictModule, NonTrainableState):
    algorithm_2d: int = eqx.field(static=True)
    algorithm_3d: int = eqx.field(static=True)
    terminal_output: bool = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract
    association_tolerance_factor: float = eqx.field(static=True)
    options_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        algorithm_2d: int = 6,
        algorithm_3d: int = 1,
        terminal_output: bool = False,
        coordinate_contract: SpatialCoordinateContract | None = None,
        association_tolerance_factor: float = 4.0,
    ):
        coordinates = (
            SpatialCoordinateContract.si()
            if coordinate_contract is None
            else coordinate_contract
        )
        if not isinstance(coordinates, SpatialCoordinateContract):
            raise TypeError(
                "coordinate_contract must be SpatialCoordinateContract or None."
            )
        factor = float(association_tolerance_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("association_tolerance_factor must be positive and finite.")
        self.algorithm_2d = int(algorithm_2d)
        self.algorithm_3d = int(algorithm_3d)
        self.terminal_output = bool(terminal_output)
        self.coordinate_contract = coordinates
        self.association_tolerance_factor = factor
        self.options_id = canonical_fingerprint(
            {
                "kind": "gmsh-options",
                "algorithm_2d": int(algorithm_2d),
                "algorithm_3d": int(algorithm_3d),
                "terminal_output": bool(terminal_output),
                "coordinate_contract": coordinates.spatial_id,
                "association_tolerance_factor": factor,
            }
        )


class GmshMeshingPlan(StrictModule, NonTrainableState):
    source: BRepModel
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    options: GmshOptions
    support: ProviderSupportReport
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: BRepModel | BRepSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        options: GmshOptions,
        support: ProviderSupportReport,
        /,
    ):
        model = _brep_model(source)
        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError("specification must be surface or volume meshing.")
        if not isinstance(options, GmshOptions):
            raise TypeError("options must be GmshOptions.")
        if not isinstance(support, ProviderSupportReport):
            raise TypeError("support must be ProviderSupportReport.")
        support.require_supported()
        self.source = model
        self.specification = specification
        self.options = options
        self.support = support
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gmsh-meshing-plan",
                "source_revision": source.report.source_revision,
                "specification": specification.specification_id,
                "options": options.options_id,
                "support": support.report_id,
            }
        )

    def execute(self, /) -> CellMeshingResult:
        return GmshProvider(self.options).execute(self)


class GmshSession(AbstractMeshingSession):
    def __init__(
        self,
        provider: GmshProvider,
        policy: MeshingExecutionPolicy,
        /,
    ):
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
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def version(self) -> str:
        return str(self._gmsh.__version__)

    def execute(self, plan: GmshMeshingPlan, /) -> CellMeshingResult:
        if self.closed:
            raise RuntimeError("Cannot execute with a closed Gmsh session.")
        if not isinstance(plan, GmshMeshingPlan):
            raise TypeError("plan must be GmshMeshingPlan.")
        return _execute_gmsh(self._gmsh, plan, self.version)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._gmsh.finalize()
        finally:
            self._closed = True
            _GMSH_LOCK.release()


class GmshProvider:
    def __init__(self, options: GmshOptions | None = None, /):
        self.options = GmshOptions() if options is None else options
        if not isinstance(self.options, GmshOptions):
            raise TypeError("options must be GmshOptions or None.")

    @property
    def info(self) -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "gmsh",
            "runtime",
            "GPL-2.0-or-later",
            operations=(MeshingOperation.MESH_SURFACE, MeshingOperation.MESH_VOLUME),
            source_kinds=(MeshingSourceKind.BREP,),
            capabilities=(
                MeshingCapability.DETERMINISTIC,
                MeshingCapability.CAD_CONFORMING,
                MeshingCapability.HIGH_ORDER_GEOMETRY,
                MeshingCapability.PERIODIC,
                MeshingCapability.MIXED_CELLS,
                MeshingCapability.BOUNDARY_LAYERS,
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

    def whole_scope(
        self, source: BRepModel | BRepSource, dimension: int, /
    ) -> MeshingScope:
        source = _brep_model(source)
        target = int(dimension)
        if target == 1:
            identifiers = np.arange(source.report.num_edges, dtype=np.int64)
        elif target == 2:
            identifiers = np.arange(source.report.num_faces, dtype=np.int64)
        elif target == 3:
            identifiers = np.asarray((0,), dtype=np.int64)
        else:
            raise ValueError("Gmsh BRep scope dimension must be one, two, or three.")
        return MeshingScope(
            source.report.source_id,
            source.report.source_revision,
            MeshingEntityKind.GEOMETRY,
            target,
            f"{source.report.source_revision}:brep:{target}",
            identifiers,
        )

    def inspect_source(
        self, source: BRepModel | BRepSource, /
    ) -> MeshingSourceDescriptor:
        model = _brep_model(source)
        closed = isinstance(source, BRepSource) or bool(
            TriangleMesh(model.mesh_vertices, model.mesh_faces).topology.watertight
        )
        return MeshingSourceDescriptor(
            source.report.source_id,
            source.report.source_revision,
            MeshingSourceKind.BREP,
            3 if closed else 2,
            3,
            closed=closed,
        )

    def validate(
        self,
        source: BRepModel | BRepSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
    ) -> ProviderSupportReport:
        descriptor = self.inspect_source(source)
        unsupported = []
        target = specification.target
        scope = (
            specification.scope
            if isinstance(specification, SurfaceMeshingSpec)
            else specification.boundary_scope
        )
        if (
            scope.source_id != descriptor.source_id
            or scope.source_revision != descriptor.source_revision
        ):
            unsupported.append("scope does not bind the supplied BRep revision")
        if (
            scope.scope_id
            != self.whole_scope(source, target.topological_dimension).scope_id
        ):
            unsupported.append(
                "Gmsh meshes the complete source; partial top-level scopes are unsupported"
            )
        if target.ambient_dimension != 3:
            unsupported.append(
                "Gmsh BRep output retains three-dimensional source coordinates"
            )
        requested = {
            *target.cell_families.required,
            *target.cell_families.preferred,
            *target.cell_families.allowed_transitions,
        }
        layers = (
            specification.layer_controls
            if isinstance(specification, VolumeMeshingSpec)
            else ()
        )
        supported = (
            {"triangle", "quadrilateral"}
            if target.topological_dimension == 2
            else {"prism", "hexahedron"}
            if layers
            else {"tetrahedron"}
        )
        if not requested <= supported:
            unsupported.append(f"Gmsh selected path supports only {sorted(supported)}")
        if target.geometry_order not in (1, 2):
            unsupported.append(
                "Gmsh canonical complete geometry elements support orders one and two"
            )
        if specification.protected_features:
            unsupported.append(
                "Gmsh protected-feature deviation contracts are not implemented"
            )
        if isinstance(specification, VolumeMeshingSpec):
            if not descriptor.closed:
                unsupported.append(
                    "Gmsh volume meshing requires a closed BRep solid, not an open surface model"
                )
            if layers:
                if specification.fill_strategy is not VolumeFillStrategy.SWEEP:
                    unsupported.append(
                        "Gmsh layers require explicit straight whole-volume SWEEP fill"
                    )
                if len(layers) != 1:
                    unsupported.append(
                        "Gmsh straight sweep accepts exactly one layer control"
                    )
                for control in layers:
                    if isinstance(control, ShellLayerControl):
                        unsupported.append(
                            "Gmsh BoundaryLayer is a 2-D field; ShellLayerControl is attached "
                            "to a volume contract and cannot certify a volume layer count"
                        )
                    elif isinstance(control, PrismLayerControl):
                        if control.termination is not LayerTerminationPolicy.REJECT:
                            unsupported.append(
                                "Gmsh straight prism sweep cannot certify COLLAPSE/TRUNCATE termination"
                            )
                        if requested != {"prism"}:
                            unsupported.append(
                                "PrismLayerControl requires prism-only output"
                            )
                        if (
                            control.surface_scope.entity_dimension != 2
                            or len(control.surface_scope.entity_ids) != 1
                        ):
                            unsupported.append(
                                "Gmsh straight prism sweep requires one planar source face"
                            )
                    elif isinstance(control, ThinRegionLayerControl):
                        if any(
                            s.entity_dimension != 2 or len(s.entity_ids) != 1
                            for s in (control.source_scope, control.target_scope)
                        ):
                            unsupported.append(
                                "Gmsh thin-region sweep requires one source face and one translated target face"
                            )
                    if (
                        not isinstance(control, ShellLayerControl)
                        and control.volume_scope.scope_id
                        != self.whole_scope(source, 3).scope_id
                    ):
                        unsupported.append(
                            "Gmsh layer volume scope must be the complete source"
                        )
            elif specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
                unsupported.append(
                    "Non-simplex Gmsh volume fill requires an explicit straight-sweep layer control"
                )
            if (
                specification.region_controls
                or specification.region_seeds
                or specification.hole_seeds
            ):
                unsupported.append(
                    "Gmsh region/material and hole-seed contracts are unsupported"
                )
        for constraint in specification.periodic_constraints:
            scopes = (constraint.source_scope, constraint.target_scope)
            if any(
                s.source_id != descriptor.source_id
                or s.source_revision != descriptor.source_revision
                or s.entity_kind is not MeshingEntityKind.GEOMETRY
                or s.entity_dimension not in (1, 2)
                for s in scopes
            ):
                unsupported.append(
                    "Gmsh periodic scopes must select source BRep curves or surfaces"
                )
            if scopes[0].entity_dimension != scopes[1].entity_dimension or len(
                scopes[0].entity_ids
            ) != len(scopes[1].entity_ids):
                unsupported.append(
                    "Gmsh periodic source/target scopes must have equal dimension and cardinality"
                )
            if np.asarray(constraint.transform).shape != (4, 4):
                unsupported.append(
                    "Gmsh periodic transforms must be 4-by-4 in source coordinates"
                )
        for control in specification.size_controls:
            if not isinstance(control, (UniformSizeControl, CurvatureSizeControl)):
                unsupported.append(
                    "Gmsh path supports uniform and curvature size controls"
                )
                continue
            if control.scope.scope_id != scope.scope_id:
                unsupported.append(
                    "Gmsh size controls must cover the whole meshed source"
                )
            if (
                isinstance(control, CurvatureSizeControl)
                and control.use_faceted_curvature
            ):
                unsupported.append(
                    "Gmsh curvature sizing uses CAD curvature, not faceted curvature"
                )
        return ProviderSupportReport(
            self.info,
            descriptor,
            specification,
            unsupported=tuple(unsupported),
        )

    def plan(
        self,
        source: BRepModel | BRepSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
    ) -> GmshMeshingPlan:
        return GmshMeshingPlan(
            source,
            specification,
            self.options,
            self.validate(source, specification),
        )

    def open_session(
        self,
        policy: MeshingExecutionPolicy | None = None,
        /,
    ) -> GmshSession:
        execution = MeshingExecutionPolicy() if policy is None else policy
        if not isinstance(execution, MeshingExecutionPolicy):
            raise TypeError("policy must be MeshingExecutionPolicy or None.")
        return GmshSession(self, execution)

    def execute(self, plan: GmshMeshingPlan, /) -> CellMeshingResult:
        with self.open_session() as session:
            return session.execute(plan)


@dataclass(frozen=True, slots=True)
class _ElementRows:
    tags: np.ndarray
    vertices: np.ndarray
    entity_tags: np.ndarray
    element_type: int
    cell_kind: str
    corner_count: int

    @property
    def block_name(self) -> str:
        return {
            "triangle": "triangles",
            "quadrilateral": "quadrilaterals",
            "tetrahedron": "tetrahedra",
            "prism": "prisms",
            "hexahedron": "hexahedra",
        }[self.cell_kind]


def _element_rows(
    gmsh, dimension: int, geometry_order: int, /
) -> tuple[_ElementRows, ...]:
    kinds = {
        "Triangle": "triangle",
        "Quadrilateral": "quadrilateral",
        "Tetrahedron": "tetrahedron",
        "Prism": "prism",
        "Hexahedron": "hexahedron",
    }
    records: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    properties = {}
    for _, entity_tag in sorted(gmsh.model.getEntities(dimension)):
        element_types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(
            dimension, entity_tag
        )
        for element_type, tag_values, node_values in zip(
            element_types, tag_blocks, node_blocks, strict=True
        ):
            element_type = int(element_type)
            name, _, order, count, _, corners = gmsh.model.mesh.getElementProperties(
                element_type
            )
            family = name.split()[0]
            if family not in kinds or int(order) != geometry_order:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    f"Gmsh returned unsupported element {name!r}; no elements may be discarded.",
                    stage=MeshingStageKind.CANONICALIZATION.value,
                )
            tags = np.asarray(tag_values, dtype=np.int64)
            nodes = np.asarray(node_values, dtype=np.int64).reshape((-1, int(count)))
            records.setdefault(element_type, []).append(
                (tags, nodes, np.full(tags.shape, entity_tag, dtype=np.int64))
            )
            properties[element_type] = (kinds[family], int(corners))
    if not records:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            f"Gmsh returned no dimension-{dimension} elements.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    result = []
    for element_type, chunks in records.items():
        tags, nodes, entities = (
            np.concatenate(values) for values in zip(*chunks, strict=True)
        )
        order = np.argsort(tags, kind="stable")
        kind, corners = properties[element_type]
        result.append(
            _ElementRows(
                tags[order], nodes[order], entities[order], element_type, kind, corners
            )
        )
    return tuple(sorted(result, key=lambda rows: (rows.corner_count, rows.block_name)))


def _local_connectivity(node_tags: np.ndarray, values: np.ndarray, /) -> np.ndarray:
    locations = np.searchsorted(node_tags, values)
    if np.any(locations >= node_tags.size) or not np.array_equal(
        node_tags[locations], values
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh element connectivity references an undeclared node tag.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return locations.astype(np.int32, copy=False)


def _geometry_permutation(gmsh, rows: _ElementRows, element, /) -> np.ndarray:
    """Map actual Gmsh reference nodes, not meshio's distinct wedge/hex ordering."""
    _, dimension, _, count, coordinates, _ = gmsh.model.mesh.getElementProperties(
        rows.element_type
    )
    source = np.asarray(coordinates, dtype=float).reshape((int(count), int(dimension)))
    if rows.cell_kind in ("quadrilateral", "hexahedron"):
        source = 0.5 * (source + 1.0)
    elif rows.cell_kind == "prism":
        source[:, 2] = 0.5 * (source[:, 2] + 1.0)
    target = np.asarray(element.reference_nodes, dtype=float)
    matches = np.max(np.abs(target[:, None] - source[None]), axis=-1) <= 2.0e-12
    if source.shape != target.shape or not np.all(np.sum(matches, axis=1) == 1):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            f"Gmsh {rows.cell_kind} geometry nodes do not match the canonical complete element.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return np.argmax(matches, axis=1).astype(np.int32)


def _audit_jacobians(gmsh, rows: tuple[_ElementRows, ...], /) -> float:
    """Audit the curved map with Gmsh's adaptive determinant extrema, not corners."""
    minimum = np.inf
    for block in rows:
        determinants = np.asarray(
            gmsh.model.mesh.getElementQualities(block.tags, "minDetJac"), dtype=float
        )
        invalid = ~np.isfinite(determinants) | (determinants <= 0.0)
        if determinants.shape != block.tags.shape or np.any(invalid):
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "Gmsh curved-element minimum Jacobian determinant is nonpositive or unavailable.",
                stage=MeshingStageKind.GEOMETRY_AUDIT.value,
                entity_ids=tuple(int(tag) for tag in block.tags[invalid])
                if determinants.shape == block.tags.shape
                else (),
            )
        minimum = min(minimum, float(np.min(determinants)))
    return minimum


@runtime_checkable
class _TopoDSEdgeCaster(Protocol):
    @staticmethod
    def Edge_s(shape: TopoDS_Shape, /) -> TopoDS_Edge: ...


def _scope_samples(
    source: BRepModel, shape, scope: MeshingScope, /
) -> tuple[np.ndarray, ...]:
    """Sample stable source entities independently of Gmsh's import tag numbering."""
    ids = np.asarray(scope.entity_ids, dtype=np.int64)
    count = (
        source.report.num_faces
        if scope.entity_dimension == 2
        else source.report.num_edges
    )
    if (
        scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension not in (1, 2)
        or scope.source_id != source.report.source_id
        or scope.source_revision != source.report.source_revision
        or np.any(ids >= count)
    ):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Scope does not select an entity of the supplied BRep revision.",
        )
    if scope.entity_dimension == 2:
        face_ids = np.asarray(source.triangle_face_ids)
        parameters = np.asarray(source.triangle_parameters)
        result = []
        for face in ids:
            triangles = np.flatnonzero(face_ids == face)
            if not triangles.size:
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Source face has no interior samples for entity resolution.",
                )
            selected = triangles[
                np.linspace(0, len(triangles) - 1, min(3, len(triangles)), dtype=int)
            ]
            uv = np.mean(parameters[selected], axis=1)
            result.append(np.asarray(source.patches[int(face)].evaluate(jnp.asarray(uv))))
        return tuple(result)
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS

    from ...geometry.brep._occt import _explore_unique

    if not isinstance(TopoDS, _TopoDSEdgeCaster):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "The CAD kernel must expose the TopoDS.Edge_s edge downcast.",
        )
    edges = _explore_unique(shape, TopAbs_EDGE, TopoDS.Edge_s)
    result = []
    for edge in ids:
        curve = BRepAdaptor_Curve(edges[int(edge)])
        parameters = np.linspace(curve.FirstParameter(), curve.LastParameter(), 5)[1:-1]
        values = [curve.Value(float(value)) for value in parameters]
        result.append(np.asarray([(point.X(), point.Y(), point.Z()) for point in values]))
    return tuple(result)


def _match_entities(
    gmsh, dimension: int, samples, candidates, tolerance: float, /
) -> tuple[int, ...]:
    result = []
    for points in samples:
        matches = []
        for tag in candidates:
            if tag in result:
                continue
            closest, _ = gmsh.model.getClosestPoint(
                dimension, tag, np.asarray(points).reshape(-1)
            )
            closest = np.asarray(closest).reshape((-1, 3))
            if (
                closest.shape == points.shape
                and np.max(np.linalg.norm(closest - points, axis=1)) <= tolerance
                and gmsh.model.isInside(dimension, tag, closest.reshape(-1))
                == len(points)
            ):
                matches.append(tag)
        if len(matches) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "BRep entity does not resolve uniquely to the imported Gmsh geometry.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        result.append(matches[0])
    return tuple(result)


def _resolve_entities(gmsh, source, shape, scope, /) -> tuple[int, ...]:
    samples = _scope_samples(source, shape, scope)
    scale = max(float(np.ptp(np.asarray(source.mesh_vertices), axis=0).max()), 1.0)
    return _match_entities(
        gmsh,
        scope.entity_dimension,
        samples,
        [tag for _, tag in gmsh.model.getEntities(scope.entity_dimension)],
        1.0e-7 * scale,
    )


def _set_periodic(gmsh, plan, shape, /):
    records = []
    slaves_used = set()
    for constraint in plan.specification.periodic_constraints:
        dimension = constraint.source_scope.entity_dimension
        masters = _resolve_entities(gmsh, plan.source, shape, constraint.source_scope)
        candidates = _resolve_entities(gmsh, plan.source, shape, constraint.target_scope)
        transform = np.asarray(constraint.transform)
        samples = _scope_samples(plan.source, shape, constraint.source_scope)
        transformed = tuple(
            points @ transform[:3, :3].T + transform[:3, 3] for points in samples
        )
        slaves = _match_entities(
            gmsh, dimension, transformed, candidates, constraint.tolerance
        )
        if set(masters) & set(slaves) or any(
            (dimension, tag) in slaves_used for tag in slaves
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Periodic self-pairs and multiply constrained slave entities are unsupported.",
            )
        gmsh.model.mesh.setPeriodic(
            dimension, list(slaves), list(masters), transform.reshape(-1).tolist()
        )
        slaves_used.update((dimension, tag) for tag in slaves)
        records.append((constraint, masters, slaves))
    return tuple(records)


def _audit_periodic(gmsh, records, node_tags, points, /):
    requested = []
    achieved = []
    for constraint, masters, slaves in records:
        dimension = constraint.source_scope.entity_dimension
        transform = np.asarray(constraint.transform)
        residual = 0.0
        pair_count = 0
        for master, slave in zip(masters, slaves, strict=True):
            actual_master, slave_nodes, master_nodes, actual_transform = (
                gmsh.model.mesh.getPeriodicNodes(
                    dimension, slave, includeHighOrderNodes=True
                )
            )
            slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
            master_nodes = np.asarray(master_nodes, dtype=np.int64)
            expected_slave, _, _ = gmsh.model.mesh.getNodes(
                dimension, slave, includeBoundary=True
            )
            expected_master, _, _ = gmsh.model.mesh.getNodes(
                dimension, master, includeBoundary=True
            )
            if (
                actual_master != master
                or not np.array_equal(np.sort(slave_nodes), np.unique(expected_slave))
                or not np.array_equal(np.sort(master_nodes), np.unique(expected_master))
                or not np.allclose(
                    np.asarray(actual_transform).reshape((4, 4)),
                    transform,
                    rtol=0.0,
                    atol=constraint.tolerance,
                )
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Gmsh periodic correspondence is not a complete high-order node bijection.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            slave_points = points[_local_connectivity(node_tags, slave_nodes)]
            master_points = points[_local_connectivity(node_tags, master_nodes)]
            mapped = master_points @ transform[:3, :3].T + transform[:3, 3]
            residual = max(
                residual,
                float(np.max(np.linalg.norm(slave_points - mapped, axis=1), initial=0.0)),
            )
            pair_count += slave_nodes.size
        if residual > constraint.tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Periodic node residual exceeds the exact requested tolerance.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        key = f"periodic:{constraint.constraint_id}"
        requested.append((f"{key}:tolerance", constraint.tolerance))
        achieved.extend(
            (
                (f"{key}:maximum_residual", residual),
                (f"{key}:node_pairs", float(pair_count)),
            )
        )
    return tuple(requested), tuple(achieved)


def _straight_sweep(gmsh, plan, shape, /):
    """Only full-volume straight extrusions; never a general boundary-layer claim."""
    controls = plan.specification.layer_controls
    if not controls:
        return None
    control = controls[0]
    volumes = gmsh.model.getEntities(3)
    if len(volumes) != 1:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Straight layers require one source solid.",
        )
    scope = (
        control.surface_scope
        if isinstance(control, PrismLayerControl)
        else control.source_scope
    )
    (base,) = _resolve_entities(gmsh, plan.source, shape, scope)
    if gmsh.model.getType(2, base) != "Plane":
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Straight layers require a planar source face.",
        )
    origin = np.asarray(gmsh.model.occ.getCenterOfMass(2, base))
    if isinstance(control, ThinRegionLayerControl):
        (target,) = _resolve_entities(gmsh, plan.source, shape, control.target_scope)
        if target == base or gmsh.model.getType(2, target) != "Plane":
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Thin-region target must be a distinct translated planar face.",
            )
        direction = np.asarray(gmsh.model.occ.getCenterOfMass(2, target)) - origin
        length = float(np.linalg.norm(direction))
        thicknesses = np.full(control.layer_count, length / control.layer_count)
    else:
        _, parameters = gmsh.model.getClosestPoint(2, base, origin)
        direction = np.asarray(gmsh.model.getNormal(base, parameters)).reshape(3)
        center = np.asarray(gmsh.model.occ.getCenterOfMass(*volumes[0]))
        if np.dot(direction, center - origin) < 0.0:
            direction = -direction
        thicknesses = control.first_layer_thickness * control.growth_rate ** np.arange(
            control.layer_count
        )
        length = float(np.sum(thicknesses))
        direction = direction / np.linalg.norm(direction) * length
    if not np.isfinite(length) or length <= 0.0:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "Layer sweep has nonpositive or nonfinite thickness.",
        )
    unit = direction / length
    # Thin-region controls specify count only, but still require a normal translation.
    _, parameters = gmsh.model.getClosestPoint(2, base, origin)
    normal = np.asarray(gmsh.model.getNormal(base, parameters)).reshape(3)
    if abs(float(np.dot(unit, normal))) < 1.0 - 1.0e-10:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Layer sweep must be normal to its planar source face.",
        )
    original = volumes[0]
    original_mass = gmsh.model.occ.getMass(*original)
    copied_base = gmsh.model.occ.copy([(2, base)])
    heights = np.cumsum(thicknesses) / length
    heights[-1] = 1.0
    extruded = gmsh.model.occ.extrude(
        copied_base,
        *direction.tolist(),
        numElements=[1] * control.layer_count,
        heights=heights.tolist(),
        recombine=True,
    )
    (generated,) = [entity for entity in extruded if entity[0] == 3]
    # Certify the source solid itself, not merely bounding boxes or volume equality.
    difference_mass = 0.0
    for left, right in ((original, generated), (generated, original)):
        left_copy = gmsh.model.occ.copy([left])
        right_copy = gmsh.model.occ.copy([right])
        difference, _ = gmsh.model.occ.cut(left_copy, right_copy)
        difference_mass += sum(
            gmsh.model.occ.getMass(dim, tag) for dim, tag in difference if dim == 3
        )
        if difference:
            gmsh.model.occ.remove(difference, recursive=True)
    if difference_mass > 1.0e-9 * original_mass:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Requested layers are not a straight extrusion filling the complete source solid "
            "(CAD symmetric difference is nonempty); partial boundary layers and general cores are unsupported.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    gmsh.model.occ.remove([original], recursive=True)
    gmsh.model.occ.synchronize()
    policy = plan.specification.target.cell_families
    if "hexahedron" in (*policy.required, *policy.preferred):
        for dim, tag in copied_base:
            gmsh.model.mesh.setRecombine(dim, tag)
    return (
        control,
        origin,
        unit,
        np.concatenate(([0.0], np.cumsum(thicknesses))),
        difference_mass / original_mass,
    )


def _audit_layers(sweep, rows, node_tags, points, /):
    if sweep is None:
        return (), ()
    control, origin, unit, levels, difference = sweep
    tolerance = 1.0e-9 * max(float(levels[-1]), 1.0)
    occupied = set()
    residual = 0.0
    measured_sums = np.zeros_like(levels)
    measured_counts = np.zeros(levels.shape, dtype=np.int64)
    for block in rows:
        if block.cell_kind not in ("prism", "hexahedron"):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Straight layers contain a non-extruded cell family.",
            )
        corners = points[
            _local_connectivity(node_tags, block.vertices[:, : block.corner_count])
        ]
        distances = (corners - origin) @ unit
        half = block.corner_count // 2
        bottom = np.mean(distances[:, :half], axis=1)
        top = np.mean(distances[:, half:], axis=1)
        lower = np.minimum(bottom, top)
        upper = np.maximum(bottom, top)
        intervals = np.argmin(np.abs(lower[:, None] - levels[None, :-1]), axis=1)
        errors = np.maximum(
            np.abs(lower - levels[intervals]), np.abs(upper - levels[intervals + 1])
        )
        residual = max(
            residual,
            float(np.max(errors)),
            float(np.max(np.ptp(distances[:, :half], axis=1))),
            float(np.max(np.ptp(distances[:, half:], axis=1))),
        )
        occupied.update(int(index) for index in intervals)
        np.add.at(measured_sums, intervals, lower)
        np.add.at(measured_sums, intervals + 1, upper)
        np.add.at(measured_counts, intervals, 1)
        np.add.at(measured_counts, intervals + 1, 1)
    if residual > tolerance or len(occupied) != control.layer_count:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Generated layer interfaces do not realize the requested count and thickness schedule.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    measured_levels = measured_sums / measured_counts
    measured_thicknesses = np.diff(measured_levels)
    requested = [("layer_count", float(control.layer_count))]
    if isinstance(control, PrismLayerControl):
        requested.extend(
            (
                ("first_layer_thickness", control.first_layer_thickness),
                ("layer_growth_rate", control.growth_rate),
            )
        )
    achieved = [
        ("layer_count", float(len(occupied))),
        ("first_layer_thickness", float(measured_thicknesses[0])),
        ("layer_interface_maximum_residual", residual),
        ("sweep_relative_cad_symmetric_difference", difference),
    ]
    if len(levels) > 2:
        achieved.append(
            (
                "layer_growth_rate",
                float(np.max(measured_thicknesses[1:] / measured_thicknesses[:-1])),
            )
        )
    return tuple(requested), tuple(achieved)


def _size_values(specification: SurfaceMeshingSpec | VolumeMeshingSpec, /):
    uniform = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if not uniform:
        curvature = tuple(
            control
            for control in specification.size_controls
            if isinstance(control, CurvatureSizeControl)
        )
        minimum = min(control.minimum_size for control in curvature)
        maximum = max(control.maximum_size for control in curvature)
        target = minimum
    else:
        minimum = min(control.minimum_size for control in uniform)
        maximum = max(control.maximum_size for control in uniform)
        target = min(control.target_size for control in uniform)
    curvature_angles = tuple(
        control.normal_angle
        for control in specification.size_controls
        if isinstance(control, CurvatureSizeControl)
    )
    curvature_points = (
        0 if not curvature_angles else int(np.ceil(2.0 * np.pi / min(curvature_angles)))
    )
    return minimum, target, maximum, curvature_points


def _boundary_association(
    source: BRepModel,
    boundary: CellMesh,
    tolerance_factor: float,
    /,
) -> tuple[GeometryAssociation, tuple[MeshZone, ...], MeshAttribute]:
    points = np.asarray(boundary.coordinates, dtype=float)
    centroids = np.concatenate(
        [
            np.mean(points[np.asarray(block.vertices, dtype=np.int32)], axis=1)
            for block in boundary.blocks
        ]
    )
    query_mesh = TriangleMesh(
        source.mesh_vertices,
        source.mesh_faces,
        source_id=f"{source.report.source_id}:association-query",
    )
    query = query_mesh.query_index().query(jnp.asarray(centroids))
    triangle_ids = np.asarray(query.face_index, dtype=np.int32)
    source_faces = np.asarray(source.triangle_face_ids, dtype=np.int32)[triangle_ids]
    residuals = np.asarray(query.distance, dtype=float)
    tolerance = max(
        source.report.linear_deflection * float(tolerance_factor),
        256.0 * np.finfo(float).eps,
    )
    resolved = residuals <= tolerance
    target_set = boundary.entity_set(2)
    source_ids = tuple(
        f"{source.report.source_revision}:face:{int(index)}" for index in source_faces
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        target_set.entity_set_id,
        target_set.entity_ids,
        source_ids,
        residuals,
        resolved=resolved,
        exact=False,
    )
    if not association.complete:
        failed = tuple(
            int(value) for value in np.asarray(target_set.entity_ids)[~resolved]
        )
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "Generated boundary faces could not be uniquely matched within tolerance.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            entity_ids=failed,
        )
    zones = []
    for face_id in np.unique(source_faces):
        selected = np.flatnonzero(source_faces == face_id)
        scope = MeshingScope(
            boundary.mesh_id,
            boundary.numeric_version,
            MeshingEntityKind.MESH,
            2,
            target_set.entity_set_id,
            np.asarray(target_set.entity_ids)[selected],
        )
        zones.append(MeshZone(f"brep-face-{int(face_id)}", MeshZoneRole.BOUNDARY, scope))
    all_scope = MeshingScope(
        boundary.mesh_id,
        boundary.numeric_version,
        MeshingEntityKind.MESH,
        2,
        target_set.entity_set_id,
        target_set.entity_ids,
    )
    attribute = MeshAttribute(
        "brep_face_index",
        MeshAttributeRole.GEOMETRY_CLASSIFICATION,
        all_scope,
        source_faces,
    )
    return association, tuple(zones), attribute


def _execute_gmsh(gmsh, plan: GmshMeshingPlan, version: str, /) -> CellMeshingResult:
    source = plan.source
    specification = plan.specification
    options = plan.options
    report = source.report
    source_path = Path(report.source_id)
    if not source_path.is_file():
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Gmsh BRep meshing requires a reopenable STEP/IGES/BREP source path.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    shape, source_format, current_revision = read_occt_shape(source_path)
    if (
        current_revision != report.source_revision
        or source_format != report.source_format
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The BRep source revision changed after import.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    limits = specification.limits
    if report.num_faces + report.num_edges + report.num_vertices > limits.maximum_faces:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "BRep entity count exceeds the meshing limit.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    minimum, target, maximum, curvature_points = _size_values(specification)
    dimension = specification.target.topological_dimension
    geometry_order = specification.target.geometry_order
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1 if options.terminal_output else 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.Algorithm", options.algorithm_2d)
    gmsh.option.setNumber("Mesh.Algorithm3D", options.algorithm_3d)
    gmsh.option.setNumber("Mesh.MeshSizeMin", minimum)
    gmsh.option.setNumber("Mesh.MeshSizeMax", maximum)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_points)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", geometry_order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    family_policy = specification.target.cell_families
    requested_kinds = {
        *family_policy.required,
        *family_policy.preferred,
        *family_policy.allowed_transitions,
    }
    # Full-quad recombination only when triangles/prisms are forbidden.
    pure_recombined = requested_kinds in ({"quadrilateral"}, {"hexahedron"})
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3 if pure_recombined else 0)
    gmsh.model.add(f"phydrax-{plan.plan_id[:12]}")
    gmsh.model.occ.importShapes(str(source_path))
    gmsh.model.occ.synchronize()
    sweep = (
        _straight_sweep(gmsh, plan, shape)
        if isinstance(specification, VolumeMeshingSpec)
        else None
    )
    if dimension == 2 and "quadrilateral" in requested_kinds:
        for entity_dimension, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(entity_dimension, tag)
    periodic_records = _set_periodic(gmsh, plan, shape)
    top_entities = sorted(gmsh.model.getEntities(dimension))
    if not top_entities:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "BRep source has no requested top-dimensional entities.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    if len(gmsh.model.getEntities()) > limits.maximum_faces:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Imported Gmsh entities exceed the declared limit.",
        )
    point_entities = gmsh.model.getEntities(0)
    if point_entities:
        gmsh.model.mesh.setSize(point_entities, target)
    for entity_dimension, tag in top_entities:
        gmsh.model.addPhysicalGroup(entity_dimension, [tag], tag=tag)
    gmsh.model.mesh.generate(dimension)

    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    order = np.argsort(node_tags, kind="stable")
    node_tags = node_tags[order]
    points = np.asarray(node_coordinates, dtype=float).reshape((-1, 3))[order]
    if points.shape[0] > limits.maximum_vertices:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Generated Gmsh mesh exceeds maximum_vertices.",
        )
    top = _element_rows(gmsh, dimension, geometry_order)
    cell_count = sum(rows.tags.size for rows in top)
    if cell_count > limits.maximum_cells:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Generated Gmsh mesh exceeds maximum_cells.",
        )
    minimum_jacobian = _audit_jacobians(gmsh, top)
    periodic_requested, periodic_achieved = _audit_periodic(
        gmsh, periodic_records, node_tags, points
    )
    layer_requested, layer_achieved = _audit_layers(sweep, top, node_tags, points)
    top_vertices = {
        rows.block_name: _local_connectivity(node_tags, rows.vertices) for rows in top
    }
    corner_nodes = np.unique(
        np.concatenate(
            [
                top_vertices[rows.block_name][:, : rows.corner_count].reshape(-1)
                for rows in top
            ]
        )
    )
    source_to_corner = np.full((points.shape[0],), -1, dtype=np.int32)
    source_to_corner[corner_nodes] = np.arange(corner_nodes.size, dtype=np.int32)
    mesh_points = points[corner_nodes]
    mesh_node_tags = node_tags[corner_nodes]
    mesh = CellMesh(
        mesh_points,
        tuple(
            CellBlock(
                rows.block_name,
                rows.cell_kind,
                source_to_corner[top_vertices[rows.block_name][:, : rows.corner_count]],
                global_ids=rows.tags,
            )
            for rows in top
        ),
        vertex_global_ids=mesh_node_tags,
        numeric_version=report.source_revision,
    )
    boundary_rows = _element_rows(gmsh, 2, geometry_order) if dimension == 3 else top
    boundary_triangles = []
    boundary_ids = []
    boundary_entities = []
    mixed_boundary = any(rows.cell_kind == "quadrilateral" for rows in boundary_rows)
    for rows in boundary_rows:
        corners = source_to_corner[
            _local_connectivity(node_tags, rows.vertices[:, : rows.corner_count])
        ]
        if np.any(corners < 0):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Boundary corner is absent from the volume mesh.",
            )
        splits = ((0, 1, 2),) if rows.cell_kind == "triangle" else ((0, 1, 2), (0, 2, 3))
        for index, split in enumerate(splits):
            boundary_triangles.append(corners[:, split])
            boundary_ids.append(2 * rows.tags + index if mixed_boundary else rows.tags)
            boundary_entities.append(rows.entity_tags)
    boundary_tags = np.concatenate(boundary_ids)
    boundary_order = np.argsort(boundary_tags, kind="stable")
    boundary_metadata = SurfaceMetadata(
        source_id=report.source_id,
        source_revision=report.source_revision,
        coordinate_contract=options.coordinate_contract,
        provenance=("gmsh-occ", plan.plan_id),
        cell_tags=tuple(
            f"gmsh-entity:{int(tag)}"
            for tag in np.concatenate(boundary_entities)[boundary_order]
        ),
    )
    boundary = SurfaceModel.from_triangles(
        mesh_points,
        np.concatenate(boundary_triangles)[boundary_order],
        boundary_metadata,
        vertex_global_ids=mesh_node_tags,
        cell_global_ids=boundary_tags[boundary_order],
        numeric_version=report.source_revision,
        repair_orientation=True,
        orient_closed_outward=dimension == 3,
    )
    if dimension == 2 and not mixed_boundary:
        mesh = boundary.mesh
    mesh = canonicalize_cell_mesh(mesh)
    association, zones, provider_attribute = _boundary_association(
        source,
        mesh if dimension == 2 else boundary.mesh,
        options.association_tolerance_factor,
    )
    if geometry_order == 1:
        geometry = CellGeometrySpec.affine(mesh)
    else:
        elements = {}
        routes = {}
        for rows in top:
            element = lagrange_element(rows.cell_kind, geometry_order)
            route = top_vertices[rows.block_name][
                :, _geometry_permutation(gmsh, rows, element)
            ]
            if dimension == 2 and not mixed_boundary:
                expected = corner_nodes[
                    np.asarray(mesh.block(rows.block_name).vertices, dtype=np.int32)
                ]
                flipped = np.any(route[:, :3] != expected, axis=1)
                if np.any(flipped):
                    reference = np.asarray(element.reference_nodes)
                    matches = (
                        np.max(
                            np.abs(reference[:, None] - reference[None, :, ::-1]), axis=-1
                        )
                        <= 2.0e-12
                    )
                    route[flipped] = route[flipped][:, np.argmax(matches, axis=1)]
            elements[rows.block_name] = element
            routes[rows.block_name] = route
        geometry = CellGeometrySpec(elements, routes, points)
    quality_evaluation = evaluate_cell_quality(mesh, mesh.coordinates)
    audit = audit_cell_mesh(
        mesh,
        geometry,
        quality_evaluation,
        associations=(association,),
        attributes=(provider_attribute,),
        zones=zones if dimension == 2 else (),
        boundary=boundary,
    )
    if not audit.passed:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=audit.quality.worst_cell_global_ids,
        )
    achieved_kinds = {block.cell_kind for block in mesh.blocks}
    connectivity = mesh.connectivity
    if not isinstance(
        connectivity,
        (
            PolygonalConnectivity,
            TetrahedralConnectivity,
            HexahedralConnectivity,
            PolyhedralConnectivity,
        ),
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh surface/volume conversion requires two- or three-dimensional connectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    connectivity_edges = np.asarray(connectivity.edges, dtype=np.int32)
    edge_lengths = np.linalg.norm(
        np.asarray(mesh.coordinates)[connectivity_edges[:, 1]]
        - np.asarray(mesh.coordinates)[connectivity_edges[:, 0]],
        axis=1,
    )
    minimum_edge = float(np.min(edge_lengths))
    maximum_edge = float(np.max(edge_lengths))
    vertex_minimum = np.full((mesh.coordinates.shape[0],), np.inf)
    vertex_maximum = np.zeros((mesh.coordinates.shape[0],), dtype=float)
    np.minimum.at(vertex_minimum, connectivity_edges[:, 0], edge_lengths)
    np.minimum.at(vertex_minimum, connectivity_edges[:, 1], edge_lengths)
    np.maximum.at(vertex_maximum, connectivity_edges[:, 0], edge_lengths)
    np.maximum.at(vertex_maximum, connectivity_edges[:, 1], edge_lengths)
    active = np.isfinite(vertex_minimum) & (vertex_minimum > 0.0)
    maximum_local_edge_ratio = float(
        np.max(vertex_maximum[active] / vertex_minimum[active], initial=1.0)
    )
    requested_growth = min(
        (
            control.maximum_growth_rate
            for control in specification.size_controls
            if isinstance(control, UniformSizeControl)
        ),
        default=1.0e300,
    )
    compliance_issues = []
    if (
        not set(family_policy.required) <= achieved_kinds
        or not achieved_kinds <= requested_kinds
        or (len(achieved_kinds) > 1 and not family_policy.allow_mixed)
    ):
        compliance_issues.append("cell_family")
    if minimum_edge < 0.5 * minimum or maximum_edge > 2.0 * maximum:
        compliance_issues.append("size_bounds")
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(compliance_issues),
        requested=(
            ("minimum_size", minimum),
            ("target_size", target),
            ("maximum_size", maximum),
            ("maximum_growth_rate", requested_growth),
            *periodic_requested,
            *layer_requested,
        ),
        achieved=(
            ("minimum_edge", minimum_edge),
            ("maximum_edge", maximum_edge),
            ("minimum_curved_jacobian_determinant", minimum_jacobian),
            ("maximum_local_edge_ratio", maximum_local_edge_ratio),
            *periodic_achieved,
            *layer_achieved,
        ),
    )
    if not compliance.passed:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "; ".join(compliance.issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(report.source_revision,),
            output_ids=(plan.support.source_descriptor_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SCOPE_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=(specification.specification_id,),
            output_ids=(
                (
                    specification.scope.scope_id
                    if isinstance(specification, SurfaceMeshingSpec)
                    else specification.boundary_scope.scope_id
                ),
            ),
        ),
        MeshingStageReport(
            MeshingStageKind.CONTROL_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=tuple(
                control.control_id for control in specification.size_controls
            ),
            output_ids=(plan.plan_id,),
        ),
        *(
            (
                MeshingStageReport(
                    MeshingStageKind.LAYER_GENERATION,
                    MeshingStageStatus.PASSED,
                    input_ids=(sweep[0].control_id,),
                    output_ids=(mesh.mesh_id,),
                    created_count=cell_count,
                ),
            )
            if sweep is not None
            else ()
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING
            if dimension == 2
            else MeshingStageKind.VOLUME_FILL,
            MeshingStageStatus.PASSED,
            input_ids=(plan.plan_id,),
            output_ids=(mesh.mesh_id,),
            created_count=cell_count,
        ),
        MeshingStageReport(
            MeshingStageKind.GEOMETRY_ASSOCIATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(association.association_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.CANONICALIZATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(mesh.topology_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.QUALITY_EVALUATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(audit.quality.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.GEOMETRY_AUDIT,
            MeshingStageStatus.PASSED,
            input_ids=(geometry.geometry_layout_id,),
            output_ids=(audit.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.TOPOLOGY_AUDIT,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.topology_id,),
            output_ids=(audit.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SPECIFICATION_COMPLIANCE,
            MeshingStageStatus.PASSED,
            input_ids=(specification.specification_id,),
            output_ids=(compliance.report_id,),
        ),
    )
    trace = MeshingTrace(stages)
    runtime = MeshingRuntimeInfo(
        plan.support.provider_id,
        version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=True,
        enforced_limits=("entities", "vertices", "cells"),
        unenforced_limits=("provider_workspace", "converted_arrays", "wall_time"),
    )
    provenance = SemanticProvenance(
        {
            "kind": "gmsh-cell-meshing-result",
            "source_revision": report.source_revision,
            "plan": plan.plan_id,
            "mesh": mesh.mesh_id,
            "association": association.association_id,
        },
        resource_ids={"source": report.source_id},
    )
    return CellMeshingResult(
        mesh,
        geometry,
        options.coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        GmshProvider(options).info,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        boundary=boundary,
        zones=zones if dimension == 2 else (),
        attributes=(provider_attribute,),
        associations=(association,),
    )


__all__ = [
    "GmshMeshingPlan",
    "GmshOptions",
    "GmshProvider",
    "GmshSession",
]
