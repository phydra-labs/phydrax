#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Immutable source and assignment inputs for mesh recertification."""

from __future__ import annotations

import math

import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..geometry._affine_plc_domain import AffinePlcDomain
from ..geometry._domain_certification import (
    coverage_region_occurrence_paths,
    CoverageDomain,
    SourceCoverageWitness,
)
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..geometry._mesh_certificates import _same_tree_content
from ..geometry._planar_coverage import PlcSourcePointWitness
from ..geometry._source_boundary_records import (
    SourceBoundaryDistanceWitness,
    SourceBoundaryQuery,
)
from ..geometry._source_fidelity import (
    validate_source_fidelity_witness,
    ValidatedSourceFidelityWitness,
)
from ..geometry._source_fidelity_certificate import SourceFidelityWitness
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ..geometry.implicit._restricted_delaunay_coverage import (
    RestrictedDelaunayBoundaryWitness,
)
from ..typing import checked
from ._certification_schedule import MeshCertificationSchedule
from ._controls import ProtectedFeature
from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
from ._scope import _SourceEntityScopes


type ScopedFidelityInput = tuple[SourceBoundaryQuery, SourceFidelityWitness, float]


def _immutable_int_vector(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    result = np.array(array, dtype=np.int64, copy=True)
    if np.any(result < 0):
        raise ValueError(f"{name} must contain nonnegative global IDs.")
    result.setflags(write=False)
    return result


def _immutable_region_indices(
    value: ArrayLike, region_count: int, name: str, /
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer region-index array.")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains values outside the int64 range.")
    result = np.array(array, dtype=np.int64, copy=True)
    if np.any(result < 0) or np.any(result >= region_count):
        raise ValueError(f"{name} must index the declared domain.region_ids table.")
    result.setflags(write=False)
    return result


def _validate_fidelity_source(
    source: SourceBoundaryQuery | None,
    schedule: MeshCertificationSchedule,
    source_id: str,
    source_revision: str,
    tolerance: float | None,
    maximum_normal_deviation: float | None,
    /,
) -> float | None:
    if source is not None:
        if not isinstance(source, SourceBoundaryQuery) or not isinstance(
            source, StrictModule
        ):
            raise TypeError("source must be an immutable SourceBoundaryQuery.")
        if (source.source_id, source.source_revision) != (source_id, source_revision):
            raise ValueError("source identity must match source_id/source_revision.")
    if "source_fidelity" in schedule.required_checks and source is None:
        raise ValueError(
            "source must be supplied when the schedule requires source fidelity."
        )
    if (source is None) != (tolerance is None):
        raise ValueError(
            "fidelity_tolerance must be supplied exactly with a fidelity source."
        )
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0.0):
        raise ValueError("fidelity_tolerance must be finite and nonnegative.")
    normal_limit = (
        None if maximum_normal_deviation is None else float(maximum_normal_deviation)
    )
    if normal_limit is not None:
        if source is None:
            raise ValueError("Normal deviation requires a fidelity source.")
        if not math.isfinite(normal_limit) or not 0.0 <= normal_limit <= math.pi:
            raise ValueError("maximum_normal_deviation must lie in [0, pi].")
    return normal_limit


def _prepare_region_assignments(
    mesh: CellMesh,
    schedule: MeshCertificationSchedule,
    domain: CoverageDomain | None,
    cell_regions: ArrayLike | None,
    source_id: str,
    /,
) -> tuple[np.ndarray, np.ndarray | None]:
    domain_required = "domain_coverage" in schedule.required_checks
    if (domain is None) != (cell_regions is None) or domain_required != (
        domain is not None
    ):
        raise ValueError(
            "domain and cell_regions must be supplied together exactly when the schedule requires coverage."
        )
    if (
        domain is not None
        and canonical_identifier(domain.source_id, "domain.source_id") != source_id
    ):
        raise ValueError("domain.source_id must match the certification source.")
    region_ids = () if domain is None else tuple(domain.region_ids)
    region_keys = (
        ()
        if domain is None
        else tuple(zip(region_ids, coverage_region_occurrence_paths(domain), strict=True))
    )
    if domain is not None and (
        not region_ids
        or any(not isinstance(value, str) or not value for value in region_ids)
        or len(set(region_keys)) != len(region_ids)
    ):
        raise ValueError(
            "domain.region_ids and occurrence paths must uniquely identify its source regions."
        )
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    cell_ids.setflags(write=False)
    regions = None
    if cell_regions is not None:
        regions = _immutable_region_indices(cell_regions, len(region_ids), "cell_regions")
        if regions.shape != cell_ids.shape:
            raise ValueError(
                "cell_regions must contain one dense region-table index per mesh cell."
            )
    return cell_ids, regions


def _prepare_junctions(
    mesh: CellMesh,
    schedule: MeshCertificationSchedule,
    junction_vertices: ArrayLike | None,
    /,
) -> np.ndarray | None:
    if "junctions" in schedule.required_checks and junction_vertices is None:
        raise ValueError(
            "junction_vertices must be explicitly supplied by junction schedules."
        )
    if junction_vertices is None:
        return None
    junctions = _immutable_int_vector(junction_vertices, "junction_vertices")
    if np.unique(junctions).size != junctions.size:
        raise ValueError("junction_vertices must be unique.")
    if np.setdiff1d(junctions, np.asarray(mesh.vertex_global_ids, dtype=np.int64)).size:
        raise ValueError("junction_vertices must name vertices of this mesh.")
    junctions = np.sort(junctions)
    junctions.setflags(write=False)
    return junctions


def _prepare_features(
    protected_features: tuple[ProtectedFeature, ...] | None,
    schedule: MeshCertificationSchedule,
    source_id: str,
    source_revision: str,
    feature_source: _SourceEntityScopes | None,
    /,
) -> tuple[ProtectedFeature, ...] | None:
    features = None if protected_features is None else tuple(protected_features)
    if features is not None and not all(
        isinstance(value, ProtectedFeature) for value in features
    ):
        raise TypeError("protected_features must contain ProtectedFeature values.")
    if features is not None:
        features = tuple(sorted(features, key=lambda feature: feature.feature_id))
    if "protected_features" in schedule.required_checks and features is None:
        raise ValueError(
            "protected_features must be explicitly supplied by its schedule."
        )
    if features is not None:
        identifiers = tuple(feature.feature_id for feature in features)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("protected_features must have unique feature identities.")
        if any(
            (feature.scope.source_id, feature.scope.source_revision)
            != (source_id, source_revision)
            for feature in features
        ):
            raise ValueError(
                "protected_features must belong to the retained source revision."
            )
    if feature_source is None:
        if features:
            raise ValueError(
                "Protected features require their authoritative source entity mapping."
            )
        return features
    if not isinstance(feature_source, _SourceEntityScopes) or not isinstance(
        feature_source, StrictModule
    ):
        raise TypeError("feature_source must be an immutable source entity owner.")
    if (feature_source.source_id, feature_source.source_revision) != (
        source_id,
        source_revision,
    ):
        raise ValueError("feature_source must match the certification source revision.")
    for feature in features or ():
        scope = feature.scope
        if (
            feature_source.scope_entity_set_id(scope.entity_dimension)
            != scope.entity_set_id
        ):
            raise ValueError(
                "Protected feature scope must belong to its source entity set."
            )
        for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist():
            feature_source.source_entity_key_for_scope(scope.entity_dimension, identifier)
    return features


def _require_coverage_witness_binding(
    witness: SourceCoverageWitness | None,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: CoverageDomain | None,
    fidelity_witness: SourceFidelityWitness | None,
    /,
) -> None:
    """Cross-bind independent metric and topology premises, without asserting proof."""
    if witness is None:
        return
    if isinstance(witness, PlcSourcePointWitness):
        if (
            not isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain))
            or witness.domain_id != domain.domain_id
            or not np.array_equal(witness.vertex_global_ids, mesh.vertex_global_ids)
        ):
            raise ValueError(
                "PLC source points must bind the piecewise-linear domain and mesh vertices."
            )
        return
    if not isinstance(domain, AnalyticImplicitProfile) or not isinstance(
        fidelity_witness, SourceBoundaryDistanceWitness
    ):
        raise ValueError(
            "Restricted coverage requires its analytic domain and distance witness."
        )
    if isinstance(witness, RestrictedDelaunayBoundaryWitness):
        root = witness
        target = (witness.mesh_id, witness.geometry_id, witness.topology_id)
        distance_id = witness.distance_witness_id
    else:
        root = witness.root_witness
        target = (
            witness.target_mesh_id,
            witness.target_geometry_id,
            witness.target_topology_id,
        )
        distance_id = witness.target_distance_witness_id
    if (
        target != (mesh.mesh_id, cell_geometry_id(geometry), mesh.topology_id)
        or (
            root.source_id,
            root.source_revision,
            root.domain_id,
            root.profile_id,
        )
        != (domain.source_id, domain.source_revision, domain.domain_id, domain.profile_id)
        or distance_id != fidelity_witness.witness_id
    ):
        raise ValueError(
            "Restricted coverage premises must bind the same source, target and distance witness."
        )


def validated_fidelity_witness(
    validated: tuple[ValidatedSourceFidelityWitness, ...],
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceFidelityWitness,
    /,
) -> SourceFidelityWitness | ValidatedSourceFidelityWitness:
    """Return the validation of ``witness`` for exactly these operands, if handed one.

    Otherwise ``witness`` itself is returned, so its consumer validates it.
    """
    for value in validated:
        if value.witness is witness and value.binds(mesh, geometry, source):
            return value
    return witness


def _validate_scoped_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    values: tuple[ScopedFidelityInput, ...],
    validated: tuple[ValidatedSourceFidelityWitness, ...],
    /,
) -> tuple[tuple[SourceBoundaryQuery, ValidatedSourceFidelityWitness, float], ...]:
    scoped: list[tuple[SourceBoundaryQuery, ValidatedSourceFidelityWitness, float]] = []
    for query, witness, tolerance in values:
        if not isinstance(query, SourceBoundaryQuery) or not isinstance(
            query, StrictModule
        ):
            raise TypeError(
                "scoped fidelity sources must be immutable SourceBoundaryQuery values."
            )
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("scoped fidelity tolerances must be finite and nonnegative.")
        token = validate_source_fidelity_witness(
            mesh,
            geometry,
            query,
            validated_fidelity_witness(validated, mesh, geometry, query, witness),
        )
        scoped.append((query, token, float(tolerance)))
    return tuple(
        sorted(
            scoped,
            key=lambda item: (
                item[0].source_id,
                item[0].source_revision,
                item[1].witness.witness_id,
                item[2],
            ),
        )
    )


class MeshCertificationInputs(StrictModule, NonTrainableState):
    """Retain actual immutable mesh, coordinate, source, and coverage inputs."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    schedule: MeshCertificationSchedule
    domain: CoverageDomain | None
    source: SourceBoundaryQuery | None
    cell_regions: np.ndarray | None
    junction_vertices: np.ndarray | None
    protected_features: tuple[ProtectedFeature, ...] | None
    scoped_fidelity: tuple[ScopedFidelityInput, ...]
    fidelity_tolerance: float | None
    maximum_normal_deviation: float | None
    fidelity_witness: SourceFidelityWitness | None
    feature_source: _SourceEntityScopes | None
    layer_core_root_geometry: LayerCoreRootGeometry | None
    source_coverage_witness: SourceCoverageWitness | None
    mesh_id: str
    topology_id: str
    geometry_id: str
    domain_id: str | None
    source_id: str
    source_revision: str
    cell_global_ids: np.ndarray
    request_id: str

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        schedule: MeshCertificationSchedule,
        /,
        *,
        source_id: str,
        source_revision: str,
        domain: CoverageDomain | None = None,
        cell_regions: ArrayLike | None = None,
        source: SourceBoundaryQuery | None = None,
        fidelity_tolerance: float | None = None,
        maximum_normal_deviation: float | None = None,
        junction_vertices: ArrayLike | None = None,
        protected_features: tuple[ProtectedFeature, ...] | None = None,
        scoped_fidelity: tuple[ScopedFidelityInput, ...] = (),
        fidelity_witness: SourceFidelityWitness | None = None,
        feature_source: _SourceEntityScopes | None = None,
        layer_core_root_geometry: LayerCoreRootGeometry | None = None,
        source_coverage_witness: SourceCoverageWitness | None = None,
        validated: tuple[ValidatedSourceFidelityWitness, ...] = (),
        computed_root: ComputedLayerCoreRoot | None = None,
    ) -> None:
        """Retain the inputs; their fidelity witnesses must validate on ``mesh``.

        ``validated`` holds validations the caller already made for exactly
        these operands; a witness with one is not validated again.
        ``feature_source`` may also retain the authoritative layer/core source
        for association replay when no features are protected; retaining it
        does not itself add a protected feature or geometric constraint.
        ``layer_core_root_geometry`` retains the independently proved root
        owner for current-to-root association replay; it is not a new source.
        """
        source_id_ = canonical_identifier(source_id, "source_id")
        source_revision_ = canonical_identifier(source_revision, "source_revision")
        normal_limit = _validate_fidelity_source(
            source,
            schedule,
            source_id_,
            source_revision_,
            fidelity_tolerance,
            maximum_normal_deviation,
        )
        if fidelity_witness is not None:
            if source is None:
                raise ValueError(
                    "A fidelity witness requires its authoritative source query."
                )
            validate_source_fidelity_witness(
                mesh,
                geometry,
                source,
                validated_fidelity_witness(
                    validated, mesh, geometry, source, fidelity_witness
                ),
            )
        cell_ids, regions = _prepare_region_assignments(
            mesh,
            schedule,
            domain,
            cell_regions,
            source_id_,
        )
        junctions = _prepare_junctions(mesh, schedule, junction_vertices)
        features = _prepare_features(
            protected_features,
            schedule,
            source_id_,
            source_revision_,
            feature_source,
        )
        if feature_source is not None or layer_core_root_geometry is not None:
            from .providers._native_sources import NativeLayerCoreSource

            if (
                type(feature_source) is NativeLayerCoreSource
                and feature_source.conforming_execution
                and layer_core_root_geometry is None
            ):
                raise ValueError(
                    "Conforming layer/core certification requires its root geometry."
                )
        if layer_core_root_geometry is not None:
            if type(layer_core_root_geometry) is not LayerCoreRootGeometry:
                raise TypeError("layer_core_root_geometry must be LayerCoreRootGeometry.")
            if type(feature_source) is not NativeLayerCoreSource:
                raise TypeError(
                    "Layer root geometry requires its actual layer/core source."
                )
            if computed_root is None:
                layer_core_root_geometry.require_current(
                    feature_source, layer_core_root_geometry.specification
                )
            elif type(computed_root) is ComputedLayerCoreRoot:
                computed_root.require(
                    feature_source,
                    layer_core_root_geometry.specification,
                    layer_core_root_geometry.mesh,
                    layer_core_root_geometry.geometry,
                    layer_core_root_geometry,
                )
            else:
                raise TypeError("computed_root must be ComputedLayerCoreRoot or None.")
        elif computed_root is not None:
            raise ValueError("A computed root requires its retained root geometry.")
        scoped = tuple(
            (query, token.witness, tolerance)
            for query, token, tolerance in _validate_scoped_fidelity(
                mesh, geometry, scoped_fidelity, validated
            )
        )
        _require_coverage_witness_binding(
            source_coverage_witness,
            mesh,
            geometry,
            domain,
            fidelity_witness,
        )
        domain_id = (
            None
            if domain is None
            else canonical_identifier(domain.domain_id, "domain_id")
        )
        self.mesh = mesh
        self.geometry = geometry
        self.schedule = schedule
        self.domain = domain
        self.source = source
        self.cell_regions = regions
        self.junction_vertices = junctions
        self.protected_features = features
        self.scoped_fidelity = tuple(scoped)
        self.fidelity_tolerance = (
            None if fidelity_tolerance is None else float(fidelity_tolerance)
        )
        self.maximum_normal_deviation = normal_limit
        self.fidelity_witness = fidelity_witness
        self.feature_source = feature_source
        self.layer_core_root_geometry = layer_core_root_geometry
        self.source_coverage_witness = source_coverage_witness
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_id = cell_geometry_id(geometry)
        self.domain_id = domain_id
        self.source_id = source_id_
        self.source_revision = source_revision_
        self.cell_global_ids = cell_ids
        self.request_id = canonical_fingerprint(
            {
                "kind": "mesh-certification-inputs",
                "schedule": schedule.schedule_id,
                "mesh": self.mesh_id,
                "topology": self.topology_id,
                "geometry": self.geometry_id,
                "domain": domain_id,
                "regions": None if regions is None else array_tree_fingerprint(regions),
                "source": (source_id_, source_revision_),
                "fidelity_tolerance": self.fidelity_tolerance,
                "maximum_normal_deviation": normal_limit,
                "fidelity_witness": None
                if fidelity_witness is None
                else fidelity_witness.witness_id,
                "source_coverage_witness": None
                if source_coverage_witness is None
                else source_coverage_witness.witness_id,
                "feature_source": None
                if feature_source is None
                else (
                    feature_source.source_id,
                    feature_source.source_revision,
                ),
                **(
                    {}
                    if layer_core_root_geometry is None
                    else {"layer_core_root_geometry": layer_core_root_geometry.root_id}
                ),
                "junctions": None
                if junctions is None
                else array_tree_fingerprint(junctions),
                "protected_features": (
                    None
                    if features is None
                    else tuple(value.feature_id for value in features)
                ),
                "scoped_fidelity": tuple(
                    (
                        query.source_id,
                        query.source_revision,
                        witness.witness_id,
                        tolerance,
                    )
                    for query, witness, tolerance in scoped
                ),
            }
        )

    @checked
    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        domain: CoverageDomain | None = None,
        source: SourceBoundaryQuery | None = None,
        validated: tuple[ValidatedSourceFidelityWitness, ...] = (),
    ) -> tuple[ValidatedSourceFidelityWitness, ...]:
        """Require these operands to match and return their witness validations.

        ``validated`` holds validations already made for exactly these operands;
        a witness with one is not validated again. The returned validations
        (retained witness first, then scoped witnesses) are handed on explicitly
        to the consumers of the same operands.
        """
        if (
            not _same_tree_content(mesh, self.mesh)
            or not _same_tree_content(geometry, self.geometry)
            or (None if domain is None else not _same_tree_content(domain, self.domain))
            or (None if source is None else not _same_tree_content(source, self.source))
            or mesh.mesh_id != self.mesh_id
            or mesh.topology_id != self.topology_id
            or cell_geometry_id(geometry) != self.geometry_id
            or (None if domain is None else domain.domain_id) != self.domain_id
            or (None if source is None else (source.source_id, source.source_revision))
            != (None if self.source is None else (self.source_id, self.source_revision))
        ):
            raise ValueError(
                "MeshCertificationInputs no longer match the retained scientific content."
            )
        tokens: list[ValidatedSourceFidelityWitness] = []
        if self.fidelity_witness is not None:
            if source is None:
                raise ValueError("A retained fidelity witness requires its source query.")
            tokens.append(
                validate_source_fidelity_witness(
                    mesh,
                    geometry,
                    source,
                    validated_fidelity_witness(
                        validated, mesh, geometry, source, self.fidelity_witness
                    ),
                )
            )
        tokens.extend(
            token
            for _, token, _ in _validate_scoped_fidelity(
                mesh, geometry, self.scoped_fidelity, validated
            )
        )
        _prepare_features(
            self.protected_features,
            self.schedule,
            self.source_id,
            self.source_revision,
            self.feature_source,
        )
        _require_coverage_witness_binding(
            self.source_coverage_witness,
            mesh,
            geometry,
            domain,
            self.fidelity_witness,
        )
        return tuple(tokens)


def validate_fidelity_witnesses(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery | None,
    fidelity_witness: SourceFidelityWitness | None,
    scoped_fidelity: tuple[ScopedFidelityInput, ...],
    /,
    *,
    validated: tuple[ValidatedSourceFidelityWitness, ...] = (),
) -> tuple[ValidatedSourceFidelityWitness, ...]:
    """Validate a certification request's fidelity witnesses once, to hand on.

    The result is passed as ``validated`` to :class:`MeshCertificationInputs`
    and its consumers for these same operands. A witness without its source is
    left to the inputs, which refuse it. Existing validations are reused only
    when they bind the same operand objects and the same witness.
    """
    tokens = []
    if fidelity_witness is not None and source is not None:
        tokens.append(
            validate_source_fidelity_witness(
                mesh,
                geometry,
                source,
                validated_fidelity_witness(
                    validated, mesh, geometry, source, fidelity_witness
                ),
            )
        )
    for query, witness, _ in scoped_fidelity:
        tokens.append(
            validate_source_fidelity_witness(
                mesh,
                geometry,
                query,
                validated_fidelity_witness(
                    (*validated, *tokens), mesh, geometry, query, witness
                ),
            )
        )
    return tuple(tokens)


__all__ = [
    "MeshCertificationInputs",
    "ScopedFidelityInput",
    "validate_fidelity_witnesses",
    "validated_fidelity_witness",
]
