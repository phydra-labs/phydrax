#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authored planar source placement for a moving-mesh epoch."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._curve_boundary import ParametricCurveBoundarySource, SegmentChartCurve
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..geometry.simplicial._regions import PlanarMeshRegion, SegmentMesh
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._association import (
    GeometryAssociation,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._certification_inputs import MeshCertificationInputs, validate_fidelity_witnesses
from ._controls import ProtectedFeature
from ._plc_planar_support import planar_region_domain
from ._scope import MeshingEntityKind, MeshingScope
from .providers._native_sources import NativePlanarSource


if TYPE_CHECKING:
    from ._result import CellMeshingResult


def _rebuild_planar_source(source: NativePlanarSource, /) -> NativePlanarSource:
    rebuilt = NativePlanarSource(
        source.region,
        source.declared_source_revision,
        embedded=source.embedded,
        region_scope_entity_ids=source.region_scope_entity_ids,
        vertex_entity_ids=source.vertex_entity_ids,
        edge_entity_ids=source.edge_entity_ids,
        embedded_vertex_entity_ids=None
        if source.embedded is None
        else source.embedded_vertex_entity_ids,
        embedded_edge_entity_ids=None
        if source.embedded is None
        else source.embedded_edge_entity_ids,
    )
    if (
        rebuilt.binding_id != source.binding_id
        or rebuilt.source_revision != source.source_revision
        or rebuilt.region_entity_id != source.region_entity_id
        or rebuilt.scope_entity_key_banks != source.scope_entity_key_banks
        or rebuilt.scope_entity_set_ids != source.scope_entity_set_ids
        or any(
            not np.array_equal(first, second)
            for first, second in zip(
                rebuilt.scope_entity_id_banks, source.scope_entity_id_banks, strict=True
            )
        )
    ):
        raise ValueError("Planar source fields do not match their retained binding.")
    return rebuilt


class NativePlanarMotionSource(StrictModule, NonTrainableState):
    """An authored source and its explicitly translated successor.

    The successor is reconstructed from source strata, never from a mesh.  The
    translation uses the same binary64 coordinate operation as motion proposals.
    """

    source: NativePlanarSource
    successor: NativePlanarSource
    translation: np.ndarray
    transition_id: str = eqx.field(static=True)

    def __init__(self, source: NativePlanarSource, translation: ArrayLike, /) -> None:
        if not isinstance(source, NativePlanarSource):
            raise TypeError("source must be NativePlanarSource.")
        _rebuild_planar_source(source)
        shift = np.asarray(translation, dtype=np.float64)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("translation must contain two finite coordinates.")
        shift = np.array(shift, copy=True)
        shift.setflags(write=False)
        region = source.region
        edges = np.asarray(region.edges, dtype=np.int64)
        offsets = np.asarray(region.loop_offsets, dtype=np.int64)
        loops = tuple(
            edges[offsets[i] : offsets[i + 1], 0].tolist()
            for i in range(offsets.size - 1)
        )
        moved_region = PlanarMeshRegion(
            np.asarray(region.vertices, dtype=np.float64) + shift,
            loops,
            feature_id=source.source_id,
        )
        embedded = source.embedded
        moved_embedded = (
            None
            if embedded is None
            else SegmentMesh(
                np.asarray(embedded.vertices, dtype=np.float64) + shift,
                np.asarray(embedded.edges),
                source_id=embedded.source_id,
            )
        )
        successor = NativePlanarSource(
            moved_region,
            source.declared_source_revision,
            embedded=moved_embedded,
            region_scope_entity_ids=source.region_scope_entity_ids,
            vertex_entity_ids=source.vertex_entity_ids,
            edge_entity_ids=source.edge_entity_ids,
            embedded_vertex_entity_ids=None
            if embedded is None
            else source.embedded_vertex_entity_ids,
            embedded_edge_entity_ids=None
            if embedded is None
            else source.embedded_edge_entity_ids,
        )
        self.source = source
        self.successor = successor
        self.translation = shift
        self.transition_id = canonical_fingerprint(
            {
                "kind": "native-planar-motion-source",
                "source": source.binding_id,
                "successor": successor.binding_id,
                "translation": array_tree_fingerprint((shift,)),
            }
        )

    @classmethod
    def from_stored_fields(
        cls: type[NativePlanarMotionSource], value: NativePlanarMotionSource, /
    ) -> NativePlanarMotionSource:
        if type(value) is not cls:
            raise TypeError(
                "stored source transition must have the exact registered type."
            )
        _rebuild_planar_source(value.successor)
        rebuilt = cls(value.source, value.translation)
        if (
            rebuilt.transition_id != value.transition_id
            or rebuilt.successor.binding_id != value.successor.binding_id
        ):
            raise ValueError(
                "Stored planar source transition differs from its authored placement."
            )
        return rebuilt


def _key_map(
    transition: NativePlanarMotionSource,
) -> dict[tuple[str, tuple[str, ...]], tuple[str, tuple[str, ...]]]:
    result = {}
    for dimension in range(3):
        for identifier in transition.source.scope_entity_ids(dimension):
            old = transition.source.source_entity_key_for_scope(
                dimension, int(identifier)
            )
            new = transition.successor.source_entity_key_for_scope(
                dimension, int(identifier)
            )
            result[old] = new
    return result


def _translated_query(
    transition: NativePlanarMotionSource, query: ParametricCurveBoundarySource, /
) -> ParametricCurveBoundarySource:
    old = transition.source
    successor = transition.successor
    old_edges = {
        old.source_entity_key_for_scope(1, int(identifier)): int(identifier)
        for identifier in old.scope_entity_ids(1)
    }
    if any(
        key not in old_edges
        for key in zip(query.curve_entity_ids, query.occurrence_paths, strict=True)
    ):
        raise ValueError(
            "Curve query contains a source entity absent from the authored source."
        )
    requested = np.asarray(
        [
            old_edges[(entity, path)]
            for entity, path in zip(
                query.curve_entity_ids, query.occurrence_paths, strict=True
            )
        ],
        dtype=np.int64,
    )
    old.validate_curve_boundary_query(query, requested)
    points, edges, vertex_ids, edge_ids = successor._curve_source_arrays()
    selected = np.flatnonzero(np.isin(edge_ids, requested))
    curves = []
    keys = []
    endpoints = []
    bounds = []
    for row in selected:
        first, last = (int(value) for value in edges[row])
        curves.append(SegmentChartCurve(points[first], points[last], 0.0, 1.0))
        keys.append(successor.source_entity_key_for_scope(1, int(edge_ids[row])))
        endpoints.append(
            (
                successor.source_entity_key_for_scope(0, int(vertex_ids[first])),
                successor.source_entity_key_for_scope(0, int(vertex_ids[last])),
            )
        )
        first_point = tuple(float(value) for value in points[first, :])
        last_point = tuple(float(value) for value in points[last, :])
        bounds.append(((first_point, first_point), (last_point, last_point)))
    result = ParametricCurveBoundarySource(
        tuple(curves),
        tuple((0.0, 1.0) for _ in curves),
        source_id=successor.source_id,
        source_revision=successor.source_revision,
        curve_entity_ids=tuple(key[0] for key in keys),
        occurrence_paths=tuple(key[1] for key in keys),
        chart_ids=tuple(key[0] for key in keys),
        endpoint_source_keys=tuple(endpoints),
        endpoint_point_bounds=tuple(bounds),
    )
    successor.validate_curve_boundary_query(result, requested)
    return result


def _rounding_bound(points: np.ndarray, shift: np.ndarray, /) -> float:
    maximum = Fraction(0)
    for point in np.asarray(points, dtype=np.float64):
        moved = point + shift
        squared = sum(
            (
                Fraction.from_float(float(value))
                + Fraction.from_float(float(delta))
                - Fraction.from_float(float(target))
            )
            ** 2
            for value, delta, target in zip(point, shift, moved, strict=True)
        )
        maximum = max(maximum, squared)
    if maximum == 0:
        return 0.0
    upper = fraction_sqrt_interval(maximum)[1]
    result = float(upper)
    if Fraction.from_float(result) < upper:
        result = float(np.nextafter(result, np.inf))
    return result


def renew_planar_motion_source(
    source: CellMeshingResult,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    transition: NativePlanarMotionSource,
    /,
) -> tuple[
    MeshCertificationInputs,
    tuple[ValidatedSourceFidelityWitness, ...],
    tuple[GeometryAssociation, ...],
]:
    """Rebind retained planar claims after authenticating the coordinate square."""
    report = source.certification
    if report is None:
        raise ValueError(
            "Source-authorized motion requires retained source certification."
        )
    report.require_current(source.mesh, source.geometry)
    original = report.inputs
    NativePlanarMotionSource.from_stored_fields(transition)
    old, successor = transition.source, transition.successor
    if (
        not isinstance(original.feature_source, NativePlanarSource)
        or original.feature_source.binding_id != old.binding_id
    ):
        raise ValueError(
            "Motion source transition does not bind the retained authored source."
        )
    if (
        original.source_id != old.source_id
        or original.source_revision != old.source_revision
    ):
        raise ValueError("Motion source transition binds a stale source revision.")
    if not np.array_equal(
        np.asarray(mesh.coordinates),
        np.asarray(source.mesh.coordinates) + transition.translation,
    ):
        raise ValueError(
            "Motion coordinates do not commute with the declared source translation."
        )
    if (
        original.domain is None
        or original.domain.domain_id != planar_region_domain(old).domain_id
    ):
        raise ValueError("Retained coverage domain is not the authored planar source.")
    if source.surface_source is not None or source.region_boundary_evidence:
        raise ValueError(
            "Planar translation cannot transport additional surface or region-boundary authority."
        )
    if (
        original.source is not None
        or original.fidelity_witness is not None
        or original.source_coverage_witness is not None
    ):
        raise ValueError(
            "This planar motion route cannot transport the retained source witness."
        )
    mapping = _key_map(transition)
    features = []
    for feature in original.protected_features or ():
        scope = feature.scope
        if scope.entity_kind is not MeshingEntityKind.GEOMETRY or scope.side_records:
            raise ValueError("Planar motion requires unsided authored geometry scopes.")
        new_scope = MeshingScope(
            successor.source_id,
            successor.source_revision,
            scope.entity_kind,
            scope.entity_dimension,
            successor.scope_entity_set_id(scope.entity_dimension),
            np.asarray(scope.entity_ids),
        )
        features.append(
            ProtectedFeature(
                new_scope,
                feature.feature_kind,
                maximum_deviation=feature.maximum_deviation,
                hard=feature.hard,
            )
        )
    scoped = []
    for query, witness, tolerance in original.scoped_fidelity:
        if not isinstance(query, ParametricCurveBoundarySource) or not isinstance(
            witness, (SourceBoundaryChartWitness, SourceBoundaryPointWitness)
        ):
            raise ValueError(
                "Planar motion requires authored point or curve-chart fidelity witnesses."
            )
        new_query = _translated_query(transition, query)
        if any(
            key not in mapping
            for key in zip(
                witness.source_entity_ids, witness.occurrence_paths, strict=True
            )
        ):
            raise ValueError("Fidelity witness contains an unknown source occurrence.")
        new_entities = tuple(
            mapping[(entity, path)][0]
            for entity, path in zip(
                witness.source_entity_ids, witness.occurrence_paths, strict=True
            )
        )
        if isinstance(witness, SourceBoundaryPointWitness):
            new_witness = SourceBoundaryPointWitness(
                mesh_id=mesh.mesh_id,
                geometry_id=cell_geometry_id(geometry),
                topology_id=mesh.topology_id,
                source_id=successor.source_id,
                source_revision=successor.source_revision,
                source_query_id=new_query.query_id,
                vertex_global_ids=witness.vertex_global_ids,
                source_entity_ids=new_entities,
                occurrence_paths=witness.occurrence_paths,
            )
        elif isinstance(witness, SourceBoundaryChartWitness):
            if witness.cad_plc is not None or witness.cad_curve_association is not None:
                raise ValueError(
                    "CAD chart witnesses require their own source transition."
                )
            if any(
                key not in mapping
                for key in zip(
                    witness.source_chart_ids, witness.occurrence_paths, strict=True
                )
            ):
                raise ValueError("Fidelity chart contains an unknown source occurrence.")
            new_charts = tuple(
                mapping[(chart, path)][0]
                for chart, path in zip(
                    witness.source_chart_ids, witness.occurrence_paths, strict=True
                )
            )
            new_witness = SourceBoundaryChartWitness(
                mesh_id=mesh.mesh_id,
                geometry_id=cell_geometry_id(geometry),
                topology_id=mesh.topology_id,
                source_id=successor.source_id,
                source_revision=successor.source_revision,
                source_query_id=new_query.query_id,
                chart_dimension=witness.chart_dimension,
                reference_dimension=witness.reference_dimension,
                cell_global_ids=witness.cell_global_ids,
                entity_global_ids=witness.entity_global_ids,
                local_entity_indices=witness.local_entity_indices,
                target_reference_vertices=witness.target_reference_vertices,
                target_reference_rational_tokens=witness.target_reference_rational_tokens,
                source_chart_ids=new_charts,
                source_entity_ids=new_entities,
                occurrence_paths=witness.occurrence_paths,
                source_parameters=witness.source_parameters,
                source_parameter_rational_tokens=witness.source_parameter_rational_tokens,
                orientations=witness.orientations,
            )
        else:
            raise ValueError(
                "Planar motion requires point or curve-chart fidelity witnesses."
            )
        scoped.append((new_query, new_witness, tolerance))
    validated = validate_fidelity_witnesses(mesh, geometry, None, None, tuple(scoped))
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        original.schedule,
        source_id=successor.source_id,
        source_revision=successor.source_revision,
        domain=planar_region_domain(successor),
        cell_regions=original.cell_regions,
        maximum_normal_deviation=original.maximum_normal_deviation,
        junction_vertices=original.junction_vertices,
        protected_features=tuple(features),
        scoped_fidelity=tuple(scoped),
        feature_source=successor,
        validated=validated,
    )
    source_points = old._curve_source_arrays()[0]
    rounding = _rounding_bound(
        np.asarray(source.mesh.coordinates), transition.translation
    ) + _rounding_bound(source_points, transition.translation)
    if rounding:
        rounding = float(np.nextafter(rounding, np.inf))
    associations = []
    for association in source.associations:
        if association.source_boundary_witness_id is not None:
            raise ValueError(
                "Moving source-bound face pieces require a renewed source witness."
            )
        if (association.source_id, association.source_revision) != (
            old.source_id,
            old.source_revision,
        ):
            raise ValueError(
                "Planar motion cannot change an unrelated association authority."
            )
        if any(side != (None, None) for side in association.source_region_sides):
            raise ValueError(
                "Planar motion cannot transport separate region-side declarations."
            )
        if any(
            key not in mapping
            for key in zip(
                association.source_entity_ids,
                association.source_occurrence_paths,
                strict=True,
            )
        ):
            raise ValueError("Association contains an unknown source occurrence.")
        keys = tuple(
            mapping[(entity, path)]
            for entity, path in zip(
                association.source_entity_ids,
                association.source_occurrence_paths,
                strict=True,
            )
        )
        dimensions = [
            dimension
            for dimension in range(source.mesh.topological_dimension + 1)
            if source.mesh.entity_set(dimension).entity_set_id
            == association.target_entity_set_id
        ]
        if len(dimensions) != 1:
            raise ValueError("Association does not bind a unique source mesh entity set.")
        residuals = np.asarray(association.residuals, dtype=np.float64).copy()
        for row, role in enumerate(association.source_roles):
            if role is not GeometrySourceEntityRole.REGION and rounding:
                residuals[row] = np.nextafter(residuals[row] + rounding, np.inf)
        associations.append(
            GeometryAssociation(
                association.association_kind,
                successor.source_id,
                successor.source_revision,
                association.target_entity_set_id,
                association.target_global_ids,
                tuple(key[0] for key in keys),
                residuals,
                resolved=association.resolved,
                ambiguous=association.ambiguous,
                exact=association.exact and not np.any(residuals),
                source_dimensions=None
                if np.all(np.asarray(association.source_dimensions) < 0)
                else association.source_dimensions,
                source_indices=None
                if np.all(np.asarray(association.source_dimensions) < 0)
                else association.source_indices,
                source_roles=association.source_roles,
                source_occurrence_paths=tuple(key[1] for key in keys),
                source_region_sides=association.source_region_sides,
                source_parameter_intervals=association.source_parameter_intervals,
                target_parameter_intervals=association.target_parameter_intervals,
                source_parameter_points=association.source_parameter_points,
                parameters=association.parameters,
                orientations=association.orientations,
                parent_dimensions=np.full(len(keys), dimensions[0], dtype=np.int8),
                parent_ids=association.target_global_ids,
                parent_association_id=association.association_id,
                provenance=GeometryAssociationProvenance.LINEAGE,
            )
        )
    return inputs, validated, tuple(associations)
