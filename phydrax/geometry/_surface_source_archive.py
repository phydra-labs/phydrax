#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cold reconstruction of a source-owned surface chart archive."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

from .._model._structure import (
    _FiniteElementSpecArchiveProjection,
    MeshingDomainArchiveRecipe,
    SurfaceSourceChartsArchiveRecipe,
    restore_row_record_recipes,
)
from .._physical import SpatialCoordinateContract
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import (
    _require_stored_chart_term_types,
    CellGeometryRestrictionSource,
    CellVertexGeometryElement,
    LayerColumnCellGeometryElement,
    PerCellRestrictedCellGeometryElement,
    RationalComposedCellGeometryElement,
    ScalarCoordinateElement,
    SplineCellGeometryElement,
)
from ..interchange._cad_archive import _reconstruct_brep_graph
from ..meshing._association import GeometryAssociation, GeometryAssociationArchiveRecipe
from ..typing import parse
from ._domain_boundary import MeshingDomainBoundarySource
from ._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSourceAccuracy,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
)
from ._surface_source_support import _SplineSurfaceCoordinateMap, SurfaceSourceCharts
from .brep._model import BRepModel


if TYPE_CHECKING:
    from ..discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ..discretization._mapped_power_geometry import MappedPowerCellGeometrySource
    from ..meshing._hex_cut_generation import _CutCoordinateMap
    from ._mapped_reference_domain import MappedReferenceDomain


def _archive_array(value: object, /) -> np.ndarray:
    if not isinstance(value, (np.ndarray, jax.Array, np.generic)):
        raise TypeError("Stored source archive data must be a numeric array.")
    return np.asarray(value)


def _domain(value: MeshingDomainArchiveRecipe, /) -> MeshingDomain:
    from .reconstruction._core import _ReconstructedTrianglePatch

    if (
        type(value) is not MeshingDomainArchiveRecipe
        or type(value.corner_points_authored) is not bool
    ):
        raise TypeError(
            "Source chart archive requires a canonical meshing-domain recipe."
        )
    patches = []
    for patch in value.patches:
        if type(patch) is not MeshingSurfacePatch:
            raise TypeError("Source domain patch must use its registered owner.")
        loops = []
        for loop in patch.loops:
            uses = []
            for use in loop:
                if type(use) not in (PatchCurveUse, PatchPoleUse):
                    raise TypeError("Source domain patch has an unregistered trim use.")
                curve = _reconstruct_brep_graph(use.pcurve)
                if type(use) is PatchCurveUse:
                    uses.append(
                        PatchCurveUse(
                            use.curve, curve, use.first, use.last, sense=use.sense
                        )
                    )
                elif type(use) is PatchPoleUse:
                    uses.append(
                        PatchPoleUse(
                            use.corner, curve, use.first, use.last, sense=use.sense
                        )
                    )
                else:
                    raise TypeError("Source domain patch has an unregistered trim use.")
            loops.append(tuple(uses))
        if type(patch.surface) is _ReconstructedTrianglePatch:
            surface = _ReconstructedTrianglePatch(_archive_array(patch.surface.corners))
            if surface.exact_corners != patch.surface.exact_corners:
                raise ValueError("Stored triangle patch differs from its exact corners.")
        else:
            surface = _reconstruct_brep_graph(patch.surface)
        patches.append(
            MeshingSurfacePatch(
                surface,
                tuple(loops),
                orientation=patch.orientation,
            )
        )
    curves = tuple(
        MeshingDomainCurve(item.start, item.end)
        for item in value.curves
        if type(item) is MeshingDomainCurve
    )
    regions = tuple(
        MeshingDomainRegion(item.name, item.boundary)
        for item in value.regions
        if type(item) is MeshingDomainRegion
    )
    if len(curves) != len(value.curves) or len(regions) != len(value.regions):
        raise TypeError("Source domain has unregistered curve or region records.")
    authority = (
        None
        if value.brep_authority is None
        else _reconstruct_brep_graph(value.brep_authority)
    )
    if authority is not None and type(authority) is not BRepModel:
        raise TypeError("Source domain B-Rep authority must retain its registered owner.")
    domain = MeshingDomain(
        tuple(patches),
        curves,
        value.corner_count,
        source_id=value.source_id,
        source_revision=value.source_revision,
        regions=regions,
        tolerance=value.tolerance,
        accuracy=parse(value.accuracy, MeshingSourceAccuracy, "accuracy"),
        source_indices=value.source_indices,
        source_occurrences=value.source_occurrences,
        region_source_indices=value.region_source_indices,
        region_source_occurrences=value.region_source_occurrences,
        source_kinds=value.source_kinds,
        authority_id=value.authority_id,
        brep_authority=authority,
        corner_points=_archive_array(value.corner_points)
        if value.corner_points_authored
        else None,
    )
    if domain.domain_id != value.expected_domain_id or not np.array_equal(
        np.asarray(domain.corner_points), np.asarray(value.corner_points)
    ):
        raise ValueError("Reconstructed meshing domain differs from its source recipe.")
    domain.require_current(domain.source_id, domain.source_revision)
    return domain


def _element(value: object, /) -> object:
    if type(value) is _FiniteElementSpecArchiveProjection:
        return value.rebuild()
    from .._model._structure import model_archive_projection
    from ..discretization.fem._reference import FiniteElementSpec

    if type(value) is FiniteElementSpec:
        projected = model_archive_projection(value)
        if type(projected) is not _FiniteElementSpecArchiveProjection:
            raise ValueError(
                "Finite element source did not yield its canonical archive recipe."
            )
        return projected.rebuild()
    if type(value) is CellVertexGeometryElement:
        return CellVertexGeometryElement(value.cell_kind, value.local_dof_count)
    if type(value) is LayerColumnCellGeometryElement:
        wall = _element(value.wall_element)
        corner = _element(value.corner_element)
        if type(wall) is not FiniteElementSpec or type(corner) is not FiniteElementSpec:
            raise TypeError("Stored layer-column basis lost its finite-element owners.")
        rebuilt = LayerColumnCellGeometryElement(wall, fiber_graph=value.fiber_graph)
        if (
            rebuilt.element_id != value.element_id
            or rebuilt.corner_element.element_id != corner.element_id
            or any(
                getattr(rebuilt, name) != getattr(value, name)
                for name in (
                    "cell_kind",
                    "conformity",
                    "local_dof_count",
                    "degree",
                    "topological_dimension",
                    "station_axis",
                    "fiber_graph",
                )
            )
        ):
            raise ValueError("Stored layer-column basis differs from its exact inputs.")
        return rebuilt
    if type(value) is SplineCellGeometryElement:
        rebuilt = SplineCellGeometryElement(
            value.anchor_ids,
            value.parameter_bounds,
            value.exact_extraction_operator,
            value.weights,
            value.source_id,
            value.source_revision,
            degrees=value.degrees,
            exact_weights=value.exact_weights,
        )
        if rebuilt.element_id != value.element_id:
            raise ValueError(
                "Stored spline geometry element differs from its exact source."
            )
        return rebuilt
    if type(value) is RationalComposedCellGeometryElement:
        source = _element(value.source_element)
        if not isinstance(source, ScalarCoordinateElement):
            raise TypeError("Stored source basis is not a scalar coordinate element.")
        return RationalComposedCellGeometryElement(
            source, value.cell_kind, value.chart_arguments
        )
    if type(value) is PerCellRestrictedCellGeometryElement:
        templates = {}
        charts = []
        for row, terms in enumerate(value.chart_coefficients):
            _require_stored_chart_term_types(terms)
            if terms not in templates:
                templates[terms] = value.chart_arguments(row)
            charts.append(templates[terms])
        source = _element(value.source_element)
        if not isinstance(source, ScalarCoordinateElement):
            raise TypeError("Stored source basis is not a scalar coordinate element.")
        return PerCellRestrictedCellGeometryElement.from_stored_fields(
            value, source, charts
        )
    raise TypeError(
        f"No source chart coordinate-element owner for {type(value).__name__}."
    )


def _mapped_reference(value: object, /) -> MappedReferenceDomain:
    from ..meshing._archive_layer_columns import LayerRecordArchiveRecipe
    from ._domain_coverage import PiecewiseLinearDomain
    from ._mapped_reference_domain import MappedReferenceDomain

    if type(value) is not MappedReferenceDomain:
        raise TypeError("Expected a canonical mapped-reference domain.")
    root = value.reference_domain
    reference = PiecewiseLinearDomain(
        root.vertices,
        root.facets,
        root.facet_regions,
        root.region_ids,
        source_id=root.source_id,
        region_occurrence_paths=root.region_occurrence_paths,
    )
    if (
        reference.domain_id != root.domain_id
        or reference.source_revision != root.source_revision
    ):
        raise ValueError("Stored reference domain differs from its source.")
    reference_mesh = CellMesh.from_stored_fields(value.reference_mesh)
    stored_geometry = value.source_geometry
    if type(stored_geometry) is LayerRecordArchiveRecipe:
        stored_geometry = stored_geometry.restore()
    source_geometry = _geometry(stored_geometry, reference_mesh)
    result = MappedReferenceDomain(
        reference,
        reference_mesh,
        source_geometry,
        value.cell_regions,
        source_id=value.source_id,
        source_revision=value.source_revision,
        coordinate_contract=_reconstruct_brep_graph(value.coordinate_contract),
        root_vertex_occurrence_paths=value.root_vertex_occurrence_paths,
        root_cell_occurrence_paths=value.root_cell_occurrence_paths,
        reference_domain_vertex_root_ids=(
            None
            if value.reference_domain_vertex_root_ids is None
            else np.asarray(value.reference_domain_vertex_root_ids, dtype=np.int64)
        ),
    )
    if result.domain_id != value.domain_id:
        raise ValueError("Stored mapped-reference domain differs from its source.")
    return result


def _power_map(
    value: object, mesh: CellMesh, /
) -> ExactPowerCellGeometrySource | MappedPowerCellGeometrySource:
    from .._model._structure import PiecewiseLinearComplexArchiveRecipe
    from ..discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ..discretization._mapped_power_geometry import MappedPowerCellGeometrySource
    from ..meshing._model_codec import restore_model_archive_value
    from ..meshing._plc_source import PiecewiseLinearComplex
    from ._mapped_reference_domain import MappedReferenceDomain

    if type(value) is MappedPowerCellGeometrySource:
        reference = _power_map(value.reference_source, mesh)
        if type(reference) is not ExactPowerCellGeometrySource:
            raise TypeError("Mapped power geometry lost its exact reference source.")
        return MappedPowerCellGeometrySource.from_stored_fields(
            value, reference_source=reference
        )
    if type(value) is not ExactPowerCellGeometrySource:
        raise TypeError("Expected a registered exact power source.")
    domain = value.domain_source
    if type(domain) in (PiecewiseLinearComplex, PiecewiseLinearComplexArchiveRecipe):
        domain = restore_model_archive_value(domain)
    elif type(domain) is MappedReferenceDomain:
        domain = _mapped_reference(domain)
    elif domain is not None:
        raise TypeError("Exact power geometry lost its source domain owner.")
    from ..discretization._periodic_cell import PeriodicCell

    periodic = None
    if value.periodic_source_context is not None and mesh.periodic_topology is not None:
        periodic = mesh.periodic_topology.cell
        if not isinstance(periodic, PeriodicCell):
            raise TypeError("Periodic power geometry requires a PeriodicCell owner.")
    return ExactPowerCellGeometrySource.from_stored_fields(
        value,
        domain_source=domain,
        target_periodic_cell=periodic,
    )


def _cut_map(value: object, target: CellMesh, /) -> _CutCoordinateMap:
    from fractions import Fraction

    from ..meshing._hex_cut_generation import _CutCoordinateMap
    from ..meshing._hex_cut_preparation import BalancedCutGridPlan
    from ..meshing._hex_frame_records import HexFrameIntegerLayout
    from ..meshing._hex_schedule import NativeHexGridSchedule
    from ..meshing._model_codec import restore_model_archive_value
    from ..meshing._plc_source import PiecewiseLinearComplex

    if type(value) is not _CutCoordinateMap:
        raise TypeError("Expected a canonical cut-coordinate map.")
    complex_ = restore_model_archive_value(value.source_complex)
    if type(complex_) is not PiecewiseLinearComplex:
        raise TypeError("Cut-coordinate source lost its PLC owner.")
    source_mesh = CellMesh.from_stored_fields(value.source_mesh)
    source_geometry = _geometry(value.source_geometry, source_mesh)
    frame = value.plan.frame_layout
    if frame is not None:
        if type(frame) is not HexFrameIntegerLayout:
            raise TypeError("Cut-coordinate frame has an unregistered owner.")
        if any(
            not isinstance(knot, Fraction)
            for axes in frame.block_axis_knots
            for axis in axes
            for knot in axis
        ):
            raise TypeError("Cut-coordinate frame knots must remain exact fractions.")
        rebuilt_frame = HexFrameIntegerLayout(
            frame.cell_global_ids,
            frame.cell_block_indices,
            frame.cell_integer_coordinates,
            np.asarray(frame.block_axes, dtype=np.float64),
            np.asarray(frame.block_origins, dtype=np.float64),
            frame.block_axis_knots,
            frame.face_connections,
            frame.face_transitions,
            frame.face_offsets,
            frame.boundary_face_parity,
            frame.source_complex_id,
            frame.source_mesh_id,
            frame.source_topology_id,
            frame.schedule_id,
            frame.frame_id,
        )
        if rebuilt_frame.layout_id != frame.layout_id:
            raise ValueError("Cut-coordinate frame differs from its exact source.")
        frame = rebuilt_frame
    schedule = NativeHexGridSchedule(
        value.schedule.route,
        minimum_scaled_jacobian=value.schedule.minimum_scaled_jacobian,
        minimum_mean_ratio=value.schedule.minimum_mean_ratio,
        maximum_aspect_ratio=value.schedule.maximum_aspect_ratio,
    )
    if schedule.schedule_id != value.schedule.schedule_id:
        raise ValueError("Cut-coordinate schedule changed.")
    plan = BalancedCutGridPlan(
        value.plan.axis_knots,
        complex_,
        source_mesh,
        schedule,
        value.plan.target_size,
        source_id=value.source_id,
        source_revision=value.source_revision,
        frame_axes=np.asarray(value.plan.frame_axes, dtype=np.float64),
        frame_origin=np.asarray(value.plan.frame_origin, dtype=np.float64),
        frame_id=value.plan.frame_id,
        frame_layout=frame,
    )
    if plan.plan_id != value.plan.plan_id:
        raise ValueError("Cut-coordinate plan differs from its exact source.")
    rebuilt = _CutCoordinateMap(
        value.source_id,
        value.source_revision,
        complex_,
        plan,
        schedule,
        source_mesh,
        source_geometry,
        value.cell_regions,
        target,
        value.node_keys,
        value.exact_vertices,
    )
    if (
        rebuilt.source_map_id != value.source_map_id
        or not np.array_equal(rebuilt.rounded_vertices, value.rounded_vertices)
        or rebuilt.node_keys != value.node_keys
        or rebuilt.exact_vertices != value.exact_vertices
    ):
        raise ValueError("Cut-coordinate source does not reproduce its exact ancestry.")
    return _CutCoordinateMap.from_stored_fields(rebuilt, target)


def _geometry(value: CellGeometrySpec, mesh: CellMesh, /) -> CellGeometrySpec:
    if type(value) is not CellGeometrySpec:
        raise TypeError("Source chart archive requires a CellGeometrySpec.")
    source_map = value.source_map
    if type(source_map) is _SplineSurfaceCoordinateMap:
        original = source_map
        source_map = _SplineSurfaceCoordinateMap(
            original.source_id,
            original.source_revision,
            original.rounded_vertices,
            original.exact_coordinates,
            value.coordinates,
            original.coefficient_keys,
            original.coefficient_authorities,
        )
        if source_map.source_map_id != original.source_map_id:
            raise ValueError("Stored spline coordinate source changed.")
    elif source_map is not None:
        from .._model._structure import (
            AffinePlcGeometrySourceArchiveRecipe,
            model_archive_projection,
        )
        from ..discretization._exact_algebraic_geometry import (
            ExactPeriodicAlgebraicSource,
        )
        from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
        from ..discretization._exact_power_geometry import ExactPowerCellGeometrySource
        from ..discretization._mapped_power_geometry import MappedPowerCellGeometrySource
        from ..meshing._curving import _ExactCoordinateBankMap
        from ..meshing._dual_extraction import _RestrictedCoordinateMap
        from ..meshing._hex_cut_generation import _CutCoordinateMap
        from ..meshing._hex_dominant import _HexDominantCoordinateMap
        from ..meshing._mixed_adaptation import _RetainedCoordinateBank
        from ..meshing._model_codec import restore_model_archive_value
        from ._affine_plc_geometry import AffinePlcGeometrySource, map_affine_plc_mesh

        if type(source_map) in (
            AffinePlcGeometrySource,
            AffinePlcGeometrySourceArchiveRecipe,
        ):
            source_map = restore_model_archive_value(model_archive_projection(source_map))
            if type(source_map) is not AffinePlcGeometrySource:
                raise TypeError("Affine PLC geometry lost its exact source owner.")
            expected_mesh, _ = map_affine_plc_mesh(
                source_map.domain, source_map.reference_mesh, witness=source_map.witness
            )
            if expected_mesh.mesh_id != mesh.mesh_id:
                raise ValueError("Affine PLC geometry does not bind the current mesh.")
        elif type(source_map) is _ExactCoordinateBankMap:
            source_map = _ExactCoordinateBankMap.from_stored_fields(source_map, mesh=mesh)
        elif type(source_map) is ExactPlcCellGeometrySource:
            source_map = ExactPlcCellGeometrySource.from_stored_fields(source_map)
        elif type(source_map) is ExactPeriodicAlgebraicSource:
            source_map = ExactPeriodicAlgebraicSource.from_stored_fields(source_map)
        elif type(source_map) is _RetainedCoordinateBank:
            source_map = _RetainedCoordinateBank.from_stored_fields(
                source_map, coordinates=value.coordinates
            )
        elif type(source_map) is _RestrictedCoordinateMap:
            original = source_map
            source_mesh = CellMesh.from_stored_fields(original.source_mesh)
            source_geometry = _geometry(original.source_geometry, source_mesh)
            source_map = _RestrictedCoordinateMap(source_mesh, source_geometry, mesh)
            if source_map.source_map_id != original.source_map_id:
                raise ValueError("Stored restricted coordinate source changed.")
            source_map = _RestrictedCoordinateMap.from_stored_fields(
                source_map, target_mesh=mesh
            )
        elif type(source_map) in (
            ExactPowerCellGeometrySource,
            MappedPowerCellGeometrySource,
        ):
            source_map = _power_map(source_map, mesh)
        elif type(source_map) is _CutCoordinateMap:
            source_map = _cut_map(source_map, mesh)
        elif type(source_map) is _HexDominantCoordinateMap:
            original = source_map
            source_mesh = CellMesh.from_stored_fields(original.source_mesh)
            source_geometry = _geometry(original.source_geometry, source_mesh)
            source_map = _HexDominantCoordinateMap(
                source_mesh,
                source_geometry,
                mesh,
                original.exact_vertex_images,
                original.node_vertex_rows,
            )
            if (
                source_map.source_map_id != original.source_map_id
                or source_map.exact_coordinate_bank != original.exact_coordinate_bank
                or not np.array_equal(
                    source_map.rounded_vertices, original.rounded_vertices
                )
            ):
                raise ValueError("Stored hex-dominant coordinate source changed.")
        else:
            raise TypeError(
                f"No source chart coordinate-map owner for {type(source_map).__name__}."
            )
    restriction = None
    if value.restriction_source is not None:
        from ..discretization._nested_reference_types import (
            ReferenceCellMapWitnesses,
            validate_reference_cell_map_witnesses,
        )

        stored = value.restriction_source
        if type(stored) is not CellGeometryRestrictionSource:
            raise TypeError("Source chart restriction must use its registered owner.")
        maps = {}
        for name, witness in stored.block_parent_reference_maps.items():
            if witness is None:
                maps[name] = None
                continue
            if type(witness) is not ReferenceCellMapWitnesses:
                raise TypeError("Source chart restriction lost its exact reference maps.")
            rebuilt = ReferenceCellMapWitnesses(
                witness.target_cell_ids,
                witness.source_cell_ids,
                witness.target_cell_kind,
                witness.source_cell_kind,
                witness.source_reference_vertices,
                witness.source_reference_exact_vertices,
                witness.source_reference_chart_arguments,
            )
            validate_reference_cell_map_witnesses(rebuilt)
            maps[name] = rebuilt
        carrier = None
        if stored.carrier_geometry is not None:
            if stored.carrier_geometry.source_mesh_id is not None and (
                stored.carrier_geometry.source_mesh_id != mesh.mesh_id
                or stored.carrier_geometry.source_topology_id != mesh.topology_id
            ):
                raise ValueError(
                    "Stored carrier geometry requires its actual mesh owner."
                )
            carrier = _geometry(stored.carrier_geometry, mesh)
        restriction = CellGeometryRestrictionSource(
            stored.source_geometry_id,
            stored.source_topology_id,
            stored.block_parent_cell_ids,
            stored.block_parent_vertex_ids,
            block_source_blocks=stored.block_source_blocks,
            carrier_publication=stored.carrier_publication,
            block_parent_reference_maps=maps,
            carrier_geometry=carrier,
        )
        if restriction.restriction_source_id != stored.restriction_source_id:
            raise ValueError("Source chart restriction differs from its exact ancestry.")
    rebuilt = CellGeometrySpec.from_stored_fields(
        value,
        source_mesh=mesh if source_map is not None else None,
        source_map=source_map,
        restriction_source=restriction,
        elements={
            name: _element(element)
            for name, element in zip(value.block_names, value.elements, strict=True)
        },
    )
    if mesh.periodic_topology is not None:
        mesh.periodic_topology.require_exact_geometry(mesh, rebuilt)
    return rebuilt


def _decode_trim_roots(value: object, /) -> tuple:
    """Decode exact source trim chords before the chart owner revalidates them."""
    from fractions import Fraction

    from .._archive_columns import ExactTensorArchiveRecipe
    from ._surface_source_support import SurfaceTrimRoot

    if type(value) is not ExactTensorArchiveRecipe or not value.rational:
        raise TypeError("Surface trim roots require a rational exact tensor.")
    if value.shape != (0,) and (
        len(value.shape) != 2 or value.shape[0] < 1 or value.shape[1] != 9
    ):
        raise ValueError("Stored surface trim-root tensor has a noncanonical shape.")
    rows = value.decode()
    roots = []
    for row in rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 9
            or any(type(value) is not Fraction for value in row)
            or any(value.denominator != 1 or value < 0 for value in row[:3])
        ):
            raise ValueError("Stored surface trim-root row is malformed.")
        roots.append(
            SurfaceTrimRoot(
                int(row[0]),
                int(row[1]),
                int(row[2]),
                row[3],
                row[4],
                (row[5], row[6]),
                (row[7], row[8]),
            )
        )
    return tuple(roots)


def restore_surface_source_charts(
    value: SurfaceSourceChartsArchiveRecipe,
    /,
) -> SurfaceSourceCharts:
    """Rebuild source chart content from registered constructors, not archive claims."""
    if type(value) is not SurfaceSourceChartsArchiveRecipe:
        raise TypeError("Expected a canonical surface-source chart archive recipe.")
    value = restore_row_record_recipes(value)
    domain = _domain(value.domain)
    from ._mesh_certificates import _same_tree_content

    source = value.boundary_source
    if not _same_tree_content(source.domain, value.domain):
        raise ValueError("Boundary query and chart recipe name different domains.")
    boundary = MeshingDomainBoundarySource(
        domain,
        source.patches,
        resolution=source.resolution,
        boundary_occurrences=(
            None
            if source.boundary_occurrences is None
            else _archive_array(source.boundary_occurrences)
        ),
        boundary_cell_ids=(
            None
            if source.boundary_cell_ids is None
            else _archive_array(source.boundary_cell_ids)
        ),
        boundary_patch_ids=(
            None
            if source.boundary_patch_ids is None
            else _archive_array(source.boundary_patch_ids)
        ),
    )
    if boundary.source_scope_id != source.expected_source_scope_id:
        raise ValueError("Reconstructed boundary source differs from its exact scope.")
    boundary.require_current()
    stored_mesh = value.mesh
    if type(stored_mesh) is not CellMesh:
        raise TypeError("Source chart root mesh must be a CellMesh.")
    faces = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in stored_mesh.blocks]
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in stored_mesh.blocks]
    )
    topology = (
        domain.multiregion_surface(
            stored_mesh.coordinates,
            faces,
            value.root_patches,
            vertex_global_ids=stored_mesh.vertex_global_ids,
            cell_global_ids=ids,
        )
        if all(block.cell_kind == "triangle" for block in stored_mesh.blocks)
        else None
    )
    mesh = CellMesh.from_stored_fields(stored_mesh, multiregion_surface=topology)
    geometry = _geometry(value.geometry, mesh)
    contract = _reconstruct_brep_graph(value.coordinate_contract)
    if type(contract) is not SpatialCoordinateContract:
        raise TypeError("Source charts lost their registered coordinate frame.")
    associations: list[GeometryAssociation] = []
    for item in value.associations:
        if type(item) is GeometryAssociationArchiveRecipe:
            associations.append(item.restore())
        elif type(item) is GeometryAssociation:
            associations.append(
                GeometryAssociationArchiveRecipe.from_association(item).restore()
            )
        else:
            raise TypeError("Source charts require registered geometry associations.")
    charts = SurfaceSourceCharts(
        domain,
        mesh,
        geometry,
        contract,
        value.root_parameters,
        value.root_patches,
        tuple(associations),
        boundary,
        source_root_cell_ids=value.source_root_cell_ids,
        boundary_occurrences=value.boundary_occurrences,
        source_piece_ids=value.source_piece_ids.decode(),
        root_parameter_exact=value.root_parameter_exact.decode(),
        trim_roots=_decode_trim_roots(value.trim_roots),
    )
    if charts.source_chart_id != value.expected_source_chart_id:
        raise ValueError("Reconstructed source charts differ from their exact recipe.")
    charts.require_root(mesh, geometry)
    charts.require_current()
    return charts
