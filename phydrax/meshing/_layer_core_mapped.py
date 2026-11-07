#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact reference-root composition for mapped layer/core construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._nested_reference as nested_reference_owner
import phydrax.discretization._periodic_topology as periodic_topology_owner
import phydrax.meshing._association as association_owner
import phydrax.meshing._canonical as canonical_owner
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._organization as organization_owner

from ..discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from ..discretization._cell_geometry_transition_types import (
    _geometry_map_expressions,
    _require_exact_reference_group,
)
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    BernsteinDomain,
    compose_expression,
    constant,
    coordinate_map_expressions,
    derivative,
    evaluate_expression,
    ExactScalar,
    Expression,
    expression_equal,
    ExpressionInput,
    multiply,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    rounded_point,
    scale,
)
from ..discretization._nested_reference_types import (
    exact_reference_cell_vertices,
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
    ReferenceCellPartition,
    validate_reference_cell_partition,
)
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ._scope import MeshingScope


if TYPE_CHECKING:
    import phydrax.meshing.providers._native_sources as sources_owner

    from .._meshcore import (
        TetMesh3D,
        TetMeshArrays,
        TetMeshCandidateQuality,
        TetMeshIdentity,
        TetMeshQuality,
        TetMeshRun,
    )
    from ..discretization._cell_geometry_transition_types import _ExactCellMaps
    from ._contracts import VolumeMeshingSpec
    from ._layer_core_root_geometry import ComputedLayerCoreRoot
    from ._volume_generation import _SourceTables, NativeVolumeSchedule


type _GeometryGroupKey = tuple[str, int, int, str, str]


def _canonical_expressions(
    expressions: tuple[ExpressionInput, ...], /
) -> tuple[Expression, ...]:
    return tuple(
        expression if isinstance(expression, RationalPolynomial) else dict(expression)
        for expression in expressions
    )


@dataclass(frozen=True, slots=True)
class _CellChart:
    target_block_index: int
    target_row: int
    target_cell_id: int
    target_cell_kind: str
    target_vertices: tuple[int, ...]
    source_block_index: int
    source_block_name: str
    source_cell_row: int
    source_cell_id: int
    source_cell_kind: str
    source_region: int
    source_region_id: str
    source_occurrence_path: tuple[str, ...]
    source_reference_vertices: tuple[tuple[Fraction, ...], ...]
    root_element: cell_geometry_owner.ScalarCoordinateElement
    base_element: cell_geometry_owner.ScalarCoordinateElement
    chart_arguments: tuple[Expression, ...]
    source_reference_chart_arguments: tuple[Expression, ...]
    source_route: np.ndarray
    physical_expressions: tuple[Expression, ...]


def _block_order_regions(mesh: CellMesh, sorted_regions: ArrayLike, /) -> np.ndarray:
    identifiers = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    regions = np.asarray(sorted_regions, dtype=np.int64)
    if regions.shape != identifiers.shape:
        raise ValueError(
            "Layer/core region data must cover the exact cell identity bank."
        )
    by_id = dict(zip(identifiers.tolist(), regions.tolist(), strict=True))
    return np.asarray(
        [
            by_id[int(identifier)]
            for block in mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist()
        ],
        dtype=np.int64,
    )


def mapped_source_lipschitz_bound(domain: MappedReferenceDomain, /) -> float:
    """Bound physical length expansion over every exact reference root map."""
    if not isinstance(domain, MappedReferenceDomain):
        raise TypeError("domain must be MappedReferenceDomain.")
    elements, routes, _ = domain.source_geometry.resolve(domain.reference_mesh)
    coefficients = np.asarray(domain.source_geometry.source_coordinates())
    domains: dict[str, BernsteinDomain] = {
        "tetrahedron": "simplex",
        "prism": "prism",
        "hexahedron": "box",
        "pyramid": "box",
    }
    upper_bound = 0.0
    for block, element, route in zip(
        domain.reference_mesh.blocks, elements, routes, strict=True
    ):
        domain_name = domains.get(reference_cell_topology(block.cell_kind).family)
        if not isinstance(element, cell_geometry_owner.ScalarCoordinateElement):
            raise TypeError(
                "Mapped layer sizing requires a scalar source coordinate map."
            )
        if domain_name is None:
            raise ValueError(
                f"Mapped layer/core sizing does not support {block.cell_kind!r} reference roots."
            )
        for row, local_route in enumerate(np.asarray(route, dtype=np.int64).tolist()):
            local = tuple(coefficients[int(index)] for index in local_route)
            expressions = coordinate_map_expressions(element, local, cell_row=row)
            if len(expressions) != 3:
                raise ValueError(
                    "Mapped layer/core sizing requires three physical coordinates."
                )
            jacobian_bounds = np.empty((3, 3), dtype=np.float64)
            for physical_axis, expression in enumerate(expressions):
                for reference_axis in range(3):
                    lower, upper = polynomial_bounds(
                        derivative(expression, reference_axis),
                        domain_name,
                        3,
                    )
                    jacobian_bounds[physical_axis, reference_axis] = max(
                        abs(lower), abs(upper)
                    )
            one_norm = max(
                math.fsum(float(value) for value in jacobian_bounds[:, axis])
                for axis in range(3)
            )
            infinity_norm = max(
                math.fsum(float(value) for value in jacobian_bounds[axis, :])
                for axis in range(3)
            )
            product_bound = float(np.nextafter(one_norm * infinity_norm, np.inf))
            cell_bound = float(np.nextafter(math.sqrt(product_bound), np.inf))
            upper_bound = max(upper_bound, cell_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("Mapped source has no finite positive physical sizing bound.")
    return float(np.nextafter(upper_bound, np.inf))


def mapped_core_quality_findings(
    construction: layer_topology.LayerCoreConstruction,
    schedule: object,
    /,
) -> tuple[tuple[tuple[str, str, int, float], ...], str]:
    """Re-evaluate the native affine-tetrahedron schedule in physical coordinates.

    This concerns the physical corner tetrahedra, not the curved map between
    corners. Whole-cell high-order validity is independently certified during
    mesh acceptance.
    """
    from .._meshcore import TetMesh3D
    from ._volume_generation import (
        _schedule_quality_rows,
        _unmet_rows,
        NativeVolumeSchedule,
    )

    if not isinstance(schedule, NativeVolumeSchedule):
        raise TypeError("Mapped core quality requires a native volume schedule.")
    core = construction.core
    core_mesh = core.mesh
    original_ids = np.sort(np.asarray(core_mesh.entity_set(3).entity_ids, dtype=np.int64))
    target_ids = np.asarray(construction.core_cell_ids, dtype=np.int64)
    if (
        original_ids.shape != target_ids.shape
        or len(set(target_ids.tolist())) != target_ids.size
    ):
        raise ValueError(
            "Physical core quality requires a bijection of core cell identities."
        )
    mapping = np.asarray(construction.core_vertex_map, dtype=np.int64)
    if mapping.shape != (core_mesh.coordinates.shape[0],):
        raise ValueError("Physical core quality requires the complete core vertex map.")
    target_coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    if (
        np.any(mapping < 0)
        or np.any(mapping >= len(target_coordinates))
        or len(set(mapping.tolist())) != mapping.size
    ):
        raise ValueError("Physical core quality has an invalid vertex ancestry.")
    actual_cells = {
        int(identifier): tuple(int(value) for value in vertices)
        for block in construction.mesh.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    tetrahedra, cell_ids = [], []
    for block in core_mesh.blocks:
        if reference_cell_topology(block.cell_kind).family != "tetrahedron":
            raise ValueError("Physical native core quality requires tetrahedral cells.")
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True):
            row = int(np.searchsorted(original_ids, int(identifier)))
            target_id = int(target_ids[row])
            local = np.asarray(vertices, dtype=np.int64)
            if actual_cells.get(target_id) != tuple(
                int(value) for value in mapping[local]
            ):
                raise ValueError(
                    "Physical core cell connectivity changed its reference ancestry."
                )
            tetrahedra.append(local)
            cell_ids.append(target_id)
    boundary_faces = np.asarray(core.boundary_faces, dtype=np.int32)
    face_sources = np.arange(len(boundary_faces), dtype=np.int64)
    with TetMesh3D(
        np.asarray(core_mesh.coordinates, dtype=np.float64),
        np.asarray(tetrahedra, dtype=np.int32),
        None,
        boundary_faces,
        face_sources,
        np.empty((0, 2), dtype=np.int32),
        np.empty((0,), dtype=np.int64),
        vertex_ids=np.arange(mapping.size, dtype=np.int64),
        cell_ids=np.asarray(cell_ids, dtype=np.int64),
    ) as state:
        snapshot_ids = np.asarray(state.arrays().vertex_ids, dtype=np.int64)
        if not np.array_equal(snapshot_ids, np.arange(mapping.size, dtype=np.int64)):
            raise ValueError(
                "Physical quality candidate changed the native vertex order."
            )
        quality = state.evaluate_candidate_quality(
            target_coordinates[mapping],
            vertex_ids=snapshot_ids,
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        return _schedule_quality_rows(_unmet_rows(quality)), quality.candidate_id


def _layer_cells_by_id(mesh: CellMesh, /) -> dict[int, tuple[str, tuple[int, ...]]]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result: dict[int, tuple[str, tuple[int, ...]]] = {}
    for block in mesh.blocks:
        if block.cell_kind not in {"prism", "pyramid"}:
            raise ValueError(
                "Mapped layer/core composition requires prism or pyramid cells."
            )
        for identifier, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            cell_id = int(identifier)
            if cell_id in result:
                raise ValueError("Layer cell global IDs must be unique across blocks.")
            result[cell_id] = (
                block.cell_kind,
                tuple(int(value) for value in vertex_ids[vertices].tolist()),
            )
    return result


def validate_layer_reference_columns(
    source: sources_owner.NativeLayerCoreSource,
    reference: sources_owner.NativeLayerCoreSource,
    domain: MappedReferenceDomain,
    /,
) -> None:
    """Prove mapped source charts preserve physical layer cell intervals."""
    physical_mesh = source.layers.mesh
    reference_mesh = reference.layers.mesh
    physical_vertex_ids = set(
        int(value)
        for value in np.asarray(physical_mesh.vertex_global_ids, dtype=np.int64)
    )
    reference_vertex_id_set = set(
        int(value)
        for value in np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64)
    )
    if physical_vertex_ids != reference_vertex_id_set:
        raise ValueError(
            "Reference layer vertices must retain physical source global IDs."
        )
    physical_cells = _layer_cells_by_id(physical_mesh)
    reference_cells = _layer_cells_by_id(reference_mesh)
    if set(physical_cells) != set(reference_cells):
        raise ValueError("Mapped reference layers must retain every physical layer cell.")
    if any(
        physical_cells[cell_id] != reference_cells[cell_id] for cell_id in physical_cells
    ):
        raise ValueError(
            "Mapped reference layer connectivity must retain source vertex identities."
        )
    physical_cell_ids = np.sort(
        np.asarray(physical_mesh.entity_set(3).entity_ids, dtype=np.int64)
    )
    reference_cell_ids = np.sort(
        np.asarray(reference_mesh.entity_set(3).entity_ids, dtype=np.int64)
    )
    physical_layer_index = np.asarray(source.layers.layer_index, dtype=np.int64)
    reference_layer_index = np.asarray(reference.layers.layer_index, dtype=np.int64)
    physical_columns = np.asarray(source.layers.column_index, dtype=np.int64)
    reference_columns = np.asarray(reference.layers.column_index, dtype=np.int64)
    if (
        physical_layer_index.shape != physical_cell_ids.shape
        or reference_layer_index.shape != reference_cell_ids.shape
        or physical_columns.shape != physical_cell_ids.shape
        or reference_columns.shape != reference_cell_ids.shape
        or not np.array_equal(physical_cell_ids, reference_cell_ids)
        or not np.array_equal(physical_layer_index, reference_layer_index)
        or not np.array_equal(physical_columns, reference_columns)
    ):
        raise ValueError(
            "Mapped reference layers must retain physical layer and column identities."
        )
    regions = _block_order_regions(reference_mesh, reference.layer_regions)
    controls = nested_reference_owner.exact_reference_chart_controls(
        domain, reference_mesh, CellGeometrySpec.affine(reference_mesh), regions
    )
    control_by_id = nested_reference_owner.reference_chart_controls_by_target_id(controls)
    root_elements, root_routes, _ = domain.source_geometry.resolve(domain.reference_mesh)
    coefficients = np.asarray(domain.source_geometry.source_coordinates())
    physical_geometry = CellGeometrySpec.affine(physical_mesh)
    physical_elements, physical_routes, _ = physical_geometry.resolve(physical_mesh)
    physical_coefficients = np.asarray(physical_geometry.source_coordinates())
    physical_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(physical_mesh.vertex_global_ids, dtype=np.int64)
        )
    }
    physical_cell_rows: dict[int, tuple[int, int]] = {}
    for block_index, block in enumerate(physical_mesh.blocks):
        for row, identifier in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            cell_id = int(identifier)
            if cell_id in physical_cell_rows:
                raise ValueError("Physical layer cell IDs must be unique across blocks.")
            physical_cell_rows[cell_id] = (block_index, row)
    reference_vertex_ids = np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64)
    cell_regions_by_id = dict(
        zip(
            np.sort(
                np.asarray(reference_mesh.entity_set(3).entity_ids, dtype=np.int64)
            ).tolist(),
            np.asarray(reference.layer_regions, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    for block in reference_mesh.blocks:
        for row, raw_identifier in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            identifier = int(raw_identifier)
            control = control_by_id[identifier]
            physical_block_index, _ = physical_cell_rows[identifier]
            if (
                control.target_cell_kind not in {"prism", "pyramid"}
                or control.target_cell_kind != block.cell_kind
                or physical_mesh.blocks[physical_block_index].cell_kind
                != control.target_cell_kind
            ):
                raise ValueError(
                    "Every mapped layer cell must retain its exact prism or pyramid reference chart."
                )
            if control.source_region != int(cell_regions_by_id[identifier]):
                raise ValueError("Mapped layer source root changed its material region.")
            root_element = root_elements[control.source_block_index]
            if not isinstance(root_element, cell_geometry_owner.ScalarCoordinateElement):
                raise TypeError("Mapped layer roots require scalar coordinate elements.")
            root_route = np.asarray(
                root_routes[control.source_block_index], dtype=np.int64
            )[control.source_cell_row]
            root_local = tuple(coefficients[int(index)] for index in root_route)
            root_expressions = coordinate_map_expressions(
                root_element, root_local, cell_row=control.source_cell_row
            )
            control_arguments = _canonical_expressions(control.arguments)
            mapped = tuple(
                compose_expression(expression, control_arguments)
                for expression in root_expressions
            )
            block_index, physical_row = physical_cell_rows[identifier]
            physical_element = physical_elements[block_index]
            if not isinstance(
                physical_element, cell_geometry_owner.ScalarCoordinateElement
            ):
                raise TypeError(
                    "Physical layer cells require scalar coordinate elements."
                )
            physical_route = np.asarray(physical_routes[block_index], dtype=np.int64)[
                physical_row
            ]
            physical_local = tuple(
                physical_coefficients[int(index)] for index in physical_route
            )
            physical_map = coordinate_map_expressions(
                physical_element, physical_local, cell_row=physical_row
            )
            target_vertices = np.asarray(block.vertices[row], dtype=np.int64)
            source_vertex_ids = reference_vertex_ids[target_vertices]
            physical_rows = np.asarray(
                [physical_vertex_rows[int(value)] for value in source_vertex_ids],
                dtype=np.int64,
            )
            for point, vertex_row in zip(
                control.source_reference_vertices, physical_rows.tolist(), strict=True
            ):
                image = tuple(
                    evaluate_expression(expression, point)
                    for expression in root_expressions
                )
                if not np.array_equal(
                    rounded_point(image),
                    np.asarray(physical_mesh.coordinates, dtype=np.float64)[vertex_row],
                ):
                    raise ValueError(
                        "Mapped root images do not retain the physical layer vertices."
                    )
            if any(
                not expression_equal(derivative(add(current, scale(initial, -1)), 2), {})
                for current, initial in zip(mapped, physical_map, strict=True)
            ):
                raise ValueError(
                    "Mapped source geometry changes a realized physical layer interval."
                )


def _mapped_wall_graph(
    source: sources_owner.NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> tuple[Polynomial, tuple[Fraction, Fraction, Fraction, Fraction]]:
    """Prove that the complete selected physical wall is one extruded quadratic graph."""
    from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from ._layer_core_controls import realized_layer_control
    from .providers._native_sources import NativeMappedHexSource, NativePlcSource

    if len(specification.layer_controls) != 1 or source.layer_control is None:
        raise ValueError("Mapped refined columns require their original wall control.")
    requested = specification.layer_controls[0]
    if realized_layer_control(source, requested).control_id != source.layers.control_id:
        raise ValueError(
            "Mapped refined columns changed the original realized wall control."
        )
    original = source.fidelity_source
    if not isinstance(original, MappedReferenceBoundarySource):
        raise ValueError("Mapped refined columns require the original mapped wall query.")
    if not isinstance(original.authority, (NativeMappedHexSource, NativePlcSource)):
        raise TypeError("Mapped wall requires its original full PLC source authority.")
    keys = tuple(
        sorted(
            original.authority.source_entity_key_for_scope(2, int(identifier))
            for identifier in np.asarray(requested.wall_scope.entity_ids, dtype=np.int64)
        )
    )
    query = original.authority.boundary_query(face_scope_keys=keys)
    if not isinstance(query, MappedReferenceBoundarySource):
        raise TypeError("Mapped wall scope did not reconstruct its exact source query.")
    query.require_current()
    triangles = []
    graph: Polynomial | None = None
    corners = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    variable = axes(1)[0]
    for row in query._selected_rows(2):
        expressions = _canonical_expressions(
            tuple(query.chart_coordinate_expressions(2, query.face_chart_ids[row]))
        )
        if len(expressions) != 3:
            raise ValueError(
                "Mapped wall graph requires three physical chart coordinates."
            )
        x, y, z = expressions
        if not isinstance(x, dict) or not isinstance(y, dict) or not isinstance(z, dict):
            raise ValueError("Mapped wall graph requires exact polynomial source charts.")
        if any(sum(exponent) > 1 for expression in (x, y) for exponent in expression):
            raise ValueError("Mapped wall graph requires affine projected source charts.")
        points_list: list[tuple[Fraction, ...]] = []
        for corner in corners:
            point_values: list[Fraction] = []
            for expression in expressions:
                value = evaluate_expression(expression, corner)
                if not isinstance(value, Fraction):
                    raise ValueError(
                        "Mapped wall graph requires rational source-chart corners."
                    )
                point_values.append(value)
            points_list.append(tuple(point_values))
        points = tuple(points_list)
        varying = next(
            (
                (i, j)
                for i in range(3)
                for j in range(i + 1, 3)
                if points[i][1] != points[j][1]
            ),
            None,
        )
        if varying is None:
            raise ValueError("Mapped wall graph has a degenerate projected source chart.")
        first, last = varying
        start, end = points[first][1], points[last][1]
        parameter = scale(add(variable, constant(-start, 1)), 1 / (end - start))
        arguments = tuple(
            add(
                constant(corners[first][axis], 1),
                scale(parameter, corners[last][axis] - corners[first][axis]),
            )
            for axis in range(2)
        )
        candidate = compose_expression(z, arguments)
        if not isinstance(candidate, dict):
            raise ValueError("Mapped wall graph is not an exact rational quadratic.")
        rational_candidate: Polynomial = {}
        for exponent, coefficient in candidate.items():
            if not isinstance(coefficient, Fraction) or exponent[0] > 2:
                raise ValueError("Mapped wall graph is not an exact rational quadratic.")
            rational_candidate[exponent] = coefficient
        if not expression_equal(z, compose_expression(rational_candidate, (y,))):
            raise ValueError("Mapped wall source charts do not share one extruded graph.")
        if graph is None:
            graph = rational_candidate
        elif not expression_equal(graph, rational_candidate):
            raise ValueError("Mapped wall source charts have different physical graphs.")
        triangles.append(tuple((point[0], point[1], Fraction(0)) for point in points))
    if graph is None or not triangles:
        raise ValueError("Mapped wall source scope has no complete graph charts.")
    xmin = min(point[0] for triangle in triangles for point in triangle)
    xmax = max(point[0] for triangle in triangles for point in triangle)
    ymin = min(point[1] for triangle in triangles for point in triangle)
    ymax = max(point[1] for triangle in triangles for point in triangle)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Mapped wall graph has a degenerate physical domain.")
    rectangle = (
        ((xmin, ymin, Fraction(0)), (xmax, ymin, Fraction(0)), (xmax, ymax, Fraction(0))),
        ((xmin, ymin, Fraction(0)), (xmax, ymax, Fraction(0)), (xmin, ymax, Fraction(0))),
    )
    exact_coplanar_triangle_partitions(tuple(triangles), rectangle)
    return graph, (xmin, xmax, ymin, ymax)


def _mapped_wall_distance_bounds(
    graph: Polynomial,
    wall_y: tuple[Fraction, Fraction],
    point_y: Fraction,
    point_z: Fraction,
    tolerance: Fraction,
    /,
) -> tuple[Fraction, Fraction, bool, Fraction | None]:
    """Enclose the true nearest distance using canonical real-root isolation."""
    from ..algebraic._exact_univariate import isolate_real_roots
    from ..linalg._hermitian_spectral import fraction_sqrt_interval

    variable = axes(1)[0]
    delta_y = add(variable, constant(-point_y, 1))
    delta_z = add(graph, constant(-point_z, 1))
    squared = add(multiply(delta_y, delta_y), multiply(delta_z, delta_z))
    stationary = derivative(squared, 0)
    curvature = derivative(stationary, 0)
    if not isinstance(squared, dict) or not isinstance(stationary, dict):
        raise ValueError("Mapped wall distance is not an exact polynomial.")
    left, right = wall_y
    coordinate = add(constant(left, 1), scale(variable, right - left))
    convexity = polynomial_bounds(compose_expression(curvature, (coordinate,)), "box", 1)
    if not math.isfinite(convexity[0]) or convexity[0] <= 0:
        raise ValueError("Mapped wall nearest-point strict convexity is unresolved.")
    first = evaluate_expression(stationary, (left,))
    last = evaluate_expression(stationary, (right,))
    if not isinstance(first, Fraction) or not isinstance(last, Fraction):
        raise ValueError("Mapped wall stationary signs are not rational.")
    if first >= 0:
        a = b = left
    elif last <= 0:
        a = b = right
    else:
        degree = max(exponent[0] for exponent in stationary)
        coefficients: list[Fraction] = []
        for index in range(degree + 1):
            coefficient = stationary.get((index,), Fraction(0))
            if not isinstance(coefficient, Fraction):
                raise ValueError("Mapped wall stationary polynomial is not rational.")
            coefficients.append(coefficient)
        while True:
            roots = isolate_real_roots(
                tuple(coefficients), tolerance=tolerance, recognize_rational_roots=False
            )
            selected = tuple(
                (max(a, left), min(b, right))
                for a, b, _ in roots
                if b >= left and a <= right
            )
            if len(selected) == 1:
                a, b = selected[0]
                break
            if not selected:
                raise ValueError("Mapped wall stationary root isolation is inconsistent.")
            tolerance /= 2
    if a == b:
        exact = evaluate_expression(squared, (a,))
        if not isinstance(exact, Fraction) or exact < 0:
            raise ValueError(
                "Mapped wall minimum is not an exact rational squared distance."
            )
        low, high = fraction_sqrt_interval(exact)
        return low, high, True, exact
    restriction = add(constant(a, 1), scale(variable, b - a))
    bounds = polynomial_bounds(compose_expression(squared, (restriction,)), "box", 1)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Mapped wall distance enclosure is not finite.")
    lower = max(Fraction(0), Fraction.from_float(bounds[0]))
    sample = evaluate_expression(squared, ((a + b) / 2,))
    if not isinstance(sample, Fraction) or sample < lower:
        raise ValueError("Mapped wall distance enclosure is inconsistent.")
    upper = min(Fraction.from_float(bounds[1]), sample)
    if upper < lower:
        raise ValueError("Mapped wall distance enclosure is inconsistent.")
    low = fraction_sqrt_interval(lower)[0]
    high = fraction_sqrt_interval(upper)[1]
    exhausted = math.nextafter(float(a), math.inf) >= float(b)
    return low, high, exhausted, None


def _certify_mapped_column_controls(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    columns: dict[tuple[tuple[int, Fraction], ...], dict[int, int]],
    coordinates: tuple[tuple[Fraction, ...], ...],
    /,
) -> None:
    """Check each actual mapped child column against its true source wall."""
    from ._layer_core_subdivision import (
        _distance_increment_at_least,
        _growth_difference_sign,
    )

    graph, (xmin, xmax, ymin, ymax) = _mapped_wall_graph(source, specification)
    control = specification.layer_controls[0]
    thicknesses = tuple(
        Fraction.from_float(float(value)) for value in control.schedule.thicknesses
    )
    absolute = Fraction.from_float(
        float(specification.size_compliance.absolute_tolerance)
    )
    relative = Fraction.from_float(
        float(specification.size_compliance.relative_tolerance)
    )
    minimum = Fraction.from_float(float(control.minimum_thickness_fraction))
    minimum = max(Fraction(0), minimum - absolute - relative * minimum)
    maximum_stretch = Fraction.from_float(float(control.maximum_corner_stretch))
    maximum_stretch += absolute + relative * maximum_stretch
    growth = control.growth_rate_bounds
    if growth is not None:
        lower, upper = map(Fraction.from_float, growth)
        slack = absolute + relative * max(abs(lower), abs(upper))
        lower, upper = max(Fraction(0), lower - slack), upper + slack
    slope = derivative(graph, 0)
    for levels in columns.values():
        if set(levels) != set(range(len(thicknesses) + 1)):
            raise ValueError(
                "A mapped child column does not cover the original layer schedule."
            )
        points = tuple(
            coordinates[levels[level]] for level in range(len(thicknesses) + 1)
        )
        x, y, wall_z = points[0]
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            raise ValueError(
                "A mapped child column starts outside the original wall scope."
            )
        if evaluate_expression(graph, (y,)) != wall_z:
            raise ValueError(
                "A mapped child column does not start on the actual source wall."
            )
        normal_slope = evaluate_expression(slope, (y,))
        if not isinstance(normal_slope, Fraction):
            raise ValueError("Mapped wall normal is not an exact rational vector.")
        if 1 + normal_slope * normal_slope > maximum_stretch * maximum_stretch:
            raise ValueError(
                "A mapped child column exceeds the original physical stretch bound."
            )
        if any(
            point[0] != x or point[1] != y or point[2] <= wall_z for point in points[1:]
        ):
            raise ValueError(
                "A mapped child column leaves the proved inward vertical source side."
            )
        tolerance = (ymax - ymin) / (2**12)
        while True:
            distances = [(Fraction(0), Fraction(0), True, Fraction(0))] + [
                _mapped_wall_distance_bounds(graph, (ymin, ymax), y, point[2], tolerance)
                for point in points[1:]
            ]
            increments = tuple(
                (current[0] - previous[1], current[1] - previous[0])
                for previous, current in zip(distances[:-1], distances[1:], strict=True)
            )
            unresolved = False
            for index, ((low, high), requested) in enumerate(
                zip(increments, thicknesses, strict=True)
            ):
                floor = requested * minimum
                before, after = distances[index][3], distances[index + 1][3]
                if before is not None and after is not None:
                    if not _distance_increment_at_least(after, before, floor):
                        raise ValueError(
                            "A mapped child column violates the original physical thickness floor."
                        )
                else:
                    if high < floor:
                        raise ValueError(
                            "A mapped child column violates the original physical thickness floor."
                        )
                    unresolved |= low < floor
            if growth is not None:
                for index, (previous, current) in enumerate(
                    zip(increments[:-1], increments[1:], strict=True)
                ):
                    a, b, c = (
                        distances[index + 2][3],
                        distances[index + 1][3],
                        distances[index][3],
                    )
                    if a is not None and b is not None and c is not None:
                        if (
                            _growth_difference_sign(a, b, c, lower) < 0
                            or _growth_difference_sign(a, b, c, upper) > 0
                        ):
                            raise ValueError(
                                "A mapped child column violates original physical growth bounds."
                            )
                    else:
                        if (
                            previous[1] <= 0
                            or current[1] < lower * previous[0]
                            or current[0] > upper * previous[1]
                        ):
                            raise ValueError(
                                "A mapped child column violates original physical growth bounds."
                            )
                        unresolved |= (
                            previous[0] <= 0
                            or current[0] < lower * previous[1]
                            or current[1] > upper * previous[0]
                        )
            if not unresolved:
                break
            if all(exhausted for _, _, exhausted, _ in distances):
                raise ValueError(
                    "Mapped child column controls are unresolved at binary64 root resolution."
                )
            tolerance /= 2


def _validate_refined_mapped_layers(
    source: sources_owner.NativeLayerCoreSource,
    reference_mesh: CellMesh,
    reference_geometry: CellGeometrySpec,
    reference_cell_regions: tuple[int, ...],
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    specification: VolumeMeshingSpec,
    partition: ReferenceCellPartition,
    /,
) -> bool:
    """Prove the actual child-prism maps, original vertices and physical controls."""

    reference = source.reference_source
    domain = source.mapped_domain
    if reference is None or not isinstance(domain, MappedReferenceDomain):
        raise ValueError(
            "Mapped layer replay requires its original reference source and domain."
        )
    validate_layer_reference_columns(source, reference, domain)
    validate_reference_cell_partition(partition)

    def maps(mesh: CellMesh, geometry: CellGeometrySpec) -> _ExactCellMaps:
        elements, routes, _ = geometry.resolve(mesh)
        return _geometry_map_expressions(mesh, geometry, (elements, routes))

    def cell_incidence(mesh: CellMesh) -> dict[int, tuple[str, tuple[int, ...]]]:
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        return {
            int(identifier): (
                block.cell_kind,
                tuple(int(vertex_ids[int(row)]) for row in vertices),
            )
            for block in mesh.blocks
            for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
        }

    if cell_incidence(reference_mesh) != cell_incidence(target_mesh):
        raise ValueError("Mapped reference and physical target cell incidence differs.")
    if not np.array_equal(
        np.asarray(reference_mesh.vertex_global_ids),
        np.asarray(target_mesh.vertex_global_ids),
    ):
        raise ValueError("Mapped reference and physical target vertex identities differ.")
    from ..geometry._mesh_certificates import _same_tree_content
    from ._topology_edit import entity_keys

    for dimension in (1, 2):

        def entity_bank(mesh: CellMesh) -> dict[tuple[int, ...], int]:
            return {
                tuple(int(value) for value in key if value >= 0): int(identifier)
                for key, identifier in zip(
                    entity_keys(mesh, dimension),
                    mesh.entity_set(dimension).entity_ids,
                    strict=True,
                )
            }

        if entity_bank(reference_mesh) != entity_bank(target_mesh):
            raise ValueError(
                "Mapped reference and physical target entity identities differ."
            )
    reference_periodic = reference_mesh.periodic_topology
    target_periodic = target_mesh.periodic_topology
    if (reference_periodic is None) != (target_periodic is None):
        raise ValueError("Mapped reference and physical target periodicity differs.")
    if reference_periodic is not None and target_periodic is not None:
        if (
            not _same_tree_content(reference_periodic.cell, target_periodic.cell)
            or not np.array_equal(
                np.asarray(reference_periodic.vertex_representatives),
                np.asarray(target_periodic.vertex_representatives),
            )
            or not np.array_equal(
                np.asarray(reference_periodic.vertex_shifts),
                np.asarray(target_periodic.vertex_shifts),
            )
        ):
            raise ValueError(
                "Mapped reference and physical target periodic actions differ."
            )
    target_regions = _block_order_regions(
        reference_mesh, np.asarray(reference_cell_regions, dtype=np.int64)
    )
    expected_blocks, expected_geometry, _, _, expected_points = _geometry_groups(
        domain, reference_mesh, reference_geometry, target_regions
    )
    if not np.array_equal(
        np.asarray(expected_points).view(np.uint64),
        np.asarray(target_mesh.coordinates).view(np.uint64),
    ):
        raise ValueError(
            "Mapped physical vertices do not reproduce their source-root maps."
        )
    expected_mesh = CellMesh(
        expected_points,
        expected_blocks,
        vertex_global_ids=reference_mesh.vertex_global_ids,
    )
    expected_maps = maps(expected_mesh, expected_geometry)
    physical_target_maps = maps(target_mesh, target_geometry)
    for block in reference_mesh.blocks:
        for identifier in np.asarray(block.global_ids, dtype=np.int64):
            expected = expected_maps.expressions(int(identifier))
            actual = physical_target_maps.expressions(int(identifier))
            if len(expected) != len(actual) or any(
                not expression_equal(first, second)
                for first, second in zip(expected, actual, strict=True)
            ):
                raise ValueError(
                    "Mapped physical cell changed its complete source-root coordinate map."
                )
    parent_geometry = CellGeometrySpec.affine(reference.layers.mesh)
    reference_parent_maps = maps(reference.layers.mesh, parent_geometry)
    reference_target_maps = maps(reference_mesh, reference_geometry)
    for group in partition.maps:
        _require_exact_reference_group(
            reference_parent_maps,
            reference_target_maps,
            group,
            exact_reference_cell_vertices(group),
        )
    _validate_mapped_layer_partition(
        source, target_mesh, target_geometry, specification, partition
    )
    return _same_tree_content(
        expected_geometry.restriction_source, target_geometry.restriction_source
    )


def _validate_mapped_layer_partition(
    source: sources_owner.NativeLayerCoreSource,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    specification: VolumeMeshingSpec,
    partition: ReferenceCellPartition,
    /,
) -> None:
    """Prove current physical prism restrictions and all physical layer controls."""
    from ._layer_core_subdivision import _layer_exact_by_id

    reference, domain = source.reference_source, source.mapped_domain
    if reference is None or not isinstance(domain, MappedReferenceDomain):
        raise ValueError(
            "Mapped layer replay requires its original reference source and domain."
        )
    validate_layer_reference_columns(source, reference, domain)
    validate_reference_cell_partition(partition)
    parent_ids = set(_layer_cells_by_id(reference.layers.mesh))
    target_ids = {
        int(value)
        for block in target_mesh.blocks
        if block.cell_kind == "prism"
        for value in np.asarray(block.global_ids, dtype=np.int64)
    }
    mapped_parents = {
        int(value)
        for group in partition.maps
        for value in np.asarray(group.source_cell_ids)
    }
    mapped_targets = {
        int(value)
        for group in partition.maps
        for value in np.asarray(group.target_cell_ids)
    }
    if mapped_parents != parent_ids or mapped_targets != target_ids:
        raise ValueError(
            "Mapped layer partition does not cover its actual original and child cells."
        )
    parent_geometry = CellGeometrySpec.affine(reference.layers.mesh)
    parent_regions = _block_order_regions(reference.layers.mesh, reference.layer_regions)
    parent_blocks, mapped_parent_geometry, _, _, parent_points = _geometry_groups(
        domain, reference.layers.mesh, parent_geometry, parent_regions
    )
    mapped_parent_mesh = CellMesh(
        parent_points,
        parent_blocks,
        vertex_global_ids=reference.layers.mesh.vertex_global_ids,
    )
    parent_resolved = mapped_parent_geometry.resolve(mapped_parent_mesh)
    physical_parent_maps = _geometry_map_expressions(
        mapped_parent_mesh,
        mapped_parent_geometry,
        (parent_resolved[0], parent_resolved[1]),
    )
    target_resolved = target_geometry.resolve(target_mesh)
    physical_target_maps = _geometry_map_expressions(
        target_mesh, target_geometry, (target_resolved[0], target_resolved[1])
    )
    target_rows = {
        int(identifier): tuple(
            int(value) for value in np.asarray(vertices, dtype=np.int64)
        )
        for block in target_mesh.blocks
        if block.cell_kind == "prism"
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    source_cells = {
        int(identifier): tuple(
            int(value) for value in np.asarray(vertices, dtype=np.int64)
        )
        for block in source.layers.mesh.blocks
        if block.cell_kind == "prism"
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    intervals = dict(
        zip(
            sorted(source_cells),
            np.asarray(source.layers.layer_index, dtype=np.int64),
            strict=True,
        )
    )
    original_columns = np.asarray(source.layers.vertex_column_ids, dtype=np.int64)
    columns: dict[tuple[tuple[int, Fraction], ...], dict[int, int]] = {}
    for group in partition.maps:
        if group.target_cell_kind != "prism" or group.source_cell_kind != "prism":
            raise ValueError(
                "Conforming mapped layer partitions require prism children and parents."
            )
        exact = exact_reference_cell_vertices(group)
        _require_exact_reference_group(
            physical_parent_maps, physical_target_maps, group, exact
        )
        for item, corners in zip(
            nested_reference_owner.nested_cell_reference_maps(group), exact, strict=True
        ):
            if item.arguments is None:
                raise ValueError(
                    "A mapped layer child lacks its exact parent reference map."
                )
            args = _canonical_expressions(item.arguments)
            if (
                len(args) != 3
                or not expression_equal(args[2], {(0, 0, 1): Fraction(1)})
                or not expression_equal(derivative(args[0], 2), {})
                or not expression_equal(derivative(args[1], 2), {})
            ):
                raise ValueError(
                    "A refined mapped layer chart changes its original column interval."
                )
            parent = source_cells[item.source_cell_id]
            child = target_rows[item.target_cell_id]
            level = int(intervals[item.source_cell_id])
            for corner in range(3):
                u, v, z = corners[corner]
                if z != 0 or corners[corner + 3] != (u, v, Fraction(1)):
                    raise ValueError(
                        "A refined prism does not retain its original transverse column."
                    )
                weights = (1 - u - v, u, v)
                key = tuple(
                    sorted(
                        (int(original_columns[parent[3 + index]]), weight)
                        for index, weight in enumerate(weights)
                        if weight
                    )
                )
                if any(identifier < 0 for identifier, _ in key):
                    raise ValueError(
                        "A refined column lacks original wall-column ancestry."
                    )
                bank = columns.setdefault(key, {})
                for child_level, vertex_row in (
                    (level, child[corner]),
                    (level + 1, child[corner + 3]),
                ):
                    previous = bank.setdefault(child_level, vertex_row)
                    if previous != vertex_row:
                        raise ValueError(
                            "Incident refined prisms disagree on their column vertices."
                        )
    source_mesh = source.layers.mesh
    original_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    target_by_id = {int(value): row for row, value in enumerate(target_vertex_ids)}
    if any(int(value) not in target_by_id for value in original_ids):
        raise ValueError("Conforming mapped layers lost an original physical vertex.")
    for row, identifier in enumerate(original_ids):
        actual = np.asarray(target_mesh.coordinates, dtype=np.float64)[
            target_by_id[int(identifier)]
        ]
        expected = np.asarray(source_mesh.coordinates, dtype=np.float64)[row]
        if not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
            raise ValueError(
                "Conforming mapped layers moved an original physical vertex."
            )
    old_periodic, new_periodic = (
        source_mesh.periodic_topology,
        target_mesh.periodic_topology,
    )
    if (old_periodic is None) != (new_periodic is None):
        raise ValueError("Conforming mapped layers changed original periodicity.")
    if old_periodic is not None and new_periodic is not None:
        old_reps = np.asarray(old_periodic.vertex_representatives, dtype=np.int64)
        new_reps = np.asarray(new_periodic.vertex_representatives, dtype=np.int64)
        old_shifts = np.asarray(old_periodic.vertex_shifts, dtype=np.int64)
        new_shifts = np.asarray(new_periodic.vertex_shifts, dtype=np.int64)
        for row, identifier in enumerate(original_ids):
            target_row = target_by_id[int(identifier)]
            if int(original_ids[old_reps[row]]) != int(
                target_vertex_ids[new_reps[target_row]]
            ) or not np.array_equal(old_shifts[row], new_shifts[target_row]):
                raise ValueError(
                    "Conforming mapped layers changed an original periodic orbit."
                )
    exact_by_id = _layer_exact_by_id(target_mesh, target_geometry)
    exact_coordinates = tuple(
        exact_by_id.get(int(identifier), (Fraction(0),) * 3)
        for identifier in target_vertex_ids
    )
    _certify_mapped_column_controls(source, specification, columns, exact_coordinates)


def _reprove_mapped_layer_endpoint(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    root: object,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    reference_cell_regions: tuple[int, ...],
    replay: object,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> None:
    """Reprove a mapped descendant through its actual root, never a rounded proxy."""
    from ..geometry._mesh_certificates import _same_tree_content
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
    from ._layer_source_association_replay import (
        _conforming_current_rows,
        LayerAssociationReplay,
    )

    if not isinstance(root, LayerCoreRootGeometry) or not isinstance(
        replay, LayerAssociationReplay
    ):
        raise TypeError(
            "Mapped descendant reproof requires actual root and layer replay owners."
        )
    if (
        root.reference_mesh is None
        or root.reference_geometry is None
        or root.reference_cell_regions is None
    ):
        raise ValueError("Mapped descendant root lacks its original reference operands.")
    if computed_root is None:
        root.require_current(source, specification)
    else:
        if type(computed_root) is not ComputedLayerCoreRoot:
            raise TypeError("Mapped descendant reproof requires an actual computed root.")
        computed_root.require(source, specification, root.mesh, root.geometry, root)
    if mesh.mesh_id == root.mesh.mesh_id:
        if tuple(reference_cell_regions) != root.reference_cell_regions:
            raise ValueError("Mapped root endpoint changed its actual material bank.")
        _validate_refined_mapped_layers(
            source,
            root.reference_mesh,
            root.reference_geometry,
            root.reference_cell_regions,
            mesh,
            geometry,
            specification,
            root.layer_root_partition,
        )
        return
    current_rows = tuple(_conforming_current_rows(root, mesh, geometry))
    actual_ids = {
        int(identifier)
        for block in mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    if (
        len(current_rows) != len(actual_ids)
        or {row[0] for row in current_rows} != actual_ids
    ):
        raise ValueError(
            "Mapped descendant root maps do not cover every actual cell exactly once."
        )
    root_ids = sorted(
        int(identifier)
        for block in root.reference_mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )
    if len(root_ids) != len(root.reference_cell_regions) or len(
        reference_cell_regions
    ) != len(actual_ids):
        raise ValueError(
            "Mapped descendant material banks do not cover their actual cells."
        )
    root_regions = dict(zip(root_ids, root.reference_cell_regions, strict=True))
    target_regions = dict(zip(sorted(actual_ids), reference_cell_regions, strict=True))
    for identifier, _, current in current_rows:
        if (
            current.source_cell_id not in root_regions
            or target_regions[identifier] != root_regions[current.source_cell_id]
        ):
            raise ValueError("Mapped descendant changed its original root material.")
    retained = {int(identifier) for identifier, _ in root.overlap_stencils.rows}
    old_ids = np.asarray(root.mesh.vertex_global_ids, dtype=np.int64)
    current_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    old_rows = {int(identifier): row for row, identifier in enumerate(old_ids)}
    current_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(current_ids)
    }
    if not retained <= set(old_rows) or not retained <= set(current_vertex_rows):
        raise ValueError("Mapped descendant lost an original source-stencil vertex.")

    def exact_vertices(
        owner_mesh: CellMesh, owner_geometry: CellGeometrySpec
    ) -> dict[int, tuple[ExactScalar, ...]]:
        found: dict[int, tuple[ExactScalar, ...]] = {}
        for block in owner_mesh.blocks:
            ids = np.asarray(block.global_ids, dtype=np.int64)
            vertices = np.asarray(block.vertices, dtype=np.int64)
            if vertices.ndim != 2:
                raise ValueError(
                    "Mapped retained-vertex replay requires standard cell connectivity."
                )
            owner_ids = np.asarray(owner_mesh.vertex_global_ids, dtype=np.int64)
            selected = [
                row
                for row in range(vertices.shape[0])
                if any(
                    int(owner_ids[int(value)]) in retained
                    for value in np.asarray(vertices[row], dtype=np.int64).reshape(-1)
                )
            ]
            if not selected:
                continue
            corners = tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in reference_cell_topology(block.cell_kind).vertices
            )
            evaluations = owner_geometry.evaluate_exact(
                owner_mesh, ids[selected], tuple(corners for _ in selected)
            )
            for cell_row, images in zip(selected, evaluations, strict=True):
                for vertex_row, point in zip(vertices[cell_row], images, strict=True):
                    identifier = int(owner_ids[int(vertex_row)])
                    if identifier not in retained:
                        continue
                    exact = tuple(point)
                    if identifier in found and found[identifier] != exact:
                        raise ValueError(
                            "Mapped incident cells disagree at a retained source vertex."
                        )
                    if not np.array_equal(
                        rounded_point(exact).view(np.uint64),
                        np.asarray(owner_mesh.coordinates, dtype=np.float64)[
                            int(vertex_row)
                        ].view(np.uint64),
                    ):
                        raise ValueError(
                            "Mapped retained source vertex changed its exact rounded image."
                        )
                    found[identifier] = exact
        if set(found) != retained:
            raise ValueError(
                "Mapped source-stencil vertices lack complete actual cell incidence."
            )
        return found

    if exact_vertices(root.mesh, root.geometry) != exact_vertices(mesh, geometry):
        raise ValueError(
            "Mapped descendant changed an original source-stencil physical image."
        )
    old_periodic, new_periodic = root.mesh.periodic_topology, mesh.periodic_topology
    if (old_periodic is None) != (new_periodic is None):
        raise ValueError("Mapped descendant changed its original periodic action.")
    if old_periodic is not None and new_periodic is not None:
        if not _same_tree_content(old_periodic.cell, new_periodic.cell):
            raise ValueError("Mapped descendant changed the original periodic group.")
        old_reps = np.asarray(old_periodic.vertex_representatives, dtype=np.int64)
        new_reps = np.asarray(new_periodic.vertex_representatives, dtype=np.int64)
        old_shifts = np.asarray(old_periodic.vertex_shifts, dtype=np.int64)
        new_shifts = np.asarray(new_periodic.vertex_shifts, dtype=np.int64)
        for identifier in retained:
            old_row, new_row = old_rows[identifier], current_vertex_rows[identifier]
            if int(old_ids[old_reps[old_row]]) != int(
                current_ids[new_reps[new_row]]
            ) or not np.array_equal(old_shifts[old_row], new_shifts[new_row]):
                raise ValueError(
                    "Mapped descendant changed a retained source periodic orbit."
                )
    partition = ReferenceCellPartition(replay.reference_maps)
    _validate_mapped_layer_partition(source, mesh, geometry, specification, partition)


def _geometry_groups(
    domain: MappedReferenceDomain,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    target_regions: np.ndarray,
    /,
) -> tuple[
    tuple[CellBlock, ...],
    CellGeometrySpec,
    tuple[ReferenceCellMapWitnesses, ...],
    tuple[_CellChart, ...],
    np.ndarray,
]:
    controls = nested_reference_owner.exact_reference_chart_controls(
        domain, target_mesh, target_geometry, target_regions
    )
    by_target_id = nested_reference_owner.reference_chart_controls_by_target_id(controls)
    source_elements, source_routes, _ = domain.source_geometry.resolve(
        domain.reference_mesh
    )
    coordinates = np.asarray(domain.source_geometry.source_coordinates())
    charts: list[_CellChart] = []
    for target_block_index, target_block in enumerate(target_mesh.blocks):
        for target_row, raw_target_id in enumerate(
            np.asarray(target_block.global_ids, dtype=np.int64).tolist()
        ):
            target_id = int(raw_target_id)
            control = by_target_id[target_id]
            root_element = source_elements[control.source_block_index]
            if not isinstance(root_element, cell_geometry_owner.ScalarCoordinateElement):
                raise TypeError("Mapped layer roots require scalar coordinate elements.")
            base_element = cell_geometry_owner.coordinate_reference_chain(root_element)[0]
            root_chart = cell_geometry_owner.restriction_chart_arguments(
                root_element,
                ancestor=base_element,
                cell_row=control.source_cell_row,
            )
            control_arguments = _canonical_expressions(control.arguments)
            chart_arguments = (
                control_arguments
                if root_chart is None
                else tuple(
                    compose_expression(argument, control_arguments)
                    for argument in _canonical_expressions(root_chart)
                )
            )
            route = np.asarray(source_routes[control.source_block_index], dtype=np.int64)[
                control.source_cell_row
            ]
            local = tuple(coordinates[int(index)] for index in route)
            physical = coordinate_map_expressions(
                root_element, local, cell_row=control.source_cell_row
            )
            charts.append(
                _CellChart(
                    target_block_index,
                    target_row,
                    target_id,
                    target_block.cell_kind,
                    tuple(
                        int(value)
                        for value in np.asarray(
                            target_block.vertices[target_row], dtype=np.int64
                        )
                    ),
                    control.source_block_index,
                    control.source_block_name,
                    control.source_cell_row,
                    control.source_cell_id,
                    control.source_cell_kind,
                    control.source_region,
                    control.source_region_id,
                    control.source_occurrence_path,
                    control.source_reference_vertices,
                    root_element,
                    base_element,
                    chart_arguments,
                    control_arguments,
                    route,
                    physical,
                )
            )
    if len(charts) != sum(block.cell_count for block in target_mesh.blocks):
        raise ValueError("Exact mapped-root controls must cover every output cell.")

    physical_points: list[np.ndarray | None] = [None] * target_mesh.coordinates.shape[0]
    exact_points: dict[int, tuple[ExactScalar, ...]] = {}
    groups: dict[_GeometryGroupKey, list[_CellChart]] = {}
    rational_elements: dict[
        _GeometryGroupKey, cell_geometry_owner.RationalComposedCellGeometryElement
    ] = {}
    for chart in charts:
        for vertex_row, reference_point in zip(
            chart.target_vertices, chart.source_reference_vertices, strict=True
        ):
            image = tuple(
                evaluate_expression(expression, reference_point)
                for expression in chart.physical_expressions
            )
            previous = exact_points.get(vertex_row)
            if previous is not None and previous != image:
                raise ValueError(
                    "Adjacent mapped source charts disagree exactly at a shared target vertex."
                )
            exact_points[vertex_row] = image
            physical_points[vertex_row] = rounded_point(image)

        target_dimension = reference_cell_topology(chart.target_cell_kind).dimension
        if any(
            isinstance(argument, RationalPolynomial) for argument in chart.chart_arguments
        ):
            rational_chart = tuple(
                argument
                if isinstance(argument, RationalPolynomial)
                else RationalPolynomial(argument, {(0,) * target_dimension: 1})
                for argument in chart.chart_arguments
            )
            element = cell_geometry_owner.RationalComposedCellGeometryElement(
                chart.base_element, chart.target_cell_kind, rational_chart
            )
            key = (
                "rational",
                chart.target_block_index,
                chart.source_block_index,
                chart.target_cell_kind,
                element.element_id,
            )
            rational_elements[key] = element
        else:
            if any(
                not isinstance(argument, Mapping) for argument in chart.chart_arguments
            ):
                raise TypeError(
                    "Mapped layer/core charts must be exact polynomial or rational expressions."
                )
            key = (
                "polynomial",
                chart.target_block_index,
                chart.source_block_index,
                chart.target_cell_kind,
                chart.base_element.element_id,
            )
        groups.setdefault(key, []).append(chart)
    if any(point is None for point in physical_points):
        raise ValueError(
            "Mapped layer/core composition left a target vertex without a root chart."
        )
    points = np.stack([point for point in physical_points if point is not None])

    result_blocks: list[CellBlock] = []
    elements: dict[str, cell_geometry_owner.CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    parent_cells: dict[str, np.ndarray] = {}
    parent_vertices: dict[str, np.ndarray] = {}
    parent_reference_maps: dict[str, ReferenceCellMapWitnesses] = {}
    source_block_names: dict[str, str] = {}
    public_maps: list[ReferenceCellMapWitnesses] = []
    ordered_groups = sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    )
    for group_index, (key, raw_records) in enumerate(ordered_groups):
        records = sorted(raw_records, key=lambda value: value.target_cell_id)
        first = records[0]
        target_name = target_mesh.blocks[first.target_block_index].name
        name = f"{target_name}:mapped:{first.source_block_name}:{group_index}"
        element = (
            rational_elements[key]
            if key[0] == "rational"
            else cell_geometry_owner.PerCellRestrictedCellGeometryElement(
                first.base_element,
                first.target_cell_kind,
                tuple(record.target_cell_id for record in records),
                tuple(record.source_cell_id for record in records),
                tuple(record.chart_arguments for record in records),
            )
        )
        block_vertices = np.stack(
            [np.asarray(record.target_vertices, dtype=np.int64) for record in records]
        )
        target_ids = np.asarray(
            [record.target_cell_id for record in records], dtype=np.int64
        )
        result_blocks.append(
            CellBlock(name, first.target_cell_kind, block_vertices, global_ids=target_ids)
        )
        elements[name] = element
        routes[name] = np.stack([record.source_route for record in records])
        root_ids = np.asarray(
            [record.source_cell_id for record in records], dtype=np.int64
        )
        parent_cells[name] = root_ids
        parent_vertices[name] = np.full(
            (len(records), block_vertices.shape[1]), -1, dtype=np.int64
        )
        source_block_names[name] = first.source_block_name
        target_kind = first.target_cell_kind
        source_kind = first.source_cell_kind
        exact_vertices = tuple(record.source_reference_vertices for record in records)
        floating_vertices = np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in exact_vertices
            ],
            dtype=np.float64,
        )
        local_map = ReferenceCellMapWitnesses(
            np.arange(len(records), dtype=np.int64),
            root_ids.copy(),
            target_kind,
            source_kind,
            floating_vertices,
            exact_vertices,
            source_reference_chart_arguments=freeze_reference_chart_arguments(
                tuple(record.source_reference_chart_arguments for record in records)
            ),
        )
        global_map = ReferenceCellMapWitnesses(
            target_ids.copy(),
            root_ids.copy(),
            target_kind,
            source_kind,
            floating_vertices.copy(),
            exact_vertices,
            source_reference_chart_arguments=freeze_reference_chart_arguments(
                tuple(record.source_reference_chart_arguments for record in records)
            ),
        )
        parent_reference_maps[name] = local_map
        public_maps.append(global_map)

    restriction = cell_geometry_owner.CellGeometryRestrictionSource(
        domain.reference_geometry_id,
        domain.reference_mesh.topology_id,
        parent_cells,
        parent_vertices,
        block_source_blocks=source_block_names,
        block_parent_reference_maps=parent_reference_maps,
    )
    geometry = CellGeometrySpec(
        elements,
        routes,
        np.asarray(domain.source_geometry.coordinates),
        restriction_source=restriction,
    )
    return tuple(result_blocks), geometry, tuple(public_maps), tuple(charts), points


def _validate_mapped_atomic_size_candidate(
    state: TetMesh3D,
    expected_identity: TetMeshIdentity,
    tables: _SourceTables,
    exact_roots: tuple[tuple[Fraction, Fraction, Fraction], ...],
    root_vertex_stencils: tuple[tuple[tuple[int, Fraction], ...], ...],
    original_reference_points: np.ndarray,
    proposal: TetMeshArrays,
    represented_reference_points: np.ndarray,
    schedule: NativeVolumeSchedule,
    /,
) -> np.ndarray:
    """Prove a source-bound native proposal's reference-metric size constraint.

    The original native field is read from the live generation, never inferred
    from request controls or physical scratch interpolation. Each complete
    changed component is an atomic cavity whose native reference-metric
    post-edit size excess must not exceed its actual pre-edit bound.
    """
    from .._meshcore import TetMesh3D, TetMeshIdentity
    from ._volume_generation import _root_normalized_candidate

    if not isinstance(state, TetMesh3D) or not isinstance(
        expected_identity, TetMeshIdentity
    ):
        raise TypeError(
            "Mapped size proof requires its actual native state and identity."
        )
    if not expected_identity.identity_bound or state.identity != expected_identity:
        raise ValueError("Mapped size proof binds a different native generation.")
    before = state.arrays()
    if state.identity != expected_identity or (
        before.generation,
        before.identity_bound,
        before.source_id,
        before.source_revision,
        before.source_map_id,
        before.frame_id,
    ) != tuple(expected_identity):
        raise ValueError("Mapped size snapshot changed its source or generation.")
    for name in (
        "points",
        "faces",
        "face_group_rows",
        "face_ids",
        "face_tolerances",
        "segments",
        "segment_group_rows",
        "segment_ids",
        "segment_tolerances",
        "face_planes",
        "face_plane_tolerances",
    ):
        actual = np.asarray(getattr(before, f"source_{name}"))
        authored = np.asarray(getattr(tables.source, name))
        if (
            actual.shape != authored.shape
            or actual.dtype != authored.dtype
            or not np.array_equal(
                np.ascontiguousarray(actual).view(np.uint8),
                np.ascontiguousarray(authored).view(np.uint8),
            )
        ):
            raise ValueError(
                "Mapped size proof source tables differ from the live native source."
            )
    if (
        not np.array_equal(tables.points, tables.source.points)
        or not np.array_equal(tables.faces, tables.source.faces)
        or not np.array_equal(tables.source_segment_vertices, tables.source.segments)
    ):
        raise ValueError(
            "Mapped size normalization tables differ from their native source banks."
        )
    normalized, _, _ = _root_normalized_candidate(
        before, tables, exact_roots, root_vertex_stencils
    )
    if not np.array_equal(
        np.asarray(original_reference_points, dtype=np.float64).view(np.uint64),
        np.asarray(normalized, dtype=np.float64).view(np.uint64),
    ):
        raise ValueError(
            "Mapped size proof original reference coordinates changed their source ancestry."
        )
    final_sizes, components = _prove_mapped_atomic_size_banks(
        before.vertex_ids,
        original_reference_points,
        before.tetrahedra,
        before.cell_ids,
        before.tetrahedron_regions,
        before.vertex_size_targets,
        before.cell_size_targets,
        proposal.vertex_ids,
        represented_reference_points,
        proposal.tetrahedra,
        proposal.cell_ids,
        proposal.tetrahedron_regions,
        proposal.vertex_ancestry_offsets,
        proposal.vertex_ancestor_ids,
    )
    _require_native_mapped_size_bounds(
        state,
        expected_identity,
        original_reference_points,
        proposal,
        represented_reference_points,
        final_sizes,
        components,
        schedule,
    )
    if state.identity != expected_identity:
        raise ValueError("Mapped size source generation changed during proof.")
    return final_sizes


def _require_native_mapped_size_bounds(
    state: TetMesh3D,
    expected_identity: TetMeshIdentity,
    original_reference_points: np.ndarray,
    proposal: TetMeshArrays,
    represented_reference_points: np.ndarray,
    final_sizes: np.ndarray,
    components: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    schedule: NativeVolumeSchedule,
    /,
) -> None:
    """Compare native reference-metric size residuals per actual changed cavity."""
    from .._meshcore import TetMesh3D

    if state.identity != expected_identity:
        raise ValueError("Mapped size evaluation binds another native generation.")
    before = state.arrays()
    old_quality = state.evaluate_candidate_quality(
        original_reference_points,
        vertex_ids=before.vertex_ids,
        radius_edge_bound=schedule.radius_edge_bound,
        min_dihedral_degrees=schedule.minimum_dihedral_degrees,
        minimum_relative_determinant=schedule.minimum_relative_determinant,
    )
    if (
        old_quality.historical_identity != expected_identity
        or state.identity != expected_identity
    ):
        raise ValueError("Mapped size evaluation changed its original native history.")
    face_labels, face_rows = np.unique(
        np.asarray(proposal.face_sources), return_inverse=True
    )
    segment_labels, segment_rows = np.unique(
        np.asarray(proposal.segment_sources), return_inverse=True
    )
    if (
        len(face_labels) > np.iinfo(np.int32).max
        or len(segment_labels) > np.iinfo(np.int32).max
    ):
        raise ValueError(
            "Mapped size evaluation source label bank exceeds native local rows."
        )
    scratch = TetMesh3D(
        represented_reference_points,
        proposal.tetrahedra,
        proposal.tetrahedron_regions,
        proposal.faces,
        face_rows.astype(np.int32),
        proposal.segments,
        segment_rows.astype(np.int32),
        vertex_ids=proposal.vertex_ids,
        cell_ids=proposal.cell_ids,
        facet_ids=proposal.facet_ids,
        face_orientations=proposal.facet_orientations,
        segment_ids=proposal.segment_ids,
        initial_vertex_sizes=final_sizes,
    )
    try:
        new_quality = scratch.evaluate_quality(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        if new_quality.identity != scratch.identity:
            raise ValueError(
                "Mapped size evaluation lost its independent candidate state."
            )
    finally:
        scratch.close()

    def residuals(
        quality: TetMeshCandidateQuality | TetMeshQuality, known: set[int]
    ) -> dict[int, float]:
        result: dict[int, float] = {}
        for identifier, criterion, reason, value in zip(
            quality.cell_ids,
            quality.criteria,
            quality.reasons,
            quality.values,
            strict=True,
        ):
            cell = int(identifier)
            if cell not in known or int(reason) == 2 or int(criterion) == 4:
                raise ValueError(
                    "Mapped native reference size evaluation is invalid or nonrepresentable."
                )
            if int(criterion) != 1:
                continue
            if cell in result or not np.isfinite(value):
                raise ValueError(
                    "Mapped native size residual is nonrepresentable or misbound."
                )
            result[cell] = float(value)
        return result

    old_ids = {int(value) for value in before.cell_ids}
    new_ids = {int(value) for value in proposal.cell_ids}
    old_values = residuals(old_quality, old_ids)
    new_values = residuals(new_quality, new_ids)
    for old_cells, new_cells in components:
        before_bound = max((old_values.get(cell, 1.0) for cell in old_cells), default=1.0)
        if any(new_values.get(cell, 1.0) > before_bound for cell in new_cells):
            raise ValueError(
                "Mapped size cavity exceeds its actual native non-worsening bound."
            )


def _prove_mapped_atomic_size_banks(
    old_vertex_ids: np.ndarray,
    original_reference_points: np.ndarray,
    original_tetrahedra: np.ndarray,
    original_cell_ids: np.ndarray,
    original_regions: np.ndarray,
    original_size_targets: np.ndarray,
    original_cell_targets: np.ndarray,
    new_vertex_ids: np.ndarray,
    represented_reference_points: np.ndarray,
    proposed_tetrahedra: np.ndarray,
    proposed_cell_ids: np.ndarray,
    proposed_regions: np.ndarray,
    proposal_ancestry_offsets: np.ndarray,
    proposal_ancestor_ids: np.ndarray,
    /,
) -> tuple[np.ndarray, tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]:
    """Reprove exact reference target ancestry and oriented material cavities."""
    from collections import defaultdict

    from ..discretization._exact_power_geometry import _barycentric

    sizes = np.asarray(original_size_targets, dtype=np.float64)
    cell_targets = np.asarray(original_cell_targets, dtype=np.float64)
    old_ids = np.asarray(old_vertex_ids, dtype=np.int64)
    new_ids = np.asarray(new_vertex_ids, dtype=np.int64)
    old_points = np.asarray(original_reference_points, dtype=np.float64)
    new_points = np.asarray(represented_reference_points, dtype=np.float64)
    old_tets = np.asarray(original_tetrahedra, dtype=np.int64)
    new_tets = np.asarray(proposed_tetrahedra, dtype=np.int64)
    old_cells = np.asarray(original_cell_ids, dtype=np.int64)
    new_cells = np.asarray(proposed_cell_ids, dtype=np.int64)
    old_regions = np.asarray(original_regions, dtype=np.int64)
    new_regions = np.asarray(proposed_regions, dtype=np.int64)
    ancestry_offsets = np.asarray(proposal_ancestry_offsets, dtype=np.int64)
    ancestor_ids = np.asarray(proposal_ancestor_ids, dtype=np.int64)
    if (
        sizes.shape != (len(old_ids),)
        or old_points.shape != (len(old_ids), 3)
        or new_points.shape != (len(new_ids), 3)
        or old_tets.shape != (len(old_cells), 4)
        or new_tets.shape != (len(new_cells), 4)
        or old_regions.shape != old_cells.shape
        or new_regions.shape != new_cells.shape
        or cell_targets.shape != old_cells.shape
        or np.any(~np.isfinite(cell_targets))
        or np.any(cell_targets < 0)
        or ancestry_offsets.shape != (len(new_ids) + 1,)
        or ancestor_ids.ndim != 1
        or ancestry_offsets[0] != 0
        or ancestry_offsets[-1] != len(ancestor_ids)
        or np.any(np.diff(ancestry_offsets) < 0)
        or len(old_tets) == 0
        or np.any(~np.isfinite(sizes))
        or np.any(sizes < 0)
        or np.any(~np.isfinite(old_points))
        or np.any(~np.isfinite(new_points))
        or np.any(old_tets < 0)
        or np.any(old_tets >= len(old_ids))
        or np.any(new_tets < 0)
        or np.any(new_tets >= len(new_ids))
        or len(set(old_ids.tolist())) != len(old_ids)
        or len(set(new_ids.tolist())) != len(new_ids)
        or len(set(old_cells.tolist())) != len(old_cells)
        or len(set(new_cells.tolist())) != len(new_cells)
    ):
        raise ValueError(
            "Mapped size proof has malformed native vertex, cell or size banks."
        )
    for cell, tetrahedron in enumerate(old_tets):
        corner_sizes = sizes[tetrahedron]
        target = float(cell_targets[cell])
        if np.any(corner_sizes == 0):
            if target != 0:
                raise ValueError(
                    "Mapped native cell target contradicts an absent corner target."
                )
        else:
            maximum = float(np.max(corner_sizes))
            lower = float(np.nextafter(0.25 * maximum, 0.0))
            if target <= 0 or target > maximum or target < lower:
                raise ValueError(
                    "Mapped native cell target contradicts its retained corner targets."
                )

    def exact_point(point: np.ndarray) -> tuple[Fraction, Fraction, Fraction]:
        return (
            Fraction.from_float(float(point[0])),
            Fraction.from_float(float(point[1])),
            Fraction.from_float(float(point[2])),
        )

    old_by_id = {int(identifier): row for row, identifier in enumerate(old_ids)}
    new_by_id = {int(identifier): row for row, identifier in enumerate(new_ids)}
    final_sizes = np.empty(len(new_ids), dtype=np.float64)
    boxes_lower = np.min(old_points[old_tets], axis=1)
    boxes_upper = np.max(old_points[old_tets], axis=1)
    for row, identifier in enumerate(new_ids):
        old_row = old_by_id.get(int(identifier))
        if old_row is not None:
            final_sizes[row] = sizes[old_row]
            continue
        point = new_points[row]
        declared = tuple(
            int(value)
            for value in ancestor_ids[ancestry_offsets[row] : ancestry_offsets[row + 1]]
        )
        if len(set(declared)) != len(declared) or any(
            value not in old_by_id for value in declared
        ):
            raise ValueError("Mapped size insertion has invalid native vertex ancestry.")
        candidates = np.flatnonzero(
            np.all((boxes_lower <= point) & (point <= boxes_upper), axis=1)
        )
        targets: list[float] = []
        for cell in candidates:
            tetrahedron = old_tets[int(cell)]
            if declared and not set(declared) <= {
                int(old_ids[index]) for index in tetrahedron
            }:
                continue
            weights = _barycentric(
                exact_point(point),
                tuple(exact_point(old_points[index]) for index in tetrahedron),
            )
            if weights is not None and all(weight >= 0 for weight in weights):
                targets.append(float(cell_targets[int(cell)]))
        if not targets:
            raise ValueError(
                "Mapped size insertion lacks an exact original reference cell."
            )
        if any(
            np.float64(value).view(np.uint64) != np.float64(targets[0]).view(np.uint64)
            for value in targets[1:]
        ):
            raise ValueError("Mapped size insertion has ambiguous original-cell targets.")
        final_sizes[row] = targets[0]

    changed_vertices = set(old_by_id) ^ set(new_by_id)
    changed_vertices.update(
        identifier
        for identifier in set(old_by_id) & set(new_by_id)
        if not np.array_equal(
            old_points[old_by_id[identifier]].view(np.uint64),
            new_points[new_by_id[identifier]].view(np.uint64),
        )
    )
    old_cell_rows = {int(identifier): row for row, identifier in enumerate(old_cells)}
    new_cell_rows = {int(identifier): row for row, identifier in enumerate(new_cells)}
    old_incidence = {
        int(identifier): (
            int(old_regions[row]),
            frozenset(int(old_ids[index]) for index in old_tets[row]),
        )
        for row, identifier in enumerate(old_cells)
    }
    new_incidence = {
        int(identifier): (
            int(new_regions[row]),
            frozenset(int(new_ids[index]) for index in new_tets[row]),
        )
        for row, identifier in enumerate(new_cells)
    }
    changed_cells = set(old_cell_rows) ^ set(new_cell_rows)
    changed_cells.update(
        identifier
        for identifier in set(old_cell_rows) & set(new_cell_rows)
        if old_incidence[identifier] != new_incidence[identifier]
        or bool(old_incidence[identifier][1] & changed_vertices)
        or bool(new_incidence[identifier][1] & changed_vertices)
    )
    old_affected = [
        old_cell_rows[identifier]
        for identifier in changed_cells
        if identifier in old_cell_rows
    ]
    new_affected = [
        new_cell_rows[identifier]
        for identifier in changed_cells
        if identifier in new_cell_rows
    ]
    if not old_affected or not new_affected:
        raise ValueError("Mapped size proof requires a complete nonempty changed cavity.")

    def cycle(values: tuple[int, int, int]) -> tuple[int, int, int]:
        return min(values, values[1:] + values[:1], values[2:] + values[:2])

    face_local = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    nodes = [(0, row) for row in old_affected] + [(1, row) for row in new_affected]
    parent = list(range(len(nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    face_nodes: dict[tuple[int, tuple[int, int, int]], list[int]] = defaultdict(list)
    node_faces: list[
        list[tuple[tuple[int, tuple[int, int, int]], tuple[int, int, int]]]
    ] = []
    for index, (side, row) in enumerate(nodes):
        ids = old_ids if side == 0 else new_ids
        tetrahedron = old_tets[row] if side == 0 else new_tets[row]
        region = int(old_regions[row] if side == 0 else new_regions[row])
        if region < 0:
            raise ValueError("Mapped size cavity has an undeclared material.")
        vertices = tuple(int(ids[int(value)]) for value in tetrahedron)
        if len(set(vertices)) != 4:
            raise ValueError("Mapped size cavity has a degenerate tetrahedron.")
        faces = []
        for local in face_local:
            oriented = (vertices[local[0]], vertices[local[1]], vertices[local[2]])
            ordered = sorted(oriented)
            key = (region, (ordered[0], ordered[1], ordered[2]))
            faces.append((key, cycle(oriented)))
            face_nodes[key].append(index)
        node_faces.append(faces)
    for indices in face_nodes.values():
        representative = find(indices[0])
        for index in indices[1:]:
            parent[find(index)] = representative
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(nodes)):
        components[find(index)].append(index)
    cavity_cells: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for component in components.values():
        banks: tuple[
            dict[tuple[int, tuple[int, int, int]], list[tuple[int, int, int]]], ...
        ] = ({}, {})
        for index in component:
            side, _ = nodes[index]
            for key, orientation in node_faces[index]:
                banks[side].setdefault(key, []).append(orientation)
        boundaries = []
        for bank in banks:
            boundary = {}
            for key, orientations in bank.items():
                if len(orientations) == 1:
                    boundary[key] = orientations[0]
                elif len(orientations) != 2 or orientations[0] != cycle(
                    (orientations[1][0], orientations[1][2], orientations[1][1])
                ):
                    raise ValueError(
                        "Mapped size cavity has duplicate or inconsistent oriented faces."
                    )
            boundaries.append(boundary)
        if not boundaries[0] or boundaries[0] != boundaries[1]:
            raise ValueError("Mapped size cavity changed its oriented material boundary.")
        if any(
            identifier in changed_vertices
            for _, face in boundaries[0]
            for identifier in face
        ):
            raise ValueError("Mapped size cavity moved a boundary vertex.")
        cavity_cells.append(
            (
                tuple(
                    sorted(
                        int(old_cells[nodes[index][1]])
                        for index in component
                        if nodes[index][0] == 0
                    )
                ),
                tuple(
                    sorted(
                        int(new_cells[nodes[index][1]])
                        for index in component
                        if nodes[index][0] == 1
                    )
                ),
            )
        )

    for row in new_affected:
        indices = new_tets[row]
        points = tuple(exact_point(new_points[index]) for index in indices)
        edges = tuple(
            tuple(points[i][axis] - points[0][axis] for axis in range(3))
            for i in (1, 2, 3)
        )
        determinant = (
            edges[0][0] * (edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1])
            - edges[0][1] * (edges[1][0] * edges[2][2] - edges[1][2] * edges[2][0])
            + edges[0][2] * (edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0])
        )
        if determinant <= 0:
            raise ValueError(
                "Mapped size cavity contains a nonpositive reference tetrahedron."
            )
    return final_sizes, tuple(cavity_cells)


def _represent_mapped_proposal_vertices(
    domain: MappedReferenceDomain,
    old_ids: np.ndarray,
    old_reference: np.ndarray,
    old_physical: np.ndarray,
    proposal: TetMeshArrays,
    /,
) -> np.ndarray:
    """Invert only changed proposal points against the actual complete root bank."""
    from ..discretization._mapped_locator import PreparedMappedCellLocator

    old_rows = {int(identifier): row for row, identifier in enumerate(old_ids)}
    new_ids = np.asarray(proposal.vertex_ids, dtype=np.int64)
    represented = np.empty_like(np.asarray(proposal.points, dtype=np.float64))
    changed = []
    for row, identifier in enumerate(new_ids):
        old = old_rows.get(int(identifier))
        if old is not None and np.array_equal(
            np.asarray(proposal.points[row]).view(np.uint64),
            np.asarray(old_physical[old]).view(np.uint64),
        ):
            represented[row] = old_reference[old]
        else:
            changed.append(row)
    if changed:
        points = np.asarray(proposal.points, dtype=np.float64)[changed]
        located: list[list[tuple[int, int, np.ndarray]]] = [[] for _ in changed]
        for block_index, block in enumerate(domain.reference_mesh.blocks):
            locator = PreparedMappedCellLocator.from_cell_geometry(
                domain.reference_mesh, domain.source_geometry, block_index
            )
            result = locator.locate(points)
            if not np.all(np.asarray(result.candidates_complete, dtype=np.bool_)):
                raise ValueError("Mapped improvement inverse-root search is incomplete.")
            for index in np.flatnonzero(np.asarray(result.successful, dtype=np.bool_)):
                if int(np.asarray(result.candidate_count)[index]) != 1:
                    raise ValueError(
                        "Mapped improvement point has ambiguous root ancestry."
                    )
                located[int(index)].append(
                    (
                        block_index,
                        int(np.asarray(result.cell_ids)[index]),
                        np.asarray(result.barycentric[index], dtype=np.float64),
                    )
                )
        coordinates = np.asarray(domain.reference_mesh.coordinates, dtype=np.float64)
        for index, candidates in enumerate(located):
            if len(candidates) != 1:
                raise ValueError(
                    "Mapped improvement point lacks one unique actual root cell."
                )
            block_index, cell, weights = candidates[0]
            rows = np.asarray(
                domain.reference_mesh.blocks[block_index].vertices[cell], dtype=np.int64
            )
            represented[changed[index]] = weights @ coordinates[rows]
    return represented


def _propose_mapped_physical_improvement(
    domain: MappedReferenceDomain,
    reference_arrays: TetMeshArrays,
    core_region_map: np.ndarray,
    protected_input_vertex_ids: np.ndarray,
    schedule: NativeVolumeSchedule,
    /,
) -> tuple[TetMeshArrays, np.ndarray, TetMeshRun, TetMeshQuality] | None:
    """Propose, but do not certify, a source-root-representable physical improvement."""
    from .._meshcore import MeshcoreStatus, TetMesh3D, TetMeshArrays, TetMeshEditStatus
    from ._volume_generation import _unmet_rows, NativeVolumeSchedule

    if not isinstance(reference_arrays, TetMeshArrays) or not isinstance(
        schedule, NativeVolumeSchedule
    ):
        raise TypeError(
            "Mapped physical improvement requires actual native arrays and volume schedule."
        )
    region_map = np.asarray(core_region_map, dtype=np.int64)
    protected_ids = np.asarray(protected_input_vertex_ids, dtype=np.int64)
    if region_map.ndim != 1 or protected_ids.ndim != 1:
        raise ValueError(
            "Mapped improvement region and protected input banks must be vectors."
        )

    def reference_mesh(
        arrays: TetMeshArrays, points: np.ndarray
    ) -> tuple[CellMesh, np.ndarray]:
        labels = np.asarray(arrays.tetrahedron_regions, dtype=np.int64)
        if np.any(labels < 0) or np.any(labels >= len(region_map)):
            raise ValueError("Mapped improvement introduced an undeclared core material.")
        mesh = CellMesh(
            points,
            (
                CellBlock(
                    "mapped-core-proposal",
                    "tetrahedron",
                    arrays.tetrahedra,
                    global_ids=arrays.cell_ids,
                ),
            ),
            vertex_global_ids=arrays.vertex_ids,
        )
        return mesh, region_map[labels]

    def mapped_points(arrays: TetMeshArrays, points: np.ndarray) -> np.ndarray:
        mesh, regions = reference_mesh(arrays, points)
        geometry = CellGeometrySpec.affine(mesh)
        return _geometry_groups(domain, mesh, geometry, regions)[4]

    old_ids = np.asarray(reference_arrays.vertex_ids, dtype=np.int64)
    if len(set(old_ids.tolist())) != len(old_ids) or not set(
        protected_ids.tolist()
    ) <= set(old_ids.tolist()):
        raise ValueError("Mapped improvement lost original protected vertex identities.")
    old_reference = np.asarray(reference_arrays.points, dtype=np.float64)
    old_physical = mapped_points(reference_arrays, old_reference)
    boundary_rows = np.unique(
        np.concatenate(
            (
                np.asarray(reference_arrays.faces, dtype=np.int64).reshape(-1),
                np.asarray(reference_arrays.segments, dtype=np.int64).reshape(-1),
            )
        )
    )
    fixed_ids = (
        set(protected_ids.tolist())
        | set(old_ids[boundary_rows].tolist())
        | set(
            old_ids[
                np.asarray(reference_arrays.protected_vertices, dtype=np.bool_)
            ].tolist()
        )
        | set(old_ids[np.asarray(reference_arrays.protection_radii) > 0].tolist())
    )
    fixed_mask = np.isin(old_ids, tuple(fixed_ids))

    def native(
        points: np.ndarray, arrays: TetMeshArrays, protected: np.ndarray | None = None
    ) -> tuple[TetMesh3D, np.ndarray, np.ndarray]:
        face_labels, face_rows = np.unique(
            np.asarray(arrays.face_sources), return_inverse=True
        )
        segment_labels, segment_rows = np.unique(
            np.asarray(arrays.segment_sources), return_inverse=True
        )
        if (
            len(face_labels) > np.iinfo(np.int32).max
            or len(segment_labels) > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Mapped improvement source label bank exceeds native local rows."
            )
        return (
            TetMesh3D(
                points,
                arrays.tetrahedra,
                arrays.tetrahedron_regions,
                arrays.faces,
                face_rows.astype(np.int32),
                arrays.segments,
                segment_rows.astype(np.int32),
                protected_vertices=protected,
                vertex_ids=arrays.vertex_ids,
                cell_ids=arrays.cell_ids,
                facet_ids=arrays.facet_ids,
                face_orientations=arrays.facet_orientations,
                segment_ids=arrays.segment_ids,
            ),
            face_labels,
            segment_labels,
        )

    state, face_labels, segment_labels = native(
        old_physical, reference_arrays, fixed_mask
    )
    try:
        initial_quality = state.evaluate_quality(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        if not _unmet_rows(initial_quality):
            return None
        initial_identity = state.identity
        run = state.improve(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        if run.edit_status is not TetMeshEditStatus.APPLIED or run.status not in (
            MeshcoreStatus.OK,
            MeshcoreStatus.UNRESOLVED,
        ):
            raise ValueError(
                "Native mapped physical improvement did not produce a valid proposal."
            )
        if state.identity == initial_identity:
            return None
        proposal = state.arrays()
        face_rows = np.asarray(proposal.face_sources, dtype=np.int64)
        segment_rows = np.asarray(proposal.segment_sources, dtype=np.int64)
        if (
            np.any(face_rows < 0)
            or np.any(face_rows >= len(face_labels))
            or np.any(segment_rows < 0)
            or np.any(segment_rows >= len(segment_labels))
        ):
            raise ValueError(
                "Mapped improvement returned an unknown native source label."
            )
        proposal = proposal._replace(
            face_sources=face_labels[face_rows],
            segment_sources=segment_labels[segment_rows],
        )
    finally:
        state.close()

    def boundary_bank(
        arrays: TetMeshArrays, rows: np.ndarray, labels: np.ndarray, ids: np.ndarray
    ) -> set[tuple]:
        vertices = np.asarray(arrays.vertex_ids, dtype=np.int64)
        return {
            (
                int(identifier),
                int(label),
                tuple(sorted(int(vertices[int(row)]) for row in item)),
            )
            for item, label, identifier in zip(rows, labels, ids, strict=True)
        }

    if boundary_bank(
        reference_arrays,
        reference_arrays.faces,
        reference_arrays.face_sources,
        reference_arrays.facet_ids,
    ) != boundary_bank(
        proposal, proposal.faces, proposal.face_sources, proposal.facet_ids
    ) or boundary_bank(
        reference_arrays,
        reference_arrays.segments,
        reference_arrays.segment_sources,
        reference_arrays.segment_ids,
    ) != boundary_bank(
        proposal, proposal.segments, proposal.segment_sources, proposal.segment_ids
    ):
        raise ValueError(
            "Mapped physical improvement changed an original boundary facet or segment."
        )
    old_rows = {int(identifier): row for row, identifier in enumerate(old_ids)}
    new_ids = np.asarray(proposal.vertex_ids, dtype=np.int64)
    new_rows = {int(identifier): row for row, identifier in enumerate(new_ids)}
    if len(new_rows) != len(new_ids) or not fixed_ids <= set(new_rows):
        raise ValueError("Mapped physical improvement removed a protected input vertex.")
    for identifier in fixed_ids:
        if not np.array_equal(
            np.asarray(proposal.points[new_rows[identifier]]).view(np.uint64),
            np.asarray(old_physical[old_rows[identifier]]).view(np.uint64),
        ):
            raise ValueError(
                "Mapped physical improvement moved a protected boundary or input vertex."
            )

    represented = _represent_mapped_proposal_vertices(
        domain, old_ids, old_reference, old_physical, proposal
    )

    reference_state, _, _ = native(represented, proposal)
    reference_state.close()

    recomposed = mapped_points(proposal, represented)
    for identifier in fixed_ids:
        if not np.array_equal(
            np.asarray(recomposed[new_rows[identifier]]).view(np.uint64),
            np.asarray(old_physical[old_rows[identifier]]).view(np.uint64),
        ):
            raise ValueError(
                "Recomposed mapped improvement changed a protected source vertex."
            )
    quality_state, _, _ = native(recomposed, proposal)
    try:
        quality = quality_state.evaluate_quality(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
    finally:
        quality_state.close()
    return proposal, represented, run, quality


def _rebind_organization(
    old_mesh: CellMesh,
    new_mesh: CellMesh,
    zones: tuple[organization_owner.MeshZone, ...],
    patches: tuple[organization_owner.MeshPatch, ...],
    labels: tuple[organization_owner.MeshLabel, ...],
    attributes: tuple[organization_owner.MeshAttribute, ...],
    /,
) -> tuple[
    tuple[organization_owner.MeshZone, ...],
    tuple[organization_owner.MeshPatch, ...],
    tuple[organization_owner.MeshLabel, ...],
    tuple[organization_owner.MeshAttribute, ...],
    tuple[dict[int, int], ...],
]:
    entity_maps: list[dict[int, int]] = []
    for dimension in range(old_mesh.topological_dimension + 1):
        old_ids = np.asarray(old_mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        new_ids = np.asarray(new_mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        if dimension in (0, old_mesh.topological_dimension):
            if not np.array_equal(np.sort(old_ids), np.sort(new_ids)):
                raise ValueError(
                    "Mapped composition changed explicit source vertex or cell identities."
                )
            entity_maps.append({int(value): int(value) for value in old_ids.tolist()})
            continue
        old_keys = canonical_owner._entity_vertex_keys(old_mesh, dimension)
        new_keys = canonical_owner._entity_vertex_keys(new_mesh, dimension)
        new_by_key = dict(zip(new_keys, new_ids.tolist(), strict=True))
        if set(old_keys) != set(new_by_key):
            raise ValueError(
                "Mapped composition changed a shared source entity incidence key."
            )
        entity_maps.append(
            {
                int(identifier): int(new_by_key[key])
                for identifier, key in zip(old_ids.tolist(), old_keys, strict=True)
            }
        )

    def scope(original: MeshingScope) -> MeshingScope:
        dimension = original.entity_dimension
        identifiers = np.asarray(original.entity_ids, dtype=np.int64)
        remapped = np.asarray(
            [
                entity_maps[dimension][int(identifier)]
                for identifier in identifiers.tolist()
            ],
            dtype=np.int64,
        )
        return layer_topology.mesh_entity_scope(new_mesh, dimension, remapped)

    rebound_zones = tuple(
        organization_owner.MeshZone(
            zone.name,
            zone.role,
            scope(zone.scope),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        for zone in zones
    )
    zone_ids = {
        previous.zone_id: current.zone_id
        for previous, current in zip(zones, rebound_zones, strict=True)
    }
    rebound_patches = tuple(
        organization_owner.MeshPatch(
            patch.name,
            scope(patch.scope),
            connected=patch.connected,
            adjacent_zone_ids=tuple(zone_ids[value] for value in patch.adjacent_zone_ids),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
            source_adjacent_region_occurrence_paths=patch.source_adjacent_region_occurrence_paths,
        )
        for patch in patches
    )
    rebound_labels = tuple(
        organization_owner.MeshLabel(label.name, scope(label.scope)) for label in labels
    )
    rebound_attributes = []
    for attribute in attributes:
        identifiers = np.asarray(attribute.scope.entity_ids, dtype=np.int64)
        remapped = np.asarray(
            [
                entity_maps[attribute.scope.entity_dimension][int(value)]
                for value in identifiers.tolist()
            ],
            dtype=np.int64,
        )
        order = np.argsort(remapped, kind="stable")
        rebound_attributes.append(
            organization_owner.MeshAttribute(
                attribute.name,
                attribute.role,
                scope(attribute.scope),
                np.asarray(attribute.values)[order],
                unit=attribute.unit,
            )
        )
    return (
        rebound_zones,
        rebound_patches,
        rebound_labels,
        tuple(rebound_attributes),
        tuple(entity_maps),
    )


def _rebind_association(
    association: association_owner.GeometryAssociation,
    old_mesh: CellMesh,
    new_mesh: CellMesh,
    entity_maps: tuple[dict[int, int], ...],
    /,
) -> association_owner.GeometryAssociation:
    dimension = next(
        (
            value
            for value in range(old_mesh.topological_dimension + 1)
            if old_mesh.entity_set(value).entity_set_id
            == association.target_entity_set_id
        ),
        None,
    )
    if dimension is None:
        raise ValueError(
            "Mapped source association is not bound to a construction entity set."
        )
    if association.source_boundary_witness_id is not None:
        raise ValueError("Mapped face pieces require a composed target witness.")
    if any(interval is not None for interval in association.target_parameter_intervals):
        raise ValueError(
            "Mapped partial-edge associations require an exact target orientation map."
        )
    target_ids = np.asarray(association.target_global_ids, dtype=np.int64)
    mapped_targets = np.asarray(
        [entity_maps[dimension][int(value)] for value in target_ids.tolist()],
        dtype=np.int64,
    )
    parent_dimensions = np.asarray(association.parent_dimensions, dtype=np.int8)
    parent_ids = np.asarray(association.parent_ids, dtype=np.int64).copy()
    for row, parent_dimension in enumerate(parent_dimensions.tolist()):
        if parent_dimension >= 0:
            parent_ids[row] = entity_maps[int(parent_dimension)][int(parent_ids[row])]
    dimensions_array = np.asarray(association.source_dimensions, dtype=np.int8)
    source_dimensions: np.ndarray | None = dimensions_array
    source_indices: np.ndarray | None = np.asarray(
        association.source_indices, dtype=np.int64
    )
    if (
        association.association_kind is not association_owner.GeometryAssociationKind.BREP
        and not np.any(dimensions_array >= 0)
    ):
        source_dimensions = None
        source_indices = None
    return association_owner.GeometryAssociation(
        association.association_kind,
        association.source_id,
        association.source_revision,
        new_mesh.entity_set(dimension).entity_set_id,
        mapped_targets,
        association.source_entity_ids,
        np.asarray(association.residuals),
        resolved=np.asarray(association.resolved),
        ambiguous=np.asarray(association.ambiguous),
        exact=association.exact,
        source_dimensions=source_dimensions,
        source_indices=source_indices,
        source_roles=association.source_roles,
        source_occurrence_paths=association.source_occurrence_paths,
        source_region_sides=association.source_region_sides,
        source_parameter_intervals=association.source_parameter_intervals,
        target_parameter_intervals=association.target_parameter_intervals,
        source_parameter_points=association.source_parameter_points,
        parameters=np.asarray(association.parameters),
        orientations=np.asarray(association.orientations),
        parent_dimensions=parent_dimensions,
        parent_ids=parent_ids,
        parent_association_id=association.parent_association_id,
        provenance=association.provenance,
    )


def _entity_on_root_boundary(
    mesh: CellMesh,
    chart: _CellChart,
    entity_key: tuple[int, ...],
    /,
) -> bool:
    """Test target-entity corners against exact root-cell facet planes."""
    root_vertices = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology(chart.source_cell_kind).vertices
    )
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
        np.asarray(chart.target_vertices, dtype=np.int64)
    ]
    if target_vertex_ids.size != len(chart.source_reference_vertices):
        raise ValueError("Mapped chart vertices must align with exact root coordinates.")
    root_by_target_vertex = {
        int(identifier): tuple(point)
        for identifier, point in zip(
            target_vertex_ids.tolist(),
            chart.source_reference_vertices,
            strict=True,
        )
    }
    try:
        entity_points = tuple(
            root_by_target_vertex[identifier] for identifier in entity_key
        )
    except KeyError as error:
        raise ValueError(
            "A mapped source entity is absent from its cell chart."
        ) from error
    topology = reference_cell_topology(chart.source_cell_kind)
    for local_face in topology.entities[2]:
        face_points = tuple(root_vertices[int(index)] for index in local_face)
        origin = face_points[0]
        first = tuple(
            value - base for value, base in zip(face_points[1], origin, strict=True)
        )
        second = tuple(
            value - base for value, base in zip(face_points[2], origin, strict=True)
        )
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        if normal == (Fraction(0), Fraction(0), Fraction(0)):
            raise ValueError(
                "A mapped root reference facet has degenerate exact coordinates."
            )
        if all(
            sum(
                (normal[axis] * (point[axis] - origin[axis]) for axis in range(3)),
                Fraction(0),
            )
            == 0
            for point in entity_points
        ):
            return True
    return False


def _mapped_root_associations(
    domain: MappedReferenceDomain,
    mesh: CellMesh,
    controls: tuple[_CellChart, ...],
    /,
) -> tuple[association_owner.GeometryAssociation, ...]:
    """Associate root-interior entities with their unique source region only."""
    control_by_target_id = {chart.target_cell_id: chart for chart in controls}
    if len(control_by_target_id) != sum(block.cell_count for block in mesh.blocks):
        raise ValueError("Mapped root charts must cover every output cell exactly once.")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    root_paths = dict(
        zip(domain.root_cell_ids, domain.root_cell_occurrence_paths, strict=True)
    )
    associations: list[association_owner.GeometryAssociation] = []
    for dimension in range(mesh.topological_dimension + 1):
        entity_set = mesh.entity_set(dimension)
        entity_ids = np.asarray(entity_set.entity_ids, dtype=np.int64)
        if dimension == mesh.topological_dimension:
            owners = {chart.target_cell_id: chart.source_cell_id for chart in controls}
            if set(owners) != set(entity_ids.tolist()):
                raise ValueError(
                    "Mapped source-root cell identities do not cover output cells."
                )
        else:
            entity_keys = (
                tuple((int(identifier),) for identifier in entity_ids.tolist())
                if dimension == 0
                else canonical_owner._entity_vertex_keys(mesh, dimension)
            )
            key_to_id = dict(zip(entity_keys, entity_ids.tolist(), strict=True))
            adjacent: dict[tuple[int, ...], list[_CellChart]] = {}
            for chart in controls:
                block = mesh.blocks[chart.target_block_index]
                local_vertices = np.asarray(
                    block.vertices[chart.target_row], dtype=np.int64
                )
                global_vertices = vertex_ids[local_vertices]
                for local_entity in reference_cell_topology(
                    chart.target_cell_kind
                ).entities[dimension]:
                    key = tuple(
                        sorted(
                            int(value)
                            for value in global_vertices[
                                np.asarray(local_entity, dtype=np.int64)
                            ].tolist()
                        )
                    )
                    adjacent.setdefault(key, []).append(chart)
            if set(adjacent) != set(key_to_id):
                raise ValueError("Mapped root charts do not cover every output entity.")
            owners: dict[int, int] = {}
            for key, charts in adjacent.items():
                source_cells = {chart.source_cell_id for chart in charts}
                if len(source_cells) != 1 or any(
                    _entity_on_root_boundary(mesh, chart, key) for chart in charts
                ):
                    continue
                owners[key_to_id[key]] = next(iter(source_cells))
        if not owners:
            continue
        target_ids = np.asarray(sorted(owners), dtype=np.int64)
        parent_ids = np.asarray(
            [owners[int(identifier)] for identifier in target_ids], dtype=np.int64
        )
        paths = tuple(root_paths[int(identifier)] for identifier in parent_ids)
        source_ids = tuple(
            domain.image_entity_id(3, int(identifier)) for identifier in parent_ids
        )
        associations.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.MAPPED_REFERENCE,
                domain.source_id,
                domain.source_revision,
                entity_set.entity_set_id,
                target_ids,
                source_ids,
                np.zeros(target_ids.shape, dtype=np.float64),
                exact=True,
                source_dimensions=np.full(target_ids.shape, 3, dtype=np.int8),
                source_indices=parent_ids,
                source_roles=(association_owner.GeometrySourceEntityRole.REGION,)
                * target_ids.size,
                source_occurrence_paths=paths,
                parent_dimensions=np.full(target_ids.shape, 3, dtype=np.int8),
                parent_ids=parent_ids,
            )
        )
    return tuple(associations)


def compose_mapped_layer_core(
    source: sources_owner.NativeLayerCoreSource,
    reference_construction: layer_topology.LayerCoreConstruction,
    /,
) -> layer_topology.LayerCoreConstruction:
    """Compose an exact reference fill through source roots without physical inversion."""
    domain = source.mapped_domain
    if not isinstance(domain, MappedReferenceDomain):
        raise TypeError(
            "Mapped layer/core composition requires its original MappedReferenceDomain."
        )
    if (
        source.reference_source is None
        or source.reference_source.source_id != domain.source_id
        or source.reference_source.source_revision != domain.source_revision
        or source.source_id != domain.source_id
        or source.source_revision != domain.source_revision
    ):
        raise ValueError(
            "Mapped layer/core source identity changed across its reference roots."
        )
    validate_layer_reference_columns(source, source.reference_source, domain)
    target_mesh = reference_construction.mesh
    block_regions = _block_order_regions(target_mesh, reference_construction.cell_regions)
    blocks, geometry, reference_maps, charts, physical_points = _geometry_groups(
        domain,
        target_mesh,
        reference_construction.geometry,
        block_regions,
    )
    explicit_entities = {
        dimension: np.asarray(
            target_mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        for dimension in range(1, target_mesh.topological_dimension)
    }
    physical_mesh = CellMesh(
        physical_points,
        blocks,
        vertex_global_ids=np.asarray(target_mesh.vertex_global_ids, dtype=np.int64),
        entity_global_ids=explicit_entities,
        numeric_version=domain.source_revision,
    )
    reference_periodic = target_mesh.periodic_topology
    physical_periodic = source.layers.mesh.periodic_topology
    if (reference_periodic is None) != (physical_periodic is None):
        raise ValueError(
            "Mapped source and reference layers must retain the same periodic topology."
        )
    if reference_periodic is not None and physical_periodic is not None:
        topology = periodic_topology_owner.PeriodicMeshTopology(
            physical_mesh,
            physical_periodic.cell,
            reference_periodic.vertex_representatives,
            reference_periodic.vertex_shifts,
        )
        physical_mesh = CellMesh(
            physical_mesh.coordinates,
            physical_mesh.blocks,
            vertex_global_ids=physical_mesh.vertex_global_ids,
            entity_global_ids=explicit_entities,
            numeric_version=physical_mesh.numeric_version,
            periodic_topology=topology,
        )
    zones, patches, labels, attributes, entity_maps = _rebind_organization(
        target_mesh,
        physical_mesh,
        reference_construction.zones,
        reference_construction.patches,
        reference_construction.labels,
        reference_construction.attributes,
    )
    associations = tuple(
        _rebind_association(value, target_mesh, physical_mesh, entity_maps)
        for value in reference_construction.associations
    ) + _mapped_root_associations(domain, physical_mesh, charts)
    return layer_topology.LayerCoreConstruction(
        physical_mesh,
        domain,
        np.asarray(reference_construction.cell_regions, dtype=np.int64),
        zones,
        patches,
        labels,
        attributes,
        associations,
        geometry,
        reference_construction.core,
        np.asarray(reference_construction.core_vertex_map, dtype=np.int64),
        np.asarray(reference_construction.core_cell_ids, dtype=np.int64),
        reference_construction.region_ids,
        source.source_id,
        source.source_revision,
        physical_mesh.periodic_topology,
        reference_construction.periodic_requested,
        reference_construction.periodic_achieved,
        reference_maps,
        domain.reference_mesh,
        domain.source_geometry,
        reference_mesh=target_mesh,
        reference_geometry=reference_construction.geometry,
        polyhedral_core=reference_construction.polyhedral_core,
        overlap_exact_geometry=reference_construction.overlap_exact_geometry,
        overlap_source_stencils=reference_construction.overlap_source_stencils,
        layer_core_column_certification=reference_construction.layer_core_column_certification,
        layer_root_partition=reference_construction.layer_root_partition,
        execution_carrier=reference_construction.execution_carrier,
    )


__all__ = ["compose_mapped_layer_core", "validate_layer_reference_columns"]
