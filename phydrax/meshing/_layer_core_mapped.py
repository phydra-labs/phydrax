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
from ..discretization._coordinate_enclosure import (
    add,
    BernsteinDomain,
    compose_expression,
    coordinate_map_expressions,
    derivative,
    evaluate_expression,
    ExactScalar,
    Expression,
    expression_equal,
    ExpressionInput,
    polynomial_bounds,
    RationalPolynomial,
    rounded_point,
    scale,
)
from ..discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ._scope import MeshingScope


if TYPE_CHECKING:
    import phydrax.meshing.providers._native_sources as sources_owner


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
    source_dimensions: np.ndarray | None = np.asarray(
        association.source_dimensions, dtype=np.int8
    )
    source_indices: np.ndarray | None = np.asarray(
        association.source_indices, dtype=np.int64
    )
    if (
        association.association_kind is not association_owner.GeometryAssociationKind.BREP
        and not np.any(source_dimensions >= 0)
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
    )


__all__ = ["compose_mapped_layer_core", "validate_layer_reference_columns"]
