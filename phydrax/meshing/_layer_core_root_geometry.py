#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One independently reproved root geometry for conforming layer transitions."""

from __future__ import annotations

from fractions import Fraction
from typing import final, NoReturn, SupportsIndex, TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._cell_geometry_transition_types import (
    _geometry_map_expressions,
    _reference_map_identity,
    _require_exact_reference_group,
)
from ..discretization._nested_reference_types import (
    exact_reference_cell_vertices,
    ReferenceCellPartition,
    validate_reference_cell_partition,
)
from ._contracts import VolumeMeshingSpec
from ._layer_core_carrier_witness import require_layer_core_specification
from ._layer_core_subdivision import (
    OverlapSourceStencils,
    RefinedColumnCertification,
    _require_overlap_provenance,
    require_refined_column_certification,
)

if TYPE_CHECKING:
    from .providers._native_sources import NativeLayerCoreSource


@final
class LayerCoreRootGeometry(StrictModule, NonTrainableState):
    """The original accepted root mesh and geometry, not a replacement source.

    Root stencils and column controls are independently revalidated against the
    actual authoring source and request whenever this owner is reconstructed.
    Descendants retain one owner and their canonical current-to-root map rather
    than a chain of prior accepted states.
    """

    mesh: CellMesh
    geometry: CellGeometrySpec
    specification: VolumeMeshingSpec
    overlap_exact_geometry: ExactPlcCellGeometrySource
    overlap_stencils: OverlapSourceStencils
    column_certification: RefinedColumnCertification
    layer_root_partition: ReferenceCellPartition
    reference_mesh: CellMesh | None
    reference_geometry: CellGeometrySpec | None
    reference_cell_regions: tuple[int, ...] | None
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        overlap_exact_geometry: ExactPlcCellGeometrySource,
        overlap_stencils: OverlapSourceStencils,
        column_certification: RefinedColumnCertification,
        layer_root_partition: ReferenceCellPartition,
        /,
        *,
        reference_mesh: CellMesh | None = None,
        reference_geometry: CellGeometrySpec | None = None,
        reference_cell_regions: tuple[int, ...] | None = None,
        _chart_match_sink: list[bool] | None = None,
    ) -> None:
        from .providers._native_sources import NativeLayerCoreSource

        if (
            type(source) is not NativeLayerCoreSource
            or type(specification) is not VolumeMeshingSpec
        ):
            raise TypeError(
                "Layer root geometry requires the original source and volume request."
            )
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Layer root geometry requires its actual mesh and coordinate map."
            )
        if not isinstance(overlap_stencils, OverlapSourceStencils) or not isinstance(
            column_certification, RefinedColumnCertification
        ):
            raise TypeError(
                "Layer root geometry requires original stencil and column evidence."
            )
        source.validate_restored()
        require_layer_core_specification(source, specification)
        if not source.conforming_execution:
            raise ValueError(
                "Only an explicitly conforming source owns this root geometry."
            )
        mapped = source.mapped_domain is not None
        if mapped:
            reference_source = source.reference_source
            if (
                reference_source is None
                or not isinstance(reference_mesh, CellMesh)
                or not isinstance(reference_geometry, CellGeometrySpec)
                or reference_cell_regions is None
            ):
                raise ValueError(
                    "Mapped root requires its actual reference source, mesh, map and regions."
                )
            from ._layer_core import _mapped_reference_specification

            reference_specification = _mapped_reference_specification(specification)
            if any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in reference_cell_regions
            ):
                raise TypeError("Reference material regions must be integer indices.")
            reference_regions = tuple(int(value) for value in reference_cell_regions)
            if len(reference_regions) != sum(
                block.cell_count for block in reference_mesh.blocks
            ) or any(
                value < 0 or value >= len(source.region_ids)
                for value in reference_regions
            ):
                raise ValueError(
                    "Reference material regions do not cover the actual root mesh."
                )
            carrier_mesh = reference_mesh
            carrier_geometry = reference_geometry
            carrier_source = reference_source
            carrier_specification = reference_specification
        else:
            if any(
                value is not None
                for value in (reference_mesh, reference_geometry, reference_cell_regions)
            ):
                raise ValueError(
                    "Unmapped roots cannot retain mapped reference operands."
                )
            reference_regions = None
            carrier_mesh = mesh
            carrier_geometry = geometry
            carrier_source = source
            carrier_specification = specification
        ExactPlcCellGeometrySource.from_stored_fields(overlap_exact_geometry)
        carrier = overlap_exact_geometry
        required = {
            vertex
            for column in column_certification.columns
            for vertex in column.target_vertices
        }
        _require_overlap_provenance(
            carrier_source, carrier_mesh, carrier, overlap_stencils, required
        )
        require_refined_column_certification(
            carrier_source,
            carrier_mesh,
            carrier_geometry,
            carrier_specification,
            column_certification,
        )
        if not isinstance(layer_root_partition, ReferenceCellPartition):
            raise TypeError(
                "Layer root geometry requires the exact authored prism partition."
            )
        validate_reference_cell_partition(layer_root_partition)
        authored_mesh = carrier_source.layers.mesh
        if any(block.cell_kind != "prism" for block in authored_mesh.blocks):
            raise ValueError("Conforming root ancestry requires original prism layers.")
        expected_source_ids = {
            int(identifier)
            for block in authored_mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        }
        expected_target_ids = {
            int(identifier)
            for block in carrier_mesh.blocks
            if block.cell_kind == "prism"
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        }
        source_ids = {
            int(identifier)
            for group in layer_root_partition.maps
            for identifier in np.asarray(group.source_cell_ids, dtype=np.int64)
        }
        target_ids = {
            int(identifier)
            for group in layer_root_partition.maps
            for identifier in np.asarray(group.target_cell_ids, dtype=np.int64)
        }
        if source_ids != expected_source_ids or target_ids != expected_target_ids:
            raise ValueError(
                "Layer root partition does not cover the actual authored and root prisms."
            )
        authored_geometry = CellGeometrySpec.affine(authored_mesh)
        authored_resolved = authored_geometry.resolve(authored_mesh)
        root_resolved = carrier_geometry.resolve(carrier_mesh)
        authored_maps = _geometry_map_expressions(
            authored_mesh, authored_geometry, (authored_resolved[0], authored_resolved[1])
        )
        root_maps = _geometry_map_expressions(
            carrier_mesh, carrier_geometry, (root_resolved[0], root_resolved[1])
        )
        carrier_rows = {
            int(identifier): carrier.exact_coordinates[row]
            for row, identifier in enumerate(
                np.asarray(carrier.vertex_global_ids, dtype=np.int64)
            )
        }
        vertex_ids = np.asarray(carrier_mesh.vertex_global_ids, dtype=np.int64)
        for block in carrier_mesh.blocks:
            reference_vertices = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(block.cell_kind).vertices
            )
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            # This evaluates the actual exact source map at every reference
            # vertex, without constructing each full symbolic cell polynomial.
            values = carrier_geometry.evaluate_exact(
                carrier_mesh, identifiers, (reference_vertices,) * len(identifiers)
            )
            for cell, cell_values in zip(block.vertices, values, strict=True):
                for vertex, actual in zip(cell, cell_values, strict=True):
                    if tuple(actual) != carrier_rows[int(vertex_ids[int(vertex)])]:
                        raise ValueError(
                            "Curved root vertex differs from its exact source carrier."
                        )
        for group in layer_root_partition.maps:
            _require_exact_reference_group(
                authored_maps, root_maps, group, exact_reference_cell_vertices(group)
            )
        mapped_chart_bank_equal = False
        if mapped:
            if reference_regions is None:
                raise ValueError("Mapped root has no reference material bank.")
            from ._layer_core_mapped import _validate_refined_mapped_layers

            mapped_chart_bank_equal = _validate_refined_mapped_layers(
                source,
                carrier_mesh,
                carrier_geometry,
                reference_regions,
                mesh,
                geometry,
                specification,
                layer_root_partition,
            )
        self.mesh = mesh
        self.geometry = geometry
        self.specification = specification
        self.overlap_exact_geometry = carrier
        self.overlap_stencils = overlap_stencils
        self.column_certification = column_certification
        self.layer_root_partition = layer_root_partition
        self.reference_mesh = reference_mesh
        self.reference_geometry = reference_geometry
        self.reference_cell_regions = reference_regions
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.root_id = canonical_fingerprint(
            {
                "kind": "layer-core-root-geometry",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "mesh": mesh.mesh_id,
                "geometry": cell_geometry_id(geometry),
                "overlap_exact_geometry": carrier.source_map_id,
                "overlap_stencils": overlap_stencils.witness_id,
                "column_certification": column_certification.certification_id,
                **(
                    {
                        "reference_mesh": reference_mesh.mesh_id,
                        "reference_geometry": cell_geometry_id(reference_geometry),
                        "reference_cell_regions": reference_regions,
                    }
                    if mapped
                    and reference_mesh is not None
                    and reference_geometry is not None
                    else {}
                ),
                "layer_root_partition": _reference_map_identity(
                    layer_root_partition.maps
                ),
            }
        )
        if _chart_match_sink is not None:
            _chart_match_sink.append(mapped_chart_bank_equal)

    def require_current(
        self,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        /,
        *,
        _chart_match_sink: list[bool] | None = None,
    ) -> None:
        """Reprove the root geometry and its binding to current authoring inputs."""
        require_layer_core_specification(source, self.specification)
        chart_match: list[bool] = []
        rebuilt = type(self)(
            source,
            specification,
            self.mesh,
            self.geometry,
            self.overlap_exact_geometry,
            self.overlap_stencils,
            self.column_certification,
            self.layer_root_partition,
            reference_mesh=self.reference_mesh,
            reference_geometry=self.reference_geometry,
            reference_cell_regions=self.reference_cell_regions,
            _chart_match_sink=chart_match,
        )
        if (
            self.source_binding_id != source.binding_id
            or self.specification_id != specification.specification_id
            or self.root_id != rebuilt.root_id
        ):
            raise ValueError(
                "Layer root geometry is stale or differs from its source authority."
            )
        if _chart_match_sink is not None:
            _chart_match_sink.extend(chart_match)


class ComputedLayerCoreRoot:
    """Operation-local root proof for the exact operands used to construct it."""

    __slots__ = (
        "source", "specification", "mesh", "geometry", "root", "mapped_chart_bank_equal"
    )

    def __init__(
        self,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        overlap_exact_geometry: ExactPlcCellGeometrySource,
        overlap_stencils: OverlapSourceStencils,
        column_certification: RefinedColumnCertification,
        layer_root_partition: ReferenceCellPartition,
        /,
        *,
        reference_mesh: CellMesh | None = None,
        reference_geometry: CellGeometrySpec | None = None,
        reference_cell_regions: tuple[int, ...] | None = None,
    ) -> None:
        self.source = source
        self.specification = specification
        self.mesh = mesh
        chart_match: list[bool] = []
        self.geometry = geometry
        self.root = LayerCoreRootGeometry(
            source,
            specification,
            mesh,
            geometry,
            overlap_exact_geometry,
            overlap_stencils,
            column_certification,
            layer_root_partition,
            reference_mesh=reference_mesh,
            reference_geometry=reference_geometry,
            reference_cell_regions=reference_cell_regions,
            _chart_match_sink=chart_match,
        )
        self.mapped_chart_bank_equal = chart_match == [True]

    @classmethod
    def reprove(
        cls,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        root: LayerCoreRootGeometry,
        /,
    ) -> ComputedLayerCoreRoot:
        """Reprove an existing root once for these exact operation operands."""
        if type(root) is not LayerCoreRootGeometry:
            raise TypeError("Root reproof requires the actual layer/core root owner.")
        chart_match: list[bool] = []
        root.require_current(source, specification, _chart_match_sink=chart_match)
        result = cls.__new__(cls)
        result.source = source
        result.specification = specification
        result.mesh = root.mesh
        result.geometry = root.geometry
        result.root = root
        result.mapped_chart_bank_equal = chart_match == [True]
        return result

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("A computed layer/core root is immutable.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("A computed layer/core root is immutable.")

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> NoReturn:
        raise TypeError("Computed layer/core root cannot be serialized.")

    def require(
        self,
        source: NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        root: LayerCoreRootGeometry,
        /,
    ) -> None:
        if (
            self.source is not source
            or self.specification is not specification
            or self.mesh is not mesh
            or self.geometry is not geometry
            or self.root is not root
        ):
            raise ValueError("Computed layer/core root binds different exact operands.")


__all__ = ["LayerCoreRootGeometry"]
