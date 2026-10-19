"""Native source-family generation for durable restart and source binding tests."""

from __future__ import annotations

import numpy as np

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.lifecycle._meshing_source_families import NativeGenerationSource
from phydrax.meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
from phydrax.meshing._mesh_part import MeshPart
from phydrax.meshing.providers._native_options import (
    NativeMeshingOptions,
    NativeMeshingRoute,
)


def native_family_generation(
    family: str,
) -> tuple[
    MeshPart,
    NativeGenerationSource,
    SurfaceMeshingSpec | VolumeMeshingSpec,
    NativeMeshingOptions,
]:
    import phydrax as phx
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
    from phydrax.geometry._source_fidelity import ParametricCurveBoundarySource
    from phydrax.geometry.brep._patches import LineCurve
    from phydrax.meshing._volume_generation import declared_plc_domain

    m = phx.meshing
    source: NativeGenerationSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    if family == "structured":
        boundaries = tuple(
            m.TransfiniteCurveControl(LineCurve(origin, direction), name, (0.0, 1.0), 2)
            for name, origin, direction in (
                ("left", (0.0, 0.0), (0.0, 1.0)),
                ("right", (1.0, 0.0), (0.0, 1.0)),
                ("bottom", (0.0, 0.0), (1.0, 0.0)),
                ("top", (0.0, 1.0), (1.0, 0.0)),
            )
        )
        block = m.TransfiniteBlock("square", boundaries)
        query = ParametricCurveBoundarySource(
            tuple(control.curve for control in boundaries),
            tuple(control.parameter_range for control in boundaries),
            source_id="restart-structured",
            source_revision="r1",
        )
        source = m.NativeStructuredSource(
            (block,),
            (),
            query.source_id,
            query.source_revision,
            fidelity_source=query,
            maximum_deviation=0.08,
        )
        scope = m.MeshingScope(
            source.source_id,
            source.source_revision,
            m.MeshingEntityKind.GEOMETRY,
            2,
            "square-domain",
            np.asarray((0,), dtype=np.int64),
        )
        specification = m.SurfaceMeshingSpec(
            m.CellMeshingTarget(2, 2, m.CellFamilyPolicy(required=("quadrilateral",))),
            scope,
            planar_embedding=phx.geometry.PlanarEmbedding(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            size_controls=(
                m.UniformSizeControl(scope, 0.5, strength=m.SizeControlStrength.SOFT),
            ),
        )
        options = m.NativeMeshingOptions("structured_transfinite")
    elif family == "surface":
        from phydrax.meshing.qualification import cylinder_sheet_surface

        source = m.NativeSurfaceSource(cylinder_sheet_surface().domain)
        scope = m.MeshingScope(
            source.source_id,
            source.source_revision,
            m.MeshingEntityKind.GEOMETRY,
            2,
            source.domain.entity_set_id(2),
            np.asarray((0,), dtype=np.int64),
        )
        specification = m.SurfaceMeshingSpec(
            m.CellMeshingTarget(2, 3, m.CellFamilyPolicy(required=("triangle",))),
            scope,
            size_controls=(
                m.UniformSizeControl(scope, 0.5, strength=m.SizeControlStrength.SOFT),
            ),
        )
        options = m.NativeMeshingOptions("parametric_surface")
    else:
        kind = "tetrahedron" if family == "plc" else "hexahedron"
        topology = reference_cell_topology(kind)
        corners = np.asarray(topology.vertices, dtype=np.float64)
        loops = topology.entities[2]
        complex_ = m.PiecewiseLinearComplex(
            corners,
            loops,
            np.arange(len(loops), dtype=np.int64),
            np.tile(np.asarray((-1, 0), dtype=np.int64), (len(loops), 1)),
            ("material",),
        )
        reference_source = m.NativePlcSource(complex_, "restart-reference", "r1")
        source = reference_source
        if family == "mapped":
            mesh = phx.discretization.CellMesh(
                corners,
                (phx.discretization.CellBlock("root", kind, np.arange(8)[None]),),
                numeric_version="r1",
            )
            element = coordinate_lagrange_element(kind, 2)
            nodes = np.asarray(element.reference_nodes)
            physical = nodes.copy()
            physical[:, 2] += 0.125 * nodes[:, 0] ** 2 * nodes[:, 1]
            geometry = CellGeometrySpec(
                {"root": element}, {"root": np.arange(len(nodes))[None]}, physical
            )
            domain = MappedReferenceDomain(
                declared_plc_domain(complex_, reference_source.source_id),
                mesh,
                geometry,
                np.asarray((0,), dtype=np.int64),
                source_id="restart-curved",
                source_revision="q2-r1",
            )
            source = m.NativeMappedHexSource(reference_source, domain)
            scope = m.MeshingScope(
                source.source_id,
                source.source_revision,
                m.MeshingEntityKind.GEOMETRY,
                2,
                domain.entity_set_id(2),
                mesh.entity_set(2).entity_ids,
            )
        else:
            if family == "polyhedral":
                source = m.NativePolyhedralSource(
                    complex_,
                    "restart-power",
                    "r1",
                    sites=np.asarray(
                        (
                            (0.25, 0.25, 0.25),
                            (0.75, 0.25, 0.25),
                            (0.25, 0.75, 0.25),
                            (0.25, 0.25, 0.75),
                        )
                    ),
                    weights=np.asarray((0.03125, -0.015625, 0.0625, 0.0)),
                )
            scope = m.MeshingScope(
                source.source_id,
                source.source_revision,
                m.MeshingEntityKind.GEOMETRY,
                2,
                "source-facets",
                np.arange(len(loops), dtype=np.int64),
            )
        cell_kind = {
            "plc": "tetrahedron",
            "polyhedral": "polyhedron",
            "mapped": "hexahedron",
        }[family]
        fill = {
            "plc": m.VolumeFillStrategy.SIMPLEX,
            "polyhedral": m.VolumeFillStrategy.POLYHEDRAL,
            "mapped": m.VolumeFillStrategy.MULTIZONE,
        }[family]
        specification = m.VolumeMeshingSpec(
            m.CellMeshingTarget(
                3,
                3,
                m.CellFamilyPolicy(required=(cell_kind,)),
                geometry_order=2 if family == "mapped" else 1,
            ),
            scope,
            fill,
            size_controls=(
                m.UniformSizeControl(
                    scope,
                    0.5 if family == "mapped" else 10.0,
                    strength=m.SizeControlStrength.SOFT,
                ),
            ),
        )
        routes: dict[str, NativeMeshingRoute] = {
            "plc": "plc_tetrahedral",
            "polyhedral": "plc_restricted_power",
            "mapped": "mapped_balanced_grid_hex",
        }
        options = m.NativeMeshingOptions(routes[family])
    result = (
        m.NativeMeshingProvider(options)
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    return m.MeshPart("generation", result), source, specification, options
