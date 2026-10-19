"""Retained polynomial and rational coordinate maps for transfer tests."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import (
    reconstruct_parametric_surface_cell_geometry,
    transition_chart_deformed_cell_geometry,
)
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionPolicy,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._surface_chart_deformation import (
    prepare_surface_chart_deformation,
)
from phydrax.discretization._surface_chart_records import (
    PreparedSurfaceChartDeformation,
    SurfaceChartWitness,
)
from phydrax.discretization.fem import (
    discontinuous_element,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._local_provider import (
    require_tabulated_coordinate_element,
)
from phydrax.discretization.fem._reference_rules import degree_aware_reference_rule
from phydrax.discretization.finite_volume import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.geometry import (
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    SpherePatch,
)
from phydrax.meshing import MeshingLimits
from phydrax.meshing._quad_generation import extract_surface_quads
from tests._support.coordinate_references import native_meshing_request


def curved_volume(kind: str) -> tuple[CellMesh, CellGeometrySpec]:
    """Return one quadratic cell with shear y += 0.05*x*z in 3D.

    In 2D the shear is y += 0.05*x*(1-x). Both preserve measure while separating
    the retained coordinate map from the corner carrier; cell 17 and vertex IDs
    starting at 100 are explicit. Provenance: mapped DG and mixed-adaptation tests.
    """
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64).copy()
    vertices[:, 1] += (
        0.05
        * vertices[:, 0]
        * (1 - vertices[:, 0] if vertices.shape[1] == 2 else vertices[:, 2])
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "volume",
                kind,
                np.arange(len(vertices), dtype=np.int64)[None],
                global_ids=np.asarray((17,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(100, 100 + len(vertices), dtype=np.int64),
    )
    element = coordinate_lagrange_element(kind, 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[:, 1] += (
        0.05 * nodes[:, 0] * (1 - nodes[:, 0] if nodes.shape[1] == 2 else nodes[:, 2])
    )
    geometry = CellGeometrySpec(
        {"volume": element},
        {"volume": np.arange(len(nodes), dtype=np.int64)[None]},
        nodes,
    )
    return mesh, geometry


def dg_space(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    degree: int,
    *,
    component_shape: tuple[int, ...] = (2,),
) -> FiniteElementDiscretization:
    """Prepare a discontinuous field using the supplied coordinate map, not the carrier."""
    return FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "u",
            {
                block.name: discontinuous_element(block.cell_kind, degree)
                for block in mesh.blocks
            },
            component_shape=component_shape,
        ),
        coordinate_spec=geometry,
    ).prepare()


def dg_content(
    space: FiniteElementDiscretization,
    geometry: CellGeometrySpec,
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate both DG components by independent degree-14 physical quadrature."""
    total = np.zeros(2, dtype=np.float64)
    elements, routes, coordinates = geometry.resolve(space.mesh)
    for field, coordinate, route, dofs in zip(
        space.elements[0], elements, routes, space.dof_maps[0].cell_dofs, strict=True
    ):
        points, weights = degree_aware_reference_rule(field.cell_kind, 14)
        basis, _ = field.tabulate(points)
        _, gradients = require_tabulated_coordinate_element(coordinate).tabulate(points)
        for cell, field_route in zip(np.asarray(route), np.asarray(dofs), strict=True):
            jacobian = np.einsum(
                "mD,qmd->qDd", np.asarray(coordinates)[cell], np.asarray(gradients)
            )
            density = (
                np.linalg.det(jacobian)
                if jacobian.shape[-2] == jacobian.shape[-1]
                else np.linalg.norm(np.cross(jacobian[..., 0], jacobian[..., 1]), axis=-1)
            )
            values = np.asarray(basis) @ coefficients[field_route]
            total += np.einsum("q,q,qk->k", np.asarray(weights), density, values)
    return total


def rational_cylinder() -> tuple[
    CellMesh, CellGeometrySpec, SplineCellGeometryElement, NDArray[np.float64]
]:
    """Return a unit-length quarter-cylinder with its rational quadratic source bank.

    Controls and weights (1, sqrt(1/2), 1) represent the circular arc; the
    polygonal carrier has area sqrt(2), whereas the represented area is pi/2.
    Provenance: test_rational_embedded_measure._cylinder.
    """
    controls = np.asarray(
        [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)],
        dtype=np.float64,
    ).reshape(-1, 3)
    u = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    v = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64)
    weights = np.tile(np.asarray((1.0, math.sqrt(0.5), 1.0), dtype=np.float64), (2, 1))
    element = SplineCellGeometryElement(
        u, v, weights, 1, 2, (1, 2), "spline", "original-r1"
    )
    corners = np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        corners,
        (
            CellBlock(
                "root", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"root": element}, {"root": np.arange(6, dtype=np.int64)[None]}, controls
    )
    return mesh, geometry, element, controls


def six_quads(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: SplineCellGeometryElement,
    controls: NDArray[np.float64],
) -> tuple[CellMesh, CellGeometrySpec]:
    """Extract six quads from two triangle restrictions of the retained cylinder map.

    Parent geometry, topology, cell, and ordered vertex IDs reach the native
    extraction; controls stay in the source basis instead of sampled vertices.
    """
    triangles = CellMesh(
        mesh.coordinates,
        tuple(
            CellBlock(
                name,
                "triangle",
                np.asarray((cell,), dtype=np.int64),
                global_ids=np.asarray((row,), dtype=np.int64),
            )
            for row, (name, cell) in enumerate((("a", (0, 1, 2)), ("b", (0, 2, 3))))
        ),
        vertex_global_ids=mesh.vertex_global_ids,
    )
    elements = {
        "a": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
        "b": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
    }
    ancestry = CellGeometryRestrictionSource(
        cell_geometry_id(geometry),
        mesh.topology_id,
        {
            name: np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
            for name in elements
        },
        {
            name: np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
                np.asarray(mesh.blocks[0].vertices)
            ]
            for name in elements
        },
    )
    charts = CellGeometrySpec(
        elements,
        {name: np.arange(6, dtype=np.int64)[None] for name in elements},
        controls,
        restriction_source=ancestry,
    )
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        result = extract_surface_quads(
            triangles, limits, source_geometry=charts, scope=scope, budget=budget
        )
    if result.geometry is None:
        raise ValueError(
            "mapped_cells.six_quads.geometry must retain its coordinate source; got None"
        )
    return result.mesh, result.geometry


def _chart_layout(
    domain: MeshingDomain,
    uv: NDArray[np.float64],
    cells: NDArray[np.int32],
    radial_scale: float,
) -> CellGeometrySpec:
    element = coordinate_lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes)
    barycentric = np.column_stack((1.0 - nodes.sum(axis=1), nodes))
    charts = barycentric @ uv[cells]
    unique, inverse = np.unique(charts.reshape((-1, 2)), axis=0, return_inverse=True)
    return CellGeometrySpec(
        {"triangles": element},
        {"triangles": inverse.reshape((2, -1))},
        radial_scale * domain.evaluate(np.zeros(len(unique), dtype=np.int32), unique),
    )


def curved_surface_chart_fv(
    radius: float = 1.0,
) -> tuple[
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeDiscretization,
    PreparedSurfaceChartDeformation,
    CellGeometryTransition,
]:
    """Return a spherical chart deformation with distinct source and target cells.

    Source radius is 0.99 of the target, and the diagonal changes. Both maps
    retain the authored face occurrence and UV witnesses. Provenance:
    test_mapped_nested_fv._curved_surface_chart_fv_fixture.
    """
    uv = np.asarray(
        ((0.125, 0.125), (0.25, 0.125), (0.25, 0.25), (0.125, 0.25)), dtype=np.float64
    )
    surface = SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), radius)
    uses = tuple(
        PatchCurveUse(i, LineCurve(uv[i], uv[(i + 1) % 4] - uv[i]), 0.0, 1.0)
        for i in range(4)
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        tuple(MeshingDomainCurve(i, (i + 1) % 4) for i in range(4)),
        4,
        source_id="regular-patch",
        source_revision="r1",
        source_occurrences=(
            tuple(("assembly", f"corner-{i}") for i in range(4)),
            tuple(("assembly", f"curve-{i}") for i in range(4)),
            (("assembly", "face-7"),),
        ),
    )
    xyz = domain.evaluate(np.zeros(4, dtype=np.int32), uv)
    old = np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.int32)
    new = np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32)
    source_ids, target_ids = (
        np.asarray((107, 13), dtype=np.int64),
        np.asarray((59, 211), dtype=np.int64),
    )
    source_mesh = CellMesh.from_triangles(0.99 * xyz, old, cell_global_ids=source_ids)
    target_mesh = CellMesh.from_triangles(xyz, new, cell_global_ids=target_ids)
    source_geometry = _chart_layout(domain, uv, old, 0.99)
    entities, paths = (
        (domain.entity_id(2, 0),) * 2,
        (domain.source_occurrences[2][0],) * 2,
    )
    reconstruction = reconstruct_parametric_surface_cell_geometry(
        source_mesh,
        source_geometry,
        target_mesh,
        _chart_layout(domain, uv, new, 1.0),
        domain,
        domain_id=domain.domain_id,
        cell_ids=target_ids,
        cell_patches=np.zeros(2, dtype=np.int32),
        cell_charts=uv[new],
        cell_geometry_entity_ids=entities,
        cell_occurrence_paths=paths,
        maximum_fidelity=0.5,
    )
    source_witness = SurfaceChartWitness(
        geometry_id=cell_geometry_id(source_geometry),
        topology_id=source_mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=jnp.asarray(source_ids),
        patches=jnp.zeros(2, dtype=jnp.int32),
        charts=jnp.asarray(uv[old]),
        geometry_entity_ids=entities,
        occurrence_paths=paths,
    )
    target_witness = SurfaceChartWitness(
        geometry_id=cell_geometry_id(reconstruction.geometry),
        topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=jnp.asarray(target_ids),
        patches=jnp.zeros(2, dtype=jnp.int32),
        charts=jnp.asarray(uv[new]),
        geometry_entity_ids=entities,
        occurrence_paths=paths,
    )
    core = prepare_surface_chart_deformation(
        source_mesh,
        source_geometry,
        target_mesh,
        reconstruction,
        domain,
        source_witness=source_witness,
        target_witness=target_witness,
        maximum_fidelity=0.5,
        maximum_displacement=1.0,
    )
    carrier = transition_chart_deformed_cell_geometry(
        source_mesh,
        source_geometry,
        target_mesh,
        reconstruction,
        core,
        policy=CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation", reconstruction_tolerance=1.0
        ),
    )
    return (
        UnstructuredFiniteVolumePlan.from_cell_mesh(source_mesh).prepare(
            cell_geometry=source_geometry
        ),
        UnstructuredFiniteVolumePlan.from_cell_mesh(target_mesh).prepare(
            cell_geometry=reconstruction.geometry
        ),
        core,
        carrier,
    )


def fv_volume_map(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map volume reference coordinates by y += 0.05*y² + 0.05*x*z.

    The determinant is 1 + 0.1*y, so the unit hexahedron has volume 1.05.
    The retained curved interior differs from the corner carrier.
    """
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, 1] += 0.05 * points[:, 1] ** 2 + 0.05 * points[:, 0] * points[:, 2]
    return points


def curved_fv_volume(kind: str) -> tuple[CellMesh, CellGeometrySpec]:
    """Return a quadratic FV volume map with retained block and cell identities.

    ``connected`` joins a hexahedron and pyramid; other kinds contain cell 17.
    Coordinates, coefficient ordering, and IDs retain the mapped FV fixture.
    """
    if kind == "connected":
        corners = np.concatenate(
            (
                np.asarray(reference_cell_topology("hexahedron").vertices),
                [[0.5, 0.5, 2.0]],
            )
        )
        blocks = (
            CellBlock(
                "hex", "hexahedron", np.arange(8)[None], global_ids=np.asarray([10])
            ),
            CellBlock(
                "cap",
                "pyramid",
                np.asarray([[4, 5, 6, 7, 8]]),
                global_ids=np.asarray([11]),
            ),
        )
    else:
        corners = np.asarray(reference_cell_topology(kind).vertices)
        blocks = (
            CellBlock(
                "volume", kind, np.arange(len(corners))[None], global_ids=np.asarray([17])
            ),
        )
    mesh = CellMesh(fv_volume_map(corners), blocks)
    elements, routes, coefficients = {}, {}, []
    offset = 0
    for block in blocks:
        element = coordinate_lagrange_element(block.cell_kind, 2)
        nodes = np.asarray(element.reference_nodes).copy()
        if kind == "connected" and block.cell_kind == "pyramid":
            nodes[:, 2] += 1
        elements[block.name] = element
        routes[block.name] = np.arange(offset, offset + len(nodes))[None]
        coefficients.append(fv_volume_map(nodes))
        offset += len(nodes)
    return mesh, CellGeometrySpec(elements, routes, np.concatenate(coefficients))
