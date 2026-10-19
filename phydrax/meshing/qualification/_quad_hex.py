# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Build curved quad/hex sources and qualify their finite-element/volume consumers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from phydrax.geometry import MappedReferenceDomain
from phydrax.linalg import LinearSolvePolicy
from phydrax.meshing import CellMeshingResult


M = phx.meshing


def source_quarter_extrusion() -> M.NativeSurfaceSource:
    """Author a rational quarter-arc extrusion."""
    from phydrax.geometry import (
        BSplineSurfacePatch,
        LineCurve,
        MeshingDomain,
        MeshingDomainCurve,
        MeshingSurfacePatch,
        PatchCurveUse,
    )

    controls = np.asarray(
        [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)],
        dtype=np.float64,
    )
    weights = np.tile(np.asarray((1.0, np.sqrt(0.5), 1.0), dtype=np.float64), (2, 1))
    surface = BSplineSurfacePatch(
        controls,
        weights,
        np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
        1,
        2,
    )
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    loop = tuple(
        PatchCurveUse(
            index, LineCurve(uv[index], uv[(index + 1) % 4] - uv[index]), 0.0, 1.0
        )
        for index in range(4)
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (loop,)),),
        tuple(MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="original-quarter-extrusion",
        source_revision="original-r1",
    )
    return M.NativeSurfaceSource(domain)


def generate_source_quads() -> CellMeshingResult:
    """Retain the authored rational surface definition through public meshing."""
    source = source_quarter_extrusion()
    domain = source.domain
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray((0,), dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(
            2, 3, M.CellFamilyPolicy(required=("quadrilateral",)), geometry_order=3
        ),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 3.0, strength=M.SizeControlStrength.SOFT),
        ),
        protected_features=(
            M.ProtectedFeature(scope, M.FeatureKind.SURFACE, maximum_deviation=0.0),
        ),
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions("parametric_surface"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )


def curved_occupied_source(
    occupied: set[tuple[int, int, int]],
    /,
    *,
    materials: bool = False,
    curvature: float = 1.0 / 64.0,
    chart_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> M.NativeMappedHexSource:
    """Author independent curved root charts, including cavity and junction parts."""
    scale = np.asarray(chart_scale, dtype=np.float64)
    topology = reference_cell_topology("hexahedron")
    offsets = np.asarray(topology.vertices, dtype=np.int64)
    origins = sorted(occupied)
    keys = sorted(
        {tuple(np.asarray(origin) + offset) for origin in origins for offset in offsets}
    )
    index = {key: row for row, key in enumerate(keys)}
    points = np.asarray(keys, dtype=np.float64) * scale
    cells = np.asarray(
        [
            [index[tuple(np.asarray(origin) + offset)] for offset in offsets]
            for origin in origins
        ],
        dtype=np.int64,
    )
    region_ids = ("left", "right") if materials else ("material",)
    regions = np.asarray(
        [int(materials and origin[0] >= 2) for origin in origins], dtype=np.int64
    )
    owners: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    for row, cell in enumerate(cells):
        for loop in topology.entities[2]:
            vertices = tuple(cell[list(loop)].tolist())
            owners.setdefault(tuple(sorted(vertices)), []).append((row, vertices))
    polygons, incidence = [], []
    for key in sorted(owners):
        values = owners[key]
        row, loop = values[0]
        if len(values) == 1:
            polygons.append(loop)
            incidence.append((-1, regions[row]))
        elif regions[row] != regions[values[1][0]]:
            polygons.append(loop)
            incidence.append((regions[values[1][0]], regions[row]))
    complex_ = M.PiecewiseLinearComplex(
        points,
        polygons,
        np.arange(len(polygons), dtype=np.int64),
        np.asarray(incidence, dtype=np.int64),
        region_ids,
    )
    reference_source = M.NativePlcSource(complex_, "reference-domain", "reference-charts")
    reference = CellMesh(
        points,
        (CellBlock("independent_roots", "hexahedron", cells),),
        numeric_version="reference-charts",
    )
    element = phx.discretization.coordinate_lagrange_element("hexahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    positions = (
        (np.asarray(origins, dtype=np.float64)[:, None, :] + nodes[None, :, :]) * scale
    ).reshape((-1, 3))
    physical = positions.copy()
    physical[:, 2] += curvature * physical[:, 0] ** 2 * physical[:, 1]
    routes = np.arange(positions.shape[0], dtype=np.int64).reshape(
        (len(origins), nodes.shape[0])
    )
    geometry = CellGeometrySpec(
        {"independent_roots": element}, {"independent_roots": routes}, physical
    )
    domain = MappedReferenceDomain(
        M.declared_plc_domain(complex_, reference_source.source_id),
        reference,
        geometry,
        regions,
        source_id="curved-polynomial-domain",
        source_revision="declared-curved-state",
    )
    return M.NativeMappedHexSource(reference_source, domain)


def solve_linear_diffusion(
    result: CellMeshingResult,
    degree: int,
    /,
    *,
    harmonic_direction: tuple[float, ...] | None = None,
    solve_policy: LinearSolvePolicy | None = None,
    cell_quadrature_order: int | None = None,
) -> float:
    """Geometry-aware harmonic solve, including straight extrusion on a curved surface."""
    mesh = result.mesh
    dimension = mesh.topological_dimension
    ambient = mesh.ambient_dimension
    weights = (
        jnp.arange(1, ambient + 1, dtype=jnp.float64)
        if harmonic_direction is None
        else jnp.asarray(harmonic_direction, dtype=jnp.float64)
    )
    if weights.shape != (ambient,):
        raise ValueError(
            f"diffusion.harmonic_direction must have shape {(ambient,)}; got {weights.shape}"
        )

    def exact(points: Array, /) -> Array:
        return points @ weights

    def source(points: Array, arguments: object, /) -> Array:
        return jnp.zeros(points.shape[:-1], dtype=points.dtype)

    family = mesh.blocks[0].cell_kind
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element(family, degree)
    )
    space = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=result.geometry
    ).prepare()
    rules = {}
    if cell_quadrature_order is not None:
        if family != "quadrilateral":
            raise ValueError(
                f"diffusion.cell_kind must equal 'quadrilateral' for the explicit rule; got {family!r}"
            )
        rule = phx.integration.ReferenceQuadrilateralRule(
            phx.integration.GaussLegendreRule(cell_quadrature_order)
        )
        rules = {block.name: rule for block in mesh.blocks}
    form = phx.equations.FiniteElementForm(
        "dual-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0, rules=rules),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(source, coefficient_id="zero-source"),
                rules=rules,
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=exact,
    )
    operator, rhs = problem.linear_system()
    policy = (
        phx.linalg.LinearSolvePolicy(
            tolerance=phx.linalg.TolerancePolicy(relative=1.0e-12, absolute=1.0e-13)
        )
        if solve_policy is None
        else solve_policy
    )
    solved = phx.linalg.solve(operator, rhs, policy=policy)
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError(
            f"diffusion.solve.successful must be true for every entry; got {solved.successful}"
        )
    values = problem.expand(solved.value)
    rule = (
        phx.integration.ReferenceQuadrilateralRule(phx.integration.GaussLegendreRule(4))
        if dimension == 2
        else phx.integration.ReferenceHexahedronRule(phx.integration.GaussLegendreRule(4))
    )
    quadrature = phx.integration.reference_rule_data(rule)
    error_squared = jnp.asarray(0.0, dtype=jnp.float64)
    for block_index in range(len(mesh.blocks)):
        geometry = space.evaluate_block_geometry(
            "u",
            block_index,
            space.default_runtime.coordinates,
            quadrature.points,
            quadrature.weights,
        )
        dofs = space.dof_maps[0].cell_dofs[block_index]
        discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
        error_squared += jnp.sum(
            geometry.physical_weights * (discrete - exact(geometry.physical_points)) ** 2
        )
    error = float(jnp.sqrt(error_squared))
    if error > 1.0e-8:
        raise RuntimeError(
            f"diffusion.manufactured_L2_error must be <= 1.e-8; got {error}"
        )
    return error


def check_stationary_euler(result: CellMeshingResult, /) -> float:
    """Exercise the native FV owner with the accepted coordinate maps."""
    mesh = result.mesh
    system = phx.equations.EulerSystem(3)
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        mesh, component_names=system.component_names
    ).prepare(cell_geometry=result.geometry)
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.SlipWallBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
        system, geometry, method, boundaries
    )
    primitive = jnp.tile(
        jnp.asarray((1.0, 0.0, 0.0, 0.0, 1.0), dtype=jnp.float64),
        (geometry.cell_count, 1),
    )
    state = system.primitive_to_conserved(primitive)
    rate = dynamics(jnp.asarray(0.0, dtype=jnp.float64), state)
    maximum = float(jnp.max(jnp.abs(rate)))
    if maximum > 1.0e-10:
        raise RuntimeError(
            f"euler.stationary_content_rate must be <= 1.e-10; got {maximum}"
        )
    return maximum
