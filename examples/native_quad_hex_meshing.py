#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native quad/all-hex generation followed by geometry-aware Q1/Q2 diffusion.

The holed plate and two-material cavity use explicit robustness-oriented PL
dual routes. A separately declared Q2 polynomial solid exercises both balanced
and frame integer-grid routes with exact curved-source restriction, global
embedding, coverage and two-sided zero fidelity evidence.
An original nonconstant-weight rational spline surface is also published as
pure quads by the existing parametric-source route, with exact source-map
inheritance and independent whole-UV coverage rather than a triangle surrogate.

The curved positive request uses HARD target size 0.5 and maximum size 0.6,
with the frozen qualification policy's p50/p95 relative tolerance 0.5 (absolute
tolerance zero). This is a distinct, documented request, not a pass of the
original zero-tolerance quantile request. Passing SizeCompliancePolicy() to
``generate_curved_hexes`` retains that stricter request: it is currently UNMET
by the bounded placement portfolio; no impossibility theorem is claimed.

These cases do not claim arbitrary CAD-to-mapped-source equivalence or universal
varying-frame/cut-template closure.
"""

from __future__ import annotations

import json
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from examples.native_tetrahedral_meshing import complex_with_cavity
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from phydrax.geometry import MappedReferenceDomain
from phydrax.linalg import LinearSolvePolicy
from phydrax.meshing import CellMeshingResult, SizeCompliancePolicy


M = phx.meshing


def generate_quads() -> CellMeshingResult:
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(
            (
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (0.8, 0.8),
                (0.8, 1.2),
                (1.2, 1.2),
                (1.2, 0.8),
            ),
            dtype=np.float64,
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id="quad-plate",
    )
    source = M.NativePlanarSource(region, "plate-revision")
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "plate-region",
        np.asarray((0,), dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("quadrilateral",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(
            M.UniformSizeControl(scope, 0.4, strength=M.SizeControlStrength.SOFT),
        ),
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions("planar_dual_quad"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )


def source_quarter_extrusion() -> M.NativeSurfaceSource:
    """Author the unchanged original rational quarter-arc extrusion."""
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
    """Retain the original rational surface definition through public meshing."""
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


def generate_hexes() -> CellMeshingResult:
    source = M.NativePlcSource(
        complex_with_cavity(), "material-cavity", "cavity-revision"
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "material-cavity-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("hexahedron",))),
        scope,
        M.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            M.UniformSizeControl(scope, 0.4, strength=M.SizeControlStrength.SOFT),
        ),
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions("plc_dual_hex"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
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
    if not occupied:
        raise ValueError("The declared mapped solid needs occupied reference roots.")
    scale = np.asarray(chart_scale, dtype=np.float64)
    if scale.shape != (3,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("chart_scale must contain three finite positive values.")
    if not np.isfinite(curvature):
        raise ValueError("curvature must be finite.")
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


def generate_curved_hexes(
    route: Literal["mapped_balanced_grid_hex", "mapped_frame_grid_hex"],
    /,
    *,
    size_compliance: SizeCompliancePolicy | None = None,
) -> CellMeshingResult:
    """Declare the exact curved source and one explicit frozen sizing policy."""
    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    domain, reference_mesh = source.domain, source.domain.reference_mesh
    entities = reference_mesh.topology.entities(2)
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        entities.entity_ids,
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        scope,
        M.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            M.UniformSizeControl(
                scope, 0.5, maximum_size=0.6, strength=M.SizeControlStrength.HARD
            ),
        ),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5)
        if size_compliance is None
        else size_compliance,
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions(route))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )


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
            "The harmonic direction must match physical coordinate dimensions."
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
                "The surface example's explicit rule requires quadrilateral cells."
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
        raise RuntimeError("Native geometry-aware diffusion solve failed.")
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
            f"Manufactured physical solution L2 error {error} exceeds 1.e-8."
        )
    return error


def check_stationary_euler(result: CellMeshingResult, /) -> float:
    """Exercise the native FV owner with the actual accepted coordinate maps."""
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
            f"Mapped-cell stationary Euler content-rate {maximum} exceeds 1.e-10."
        )
    return maximum


def main() -> None:
    report: dict[str, dict[str, int | float | bool | str]] = {}
    cases = (
        ("quad_plate", generate_quads(), "quadrilateral", 1, None, None, None),
        (
            "original_rational_surface_quads",
            generate_source_quads(),
            "quadrilateral",
            2,
            (1.0, 0.0, 0.0),
            LinearSolvePolicy(
                phx.linalg.DenseLU(),
                tolerance=phx.linalg.TolerancePolicy(relative=1.0e-12, absolute=1.0e-13),
            ),
            8,
        ),
        ("hex_material_cavity", generate_hexes(), "hexahedron", 1, None, None, None),
        (
            "balanced_curved_hex",
            generate_curved_hexes("mapped_balanced_grid_hex"),
            "hexahedron",
            2,
            None,
            None,
            None,
        ),
        (
            "frame_curved_hex",
            generate_curved_hexes("mapped_frame_grid_hex"),
            "hexahedron",
            2,
            None,
            None,
            None,
        ),
    )
    for name, result, expected_family, degree, direction, policy, order in cases:
        mesh = result.mesh
        certification = result.certification
        if certification is None:
            raise RuntimeError(
                "Native pure-family generation omitted its required certification report."
            )
        if {block.cell_kind for block in mesh.blocks} != {expected_family}:
            raise RuntimeError("A required pure-family request was not met.")
        report[name] = {
            "family": mesh.blocks[0].cell_kind,
            "cells": sum(block.cell_count for block in mesh.blocks),
            "audit_passed": result.audit.passed,
            "certification_passed": certification.passed,
            "manufactured_physical_solution_error": solve_linear_diffusion(
                result,
                degree,
                harmonic_direction=direction,
                solve_policy=policy,
                cell_quadrature_order=order,
            ),
        }
        if mesh.topological_dimension == 3:
            report[name]["stationary_fv_euler_content_rate"] = check_stationary_euler(
                result
            )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
