from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Callable, cast

import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    lagrange_element,
    PolyhedralConnectivity,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    swept_coordinate_element,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.geometry import (
    BSplineSurfacePatch,
    MappedReferenceDomain,
    PiecewiseLinearDomain,
)
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._piecewise_linear_boundary import (
    PiecewiseLinearBoundarySource,
    PiecewiseLinearDomainBoundaryOwner,
)
from phydrax.geometry.brep import CircleCurve, LineCurve, PlanePatch
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshQualityTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    BlockInterfaceControl,
    FeatureKind,
    HoleSeed,
    PatchControl,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
    TransfiniteCurveControl,
    TransfiniteSurfaceControl,
)
from phydrax.meshing._organization import MeshZoneRole, RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._structured import TransfiniteBlock
from phydrax.meshing._sweep import SweepControl, SweepMapKind
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import (
    NativeMappedHexSource,
    NativePlanarSource,
    NativePlcSource,
    NativeStructuredSource,
    NativeSweepSource,
)

from .._runtime import accepted_result_record, runtime_identity


_CONTRACT = SpatialCoordinateContract.si()
_PROFILE_PREFIX = "native-family-lifecycle:"
_BASE_RESOLUTION = 4
_PROFILE_TOLERANCE = 1.0e-10
_QUALITY_LIMITS = {
    "minimum_scaled_jacobian": 1.0e-8,
    "minimum_mean_ratio": 0.01,
    "maximum_aspect_ratio": 100.0,
}

# Distinct pure-family consumers cover native structured, sweep, multiblock,
# dual conversion and curved/material routes without multiplying every case by
# identical FE work. Mixed-transition consumption is qualified separately.
_FE_PROFILES = frozenset(
    {
        "dual-quad",
        "dual-hex",
        "transfinite-quad",
        "sweep-hex",
        "multiblock-reversed-hex",
        "curved-material-hex",
        "curved-frame-sweep-hex",
        "nonsweepable-dual-hex",
    }
)

# Administrative materialization bound for the exact sparse Cholesky
# preconditioner (not a scientific tolerance). The library default of 2,000,000
# refused the restored-donor curved-material-hex Q2 system, whose measured
# nested-dissection factor needs 6,484,805 nonzeros (36,743,690 bytes).
_FAMILY_FACTOR_NNZ_BUDGET = 6_500_000
_FAMILY_FACTOR_NNZ_BASIS = (
    "measured curved-material-hex factor_nnz=6,484,805 (bg4756), rounded up"
)


def _solve_family_affine_state(
    result: Any, /
) -> tuple[dict[str, object], np.ndarray, Any]:
    kinds = sorted({block.cell_kind for block in result.mesh.blocks if block.cell_count})
    if not kinds:
        raise ValueError("The family FE witness requires at least one actual cell.")
    degree = max(
        int(getattr(element, "degree", 1)) for element in result.geometry.elements
    )
    # Declared mixed families (hex-dominant transitions) use one conforming
    # Lagrange element per actual block; the affine field lies in every space.
    elements = (
        lagrange_element(kinds[0], degree)
        if len(kinds) == 1
        else {
            block.name: lagrange_element(block.cell_kind, degree)
            for block in result.mesh.blocks
        }
    )
    field = phx.discretization.FiniteElementFieldSpec("u", elements)
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        "native-family-affine-harmonic",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"positive_definite": "asserted"},
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=lambda points: points[..., 0]
    )
    operator, rhs = problem.linear_system(storage="sparse")
    prepared = phx.linalg.prepare(
        operator,
        phx.linalg.LinearSolvePolicy(
            phx.linalg.PCG(),
            tolerance=phx.linalg.TolerancePolicy(relative=1.0e-14, absolute=0.0),
            preconditioning=phx.linalg.PreconditioningPolicy(
                phx.linalg.SparseFactorizationPreconditionerBuilder(
                    phx.linalg.SparseFactorizationPolicy(
                        "cholesky",
                        ordering="nested-dissection",
                        max_factor_nnz=_FAMILY_FACTOR_NNZ_BUDGET,
                    )
                )
            ),
        ),
    )
    solved = phx.linalg.solve(prepared, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    expected = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)[:, 0]
    error = float(np.max(np.abs(values - expected), initial=0.0))
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    normalized = residual / max(1.0, float(jnp.linalg.norm(rhs)))
    if (
        not bool(jnp.all(solved.successful))
        or values.shape != expected.shape
        or error > 2.0e-9
        or normalized > 2.0e-9
    ):
        raise RuntimeError(
            f"Family FE affine-field solve failed: {error=}, {normalized=}."
        )
    record = {
        "form": form.form_id,
        "cell_kind": kinds[0] if len(kinds) == 1 else "mixed",
        "cell_kinds": kinds,
        "field_degree": degree,
        "cell_count": int(sum(block.cell_count for block in result.mesh.blocks)),
        "factor_nnz_budget": _FAMILY_FACTOR_NNZ_BUDGET,
        "factor_nnz_budget_basis": _FAMILY_FACTOR_NNZ_BASIS,
        "dofs": int(values.size),
        "exact_field": "u=x",
        "maximum_error": error,
        "normalized_residual": normalized,
        "tolerance": 2.0e-9,
    }
    return record, values, field


def _solve_family_affine_field(result: Any, /) -> dict[str, object]:
    return _solve_family_affine_state(result)[0]


def _solve_quarter_quadratic(result: Any, /) -> tuple[dict[str, object], np.ndarray, Any]:
    """Reproduce the donor's axial x² diffusion and physical L2 oracle."""
    degree = max(
        int(getattr(element, "degree", 1)) for element in result.geometry.elements
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u", lagrange_element("quadrilateral", degree)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()

    def exact(points: Any) -> Any:
        return points[..., 0] ** 2

    def source(points: Any, parameters: Any) -> Any:
        del parameters
        return jnp.full(points.shape[:-1], -2.0, dtype=points.dtype)

    form = phx.equations.FiniteElementForm(
        "exact-family-manufactured-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    source, coefficient_id="original-extrusion-axial-quadratic-source"
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=exact,
    )
    operator, rhs = problem.linear_system(storage="sparse")
    solved = phx.linalg.solve(
        operator,
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            tolerance=phx.linalg.TolerancePolicy(relative=1.0e-12, absolute=1.0e-13)
        ),
    )
    values = problem.expand(solved.value)
    quadrature = phx.integration.reference_rule_data(
        phx.integration.ReferenceQuadrilateralRule(phx.integration.GaussLegendreRule(6))
    )
    squared = 0.0
    inventory = 0.0
    for index in range(len(result.mesh.blocks)):
        geometry = space.evaluate_block_geometry(
            "u",
            index,
            space.default_runtime.coordinates,
            quadrature.points,
            quadrature.weights,
        )
        dofs = space.dof_maps[0].cell_dofs[index]
        discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
        weights = np.asarray(geometry.physical_weights)
        squared += float(
            np.sum(weights * np.asarray(discrete - exact(geometry.physical_points)) ** 2)
        )
        inventory += float(np.sum(weights * np.asarray(discrete)))
    error = float(np.sqrt(squared))
    if not bool(jnp.all(solved.successful)) or not np.isfinite(error) or error > 0.05:
        raise RuntimeError(f"Original quarter-extrusion quadratic FE L2 error: {error}.")
    record = {
        "form": form.form_id,
        "field_degree": degree,
        "exact_field": "u=x^2",
        "source": "-2",
        "physical_l2_error": error,
        "physical_inventory": inventory,
        "quadrature_order": 6,
        "l2_tolerance": 0.05,
    }
    return record, np.asarray(values, dtype=np.float64), field


@dataclass(frozen=True)
class _FamilyInput:
    source: Any
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    options: NativeMeshingOptions
    expected_route: str
    physical_input: dict[str, object]
    expected_measure: float
    expected_regions: dict[str, float] = field(default_factory=dict)
    expected_domain: Any | None = None
    fidelity_tolerance: float | None = None
    require_region_coverage: bool = False
    require_transition: bool = False
    quality_limits: dict[str, float] = field(default_factory=dict)
    # Donor obligation inputs, derived from the authored source only.
    nonsweepability_required: bool = False
    authored_topology: dict[str, Any] | None = None
    source_construction_proof: dict[str, Any] | None = None
    authored_boundary: dict[str, Any] | None = None


def _profile_ids() -> tuple[str, ...]:
    migration_path = Path(__file__).resolve().parents[1] / "native_case_migration.json"
    with migration_path.open("r", encoding="utf-8") as stream:
        migration = json.load(stream)
    workflows = migration["workflow_contracts"]
    workflow = next(
        item for item in workflows if item["workflow"] == "general-quads-hexes"
    )
    return tuple(workflow["profiles"])


def _size_control(scope: MeshingScope, resolution: int) -> UniformSizeControl:
    return UniformSizeControl(
        scope,
        1.0 / resolution,
        maximum_size=2.0 / resolution,
        strength=SizeControlStrength.HARD,
    )


def _size_compliance() -> SizeCompliancePolicy:
    return SizeCompliancePolicy(relative_tolerance=0.95, target_statistics=("p50",))


def _surface_specification(
    source: Any,
    resolution: int,
    family: str,
    *,
    ambient_dimension: int = 2,
    geometry_order: int = 1,
    planar_embedding: Any | None = None,
    protected_features: tuple[ProtectedFeature, ...] = (),
) -> SurfaceMeshingSpec:
    scope = MeshingScope.whole(source, 2)
    return SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            ambient_dimension,
            CellFamilyPolicy(required=(family,)),
            geometry_order=geometry_order,
        ),
        scope,
        planar_embedding=planar_embedding,
        size_controls=(_size_control(scope, resolution),),
        protected_features=protected_features,
        size_compliance=_size_compliance(),
        quality_target=MeshQualityTarget(minimum_angle=0.01, hard=True),
    )


def _volume_specification(
    source: Any,
    resolution: int,
    family: str,
    *,
    geometry_order: int = 1,
    fill_strategy: VolumeFillStrategy = VolumeFillStrategy.MULTIZONE,
    region_seeds: tuple[RegionSeed, ...] = (),
    region_controls: tuple[RegionControl, ...] = (),
    patch_controls: tuple[PatchControl, ...] = (),
    hole_seeds: tuple[HoleSeed, ...] = (),
    mixed: bool = False,
) -> VolumeMeshingSpec:
    scope = MeshingScope.whole(source, 2)
    policy = (
        CellFamilyPolicy(
            required=("hexahedron",),
            allowed_transitions=("pyramid", "tetrahedron"),
            allow_mixed=True,
        )
        if mixed
        else CellFamilyPolicy(required=(family,))
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, policy, geometry_order=geometry_order),
        scope,
        fill_strategy,
        size_controls=(_size_control(scope, resolution),),
        size_compliance=_size_compliance(),
        region_seeds=region_seeds,
        region_controls=region_controls,
        patch_controls=patch_controls,
        hole_seeds=hole_seeds,
    )


def _curve_control(
    curve: Any,
    name: str,
    interval: tuple[float, float],
    intervals: int,
) -> TransfiniteCurveControl:
    return TransfiniteCurveControl(curve, name, interval, intervals)


def _rectangle(name: str, left: float, right: float, resolution: int) -> TransfiniteBlock:
    return TransfiniteBlock(
        name,
        tuple(
            _curve_control(
                LineCurve(origin, direction), f"{name}:{edge}", (0.0, 1.0), intervals
            )
            for edge, origin, direction, intervals in (
                ("left", (left, 0.0), (0.0, 1.0), resolution),
                ("right", (right, 0.0), (0.0, 1.0), resolution),
                ("bottom", (left, 0.0), (right - left, 0.0), resolution),
                ("top", (left, 1.0), (right - left, 0.0), resolution),
            )
        ),
    )


def _pwl_query(domain: PiecewiseLinearDomain) -> PiecewiseLinearBoundarySource:
    return PiecewiseLinearBoundarySource(PiecewiseLinearDomainBoundaryOwner(domain), 2)


def _cube_domain(source_id: str = "box") -> PiecewiseLinearDomain:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    facets = np.asarray(
        tuple(triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))),
        dtype=np.int64,
    )
    return PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1)),
        ("solid",),
        source_id=source_id,
    )


def _cube_profile(resolution: int, family: str) -> CellMesh:
    axis = np.linspace(0.0, 1.0, resolution + 1)
    points = np.asarray(tuple((u, v, 0.0) for u in axis for v in axis), dtype=np.float64)
    cells: list[tuple[int, ...]] = []
    for i in range(resolution):
        for j in range(resolution):
            a = i * (resolution + 1) + j
            b = (i + 1) * (resolution + 1) + j
            if family == "quadrilateral":
                cells.append((a, b, b + 1, a + 1))
            else:
                cells.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    kind = "quadrilateral" if family == "quadrilateral" else "triangle"
    block = CellBlock(
        "profile",
        kind,
        np.asarray(cells, dtype=np.int32),
        global_ids=np.arange(len(cells), dtype=np.int64) + 500,
    )
    return CellMesh(
        points,
        (block,),
        vertex_global_ids=np.arange(points.shape[0], dtype=np.int64) + 40,
    )


def _cube_domain_vertex_map(
    domain: PiecewiseLinearDomain, profile: CellMesh, resolution: int
) -> np.ndarray:
    profile_points = np.asarray(profile.coordinates, dtype=np.float64).reshape((-1, 3))
    profile_ids = np.asarray(profile.vertex_global_ids, dtype=np.int64).reshape((-1,))
    row_by_xy: dict[tuple[float, float], int] = {}
    for row in range(profile_points.shape[0]):
        row_by_xy[(float(profile_points[row, 0]), float(profile_points[row, 1]))] = int(
            profile_ids[row]
        )
    domain_points = np.asarray(domain.vertices, dtype=np.float64).reshape((-1, 3))
    mapping: list[tuple[int, int]] = []
    for row in range(domain_points.shape[0]):
        x = float(domain_points[row, 0])
        y = float(domain_points[row, 1])
        z = float(domain_points[row, 2])
        mapping.append((row_by_xy[(x, y)], 0 if z == 0.0 else resolution))
    return np.asarray(mapping, dtype=np.int64)


def _plc(
    vertices: np.ndarray,
    polygons: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    polygon_facets: np.ndarray,
    facet_regions: np.ndarray,
    region_ids: tuple[str, ...],
    *,
    vertex_prefix: str,
    facet_ids: tuple[str, ...],
) -> PiecewiseLinearComplex:
    return PiecewiseLinearComplex(
        vertices,
        polygons,
        polygon_facets,
        facet_regions,
        region_ids,
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(
            f"{vertex_prefix}-vertex-{row}" for row in range(vertices.shape[0])
        ),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=facet_ids,
        facet_occurrence_paths=((),) * len(facet_ids),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),) * len(region_ids),
    )


def _cavity_plc() -> PiecewiseLinearComplex:
    box_loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )

    def box(
        lower: tuple[float, float, float], upper: tuple[float, float, float]
    ) -> np.ndarray:
        return np.asarray(
            [
                [(upper if index >> axis & 1 else lower)[axis] for axis in range(3)]
                for index in range(8)
            ],
            dtype=np.float64,
        )

    left = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    right = box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    cavity = box((1.25, 0.25, 0.25), (1.75, 0.75, 0.75))
    vertices = np.concatenate((left, right[[1, 3, 5, 7]], cavity))
    right_index = {0: 1, 2: 3, 4: 5, 6: 7, 1: 8, 3: 9, 5: 10, 7: 11}
    polygons: list[tuple[int, ...]] = []
    polygon_facets: list[int] = []
    for loop in box_loops:
        if loop != (1, 3, 7, 5):
            polygons.append(loop)
            polygon_facets.append(0)
        if loop != (0, 4, 6, 2):
            polygons.append(tuple(right_index[index] for index in loop))
            polygon_facets.append(1)
    polygons.append((1, 3, 7, 5))
    polygon_facets.append(2)
    polygons.extend(tuple(index + 12 for index in loop) for loop in box_loops)
    polygon_facets.extend([3] * len(box_loops))
    return _plc(
        vertices,
        polygons,
        np.asarray(polygon_facets, dtype=np.int64),
        np.asarray(((-1, 0), (-1, 1), (1, 0), (1, -1)), dtype=np.int64),
        ("left", "right"),
        vertex_prefix="material-cavity",
        facet_ids=("left-outer", "right-outer", "material-interface", "right-cavity"),
    )


def _nonsweepable_plc() -> tuple[PiecewiseLinearComplex, float]:
    tetrahedron = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64
    )
    vertices = np.concatenate(
        (
            tetrahedron,
            (2 * tetrahedron + (2, 2, 2)) / 16,
            (2 * tetrahedron + (7, 2, 2)) / 16,
        )
    )
    outward = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    polygons: list[tuple[int, ...]] = list(outward)
    for offset in (4, 8):
        polygons.extend(
            (offset + face[2], offset + face[1], offset + face[0]) for face in outward
        )
    complex_ = _plc(
        vertices,
        polygons,
        np.arange(12, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (12, 1)),
        ("solid",),
        vertex_prefix="nonsweepable-dual-hex",
        facet_ids=tuple(f"nonsweepable-shell-facet-{row}" for row in range(12)),
    )
    outer = Fraction(1, 6)
    cavity = Fraction(1, 6) * Fraction(1, 8) ** 3
    return complex_, float(outer - 2 * cavity)


def _mapped_reference_domain(
    complex_: PiecewiseLinearComplex,
    points: np.ndarray,
    cells: np.ndarray,
    cell_regions: np.ndarray,
    root_vertex_ids: np.ndarray,
    root_cell_ids: np.ndarray,
    root_geometry: CellGeometrySpec,
    source_id: str,
    source_revision: str,
) -> MappedReferenceDomain:
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    owners = np.asarray(complex_.polygon_facets, dtype=np.int64)
    facet_sides = np.asarray(complex_.facet_regions, dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    region_sides: list[tuple[int, int]] = []
    for polygon in range(owners.size):
        loop = polygons[offsets[polygon] : offsets[polygon + 1]]
        facet_row = int(owners[polygon])
        side: tuple[int, int] = (
            int(facet_sides[facet_row, 1]),
            int(facet_sides[facet_row, 0]),
        )
        for index in range(1, loop.size - 1):
            triangles.append((int(loop[0]), int(loop[index]), int(loop[index + 1])))
            region_sides.append(side)
    reference_domain = PiecewiseLinearDomain(
        points,
        np.asarray(triangles, dtype=np.int64),
        np.asarray(region_sides, dtype=np.int64),
        complex_.region_ids,
        source_id=source_id,
    )
    root_mesh = CellMesh(
        points,
        (CellBlock("root", "hexahedron", cells, global_ids=root_cell_ids),),
        vertex_global_ids=root_vertex_ids,
    )
    return MappedReferenceDomain(
        reference_domain,
        root_mesh,
        root_geometry,
        cell_regions,
        source_id=source_id,
        source_revision=source_revision,
        coordinate_contract=_CONTRACT,
        reference_domain_vertex_root_ids=root_vertex_ids,
    )


def _curved_occupied_source(
    kind: str,
) -> tuple[NativeMappedHexSource, dict[str, float], dict[str, object]]:
    if kind == "curved-nonsweepable-hex":
        occupied = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
        materials = False
    else:
        occupied = {
            origin for origin in product(range(3), repeat=3) if origin != (1, 1, 1)
        }
        materials = kind == "curved-material-hex"
    topology = reference_cell_topology("hexahedron")
    offsets = np.asarray(topology.vertices, dtype=np.int64)
    origins = sorted(occupied)
    keys = sorted(
        {
            tuple((np.asarray(origin) + offset).tolist())
            for origin in origins
            for offset in offsets
        }
    )
    index = {key: row for row, key in enumerate(keys)}
    points = np.asarray(keys, dtype=np.float64)
    cells = np.asarray(
        [
            [index[tuple((np.asarray(origin) + offset).tolist())] for offset in offsets]
            for origin in origins
        ],
        dtype=np.int32,
    )
    region_ids = (
        ("left", "right")
        if materials
        else (("material",) if kind != "curved-nonsweepable-hex" else ("solid",))
    )
    cell_regions = np.asarray(
        [int(materials and origin[0] >= 2) for origin in origins], dtype=np.int64
    )
    owners: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    for row, cell in enumerate(cells):
        for face in topology.entities[2]:
            face_vertices = tuple(int(value) for value in cell[list(face)])
            owners.setdefault(tuple(sorted(face_vertices)), []).append(
                (row, face_vertices)
            )
    polygons: list[tuple[int, ...]] = []
    polygon_facets: list[int] = []
    facet_sides: list[tuple[int, int]] = []
    for values in owners.values():
        row, oriented = values[0]
        region = int(cell_regions[row])
        if len(values) == 1:
            sides = (-1, region)
        else:
            other_row = int(values[1][0])
            other_region = int(cell_regions[other_row])
            if other_region == region:
                continue
            sides = (other_region, region)
        facet_index = len(facet_sides)
        polygons.extend(
            (
                (oriented[0], oriented[1], oriented[2]),
                (oriented[0], oriented[2], oriented[3]),
            )
        )
        polygon_facets.extend((facet_index, facet_index))
        facet_sides.append(sides)
    complex_ = _plc(
        points,
        polygons,
        np.asarray(polygon_facets, dtype=np.int64),
        np.asarray(facet_sides, dtype=np.int64),
        region_ids,
        vertex_prefix=f"{kind}-root",
        facet_ids=tuple(f"{kind}-facet-{row}" for row in range(len(facet_sides))),
    )
    source_id, source_revision = "curved-polynomial-domain", "declared-curved-state"
    root_vertex_ids = np.arange(points.shape[0], dtype=np.int64) + 1000
    root_cell_ids = np.arange(cells.shape[0], dtype=np.int64) + 900
    element = lagrange_element("hexahedron", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    positions = (
        np.asarray(origins, dtype=np.float64)[:, None, :] + reference_nodes[None, :, :]
    ).reshape((-1, 3))
    physical = positions.copy()
    physical[:, 2] += (1.0 / 64.0) * physical[:, 0] ** 2 * physical[:, 1]
    routes = np.arange(physical.shape[0], dtype=np.int32).reshape(
        (cells.shape[0], element.local_dof_count)
    )
    root_geometry = CellGeometrySpec({"root": element}, {"root": routes}, physical)
    domain = _mapped_reference_domain(
        complex_,
        points,
        cells,
        cell_regions,
        root_vertex_ids,
        root_cell_ids,
        root_geometry,
        source_id,
        source_revision,
    )
    reference = NativePlcSource(complex_, source_id, source_revision)
    source = NativeMappedHexSource(
        reference,
        domain,
        plc_vertex_root_global_ids=root_vertex_ids,
    )
    region_measures = {
        name: float(sum(int(cell_regions[row] == region) for row in range(len(origins))))
        for region, name in enumerate(region_ids)
    }
    details = {
        "root_cell_origins": [list(origin) for origin in origins],
        "root_cell_count": len(origins),
        "root_cell_family": "hexahedron",
        "root_geometry_order": 2,
        "physical_map": "(x, y, z) -> (x, y, z + x^2*y/64)",
        "jacobian_determinant": 1.0,
        "material_partition": "x-index >= 2 is right; all other occupied roots are left"
        if materials
        else None,
        "curvature": 1.0 / 64.0,
    }
    return source, region_measures, details


def _quarter_extrusion() -> MeshingDomain:
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
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (loop,)),),
        tuple(MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="original-quarter-extrusion",
        source_revision="original-r1",
    )


def _planar_dual_quad(resolution: int) -> _FamilyInput:
    vertices = np.asarray(
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
    )
    region = PlanarMeshRegion(
        vertices, ((0, 1, 2, 3), (4, 5, 6, 7)), feature_id="quad-plate"
    )
    source = NativePlanarSource(
        region,
        "plate-revision",
        vertex_entity_ids=np.asarray((100, 101, 102, 103, 104, 105, 106, 107)),
        edge_entity_ids=np.asarray((200, 201, 202, 203, 204, 205, 206, 207)),
    )
    specification = _surface_specification(source, resolution, "quadrilateral")
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("planar_dual_quad"),
        "planar_dual_quad",
        {
            "fixture": "NativePlanarSource:holed-plate",
            "outer_loop_m": vertices[:4].tolist(),
            "hole_loop_m": vertices[4:].tolist(),
            "domain_area_m2": 3.84,
            "resolution": resolution,
            "size_target_m": 1.0 / resolution,
            "maximum_size_m": 2.0 / resolution,
        },
        3.84,
        fidelity_tolerance=0.0,
        quality_limits={"minimum_angle": 0.01},
        authored_boundary=_authored_planar_boundary(
            vertices, ((0, 1, 2, 3), (4, 5, 6, 7))
        ),
    )


def _structured_square(resolution: int, multiblock: bool) -> _FamilyInput:
    source_id = "unit-square"
    if multiblock:
        blocks = (
            _rectangle("left", 0.0, 0.5, resolution),
            _rectangle("right", 0.5, 1.0, resolution),
        )
        interfaces = (BlockInterfaceControl("left", 1, "right", 0, (0,), (False,)),)
        fixture = "NativeStructuredSource:two-glued-transfinite-blocks"
    else:
        blocks = (_rectangle("square", 0.0, 1.0, resolution),)
        interfaces = ()
        fixture = "NativeStructuredSource:single-transfinite-block"
    domain = PiecewiseLinearDomain(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("square",),
        source_id=source_id,
    )
    revision = domain.source_revision
    interface_faces = {
        face
        for interface in interfaces
        for face in (
            (interface.first_block, interface.first_face),
            (interface.second_block, interface.second_face),
        )
    }
    boundaries = tuple(
        cast(TransfiniteCurveControl, control)
        for block in blocks
        for face, control in enumerate(block.boundaries)
        if (block.name, face) not in interface_faces
    )
    boundary = ParametricCurveBoundarySource(
        tuple(control.curve for control in boundaries),
        tuple(control.parameter_range for control in boundaries),
        source_id=source_id,
        source_revision=revision,
        curve_entity_ids=tuple(control.source_id for control in boundaries),
        occurrence_paths=((),) * len(boundaries),
        chart_ids=tuple(control.source_id for control in boundaries),
    )
    source = NativeStructuredSource(
        blocks,
        interfaces,
        source_id,
        revision,
        fidelity_source=boundary,
        maximum_deviation=0.08,
        domain=domain,
    )
    embedding = phx.geometry.PlanarEmbedding(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    specification = _surface_specification(
        source,
        resolution,
        "quadrilateral",
        planar_embedding=embedding,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("structured_transfinite"),
        "structured_transfinite",
        {
            "fixture": fixture,
            "domain_m": ((0.0, 0.0), (1.0, 1.0)),
            "blocks": [block.name for block in blocks],
            "block_interfaces": [interface.control_id for interface in interfaces],
            "intervals_per_block_axis": resolution,
            "physical_domain_area_m2": 1.0,
            "maximum_deviation_m": 0.08,
            "resolution": resolution,
        },
        1.0,
        fidelity_tolerance=0.08,
        quality_limits={"minimum_angle": 0.01},
    )


def _annular_blocks(resolution: int) -> _FamilyInput:
    inner = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    outer = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0)
    angular = 4 * resolution
    angles = np.concatenate(
        tuple(
            np.linspace(index * np.pi / 2.0, (index + 1) * np.pi / 2.0, angular + 1)[:-1]
            for index in range(4)
        )
    )
    points = np.concatenate(
        (
            np.asarray(inner.evaluate(jnp.asarray(angles)), dtype=np.float64),
            np.asarray(outer.evaluate(jnp.asarray(angles)), dtype=np.float64),
        )
    )
    count = angles.size
    facets = tuple(((index + 1) % count, index) for index in range(count)) + tuple(
        (count + index, count + (index + 1) % count) for index in range(count)
    )
    domain = PiecewiseLinearDomain(
        points,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(facets), 1)),
        ("annulus",),
        source_id="annular-block-source",
    )
    revision = domain.source_revision
    query = ParametricCurveBoundarySource(
        (inner, outer),
        ((0.0, 2.0 * np.pi),) * 2,
        source_id=domain.source_id,
        source_revision=revision,
        curve_entity_ids=("inner", "outer"),
        occurrence_paths=((), ()),
        chart_ids=("annulus-inner-chart", "annulus-outer-chart"),
    )
    blocks = []
    for index in range(4):
        first, last = index * np.pi / 2.0, (index + 1) * np.pi / 2.0
        first_ray = np.asarray((np.cos(first), np.sin(first)))
        last_ray = np.asarray((np.cos(last), np.sin(last)))
        blocks.append(
            TransfiniteBlock(
                f"sector:{index}",
                (
                    _curve_control(inner, "inner", (first, last), angular),
                    _curve_control(outer, "outer", (first, last), angular),
                    _curve_control(
                        LineCurve(first_ray, first_ray),
                        f"ray:{index}",
                        (0.0, 1.0),
                        resolution,
                    ),
                    _curve_control(
                        LineCurve(last_ray, last_ray),
                        f"ray:{(index + 1) % 4}",
                        (0.0, 1.0),
                        resolution,
                    ),
                ),
            )
        )
    interfaces = tuple(
        BlockInterfaceControl(
            f"sector:{index}", 3, f"sector:{(index + 1) % 4}", 2, (0,), (False,)
        )
        for index in range(4)
    )
    source = NativeStructuredSource(
        tuple(blocks),
        interfaces,
        domain.source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=0.08,
        domain=domain,
    )
    exact_area = (
        sum(
            Fraction(float(points[a, 0])) * Fraction(float(points[b, 1]))
            - Fraction(float(points[a, 1])) * Fraction(float(points[b, 0]))
            for a, b in facets
        )
        / 2
    )
    specification = _surface_specification(
        source,
        resolution,
        "quadrilateral",
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("structured_transfinite"),
        "structured_transfinite",
        {
            "fixture": "NativeStructuredSource:source-circle-annular-block-cycle",
            "inner_radius_m": 1.0,
            "outer_radius_m": 2.0,
            "sector_count": 4,
            "angular_intervals_per_quadrant": angular,
            "radial_intervals": resolution,
            "polygon_domain_area_m2": float(exact_area),
            "continuous_circle_domain_area_m2": 3.0 * np.pi,
            "maximum_deviation_m": 0.08,
            "resolution": resolution,
        },
        float(exact_area),
        expected_regions={"annulus": float(exact_area)},
        expected_domain=domain,
        fidelity_tolerance=0.08,
        quality_limits={"minimum_angle": 0.01},
    )


def _reversed_multiblock(resolution: int, materials: bool) -> _FamilyInput:
    width = 2.0 if materials else 1.0
    split = width / 2.0
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (split, 0.0, 0.0),
            (split, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (split, 0.0, 1.0),
            (split, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (width, 0.0, 0.0),
            (width, 1.0, 0.0),
            (width, 0.0, 1.0),
            (width, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    source_cells = ((0, 1, 2, 3, 4, 5, 6, 7), (1, 8, 9, 2, 5, 10, 11, 6))
    face_template = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    face_rows: dict[tuple[int, ...], tuple[tuple[int, ...], list[int]]] = {}
    for material, cell in enumerate(source_cells):
        for face in face_template:
            row = tuple(cell[index] for index in face)
            key = tuple(sorted(row))
            if key in face_rows:
                face_rows[key][1][1] = material
            else:
                face_rows[key] = (row, [material, -1])
    facets: list[tuple[int, int, int]] = []
    adjacency: list[tuple[int, int]] = []
    for row, labels in face_rows.values():
        if not materials and labels[1] >= 0:
            continue
        a, b, c, d = row
        facets.extend(((a, b, c), (a, c, d)))
        if materials:
            positive, negative = int(labels[0]), int(labels[1])
            adjacency.extend(((positive, negative), (positive, negative)))
        else:
            adjacency.extend(((0, -1), (0, -1)))
    region_ids = ("left", "right") if materials else ("solid",)
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64),
        np.asarray(adjacency, dtype=np.int64),
        region_ids,
        source_id="reversed-block-source",
    )
    query = _pwl_query(domain)
    blocks = []
    for name, origin, diagonal in (
        ("left", (0.0, 0.0, 0.0), (split, 1.0, 1.0)),
        ("right", (split, 1.0, 1.0), (split, -1.0, -1.0)),
    ):
        matrix = np.diag(np.asarray(diagonal, dtype=np.float64))
        controls = []
        for axis in range(3):
            other = tuple(index for index in range(3) if index != axis)
            for side in range(2):
                patch = PlanePatch(
                    np.asarray(origin) + side * matrix[:, axis],
                    matrix[:, other[0]],
                    matrix[:, other[1]],
                )
                controls.append(
                    TransfiniteSurfaceControl(
                        patch,
                        f"{name}:{2 * axis + side}",
                        np.asarray(((0.0, 0.0), (1.0, 1.0))),
                        (resolution, resolution),
                    )
                )
        blocks.append(TransfiniteBlock(name, tuple(controls)))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0, 1), (True, True))
    block_regions = (("left", "left"), ("right", "right")) if materials else ()
    source = NativeStructuredSource(
        tuple(blocks),
        (interface,),
        domain.source_id,
        domain.source_revision,
        fidelity_source=query,
        maximum_deviation=0.2,
        domain=domain,
        block_regions=block_regions,
    )
    regions: tuple[RegionControl, ...] = ()
    patches: tuple[PatchControl, ...] = ()
    if materials:
        # Donor RegionControls over the two source materials and its explicit
        # material-interface PatchControl, bound to the PWL owner entity banks.
        owner = query.owner
        if not isinstance(owner, PiecewiseLinearDomainBoundaryOwner):
            raise TypeError("Reversed material source lost its PWL entity owner.")
        kind = MeshingScope.whole(source, 2).entity_kind
        regions = tuple(
            RegionControl(
                MeshingScope(
                    source.source_id,
                    source.source_revision,
                    kind,
                    3,
                    owner.scope_entity_set_id(3),
                    np.asarray((index,), dtype=np.int64),
                ),
                name,
                f"material-{name}",
                RegionRole.SOLID,
            )
            for index, name in enumerate(region_ids)
        )
        sides = np.asarray(domain.facet_regions, dtype=np.int64)
        interface_ids = np.flatnonzero(np.all(sides >= 0, axis=1)).astype(np.int64)
        patches = (
            PatchControl(
                "material-interface",
                MeshingScope(
                    source.source_id,
                    source.source_revision,
                    kind,
                    2,
                    owner.scope_entity_set_id(2),
                    interface_ids,
                ),
                ("left", "right"),
            ),
        )
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        region_controls=regions,
        patch_controls=patches,
    )
    expected_regions = {"left": 1.0, "right": 1.0} if materials else {"solid": 1.0}
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("structured_transfinite"),
        "structured_transfinite",
        {
            "fixture": "NativeStructuredSource:reversed-two-material-boxes"
            if materials
            else "NativeStructuredSource:reversed-logical-face-boxes",
            "box_union_m": ((0.0, 0.0, 0.0), (width, 1.0, 1.0)),
            "block_widths_m": (split, split),
            "reversed_right_logical_axes": ("y", "z"),
            "interface_permutation": (0, 1),
            "interface_flips": (True, True),
            "material_regions": expected_regions if materials else {"solid": 1.0},
            "resolution": resolution,
            "maximum_deviation_m": 0.2,
        },
        sum(expected_regions.values()),
        expected_regions=expected_regions,
        expected_domain=domain,
        fidelity_tolerance=0.2,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _sweep_source(resolution: int, family: str) -> _FamilyInput:
    domain = _cube_domain()
    profile = _cube_profile(resolution, family)
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        phx.meshing.LayerSchedule(
            np.full(resolution, 1.0 / resolution, dtype=np.float64)
        ),
        "box-extrusion",
    )
    source = NativeSweepSource(
        profile,
        control,
        domain.source_id,
        domain.source_revision,
        fidelity_source=_pwl_query(domain),
        maximum_deviation=0.2,
        domain=domain,
        domain_vertex_map=_cube_domain_vertex_map(domain, profile, resolution),
    )
    specification = _volume_specification(
        source,
        resolution,
        "prism" if family == "triangle" else "hexahedron",
        fill_strategy=VolumeFillStrategy.SWEEP,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("sweep"),
        "sweep",
        {
            "fixture": "NativeSweepSource:declared-triangle-extrusion"
            if family == "triangle"
            else "NativeSweepSource:declared-square-extrusion",
            "base_profile": "unit-square triangulated into two triangles per grid square"
            if family == "triangle"
            else "unit-square quadrilateral grid",
            "base_profile_resolution": resolution,
            "extrusion_axis": (0.0, 0.0, 1.0),
            "layer_thicknesses_m": [1.0 / resolution] * resolution,
            "domain_vertices_m": np.asarray(domain.vertices).tolist(),
            "domain_volume_m3": 1.0,
            "domain_vertex_map": np.asarray(
                source.domain_vertex_map, dtype=np.int64
            ).tolist(),
            "maximum_deviation_m": 0.2,
        },
        1.0,
        expected_regions={"solid": 1.0},
        expected_domain=domain,
        fidelity_tolerance=0.2,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _frame_sweep_source(resolution: int) -> _FamilyInput:
    axis = np.asarray(
        tuple(index / resolution for index in range(resolution + 1)), dtype=np.float64
    )
    uv = np.asarray(tuple((u, v) for u in axis for v in axis), dtype=np.float64)
    profile_cells = np.asarray(
        tuple(
            (
                i * (resolution + 1) + j,
                (i + 1) * (resolution + 1) + j,
                (i + 1) * (resolution + 1) + j + 1,
                i * (resolution + 1) + j + 1,
            )
            for i in range(resolution)
            for j in range(resolution)
        ),
        dtype=np.int32,
    )
    xy = np.column_stack(
        (uv[:, 0] + 0.25 * uv[:, 1], uv[:, 1] + 0.125 * uv[:, 0] * (1.0 - uv[:, 0]))
    )
    profile_points = np.column_stack((xy, np.zeros(xy.shape[0], dtype=np.float64)))
    profile = CellMesh(
        profile_points,
        (
            CellBlock(
                "profile",
                "quadrilateral",
                profile_cells,
                global_ids=np.arange(resolution**2, dtype=np.int64) + 500,
            ),
        ),
        vertex_global_ids=np.arange(uv.shape[0], dtype=np.int64) + 40,
    )
    element = lagrange_element("quadrilateral", 2)
    local = np.asarray(element.reference_nodes, dtype=np.float64)
    controls = []
    for i in range(resolution):
        for j in range(resolution):
            u, v = (i + local[:, 0]) / resolution, (j + local[:, 1]) / resolution
            controls.append(
                np.column_stack(
                    (u + 0.25 * v, v + 0.125 * u * (1.0 - u), np.zeros_like(u))
                )
            )
    bank = np.concatenate(controls)
    routes = np.arange(bank.shape[0], dtype=np.int32).reshape(
        (resolution**2, element.local_dof_count)
    )
    profile_geometry = CellGeometrySpec({"profile": element}, {"profile": routes}, bank)
    angles = 0.15 * axis
    frames = np.asarray(
        tuple(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
            for angle in angles
        ),
        dtype=np.float64,
    )
    translations = axis[:, None] * np.asarray((0.1, -0.05, 1.0), dtype=np.float64)
    control = SweepControl(
        SweepMapKind.FRAMES,
        phx.meshing.LayerSchedule(
            np.full(resolution, 1.0 / resolution, dtype=np.float64)
        ),
        "authored-curved-profile-absolute-frames",
        frames=frames,
        translations=translations,
    )
    reference_points = np.asarray(
        tuple((u, v, w) for w in axis for u, v in uv), dtype=np.float64
    )
    root_cells = np.asarray(
        tuple(
            tuple(profile_cell + layer * uv.shape[0])
            + tuple(profile_cell + (layer + 1) * uv.shape[0])
            for layer in range(resolution)
            for profile_cell in profile_cells
        ),
        dtype=np.int32,
    )
    root_cell_ids = np.arange(root_cells.shape[0], dtype=np.int64) + 900
    root_vertex_ids = np.arange(reference_points.shape[0], dtype=np.int64) + 100
    reference_mesh = CellMesh(
        reference_points,
        (CellBlock("root", "hexahedron", root_cells, global_ids=root_cell_ids),),
        vertex_global_ids=root_vertex_ids,
    )
    cube_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    reference_facets = np.asarray(
        tuple(triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))),
        dtype=np.int64,
    )
    reference_domain = PiecewiseLinearDomain(
        cube_vertices,
        reference_facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (reference_facets.shape[0], 1)),
        ("solid",),
        source_id="curved-frame-reference",
    )
    transformed = (
        bank[None, :, :] @ np.swapaxes(frames, -1, -2) + translations[:, None, :]
    )
    root_controls = np.concatenate(
        tuple(
            np.concatenate((transformed[layer, route], transformed[layer + 1, route]))
            for layer in range(resolution)
            for route in routes
        )
    )
    root_element = swept_coordinate_element(element)
    root_routes = np.arange(root_controls.shape[0], dtype=np.int32).reshape(
        root_cells.shape[0], root_element.local_dof_count
    )
    root_geometry = CellGeometrySpec(
        {"root": root_element}, {"root": root_routes}, root_controls
    )
    # The authored map interpolates the binary64 station controls; its
    # intermediate frames are not rigid rotations. Integrate that source map
    # exactly rather than assigning it the idealized unit-rotation volume.
    from phydrax.discretization._cell_geometry import coordinate_effective_basis
    from phydrax.discretization._coordinate_enclosure import (
        linear_combinations,
        RationalPolynomial,
    )
    from phydrax.discretization._mapped_geometry_measure import (
        _coordinate_jacobian,
        _determinant,
        _polynomial_integral,
    )

    basis = coordinate_effective_basis(root_element)
    exact_controls = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in root_controls
    )
    source_volume = Fraction(0)
    for route in root_routes:
        components = tuple(
            tuple(exact_controls[int(index)][axis] for index in route)
            for axis in range(3)
        )
        coordinates = linear_combinations(basis, components)
        determinant = _determinant(_coordinate_jacobian(coordinates, 3), 3)
        if isinstance(determinant, RationalPolynomial):
            raise ValueError("Authored sweep root map must remain polynomial.")
        source_volume += _polynomial_integral(determinant, "hexahedron")
    revision = canonical_fingerprint(
        {
            "kind": "authored-curved-frame-source",
            "profile_basis": element.element_id,
            "profile_controls": bank,
            "frames": frames,
            "translations": translations,
        }
    )
    reference_points_array = np.asarray(reference_points, dtype=np.float64).reshape(
        (-1, 3)
    )
    reference_vertex_row: dict[tuple[float, float, float], int] = {}
    for row in range(reference_points_array.shape[0]):
        key = (
            float(reference_points_array[row, 0]),
            float(reference_points_array[row, 1]),
            float(reference_points_array[row, 2]),
        )
        reference_vertex_row[key] = row
    reference_domain_vertices = np.asarray(cube_vertices, dtype=np.float64).reshape(
        (-1, 3)
    )
    reference_domain_root_ids = np.empty(
        (reference_domain_vertices.shape[0],), dtype=np.int64
    )
    for row in range(reference_domain_vertices.shape[0]):
        key = (
            float(reference_domain_vertices[row, 0]),
            float(reference_domain_vertices[row, 1]),
            float(reference_domain_vertices[row, 2]),
        )
        reference_domain_root_ids[row] = root_vertex_ids[reference_vertex_row[key]]
    source_id = "curved-frame-source"
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        root_geometry,
        np.zeros(root_cells.shape[0], dtype=np.int64),
        source_id=source_id,
        source_revision=revision,
        coordinate_contract=_CONTRACT,
        reference_domain_vertex_root_ids=reference_domain_root_ids,
    )
    reference_polygons: tuple[tuple[int, int, int], ...] = tuple(
        triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))
    )
    reference_complex = _plc(
        cube_vertices,
        reference_polygons,
        np.arange(len(reference_polygons), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(reference_polygons), 1)),
        ("solid",),
        vertex_prefix=source_id,
        facet_ids=tuple(
            f"{source_id}:facet:{row}" for row in range(len(reference_polygons))
        ),
    )
    reference_source = NativePlcSource(reference_complex, source_id, revision)
    mapped_authority = NativeMappedHexSource(
        reference_source,
        domain,
        plc_vertex_root_global_ids=reference_domain_root_ids,
    )
    query = mapped_authority.boundary_query()
    correspondence = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        reference_mesh.topology_id,
        {"sweep:profile": root_cell_ids},
        {"sweep:profile": root_vertex_ids[root_cells]},
    )
    source = NativeSweepSource(
        profile,
        control,
        source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=1.0e-10,
        domain=domain,
        profile_geometry=profile_geometry,
        root_correspondence=correspondence,
    )
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        geometry_order=2,
        fill_strategy=VolumeFillStrategy.SWEEP,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("sweep"),
        "sweep",
        {
            "fixture": "NativeSweepSource:authored-curved-profile-absolute-frames",
            "profile_map": "(u,v) -> (u+v/4, v+u*(1-u)/8, 0)",
            "station_frames": np.asarray(frames).tolist(),
            "station_translations_m": np.asarray(translations).tolist(),
            "layer_thicknesses_m": [1.0 / resolution] * resolution,
            "reference_domain_m": "unit cube; exact determinant integral of binary64 station-interpolated source controls",
            "root_hex_count": int(root_cells.shape[0]),
            "geometry_order": 2,
            "maximum_deviation_m": 1.0e-10,
            "resolution": resolution,
        },
        float(source_volume),
        expected_regions={"solid": float(source_volume)},
        expected_domain=domain,
        fidelity_tolerance=1.0e-10,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _source_scope(source: Any, dimension: int, identifiers: Any) -> MeshingScope:
    boundary = MeshingScope.whole(source, 2)
    return MeshingScope(
        source.source_id,
        source.source_revision,
        boundary.entity_kind,
        dimension,
        source.scope_entity_set_id(dimension),
        np.sort(np.asarray(identifiers, dtype=np.int64)),
    )


def _plc_region_controls(
    source: NativePlcSource,
) -> tuple[RegionControl, ...]:
    """Donor RegionControl per authored PLC region, bound by its stable region ID."""
    complex_ = source.complex
    return tuple(
        RegionControl(
            _source_scope(source, 3, (int(complex_.region_entity_ids[index]),)),
            name,
            f"material-{name}",
            RegionRole.SOLID,
        )
        for index, name in enumerate(complex_.region_ids)
    )


def _plc_source_case(name: str, resolution: int) -> _FamilyInput:
    complex_ = _cavity_plc()
    source = NativePlcSource(complex_, "material-cavity", "cavity-revision")
    interface_id = int(complex_.facet_entity_ids[2])
    cavity_id = int(complex_.facet_entity_ids[3])
    interface_scope = _source_scope(source, 2, (interface_id,))
    cavity_scope = _source_scope(source, 2, (cavity_id,))
    regions = _plc_region_controls(source)
    patches = (
        PatchControl("material-interface", interface_scope, ("left", "right")),
        PatchControl("cavity-wall", cavity_scope, ("right",)),
    )
    holes = (HoleSeed(np.asarray((1.5, 0.5, 0.5)), cavity_scope),)
    if name == "hex-dominant":
        route = "plc_hex_dominant"
        fill = VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION
        options = NativeMeshingOptions(route, hex_core_fraction=0.8)
    else:
        route = "plc_dual_hex"
        # MULTIZONE is now refused for PLC sources; SIMPLEX is the approved cutover.
        fill = VolumeFillStrategy.SIMPLEX
        options = NativeMeshingOptions(route)
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        fill_strategy=fill,
        region_controls=regions,
        patch_controls=patches,
        hole_seeds=holes,
        mixed=name == "hex-dominant",
    )
    expected = {"left": 1.0, "right": 0.875}
    return _FamilyInput(
        source,
        specification,
        options,
        route,
        {
            "fixture": "NativePlcSource:two-unit-box-material-interface-with-right-box-cavity",
            "outer_box_m": ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            "left_region_m": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "right_region_m": ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            "cavity_m": ((1.25, 0.25, 0.25), (1.75, 0.75, 0.75)),
            "material_interface_facet_id": interface_id,
            "cavity_facet_id": cavity_id,
            "region_controls": [control.region_name for control in regions],
            "patch_controls": [patch.name for patch in patches],
            "hole_seed_point_m": (1.5, 0.5, 0.5),
            "native_options": "donor route defaults",
            "size_target_m": 1.0 / resolution,
            "maximum_size_m": 2.0 / resolution,
            "resolution": resolution,
            "hex_core_fraction": 0.8 if name == "hex-dominant" else None,
        },
        sum(expected.values()),
        expected_regions=expected,
        require_region_coverage=True,
        fidelity_tolerance=0.0,
        require_transition=name == "hex-dominant",
        quality_limits=dict(_QUALITY_LIMITS),
    )


_NONSWEEPABLE_PATCH_NAMES = ("outer-wall", "cavity-wall-0", "cavity-wall-1")


def _nonsweepable_dual_hex(resolution: int) -> _FamilyInput:
    complex_, exact_volume = _nonsweepable_plc()
    proof = _nonsweepable_source_proof(complex_)
    if Fraction(proof["solid_volume"]) != Fraction(exact_volume):
        raise ValueError("Nonsweepable exact source volume disagrees with its proof.")
    source = NativePlcSource(
        complex_, "qualification-nonsweepable-dual-hex", "dyadic-two-cavities-r1"
    )
    facet_ids = np.asarray(complex_.facet_entity_ids, dtype=np.int64)
    shell_scopes = tuple(
        _source_scope(source, 2, facet_ids[offset : offset + 4]) for offset in (0, 4, 8)
    )
    patches = tuple(
        PatchControl(name, scope, ("solid",))
        for name, scope in zip(_NONSWEEPABLE_PATCH_NAMES, shell_scopes, strict=True)
    )
    cavity_vertices = np.asarray(complex_.vertices, dtype=np.float64)
    holes = (
        HoleSeed(cavity_vertices[4:8].mean(axis=0), shell_scopes[1]),
        HoleSeed(cavity_vertices[8:12].mean(axis=0), shell_scopes[2]),
    )
    regions = _plc_region_controls(source)
    route = "plc_dual_hex"
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        fill_strategy=VolumeFillStrategy.SIMPLEX,
        region_controls=regions,
        patch_controls=patches,
        hole_seeds=holes,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions(route),
        route,
        {
            "fixture": "NativePlcSource:unit-tetrahedron-two-enclosed-tetrahedral-cavities",
            "outer_tetrahedron_vertices_m": np.asarray(complex_.vertices[:4]).tolist(),
            "cavity_tetrahedron_vertices_m": [
                np.asarray(complex_.vertices[4:8]).tolist(),
                np.asarray(complex_.vertices[8:12]).tolist(),
            ],
            "boundary_components": 3,
            "cavities_strictly_enclosed_and_disjoint": True,
            "exact_rational_domain_volume_m3": proof["solid_volume"],
            "region_controls": [control.region_name for control in regions],
            "patch_controls": list(_NONSWEEPABLE_PATCH_NAMES),
            "native_options": "donor route defaults",
            "resolution": resolution,
        },
        exact_volume,
        expected_regions={"solid": exact_volume},
        fidelity_tolerance=0.0,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
        nonsweepability_required=True,
        authored_topology=_authored_topology(complex_),
        source_construction_proof=proof,
    )


def _mapped_hex_case(name: str, resolution: int) -> _FamilyInput:
    source, region_measures, details = _curved_occupied_source(name)
    # Donor RegionControl per authored root region. The current mapped route
    # binds a region scope by the reference PLC region entity, not by root-cell
    # IDs (those are refused as "mapped source region identity").
    complex_ = source.reference.complex
    regions = tuple(
        RegionControl(
            _source_scope(source, 3, (int(complex_.region_entity_ids[index]),)),
            region,
            f"material-{region}",
            RegionRole.SOLID,
        )
        for index, region in enumerate(complex_.region_ids)
    )
    # Prove each bound region entity denotes exactly the donor root-cell scope
    # (domain.cell_regions == donor region index) and the donor material.
    domain = source.domain
    root_ids = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in domain.reference_mesh.blocks
        ]
    )
    cell_regions = np.asarray(domain.cell_regions, dtype=np.int64)
    region_rows = {
        int(identifier): row for row, identifier in enumerate(complex_.region_entity_ids)
    }
    root_scope_equality = []
    for donor_index, donor_region in enumerate(domain.region_ids):
        (control,) = (item for item in regions if item.region_name == donor_region)
        (bound_entity,) = np.asarray(control.scope.entity_ids, dtype=np.int64).tolist()
        bound_region = complex_.region_ids[region_rows[int(bound_entity)]]
        donor_cells = np.sort(root_ids[cell_regions == donor_index])
        bound_cells = np.sort(
            root_ids[cell_regions == list(domain.region_ids).index(bound_region)]
        )
        if (
            bound_region != donor_region
            or control.material_id != f"material-{donor_region}"
            or donor_cells.size == 0
            or not np.array_equal(bound_cells, donor_cells)
        ):
            raise ValueError(
                f"Region entity scope for {donor_region!r} differs from the donor root cells."
            )
        root_scope_equality.append(
            {
                "region": donor_region,
                "bound_region_entity_id": int(bound_entity),
                "material_id": control.material_id,
                "donor_root_cell_ids": donor_cells.tolist(),
                "bound_root_cell_ids_equal": True,
            }
        )
    route = (
        "mapped_balanced_grid_hex"
        if name == "curved-cavity-hex"
        else "mapped_frame_grid_hex"
    )
    options = NativeMeshingOptions(route)
    grid = options.grid_schedule
    if grid is None:
        raise RuntimeError("Mapped grid route omitted its default schedule.")
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        geometry_order=2,
        region_controls=regions,
    )
    return _FamilyInput(
        source,
        specification,
        options,
        route,
        {
            "fixture": f"NativeMappedHexSource:{name}",
            **details,
            "source_map_revision": source.source_revision,
            "requested_geometry_order": 2,
            "resolution": resolution,
            "region_controls": [control.region_name for control in regions],
            "region_scope_root_cell_equality": root_scope_equality,
            "grid_schedule": {
                "route": grid.route,
                "source": "donor route default",
                "minimum_scaled_jacobian": grid.minimum_scaled_jacobian,
                "minimum_mean_ratio": grid.minimum_mean_ratio,
                "maximum_aspect_ratio": grid.maximum_aspect_ratio,
            },
        },
        sum(region_measures.values()),
        expected_regions=region_measures,
        expected_domain=source.domain,
        fidelity_tolerance=0.0,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _quarter_surface_case(resolution: int) -> _FamilyInput:
    source = _quarter_extrusion()
    scope = MeshingScope.whole(source, 2)
    protected = (ProtectedFeature(scope, FeatureKind.SURFACE, maximum_deviation=0.0),)
    specification = _surface_specification(
        source,
        resolution,
        "quadrilateral",
        ambient_dimension=3,
        geometry_order=3,
        protected_features=protected,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("parametric_surface"),
        "parametric_surface",
        {
            "fixture": "NativeSurfaceSource:original-quarter-extrusion",
            "rational_quarter_arc_radius_m": 1.0,
            "extrusion_length_m": 1.0,
            "surface_control_points_m": np.asarray(
                [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)]
            ).tolist(),
            "rational_weights": (1.0, float(np.sqrt(0.5)), 1.0),
            "exact_surface_area_m2": float(np.pi / 2.0),
            "geometry_order": 3,
            "maximum_deviation_m": 0.0,
            "resolution": resolution,
        },
        float(np.pi / 2.0),
        fidelity_tolerance=0.0,
        quality_limits={"minimum_angle": 0.01},
        # One authored patch: every loop is a source boundary loop of the surface.
        authored_boundary={"loops": len(source.patches[0].loops)},
    )


def _make_profile(profile_id: str) -> _FamilyInput:
    case_name = profile_id.removeprefix(_PROFILE_PREFIX)
    larger = case_name.endswith(":larger")
    name = case_name.removesuffix(":larger")
    resolution = _BASE_RESOLUTION * (2 if larger else 1)
    if name == "dual-quad":
        return _planar_dual_quad(resolution)
    if name == "curved-quarter-extrusion-quad":
        return _quarter_surface_case(resolution)
    if name in ("dual-hex", "hex-dominant"):
        return _plc_source_case(name, resolution)
    if name == "nonsweepable-dual-hex":
        return _nonsweepable_dual_hex(resolution)
    if name == "curved-frame-sweep-hex":
        return _frame_sweep_source(resolution)
    if name == "multiblock-reversed-hex":
        return _reversed_multiblock(resolution, False)
    if name == "multiblock-material-hex":
        return _reversed_multiblock(resolution, True)
    if name == "multiblock-cycle-quad":
        return _annular_blocks(resolution)
    if name in ("curved-nonsweepable-hex", "curved-cavity-hex", "curved-material-hex"):
        return _mapped_hex_case(name, resolution)
    if name == "transfinite-quad":
        return _structured_square(resolution, False)
    if name == "multiblock-quad":
        return _structured_square(resolution, True)
    if name == "sweep-hex":
        return _sweep_source(resolution, "quadrilateral")
    if name == "sweep-prism":
        return _sweep_source(resolution, "triangle")
    raise ValueError(f"Unregistered native family source profile {profile_id!r}.")


def _source_record(source: Any) -> dict[str, object]:
    try:
        domain = getattr(source, "domain", None)
    except (AttributeError, ValueError):
        domain = None
    if domain is None and isinstance(source, MeshingDomain):
        domain = source
    result: dict[str, object] = {
        "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
        "source_id": getattr(source, "source_id", None),
        "source_revision": getattr(source, "source_revision", None),
        "binding_id": getattr(source, "binding_id", None),
        "domain_id": getattr(domain, "domain_id", None),
    }
    complex_ = getattr(source, "complex", None)
    if complex_ is not None:
        result.update(
            {
                "complex_id": complex_.complex_id,
                "source_vertices": int(np.asarray(complex_.vertices).shape[0]),
                "source_vertex_coordinates_m": np.asarray(complex_.vertices).tolist(),
                "source_vertex_ids": tuple(complex_.vertex_ids),
                "source_facets": int(complex_.facet_count),
                "source_facet_ids": tuple(complex_.facet_ids),
                "source_facet_region_sides": np.asarray(complex_.facet_regions).tolist(),
                "source_polygon_offsets": np.asarray(complex_.polygon_offsets).tolist(),
                "source_polygon_vertices": np.asarray(complex_.polygon_vertices).tolist(),
                "source_polygon_facet_indices": np.asarray(
                    complex_.polygon_facets
                ).tolist(),
                "source_regions": tuple(complex_.region_ids),
            }
        )
    if isinstance(domain, PiecewiseLinearDomain):
        result["domain_region_ids"] = tuple(domain.region_ids)
        result["domain_vertices"] = np.asarray(domain.vertices).tolist()
        result["domain_facets"] = np.asarray(domain.facets).tolist()
        result["domain_facet_region_sides"] = np.asarray(domain.facet_regions).tolist()
    elif isinstance(domain, MappedReferenceDomain):
        reference = domain.reference_domain
        root_mesh = domain.reference_mesh
        result["domain_region_ids"] = tuple(domain.region_ids)
        result["reference_domain_vertices"] = np.asarray(reference.vertices).tolist()
        result["reference_domain_facets"] = np.asarray(reference.facets).tolist()
        result["reference_domain_facet_region_sides"] = np.asarray(
            reference.facet_regions
        ).tolist()
        result["root_coordinates"] = np.asarray(root_mesh.coordinates).tolist()
        result["root_vertex_ids"] = np.asarray(root_mesh.vertex_global_ids).tolist()
        result["root_cell_connectivity"] = {
            block.name: np.asarray(block.vertices).tolist() for block in root_mesh.blocks
        }
        result["root_cell_regions"] = np.asarray(domain.cell_regions).tolist()
        result["root_physical_coordinate_dofs"] = np.asarray(
            domain.source_geometry.coordinates
        ).tolist()
        result["mapped_domain_id"] = domain.domain_id
    blocks = getattr(source, "blocks", None)
    if blocks is not None:
        result["block_ids"] = tuple(block.name for block in blocks)
        result["block_intervals"] = {
            block.name: tuple(block.intervals) for block in blocks
        }
        result["interface_ids"] = tuple(
            interface.control_id for interface in source.interfaces
        )
    profile = getattr(source, "profile", None)
    if profile is not None:
        result["profile_mesh_id"] = profile.mesh_id
        result["profile_coordinates"] = np.asarray(profile.coordinates).tolist()
        result["profile_cell_families"] = tuple(
            block.cell_kind for block in profile.blocks
        )
        result["profile_cell_connectivity"] = {
            block.name: np.asarray(block.vertices).tolist() for block in profile.blocks
        }
        result["profile_cell_count"] = sum(block.cell_count for block in profile.blocks)
        result["profile_geometry_dofs"] = np.asarray(
            source.profile_geometry.coordinates
        ).tolist()
    return result


def _request_record(case: _FamilyInput) -> dict[str, object]:
    specification = case.specification
    target = specification.target
    sizes = tuple(
        {
            "type": type(control).__name__,
            "target_size_m": getattr(control, "target_size", None),
            "maximum_size_m": getattr(control, "maximum_size", None),
            "strength": getattr(getattr(control, "strength", None), "value", None),
        }
        for control in specification.size_controls
    )
    families = target.cell_families
    schedule_records: dict[str, object] = {}
    schedule_fields = (
        "route",
        "radius_edge_bound",
        "minimum_dihedral_degrees",
        "minimum_scaled_jacobian",
        "minimum_mean_ratio",
        "maximum_aspect_ratio",
        "quality_angle_degrees",
        "spacing_fraction",
        "minimum_edge_samples",
        "optimization_steps",
    )
    for schedule_name in (
        "curve_schedule",
        "surface_schedule",
        "volume_schedule",
        "structured_schedule",
        "grid_schedule",
    ):
        schedule = getattr(case.options, schedule_name)
        if schedule is None:
            continue
        schedule_record: dict[str, object] = {
            "schedule_id": getattr(schedule, "schedule_id", None)
        }
        for field_name in schedule_fields:
            if hasattr(schedule, field_name):
                value = getattr(schedule, field_name)
                schedule_record[field_name] = getattr(value, "value", value)
        schedule_records[schedule_name] = schedule_record
    quality_target = (
        specification.quality_target
        if isinstance(specification, SurfaceMeshingSpec)
        else None
    )
    return {
        "specification_id": specification.specification_id,
        "requested_route": case.expected_route,
        "options": {
            "route": case.options.route,
            "hex_core_fraction": case.options.hex_core_fraction,
            "schedules": schedule_records,
        },
        "topological_dimension": target.topological_dimension,
        "ambient_dimension": target.ambient_dimension,
        "geometry_order": target.geometry_order,
        "family_policy": {
            "required": tuple(families.required),
            "preferred": tuple(families.preferred),
            "allowed_transitions": tuple(families.allowed_transitions),
            "allow_mixed": families.allow_mixed,
        },
        "fill_strategy": getattr(
            getattr(specification, "fill_strategy", None), "value", None
        ),
        "size_controls": sizes,
        "size_compliance": {
            "relative_tolerance": specification.size_compliance.relative_tolerance,
            "target_statistics": tuple(specification.size_compliance.target_statistics),
        },
        "quality_target": None
        if quality_target is None
        else {
            "minimum_angle_radians": quality_target.minimum_angle,
            "hard": quality_target.hard,
        },
    }


def _measure_coverage(result: Any) -> dict[str, float]:
    certification = result.certification
    coverage = None if certification is None else certification.coverage
    if coverage is None:
        return {}
    return {
        name: float(measure)
        for name, measure in zip(
            coverage.region_ids,
            coverage.achieved_region_measures,
            strict=True,
        )
        if measure is not None
    }


def _family_counts(result: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in result.mesh.blocks:
        counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
    return counts


def _fidelity_record(case: _FamilyInput, result: Any) -> dict[str, object]:
    certification = result.certification
    fidelity = None if certification is None else certification.fidelity
    record: dict[str, object] = {
        "declared_maximum_deviation_m": case.fidelity_tolerance,
        "certificate_present": fidelity is not None,
    }
    if fidelity is not None:
        record.update(
            {
                "mesh_to_source_upper_m": float(fidelity.mesh_to_source_upper),
                "source_to_mesh_upper_m": float(fidelity.source_to_mesh_upper),
                "certificate_id": fidelity.certificate_id,
            }
        )
        if case.fidelity_tolerance is not None and (
            fidelity.mesh_to_source_upper > case.fidelity_tolerance + 1.0e-12
            or fidelity.source_to_mesh_upper > case.fidelity_tolerance + 1.0e-12
        ):
            raise RuntimeError(
                "Source-fidelity certificate exceeds the profile's declared maximum deviation."
            )
    elif case.fidelity_tolerance is not None:
        if (
            isinstance(case.source, NativePlcSource)
            and case.source.fidelity_source is None
        ):
            record["fidelity_basis"] = (
                "exact retained PLC facet identity and complete source associations"
            )
        elif isinstance(case.source, NativeMappedHexSource):
            record["fidelity_basis"] = (
                "exact mapped root-cell compositions, PLC-to-root vertex relation, and authored source/domain identity"
            )
        elif (
            isinstance(case.source, NativePlanarSource) and case.fidelity_tolerance == 0.0
        ):
            record["fidelity_basis"] = (
                "exact authored planar loops, source-edge associations, and zero-deviation source contract"
            )
        else:
            raise RuntimeError(
                "A source-bound fidelity certificate is required by this profile."
            )
    return record


# Donor c56614a9 family.py obligations. Classes derive from declared route,
# family and nonsweepability fields, never from a route outcome.
_FEATURE_QUAD_ROUTES = frozenset(("planar_dual_quad", "parametric_surface"))
_PHYSICAL_EDGE_PROFILES = frozenset(
    ("curved-quarter-extrusion-quad", "curved-frame-sweep-hex")
)
_MATERIAL_PROFILES = frozenset(
    (
        "dual-hex",
        "hex-dominant",
        "curved-material-hex",
        "nonsweepable-dual-hex",
        "multiblock-material-hex",
    )
)
_STRICT_ASSOCIATION_PROFILES = frozenset(
    (
        "dual-quad",
        "dual-hex",
        "nonsweepable-dual-hex",
        "hex-dominant",
        "curved-quarter-extrusion-quad",
    )
)
_EDGE_POLYLINE_SEGMENTS = 64


def _obligation_class(case: _FamilyInput) -> str:
    policy = case.specification.target.cell_families
    if policy.allow_mixed:
        return "declared-mixed-family-policy"
    if case.nonsweepability_required:
        return "general-nonsweepable-all-hex"
    if (
        set(policy.required) == {"quadrilateral"}
        and case.expected_route in _FEATURE_QUAD_ROUTES
    ):
        return "feature-conforming-quad"
    if case.expected_route == "sweep":
        return "declared-sweep"
    if set(policy.required) == {"hexahedron"}:
        return "declared-exact-all-hex"
    return "declared-structured-multiblock"


def _nonsweepable_source_proof(complex_: PiecewiseLinearComplex) -> dict[str, Any]:
    """Check exact source premises without using the mesh topology gate.

    A nondegenerate tetrahedron is a closed PL 3-ball. Strict barycentric
    containment and a strict separating plane make the two removed balls
    disjoint interior balls. Their complement in the outer ball is connected
    and has precisely the three checked spherical boundary components.
    Fractions represent the actual binary64 source coordinates.
    """
    points = tuple(
        tuple(Fraction(float(value)) for value in point) for point in complex_.vertices
    )

    def determinant(
        a: tuple[Fraction, ...],
        b: tuple[Fraction, ...],
        c: tuple[Fraction, ...],
        d: tuple[Fraction, ...],
    ) -> Fraction:
        u, v, w = tuple(
            tuple(point[axis] - a[axis] for axis in range(3)) for point in (b, c, d)
        )
        return (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
        )

    if (
        len(points) != 12
        or complex_.region_ids != ("solid",)
        or not np.array_equal(complex_.polygon_offsets, np.arange(13) * 3)
        or not np.array_equal(complex_.polygon_facets, np.arange(12))
        or not np.array_equal(complex_.facet_regions, np.tile((-1, 0), (12, 1)))
        or np.asarray(complex_.segments).size
    ):
        raise ValueError(
            "Nonsweepable source must retain its twelve oriented solid/void facets."
        )
    tetrahedra = tuple(points[offset : offset + 4] for offset in (0, 4, 8))
    determinants = tuple(determinant(*tetrahedron) for tetrahedron in tetrahedra)
    if any(value <= 0 for value in determinants):
        raise ValueError("Nonsweepable source tetrahedra must have positive volume.")
    outer = tetrahedra[0]
    barycentric_minima = []
    for cavity in tetrahedra[1:]:
        weights = []
        for point in cavity:
            for vertex in range(4):
                replaced = list(outer)
                replaced[vertex] = point
                weights.append(determinant(*replaced) / determinants[0])
        barycentric_minima.append(min(weights))
    separation = min(point[0] for point in tetrahedra[2]) - max(
        point[0] for point in tetrahedra[1]
    )
    if min(barycentric_minima) <= 0 or separation <= 0:
        raise ValueError(
            "Nonsweepable cavities must be strictly enclosed and strictly disjoint."
        )
    polygons = np.asarray(complex_.polygon_vertices).reshape(12, 3)
    origin = (Fraction(0),) * 3
    shells = []
    signed_volumes = []
    for shell, offset in enumerate((0, 4, 8)):
        faces = polygons[offset : offset + 4]
        vertices = set(range(offset, offset + 4))
        edges: dict[tuple[int, int], list[int]] = {}
        volume = Fraction(0)
        for face in faces:
            ids = tuple(map(int, face))
            if len(set(ids)) != 3 or not set(ids) < vertices:
                raise ValueError(
                    "Nonsweepable shell faces must bound their own tetrahedron."
                )
            opposite = next(iter(vertices - set(ids)))
            a, b, c = (points[index] for index in ids)
            side = determinant(a, b, c, points[opposite])
            if side >= 0 if shell == 0 else side <= 0:
                raise ValueError(
                    "Nonsweepable boundary orientation must point from material to void."
                )
            volume += determinant(origin, a, b, c) / 6
            for u, v in zip(ids, (*ids[1:], ids[0]), strict=True):
                edges.setdefault((min(u, v), max(u, v)), []).append(1 if u < v else -1)
        euler = len(vertices) - len(edges) + len(faces)
        if euler != 2 or any(
            len(sides) != 2 or sum(sides) != 0 for sides in edges.values()
        ):
            raise ValueError(
                "Nonsweepable boundary shells must be closed oriented tetrahedral spheres."
            )
        if volume != determinants[shell] / (6 if shell == 0 else -6):
            raise ValueError(
                "Nonsweepable shell signed volume must match its exact tetrahedron."
            )
        signed_volumes.append(volume)
        shells.append(
            {
                "shell": "outer" if shell == 0 else f"cavity-{shell - 1}",
                "euler_characteristic": euler,
                "signed_volume": str(volume),
                "orientation": "outward" if shell == 0 else "inward",
            }
        )
    solid_volume = sum(signed_volumes, Fraction(0))
    if solid_volume <= 0:
        raise ValueError("Nonsweepable solid must have positive exact volume.")
    return {
        "arithmetic": "exact-rational-of-authored-binary64-coordinates",
        "solid_connected": True,
        "connectedness_basis": "PL-ball-minus-two-disjoint-strictly-interior-PL-balls",
        "strict_cavity_containment": True,
        "minimum_outer_barycentric_coordinates": [str(v) for v in barycentric_minima],
        "strict_cavity_disjointness": True,
        "separating_axis": 0,
        "separation": str(separation),
        "boundary_components": shells,
        "solid_volume": str(solid_volume),
    }


def _loop_arrays(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray([loop.shape[0] for loop in loops], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    flat = (
        np.concatenate(loops).astype(np.int64)
        if loops
        else np.empty((0,), dtype=np.int64)
    )
    return offsets, flat


def _face_components(
    offsets: np.ndarray, flat: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label polygon components through shared undirected edges."""
    count = offsets.shape[0] - 1
    if count == 0:
        empty = np.empty((0,), dtype=np.int64)
        return 0, empty, empty, empty, np.empty((0, 2), dtype=np.int64)
    face = np.repeat(np.arange(count, dtype=np.int64), np.diff(offsets))
    successor = np.arange(flat.shape[0], dtype=np.int64) + 1
    successor[offsets[1:] - 1] = offsets[:-1]
    directed = np.stack((flat, flat[successor]), axis=1)
    edges, edge_index = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    edge_index = edge_index.reshape(-1)
    graph = coo_matrix(
        (np.ones(face.shape[0]), (face, count + edge_index)),
        shape=(count + edges.shape[0], count + edges.shape[0]),
    )
    components, labels = connected_components(graph, directed=False)
    return int(components), labels[:count], edge_index, successor, directed


def _surface_vertex_links(flat: np.ndarray, successor: np.ndarray) -> bool:
    """Every boundary vertex must have one circular link, not pinched sheets."""
    if flat.size == 0:
        return True
    predecessor = np.empty_like(successor)
    predecessor[successor] = np.arange(flat.size, dtype=np.int64)
    endpoints = np.concatenate(
        (
            np.stack((flat, flat[predecessor]), axis=1),
            np.stack((flat, flat[successor]), axis=1),
        ),
        axis=0,
    )
    link_nodes, inverse = np.unique(endpoints, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    first, second = inverse[: flat.size], inverse[flat.size :]
    degree = np.bincount(inverse, minlength=link_nodes.shape[0])
    graph = coo_matrix(
        (np.ones(flat.size, dtype=np.int8), (first, second)),
        shape=(link_nodes.shape[0], link_nodes.shape[0]),
    )
    _, labels = connected_components(graph, directed=False)
    vertices, vertex_index = np.unique(link_nodes[:, 0], return_inverse=True)
    lower = np.full(vertices.size, link_nodes.shape[0], dtype=np.int64)
    upper = np.full(vertices.size, -1, dtype=np.int64)
    np.minimum.at(lower, vertex_index, labels)
    np.maximum.at(upper, vertex_index, labels)
    return bool(np.all(degree == 2) and np.all(lower == upper))


def _surface_topology(
    points: np.ndarray, offsets: np.ndarray, flat: np.ndarray
) -> dict[str, Any]:
    """Classify an outward-oriented closed polygon surface independently of its producer."""
    components, labels, edge_index, successor, directed = _face_components(offsets, flat)
    edge_count = int(edge_index.max(initial=-1)) + 1
    uses = np.bincount(edge_index, minlength=edge_count)
    forward = np.bincount(
        edge_index,
        weights=(directed[:, 0] < directed[:, 1]).astype(np.float64),
        minlength=edge_count,
    )
    manifold = bool(
        np.all(uses == 2)
        and np.all(forward == 1.0)
        and _surface_vertex_links(flat, successor)
    )
    face = np.repeat(np.arange(offsets.shape[0] - 1), np.diff(offsets))
    face_label = labels[face]
    edge_label = np.zeros((edge_count,), dtype=np.int64)
    edge_label[edge_index] = face_label
    vertex_pairs = np.unique(np.stack((face_label, flat), axis=1), axis=0)
    vertices = np.bincount(vertex_pairs[:, 0], minlength=components)
    edges = np.bincount(edge_label, minlength=components)
    faces = np.bincount(labels, minlength=components)
    # Fan triangles about each loop's first vertex give the enclosed signed volume.
    first = np.repeat(flat[offsets[:-1]], np.diff(offsets))
    interior = np.ones(flat.shape[0], dtype=np.bool_)
    interior[offsets[:-1]] = False
    interior[offsets[1:] - 1] = False
    a, b, c = (
        points[first[interior]],
        points[flat[interior]],
        points[flat[successor[interior]]],
    )
    volume = np.bincount(
        face_label[interior],
        weights=np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0,
        minlength=components,
    )
    shells = sorted(
        (
            "outer" if volume[index] > 0.0 else "cavity",
            int(vertices[index] - edges[index] + faces[index]),
        )
        for index in range(components)
    )
    return {
        "closed_oriented_manifold": manifold,
        "components": [
            {"shell": shell, "euler_characteristic": euler} for shell, euler in shells
        ],
        "outer_shells": sum(shell == "outer" for shell, _ in shells),
        "enclosed_cavities": sum(shell == "cavity" for shell, _ in shells),
    }


def _authored_topology(complex_: PiecewiseLinearComplex) -> dict[str, Any]:
    """Derive solid, cavity, interface and nonsweepability facts from the authored PLC."""
    points = np.asarray(complex_.vertices, dtype=np.float64)
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    flat = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    incidence = np.asarray(complex_.facet_regions, dtype=np.int64)[
        np.asarray(complex_.polygon_facets, dtype=np.int64)
    ]
    loops = [flat[offsets[index] : offsets[index + 1]] for index in range(len(incidence))]
    void = incidence < 0
    # The right-hand normal points into incidence[:, 0]; outward means toward void.
    boundary = [
        loop if void[index, 0] else loop[::-1]
        for index, loop in enumerate(loops)
        if void[index, 0] != void[index, 1]
    ]
    interface = [
        loop
        for index, loop in enumerate(loops)
        if not void[index].any() and incidence[index, 0] != incidence[index, 1]
    ]
    topology = _surface_topology(points, *_loop_arrays(boundary))
    boundary_points = points[np.concatenate(boundary)]
    lower, upper = boundary_points.min(axis=0), boundary_points.max(axis=0)
    excluded = [
        axis
        for axis in range(3)
        if any(
            np.all(points[loop, axis] == points[loop[0], axis])
            and lower[axis] < points[loop[0], axis] < upper[axis]
            for loop in boundary
        )
    ]
    euler = tuple(
        component["euler_characteristic"] for component in topology["components"]
    )
    interval_product_excluded = len(euler) > 2 or (
        len(euler) == 2 and euler[0] != euler[1]
    )
    circle_product_excluded = any(value != 0 for value in euler)
    product_obstruction = (
        topology["closed_oriented_manifold"]
        and interval_product_excluded
        and circle_product_excluded
    )
    return {
        "boundary": topology,
        "interface_components": _face_components(*_loop_arrays(interface))[0],
        "nonsweepable_witness": {
            "axes_with_intermediate_caps": excluded,
            "interval_product_excluded": interval_product_excluded,
            "circle_product_excluded": circle_product_excluded,
            "requires_connected_solid": True,
            "witness": "boundary-component-product-obstruction"
            if product_obstruction
            else None,
            "scope": (
                "For a connected compact surface S, the boundary of S x I is connected "
                "or consists of two homeomorphic closed surfaces. The boundary of "
                "S x S1 is empty or consists only of tori. With independently proved "
                "solid connectivity, these boundary invariants exclude both regular sweeps."
            ),
        },
    }


def _published_topology(mesh: CellMesh, regions: np.ndarray) -> dict[str, Any]:
    """Recompute boundary/interface topology from published hexahedral connectivity only."""
    if any(block.cell_kind != "hexahedron" for block in mesh.blocks):
        raise RuntimeError("Published all-hex topology requires only hexahedra.")
    table = np.asarray(reference_cell_topology("hexahedron").entities[2], dtype=np.int64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    loops = cells[:, table].reshape((-1, table.shape[1]))
    owner = np.repeat(np.arange(cells.shape[0], dtype=np.int64), table.shape[0])
    _, inverse, counts = np.unique(
        np.sort(loops, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    once = counts[inverse] == 1
    twice = np.flatnonzero(counts[inverse] == 2)
    order = twice[np.argsort(inverse[twice], kind="stable")].reshape((-1, 2))
    different = regions[owner[order[:, 0]]] != regions[owner[order[:, 1]]]
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    boundary = _surface_topology(
        points,
        np.arange(0, 4 * (int(np.count_nonzero(once)) + 1), 4, dtype=np.int64),
        loops[once].reshape(-1),
    )
    interface = loops[order[different, 0]]
    adjacent = owner[order]
    cell_graph = coo_matrix(
        (
            np.ones(adjacent.shape[0], dtype=np.int8),
            (adjacent[:, 0], adjacent[:, 1]),
        ),
        shape=(cells.shape[0], cells.shape[0]),
    )
    solid_components = connected_components(
        cell_graph, directed=False, return_labels=False
    )
    return {
        "boundary": boundary,
        "interface_components": _face_components(
            np.arange(0, 4 * (interface.shape[0] + 1), 4, dtype=np.int64),
            interface.reshape(-1),
        )[0],
        "nonconforming_faces": int(np.count_nonzero(counts > 2)),
        "solid_components": int(solid_components),
    }


def _topology_acceptance(
    case: _FamilyInput, mesh: CellMesh, regions: np.ndarray
) -> dict[str, object]:
    """Compare published topology with the independently authored source topology."""
    authored = case.authored_topology
    if authored is None:
        raise RuntimeError("Nonsweepable obligation lacks its authored topology.")
    published = _published_topology(mesh, regions)
    construction = case.source_construction_proof
    nonsweepability_proved = (
        authored["nonsweepable_witness"]["witness"] is not None
        and isinstance(construction, dict)
        and construction.get("solid_connected") is True
        and published["solid_components"] == 1
    )
    if (
        not authored["boundary"]["closed_oriented_manifold"]
        or (case.nonsweepability_required and not nonsweepability_proved)
        or not published["boundary"]["closed_oriented_manifold"]
        or published["nonconforming_faces"] != 0
        or published["boundary"] != authored["boundary"]
        or published["interface_components"] != authored["interface_components"]
    ):
        raise RuntimeError(
            "Published all-hex topology disagrees with its authored topology or "
            f"required product-topology obstruction: authored={authored!r}, "
            f"published={published!r}."
        )
    return {
        "authored_topology_match": True,
        "nonsweepability_proved": nonsweepability_proved,
        "source_construction_proof": construction,
        "authored_topology": authored,
        "published_topology": published,
        "nonsweepable_witness": authored["nonsweepable_witness"]["witness"],
    }


def _authored_planar_boundary(
    vertices: np.ndarray, loops: tuple[tuple[int, ...], ...]
) -> dict[str, Any]:
    """Authored feature segments and loop orientation of a planar source."""
    areas = [
        0.5
        * float(
            np.sum(
                vertices[list(loop), 0] * np.roll(vertices[list(loop), 1], -1)
                - np.roll(vertices[list(loop), 0], -1) * vertices[list(loop), 1]
            )
        )
        for loop in loops
    ]
    return {
        "loops": len(loops),
        "outer_loops": sum(area > 0.0 for area in areas),
        "holes": sum(area < 0.0 for area in areas),
        "segments": [
            [
                [float(value) for value in vertices[loop[index]]],
                [float(value) for value in vertices[loop[(index + 1) % len(loop)]]],
            ]
            for loop in loops
            for index in range(len(loop))
        ],
    }


def _quad_feature_acceptance(case: _FamilyInput, mesh: CellMesh) -> dict[str, object]:
    """Recompute closed boundary loops and feature-segment coverage from published quads."""
    authored = case.authored_boundary
    if authored is None:
        raise RuntimeError("Feature-conforming quad obligation lacks authored loops.")
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    directed = np.stack(
        (cells.reshape(-1), np.roll(cells, -1, axis=1).reshape(-1)), axis=1
    )
    _, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    boundary = directed[counts[inverse] == 1]
    undirected = np.unique(np.sort(directed, axis=1), axis=0)
    vertex_count = int(np.asarray(mesh.coordinates).shape[0])
    valence = np.bincount(undirected.reshape(-1), minlength=vertex_count)
    degree = np.bincount(boundary.reshape(-1), minlength=vertex_count)
    on_boundary = degree > 0
    used = np.zeros((vertex_count,), dtype=np.bool_)
    used[cells.reshape(-1)] = True
    irregular = int(np.count_nonzero(used & ~on_boundary & (valence != 4)))
    graph = coo_matrix(
        (np.ones(boundary.shape[0]), (boundary[:, 0], boundary[:, 1])),
        shape=(vertex_count, vertex_count),
    )
    _, labels = connected_components(graph, directed=False)
    loop_labels = np.unique(labels[on_boundary])
    closed = bool(np.all(degree[on_boundary] == 2))
    record: dict[str, object] = {
        "closed_boundary_loops": closed,
        "boundary_loops": int(loop_labels.shape[0]),
        "irregular_interior_vertices": irregular,
    }
    passed = closed and loop_labels.shape[0] == authored["loops"]
    if "segments" in authored:
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        start, end = points[boundary[:, 0]], points[boundary[:, 1]]
        area = np.bincount(
            labels[boundary[:, 0]],
            weights=0.5 * (start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1]),
            minlength=int(labels.max(initial=-1)) + 1,
        )[loop_labels]
        # The loop areas sum to the signed domain area; normalize the cell orientation.
        area = area if float(np.sum(area)) > 0.0 else -area
        segments = np.asarray(authored["segments"], dtype=np.float64)
        origin, span = segments[:, 0], segments[:, 1] - segments[:, 0]
        lengths = np.linalg.norm(span, axis=1)
        scale = float(np.max(lengths))
        tolerance = 64.0 * np.finfo(np.float64).eps * scale

        def offsets(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            relative = values[:, None, :] - origin[None, :, :]
            cross = (
                span[None, :, 0] * relative[..., 1] - span[None, :, 1] * relative[..., 0]
            ) / lengths[None, :]
            along = np.einsum("psk,sk->ps", relative, span) / lengths[None, :] ** 2
            return cross, along

        cross_start, along_start = offsets(start)
        cross_end, along_end = offsets(end)
        slack = tolerance / scale
        on_segment = (
            (np.abs(cross_start) <= tolerance)
            & (np.abs(cross_end) <= tolerance)
            & (np.minimum(along_start, along_end) >= -slack)
            & (np.maximum(along_start, along_end) <= 1.0 + slack)
        )
        carried = on_segment.sum(axis=1) >= 1
        owner = np.argmax(on_segment, axis=1)
        covered = np.bincount(
            owner[carried],
            weights=np.linalg.norm(end - start, axis=1)[carried],
            minlength=segments.shape[0],
        )
        coverage_defect = float(np.max(np.abs(covered - lengths)))
        record.update(
            {
                "outer_loops": int(np.count_nonzero(area > 0.0)),
                "holes": int(np.count_nonzero(area < 0.0)),
                "boundary_edges_on_authored_features": bool(np.all(carried)),
                "maximum_feature_coverage_defect": coverage_defect,
            }
        )
        passed = (
            passed
            and bool(np.all(carried))
            and coverage_defect <= tolerance * segments.shape[0]
            and record["outer_loops"] == authored["outer_loops"]
            and record["holes"] == authored["holes"]
            and (authored["holes"] > 0 or irregular > 0)
        )
    if not passed:
        summary = {key: value for key, value in authored.items() if key != "segments"}
        raise RuntimeError(
            "Published quadrilaterals do not close onto the authored feature loops: "
            f"authored={summary!r}, published={record!r}."
        )
    return {**record, "authored_feature_match": True}


def _independent_regions(name: str, mesh: CellMesh) -> np.ndarray:
    """Donor centroid classification of authored material partitions."""
    centers = np.concatenate(
        [
            np.asarray(mesh.coordinates)[np.asarray(block.vertices)].mean(axis=1)
            for block in mesh.blocks
        ]
    )
    if name in ("dual-hex", "hex-dominant", "multiblock-material-hex"):
        return (centers[:, 0] > 1.0).astype(np.int64)
    if name == "curved-material-hex":
        return (centers[:, 0] > 2.0).astype(np.int64)
    return np.zeros(centers.shape[0], dtype=np.int64)


def _material_acceptance(
    name: str, case: _FamilyInput, result: Any, regions: np.ndarray
) -> dict[str, object]:
    """Donor zone/patch organization obligations for material profiles."""
    mesh = result.mesh
    record: dict[str, object] = {"status": "single-domain"}
    if name not in _MATERIAL_PROFILES:
        return record
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    zones = {zone.name: zone for zone in result.zones if zone.role is MeshZoneRole.REGION}
    region_names = tuple(case.expected_regions)
    if set(zones) != set(region_names):
        raise RuntimeError(f"Family publication lost a material: {sorted(zones)}.")
    for index, region in enumerate(region_names):
        zone = zones[region]
        if (
            zone.material_id != f"material-{region}"
            or zone.region_role is not RegionRole.SOLID
            or not np.array_equal(
                np.sort(np.asarray(zone.scope.entity_ids)),
                np.sort(ids[regions == index]),
            )
        ):
            raise RuntimeError(
                f"Material zone {region!r} does not own exactly its independently "
                "classified cells."
            )
    patches = {patch.name: patch for patch in result.patches}
    if name == "nonsweepable-dual-hex":
        cavity_names = _NONSWEEPABLE_PATCH_NAMES[1:]
        if not all(patch in patches for patch in cavity_names):
            raise RuntimeError(
                f"The two authored cavities must retain distinct patches: {sorted(patches)}."
            )
        first, second = (
            np.asarray(patches[patch].scope.entity_ids) for patch in cavity_names
        )
        if first.size == 0 or second.size == 0 or np.intersect1d(first, second).size:
            raise RuntimeError(
                "Distinct cavity patches must own disjoint, represented face sets."
            )
    if name in ("dual-hex", "hex-dominant") and not {
        "material-interface",
        "cavity-wall",
    } <= set(patches):
        raise RuntimeError(
            f"Family result lost its declared interface or cavity-wall patch: {sorted(patches)}."
        )
    if name == "multiblock-material-hex" and "material-interface" not in patches:
        raise RuntimeError("Reversed multiblock publication lost its material interface.")
    record = {
        "status": "exclusive-source-material-coverage",
        "material_ids": [zones[region].material_id for region in region_names],
        "patches": sorted(patches),
    }
    if name == "hex-dominant":
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolyhedralConnectivity):
            raise RuntimeError(
                "Hex-dominant publication must retain packed mixed-cell face incidence."
            )
        face_ids = np.asarray(mesh.entity_set(2).entity_ids)
        owners = np.asarray(connectivity.face_owner, dtype=np.int64)
        neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
        incidence = np.asarray(connectivity.face_cell_counts)
        if np.any((incidence != 1) & (incidence != 2)):
            raise RuntimeError(
                "Mixed transitions have nonconforming or nonmanifold face incidence."
            )
        internal = neighbors >= 0
        material_faces = internal & (regions[owners] != regions[np.maximum(neighbors, 0)])
        if not np.array_equal(
            np.sort(np.asarray(patches["material-interface"].scope.entity_ids)),
            np.sort(face_ids[material_faces]),
        ):
            raise RuntimeError(
                "Declared material interface does not cover exactly the conforming "
                "mixed-cell transition faces."
            )
        record["conforming_material_interface_face_count"] = int(
            np.count_nonzero(material_faces)
        )
    return record


def _mesh_edges(mesh: CellMesh) -> tuple[np.ndarray, list[tuple[int, Any, Any]]]:
    """Unique cell edges and one owning cell/reference segment per edge."""
    owners: dict[tuple[int, int], tuple[int, Any, Any]] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        reference = np.asarray(topology.vertices, dtype=np.float64)
        for identifier, row in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            for first, second in topology.entities[1]:
                a, b = int(row[first]), int(row[second])
                key = (min(a, b), max(a, b))
                if key not in owners:
                    start, end = (
                        (reference[first], reference[second])
                        if a < b
                        else (reference[second], reference[first])
                    )
                    owners[key] = (identifier, start, end)
    keys = sorted(owners)
    return np.asarray(keys, dtype=np.int64), [owners[key] for key in keys]


def _physical_edge_lengths(
    mesh: CellMesh, geometry: CellGeometrySpec, edges: np.ndarray, owners: list[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inscribed-polyline physical edge lengths through the published coordinate map.

    The polyline through mapped points is a strict lower bound of the arc
    length. The upper value adds the observed polyline refinement change, a
    Richardson-type margin for smooth maps; it is an estimate, not a proof.
    """
    parameters = np.linspace(0.0, 1.0, _EDGE_POLYLINE_SEGMENTS + 1)
    cells = np.asarray([owner[0] for owner in owners], dtype=np.int64)
    start = np.asarray([owner[1] for owner in owners], dtype=np.float64)
    end = np.asarray([owner[2] for owner in owners], dtype=np.float64)
    references = start[:, None, :] + parameters[None, :, None] * (end - start)[:, None, :]
    points = geometry.evaluate(mesh, cells, references)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    scale = float(np.max(np.abs(coordinates))) or 1.0
    if not np.allclose(
        points[:, 0], coordinates[edges[:, 0]], rtol=0.0, atol=1.0e-12 * scale
    ) or not np.allclose(
        points[:, -1], coordinates[edges[:, 1]], rtol=0.0, atol=1.0e-12 * scale
    ):
        raise RuntimeError("Published coordinate map does not retain its edge corners.")
    fine = np.sum(np.linalg.norm(np.diff(points, axis=1), axis=-1), axis=1)
    coarse = np.sum(np.linalg.norm(np.diff(points[:, ::2], axis=1), axis=-1), axis=1)
    return fine, fine, fine + np.abs(fine - coarse)


def _independent_quality(name: str, case: _FamilyInput, result: Any) -> dict[str, object]:
    """Donor frozen quality and hard size budget, recomputed from the mesh."""
    mesh = result.mesh
    values = phx.meshing.evaluate_cell_quality(mesh)
    minimum_scaled = float(np.min(np.asarray(values.scaled_jacobian)))
    minimum_mean = float(np.min(np.asarray(values.mean_ratios)))
    maximum_aspect = float(np.max(np.asarray(values.aspect_ratios)))
    sampled_valid = bool(np.all(np.asarray(values.sampled_valid)))
    edges, owners = _mesh_edges(mesh)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    lengths = np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
    edge_lower: np.ndarray | None = None
    edge_upper: np.ndarray | None = None
    if name in _PHYSICAL_EDGE_PROFILES:
        lengths, edge_lower, edge_upper = _physical_edge_lengths(
            mesh, result.geometry, edges, owners
        )
    specification = case.specification
    (control,) = specification.size_controls
    if not isinstance(control, UniformSizeControl) or control.maximum_size is None:
        raise RuntimeError("Family quality requires the authored hard uniform size.")
    target_size = float(control.target_size)
    maximum_size = float(control.maximum_size)
    tolerance = float(specification.size_compliance.relative_tolerance)
    minimum_angle = (
        specification.quality_target.minimum_angle
        if isinstance(specification, SurfaceMeshingSpec)
        and specification.quality_target is not None
        else None
    )
    median, maximum = float(np.median(lengths)), float(np.max(lengths))
    size_passed = abs(median - target_size) <= tolerance * target_size and (
        maximum <= maximum_size * (1.0 + tolerance)
    )
    if edge_lower is not None and edge_upper is not None:
        size_passed = size_passed and (
            float(np.median(edge_lower)) >= target_size * (1.0 - tolerance)
            and float(np.median(edge_upper)) <= target_size * (1.0 + tolerance)
            and float(np.max(edge_upper)) <= maximum_size * (1.0 + tolerance)
        )
    observed_angle = float(np.min(np.asarray(values.minimum_angle)))
    angle_passed = minimum_angle is None or observed_angle >= minimum_angle
    record: dict[str, object] = {
        "evaluator": "phydrax.meshing.evaluate_cell_quality(published vertices)",
        "minimum_scaled_jacobian": minimum_scaled,
        "minimum_mean_ratio": minimum_mean,
        "maximum_aspect_ratio": maximum_aspect,
        "sampled_valid": sampled_valid,
        "minimum_angle_radians": observed_angle,
        "required_minimum_angle_radians": minimum_angle,
        "edge_count": int(edges.shape[0]),
        "edge_length_basis": "physical inscribed polyline"
        if edge_lower is not None
        else "vertex chord",
        "median_edge": median,
        "maximum_edge": maximum,
        "target_size": target_size,
        "maximum_size": maximum_size,
        "size_relative_tolerance": tolerance,
        "size_passed": size_passed,
        "angle_passed": angle_passed,
    }
    if edge_lower is not None and edge_upper is not None:
        record.update(
            {
                "median_edge_lower": float(np.median(edge_lower)),
                "median_edge_upper_estimate": float(np.median(edge_upper)),
                "maximum_edge_upper_estimate": float(np.max(edge_upper)),
            }
        )
    if (
        not sampled_valid
        or not np.all(
            np.isfinite((minimum_scaled, minimum_mean, maximum_aspect, median, maximum))
        )
        or minimum_scaled < _QUALITY_LIMITS["minimum_scaled_jacobian"]
        or minimum_mean < _QUALITY_LIMITS["minimum_mean_ratio"]
        or maximum_aspect > _QUALITY_LIMITS["maximum_aspect_ratio"]
        or not size_passed
        or not angle_passed
    ):
        raise RuntimeError(
            f"Frozen independent family quality or hard size budget failed: {record!r}."
        )
    return record


def _accept_profile(
    profile_id: str, case: _FamilyInput, plan: Any, result: Any
) -> dict[str, object]:
    if plan.route != case.expected_route:
        raise RuntimeError(
            f"Native source/request resolved to route {plan.route!r}, expected exact profile route {case.expected_route!r}."
        )
    accepted = accepted_result_record(result)
    counts = _family_counts(result)
    policy = case.specification.target.cell_families
    required = set(policy.required)
    permitted = required | set(policy.preferred) | set(policy.allowed_transitions)
    if not required <= set(counts):
        raise RuntimeError(
            f"Required cell-family policy {sorted(required)} is absent from {counts}."
        )
    if policy.allow_mixed:
        if not set(counts) <= permitted:
            raise RuntimeError(
                f"Output contains a cell family outside the declared transition policy: {counts}."
            )
        if case.require_transition and not (
            set(counts) & set(policy.allowed_transitions)
        ):
            raise RuntimeError(
                "The hex-dominant source produced no declared transition cells."
            )
    elif set(counts) != required:
        raise RuntimeError(
            f"Pure-family request produced {counts}, expected only {sorted(required)}."
        )

    measures = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    if measures.size == 0 or not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
        raise RuntimeError(
            "Independent cell measures contain an empty, non-finite, or non-positive value."
        )
    measured_total = float(np.sum(measures, dtype=np.float64))
    relative_error = abs(measured_total - case.expected_measure) / case.expected_measure
    if relative_error > _PROFILE_TOLERANCE:
        raise RuntimeError(
            f"Independent physical measure {measured_total:.17g} differs from authored oracle {case.expected_measure:.17g} by relative error {relative_error:.17g}."
        )

    measured_regions = _measure_coverage(result)
    if case.require_region_coverage:
        if set(measured_regions) != set(case.expected_regions):
            raise RuntimeError(
                f"Independent region coverage names {sorted(measured_regions)}, expected {sorted(case.expected_regions)}."
            )
    region_errors: dict[str, float] = {}
    for region, expected in case.expected_regions.items():
        if region not in measured_regions:
            if case.require_region_coverage:
                raise RuntimeError(
                    f"Independent material measure for source region {region!r} is missing."
                )
            continue
        error = abs(measured_regions[region] - expected) / expected
        region_errors[region] = error
        if error > _PROFILE_TOLERANCE:
            raise RuntimeError(
                f"Source region {region!r} measure {measured_regions[region]:.17g} differs from its independent oracle {expected:.17g}."
            )
    if case.require_region_coverage and len(case.expected_regions) > 1:
        expected_materials = {f"material-{region}" for region in case.expected_regions}
        observed_materials = {
            zone.material_id for zone in result.zones if zone.material_id is not None
        }
        if observed_materials != expected_materials:
            raise RuntimeError(
                f"Material ancestry {sorted(observed_materials)} differs from expected {sorted(expected_materials)}."
            )

    associations = tuple(result.associations)
    if not associations or not any(item.complete for item in associations):
        raise RuntimeError(
            "Accepted native result omitted complete source ancestry associations."
        )
    source_id = getattr(case.source, "source_id", None)
    source_revision = getattr(case.source, "source_revision", None)
    matching_associations = tuple(
        item
        for item in associations
        if item.source_id == source_id and item.source_revision == source_revision
    )
    if not matching_associations or not any(
        item.complete for item in matching_associations
    ):
        raise RuntimeError(
            "Complete source associations do not retain this profile's source/revision identity."
        )
    source_roles = sorted(
        {
            getattr(role, "value", str(role))
            for association in matching_associations
            for role in association.source_roles
        }
    )

    quality = result.quality
    quality_limits = {**_QUALITY_LIMITS, **case.quality_limits}
    quality_record: dict[str, object] = {
        "minimum_measure": quality.minimum_measure,
        "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
        "minimum_mean_ratio": quality.minimum_mean_ratio,
        "maximum_aspect_ratio": quality.maximum_aspect_ratio,
        "sampled_invalid_count": quality.sampled_invalid_count,
        "hard_thresholds": quality_limits,
    }
    if (
        quality.sampled_invalid_count != 0
        or quality.minimum_measure <= 0.0
        or not np.all(
            np.isfinite(
                (
                    quality.minimum_measure,
                    quality.minimum_scaled_jacobian,
                    quality.minimum_mean_ratio,
                    quality.maximum_aspect_ratio,
                )
            )
        )
    ):
        raise RuntimeError(
            "Independent cell-quality report found invalid or non-positive cells."
        )
    if (
        "minimum_scaled_jacobian" in quality_limits
        and quality.minimum_scaled_jacobian < quality_limits["minimum_scaled_jacobian"]
    ):
        raise RuntimeError(
            "Minimum scaled Jacobian is below the authored hex-grid schedule."
        )
    if (
        "minimum_mean_ratio" in quality_limits
        and quality.minimum_mean_ratio < quality_limits["minimum_mean_ratio"]
    ):
        raise RuntimeError("Minimum mean ratio is below the authored hex-grid schedule.")
    if (
        "maximum_aspect_ratio" in quality_limits
        and quality.maximum_aspect_ratio > quality_limits["maximum_aspect_ratio"]
    ):
        raise RuntimeError("Maximum aspect ratio exceeds the authored hex-grid schedule.")
    if "minimum_angle" in quality_limits:
        minimum_angle = quality.minimum_angle
        if (
            minimum_angle is None
            or not np.isfinite(minimum_angle)
            or minimum_angle < quality_limits["minimum_angle"]
        ):
            raise RuntimeError(
                "Minimum angle is below the authored profile quality target."
            )
        quality_record["minimum_angle_radians"] = minimum_angle

    fidelity = _fidelity_record(case, result)
    if (
        case.expected_domain is not None
        and result.certification.inputs.domain is not case.expected_domain
    ):
        raise RuntimeError(
            "Native certification lost the exact authored source-domain object."
        )
    mapped_source_map = result.geometry.source_map
    if isinstance(case.source, NativeMappedHexSource):
        from phydrax.geometry._embedding_cells import prepare_embedding_cells
        from phydrax.geometry._restricted_embedding import (
            reference_chart_composition_equal,
        )

        domain = case.source.domain
        restriction = result.geometry.restriction_source
        if (
            (domain.source_id, domain.source_revision) != (source_id, source_revision)
            or result.certification.inputs.domain is not domain
            or restriction is None
            or restriction.source_geometry_id != cell_geometry_id(domain.source_geometry)
            or restriction.source_topology_id != domain.reference_mesh.topology_id
        ):
            raise RuntimeError(
                "Mapped output lost its authored domain or root restriction."
            )
        parents = {
            cell.global_id: cell
            for cell in prepare_embedding_cells(
                domain.reference_mesh, domain.source_geometry
            )
        }
        targets = {
            cell.global_id: cell
            for cell in prepare_embedding_cells(result.mesh, result.geometry)
        }
        visited: set[int] = set()
        for block_name, witness in restriction.block_parent_reference_maps.items():
            if witness is None or witness.source_reference_chart_arguments is None:
                raise RuntimeError("Mapped output omitted exact root chart arguments.")
            parent_ids = np.asarray(restriction.block_parent_cell_ids[block_name])
            if not np.array_equal(parent_ids, np.asarray(witness.source_cell_ids)):
                raise RuntimeError(
                    "Mapped output chart parent IDs disagree with its restriction."
                )
            for target_id, parent_id, arguments in zip(
                np.asarray(witness.target_cell_ids),
                parent_ids,
                witness.source_reference_chart_arguments,
                strict=True,
            ):
                target_id, parent_id = int(target_id), int(parent_id)
                if (
                    target_id in visited
                    or target_id not in targets
                    or parent_id not in parents
                    or reference_chart_composition_equal(
                        parents[parent_id], targets[target_id], tuple(arguments)
                    )
                    is not True
                ):
                    raise RuntimeError(
                        "Mapped output is not the exact root-map composition."
                    )
                visited.add(target_id)
        if visited != set(targets):
            raise RuntimeError(
                "Mapped output root charts do not cover every target cell."
            )
        source_map = domain.source_geometry.source_map
        if source_map is not None and (
            mapped_source_map is None
            or (mapped_source_map.source_id, mapped_source_map.source_revision)
            != (source_map.source_id, source_map.source_revision)
            or getattr(mapped_source_map, "source_geometry", None)
            is not domain.source_geometry
            or getattr(mapped_source_map, "source_mesh", None)
            is not domain.reference_mesh
        ):
            raise RuntimeError("Mapped output lost its authored coordinate source map.")
    elif isinstance(case.source, NativeSweepSource) and isinstance(
        case.source.domain, MappedReferenceDomain
    ):
        from phydrax.discretization._coordinate_enclosure import axes
        from phydrax.geometry._embedding_cells import prepare_embedding_cells
        from phydrax.geometry._restricted_embedding import (
            reference_chart_composition_equal,
        )

        domain = case.source.domain
        restriction = result.geometry.restriction_source
        if (
            restriction is None
            or restriction.source_geometry_id != cell_geometry_id(domain.source_geometry)
            or restriction.source_topology_id != domain.reference_mesh.topology_id
            or result.certification.inputs.domain is not domain
        ):
            raise RuntimeError("Mapped sweep lost its original root geometry or domain.")
        parents = {
            cell.global_id: cell
            for cell in prepare_embedding_cells(
                domain.reference_mesh, domain.source_geometry
            )
        }
        targets = {
            cell.global_id: cell
            for cell in prepare_embedding_cells(result.mesh, result.geometry)
        }
        visited: set[int] = set()
        for block in result.mesh.blocks:
            parent_ids = np.asarray(restriction.block_parent_cell_ids[block.name])
            for target_id, parent_id in zip(block.global_ids, parent_ids, strict=True):
                target_id, parent_id = int(target_id), int(parent_id)
                if (
                    target_id in visited
                    or target_id not in targets
                    or parent_id not in parents
                    or reference_chart_composition_equal(
                        parents[parent_id], targets[target_id], axes(3)
                    )
                    is not True
                ):
                    raise RuntimeError(
                        "Mapped sweep is not the exact authored root-map composition."
                    )
                visited.add(target_id)
        if visited != set(targets):
            raise RuntimeError("Mapped sweep root maps omit target cells.")
        source_map = domain.source_geometry.source_map
        if source_map is not None and (
            mapped_source_map is None
            or (mapped_source_map.source_id, mapped_source_map.source_revision)
            != (source_map.source_id, source_map.source_revision)
        ):
            raise RuntimeError("Mapped sweep lost its authored coordinate source map.")
    family_name = profile_id.removeprefix(_PROFILE_PREFIX).removesuffix(":larger")
    certification = result.certification
    if (
        certification is None
        or not certification.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise RuntimeError("Family publication lacks strict independent certification.")
    if family_name in _STRICT_ASSOCIATION_PROFILES and any(
        (item.source_id, item.source_revision) != (source_id, source_revision)
        for item in associations
    ):
        raise RuntimeError("Family result lost its source/revision associations.")
    regions = _independent_regions(family_name, result.mesh)
    published_regions = certification.inputs.cell_regions
    if published_regions is not None and not np.array_equal(
        regions, np.asarray(published_regions)
    ):
        raise RuntimeError(
            "Published cell-material assignments disagree with independent geometry."
        )
    material_record = _material_acceptance(family_name, case, result, regions)
    if isinstance(case.source, NativeMappedHexSource):
        restriction = result.geometry.restriction_source
        parents = {
            int(target): int(parent)
            for block in result.mesh.blocks
            for target, parent in zip(
                np.asarray(block.global_ids).tolist(),
                np.asarray(restriction.block_parent_cell_ids[block.name]).tolist(),
                strict=True,
            )
        }
        region_zones = {
            zone.name: zone for zone in result.zones if zone.role is MeshZoneRole.REGION
        }
        equality = case.physical_input["region_scope_root_cell_equality"]
        for entry in cast(list[dict[str, Any]], equality):
            zone = region_zones.get(entry["region"])
            if (
                zone is None
                or zone.material_id != entry["material_id"]
                or {parents[int(cell)] for cell in np.asarray(zone.scope.entity_ids)}
                != set(entry["donor_root_cell_ids"])
            ):
                raise RuntimeError(
                    f"Published region {entry['region']!r} does not cover exactly the "
                    "donor root-cell scope with its donor material."
                )
        material_record["donor_root_cell_scope_equality"] = equality
    independent_quality = _independent_quality(family_name, case, result)
    obligation = _obligation_class(case)
    class_record: dict[str, object] = {}
    if obligation == "general-nonsweepable-all-hex":
        class_record["topology"] = _topology_acceptance(case, result.mesh, regions)
    elif obligation == "feature-conforming-quad":
        class_record["feature_topology"] = _quad_feature_acceptance(case, result.mesh)

    return {
        "status": "passed",
        "profile_id": profile_id,
        "expected_admission": "positive",
        "route": plan.route,
        "plan_id": plan.plan_id,
        "source_provenance": _source_record(case.source),
        "obligation_class": obligation,
        **class_record,
        "material_organization": material_record,
        "independent_quality": independent_quality,
        "physical_domain_oracle": {
            "independent_total_measure": case.expected_measure,
            "measured_total_measure": measured_total,
            "relative_error": relative_error,
            "tolerance_relative": _PROFILE_TOLERANCE,
            "independent_region_measures": dict(case.expected_regions),
            "measured_region_measures": measured_regions,
            "region_relative_errors": region_errors,
        },
        "family_inventory": counts,
        "quality_oracle": quality_record,
        "ancestry": {
            "association_count": len(associations),
            "complete_association_count": sum(
                bool(item.complete) for item in associations
            ),
            "matching_source_association_count": len(matching_associations),
            "source_roles": source_roles,
        },
        "source_fidelity": fidelity,
        "accepted_native_evidence": accepted,
    }


def _run_profile(
    profile_id: str,
    *,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    capture_result: Callable[[str, Any], None] | None = None,
) -> dict[str, object]:
    phase = "source-and-request-construction"

    def report_phase() -> None:
        if checkpoint is not None:
            checkpoint({"profile_id": profile_id, "phase": phase, "event": "phase-start"})

    report_phase()
    case: _FamilyInput | None = None
    scenario_name = profile_id.removeprefix(_PROFILE_PREFIX)
    larger = scenario_name.endswith(":larger")
    family_case = scenario_name.removesuffix(":larger")
    resolution_scale = 2 if larger else 1
    resolved_resolution = _BASE_RESOLUTION * resolution_scale
    source_provenance: dict[str, object] = {
        "migration_registry": "tools/meshing_qualification/native_case_migration.json#general-quads-hexes",
        "profile_identity": profile_id,
        "fixture_origin": "donor native-family-lifecycle source registry; ported onto current native public APIs",
        "api_contract_sources": (
            "tests/unit/meshing/test_native_general_families.py",
            "tests/unit/meshing/test_native_structured_routes.py",
        ),
    }
    run_record: dict[str, object] = {
        "profile_id": profile_id,
        "expected_admission": "positive",
        "source_provenance": source_provenance,
        "accepted_native_evidence": None,
        "profile_controls": {
            "scenario_controls": {"family_case": family_case},
            "resolution_scale": resolution_scale,
            "base_resolution": _BASE_RESOLUTION,
            "resolved_resolution": resolved_resolution,
            "size_target_m": 1.0 / resolved_resolution,
            "maximum_size_m": 2.0 / resolved_resolution,
            "size_strength": "hard",
            "size_compliance": {
                "relative_tolerance": 0.95,
                "target_statistics": ("p50",),
            },
        },
    }
    try:
        case = _make_profile(profile_id)
        source_provenance.update(_source_record(case.source))
        run_record["source_input"] = case.physical_input
        run_record["request"] = _request_record(case)
        run_record["oracles_and_tolerances"] = {
            "independent_physical_measure": case.expected_measure,
            "independent_region_measures": dict(case.expected_regions),
            "region_measure_relative_tolerance": _PROFILE_TOLERANCE,
            "source_fidelity_maximum_deviation_m": case.fidelity_tolerance,
            "cell_quality_hard_limits": {**_QUALITY_LIMITS, **case.quality_limits},
            "family_policy": "exact required family; only explicitly named hex-dominant transitions are permitted",
        }
        phase = "native-plan"
        report_phase()
        plan = NativeMeshingProvider(case.options).plan(
            case.source,
            case.specification,
            coordinate_contract=_CONTRACT,
        )
        run_record["plan_route"] = plan.route
        phase = "native-execute"
        report_phase()
        result = plan.execute()
        run_record["native_result"] = {
            "result_id": result.result_id,
            "topology_id": result.mesh.topology_id,
            "provider": result.provider.name,
            "audit_passed": result.audit.passed,
            "certification_passed": result.certification is not None
            and result.certification.passed,
            "compliance_passed": result.compliance.passed,
        }
        if capture_result is not None:
            phase = "capture-native-result"
            report_phase()
            capture_result(profile_id, result)
        phase = "independent-acceptance"
        report_phase()
        accepted = _accept_profile(profile_id, case, plan, result)
        family_name = profile_id.removeprefix(_PROFILE_PREFIX).removesuffix(":larger")
        if family_name == "curved-quarter-extrusion-quad":
            phase = "physical-fe-quadratic"
            report_phase()
            accepted["physical_fe"] = _solve_quarter_quadratic(result)[0]
        if family_name in _FE_PROFILES and not profile_id.endswith(":larger"):
            phase = "physical-fe-solve"
            report_phase()
            accepted["physical_fe"] = _solve_family_affine_field(result)
        run_record.update(accepted)
        run_record["source_input"] = case.physical_input
        run_record["request"] = _request_record(case)
        return run_record
    except Exception as error:
        category = getattr(error, "category", None)
        run_record.update(
            {
                "status": "failed",
                "failure": {
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "error_category": getattr(category, "value", category),
                    "api_or_oracle_error": str(error),
                },
            }
        )
        if case is not None:
            source_provenance.update(_source_record(case.source))
            run_record["source_input"] = case.physical_input
            run_record["request"] = _request_record(case)
            run_record["oracles_and_tolerances"] = {
                "independent_physical_measure": case.expected_measure,
                "independent_region_measures": dict(case.expected_regions),
                "region_measure_relative_tolerance": _PROFILE_TOLERANCE,
                "source_fidelity_maximum_deviation_m": case.fidelity_tolerance,
                "cell_quality_hard_limits": {**_QUALITY_LIMITS, **case.quality_limits},
                "family_policy": "exact required family; only explicitly named hex-dominant transitions are permitted",
            }
        return run_record


def native_general_families(
    *,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    capture_result: Callable[[str, Any], None] | None = None,
) -> dict[str, object]:
    """Run every positive native general-quad/hex source profile as an independent gate."""
    profile_ids = _profile_ids()
    records: dict[str, dict[str, object]] = {}
    for profile_id in profile_ids:
        record = _run_profile(
            profile_id, checkpoint=checkpoint, capture_result=capture_result
        )
        records[profile_id] = record
        if checkpoint is not None:
            checkpoint(
                {"profile_id": profile_id, "event": "profile-complete", "record": record}
            )
    runtime = runtime_identity()
    return {
        "status": "passed"
        if all(record["status"] == "passed" for record in records.values())
        else "failed",
        "workflow": "all original native general quadrilateral and hexahedron source profiles",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#general-quads-hexes",
        "expected_admission": "positive",
        "mandatory_profile_count": len(profile_ids),
        "passed_profile_count": sum(
            record["status"] == "passed" for record in records.values()
        ),
        "failed_profile_count": sum(
            record["status"] != "passed" for record in records.values()
        ),
        "profile_run_records": records,
        "runtime": runtime,
        "meshcore_identity": runtime["meshcore"],
    }
