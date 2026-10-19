#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Request-scoped surface metric sources and physical reaction preparation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from math import inf
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax._differentiation import BranchDifferentiationPolicy
from phydrax._fingerprint import canonical_fingerprint
from phydrax._frozendict import frozendict
from phydrax._meshcore import (
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax._strict import StrictModule
from phydrax._validation import positive_finite_float
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import SplineCellGeometryElement
from phydrax.discretization._cell_geometry_transfer import (
    reconstruct_parametric_surface_cell_geometry,
)
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransitionPolicy,
)
from phydrax.discretization._chart_reconstruction import SurfaceGeometryReconstruction
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    Expression,
    RationalPolynomial,
)
from phydrax.discretization._surface_chart_records import PreparedSurfaceChartDeformation
from phydrax.discretization._views import FieldTracePolicy
from phydrax.discretization.fem import FiniteElementDiscretization
from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
from phydrax.discretization.fem._surface_chart_transfer import (
    prepare_surface_chart_field_transfer,
)
from phydrax.discretization.fem._topology_transfer import FiniteElementFieldTransfer
from phydrax.discretization.finite_volume import (
    prepare_unstructured_conservative_remap,
    UnstructuredConservativeRemapPlan,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._remap_evidence import (
    MappedSurfaceChartRemapEvidence,
    PreparedUnstructuredConservativeRemap,
)
from phydrax.equations import CompiledFiniteElementProblem
from phydrax.geometry import MeshingDomain
from phydrax.geometry._sphere_material_atlas import SphereMaterialCellAtlas
from phydrax.geometry._surface_source_support import (
    prepare_surface_source_root_atlas,
    PreparedSurfaceSourceSupport,
    SurfaceSourceCharts,
    SurfaceSourceRootAtlas,
)
from phydrax.lifecycle._meshing_field_records import (
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from phydrax.linalg._spaces import ArraySpace
from phydrax.meshing._adaptation import MeshAdaptationPolicy, MeshAdaptationRoute
from phydrax.meshing._association import SurfaceAssociationTransfer
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._mesh_part import MeshPart
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._surface_metric import prepare_surface_metric_source
from phydrax.meshing.qualification import AnalyticSurface
from phydrax.solver import MaterialTopologyTransferResult
from tests._support.sphere_material import accepted_sphere_material


# Locked empirical acceptance constants from the pre-remediation
# tests/unit/meshing/test_surface_metric.py scenarios: analytic sphere/torus
# chart evaluation, rational curve residual reconstruction, coordinate-map
# reconstruction, material field actions and physical reaction residuals.
# These names preserve their recorded values; authored geometry bounds remain
# the domain's tolerance or the request's fidelity bound.
GEOMETRY_EVALUATION_ATOL = 1.0e-12
SOURCE_RESIDUAL_RTOL = 1.0e-12
SOURCE_RESIDUAL_ATOL = 1.0e-14
COORDINATE_RECONSTRUCTION_RTOL = 1.0e-13
METRIC_ARC_RTOL = 1.0e-10
MATERIAL_FIELD_RTOL = 1.0e-12
MATERIAL_FIELD_ATOL = 1.0e-12
REACTION_RESIDUAL_RTOL = 1.0e-8
REACTION_RESIDUAL_ATOL = 1.0e-10


def rational_patch_point(
    controls: np.ndarray, weights: np.ndarray, u: Fraction, v: Fraction
) -> tuple[Fraction, ...]:
    """Evaluate the authored degree-one rational tensor patch in exact arithmetic."""
    basis = ((1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v)
    weighted = tuple(
        value * Fraction(float(weight))
        for value, weight in zip(basis, weights.flat, strict=True)
    )
    denominator = sum(weighted, Fraction(0))
    return tuple(
        sum(
            (
                weight * Fraction(float(point[axis]))
                for weight, point in zip(weighted, controls.reshape((4, 3)), strict=True)
            ),
            Fraction(0),
        )
        / denominator
        for axis in range(3)
    )


def rational_trim_point(
    domain: MeshingDomain, curve: int, parameter: float
) -> np.ndarray:
    """Evaluate the authored quadratic Bezier trim and bilinear patch, without atlases."""
    patch, loop, position = domain.curve_owners[curve]
    use = domain.patches[patch].loops[loop][position]
    if not isinstance(use, phx.geometry.PatchCurveUse) or not isinstance(
        use.pcurve, phx.geometry.BSplineCurve
    ):
        raise TypeError(
            "rational_reference.curve must be an authored Bezier curve; got an incompatible boundary use"
        )
    surface = domain.patches[patch].surface
    if not isinstance(surface, phx.geometry.BSplineSurfacePatch):
        raise TypeError(
            "rational_reference.surface must be an authored bilinear patch; got an incompatible surface"
        )
    t = Fraction(parameter)
    basis = ((1 - t) ** 2, 2 * t * (1 - t), t**2)
    weighted = tuple(
        value * Fraction(float(weight))
        for value, weight in zip(basis, np.asarray(use.pcurve.weights), strict=True)
    )
    uv = tuple(
        sum(
            (
                weight * Fraction(float(point[axis]))
                for weight, point in zip(
                    weighted, np.asarray(use.pcurve.control_points), strict=True
                )
            ),
            Fraction(0),
        )
        / sum(weighted, Fraction(0))
        for axis in range(2)
    )
    return np.asarray(
        rational_patch_point(
            np.asarray(surface.control_points), np.asarray(surface.weights), *uv
        ),
        dtype=np.float64,
    )


def require_exact_rational_patch_map(
    expressions: tuple[Expression, ...],
    charts: np.ndarray,
    controls: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Prove quotient identity on a degree-sufficient rational tensor grid.

    Affine UV arguments give each authored bilinear numerator and denominator
    degree at most two per reference axis. Cross multiplying a degree-d represented
    quotient therefore gives degree at most d+2. The d+3 distinct grid nodes
    per axis prove the whole polynomial is zero, not merely sampled agreement.
    """
    exact_charts = tuple(tuple(Fraction(float(value)) for value in row) for row in charts)
    for axis, expression in enumerate(expressions):
        if isinstance(expression, RationalPolynomial):
            numerator, denominator = expression.numerator, expression.denominator
        else:
            numerator, denominator = expression, {(0, 0): Fraction(1)}
        degrees = tuple(
            max((index[dimension] for index in (*numerator, *denominator)), default=0) + 2
            for dimension in range(2)
        )
        for i in range(degrees[0] + 1):
            for j in range(degrees[1] + 1):
                point = (Fraction(i, 4 * degrees[0]), Fraction(j, 4 * degrees[1]))
                uv = tuple(
                    exact_charts[0][dimension]
                    + point[0] * (exact_charts[1][dimension] - exact_charts[0][dimension])
                    + point[1] * (exact_charts[2][dimension] - exact_charts[0][dimension])
                    for dimension in range(2)
                )
                top = sum(
                    (
                        coefficient * point[0] ** index[0] * point[1] ** index[1]
                        for index, coefficient in numerator.items()
                    ),
                    Fraction(0),
                )
                bottom = sum(
                    (
                        coefficient * point[0] ** index[0] * point[1] ** index[1]
                        for index, coefficient in denominator.items()
                    ),
                    Fraction(0),
                )
                expected = rational_patch_point(controls, weights, *uv)[axis]
                assert bottom != 0 and top == bottom * expected


def material_triangle_weights(
    triangles: np.ndarray, point: np.ndarray
) -> tuple[int, np.ndarray]:
    """Select the first authored cover and solve its affine barycentric weights."""
    for row, triangle in enumerate(triangles):
        barycentric = np.linalg.solve(
            np.vstack((triangle.T, np.ones(3, dtype=np.float64))),
            np.asarray((*point, 1.0), dtype=np.float64),
        )
        if np.min(barycentric) < -64 * np.finfo(np.float64).eps:
            continue
        return row, barycentric
    raise ValueError(
        "material_reference.coverage must contain every target node; got an uncovered target node"
    )


def material_p1_reference_matrix(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    chart: PreparedSurfaceChartDeformation,
) -> np.ndarray:
    """Use host barycentric solves on material triangles, never transfer rows."""
    source_routes = np.concatenate(
        [np.asarray(rows) for rows in source.dof_maps[0].cell_dofs]
    )
    target_routes = np.concatenate(
        [np.asarray(rows) for rows in target.dof_maps[0].cell_dofs]
    )
    matrix = np.zeros(
        (target.dof_maps[0].global_dof_count, source.dof_maps[0].global_dof_count),
        dtype=np.float64,
    )
    assigned = np.zeros(matrix.shape[0], dtype=np.bool_)
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in source.mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in target.mesh.blocks]
    )
    chart_source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in chart.source_mesh.blocks]
    )
    chart_target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in chart.target_mesh.blocks]
    )
    for occurrence in chart.occurrences:
        old_triangles = np.asarray(
            occurrence.exact_source_material_charts, dtype=np.float64
        )
        new_triangles = np.asarray(
            occurrence.exact_target_material_charts, dtype=np.float64
        )
        for local_row, physical_row in enumerate(np.asarray(occurrence.target_rows)):
            target_row = int(
                np.flatnonzero(target_ids == chart_target_ids[physical_row])[0]
            )
            for slot, point in enumerate(new_triangles[local_row]):
                dof = target_routes[target_row, slot]
                old_local, barycentric = material_triangle_weights(old_triangles, point)
                old_physical = np.asarray(occurrence.source_rows)[old_local]
                old_row = int(
                    np.flatnonzero(source_ids == chart_source_ids[old_physical])[0]
                )
                candidate = np.zeros(matrix.shape[1], dtype=np.float64)
                candidate[source_routes[old_row]] = barycentric
                if assigned[dof]:
                    np.testing.assert_allclose(
                        matrix[dof],
                        candidate,
                        rtol=MATERIAL_FIELD_RTOL,
                        atol=MATERIAL_FIELD_ATOL,
                    )
                else:
                    matrix[dof] = candidate
                    assigned[dof] = True
    assert np.all(assigned)
    return matrix


@contextmanager
def host_coordinate_phase(
    budget: CoordinateEnclosureBudget,
) -> Iterator[None]:
    if not budget.ticket.settled:
        budget.ticket.settle()
    with budget.retained_storage(lambda size: HostBatchTicket(0, bytes_upper=size)):
        with budget.bind_ticket(
            HostBatchTicket(
                budget.maximum_work_units - budget.work_units,
                bytes_upper=budget.maximum_memory_bytes
                - budget.retained_basis_bytes
                - budget.temporary_bytes_upper,
            )
        ):
            yield


@contextmanager
def native_coordinate_phase(
    scope: NativeExecutionScope, budget: CoordinateEnclosureBudget
) -> Iterator[None]:
    if not budget.ticket.settled:
        budget.ticket.settle()
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        remaining = scope.remaining()
        with budget.bind_ticket(
            scope.batch(
                min(
                    budget.maximum_work_units - budget.work_units,
                    remaining.remaining_work_units,
                ),
                queries_upper=remaining.remaining_geometry_queries,
                cavity_upper=remaining.remaining_cavity_cells,
                bytes_upper=min(
                    budget.maximum_memory_bytes
                    - budget.retained_basis_bytes
                    - budget.temporary_bytes_upper,
                    remaining.remaining_managed_bytes,
                ),
            )
        ):
            with budget.ticket.native_target():
                yield


@contextmanager
def native_coordinate_resources(
    scope: NativeExecutionScope, limits: MeshingLimits
) -> Iterator[CoordinateEnclosureBudget]:
    budget = CoordinateEnclosureBudget(
        HostBatchTicket(0),
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )
    with native_coordinate_phase(scope, budget):
        yield budget


@contextmanager
def surface_resources() -> Iterator[
    tuple[NativeExecutionScope, CoordinateEnclosureBudget]
]:
    unlimited = (1 << 64) - 1
    with NativeExecutionScope(
        NativeExecutionLimits(unlimited, unlimited, unlimited, unlimited, inf)
    ) as scope:
        budget = CoordinateEnclosureBudget(
            HostBatchTicket(0),
            maximum_work_units=unlimited,
            maximum_memory_bytes=unlimited,
        )
        with native_coordinate_phase(scope, budget):
            yield scope, budget


def surface_patch(surface: AnalyticSurface) -> tuple[CellMesh, np.ndarray, np.ndarray]:
    chart_points = np.asarray(
        ((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)), dtype=np.float64
    )
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    points = surface.domain.evaluate(np.zeros((4,), dtype=np.int32), chart_points)
    return (
        CellMesh.from_triangles(points, cells),
        np.zeros((2,), dtype=np.int32),
        chart_points[cells],
    )


class SphereMetricSource(NamedTuple):
    domain: MeshingDomain
    source: CellMeshingResult
    atlas: SphereMaterialCellAtlas
    transfer: SurfaceAssociationTransfer
    policy: MeshAdaptationPolicy


def prepare_sphere_metric_source(size: float) -> SphereMetricSource:
    """Retain native corner charts, not a manufactured latitude atlas."""
    domain, source, atlas = accepted_sphere_material(size, reuse_preparation=False)
    retained = source.surface_source
    if not isinstance(retained, SurfaceSourceCharts):
        raise TypeError(
            "surface_metric.fixture.source_charts must retain the producer chart data; got a sphere result without source charts"
        )
    retained.require_root(source.mesh, source.geometry)
    charts, boundary = retained.root_parameters, retained.boundary_source
    transfer = SurfaceAssociationTransfer(
        PreparedSurfaceSourceSupport(
            domain,
            source,
            jnp.asarray(charts),
            boundary,
            maximum_support_queries=1_280_000,
        )
    )
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=640_000,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=4,
        relocation=False,
        limits=limits,
        geometry_transition=CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation",
            reconstruction_tolerance=0.32,
            maximum_evaluations=640_000,
        ),
    )
    return SphereMetricSource(domain, source, atlas, transfer, policy)


def sphere_native_scope(limits: MeshingLimits) -> NativeExecutionScope:
    return NativeExecutionScope(limits.native_limits())


class UVMetricSource(NamedTuple):
    domain: MeshingDomain
    source: CellMeshingResult
    transfer: SurfaceAssociationTransfer
    policy: MeshAdaptationPolicy


def prepare_rational_trim_metric_source() -> UVMetricSource:
    """Prepare a rational patch and trim under the fixture request capacities."""
    geometry, meshing = phx.geometry, phx.meshing
    controls = np.asarray(
        (
            ((0.75, 0.5), (0.75, 0.75), (0.5, 0.75)),
            ((0.5, 0.75), (0.25, 0.75), (0.25, 0.5)),
            ((0.25, 0.5), (0.25, 0.25), (0.5, 0.25)),
            ((0.5, 0.25), (0.75, 0.25), (0.75, 0.5)),
        ),
        dtype=np.float64,
    )
    loop = tuple(
        geometry.PatchCurveUse(
            index,
            geometry.BSplineCurve.bezier(points, (1.0, 0.75, 1.0)),
            0.0,
            1.0,
        )
        for index, points in enumerate(controls)
    )
    patch = geometry.BSplineSurfacePatch(
        np.asarray(
            (((0.0, 0.0, 0.0), (0.0, 1.0, 0.05)), ((1.0, 0.0, 0.1), (1.0, 1.0, 0.2))),
            dtype=np.float64,
        ),
        np.asarray(((1.0, 0.9), (1.1, 1.0)), dtype=np.float64),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        1,
    )
    domain = geometry.MeshingDomain(
        (geometry.MeshingSurfacePatch(patch, (loop,)),),
        tuple(geometry.MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="native-rational-trim-metric",
        source_revision="authored",
    )
    scope = meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    limits = MeshingLimits(
        maximum_vertices=20_000,
        maximum_edges=160_000,
        maximum_faces=160_000,
        maximum_cells=160_000,
        maximum_work_units=640_000,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(2, 3, meshing.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            meshing.UniformSizeControl(
                scope, 0.16, maximum_size=0.24, strength=meshing.SizeControlStrength.HARD
            ),
        ),
        size_compliance=meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=(
            meshing.ProtectedFeature(
                scope, meshing.FeatureKind.SURFACE, maximum_deviation=0.01
            ),
        ),
        limits=limits,
    )
    source = (
        meshing.NativeMeshingProvider(meshing.NativeMeshingOptions("parametric_surface"))
        .plan(
            meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    if not isinstance(source, CellMeshingResult):
        raise TypeError(
            "surface_metric.fixture must retain accepted source geometry and field preparation; got the authored rational surface plan must produce its cell result"
        )
    retained = source.surface_source
    if not isinstance(retained, SurfaceSourceCharts):
        raise TypeError(
            "surface_metric.fixture.source_charts must retain the producer chart data; got a rational surface result without source charts"
        )
    retained.require_root(source.mesh, source.geometry)
    boundary = retained.boundary_source
    atlas = prepare_surface_source_root_atlas(
        domain,
        boundary.patches,
        source.coordinate_contract,
        maximum_support_queries=limits.maximum_geometry_queries,
    )
    transfer = SurfaceAssociationTransfer(
        PreparedSurfaceSourceSupport(
            domain,
            atlas,
            atlas.root_parameters,
            boundary,
            maximum_support_queries=limits.maximum_geometry_queries,
        )
    )
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        association_transfer=transfer,
        maximum_passes=4,
        relocation=False,
        limits=limits,
        geometry_transition=CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation",
            reconstruction_tolerance=0.02,
            maximum_evaluations=640_000,
        ),
    )
    return UVMetricSource(domain, source, transfer, policy)


def surface_reaction_load(points: Array, context: object) -> Array:
    del context
    return 1.0 + points[..., 0] ** 2


def solve_surface_reaction(
    prepared: FiniteElementDiscretization,
    seed: Array | None = None,
) -> tuple[CompiledFiniteElementProblem, Array]:
    form = phx.equations.FiniteElementForm(
        "surface-reaction",
        "temperature",
        (
            phx.equations.MassAction("temperature", 1.0),
            phx.equations.SourceAction(
                "temperature",
                phx.equations.coefficient(
                    surface_reaction_load, coefficient_id="ambient-temperature-reaction"
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, prepared)
    system, raw_rhs = problem.linear_system()
    space = prepared.field_spaces[0].vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError(
            "surface_metric.fixture must prepare one array coefficient space for temperature; got the single temperature field requires its array coefficient space"
        )
    rhs = space.validate(raw_rhs)
    initial = space.validate(seed) if seed is not None else jnp.zeros_like(rhs)
    solved = phx.linalg.solve(system, rhs - space.validate(system.operator.mv(initial)))
    if not bool(solved.successful):
        raise ValueError(
            "surface_metric.fixture must report a successful native reaction solve; got the physical surface reaction solve did not succeed"
        )
    value = initial + space.validate(solved.value)
    correction = phx.linalg.solve(system, rhs - space.validate(system.operator.mv(value)))
    if not bool(correction.successful):
        raise ValueError(
            "surface_metric.fixture must report a successful native reaction solve; got the physical surface reaction refinement did not succeed"
        )
    value += space.validate(correction.value)
    np.testing.assert_allclose(
        np.asarray(space.validate(system.operator.mv(value))),
        np.asarray(rhs),
        rtol=REACTION_RESIDUAL_RTOL,
        atol=REACTION_RESIDUAL_ATOL,
    )
    return problem, space.validate(problem.expand(value))


class RationalMetricScenario(NamedTuple):
    source_fixture: UVMetricSource
    before: np.ndarray
    field: phx.discretization.FiniteElementFieldSpec
    source_space: FiniteElementDiscretization
    source_problem: CompiledFiniteElementProblem
    source_temperature: Array
    adaptation: phx.meshing.MeshAdaptationResult
    target: CellMeshingResult
    chart: PreparedSurfaceChartDeformation


class RationalMetricFieldState(NamedTuple):
    scenario: RationalMetricScenario
    old_fv: UnstructuredFiniteVolumeDiscretization
    new_fv: UnstructuredFiniteVolumeDiscretization
    remap: PreparedUnstructuredConservativeRemap
    remap_plan: UnstructuredConservativeRemapPlan
    history: Array
    source_material: phx.equations.MaterialTransaction
    carried_history: Array
    bound: float
    prepared: FiniteElementDiscretization
    transfer: FiniteElementFieldTransfer
    carried_temperature: Array
    problem: CompiledFiniteElementProblem
    temperature: Array
    material: phx.equations.MaterialTransaction


class RationalMetricArchive(NamedTuple):
    fields: RationalMetricFieldState
    part: MeshPart
    carrier: CellMeshingResult
    declaration: MeshingFiniteElementFieldDeclaration
    adaptation_event: dict[str, Any]
    archive_records: dict[str, Any]


def prepare_rational_metric_scenario(
    source_fixture: UVMetricSource,
    *,
    metric_size: float = 0.14,
) -> RationalMetricScenario:
    """Prepare scalar-field metric adaptation at the requested positive size."""
    metric_size = positive_finite_float(metric_size, "surface_metric.fixture.metric_size")
    domain, source, _, policy = source_fixture
    before = np.asarray(source.geometry.coordinates).copy()
    field = phx.discretization.FiniteElementFieldSpec(
        "temperature", phx.discretization.lagrange_element("triangle", 1)
    )
    source_space = phx.discretization.FiniteElementPlan(
        source.mesh, field, coordinate_spec=source.geometry
    ).prepare()
    source_problem, source_temperature = solve_surface_reaction(source_space)
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) / metric_size**2,
            (source.mesh.coordinates.shape[0], 3, 3),
        ),
        minimum_size=metric_size,
        maximum_size=metric_size,
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=source.coordinate_contract
            ),
            policy=policy,
        )
    )
    if not isinstance(adaptation.target, CellMeshingResult):
        raise TypeError(
            "surface_metric.fixture.target must be a cell result; got incompatible carrier"
        )
    target = adaptation.target
    if adaptation.geometry_transition is None:
        raise TypeError(
            "surface_metric.fixture.geometry_transition must retain a chart deformation; got None"
        )
    chart = adaptation.geometry_transition.chart_deformation
    if not isinstance(chart, PreparedSurfaceChartDeformation):
        raise TypeError(
            "surface_metric.fixture.chart must be a UV deformation; got incompatible map"
        )
    return RationalMetricScenario(
        source_fixture,
        before,
        field,
        source_space,
        source_problem,
        source_temperature,
        adaptation,
        target,
        chart,
    )


def prepare_rational_metric_fields(
    scenario: RationalMetricScenario,
) -> RationalMetricFieldState:
    source = scenario.source_fixture.source
    field, source_space, source_temperature, adaptation, chart = (
        scenario.field,
        scenario.source_space,
        scenario.source_temperature,
        scenario.adaptation,
        scenario.chart,
    )
    old_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(
        source.mesh, component_names=("temperature",)
    ).prepare(cell_geometry=source.geometry)
    new_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(
        scenario.target.mesh, component_names=("temperature",)
    ).prepare(cell_geometry=scenario.target.geometry)
    remap = prepare_unstructured_conservative_remap(
        old_fv,
        new_fv,
        provenance="native-rational-material-history",
        surface_chart_deformation=chart,
        adaptation=adaptation,
        policy=phx.geometry.CommonRefinementPolicy(
            maximum_exact_work=640_000, maximum_memory_bytes=81_920_000
        ),
    )
    if (
        not isinstance(remap.evidence, MappedSurfaceChartRemapEvidence)
        or not remap.succeeded
        or remap.plan is None
    ):
        raise TypeError(
            "surface_metric.fixture.remap must provide certified material correspondence; got refused remap"
        )
    remap_plan = remap.plan
    history = (1.0 + old_fv.cell_centers[:, 0] ** 2)[:, None]
    source_material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-temperature-history"),
                "surface-temperature-history",
                history,
            ),
        )
    )
    carried_history = remap.plan.apply(
        source_material.state("surface-temperature-history").committed
    )
    bound = float(remap.plan.report.total_source_coverage_error_bound) * float(
        np.max(np.abs(history))
    )
    bound += (
        np.finfo(np.float64).eps
        * 32
        * float(np.sum(np.abs(np.asarray(old_fv.cell_volumes)[:, None] * history)))
    )
    prepared = phx.discretization.FiniteElementPlan(
        scenario.target.mesh,
        field,
        coordinate_spec=scenario.target.geometry,
    ).prepare(numeric_version=scenario.target.mesh.numeric_version)
    transfer = prepare_surface_chart_field_transfer(
        source_space,
        prepared,
        chart,
        field_name="temperature",
        source_geometry=source.geometry,
        target_geometry=scenario.target.geometry,
        semantics="intensive",
        maximum_work=640_000,
    )
    carried_temperature = transfer.transfer.apply(source_temperature)
    problem, temperature = solve_surface_reaction(prepared, carried_temperature)
    material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-temperature-history"),
                "surface-temperature-history",
                carried_history,
            ),
        )
    )
    return RationalMetricFieldState(
        scenario,
        old_fv,
        new_fv,
        remap,
        remap_plan,
        history,
        source_material,
        carried_history,
        bound,
        prepared,
        transfer,
        carried_temperature,
        problem,
        temperature,
        material,
    )


def prepare_rational_metric_archive(
    fields: RationalMetricFieldState,
) -> RationalMetricArchive:
    scenario = fields.scenario
    domain = scenario.source_fixture.domain
    adaptation, field = scenario.adaptation, scenario.field
    prepared, temperature, carried_history = (
        fields.prepared,
        fields.temperature,
        fields.carried_history,
    )
    part = MeshPart(domain.source_id, scenario.target)
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult) or not isinstance(
        carrier.geometry, CellGeometrySpec
    ):
        raise TypeError(
            "surface_metric.fixture must retain accepted source geometry and field preparation; got the archived temperature owner requires its cell geometry carrier"
        )
    coefficient_binding = prepare_meshing_field_index_binding(
        carrier, "mesh/vertex_ids", field_space=prepared.field_spaces[0]
    )
    history_binding = prepare_meshing_field_index_binding(carrier, "mesh/cell_ids")
    roles = (
        MeshingFieldStateRole(
            "temperature",
            "temperature",
            "coefficients",
            temperature.shape,
            np.dtype(np.float64),
            1,
            index_binding=coefficient_binding,
        ),
        MeshingFieldStateRole(
            "material-temperature",
            "temperature",
            "material-history",
            carried_history.shape,
            np.dtype(np.float64),
            1,
            value_units=(phx.units.KELVIN,),
            index_binding=history_binding,
        ),
        MeshingFieldStateRole(
            "accepted-epoch", "temperature", "state-epoch", (), np.dtype(np.int64), 1
        ),
    )
    declaration = MeshingFiniteElementFieldDeclaration(
        part_name=part.name,
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict(
            {"temperature": prepared.field_spaces[0].field_space_id}
        ),
        value_units=frozendict({"temperature": (phx.units.KELVIN,)}),
        maximum_derivative_orders=frozendict({"temperature": 1}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version=carrier.mesh.numeric_version,
        finite_element_fields=(field,),
        precision_policy=FiniteElementPrecisionPolicy(),
    )
    if carrier.certification is None:
        raise TypeError(
            "surface_metric.fixture must retain accepted source geometry and field preparation; got the rational archive requires its accepted certification"
        )
    if adaptation.transition is None or adaptation.lineage is None:
        raise TypeError(
            "surface_metric.fixture must retain accepted source geometry and field preparation; got the rational archive requires its accepted source-to-target event"
        )
    event_values = {
        "source_result_id": adaptation.source.result_id,
        "source_mesh_id": adaptation.source.mesh.mesh_id,
        "source_topology_id": adaptation.source.mesh.topology_id,
        "target_result_id": carrier.result_id,
        "target_mesh_id": carrier.mesh.mesh_id,
        "target_topology_id": carrier.mesh.topology_id,
        "transition_id": adaptation.transition.transition_id,
    }
    adaptation_event: dict[str, Any] = {
        **event_values,
        "lineage": adaptation.lineage,
        "accepted_event_id": canonical_fingerprint(
            {
                "kind": "accepted-serial-adaptation-event",
                **event_values,
                "lineage": adaptation.lineage.lineage_id,
            }
        ),
    }
    archive_records = {
        "accepted_target": carrier,
        "certification_inputs": carrier.certification.request,
        "report": carrier.certification,
        "associations": carrier.associations,
        "field_declarations": {part.name: declaration},
        "accepted_data": {
            "adaptation_event": adaptation_event,
            "fields": {
                "temperature": np.asarray(temperature),
                "material-temperature": np.asarray(carried_history),
                "accepted-epoch": np.asarray(1, dtype=np.int64),
            },
        },
    }
    return RationalMetricArchive(
        fields, part, carrier, declaration, adaptation_event, archive_records
    )


class RationalMaterialHistory(StrictModule):
    adaptation: phx.meshing.MeshAdaptationResult
    old_fv: UnstructuredFiniteVolumeDiscretization
    new_fv: UnstructuredFiniteVolumeDiscretization
    chart: PreparedSurfaceChartDeformation
    remap_plan: UnstructuredConservativeRemapPlan
    history: Array
    carried_temperature: Array
    bound: float
    target: CellMeshingResult

    def move_material(
        self,
        value: phx.equations.MaterialTransaction,
        lineage: object,
        args: object,
        *,
        source_workspace: NativeHostStorageWorkspace,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
    ) -> MaterialTopologyTransferResult:
        adaptation = self.adaptation
        old_fv = self.old_fv
        new_fv = self.new_fv
        chart = self.chart
        del lineage, args
        executed_remap = prepare_unstructured_conservative_remap(
            old_fv,
            new_fv,
            provenance="native-rational-transaction-history",
            surface_chart_deformation=chart,
            adaptation=adaptation,
            policy=phx.geometry.CommonRefinementPolicy(
                maximum_exact_work=640_000, maximum_memory_bytes=81_920_000
            ),
            source_workspace=source_workspace,
            scope=scope,
            budget=budget,
        )
        if not executed_remap.succeeded or executed_remap.plan is None:
            raise ValueError(
                "surface_metric.fixture.material_remap must succeed within the transaction allowance; got a refused history remap"
            )
        prior = value.state("surface-temperature-history")
        moved = executed_remap.plan.apply(prior.committed)
        candidate = phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    prior.site_id,
                    prior.model_id,
                    moved,
                    state_version=prior.state_version + 1,
                ),
            )
        )
        return MaterialTopologyTransferResult(
            materials=candidate, remaps=(executed_remap,)
        )

    def certify_material(
        self,
        mesh: CellMesh,
        values: tuple[Array, ...],
        candidate: phx.equations.MaterialTransaction | None,
        lineage: object,
        args: object,
    ) -> bool:
        target = self.target
        old_fv, new_fv = self.old_fv, self.new_fv
        history = self.history
        carried_temperature = self.carried_temperature
        bound = self.bound
        del lineage, args
        if candidate is None or mesh.mesh_id != target.mesh.mesh_id:
            return False
        actual_history = candidate.state("surface-temperature-history").committed
        actual_defect = np.max(
            np.abs(
                np.asarray(old_fv.cell_volumes) @ np.asarray(history)
                - np.asarray(new_fv.cell_volumes) @ np.asarray(actual_history)
            )
        )
        return bool(
            target.audit.passed
            and target.certification is not None
            and target.certification.passed
            and actual_defect <= bound
            and np.all(np.isfinite(np.asarray(values[0])))
            and np.allclose(
                np.asarray(values[0]),
                np.asarray(carried_temperature),
                rtol=MATERIAL_FIELD_RTOL,
                atol=MATERIAL_FIELD_ATOL,
            )
        )


class RationalSurfaceReference(NamedTuple):
    domain: MeshingDomain
    source: CellMeshingResult
    atlas: SurfaceSourceRootAtlas
    original_controls: np.ndarray
    source_coordinates: np.ndarray
    original: SplineCellGeometryElement
    reconstruction: SurfaceGeometryReconstruction
    mesh: CellMesh
    ledger: CoordinateEnclosureBudget


@contextmanager
def rational_surface_reconstruction_reference(
    source_fixture: UVMetricSource,
) -> Iterator[RationalSurfaceReference]:
    domain, source, transfer, policy = source_fixture
    atlas = transfer.support.original
    assert isinstance(atlas, SurfaceSourceRootAtlas)
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_resources(native_scope, policy.limits) as ledger,
    ):
        witness = prepare_surface_metric_source(source, transfer, budget=ledger)
    original_controls = np.asarray(atlas.geometry.coordinates).copy()
    source_coordinates = np.asarray(source.geometry.coordinates).copy()
    originals, _, _ = atlas.geometry.resolve(atlas.mesh)
    original = originals[0]
    assert isinstance(original, SplineCellGeometryElement)
    ledger = CoordinateEnclosureBudget(
        HostBatchTicket(
            policy.limits.maximum_work_units,
            bytes_upper=policy.limits.maximum_scratch_bytes,
        ),
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_memory_bytes=policy.limits.maximum_scratch_bytes,
    )
    with (
        sphere_native_scope(policy.limits) as native_scope,
        native_coordinate_phase(native_scope, ledger),
    ):
        reconstruction = reconstruct_parametric_surface_cell_geometry(
            source.mesh,
            source.geometry,
            source.mesh,
            None,
            domain,
            domain_id=domain.domain_id,
            cell_ids=np.asarray(witness.cell_global_ids),
            cell_patches=np.asarray(witness.patches),
            cell_charts=np.asarray(witness.charts),
            cell_geometry_entity_ids=witness.geometry_entity_ids,
            cell_occurrence_paths=witness.occurrence_paths,
            maximum_fidelity=0.01,
            policy=policy.geometry_transition,
            source_atlas=atlas,
        )
        mesh = reconstruction.target_mesh
        yield RationalSurfaceReference(
            domain,
            source,
            atlas,
            original_controls,
            source_coordinates,
            original,
            reconstruction,
            mesh,
            ledger,
        )
