from __future__ import annotations

from collections.abc import Callable
from math import pi
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
)
from phydrax.geometry.brep import (
    brep_cylinder,
    brep_planar_face,
    CylinderPatch,
    LineCurve,
    PlanarEmbedding,
    PlanarProfile,
    PlanePatch,
    ProfileLoop,
    ProfilePlane,
    SpherePatch,
    TorusPatch,
)
from phydrax.geometry.simplicial import PlanarMeshRegion, SegmentMesh
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveJunction,
    CurveMeshingSpec,
    SurfaceMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import (
    NativeCurveSource,
    NativePlanarSource,
)

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


_TAU = 2.0 * pi
_HALF_PI = 0.5 * pi
_SOURCE_REVISION = "native-planar-feature-source-v1"
_FIDELITY_TOLERANCE_M = 0.02
_PDE_RESIDUAL_TOLERANCE = 2.0e-8
_PLANAR_MEASURE_RELATIVE_TOLERANCE = 1.0e-10

_PROFILE_IDS = (
    "native-surface-pde:annulus",
    "native-surface-pde:annulus:larger",
    "native-surface-pde:capped_cylinder",
    "native-surface-pde:capped_cylinder:larger",
    "native-surface-pde:capped_cylinder_features",
    "native-surface-pde:capped_cylinder_features:larger",
    "native-surface-pde:cylinder_sheet",
    "native-surface-pde:cylinder_sheet:larger",
    "native-surface-pde:folded_plates",
    "native-surface-pde:folded_plates:larger",
    "native-surface-pde:torus",
    "native-surface-pde:torus:larger",
    "planar-hole-feature",
    "planar-hole-feature:larger",
    "planar-thin-channel",
    "sphere-feature-surface",
    "sphere-feature-surface:larger",
)


class _SurfaceDefinition(NamedTuple):
    domain: MeshingDomain
    area: float
    distance: Callable[[np.ndarray], np.ndarray]
    curvature: float
    closed: bool
    euler: int
    boundary_loops: int


def _profile_revision(profile: str, /) -> str:
    return f"{_SOURCE_REVISION}:{profile}"


def _side(
    curve: int,
    origin: tuple[float, float],
    direction: tuple[float, float],
    first: float,
    last: float,
) -> PatchCurveUse:
    return PatchCurveUse(curve, LineCurve(origin, direction), first, last)


def _sphere_domain(source_id: str, revision: str, /) -> MeshingDomain:
    radius = 1.0
    patch = SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), radius)
    loop = (
        PatchPoleUse(
            0,
            LineCurve((0.0, -_HALF_PI), (_TAU, 0.0)),
            0.0,
            1.0,
        ),
        _side(0, (_TAU, 0.0), (0.0, 1.0), -_HALF_PI, _HALF_PI),
        PatchPoleUse(
            1,
            LineCurve((_TAU, _HALF_PI), (-_TAU, 0.0)),
            0.0,
            1.0,
        ),
        _side(0, (0.0, 0.0), (0.0, 1.0), _HALF_PI, -_HALF_PI),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        (MeshingDomainCurve(0, 1),),
        2,
        source_id=source_id,
        source_revision=revision,
        regions=(MeshingDomainRegion("ball", ((0, 1),)),),
    )


def _torus_domain(source_id: str, revision: str, /) -> MeshingDomain:
    major, minor = 2.0, 0.6
    patch = TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), major, minor)
    loop = (
        _side(0, (0.0, 0.0), (1.0, 0.0), 0.0, _TAU),
        _side(1, (_TAU, 0.0), (0.0, 1.0), 0.0, _TAU),
        _side(0, (0.0, _TAU), (1.0, 0.0), _TAU, 0.0),
        _side(1, (0.0, 0.0), (0.0, 1.0), _TAU, 0.0),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        (MeshingDomainCurve(0, 0), MeshingDomainCurve(0, 0)),
        1,
        source_id=source_id,
        source_revision=revision,
        regions=(MeshingDomainRegion("ring", ((0, 1),)),),
    )


def _capped_cylinder_domain(source_id: str, revision: str, /) -> MeshingDomain:
    source_id = f"{source_id}:{revision}"
    model = brep_cylinder(
        0.8,
        1.5,
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id=source_id,
    )
    return MeshingDomain.from_brep(model)


def _annulus_domain(source_id: str, revision: str, /) -> MeshingDomain:
    source_id = f"{source_id}:{revision}"
    model = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.circle((0.0, 0.0), 1.0),
            (ProfileLoop.circle((0.0, 0.0), 0.4),),
        ),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id=source_id,
    )
    return MeshingDomain.from_brep(model)


def _cylinder_sheet_domain(source_id: str, revision: str, /) -> MeshingDomain:
    radius, height = 1.0, 1.0
    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), radius)
    angle = _HALF_PI
    loop = (
        _side(0, (0.0, 0.0), (1.0, 0.0), 0.0, angle),
        _side(1, (angle, 0.0), (0.0, 1.0), 0.0, height),
        _side(2, (0.0, height), (1.0, 0.0), angle, 0.0),
        _side(3, (0.0, 0.0), (0.0, 1.0), height, 0.0),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(3, 0),
        ),
        4,
        source_id=source_id,
        source_revision=revision,
    )


def _folded_plates_domain(source_id: str, revision: str, /) -> MeshingDomain:
    first = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                _side(1, (0.0, 0.0), (1.0, 0.0), -1.0, 0.0),
                _side(0, (0.0, 0.0), (0.0, 1.0), 0.0, 1.0),
                _side(2, (0.0, 1.0), (1.0, 0.0), 0.0, -1.0),
                _side(3, (-1.0, 0.0), (0.0, 1.0), 1.0, 0.0),
            ),
        ),
    )
    second = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (0, 0, 1), (0, 1, 0)),
        (
            (
                _side(4, (0.0, 0.0), (1.0, 0.0), 0.0, 1.0),
                _side(5, (1.0, 0.0), (0.0, 1.0), 0.0, 1.0),
                _side(6, (0.0, 1.0), (1.0, 0.0), 1.0, 0.0),
                _side(0, (0.0, 0.0), (0.0, 1.0), 1.0, 0.0),
            ),
        ),
    )
    return MeshingDomain(
        (first, second),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(2, 0),
            MeshingDomainCurve(1, 3),
            MeshingDomainCurve(3, 2),
            MeshingDomainCurve(0, 4),
            MeshingDomainCurve(4, 5),
            MeshingDomainCurve(5, 1),
        ),
        6,
        source_id=source_id,
        source_revision=revision,
    )


def _surface_definition(profile: str, geometry: str, /) -> _SurfaceDefinition:
    source_id = f"qualification:{profile}"
    revision = _profile_revision(profile)
    match geometry:
        case "sphere":
            domain = _sphere_domain(source_id, revision)
            area = 4.0 * pi
            distance = lambda points: np.abs(np.linalg.norm(points, axis=1) - 1.0)
            return _SurfaceDefinition(domain, area, distance, 1.0, True, 2, 0)
        case "torus":
            major, minor = 2.0, 0.6
            domain = _torus_domain(source_id, revision)

            def distance(points: np.ndarray, /) -> np.ndarray:
                ring = np.hypot(points[:, 0], points[:, 1]) - major
                return np.abs(np.hypot(ring, points[:, 2]) - minor)

            return _SurfaceDefinition(
                domain, 4.0 * pi**2 * major * minor, distance, 1.0 / minor, True, 0, 0
            )
        case "capped_cylinder":
            radius, height = 0.8, 1.5
            domain = _capped_cylinder_domain(source_id, revision)

            def distance(points: np.ndarray, /) -> np.ndarray:
                radial = np.hypot(points[:, 0], points[:, 1])
                wall = np.where(
                    (points[:, 2] >= 0.0) & (points[:, 2] <= height),
                    np.abs(radial - radius),
                    np.inf,
                )
                caps = np.where(
                    radial <= radius,
                    np.minimum(np.abs(points[:, 2]), np.abs(points[:, 2] - height)),
                    np.inf,
                )
                return np.minimum(wall, caps)

            area = _TAU * radius * height + 2.0 * pi * radius**2
            return _SurfaceDefinition(domain, area, distance, 1.0 / radius, True, 2, 0)
        case "annulus":
            outer, inner = 1.0, 0.4
            domain = _annulus_domain(source_id, revision)
            distance = lambda points: np.abs(points[:, 2])
            return _SurfaceDefinition(
                domain, pi * (outer**2 - inner**2), distance, 1.0 / inner, False, 0, 2
            )
        case "cylinder_sheet":
            radius, height = 1.0, 1.0
            domain = _cylinder_sheet_domain(source_id, revision)
            distance = lambda points: np.abs(
                np.hypot(points[:, 0], points[:, 1]) - radius
            )
            return _SurfaceDefinition(
                domain, radius * _HALF_PI * height, distance, 1.0, False, 1, 1
            )
        case "folded_plates":
            domain = _folded_plates_domain(source_id, revision)

            def distance(points: np.ndarray, /) -> np.ndarray:
                on_first = np.where(points[:, 0] <= 0.0, np.abs(points[:, 2]), np.inf)
                on_second = np.where(points[:, 2] >= 0.0, np.abs(points[:, 0]), np.inf)
                return np.minimum(on_first, on_second)

            return _SurfaceDefinition(domain, 2.0, distance, 0.0, False, 1, 1)
        case _:
            raise ValueError(f"Unknown native surface profile geometry {geometry!r}.")


def _surface_area_oracle(geometry: str, /) -> float:
    match geometry:
        case "sphere":
            return 4.0 * pi
        case "torus":
            return 4.0 * pi**2 * 2.0 * 0.6
        case "capped_cylinder":
            return _TAU * 0.8 * 1.5 + 2.0 * pi * 0.8**2
        case "annulus":
            return pi * (1.0**2 - 0.4**2)
        case "cylinder_sheet":
            return _HALF_PI
        case "folded_plates":
            return 2.0
        case _:
            raise ValueError(f"Unknown analytic surface measure {geometry!r}.")


def _surface_parameters(geometry: str, /) -> dict[str, float]:
    match geometry:
        case "sphere":
            return {"radius_m": 1.0}
        case "torus":
            return {"major_radius_m": 2.0, "minor_radius_m": 0.6}
        case "capped_cylinder":
            return {"radius_m": 0.8, "height_m": 1.5}
        case "annulus":
            return {"outer_radius_m": 1.0, "inner_radius_m": 0.4}
        case "cylinder_sheet":
            return {"radius_m": 1.0, "height_m": 1.0, "sweep_angle_radians": _HALF_PI}
        case "folded_plates":
            return {"plate_side_m": 1.0, "dihedral_angle_radians": _HALF_PI}
        case _:
            raise ValueError(f"Unknown analytic surface parameters {geometry!r}.")


def _planar_definition(
    profile: str, /
) -> tuple[NativePlanarSource, float, dict[str, object]]:
    source_id = f"qualification:{profile}"
    if profile.startswith("planar-hole-feature"):
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
        loops = ((0, 1, 2, 3), (4, 5, 6, 7))
        region = PlanarMeshRegion(vertices, loops, feature_id=source_id)
        embedded = SegmentMesh(
            jnp.asarray(((0.2, 0.2), (0.5, 1.7)), dtype=jnp.float64),
            jnp.asarray(((0, 1),), dtype=jnp.int32),
            source_id=f"{source_id}:embedded-segment",
        )
        source = NativePlanarSource(
            region,
            _profile_revision(profile),
            embedded=embedded,
        )
        area = 3.84
        physical_input: dict[str, object] = {
            "geometry": "square-minus-square-hole-with-embedded-segment",
            "outer_vertices": 4,
            "hole_vertices": 4,
            "embedded_segment": ((0.2, 0.2), (0.5, 1.7)),
            "embedded_source_id": embedded.source_id,
            "analytic_region": "plate",
        }
        return source, area, physical_input
    if profile == "planar-thin-channel":
        vertices = np.asarray(
            (
                (0.0, 0.0),
                (0.4, 0.0),
                (0.4, 0.19),
                (0.6, 0.19),
                (0.6, 0.0),
                (1.0, 0.0),
                (1.0, 0.4),
                (0.6, 0.4),
                (0.6, 0.21),
                (0.4, 0.21),
                (0.4, 0.4),
                (0.0, 0.4),
            ),
            dtype=np.float64,
        )
        region = PlanarMeshRegion(
            vertices, (tuple(range(vertices.shape[0])),), feature_id=source_id
        )
        source = NativePlanarSource(region, _profile_revision(profile))
        return (
            source,
            0.324,
            {
                "geometry": "analytic-dumbbell-thin-channel",
                "channel_width": 0.02,
                "outer_bounds": ((0.0, 0.0), (1.0, 0.4)),
                "analytic_region": "plate",
            },
        )
    raise ValueError(f"Unknown native planar profile {profile!r}.")


def _triangle_data(
    result: phx.meshing.CellMeshingResult, /
) -> tuple[np.ndarray, np.ndarray]:
    blocks = tuple(result.mesh.blocks)
    if not blocks or any(block.cell_kind != "triangle" for block in blocks):
        raise RuntimeError(
            "The native planar-feature request did not produce only triangles."
        )
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in blocks), axis=0
    )
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise RuntimeError(
            "The native planar-feature mesh has invalid triangle coordinates."
        )
    return coordinates, cells


def _components(edges: np.ndarray, /) -> int:
    if edges.size == 0:
        return 0
    vertices = np.unique(edges)
    parent = {int(vertex): int(vertex) for vertex in vertices.tolist()}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    for first, second in np.asarray(edges, dtype=np.int64).tolist():
        first_root, second_root = find(int(first)), find(int(second))
        if first_root != second_root:
            parent[second_root] = first_root
    return len({find(int(vertex)) for vertex in vertices.tolist()})


def _topology_record(
    result: phx.meshing.CellMeshingResult,
    expected_euler: int,
    expected_boundary_loops: int,
    /,
) -> dict[str, object]:
    coordinates, cells = _triangle_data(result)
    edge_rows = np.sort(
        np.concatenate((cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)]), axis=0),
        axis=1,
    )
    edges, incidence = np.unique(edge_rows, axis=0, return_counts=True)
    if np.any((incidence != 1) & (incidence != 2)):
        raise RuntimeError(
            "Published triangle topology contains a non-manifold edge incidence."
        )
    used_vertices = np.unique(cells)
    euler = int(used_vertices.size - edges.shape[0] + cells.shape[0])
    boundary = edges[incidence == 1]
    boundary_counts = (
        np.unique(boundary, return_counts=True)[1]
        if boundary.size
        else np.empty((0,), dtype=np.int64)
    )
    boundary_cycles = _components(boundary)
    connected_components = _components(edges)
    if boundary.size and np.any(boundary_counts != 2):
        raise RuntimeError(
            "Published source boundary is not a disjoint union of closed loops."
        )
    if (
        euler != expected_euler
        or boundary_cycles != expected_boundary_loops
        or connected_components != 1
    ):
        raise RuntimeError(
            "Published source topology differs from the analytic profile: "
            f"Euler={euler} (expected {expected_euler}), boundary loops="
            f"{boundary_cycles} (expected {expected_boundary_loops}), "
            f"components={connected_components}."
        )
    return {
        "vertices": int(used_vertices.size),
        "edges": int(edges.shape[0]),
        "faces": int(cells.shape[0]),
        "euler_characteristic": euler,
        "boundary_edges": int(boundary.shape[0]),
        "boundary_loops": boundary_cycles,
        "connected_components": connected_components,
        "expected_euler_characteristic": expected_euler,
        "expected_boundary_loops": expected_boundary_loops,
    }


def _association_record(
    result: phx.meshing.CellMeshingResult,
    source_id: str,
    source_revision: str,
    expected_entity_ids: set[str],
    /,
) -> dict[str, object]:
    associations = tuple(result.associations)
    matching = tuple(
        value
        for value in associations
        if value.source_id == source_id and value.source_revision == source_revision
    )
    if not matching or any(not value.complete for value in matching):
        raise RuntimeError(
            "Native generation omitted complete source entity associations."
        )
    actual = {identifier for value in matching for identifier in value.source_entity_ids}
    missing = sorted(expected_entity_ids - actual)
    if missing:
        raise RuntimeError(
            f"Native generation omitted authored source entities: {missing!r}."
        )
    return {
        "association_count": len(matching),
        "complete": True,
        "expected_source_entity_count": len(expected_entity_ids),
        "associated_source_entity_count": len(expected_entity_ids & actual),
        "source_entity_ids": tuple(sorted(expected_entity_ids)),
    }


def _domain_source_entities(domain: MeshingDomain, /) -> set[str]:
    return {
        domain.entity_id(dimension, slot)
        for dimension in (0, 1, 2)
        for slot in range(len(domain.source_indices[dimension]))
    }


def _planar_source_entities(source: NativePlanarSource, /) -> set[str]:
    identifiers = {
        source.source_entity_key_for_scope(dimension, int(entity_id))[0]
        for dimension in (0, 1, 2)
        for entity_id in source.scope_entity_ids(dimension).tolist()
    }
    identifiers.add(source.region_entity_id)
    return identifiers


def _solve_manufactured_surface(
    result: phx.meshing.CellMeshingResult,
    profile: str,
    geometry: str,
    /,
) -> dict[str, object]:
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    space = FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()

    def exact(points: Any) -> Any:
        if geometry == "sphere":
            return points[..., 0] / jnp.linalg.norm(points, axis=-1)
        if geometry == "torus":
            return points[..., 2]
        if geometry == "capped_cylinder":
            z = points[..., 2]
            return z * z * (1.5 - z) ** 2
        if geometry == "annulus":
            return points[..., 0]
        if geometry == "cylinder_sheet":
            return points[..., 2]
        if geometry == "folded_plates":
            return points[..., 1]
        raise ValueError(f"Unknown manufactured surface law {geometry!r}.")

    closed = geometry in ("sphere", "torus", "capped_cylinder")
    actions: list[Any] = [phx.equations.DiffusionAction("u", 1.0)]
    if closed:
        actions.append(phx.equations.MassAction("u", 1.0))

        def forcing(points: Any, context: Any) -> Any:
            del context
            if geometry == "sphere":
                return 3.0 * exact(points)
            if geometry == "torus":
                rho = jnp.linalg.norm(points[..., :2], axis=-1)
                z = points[..., 2]
                return z * (1.0 + (2.0 * rho - 2.0) / (0.6**2 * rho))
            z = points[..., 2]
            second = 2.0 * 1.5**2 - 12.0 * 1.5 * z + 12.0 * z * z
            return jnp.where((z == 0.0) | (z == 1.5), 0.0, exact(points) - second)

        source = phx.equations.coefficient(
            forcing,
            coefficient_id=f"{profile}:manufactured-source",
        )
        actions.append(phx.equations.SourceAction("u", source))
    form = phx.equations.FiniteElementForm(
        f"{profile.replace(':', '-')}-manufactured-surface",
        "u",
        tuple(actions),
    )
    if closed:
        problem = phx.equations.compile_finite_element_problem(form, space)
    else:
        constraint = phx.discretization.dirichlet_constraint(space, "u")
        problem = phx.equations.compile_finite_element_problem(
            form,
            space,
            constraint=constraint,
            dirichlet_values=exact,
        )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    dof_coordinates = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    expected = np.asarray(exact(jnp.asarray(dof_coordinates)), dtype=np.float64)
    if values.shape != expected.shape or not np.all(np.isfinite(values)):
        raise RuntimeError(
            "Native surface FE consumer returned malformed manufactured coefficients."
        )
    solution_error = float(np.max(np.abs(values - expected), initial=0.0))
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    rhs_norm = float(jnp.linalg.norm(rhs))
    normalized_residual = residual / max(1.0, rhs_norm)
    if (
        not bool(jnp.all(solved.successful))
        or not np.isfinite(solution_error)
        or normalized_residual > _PDE_RESIDUAL_TOLERANCE
    ):
        raise RuntimeError(
            "Native surface PDE failed its manufactured solve/residual oracle: "
            f"solution error={solution_error}, normalized residual={normalized_residual}."
        )
    if not closed and solution_error > _PDE_RESIDUAL_TOLERANCE:
        raise RuntimeError(
            f"Exact affine open-surface field error {solution_error} exceeds {_PDE_RESIDUAL_TOLERANCE}."
        )
    return {
        "form": form.form_id,
        "field": field.name,
        "dofs": int(values.size),
        "successful": bool(jnp.all(solved.successful)),
        "manufactured_field": geometry,
        "maximum_dof_solution_error": solution_error,
        "residual_norm": residual,
        "normalized_residual": normalized_residual,
        "normalized_residual_tolerance": _PDE_RESIDUAL_TOLERANCE,
        "solution_error_tolerance": _PDE_RESIDUAL_TOLERANCE if not closed else None,
    }


def _qualify_surface(profile: str, geometry: str, /) -> dict[str, object]:
    definition = _surface_definition(profile, geometry)
    area_oracle = _surface_area_oracle(geometry)
    source_measure_relative_error = abs(definition.area - area_oracle) / area_oracle
    if source_measure_relative_error > _PLANAR_MEASURE_RELATIVE_TOLERANCE:
        raise RuntimeError(
            f"Analytic source area {definition.area} differs from the independent "
            f"{geometry} oracle {area_oracle}."
        )
    domain = definition.domain
    scope = MeshingScope.whole(domain, 2)
    scale = 2 if profile.endswith(":larger") else 1
    if geometry == "sphere":
        resolution = 4 * scale
        size = 1.6 / resolution
    else:
        authored_sizes = {
            "torus": 0.45,
            "capped_cylinder": 0.3,
            "annulus": 0.2,
            "cylinder_sheet": 0.25,
            "folded_plates": 0.3,
        }
        size = authored_sizes[geometry] / scale
    combined = profile.startswith("native-surface-pde:capped_cylinder_features")
    sphere_feature = profile.startswith("sphere-feature-surface")
    protected_deviation = 0.25 * size * size
    requested_deviation = (
        min(_FIDELITY_TOLERANCE_M, protected_deviation)
        if combined or sphere_feature
        else _FIDELITY_TOLERANCE_M
    )
    controls = [
        UniformSizeControl(
            scope,
            size,
            maximum_size=1.5 * size if combined or sphere_feature else None,
            strength=SizeControlStrength.HARD
            if combined or sphere_feature
            else SizeControlStrength.SOFT,
        )
    ]
    protected_features: tuple[ProtectedFeature, ...] = ()
    compliance = None
    if combined:
        curve_scope = MeshingScope(
            domain.source_id,
            domain.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            1,
            domain.entity_set_id(1),
            np.asarray(domain.scope_indices(1), dtype=np.int64),
        )
        protected_features = (
            ProtectedFeature(
                scope,
                FeatureKind.SURFACE,
                maximum_deviation=protected_deviation,
            ),
            ProtectedFeature(
                curve_scope,
                FeatureKind.CURVE,
                maximum_deviation=protected_deviation,
            ),
        )
        compliance = SizeCompliancePolicy(
            relative_tolerance=0.5,
            target_statistics=("p50",),
        )
    elif sphere_feature:
        protected_features = (
            ProtectedFeature(
                scope,
                FeatureKind.SURFACE,
                maximum_deviation=protected_deviation,
            ),
        )
        compliance = SizeCompliancePolicy(
            relative_tolerance=0.5,
            target_statistics=("p50",),
        )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=tuple(controls),
        protected_features=protected_features,
        size_compliance=compliance,
        maximum_deviation=requested_deviation,
    )
    options = NativeMeshingOptions("parametric_surface")
    plan = NativeMeshingProvider(options).plan(
        domain,
        specification,
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    topology = _topology_record(result, definition.euler, definition.boundary_loops)
    certification = result.certification
    charts = result.surface_source
    if certification is None or certification.fidelity is None or charts is None:
        raise RuntimeError(
            "Native parametric surface omitted source-chart fidelity evidence."
        )
    source_query = certification.inputs.source
    if source_query is None:
        raise RuntimeError("Native surface certification omitted its source boundary query.")
    charts.require_bound(result.mesh, result.geometry, source_query)
    if charts.domain.content_domain_id() != domain.content_domain_id():
        raise RuntimeError(
            "Native surface publication changed its authored trimmed source domain."
        )
    fidelity = certification.fidelity
    if (fidelity.binding.source_id, fidelity.binding.source_revision) != (
        domain.source_id,
        domain.source_revision,
    ):
        raise RuntimeError(
            "Native surface fidelity certificate changed the profile source identity."
        )
    chart_coverage = fidelity.coverage
    if chart_coverage is None or not chart_coverage.complete:
        raise RuntimeError(
            "Native surface certificate omitted complete source-chart coverage."
        )
    if (
        fidelity.tolerance > requested_deviation
        or not np.isfinite(fidelity.mesh_to_source_upper)
        or fidelity.mesh_to_source_upper > requested_deviation
    ):
        raise RuntimeError(
            f"Native surface source-fidelity certificate exceeds {requested_deviation} m."
        )
    association = _association_record(
        result,
        domain.source_id,
        domain.source_revision,
        _domain_source_entities(domain),
    )
    coordinates, cells = _triangle_data(result)
    edge_midpoints = np.concatenate(
        (
            0.5 * (coordinates[cells[:, 0]] + coordinates[cells[:, 1]]),
            0.5 * (coordinates[cells[:, 1]] + coordinates[cells[:, 2]]),
            0.5 * (coordinates[cells[:, 2]] + coordinates[cells[:, 0]]),
            np.mean(coordinates[cells], axis=1),
        ),
        axis=0,
    )
    sampled_deviation = float(np.max(definition.distance(edge_midpoints), initial=0.0))
    deviation_roundoff = (
        64.0
        * np.finfo(np.float64).eps
        * (1.0 + float(np.max(np.abs(coordinates), initial=0.0)))
    )
    if sampled_deviation > fidelity.mesh_to_source_upper + deviation_roundoff:
        raise RuntimeError(
            "Independent source-distance samples falsify the native fidelity upper bound."
        )
    if sampled_deviation > _FIDELITY_TOLERANCE_M + deviation_roundoff:
        raise RuntimeError(
            "Independent source-distance samples exceed the 0.02 m profile tolerance."
        )
    expected_labels = {f"region:{region.name}" for region in domain.regions}
    actual_labels = {
        label.name for label in result.labels if label.name.startswith("region:")
    }
    if actual_labels != expected_labels:
        raise RuntimeError(
            f"Native surface region labels differ from the source request: {actual_labels!r} != {expected_labels!r}."
        )
    measured_area = float(
        np.sum(np.asarray(result.quality.evaluation.measures, dtype=np.float64))
    )
    area_tolerance = definition.area * max(
        0.25 * (definition.curvature * size) ** 2,
        1.0e-12,
    )
    if (
        not np.isfinite(measured_area)
        or abs(measured_area - definition.area) > area_tolerance
    ):
        raise RuntimeError(
            f"Native surface area {measured_area} differs from analytic source area "
            f"{definition.area} beyond {area_tolerance}."
        )
    pde = _solve_manufactured_surface(result, profile, geometry)
    return {
        "profile": profile,
        "status": "passed",
        "producer": "NativeMeshingProvider",
        "route": plan.route,
        "plan_id": plan.plan_id,
        "source": {
            "kind": (
                "native BRep converted to MeshingDomain"
                if geometry in ("annulus", "capped_cylinder")
                else "revision-bound analytic MeshingDomain"
            ),
            "source_id": domain.source_id,
            "source_revision": domain.source_revision,
            "geometry": geometry,
            "physical_parameters": _surface_parameters(geometry),
            "patch_count": len(domain.patches),
            "curve_count": len(domain.curves),
            "corner_count": int(domain.corner_points.shape[0]),
            "region_ids": tuple(region.name for region in domain.regions),
        },
        "controls": {
            "resolution_scale": scale,
            "target_size_m": size,
            "size_strength": "hard" if combined or sphere_feature else "soft",
            "maximum_size_m": 1.5 * size if combined or sphere_feature else None,
            "size_compliance_relative_tolerance": 0.5
            if combined or sphere_feature
            else None,
            "protected_feature_deviation_m": protected_deviation
            if combined or sphere_feature
            else None,
            "requested_source_deviation_m": requested_deviation,
            "maximum_source_deviation_m": _FIDELITY_TOLERANCE_M,
        },
        "topology": topology,
        "exact_source_measure": {
            "kind": "analytic-surface-area",
            "value": definition.area,
            "independent_analytic_area": area_oracle,
            "relative_error": source_measure_relative_error,
            "relative_tolerance": _PLANAR_MEASURE_RELATIVE_TOLERANCE,
            "mesh_surface_area": measured_area,
            "mesh_area_approximation_bound": area_tolerance,
        },
        "source_association": association,
        "source_fidelity": {
            "certificate_id": fidelity.certificate_id,
            "source_chart_id": charts.source_chart_id,
            "chart_coverage_complete": chart_coverage.complete,
            "certified_deviation_upper_m": float(fidelity.mesh_to_source_upper),
            "certified_tolerance_m": float(fidelity.tolerance),
            "independent_sampled_deviation_m": sampled_deviation,
            "tolerance_m": _FIDELITY_TOLERANCE_M,
            "requested_deviation_m": requested_deviation,
        },
        "finite_element_consumer": pde,
        "accepted": accepted,
    }


def _planar_measure(region: PlanarMeshRegion, /) -> float:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    signed = 0.0
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        loop = vertices[np.asarray(region.edges, dtype=np.int64)[start:stop, 0]]
        signed += 0.5 * float(
            np.sum(
                loop[:, 0] * np.roll(loop[:, 1], -1)
                - np.roll(loop[:, 0], -1) * loop[:, 1]
            )
        )
    return signed


def _point_segment_distances(
    points: np.ndarray,
    vertices: np.ndarray,
    edges: np.ndarray,
    /,
) -> np.ndarray:
    distances = np.full((points.shape[0],), np.inf, dtype=np.float64)
    for first, last in np.asarray(edges, dtype=np.int64).tolist():
        start = vertices[int(first)]
        direction = vertices[int(last)] - start
        parameter = np.clip(
            np.sum((points - start) * direction, axis=1) / np.dot(direction, direction),
            0.0,
            1.0,
        )
        nearest = start + parameter[:, None] * direction
        distances = np.minimum(distances, np.linalg.norm(points - nearest, axis=1))
    return distances


def _planar_source_fidelity(
    result: phx.meshing.CellMeshingResult,
    source: NativePlanarSource,
    /,
) -> dict[str, object]:
    coordinates, cells = _triangle_data(result)
    if coordinates.shape[1] != 2:
        raise RuntimeError(
            "Native planar source publication changed its authored 2D frame."
        )
    triangles = coordinates[cells]
    edge_points = np.concatenate(
        (
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
            np.mean(triangles, axis=1),
        ),
        axis=0,
    )
    samples = np.concatenate((coordinates, edge_points), axis=0)
    source_vertices = np.asarray(source.region.vertices, dtype=np.float64)
    source_edges = np.asarray(source.region.edges, dtype=np.int64)
    offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
    inside = np.zeros((samples.shape[0],), dtype=np.bool_)
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        loop = source_vertices[source_edges[start:stop, 0]]
        following = np.roll(loop, -1, axis=0)
        vertical = following[:, 1] - loop[:, 1]
        safe_vertical = np.where(vertical == 0.0, 1.0, vertical)
        crosses = (loop[:, 1, None] > samples[None, :, 1]) != (
            following[:, 1, None] > samples[None, :, 1]
        )
        x_crossings = loop[:, 0, None] + (
            (samples[None, :, 1] - loop[:, 1, None])
            * (following[:, 0] - loop[:, 0])[:, None]
            / safe_vertical[:, None]
        )
        loop_inside = np.logical_xor.reduce(
            crosses & (samples[None, :, 0] < x_crossings),
            axis=0,
        )
        inside ^= loop_inside
    boundary_distance = _point_segment_distances(samples, source_vertices, source_edges)
    boundary_tolerance = 1.0e-12
    outside = ~inside & (boundary_distance > boundary_tolerance)
    if np.any(outside):
        raise RuntimeError(
            "Native planar triangles leave or cross the authored source region."
        )
    maximum_source_domain_distance = float(
        np.max(np.where(inside, 0.0, boundary_distance), initial=0.0)
    )
    if maximum_source_domain_distance > boundary_tolerance:
        raise RuntimeError(
            "Native planar source-domain distance exceeds the exact-boundary tolerance."
        )
    mesh_edges = np.sort(
        np.concatenate(
            (cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)]),
            axis=0,
        ),
        axis=1,
    )
    unique_edges, incidence = np.unique(mesh_edges, axis=0, return_counts=True)
    boundary_vertices = np.unique(unique_edges[incidence == 1])
    maximum_boundary_deviation = float(
        np.max(
            _point_segment_distances(
                coordinates[boundary_vertices],
                source_vertices,
                source_edges,
            ),
            initial=0.0,
        )
    )
    if maximum_boundary_deviation > boundary_tolerance:
        raise RuntimeError(
            "Native planar boundary vertices departed from authored source edges."
        )
    embedded_evidence = None
    if source.embedded is not None:
        embedded_label = next(
            (label for label in result.labels if label.name == "embedded"),
            None,
        )
        edge_set = result.mesh.entity_set(1)
        if (
            embedded_label is None
            or embedded_label.scope.entity_set_id != edge_set.entity_set_id
        ):
            raise RuntimeError(
                "Native planar source omitted its embedded-segment edge selection."
            )
        edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)
        row_by_id = {
            int(identifier): row for row, identifier in enumerate(edge_ids.tolist())
        }
        selected_ids = np.asarray(embedded_label.scope.entity_ids, dtype=np.int64)
        if selected_ids.size == 0 or any(
            int(identifier) not in row_by_id for identifier in selected_ids.tolist()
        ):
            raise RuntimeError(
                "Native planar embedded-segment label has no published source edges."
            )
        selected_rows = np.asarray(
            [row_by_id[int(identifier)] for identifier in selected_ids.tolist()],
            dtype=np.int64,
        )
        connectivity = result.mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise RuntimeError("Native planar mesh omitted polygonal edge connectivity.")
        connectivity_edges = np.asarray(connectivity.edges, dtype=np.int64)
        embedded_edges = connectivity_edges[selected_rows]
        embedded_vertices = np.asarray(source.embedded.vertices, dtype=np.float64)
        authored_edges = np.asarray(source.embedded.edges, dtype=np.int64)
        feature_points = coordinates[embedded_edges].reshape((-1, 2))
        feature_deviation = float(
            np.max(
                _point_segment_distances(
                    feature_points, embedded_vertices, authored_edges
                ),
                initial=0.0,
            )
        )
        feature_length = float(
            np.sum(
                np.linalg.norm(
                    coordinates[embedded_edges[:, 1]] - coordinates[embedded_edges[:, 0]],
                    axis=1,
                )
            )
        )
        authored_length = float(
            np.sum(
                np.linalg.norm(
                    embedded_vertices[authored_edges[:, 1]]
                    - embedded_vertices[authored_edges[:, 0]],
                    axis=1,
                )
            )
        )
        if feature_deviation > boundary_tolerance or not np.isclose(
            feature_length, authored_length, rtol=0.0, atol=boundary_tolerance
        ):
            raise RuntimeError(
                "Native planar meshing changed the embedded source segment."
            )
        embedded_evidence = {
            "source_segment_length": authored_length,
            "published_segment_length": feature_length,
            "maximum_segment_deviation": feature_deviation,
            "tolerance": boundary_tolerance,
            "published_edge_count": int(selected_rows.size),
        }
    return {
        "kind": "exact-piecewise-linear-planar-source",
        "sampled_source_domain_points": int(samples.shape[0]),
        "maximum_source_domain_distance_m": maximum_source_domain_distance,
        "outside_source_domain_points": int(np.count_nonzero(outside)),
        "maximum_boundary_vertex_deviation": maximum_boundary_deviation,
        "boundary_tolerance": boundary_tolerance,
        "embedded_feature": embedded_evidence,
    }


def _solve_planar_field(
    result: phx.meshing.CellMeshingResult,
    profile: str,
    /,
) -> dict[str, object]:
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    space = FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        f"{profile.replace(':', '-')}-manufactured-planar",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=lambda points: points[..., 0],
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    dof_coordinates = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    expected = dof_coordinates[:, 0]
    error = float(np.max(np.abs(values - expected), initial=0.0))
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    rhs_norm = float(jnp.linalg.norm(rhs))
    normalized_residual = residual / max(1.0, rhs_norm)
    if (
        not bool(jnp.all(solved.successful))
        or values.shape != expected.shape
        or error > _PDE_RESIDUAL_TOLERANCE
        or normalized_residual > _PDE_RESIDUAL_TOLERANCE
    ):
        raise RuntimeError(
            "Native planar-feature FE solve failed the exact affine field/residual oracle: "
            f"error={error}, normalized residual={normalized_residual}."
        )
    return {
        "form": form.form_id,
        "field": field.name,
        "dofs": int(values.size),
        "successful": bool(jnp.all(solved.successful)),
        "manufactured_field": "u=x",
        "maximum_solution_error": error,
        "normalized_residual": normalized_residual,
        "normalized_residual_tolerance": _PDE_RESIDUAL_TOLERANCE,
        "solution_error_tolerance": _PDE_RESIDUAL_TOLERANCE,
    }


def _qualify_planar(profile: str, /) -> dict[str, object]:
    source, expected_area, physical_input = _planar_definition(profile)
    scope = MeshingScope.whole(source, 2)
    larger = profile.endswith(":larger")
    resolution = 8 if larger else 1
    size = 0.8 / resolution
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(
            UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=SizeControlStrength.HARD,
            ),
        ),
        size_compliance=SizeCompliancePolicy(
            relative_tolerance=0.5,
            target_statistics=("p50",),
        ),
        quality_target=phx.meshing.MeshQualityTarget(minimum_angle=np.radians(25.0)),
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions("planar_constrained_delaunay")
    ).plan(
        source,
        specification,
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    expected_labels = {"region:0"}
    if source.embedded is not None:
        expected_labels.add("embedded")
    actual_labels = {label.name for label in result.labels}
    if actual_labels != expected_labels:
        raise RuntimeError(
            f"Native planar region labels differ from the source request: {actual_labels!r} != {expected_labels!r}."
        )
    topology = _topology_record(
        result,
        0 if source.region.loop_offsets.shape[0] == 3 else 1,
        2 if source.region.loop_offsets.shape[0] == 3 else 1,
    )
    association = _association_record(
        result,
        source.source_id,
        source.source_revision,
        _planar_source_entities(source),
    )
    source_fidelity = _planar_source_fidelity(result, source)
    measured_area = float(
        np.sum(np.asarray(result.quality.evaluation.measures, dtype=np.float64))
    )
    source_area = _planar_measure(source.region)
    if not np.isclose(
        source_area, expected_area, rtol=0.0, atol=1.0e-14
    ) or not np.isclose(
        measured_area,
        expected_area,
        rtol=_PLANAR_MEASURE_RELATIVE_TOLERANCE,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"Native planar region area {measured_area} disagrees with the exact "
            f"source oracle {expected_area} (source-loop measure {source_area})."
        )
    pde = _solve_planar_field(result, profile)
    return {
        "profile": profile,
        "status": "passed",
        "producer": "NativeMeshingProvider",
        "route": plan.route,
        "plan_id": plan.plan_id,
        "source": {
            "kind": "NativePlanarSource",
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "declared_source_revision": source.declared_source_revision,
            **physical_input,
        },
        "controls": {
            "resolution_scale": resolution,
            "target_size_m": size,
            "maximum_size_m": 1.5 * size,
            "minimum_angle_radians": float(np.radians(25.0)),
            "size_compliance_relative_tolerance": 0.5,
            "size_strength": "hard",
        },
        "topology": topology,
        "exact_source_measure": {
            "kind": "analytic-planar-region-area",
            "region": "plate",
            "value": expected_area,
            "source_loop_measure": source_area,
            "mesh_area": measured_area,
            "relative_tolerance": _PLANAR_MEASURE_RELATIVE_TOLERANCE,
        },
        "source_association": association,
        "source_fidelity": source_fidelity,
        "finite_element_consumer": pde,
        "accepted": accepted,
    }


def _qualify_junction_network() -> dict[str, object]:
    curves = SegmentMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)), dtype=jnp.float64),
        jnp.asarray(((0, 1), (0, 2), (0, 3)), dtype=jnp.int32),
        source_id="qualification:planar-junction-network",
    )
    source = NativeCurveSource(curves, "revision:planar-junction-network-v1")
    scope = MeshingScope.whole(source, 1)
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(
            UniformSizeControl(
                scope,
                0.25,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        junctions=(
            CurveJunction(
                "three-arm-junction",
                ((0, "start"), (1, "start"), (2, "start")),
            ),
        ),
        maximum_deviation=0.0,
    )
    plan = NativeMeshingProvider(NativeMeshingOptions("curve_arc_length")).plan(
        source,
        specification,
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    if not result.mesh.blocks or any(
        block.cell_kind != "interval" for block in result.mesh.blocks
    ):
        raise RuntimeError("Native curve generation did not preserve interval cells.")
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    edges = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in result.mesh.blocks),
        axis=0,
    )
    edge_vectors = coordinates[edges[:, 1]] - coordinates[edges[:, 0]]
    length = float(np.sum(np.linalg.norm(edge_vectors, axis=1)))
    center_rows = np.flatnonzero(np.linalg.norm(coordinates, axis=1) <= 1.0e-14)
    if center_rows.size != 1:
        raise RuntimeError(
            "Native curve mesh did not retain one exact three-way junction vertex."
        )
    center = int(center_rows[0])
    center_degree = int(np.count_nonzero(np.any(edges == center, axis=1)))
    if center_degree != 3 or not np.isclose(length, 3.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"Native curve network changed its three-arm junction/length: degree={center_degree}, length={length}."
        )
    segments = np.asarray(curves.edges, dtype=np.int64)
    residuals = []
    source_points = np.asarray(curves.vertices, dtype=np.float64)
    for point in coordinates:
        candidates = []
        for first, last in segments.tolist():
            start, end = source_points[int(first)], source_points[int(last)]
            direction = end - start
            parameter = np.clip(
                np.dot(point - start, direction) / np.dot(direction, direction), 0.0, 1.0
            )
            candidates.append(
                float(np.linalg.norm(point - (start + parameter * direction)))
            )
        residuals.append(min(candidates))
    maximum_line_residual = float(np.max(residuals, initial=0.0))
    if maximum_line_residual > 1.0e-12:
        raise RuntimeError(
            "Native curve vertices departed from their authored junction branches."
        )
    expected_ids = {
        source.source_entity_key_for_scope(1, int(identifier))[0]
        for identifier in source.scope_entity_ids(1).tolist()
    }
    association = _association_record(
        result,
        source.source_id,
        source.source_revision,
        expected_ids,
    )
    return {
        "status": "passed",
        "producer": "NativeMeshingProvider",
        "route": plan.route,
        "plan_id": plan.plan_id,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "physical_input": "three straight unit arms sharing one authored source endpoint",
        "junction_id": specification.junctions[0].junction_id,
        "junction_valence": center_degree,
        "analytic_total_length": 3.0,
        "measured_total_length": length,
        "maximum_source_line_residual": maximum_line_residual,
        "source_association": association,
        "accepted": accepted,
    }


def native_planar_feature_matrix() -> dict[str, object]:
    """Run every recovered planar-features profile and its native PDE consumer."""
    records: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    try:
        meshcore = require_meshcore()
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        records = {
            profile: {
                "profile": profile,
                "status": "failed",
                "stage": "native-runtime-prerequisite",
                "failure": detail,
            }
            for profile in _PROFILE_IDS
        }
        return {
            "status": "failed",
            "workflow": "planar-features",
            "source_registry": "tools/meshing_qualification/native_case_migration.json#planar-features",
            "profile_count": len(_PROFILE_IDS),
            "profiles": records,
            "junction_network": {
                "status": "failed",
                "stage": "native-runtime-prerequisite",
                "failure": detail,
            },
            "failures": [
                {"profile": profile, "failure": detail} for profile in _PROFILE_IDS
            ],
            "runtime": runtime_identity(),
        }
    for profile in _PROFILE_IDS:
        try:
            if profile.startswith("native-surface-pde:"):
                geometry = profile.removeprefix("native-surface-pde:").removesuffix(
                    ":larger"
                )
                if geometry == "capped_cylinder_features":
                    geometry = "capped_cylinder"
                records[profile] = _qualify_surface(profile, geometry)
            elif profile.startswith("sphere-feature-surface"):
                records[profile] = _qualify_surface(profile, "sphere")
            else:
                records[profile] = _qualify_planar(profile)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            records[profile] = {
                "profile": profile,
                "status": "failed",
                "failure": failure,
            }
            failures.append({"profile": profile, "failure": failure})
    try:
        junction_network = _qualify_junction_network()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        junction_network = {"status": "failed", "failure": failure}
        failures.append(
            {"profile": "planar-feature-junction-network", "failure": failure}
        )
    status = "passed" if not failures and len(records) == len(_PROFILE_IDS) else "failed"
    return {
        "status": status,
        "workflow": "planar-features",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#planar-features",
        "source_provenance": {
            "record": "tools/meshing_qualification/native_case_migration.json",
            "donor_source_branch": "feature/native-meshing-20260929",
            "donor_source_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
            "physical_inputs": "profile-bound analytic planar regions and source-chart trimmed surfaces",
        },
        "profile_count": len(_PROFILE_IDS),
        "completed_profile_count": sum(
            record.get("status") == "passed" for record in records.values()
        ),
        "profiles": records,
        "junction_network": junction_network,
        "failures": failures,
        "tolerances": {
            "analytic_measure_relative": _PLANAR_MEASURE_RELATIVE_TOLERANCE,
            "native_surface_maximum_deviation_m": _FIDELITY_TOLERANCE_M,
            "pde_normalized_residual": _PDE_RESIDUAL_TOLERANCE,
        },
        "native_runtime": meshcore,
        "runtime": runtime_identity(),
    }
