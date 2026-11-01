from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.reconstruction import (
    AuthoredTriangleSheet,
    reconstruct_surface_region,
    ReconstructionRobustness,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
    UniformSizeControl,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingProvider

from .._runtime import (
    accepted_result_record,
    require_meshcore,
    runtime_identity,
    single_cell_kind,
)


_CENTER = np.asarray((0.07, -0.03, 0.02), dtype=np.float64)
_CAP_DIRECTION = np.asarray((0.6, 0.0, 0.8), dtype=np.float64)
_CAP_DIRECTION /= np.linalg.norm(_CAP_DIRECTION)
_TORUS_RADII = (0.32, 0.17)
_ELLIPSOID_AXES = np.asarray((0.32, 0.24, 0.18), dtype=np.float64)


def _rotation() -> np.ndarray:
    axis = np.ones((3,), dtype=np.float64) / np.sqrt(3.0)
    angle = np.deg2rad(40.0)
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _directions(count: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    radial = np.sqrt(1.0 - z * z)
    return np.stack((radial * np.cos(angle), radial * np.sin(angle), z), axis=1), angle


def _surface_samples(kind: str, count: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "lobed-sphere":
        direction, angle = _directions(count)
        radial_squared = direction[:, 0] ** 2 + direction[:, 1] ** 2
        radius = 0.2 * (1.0 + 0.04 * np.sin(3.0 * angle) * radial_squared)
        return _CENTER + direction * radius[:, None]
    if kind == "torus":
        major, minor = _TORUS_RADII
        accepted: list[np.ndarray] = []
        total = 0
        while total < count:
            u, v, keep = rng.uniform(0.0, 1.0, size=(3, 2 * count))
            u, v = 2.0 * np.pi * u, 2.0 * np.pi * v
            ring = major + minor * np.cos(v)
            selected = keep < ring / (major + minor)
            block = np.stack(
                (ring * np.cos(u), ring * np.sin(u), minor * np.sin(v)), axis=1
            )[selected]
            accepted.append(block)
            total += block.shape[0]
        return _CENTER + np.concatenate(accepted)[:count]
    if kind == "rotated-ellipsoid":
        rotation = _rotation()
        accepted = []
        total = 0
        while total < count:
            direction = rng.normal(size=(2 * count, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            weight = np.min(_ELLIPSOID_AXES) * np.linalg.norm(
                direction / _ELLIPSOID_AXES, axis=1
            )
            chosen = rng.uniform(size=weight.shape) < weight
            block = (direction[chosen] * _ELLIPSOID_AXES) @ rotation.T
            accepted.append(block)
            total += block.shape[0]
        return _CENTER + np.concatenate(accepted)[:count]
    raise ValueError(f"Unknown reconstruction source {kind!r}.")


def _defective_samples(
    kind: str, count: int, seed: int, cap_radius: float
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    clean = _surface_samples(kind, count, rng)
    local = clean - _CENTER
    peak = float(np.max(local @ _CAP_DIRECTION))
    cap_center = peak * _CAP_DIRECTION
    cap = np.linalg.norm(local - cap_center, axis=1) < cap_radius
    kept = ~cap
    x_extent = float(np.max(np.abs(local[:, 0])))
    low_density = local[:, 0] < -0.5 * x_extent
    alternating = (np.arange(count) % 2) == 0
    kept &= ~low_density | alternating
    observed = clean[kept]
    observed = observed + rng.normal(0.0, 0.0008, size=observed.shape)
    outlier_count = max(8, count // 32)
    directions = rng.normal(size=(outlier_count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    maximum_radius = float(np.max(np.linalg.norm(local, axis=1)))
    offsets = np.linspace(0.35, 0.55, outlier_count, dtype=np.float64)
    outliers = _CENTER + directions * (maximum_radius + offsets)[:, None]
    return np.concatenate((observed, outliers)), {
        "seed": seed,
        "source_samples": count,
        "missing_cap_samples": int(np.count_nonzero(cap)),
        "density_band_samples": int(np.count_nonzero(low_density)),
        "retained_before_noise": int(observed.shape[0]),
        "injected_outliers": outlier_count,
        "noise_standard_deviation": 0.0008,
        "cap_direction": _CAP_DIRECTION.tolist(),
        "cap_radius": cap_radius,
        "density_drop": "every second sample in x < -0.5 max(|x|)",
    }


def _source_deviation(kind: str, coordinates: np.ndarray) -> np.ndarray:
    local = coordinates - _CENTER
    if kind == "lobed-sphere":
        radius = np.linalg.norm(local, axis=1)
        angle = np.arctan2(local[:, 1], local[:, 0])
        planar_radius_squared = local[:, 0] ** 2 + local[:, 1] ** 2
        polar_factor = np.divide(
            planar_radius_squared,
            radius**2,
            out=np.zeros_like(radius),
            where=radius > 0,
        )
        expected = 0.2 * (1.0 + 0.04 * np.sin(3.0 * angle) * polar_factor)
        return np.abs(radius - expected)
    if kind == "torus":
        major, minor = _TORUS_RADII
        ring_distance = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2) - major
        return np.abs(np.sqrt(ring_distance**2 + local[:, 2] ** 2) - minor)
    if kind == "rotated-ellipsoid":
        local = local @ _rotation()
        scaled_radius = np.linalg.norm(local / _ELLIPSOID_AXES, axis=1)
        return np.abs(scaled_radius - 1.0) * float(np.max(_ELLIPSOID_AXES))
    raise ValueError(f"Unknown reconstruction source {kind!r}.")


def _affine_harmonic_solve(
    result: phx.meshing.CellMeshingResult, form_id: str, /
) -> dict[str, object]:
    """Solve Laplace with affine Dirichlet data; P1 reproduces it to roundoff."""
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element(
            single_cell_kind(result.mesh, "tetrahedron"), 1
        ),
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        form_id, "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=lambda values: (
            1.0 + 2.0 * values[..., 0] - values[..., 1] + 0.5 * values[..., 2]
        ),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(realization="sparse"),
    )
    # The oracle is roundoff reproduction of affine data, so the Dirichlet-reduced
    # stiffness is handed over as sparse coordinates and factorized by the native
    # sparse LU. The matrix-free action applies the linearized residual program per
    # Krylov step and holds its whole quadrature workset per step. The right-hand
    # side is the residual program at the lift, so the affine reproduction below
    # checks the stored coordinates against it; the residual is read from them.
    operator, rhs = problem.linear_system(storage="sparse")
    prepared = phx.linalg.prepare(
        operator,
        phx.linalg.LinearSolvePolicy(phx.linalg.SparseLU(provider="jax-cpu")),
    )
    solve = phx.linalg.solve(prepared, rhs)
    values = np.asarray(problem.expand(solve.value), dtype=np.float64)
    points = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    exact = 1.0 + 2.0 * points[:, 0] - points[:, 1] + 0.5 * points[:, 2]
    error = float(np.max(np.abs(values - exact), initial=0.0))
    residual = float(jnp.linalg.norm(operator.operator.mv(solve.value) - rhs))
    rhs_norm = float(jnp.linalg.norm(rhs))
    if (
        not bool(jnp.all(solve.successful))
        or values.shape != exact.shape
        or error > 2.0e-9
        or residual > 2.0e-8 * max(1.0, rhs_norm)
    ):
        raise RuntimeError(
            f"{form_id} FE solve failed its affine solution/residual oracle: {error=}, {residual=}, {rhs_norm=}."
        )
    return {
        "form": form.form_id,
        "successful": True,
        "maximum_affine_error": error,
        "normalized_residual": residual / max(1.0, rhs_norm),
        "tolerance": 2.0e-9,
        "solver": {
            "backend": str(prepared.plan.backend),
            "method": str(prepared.plan.method),
            "status": int(np.asarray(solve.status)),
            "unknowns": int(operator.operator.source.size),
        },
    }


def _reconstruct_case(kind: str, scale: int, /) -> dict[str, object]:
    count = 320 * scale
    spacing = 0.08 / scale
    coverage_distance = spacing
    # The missing cap must exceed the declared coverage distance, or no surface
    # point lies farther than it from every sample and the gap is undetectable.
    cap_radius = 1.5 * coverage_distance
    points, input_record = _defective_samples(
        kind, count, 1709 + scale + len(kind), cap_radius
    )
    robustness = ReconstructionRobustness(
        outlier_std_ratio=2.0,
        coverage_distance=coverage_distance,
        incomplete_sampling="close",
        thin_feature_distance=2.0 * spacing,
        component_fit_tolerance=0.4,
    )
    source = reconstruct_surface_region(
        points,
        normal_orientation="propagate",
        recenter=True,
        neighborhood_size=16,
        sample_spacing=spacing,
        screening=4.0,
        discretization="octree",
        robustness=robustness,
        feature_id=f"native-qualification:{kind}:scale-{scale}",
    )
    report = source.report
    expected_euler = 0 if kind == "torus" else 2
    if (
        report.connected_components != 1
        or not report.watertight
        or not report.winding_consistent
        or report.euler_characteristic != expected_euler
        or report.normal_evidence is None
        or report.poisson_evidence is None
        or not report.poisson_evidence.solver_converged
        or report.poisson_evidence.solver_relative_residual > 1.0e-7
        or report.outlier_evidence is None
        or not report.outlier_evidence.removed_indices
        or report.coverage_evidence is None
        or report.coverage_evidence.unsupported_patches < 1
        or report.deviation_evidence is None
    ):
        raise RuntimeError(
            f"{kind} reconstruction failed topology, native-solve, outlier, coverage, or deviation acceptance: {report!r}."
        )
    region = source.source
    source_vertex_values = getattr(region, "vertices", None)
    if source_vertex_values is None:
        raise TypeError("Reconstruction source lost its original point-cloud vertices.")
    source_vertices = np.asarray(source_vertex_values, dtype=np.float64) + np.asarray(
        report.recenter_offset, dtype=np.float64
    )
    source_deviation = float(
        np.max(_source_deviation(kind, source_vertices), initial=0.0)
    )
    source_deviation_tolerance = 0.14 / scale
    if source_deviation > source_deviation_tolerance:
        raise RuntimeError(
            f"{kind} reconstructed boundary deviates {source_deviation:.6g} m from its independent analytic source; tolerance is {source_deviation_tolerance:.6g} m."
        )

    contract = SpatialCoordinateContract.si()
    scope = MeshingScope.whole(source, 2)
    # The reconstructed route conforms to the source triangles and can only
    # red-refine them. They are already finer than 2 * spacing, so a hard
    # target there would demand coarsening; the size is a preference.
    size_control = UniformSizeControl(
        scope, 2.0 * spacing, strength=SizeControlStrength.SOFT
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(size_control,),
        maximum_deviation=source_deviation_tolerance,
    )
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=contract
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    # The reconstructed route certifies fidelity against the reconstructed
    # triangle source itself; it publishes no MeshingDomain chart bank.
    if result.certification is None or result.certification.fidelity is None:
        raise RuntimeError(
            f"{kind} native surface consumption omitted source fidelity evidence."
        )
    # The soft size preference stays unmet, as declared: the result keeps the
    # source triangles unrefined and reports a median edge finer than requested.
    achieved_size = dict(result.compliance.achieved)
    median_edge = achieved_size.get(f"size:{size_control.control_id}:p50_edge")
    if (
        median_edge is None
        or not median_edge < size_control.target_size
        or accepted["cells"] != report.output_cells
    ):
        raise RuntimeError(
            f"{kind} reconstructed surface size evidence changed: median edge {median_edge}, "
            f"target {size_control.target_size}, cells {accepted['cells']} vs source {report.output_cells}."
        )
    mesh_vertices = np.asarray(result.mesh.coordinates, dtype=np.float64) + np.asarray(
        report.recenter_offset, dtype=np.float64
    )
    mesh_deviation = float(np.max(_source_deviation(kind, mesh_vertices), initial=0.0))
    if mesh_deviation > source_deviation_tolerance:
        raise RuntimeError(
            f"{kind} native surface mesh deviates {mesh_deviation:.6g} m from its independent analytic source; tolerance is {source_deviation_tolerance:.6g} m."
        )
    # The native envelope certifies only an offset that dominates its sampling
    # error, sqrt(3) * spacing + certificate_spacing; the certified two-sided
    # bound offset + error must stay within the declared maximum deviation.
    envelope = phx.meshing.SurfaceEnvelopeSource(
        source,
        source_id=f"native-qualification:{kind}:scale-{scale}:volume-envelope",
        coordinate_contract=contract,
        offset=1.5 * spacing,
        spacing=0.5 * spacing,
        certificate_spacing=0.5 * spacing,
        maximum_deviation=2.0 * source_deviation_tolerance,
    )
    volume_scope = MeshingScope.whole(envelope, 2)
    volume_specification = phx.meshing.VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        volume_scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(volume_scope, 2.0 * spacing),),
        maximum_deviation=envelope.maximum_deviation,
    )
    # The envelope volume keeps the scientific tetrahedral quality target; a
    # residual unmet cell is a failed profile, not a relaxed schedule.
    volume_schedule = NativeVolumeSchedule(
        radius_edge_bound=2.0, minimum_dihedral_degrees=10.0
    )
    volume_plan = NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(
            "surface_envelope_tetrahedral", volume_schedule=volume_schedule
        )
    ).plan(envelope, volume_specification, coordinate_contract=contract)
    volume_result = volume_plan.execute()
    volume_accepted = accepted_result_record(volume_result)
    fe = _affine_harmonic_solve(volume_result, f"native-reconstruction-{kind}-harmonic")
    envelope_volume = float(
        np.sum(np.asarray(volume_result.quality.evaluation.measures, dtype=np.float64))
    )
    return {
        "status": "passed",
        "profile": f"{kind}:scale-{scale}",
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_input": input_record,
        "provenance": {
            "kind": "deterministic-analytic-geometry-with-declared-noise-density-gap-and-outliers",
            "seed": input_record["seed"],
            "source_family": kind,
        },
        "oracle": {
            "topological_components": 1,
            "euler_characteristic": expected_euler,
            "watertight": report.watertight,
            "source_deviation_tolerance_m": source_deviation_tolerance,
            "reconstruction_maximum_source_deviation_m": source_deviation,
            "native_surface_mesh_maximum_source_deviation_m": mesh_deviation,
            "poisson_relative_residual": report.poisson_evidence.solver_relative_residual,
            "sample_to_surface_max_m": report.deviation_evidence.sample_to_surface_max,
            "surface_to_sample_max_m": report.deviation_evidence.surface_to_sample_max,
            "unsupported_repair_patches": report.coverage_evidence.unsupported_patches,
            "missing_cap_radius_m": cap_radius,
            "surface_size_preference_m": size_control.target_size,
            "surface_median_edge_m": median_edge,
        },
        "required_features": [
            "no supplied normals",
            "isotropic Gaussian perturbation",
            "one spatial density drop",
            "isolated outliers with independently recorded removal",
            "one incomplete surface cap with reported Poisson closure",
            "connected-component and genus oracle",
            "native reconstruction and source-fidelity-certified native surface meshing",
            "native closed-surface envelope -> tetrahedral mesh -> fresh manufactured FE solve",
        ],
        "native_surface_result": accepted,
        "reconstruction": {
            "algorithm": report.algorithm,
            "input_points": report.input_points,
            "retained_points": report.retained_points,
            "vertices": report.output_vertices,
            "faces": report.output_cells,
            "normal_estimated": report.normal_evidence is not None,
            "outliers_removed": len(report.outlier_evidence.removed_indices),
            "repair_interpretation": report.repair_interpretation,
        },
        "native_volume_consumer": {
            "route": volume_plan.route,
            "envelope_id": envelope.envelope_id,
            "offset_m": envelope.offset,
            "spacing_m": envelope.spacing,
            "maximum_deviation_m": envelope.maximum_deviation,
            "radius_edge_bound": volume_schedule.radius_edge_bound,
            "minimum_dihedral_degrees": volume_schedule.minimum_dihedral_degrees,
            "measured_volume_m3": envelope_volume,
            "accepted": volume_accepted,
            "fe": fe,
        },
        "runtime": runtime_identity(),
    }


def native_reconstruction() -> dict[str, object]:
    meshcore = require_meshcore()
    cases: list[dict[str, object]] = []
    failures: list[str] = []
    for kind in ("lobed-sphere", "torus", "rotated-ellipsoid"):
        for scale in (1, 2):
            try:
                cases.append(_reconstruct_case(kind, scale))
            except Exception as error:
                failures.append(f"{kind}:scale-{scale}: {type(error).__name__}: {error}")
    return {
        "status": "failed" if failures else "passed",
        "workflow": "native noisy-surface reconstruction -> certified native surface meshing",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#imperfect-surface",
        "cases": cases,
        "failures": failures,
        "meshcore_identity": meshcore,
        "runtime": runtime_identity(),
    }


# Donor `scenarios/envelope.py` open-sheet fixture (commit c56614a9f): one open
# triangle with an authored feature edge, thickened as the sampled
# unsigned-distance sublevel with rounding and topology change permitted.
_ENVELOPE_SHEET_VERTICES = ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0))
_ENVELOPE_SHEET_TRIANGLES = ((0, 1, 2),)
_ENVELOPE_OFFSET = 0.263
_ENVELOPE_MAXIMUM_DEVIATION = 0.55
_ENVELOPE_RESOLUTION = 4


def _point_triangle_distances(points: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    """Brute-force distance from points to a triangle soup by Voronoi-region tests."""
    p = points[:, None, :]
    a, b, c = (triangles[None, :, k, :] for k in range(3))
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, axis=-1), np.sum(ac * ap, axis=-1)
    bp, cp = p - b, p - c
    d3, d4 = np.sum(ab * bp, axis=-1), np.sum(ac * bp, axis=-1)
    d5, d6 = np.sum(ab * cp, axis=-1), np.sum(ac * cp, axis=-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = va + vb + vc
        closest = (
            a + (vb / denominator)[..., None] * ab + (vc / denominator)[..., None] * ac
        )
        on_ab = a + (d1 / (d1 - d3))[..., None] * ab
        on_ac = a + (d2 / (d2 - d6))[..., None] * ac
        on_bc = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None] * (c - b)
    conditions = (
        (d1 <= 0.0) & (d2 <= 0.0),
        (d3 >= 0.0) & (d4 <= d3),
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
        (d6 >= 0.0) & (d5 <= d6),
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
        (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
    )
    choices = (a, b, on_ab, c, on_ac, on_bc)
    for condition, choice in zip(reversed(conditions), reversed(choices), strict=True):
        closest = np.where(
            condition[..., None], np.broadcast_to(choice, closest.shape), closest
        )
    return np.min(np.linalg.norm(p - closest, axis=-1), axis=1)


def _triangle_samples(triangles: np.ndarray, divisions: int, /) -> np.ndarray:
    weights = np.asarray(
        [
            (i, j, divisions - i - j)
            for i in range(divisions + 1)
            for j in range(divisions + 1 - i)
        ],
        dtype=np.float64,
    ) / float(divisions)
    return np.einsum("wk,tkd->twd", weights, triangles).reshape((-1, 3))


def _flat_convex_offset_volume(area: float, perimeter: float, radius: float, /) -> float:
    """Steiner volume of the closed ``radius`` neighbourhood of a planar convex set."""
    return (
        2.0 * area * radius
        + 0.5 * np.pi * perimeter * radius**2
        + 4.0 / 3.0 * np.pi * radius**3
    )


def native_envelope() -> dict[str, object]:
    meshcore = require_meshcore()
    vertices = np.asarray(_ENVELOPE_SHEET_VERTICES, dtype=np.float64)
    triangles = np.asarray(_ENVELOPE_SHEET_TRIANGLES, dtype=np.int64)
    sheet = AuthoredTriangleSheet(
        vertices, triangles, feature_id="qualification-open-envelope-sheet"
    )
    repair = sheet.repair
    if (
        sheet.report.watertight
        or not sheet.report.winding_consistent
        or sheet.report.connected_components != 1
        or sheet.report.euler_characteristic != 1
        or repair.boundary_edges != 3
        or repair.repaired_triangles != 1
    ):
        raise RuntimeError(
            f"Authored open sheet repair disagrees with its open one-triangle topology: {repair!r}, {sheet.report!r}."
        )
    contract = SpatialCoordinateContract.si()
    source = phx.meshing.SurfaceEnvelopeSource(
        sheet,
        source_id="native-qualification:open-sheet-envelope",
        coordinate_contract=contract,
        offset=_ENVELOPE_OFFSET,
        spacing=0.44 / _ENVELOPE_RESOLUTION,
        certificate_spacing=0.2 / _ENVELOPE_RESOLUTION,
        maximum_deviation=_ENVELOPE_MAXIMUM_DEVIATION,
    )
    # bounds[0]: input-to-envelope upper; bounds[2] <= envelope-to-input <= bounds[1].
    certified_bounds = np.asarray(source.result.bounds, dtype=np.float64)
    input_to_envelope_bound = float(certified_bounds[0])
    envelope_upper_bound = float(certified_bounds[1])
    envelope_lower_bound = float(certified_bounds[2])
    # Independent two-sided deviation against the AUTHORED triangles.
    authored_triangles = vertices[triangles]
    envelope_vertices = np.asarray(source.domain.vertices, dtype=np.float64)
    envelope_triangles = envelope_vertices[
        np.asarray(source.domain.facets, dtype=np.int64)
    ]
    input_to_envelope = float(
        np.max(
            _point_triangle_distances(
                _triangle_samples(authored_triangles, 16), envelope_triangles
            )
        )
    )
    envelope_to_input = _point_triangle_distances(
        np.concatenate((envelope_vertices, envelope_triangles.mean(axis=1))),
        authored_triangles,
    )
    roundoff = 1.0e-12
    if (
        input_to_envelope > input_to_envelope_bound + roundoff
        or float(np.min(envelope_to_input)) < envelope_lower_bound - roundoff
        or float(np.max(envelope_to_input)) > envelope_upper_bound + roundoff
        or max(input_to_envelope_bound, envelope_upper_bound)
        > _ENVELOPE_MAXIMUM_DEVIATION
        or envelope_lower_bound <= 0.0
    ):
        raise RuntimeError(
            "Open-sheet envelope violates its independent input/envelope deviation oracle: "
            f"{input_to_envelope=}, envelope_to_input=[{np.min(envelope_to_input)}, {np.max(envelope_to_input)}], "
            f"certified=({input_to_envelope_bound}, {envelope_upper_bound}, {envelope_lower_bound})."
        )
    scope = MeshingScope.whole(source, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, source.spacing),),
        maximum_deviation=source.maximum_deviation,
    )
    plan = NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("surface_envelope_tetrahedral")
    ).plan(source, specification, coordinate_contract=contract)
    result = plan.execute()
    accepted = accepted_result_record(result)
    achieved = dict(result.compliance.achieved)
    repair_quantities = {
        name: achieved.get(f"source_topology_repair:{name}")
        for name, _ in repair.quantities
    }
    if any(repair_quantities[name] != float(value) for name, value in repair.quantities):
        raise RuntimeError(
            f"Envelope volume result lost the authored topology-repair interpretation: {repair_quantities}."
        )
    measures = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    volume = float(np.sum(measures))
    edge_lengths = np.linalg.norm(
        authored_triangles[0] - np.roll(authored_triangles[0], -1, axis=0), axis=1
    )
    area = 0.5 * float(
        np.linalg.norm(
            np.cross(
                authored_triangles[0, 1] - authored_triangles[0, 0],
                authored_triangles[0, 2] - authored_triangles[0, 0],
            )
        )
    )
    perimeter = float(np.sum(edge_lengths))
    # The meshed region contains {d < lower} and lies inside {d <= upper}.
    volume_lower = _flat_convex_offset_volume(area, perimeter, envelope_lower_bound)
    volume_upper = _flat_convex_offset_volume(area, perimeter, envelope_upper_bound)
    if np.any(measures <= 0.0) or not volume_lower <= volume <= volume_upper:
        raise RuntimeError(
            f"Envelope tetrahedra integrate {volume} m^3 outside the Steiner offset bracket [{volume_lower}, {volume_upper}] m^3."
        )
    fe = _affine_harmonic_solve(result, "native-open-sheet-envelope-harmonic")
    return {
        "status": "passed",
        "workflow": "native imperfect-surface envelope -> tetrahedral volume mesh",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#imperfect-surface",
        "profile": "native-envelope-lifecycle",
        "source_input": {
            "vertices": vertices.tolist(),
            "triangles": triangles.tolist(),
            "provenance": "PHYDRA synthetic open sheet; donor envelope qualification fixture",
            "interpretation": "sampled unsigned-distance sublevel; feature rounding and topology change permitted",
        },
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "original_source_id": sheet.source_id,
        "original_source_revision": sheet.source_revision,
        "repair_interpretation": sheet.report.repair_interpretation,
        "topology_repair": dict(repair.quantities),
        "envelope": {
            "envelope_id": source.envelope_id,
            "offset_m": source.offset,
            "spacing_m": source.spacing,
            "certificate_spacing_m": source.certificate_spacing,
            "maximum_deviation_m": source.maximum_deviation,
            "certified_input_to_envelope_upper_m": input_to_envelope_bound,
            "certified_envelope_to_input_bounds_m": [
                envelope_lower_bound,
                envelope_upper_bound,
            ],
        },
        "oracle": {
            "independent_input_to_envelope_max_m": input_to_envelope,
            "independent_envelope_to_input_min_m": float(np.min(envelope_to_input)),
            "independent_envelope_to_input_max_m": float(np.max(envelope_to_input)),
            "measured_volume_m3": volume,
            "steiner_volume_bracket_m3": [volume_lower, volume_upper],
        },
        "native_volume_result": accepted,
        "fe": fe,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
