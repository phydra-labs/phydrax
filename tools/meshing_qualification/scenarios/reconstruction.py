from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.reconstruction import (
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
from phydrax.meshing.providers import NativeMeshingProvider

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


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
    kind: str, count: int, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    clean = _surface_samples(kind, count, rng)
    local = clean - _CENTER
    peak = float(np.max(local @ _CAP_DIRECTION))
    cap_center = peak * _CAP_DIRECTION
    cap = np.linalg.norm(local - cap_center, axis=1) < 0.06
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
        "cap_radius": 0.06,
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


def _reconstruct_case(kind: str, scale: int, /) -> dict[str, object]:
    count = 320 * scale
    spacing = 0.08 / scale
    coverage_distance = 2.0 * spacing
    points, input_record = _defective_samples(kind, count, 1709 + scale + len(kind))
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
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 2.0 * spacing),),
        maximum_deviation=source_deviation_tolerance,
    )
    plan = NativeMeshingProvider().plan(
        source, specification, coordinate_contract=contract
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    if (
        result.surface_source is None
        or result.certification is None
        or result.certification.fidelity is None
    ):
        raise RuntimeError(
            f"{kind} native surface consumption omitted source fidelity evidence."
        )
    mesh_vertices = np.asarray(result.mesh.coordinates, dtype=np.float64) + np.asarray(
        report.recenter_offset, dtype=np.float64
    )
    mesh_deviation = float(np.max(_source_deviation(kind, mesh_vertices), initial=0.0))
    if mesh_deviation > source_deviation_tolerance:
        raise RuntimeError(
            f"{kind} native surface mesh deviates {mesh_deviation:.6g} m from its independent analytic source; tolerance is {source_deviation_tolerance:.6g} m."
        )
    envelope = phx.meshing.SurfaceEnvelopeSource(
        source,
        source_id=f"native-qualification:{kind}:scale-{scale}:volume-envelope",
        coordinate_contract=contract,
        offset=0.02 / scale,
        spacing=2.0 * spacing,
        certificate_spacing=spacing,
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
    volume_plan = NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("surface_envelope_tetrahedral")
    ).plan(envelope, volume_specification, coordinate_contract=contract)
    volume_result = volume_plan.execute()
    volume_accepted = accepted_result_record(volume_result)
    fe_field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    fe_space = phx.discretization.FiniteElementPlan(
        volume_result.mesh,
        fe_field,
        coordinate_spec=volume_result.geometry,
    ).prepare()
    fe_constraint = phx.discretization.dirichlet_constraint(fe_space, "u")
    fe_form = phx.equations.FiniteElementForm(
        f"native-reconstruction-{kind}-harmonic",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
    )
    fe_problem = phx.equations.compile_finite_element_problem(
        fe_form,
        fe_space,
        constraint=fe_constraint,
        dirichlet_values=lambda values: (
            1.0 + 2.0 * values[..., 0] - values[..., 1] + 0.5 * values[..., 2]
        ),
    )
    fe_operator, fe_rhs = fe_problem.linear_system()
    fe_solve = phx.linalg.solve(fe_operator, fe_rhs)
    fe_values = np.asarray(fe_problem.expand(fe_solve.value), dtype=np.float64)
    fe_points = np.asarray(fe_space.dof_maps[0].dof_coordinates, dtype=np.float64)
    fe_exact = 1.0 + 2.0 * fe_points[:, 0] - fe_points[:, 1] + 0.5 * fe_points[:, 2]
    fe_error = float(np.max(np.abs(fe_values - fe_exact), initial=0.0))
    fe_residual = float(jnp.linalg.norm(fe_problem.residual(fe_solve.value)))
    fe_rhs_norm = float(jnp.linalg.norm(fe_rhs))
    if (
        not bool(jnp.all(fe_solve.successful))
        or fe_values.shape != fe_exact.shape
        or fe_error > 2.0e-9
        or fe_residual > 2.0e-8 * max(1.0, fe_rhs_norm)
    ):
        raise RuntimeError(
            f"{kind} reconstructed-envelope FE solve failed its affine solution/residual oracle: {fe_error=}, {fe_residual=}, {fe_rhs_norm=}."
        )
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
            "measured_volume_m3": envelope_volume,
            "accepted": volume_accepted,
            "fe_form": fe_form.form_id,
            "fe_successful": bool(jnp.all(fe_solve.successful)),
            "fe_maximum_affine_error": fe_error,
            "fe_normalized_residual": fe_residual / max(1.0, fe_rhs_norm),
            "fe_tolerance": 2.0e-9,
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


def native_envelope() -> dict[str, object]:
    meshcore = require_meshcore()
    points, input_record = _defective_samples("lobed-sphere", 320, 1723)
    reconstructed = reconstruct_surface_region(
        points,
        neighborhood_size=16,
        sample_spacing=0.08,
        robustness=ReconstructionRobustness(
            outlier_std_ratio=2.0,
            coverage_distance=0.16,
            incomplete_sampling="close",
            thin_feature_distance=0.16,
            component_fit_tolerance=0.4,
        ),
        feature_id="native-qualification:imperfect-envelope-source",
    )
    source = phx.meshing.SurfaceEnvelopeSource(
        reconstructed,
        source_id="native-qualification:reconstructed-surface-envelope",
        coordinate_contract=SpatialCoordinateContract.si(),
        offset=0.02,
        spacing=0.08,
        certificate_spacing=0.04,
        maximum_deviation=0.16,
    )
    scope = MeshingScope.whole(source, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 0.12),),
        maximum_deviation=source.maximum_deviation,
    )
    plan = NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("surface_envelope_tetrahedral")
    ).plan(source, specification, coordinate_contract=source.coordinate_contract)
    result = plan.execute()
    accepted = accepted_result_record(result)
    envelope_record = {
        "status": "passed",
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "envelope_id": source.envelope_id,
        "offset_m": source.offset,
        "spacing_m": source.spacing,
        "certificate_spacing_m": source.certificate_spacing,
        "maximum_deviation_m": source.maximum_deviation,
        "native_volume_result": accepted,
        "independent_volume_m3": float(
            np.sum(np.asarray(result.quality.evaluation.measures))
        ),
    }
    # The recovered contract requires repair of an imperfect open sheet. The
    # current SurfaceEnvelopeSource accepts only a reconstructed closed MeshRegion;
    # keep that missing source representation visible as a failing gate.
    return {
        "status": "failed",
        "workflow": "native imperfect-surface envelope -> tetrahedral volume mesh",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#imperfect-surface",
        "profile": "native-envelope-lifecycle",
        "actual_closed_surface_envelope": envelope_record,
        "source_input": input_record,
        "missing_production_contracts": [
            "native envelope construction directly from the authored imperfect open triangle sheet, with explicit topology-repair interpretation and independent input-to-envelope fidelity oracle"
        ],
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
