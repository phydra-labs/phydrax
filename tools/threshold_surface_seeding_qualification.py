"""Qualification campaign for threshold-label to explicit-surface seeding."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.foams import (
    FoamEquilibriumPlan,
    FoamMaterialPlan,
    PreparedFoamEquilibrium,
)
from phydrax.geometry.multiregion_surface import (
    LabelFieldSurfaceExtractionPlan,
    LabelFieldSurfaceExtractionStatus,
    MultiRegionSurfaceValidationPolicy,
)
from phydrax.qualification import QualificationRuntimeIdentity
from phydrax.threshold_dynamics import LabelFieldState


RADIUS = 0.3


def _runtime() -> dict[str, Any]:
    identity = QualificationRuntimeIdentity(
        f"phydrax-{importlib.metadata.version('phydrax')}",
        f"python-{platform.python_version()}-jax-{jax.__version__}-"
        f"numpy-{np.__version__}",
        jax.default_backend(),
        f"devices-{jax.device_count()}",
        "float64-host-certification",
    )
    return dict(identity.to_record())


def _image_affine(
    spacing: float, *, origin: tuple[float, float, float] | None = None
) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] *= spacing
    affine[:3, 3] = (0.5 * spacing,) * 3 if origin is None else np.asarray(origin)
    return affine


def _state(labels: np.ndarray, label_ids: tuple[str, ...]) -> LabelFieldState:
    counts = np.bincount(labels.reshape(-1), minlength=len(label_ids))
    return LabelFieldState(
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray(counts > 0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        label_ids=label_ids,
        route_id="surface-seeding-qualification-route",
        prepared_id="surface-seeding-qualification-prepared",
        site_id="surface-seeding-qualification-sites",
    )


def _sphere_labels(resolution: int) -> tuple[np.ndarray, float]:
    spacing = 1.0 / resolution
    axis = (np.arange(resolution) + 0.5) * spacing
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return (
        (np.linalg.norm(points - 0.5, axis=-1) >= RADIUS).astype(np.int32),
        spacing,
    )


def _sphere_row(resolution: int) -> dict[str, Any]:
    labels, spacing = _sphere_labels(resolution)
    plan = LabelFieldSurfaceExtractionPlan(
        ("bubble", "ambient"),
        index_to_world=_image_affine(spacing),
        boundary_region_id="ambient",
        validation_policy=MultiRegionSurfaceValidationPolicy(
            profile="manifold_two_region"
        ),
    )
    result = plan.prepare(labels.shape).extract(_state(labels, ("bubble", "ambient")))
    volume = result.evidence.extracted_finite_region_volumes[0]
    area = result.evidence.extracted_pair_areas[0]
    return {
        "resolution": resolution,
        "status": result.evidence.status.name,
        "accepted": result.accepted,
        "vertices": result.evidence.capacity.required.vertex,
        "faces": result.evidence.capacity.required.face,
        "ambiguous_cells": result.evidence.ambiguous_cell_count,
        "volume": volume,
        "volume_relative_error": abs(volume / (4.0 * np.pi * RADIUS**3 / 3.0) - 1.0),
        "area": area,
        "area_relative_error": abs(area / (4.0 * np.pi * RADIUS**2) - 1.0),
        "collision_certified": result.evidence.collision_certified,
        "lineage_id": result.lineage.lineage_id,
        "source_binding_id": result.lineage.source_binding_id,
        "evidence_id": result.evidence.evidence_id,
    }


def _three_cell_labels(resolution: int) -> np.ndarray:
    spacing = 1.0 / resolution
    axis = (np.arange(resolution) + 0.5) * spacing
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    centers = np.asarray(((0.25, 0.25, 0.5), (0.75, 0.25, 0.5), (0.5, 0.78, 0.5)))
    return np.argmin(
        np.sum((points[..., None, :] - centers) ** 2, axis=-1), axis=-1
    ).astype(np.int32)


def _reflection_shear() -> np.ndarray:
    return np.asarray(
        (
            (-1.0, 0.20, 0.05, 1.2),
            (0.15, 0.90, 0.25, -0.4),
            (0.02, 0.10, 1.10, 0.7),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _junction_and_preparation() -> dict[str, Any]:
    resolution = 3
    labels = _three_cell_labels(resolution)
    spacing = 1.0 / resolution
    ids = ("cell-a", "cell-b", "cell-c")
    policy = MultiRegionSurfaceValidationPolicy(profile="dry_foam")
    base_affine = _image_affine(spacing)
    prepared = LabelFieldSurfaceExtractionPlan(
        ids, index_to_world=base_affine, validation_policy=policy
    ).prepare(labels.shape)
    state = _state(labels, ids)
    result = prepared.extract(state)
    repeated = prepared.extract(state)
    validation = result.evidence.validation
    equilibrium_prepared = False
    if result.accepted and result.topology is not None and result.state is not None:
        if result.surface is None:
            raise RuntimeError("Accepted extraction did not expose its surface.")
        material = FoamMaterialPlan.soap_film(result.topology.region_ids, 0.025)
        equilibrium = PreparedFoamEquilibrium(
            FoamEquilibriumPlan(), result.surface, material, result.state
        )
        equilibrium_prepared = (
            equilibrium.surface.prepared_id == result.surface.prepared_id
        )
    deterministic = (
        result.candidate_seed is not None
        and repeated.candidate_seed is not None
        and result.candidate_seed.seed_id == repeated.candidate_seed.seed_id
        and result.lineage.lineage_id == repeated.lineage.lineage_id
    )
    transformed = (
        LabelFieldSurfaceExtractionPlan(
            ids,
            index_to_world=_reflection_shear() @ base_affine,
            validation_policy=policy,
        )
        .prepare(labels.shape)
        .extract(state)
    )
    affine_covariant = False
    if (
        result.accepted
        and transformed.accepted
        and result.candidate_seed is not None
        and transformed.candidate_seed is not None
    ):
        transform = _reflection_shear()
        expected = (
            np.asarray(result.candidate_seed.positions) @ transform[:3, :3].T
            + transform[:3, 3]
        )
        affine_covariant = bool(
            np.allclose(
                transformed.candidate_seed.positions,
                expected,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.array_equal(
                np.sort(transformed.candidate_seed.faces, axis=1),
                np.sort(result.candidate_seed.faces, axis=1),
            )
            and np.array_equal(
                transformed.candidate_seed.face_labels,
                result.candidate_seed.face_labels,
            )
        )
    passed = bool(
        result.accepted
        and validation is not None
        and validation.label_orientation_consistent
        and validation.maximum_edge_valence == 3
        and validation.maximum_vertex_regions == 4
        and result.evidence.ambiguous_cell_count > 0
        and result.evidence.ambiguity_resolved
        and result.evidence.collision_certified
        and deterministic
        and equilibrium_prepared
        and affine_covariant
    )
    return {
        "status": result.evidence.status.name,
        "accepted": result.accepted,
        "ambiguous_cells": result.evidence.ambiguous_cell_count,
        "multi_label_tetrahedra": result.evidence.multi_label_tetrahedron_count,
        "maximum_edge_valence": None
        if validation is None
        else validation.maximum_edge_valence,
        "maximum_vertex_regions": None
        if validation is None
        else validation.maximum_vertex_regions,
        "orientation_consistent": False
        if validation is None
        else validation.label_orientation_consistent,
        "collision_certified": result.evidence.collision_certified,
        "deterministic": deterministic,
        "affine_covariant": affine_covariant,
        "equilibrium_prepared": equilibrium_prepared,
        "passed": passed,
    }


def _tiny_material() -> dict[str, Any]:
    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[2, 2, 2] = 1
    ids = ("ambient", "tiny-material")
    affine = _reflection_shear() @ _image_affine(0.2)
    result = (
        LabelFieldSurfaceExtractionPlan(
            ids,
            index_to_world=affine,
            boundary_region_id="ambient",
            validation_policy=MultiRegionSurfaceValidationPolicy(
                profile="manifold_two_region"
            ),
        )
        .prepare(labels.shape)
        .extract(_state(labels, ids))
    )
    retained = (
        result.accepted
        and result.lineage.source_label_counts == (124, 1)
        and "tiny-material" in result.lineage.region_ids
        and result.candidate_seed is not None
        and bool(np.any(np.all(result.candidate_seed.face_labels == (0, 1), axis=1)))
    )
    determinant = abs(float(np.linalg.det(affine[:3, :3])))
    measure = bool(
        retained
        and np.isclose(
            result.evidence.source_finite_region_volumes[0],
            determinant,
            rtol=1.0e-12,
        )
    )
    return {
        "status": result.evidence.status.name,
        "accepted": result.accepted,
        "source_label_counts": result.lineage.source_label_counts,
        "tiny_material_retained": retained,
        "affine_voxel_measure_preserved": measure,
        "passed": bool(retained and measure),
    }


def _source_semantics() -> dict[str, Any]:
    labels = np.indices((2, 2, 2)).sum(axis=0).astype(np.int32) % 3
    state = _state(labels, ("a", "b", "c"))
    unsupported = (
        LabelFieldSurfaceExtractionPlan(
            ("a", "b", "c"),
            index_to_world=np.eye(4),
            validation_policy=MultiRegionSurfaceValidationPolicy(
                profile="manifold_two_region"
            ),
        )
        .prepare(labels.shape)
        .extract(state)
    )
    incomplete = np.asarray(((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)))
    try:
        LabelFieldSurfaceExtractionPlan(
            ("a", "b", "c"), index_to_world=np.eye(4)
        ).prepare((2, 2, 2), site_coordinates=incomplete)
    except ValueError as error:
        missing_samples_rejected = "missing categorical samples" in str(error)
    else:
        missing_samples_rejected = False
    passed = bool(
        unsupported.evidence.status
        is LabelFieldSurfaceExtractionStatus.UNSUPPORTED_VALENCE
        and unsupported.surface is None
        and missing_samples_rejected
    )
    return {
        "unsupported_valence_status": unsupported.evidence.status.name,
        "unknown_sparse_samples_rejected": missing_samples_rejected,
        "passed": passed,
    }


def run() -> dict[str, Any]:
    rows = [_sphere_row(resolution) for resolution in (5, 7, 9)]
    volume_errors = [row["volume_relative_error"] for row in rows]
    area_errors = [row["area_relative_error"] for row in rows]
    volume_monotone = all(
        volume_errors[index + 1] < volume_errors[index] for index in range(len(rows) - 1)
    )
    area_monotone = all(
        area_errors[index + 1] < area_errors[index] for index in range(len(rows) - 1)
    )
    resolution_ratio = rows[-1]["resolution"] / rows[0]["resolution"]
    volume_endpoint_order = float(
        np.log(volume_errors[0] / volume_errors[-1]) / np.log(resolution_ratio)
    )
    area_endpoint_order = float(
        np.log(area_errors[0] / area_errors[-1]) / np.log(resolution_ratio)
    )
    sphere_passed = bool(
        all(row["accepted"] and row["collision_certified"] for row in rows)
        and volume_endpoint_order > 0.0
        and area_endpoint_order > 0.0
        and volume_errors[-1] < 0.1
        and area_errors[-1] < 0.25
    )
    sphere = {
        "reference": "voxelized sphere: V=4 pi R^3/3, A=4 pi R^2",
        "rows": rows,
        "volume_observed_orders": [
            float(
                np.log(volume_errors[index] / volume_errors[index + 1])
                / np.log(rows[index + 1]["resolution"] / rows[index]["resolution"])
            )
            for index in range(len(rows) - 1)
        ],
        "area_observed_orders": [
            float(
                np.log(area_errors[index] / area_errors[index + 1])
                / np.log(rows[index + 1]["resolution"] / rows[index]["resolution"])
            )
            for index in range(len(rows) - 1)
        ],
        "volume_endpoint_order": volume_endpoint_order,
        "area_endpoint_order": area_endpoint_order,
        "volume_error_monotone": volume_monotone,
        "area_error_monotone": area_monotone,
        "admission": {
            "finest_volume_relative_error": 0.1,
            "finest_area_relative_error": 0.25,
            "positive_endpoint_orders": True,
            "adjacent_monotonicity_required": False,
        },
        "passed": sphere_passed,
    }
    junction = _junction_and_preparation()
    source_semantics = _source_semantics()
    tiny_material = _tiny_material()
    successful = bool(
        sphere_passed
        and junction["passed"]
        and source_semantics["passed"]
        and tiny_material["passed"]
    )
    return {
        "runtime": _runtime(),
        "source": {
            "algorithm": (
                "Junction-conforming multi-label marching tetrahedra on a "
                "Freudenthal cube split"
            ),
            "semantics": "categorical labels with full image index-to-world affine",
            "reference": ("Da, Batty and Grinspun, ACM TOG 33, 112 (2014), ideas only"),
            "copying_permitted": False,
        },
        "sphere_convergence": sphere,
        "junction_and_foam_preparation": junction,
        "source_semantics": source_semantics,
        "tiny_material": tiny_material,
        "successful": successful,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n")
    print(encoded)
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
