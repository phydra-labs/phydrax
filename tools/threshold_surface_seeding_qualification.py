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
    MultiRegionSurfaceCapacityPlan,
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


def _capacity(name: str, resolution: int) -> MultiRegionSurfaceCapacityPlan:
    scale = resolution**2
    return MultiRegionSurfaceCapacityPlan(
        vertex_capacity=160 * scale,
        edge_capacity=480 * scale,
        face_capacity=320 * scale,
        region_capacity=8,
        region_pair_capacity=28,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=6,
        resource_id=name,
    )


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
        _capacity(f"sphere-{resolution}", resolution),
        spacing=(spacing, spacing, spacing),
        origin=(0.5 * spacing, 0.5 * spacing, 0.5 * spacing),
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


def _junction_and_preparation() -> dict[str, Any]:
    resolution = 3
    labels = _three_cell_labels(resolution)
    spacing = 1.0 / resolution
    plan = LabelFieldSurfaceExtractionPlan(
        ("cell-a", "cell-b", "cell-c"),
        _capacity("junction", resolution),
        spacing=(spacing, spacing, spacing),
        origin=(0.5 * spacing, 0.5 * spacing, 0.5 * spacing),
        validation_policy=MultiRegionSurfaceValidationPolicy(profile="dry_foam"),
    )
    prepared = plan.prepare(labels.shape)
    state = _state(labels, ("cell-a", "cell-b", "cell-c"))
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
        "equilibrium_prepared": equilibrium_prepared,
        "passed": passed,
    }


def _refusals() -> dict[str, Any]:
    labels = np.indices((2, 2, 2)).sum(axis=0).astype(np.int32) % 3
    state = _state(labels, ("a", "b", "c"))
    ambiguity = (
        LabelFieldSurfaceExtractionPlan(
            ("a", "b", "c"),
            _capacity("ambiguity", 2),
            spacing=(1.0, 1.0, 1.0),
            maximum_ambiguous_cells=0,
        )
        .prepare(labels.shape)
        .extract(state)
    )
    tiny = MultiRegionSurfaceCapacityPlan(
        vertex_capacity=1,
        edge_capacity=1,
        face_capacity=1,
        region_capacity=4,
        region_pair_capacity=6,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=6,
        resource_id="capacity-refusal",
    )
    capacity = (
        LabelFieldSurfaceExtractionPlan(("a", "b", "c"), tiny, spacing=(1.0, 1.0, 1.0))
        .prepare(labels.shape)
        .extract(state)
    )
    passed = bool(
        ambiguity.evidence.status
        is LabelFieldSurfaceExtractionStatus.AMBIGUITY_CAPACITY_EXCEEDED
        and ambiguity.surface is None
        and capacity.evidence.status
        is LabelFieldSurfaceExtractionStatus.CAPACITY_EXCEEDED
        and capacity.surface is None
        and not capacity.evidence.capacity.admitted
    )
    return {
        "ambiguity_status": ambiguity.evidence.status.name,
        "ambiguous_cells": ambiguity.evidence.ambiguous_cell_count,
        "capacity_status": capacity.evidence.status.name,
        "capacity_exceeded": capacity.evidence.capacity.exceeded,
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
    refusals = _refusals()
    successful = bool(sphere_passed and junction["passed"] and refusals["passed"])
    return {
        "runtime": _runtime(),
        "source": {
            "algorithm": (
                "Junction-conforming multi-label marching tetrahedra on a "
                "Freudenthal cube split"
            ),
            "reference": ("Da, Batty and Grinspun, ACM TOG 33, 112 (2014), ideas only"),
            "copying_permitted": False,
        },
        "sphere_convergence": sphere,
        "junction_and_foam_preparation": junction,
        "resource_refusals": refusals,
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
