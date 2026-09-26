#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Lagrangian motion, monitored remeshing, and conservative remap of a cell field.

A graded triangulated annulus carries a conserved finite-volume density: every
cell keeps its mass while the mesh moves, so the averages change only with the
cell areas. The inner ring (a stirring obstacle) rotates a little every step. A
``FiniteElementMeshMotionPlan`` realizes the rotated inner boundary and extends
it into the interior through the explicit stiffened ``LINEAR_ELASTICITY`` route,
and ``advance_mesh_motion`` assesses every proposal with a ``MeshMotionMonitor``
under an explicit ``MeshMotionMonitorPolicy``: the growing twist first degrades
the cell quality below the relocation threshold, and when fixed-topology
relocation cannot restore it the advance remeshes through the planar
``NATIVE_METRIC_2D`` route toward the reference cell size.

Relocated and remeshed meshes cover exactly the moved domain, so the density is
remapped on their certified common refinement: the bound-preserving limited P1
remap (``UnstructuredSecondOrderRemapPlan``) is the transfer that the loop
carries forward; the first-order P0 remap and the Galerkin L2 projection of the
piecewise-constant field (``prepare_l2_projection_target`` and
``prepare_l2_projection_transfer``) are compared on the same refinement. After a
remesh the adapted mesh becomes the new motion and monitor reference.

Run with the native meshing core available (``phydrax[meshcore]`` or
``PHYDRAX_MESHCORE_LIBRARY``)::

    python examples/ale_conservative_remesh.py
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


D = phx.discretization
INNER_RADIUS, OUTER_RADIUS = 0.3, 1.0
STEP_ANGLE = np.deg2rad(12.0)
STEPS = 10


def annulus(rings: int, sectors: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Graded polar triangulation whose smallest cells hug the obstacle."""
    radii = INNER_RADIUS + (OUTER_RADIUS - INNER_RADIUS) * (
        np.linspace(0.0, 1.0, rings + 1) ** 1.5
    )
    angles = 2.0 * np.pi * np.arange(sectors) / sectors
    points = np.stack(
        (
            np.outer(radii, np.cos(angles)).ravel(),
            np.outer(radii, np.sin(angles)).ravel(),
        ),
        axis=1,
    )
    index = np.arange(points.shape[0], dtype=np.int32).reshape(rings + 1, sectors)
    following = np.roll(index, -1, axis=1)
    a, b = index[:-1].ravel(), following[:-1].ravel()
    c, d = following[1:].ravel(), index[1:].ravel()
    return points, np.concatenate((np.stack((a, d, c), 1), np.stack((a, c, b), 1)))


class RotatingObstacle(phx.StrictModule):
    """Boundary provider: the inner ring rotates by the design angle."""

    reference_points: jax.Array
    inner: jax.Array
    mapping_id: str = eqx.field(static=True)

    def __init__(self, points: np.ndarray, /):
        radius = np.linalg.norm(points, axis=1)
        self.reference_points = jnp.asarray(points, dtype=jnp.float64)
        self.inner = jnp.asarray(radius < 0.5 * (INNER_RADIUS + OUTER_RADIUS))
        self.mapping_id = "rotating-obstacle"

    def rotated(self, angle: jax.Array, /) -> jax.Array:
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        rotation = jnp.stack((jnp.stack((cosine, -sine)), jnp.stack((sine, cosine))))
        turned = phx.ein.contract("ij,pj->pi", rotation, self.reference_points)
        return jnp.where(self.inner[:, None], turned, self.reference_points)

    def realize(self, angle: jax.Array, /) -> D.FiniteElementBoundaryRealization:
        points = self.rotated(angle)
        return D.FiniteElementBoundaryRealization(
            points,
            points,
            accepted=True,
            refresh_required=False,
            status=0,
            mapping_id=self.mapping_id,
        )


def motion_plan(mesh: D.CellMesh, /) -> D.FiniteElementMeshMotionPlan:
    boundary = np.asarray(mesh.topology.entities(0).subset("boundary").mask)
    discretization = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("x", D.lagrange_element("triangle", 1))
    ).prepare()
    return D.FiniteElementMeshMotionPlan(
        discretization,
        RotatingObstacle(np.asarray(mesh.coordinates)[boundary]),
        policy=D.FiniteElementMeshMotionPolicy(
            route=D.FiniteElementMeshMotionRoute.LINEAR_ELASTICITY,
            stiffening_exponent=1.0,
            validity=D.MotionValidityPolicy(
                minimum_relative_jacobian=0.0, maximum_displacement_fraction=None
            ),
        ),
    )


def finite_volume(mesh: D.CellMesh, /) -> D.UnstructuredFiniteVolumeDiscretization:
    return D.UnstructuredFiniteVolumePlan(
        np.asarray(mesh.coordinates, dtype=np.float64),
        triangles=np.asarray(mesh.blocks[0].vertices, dtype=np.int32),
    ).prepare()


def dg0(mesh: D.CellMesh, /) -> D.FiniteElementDiscretization:
    return D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("rho", D.discontinuous_element("triangle", 0))
    ).prepare()


def remap(source_mesh: D.CellMesh, target_mesh: D.CellMesh, density, label, /):
    """Limited P1 remap of ``density`` with P0 and L2-projection comparisons."""
    source, target = finite_volume(source_mesh), finite_volume(target_mesh)
    prepared = D.prepare_unstructured_conservative_remap(
        source,
        target,
        provenance=label,
        policy=phx.geometry.CommonRefinementPolicy(overlap_simplices=True),
    )
    if not prepared.succeeded:
        raise RuntimeError(f"{label}: common refinement failed: {prepared.reason}.")
    limited = D.UnstructuredSecondOrderRemapPlan(
        prepared.plan, prepared.refinement, source
    ).apply(density)
    first_order = prepared.plan.apply(density)
    projection = D.prepare_l2_projection_transfer(
        dg0(source.mesh),
        D.prepare_l2_projection_target(dg0(target.mesh), field_name="rho"),
        prepared.refinement,
        field_name="rho",
    )
    projected = projection.apply(density)
    source_volumes = np.asarray(source.cell_volumes)
    target_volumes = np.asarray(target.cell_volumes)
    mass = float(np.sum(source_volumes * np.asarray(density)))
    values = np.asarray(limited.values)
    evidence = prepared.evidence
    record = {
        "source_cells": source.cell_count,
        "target_cells": target.cell_count,
        "overlap_entries": prepared.refinement.entry_count,
        "maximum_relative_coverage_defect": max(
            evidence.maximum_relative_source_defect,
            evidence.maximum_relative_target_defect,
        ),
        "limited_cells": int(limited.limited_count),
        "minimum_limiter_factor": float(limited.minimum_limiter_factor),
        "conservation_residual_restored": bool(np.all(limited.restored)),
        "relative_mass_residual": abs(float(np.sum(target_volumes * values)) - mass)
        / mass,
        "p0_relative_mass_residual": abs(
            float(np.sum(target_volumes * np.asarray(first_order))) - mass
        )
        / mass,
        "l2_projection_relative_mass_residual": abs(
            float(np.sum(target_volumes * np.asarray(projected))) - mass
        )
        / mass,
        "l2_projection_minus_p0": float(
            np.max(np.abs(np.asarray(projected) - np.asarray(first_order)))
        ),
        "l2_projection_conservative": projection.conservative,
        "source_bounds": [float(np.min(density)), float(np.max(density))],
        "target_bounds": [float(np.min(values)), float(np.max(values))],
    }
    tolerance = 1.0e-12
    if (
        record["relative_mass_residual"] > tolerance
        or record["p0_relative_mass_residual"] > tolerance
        or record["l2_projection_relative_mass_residual"] > 1.0e-10
        or record["l2_projection_minus_p0"] > 1.0e-10
        or not projection.conservative
        or record["target_bounds"][0] < record["source_bounds"][0] - tolerance
        or record["target_bounds"][1] > record["source_bounds"][1] + tolerance
    ):
        raise RuntimeError(f"{label}: remap violated conservation or bounds: {record}")
    return jnp.asarray(values), record


contract = phx.SpatialCoordinateContract.si()
points, triangles = annulus(4, 16)
reference = phx.meshing.certify_cell_mesh(
    D.CellMesh.from_triangles(points, triangles), contract
)
monitor_policy = phx.meshing.MeshMotionMonitorPolicy(
    relocation_quality_ratio=0.6, remesh_quality_ratio=0.3
)
adaptation_policy = phx.meshing.MeshAdaptationPolicy(
    phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
)
monitor = phx.meshing.MeshMotionMonitor(reference.mesh, policy=monitor_policy)
plan = motion_plan(reference.mesh)
realize = eqx.filter_jit(lambda plan, angle: plan.realize(angle, numeric_version="step"))
centroids = np.mean(points[triangles], axis=1)
volumes = np.asarray(phx.meshing.evaluate_cell_quality(reference.mesh).measures)
density = jnp.asarray(1.0 + np.exp(-8.0 * np.sum((centroids - (0.0, 0.6)) ** 2, axis=1)))
masses = volumes * np.asarray(density)
total_mass = float(np.sum(masses))
angle = 0.0
steps = []
for step in range(STEPS):
    angle += STEP_ANGLE
    realization = realize(plan, jnp.asarray(angle))
    if not bool(realization.accepted):
        raise RuntimeError(f"Step {step}: the motion route rejected the proposal.")
    coordinates = np.asarray(realization.coordinates)
    boundary = np.asarray(plan.extension.boundary_indices)
    boundary_residual = float(
        np.max(
            np.abs(
                coordinates[boundary]
                - np.asarray(plan.boundary_provider.rotated(jnp.asarray(angle)))
            )
        )
    )
    advance = phx.meshing.advance_mesh_motion(
        monitor,
        reference,
        coordinates,
        boundary_residual=boundary_residual,
        adaptation_policy=adaptation_policy,
    )
    if not advance.accepted:
        raise RuntimeError(f"Step {step}: {advance.decision.value} was not accepted.")
    moved = monitor.moved_mesh(coordinates)
    moved_volumes = np.asarray(phx.meshing.evaluate_cell_quality(moved).measures)
    lagrangian = jnp.asarray(masses / moved_volumes)
    first = advance.assessments[0]
    record = {
        "step": step,
        "angle_degrees": float(np.rad2deg(angle)),
        "decision": advance.decision.value,
        "assessments": [value.decision.value for value in advance.assessments],
        "reasons": list(first.reasons),
        "minimum_jacobian_ratio": first.minimum_jacobian_ratio,
        "minimum_quality_ratio": first.minimum_quality_ratio,
        "extension_successful": bool(realization.evidence.extension.successful),
        "boundary_residual": boundary_residual,
    }
    match advance.decision:
        case phx.meshing.MeshMotionDecision.ACCEPT_MOTION:
            density = lagrangian
            carried = moved_volumes
        case (
            phx.meshing.MeshMotionDecision.RELOCATE
            | phx.meshing.MeshMotionDecision.REMESH
        ):
            record["relocation"] = advance.relocation.status.value
            if advance.adaptation is not None:
                evidence = advance.adaptation.evidence
                record["adaptation"] = {
                    "status": advance.adaptation.status.value,
                    "passes": evidence.passes,
                    "splits": evidence.splits,
                    "collapses": evidence.collapses,
                    "flips": evidence.flips,
                    "unit_fraction": evidence.unit_fraction,
                }
            # Relocated and remeshed meshes cover the moved domain exactly.
            density, record["remap"] = remap(
                moved, advance.result.mesh, lagrangian, f"{advance.decision}-{step}"
            )
            carried = np.asarray(
                phx.meshing.evaluate_cell_quality(advance.result.mesh).measures
            )
        case decision:
            raise RuntimeError(f"Step {step}: unexpected decision {decision!r}.")
    result = advance.result
    quality = result.quality
    record["mesh"] = {
        "cells": result.mesh.blocks[0].cell_count,
        "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
        "minimum_mean_ratio": quality.minimum_mean_ratio,
        "certified": result.audit.passed,
    }
    masses = carried * np.asarray(density)
    record["relative_mass_residual"] = (
        abs(float(np.sum(masses)) - total_mass) / total_mass
    )
    if record["relative_mass_residual"] > 1.0e-12:
        raise RuntimeError(f"Step {step}: total mass drifted: {record}")
    steps.append(record)
    if advance.decision is not phx.meshing.MeshMotionDecision.ACCEPT_MOTION:
        reference = result
        monitor = phx.meshing.MeshMotionMonitor(reference.mesh, policy=monitor_policy)
        plan = motion_plan(reference.mesh)
        angle = 0.0
remeshes = sum(record["decision"] == "remesh" for record in steps)
if remeshes == 0:
    raise RuntimeError("The monitor never requested a remesh.")
print(
    json.dumps(
        {
            "route": "linear_elasticity",
            "steps": steps,
            "remeshes": remeshes,
            "final_cells": reference.mesh.blocks[0].cell_count,
            "total_mass": total_mass,
        },
        indent=2,
    )
)
