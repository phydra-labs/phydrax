#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# ty: ignore[unresolved-import]
from geometry_realization_qualification import _motion_setup

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_repeated,
    measure_synchronized,
)
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry._meshing_domain import MeshingDomain, SourcePreparation
from phydrax.geometry.brep import (
    brep_box,
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
    BRepGeometry,
    BRepModel,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._patches import BSplineCurve, BSplineSurfacePatch, LineCurve
from phydrax.geometry.brep._tessellation import tessellate_brep_model
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
)
from phydrax.operators.integral.layer_potential._scalar_calderon3d import (
    prepare_scalar_calderon_dp0_3d,
)
from phydrax.solver._fem_bem_scalar import prepare_scalar_laplace_fem_bem_3d


def _surface_case(count: int, *, warmup: int, repeats: int) -> Any:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id=f"sphere-{count}",
    ).compile()
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]]))
    plan, discovery_seconds = measure_synchronized(
        lambda: phx.geometry.discover_implicit_surface(
            geometry,
            grid,
            source_id=f"sphere-surface-{count}",
        )
    )
    realization, first_realization_seconds = measure_synchronized(
        lambda: plan.realize(geometry.state)
    )
    _, realization_distribution = measure_repeated(
        lambda: plan.realize(geometry.state),
        warmup=warmup,
        repeats=repeats,
    )
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId(f"sphere-{count}", "radius")
    )

    def vertices(radius: Any) -> Any:
        state = geometry.state.replace_at(radius_index, radius)
        return plan.realize(state).proposed_vertices

    _, first_jvp_seconds = measure_synchronized(
        lambda: jax.jvp(
            vertices,
            (jnp.asarray(0.75),),
            (jnp.asarray(1.0),),
        )
    )
    _, jvp_distribution = measure_repeated(
        lambda: jax.jvp(
            vertices,
            (jnp.asarray(0.75),),
            (jnp.asarray(1.0),),
        ),
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "accepted": bool(realization.accepted),
        "discovery_seconds": discovery_seconds,
        "faces": realization.faces.shape[0],
        "first_jvp_seconds": first_jvp_seconds,
        "first_realization_seconds": first_realization_seconds,
        "intersection_pairs": plan.intersection_pairs.shape[0],
        "jvp": jvp_distribution.to_seconds_dict(),
        "lattice_points": plan.grid_points.shape[0],
        "logical_plan_bytes": logical_array_bytes(plan),
        "minimum_face_area": float(realization.evidence.minimum_face_area),
        "realization": realization_distribution.to_seconds_dict(),
        "resolution_per_axis": count,
        "vertices": realization.vertices.shape[0],
    }


def _finite_element_case(*, warmup: int, repeats: int) -> Any:
    geometry, discretization, motion, radius_index = _motion_setup()
    design = geometry.state.replace_at(radius_index, jnp.asarray(1.1))
    realization, first_realization_seconds = measure_synchronized(
        lambda: motion.realize(design)
    )
    _, realization_distribution = measure_repeated(
        lambda: motion.realize(design),
        warmup=warmup,
        repeats=repeats,
    )

    def area(radius: Any) -> Any:
        current = geometry.state.replace_at(radius_index, radius)
        moved = motion.realize(current)
        blocks = discretization.evaluate_geometry("u", moved.runtime.coordinates)
        return sum(jnp.sum(block.physical_weights) for block in blocks)

    derivative, first_derivative_seconds = measure_synchronized(
        lambda: jax.grad(area)(jnp.asarray(1.1))
    )
    _, derivative_distribution = measure_repeated(
        lambda: jax.grad(area)(jnp.asarray(1.1)),
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "accepted": bool(realization.accepted),
        "boundary_vertices": motion.extension.boundary_indices.shape[0],
        "derivative": derivative_distribution.to_seconds_dict(),
        "first_derivative_seconds": first_derivative_seconds,
        "first_realization_seconds": first_realization_seconds,
        "interior_vertices": motion.extension.interior_indices.shape[0],
        "logical_plan_bytes": logical_array_bytes(motion),
        "minimum_relative_jacobian": float(
            realization.evidence.geometry.minimum_relative_jacobian
        ),
        "realization": realization_distribution.to_seconds_dict(),
        "shape_derivative": float(derivative),
        "vertices": realization.coordinates.shape[0],
    }


def _sharp_measure_case(count: int, *, warmup: int, repeats: int) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count),
            phx.discretization.UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(discretization).prepare()

    def plane(points: Any, time: Any, args: Any) -> Any:
        del time, args
        return points[..., 0] - 0.3125

    plan = phx.discretization.MACExactSDFMeasurePlan(
        mac,
        plane,
        # ty: ignore[invalid-argument-type]
        phx.geometry.ExactSDFEnclosureCertificate(
            phx.geometry.exact_signed_distance_certificate(smooth=True)
        ),
        source_id=f"benchmark-plane-{count}",
        subdivisions=8,
    )
    realization, first_seconds = measure_synchronized(plan.prepare)
    _, distribution = measure_repeated(
        plan.prepare,
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "accepted": bool(realization.accepted),
        "bound_width": float(
            jnp.sum(
                realization.cell_fluid_measure_upper
                - realization.cell_fluid_measure_lower
            )
        ),
        "cells": count * count,
        "first_realization_seconds": first_seconds,
        "logical_plan_bytes": logical_array_bytes(plan),
        "realization": distribution.to_seconds_dict(),
        "resolution_per_axis": count,
        "topology_resolved": bool(realization.evidence.topology_resolved),
    }


def _placement_case() -> Any:
    """Time separate native placement, world evidence and scientific consumers."""
    box = brep_box(
        (0, 0, 0), (1, 1, 1), coordinate_contract=SpatialCoordinateContract.si()
    )
    patches = []
    for index, original in enumerate(box.patches):
        bounds = np.asarray(box.parameter_bounds[index], dtype=np.float64)
        parameters = np.asarray(
            [[(u, v) for v in bounds[:, 1]] for u in bounds[:, 0]], dtype=np.float64
        )
        control = np.asarray(original.evaluate(jnp.asarray(parameters)))
        exact = tuple(
            tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in row
            )
            for row in control
        )
        u = tuple(Fraction.from_float(float(value)) for value in bounds[:, 0])
        v = tuple(Fraction.from_float(float(value)) for value in bounds[:, 1])
        patches.append(
            BSplineSurfacePatch(
                control,
                np.ones((2, 2)),
                np.asarray((u[0], u[0], u[1], u[1]), dtype=np.float64),
                np.asarray((v[0], v[0], v[1], v[1]), dtype=np.float64),
                1,
                1,
                exact_control_points=exact,
                exact_weights=((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))),
                exact_u_knots=(u[0], u[0], u[1], u[1]),
                exact_v_knots=(v[0], v[0], v[1], v[1]),
                control_point_ids=(
                    (4 * index, 4 * index + 1),
                    (4 * index + 2, 4 * index + 3),
                ),
            )
        )
    graph = BRepAssemblyGraph(
        (
            BRepAssemblyUse("first", definition_id="part"),
            BRepAssemblyUse(
                "second",
                definition_id="part",
                placement=BRepAffinePlacement.from_matrix(
                    (
                        (1, 0, 0, 1),
                        (0, 1, 0, Fraction(41, 20)),
                        (0, 0, 1, 0),
                        (0, 0, 0, 1),
                    )
                ),
            ),
        ),
        (BRepAssemblyDefinition("part", (BRepAssemblyUse("leaf", solid_index=0),)),),
        1,
    )
    geometry = box.geometry
    pcurves = []
    for coedge, curve in enumerate(geometry.pcurves):
        if not isinstance(curve, LineCurve):
            raise TypeError("The box trim fixture requires native line pcurves.")
        edge = geometry.coedge_edges[coedge]
        start, end = (
            Fraction.from_float(float(value)) for value in geometry.edge_ranges[edge]
        )
        controls = tuple(
            tuple(
                origin + parameter * direction
                for origin, direction in zip(
                    curve.origin_exact, curve.direction_exact, strict=True
                )
            )
            for parameter in (start, end)
        )
        knots = (start, start, end, end)
        pcurves.append(
            BSplineCurve(
                np.asarray(controls, dtype=np.float64),
                np.ones(2),
                np.asarray(knots, dtype=np.float64),
                1,
                exact_control_points=controls,
                exact_weights=(Fraction(1), Fraction(1)),
                exact_knots=knots,
            )
        )
    geometry = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=tuple(pcurves),
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=graph,
    )
    source = BRepModel(
        geometry=geometry,
        patches=tuple(patches),
        parameter_bounds=box.parameter_bounds,
        orientation=box.orientation,
        physical_tags=box.physical_tags,
        coordinate_contract=box.coordinate_contract,
        report=box.report,
        trim_domains=box.trim_domains,
    )
    pose = BRepAffinePlacement.from_matrix(
        ((0, -1, 0, 2), (1, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    policy = BRepTessellationPolicy(
        linear_deflection=0.1, angular_deflection=0.3, trim_samples_per_edge=3
    )

    def local_prepare() -> Any:
        preparation = SourcePreparation(MeshingDomain.from_brep(source))
        for patch in source.patches:
            preparation.surface_spans(patch)
        for pcurve in pcurves:
            preparation.curve_spans(pcurve)
        return preparation

    local, local_seconds = measure_synchronized(local_prepare)

    def place() -> Any:
        first = phx.geometry.edit_brep_occurrence_placement(
            source,
            ("first", "leaf"),
            BRepAffinePlacement.from_matrix(
                ((1, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
            ),
        )
        return phx.geometry.edit_brep_occurrence_placement(
            first.model, ("first", "leaf"), pose
        )

    edited, placement_seconds = measure_synchronized(place)
    reused, reuse_seconds = measure_synchronized(
        lambda: local.for_placement_domain(MeshingDomain.from_brep(edited.model))
    )
    reused_count = sum(
        reused.surface_spans(patch) is local.surface_spans(patch)
        for patch in source.patches
    )
    reused_trim_count = sum(
        reused.curve_spans(pcurve) is local.curve_spans(pcurve)
        for pcurve in pcurves
    )

    def world(model: Any, preparation: Any) -> Any:
        return tessellate_brep_model(model, policy, preparation=preparation)

    (tessellation, certificate), world_seconds = measure_synchronized(
        lambda: world(edited.model, reused)
    )
    (cold_tessellation, cold_certificate), cold_world_seconds = measure_synchronized(
        lambda: world(
            edited.model, SourcePreparation(MeshingDomain.from_brep(edited.model))
        )
    )
    np.testing.assert_array_equal(tessellation.vertices, cold_tessellation.vertices)
    np.testing.assert_array_equal(tessellation.triangles, cold_tessellation.triangles)
    np.testing.assert_array_equal(
        tessellation.triangle_occurrence_ids, cold_tessellation.triangle_occurrence_ids
    )
    if certificate.certificate_id != cold_certificate.certificate_id:
        raise ValueError("Cold and reused placement world proofs differ.")

    def coordinates(model: Any) -> Any:
        occurrence = next(
            item
            for item in model.geometry.iter_occurrences()
            if item.path == ("first", "leaf")
        )
        points = (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        )
        return jnp.asarray(
            [
                tuple(float(value) for value in occurrence.placement.apply(point))
                for point in points
            ],
            dtype=jnp.float64,
        )

    faces = jnp.asarray(
        (
            (0, 2, 3),
            (0, 3, 1),
            (4, 5, 7),
            (4, 7, 6),
            (0, 1, 5),
            (0, 5, 4),
            (2, 6, 7),
            (2, 7, 3),
            (0, 4, 6),
            (0, 6, 2),
            (1, 3, 7),
            (1, 7, 5),
        ),
        dtype=jnp.int32,
    )
    cells = jnp.asarray(
        (
            (0, 1, 3, 7),
            (0, 3, 2, 7),
            (0, 2, 6, 7),
            (0, 6, 4, 7),
            (0, 4, 5, 7),
            (0, 5, 1, 7),
        ),
        dtype=jnp.int32,
    )

    def coupling(model: Any, *, numeric_version: str) -> Any:
        points = coordinates(model)
        mesh = phx.discretization.CellMesh.from_tetrahedra(points, cells)
        fem = FiniteElementPlan(
            mesh, FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
        ).prepare(numeric_version=numeric_version)
        surface = phx.geometry.MeshRegion(points, faces, feature_id="placement-benchmark")
        bem = prepare_scalar_calderon_dp0_3d(
            surface,
            policy=LaplaceSingleLayerDP0GalerkinPolicy3D(
                regular_order=3,
                singular_order=3,
                near_order=3,
                near_ratio=1.0,
                absolute_tolerance=2.0e-3,
                relative_tolerance=2.0e-3,
                target_block_size=4,
                source_block_size=4,
            ),
            numeric_version=numeric_version,
        )
        prepared = prepare_scalar_laplace_fem_bem_3d(fem, surface, bem)
        gradient = jnp.asarray((0.75, -0.5, 0.25), dtype=points.dtype)
        coefficients = 1.25 + points @ gradient
        return prepared.interface.trace(coefficients), prepared.interface.conormal(
            coefficients
        )

    def contact(model: Any, mesh: Any) -> Any:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        triangle_occurrences = np.asarray(mesh.triangle_occurrence_ids)
        occurrence_ids = {
            occurrence.path: index
            for index, occurrence in enumerate(model.geometry.iter_occurrences())
        }
        space = phx.linalg.ArraySpace((1, 3), dtype=vertices.dtype)
        surfaces = []
        counts = []
        for body, path in enumerate((("first", "leaf"), ("second", "leaf"))):
            occurrence_id = occurrence_ids[path]
            selected = triangles[triangle_occurrences == occurrence_id]
            vertex_indices = np.unique(selected)
            local_faces = np.searchsorted(vertex_indices, selected)
            points = jnp.asarray(vertices[vertex_indices])
            count = len(vertex_indices)
            counts.append((count, len(selected)))
            topology = phx.discretization.CollisionSurfacePlan(
                jnp.asarray(vertex_indices, dtype=jnp.int64),
                ambient_dimension=3,
                faces=jnp.asarray(local_faces, dtype=jnp.int32),
                body_ids=body,
                participant_ids=body,
            )
            surfaces.append(
                phx.discretization.PreparedCollisionSurface(
                    topology,
                    points,
                    phx.discretization.static_collision_operator(space, count, 3),
                )
            )
        scene = phx.discretization.PreparedCollisionScene(tuple(surfaces))
        total_vertices = sum(count for count, _ in counts)
        total_faces = sum(count for _, count in counts)
        search = phx.discretization.SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=3 * total_faces * total_vertices,
            edge_edge_capacity=9 * total_faces**2,
            face_vertex_capacity=total_faces * total_vertices,
            activation_distance=0.1,
        )
        return search.build(scene, scene.positions(space.zeros()))

    epoch, contact_seconds = measure_synchronized(
        lambda: contact(edited.model, tessellation)
    )
    traces, coupling_seconds = measure_synchronized(
        lambda: coupling(edited.model, numeric_version=edited.model.source_revision)
    )
    cold_epoch, cold_contact_seconds = measure_synchronized(
        lambda: contact(edited.model, cold_tessellation)
    )
    cold_traces, cold_coupling_seconds = measure_synchronized(
        lambda: coupling(
            edited.model, numeric_version="cold-" + edited.model.source_revision
        )
    )
    np.testing.assert_array_equal(
        epoch.reference_positions, cold_epoch.reference_positions
    )
    np.testing.assert_array_equal(epoch.status, cold_epoch.status)
    np.testing.assert_array_equal(epoch.complete, cold_epoch.complete)
    for actual, cold_batch in zip(epoch.batches, cold_epoch.batches, strict=True):
        np.testing.assert_array_equal(actual.valid, cold_batch.valid)
        np.testing.assert_array_equal(actual.vertex_indices, cold_batch.vertex_indices)
    for actual, cold_value in zip(traces, cold_traces, strict=True):
        np.testing.assert_allclose(actual, cold_value)

    def end_to_end() -> Any:
        preparation = local_prepare()
        result = place()
        reused_preparation = preparation.for_placement_domain(
            MeshingDomain.from_brep(result.model)
        )
        tessellated = world(result.model, reused_preparation)
        return (
            tessellated,
            contact(result.model, tessellated[0]),
            coupling(result.model, numeric_version=result.model.source_revision),
        )

    _, total_seconds = measure_synchronized(end_to_end)
    return {
        "local_preparation_seconds": local_seconds,
        "placement_seconds": placement_seconds,
        "local_span_reuse_seconds": reuse_seconds,
        "local_spans_reused": reused_count,
        "local_trim_spans_reused": reused_trim_count,
        "world_tessellation_and_proof_seconds": world_seconds,
        "cold_world_tessellation_and_proof_seconds": cold_world_seconds,
        "cold_contact_search_seconds": cold_contact_seconds,
        "cold_coupling_preparation_and_trace_seconds": cold_coupling_seconds,
        "cold_reuse_parity": True,
        "contact_search_seconds": contact_seconds,
        "coupling_preparation_and_trace_seconds": coupling_seconds,
        "end_to_end_seconds": total_seconds,
        "world_triangles": int(tessellation.triangles.shape[0]),
        "world_certificate_status": certificate.status,
        "contact_candidates": int(epoch.candidate_count),
        "contact_successful": bool(epoch.successful),
        "trace": traces[0].tolist(),
        "conormal": traces[1].tolist(),
    }


def run(*, smoke: bool, warmup: int, repeats: int) -> Any:
    resolutions = (7,) if smoke else (7, 9)
    return {
        "environment": capture_environment().to_dict(),
        "finite_element": _finite_element_case(warmup=warmup, repeats=repeats),
        "placement": _placement_case(),
        "kind": "geometry-realization-record-only-timing",
        "record_only": True,
        "surface": [
            _surface_case(count, warmup=warmup, repeats=repeats) for count in resolutions
        ],
        "sharp_measure": _sharp_measure_case(
            8 if smoke else 16,
            warmup=warmup,
            repeats=repeats,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record fixed-topology geometry realization timings."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/geometry_realization.json"),
    )
    arguments = parser.parse_args()
    payload = run(
        smoke=arguments.smoke,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    print(rendered)
    write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
