#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic


def _motion_setup():
    coordinates = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ]
    )
    triangles = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    mesh = phx.discretization.CellMesh.from_triangles(coordinates, triangles)
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    geometry = phx.geometry.Circle(
        (0.0, 0.0),
        1.0,
        feature_id="circle",
    ).compile()
    projection = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        coordinates[:4],
        0.3,
        source_id="circle-boundary",
    )
    motion = phx.discretization.FiniteElementMeshMotionPlan(
        discretization,
        projection,
    )
    radius_index = geometry.schema.index(phx.geometry.ParameterId("circle", "radius"))
    return geometry, discretization, motion, radius_index


def _surface_setup():
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="sphere",
    ).compile()
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(7) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]]))
    plan = phx.geometry.discover_implicit_surface(
        geometry,
        grid,
        policy=phx.geometry.ImplicitSurfacePolicy(maximum_intersection_pairs=500_000),
        source_id="sphere-surface",
    )
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    return geometry, plan, radius_index


def _case(case_id, passed, **metrics):
    return {
        "case_id": case_id,
        "passed": bool(passed),
        "metrics": {
            name: bool(value)
            if isinstance(value, (bool, np.bool_))
            else int(value)
            if isinstance(value, (int, np.integer))
            else float(value)
            for name, value in metrics.items()
        },
    }


def run():
    cases = []
    extrusion = (
        phx.geometry.Circle((0.0, 0.0), 2.0, feature_id="profile")
        .extruded(6.0, feature_id="extrusion")
        .compile()
    )
    extrusion_error = float(
        jnp.max(
            jnp.abs(
                extrusion.signed_distance(jnp.asarray([[3.0, 0.0, 4.0]])) - jnp.sqrt(2.0)
            )
        )
    )
    cases.append(
        _case(
            "extrusion_exactness",
            extrusion_error < 1.0e-6,
            field_error=extrusion_error,
            volume_error=abs(float(extrusion.measure - 24.0 * jnp.pi)),
        )
    )
    revolution = (
        phx.geometry.Circle((2.0, 0.0), 0.5, feature_id="radial-profile")
        .revolved()
        .compile()
    )
    revolution_error = float(
        jnp.max(
            jnp.abs(
                revolution.signed_distance(
                    jnp.asarray([[2.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
                )
                - jnp.asarray([0.0, 1.5])
            )
        )
    )
    cases.append(
        _case(
            "revolution_exactness",
            bool(revolution.validity().accepted) and revolution_error < 1.0e-6,
            field_error=revolution_error,
        )
    )
    invalid_revolution = (
        phx.geometry.Circle((0.25, 0.0), 0.5, feature_id="invalid-profile")
        .revolved()
        .compile()
    )
    cases.append(
        _case(
            "geometry_validity",
            not bool(invalid_revolution.validity().accepted),
            invalid_detected=not bool(invalid_revolution.validity().accepted),
        )
    )

    geometry, discretization, motion, radius_index = _motion_setup()

    def projected(radius):
        design = geometry.state.replace_at(radius_index, radius)
        return motion.boundary_provider.realize(design).proposed_points

    _, projection_tangent = jax.jvp(
        projected,
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )
    projection_error = float(
        jnp.max(jnp.abs(projection_tangent - motion.boundary_provider.reference_points))
    )
    cases.append(
        _case(
            "implicit_projection_derivative",
            projection_error < 1.0e-6,
            maximum_error=projection_error,
        )
    )

    sphere, surface_plan, sphere_radius_index = _surface_setup()
    surface = surface_plan.realize(sphere.state)
    mesh = surface.to_triangle_mesh()
    cases.append(
        _case(
            "implicit_surface_topology",
            bool(surface.accepted) and mesh.topology.watertight,
            vertices=surface.vertices.shape[0],
            faces=surface.faces.shape[0],
            components=mesh.topology.num_face_components,
            minimum_face_area=float(surface.evidence.minimum_face_area),
        )
    )
    expired_surface = surface_plan.realize(
        sphere.state.replace_at(sphere_radius_index, jnp.asarray(1.2))
    )
    cases.append(
        _case(
            "implicit_surface_refresh",
            (not bool(expired_surface.accepted))
            and bool(expired_surface.refresh_required),
            accepted=bool(expired_surface.accepted),
            refresh_required=bool(expired_surface.refresh_required),
        )
    )

    def coordinates(radius):
        design = geometry.state.replace_at(radius_index, radius)
        return motion.realize(design).proposed_coordinates

    radius = jnp.asarray(1.05)
    cotangent = jnp.arange(10, dtype=float).reshape((5, 2)) / 10.0
    _, tangent = jax.jvp(coordinates, (radius,), (jnp.asarray(1.0),))
    _, pullback = jax.vjp(coordinates, radius)
    left = jnp.sum(tangent * cotangent)
    right = pullback(cotangent)[0]
    duality_defect = float(jnp.abs(left - right))
    cases.append(
        _case(
            "mesh_motion_duality",
            duality_defect < 1.0e-6,
            work_duality_defect=duality_defect,
        )
    )

    def area(radius_value):
        design = geometry.state.replace_at(radius_index, radius_value)
        realization = motion.realize(design)
        blocks = discretization.evaluate_geometry("u", realization.runtime.coordinates)
        return sum(jnp.sum(block.physical_weights) for block in blocks)

    analytic = jax.grad(area)(radius)
    step = jnp.asarray(1.0e-4)
    finite_difference = (area(radius + step) - area(radius - step)) / (2.0 * step)
    derivative_defect = float(jnp.abs(analytic - finite_difference))
    cases.append(
        _case(
            "finite_element_shape_derivative",
            derivative_defect < 1.0e-3,
            adjoint_value=float(analytic),
            finite_difference=float(finite_difference),
            absolute_defect=derivative_defect,
        )
    )
    expired_motion = motion.realize(
        geometry.state.replace_at(radius_index, jnp.asarray(1.8))
    )
    cases.append(
        _case(
            "invalid_candidate_rejection",
            (not bool(expired_motion.accepted))
            and jnp.array_equal(
                expired_motion.coordinates,
                motion.reference_coordinates,
            ),
            accepted=bool(expired_motion.accepted),
            refresh_required=bool(expired_motion.refresh_required),
        )
    )
    accepted_motion = motion.realize(
        geometry.state.replace_at(radius_index, jnp.asarray(1.1))
    )
    cases.append(
        _case(
            "end_to_end_state_design",
            bool(accepted_motion.accepted)
            and accepted_motion.runtime.topology_id == motion.topology_id,
            accepted=bool(accepted_motion.accepted),
            minimum_relative_jacobian=float(
                accepted_motion.evidence.geometry.minimum_relative_jacobian
            ),
        )
    )
    cases.append(
        _case(
            "invalid_candidate_finiteness",
            bool(jnp.all(jnp.isfinite(expired_motion.runtime.coordinates))),
            finite=bool(jnp.all(jnp.isfinite(expired_motion.runtime.coordinates))),
        )
    )
    sharp_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8),
            phx.discretization.UniformCellAxisSpec(8),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    sharp_finite_volume = phx.discretization.FiniteVolumePlan(sharp_grid).prepare()
    sharp_mac = phx.discretization.MACOperatorPlan(sharp_finite_volume).prepare()

    def plane_signed_distance(points, time, args):
        del time, args
        return points[..., 0] - 0.3125

    sharp = phx.discretization.MACExactSDFMeasurePlan(
        sharp_mac,
        plane_signed_distance,
        phx.geometry.ExactSDFEnclosureCertificate(
            phx.geometry.exact_signed_distance_certificate(smooth=True)
        ),
        source_id="qualified-plane",
        subdivisions=16,
    ).prepare()
    volume_lower = float(jnp.sum(sharp.cell_fluid_measure_lower))
    volume_upper = float(jnp.sum(sharp.cell_fluid_measure_upper))
    exact_volume = 0.6875
    cases.append(
        _case(
            "qualified_sharp_fluid_measure",
            bool(sharp.accepted)
            and volume_lower <= exact_volume <= volume_upper
            and bool(sharp.evidence.topology_resolved),
            volume_lower=volume_lower,
            volume_upper=volume_upper,
            exact_volume=exact_volume,
            bound_width=volume_upper - volume_lower,
            topology_resolved=bool(sharp.evidence.topology_resolved),
        )
    )
    return {
        "kind": "geometry-realization-qualification",
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/geometry_realization_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = run()
    write_json_atomic(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
