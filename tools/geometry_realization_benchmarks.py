#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from geometry_realization_qualification import _motion_setup

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_repeated,
    measure_synchronized,
)


def _surface_case(count: int, *, warmup: int, repeats: int):
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id=f"sphere-{count}",
    ).compile()
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]]))
    policy = phx.geometry.ImplicitSurfacePolicy(maximum_intersection_pairs=2_000_000)
    plan, discovery_seconds = measure_synchronized(
        lambda: phx.geometry.discover_implicit_surface(
            geometry,
            grid,
            policy=policy,
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

    def vertices(radius):
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
        "faces": int(realization.faces.shape[0]),
        "first_jvp_seconds": first_jvp_seconds,
        "first_realization_seconds": first_realization_seconds,
        "intersection_pairs": int(plan.intersection_pairs.shape[0]),
        "jvp": jvp_distribution.to_seconds_dict(),
        "lattice_points": int(plan.grid_points.shape[0]),
        "logical_plan_bytes": logical_array_bytes(plan),
        "minimum_face_area": float(realization.evidence.minimum_face_area),
        "realization": realization_distribution.to_seconds_dict(),
        "resolution_per_axis": count,
        "vertices": int(realization.vertices.shape[0]),
    }


def _finite_element_case(*, warmup: int, repeats: int):
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

    def area(radius):
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
        "boundary_vertices": int(motion.boundary_indices.shape[0]),
        "derivative": derivative_distribution.to_seconds_dict(),
        "first_derivative_seconds": first_derivative_seconds,
        "first_realization_seconds": first_realization_seconds,
        "interior_vertices": int(motion.interior_indices.shape[0]),
        "logical_plan_bytes": logical_array_bytes(motion),
        "minimum_relative_jacobian": float(
            realization.evidence.geometry.minimum_relative_jacobian
        ),
        "realization": realization_distribution.to_seconds_dict(),
        "shape_derivative": float(derivative),
        "vertices": int(realization.coordinates.shape[0]),
    }


def _sharp_measure_case(count: int, *, warmup: int, repeats: int):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count),
            phx.discretization.UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(discretization).prepare()

    def plane(points, time, args):
        del time, args
        return points[..., 0] - 0.3125

    plan = phx.discretization.MACExactSDFMeasurePlan(
        mac,
        plane,
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


def run(*, smoke: bool, warmup: int, repeats: int):
    resolutions = (7,) if smoke else (7, 9)
    return {
        "environment": capture_environment().to_dict(),
        "finite_element": _finite_element_case(warmup=warmup, repeats=repeats),
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


def main():
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
