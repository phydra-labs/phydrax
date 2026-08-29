#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _stefan(points):
    started = time.perf_counter()
    result = phx.applications.free_boundary.ExactStefanBenchmark().run(
        points_per_block=points,
        key=jr.key(0),
    )
    return {
        "seconds": time.perf_counter() - started,
        "explicit_loss": float(result.explicit.total),
        "implicit_loss": float(result.implicit.total),
        "reference_loss": float(result.reference.total),
        "best_representation": result.best_representation,
    }


def _instability():
    times = jnp.linspace(0.0, 1.0, 16)
    modes = jnp.asarray((2, 3, 4))
    rates = jnp.asarray((0.2, 0.05, -0.1))
    initial = jnp.asarray((0.1, 0.05, 0.02))
    exact = initial[None, :] * jnp.exp(times[:, None] * rates[None, :])
    report = phx.applications.free_boundary.mullins_sekerka_benchmark(
        exact,
        times,
        modes,
        rates,
        initial,
    )
    return {
        "relative_l2_error": float(report.relative_l2_error),
        "maximum_relative_mode_error": float(report.maximum_relative_mode_error),
        "dominant_mode": int(report.predicted_dominant_mode),
    }


def _topology():
    report = phx.applications.free_boundary.topology_event_benchmark(
        jnp.asarray((1, 1, 2, 2)),
        jnp.asarray((1, 1, 2, 2)),
        jnp.asarray((0.0, 0.25, 0.5, 0.75)),
    )
    return {
        "component_accuracy": float(report.component_count_correct),
        "event_time_error": float(report.event_time_error),
        "event_order_correct": bool(report.event_order_correct),
    }


def _bubble():
    angle = jnp.linspace(0.0, 2.0 * jnp.pi, 129)[:-1]
    contour = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    report = phx.applications.free_boundary.hysing_bubble_benchmark(
        jnp.ones((4,)),
        jnp.asarray(((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))),
        jnp.full((4,), jnp.pi / 4.0),
        jnp.full((4,), 1.0),
        contour,
    )
    return {
        "area": float(report.area),
        "circularity": float(report.circularity),
        "centroid": [float(value) for value in report.centroid],
        "mean_rise_velocity": float(report.mean_rise_velocity),
    }


def _fsi_obstacle_fracture():
    time = jnp.arange(128) * 0.01
    signal = jnp.sin(2.0 * jnp.pi * 2.0 * time)
    fsi = phx.applications.free_boundary.turek_hron_fsi_benchmark(
        signal,
        signal,
        signal,
        signal,
        signal,
        signal,
        0.01,
    )
    obstacle = phx.applications.free_boundary.obstacle_complementarity_benchmark(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 0.0)),
    )
    crack = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    fracture = phx.applications.free_boundary.phase_field_fracture_benchmark(
        jnp.asarray(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))),
        jnp.asarray((0.0, 1.0, 2.0)),
        jnp.asarray((0.0, 1.0, 2.0)),
        jnp.asarray((0.0, 0.5, 1.0)),
        crack,
        crack,
    )
    return {
        "fsi_tip_relative_l2": float(fsi.tip_relative_l2),
        "fsi_frequency_error": float(fsi.dominant_frequency_error),
        "obstacle_complementarity": float(obstacle.complementarity_residual),
        "fracture_irreversibility": float(fracture.irreversibility_violation),
        "fracture_path_hausdorff": float(fracture.crack_path_hausdorff),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.points < 8:
        raise ValueError("points must be at least eight.")
    points = min(arguments.points, 16) if arguments.smoke else arguments.points
    report = {
        "stefan": _stefan(points),
        "mullins_sekerka": _instability(),
        "topology": _topology(),
        "hysing": _bubble(),
        "fsi_obstacle_fracture": _fsi_obstacle_fracture(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
