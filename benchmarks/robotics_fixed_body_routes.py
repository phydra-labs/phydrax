"""Benchmark articulated fixed-body muscle-route JVP and force pullback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from benchmarks._runtime import measure_lower_and_compile, measure_repeated
from phydrax.applications.robotics import FixedBodyRoutePlan, parse_urdf_text


def _urdf(joints: int) -> str:
    links = []
    for index in range(joints + 1):
        links.append(
            f'<link name="link-{index}"><inertial><mass value="1"/>'
            '<inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>'
            "</inertial></link>"
        )
    edges = []
    for index in range(joints):
        edges.append(
            f'<joint name="joint-{index}" type="revolute">'
            f'<parent link="link-{index}"/><child link="link-{index + 1}"/>'
            '<origin xyz="0.25 0 0"/><axis xyz="0 0 1"/>'
            '<limit lower="-2.5" upper="2.5" effort="500" velocity="20"/>'
            "</joint>"
        )
    return '<robot name="route-benchmark">' + "".join(links + edges) + "</robot>"


def _case(joints: int, routes: int):
    adaptation = parse_urdf_text(_urdf(joints))
    particles = adaptation.particles.prepare()
    bodies = adaptation.bodies.prepare(particles)
    graph = adaptation.joints.prepare(bodies, adaptation.reference)
    articulation = adaptation.articulation.prepare(graph, adaptation.reference)
    names = tuple(f"route-{index}" for index in range(routes))
    offsets = tuple(2 * index for index in range(routes + 1))
    root = int(adaptation.link_ids.id_for_name("link-0"))
    body_ids = []
    local = []
    for index in range(routes):
        target = 1 + index % joints
        body_ids.extend((root, int(adaptation.link_ids.id_for_name(f"link-{target}"))))
        local.extend(((0.0, 0.02 * (index % 5), 0.0), (0.1, -0.01 * (index % 7), 0.0)))
    route = FixedBodyRoutePlan(names, offsets, body_ids).prepare(
        articulation, jnp.asarray(local)
    )
    configuration = jnp.linspace(-0.3, 0.3, articulation.nq)
    velocity = jnp.linspace(0.2, -0.2, articulation.nv)
    tension = jnp.linspace(100.0, 2500.0, routes)
    return route, configuration, velocity, tension


def benchmark(
    joints: int, routes: int, warmup: int, repeats: int
) -> dict[str, object]:
    route, configuration, velocity, tension = _case(joints, routes)
    evaluate = eqx.filter_jit(route.evaluate)
    pullback = eqx.filter_jit(route.tensile_force_pullback)
    compiled_evaluate, evaluation_compilation = measure_lower_and_compile(
        lambda: evaluate.lower(configuration, velocity),
        lambda lowered: lowered.compile(),
    )
    compiled_pullback, pullback_compilation = measure_lower_and_compile(
        lambda: pullback.lower(configuration, velocity, tension),
        lambda lowered: lowered.compile(),
    )
    evaluated, evaluation_execution = measure_repeated(
        lambda: compiled_evaluate(configuration, velocity),
        warmup=warmup,
        repeats=repeats,
    )
    (load, evidence), pullback_execution = measure_repeated(
        lambda: compiled_pullback(configuration, velocity, tension),
        warmup=warmup,
        repeats=repeats,
    )
    evaluation_median = evaluation_execution.median_seconds
    pullback_median = pullback_execution.median_seconds
    assert evaluation_median is not None
    assert pullback_median is not None
    return {
        "benchmark": "robotics-fixed-body-routes",
        "joint_count": joints,
        "route_count": routes,
        "point_count": route.point_capacity,
        "warmup": warmup,
        "repeats": repeats,
        "timing_scope": (
            "synchronized execution of explicitly lowered and compiled executables"
        ),
        "evaluation_lower_ms": 1000.0 * evaluation_compilation.lowering_seconds,
        "evaluation_compile_ms": 1000.0 * evaluation_compilation.compilation_seconds,
        "evaluation_run": evaluation_execution.to_milliseconds_dict(),
        "evaluations_per_second": 1.0 / evaluation_median,
        "pullback_lower_ms": 1000.0 * pullback_compilation.lowering_seconds,
        "pullback_compile_ms": 1000.0 * pullback_compilation.compilation_seconds,
        "pullback_run": pullback_execution.to_milliseconds_dict(),
        "pullbacks_per_second": 1.0 / pullback_median,
        "virtual_power_successful": bool(evidence.successful),
        "route_length_checksum_m": float(jnp.sum(evaluated.route_lengths_m)),
        "load_checksum_Nm": float(jnp.sum(load)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", type=int, default=30)
    parser.add_argument("--routes", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/robotics_fixed_body_routes.json"),
    )
    arguments = parser.parse_args()
    if (
        min(arguments.joints, arguments.routes, arguments.repeats) <= 0
        or arguments.warmup < 0
    ):
        raise ValueError(
            "joints, routes, and repeats must be positive; warmup must be nonnegative."
        )
    payload = benchmark(
        arguments.joints,
        arguments.routes,
        arguments.warmup,
        arguments.repeats,
    )
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["virtual_power_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
