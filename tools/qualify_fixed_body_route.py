"""Qualify fixed body-route derivatives and virtual power independently."""

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import numpy as np

from phydrax.applications.robotics import FixedBodyRoutePlan, parse_urdf_text


_URDF = """
<robot name="route-qualification">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="link">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="link"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
</robot>
"""


def _prepared_route():
    adaptation = parse_urdf_text(_URDF)
    particles = adaptation.particles.prepare()
    bodies = adaptation.bodies.prepare(particles)
    graph = adaptation.joints.prepare(bodies, adaptation.reference)
    articulation = adaptation.articulation.prepare(graph, adaptation.reference)
    base = int(adaptation.link_ids.id_for_name("base"))
    link = int(adaptation.link_ids.id_for_name("link"))
    route = FixedBodyRoutePlan(("route",), (0, 2), (base, link)).prepare(
        articulation, jnp.asarray([[0.0, 0.7, 0.0], [0.5, -0.2, 0.0]])
    )
    return route


def qualify(step: float) -> dict[str, object]:
    route = _prepared_route()
    configuration = jnp.asarray([0.4])
    velocity = jnp.asarray([-0.8])
    evaluation = route.evaluate(configuration, velocity)
    plus = route.lengths(configuration + step * velocity)
    minus = route.lengths(configuration - step * velocity)
    finite_difference = (plus - minus) / (2.0 * step)
    derivative_error = float(
        np.max(
            np.abs(
                np.asarray(evaluation.route_length_rates_m_per_s)
                - np.asarray(finite_difference)
            )
        )
    )
    _, power = route.tensile_force_pullback(
        configuration, velocity, jnp.asarray([750.0])
    )
    power_error = float(np.abs(np.asarray(power.power_residual_W)))
    tolerance = max(1.0e-8, 10.0 * step * step)
    return {
        "construction": "piecewise Euclidean length through body-fixed points",
        "sign": "positive tension uses Q=-J_L^T T and route power -T*dL/dt",
        "central_difference_step": step,
        "jvp_vs_central_difference_max_abs_m_per_s": derivative_error,
        "virtual_power_residual_W": power_error,
        "tolerance": tolerance,
        "passed": bool(evaluation.successful[0])
        and bool(power.successful)
        and derivative_error <= tolerance
        and power_error <= tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=1.0e-4)
    arguments = parser.parse_args()
    if not np.isfinite(arguments.step) or arguments.step <= 0.0:
        raise ValueError("--step must be positive and finite.")
    report = qualify(arguments.step)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
