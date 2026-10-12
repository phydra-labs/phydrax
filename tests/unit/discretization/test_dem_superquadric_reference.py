#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Recorded pre-cutover SQ trajectories, not a second stepping implementation."""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

import phydrax as phx
from phydrax.discretization import DEMRuntimeState, RigidContactGeometry
from phydrax.equations import CompiledDiscreteElementProblem


def _unilateral_damping(stiffness: float, mass: float, restitution: float) -> float:
    """Independent release-at-zero-force oscillator calibration.

    The release angle is 2 acos(zeta), so e=exp(-2 zeta acos(zeta)/
    sqrt(1-zeta²)). This uses the closed oscillator equation, not a DEM law.
    """

    def residual(ratio: float) -> float:
        return float(
            np.exp(-2.0 * ratio * np.arccos(ratio) / np.sqrt(1.0 - ratio**2))
            - restitution
        )

    ratio = brentq(residual, 0.0, 1.0 - 1.0e-10)
    return float(2.0 * ratio * np.sqrt(stiffness * mass))


def _reference_normal_speed(
    geometry: RigidContactGeometry,
    velocity: np.ndarray,
    angular: np.ndarray,
    *,
    stationary_wall: bool = False,
) -> np.ndarray:
    left = np.asarray(geometry.left_owner_index, dtype=np.int32)
    right = np.asarray(geometry.right_owner_index, dtype=np.int32)
    right_index = np.maximum(right, 0)
    left_velocity = velocity[left] + np.cross(
        angular[left], np.asarray(geometry.left_contact_arm)
    )
    right_velocity = (
        np.zeros_like(left_velocity)
        if stationary_wall
        else velocity[right_index]
        + np.cross(angular[right_index], np.asarray(geometry.right_contact_arm))
    )
    return np.sum((left_velocity - right_velocity) * np.asarray(geometry.normal), axis=-1)


def _common_point_normal_moment(
    geometry: RigidContactGeometry,
    force: np.ndarray,
    count: int,
    *,
    stationary_wall: bool = False,
) -> np.ndarray:
    moment = np.zeros((count, 3), dtype=np.float64)
    left = np.asarray(geometry.left_owner_index, dtype=np.int32)
    right = np.asarray(geometry.right_owner_index, dtype=np.int32)
    np.add.at(moment, left, np.cross(np.asarray(geometry.left_owner_arm), force))
    particle_right = (right >= 0) & ~np.asarray(stationary_wall, dtype=np.bool_)
    np.add.at(
        moment,
        right[particle_right],
        np.cross(
            np.asarray(geometry.right_owner_arm)[particle_right], -force[particle_right]
        ),
    )
    return moment


def _compile_recorded_case(
    inputs: dict[str, Any], scenario: str
) -> tuple[CompiledDiscreteElementProblem, DEMRuntimeState]:
    """Prepare the recorded physical inputs without a reference stepping engine."""
    count = len(inputs["masses"])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int64),
        jnp.asarray(inputs["masses"], dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    shapes = phx.discretization.SuperquadricSetPlan(
        jnp.asarray(inputs["semi_axes"], dtype=jnp.float64),
        jnp.asarray(inputs["first_exponent"], dtype=jnp.float64),
        jnp.asarray(inputs["second_exponent"], dtype=jnp.float64),
        jnp.asarray(inputs["material_ids"], dtype=jnp.int32),
        contact_geometry=phx.discretization.SuperquadricContactPlan(
            iterations=inputs["iterations"]
        ),
    )
    walls = (
        ()
        if scenario == "pair"
        else (
            phx.discretization.TriangleWallPlan(
                jnp.asarray(inputs["wall_vertices"], dtype=jnp.float64),
                jnp.asarray(inputs["wall_faces"], dtype=jnp.int32),
                jnp.zeros((len(inputs["wall_faces"]),), dtype=jnp.int32),
            ),
        )
    )
    compiled = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "recorded-superquadric-parity",
            phx.equations.DEMMaterialTable(
                jnp.asarray(inputs["young_modulus"], dtype=jnp.float64),
                jnp.asarray(inputs["poisson_ratio"], dtype=jnp.float64),
                jnp.asarray(inputs["restitution"], dtype=jnp.float64),
                jnp.asarray(inputs["friction"], dtype=jnp.float64),
            ),
            gravity=jnp.zeros((3,), dtype=jnp.float64),
            walls=walls,
        ),
        particles,
        shapes,
        phx.discretization.CompliantDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LinearSpringDashpotNormalPlan(
                    inputs["normal_stiffness"]
                ),
                tangential=phx.discretization.CundallStrackTangentialPlan(
                    inputs["tangential_stiffness"]
                ),
            ),
            wall_geometry=phx.discretization.SuperquadricTriangleContactPlan(),
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2
        ),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray(inputs["position"], dtype=jnp.float64),
        jnp.asarray(inputs["velocity"], dtype=jnp.float64),
        jnp.asarray(inputs["angular_velocity"], dtype=jnp.float64),
        orientation=jnp.asarray(inputs["orientation"], dtype=jnp.float64),
    )
    return compiled, state


@pytest.mark.parametrize(
    "scenario", ["pair", "triangle_wall"], ids=["pair", "triangle-wall"]
)
def test_shared_step_matches_recorded_superquadric_trajectory(scenario: str) -> None:
    path = Path(__file__).parents[2] / "fixtures" / "dem" / "superquadric_reference.json"
    reference: dict[str, Any] = json.loads(path.read_text())
    case = reference[scenario]
    inputs = case["inputs"]
    count = len(inputs["masses"])
    compiled, state = _compile_recorded_case(inputs, scenario)
    dt = jnp.asarray(inputs["dt"], dtype=jnp.float64)
    step = float(dt)
    masses = np.asarray(inputs["masses"], dtype=np.float64)
    minimum_mass = float(np.min(masses))
    effective_mass = float(1.0 / np.sum(1.0 / masses))
    damping = _unilateral_damping(
        inputs["normal_stiffness"],
        effective_mass,
        inputs["restitution"][0][0],
    )
    force_scale = float(
        max(
            np.max(np.linalg.norm(np.asarray(output["force"]), axis=-1))
            for output in case["outputs"]
        )
    )
    assert (
        step
        * (np.sqrt(inputs["normal_stiffness"] / minimum_mass) + damping / minimum_mass)
        < 0.01
    )
    inertia_z = np.asarray(state.body_properties.inertia_com)[:, 2, 2]
    angular_correction = np.zeros((count, 3), dtype=np.float64)
    angle_correction = np.zeros((count,), dtype=np.float64)
    previous_normal_moment = np.zeros((count, 3), dtype=np.float64)
    for index, expected in enumerate(case["outputs"]):
        detail = compiled.dynamics.step_detailed(
            jnp.asarray(index, dtype=jnp.int32),
            index * dt,
            state,
            dt,
            None,
        )
        assert bool(detail.successful) == expected["successful"], f"step {index}"
        state = detail.accepted_state
        actual = {
            "position": state.kinematics.position,
            "velocity": state.kinematics.velocity,
            "orientation": state.kinematics.orientation,
            "angular_velocity": state.kinematics.angular_velocity,
            "force": detail.evaluation.loads.total.force,
            "torque": detail.evaluation.loads.total.torque,
            "pair_gap": detail.evaluation.geometry.geometry.gap,
            "pair_force": detail.evaluation.particle_contact.pair_force,
        }
        geometry = detail.evaluation.geometry.geometry
        reference_velocity = np.asarray(expected["velocity"], dtype=np.float64)
        reference_angular = np.asarray(expected["angular_velocity"], dtype=np.float64)
        pair_force = np.asarray(expected["pair_force"], dtype=np.float64).reshape((-1, 3))
        reference_pair_gap = np.asarray(expected["pair_gap"], dtype=np.float64)
        pair_force = pair_force + (
            inputs["normal_stiffness"]
            * (np.asarray(geometry.overlap) - np.maximum(-reference_pair_gap, 0.0))
            - damping
            * (
                np.asarray(geometry.normal_velocity)
                - _reference_normal_speed(geometry, reference_velocity, reference_angular)
            )
        )[:, None] * np.asarray(geometry.normal)
        corrected_force = np.zeros((count, 3), dtype=np.float64)
        np.add.at(corrected_force, np.asarray(geometry.left_owner_index), pair_force)
        np.add.at(corrected_force, np.asarray(geometry.right_owner_index), -pair_force)
        normal_moment = _common_point_normal_moment(
            geometry,
            np.asarray(detail.evaluation.particle_contact.normal_force),
            count,
        )
        if scenario == "triangle_wall":
            boundary = detail.evaluation.boundaries[0]
            wall_geometry = boundary.geometry
            wall_force = np.asarray(expected["wall_force"], dtype=np.float64) + (
                inputs["normal_stiffness"]
                * (
                    np.asarray(wall_geometry.overlap)
                    - np.maximum(-np.asarray(expected["wall_gap"]), 0.0)
                )
                - damping
                * (
                    np.asarray(wall_geometry.normal_velocity)
                    - _reference_normal_speed(
                        wall_geometry,
                        reference_velocity,
                        reference_angular,
                        stationary_wall=True,
                    )
                )
            )[:, None] * np.asarray(wall_geometry.normal)
            np.add.at(
                corrected_force, np.asarray(wall_geometry.left_owner_index), wall_force
            )
            normal_moment += _common_point_normal_moment(
                wall_geometry,
                np.asarray(boundary.contact.normal_force),
                count,
                stationary_wall=True,
            )

        # The deleted engine omitted normal lever moments and reported loads at
        # full-step velocity. The shared KDK reports its half-kick contact sample.
        # Correct those explicit measurement/model defects; do not hide them in
        # a blanket trajectory tolerance. Both recorded cases rotate about the
        # principal z axis, giving an analytic normal-impulse correction.
        angle_correction += step * (
            angular_correction[:, 2]
            + 0.5 * step * previous_normal_moment[:, 2] / inertia_z
        )
        angular_correction += (
            0.5 * step * (previous_normal_moment + normal_moment) / inertia_z[:, None]
        )
        previous_normal_moment = normal_moment
        expected_orientation = np.asarray(
            expected["orientation"], dtype=np.float64
        ).copy()
        cosine = np.cos(0.5 * angle_correction)
        sine = np.sin(0.5 * angle_correction)
        scalar = expected_orientation[:, 0].copy()
        axial = expected_orientation[:, 3].copy()
        expected_orientation[:, 0] = cosine * scalar - sine * axial
        expected_orientation[:, 3] = sine * scalar + cosine * axial
        np.testing.assert_allclose(
            actual["orientation"], expected_orientation, rtol=2.0e-8, atol=2.0e-10
        )
        np.testing.assert_allclose(
            actual["angular_velocity"],
            reference_angular + angular_correction,
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            actual["torque"],
            np.asarray(expected["torque"]) + normal_moment,
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            actual["force"], corrected_force, rtol=2.0e-8, atol=2.0e-10
        )
        np.testing.assert_allclose(
            actual["pair_force"], pair_force, rtol=2.0e-8, atol=2.0e-10
        )
        np.testing.assert_allclose(
            actual["pair_gap"], reference_pair_gap, rtol=2.0e-8, atol=2.0e-10
        )
        # Half/full damping samples differ by at most gamma*dt*F/m; summing
        # their half impulses yields O(n² dt² gamma F/m²) velocity error.
        # The fixed fixture resolves every oscillator by >1000 time steps.
        velocity_error = (
            2.0 * (index + 1) ** 2 * step**2 * damping * force_scale / minimum_mass**2
        )
        position_error = (index + 1) * step * velocity_error
        np.testing.assert_allclose(
            actual["velocity"],
            reference_velocity,
            rtol=2.0e-8,
            atol=2.0e-10 + velocity_error,
        )
        np.testing.assert_allclose(
            actual["position"],
            expected["position"],
            rtol=2.0e-8,
            atol=2.0e-10 + position_error,
        )
        np.testing.assert_array_equal(
            state.particle_history.pair_keys,
            np.asarray(expected["pair_keys"], dtype=np.int64).reshape((-1, 5)),
        )
        if scenario == "triangle_wall":
            boundary = detail.evaluation.boundaries[0]
            np.testing.assert_allclose(
                boundary.geometry.gap, expected["wall_gap"], rtol=2.0e-8, atol=2.0e-10
            )
            np.testing.assert_allclose(
                boundary.particle_force, wall_force, rtol=2.0e-8, atol=2.0e-10
            )
            np.testing.assert_allclose(
                boundary.reaction_force,
                -np.sum(wall_force, axis=0),
                rtol=2.0e-8,
                atol=2.0e-10,
            )
            np.testing.assert_allclose(
                boundary.reaction_torque,
                expected["wall_reaction_torque"],
                rtol=2.0e-8,
                atol=2.0e-10,
            )
            np.testing.assert_array_equal(
                state.boundary_histories[0].pair_keys, expected["wall_keys"]
            )
