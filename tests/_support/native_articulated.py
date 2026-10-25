#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic native articulated contact fixtures shared by step/plant tests.

Every fixture is a direct native model whose contact mechanics have a closed
form: a vertical slider on a plane, a planar (x, z) slider with Coulomb
friction, a horizontal capsule bar on a plane and a hinge pendulum sweeping
past a static obstacle. Tolerances are explicit physical quantities.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract
from phydrax.applications.contact._articulated_step import (
    ArticulatedContactCertificatePolicy,
    ArticulatedContactStepPolicy,
)
from phydrax.applications.contact._cone import ContactConeSolverPlan
from phydrax.applications.robotics._native_model import (
    lower_native_robot_tree,
    NativeRobotModel,
    NativeTreeJoint,
    RobotCollisionGeometryPlan,
    RobotContactMaterialTable,
    RobotFramePlan,
    RobotNameIDMap,
)
from phydrax.discretization.contact._articulated_geometry import (
    ArticulatedPrimitiveKind,
)
from phydrax.discretization.contact._ccd import AdmissiblePathPlan
from phydrax.discretization.particle._reduced_dynamics import (
    ReducedSemiImplicitVelocityEulerStepPolicy,
)
from phydrax.discretization.particle._rigid_unilateral import ReducedJointStopPlan
from phydrax.linalg import MaterializationPolicy
from phydrax.linalg.svd import DenseSVD, SVDSolvePolicy


GAP_TOLERANCE = 1.0e-9
WORK_TOLERANCE = 1.0e-8


def certificate_policy(**overrides: float) -> ArticulatedContactCertificatePolicy:
    values: dict[str, float] = {
        "contact_reference_velocity": 1.0,
        "hinge_reference_velocity": 1.0,
        "prismatic_reference_velocity": 1.0,
        "contact_impulse_tolerance": 1.0e-9,
        "hinge_impulse_tolerance": 1.0e-9,
        "prismatic_impulse_tolerance": 1.0e-9,
        "contact_gap_tolerance": GAP_TOLERANCE,
        "hinge_gap_tolerance": GAP_TOLERANCE,
        "prismatic_gap_tolerance": GAP_TOLERANCE,
        "slip_speed_tolerance": 1.0e-9,
        "complementarity_work_tolerance": WORK_TOLERANCE,
        "friction_work_tolerance": WORK_TOLERANCE,
        "contact_work_tolerance": WORK_TOLERANCE,
        "contact_energy_tolerance": WORK_TOLERANCE,
        "ledger_tolerance": 1.0e-8,
        "energy_surplus_tolerance": 1.0e-8,
        "momentum_energy_tolerance": 1.0e-14,
        "discretization_energy_budget": 1.0,
    }
    values.update(overrides)
    return ArticulatedContactCertificatePolicy(**values)


def step_policy(**overrides: Any) -> ArticulatedContactStepPolicy:
    certificate = overrides.pop("certificate", certificate_policy())
    keywords: dict[str, Any] = {
        "cone_solver": ContactConeSolverPlan(
            maximum_iterations=2000,
            absolute_tolerance=1.0e-14,
            relative_tolerance=1.0e-13,
        ),
        "smooth": ReducedSemiImplicitVelocityEulerStepPolicy(
            maximum_step_size=1.0,
            absolute_energy_tolerance=1.0,
            relative_energy_tolerance=1.0e-2,
        ),
        "maximum_root_iterations": 200,
        "root_absolute_tolerance": 1.0e-13,
        "root_relative_tolerance": 1.0e-12,
        "maximum_subdivision_depth": 8,
        "maximum_segment_attempts": 64,
        "minimum_segment_duration": 1.0e-6,
        # Explicit bounded native dense SVD certifies derivative regularity of
        # these few-row fixtures; production profiles declare their own bound.
        "derivative_regularity": SVDSolvePolicy(
            DenseSVD(algorithm="qr"),
            materialization=MaterializationPolicy(max_entries=4096),
        ),
    }
    keywords.update(overrides)
    return ArticulatedContactStepPolicy(
        certificate,
        AdmissiblePathPlan(GAP_TOLERANCE, maximum_intervals=4096),
        **keywords,
    )


def _frames(body_id: int) -> RobotFramePlan:
    return RobotFramePlan(
        RobotNameIDMap(("tool",)),
        np.asarray([body_id], dtype=np.int64),
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )


def _materials(static: float, dynamic: float) -> RobotContactMaterialTable:
    return RobotContactMaterialTable(
        np.asarray([[1, 1]], dtype=np.int64), np.asarray([static]), np.asarray([dynamic])
    )


def _identity(count: int, scale: float) -> np.ndarray:
    return np.tile(scale * np.eye(3), (count, 1, 1))


_ROUTE_BUDGETS: dict[str, tuple[int, int, int, int, int, int]] = {}


def _declare(
    budget: tuple[int, int, int, int, int, int], model: NativeRobotModel, /
) -> NativeRobotModel:
    """Record the fixture's explicit witness-row budget beside its geometry.

    Order: vertex-vertex, edge-vertex, edge-edge, face-vertex, vertex-plane,
    edge-plane rows (capsule-plane pairs use two endpoint rows).
    """

    _ROUTE_BUDGETS[model.model_id] = budget
    return model


def route_budget(model: NativeRobotModel, /) -> tuple[int, int, int, int, int, int]:
    """The declared route capacities of one fixture model."""

    return _ROUTE_BUDGETS[model.model_id]


def dof(prepared: Any, joint_name: str) -> int:
    """Generalized velocity index of one named scalar joint."""

    identifier = prepared.model.joint_names.ids[
        prepared.model.joint_names.names.index(joint_name)
    ]
    edges = [int(value) for value in np.asarray(prepared.articulation.joint_ids)]
    return prepared.articulation.joint_velocity_slices[edges.index(identifier)].start


def slider_model(
    *,
    mass: float = 2.0,
    radius: float = 0.1,
    friction: tuple[float, float] = (0.8, 0.6),
    stop_bounds: tuple[float, float] | None = None,
    floor: bool = True,
) -> NativeRobotModel:
    """Vertical prismatic slider with a sphere touching the plane ``z = 0``.

    The slider COM starts at height ``radius`` so the sphere's physical gap is
    zero at ``q = 0``; ``q`` is the upward displacement.
    """

    bodies = RobotNameIDMap(("base", "slider"))
    base, slider = bodies.ids
    joints = RobotNameIDMap(("lift",))
    collision = RobotCollisionGeometryPlan(
        RobotNameIDMap(("ball",)),
        (ArticulatedPrimitiveKind.SPHERE,),
        np.asarray([slider], dtype=np.int64),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        np.asarray([radius]),
        np.asarray([1], dtype=np.int64),
        materials=_materials(*friction),
        **(
            {
                "plane_names": RobotNameIDMap(("floor",)),
                "plane_normals": np.asarray([[0.0, 0.0, 1.0]]),
                "plane_offsets": np.asarray([0.0]),
                "plane_material_ids": np.asarray([1], dtype=np.int64),
            }
            if floor
            else {}
        ),
    )
    stops = (
        None
        if stop_bounds is None
        else ReducedJointStopPlan(
            np.asarray([joints.ids[0]], dtype=np.int64),
            np.asarray([stop_bounds[0]]),
            np.asarray([stop_bounds[1]]),
        )
    )
    return _declare(
        (0, 0, 0, 0, 1 if floor else 0, 0),
        lower_native_robot_tree(
            "slider",
            bodies,
            np.asarray([10.0, mass]),
            _identity(2, 0.1),
            np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, radius]]),
            np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
            "base",
            (
                NativeTreeJoint(
                    "lift", "prismatic", "base", "slider", axis=(0.0, 0.0, 1.0)
                ),
            ),
            DimensionalScaleContract.si(),
            _frames(slider),
            collision=collision,
            stops=stops,
        ),
    )


def planar_slider_model(
    *,
    carriage_mass: float = 3.0,
    slider_mass: float = 1.0,
    radius: float = 0.1,
    friction: tuple[float, float] = (0.8, 0.5),
    drive_bounds: tuple[float, float] | None = None,
) -> NativeRobotModel:
    """Carriage sliding along x carrying a vertical slider whose sphere rests on z = 0."""

    bodies = RobotNameIDMap(("base", "carriage", "slider"))
    base, carriage, slider = bodies.ids
    joints = RobotNameIDMap(("drive", "lift"))
    collision = RobotCollisionGeometryPlan(
        RobotNameIDMap(("ball",)),
        (ArticulatedPrimitiveKind.SPHERE,),
        np.asarray([slider], dtype=np.int64),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        np.asarray([radius]),
        np.asarray([1], dtype=np.int64),
        materials=_materials(*friction),
        plane_names=RobotNameIDMap(("floor",)),
        plane_normals=np.asarray([[0.0, 0.0, 1.0]]),
        plane_offsets=np.asarray([0.0]),
        plane_material_ids=np.asarray([1], dtype=np.int64),
    )
    stops = (
        None
        if drive_bounds is None
        else ReducedJointStopPlan(
            np.asarray([joints.ids[joints.names.index("drive")]], dtype=np.int64),
            np.asarray([drive_bounds[0]]),
            np.asarray([drive_bounds[1]]),
        )
    )
    del base, carriage
    return _declare(
        (0, 0, 0, 0, 1, 0),
        lower_native_robot_tree(
            "planar-slider",
            bodies,
            np.asarray([10.0, carriage_mass, slider_mass]),
            _identity(3, 0.1),
            np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, radius]]),
            np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (3, 1)),
            "base",
            (
                NativeTreeJoint(
                    "drive", "prismatic", "base", "carriage", axis=(1.0, 0.0, 0.0)
                ),
                NativeTreeJoint(
                    "lift", "prismatic", "carriage", "slider", axis=(0.0, 0.0, 1.0)
                ),
            ),
            DimensionalScaleContract.si(),
            _frames(slider),
            collision=collision,
            stops=stops,
        ),
    )


def capsule_bar_model(
    *, mass: float = 2.0, half_length: float = 0.4, radius: float = 0.05
) -> NativeRobotModel:
    """Horizontal capsule bar on a vertical slider, touching z = 0 along its axis."""

    bodies = RobotNameIDMap(("bar", "base"))
    bar, base = bodies.ids
    collision = RobotCollisionGeometryPlan(
        RobotNameIDMap(("rod",)),
        (ArticulatedPrimitiveKind.CAPSULE,),
        np.asarray([bar], dtype=np.int64),
        np.asarray([[-half_length, 0.0, 0.0]]),
        np.asarray([[half_length, 0.0, 0.0]]),
        np.asarray([radius]),
        np.asarray([1], dtype=np.int64),
        materials=_materials(0.8, 0.6),
        plane_names=RobotNameIDMap(("floor",)),
        plane_normals=np.asarray([[0.0, 0.0, 1.0]]),
        plane_offsets=np.asarray([0.0]),
        plane_material_ids=np.asarray([1], dtype=np.int64),
    )
    del base
    return _declare(
        (0, 0, 0, 0, 0, 2),
        lower_native_robot_tree(
            "capsule-bar",
            bodies,
            np.asarray([mass, 10.0]),
            np.stack((np.diag([0.01, 0.1, 0.1]), 0.1 * np.eye(3))),
            np.asarray([[0.0, 0.0, radius], [0.0, 0.0, -1.0]]),
            np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
            "base",
            (NativeTreeJoint("lift", "prismatic", "base", "bar", axis=(0.0, 0.0, 1.0)),),
            DimensionalScaleContract.si(),
            _frames(bar),
            collision=collision,
        ),
    )


def pendulum_model(
    *,
    length: float = 1.0,
    radius: float = 0.1,
    obstacle_angle: float | None = 0.5 * np.pi,
    stop_bounds: tuple[float, float] | None = None,
) -> NativeRobotModel:
    """Hinge about z at the origin; link sphere at ``length``; optional obstacle.

    The static obstacle sphere is attached to the fixed base at angle
    ``obstacle_angle`` on the swept circle, so the rotating link collides with
    it partway through a full turn while full-turn endpoints remain clear.
    """

    bodies = RobotNameIDMap(("base", "link"))
    base, link = bodies.ids
    names = ["tip"]
    kinds = [ArticulatedPrimitiveKind.SPHERE]
    owners = [link]
    centers = [[0.0, 0.0, 0.0]]
    radii = [radius]
    if obstacle_angle is not None:
        names.append("obstacle")
        kinds.append(ArticulatedPrimitiveKind.SPHERE)
        owners.append(base)
        # Base COM sits at (0, 0, -1); express the obstacle in its COM frame.
        centers.append(
            [
                length * np.cos(obstacle_angle),
                length * np.sin(obstacle_angle),
                1.0,
            ]
        )
        radii.append(radius)
    order = np.argsort(names)
    collision = RobotCollisionGeometryPlan(
        RobotNameIDMap(tuple(names[index] for index in order)),
        tuple(kinds[index] for index in order),
        np.asarray([owners[index] for index in order], dtype=np.int64),
        np.asarray([centers[index] for index in order]),
        np.asarray([centers[index] for index in order]),
        np.asarray([radii[index] for index in order]),
        np.ones((len(names),), dtype=np.int64),
        materials=_materials(0.0, 0.0),
        plane_names=RobotNameIDMap(("floor",)),
        plane_normals=np.asarray([[0.0, 0.0, 1.0]]),
        plane_offsets=np.asarray([-0.5]),
        plane_material_ids=np.asarray([1], dtype=np.int64),
    )
    return _declare(
        (0 if obstacle_angle is None else 1, 0, 0, 0, 1, 0),
        lower_native_robot_tree(
            "pendulum",
            bodies,
            np.asarray([10.0, 1.0]),
            _identity(2, 0.01),
            np.asarray([[0.0, 0.0, -1.0], [length, 0.0, 0.0]]),
            np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
            "base",
            (
                NativeTreeJoint(
                    "swing",
                    "hinge",
                    "base",
                    "link",
                    anchor=(0.0, 0.0, 0.0),
                    axis=(0.0, 0.0, 1.0),
                ),
            ),
            DimensionalScaleContract.si(),
            _frames(link),
            collision=collision,
            stops=(
                None
                if stop_bounds is None
                else ReducedJointStopPlan(
                    np.asarray([0], dtype=np.int64),
                    np.asarray([stop_bounds[0]]),
                    np.asarray([stop_bounds[1]]),
                )
            ),
        ),
    )


def zero_gravity() -> jnp.ndarray:
    return jnp.zeros((3,))


def standard_gravity() -> jnp.ndarray:
    return jnp.asarray([0.0, 0.0, -9.81])


LBVH_TRAVERSAL_VISITS = 256


def contact_step(
    model: NativeRobotModel,
    policy: ArticulatedContactStepPolicy | None = None,
    /,
    *,
    search_backend: str = "exhaustive",
    **keywords: Any,
) -> tuple[Any, Any]:
    """Prepare the native model and its coupled contact step.

    Route capacities are the fixture's declared budget; the LBVH backend also
    receives an explicit traversal-visit budget.
    """

    from phydrax.applications.contact._articulated_step import (
        PreparedArticulatedContactStep,
    )

    prepared = model.prepare()
    collision = model.collision
    materials: dict[str, Any] = {}
    if collision is not None:
        materials = {
            "material_pairs": np.asarray(collision.materials.material_pairs),
            "static_friction": np.asarray(collision.materials.static_friction),
            "dynamic_friction": np.asarray(collision.materials.dynamic_friction),
            "activation_distance": keywords.pop("activation_distance", 0.05),
            "route_capacities": keywords.pop("route_capacities", route_budget(model)),
            "search_backend": search_backend,
            "maximum_traversal_visits": (
                LBVH_TRAVERSAL_VISITS if search_backend == "lbvh" else None
            ),
        }
    step = PreparedArticulatedContactStep(
        prepared.articulation,
        prepared.coordinates,
        step_policy() if policy is None else policy,
        geometry=prepared.collision,
        stops=prepared.stops,
        motors=prepared.motors,
        damping=prepared.damping,
        **materials,
        **keywords,
    )
    return prepared, step
