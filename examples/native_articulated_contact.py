"""Drop a native capsule bar onto a plane and inspect its contact evidence.

A horizontal capsule on a vertical prismatic joint falls 5 mm under gravity
onto a static plane with the native position-level inelastic Coulomb
hard-contact law. Each step reports its status, covered time, the two capsule
endpoint supports and the mechanical-energy ledger, including the explicit
inelastic correction loss. Any refused step raises instead of being hidden.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

from phydrax import DimensionalScaleContract
from phydrax.applications.contact import (
    ArticulatedActuationMode,
    ArticulatedContactCertificatePolicy,
    ArticulatedContactStepPolicy,
    ContactConeSolverPlan,
)
from phydrax.applications.robotics import (
    lower_native_robot_tree,
    NativeArticulatedPlantPlan,
    NativeRobotModel,
    NativeTreeJoint,
    prepare_articulated_plant,
    RobotCollisionGeometryPlan,
    RobotContactMaterialTable,
    RobotFramePlan,
    RobotNameIDMap,
)
from phydrax.discretization.contact import (
    AdmissiblePathPlan,
    ArticulatedPrimitiveKind,
)
from phydrax.discretization.particle import (
    ReducedSemiImplicitVelocityEulerStepPolicy,
)
from phydrax.dynamics import PlantStepContext


GAP = 1.0e-9
STEP = 0.005
RADIUS = 0.05


def build_model() -> NativeRobotModel:
    bodies = RobotNameIDMap(("bar", "base"))
    bar, _ = bodies.ids
    collision = RobotCollisionGeometryPlan(
        RobotNameIDMap(("rod",)),
        (ArticulatedPrimitiveKind.CAPSULE,),
        np.asarray([bar], dtype=np.int64),
        np.asarray([[-0.4, 0.0, 0.0]]),
        np.asarray([[0.4, 0.0, 0.0]]),
        np.asarray([RADIUS]),
        np.asarray([1], dtype=np.int64),
        materials=RobotContactMaterialTable(
            np.asarray([[1, 1]], dtype=np.int64), np.asarray([0.8]), np.asarray([0.6])
        ),
        plane_names=RobotNameIDMap(("floor",)),
        plane_normals=np.asarray([[0.0, 0.0, 1.0]]),
        plane_offsets=np.asarray([0.0]),
        plane_material_ids=np.asarray([1], dtype=np.int64),
    )
    return lower_native_robot_tree(
        "capsule-bar",
        bodies,
        np.asarray([2.0, 10.0]),
        np.stack((np.diag([0.01, 0.1, 0.1]), 0.1 * np.eye(3))),
        np.asarray([[0.0, 0.0, RADIUS], [0.0, 0.0, -1.0]]),
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
        "base",
        (NativeTreeJoint("lift", "prismatic", "base", "bar", axis=(0.0, 0.0, 1.0)),),
        DimensionalScaleContract.si(),
        RobotFramePlan(
            RobotNameIDMap(("bar_frame",)),
            np.asarray([bar], dtype=np.int64),
            np.zeros((1, 3)),
            np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        ),
        collision=collision,
    )


def build_policy() -> ArticulatedContactStepPolicy:
    certificate = ArticulatedContactCertificatePolicy(
        contact_reference_velocity=1.0,
        hinge_reference_velocity=1.0,
        prismatic_reference_velocity=1.0,
        contact_impulse_tolerance=1.0e-9,
        hinge_impulse_tolerance=1.0e-9,
        prismatic_impulse_tolerance=1.0e-9,
        contact_gap_tolerance=GAP,
        hinge_gap_tolerance=GAP,
        prismatic_gap_tolerance=GAP,
        slip_speed_tolerance=1.0e-9,
        complementarity_work_tolerance=1.0e-8,
        friction_work_tolerance=1.0e-8,
        contact_work_tolerance=1.0e-8,
        contact_energy_tolerance=1.0e-8,
        ledger_tolerance=1.0e-8,
        energy_surplus_tolerance=1.0e-8,
        momentum_energy_tolerance=1.0e-14,
        discretization_energy_budget=1.0,
    )
    return ArticulatedContactStepPolicy(
        certificate,
        AdmissiblePathPlan(GAP),
        cone_solver=ContactConeSolverPlan(
            maximum_iterations=2000,
            absolute_tolerance=1.0e-14,
            relative_tolerance=1.0e-13,
        ),
        smooth=ReducedSemiImplicitVelocityEulerStepPolicy(
            maximum_step_size=1.0, absolute_energy_tolerance=1.0
        ),
        root_absolute_tolerance=1.0e-13,
        root_relative_tolerance=1.0e-12,
    )


def main() -> None:
    plan = NativeArticulatedPlantPlan(
        build_policy(),
        gravity=jnp.asarray([0.0, 0.0, -9.81]),
        actuation_mode=ArticulatedActuationMode.AUTONOMOUS,
        activation_distance=0.02,
        # Device LBVH route with explicit budgets: one capsule-plane pair has
        # two endpoint witness rows; traversal work is bounded by 256 visits.
        search_backend="lbvh",
        route_capacities=(0, 0, 0, 0, 0, 2),
        maximum_traversal_visits=256,
        initial_lifted_configuration=[0.005],
    )
    plant = prepare_articulated_plant(build_model(), plan)
    parameters = plant.bind_parameters()
    reset = plant.reset(jax.random.key(0), parameters)
    if not bool(reset.successful):
        raise RuntimeError(f"reset refused with status {int(reset.status)}")
    state = reset.accepted_state
    records = []
    for _ in range(12):
        context = PlantStepContext(state.time, state.time + STEP, state.step_index)
        # The plant's compiled step body is reused by every repeated step.
        result = plant.step(context, state, None, parameters)
        evidence = result.evidence.step
        if not bool(result.successful):
            raise RuntimeError(
                f"step {int(state.step_index)} refused with status {int(result.status)}"
            )
        segment = evidence.last_segment
        valid = np.asarray(segment.block_valid)
        supports = np.asarray(segment.impulse)[valid, 0]
        records.append(
            {
                "time": float(result.accepted_state.time),
                "height": float(result.accepted_state.payload.configuration[0]),
                "velocity": float(result.accepted_state.payload.velocity[0]),
                "covered_fraction": float(evidence.covered_fraction),
                "segments": int(evidence.accepted_segments),
                "free_bypass": bool(evidence.free_bypass),
                "endpoint_supports": supports.tolist(),
                "contact_work": float(evidence.ledger.contact_work),
                "correction_loss": float(evidence.ledger.correction_loss),
                "ledger_identity_residual": float(evidence.ledger.identity_residual),
            }
        )
        state = result.accepted_state
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
