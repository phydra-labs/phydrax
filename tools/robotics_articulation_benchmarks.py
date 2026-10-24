"""Benchmark native reduced articulation, prepared mass and the native contact plant.

The default invocation keeps the legacy smooth fixed-base suite (forward
kinematics, frame-Jacobian actions, RNEA, ABA and a status-aware smooth
rollout). ``--campaign quick`` and ``--campaign scaling`` run explicit,
bounded diagnostic rows with phase-separated evidence:

* prepared-mass rows: host topology arrays, plan construction, preparation,
  explicit host-to-device transfer, mass binding, reused vector,
  repeated-vector and multiple-RHS inverse actions, matched preparation-inside-map
  work, independent ``M(q)`` actions, configuration JVP/VJP, smooth ABA/RNEA
  and an independent dense ``JᵀGJ`` reference;
* native plant rows: host model construction, model and plant preparation,
  explicit transfer, reset, the canonical and AOT plant step, public step
  components, the single-case contact step with fixed-branch JVP/VJP, and the
  complete-state ``PlantControlDynamics`` rollout.

Every compiled phase reports lowering, compilation, first and warmed
execution, XLA compiler estimates and logical array bytes. Provider-external
memory is never measured here and is reported as such. The native contact law
is Phydrax's position-level inelastic hard contact; no provider-law
equivalence is claimed. Unsuccessful, refused and inconclusive work is
reported beside its timing and never enters a time ratio.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import JaxRuntimeError

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import DimensionalScaleContract
from phydrax._jax_config import configuration_values
from phydrax.applications.contact import (
    ArticulatedActuationMode,
    ArticulatedContactCertificatePolicy,
    ArticulatedContactStepPolicy,
    ArticulatedContactStepStatus,
    ContactConeSolverPlan,
)
from phydrax.applications.robotics import (
    ArticulatedPlantCommand,
    ArticulatedPlantStatus,
    lower_native_robot_tree,
    NativeArticulatedPlantPlan,
    NativeRobotModel,
    NativeTreeJoint,
    parse_urdf_text,
    prepare_articulated_plant,
    PreparedArticulatedPlant,
    PreparedNativeRobotModel,
    RobotCollisionGeometryPlan,
    RobotContactMaterialTable,
    RobotFramePlan,
    RobotNameIDMap,
)
from phydrax.control import (
    DiscreteControlDynamics,
    PiecewiseConstantControlParameterization,
    PlantControlDynamics,
)
from phydrax.discretization import (
    AdmissiblePathPlan,
    AdmissiblePathStatus,
    ArticulatedMassPolicy,
    ArticulatedMassStatus,
    ArticulatedPrimitiveKind,
    ContactSearchStatus,
    FixedJointSetPlan,
    HingeJointSetPlan,
    ParticleSetPlan,
    prepare_articulated_mass,
    PreparedReducedArticulation,
    PrismaticJointSetPlan,
    reduced_forward_dynamics,
    reduced_inverse_dynamics,
    reduced_mass_matrix,
    reduced_semi_implicit_velocity_euler_step,
    ReducedArticulationPlan,
    ReducedJointStopPlan,
    ReducedSemiImplicitVelocityEulerStepPolicy,
    RigidBodySetPlan,
    RigidJointGraphPlan,
)
from phydrax.dynamics import (
    ArrayPyTreeSchema,
    ControlVectorCodec,
    DiscreteSystem,
    DiscreteTransitionResult,
    PlantRuntimeState,
    PlantStateVectorCodec,
    PlantStepContext,
    StateLayout,
    TimeGrid,
)
from phydrax.linalg import ArraySpace, MaterializationPolicy
from phydrax.linalg.svd import DenseSVD, SVDSolvePolicy


def _link_xml(index: int, /) -> str:
    mass = 1.0 + 0.05 * index
    inertia = 0.02 + 0.001 * index
    return f"""
  <link name="link_{index:02d}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.8f}"/>
      <inertia ixx="{inertia:.8f}" ixy="0" ixz="0"
               iyy="{inertia:.8f}" iyz="0" izz="{inertia:.8f}"/>
    </inertial>
  </link>"""


def _joint_xml(index: int, /) -> str:
    kind = "revolute" if index % 2 == 0 else "prismatic"
    axis = "0 0 1" if kind == "revolute" else "1 0 0"
    lower, upper = ("-2.5", "2.5") if kind == "revolute" else ("-0.2", "0.3")
    return f"""
  <joint name="joint_{index:02d}" type="{kind}">
    <parent link="link_{index:02d}"/>
    <child link="link_{index + 1:02d}"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit lower="{lower}" upper="{upper}" effort="50" velocity="5"/>
    <dynamics damping="0.01"/>
  </joint>"""


def _chain_urdf(joint_count: int, /) -> str:
    links = "".join(_link_xml(index) for index in range(joint_count + 1))
    joints = "".join(_joint_xml(index) for index in range(joint_count))
    return f'<robot name="benchmark-chain-{joint_count}">{links}{joints}\n</robot>'


def _prepare_chain(joint_count: int, /) -> Any:
    adaptation = parse_urdf_text(_chain_urdf(joint_count), root_policy="fixed_world")
    articulation = adaptation.model.prepare().articulation
    return adaptation, articulation


def _synchronize(value: Any, /) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(
    operation: Callable[[], Any],
    /,
    *,
    repeats: int,
) -> tuple[Any, dict[str, float]]:
    warmup = operation()
    _synchronize(warmup)
    samples = []
    result = warmup
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        _synchronize(result)
        samples.append((time.perf_counter_ns() - started) * 1.0e-6)
    values = np.asarray(samples, dtype="float64")
    return result, {
        "maximum_ms": round(float(np.max(values)), 6),
        "median_ms": round(float(np.median(values)), 6),
        "minimum_ms": round(float(np.min(values)), 6),
    }


def _scalar_bool(value: Any, /) -> bool:
    return bool(np.asarray(value))


def _scalar_float(value: Any, /) -> float:
    return float(np.asarray(value))


def _rollout_workload(
    articulation: Any,
    configuration: Any,
    /,
    *,
    batch_size: int,
    step_count: int,
) -> Any:
    if articulation.state_layout is None or articulation.input_layout is None:
        raise ValueError("rollout benchmarking requires at least one moving joint")
    dtype = articulation.reference_position.dtype
    gravity = jnp.zeros((3,), dtype=dtype)
    policy = ReducedSemiImplicitVelocityEulerStepPolicy(maximum_step_size=0.01)

    def transition(
        context: Any, packed_state: Any, generalized_effort: Any, args: Any
    ) -> Any:
        tree, world_gravity = args
        source = tree.unpack_state(packed_state)
        result = reduced_semi_implicit_velocity_euler_step(
            tree,
            source,
            generalized_effort,
            world_gravity,
            context.duration,
            policy=policy,
        )
        return DiscreteTransitionResult(
            tree.pack_state(result.candidate_state),
            tree.pack_state(result.accepted_state),
            result.successful,
            result.status,
        )

    system = DiscreteSystem(
        transition,
        state_layout=articulation.state_layout,
        input_layout=articulation.input_layout,
        system_id=f"benchmark:robotics-articulation:{articulation.prepared_id}",
    )
    dynamics = DiscreteControlDynamics(
        system,
        method_id="benchmark:status-aware-semi-implicit-velocity-euler-rollout",
    )
    step_size = 1.0e-3
    grid = TimeGrid(
        jnp.arange(step_count + 1, dtype=dtype) * step_size,
        time_id=f"benchmark:robotics-rollout:{step_count}",
    )
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        articulation.input_layout.shape,
        parameterization_id="benchmark:zero-generalized-effort",
    )
    source = articulation.pack_state(
        configuration,
        jnp.zeros((articulation.nv,), dtype=dtype),
    )
    initial_states = jnp.broadcast_to(
        source,
        (batch_size, articulation.state_size),
    )
    controls = jnp.zeros(
        (batch_size, step_count, articulation.nv),
        dtype=dtype,
    )

    problem_id = f"benchmark:robotics-rollout:{articulation.prepared_id}"

    def rollout(
        dynamics_: DiscreteControlDynamics,
        grid_: TimeGrid,
        parameterization_: PiecewiseConstantControlParameterization,
        parameters: tuple[PreparedReducedArticulation, jax.Array],
        initial: jax.Array,
        coefficients: jax.Array,
    ) -> Any:
        return dynamics_.rollout(
            grid_,
            initial,
            parameterization_,
            coefficients,
            args=parameters,
            problem_id=problem_id,
        )

    arguments = (
        dynamics,
        grid,
        parameterization,
        (articulation, gravity),
        initial_states,
        controls,
    )
    return eqx.filter_jit(rollout), arguments


def _run_case(
    joint_count: int,
    /,
    *,
    repeats: int,
    rollout_batch_size: int,
    rollout_step_count: int,
) -> dict[str, Any]:
    (adaptation, articulation), preparation_timing = _measure(
        lambda: _prepare_chain(joint_count),
        repeats=repeats,
    )
    dtype = articulation.reference_position.dtype
    configuration = jnp.linspace(0.025, 0.15, articulation.nq, dtype=dtype)
    velocity = jnp.linspace(-0.08, 0.12, articulation.nv, dtype=dtype)
    target_acceleration = jnp.linspace(0.03, -0.04, articulation.nv, dtype=dtype)
    gravity = jnp.asarray([0.0, -9.81, 0.0], dtype=dtype)
    tip_body_id = int(np.asarray(articulation.body_ids[-1]))
    frame_load = jnp.linspace(0.1, 0.6, 6, dtype=dtype)

    compiled_fk = jax.jit(PreparedReducedArticulation.forward_kinematics)

    def jacobian_actions(
        tree: PreparedReducedArticulation,
        point: jax.Array,
        rate: jax.Array,
        load: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        operator = tree.frame_jacobian_operator(point, tip_body_id)
        return operator.mv(rate), operator.transpose_mv(load)

    compiled_jacobian = jax.jit(jacobian_actions)
    compiled_inverse = jax.jit(reduced_inverse_dynamics)
    compiled_forward = jax.jit(reduced_forward_dynamics)

    kinematics, fk_timing = _measure(
        lambda: compiled_fk(articulation, configuration, velocity),
        repeats=repeats,
    )
    (frame_velocity, generalized_load), jacobian_timing = _measure(
        lambda: compiled_jacobian(articulation, configuration, velocity, frame_load),
        repeats=repeats,
    )
    inverse, inverse_timing = _measure(
        lambda: compiled_inverse(
            articulation, configuration, velocity, target_acceleration, gravity
        ),
        repeats=repeats,
    )
    forward, forward_timing = _measure(
        lambda: compiled_forward(
            articulation,
            configuration,
            velocity,
            inverse.generalized_effort,
            gravity,
        ),
        repeats=repeats,
    )

    compiled_rollout, rollout_arguments = _rollout_workload(
        articulation,
        configuration,
        batch_size=rollout_batch_size,
        step_count=rollout_step_count,
    )
    trajectory, rollout_timing = _measure(
        lambda: compiled_rollout(*rollout_arguments),
        repeats=repeats,
    )

    contact_power = jnp.vdot(frame_velocity, frame_load).real
    generalized_power = jnp.vdot(velocity, generalized_load).real
    jacobian_power_residual = jnp.abs(contact_power - generalized_power)
    jacobian_power_scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(contact_power), jnp.abs(generalized_power)),
    )
    jacobian_relative_residual = jacobian_power_residual / jacobian_power_scale
    acceleration_error = jnp.max(
        jnp.abs(forward.acceleration - target_acceleration),
        initial=0.0,
    )
    rollout_valid = jnp.all(trajectory.valid)
    rollout_successful = jnp.all(trajectory.status == 0)
    finite_timings = all(
        math.isfinite(value)
        for timing in (
            preparation_timing,
            fk_timing,
            jacobian_timing,
            inverse_timing,
            forward_timing,
            rollout_timing,
        )
        for value in timing.values()
    )
    passed = (
        adaptation.report.negotiation.valid
        and not adaptation.report.losses
        and _scalar_bool(kinematics.successful)
        and _scalar_float(jacobian_relative_residual) <= 1.0e-6
        and _scalar_bool(inverse.successful)
        and _scalar_bool(forward.successful)
        and _scalar_float(acceleration_error) <= 5.0e-5
        and _scalar_bool(rollout_valid)
        and _scalar_bool(rollout_successful)
        and finite_timings
    )

    return {
        "evidence": {
            "forward_dynamics_status": int(np.asarray(forward.status)),
            "forward_dynamics_successful": _scalar_bool(forward.successful),
            "forward_inverse_relative_residual": _scalar_float(
                forward.relative_inverse_forward_residual
            ),
            "forward_kinematics_successful": _scalar_bool(kinematics.successful),
            "inverse_dynamics_decomposition_residual": _scalar_float(
                inverse.decomposition_residual
            ),
            "inverse_dynamics_status": int(np.asarray(inverse.status)),
            "inverse_dynamics_successful": _scalar_bool(inverse.successful),
            "jacobian_relative_power_residual": _scalar_float(jacobian_relative_residual),
            "maximum_acceleration_reconstruction_error": _scalar_float(
                acceleration_error
            ),
            "minimum_articulated_inertia": _scalar_float(
                forward.minimum_articulated_inertia
            ),
            "rollout_all_nodes_valid": _scalar_bool(rollout_valid),
            "rollout_backend_status": np.asarray(trajectory.backend_status).tolist(),
            "rollout_status": np.asarray(trajectory.status).tolist(),
            "urdf_loss_count": len(adaptation.report.losses),
            "urdf_negotiation_valid": adaptation.report.negotiation.valid,
        },
        "passed": passed,
        "size": {
            "body_count": joint_count + 1,
            "generalized_coordinates": articulation.nq,
            "generalized_velocities": articulation.nv,
            "joint_count": joint_count,
        },
        "timings_ms": {
            "forward_dynamics": forward_timing,
            "forward_kinematics": fk_timing,
            "frame_jacobian_actions": jacobian_timing,
            "inverse_dynamics": inverse_timing,
            "preparation": preparation_timing,
            "status_aware_batched_rollout": rollout_timing,
        },
        "work": {
            "rollout_batch_size": rollout_batch_size,
            "rollout_step_count": rollout_step_count,
            "rollout_transition_count": rollout_batch_size * rollout_step_count,
            "timed_calls_per_operation": repeats,
            "warmup_calls_per_operation": 1,
        },
    }


def run_benchmarks(*, smoke: bool = False) -> dict[str, Any]:
    joint_counts = (2,) if smoke else (2, 4, 8)
    repeats = 1 if smoke else 5
    rollout_batch_size = 2 if smoke else 32
    rollout_step_count = 2 if smoke else 32
    results = [
        _run_case(
            joint_count,
            repeats=repeats,
            rollout_batch_size=rollout_batch_size,
            rollout_step_count=rollout_step_count,
        )
        for joint_count in joint_counts
    ]
    return {
        "benchmark": "native-fixed-base-hinge-prismatic-articulation",
        "execution": "jax-jit-with-one-untimed-warmup",
        "passed": all(result["passed"] for result in results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Phase-separated campaign
# ---------------------------------------------------------------------------

type _Campaign = Literal["quick", "scaling"]
type _Section = Literal["all", "mass", "native"]
type _Topology = Literal["chain", "branched"]
type _Precision = Literal["float32", "float64"]
type _Scene = Literal[
    "sphere-chain", "sphere-branched", "capsule-bar", "capsule-bar-stop"
]
type _SearchBackend = Literal["exhaustive", "lbvh"]

_PROVIDER_MEMORY: dict[str, str] = {
    "status": "not-measured",
    "reason": (
        "No allocator or provider-external memory probe is attached. Compiler "
        "buffer sizes are XLA estimates, not vendor allocations, and logical "
        "bytes count unique array payloads only."
    ),
}
_CONTACT_LAW: dict[str, str] = {
    "declared": "position-level-inelastic-coulomb-hard-contact-coupled-stops",
    "normal_law": "position-level gap/h normal law with native hard inelastic contact",
    "equivalence_claim": (
        "none: native Phydrax hard contact is not a MuJoCo or other provider "
        "contact-law equivalent"
    ),
}
_STEP_OWNERSHIP: dict[str, Any] = {
    "ownership": "inseparable",
    "stages": [
        "route search through canonical articulated kinematics",
        "narrow phase",
        "coupled root and route epochs",
        "admissible-path certification",
        "operator cone response",
        "manifold and stop history commit",
    ],
    "reason": (
        "These stages execute inside one compiled PreparedArticulatedContactStep "
        "body and no public operation isolates their device time, so no per-stage "
        "value is reported. Timed public components (static admissibility, "
        "forward kinematics, mass binding, single-case contact step) are separate "
        "operations, not an additive decomposition of the plant step."
    ),
}
_STEP_SECONDS = 0.005
_RADIUS = 0.05
_GAP_TOLERANCE = 1.0e-9
_WORK_TOLERANCE = 1.0e-8
_ACTIVATION_DISTANCE = 0.02
_SEED = 20261022
_JOINT_AXES = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
_MASS_TOLERANCES: dict[str, dict[str, float]] = {
    "float64": {
        "action_agreement": 1.0e-10,
        "mass_action_recovery": 1.0e-8,
        "jvp_vjp_duality": 1.0e-9,
        "finite_difference": 1.0e-5,
        "inverse_reconstruction": 1.0e-6,
        "dense_reference": 1.0e-8,
        "baseline_agreement": 1.0e-8,
    },
    "float32": {
        "action_agreement": 1.0e-4,
        "mass_action_recovery": 1.0e-3,
        "jvp_vjp_duality": 1.0e-3,
        "finite_difference": 5.0e-2,
        "inverse_reconstruction": 1.0e-5,
        "dense_reference": 1.0e-3,
        "baseline_agreement": 1.0e-3,
    },
}
_NATIVE_DUALITY_TOLERANCE = 1.0e-8
_REGULARITY_MAX_ENTRIES = 65_536
_ROLLOUT_RETAINED_LIMIT = 8 * 1024**3


@dataclass(frozen=True, slots=True)
class MassRow:
    """One prepared-mass/smooth-dynamics diagnostic row."""

    topology: _Topology
    joint_count: int
    precision: _Precision
    rhs_count: int

    @property
    def row_id(self) -> str:
        return (
            f"mass-{self.topology}-j{self.joint_count}-{self.precision}"
            f"-rhs{self.rhs_count}"
        )


@dataclass(frozen=True, slots=True)
class NativeRow:
    """One native complete-plant diagnostic row (float64 native model)."""

    scene: _Scene
    joint_count: int
    capacity_factor: int
    initial_gap: float
    batch_size: int
    horizon: int
    search_backend: _SearchBackend = "exhaustive"

    @property
    def row_id(self) -> str:
        return (
            f"native-{self.scene}-j{self.joint_count}-cap{self.capacity_factor}"
            f"-gap{self.initial_gap:g}-b{self.batch_size}-h{self.horizon}"
            f"-{self.search_backend}"
        )


@dataclass(frozen=True, slots=True)
class CampaignSettings:
    repeats: int
    warmup: int
    dense_limit: int
    baseline: ModuleType | None
    native_details: bool = True


class _Stage:
    """Mutable label of the phase currently executing in one row."""

    __slots__ = ("name",)

    def __init__(self) -> None:
        self.name = "row-setup"


_QUICK_MASS_ROWS: tuple[MassRow, ...] = (
    MassRow("chain", 4, "float64", 8),
    MassRow("branched", 7, "float64", 8),
    MassRow("chain", 4, "float32", 8),
)
_SCALING_MASS_ROWS: tuple[MassRow, ...] = (
    MassRow("chain", 4, "float64", 8),
    MassRow("chain", 16, "float64", 8),
    MassRow("chain", 64, "float64", 8),
    MassRow("chain", 256, "float64", 8),
    MassRow("branched", 15, "float64", 8),
    MassRow("branched", 63, "float64", 8),
    MassRow("branched", 255, "float64", 8),
    MassRow("chain", 64, "float64", 1),
    MassRow("chain", 64, "float64", 64),
    MassRow("chain", 64, "float32", 8),
    MassRow("branched", 63, "float32", 8),
)
_QUICK_NATIVE_ROWS: tuple[NativeRow, ...] = (
    NativeRow("sphere-chain", 2, 1, 1.0e-4, 1, 4),
    NativeRow("sphere-branched", 3, 1, 1.0e-4, 2, 4),
    NativeRow("capsule-bar", 1, 1, 1.0e-4, 1, 4),
    NativeRow("capsule-bar-stop", 1, 1, 1.0e-4, 1, 4),
    NativeRow("sphere-branched", 3, 2, 0.5, 2, 4),
)
_SCALING_NATIVE_ROWS: tuple[NativeRow, ...] = (
    NativeRow("sphere-chain", 2, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-chain", 4, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-chain", 8, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-chain", 16, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-branched", 3, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-branched", 5, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-branched", 9, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-branched", 17, 1, 1.0e-4, 1, 16),
    NativeRow("sphere-branched", 9, 3, 1.0e-4, 1, 16),
    NativeRow("sphere-branched", 9, 1, 0.5, 1, 16),
    NativeRow("sphere-branched", 5, 1, 1.0e-4, 16, 16),
    NativeRow("sphere-branched", 5, 1, 1.0e-4, 64, 16),
    NativeRow("sphere-branched", 5, 1, 1.0e-4, 8, 8),
    NativeRow("sphere-branched", 5, 1, 1.0e-4, 8, 64),
    # Fixed mechanics isolate the case-capacity and recurrence-length axes.
    NativeRow("capsule-bar", 1, 1, 1.0e-4, 1, 32),
    NativeRow("capsule-bar", 1, 1, 1.0e-4, 8, 32),
    NativeRow("capsule-bar", 1, 1, 1.0e-4, 64, 32),
    NativeRow("capsule-bar", 1, 1, 1.0e-4, 8, 8),
    NativeRow("capsule-bar", 1, 1, 1.0e-4, 8, 64),
    NativeRow("capsule-bar-stop", 1, 1, 1.0e-4, 8, 32),
    NativeRow("sphere-chain", 8, 1, 1.0e-4, 1, 16, "lbvh"),
)


# ------------------------------------------------------------ JSON evidence


def _real(value: Any, /) -> float | str:
    """Finite floats stay numeric; non-finite values are named, never dropped."""

    number = float(np.asarray(value))
    if math.isfinite(number):
        return number
    if math.isnan(number):
        return "nan"
    return "inf" if number > 0.0 else "-inf"


def _reduce(values: Any, reducer: Callable[[np.ndarray], Any], /) -> float | str | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return None if array.size == 0 else _real(reducer(array))


def _count(values: Any, /) -> int:
    return int(np.sum(np.asarray(values).astype(np.int64)))


def _enum_namer(kind: type[IntEnum], /) -> Callable[[int], str]:
    def name(value: int, /) -> str:
        try:
            return kind(value).name
        except ValueError:
            return f"code_{value}"

    return name


def _code(value: int, /) -> str:
    return str(value)


def _plant_status_name(value: int, /) -> str:
    if value == int(ArticulatedPlantStatus.REVISION_MISMATCH):
        return ArticulatedPlantStatus.REVISION_MISMATCH.name
    return _enum_namer(ArticulatedContactStepStatus)(value)


def _histogram(values: Any, name: Callable[[int], str], /) -> dict[str, int]:
    flat = np.asarray(values).reshape(-1)
    unique, counts = np.unique(flat, return_counts=True)
    return {
        name(int(value)): int(count)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def _relative_difference(actual: Any, expected: Any, /) -> float | str:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    scale = max(float(np.max(np.abs(right), initial=0.0)), np.finfo(np.float64).tiny)
    return _real(np.max(np.abs(left - right), initial=0.0) / scale)


def _within(value: float | str | None, tolerance: float, /) -> bool:
    return isinstance(value, float) and value <= tolerance


def _warm_median(phase: Mapping[str, Any], /) -> float | None:
    warm = phase.get("warm")
    if not isinstance(warm, Mapping):
        return None
    median = warm.get("median_seconds")
    return median if isinstance(median, float) and median > 0.0 else None


def _ratio(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
    matched: bool,
    reason: str,
    /,
) -> dict[str, Any]:
    """A warm-median time ratio only for successful, numerically matched work."""

    top = _warm_median(numerator)
    bottom = _warm_median(denominator)
    if not matched:
        return {"value": None, "withheld_reason": reason}
    if top is None or bottom is None:
        return {"value": None, "withheld_reason": "a warm median was unavailable"}
    return {"value": top / bottom, "withheld_reason": None}


# -------------------------------------------------------- measured phases


def _cost_mapping(value: Any, /) -> Mapping[str, float] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Mapping):
        return value[0]
    return None


def _aot_phase(
    name: str,
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    settings: CampaignSettings,
    stage: _Stage,
    /,
) -> tuple[Any, dict[str, Any], Callable[..., Any]]:
    """Cold in-process wrap, lower, compile, first and warm execution of one body.

    Every numerical model leaf and argument is a dynamic executable argument;
    nothing numerical is closed over as a compile-time constant.
    """

    jax.clear_caches()
    stage.name = f"{name}:wrap"
    jitted, wrap_seconds = measure_host(lambda: eqx.filter_jit(function))
    stage.name = f"{name}:lower-compile"
    executable, compilation = measure_lower_and_compile(
        lambda: jitted.lower(*arguments),  # ty: ignore[unresolved-attribute]
        lambda lowered: lowered.compile(),
    )
    stage.name = f"{name}:first-execution"
    value, first_seconds = measure_synchronized(lambda: executable(*arguments))
    stage.name = f"{name}:warm-execution"
    _, warm = measure_repeated(
        lambda: executable(*arguments),
        warmup=settings.warmup,
        repeats=settings.repeats,
    )
    stage.name = f"{name}:compiler-evidence"
    evidence = compiler_evidence(
        _cost_mapping(executable.compiled.cost_analysis()),
        executable.compiled.memory_analysis(),
        source="jax-aot-lowered-compiled-executable",
        unavailable_reason="The active JAX backend did not report every compiler estimate.",
    )
    record = {
        "cache_state": "cold in-process (jax.clear_caches() immediately before wrap)",
        "wrap_seconds": wrap_seconds,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_seconds": first_seconds,
        "cold_end_to_end_seconds": (
            wrap_seconds
            + compilation.lowering_seconds
            + compilation.compilation_seconds
            + first_seconds
        ),
        "cold_end_to_end_scope": "sum of the measured wrap, lowering, compilation and first synchronized execution",
        "warm": warm.to_seconds_dict(),
        "warmup_calls": settings.warmup,
        "compiler": asdict(evidence),
        "compiler_estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "logical_argument_bytes": logical_array_bytes(arguments),
        "logical_result_bytes": logical_array_bytes(value),
        "provider_memory": _PROVIDER_MEMORY,
    }
    return value, record, executable


def _canonical_phase(
    name: str,
    operation: Callable[[], Any],
    settings: CampaignSettings,
    stage: _Stage,
    /,
) -> tuple[Any, dict[str, Any]]:
    """Canonical public entrypoint: first call on a cold in-process cache, then warm."""

    jax.clear_caches()
    stage.name = f"{name}:canonical-first"
    value, first_seconds = measure_synchronized(operation)
    stage.name = f"{name}:canonical-warm"
    _, warm = measure_repeated(
        operation, warmup=settings.warmup, repeats=settings.repeats
    )
    return value, {
        "cache_state": "cold in-process (jax.clear_caches() immediately before first call)",
        "first_call_seconds": first_seconds,
        "first_call_scope": "host validation, canonical trace, lowering, compilation and synchronized execution; not separable at this entrypoint",
        "warm": warm.to_seconds_dict(),
        "warmup_calls": settings.warmup,
        "provider_memory": _PROVIDER_MEMORY,
    }


def _transfer_record(tree: Any, stage: _Stage, /) -> dict[str, Any]:
    """Synchronized device placement of host copies of every prepared array leaf."""

    stage.name = "explicit-transfer"
    host = tuple(
        np.asarray(leaf)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, (jax.Array, np.ndarray))
        and not jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
    )
    _, seconds = measure_synchronized(lambda: jax.device_put(host))
    return {
        "seconds": seconds,
        "copied_leaf_count": len(host),
        "copied_bytes": sum(int(leaf.nbytes) for leaf in host),
        "scope": (
            "jax.device_put of host copies of every prepared array leaf; transfers "
            "performed inside plan construction and preparation remain inside "
            "those phases"
        ),
    }


def _refusal_status(error: Exception, /) -> str | None:
    if isinstance(error, (TypeError, ValueError, eqx.EquinoxRuntimeError)):
        return "refused"
    if isinstance(error, JaxRuntimeError) and "RESOURCE_EXHAUSTED" in str(error):
        return "resource-refused"
    return None


def _refusal(error: Exception, status: str, stage: _Stage, /) -> dict[str, Any]:
    return {
        "status": status,
        "stage": stage.name,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _section(
    stage: _Stage, build: Callable[[], dict[str, Any]] | None, /
) -> dict[str, Any]:
    """Run one optional evidence section; a refusal is recorded, never hidden."""

    if build is None:
        return {"status": "not-requested"}
    try:
        return build()
    except Exception as error:
        status = _refusal_status(error)
        if status is None:
            raise
        return _refusal(error, status, stage)


# ------------------------------------------------------ prepared-mass rows


@dataclass(frozen=True, slots=True)
class _TreeArrays:
    """Host topology of one fixed-base tree with large stable IDs."""

    body_ids: np.ndarray
    parent_indices: np.ndarray
    masses: np.ndarray
    inertia: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray
    joint_ids: np.ndarray
    joint_kinds: tuple[str, ...]
    anchors: np.ndarray
    axes: np.ndarray
    dtype: np.dtype

    @property
    def root_id(self) -> int:
        return int(self.body_ids[0])


def _body_inertia(index: int, /) -> np.ndarray:
    a = 0.02 + 0.001 * (index % 5)
    b = 0.025 + 0.001 * (index % 3)
    c = 0.03
    d = 0.002
    return np.asarray([[a, d, 0.0], [d, b, 0.5 * d], [0.0, 0.5 * d, c]])


def _tree_arrays(
    topology: _Topology, joint_count: int, dtype: np.dtype, /
) -> _TreeArrays:
    """Chain or binary tree; every fifth joint is fixed, others hinge/prismatic."""

    if joint_count < 1:
        raise ValueError("Mass rows require at least one joint.")
    count = joint_count + 1
    parents = np.full((count,), -1, dtype=np.int64)
    positions = np.zeros((count, 3), dtype=np.float64)
    for index in range(1, count):
        parent = index - 1 if topology == "chain" else (index - 1) // 2
        parents[index] = parent
        if topology == "chain":
            offset = np.asarray([0.2, 0.03 * math.sin(index), 0.01])
        else:
            depth = int(math.floor(math.log2(index + 1)))
            side = 1.0 if index % 2 else -1.0
            offset = np.asarray([0.25, side * 0.4 / (depth + 1), 0.05])
        positions[index] = positions[parent] + offset
    children = np.arange(1, count)
    kinds = tuple(
        "fixed" if joint % 5 == 4 else ("hinge" if joint % 2 == 0 else "prismatic")
        for joint in range(joint_count)
    )
    return _TreeArrays(
        body_ids=100_003 + 7_919 * np.arange(count, dtype=np.int64),
        parent_indices=parents,
        masses=np.asarray(
            [5.0] + [1.0 + 0.05 * (index % 7) for index in range(1, count)]
        ).astype(dtype),
        inertia=np.stack([_body_inertia(index) for index in range(count)]).astype(dtype),
        positions=positions.astype(dtype),
        orientations=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (count, 1)).astype(dtype),
        joint_ids=50_000 + 3 * np.arange(joint_count, dtype=np.int64),
        joint_kinds=kinds,
        anchors=(0.5 * (positions[parents[children]] + positions[children])).astype(
            dtype
        ),
        axes=_JOINT_AXES[np.arange(joint_count) % _JOINT_AXES.shape[0]].astype(dtype),
        dtype=dtype,
    )


def _tree_plans(
    arrays: _TreeArrays, /
) -> tuple[
    ParticleSetPlan, RigidBodySetPlan, RigidJointGraphPlan, ReducedArticulationPlan
]:
    count = arrays.body_ids.shape[0]
    children = np.arange(1, count)
    parent_ids = arrays.body_ids[arrays.parent_indices[children]]
    child_ids = arrays.body_ids[children]
    kinds = np.asarray(arrays.joint_kinds)
    fixed = np.flatnonzero(kinds == "fixed")
    hinge = np.flatnonzero(kinds == "hinge")
    prismatic = np.flatnonzero(kinds == "prismatic")
    particles = ParticleSetPlan(
        arrays.body_ids,
        arrays.masses,
        ambient_dimension=3,
        name="benchmark-tree-links",
        coordinate_dtype=arrays.dtype.name,
    )
    bodies = RigidBodySetPlan(
        np.zeros((count,), dtype=np.int64),
        arrays.inertia,
        fixed_mask=arrays.body_ids == arrays.root_id,
        name="benchmark-tree-bodies",
    )
    graph = RigidJointGraphPlan(
        fixed=(
            None
            if fixed.size == 0
            else FixedJointSetPlan(
                arrays.joint_ids[fixed], parent_ids[fixed], child_ids[fixed]
            )
        ),
        hinge=(
            None
            if hinge.size == 0
            else HingeJointSetPlan(
                arrays.joint_ids[hinge],
                parent_ids[hinge],
                child_ids[hinge],
                arrays.anchors[hinge],
                arrays.axes[hinge],
            )
        ),
        prismatic=(
            None
            if prismatic.size == 0
            else PrismaticJointSetPlan(
                arrays.joint_ids[prismatic],
                parent_ids[prismatic],
                child_ids[prismatic],
                arrays.anchors[prismatic],
                arrays.axes[prismatic],
            )
        ),
    )
    articulation = ReducedArticulationPlan(
        arrays.root_id, arrays.joint_ids, parent_ids, child_ids
    )
    return particles, bodies, graph, articulation


def _prepare_tree(
    plans: tuple[
        ParticleSetPlan, RigidBodySetPlan, RigidJointGraphPlan, ReducedArticulationPlan
    ],
    arrays: _TreeArrays,
    /,
) -> PreparedReducedArticulation:
    particles_plan, bodies_plan, graph_plan, articulation_plan = plans
    particles = particles_plan.prepare()
    bodies = bodies_plan.prepare(particles)
    zeros = np.zeros_like(arrays.positions)
    reference = bodies.kinematics(arrays.positions, zeros, arrays.orientations, zeros)
    graph = graph_plan.prepare(bodies, reference)
    return articulation_plan.prepare(graph, reference)


def _tree_shape(parents: np.ndarray, /) -> dict[str, int]:
    depth = np.zeros(parents.shape, dtype=np.int64)
    for index in range(1, parents.size):
        depth[index] = depth[parents[index]] + 1
    children = np.bincount(parents[1:], minlength=parents.size)
    return {
        "maximum_depth": int(depth.max()),
        "maximum_branching": int(children.max()),
        "leaf_count": int(np.sum(children == 0)),
    }


def _baseline_comparison(
    settings: CampaignSettings,
    stage: _Stage,
    arguments: tuple[Any, ...],
    current: Any,
    current_phase: Mapping[str, Any],
    tolerance: float,
    residual_tolerance: float,
    /,
) -> dict[str, Any]:
    """Pinned old forward dynamics on the identical prepared articulation and inputs."""

    if settings.baseline is None:
        return {
            "status": "not-requested",
            "reason": "No --baseline-reference was supplied; no comparison is claimed.",
        }
    old_forward = settings.baseline.reduced_forward_dynamics

    def baseline_forward(
        articulation: Any, q: Any, v: Any, tau: Any, gravity: Any
    ) -> Any:
        return old_forward(
            articulation, q, v, tau, gravity, residual_tolerance=residual_tolerance
        )

    try:
        value, phase, _ = _aot_phase(
            "baseline-forward-dynamics", baseline_forward, arguments, settings, stage
        )
    except Exception as error:
        status = _refusal_status(error)
        if status is None:
            raise
        return _refusal(error, status, stage)
    agreement = _relative_difference(value.acceleration, current.acceleration)
    baseline_successful = bool(np.asarray(value.successful))
    current_successful = bool(np.asarray(current.successful))
    matched = baseline_successful and current_successful and _within(agreement, tolerance)
    return {
        "status": "measured",
        "scope": "same-model smooth reduced_forward_dynamics on identical prepared input data",
        "phase": phase,
        "baseline_successful": baseline_successful,
        "baseline_status": int(np.asarray(value.status)),
        "acceleration_relative_difference": agreement,
        "agreement_tolerance": tolerance,
        "inverse_reconstruction_tolerance": residual_tolerance,
        "matched_work": matched,
        "baseline_over_current_warm_median": _ratio(
            phase,
            current_phase,
            matched,
            "baseline or current forward dynamics failed or accelerations disagree",
        ),
    }


def _mass_record(
    row: MassRow, settings: CampaignSettings, stage: _Stage, /
) -> dict[str, Any]:
    precision = np.dtype(row.precision)
    tolerances = _MASS_TOLERANCES[row.precision]
    stage.name = "host-topology-arrays"
    arrays, host_seconds = measure_host(
        lambda: _tree_arrays(row.topology, row.joint_count, precision)
    )
    stage.name = "plan-construction"
    plans, plan_seconds = measure_synchronized(lambda: _tree_plans(arrays))
    stage.name = "preparation"
    articulation, preparation_seconds = measure_synchronized(
        lambda: _prepare_tree(plans, arrays)
    )
    transfer = _transfer_record(articulation, stage)

    stage.name = "inputs"
    dtype = np.dtype(articulation.reference_position.dtype)
    nv = articulation.nv
    generator = np.random.default_rng(_SEED + row.joint_count)
    coordinates = np.arange(1, nv + 1, dtype=np.float64)
    q = jnp.asarray(0.3 * np.sin(coordinates), dtype=dtype)
    velocity = jnp.asarray(0.2 * np.cos(coordinates), dtype=dtype)
    rows_host = generator.standard_normal((row.rhs_count, nv))
    efforts = jnp.asarray(rows_host, dtype=dtype)
    columns = jnp.asarray(np.ascontiguousarray(rows_host.T), dtype=dtype)
    effort = jnp.asarray(rows_host[0], dtype=dtype)
    pair = generator.standard_normal((2, nv))
    direction = jnp.asarray(generator.standard_normal(nv), dtype=dtype)
    cotangent = jnp.asarray(generator.standard_normal(nv), dtype=dtype)
    gravity = jnp.asarray([0.0, 0.0, -9.81], dtype=dtype)
    policy = ArticulatedMassPolicy(maximum_rhs=max(64, row.rhs_count))

    def bind(articulation_: Any, q_: Any) -> Any:
        return prepare_articulated_mass(articulation_, q_, policy=policy)

    def vector_action(mass_: Any, articulation_: Any, q_: Any, effort_: Any) -> Any:
        return mass_.apply_inverse(articulation_, q_, effort_)

    def multiple_action(mass_: Any, articulation_: Any, q_: Any, columns_: Any) -> Any:
        return mass_.apply_inverse(articulation_, q_, columns_, rhs_axis=1)

    def repeated_actions(mass_: Any, articulation_: Any, q_: Any, rows_: Any) -> Any:
        def one(value: Any) -> Any:
            return mass_.apply_inverse(articulation_, q_, value)

        return jax.lax.map(one, rows_)

    def refactored_actions(articulation_: Any, q_: Any, rows_: Any) -> Any:
        def one(value: Any) -> Any:
            return prepare_articulated_mass(
                articulation_, q_, policy=policy
            ).apply_inverse(articulation_, q_, value)

        return jax.lax.map(one, rows_)

    def mass_action(mass_: Any, articulation_: Any, q_: Any, velocity_: Any) -> Any:
        return mass_.apply_mass(articulation_, q_, velocity_)

    def primal(articulation_: Any, q_: Any, effort_: Any) -> Any:
        return (
            prepare_articulated_mass(articulation_, q_, policy=policy)
            .apply_inverse(articulation_, q_, effort_)
            .value
        )

    def configuration_jvp(
        articulation_: Any, q_: Any, effort_: Any, direction_: Any
    ) -> Any:
        def inverse(point: Any) -> Any:
            return primal(articulation_, point, effort_)

        return jax.jvp(inverse, (q_,), (direction_,))[1]

    def configuration_vjp(
        articulation_: Any, q_: Any, effort_: Any, cotangent_: Any
    ) -> Any:
        def inverse(point: Any) -> Any:
            return primal(articulation_, point, effort_)

        return jax.vjp(inverse, q_)[1](cotangent_)[0]

    def forward(articulation_: Any, q_: Any, v_: Any, tau_: Any, gravity_: Any) -> Any:
        return reduced_forward_dynamics(
            articulation_,
            q_,
            v_,
            tau_,
            gravity_,
            residual_tolerance=tolerances["inverse_reconstruction"],
        )

    def inverse_dynamics(
        articulation_: Any, q_: Any, v_: Any, acceleration_: Any, gravity_: Any
    ) -> Any:
        return reduced_inverse_dynamics(
            articulation_,
            q_,
            v_,
            acceleration_,
            gravity_,
            power_tolerance=tolerances["inverse_reconstruction"],
        )

    def dense(articulation_: Any, q_: Any, columns_: Any) -> Any:
        return jnp.linalg.solve(reduced_mass_matrix(articulation_, q_).matrix, columns_)

    phases: dict[str, Any] = {}
    mass, phases["mass_bind"], _ = _aot_phase(
        "mass-bind", bind, (articulation, q), settings, stage
    )
    vector, phases["reused_vector_action"], _ = _aot_phase(
        "reused-vector-action",
        vector_action,
        (mass, articulation, q, effort),
        settings,
        stage,
    )
    multiple, phases["reused_multiple_rhs_action"], _ = _aot_phase(
        "reused-multiple-rhs-action",
        multiple_action,
        (mass, articulation, q, columns),
        settings,
        stage,
    )
    repeated, phases["reused_repeated_vector_actions"], _ = _aot_phase(
        "reused-repeated-vector-actions",
        repeated_actions,
        (mass, articulation, q, efforts),
        settings,
        stage,
    )
    refactored, phases["refactor_per_vector_actions"], _ = _aot_phase(
        "refactor-per-vector-actions",
        refactored_actions,
        (articulation, q, efforts),
        settings,
        stage,
    )
    product, phases["independent_mass_action"], _ = _aot_phase(
        "independent-mass-action",
        mass_action,
        (mass, articulation, q, vector.value),
        settings,
        stage,
    )
    _, phases["bind_and_vector_action"], primal_executable = _aot_phase(
        "bind-and-vector-action", primal, (articulation, q, effort), settings, stage
    )
    tangent, phases["configuration_jvp"], _ = _aot_phase(
        "configuration-jvp",
        configuration_jvp,
        (articulation, q, effort, direction),
        settings,
        stage,
    )
    pullback, phases["configuration_vjp"], _ = _aot_phase(
        "configuration-vjp",
        configuration_vjp,
        (articulation, q, effort, cotangent),
        settings,
        stage,
    )
    forward_arguments = (articulation, q, velocity, effort, gravity)
    forward_result, phases["smooth_forward_dynamics"], _ = _aot_phase(
        "smooth-forward-dynamics", forward, forward_arguments, settings, stage
    )
    inverse_result, phases["smooth_inverse_dynamics"], _ = _aot_phase(
        "smooth-inverse-dynamics",
        inverse_dynamics,
        (articulation, q, velocity, forward_result.candidate_acceleration, gravity),
        settings,
        stage,
    )
    dense_difference: float | str | None = None
    if row.joint_count <= settings.dense_limit:
        dense_value, phases["independent_dense_reference"], _ = _aot_phase(
            "independent-dense-reference",
            dense,
            (articulation, q, columns),
            settings,
            stage,
        )
        dense_difference = _relative_difference(multiple.value, dense_value)
    else:
        phases["independent_dense_reference"] = {
            "status": "not-run",
            "reason": (
                f"joint_count {row.joint_count} exceeds --dense-limit "
                f"{settings.dense_limit}"
            ),
        }

    stage.name = "finite-difference"
    step = 1.0e-6 if dtype == np.float64 else 1.0e-3
    plus = primal_executable(articulation, q + step * direction, effort)
    minus = primal_executable(articulation, q - step * direction, effort)
    central = (np.asarray(plus, np.float64) - np.asarray(minus, np.float64)) / (
        2.0 * step
    )
    stage.name = "adjoint-evidence"
    adjoint = mass.adjoint_evidence(
        jnp.asarray(pair[0], dtype=dtype), jnp.asarray(pair[1], dtype=dtype)
    )
    left = float(
        np.vdot(np.asarray(cotangent, np.float64), np.asarray(tangent, np.float64))
    )
    right = float(
        np.vdot(np.asarray(pullback, np.float64), np.asarray(direction, np.float64))
    )
    duality = _real(
        abs(left - right) / max(abs(left), abs(right), np.finfo(np.float64).tiny)
    )
    finite_difference = _relative_difference(tangent, central)
    repeated_vs_multiple = _relative_difference(
        repeated.value, np.asarray(multiple.value).T
    )
    refactored_vs_reused = _relative_difference(refactored.value, repeated.value)
    recovery = _relative_difference(product.value, effort)
    reconstruction = _relative_difference(inverse_result.candidate_effort, effort)
    stage.name = "baseline"
    baseline = _baseline_comparison(
        settings,
        stage,
        forward_arguments,
        forward_result,
        phases["smooth_forward_dynamics"],
        tolerances["baseline_agreement"],
        tolerances["inverse_reconstruction"],
    )

    mass_status = _enum_namer(ArticulatedMassStatus)
    repeated_ok = bool(np.all(np.asarray(repeated.successful)))
    refactored_ok = bool(np.all(np.asarray(refactored.successful)))
    multiple_ok = bool(np.asarray(multiple.successful))
    checks: dict[str, bool | None] = {
        "mass_preparation_successful": bool(np.asarray(mass.successful)),
        "vector_action_successful": bool(np.asarray(vector.successful)),
        "multiple_rhs_action_successful": multiple_ok,
        "repeated_vector_actions_successful": repeated_ok,
        "refactored_actions_successful": refactored_ok,
        "repeated_matches_multiple_rhs": _within(
            repeated_vs_multiple, tolerances["action_agreement"]
        ),
        "refactored_matches_reused": _within(
            refactored_vs_reused, tolerances["action_agreement"]
        ),
        "independent_mass_action_recovers_effort": bool(np.asarray(product.successful))
        and _within(recovery, tolerances["mass_action_recovery"]),
        "adjoint_symmetry_valid": bool(np.asarray(adjoint.valid)),
        "jvp_vjp_duality": _within(duality, tolerances["jvp_vjp_duality"]),
        "jvp_central_difference": _within(
            finite_difference, tolerances["finite_difference"]
        ),
        "smooth_forward_dynamics_successful": bool(np.asarray(forward_result.successful)),
        "smooth_inverse_reconstructs_effort": bool(np.asarray(inverse_result.successful))
        and _within(reconstruction, tolerances["inverse_reconstruction"]),
        "independent_dense_reference": (
            None
            if dense_difference is None
            else _within(dense_difference, tolerances["dense_reference"])
        ),
    }
    failed = sorted(name for name, value in checks.items() if value is False)
    reuse_matched = (
        repeated_ok and refactored_ok and bool(checks["refactored_matches_reused"])
    )
    schedule_matched = (
        repeated_ok and multiple_ok and bool(checks["repeated_matches_multiple_rhs"])
    )
    kinds = list(arrays.joint_kinds)
    return {
        "kind": "prepared-mass",
        "row_id": row.row_id,
        "row": asdict(row),
        "status": "completed",
        "identity": {
            "articulation_prepared_id": articulation.prepared_id,
            "generalized_velocity_space_id": articulation.generalized_velocity_space.space_id,
            "generalized_effort_space_id": articulation.generalized_effort_space.space_id,
            "mass_id": mass.mass_id,
            "mass_policy": {
                "maximum_rhs": policy.maximum_rhs,
                "relative_pivot_tolerance": policy.resolved_pivot_tolerance(dtype),
                "residual_tolerance": policy.resolved_residual_tolerance(dtype),
            },
        },
        "size": {
            "topology": row.topology,
            "body_count": row.joint_count + 1,
            "joint_count": row.joint_count,
            "joint_kinds": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
            "generalized_velocities": nv,
            "rhs_count": row.rhs_count,
            "dtype": dtype.name,
            **_tree_shape(arrays.parent_indices),
        },
        "host_phases": {
            "topology_array_seconds": host_seconds,
            "plan_construction_seconds": plan_seconds,
            "plan_construction_scope": "public plan constructors, including their own host validation and array placement",
            "preparation_seconds": preparation_seconds,
            "preparation_scope": "synchronized particle, body, reference-kinematics, joint-graph and articulation preparation",
            "explicit_transfer": transfer,
        },
        "phases": phases,
        "factor_reuse": {
            "matched_work": reuse_matched,
            "refactor_over_reused_warm_median": _ratio(
                phases["refactor_per_vector_actions"],
                phases["reused_repeated_vector_actions"],
                reuse_matched,
                "refactored or reused actions failed or disagree",
            ),
            "repeated_vector_over_multiple_rhs_warm_median": _ratio(
                phases["reused_repeated_vector_actions"],
                phases["reused_multiple_rhs_action"],
                schedule_matched,
                "repeated-vector or multiple-RHS actions failed or disagree",
            ),
            "scope": f"{row.rhs_count} effort right-hand sides per call in one executable",
            "compiler_scope": (
                "The comparison spells preparation inside lax.map, but XLA may hoist "
                "loop-invariant factor work. Timings compare the actual executables; "
                "no factorization execution count is inferred from Python placement."
            ),
        },
        "evidence": {
            "mass_status": mass_status(int(np.asarray(mass.status))),
            "minimum_relative_pivot": _real(mass.evidence.minimum_relative_pivot),
            "relative_pivot_tolerance": _real(mass.evidence.relative_pivot_tolerance),
            "vector_status": mass_status(int(np.asarray(vector.status))),
            "vector_relative_residual": _real(vector.relative_residual),
            "multiple_rhs_status": mass_status(int(np.asarray(multiple.status))),
            "multiple_rhs_relative_residual_max": _reduce(
                multiple.relative_residual, np.max
            ),
            "repeated_relative_residual_max": _reduce(repeated.relative_residual, np.max),
            "refactored_relative_residual_max": _reduce(
                refactored.relative_residual, np.max
            ),
            "repeated_vs_multiple_relative_difference": repeated_vs_multiple,
            "refactored_vs_reused_relative_difference": refactored_vs_reused,
            "mass_action_effort_relative_difference": recovery,
            "adjoint_residual": _real(adjoint.residual),
            "adjoint_scale": _real(adjoint.scale),
            "jvp_vjp_duality_relative_residual": duality,
            "jvp_central_difference_relative_error": finite_difference,
            "central_difference_step": step,
            "inverse_reconstruction_input": "retained candidate acceleration, including on refusal",
            "smooth_forward_status": int(np.asarray(forward_result.status)),
            "smooth_forward_relative_inverse_residual": _real(
                forward_result.relative_inverse_forward_residual
            ),
            "smooth_inverse_effort_relative_difference": reconstruction,
            "smooth_inverse_status": int(np.asarray(inverse_result.status)),
            "smooth_inverse_decomposition_residual": _real(
                inverse_result.decomposition_residual
            ),
            "smooth_inverse_external_power_residual": _real(
                inverse_result.external_power_residual
            ),
            "dense_reference_relative_difference": dense_difference,
            "dense_reference_scope": "materialized JᵀGJ from reduced_mass_matrix and an LU solve; independent of the articulated pivots",
        },
        "baseline": baseline,
        "logical_retained_bytes": {
            "prepared_articulation": logical_array_bytes(articulation),
            "prepared_mass": logical_array_bytes(mass),
            "mass_factor": logical_array_bytes(mass.factor),
            "mass_binding": logical_array_bytes(mass.binding),
            "dense_matrix_equivalent": nv * nv * dtype.itemsize,
            "dense_matrix_equivalent_scope": "analytic nv*nv payload for comparison only; never allocated by the prepared mass",
        },
        "provider_memory": _PROVIDER_MEMORY,
        "scientific": {
            "status": "passed" if not failed else "unsuccessful",
            "checks": checks,
            "failed_checks": failed,
            "tolerances": tolerances,
        },
    }


# ------------------------------------------------------- native plant rows


@dataclass(frozen=True, slots=True)
class _NativeScene:
    model: NativeRobotModel
    lift_joint: str
    route_capacities: tuple[int, int, int, int, int, int]
    joint_kinds: tuple[str, ...]


def _materials() -> RobotContactMaterialTable:
    return RobotContactMaterialTable(
        np.asarray([[1, 1]], dtype=np.int64), np.asarray([0.8]), np.asarray([0.6])
    )


def _frame(body_id: int, /) -> RobotFramePlan:
    return RobotFramePlan(
        RobotNameIDMap(("tool",)),
        np.asarray([body_id], dtype=np.int64),
        np.zeros((1, 3)),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )


def _floor() -> dict[str, Any]:
    return {
        "plane_names": RobotNameIDMap(("floor",)),
        "plane_normals": np.asarray([[0.0, 0.0, 1.0]]),
        "plane_offsets": np.zeros((1,)),
        "plane_material_ids": np.ones((1,), dtype=np.int64),
    }


def _sphere_scene(row: NativeRow, /) -> _NativeScene:
    """Prismatic-z bodies carrying spheres just above the plane z = 0.

    ``sphere-chain`` stacks every slider on its predecessor; ``sphere-branched``
    hangs ``joint_count - 1`` sliders from one lifted carriage.
    """

    up = (0.0, 0.0, 1.0)
    if row.scene == "sphere-chain":
        if row.joint_count < 1:
            raise ValueError("sphere-chain rows require joint_count >= 1.")
        carriers = tuple(f"link_{index:02d}" for index in range(row.joint_count))
        names = ("base", *carriers)
        centers = [(0.0, 0.0, -1.0)] + [
            (0.3 * index, 0.0, _RADIUS) for index in range(row.joint_count)
        ]
        masses = [10.0] + [1.0] * row.joint_count
        parents = ("base", *carriers[:-1])
        joints = tuple(
            NativeTreeJoint(
                f"slide_{index:02d}",
                "prismatic",
                parents[index],
                carriers[index],
                anchor=centers[index + 1],
                axis=up,
            )
            for index in range(row.joint_count)
        )
        lift = "slide_00"
    else:
        tines = row.joint_count - 1
        if tines < 1:
            raise ValueError("sphere-branched rows require joint_count >= 2.")
        carriers = tuple(f"tine_{index:02d}" for index in range(tines))
        names = ("base", "carriage", *carriers)
        middle = 0.5 * (tines - 1)
        centers = [(0.0, 0.0, -1.0), (0.0, 0.0, 0.3)] + [
            (0.3 * (index - middle), 0.0, _RADIUS) for index in range(tines)
        ]
        masses = [10.0, 2.0] + [1.0] * tines
        joints = (
            NativeTreeJoint(
                "lift", "prismatic", "base", "carriage", anchor=centers[1], axis=up
            ),
        ) + tuple(
            NativeTreeJoint(
                f"tine_slide_{index:02d}",
                "prismatic",
                "carriage",
                carriers[index],
                anchor=centers[index + 2],
                axis=up,
            )
            for index in range(tines)
        )
        lift = "lift"
    bodies = RobotNameIDMap(names)
    carrier_ids = np.asarray(
        [bodies.id_for_name(name) for name in carriers], dtype=np.int64
    )
    count = carrier_ids.size
    excluded = np.asarray(
        [
            (int(carrier_ids[first]), int(carrier_ids[second]))
            for first in range(count)
            for second in range(first + 1, count)
        ],
        dtype=np.int64,
    ).reshape((-1, 2))
    collision = RobotCollisionGeometryPlan(
        RobotNameIDMap(tuple(f"sphere_{index:02d}" for index in range(count))),
        (ArticulatedPrimitiveKind.SPHERE,) * count,
        carrier_ids,
        np.zeros((count, 3)),
        np.zeros((count, 3)),
        np.full((count,), _RADIUS),
        np.ones((count,), dtype=np.int64),
        materials=_materials(),
        excluded_body_pairs=excluded,
        **_floor(),
    )
    model = lower_native_robot_tree(
        f"benchmark-{row.scene}-{row.joint_count}",
        bodies,
        np.asarray(masses),
        np.tile(0.01 * np.eye(3), (len(names), 1, 1)),
        np.asarray(centers),
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (len(names), 1)),
        "base",
        joints,
        DimensionalScaleContract.si(),
        _frame(int(carrier_ids[-1])),
        collision=collision,
    )
    return _NativeScene(
        model,
        lift,
        (0, 0, 0, 0, count * row.capacity_factor, 0),
        tuple(joint.kind for joint in joints),
    )


def _capsule_scene(row: NativeRow, /) -> _NativeScene:
    """Horizontal capsule on a vertical slider: one pair with two plane witnesses.

    ``capsule-bar-stop`` adds a lower joint stop at the contact configuration,
    coupling both witness rows with a stop row.
    """

    if row.joint_count != 1:
        raise ValueError("Capsule rows have exactly one joint.")
    bodies = RobotNameIDMap(("bar", "base"))
    bar = int(bodies.id_for_name("bar"))
    collision = RobotCollisionGeometryPlan(
        RobotNameIDMap(("rod",)),
        (ArticulatedPrimitiveKind.CAPSULE,),
        np.asarray([bar], dtype=np.int64),
        np.asarray([[-0.4, 0.0, 0.0]]),
        np.asarray([[0.4, 0.0, 0.0]]),
        np.asarray([_RADIUS]),
        np.ones((1,), dtype=np.int64),
        materials=_materials(),
        **_floor(),
    )
    stops = (
        ReducedJointStopPlan(
            np.asarray([0], dtype=np.int64), np.asarray([0.0]), np.asarray([1.0])
        )
        if row.scene == "capsule-bar-stop"
        else None
    )
    model = lower_native_robot_tree(
        f"benchmark-{row.scene}",
        bodies,
        np.asarray([2.0, 10.0]),
        np.stack((np.diag([0.01, 0.1, 0.1]), 0.1 * np.eye(3))),
        np.asarray([[0.0, 0.0, _RADIUS], [0.0, 0.0, -1.0]]),
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
        "base",
        (NativeTreeJoint("lift", "prismatic", "base", "bar", axis=(0.0, 0.0, 1.0)),),
        DimensionalScaleContract.si(),
        _frame(bar),
        collision=collision,
        stops=stops,
    )
    return _NativeScene(
        model, "lift", (0, 0, 0, 0, 0, 2 * row.capacity_factor), ("prismatic",)
    )


def _native_scene(row: NativeRow, /) -> _NativeScene:
    if row.scene in ("capsule-bar", "capsule-bar-stop"):
        return _capsule_scene(row)
    return _sphere_scene(row)


def _native_step_policy() -> ArticulatedContactStepPolicy:
    certificate = ArticulatedContactCertificatePolicy(
        contact_reference_velocity=1.0,
        hinge_reference_velocity=1.0,
        prismatic_reference_velocity=1.0,
        contact_impulse_tolerance=1.0e-9,
        hinge_impulse_tolerance=1.0e-9,
        prismatic_impulse_tolerance=1.0e-9,
        contact_gap_tolerance=_GAP_TOLERANCE,
        hinge_gap_tolerance=_GAP_TOLERANCE,
        prismatic_gap_tolerance=_GAP_TOLERANCE,
        slip_speed_tolerance=1.0e-9,
        complementarity_work_tolerance=_WORK_TOLERANCE,
        friction_work_tolerance=_WORK_TOLERANCE,
        contact_work_tolerance=_WORK_TOLERANCE,
        contact_energy_tolerance=_WORK_TOLERANCE,
        ledger_tolerance=1.0e-8,
        energy_surplus_tolerance=1.0e-8,
        momentum_energy_tolerance=1.0e-14,
        discretization_energy_budget=1.0,
    )
    return ArticulatedContactStepPolicy(
        certificate,
        AdmissiblePathPlan(_GAP_TOLERANCE, maximum_intervals=4096),
        cone_solver=ContactConeSolverPlan(
            maximum_iterations=2000,
            absolute_tolerance=1.0e-14,
            relative_tolerance=1.0e-13,
        ),
        smooth=ReducedSemiImplicitVelocityEulerStepPolicy(
            maximum_step_size=1.0,
            absolute_energy_tolerance=1.0,
            relative_energy_tolerance=1.0e-2,
        ),
        maximum_root_iterations=200,
        root_absolute_tolerance=1.0e-13,
        root_relative_tolerance=1.0e-12,
        maximum_subdivision_depth=8,
        maximum_segment_attempts=64,
        minimum_segment_duration=1.0e-6,
        derivative_regularity=SVDSolvePolicy(
            DenseSVD(algorithm="qr"),
            materialization=MaterializationPolicy(max_entries=_REGULARITY_MAX_ENTRIES),
        ),
    )


def _dof_index(prepared: PreparedNativeRobotModel, joint_name: str, /) -> int:
    names = prepared.model.joint_names
    identifier = names.ids[names.names.index(joint_name)]
    edges = [int(value) for value in np.asarray(prepared.articulation.joint_ids)]
    return int(prepared.articulation.joint_velocity_slices[edges.index(identifier)].start)


def _case(tree: Any, index: int, /) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _epoch_evidence(epoch: Any, /) -> dict[str, Any]:
    return {
        "complete_cases": _count(epoch.complete),
        "search_status": _histogram(epoch.status, _enum_namer(ContactSearchStatus)),
        "candidate_count_max": _reduce(epoch.candidate_count, np.max),
        "overflow_count_max": _reduce(epoch.overflow_count, np.max),
    }


def _path_evidence(path: Any, /) -> dict[str, Any]:
    return {
        "status": _histogram(path.status, _enum_namer(AdmissiblePathStatus)),
        "certified_prefix_min": _reduce(path.certified_prefix, np.min),
        "minimum_initial_gap_min": _reduce(path.minimum_initial_gap, np.min),
        "minimum_lower_gap_min": _reduce(path.minimum_lower_gap, np.min),
        "interval_count_max": _reduce(path.interval_count, np.max),
        "guarantee_successful_cases": _count(path.guarantee.successful),
        "finite_cases": _count(path.finite),
    }


def _rollback_evidence(source: PlantRuntimeState, result: Any, /) -> dict[str, Any]:
    """Failed cases must keep their complete accepted payload and clock."""

    failed = ~np.asarray(result.successful, dtype=np.bool_).reshape(-1)
    cases = failed.size
    unchanged = np.asarray(source.time).reshape(-1) == np.asarray(
        result.accepted_state.time
    ).reshape(-1)
    for before, after in zip(
        jax.tree.leaves(source.payload),
        jax.tree.leaves(result.accepted_state.payload),
        strict=True,
    ):
        unchanged &= np.all(
            np.asarray(before).reshape((cases, -1))
            == np.asarray(after).reshape((cases, -1)),
            axis=1,
        )
    return {
        "failed_cases": int(failed.sum()),
        "failed_cases_with_unchanged_payload_and_clock": int(np.sum(failed & unchanged)),
        "verified": bool(np.all(unchanged[failed])),
    }


def _step_evidence(
    result: Any,
    source: PlantRuntimeState,
    contact_capacity: int,
    stop_capacity: int,
    /,
) -> dict[str, Any]:
    step = result.evidence.step
    segment = step.last_segment
    root = segment.root
    ledger = step.ledger
    active_contacts = np.asarray(segment.active_contact_count).reshape(-1)
    active_stops = np.asarray(segment.active_stop_count).reshape(-1)
    return {
        "case_count": int(active_contacts.size),
        "successful_cases": _count(result.successful),
        "plant_status": _histogram(result.status, _plant_status_name),
        "contact_step_status": _histogram(step.status, _plant_status_name),
        "revision_valid_cases": _count(result.evidence.revision_valid),
        "covered_fraction_min": _reduce(step.covered_fraction, np.min),
        "covered_fraction_max": _reduce(step.covered_fraction, np.max),
        "accepted_segments_max": _reduce(step.accepted_segments, np.max),
        "attempted_segments_max": _reduce(step.attempted_segments, np.max),
        "root_iterations_max": _reduce(step.root_iterations, np.max),
        "route_rebuilds_max": _reduce(step.route_rebuilds, np.max),
        "request_budgets_successful_cases": _count(step.request_budgets_successful),
        "free_bypass_cases": _count(step.free_bypass),
        "fixed_branch_candidate_cases": _count(step.fixed_branch_candidate),
        "fixed_branch_candidate_scope": "primal discrete-branch precondition only; not derivative admission",
        "last_segment": {
            "root_attempted_cases": _count(root.attempted),
            "root_status_codes": _histogram(root.status, _code),
            "root_iterations_max": _reduce(root.iterations, np.max),
            "root_residual_norm_max": _reduce(root.residual_norm, np.max),
            "routes_stable_cases": _count(root.routes_stable),
            "search_successful_cases": _count(root.search_successful),
            "narrow_phase_successful_cases": _count(root.narrow_phase_successful),
            "warm_history_successful_cases": _count(root.warm_history_successful),
            "route_epoch": (
                None
                if segment.route_epoch is None
                else _epoch_evidence(segment.route_epoch)
            ),
            "coverage_epoch": (
                None
                if segment.coverage_epoch is None
                else _epoch_evidence(segment.coverage_epoch)
            ),
            "path": None if segment.path is None else _path_evidence(segment.path),
            "active_contacts_max": int(active_contacts.max(initial=0)),
            "active_contacts_total": int(active_contacts.sum()),
            "contact_capacity_per_case": contact_capacity,
            "active_stops_max": int(active_stops.max(initial=0)),
            "stop_capacity_per_case": stop_capacity,
            "activation_events": _count(segment.activation_event),
            "mass_status": _histogram(
                segment.mass_status, _enum_namer(ArticulatedMassStatus)
            ),
            "passed_cases": {
                name: _count(getattr(segment, name))
                for name in (
                    "momentum_successful",
                    "gaps_successful",
                    "cone_successful",
                    "contact_work_successful",
                    "energy_successful",
                    "history_successful",
                    "finite",
                )
            },
        },
        "energy_ledger_joules": {
            "initial_energy_min": _reduce(ledger.initial_energy, np.min),
            "final_energy_max": _reduce(ledger.final_energy, np.max),
            "source_work_total": _reduce(ledger.source_work, np.sum),
            "contact_work_total": _reduce(ledger.contact_work, np.sum),
            "friction_work_total": _reduce(ledger.friction_work, np.sum),
            "correction_loss_total": _reduce(ledger.correction_loss, np.sum),
            "identity_residual_max_abs": _reduce(
                np.abs(np.asarray(ledger.identity_residual)), np.max
            ),
            "positive_energy_surplus_max": _reduce(
                ledger.positive_energy_surplus, np.max
            ),
            "discretization_charge_max": _reduce(ledger.discretization_charge, np.max),
            "roundoff_allowance_max": _reduce(ledger.roundoff_allowance, np.max),
        },
        "rollback": _rollback_evidence(source, result),
    }


def _admissibility_evidence(admissibility: Any, /) -> dict[str, Any]:
    return {
        "successful": bool(np.asarray(admissibility.successful)),
        "status": _plant_status_name(int(np.asarray(admissibility.status))),
        "minimum_contact_gap": _reduce(admissibility.minimum_contact_gap, np.min),
        "minimum_stop_gap": _reduce(admissibility.minimum_stop_gap, np.min),
        "winding_valid": bool(np.asarray(admissibility.winding_valid)),
        "epoch": None
        if admissibility.epoch is None
        else _epoch_evidence(admissibility.epoch),
        "path": None
        if admissibility.path is None
        else _path_evidence(admissibility.path),
    }


def _single_step_evidence(result: Any, /) -> dict[str, Any]:
    return {
        "successful": bool(np.asarray(result.successful)),
        "status": _plant_status_name(int(np.asarray(result.status))),
        "covered_fraction": _real(result.covered_fraction),
        "accepted_segments": int(np.asarray(result.accepted_segments)),
        "attempted_segments": int(np.asarray(result.attempted_segments)),
        "free_bypass": bool(np.asarray(result.free_bypass)),
        "fixed_branch_candidate": bool(np.asarray(result.fixed_branch_candidate)),
        "active_contacts": int(np.asarray(result.last_segment.active_contact_count)),
        "active_stops": int(np.asarray(result.last_segment.active_stop_count)),
    }


def _native_derivative(
    contact_step: Any,
    result: Any,
    actuation: Any,
    gravity: Any,
    settings: CampaignSettings,
    stage: _Stage,
    phases: dict[str, Any],
    /,
) -> dict[str, Any]:
    """Fixed-branch JVP/VJP of one single-case step; refused branches claim nothing."""

    dtype = contact_step.dtype
    nv = contact_step.articulation.nv
    generator = np.random.default_rng(_SEED + 1)
    q_dot = jnp.asarray(generator.standard_normal(nv), dtype=dtype)
    v_dot = jnp.asarray(generator.standard_normal(nv), dtype=dtype)
    q_bar = jnp.asarray(generator.standard_normal(nv), dtype=dtype)
    v_bar = jnp.asarray(generator.standard_normal(nv), dtype=dtype)
    impulse_bar = jnp.asarray(
        generator.standard_normal(result.last_segment.impulse.shape), dtype=dtype
    )

    def step_jvp(
        step_: Any, result_: Any, actuation_: Any, gravity_: Any, q_dot_: Any, v_dot_: Any
    ) -> Any:
        return step_.step_jvp(
            result_,
            actuation_,
            gravity_,
            (None, q_dot_, v_dot_, jnp.zeros_like(actuation_), jnp.zeros_like(gravity_)),
        )

    def step_vjp(
        step_: Any,
        result_: Any,
        actuation_: Any,
        gravity_: Any,
        q_bar_: Any,
        v_bar_: Any,
        impulse_bar_: Any,
    ) -> Any:
        return step_.step_vjp(
            result_,
            actuation_,
            gravity_,
            (q_bar_, v_bar_, impulse_bar_),
            friction_cotangent=True,
        )

    tangent, phases["fixed_branch_step_jvp"], _ = _aot_phase(
        "fixed-branch-step-jvp",
        step_jvp,
        (contact_step, result, actuation, gravity, q_dot, v_dot),
        settings,
        stage,
    )
    cotangent, phases["fixed_branch_step_vjp"], _ = _aot_phase(
        "fixed-branch-step-vjp",
        step_vjp,
        (contact_step, result, actuation, gravity, q_bar, v_bar, impulse_bar),
        settings,
        stage,
    )
    candidate = bool(np.asarray(result.fixed_branch_candidate))
    jvp_ok = bool(np.asarray(tangent.successful))
    vjp_ok = bool(np.asarray(cotangent.successful))
    duality: float | str | None = None
    if jvp_ok and vjp_ok:
        left = sum(
            float(np.vdot(np.asarray(bar, np.float64), np.asarray(dot, np.float64)))
            for bar, dot in (
                (q_bar, tangent.configuration_tangent),
                (v_bar, tangent.velocity_tangent),
                (impulse_bar, tangent.impulse_tangent),
            )
        )
        right = sum(
            float(np.vdot(np.asarray(bar, np.float64), np.asarray(dot, np.float64)))
            for bar, dot in (
                (cotangent.configuration_cotangent, q_dot),
                (cotangent.velocity_cotangent, v_dot),
            )
        )
        duality = _real(
            abs(left - right) / max(abs(left), abs(right), np.finfo(np.float64).tiny)
        )
    if jvp_ok and vjp_ok:
        status = "admitted"
    elif jvp_ok != vjp_ok:
        status = "inconsistent-admission"
    elif candidate:
        status = "candidate-refused"
    else:
        status = "not-candidate"
    step_cotangent = cotangent.step_cotangent
    coupled = contact_step.articulation.nv + 3 * contact_step.block_count
    return {
        "status": status,
        "status_scope": (
            "admitted requires a successful JVP and VJP; fixed_branch_candidate is "
            "only the primal discrete-branch precondition and is never counted as "
            "admission"
        ),
        "fixed_branch_candidate": candidate,
        "jvp": _derivative_admission(tangent),
        "vjp": {
            **_derivative_admission(cotangent),
            "friction_cotangent": bool(cotangent.friction_cotangent),
            "friction_table_cotangents_formed": (
                step_cotangent.static_friction_table is not None
                and step_cotangent.dynamic_friction_table is not None
            ),
        },
        "probe_scope": (
            "configuration and velocity tangents with zero step, actuation and "
            "gravity tangents; duality pairs accepted configuration, velocity and "
            "block impulse cotangents"
        ),
        "duality_relative_residual": duality,
        "duality_tolerance": _NATIVE_DUALITY_TOLERANCE,
        "duality_withheld_reason": (
            None if duality is not None else "no derivative is admitted for this step"
        ),
        "regularity_policy": {
            "method": "native dense SVD (qr) via the coupled (v1, p) rank certificate",
            "materialization_max_entries": _REGULARITY_MAX_ENTRIES,
            "coupled_unknowns": coupled,
            "coupled_jacobian_entries": coupled**2,
            "coupled_jacobian_within_bound": coupled**2 <= _REGULARITY_MAX_ENTRIES,
            "scope": "unknowns are nv + 3 * (contact + stop blocks); an over-budget certificate refuses before materializing",
        },
    }


def _rank_certificate(certificate: Any, /) -> dict[str, Any] | None:
    """Native SVD rank evidence and its declared resources, or ``None`` if absent."""

    if certificate is None:
        return None
    cost = certificate.cost
    return {
        "route": certificate.route,
        "reason": certificate.reason,
        "rows": certificate.rows,
        "columns": certificate.columns,
        "rank": int(np.asarray(certificate.rank)),
        "fixed_rank": bool(np.asarray(certificate.fixed_rank)),
        "converged": bool(np.asarray(certificate.converged)),
        "svd_status": int(np.asarray(certificate.svd_status)),
        "smallest_retained_singular_value": _real(
            certificate.smallest_retained_singular_value
        ),
        "largest_discarded_singular_value": _real(
            certificate.largest_discarded_singular_value
        ),
        "singular_value_error_bound": _real(certificate.singular_value_error_bound),
        "gap_margin": _real(certificate.gap_margin),
        "resources": {
            "storage_bytes": cost.storage_bytes,
            "original_operator_bytes": cost.original_operator_bytes,
            "preparation_workspace_bytes": cost.preparation_workspace_bytes,
            "apply_workspace_bytes": cost.apply_workspace_bytes,
            "operator_matvec_count": cost.operator_matvec_count,
            "adjoint_matvec_count": cost.adjoint_matvec_count,
        },
    }


def _derivative_admission(result: Any, /) -> dict[str, Any]:
    regularity = result.coupled_regularity
    return {
        "successful": bool(np.asarray(result.successful)),
        "fixed_branch_candidate": bool(np.asarray(result.fixed_branch_candidate)),
        "binding_current": bool(np.asarray(result.binding_current)),
        "branch_admitted": bool(np.asarray(result.branch_admitted)),
        "coupled_regular": bool(np.asarray(result.coupled_regular)),
        "linear_successful": bool(np.asarray(result.linear_successful)),
        "coupled_regularity": _rank_certificate(regularity),
        "coupled_regularity_absent_reason": (
            None
            if regularity is not None
            else "no contact/stop blocks (explicit Euler map) or no regularity policy"
        ),
    }


def _rollout_dynamics(
    plant: PreparedArticulatedPlant, parameters: Any, /
) -> PlantControlDynamics:
    """Complete-state controller codecs over every payload leaf of the plant."""

    template = plant.reset_fallback
    schema = ArrayPyTreeSchema.from_tree(template, case_ndim=0)
    dtype = plant.contact_step.dtype
    size = sum(math.prod(leaf.shape) for leaf in schema.leaves if leaf.dtype.kind in "fc")
    layout = StateLayout(
        (size,),
        axes=("coordinate",),
        local_space=ArraySpace((size,), dtype=dtype),
        tangent_space=ArraySpace((size,), dtype=dtype),
    )
    identities = {
        "semantic_provenance": plant.semantic_provenance,
        "numeric_revision": plant.numeric_revision,
        "executable_signature": plant.execution_signature,
    }
    state_codec = PlantStateVectorCodec(schema, layout, template, **identities)
    command = ArticulatedPlantCommand(
        jnp.zeros(
            (plant.contact_step.actuation_size,), dtype=plant.initial_state.command.dtype
        )
    )
    control_codec = ControlVectorCodec(
        ArrayPyTreeSchema.from_tree(command, case_ndim=0), **identities
    )
    return PlantControlDynamics(
        plant,
        state_codec,
        control_codec,
        parameters=parameters,
        retention="endpoint",
        maximum_retained_bytes=_ROLLOUT_RETAINED_LIMIT,
        method_id="benchmark:native-complete-plant-rollout",
    )


def _rollout_evidence(trajectory: Any, contact_capacity: int, /) -> dict[str, Any]:
    valid = np.asarray(trajectory.valid)
    cases = int(valid.shape[0])
    transition = trajectory.transition_evidence
    attempted = np.asarray(transition.attempted, dtype=np.bool_)
    history = trajectory.plant_history
    final = None if history is None else history.final_state
    manifold = None if final is None else final.payload.contact_history
    active = (
        np.zeros((cases, 0), dtype=np.bool_)
        if manifold is None
        else np.asarray(manifold.active).reshape((cases, -1))
    )
    return {
        "case_count": cases,
        "all_nodes_valid_cases": int(np.sum(np.all(valid, axis=-1))),
        "attempted_transitions": _count(attempted),
        "successful_transitions": _count(transition.successful),
        "transition_status": _histogram(
            np.asarray(transition.status)[attempted], _plant_status_name
        ),
        "first_failure_step": _histogram(transition.first_failure_step, _code),
        "control_status": _histogram(trajectory.status, _code),
        "final_active_contacts_total": _count(active),
        "final_active_contacts_max_per_case": int(active.sum(axis=-1).max(initial=0)),
        "contact_capacity_per_case": contact_capacity,
        "final_time_min": None if final is None else _reduce(final.time, np.min),
        "final_time_max": None if final is None else _reduce(final.time, np.max),
        "final_configuration_finite": (
            None
            if final is None
            else bool(np.all(np.isfinite(np.asarray(final.payload.configuration))))
        ),
    }


def _same_payload(left: PlantRuntimeState, right: PlantRuntimeState, /) -> bool:
    return all(
        np.array_equal(np.asarray(first), np.asarray(second))
        for first, second in zip(
            jax.tree.leaves(left.payload), jax.tree.leaves(right.payload), strict=True
        )
    )


def _native_record(
    row: NativeRow, settings: CampaignSettings, stage: _Stage, /
) -> dict[str, Any]:
    stage.name = "host-model-construction"
    scene, model_seconds = measure_synchronized(lambda: _native_scene(row))
    stage.name = "model-preparation"
    prepared, model_preparation_seconds = measure_synchronized(
        lambda: scene.model.prepare()
    )
    articulation = prepared.articulation
    dtype = np.dtype(articulation.reference_position.dtype)
    nv = articulation.nv
    lifted = np.zeros((articulation.nq,), dtype=dtype)
    lifted[_dof_index(prepared, scene.lift_joint)] = row.initial_gap
    visits = (
        256 * scene.model.manifest.primitive_count
        if row.search_backend == "lbvh"
        else None
    )
    policy = _native_step_policy()
    plan = NativeArticulatedPlantPlan(
        policy,
        gravity=np.asarray([0.0, 0.0, -9.81]),
        actuation_mode=ArticulatedActuationMode.GENERALIZED_EFFORT,
        initial_lifted_configuration=lifted,
        initial_velocity=np.zeros((nv,), dtype=dtype),
        case_ndim=1,
        search_backend=row.search_backend,
        activation_distance=_ACTIVATION_DISTANCE,
        route_capacities=scene.route_capacities,
        maximum_traversal_visits=visits,
    )
    stage.name = "plant-preparation"
    plant, plant_preparation_seconds = measure_synchronized(
        lambda: prepare_articulated_plant(prepared, plan)
    )
    transfer = _transfer_record(plant, stage)
    contact_step = plant.contact_step
    parameters = plant.bind_parameters()
    batch = row.batch_size
    keys = jax.random.split(jax.random.key(_SEED), batch)
    times = jnp.zeros((batch,), dtype=dtype)

    def reset(plant_: Any, keys_: Any, parameters_: Any, times_: Any) -> Any:
        return plant_.reset(keys_, parameters_, case_shape=(batch,), initial_time=times_)

    phases: dict[str, Any] = {}
    reset_result, phases["plant_reset"], _ = _aot_phase(
        "plant-reset", reset, (plant, keys, parameters, times), settings, stage
    )
    source = reset_result.accepted_state
    context = PlantStepContext(
        source.time, source.time + _STEP_SECONDS, source.step_index
    )
    command = ArticulatedPlantCommand(
        jnp.zeros((batch, contact_step.actuation_size), dtype=dtype)
    )
    canonical, phases["plant_step_canonical"] = _canonical_phase(
        "plant-step",
        lambda: plant.step(context, source, command, parameters),
        settings,
        stage,
    )

    def plant_step(
        plant_: Any, context_: Any, source_: Any, command_: Any, parameters_: Any
    ) -> Any:
        return plant_.step(context_, source_, command_, parameters_)

    step_result, phases["plant_step"], _ = _aot_phase(
        "plant-step",
        plant_step,
        (plant, context, source, command, parameters),
        settings,
        stage,
    )

    def components() -> dict[str, Any]:
        case_source = _case(source.payload, 0).mechanics

        def admissibility(step_: Any, source_: Any) -> Any:
            return step_.admissibility(source_)

        def kinematics(articulation_: Any, q_: Any) -> Any:
            return articulation_.forward_kinematics(q_)

        def mass_bind(articulation_: Any, q_: Any) -> Any:
            return prepare_articulated_mass(articulation_, q_)

        admissible, phases["component_static_admissibility"], _ = _aot_phase(
            "component-static-admissibility",
            admissibility,
            (contact_step, case_source),
            settings,
            stage,
        )
        _, phases["component_forward_kinematics"], _ = _aot_phase(
            "component-forward-kinematics",
            kinematics,
            (articulation, case_source.configuration),
            settings,
            stage,
        )
        _, phases["component_mass_bind"], _ = _aot_phase(
            "component-mass-bind",
            mass_bind,
            (articulation, case_source.configuration),
            settings,
            stage,
        )
        return {
            "status": "completed",
            "static_admissibility_case0": _admissibility_evidence(admissible),
        }

    def single_case() -> dict[str, Any]:
        accepted_case = _case(step_result.accepted_state.payload, 0).mechanics
        actuation = jnp.zeros((contact_step.actuation_size,), dtype=dtype)
        gravity = plant.default_parameters.gravity
        duration = jnp.asarray(_STEP_SECONDS, dtype=dtype)

        def single_step(
            step_: Any, source_: Any, duration_: Any, actuation_: Any, gravity_: Any
        ) -> Any:
            return step_.step(source_, duration_, actuation_, gravity_)

        single, phases["single_case_contact_step"], _ = _aot_phase(
            "single-case-contact-step",
            single_step,
            (contact_step, accepted_case, duration, actuation, gravity),
            settings,
            stage,
        )
        return {
            "status": "completed",
            "source": "accepted case 0 after the timed plant step",
            "step": _single_step_evidence(single),
            "fixed_branch_derivative": _native_derivative(
                contact_step, single, actuation, gravity, settings, stage, phases
            ),
        }

    def rollout_section() -> dict[str, Any]:
        stage.name = "rollout-binding"
        dynamics, binding_seconds = measure_synchronized(
            lambda: _rollout_dynamics(plant, parameters)
        )
        grid = TimeGrid(
            jnp.arange(row.horizon + 1, dtype=dtype) * _STEP_SECONDS,
            time_id=f"benchmark:native-rollout:h{row.horizon}",
        )
        parameterization = PiecewiseConstantControlParameterization(
            grid,
            (contact_step.actuation_size,),
            parameterization_id="benchmark:native-zero-generalized-effort",
        )
        coefficients = jnp.zeros(
            (batch, row.horizon, contact_step.actuation_size), dtype=dtype
        )
        problem_id = f"benchmark:{row.row_id}"
        _, phases["plant_rollout_canonical"] = _canonical_phase(
            "plant-rollout",
            lambda: dynamics.rollout(
                grid, source, parameterization, coefficients, problem_id=problem_id
            ),
            settings,
            stage,
        )

        def rollout(
            dynamics_: Any,
            grid_: Any,
            initial_: Any,
            parameterization_: Any,
            coefficients_: Any,
        ) -> Any:
            return dynamics_.rollout(
                grid_, initial_, parameterization_, coefficients_, problem_id=problem_id
            )

        trajectory, phases["plant_rollout"], _ = _aot_phase(
            "plant-rollout",
            rollout,
            (dynamics, grid, source, parameterization, coefficients),
            settings,
            stage,
        )
        evidence = _rollout_evidence(trajectory, contact_step.contact_capacity)
        successful = evidence["successful_transitions"]
        median = _warm_median(phases["plant_rollout"])
        return {
            "status": "completed",
            "codec_binding_seconds": binding_seconds,
            "evidence": evidence,
            "throughput": {
                "attempted_transitions": evidence["attempted_transitions"],
                "successful_transitions": successful,
                "successful_transitions_per_warm_rollout_second": (
                    None if median is None else successful / median
                ),
                "scope": "successful transitions only; failed or refused transitions never count",
            },
            "declared_retained_storage_bytes": dynamics.retained_storage_bytes(
                source, row.horizon
            ),
            "declared_retained_scope": "PlantControlDynamics.retained_storage_bytes for endpoint retention",
        }

    sections: dict[str, Any] = {
        "components": _section(stage, components if settings.native_details else None),
        "single_case": _section(stage, single_case if settings.native_details else None),
        "rollout": _section(stage, rollout_section),
    }

    stage.name = "evidence"
    reset_successful = _count(reset_result.successful)
    step_evidence = _step_evidence(
        step_result, source, contact_step.contact_capacity, contact_step.stop_capacity
    )
    reasons: list[str] = []
    if reset_successful < batch:
        reasons.append(f"reset refused for {batch - reset_successful} of {batch} cases")
    if step_evidence["successful_cases"] < batch:
        reasons.append(
            f"plant step unsuccessful for {batch - step_evidence['successful_cases']} "
            f"of {batch} cases"
        )
    if not step_evidence["rollback"]["verified"]:
        reasons.append("a failed case changed its accepted payload or clock")
    for name, section in sections.items():
        if section["status"] not in ("completed", "not-requested"):
            reasons.append(f"{name} section {section['status']} at {section['stage']}")
    rollout = sections["rollout"]
    if rollout["status"] == "completed":
        valid_cases = rollout["evidence"]["all_nodes_valid_cases"]
        if valid_cases < batch:
            reasons.append(f"rollout invalid for {batch - valid_cases} of {batch} cases")
    single = sections["single_case"]
    if single["status"] == "completed":
        derivative = single["fixed_branch_derivative"]
        if derivative["status"] == "inconsistent-admission":
            reasons.append("fixed-branch JVP and VJP disagree on admission")
        if derivative["fixed_branch_candidate"] and not (
            derivative["jvp"]["binding_current"] and derivative["vjp"]["binding_current"]
        ):
            reasons.append(
                "derivative binding rejected the exact step, actuation and gravity inputs"
            )
        duality = derivative["duality_relative_residual"]
        if duality is not None and not _within(duality, _NATIVE_DUALITY_TOLERANCE):
            reasons.append("fixed-branch JVP/VJP duality exceeds its tolerance")
    manifest = scene.model.manifest
    return {
        "kind": "native-plant",
        "row_id": row.row_id,
        "row": asdict(row),
        "status": "completed",
        "law": _CONTACT_LAW,
        "identity": {
            "model_id": scene.model.model_id,
            "prepared_model_id": prepared.prepared_id,
            "articulation_prepared_id": articulation.prepared_id,
            "generalized_velocity_space_id": articulation.generalized_velocity_space.space_id,
            "plant_id": plant.plant_id,
            "execution_signature_id": plant.execution_signature.signature_id,
            "contact_step_id": contact_step.step_id,
            "contact_catalogue_id": contact_step.catalogue_id,
            "step_policy_id": policy.policy_id,
        },
        "size": {
            "scene": row.scene,
            "body_count": manifest.body_count,
            "joint_count": manifest.joint_count,
            "joint_kinds": list(scene.joint_kinds),
            "generalized_velocities": nv,
            "primitive_count": manifest.primitive_count,
            "plane_count": manifest.plane_count,
            "route_capacities": list(scene.route_capacities),
            "contact_capacity": contact_step.contact_capacity,
            "stop_capacity": contact_step.stop_capacity,
            "history_capacity": contact_step.history_capacity,
            "batch_size": batch,
            "horizon": row.horizon,
            "step_seconds": _STEP_SECONDS,
            "initial_gap_m": row.initial_gap,
            "activation_distance_m": _ACTIVATION_DISTANCE,
            "search_backend": row.search_backend,
            "maximum_traversal_visits": visits,
            "dtype": dtype.name,
            "dtype_scope": "lower_native_robot_tree prepares float64 coordinates; precision varies only in prepared-mass rows",
        },
        "host_phases": {
            "model_construction_seconds": model_seconds,
            "model_construction_scope": "host declarations and lower_native_robot_tree, including plan-owned array placement",
            "model_preparation_seconds": model_preparation_seconds,
            "plant_preparation_seconds": plant_preparation_seconds,
            "explicit_transfer": transfer,
        },
        "phases": phases,
        "plant_step_internal_phases": _STEP_OWNERSHIP,
        "evidence": {
            "reset_successful_cases": reset_successful,
            "reset_status": _histogram(reset_result.status, _plant_status_name),
            "plant_step": step_evidence,
            "aot_matches_canonical_step_bitwise": _same_payload(
                step_result.accepted_state, canonical.accepted_state
            ),
            **sections,
        },
        "logical_retained_bytes": {
            "prepared_plant": logical_array_bytes(plant),
            "runtime_state": logical_array_bytes(source),
        },
        "provider_memory": _PROVIDER_MEMORY,
        "scientific": {
            "status": "passed" if not reasons else "unsuccessful",
            "reasons": reasons,
            "scope": (
                "complete requested phases, including scalar components and derivatives"
                if settings.native_details
                else "reset, public step and complete-state rollout; scalar component and derivative measurements not requested"
            ),
        },
    }


# ------------------------------------------------------------ campaign


def _campaign_rows(
    campaign: _Campaign, /
) -> tuple[tuple[MassRow, ...], tuple[NativeRow, ...]]:
    if campaign == "quick":
        return _QUICK_MASS_ROWS, _QUICK_NATIVE_ROWS
    return _SCALING_MASS_ROWS, _SCALING_NATIVE_ROWS


def _selected_rows(
    campaign: _Campaign, section: _Section, row_ids: Sequence[str], /
) -> tuple[MassRow | NativeRow, ...]:
    mass_rows, native_rows = _campaign_rows(campaign)
    rows: list[MassRow | NativeRow] = []
    if section in ("all", "mass"):
        rows.extend(mass_rows)
    if section in ("all", "native"):
        rows.extend(native_rows)
    if not row_ids:
        return tuple(rows)
    requested = set(row_ids)
    available = {row.row_id for row in rows}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            f"Unknown {campaign}/{section} rows {unknown}; available: {sorted(available)}"
        )
    return tuple(row for row in rows if row.row_id in requested)


def _run_row(row: MassRow | NativeRow, settings: CampaignSettings, /) -> dict[str, Any]:
    stage = _Stage()
    jax.clear_caches()
    started = time.perf_counter_ns()
    try:
        if isinstance(row, MassRow):
            record = _mass_record(row, settings, stage)
        else:
            record = _native_record(row, settings, stage)
    except Exception as error:
        status = _refusal_status(error)
        if status is None:
            raise
        record = {
            "kind": "prepared-mass" if isinstance(row, MassRow) else "native-plant",
            "row_id": row.row_id,
            "row": asdict(row),
            "status": status,
            "refusal": _refusal(error, status, stage),
            "provider_memory": _PROVIDER_MEMORY,
            "scientific": {
                "status": "not-evaluated",
                "reasons": ["the row was refused before its evidence was complete"],
            },
        }
    record["row_wall_seconds"] = (time.perf_counter_ns() - started) / 1.0e9
    jax.clear_caches()
    return record


def _load_baseline(path: Path | None, /) -> tuple[ModuleType | None, dict[str, Any]]:
    """Load the pinned old reduced-dynamics owner beside the current package."""

    if path is None:
        return None, {
            "status": "not-requested",
            "reason": "No --baseline-reference was supplied; no speed comparison is claimed.",
        }
    resolved = path.resolve()
    record: dict[str, Any] = {"path": str(resolved)}
    if not resolved.is_file():
        return None, {**record, "status": "refused", "reason": "file does not exist"}
    record["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    name = "phydrax.discretization.particle._benchmark_reduced_dynamics_baseline"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        return None, {**record, "status": "refused", "reason": "no module loader"}
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, AttributeError, NameError, SyntaxError, TypeError) as error:
        del sys.modules[name]
        return None, {
            **record,
            "status": "refused",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    if not callable(getattr(module, "reduced_forward_dynamics", None)):
        del sys.modules[name]
        return None, {
            **record,
            "status": "refused",
            "reason": "module has no reduced_forward_dynamics",
        }
    return module, {
        **record,
        "status": "loaded",
        "module": name,
        "scope": "reduced_forward_dynamics on identical prepared articulations and inputs in every mass row",
    }


def _cache_policy(allow_persistent: bool, /) -> dict[str, Any]:
    settings = configuration_values()
    directory = settings["jax_compilation_cache_dir"]
    enabled = bool(settings["jax_enable_compilation_cache"])
    if not allow_persistent and enabled and directory:
        raise ValueError(
            "Cold campaigns require launch-time JAX_ENABLE_COMPILATION_CACHE=false "
            "when a persistent cache directory is configured."
        )
    return {
        "policy": "inherited" if allow_persistent else "persistent-cache-not-active",
        "persistent_compilation_cache_enabled": enabled,
        "persistent_compilation_cache_dir": None if directory is None else str(directory),
        "in_process": (
            "jax.clear_caches() before every row, compiled phase and canonical "
            "first call; AOT executables are lowered and compiled per phase"
        ),
    }


def run_campaign(
    campaign: _Campaign,
    /,
    *,
    section: _Section = "all",
    row_ids: Sequence[str] = (),
    repeats: int | None = None,
    warmup: int = 1,
    dense_limit: int = 64,
    baseline_reference: Path | None = None,
    allow_persistent_cache: bool = False,
    native_details: bool = True,
) -> dict[str, Any]:
    """Run explicit diagnostic rows and return one canonical JSON-ready record."""

    resolved_repeats = (3 if campaign == "quick" else 7) if repeats is None else repeats
    if resolved_repeats < 1 or warmup < 0 or dense_limit < 0:
        raise ValueError("repeats must be positive; warmup and dense_limit nonnegative.")
    if not isinstance(native_details, bool):
        raise TypeError("native_details must be a boolean.")
    rows = _selected_rows(campaign, section, row_ids)
    cache_policy = _cache_policy(allow_persistent_cache)
    baseline, baseline_record = _load_baseline(baseline_reference)
    settings = CampaignSettings(
        resolved_repeats, warmup, dense_limit, baseline, native_details
    )
    records = [_run_row(row, settings) for row in rows]
    driver = Path(__file__).resolve()
    statuses = [record["status"] for record in records]
    scientific = [record["scientific"]["status"] for record in records]
    return {
        "benchmark": "native-articulation-mass-and-contact-plant",
        "campaign": campaign,
        "settings": {
            "section": section,
            "row_filter": sorted(row_ids),
            "repeats": resolved_repeats,
            "warmup": warmup,
            "dense_limit": dense_limit,
            "seed": _SEED,
            "native_step_seconds": _STEP_SECONDS,
            "native_details": native_details,
        },
        "cache_policy": cache_policy,
        "baseline": baseline_record,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            driver.parents[1],
            driver,
            ("baseline", "cache_policy", "environment", "records", "settings"),
        ).to_dict(),
        "records": records,
        "summary": {
            "rows": len(records),
            "completed": statuses.count("completed"),
            "refused": statuses.count("refused"),
            "resource_refused": statuses.count("resource-refused"),
            "scientific_passed": scientific.count("passed"),
            "scientific_unsuccessful": scientific.count("unsuccessful"),
            "scientific_not_evaluated": scientific.count("not-evaluated"),
        },
        "interpretation": (
            "Capacity-local diagnostic rows. Phase times are separate measurements; "
            "AOT cold end-to-end is the sum of its measured phases and canonical "
            "first calls are measured whole. Time ratios appear only for successful, "
            "numerically matched work. Same-model smooth measurements (mass and "
            "baseline rows) are separate from native-contact throughput, which is "
            "native hard-contact physics with no provider-law equivalence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark native fixed-base reduced articulation. Without --campaign "
            "the legacy smooth suite runs; --campaign runs phase-separated mass and "
            "native contact plant rows."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="legacy suite only: run one two-joint case with minimal repetitions and rollout work",
    )
    parser.add_argument(
        "--campaign",
        choices=("quick", "scaling"),
        default=None,
        help="run the bounded quick rows or the explicit scaling rows",
    )
    parser.add_argument("--section", choices=("all", "mass", "native"), default="all")
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        help="run only this row id (repeatable); see --list-rows",
    )
    parser.add_argument(
        "--list-rows", action="store_true", help="print the selected row ids and exit"
    )
    parser.add_argument(
        "--repeats", type=int, default=None, help="warm samples per phase"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="untimed warm calls per phase"
    )
    parser.add_argument(
        "--dense-limit",
        type=int,
        default=64,
        help="largest joint count for the independent dense mass reference",
    )
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=None,
        help="pinned old reduced-dynamics module compared on identical inputs",
    )
    parser.add_argument(
        "--allow-persistent-cache",
        action="store_true",
        help="allow an active persistent cache; otherwise launch with it disabled",
    )
    parser.add_argument(
        "--native-execution-only",
        action="store_true",
        help="measure reset, step and rollout without repeating scalar component/AD phases",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="also write the JSON here"
    )
    arguments = parser.parse_args(argv)
    if arguments.campaign is None:
        campaign_only = (
            arguments.section != "all"
            or arguments.row
            or arguments.list_rows
            or arguments.repeats is not None
            or arguments.warmup != 1
            or arguments.dense_limit != 64
            or arguments.baseline_reference is not None
            or arguments.allow_persistent_cache
            or arguments.native_execution_only
            or arguments.output is not None
        )
        if campaign_only:
            parser.error(
                "campaign options require --campaign quick or --campaign scaling"
            )
        print(
            json.dumps(
                run_benchmarks(smoke=arguments.smoke),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.smoke:
        parser.error("--smoke selects the legacy suite; use --campaign quick instead")
    if arguments.list_rows:
        rows = _selected_rows(arguments.campaign, arguments.section, arguments.row)
        print(json.dumps([row.row_id for row in rows], indent=2))
        return
    payload = run_campaign(
        arguments.campaign,
        section=arguments.section,
        row_ids=tuple(arguments.row),
        repeats=arguments.repeats,
        warmup=arguments.warmup,
        dense_limit=arguments.dense_limit,
        baseline_reference=arguments.baseline_reference,
        allow_persistent_cache=arguments.allow_persistent_cache,
        native_details=not arguments.native_execution_only,
    )
    if arguments.output is not None:
        print(write_json_atomic(arguments.output, payload))
    else:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
