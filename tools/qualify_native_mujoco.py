"""Qualify native robot models adapted from compiled MuJoCo against MuJoCo/MJX.

Each fixture is compiled once; that exact ``MjModel`` is adapted with
`adapt_mujoco_model` and both sides are evaluated at identical mapped states
and inputs. Tiers follow the native robotics plan:

* ``smooth-model-mapping``: body-origin/COM/site poses, site Jacobians,
  virtual work, mass matrix and mass actions, inverse dynamics (RNEA), bias
  and forward dynamics (ABA) while the provider has no active constraint;
* ``smooth-actuation``: motor transmission, clamps and source power;
* ``matched-stepping``: one step, only when timestep, integrator, damping
  treatment and laws match (MuJoCo Euler without implicit damping and no
  executed contact or limit law); other requests are refused, not passed;
* ``different-native-contact-physics``: same compiled geometry and source
  state under an explicitly different contact law. Geometry and signed
  distances are compared at identical states; the native plant
  (`prepare_articulated_plant`) then executes the fixture's declared rollout
  and is checked against physical invariants (non-penetration, energy ledger,
  non-positive contact work, inelastic rest, time-step refinement, LBVH versus
  exhaustive search parity), while MuJoCo is checked only against its own
  rest invariant. Trajectories and contact forces are never compared;
* ``fixed-branch-ad``: native JVP/VJP against provider Jacobians in the shared
  smooth domain.

MJX comparisons run when the release-matched provider pair is available. A
missing provider is reported as unavailable, never as a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    measure_repeated,
)
from phydrax import config as pxconfig
from phydrax._fingerprint import canonical_fingerprint
from phydrax._jax_config import x64_enabled
from phydrax.applications.contact._articulated_step import (
    ArticulatedActuationMode,
    ArticulatedContactCertificatePolicy,
    ArticulatedContactSearchBackend,
    ArticulatedContactStepPolicy,
)
from phydrax.applications.contact._cone import ContactConeSolverPlan
from phydrax.applications.robotics import mjx_availability
from phydrax.applications.robotics._articulated_plant import (
    ArticulatedPlantCommand,
    NativeArticulatedPlantPlan,
    prepare_articulated_plant,
    PreparedArticulatedPlant,
)
from phydrax.applications.robotics._mujoco_model import (
    adapt_mujoco_model,
    mujoco_host_availability,
    MujocoJointStopTransformation,
    MujocoModelAdaptationError,
    MujocoModelEvidence,
    MujocoNativeContactTransformation,
)
from phydrax.applications.robotics._native_model import (
    PreparedNativeRobotModel,
    RobotAdaptation,
    RobotContactMaterialTable,
)
from phydrax.discretization import (
    reduced_bias_terms,
    reduced_forward_dynamics,
    reduced_inverse_dynamics,
    reduced_mass_matrix,
    reduced_semi_implicit_velocity_euler_step,
    ReducedArticulationState,
)
from phydrax.discretization.contact._articulated_geometry import (
    ArticulatedPrimitiveKind,
)
from phydrax.discretization.contact._ccd import AdmissiblePathPlan
from phydrax.discretization.particle._articulated_mass import (
    prepare_articulated_mass,
)
from phydrax.discretization.particle._reduced_dynamics import (
    reduced_energy,
    ReducedSemiImplicitVelocityEulerStepPolicy,
)
from phydrax.dynamics import PlantStepContext
from phydrax.qualification import QualificationRuntimeIdentity
from phydrax.typing import parse


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    _ROOT / "tests" / "fixtures" / "robotics" / "native_mujoco_profiles.json"
)


def _array(value: Any, /) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _comparison(
    name: str,
    native: Any,
    provider: Any,
    tolerance: float,
    /,
    *,
    relative: bool = False,
) -> dict[str, object]:
    native_ = _array(native)
    provider_ = _array(provider)
    if native_.shape != provider_.shape:
        return {
            "quantity": name,
            "passed": False,
            "reason": f"shape mismatch {native_.shape} != {provider_.shape}",
        }
    error = float(np.max(np.abs(native_ - provider_), initial=0.0))
    scale = 1.0
    if relative:
        scale = max(
            float(np.max(np.abs(native_), initial=0.0)),
            float(np.max(np.abs(provider_), initial=0.0)),
        )
        scale = scale if scale > 0.0 else 1.0
    measured = error / scale
    finite = bool(np.all(np.isfinite(native_)) and np.all(np.isfinite(provider_)))
    return {
        "quantity": name,
        "error": measured,
        "error_kind": "relative-max" if relative else "absolute-max",
        "tolerance": float(tolerance),
        "passed": finite and measured <= float(tolerance),
    }


def _tier(
    comparisons: Sequence[dict[str, object]],
    /,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "status": "passed"
        if comparisons and all(bool(item["passed"]) for item in comparisons)
        else ("failed" if comparisons else "not-evaluated"),
        "comparisons": list(comparisons),
    }
    if not comparisons:
        record["reason"] = "no comparison was applicable at the sampled states"
    if extra:
        record.update(extra)
    return record


def _not_evaluated(reason: str, /) -> dict[str, object]:
    return {"status": "not-evaluated", "reason": reason, "comparisons": []}


def _refused(reason: str, /) -> dict[str, object]:
    return {"status": "refused", "reason": reason, "comparisons": []}


def matched_stepping_refusal(evidence: MujocoModelEvidence, /) -> str | None:
    """Return why one-step parity is invalid for this source, or ``None``."""

    if evidence.integrator != "euler":
        return (
            f"MuJoCo integrator {evidence.integrator!r} is not the native "
            "semi-implicit velocity Euler step"
        )
    if evidence.contact_mode not in ("no-compiled-contact",):
        return f"contact law differs ({evidence.contact_mode}); trajectories are not comparable"
    if evidence.joint_limit_mode == "native-hard-stops":
        return "MuJoCo regularized limits and native hard stops are different laws"
    if (
        evidence.damping_executed
        and "eulerdamp" not in evidence.disabled_flags
        and any(joint.damping != 0.0 for joint in evidence.joints)
    ):
        return "MuJoCo Euler treats joint damping implicitly (eulerdamp enabled)"
    if "actuation" in evidence.disabled_flags:
        return "source actuation is disabled"
    return None


def _adapt(model: Any, spec: Mapping[str, Any], /) -> RobotAdaptation:
    policy = spec["adaptation"]
    contact = None
    if policy["contact"] is not None:
        declared = policy["contact"]
        contact = MujocoNativeContactTransformation(
            RobotContactMaterialTable(
                np.asarray(declared["material_pairs"], dtype=np.int64),
                np.asarray(declared["static_friction"], dtype=np.float64),
                np.asarray(declared["dynamic_friction"], dtype=np.float64),
            ),
            declared["rationale"],
            default_material_id=declared["default_material_id"],
        )
    stops = (
        None
        if policy["joint_stops"] is None
        else MujocoJointStopTransformation(policy["joint_stops"]["rationale"])
    )
    return adapt_mujoco_model(model, contact=contact, joint_stops=stops)


def _native_inputs(
    evidence: MujocoModelEvidence,
    prepared: PreparedNativeRobotModel,
    rng: np.random.Generator,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample q, v, a, tau in native DOF order and raw MuJoCo ctrl."""

    articulation = prepared.articulation
    joint_by_id = {joint.native_joint_id: joint for joint in evidence.joints}
    ordered = np.asarray(articulation.joint_ids)[
        np.asarray(articulation.dof_joint_indices)
    ]
    lower = []
    upper = []
    for identifier in ordered:
        joint = joint_by_id[int(identifier)]
        span = 0.6 if joint.kind == "hinge" else 0.05
        low, high = -span, span
        if joint.limited:
            low = max(low, 0.8 * (joint.range[0] - joint.qpos0))
            high = min(high, 0.8 * (joint.range[1] - joint.qpos0))
        lower.append(low)
        upper.append(high)
    q = rng.uniform(np.asarray(lower), np.asarray(upper))
    v = rng.uniform(-1.0, 1.0, size=q.shape)
    a = rng.uniform(-2.0, 2.0, size=q.shape)
    tau = rng.uniform(-0.5, 0.5, size=q.shape)
    ctrl = np.zeros((len(evidence.actuators),))
    for actuator in evidence.actuators:
        low, high = actuator.command_range
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = -2.0, 2.0
        center = 0.5 * (low + high)
        half = 0.75 * (high - low)
        ctrl[actuator.mujoco_actuator_id] = rng.uniform(center - half, center + half)
    return q, v, a, tau, ctrl


def _native_control(evidence: MujocoModelEvidence, ctrl: np.ndarray, /) -> np.ndarray:
    command = np.zeros((len(evidence.actuators),))
    for actuator in evidence.actuators:
        command[actuator.native_actuator_id] = ctrl[actuator.mujoco_actuator_id]
    return command


def _passive_and_motor_effort(
    evidence: MujocoModelEvidence,
    prepared: PreparedNativeRobotModel,
    v: np.ndarray,
    ctrl: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    motor = np.zeros_like(v)
    if prepared.motors is not None:
        motor = _array(
            prepared.motors.evaluate(
                jnp.asarray(_native_control(evidence, ctrl)), jnp.asarray(v)
            ).generalized_effort
        )
    damping = np.zeros_like(v)
    if prepared.damping is not None:
        damping = _array(prepared.damping.evaluate(jnp.asarray(v)).generalized_effort)
    return motor, damping


def _set_provider_state(
    mujoco: Any,
    model: Any,
    data: Any,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray],
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    ctrl: np.ndarray,
    /,
) -> None:
    qpos_address, dof_address, qpos0 = maps
    mujoco.mj_resetData(model, data)
    data.qpos[:] = np.asarray(model.qpos0)
    data.qpos[qpos_address] = q + qpos0
    data.qvel[:] = 0.0
    data.qvel[dof_address] = v
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[dof_address] = tau
    if ctrl.size:
        data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)


def _frames(
    prepared: PreparedNativeRobotModel, /
) -> list[tuple[str, str, str, int, np.ndarray]]:
    plan = prepared.model.frames
    local = _array(plan.local_transforms())
    frames = []
    for index, name in enumerate(plan.names.names):
        kind, source = name.split(":", 1)
        frames.append(
            (name, kind, source, int(np.asarray(plan.body_ids)[index]), local[index])
        )
    return frames


def _smooth_mapping(
    mujoco: Any,
    model: Any,
    adaptation: RobotAdaptation,
    prepared: PreparedNativeRobotModel,
    samples: Sequence[tuple[np.ndarray, ...]],
    tolerance: Mapping[str, float],
    /,
) -> dict[str, object]:
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    articulation = prepared.articulation
    maps = evidence.coordinate_map(articulation)
    _, dof_address, _ = maps
    gravity = jnp.asarray(evidence.gravity)
    data = mujoco.MjData(model)
    body_id = {body.name: body.mujoco_body_id for body in evidence.bodies}
    native_body = {body.name: body.native_body_id for body in evidence.bodies}
    comparisons: list[dict[str, object]] = []
    skipped = 0
    for sample, (q, v, a, tau, ctrl) in enumerate(samples):
        _set_provider_state(mujoco, model, data, maps, q, v, tau, ctrl)
        qj = jnp.asarray(q)
        transforms = _array(articulation.body_transforms(qj))
        for name, kind, source, owner, local in _frames(prepared):
            world = transforms[owner] @ local
            if kind == "body":
                provider_position = data.xpos[body_id[source]]
                provider_rotation = data.xmat[body_id[source]].reshape((3, 3))
            else:
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, source)
                provider_position = data.site_xpos[site]
                provider_rotation = data.site_xmat[site].reshape((3, 3))
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:position",
                    world[:3, 3],
                    provider_position,
                    tolerance["position_m"],
                )
            )
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:rotation",
                    world[:3, :3],
                    provider_rotation,
                    tolerance["rotation"],
                )
            )
        for body in evidence.bodies:
            index = native_body[body.name]
            comparisons.append(
                _comparison(
                    f"{sample}:com:{body.name}:position",
                    transforms[index][:3, 3],
                    data.xipos[body.mujoco_body_id],
                    tolerance["position_m"],
                )
            )
            comparisons.append(
                _comparison(
                    f"{sample}:com:{body.name}:rotation",
                    transforms[index][:3, :3],
                    data.ximat[body.mujoco_body_id].reshape((3, 3)),
                    tolerance["rotation"],
                )
            )
        nv = articulation.nv
        basis = np.eye(nv)
        rng = np.random.default_rng(sample)
        for name, kind, source, owner, local in _frames(prepared):
            if kind != "site":
                continue
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, source)
            jacp = np.zeros((3, int(model.nv)))
            jacr = np.zeros((3, int(model.nv)))
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            operator = articulation.frame_jacobian_operator(qj, owner, local[:3, 3])
            native = np.stack(
                [_array(operator.mv(jnp.asarray(basis[k]))) for k in range(nv)], axis=1
            )
            provider = np.concatenate((jacp, jacr), axis=0)[:, dof_address]
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:jacobian", native, provider, tolerance["jacobian"]
                )
            )
            force = rng.uniform(-3.0, 3.0, size=(3,))
            torque = rng.uniform(-1.0, 1.0, size=(3,))
            body = int(model.site_bodyid[site])
            generalized = np.zeros((int(model.nv),))
            mujoco.mj_applyFT(
                model, data, force, torque, data.site_xpos[site], body, generalized
            )
            native_work = _array(
                operator.transpose_mv(jnp.asarray(np.concatenate((force, torque))))
            )
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:virtual-work",
                    native_work,
                    generalized[dof_address],
                    tolerance["effort_relative"],
                    relative=True,
                )
            )
        full = np.zeros((model.nv, model.nv), dtype=np.float64)
        mujoco.mj_fullM(model, data, full)
        provider_mass = full[np.ix_(dof_address, dof_address)]
        native_mass = _array(reduced_mass_matrix(articulation, qj).matrix)
        comparisons.append(
            _comparison(
                f"{sample}:mass-matrix",
                native_mass,
                provider_mass,
                tolerance["mass_relative"],
                relative=True,
            )
        )
        mass = prepare_articulated_mass(articulation, qj)
        product = mass.apply_mass(articulation, qj, jnp.asarray(v))
        provider_product = np.zeros((int(model.nv),))
        velocity = np.zeros((int(model.nv),))
        velocity[dof_address] = v
        mujoco.mj_mulM(model, data, provider_product, velocity)
        comparisons.append(
            _comparison(
                f"{sample}:mass-action",
                product.value,
                provider_product[dof_address],
                tolerance["mass_relative"],
                relative=True,
            )
        )
        bias = reduced_bias_terms(articulation, qj, jnp.asarray(v), gravity)
        comparisons.append(
            _comparison(
                f"{sample}:bias",
                bias.total,
                data.qfrc_bias[dof_address],
                tolerance["effort_relative"],
                relative=True,
            )
        )
        acceleration = np.zeros((int(model.nv),))
        acceleration[dof_address] = a
        data.qacc[:] = acceleration
        inverse = np.zeros((int(model.nv),))
        mujoco.mj_rne(model, data, 1, inverse)
        native_inverse = reduced_inverse_dynamics(
            articulation, qj, jnp.asarray(v), jnp.asarray(a), gravity
        )
        comparisons.append(
            _comparison(
                f"{sample}:inverse-dynamics",
                native_inverse.generalized_effort,
                inverse[dof_address],
                tolerance["effort_relative"],
                relative=True,
            )
        )
        _set_provider_state(mujoco, model, data, maps, q, v, tau, ctrl)
        if int(data.nefc) or int(data.ncon):
            skipped += 1
            continue
        motor, damping = _passive_and_motor_effort(evidence, prepared, v, ctrl)
        effort = tau + motor + damping
        forward = reduced_forward_dynamics(
            articulation, qj, jnp.asarray(v), jnp.asarray(effort), gravity
        )
        comparisons.append(
            _comparison(
                f"{sample}:forward-dynamics",
                forward.acceleration,
                data.qacc[dof_address],
                tolerance["acceleration_relative"],
                relative=True,
            )
        )
        comparisons.append(
            {
                "quantity": f"{sample}:forward-dynamics:native-status",
                "passed": bool(np.asarray(forward.successful)),
            }
        )
    return _tier(
        comparisons,
        extra={"forward_dynamics_skipped_for_active_provider_constraints": skipped},
    )


def _smooth_actuation(
    mujoco: Any,
    model: Any,
    adaptation: RobotAdaptation,
    prepared: PreparedNativeRobotModel,
    samples: Sequence[tuple[np.ndarray, ...]],
    tolerance: Mapping[str, float],
    /,
) -> dict[str, object]:
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    if prepared.motors is None:
        return _not_evaluated("the model declares no actuators")
    maps = evidence.coordinate_map(prepared.articulation)
    _, dof_address, _ = maps
    data = mujoco.MjData(model)
    comparisons: list[dict[str, object]] = []
    for sample, (q, v, _, _, ctrl) in enumerate(samples):
        _set_provider_state(mujoco, model, data, maps, q, v, np.zeros_like(q), ctrl)
        evaluation = prepared.motors.evaluate(
            jnp.asarray(_native_control(evidence, ctrl)), jnp.asarray(v)
        )
        comparisons.append(
            _comparison(
                f"{sample}:generalized-actuator-effort",
                evaluation.generalized_effort,
                data.qfrc_actuator[dof_address],
                tolerance["actuation_relative"],
                relative=True,
            )
        )
        comparisons.append(
            _comparison(
                f"{sample}:actuator-source-power",
                jnp.sum(evaluation.source_power),
                float(np.dot(data.qfrc_actuator, data.qvel)),
                tolerance["power_relative"],
                relative=True,
            )
        )
        if prepared.damping is not None and evidence.damping_executed:
            damping = prepared.damping.evaluate(jnp.asarray(v))
            comparisons.append(
                _comparison(
                    f"{sample}:damping-effort",
                    damping.generalized_effort,
                    data.qfrc_damper[dof_address]
                    if hasattr(data, "qfrc_damper")
                    else data.qfrc_passive[dof_address],
                    tolerance["actuation_relative"],
                    relative=True,
                )
            )
            comparisons.append(
                _comparison(
                    f"{sample}:damping-dissipation",
                    jnp.sum(damping.dissipation_rate),
                    -float(np.dot(data.qfrc_passive, data.qvel)),
                    tolerance["power_relative"],
                    relative=True,
                )
            )
    return _tier(comparisons)


def _matched_stepping(
    mujoco: Any,
    model: Any,
    adaptation: RobotAdaptation,
    prepared: PreparedNativeRobotModel,
    samples: Sequence[tuple[np.ndarray, ...]],
    tolerance: Mapping[str, float],
    /,
) -> dict[str, object]:
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    refusal = matched_stepping_refusal(evidence)
    if refusal is not None:
        return _refused(refusal)
    articulation = prepared.articulation
    maps = evidence.coordinate_map(articulation)
    qpos_address, dof_address, qpos0 = maps
    gravity = jnp.asarray(evidence.gravity)
    step = evidence.timestep
    data = mujoco.MjData(model)
    comparisons: list[dict[str, object]] = []
    for sample, (q, v, _, tau, ctrl) in enumerate(samples):
        _set_provider_state(mujoco, model, data, maps, q, v, tau, ctrl)
        if int(data.nefc) or int(data.ncon):
            continue
        motor, damping = _passive_and_motor_effort(evidence, prepared, v, ctrl)
        result = reduced_semi_implicit_velocity_euler_step(
            articulation,
            ReducedArticulationState(jnp.asarray(q), jnp.asarray(v)),
            jnp.asarray(tau + motor + damping),
            gravity,
            step,
        )
        start_time = float(data.time)
        mujoco.mj_step(model, data)
        comparisons.append(
            _comparison(
                f"{sample}:configuration",
                result.candidate_state.configuration,
                data.qpos[qpos_address] - qpos0,
                tolerance["step_state"],
            )
        )
        comparisons.append(
            _comparison(
                f"{sample}:velocity",
                result.candidate_state.velocity,
                data.qvel[dof_address],
                tolerance["step_state"],
            )
        )
        comparisons.append(
            _comparison(
                f"{sample}:clock",
                start_time + step,
                float(data.time),
                tolerance["step_state"],
            )
        )
    return _tier(
        comparisons,
        extra={
            "timestep": step,
            "integrator": "mujoco euler (eulerdamp off) vs native semi-implicit velocity euler",
        },
    )


def _minimum_plane_gap(collision: Any, transforms: np.ndarray, /) -> float:
    """Smallest native signed distance of any primitive to any static plane."""

    body_ids = np.asarray(collision.body_ids)
    first = _array(collision.local_first)
    second = _array(collision.local_second)
    radii = _array(collision.radii)
    normals = _array(collision.plane_normals)
    offsets = _array(collision.plane_offsets)
    gaps = [np.inf]
    for index in range(collision.count):
        transform = transforms[int(body_ids[index])]
        for point in (first[index], second[index]):
            world = transform[:3, :3] @ point + transform[:3, 3]
            for plane in range(collision.plane_count):
                gaps.append(float(normals[plane] @ world - offsets[plane]) - radii[index])
    return float(min(gaps))


def _contact_geometry(
    mujoco: Any,
    model: Any,
    evidence: MujocoModelEvidence,
    prepared: PreparedNativeRobotModel,
    samples: Sequence[tuple[np.ndarray, ...]],
    tolerance: Mapping[str, float],
    /,
) -> list[dict[str, object]]:
    """Same source state: native vs compiled geometry and signed distances."""

    collision = prepared.model.collision
    assert collision is not None
    articulation = prepared.articulation
    maps = evidence.coordinate_map(articulation)
    data = mujoco.MjData(model)
    geom_id = {geom.native_name: geom.mujoco_geom_id for geom in evidence.geoms}
    body_ids = np.asarray(collision.body_ids)
    first = _array(collision.local_first)
    second = _array(collision.local_second)
    radii = _array(collision.radii)
    normals = _array(collision.plane_normals)
    offsets = _array(collision.plane_offsets)
    comparisons: list[dict[str, object]] = []
    for sample, (q, v, _, _, ctrl) in enumerate(samples):
        _set_provider_state(mujoco, model, data, maps, q, v, np.zeros_like(q), ctrl)
        transforms = _array(articulation.body_transforms(jnp.asarray(q)))
        for plane_index, plane_name in enumerate(collision.plane_names.names):
            geom = geom_id[plane_name]
            provider_normal = data.geom_xmat[geom].reshape((3, 3))[:, 2]
            comparisons.append(
                _comparison(
                    f"{sample}:{plane_name}:normal",
                    normals[plane_index],
                    provider_normal,
                    tolerance["rotation"],
                )
            )
            comparisons.append(
                _comparison(
                    f"{sample}:{plane_name}:offset",
                    offsets[plane_index],
                    float(provider_normal @ data.geom_xpos[geom]),
                    tolerance["position_m"],
                )
            )
        for index, name in enumerate(collision.names.names):
            geom = geom_id[name]
            transform = transforms[int(body_ids[index])]
            world_first = transform[:3, :3] @ first[index] + transform[:3, 3]
            world_second = transform[:3, :3] @ second[index] + transform[:3, 3]
            axis = data.geom_xmat[geom].reshape((3, 3))[:, 2]
            capsule = collision.kinds[index] == ArticulatedPrimitiveKind.CAPSULE
            half = float(model.geom_size[geom, 1]) if capsule else 0.0
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:segment",
                    np.stack((world_first, world_second)),
                    np.stack(
                        (
                            data.geom_xpos[geom] - half * axis,
                            data.geom_xpos[geom] + half * axis,
                        )
                    ),
                    tolerance["position_m"],
                )
            )
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:radius",
                    radii[index],
                    float(model.geom_size[geom, 0]),
                    tolerance["position_m"],
                )
            )
            for plane_index, plane_name in enumerate(collision.plane_names.names):
                native_distance = (
                    min(
                        float(normals[plane_index] @ point - offsets[plane_index])
                        for point in (world_first, world_second)
                    )
                    - radii[index]
                )
                provider_distance = float(
                    mujoco.mj_geomDistance(
                        model, data, geom_id[plane_name], geom, 10.0, None
                    )
                )
                comparisons.append(
                    _comparison(
                        f"{sample}:{name}-{plane_name}:signed-distance",
                        native_distance,
                        provider_distance,
                        tolerance["signed_distance_m"],
                    )
                )
    return comparisons


def _step_policy(config: Mapping[str, Any], /) -> ArticulatedContactStepPolicy:
    """The fixture's declared native step policy; nothing is defaulted silently."""

    root = config["root"]
    return ArticulatedContactStepPolicy(
        ArticulatedContactCertificatePolicy(**config["certificate"]),
        AdmissiblePathPlan(
            float(config["path"]["gap_tolerance_m"]),
            maximum_intervals=int(config["path"]["maximum_intervals"]),
        ),
        cone_solver=ContactConeSolverPlan(**config["cone_solver"]),
        smooth=ReducedSemiImplicitVelocityEulerStepPolicy(**config["smooth"]),
        maximum_root_iterations=int(root["maximum_root_iterations"]),
        root_absolute_tolerance=float(root["root_absolute_tolerance"]),
        root_relative_tolerance=float(root["root_relative_tolerance"]),
        maximum_subdivision_depth=int(root["maximum_subdivision_depth"]),
        maximum_segment_attempts=int(root["maximum_segment_attempts"]),
        minimum_segment_duration=float(root["minimum_segment_duration"]),
    )


def _route_capacities(value: object, /) -> tuple[int, int, int, int, int, int]:
    """The six declared witness-row capacities, in route-kind order."""

    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 6:
        raise ValueError("route_capacities must list exactly six integers.")
    items = tuple(value)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        raise TypeError("route_capacities must contain integers.")
    return (
        int(items[0]),
        int(items[1]),
        int(items[2]),
        int(items[3]),
        int(items[4]),
        int(items[5]),
    )


def _native_rollout(
    prepared: PreparedNativeRobotModel,
    evidence: MujocoModelEvidence,
    config: Mapping[str, Any],
    backend: object,
    duration: float,
    steps: int,
    /,
) -> tuple[PreparedArticulatedPlant, bool, list[Any]]:
    """Reset at the compiled reference and roll out through the native plant."""

    search_backend = parse(backend, ArticulatedContactSearchBackend, "search_backend")
    motors = prepared.motors is not None
    plan = NativeArticulatedPlantPlan(
        _step_policy(config),
        gravity=np.asarray(evidence.gravity),
        actuation_mode=ArticulatedActuationMode.MAPPED_MOTORS
        if motors
        else ArticulatedActuationMode.AUTONOMOUS,
        search_backend=search_backend,
        activation_distance=float(config["activation_distance_m"]),
        route_capacities=_route_capacities(config["route_capacities"]),
        maximum_traversal_visits=int(config["maximum_traversal_visits"])
        if search_backend == "lbvh"
        else None,
    )
    plant = prepare_articulated_plant(prepared, plan)
    command = (
        ArticulatedPlantCommand(jnp.zeros((len(evidence.actuators),))) if motors else None
    )
    parameters = plant.bind_parameters()
    reset = plant.reset(jax.random.key(0), parameters)
    state = reset.accepted_state
    results = []
    for _ in range(steps):
        context = PlantStepContext(state.time, state.time + duration, state.step_index)
        result = plant.step(context, state, command, parameters)
        results.append(result)
        state = result.accepted_state
    return plant, bool(np.asarray(reset.successful)), results


def _check(name: str, passed: bool, /, **values: float) -> dict[str, object]:
    return {"quantity": name, "passed": bool(passed), **values}


def _contact_physics(
    mujoco: Any,
    model: Any,
    adaptation: RobotAdaptation,
    prepared: PreparedNativeRobotModel,
    samples: Sequence[tuple[np.ndarray, ...]],
    tolerance: Mapping[str, float],
    config: Mapping[str, Any] | None,
    /,
) -> dict[str, object]:
    """Different laws: geometry identity plus native step invariants and rest.

    The native plant is executed under its own inelastic hard-contact law and
    checked against physical invariants (non-penetration, energy ledger,
    non-positive contact work, rest, time-step refinement and search-backend
    parity); MuJoCo is simulated under its soft law and checked only against
    its own rest invariant. Trajectories and forces are never compared.
    """

    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    collision = prepared.model.collision
    if collision is None:
        return _not_evaluated("the model lowers no native contact geometry")
    law = {
        "native": collision.contact_law,
        "provider": "mujoco regularized soft contact",
    }
    geometry = _contact_geometry(mujoco, model, evidence, prepared, samples, tolerance)
    if config is None:
        record = _not_evaluated(
            "the fixture declares no native contact rollout; geometry alone is "
            "not a contact-physics pass"
        )
        record.update({"geometry_prerequisite": _tier(geometry), "law": law})
        return record
    limits = config["tolerances"]
    articulation = prepared.articulation
    gravity = jnp.asarray(evidence.gravity)
    plant, reset_ok, results = _native_rollout(
        prepared,
        evidence,
        config,
        config["search_backend"],
        float(config["step_duration_s"]),
        int(config["steps"]),
    )
    checks: list[dict[str, object]] = [_check("native-reset-admissible", reset_ok)]
    previous_energy: float | None = None
    contact_seen = 0.0
    for index, result in enumerate(results):
        step = result.evidence.step
        ledger = step.ledger
        payload = result.accepted_state.payload
        configuration = payload.configuration
        transforms = _array(articulation.body_transforms(configuration))
        gap = _minimum_plane_gap(collision, transforms)
        energy = float(
            reduced_energy(articulation, configuration, payload.velocity, gravity).total
        )
        checks.append(
            _check(
                f"{index}:step-successful-full-interval",
                bool(np.asarray(result.successful))
                and float(np.asarray(step.covered_fraction)) == 1.0,
                status=int(np.asarray(result.status)),
            )
        )
        checks.append(
            _check(
                f"{index}:non-penetration",
                gap >= -limits["penetration_m"],
                minimum_gap_m=gap,
            )
        )
        checks.append(
            _check(
                f"{index}:ledger-identity",
                abs(float(np.asarray(ledger.identity_residual)))
                <= limits["ledger_identity_J"],
                identity_residual_J=float(np.asarray(ledger.identity_residual)),
            )
        )
        checks.append(
            _check(
                f"{index}:non-positive-contact-work",
                float(np.asarray(ledger.positive_contact_work))
                <= limits["positive_contact_work_J"]
                and float(np.asarray(ledger.positive_energy_surplus))
                <= limits["positive_contact_work_J"],
                positive_contact_work_J=float(np.asarray(ledger.positive_contact_work)),
            )
        )
        checks.append(
            _check(
                f"{index}:inelastic-correction-loss",
                float(np.asarray(ledger.correction_loss))
                >= -float(np.asarray(ledger.roundoff_allowance)),
                correction_loss_J=float(np.asarray(ledger.correction_loss)),
            )
        )
        if previous_energy is not None:
            increase = energy - previous_energy - float(np.asarray(ledger.source_work))
            checks.append(
                _check(
                    f"{index}:energy-not-created",
                    increase <= limits["energy_increase_J"],
                    energy_increase_J=increase,
                )
            )
        previous_energy = energy
        contact_seen = max(
            contact_seen,
            float(np.sum(np.asarray(payload.sensors.contact_normal_impulse))),
        )
    final = results[-1].accepted_state.payload
    final_gap = _minimum_plane_gap(
        collision, _array(articulation.body_transforms(final.configuration))
    )
    final_speed = float(np.max(np.abs(np.asarray(final.velocity)), initial=0.0))
    checks.append(_check("contact-established", contact_seen > 0.0, impulse=contact_seen))
    checks.append(
        _check(
            "native-rest-on-contact",
            final_speed <= limits["rest_speed_m_per_s"]
            and abs(final_gap) <= limits["rest_gap_m"],
            final_speed=final_speed,
            final_gap_m=final_gap,
        )
    )
    _, refined_ok, refined = _native_rollout(
        prepared,
        evidence,
        config,
        config["search_backend"],
        float(config["refined_step_duration_s"]),
        int(config["refined_steps"]),
    )
    refined_final = refined[-1].accepted_state.payload
    refined_gap = _minimum_plane_gap(
        collision, _array(articulation.body_transforms(refined_final.configuration))
    )
    checks.append(
        _check(
            "time-step-refinement-rest",
            refined_ok
            and all(bool(np.asarray(item.successful)) for item in refined)
            and abs(refined_gap - final_gap) <= limits["refinement_rest_gap_m"],
            refined_final_gap_m=refined_gap,
        )
    )
    _, reference_ok, reference = _native_rollout(
        prepared,
        evidence,
        config,
        config["reference_search_backend"],
        float(config["step_duration_s"]),
        int(config["steps"]),
    )
    reference_final = reference[-1].accepted_state.payload
    checks.append(
        _comparison(
            "search-backend-parity:configuration",
            final.configuration,
            reference_final.configuration,
            tolerance["step_state"],
        )
    )
    checks.append(
        _comparison(
            "search-backend-parity:velocity",
            final.velocity,
            reference_final.velocity,
            tolerance["step_state"],
        )
    )
    checks.append(
        _check(
            "search-backend-reference-steps",
            reference_ok and all(bool(np.asarray(item.successful)) for item in reference),
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    settle_steps = int(round(limits["provider_settle_s"] / float(model.opt.timestep)))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    geom_id = {geom.native_name: geom.mujoco_geom_id for geom in evidence.geoms}
    provider_gaps = [
        float(
            mujoco.mj_geomDistance(
                model, data, geom_id[plane], geom_id[primitive], 10.0, None
            )
        )
        for plane in collision.plane_names.names
        for primitive in collision.names.names
    ]
    provider_gap = min(provider_gaps)
    checks.append(
        _check(
            "provider-own-law-rest",
            abs(provider_gap) <= limits["provider_rest_gap_m"]
            and float(np.max(np.abs(data.qvel), initial=0.0))
            <= limits["provider_rest_speed_m_per_s"],
            provider_rest_gap_m=provider_gap,
        )
    )
    geometry_tier = _tier(geometry)
    invariant_tier = _tier(checks)
    passed = geometry_tier["status"] == "passed" and invariant_tier["status"] == "passed"
    return {
        "status": "passed" if passed else "failed",
        "comparisons": geometry + checks,
        "geometry_prerequisite": geometry_tier,
        "native_step_invariants": invariant_tier,
        "plant_id": plant.plant_id,
        "search_backend": config["search_backend"],
        "reference_search_backend": config["reference_search_backend"],
        "law": law,
        "rest_gap_law_difference_m": final_gap - provider_gap,
        "not_compared": [
            "contact forces",
            "contact trajectories",
            "impulses and complementarity residuals",
        ],
        "reason_not_compared": "the native and provider contact laws differ by declaration",
    }


def _fixed_branch_ad(
    mujoco: Any,
    model: Any,
    adaptation: RobotAdaptation,
    prepared: PreparedNativeRobotModel,
    samples: Sequence[tuple[np.ndarray, ...]],
    tolerance: Mapping[str, float],
    /,
) -> dict[str, object]:
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    articulation = prepared.articulation
    maps = evidence.coordinate_map(articulation)
    _, dof_address, _ = maps
    data = mujoco.MjData(model)
    comparisons: list[dict[str, object]] = []
    for sample, (q, v, _, _, ctrl) in enumerate(samples):
        _set_provider_state(mujoco, model, data, maps, q, v, np.zeros_like(q), ctrl)
        rng = np.random.default_rng(1000 + sample)
        for name, kind, source, owner, local in _frames(prepared):
            if kind != "site":
                continue
            local_ = jnp.asarray(local)

            def site_position(configuration: jax.Array) -> jax.Array:
                return articulation.frame_transform(configuration, owner, local_)[:3, 3]

            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, source)
            jacp = np.zeros((3, int(model.nv)))
            jacr = np.zeros((3, int(model.nv)))
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            tangent = rng.uniform(-1.0, 1.0, size=q.shape)
            _, jvp = jax.jvp(site_position, (jnp.asarray(q),), (jnp.asarray(tangent),))
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:jvp",
                    jvp,
                    jacp[:, dof_address] @ tangent,
                    tolerance["jacobian"],
                )
            )
            cotangent = rng.uniform(-1.0, 1.0, size=(3,))
            _, pullback = jax.vjp(site_position, jnp.asarray(q))
            (vjp,) = pullback(jnp.asarray(cotangent))
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:vjp",
                    vjp,
                    jacp[:, dof_address].T @ cotangent,
                    tolerance["jacobian"],
                )
            )
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:adjoint-identity",
                    float(jnp.vdot(jnp.asarray(cotangent), jvp)),
                    float(jnp.vdot(vjp, jnp.asarray(tangent))),
                    tolerance["adjoint_relative"],
                    relative=True,
                )
            )
    return _tier(comparisons)


def _mjx_tier(
    mujoco: Any,
    model: Any,
    adaptation: RobotAdaptation,
    prepared: PreparedNativeRobotModel,
    samples: Sequence[tuple[np.ndarray, ...]],
    tolerances: Mapping[str, Mapping[str, float]],
    /,
) -> dict[str, object]:
    availability = mjx_availability()
    if not availability.available:
        return {
            "status": "provider-unavailable",
            "reason": availability.reason,
            "versions": [list(item) for item in availability.versions],
            "comparisons": [],
        }
    import mujoco.mjx as mjx

    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    articulation = prepared.articulation
    maps = evidence.coordinate_map(articulation)
    qpos_address, dof_address, qpos0 = maps
    gravity = jnp.asarray(evidence.gravity)
    device = jax.devices("cpu")[0]
    try:
        device_model = mjx.put_model(model, device=device, impl="jax")
        template = mjx.make_data(device_model, device=device, impl="jax")
    except (NotImplementedError, ValueError) as error:
        return {
            "status": "provider-unsupported",
            "reason": f"{type(error).__name__}: {error}",
            "comparisons": [],
        }
    dtype = str(template.qpos.dtype)
    tolerance = tolerances["float64" if dtype == "float64" else "float32"]
    host = mujoco.MjData(model)
    comparisons: list[dict[str, object]] = []
    for sample, (q, v, _, tau, ctrl) in enumerate(samples):
        _set_provider_state(mujoco, model, host, maps, q, v, tau, ctrl)
        qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
        qpos[qpos_address] = q + qpos0
        qvel = np.zeros((int(model.nv),))
        qvel[dof_address] = v
        applied = np.zeros((int(model.nv),))
        applied[dof_address] = tau
        data = template.replace(
            qpos=jnp.asarray(qpos, dtype=template.qpos.dtype),
            qvel=jnp.asarray(qvel, dtype=template.qvel.dtype),
            qfrc_applied=jnp.asarray(applied, dtype=template.qfrc_applied.dtype),
            ctrl=jnp.asarray(ctrl, dtype=template.ctrl.dtype),
        )
        data = mjx.forward(device_model, data)
        transforms = _array(articulation.body_transforms(jnp.asarray(q)))
        for body in evidence.bodies:
            comparisons.append(
                _comparison(
                    f"{sample}:com:{body.name}:position",
                    transforms[body.native_body_id][:3, 3],
                    np.asarray(data.xipos)[body.mujoco_body_id],
                    tolerance["position_m"],
                )
            )
        for name, kind, source, owner, local in _frames(prepared):
            if kind != "site":
                continue
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, source)
            comparisons.append(
                _comparison(
                    f"{sample}:{name}:position",
                    (transforms[owner] @ local)[:3, 3],
                    np.asarray(data.site_xpos)[site],
                    tolerance["position_m"],
                )
            )
        bias = reduced_bias_terms(articulation, jnp.asarray(q), jnp.asarray(v), gravity)
        comparisons.append(
            _comparison(
                f"{sample}:bias",
                bias.total,
                np.asarray(data.qfrc_bias)[dof_address],
                tolerance["effort_relative"],
                relative=True,
            )
        )
        if int(host.nefc) or int(host.ncon):
            continue
        motor, damping = _passive_and_motor_effort(evidence, prepared, v, ctrl)
        comparisons.append(
            _comparison(
                f"{sample}:generalized-actuator-effort",
                motor,
                np.asarray(data.qfrc_actuator)[dof_address],
                tolerance["actuation_relative"],
                relative=True,
            )
        )
        forward = reduced_forward_dynamics(
            articulation,
            jnp.asarray(q),
            jnp.asarray(v),
            jnp.asarray(tau + motor + damping),
            gravity,
        )
        comparisons.append(
            _comparison(
                f"{sample}:forward-dynamics",
                forward.acceleration,
                np.asarray(data.qacc)[dof_address],
                tolerance["acceleration_relative"],
                relative=True,
            )
        )
    return _tier(
        comparisons,
        extra={
            "dtype": dtype,
            "versions": [list(item) for item in availability.versions],
            "device": f"{device.platform}:{device.id}",
        },
    )


def _timings(
    mujoco: Any,
    model: Any,
    adaptation: RobotAdaptation,
    prepared: PreparedNativeRobotModel,
    sample: tuple[np.ndarray, ...],
    repeats: int,
    /,
) -> dict[str, object]:
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    articulation = prepared.articulation
    q, v, _, tau, ctrl = sample
    gravity = jnp.asarray(evidence.gravity)

    @jax.jit
    def native_forward(
        configuration: jax.Array, velocity: jax.Array, effort: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        result = reduced_forward_dynamics(
            articulation, configuration, velocity, effort, gravity
        )
        return result.acceleration, result.successful, result.status

    arguments = (jnp.asarray(q), jnp.asarray(v), jnp.asarray(tau))
    result, native = measure_repeated(
        lambda: native_forward(*arguments), warmup=1, repeats=repeats
    )
    if not bool(result[1]):
        raise RuntimeError(
            f"Native timing cannot qualify failed forward dynamics: {int(result[2])}."
        )
    data = mujoco.MjData(model)
    maps = evidence.coordinate_map(articulation)
    _set_provider_state(mujoco, model, data, maps, q, v, tau, ctrl)
    _, provider = measure_repeated(
        lambda: mujoco.mj_forward(model, data), warmup=1, repeats=repeats
    )
    return {
        "native_forward_dynamics_jit": native.to_milliseconds_dict(),
        "native_successful": bool(result[1]),
        "native_status": int(result[2]),
        "mujoco_host_mj_forward": provider.to_milliseconds_dict(),
        "repeats": repeats,
    }


_TIERS: dict[str, Callable[..., dict[str, object]]] = {
    "smooth-model-mapping": _smooth_mapping,
    "smooth-actuation": _smooth_actuation,
    "matched-stepping": _matched_stepping,
    "fixed-branch-ad": _fixed_branch_ad,
}


def _source_sha256(path: Path, /) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class _FixtureOutcome:
    """Typed summary of one fixture beside its JSON record."""

    name: str
    adapted: bool
    source_hash_matches: bool
    tier_statuses: tuple[tuple[str, str], ...]
    record: dict[str, object]


@dataclass(frozen=True, slots=True)
class _InvalidOutcome:
    """Typed summary of one deliberately invalid parity request."""

    fixture: str
    tier: str
    refused: bool
    record: dict[str, object]


def _tier_status(tier: Mapping[str, object], /) -> str:
    """Effective status; a failed geometry prerequisite fails its tier."""

    status = tier["status"]
    if not isinstance(status, str):
        raise TypeError("Tier status must be a string.")
    prerequisite = tier.get("geometry_prerequisite")
    if isinstance(prerequisite, Mapping) and prerequisite.get("status") == "failed":
        return "failed"
    return status


def qualify_fixture(
    mujoco: Any,
    spec: Mapping[str, Any],
    profile: Mapping[str, Any],
    fixture_root: Path,
    /,
    *,
    repeats: int = 3,
) -> _FixtureOutcome:
    """Compile one fixture once, adapt that exact model and run its tiers."""

    path = fixture_root / spec["file"]
    source_hash = _source_sha256(path)
    record: dict[str, object] = {
        "fixture": spec["name"],
        "source_sha256": source_hash,
        "source_sha256_matches_profile": source_hash == spec["source_sha256"],
    }
    model = mujoco.MjModel.from_xml_path(str(path))
    adaptation = _adapt(model, spec)
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    prepared = adaptation.model.prepare()
    tolerance = profile["tolerances"]["float64"]
    seed = int(profile["provenance"]["state_seed"])
    rng = np.random.default_rng(seed)
    samples = [
        _native_inputs(evidence, prepared, rng)
        for _ in range(int(profile["provenance"]["state_count"]))
    ]
    record["identity"] = {
        "source_id": evidence.source_id,
        "model_sha256": evidence.model_sha256,
        "mujoco_version": evidence.mujoco_version,
        "native_model_id": adaptation.model.model_id,
        "prepared_id": prepared.prepared_id,
        "evidence_id": evidence.evidence_id,
        "report_id": adaptation.report.report_id,
        "report_status": adaptation.report.status.name,
        "losses": [
            {
                "path": loss.path,
                "category": loss.category,
                "changes_interpretation": loss.changes_interpretation,
            }
            for loss in adaptation.report.losses
        ],
    }
    record["law"] = {
        "contact": evidence.contact_mode,
        "joint_limits": evidence.joint_limit_mode,
        "damping_executed": evidence.damping_executed,
        "integrator": evidence.integrator,
        "timestep": evidence.timestep,
        "gravity": list(evidence.gravity),
        "disabled_flags": list(evidence.disabled_flags),
        "enabled_flags": list(evidence.enabled_flags),
    }
    record["tolerances"] = dict(tolerance)
    tiers: dict[str, dict[str, object]] = {}
    for tier in spec["tiers"]:
        if tier == "different-native-contact-physics":
            tiers[tier] = _contact_physics(
                mujoco,
                model,
                adaptation,
                prepared,
                samples,
                tolerance,
                spec.get("native_contact"),
            )
            continue
        tiers[tier] = _TIERS[tier](
            mujoco, model, adaptation, prepared, samples, tolerance
        )
    tiers["mjx-smooth-model-mapping"] = _mjx_tier(
        mujoco, model, adaptation, prepared, samples, profile["tolerances"]
    )
    record["tiers"] = tiers
    record["timings"] = _timings(mujoco, model, adaptation, prepared, samples[0], repeats)
    return _FixtureOutcome(
        str(spec["name"]),
        True,
        source_hash == spec["source_sha256"],
        tuple((name, _tier_status(tier)) for name, tier in tiers.items()),
        record,
    )


def _invalid_request(
    mujoco: Any,
    request: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    fixture_root: Path,
    /,
) -> _InvalidOutcome:
    spec = specs[request["fixture"]]
    model = mujoco.MjModel.from_xml_path(str(fixture_root / spec["file"]))
    override = request.get("integrator_override")
    if override is not None:
        model.opt.integrator = int(getattr(mujoco.mjtIntegrator, f"mjINT_{override}"))
    adaptation = _adapt(model, spec)
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    if request["tier"] != "matched-stepping":
        raise ValueError(f"Unknown invalid-request tier {request['tier']!r}.")
    refusal = matched_stepping_refusal(evidence)
    return _InvalidOutcome(
        str(request["fixture"]),
        str(request["tier"]),
        refusal is not None,
        {
            "fixture": request["fixture"],
            "tier": request["tier"],
            "integrator_override": override,
            "expected_reason": request["reason"],
            "status": "refused" if refusal is not None else "unexpectedly-admitted",
            "reason": refusal,
            "passed": refusal is not None,
        },
    )


def _runtime_evidence() -> dict[str, object]:
    """Bind actual execution and source identity, without a resolver attestation.

    The generic qualification identity owns these observations. Battery release
    qualification additionally requires its selected dependency lock; a universal
    multi-extra lock is not such an attestation.
    """
    environment = capture_environment()
    identity = capture_benchmark_identity(
        _ROOT, Path(__file__), ("fixtures", "invalid_requests", "runtime_identity")
    )
    settings = pxconfig.snapshot()
    topology = canonical_fingerprint(
        {
            "kind": "native-robotics-execution-topology",
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "devices": [device.to_dict() for device in environment.devices],
        }
    )
    precision = canonical_fingerprint(
        {
            "kind": "native-robotics-execution-precision",
            "x64_enabled": environment.x64_enabled,
            "default_float_dtype": environment.default_float_dtype,
            "matmul_precision": settings.matmul_precision,
            "dtype_promotion": settings.dtype_promotion,
            "rank_promotion": settings.rank_promotion,
        }
    )
    runtime = QualificationRuntimeIdentity(
        identity.fingerprint,
        environment.fingerprint,
        environment.backend,
        topology,
        precision,
    )
    return {
        "identity": identity.to_dict(),
        "environment": environment.to_dict(),
        "runtime_identity": dict(runtime.to_record()),
        "runtime_identity_verified": True,
        "dependency_lock_attestation": "not-claimed",
    }


def qualify(
    profile_path: Path = DEFAULT_PROFILE, /, *, repeats: int = 3
) -> dict[str, object]:
    """Run every profile fixture and invalid request in float64; never pass by default."""

    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        return _qualify(profile_path, repeats)


def _qualify(profile_path: Path, repeats: int, /) -> dict[str, object]:
    """Run the profile with the active numerical configuration."""

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    fixture_root = profile_path.parent
    host = mujoco_host_availability()
    report: dict[str, object] = {
        "kind": "native-mujoco-qualification",
        "profile": str(profile_path.relative_to(_ROOT))
        if profile_path.is_relative_to(_ROOT)
        else str(profile_path),
        "profile_sha256": _source_sha256(profile_path),
        "runtime_identity": None,
        "runtime_identity_verified": False,
        "jax": {
            "backend": jax.default_backend(),
            "x64": x64_enabled(),
            "devices": [f"{device.platform}:{device.id}" for device in jax.devices()],
        },
        "mujoco_host": {
            "available": host.available,
            "reason": host.reason,
            "versions": [list(item) for item in host.versions],
        },
    }
    if not host.available:
        report["status"] = "provider-unavailable"
        return report
    report.update(_runtime_evidence())
    import mujoco

    specs = {spec["name"]: spec for spec in profile["fixtures"]}
    fixtures: list[_FixtureOutcome] = []
    for spec in profile["fixtures"]:
        try:
            fixtures.append(
                qualify_fixture(mujoco, spec, profile, fixture_root, repeats=repeats)
            )
        except MujocoModelAdaptationError as error:
            fixtures.append(
                _FixtureOutcome(
                    str(spec["name"]),
                    False,
                    True,
                    (),
                    {
                        "fixture": spec["name"],
                        "status": "adaptation-refused",
                        "reason": str(error),
                    },
                )
            )
    invalid = [
        _invalid_request(mujoco, request, specs, fixture_root)
        for request in profile["invalid_requests"]
    ]
    report["fixtures"] = [outcome.record for outcome in fixtures]
    report["invalid_requests"] = [outcome.record for outcome in invalid]
    failed: list[str] = []
    unevaluated: list[str] = []
    for outcome in fixtures:
        if not outcome.adapted:
            failed.append(f"{outcome.name}:adaptation")
        if not outcome.source_hash_matches:
            failed.append(f"{outcome.name}:source-hash")
        for name, status in outcome.tier_statuses:
            if status == "failed":
                failed.append(f"{outcome.name}:{name}")
            elif status != "passed":
                unevaluated.append(f"{outcome.name}:{name}")
    failed.extend(
        f"{outcome.fixture}:invalid-{outcome.tier}"
        for outcome in invalid
        if not outcome.refused
    )
    report["failed"] = failed
    report["not_passed_unevaluated"] = unevaluated
    report["status"] = "failed" if failed else "qualified-evaluated-tiers"
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args(argv)
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    report = qualify(arguments.profile.resolve(), repeats=arguments.repeats)
    if arguments.output is not None:
        print(write_json_atomic(arguments.output, report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] == "provider-unavailable":
        raise SystemExit(2)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
