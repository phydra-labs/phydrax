#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_tree import ArrayPyTreeSchema
from phydrax._fingerprint import canonical_fingerprint
from phydrax._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.applications.robotics._hybrid_soft import (
    AbstractHybridPlantPort,
    AttachmentFrameState,
    AttachmentWrenchCommand,
    FloatingReducedRodPlantPort,
    FrameWrench,
    HybridRigidSoftPlant,
    HybridRigidSoftStatus,
    PreparedReducedRodPlantPort,
    RigidFrameAttachmentPlan,
    RigidSoftAttachmentPlan,
    SoftEndpointAttachmentPlan,
    SynchronizedStepPolicy,
    TendonDrivenRodPlantPort,
)
from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_floating import (
    FloatingReducedRodPlan,
    FloatingReducedRodPlant,
    FloatingReducedRodPlantControl,
    FloatingReducedRodState,
    prepare_floating_reduced_rod,
)
from phydrax.applications.solid_mechanics._rod_plant import (
    prepare_reduced_rod_plant,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_integrators import (
    ReducedRodSemiImplicitVelocityEuler,
)
from phydrax.applications.solid_mechanics._rod_reduction import ReducedRodPlan
from phydrax.applications.solid_mechanics._rod_tendon import (
    FrictionlessElasticTendonPlan,
    prepare_frictionless_elastic_tendon,
    RodMaterialStation,
    TendonRoutePlan,
)
from phydrax.applications.solid_mechanics._rod_tendon_plant import (
    prepare_tendon_driven_rod_plant,
)
from phydrax.dynamics._plant import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantStepContext,
)


class _BodyPayload(StrictModule):
    position: jax.Array
    rotation: jax.Array
    linear_velocity: jax.Array
    angular_velocity: jax.Array
    applied_force: jax.Array
    applied_moment: jax.Array
    topology_marker: jax.Array
    observed_duration: jax.Array


class _BodyCommands(StrictModule):
    force: jax.Array
    moment: jax.Array
    translation: jax.Array
    fail: jax.Array
    change_topology: jax.Array


class _FakePlant(AbstractDiscretePlant, NonTrainableState):
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: _BodyPayload
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    parameters: PlantParameters

    def __init__(self, name: str, fallback: _BodyPayload, /):
        state_schema = ArrayPyTreeSchema.from_tree(fallback, case_ndim=0)
        command_template = _BodyCommands(
            jnp.zeros((3,), dtype=fallback.position.dtype),
            jnp.zeros((3,), dtype=fallback.position.dtype),
            jnp.zeros((3,), dtype=fallback.position.dtype),
            jnp.asarray(False),
            jnp.asarray(False),
        )
        control_schema = ArrayPyTreeSchema.from_tree(command_template, case_ndim=0)
        parameter_schema = ArrayPyTreeSchema.from_tree((), case_ndim=0)
        semantic = SemanticProvenance({"kind": "hybrid-test-child", "name": name})
        revision = NumericRevision(semantic, {"name": name})
        signature = ExecutableSignature(
            shapes=tuple(
                (f"state:{leaf.path}", leaf.shape) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.shape) for leaf in control_schema.leaves
            ),
            dtypes=tuple(
                (f"state:{leaf.path}", leaf.dtype) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.dtype) for leaf in control_schema.leaves
            ),
            topology_ids={"body": canonical_fingerprint({"name": name})},
        )
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = fallback
        self.semantic_provenance = semantic
        self.numeric_revision = revision
        self.execution_signature = signature
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.parameters = PlantParameters((), parameter_schema.schema_id, revision)

    def propose_reset(
        self,
        keys: jax.Array,
        parameters: Any,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: jax.Array,
    ) -> PlantProposal:
        del keys, parameters, initial_time
        payload = jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(value, case_shape + value.shape),
            self.reset_fallback,
        )
        attempted = jnp.ones(case_shape, dtype=bool)
        status = jnp.zeros(case_shape, dtype=jnp.int32)
        return PlantProposal(payload, payload, attempted, attempted, status, status, ())

    def propose_step(
        self,
        context: PlantStepContext,
        source: _BodyPayload,
        commands: _BodyCommands,
        parameters: Any,
        keys: jax.Array,
        /,
    ) -> PlantProposal:
        del parameters, keys
        candidate = _BodyPayload(
            source.position + commands.translation,
            source.rotation,
            source.linear_velocity,
            source.angular_velocity,
            commands.force,
            commands.moment,
            source.topology_marker + commands.change_topology.astype(jnp.int32),
            context.duration,
        )
        attempted = jnp.ones(context.duration.shape, dtype=bool)
        successful = attempted & ~commands.fail
        status = jnp.where(successful, 0, 73).astype(jnp.int32)
        return PlantProposal(
            candidate,
            candidate,
            attempted,
            successful,
            status,
            status,
            (),
        )


class _FakePort(AbstractHybridPlantPort, NonTrainableState):
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    control_schema_id: str | None = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    step_policy_id: str = eqx.field(static=True)
    frame_ids: tuple[str, ...] = eqx.field(static=True)
    supports_external_wrenches: bool = eqx.field(static=True)

    def __init__(
        self,
        plant: _FakePlant,
        frame_id: str,
        policy: SynchronizedStepPolicy,
        /,
        *,
        state_schema_id: str | None = None,
        topology_id: str = "fixed-body-topology",
    ):
        self.semantic_provenance_id = plant.semantic_provenance.semantic_id
        self.numeric_revision_id = plant.numeric_revision.revision_id
        self.state_schema_id = (
            plant.state_schema.schema_id if state_schema_id is None else state_schema_id
        )
        self.control_schema_id = plant.control_schema.schema_id
        self.execution_signature_id = plant.execution_signature.signature_id
        self.topology_id = topology_id
        self.step_policy_id = policy.policy_id
        self.frame_ids = (frame_id,)
        self.supports_external_wrenches = True

    def frame_state(
        self, payload: _BodyPayload, frame_id: str, /
    ) -> AttachmentFrameState:
        if frame_id != self.frame_ids[0]:
            raise ValueError("Unknown test frame.")
        return AttachmentFrameState(
            payload.position,
            payload.rotation,
            payload.linear_velocity,
            payload.angular_velocity,
        )

    def apply_frame_wrenches(
        self,
        payload: _BodyPayload,
        commands: _BodyCommands,
        wrenches: tuple[FrameWrench, ...],
        /,
    ) -> _BodyCommands:
        del payload
        force = commands.force
        moment = commands.moment
        for wrench in wrenches:
            if wrench.frame_id != self.frame_ids[0]:
                raise ValueError("Wrench was routed to the wrong test frame.")
            force = force + wrench.force
            moment = moment + wrench.moment
        return _BodyCommands(
            force,
            moment,
            commands.translation,
            commands.fail,
            commands.change_topology,
        )

    def topology_unchanged(
        self, source: _BodyPayload, candidate: _BodyPayload, /
    ) -> jax.Array:
        return source.topology_marker == candidate.topology_marker


def _payload(position, linear_velocity) -> _BodyPayload:
    return _BodyPayload(
        jnp.asarray(position, dtype=float),
        jnp.eye(3),
        jnp.asarray(linear_velocity, dtype=float),
        jnp.asarray((0.0, 0.0, 2.0)),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray(5, dtype=jnp.int32),
        jnp.asarray(0.0),
    )


def _child_commands(
    *,
    translation=(0.0, 0.0, 0.0),
    fail=False,
    change_topology=False,
) -> _BodyCommands:
    return _BodyCommands(
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray(translation, dtype=float),
        jnp.asarray(fail),
        jnp.asarray(change_topology),
    )


def _profile(*, name: str = "primary"):
    rigid = _FakePlant(f"{name}-rigid", _payload((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    soft = _FakePlant(f"{name}-soft", _payload((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)))
    policy = SynchronizedStepPolicy(fixed_duration=0.1)
    rigid_port = _FakePort(
        rigid, "rigid-link", policy, topology_id=f"{name}-rigid-topology"
    )
    soft_port = _FakePort(soft, "rod-end", policy, topology_id=f"{name}-soft-topology")
    attachment = RigidSoftAttachmentPlan(
        RigidFrameAttachmentPlan("rigid-link", jnp.asarray((1.0, 0.0, 0.0)), jnp.eye(3)),
        SoftEndpointAttachmentPlan("rod-end", jnp.asarray((0.0, 0.0, 0.0)), jnp.eye(3)),
        position_tolerance=1.0e-6,
        rotation_tolerance=1.0e-6,
        velocity_tolerance=1.0e-6,
        balance_tolerance=1.0e-6,
        power_tolerance=1.0e-6,
    )
    plant = HybridRigidSoftPlant(
        rigid,
        rigid.parameters,
        rigid_port,
        _child_commands(),
        soft,
        soft.parameters,
        soft_port,
        _child_commands(),
        (attachment,),
        policy,
    )
    return plant, rigid, soft, rigid_port, soft_port, attachment, policy


def _step_commands(
    plant: HybridRigidSoftPlant,
    attachment: RigidSoftAttachmentPlan,
    *,
    rigid: _BodyCommands | None = None,
    soft: _BodyCommands | None = None,
) -> Any:
    return plant.commands(
        _child_commands() if rigid is None else rigid,
        _child_commands() if soft is None else soft,
        (
            AttachmentWrenchCommand(
                jnp.asarray((0.0, 2.0, 0.0)),
                jnp.asarray((0.0, 0.0, 3.0)),
                attachment.attachment_id,
            ),
        ),
    )


def _reset(plant: HybridRigidSoftPlant):
    result = plant.reset(jax.random.key(3), plant.parameters)
    assert bool(result.successful)
    return result.accepted_state


def _assert_same_tree(left, right):
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def _prepared_floating_hybrid_rod():
    dtype = jnp.float32
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0), (1.7, 0.0, 0.0)),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.asarray((0.8, 1.1, 0.9), dtype=dtype),
            jnp.asarray(
                (
                    ((0.20, 0.01, 0.0), (0.01, 0.25, 0.0), (0.0, 0.0, 0.30)),
                    ((0.24, 0.0, 0.01), (0.0, 0.31, 0.0), (0.01, 0.0, 0.27)),
                ),
                dtype=dtype,
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((70.0, 55.0, 45.0), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.asarray(
                (((8.0, 0.0, 0.0), (0.0, 9.0, 0.0), (0.0, 0.0, 10.0)),),
                dtype=dtype,
            ),
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        components=("nu_y", "kappa_z"),
        component_scales=jnp.ones((6,), dtype=dtype),
    )
    return prepare_floating_reduced_rod(
        rod, FloatingReducedRodPlan(ReducedRodPlan(basis))
    )


def _prepared_tendon_hybrid_plant():
    floating = _prepared_floating_hybrid_rod()
    base = prepare_reduced_rod_plant(
        floating.fixed_base_dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.01, energy_balance_tolerance=1.0
        ),
    )
    route = TendonRoutePlan(
        (
            RodMaterialStation(0, 0.0, jnp.asarray((0.0, 0.0, 0.0), dtype=jnp.float32)),
            RodMaterialStation(1, 1.0, jnp.asarray((0.0, 0.0, 0.0), dtype=jnp.float32)),
        )
    )
    tendon = prepare_frictionless_elastic_tendon(
        FrictionlessElasticTendonPlan(
            route,
            10.0,
            free_length_bounds=(1.0, 3.0),
            payout_rate_bounds=(-0.5, 0.5),
            tendon_length_bounds=(1.5, 2.5),
            maximum_tension=20.0,
            power_tolerance=1.0e-5,
        ),
        floating.reduction,
    )
    return prepare_tendon_driven_rod_plant(base, (tendon,), (1.7,))


def test_attachment_kinematics_wrench_moment_and_power_close_exactly():
    plant, _, _, _, _, attachment, _ = _profile()
    source = _reset(plant)
    kinematics = plant.attachment_kinematics(source)[0]
    np.testing.assert_allclose(kinematics.rigid_attachment.position, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        kinematics.rigid_attachment.linear_velocity, (1.0, 2.0, 0.0)
    )
    np.testing.assert_allclose(kinematics.position_residual, 0.0)
    np.testing.assert_allclose(kinematics.rotation_residual, 0.0)
    assert bool(kinematics.successful)

    result = plant.step(
        PlantStepContext(0.0, 0.1, 0),
        source,
        _step_commands(plant, attachment),
        plant.parameters,
    )
    assert bool(result.successful)
    route = result.evidence.wrench_routes[0]
    np.testing.assert_allclose(route.rigid_at_attachment.force, (0.0, -2.0, 0.0))
    np.testing.assert_allclose(route.soft_at_attachment.force, (0.0, 2.0, 0.0))
    np.testing.assert_allclose(route.rigid_at_attachment.moment, (0.0, 0.0, -3.0))
    np.testing.assert_allclose(route.soft_at_attachment.moment, (0.0, 0.0, 3.0))
    np.testing.assert_allclose(route.rigid_at_parent.moment, (0.0, 0.0, -5.0))
    np.testing.assert_allclose(route.soft_at_parent.moment, (0.0, 0.0, 3.0))
    np.testing.assert_allclose(route.force_balance_residual, 0.0)
    np.testing.assert_allclose(route.moment_balance_residual, 0.0)
    np.testing.assert_allclose(route.rigid_power, -10.0)
    np.testing.assert_allclose(route.soft_power, 10.0)
    np.testing.assert_allclose(route.power_residual, 0.0)
    assert bool(route.balanced)
    assert bool(route.power_conserving)
    np.testing.assert_allclose(
        result.candidate_state.payload.rigid.applied_force, (0.0, -2.0, 0.0)
    )
    np.testing.assert_allclose(
        result.candidate_state.payload.rigid.applied_moment, (0.0, 0.0, -5.0)
    )
    np.testing.assert_allclose(
        result.candidate_state.payload.soft.applied_force, (0.0, 2.0, 0.0)
    )
    np.testing.assert_allclose(
        result.candidate_state.payload.soft.applied_moment, (0.0, 0.0, 3.0)
    )
    np.testing.assert_allclose(
        result.candidate_state.payload.rigid.observed_duration, 0.1
    )
    np.testing.assert_allclose(result.candidate_state.payload.soft.observed_duration, 0.1)
    np.testing.assert_allclose(result.accepted_state.time, 0.1)
    np.testing.assert_array_equal(result.accepted_state.step_index, 1)


def test_one_child_failure_atomically_rolls_back_both_complete_states():
    plant, _, _, _, _, attachment, _ = _profile()
    source = _reset(plant)
    commands = _step_commands(
        plant,
        attachment,
        soft=_child_commands(fail=True),
    )
    result = plant.step(PlantStepContext(0.0, 0.1, 0), source, commands, plant.parameters)
    assert not bool(result.successful)
    assert int(result.status) == int(HybridRigidSoftStatus.SOFT_STEP_FAILED)
    assert bool(result.evidence.rigid.successful)
    assert not bool(result.evidence.soft.successful)
    _assert_same_tree(result.accepted_state.payload, source.payload)
    np.testing.assert_array_equal(result.accepted_state.time, source.time)
    np.testing.assert_array_equal(result.accepted_state.step_index, source.step_index)
    np.testing.assert_array_equal(
        jax.random.key_data(result.accepted_state.key),
        jax.random.key_data(source.key),
    )
    assert (
        np.linalg.norm(np.asarray(result.candidate_state.payload.rigid.applied_moment))
        > 0.0
    )
    assert (
        np.linalg.norm(np.asarray(result.candidate_state.payload.soft.applied_moment))
        > 0.0
    )


def test_attachment_drift_and_topology_change_each_reject_the_joint_commit():
    plant, _, _, _, _, attachment, _ = _profile()
    source = _reset(plant)
    drift = plant.step(
        PlantStepContext(0.0, 0.1, 0),
        source,
        _step_commands(
            plant,
            attachment,
            soft=_child_commands(translation=(0.01, 0.0, 0.0)),
        ),
        plant.parameters,
    )
    assert not bool(drift.successful)
    assert int(drift.status) == int(HybridRigidSoftStatus.INVALID_ACCEPTED_ATTACHMENT)
    _assert_same_tree(drift.accepted_state.payload, source.payload)
    assert not bool(drift.evidence.accepted_attachments[0].successful)

    topology = plant.step(
        PlantStepContext(0.0, 0.1, 0),
        source,
        _step_commands(
            plant,
            attachment,
            rigid=_child_commands(change_topology=True),
        ),
        plant.parameters,
    )
    assert not bool(topology.successful)
    assert int(topology.status) == int(HybridRigidSoftStatus.TOPOLOGY_CHANGED)
    _assert_same_tree(topology.accepted_state.payload, source.payload)
    assert not bool(topology.evidence.topology_unchanged)


def test_fixed_duration_policy_is_enforced_on_the_shared_context():
    plant, _, _, _, _, attachment, _ = _profile()
    source = _reset(plant)
    result = plant.step(
        PlantStepContext(0.0, 0.2, 0),
        source,
        _step_commands(plant, attachment),
        plant.parameters,
    )
    assert not bool(result.successful)
    assert int(result.status) == int(HybridRigidSoftStatus.INCOMPATIBLE_DURATION)
    _assert_same_tree(result.accepted_state.payload, source.payload)
    np.testing.assert_allclose(
        result.candidate_state.payload.rigid.observed_duration, 0.2
    )
    np.testing.assert_allclose(result.candidate_state.payload.soft.observed_duration, 0.2)


def test_port_policy_schema_parameter_and_runtime_provenance_mismatches_reject():
    plant, rigid, soft, rigid_port, soft_port, attachment, policy = _profile()
    incompatible = SynchronizedStepPolicy(fixed_duration=0.2)
    bad_policy_port = _FakePort(
        soft, "rod-end", incompatible, topology_id="primary-soft-topology"
    )
    with pytest.raises(ValueError, match="incompatible step policy"):
        HybridRigidSoftPlant(
            rigid,
            rigid.parameters,
            rigid_port,
            _child_commands(),
            soft,
            soft.parameters,
            bad_policy_port,
            _child_commands(),
            (attachment,),
            policy,
        )

    bad_schema_port = _FakePort(
        rigid,
        "rigid-link",
        policy,
        state_schema_id="foreign-state-schema",
        topology_id="primary-rigid-topology",
    )
    with pytest.raises(ValueError, match="identities"):
        HybridRigidSoftPlant(
            rigid,
            rigid.parameters,
            bad_schema_port,
            _child_commands(),
            soft,
            soft.parameters,
            soft_port,
            _child_commands(),
            (attachment,),
            policy,
        )

    foreign_semantic = SemanticProvenance({"kind": "foreign-parameters"})
    foreign_revision = NumericRevision(foreign_semantic, {})
    foreign_parameters = PlantParameters(
        (), rigid.parameter_schema.schema_id, foreign_revision
    )
    with pytest.raises(ValueError, match="numeric revision"):
        HybridRigidSoftPlant(
            rigid,
            foreign_parameters,
            rigid_port,
            _child_commands(),
            soft,
            soft.parameters,
            soft_port,
            _child_commands(),
            (attachment,),
            policy,
        )

    foreign_plant, _, _, _, _, foreign_attachment, _ = _profile(name="foreign")
    source = _reset(plant)
    with pytest.raises(ValueError, match="does not match this plant"):
        foreign_plant.step(
            PlantStepContext(0.0, 0.1, 0),
            source,
            _step_commands(foreign_plant, foreign_attachment),
            foreign_plant.parameters,
        )


def test_floating_rod_port_exposes_endpoint_twist_and_adds_exact_dual_wrench():
    prepared = _prepared_floating_hybrid_rod()
    initial = prepared.initialize_state()
    moving = FloatingReducedRodState(
        initial.base_pose,
        initial.coefficients,
        jnp.asarray((0.3, -0.2, 0.1, 0.0, 0.0, 0.4), dtype=jnp.float32),
        jnp.asarray((0.2, -0.1), dtype=jnp.float32),
    )
    plant = FloatingReducedRodPlant(prepared, initial_state=moving)
    policy = SynchronizedStepPolicy(fixed_duration=1.0e-4)
    port = FloatingReducedRodPlantPort(
        plant,
        policy,
        base_frame_id="floating-root",
        tip_frame_id="rod-tip",
    )
    payload = plant.initial_payload
    frame = port.frame_state(payload, "rod-tip")
    native = prepared.lift(moving)
    tip_node = int(np.asarray(prepared.reduction.path_node_ids[-1]))

    np.testing.assert_allclose(frame.position, native.positions[tip_node])
    np.testing.assert_allclose(frame.rotation, jnp.eye(3, dtype=jnp.float32))
    np.testing.assert_allclose(frame.linear_velocity, native.velocities[tip_node])
    np.testing.assert_allclose(frame.angular_velocity, native.angular_velocities[-1])

    original = FloatingReducedRodPlantControl(
        jnp.linspace(-0.2, 0.3, prepared.tangent_size, dtype=jnp.float32)
    )
    wrench = FrameWrench(
        jnp.asarray((0.4, -0.5, 0.6), dtype=jnp.float32),
        jnp.asarray((-0.3, 0.2, 0.1), dtype=jnp.float32),
        "rod-tip",
    )
    routed = port.apply_frame_wrenches(payload, original, (wrench,))
    native_forces = jnp.zeros_like(native.velocities).at[tip_node].set(wrench.force)
    native_moments = jnp.zeros_like(native.angular_velocities).at[-1].set(wrench.moment)
    expected_delta = prepared.effort_pullback_operator(moving).mv(
        (native_forces, native_moments)
    )

    np.testing.assert_allclose(routed.effort, original.effort + expected_delta)
    np.testing.assert_allclose(
        jnp.vdot(expected_delta, moving.velocity).real,
        jnp.vdot(wrench.force, frame.linear_velocity).real
        + jnp.vdot(wrench.moment, frame.angular_velocity).real,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_array_equal(
        original.effort,
        jnp.linspace(-0.2, 0.3, prepared.tangent_size, dtype=jnp.float32),
    )


def test_two_real_floating_rod_ports_compose_and_retain_child_transactions():
    prepared = _prepared_floating_hybrid_rod()
    rigid = FloatingReducedRodPlant(prepared)
    soft = FloatingReducedRodPlant(prepared)
    policy = SynchronizedStepPolicy(fixed_duration=1.0e-4)
    rigid_port = FloatingReducedRodPlantPort(
        rigid, policy, base_frame_id="rigid-root", tip_frame_id="rigid-tip"
    )
    soft_port = FloatingReducedRodPlantPort(
        soft, policy, base_frame_id="soft-root", tip_frame_id="soft-tip"
    )
    attachment = RigidSoftAttachmentPlan(
        RigidFrameAttachmentPlan(
            "rigid-tip", jnp.zeros((3,), dtype=jnp.float32), jnp.eye(3, dtype=jnp.float32)
        ),
        SoftEndpointAttachmentPlan(
            "soft-tip", jnp.zeros((3,), dtype=jnp.float32), jnp.eye(3, dtype=jnp.float32)
        ),
        position_tolerance=1.0,
        rotation_tolerance=1.0,
        velocity_tolerance=1.0,
        balance_tolerance=2.0e-6,
        power_tolerance=2.0e-6,
    )
    hybrid = HybridRigidSoftPlant(
        rigid,
        rigid.bind_parameters(),
        rigid_port,
        rigid.zero_control(),
        soft,
        soft.bind_parameters(),
        soft_port,
        soft.zero_control(),
        (attachment,),
        policy,
    )
    source = _reset(hybrid)
    commands = hybrid.commands(
        rigid.zero_control(),
        soft.zero_control(),
        (
            AttachmentWrenchCommand(
                jnp.asarray((0.0, 0.2, 0.0), dtype=jnp.float32),
                jnp.asarray((0.0, 0.0, 0.05), dtype=jnp.float32),
                attachment.attachment_id,
            ),
        ),
    )
    result = hybrid.step(
        PlantStepContext(
            source.time,
            source.time + jnp.asarray(1.0e-4, dtype=source.time.dtype),
            source.step_index,
        ),
        source,
        commands,
        hybrid.parameters,
    )

    assert bool(result.successful)
    route = result.evidence.wrench_routes[0]
    rigid_routed = rigid_port.apply_frame_wrenches(
        source.payload.rigid, commands.rigid, (route.rigid_at_parent,)
    )
    soft_routed = soft_port.apply_frame_wrenches(
        source.payload.soft, commands.soft, (route.soft_at_parent,)
    )
    np.testing.assert_allclose(
        result.evidence.rigid.evidence.dynamics.evaluation.forces.direct_effort,
        rigid_routed.effort,
    )
    np.testing.assert_allclose(
        result.evidence.soft.evidence.dynamics.evaluation.forces.direct_effort,
        soft_routed.effort,
    )
    np.testing.assert_allclose(route.rigid_power + route.soft_power, 0.0)
    _assert_same_tree(
        result.candidate_state.payload.rigid,
        result.evidence.rigid.candidate_state.payload,
    )
    _assert_same_tree(
        result.candidate_state.payload.soft,
        result.evidence.soft.candidate_state.payload,
    )
    _assert_same_tree(
        result.accepted_state.payload.rigid,
        result.evidence.rigid.accepted_state.payload,
    )
    _assert_same_tree(
        result.accepted_state.payload.soft,
        result.evidence.soft.accepted_state.payload,
    )


def test_passive_reduced_rod_port_is_rejected_before_hybrid_composition():
    prepared = _prepared_floating_hybrid_rod()
    passive = prepare_reduced_rod_plant(
        prepared.fixed_base_dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.01, energy_balance_tolerance=1.0
        ),
    )
    floating = FloatingReducedRodPlant(prepared)
    policy = SynchronizedStepPolicy(fixed_duration=1.0e-4)
    passive_port = PreparedReducedRodPlantPort(passive, policy)
    floating_port = FloatingReducedRodPlantPort(floating, policy)
    attachment = RigidSoftAttachmentPlan(
        RigidFrameAttachmentPlan(
            "tip", jnp.zeros((3,), dtype=jnp.float32), jnp.eye(3, dtype=jnp.float32)
        ),
        SoftEndpointAttachmentPlan(
            "tip", jnp.zeros((3,), dtype=jnp.float32), jnp.eye(3, dtype=jnp.float32)
        ),
    )

    assert passive.control_schema is None
    assert passive_port.supports_external_wrenches is False
    with pytest.raises(ValueError, match="cannot accept attachment-frame wrenches"):
        HybridRigidSoftPlant(
            passive,
            passive.bind_parameters(),
            passive_port,
            None,
            floating,
            floating.bind_parameters(),
            floating_port,
            floating.zero_control(),
            (attachment,),
            policy,
        )


def test_tendon_port_preserves_payout_commands_and_adds_only_endpoint_effort():
    plant = _prepared_tendon_hybrid_plant()
    policy = SynchronizedStepPolicy(fixed_duration=1.0e-4)
    port = TendonDrivenRodPlantPort(
        plant, policy, base_frame_id="tendon-root", tip_frame_id="tendon-tip"
    )
    payload = plant.initial_state
    reduction = plant.base_plant.dynamics.reduction
    native = reduction.lift(payload.reduced_state)
    tip_node = int(np.asarray(reduction.path_node_ids[-1]))
    frame = port.frame_state(payload, "tendon-tip")
    original = plant.command(
        (0.25,),
        external_effort=jnp.asarray((0.1, -0.2), dtype=jnp.float32),
    )
    wrench = FrameWrench(
        jnp.asarray((0.0, 0.4, -0.1), dtype=jnp.float32),
        jnp.asarray((0.0, 0.0, 0.2), dtype=jnp.float32),
        "tendon-tip",
    )
    routed = port.apply_frame_wrenches(payload, original, (wrench,))
    native_forces = jnp.zeros_like(native.velocities).at[tip_node].set(wrench.force)
    native_moments = jnp.zeros_like(native.angular_velocities).at[-1].set(wrench.moment)
    expected_delta = reduction.pullback_loads(
        payload.reduced_state.coefficients, native_forces, native_moments
    )

    plant.control_schema.validate(routed)
    np.testing.assert_array_equal(
        routed.tendon_commands[0].payout_rate,
        original.tendon_commands[0].payout_rate,
    )
    np.testing.assert_allclose(
        routed.external_effort, original.external_effort + expected_delta
    )
    np.testing.assert_array_equal(
        original.external_effort, jnp.asarray((0.1, -0.2), dtype=jnp.float32)
    )
    np.testing.assert_allclose(
        jnp.vdot(expected_delta, payload.reduced_state.coefficient_velocities).real,
        jnp.vdot(wrench.force, frame.linear_velocity).real
        + jnp.vdot(wrench.moment, frame.angular_velocity).real,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
