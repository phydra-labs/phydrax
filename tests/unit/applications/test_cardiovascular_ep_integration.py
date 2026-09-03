#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.electrophysiology._integration import (
    commit_physical_monodomain_candidate,
    EventAlignedMultirateSchedule,
    ExplicitReferenceDiffusion,
    ImplicitThetaDiffusion,
    initialize_physical_monodomain,
    integrate_physical_monodomain,
    LieSplit,
    MonodomainMacroInputs,
    PhysicalMonodomainPlan,
    PhysicalMonodomainSpatialBinding,
    propose_physical_monodomain_step,
    PublicDiffusionOperatorInput,
    rollback_physical_monodomain,
    ScheduledMonodomainInputs,
    step_physical_monodomain,
    StrangSplit,
    TensorDiffusionOperatorInput,
    zero_monodomain_macro_inputs,
    zero_scheduled_monodomain_inputs,
)
from phydrax.applications.cardiovascular.electrophysiology._reaction import (
    CardiacReactionEvaluation,
    CardiacReactionParameterLayout,
    CardiacReactionStateLayout,
    plan_reaction,
    prepare_reaction,
)
from phydrax.applications.cardiovascular.electrophysiology._regional_assignment import (
    AblationLesion,
    AnatomicalRegionSelector,
    DiffuseFibrosis,
    RegionalAssignmentRule,
    RegionalElectrophysiologyPlan,
    RegionalHeterogeneity,
    RegionalPhenotype,
    ScarBorderZone,
    ScarCore,
)
from phydrax.applications.cardiovascular.electrophysiology._ventricular_models import (
    TenTusscherPanfilov2006Model,
)
from phydrax.equations import TensorDiffusionAction
from phydrax.linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    OperatorProperties,
)


class _AffineReactionModel:
    state_layout = CardiacReactionStateLayout(
        ("voltage_mV", "gate", "pool"),
        ("mV", "1", "mM"),
        ("gate",),
        ("pool",),
    )
    parameter_layout = CardiacReactionParameterLayout(
        ("forcing", "gate_rate", "pool_rate"),
        ("mV/ms", "1/ms", "1/ms"),
    )
    current_names = ("affine",)
    membrane_capacitance_uF_per_mm2 = 1.0
    membrane_surface_to_volume_per_mm = 1.0

    def __init__(self, forcing: float, model_id: str):
        self.default_parameters = jnp.asarray((forcing, 2.0, 0.5), dtype=jnp.float32)
        self.model_id = model_id

    def initialize(self, batch_shape=(), *, dtype=None):
        resolved_dtype = jnp.float32 if dtype is None else dtype
        voltage = jnp.zeros(batch_shape, dtype=resolved_dtype)
        gate = jnp.ones(batch_shape, dtype=resolved_dtype)
        pool = jnp.ones(batch_shape, dtype=resolved_dtype)
        return jnp.stack((voltage, gate, pool), axis=-1)

    def _parameters(self, state, parameters):
        values = self.default_parameters if parameters is None else parameters
        return jnp.asarray(values, dtype=state.dtype)

    def evaluate(
        self,
        state,
        parameters=None,
        *,
        stimulus_current_uA_per_mm2=0.0,
    ):
        values = self._parameters(state, parameters)
        forcing, gate_rate, pool_rate = values
        gate = state[..., 1]
        pool = state[..., 2]
        stimulus = jnp.asarray(stimulus_current_uA_per_mm2, dtype=state.dtype)
        voltage_rate = forcing + stimulus / self.membrane_capacitance_uF_per_mm2
        voltage_rate = jnp.broadcast_to(voltage_rate, state.shape[:-1])
        gate_derivative = -gate_rate * gate
        pool_derivative = -pool_rate * pool
        state_rate = jnp.stack((voltage_rate, gate_derivative, pool_derivative), axis=-1)
        outward = -jnp.broadcast_to(forcing, state.shape[:-1])
        current_density = outward[..., None]
        zeros = jnp.zeros_like(outward)
        valid = jnp.all(jnp.isfinite(state), axis=-1) & (pool >= 0.0)
        return CardiacReactionEvaluation(
            state_rate,
            jnp.zeros_like(gate)[..., None],
            jnp.broadcast_to((1.0 / gate_rate)[None], gate.shape + (1,)),
            current_density,
            outward,
            pool,
            pool_derivative,
            zeros,
            zeros,
            zeros,
            valid,
            self.current_names,
            self.model_id,
        )

    def rates(
        self,
        state,
        parameters=None,
        *,
        stimulus_current_uA_per_mm2=0.0,
    ):
        return self.evaluate(
            state,
            parameters,
            stimulus_current_uA_per_mm2=stimulus_current_uA_per_mm2,
        ).state_rate

    def exact_gate_update(self, state, dt_ms, parameters=None):
        values = self._parameters(state, parameters)
        gate = state[..., 1] * jnp.exp(-values[1] * dt_ms)
        return state.at[..., 1].set(gate)

    def currents(self, state, parameters=None):
        return self.evaluate(state, parameters).current_density_uA_per_mm2

    def admissible(self, state, parameters=None):
        return self.evaluate(state, parameters).valid

    def validate_state(self, state, parameters=None):
        self.state_layout.require_shape(state)


def _operator(matrix: np.ndarray, operator_id: str = "test-diffusion"):
    return DenseLinearOperator(
        jnp.asarray(matrix, dtype=jnp.float32),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_semidefinite": "asserted",
            },
        ),
        operator_id=operator_id,
    )


def _two_workset_assignment(*, second_effect=None):
    effect = RegionalHeterogeneity(1) if second_effect is None else second_effect
    plan = RegionalElectrophysiologyPlan(
        2,
        (RegionalPhenotype("left", 0), RegionalPhenotype("right", 1)),
        rules=(
            RegionalAssignmentRule(
                AnatomicalRegionSelector((1,)),
                effect,
                rule_id="right-workset",
            ),
        ),
    )
    return plan.prepare(
        np.asarray((101, 205), dtype=np.int64),
        np.asarray((0, 1), dtype=np.int32),
    )


def _runtime(
    *,
    macro_dt_ms: float,
    macro_step_count: int,
    splitting,
    assignment=None,
    matrix=None,
    event_ticks=(0,),
    checkpoint_stride=1,
):
    regional = _two_workset_assignment() if assignment is None else assignment
    diffusion_matrix = (
        np.asarray(((1.0, -1.0), (-1.0, 1.0))) if matrix is None else np.asarray(matrix)
    )
    operator = _operator(diffusion_matrix)
    spatial = PhysicalMonodomainSpatialBinding(
        jnp.ones((2,), dtype=jnp.float32),
        PublicDiffusionOperatorInput(
            operator,
            regional.runtime_id,
            input_id="test-public-diffusion",
        ),
        binding_id="test-spatial-binding",
    )
    schedule = EventAlignedMultirateSchedule(
        macro_dt_ms / 2.0,
        2,
        macro_step_count,
        event_ticks=event_ticks,
        checkpoint_stride=checkpoint_stride,
    )
    integration_plan = PhysicalMonodomainPlan(
        2,
        schedule,
        splitting,
        ImplicitThetaDiffusion(0.5, LinearSolvePolicy(DenseLU())),
        residual_tolerance=2.0e-5,
        checkpoint_capacity=3,
    )
    reactions = (
        prepare_reaction(
            plan_reaction(_AffineReactionModel(1.0, "affine-left"), 1, dtype=np.float32)
        ),
        prepare_reaction(
            plan_reaction(_AffineReactionModel(-0.5, "affine-right"), 1, dtype=np.float32)
        ),
    )
    return integration_plan.prepare(spatial, regional, reactions)


def test_regional_effect_types_are_distinct_and_assignment_is_complete():
    effects = (
        ScarCore(0),
        ScarBorderZone(0),
        DiffuseFibrosis(0),
        AblationLesion(0),
    )
    plan = RegionalElectrophysiologyPlan(
        6,
        (RegionalPhenotype("ventricular", 0),),
        rules=tuple(
            RegionalAssignmentRule(
                AnatomicalRegionSelector((code,), stable_node_ids=(100 + code,)),
                effect,
                rule_id=f"effect-{code}",
            )
            for code, effect in enumerate(effects, start=1)
        ),
    )
    prepared = plan.prepare(
        np.asarray((100, 101, 102, 103, 104, 105), dtype=np.int64),
        np.asarray((0, 1, 2, 3, 4, 0), dtype=np.int32),
    )

    assert bool(prepared.evidence.complete)
    assert prepared.workset_ids == tuple(dict.fromkeys(prepared.workset_ids))
    np.testing.assert_array_equal(
        prepared.evidence.tissue_node_counts,
        np.asarray((2, 1, 1, 1, 1), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        prepared.tissue_codes,
        np.asarray((0, 1, 2, 3, 4, 0), dtype=np.int32),
    )
    assert type(effects[0]) is not type(effects[1])
    assert type(effects[1]) is not type(effects[2])


def test_homogeneous_worksets_route_reaction_and_exact_gate_lanes():
    second_effect = ScarBorderZone(
        1,
        capacitance_scale=2.0,
        ionic_current_scale=0.5,
        state_update_scale=1.0,
    )
    runtime = _runtime(
        macro_dt_ms=0.1,
        macro_step_count=1,
        splitting=LieSplit(),
        assignment=_two_workset_assignment(second_effect=second_effect),
        matrix=np.zeros((2, 2)),
    )
    state = initialize_physical_monodomain(runtime)
    stimulus = jnp.asarray(((2.0, 2.0), (2.0, 2.0)), dtype=jnp.float32)
    result = step_physical_monodomain(runtime, state, MonodomainMacroInputs(stimulus))

    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(
        result.evidence.workset_node_count, np.asarray((1, 1), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        result.evidence.reaction_rate_call_count,
        np.asarray((2, 2), dtype=np.int32),
    )
    # Node 0: +1 mV/ms ionic and +2 mV/ms inward stimulus.
    # Node 1: 0.5*(-0.5)/2 ionic and +2/2 stimulus, all in mV/ms.
    np.testing.assert_allclose(
        result.state.voltage_mV,
        np.asarray((0.3, 0.0875), dtype=np.float32),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        result.state.local_states[0][0, 0],
        math.exp(-0.2),
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        result.state.local_states[0][0, 1],
        0.950625,
        rtol=2.0e-6,
    )


def test_strang_split_converges_faster_than_lie_for_affine_monodomain():
    def error(macro_dt_ms, splitting):
        step_count = int(round(1.0 / macro_dt_ms))
        runtime = _runtime(
            macro_dt_ms=macro_dt_ms,
            macro_step_count=step_count,
            splitting=splitting,
        )
        initial = initialize_physical_monodomain(runtime)
        inputs = ScheduledMonodomainInputs(
            jnp.zeros((runtime.plan.schedule.total_tick_count, 2), dtype=jnp.float32)
        )
        result = integrate_physical_monodomain(runtime, initial, inputs)
        mean = 0.25
        difference = 0.75 * (1.0 - math.exp(-2.0))
        exact = np.asarray(
            (mean + 0.5 * difference, mean - 0.5 * difference),
            dtype=np.float32,
        )
        return float(np.linalg.norm(np.asarray(result.state.voltage_mV) - exact))

    lie_coarse = error(0.2, LieSplit())
    lie_fine = error(0.1, LieSplit())
    strang_coarse = error(0.2, StrangSplit())
    strang_fine = error(0.1, StrangSplit())

    assert lie_fine < 0.6 * lie_coarse
    assert strang_fine < 0.35 * strang_coarse
    assert strang_fine < 0.2 * lie_fine


def test_event_misalignment_rolls_back_and_retains_original_evidence():
    runtime = _runtime(
        macro_dt_ms=0.2,
        macro_step_count=2,
        splitting=StrangSplit(),
        event_ticks=(0, 2),
        checkpoint_stride=1,
    )
    state = initialize_physical_monodomain(runtime)
    misaligned = jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float32)
    result = step_physical_monodomain(runtime, state, MonodomainMacroInputs(misaligned))

    assert not bool(result.evidence.successful)
    assert bool(result.evidence.rolled_back)
    assert not bool(result.evidence.checkpoint_written)
    assert not bool(result.evidence.event_aligned)
    assert np.all(np.isfinite(np.asarray(result.evidence.diffusion_residual_norm_uA)))
    np.testing.assert_array_equal(result.state.voltage_mV, state.voltage_mV)
    np.testing.assert_array_equal(result.state.tick, state.tick)


def test_checkpoint_restores_complete_accepted_state():
    runtime = _runtime(
        macro_dt_ms=0.1,
        macro_step_count=2,
        splitting=LieSplit(),
        checkpoint_stride=1,
    )
    state = initialize_physical_monodomain(runtime)
    first = step_physical_monodomain(
        runtime, state, MonodomainMacroInputs(jnp.zeros((2, 2), dtype=jnp.float32))
    )
    assert bool(first.evidence.checkpoint_written)
    altered = type(first.state)(
        first.state.voltage_mV.at[0].set(jnp.nan),
        tuple(local + 3.0 for local in first.state.local_states),
        first.state.tick + 1,
        first.state.macro_step_index,
        first.state.last_applied_inward_current_uA_per_mm3.at[0].set(jnp.nan),
        first.state.has_previous_stimulus,
        first.state.checkpoints,
        first.state.runtime_id,
    )
    rolled_back = rollback_physical_monodomain(runtime, altered)

    assert bool(rolled_back.evidence.restored)
    np.testing.assert_allclose(rolled_back.state.voltage_mV, first.state.voltage_mV)
    np.testing.assert_allclose(
        rolled_back.state.local_states[0], first.state.local_states[0]
    )
    assert int(rolled_back.state.tick) == int(first.state.tick)

    corrupted_latest = eqx.tree_at(
        lambda current: current.checkpoints.voltage_mV,
        altered,
        altered.checkpoints.voltage_mV.at[1, 0].set(jnp.nan),
    )
    fallback = rollback_physical_monodomain(runtime, corrupted_latest)
    assert bool(fallback.evidence.restored)
    assert int(fallback.evidence.source_slot) == 0
    np.testing.assert_allclose(fallback.state.voltage_mV, state.voltage_mV)
    assert int(fallback.state.tick) == int(state.tick)


def test_explicit_reference_has_no_implicit_method_fallback():
    assignment = _two_workset_assignment()
    operator = _operator(np.zeros((2, 2)), "zero-reference-diffusion")
    spatial = PhysicalMonodomainSpatialBinding(
        jnp.ones((2,), dtype=jnp.float32),
        PublicDiffusionOperatorInput(
            operator,
            assignment.runtime_id,
            input_id="explicit-reference-input",
        ),
        binding_id="explicit-reference-spatial",
    )
    plan = PhysicalMonodomainPlan(
        2,
        EventAlignedMultirateSchedule(0.05, 2, 1),
        LieSplit(),
        ExplicitReferenceDiffusion(0.1),
    )
    reactions = tuple(
        prepare_reaction(
            plan_reaction(_AffineReactionModel(forcing, model_id), 1, dtype=np.float32)
        )
        for forcing, model_id in ((1.0, "explicit-left"), (-0.5, "explicit-right"))
    )
    runtime = plan.prepare(spatial, assignment, reactions)

    assert runtime.full_diffusion_solve is None
    assert runtime.half_diffusion_solve is None
    with pytest.raises(ValueError, match="explicit linear method"):
        ImplicitThetaDiffusion(0.5, LinearSolvePolicy())


def test_generic_tensor_diffusion_action_is_bound_without_local_assembly():
    assignment = _two_workset_assignment()
    operator = _operator(np.zeros((2, 2)), "tensor-action-discretization")
    action = TensorDiffusionAction(
        "voltage_mV",
        jnp.eye(3, dtype=jnp.float32),
        action_id="cardiac-conductivity-tensor",
    )
    bound = TensorDiffusionOperatorInput(
        action,
        operator,
        assignment.runtime_id,
        tensor_action_id=action.action_id,
        input_id="tensor-action-input",
    )

    assert bound.tensor_diffusion_action is action
    assert bound.operator is operator
    assert bound.regional_assignment_id == assignment.runtime_id


def test_spatial_binding_identity_changes_with_physical_content():
    assignment = _two_workset_assignment()
    first_operator = _operator(np.zeros((2, 2)), "shared-operator-label")
    second_operator = _operator(
        np.asarray(((1.0, -1.0), (-1.0, 1.0))),
        "shared-operator-label",
    )

    def bind(volumes, operator):
        return PhysicalMonodomainSpatialBinding(
            jnp.asarray(volumes, dtype=jnp.float32),
            PublicDiffusionOperatorInput(
                operator,
                assignment.runtime_id,
                input_id="shared-input-label",
            ),
            binding_id="shared-user-label",
        )

    baseline = bind((1.0, 1.0), first_operator)
    changed_volume = bind((1.0, 2.0), first_operator)
    changed_operator = bind((1.0, 1.0), second_operator)

    assert baseline.binding_id != "shared-user-label"
    assert baseline.binding_id != changed_volume.binding_id
    assert baseline.binding_id != changed_operator.binding_id
    baseline_runtime = _runtime(
        macro_dt_ms=0.1,
        macro_step_count=1,
        splitting=LieSplit(),
        matrix=np.zeros((2, 2)),
    )
    changed_runtime = _runtime(
        macro_dt_ms=0.1,
        macro_step_count=1,
        splitting=LieSplit(),
        matrix=np.asarray(((1.0, -1.0), (-1.0, 1.0))),
    )
    baseline_state = initialize_physical_monodomain(baseline_runtime)
    changed_state = initialize_physical_monodomain(changed_runtime)
    assert baseline_runtime.runtime_id != changed_runtime.runtime_id
    assert baseline_state.runtime_id != changed_state.runtime_id
    assert baseline_state.checkpoints.runtime_id != changed_state.checkpoints.runtime_id


def test_candidate_commit_rejects_a_different_complete_source_state():
    runtime = _runtime(
        macro_dt_ms=0.1,
        macro_step_count=2,
        splitting=LieSplit(),
    )
    source = initialize_physical_monodomain(runtime)
    candidate = propose_physical_monodomain_step(
        runtime, source, zero_monodomain_macro_inputs(runtime)
    )
    different_source = eqx.tree_at(
        lambda state: state.voltage_mV,
        source,
        source.voltage_mV.at[0].set(1.0),
    )

    with pytest.raises((ValueError, RuntimeError), match="source state does not match"):
        commit_physical_monodomain_candidate(runtime, different_source, candidate)
    different_checkpoint = eqx.tree_at(
        lambda state: state.checkpoints.valid,
        source,
        source.checkpoints.valid.at[1].set(True),
    )
    with pytest.raises((ValueError, RuntimeError), match="source state does not match"):
        commit_physical_monodomain_candidate(runtime, different_checkpoint, candidate)


def test_cadence_and_horizon_are_enforced():
    runtime = _runtime(
        macro_dt_ms=0.1,
        macro_step_count=1,
        splitting=LieSplit(),
    )
    initial = initialize_physical_monodomain(runtime)
    inconsistent = eqx.tree_at(
        lambda state: state.tick,
        initial,
        jnp.asarray(1, dtype=jnp.int32),
    )
    with pytest.raises((ValueError, RuntimeError), match="fixed integer cadence"):
        step_physical_monodomain(
            runtime, inconsistent, zero_monodomain_macro_inputs(runtime)
        )

    accepted = step_physical_monodomain(
        runtime, initial, zero_monodomain_macro_inputs(runtime)
    ).state
    with pytest.raises((ValueError, RuntimeError), match="schedule horizon"):
        step_physical_monodomain(runtime, accepted, zero_monodomain_macro_inputs(runtime))
    with pytest.raises((ValueError, RuntimeError), match="initialized schedule state"):
        integrate_physical_monodomain(
            runtime, accepted, zero_scheduled_monodomain_inputs(runtime)
        )


def test_tp06_prepared_reaction_advances_through_physical_monodomain():
    assignment = RegionalElectrophysiologyPlan(
        1, (RegionalPhenotype("tp06-epicardium", 0),)
    ).prepare(
        np.asarray((9001,), dtype=np.int64),
        np.asarray((0,), dtype=np.int32),
    )
    diffusion = _operator(np.zeros((1, 1)), "tp06-zero-diffusion")
    spatial = PhysicalMonodomainSpatialBinding(
        jnp.ones((1,), dtype=jnp.float32),
        PublicDiffusionOperatorInput(
            diffusion,
            assignment.runtime_id,
            input_id="tp06-production-diffusion",
        ),
        binding_id="tp06-production-spatial",
    )
    reaction = prepare_reaction(
        plan_reaction(TenTusscherPanfilov2006Model(), 1, dtype=np.float32)
    )
    plan = PhysicalMonodomainPlan(
        1,
        EventAlignedMultirateSchedule(0.02, 1, 1, checkpoint_stride=1),
        LieSplit(),
        ExplicitReferenceDiffusion(0.02),
        checkpoint_capacity=2,
    )
    runtime = plan.prepare(spatial, assignment, (reaction,))
    initial = initialize_physical_monodomain(runtime)
    workset = runtime.worksets[0]
    initial_voltage = initial.voltage_mV[workset.node_indices]
    initial_local = initial.local_states[0]
    ionic_voltage_rate = reaction.rates(initial_voltage, initial_local, 0.0)[0]
    initial_current = reaction.currents(initial_voltage, initial_local)
    stimulus = jnp.asarray(((80.0,),), dtype=jnp.float32)
    result = step_physical_monodomain(runtime, initial, MonodomainMacroInputs(stimulus))
    final_voltage = result.state.voltage_mV[workset.node_indices]
    final_local = result.state.local_states[0]
    final_current = reaction.currents(final_voltage, final_local)
    expected_voltage = initial_voltage + 0.02 * (ionic_voltage_rate + 80.0 / 1.4)

    assert bool(result.evidence.successful)
    assert bool(result.evidence.checkpoint_written)
    assert workset.volumetric_capacitance_uF_per_mm3 == pytest.approx(1.4)
    assert float(runtime.lumped_capacitance_uF[0]) == pytest.approx(1.4)
    np.testing.assert_allclose(final_voltage, expected_voltage, rtol=2.0e-5, atol=2.0e-5)
    assert float(final_voltage[0]) > float(initial_voltage[0])
    assert not np.array_equal(np.asarray(final_local), np.asarray(initial_local))
    assert float(final_local[0, 12]) != pytest.approx(
        float(initial_local[0, 12]), abs=1.0e-12
    )
    assert float(final_current[0]) != pytest.approx(float(initial_current[0]), abs=1.0e-7)
    assert bool(reaction.admissible(final_voltage, final_local)[0])
    assert int(result.state.checkpoints.tick[1]) == 1
