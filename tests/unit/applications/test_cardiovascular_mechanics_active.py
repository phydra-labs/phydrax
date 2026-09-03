#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.applications.cardiovascular.electrophysiology._reaction import (
    CardiacReactionEvaluation,
)
from phydrax.applications.cardiovascular.mechanics._active_strain import (
    ActiveStrainPlan,
)
from phydrax.applications.cardiovascular.mechanics._active_stress import (
    ActiveStressPlan,
)
from phydrax.applications.cardiovascular.mechanics._contraction import (
    ActivationDrivenContractionPlan,
    CalciumDrivenFirstOrderContractionPlan,
    ContractionState,
    LandLengthVelocityContractionPlan,
    prepare_contraction,
    PrescribedTensionContractionPlan,
)
from phydrax.applications.cardiovascular.mechanics._electromechanics import (
    ActivationEPToMechanicsPort,
    BidirectionalElectromechanicsPlan,
    ElectricalWindowCandidate,
    ElectromechanicsCadence,
    MechanicalWindowCandidate,
    OneWayElectromechanicsPlan,
    StretchMechanicsToEPPort,
)
from phydrax.discretization import (
    DiscreteFieldSpace,
    FieldTransfer,
    TensorDofLayout,
    TransferProperties,
)
from phydrax.linalg import ArraySpace, DenseLinearOperator
from phydrax.solver import coupling


def _field(name, count):
    vector = ArraySpace((count,), dtype=jnp.float64, space_id=f"{name}-vector")
    return DiscreteFieldSpace(
        name,
        f"{name}-support",
        TensorDofLayout(("node",), (count,), layout_id=f"{name}-layout"),
        vector,
        representation="point_value",
        field_space_id=f"{name}-field",
    )


def _transfer(source, target, matrix, name):
    operator = DenseLinearOperator(
        jnp.asarray(matrix, dtype=jnp.float64),
        source=source.vector_space,
        target=target.vector_space,
        operator_id=f"{name}-operator",
    )
    return FieldTransfer(
        source,
        target,
        operator,
        properties=TransferProperties(
            constant_preserving=True,
            positivity_preserving=True,
            exact_on=("constants",),
        ),
        transfer_id=name,
    )


def _reaction(calcium, model_id="ord-2011"):
    calcium = jnp.asarray(calcium, dtype=jnp.float64)
    batch = calcium.shape
    return CardiacReactionEvaluation(
        state_rate=jnp.zeros((*batch, 2), dtype=jnp.float64),
        gate_steady_state=jnp.zeros((*batch, 1), dtype=jnp.float64),
        gate_time_constant_ms=jnp.ones((*batch, 1), dtype=jnp.float64),
        current_density_uA_per_mm2=jnp.zeros((*batch, 1), dtype=jnp.float64),
        total_outward_current_uA_per_mm2=jnp.zeros(batch, dtype=jnp.float64),
        calcium_cytosol_mM=calcium,
        calcium_cytosol_rate_mM_per_ms=jnp.zeros(batch, dtype=jnp.float64),
        calcium_sr_flux_mM_per_ms=jnp.zeros(batch, dtype=jnp.float64),
        calcium_membrane_current_uA_per_mm2=jnp.zeros(batch, dtype=jnp.float64),
        charge_balance_residual_uA_per_mm2=jnp.zeros(batch, dtype=jnp.float64),
        valid=jnp.ones(batch, dtype=bool),
        current_names=("I_test",),
        model_id=model_id,
    )


def test_all_contraction_fidelities_are_named_and_transactional():
    state = ContractionState.resting((2,), dtype=jnp.float64)
    prescribed = prepare_contraction(PrescribedTensionContractionPlan(), state)
    prescribed_candidate = prescribed.candidate(
        state, jnp.asarray([4.0, -2.0]), jnp.ones(2), 1.0
    )
    assert prescribed.plan.fidelity_id == "prescribed-tension"
    assert jnp.allclose(prescribed_candidate.active_tension, jnp.asarray([4.0, 0.0]))

    activation = prepare_contraction(
        ActivationDrivenContractionPlan(80.0, activation_time=10.0), state
    )
    activated = activation.candidate(state, jnp.ones(2), jnp.ones(2), 10.0)
    assert activation.plan.fidelity_id == "activation-driven-first-order"
    assert jnp.all((activated.active_tension > 0.0) & (activated.active_tension < 80.0))
    accepted = activation.commit(activated)
    failed = activation.candidate(accepted, jnp.asarray([jnp.nan, 1.0]), jnp.ones(2), 1.0)
    rolled_back = activation.commit(failed)
    assert not bool(failed.successful)
    assert jnp.allclose(rolled_back.activation, accepted.activation)
    checkpoint = activation.checkpoint(accepted, 10.0, 1)
    assert jnp.allclose(activation.restore(checkpoint).activation, accepted.activation)

    calcium = CalciumDrivenFirstOrderContractionPlan(
        90.0, 5.0e-4, ionic_model_id="ord-2011"
    )
    assert calcium.fidelity_id == "calcium-driven-first-order"
    land = LandLengthVelocityContractionPlan(100.0, 5.0e-4, ionic_model_id="ord-2011")
    assert land.fidelity_id == "land-length-velocity-calcium"


def test_land_consumes_live_compatible_reaction_calcium_and_length_velocity():
    state = ContractionState.resting((2,), dtype=jnp.float64)
    prepared = prepare_contraction(
        LandLengthVelocityContractionPlan(
            100.0,
            5.0e-4,
            calcium_binding_time=5.0,
            length_sensitivity=2.0,
            velocity_sensitivity=1.0,
            ionic_model_id="ord-2011",
        ),
        state,
    )
    reaction = _reaction([1.0e-3, 1.0e-3])
    candidate = prepared.candidate_from_reaction(
        state,
        reaction,
        jnp.asarray([1.1, 0.9]),
        5.0,
        shortening_velocity=jnp.asarray([0.0, 0.1]),
    )
    assert bool(candidate.successful)
    assert candidate.evidence.live_calcium_consumed
    assert candidate.active_tension[0] > candidate.active_tension[1]
    incompatible = _reaction([1.0e-3, 1.0e-3], model_id="tp06")
    rejected = prepared.candidate_from_reaction(state, incompatible, jnp.ones(2), 5.0)
    assert not bool(rejected.successful)


def test_active_stress_and_active_strain_are_separate_active_mechanics_routes():
    state = ContractionState.resting((1,), dtype=jnp.float64)
    contraction = prepare_contraction(
        ActivationDrivenContractionPlan(50.0, activation_time=1.0), state
    ).candidate(state, jnp.ones(1), jnp.ones(1), 10.0)
    fiber = jnp.asarray([[2.0, 0.0, 0.0]], dtype=jnp.float64)
    sheet = jnp.asarray([[1.0, 3.0, 0.0]], dtype=jnp.float64)
    deformation = jnp.eye(3, dtype=jnp.float64)[None, ...]

    stress = ActiveStressPlan(sheet_tension_fraction=0.1).prepare(fiber, sheet)
    assert jnp.allclose(stress.reference_fiber, jnp.asarray([[1.0, 0.0, 0.0]]))
    assert jnp.allclose(stress.reference_sheet, jnp.asarray([[0.0, 1.0, 0.0]]))
    stress_candidate = stress.candidate(stress.resting_state(), contraction, deformation)
    assert bool(stress_candidate.successful)
    assert stress_candidate.evidence.claim == "active-mechanics-only"
    assert stress_candidate.candidate_state.cauchy_stress[0, 0, 0] > 0.0
    assert stress_candidate.candidate_state.cauchy_stress[0, 1, 1] > 0.0

    strain = ActiveStrainPlan(0.2).prepare(fiber, sheet)
    strain_candidate = strain.candidate(strain.resting_state(), contraction, deformation)
    assert bool(strain_candidate.successful)
    assert strain_candidate.evidence.claim == "active-mechanics-only"
    assert strain_candidate.candidate_state.active_deformation_gradient[0, 0, 0] < 1.0
    assert strain_candidate.evidence.determinant_residual < 1.0e-6
    assert strain_candidate.evidence.reconstruction_residual < 1.0e-6


def test_one_way_multimesh_transfer_cadence_and_rollback():
    ep_field = _field("ep", 2)
    mechanics_field = _field("mechanics", 3)
    interpolation = _transfer(
        ep_field,
        mechanics_field,
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        "ep-mechanics-interpolation",
    )
    port = ActivationEPToMechanicsPort(ep_field, mechanics_field, transfer=interpolation)
    cadence = ElectromechanicsCadence(4)
    plan = OneWayElectromechanicsPlan(
        port, ActivationDrivenContractionPlan(60.0), cadence
    )

    def ep_advance(window, state, stretch, substeps, args):
        del window, stretch, args
        next_state = state + 1.0
        return ElectricalWindowCandidate(
            next_state,
            next_state,
            successful=True,
            work=substeps,
            completed_substeps=substeps,
        )

    def mechanics_advance(window, state, drive, substeps, args):
        del window, state, args
        return MechanicalWindowCandidate(
            drive,
            jnp.ones_like(drive),
            successful=True,
            work=substeps,
            completed_substeps=substeps,
        )

    prepared = plan.prepare(
        ep_advance,
        mechanics_advance,
        jnp.zeros(2, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        t0=0.0,
        t1=2.0,
        coupling_window=1.0,
    )
    run = prepared.solve()
    assert bool(run.solution.successful)
    assert prepared.preparation.forward_transfer_id == interpolation.transfer_id
    assert prepared.preparation.electrophysiology_substeps == 4
    assert jnp.allclose(run.solution.final_state.participant_states[1], jnp.ones(3) * 2.0)
    assert jnp.all(run.evidence.participant_evaluations > 0)

    def failing_mechanics(window, state, drive, substeps, args):
        del window, drive, args
        return MechanicalWindowCandidate(
            state + 10.0,
            jnp.ones_like(state),
            successful=False,
            status=9,
            completed_substeps=substeps,
        )

    failed = plan.prepare(
        ep_advance,
        failing_mechanics,
        jnp.zeros(2, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        t0=0.0,
        t1=1.0,
        coupling_window=1.0,
    ).solve()
    assert not bool(failed.solution.successful)
    assert bool(failed.evidence.rolled_back)
    assert jnp.allclose(failed.solution.final_state.participant_states[1], 0.0)


def _bidirectional(cadence):
    ep_field = _field(f"ep-{cadence.electrophysiology_substeps}", 2)
    mechanics_field = _field(f"mechanics-{cadence.electrophysiology_substeps}", 3)
    forward_transfer = _transfer(
        ep_field,
        mechanics_field,
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        f"forward-{cadence.electrophysiology_substeps}",
    )
    backward_transfer = _transfer(
        mechanics_field,
        ep_field,
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        f"backward-{cadence.electrophysiology_substeps}",
    )
    forward = ActivationEPToMechanicsPort(
        ep_field, mechanics_field, transfer=forward_transfer
    )
    backward = StretchMechanicsToEPPort(
        mechanics_field, ep_field, transfer=backward_transfer
    )
    return BidirectionalElectromechanicsPlan(
        forward,
        backward,
        ActivationDrivenContractionPlan(60.0),
        cadence,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-6,
        maximum_iterations=30,
    )


def _ep_feedback(window, state, stretch, substeps, args):
    del window, state, args
    drive = 0.5 + 0.2 * stretch
    return ElectricalWindowCandidate(
        drive,
        drive,
        successful=True,
        work=substeps,
        completed_substeps=substeps,
    )


def _mechanics_feedback(window, state, drive, substeps, args):
    del window, state, args
    stretch = 1.0 - 0.1 * drive
    return MechanicalWindowCandidate(
        stretch,
        stretch,
        successful=True,
        work=substeps,
        completed_substeps=substeps,
    )


def test_bidirectional_cube_cadence_refinement_and_restart():
    coarse = _bidirectional(ElectromechanicsCadence(2)).prepare(
        _ep_feedback,
        _mechanics_feedback,
        jnp.zeros(2, dtype=jnp.float64),
        jnp.ones(3, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
        t0=0.0,
        t1=2.0,
        coupling_window=1.0,
        rollout=coupling.CouplingRolloutPlan(retention="checkpoints"),
    )
    coarse_run = coarse.solve()
    assert bool(coarse_run.solution.successful)
    assert jnp.max(coarse_run.evidence.interface_residual_norms) < 1.0e-5
    assert jnp.all(coarse_run.solution.converged)

    refined = (
        _bidirectional(ElectromechanicsCadence(4))
        .prepare(
            _ep_feedback,
            _mechanics_feedback,
            jnp.zeros(2, dtype=jnp.float64),
            jnp.ones(3, dtype=jnp.float64),
            jnp.zeros(3, dtype=jnp.float64),
            jnp.ones(2, dtype=jnp.float64),
            t0=0.0,
            t1=2.0,
            coupling_window=1.0,
        )
        .solve()
    )
    assert bool(refined.solution.successful)
    assert jnp.allclose(
        coarse_run.solution.final_state.exchange_values[0],
        refined.solution.final_state.exchange_values[0],
        atol=2.0e-6,
    )

    restart = coarse.restart(coarse_run.solution.final_state, 3.0).solve()
    assert bool(restart.solution.successful)
    assert float(restart.solution.final_state.time) == pytest.approx(3.0)
    assert int(restart.solution.final_state.window_index) == 1
    assert jnp.allclose(
        restart.solution.final_state.exchange_values[0],
        coarse_run.solution.final_state.exchange_values[0],
        atol=2.0e-6,
    )
