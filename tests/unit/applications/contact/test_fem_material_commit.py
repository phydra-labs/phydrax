#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._strict import StrictModule
from phydrax.applications.contact import (
    FiniteElementContactParticipant,
    MonolithicContactPlan,
    MonolithicContactState,
)
from phydrax.applications.solid_mechanics import (
    FiniteElementDynamicsExecutionContext,
    FiniteElementDynamicsState,
    FiniteStrainJ2MPMConstitutivePlan,
    FiniteStrainJ2Parameters,
)
from phydrax.equations import (
    CompiledFiniteElementProblem,
    FiniteElementExecutionContext,
    MaterialTransaction,
    MPMConstitutiveResponse,
)
from phydrax.equations.fem import FiniteElementAuxiliaryEvaluation


pytestmark = pytest.mark.strict_jax


@final
class _Plasticity(StrictModule):
    material: FiniteStrainJ2MPMConstitutivePlan
    parameters: FiniteStrainJ2Parameters

    def response(
        self, deformation: Array, dynamics: FiniteElementDynamicsExecutionContext
    ) -> MPMConstitutiveResponse:
        materials = dynamics.materials
        if materials is None:
            raise ValueError("The plastic FEM case requires committed materials.")
        return self.material.evaluate(
            deformation,
            materials.states[0].committed,
            jnp.asarray(1.0, dtype=deformation.dtype),
            self.parameters,
            dynamics.end_time,
            dynamics.step_size,
        )

    def residual(
        self,
        values: tuple[Array, ...],
        gradients: tuple[Array, ...],
        points: Array,
        weights: Array,
        test_basis: Array,
        test_gradients: Array,
        context: FiniteElementExecutionContext,
    ) -> Array:
        del values, points, test_basis
        dynamics = context.user_args
        if not isinstance(dynamics, FiniteElementDynamicsExecutionContext):
            raise TypeError("Plastic FEM requires a dynamics execution context.")
        # One affine P1 tetrahedron has the same deformation at every point.
        deformation = jnp.eye(3, dtype=gradients[0].dtype) + gradients[0][0, 0]
        response = self.response(deformation, dynamics)
        return phx.ein.contract(
            "cq,cqib,ab->cia", weights, test_gradients, response.first_piola
        )

    def auxiliary(
        self, displacement: Array, context: FiniteElementExecutionContext
    ) -> FiniteElementAuxiliaryEvaluation:
        dynamics = context.user_args
        if not isinstance(dynamics, FiniteElementDynamicsExecutionContext):
            raise TypeError("Plastic FEM requires a dynamics execution context.")
        materials = dynamics.materials
        if materials is None:
            raise ValueError("The plastic FEM case requires committed materials.")
        deformation = (
            jnp.eye(3, dtype=displacement.dtype) + (displacement[1:] - displacement[:1]).T
        )
        response = self.response(deformation, dynamics)
        return FiniteElementAuxiliaryEvaluation(
            materials.with_trials({"plastic": response.trial_state}),
            successful=response.successful,
            admissible=response.admissible & dynamics.user_args["valid"],
        )

    def update(
        self,
        displacement: Array,
        velocity: Array,
        acceleration: Array,
        time: Array,
        step_size: Array,
        previous: MaterialTransaction | None,
        args: dict[str, Array],
    ) -> MaterialTransaction:
        del velocity, acceleration
        if previous is None:
            raise ValueError("The plastic FEM case requires committed materials.")
        dynamics = FiniteElementDynamicsExecutionContext(
            previous, time - step_size, time, step_size, args
        )
        deformation = (
            jnp.eye(3, dtype=displacement.dtype) + (displacement[1:] - displacement[:1]).T
        )
        response = self.response(deformation, dynamics)
        return previous.with_trials({"plastic": response.trial_state})


def _case() -> tuple[
    CompiledFiniteElementProblem,
    FiniteElementContactParticipant,
    FiniteElementDynamicsState,
    _Plasticity,
]:
    points = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=jnp.float64,
    )
    cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_tetrahedra(points, cells)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()
    plasticity = _Plasticity(
        FiniteStrainJ2MPMConstitutivePlan(),
        FiniteStrainJ2Parameters(0.2, 0.8, 0.002, 0.02),
    )
    form = phx.equations.FiniteElementForm(
        "plastic-contact-history",
        "u",
        (
            phx.equations.CellResidualAction(
                "u", ("u",), plasticity.residual, action_id="j2-stress"
            ),
        ),
        auxiliary_evaluator=plasticity.auxiliary,
        auxiliary_id="j2-trial",
    )
    problem = phx.equations.compile_finite_element_problem(form, discretization)
    participant = phx.applications.contact.prepare_finite_element_contact_participant(
        problem,
        inversion=phx.discretization.SimplexInversionStepPlan(cells, points),
    )
    materials = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("plastic"),
                "j2",
                plasticity.material.initialize_state((), jnp.float64),
            ),
        )
    )
    state = FiniteElementDynamicsState(
        jnp.zeros_like(points),
        points @ jnp.diag(jnp.asarray((4.0, -2.0, -2.0), dtype=points.dtype)),
        jnp.zeros_like(points),
        materials=materials,
    )
    return problem, participant, state, plasticity


def _plan(
    participant: FiniteElementContactParticipant, *, envelope_radius: float = 1.0
) -> MonolithicContactPlan:
    scene = phx.discretization.ContactParticipantScene((participant.geometry,))
    rest = scene.positions((jnp.zeros(participant.dof_space.shape, dtype=jnp.float64),))
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.01, 1.0).prepare(
        scene, rest_positions=rest
    )
    return MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        phx.discretization.SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=16,
            edge_edge_capacity=16,
            face_vertex_capacity=16,
            activation_distance=0.01,
            envelope_radius=envelope_radius,
        ),
        phx.discretization.InclusionCCDPlan(),
        geometry_owner_indices=(0,),
        fixed_geometry_states=(None,),
        policy=phx.applications.contact.ContactSolvePolicy(
            merit="residual",
            termination=phx.nonlinear.NonlinearTermination(
                absolute_residual=1e-11,
                relative_residual=1e-11,
                maximum_steps=30,
            ),
        ),
    )


def _history(state: MonolithicContactState) -> FiniteElementDynamicsState:
    history = state.histories[0]
    if not isinstance(history, FiniteElementDynamicsState):
        raise TypeError("The accepted FEM history must retain its dynamics state.")
    return history


def test_two_accepted_plastic_steps_match_standalone_fem_dynamics() -> None:
    problem, participant, initial, plasticity = _case()
    plan = _plan(participant)
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan, (initial.displacement,), (initial,)
    )
    args = {"valid": jnp.asarray(True, dtype=jnp.bool_)}
    standalone_plan = phx.applications.solid_mechanics.prepare_finite_element_dynamics(
        problem,
        initial,
        0.05,
        args=args,
        nonlinear_method=plan.policy.method,
        nonlinear_termination=plan.policy.termination,
        material_update=plasticity.update,
        material_update_id="j2-trial",
    )
    standalone = initial
    previous_alpha = 0.0
    for step in (1, 2):
        previous = state
        result = phx.applications.contact.solve_monolithic_contact_step(
            phx.applications.contact.prepare_monolithic_contact_step(
                plan, state, 0.05, args=args
            )
        )
        reference = phx.applications.solid_mechanics.solve_finite_element_dynamics_step(
            phx.applications.solid_mechanics.prepare_finite_element_dynamics_step(
                standalone_plan, standalone, 0.05, args=args
            )
        )
        assert bool(result.accepted)
        assert bool(reference.accepted)
        state = result.commit()
        standalone = reference.promote()
        assert eqx.tree_equal(result.commit(), state)
        assert eqx.tree_equal(result.rollback(), previous)
        history = _history(state)
        assert history.materials is not None
        assert standalone.materials is not None
        material = history.materials.states[0]
        alpha = float(material.committed[-1])
        assert alpha > previous_alpha
        previous_alpha = alpha
        assert material.state_version == step
        np.testing.assert_array_equal(material.trial, material.committed)
        for actual, expected in (
            (history.displacement, standalone.displacement),
            (history.velocity, standalone.velocity),
            (history.acceleration, standalone.acceleration),
        ):
            # Compare at the field scale: Newmark acceleration subtracts
            # nearly equal displacements and magnifies root roundoff by dt^-2.
            scale = np.max(np.abs(expected))
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7 * scale)
        np.testing.assert_allclose(
            material.committed,
            standalone.materials.states[0].committed,
            rtol=1e-7,
            atol=1e-9,
        )
        np.testing.assert_array_equal(history.displacement, state.coordinates[0])
        assert int(history.step) == int(state.step) == step
        assert int(history.state_version) == int(state.state_version) == step
        np.testing.assert_allclose(history.time, state.time)
        candidate = _history(result.candidate)
        assert candidate.materials is not None
        assert candidate.materials.states[0].state_version == step - 1
        np.testing.assert_array_equal(
            candidate.materials.states[0].trial, material.committed
        )
        old_history = _history(previous)
        assert old_history.materials is not None
        assert old_history.materials.states[0].state_version == step - 1
        staged = _history(result.accepted_state)
        assert staged.materials is not None
        assert staged.materials.states[0].state_version == step - 1
        np.testing.assert_array_equal(
            staged.materials.states[0].committed,
            old_history.materials.states[0].committed,
        )
        np.testing.assert_array_equal(
            staged.materials.states[0].trial, material.committed
        )


@pytest.mark.parametrize("failure", ("participant", "global-envelope"))
def test_rejected_plastic_step_preserves_committed_fem_revision(failure: str) -> None:
    _, participant, initial, _ = _case()
    plan = _plan(participant)
    valid_args = {"valid": jnp.asarray(True, dtype=jnp.bool_)}
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan, (initial.displacement,), (initial,)
    )
    first = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(
            plan, state, 0.05, args=valid_args
        )
    )
    assert bool(first.accepted)
    previous = first.commit()
    if failure == "global-envelope":
        plan = _plan(participant, envelope_radius=1e-6)
        previous = phx.applications.contact.initialize_monolithic_contact_state(
            plan, previous.coordinates, previous.histories, time=previous.time
        )
    rejected = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(
            plan,
            previous,
            0.05,
            args={"valid": jnp.asarray(failure != "participant", dtype=jnp.bool_)},
        )
    )
    assert not bool(rejected.accepted)
    assert bool(rejected.rollback_applied)
    assert eqx.tree_equal(rejected.accepted_state, previous)
    assert eqx.tree_equal(rejected.commit(), previous)
    assert eqx.tree_equal(rejected.rollback(), previous)
    if failure == "participant":
        assert int(rejected.rejection_reasons) & int(
            phx.applications.contact.ContactRejectionReason.PARTICIPANT
        )
    else:
        assert all(
            bool(evidence.successful) and bool(evidence.admissible)
            for evidence in rejected.participant_evidence
        )


def test_equilibrium_commits_material_version_without_advancing_fem_clock() -> None:
    _, participant, initial, _ = _case()
    plan = _plan(participant)
    assert initial.materials is not None
    stationary = FiniteElementDynamicsState(
        initial.displacement,
        jnp.zeros_like(initial.velocity),
        initial.acceleration,
        materials=initial.materials,
        time=2.0,
        step=7,
        state_version=9,
    )
    previous = phx.applications.contact.initialize_monolithic_contact_state(
        plan, (stationary.displacement,), (stationary,), time=2.0
    )
    result = phx.applications.contact.solve_monolithic_contact_equilibrium(
        phx.applications.contact.prepare_monolithic_contact_equilibrium(
            plan, previous, args={"valid": jnp.asarray(True, dtype=jnp.bool_)}
        )
    )
    assert bool(result.accepted)
    history = _history(result.commit())
    assert history.materials is not None
    assert history.materials.states[0].state_version == 1
    assert int(history.step) == 7
    assert int(history.state_version) == 10
    np.testing.assert_array_equal(history.time, stationary.time)
    np.testing.assert_array_equal(
        history.materials.states[0].trial, history.materials.states[0].committed
    )
