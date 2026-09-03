#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import DiscreteFieldSpace, FieldTransfer
from ....nonlinear import FixedPointIteration, NonlinearTermination
from ....solver import coupling
from ._contraction import (
    ActivationDrivenContractionPlan,
    CalciumDrivenFirstOrderContractionPlan,
    ContractionPlan,
    LandLengthVelocityContractionPlan,
)


class ElectromechanicsCadence(StrictModule, NonTrainableState):
    """Declared fixed EP/mechanics work per physical coupling window."""

    electrophysiology_substeps: int = eqx.field(static=True)
    mechanics_substeps: int = eqx.field(static=True)
    cadence_id: str = eqx.field(static=True)

    def __init__(
        self,
        electrophysiology_substeps: int,
        /,
        *,
        mechanics_substeps: int = 1,
    ):
        ep = int(electrophysiology_substeps)
        mechanics = int(mechanics_substeps)
        if ep <= 0 or mechanics <= 0:
            raise ValueError("Electromechanics cadence counts must be positive.")
        self.electrophysiology_substeps = ep
        self.mechanics_substeps = mechanics
        self.cadence_id = canonical_fingerprint(
            {
                "kind": "cardiac-electromechanics-cadence",
                "electrophysiology_substeps": ep,
                "mechanics_substeps": mechanics,
            }
        )

    @property
    def ratio(self) -> float:
        return self.electrophysiology_substeps / self.mechanics_substeps


class ActivationEPToMechanicsPort(StrictModule, NonTrainableState):
    """Dimensionless activation exchange from an EP support to a mechanics support."""

    source: DiscreteFieldSpace
    target: DiscreteFieldSpace
    transfer: FieldTransfer | None
    source_port_id: str = eqx.field(static=True)
    target_port_id: str = eqx.field(static=True)
    exchange_id: str = eqx.field(static=True)
    reference_scale: float = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True, default="activation-fraction")

    def __init__(
        self,
        source: DiscreteFieldSpace,
        target: DiscreteFieldSpace,
        /,
        *,
        transfer: FieldTransfer | None = None,
        source_port_id: str = "ep.activation.output",
        target_port_id: str = "mechanics.activation.input",
        exchange_id: str = "ep-to-mechanics-activation",
        reference_scale: float = 1.0,
    ):
        _validate_exchange_spaces(source, target, transfer)
        scale = _positive_scale(reference_scale)
        self.source = source
        self.target = target
        self.transfer = transfer
        self.source_port_id = _identifier(source_port_id, "source_port_id")
        self.target_port_id = _identifier(target_port_id, "target_port_id")
        self.exchange_id = _identifier(exchange_id, "exchange_id")
        self.reference_scale = scale


class CalciumEPToMechanicsPort(StrictModule, NonTrainableState):
    """Live cytosolic Ca exchange from a compatible ionic model to contraction."""

    source: DiscreteFieldSpace
    target: DiscreteFieldSpace
    transfer: FieldTransfer | None
    source_port_id: str = eqx.field(static=True)
    target_port_id: str = eqx.field(static=True)
    exchange_id: str = eqx.field(static=True)
    reference_scale: float = eqx.field(static=True)
    ionic_model_id: str = eqx.field(static=True)
    calcium_unit: str = eqx.field(static=True, default="mM")
    quantity_id: str = eqx.field(static=True, default="cytosolic-calcium")

    def __init__(
        self,
        source: DiscreteFieldSpace,
        target: DiscreteFieldSpace,
        /,
        *,
        ionic_model_id: str,
        transfer: FieldTransfer | None = None,
        source_port_id: str = "ep.calcium-cytosol.output",
        target_port_id: str = "mechanics.calcium-cytosol.input",
        exchange_id: str = "ep-to-mechanics-calcium",
        reference_scale: float = 1.0e-4,
        calcium_unit: str = "mM",
    ):
        _validate_exchange_spaces(source, target, transfer)
        self.source = source
        self.target = target
        self.transfer = transfer
        self.source_port_id = _identifier(source_port_id, "source_port_id")
        self.target_port_id = _identifier(target_port_id, "target_port_id")
        self.exchange_id = _identifier(exchange_id, "exchange_id")
        self.reference_scale = _positive_scale(reference_scale)
        self.ionic_model_id = _identifier(ionic_model_id, "ionic_model_id")
        self.calcium_unit = _identifier(calcium_unit, "calcium_unit")


class StretchMechanicsToEPPort(StrictModule, NonTrainableState):
    """Dimensionless fiber stretch feedback from mechanics to EP."""

    source: DiscreteFieldSpace
    target: DiscreteFieldSpace
    transfer: FieldTransfer | None
    source_port_id: str = eqx.field(static=True)
    target_port_id: str = eqx.field(static=True)
    exchange_id: str = eqx.field(static=True)
    reference_scale: float = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True, default="fiber-stretch")

    def __init__(
        self,
        source: DiscreteFieldSpace,
        target: DiscreteFieldSpace,
        /,
        *,
        transfer: FieldTransfer | None = None,
        source_port_id: str = "mechanics.fiber-stretch.output",
        target_port_id: str = "ep.fiber-stretch.input",
        exchange_id: str = "mechanics-to-ep-stretch",
        reference_scale: float = 1.0,
    ):
        _validate_exchange_spaces(source, target, transfer)
        self.source = source
        self.target = target
        self.transfer = transfer
        self.source_port_id = _identifier(source_port_id, "source_port_id")
        self.target_port_id = _identifier(target_port_id, "target_port_id")
        self.exchange_id = _identifier(exchange_id, "exchange_id")
        self.reference_scale = _positive_scale(reference_scale)


EPToMechanicsPort: TypeAlias = ActivationEPToMechanicsPort | CalciumEPToMechanicsPort


class ElectricalWindowCandidate(StrictModule):
    """One EP participant candidate and its activation or live-Ca endpoint field."""

    candidate_state: Any
    drive: Array
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array
    work: Array
    completed_substeps: Array

    def __init__(
        self,
        candidate_state: Any,
        drive: ArrayLike,
        /,
        *,
        successful: ArrayLike,
        status: ArrayLike = 0,
        residual_norm: ArrayLike = 0.0,
        iterations: ArrayLike = 0,
        work: ArrayLike = 0,
        completed_substeps: ArrayLike,
    ):
        self.candidate_state = candidate_state
        self.drive = jnp.asarray(drive)
        self.successful = _scalar(successful, bool)
        self.status = _scalar(status, jnp.int32)
        self.residual_norm = _scalar(residual_norm, self.drive.dtype)
        self.iterations = _scalar(iterations, jnp.int32)
        self.work = _scalar(work, jnp.int32)
        self.completed_substeps = _scalar(completed_substeps, jnp.int32)


class MechanicalWindowCandidate(StrictModule):
    """One mechanics participant candidate and its fiber-stretch endpoint field."""

    candidate_state: Any
    stretch: Array
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array
    work: Array
    completed_substeps: Array

    def __init__(
        self,
        candidate_state: Any,
        stretch: ArrayLike,
        /,
        *,
        successful: ArrayLike,
        status: ArrayLike = 0,
        residual_norm: ArrayLike = 0.0,
        iterations: ArrayLike = 0,
        work: ArrayLike = 0,
        completed_substeps: ArrayLike,
    ):
        self.candidate_state = candidate_state
        self.stretch = jnp.asarray(stretch)
        self.successful = _scalar(successful, bool)
        self.status = _scalar(status, jnp.int32)
        self.residual_norm = _scalar(residual_norm, self.stretch.dtype)
        self.iterations = _scalar(iterations, jnp.int32)
        self.work = _scalar(work, jnp.int32)
        self.completed_substeps = _scalar(completed_substeps, jnp.int32)


class ElectromechanicsPreparationEvidence(StrictModule, NonTrainableState):
    forward_transfer_id: str | None = eqx.field(static=True)
    backward_transfer_id: str | None = eqx.field(static=True)
    forward_quantity_id: str = eqx.field(static=True)
    electrophysiology_substeps: int = eqx.field(static=True)
    mechanics_substeps: int = eqx.field(static=True)
    fixed_topology: bool = eqx.field(static=True)
    bidirectional: bool = eqx.field(static=True)
    differentiation_policy_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class ElectromechanicsEvidence(StrictModule):
    interface_residual_norms: Array
    participant_statuses: Array
    participant_evaluations: Array
    coupling_iterations: Array
    window_statuses: Array
    cadence_ratio: Array
    declared_participant_work_per_window: Array
    work_accounting_complete: Array
    successful: Array
    rolled_back: Array
    preparation_id: str = eqx.field(static=True)
    work_semantics: str = eqx.field(
        static=True, default="participant-evaluation-counts-and-callback-work"
    )


class ElectromechanicsRun(StrictModule):
    solution: coupling.CouplingSolution
    evidence: ElectromechanicsEvidence


class PreparedElectromechanics(StrictModule, NonTrainableState):
    """Prepared native coupling problem; execution is owned by solver.coupling."""

    problem: coupling.CouplingProblem
    native: coupling.PreparedCoupling
    rollout: coupling.CouplingRolloutPlan
    cadence: ElectromechanicsCadence
    preparation: ElectromechanicsPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: coupling.CouplingProblem,
        rollout: coupling.CouplingRolloutPlan,
        cadence: ElectromechanicsCadence,
        preparation: ElectromechanicsPreparationEvidence,
        /,
    ):
        if not isinstance(problem, coupling.CouplingProblem):
            raise TypeError("Prepared electromechanics requires CouplingProblem.")
        if not isinstance(rollout, coupling.CouplingRolloutPlan):
            raise TypeError("Prepared electromechanics requires CouplingRolloutPlan.")
        self.native = problem.prepare()
        self.problem = problem
        self.rollout = rollout
        self.cadence = cadence
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiac-electromechanics",
                "problem": problem.problem_id,
                "rollout": rollout.plan_id,
                "cadence": cadence.cadence_id,
                "preparation": preparation.evidence_id,
            }
        )

    def solve(self, /) -> ElectromechanicsRun:
        solution = self.rollout.rollout(
            self.native,
            window_count=self.problem.window_count,
            window_size=self.problem.window_size,
            args=self.problem.args,
        )
        evidence = ElectromechanicsEvidence(
            solution.exchange_residual_norms,
            solution.participant_statuses,
            solution.participant_evaluations,
            solution.coupling_iterations,
            solution.statuses,
            jnp.asarray(self.cadence.ratio),
            jnp.asarray(
                (
                    self.cadence.electrophysiology_substeps,
                    self.cadence.mechanics_substeps,
                ),
                dtype=jnp.int32,
            ),
            jnp.asarray(True),
            solution.successful,
            ~solution.successful,
            self.preparation.evidence_id,
        )
        return ElectromechanicsRun(solution, evidence)

    def restart(
        self,
        checkpoint: coupling.CouplingState,
        t1: float,
        /,
        *,
        rollout: coupling.CouplingRolloutPlan | None = None,
    ) -> PreparedElectromechanics:
        if not isinstance(checkpoint, coupling.CouplingState):
            raise TypeError("Electromechanics restart requires CouplingState checkpoint.")
        if (
            checkpoint.subsystem_ids != self.native.reference_state.subsystem_ids
            or checkpoint.exchange_ids != self.native.reference_state.exchange_ids
        ):
            raise ValueError(
                "Electromechanics checkpoint graph identity is incompatible."
            )
        start = float(checkpoint.time)
        end = float(t1)
        if not isfinite(end) or end <= start:
            raise ValueError(
                "Restart end time must be finite and exceed checkpoint time."
            )
        restarted_problem = coupling.CouplingProblem(
            self.problem.graph,
            checkpoint.participant_states,
            checkpoint.exchange_values,
            self.problem.policy,
            t0=start,
            t1=end,
            window_size=self.problem.window_size,
            differentiation=self.problem.differentiation,
            args=self.problem.args,
            resources=self.problem.resources,
            problem_id=f"{self.problem.problem_id}:restart:{int(checkpoint.window_index)}",
        )
        return PreparedElectromechanics(
            restarted_problem,
            self.rollout if rollout is None else rollout,
            self.cadence,
            self.preparation,
        )


class OneWayElectromechanicsPlan(StrictModule, NonTrainableState):
    """EP→mechanics partitioned multirate coupling with no mechanics feedback."""

    forward_port: EPToMechanicsPort
    contraction_plan: ContractionPlan
    cadence: ElectromechanicsCadence
    differentiation: coupling.CouplingDifferentiationPolicy
    plan_id: str = eqx.field(static=True)
    fidelity_id: str = eqx.field(static=True, default="one-way-electromechanics")

    def __init__(
        self,
        forward_port: EPToMechanicsPort,
        contraction_plan: ContractionPlan,
        cadence: ElectromechanicsCadence,
        /,
        *,
        differentiation: coupling.CouplingDifferentiationPolicy | None = None,
    ):
        _validate_forward_contraction(forward_port, contraction_plan)
        if not isinstance(cadence, ElectromechanicsCadence):
            raise TypeError("One-way electromechanics requires ElectromechanicsCadence.")
        differentiation_ = (
            coupling.CouplingDifferentiationPolicy("none")
            if differentiation is None
            else differentiation
        )
        self.forward_port = forward_port
        self.contraction_plan = contraction_plan
        self.cadence = cadence
        self.differentiation = differentiation_
        self.plan_id = _electromechanics_plan_id(
            self.fidelity_id,
            forward_port,
            None,
            contraction_plan,
            cadence,
            differentiation_,
        )

    def prepare(
        self,
        electrophysiology_advance: Any,
        mechanics_advance: Any,
        initial_electrophysiology_state: Any,
        initial_mechanics_state: Any,
        initial_drive_on_mechanics: ArrayLike,
        /,
        *,
        t0: float,
        t1: float,
        coupling_window: float,
        args: Any = None,
        rollout: coupling.CouplingRolloutPlan | None = None,
    ) -> PreparedElectromechanics:
        if not callable(electrophysiology_advance) or not callable(mechanics_advance):
            raise TypeError("Electromechanics participant advances must be callable.")
        source_port, target_port = _forward_coupling_ports(self.forward_port)
        capabilities = _capabilities(self.differentiation)
        cadence = self.cadence

        def ep_adapter(window, state, inputs, runtime_args):
            del inputs
            result = electrophysiology_advance(
                window,
                state,
                None,
                cadence.electrophysiology_substeps,
                runtime_args,
            )
            _require_electrical_candidate(result)
            complete = result.completed_substeps == cadence.electrophysiology_substeps
            return coupling.CouplingSubsystemResult(
                result.candidate_state,
                (result.drive,),
                successful=result.successful & complete,
                status=jnp.where(complete, result.status, 6),
                residual_norm=result.residual_norm,
                iterations=result.iterations,
                work=result.work,
            )

        def mechanics_adapter(window, state, inputs, runtime_args):
            result = mechanics_advance(
                window,
                state,
                inputs[0],
                cadence.mechanics_substeps,
                runtime_args,
            )
            _require_mechanical_candidate(result)
            complete = result.completed_substeps == cadence.mechanics_substeps
            return coupling.CouplingSubsystemResult(
                result.candidate_state,
                (),
                successful=result.successful & complete,
                status=jnp.where(complete, result.status, 6),
                residual_norm=result.residual_norm,
                iterations=result.iterations,
                work=result.work,
            )

        ep = coupling.CallableCouplingSubsystem(
            ep_adapter,
            subsystem_id="electrophysiology",
            output_ports=(source_port,),
            capabilities=capabilities,
            discretization_bundle_id=self.forward_port.source.support_id,
        )
        mechanics = coupling.CallableCouplingSubsystem(
            mechanics_adapter,
            subsystem_id="mechanics",
            input_ports=(target_port,),
            capabilities=capabilities,
            discretization_bundle_id=self.forward_port.target.support_id,
        )
        exchange = _forward_exchange(self.forward_port)
        graph = coupling.CouplingGraph((ep, mechanics), (exchange,))
        policy = coupling.ExplicitCouplingPolicy(
            coupling.CouplingSweep(
                "gauss-seidel",
                subsystem_order=("electrophysiology", "mechanics"),
            )
        )
        problem = coupling.CouplingProblem(
            graph,
            (initial_electrophysiology_state, initial_mechanics_state),
            (jnp.asarray(initial_drive_on_mechanics),),
            policy,
            t0=t0,
            t1=t1,
            window_size=coupling_window,
            differentiation=self.differentiation,
            args=args,
            problem_id=self.plan_id,
        )
        return _prepared(
            problem,
            self.cadence,
            self.forward_port,
            None,
            self.differentiation,
            rollout,
        )


class BidirectionalElectromechanicsPlan(StrictModule, NonTrainableState):
    """Implicit EP↔mechanics partitioned multirate coupling with stretch feedback."""

    forward_port: EPToMechanicsPort
    backward_port: StretchMechanicsToEPPort
    contraction_plan: ContractionPlan
    cadence: ElectromechanicsCadence
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    differentiation: coupling.CouplingDifferentiationPolicy
    plan_id: str = eqx.field(static=True)
    fidelity_id: str = eqx.field(static=True, default="bidirectional-electromechanics")

    def __init__(
        self,
        forward_port: EPToMechanicsPort,
        backward_port: StretchMechanicsToEPPort,
        contraction_plan: ContractionPlan,
        cadence: ElectromechanicsCadence,
        /,
        *,
        absolute_tolerance: float = 1.0e-7,
        relative_tolerance: float = 1.0e-6,
        maximum_iterations: int = 30,
        differentiation: coupling.CouplingDifferentiationPolicy | None = None,
    ):
        _validate_forward_contraction(forward_port, contraction_plan)
        if not isinstance(backward_port, StretchMechanicsToEPPort):
            raise TypeError(
                "Bidirectional coupling requires typed stretch feedback port."
            )
        if not isinstance(cadence, ElectromechanicsCadence):
            raise TypeError("Bidirectional coupling requires ElectromechanicsCadence.")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        maximum = int(maximum_iterations)
        if (
            not isfinite(absolute)
            or not isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
            or absolute + relative <= 0.0
            or maximum <= 0
        ):
            raise ValueError("Bidirectional coupling convergence controls are invalid.")
        differentiation_ = (
            coupling.CouplingDifferentiationPolicy("none")
            if differentiation is None
            else differentiation
        )
        if differentiation_.mode == "implicit":
            raise ValueError(
                "Bidirectional fixed-point electromechanics supports only none or "
                "algorithmic differentiation."
            )
        self.forward_port = forward_port
        self.backward_port = backward_port
        self.contraction_plan = contraction_plan
        self.cadence = cadence
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.maximum_iterations = maximum
        self.differentiation = differentiation_
        self.plan_id = _electromechanics_plan_id(
            self.fidelity_id,
            forward_port,
            backward_port,
            contraction_plan,
            cadence,
            differentiation_,
        )

    def prepare(
        self,
        electrophysiology_advance: Any,
        mechanics_advance: Any,
        initial_electrophysiology_state: Any,
        initial_mechanics_state: Any,
        initial_drive_on_mechanics: ArrayLike,
        initial_stretch_on_ep: ArrayLike,
        /,
        *,
        t0: float,
        t1: float,
        coupling_window: float,
        args: Any = None,
        rollout: coupling.CouplingRolloutPlan | None = None,
    ) -> PreparedElectromechanics:
        if not callable(electrophysiology_advance) or not callable(mechanics_advance):
            raise TypeError("Electromechanics participant advances must be callable.")
        forward_source, forward_target = _forward_coupling_ports(self.forward_port)
        backward_source, backward_target = _backward_coupling_ports(self.backward_port)
        capabilities = _capabilities(self.differentiation)
        cadence = self.cadence

        def ep_adapter(window, state, inputs, runtime_args):
            result = electrophysiology_advance(
                window,
                state,
                inputs[0],
                cadence.electrophysiology_substeps,
                runtime_args,
            )
            _require_electrical_candidate(result)
            complete = result.completed_substeps == cadence.electrophysiology_substeps
            return coupling.CouplingSubsystemResult(
                result.candidate_state,
                (result.drive,),
                successful=result.successful & complete,
                status=jnp.where(complete, result.status, 6),
                residual_norm=result.residual_norm,
                iterations=result.iterations,
                work=result.work,
            )

        def mechanics_adapter(window, state, inputs, runtime_args):
            result = mechanics_advance(
                window,
                state,
                inputs[0],
                cadence.mechanics_substeps,
                runtime_args,
            )
            _require_mechanical_candidate(result)
            complete = result.completed_substeps == cadence.mechanics_substeps
            return coupling.CouplingSubsystemResult(
                result.candidate_state,
                (result.stretch,),
                successful=result.successful & complete,
                status=jnp.where(complete, result.status, 6),
                residual_norm=result.residual_norm,
                iterations=result.iterations,
                work=result.work,
            )

        ep = coupling.CallableCouplingSubsystem(
            ep_adapter,
            subsystem_id="electrophysiology",
            input_ports=(backward_target,),
            output_ports=(forward_source,),
            capabilities=capabilities,
            discretization_bundle_id=self.forward_port.source.support_id,
        )
        mechanics = coupling.CallableCouplingSubsystem(
            mechanics_adapter,
            subsystem_id="mechanics",
            input_ports=(forward_target,),
            output_ports=(backward_source,),
            capabilities=capabilities,
            discretization_bundle_id=self.forward_port.target.support_id,
        )
        graph = coupling.CouplingGraph(
            (ep, mechanics),
            (
                _forward_exchange(self.forward_port),
                _backward_exchange(self.backward_port),
            ),
        )
        policy = coupling.ImplicitCouplingPolicy(
            FixedPointIteration(),
            NonlinearTermination(
                absolute_residual=self.absolute_tolerance,
                relative_residual=self.relative_tolerance,
                maximum_steps=self.maximum_iterations,
            ),
            (
                coupling.CouplingTolerance(
                    self.forward_port.target_port_id,
                    absolute=self.absolute_tolerance,
                    relative=self.relative_tolerance,
                ),
                coupling.CouplingTolerance(
                    self.backward_port.target_port_id,
                    absolute=self.absolute_tolerance,
                    relative=self.relative_tolerance,
                ),
            ),
            fixed_point_sweep=coupling.CouplingSweep(
                "gauss-seidel",
                subsystem_order=("electrophysiology", "mechanics"),
            ),
        )
        problem = coupling.CouplingProblem(
            graph,
            (initial_electrophysiology_state, initial_mechanics_state),
            (
                jnp.asarray(initial_drive_on_mechanics),
                jnp.asarray(initial_stretch_on_ep),
            ),
            policy,
            t0=t0,
            t1=t1,
            window_size=coupling_window,
            differentiation=self.differentiation,
            args=args,
            problem_id=self.plan_id,
        )
        return _prepared(
            problem,
            self.cadence,
            self.forward_port,
            self.backward_port,
            self.differentiation,
            rollout,
        )


def _identifier(value: str, role: str) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


def _positive_scale(value: float) -> float:
    scale = float(value)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("Coupling reference scale must be finite and positive.")
    return scale


def _scalar(value: ArrayLike, dtype: Any) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != ():
        raise ValueError("Participant evidence must be scalar.")
    return result


def _validate_exchange_spaces(
    source: DiscreteFieldSpace,
    target: DiscreteFieldSpace,
    transfer: FieldTransfer | None,
) -> None:
    if not isinstance(source, DiscreteFieldSpace) or not isinstance(
        target, DiscreteFieldSpace
    ):
        raise TypeError("Typed electromechanics ports require DiscreteFieldSpace values.")
    if transfer is None:
        if source.vector_space.space_id != target.vector_space.space_id:
            raise ValueError(
                "Different electromechanics meshes require an explicit FieldTransfer."
            )
    elif not isinstance(transfer, FieldTransfer):
        raise TypeError("Electromechanics transfer must be FieldTransfer or None.")
    elif (
        transfer.source.field_space_id != source.field_space_id
        or transfer.target.field_space_id != target.field_space_id
    ):
        raise ValueError(
            "Electromechanics transfer source/target spaces are inconsistent."
        )


def _validate_forward_contraction(port: EPToMechanicsPort, plan: ContractionPlan) -> None:
    if isinstance(port, ActivationEPToMechanicsPort):
        if not isinstance(plan, ActivationDrivenContractionPlan):
            raise TypeError(
                "Activation EP ports require ActivationDrivenContractionPlan."
            )
    elif isinstance(port, CalciumEPToMechanicsPort):
        if not isinstance(
            plan,
            (CalciumDrivenFirstOrderContractionPlan, LandLengthVelocityContractionPlan),
        ):
            raise TypeError("Calcium EP ports require a calcium-driven contraction plan.")
        if (
            plan.ionic_model_id != port.ionic_model_id
            or plan.calcium_unit != port.calcium_unit
        ):
            raise ValueError("Contraction plan is incompatible with the live-Ca EP port.")
    else:
        raise TypeError("Unknown typed EP-to-mechanics port.")


def _generic_port(
    port_id: str,
    direction: str,
    field: DiscreteFieldSpace,
    reference_scale: float,
) -> coupling.CouplingPort:
    return coupling.CouplingPort(
        port_id,
        direction,
        field.vector_space,
        field_space=field,
        reference_scale=reference_scale,
    )


def _forward_coupling_ports(
    port: EPToMechanicsPort,
) -> tuple[coupling.CouplingPort, coupling.CouplingPort]:
    return (
        _generic_port(port.source_port_id, "output", port.source, port.reference_scale),
        _generic_port(port.target_port_id, "input", port.target, port.reference_scale),
    )


def _backward_coupling_ports(
    port: StretchMechanicsToEPPort,
) -> tuple[coupling.CouplingPort, coupling.CouplingPort]:
    return (
        _generic_port(port.source_port_id, "output", port.source, port.reference_scale),
        _generic_port(port.target_port_id, "input", port.target, port.reference_scale),
    )


def _forward_exchange(port: EPToMechanicsPort) -> coupling.CouplingExchange:
    return coupling.CouplingExchange(
        port.exchange_id,
        port.source_port_id,
        port.target_port_id,
        transfer=port.transfer,
        requirement=coupling.CouplingTransferRequirement(
            constant_preserving=True,
            positivity_preserving=True,
        ),
    )


def _backward_exchange(port: StretchMechanicsToEPPort) -> coupling.CouplingExchange:
    return coupling.CouplingExchange(
        port.exchange_id,
        port.source_port_id,
        port.target_port_id,
        transfer=port.transfer,
        requirement=coupling.CouplingTransferRequirement(constant_preserving=True),
    )


def _capabilities(
    differentiation: coupling.CouplingDifferentiationPolicy,
) -> coupling.CouplingSubsystemCapabilities:
    return coupling.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=differentiation.mode != "none",
        deterministic_replay=True,
        fixed_topology=True,
        counts_complete=True,
    )


def _require_electrical_candidate(value: Any) -> None:
    if not isinstance(value, ElectricalWindowCandidate):
        raise TypeError("EP callback must return ElectricalWindowCandidate.")


def _require_mechanical_candidate(value: Any) -> None:
    if not isinstance(value, MechanicalWindowCandidate):
        raise TypeError("Mechanics callback must return MechanicalWindowCandidate.")


def _electromechanics_plan_id(
    fidelity: str,
    forward: EPToMechanicsPort,
    backward: StretchMechanicsToEPPort | None,
    contraction: ContractionPlan,
    cadence: ElectromechanicsCadence,
    differentiation: coupling.CouplingDifferentiationPolicy,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "cardiac-electromechanics",
            "fidelity": fidelity,
            "forward_exchange": forward.exchange_id,
            "forward_transfer": None
            if forward.transfer is None
            else forward.transfer.transfer_id,
            "backward_exchange": None if backward is None else backward.exchange_id,
            "backward_transfer": (
                None
                if backward is None or backward.transfer is None
                else backward.transfer.transfer_id
            ),
            "contraction": contraction.plan_id,
            "cadence": cadence.cadence_id,
            "differentiation": differentiation.policy_id,
        }
    )


def _prepared(
    problem: coupling.CouplingProblem,
    cadence: ElectromechanicsCadence,
    forward: EPToMechanicsPort,
    backward: StretchMechanicsToEPPort | None,
    differentiation: coupling.CouplingDifferentiationPolicy,
    rollout: coupling.CouplingRolloutPlan | None,
) -> PreparedElectromechanics:
    evidence_id = canonical_fingerprint(
        {
            "kind": "cardiac-electromechanics-preparation",
            "problem": problem.problem_id,
            "forward_transfer": None
            if forward.transfer is None
            else forward.transfer.transfer_id,
            "backward_transfer": (
                None
                if backward is None or backward.transfer is None
                else backward.transfer.transfer_id
            ),
            "cadence": cadence.cadence_id,
            "fixed_topology": True,
            "differentiation": differentiation.policy_id,
        }
    )
    evidence = ElectromechanicsPreparationEvidence(
        None if forward.transfer is None else forward.transfer.transfer_id,
        None
        if backward is None or backward.transfer is None
        else backward.transfer.transfer_id,
        forward.quantity_id,
        cadence.electrophysiology_substeps,
        cadence.mechanics_substeps,
        True,
        backward is not None,
        differentiation.policy_id,
        evidence_id,
    )
    rollout_ = (
        coupling.CouplingRolloutPlan(retention="trajectory")
        if rollout is None
        else rollout
    )
    return PreparedElectromechanics(problem, rollout_, cadence, evidence)


__all__ = [
    "ActivationEPToMechanicsPort",
    "BidirectionalElectromechanicsPlan",
    "CalciumEPToMechanicsPort",
    "ElectricalWindowCandidate",
    "ElectromechanicsCadence",
    "ElectromechanicsEvidence",
    "ElectromechanicsPreparationEvidence",
    "ElectromechanicsRun",
    "EPToMechanicsPort",
    "MechanicalWindowCandidate",
    "OneWayElectromechanicsPlan",
    "PreparedElectromechanics",
    "StretchMechanicsToEPPort",
]
