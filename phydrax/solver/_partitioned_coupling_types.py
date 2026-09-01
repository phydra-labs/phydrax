#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscreteFieldSpace, FieldTransfer
from ..dynamics import TimeGrid
from ..linalg import AbstractVectorSpace
from ..nonlinear import (
    AbstractNonlinearMethod,
    FixedPointIteration,
    ImplicitRootDerivativePolicy,
    NonlinearTermination,
)


CouplingDirection: TypeAlias = Literal["input", "output"]
CouplingSweepKind: TypeAlias = Literal["jacobi", "gauss-seidel"]
CouplingDifferentiationMode: TypeAlias = Literal["none", "algorithmic", "implicit"]


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


def _scalar(value: Any, role: str, /, *, dtype: Any | None = None) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != ():
        raise ValueError(f"{role} must be a scalar array.")
    return result


def _array_tree(value: Any, role: str, /) -> Any:
    leaves, treedef = jax.tree.flatten(value)
    if not leaves:
        raise ValueError(f"{role} must contain at least one array leaf.")
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    return jax.tree.unflatten(treedef, arrays)


def _termination_payload(termination: NonlinearTermination, /) -> dict[str, Any]:
    return {
        "absolute_residual": termination.absolute_residual,
        "relative_residual": termination.relative_residual,
        "absolute_step": termination.absolute_step,
        "relative_step": termination.relative_step,
        "maximum_steps": termination.maximum_steps,
        "maximum_evaluations": termination.maximum_evaluations,
        "maximum_linear_iterations": termination.maximum_linear_iterations,
        "divergence_factor": termination.divergence_factor,
    }


class CouplingStatus(IntEnum):
    """Terminal status of one transactional coupling window."""

    SUCCESS = 0
    PARTICIPANT_FAILURE = 1
    NONFINITE_EVALUATION = 2
    NONLINEAR_FAILURE = 3
    WORK_EXHAUSTED = 4
    CERTIFICATION_FAILURE = 5


_COUPLING_STATUS_MESSAGES = {
    CouplingStatus.SUCCESS: "coupling window completed successfully",
    CouplingStatus.PARTICIPANT_FAILURE: "a coupling participant failed",
    CouplingStatus.NONFINITE_EVALUATION: "coupling evaluation produced non-finite data",
    CouplingStatus.NONLINEAR_FAILURE: "implicit coupling solve failed",
    CouplingStatus.WORK_EXHAUSTED: "implicit coupling exhausted its work limit",
    CouplingStatus.CERTIFICATION_FAILURE: (
        "candidate coupling state failed physical interface certification"
    ),
}


def coupling_status_message(status: int | CouplingStatus, /) -> str:
    return _COUPLING_STATUS_MESSAGES[CouplingStatus(int(status))]


class CouplingSubsystemCapabilities(StrictModule):
    """Static transformation and execution capabilities of one participant."""

    jit: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    deterministic_replay: bool = eqx.field(static=True)
    fixed_topology: bool = eqx.field(static=True)
    supports_endpoint: bool = eqx.field(static=True)
    supports_waveform: bool = eqx.field(static=True)
    counts_complete: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        jit: bool,
        differentiable: bool,
        deterministic_replay: bool,
        fixed_topology: bool,
        supports_endpoint: bool = True,
        supports_waveform: bool = False,
        counts_complete: bool = True,
    ):
        self.jit = bool(jit)
        self.differentiable = bool(differentiable)
        self.deterministic_replay = bool(deterministic_replay)
        self.fixed_topology = bool(fixed_topology)
        self.supports_endpoint = bool(supports_endpoint)
        self.supports_waveform = bool(supports_waveform)
        self.counts_complete = bool(counts_complete)


class CouplingPort(StrictModule, NonTrainableState):
    """One exact participant input or output space."""

    space: AbstractVectorSpace
    field_space: DiscreteFieldSpace | None
    sample_grid: TimeGrid | None
    reference_scale: float = eqx.field(static=True)
    direction: CouplingDirection = eqx.field(static=True)
    temporal_interpolation: Literal["held", "linear"] = eqx.field(static=True)
    port_id: str = eqx.field(static=True)

    def __init__(
        self,
        port_id: str,
        direction: CouplingDirection,
        space: AbstractVectorSpace,
        /,
        *,
        field_space: DiscreteFieldSpace | None = None,
        sample_grid: TimeGrid | None = None,
        temporal_interpolation: Literal["held", "linear"] = "linear",
        reference_scale: float,
    ):
        if direction not in ("input", "output"):
            raise ValueError("Coupling port direction must be 'input' or 'output'.")
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("Coupling port space must be an AbstractVectorSpace.")
        if field_space is not None:
            if not isinstance(field_space, DiscreteFieldSpace):
                raise TypeError(
                    "Coupling port field_space must be a DiscreteFieldSpace or None."
                )
            if field_space.vector_space.space_id != space.space_id:
                raise ValueError(
                    "Coupling field_space vector space must equal the declared port space."
                )
        if sample_grid is not None:
            if not isinstance(sample_grid, TimeGrid):
                raise TypeError("Coupling port sample_grid must be a TimeGrid or None.")
            if float(sample_grid.times[0]) != 0.0 or float(sample_grid.times[-1]) <= 0.0:
                raise ValueError(
                    "Coupling port sample_grid must span relative time zero to a "
                    "positive endpoint."
                )
        if temporal_interpolation not in ("held", "linear"):
            raise ValueError(
                "Coupling temporal_interpolation must be 'held' or 'linear'."
            )
        scale = float(reference_scale)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("Coupling port reference_scale must be finite and positive.")
        self.space = space
        self.field_space = field_space
        self.sample_grid = sample_grid
        self.reference_scale = scale
        self.direction = direction
        self.temporal_interpolation = temporal_interpolation
        self.port_id = _identifier(port_id, "Coupling port_id")


class CouplingTransferRequirement(StrictModule, NonTrainableState):
    """Evidence a field exchange requires from its supplied transfer."""

    conservative: bool = eqx.field(static=True)
    constant_preserving: bool = eqx.field(static=True)
    positivity_preserving: bool = eqx.field(static=True)
    adjoint_paired: bool = eqx.field(static=True)
    minimum_exactness_degree: int | None = eqx.field(static=True)
    requirement_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        conservative: bool = False,
        constant_preserving: bool = False,
        positivity_preserving: bool = False,
        adjoint_paired: bool = False,
        minimum_exactness_degree: int | None = None,
    ):
        degree = (
            None if minimum_exactness_degree is None else int(minimum_exactness_degree)
        )
        if degree is not None and degree < 0:
            raise ValueError("minimum_exactness_degree must be non-negative or None.")
        self.conservative = bool(conservative)
        self.constant_preserving = bool(constant_preserving)
        self.positivity_preserving = bool(positivity_preserving)
        self.adjoint_paired = bool(adjoint_paired)
        self.minimum_exactness_degree = degree
        self.requirement_id = canonical_fingerprint(
            {
                "kind": "coupling-transfer-requirement",
                "conservative": self.conservative,
                "constant_preserving": self.constant_preserving,
                "positivity_preserving": self.positivity_preserving,
                "adjoint_paired": self.adjoint_paired,
                "minimum_exactness_degree": degree,
            }
        )


class CouplingExchange(StrictModule, NonTrainableState):
    """One directed output-to-input exchange with no implicit mapping fallback."""

    transfer: FieldTransfer | None
    source_port_id: str = eqx.field(static=True)
    target_port_id: str = eqx.field(static=True)
    use_adjoint: bool = eqx.field(static=True)
    requirement: CouplingTransferRequirement | None
    exchange_id: str = eqx.field(static=True)

    def __init__(
        self,
        exchange_id: str,
        source_port_id: str,
        target_port_id: str,
        /,
        *,
        transfer: FieldTransfer | None = None,
        use_adjoint: bool = False,
        requirement: CouplingTransferRequirement | None = None,
    ):
        if transfer is not None and not isinstance(transfer, FieldTransfer):
            raise TypeError("Coupling exchange transfer must be a FieldTransfer or None.")
        if transfer is None and use_adjoint:
            raise ValueError(
                "A direct coupling exchange cannot request an adjoint action."
            )
        if requirement is not None and not isinstance(
            requirement, CouplingTransferRequirement
        ):
            raise TypeError(
                "Coupling exchange requirement must be a CouplingTransferRequirement or None."
            )
        self.transfer = transfer
        self.source_port_id = _identifier(source_port_id, "source_port_id")
        self.target_port_id = _identifier(target_port_id, "target_port_id")
        self.use_adjoint = bool(use_adjoint)
        self.requirement = requirement
        self.exchange_id = _identifier(exchange_id, "Coupling exchange_id")


class CouplingTolerance(StrictModule, NonTrainableState):
    """Physical certification threshold for one target input port."""

    absolute: float = eqx.field(static=True)
    relative: float = eqx.field(static=True)
    port_id: str = eqx.field(static=True)
    tolerance_id: str = eqx.field(static=True)

    def __init__(
        self,
        port_id: str,
        /,
        *,
        absolute: float,
        relative: float = 0.0,
    ):
        absolute_ = float(absolute)
        relative_ = float(relative)
        if (
            not isfinite(absolute_)
            or not isfinite(relative_)
            or absolute_ < 0.0
            or relative_ < 0.0
        ):
            raise ValueError("Coupling tolerances must be finite and non-negative.")
        if absolute_ == 0.0 and relative_ == 0.0:
            raise ValueError("At least one coupling tolerance must be positive.")
        identifier = _identifier(port_id, "Coupling tolerance port_id")
        self.absolute = absolute_
        self.relative = relative_
        self.port_id = identifier
        self.tolerance_id = canonical_fingerprint(
            {
                "kind": "coupling-tolerance",
                "port": identifier,
                "absolute": absolute_,
                "relative": relative_,
            }
        )


class CouplingSweep(StrictModule, NonTrainableState):
    """Numerical block-Jacobi or ordered block-Gauss--Seidel semantics."""

    kind: CouplingSweepKind = eqx.field(static=True)
    subsystem_order: tuple[str, ...] = eqx.field(static=True)
    sweep_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CouplingSweepKind,
        /,
        *,
        subsystem_order: tuple[str, ...] = (),
    ):
        if kind not in ("jacobi", "gauss-seidel"):
            raise ValueError("Coupling sweep kind must be 'jacobi' or 'gauss-seidel'.")
        order = tuple(
            _identifier(value, "sweep subsystem ID") for value in subsystem_order
        )
        if len(set(order)) != len(order):
            raise ValueError("Coupling sweep subsystem IDs must be unique.")
        if kind == "jacobi" and order:
            raise ValueError("A Jacobi coupling sweep must not declare an order.")
        if kind == "gauss-seidel" and not order:
            raise ValueError("A Gauss--Seidel coupling sweep requires an explicit order.")
        self.kind = kind
        self.subsystem_order = order
        self.sweep_id = canonical_fingerprint(
            {"kind": "coupling-sweep", "sweep": kind, "order": list(order)}
        )


class AbstractCouplingPolicy(StrictModule, NonTrainableState):
    policy_id: AbstractAttribute[str]


class ExplicitCouplingPolicy(AbstractCouplingPolicy):
    """Exactly one declared partitioned sweep per physical time window."""

    sweep: CouplingSweep
    policy_id: str = eqx.field(static=True)

    def __init__(self, sweep: CouplingSweep, /):
        if not isinstance(sweep, CouplingSweep):
            raise TypeError("Explicit coupling requires a CouplingSweep.")
        self.sweep = sweep
        self.policy_id = canonical_fingerprint(
            {"kind": "explicit-coupling-policy", "sweep": sweep.sweep_id}
        )


class ImplicitCouplingPolicy(AbstractCouplingPolicy):
    """Bounded fixed-point or general-root interface solve."""

    method: FixedPointIteration | AbstractNonlinearMethod
    fixed_point_sweep: CouplingSweep | None
    termination: NonlinearTermination
    tolerances: tuple[CouplingTolerance, ...]
    derivative_policy: ImplicitRootDerivativePolicy | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: FixedPointIteration | AbstractNonlinearMethod,
        termination: NonlinearTermination,
        tolerances: tuple[CouplingTolerance, ...],
        /,
        *,
        fixed_point_sweep: CouplingSweep | None = None,
        derivative_policy: ImplicitRootDerivativePolicy | None = None,
    ):
        if not isinstance(method, (FixedPointIteration, AbstractNonlinearMethod)):
            raise TypeError(
                "Implicit coupling method must be FixedPointIteration or "
                "AbstractNonlinearMethod."
            )
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("Implicit coupling termination must be NonlinearTermination.")
        tolerances_ = tuple(tolerances)
        if not tolerances_ or any(
            not isinstance(value, CouplingTolerance) for value in tolerances_
        ):
            raise TypeError(
                "Implicit coupling tolerances must contain CouplingTolerance values."
            )
        port_ids = tuple(value.port_id for value in tolerances_)
        if len(set(port_ids)) != len(port_ids):
            raise ValueError("Implicit coupling tolerance port IDs must be unique.")
        if isinstance(method, FixedPointIteration):
            if not isinstance(fixed_point_sweep, CouplingSweep):
                raise TypeError(
                    "Fixed-point implicit coupling requires fixed_point_sweep."
                )
            if derivative_policy is not None:
                raise ValueError(
                    "Fixed-point implicit coupling does not expose implicit derivatives."
                )
        elif fixed_point_sweep is not None:
            raise ValueError(
                "General-root implicit coupling uses the simultaneous physical residual "
                "and must not declare fixed_point_sweep."
            )
        if derivative_policy is not None and not isinstance(
            derivative_policy, ImplicitRootDerivativePolicy
        ):
            raise TypeError(
                "derivative_policy must be ImplicitRootDerivativePolicy or None."
            )
        self.method = method
        self.fixed_point_sweep = fixed_point_sweep
        self.termination = termination
        self.tolerances = tolerances_
        self.derivative_policy = derivative_policy
        self.policy_id = canonical_fingerprint(
            {
                "kind": "implicit-coupling-policy",
                "method": method.method_id,
                "fixed_point_sweep": (
                    None if fixed_point_sweep is None else fixed_point_sweep.sweep_id
                ),
                "termination": _termination_payload(termination),
                "tolerances": [value.tolerance_id for value in tolerances_],
                "derivative_policy": (
                    None
                    if derivative_policy is None
                    else {
                        "tangent": (
                            None
                            if derivative_policy.tangent_linear_policy is None
                            else canonical_fingerprint(
                                {
                                    "kind": "linear-solve-policy",
                                    "representation": repr(
                                        derivative_policy.tangent_linear_policy
                                    ),
                                }
                            )
                        ),
                        "adjoint": (
                            None
                            if derivative_policy.adjoint_linear_policy is None
                            else canonical_fingerprint(
                                {
                                    "kind": "linear-solve-policy",
                                    "representation": repr(
                                        derivative_policy.adjoint_linear_policy
                                    ),
                                }
                            )
                        ),
                    }
                ),
            }
        )


class CouplingDifferentiationPolicy(StrictModule, NonTrainableState):
    """Explicit derivative semantics for one coupled solution map."""

    mode: CouplingDifferentiationMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, mode: CouplingDifferentiationMode = "none", /):
        if mode not in ("none", "algorithmic", "implicit"):
            raise ValueError(
                "Coupling differentiation mode must be 'none', 'algorithmic', or "
                "'implicit'."
            )
        self.mode = mode
        self.policy_id = f"coupling-differentiation:{mode}"


class CouplingWindow(StrictModule, NonTrainableState):
    """One fixed physical coupling interval."""

    index: Array
    start: Array
    end: Array

    def __init__(self, index: Any, start: Any, end: Any, /):
        start_ = _scalar(start, "Coupling window start")
        end_ = _scalar(end, "Coupling window end", dtype=start_.dtype)
        index_ = _scalar(index, "Coupling window index", dtype=jnp.int32)
        start_ = eqx.error_if(
            start_,
            ~jnp.isfinite(start_) | ~jnp.isfinite(end_) | (end_ <= start_),
            "Coupling window requires finite end > start.",
        )
        self.index = index_
        self.start = start_
        self.end = end_

    @property
    def size(self) -> Array:
        return self.end - self.start


class CouplingSubsystemResult(StrictModule):
    """One participant candidate and endpoint outputs for a frozen window."""

    candidate_state: Any
    outputs: tuple[Any, ...]
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array
    work: Array
    auxiliary: Any

    def __init__(
        self,
        candidate_state: Any,
        outputs: tuple[Any, ...],
        /,
        *,
        successful: Any,
        status: Any,
        residual_norm: Any = 0.0,
        iterations: Any = 0,
        work: Any = 0,
        auxiliary: Any = None,
    ):
        self.candidate_state = _array_tree(candidate_state, "candidate_state")
        self.outputs = tuple(outputs)
        self.successful = _scalar(successful, "participant successful", dtype=bool)
        self.status = _scalar(status, "participant status", dtype=jnp.int32)
        self.residual_norm = _scalar(residual_norm, "participant residual_norm")
        self.iterations = _scalar(iterations, "participant iterations", dtype=jnp.int32)
        self.work = _scalar(work, "participant work", dtype=jnp.int32)
        self.auxiliary = auxiliary


class AbstractCouplingSubsystem(StrictModule, NonTrainableState):
    """Pure prepared subsystem map over one coupling window."""

    subsystem_id: AbstractAttribute[str]
    input_ports: AbstractAttribute[tuple[CouplingPort, ...]]
    output_ports: AbstractAttribute[tuple[CouplingPort, ...]]
    capabilities: AbstractAttribute[CouplingSubsystemCapabilities]
    discretization_bundle_id: AbstractAttribute[str | None]

    @abc.abstractmethod
    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        raise NotImplementedError


class CallableCouplingSubsystem(AbstractCouplingSubsystem):
    """Explicit-ID adapter for one pure participant window callback."""

    advance: Any
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        advance: Any,
        /,
        *,
        subsystem_id: str,
        input_ports: tuple[CouplingPort, ...] = (),
        output_ports: tuple[CouplingPort, ...] = (),
        capabilities: CouplingSubsystemCapabilities,
        discretization_bundle_id: str | None = None,
    ):
        if not callable(advance):
            raise TypeError("Callable coupling subsystem advance must be callable.")
        inputs = tuple(input_ports)
        outputs = tuple(output_ports)
        if any(not isinstance(port, CouplingPort) for port in (*inputs, *outputs)):
            raise TypeError("Subsystem ports must contain CouplingPort values.")
        if any(port.direction != "input" for port in inputs):
            raise ValueError("Subsystem input_ports must all have direction='input'.")
        if any(port.direction != "output" for port in outputs):
            raise ValueError("Subsystem output_ports must all have direction='output'.")
        port_ids = tuple(port.port_id for port in (*inputs, *outputs))
        if len(set(port_ids)) != len(port_ids):
            raise ValueError("Subsystem port IDs must be unique.")
        if not isinstance(capabilities, CouplingSubsystemCapabilities):
            raise TypeError("capabilities must be CouplingSubsystemCapabilities.")
        bundle_id = (
            None
            if discretization_bundle_id is None
            else _identifier(discretization_bundle_id, "discretization_bundle_id")
        )
        self.advance = advance
        self.input_ports = inputs
        self.output_ports = outputs
        self.capabilities = capabilities
        self.subsystem_id = _identifier(subsystem_id, "Coupling subsystem_id")
        self.discretization_bundle_id = bundle_id

    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        result = self.advance(window, start_state, inputs, args)
        if not isinstance(result, CouplingSubsystemResult):
            raise TypeError(
                "Coupling participant callback must return CouplingSubsystemResult."
            )
        return result


class CouplingState(StrictModule):
    """Accepted participant states and target exchange values at one window boundary."""

    participant_states: tuple[Any, ...]
    exchange_values: tuple[Any, ...]
    time: Array
    window_index: Array
    subsystem_ids: tuple[str, ...] = eqx.field(static=True)
    exchange_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        participant_states: tuple[Any, ...],
        exchange_values: tuple[Any, ...],
        time: Any,
        window_index: Any,
        /,
        *,
        subsystem_ids: tuple[str, ...],
        exchange_ids: tuple[str, ...],
    ):
        states = tuple(
            _array_tree(value, f"participant_states[{index}]")
            for index, value in enumerate(participant_states)
        )
        values = tuple(
            _array_tree(value, f"exchange_values[{index}]")
            for index, value in enumerate(exchange_values)
        )
        subsystem_ids_ = tuple(subsystem_ids)
        exchange_ids_ = tuple(exchange_ids)
        if len(states) != len(subsystem_ids_):
            raise ValueError("One participant state is required per subsystem ID.")
        if len(values) != len(exchange_ids_):
            raise ValueError("One target value is required per exchange ID.")
        self.participant_states = states
        self.exchange_values = values
        self.time = _scalar(time, "Coupling state time")
        self.window_index = _scalar(
            window_index, "Coupling state window_index", dtype=jnp.int32
        )
        self.subsystem_ids = subsystem_ids_
        self.exchange_ids = exchange_ids_


class CouplingWindowDiagnostics(StrictModule):
    """Physical block residuals and exact participant work for one window."""

    exchange_residual_norms: Array
    normalized_exchange_residual_norms: Array
    exchange_thresholds: Array
    exchange_certified: Array
    participant_statuses: Array
    participant_residual_norms: Array
    participant_iterations: Array
    participant_work: Array
    participant_evaluations: Array
    transfer_applications: Array
    coupling_iterations: Array
    nonlinear_residual_evaluations: Array
    counts_complete: bool = eqx.field(static=True)


class CouplingProvenance(StrictModule, NonTrainableState):
    problem_id: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    differentiation_policy_id: str = eqx.field(static=True)
    numeric_version: Array


class CouplingWindowResult(StrictModule):
    """One coupling candidate, atomic accepted state, and retained evidence."""

    candidate_state: CouplingState
    accepted_state: CouplingState
    successful: Array
    converged: Array
    status: Array
    nonlinear_status: Array
    diagnostics: CouplingWindowDiagnostics
    provenance: CouplingProvenance


__all__ = [
    "AbstractCouplingPolicy",
    "AbstractCouplingSubsystem",
    "CallableCouplingSubsystem",
    "CouplingDifferentiationMode",
    "CouplingDifferentiationPolicy",
    "CouplingDirection",
    "CouplingExchange",
    "CouplingPort",
    "CouplingProvenance",
    "CouplingState",
    "CouplingStatus",
    "CouplingSubsystemCapabilities",
    "CouplingSubsystemResult",
    "CouplingSweep",
    "CouplingSweepKind",
    "CouplingTolerance",
    "CouplingTransferRequirement",
    "CouplingWindow",
    "CouplingWindowDiagnostics",
    "CouplingWindowResult",
    "ExplicitCouplingPolicy",
    "ImplicitCouplingPolicy",
    "coupling_status_message",
]
