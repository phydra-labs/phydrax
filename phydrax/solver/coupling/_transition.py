#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Accepted discrete transitions and observation ports of a coupled transient.

A prepared coupled transient (``prepare_coupled_transient``) is integrated
between two physical times by one native DAE solve prepared once on a template
window; the window's save times are rebound to each ``[source, target]``, the
pattern of the partitioned DAE participant. The step certifies the original
coupled transient rows at every sample and is accepted only when the native
solve succeeded and the certificate holds; a failed step keeps its candidate as
evidence and rolls back to the source state. It is a Markov map of the native
transient coordinates: every window starts the native method from its own
start state, with no history carried across windows.

The runtime DAE arguments are the values of the prepared problem's refresh
parameter bindings (P6), keyed by binding ID; one binding may be the declared
control, supplied as the control input. The transient's ``arguments`` callable
receives this mapping and binds it with ``PreparedCoupledProblem.bind_arguments``.

``discrete_system`` exposes the step through the ``DiscreteSystem`` transition
ABI (control, linearization, and MPC consumers); ``CoupledObservationPort``
evaluates the P6 observation bindings of the prepared problem on a transient
state and exposes them through the control output and state-space location
ABIs. The stochastic kernel ``phydrax.stochastic.CoupledTransitionKernel`` adds
declared process noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._admissibility import guard_derivative_validity, refuse_derivative_dependencies
from ..._fingerprint import canonical_fingerprint
from ..._observation_covariance import DiagonalCovarianceAction
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_finite_float
from ...dynamics import (
    DiscreteStepContext,
    DiscreteSystem,
    InputLayout,
    StateLayout,
    TimeGrid,
)
from ...dynamics._system import DiscreteTransitionResult
from ...measurement import PreparedQuantityField
from ...observation import CholeskyCovarianceAction, MeasurementComparisonPlan
from ...typing import checked
from .._differential_algebraic import (
    DAESolvePolicy,
    prepare_dae,
    PreparedDAESolve,
    solve_dae,
)
from ._observations import PreparedFluxObservation
from ._parameters import ParameterBinding
from ._transient import _certificate, NestedBlocks, PreparedCoupledTransient


CoupledTransitionStatus: TypeAlias = Literal[
    "success", "native-failure", "certificate-failure", "nonfinite"
]
COUPLED_TRANSITION_SUCCESS = 0
COUPLED_TRANSITION_NATIVE_FAILURE = 1
COUPLED_TRANSITION_CERTIFICATE_FAILURE = 2
COUPLED_TRANSITION_NONFINITE = 3

type ParameterValues = Mapping[str, ArrayLike]


def coupled_transition_status_name(value: int, /) -> CoupledTransitionStatus:
    """Name of one ``CoupledTransitionStep.status`` code."""
    match value:
        case 0:
            return "success"
        case 1:
            return "native-failure"
        case 2:
            return "certificate-failure"
        case 3:
            return "nonfinite"
        case _:
            raise ValueError(f"Unknown coupled transition status {value!r}.")


@final
class CoupledTransitionStep(StrictModule):
    """One attempted step: candidate, accepted state, and its evidence.

    ``accepted`` is the candidate when ``successful`` and the unchanged source
    state otherwise. ``residual_ratio`` is the largest certified owner-row
    defect relative to its term scale over the window samples;
    ``native_status`` is the native DAE termination status.
    """

    candidate: Array
    accepted: Array
    successful: Array
    status: Array
    native_status: Array
    residual_ratio: Array


def _refresh_bindings(
    transient: PreparedCoupledTransient, /
) -> tuple[ParameterBinding, ...]:
    parameters = transient.prepared.parameters
    return tuple(
        binding
        for binding in parameters.bindings
        if binding.binding_id not in parameters.fixed_ids
    )


def _control_binding(
    bindings: tuple[ParameterBinding, ...], control: str | None, /
) -> ParameterBinding | None:
    if control is None:
        return None
    for binding in bindings:
        if binding.binding_id == control:
            return binding
    raise ValueError(
        f"Control {control!r} is not a refresh parameter binding of the coupled "
        f"problem; refresh bindings are {sorted(b.binding_id for b in bindings)}."
    )


_INCIDENCE_PROBES = 3


def _require_control_incidence(
    transient: PreparedCoupledTransient,
    control: str,
    parameters: dict[str, Array],
    window: float,
    /,
) -> None:
    """Refuse a control that does not enter the coupled transient rows.

    Incidence is structural: a component enters when the rows depend on it at
    some point. It is probed at the origin and at fixed pseudo-random states,
    rates, and times within one window, so controls entering through products
    with the state or time (``u z``, ``u t``, ``u sin(w t)``) are not mistaken
    for silent ones. A non-finite probe proves nothing and is ignored.
    """
    space = transient.state_space
    origin = space.flatten(space.zeros())
    state_key, rate_key, time_key = jax.random.split(jax.random.key(0), 3)
    shape = (_INCIDENCE_PROBES, origin.size)
    states = jnp.concatenate(
        (origin[None], jax.random.normal(state_key, shape, dtype=origin.dtype))
    )
    rates = jnp.concatenate(
        (origin[None], jax.random.normal(rate_key, shape, dtype=origin.dtype))
    )
    times = window * jnp.concatenate(
        (
            jnp.zeros((1,), dtype=origin.dtype),
            jax.random.uniform(time_key, (_INCIDENCE_PROBES,), dtype=origin.dtype),
        )
    )
    value = parameters[control]
    basis = jnp.eye(value.size, dtype=value.dtype).reshape((value.size, *value.shape))

    def probe(time: Array, state: Array, rate: Array) -> Array:
        def rows(control_value: Array) -> Array:
            return transient.native_state(
                transient.residual(
                    time,
                    space.unflatten(state),
                    space.unflatten(rate),
                    {**parameters, control: control_value},
                )
            )

        return jax.vmap(lambda tangent: jax.jvp(rows, (value,), (tangent,))[1])(basis)

    incidence = np.asarray(jax.vmap(probe)(times, states, rates))
    # Host admission boundary: the declaration is checked once at preparation.
    magnitude = np.where(np.isfinite(incidence), np.abs(incidence), 0.0)
    columns = np.max(magnitude, axis=(0, 2))
    silent = np.flatnonzero(~(columns > 0.0))
    if silent.size:
        raise ValueError(
            f"Control {control!r} components {silent.tolist()} do not enter the "
            "coupled transient rows at any probed state, rate, or time; bind the "
            "parameter to its owners' runtime inputs through "
            "PreparedCoupledProblem.bind_arguments."
        )


@final
class PreparedCoupledTransition(StrictModule, NonTrainableState):
    """A coupled transient prepared as a native discrete transition.

    ``solve`` is the native DAE solve prepared once on a template window of
    ``substeps`` uniform steps; ``normalized_times`` are its save times on
    ``[0, 1]``. ``parameter_ids`` are the refresh parameter bindings supplied
    at every step (``control`` among them when declared). Derivatives flow
    through the native solve with respect to the state and the parameter
    values; derivatives through the step times are refused, and derivatives of
    a non-accepted step are NaN. The prepared transition is fixed structure
    (``NonTrainableState``): trainable values enter only as refresh parameter
    values at each step.
    """

    transient: PreparedCoupledTransient
    solve: PreparedDAESolve
    control_binding: ParameterBinding | None
    parameter_ids: tuple[str, ...] = eqx.field(static=True)
    normalized_times: tuple[float, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    @property
    def control(self) -> str | None:
        return None if self.control_binding is None else self.control_binding.binding_id

    @property
    def state_size(self) -> int:
        return self.transient.state_space.size

    @property
    def state_layout(self) -> StateLayout:
        """Euclidean native transient coordinates (exact retraction geometry)."""
        return StateLayout(
            (self.state_size,),
            axes=("coupled-state",),
            layout_id=f"{self.transition_id}:state",
        )

    @property
    def input_layout(self) -> InputLayout | None:
        """The control binding's port as a control input, or ``None``."""
        binding = self.control_binding
        if binding is None:
            return None
        return InputLayout(
            binding.port.event_shape,
            component_names=binding.port.component_ids,
            roles="control",
            layout_id=f"{self.transition_id}:control:{binding.port.port_id}",
        )

    def parameter_values(
        self, control: ArrayLike | None, others: ParameterValues | None, /
    ) -> dict[str, Array]:
        """Runtime parameter mapping from a control input and the other bindings."""
        supplied = {} if others is None else dict(others)
        name = self.control
        if name is not None:
            if control is None:
                raise ValueError(f"The coupled transition requires control {name!r}.")
            if name in supplied:
                raise ValueError(
                    f"Control {name!r} is the transition input, not an argument."
                )
            supplied[name] = control
        elif control is not None:
            raise ValueError("An autonomous coupled transition accepts no control.")
        return self._complete(supplied)

    def _complete(self, supplied: ParameterValues, /) -> dict[str, Array]:
        if set(supplied) != set(self.parameter_ids):
            raise ValueError(
                "Coupled transition parameters must be exactly "
                f"{list(self.parameter_ids)}; received {sorted(supplied)}."
            )
        return {key: jnp.asarray(supplied[key]) for key in self.parameter_ids}

    def step(
        self,
        source: ArrayLike,
        target: ArrayLike,
        state: ArrayLike,
        parameters: ParameterValues,
        /,
    ) -> CoupledTransitionStep:
        """Integrate from ``source`` to ``target`` and certify the original rows.

        ``parameters`` maps every refresh binding ID (the control included) to
        its value on this step.
        """
        values = self._complete(parameters)
        start = jnp.asarray(source, dtype=jnp.float64)
        end = jnp.asarray(target, dtype=jnp.float64)
        if start.shape or end.shape:
            raise ValueError("Coupled transition step times must be scalar.")
        initial = jnp.asarray(state, dtype=jnp.float64)
        if initial.shape != (self.state_size,):
            raise ValueError(
                f"Coupled transition state must have shape {(self.state_size,)}; "
                f"received {initial.shape}."
            )
        return _step(self, start, end, initial, values)

    def discrete_system(self, /, *, system_id: str | None = None) -> DiscreteSystem:
        """The step through the ``DiscreteSystem`` transition ABI.

        A controlled transition reads the control from the input and every
        other refresh parameter from the system ``args`` mapping; an
        autonomous transition reads all of them from ``args``. Step intervals
        are the ``DiscreteStepContext`` source and target times.
        """
        transition = (
            _AutonomousDiscreteStep(self)
            if self.control is None
            else _ControlledDiscreteStep(self)
        )
        return DiscreteSystem(
            transition,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            system_id=self.transition_id if system_id is None else system_id,
        )

    def observation_port(self, binding_ids: Sequence[str], /) -> CoupledObservationPort:
        """Observation port over the prepared problem's observation bindings."""
        return CoupledObservationPort(self, binding_ids)


def _step(
    transition: PreparedCoupledTransition,
    source: Array,
    target: Array,
    state: Array,
    parameters: dict[str, Array],
    /,
) -> CoupledTransitionStep:
    # The window is rebuilt from these times; a reversed or nonfinite interval
    # is refused before the native solve instead of surfacing as a grid error.
    target = eqx.error_if(
        target,
        ~(jnp.isfinite(source) & jnp.isfinite(target) & (target > source)),
        "A coupled transition step requires finite target > source.",
    )
    times = source + (target - source) * jnp.asarray(
        transition.normalized_times, dtype=source.dtype
    )
    times = refuse_derivative_dependencies(
        times,
        times,
        message=(
            "a coupled transition step is not differentiated with respect to its "
            "physical step times; the native solve stops time derivatives."
        ),
    )
    window = eqx.tree_at(lambda solve: solve.time_grid.times, transition.solve, times)
    solution = solve_dae(window, args=parameters, initial_state=state)
    certificate = _certificate(
        transition.transient, solution, parameters, transition.tolerance
    )
    candidate = solution.states[-1]
    native = jnp.asarray(solution.successful)
    certified = jnp.all(certificate.accepted)
    finite = jnp.all(jnp.isfinite(candidate))
    successful = native & certified & finite
    status = jnp.where(
        ~native,
        COUPLED_TRANSITION_NATIVE_FAILURE,
        jnp.where(
            ~certified,
            COUPLED_TRANSITION_CERTIFICATE_FAILURE,
            jnp.where(finite, COUPLED_TRANSITION_SUCCESS, COUPLED_TRANSITION_NONFINITE),
        ),
    ).astype(jnp.int32)
    candidate, accepted = guard_derivative_validity(
        (candidate, jnp.where(successful, candidate, state)),
        successful,
        dependencies=(state, parameters),
        message="Derivative of a non-accepted coupled transition step.",
    )
    ratios = certificate.residual_norms / jnp.where(
        certificate.scales > 0.0, certificate.scales, 1.0
    )
    return CoupledTransitionStep(
        candidate=candidate,
        accepted=accepted,
        successful=successful,
        status=status,
        native_status=jnp.asarray(solution.termination_status, dtype=jnp.int32),
        residual_ratio=jnp.max(ratios),
    )


@final
class _ControlledDiscreteStep(StrictModule):
    transition: PreparedCoupledTransition

    def __call__(
        self, context: DiscreteStepContext, state: Array, inputs: Array, args: Any, /
    ) -> DiscreteTransitionResult:
        values = self.transition.parameter_values(inputs, _mapping(args))
        return _result(self.transition, context, state, values)


@final
class _AutonomousDiscreteStep(StrictModule):
    transition: PreparedCoupledTransition

    def __call__(
        self, context: DiscreteStepContext, state: Array, args: Any, /
    ) -> DiscreteTransitionResult:
        values = self.transition.parameter_values(None, _mapping(args))
        return _result(self.transition, context, state, values)


def _mapping(args: Any, /) -> ParameterValues | None:
    if args is None or isinstance(args, Mapping):
        return args
    raise TypeError(
        "Coupled transition args are the other refresh parameter values, keyed by "
        f"binding ID; received {type(args).__name__}."
    )


def _result(
    transition: PreparedCoupledTransition,
    context: DiscreteStepContext,
    state: Array,
    parameters: dict[str, Array],
    /,
) -> DiscreteTransitionResult:
    step = _step(
        transition,
        jnp.asarray(context.source, dtype=jnp.float64),
        jnp.asarray(context.target, dtype=jnp.float64),
        jnp.asarray(state, dtype=jnp.float64),
        parameters,
    )
    return DiscreteTransitionResult(
        step.candidate, step.accepted, step.successful, step.status
    )


def prepare_coupled_transition(
    transient: PreparedCoupledTransient,
    /,
    *,
    step: float,
    policy: DAESolvePolicy,
    parameters: ParameterValues,
    control: str | None = None,
    substeps: int = 1,
    tolerance: float = 1.0e-6,
) -> PreparedCoupledTransition:
    """Prepare a coupled transient as a native discrete transition (host step).

    ``step`` is the template window length used to prepare the native solve
    (its step-ratio and capacity checks); every call rebinds the window to its
    own interval. ``parameters`` gives a reference value of every refresh
    parameter binding of the prepared problem; ``control`` names the binding
    supplied as the control input. A control that does not enter the coupled
    transient rows is refused.
    """
    if not isinstance(transient, PreparedCoupledTransient):
        raise TypeError("transient must be a PreparedCoupledTransient.")
    if not isinstance(policy, DAESolvePolicy):
        raise TypeError("policy must be a DAESolvePolicy.")
    length = positive_finite_float(step, "step")
    if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps < 1:
        raise ValueError("substeps must be a positive integer.")
    tolerance_ = positive_finite_float(tolerance, "tolerance")
    bindings = _refresh_bindings(transient)
    control_binding = _control_binding(
        bindings, None if control is None else canonical_identifier(control, "control")
    )
    if control_binding is not None and control_binding.port.event_shape == ():
        raise ValueError(
            f"Control {control_binding.binding_id!r} needs a rank-one port event "
            "shape; control inputs are vectors."
        )
    reference = dict(parameters)
    # Validates keys, shapes, dtypes, and authorities of every refresh value.
    transient.prepared.bind_arguments(parameters=reference)
    values = {key: jnp.asarray(value) for key, value in reference.items()}
    if control_binding is not None:
        _require_control_incidence(transient, control_binding.binding_id, values, length)
    normalized = np.linspace(0.0, 1.0, substeps + 1)
    template = TimeGrid(length * normalized, time_id="coupled-transition-window")
    solve = prepare_dae(
        transient.problem(transient.state_space.zeros(), parameters=values),
        template,
        policy=policy,
    )
    identity = canonical_fingerprint(
        {
            "kind": "coupled-transition",
            "transient": transient.transient_id,
            "solve": solve.prepared_id,
            "control": None if control_binding is None else control_binding.binding_id,
            "parameters": sorted(values),
            "substeps": substeps,
            "tolerance": tolerance_,
        }
    )
    return PreparedCoupledTransition(
        transient=transient,
        solve=solve,
        control_binding=control_binding,
        parameter_ids=tuple(binding.binding_id for binding in bindings),
        normalized_times=tuple(float(value) for value in normalized),
        tolerance=tolerance_,
        transition_id=identity,
    )


def _covariance_block(plan: MeasurementComparisonPlan, /) -> np.ndarray:
    """Dense covariance of one plan's declared noise model (host)."""
    observed = plan.observed
    match plan.noise_model:
        case "independent_uncertainty":
            if observed.standard_uncertainty is None:
                raise ValueError("Independent uncertainty requires standard values.")
            std = np.asarray(observed.standard_uncertainty, dtype=np.float64).ravel()
            return np.diag(std**2)
        case "covariance":
            if not bool(np.all(np.asarray(observed.valid_mask))):
                raise ValueError(
                    f"Plan {plan.plan_id!r} declares correlated noise on a subset of "
                    "its values; a state-space observation covers every value."
                )
            covariance = plan.covariance
            if isinstance(covariance, DiagonalCovarianceAction):
                return np.diag(np.asarray(covariance.variance, dtype=np.float64))
            if isinstance(covariance, CholeskyCovarianceAction):
                lower = np.asarray(covariance.lower_cholesky, dtype=np.float64)
                return lower @ lower.T
            raise ValueError(
                f"Plan {plan.plan_id!r} declares a {type(covariance).__name__}; only "
                "diagonal and dense Cholesky covariances have an exact dense "
                "state-space observation covariance."
            )
        case "unquantified" | "reference_weighting":
            raise ValueError(
                f"Plan {plan.plan_id!r} declares no measurement noise "
                f"({plan.noise_model!r}); a Gaussian state-space likelihood needs "
                "declared uncertainty or covariance."
            )
        case _:
            raise ValueError(f"Unknown noise model {plan.noise_model!r}.")


def _measure(
    transition: PreparedCoupledTransition,
    binding_ids: tuple[str, ...],
    time: ArrayLike,
    state: ArrayLike,
    parameters: ParameterValues,
    /,
) -> tuple[PreparedQuantityField, ...]:
    """Observation bindings on the full fields (lifts included) at ``time``."""
    transient = transition.transient
    prepared = transient.prepared
    time_ = jnp.asarray(time, dtype=jnp.float64)
    view: NestedBlocks = transient.state_view(jnp.asarray(state, dtype=jnp.float64))
    arguments = transient.core.owner_arguments(time_, dict(parameters))
    fields: dict[tuple[str, str], Array] = {}
    results = []
    for item in binding_ids:
        observation = prepared.observation(item)
        key = (observation.component, observation.field)
        if key not in fields:
            fields[key] = prepared.field(key[0], key[1], view, arguments)
        results.append(observation.evaluate(fields, arguments))
    return tuple(results)


def _predicted(
    transition: PreparedCoupledTransition, binding_ids: tuple[str, ...], /
) -> tuple[PreparedQuantityField, ...]:
    """Shapes and static measurement identities of the predicted fields."""
    values = {
        binding.binding_id: jax.ShapeDtypeStruct(binding.port.event_shape, jnp.float64)
        for binding in _refresh_bindings(transition.transient)
    }
    state = jax.ShapeDtypeStruct((transition.state_size,), jnp.float64)
    return eqx.filter_eval_shape(
        lambda z, p: _measure(transition, binding_ids, 0.0, z, p), state, values
    )


@final
class CoupledObservationPort(StrictModule):
    """P6 observation bindings evaluated on native coupled transient states.

    ``measure`` returns the prepared quantity fields (with their measurement
    identities); ``values`` concatenates their flattened values in
    ``binding_ids`` order. The port is the control output ABI through
    ``__call__(time, state, control, args)`` and the state-space location ABI
    through ``location(state, time, context)`` (parameters from
    ``context.args``). ``noise_covariance`` derives the Gaussian observation
    covariance from the declared noise of measurement comparison plans whose
    observed identities equal the predicted ones.

    Both ABIs are plain value vectors with no validity channel, so every
    binding must predict every sample: masked point observations with
    unlocated or inactive samples are refused. Flux-content bindings are
    refused too: they pair the steady residual reaction, while the transient
    reaction also carries the capacity action ``C u'``.
    """

    transition: PreparedCoupledTransition
    binding_ids: tuple[str, ...] = eqx.field(static=True)
    observation_size: int = eqx.field(static=True)

    @checked
    def __init__(
        self, transition: PreparedCoupledTransition, binding_ids: Sequence[str], /
    ) -> None:
        ids = tuple(canonical_identifier(item, "binding_id") for item in binding_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("binding_ids must name distinct observation bindings.")
        prepared = transition.transient.prepared
        for item in ids:
            observation = prepared.observation(item)
            if isinstance(observation, PreparedFluxObservation):
                raise ValueError(
                    f"Observation {item!r} is a flux content of the steady residual "
                    "reaction; a transient reaction also carries the capacity action "
                    "C u', so a coupled-transition port does not observe it."
                )
            if not observation.complete:
                raise ValueError(
                    f"Observation {item!r} has samples that are never valid; the "
                    "port's value vector has no validity channel, so every sample "
                    "must be predicted. Restrict the support to the observed samples."
                )
        self.observation_size = sum(
            prod(field.values.shape) for field in _predicted(transition, ids)
        )
        self.transition = transition
        self.binding_ids = ids

    def measure(
        self, time: ArrayLike, state: ArrayLike, parameters: ParameterValues, /
    ) -> tuple[PreparedQuantityField, ...]:
        return _measure(self.transition, self.binding_ids, time, state, parameters)

    def values(
        self, time: ArrayLike, state: ArrayLike, parameters: ParameterValues, /
    ) -> Array:
        measured = self.measure(time, state, parameters)
        return jnp.concatenate([jnp.ravel(field.values) for field in measured])

    def __call__(self, time: Array, state: Array, control: Array, args: Any, /) -> Array:
        """Control output ABI: parameters are the control and ``args``."""
        return self.values(
            time, state, self.transition.parameter_values(control, _mapping(args))
        )

    def location(self, state: Array, time: Array, context: Any, /) -> Array:
        """State-space location ABI: parameters are ``context.args``."""
        values = self.transition.parameter_values(None, _mapping(context.args))
        return self.values(time, state, values)

    def noise_covariance(self, plans: Sequence[MeasurementComparisonPlan], /) -> Array:
        """Dense observation covariance from the plans' declared noise (host).

        ``plans`` align with ``binding_ids``; each observed field must carry
        the predicted quantity, layout, support, sampling, and unit identities.
        """
        plans_ = tuple(plans)
        if len(plans_) != len(self.binding_ids):
            raise ValueError("Provide one comparison plan per observation binding.")
        blocks = []
        for item, plan, predicted in zip(
            self.binding_ids,
            plans_,
            _predicted(self.transition, self.binding_ids),
            strict=True,
        ):
            if not isinstance(plan, MeasurementComparisonPlan):
                raise TypeError("plans must be MeasurementComparisonPlan values.")
            observed = plan.observed
            for role, left, right in (
                ("quantity", predicted.compatibility_id, observed.compatibility_id),
                ("layout", predicted.layout_id, observed.layout_id),
                ("support", predicted.support_id, observed.support_id),
                ("sampling", predicted.sampling_id, observed.sampling_id),
                ("unit", predicted.unit_id, observed.unit_id),
            ):
                if left != right:
                    raise ValueError(
                        f"Observation {item!r}: predicted and observed {role} "
                        "identities differ."
                    )
            blocks.append(_covariance_block(plan))
        size = sum(block.shape[0] for block in blocks)
        covariance = np.zeros((size, size))
        offset = 0
        for block in blocks:
            stop = offset + block.shape[0]
            covariance[offset:stop, offset:stop] = block
            offset = stop
        return jnp.asarray(covariance)


__all__ = [
    "COUPLED_TRANSITION_CERTIFICATE_FAILURE",
    "COUPLED_TRANSITION_NATIVE_FAILURE",
    "COUPLED_TRANSITION_NONFINITE",
    "COUPLED_TRANSITION_SUCCESS",
    "CoupledObservationPort",
    "CoupledTransitionStatus",
    "CoupledTransitionStep",
    "PreparedCoupledTransition",
    "coupled_transition_status_name",
    "prepare_coupled_transition",
]
