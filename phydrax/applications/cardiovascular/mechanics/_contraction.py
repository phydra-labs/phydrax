#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ..electrophysiology._reaction import CardiacReactionEvaluation


class ContractionState(StrictModule):
    """Fixed-shape local state shared by all active-contraction fidelities."""

    activation: Array
    calcium_bound: Array
    distortion: Array
    previous_stretch: Array

    def __init__(
        self,
        activation: ArrayLike,
        calcium_bound: ArrayLike,
        distortion: ArrayLike,
        previous_stretch: ArrayLike,
        /,
    ):
        activation_ = jnp.asarray(activation)
        calcium_bound_ = jnp.asarray(calcium_bound, dtype=activation_.dtype)
        distortion_ = jnp.asarray(distortion, dtype=activation_.dtype)
        previous_stretch_ = jnp.asarray(previous_stretch, dtype=activation_.dtype)
        if not (
            activation_.shape
            == calcium_bound_.shape
            == distortion_.shape
            == previous_stretch_.shape
        ):
            raise ValueError("Contraction state fields must have one fixed common shape.")
        self.activation = activation_
        self.calcium_bound = calcium_bound_
        self.distortion = distortion_
        self.previous_stretch = previous_stretch_

    @classmethod
    def resting(
        cls,
        shape: tuple[int, ...],
        /,
        *,
        dtype: Any = jnp.float64,
        reference_stretch: float = 1.0,
    ) -> ContractionState:
        shape_ = tuple(int(value) for value in shape)
        if any(value <= 0 for value in shape_):
            raise ValueError("Contraction state shape entries must be positive.")
        zeros = jnp.zeros(shape_, dtype=dtype)
        stretch = jnp.full(shape_, reference_stretch, dtype=dtype)
        return cls(zeros, zeros, zeros, stretch)


class ContractionEvidence(StrictModule):
    """Local numerical and physical evidence for one contraction candidate."""

    drive: Array
    target_activation: Array
    active_tension: Array
    length_factor: Array
    velocity_factor: Array
    state_increment_norm: Array
    finite: Array
    bounds_satisfied: Array
    successful: Array
    fidelity_id: str = eqx.field(static=True)
    live_calcium_consumed: bool = eqx.field(static=True)


class ContractionCandidate(StrictModule):
    """Uncommitted state and tension; failed candidates retain the old accepted state."""

    previous_state: ContractionState
    candidate_state: ContractionState
    active_tension: Array
    evidence: ContractionEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class ContractionCheckpoint(StrictModule):
    state: ContractionState
    time: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: ContractionState,
        time: ArrayLike,
        step_index: ArrayLike,
        /,
        *,
        plan_id: str,
    ):
        if not isinstance(state, ContractionState):
            raise TypeError("Contraction checkpoint state must be ContractionState.")
        time_ = jnp.asarray(time)
        index = jnp.asarray(step_index, dtype=jnp.int32)
        if time_.shape != () or index.shape != ():
            raise ValueError("Contraction checkpoint time and step index must be scalar.")
        self.state = state
        self.time = time_
        self.step_index = index
        self.plan_id = str(plan_id)


class PrescribedTensionContractionPlan(StrictModule, NonTrainableState):
    """Prescribed active tension in kPa; this is active mechanics, not electromechanics."""

    allow_compression: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    fidelity_id: str = eqx.field(static=True, default="prescribed-tension")

    def __init__(self, /, *, allow_compression: bool = False):
        self.allow_compression = bool(allow_compression)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiac-contraction",
                "fidelity": self.fidelity_id,
                "allow_compression": self.allow_compression,
                "stress_unit": "kPa",
            }
        )


class ActivationDrivenContractionPlan(StrictModule, NonTrainableState):
    """First-order tension activation driven by a dimensionless prescribed signal."""

    peak_tension: float = eqx.field(static=True)
    activation_time: float = eqx.field(static=True)
    relaxation_time: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    fidelity_id: str = eqx.field(static=True, default="activation-driven-first-order")

    def __init__(
        self,
        peak_tension: float,
        /,
        *,
        activation_time: float = 30.0,
        relaxation_time: float = 80.0,
    ):
        values = (float(peak_tension), float(activation_time), float(relaxation_time))
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Activation-driven contraction parameters must be positive.")
        self.peak_tension, self.activation_time, self.relaxation_time = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiac-contraction",
                "fidelity": self.fidelity_id,
                "peak_tension_kPa": self.peak_tension,
                "activation_time_ms": self.activation_time,
                "relaxation_time_ms": self.relaxation_time,
            }
        )


class CalciumDrivenFirstOrderContractionPlan(StrictModule, NonTrainableState):
    """First-order active tension driven by live free cytosolic Ca concentration."""

    peak_tension: float = eqx.field(static=True)
    half_activation_calcium: float = eqx.field(static=True)
    hill_exponent: float = eqx.field(static=True)
    activation_time: float = eqx.field(static=True)
    relaxation_time: float = eqx.field(static=True)
    calcium_unit: str = eqx.field(static=True)
    ionic_model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    fidelity_id: str = eqx.field(static=True, default="calcium-driven-first-order")

    def __init__(
        self,
        peak_tension: float,
        half_activation_calcium: float,
        /,
        *,
        hill_exponent: float = 2.0,
        activation_time: float = 25.0,
        relaxation_time: float = 75.0,
        calcium_unit: str = "mM",
        ionic_model_id: str,
    ):
        values = (
            float(peak_tension),
            float(half_activation_calcium),
            float(hill_exponent),
            float(activation_time),
            float(relaxation_time),
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Calcium-driven contraction parameters must be positive.")
        if not calcium_unit or not ionic_model_id:
            raise ValueError("Calcium unit and compatible ionic model ID are required.")
        (
            self.peak_tension,
            self.half_activation_calcium,
            self.hill_exponent,
            self.activation_time,
            self.relaxation_time,
        ) = values
        self.calcium_unit = str(calcium_unit)
        self.ionic_model_id = str(ionic_model_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiac-contraction",
                "fidelity": self.fidelity_id,
                "peak_tension_kPa": self.peak_tension,
                "half_activation_calcium": self.half_activation_calcium,
                "hill_exponent": self.hill_exponent,
                "activation_time_ms": self.activation_time,
                "relaxation_time_ms": self.relaxation_time,
                "calcium_unit": self.calcium_unit,
                "ionic_model_id": self.ionic_model_id,
            }
        )


class LandLengthVelocityContractionPlan(StrictModule, NonTrainableState):
    """Land-class Ca, sarcomere-length, and shortening-velocity active tension."""

    peak_tension: float = eqx.field(static=True)
    half_activation_calcium: float = eqx.field(static=True)
    hill_exponent: float = eqx.field(static=True)
    calcium_binding_time: float = eqx.field(static=True)
    distortion_time: float = eqx.field(static=True)
    length_sensitivity: float = eqx.field(static=True)
    velocity_sensitivity: float = eqx.field(static=True)
    minimum_length_factor: float = eqx.field(static=True)
    reference_stretch: float = eqx.field(static=True)
    calcium_unit: str = eqx.field(static=True)
    ionic_model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    fidelity_id: str = eqx.field(static=True, default="land-length-velocity-calcium")

    def __init__(
        self,
        peak_tension: float,
        half_activation_calcium: float,
        /,
        *,
        hill_exponent: float = 2.5,
        calcium_binding_time: float = 20.0,
        distortion_time: float = 35.0,
        length_sensitivity: float = 1.8,
        velocity_sensitivity: float = 0.35,
        minimum_length_factor: float = 0.05,
        reference_stretch: float = 1.0,
        calcium_unit: str = "mM",
        ionic_model_id: str,
    ):
        positive = (
            float(peak_tension),
            float(half_activation_calcium),
            float(hill_exponent),
            float(calcium_binding_time),
            float(distortion_time),
            float(length_sensitivity),
            float(reference_stretch),
        )
        minimum = float(minimum_length_factor)
        velocity = float(velocity_sensitivity)
        if any(not isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Land contraction positive parameters are invalid.")
        if not isfinite(minimum) or minimum <= 0.0 or minimum > 1.0:
            raise ValueError("minimum_length_factor must lie in (0, 1].")
        if not isfinite(velocity) or velocity < 0.0:
            raise ValueError("velocity_sensitivity must be finite and non-negative.")
        if not calcium_unit or not ionic_model_id:
            raise ValueError("Land contraction requires calcium unit and ionic model ID.")
        (
            self.peak_tension,
            self.half_activation_calcium,
            self.hill_exponent,
            self.calcium_binding_time,
            self.distortion_time,
            self.length_sensitivity,
            self.reference_stretch,
        ) = positive
        self.velocity_sensitivity = velocity
        self.minimum_length_factor = minimum
        self.calcium_unit = str(calcium_unit)
        self.ionic_model_id = str(ionic_model_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiac-contraction",
                "fidelity": self.fidelity_id,
                "peak_tension_kPa": self.peak_tension,
                "half_activation_calcium": self.half_activation_calcium,
                "hill_exponent": self.hill_exponent,
                "calcium_binding_time_ms": self.calcium_binding_time,
                "distortion_time_ms": self.distortion_time,
                "length_sensitivity": self.length_sensitivity,
                "velocity_sensitivity_ms": self.velocity_sensitivity,
                "minimum_length_factor": self.minimum_length_factor,
                "reference_stretch": self.reference_stretch,
                "calcium_unit": self.calcium_unit,
                "ionic_model_id": self.ionic_model_id,
            }
        )


ContractionPlan: TypeAlias = (
    PrescribedTensionContractionPlan
    | ActivationDrivenContractionPlan
    | CalciumDrivenFirstOrderContractionPlan
    | LandLengthVelocityContractionPlan
)


def _safe_hill(calcium: Array, half: float, exponent: float) -> Array:
    positive = jnp.maximum(calcium, 0.0)
    numerator = positive**exponent
    denominator = numerator + jnp.asarray(half, dtype=positive.dtype) ** exponent
    return numerator / jnp.maximum(denominator, jnp.finfo(positive.dtype).tiny)


def _exact_first_order(
    current: Array, target: Array, time_step: Array, tau: Array
) -> Array:
    return target + (current - target) * jnp.exp(-time_step / tau)


def _state_norm(candidate: ContractionState, previous: ContractionState) -> Array:
    leaves = jax.tree.leaves(
        jax.tree.map(lambda new, old: new - old, candidate, previous)
    )
    return jnp.sqrt(sum(jnp.sum(value * value) for value in leaves))


def _select_state(
    successful: Array, candidate: ContractionState, previous: ContractionState
) -> ContractionState:
    return jax.tree.map(
        lambda new, old: jnp.where(successful, new, old), candidate, previous
    )


class PreparedContraction(StrictModule, NonTrainableState):
    """Prepared fixed-shape transactional contraction kernel."""

    plan: ContractionPlan
    reference_state: ContractionState
    field_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ContractionPlan, state: ContractionState, /):
        if not isinstance(
            plan,
            (
                PrescribedTensionContractionPlan,
                ActivationDrivenContractionPlan,
                CalciumDrivenFirstOrderContractionPlan,
                LandLengthVelocityContractionPlan,
            ),
        ):
            raise TypeError("Unknown contraction fidelity plan.")
        if not isinstance(state, ContractionState):
            raise TypeError("Prepared contraction requires a ContractionState.")
        dtype = np.dtype(state.activation.dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Contraction state must use a floating-point dtype.")
        self.plan = plan
        self.reference_state = state
        self.field_shape = state.activation.shape
        self.dtype = dtype
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiac-contraction",
                "plan": plan.plan_id,
                "shape": list(self.field_shape),
                "dtype": dtype.str,
            }
        )

    def candidate(
        self,
        state: ContractionState,
        drive: ArrayLike,
        stretch: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        shortening_velocity: ArrayLike | None = None,
        ionic_model_id: str | None = None,
        calcium_unit: str | None = None,
    ) -> ContractionCandidate:
        if not isinstance(state, ContractionState):
            raise TypeError("Contraction candidate state must be ContractionState.")
        drive_ = jnp.asarray(drive, dtype=self.dtype)
        stretch_ = jnp.asarray(stretch, dtype=self.dtype)
        dt = jnp.asarray(time_step, dtype=self.dtype)
        if (
            state.activation.shape != self.field_shape
            or drive_.shape != self.field_shape
            or stretch_.shape != self.field_shape
            or dt.shape != ()
        ):
            raise ValueError("Contraction candidate inputs violate the prepared shape.")
        velocity = (
            (state.previous_stretch - stretch_) / dt
            if shortening_velocity is None
            else jnp.asarray(shortening_velocity, dtype=self.dtype)
        )
        if velocity.shape != self.field_shape:
            raise ValueError("Shortening velocity violates the prepared field shape.")

        plan = self.plan
        one = jnp.ones_like(drive_)
        zero = jnp.zeros_like(drive_)
        live_calcium = isinstance(
            plan,
            (CalciumDrivenFirstOrderContractionPlan, LandLengthVelocityContractionPlan),
        )
        compatible = True
        if live_calcium:
            compatible = (
                ionic_model_id == plan.ionic_model_id
                and calcium_unit == plan.calcium_unit
            )
        if isinstance(plan, PrescribedTensionContractionPlan):
            tension = drive_ if plan.allow_compression else jnp.maximum(drive_, 0.0)
            target = tension
            candidate = ContractionState(tension, state.calcium_bound, zero, stretch_)
            length_factor, velocity_factor = one, one
            bounds = jnp.all(plan.allow_compression | (tension >= 0.0))
        elif isinstance(plan, ActivationDrivenContractionPlan):
            target = jnp.clip(drive_, 0.0, 1.0)
            tau = jnp.where(
                target >= state.activation, plan.activation_time, plan.relaxation_time
            )
            activation = _exact_first_order(state.activation, target, dt, tau)
            tension = plan.peak_tension * activation
            candidate = ContractionState(activation, state.calcium_bound, zero, stretch_)
            length_factor, velocity_factor = one, one
            bounds = jnp.all((activation >= 0.0) & (activation <= 1.0))
        elif isinstance(plan, CalciumDrivenFirstOrderContractionPlan):
            target = _safe_hill(drive_, plan.half_activation_calcium, plan.hill_exponent)
            tau = jnp.where(
                target >= state.activation, plan.activation_time, plan.relaxation_time
            )
            activation = _exact_first_order(state.activation, target, dt, tau)
            tension = plan.peak_tension * activation
            candidate = ContractionState(activation, target, zero, stretch_)
            length_factor, velocity_factor = one, one
            bounds = jnp.all((activation >= 0.0) & (activation <= 1.0))
        else:
            target = _safe_hill(drive_, plan.half_activation_calcium, plan.hill_exponent)
            bound = _exact_first_order(
                state.calcium_bound,
                target,
                dt,
                jnp.asarray(plan.calcium_binding_time, dtype=dt.dtype),
            )
            length_factor = jnp.maximum(
                plan.minimum_length_factor,
                1.0 + plan.length_sensitivity * (stretch_ - plan.reference_stretch),
            )
            distortion_target = -plan.velocity_sensitivity * velocity
            distortion = _exact_first_order(
                state.distortion,
                distortion_target,
                dt,
                jnp.asarray(plan.distortion_time, dtype=dt.dtype),
            )
            velocity_factor = jnp.maximum(0.0, 1.0 + distortion)
            activation = jnp.clip(bound * length_factor, 0.0, 1.0)
            tension = plan.peak_tension * bound * length_factor * velocity_factor
            candidate = ContractionState(activation, bound, distortion, stretch_)
            bounds = jnp.all((bound >= 0.0) & (bound <= 1.0)) & jnp.all(
                length_factor > 0.0
            )

        leaves = jax.tree.leaves(candidate)
        finite = (
            jnp.all(jnp.isfinite(dt))
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(drive_))
            & jnp.all(jnp.isfinite(stretch_))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(tension))
            & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))
        )
        successful = finite & bounds & jnp.asarray(compatible)
        evidence = ContractionEvidence(
            drive_,
            target,
            tension,
            length_factor,
            velocity_factor,
            _state_norm(candidate, state),
            finite,
            bounds,
            successful,
            plan.fidelity_id,
            live_calcium,
        )
        return ContractionCandidate(state, candidate, tension, evidence, self.prepared_id)

    def candidate_from_reaction(
        self,
        state: ContractionState,
        reaction: CardiacReactionEvaluation,
        stretch: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        shortening_velocity: ArrayLike | None = None,
    ) -> ContractionCandidate:
        """Consume live Ca from a concrete compatible ionic-model evaluation."""

        if not isinstance(reaction, CardiacReactionEvaluation):
            raise TypeError("Ca-driven contraction requires CardiacReactionEvaluation.")
        plan = self.plan
        if not isinstance(
            plan,
            (CalciumDrivenFirstOrderContractionPlan, LandLengthVelocityContractionPlan),
        ):
            raise TypeError(
                "Only Ca-driven first-order and Land contraction consume reactions."
            )
        candidate = self.candidate(
            state,
            reaction.calcium_cytosol_mM,
            stretch,
            time_step,
            shortening_velocity=shortening_velocity,
            ionic_model_id=reaction.model_id,
            calcium_unit="mM",
        )
        successful = candidate.evidence.successful & jnp.all(reaction.valid)
        evidence = ContractionEvidence(
            candidate.evidence.drive,
            candidate.evidence.target_activation,
            candidate.evidence.active_tension,
            candidate.evidence.length_factor,
            candidate.evidence.velocity_factor,
            candidate.evidence.state_increment_norm,
            candidate.evidence.finite,
            candidate.evidence.bounds_satisfied,
            successful,
            candidate.evidence.fidelity_id,
            True,
        )
        return ContractionCandidate(
            candidate.previous_state,
            candidate.candidate_state,
            candidate.active_tension,
            evidence,
            candidate.plan_id,
        )

    def commit(self, candidate: ContractionCandidate, /) -> ContractionState:
        if not isinstance(candidate, ContractionCandidate):
            raise TypeError("commit requires ContractionCandidate evidence.")
        if candidate.plan_id != self.prepared_id:
            raise ValueError("Contraction candidate belongs to another prepared plan.")
        return _select_state(
            candidate.evidence.successful,
            candidate.candidate_state,
            candidate.previous_state,
        )

    def checkpoint(
        self, state: ContractionState, time: ArrayLike, step_index: ArrayLike, /
    ) -> ContractionCheckpoint:
        if state.activation.shape != self.field_shape:
            raise ValueError("Checkpoint state violates prepared contraction shape.")
        return ContractionCheckpoint(state, time, step_index, plan_id=self.prepared_id)

    def restore(self, checkpoint: ContractionCheckpoint, /) -> ContractionState:
        if not isinstance(checkpoint, ContractionCheckpoint):
            raise TypeError("restore requires ContractionCheckpoint.")
        if checkpoint.plan_id != self.prepared_id:
            raise ValueError("Contraction checkpoint belongs to another prepared plan.")
        if checkpoint.state.activation.shape != self.field_shape:
            raise ValueError("Contraction checkpoint shape is incompatible.")
        return checkpoint.state


def prepare_contraction(
    plan: ContractionPlan,
    state: ContractionState,
    /,
) -> PreparedContraction:
    """Freeze one contraction fidelity and local-state shape before execution."""

    return PreparedContraction(plan, state)


__all__ = [
    "ActivationDrivenContractionPlan",
    "CalciumDrivenFirstOrderContractionPlan",
    "ContractionCandidate",
    "ContractionCheckpoint",
    "ContractionEvidence",
    "ContractionPlan",
    "ContractionState",
    "LandLengthVelocityContractionPlan",
    "PrescribedTensionContractionPlan",
    "PreparedContraction",
    "prepare_contraction",
]
