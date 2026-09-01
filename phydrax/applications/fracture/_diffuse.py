#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import CellResidualAction, FiniteElementForm
from ...equations.fem import symmetric_gradient


class PhaseFieldFractureParameters(StrictModule, NonTrainableState):
    """Isotropic AT2 fracture parameters shared by every diffuse realization."""

    lame_lambda: Array
    shear_modulus: Array
    critical_energy_release_rate: Array
    length_scale: Array
    residual_stiffness: Array

    def __init__(
        self,
        lame_lambda: ArrayLike,
        shear_modulus: ArrayLike,
        critical_energy_release_rate: ArrayLike,
        length_scale: ArrayLike,
        /,
        *,
        residual_stiffness: ArrayLike = 1.0e-8,
    ):
        values = tuple(
            np.asarray(value)
            for value in (
                lame_lambda,
                shear_modulus,
                critical_energy_release_rate,
                length_scale,
                residual_stiffness,
            )
        )
        if any(value.shape != () or not np.isfinite(value) for value in values):
            raise ValueError("Phase-field fracture parameters must be finite scalars.")
        if values[0] < 0.0 or any(value <= 0.0 for value in values[1:4]):
            raise ValueError(
                "Phase-field elastic and regularization data are inadmissible."
            )
        if not 0.0 <= values[4] < 1.0:
            raise ValueError("Residual fracture stiffness must lie in [0, 1).")
        (
            self.lame_lambda,
            self.shear_modulus,
            self.critical_energy_release_rate,
            self.length_scale,
            self.residual_stiffness,
        ) = tuple(jnp.asarray(value) for value in values)

    def degradation(self, damage: ArrayLike, /) -> Array:
        damage_ = jnp.asarray(damage)
        return (1.0 - self.residual_stiffness) * (
            1.0 - damage_
        ) ** 2 + self.residual_stiffness

    def tensile_energy(self, strain: ArrayLike, /) -> Array:
        strain_ = jnp.asarray(strain)
        if strain_.ndim < 2 or strain_.shape[-2] != strain_.shape[-1]:
            raise ValueError("Strain must end in a square tensor layout.")
        dimension = strain_.shape[-1]
        trace = jnp.trace(strain_, axis1=-2, axis2=-1)
        deviator = strain_ - trace[..., None, None] * jnp.eye(dimension) / dimension
        bulk = self.lame_lambda + 2.0 * self.shear_modulus / dimension
        return 0.5 * bulk * jnp.maximum(trace, 0.0) ** 2 + self.shear_modulus * jnp.sum(
            deviator**2, axis=(-2, -1)
        )


class PhaseFieldHistoryState(StrictModule, NonTrainableState):
    """History and damage from the last accepted diffuse-fracture step."""

    history: Array
    accepted_damage: Array
    state_version: int = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        history: ArrayLike,
        accepted_damage: ArrayLike,
        /,
        *,
        state_version: int = 0,
    ):
        history_ = np.asarray(history)
        damage = np.asarray(accepted_damage)
        version = int(state_version)
        if history_.ndim != 2 or damage.ndim < 1 or version < 0:
            raise ValueError("Phase-field history/damage layouts or version are invalid.")
        if not np.all(np.isfinite(history_)) or np.any(history_ < 0.0):
            raise ValueError(
                "Accepted phase-field history must be finite and nonnegative."
            )
        if not np.all(np.isfinite(damage)) or np.any((damage < 0.0) | (damage > 1.0)):
            raise ValueError("Accepted phase-field damage must lie in [0, 1].")
        self.history = jnp.asarray(history_)
        self.accepted_damage = jnp.asarray(damage)
        self.state_version = version
        self.state_id = canonical_fingerprint(
            {
                "kind": "accepted-phase-field-fracture-history",
                "history": history_.tolist(),
                "damage": damage.tolist(),
                "state_version": version,
            }
        )

    def transaction(
        self,
        tensile_energy: ArrayLike,
        damage: ArrayLike,
        /,
        *,
        accepted: bool,
    ) -> PhaseFieldHistoryTransaction:
        """Build a candidate without mutating accepted history or damage."""

        energy = jnp.asarray(tensile_energy)
        damage_ = jnp.asarray(damage)
        if energy.shape != self.history.shape:
            raise ValueError("Tensile energy must preserve the accepted history layout.")
        if damage_.shape != self.accepted_damage.shape:
            raise ValueError("Trial damage must preserve the accepted damage layout.")
        return PhaseFieldHistoryTransaction(
            jnp.maximum(self.history, energy),
            damage_,
            base_state_version=self.state_version,
            accepted=accepted,
        )


class PhaseFieldHistoryTransaction(StrictModule, NonTrainableState):
    """A trial history update that changes accepted state only on commit."""

    trial_history: Array
    trial_damage: Array
    base_state_version: int = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        trial_history: ArrayLike,
        trial_damage: ArrayLike,
        /,
        *,
        base_state_version: int,
        accepted: bool,
    ):
        history = np.asarray(trial_history)
        damage = np.asarray(trial_damage)
        version = int(base_state_version)
        if history.ndim != 2 or damage.ndim < 1 or version < 0:
            raise ValueError("Trial phase-field history/damage layouts are invalid.")
        if not np.all(np.isfinite(history)) or np.any(history < 0.0):
            raise ValueError("Trial phase-field history must be finite and nonnegative.")
        if not np.all(np.isfinite(damage)) or np.any((damage < 0.0) | (damage > 1.0)):
            raise ValueError("Trial phase-field damage must lie in [0, 1].")
        self.trial_history = jnp.asarray(history)
        self.trial_damage = jnp.asarray(damage)
        self.base_state_version = version
        self.accepted = bool(accepted)
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "phase-field-fracture-history-transaction",
                "history": history.tolist(),
                "damage": damage.tolist(),
                "base_state_version": version,
                "accepted": bool(accepted),
            }
        )

    def commit(self, current: PhaseFieldHistoryState, /) -> PhaseFieldHistoryState:
        if not isinstance(current, PhaseFieldHistoryState):
            raise TypeError("current must be a PhaseFieldHistoryState.")
        if current.state_version != self.base_state_version:
            raise ValueError(
                "Phase-field history transaction targets a stale state version."
            )
        if self.trial_history.shape != current.history.shape:
            raise ValueError("A history transaction must preserve the history layout.")
        if self.trial_damage.shape != current.accepted_damage.shape:
            raise ValueError("A history transaction must preserve the damage layout.")
        if not self.accepted:
            return current
        if bool(jnp.any(self.trial_damage < current.accepted_damage)):
            raise ValueError("Accepted phase-field damage cannot decrease.")
        return PhaseFieldHistoryState(
            jnp.maximum(current.history, self.trial_history),
            self.trial_damage,
            state_version=current.state_version + 1,
        )


class FixedHistoryNeuralBlock(StrictModule, NonTrainableState):
    """Bounded neural damage evaluated against one immutable accepted history."""

    fixed_history: Array
    damage: Array
    logits: Array
    base_state_version: int = eqx.field(static=True)

    def __init__(
        self,
        fixed_history: ArrayLike,
        damage: ArrayLike,
        logits: ArrayLike,
        /,
        *,
        base_state_version: int,
    ):
        history = jnp.asarray(fixed_history)
        damage_ = jnp.asarray(damage)
        logits_ = jnp.asarray(logits)
        version = int(base_state_version)
        if history.ndim != 2 or damage_.ndim < 1 or damage_.shape != logits_.shape:
            raise ValueError("Fixed-history neural block layouts are incompatible.")
        if version < 0:
            raise ValueError("Fixed neural history state version must be nonnegative.")
        history = eqx.error_if(
            history,
            jnp.any(~jnp.isfinite(history) | (history < 0.0)),
            "Fixed neural history must be finite and nonnegative.",
        )
        damage_ = eqx.error_if(
            damage_,
            jnp.any(~jnp.isfinite(damage_) | (damage_ < 0.0) | (damage_ > 1.0)),
            "Neural phase-field damage must lie in [0, 1].",
        )
        logits_ = eqx.error_if(
            logits_,
            jnp.any(~jnp.isfinite(logits_)),
            "Neural phase-field logits must be finite.",
        )
        self.fixed_history = history
        self.damage = damage_
        self.logits = logits_
        self.base_state_version = version

    def transaction(
        self,
        tensile_energy: ArrayLike,
        /,
        *,
        accepted: bool,
    ) -> PhaseFieldHistoryTransaction:
        energy = jnp.asarray(tensile_energy)
        if energy.shape != self.fixed_history.shape:
            raise ValueError("Tensile energy must preserve the fixed history layout.")
        return PhaseFieldHistoryTransaction(
            jnp.maximum(self.fixed_history, energy),
            self.damage,
            base_state_version=self.base_state_version,
            accepted=accepted,
        )


class BoundedNeuralFixedHistoryController(StrictModule):
    """Map pointwise neural logits into the irreversible interval [d_n, 1]."""

    network: object
    controller_id: str = eqx.field(static=True)

    def __init__(self, network: object, /, *, controller_id: str):
        if not callable(network):
            raise TypeError("network must be callable.")
        identifier = str(controller_id)
        if not identifier:
            raise ValueError("controller_id must be non-empty.")
        self.network = network
        self.controller_id = canonical_fingerprint(
            {"kind": "bounded-neural-fixed-history-controller", "declared_id": identifier}
        )

    def evaluate(
        self,
        features: ArrayLike,
        accepted: PhaseFieldHistoryState,
        /,
    ) -> FixedHistoryNeuralBlock:
        if not isinstance(accepted, PhaseFieldHistoryState):
            raise TypeError("accepted must be a PhaseFieldHistoryState.")
        features_ = jnp.asarray(features)
        damage_shape = accepted.accepted_damage.shape
        if (
            features_.ndim != len(damage_shape) + 1
            or features_.shape[:-1] != damage_shape
        ):
            raise ValueError(
                "Neural features must append one feature axis to damage layout."
            )
        flat_features = features_.reshape((-1, features_.shape[-1]))
        flat_logits = jax.vmap(self.network)(flat_features)
        if flat_logits.shape not in (
            (flat_features.shape[0],),
            (flat_features.shape[0], 1),
        ):
            raise ValueError(
                "The fixed-history network must return one scalar logit per point."
            )
        logits = jnp.reshape(flat_logits, damage_shape)
        lower = accepted.accepted_damage
        damage = lower + (1.0 - lower) * jax.nn.sigmoid(logits)
        return FixedHistoryNeuralBlock(
            accepted.history,
            damage,
            logits,
            base_state_version=accepted.state_version,
        )


class PhaseFieldFractureModel(StrictModule, NonTrainableState):
    """Shared diffuse-fracture constitutive model and finite-element form factory."""

    parameters: PhaseFieldFractureParameters
    displacement_field: str = eqx.field(static=True)
    damage_field: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: PhaseFieldFractureParameters,
        /,
        *,
        displacement_field: str = "displacement",
        damage_field: str = "damage",
        model_id: str = "phase-field-fracture",
    ):
        if not isinstance(parameters, PhaseFieldFractureParameters):
            raise TypeError("parameters must be PhaseFieldFractureParameters.")
        displacement = str(displacement_field)
        damage = str(damage_field)
        identifier = str(model_id)
        if not displacement or not damage or displacement == damage or not identifier:
            raise ValueError(
                "Phase-field fracture field names and model_id must be distinct and non-empty."
            )
        self.parameters = parameters
        self.displacement_field = displacement
        self.damage_field = damage
        self.model_id = canonical_fingerprint(
            {
                "kind": "phase-field-fracture-model",
                "declared_id": identifier,
                "displacement_field": displacement,
                "damage_field": damage,
                "parameters": {
                    "lame_lambda": float(parameters.lame_lambda),
                    "shear_modulus": float(parameters.shear_modulus),
                    "critical_energy_release_rate": float(
                        parameters.critical_energy_release_rate
                    ),
                    "length_scale": float(parameters.length_scale),
                    "residual_stiffness": float(parameters.residual_stiffness),
                },
            }
        )

    def form(
        self,
        fixed_history: ArrayLike,
        /,
        *,
        form_id: str | None = None,
    ) -> FiniteElementForm:
        history = jnp.asarray(fixed_history)
        if history.ndim != 2:
            raise ValueError(
                "Phase-field fracture history must have cell/quadrature shape."
            )
        parameters = self.parameters

        def equilibrium(
            values, gradients, points, weights, test_basis, test_gradients, context
        ):
            displacement_gradient, _ = gradients
            _, damage = values
            strain = symmetric_gradient(displacement_gradient)
            dimension = strain.shape[-1]
            trace = jnp.trace(strain, axis1=-2, axis2=-1)
            identity = jnp.eye(dimension, dtype=strain.dtype)
            positive_trace = jnp.maximum(trace, 0.0)
            negative_trace = jnp.minimum(trace, 0.0)
            deviator = strain - trace[..., None, None] * identity / dimension
            bulk = parameters.lame_lambda + 2.0 * parameters.shear_modulus / dimension
            stress_plus = (
                bulk * positive_trace[..., None, None] * identity
                + 2.0 * parameters.shear_modulus * deviator
            )
            stress_minus = bulk * negative_trace[..., None, None] * identity
            stress = (
                parameters.degradation(damage)[..., None, None] * stress_plus
                + stress_minus
            )
            return oe.contract("cq,cqib,cqab->cia", weights, test_gradients, stress)

        def damage_residual(
            values, gradients, points, weights, test_basis, test_gradients, context
        ):
            _, damage = values
            _, damage_gradient = gradients
            if history.shape[0] != damage.shape[0] or history.shape[1] not in (
                1,
                damage.shape[1],
            ):
                raise ValueError(
                    "Phase-field history must match damage cells/quadrature."
                )
            history_values = jnp.broadcast_to(history, damage.shape)
            local = (
                parameters.critical_energy_release_rate / parameters.length_scale * damage
                - 2.0 * (1.0 - damage) * history_values
            )
            return (
                oe.contract("cq,cq,qi->ci", weights, local, test_basis)
                + parameters.critical_energy_release_rate
                * parameters.length_scale
                * oe.contract("cq,cqid,cqd->ci", weights, test_gradients, damage_gradient)
            )

        identifier = self.model_id if form_id is None else str(form_id)
        if not identifier:
            raise ValueError("Phase-field fracture form_id must be non-empty.")
        return FiniteElementForm(
            identifier,
            (self.displacement_field, self.damage_field),
            (
                CellResidualAction(
                    self.displacement_field,
                    (self.displacement_field, self.damage_field),
                    equilibrium,
                    action_id="fracture-equilibrium",
                ),
                CellResidualAction(
                    self.damage_field,
                    (self.displacement_field, self.damage_field),
                    damage_residual,
                    action_id="fracture-damage",
                ),
            ),
        )


__all__ = [
    "BoundedNeuralFixedHistoryController",
    "FixedHistoryNeuralBlock",
    "PhaseFieldFractureModel",
    "PhaseFieldFractureParameters",
    "PhaseFieldHistoryState",
    "PhaseFieldHistoryTransaction",
]
