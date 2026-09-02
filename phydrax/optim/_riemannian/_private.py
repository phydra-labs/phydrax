#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ...metrix._privacy import RDPLedger, TangentNoiseFrame
from ._parameter_geometry import ParameterGeometry


class PrivateRiemannianStepEvidence(StrictModule):
    per_example_norms: Array
    clipping_scales: Array
    sensitivity: Array
    noise_standard_deviation: Array
    finite: Array
    accepted: Array
    key_fingerprint: Array
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        per_example_norms: Array,
        clipping_scales: Array,
        sensitivity: Array,
        noise_standard_deviation: Array,
        finite: Array,
        accepted: Array,
        key_fingerprint: Array,
        frame_id: str,
    ):
        self.per_example_norms = jnp.asarray(per_example_norms)
        self.clipping_scales = jnp.asarray(clipping_scales)
        self.sensitivity = jnp.asarray(sensitivity)
        self.noise_standard_deviation = jnp.asarray(noise_standard_deviation)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.accepted = jnp.asarray(accepted, dtype=bool)
        self.key_fingerprint = jnp.asarray(key_fingerprint, dtype=jnp.uint32)
        self.frame_id = str(frame_id)


class PrivateRiemannianSGDState(StrictModule):
    step: Array
    ledger: RDPLedger
    evidence: PrivateRiemannianStepEvidence

    def __init__(
        self, step: Array, ledger: RDPLedger, evidence: PrivateRiemannianStepEvidence, /
    ):
        self.step = jnp.asarray(step, dtype=jnp.int32)
        self.ledger = ledger
        self.evidence = evidence


class PrivateRiemannianSGD(StrictModule):
    """Conditional DP-SGD with per-example metric clipping and tangent noise."""

    parameter_geometry: ParameterGeometry
    frame: TangentNoiseFrame
    initial_ledger: RDPLedger
    learning_rate: float = eqx.field(static=True)
    clipping_norm: float = eqx.field(static=True)
    noise_multiplier: float = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_geometry: ParameterGeometry,
        frame: TangentNoiseFrame,
        ledger: RDPLedger,
        /,
        *,
        learning_rate: float,
        clipping_norm: float,
        noise_multiplier: float,
        batch_size: int,
    ):
        if not isinstance(parameter_geometry, ParameterGeometry):
            raise TypeError("parameter_geometry must be a ParameterGeometry.")
        if not isinstance(frame, TangentNoiseFrame) or not isinstance(ledger, RDPLedger):
            raise TypeError("frame and ledger must use GTA privacy evidence types.")
        if (
            min(float(learning_rate), float(clipping_norm), float(noise_multiplier))
            <= 0.0
            or int(batch_size) < 1
        ):
            raise ValueError(
                "learning_rate, clipping_norm, noise_multiplier and batch_size must be positive."
            )
        self.parameter_geometry = parameter_geometry
        self.frame = frame
        self.initial_ledger = ledger
        self.learning_rate = float(learning_rate)
        self.clipping_norm = float(clipping_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.batch_size = int(batch_size)
        self.optimizer_id = "private-riemannian-sgd"

    def init(self, parameters: PyTree, /) -> PrivateRiemannianSGDState:
        self.parameter_geometry.validate(parameters)
        if not bool(self.parameter_geometry.contains(parameters)):
            raise ValueError(
                "Initial private optimizer parameters are outside ParameterGeometry."
            )
        zero = jnp.asarray(0.0)
        evidence = PrivateRiemannianStepEvidence(
            per_example_norms=jnp.zeros((self.batch_size,)),
            clipping_scales=jnp.ones((self.batch_size,)),
            sensitivity=self.clipping_norm / self.batch_size,
            noise_standard_deviation=self.clipping_norm
            * self.noise_multiplier
            / self.batch_size,
            finite=True,
            accepted=True,
            key_fingerprint=jnp.asarray(0, dtype=jnp.uint32),
            frame_id=self.frame.frame_id,
        )
        return PrivateRiemannianSGDState(
            jnp.asarray(0, dtype=jnp.int32), self.initial_ledger, evidence
        )

    def update(
        self,
        per_example_gradients: PyTree,
        state: PrivateRiemannianSGDState,
        parameters: PyTree,
        key: Array,
        /,
    ) -> tuple[PyTree, PrivateRiemannianSGDState]:
        leaves = jax.tree.leaves(per_example_gradients)
        if not leaves or any(leaf.shape[:1] != (self.batch_size,) for leaf in leaves):
            raise ValueError(
                "Every per-example gradient leaf must have the fixed leading batch_size."
            )
        rgradients = jax.vmap(
            lambda gradient: self.parameter_geometry.egrad_to_rgrad(parameters, gradient)
        )(per_example_gradients)
        norms = jax.vmap(
            lambda tangent: self.parameter_geometry.norm(parameters, tangent)
        )(rgradients)
        tiny = jnp.finfo(norms.dtype).tiny
        scales = jnp.minimum(
            1.0,
            jnp.asarray(self.clipping_norm, dtype=norms.dtype) / jnp.maximum(norms, tiny),
        )
        clipped = jax.tree.map(
            lambda leaf: (
                leaf * scales.reshape((self.batch_size,) + (1,) * (leaf.ndim - 1))
            ),
            rgradients,
        )
        mean = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), clipped)
        noise = self.parameter_geometry.project_tangent(
            parameters, self.frame.sample(parameters, key)
        )
        noise_scale = jnp.asarray(
            self.clipping_norm * self.noise_multiplier / self.batch_size,
            dtype=norms.dtype,
        )
        noised = jax.tree.map(
            lambda average, sample: average + noise_scale * sample, mean, noise
        )
        tangent_step = jax.tree.map(
            lambda leaf: -jnp.asarray(self.learning_rate, dtype=leaf.real.dtype) * leaf,
            noised,
        )
        proposed = self.parameter_geometry.retract(parameters, tangent_step)
        finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(proposed))
            )
        )
        accepted = finite & self.parameter_geometry.contains(proposed)
        safe = jax.tree.map(
            lambda old, new: jnp.where(accepted, new, old), parameters, proposed
        )
        sensitivity = jnp.asarray(self.clipping_norm / self.batch_size, dtype=norms.dtype)
        ledger = state.ledger.compose_gaussian(sensitivity, noise_scale)
        evidence = PrivateRiemannianStepEvidence(
            per_example_norms=norms,
            clipping_scales=scales,
            sensitivity=sensitivity,
            noise_standard_deviation=noise_scale,
            finite=finite,
            accepted=accepted,
            key_fingerprint=jnp.bitwise_xor.reduce(jax.random.key_data(key)),
            frame_id=self.frame.frame_id,
        )
        return safe, PrivateRiemannianSGDState(state.step + 1, ledger, evidence)


__all__ = [
    "PrivateRiemannianSGD",
    "PrivateRiemannianSGDState",
    "PrivateRiemannianStepEvidence",
]
