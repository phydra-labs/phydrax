# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._strict import StrictModule
from ...._trainable import partition_trainable
from ....ml._numerics import fit_weighted_subspace
from ..._keys import EvalKey, fold_in_eval_key
from ...parameters import ParameterSubspace
from .._port_hamiltonian import PortHamiltonianVectorField


class PortHamiltonianResidualClosure(StrictModule):
    """Compose baseline and residual through energy, skew, PSD, and forcing parts."""

    base: PortHamiltonianVectorField
    residual: PortHamiltonianVectorField
    state_size: int = eqx.field(static=True)
    control_size: int | None = eqx.field(static=True)

    def __init__(
        self, base: PortHamiltonianVectorField, residual: PortHamiltonianVectorField, /
    ):
        if not isinstance(base, PortHamiltonianVectorField) or not isinstance(
            residual, PortHamiltonianVectorField
        ):
            raise TypeError(
                "base and residual must be PortHamiltonianVectorField values."
            )
        if base.state_size != residual.state_size:
            raise ValueError("Baseline and residual state sizes must match.")
        if base.control_size != residual.control_size:
            raise ValueError("Baseline and residual control contracts must match.")
        self.base = base
        self.residual = residual
        self.state_size = base.state_size
        self.control_size = base.control_size

    def _split(self, value: Array | tuple[Array, Array], /) -> tuple[Array, Array | None]:
        state, control = self.base._split_input(value)
        self.residual._split_input(value)
        return state, control

    def energy(self, state: ArrayLike, /, *, key: EvalKey = None) -> Array:
        value = jnp.asarray(state)
        return self.base.energy(
            value, key=fold_in_eval_key(key, 0)
        ) + self.residual.energy(value, key=fold_in_eval_key(key, 1))

    def energy_gradient(self, state: ArrayLike, /, *, key: EvalKey = None) -> Array:
        value = jnp.asarray(state)
        return jax.grad(lambda item: self.energy(item, key=key))(value)

    def interconnection_matrix(
        self, state: ArrayLike, /, *, key: EvalKey = None
    ) -> Array:
        value = jnp.asarray(state)
        return self.base.interconnection_matrix(
            value, key=fold_in_eval_key(key, 2)
        ) + self.residual.interconnection_matrix(value, key=fold_in_eval_key(key, 3))

    def dissipation_matrix(self, state: ArrayLike, /, *, key: EvalKey = None) -> Array:
        value = jnp.asarray(state)
        return self.base.dissipation_matrix(
            value, key=fold_in_eval_key(key, 4)
        ) + self.residual.dissipation_matrix(value, key=fold_in_eval_key(key, 5))

    def forcing_vector(
        self, state: ArrayLike, control: Array | None = None, /, *, key: EvalKey = None
    ) -> Array:
        value = jnp.asarray(state)
        forcing = self.base.forcing_vector(
            value, key=fold_in_eval_key(key, 6)
        ) + self.residual.forcing_vector(value, key=fold_in_eval_key(key, 7))
        if control is not None:
            control_map = self.base.control_map(
                value, key=fold_in_eval_key(key, 8)
            ) + self.residual.control_map(value, key=fold_in_eval_key(key, 9))
            forcing = forcing + contract("ij,j->i", control_map, control)
        return forcing

    def __call__(
        self, value: Array | tuple[Array, Array], /, *, key: EvalKey = None
    ) -> Array:
        state, control = self._split(value)
        gradient = self.energy_gradient(state, key=key)
        skew = self.interconnection_matrix(state, key=key)
        dissipation = self.dissipation_matrix(state, key=key)
        forcing = self.forcing_vector(state, control, key=key)
        return contract("ij,j->i", skew - dissipation, gradient) + forcing

    def energy_balance_residual(
        self, value: Array | tuple[Array, Array], /, *, key: EvalKey = None
    ) -> Array:
        state, control = self._split(value)
        gradient = self.energy_gradient(state, key=key)
        dissipation = self.dissipation_matrix(state, key=key)
        forcing = self.forcing_vector(state, control, key=key)
        rate = jnp.vdot(gradient, self(value, key=key)).real
        loss = jnp.vdot(
            gradient,
            contract("ij,j->i", dissipation, gradient),
        ).real
        power = jnp.vdot(gradient, forcing).real
        return rate + loss - power

    def residual_parameter_subspace(self) -> ParameterSubspace:
        trainable, _ = partition_trainable(self.residual)
        paths = tuple(
            jax.tree_util.keystr(path)
            for path, leaf in jax.tree_util.tree_flatten_with_path(trainable)[0]
            if eqx.is_inexact_array(leaf)
        )
        if not paths:
            raise ValueError("Residual closure has no trainable residual leaves.")
        return ParameterSubspace.from_leaf_paths(self.residual, paths)


class FixedSubspaceProjectionReport(StrictModule):
    """Weighted orthogonality and snapshot-fit evidence for a fixed subspace."""

    retained_energy: Array
    residual_energy: Array
    numerical_rank: Array
    orthogonality_error: Array
    valid: Array


class FixedSubspaceOnsagerModel(StrictModule):
    """Structured latent dynamics restricted to a fixed affine physical subspace."""

    mean: Array
    basis: Array
    metric: Array
    latent_dynamics: Any
    report: FixedSubspaceProjectionReport
    state_size: int = eqx.field(static=True)
    latent_size: int = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        basis: ArrayLike,
        metric: ArrayLike,
        latent_dynamics: Any,
        /,
        *,
        report: FixedSubspaceProjectionReport | None = None,
        orthogonality_tolerance: float = 1.0e-5,
    ):
        mean_array = jnp.asarray(mean)
        basis_array = jnp.asarray(basis)
        metric_array = jnp.asarray(metric)
        if mean_array.ndim != 1 or basis_array.ndim != 2:
            raise ValueError(
                "mean and basis must have shapes (state,) and (state, latent)."
            )
        state_size, latent_size = basis_array.shape
        if mean_array.shape != (state_size,) or metric_array.shape != (
            state_size,
            state_size,
        ):
            raise ValueError("Mean, basis, and metric state dimensions must agree.")
        gram = contract("ia,ij,jb->ab", basis_array, metric_array, basis_array)
        error = jnp.max(jnp.abs(gram - jnp.eye(latent_size, dtype=gram.dtype)))
        if not bool(jnp.isfinite(error)) or float(error) > float(orthogonality_tolerance):
            raise ValueError("Basis is not weighted orthonormal in the declared metric.")
        if report is None:
            report = FixedSubspaceProjectionReport(
                jnp.asarray(1.0),
                jnp.asarray(0.0),
                jnp.asarray(latent_size, dtype=jnp.int32),
                error,
                jnp.asarray(True),
            )
        self.mean = mean_array
        self.basis = basis_array
        self.metric = metric_array
        self.latent_dynamics = latent_dynamics
        self.report = report
        self.state_size = int(state_size)
        self.latent_size = int(latent_size)

    @classmethod
    def fit(
        cls,
        snapshots: ArrayLike,
        latent_size: int,
        latent_dynamics: Any,
        /,
        *,
        sample_weights: ArrayLike | None = None,
    ) -> "FixedSubspaceOnsagerModel":
        values = jnp.asarray(snapshots)
        if values.ndim != 2:
            raise ValueError("snapshots must have shape (sample, state).")
        weights = (
            jnp.ones(values.shape[:-1], dtype=values.real.dtype)
            if sample_weights is None
            else jnp.asarray(sample_weights, dtype=values.real.dtype)
        )
        fit = fit_weighted_subspace(values, weights, rank=int(latent_size), centered=True)
        if not bool(fit.valid):
            raise ValueError("Weighted subspace fitting did not produce a valid basis.")
        basis = jnp.swapaxes(fit.components, -1, -2)
        report = FixedSubspaceProjectionReport(
            fit.retained_energy,
            fit.residual_energy,
            fit.numerical_rank,
            fit.orthogonality_error,
            fit.valid,
        )
        return cls(
            fit.offset,
            basis,
            jnp.eye(values.shape[-1], dtype=values.dtype),
            latent_dynamics,
            report=report,
        )

    def encode(self, state: ArrayLike, /) -> Array:
        centered = jnp.asarray(state) - self.mean
        if centered.shape[-1:] != (self.state_size,):
            raise ValueError("Physical state must end in state_size.")
        return contract(
            "ia,...i->...a", self.basis, contract("ij,...j->...i", self.metric, centered)
        )

    def decode(self, latent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        if value.shape[-1:] != (self.latent_size,):
            raise ValueError("Latent state must end in latent_size.")
        return self.mean + contract("ia,...a->...i", self.basis, value)

    def latent_vector_field(self, latent: ArrayLike, /, *, key: EvalKey = None) -> Array:
        return jnp.asarray(self.latent_dynamics(jnp.asarray(latent), key=key))

    def __call__(self, state: ArrayLike, /, *, key: EvalKey = None) -> Array:
        latent_rate = self.latent_vector_field(self.encode(state), key=key)
        return contract("ia,...a->...i", self.basis, latent_rate)


class AutoencodedOnsagerDiagnostics(StrictModule):
    reconstruction_error: Array
    reconstruction_valid: Array
    latent_guarantees_only: bool = eqx.field(static=True)
    pullback_metric_validated: bool = eqx.field(static=True)


class AutoencodedOnsagerModel(StrictModule):
    """Evolve a structured latent model and decode its observable tangent by JVP."""

    encoder: Any
    decoder: Any
    latent_dynamics: Any
    latent_layout: Any
    reconstruction_tolerance: float = eqx.field(static=True)
    pullback_metric_validated: bool = eqx.field(static=True)

    def __init__(
        self,
        encoder: Any,
        decoder: Any,
        latent_dynamics: Any,
        latent_layout: Any,
        /,
        *,
        reconstruction_tolerance: float,
        pullback_metric_validated: bool = False,
    ):
        tolerance = float(reconstruction_tolerance)
        if not jnp.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("reconstruction_tolerance must be finite and nonnegative.")
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dynamics = latent_dynamics
        self.latent_layout = latent_layout
        self.reconstruction_tolerance = tolerance
        self.pullback_metric_validated = bool(pullback_metric_validated)

    def encode(self, state: ArrayLike, /, *, key: EvalKey = None) -> Array:
        return jnp.asarray(self.encoder(jnp.asarray(state), key=fold_in_eval_key(key, 0)))

    def decode(self, latent: ArrayLike, /, *, key: EvalKey = None) -> Array:
        return jnp.asarray(
            self.decoder(jnp.asarray(latent), key=fold_in_eval_key(key, 1))
        )

    def latent_vector_field(self, latent: ArrayLike, /, *, key: EvalKey = None) -> Array:
        return jnp.asarray(
            self.latent_dynamics(jnp.asarray(latent), key=fold_in_eval_key(key, 2))
        )

    def decoded_tangent(self, latent: ArrayLike, /, *, key: EvalKey = None) -> Array:
        value = jnp.asarray(latent)
        rate = self.latent_vector_field(value, key=key)
        _, tangent = jax.jvp(lambda item: self.decode(item, key=key), (value,), (rate,))
        return tangent

    def __call__(self, state: ArrayLike, /, *, key: EvalKey = None) -> Array:
        latent = self.encode(state, key=key)
        return self.decoded_tangent(latent, key=key)

    def diagnostics(
        self, state: ArrayLike, /, *, key: EvalKey = None
    ) -> AutoencodedOnsagerDiagnostics:
        value = jnp.asarray(state)
        reconstruction = self.decode(self.encode(value, key=key), key=key)
        error = jnp.sqrt(jnp.sum(jnp.abs(reconstruction - value) ** 2))
        valid = jnp.isfinite(error) & (error <= self.reconstruction_tolerance)
        return AutoencodedOnsagerDiagnostics(
            error, valid, True, self.pullback_metric_validated
        )

    def require_reconstruction(
        self, state: ArrayLike, /, *, key: EvalKey = None
    ) -> Array:
        diagnostics = self.diagnostics(state, key=key)
        return eqx.error_if(
            diagnostics.reconstruction_error,
            ~diagnostics.reconstruction_valid,
            "Autoencoded Onsager reconstruction exceeds the declared tolerance.",
        )


__all__ = [
    "AutoencodedOnsagerDiagnostics",
    "AutoencodedOnsagerModel",
    "FixedSubspaceOnsagerModel",
    "FixedSubspaceProjectionReport",
    "PortHamiltonianResidualClosure",
]
