#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-mode Gaussian reductions and PPT entanglement diagnostics.

Quadratures use the interleaved ``(q0, p0, q1, p1, ...)`` convention of
:mod:`phydrax.metrix._bosonic_gaussian`.  Logarithms are natural logarithms.
A Gaussian partial transpose flips the momentum of every transposed mode; the
result is diagnostic data and is not represented as a physical Gaussian state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jspecial
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._bosonic_gaussian import BosonicGaussianState, canonical_commutation_matrix


def _mode_tuple(modes: Sequence[int], mode_count: int, name: str, /) -> tuple[int, ...]:
    values = tuple(modes)
    if not values:
        raise ValueError(f"{name} must contain at least one mode.")
    if any(
        isinstance(mode, bool) or not isinstance(mode, (int, np.integer))
        for mode in values
    ):
        raise TypeError(f"{name} must contain integer mode indices.")
    normalized = tuple(values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} cannot contain duplicate modes.")
    if any(mode < 0 or mode >= mode_count for mode in normalized):
        raise ValueError(f"{name} contains a mode outside the declared support.")
    return normalized


def _quadrature_indices(modes: tuple[int, ...], /) -> np.ndarray:
    return np.asarray(
        tuple(index for mode in modes for index in (2 * mode, 2 * mode + 1)),
        dtype=np.int32,
    )


def _square_covariance(covariance: ArrayLike, /) -> Array:
    value = jnp.asarray(covariance)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] % 2:
        raise ValueError("Gaussian covariance must be an even square matrix.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype("float64")
    return value


def _symplectic_eigenvalues(covariance: Array, omega: Array, /) -> Array:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = jnp.linalg.eigvals(1j * omega @ symmetric)
    ordered = jnp.sort(jnp.abs(eigenvalues))
    return jnp.mean(ordered.reshape((-1, 2)), axis=1)


def symplectic_eigenvalues(covariance: ArrayLike, /) -> Array:
    """Return one positive symplectic eigenvalue per interleaved mode."""

    value = _square_covariance(covariance)
    omega = canonical_commutation_matrix(value.shape[0] // 2, dtype=value.real.dtype)
    return _symplectic_eigenvalues(value, omega)


def select_gaussian_subsystem(
    state: BosonicGaussianState,
    modes: Sequence[int],
    /,
) -> BosonicGaussianState:
    """Select and reorder modes without changing the state's hbar convention."""

    if not isinstance(state, BosonicGaussianState):
        raise TypeError("state must be a BosonicGaussianState.")
    selected = _mode_tuple(modes, state.mode_count, "modes")
    indices = jnp.asarray(_quadrature_indices(selected))
    return BosonicGaussianState(
        jnp.take(state.mean, indices, axis=0),
        jnp.take(jnp.take(state.covariance, indices, axis=0), indices, axis=1),
        hbar=state.hbar,
        geometry_precision=state.geometry_precision,
        hermitian_precision=state.hermitian_precision,
    )


def gaussian_partial_transpose(
    covariance: ArrayLike,
    transposed_modes: Sequence[int],
    /,
) -> Array:
    """Apply phase-space partial transpose to a covariance matrix."""

    value = _square_covariance(covariance)
    mode_count = value.shape[0] // 2
    selected = _mode_tuple(transposed_modes, mode_count, "transposed_modes")
    signs = jnp.ones((value.shape[0],), dtype=value.real.dtype)
    signs = signs.at[jnp.asarray(tuple(2 * mode + 1 for mode in selected))].set(-1.0)
    return signs[:, None] * value * signs[None, :]


def gaussian_partial_transpose_mean(
    mean: ArrayLike,
    transposed_modes: Sequence[int],
    /,
) -> Array:
    """Apply the same momentum reflection to Gaussian first moments."""

    value = jnp.asarray(mean)
    if value.ndim != 1 or value.shape[0] % 2:
        raise ValueError("Gaussian mean must be an even rank-one vector.")
    selected = _mode_tuple(transposed_modes, value.shape[0] // 2, "transposed_modes")
    signs = jnp.ones(value.shape, dtype=value.real.dtype)
    signs = signs.at[jnp.asarray(tuple(2 * mode + 1 for mode in selected))].set(-1.0)
    return signs * value


def gaussian_ppt_margins(
    covariance: ArrayLike,
    transposed_modes: Sequence[int],
    /,
    *,
    hbar: float = 1.0,
) -> Array:
    """Return ``nu_tilde - hbar/2`` for every partially transposed mode."""

    hbar_ = float(hbar)
    if not math.isfinite(hbar_) or hbar_ <= 0.0:
        raise ValueError("hbar must be finite and positive.")
    transposed = gaussian_partial_transpose(covariance, transposed_modes)
    return symplectic_eigenvalues(transposed) - 0.5 * hbar_


def gaussian_logarithmic_negativity(
    covariance: ArrayLike,
    transposed_modes: Sequence[int],
    /,
    *,
    hbar: float = 1.0,
) -> Array:
    """Return Gaussian logarithmic negativity in nats."""

    hbar_ = float(hbar)
    if not math.isfinite(hbar_) or hbar_ <= 0.0:
        raise ValueError("hbar must be finite and positive.")
    spectrum = symplectic_eigenvalues(
        gaussian_partial_transpose(covariance, transposed_modes)
    )
    scaled = 2.0 * spectrum / hbar_
    return jnp.sum(jnp.maximum(-jnp.log(scaled), 0.0))


def _entropy_from_physical_spectrum(values: Array, hbar: float, /) -> Array:
    occupation = jnp.maximum(values / hbar - 0.5, 0.0)
    return jnp.sum(
        jspecial.xlogy(occupation + 1.0, occupation + 1.0)
        - jspecial.xlogy(occupation, occupation)
    )


def gaussian_entropy_from_symplectic(
    spectrum: ArrayLike,
    /,
    *,
    hbar: float = 1.0,
    tolerance: float = 1e-9,
) -> Array:
    """Return the von Neumann entropy in nats from physical symplectic values."""

    hbar_ = float(hbar)
    tolerance_ = float(tolerance)
    if not math.isfinite(hbar_) or hbar_ <= 0.0:
        raise ValueError("hbar must be finite and positive.")
    if not math.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    values = jnp.asarray(spectrum)
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("spectrum must be a nonempty rank-one array.")
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | jnp.any(values < 0.5 * hbar_ - tolerance_),
        "Gaussian entropy requires a finite physical symplectic spectrum.",
    )
    return _entropy_from_physical_spectrum(values, hbar_)


def gaussian_entropy(
    covariance: ArrayLike,
    /,
    *,
    hbar: float = 1.0,
    tolerance: float = 1e-9,
) -> Array:
    """Return the finite-mode Gaussian von Neumann entropy in nats."""

    return gaussian_entropy_from_symplectic(
        symplectic_eigenvalues(covariance),
        hbar=hbar,
        tolerance=tolerance,
    )


class GaussianSubsystemPlan(StrictModule, NonTrainableState):
    """Immutable host plan for one bounded finite Gaussian reduction."""

    mode_count: int = eqx.field(static=True)
    modes: tuple[int, ...] = eqx.field(static=True)
    maximum_covariance_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_count: int,
        modes: Sequence[int],
        /,
        *,
        maximum_covariance_elements: int = 1_048_576,
    ):
        if isinstance(mode_count, bool) or not isinstance(mode_count, (int, np.integer)):
            raise TypeError("mode_count must be an integer.")
        count = int(mode_count)
        if count < 1:
            raise ValueError("mode_count must be positive.")
        selected = _mode_tuple(modes, count, "modes")
        if isinstance(maximum_covariance_elements, bool) or not isinstance(
            maximum_covariance_elements, (int, np.integer)
        ):
            raise TypeError("maximum_covariance_elements must be an integer.")
        maximum = int(maximum_covariance_elements)
        required = (2 * len(selected)) ** 2
        if maximum < 1 or required > maximum:
            raise ValueError("Gaussian subsystem exceeds maximum_covariance_elements.")
        self.mode_count = count
        self.modes = selected
        self.maximum_covariance_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-subsystem-plan",
                "mode_count": count,
                "modes": selected,
                "maximum_covariance_elements": maximum,
            }
        )

    def prepare(self, /) -> "PreparedGaussianSubsystem":
        return PreparedGaussianSubsystem(self)


class PreparedGaussianSubsystem(StrictModule, NonTrainableState):
    """Prepared gather indices for repeated JAX-compatible reductions."""

    plan: GaussianSubsystemPlan
    quadrature_indices: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: GaussianSubsystemPlan, /):
        if not isinstance(plan, GaussianSubsystemPlan):
            raise TypeError("plan must be a GaussianSubsystemPlan.")
        indices = _quadrature_indices(plan.modes)
        self.plan = plan
        self.quadrature_indices = jnp.asarray(indices)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-gaussian-subsystem",
                "plan": plan.plan_id,
                "quadrature_indices": tuple(indices),
            }
        )

    def select(self, state: BosonicGaussianState, /) -> BosonicGaussianState:
        if not isinstance(state, BosonicGaussianState):
            raise TypeError("state must be a BosonicGaussianState.")
        if state.mode_count != self.plan.mode_count:
            raise ValueError("Gaussian state mode count does not match the plan.")
        indices = self.quadrature_indices
        return BosonicGaussianState(
            jnp.take(state.mean, indices, axis=0),
            jnp.take(jnp.take(state.covariance, indices, axis=0), indices, axis=1),
            hbar=state.hbar,
            geometry_precision=state.geometry_precision,
            hermitian_precision=state.hermitian_precision,
        )


class GaussianEntanglementReport(StrictModule, NonTrainableState):
    """Runtime values and explicit PPT/physicality status for one reduction."""

    subsystem_mean: Array
    subsystem_covariance: Array
    partial_transpose_mean: Array
    partial_transpose_covariance: Array
    symplectic_eigenvalues: Array
    partial_transpose_symplectic_eigenvalues: Array
    ppt_margins: Array
    minimum_ppt_margin: Array
    logarithmic_negativity: Array
    entropy: Array
    physical: Array
    ppt: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    criterion: str = eqx.field(static=True)


class GaussianEntanglementPlan(StrictModule, NonTrainableState):
    """Plan a selected subsystem and a partial transpose within that subsystem."""

    mode_count: int = eqx.field(static=True)
    subsystem_modes: tuple[int, ...] = eqx.field(static=True)
    transposed_modes: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_covariance_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_count: int,
        subsystem_modes: Sequence[int],
        /,
        *,
        transposed_modes: Sequence[int],
        tolerance: float = 1e-9,
        maximum_covariance_elements: int = 1_048_576,
    ):
        subsystem = GaussianSubsystemPlan(
            mode_count,
            subsystem_modes,
            maximum_covariance_elements=maximum_covariance_elements,
        )
        transposed_global = _mode_tuple(
            transposed_modes, subsystem.mode_count, "transposed_modes"
        )
        if not set(transposed_global).issubset(subsystem.modes):
            raise ValueError("transposed_modes must be contained in subsystem_modes.")
        if len(transposed_global) == len(subsystem.modes):
            raise ValueError(
                "transposed_modes must leave a nonempty complementary partition."
            )
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        self.mode_count = subsystem.mode_count
        self.subsystem_modes = subsystem.modes
        self.transposed_modes = transposed_global
        self.tolerance = tolerance_
        self.maximum_covariance_elements = subsystem.maximum_covariance_elements
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-entanglement-plan",
                "mode_count": self.mode_count,
                "subsystem_modes": self.subsystem_modes,
                "transposed_modes": self.transposed_modes,
                "tolerance": tolerance_,
                "maximum_covariance_elements": self.maximum_covariance_elements,
            }
        )

    def prepare(self, /) -> "PreparedGaussianEntanglement":
        return PreparedGaussianEntanglement(self)


class PreparedGaussianEntanglement(StrictModule, NonTrainableState):
    """Prepared fixed-shape Gaussian reduction and PPT execution."""

    plan: GaussianEntanglementPlan
    quadrature_indices: Array
    transpose_signs: Array
    symplectic_form: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: GaussianEntanglementPlan, /):
        if not isinstance(plan, GaussianEntanglementPlan):
            raise TypeError("plan must be a GaussianEntanglementPlan.")
        indices = _quadrature_indices(plan.subsystem_modes)
        local_lookup = {
            global_mode: local_mode
            for local_mode, global_mode in enumerate(plan.subsystem_modes)
        }
        signs = np.ones((2 * len(plan.subsystem_modes),), dtype=np.float64)
        for global_mode in plan.transposed_modes:
            signs[2 * local_lookup[global_mode] + 1] = -1.0
        self.plan = plan
        self.quadrature_indices = jnp.asarray(indices)
        self.transpose_signs = jnp.asarray(signs)
        self.symplectic_form = canonical_commutation_matrix(len(plan.subsystem_modes))
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-gaussian-entanglement",
                "plan": plan.plan_id,
                "indices": tuple(indices),
                "transpose_signs": tuple(float(value) for value in signs),
            }
        )

    def evaluate(self, state: BosonicGaussianState, /) -> GaussianEntanglementReport:
        """Execute reduction and report physical, PPT, and entropic evidence."""

        if not isinstance(state, BosonicGaussianState):
            raise TypeError("state must be a BosonicGaussianState.")
        if state.mode_count != self.plan.mode_count:
            raise ValueError("Gaussian state mode count does not match the plan.")
        indices = self.quadrature_indices
        mean = jnp.take(state.mean, indices, axis=0)
        covariance = jnp.take(
            jnp.take(state.covariance, indices, axis=0), indices, axis=1
        )
        signs = self.transpose_signs.astype(covariance.real.dtype)
        transposed_mean = signs * mean
        transposed_covariance = signs[:, None] * covariance * signs[None, :]
        omega = self.symplectic_form.astype(covariance.real.dtype)
        physical_spectrum = _symplectic_eigenvalues(covariance, omega)
        transposed_spectrum = _symplectic_eigenvalues(transposed_covariance, omega)
        margins = transposed_spectrum - 0.5 * state.hbar
        minimum = jnp.min(margins)
        scaled = 2.0 * transposed_spectrum / state.hbar
        negativity_value = jnp.sum(jnp.maximum(-jnp.log(scaled), 0.0))
        physical = jnp.asarray(state.valid, dtype=jnp.bool_) & jnp.all(
            physical_spectrum >= 0.5 * state.hbar - self.plan.tolerance
        )
        entropy_value = _entropy_from_physical_spectrum(physical_spectrum, state.hbar)
        logarithmic_negativity = jnp.where(physical, negativity_value, jnp.nan)
        entropy = jnp.where(physical, entropy_value, jnp.nan)
        finite = (
            jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(jnp.isfinite(transposed_spectrum))
            & jnp.isfinite(logarithmic_negativity)
            & jnp.isfinite(entropy)
        )
        ppt = physical & (minimum >= -self.plan.tolerance)
        complementary_count = len(self.plan.subsystem_modes) - len(
            self.plan.transposed_modes
        )
        criterion = (
            "Gaussian-PPT-necessary-and-sufficient-for-one-versus-N-modes"
            if min(len(self.plan.transposed_modes), complementary_count) == 1
            else "Gaussian-PPT-necessary-for-general-bipartitions"
        )
        return GaussianEntanglementReport(
            subsystem_mean=mean,
            subsystem_covariance=covariance,
            partial_transpose_mean=transposed_mean,
            partial_transpose_covariance=transposed_covariance,
            symplectic_eigenvalues=physical_spectrum,
            partial_transpose_symplectic_eigenvalues=transposed_spectrum,
            ppt_margins=margins,
            minimum_ppt_margin=minimum,
            logarithmic_negativity=logarithmic_negativity,
            entropy=entropy,
            physical=physical,
            ppt=ppt,
            valid=finite & physical,
            plan_id=self.plan.plan_id,
            criterion=criterion,
        )


__all__ = [
    "GaussianEntanglementPlan",
    "GaussianEntanglementReport",
    "GaussianSubsystemPlan",
    "PreparedGaussianEntanglement",
    "PreparedGaussianSubsystem",
    "gaussian_entropy",
    "gaussian_entropy_from_symplectic",
    "gaussian_logarithmic_negativity",
    "gaussian_partial_transpose",
    "gaussian_partial_transpose_mean",
    "gaussian_ppt_margins",
    "select_gaussian_subsystem",
    "symplectic_eigenvalues",
]
