#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import eigen as eigen_linalg
from ._maxwell_frequency import _verified_dense_operator


class MaxwellModeResult(StrictModule):
    propagation_constants: Array
    modes: Array
    residuals: Array
    power_norms: Array
    status: Array
    diagnostics: eigen_linalg.EigenSolveDiagnostics
    result_id: str = eqx.field(static=True)


class TransverseMaxwellModePlan(StrictModule, NonTrainableState):
    """Budgeted generalized Hermitian transverse mode solve."""

    operator: Array
    mass: Array
    mode_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: ArrayLike,
        mass: ArrayLike,
        mode_count: int,
        /,
        *,
        maximum_dofs: int = 4096,
    ):
        operator_ = np.asarray(operator)
        mass_ = np.asarray(mass)
        if operator_.ndim != 2 or operator_.shape[0] != operator_.shape[1]:
            raise ValueError("Mode operator must be square.")
        if mass_.shape != operator_.shape:
            raise ValueError("Mode mass must match operator shape.")
        if operator_.shape[0] > int(maximum_dofs):
            raise ValueError("Mode solve exceeds maximum_dofs.")
        count = int(mode_count)
        if count <= 0 or count > operator_.shape[0]:
            raise ValueError("mode_count is outside the operator dimension.")
        if np.any(~np.isfinite(operator_)) or np.any(~np.isfinite(mass_)):
            raise ValueError("Mode operator/mass must be finite.")
        tolerance = np.finfo(operator_.real.dtype).eps * max(
            1.0, np.linalg.norm(operator_)
        )
        if not np.allclose(
            operator_, operator_.conj().T, rtol=1e-10, atol=64 * tolerance
        ):
            raise ValueError("Mode operator must be Hermitian.")
        if not np.allclose(mass_, mass_.conj().T, rtol=1e-10, atol=64 * tolerance):
            raise ValueError("Mode mass must be Hermitian.")
        if np.linalg.eigvalsh(mass_)[0] <= 64 * tolerance:
            raise ValueError("Mode mass must be positive definite.")
        self.operator = jnp.asarray(operator_)
        self.mass = jnp.asarray(mass_)
        self.mode_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transverse-maxwell-mode-plan",
                "operator": array_tree_fingerprint(operator_),
                "mass": array_tree_fingerprint(mass_),
                "mode_count": count,
            }
        )

    def solve(self, /) -> MaxwellModeResult:
        problem = eigen_linalg.GeneralizedEigenproblem(
            _verified_dense_operator(
                self.operator,
                "transverse Maxwell mode operator",
            ),
            _verified_dense_operator(
                self.mass,
                "transverse Maxwell mode mass",
                positive_definite=True,
            ),
        )
        solved = eigen_linalg.eigensolve(
            problem,
            policy=eigen_linalg.EigenSolvePolicy(
                eigen_linalg.DenseEigh(),
                count=self.mode_count,
                which="largest-algebraic",
            ),
        )
        values = jnp.real(solved.eigenvalues)
        modes = solved.eigenvectors
        power_norms = jnp.real(jnp.sum(jnp.conj(modes) * (self.mass @ modes), axis=0))
        return MaxwellModeResult(
            propagation_constants=jnp.sqrt(jnp.maximum(values, 0.0)),
            modes=modes,
            residuals=solved.diagnostics.residual_norms,
            power_norms=power_norms,
            status=solved.status,
            diagnostics=solved.diagnostics,
            result_id=canonical_fingerprint(
                {
                    "kind": "maxwell-mode-result",
                    "plan": self.plan_id,
                    "eigen_plan": solved.provenance.plan_id,
                }
            ),
        )


class MaxwellModeSource(StrictModule):
    """Charge-compatible modal current source with a scalar time envelope."""

    mode: Array
    envelope: Callable[[Array, Any], ArrayLike] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: ArrayLike,
        envelope: Callable[[Array, Any], ArrayLike],
        /,
    ):
        mode_ = jnp.asarray(mode)
        if mode_.ndim != 1 or not callable(envelope):
            raise TypeError("Mode source requires a vector mode and callable envelope.")
        self.mode = mode_
        self.envelope = envelope
        self.source_id = canonical_fingerprint(
            {"kind": "maxwell-mode-source", "mode": array_tree_fingerprint(mode_)}
        )

    def __call__(self, time: Array, coordinates: Array, args: Any, /) -> Array:
        del coordinates
        amplitude = jnp.asarray(self.envelope(time, args))
        if amplitude.shape != ():
            raise ValueError("Mode source envelope must return a scalar.")
        return amplitude * self.mode


class MaxwellModeDecomposition(StrictModule, NonTrainableState):
    modes: Array
    mass: Array
    decomposition_id: str = eqx.field(static=True)

    def __init__(self, modes: ArrayLike, mass: ArrayLike, /):
        modes_ = jnp.asarray(modes)
        mass_ = jnp.asarray(mass)
        if modes_.ndim != 2 or mass_.shape != (modes_.shape[0], modes_.shape[0]):
            raise ValueError("Mode decomposition shapes are incompatible.")
        gram = jnp.conj(modes_.T) @ mass_ @ modes_
        if not np.allclose(np.asarray(gram), np.eye(modes_.shape[1]), atol=1e-8):
            raise ValueError("Mode basis must be mass-orthonormal.")
        self.modes = modes_
        self.mass = mass_
        self.decomposition_id = canonical_fingerprint(
            {
                "kind": "maxwell-mode-decomposition",
                "modes": array_tree_fingerprint(modes_),
                "mass": array_tree_fingerprint(mass_),
            }
        )

    def amplitudes(self, field: ArrayLike, /) -> Array:
        value = jnp.asarray(field)
        if value.shape != self.modes.shape[:1]:
            raise ValueError("Field shape does not match mode basis.")
        return jnp.conj(self.modes.T) @ (self.mass @ value)


class MaxwellScatteringParameters(StrictModule):
    incident: Array
    reflected: Array
    transmitted: Array
    s11: Array
    s21: Array
    power_balance: Array


def scattering_parameters(
    incident: ArrayLike,
    reflected: ArrayLike,
    transmitted: ArrayLike,
    /,
) -> MaxwellScatteringParameters:
    incident_ = jnp.asarray(incident)
    reflected_ = jnp.asarray(reflected)
    transmitted_ = jnp.asarray(transmitted)
    if incident_.shape != reflected_.shape or incident_.shape != transmitted_.shape:
        raise ValueError("Scattering amplitudes must have matching shapes.")
    safe = jnp.where(jnp.abs(incident_) > 0.0, incident_, 1.0)
    s11 = reflected_ / safe
    s21 = transmitted_ / safe
    valid = jnp.abs(incident_) > 0.0
    s11 = jnp.where(valid, s11, jnp.nan)
    s21 = jnp.where(valid, s21, jnp.nan)
    balance = jnp.abs(s11) ** 2 + jnp.abs(s21) ** 2
    return MaxwellScatteringParameters(
        incident_, reflected_, transmitted_, s11, s21, balance
    )


class MaxwellNearToFarPlan(StrictModule, NonTrainableState):
    """Homogeneous-exterior Huygens surface far-field transform."""

    positions: Array
    normals: Array
    weights: Array
    directions: Array
    wavenumbers: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        normals: ArrayLike,
        weights: ArrayLike,
        directions: ArrayLike,
        wavenumbers: ArrayLike,
        /,
    ):
        positions_ = jnp.asarray(positions, dtype=float)
        normals_ = jnp.asarray(normals, dtype=float)
        weights_ = jnp.asarray(weights, dtype=float)
        directions_ = jnp.asarray(directions, dtype=float)
        wavenumbers_ = jnp.asarray(wavenumbers, dtype=float)
        if positions_.ndim != 2 or positions_.shape[1] != 3:
            raise ValueError("Near-to-far positions must have shape (surface, 3).")
        if normals_.shape != positions_.shape or weights_.shape != positions_.shape[:1]:
            raise ValueError("Near-to-far surface arrays are incompatible.")
        if directions_.ndim != 2 or directions_.shape[1] != 3:
            raise ValueError("Far-field directions must have shape (directions, 3).")
        if wavenumbers_.ndim != 1 or wavenumbers_.size == 0:
            raise ValueError("wavenumbers must be a nonempty vector.")
        normal_norm = jnp.linalg.norm(normals_, axis=1)
        direction_norm = jnp.linalg.norm(directions_, axis=1)
        normals_ = normals_ / normal_norm[:, None]
        directions_ = directions_ / direction_norm[:, None]
        invalid = (
            jnp.any(~jnp.isfinite(positions_))
            | jnp.any(~jnp.isfinite(normals_))
            | jnp.any(~jnp.isfinite(weights_))
            | jnp.any(~jnp.isfinite(directions_))
            | jnp.any(~jnp.isfinite(wavenumbers_))
            | jnp.any(normal_norm <= 0.0)
            | jnp.any(direction_norm <= 0.0)
            | jnp.any(weights_ <= 0.0)
            | jnp.any(wavenumbers_ < 0.0)
        )
        positions_ = eqx.error_if(
            positions_, invalid, "Near-to-far geometry/frequencies are invalid."
        )
        self.positions = positions_
        self.normals = normals_
        self.weights = weights_
        self.directions = directions_
        self.wavenumbers = wavenumbers_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-near-to-far",
                "positions": array_tree_fingerprint(positions_),
                "directions": array_tree_fingerprint(directions_),
                "wavenumbers": array_tree_fingerprint(wavenumbers_),
            }
        )

    def transform(self, electric: ArrayLike, magnetic: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        magnetic_ = jnp.asarray(magnetic)
        if (
            electric_.shape != self.positions.shape
            or magnetic_.shape != self.positions.shape
        ):
            raise ValueError("Near-to-far fields must have shape (surface, 3).")
        electric_current = jnp.cross(self.normals, magnetic_)
        magnetic_current = -jnp.cross(self.normals, electric_)
        phase_argument = self.directions @ self.positions.T
        phase = jnp.exp(
            -1j * self.wavenumbers[:, None, None] * phase_argument[None, :, :]
        )
        projected_electric = electric_current[None, None, :, :]
        projected_magnetic = jnp.cross(
            self.directions[None, :, None, :],
            magnetic_current[None, None, :, :],
        )
        integrand = projected_electric + projected_magnetic
        return jnp.sum(
            phase[..., None] * self.weights[None, None, :, None] * integrand,
            axis=2,
        )


__all__ = [
    "MaxwellModeDecomposition",
    "MaxwellModeResult",
    "MaxwellModeSource",
    "MaxwellNearToFarPlan",
    "MaxwellScatteringParameters",
    "TransverseMaxwellModePlan",
    "scattering_parameters",
]
