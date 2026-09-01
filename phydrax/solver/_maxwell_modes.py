#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import eigen as eigen_linalg
from ._maxwell_frequency import _verified_dense_operator
from ._maxwell_observers import ModeAmplitudeObserverPlan
from ._maxwell_sources import (
    AbstractMaxwellSourcePlan,
    MaxwellPairedCurrentSourcePlan,
    PreparedMaxwellSource,
)


class MaxwellModeResult(StrictModule):
    propagation_constants: Array
    electric_modes: Array
    magnetic_modes: Array
    residuals: Array
    mass_norms: Array
    signed_powers: Array
    status: Array
    diagnostics: eigen_linalg.EigenSolveDiagnostics
    result_id: str = eqx.field(static=True)


class TransverseMaxwellModePlan(StrictModule, NonTrainableState):
    """Budgeted generalized Hermitian transverse mode solve."""

    operator: Array
    mass: Array
    mode_count: int = eqx.field(static=True)
    magnetic_reconstruction: Array
    power_pairing: Array
    propagation_direction: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: ArrayLike,
        mass: ArrayLike,
        mode_count: int,
        /,
        *,
        magnetic_reconstruction: ArrayLike,
        power_pairing: ArrayLike,
        propagation_direction: int = 1,
        maximum_dofs: int = 4096,
    ):
        operator_ = np.asarray(operator)
        mass_ = np.asarray(mass)
        reconstruction = np.asarray(magnetic_reconstruction)
        pairing = np.asarray(power_pairing)
        if operator_.ndim != 2 or operator_.shape[0] != operator_.shape[1]:
            raise ValueError("Mode operator must be square.")
        if mass_.shape != operator_.shape:
            raise ValueError("Mode mass must match operator shape.")
        if reconstruction.ndim != 2 or reconstruction.shape[1] != operator_.shape[0]:
            raise ValueError("Mode magnetic reconstruction has the wrong electric axis.")
        if pairing.shape != (operator_.shape[0], reconstruction.shape[0]):
            raise ValueError("Mode power pairing must map magnetic to electric traces.")
        if propagation_direction not in (-1, 1):
            raise ValueError("Mode propagation_direction must be -1 or +1.")
        maximum = int(maximum_dofs)
        if maximum <= 0 or operator_.shape[0] > maximum:
            raise ValueError("Mode solve exceeds maximum_dofs.")
        count = int(mode_count)
        if count <= 0 or count > operator_.shape[0]:
            raise ValueError("mode_count is outside the operator dimension.")
        if (
            np.any(~np.isfinite(operator_))
            or np.any(~np.isfinite(mass_))
            or np.any(~np.isfinite(reconstruction))
            or np.any(~np.isfinite(pairing))
        ):
            raise ValueError(
                "Mode operator, mass, reconstruction, and pairing must be finite."
            )
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
        self.magnetic_reconstruction = jnp.asarray(reconstruction)
        self.power_pairing = jnp.asarray(pairing)
        self.propagation_direction = int(propagation_direction)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transverse-maxwell-mode-plan",
                "operator": array_tree_fingerprint(operator_),
                "mass": array_tree_fingerprint(mass_),
                "mode_count": count,
                "magnetic_reconstruction": array_tree_fingerprint(reconstruction),
                "power_pairing": array_tree_fingerprint(pairing),
                "propagation_direction": propagation_direction,
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
        electric_modes = solved.eigenvectors
        magnetic_modes = self.propagation_direction * (
            self.magnetic_reconstruction @ electric_modes
        )
        signed_power = 0.5 * jnp.real(
            jnp.sum(
                jnp.conj(electric_modes) * (self.power_pairing @ magnetic_modes),
                axis=0,
            )
        )
        invalid_power = jnp.any(~jnp.isfinite(signed_power)) | jnp.any(
            jnp.abs(signed_power) <= jnp.finfo(signed_power.dtype).eps
        )
        electric_modes = eqx.error_if(
            electric_modes,
            invalid_power,
            "Propagating Maxwell modes require finite nonzero signed power.",
        )
        normalization = jnp.sqrt(jnp.abs(signed_power))
        electric_modes = electric_modes / normalization[None, :]
        magnetic_modes = magnetic_modes / normalization[None, :]
        mass_norms = jnp.real(
            jnp.sum(jnp.conj(electric_modes) * (self.mass @ electric_modes), axis=0)
        )
        return MaxwellModeResult(
            propagation_constants=jnp.sqrt(jnp.maximum(values, 0.0)),
            electric_modes=electric_modes,
            magnetic_modes=magnetic_modes,
            residuals=solved.diagnostics.residual_norms,
            mass_norms=mass_norms,
            signed_powers=jnp.sign(signed_power),
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


class MaxwellHuygensSourcePlan(AbstractMaxwellSourcePlan, NonTrainableState):
    """Orientation-certified paired equivalent currents on one discrete surface."""

    paired: MaxwellPairedCurrentSourcePlan
    direction: int = eqx.field(static=True)
    signed_power: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_indices: ArrayLike,
        magnetic_trace: ArrayLike,
        magnetic_indices: ArrayLike,
        electric_trace: ArrayLike,
        /,
        *,
        signed_power: float,
        direction: int = 1,
        angular_frequency: ArrayLike = 0.0,
        phase: ArrayLike = 0.0,
        amplitude: ArrayLike = 1.0,
        control_key: str | None = None,
        magnetic_closedness_preserving: bool = False,
    ):
        if direction not in (-1, 1):
            raise ValueError("Huygens launch direction must be -1 or +1.")
        power = float(signed_power)
        if not np.isfinite(power) or power == 0.0:
            raise ValueError("Huygens launch requires finite nonzero signed power.")
        identifier = canonical_fingerprint(
            {
                "kind": "maxwell-huygens-source-plan",
                "electric_indices": array_tree_fingerprint(electric_indices),
                "magnetic_indices": array_tree_fingerprint(magnetic_indices),
                "direction": direction,
                "signed_power": power,
            }
        )
        self.paired = MaxwellPairedCurrentSourcePlan(
            electric_indices,
            direction * jnp.asarray(magnetic_trace),
            magnetic_indices,
            -direction * jnp.asarray(electric_trace),
            angular_frequency=angular_frequency,
            phase=phase,
            amplitude=amplitude / np.sqrt(abs(power)),
            control_key=control_key,
            magnetic_closedness_preserving=magnetic_closedness_preserving,
            source_id=identifier,
        )
        self.direction, self.signed_power, self.source_id = (
            int(direction),
            power,
            identifier,
        )

    def prepare(self, bridge, layout, /) -> PreparedMaxwellSource:
        return self.paired.prepare(bridge, layout)


class MaxwellModePortResponse(StrictModule):
    incident: Array
    reflected: Array
    transmitted: Array
    reflection: Array
    transmission: Array
    power_balance: Array


class MaxwellModePortPlan(StrictModule):
    source: MaxwellHuygensSourcePlan
    observer: ModeAmplitudeObserverPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MaxwellHuygensSourcePlan,
        observer: ModeAmplitudeObserverPlan,
        /,
    ):
        if not isinstance(source, MaxwellHuygensSourcePlan) or not isinstance(
            observer, ModeAmplitudeObserverPlan
        ):
            raise TypeError("Mode port requires a paired mode source and modal observer.")
        self.source, self.observer = source, observer
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-mode-port-plan",
                "source": source.source_id,
                "observer": observer.plan_id,
            }
        )

    def response(
        self,
        incident: ArrayLike,
        reflected: ArrayLike,
        transmitted: ArrayLike,
        /,
    ) -> MaxwellModePortResponse:
        incident_, reflected_, transmitted_ = (
            jnp.asarray(value) for value in (incident, reflected, transmitted)
        )
        if incident_.shape != reflected_.shape or incident_.shape != transmitted_.shape:
            raise ValueError("Mode-port amplitude arrays must have matching shapes.")
        valid = jnp.abs(incident_) > 0.0
        safe = jnp.where(valid, incident_, 1.0)
        reflection = jnp.where(valid, reflected_ / safe, jnp.nan)
        transmission = jnp.where(valid, transmitted_ / safe, jnp.nan)
        return MaxwellModePortResponse(
            incident_,
            reflected_,
            transmitted_,
            reflection,
            transmission,
            jnp.abs(reflection) ** 2 + jnp.abs(transmission) ** 2,
        )


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
    "MaxwellHuygensSourcePlan",
    "MaxwellModeDecomposition",
    "MaxwellModePortPlan",
    "MaxwellModePortResponse",
    "MaxwellModeResult",
    "MaxwellNearToFarPlan",
    "TransverseMaxwellModePlan",
]
