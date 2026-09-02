#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spectral._incompressible import PeriodicLerayProjector
from ...stochastic._ou import OrnsteinUhlenbeckRealization


def _periodic_modal_geometry(
    projector: PeriodicLerayProjector, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    discretization = projector.discretization
    shape = discretization.modal_shape
    modal_size = prod(shape)
    multi_indices = np.indices(shape, dtype=np.int64).reshape((len(shape), -1))
    conjugate_multi = np.stack(
        tuple(
            np.asarray(axis.modes.conjugate_indices, dtype=np.int64)[
                multi_indices[axis_index]
            ]
            for axis_index, axis in enumerate(discretization.axes)
        ),
        axis=0,
    )
    conjugates = np.ravel_multi_index(conjugate_multi, shape)
    if conjugates.shape != (modal_size,):
        raise ValueError("Periodic modal conjugate geometry is inconsistent.")
    magnitude = np.sqrt(np.asarray(projector.wavenumber_squared, dtype=float))
    admissible = np.asarray(projector.admissibility_mask, dtype=bool)
    volume = float(np.prod([float(axis.length) for axis in discretization.axes]))
    if (
        magnitude.shape != shape
        or admissible.shape != shape
        or not np.isfinite(volume)
        or volume <= 0.0
    ):
        raise ValueError("Periodic projector geometry is invalid.")
    return magnitude, admissible, conjugates, volume


def _forced_mask(
    magnitude: np.ndarray,
    admissible: np.ndarray,
    minimum_wavenumber: float,
    maximum_wavenumber: float,
    /,
) -> np.ndarray:
    return (
        admissible
        & (magnitude > 0.0)
        & (magnitude >= minimum_wavenumber)
        & (magnitude <= maximum_wavenumber)
    )


def _hermitian_defect(value: Array, conjugate_indices: Array, /) -> Array:
    flat = value.reshape((-1, value.shape[-1]))
    return jnp.max(
        jnp.abs(flat - jnp.conj(flat[conjugate_indices])),
        initial=0.0,
    )


def _hermitian_projection(value: Array, conjugate_indices: Array, /) -> Array:
    flat = value.reshape((-1, value.shape[-1]))
    projected = 0.5 * (flat + jnp.conj(flat[conjugate_indices]))
    return projected.reshape(value.shape)


def _native_real_inner_product(left: Array, right: Array, /) -> Array:
    return jnp.real(oe.contract("...i,...i->", jnp.conj(left), right))


class ConstantPowerFourierForcingResult(StrictModule):
    forcing: Array
    forced_velocity: Array
    forced_energy: Array
    requested_power_density: Array
    requested_total_power: Array
    actual_power_density: Array
    actual_total_power: Array
    power_defect: Array
    divergence_norm: Array
    input_reality_defect: Array
    forcing_reality_defect: Array
    active: Array
    finite: Array
    successful: Array
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)


class ConstantPowerFourierForcingPlan(StrictModule, NonTrainableState):
    """Prepared low-shell forcing with exact semidiscrete mean power input.

    ``power_input`` is volume-mean power, while ``minimum_forced_energy`` is
    the native total ``0.5 * ||u_F||**2`` activation threshold.
    """

    projector: PeriodicLerayProjector
    forced_mask: Array
    conjugate_indices: Array
    minimum_wavenumber: float = eqx.field(static=True)
    maximum_wavenumber: float = eqx.field(static=True)
    power_input: float = eqx.field(static=True)
    minimum_forced_energy: float = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    reality_tolerance: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    maximum_preparation_bytes: int = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)

    def __init__(
        self,
        projector: PeriodicLerayProjector,
        /,
        *,
        maximum_wavenumber: float,
        power_input: float,
        minimum_forced_energy: float,
        minimum_wavenumber: float = 0.0,
        reality_tolerance: float = 1.0e-10,
        power_tolerance: float = 1.0e-10,
        maximum_preparation_bytes: int = 256 * 1024 * 1024,
    ):
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be a PeriodicLerayProjector.")
        minimum_wave = float(minimum_wavenumber)
        maximum_wave = float(maximum_wavenumber)
        injection = float(power_input)
        minimum_energy = float(minimum_forced_energy)
        reality = float(reality_tolerance)
        power_error = float(power_tolerance)
        maximum_bytes = int(maximum_preparation_bytes)
        if (
            not np.isfinite(minimum_wave)
            or minimum_wave < 0.0
            or not np.isfinite(maximum_wave)
            or maximum_wave <= minimum_wave
            or not np.isfinite(injection)
            or injection <= 0.0
            or not np.isfinite(minimum_energy)
            or minimum_energy <= 0.0
            or not np.isfinite(reality)
            or reality < 0.0
            or not np.isfinite(power_error)
            or power_error < 0.0
            or maximum_bytes <= 0
        ):
            raise ValueError("Constant-power Fourier forcing parameters are invalid.")
        magnitude, admissible, conjugates, volume = _periodic_modal_geometry(projector)
        mask = _forced_mask(
            magnitude,
            admissible,
            minimum_wave,
            maximum_wave,
        )
        if not np.any(mask):
            raise ValueError(
                "The declared forcing interval contains no admissible modes."
            )
        flat_mask = mask.reshape((-1,))
        if np.any(flat_mask != flat_mask[conjugates]):
            raise ValueError("The declared forcing interval is not Hermitian closed.")
        index_dtype = np.dtype(np.int32)
        preparation_bytes = int(mask.nbytes + conjugates.size * index_dtype.itemsize)
        if preparation_bytes > maximum_bytes:
            raise ValueError(
                "Constant-power forcing preparation exceeds maximum_preparation_bytes."
            )
        self.projector = projector
        self.forced_mask = jnp.asarray(mask)
        self.conjugate_indices = jnp.asarray(conjugates, dtype=jnp.int32)
        self.minimum_wavenumber = minimum_wave
        self.maximum_wavenumber = maximum_wave
        self.power_input = injection
        self.minimum_forced_energy = minimum_energy
        self.volume = volume
        self.reality_tolerance = reality
        self.power_tolerance = power_error
        self.preparation_bytes = preparation_bytes
        self.maximum_preparation_bytes = maximum_bytes
        self.discretization_id = projector.discretization.prepared_id
        self.projector_id = projector.projector_id
        self.forcing_id = canonical_fingerprint(
            {
                "kind": "constant-power-fourier-forcing",
                "discretization": self.discretization_id,
                "projector": self.projector_id,
                "minimum_wavenumber": minimum_wave,
                "maximum_wavenumber": maximum_wave,
                "power_input": injection,
                "minimum_forced_energy": minimum_energy,
                "zero_policy": "exclude",
                "nyquist_policy": "exclude-all-axes",
                "reality_tolerance": reality,
                "power_tolerance": power_error,
                "forced_mask": array_tree_fingerprint(mask),
            }
        )

    def evaluate(self, velocity: ArrayLike, /) -> ConstantPowerFourierForcingResult:
        value = self.projector.validate_state(velocity)
        finite_input = jnp.all(jnp.isfinite(value))
        clean = jnp.where(finite_input, value, jnp.zeros_like(value))
        input_reality_defect = _hermitian_defect(clean, self.conjugate_indices)
        admissible_velocity = self.projector.project(clean)
        admissible_velocity = _hermitian_projection(
            admissible_velocity,
            self.conjugate_indices,
        )
        forced_velocity = admissible_velocity * self.forced_mask[..., None]
        norm_squared = _native_real_inner_product(forced_velocity, forced_velocity)
        forced_energy = 0.5 * norm_squared
        input_valid = finite_input & (input_reality_defect <= self.reality_tolerance)
        active = input_valid & (forced_energy >= self.minimum_forced_energy)
        requested_total = jnp.asarray(
            self.volume * self.power_input,
            dtype=forced_velocity.real.dtype,
        )
        safe_norm = jnp.where(norm_squared > 0.0, norm_squared, 1.0)
        scale = requested_total / safe_norm
        candidate = scale.astype(forced_velocity.dtype) * forced_velocity
        forcing = jnp.where(active, candidate, jnp.zeros_like(candidate))
        actual_total = _native_real_inner_product(value, forcing)
        actual_density = actual_total / self.volume
        power_defect = actual_density - self.power_input
        divergence_norm = self.projector.divergence_norm(forcing)
        forcing_reality_defect = _hermitian_defect(forcing, self.conjugate_indices)
        finite = (
            jnp.all(jnp.isfinite(forcing))
            & jnp.isfinite(forced_energy)
            & jnp.isfinite(actual_density)
            & jnp.isfinite(divergence_norm)
            & jnp.isfinite(forcing_reality_defect)
        )
        power_scale = jnp.maximum(jnp.abs(self.power_input), 1.0)
        successful = (
            active
            & finite
            & (jnp.abs(power_defect) <= self.power_tolerance * power_scale)
            & (forcing_reality_defect <= self.reality_tolerance)
        )
        return ConstantPowerFourierForcingResult(
            forcing=forcing,
            forced_velocity=forced_velocity,
            forced_energy=forced_energy,
            requested_power_density=jnp.asarray(
                self.power_input,
                dtype=forced_velocity.real.dtype,
            ),
            requested_total_power=requested_total,
            actual_power_density=actual_density,
            actual_total_power=actual_total,
            power_defect=power_defect,
            divergence_norm=divergence_norm,
            input_reality_defect=input_reality_defect,
            forcing_reality_defect=forcing_reality_defect,
            active=active,
            finite=finite,
            successful=successful,
            discretization_id=self.discretization_id,
            projector_id=self.projector_id,
            forcing_id=self.forcing_id,
        )


class SolenoidalHermitianFourierBasis(StrictModule, NonTrainableState):
    """Orthonormal independent-real basis for selected solenoidal mode pairs."""

    projector: PeriodicLerayProjector
    representative_indices: Array
    partner_indices: Array
    polarizations: Array
    forced_mask: Array
    minimum_wavenumber: float = eqx.field(static=True)
    maximum_wavenumber: float = eqx.field(static=True)
    pair_count: int = eqx.field(static=True)
    polarization_count: int = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    maximum_preparation_bytes: int = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        projector: PeriodicLerayProjector,
        /,
        *,
        maximum_wavenumber: float,
        minimum_wavenumber: float = 0.0,
        maximum_preparation_bytes: int = 256 * 1024 * 1024,
    ):
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be a PeriodicLerayProjector.")
        minimum_wave = float(minimum_wavenumber)
        maximum_wave = float(maximum_wavenumber)
        maximum_bytes = int(maximum_preparation_bytes)
        if (
            not np.isfinite(minimum_wave)
            or minimum_wave < 0.0
            or not np.isfinite(maximum_wave)
            or maximum_wave <= minimum_wave
            or maximum_bytes <= 0
        ):
            raise ValueError("Solenoidal Fourier-basis parameters are invalid.")
        magnitude, admissible, conjugates, volume = _periodic_modal_geometry(projector)
        mask = _forced_mask(
            magnitude,
            admissible,
            minimum_wave,
            maximum_wave,
        )
        flat_mask = mask.reshape((-1,))
        flat_indices = np.arange(flat_mask.size, dtype=np.int64)
        representatives = flat_indices[flat_mask & (flat_indices < conjugates)]
        partners = conjugates[representatives]
        if representatives.size == 0 or np.any(~flat_mask[partners]):
            raise ValueError(
                "The declared interval contains no complete admissible Fourier pairs."
            )
        dimension = projector.spatial_dimension
        polarization_count = dimension - 1
        flat_waves = tuple(
            np.asarray(wave, dtype=float).reshape((-1,)) for wave in projector.wavenumbers
        )
        polarizations = np.empty(
            (representatives.size, polarization_count, dimension),
            dtype=np.asarray(projector.wavenumber_squared).dtype,
        )
        for row, modal_index in enumerate(representatives):
            wave = np.asarray(
                [component[modal_index] for component in flat_waves],
                dtype=float,
            )
            unit = wave / np.linalg.norm(wave)
            if dimension == 2:
                polarizations[row, 0] = np.asarray((-unit[1], unit[0]))
            else:
                pivot = int(np.argmin(np.abs(unit)))
                reference = np.zeros((3,), dtype=float)
                reference[pivot] = 1.0
                first = np.cross(unit, reference)
                first /= np.linalg.norm(first)
                second = np.cross(unit, first)
                second /= np.linalg.norm(second)
                polarizations[row, 0] = first
                polarizations[row, 1] = second
        coordinate_size = int(2 * representatives.size * polarization_count)
        index_dtype = np.dtype(np.int32)
        preparation_bytes = int(
            2 * representatives.size * index_dtype.itemsize
            + polarizations.nbytes
            + mask.nbytes
        )
        if preparation_bytes > maximum_bytes:
            raise ValueError(
                "Solenoidal Fourier basis exceeds maximum_preparation_bytes."
            )
        self.projector = projector
        self.representative_indices = jnp.asarray(representatives, dtype=jnp.int32)
        self.partner_indices = jnp.asarray(partners, dtype=jnp.int32)
        self.polarizations = jnp.asarray(polarizations)
        self.forced_mask = jnp.asarray(mask)
        self.minimum_wavenumber = minimum_wave
        self.maximum_wavenumber = maximum_wave
        self.pair_count = int(representatives.size)
        self.polarization_count = polarization_count
        self.coordinate_size = coordinate_size
        self.volume = volume
        self.preparation_bytes = preparation_bytes
        self.maximum_preparation_bytes = maximum_bytes
        self.discretization_id = projector.discretization.prepared_id
        self.projector_id = projector.projector_id
        self.basis_id = canonical_fingerprint(
            {
                "kind": "solenoidal-hermitian-fourier-basis",
                "discretization": self.discretization_id,
                "projector": self.projector_id,
                "minimum_wavenumber": minimum_wave,
                "maximum_wavenumber": maximum_wave,
                "zero_policy": "exclude",
                "nyquist_policy": "exclude-all-axes",
                "representatives": array_tree_fingerprint(representatives),
                "partners": array_tree_fingerprint(partners),
                "polarizations": array_tree_fingerprint(polarizations),
                "normalization": "native-modal-isometry",
            }
        )

    def evaluate(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != (self.coordinate_size,):
            raise ValueError(
                f"OU coefficients must have shape ({self.coordinate_size},)."
            )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("OU coefficients must be independent real coordinates.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "OU coefficients must be finite.",
        )
        coordinates = values.reshape((self.pair_count, 2, self.polarization_count))
        real = oe.contract("mp,mpd->md", coordinates[:, 0], self.polarizations)
        imaginary = oe.contract("mp,mpd->md", coordinates[:, 1], self.polarizations)
        scale = jnp.sqrt(jnp.asarray(2.0, dtype=values.dtype))
        representatives = jax.lax.complex(real / scale, imaginary / scale)
        dtype = jnp.dtype(self.projector.discretization.plan.precision.coefficient_dtype)
        flat = jnp.zeros(
            (
                prod(self.projector.discretization.modal_shape),
                self.projector.spatial_dimension,
            ),
            dtype=dtype,
        )
        flat = flat.at[self.representative_indices].set(representatives.astype(dtype))
        flat = flat.at[self.partner_indices].set(jnp.conj(representatives).astype(dtype))
        return flat.reshape(self.projector.state_shape)

    def analyze(self, field: ArrayLike, /) -> Array:
        value = self.projector.validate_state(field)
        flat = value.reshape((-1, self.projector.spatial_dimension))
        amplitudes = oe.contract(
            "md,mpd->mp",
            flat[self.representative_indices],
            self.polarizations.astype(value.dtype),
        )
        scale = jnp.sqrt(jnp.asarray(2.0, dtype=amplitudes.real.dtype))
        return jnp.stack(
            (scale * jnp.real(amplitudes), scale * jnp.imag(amplitudes)),
            axis=1,
        ).reshape((-1,))

    def project(self, field: ArrayLike, /) -> Array:
        return self.evaluate(self.analyze(field))


class SolenoidalOUForcingState(StrictModule):
    time: Array
    coefficients: Array
    basis_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)


class SolenoidalOUForcingAdvance(StrictModule):
    state: SolenoidalOUForcingState
    start_coefficients: Array
    half_coefficients: Array
    end_coefficients: Array
    start_forcing: Array
    half_forcing: Array
    end_forcing: Array
    finite: Array
    successful: Array
    basis_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)


class SolenoidalOUForcingPlan(StrictModule, NonTrainableState):
    """Exact-transition OU continuation in an orthonormal real modal basis.

    ``rms_acceleration`` is the stationary expected volume RMS. Coefficients
    retain their exact OU amplitudes; no instantaneous normalization is used.
    """

    basis: SolenoidalHermitianFourierBasis
    correlation_time: float = eqx.field(static=True)
    rms_acceleration: float = eqx.field(static=True)
    coefficient_scale: float = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: SolenoidalHermitianFourierBasis,
        /,
        *,
        correlation_time: float,
        rms_acceleration: float,
    ):
        if not isinstance(basis, SolenoidalHermitianFourierBasis):
            raise TypeError("basis must be a SolenoidalHermitianFourierBasis.")
        correlation = float(correlation_time)
        rms = float(rms_acceleration)
        if (
            not np.isfinite(correlation)
            or correlation <= 0.0
            or not np.isfinite(rms)
            or rms < 0.0
        ):
            raise ValueError("Solenoidal OU forcing parameters are invalid.")
        coefficient_scale = rms * np.sqrt(basis.volume / basis.coordinate_size)
        self.basis = basis
        self.correlation_time = correlation
        self.rms_acceleration = rms
        self.coefficient_scale = float(coefficient_scale)
        self.basis_id = basis.basis_id
        self.discretization_id = basis.discretization_id
        self.projector_id = basis.projector_id
        self.forcing_id = canonical_fingerprint(
            {
                "kind": "solenoidal-ou-fourier-forcing",
                "basis": basis.basis_id,
                "correlation_time": correlation,
                "rms_acceleration": rms,
                "coefficient_scale": coefficient_scale,
                "instantaneous_normalization": "none",
                "transition": "exact-ou",
            }
        )

    def _validate_realization(self, realization: OrnsteinUhlenbeckRealization, /) -> None:
        if not isinstance(realization, OrnsteinUhlenbeckRealization):
            raise TypeError("OU forcing requires OrnsteinUhlenbeckRealization.")
        if realization.sample_shape or realization.noise_shape != (
            self.basis.coordinate_size,
        ):
            raise ValueError("OU realization shape does not match the modal basis.")

    def initialize(
        self,
        time: ArrayLike,
        realization: OrnsteinUhlenbeckRealization,
        /,
        *,
        coefficients: ArrayLike | None = None,
    ) -> SolenoidalOUForcingState:
        self._validate_realization(realization)
        dtype = self.basis.polarizations.dtype
        time_ = jnp.asarray(time, dtype=dtype)
        if time_.shape:
            raise ValueError("OU forcing initialization time must be scalar.")
        values = (
            jnp.zeros((self.basis.coordinate_size,), dtype=dtype)
            if coefficients is None
            else jnp.asarray(coefficients, dtype=dtype)
        )
        if values.shape != (self.basis.coordinate_size,):
            raise ValueError("Initial OU coefficients do not match the modal basis.")
        values = eqx.error_if(
            values,
            ~jnp.isfinite(time_) | jnp.any(~jnp.isfinite(values)),
            "Initial OU continuation state must be finite.",
        )
        return SolenoidalOUForcingState(
            time=time_,
            coefficients=values,
            basis_id=self.basis_id,
            realization_id=realization.realization_id,
            forcing_id=self.forcing_id,
        )

    def evaluate(self, state: SolenoidalOUForcingState, /) -> Array:
        if not isinstance(state, SolenoidalOUForcingState):
            raise TypeError("state must be a SolenoidalOUForcingState.")
        if state.basis_id != self.basis_id or state.forcing_id != self.forcing_id:
            raise ValueError("OU continuation state belongs to another forcing plan.")
        return self.coefficient_scale * self.basis.evaluate(state.coefficients)

    def advance(
        self,
        state: SolenoidalOUForcingState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        realization: OrnsteinUhlenbeckRealization,
        /,
    ) -> SolenoidalOUForcingAdvance:
        self._validate_realization(realization)
        if not isinstance(state, SolenoidalOUForcingState):
            raise TypeError("state must be a SolenoidalOUForcingState.")
        if (
            state.basis_id != self.basis_id
            or state.forcing_id != self.forcing_id
            or state.realization_id != realization.realization_id
        ):
            raise ValueError("OU state, forcing plan, and realization disagree.")
        start = jnp.asarray(start_time, dtype=state.time.dtype)
        end = jnp.asarray(end_time, dtype=state.time.dtype)
        if start.shape or end.shape:
            raise ValueError("OU forcing interval bounds must be scalar.")
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start)
            | ~jnp.isfinite(end)
            | (end < start)
            | (state.time != start),
            "OU forcing interval must continue the exact accepted state time.",
        )
        half = start + 0.5 * (end - start)
        half_coefficients = realization.transition(
            state.coefficients,
            start,
            half,
            jnp.asarray(self.correlation_time, dtype=start.dtype),
        )
        end_coefficients = realization.transition(
            state.coefficients,
            start,
            end,
            jnp.asarray(self.correlation_time, dtype=start.dtype),
        )
        start_forcing = self.coefficient_scale * self.basis.evaluate(state.coefficients)
        half_forcing = self.coefficient_scale * self.basis.evaluate(half_coefficients)
        end_forcing = self.coefficient_scale * self.basis.evaluate(end_coefficients)
        finite = (
            jnp.all(jnp.isfinite(half_coefficients))
            & jnp.all(jnp.isfinite(end_coefficients))
            & jnp.all(jnp.isfinite(start_forcing))
            & jnp.all(jnp.isfinite(half_forcing))
            & jnp.all(jnp.isfinite(end_forcing))
        )
        next_state = SolenoidalOUForcingState(
            time=end,
            coefficients=end_coefficients,
            basis_id=self.basis_id,
            realization_id=realization.realization_id,
            forcing_id=self.forcing_id,
        )
        return SolenoidalOUForcingAdvance(
            state=next_state,
            start_coefficients=state.coefficients,
            half_coefficients=half_coefficients,
            end_coefficients=end_coefficients,
            start_forcing=start_forcing,
            half_forcing=half_forcing,
            end_forcing=end_forcing,
            finite=finite,
            successful=finite,
            basis_id=self.basis_id,
            realization_id=realization.realization_id,
            forcing_id=self.forcing_id,
        )


__all__ = [
    "ConstantPowerFourierForcingPlan",
    "ConstantPowerFourierForcingResult",
    "SolenoidalHermitianFourierBasis",
    "SolenoidalOUForcingAdvance",
    "SolenoidalOUForcingPlan",
    "SolenoidalOUForcingState",
]
