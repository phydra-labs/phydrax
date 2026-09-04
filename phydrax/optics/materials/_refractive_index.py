#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ArtifactManifest


ExtrapolationPolicy: TypeAlias = Literal["reject", "clamp", "continue"]
PassiveBranch: TypeAlias = Literal["positive-imaginary", "as-given"]


class AngularFrequencyValidity(StrictModule, NonTrainableState):
    """Closed angular-frequency interval and its explicit exterior policy.

    Frequencies are angular frequencies in rad/s. ``continue`` means analytic
    continuation for analytic laws and endpoint-linear continuation for a
    tabulated law; it is never an implicit constant extension.
    """

    minimum: Array
    maximum: Array
    extrapolation: ExtrapolationPolicy = eqx.field(static=True)

    def __init__(
        self,
        minimum: ArrayLike,
        maximum: ArrayLike,
        /,
        *,
        extrapolation: ExtrapolationPolicy = "reject",
    ):
        lower = jnp.asarray(minimum)
        upper = jnp.asarray(maximum)
        if lower.ndim != 0 or upper.ndim != 0:
            raise ValueError("Angular-frequency validity bounds must be scalars.")
        if jnp.issubdtype(lower.dtype, jnp.integer):
            lower = lower.astype(jnp.float32)
        elif not jnp.issubdtype(lower.dtype, jnp.floating):
            raise TypeError("minimum must be a real scalar angular frequency.")
        if jnp.issubdtype(upper.dtype, jnp.integer):
            upper = upper.astype(jnp.float32)
        elif not jnp.issubdtype(upper.dtype, jnp.floating):
            raise TypeError("maximum must be a real scalar angular frequency.")
        if not bool(jnp.isfinite(lower) & jnp.isfinite(upper)):
            raise ValueError("Angular-frequency validity bounds must be finite.")
        if not bool((lower > 0) & (upper > lower)):
            raise ValueError("Validity requires 0 < minimum < maximum.")
        if extrapolation not in ("reject", "clamp", "continue"):
            raise ValueError("extrapolation must be 'reject', 'clamp', or 'continue'.")
        self.minimum = lower
        self.maximum = upper
        self.extrapolation = extrapolation


class RefractiveIndexProvenance(StrictModule, NonTrainableState):
    """Identity of one refractive-index record backed by an artifact manifest."""

    manifest: ArtifactManifest
    record_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(self, manifest: ArtifactManifest, /, *, record_id: str):
        if not isinstance(manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest.")
        identifier = str(record_id).strip()
        if not identifier:
            raise ValueError("record_id must be non-empty.")
        self.manifest = manifest
        self.record_id = identifier
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "refractive-index-provenance",
                "manifest_id": manifest.manifest_id,
                "record_id": identifier,
            }
        )


class RefractiveIndexEvaluation(StrictModule):
    """JAX-compatible value, validity status, and branch evidence.

    Status values are 0 for in-range evaluation, 1 for clamped evaluation,
    2 for continued evaluation, 3 for rejected frequency, and 4 for a
    non-finite law value. The time convention is exp(-i omega t), so the
    passive square-root sheet has Im(n) > 0, with Re(n) >= 0 when Im(n) = 0.
    """

    angular_frequency: Array
    evaluated_angular_frequency: Array
    refractive_index: Array
    within_validity: Array
    accepted: Array
    extrapolated: Array
    status: Array
    law_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    extrapolation: ExtrapolationPolicy = eqx.field(static=True)
    passive_branch: PassiveBranch = eqx.field(static=True)


class AbstractRefractiveIndexLaw(StrictModule):
    """Abstract scalar isotropic refractive-index law in angular frequency."""

    validity: AngularFrequencyValidity
    reference_wave_speed: Array
    provenance: RefractiveIndexProvenance
    law_id: str = eqx.field(static=True)
    passive_branch: PassiveBranch = eqx.field(static=True)

    @abc.abstractmethod
    def refractive_index_at(self, angular_frequency: Array, /) -> Array:
        """Evaluate the mathematical law without applying validity policy."""
        raise NotImplementedError


def _common_law_values(
    validity: AngularFrequencyValidity,
    reference_wave_speed: ArrayLike,
    provenance: RefractiveIndexProvenance,
    law_id: str,
    passive_branch: PassiveBranch,
) -> tuple[Array, str, PassiveBranch]:
    if not isinstance(validity, AngularFrequencyValidity):
        raise TypeError("validity must be an AngularFrequencyValidity.")
    speed = jnp.asarray(reference_wave_speed)
    if speed.ndim != 0:
        raise ValueError("reference_wave_speed must be scalar.")
    if jnp.issubdtype(speed.dtype, jnp.integer):
        speed = speed.astype(jnp.float32)
    elif not jnp.issubdtype(speed.dtype, jnp.floating):
        raise TypeError("reference_wave_speed must be real-valued.")
    if not bool(jnp.isfinite(speed) & (speed > 0)):
        raise ValueError("reference_wave_speed must be positive and finite.")
    if not isinstance(provenance, RefractiveIndexProvenance):
        raise TypeError("provenance must be a RefractiveIndexProvenance.")
    identifier = str(law_id).strip()
    if not identifier:
        raise ValueError("law_id must be non-empty.")
    if passive_branch not in ("positive-imaginary", "as-given"):
        raise ValueError("passive_branch must be 'positive-imaginary' or 'as-given'.")
    return speed, identifier, passive_branch


def _real_vector(value: ArrayLike, name: str, /, *, minimum_size: int = 1) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or array.shape[0] < minimum_size:
        raise ValueError(f"{name} must be a rank-one array of sufficient length.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        if jnp.issubdtype(array.dtype, jnp.integer):
            array = array.astype(jnp.float32)
        else:
            raise TypeError(f"{name} must be real-valued.")
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _passive_square_root(value: Array, /) -> Array:
    complex_dtype = jnp.result_type(value.dtype, jnp.complex64)
    root = jnp.sqrt(value.astype(complex_dtype))
    wrong_sheet = (jnp.imag(root) < 0) | ((jnp.imag(root) == 0) & (jnp.real(root) < 0))
    return jnp.where(wrong_sheet, -root, root)


def _on_passive_branch(value: Array, /) -> Array:
    imaginary = jnp.imag(value)
    return (imaginary > 0) | ((imaginary == 0) & (jnp.real(value) >= 0))


class ConstantRefractiveIndex(AbstractRefractiveIndexLaw):
    """Frequency-independent complex scalar index supplied as given."""

    refractive_index: Array

    def __init__(
        self,
        refractive_index: ArrayLike,
        /,
        *,
        validity: AngularFrequencyValidity,
        reference_wave_speed: ArrayLike,
        provenance: RefractiveIndexProvenance,
        law_id: str,
        passive_branch: PassiveBranch = "as-given",
    ):
        speed, identifier, branch = _common_law_values(
            validity, reference_wave_speed, provenance, law_id, passive_branch
        )
        index = jnp.asarray(refractive_index)
        if index.ndim != 0 or not jnp.issubdtype(index.dtype, jnp.number):
            raise TypeError("refractive_index must be a numeric scalar.")
        if not bool(jnp.isfinite(jnp.real(index)) & jnp.isfinite(jnp.imag(index))):
            raise ValueError("refractive_index must be finite.")
        if branch == "positive-imaginary" and not bool(_on_passive_branch(index)):
            raise ValueError(
                "A passive-branch constant index must have Im(n) > 0, "
                "or Im(n) = 0 and Re(n) >= 0."
            )
        self.refractive_index = index
        self.validity = validity
        self.reference_wave_speed = speed
        self.provenance = provenance
        self.law_id = identifier
        self.passive_branch = branch

    def refractive_index_at(self, angular_frequency: Array, /) -> Array:
        return jnp.broadcast_to(self.refractive_index, angular_frequency.shape)


class CauchyRefractiveIndex(AbstractRefractiveIndexLaw):
    """Cauchy law in even powers of a dimensionless inverse wavelength.

    If ``a`` is the coefficient vector and ``s`` is ``wavelength_scale``, then
    n(lambda) = sum_j a[j] (s / lambda) ** (2 j), with
    lambda = 2 pi reference_wave_speed / angular_frequency.
    """

    coefficients: Array
    wavelength_scale: Array

    def __init__(
        self,
        coefficients: ArrayLike,
        wavelength_scale: ArrayLike,
        /,
        *,
        validity: AngularFrequencyValidity,
        reference_wave_speed: ArrayLike,
        provenance: RefractiveIndexProvenance,
        law_id: str,
    ):
        speed, identifier, branch = _common_law_values(
            validity,
            reference_wave_speed,
            provenance,
            law_id,
            "positive-imaginary",
        )
        coefficients_ = _real_vector(coefficients, "coefficients")
        scale = jnp.asarray(wavelength_scale)
        if scale.ndim != 0 or not jnp.issubdtype(scale.dtype, jnp.floating):
            raise TypeError("wavelength_scale must be a real scalar in metres.")
        if not bool(jnp.isfinite(scale) & (scale > 0)):
            raise ValueError("wavelength_scale must be positive and finite.")
        self.coefficients = coefficients_
        self.wavelength_scale = scale
        self.validity = validity
        self.reference_wave_speed = speed
        self.provenance = provenance
        self.law_id = identifier
        self.passive_branch = branch

    def refractive_index_at(self, angular_frequency: Array, /) -> Array:
        wavelength = (
            2.0 * jnp.asarray(jnp.pi) * self.reference_wave_speed / angular_frequency
        )
        inverse_wavelength = self.wavelength_scale / wavelength
        powers = jnp.arange(self.coefficients.shape[0], dtype=wavelength.dtype)
        return jnp.sum(
            self.coefficients * inverse_wavelength[..., None] ** (2.0 * powers),
            axis=-1,
        )


class SellmeierRefractiveIndex(AbstractRefractiveIndexLaw):
    """Sellmeier law with dimensionless strengths and metre resonances."""

    strengths: Array
    resonance_wavelengths: Array

    def __init__(
        self,
        strengths: ArrayLike,
        resonance_wavelengths: ArrayLike,
        /,
        *,
        validity: AngularFrequencyValidity,
        reference_wave_speed: ArrayLike,
        provenance: RefractiveIndexProvenance,
        law_id: str,
    ):
        speed, identifier, branch = _common_law_values(
            validity,
            reference_wave_speed,
            provenance,
            law_id,
            "positive-imaginary",
        )
        strengths_ = _real_vector(strengths, "strengths")
        resonances = _real_vector(resonance_wavelengths, "resonance_wavelengths")
        if strengths_.shape != resonances.shape:
            raise ValueError("strengths and resonance_wavelengths must have equal shape.")
        if not bool(jnp.all(resonances > 0)):
            raise ValueError("resonance_wavelengths must be positive metres.")
        self.strengths = strengths_
        self.resonance_wavelengths = resonances
        self.validity = validity
        self.reference_wave_speed = speed
        self.provenance = provenance
        self.law_id = identifier
        self.passive_branch = branch

    def refractive_index_at(self, angular_frequency: Array, /) -> Array:
        wavelength = (
            2.0 * jnp.asarray(jnp.pi) * self.reference_wave_speed / angular_frequency
        )
        wavelength_squared = wavelength[..., None] ** 2
        resonances_squared = self.resonance_wavelengths**2
        relative_permittivity = 1.0 + jnp.sum(
            self.strengths
            * wavelength_squared
            / (wavelength_squared - resonances_squared),
            axis=-1,
        )
        return _passive_square_root(relative_permittivity)


class LorentzDrudeRefractiveIndex(AbstractRefractiveIndexLaw):
    """Passive-branch Lorentz-Drude scalar index.

    The relative permittivity is epsilon_inf + sum_j strength[j] /
    (resonance[j]**2 - omega**2 - i damping[j] omega). A zero resonance is a
    Drude pole. Strengths have units rad^2/s^2 and all other spectral
    parameters have units rad/s.
    """

    epsilon_infinity: Array
    strengths: Array
    resonance_angular_frequencies: Array
    damping_angular_frequencies: Array

    def __init__(
        self,
        epsilon_infinity: ArrayLike,
        strengths: ArrayLike,
        resonance_angular_frequencies: ArrayLike,
        damping_angular_frequencies: ArrayLike,
        /,
        *,
        validity: AngularFrequencyValidity,
        reference_wave_speed: ArrayLike,
        provenance: RefractiveIndexProvenance,
        law_id: str,
    ):
        speed, identifier, branch = _common_law_values(
            validity,
            reference_wave_speed,
            provenance,
            law_id,
            "positive-imaginary",
        )
        epsilon = jnp.asarray(epsilon_infinity)
        if epsilon.ndim != 0 or not jnp.issubdtype(epsilon.dtype, jnp.floating):
            raise TypeError("epsilon_infinity must be a real scalar.")
        if not bool(jnp.isfinite(epsilon) & (epsilon > 0)):
            raise ValueError("epsilon_infinity must be positive and finite.")
        strengths_ = _real_vector(strengths, "strengths")
        resonances = _real_vector(
            resonance_angular_frequencies, "resonance_angular_frequencies"
        )
        dampings = _real_vector(
            damping_angular_frequencies, "damping_angular_frequencies"
        )
        if strengths_.shape != resonances.shape or strengths_.shape != dampings.shape:
            raise ValueError("Lorentz-Drude parameter arrays must have equal shape.")
        if not bool(
            jnp.all(strengths_ >= 0) & jnp.all(resonances >= 0) & jnp.all(dampings >= 0)
        ):
            raise ValueError(
                "Lorentz-Drude strengths, resonances, and damping must be nonnegative."
            )
        self.epsilon_infinity = epsilon
        self.strengths = strengths_
        self.resonance_angular_frequencies = resonances
        self.damping_angular_frequencies = dampings
        self.validity = validity
        self.reference_wave_speed = speed
        self.provenance = provenance
        self.law_id = identifier
        self.passive_branch = branch

    def refractive_index_at(self, angular_frequency: Array, /) -> Array:
        omega = angular_frequency[..., None]
        denominator = (
            self.resonance_angular_frequencies**2
            - omega**2
            - 1j * self.damping_angular_frequencies * omega
        )
        relative_permittivity = self.epsilon_infinity + jnp.sum(
            self.strengths / denominator, axis=-1
        )
        return _passive_square_root(relative_permittivity)


class TabulatedComplexRefractiveIndex(AbstractRefractiveIndexLaw):
    """Piecewise-linear complex index sampled at increasing angular frequencies."""

    angular_frequencies: Array
    refractive_indices: Array

    def __init__(
        self,
        angular_frequencies: ArrayLike,
        refractive_indices: ArrayLike,
        /,
        *,
        validity: AngularFrequencyValidity,
        reference_wave_speed: ArrayLike,
        provenance: RefractiveIndexProvenance,
        law_id: str,
        passive_branch: PassiveBranch = "as-given",
    ):
        speed, identifier, branch = _common_law_values(
            validity, reference_wave_speed, provenance, law_id, passive_branch
        )
        frequencies = _real_vector(
            angular_frequencies, "angular_frequencies", minimum_size=2
        )
        indices = jnp.asarray(refractive_indices)
        if indices.ndim != 1 or indices.shape != frequencies.shape:
            raise ValueError(
                "refractive_indices must be rank one and match angular_frequencies."
            )
        if not jnp.issubdtype(indices.dtype, jnp.number):
            raise TypeError("refractive_indices must be numeric.")
        if not bool(
            jnp.all(jnp.isfinite(jnp.real(indices)))
            & jnp.all(jnp.isfinite(jnp.imag(indices)))
        ):
            raise ValueError("refractive_indices must be finite.")
        if not bool(jnp.all(frequencies > 0) & jnp.all(jnp.diff(frequencies) > 0)):
            raise ValueError(
                "angular_frequencies must be positive and strictly increasing."
            )
        if not bool(
            (validity.minimum >= frequencies[0]) & (validity.maximum <= frequencies[-1])
        ):
            raise ValueError("validity must lie within the tabulated frequency span.")
        if branch == "positive-imaginary" and not bool(
            jnp.all(_on_passive_branch(indices))
        ):
            raise ValueError(
                "Passive tabulated indices must have Im(n) > 0, "
                "or Im(n) = 0 and Re(n) >= 0."
            )
        self.angular_frequencies = frequencies
        self.refractive_indices = indices
        self.validity = validity
        self.reference_wave_speed = speed
        self.provenance = provenance
        self.law_id = identifier
        self.passive_branch = branch

    def refractive_index_at(self, angular_frequency: Array, /) -> Array:
        right = jnp.searchsorted(
            self.angular_frequencies, angular_frequency, side="right"
        )
        left = jnp.clip(right - 1, 0, self.angular_frequencies.shape[0] - 2)
        omega_left = self.angular_frequencies[left]
        omega_right = self.angular_frequencies[left + 1]
        value_left = self.refractive_indices[left]
        value_right = self.refractive_indices[left + 1]
        fraction = (angular_frequency - omega_left) / (omega_right - omega_left)
        return value_left + fraction * (value_right - value_left)


def evaluate_refractive_index(
    law: AbstractRefractiveIndexLaw,
    angular_frequency: ArrayLike,
    /,
) -> RefractiveIndexEvaluation:
    """Evaluate one law with explicit validity, extrapolation, and branch evidence."""
    if not isinstance(law, AbstractRefractiveIndexLaw):
        raise TypeError("law must be an AbstractRefractiveIndexLaw.")
    omega = jnp.asarray(angular_frequency)
    if not jnp.issubdtype(omega.dtype, jnp.floating):
        if jnp.issubdtype(omega.dtype, jnp.integer):
            omega = omega.astype(
                jnp.result_type(law.reference_wave_speed.dtype, jnp.float32)
            )
        else:
            raise TypeError("angular_frequency must be a real array in rad/s.")
    finite_positive = jnp.isfinite(omega) & (omega > 0)
    within = (
        finite_positive
        & (omega >= law.validity.minimum)
        & (omega <= law.validity.maximum)
    )
    if law.validity.extrapolation == "clamp":
        evaluated_omega = jnp.clip(omega, law.validity.minimum, law.validity.maximum)
        policy_accepts = finite_positive
        exterior_status = jnp.asarray(1, dtype=jnp.int32)
    elif law.validity.extrapolation == "continue":
        evaluated_omega = omega
        policy_accepts = finite_positive
        exterior_status = jnp.asarray(2, dtype=jnp.int32)
    else:
        evaluated_omega = omega
        policy_accepts = within
        exterior_status = jnp.asarray(3, dtype=jnp.int32)
    index = law.refractive_index_at(evaluated_omega)
    index_finite = jnp.isfinite(jnp.real(index)) & jnp.isfinite(jnp.imag(index))
    accepted = policy_accepts & index_finite
    complex_dtype = jnp.result_type(index.dtype, jnp.complex64)
    complex_index = index.astype(complex_dtype)
    rejected_value = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=complex_dtype)
    value = jnp.where(accepted, complex_index, rejected_value)
    status = jnp.where(~finite_positive, 3, jnp.where(within, 0, exterior_status)).astype(
        jnp.int32
    )
    status = jnp.where(policy_accepts & ~index_finite, 4, status).astype(jnp.int32)
    return RefractiveIndexEvaluation(
        angular_frequency=omega,
        evaluated_angular_frequency=evaluated_omega,
        refractive_index=value,
        within_validity=within,
        accepted=accepted,
        extrapolated=accepted & ~within,
        status=status,
        law_id=law.law_id,
        provenance_id=law.provenance.provenance_id,
        extrapolation=law.validity.extrapolation,
        passive_branch=law.passive_branch,
    )


def medium_wavenumber(
    law: AbstractRefractiveIndexLaw,
    angular_frequency: ArrayLike,
    /,
) -> Array:
    """Return k = n(omega) omega / reference_wave_speed in rad/m.

    Rejected evaluations remain complex NaNs; callers needing status should keep
    the corresponding :class:`RefractiveIndexEvaluation`.
    """
    evaluation = evaluate_refractive_index(law, angular_frequency)
    return (
        evaluation.refractive_index
        * evaluation.angular_frequency
        / law.reference_wave_speed
    )


__all__ = [
    "AbstractRefractiveIndexLaw",
    "AngularFrequencyValidity",
    "ConstantRefractiveIndex",
    "CauchyRefractiveIndex",
    "ExtrapolationPolicy",
    "LorentzDrudeRefractiveIndex",
    "PassiveBranch",
    "RefractiveIndexEvaluation",
    "RefractiveIndexProvenance",
    "SellmeierRefractiveIndex",
    "TabulatedComplexRefractiveIndex",
    "evaluate_refractive_index",
    "medium_wavenumber",
]
