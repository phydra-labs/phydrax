#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""3+1 Weyl extraction, spin-weighted waves, strain, and radial extrapolation."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein
import phydrax.linalg as la

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._spectral._spherical import (
    SphericalExecution,
    SphericalHarmonicPlan,
    SphericalSampling,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._spacetime_conventions import RelativityConvention


_METRIC_SOLVE = la.SmallLinearSolvePlan(3)
_LEVI_CIVITA_SYMBOL = jnp.asarray(
    (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
)


class WaveConvergenceEvidence(StrictModule):
    absolute_error: Array
    relative_error: Array
    tolerance: Array
    finite: Array
    converged: Array


class WeylCurvatureEvidence(StrictModule):
    electric: Array
    magnetic: Array
    electric_trace_error: Array
    magnetic_trace_error: Array
    electric_symmetry_error: Array
    magnetic_symmetry_error: Array
    metric_inversion_valid: Array
    finite: Array
    physically_valid: Array
    derivative_valid: Array
    convention_id: str = eqx.field(static=True)


class Psi4TetradConvention(StrictModule, NonTrainableState):
    """A null-tetrad convention layered on the shared spacetime convention.

    The dyad is ``m=(e_theta+i*dyad_orientation*e_phi)/sqrt(2)`` and this module
    defines ``Psi4=psi4_sign*(E-i*magnetic_sign*B)_ij conj(m)^i conj(m)^j``.
    """

    relativity: RelativityConvention
    dyad_orientation: int = eqx.field(static=True)
    psi4_sign: int = eqx.field(static=True)
    magnetic_sign: int = eqx.field(static=True)
    strain_polarization: Literal["h_plus-minus-i-h_cross"] = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(
        self,
        relativity: RelativityConvention | None = None,
        /,
        *,
        dyad_orientation: int = 1,
        psi4_sign: int = 1,
        magnetic_sign: int = 1,
    ):
        relativity_ = RelativityConvention() if relativity is None else relativity
        if not isinstance(relativity_, RelativityConvention):
            raise TypeError("relativity must be a RelativityConvention or None.")
        values = (int(dyad_orientation), int(psi4_sign), int(magnetic_sign))
        if any(value not in (-1, 1) for value in values):
            raise ValueError("Tetrad orientation and sign choices must be +1 or -1.")
        self.relativity = relativity_
        self.dyad_orientation, self.psi4_sign, self.magnetic_sign = values
        self.strain_polarization = "h_plus-minus-i-h_cross"
        self.convention_id = canonical_fingerprint(
            {
                "kind": "psi4-tetrad-convention",
                "relativity": relativity_.convention_id,
                "dyad_orientation": values[0],
                "psi4_sign": values[1],
                "magnetic_sign": values[2],
                "strain_polarization": self.strain_polarization,
            }
        )


class SpinWeightedMultipoles(StrictModule):
    coefficients: Array
    valid_modes: Array
    reconstruction: Array
    convergence: WaveConvergenceEvidence
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class SpinWeightedMultipolePlan(StrictModule, NonTrainableState):
    """Prepared exact-sampling transform with a fixed ``(ell,m)`` capacity."""

    transform: SphericalHarmonicPlan
    degrees: Array
    orders: Array
    valid_modes: Array
    solid_angle_weights: Array
    spin: int = eqx.field(static=True)
    bandlimit: int = eqx.field(static=True)
    reconstruction_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bandlimit: int,
        /,
        *,
        spin: int = -2,
        sampling: SphericalSampling = "gl",
        execution: SphericalExecution = "recursive",
        reconstruction_tolerance: float = 1.0e-8,
        max_precompute_bytes: int = 512 * 1024**2,
    ):
        bandlimit_ = int(bandlimit)
        spin_ = int(spin)
        tolerance = float(reconstruction_tolerance)
        if tolerance <= 0.0:
            raise ValueError("Multipole reconstruction tolerance must be positive.")
        transform = SphericalHarmonicPlan(
            bandlimit_,
            sampling=sampling,
            spin=spin_,
            reality=False,
            execution=execution,
            max_precompute_bytes=max_precompute_bytes,
        )
        degree = np.arange(bandlimit_, dtype=np.int32)[:, None]
        order = np.arange(-(bandlimit_ - 1), bandlimit_, dtype=np.int32)[None, :]
        shape = transform.coefficient_shape
        degrees = np.broadcast_to(degree, shape)
        orders = np.broadcast_to(order, shape)
        valid = (np.abs(orders) <= degrees) & (degrees >= abs(spin_))
        weights = np.asarray(transform.theta_quadrature_weights)[:, None] * np.asarray(
            transform.phi_quadrature_weights
        )[None, :]
        self.transform = transform
        self.degrees = jnp.asarray(degrees)
        self.orders = jnp.asarray(orders)
        self.valid_modes = jnp.asarray(valid)
        self.solid_angle_weights = jnp.asarray(weights)
        self.spin = spin_
        self.bandlimit = bandlimit_
        self.reconstruction_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spin-weighted-multipole-plan",
                "transform": transform.transform_id,
                "reconstruction_tolerance": tolerance,
            }
        )

    def analyze(self, samples: ArrayLike, /) -> SpinWeightedMultipoles:
        values = jnp.asarray(samples)
        if values.shape[-2:] != self.transform.sample_shape:
            raise ValueError("Spin-weighted samples do not match transform capacity.")
        coefficients = self.transform.analysis(values)
        mask_shape = (1,) * (coefficients.ndim - 2) + self.valid_modes.shape
        coefficients = jnp.where(self.valid_modes.reshape(mask_shape), coefficients, 0.0)
        reconstruction = self.transform.synthesis(coefficients)
        difference = reconstruction - values
        weight_shape = (1,) * (values.ndim - 2) + self.solid_angle_weights.shape
        weights = self.solid_angle_weights.reshape(weight_shape)
        absolute_error = jnp.sqrt(jnp.sum(weights * jnp.abs(difference) ** 2))
        norm = jnp.sqrt(jnp.sum(weights * jnp.abs(values) ** 2))
        relative_error = absolute_error / jnp.maximum(
            norm, jnp.finfo(values.real.dtype).tiny
        )
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(coefficients))
            & jnp.isfinite(relative_error)
        )
        converged = finite & (relative_error <= self.reconstruction_tolerance)
        evidence = WaveConvergenceEvidence(
            absolute_error,
            relative_error,
            jnp.asarray(self.reconstruction_tolerance, dtype=relative_error.dtype),
            finite,
            converged,
        )
        return SpinWeightedMultipoles(
            coefficients,
            self.valid_modes,
            reconstruction,
            evidence,
            finite,
            converged,
            finite,
            converged,
            converged,
            self.plan_id,
        )

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape[-2:] != self.transform.coefficient_shape:
            raise ValueError("Multipole coefficients do not match transform capacity.")
        mask_shape = (1,) * (values.ndim - 2) + self.valid_modes.shape
        return self.transform.synthesis(
            jnp.where(self.valid_modes.reshape(mask_shape), values, 0.0)
        )


class Psi4ExtractionResult(StrictModule):
    psi4: Array
    multipoles: SpinWeightedMultipoles
    tetrad_orthonormality_error: Array
    tetrad_orientation: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)


class Psi4ExtractionPlan(StrictModule, NonTrainableState):
    multipole_plan: SpinWeightedMultipolePlan
    convention: Psi4TetradConvention
    tetrad_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        multipole_plan: SpinWeightedMultipolePlan,
        convention: Psi4TetradConvention | None = None,
        /,
        *,
        tetrad_tolerance: float = 1.0e-7,
    ):
        if not isinstance(multipole_plan, SpinWeightedMultipolePlan) or multipole_plan.spin != -2:
            raise TypeError("Psi4 extraction requires a spin -2 multipole plan.")
        convention_ = Psi4TetradConvention() if convention is None else convention
        if not isinstance(convention_, Psi4TetradConvention):
            raise TypeError("convention must be a Psi4TetradConvention or None.")
        tolerance = float(tetrad_tolerance)
        if tolerance <= 0.0:
            raise ValueError("tetrad_tolerance must be positive.")
        self.multipole_plan = multipole_plan
        self.convention = convention_
        self.tetrad_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "psi4-extraction-plan",
                "multipoles": multipole_plan.plan_id,
                "convention": convention_.convention_id,
                "tetrad_tolerance": tolerance,
            }
        )

    def extract(
        self,
        curvature: WeylCurvatureEvidence,
        radial_leg: ArrayLike,
        theta_leg: ArrayLike,
        phi_leg: ArrayLike,
        /,
        *,
        spatial_metric: ArrayLike | None = None,
    ) -> Psi4ExtractionResult:
        radial = jnp.asarray(radial_leg)
        theta = jnp.asarray(theta_leg, dtype=radial.dtype)
        phi = jnp.asarray(phi_leg, dtype=radial.dtype)
        sample_shape = self.multipole_plan.transform.sample_shape
        expected = sample_shape + (3,)
        if radial.shape != expected or theta.shape != expected or phi.shape != expected:
            raise ValueError("Tetrad legs must have the spherical sample shape plus (3,).")
        if curvature.electric.shape != sample_shape + (3, 3):
            raise ValueError("Weyl samples do not match the multipole grid.")
        metric = (
            jnp.broadcast_to(jnp.eye(3, dtype=radial.dtype), sample_shape + (3, 3))
            if spatial_metric is None
            else jnp.asarray(spatial_metric, dtype=radial.dtype)
        )
        if metric.shape != sample_shape + (3, 3):
            raise ValueError("Spatial metric does not match the extraction sphere.")
        frame = jnp.stack((radial, theta, phi), axis=-2)
        gram = ein.contract("...ai,...ij,...bj->...ab", frame, metric, frame)
        identity = jnp.eye(3, dtype=gram.dtype)
        orthonormality_error = jnp.max(jnp.abs(gram - identity))
        orientation = la.determinant_small_linear(_METRIC_SOLVE, frame)
        dyad = (
            theta + 1j * float(self.convention.dyad_orientation) * phi
        ) / jnp.sqrt(2.0)
        radiative_weyl = curvature.electric - 1j * float(
            self.convention.magnetic_sign
        ) * curvature.magnetic
        psi4 = float(self.convention.psi4_sign) * ein.contract(
            "...i,...ij,...j->...", jnp.conj(dyad), radiative_weyl, jnp.conj(dyad)
        )
        multipoles = self.multipole_plan.analyze(psi4)
        finite = (
            curvature.finite
            & jnp.all(jnp.isfinite(frame))
            & jnp.all(jnp.isfinite(psi4))
            & jnp.isfinite(orthonormality_error)
        )
        tetrad_valid = (
            orthonormality_error <= self.tetrad_tolerance
        ) & jnp.all(
            orientation
            * float(self.convention.relativity.azimuthal_orientation)
            > 0.0
        )
        physically_valid = curvature.physically_valid & finite & tetrad_valid
        qualified = physically_valid & multipoles.qualified
        return Psi4ExtractionResult(
            psi4,
            multipoles,
            orthonormality_error,
            jnp.min(orientation),
            finite,
            physically_valid,
            qualified,
            qualified & curvature.derivative_valid & multipoles.derivative_valid,
            self.plan_id,
            self.convention.convention_id,
        )


class StrainIntegrationResult(StrictModule):
    strain: Array
    strain_spectrum: Array
    filtered_psi4: Array
    reconstructed_psi4: Array
    convergence: WaveConvergenceEvidence
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class FixedFrequencyStrainPlan(StrictModule, NonTrainableState):
    """Twice integrate Psi4 with an explicit fixed-frequency high-pass filter."""

    angular_frequencies: Array
    passband: Array
    sample_count: int = eqx.field(static=True)
    sample_interval: float = eqx.field(static=True)
    low_frequency_cutoff: float = eqx.field(static=True)
    reconstruction_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int,
        sample_interval: float,
        low_frequency_cutoff: float,
        /,
        *,
        reconstruction_tolerance: float = 1.0e-8,
    ):
        count = int(sample_count)
        interval = float(sample_interval)
        cutoff = float(low_frequency_cutoff)
        tolerance = float(reconstruction_tolerance)
        nyquist = np.pi / interval if interval > 0.0 else 0.0
        if (
            count < 4
            or interval <= 0.0
            or cutoff <= 0.0
            or cutoff >= nyquist
            or tolerance <= 0.0
        ):
            raise ValueError("Strain integration capacity, frequencies, or tolerance are invalid.")
        omega = 2.0 * np.pi * np.fft.fftfreq(count, d=interval)
        passband = np.abs(omega) >= cutoff
        self.angular_frequencies = jnp.asarray(omega)
        self.passband = jnp.asarray(passband)
        self.sample_count = count
        self.sample_interval = interval
        self.low_frequency_cutoff = cutoff
        self.reconstruction_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-frequency-strain-integration",
                "sample_count": count,
                "sample_interval": interval,
                "low_frequency_cutoff": cutoff,
                "reconstruction_tolerance": tolerance,
            }
        )

    def integrate(self, psi4: ArrayLike, /) -> StrainIntegrationResult:
        values = jnp.asarray(psi4)
        if values.shape[0] != self.sample_count:
            raise ValueError("Psi4 time axis does not match integration capacity.")
        spectrum = jnp.fft.fft(values, axis=0)
        trailing = (1,) * (values.ndim - 1)
        omega = self.angular_frequencies.reshape((self.sample_count,) + trailing)
        passband = self.passband.reshape((self.sample_count,) + trailing)
        safe_omega_squared = jnp.where(passband, omega**2, 1.0)
        filtered_spectrum = jnp.where(passband, spectrum, 0.0)
        strain_spectrum = -filtered_spectrum / safe_omega_squared
        strain = jnp.fft.ifft(strain_spectrum, axis=0)
        filtered_psi4 = jnp.fft.ifft(filtered_spectrum, axis=0)
        reconstructed = jnp.fft.ifft(-omega**2 * strain_spectrum, axis=0)
        difference = reconstructed - filtered_psi4
        absolute_error = jnp.sqrt(jnp.sum(jnp.abs(difference) ** 2))
        norm = jnp.sqrt(jnp.sum(jnp.abs(filtered_psi4) ** 2))
        relative_error = absolute_error / jnp.maximum(
            norm, jnp.finfo(values.real.dtype).tiny
        )
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(strain))
            & jnp.isfinite(relative_error)
        )
        converged = finite & (relative_error <= self.reconstruction_tolerance)
        evidence = WaveConvergenceEvidence(
            absolute_error,
            relative_error,
            jnp.asarray(self.reconstruction_tolerance, dtype=relative_error.dtype),
            finite,
            converged,
        )
        return StrainIntegrationResult(
            strain,
            strain_spectrum,
            filtered_psi4,
            reconstructed,
            evidence,
            finite,
            converged,
            finite,
            converged,
            converged,
            self.plan_id,
        )


class FiniteRadiusExtrapolationResult(StrictModule):
    asymptotic_waveform: Array
    radial_coefficients: Array
    fitted_scaled_waveforms: Array
    fit_convergence: WaveConvergenceEvidence
    order_convergence: WaveConvergenceEvidence
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class FiniteRadiusExtrapolationPlan(StrictModule, NonTrainableState):
    """Polynomial ``1/r`` extrapolation of a peeling-scaled radiative field."""

    radii: Array
    design: Array
    projection: Array
    lower_order_projection: Array
    order: int = eqx.field(static=True)
    radial_power: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    radius_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radii: ArrayLike,
        order: int,
        /,
        *,
        radial_power: float = 1.0,
        convergence_tolerance: float = 5.0e-3,
    ):
        radii_ = np.asarray(radii, dtype=float).reshape((-1,))
        order_ = int(order)
        radial_power_ = float(radial_power)
        tolerance = float(convergence_tolerance)
        if (
            order_ < 1
            or radii_.size < order_ + 2
            or np.any(~np.isfinite(radii_))
            or np.any(radii_ <= 0.0)
            or np.any(np.diff(radii_) <= 0.0)
            or not np.isfinite(radial_power_)
            or tolerance <= 0.0
        ):
            raise ValueError("Finite-radius extrapolation support or tolerance is invalid.")
        inverse_radius = 1.0 / radii_
        design = np.stack(tuple(inverse_radius**power for power in range(order_ + 1)), axis=-1)
        projection, _, rank, _ = np.linalg.lstsq(
            design, np.eye(radii_.size), rcond=None
        )
        lower_design = design[:, :order_]
        lower_projection, _, lower_rank, _ = np.linalg.lstsq(
            lower_design, np.eye(radii_.size), rcond=None
        )
        if rank != order_ + 1 or lower_rank != order_:
            raise ValueError("Finite-radius extrapolation design is rank deficient.")
        self.radii = jnp.asarray(radii_)
        self.design = jnp.asarray(design)
        self.projection = jnp.asarray(projection)
        self.lower_order_projection = jnp.asarray(lower_projection)
        self.order = order_
        self.radial_power = radial_power_
        self.convergence_tolerance = tolerance
        self.radius_capacity = int(radii_.size)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-radius-wave-extrapolation",
                "radii": array_tree_fingerprint(radii_),
                "order": order_,
                "radial_power": radial_power_,
                "convergence_tolerance": tolerance,
            }
        )

    def extrapolate(
        self, finite_radius_waveforms: ArrayLike, /
    ) -> FiniteRadiusExtrapolationResult:
        values = jnp.asarray(finite_radius_waveforms)
        if values.shape[0] != self.radius_capacity:
            raise ValueError("Waveform radius axis does not match extrapolation capacity.")
        scale_shape = (self.radius_capacity,) + (1,) * (values.ndim - 1)
        scaled = values * self.radii.reshape(scale_shape) ** self.radial_power
        coefficients = ein.contract("kr,r...->k...", self.projection, scaled)
        fitted = ein.contract("rk,k...->r...", self.design, coefficients)
        asymptotic = coefficients[0]
        lower_coefficients = ein.contract(
            "kr,r...->k...", self.lower_order_projection, scaled
        )
        lower_asymptotic = lower_coefficients[0]

        fit_difference = fitted - scaled
        fit_absolute = jnp.sqrt(jnp.sum(jnp.abs(fit_difference) ** 2))
        data_norm = jnp.sqrt(jnp.sum(jnp.abs(scaled) ** 2))
        fit_relative = fit_absolute / jnp.maximum(
            data_norm, jnp.finfo(values.real.dtype).tiny
        )
        order_absolute = jnp.sqrt(jnp.sum(jnp.abs(asymptotic - lower_asymptotic) ** 2))
        asymptotic_norm = jnp.sqrt(jnp.sum(jnp.abs(asymptotic) ** 2))
        order_relative = order_absolute / jnp.maximum(
            asymptotic_norm, jnp.finfo(values.real.dtype).tiny
        )
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(coefficients))
            & jnp.isfinite(fit_relative)
            & jnp.isfinite(order_relative)
        )
        fit_converged = finite & (fit_relative <= self.convergence_tolerance)
        order_converged = finite & (order_relative <= self.convergence_tolerance)
        fit_evidence = WaveConvergenceEvidence(
            fit_absolute,
            fit_relative,
            jnp.asarray(self.convergence_tolerance, dtype=fit_relative.dtype),
            finite,
            fit_converged,
        )
        order_evidence = WaveConvergenceEvidence(
            order_absolute,
            order_relative,
            jnp.asarray(self.convergence_tolerance, dtype=order_relative.dtype),
            finite,
            order_converged,
        )
        converged = fit_converged & order_converged
        return FiniteRadiusExtrapolationResult(
            asymptotic,
            coefficients,
            fitted,
            fit_evidence,
            order_evidence,
            finite,
            converged,
            finite,
            converged,
            converged,
            self.plan_id,
        )


def vacuum_weyl_curvature(
    spatial_metric: ArrayLike,
    spatial_ricci: ArrayLike,
    extrinsic_curvature: ArrayLike,
    covariant_derivative_extrinsic_curvature: ArrayLike,
    /,
    *,
    convention: RelativityConvention | None = None,
) -> WeylCurvatureEvidence:
    """Construct vacuum electric/magnetic Weyl tensors from one 3+1 slice.

    ``covariant_derivative_extrinsic_curvature[..., k, l, j]`` is ``D_k K_lj``.
    Input ``K`` follows the shared convention and is converted internally to the
    standard ``K=-L_n gamma/2`` convention before the magnetic projection.
    """
    convention_ = RelativityConvention() if convention is None else convention
    if not isinstance(convention_, RelativityConvention):
        raise TypeError("convention must be a RelativityConvention or None.")
    metric = jnp.asarray(spatial_metric)
    ricci = jnp.asarray(spatial_ricci, dtype=metric.dtype)
    curvature = jnp.asarray(extrinsic_curvature, dtype=metric.dtype)
    derivative = jnp.asarray(
        covariant_derivative_extrinsic_curvature, dtype=metric.dtype
    )
    incompatible_tensors = (
        metric.shape[-2:] != (3, 3)
        or ricci.shape != metric.shape
        or curvature.shape != metric.shape
    )
    if incompatible_tensors:
        raise ValueError(
            "Metric, Ricci, and extrinsic-curvature shapes must end in (3, 3)."
        )
    if derivative.shape != metric.shape[:-2] + (3, 3, 3):
        raise ValueError("Covariant K derivative must end in (3, 3, 3).")
    inverse_result = la.inverse_small_linear(_METRIC_SOLVE, metric)
    inverse_metric = inverse_result.value
    trace_curvature = ein.contract("...ij,...ij->...", inverse_metric, curvature)
    mixed_curvature = ein.contract("...ik,...kj->...ij", curvature, inverse_metric)
    curvature_square = ein.contract("...ik,...kj->...ij", mixed_curvature, curvature)
    riemann_sign = float(convention_.riemann_sign)
    electric_raw = riemann_sign * (
        ricci + trace_curvature[..., None, None] * curvature - curvature_square
    )

    determinant = la.determinant_small_linear(_METRIC_SOLVE, metric)
    safe_sqrt_determinant = jnp.sqrt(jnp.maximum(determinant, 0.0))
    safe_sqrt_determinant = jnp.where(safe_sqrt_determinant > 0.0, safe_sqrt_determinant, 1.0)
    spatial_orientation = float(
        convention_.spacetime_orientation * convention_.future_time_orientation
    )
    upper_epsilon = (
        spatial_orientation
        * _LEVI_CIVITA_SYMBOL
        / safe_sqrt_determinant[..., None, None, None]
    )
    mixed_epsilon = ein.contract("...im,...mkl->...ikl", metric, upper_epsilon)
    standard_derivative = -float(convention_.extrinsic_curvature_sign) * derivative
    magnetic_raw = riemann_sign * ein.contract(
        "...ikl,...klj->...ij", mixed_epsilon, standard_derivative
    )

    electric_symmetric = 0.5 * (electric_raw + jnp.swapaxes(electric_raw, -1, -2))
    magnetic_symmetric = 0.5 * (magnetic_raw + jnp.swapaxes(magnetic_raw, -1, -2))
    electric_trace = ein.contract("...ij,...ij->...", inverse_metric, electric_symmetric)
    magnetic_trace = ein.contract("...ij,...ij->...", inverse_metric, magnetic_symmetric)
    electric = electric_symmetric - electric_trace[..., None, None] * metric / 3.0
    magnetic = magnetic_symmetric - magnetic_trace[..., None, None] * metric / 3.0
    electric_trace_error = jnp.max(
        jnp.abs(ein.contract("...ij,...ij->...", inverse_metric, electric))
    )
    magnetic_trace_error = jnp.max(
        jnp.abs(ein.contract("...ij,...ij->...", inverse_metric, magnetic))
    )
    electric_symmetry_error = jnp.max(jnp.abs(electric - jnp.swapaxes(electric, -1, -2)))
    magnetic_symmetry_error = jnp.max(jnp.abs(magnetic - jnp.swapaxes(magnetic, -1, -2)))
    finite = (
        jnp.all(jnp.isfinite(metric))
        & jnp.all(jnp.isfinite(electric))
        & jnp.all(jnp.isfinite(magnetic))
        & jnp.isfinite(electric_trace_error)
        & jnp.isfinite(magnetic_trace_error)
    )
    metric_valid = jnp.all(inverse_result.successful) & jnp.all(determinant > 0.0)
    tolerance = 128.0 * jnp.finfo(metric.real.dtype).eps
    physically_valid = (
        finite
        & metric_valid
        & (electric_trace_error <= tolerance * jnp.maximum(jnp.max(jnp.abs(electric)), 1.0))
        & (magnetic_trace_error <= tolerance * jnp.maximum(jnp.max(jnp.abs(magnetic)), 1.0))
    )
    return WeylCurvatureEvidence(
        electric,
        magnetic,
        electric_trace_error,
        magnetic_trace_error,
        electric_symmetry_error,
        magnetic_symmetry_error,
        metric_valid,
        finite,
        physically_valid,
        finite & metric_valid,
        convention_.convention_id,
    )


__all__ = [
    "FiniteRadiusExtrapolationPlan",
    "FiniteRadiusExtrapolationResult",
    "FixedFrequencyStrainPlan",
    "Psi4ExtractionPlan",
    "Psi4ExtractionResult",
    "Psi4TetradConvention",
    "SpinWeightedMultipolePlan",
    "SpinWeightedMultipoles",
    "StrainIntegrationResult",
    "WaveConvergenceEvidence",
    "WeylCurvatureEvidence",
    "vacuum_weyl_curvature",
]
