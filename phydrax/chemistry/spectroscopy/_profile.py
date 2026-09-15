#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared spectroscopy axes and area-audited normalized line profiles."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.special import ndtr, voigt_profile

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import conversion_factor, ELECTRONVOLT, HARTREE, UnitDefinition


_EV_TO_INVERSE_CENTIMETER = 8065.54393734921
_EV_NANOMETER = 1239.8419843320026
_EV_TO_TERAHERTZ = 241.7989242084918


class SpectralAxis(StrEnum):
    ENERGY_EV = "energy-eV"
    WAVENUMBER_CM1 = "wavenumber-cm-1"
    WAVELENGTH_NM = "wavelength-nm"
    FREQUENCY_THZ = "frequency-THz"
    ANGULAR_FREQUENCY_AU = "angular-frequency-au"


class SpectralLineShape(StrEnum):
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"


def transition_energy_axis(
    energies: ArrayLike,
    energy_unit: UnitDefinition,
    axis: SpectralAxis,
    /,
) -> Array:
    if not isinstance(energy_unit, UnitDefinition) or not isinstance(axis, SpectralAxis):
        raise TypeError("Energy unit and spectral axis must be typed values.")
    values = jnp.asarray(energies)
    energy_ev = values * float(conversion_factor(energy_unit, ELECTRONVOLT))
    if axis is SpectralAxis.ENERGY_EV:
        return energy_ev
    if axis is SpectralAxis.WAVENUMBER_CM1:
        return energy_ev * _EV_TO_INVERSE_CENTIMETER
    if axis is SpectralAxis.WAVELENGTH_NM:
        if bool(jnp.any(energy_ev <= 0.0)):
            raise ValueError(
                "Wavelength conversion requires positive transition energies."
            )
        return _EV_NANOMETER / energy_ev
    if axis is SpectralAxis.FREQUENCY_THZ:
        return energy_ev * _EV_TO_TERAHERTZ
    return values * float(conversion_factor(energy_unit, HARTREE))


class SpectralProfileResult(StrictModule, NonTrainableState):
    grid: Array
    intensity_density: Array
    integrated_intensity: Array
    expected_intensity: Array
    area_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: ArrayLike,
        intensity_density: ArrayLike,
        integrated_intensity: ArrayLike,
        expected_intensity: ArrayLike,
        area_residual: ArrayLike,
        successful: ArrayLike,
        plan_id: str,
        /,
    ):
        grid_ = jnp.asarray(grid)
        intensity = jnp.asarray(intensity_density, dtype=grid_.dtype)
        if grid_.ndim != 1 or intensity.shape != grid_.shape:
            raise ValueError("Spectral grid and intensity density must align.")
        self.grid = grid_
        self.intensity_density = intensity
        self.integrated_intensity = jnp.asarray(
            integrated_intensity, dtype=grid_.dtype
        ).reshape(())
        self.expected_intensity = jnp.asarray(
            expected_intensity, dtype=grid_.dtype
        ).reshape(())
        self.area_residual = jnp.asarray(area_residual, dtype=grid_.dtype).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "spectral-profile-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "grid": np.asarray(grid_),
                        "intensity": np.asarray(intensity),
                        "integrated": np.asarray(self.integrated_intensity),
                        "expected": np.asarray(self.expected_intensity),
                        "area_residual": np.asarray(self.area_residual),
                    }
                ),
            }
        )


class SpectralProfilePlan(StrictModule, NonTrainableState):
    line_shape: SpectralLineShape = eqx.field(static=True)
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    gaussian_fwhm: float = eqx.field(static=True)
    lorentzian_fwhm: float = eqx.field(static=True)
    area_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        line_shape: SpectralLineShape,
        minimum: float,
        maximum: float,
        /,
        *,
        grid_size: int = 2001,
        fwhm: float,
        lorentzian_fwhm: float | None = None,
        area_tolerance: float = 5.0e-3,
    ):
        if not isinstance(line_shape, SpectralLineShape):
            raise TypeError("line_shape must be SpectralLineShape.")
        lower = float(minimum)
        upper = float(maximum)
        gaussian = float(fwhm)
        lorentzian = gaussian if lorentzian_fwhm is None else float(lorentzian_fwhm)
        tolerance = float(area_tolerance)
        size = int(grid_size)
        if (
            any(
                not isfinite(value)
                for value in (lower, upper, gaussian, lorentzian, tolerance)
            )
            or upper <= lower
            or gaussian <= 0.0
            or lorentzian <= 0.0
            or tolerance <= 0.0
            or size < 3
        ):
            raise ValueError(
                "Spectral support, widths, tolerance, or grid size is invalid."
            )
        self.line_shape = line_shape
        self.minimum = lower
        self.maximum = upper
        self.grid_size = size
        self.gaussian_fwhm = gaussian
        self.lorentzian_fwhm = lorentzian
        self.area_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-profile-plan",
                "line_shape": line_shape.value,
                "support": [lower, upper],
                "grid_size": size,
                "gaussian_fwhm": gaussian,
                "lorentzian_fwhm": lorentzian,
                "area_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        line_positions: ArrayLike,
        line_intensities: ArrayLike,
        /,
    ) -> SpectralProfileResult:
        positions = np.asarray(line_positions, dtype=float).reshape((-1,))
        strengths = np.asarray(line_intensities, dtype=float).reshape((-1,))
        if positions.shape != strengths.shape or np.any(strengths < 0.0):
            raise ValueError("Spectral lines and non-negative intensities must align.")
        grid = np.linspace(self.minimum, self.maximum, self.grid_size)
        offsets = grid[:, None] - positions[None, :]
        gaussian_sigma = self.gaussian_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if self.line_shape is SpectralLineShape.GAUSSIAN:
            profiles = np.exp(-0.5 * (offsets / gaussian_sigma) ** 2) / (
                gaussian_sigma * np.sqrt(2.0 * np.pi)
            )
            captured = ndtr((self.maximum - positions) / gaussian_sigma) - ndtr(
                (self.minimum - positions) / gaussian_sigma
            )
        elif self.line_shape is SpectralLineShape.LORENTZIAN:
            gamma = 0.5 * self.lorentzian_fwhm
            profiles = gamma / (np.pi * (offsets**2 + gamma**2))
            captured = (
                np.arctan((self.maximum - positions) / gamma)
                - np.arctan((self.minimum - positions) / gamma)
            ) / np.pi
        else:
            gamma = 0.5 * self.lorentzian_fwhm
            profiles = voigt_profile(offsets, gaussian_sigma, gamma)
            captured = np.trapezoid(profiles, grid, axis=0)
        raw_areas = np.trapezoid(profiles, grid, axis=0)
        valid_areas = raw_areas > np.finfo(float).tiny
        profiles = profiles / np.where(valid_areas, raw_areas, 1.0)[None, :]
        intensity = profiles @ strengths
        integrated = np.trapezoid(intensity, grid)
        expected = float(np.sum(strengths))
        integration_residual = abs(integrated - expected) / max(
            expected, np.finfo(float).tiny
        )
        active = strengths > 0.0
        support_residual = np.max(
            np.where(active, np.abs(1.0 - captured), 0.0), initial=0.0
        )
        residual = max(integration_residual, float(support_residual))
        lines_inside = np.all(
            ~active | ((positions > self.minimum) & (positions < self.maximum))
        )
        grid_spacing = (self.maximum - self.minimum) / (self.grid_size - 1)
        resolved = grid_spacing <= min(self.gaussian_fwhm, self.lorentzian_fwhm)
        successful = (
            lines_inside
            and np.all(~active | valid_areas)
            and resolved
            and np.all(np.isfinite(intensity))
            and residual <= self.area_tolerance
        )
        return SpectralProfileResult(
            grid,
            intensity,
            integrated,
            expected,
            residual,
            successful,
            self.plan_id,
        )


__all__ = [
    "SpectralAxis",
    "SpectralLineShape",
    "SpectralProfilePlan",
    "SpectralProfileResult",
    "transition_energy_axis",
]
