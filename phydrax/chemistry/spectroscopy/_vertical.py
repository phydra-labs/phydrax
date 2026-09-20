#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Axis-correct vertical molecular UV--visible absorption spectra."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition
from ..excited import ElectronicManifoldResult
from ._profile import (
    SpectralAxis,
    SpectralLineShape,
    SpectralProfilePlan,
    transition_energy_axis,
)


class UVVisibleSpectrumResult(StrictModule, NonTrainableState):
    line_positions: Array
    oscillator_strengths: Array
    transition_dipoles: Array
    grid: Array
    oscillator_strength_density: Array
    integrated_strength: Array
    expected_strength: Array
    area_residual: Array
    successful: Array
    axis: SpectralAxis = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        line_positions: ArrayLike,
        oscillator_strengths: ArrayLike,
        transition_dipoles: ArrayLike,
        grid: ArrayLike,
        oscillator_strength_density: ArrayLike,
        integrated_strength: ArrayLike,
        expected_strength: ArrayLike,
        area_residual: ArrayLike,
        successful: ArrayLike,
        axis: SpectralAxis,
        source_result_id: str,
        /,
    ):
        positions = jnp.asarray(line_positions)
        strengths = jnp.asarray(oscillator_strengths, dtype=positions.dtype)
        dipoles = jnp.asarray(transition_dipoles, dtype=positions.dtype)
        grid_ = jnp.asarray(grid, dtype=positions.dtype)
        density = jnp.asarray(oscillator_strength_density, dtype=positions.dtype)
        if (
            positions.ndim != 1
            or strengths.shape != positions.shape
            or dipoles.shape
            != (
                positions.size,
                3,
            )
        ):
            raise ValueError(
                "UV-visible line positions, strengths, and dipoles must align."
            )
        if grid_.ndim != 1 or density.shape != grid_.shape:
            raise ValueError("UV-visible grid and density must align.")
        if not isinstance(axis, SpectralAxis):
            raise TypeError("axis must be SpectralAxis.")
        self.line_positions = positions
        self.oscillator_strengths = strengths
        self.transition_dipoles = dipoles
        self.grid = grid_
        self.oscillator_strength_density = density
        self.integrated_strength = jnp.asarray(
            integrated_strength, dtype=positions.dtype
        ).reshape(())
        self.expected_strength = jnp.asarray(
            expected_strength, dtype=positions.dtype
        ).reshape(())
        self.area_residual = jnp.asarray(area_residual, dtype=positions.dtype).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.axis = axis
        self.result_id = canonical_fingerprint(
            {
                "kind": "uv-visible-spectrum-result",
                "axis": axis.value,
                "source": str(source_result_id),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "line_positions": np.asarray(positions),
                        "oscillator_strengths": np.asarray(strengths),
                        "transition_dipoles": np.asarray(dipoles),
                        "grid": np.asarray(grid_),
                        "density": np.asarray(density),
                        "integrated_strength": np.asarray(self.integrated_strength),
                        "expected_strength": np.asarray(self.expected_strength),
                        "area_residual": np.asarray(self.area_residual),
                    }
                ),
            }
        )


class UVVisibleSpectrumPlan(StrictModule, NonTrainableState):
    axis: SpectralAxis = eqx.field(static=True)
    line_shape: SpectralLineShape = eqx.field(static=True)
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    fwhm: float = eqx.field(static=True)
    area_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: SpectralAxis,
        line_shape: SpectralLineShape,
        minimum: float,
        maximum: float,
        /,
        *,
        grid_size: int = 2001,
        fwhm: float,
        area_tolerance: float = 5.0e-3,
    ):
        if not isinstance(axis, SpectralAxis) or not isinstance(
            line_shape, SpectralLineShape
        ):
            raise TypeError("axis and line_shape must be typed spectral enums.")
        lower = float(minimum)
        upper = float(maximum)
        width = float(fwhm)
        tolerance = float(area_tolerance)
        size = int(grid_size)
        if any(not isfinite(value) for value in (lower, upper, width, tolerance)):
            raise ValueError("Spectrum support, width, and tolerance must be finite.")
        if lower <= 0.0 or upper <= lower or width <= 0.0 or tolerance <= 0.0 or size < 3:
            raise ValueError(
                "Spectrum support, width, tolerance, or grid size is invalid."
            )
        self.axis = axis
        self.line_shape = line_shape
        self.minimum = lower
        self.maximum = upper
        self.grid_size = size
        self.fwhm = width
        self.area_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "uv-visible-spectrum-plan",
                "axis": axis.value,
                "line_shape": line_shape.value,
                "support": [lower, upper],
                "grid_size": size,
                "fwhm": width,
                "area_tolerance": tolerance,
            }
        )

    def _line_positions(self, energies: Array, energy_unit: UnitDefinition, /) -> Array:
        return transition_energy_axis(energies, energy_unit, self.axis)

    def evaluate(
        self,
        manifold: ElectronicManifoldResult,
        /,
    ) -> UVVisibleSpectrumResult:
        if not isinstance(manifold, ElectronicManifoldResult):
            raise TypeError("manifold must be ElectronicManifoldResult.")
        if not bool(manifold.successful):
            raise ValueError(
                "UV-visible spectra require a successful excited-state manifold."
            )
        positions = self._line_positions(
            manifold.excitation_energies, manifold.energy_unit
        )
        strengths = jnp.maximum(manifold.oscillator_strengths, 0.0)
        profile = SpectralProfilePlan(
            self.line_shape,
            self.minimum,
            self.maximum,
            grid_size=self.grid_size,
            fwhm=self.fwhm,
            area_tolerance=self.area_tolerance,
        ).evaluate(positions, strengths)
        dipoles = manifold.electric_transition_dipoles
        return UVVisibleSpectrumResult(
            positions,
            strengths,
            dipoles,
            profile.grid,
            profile.intensity_density,
            profile.integrated_intensity,
            profile.expected_intensity,
            profile.area_residual,
            profile.successful,
            self.axis,
            manifold.result_id,
        )


__all__ = [
    "SpectralAxis",
    "SpectralLineShape",
    "UVVisibleSpectrumPlan",
    "UVVisibleSpectrumResult",
]
