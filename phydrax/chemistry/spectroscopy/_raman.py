#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nonresonant Placzek Raman activities from polarizability derivatives."""

from __future__ import annotations

import abc
from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure, AtomisticSystemPlan
from ...units import derived_unit, UnitDefinition
from .._optimization import _require_structure_matches_system
from ..electronic_structure._kohn_sham import NativeLDAPlan, StaticPolarizabilityResult
from ..vibration._harmonic import VibrationalAnalysisResult
from ._profile import SpectralLineShape, SpectralProfilePlan


class AbstractPolarizabilityProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    system_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(self, positions: ArrayLike, /) -> StaticPolarizabilityResult:
        raise NotImplementedError


PolarizabilityEvaluator = Callable[[ArrayLike], StaticPolarizabilityResult]


class CallablePolarizabilityProvider(AbstractPolarizabilityProvider):
    evaluator: PolarizabilityEvaluator
    provider_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self, evaluator: PolarizabilityEvaluator, provider_id: str, system_id: str, /
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        system = str(system_id).strip()
        if not provider or not system:
            raise ValueError("Polarizability provider/system IDs must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider
        self.system_id = system

    def evaluate(self, positions: ArrayLike, /) -> StaticPolarizabilityResult:
        result = self.evaluator(positions)
        if not isinstance(result, StaticPolarizabilityResult):
            raise TypeError("Polarizability evaluator returned an invalid result.")
        return result


class NativeLDAPolarizabilityProvider(AbstractPolarizabilityProvider):
    plan: NativeLDAPlan
    provider_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(self, plan: NativeLDAPlan, /):
        if not isinstance(plan, NativeLDAPlan):
            raise TypeError("plan must be NativeLDAPlan.")
        self.plan = plan
        self.system_id = plan.system.system_id
        self.provider_id = canonical_fingerprint(
            {"kind": "native-lda-polarizability-provider", "plan": plan.plan_id}
        )

    def evaluate(self, positions: ArrayLike, /) -> StaticPolarizabilityResult:
        return self.plan.static_polarizability(positions)


class RamanSpectrumResult(StrictModule, NonTrainableState):
    wavenumbers: Array
    polarizability_derivatives: Array
    isotropic_invariants: Array
    anisotropy_invariants: Array
    activities: Array
    depolarization_ratios: Array
    stokes_relative_intensities: Array
    anti_stokes_relative_intensities: Array
    grid: Array
    broadened_intensity: Array
    area_residual: Array
    successful: Array
    activity_unit: UnitDefinition
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavenumbers: ArrayLike,
        polarizability_derivatives: ArrayLike,
        isotropic_invariants: ArrayLike,
        anisotropy_invariants: ArrayLike,
        activities: ArrayLike,
        depolarization_ratios: ArrayLike,
        stokes_relative_intensities: ArrayLike,
        anti_stokes_relative_intensities: ArrayLike,
        grid: ArrayLike,
        broadened_intensity: ArrayLike,
        area_residual: ArrayLike,
        successful: ArrayLike,
        activity_unit: UnitDefinition,
        source_result_ids: tuple[str, ...],
        /,
    ):
        waves = jnp.asarray(wavenumbers)
        derivatives = jnp.asarray(polarizability_derivatives, dtype=waves.dtype)
        mode_count = waves.size
        vectors = tuple(
            jnp.asarray(value, dtype=waves.dtype)
            for value in (
                isotropic_invariants,
                anisotropy_invariants,
                activities,
                depolarization_ratios,
                stokes_relative_intensities,
                anti_stokes_relative_intensities,
            )
        )
        if derivatives.shape != (mode_count, 3, 3) or any(
            value.shape != (mode_count,) for value in vectors
        ):
            raise ValueError("Raman tensors and invariants must align by mode.")
        grid_ = jnp.asarray(grid, dtype=waves.dtype)
        broadened = jnp.asarray(broadened_intensity, dtype=waves.dtype)
        if grid_.ndim != 1 or broadened.shape != grid_.shape:
            raise ValueError("Raman grid and broadened intensity must align.")
        area_residual_ = jnp.asarray(area_residual, dtype=waves.dtype).reshape(())
        if not isinstance(activity_unit, UnitDefinition):
            raise TypeError("activity_unit must be UnitDefinition.")
        self.wavenumbers = waves
        self.polarizability_derivatives = derivatives
        (
            self.isotropic_invariants,
            self.anisotropy_invariants,
            self.activities,
            self.depolarization_ratios,
            self.stokes_relative_intensities,
            self.anti_stokes_relative_intensities,
        ) = vectors
        self.grid = grid_
        self.broadened_intensity = broadened
        self.area_residual = area_residual_
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.activity_unit = activity_unit
        self.source_result_ids = source_result_ids
        self.result_id = canonical_fingerprint(
            {
                "kind": "raman-spectrum-result",
                "unit": activity_unit.unit_id,
                "sources": list(source_result_ids),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "wavenumbers": np.asarray(waves),
                        "derivatives": np.asarray(derivatives),
                        "activities": np.asarray(self.activities),
                        "depolarization": np.asarray(self.depolarization_ratios),
                        "stokes": np.asarray(self.stokes_relative_intensities),
                        "anti_stokes": np.asarray(self.anti_stokes_relative_intensities),
                        "grid": np.asarray(grid_),
                        "area_residual": np.asarray(area_residual_),
                        "broadened": np.asarray(broadened),
                    }
                ),
            }
        )


class RamanSpectrumPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    provider: AbstractPolarizabilityProvider
    line_shape: SpectralLineShape = eqx.field(static=True)
    normal_coordinate_displacement: float = eqx.field(static=True)
    laser_wavenumber: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    fwhm: float = eqx.field(static=True)
    grid_minimum: float = eqx.field(static=True)
    grid_maximum: float = eqx.field(static=True)
    polarizability_symmetry_tolerance: float = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        provider: AbstractPolarizabilityProvider,
        /,
        *,
        normal_coordinate_displacement: float = 1.0e-3,
        laser_wavenumber: float = 18797.0,
        temperature: float = 298.15,
        fwhm: float = 10.0,
        grid_minimum: float = 0.0,
        polarizability_symmetry_tolerance: float = 1.0e-8,
        grid_maximum: float = 4000.0,
        grid_size: int = 4001,
        line_shape: SpectralLineShape = SpectralLineShape.GAUSSIAN,
    ):
        if provider.system_id != system.system_id:
            raise ValueError("Raman provider belongs to another system.")
        if not isinstance(line_shape, SpectralLineShape):
            raise TypeError("line_shape must be SpectralLineShape.")
        values = tuple(
            float(value)
            for value in (
                normal_coordinate_displacement,
                laser_wavenumber,
                temperature,
                fwhm,
                grid_minimum,
                grid_maximum,
                polarizability_symmetry_tolerance,
            )
        )
        if any(not isfinite(value) for value in values) or any(
            value <= 0.0 for value in (*values[:4], values[6])
        ):
            raise ValueError(
                "Raman displacement, laser, temperature, and width are invalid."
            )
        if values[4] < 0.0 or values[5] <= values[4] or int(grid_size) < 2:
            raise ValueError("Raman grid support is invalid.")
        self.system = system
        self.provider = provider
        self.line_shape = line_shape
        (
            self.normal_coordinate_displacement,
            self.laser_wavenumber,
            self.temperature,
            self.fwhm,
            self.grid_minimum,
            self.grid_maximum,
        ) = values[:6]
        self.polarizability_symmetry_tolerance = values[6]
        self.grid_size = int(grid_size)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "raman-spectrum-plan",
                "system": system.system_id,
                "line_shape": line_shape.value,
                "provider": provider.provider_id,
                "displacement": values[0],
                "laser_wavenumber": values[1],
                "temperature": values[2],
                "fwhm": values[3],
                "grid": [values[4], values[5], int(grid_size)],
                "polarizability_symmetry_tolerance": values[6],
            }
        )

    def evaluate(
        self,
        structure: AtomicStructure,
        vibration: VibrationalAnalysisResult,
        /,
    ) -> RamanSpectrumResult:
        _require_structure_matches_system(structure, self.system)
        if not bool(vibration.successful):
            raise ValueError(
                "Raman spectroscopy requires successful vibrational analysis."
            )
        if (
            vibration.source_system_id != self.system.system_id
            or vibration.source_geometry_id != structure.structure_id
            or vibration.units.unit_system_id != self.system.units.unit_system_id
        ):
            raise ValueError("Raman vibration belongs to another system or geometry.")
        positions = np.asarray(structure.positions)
        masses = np.asarray(vibration.reduced_masses)
        modes = np.asarray(vibration.normal_modes)
        positive = np.asarray(vibration.wavenumbers) > 0.0
        waves = np.asarray(vibration.wavenumbers)[positive]
        selected_modes = modes[..., positive]
        selected_masses = masses[positive]
        if np.any(waves >= self.laser_wavenumber):
            raise ValueError("Stokes Raman shifts must lie below the laser wavenumber.")
        derivatives = []
        source_ids: list[str] = []
        successful = True
        unit: UnitDefinition | None = None
        for mode_index in range(waves.size):
            mass_normalized = selected_modes[..., mode_index] / np.sqrt(
                selected_masses[mode_index]
            )
            displacement = self.normal_coordinate_displacement * mass_normalized
            plus = self.provider.evaluate(positions + displacement)
            minus = self.provider.evaluate(positions - displacement)
            if unit is None:
                unit = plus.unit
            if (
                plus.unit.unit_id != minus.unit.unit_id
                or plus.unit.unit_id != unit.unit_id
            ):
                raise ValueError(
                    "Raman polarizability units changed across displacements."
                )
            derivatives.append(
                (np.asarray(plus.tensor) - np.asarray(minus.tensor))
                / (2.0 * self.normal_coordinate_displacement)
            )
            source_ids.extend((plus.result_id, minus.result_id))
            successful = (
                successful
                and bool(plus.successful)
                and bool(minus.successful)
                and float(plus.symmetry_residual)
                <= self.polarizability_symmetry_tolerance
                and float(minus.symmetry_residual)
                <= self.polarizability_symmetry_tolerance
            )
        if unit is None:
            raise ValueError(
                "Raman analysis requires at least one positive vibrational mode."
            )
        derivative = np.asarray(derivatives)
        diagonal = np.diagonal(derivative, axis1=1, axis2=2)
        isotropic = np.trace(derivative, axis1=1, axis2=2) / 3.0
        anisotropy = 0.5 * (
            (diagonal[:, 0] - diagonal[:, 1]) ** 2
            + (diagonal[:, 1] - diagonal[:, 2]) ** 2
            + (diagonal[:, 2] - diagonal[:, 0]) ** 2
            + 6.0
            * (
                derivative[:, 0, 1] ** 2
                + derivative[:, 1, 2] ** 2
                + derivative[:, 2, 0] ** 2
            )
        )
        activity = 45.0 * isotropic**2 + 7.0 * anisotropy
        denominator = 45.0 * isotropic**2 + 4.0 * anisotropy
        depolarization = np.where(denominator > 0.0, 3.0 * anisotropy / denominator, 0.0)
        constants = 1.438776877
        occupation = 1.0 / np.expm1(constants * waves / self.temperature)
        stokes = (
            activity * (self.laser_wavenumber - waves) ** 4 * (occupation + 1.0) / waves
        )
        anti_stokes = activity * (self.laser_wavenumber + waves) ** 4 * occupation / waves
        maximum = max(float(np.max(stokes, initial=0.0)), 1.0)
        stokes /= maximum
        anti_stokes /= maximum
        profile = SpectralProfilePlan(
            self.line_shape,
            self.grid_minimum,
            self.grid_maximum,
            grid_size=self.grid_size,
            fwhm=self.fwhm,
        ).evaluate(waves, stokes)
        grid = np.asarray(profile.grid)
        broadened = np.asarray(profile.intensity_density)
        normal_coordinate_unit = derived_unit(
            f"sqrt({self.system.units.mass_unit.symbol})*{self.system.units.scale.length_unit.symbol}",
            ((self.system.units.mass_unit, 1), (self.system.units.scale.length_unit, 2)),
        )
        derivative_unit = derived_unit(
            f"{unit.symbol}^2/{normal_coordinate_unit.symbol}^2",
            ((unit, 2), (normal_coordinate_unit, -1)),
        )
        successful = (
            successful
            and bool(profile.successful)
            and np.all(np.isfinite(derivative))
            and np.all(np.isfinite(broadened))
        )
        return RamanSpectrumResult(
            waves,
            derivative,
            isotropic,
            anisotropy,
            activity,
            depolarization,
            stokes,
            anti_stokes,
            grid,
            broadened,
            profile.area_residual,
            successful,
            derivative_unit,
            tuple(source_ids),
        )


__all__ = [
    "AbstractPolarizabilityProvider",
    "CallablePolarizabilityProvider",
    "NativeLDAPolarizabilityProvider",
    "RamanSpectrumPlan",
    "RamanSpectrumResult",
]
