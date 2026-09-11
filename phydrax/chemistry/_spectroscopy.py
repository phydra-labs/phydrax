#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-difference dipole derivatives and vibrational infrared spectra."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomicStructure, AtomisticSystemPlan, AtomisticUnitSystem
from ..ein import contract
from ..execution import HostTaskExecutor, InlineTaskExecutor
from ..units import derived_unit, INVERSE_CENTIMETER, UnitDefinition
from ._optimization import _require_structure_matches_system
from ._provider import AbstractPreparedElectronicCalculation
from ._result import ElectronicGroundStatePropertyEvaluation
from ._units import dipole_derivative_unit
from ._vibration import VibrationalAnalysisResult


class IRSpectrumResult(StrictModule, NonTrainableState):
    wavenumbers: Array
    line_strengths: Array
    grid: Array
    intensity: Array
    dipole_derivative: Array
    successful: Array
    line_strength_unit: UnitDefinition
    units: AtomisticUnitSystem
    wavenumber_unit: UnitDefinition
    intensity_unit: UnitDefinition
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavenumbers,
        line_strengths,
        grid,
        intensity,
        dipole_derivative,
        successful,
        line_strength_unit: UnitDefinition,
        units: AtomisticUnitSystem,
        source_result_ids: tuple[str, ...],
        plan_id: str,
        /,
    ):
        waves = jnp.asarray(wavenumbers)
        strengths = jnp.asarray(line_strengths, dtype=waves.dtype)
        grid_ = jnp.asarray(grid, dtype=waves.dtype)
        intensity_ = jnp.asarray(intensity, dtype=waves.dtype)
        derivative = jnp.asarray(dipole_derivative, dtype=waves.dtype)
        if waves.ndim != 1 or strengths.shape != waves.shape:
            raise ValueError("IR line positions and strengths must align.")
        if grid_.ndim != 1 or intensity_.shape != grid_.shape:
            raise ValueError("IR grid and broadened intensity must align.")
        if derivative.ndim != 3 or derivative.shape[0] != 3 or derivative.shape[2] != 3:
            raise ValueError("dipole_derivative must have shape (3, atom_capacity, 3).")
        if not isinstance(line_strength_unit, UnitDefinition):
            raise TypeError("line_strength_unit must be UnitDefinition.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        sources = tuple(str(value).strip() for value in source_result_ids)
        if not sources or any(not value for value in sources):
            raise ValueError("IR source result IDs must be non-empty.")
        successful_ = jnp.asarray(successful, dtype=bool).reshape(())
        self.wavenumbers = waves
        self.line_strengths = strengths
        self.grid = grid_
        self.intensity = intensity_
        self.wavenumber_unit = INVERSE_CENTIMETER
        self.intensity_unit = derived_unit(
            f"{line_strength_unit.symbol}/{INVERSE_CENTIMETER.symbol}",
            ((line_strength_unit, 1), (INVERSE_CENTIMETER, -1)),
        )
        self.dipole_derivative = derivative
        self.successful = successful_
        self.line_strength_unit = line_strength_unit
        self.units = units
        self.source_result_ids = sources
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "ir-spectrum-result",
                "plan": self.plan_id,
                "line_strength_unit": line_strength_unit.unit_id,
                "sources": list(sources),
                "wavenumber_unit": INVERSE_CENTIMETER.unit_id,
                "intensity_unit": self.intensity_unit.unit_id,
                "successful": bool(successful_),
                "arrays": array_tree_fingerprint(
                    {
                        "wavenumbers": np.asarray(waves),
                        "line_strengths": np.asarray(strengths),
                        "grid": np.asarray(grid_),
                        "intensity": np.asarray(intensity_),
                        "dipole_derivative": np.asarray(derivative),
                    }
                ),
            }
        )


class IRSpectrumPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    calculation: AbstractPreparedElectronicCalculation
    displacement: float = eqx.field(static=True)
    minimum_wavenumber: float = eqx.field(static=True)
    maximum_wavenumber: float = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    fwhm: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        calculation: AbstractPreparedElectronicCalculation,
        /,
        *,
        displacement: float = 1.0e-3,
        minimum_wavenumber: float = 0.0,
        maximum_wavenumber: float = 4000.0,
        grid_size: int = 4001,
        fwhm: float = 10.0,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if not isinstance(calculation, AbstractPreparedElectronicCalculation):
            raise TypeError("calculation must be a prepared electronic calculation.")
        if calculation.calculation.system.system_id != system.system_id:
            raise ValueError("IR calculation belongs to another system.")
        from ._properties import ElectronicProperty

        if not calculation.calculation.request.requires(ElectronicProperty.DIPOLE):
            raise ValueError("IR spectroscopy requires a dipole property request.")
        step = float(displacement)
        lower = float(minimum_wavenumber)
        upper = float(maximum_wavenumber)
        size = int(grid_size)
        width = float(fwhm)
        if any(not isfinite(value) for value in (step, lower, upper, width)):
            raise ValueError("IR plan values must be finite.")
        if step <= 0.0 or lower < 0.0 or upper <= lower or size < 2 or width <= 0.0:
            raise ValueError("IR displacement, support, grid, or width is invalid.")
        self.system = system
        self.calculation = calculation
        self.displacement = step
        self.minimum_wavenumber = lower
        self.maximum_wavenumber = upper
        self.grid_size = size
        self.fwhm = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ir-spectrum-plan",
                "system": system.system_id,
                "calculation": calculation.prepared_id,
                "displacement": step,
                "support": [lower, upper],
                "grid_size": size,
                "fwhm": width,
            }
        )

    def evaluate(
        self,
        structure: AtomicStructure,
        vibration: VibrationalAnalysisResult,
        /,
        *,
        executor: HostTaskExecutor | None = None,
    ) -> IRSpectrumResult:
        if not isinstance(structure, AtomicStructure):
            raise TypeError("structure must be AtomicStructure.")
        if not isinstance(vibration, VibrationalAnalysisResult):
            raise TypeError("vibration must be VibrationalAnalysisResult.")
        _require_structure_matches_system(structure, self.system)
        if not bool(vibration.successful):
            raise ValueError("IR spectroscopy requires successful vibrational analysis.")
        positions = np.asarray(structure.positions, dtype=np.dtype(self.system.coordinate_dtype))
        active = np.asarray(self.system.active_mask, dtype=bool)
        coordinates = tuple(
            (int(atom), component)
            for atom in np.flatnonzero(active)
            for component in range(3)
        )
        cell = None if structure.cell is None else np.asarray(structure.cell)
        owned_executor = executor is None
        selected: HostTaskExecutor = InlineTaskExecutor() if executor is None else executor

        def displaced(atom: int, component: int, direction: int):
            candidate = positions.copy()
            candidate[atom, component] += direction * self.displacement
            result = self.calculation.evaluate(candidate, cell)
            if not isinstance(result, ElectronicGroundStatePropertyEvaluation):
                raise TypeError("IR electronic calculation must return dipole properties.")
            return result

        handles = []
        for atom, component in coordinates:
            for direction in (-1, 1):
                handles.append(
                    (
                        atom,
                        component,
                        direction,
                        selected.submit(
                            f"{self.plan_id}:{atom}:{component}:{direction:+d}",
                            displaced,
                            atom,
                            component,
                            direction,
                            byte_count=positions.nbytes,
                        ),
                    )
                )
        values = tuple(
            (atom, component, direction, handle.result())
            for atom, component, direction, handle in handles
        )
        if owned_executor:
            selected.close()
        by_coordinate = {
            (atom, component, direction): value
            for atom, component, direction, value in values
        }
        derivative = np.zeros((3, active.size, 3), dtype=positions.dtype)
        source_ids: list[str] = []
        successful = True
        for atom, component in coordinates:
            minus = by_coordinate[(atom, component, -1)]
            plus = by_coordinate[(atom, component, 1)]
            derivative[:, atom, component] = (
                np.asarray(plus.dipole) - np.asarray(minus.dipole)
            ) / (2.0 * self.displacement)
            source_ids.extend((minus.result_id, plus.result_id))
            successful = successful and bool(minus.successful) and bool(plus.successful)
        inverse_root_mass = 1.0 / np.sqrt(np.asarray(vibration.reduced_masses))
        mass_normalized_modes = (
            np.asarray(vibration.normal_modes) * inverse_root_mass[None, None, :]
        )
        projected = contract(
            "cai,aim->cm",
            jnp.asarray(derivative),
            jnp.asarray(mass_normalized_modes),
        )
        strengths = np.asarray(jnp.sum(projected**2, axis=0))
        waves = np.asarray(vibration.wavenumbers)
        positive = waves > 0.0
        line_waves = waves[positive]
        line_strengths = strengths[positive]
        grid = np.linspace(
            self.minimum_wavenumber,
            self.maximum_wavenumber,
            self.grid_size,
        )
        sigma = self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        offsets = grid[:, None] - line_waves[None, :]
        profiles = np.exp(-0.5 * (offsets / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        intensity = profiles @ line_strengths
        strength_unit = derived_unit(
            f"{self.system.units.charge_unit.symbol}^2/{self.system.units.mass_unit.symbol}",
            ((dipole_derivative_unit(self.system.units), 2), (self.system.units.mass_unit, -1)),
        )
        successful = successful and np.all(np.isfinite(derivative)) and np.all(
            np.isfinite(intensity)
        )
        return IRSpectrumResult(
            line_waves,
            line_strengths,
            grid,
            intensity,
            derivative,
            successful,
            strength_unit,
            self.system.units,
            tuple(source_ids),
            self.plan_id,
        )


__all__ = ["IRSpectrumPlan", "IRSpectrumResult"]
