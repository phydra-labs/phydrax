#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral molecular potential-energy surfaces."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomisticUnitSystem
from ._provider import AbstractPreparedElectronicCalculation
from ._result import (
    ElectronicEnergyEvaluation,
    ElectronicEnergyForceEvaluation,
    ElectronicEnergyForceHessianEvaluation,
    ElectronicGroundStatePropertyEvaluation,
)


class PotentialEnergySurfaceCapabilities(StrictModule, NonTrainableState):
    energy: bool = eqx.field(static=True)
    forces: bool = eqx.field(static=True)
    hessian: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        energy: bool = True,
        forces: bool = True,
        hessian: bool = False,
        conservative: bool = True,
        differentiable: bool = False,
    ):
        if not energy:
            raise ValueError("A potential-energy surface must provide energy.")
        if hessian and not forces:
            raise ValueError("A Hessian-capable surface must provide forces.")
        values = {
            "energy": bool(energy),
            "forces": bool(forces),
            "hessian": bool(hessian),
            "conservative": bool(conservative),
            "differentiable": bool(differentiable),
        }
        for name, value in values.items():
            setattr(self, name, value)
        self.capabilities_id = canonical_fingerprint(
            {"kind": "potential-energy-surface-capabilities", **values}
        )


class PotentialEnergySurfaceEvaluation(StrictModule, NonTrainableState):
    energy: Array
    forces: Array
    hessian: Array | None
    successful: Array
    provider_id: str = eqx.field(static=True)
    source_result_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        forces: ArrayLike,
        hessian: ArrayLike | None,
        successful: ArrayLike,
        /,
        *,
        provider_id: str,
        source_result_id: str,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        forces_ = jnp.asarray(forces, dtype=energy_.dtype)
        if forces_.ndim != 2 or forces_.shape[-1] != 3:
            raise ValueError("Surface forces must have shape (atom_capacity, 3).")
        hessian_ = None if hessian is None else jnp.asarray(hessian, dtype=energy_.dtype)
        expected_hessian = forces_.shape + forces_.shape
        if hessian_ is not None and hessian_.shape != expected_hessian:
            raise ValueError(f"Surface Hessian must have shape {expected_hessian}.")
        successful_ = jnp.asarray(successful, dtype=bool).reshape(())
        provider = str(provider_id).strip()
        source = str(source_result_id).strip()
        if not provider or not source:
            raise ValueError("Surface provider and source-result IDs must be non-empty.")
        if bool(successful_) and (
            not bool(jnp.isfinite(energy_))
            or not bool(jnp.all(jnp.isfinite(forces_)))
            or (hessian_ is not None and not bool(jnp.all(jnp.isfinite(hessian_))))
        ):
            raise ValueError("A successful surface evaluation must be finite.")
        self.energy = energy_
        self.forces = forces_
        self.hessian = hessian_
        self.successful = successful_
        self.provider_id = provider
        self.source_result_id = source
        self.evaluation_id = canonical_fingerprint(
            {
                "kind": "potential-energy-surface-evaluation",
                "provider": provider,
                "source_result": source,
                "successful": bool(successful_),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy_),
                        "forces": np.asarray(forces_),
                        "hessian": None if hessian_ is None else np.asarray(hessian_),
                    }
                ),
            }
        )


class AbstractPreparedPotentialEnergySurface(StrictModule, NonTrainableState):
    system_id: AbstractAttribute[str]
    units: AbstractAttribute[AtomisticUnitSystem]
    provider_id: AbstractAttribute[str]
    surface_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[PotentialEnergySurfaceCapabilities]

    @abc.abstractmethod
    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        raise NotImplementedError


class ElectronicPotentialEnergySurface(AbstractPreparedPotentialEnergySurface):
    calculation: AbstractPreparedElectronicCalculation
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(self, calculation: AbstractPreparedElectronicCalculation, /):
        if not isinstance(calculation, AbstractPreparedElectronicCalculation):
            raise TypeError(
                "calculation must implement AbstractPreparedElectronicCalculation."
            )
        request = calculation.calculation.request
        from ._properties import ElectronicProperty

        if not request.requires(ElectronicProperty.FORCES):
            raise ValueError(
                "Electronic potential-energy surfaces require a force property request."
            )
        capabilities = PotentialEnergySurfaceCapabilities(
            forces=request.requires(ElectronicProperty.FORCES),
            hessian=request.requires(ElectronicProperty.HESSIAN),
            conservative=calculation.capabilities.conservative_forces,
            differentiable=calculation.capabilities.differentiable,
        )
        self.calculation = calculation
        self.system_id = calculation.calculation.system.system_id
        self.units = calculation.calculation.system.units
        self.provider_id = calculation.provider_id
        self.capabilities = capabilities
        self.surface_id = canonical_fingerprint(
            {
                "kind": "electronic-potential-energy-surface",
                "calculation": calculation.prepared_id,
                "capabilities": capabilities.capabilities_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        result = self.calculation.evaluate(positions, cell_vectors)
        if isinstance(result, ElectronicEnergyForceHessianEvaluation):
            forces, hessian = result.forces, result.hessian
        elif isinstance(
            result, (ElectronicEnergyForceEvaluation, ElectronicGroundStatePropertyEvaluation)
        ):
            forces, hessian = result.forces, None
        elif isinstance(result, ElectronicEnergyEvaluation):
            raise ValueError("Potential-energy surface evaluation requires forces.")
        else:
            raise TypeError("Electronic calculation returned an unsupported result type.")
        return PotentialEnergySurfaceEvaluation(
            result.energy,
            forces,
            hessian,
            result.successful,
            provider_id=result.header.provider_id,
            source_result_id=result.result_id,
        )


SurfaceEvaluator = Callable[[ArrayLike, ArrayLike | None], PotentialEnergySurfaceEvaluation]


class CallablePotentialEnergySurface(AbstractPreparedPotentialEnergySurface):
    evaluator: SurfaceEvaluator
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        evaluator: SurfaceEvaluator,
        system_id: str,
        units: AtomisticUnitSystem,
        provider_id: str,
        capabilities: PotentialEnergySurfaceCapabilities,
        /,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        identifiers = tuple(str(value).strip() for value in (system_id, provider_id))
        if any(not value for value in identifiers):
            raise ValueError("Surface identifiers must be non-empty.")
        if not isinstance(capabilities, PotentialEnergySurfaceCapabilities):
            raise TypeError("capabilities must be PotentialEnergySurfaceCapabilities.")
        self.evaluator = evaluator
        self.system_id, self.provider_id = identifiers
        self.units = units
        self.capabilities = capabilities
        self.surface_id = canonical_fingerprint(
            {
                "kind": "callable-potential-energy-surface",
                "system": self.system_id,
                "provider": self.provider_id,
                "units": units.unit_system_id,
                "capabilities": capabilities.capabilities_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        result = self.evaluator(positions, cell_vectors)
        if not isinstance(result, PotentialEnergySurfaceEvaluation):
            raise TypeError("Surface evaluator must return PotentialEnergySurfaceEvaluation.")
        if result.provider_id != self.provider_id:
            raise ValueError("Surface evaluator changed provider identity.")
        return result


class CompositePotentialEnergySurface(AbstractPreparedPotentialEnergySurface):
    surfaces: tuple[AbstractPreparedPotentialEnergySurface, ...]
    coefficients: tuple[float, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        surfaces: Sequence[AbstractPreparedPotentialEnergySurface],
        coefficients: Sequence[float],
        /,
    ):
        surfaces_ = tuple(surfaces)
        coefficients_ = tuple(float(value) for value in coefficients)
        if not surfaces_ or any(
            not isinstance(value, AbstractPreparedPotentialEnergySurface)
            for value in surfaces_
        ):
            raise TypeError("surfaces must contain prepared potential-energy surfaces.")
        if len(surfaces_) != len(coefficients_) or any(
            not np.isfinite(value) for value in coefficients_
        ):
            raise ValueError("Composite surface coefficients must align and be finite.")
        system = surfaces_[0].system_id
        units = surfaces_[0].units
        if any(value.system_id != system for value in surfaces_[1:]):
            raise ValueError("Composite surfaces must share one system identity.")
        if any(value.units.unit_system_id != units.unit_system_id for value in surfaces_[1:]):
            raise ValueError("Composite surfaces must share one unit system.")
        capabilities = PotentialEnergySurfaceCapabilities(
            forces=all(value.capabilities.forces for value in surfaces_),
            hessian=all(value.capabilities.hessian for value in surfaces_),
            conservative=all(value.capabilities.conservative for value in surfaces_),
            differentiable=all(value.capabilities.differentiable for value in surfaces_),
        )
        self.surfaces = surfaces_
        self.coefficients = coefficients_
        self.system_id = system
        self.units = units
        self.provider_id = canonical_fingerprint(
            {
                "kind": "composite-surface-provider",
                "providers": [value.provider_id for value in surfaces_],
            }
        )
        self.capabilities = capabilities
        self.surface_id = canonical_fingerprint(
            {
                "kind": "composite-potential-energy-surface",
                "surfaces": [value.surface_id for value in surfaces_],
                "coefficients": list(coefficients_),
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> PotentialEnergySurfaceEvaluation:
        values = tuple(surface.evaluate(positions, cell_vectors) for surface in self.surfaces)
        energy = sum(
            coefficient * value.energy
            for coefficient, value in zip(self.coefficients, values, strict=True)
        )
        forces = sum(
            coefficient * value.forces
            for coefficient, value in zip(self.coefficients, values, strict=True)
        )
        if self.capabilities.hessian:
            hessian_values = tuple(value.hessian for value in values)
            if any(value is None for value in hessian_values):
                raise ValueError("Hessian-capable composite component omitted its Hessian.")
            first_hessian = hessian_values[0]
            if first_hessian is None:
                raise RuntimeError("Composite Hessian validation lost its first value.")
            hessian = jnp.zeros_like(first_hessian)
            for coefficient, value in zip(
                self.coefficients, hessian_values, strict=True
            ):
                if value is None:
                    raise RuntimeError("Composite Hessian validation changed.")
                hessian = hessian + coefficient * value
        else:
            hessian = None
        successful = jnp.all(jnp.stack(tuple(value.successful for value in values)))
        source_id = canonical_fingerprint(
            {
                "kind": "composite-surface-source",
                "components": [value.source_result_id for value in values],
                "coefficients": list(self.coefficients),
            }
        )
        return PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            hessian,
            successful,
            provider_id=self.provider_id,
            source_result_id=source_id,
        )


__all__ = [
    "AbstractPreparedPotentialEnergySurface",
    "CallablePotentialEnergySurface",
    "CompositePotentialEnergySurface",
    "ElectronicPotentialEnergySurface",
    "PotentialEnergySurfaceCapabilities",
    "PotentialEnergySurfaceEvaluation",
]
