#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


@dataclass(frozen=True, slots=True)
class EquilibriumProblem:
    component_names: tuple[str, ...]
    phase_names: tuple[str, ...]
    temperature_k: Array
    pressure_pa: Array
    composition: Array

    @classmethod
    def create(cls, components, phases, temperature_k, pressure_pa, composition):
        component_values = tuple(components)
        phase_values = tuple(phases)
        if (
            not component_values
            or not phase_values
            or any(not isinstance(value, str) or not value for value in component_values)
            or any(not isinstance(value, str) or not value for value in phase_values)
            or len(set(component_values)) != len(component_values)
            or len(set(phase_values)) != len(phase_values)
        ):
            raise ValueError(
                "Equilibrium component and phase identities must be unique and nonempty."
            )
        temperature = jnp.asarray(temperature_k)
        pressure = jnp.asarray(pressure_pa)
        composition_ = jnp.asarray(composition)
        if composition_.ndim < 1 or composition_.shape[-1] != len(component_values):
            raise ValueError("Equilibrium composition must align with components.")
        composition_ = eqx.error_if(
            composition_,
            jnp.any(~jnp.isfinite(composition_) | (composition_ < 0))
            | jnp.any(~jnp.isfinite(temperature) | (temperature <= 0))
            | jnp.any(~jnp.isfinite(pressure) | (pressure <= 0))
            | jnp.any(~jnp.isclose(jnp.sum(composition_, axis=-1), 1.0)),
            "Equilibrium temperature/pressure/composition are outside physical bounds.",
        )
        return cls(
            component_values,
            phase_values,
            temperature,
            pressure,
            composition_,
        )


@dataclass(frozen=True, slots=True)
class EquilibriumResult:
    phase_fractions: Array
    phase_compositions: Array
    chemical_potentials_j_mol: Array
    successful: Array
    provider_id: str


@dataclass(frozen=True, slots=True)
class CallableEquilibriumProvider:
    provider_id: str
    evaluate: Callable[[EquilibriumProblem], EquilibriumResult]

    def __call__(self, problem: EquilibriumProblem):
        result = self.evaluate(problem)
        if (
            not isinstance(result, EquilibriumResult)
            or result.provider_id != self.provider_id
        ):
            raise ValueError("Equilibrium provider returned invalid ownership.")
        return result


__all__ = ["CallableEquilibriumProvider", "EquilibriumProblem", "EquilibriumResult"]
