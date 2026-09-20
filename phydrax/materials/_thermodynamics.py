#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Callable
from dataclasses import dataclass

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
        value = cls(
            tuple(components),
            tuple(phases),
            jnp.asarray(temperature_k),
            jnp.asarray(pressure_pa),
            jnp.asarray(composition),
        )
        if value.composition.shape[-1] != len(value.component_names) or not bool(
            jnp.allclose(jnp.sum(value.composition, axis=-1), 1)
        ):
            raise ValueError("Equilibrium composition must align and normalize.")
        return value


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
