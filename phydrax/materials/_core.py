#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Engineering material identity, phase state, history, and homogenization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import CapabilityProfile, SupportTuple


HomogenizationBound: TypeAlias = Literal["voigt", "reuss", "hill"]


@dataclass(frozen=True, slots=True)
class MaterialRecord:
    material_id: str
    revision: str
    composition: tuple[tuple[str, float], ...]
    source_ids: tuple[str, ...] = ()

    @classmethod
    def create(cls, material_id, revision, composition, /, *, source_ids=()):
        return cls(
            str(material_id).strip(),
            str(revision).strip(),
            tuple(
                sorted(
                    (str(name), float(fraction)) for name, fraction in composition.items()
                )
            ),
            tuple(sorted(str(value) for value in source_ids)),
        )

    def __post_init__(self) -> None:
        if not self.material_id or not self.revision or not self.composition:
            raise ValueError("Material identity, revision, and composition are required.")
        if any(not np.isfinite(value) or value < 0.0 for _, value in self.composition):
            raise ValueError("Material fractions must be finite and non-negative.")
        if not np.isclose(
            sum(value for _, value in self.composition), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Material composition must sum to one.")

    @property
    def record_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "material-record",
                "material_id": self.material_id,
                "revision": self.revision,
                "composition": self.composition,
                "source_ids": self.source_ids,
            }
        )


class MaterialState(StrictModule, NonTrainableState):
    temperature_k: Array
    pressure_pa: Array
    phase_fractions: Array
    internal_variables: Array

    def __init__(
        self, temperature_k, pressure_pa, phase_fractions, internal_variables=()
    ):
        fractions = jnp.asarray(phase_fractions)
        if fractions.ndim < 1:
            raise ValueError("Phase fractions require a trailing phase axis.")
        self.temperature_k = jnp.asarray(temperature_k)
        self.pressure_pa = jnp.asarray(pressure_pa)
        self.phase_fractions = fractions
        self.internal_variables = jnp.asarray(internal_variables)

    @property
    def admissible(self) -> Array:
        return (
            jnp.all(jnp.isfinite(self.temperature_k))
            & jnp.all(self.temperature_k > 0.0)
            & jnp.all(jnp.isfinite(self.pressure_pa))
            & jnp.all(self.phase_fractions >= 0.0)
            & jnp.allclose(jnp.sum(self.phase_fractions, axis=-1), 1.0)
        )


class MaterialHistory(StrictModule, NonTrainableState):
    times_s: Array
    temperatures_k: Array
    phase_fractions: Array
    history_id: str = eqx.field(static=True)

    def __init__(self, times_s, temperatures_k, phase_fractions, /):
        times = np.asarray(times_s, dtype=float)
        temperatures = np.asarray(temperatures_k, dtype=float)
        fractions = np.asarray(phase_fractions, dtype=float)
        if times.ndim != 1 or times.size < 1 or np.any(np.diff(times) <= 0.0):
            raise ValueError("Material-history times must be strictly increasing.")
        if temperatures.shape[0] != times.size or fractions.shape[0] != times.size:
            raise ValueError("Material-history fields must align with time.")
        if (
            np.any(temperatures <= 0.0)
            or np.any(fractions < 0.0)
            or not np.allclose(fractions.sum(axis=-1), 1.0)
        ):
            raise ValueError("Material history is thermodynamically inadmissible.")
        self.times_s = jnp.asarray(times)
        self.temperatures_k = jnp.asarray(temperatures)
        self.phase_fractions = jnp.asarray(fractions)
        self.history_id = canonical_fingerprint(
            {
                "kind": "material-history",
                "times_s": times.tolist(),
                "temperatures_k": temperatures.tolist(),
                "phase_fractions": fractions.tolist(),
            }
        )


def homogenize_scalar(
    properties: ArrayLike, fractions: ArrayLike, bound: HomogenizationBound = "hill"
) -> Array:
    values = jnp.asarray(properties)
    weights = jnp.asarray(fractions)
    if values.shape != weights.shape:
        raise ValueError("Properties and fractions must align.")
    voigt = jnp.sum(weights * values, axis=-1)
    reuss = 1.0 / jnp.sum(weights / values, axis=-1)
    if bound == "voigt":
        return voigt
    if bound == "reuss":
        return reuss
    if bound == "hill":
        return 0.5 * (voigt + reuss)
    raise ValueError(f"Unknown homogenization bound {bound!r}.")


def jmak_fraction(time_s: ArrayLike, rate_s_inv: ArrayLike, exponent: float, /) -> Array:
    if exponent <= 0.0:
        raise ValueError("JMAK exponent must be positive.")
    return 1.0 - jnp.exp(
        -(
            jnp.maximum(jnp.asarray(rate_s_inv) * jnp.asarray(time_s), 0.0)
            ** float(exponent)
        )
    )


def koistinen_marburger_fraction(
    temperature_k: ArrayLike, start_temperature_k: float, coefficient_k_inv: float, /
) -> Array:
    if coefficient_k_inv <= 0.0:
        raise ValueError("Koistinen-Marburger coefficient must be positive.")
    undercooling = jnp.maximum(
        float(start_temperature_k) - jnp.asarray(temperature_k), 0.0
    )
    return 1.0 - jnp.exp(-float(coefficient_k_inv) * undercooling)


def materials_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("materials.identity-history", {"state": "temperature-pressure-phase-history"}),
        ("materials.homogenization", {"methods": "voigt-reuss-hill"}),
        ("materials.phase-transformation", {"methods": "jmak-koistinen-marburger"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("analytic-control", "conservation", "public-workflow"),
        )
        for name, attrs in specs
    )


__all__ = [
    "HomogenizationBound",
    "MaterialHistory",
    "MaterialRecord",
    "MaterialState",
    "homogenize_scalar",
    "jmak_fraction",
    "koistinen_marburger_fraction",
    "materials_candidate_profiles",
]
