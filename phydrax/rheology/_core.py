#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generalized-Newtonian, viscoelastic, thixotropic, and yield rheology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


GeneralizedNewtonianKind: TypeAlias = Literal[
    "power-law", "carreau-yasuda", "bingham", "herschel-bulkley"
]
ViscoelasticKind: TypeAlias = Literal[
    "oldroyd-b",
    "giesekus",
    "fene-p",
    "fene-cr",
    "linear-ptt",
    "exponential-ptt",
    "rolie-poly",
]


@dataclass(frozen=True, slots=True)
class GeneralizedNewtonianLaw:
    kind: GeneralizedNewtonianKind
    zero_shear_viscosity_pa_s: float
    infinite_shear_viscosity_pa_s: float = 0.0
    consistency_pa_s_n: float = 1.0
    power_index: float = 1.0
    time_constant_s: float = 1.0
    yasuda_exponent: float = 2.0
    yield_stress_pa: float = 0.0
    regularization_s: float = 1.0e3

    def viscosity(self, shear_rate_s_inv: ArrayLike, /) -> Array:
        rate = jnp.maximum(jnp.asarray(shear_rate_s_inv), 0.0)
        safe = jnp.maximum(rate, 1.0e-30)
        if self.kind == "power-law":
            return self.consistency_pa_s_n * safe ** (self.power_index - 1.0)
        if self.kind == "carreau-yasuda":
            return self.infinite_shear_viscosity_pa_s + (
                self.zero_shear_viscosity_pa_s - self.infinite_shear_viscosity_pa_s
            ) * (1.0 + (self.time_constant_s * rate) ** self.yasuda_exponent) ** (
                (self.power_index - 1.0) / self.yasuda_exponent
            )
        plastic = self.consistency_pa_s_n * safe ** (self.power_index - 1.0)
        yielded = (
            self.yield_stress_pa * (1.0 - jnp.exp(-self.regularization_s * rate)) / safe
        )
        if self.kind in ("bingham", "herschel-bulkley"):
            return plastic + yielded
        raise ValueError(f"Unknown generalized-Newtonian law {self.kind!r}.")


@dataclass(frozen=True, slots=True)
class ViscoelasticLaw:
    kind: ViscoelasticKind
    polymer_viscosity_pa_s: float
    relaxation_time_s: float
    finite_extensibility: float = 100.0
    mobility_factor: float = 0.0
    ptt_epsilon: float = 0.0

    def stress(self, conformation: ArrayLike, /) -> Array:
        conformation_ = jnp.asarray(conformation)
        dimension = conformation_.shape[-1]
        identity = jnp.eye(dimension, dtype=conformation_.dtype)
        if self.kind in ("fene-p", "fene-cr"):
            trace = jnp.trace(conformation_, axis1=-2, axis2=-1)
            factor = (self.finite_extensibility - dimension) / jnp.maximum(
                self.finite_extensibility - trace, 1.0e-12
            )
            return (
                self.polymer_viscosity_pa_s
                / self.relaxation_time_s
                * (factor[..., None, None] * conformation_ - identity)
            )
        return (
            self.polymer_viscosity_pa_s
            / self.relaxation_time_s
            * (conformation_ - identity)
        )

    def relaxation(self, conformation: ArrayLike, /) -> Array:
        conformation_ = jnp.asarray(conformation)
        dimension = conformation_.shape[-1]
        identity = jnp.eye(dimension, dtype=conformation_.dtype)
        delta = conformation_ - identity
        if self.kind == "giesekus":
            return (
                -(delta + self.mobility_factor * (delta @ delta)) / self.relaxation_time_s
            )
        if self.kind in ("fene-p", "fene-cr"):
            trace = jnp.trace(conformation_, axis1=-2, axis2=-1)
            factor = (self.finite_extensibility - dimension) / jnp.maximum(
                self.finite_extensibility - trace, 1.0e-12
            )
            return (
                -(factor[..., None, None] * conformation_ - identity)
                / self.relaxation_time_s
            )
        if self.kind in ("linear-ptt", "exponential-ptt"):
            trace_delta = jnp.trace(delta, axis1=-2, axis2=-1)
            factor = 1.0 + self.ptt_epsilon * trace_delta
            if self.kind == "exponential-ptt":
                factor = jnp.exp(self.ptt_epsilon * trace_delta)
            return -factor[..., None, None] * delta / self.relaxation_time_s
        return -delta / self.relaxation_time_s

    def upper_convected_rate(
        self, conformation: ArrayLike, velocity_gradient: ArrayLike, /
    ) -> Array:
        conformation_ = jnp.asarray(conformation)
        gradient = jnp.asarray(velocity_gradient)
        return (
            gradient @ conformation_
            + conformation_ @ jnp.swapaxes(gradient, -1, -2)
            + self.relaxation(conformation_)
        )

    def free_energy_density(self, conformation: ArrayLike, /) -> Array:
        value = jnp.asarray(conformation)
        dimension = value.shape[-1]
        return (
            0.5
            * self.polymer_viscosity_pa_s
            / self.relaxation_time_s
            * (
                jnp.trace(value, axis1=-2, axis2=-1)
                - jnp.linalg.slogdet(value)[1]
                - dimension
            )
        )


@dataclass(frozen=True, slots=True)
class ThixotropicLaw:
    build_rate_s_inv: float
    breakdown_coefficient: float
    exponent: float = 1.0

    def rate(self, structure: ArrayLike, shear_rate_s_inv: ArrayLike, /) -> Array:
        value = jnp.asarray(structure)
        rate = jnp.asarray(shear_rate_s_inv)
        return (
            self.build_rate_s_inv * (1.0 - value)
            - self.breakdown_coefficient * rate**self.exponent * value
        )


def log_conformation(conformation: ArrayLike, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(jnp.asarray(conformation))
    if jnp.any(eigenvalues <= 0.0):
        raise ValueError("Log conformation requires a positive-definite tensor.")
    return (eigenvectors * jnp.log(eigenvalues)[..., None, :]) @ jnp.swapaxes(
        eigenvectors, -1, -2
    )


def exp_conformation(logarithm: ArrayLike, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(jnp.asarray(logarithm))
    return (eigenvectors * jnp.exp(eigenvalues)[..., None, :]) @ jnp.swapaxes(
        eigenvectors, -1, -2
    )


def rheology_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        (
            "rheology.generalized-newtonian",
            {"laws": "power-carreau-bingham-herschel-bulkley"},
        ),
        ("rheology.viscoelastic", {"laws": "oldroyd-giesekus-fene-ptt-rolie-poly"}),
        ("rheology.thixotropic", {"state": "scalar-structure"}),
        ("rheology.log-conformation", {"stabilization": "spectral-log-exp"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("analytic-rheometry", "energy-dissipation", "jit-derivative"),
        )
        for name, attrs in specs
    )


__all__ = [
    "GeneralizedNewtonianKind",
    "GeneralizedNewtonianLaw",
    "ThixotropicLaw",
    "ViscoelasticKind",
    "ViscoelasticLaw",
    "exp_conformation",
    "log_conformation",
    "rheology_candidate_profiles",
]
