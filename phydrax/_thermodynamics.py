#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._fingerprint import canonical_fingerprint
from ._phase_field import AbstractBulkFreeEnergy, DoubleWellFreeEnergy
from ._strict import AbstractAttribute, StrictModule
from ._trainable import NonTrainableState


_INTERFACE_QUADRATURE_NODES, _INTERFACE_QUADRATURE_WEIGHTS = (
    np.polynomial.legendre.leggauss(32)
)


class ThermodynamicForceRepresentation(StrEnum):
    """Discrete hydrodynamic representation of one thermodynamic closure."""

    CHEMICAL_POTENTIAL_GRADIENT = "chemical_potential_gradient"
    STRESS_DIVERGENCE = "stress_divergence"


class BinaryThermodynamicParameters(StrictModule):
    """Differentiable coefficients shared by phase-field and kinetic models."""

    bulk_scale: Array
    gradient_coefficient: Array
    wetting_strength: Array

    def __init__(
        self,
        bulk_scale: ArrayLike,
        gradient_coefficient: ArrayLike,
        /,
        *,
        wetting_strength: ArrayLike = 0.0,
    ):
        bulk = jnp.asarray(bulk_scale)
        gradient = jnp.asarray(gradient_coefficient, dtype=bulk.dtype)
        wetting = jnp.asarray(wetting_strength, dtype=bulk.dtype)
        if any(value.shape != () for value in (bulk, gradient, wetting)):
            raise ValueError("Binary thermodynamic coefficients must be scalar arrays.")
        if not jnp.issubdtype(bulk.dtype, jnp.inexact):
            raise TypeError("Binary thermodynamic coefficients require an inexact dtype.")
        bulk = eqx.error_if(
            bulk,
            ~jnp.isfinite(bulk) | (bulk <= 0.0),
            "bulk_scale must be finite and positive.",
        )
        gradient = eqx.error_if(
            gradient,
            ~jnp.isfinite(gradient) | (gradient <= 0.0),
            "gradient_coefficient must be finite and positive.",
        )
        wetting = eqx.error_if(
            wetting,
            ~jnp.isfinite(wetting),
            "wetting_strength must be finite.",
        )
        self.bulk_scale = bulk
        self.gradient_coefficient = gradient
        self.wetting_strength = wetting


class BinaryThermodynamicLocalFields(StrictModule):
    """One internally consistent local free-energy evaluation."""

    bulk_energy_density: Array
    gradient_energy_density: Array
    chemical_potential: Array
    symmetric_stress: Array


class AbstractKineticThermodynamicClosure(StrictModule, NonTrainableState):
    """Constitutive source for energy, variational derivative, and stress."""

    closure_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate_local(
        self,
        phase: Array,
        gradient: Array,
        laplacian: Array,
        parameters: BinaryThermodynamicParameters,
        /,
    ) -> BinaryThermodynamicLocalFields:
        raise NotImplementedError

    @abc.abstractmethod
    def characteristic_interface_width(
        self,
        parameters: BinaryThermodynamicParameters,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def planar_surface_tension(
        self,
        parameters: BinaryThermodynamicParameters,
        /,
    ) -> Array:
        raise NotImplementedError


class BinaryPhaseThermodynamicClosure(AbstractKineticThermodynamicClosure):
    """Scalar binary-phase closure backed by one canonical bulk potential."""

    free_energy: AbstractBulkFreeEnergy
    closure_id: str = eqx.field(static=True)

    def __init__(self, free_energy: AbstractBulkFreeEnergy | None = None, /):
        potential = DoubleWellFreeEnergy() if free_energy is None else free_energy
        if not isinstance(potential, AbstractBulkFreeEnergy):
            raise TypeError("free_energy must implement AbstractBulkFreeEnergy.")
        self.free_energy = potential
        self.closure_id = canonical_fingerprint(
            {
                "kind": "binary-phase-thermodynamic-closure",
                "free_energy": potential.free_energy_id,
            }
        )

    def evaluate_local(
        self,
        phase: Array,
        gradient: Array,
        laplacian: Array,
        parameters: BinaryThermodynamicParameters,
        /,
    ) -> BinaryThermodynamicLocalFields:
        if not isinstance(parameters, BinaryThermodynamicParameters):
            raise TypeError("parameters must be BinaryThermodynamicParameters.")
        phi = jnp.asarray(phase)
        grad = jnp.asarray(gradient, dtype=phi.dtype)
        lap = jnp.asarray(laplacian, dtype=phi.dtype)
        if grad.shape[:-1] != phi.shape or lap.shape != phi.shape:
            raise ValueError("Binary thermodynamic fields have incompatible shapes.")
        bulk = parameters.bulk_scale.astype(phi.dtype)
        kappa = parameters.gradient_coefficient.astype(phi.dtype)
        bulk_density = bulk * self.free_energy.density(phi)
        chemical = bulk * self.free_energy.derivative(phi) - kappa * lap
        gradient_squared = ein.contract("...d,...d->...", grad, grad)
        gradient_density = 0.5 * kappa * gradient_squared
        isotropic_pressure = phi * chemical - bulk_density - gradient_density
        identity = jnp.eye(grad.shape[-1], dtype=phi.dtype)
        stress = isotropic_pressure[..., None, None] * identity + kappa * ein.contract(
            "...a,...b->...ab", grad, grad
        )
        return BinaryThermodynamicLocalFields(
            bulk_density,
            gradient_density,
            chemical,
            stress,
        )

    def characteristic_interface_width(
        self,
        parameters: BinaryThermodynamicParameters,
        /,
    ) -> Array:
        if not isinstance(parameters, BinaryThermodynamicParameters):
            raise TypeError("parameters must be BinaryThermodynamicParameters.")
        dtype = parameters.bulk_scale.dtype
        if isinstance(self.free_energy, DoubleWellFreeEnergy):
            effective_bulk = parameters.bulk_scale * self.free_energy.scale.astype(dtype)
            barrier = 0.25 * effective_bulk
        else:
            barrier = parameters.bulk_scale * self.free_energy.density(
                jnp.asarray(0.0, dtype=dtype)
            )
        width = jnp.sqrt(parameters.gradient_coefficient / (2.0 * barrier))
        return eqx.error_if(
            width,
            ~jnp.isfinite(barrier)
            | (barrier <= 0.0)
            | ~jnp.isfinite(width)
            | (width <= 0.0),
            "Binary free energy has no finite positive central interface barrier.",
        )

    def planar_surface_tension(
        self,
        parameters: BinaryThermodynamicParameters,
        /,
    ) -> Array:
        if not isinstance(parameters, BinaryThermodynamicParameters):
            raise TypeError("parameters must be BinaryThermodynamicParameters.")
        dtype = parameters.bulk_scale.dtype
        if isinstance(self.free_energy, DoubleWellFreeEnergy):
            effective_bulk = parameters.bulk_scale * self.free_energy.scale.astype(dtype)
            tension = (
                2.0
                * jnp.sqrt(2.0 * effective_bulk * parameters.gradient_coefficient)
                / 3.0
            )
            return eqx.error_if(
                tension,
                ~jnp.isfinite(tension) | (tension <= 0.0),
                "Double-well surface tension must be finite and positive.",
            )
        nodes = jnp.asarray(_INTERFACE_QUADRATURE_NODES, dtype=dtype)
        weights = jnp.asarray(_INTERFACE_QUADRATURE_WEIGHTS, dtype=dtype)
        density = parameters.bulk_scale * self.free_energy.density(nodes)
        endpoints = parameters.bulk_scale * self.free_energy.density(
            jnp.asarray((-1.0, 1.0), dtype=dtype)
        )
        scale = jnp.maximum(jnp.max(jnp.abs(density)), 1.0)
        tolerance = 256.0 * jnp.finfo(dtype).eps * scale
        tension = jnp.sqrt(2.0 * parameters.gradient_coefficient) * jnp.sum(
            weights * jnp.sqrt(jnp.maximum(density, 0.0))
        )
        return eqx.error_if(
            tension,
            jnp.any(~jnp.isfinite(density))
            | jnp.any(density < -tolerance)
            | jnp.any(~jnp.isfinite(endpoints))
            | jnp.any(jnp.abs(endpoints) > tolerance)
            | ~jnp.isfinite(tension)
            | (tension <= 0.0),
            "Binary free energy is not a finite nonnegative double-well on [-1, 1].",
        )


__all__ = [
    "AbstractKineticThermodynamicClosure",
    "BinaryPhaseThermodynamicClosure",
    "BinaryThermodynamicLocalFields",
    "BinaryThermodynamicParameters",
    "ThermodynamicForceRepresentation",
]
