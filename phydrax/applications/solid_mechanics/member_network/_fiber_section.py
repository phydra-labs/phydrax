#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class FiberSectionGeometry(StrictModule, NonTrainableState):
    coordinates: Array
    areas: Array
    material_indices: Array
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        areas: ArrayLike,
        material_indices: ArrayLike,
        /,
        *,
        section_id: str = "fiber-section",
    ):
        coordinates_ = jnp.asarray(coordinates)
        areas_ = jnp.asarray(areas, dtype=coordinates_.dtype)
        materials = jnp.asarray(material_indices, dtype=jnp.int32)
        if coordinates_.ndim != 2 or coordinates_.shape[1] != 2:
            raise ValueError("Fiber coordinates must have shape (fibers, 2).")
        if areas_.shape != (coordinates_.shape[0],) or materials.shape != areas_.shape:
            raise ValueError("Fiber areas/material indices must align with coordinates.")
        if bool(jnp.any(~jnp.isfinite(coordinates_)) | jnp.any(areas_ <= 0.0)):
            raise ValueError("Fiber geometry must be finite with positive areas.")
        self.coordinates = coordinates_
        self.areas = areas_
        self.material_indices = materials
        self.section_id = str(section_id)


class BilinearFiberMaterial(StrictModule, NonTrainableState):
    young_modulus: Array
    yield_strength: Array
    isotropic_hardening: Array
    kinematic_hardening: Array
    fracture_strain: Array
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        young_modulus: ArrayLike,
        yield_strength: ArrayLike,
        /,
        *,
        isotropic_hardening: ArrayLike = 0.0,
        kinematic_hardening: ArrayLike = 0.0,
        fracture_strain: ArrayLike = jnp.inf,
        material_id: str = "bilinear-fiber-material",
    ):
        young = jnp.asarray(young_modulus)
        yield_ = jnp.asarray(yield_strength, dtype=young.dtype)
        isotropic = jnp.asarray(isotropic_hardening, dtype=young.dtype)
        kinematic = jnp.asarray(kinematic_hardening, dtype=young.dtype)
        fracture = jnp.asarray(fracture_strain, dtype=young.dtype)
        if any(
            value.shape != () for value in (young, yield_, isotropic, kinematic, fracture)
        ):
            raise ValueError("Fiber material values must be scalar.")
        if bool(
            (young <= 0.0)
            | (yield_ <= 0.0)
            | (isotropic < 0.0)
            | (kinematic < 0.0)
            | (fracture <= 0.0)
        ):
            raise ValueError("Fiber material values are inadmissible.")
        self.young_modulus = young
        self.yield_strength = yield_
        self.isotropic_hardening = isotropic
        self.kinematic_hardening = kinematic
        self.fracture_strain = fracture
        self.material_id = str(material_id)


class FiberMaterialHistory(StrictModule):
    plastic_strain: Array
    accumulated_plastic_strain: Array
    backstress: Array
    damage: Array

    @classmethod
    def zeros(cls, fiber_count: int, dtype) -> FiberMaterialHistory:
        zeros = jnp.zeros((fiber_count,), dtype=dtype)
        return cls(zeros, zeros, zeros, zeros)


class FiberSectionState(StrictModule):
    generalized_strain: Array
    fiber_strain: Array
    fiber_stress: Array
    fiber_tangent: Array
    axial_force: Array
    moment_y: Array
    moment_z: Array
    tangent: Array
    yielded: Array
    fractured: Array
    trial_history: FiberMaterialHistory
    plastic_dissipation: Array


class FiberSectionTransaction(StrictModule):
    committed: FiberMaterialHistory
    trial: FiberMaterialHistory

    def commit(self, /) -> FiberSectionTransaction:
        return FiberSectionTransaction(self.trial, self.trial)

    def discard_trial(self, /) -> FiberSectionTransaction:
        return FiberSectionTransaction(self.committed, self.committed)


def evaluate_fiber_section(
    geometry: FiberSectionGeometry,
    materials: tuple[BilinearFiberMaterial, ...],
    generalized_strain: ArrayLike,
    transaction: FiberSectionTransaction,
    /,
) -> tuple[FiberSectionState, FiberSectionTransaction]:
    """Return trial section resultants and a consistent elastoplastic tangent."""
    strain = jnp.asarray(generalized_strain)
    if strain.shape != (3,):
        raise ValueError("generalized_strain must be [axial, curvature_y, curvature_z].")
    fiber_count = geometry.areas.size
    for field in (
        transaction.committed.plastic_strain,
        transaction.committed.accumulated_plastic_strain,
        transaction.committed.backstress,
        transaction.committed.damage,
    ):
        if field.shape != (fiber_count,):
            raise ValueError("Fiber material history has the wrong shape.")
    young = jnp.asarray([value.young_modulus for value in materials])[
        geometry.material_indices
    ]
    yield_strength = jnp.asarray([value.yield_strength for value in materials])[
        geometry.material_indices
    ]
    isotropic = jnp.asarray([value.isotropic_hardening for value in materials])[
        geometry.material_indices
    ]
    kinematic = jnp.asarray([value.kinematic_hardening for value in materials])[
        geometry.material_indices
    ]
    fracture = jnp.asarray([value.fracture_strain for value in materials])[
        geometry.material_indices
    ]
    y, z = geometry.coordinates[:, 0], geometry.coordinates[:, 1]
    fiber_strain = strain[0] - strain[1] * z + strain[2] * y
    committed = transaction.committed
    trial_stress = young * (fiber_strain - committed.plastic_strain)
    relative = trial_stress - committed.backstress
    threshold = yield_strength + isotropic * committed.accumulated_plastic_strain
    yield_function = jnp.abs(relative) - threshold
    yielded = yield_function > 0.0
    direction = jnp.sign(relative)
    denominator = young + isotropic + kinematic
    increment = jnp.where(yielded, yield_function / denominator, 0.0)
    plastic_strain = committed.plastic_strain + increment * direction
    accumulated = committed.accumulated_plastic_strain + increment
    backstress = committed.backstress + kinematic * increment * direction
    stress = trial_stress - young * increment * direction
    tangent = jnp.where(
        yielded,
        young * (isotropic + kinematic) / denominator,
        young,
    )
    fractured = jnp.abs(fiber_strain) >= fracture
    damage = jnp.where(fractured, 1.0, committed.damage)
    stress = (1.0 - damage) * stress
    tangent = (1.0 - damage) * tangent
    basis = jnp.stack((jnp.ones_like(y), -z, y), axis=-1)
    weighted_stress = geometry.areas * stress
    resultants = ein.contract("fi,f->i", basis, weighted_stress)
    section_tangent = ein.contract("fi,f,fj->ij", basis, geometry.areas * tangent, basis)
    dissipation = jnp.sum(
        geometry.areas
        * increment
        * (yield_strength + isotropic * committed.accumulated_plastic_strain)
    )
    trial_history = FiberMaterialHistory(plastic_strain, accumulated, backstress, damage)
    state = FiberSectionState(
        strain,
        fiber_strain,
        stress,
        tangent,
        resultants[0],
        resultants[1],
        resultants[2],
        section_tangent,
        yielded,
        fractured,
        trial_history,
        dissipation,
    )
    return state, FiberSectionTransaction(committed, trial_history)


__all__ = [
    "BilinearFiberMaterial",
    "FiberMaterialHistory",
    "FiberSectionGeometry",
    "FiberSectionState",
    "FiberSectionTransaction",
    "evaluate_fiber_section",
]
