#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..solid_mechanics import MixedHyperelasticLaw


class LinearElasticPhaseMaterial(StrictModule, NonTrainableState):
    stiffness: Array
    eigenstrain: Array
    dimension: int = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness: ArrayLike,
        eigenstrain: ArrayLike,
        /,
        *,
        material_id: str,
    ):
        tensor = np.asarray(stiffness)
        transformation = np.asarray(eigenstrain)
        identifier = str(material_id)
        if (
            tensor.ndim != 4
            or len(set(tensor.shape)) != 1
            or tensor.shape[0] not in (2, 3)
            or transformation.shape != tensor.shape[:2]
            or np.any(~np.isfinite(tensor))
            or np.any(~np.isfinite(transformation))
            or not identifier
        ):
            raise ValueError("Linear elastic phase material is invalid.")
        dimension = tensor.shape[0]
        major = np.transpose(tensor, (2, 3, 0, 1))
        minor_left = np.transpose(tensor, (1, 0, 2, 3))
        minor_right = np.transpose(tensor, (0, 1, 3, 2))
        scale = max(float(np.max(np.abs(tensor))), 1.0)
        tolerance = 256.0 * np.finfo(tensor.dtype).eps * scale
        if (
            np.max(np.abs(tensor - major)) > tolerance
            or np.max(np.abs(tensor - minor_left)) > tolerance
            or np.max(np.abs(tensor - minor_right)) > tolerance
        ):
            raise ValueError("Elastic stiffness lacks major/minor symmetry.")
        symmetric_strains = []
        for row in range(dimension):
            for column in range(row, dimension):
                basis = np.zeros((dimension, dimension))
                basis[row, column] = 1.0
                basis[column, row] = 1.0
                basis /= np.sqrt(np.sum(basis * basis))
                symmetric_strains.append(basis)
        gram = np.asarray(
            [
                [
                    ein.contract("ij,ijkl,kl", left, tensor, right)
                    for right in symmetric_strains
                ]
                for left in symmetric_strains
            ]
        )
        if np.min(np.linalg.eigvalsh(0.5 * (gram + gram.T))) <= 0.0:
            raise ValueError("Elastic stiffness must be positive definite.")
        self.stiffness = jnp.asarray(tensor)
        self.eigenstrain = jnp.asarray(0.5 * (transformation + transformation.T))
        self.dimension = dimension
        self.material_id = canonical_fingerprint(
            {
                "kind": "linear-elastic-phase-material",
                "declared_id": identifier,
                "stiffness": array_tree_fingerprint(tensor),
                "eigenstrain": array_tree_fingerprint(transformation),
            }
        )


class PhaseMechanicalEvaluation(StrictModule):
    strain: Array
    elastic_strain: Array
    stiffness: Array
    eigenstrain: Array
    energy: Array
    stress: Array
    phase_force: Array
    finite: Array
    stable: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class PhaseMechanicalModel(StrictModule, NonTrainableState):
    materials: tuple[LinearElasticPhaseMaterial, ...]
    phase_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, materials: Sequence[LinearElasticPhaseMaterial], /):
        values = tuple(materials)
        if (
            len(values) < 2
            or any(not isinstance(value, LinearElasticPhaseMaterial) for value in values)
            or len({value.dimension for value in values}) != 1
            or len({value.material_id for value in values}) != len(values)
        ):
            raise ValueError("Phase mechanical materials are incompatible.")
        self.materials = values
        self.phase_count = len(values)
        self.dimension = values[0].dimension
        self.model_id = canonical_fingerprint(
            {
                "kind": "phase-mechanical-model",
                "materials": [value.material_id for value in values],
            }
        )

    def _energy(self, logits: Array, strain: Array) -> Array:
        weights = jax.nn.softmax(logits)
        stiffness = sum(
            (
                weight * material.stiffness.astype(strain.dtype)
                for weight, material in zip(weights[1:], self.materials[1:], strict=True)
            ),
            start=weights[0] * self.materials[0].stiffness.astype(strain.dtype),
        )
        eigenstrain = sum(
            (
                weight * material.eigenstrain.astype(strain.dtype)
                for weight, material in zip(weights[1:], self.materials[1:], strict=True)
            ),
            start=weights[0] * self.materials[0].eigenstrain.astype(strain.dtype),
        )
        elastic = strain - eigenstrain
        return 0.5 * ein.contract("ij,ijkl,kl->", elastic, stiffness, elastic)

    def evaluate(
        self,
        phase_logits: ArrayLike,
        displacement_gradient: ArrayLike,
        /,
    ) -> PhaseMechanicalEvaluation:
        logits = jnp.asarray(phase_logits)
        gradient = jnp.asarray(displacement_gradient, dtype=logits.dtype)
        if logits.shape != (self.phase_count,) or gradient.shape != (
            self.dimension,
            self.dimension,
        ):
            raise ValueError("Phase-mechanical field shapes are incompatible.")
        strain = 0.5 * (gradient + gradient.T)
        weights = jax.nn.softmax(logits)
        stiffness = sum(
            (
                weight * material.stiffness.astype(strain.dtype)
                for weight, material in zip(weights[1:], self.materials[1:], strict=True)
            ),
            start=weights[0] * self.materials[0].stiffness.astype(strain.dtype),
        )
        eigenstrain = sum(
            (
                weight * material.eigenstrain.astype(strain.dtype)
                for weight, material in zip(weights[1:], self.materials[1:], strict=True)
            ),
            start=weights[0] * self.materials[0].eigenstrain.astype(strain.dtype),
        )
        elastic = strain - eigenstrain
        energy, stress = jax.value_and_grad(
            lambda value: (
                0.5
                * ein.contract(
                    "ij,ijkl,kl->", value - eigenstrain, stiffness, value - eigenstrain
                )
            )
        )(strain)
        phase_force = jax.grad(self._energy, argnums=0)(logits, strain)
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(stress))
            & jnp.all(jnp.isfinite(phase_force))
        )
        stable = jnp.asarray(True)
        return PhaseMechanicalEvaluation(
            strain,
            elastic,
            stiffness,
            eigenstrain,
            energy,
            stress,
            phase_force,
            finite,
            stable,
            finite & stable,
            self.model_id,
        )


class PhaseHyperelasticEvaluation(StrictModule):
    energy: Array
    first_piola: Array
    phase_force: Array
    minimum_jacobian: Array
    finite: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class PhaseHyperelasticModel(StrictModule, NonTrainableState):
    laws: tuple[MixedHyperelasticLaw, ...]
    law_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        laws: Sequence[MixedHyperelasticLaw],
        law_ids: Sequence[str],
        /,
    ):
        values = tuple(laws)
        identifiers = tuple(str(value) for value in law_ids)
        if (
            len(values) < 2
            or any(not isinstance(value, MixedHyperelasticLaw) for value in values)
            or len(identifiers) != len(values)
            or any(not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError(
                "Phase hyperelastic model requires identified material laws."
            )
        self.laws = values
        self.law_ids = identifiers
        self.model_id = canonical_fingerprint(
            {
                "kind": "phase-hyperelastic-model",
                "laws": identifiers,
            }
        )

    def evaluate(
        self,
        phase_logits: ArrayLike,
        deformation_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> PhaseHyperelasticEvaluation:
        logits = jnp.asarray(phase_logits)
        deformation = jnp.asarray(deformation_gradient, dtype=logits.dtype)
        pressure_ = jnp.asarray(pressure, dtype=logits.dtype)
        if logits.shape != (len(self.laws),):
            raise ValueError("Hyperelastic phase logits have incompatible shape.")
        weights = jax.nn.softmax(logits)
        responses = tuple(law.evaluate(deformation, pressure_) for law in self.laws)
        energy = sum(
            (
                weight * response.mixed_energy
                for weight, response in zip(weights[1:], responses[1:], strict=True)
            ),
            start=weights[0] * responses[0].mixed_energy,
        )
        stress = sum(
            (
                weight * response.first_piola
                for weight, response in zip(weights[1:], responses[1:], strict=True)
            ),
            start=weights[0] * responses[0].first_piola,
        )
        phase_force = jax.grad(
            lambda values: ein.contract(
                "p,p->",
                jax.nn.softmax(values),
                jnp.stack(tuple(response.mixed_energy for response in responses)),
            )
        )(logits)
        minimum_jacobian = jnp.min(
            jnp.stack(tuple(response.evidence.jacobian for response in responses))
        )
        finite = jnp.isfinite(energy) & jnp.all(jnp.isfinite(stress))
        successful = finite & jnp.all(
            jnp.stack(tuple(response.evidence.valid for response in responses))
        )
        return PhaseHyperelasticEvaluation(
            energy,
            stress,
            phase_force,
            minimum_jacobian,
            finite,
            successful,
            self.model_id,
        )


__all__ = [
    "LinearElasticPhaseMaterial",
    "PhaseHyperelasticEvaluation",
    "PhaseHyperelasticModel",
    "PhaseMechanicalEvaluation",
    "PhaseMechanicalModel",
]
