#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import LatticeHarmonicDiscretization
from ._contracts import FrequencyMaxwellMaterial


class LateralTransformationOpticsPMLPlan(StrictModule, NonTrainableState):
    stretch_profile: Any = eqx.field(static=True)
    region_mask: Any = eqx.field(static=True)
    pml_id: str = eqx.field(static=True)

    def __init__(self, stretch_profile: Any, region_mask: Any, /, *, pml_id: str):
        if not callable(stretch_profile) and not eqx.is_array(stretch_profile):
            raise TypeError("stretch_profile must be callable or an explicit array.")
        if not callable(region_mask) and not eqx.is_array(region_mask):
            raise TypeError("region_mask must be callable or an explicit array.")
        identifier = str(pml_id)
        if not identifier:
            raise ValueError("pml_id must be non-empty.")
        self.stretch_profile = stretch_profile
        self.region_mask = region_mask
        self.pml_id = identifier


class TransformationOpticsPMLEvidence(StrictModule):
    minimum_determinant_magnitude: Array
    minimum_orientation: Array
    minimum_absorption_sign: Array
    seam_defect: Array
    finite: Array
    passive: Array
    successful: Array
    pml_id: str = eqx.field(static=True)


class TransformedFourierModalMaterial(StrictModule):
    material: FrequencyMaxwellMaterial
    evidence: TransformationOpticsPMLEvidence


def _tensor_samples(value: Array, lattice: LatticeHarmonicDiscretization, /) -> Array:
    array = jnp.asarray(value)
    shape = lattice.plan.sample_shape
    if array.ndim == 0:
        return jnp.broadcast_to(array * jnp.eye(3, dtype=array.dtype), (*shape, 3, 3))
    if array.shape == (3, 3):
        return jnp.broadcast_to(array, (*shape, 3, 3))
    if array.shape == shape:
        return array[..., None, None] * jnp.eye(3, dtype=array.dtype)
    if array.shape == (*shape, 3, 3):
        return array
    raise ValueError("Fourier TO material block has an incompatible sampled shape.")


def _determinant(matrix: Array, /) -> Array:
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def _seam_defect(jacobian: Array, periodic_dimension: int, /) -> Array:
    defects = []
    for axis in range(periodic_dimension):
        defects.append(
            jnp.max(
                jnp.abs(
                    jnp.take(jacobian, 0, axis=axis) - jnp.take(jacobian, -1, axis=axis)
                )
            )
        )
    return jnp.max(jnp.stack(tuple(defects)), initial=0.0)


def transform_fourier_modal_material(
    material: FrequencyMaxwellMaterial,
    lattice: LatticeHarmonicDiscretization,
    plan: LateralTransformationOpticsPMLPlan,
    /,
) -> TransformedFourierModalMaterial:
    """Apply one declared complex-coordinate Jacobian to all constitutive blocks."""

    coordinates = lattice.physical_coordinates
    jacobian = (
        plan.stretch_profile(coordinates)
        if callable(plan.stretch_profile)
        else jnp.asarray(plan.stretch_profile)
    )
    mask = (
        plan.region_mask(coordinates)
        if callable(plan.region_mask)
        else jnp.asarray(plan.region_mask)
    )
    jacobian = jnp.asarray(jacobian)
    mask = jnp.asarray(mask, dtype=bool)
    shape = lattice.plan.sample_shape
    if jacobian.shape == (*shape, 3):
        jacobian = jacobian[..., :, None] * jnp.eye(3, dtype=jacobian.dtype)
    if jacobian.shape != (*shape, 3, 3) or mask.shape != shape:
        raise ValueError("TO-PML Jacobian/mask must match the lattice sample shape.")
    identity = jnp.broadcast_to(jnp.eye(3, dtype=jacobian.dtype), jacobian.shape)
    jacobian = jnp.where(mask[..., None, None], jacobian, identity)
    determinant = _determinant(jacobian)
    epsilon = jnp.finfo(jacobian.real.dtype).eps
    finite = jnp.all(jnp.isfinite(jacobian)) & jnp.all(jnp.isfinite(determinant))
    orientation = jnp.real(determinant)
    absorption = jnp.min(
        jnp.where(
            mask[..., None],
            jnp.imag(jnp.diagonal(jacobian, axis1=-2, axis2=-1)),
            jnp.inf,
        )
    )
    absorption = jnp.where(jnp.any(mask), absorption, 0.0)
    seam = _seam_defect(jacobian, lattice.plan.layout.periodic_dimension)
    passive = (absorption >= -64.0 * epsilon) & (jnp.min(orientation) > 0.0)
    valid = (
        finite
        & passive
        & (jnp.min(jnp.abs(determinant)) > 64.0 * epsilon)
        & (seam <= 256.0 * epsilon * jnp.maximum(jnp.max(jnp.abs(jacobian)), 1.0))
    )
    jacobian = eqx.error_if(
        jacobian,
        ~valid,
        "TO-PML transform is nonfinite, singular, orientation reversing, or active.",
    )

    def transform(block: Array) -> Array:
        samples = _tensor_samples(block, lattice).astype(jacobian.dtype)
        transformed = (
            contract("...ia,...ab,...jb->...ij", jacobian, samples, jacobian)
            / determinant[..., None, None]
        )
        return jnp.where(mask[..., None, None], transformed, samples)

    transformed = FrequencyMaxwellMaterial(
        transform(material.permittivity),
        transform(material.permeability),
        magnetoelectric_xi=transform(material.magnetoelectric_xi),
        magnetoelectric_zeta=transform(material.magnetoelectric_zeta),
        material_id=canonical_fingerprint(
            {
                "kind": "transformed-fourier-modal-material",
                "source": material.material_id,
                "pml": plan.pml_id,
            }
        ),
        material_role="artificial_pml",
        origin_evidence_id=plan.pml_id,
        passive=material.passive,
        reciprocal=material.reciprocal,
    )
    evidence = TransformationOpticsPMLEvidence(
        jnp.min(jnp.abs(determinant)),
        jnp.min(orientation),
        absorption,
        seam,
        finite,
        passive,
        valid,
        plan.pml_id,
    )
    return TransformedFourierModalMaterial(transformed, evidence)


__all__ = [
    "LateralTransformationOpticsPMLPlan",
    "TransformationOpticsPMLEvidence",
    "TransformedFourierModalMaterial",
    "transform_fourier_modal_material",
]
