#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import LatticeHarmonicDiscretization
from ....geometry import CompiledGeometry, GeometryCapability
from ....geometry._interface import regularized_heaviside_values
from ._contracts import FrequencyMaxwellMaterial


class FourierModalRasterizationPolicy(StrictModule, NonTrainableState):
    """Subpixel material averaging on one fixed two-dimensional lattice grid."""

    samples_per_axis: int = eqx.field(static=True)
    smoothing_width: float | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        samples_per_axis: int = 1,
        smoothing_width: float | None = None,
    ):
        if isinstance(samples_per_axis, bool) or not isinstance(samples_per_axis, int):
            raise TypeError("samples_per_axis must be an integer.")
        if samples_per_axis <= 0:
            raise ValueError("samples_per_axis must be positive.")
        width = None if smoothing_width is None else float(smoothing_width)
        if width is not None and (not isfinite(width) or width <= 0.0):
            raise ValueError("smoothing_width must be finite and positive.")
        self.samples_per_axis = int(samples_per_axis)
        self.smoothing_width = width
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-rasterization-policy",
                "samples_per_axis": self.samples_per_axis,
                "smoothing_width": width,
            }
        )


class FourierModalRasterizationPlan(StrictModule, NonTrainableState):
    """Prepared physical subpixel coordinates for one harmonic discretization."""

    harmonics: LatticeHarmonicDiscretization
    sample_points_xy: Array
    policy: FourierModalRasterizationPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        harmonics: LatticeHarmonicDiscretization,
        policy: FourierModalRasterizationPolicy | None = None,
        /,
    ):
        if not isinstance(harmonics, LatticeHarmonicDiscretization):
            raise TypeError("harmonics must be a LatticeHarmonicDiscretization.")
        if harmonics.periodic_dimension != 2:
            raise ValueError("Geometry rasterization currently requires a 2-D lattice.")
        policy_ = FourierModalRasterizationPolicy() if policy is None else policy
        if not isinstance(policy_, FourierModalRasterizationPolicy):
            raise TypeError("policy must be FourierModalRasterizationPolicy or None.")

        count = policy_.samples_per_axis
        shape = harmonics.sample_shape
        offsets = np.asarray(
            tuple(product(*(range(count) for _ in range(2)))), dtype=float
        )
        offsets = (offsets + 0.5) / count - 0.5
        offsets = offsets / np.asarray(shape, dtype=float)
        fractional = harmonics.fractional_coordinates[..., None, :] + jnp.asarray(
            offsets, dtype=harmonics.fractional_coordinates.dtype
        )
        physical = contract("...sp,pd->...sd", fractional, harmonics.primitive_vectors)
        self.harmonics = harmonics
        self.sample_points_xy = physical
        self.policy = policy_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-geometry-rasterization",
                "harmonics": harmonics.preparation_id,
                "policy": policy_.policy_id,
            }
        )


class FourierModalRasterizationEvidence(StrictModule, NonTrainableState):
    """Geometry, smoothing, and differentiability evidence for one material grid."""

    plan_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    geometry_kind: str = eqx.field(static=True)
    zero_set_accuracy: str = eqx.field(static=True)
    sign_reliability: str = eqx.field(static=True)
    distance_semantics: str = eqx.field(static=True)
    field_regularity: str = eqx.field(static=True)
    samples_per_pixel: int = eqx.field(static=True)
    smoothing_width: float | None = eqx.field(static=True)
    parameter_differentiable: bool = eqx.field(static=True)


class FourierModalRasterizationResult(StrictModule):
    """Rasterized material and its physical fill-fraction evidence."""

    material: FrequencyMaxwellMaterial
    fill_fraction: Array
    evidence: FourierModalRasterizationEvidence


def _scalar_material_value(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape != () or not jnp.issubdtype(result.dtype, jnp.number):
        raise TypeError(f"{name} must be one numeric scalar.")
    return result


def rasterize_fourier_modal_material(
    plan: FourierModalRasterizationPlan,
    geometry: CompiledGeometry,
    /,
    *,
    inside_permittivity: ArrayLike,
    outside_permittivity: ArrayLike = 1.0,
    inside_permeability: ArrayLike = 1.0,
    outside_permeability: ArrayLike = 1.0,
    z: ArrayLike = 0.0,
    material_id: str,
    passive: bool | None = None,
    reciprocal: bool | None = None,
) -> FourierModalRasterizationResult:
    """Rasterize one compiled geometry into a smooth or sharp periodic material."""

    if not isinstance(plan, FourierModalRasterizationPlan):
        raise TypeError("plan must be a FourierModalRasterizationPlan.")
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be a CompiledGeometry.")
    identifier = str(material_id)
    if not identifier:
        raise ValueError("material_id must be non-empty.")
    if geometry.ambient_dimension not in (2, 3):
        raise ValueError("Geometry rasterization requires ambient dimension 2 or 3.")

    points_xy = plan.sample_points_xy
    if geometry.ambient_dimension == 2:
        points = points_xy
    else:
        z_ = jnp.asarray(z)
        if z_.shape != () or not jnp.issubdtype(z_.dtype, jnp.floating):
            raise TypeError("z must be one real scalar for a 3-D geometry.")
        points = jnp.concatenate(
            (
                points_xy,
                jnp.broadcast_to(z_, points_xy.shape[:-1] + (1,)),
            ),
            axis=-1,
        )

    if plan.policy.smoothing_width is None:
        geometry.require(GeometryCapability.REGION_QUERY)
        samples = geometry.contains(points).astype(points.dtype)
        differentiable = False
    else:
        samples = 1.0 - regularized_heaviside_values(
            geometry.boundary_field(points),
            width=plan.policy.smoothing_width,
        )
        differentiable = geometry.field_certificate.parameter_differentiable
    fill = jnp.mean(samples, axis=-1)

    epsilon_in = _scalar_material_value(inside_permittivity, "inside_permittivity")
    epsilon_out = _scalar_material_value(outside_permittivity, "outside_permittivity")
    mu_in = _scalar_material_value(inside_permeability, "inside_permeability")
    mu_out = _scalar_material_value(outside_permeability, "outside_permeability")
    epsilon = epsilon_out + fill * (epsilon_in - epsilon_out)
    permeability = mu_out + fill * (mu_in - mu_out)
    material = FrequencyMaxwellMaterial(
        epsilon,
        permeability,
        material_id=identifier,
        material_role="physical",
        origin_evidence_id=plan.plan_id,
        passive=passive,
        reciprocal=reciprocal,
    )
    certificate = geometry.field_certificate
    evidence = FourierModalRasterizationEvidence(
        plan.plan_id,
        identifier,
        geometry.kind.value,
        certificate.zero_set_accuracy.value,
        certificate.sign_reliability.value,
        certificate.distance_semantics.value,
        certificate.regularity.value,
        plan.policy.samples_per_axis**2,
        plan.policy.smoothing_width,
        differentiable,
    )
    return FourierModalRasterizationResult(material, fill, evidence)


__all__ = [
    "FourierModalRasterizationEvidence",
    "FourierModalRasterizationPlan",
    "FourierModalRasterizationPolicy",
    "FourierModalRasterizationResult",
    "rasterize_fourier_modal_material",
]
