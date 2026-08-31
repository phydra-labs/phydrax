#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._gaussian_diffusion import AbstractGaussianDiffusion
from ._spatial_noise import SpatialNoiseBasis
from ._subspace_diffusion import AffineSubspaceLayout, SubspaceGaussianDiffusion


class FieldNoiseGeometry(StrictModule):
    """Quadrature-aware finite-rank field coordinates from a spatial noise basis."""

    basis: SpatialNoiseBasis
    layout: AffineSubspaceLayout
    field_space_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: SpatialNoiseBasis,
        /,
        *,
        field_space_id: str | None = None,
    ):
        if not isinstance(basis, SpatialNoiseBasis):
            raise TypeError("basis must be a SpatialNoiseBasis.")
        if jnp.iscomplexobj(basis.modes):
            raise TypeError(
                "FieldNoiseGeometry requires a real basis; use explicit complex coordinates."
            )
        if bool(jnp.any(basis.eigenvalues <= 0.0)):
            raise ValueError("Field diffusion requires strictly positive retained eigenvalues.")
        space = field_space_id or basis.field_space_id
        if not isinstance(space, str) or not space:
            raise ValueError("field_space_id must be supplied by the basis or caller.")
        scaled_modes = basis.modes.reshape((-1, basis.modes.shape[-1])) * jnp.sqrt(
            basis.eigenvalues
        )[None, :]
        layout = AffineSubspaceLayout(
            jnp.zeros(basis.state_shape, dtype=basis.modes.dtype),
            scaled_modes,
            event_shape=basis.state_shape,
            quadrature_weights=basis.quadrature_weights,
            layout_id=canonical_fingerprint(
                {
                    "kind": "field-noise-subspace",
                    "basis_id": basis.basis_id,
                    "field_space_id": space,
                }
            ),
        )
        self.basis = basis
        self.layout = layout
        self.field_space_id = space
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "field-noise-geometry",
                "basis_id": basis.basis_id,
                "precision_id": basis.precision.policy_id,
                "field_space_id": space,
            }
        )

    @property
    def rank(self) -> int:
        return self.layout.rank

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.layout.event_shape

    def coefficients(self, field: ArrayLike, /) -> tuple[Array, Array]:
        return self.layout.project(field)

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        return self.layout.synthesize(coefficients)

    def transfer(self, field: ArrayLike, target: "FieldNoiseGeometry", /) -> Array:
        if not isinstance(target, FieldNoiseGeometry):
            raise TypeError("target must be a FieldNoiseGeometry.")
        if self.basis.mode_ids != target.basis.mode_ids:
            raise ValueError("Field transfer requires identical ordered mode IDs.")
        if not jnp.allclose(self.basis.eigenvalues, target.basis.eigenvalues):
            raise ValueError("Field transfer requires identical retained covariance spectrum.")
        coefficients, residual = self.coefficients(field)
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(residual > 1e-7),
            "Source field lies outside the retained noise space.",
        )
        return target.synthesize(coefficients)


class FieldGaussianDiffusion(StrictModule):
    """Coefficient-space Gaussian diffusion with mesh-explicit field synthesis."""

    geometry: FieldNoiseGeometry
    coefficient_process: AbstractGaussianDiffusion
    subspace_process: SubspaceGaussianDiffusion
    process_id: str = eqx.field(static=True)

    def __init__(self, geometry, coefficient_process, /, *, process_id: str | None = None):
        if not isinstance(geometry, FieldNoiseGeometry):
            raise TypeError("geometry must be a FieldNoiseGeometry.")
        if not isinstance(coefficient_process, AbstractGaussianDiffusion):
            raise TypeError("coefficient_process must implement AbstractGaussianDiffusion.")
        if coefficient_process.state_shape != (geometry.rank,):
            raise ValueError("Coefficient diffusion dimension must match field noise rank.")
        identifier = process_id or canonical_fingerprint(
            {
                "kind": "field-gaussian-diffusion",
                "geometry_id": geometry.geometry_id,
                "coefficient_process_id": coefficient_process.process_id,
            }
        )
        self.geometry = geometry
        self.coefficient_process = coefficient_process
        self.subspace_process = SubspaceGaussianDiffusion(
            geometry.layout,
            coefficient_process,
            process_id=identifier,
        )
        self.process_id = identifier

    def perturb(self, key: Key[Array, ""], field: ArrayLike, /, *, time: ArrayLike) -> Array:
        return self.subspace_process.perturb(key, field, time=time)

    def conditional_coefficient_score(self, perturbed, clean, /, *, time):
        return self.subspace_process.conditional_coefficient_score(
            perturbed,
            clean,
            time=time,
        )

    def coefficient_score_to_field(self, coefficient_score: ArrayLike, /) -> Array:
        score = jnp.asarray(coefficient_score)
        if score.shape[-1:] != (self.geometry.rank,):
            raise ValueError("Coefficient score has an incompatible retained rank.")
        tangent_coefficients = self.geometry.layout.solve_gram(score)
        flat = oe.contract("ir,...r->...i", self.geometry.layout.basis, tangent_coefficients)
        return flat.reshape(score.shape[:-1] + self.geometry.state_shape)


__all__ = ["FieldGaussianDiffusion", "FieldNoiseGeometry"]
