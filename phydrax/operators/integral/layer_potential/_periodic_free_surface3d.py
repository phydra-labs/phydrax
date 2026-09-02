#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import PeriodicCell
from ._free_surface_green3d import FreeSurfaceGreenRepresentation3D


class PeriodicFreeSurfaceGreenEvidence3D(StrictModule, NonTrainableState):
    image_count: int = eqx.field(static=True)
    image_cutoff: int = eqx.field(static=True)
    exact_finite_image_sum: bool = eqx.field(static=True)
    infinite_lattice_certified: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PeriodicFreeSurfaceGreen3D(StrictModule, NonTrainableState):
    """Fixed finite-image Bloch-periodic free-surface Green action.

    The class is deliberately a finite-image periodic approximation, not an
    Ewald/infinite-lattice certificate.  Its identity and evidence expose that
    distinction so downstream hydrodynamics cannot promote the claim.
    """

    base: FreeSurfaceGreenRepresentation3D
    cell: PeriodicCell
    bloch_wavevector: Array
    image_indices: Array
    translations: Array
    phases: Array
    evidence: PeriodicFreeSurfaceGreenEvidence3D
    representation_id: str = eqx.field(static=True)

    def value(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        target_ = jnp.asarray(target)
        source_ = jnp.asarray(source)
        if target_.shape != (3,) or source_.shape != (3,):
            raise ValueError("Periodic free-surface points must have shape (3,).")
        shifted_sources = source_[None, :] + self.translations
        values = jax.vmap(self.base.value, in_axes=(None, 0))(target_, shifted_sources)
        return jnp.sum(self.phases * values)

    def target_gradient(self, target: ArrayLike, source: ArrayLike, /) -> Array:
        target_ = jnp.asarray(target)
        source_ = jnp.asarray(source)
        if target_.shape != (3,) or source_.shape != (3,):
            raise ValueError("Periodic free-surface points must have shape (3,).")
        shifted_sources = source_[None, :] + self.translations
        values = jax.vmap(self.base.target_gradient, in_axes=(None, 0))(
            target_, shifted_sources
        )
        return jnp.sum(self.phases[:, None] * values, axis=0)


def prepare_periodic_free_surface_green_3d(
    base: FreeSurfaceGreenRepresentation3D,
    cell: PeriodicCell,
    /,
    *,
    image_cutoff: int,
    bloch_wavevector: ArrayLike | None = None,
    maximum_images: int = 4096,
) -> PeriodicFreeSurfaceGreen3D:
    """Prepare the first rank-two-in-R3 PeriodicCell consumer."""

    if not isinstance(base, FreeSurfaceGreenRepresentation3D):
        raise TypeError("base must be FreeSurfaceGreenRepresentation3D.")
    if not isinstance(cell, PeriodicCell):
        raise TypeError("cell must be PeriodicCell.")
    if cell.rank != 2 or cell.ambient_dimension != 3 or not cell.fully_periodic:
        raise ValueError(
            "Periodic free-surface Green requires a fully periodic rank-2 cell in R3."
        )
    vectors = np.asarray(cell.vectors, dtype=float)
    scale = max(float(np.linalg.norm(vectors)), 1.0)
    if np.max(np.abs(vectors[:, 2])) > 64.0 * np.finfo(float).eps * scale:
        raise ValueError(
            "Free-surface periodic vectors must lie in the horizontal plane."
        )
    cutoff = int(image_cutoff)
    limit = int(maximum_images)
    if cutoff < 0 or limit <= 0:
        raise ValueError("image_cutoff must be nonnegative and maximum_images positive.")
    indices = np.asarray(
        tuple(product(range(-cutoff, cutoff + 1), repeat=2)), dtype=np.int32
    )
    if indices.shape[0] > limit:
        raise ValueError("Periodic free-surface image capacity exceeded.")
    wavevector = (
        np.zeros((3,), dtype=float)
        if bloch_wavevector is None
        else np.asarray(bloch_wavevector, dtype=float)
    )
    if wavevector.shape != (3,) or np.any(~np.isfinite(wavevector)):
        raise ValueError("bloch_wavevector must be one finite R3 vector.")
    if abs(float(wavevector[2])) > 64.0 * np.finfo(float).eps * max(
        float(np.linalg.norm(wavevector)), 1.0
    ):
        raise ValueError("Free-surface Bloch wavevectors must be horizontal.")
    translations = indices @ vectors
    phases = np.exp(1j * (translations @ wavevector))
    evidence_id = canonical_fingerprint(
        {
            "kind": "periodic-free-surface-green-evidence-3d",
            "base": base.representation_id,
            "cell": cell.cell_id,
            "indices": array_tree_fingerprint(indices),
            "bloch": array_tree_fingerprint(wavevector),
            "exact_finite_image_sum": True,
            "infinite_lattice_certified": False,
        }
    )
    evidence = PeriodicFreeSurfaceGreenEvidence3D(
        image_count=int(indices.shape[0]),
        image_cutoff=cutoff,
        exact_finite_image_sum=True,
        infinite_lattice_certified=False,
        evidence_id=evidence_id,
    )
    return PeriodicFreeSurfaceGreen3D(
        base=base,
        cell=cell,
        bloch_wavevector=jnp.asarray(wavevector),
        image_indices=jnp.asarray(indices),
        translations=jnp.asarray(translations),
        phases=jnp.asarray(phases),
        evidence=evidence,
        representation_id=canonical_fingerprint(
            {
                "kind": "periodic-free-surface-green-3d",
                "base": base.representation_id,
                "cell": cell.cell_id,
                "evidence": evidence_id,
            }
        ),
    )


__all__ = [
    "PeriodicFreeSurfaceGreen3D",
    "PeriodicFreeSurfaceGreenEvidence3D",
    "prepare_periodic_free_surface_green_3d",
]
