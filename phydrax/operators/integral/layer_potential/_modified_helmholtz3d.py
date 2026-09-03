#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._surface3d import SurfacePanelization3D


class ModifiedHelmholtzLayerPotential3D(StrictModule, NonTrainableState):
    panelization: SurfacePanelization3D
    density: Array
    decay: float = eqx.field(static=True)
    minimum_clearance: float = eqx.field(static=True)
    kind: Literal["single", "double"] = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __call__(self, target: ArrayLike, /) -> Array:
        point = jnp.asarray(target, dtype=self.panelization.points.dtype)
        if point.shape != (3,):
            raise ValueError("Modified-Helmholtz target must have shape (3,).")
        differences = point[None, :] - self.panelization.points
        radii = jnp.linalg.norm(differences, axis=1)
        point = eqx.error_if(
            point,
            jnp.min(radii) <= self.minimum_clearance,
            "Target violates the prepared off-surface clearance.",
        )
        exponential = jnp.exp(-self.decay * radii)
        if self.kind == "single":
            kernels = exponential / (4.0 * jnp.pi * radii)
        else:
            kernels = (
                exponential
                * (1.0 + self.decay * radii)
                * jnp.sum(differences * self.panelization.normals, axis=1)
                / (4.0 * jnp.pi * radii**3)
            )
        return contract(
            "n,n,n->", kernels, self.panelization.weights, self.density, backend="jax"
        )

    def evaluate(self, targets: ArrayLike, /) -> Array:
        values = jnp.asarray(targets)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("targets must have shape (target_count, 3).")
        return jax.vmap(self)(values)


def prepare_modified_helmholtz_reconstruction_3d(
    panelization: SurfacePanelization3D,
    density: ArrayLike,
    /,
    *,
    decay: float,
    kind: Literal["single", "double"] = "single",
    minimum_clearance: float,
) -> ModifiedHelmholtzLayerPotential3D:
    if not isinstance(panelization, SurfacePanelization3D):
        raise TypeError("panelization must be SurfacePanelization3D.")
    values = jnp.asarray(density)
    if values.shape != (panelization.node_count,):
        raise ValueError("density must have one value per panelization node.")
    decay_ = float(decay)
    clearance = float(minimum_clearance)
    if (
        not np.isfinite(decay_)
        or decay_ <= 0
        or not np.isfinite(clearance)
        or clearance <= 0
        or kind not in ("single", "double")
    ):
        raise ValueError(
            "Modified-Helmholtz parameters violate the positive off-surface envelope."
        )
    return ModifiedHelmholtzLayerPotential3D(
        panelization=panelization,
        density=values,
        decay=decay_,
        minimum_clearance=clearance,
        kind=kind,
        representation_id=canonical_fingerprint(
            {
                "kind": "modified-helmholtz-layer-potential-3d",
                "panelization": panelization.panelization_id,
                "decay": decay_,
                "layer": kind,
                "clearance": clearance,
            }
        ),
    )


__all__ = [
    "ModifiedHelmholtzLayerPotential3D",
    "prepare_modified_helmholtz_reconstruction_3d",
]
