#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ._common import SmoothingPatchGeometry, SmoothingPatchLayout
from ._moments import boundary_moment, smoothed_symmetric_gradient_matrix
from ._stabilization import SmoothingStabilizationPolicy


def smoothing_strain_matrix(
    layout: SmoothingPatchLayout,
    geometry: SmoothingPatchGeometry,
    /,
) -> Array:
    """Return Bbar[patch, engineering-strain, flattened local vector DOF]."""

    moment = boundary_moment(layout, geometry)
    per_dof = smoothed_symmetric_gradient_matrix(moment)
    return jnp.transpose(per_dof, (0, 2, 1, 3)).reshape(
        (per_dof.shape[0], 3, 2 * per_dof.shape[1])
    )


def smoothing_local_stiffness(
    layout: SmoothingPatchLayout,
    geometry: SmoothingPatchGeometry,
    constitutive: ArrayLike,
    /,
    *,
    compatible_local_stiffness: ArrayLike | None = None,
    stabilization: SmoothingStabilizationPolicy | None = None,
) -> Array:
    constitutive_ = jnp.asarray(constitutive)
    if constitutive_.shape != (3, 3):
        raise ValueError("2-D elasticity constitutive matrix must have shape (3, 3).")
    strain = smoothing_strain_matrix(layout, geometry)
    local = oe.contract(
        "p,psi,st,ptj->pij",
        geometry.area,
        strain,
        constitutive_,
        strain,
    )
    if stabilization is None or stabilization.kind == "none":
        return local
    if compatible_local_stiffness is None:
        raise ValueError(
            "Active smoothing stabilization needs compatible local stiffness."
        )
    return stabilization.apply(local, compatible_local_stiffness)


def assemble_smoothing_stiffness(
    layout: SmoothingPatchLayout,
    local_stiffness: ArrayLike,
    global_node_count: int,
    /,
) -> Array:
    local = jnp.asarray(local_stiffness)
    if local.shape[0] != layout.dof_routes.shape[0]:
        raise ValueError("Local smoothing tensors must match patch count.")
    global_size = 2 * int(global_node_count)
    result = jnp.zeros((global_size, global_size), dtype=local.dtype)
    for patch in range(layout.dof_routes.shape[0]):
        nodes = layout.dof_routes[patch]
        valid = layout.dof_valid[patch]
        flat = (2 * nodes[:, None] + jnp.arange(2)[None, :]).reshape((-1,))
        active = jnp.repeat(valid, 2)
        safe = jnp.where(active, flat, 0)
        mask = active[:, None] & active[None, :]
        values = jnp.where(mask, local[patch], 0.0)
        result = result.at[safe[:, None], safe[None, :]].add(values)
    return result


def smoothing_internal_force(
    stiffness: ArrayLike,
    displacement: ArrayLike,
    /,
) -> Array:
    matrix = jnp.asarray(stiffness)
    displacement_ = jnp.asarray(displacement)
    if matrix.shape != (displacement_.size, displacement_.size):
        raise ValueError("Smoothed stiffness/displacement shapes are incompatible.")
    return (matrix @ displacement_.reshape((-1,))).reshape(displacement_.shape)


def plane_stress_matrix(young: float, poisson: float, /) -> Array:
    young_ = float(young)
    poisson_ = float(poisson)
    if young_ <= 0.0 or not (-1.0 < poisson_ < 0.5):
        raise ValueError("Invalid isotropic plane-stress constants.")
    factor = young_ / (1.0 - poisson_**2)
    return factor * jnp.asarray(
        [[1.0, poisson_, 0.0], [poisson_, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson_)]]
    )


def plane_strain_matrix(young: float, poisson: float, /) -> Array:
    young_ = float(young)
    poisson_ = float(poisson)
    if young_ <= 0.0 or not (-1.0 < poisson_ < 0.5):
        raise ValueError("Invalid isotropic plane-strain constants.")
    factor = young_ / ((1.0 + poisson_) * (1.0 - 2.0 * poisson_))
    return factor * jnp.asarray(
        [
            [1.0 - poisson_, poisson_, 0.0],
            [poisson_, 1.0 - poisson_, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson_)],
        ]
    )


__all__ = [
    "assemble_smoothing_stiffness",
    "plane_strain_matrix",
    "plane_stress_matrix",
    "smoothing_internal_force",
    "smoothing_local_stiffness",
    "smoothing_strain_matrix",
]
