#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import ArraySpace, FunctionLinearOperator
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
    local = ein.contract(
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


class SmoothedElasticityOperator(StrictModule, NonTrainableState):
    """Matrix-free patch gather/action/scatter for 2-D smoothed elasticity."""

    layout: SmoothingPatchLayout
    local_stiffness: Array
    global_node_count: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: SmoothingPatchLayout,
        local_stiffness: ArrayLike,
        global_node_count: int,
        /,
    ):
        local = jnp.asarray(local_stiffness)
        count = int(global_node_count)
        local_width = 2 * int(layout.dof_routes.shape[1])
        if local.shape != (
            layout.dof_routes.shape[0],
            local_width,
            local_width,
        ):
            raise ValueError("Smoothed local stiffness shape is incompatible.")
        if count <= 0:
            raise ValueError("global_node_count must be positive.")
        self.layout = layout
        self.local_stiffness = local
        self.global_node_count = count
        self.operator_id = canonical_fingerprint(
            {
                "kind": "smoothed-elasticity-operator",
                "layout": layout.layout_id,
                "patches": int(local.shape[0]),
                "local_width": local_width,
                "global_nodes": count,
            }
        )

    def mv(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement)
        original_shape = value.shape
        flat_value = value.reshape((-1,))
        if flat_value.shape != (2 * self.global_node_count,):
            raise ValueError("Smoothed displacement has invalid global shape.")
        nodes = self.layout.dof_routes
        valid = self.layout.dof_valid
        flat_routes = (2 * nodes[..., None] + jnp.arange(2)).reshape((nodes.shape[0], -1))
        flat_valid = jnp.repeat(valid, 2, axis=1)
        safe = jnp.where(flat_valid, flat_routes, 0)
        local_value = jnp.where(flat_valid, flat_value[safe], 0.0)
        local_result = ein.contract("pij,pj->pi", self.local_stiffness, local_value)
        local_result = jnp.where(flat_valid, local_result, 0.0)
        result = jnp.zeros_like(flat_value).at[safe].add(local_result)
        return result.reshape(original_shape)

    def diagonal(self, /) -> Array:
        nodes = self.layout.dof_routes
        valid = self.layout.dof_valid
        flat_routes = (2 * nodes[..., None] + jnp.arange(2)).reshape((nodes.shape[0], -1))
        flat_valid = jnp.repeat(valid, 2, axis=1)
        safe = jnp.where(flat_valid, flat_routes, 0)
        local = jnp.diagonal(self.local_stiffness, axis1=-2, axis2=-1)
        return (
            jnp.zeros((2 * self.global_node_count,), dtype=local.dtype)
            .at[safe]
            .add(jnp.where(flat_valid, local, 0.0))
        )

    def materialize(self, /, *, max_entries: int = 4_000_000) -> Array:
        size = 2 * self.global_node_count
        if size * size > int(max_entries):
            raise ValueError(
                "Dense smoothing materialization exceeds the explicit entry budget."
            )
        return assemble_smoothing_stiffness(
            self.layout,
            self.local_stiffness,
            self.global_node_count,
        )

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        space = ArraySpace(
            (2 * self.global_node_count,), dtype=self.local_stiffness.dtype
        )
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            operator_id=self.operator_id,
        )


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
