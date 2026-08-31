#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization.vem import (
    VirtualElementDiscretization,
    VirtualElementRuntimeData,
)


class VirtualElementReconstruction(StrictModule):
    runtime: VirtualElementRuntimeData
    l2_coefficients: tuple[Array, ...]
    h1_coefficients: tuple[Array, ...]
    state: Array
    runtime_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)


def project_virtual_element_field(
    discretization: VirtualElementDiscretization,
    state: ArrayLike,
    /,
    *,
    runtime: VirtualElementRuntimeData | None = None,
) -> VirtualElementReconstruction:
    if not isinstance(discretization, VirtualElementDiscretization):
        raise TypeError("discretization must be VirtualElementDiscretization.")
    runtime_ = discretization.default_runtime if runtime is None else runtime
    values = discretization.field_space.vector_space.validate(state)
    l2 = []
    h1 = []
    for projection, gathers in zip(
        runtime_.projections, discretization.dof_map.cell_dofs, strict=True
    ):
        local = values[gathers]
        l2.append(oe.contract("cai,ci->ca", projection.l2_coefficients, local))
        h1.append(oe.contract("cai,ci->ca", projection.h1_coefficients, local))
    return VirtualElementReconstruction(
        l2_coefficients=tuple(l2),
        h1_coefficients=tuple(h1),
        state=values,
        runtime_id=runtime_.runtime_id,
        runtime=runtime_,
        field_space_id=discretization.field_space.field_space_id,
        reconstruction_id=canonical_fingerprint(
            {
                "kind": "projected-virtual-element-field",
                "runtime": runtime_.runtime_id,
                "field_space": discretization.field_space.field_space_id,
            }
        ),
    )


def evaluate_virtual_element_reconstruction(
    reconstruction: VirtualElementReconstruction,
    discretization: VirtualElementDiscretization,
    block_index: int,
    points: ArrayLike,
    /,
    *,
    cell_indices: ArrayLike | None = None,
) -> tuple[Array, Array]:
    block = int(block_index)
    runtime = reconstruction.runtime
    if reconstruction.runtime_id != runtime.runtime_id:
        raise ValueError("Reconstruction runtime does not match the selected geometry.")
    geometry = runtime.geometries[block]
    projection = runtime.projections[block]
    points_ = jnp.asarray(points)
    indices = (
        jnp.arange(geometry.areas.size, dtype=jnp.int32)
        if cell_indices is None
        else jnp.asarray(cell_indices, dtype=jnp.int32)
    )
    if points_.ndim != 3 or points_.shape[0] != indices.size or points_.shape[-1] != 2:
        raise ValueError(
            "Reconstruction points require shape (selected_cells, points, 2)."
        )
    basis = projection.basis.evaluate(
        points_,
        geometry.centroids[indices],
        geometry.characteristic_lengths[indices],
    )
    gradient_basis = projection.basis.gradient(
        points_,
        geometry.centroids[indices],
        geometry.characteristic_lengths[indices],
    )
    value = oe.contract(
        "cqa,ca->cq", basis, reconstruction.l2_coefficients[block][indices]
    )
    gradient = oe.contract(
        "cqad,ca->cqd", gradient_basis, reconstruction.h1_coefficients[block][indices]
    )
    return value, gradient


def evaluate_virtual_element_trace(
    reconstruction: VirtualElementReconstruction,
    discretization: VirtualElementDiscretization,
    edge_indices: ArrayLike,
    parameters: ArrayLike,
    /,
) -> Array:
    from ...integration import GaussLobattoLegendreRule, interval_rule_data

    edges = jnp.asarray(edge_indices, dtype=jnp.int32)
    parameters_ = jnp.asarray(parameters)
    if parameters_.ndim != 1:
        raise ValueError("Trace parameters must be one rank-1 array on [-1, 1].")
    degree = discretization.field.element.degree
    nodes = jnp.asarray(interval_rule_data(GaussLobattoLegendreRule(degree + 1)).nodes)
    basis = []
    for index in range(nodes.size):
        value = jnp.ones_like(parameters_)
        for other in range(nodes.size):
            if other != index:
                value = (
                    value * (parameters_ - nodes[other]) / (nodes[index] - nodes[other])
                )
        basis.append(value)
    basis_values = jnp.stack(tuple(basis), axis=-1)
    connectivity = jnp.asarray(discretization.mesh.connectivity.edges, dtype=jnp.int32)[
        edges
    ]
    routes = [connectivity[:, 0]]
    offset = discretization.dof_map.vertex_dof_count
    for interior in range(degree - 1):
        routes.append(offset + edges * (degree - 1) + interior)
    routes.append(connectivity[:, 1])
    gathered = reconstruction.state[jnp.stack(tuple(routes), axis=1)]
    return oe.contract("qi,ei->eq", basis_values, gathered)


__all__ = [
    "VirtualElementReconstruction",
    "evaluate_virtual_element_reconstruction",
    "evaluate_virtual_element_trace",
    "project_virtual_element_field",
]
