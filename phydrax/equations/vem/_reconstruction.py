#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._polynomial import ScaledMonomialBasis
from ..._strict import StrictModule
from ...discretization.vem import (
    VirtualElementDiscretization,
    VirtualElementRuntimeData,
)


class VirtualElementReconstruction(StrictModule):
    runtime: VirtualElementRuntimeData
    l2_coefficients: tuple[Array, ...]
    h1_coefficients: tuple[Array, ...]
    differential_coefficients: tuple[Array, ...]
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
    """Project global functional DOFs into the family's polynomial images."""
    if not isinstance(discretization, VirtualElementDiscretization):
        raise TypeError("discretization must be VirtualElementDiscretization.")
    runtime_ = discretization.default_runtime if runtime is None else runtime
    if not isinstance(runtime_, VirtualElementRuntimeData):
        raise TypeError("runtime must be VirtualElementRuntimeData.")
    family = discretization.field.element.family
    if (
        runtime_.topology_id != discretization.mesh.topology_id
        or runtime_.geometry_layout_id != discretization.mesh.geometry_layout_id
        or any(projection.family != family for projection in runtime_.projections)
    ):
        raise ValueError("VEM reconstruction runtime is incompatible with the space.")
    values = discretization.field_space.vector_space.validate(state)
    l2 = []
    h1 = []
    differential = []
    for projection, gathers, orientations in zip(
        runtime_.projections,
        discretization.dof_map.cell_dofs,
        discretization.dof_map.orientations,
        strict=True,
    ):
        local = values[gathers] * orientations
        l2.append(oe.contract("cai,ci->ca", projection.l2_coefficients, local))
        h1.append(oe.contract("cai,ci->ca", projection.h1_coefficients, local))
        differential.append(
            oe.contract("cai,ci->ca", projection.differential_coefficients, local)
        )
    return VirtualElementReconstruction(
        l2_coefficients=tuple(l2),
        h1_coefficients=tuple(h1),
        differential_coefficients=tuple(differential),
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
) -> tuple[Array, Array | None]:
    """Evaluate values and the family derivative (gradient, divergence, or curl).

    Discontinuous L2 reconstructions return ``None`` for the undefined
    derivative.
    """
    if not isinstance(reconstruction, VirtualElementReconstruction):
        raise TypeError("reconstruction must be VirtualElementReconstruction.")
    if not isinstance(discretization, VirtualElementDiscretization):
        raise TypeError("discretization must be VirtualElementDiscretization.")
    if reconstruction.field_space_id != discretization.field_space.field_space_id:
        raise ValueError("Reconstruction field space does not match the discretization.")
    block = int(block_index)
    runtime = reconstruction.runtime
    if reconstruction.runtime_id != runtime.runtime_id:
        raise ValueError("Reconstruction runtime does not match the selected geometry.")
    if block < 0 or block >= len(runtime.projections):
        raise IndexError("Virtual-element block index is out of range.")
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
    coefficients = reconstruction.l2_coefficients[block][indices]
    if projection.polynomial_value_shape == (2,):
        vector_coefficients = coefficients.reshape(
            (coefficients.shape[0], 2, projection.basis.feature_count)
        )
        value = oe.contract("cqa,cda->cqd", basis, vector_coefficients)
        differential_basis = ScaledMonomialBasis(2, projection.differential_degree)
        differential_values = differential_basis.evaluate(
            points_,
            geometry.centroids[indices],
            geometry.characteristic_lengths[indices],
        )
        differential = oe.contract(
            "cqa,ca->cq",
            differential_values,
            reconstruction.differential_coefficients[block][indices],
        )
        return value, differential
    value = oe.contract("cqa,ca->cq", basis, coefficients)
    if projection.family == "DiscontinuousL2":
        return value, None
    gradient_basis = projection.basis.gradient(
        points_,
        geometry.centroids[indices],
        geometry.characteristic_lengths[indices],
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
    """Evaluate the value, canonical normal, or canonical tangential edge trace."""
    if not isinstance(reconstruction, VirtualElementReconstruction):
        raise TypeError("reconstruction must be VirtualElementReconstruction.")
    if not isinstance(discretization, VirtualElementDiscretization):
        raise TypeError("discretization must be VirtualElementDiscretization.")
    if reconstruction.field_space_id != discretization.field_space.field_space_id:
        raise ValueError("Reconstruction field space does not match the discretization.")
    trace_kind = discretization.field.element.trace_kind
    if trace_kind == "none":
        raise ValueError("Discontinuous L2 virtual elements have no boundary trace.")
    edges = jnp.asarray(edge_indices, dtype=jnp.int32)
    if edges.ndim != 1:
        raise ValueError("Trace edge indices must be one rank-1 array.")
    parameters_ = jnp.asarray(parameters)
    if parameters_.ndim != 1:
        raise ValueError("Trace parameters must be one rank-1 array on [-1, 1].")
    degree = discretization.field.element.degree
    connectivity = jnp.asarray(discretization.mesh.connectivity.edges, dtype=jnp.int32)[
        edges
    ]
    offset = discretization.dof_map.vertex_dof_count
    if trace_kind == "value":
        from ...integration import GaussLobattoLegendreRule, interval_rule_data

        nodes = jnp.asarray(
            interval_rule_data(GaussLobattoLegendreRule(degree + 1)).nodes
        )
        basis = []
        for index in range(nodes.size):
            value = jnp.ones_like(parameters_)
            for other in range(nodes.size):
                if other != index:
                    value = (
                        value
                        * (parameters_ - nodes[other])
                        / (nodes[index] - nodes[other])
                    )
            basis.append(value)
        basis_values = jnp.stack(tuple(basis), axis=-1)
        routes = [connectivity[:, 0]]
        for interior in range(degree - 1):
            routes.append(offset + edges * (degree - 1) + interior)
        routes.append(connectivity[:, 1])
        gathered = reconstruction.state[jnp.stack(tuple(routes), axis=1)]
        return oe.contract("qi,ei->eq", basis_values, gathered)
    values = [jnp.ones_like(parameters_)]
    if degree:
        values.append(parameters_)
    for order in range(2, degree + 1):
        values.append(
            (
                (2 * order - 1) * parameters_ * values[-1]
                - (order - 1) * values[-2]
            )
            / order
        )
    dual = 2 * jnp.arange(degree + 1, dtype=parameters_.dtype) + 1
    basis_values = jnp.stack(tuple(values), axis=-1) * dual
    modes = jnp.arange(degree + 1, dtype=jnp.int32)
    routes = offset + edges[:, None] * (degree + 1) + modes[None, :]
    gathered = reconstruction.state[routes]
    return oe.contract("qi,ei->eq", basis_values, gathered)


__all__ = [
    "VirtualElementReconstruction",
    "evaluate_virtual_element_reconstruction",
    "evaluate_virtual_element_trace",
    "project_virtual_element_field",
]
