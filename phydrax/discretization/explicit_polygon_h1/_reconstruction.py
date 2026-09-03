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
from ...linalg import inverse_small_linear, SmallLinearSolvePlan
from ._space import ExplicitPolygonH1Discretization, ExplicitPolygonH1RuntimeData


class ExplicitPolygonH1Reconstruction(StrictModule):
    """One explicit polygon state bound to the geometry that reconstructs it."""

    runtime: ExplicitPolygonH1RuntimeData
    state: Array
    runtime_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)


def prepare_explicit_polygon_h1_reconstruction(
    discretization: ExplicitPolygonH1Discretization,
    state: ArrayLike,
    /,
    *,
    runtime: ExplicitPolygonH1RuntimeData | None = None,
) -> ExplicitPolygonH1Reconstruction:
    if not isinstance(discretization, ExplicitPolygonH1Discretization):
        raise TypeError("discretization must be ExplicitPolygonH1Discretization.")
    runtime_ = discretization.default_runtime if runtime is None else runtime
    discretization.validate_local_runtime(runtime_)
    values = discretization.field_space.vector_space.validate(state)
    return ExplicitPolygonH1Reconstruction(
        runtime=runtime_,
        state=values,
        runtime_id=runtime_.runtime_id,
        field_space_id=discretization.field_space.field_space_id,
        reconstruction_id=canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-reconstruction",
                "runtime": runtime_.runtime_id,
                "field_space": discretization.field_space.field_space_id,
            }
        ),
    )


def evaluate_explicit_polygon_h1_reconstruction(
    reconstruction: ExplicitPolygonH1Reconstruction,
    discretization: ExplicitPolygonH1Discretization,
    block_index: int,
    points: ArrayLike,
    /,
    *,
    cell_indices: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Evaluate the continuous field and one deterministic piecewise gradient."""
    if not isinstance(reconstruction, ExplicitPolygonH1Reconstruction):
        raise TypeError("reconstruction must be ExplicitPolygonH1Reconstruction.")
    if not isinstance(discretization, ExplicitPolygonH1Discretization):
        raise TypeError("discretization must be ExplicitPolygonH1Discretization.")
    if reconstruction.field_space_id != discretization.field_space.field_space_id:
        raise ValueError("Reconstruction field space does not match discretization.")
    runtime = reconstruction.runtime
    if reconstruction.runtime_id != runtime.runtime_id:
        raise ValueError("Reconstruction runtime identity is stale.")
    block = int(block_index)
    if block < 0 or block >= len(runtime.bases):
        raise IndexError("Explicit polygon block index is out of range.")
    basis_data = runtime.bases[block]
    geometry = runtime.geometries[block]
    query = jnp.asarray(points)
    indices = (
        jnp.arange(geometry.vertices.shape[0], dtype=jnp.int32)
        if cell_indices is None
        else jnp.asarray(cell_indices, dtype=jnp.int32)
    )
    if query.ndim != 3 or query.shape[0] != indices.size or query.shape[-1] != 2:
        raise ValueError(
            "Reconstruction points require shape (selected_cells, points, 2)."
        )
    vertices = geometry.vertices[indices]
    witness = basis_data.witness[indices]
    following = jnp.roll(vertices, -1, axis=1)
    axis_one = vertices - witness[:, None, :]
    axis_two = following - witness[:, None, :]
    jacobians = jnp.stack((axis_one, axis_two), axis=-1)
    inverse = inverse_small_linear(
        SmallLinearSolvePlan(
            2,
            singular_tolerance=float(
                discretization.qualification_policy.tolerance_multiplier
                * jnp.finfo(discretization.precision_policy.factorization_dtype).eps
            ),
            maximum_condition=discretization.qualification_policy.maximum_condition_number,
        ),
        jacobians,
    )
    relative = query[:, :, None, :] - witness[:, None, None, :]
    reference = oe.contract("cpnd,cnrd->cpnr", relative, inverse.value)
    barycentric = jnp.concatenate(
        (
            1.0 - jnp.sum(reference, axis=-1, keepdims=True),
            reference,
        ),
        axis=-1,
    )
    tolerance = (
        discretization.qualification_policy.tolerance_multiplier
        * jnp.finfo(query.dtype).eps
    )
    inside = jnp.all(barycentric >= -tolerance, axis=-1)
    valid = jnp.any(inside, axis=-1)
    choice = jnp.argmax(inside, axis=-1)
    query = eqx.error_if(
        query,
        jnp.any(~valid) | jnp.any(~inverse.successful),
        "Reconstruction point lies outside an admissible polygon fan.",
    )
    prolongation = basis_data.prolongation[indices]
    arity = basis_data.arity
    local_prolongations = []
    for triangle in range(arity):
        routes = jnp.asarray((arity, triangle, (triangle + 1) % arity))
        local_prolongations.append(prolongation[:, routes, :])
    local_prolongation = jnp.stack(tuple(local_prolongations), axis=1)
    all_values = oe.contract("cpna,cnai->cpni", barycentric, local_prolongation)
    reference_gradient = jnp.asarray(
        ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)), dtype=query.dtype
    )
    local_reference_gradients = oe.contract(
        "ar,cnai->cnir", reference_gradient, local_prolongation
    )
    all_gradients = oe.contract(
        "cnir,cnrd->cnid", local_reference_gradients, inverse.value
    )
    cell_rows = jnp.arange(indices.size)[:, None]
    point_rows = jnp.arange(query.shape[1])[None, :]
    selected_values = all_values[cell_rows, point_rows, choice]
    selected_gradients = all_gradients[cell_rows, choice]
    gathers = discretization.dof_map.cell_dofs[block][indices, :arity]
    coefficients = reconstruction.state[gathers]
    values = oe.contract("cpi,ci...->cp...", selected_values, coefficients)
    gradients = oe.contract("cpid,ci...->cp...d", selected_gradients, coefficients)
    return values, gradients


def evaluate_explicit_polygon_h1_trace(
    reconstruction: ExplicitPolygonH1Reconstruction,
    discretization: ExplicitPolygonH1Discretization,
    edge_indices: ArrayLike,
    parameters: ArrayLike,
    /,
) -> Array:
    if not isinstance(reconstruction, ExplicitPolygonH1Reconstruction):
        raise TypeError("reconstruction must be ExplicitPolygonH1Reconstruction.")
    if reconstruction.field_space_id != discretization.field_space.field_space_id:
        raise ValueError("Trace reconstruction field space is incompatible.")
    edges = jnp.asarray(edge_indices, dtype=jnp.int32)
    parameter = jnp.asarray(parameters)
    if edges.ndim != 1 or parameter.ndim != 1:
        raise ValueError("Trace edges and parameters must be rank-one arrays.")
    connectivity = jnp.asarray(discretization.mesh.connectivity.edges, dtype=jnp.int32)[
        edges
    ]
    start = reconstruction.state[connectivity[:, 0]]
    stop = reconstruction.state[connectivity[:, 1]]
    value_rank = start.ndim - 1
    start_weight = (1.0 - parameter).reshape((1, parameter.size) + (1,) * value_rank)
    stop_weight = parameter.reshape((1, parameter.size) + (1,) * value_rank)
    return start[:, None] * start_weight + stop[:, None] * stop_weight


__all__ = [
    "ExplicitPolygonH1Reconstruction",
    "evaluate_explicit_polygon_h1_reconstruction",
    "evaluate_explicit_polygon_h1_trace",
    "prepare_explicit_polygon_h1_reconstruction",
]
