#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._differentiation import DerivativeRegularity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, inverse_small_linear, SmallLinearSolvePlan
from .._polygon_query import (
    AbstractPolygonLocalBasis,
    fan_barycentric_gradients,
    LocatedPolygonPoints,
    polygon_fan_slots,
    polygon_mesh_support_geometry,
    polygon_value_port,
    polygonal_connectivity_of,
    PolygonFieldReconstructionKernel,
    PreparedPolygonMeshLocator,
)
from .._views import FieldTracePolicy, PreparedFieldReconstruction
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
    reference = ein.contract("cpnd,cnrd->cpnr", relative, inverse.value)
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
    all_values = ein.contract("cpna,cnai->cpni", barycentric, local_prolongation)
    reference_gradient = jnp.asarray(
        ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)), dtype=query.dtype
    )
    local_reference_gradients = ein.contract(
        "ar,cnai->cnir", reference_gradient, local_prolongation
    )
    all_gradients = ein.contract(
        "cnir,cnrd->cnid", local_reference_gradients, inverse.value
    )
    cell_rows = jnp.arange(indices.size)[:, None]
    point_rows = jnp.arange(query.shape[1])[None, :]
    selected_values = all_values[cell_rows, point_rows, choice]
    selected_gradients = all_gradients[cell_rows, choice]
    gathers = discretization.dof_map.cell_dofs[block][indices, :arity]
    coefficients = reconstruction.state[gathers]
    values = ein.contract("cpi,ci...->cp...", selected_values, coefficients)
    gradients = ein.contract("cpid,ci...->cp...d", selected_gradients, coefficients)
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
    polygonal = polygonal_connectivity_of(discretization.mesh)
    connectivity = jnp.asarray(polygonal.edges, dtype=jnp.int32)[edges]
    start = reconstruction.state[connectivity[:, 0]]
    stop = reconstruction.state[connectivity[:, 1]]
    value_rank = start.ndim - 1
    start_weight = (1.0 - parameter).reshape((1, parameter.size) + (1,) * value_rank)
    stop_weight = parameter.reshape((1, parameter.size) + (1,) * value_rank)
    return start[:, None] * start_weight + stop[:, None] * stop_weight


@final
class _ExplicitPolygonH1FanBasis(AbstractPolygonLocalBasis, NonTrainableState):
    """Condensed fan-P1 basis of every explicit polygon cell at located points.

    `prolongation` maps the padded vertex values of each cell to its fan nodes:
    rows `0..width-1` are the vertices and row `width` is the condensed witness.
    """

    prolongation: Array
    following: Array
    width: int = eqx.field(static=True)
    _basis_id: str = eqx.field(static=True)

    def __init__(
        self, prolongation: ArrayLike, following: ArrayLike, /, *, basis_id: str
    ) -> None:
        values = np.asarray(prolongation)
        routes = np.asarray(following, dtype=np.int32)
        if values.ndim != 3 or values.shape[1] != values.shape[2] + 1:
            raise ValueError(
                "Fan prolongations must have shape (cells, width + 1, width)."
            )
        if routes.shape != (values.shape[0], values.shape[2]):
            raise ValueError("Fan successor routes must have shape (cells, width).")
        self.prolongation = jnp.asarray(values)
        self.following = jnp.asarray(routes)
        self.width = values.shape[2]
        self._basis_id = basis_id

    @property
    def basis_id(self) -> str:
        return self._basis_id

    @property
    def local_width(self) -> int:
        return self.width

    @property
    def value_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def fan_continuity(self) -> int | None:
        return 0

    def weights(
        self, located: LocatedPolygonPoints, derivative: tuple[int, ...], /
    ) -> Array:
        cells = located.cells
        fans = located.fans
        nodes = jnp.stack(
            (
                jnp.full_like(fans, self.width),
                fans,
                self.following[cells, fans],
            ),
            axis=1,
        )
        local = self.prolongation[cells[:, None], nodes]
        if sum(derivative) == 0:
            coordinates = located.barycentric
        else:
            axis = derivative.index(1)
            coordinates = fan_barycentric_gradients(located.fan_inverse)[:, :, axis]
        return ein.contract("na,nal->nl", coordinates, local)


def _padded_fan_data(
    discretization: ExplicitPolygonH1Discretization,
    runtime: ExplicitPolygonH1RuntimeData,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host `(prolongation, witness, valid, cell_dofs)` over the global cells."""
    width = discretization.dof_map.local_width
    prolongations = []
    for block, basis in zip(discretization.mesh.blocks, runtime.bases, strict=True):
        values = np.asarray(basis.prolongation)
        padded = np.zeros((block.cell_count, width + 1, width), dtype=values.dtype)
        padded[:, : block.arity, : block.arity] = values[:, : block.arity]
        padded[:, width, : block.arity] = values[:, block.arity]
        prolongations.append(padded)
    return (
        np.concatenate(prolongations, axis=0),
        np.concatenate(tuple(np.asarray(basis.witness) for basis in runtime.bases)),
        np.concatenate(
            tuple(np.asarray(basis.evidence.passed) for basis in runtime.bases)
        ),
        np.concatenate(
            tuple(np.asarray(dofs) for dofs in discretization.dof_map.cell_dofs)
        ),
    )


def prepare_explicit_polygon_h1_field_reconstruction(
    discretization: ExplicitPolygonH1Discretization,
    /,
    *,
    runtime: ExplicitPolygonH1RuntimeData | None = None,
    value_port: ValuePort | None = None,
) -> PreparedFieldReconstruction:
    """Prepare the exact, evidenced coordinate reconstruction of the field.

    Points are located in the witness fans of the star-shaped cells (bounded
    BVH candidates, fixed point batches) and evaluated with the condensed
    fan-P1 basis: values are exact and `C^0`; first derivatives are exact and
    piecewise constant on the fan triangles. A derivative on a shared cell
    edge needs a bound trace side, and a derivative on an interior fan edge
    (including polygon vertices) is `SIDE_UNRESOLVED` for every side because
    the discrete gradient jumps there; no fan triangle is chosen silently.
    The support geometry is the region covered by the fans of `runtime`.
    """
    if not isinstance(discretization, ExplicitPolygonH1Discretization):
        raise TypeError("discretization must be ExplicitPolygonH1Discretization.")
    runtime_ = discretization.default_runtime if runtime is None else runtime
    discretization.validate_local_runtime(runtime_)
    space = discretization.field_space.vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError("Explicit polygon H1 fields are array valued.")
    connectivity = polygonal_connectivity_of(discretization.mesh)
    cells = np.asarray(connectivity.cell_vertices, dtype=np.int32)
    arity = np.asarray(connectivity.cell_kinds, dtype=np.int32)
    coordinates = np.asarray(runtime_.coordinates)
    prolongation, witness, valid, cell_dofs = _padded_fan_data(discretization, runtime_)
    policy = discretization.qualification_policy
    locator = PreparedPolygonMeshLocator(
        coordinates,
        cells,
        arity,
        witness,
        valid,
        tolerance=policy.tolerance_multiplier
        * float(
            np.finfo(np.dtype(discretization.precision_policy.factorization_dtype)).eps
        ),
        maximum_condition=policy.maximum_condition_number,
    )
    field_space_id = discretization.field_space.field_space_id
    support_id = canonical_fingerprint(
        {
            "kind": "explicit-polygon-h1-support",
            "topology": discretization.mesh.topology_id,
            "coordinates": array_tree_fingerprint(coordinates),
        }
    )
    _, following = polygon_fan_slots(arity, discretization.dof_map.local_width)
    basis = _ExplicitPolygonH1FanBasis(
        prolongation,
        following,
        basis_id=canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-fan-basis",
                "runtime": runtime_.runtime_id,
                "prolongation": array_tree_fingerprint(prolongation),
            }
        ),
    )
    kernel = PolygonFieldReconstructionKernel(
        locator,
        basis,
        cell_dofs,
        continuity=0,
        global_dof_count=discretization.dof_map.global_dof_count,
        field_space_id=field_space_id,
    )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=polygon_mesh_support_geometry(
            coordinates, cells, arity, witness, support_id
        ),
        value_port=polygon_value_port(
            discretization.field.name,
            space.shape[1:],
            "explicit-polygon-h1-field",
            field_space_id,
            value_port,
        ),
        regularity=DerivativeRegularity.piecewise_polynomial(
            continuity=0, degree_bound=1
        ),
        trace_policy=FieldTracePolicy("cell-sided"),
        coefficient_shape=space.shape,
        physical_dimension=2,
        maximum_derivative_order=1,
        field_space_id=field_space_id,
        support_id=support_id,
    )


__all__ = [
    "ExplicitPolygonH1Reconstruction",
    "evaluate_explicit_polygon_h1_reconstruction",
    "evaluate_explicit_polygon_h1_trace",
    "prepare_explicit_polygon_h1_field_reconstruction",
    "prepare_explicit_polygon_h1_reconstruction",
]
