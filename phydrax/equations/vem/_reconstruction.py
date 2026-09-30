#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Projected interior reconstructions and exact edge traces of VEM fields.

Virtual-element interiors are not evaluated directly: the H1 energy projection
and the (enhanced) L2 projection are computable polynomial images of the
virtual field and are published as separately labeled channels. Exact edge
traces are a different capability owned by the discretization.
"""

from __future__ import annotations

import math
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._differentiation import DerivativeRegularity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model._ports import ValuePort
from ..._polynomial import ScaledMonomialBasis
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import FieldTracePolicy, PreparedFieldReconstruction
from ...discretization._polygon_query import (
    AbstractPolygonLocalBasis,
    LocatedPolygonPoints,
    polygon_mesh_support_geometry,
    polygon_value_port,
    polygonal_connectivity_of,
    PolygonFieldReconstructionKernel,
    PreparedPolygonMeshLocator,
)
from ...discretization.vem import (
    VirtualElementDiscretization,
    VirtualElementRuntimeData,
)
from ...discretization.vem._space import virtual_element_runtime_matches
from ...typing import parse


VirtualElementReconstructionChannel: TypeAlias = Literal["h1-projection", "l2-projection"]


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
    if not virtual_element_runtime_matches(discretization, runtime_):
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
        l2.append(ein.contract("cai,ci->ca", projection.l2_coefficients, local))
        h1.append(ein.contract("cai,ci->ca", projection.h1_coefficients, local))
        differential.append(
            ein.contract("cai,ci->ca", projection.differential_coefficients, local)
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
        value = ein.contract("cqa,cda->cqd", basis, vector_coefficients)
        differential_basis = ScaledMonomialBasis(2, projection.differential_degree)
        differential_values = differential_basis.evaluate(
            points_,
            geometry.centroids[indices],
            geometry.characteristic_lengths[indices],
        )
        differential = ein.contract(
            "cqa,ca->cq",
            differential_values,
            reconstruction.differential_coefficients[block][indices],
        )
        return value, differential
    value = ein.contract("cqa,ca->cq", basis, coefficients)
    if projection.family == "DiscontinuousL2":
        return value, None
    gradient_basis = projection.basis.gradient(
        points_,
        geometry.centroids[indices],
        geometry.characteristic_lengths[indices],
    )
    gradient = ein.contract(
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
    """Evaluate the value, canonical normal, or canonical tangential edge trace.

    `parameters` lie in `[-1, 1]` along each edge from its lower to its higher
    vertex index; normal and tangential traces refer to that canonical edge
    orientation.
    """
    if not isinstance(reconstruction, VirtualElementReconstruction):
        raise TypeError("reconstruction must be VirtualElementReconstruction.")
    if not isinstance(discretization, VirtualElementDiscretization):
        raise TypeError("discretization must be VirtualElementDiscretization.")
    if reconstruction.field_space_id != discretization.field_space.field_space_id:
        raise ValueError("Reconstruction field space does not match the discretization.")
    if discretization.field.element.trace_kind == "none":
        raise ValueError("Discontinuous L2 virtual elements have no boundary trace.")
    edges = jnp.asarray(edge_indices, dtype=jnp.int32)
    if edges.ndim != 1:
        raise ValueError("Trace edge indices must be one rank-1 array.")
    parameters_ = jnp.asarray(parameters)
    if parameters_.ndim != 1:
        raise ValueError("Trace parameters must be one rank-1 array on [-1, 1].")
    basis = discretization.field.element.edge_trace_basis(parameters_)
    gathered = reconstruction.state[discretization.edge_trace_routes(edges)]
    return ein.contract("qi,ei->eq", basis, gathered)


# Fan location reuses the explicit polygon qualification defaults: a relative
# fan-coordinate tolerance of 4096 unit roundoffs and a Jacobian condition cap.
_LOCATION_TOLERANCE_MULTIPLIER = 4096.0
_LOCATION_MAXIMUM_CONDITION = 1.0e12


@final
class _VirtualElementProjectionBasis(AbstractPolygonLocalBasis, NonTrainableState):
    """Projected polynomial of every VEM cell as weights of its oriented DOFs.

    `coefficients[c]` maps the oriented local DOFs of cell `c` to the scaled
    monomial coefficients of the projection (component-major for vector
    families); padded local slots are zero.
    """

    monomials: ScaledMonomialBasis
    coefficients: Array
    centroids: Array
    scales: Array
    projected_shape: tuple[int, ...] = eqx.field(static=True)
    _basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        monomials: ScaledMonomialBasis,
        coefficients: ArrayLike,
        centroids: ArrayLike,
        scales: ArrayLike,
        /,
        *,
        projected_shape: tuple[int, ...],
        basis_id: str,
    ) -> None:
        values = np.asarray(coefficients)
        components = math.prod(projected_shape)
        if values.ndim != 3 or values.shape[1] != components * monomials.feature_count:
            raise ValueError(
                "Projection coefficients must have shape "
                "(cells, components * monomials, local_width)."
            )
        self.monomials = monomials
        self.coefficients = jnp.asarray(values)
        self.centroids = jnp.asarray(centroids)
        self.scales = jnp.asarray(scales)
        self.projected_shape = projected_shape
        self._basis_id = basis_id

    @property
    def basis_id(self) -> str:
        return self._basis_id

    @property
    def local_width(self) -> int:
        return self.coefficients.shape[2]

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.projected_shape

    @property
    def fan_continuity(self) -> int | None:
        return None

    def cell_weights(
        self, cells: Array, points: Array, derivative: tuple[int, ...], /
    ) -> Array:
        """Weights `(points, local_width, *value_shape)` at points of known cells."""
        centers = self.centroids[cells]
        scales = self.scales[cells]
        if sum(derivative) == 0:
            monomials = self.monomials.evaluate(points, centers, scales)
        else:
            monomials = self.monomials.gradient(points, centers, scales)[
                ..., derivative.index(1)
            ]
        coefficients = self.coefficients[cells].reshape(
            (cells.shape[0], -1, self.monomials.feature_count, self.local_width)
        )
        weights = ein.contract("na,ncal->nlc", monomials, coefficients)
        return weights.reshape(weights.shape[:2] + self.projected_shape)

    def weights(
        self, located: LocatedPolygonPoints, derivative: tuple[int, ...], /
    ) -> Array:
        return self.cell_weights(located.cells, located.points, derivative)


def _require_channel(
    discretization: VirtualElementDiscretization,
    channel: VirtualElementReconstructionChannel,
    /,
) -> None:
    family = discretization.field.element.family
    match channel:
        case "h1-projection":
            if family != "ConformingH1":
                raise ValueError(
                    "The H1 energy projection is defined for scalar ConformingH1 "
                    f"virtual elements; {family} fields publish "
                    "channel='l2-projection'."
                )
        case "l2-projection":
            pass
        case _:
            assert_never(channel)


def _padded_projection_data(
    discretization: VirtualElementDiscretization,
    runtime: VirtualElementRuntimeData,
    channel: VirtualElementReconstructionChannel,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Host oriented projection maps and cell DOFs padded to one local width."""
    element = discretization.field.element
    width = max(
        element.local_dof_count(block.arity) for block in discretization.mesh.blocks
    )
    maps = []
    routes = []
    for projection, dofs, orientation in zip(
        runtime.projections,
        discretization.dof_map.cell_dofs,
        discretization.dof_map.orientations,
        strict=True,
    ):
        match channel:
            case "h1-projection":
                coefficients = np.asarray(projection.h1_coefficients)
            case "l2-projection":
                coefficients = np.asarray(projection.l2_coefficients)
            case _:
                assert_never(channel)
        oriented = coefficients * np.asarray(orientation)[:, None, :]
        padding = width - oriented.shape[2]
        maps.append(np.pad(oriented, ((0, 0), (0, 0), (0, padding))))
        routes.append(np.pad(np.asarray(dofs), ((0, 0), (0, padding))))
    return np.concatenate(maps, axis=0), np.concatenate(routes, axis=0)


def _polygon_locator(
    discretization: VirtualElementDiscretization,
    runtime: VirtualElementRuntimeData,
    /,
) -> tuple[PreparedPolygonMeshLocator, np.ndarray]:
    """Fan locator over the star-kernel witnesses of the runtime geometry."""
    connectivity = polygonal_connectivity_of(discretization.mesh)
    witness = np.concatenate(
        tuple(
            np.asarray(
                ein.contract(
                    "cv,cvd->cd", triangulation.witness_weights, geometry.vertices
                )
            )
            for triangulation, geometry in zip(
                discretization.triangulations, runtime.geometries, strict=True
            )
        )
    )
    valid = np.concatenate(
        tuple(
            np.asarray(geometry.evidence.valid & projection.evidence.factorization_valid)
            for geometry, projection in zip(
                runtime.geometries, runtime.projections, strict=True
            )
        )
    )
    dtype = np.dtype(discretization.precision_policy.geometry_dtype)
    locator = PreparedPolygonMeshLocator(
        np.asarray(runtime.coordinates),
        np.asarray(connectivity.cell_vertices, dtype=np.int32),
        np.asarray(connectivity.cell_kinds, dtype=np.int32),
        witness,
        valid,
        tolerance=_LOCATION_TOLERANCE_MULTIPLIER * float(np.finfo(dtype).eps),
        maximum_condition=_LOCATION_MAXIMUM_CONDITION,
    )
    return locator, witness


def virtual_element_projection_basis(
    discretization: VirtualElementDiscretization,
    runtime: VirtualElementRuntimeData,
    channel: VirtualElementReconstructionChannel,
    /,
) -> tuple[_VirtualElementProjectionBasis, np.ndarray]:
    """Return the projected-channel local basis and its padded cell DOF routes."""
    maps, routes = _padded_projection_data(discretization, runtime, channel)
    projection = runtime.projections[0]
    projected_shape = (
        () if channel == "h1-projection" else projection.polynomial_value_shape
    )
    basis = _VirtualElementProjectionBasis(
        projection.basis,
        maps,
        jnp.concatenate(tuple(geometry.centroids for geometry in runtime.geometries)),
        jnp.concatenate(
            tuple(geometry.characteristic_lengths for geometry in runtime.geometries)
        ),
        projected_shape=projected_shape,
        basis_id=canonical_fingerprint(
            {
                "kind": "virtual-element-projection-basis",
                "channel": channel,
                "runtime": runtime.runtime_id,
                "maps": array_tree_fingerprint(maps),
            }
        ),
    )
    return basis, routes


def prepare_virtual_element_field_reconstruction(
    discretization: VirtualElementDiscretization,
    /,
    *,
    channel: VirtualElementReconstructionChannel,
    runtime: VirtualElementRuntimeData | None = None,
    value_port: ValuePort | None = None,
) -> PreparedFieldReconstruction:
    """Prepare one labeled projected interior channel of a VEM field.

    The virtual interior basis is never evaluated: `"h1-projection"` evaluates
    the energy projection `Pi^nabla u` of scalar `ConformingH1` fields and
    `"l2-projection"` the (enhanced) L2 projection `Pi^0 u` of every family
    (vector valued for H(div)/H(curl)). The route is linear in the DOFs:
    projection coefficients times the oriented local DOFs times the scaled
    monomials at the located point, with values and first derivatives exact
    for the projected polynomial. The reconstruction's `approximation` is the
    channel, so consumers can distinguish it from the exact edge traces of
    `VirtualElementDiscretization.prepare_side_trace`. The projections are
    discontinuous across cells (`C^-1`, cell-sided): points on shared edges
    need a bound trace side.
    """
    if not isinstance(discretization, VirtualElementDiscretization):
        raise TypeError("discretization must be VirtualElementDiscretization.")
    channel_ = parse(channel, VirtualElementReconstructionChannel, "channel")
    _require_channel(discretization, channel_)
    runtime_ = discretization.default_runtime if runtime is None else runtime
    if not virtual_element_runtime_matches(discretization, runtime_):
        raise ValueError("VEM reconstruction runtime is incompatible with the space.")
    locator, witness = _polygon_locator(discretization, runtime_)
    basis, routes = virtual_element_projection_basis(discretization, runtime_, channel_)
    field_space_id = discretization.field_space.field_space_id
    coordinates = np.asarray(runtime_.coordinates)
    support_id = canonical_fingerprint(
        {
            "kind": "virtual-element-support",
            "topology": discretization.mesh.topology_id,
            "coordinates": array_tree_fingerprint(coordinates),
        }
    )
    connectivity = polygonal_connectivity_of(discretization.mesh)
    kernel = PolygonFieldReconstructionKernel(
        locator,
        basis,
        routes,
        continuity=-1,
        global_dof_count=discretization.dof_map.global_dof_count,
        field_space_id=field_space_id,
    )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=polygon_mesh_support_geometry(
            coordinates,
            np.asarray(connectivity.cell_vertices, dtype=np.int32),
            np.asarray(connectivity.cell_kinds, dtype=np.int32),
            witness,
            support_id,
        ),
        value_port=polygon_value_port(
            discretization.field.name,
            basis.value_shape,
            f"virtual-element-{channel_}",
            field_space_id,
            value_port,
            form=discretization.field.element.value_spec,
        ),
        regularity=DerivativeRegularity.piecewise_polynomial(
            continuity=-1, degree_bound=discretization.field.element.degree
        ),
        trace_policy=FieldTracePolicy("cell-sided"),
        coefficient_shape=(discretization.dof_map.global_dof_count,),
        physical_dimension=2,
        maximum_derivative_order=1,
        field_space_id=field_space_id,
        support_id=support_id,
        approximation=channel_,
    )


__all__ = [
    "VirtualElementReconstruction",
    "VirtualElementReconstructionChannel",
    "evaluate_virtual_element_reconstruction",
    "evaluate_virtual_element_trace",
    "prepare_virtual_element_field_reconstruction",
    "project_virtual_element_field",
    "virtual_element_projection_basis",
]
