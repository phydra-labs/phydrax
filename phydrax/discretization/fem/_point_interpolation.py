#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-element point evaluation from native tabulation and DOF routes.

`PreparedFiniteElementPointInterpolation` owns fixed host-cell routes (own
quadrature or node points) and their exact transpose scatter. The field
reconstruction kernel evaluates the same route formula at arbitrary physical
points located by an `AbstractCellLocator`; it never searches for a nearest
cell.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._differentiation import DerivativeRegularity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace
from .._simplicial_locator import (
    AbstractCellLocator,
    CellLocationStatus,
    PreparedSimplicialCellLocator,
    SimplicialLocationPolicy,
)
from .._view_support import (
    cell_location_status,
    simplicial_mesh_support_geometry,
    verify_mesh_support_geometry,
)
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTracePolicy,
    FieldTraceSide,
    InterpolationTransposeEvidence,
    PreparedFieldReconstruction,
    transpose_duality_evidence,
)
from ._cell_map import PreparedFiniteElementCellMap
from ._generic import FiniteElementDiscretization, FiniteElementRuntimeData
from ._reference import FiniteElementSpec


_SIMPLICES = ("triangle", "tetrahedron")


def _scalar_basis_element(
    discretization: FiniteElementDiscretization,
    field_name: str,
    block_index: int,
    /,
) -> FiniteElementSpec:
    field_index = discretization._field_index(field_name)
    element = discretization.elements[field_index][block_index]
    if (
        element.conformity not in ("H1", "L2")
        or element.mapping != "identity"
        or element.value_shape
    ):
        raise ValueError(
            "FE point evaluation requires a scalar-basis identity-mapped H1 or L2 field."
        )
    return element


def _field_array_space(
    discretization: FiniteElementDiscretization, field_name: str, /
) -> ArraySpace:
    field_index = discretization._field_index(field_name)
    space = discretization.field_spaces[field_index].vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError("FE point evaluation requires an array-valued FE field.")
    return space


def finite_element_point_weights(
    element: FiniteElementSpec,
    orientation: Array,
    reference_points: Array,
    inverse_jacobian: Array | None,
    derivative_axis: int | None,
    /,
) -> Array:
    """Oriented basis values or physical basis derivatives at reference points.

    `inverse_jacobian` has shape `(points, reference_dimension, ambient)`; the
    physical derivative along `derivative_axis` is the chain rule of the native
    reference gradients.
    """
    basis, gradients = element.tabulate(reference_points)
    if derivative_axis is None:
        weights = basis
    else:
        weights = contract(
            "plr,pr->pl", gradients, inverse_jacobian[:, :, derivative_axis]
        )
    return weights * orientation


class PreparedFiniteElementPointInterpolation(StrictModule, NonTrainableState):
    """Fixed FE point routes with an exact algebraic transpose scatter.

    Routes evaluate the field (or its physical derivative along
    `derivative_axis`) at fixed cell/reference points of one block, such as the
    element's own quadrature or node points. Fields have coefficient shape
    `(global_dof_count, *components)`.
    """

    discretization: FiniteElementDiscretization
    field_name: str = eqx.field(static=True)
    dof_routes: Array
    weights: Array
    reference_positions: Array
    dof_reference_positions: Array
    derivative_axis: int | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        field_name: str,
        dof_routes: ArrayLike,
        weights: ArrayLike,
        reference_positions: ArrayLike,
        dof_reference_positions: ArrayLike,
        /,
        *,
        derivative_axis: int | None = None,
        tolerance: float = 1.0e-10,
        prepared_id: str | None = None,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        name = str(field_name)
        field_space = _field_array_space(discretization, name)
        dimension = discretization.mesh.ambient_dimension
        routes = np.asarray(dof_routes, dtype=np.int32)
        weights_ = np.asarray(weights)
        positions = np.asarray(reference_positions)
        dof_positions = np.asarray(dof_reference_positions)
        limit = float(tolerance)
        if derivative_axis is not None and (
            isinstance(derivative_axis, bool)
            or not isinstance(derivative_axis, int)
            or not 0 <= derivative_axis < dimension
        ):
            raise ValueError(f"derivative_axis must be None or in [0, {dimension}).")
        if routes.ndim != 2 or routes.shape[0] == 0:
            raise ValueError("Interpolation routes must be a nonempty rank-2 array.")
        if weights_.shape != routes.shape:
            raise ValueError("Interpolation weights must match fixed route shape.")
        if positions.shape != (routes.shape[0], dimension):
            raise ValueError("Reference points must match interpolation count/dimension.")
        if dof_positions.shape != (field_space.shape[0], dimension):
            raise ValueError("DOF reference positions must match the FE field DOFs.")
        if (
            np.any(routes < 0)
            or np.any(routes >= field_space.shape[0])
            or np.any(~np.isfinite(weights_))
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(dof_positions))
        ):
            raise ValueError(
                "Interpolation routes and geometric data must be finite/valid."
            )
        # Values reproduce constants; derivatives annihilate them.
        target = 1.0 if derivative_axis is None else 0.0
        scale = 1.0 if derivative_axis is None else max(np.max(np.abs(weights_)), 1.0)
        partition_defect = np.max(np.abs(np.sum(weights_, axis=1) - target)) / scale
        if not isfinite(limit) or limit < 0.0 or partition_defect > limit:
            raise ValueError(
                "FE interpolation must reproduce constants within tolerance."
            )
        generated = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-point-interpolation",
                "discretization": discretization.prepared_id,
                "field": name,
                "routes": array_tree_fingerprint(routes),
                "weights": array_tree_fingerprint(weights_),
                "reference_positions": array_tree_fingerprint(positions),
                "derivative_axis": derivative_axis,
                "tolerance": limit.hex(),
            }
        )
        identifier = generated if prepared_id is None else str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be non-empty or None.")
        dtype = field_space.dtype
        self.discretization = discretization
        self.field_name = name
        self.dof_routes = jnp.asarray(routes)
        self.weights = jnp.asarray(weights_, dtype=dtype)
        self.reference_positions = jnp.asarray(positions, dtype=dtype)
        self.dof_reference_positions = jnp.asarray(dof_positions, dtype=dtype)
        self.derivative_axis = derivative_axis
        self.tolerance = limit
        self.prepared_id = identifier

    @property
    def attachment_count(self) -> int:
        return self.dof_routes.shape[0]

    @property
    def ambient_dimension(self) -> int:
        return self.reference_positions.shape[1]

    @property
    def field_space(self) -> ArraySpace:
        return _field_array_space(self.discretization, self.field_name)

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.field_space.shape[1:]

    def interpolate(self, coefficients: ArrayLike, /) -> Array:
        values = self.field_space.validate(coefficients)
        return contract("ai,ai...->a...", self.weights, values[self.dof_routes])

    def transpose_scatter(self, point_dual: ArrayLike, /) -> Array:
        dual = jnp.asarray(point_dual, dtype=self.weights.dtype)
        if dual.shape != (self.attachment_count, *self.value_shape):
            raise ValueError("Point dual must match attachment count and value shape.")
        payload = contract("ai,a...->ai...", self.weights, dual)
        return (
            jnp.zeros(self.field_space.shape, dtype=payload.dtype)
            .at[self.dof_routes]
            .add(payload)
        )

    def duality_evidence(
        self,
        coefficients: ArrayLike,
        point_dual: ArrayLike,
        /,
    ) -> InterpolationTransposeEvidence:
        values = self.field_space.validate(coefficients)
        dual = jnp.asarray(point_dual, dtype=values.dtype)
        return transpose_duality_evidence(
            self.interpolate(values),
            dual,
            values,
            self.transpose_scatter(dual),
            tolerance=self.tolerance,
        )

    def deformed_dof_positions(self, displacement: ArrayLike, /) -> Array:
        value = self.field_space.validate(displacement)
        return self.dof_reference_positions + value


def _realized_runtime(
    discretization: FiniteElementDiscretization,
    runtime: FiniteElementRuntimeData | None,
    /,
) -> FiniteElementRuntimeData:
    realized = discretization.default_runtime if runtime is None else runtime
    if not isinstance(realized, FiniteElementRuntimeData):
        raise TypeError("runtime must be FiniteElementRuntimeData or None.")
    if (
        realized.topology_id != discretization.mesh.topology_id
        or realized.geometry_layout_id
        != discretization.default_runtime.geometry_layout_id
    ):
        raise ValueError("Interpolation runtime does not match the FE discretization.")
    return realized


def prepare_finite_element_point_interpolation(
    discretization: FiniteElementDiscretization,
    field_name: str,
    block_name: str,
    cell_indices: ArrayLike,
    reference_points: ArrayLike,
    /,
    *,
    runtime: FiniteElementRuntimeData | None = None,
    derivative_axis: int | None = None,
    tolerance: float = 1.0e-10,
) -> PreparedFiniteElementPointInterpolation:
    """Prepare fixed block-local cell/point routes from native tabulation.

    With `derivative_axis`, the routes evaluate the exact physical derivative
    along that axis inside each named cell (the cell interior limit).
    """
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    block = str(block_name)
    if block not in dof_map.block_names:
        raise KeyError(f"Unknown FE block {block!r} for field {field_name!r}.")
    block_index = dof_map.block_names.index(block)
    element = _scalar_basis_element(discretization, field_name, block_index)
    cells = np.asarray(cell_indices, dtype=np.int32)
    points = np.asarray(reference_points)
    cell_count = discretization.mesh.blocks[block_index].cell_count
    if cells.ndim != 1 or cells.size == 0:
        raise ValueError("cell_indices must be a nonempty rank-1 array.")
    if points.shape != (cells.size, element.topological_dimension):
        raise ValueError("reference_points must provide one point per selected cell.")
    if np.any(cells < 0) or np.any(cells >= cell_count) or np.any(~np.isfinite(points)):
        raise ValueError("Interpolation cell routes/reference points are invalid.")
    realized = _realized_runtime(discretization, runtime)
    cell_map = PreparedFiniteElementCellMap(discretization, block_index)
    evaluation = cell_map.evaluate(
        realized.coordinates, jnp.asarray(cells), jnp.asarray(points)
    )
    if not bool(np.all(np.asarray(evaluation.valid))):
        raise ValueError("Interpolation cells have an invalid coordinate map.")
    orientation = jnp.asarray(dof_map.orientations[block_index])[cells]
    weights = finite_element_point_weights(
        element,
        orientation,
        jnp.asarray(points),
        evaluation.inverse_jacobian,
        derivative_axis,
    )
    routes = np.asarray(dof_map.cell_dofs[block_index])[cells]
    dof_positions = np.asarray(
        dof_map.evaluate_coordinates(discretization.mesh, realized.coordinates)
    )
    return PreparedFiniteElementPointInterpolation(
        discretization,
        field_name,
        routes,
        np.asarray(weights),
        np.asarray(evaluation.physical_points),
        dof_positions,
        derivative_axis=derivative_axis,
        tolerance=tolerance,
    )


class _FiniteElementRoute(StrictModule):
    dof_routes: Array
    weights: Array


@final
class FiniteElementFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Located evaluation of one scalar-basis FE field on one cell block.

    Each query point is located by the block's `AbstractCellLocator`; every
    containing cell contributes a candidate route built from native tabulation
    and oriented DOF routes. Values and first physical derivatives are exact
    inside cells. On a non-smooth locus (a point in several cells) a derivative
    above the field's continuity needs a bound trace side.
    """

    locator: AbstractCellLocator
    element: FiniteElementSpec
    cell_dofs: Array
    orientations: Array
    continuity: int = eqx.field(static=True)
    global_dof_count: int = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        locator: AbstractCellLocator,
        element: FiniteElementSpec,
        cell_dofs: ArrayLike,
        orientations: ArrayLike,
        /,
        *,
        continuity: int,
        global_dof_count: int,
        field_space_id: str,
    ):
        if not isinstance(locator, AbstractCellLocator):
            raise TypeError("locator must be an AbstractCellLocator.")
        if not isinstance(element, FiniteElementSpec):
            raise TypeError("element must be a FiniteElementSpec.")
        routes = jnp.asarray(cell_dofs)
        signs = jnp.asarray(orientations)
        expected = (locator.cell_map.cell_count, element.local_dof_count)
        if routes.shape != expected or signs.shape != expected:
            raise ValueError("Cell DOF routes and orientations must be (cells, local).")
        if continuity not in (-1, 0):
            raise ValueError("FE field continuity must be -1 (L2) or 0 (H1).")
        self.locator = locator
        self.element = element
        self.cell_dofs = routes
        self.orientations = signs
        self.continuity = continuity
        self.global_dof_count = global_dof_count
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "finite-element-field-reconstruction-kernel",
                "locator": locator.locator_id,
                "element": element.element_id,
                "field_space": field_space_id,
                "continuity": continuity,
            }
        )

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return self.locator.cell_map.cell_count

    @property
    def support_coverage(self) -> str:
        return "complete"

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[_FiniteElementRoute, FieldQueryEvidence]:
        order = sum(derivative)
        axis = None if order == 0 else derivative.index(1)
        mask = None if side is None else side.cell_mask
        location = self.locator.locate(points, cell_mask=mask)
        candidates = location.candidate_cells
        accepted = candidates >= 0
        count = jnp.sum(accepted, axis=1, dtype=jnp.int32)
        safe = jnp.where(accepted, candidates, 0)
        flat_cells = safe.reshape((-1,))
        reference = location.candidate_reference.reshape(
            (-1, self.locator.cell_map.reference_dimension)
        )
        inverse_jacobian = None
        if axis is not None:
            inverse_jacobian = self.locator.cell_map.evaluate(
                self.locator.coordinates, flat_cells, reference
            ).inverse_jacobian
        weights = finite_element_point_weights(
            self.element,
            self.orientations[flat_cells],
            reference,
            inverse_jacobian,
            axis,
        ).reshape((*candidates.shape, self.element.local_dof_count))
        if side is not None and side.side == "average":
            share = accepted / jnp.maximum(count, 1)[:, None]
        else:
            first = jnp.argmax(accepted, axis=1)
            share = (
                jnp.arange(candidates.shape[1])[None, :] == first[:, None]
            ) & accepted
        share = share.astype(weights.dtype)
        route = _FiniteElementRoute(self.cell_dofs[safe], weights * share[:, :, None])
        ambiguous = (count > 1) & (order > self.continuity)
        status = cell_location_status(location.status)
        if side is None:
            side_status = FieldQueryStatus.SIDE_REQUIRED
        else:
            side_status = FieldQueryStatus.SIDE_UNRESOLVED
        if side is None or side.side != "average":
            status = jnp.where(
                (status == int(FieldQueryStatus.VALID)) & ambiguous,
                int(side_status),
                status,
            )
        evidence = FieldQueryEvidence(
            status, location.jacobian_condition, count, kernel_id=self.kernel_id
        )
        return route, evidence

    def apply(self, route: _FiniteElementRoute, coefficients: Array, /) -> Array:
        return contract("pcl,pcl...->p...", route.weights, coefficients[route.dof_routes])

    def transpose(self, route: _FiniteElementRoute, cotangent: Array, /) -> Array:
        payload = contract("pcl,p...->pcl...", route.weights, cotangent)
        shape = (self.global_dof_count, *cotangent.shape[1:])
        return jnp.zeros(shape, dtype=payload.dtype).at[route.dof_routes].add(payload)

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        location = self.locator.locate(jnp.asarray(sites))
        status = np.asarray(location.status)
        if np.any(status != int(CellLocationStatus.LOCATED)):
            raise ValueError("Every trace site must lie in the FE support.")
        containing = np.asarray(location.candidate_cells)
        if side == "average":
            if cell_ids is not None:
                raise ValueError("Average traces combine every containing cell.")
            return None, np.full(sites.shape[0], -1, dtype=np.int32)
        if cell_ids is None:
            single = np.sum(containing >= 0, axis=1) == 1
            if side != "owner" or not np.all(single):
                raise ValueError(
                    f"{side!r} traces at sites shared by several cells (or without a "
                    "neighbor cell) require explicit cell_ids."
                )
            site_cells = np.max(containing, axis=1).astype(np.int32)
        else:
            if np.any(np.all(containing != cell_ids[:, None], axis=1)):
                raise ValueError("cell_ids must name a cell containing each trace site.")
            site_cells = cell_ids
        mask = np.zeros((self.cell_count,), dtype=np.bool_)
        mask[site_cells] = True
        masked = self.locator.locate(jnp.asarray(sites), cell_mask=jnp.asarray(mask))
        if np.any(np.asarray(masked.candidate_count) != 1) or np.any(
            np.asarray(masked.cell_ids) != site_cells
        ):
            raise ValueError(
                "The side cells do not resolve every trace site to exactly its "
                "declared cell; the side region is ambiguous."
            )
        return mask, site_cells


def prepare_finite_element_field_reconstruction(
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
    *,
    block_name: str | None = None,
    locator: AbstractCellLocator | None = None,
    location_policy: SimplicialLocationPolicy | None = None,
    support_geometry: Any = None,
    value_port: ValuePort | None = None,
    runtime: FiniteElementRuntimeData | None = None,
    support_tolerance: float = 1.0e-9,
) -> PreparedFieldReconstruction:
    """Prepare an evidenced coordinate reconstruction of one FE field.

    Arbitrary points are located on triangle/tetrahedron blocks by a
    `PreparedSimplicialCellLocator` (built from `location_policy`); other cell
    kinds require an explicit `locator` implementing `AbstractCellLocator`.
    `support_geometry` defaults to the region derived from affine simplicial
    meshes; an explicit geometry must be covered by the mesh (vertices inside,
    equal measure). Regularity is `C^0` (H1) or `C^-1` (L2) with polynomial
    pieces of the element degree on affine cells and smooth pieces otherwise.
    """
    from ...geometry import CompiledGeometry

    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    if block_name is None:
        if len(dof_map.block_names) != 1:
            raise ValueError("Multi-block FE fields require an explicit block_name.")
        block_index = 0
    else:
        if block_name not in dof_map.block_names:
            raise KeyError(f"Unknown FE block {block_name!r} for field {field_name!r}.")
        block_index = dof_map.block_names.index(block_name)
    if len(dof_map.block_names) != 1:
        raise ValueError(
            "A field view covers one complete support; multi-block FE fields need one "
            "view per block support."
        )
    element = _scalar_basis_element(discretization, field_name, block_index)
    realized = _realized_runtime(discretization, runtime)
    cell_map = PreparedFiniteElementCellMap(discretization, block_index)
    if locator is None:
        if cell_map.coordinate_element.cell_kind not in _SIMPLICES:
            raise ValueError(
                f"Arbitrary-point evaluation on {cell_map.coordinate_element.cell_kind!r} "
                "cells requires an explicit AbstractCellLocator inverse provider."
            )
        policy = (
            SimplicialLocationPolicy(min(cell_map.cell_count, 16), 16, 1)
            if location_policy is None
            else location_policy
        )
        locator = PreparedSimplicialCellLocator(cell_map, realized.coordinates, policy)
    elif not isinstance(locator, AbstractCellLocator):
        raise TypeError("locator must be an AbstractCellLocator or None.")
    elif locator.cell_map.cell_map_id != cell_map.cell_map_id or not np.array_equal(
        np.asarray(locator.coordinates), np.asarray(realized.coordinates)
    ):
        raise ValueError("The locator does not invert this FE block and runtime.")
    space = _field_array_space(discretization, field_name)
    field_space_id = canonical_fingerprint(
        {
            "kind": "finite-element-field-space",
            "discretization": discretization.prepared_id,
            "field": str(field_name),
        }
    )
    support_id = canonical_fingerprint(
        {
            "kind": "finite-element-support",
            "topology": cell_map.topology_id,
            "cell_map": cell_map.cell_map_id,
            "coordinates": array_tree_fingerprint(np.asarray(realized.coordinates)),
        }
    )
    if support_geometry is None:
        geometry = simplicial_mesh_support_geometry(locator, support_id)
    elif isinstance(support_geometry, CompiledGeometry):
        verify_mesh_support_geometry(
            support_geometry, locator, tolerance=support_tolerance
        )
        geometry = support_geometry
    else:
        raise TypeError("support_geometry must be a CompiledGeometry or None.")
    continuity = 0 if element.conformity == "H1" else -1
    affine = cell_map.coordinate_element.degree == 1
    if element.degree == 0:
        regularity = DerivativeRegularity.piecewise_polynomial(
            continuity=-1, degree_bound=0
        )
    elif affine:
        regularity = DerivativeRegularity.piecewise_polynomial(
            continuity=continuity, degree_bound=element.degree
        )
    else:
        regularity = DerivativeRegularity.piecewise_smooth(continuity=continuity)
    components = space.shape[1:]
    port = (
        ValuePort(
            str(field_name),
            event_shape=components,
            component_ids=(
                (str(field_name),)
                if not components
                else tuple(
                    f"{field_name}[{index}]" for index in range(int(np.prod(components)))
                )
            ),
            representation="finite-element-field",
            space_id=field_space_id,
        )
        if value_port is None
        else value_port
    )
    if not isinstance(port, ValuePort):
        raise TypeError("value_port must be a ValuePort or None.")
    if port.event_shape != components:
        raise ValueError("value_port event_shape must equal the FE component shape.")
    kernel = FiniteElementFieldReconstructionKernel(
        locator,
        element,
        dof_map.cell_dofs[block_index],
        dof_map.orientations[block_index],
        continuity=continuity,
        global_dof_count=dof_map.global_dof_count,
        field_space_id=field_space_id,
    )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=geometry,
        value_port=port,
        regularity=regularity,
        trace_policy=FieldTracePolicy("cell-sided"),
        coefficient_shape=space.shape,
        physical_dimension=discretization.mesh.ambient_dimension,
        maximum_derivative_order=min(element.degree, 1),
        field_space_id=field_space_id,
        support_id=support_id,
    )


__all__ = [
    "FiniteElementFieldReconstructionKernel",
    "PreparedFiniteElementPointInterpolation",
    "finite_element_point_weights",
    "prepare_finite_element_field_reconstruction",
    "prepare_finite_element_point_interpolation",
]
