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

from dataclasses import dataclass
from math import isfinite
from typing import Any, assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._differentiation import DerivativeRegularity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._algebra import map_reference_values
from ...exterior._form_type import FormType, FormValueSpec
from ...linalg import ArraySpace
from ...typing import parse
from .._integration_domain import IntegrationDomain
from .._reference_cell import FacetShape, reference_cell_topology, ReferenceCellTopology
from .._side_actions import (
    FacetTraceRule,
    PreparedTraceAction,
    SideActionDescriptor,
    SideGatherRoute,
    SideTraceQuantity,
)
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
from ._reference_operator import (
    _facet_corner_parameters,
    _reference_facet_shape,
    reference_facet_embedding,
)


_SIMPLICES = ("triangle", "tetrahedron")


def _field_element(
    discretization: FiniteElementDiscretization,
    field_name: str,
    block_index: int,
    /,
) -> FiniteElementSpec:
    field_index = discretization._field_index(field_name)
    element = discretization.elements[field_index][block_index]
    if element.form_basis is None and (
        element.mapping != "identity"
        or element.value_shape
        or element.value_spec.form_type.twist == "twisted"
    ):
        raise ValueError(
            "Mapped or twisted FE fields require canonical form-basis metadata."
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
    *,
    jacobian: Array | None = None,
    transform: Array | None = None,
) -> Array:
    """Oriented basis values or physical basis derivatives at reference points.

    `inverse_jacobian` has shape `(points, reference_dimension, ambient)`; the
    physical derivative along `derivative_axis` is the chain rule of the native
    reference gradients.
    """
    basis, gradients = element.tabulate(reference_points)
    if element.mapping != "identity" or element.value_spec.form_type.twist == "twisted":
        if jacobian is None:
            raise TypeError("Mapped point weights require a coordinate Jacobian.")
        basis = map_reference_values(basis, element.value_spec, jacobian[:, None])
        if derivative_axis is not None:
            gradients = jnp.stack(
                tuple(
                    map_reference_values(
                        gradients[..., axis], element.value_spec, jacobian[:, None]
                    )
                    for axis in range(element.topological_dimension)
                ),
                axis=-1,
            )
    if derivative_axis is None:
        weights = basis
    else:
        if inverse_jacobian is None:
            raise TypeError("derivative_axis requires inverse_jacobian.")
        flat = gradients.reshape(
            (gradients.shape[0], gradients.shape[1], -1, gradients.shape[-1])
        )
        weights = contract("plvr,pr->plv", flat, inverse_jacobian[:, :, derivative_axis])
        weights = weights.reshape((*basis.shape[:2], *basis.shape[2:]))
    if transform is not None:
        flat = weights.reshape((weights.shape[0], weights.shape[1], -1))
        weights = contract("plv,plj->pjv", flat, transform).reshape(weights.shape)
    return weights * orientation.reshape(
        (*orientation.shape, *((1,) * (weights.ndim - orientation.ndim)))
    )


@final
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
    ) -> None:
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
        if weights_.shape[:2] != routes.shape:
            raise ValueError(
                "Interpolation weights must match the fixed point/local axes."
            )
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
        element = discretization.elements[discretization._field_index(name)][0]
        if element.form_basis is None:
            target = 1.0 if derivative_axis is None else 0.0
            scale = 1.0 if derivative_axis is None else max(np.max(np.abs(weights_)), 1.0)
            defect = np.max(np.abs(np.sum(weights_, axis=1) - target)) / scale
            if not isfinite(limit) or limit < 0.0 or defect > limit:
                raise ValueError(
                    "FE interpolation must reproduce constants within tolerance."
                )
        elif not isfinite(limit) or limit < 0.0:
            raise ValueError("Interpolation tolerance must be finite and nonnegative.")
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
        self.reference_positions = jnp.asarray(positions)
        self.dof_reference_positions = jnp.asarray(dof_positions)
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
        return self.weights.shape[2:] + self.field_space.shape[1:]

    def interpolate(self, coefficients: ArrayLike, /) -> Array:
        values = self.field_space.validate(coefficients)
        if self.weights.ndim == 2:
            return contract("ai,ai...->a...", self.weights, values[self.dof_routes])
        return contract("aiv,ai->av", self.weights, values[self.dof_routes])

    def transpose_scatter(self, point_dual: ArrayLike, /) -> Array:
        dual = jnp.asarray(point_dual, dtype=self.weights.dtype)
        if dual.shape != (self.attachment_count, *self.value_shape):
            raise ValueError("Point dual must match attachment count and value shape.")
        payload = (
            contract("ai,a...->ai...", self.weights, dual)
            if self.weights.ndim == 2
            else contract("aiv,av->ai", self.weights, dual)
        )
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
    element = _field_element(discretization, field_name, block_index)
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
        jacobian=evaluation.jacobian,
        transform=dof_map.cell_transforms[block_index][cells],
    )
    if element.form_basis is not None and derivative_axis is not None:
        value_shape = weights.shape[2:]

        def mapped(reference: Array, cell: Array, /) -> Array:
            geometry = cell_map.evaluate(
                realized.coordinates, cell[None], reference[None]
            )
            values = map_reference_values(
                element.tabulate(reference[None])[0],
                element.value_spec,
                geometry.jacobian[:, None],
            )[0]
            transformed = contract(
                "iv,ij->jv",
                values.reshape((element.local_dof_count, -1)),
                dof_map.cell_transforms[block_index][cell],
            )
            return transformed

        gradient = jax.vmap(jax.jacfwd(mapped, argnums=0))(
            jnp.asarray(points), jnp.asarray(cells)
        )
        weights = contract(
            "plvr,pr->plv", gradient, evaluation.inverse_jacobian[:, :, derivative_axis]
        )
        weights = weights.reshape((cells.size, element.local_dof_count, *value_shape))
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
    ) -> None:
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
    whole_mesh = block_name is None and len(dof_map.block_names) > 1
    if block_name is None:
        block_index = 0
    else:
        if block_name not in dof_map.block_names:
            raise KeyError(f"Unknown FE block {block_name!r} for field {field_name!r}.")
        block_index = dof_map.block_names.index(block_name)
    element = _field_element(discretization, field_name, block_index)
    realized = _realized_runtime(discretization, runtime)
    if whole_mesh:
        if any(
            other.tabulator_id != element.tabulator_id
            or other.degree != element.degree
            or other.value_spec.value_spec_id != element.value_spec.value_spec_id
            for other in discretization.elements[field_index]
        ):
            raise ValueError(
                "A whole-support reconstruction requires one canonical reference field basis."
            )
        cell_map = PreparedFiniteElementCellMap(discretization, None)
        cell_dofs = jnp.concatenate(dof_map.cell_dofs, axis=0)
        cell_transforms = jnp.concatenate(dof_map.cell_transforms, axis=0)
        orientations = jnp.concatenate(dof_map.orientations, axis=0)
    else:
        cell_map = PreparedFiniteElementCellMap(discretization, block_index)
        cell_dofs = dof_map.cell_dofs[block_index]
        cell_transforms = dof_map.cell_transforms[block_index]
        orientations = dof_map.orientations[block_index]
    if locator is None:
        kind = cell_map.coordinate_element.cell_kind
        policy = (
            SimplicialLocationPolicy(min(cell_map.cell_count, 16), 16, 1)
            if location_policy is None
            else location_policy
        )
        if kind in ("quadrilateral", "hexahedron") or kind.startswith("tensor:"):
            from ._form_reconstruction import _TensorCellLocator

            locator = _TensorCellLocator(cell_map, realized.coordinates, policy)
        else:
            locator = PreparedSimplicialCellLocator(
                cell_map, realized.coordinates, policy
            )
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
    scientific = element.value_spec.form_type
    physical_spec = FormValueSpec(
        FormType(
            scientific.dimension,
            scientific.degree,
            twist=scientific.twist,
            fiber_shape=scientific.fiber_shape,
            ambient_dimension=discretization.mesh.ambient_dimension,
        ),
        proxy=element.value_spec.proxy,
    )
    components = physical_spec.value_shape + space.shape[1:]
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
            form=physical_spec if element.form_basis is not None else None,
        )
        if value_port is None
        else value_port
    )
    if not isinstance(port, ValuePort):
        raise TypeError("value_port must be a ValuePort or None.")
    if port.event_shape != components:
        raise ValueError("value_port event_shape must equal the FE component shape.")
    if element.form_basis is not None:
        from ._form_reconstruction import FormFieldReconstructionKernel

        kernel = FormFieldReconstructionKernel(
            locator,
            element,
            cell_dofs,
            cell_transforms,
            global_dof_count=dof_map.global_dof_count,
            field_space_id=field_space_id,
        )
    else:
        kernel = FiniteElementFieldReconstructionKernel(
            locator,
            element,
            cell_dofs,
            orientations,
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


# Owner-facet corners spanned by the unit facet parameter axes.
_FACET_AXIS_CORNERS: dict[FacetShape, tuple[int, ...]] = {
    "point": (),
    "edge": (1,),
    "triangle": (1, 2),
    "quadrilateral": (1, 3),
}


@dataclass(frozen=True, slots=True)
class _SideSelection:
    """Host routes of the selected facets on the traced and owner sides."""

    neighbor: bool
    side_cells: np.ndarray
    side_local: np.ndarray
    owner_cells: np.ndarray
    owner_local: np.ndarray


@dataclass(frozen=True, slots=True)
class _SideGeometry:
    """Reference and physical geometry of the side cells at the facet sites.

    `inverse_jacobian[f, q, r, i]` is `d xi_r / d x_i` of the side-cell map.
    """

    reference: np.ndarray
    sites: np.ndarray
    scaled_normals: np.ndarray
    inverse_jacobian: np.ndarray
    coordinate_degree: int


@dataclass(frozen=True, slots=True)
class _SideTabulation:
    """Padded per-facet gathers, oriented basis values, and side geometry."""

    dofs: np.ndarray
    valid: np.ndarray
    basis: np.ndarray
    sites: np.ndarray
    weights: np.ndarray
    normals: np.ndarray
    trace_degree: int | None


def _side_trace_field(
    discretization: FiniteElementDiscretization,
    field_name: str,
    quantity: SideTraceQuantity,
    /,
) -> tuple[int, ArraySpace]:
    """Validate the traced field and the requested trace quantity."""
    field_index = discretization._field_index(field_name)
    space = _field_array_space(discretization, field_name)
    components = discretization.elements[field_index][0].value_shape + space.shape[1:]
    dimension = discretization.mesh.ambient_dimension
    if discretization.mesh.topological_dimension != dimension:
        raise ValueError(
            "Side traces require a mesh whose topological and ambient dimensions "
            "agree; embedded manifolds have no unique facet normal."
        )
    match quantity:
        case "value":
            pass
        case "normal" | "tangential":
            if components != (dimension,):
                raise ValueError(
                    f"{quantity!r} traces require a vector field with component "
                    f"shape ({dimension},)."
                )
        case "conormal-flux":
            raise ValueError(
                "Conormal fluxes are published by compiled physics owners through "
                "prepare_conormal_flux, not by the discretization."
            )
        case _:
            assert_never(quantity)
    return field_index, space


def _require_own_facet_domain(
    domain: IntegrationDomain, base: IntegrationDomain, /
) -> None:
    """Refuse facet domains whose routes differ from this discretization's."""
    facets = np.asarray(domain.entity_indices, dtype=np.int32)
    base_facets = np.asarray(base.entity_indices, dtype=np.int32)
    rows = np.minimum(np.searchsorted(base_facets, facets), base_facets.size - 1)
    routes = (
        (domain.owner_cells, base.owner_cells),
        (domain.neighbor_cells, base.neighbor_cells),
        (domain.owner_local_entities, base.owner_local_entities),
        (domain.neighbor_local_entities, base.neighbor_local_entities),
    )
    if (
        domain.support_id != base.support_id
        or domain.entity_set_id != base.entity_set_id
        or base_facets.size == 0
        or not np.array_equal(base_facets[rows], facets)
        or any(
            not np.array_equal(np.asarray(value), np.asarray(expected)[rows])
            for value, expected in routes
        )
    ):
        raise ValueError(
            "The facet domain was not produced by this finite-element discretization."
        )


def _side_trace_selection(
    discretization: FiniteElementDiscretization,
    domain: IntegrationDomain,
    side: FieldTraceSide,
    /,
) -> _SideSelection:
    """Verify the domain belongs to this discretization and resolve its side."""
    if not isinstance(domain, IntegrationDomain):
        raise TypeError("domain must be an IntegrationDomain.")
    if domain.entity_indices.size == 0:
        raise ValueError("A side trace requires at least one selected facet.")
    match domain.kind:
        case "exterior_facet":
            _require_own_facet_domain(domain, discretization.exterior_facet_domain)
        case "interior_facet":
            _require_own_facet_domain(domain, discretization.interior_facet_domain)
        case _:
            raise ValueError("Side traces act on exterior or interior facet domains.")
    owner_cells = np.asarray(domain.owner_cells, dtype=np.int32)
    owner_local = np.asarray(domain.owner_local_entities, dtype=np.int32)
    match side:
        case "owner":
            return _SideSelection(
                False, owner_cells, owner_local, owner_cells, owner_local
            )
        case "neighbor":
            if domain.kind != "interior_facet":
                raise ValueError(
                    "Exterior facets have no neighbor side; trace the owner side."
                )
            return _SideSelection(
                True,
                np.asarray(domain.neighbor_cells, dtype=np.int32),
                np.asarray(domain.neighbor_local_entities, dtype=np.int32),
                owner_cells,
                owner_local,
            )
        case "average":
            raise ValueError(
                "Average side traces are not prepared; compose the owner and "
                "neighbor traces of the interior facets instead."
            )
        case _:
            assert_never(side)


def _cell_blocks(
    discretization: FiniteElementDiscretization, cells: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Map global cell indices to `(block index, block-local cell index)`."""
    counts = [block.cell_count for block in discretization.mesh.blocks]
    starts = np.concatenate(([0], np.cumsum(counts)))
    blocks = np.searchsorted(starts, cells, side="right") - 1
    return blocks.astype(np.int32), (cells - starts[blocks]).astype(np.int32)


def _local_facet_vertices(
    discretization: FiniteElementDiscretization,
    cells: np.ndarray,
    local: np.ndarray,
    /,
) -> list[tuple[int, ...]]:
    """Global mesh vertices of each local facet in reference-topology order."""
    blocks, block_cells = _cell_blocks(discretization, cells)
    vertices = [np.asarray(block.vertices) for block in discretization.mesh.blocks]
    facets = [
        reference_cell_topology(block.cell_kind).entities[block.topological_dimension - 1]
        for block in discretization.mesh.blocks
    ]
    return [
        tuple(int(vertex) for vertex in vertices[block][cell, list(facets[block][facet])])
        for block, cell, facet in zip(blocks, block_cells, local, strict=True)
    ]


def _side_parameters(
    discretization: FiniteElementDiscretization,
    selection: _SideSelection,
    rule: FacetTraceRule,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[FacetShape, ...]]:
    """Owner and traced-side facet parameters of the canonical sites.

    The canonical sites are the owner cell's local-facet rule points. On the
    neighbor side the same points are addressed through the affine
    (dihedral) map between the two local-facet parametrizations, fixed by the
    shared global facet vertices. Returns `(owner, side, rule_weights, shapes)`.
    """
    blocks, _ = _cell_blocks(discretization, selection.owner_cells)
    shapes = tuple(
        _reference_facet_shape(discretization.mesh.blocks[block].cell_kind, int(local))
        for block, local in zip(blocks, selection.owner_local, strict=True)
    )
    references = {shape: rule.reference(shape) for shape in set(shapes)}
    if len({value[0].shape for value in references.values()}) != 1:
        raise ValueError("The selected facets need one common rule site count.")
    owner = np.stack([references[shape][0] for shape in shapes])
    weights = np.stack([references[shape][1] for shape in shapes])
    if not selection.neighbor:
        return owner, owner, weights, shapes
    owner_vertices = _local_facet_vertices(
        discretization, selection.owner_cells, selection.owner_local
    )
    side_vertices = _local_facet_vertices(
        discretization, selection.side_cells, selection.side_local
    )
    mapped = np.empty_like(owner)
    for index, (shape, first, second) in enumerate(
        zip(shapes, owner_vertices, side_vertices, strict=True)
    ):
        if sorted(first) != sorted(second):
            raise ValueError(
                "Owner and neighbor local facets do not share their vertices; "
                "periodic or nonconforming facets need an explicit transfer."
            )
        reference_corners = _facet_corner_parameters(shape)
        corners = reference_corners[[second.index(v) for v in first]]
        if isinstance(shape, ReferenceCellTopology):
            dimension = reference_corners.shape[1]
            origin = int(np.flatnonzero(np.all(reference_corners == 0, axis=1))[0])
            axis_corners = [
                int(np.flatnonzero(np.all(reference_corners == axis, axis=1))[0])
                for axis in np.eye(dimension, dtype=np.float64)
            ]
            axes = corners[axis_corners] - corners[origin]
            mapped[index] = corners[origin] + owner[index] @ axes
        else:
            axes = corners[list(_FACET_AXIS_CORNERS[shape])] - corners[0]
            mapped[index] = corners[0] + owner[index] @ axes
    return owner, mapped, weights, shapes


def _facet_support_columns(element: FiniteElementSpec, local_facet: int, /) -> np.ndarray:
    """Local DOFs of one side cell whose basis has a nonzero trace on the facet.

    H1 fields use the facet-closure entity DOFs; every other local basis
    function must vanish on the facet (verified on a unisolvent facet probe).
    L2 fields keep the local DOFs whose probe tabulation is not identically
    zero.
    """
    if element.form_basis is not None:
        return np.arange(element.local_dof_count, dtype=np.int32)
    shape = _reference_facet_shape(element.cell_kind, local_facet)
    probe, _ = FacetTraceRule(points=element.degree + 2).reference(shape)
    points, _ = reference_facet_embedding(element.cell_kind, local_facet, probe)
    values = np.abs(np.asarray(element.tabulate(points)[0]))
    nonzero = np.max(values, axis=0) > 1.0e-10 * max(np.max(values), 1.0)
    if element.conformity != "H1":
        return np.flatnonzero(nonzero).astype(np.int32)
    topology = reference_cell_topology(element.cell_kind)
    facet = set(topology.entities[topology.dimension - 1][local_facet])
    closure = sorted(
        dof
        for dimension, entities in enumerate(topology.entities[: topology.dimension])
        for entity, vertices in enumerate(entities)
        if set(vertices) <= facet
        for dof in element.entity_dofs[dimension][entity]
    )
    columns = np.asarray(closure, dtype=np.int32)
    outside = np.ones(values.shape[1], dtype=np.bool_)
    outside[columns] = False
    if np.any(nonzero & outside):
        raise ValueError(
            "An H1 basis function outside the facet closure has a nonzero trace; "
            "the element is not trace-conforming."
        )
    return columns


def _facet_trace_degree(
    element: FiniteElementSpec,
    coordinate_degree: int,
    shape: FacetShape,
    corners: np.ndarray,
    /,
) -> int | None:
    """Polynomial degree of the trace along affinely mapped facets, else None.

    `corners` holds the physical facet corners `(facets, corners, dimension)`.
    Tensor (quadrilateral) facet traces have total degree twice the element
    degree; rational pyramid bases and curved coordinate maps give None.
    """
    if element.cell_kind == "pyramid" or coordinate_degree != 1:
        return None
    if isinstance(shape, ReferenceCellTopology):
        if shape.name.startswith("simplex:"):
            return element.degree
        reference = np.asarray(shape.vertices)
        augmented = np.c_[reference, np.ones((reference.shape[0],))]
        affine = np.linalg.lstsq(
            augmented,
            corners.transpose((1, 0, 2)).reshape((reference.shape[0], -1)),
            rcond=None,
        )[0]
        represented = (
            (augmented @ affine)
            .reshape((reference.shape[0], corners.shape[0], corners.shape[2]))
            .transpose((1, 0, 2))
        )
        if np.max(np.abs(represented - corners)) > 1e-12 * max(
            np.max(np.abs(corners)), 1
        ):
            return None
        return reference.shape[1] * element.degree
    match shape:
        case "point":
            return 0
        case "edge" | "triangle":
            return element.degree
        case "quadrilateral":
            # A bilinear facet map is affine exactly on parallelograms.
            defect = corners[:, 0] - corners[:, 1] + corners[:, 2] - corners[:, 3]
            scale = np.max(np.abs(corners - corners[:, :1]))
            if np.max(np.abs(defect)) > 1.0e-12 * scale:
                return None
            return 2 * element.degree
        case _:
            raise ValueError(f"Unsupported facet shape {shape!r}.")


def _side_geometry(
    discretization: FiniteElementDiscretization,
    coordinates: Array,
    block: int,
    block_cells: np.ndarray,
    local: np.ndarray,
    parameters: np.ndarray,
    /,
) -> _SideGeometry:
    """Embed facet parameters into the side cells and map them physically.

    The measure-scaled outward normal is the cofactor (Nanson) map
    `n ds = det(J) J^{-T} N_ref ds_ref` of the coordinate element Jacobian.
    """
    kind = discretization.mesh.blocks[block].cell_kind
    count, sites, _ = parameters.shape
    dimension = discretization.mesh.topological_dimension
    reference = np.empty((count, sites, dimension))
    scaled = np.empty((count, sites, dimension))
    for facet in np.unique(local).tolist():
        rows = np.flatnonzero(local == facet)
        points, normals = reference_facet_embedding(
            kind, facet, parameters[rows].reshape((rows.size * sites, dimension - 1))
        )
        reference[rows] = np.asarray(points).reshape((rows.size, sites, dimension))
        scaled[rows] = np.asarray(normals).reshape((rows.size, sites, dimension))
    cell_map = PreparedFiniteElementCellMap(discretization, block)
    evaluation = cell_map.evaluate(
        coordinates,
        jnp.asarray(np.repeat(block_cells, sites)),
        jnp.asarray(reference.reshape((-1, dimension))),
    )
    if not bool(np.all(np.asarray(evaluation.valid))):
        raise ValueError("A side cell has an invalid coordinate map at its facet.")
    physical = contract(
        "p,pri,pr->pi",
        jnp.abs(evaluation.determinant),
        evaluation.inverse_jacobian,
        jnp.asarray(scaled.reshape((-1, dimension))),
    )
    return _SideGeometry(
        reference,
        np.asarray(evaluation.physical_points).reshape((count, sites, dimension)),
        np.asarray(physical).reshape((count, sites, dimension)),
        np.asarray(evaluation.inverse_jacobian).reshape(
            (count, sites, dimension, dimension)
        ),
        cell_map.coordinate_element.degree,
    )


def _block_side_gathers(
    discretization: FiniteElementDiscretization,
    field_index: int,
    block: int,
    block_cells: np.ndarray,
    local: np.ndarray,
    reference: np.ndarray,
    coordinates: Array,
    /,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-facet global DOF gathers and oriented basis values on the support."""
    element = discretization.elements[field_index][block]
    dof_map = discretization.dof_maps[field_index]
    count, sites, dimension = reference.shape
    points = jnp.asarray(reference.reshape((-1, dimension)))
    cell_map = PreparedFiniteElementCellMap(discretization, block)
    paired_cells = jnp.repeat(jnp.asarray(block_cells), sites)
    evaluation = cell_map.evaluate(coordinates, paired_cells, points)
    tabulated = finite_element_point_weights(
        element,
        jnp.asarray(dof_map.orientations[block])[paired_cells],
        points,
        evaluation.inverse_jacobian,
        None,
        jacobian=evaluation.jacobian,
        transform=dof_map.cell_transforms[block][paired_cells],
    )
    basis = np.asarray(tabulated).reshape(
        (count, sites, element.local_dof_count, *element.value_shape)
    )
    routes = np.asarray(dof_map.cell_dofs[block])[block_cells]
    columns = {
        facet: _facet_support_columns(element, facet)
        for facet in np.unique(local).tolist()
    }
    gathers = [routes[row, columns[int(facet)]] for row, facet in enumerate(local)]
    values = [basis[row][:, columns[int(facet)]] for row, facet in enumerate(local)]
    return gathers, values


def _block_trace_degree(
    discretization: FiniteElementDiscretization,
    field_index: int,
    coordinates: np.ndarray,
    coordinate_degree: int,
    block: int,
    block_cells: np.ndarray,
    local: np.ndarray,
    /,
) -> int | None:
    """Largest trace degree of one block's selected facets, None if not polynomial."""
    element = discretization.elements[field_index][block]
    topology = reference_cell_topology(element.cell_kind)
    geometry_dofs = np.asarray(discretization.coordinate_dofs[block])[block_cells]
    degrees = []
    for facet in np.unique(local).tolist():
        vertices = list(topology.entities[topology.dimension - 1][facet])
        corners = coordinates[geometry_dofs[local == facet][:, vertices]]
        shape = _reference_facet_shape(element.cell_kind, facet)
        degrees.append(_facet_trace_degree(element, coordinate_degree, shape, corners))
    return _combined_degree(degrees)


def _combined_degree(degrees: list[int | None], /) -> int | None:
    """Largest polynomial degree, or None when any part is not polynomial."""
    known = [degree for degree in degrees if degree is not None]
    return None if len(known) != len(degrees) else max(known)


def _side_tabulation(
    discretization: FiniteElementDiscretization,
    field_index: int,
    coordinates: Array,
    cells: np.ndarray,
    local: np.ndarray,
    parameters: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> _SideTabulation:
    """Group facets by side-cell block and pad local widths with zero weights."""
    blocks, block_cells = _cell_blocks(discretization, cells)
    count, sites = rule_weights.shape
    dimension = discretization.mesh.ambient_dimension
    gathers: list[np.ndarray] = [np.empty((0,), dtype=np.int32)] * count
    values: list[np.ndarray] = [np.empty((sites, 0))] * count
    physical_sites = np.empty((count, sites, dimension))
    scaled = np.empty((count, sites, dimension))
    degrees: list[int | None] = []
    for block in np.unique(blocks).tolist():
        rows = np.flatnonzero(blocks == block)
        geometry = _side_geometry(
            discretization,
            coordinates,
            block,
            block_cells[rows],
            local[rows],
            parameters[rows],
        )
        block_gathers, block_values = _block_side_gathers(
            discretization,
            field_index,
            block,
            block_cells[rows],
            local[rows],
            geometry.reference,
            coordinates,
        )
        for row, gather, value in zip(rows, block_gathers, block_values, strict=True):
            gathers[row] = gather
            values[row] = value
        physical_sites[rows] = geometry.sites
        scaled[rows] = geometry.scaled_normals
        degrees.append(
            _block_trace_degree(
                discretization,
                field_index,
                np.asarray(coordinates),
                geometry.coordinate_degree,
                block,
                block_cells[rows],
                local[rows],
            )
        )
    width = max(gather.size for gather in gathers)
    dofs = np.empty((count, width), dtype=np.int32)
    valid = np.zeros((count, width), dtype=np.bool_)
    value_shape = discretization.elements[field_index][0].value_shape
    basis = np.zeros((count, sites, width, *value_shape), dtype=np.float64)
    for row, (gather, value) in enumerate(zip(gathers, values, strict=True)):
        # Padded slots repeat a real row of the facet and carry zero weight.
        dofs[row] = gather[0]
        dofs[row, : gather.size] = gather
        valid[row, : gather.size] = True
        basis[row, :, : gather.size] = value
    measure = np.linalg.norm(scaled, axis=-1)
    return _SideTabulation(
        dofs,
        valid,
        basis,
        physical_sites,
        rule_weights * measure,
        scaled / measure[..., None],
        _combined_degree(degrees),
    )


def _trace_route_weights(
    basis: np.ndarray, normals: np.ndarray, quantity: SideTraceQuantity, /
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Contracted route weights `(facets, sites, local, *value, *component)`."""
    dimension = normals.shape[-1]
    if basis.ndim == 4:
        match quantity:
            case "normal":
                return np.sum(basis * normals[:, :, None, :], axis=-1), ()
            case "tangential" if dimension == 2:
                tangent = np.stack((-normals[..., 1], normals[..., 0]), axis=-1)
                return np.sum(basis * tangent[:, :, None, :], axis=-1), ()
            case "tangential":
                normal_part = np.sum(basis * normals[:, :, None, :], axis=-1)
                return basis - normal_part[..., None] * normals[:, :, None, :], (
                    dimension,
                )
            case "value" | "conormal-flux":
                raise ValueError(f"{quantity!r} traces are not contracted routes.")
            case _:
                assert_never(quantity)
    match quantity:
        case "normal":
            return basis[..., None] * normals[:, :, None, :], ()
        case "tangential" if dimension == 2:
            tangent = np.stack((-normals[..., 1], normals[..., 0]), axis=-1)
            return basis[..., None] * tangent[:, :, None, :], ()
        case "tangential":
            projector = np.eye(dimension) - normals[..., :, None] * normals[..., None, :]
            return basis[..., None, None] * projector[:, :, None], (dimension,)
        case "value" | "conormal-flux":
            raise ValueError(f"{quantity!r} traces are not contracted routes.")
        case _:
            assert_never(quantity)


def _trace_route(
    tabulation: _SideTabulation,
    space: ArraySpace,
    quantity: SideTraceQuantity,
    /,
) -> SideGatherRoute:
    """Componentwise value route or normal-contracted vector route."""
    if tabulation.basis.ndim > 3:
        if quantity == "value":
            weights, value_shape = tabulation.basis, tabulation.basis.shape[3:]
        else:
            weights, value_shape = _trace_route_weights(
                tabulation.basis, tabulation.normals, quantity
            )
        return SideGatherRoute(
            tabulation.dofs,
            weights.astype(space.dtype),
            coefficient_shape=space.shape,
            mode="contracted",
            value_shape=value_shape,
        )
    if quantity == "value":
        return SideGatherRoute(
            tabulation.dofs,
            tabulation.basis.astype(space.dtype),
            coefficient_shape=space.shape,
            value_shape=space.shape[1:],
        )
    weights, value_shape = _trace_route_weights(
        tabulation.basis, tabulation.normals, quantity
    )
    return SideGatherRoute(
        tabulation.dofs,
        weights.astype(space.dtype),
        coefficient_shape=space.shape,
        mode="contracted",
        value_shape=value_shape,
    )


def _owner_site_agreement(
    discretization: FiniteElementDiscretization,
    field_index: int,
    coordinates: Array,
    selection: _SideSelection,
    owner_parameters: np.ndarray,
    rule_weights: np.ndarray,
    side: _SideTabulation,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Owner sites and measure, verified against the neighbor embedding."""
    owner = _side_tabulation(
        discretization,
        field_index,
        coordinates,
        selection.owner_cells,
        selection.owner_local,
        owner_parameters,
        rule_weights,
    )
    scale = max(float(np.max(np.ptp(np.asarray(coordinates), axis=0))), 1.0)
    if np.max(np.abs(owner.sites - side.sites)) > 1.0e-10 * scale:
        raise ValueError(
            "Owner and neighbor embeddings of the facet sites disagree; the mesh "
            "geometry is not conforming across the selected facets."
        )
    return owner.sites, owner.weights


def prepare_finite_element_side_trace(
    discretization: FiniteElementDiscretization,
    field_name: str,
    domain: IntegrationDomain,
    /,
    *,
    rule: FacetTraceRule,
    quantity: SideTraceQuantity = "value",
    side: FieldTraceSide = "owner",
    runtime: FiniteElementRuntimeData | None = None,
) -> PreparedTraceAction:
    """Prepare the exact trace of one FE field on selected exterior/interior facets.

    Supports scalar and compatible polynomial form fields of every degree on
    simplex and tensor-product blocks. Piola values use the actual side runtime.
    Sites are the owner cell's local-facet rule points, shared by the
    `"owner"` and `"neighbor"` sides of an interior facet; normals point out of
    the traced side cell and weights are the physical facet measure. `"value"`
    traces act componentwise; `"normal"` (`u . n`) and `"tangential"`
    (`u . tau` with `tau = (-n_y, n_x)` in 2-D, `u - (u . n) n` in 3-D) traces
    of vector fields contract the components against the outward normal. The
    route gathers each facet's side-cell DOFs (facet-closure DOFs for H1); no
    global coefficient-by-site matrix is formed.
    """
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    if not isinstance(rule, FacetTraceRule):
        raise TypeError("rule must be a FacetTraceRule.")
    quantity = parse(quantity, SideTraceQuantity, "quantity")
    side = parse(side, FieldTraceSide, "side")
    field_index, space = _side_trace_field(discretization, field_name, quantity)
    selection = _side_trace_selection(discretization, domain, side)
    realized = _realized_runtime(discretization, runtime)
    owner_parameters, parameters, rule_weights, shapes = _side_parameters(
        discretization, selection, rule
    )
    tabulation = _side_tabulation(
        discretization,
        field_index,
        realized.coordinates,
        selection.side_cells,
        selection.side_local,
        parameters,
        rule_weights,
    )
    sites, weights = tabulation.sites, tabulation.weights
    if selection.neighbor:
        sites, weights = _owner_site_agreement(
            discretization,
            field_index,
            realized.coordinates,
            selection,
            owner_parameters,
            rule_weights,
            tabulation,
        )
    exact_degrees = [rule.exact_degree(shape) for shape in set(shapes)]
    known_exact = [degree for degree in exact_degrees if degree is not None]
    descriptor = SideActionDescriptor(
        owner_id=discretization.prepared_id,
        field_space_id=discretization.field_spaces[field_index].field_space_id,
        quantity=quantity,
        representation="quadrature-values",
        orientation="unoriented" if quantity == "value" else "outward",
        approximation="exact",
        side=side,
        domain=domain,
        revision_id=finite_element_side_revision(realized),
        rule=rule,
        trace_degree=tabulation.trace_degree,
        quadrature_exact_degree=(
            min(known_exact) if len(known_exact) == len(exact_degrees) else None
        ),
    )
    element = discretization.elements[field_index][0]
    form = None
    if element.form_basis is not None:
        dimension = discretization.mesh.ambient_dimension
        twist = element.value_spec.form_type.twist
        if quantity == "normal" or (quantity == "tangential" and dimension == 2):
            form = FormValueSpec(
                FormType(dimension - 1, dimension - 1, twist=twist), proxy="density"
            )
        elif quantity == "tangential":
            form = FormValueSpec(
                FormType(dimension - 1, 1, twist=twist, ambient_dimension=dimension),
                proxy="circulation",
            )
        else:
            form = element.value_spec
    return PreparedTraceAction(
        descriptor,
        _trace_route(tabulation, space, quantity),
        space,
        sites=sites,
        weights=weights,
        normals=tabulation.normals,
        support_rows=np.unique(tabulation.dofs[tabulation.valid]),
        form=form,
    )


def finite_element_side_revision(runtime: FiniteElementRuntimeData, /) -> str:
    """Geometry revision of FE side actions prepared on one runtime realization."""
    return canonical_fingerprint(
        {
            "kind": "finite-element-side-geometry",
            "runtime": runtime.runtime_id,
            "coordinates": array_tree_fingerprint(np.asarray(runtime.coordinates)),
        }
    )


@final
class FiniteElementSideGradient(StrictModule, NonTrainableState):
    """Physical gradient of one scalar finite-element field at facet sites.

    `route` gathers every local DOF of each facet's side cell (a gradient on a
    facet depends on the whole cell, not only on its facet closure) and
    contracts it with the oriented physical basis gradients at the sites into
    `(facets, sites, dimension)` values; `route.transpose` is its exact
    scatter-add. `reference` holds the sites in the side cells' reference
    coordinates, `side_blocks`/`side_block_cells` locate the side cells in the
    mesh blocks, and `valid` marks the real local slots of padded cells.
    `normals` point out of the side cells and `weights` are the physical facet
    measure of the rule. `gradient_degree` bounds the polynomial degree of the
    gradient along every facet; it is None when a side-cell map is not affine
    along a facet or the element trace is not polynomial. `exact_degree` is the
    polynomial degree the facet rule integrates exactly on every selected facet
    shape (None on point facets, where the rule is a point evaluation).
    """

    route: SideGatherRoute
    sites: Array
    weights: Array
    normals: Array
    reference: Array
    side_cells: Array
    side_blocks: Array
    side_block_cells: Array
    valid: Array
    field_name: str = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    gradient_degree: int | None = eqx.field(static=True)
    exact_degree: int | None = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class _GradientRows:
    """Per-facet whole-cell gathers and oriented physical basis gradients."""

    gathers: list[np.ndarray]
    gradients: list[np.ndarray]
    reference: np.ndarray
    sites: np.ndarray
    scaled_normals: np.ndarray
    gradient_degree: int | None


def _gradient_degree(
    element: FiniteElementSpec, value_degree: int | None, inverse_jacobian: np.ndarray, /
) -> int | None:
    """Degree of the gradient along facets whose side-cell map is affine there.

    A simplex gradient loses one degree; a tensor-cell normal derivative keeps
    the tangential degree of the trace. A Jacobian that varies along a facet
    makes the physical gradient rational.
    """
    if value_degree is None:
        return None
    spread = float(np.max(np.abs(inverse_jacobian - inverse_jacobian[:, :1])))
    if spread > 1.0e-10 * max(float(np.max(np.abs(inverse_jacobian))), 1.0e-300):
        return None
    if element.cell_kind in (*_SIMPLICES, "interval"):
        return max(value_degree - 1, 0)
    return value_degree


def _side_gradient_rows(
    discretization: FiniteElementDiscretization,
    field_index: int,
    coordinates: Array,
    cells: np.ndarray,
    local: np.ndarray,
    parameters: np.ndarray,
    /,
) -> _GradientRows:
    """Tabulate every side cell's physical basis gradients block by block."""
    blocks, block_cells = _cell_blocks(discretization, cells)
    count, sites = parameters.shape[:2]
    dimension = discretization.mesh.ambient_dimension
    gathers: list[np.ndarray] = [np.empty((0,), dtype=np.int32)] * count
    gradients: list[np.ndarray] = [np.empty((sites, 0, dimension))] * count
    reference = np.empty((count, sites, dimension))
    physical_sites = np.empty((count, sites, dimension))
    scaled = np.empty((count, sites, dimension))
    degrees: list[int | None] = []
    dof_map = discretization.dof_maps[field_index]
    for block in np.unique(blocks).tolist():
        rows = np.flatnonzero(blocks == block)
        geometry = _side_geometry(
            discretization,
            coordinates,
            block,
            block_cells[rows],
            local[rows],
            parameters[rows],
        )
        element = discretization.elements[field_index][block]
        tabulated = element.tabulate(
            jnp.asarray(geometry.reference.reshape((-1, dimension)))
        )[1]
        physical = np.asarray(
            contract(
                "plr,pri->pli",
                tabulated,
                jnp.asarray(
                    geometry.inverse_jacobian.reshape((-1, dimension, dimension))
                ),
            )
        ).reshape((rows.size, sites, -1, dimension))
        orientation = np.asarray(dof_map.orientations[block])[block_cells[rows]]
        physical = physical * orientation[:, None, :, None]
        routes = np.asarray(dof_map.cell_dofs[block])[block_cells[rows]]
        for position, row in enumerate(rows):
            gathers[row] = routes[position]
            gradients[row] = physical[position]
        reference[rows] = geometry.reference
        physical_sites[rows] = geometry.sites
        scaled[rows] = geometry.scaled_normals
        value_degree = _block_trace_degree(
            discretization,
            field_index,
            np.asarray(coordinates),
            geometry.coordinate_degree,
            block,
            block_cells[rows],
            local[rows],
        )
        degrees.append(_gradient_degree(element, value_degree, geometry.inverse_jacobian))
    return _GradientRows(
        gathers, gradients, reference, physical_sites, scaled, _combined_degree(degrees)
    )


def prepare_finite_element_side_gradient(
    discretization: FiniteElementDiscretization,
    field_name: str,
    domain: IntegrationDomain,
    /,
    *,
    rule: FacetTraceRule,
    side: FieldTraceSide = "owner",
    runtime: FiniteElementRuntimeData | None = None,
) -> FiniteElementSideGradient:
    """Prepare the physical gradient of one scalar FE field on selected facets.

    The sites are the side cell's local-facet rule points (the owner's points
    mapped into the neighbor on interior facets, as for side traces). Only
    identity-mapped scalar-basis H1/L2 fields are supported.
    """
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    if not isinstance(rule, FacetTraceRule):
        raise TypeError("rule must be a FacetTraceRule.")
    side = parse(side, FieldTraceSide, "side")
    field_index, space = _side_trace_field(discretization, field_name, "value")
    if space.shape[1:]:
        raise ValueError("Side gradients are prepared for scalar finite-element fields.")
    selection = _side_trace_selection(discretization, domain, side)
    realized = _realized_runtime(discretization, runtime)
    _, parameters, rule_weights, shapes = _side_parameters(
        discretization, selection, rule
    )
    exact = [rule.exact_degree(shape) for shape in set(shapes)]
    known = [degree for degree in exact if degree is not None]
    rows = _side_gradient_rows(
        discretization,
        field_index,
        realized.coordinates,
        selection.side_cells,
        selection.side_local,
        parameters,
    )
    count, sites = rule_weights.shape
    dimension = discretization.mesh.ambient_dimension
    width = max(gather.size for gather in rows.gathers)
    dofs = np.empty((count, width), dtype=np.int32)
    valid = np.zeros((count, width), dtype=np.bool_)
    weights = np.zeros((count, sites, width, dimension))
    for row, (gather, gradient) in enumerate(
        zip(rows.gathers, rows.gradients, strict=True)
    ):
        # Padded slots repeat a real row of the cell and carry zero weight.
        dofs[row] = gather[0]
        dofs[row, : gather.size] = gather
        valid[row, : gather.size] = True
        weights[row, :, : gather.size] = gradient
    measure = np.linalg.norm(rows.scaled_normals, axis=-1)
    blocks, block_cells = _cell_blocks(discretization, selection.side_cells)
    return FiniteElementSideGradient(
        route=SideGatherRoute(
            dofs,
            weights.astype(space.dtype),
            coefficient_shape=space.shape,
            mode="contracted",
            value_shape=(dimension,),
        ),
        sites=jnp.asarray(rows.sites.astype(space.dtype)),
        weights=jnp.asarray((rule_weights * measure).astype(space.dtype)),
        normals=jnp.asarray(
            (rows.scaled_normals / measure[..., None]).astype(space.dtype)
        ),
        reference=jnp.asarray(rows.reference.astype(space.dtype)),
        side_cells=jnp.asarray(selection.side_cells.astype(np.int32)),
        side_blocks=jnp.asarray(blocks),
        side_block_cells=jnp.asarray(block_cells),
        valid=jnp.asarray(valid),
        field_name=discretization.field_spaces[field_index].name,
        rule_id=rule.rule_id,
        revision_id=finite_element_side_revision(realized),
        runtime_id=realized.runtime_id,
        gradient_degree=rows.gradient_degree,
        exact_degree=min(known) if known else None,
    )


__all__ = [
    "FiniteElementFieldReconstructionKernel",
    "FiniteElementSideGradient",
    "PreparedFiniteElementPointInterpolation",
    "finite_element_point_weights",
    "finite_element_side_revision",
    "prepare_finite_element_field_reconstruction",
    "prepare_finite_element_point_interpolation",
    "prepare_finite_element_side_gradient",
    "prepare_finite_element_side_trace",
]
