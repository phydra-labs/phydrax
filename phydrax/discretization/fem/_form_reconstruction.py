# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...exterior._algebra import map_reference_values
from ...exterior._form_type import FormType, FormValueSpec
from ...typing import Dim, Float, Int32
from .._cell_geometry_validity import _bernstein_plan
from .._simplicial_locator import (
    AbstractCellLocator,
    CellLocationResult,
    CellLocationStatus,
    SimplicialLocationPolicy,
)
from .._view_support import cell_location_status
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTraceSide,
)
from ._cell_map import PreparedFiniteElementCellMap
from ._reference import FiniteElementSpec


class FormCellDim(Dim):
    """Prepared finite-element cells."""


class FormLocalDim(Dim):
    """Local polynomial moment coordinates."""


class FormQueryDim(Dim):
    """Physical field-query sites."""


class FormCandidateDim(Dim):
    """Side-resolving candidate cells."""


class FormComponentDim(Dim):
    """Physical proxy components at a query."""


class FormVertexDim(Dim):
    """Coordinate-map vertices."""


class FormAmbientDim(Dim):
    """Physical ambient coordinate axes."""


@final
class _TensorCellLocator(AbstractCellLocator, NonTrainableState):
    """Bounded native BVH and damped-Newton tensor coordinate queries."""

    __strict_contract__ = True
    cell_map: PreparedFiniteElementCellMap
    coordinates: Float[FormVertexDim, FormAmbientDim]
    bvh: PackedBVH
    policy: SimplicialLocationPolicy
    locator_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_map: PreparedFiniteElementCellMap,
        coordinates: Array,
        policy: SimplicialLocationPolicy,
        /,
    ) -> None:
        kind = cell_map.coordinate_element.cell_kind
        if kind not in ("quadrilateral", "hexahedron") and not kind.startswith("tensor:"):
            raise ValueError(
                "Tensor location requires a tensor-product coordinate chart."
            )
        if coordinates.shape != (cell_map.coordinate_count, cell_map.ambient_dimension):
            raise ValueError(
                "Tensor locator coordinates do not match the prepared chart."
            )
        if not isinstance(policy, SimplicialLocationPolicy):
            raise TypeError("Tensor queries require the owning bounded location policy.")
        plan = _bernstein_plan(
            "box", (cell_map.coordinate_element.degree,) * cell_map.reference_dimension
        )
        basis = np.asarray(
            cell_map.coordinate_element.tabulate(plan.nodes)[0], dtype=np.float64
        )
        cell_coordinates = np.asarray(coordinates, dtype=np.float64)[
            np.asarray(cell_map.coordinate_dofs)
        ]
        control = contract(
            "rm,mn,cna->cra", plan.coefficients_from_values, basis, cell_coordinates
        )
        rounding = (
            2
            * plan.conversion_norm
            * np.max(np.sum(np.abs(basis), axis=1))
            * sum(basis.shape)
            * np.finfo(np.dtype(coordinates.dtype)).eps
            * np.max(np.abs(cell_coordinates), axis=(1, 2))
        )
        bvh = prepare_bvh(
            np.min(control, axis=1) - rounding[:, None],
            np.max(control, axis=1) + rounding[:, None],
            policy=BVHBuildPolicy(leaf_size=min(16, cell_map.cell_count)),
            dtype=coordinates.dtype,
        )
        self.cell_map, self.coordinates, self.bvh, self.policy = (
            cell_map,
            coordinates,
            bvh,
            policy,
        )
        self.locator_id = canonical_fingerprint(
            {
                "kind": "tensor-fe-locator",
                "cell_map": cell_map.cell_map_id,
                "coordinates": array_tree_fingerprint(coordinates),
                "policy": policy.policy_id,
            }
        )

    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> CellLocationResult:
        sites = jnp.asarray(points)
        if sites.ndim != 2 or sites.shape[1] != self.cell_map.ambient_dimension:
            raise ValueError("Tensor query sites must have shape (points,ambient).")
        capacity = min(self.policy.maximum_candidates, self.cell_map.cell_count)
        candidates, valid, complete = point_select_leaf_items(
            sites,
            bvh=self.bvh,
            maximum_candidates=capacity,
            tolerance=self.policy.reference_tolerance,
        )
        safe = jnp.where(valid, candidates, 0)
        if cell_mask is not None:
            mask = jnp.asarray(cell_mask)
            if mask.shape != (self.cell_map.cell_count,) or mask.dtype != jnp.bool_:
                raise ValueError(
                    "Tensor side masks must be Boolean and match cell count."
                )
            valid &= mask[safe]
        reference, iterations, geometry_valid, residual, condition = (
            self._invert_candidates(sites, safe, valid)
        )
        return self._location_result(
            sites,
            candidates,
            valid,
            complete,
            reference,
            iterations,
            geometry_valid,
            residual,
            condition,
        )

    def _invert_candidates(
        self, sites: Array, safe: Array, valid: Array, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        count, capacity = safe.shape
        dimension = self.cell_map.reference_dimension
        pool = jnp.concatenate(
            (
                jnp.full((1, dimension), 0.5, dtype=sites.dtype),
                self.cell_map.coordinate_element.reference_nodes,
            ),
            axis=0,
        )
        seeds = min(self.policy.maximum_seeds, pool.shape[0])
        reference = jnp.broadcast_to(
            pool[:seeds], (count, capacity, seeds, dimension)
        ).reshape((-1, dimension))
        ids = jnp.repeat(safe.reshape((-1,)), seeds)
        physical = jnp.repeat(sites, capacity * seeds, axis=0)
        lane_valid = jnp.repeat(valid.reshape((-1,)), seeds)
        initial_done = ~lane_valid
        initial_iterations = jnp.full(
            lane_valid.shape, self.policy.maximum_iterations, dtype=jnp.int32
        )

        def step(
            index: int, state: tuple[Array, Array, Array], /
        ) -> tuple[Array, Array, Array]:
            reference_, done, iterations = state
            evaluation = self.cell_map.evaluate(self.coordinates, ids, reference_)
            residual = evaluation.physical_points - physical
            converged = (
                evaluation.valid
                & lane_valid
                & (jnp.linalg.norm(residual, axis=-1) <= self.policy.residual_tolerance)
            )
            iterations = jnp.where(~done & converged, index, iterations)
            done |= converged
            update = contract("prd,pd->pr", evaluation.inverse_jacobian, residual)
            scale = jnp.minimum(
                1,
                self.policy.trust_radius
                / jnp.maximum(
                    jnp.linalg.norm(update, axis=-1), jnp.finfo(sites.dtype).tiny
                ),
            )
            reference_ = jnp.where(
                done[:, None], reference_, reference_ - update * scale[:, None]
            )
            return reference_, done, iterations

        def bounded_step(
            index: int, state: tuple[Array, Array, Array], /
        ) -> tuple[Array, Array, Array]:
            return jax.lax.cond(
                jnp.all(state[1]),
                lambda carry: carry,
                lambda carry: step(index, carry),
                state,
            )

        reference, _, iterations = jax.lax.fori_loop(
            0,
            self.policy.maximum_iterations,
            bounded_step,
            (reference, initial_done, initial_iterations),
        )
        evaluation = self.cell_map.evaluate(self.coordinates, ids, reference)
        residual = jnp.linalg.norm(
            evaluation.physical_points - physical, axis=-1
        ).reshape((count, capacity, seeds))
        condition = (
            jnp.linalg.norm(evaluation.jacobian, axis=(-2, -1))
            * jnp.linalg.norm(evaluation.inverse_jacobian, axis=(-2, -1))
        ).reshape((count, capacity, seeds))
        reference = reference.reshape((count, capacity, seeds, dimension))
        return reference, iterations, evaluation.valid, residual, condition

    def _location_result(
        self,
        sites: Array,
        candidates: Array,
        valid: Array,
        complete: Array,
        reference: Array,
        iterations: Array,
        geometry_valid: Array,
        residual: Array,
        condition: Array,
        /,
    ) -> CellLocationResult:
        count, capacity, seeds = residual.shape
        converged = residual <= self.policy.residual_tolerance
        accepted = (
            converged
            & geometry_valid.reshape((count, capacity, seeds))
            & valid[..., None]
        )
        accepted &= jnp.all(
            (reference >= -self.policy.reference_tolerance)
            & (reference <= 1 + self.policy.reference_tolerance),
            axis=-1,
        )
        seed = jnp.argmax(accepted, axis=-1)
        candidate_valid = jnp.any(accepted, axis=-1)
        candidate_reference = jnp.take_along_axis(
            reference, seed[..., None, None], axis=2
        ).squeeze(axis=2)
        candidate_residual = jnp.take_along_axis(
            residual, seed[..., None], axis=2
        ).squeeze(axis=2)
        candidate_condition = jnp.take_along_axis(
            condition, seed[..., None], axis=2
        ).squeeze(axis=2)
        candidate_iterations = jnp.take_along_axis(
            iterations.reshape((count, capacity, seeds)), seed[..., None], axis=2
        ).squeeze(axis=2)
        number = jnp.sum(candidate_valid, axis=1, dtype=jnp.int32)
        first = jnp.argmin(
            jnp.where(candidate_valid, candidates, self.cell_map.cell_count), axis=1
        )
        located = number > 0
        selected = jnp.where(located, candidates[jnp.arange(count), first], -1).astype(
            jnp.int32
        )
        selected_reference = candidate_reference[jnp.arange(count), first]
        status = jnp.where(
            located, int(CellLocationStatus.LOCATED), int(CellLocationStatus.OUTSIDE)
        )
        exhausted = jnp.any(valid[..., None] & ~converged, axis=(1, 2))
        status = jnp.where(
            ~located & exhausted, int(CellLocationStatus.INVERSE_MAP_EXHAUSTED), status
        )
        status = jnp.where(complete, status, int(CellLocationStatus.RESOURCE_EXCEEDED))
        status = jnp.where(
            jnp.all(jnp.isfinite(sites), axis=-1),
            status,
            int(CellLocationStatus.NONFINITE),
        ).astype(jnp.int32)
        successful = status == int(CellLocationStatus.LOCATED)
        return CellLocationResult(
            selected,
            selected_reference,
            self.cell_map.coordinate_element.tabulate(selected_reference)[0],
            candidate_residual[jnp.arange(count), first],
            candidate_iterations[jnp.arange(count), first],
            candidate_condition[jnp.arange(count), first],
            located,
            seed[jnp.arange(count), first] > 0,
            number,
            status,
            successful,
            jnp.where(candidate_valid, candidates, -1),
            candidate_reference,
            locator_id=self.locator_id,
        )


@final
class _FormRoute(StrictModule):
    __strict_contract__ = True
    dofs: Int32[FormQueryDim, FormCandidateDim, FormLocalDim]
    weights: Float[FormQueryDim, FormCandidateDim, FormLocalDim, FormComponentDim]


@final
class FormFieldReconstructionKernel(AbstractFieldReconstructionKernel, NonTrainableState):
    """Cell-sided compatible reconstruction, including coordinate-map derivatives."""

    __strict_contract__ = True
    locator: AbstractCellLocator
    element: FiniteElementSpec
    value_spec: FormValueSpec = eqx.field(static=True)
    cell_dofs: Int32[FormCellDim, FormLocalDim]
    cell_transforms: Float[FormCellDim, FormLocalDim, FormLocalDim]
    global_dof_count: int = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        locator: AbstractCellLocator,
        element: FiniteElementSpec,
        cell_dofs: ArrayLike,
        cell_transforms: ArrayLike,
        /,
        *,
        global_dof_count: int,
        field_space_id: str,
    ) -> None:
        if not isinstance(locator, AbstractCellLocator) or not isinstance(
            element, FiniteElementSpec
        ):
            raise TypeError(
                "Form reconstruction requires a cell locator and form element."
            )
        if element.form_basis is None:
            raise ValueError(
                "Form reconstruction requires canonical polynomial form metadata."
            )
        routes, transforms = jnp.asarray(cell_dofs), jnp.asarray(cell_transforms)
        expected = (locator.cell_map.cell_count, element.local_dof_count)
        if routes.shape != expected or transforms.shape != (*expected, expected[1]):
            raise ValueError("Form cell routes/transforms have incompatible shapes.")
        if global_dof_count < 1 or not field_space_id:
            raise ValueError("Form reconstruction requires nonempty scientific identity.")
        form = element.value_spec.form_type
        physical_type = FormType(
            form.dimension,
            form.degree,
            twist=form.twist,
            fiber_shape=form.fiber_shape,
            ambient_dimension=locator.cell_map.ambient_dimension,
        )
        value_spec = FormValueSpec(physical_type, proxy=element.value_spec.proxy)
        self.locator, self.element = locator, element
        self.value_spec = value_spec
        self.cell_dofs, self.cell_transforms = routes, transforms
        self.global_dof_count = global_dof_count
        self._kernel_id = canonical_fingerprint(
            {
                "kind": "form-field-reconstruction",
                "locator": locator.locator_id,
                "element": element.element_id,
                "field_space": field_space_id,
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

    def _basis(self, cell: Array, reference: Array, /) -> Array:
        evaluation = self.locator.cell_map.evaluate(
            self.locator.coordinates, cell[None], reference[None]
        )
        values = self.element.tabulate(reference[None])[0]
        mapped = map_reference_values(
            values, self.element.value_spec, evaluation.jacobian[:, None]
        )[0]
        return contract(
            "iv,ij->jv",
            mapped.reshape((self.element.local_dof_count, -1)),
            self.cell_transforms[cell],
        )

    def locate(
        self, points: Array, derivative: tuple[int, ...], side: FieldSideBinding | None, /
    ) -> tuple[_FormRoute, FieldQueryEvidence]:
        order = sum(derivative)
        if order > 1:
            raise ValueError(
                "Prepared form reconstruction supports values and first derivatives."
            )
        location = self.locator.locate(
            points, cell_mask=None if side is None else side.cell_mask
        )
        candidates = location.candidate_cells
        accepted = candidates >= 0
        safe = jnp.where(accepted, candidates, 0)
        count = jnp.sum(accepted, axis=1, dtype=jnp.int32)
        cells = safe.reshape((-1,))
        reference = location.candidate_reference.reshape(
            (-1, self.locator.cell_map.reference_dimension)
        )
        if order == 0:
            weights = jax.vmap(self._basis)(cells, reference)
        else:
            axis = derivative.index(1)
            reference_gradient = jax.vmap(jax.jacfwd(self._basis, argnums=1))(
                cells, reference
            )
            inverse = self.locator.cell_map.evaluate(
                self.locator.coordinates, cells, reference
            ).inverse_jacobian
            weights = contract("plvr,pr->plv", reference_gradient, inverse[:, :, axis])
        weights = weights.reshape((*candidates.shape, self.element.local_dof_count, -1))
        if side is not None and side.side == "average":
            share = accepted / jnp.maximum(count, 1)[:, None]
        else:
            share = (
                jnp.arange(candidates.shape[1])[None]
                == jnp.argmax(accepted, axis=1)[:, None]
            ) & accepted
        route = _FormRoute(self.cell_dofs[safe], weights * share[:, :, None, None])
        status = cell_location_status(location.status)
        continuity = (
            0
            if self.element.value_spec.form_type.degree == 0
            and self.element.continuity == "conforming"
            else -1
        )
        ambiguous = (count > 1) & (order > continuity)
        if side is None or side.side != "average":
            status = jnp.where(
                (status == int(FieldQueryStatus.VALID)) & ambiguous,
                int(
                    FieldQueryStatus.SIDE_REQUIRED
                    if side is None
                    else FieldQueryStatus.SIDE_UNRESOLVED
                ),
                status,
            )
        return route, FieldQueryEvidence(
            status, location.jacobian_condition, count, kernel_id=self.kernel_id
        )

    def apply(self, route: _FormRoute, coefficients: Array, /) -> Array:
        values = contract("pclv,pcl->pv", route.weights, coefficients[route.dofs])
        return values.reshape((values.shape[0], *self.value_spec.value_shape))

    def transpose(self, route: _FormRoute, cotangent: Array, /) -> Array:
        payload = contract(
            "pclv,pv->pcl",
            jnp.conj(route.weights),
            cotangent.reshape((cotangent.shape[0], -1)),
        )
        return (
            jnp.zeros((self.global_dof_count,), dtype=payload.dtype)
            .at[route.dofs]
            .add(payload)
        )

    def bind_side(
        self, sites: np.ndarray, side: FieldTraceSide, cell_ids: np.ndarray | None, /
    ) -> tuple[np.ndarray | None, np.ndarray]:
        location = self.locator.locate(jnp.asarray(sites))
        if np.any(np.asarray(location.status) != int(CellLocationStatus.LOCATED)):
            raise ValueError("Every form trace site must lie in the FE support.")
        containing = np.asarray(location.candidate_cells)
        if side == "average":
            if cell_ids is not None:
                raise ValueError("Average form traces combine all containing cells.")
            return None, np.full(sites.shape[0], -1, dtype=np.int32)
        if cell_ids is None:
            if side != "owner" or np.any(np.sum(containing >= 0, axis=1) != 1):
                raise ValueError(
                    "Shared form trace sites require explicit side cell_ids."
                )
            selected = np.max(containing, axis=1).astype(np.int32)
        else:
            if np.any(np.all(containing != cell_ids[:, None], axis=1)):
                raise ValueError("Trace cell_ids must contain their declared sites.")
            selected = cell_ids
        mask = np.zeros((self.cell_count,), dtype=np.bool_)
        mask[selected] = True
        resolved = self.locator.locate(jnp.asarray(sites), cell_mask=jnp.asarray(mask))
        if np.any(np.asarray(resolved.candidate_count) != 1) or np.any(
            np.asarray(resolved.cell_ids) != selected
        ):
            raise ValueError("The declared form trace side remains ambiguous.")
        return mask, selected


__all__ = ["FormFieldReconstructionKernel"]
