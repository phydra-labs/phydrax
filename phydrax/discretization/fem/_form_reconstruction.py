# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...exterior._algebra import map_reference_values
from ...exterior._form_type import FormType, FormValueSpec
from ...typing import Dim, Float, Int32
from .._cell_geometry import coordinate_tabulation
from .._cell_locator import AbstractCellLocator, CellLocationStatus
from .._mapped_locator import PreparedMappedCellLocator, PreparedMappedCellMap
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


# Query rows (point x candidate) whose basis is evaluated at once: the vmapped
# cell-map evaluation materializes per-row geometry, so rows are processed in
# chunks and the working set stays bounded whatever the number of points.
_BASIS_ROW_BUDGET = 1 << 12


def _row_chunked(
    function: Callable[[Array, Array], Array], cells: Array, points: Array, /
) -> Array:
    rows = cells.shape[0]
    parts = [
        function(
            cells[start : start + _BASIS_ROW_BUDGET],
            points[start : start + _BASIS_ROW_BUDGET],
        )
        for start in range(0, rows, _BASIS_ROW_BUDGET)
    ]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)


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
        if not isinstance(
            locator.cell_map,
            (PreparedFiniteElementCellMap, PreparedMappedCellMap),
        ):
            raise TypeError(
                "Form reconstruction requires a prepared coordinate cell map."
            )
        if isinstance(locator.cell_map, PreparedMappedCellMap) and not isinstance(
            locator, PreparedMappedCellLocator
        ):
            raise TypeError(
                "Mapped form reconstruction requires PreparedMappedCellLocator."
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
        cell_map = self.locator.cell_map
        if not isinstance(cell_map, PreparedFiniteElementCellMap):
            raise TypeError(
                "Form reconstruction requires a prepared finite-element cell map."
            )
        evaluation = cell_map.evaluate(
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

    def _mapped_geometry_jacobian(self, cell_row: int, reference: Array, /) -> Array:
        locator = self.locator
        if not isinstance(locator, PreparedMappedCellLocator) or not isinstance(
            locator.cell_map, PreparedMappedCellMap
        ):
            raise TypeError(
                "Mapped form evaluation requires a prepared mapped cell locator."
            )
        _, gradients = coordinate_tabulation(
            locator.cell_map.coordinate_element,
            reference[None],
            cell_row=cell_row,
        )
        local_coordinates = locator.coordinates[
            locator.cell_map.coordinate_dofs[cell_row]
        ]
        return contract("plr,la->ar", gradients, local_coordinates)[0]

    def _mapped_basis(self, cell_row: int, reference: Array, /) -> Array:
        jacobian = self._mapped_geometry_jacobian(cell_row, reference)
        values = self.element.tabulate(reference[None])[0]
        mapped = map_reference_values(
            values, self.element.value_spec, jacobian[None, None]
        )[0]
        return contract(
            "iv,ij->jv",
            mapped.reshape((self.element.local_dof_count, -1)),
            self.cell_transforms[cell_row],
        )

    def _mapped_inverse_jacobian(self, cell_row: int, reference: Array, /) -> Array:
        jacobian = self._mapped_geometry_jacobian(cell_row, reference)
        reference_dimension = jacobian.shape[1]
        if reference_dimension == jacobian.shape[0]:
            return jnp.linalg.solve(
                jacobian,
                jnp.eye(reference_dimension, dtype=jacobian.dtype),
            )
        return jnp.linalg.solve(jacobian.T @ jacobian, jacobian.T)

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
        accepted = location.candidate_cells >= 0
        count = jnp.sum(accepted, axis=1, dtype=jnp.int32)
        if side is not None and side.side == "average":
            candidates = location.candidate_cells
            candidate_reference = location.candidate_reference
            share = accepted / jnp.maximum(count, 1)[:, None]
        else:
            # Without averaging only the first containing candidate carries
            # weight, so the route keeps that one candidate and the basis is
            # evaluated for it alone (the others would be multiplied by zero).
            chosen = jnp.argmax(accepted, axis=1)[:, None]
            candidates = jnp.take_along_axis(location.candidate_cells, chosen, axis=1)
            candidate_reference = jnp.take_along_axis(
                location.candidate_reference, chosen[:, :, None], axis=1
            )
            share = jnp.take_along_axis(accepted, chosen, axis=1)
        safe = jnp.where(candidates >= 0, candidates, 0)
        cells = safe.reshape((-1,))
        reference = candidate_reference.reshape(
            (-1, self.locator.cell_map.reference_dimension)
        )
        if isinstance(self.locator, PreparedMappedCellLocator) and isinstance(
            self.locator.cell_map, PreparedMappedCellMap
        ):
            cell_ids = np.asarray(cells, dtype=np.int64)
            weights = jnp.zeros_like(self.element.tabulate(reference)[0])
            for cell_id in np.unique(cell_ids):
                cell_row = int(cell_id)
                rows = jnp.asarray(np.flatnonzero(cell_ids == cell_row), dtype=jnp.int32)
                references = reference[rows]

                def basis_fn(point: Array, /) -> Array:
                    return self._mapped_basis(cell_row, point)

                if order == 0:
                    group_weights = jax.vmap(basis_fn)(references)
                else:
                    axis = derivative.index(1)
                    reference_gradient = jax.vmap(jax.jacfwd(basis_fn, argnums=0))(
                        references
                    )

                    def inverse_jacobian_fn(point: Array, /) -> Array:
                        return self._mapped_inverse_jacobian(cell_row, point)

                    inverse = jax.vmap(inverse_jacobian_fn)(references)
                    group_weights = contract(
                        "plvr,pr->plv",
                        reference_gradient,
                        inverse[:, :, axis],
                    )
                weights = weights.at[rows].set(group_weights)
        elif order == 0:
            weights = _row_chunked(jax.vmap(self._basis), cells, reference)
        else:
            axis = derivative.index(1)
            reference_gradient = _row_chunked(
                jax.vmap(jax.jacfwd(self._basis, argnums=1)), cells, reference
            )
            cell_map = self.locator.cell_map
            if not isinstance(cell_map, PreparedFiniteElementCellMap):
                raise TypeError(
                    "Form reconstruction requires a prepared finite-element cell map."
                )
            coordinates = self.locator.coordinates

            def inverse_jacobian(rows: Array, points: Array, /) -> Array:
                return cell_map.evaluate(coordinates, rows, points).inverse_jacobian

            inverse = _row_chunked(inverse_jacobian, cells, reference)
            weights = contract("plvr,pr->plv", reference_gradient, inverse[:, :, axis])
        weights = weights.reshape((*candidates.shape, self.element.local_dof_count, -1))
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
