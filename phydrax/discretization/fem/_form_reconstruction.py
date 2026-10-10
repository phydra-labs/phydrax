# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

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
from .._simplicial_locator import (
    AbstractCellLocator,
    CellLocationStatus,
)
from .._view_support import cell_location_status
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTraceSide,
)
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
