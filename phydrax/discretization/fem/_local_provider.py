#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._coordinate_measure import CoordinateMeasurePlan
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...geometry._source_array_identity import source_arrays_equal
from .._cell_geometry import PerCellRestrictedCellGeometryElement
from .._cell_geometry_validity import cell_jacobian_inverse_and_measure
from .._integration_domain import IntegrationDomain
from .._local_variational import (
    LocalFieldBinding,
    LocalGeometryActions,
    LocalMetricResult,
    LocalReferenceActions,
    LocalVariationalCapabilities,
    LocalVariationalOffer,
    PreparedLocalRegion,
)
from ._source_surface import SurfaceFiniteElementGeometry


if TYPE_CHECKING:
    from ._generic import FiniteElementDiscretization

# Quadrature-point Jacobians inverted per scanned batch while realizing a
# local metric. The unrolled small inverse materializes dozens of
# intermediates per point; batching keeps them this many points wide instead
# of the whole region (cells x quadrature points).
_JACOBIAN_INVERSE_BATCH = 1 << 12


def _jacobian_inverse_batch(
    carry: None, jacobian: Array, /
) -> tuple[None, tuple[Array, Array, Array]]:
    inverse, _, measure, _, successful = cell_jacobian_inverse_and_measure(jacobian)
    return carry, (inverse, measure, successful)


def _batched_jacobian_inverse(jacobian: Array, /) -> tuple[Array, Array, Array]:
    """Pseudo-inverse, measure and evidence of ``(..., ambient, reference)`` rows.

    Row by row this is `cell_jacobian_inverse_and_measure`; regions wider than
    one batch are scanned in batches by a module-level body, the tail padded
    with exact identity Jacobians whose rows are dropped.
    """
    leading = jacobian.shape[:-2]
    rows = jacobian.reshape((-1,) + jacobian.shape[-2:])
    count = rows.shape[0]
    if count <= _JACOBIAN_INVERSE_BATCH:
        _, (inverse, measure, successful) = _jacobian_inverse_batch(None, rows)
    else:
        batches = -(-count // _JACOBIAN_INVERSE_BATCH)
        padding = jnp.broadcast_to(
            jnp.eye(rows.shape[1], rows.shape[2], dtype=rows.dtype),
            (batches * _JACOBIAN_INVERSE_BATCH - count,) + rows.shape[1:],
        )
        _, (inverse, measure, successful) = jax.lax.scan(
            _jacobian_inverse_batch,
            None,
            jnp.concatenate((rows, padding)).reshape(
                (batches, _JACOBIAN_INVERSE_BATCH) + rows.shape[1:]
            ),
        )
        inverse = inverse.reshape((-1,) + inverse.shape[2:])[:count]
        measure = measure.reshape((-1,))[:count]
        successful = successful.reshape((-1,))[:count]
    return (
        inverse.reshape(leading + inverse.shape[1:]),
        measure.reshape(leading),
        successful.reshape(leading),
    )


def _tabulation_hessians(
    tabulate: Callable[[Array], tuple[Array, Array]], points: Array, /
) -> Array:
    """Differentiate an owned reference-gradient tabulation."""

    def point_gradient(point: Array) -> Array:
        return tabulate(point[None, :])[1][0]

    return jax.vmap(jax.jacfwd(point_gradient))(points)


def _coordinate_actions(
    coordinate_element: object,
    points: Array,
    cell_rows: np.ndarray,
    /,
    *,
    hessians: bool,
) -> tuple[Array, Array, Array | None]:
    """Coordinate basis, gradients and optional Hessians at reference points.

    Shared charts give ``(q, i)``-leading arrays; exact per-cell charts give
    ``(cell, q, i)``-leading arrays for the block rows ``cell_rows``, tabulated
    once per distinct chart.
    """
    if isinstance(coordinate_element, PerCellRestrictedCellGeometryElement):
        rows = np.asarray(cell_rows)
        basis, gradients = coordinate_element.tabulate_cells(rows, points)
        if not hessians:
            return basis, gradients, None

        def point_gradients(point: Array) -> Array:
            return coordinate_element.tabulate_cells(rows, point[None, :])[1][:, 0]

        # (q, cell, i, r, s) -> (cell, q, i, r, s)
        second = jax.vmap(jax.jacfwd(point_gradients))(points)
        return basis, gradients, jnp.moveaxis(second, 0, 1)
    tabulate = getattr(coordinate_element, "tabulate", None)
    if not callable(tabulate):
        raise TypeError(
            f"{type(coordinate_element).__name__} has no scalar coordinate tabulator."
        )
    basis, gradients = tabulate(points)
    return (
        basis,
        gradients,
        _tabulation_hessians(tabulate, points) if hessians else None,
    )


@final
class FiniteElementReferenceActions(LocalReferenceActions):
    """Dense FE reference actions; runtime is intentionally ignored."""

    action_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    kernel_modes: tuple[str, ...] = eqx.field(static=True)
    basis_values: Array
    basis_gradients: Array
    basis_hessians: Array
    coefficient_transform: Array

    def __init__(
        self,
        basis_values: ArrayLike,
        basis_gradients: ArrayLike,
        /,
        *,
        basis_hessians: ArrayLike | None = None,
        coefficient_transform: ArrayLike | None = None,
        maximum_derivative_order: int,
        kernel_modes: Sequence[str],
        action_id: str | None = None,
    ) -> None:
        values = jnp.asarray(basis_values)
        gradients = jnp.asarray(basis_gradients)
        hessians = (
            jnp.empty((0,), dtype=gradients.dtype)
            if basis_hessians is None
            else jnp.asarray(basis_hessians)
        )
        if values.ndim not in (2, 3) or gradients.ndim != values.ndim + 1:
            raise ValueError("FE reference values and gradients have invalid rank.")
        if values.shape != gradients.shape[:-1]:
            raise ValueError("FE reference values and gradients disagree.")
        if hessians.size and hessians.shape != gradients.shape + (gradients.shape[-1],):
            raise ValueError("FE reference Hessians have incompatible axes.")
        width = values.shape[-1]
        point_count = values.shape[-2]
        transform = (
            jnp.empty((0,), dtype=values.dtype)
            if coefficient_transform is None
            else jnp.asarray(coefficient_transform, dtype=values.dtype)
        )
        if transform.size and (
            transform.ndim != 3 or transform.shape[-2:] != (width, width)
        ):
            raise ValueError(
                "FE coefficient transforms must have shape (cells,width,width)."
            )
        derivative_order = int(maximum_derivative_order)
        modes = tuple(dict.fromkeys(str(value) for value in kernel_modes))
        if derivative_order < 0 or derivative_order > 2:
            raise ValueError(
                "FE reference actions support derivative orders zero to two."
            )
        if derivative_order == 2 and not hessians.size:
            raise ValueError("Second-order FE actions require reference Hessians.")
        if not modes or any(
            value not in ("dense", "partial", "sum_factorized", "collocated")
            for value in modes
        ):
            raise ValueError("FE reference kernel modes are invalid.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-element-reference-actions",
                    "values": array_tree_fingerprint(values),
                    "gradients": array_tree_fingerprint(gradients),
                    "hessians": (
                        None if not hessians.size else array_tree_fingerprint(hessians)
                    ),
                    "coefficient_transform": array_tree_fingerprint(transform),
                    "maximum_derivative_order": derivative_order,
                    "kernel_modes": modes,
                }
            )
            if action_id is None
            else str(action_id)
        )
        if not identifier:
            raise ValueError("action_id must be non-empty.")
        self.realization_id = "finite-element-dense"
        self.action_id = identifier
        self.local_width = width
        self.point_count = point_count
        self.maximum_derivative_order = derivative_order
        self.kernel_modes = modes
        self.basis_values = values
        self.basis_gradients = gradients
        self.basis_hessians = hessians
        self.coefficient_transform = transform

    def _reference_coefficients(self, coefficients: Array, /) -> Array:
        if not self.coefficient_transform.size:
            return coefficients
        return ein.contract("cij,cj...->ci...", self.coefficient_transform, coefficients)

    def _gathered_coefficients(self, coefficients: Array, /) -> Array:
        if not self.coefficient_transform.size:
            return coefficients
        return ein.contract("cij,ci...->cj...", self.coefficient_transform, coefficients)

    def realize_reference_actions(
        self, runtime: object, /
    ) -> FiniteElementReferenceActions:
        del runtime
        return self

    def interpolate(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        del runtime
        coefficients = self._reference_coefficients(jnp.asarray(local_coefficients))
        if self.basis_values.ndim == 2:
            return ein.contract("qi,ci...->cq...", self.basis_values, coefficients)
        return ein.contract("cqi,ci...->cq...", self.basis_values, coefficients)

    def interpolate_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        del runtime
        values_ = jnp.asarray(values)
        if self.basis_values.ndim == 2:
            local = ein.contract("qi,cq...->ci...", self.basis_values, values_)
        else:
            local = ein.contract("cqi,cq...->ci...", self.basis_values, values_)
        return self._gathered_coefficients(local)

    def reference_gradient(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        del runtime
        coefficients = self._reference_coefficients(jnp.asarray(local_coefficients))
        if self.basis_gradients.ndim == 3:
            return ein.contract("qir,ci...->cq...r", self.basis_gradients, coefficients)
        return ein.contract("cqir,ci...->cq...r", self.basis_gradients, coefficients)

    def reference_gradient_transpose(
        self, runtime: object, gradients: ArrayLike, /
    ) -> Array:
        del runtime
        gradients_ = jnp.asarray(gradients)
        if self.basis_gradients.ndim == 3:
            local = ein.contract("qir,cq...r->ci...", self.basis_gradients, gradients_)
        else:
            local = ein.contract("cqir,cq...r->ci...", self.basis_gradients, gradients_)
        return self._gathered_coefficients(local)

    def reference_hessian(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        del runtime
        if not self.basis_hessians.size:
            raise ValueError("FE reference Hessian actions were not prepared.")
        coefficients = self._reference_coefficients(jnp.asarray(local_coefficients))
        if self.basis_hessians.ndim == 4:
            return ein.contract("qirs,ci...->cq...rs", self.basis_hessians, coefficients)
        return ein.contract("cqirs,ci...->cq...rs", self.basis_hessians, coefficients)

    def reference_hessian_transpose(
        self, runtime: object, hessians: ArrayLike, /
    ) -> Array:
        del runtime
        if not self.basis_hessians.size:
            raise ValueError("FE reference Hessian transpose actions were not prepared.")
        values = jnp.asarray(hessians)
        if self.basis_hessians.ndim == 4:
            local = ein.contract("qirs,cq...rs->ci...", self.basis_hessians, values)
        else:
            local = ein.contract("cqirs,cq...rs->ci...", self.basis_hessians, values)
        return self._gathered_coefficients(local)

    def trace(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        return self.interpolate(runtime, local_coefficients)

    def trace_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        return self.interpolate_transpose(runtime, values)


@final
class FiniteElementGeometryActions(LocalGeometryActions):
    """Fixed FE coordinate routes with runtime coordinates supplied at realization."""

    action_id: str = eqx.field(static=True)
    runtime_layout_id: str = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    domain_kind: str = eqx.field(static=True)
    coordinate_basis: Array
    coordinate_gradients: Array
    coordinate_gathers: Array
    coordinate_hessians: Array
    reference_weights: Array
    coordinate_measure: CoordinateMeasurePlan | None

    def __init__(
        self,
        runtime_layout_id: str,
        domain_kind: str,
        coordinate_basis: ArrayLike,
        coordinate_gradients: ArrayLike,
        coordinate_gathers: ArrayLike,
        reference_weights: ArrayLike,
        /,
        *,
        coordinate_hessians: ArrayLike | None = None,
        coordinate_measure: CoordinateMeasurePlan | None = None,
    ) -> None:
        layout = str(runtime_layout_id)
        kind = str(domain_kind)
        basis = jnp.asarray(coordinate_basis)
        gradients = jnp.asarray(coordinate_gradients)
        hessians = (
            jnp.empty((0,), dtype=gradients.dtype)
            if coordinate_hessians is None
            else jnp.asarray(coordinate_hessians)
        )
        gathers = jnp.asarray(coordinate_gathers, dtype=jnp.int32)
        weights = jnp.asarray(reference_weights)
        # Shared charts store (q, i)-leading tabulations; exact per-cell charts
        # store one (cell, q, i)-leading tabulation per gathered cell.
        per_cell = basis.ndim == 3
        if (
            not layout
            or kind != "cell"
            or basis.ndim not in (2, 3)
            or gradients.ndim != basis.ndim + 1
            or gradients.shape[: basis.ndim] != basis.shape
            or (
                hessians.size
                and hessians.shape != gradients.shape + (gradients.shape[-1],)
            )
            or gathers.ndim != 2
            or gathers.shape[1] != basis.shape[-1]
            or (per_cell and basis.shape[0] != gathers.shape[0])
            or weights.shape != (basis.shape[-2],)
        ):
            raise ValueError("FE local geometry actions are inconsistent.")
        self.runtime_layout_id = layout
        self.domain_kind = kind
        self.entity_count = gathers.shape[0]
        self.coordinate_basis = basis
        self.coordinate_gradients = gradients
        self.coordinate_gathers = gathers
        self.coordinate_hessians = hessians
        self.reference_weights = weights
        self.coordinate_measure = coordinate_measure
        self.action_id = canonical_fingerprint(
            {
                "kind": "finite-element-local-geometry-actions",
                "runtime_layout": layout,
                "domain_kind": kind,
                "coordinate_basis": array_tree_fingerprint(basis),
                "coordinate_gradients": array_tree_fingerprint(gradients),
                "coordinate_gathers": array_tree_fingerprint(gathers),
                "coordinate_hessians": (
                    None if not hessians.size else array_tree_fingerprint(hessians)
                ),
                "reference_weights": array_tree_fingerprint(weights),
                **(
                    {"coordinate_measure": coordinate_measure.plan_id}
                    if coordinate_measure is not None
                    and coordinate_measure.kind != "cartesian"
                    else {}
                ),
            }
        )

    def realize(self, runtime: object, /) -> LocalMetricResult:
        from ._generic import FiniteElementRuntimeData

        if not isinstance(runtime, FiniteElementRuntimeData):
            raise TypeError("runtime must be FiniteElementRuntimeData.")
        coordinates = jnp.asarray(runtime.coordinates)[self.coordinate_gathers]
        cells = "c" if self.coordinate_basis.ndim == 3 else ""
        points = ein.contract(f"{cells}qi,cid->cqd", self.coordinate_basis, coordinates)
        jacobian = ein.contract(
            f"{cells}qir,cid->cqdr", self.coordinate_gradients, coordinates
        )
        inverse, measure, successful = _batched_jacobian_inverse(jacobian)
        measure = eqx.error_if(
            measure,
            jnp.any(~successful | ~jnp.isfinite(measure) | (measure <= 0.0)),
            "Finite-element metric measure must be positive and finite.",
        )
        inverse_hessian = None
        if self.coordinate_hessians.size:
            mapping_hessian = ein.contract(
                f"{cells}qirs,cid->cqdrs", self.coordinate_hessians, coordinates
            )
            inverse_hessian = -ein.contract(
                "cqrd,cqdst,cqsa,cqtb->cqrab",
                inverse,
                mapping_hessian,
                inverse,
                inverse,
            )
        weights = measure * self.reference_weights[None, :]
        if self.coordinate_measure is not None:
            weights = self.coordinate_measure.weight(
                points, weights, frame=runtime.coordinate_frame
            )
            radius_axis = self.coordinate_measure.radial_axis
            if radius_axis is not None:
                valid = jnp.all(jnp.isfinite(coordinates)) & jnp.all(
                    coordinates[..., radius_axis] >= 0.0
                )
                weights = jnp.where(valid, weights, jnp.nan)
        return LocalMetricResult(
            points,
            weights,
            jacobian,
            inverse,
            inverse_hessian=inverse_hessian,
        )


@final
class SourceSurfaceGeometryActions(LocalGeometryActions):
    """Source immersion evaluated on the selected fixed carrier cell rows."""

    action_id: str = eqx.field(static=True)
    runtime_layout_id: str = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    domain_kind: str = eqx.field(static=True)
    source: SurfaceFiniteElementGeometry
    block_index: int = eqx.field(static=True)
    rows: Array
    reference_points: Array
    reference_weights: Array
    cached_points: Array
    cached_jacobian: Array
    cached_inverse_jacobian: Array
    cached_measure: Array
    cache_reference: tuple[Array, ...]

    def __init__(
        self,
        source: SurfaceFiniteElementGeometry,
        block_index: int,
        rows: ArrayLike,
        points: ArrayLike,
        weights: ArrayLike,
        /,
    ) -> None:
        from ...geometry._brep_capped_material import BRepCappedMaterialMap

        self.source = source
        self.block_index = int(block_index)
        self.rows = jnp.asarray(rows, dtype=jnp.int32)
        self.reference_points = jnp.asarray(points)
        self.reference_weights = jnp.asarray(weights)
        host_rows = np.asarray(rows, dtype=np.int32)
        if (
            isinstance(source.material_map, BRepCappedMaterialMap)
            and host_rows.size != source.parameters[block_index].shape[0]
        ):
            material_rows = source.material_map.prepare_rows(block_index, host_rows)
            physical, jacobian = material_rows.evaluate(self.reference_points)
        else:
            physical, jacobian = source.evaluate(block_index, self.reference_points)
            physical, jacobian = physical[self.rows], jacobian[self.rows]
        inverse, measure, successful = _batched_jacobian_inverse(jacobian)
        measure = eqx.error_if(
            measure,
            jnp.any(~successful | ~jnp.isfinite(measure) | (measure <= 0.0)),
            "Finite-element source metric must be positive and finite.",
        )
        self.cached_points = physical
        self.cached_jacobian = jacobian
        self.cached_inverse_jacobian = inverse
        self.cached_measure = measure
        self.cache_reference = (
            physical, jacobian, inverse, measure,
            self.reference_points, self.reference_weights, self.rows,
        )
        self.runtime_layout_id = source.geometry_layout_id
        self.entity_count = int(self.rows.shape[0])
        self.domain_kind = "cell"
        self.action_id = canonical_fingerprint({
            "kind": "source-surface-local-geometry",
            "source": source.source_geometry_id,
            "block": self.block_index,
            "rows": array_tree_fingerprint(self.rows),
            "points": array_tree_fingerprint(self.reference_points),
            "weights": array_tree_fingerprint(self.reference_weights),
        })

    def realize(self, runtime: object, /) -> LocalMetricResult:
        from ._generic import FiniteElementRuntimeData

        if not isinstance(runtime, FiniteElementRuntimeData):
            raise TypeError("runtime must be FiniteElementRuntimeData.")
        if runtime.geometry_layout_id != self.runtime_layout_id:
            raise ValueError("Source surface runtime has a different carrier layout.")
        coordinates = self.source.require_carrier(runtime.coordinates)
        cached = (
            self.cached_points,
            self.cached_jacobian,
            self.cached_inverse_jacobian,
            self.cached_measure,
        )
        changed = self.source.source_changed() | ~source_arrays_equal(
            coordinates, self.source.carrier_coordinates
        )
        current_values = (*cached, self.reference_points, self.reference_weights, self.rows)
        for current, original in zip(current_values, self.cache_reference, strict=True):
            if current.shape != original.shape or current.dtype != original.dtype:
                raise ValueError("Source surface metric cache layout changed after proof.")
            changed = changed | ~source_arrays_equal(current, original)
        points, jacobian, inverse, measure = eqx.error_if(
            cached, changed, "Source surface metric or carrier changed after proof."
        )
        return LocalMetricResult(
            points, measure * self.reference_weights[None, :], jacobian, inverse
        )


@final
class FiniteElementLocalProvider(StrictModule):
    """Adapter from stable FE storage to the prepared-local contract."""

    discretization: FiniteElementDiscretization

    def __init__(self, discretization: FiniteElementDiscretization, /) -> None:
        from ._generic import FiniteElementDiscretization

        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        self.discretization = discretization

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        common_semantics = (
            "exact_interpolation_transpose",
            "exact_gradient_transpose",
            "exact_trace_transpose",
        )
        return LocalVariationalCapabilities(
            "finite-element-local-provider",
            (
                LocalVariationalOffer(
                    "prepared-local",
                    ("cell",),
                    ("diffusion", "tensor-diffusion", "mass", "source"),
                    ("value", "grad"),
                    ("value", "gradient"),
                    (
                        "dense",
                        "partial",
                        "sum_factorized",
                        "collocated",
                    ),
                    ("matrix_free", "sparse"),
                    ("finite-element-dense",),
                    automatic_kernel_mode="dense",
                    automatic_operator_realization="sparse",
                    automatic_reference_realization_id="finite-element-dense",
                    action_semantics=common_semantics,
                ),
                LocalVariationalOffer(
                    "native",
                    ("cell", "exterior_facet", "interior_facet"),
                    (
                        "diffusion",
                        "mass",
                        "source",
                        "boundary-load",
                        "cell-residual",
                        "pairwise-volume-flux",
                        "interior-facet",
                        "exterior-facet",
                        "sipg-facet",
                        "cell-energy",
                        "functional",
                        "cell-bilinear",
                        "operator-action",
                    ),
                    (
                        "value",
                        "grad",
                        "sym-grad",
                        "div",
                        "curl",
                        "normal-trace",
                        "tangential-trace",
                        "jump",
                        "average",
                    ),
                    ("value", "gradient", "trace", "curl", "divergence"),
                    (
                        "dense",
                        "partial",
                        "sum_factorized",
                        "collocated",
                    ),
                    ("matrix_free", "sparse"),
                    ("finite-element-native",),
                    automatic_kernel_mode="dense",
                    automatic_operator_realization="sparse",
                    automatic_reference_realization_id="finite-element-native",
                    action_semantics=common_semantics,
                    material_modes=("none", "local"),
                    history_modes=("none", "local"),
                    explicit_rules=True,
                ),
            ),
        )

    def local_field_binding(self, name: str, /) -> LocalFieldBinding:
        discretization = self.discretization
        field_index = discretization._field_index(name)
        dof_map = discretization.dof_maps[field_index]
        elements = discretization.elements[field_index]
        widths = tuple(element.local_dof_count for element in elements)
        value_shapes = tuple(element.value_shape for element in elements)
        if len(set(value_shapes)) != 1:
            raise ValueError("One local field binding requires a uniform FE value shape.")
        component_shape = tuple(dof_map.component_shape)
        execution_shape = component_shape
        public_shape = component_shape
        return LocalFieldBinding(
            str(name),
            discretization.field_spaces[field_index],
            component_shape=component_shape,
            public_shape=public_shape,
            execution_shape=execution_shape,
            local_width=max(widths),
            layout_id=canonical_fingerprint(
                {
                    "kind": "finite-element-local-field-layout",
                    "dof_map": dof_map.dof_map_id,
                    "elements": [element.element_id for element in elements],
                }
            ),
        )

    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        from ._generic import _degree_aware_reference_rule

        discretization = self.discretization
        if not isinstance(domain, IntegrationDomain) or domain.kind != "cell":
            raise ValueError("The generic FE local provider currently prepares cells.")
        if (
            domain.support_id != discretization.cell_domain.support_id
            or domain.entity_set_id != discretization.cell_domain.entity_set_id
        ):
            raise ValueError("Finite-element cell domain belongs to another support.")
        names = tuple(str(name) for name in field_names)
        if any(
            element.value_spec.form_type.degree != 0
            or (
                element.continuity != "conforming"
                and not (
                    element.continuity == "discontinuous"
                    and element.form_basis is None
                    and element.mapping == "identity"
                    and element.value_spec.form_type.twist == "untwisted"
                )
            )
            for name in names
            for element in discretization.elements[discretization._field_index(name)]
        ):
            raise ValueError(
                "Generic prepared-local FE execution requires conforming scalar "
                "or identity-mapped discontinuous scalar elements."
            )
        mode = "dense" if str(kernel_mode) == "auto" else str(kernel_mode)
        if mode not in ("dense", "partial", "sum_factorized", "collocated"):
            raise ValueError("Unknown FE prepared-local kernel mode.")
        entity_set = set(int(value) for value in np.asarray(domain.entity_indices))
        regions = []
        cell_offset = 0
        for block_index, block in enumerate(discretization.mesh.blocks):
            block_cells = np.arange(
                cell_offset, cell_offset + block.cell_count, dtype=np.int32
            )
            cell_offset += block.cell_count
            selected = np.asarray(
                [
                    index
                    for index, cell in enumerate(block_cells)
                    if int(cell) in entity_set
                ],
                dtype=np.int32,
            )
            if selected.size == 0 and entity_set:
                continue
            # ``selected`` enumerates distinct block rows in increasing order.
            # A nonempty full count therefore names exactly the identity route.
            all_block_rows = bool(block.cell_count and selected.size == block.cell_count)
            entities = block_cells[selected]
            rows_by_entity = {
                int(entity): row
                for row, entity in enumerate(np.asarray(domain.entity_indices))
            }
            domain_rows = np.asarray(
                [rows_by_entity[int(entity)] for entity in entities], dtype=np.int32
            )
            degrees = tuple(
                discretization.elements[discretization._field_index(name)][
                    block_index
                ].degree
                for name in names
            )
            degree = max(degrees)
            if (
                discretization.coordinate_measure is not None
                and discretization.coordinate_measure.kind == "spherical-radial"
            ):
                degree += 1
            points, weights = _degree_aware_reference_rule(
                block.cell_kind,
                degree,
                # ty: ignore[unresolved-attribute]
                geometry_degree=discretization.coordinate_elements[block_index].degree,
            )
            references = []
            gathers = []
            for name in names:
                field_index = discretization._field_index(name)
                element = discretization.elements[field_index][block_index]
                basis, gradients = element.tabulate(points)
                hessians = (
                    _tabulation_hessians(element.tabulate, points)
                    if maximum_derivative_order == 2
                    else None
                )
                coefficient_transform = None
                if element.form_basis is not None:
                    coefficient_transform = discretization.dof_maps[
                        field_index
                    ].cell_transforms[block_index]
                    if not all_block_rows:
                        coefficient_transform = coefficient_transform[selected]
                references.append(
                    FiniteElementReferenceActions(
                        basis,
                        gradients,
                        basis_hessians=hessians,
                        coefficient_transform=coefficient_transform,
                        maximum_derivative_order=maximum_derivative_order,
                        kernel_modes=(mode,),
                        action_id=canonical_fingerprint(
                            {
                                "kind": "finite-element-local-reference-actions",
                                "element": element.element_id,
                                "dof_map": discretization.dof_maps[
                                    field_index
                                ].dof_map_id,
                                "selected_cells": selected.tolist(),
                                "points": array_tree_fingerprint(points),
                                "maximum_derivative_order": maximum_derivative_order,
                                "kernel_mode": mode,
                            }
                        ),
                    )
                )
                cell_dofs = discretization.dof_maps[field_index].cell_dofs[block_index]
                gathers.append(cell_dofs if all_block_rows else cell_dofs[selected])
            if discretization.source_surface is not None:
                if maximum_derivative_order == 2:
                    raise ValueError("Source surface local geometry does not provide inverse-map Hessians.")
                geometry = SourceSurfaceGeometryActions(
                    discretization.source_surface, block_index, selected, points, weights
                )
            else:
                coordinate_basis, coordinate_gradients, coordinate_hessians = (
                    _coordinate_actions(
                        discretization.coordinate_elements[block_index],
                        points,
                        selected,
                        hessians=maximum_derivative_order == 2,
                    )
                )
                geometry = FiniteElementGeometryActions(
                    discretization.default_runtime.geometry_layout_id,
                    domain.kind,
                    coordinate_basis,
                    coordinate_gradients,
                    discretization.coordinate_dofs[block_index][selected],
                    weights,
                    coordinate_hessians=coordinate_hessians,
                    coordinate_measure=discretization.coordinate_measure,
                )
            selected_domain = IntegrationDomain(
                "cell",
                entities,
                domain.support_id,
                domain.entity_set_id,
                owner_cells=np.asarray(domain.owner_cells)[domain_rows],
                neighbor_cells=np.asarray(domain.neighbor_cells)[domain_rows],
                owner_local_entities=np.asarray(domain.owner_local_entities)[domain_rows],
                neighbor_local_entities=np.asarray(domain.neighbor_local_entities)[
                    domain_rows
                ],
                selection_id=domain.selection_id,
            )
            regions.append(
                PreparedLocalRegion(
                    selected_domain,
                    names,
                    gathers,
                    references,
                    geometry,
                    block_name=block.name,
                    cell_kind=block.cell_kind,
                )
            )
        if not regions:
            raise ValueError("The local domain selects no finite-element cells.")
        return tuple(regions)

    def validate_local_runtime(self, runtime: object, /) -> None:
        from ._generic import FiniteElementRuntimeData

        discretization = self.discretization
        if not isinstance(runtime, FiniteElementRuntimeData):
            raise TypeError("runtime must be FiniteElementRuntimeData.")
        if (
            runtime.topology_id != discretization.mesh.topology_id
            or runtime.geometry_layout_id
            != discretization.default_runtime.geometry_layout_id
            or runtime.coordinates.shape
            != discretization.default_runtime.coordinates.shape
        ):
            raise ValueError("Finite-element runtime does not match the prepared layout.")
        measure = discretization.coordinate_measure
        if measure is not None and runtime.coordinate_frame != measure.frame:
            raise ValueError(
                "Runtime coordinate frame must match the coordinate measure."
            )


__all__ = [
    "FiniteElementGeometryActions",
    "FiniteElementLocalProvider",
    "FiniteElementReferenceActions",
]
