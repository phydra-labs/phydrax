#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization.fem._reference_rules as reference_rules_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
import phydrax.ein as ein

from ..._coordinate_measure import CoordinateMeasurePlan
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ...typing import checked, parse
from .. import (
    _cell_geometry as imported_cell_geometry,
    _coordinate_enclosure as imported_coordinate_enclosure,
)
from .._coordinate_budget import CoordinateEnclosureBudget
from .._coordinate_enclosure import coordinate_estimate_bank, source_basis_estimate
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
from . import _generic as imported_generic


if TYPE_CHECKING:
    from phydrax.discretization._reference_cell_types import CellGeometryElement
    from phydrax.discretization.fem._reference_types import FiniteElementSpec

    from .._cell_geometry import (
        BarycentricCellGeometryElement,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from .._coordinate_enclosure import Polynomial
    from ._generic import FiniteElementDiscretization


type FiniteElementKernelMode = Literal["dense", "partial", "sum_factorized", "collocated"]


def require_tabulated_coordinate_element(
    element: CellGeometryElement,
) -> (
    reference_types_owner.FiniteElementSpec
    | imported_cell_geometry.BarycentricCellGeometryElement
    | imported_cell_geometry.RestrictedCellGeometryElement
    | imported_cell_geometry.PolynomialComposedCellGeometryElement
    | imported_cell_geometry.RationalComposedCellGeometryElement
    | imported_cell_geometry.SplineCellGeometryElement
    | imported_cell_geometry.LayerColumnCellGeometryElement
):

    match element:
        case (
            reference_types_owner.FiniteElementSpec()
            | imported_cell_geometry.BarycentricCellGeometryElement()
            | imported_cell_geometry.RestrictedCellGeometryElement()
            | imported_cell_geometry.PolynomialComposedCellGeometryElement()
            | imported_cell_geometry.RationalComposedCellGeometryElement()
            | imported_cell_geometry.SplineCellGeometryElement()
            | imported_cell_geometry.LayerColumnCellGeometryElement()
        ):
            return element
        case _:
            raise TypeError(
                "require_tabulated_coordinate_element.element must support scalar coordinate tabulation; got Coordinate element must support scalar coordinate tabulation."
            )


def _tabulation_hessians(
    element: FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    points: Array,
    /,
) -> Array:
    """Differentiate the element's owned reference-gradient action."""

    def point_gradient(point: Array) -> Array:
        return element.tabulate(point[None, :])[1][0]

    return jax.vmap(jax.jacfwd(point_gradient))(points)


@final
class FiniteElementReferenceActions(LocalReferenceActions):
    """Dense FE reference actions; runtime is intentionally ignored."""

    action_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    kernel_modes: tuple[FiniteElementKernelMode, ...] = eqx.field(static=True)
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
            raise ValueError(
                "FiniteElementReferenceActions.configuration must have valid fe reference values and gradients; got FE reference values and gradients have invalid rank."
            )
        if values.shape != gradients.shape[:-1]:
            raise ValueError(
                "FiniteElementReferenceActions.configuration must agree with the declared geometry, topology, and field layout; got FE reference values and gradients disagree."
            )
        if hessians.size and hessians.shape != gradients.shape + (gradients.shape[-1],):
            raise ValueError(
                "FiniteElementReferenceActions.configuration must agree with the declared geometry, topology, and field layout; got FE reference Hessians have incompatible axes."
            )
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
                "FiniteElementReferenceActions.configuration must have shape (cells,width,width); got FE coefficient transforms must have shape (cells,width,width)."
            )
        derivative_order = int(maximum_derivative_order)
        modes = tuple(dict.fromkeys(str(value) for value in kernel_modes))
        if derivative_order < 0 or derivative_order > 2:
            raise ValueError(
                "FiniteElementReferenceActions.configuration must use a derivative order between zero and two; got FE reference actions support derivative orders zero to two."
            )
        if derivative_order == 2 and not hessians.size:
            raise ValueError(
                "FiniteElementReferenceActions.configuration must provide reference Hessians; got Second-order FE actions require reference Hessians."
            )
        if not modes:
            raise ValueError(
                "FiniteElementReferenceActions.configuration must have valid fe reference kernel modes; got FE reference kernel modes are invalid."
            )
        modes = tuple(
            parse(
                value,
                FiniteElementKernelMode,
                "FiniteElementReferenceActions.kernel_modes",
            )
            for value in modes
        )
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
            raise ValueError(
                "FiniteElementReferenceActions.action_id must be non-empty; got action_id must be non-empty."
            )
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
            raise ValueError(
                "FiniteElementReferenceActions.reference_hessian.configuration must include prepared reference Hessians; got FE reference Hessian actions were not prepared."
            )
        coefficients = self._reference_coefficients(jnp.asarray(local_coefficients))
        if self.basis_hessians.ndim == 4:
            return ein.contract("qirs,ci...->cq...rs", self.basis_hessians, coefficients)
        return ein.contract("cqirs,ci...->cq...rs", self.basis_hessians, coefficients)

    def reference_hessian_transpose(
        self, runtime: object, hessians: ArrayLike, /
    ) -> Array:
        del runtime
        if not self.basis_hessians.size:
            raise ValueError(
                "FiniteElementReferenceActions.reference_hessian_transpose.configuration must include prepared reference Hessian transpose actions; got FE reference Hessian transpose actions were not prepared."
            )
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
        if (
            not layout
            or kind != "cell"
            or basis.ndim not in (2, 3)
            or gradients.ndim != basis.ndim + 1
            or gradients.shape[:-1] != basis.shape
            or (
                hessians.size
                and hessians.shape != gradients.shape + (gradients.shape[-1],)
            )
            or gathers.ndim != 2
            or gathers.shape[1] != basis.shape[-1]
            or weights.shape != (basis.shape[-2],)
            or (basis.ndim == 3 and basis.shape[0] != gathers.shape[0])
        ):
            raise ValueError(
                "FiniteElementGeometryActions.configuration must agree with the declared geometry, topology, and field layout; got FE local geometry actions are inconsistent."
            )
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

        if not isinstance(runtime, imported_generic.FiniteElementRuntimeData):
            raise TypeError(
                "FiniteElementGeometryActions.realize.runtime must be FiniteElementRuntimeData; got runtime must be FiniteElementRuntimeData."
            )
        coordinates = jnp.asarray(runtime.coordinates)[self.coordinate_gathers]
        cellwise = self.coordinate_basis.ndim == 3
        points = ein.contract(
            "cqi,cid->cqd" if cellwise else "qi,cid->cqd",
            self.coordinate_basis,
            coordinates,
        )
        jacobian = ein.contract(
            "cqir,cid->cqdr" if cellwise else "qir,cid->cqdr",
            self.coordinate_gradients,
            coordinates,
        )
        dimension = jacobian.shape[-1]
        square = jacobian.shape[-2] == dimension
        matrix = (
            jacobian if square else ein.contract("cqdi,cqdj->cqij", jacobian, jacobian)
        )
        if dimension <= 4:
            result = inverse_small_linear(SmallLinearSolvePlan(dimension), matrix)
            inverse_matrix = result.value
            successful = result.successful
            measure = (
                jnp.abs(result.determinant)
                if square
                else jnp.sqrt(jnp.where(successful, result.determinant, 0.0))
            )
        else:
            space = ArraySpace(
                (dimension,),
                dtype=matrix.dtype,
                space_id=f"{self.action_id}:reference-tangent",
            )
            operator = DenseLinearOperator(
                matrix,
                source=space,
                target=space,
                operator_id=f"{self.action_id}:jacobian"
                if square
                else f"{self.action_id}:metric",
            )
            prepared = factorize(operator, FactorizationPolicy("lu"))
            result = prepared.materialize_inverse()
            inverse_matrix = result.value
            successful = result.successful
            if not square:
                successful = successful & (prepared.determinant_sign() > 0.0)
            measure = jnp.exp(prepared.log_abs_determinant() * (1.0 if square else 0.5))
        measure = eqx.error_if(
            measure,
            jnp.any(~successful | ~jnp.isfinite(measure) | (measure <= 0.0)),
            "Finite-element metric measure must be positive and finite.",
        )
        inverse = (
            inverse_matrix
            if square
            else ein.contract("cqij,cqdj->cqid", inverse_matrix, jacobian)
        )
        inverse_hessian = None
        if self.coordinate_hessians.size:
            mapping_hessian = ein.contract(
                "cqirs,cid->cqdrs" if cellwise else "qirs,cid->cqdrs",
                self.coordinate_hessians,
                coordinates,
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


def _scalar_dg_reference_polynomials(
    element: FiniteElementSpec,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:

    if (
        element.conformity != "L2"
        or element.mapping != "identity"
        or element.representation != "point_value"
        or element.value_shape
        or any(dofs for dimension in element.entity_dofs[:-1] for dofs in dimension)
        or element.entity_dofs[-1] != (tuple(range(element.local_dof_count)),)
    ):
        raise ValueError(
            "scalar_dg_reference_polynomials.configuration must provide represented scalar point-value cell functionals; got Prepared-local L2 execution requires represented scalar point-value cell functionals."
        )
    basis = imported_coordinate_enclosure.source_basis(element, budget=budget)
    if basis is None or len(basis) != element.local_dof_count:
        raise ValueError(
            "scalar_dg_reference_polynomials.configuration must provide its declared owned scalar reference source; got Prepared-local DG execution requires its declared owned scalar reference source."
        )
    return basis


def _coordinate_reference_polynomials(
    element: CellGeometryElement,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:

    if isinstance(element, imported_cell_geometry.RestrictedCellGeometryElement):
        restricted = mapped_measure_owner.mapped_expression_restriction(
            _coordinate_reference_polynomials(element.source_element, budget=budget),
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.matrix),
            np.asarray(element.offset),
            budget=budget,
        )
        if any(
            isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
            for value in restricted
        ):
            raise ValueError(
                "coordinate_reference_polynomials.configuration must have polynomial coordinate restrictions; got Polynomial-algebraic local quadrature cannot certify a rational coordinate restriction."
            )
        return tuple(
            value
            for value in restricted
            if not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
        )
    basis = imported_coordinate_enclosure.source_basis(element, budget=budget)
    if basis is None:
        raise ValueError(
            "coordinate_reference_polynomials.configuration must provide its declared polynomial reference source; got Prepared-local DG geometry requires its declared polynomial reference source."
        )
    return basis


def _scalar_cell_quadrature_degree(
    elements: Sequence[FiniteElementSpec],
    coordinate_element: FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    *,
    budget: CoordinateEnclosureBudget,
) -> int:
    """Rule width for scalar mass/energy products, independent of runtime coefficients.

    algebraic source-basis degree bounds include affine restriction composition;
    nominal root degree or cancellation in today's coordinates cannot understate
    tomorrow's density degree. Nonlinear densities still use explicit user rules.
    """

    geometry = _coordinate_reference_polynomials(coordinate_element, budget=budget)
    candidates = tuple(
        _scalar_dg_reference_polynomials(element, budget=budget)
        if element.conformity == "L2"
        else imported_coordinate_enclosure.source_basis(element, budget=budget)
        for element in elements
    )
    fields_list: list[tuple[Polynomial, ...]] = []
    for basis in candidates:
        if basis is None:
            raise ValueError(
                "scalar_cell_quadrature_degree.configuration must provide declared polynomial field sources; got Mixed scalar DG cell execution requires declared polynomial field sources."
            )
        fields_list.append(basis)
    fields = tuple(fields_list)
    kind, dimension = (
        coordinate_element.cell_kind,
        coordinate_element.topological_dimension,
    )

    def degree(bases: Sequence[tuple[Polynomial, ...]], axes: tuple[int, ...]) -> int:
        return max(
            (
                sum(index[axis] for axis in axes)
                for basis in bases
                for polynomial in basis
                for index in polynomial
            ),
            default=0,
        )

    if kind in ("interval", "triangle", "tetrahedron"):
        field_degree = degree(fields, tuple(range(dimension)))
        geometry_degree = degree((geometry,), tuple(range(dimension)))
        count = field_degree + (dimension * geometry_degree + 1) // 2
    elif kind == "prism":
        count = max(
            degree(fields, axes) + (3 * degree((geometry,), axes) + 1) // 2
            for axes in ((0, 1), (2,))
        )
    else:
        count = max(
            degree(fields, (axis,)) + (dimension * degree((geometry,), (axis,)) + 1) // 2
            for axis in range(dimension)
        )
    count = max(count, 2)
    # The Duffy rule includes one extra tetrahedral axis node.
    return max(0, count - (2 if kind == "tetrahedron" else 1))


@final
class FiniteElementLocalProvider(StrictModule):
    """Adapter from stable FE storage to the prepared-local contract."""

    discretization: FiniteElementDiscretization

    @checked
    def __init__(
        self, discretization: imported_generic.FiniteElementDiscretization, /
    ) -> None:

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
        field_index = discretization.field_index(name)
        dof_map = discretization.dof_maps[field_index]
        elements = discretization.elements[field_index]
        widths = tuple(element.local_dof_count for element in elements)
        value_shapes = tuple(element.value_shape for element in elements)
        if len(set(value_shapes)) != 1:
            raise ValueError(
                "FiniteElementLocalProvider.local_field_binding.configuration must provide a uniform FE value shape; got One local field binding requires a uniform FE value shape."
            )
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

    def _prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[PreparedLocalRegion, ...]:

        discretization = self.discretization
        if domain.kind != "cell":
            raise ValueError(
                "FiniteElementLocalProvider._prepare_local_regions.configuration must select cell integration domains; got The generic FE local provider currently prepares cells."
            )
        if (
            domain.support_id != discretization.cell_domain.support_id
            or domain.entity_set_id != discretization.cell_domain.entity_set_id
        ):
            raise ValueError(
                "FiniteElementLocalProvider._prepare_local_regions.domain must match the prepared cell support; got Finite-element cell domain belongs to another support."
            )
        names = tuple(str(name) for name in field_names)
        bindings = tuple(discretization.local_field_binding(name) for name in names)
        has_dg = False
        for name, binding in zip(names, bindings, strict=True):
            if binding.conformity == "H1":
                continue
            if binding.conformity != "L2":
                raise ValueError(
                    "FiniteElementLocalProvider._prepare_local_regions.configuration must provide H1 or implemented scalar DG fields; got Prepared-local cell execution requires H1 or implemented scalar DG fields."
                )
            has_dg = True
            field_index = discretization.field_index(name)
            dofs = discretization.dof_maps[field_index]
            for element in discretization.elements[field_index]:
                _scalar_dg_reference_polynomials(element, budget=budget)
            routes = np.concatenate(
                [np.asarray(route).reshape(-1) for route in dofs.cell_dofs]
            )
            if (
                dofs.association != "cell"
                or routes.size != dofs.global_dof_count
                or np.unique(routes).size != routes.size
                or any(np.any(np.asarray(signs) != 1) for signs in dofs.orientations)
                or any(
                    not np.array_equal(
                        np.asarray(transform),
                        np.broadcast_to(np.eye(transform.shape[-1]), transform.shape),
                    )
                    for transform in dofs.cell_transforms
                )
            ):
                raise ValueError(
                    "FiniteElementLocalProvider._prepare_local_regions.configuration must provide full unshared identity-oriented cell routes; got Scalar DG local actions require full unshared identity-oriented cell routes."
                )
        mode = "dense" if str(kernel_mode) == "auto" else str(kernel_mode)
        mode = parse(
            mode, FiniteElementKernelMode, "FiniteElementLocalProvider.kernel_mode"
        )
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
            if selected.size == 0:
                continue
            entities = block_cells[selected]
            rows_by_entity = {
                int(entity): row
                for row, entity in enumerate(np.asarray(domain.entity_indices))
            }
            domain_rows = np.asarray(
                [rows_by_entity[int(entity)] for entity in entities], dtype=np.int32
            )
            elements = tuple(
                discretization.elements[discretization.field_index(name)][block_index]
                for name in names
            )
            coordinate_element = require_tabulated_coordinate_element(
                discretization.coordinate_elements[block_index]
            )
            degree = (
                _scalar_cell_quadrature_degree(
                    elements, coordinate_element, budget=budget
                )
                if has_dg
                else max(element.degree for element in elements)
            )
            if (
                discretization.coordinate_measure is not None
                and discretization.coordinate_measure.kind == "spherical-radial"
            ):
                degree += 1
            points, weights = reference_rules_owner.degree_aware_reference_rule(
                block.cell_kind, degree
            )
            references = []
            gathers = []
            for name in names:
                field_index = discretization.field_index(name)
                element = discretization.elements[field_index][block_index]
                basis, gradients = element.tabulate(points)
                hessians = (
                    _tabulation_hessians(element, points)
                    if maximum_derivative_order == 2
                    else None
                )
                references.append(
                    FiniteElementReferenceActions(
                        basis,
                        gradients,
                        basis_hessians=hessians,
                        coefficient_transform=(
                            jax.device_put(
                                np.asarray(
                                    discretization.dof_maps[field_index].cell_transforms[
                                        block_index
                                    ]
                                )[selected]
                            )
                            if element.form_basis is not None
                            else None
                        ),
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
                gathers.append(
                    np.asarray(
                        discretization.dof_maps[field_index].cell_dofs[block_index]
                    )[selected]
                )
            coordinate_basis, coordinate_gradients = coordinate_element.tabulate(points)
            coordinate_hessians = (
                _tabulation_hessians(coordinate_element, points)
                if maximum_derivative_order == 2
                else None
            )
            geometry = FiniteElementGeometryActions(
                discretization.default_runtime.geometry_layout_id,
                domain.kind,
                coordinate_basis,
                coordinate_gradients,
                jax.device_put(
                    np.asarray(discretization.coordinate_dofs[block_index])[selected]
                ),
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
            raise ValueError(
                "FiniteElementLocalProvider._prepare_local_regions.domain must select a supported declared route; got The local domain selects no finite-element cells."
            )
        return _coalesce_local_regions(tuple(regions))

    @checked
    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[PreparedLocalRegion, ...]:
        """Prepare host regions with an explicit input-derived source reservation."""
        if budget is not None:
            return self._prepare_local_regions(
                domain,
                field_names=field_names,
                maximum_derivative_order=maximum_derivative_order,
                kernel_mode=kernel_mode,
                budget=budget,
            )
        fields = tuple(
            self.discretization.elements[self.discretization.field_index(str(name))]
            for name in field_names
        )
        estimates = []
        if any(element.conformity == "L2" for group in fields for element in group):
            for group in fields:
                for element in group:
                    estimate = source_basis_estimate(element)
                    if estimate is not None:
                        # The admission loop and per-block quadrature sizing
                        # each prepare this source; request caching only lowers it.
                        estimates.extend((estimate, estimate))
            for element in self.discretization.coordinate_elements:
                estimate = source_basis_estimate(
                    require_tabulated_coordinate_element(element)
                )
                if estimate is not None:
                    estimates.append(estimate)
        estimate = coordinate_estimate_bank(tuple(estimates))
        with HostBatchTicket(
            estimate.work_upper, bytes_upper=estimate.bytes_upper
        ) as ticket:
            ledger = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=estimate.work_upper,
                maximum_memory_bytes=estimate.bytes_upper,
            )
            with ledger.activate():
                return self._prepare_local_regions(
                    domain,
                    field_names=field_names,
                    maximum_derivative_order=maximum_derivative_order,
                    kernel_mode=kernel_mode,
                    budget=ledger,
                )

    def validate_local_runtime(self, runtime: object, /) -> None:

        discretization = self.discretization
        if not isinstance(runtime, imported_generic.FiniteElementRuntimeData):
            raise TypeError(
                "FiniteElementGeometryActions.realize.runtime must be FiniteElementRuntimeData; got runtime must be FiniteElementRuntimeData."
            )
        if (
            runtime.topology_id != discretization.mesh.topology_id
            or runtime.geometry_layout_id
            != discretization.default_runtime.geometry_layout_id
            or runtime.coordinates.shape
            != discretization.default_runtime.coordinates.shape
        ):
            raise ValueError(
                "FiniteElementLocalProvider.validate_local_runtime.runtime must match the prepared layout; got Finite-element runtime does not match the prepared layout."
            )
        measure = discretization.coordinate_measure
        if measure is not None and runtime.coordinate_frame != measure.frame:
            raise ValueError(
                "FiniteElementLocalProvider.validate_local_runtime.configuration must match the coordinate measure; got Runtime coordinate frame must match the coordinate measure."
            )


__all__ = [
    "FiniteElementGeometryActions",
    "FiniteElementLocalProvider",
    "FiniteElementReferenceActions",
]


def _finite_element_region_parts(
    region: PreparedLocalRegion,
) -> tuple[FiniteElementGeometryActions, tuple[FiniteElementReferenceActions, ...]]:
    match region.geometry_actions:
        case FiniteElementGeometryActions() as geometry:
            pass
        case _:
            raise TypeError(
                "finite_element.region.geometry must retain FE coordinate actions; got another geometry owner"
            )
    references = []
    for reference in region.reference_actions:
        match reference:
            case FiniteElementReferenceActions():
                references.append(reference)
            case _:
                raise TypeError(
                    "finite_element.region.references must retain FE field actions; got another reference owner"
                )
    return geometry, tuple(references)


def _local_region_numerical_signature(region: PreparedLocalRegion) -> str:
    geometry, references = _finite_element_region_parts(region)
    return canonical_fingerprint(
        {
            "fields": region.field_names,
            "cell_kind": region.cell_kind,
            "domain": (
                region.domain.kind,
                region.domain.support_id,
                region.domain.entity_set_id,
                region.domain.selection_id,
            ),
            "references": tuple(
                (
                    array_tree_fingerprint(
                        (
                            reference.basis_values,
                            reference.basis_gradients,
                            reference.basis_hessians,
                        )
                    ),
                    reference.maximum_derivative_order,
                    reference.kernel_modes,
                    reference.coefficient_transform.shape[1:],
                    reference.coefficient_transform.dtype.str,
                )
                for reference in references
            ),
            "geometry": (
                geometry.runtime_layout_id,
                geometry.coordinate_basis.shape,
                geometry.coordinate_gradients.shape,
                geometry.coordinate_hessians.shape,
                geometry.coordinate_basis.dtype.str,
                array_tree_fingerprint(geometry.reference_weights),
                None
                if geometry.coordinate_measure is None
                else geometry.coordinate_measure.plan_id,
            ),
        }
    )


def _merge_local_region_run(
    regions: tuple[PreparedLocalRegion, ...],
) -> PreparedLocalRegion:
    """Merge a consecutive homogeneous run, retaining its ordered cell routes."""
    if len(regions) == 1:
        return regions[0]
    parts = tuple(_finite_element_region_parts(region) for region in regions)
    first, (geometry, references) = regions[0], parts[0]
    merged_references = tuple(
        FiniteElementReferenceActions(
            reference.basis_values,
            reference.basis_gradients,
            basis_hessians=reference.basis_hessians,
            coefficient_transform=(
                np.concatenate(
                    tuple(
                        np.asarray(refs[index].coefficient_transform) for _, refs in parts
                    )
                )
                if reference.coefficient_transform.size
                else None
            ),
            maximum_derivative_order=reference.maximum_derivative_order,
            kernel_modes=reference.kernel_modes,
            action_id=canonical_fingerprint(
                tuple(refs[index].action_id for _, refs in parts)
            ),
        )
        for index, reference in enumerate(references)
    )

    def cell_tables(tables: tuple[Array, ...]) -> np.ndarray:
        return np.concatenate(
            tuple(
                np.broadcast_to(np.asarray(table), (owner.entity_count,) + table.shape)
                for table, (owner, _) in zip(tables, parts, strict=True)
            )
        )

    merged_geometry = FiniteElementGeometryActions(
        geometry.runtime_layout_id,
        geometry.domain_kind,
        cell_tables(tuple(owner.coordinate_basis for owner, _ in parts)),
        cell_tables(tuple(owner.coordinate_gradients for owner, _ in parts)),
        np.concatenate(tuple(np.asarray(owner.coordinate_gathers) for owner, _ in parts)),
        geometry.reference_weights,
        coordinate_hessians=cell_tables(
            tuple(owner.coordinate_hessians for owner, _ in parts)
        )
        if geometry.coordinate_hessians.size
        else None,
        coordinate_measure=geometry.coordinate_measure,
    )
    domain = IntegrationDomain(
        first.domain.kind,
        np.concatenate(tuple(np.asarray(region.entity_indices) for region in regions)),
        first.domain.support_id,
        first.domain.entity_set_id,
        owner_cells=np.concatenate(
            tuple(np.asarray(region.owner_cells) for region in regions)
        ),
        neighbor_cells=np.concatenate(
            tuple(np.asarray(region.neighbor_cells) for region in regions)
        ),
        owner_local_entities=np.concatenate(
            tuple(np.asarray(region.owner_local_entities) for region in regions)
        ),
        neighbor_local_entities=np.concatenate(
            tuple(np.asarray(region.neighbor_local_entities) for region in regions)
        ),
        selection_id=first.domain.selection_id,
    )
    return PreparedLocalRegion(
        domain,
        first.field_names,
        tuple(
            np.concatenate(
                tuple(np.asarray(region.field_gathers[index]) for region in regions)
            )
            for index in range(len(first.field_names))
        ),
        merged_references,
        merged_geometry,
        block_name=canonical_fingerprint(tuple(region.block_name for region in regions)),
        cell_kind=first.cell_kind,
    )


def _coalesce_local_regions(
    regions: tuple[PreparedLocalRegion, ...],
) -> tuple[PreparedLocalRegion, ...]:
    """Bucket consecutive numerical signatures without reordering cell lanes."""
    runs: list[list[PreparedLocalRegion]] = []
    previous: str | None = None
    for region in regions:
        signature = _local_region_numerical_signature(region)
        if signature != previous:
            runs.append([])
            previous = signature
        runs[-1].append(region)
    return tuple(_merge_local_region_run(tuple(run)) for run in runs)
