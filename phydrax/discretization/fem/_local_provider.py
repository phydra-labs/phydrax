#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
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


if TYPE_CHECKING:
    from .._cell_geometry import (
        BarycentricCellGeometryElement,
        CellGeometryElement,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from .._coordinate_enclosure import Polynomial
    from ._generic import FiniteElementDiscretization
    from ._reference import FiniteElementSpec


def _require_tabulated_coordinate_element(
    element: CellGeometryElement,
) -> (
    FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement
):
    from .._cell_geometry import (
        BarycentricCellGeometryElement,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from ._reference import FiniteElementSpec

    if not isinstance(
        element,
        (
            FiniteElementSpec,
            BarycentricCellGeometryElement,
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
            SplineCellGeometryElement,
            LayerColumnCellGeometryElement,
        ),
    ):
        raise TypeError("Coordinate element must support scalar coordinate tabulation.")
    return element


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
            or basis.ndim != 2
            or gradients.ndim != 3
            or gradients.shape[:2] != basis.shape
            or (
                hessians.size
                and hessians.shape != gradients.shape + (gradients.shape[-1],)
            )
            or gathers.ndim != 2
            or gathers.shape[1] != basis.shape[1]
            or weights.shape != (basis.shape[0],)
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
            }
        )

    def realize(self, runtime: object, /) -> LocalMetricResult:
        from ._generic import FiniteElementRuntimeData

        if not isinstance(runtime, FiniteElementRuntimeData):
            raise TypeError("runtime must be FiniteElementRuntimeData.")
        coordinates = jnp.asarray(runtime.coordinates)[self.coordinate_gathers]
        points = ein.contract("qi,cid->cqd", self.coordinate_basis, coordinates)
        jacobian = ein.contract("qir,cid->cqdr", self.coordinate_gradients, coordinates)
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
                "qirs,cid->cqdrs", self.coordinate_hessians, coordinates
            )
            inverse_hessian = -ein.contract(
                "cqrd,cqdst,cqsa,cqtb->cqrab",
                inverse,
                mapping_hessian,
                inverse,
                inverse,
            )
        return LocalMetricResult(
            points,
            measure * self.reference_weights[None, :],
            jacobian,
            inverse,
            inverse_hessian=inverse_hessian,
        )


def _scalar_dg_reference_polynomials(
    element: FiniteElementSpec,
) -> tuple[Polynomial, ...]:
    from .._coordinate_enclosure import source_basis

    if (
        element.conformity != "L2"
        or element.mapping != "identity"
        or element.representation != "point_value"
        or element.value_shape
        or any(dofs for dimension in element.entity_dofs[:-1] for dofs in dimension)
        or element.entity_dofs[-1] != (tuple(range(element.local_dof_count)),)
    ):
        raise ValueError(
            "Prepared-local L2 execution requires actual scalar point-value cell functionals."
        )
    basis = source_basis(element)
    if basis is None or len(basis) != element.local_dof_count:
        raise ValueError(
            "Prepared-local DG execution requires its canonical owned scalar reference source."
        )
    return basis


def _coordinate_reference_polynomials(
    element: CellGeometryElement,
) -> tuple[Polynomial, ...]:
    from .._cell_geometry import RestrictedCellGeometryElement
    from .._cell_geometry_transfer import _mapped_expression_restriction
    from .._coordinate_enclosure import RationalPolynomial, source_basis

    if isinstance(element, RestrictedCellGeometryElement):
        restricted = _mapped_expression_restriction(
            _coordinate_reference_polynomials(element.source_element),
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.matrix),
            np.asarray(element.offset),
        )
        if any(isinstance(value, RationalPolynomial) for value in restricted):
            raise ValueError(
                "Polynomial-exact local quadrature cannot certify a genuinely rational coordinate restriction."
            )
        return tuple(
            value for value in restricted if not isinstance(value, RationalPolynomial)
        )
    basis = source_basis(element)
    if basis is None:
        raise ValueError(
            "Prepared-local DG geometry requires its authoritative polynomial reference source."
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
) -> int:
    """Rule width for scalar mass/energy products, independent of runtime coefficients.

    Exact source-basis degree bounds include affine restriction composition;
    nominal root degree or cancellation in today's coordinates cannot understate
    tomorrow's density degree. Nonlinear densities still use explicit user rules.
    """
    from .._coordinate_enclosure import source_basis

    geometry = _coordinate_reference_polynomials(coordinate_element)
    candidates = tuple(
        _scalar_dg_reference_polynomials(element)
        if element.conformity == "L2"
        else source_basis(element)
        for element in elements
    )
    fields_list: list[tuple[Polynomial, ...]] = []
    for basis in candidates:
        if basis is None:
            raise ValueError(
                "Mixed scalar DG cell execution requires canonical polynomial field sources."
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
    # The owning Duffy rule includes one extra tetrahedral axis node.
    return max(0, count - (2 if kind == "tetrahedron" else 1))


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
        bindings = tuple(discretization.local_field_binding(name) for name in names)
        has_dg = False
        for name, binding in zip(names, bindings, strict=True):
            if binding.conformity == "H1":
                continue
            if binding.conformity != "L2":
                raise ValueError(
                    "Prepared-local cell execution requires H1 or implemented scalar DG fields."
                )
            has_dg = True
            field_index = discretization._field_index(name)
            dofs = discretization.dof_maps[field_index]
            for element in discretization.elements[field_index]:
                _scalar_dg_reference_polynomials(element)
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
                    "Scalar DG local actions require complete unshared identity-oriented cell routes."
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
                discretization.elements[discretization._field_index(name)][block_index]
                for name in names
            )
            coordinate_element = _require_tabulated_coordinate_element(
                discretization.coordinate_elements[block_index]
            )
            degree = (
                _scalar_cell_quadrature_degree(elements, coordinate_element)
                if has_dg
                else max(element.degree for element in elements)
            )
            points, weights = _degree_aware_reference_rule(block.cell_kind, degree)
            references = []
            gathers = []
            for name in names:
                field_index = discretization._field_index(name)
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
                            discretization.dof_maps[field_index].cell_transforms[
                                block_index
                            ][selected]
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
                    discretization.dof_maps[field_index].cell_dofs[block_index][selected]
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
                discretization.coordinate_dofs[block_index][selected],
                weights,
                coordinate_hessians=coordinate_hessians,
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


__all__ = [
    "FiniteElementGeometryActions",
    "FiniteElementLocalProvider",
    "FiniteElementReferenceActions",
]
