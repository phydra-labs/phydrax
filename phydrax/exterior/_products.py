#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Cell products obtained by reconstruction and oriented chain integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product as cartesian_product
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization._cell_complex import simplicial_cell_geometry
from ..discretization._cell_de_rham import AbstractCellDeRhamComplex
from ..discretization._cubical_whitney import CubicalSplineWhitneyKernel
from ..discretization._structured_cochain import StructuredCochainBridge
from ..discretization.fem._simplicial_whitney_chains import SimplicialWhitneyKernel
from ..linalg._compound import compound_matrix
from ..topology._advanced import CellDiagonalApproximation
from ..typing import parse
from ._algebra import interior, wedge
from ._chains import PreparedChainQuery
from ._complex import ComplexBoundary, DiscreteForm
from ._form_type import FiberProduct, FormTwist, FormType


type LieDerivativeMethod = Literal["cartan", "semi-lagrangian"]
type ProductVectorField = ArrayLike | Callable[[Array], Array]
type WhitneyKernel = CubicalSplineWhitneyKernel | SimplicialWhitneyKernel


def _quadrature(degree: int, order: int, simplex: bool, /) -> tuple[Array, Array]:
    if degree == 0:
        return jnp.zeros((1, 0), dtype=jnp.float64), jnp.ones((1,), dtype=jnp.float64)
    points, weights = np.polynomial.legendre.leggauss(order)
    points = (points + 1) / 2
    weights = weights / 2
    indices = np.asarray(tuple(cartesian_product(range(order), repeat=degree)))
    nodes = points[indices]
    mass = np.prod(weights[indices], axis=1)
    if simplex:
        # Duffy's cube-to-simplex map includes its Jacobian in the weights.
        remaining = np.ones((nodes.shape[0],), dtype=np.float64)
        mapped = np.empty_like(nodes)
        for axis in range(degree):
            mapped[:, axis] = remaining * nodes[:, axis]
            mass *= (1 - nodes[:, axis]) ** (degree - axis - 1)
            remaining *= 1 - nodes[:, axis]
        nodes = mapped
    return jnp.asarray(nodes), jnp.asarray(mass)


def _cubical_vertices(kernel: CubicalSplineWhitneyKernel, degree: int, /) -> Array:
    bridge = kernel.bridge
    blocks = []
    for axes, shape in zip(
        bridge.orientations[degree], bridge.orientation_shapes[degree], strict=True
    ):
        lattice = jnp.asarray(
            np.indices(shape).reshape((len(shape), -1)).T, dtype=jnp.int32
        )
        corners = []
        for bits in cartesian_product((0, 1), repeat=degree):
            coordinates = []
            for axis, grid_axis in enumerate(bridge.grid.structured_axes):
                index = lattice[:, axis]
                value = grid_axis.point_coordinates[index]
                if axis in axes:
                    value = (
                        value + bits[axes.index(axis)] * grid_axis.interval_widths[index]
                    )
                coordinates.append(value)
            corners.append(jnp.stack(coordinates, axis=-1))
        blocks.append(jnp.stack(corners, axis=1))
    return jnp.concatenate(blocks, axis=0)


def _simplex_vertices(
    kernel: SimplicialWhitneyKernel, degree: int, /
) -> tuple[Array, Array]:
    topology = (
        kernel.complex.topology
        if isinstance(kernel.complex, AbstractCellDeRhamComplex)
        else kernel.complex
    )
    rows, orientations = simplicial_cell_geometry(topology)
    return kernel.locator.coordinates[jnp.asarray(rows[degree])], jnp.asarray(
        orientations[degree]
    )


def _chain_map(vertices: Array, coordinates: Array, simplex: bool, /) -> Array:
    if simplex:
        return vertices[0] + coordinates @ (vertices[1:] - vertices[0])
    degree = coordinates.shape[0]
    bits = jnp.asarray(
        tuple(cartesian_product((0, 1), repeat=degree)), dtype=jnp.int32
    ).reshape((2**degree, degree))
    weights = jnp.prod(
        jnp.where(bits == 1, coordinates[None, :], 1 - coordinates[None, :]), axis=1
    )
    return weights @ vertices


def _admitted_gather(query: PreparedChainQuery, values: Array, /) -> Array:
    result = query.gather(values)
    return eqx.error_if(
        result,
        jnp.any(~query.successful | query.overflow),
        "Whitney chain query left its domain or exceeded traversal capacity.",
    )


def _gather_fibers(query: PreparedChainQuery, values: Array, /) -> Array:
    if values.ndim == 1:
        return _admitted_gather(query, values)
    fibers = values.reshape((values.shape[0], -1))

    def gather_column(column: Array, /) -> Array:
        return _admitted_gather(query, column)

    result = jax.vmap(gather_column, in_axes=1, out_axes=-1)(fibers)
    return result.reshape((query.indices.shape[0], *query.value_shape, *values.shape[1:]))


def _validate_product_kernel(
    complex: AbstractCellDeRhamComplex, kernel: WhitneyKernel, /
) -> None:
    if kernel.dof_counts != complex.cell_counts:
        raise ValueError(
            "Whitney kernel degree coordinates do not match the realization."
        )
    if isinstance(kernel, CubicalSplineWhitneyKernel):
        kernel_realization = (
            kernel.bridge.realization_id
            if isinstance(complex, StructuredCochainBridge)
            else kernel.bridge.cochain.realization_id
        )
        if kernel_realization != complex.realization_id:
            raise ValueError("Whitney kernel belongs to a different realization.")
        if kernel.shape_order != 1:
            raise ValueError(
                "Cell products require lowest-order cubical Whitney reconstruction."
            )
    else:
        topology = (
            kernel.complex.topology
            if isinstance(kernel.complex, AbstractCellDeRhamComplex)
            else kernel.complex
        )
        if topology.topology_id != complex.topology.topology_id:
            raise ValueError("Whitney kernel belongs to a different topology.")


def _admit_whitney_product(
    complex: AbstractCellDeRhamComplex,
    kernel: WhitneyKernel,
    quadrature_order: int,
    boundary: ComplexBoundary,
    /,
) -> ComplexBoundary:
    if not isinstance(complex, AbstractCellDeRhamComplex):
        raise TypeError("Whitney products require a cell de Rham realization.")
    if not isinstance(kernel, (CubicalSplineWhitneyKernel, SimplicialWhitneyKernel)):
        raise TypeError(
            "Whitney products require an owning cubical or simplicial kernel."
        )
    if (
        not isinstance(quadrature_order, int)
        or isinstance(quadrature_order, bool)
        or quadrature_order < 1
    ):
        raise ValueError("quadrature_order must be a positive integer.")
    boundary = parse(boundary, ComplexBoundary, "boundary")
    _validate_product_kernel(complex, kernel)
    return boundary


def _prepare_product_geometry(
    complex: AbstractCellDeRhamComplex, kernel: WhitneyKernel, /
) -> tuple[bool, tuple[Array, ...], tuple[Array, ...]]:
    if isinstance(kernel, CubicalSplineWhitneyKernel):
        simplex = False
        vertices = tuple(
            _cubical_vertices(kernel, degree) for degree in range(complex.dimension + 1)
        )
        orientations = tuple(
            jnp.ones((count,), dtype=jnp.float64) for count in kernel.dof_counts
        )
    else:
        simplex = True
        geometry = tuple(
            _simplex_vertices(kernel, degree) for degree in range(complex.dimension + 1)
        )
        vertices = tuple(item[0] for item in geometry)
        orientations = tuple(item[1] for item in geometry)
    if any(value.shape[-1] != complex.dimension for value in vertices):
        raise ValueError("Cell products require intrinsic Cartesian coordinates.")
    return simplex, vertices, orientations


def _chain_geometry(
    cells: Array, nodes: Array, degree: int, simplex: bool, /
) -> tuple[Array, Array]:
    def at_cell(cell: Array, /) -> tuple[Array, Array]:
        def at_node(node: Array, /) -> Array:
            return _chain_map(cell, node, simplex)

        return jax.vmap(at_node)(nodes), jax.vmap(jax.jacfwd(at_node))(nodes)

    points, jacobians = jax.vmap(at_cell)(cells)
    blades = compound_matrix(jacobians, degree)[..., 0]
    return points, blades


def _prepare_reconstruction_queries(
    kernel: WhitneyKernel, points: tuple[Array, ...], dimension: int, /
) -> tuple[tuple[PreparedChainQuery, ...], ...]:
    # Wedges need source degrees <= target; contraction needs target + 1.
    return tuple(
        tuple(
            kernel.evaluate(point.reshape((-1, dimension)), source_degree)
            for source_degree in range(min(target_degree + 1, dimension) + 1)
        )
        for target_degree, point in enumerate(points)
    )


@final
class WhitneyProductPlan(StrictModule):
    """Prepared geometry for genuine Whitney products and pullback chains.

    ``quadrature_order`` is the number of Gauss nodes per reference direction.
    Cubical cells use tensor Gauss quadrature; simplices use the Duffy map.
    Semi-Lagrangian degrees zero and one instead use exact point/segment
    integration, including every crossed source cell. The flow approximation
    backtracks vertices by ``x - step * X(x)`` and integrates the resulting
    affine simplex or multilinear cubical chain, not scalar samples.

    Pass the plan as a dynamic PyTree argument to ``eqx.filter_jit`` when
    compiling repeated products: prepared reconstruction coefficients, query
    status, geometry blades, and masks remain numeric inputs and support
    differentiation. Capturing a plan in a ``jax.jit`` closure instead fixes
    those numeric leaves to that prepared snapshot.
    """

    complex: AbstractCellDeRhamComplex
    kernel: WhitneyKernel
    vertices: tuple[Array, ...]
    orientations: tuple[Array, ...]
    nodes: tuple[Array, ...]
    weights: tuple[Array, ...]
    masks: tuple[Array, ...]
    points: tuple[Array, ...]
    blades: tuple[Array, ...]
    reconstruction_queries: tuple[tuple[PreparedChainQuery, ...], ...]
    vector_queries: tuple[PreparedChainQuery, ...]
    vertex_vector_queries: tuple[PreparedChainQuery, ...]
    simplex: bool = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: AbstractCellDeRhamComplex,
        kernel: WhitneyKernel,
        /,
        *,
        quadrature_order: int = 4,
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        boundary = _admit_whitney_product(complex, kernel, quadrature_order, boundary)
        simplex, vertices, orientations = _prepare_product_geometry(complex, kernel)
        quadrature = tuple(
            _quadrature(degree, quadrature_order, simplex)
            for degree in range(complex.dimension + 1)
        )
        geometry = tuple(
            _chain_geometry(cell, rule[0], degree, simplex)
            for degree, (cell, rule) in enumerate(zip(vertices, quadrature, strict=True))
        )
        points = tuple(item[0] for item in geometry)
        reconstruction_queries = _prepare_reconstruction_queries(
            kernel, points, complex.dimension
        )
        vector_queries = tuple(
            kernel.integrate_points(point.reshape((-1, complex.dimension)))
            for point in points
        )
        vertex_vector_queries = tuple(
            kernel.integrate_points(cell.reshape((-1, complex.dimension)))
            for cell in vertices
        )
        self.complex = complex
        self.kernel = kernel
        self.vertices = vertices
        self.orientations = orientations
        self.nodes = tuple(item[0] for item in quadrature)
        self.weights = tuple(item[1] for item in quadrature)
        self.points = points
        self.blades = tuple(item[1] for item in geometry)
        self.reconstruction_queries = reconstruction_queries
        self.vector_queries = vector_queries
        self.vertex_vector_queries = vertex_vector_queries
        self.masks = tuple(
            entity.active_mask
            & (~mask if boundary == "relative" else jnp.ones(mask.shape, dtype=jnp.bool_))
            for entity, mask in zip(
                complex.topology.entity_sets, complex.boundary_masks, strict=True
            )
        )
        self.simplex = simplex
        self.quadrature_order = quadrature_order
        self.boundary = boundary
        self.plan_id = canonical_fingerprint(
            {
                "kind": "whitney-products",
                "realization": complex.realization_id,
                "kernel": kernel.kernel_id,
                "quadrature_order": quadrature_order,
                "boundary": boundary,
            }
        )

    def _mask(self, values: Array, degree: int, /) -> Array:
        if (
            values.ndim == 0
            or degree < 0
            or degree > self.complex.dimension
            or values.shape[0] != self.kernel.dof_counts[degree]
        ):
            raise ValueError("Product coordinates do not match their declared degree.")
        return jnp.where(
            self.masks[degree].reshape((-1, *(1,) * (values.ndim - 1))), values, 0
        )

    def _geometry(
        self, degree: int, /, *, vertices: Array | None = None
    ) -> tuple[Array, Array]:
        if vertices is None:
            return self.points[degree], self.blades[degree]
        return _chain_geometry(vertices, self.nodes[degree], degree, self.simplex)

    def _evaluate(
        self,
        values: Array,
        points: Array,
        degree: int,
        /,
        *,
        query: PreparedChainQuery | None = None,
    ) -> Array:
        if query is None:
            query = self.kernel.evaluate(
                points.reshape((-1, self.complex.dimension)), degree
            )
        masked = self._mask(values, degree)
        if values.ndim == 1:
            result = _admitted_gather(query, masked)
            if result.ndim == 1:
                result = result[:, None]
            return result.reshape((*points.shape[:-1], result.shape[-1]))
        result = _gather_fibers(query, masked)
        if len(query.value_shape) == 0:
            result = result[:, None]
        return result.reshape(
            (
                *points.shape[:-1],
                FormType(self.complex.dimension, degree).component_count,
                *values.shape[1:],
            )
        )

    def _vector(
        self,
        X: ProductVectorField,
        points: Array,
        /,
        *,
        query: PreparedChainQuery | None = None,
    ) -> Array:
        flat = points.reshape((-1, self.complex.dimension))
        if callable(X):
            result = jnp.asarray(jax.vmap(X)(flat))
        else:
            vector = jnp.asarray(X)
            if vector.shape == (self.complex.dimension,):
                result = jnp.broadcast_to(vector, flat.shape)
            elif vector.shape == (self.kernel.dof_counts[0], self.complex.dimension):
                if query is None:
                    query = self.kernel.integrate_points(flat)
                result = _gather_fibers(query, vector)
            else:
                raise ValueError(
                    "X must be a Cartesian vector, a nodal vector field, or a pointwise callable."
                )
        if result.shape != flat.shape:
            raise ValueError("X must return one intrinsic Cartesian vector per point.")
        return result.reshape(points.shape)

    def _integrate(self, values: Array, blades: Array, degree: int, /) -> Array:
        extra = (1,) * (values.ndim - 3)
        integrand = jnp.sum(values * blades.reshape((*blades.shape, *extra)), axis=2)
        weights = self.weights[degree].reshape((1, -1, *extra))
        result = jnp.sum(integrand * weights, axis=1)
        result = result * self.orientations[degree].reshape((-1, *extra))
        return self._mask(result, degree)

    def wedge(
        self,
        a: Array,
        b: Array,
        left_degree: int,
        right_degree: int,
        /,
        *,
        product: FiberProduct = "scalar",
        left_twist: FormTwist = "untwisted",
        right_twist: FormTwist = "untwisted",
    ) -> Array:
        left_type = FormType(
            self.complex.dimension, left_degree, twist=left_twist, fiber_shape=a.shape[1:]
        )
        right_type = FormType(
            self.complex.dimension,
            right_degree,
            twist=right_twist,
            fiber_shape=b.shape[1:],
        )
        result_type = left_type.wedge_type(right_type, product=product)
        if (
            left_type.twist != self.complex.primal_twist
            or right_type.twist != self.complex.primal_twist
            or result_type.twist != self.complex.primal_twist
        ):
            raise ValueError(
                "Whitney wedge requires primal input and output placement; dual geometric reconstruction is not supplied."
            )
        points, blades = self._geometry(result_type.degree)
        values = wedge(
            self._evaluate(
                a,
                points,
                left_degree,
                query=self.reconstruction_queries[result_type.degree][left_degree],
            ),
            self._evaluate(
                b,
                points,
                right_degree,
                query=self.reconstruction_queries[result_type.degree][right_degree],
            ),
            left_type,
            right_type,
            product=product,
        )
        return self._integrate(values, blades, result_type.degree)

    def interior(self, X: ProductVectorField, a: Array, degree: int, /) -> Array:
        form_type = FormType(self.complex.dimension, degree, fiber_shape=a.shape[1:])
        result_type = form_type.interior_type()
        points, blades = self._geometry(result_type.degree)
        values = interior(
            self._vector(X, points, query=self.vector_queries[result_type.degree]),
            self._evaluate(
                a,
                points,
                degree,
                query=self.reconstruction_queries[result_type.degree][degree],
            ),
            form_type,
        )
        return self._integrate(values, blades, result_type.degree)

    def _derivative(self, values: Array, degree: int, /) -> Array:
        if values.ndim == 1:
            return self.complex.exterior_derivative(
                degree, values, boundary=self.boundary
            )

        def differentiate(column: Array, /) -> Array:
            return self.complex.exterior_derivative(
                degree, column, boundary=self.boundary
            )

        result = jax.vmap(differentiate, in_axes=1, out_axes=1)(
            values.reshape((values.shape[0], -1))
        )
        return result.reshape((self.kernel.dof_counts[degree + 1], *values.shape[1:]))

    def lie(
        self,
        X: ProductVectorField,
        a: Array,
        degree: int,
        /,
        *,
        method: LieDerivativeMethod = "cartan",
        step: ArrayLike | None = None,
    ) -> Array:
        method = parse(method, LieDerivativeMethod, "method")
        a = self._mask(a, degree)
        match method:
            case "cartan":
                result = jnp.zeros_like(a)
                if degree > 0:
                    contraction = self.interior(X, a, degree)
                    result = result + self._derivative(contraction, degree - 1)
                if degree < self.complex.dimension:
                    derivative = self._derivative(a, degree)
                    result = result + self.interior(X, derivative, degree + 1)
                return self._mask(result, degree)
            case "semi-lagrangian":
                if step is None:
                    raise ValueError(
                        "Semi-Lagrangian differentiation requires a nonzero step."
                    )
                dt = jnp.asarray(step)
                if dt.shape != () or not jnp.issubdtype(dt.dtype, jnp.floating):
                    raise ValueError("step must be a real floating scalar.")
                dt = eqx.error_if(
                    dt, ~jnp.isfinite(dt) | (dt == 0), "step must be finite and nonzero."
                )
                vertices = self.vertices[degree]
                backtracked = vertices - dt * self._vector(
                    X, vertices, query=self.vertex_vector_queries[degree]
                )
                if degree == 0:
                    pulled = _gather_fibers(
                        self.kernel.integrate_points(backtracked[:, 0]), a
                    )
                elif degree == 1:
                    query = self.kernel.integrate_segments(
                        backtracked[:, 0], backtracked[:, 1]
                    )
                    pulled = _gather_fibers(query, a) * self.orientations[degree].reshape(
                        (-1, *(1,) * (a.ndim - 1))
                    )
                else:
                    points, blades = self._geometry(degree, vertices=backtracked)
                    pulled = self._integrate(
                        self._evaluate(a, points, degree), blades, degree
                    )
                # Backward pullback differentiates to minus the Lie derivative.
                return self._mask((a - pulled) / dt, degree)


def _product_plan(
    complex: WhitneyProductPlan | StructuredCochainBridge, /
) -> WhitneyProductPlan:
    if isinstance(complex, WhitneyProductPlan):
        return complex
    if isinstance(complex, StructuredCochainBridge):
        return WhitneyProductPlan(complex, CubicalSplineWhitneyKernel(complex, 1))
    raise TypeError(
        "Numerical products require WhitneyProductPlan or StructuredCochainBridge with explicit chain geometry."
    )


def _validate_form(complex: AbstractCellDeRhamComplex, a: DiscreteForm, /) -> None:
    if not isinstance(a, DiscreteForm):
        raise TypeError("Cell products require DiscreteForm operands.")
    if a.realization_id != complex.realization_id:
        raise ValueError("Discrete form belongs to a different realization.")
    if (
        a.form_type.dimension != complex.dimension
        or a.form_type.ambient_dimension != complex.dimension
        or a.form_type.fiber_shape
    ):
        raise ValueError("Cell products require intrinsic scalar-coefficient forms.")
    if a.form_type.twist != complex.primal_twist:
        raise ValueError(
            "Cell products require primal operands; dual geometric reconstruction is not supplied."
        )
    if a.values.shape != (complex.cell_counts[a.form_type.degree],):
        raise ValueError("Discrete form coordinates do not match the declared degree.")


def cochain_cup_product(
    complex: AbstractCellDeRhamComplex,
    a: DiscreteForm,
    b: DiscreteForm,
    /,
    *,
    diagonal: CellDiagonalApproximation | Sequence[CellDiagonalApproximation],
) -> DiscreteForm:
    """Apply a declared cellular diagonal over real or complex coefficients."""
    _validate_form(complex, a)
    _validate_form(complex, b)
    result_type = a.form_type.wedge_type(b.form_type)
    if result_type.twist != complex.primal_twist:
        raise ValueError(
            "Cup output requires a dual-storage diagonal, which is not supplied."
        )
    terms = (
        (diagonal,)
        if isinstance(diagonal, CellDiagonalApproximation)
        else tuple(diagonal)
    )
    if any(not isinstance(term, CellDiagonalApproximation) for term in terms):
        raise TypeError("diagonal must contain CellDiagonalApproximation terms.")
    if any(term.topology_id != complex.topology.topology_id for term in terms):
        raise ValueError("Diagonal belongs to a different topology.")
    matching = tuple(
        term
        for term in terms
        if term.left_degree == a.form_type.degree
        and term.right_degree == b.form_type.degree
    )
    if not matching:
        raise ValueError("Diagonal does not contain the requested bidegree.")
    dtype = jnp.result_type(a.values, b.values)
    result = jnp.zeros((complex.cell_counts[result_type.degree],), dtype=dtype)
    for term in matching:
        values = (
            term.coefficients.astype(dtype)
            * a.values[term.left_cells]
            * b.values[term.right_cells]
        )
        result = result.at[term.source_cells].add(values)
    return DiscreteForm(complex.realization_id, result_type, result)


def whitney_wedge(
    complex: WhitneyProductPlan | StructuredCochainBridge,
    a: DiscreteForm,
    b: DiscreteForm,
    /,
) -> DiscreteForm:
    """Integrate the wedge of both Whitney reconstructions over each cell."""
    plan = _product_plan(complex)
    _validate_form(plan.complex, a)
    _validate_form(plan.complex, b)
    result_type = a.form_type.wedge_type(b.form_type)
    values = plan.wedge(
        a.values,
        b.values,
        a.form_type.degree,
        b.form_type.degree,
        left_twist=a.form_type.twist,
        right_twist=b.form_type.twist,
    )
    return DiscreteForm(plan.complex.realization_id, result_type, values)


def interior_product(
    complex: WhitneyProductPlan | StructuredCochainBridge,
    X: ProductVectorField,
    a: DiscreteForm,
    /,
) -> DiscreteForm:
    """Integrate Cartesian contraction of a Whitney reconstruction."""
    plan = _product_plan(complex)
    _validate_form(plan.complex, a)
    result_type = a.form_type.interior_type()
    return DiscreteForm(
        plan.complex.realization_id,
        result_type,
        plan.interior(X, a.values, a.form_type.degree),
    )


def lie_derivative(
    complex: WhitneyProductPlan | StructuredCochainBridge,
    X: ProductVectorField,
    a: DiscreteForm,
    /,
    *,
    method: LieDerivativeMethod = "cartan",
    step: ArrayLike | None = None,
) -> DiscreteForm:
    """Cartan's discrete formula or a backtracked-chain pullback difference."""
    plan = _product_plan(complex)
    _validate_form(plan.complex, a)
    return DiscreteForm(
        plan.complex.realization_id,
        a.form_type,
        plan.lie(X, a.values, a.form_type.degree, method=method, step=step),
    )
