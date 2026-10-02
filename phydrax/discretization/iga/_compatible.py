#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import prod
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import NumericRevision, SemanticProvenance
from ..._interpolation._bspline_grid import BSplineGrid
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...diagnostics import Diagnostic
from ...ein import contract
from ...exterior import FormTwist
from ...exterior._basis import exterior_indices
from ...exterior._complex import ComplexBoundary
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    ComplexMap,
    DenseLinearOperator,
    FailurePolicy,
    FunctionLinearOperator,
    HilbertComplex,
    KroneckerLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    materialize,
    OperatorPairing,
    OperatorProperties,
    PCG,
    prepare,
    solve,
    TolerancePolicy,
)
from ...linalg._complexes import (
    codifferential as complex_codifferential,
    hodge_laplacian as complex_hodge_laplacian,
    HodgeLaplacianPart,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import checked, Dim, Float64, parse, Scope
from .._cell_de_rham import AbstractCellDeRhamComplex
from .._topology import CellComplexTopology, EntitySet, OrientedIncidence
from ._compatible_basis import (
    _PreparedSplineFunctional,
    apply_functionals,
    basis,
    component_functionals,
    pullback_values,
    tensor_basis,
    tensor_quadrature,
)


BoundarySide: TypeAlias = Literal["lower", "upper"]


class SplineQuadraturePointDim(Dim):
    """Prepared tensor quadrature query extent."""


class SplineParameterDim(Dim):
    """Intrinsic chart coordinate extent."""


def _array_identity(value: ArrayLike, /) -> dict[str, object]:
    return array_tree_fingerprint(np.asarray(value))


def _max_abs(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if array.size == 0:
        return jnp.asarray(0.0, dtype=array.dtype)
    return jnp.max(jnp.abs(array))


def _reduced_grid(grid: BSplineGrid, /) -> BSplineGrid:
    if grid.degree < 1:
        raise ValueError("Spline de Rham axes require degree at least one.")
    return BSplineGrid(grid.knots[1:-1], grid.degree - 1)


def _face_restriction(
    shape: tuple[int, ...],
    axis: int,
    side: BoundarySide,
    orientation: int,
    /,
) -> np.ndarray:
    face_shape = shape[:axis] + shape[axis + 1 :]
    matrix = np.zeros((prod(face_shape), prod(shape)), dtype=np.float64)
    fixed = 0 if side == "lower" else shape[axis] - 1
    for face_index in np.ndindex(face_shape):
        volume_index = face_index[:axis] + (fixed,) + face_index[axis:]
        row = np.ravel_multi_index(face_index, face_shape)
        matrix[row, np.ravel_multi_index(volume_index, shape)] = float(orientation)
    return matrix


def _rank(matrix: np.ndarray, tolerance: float, /) -> int:
    if matrix.size == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular_values > tolerance))


def _nullspace(matrix: np.ndarray, tolerance: float, /) -> np.ndarray:
    column_count = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.eye(column_count)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return right[rank:].T.copy()


def _range_basis(matrix: np.ndarray, tolerance: float, /) -> np.ndarray:
    if matrix.shape[1] == 0 or matrix.size == 0:
        return np.zeros((matrix.shape[0], 0))
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return left[:, :rank].copy()


@final
class SplineFormComponent(StrictModule, NonTrainableState):
    """One polynomial tensor component of a spline differential-form space."""

    periodic: tuple[bool, ...] = eqx.field(static=True)

    form_degree: int = eqx.field(static=True)
    component_axes: tuple[int, ...] = eqx.field(static=True)
    grids: tuple[BSplineGrid, ...]
    coefficient_shape: tuple[int, ...] = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    coefficient_kind: str = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        form_degree: int,
        component_axes: Sequence[int],
        grids: Sequence[BSplineGrid],
        /,
        *,
        periodic: tuple[bool, ...] | None = None,
    ) -> None:
        degree = int(form_degree)
        axes = tuple(component_axes)
        grids_ = tuple(grids)
        dimension = len(grids_)
        if degree < 0 or degree > dimension or len(axes) != degree:
            raise ValueError("Form component degree and component axes disagree.")
        if axes != tuple(sorted(axes)) or len(set(axes)) != len(axes):
            raise ValueError("Form component axes must be unique and increasing.")
        if any(axis < 0 or axis >= dimension for axis in axes):
            raise ValueError("Form component axis lies outside the parameter dimension.")
        if any(not isinstance(grid, BSplineGrid) for grid in grids_):
            raise TypeError("Form component grids must be BSplineGrid values.")
        periodic_ = (False,) * dimension if periodic is None else periodic
        if len(periodic_) != dimension:
            raise ValueError("Periodic flags must match the spline dimension.")
        shape = tuple(
            grid.coefficient_count - (periodic_[axis] and axis not in axes)
            for axis, grid in enumerate(grids_)
        )
        self.periodic = periodic_
        self.form_degree = degree
        self.component_axes = axes
        self.grids = grids_
        self.coefficient_shape = shape
        self.coefficient_count = prod(shape)
        self.coefficient_kind = "polynomial"
        self.component_id = canonical_fingerprint(
            {
                "kind": "spline-form-component",
                "form_degree": degree,
                "component_axes": list(axes),
                "degrees": [grid.degree for grid in grids_],
                "knots": [_array_identity(grid.knots) for grid in grids_],
                "coefficient_kind": "polynomial",
                "periodic": periodic_,
            }
        )


@final
class SplineDifferentialSpace(StrictModule, NonTrainableState):
    """All degree-reduced polynomial tensor components of one k-form space."""

    form_degree: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    components: tuple[SplineFormComponent, ...]
    component_offsets: tuple[int, ...] = eqx.field(static=True)
    dof_count: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        form_degree: int,
        dimension: int,
        components: Sequence[SplineFormComponent],
        /,
    ) -> None:
        degree = int(form_degree)
        dimension_ = int(dimension)
        components_ = tuple(components)
        expected_axes = exterior_indices(dimension_, degree)
        if tuple(component.component_axes for component in components_) != expected_axes:
            raise ValueError(
                "Spline form components do not have canonical axis ordering."
            )
        if any(
            component.form_degree != degree or len(component.grids) != dimension_
            for component in components_
        ):
            raise ValueError("Spline form component belongs to another space.")
        offsets: list[int] = []
        offset = 0
        for component in components_:
            offsets.append(offset)
            offset += component.coefficient_count
        self.form_degree = degree
        self.dimension = dimension_
        self.components = components_
        self.component_offsets = tuple(offsets)
        self.dof_count = offset
        self.space_id = canonical_fingerprint(
            {
                "kind": "spline-differential-space",
                "form_degree": degree,
                "dimension": dimension_,
                "components": [component.component_id for component in components_],
            }
        )

    def component_slice(self, component_axes: Sequence[int], /) -> slice:
        axes = tuple(component_axes)
        for offset, component in zip(
            self.component_offsets, self.components, strict=True
        ):
            if component.component_axes == axes:
                return slice(offset, offset + component.coefficient_count)
        raise ValueError(f"No component with axes {axes!r} belongs to this space.")


def _sparse_operator(
    matrix: np.ndarray,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    identifier: str,
    /,
) -> SparseCoordinateOperator:
    rows, columns = np.nonzero(matrix)
    relation = EdgeRelation(
        columns, rows, source_size=source.size, target_size=target.size
    )
    return SparseCoordinateOperator(
        relation,
        matrix[rows, columns],
        source=source,
        target=target,
        operator_id=identifier,
    )


def _operator_matrix(operator: AbstractLinearOperator, /) -> Array:
    return materialize(operator, MaterializationPolicy())


def _paired_space(
    operator: AbstractLinearOperator, identifier: str, policy: LinearSolvePolicy | None, /
) -> ArraySpace:
    if operator.source.size == 0:
        return ArraySpace((0,), dtype=jnp.float64, space_id=identifier)
    policy_ = (
        LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=1e-12, absolute=0.0, max_steps=max(1, 8 * operator.source.size)
            ),
            failure=FailurePolicy("error"),
        )
        if policy is None
        else policy
    )
    prepared = prepare(LinearSystem(operator, problem_id=f"{identifier}:system"), policy_)
    pairing = OperatorPairing(
        operator, prepared_inverse=prepared, pairing_id=f"{identifier}:pairing"
    )
    return ArraySpace(
        (operator.source.size,), dtype=jnp.float64, pairing=pairing, space_id=identifier
    )


def _gram_properties() -> OperatorProperties:
    return OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={"self_adjoint": "construction", "positive_definite": "construction"},
    )


def _make_hilbert(
    matrices: tuple[np.ndarray, ...],
    grams: tuple[AbstractLinearOperator, ...],
    identifier: str,
    policy: LinearSolvePolicy | None,
    /,
) -> HilbertComplex:
    spaces = tuple(
        _paired_space(gram, f"{identifier}:degree:{k}", policy)
        for k, gram in enumerate(grams)
    )
    differentials = tuple(
        _sparse_operator(matrix, spaces[k], spaces[k + 1], f"{identifier}:d:{k}")
        for k, matrix in enumerate(matrices)
    )
    return HilbertComplex(spaces, differentials, complex_id=identifier)


def _topology(
    counts: tuple[int, ...], matrices: tuple[np.ndarray, ...], identifier: str, /
) -> CellComplexTopology:
    entities = tuple(
        EntitySet(
            f"spline-degree-{k}",
            k,
            np.arange(count, dtype=np.int64),
            entity_set_id=f"{identifier}:entities:{k}",
        )
        for k, count in enumerate(counts)
    )
    incidences = []
    for k, matrix in enumerate(matrices):
        rows, columns = np.nonzero(matrix)
        relation = EdgeRelation(
            columns, rows, source_size=counts[k], target_size=counts[k + 1]
        )
        incidences.append(
            OrientedIncidence(
                k + 1, entities[k], entities[k + 1], relation, matrix[rows, columns]
            )
        )
    return CellComplexTopology(
        entities, tuple(incidences), topology_id=f"{identifier}:topology"
    )


class _SplineComplex(AbstractCellDeRhamComplex):
    """Shared native paired realization, not a second exterior-calculus protocol."""

    __strict_abstract__ = True

    dimension: int = eqx.field(static=True)
    primal_twist: FormTwist = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)
    dof_counts: tuple[int, ...] = eqx.field(static=True)
    d_squared_defects: Array
    topology: CellComplexTopology
    boundary_masks: tuple[Array, ...]
    absolute_complex: HilbertComplex
    relative_complex: HilbertComplex
    active_coordinates: tuple[Array, ...]
    coordinate_restrictions: tuple[SparseCoordinateOperator, ...]
    boundary_traces: tuple[ComplexMap, ...]
    boundary_faces: tuple[tuple[int, BoundarySide], ...] = eqx.field(static=True)

    @abc.abstractmethod
    def __init__(self) -> None:
        """Concrete realizations admit their bases or assembly before binding."""
        raise NotImplementedError

    def hilbert_complex(
        self, /, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        match parse(boundary, ComplexBoundary, "boundary"):
            case "absolute":
                return self.absolute_complex
            case "relative":
                return self.relative_complex

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return jnp.asarray(self.absolute_complex.space(degree).riesz(jnp.asarray(values)))

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return jnp.asarray(
            self.absolute_complex.space(degree).inverse_riesz(jnp.asarray(values))
        )

    def exterior_derivative(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        operator = self.hilbert_complex(boundary=boundary).differential(degree)
        return self._conditioned_action(degree, degree + 1, values, operator, boundary)

    def codifferential(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        operator = complex_codifferential(self.hilbert_complex(boundary=boundary), degree)
        return self._conditioned_action(degree, degree - 1, values, operator, boundary)

    def hodge_laplacian(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        part: HodgeLaplacianPart = "complete",
    ) -> Array:
        operator = complex_hodge_laplacian(
            self.hilbert_complex(boundary=boundary), degree, part=part
        )
        return self._conditioned_action(degree, degree, values, operator, boundary)

    def _conditioned_action(
        self,
        degree: int,
        output_degree: int,
        values: ArrayLike,
        operator: AbstractLinearOperator,
        boundary: ComplexBoundary,
        /,
    ) -> Array:
        array = jnp.asarray(values)
        self.absolute_complex.space(degree).validate(array)
        if parse(boundary, ComplexBoundary, "boundary") == "absolute":
            return operator.mv(array)
        source = self.coordinate_restrictions[degree]
        target = self.coordinate_restrictions[output_degree]
        return target.transpose_mv(operator.mv(source.mv(array)))

    def dof_count(self, degree: int, /) -> int:
        return self.absolute_complex.space(degree).size

    def active_indices(
        self, degree: int, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        match parse(boundary, ComplexBoundary, "boundary"):
            case "absolute":
                return jnp.arange(self.dof_count(degree), dtype=jnp.int32)
            case "relative":
                return self.active_coordinates[degree]

    def trace(self, normal_axis: int, side: BoundarySide, /) -> ComplexMap:
        face = (normal_axis, parse(side, BoundarySide, "side"))
        for face_, trace in zip(self.boundary_faces, self.boundary_traces, strict=True):
            if face_ == face:
                return trace
        raise ValueError("The requested spline boundary face is unavailable.")

    def trace_complex_map(
        self, /, *, boundary_mask: ArrayLike | None = None
    ) -> ComplexMap:
        selected = (
            np.ones((len(self.boundary_traces),), dtype=np.bool_)
            if boundary_mask is None
            else np.asarray(boundary_mask, dtype=np.bool_)
        )
        if selected.shape != (len(self.boundary_traces),) or not np.any(selected):
            raise ValueError(
                "Spline trace selection must select at least one declared boundary face."
            )
        traces = tuple(
            trace
            for active, trace in zip(selected, self.boundary_traces, strict=True)
            if active
        )
        identifier = (
            f"{self.realization_id}:boundary:{canonical_fingerprint(selected.tolist())}"
        )
        grams, differentials = [], []
        for degree in range(self.dimension):
            sizes = tuple(trace.target.space(degree).size for trace in traces)
            offsets = tuple(sum(sizes[:i]) for i in range(len(sizes)))
            coordinates = ArraySpace(
                (sum(sizes),),
                dtype=jnp.float64,
                space_id=f"{identifier}:{degree}:coordinates",
            )
            spaces = tuple(trace.target.space(degree) for trace in traces)

            def action(
                value: Array,
                *,
                sizes: tuple[int, ...] = sizes,
                offsets: tuple[int, ...] = offsets,
                spaces: tuple[AbstractVectorSpace, ...] = spaces,
            ) -> Array:
                return jnp.concatenate(
                    tuple(
                        space.flatten(
                            space.riesz(space.unflatten(value[offset : offset + size]))
                        )
                        for space, offset, size in zip(
                            spaces, offsets, sizes, strict=True
                        )
                    )
                )

            grams.append(
                FunctionLinearOperator(
                    action,
                    source=coordinates,
                    target=coordinates,
                    transpose_action=action,
                    properties=_gram_properties(),
                    operator_id=f"{identifier}:{degree}:gram",
                )
            )
            if degree < self.dimension - 1:
                rows = sum(trace.target.space(degree + 1).size for trace in traces)
                matrix = np.zeros((rows, sum(sizes)), dtype=np.float64)
                row_offset = 0
                for trace, column_offset, size in zip(
                    traces, offsets, sizes, strict=True
                ):
                    block = np.asarray(
                        _operator_matrix(trace.target.differential(degree))
                    )
                    matrix[
                        row_offset : row_offset + block.shape[0],
                        column_offset : column_offset + size,
                    ] = block
                    row_offset += block.shape[0]
                differentials.append(matrix)
        target = _make_hilbert(tuple(differentials), tuple(grams), identifier, None)
        maps = tuple(
            _sparse_operator(
                np.concatenate(
                    tuple(
                        np.asarray(_operator_matrix(trace.maps[degree]))
                        for trace in traces
                    ),
                    axis=0,
                ),
                self.absolute_complex.space(degree),
                target.space(degree),
                f"{identifier}:trace:{degree}",
            )
            for degree in range(self.dimension)
        )
        return ComplexMap(
            self.absolute_complex, target, maps, map_id=f"{identifier}:trace"
        )

    def _bind(
        self,
        counts: tuple[int, ...],
        matrices: tuple[np.ndarray, ...],
        grams: tuple[AbstractLinearOperator, ...],
        masks: tuple[np.ndarray, ...],
        identifier: str,
        twist: FormTwist,
        policy: LinearSolvePolicy | None,
        /,
    ) -> None:
        topology = _topology(counts, matrices, identifier)
        closure_defects = tuple(
            float(
                np.max(
                    np.abs((left.scipy_boundary() @ right.scipy_boundary()).data),
                    initial=0.0,
                )
            )
            for left, right in zip(
                topology.incidences[:-1], topology.incidences[1:], strict=True
            )
        )
        absolute = _make_hilbert(matrices, grams, identifier, policy)
        indices = tuple(np.flatnonzero(~mask).astype(np.int32) for mask in masks)
        for k, matrix in enumerate(matrices):
            if np.any(matrix[np.ix_(np.flatnonzero(masks[k + 1]), indices[k])]):
                raise ValueError(
                    "Spline boundary masks do not define a relative subcomplex."
                )
        relative_grams: list[AbstractLinearOperator] = []
        for k, (gram, active) in enumerate(zip(grams, indices, strict=True)):
            coords = ArraySpace(
                (active.size,),
                dtype=jnp.float64,
                space_id=f"{identifier}:relative:{k}:coordinates",
            )
            relation = EdgeRelation(
                active,
                np.arange(active.size, dtype=np.int32),
                source_size=counts[k],
                target_size=active.size,
            )
            restriction = SparseCoordinateOperator(
                relation,
                jnp.ones((active.size,), dtype=jnp.float64),
                source=gram.source,
                target=coords,
                operator_id=f"{identifier}:relative:{k}:restriction",
            )

            def action(
                value: Array,
                *,
                restriction: SparseCoordinateOperator = restriction,
                gram: AbstractLinearOperator = gram,
            ) -> Array:
                return restriction.mv(gram.mv(restriction.transpose_mv(value)))

            relative_grams.append(
                FunctionLinearOperator(
                    action,
                    source=coords,
                    target=coords,
                    transpose_action=action,
                    properties=_gram_properties(),
                    operator_id=f"{identifier}:relative:{k}:gram",
                )
            )
        relative_matrices = tuple(
            matrix[np.ix_(indices[k + 1], indices[k])]
            for k, matrix in enumerate(matrices)
        )
        self.dimension = len(counts) - 1
        self.primal_twist = twist
        self.realization_id = identifier
        self.complex_id = identifier
        self.dof_counts = counts
        self.d_squared_defects = jnp.asarray(closure_defects, dtype=jnp.float64)
        self.topology = topology
        self.boundary_masks = tuple(jnp.asarray(mask) for mask in masks)
        self.absolute_complex = absolute
        self.relative_complex = _make_hilbert(
            relative_matrices, tuple(relative_grams), f"{identifier}:relative", policy
        )
        self.active_coordinates = tuple(jnp.asarray(active) for active in indices)
        restrictions = []
        for k, active in enumerate(indices):
            relation = EdgeRelation(
                active,
                np.arange(active.size, dtype=np.int32),
                source_size=counts[k],
                target_size=active.size,
            )
            restrictions.append(
                SparseCoordinateOperator(
                    relation,
                    jnp.ones((active.size,), dtype=jnp.float64),
                    source=absolute.space(k),
                    target=self.relative_complex.space(k),
                    operator_id=f"{identifier}:active-restriction:{k}",
                )
            )
        self.coordinate_restrictions = tuple(restrictions)


def _differentials(
    spaces: tuple[SplineDifferentialSpace, ...], periodic: tuple[bool, ...], /
) -> tuple[np.ndarray, ...]:
    matrices = []
    for degree, source in enumerate(spaces[:-1]):
        target = spaces[degree + 1]
        matrix = np.zeros((target.dof_count, source.dof_count), dtype=np.float64)
        for offset, component in zip(
            source.component_offsets, source.components, strict=True
        ):
            for axis in range(source.dimension):
                if axis in component.component_axes:
                    continue
                axes = tuple(sorted((*component.component_axes, axis)))
                block_shape = target.components[
                    exterior_indices(source.dimension, degree + 1).index(axes)
                ].coefficient_shape
                block = np.zeros(
                    (prod(block_shape), component.coefficient_count), dtype=np.float64
                )
                sign = (-1) ** sum(
                    existing < axis for existing in component.component_axes
                )
                for target_index in np.ndindex(block_shape):
                    lower = list(target_index)
                    upper = list(target_index)
                    upper[axis] = (
                        (upper[axis] + 1) % component.coefficient_shape[axis]
                        if periodic[axis]
                        else upper[axis] + 1
                    )
                    row = np.ravel_multi_index(target_index, block_shape)
                    block[
                        row,
                        np.ravel_multi_index(tuple(lower), component.coefficient_shape),
                    ] -= sign
                    block[
                        row,
                        np.ravel_multi_index(tuple(upper), component.coefficient_shape),
                    ] += sign
                matrix[
                    target.component_slice(axes),
                    offset : offset + component.coefficient_count,
                ] = block
        matrices.append(matrix)
    return tuple(matrices)


def _separable_grams(
    spaces: tuple[SplineDifferentialSpace, ...],
    periodic: tuple[bool, ...],
    identifier: str,
    /,
) -> tuple[
    tuple[AbstractLinearOperator, ...], tuple[tuple[KroneckerLinearOperator, ...], ...]
]:
    grams: list[AbstractLinearOperator] = []
    all_blocks: list[tuple[KroneckerLinearOperator, ...]] = []
    for k, space in enumerate(spaces):
        blocks = []
        for c, component in enumerate(space.components):
            factors = []
            for axis, grid in enumerate(component.grids):
                points, weights = grid.quadrature(2 * grid.degree + 2)
                values = basis(
                    grid,
                    points,
                    differential=axis in component.component_axes,
                    periodic=periodic[axis] and axis not in component.component_axes,
                )
                matrix = values.T @ (weights[:, None] * values)
                coordinates = ArraySpace(
                    (matrix.shape[0],),
                    dtype=jnp.float64,
                    space_id=f"{identifier}:{k}:{c}:{axis}:coordinates",
                )
                factors.append(
                    DenseLinearOperator(
                        matrix,
                        source=coordinates,
                        target=coordinates,
                        properties=_gram_properties(),
                        operator_id=f"{identifier}:{k}:{c}:{axis}:gram",
                    )
                )
            blocks.append(
                KroneckerLinearOperator(
                    tuple(factors), operator_id=f"{identifier}:{k}:{c}:kronecker"
                )
            )
        blocks_ = tuple(blocks)
        coordinates = ArraySpace(
            (space.dof_count,),
            dtype=jnp.float64,
            space_id=f"{identifier}:{k}:coordinates",
        )

        def action(
            value: Array,
            *,
            blocks: tuple[KroneckerLinearOperator, ...] = blocks_,
            space: SplineDifferentialSpace = space,
        ) -> Array:
            return jnp.concatenate(
                tuple(
                    block.mv(
                        value[offset : offset + component.coefficient_count].reshape(
                            component.coefficient_shape
                        )
                    ).reshape((-1,))
                    for block, offset, component in zip(
                        blocks, space.component_offsets, space.components, strict=True
                    )
                )
            )

        grams.append(
            FunctionLinearOperator(
                action,
                source=coordinates,
                target=coordinates,
                transpose_action=action,
                properties=_gram_properties(),
                operator_id=f"{identifier}:{k}:gram",
            )
        )
        all_blocks.append(blocks_)
    return tuple(grams), tuple(all_blocks)


@final
class _SplineQuadrature(StrictModule, NonTrainableState):
    """Fixed quadrature routes and basis values reused by geometry refresh."""

    __strict_contract__ = True

    spaces: tuple[SplineDifferentialSpace, ...]
    points: Float64[SplineQuadraturePointDim, SplineParameterDim]
    weights: Float64[SplineQuadraturePointDim]
    reference_bases: tuple[Array, ...]

    def __init__(
        self,
        spaces: tuple[SplineDifferentialSpace, ...],
        base: tuple[BSplineGrid, ...],
        periodic: tuple[bool, ...],
        degree: int,
        /,
    ) -> None:
        points, weights = tensor_quadrature(base, degree)
        references = []
        for space in spaces:
            reference = jnp.zeros(
                (points.shape[0], len(space.components), space.dof_count),
                dtype=jnp.float64,
            )
            for c, (offset, component) in enumerate(
                zip(space.component_offsets, space.components, strict=True)
            ):
                values = tensor_basis(
                    component.grids, points, component.component_axes, periodic
                )
                reference = reference.at[
                    :, c, offset : offset + component.coefficient_count
                ].set(values)
            references.append(reference)
        scope = Scope()
        self.spaces = spaces
        self.points = parse(
            points,
            Float64[SplineQuadraturePointDim, SplineParameterDim],
            "points",
            scope=scope,
        )
        self.weights = parse(
            weights, Float64[SplineQuadraturePointDim], "weights", scope=scope
        )
        self.reference_bases = tuple(references)


def _mapped_grams(
    quadrature: _SplineQuadrature, geometry: Callable[[Array], Array], identifier: str, /
) -> tuple[AbstractLinearOperator, ...]:
    from ...linalg import compound_matrix, DenseLU, RHSLayout

    points, weights = quadrature.points, quadrature.weights
    jacobians = jax.vmap(jax.jacfwd(geometry))(points)
    metrics = jnp.swapaxes(jacobians, -1, -2) @ jacobians
    determinant = compound_matrix(metrics, len(quadrature.spaces) - 1)[:, 0, 0]
    determinant = eqx.error_if(
        determinant,
        jnp.any(~jnp.isfinite(metrics)) | jnp.any(determinant <= 0.0),
        "Spline geometry must have finite full-rank tangent Jacobians at quadrature points.",
    )
    volume = jnp.sqrt(determinant)
    grams = []
    for k, (space, reference) in enumerate(
        zip(quadrature.spaces, quadrature.reference_bases, strict=True)
    ):
        compound = compound_matrix(metrics, k)

        def metric_solve(matrix: Array, rhs: Array) -> Array:
            operator = DenseLinearOperator(matrix, operator_id=f"{identifier}:{k}:metric")
            return solve(
                LinearSystem(operator, problem_id=f"{identifier}:{k}:metric-system"),
                rhs,
                policy=LinearSolvePolicy(DenseLU(), failure=FailurePolicy("error")),
                rhs_layout=RHSLayout((space.dof_count,)),
            ).value

        weighted = jax.vmap(metric_solve)(compound, reference)
        matrix = contract("qci,qcj,q->ij", reference, weighted, weights * volume)
        coordinates = ArraySpace(
            (space.dof_count,),
            dtype=jnp.float64,
            space_id=f"{identifier}:{k}:coordinates",
        )
        grams.append(
            DenseLinearOperator(
                matrix,
                source=coordinates,
                target=coordinates,
                properties=_gram_properties(),
                operator_id=f"{identifier}:{k}:gram",
            )
        )
    return tuple(grams)


def _refresh_hilbert(
    template: HilbertComplex,
    grams: tuple[AbstractLinearOperator, ...],
    policy: LinearSolvePolicy | None,
    /,
) -> HilbertComplex:
    spaces = tuple(
        _paired_space(gram, template.space(k).space_id, policy)
        for k, gram in enumerate(grams)
    )
    differentials = tuple(
        eqx.tree_at(
            lambda operator: (operator.source, operator.target),
            operator,
            (spaces[k], spaces[k + 1]),
        )
        for k, operator in enumerate(template.differentials)
    )
    return HilbertComplex(spaces, differentials, complex_id=template.complex_id)


@final
class SplineDeRhamComplex(_SplineComplex):
    """Tensor spline differential forms with exact sparse d and physical L2 pairing."""

    base_grids: tuple[BSplineGrid, ...]
    spaces: tuple[SplineDifferentialSpace, ...]
    periodic: tuple[bool, ...] = eqx.field(static=True)
    geometry: Callable[[Array], Array] | None
    geometry_id: str = eqx.field(static=True)
    quadrature_degree: int = eqx.field(static=True)
    gram_blocks: tuple[tuple[KroneckerLinearOperator, ...], ...]
    functional_queries: tuple[Array, ...]
    functional_routes: tuple[Array, ...]
    functional_solvers: tuple[_PreparedSplineFunctional, ...]
    quadrature: _SplineQuadrature | None
    face_quadratures: tuple[_SplineQuadrature | None, ...]
    hodge_policy: LinearSolvePolicy | None

    def __init__(
        self,
        grids: Sequence[BSplineGrid],
        /,
        *,
        periodic: Sequence[bool] | None = None,
        geometry: Callable[[Array], Array] | None = None,
        geometry_id: str | None = None,
        twist: FormTwist = "untwisted",
        quadrature_degree: int | None = None,
        hodge_policy: LinearSolvePolicy | None = None,
    ) -> None:
        grids_ = tuple(grids)
        dimension = len(grids_)
        if dimension < 1 or dimension > 3:
            raise ValueError("Spline complexes require one to three axes.")
        if any(not isinstance(grid, BSplineGrid) for grid in grids_):
            raise TypeError("Spline axes must be BSplineGrid values.")
        if any(grid.degree < 1 for grid in grids_):
            raise ValueError("Spline de Rham axes require degree at least one.")
        periodic_ = (False,) * dimension if periodic is None else tuple(periodic)
        if len(periodic_) != dimension or any(
            not isinstance(flag, bool) for flag in periodic_
        ):
            raise ValueError("Periodic flags must match the spline dimension.")
        if geometry is not None and (geometry_id is None or not geometry_id.strip()):
            raise ValueError("Mapped spline geometry requires an explicit geometry_id.")
        if geometry is not None:
            if not callable(geometry):
                raise TypeError("Spline geometry must be callable.")
            shape = jax.eval_shape(
                geometry, jax.ShapeDtypeStruct((dimension,), jnp.float64)
            )
            if len(shape.shape) != 1 or shape.shape[0] < dimension:
                raise ValueError(
                    "Spline geometry must return ambient coordinates of dimension at least the parameter dimension."
                )
        twist_ = parse(twist, FormTwist, "twist")
        reduced = tuple(_reduced_grid(grid) for grid in grids_)
        spaces = tuple(
            SplineDifferentialSpace(
                k,
                dimension,
                tuple(
                    SplineFormComponent(
                        k,
                        axes,
                        tuple(
                            reduced[axis] if axis in axes else grids_[axis]
                            for axis in range(dimension)
                        ),
                        periodic=periodic_,
                    )
                    for axes in exterior_indices(dimension, k)
                ),
            )
            for k in range(dimension + 1)
        )
        matrices = _differentials(spaces, periodic_)
        order = (
            2 * max(grid.degree for grid in grids_) + 4
            if quadrature_degree is None
            else quadrature_degree
        )
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or order < 2 * max(grid.degree for grid in grids_)
        ):
            raise ValueError(
                "Spline quadrature degree must be an integer resolving every squared component basis."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "spline-de-rham",
                "spaces": [space.space_id for space in spaces],
                "periodic": periodic_,
                "geometry": geometry_id,
                "twist": twist_,
                "quadrature": order,
            }
        )
        if geometry is None:
            grams, blocks = _separable_grams(spaces, periodic_, identifier)
            quadrature = None
        else:
            quadrature = _SplineQuadrature(spaces, grids_, periodic_, order)
            grams, blocks = _mapped_grams(quadrature, geometry, identifier), ()
        masks = []
        for space in spaces:
            mask = np.zeros((space.dof_count,), dtype=np.bool_)
            for offset, component in zip(
                space.component_offsets, space.components, strict=True
            ):
                indices = np.indices(component.coefficient_shape)
                local = np.zeros(component.coefficient_shape, dtype=np.bool_)
                for axis in range(dimension):
                    if axis not in component.component_axes and not periodic_[axis]:
                        local |= (indices[axis] == 0) | (
                            indices[axis] == component.coefficient_shape[axis] - 1
                        )
                mask[offset : offset + component.coefficient_count] = local.reshape((-1,))
            masks.append(mask)
        self._bind(
            tuple(space.dof_count for space in spaces),
            matrices,
            grams,
            tuple(masks),
            identifier,
            twist_,
            hodge_policy,
        )
        self.base_grids = grids_
        self.spaces = spaces
        self.periodic = periodic_
        self.geometry = geometry
        self.geometry_id = "identity" if geometry_id is None else geometry_id
        self.quadrature_degree = order
        self.gram_blocks = blocks
        queries, routes, solvers = [], [], []
        for space in spaces:
            for component in space.components:
                points, route, _ = component_functionals(
                    grids_,
                    component.component_axes,
                    periodic_,
                    max(grid.degree for grid in grids_) + 2,
                )
                queries.append(points)
                routes.append(route)
                solvers.append(
                    _PreparedSplineFunctional(
                        grids_,
                        component.grids,
                        component.component_axes,
                        periodic_,
                        f"{component.component_id}:functionals",
                        max(grid.degree for grid in grids_) + 2,
                    )
                )
        self.functional_queries, self.functional_routes, self.functional_solvers = (
            tuple(queries),
            tuple(routes),
            tuple(solvers),
        )
        self.quadrature, self.hodge_policy = quadrature, hodge_policy
        faces = tuple(
            (axis, parse(side, BoundarySide, "side"))
            for axis in range(dimension)
            if not periodic_[axis]
            for side in ("lower", "upper")
        )
        self.boundary_faces = faces
        prepared_faces = tuple(
            self._prepare_trace(axis, side, hodge_policy)
            for axis, side in self.boundary_faces
        )
        self.boundary_traces = tuple(value[0] for value in prepared_faces)
        self.face_quadratures = tuple(value[1] for value in prepared_faces)

    def _prepare_trace(
        self, axis: int, side: BoundarySide, policy: LinearSolvePolicy | None, /
    ) -> tuple[ComplexMap, _SplineQuadrature | None]:
        orientation = ((-1) ** axis) * (1 if side == "upper" else -1)
        if self.dimension == 1:
            identifier = f"{self.realization_id}:point:{side}"
            space = ArraySpace(
                (1,), dtype=jnp.float64, space_id=f"{identifier}:coordinates"
            )
            target = HilbertComplex((space,), (), complex_id=identifier)
            matrix = np.zeros((1, self.dof_count(0)), dtype=np.float64)
            matrix[0, 0 if side == "lower" else -1] = (
                orientation if self.primal_twist == "twisted" else 1.0
            )
            operator = _sparse_operator(
                matrix, self.absolute_complex.space(0), space, f"{identifier}:restriction"
            )
            return ComplexMap(
                self.absolute_complex, target, (operator,), map_id=f"{identifier}:trace"
            ), None
        tangential_axes = tuple(value for value in range(self.dimension) if value != axis)
        face_grids = self.base_grids[:axis] + self.base_grids[axis + 1 :]
        reflection_sum = face_grids[0].knots[0] + face_grids[0].knots[-1]
        if orientation < 0:
            reflected = BSplineGrid(
                reflection_sum - face_grids[0].knots[::-1], face_grids[0].degree
            )
            face_grids = (reflected, *face_grids[1:])
        boundary_value = self.base_grids[axis].active_interval[
            0 if side == "lower" else 1
        ]
        geometry = self.geometry

        def face_geometry(point: Array) -> Array:
            parameter = (
                jnp.zeros((self.dimension,), dtype=point.dtype)
                .at[jnp.asarray(tangential_axes)]
                .set(point)
            )
            parameter = parameter.at[axis].set(boundary_value)
            if orientation < 0:
                parameter = parameter.at[tangential_axes[0]].set(
                    reflection_sum - point[0]
                )
            return parameter if geometry is None else geometry(parameter)

        target = SplineDeRhamComplex(
            face_grids,
            periodic=self.periodic[:axis] + self.periodic[axis + 1 :],
            geometry=face_geometry,
            geometry_id=f"{self.geometry_id}:face:{axis}:{side}",
            twist=self.primal_twist,
            hodge_policy=policy,
        )
        maps = []
        for k, target_space in enumerate(target.spaces):
            source_space = self.spaces[k]
            matrix = np.zeros(
                (target_space.dof_count, source_space.dof_count), dtype=np.float64
            )
            for component in source_space.components:
                if axis in component.component_axes:
                    continue
                target_axes = tuple(
                    value - (value > axis) for value in component.component_axes
                )
                block = _face_restriction(component.coefficient_shape, axis, side, 1)
                shape = (
                    component.coefficient_shape[:axis]
                    + component.coefficient_shape[axis + 1 :]
                )
                if orientation < 0:
                    permutation = np.arange(shape[0] - 1, -1, -1, dtype=np.int32)
                    if self.periodic[tangential_axes[0]] and 0 not in target_axes:
                        permutation = np.concatenate(
                            (
                                np.asarray([0], dtype=np.int32),
                                np.arange(shape[0] - 1, 0, -1, dtype=np.int32),
                            )
                        )
                    block = np.take(
                        block.reshape((*shape, component.coefficient_count)),
                        permutation,
                        axis=0,
                    ).reshape(block.shape)
                    if 0 in target_axes:
                        block = -block
                if self.primal_twist == "twisted":
                    block = orientation * block
                matrix[
                    target_space.component_slice(target_axes),
                    source_space.component_slice(component.component_axes),
                ] = block
            maps.append(
                _sparse_operator(
                    matrix,
                    self.absolute_complex.space(k),
                    target.absolute_complex.space(k),
                    f"{self.realization_id}:trace:{axis}:{side}:{k}",
                )
            )
        if target.quadrature is None:
            raise RuntimeError("Prepared spline face must carry a mapped quadrature.")
        return ComplexMap(
            self.absolute_complex,
            target.absolute_complex,
            tuple(maps),
            map_id=f"{self.realization_id}:trace:{axis}:{side}",
        ), target.quadrature

    def interpolant(self, degree: int, form: Callable[[Array], Array], /) -> Array:
        self.absolute_complex.space(degree)
        space = self.spaces[degree]
        component_start = sum(len(other.components) for other in self.spaces[:degree])
        coefficients = []
        for c, component in enumerate(space.components):
            index = component_start + c
            values = pullback_values(
                form,
                self.functional_queries[index],
                component.component_axes,
                self.geometry,
                self.primal_twist,
            )
            rhs = apply_functionals(
                values, self.functional_routes[index], component.coefficient_count
            )
            coefficients.append(self.functional_solvers[index].solve(rhs))
        return jnp.concatenate(tuple(coefficients))

    def refresh_geometry(
        self, geometry: Callable[[Array], Array], /
    ) -> SplineDeRhamComplex:
        """Refresh numeric metric leaves on fixed bases, routes, and binding ids."""
        quadrature = self.quadrature
        if quadrature is None:
            raise ValueError(
                "Geometry refresh requires a mapped realization admitted at preparation."
            )
        grams = _mapped_grams(quadrature, geometry, self.realization_id)
        absolute = _refresh_hilbert(self.absolute_complex, grams, self.hodge_policy)
        relative_grams = []
        for k, (gram, restriction) in enumerate(
            zip(grams, self.coordinate_restrictions, strict=True)
        ):
            old_space = self.relative_complex.space(k)
            if not isinstance(old_space, ArraySpace):
                raise TypeError("Spline relative coordinates must be array spaces.")
            coordinates = ArraySpace(
                (old_space.size,),
                dtype=jnp.float64,
                space_id=f"{self.realization_id}:relative:{k}:coordinates",
            )

            def action(
                value: Array,
                *,
                gram: AbstractLinearOperator = gram,
                restriction: SparseCoordinateOperator = restriction,
            ) -> Array:
                return restriction.mv(gram.mv(restriction.transpose_mv(value)))

            relative_grams.append(
                FunctionLinearOperator(
                    action,
                    source=coordinates,
                    target=coordinates,
                    transpose_action=action,
                    properties=_gram_properties(),
                    operator_id=f"{self.realization_id}:relative:{k}:gram",
                )
            )
        relative = _refresh_hilbert(
            self.relative_complex, tuple(relative_grams), self.hodge_policy
        )
        restrictions = tuple(
            eqx.tree_at(
                lambda operator: (operator.source, operator.target),
                operator,
                (absolute.space(k), relative.space(k)),
            )
            for k, operator in enumerate(self.coordinate_restrictions)
        )
        traces = []
        for (axis, side), trace, face_quadrature in zip(
            self.boundary_faces, self.boundary_traces, self.face_quadratures, strict=True
        ):
            if face_quadrature is None:
                maps = tuple(
                    eqx.tree_at(
                        lambda operator: operator.source, operator, absolute.space(k)
                    )
                    for k, operator in enumerate(trace.maps)
                )
                traces.append(
                    ComplexMap(absolute, trace.target, maps, map_id=trace.map_id)
                )
                continue
            tangential_axes = tuple(
                value for value in range(self.dimension) if value != axis
            )
            orientation = ((-1) ** axis) * (1 if side == "upper" else -1)
            first_grid = self.base_grids[tangential_axes[0]]
            reflection_sum = first_grid.knots[0] + first_grid.knots[-1]
            boundary_value = self.base_grids[axis].active_interval[
                0 if side == "lower" else 1
            ]

            def face_geometry(
                point: Array,
                *,
                axes: tuple[int, ...] = tangential_axes,
                axis: int = axis,
                orientation: int = orientation,
                reflection_sum: Array = reflection_sum,
                boundary_value: Array = boundary_value,
            ) -> Array:
                parameter = (
                    jnp.zeros((self.dimension,), dtype=point.dtype)
                    .at[jnp.asarray(axes)]
                    .set(point)
                    .at[axis]
                    .set(boundary_value)
                )
                if orientation < 0:
                    parameter = parameter.at[axes[0]].set(reflection_sum - point[0])
                return geometry(parameter)

            face_grams = _mapped_grams(
                face_quadrature, face_geometry, trace.target.complex_id
            )
            face_target = _refresh_hilbert(trace.target, face_grams, self.hodge_policy)
            maps = tuple(
                eqx.tree_at(
                    lambda operator: (operator.source, operator.target),
                    operator,
                    (absolute.space(k), face_target.space(k)),
                )
                for k, operator in enumerate(trace.maps)
            )
            traces.append(ComplexMap(absolute, face_target, maps, map_id=trace.map_id))
        return eqx.tree_at(
            lambda complex_: (
                complex_.geometry,
                complex_.absolute_complex,
                complex_.relative_complex,
                complex_.coordinate_restrictions,
                complex_.boundary_traces,
            ),
            self,
            (geometry, absolute, relative, restrictions, tuple(traces)),
            is_leaf=lambda value: value is None,
        )

    def reconstruction(
        self, degree: int, coefficients: ArrayLike, points: ArrayLike, /
    ) -> Array:
        self.absolute_complex.space(degree)
        space = self.spaces[degree]
        values = jnp.asarray(coefficients)
        points_ = jnp.asarray(points)
        if (
            values.shape != (space.dof_count,)
            or points_.ndim != 2
            or points_.shape[1] != self.dimension
        ):
            raise ValueError(
                "Spline reconstruction coefficient/query dimensions disagree."
            )
        reference = jnp.stack(
            tuple(
                tensor_basis(
                    component.grids, points_, component.component_axes, self.periodic
                )
                @ values[offset : offset + component.coefficient_count]
                for offset, component in zip(
                    space.component_offsets, space.components, strict=True
                )
            ),
            axis=-1,
        )
        if self.geometry is None:
            return reference
        from ...exterior import FormType, FormValueSpec, map_reference_values

        geometry = self.geometry

        def physical_values(value: Array, point: Array) -> Array:
            return map_reference_values(value, spec, jax.jacfwd(geometry)(point))

        spec = FormValueSpec(
            FormType(self.dimension, degree, twist=self.primal_twist), proxy="components"
        )
        return jax.vmap(physical_values)(reference, points_)

    def transfer(
        self, target: SplineDeRhamComplex, /, *, tolerance: float = 1e-10
    ) -> ComplexMap:
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Transfer tolerance must be finite and positive.")
        if (
            target.dimension != self.dimension
            or target.periodic != self.periodic
            or target.geometry_id != self.geometry_id
            or target.primal_twist != self.primal_twist
        ):
            raise ValueError(
                "Spline transfer requires the same chart, periodicity, and twist."
            )
        maps = []
        quadrature_order = (
            max(
                max(grid.degree for grid in self.base_grids),
                max(grid.degree for grid in target.base_grids),
            )
            + 2
        )
        partitions = tuple(grid.breakpoints for grid in self.base_grids)
        for degree in range(self.dimension + 1):
            source_space, target_space = self.spaces[degree], target.spaces[degree]
            matrix = jnp.zeros(
                (target_space.dof_count, source_space.dof_count), dtype=jnp.float64
            )
            solver_start = sum(len(space.components) for space in target.spaces[:degree])
            for c, (source_component, target_component) in enumerate(
                zip(source_space.components, target_space.components, strict=True)
            ):
                points, route, _ = component_functionals(
                    target.base_grids,
                    target_component.component_axes,
                    target.periodic,
                    quadrature_order,
                    partitions=partitions,
                )
                values = tensor_basis(
                    source_component.grids,
                    points,
                    source_component.component_axes,
                    self.periodic,
                )

                def functional(column: Array) -> Array:
                    return apply_functionals(
                        column, route, target_component.coefficient_count
                    )

                rhs = jax.vmap(functional, in_axes=1, out_axes=1)(values)
                block = jax.vmap(
                    target.functional_solvers[solver_start + c].solve,
                    in_axes=1,
                    out_axes=1,
                )(rhs)
                matrix = matrix.at[
                    target_space.component_slice(target_component.component_axes),
                    source_space.component_slice(source_component.component_axes),
                ].set(block)
            maps.append(
                DenseLinearOperator(
                    matrix,
                    source=self.absolute_complex.space(degree),
                    target=target.absolute_complex.space(degree),
                    operator_id=f"{self.realization_id}:transfer:{target.realization_id}:{degree}",
                )
            )
        transfer = ComplexMap(
            self.absolute_complex,
            target.absolute_complex,
            tuple(maps),
            map_id=f"{self.realization_id}:transfer:{target.realization_id}",
        )
        for degree in range(self.dimension):
            defect = (
                maps[degree + 1].matrix
                @ _operator_matrix(self.absolute_complex.differential(degree))
                - _operator_matrix(target.absolute_complex.differential(degree))
                @ maps[degree].matrix
            )
            if np.max(np.abs(np.asarray(defect)), initial=0.0) > tolerance:
                raise ValueError("Spline spaces do not admit this commuting transfer.")
        return transfer


@final
class AssembledSplineDeRhamComplex(_SplineComplex):
    """Admitted multipatch coefficient complex with metric-only Gram Hodges."""

    source_complex_ids: tuple[str, ...] = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        dof_counts: Sequence[int],
        exterior_derivatives: Sequence[ArrayLike],
        source_complex_ids: Sequence[str],
        assembly_id: str,
        /,
        *,
        gram_operators: Sequence[AbstractLinearOperator],
        boundary_masks: Sequence[ArrayLike] | None = None,
        boundary_traces: Sequence[ComplexMap] = (),
        boundary_faces: Sequence[tuple[int, BoundarySide]] = (),
        twist: FormTwist = "untwisted",
        hodge_policy: LinearSolvePolicy | None = None,
    ) -> None:
        counts = tuple(dof_counts)
        matrices = tuple(
            np.asarray(value, dtype=np.float64) for value in exterior_derivatives
        )
        grams = tuple(gram_operators)
        source_ids = tuple(source_complex_ids)
        faces = tuple(
            (axis, parse(side, BoundarySide, "side")) for axis, side in boundary_faces
        )
        twist_ = parse(twist, FormTwist, "twist")
        if len(set(faces)) != len(faces) or any(
            axis < 0 or axis >= dimension for axis, _ in faces
        ):
            raise ValueError("Assembled spline boundary face declarations are invalid.")
        if (
            dimension < 1
            or dimension > 3
            or len(counts) != dimension + 1
            or len(matrices) != dimension
            or len(grams) != len(counts)
        ):
            raise ValueError("Assembled spline complex degree counts disagree.")
        if not assembly_id or not source_ids or any(not value for value in source_ids):
            raise ValueError(
                "Assembled spline complex requires explicit source and assembly identities."
            )
        for k, matrix in enumerate(matrices):
            if matrix.shape != (counts[k + 1], counts[k]) or np.any(~np.isfinite(matrix)):
                raise ValueError(
                    "Assembled spline differential dimensions or values are invalid."
                )
        masks = (
            tuple(np.zeros((count,), dtype=np.bool_) for count in counts)
            if boundary_masks is None
            else tuple(np.asarray(value, dtype=np.bool_) for value in boundary_masks)
        )
        if len(masks) != len(counts) or any(
            mask.shape != (count,) for mask, count in zip(masks, counts, strict=True)
        ):
            raise ValueError(
                "Assembled spline boundary masks disagree with coefficients."
            )
        if len(boundary_traces) != len(boundary_faces):
            raise ValueError(
                "Assembled spline boundary maps and face declarations disagree."
            )
        if any(
            gram.source.size != count or gram.target.size != count
            for gram, count in zip(grams, counts, strict=True)
        ):
            raise ValueError(
                "Assembled spline Gram dimensions disagree with coefficients."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "assembled-spline",
                "assembly": assembly_id,
                "sources": source_ids,
                "d": [_array_identity(matrix) for matrix in matrices],
                "grams": [gram.operator_id for gram in grams],
                "boundary_masks": [_array_identity(mask) for mask in masks],
                "faces": faces,
                "twist": twist_,
            }
        )
        self._bind(counts, matrices, grams, masks, identifier, twist_, hodge_policy)
        self.source_complex_ids = source_ids
        self.assembly_id = assembly_id
        self.boundary_traces = tuple(boundary_traces)
        self.boundary_faces = faces


class RelativeCohomologyEvidence(StrictModule, NonTrainableState):
    """Relative subcomplex, nullspaces, images, and quotient cohomology evidence."""

    complex_id: str = eqx.field(static=True)
    boundary_faces: tuple[tuple[int, BoundarySide], ...] = eqx.field(static=True)
    restriction_bases: tuple[Array, ...]
    restricted_derivatives: tuple[Array, ...]
    nullspace_bases: tuple[Array, ...]
    cohomology_bases: tuple[Array, ...]
    derivative_ranks: tuple[int, ...] = eqx.field(static=True)
    nullities: tuple[int, ...] = eqx.field(static=True)
    betti_numbers: tuple[int, ...] = eqx.field(static=True)
    closure_defects: Array
    tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        complex_: _SplineComplex,
        boundary_faces: Sequence[tuple[int, BoundarySide]],
        /,
        *,
        tolerance: float = 1e-12,
    ) -> None:
        faces = tuple((int(axis), side) for axis, side in boundary_faces)
        tolerance_ = float(tolerance)
        if tolerance_ <= 0.0 or not np.isfinite(tolerance_):
            raise ValueError("Relative cohomology tolerance must be positive and finite.")
        if len(set(faces)) != len(faces):
            raise ValueError("Relative boundary faces must be unique.")
        validated_faces: list[tuple[int, BoundarySide]] = []
        for axis, side in faces:
            if axis < 0 or axis >= complex_.dimension:
                raise ValueError("Relative boundary face is invalid.")
            validated_faces.append((axis, parse(side, BoundarySide, "side")))

        restrictions: list[np.ndarray] = []
        for degree in range(complex_.dimension + 1):
            matrices = [
                np.asarray(_operator_matrix(complex_.trace(axis, side).maps[degree]))
                for axis, side in validated_faces
                if degree < complex_.dimension
            ]
            trace_matrix = (
                np.concatenate(matrices, axis=0)
                if matrices
                else np.zeros((0, complex_.dof_count(degree)))
            )
            restrictions.append(_nullspace(trace_matrix, tolerance_))

        restricted_derivatives: list[np.ndarray] = []
        closure_defects: list[float] = []
        for degree, derivative in enumerate(complex_.absolute_complex.differentials):
            source_basis = restrictions[degree]
            target_basis = restrictions[degree + 1]
            derivative_host = np.asarray(_operator_matrix(derivative))
            restricted = target_basis.T @ derivative_host @ source_basis
            closure = (
                (np.eye(complex_.dof_count(degree + 1)) - target_basis @ target_basis.T)
                @ derivative_host
                @ source_basis
            )
            restricted_derivatives.append(restricted)
            closure_defects.append(float(np.max(np.abs(closure), initial=0.0)))

        derivative_ranks = tuple(
            _rank(derivative, tolerance_) for derivative in restricted_derivatives
        )
        nullspaces: list[np.ndarray] = []
        cohomologies: list[np.ndarray] = []
        nullities: list[int] = []
        betti: list[int] = []
        for degree in range(complex_.dimension + 1):
            restricted_dimension = restrictions[degree].shape[1]
            outgoing = (
                restricted_derivatives[degree]
                if degree < complex_.dimension
                else np.zeros((0, restricted_dimension))
            )
            kernel = _nullspace(outgoing, tolerance_)
            previous = (
                restricted_derivatives[degree - 1]
                if degree > 0
                else np.zeros((restricted_dimension, 0))
            )
            image = _range_basis(previous, tolerance_)
            quotient_candidates = (
                np.eye(restricted_dimension) - image @ image.T
            ) @ kernel
            quotient = _range_basis(quotient_candidates, tolerance_)
            full_kernel = restrictions[degree] @ kernel
            full_quotient = restrictions[degree] @ quotient
            nullspaces.append(full_kernel)
            cohomologies.append(full_quotient)
            nullity = kernel.shape[1]
            nullities.append(nullity)
            betti.append(nullity - (derivative_ranks[degree - 1] if degree > 0 else 0))

        self.complex_id = complex_.complex_id
        self.boundary_faces = tuple(validated_faces)
        self.restriction_bases = tuple(jnp.asarray(value) for value in restrictions)
        self.restricted_derivatives = tuple(
            jnp.asarray(value) for value in restricted_derivatives
        )
        self.nullspace_bases = tuple(jnp.asarray(value) for value in nullspaces)
        self.cohomology_bases = tuple(jnp.asarray(value) for value in cohomologies)
        self.derivative_ranks = derivative_ranks
        self.nullities = tuple(nullities)
        self.betti_numbers = tuple(betti)
        self.closure_defects = jnp.asarray(closure_defects)
        self.tolerance = tolerance_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "relative-cohomology-evidence",
                "complex": complex_.complex_id,
                "boundary_faces": [list(face) for face in faces],
                "restriction_bases": [_array_identity(value) for value in restrictions],
                "restricted_derivatives": [
                    _array_identity(value) for value in restricted_derivatives
                ],
                "nullspace_bases": [_array_identity(value) for value in nullspaces],
                "cohomology_bases": [_array_identity(value) for value in cohomologies],
                "derivative_ranks": list(derivative_ranks),
                "nullities": nullities,
                "betti_numbers": betti,
                "tolerance": tolerance_,
            }
        )

    @classmethod
    def full_boundary(
        cls,
        complex_: _SplineComplex,
        /,
        *,
        tolerance: float = 1e-12,
    ) -> RelativeCohomologyEvidence:
        faces = complex_.boundary_faces
        return cls(complex_, faces, tolerance=tolerance)


class CompatibleQualificationPolicy(StrictModule, NonTrainableState):
    """Fail-closed algebraic, topological, and stability gates for M1--M4."""

    expected_relative_betti: tuple[int, ...] = eqx.field(static=True)
    algebra_tolerance: float = eqx.field(static=True)
    maximum_friedrichs_constant: float = eqx.field(static=True)
    maximum_projector_norm: float = eqx.field(static=True)
    maximum_discrete_compactness_bound: float = eqx.field(static=True)

    def __init__(
        self,
        expected_relative_betti: Sequence[int],
        /,
        *,
        algebra_tolerance: float = 1e-11,
        maximum_friedrichs_constant: float = 1e8,
        maximum_projector_norm: float = 1e4,
        maximum_discrete_compactness_bound: float = 1e10,
    ) -> None:
        betti = tuple(expected_relative_betti)
        tolerance = float(algebra_tolerance)
        friedrichs = float(maximum_friedrichs_constant)
        projector = float(maximum_projector_norm)
        compactness = float(maximum_discrete_compactness_bound)
        if not betti or any(value < 0 for value in betti):
            raise ValueError("Expected relative Betti numbers must be nonnegative.")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (tolerance, friedrichs, projector, compactness)
        ):
            raise ValueError(
                "Compatible qualification bounds must be positive and finite."
            )
        self.expected_relative_betti = betti
        self.algebra_tolerance = tolerance
        self.maximum_friedrichs_constant = friedrichs
        self.maximum_projector_norm = projector
        self.maximum_discrete_compactness_bound = compactness

    @classmethod
    def contractible_full_boundary(
        cls, dimension: int, /
    ) -> CompatibleQualificationPolicy:
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Compatible qualification requires dimension two or three.")
        return cls((0,) * dimension_ + (1,))


class CompatibleQualificationEvidence(StrictModule, NonTrainableState):
    """Computed D², projector, relative-kernel, compactness, and Friedrichs evidence."""

    complex_id: str = eqx.field(static=True)
    projector_map_id: str = eqx.field(static=True)
    relative_evidence_id: str = eqx.field(static=True)
    numeric_revision: NumericRevision
    d_squared_defects: Array
    projector_commuting_defects: Array
    inclusion_commuting_defects: Array
    projector_retraction_defects: Array
    source_projection_defects: Array
    relative_closure_defects: Array
    relative_betti_numbers: tuple[int, ...] = eqx.field(static=True)
    complement_dimensions: tuple[int, ...] = eqx.field(static=True)
    minimum_complement_singular_values: Array
    friedrichs_constants: Array
    projector_operator_norms: Array
    finite_level_discrete_compactness_bounds: Array
    diagnostics: tuple[Diagnostic, ...]
    profile_codes: tuple[str, ...] = eqx.field(static=True)
    qualified: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        complex_id: str,
        projector_map_id: str,
        relative_evidence_id: str,
        numeric_revision: NumericRevision,
        d_squared_defects: ArrayLike,
        projector_commuting_defects: ArrayLike,
        inclusion_commuting_defects: ArrayLike,
        projector_retraction_defects: ArrayLike,
        source_projection_defects: ArrayLike,
        relative_closure_defects: ArrayLike,
        relative_betti_numbers: Sequence[int],
        complement_dimensions: Sequence[int],
        minimum_complement_singular_values: ArrayLike,
        friedrichs_constants: ArrayLike,
        projector_operator_norms: ArrayLike,
        finite_level_discrete_compactness_bounds: ArrayLike,
        diagnostics: Sequence[Diagnostic],
        qualified: bool,
    ) -> None:
        diagnostics_ = tuple(diagnostics)
        if any(not isinstance(value, Diagnostic) for value in diagnostics_):
            raise TypeError("Compatible evidence diagnostics have invalid types.")
        self.complex_id = str(complex_id)
        self.projector_map_id = str(projector_map_id)
        self.relative_evidence_id = str(relative_evidence_id)
        self.numeric_revision = numeric_revision
        self.d_squared_defects = jnp.asarray(d_squared_defects)
        self.projector_commuting_defects = jnp.asarray(projector_commuting_defects)
        self.inclusion_commuting_defects = jnp.asarray(inclusion_commuting_defects)
        self.projector_retraction_defects = jnp.asarray(projector_retraction_defects)
        self.source_projection_defects = jnp.asarray(source_projection_defects)
        self.relative_closure_defects = jnp.asarray(relative_closure_defects)
        self.relative_betti_numbers = tuple(relative_betti_numbers)
        self.complement_dimensions = tuple(complement_dimensions)
        self.minimum_complement_singular_values = jnp.asarray(
            minimum_complement_singular_values
        )
        self.friedrichs_constants = jnp.asarray(friedrichs_constants)
        self.projector_operator_norms = jnp.asarray(projector_operator_norms)
        self.finite_level_discrete_compactness_bounds = jnp.asarray(
            finite_level_discrete_compactness_bounds
        )
        self.diagnostics = diagnostics_
        self.profile_codes = ("M1", "M2", "M3", "M4")
        self.qualified = bool(qualified)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "compatible-qualification-evidence",
                "complex": self.complex_id,
                "projector": self.projector_map_id,
                "relative": self.relative_evidence_id,
                "numeric_revision": numeric_revision.revision_id,
                "d_squared": _array_identity(self.d_squared_defects),
                "projector_commuting": _array_identity(self.projector_commuting_defects),
                "relative_betti": list(self.relative_betti_numbers),
                "friedrichs": _array_identity(self.friedrichs_constants),
                "projector_norms": _array_identity(self.projector_operator_norms),
                "compactness": _array_identity(
                    self.finite_level_discrete_compactness_bounds
                ),
                "diagnostics": [value.diagnostic_id for value in diagnostics_],
                "qualified": self.qualified,
            }
        )


def _gate_diagnostic(
    code: str,
    passed: bool,
    message: str,
    entity_id: str,
    value: float,
    tolerance: float,
    /,
) -> Diagnostic:
    return Diagnostic(
        code,
        "info" if passed else "error",
        "qualification",
        message,
        entity_ids=(entity_id,),
        value=value,
        tolerance=tolerance,
        remediation=None if passed else "Regenerate the compatible complex and evidence.",
    )


def _complex_numeric_revision(complex_: _SplineComplex, /) -> NumericRevision:
    """Return the canonical revision of a complex's derivative and trace matrices."""
    return NumericRevision(
        SemanticProvenance(
            {
                "kind": "spline-de-rham-complex",
                "dimension": complex_.dimension,
                "dof_counts": complex_.dof_counts,
                "traces": tuple(trace.map_id for trace in complex_.boundary_traces),
            }
        ),
        {
            "differentials": tuple(
                _operator_matrix(operator)
                for operator in complex_.absolute_complex.differentials
            ),
            "boundary_traces": tuple(
                tuple(_operator_matrix(operator) for operator in trace.maps)
                for trace in complex_.boundary_traces
            ),
            "metric_grams": tuple(
                jax.vmap(complex_.hodge_star, in_axes=(None, 1), out_axes=1)(
                    degree, jnp.eye(count, dtype=jnp.float64)
                )
                for degree, count in enumerate(complex_.dof_counts)
            ),
        },
    )


def qualify_compatible_complex(
    complex_: _SplineComplex,
    projector: ComplexMap,
    relative: RelativeCohomologyEvidence,
    policy: CompatibleQualificationPolicy,
    /,
    *,
    inclusion: ComplexMap,
) -> CompatibleQualificationEvidence:
    """Produce fail-closed qualification evidence without publishing a profile.

    The evidence binds the canonical `NumericRevision` of the complex's exterior
    derivative and boundary-trace matrices.
    """
    if not isinstance(complex_, _SplineComplex):
        raise TypeError("Compatible qualification requires a spline de Rham complex.")
    if not isinstance(projector, ComplexMap) or not isinstance(inclusion, ComplexMap):
        raise TypeError(
            "Compatible qualification requires projection and inclusion ComplexMaps."
        )
    if not isinstance(relative, RelativeCohomologyEvidence):
        raise TypeError("Compatible qualification requires relative cohomology evidence.")
    if not isinstance(policy, CompatibleQualificationPolicy):
        raise TypeError("Compatible qualification requires a policy.")
    if projector.target.complex_id != complex_.complex_id:
        raise ValueError("Projector evidence belongs to another complex.")
    if (
        projector.degree_offset != 0
        or inclusion.degree_offset != 0
        or projector.source.top_degree != complex_.dimension
    ):
        raise ValueError(
            "Spline qualification requires degree-preserving maps of equal-length complexes."
        )
    if (
        inclusion.source.complex_id != projector.target.complex_id
        or inclusion.target.complex_id != projector.source.complex_id
    ):
        raise ValueError("Projection and inclusion must have reversed endpoints.")
    if relative.complex_id != complex_.complex_id:
        raise ValueError("Relative evidence belongs to another complex.")
    if len(policy.expected_relative_betti) != complex_.dimension + 1:
        raise ValueError("Expected relative Betti numbers have the wrong dimension.")

    tolerance = policy.algebra_tolerance
    d_squared = np.asarray(complex_.d_squared_defects, dtype=np.float64)
    projections = tuple(_operator_matrix(operator) for operator in projector.maps)
    inclusions = tuple(_operator_matrix(operator) for operator in inclusion.maps)
    projection_defects, inclusion_defects, retraction_defects, source_defects = (
        [],
        [],
        [],
        [],
    )
    for degree in range(complex_.dimension):
        source_d = _operator_matrix(projector.source.differential(degree))
        target_d = _operator_matrix(complex_.absolute_complex.differential(degree))
        projection_defects.append(
            _max_abs(target_d @ projections[degree] - projections[degree + 1] @ source_d)
        )
        inclusion_defects.append(
            _max_abs(source_d @ inclusions[degree] - inclusions[degree + 1] @ target_d)
        )
    for degree, (projection, embedding) in enumerate(
        zip(projections, inclusions, strict=True)
    ):
        retraction_defects.append(
            _max_abs(
                projection @ embedding
                - jnp.eye(complex_.dof_count(degree), dtype=jnp.float64)
            )
        )
        source_projection = embedding @ projection
        source_defects.append(
            _max_abs(source_projection @ source_projection - source_projection)
        )
    projector_defect = np.asarray(projection_defects, dtype=np.float64)
    inclusion_defect = np.asarray(inclusion_defects, dtype=np.float64)
    retraction_defect = np.asarray(retraction_defects, dtype=np.float64)
    source_projection_defect = np.asarray(source_defects, dtype=np.float64)
    relative_closure = np.asarray(relative.closure_defects, dtype=np.float64)

    minimum_singular_values: list[float] = []
    friedrichs_constants: list[float] = []
    complement_dimensions: list[int] = []
    for derivative in complex_.absolute_complex.differentials:
        singular_values = np.linalg.svd(
            np.asarray(_operator_matrix(derivative)), compute_uv=False
        )
        positive = singular_values[singular_values > tolerance]
        complement_dimensions.append(positive.size)
        if positive.size:
            minimum = float(np.min(positive))
            minimum_singular_values.append(minimum)
            friedrichs_constants.append(1.0 / minimum)
        else:
            minimum_singular_values.append(float("inf"))
            friedrichs_constants.append(0.0)
    projector_norms = [
        float(np.linalg.norm(np.asarray(value), ord=2)) for value in projections
    ]
    compactness_bounds = [
        projector_norms[degree] * friedrichs_constants[degree]
        for degree in range(complex_.dimension)
    ]

    maximum_algebra_defect = max(
        (
            float(np.max(d_squared, initial=0.0)),
            float(np.max(projector_defect, initial=0.0)),
            float(np.max(inclusion_defect, initial=0.0)),
            float(np.max(retraction_defect, initial=0.0)),
            float(np.max(source_projection_defect, initial=0.0)),
            float(np.max(relative_closure, initial=0.0)),
        )
    )
    algebra_passed = maximum_algebra_defect <= tolerance
    topology_passed = relative.betti_numbers == policy.expected_relative_betti
    maximum_friedrichs = max(friedrichs_constants, default=0.0)
    friedrichs_passed = maximum_friedrichs <= policy.maximum_friedrichs_constant
    maximum_projector = max(projector_norms, default=0.0)
    projector_passed = maximum_projector <= policy.maximum_projector_norm
    maximum_compactness = max(compactness_bounds, default=0.0)
    compactness_passed = maximum_compactness <= policy.maximum_discrete_compactness_bound
    diagnostics = (
        _gate_diagnostic(
            "IGA-COMPATIBLE-ALGEBRA",
            algebra_passed,
            "D squared, commuting, retraction, and relative-closure defects.",
            complex_.complex_id,
            maximum_algebra_defect,
            tolerance,
        ),
        _gate_diagnostic(
            "IGA-COMPATIBLE-RELATIVE",
            topology_passed,
            "Relative cohomology agrees with the declared domain pair.",
            complex_.complex_id,
            float(
                sum(
                    abs(a - b)
                    for a, b in zip(
                        relative.betti_numbers,
                        policy.expected_relative_betti,
                        strict=True,
                    )
                )
            ),
            0.0,
        ),
        _gate_diagnostic(
            "IGA-COMPATIBLE-FRIEDRICHS",
            friedrichs_passed,
            "Finite-level complement satisfies the declared Friedrichs bound.",
            complex_.complex_id,
            maximum_friedrichs,
            policy.maximum_friedrichs_constant,
        ),
        _gate_diagnostic(
            "IGA-COMPATIBLE-PROJECTOR",
            projector_passed,
            "Commuting projector operator norms satisfy the declared bound.",
            complex_.complex_id,
            maximum_projector,
            policy.maximum_projector_norm,
        ),
        _gate_diagnostic(
            "IGA-COMPATIBLE-COMPACTNESS",
            compactness_passed,
            "Finite-level discrete compact-complement bounds satisfy policy.",
            complex_.complex_id,
            maximum_compactness,
            policy.maximum_discrete_compactness_bound,
        ),
    )
    qualified = all(
        (
            algebra_passed,
            topology_passed,
            friedrichs_passed,
            projector_passed,
            compactness_passed,
        )
    )
    return CompatibleQualificationEvidence(
        complex_id=complex_.complex_id,
        projector_map_id=projector.map_id,
        relative_evidence_id=relative.evidence_id,
        numeric_revision=_complex_numeric_revision(complex_),
        d_squared_defects=d_squared,
        projector_commuting_defects=projector_defect,
        inclusion_commuting_defects=inclusion_defect,
        projector_retraction_defects=retraction_defect,
        source_projection_defects=source_projection_defect,
        relative_closure_defects=relative_closure,
        relative_betti_numbers=relative.betti_numbers,
        complement_dimensions=complement_dimensions,
        minimum_complement_singular_values=jnp.asarray(minimum_singular_values),
        friedrichs_constants=jnp.asarray(friedrichs_constants),
        projector_operator_norms=jnp.asarray(projector_norms),
        finite_level_discrete_compactness_bounds=jnp.asarray(compactness_bounds),
        diagnostics=diagnostics,
        qualified=qualified,
    )


__all__ = [
    "AssembledSplineDeRhamComplex",
    "BoundarySide",
    "CompatibleQualificationEvidence",
    "CompatibleQualificationPolicy",
    "RelativeCohomologyEvidence",
    "SplineDeRhamComplex",
    "SplineDifferentialSpace",
    "SplineFormComponent",
    "qualify_compatible_complex",
]
