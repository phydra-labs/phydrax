#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation._bspline_grid import BSplineGrid
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...diagnostics import Diagnostic
from ...lifecycle._models import NumericRevision


BoundarySide: TypeAlias = Literal["lower", "upper"]
PiolaKind: TypeAlias = Literal["h1", "hcurl", "hdiv", "l2"]


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


def _axis_difference(shape: tuple[int, ...], axis: int, /) -> np.ndarray:
    target_shape = tuple(
        size - 1 if index == axis else size for index, size in enumerate(shape)
    )
    matrix = np.zeros((prod(target_shape), prod(shape)), dtype=float)
    for target_index in np.ndindex(target_shape):
        lower = list(target_index)
        upper = list(target_index)
        upper[axis] += 1
        row = np.ravel_multi_index(target_index, target_shape)
        matrix[row, np.ravel_multi_index(tuple(lower), shape)] = -1.0
        matrix[row, np.ravel_multi_index(tuple(upper), shape)] = 1.0
    return matrix


def _face_restriction(
    shape: tuple[int, ...],
    axis: int,
    side: BoundarySide,
    orientation: int,
    /,
) -> np.ndarray:
    face_shape = shape[:axis] + shape[axis + 1 :]
    matrix = np.zeros((prod(face_shape), prod(shape)), dtype=float)
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
    column_count = int(matrix.shape[1])
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


class SplineFormComponent(StrictModule, NonTrainableState):
    """One polynomial tensor component of a spline differential-form space."""

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
    ):
        degree = int(form_degree)
        axes = tuple(int(axis) for axis in component_axes)
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
        shape = tuple(grid.coefficient_count for grid in grids_)
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
            }
        )


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
    ):
        degree = int(form_degree)
        dimension_ = int(dimension)
        components_ = tuple(components)
        expected_axes = tuple(combinations(range(dimension_), degree))
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
        axes = tuple(int(axis) for axis in component_axes)
        for offset, component in zip(
            self.component_offsets, self.components, strict=True
        ):
            if component.component_axes == axes:
                return slice(offset, offset + component.coefficient_count)
        raise ValueError(f"No component with axes {axes!r} belongs to this space.")


class SignedSplineTrace(StrictModule, NonTrainableState):
    """Oriented tangential pullback from one spline k-form space to a face."""

    form_degree: int = eqx.field(static=True)
    normal_axis: int = eqx.field(static=True)
    side: BoundarySide = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    source_dof_count: int = eqx.field(static=True)
    target_component_axes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    matrix: Array
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        form_degree: int,
        normal_axis: int,
        side: BoundarySide,
        source_dof_count: int,
        target_component_axes: Sequence[Sequence[int]],
        matrix: ArrayLike,
        /,
    ):
        degree = int(form_degree)
        axis = int(normal_axis)
        side_ = str(side)
        count = int(source_dof_count)
        target_axes = tuple(
            tuple(int(value) for value in axes) for axes in target_component_axes
        )
        matrix_ = jnp.asarray(matrix)
        if side_ not in ("lower", "upper"):
            raise ValueError("Spline trace side must be lower or upper.")
        if matrix_.ndim != 2 or matrix_.shape[1] != count:
            raise ValueError("Spline trace matrix has an invalid source dimension.")
        orientation = ((-1) ** axis) * (1 if side_ == "upper" else -1)
        self.form_degree = degree
        self.normal_axis = axis
        self.side = side_  # type: ignore[assignment]
        self.orientation = orientation
        self.source_dof_count = count
        self.target_component_axes = target_axes
        self.matrix = matrix_
        self.trace_id = canonical_fingerprint(
            {
                "kind": "signed-spline-trace",
                "form_degree": degree,
                "normal_axis": axis,
                "side": side_,
                "source_dof_count": count,
                "target_component_axes": [list(value) for value in target_axes],
                "matrix": _array_identity(matrix_),
            }
        )

    def apply(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape[-1] != self.source_dof_count:
            raise ValueError("Trace coefficients have the wrong trailing dimension.")
        return values @ jnp.swapaxes(self.matrix, -1, -2)


class AbstractSplineDeRhamComplex(StrictModule, NonTrainableState):
    """Common algebra carried by tensor and assembled multipatch complexes."""

    dimension: int = eqx.field(static=True)
    dof_counts: tuple[int, ...] = eqx.field(static=True)
    exterior_derivatives: tuple[Array, ...]
    boundary_traces: tuple[SignedSplineTrace, ...]
    complex_id: str = eqx.field(static=True)

    def exterior_derivative(self, form_degree: int, /) -> Array:
        degree = int(form_degree)
        if degree < 0 or degree >= self.dimension:
            raise ValueError("Exterior derivative degree lies outside the complex.")
        return self.exterior_derivatives[degree]

    def dof_count(self, form_degree: int, /) -> int:
        degree = int(form_degree)
        if degree < 0 or degree > self.dimension:
            raise ValueError("Form degree lies outside the complex.")
        return self.dof_counts[degree]

    def trace(
        self, form_degree: int, normal_axis: int, side: BoundarySide, /
    ) -> SignedSplineTrace:
        degree = int(form_degree)
        axis = int(normal_axis)
        side_ = str(side)
        for trace in self.boundary_traces:
            if (
                trace.form_degree == degree
                and trace.normal_axis == axis
                and trace.side == side_
            ):
                return trace
        raise ValueError("The requested boundary trace is not available.")

    @property
    def d_squared_defects(self) -> Array:
        defects = [
            _max_abs(right @ left)
            for left, right in zip(
                self.exterior_derivatives[:-1],
                self.exterior_derivatives[1:],
                strict=True,
            )
        ]
        return jnp.stack(defects) if defects else jnp.zeros((0,))


class SplineDeRhamComplex(AbstractSplineDeRhamComplex):
    """Exact 2D/3D tensor spline de Rham sequence with polynomial components."""

    base_grids: tuple[BSplineGrid, ...]
    spaces: tuple[SplineDifferentialSpace, ...]

    def __init__(self, grids: Sequence[BSplineGrid], /):
        grids_ = tuple(grids)
        dimension = len(grids_)
        if dimension not in (2, 3):
            raise ValueError("Spline de Rham complexes require dimension two or three.")
        if any(not isinstance(grid, BSplineGrid) for grid in grids_):
            raise TypeError("Spline de Rham axes must be BSplineGrid values.")
        if any(grid.degree < 1 for grid in grids_):
            raise ValueError("Spline de Rham axes require degree at least one.")
        reduced = tuple(_reduced_grid(grid) for grid in grids_)
        spaces: list[SplineDifferentialSpace] = []
        for form_degree in range(dimension + 1):
            components = []
            for component_axes in combinations(range(dimension), form_degree):
                component_grids = tuple(
                    reduced[axis] if axis in component_axes else grids_[axis]
                    for axis in range(dimension)
                )
                components.append(
                    SplineFormComponent(form_degree, component_axes, component_grids)
                )
            spaces.append(
                SplineDifferentialSpace(form_degree, dimension, tuple(components))
            )

        derivatives: list[Array] = []
        for form_degree in range(dimension):
            source = spaces[form_degree]
            target = spaces[form_degree + 1]
            derivative = np.zeros((target.dof_count, source.dof_count))
            for source_offset, component in zip(
                source.component_offsets, source.components, strict=True
            ):
                for axis in range(dimension):
                    if axis in component.component_axes:
                        continue
                    target_axes = tuple(sorted(component.component_axes + (axis,)))
                    target_slice = target.component_slice(target_axes)
                    sign = (-1) ** sum(
                        existing_axis < axis for existing_axis in component.component_axes
                    )
                    block = sign * _axis_difference(component.coefficient_shape, axis)
                    source_slice = slice(
                        source_offset,
                        source_offset + component.coefficient_count,
                    )
                    derivative[target_slice, source_slice] = block
            derivatives.append(jnp.asarray(derivative))

        traces: list[SignedSplineTrace] = []
        for form_degree, space in enumerate(spaces):
            for axis in range(dimension):
                for side in ("lower", "upper"):
                    orientation = ((-1) ** axis) * (1 if side == "upper" else -1)
                    tangential = tuple(
                        component
                        for component in space.components
                        if axis not in component.component_axes
                    )
                    row_count = sum(
                        prod(
                            component.coefficient_shape[:axis]
                            + component.coefficient_shape[axis + 1 :]
                        )
                        for component in tangential
                    )
                    matrix = np.zeros((row_count, space.dof_count))
                    row_offset = 0
                    for component in tangential:
                        source_slice = space.component_slice(component.component_axes)
                        block = _face_restriction(
                            component.coefficient_shape,
                            axis,
                            side,  # type: ignore[arg-type]
                            orientation,
                        )
                        matrix[row_offset : row_offset + block.shape[0], source_slice] = (
                            block
                        )
                        row_offset += block.shape[0]
                    target_axes = tuple(
                        tuple(
                            component_axis
                            for component_axis in component.component_axes
                            if component_axis != axis
                        )
                        for component in tangential
                    )
                    traces.append(
                        SignedSplineTrace(
                            form_degree,
                            axis,
                            side,  # type: ignore[arg-type]
                            space.dof_count,
                            target_axes,
                            matrix,
                        )
                    )

        identity = canonical_fingerprint(
            {
                "kind": "spline-de-rham-complex",
                "dimension": dimension,
                "base_degrees": [grid.degree for grid in grids_],
                "base_knots": [_array_identity(grid.knots) for grid in grids_],
                "spaces": [space.space_id for space in spaces],
                "derivatives": [_array_identity(value) for value in derivatives],
                "traces": [trace.trace_id for trace in traces],
            }
        )
        self.dimension = dimension
        self.dof_counts = tuple(space.dof_count for space in spaces)
        self.exterior_derivatives = tuple(derivatives)
        self.boundary_traces = tuple(traces)
        self.complex_id = identity
        self.base_grids = grids_
        self.spaces = tuple(spaces)

    def space(self, form_degree: int, /) -> SplineDifferentialSpace:
        degree = int(form_degree)
        if degree < 0 or degree > self.dimension:
            raise ValueError("Form degree lies outside the spline complex.")
        return self.spaces[degree]


class AssembledSplineDeRhamComplex(AbstractSplineDeRhamComplex):
    """A matching multipatch quotient of local spline de Rham complexes."""

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
        boundary_traces: Sequence[SignedSplineTrace] = (),
    ):
        dimension_ = int(dimension)
        counts = tuple(int(value) for value in dof_counts)
        derivatives = tuple(jnp.asarray(value) for value in exterior_derivatives)
        source_ids = tuple(str(value) for value in source_complex_ids)
        assembly_id_ = str(assembly_id).strip()
        traces = tuple(boundary_traces)
        if dimension_ not in (2, 3) or len(counts) != dimension_ + 1:
            raise ValueError("Assembled de Rham dimensions are invalid.")
        if len(derivatives) != dimension_:
            raise ValueError("Assembled de Rham derivative count is invalid.")
        for degree, derivative in enumerate(derivatives):
            if derivative.shape != (counts[degree + 1], counts[degree]):
                raise ValueError("Assembled exterior derivative shape is invalid.")
        if not source_ids or any(not value for value in source_ids):
            raise ValueError("Assembled de Rham source IDs must be non-empty.")
        if not assembly_id_:
            raise ValueError("Assembled de Rham assembly_id must be non-empty.")
        if any(not isinstance(trace, SignedSplineTrace) for trace in traces):
            raise TypeError("Assembled boundary traces must be SignedSplineTrace values.")
        self.dimension = dimension_
        self.dof_counts = counts
        self.exterior_derivatives = derivatives
        self.boundary_traces = traces
        self.source_complex_ids = source_ids
        self.assembly_id = assembly_id_
        self.complex_id = canonical_fingerprint(
            {
                "kind": "assembled-spline-de-rham-complex",
                "dimension": dimension_,
                "dof_counts": list(counts),
                "derivatives": [_array_identity(value) for value in derivatives],
                "source_complex_ids": list(source_ids),
                "assembly_id": assembly_id_,
                "traces": [trace.trace_id for trace in traces],
            }
        )


class SplinePiolaMap(StrictModule, NonTrainableState):
    """Dimension-correct scalar, covariant, contravariant, or density pullback."""

    dimension: int = eqx.field(static=True)
    kind: PiolaKind = eqx.field(static=True)

    def __init__(self, dimension: int, kind: PiolaKind, /):
        dimension_ = int(dimension)
        kind_ = str(kind)
        if dimension_ not in (2, 3):
            raise ValueError("Spline Piola maps require dimension two or three.")
        if kind_ not in ("h1", "hcurl", "hdiv", "l2"):
            raise ValueError("Unknown spline Piola map kind.")
        self.dimension = dimension_
        self.kind = kind_  # type: ignore[assignment]

    def push_forward(self, jacobian: ArrayLike, values: ArrayLike, /) -> Array:
        matrix = jnp.asarray(jacobian)
        values_ = jnp.asarray(values)
        if matrix.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError("Piola Jacobian has the wrong trailing dimensions.")
        if self.kind in ("hcurl", "hdiv") and values_.shape[-1] != self.dimension:
            raise ValueError("Vector Piola values have the wrong trailing dimension.")
        if self.kind == "h1":
            return values_
        determinant = jnp.linalg.det(matrix)
        if self.kind == "hcurl":
            return jnp.linalg.solve(jnp.swapaxes(matrix, -1, -2), values_[..., None])[
                ..., 0
            ]
        if self.kind == "hdiv":
            return (matrix @ values_[..., None])[..., 0] / determinant[..., None]
        if values_.shape == determinant.shape:
            return values_ / determinant
        if values_.shape[-1:] == (1,) and values_.shape[:-1] == determinant.shape:
            return values_ / determinant[..., None]
        raise ValueError("L2 Piola values must be scalar densities.")

    def pull_back(self, jacobian: ArrayLike, values: ArrayLike, /) -> Array:
        matrix = jnp.asarray(jacobian)
        values_ = jnp.asarray(values)
        if matrix.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError("Piola Jacobian has the wrong trailing dimensions.")
        if self.kind in ("hcurl", "hdiv") and values_.shape[-1] != self.dimension:
            raise ValueError("Vector Piola values have the wrong trailing dimension.")
        if self.kind == "h1":
            return values_
        determinant = jnp.linalg.det(matrix)
        if self.kind == "hcurl":
            return (jnp.swapaxes(matrix, -1, -2) @ values_[..., None])[..., 0]
        if self.kind == "hdiv":
            return (
                determinant[..., None]
                * jnp.linalg.solve(matrix, values_[..., None])[..., 0]
            )
        if values_.shape == determinant.shape:
            return determinant * values_
        if values_.shape[-1:] == (1,) and values_.shape[:-1] == determinant.shape:
            return determinant[..., None] * values_
        raise ValueError("L2 Piola values must be scalar densities.")


class CommutingProjectorContract(StrictModule, NonTrainableState):
    """Projection/retraction data between a source cochain complex and spline target."""

    target: AbstractSplineDeRhamComplex
    source_dof_counts: tuple[int, ...] = eqx.field(static=True)
    source_derivatives: tuple[Array, ...]
    projectors: tuple[Array, ...]
    inclusions: tuple[Array, ...]
    projection_commuting_defects: Array
    inclusion_commuting_defects: Array
    retraction_defects: Array
    source_projection_defects: Array
    source_id: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: AbstractSplineDeRhamComplex,
        source_dof_counts: Sequence[int],
        source_derivatives: Sequence[ArrayLike],
        projectors: Sequence[ArrayLike],
        inclusions: Sequence[ArrayLike],
        /,
        *,
        source_id: str,
    ):
        if not isinstance(target, AbstractSplineDeRhamComplex):
            raise TypeError(
                "Commuting projector target must be a spline de Rham complex."
            )
        source_counts = tuple(int(value) for value in source_dof_counts)
        source_d = tuple(jnp.asarray(value) for value in source_derivatives)
        projectors_ = tuple(jnp.asarray(value) for value in projectors)
        inclusions_ = tuple(jnp.asarray(value) for value in inclusions)
        source_id_ = str(source_id).strip()
        dimension = target.dimension
        if len(source_counts) != dimension + 1 or any(
            value < 1 for value in source_counts
        ):
            raise ValueError("Projector source dimensions are invalid.")
        if len(source_d) != dimension:
            raise ValueError("Projector source derivative count is invalid.")
        if len(projectors_) != dimension + 1 or len(inclusions_) != dimension + 1:
            raise ValueError("Projector/retraction count is invalid.")
        for degree, derivative in enumerate(source_d):
            if derivative.shape != (source_counts[degree + 1], source_counts[degree]):
                raise ValueError("Projector source derivative shape is invalid.")
        for degree, (projector, inclusion) in enumerate(
            zip(projectors_, inclusions_, strict=True)
        ):
            expected_projector = (target.dof_count(degree), source_counts[degree])
            expected_inclusion = (source_counts[degree], target.dof_count(degree))
            if (
                projector.shape != expected_projector
                or inclusion.shape != expected_inclusion
            ):
                raise ValueError("Projector or inclusion shape is invalid.")
        if not source_id_:
            raise ValueError("Commuting projector source_id must be non-empty.")

        projection_defects = []
        inclusion_defects = []
        for degree in range(dimension):
            projection_defects.append(
                _max_abs(
                    target.exterior_derivative(degree) @ projectors_[degree]
                    - projectors_[degree + 1] @ source_d[degree]
                )
            )
            inclusion_defects.append(
                _max_abs(
                    source_d[degree] @ inclusions_[degree]
                    - inclusions_[degree + 1] @ target.exterior_derivative(degree)
                )
            )
        retraction_defects = []
        source_projection_defects = []
        for degree in range(dimension + 1):
            target_identity = jnp.eye(target.dof_count(degree))
            retraction_defects.append(
                _max_abs(projectors_[degree] @ inclusions_[degree] - target_identity)
            )
            source_projection = inclusions_[degree] @ projectors_[degree]
            source_projection_defects.append(
                _max_abs(source_projection @ source_projection - source_projection)
            )
        self.target = target
        self.source_dof_counts = source_counts
        self.source_derivatives = source_d
        self.projectors = projectors_
        self.inclusions = inclusions_
        self.projection_commuting_defects = jnp.stack(projection_defects)
        self.inclusion_commuting_defects = jnp.stack(inclusion_defects)
        self.retraction_defects = jnp.stack(retraction_defects)
        self.source_projection_defects = jnp.stack(source_projection_defects)
        self.source_id = source_id_
        self.contract_id = canonical_fingerprint(
            {
                "kind": "commuting-projector-contract",
                "target": target.complex_id,
                "source_id": source_id_,
                "source_dof_counts": list(source_counts),
                "source_derivatives": [_array_identity(value) for value in source_d],
                "projectors": [_array_identity(value) for value in projectors_],
                "inclusions": [_array_identity(value) for value in inclusions_],
            }
        )

    @classmethod
    def identity(
        cls, target: AbstractSplineDeRhamComplex, /
    ) -> CommutingProjectorContract:
        counts = target.dof_counts
        identities = tuple(jnp.eye(count) for count in counts)
        return cls(
            target,
            counts,
            target.exterior_derivatives,
            identities,
            identities,
            source_id=target.complex_id,
        )

    def project(self, form_degree: int, values: ArrayLike, /) -> Array:
        degree = int(form_degree)
        if degree < 0 or degree > self.target.dimension:
            raise ValueError("Projection degree lies outside the complex.")
        values_ = jnp.asarray(values)
        if values_.shape[-1] != self.source_dof_counts[degree]:
            raise ValueError("Projected cochains have the wrong trailing dimension.")
        return values_ @ jnp.swapaxes(self.projectors[degree], -1, -2)


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

    def __init__(
        self,
        complex_: AbstractSplineDeRhamComplex,
        boundary_faces: Sequence[tuple[int, BoundarySide]],
        /,
        *,
        tolerance: float = 1e-12,
    ):
        if not isinstance(complex_, AbstractSplineDeRhamComplex):
            raise TypeError("Relative evidence requires a spline de Rham complex.")
        faces = tuple((int(axis), str(side)) for axis, side in boundary_faces)
        tolerance_ = float(tolerance)
        if tolerance_ <= 0.0 or not np.isfinite(tolerance_):
            raise ValueError("Relative cohomology tolerance must be positive and finite.")
        if len(set(faces)) != len(faces):
            raise ValueError("Relative boundary faces must be unique.")
        for axis, side in faces:
            if axis < 0 or axis >= complex_.dimension or side not in ("lower", "upper"):
                raise ValueError("Relative boundary face is invalid.")

        restrictions: list[np.ndarray] = []
        for degree in range(complex_.dimension + 1):
            matrices = [
                np.asarray(complex_.trace(degree, axis, side).matrix)
                for axis, side in faces
            ]
            trace_matrix = (
                np.concatenate(matrices, axis=0)
                if matrices
                else np.zeros((0, complex_.dof_count(degree)))
            )
            restrictions.append(_nullspace(trace_matrix, tolerance_))

        restricted_derivatives: list[np.ndarray] = []
        closure_defects: list[float] = []
        for degree, derivative in enumerate(complex_.exterior_derivatives):
            source_basis = restrictions[degree]
            target_basis = restrictions[degree + 1]
            derivative_host = np.asarray(derivative)
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
            nullity = int(kernel.shape[1])
            nullities.append(nullity)
            betti.append(nullity - (derivative_ranks[degree - 1] if degree > 0 else 0))

        self.complex_id = complex_.complex_id
        self.boundary_faces = tuple(
            (axis, side)
            for axis, side in faces  # type: ignore[misc]
        )
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
        complex_: AbstractSplineDeRhamComplex,
        /,
        *,
        tolerance: float = 1e-12,
    ) -> RelativeCohomologyEvidence:
        faces = tuple(
            (axis, side)
            for axis in range(complex_.dimension)
            for side in ("lower", "upper")
        )
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
    ):
        betti = tuple(int(value) for value in expected_relative_betti)
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
    projector_contract_id: str = eqx.field(static=True)
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

    def __init__(
        self,
        *,
        complex_id: str,
        projector_contract_id: str,
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
    ):
        if not isinstance(numeric_revision, NumericRevision):
            raise TypeError("Compatible evidence requires a NumericRevision.")
        diagnostics_ = tuple(diagnostics)
        if any(not isinstance(value, Diagnostic) for value in diagnostics_):
            raise TypeError("Compatible evidence diagnostics have invalid types.")
        self.complex_id = str(complex_id)
        self.projector_contract_id = str(projector_contract_id)
        self.relative_evidence_id = str(relative_evidence_id)
        self.numeric_revision = numeric_revision
        self.d_squared_defects = jnp.asarray(d_squared_defects)
        self.projector_commuting_defects = jnp.asarray(projector_commuting_defects)
        self.inclusion_commuting_defects = jnp.asarray(inclusion_commuting_defects)
        self.projector_retraction_defects = jnp.asarray(projector_retraction_defects)
        self.source_projection_defects = jnp.asarray(source_projection_defects)
        self.relative_closure_defects = jnp.asarray(relative_closure_defects)
        self.relative_betti_numbers = tuple(
            int(value) for value in relative_betti_numbers
        )
        self.complement_dimensions = tuple(int(value) for value in complement_dimensions)
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
                "projector": self.projector_contract_id,
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


def qualify_compatible_complex(
    complex_: AbstractSplineDeRhamComplex,
    projector: CommutingProjectorContract,
    relative: RelativeCohomologyEvidence,
    numeric_revision: NumericRevision,
    policy: CompatibleQualificationPolicy,
    /,
) -> CompatibleQualificationEvidence:
    """Produce fail-closed qualification evidence without publishing a profile."""

    if not isinstance(complex_, AbstractSplineDeRhamComplex):
        raise TypeError("Compatible qualification requires a spline de Rham complex.")
    if not isinstance(projector, CommutingProjectorContract):
        raise TypeError("Compatible qualification requires a projector contract.")
    if not isinstance(relative, RelativeCohomologyEvidence):
        raise TypeError("Compatible qualification requires relative cohomology evidence.")
    if not isinstance(numeric_revision, NumericRevision):
        raise TypeError("Compatible qualification requires a NumericRevision.")
    if not isinstance(policy, CompatibleQualificationPolicy):
        raise TypeError("Compatible qualification requires a policy.")
    if projector.target.complex_id != complex_.complex_id:
        raise ValueError("Projector evidence belongs to another complex.")
    if relative.complex_id != complex_.complex_id:
        raise ValueError("Relative evidence belongs to another complex.")
    if numeric_revision.content_digest != complex_.complex_id:
        raise ValueError("Numeric revision does not identify the compatible complex.")
    if len(policy.expected_relative_betti) != complex_.dimension + 1:
        raise ValueError("Expected relative Betti numbers have the wrong dimension.")

    tolerance = policy.algebra_tolerance
    d_squared = np.asarray(complex_.d_squared_defects, dtype=float)
    projector_defect = np.asarray(projector.projection_commuting_defects, dtype=float)
    inclusion_defect = np.asarray(projector.inclusion_commuting_defects, dtype=float)
    retraction_defect = np.asarray(projector.retraction_defects, dtype=float)
    source_projection_defect = np.asarray(
        projector.source_projection_defects, dtype=float
    )
    relative_closure = np.asarray(relative.closure_defects, dtype=float)

    minimum_singular_values: list[float] = []
    friedrichs_constants: list[float] = []
    complement_dimensions: list[int] = []
    for derivative in complex_.exterior_derivatives:
        singular_values = np.linalg.svd(np.asarray(derivative), compute_uv=False)
        positive = singular_values[singular_values > tolerance]
        complement_dimensions.append(int(positive.size))
        if positive.size:
            minimum = float(np.min(positive))
            minimum_singular_values.append(minimum)
            friedrichs_constants.append(1.0 / minimum)
        else:
            minimum_singular_values.append(float("inf"))
            friedrichs_constants.append(0.0)
    projector_norms = [
        float(np.linalg.norm(np.asarray(value), ord=2)) for value in projector.projectors
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
        projector_contract_id=projector.contract_id,
        relative_evidence_id=relative.evidence_id,
        numeric_revision=numeric_revision,
        d_squared_defects=d_squared,
        projector_commuting_defects=projector_defect,
        inclusion_commuting_defects=inclusion_defect,
        projector_retraction_defects=retraction_defect,
        source_projection_defects=source_projection_defect,
        relative_closure_defects=relative_closure,
        relative_betti_numbers=relative.betti_numbers,
        complement_dimensions=complement_dimensions,
        minimum_complement_singular_values=minimum_singular_values,
        friedrichs_constants=friedrichs_constants,
        projector_operator_norms=projector_norms,
        finite_level_discrete_compactness_bounds=compactness_bounds,
        diagnostics=diagnostics,
        qualified=qualified,
    )


__all__ = [
    "AbstractSplineDeRhamComplex",
    "AssembledSplineDeRhamComplex",
    "BoundarySide",
    "CommutingProjectorContract",
    "CompatibleQualificationEvidence",
    "CompatibleQualificationPolicy",
    "PiolaKind",
    "RelativeCohomologyEvidence",
    "SignedSplineTrace",
    "SplineDeRhamComplex",
    "SplineDifferentialSpace",
    "SplineFormComponent",
    "SplinePiolaMap",
    "qualify_compatible_complex",
]
