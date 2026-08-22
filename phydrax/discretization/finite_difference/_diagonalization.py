#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearTransform,
    ArraySpace,
    CompatibilityPolicy,
    FFTLinearTransform,
    FunctionLinearOperator,
    GaugePolicy,
    PreparedTransformDiagonalSolve,
    RealTrigonometricTransform,
    SimilarityScaledLinearTransform,
    TensorLinearTransform,
    TransformDiagonalRepresentation,
    TransformDiagonalSolvePlan,
    TransformDiagonalSolveResult,
)
from .._tensor_support import PreparedTensorGrid


FDBoundaryKind: TypeAlias = Literal["periodic", "dirichlet", "neumann"]
FDBoundaryPair: TypeAlias = tuple[FDBoundaryKind, FDBoundaryKind]


class FDTransformAxisReport(StrictModule, NonTrainableState):
    """Analytic certificate for one uniform three-point second difference."""

    axis: str = eqx.field(static=True)
    primary_entity: Literal["point", "interval"] = eqx.field(static=True)
    lower_boundary: FDBoundaryKind = eqx.field(static=True)
    upper_boundary: FDBoundaryKind = eqx.field(static=True)
    transform_family: Literal["fft", "dct", "dst"] = eqx.field(static=True)
    transform_type: int | None = eqx.field(static=True)
    full_count: int = eqx.field(static=True)
    unknown_count: int = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    nullspace_dimension: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        axis: str,
        primary_entity: Literal["point", "interval"],
        lower_boundary: FDBoundaryKind,
        upper_boundary: FDBoundaryKind,
        transform_family: Literal["fft", "dct", "dst"],
        transform_type: int | None,
        full_count: int,
        unknown_count: int,
        spacing: float,
        nullspace_dimension: int,
    ):
        self.axis = axis
        self.primary_entity = primary_entity
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.transform_family = transform_family
        self.transform_type = transform_type
        self.full_count = int(full_count)
        self.unknown_count = int(unknown_count)
        self.spacing = float(spacing)
        self.nullspace_dimension = int(nullspace_dimension)
        self.exact = True
        self.report_id = canonical_fingerprint(
            {
                "kind": "fd-transform-axis-report",
                "axis": axis,
                "primary_entity": primary_entity,
                "boundaries": [lower_boundary, upper_boundary],
                "transform": [transform_family, transform_type],
                "full_count": int(full_count),
                "unknown_count": int(unknown_count),
                "spacing": float(spacing),
                "nullspace_dimension": int(nullspace_dimension),
            }
        )


class _FDSecondDifferenceAxis(StrictModule, NonTrainableState):
    axis: int = eqx.field(static=True)
    primary_entity: Literal["point", "interval"] = eqx.field(static=True)
    lower_boundary: FDBoundaryKind = eqx.field(static=True)
    upper_boundary: FDBoundaryKind = eqx.field(static=True)
    full_count: int = eqx.field(static=True)
    unknown_count: int = eqx.field(static=True)
    spacing: float = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        primary_entity: Literal["point", "interval"],
        boundaries: FDBoundaryPair,
        full_count: int,
        unknown_count: int,
        spacing: float,
        /,
    ):
        self.axis = int(axis)
        self.primary_entity = primary_entity
        self.lower_boundary, self.upper_boundary = boundaries
        self.full_count = int(full_count)
        self.unknown_count = int(unknown_count)
        self.spacing = float(spacing)

    def apply(self, values: Array, /) -> Array:
        moved = jnp.moveaxis(values, self.axis, 0)
        inverse_spacing_squared = 1.0 / self.spacing**2
        if self.lower_boundary == "periodic":
            second = (
                jnp.roll(moved, -1, axis=0) - 2.0 * moved + jnp.roll(moved, 1, axis=0)
            )
            return jnp.moveaxis(inverse_spacing_squared * second, 0, self.axis)
        if self.primary_entity == "interval":
            lower_ghost = -moved[:1] if self.lower_boundary == "dirichlet" else moved[:1]
            upper_ghost = (
                -moved[-1:] if self.upper_boundary == "dirichlet" else moved[-1:]
            )
            left = jnp.concatenate((lower_ghost, moved[:-1]), axis=0)
            right = jnp.concatenate((moved[1:], upper_ghost), axis=0)
            second = left - 2.0 * moved + right
            return jnp.moveaxis(inverse_spacing_squared * second, 0, self.axis)
        zero = jnp.zeros_like(moved[:1])
        full = moved
        if self.lower_boundary == "dirichlet":
            full = jnp.concatenate((zero, full), axis=0)
        if self.upper_boundary == "dirichlet":
            full = jnp.concatenate((full, zero), axis=0)
        second = jnp.zeros_like(full)
        if self.full_count > 2:
            second = second.at[1:-1].set(full[:-2] - 2.0 * full[1:-1] + full[2:])
        if self.lower_boundary == "neumann":
            second = second.at[0].set(2.0 * (full[1] - full[0]))
        if self.upper_boundary == "neumann":
            second = second.at[-1].set(2.0 * (full[-2] - full[-1]))
        start = 1 if self.lower_boundary == "dirichlet" else 0
        stop = self.full_count - (1 if self.upper_boundary == "dirichlet" else 0)
        return jnp.moveaxis(inverse_spacing_squared * second[start:stop], 0, self.axis)


class FDLaplacianDiagonalization(StrictModule, NonTrainableState):
    """Certified tensor-product diagonalization of uniform FD2 Laplacians."""

    grid: PreparedTensorGrid
    boundaries: tuple[FDBoundaryPair, ...] = eqx.field(static=True)
    axis_actions: tuple[_FDSecondDifferenceAxis, ...]
    axis_reports: tuple[FDTransformAxisReport, ...]
    transforms: tuple[AbstractLinearTransform, ...]
    transform: TensorLinearTransform
    modal_values: Array
    space: ArraySpace
    operator: FunctionLinearOperator
    representation: TransformDiagonalRepresentation
    unknown_shape: tuple[int, ...] = eqx.field(static=True)
    unknown_coordinates: tuple[Array, ...]
    nullspace_dimension: int = eqx.field(static=True)
    diagonalization_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        boundaries: Mapping[str, FDBoundaryPair] | Sequence[FDBoundaryPair],
        /,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        pairs = _normalize_boundaries(grid, boundaries)
        dtype = np.dtype(grid.points.dtype)
        actions = []
        reports = []
        transforms = []
        spectra = []
        coordinates = []
        for axis_index, (axis_name, structured_axis, pair) in enumerate(
            zip(grid.axis_names, grid.structured_axes, pairs, strict=True)
        ):
            action, report, transform, spectrum, axis_coordinates = _prepare_axis(
                axis_index,
                axis_name,
                structured_axis,
                pair,
                dtype,
            )
            actions.append(action)
            reports.append(report)
            transforms.append(transform)
            spectra.append(spectrum)
            coordinates.append(axis_coordinates)
        transforms_ = tuple(transforms)
        tensor_transform = TensorLinearTransform(transforms_)
        unknown_shape = tuple(action.unknown_count for action in actions)
        modal_values = jnp.zeros(unknown_shape, dtype=dtype)
        for axis, spectrum in enumerate(spectra):
            shape = [1] * len(unknown_shape)
            shape[axis] = spectrum.size
            modal_values = modal_values + spectrum.reshape(shape)
        space = ArraySpace(unknown_shape, dtype=dtype)
        actions_ = tuple(actions)

        def laplacian_action(values):
            result = jnp.zeros_like(values)
            for action in actions_:
                result = result + action.apply(values)
            return result

        identifier = canonical_fingerprint(
            {
                "kind": "fd-laplacian-diagonalization",
                "grid": grid.prepared_id,
                "axis_reports": [report.report_id for report in reports],
                "transform": tensor_transform.transform_id,
            }
        )
        operator = FunctionLinearOperator(
            laplacian_action,
            source=space,
            target=space,
            operator_id=f"{identifier}:operator",
        )
        representation = TransformDiagonalRepresentation.from_transform(
            operator,
            modal_values,
            tensor_transform,
            representation_id=f"{identifier}:representation",
        )
        self.grid = grid
        self.boundaries = pairs
        self.axis_actions = actions_
        self.axis_reports = tuple(reports)
        self.transforms = transforms_
        self.transform = tensor_transform
        self.modal_values = modal_values
        self.space = space
        self.operator = operator
        self.representation = representation
        self.unknown_shape = unknown_shape
        self.unknown_coordinates = tuple(coordinates)
        self.nullspace_dimension = int(
            all(report.nullspace_dimension == 1 for report in reports)
        )
        self.diagonalization_id = identifier

    def apply(
        self,
        values: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ) -> Array:
        value = self.space.validate(jnp.asarray(values))
        return self.operator.mv(value) + self.boundary_forcing(boundary_values)

    def boundary_forcing(
        self,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
        /,
    ) -> Array:
        values = {} if boundary_values is None else dict(boundary_values)
        unknown = set(values).difference(self.grid.axis_names)
        if unknown:
            raise ValueError(f"Boundary data names unknown axes {sorted(unknown)!r}.")
        result = jnp.zeros(self.unknown_shape, dtype=self.space.dtype)
        for action, axis_name in zip(
            self.axis_actions, self.grid.axis_names, strict=True
        ):
            if action.lower_boundary == "periodic":
                if axis_name in values:
                    raise ValueError(
                        "Periodic axes do not accept physical boundary data."
                    )
                continue
            lower, upper = values.get(axis_name, (0.0, 0.0))
            boundary_shape = (
                self.unknown_shape[: action.axis] + self.unknown_shape[action.axis + 1 :]
            )
            lower_value = _broadcast_boundary(lower, boundary_shape, result.dtype)
            upper_value = _broadcast_boundary(upper, boundary_shape, result.dtype)
            moved = jnp.moveaxis(result, action.axis, 0)
            lower_scale, upper_scale = _boundary_scales(action)
            moved = moved.at[0].add(lower_scale * lower_value)
            moved = moved.at[-1].add(upper_scale * upper_value)
            result = jnp.moveaxis(moved, 0, action.axis)
        return result


class FDLaplacianSolvePlan(StrictModule, NonTrainableState):
    """Reusable direct solve for a scaled, shifted certified FD Laplacian."""

    diagonalization: FDLaplacianDiagonalization
    representation: TransformDiagonalRepresentation
    prepared: PreparedTransformDiagonalSolve
    operator_scale: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        diagonalization: FDLaplacianDiagonalization,
        /,
        *,
        operator_scale: ArrayLike = 1.0,
        diagonal_shift: ArrayLike = 0.0,
        compatibility: CompatibilityPolicy = "error",
        gauge: GaugePolicy = "minimum_norm",
        zero_tolerance: float = 1e-10,
    ):
        if not isinstance(diagonalization, FDLaplacianDiagonalization):
            raise TypeError("diagonalization must be FDLaplacianDiagonalization.")
        scale = jnp.asarray(operator_scale, dtype=diagonalization.space.dtype)
        shift = jnp.asarray(diagonal_shift, dtype=diagonalization.space.dtype)
        if (
            scale.shape != ()
            or not bool(np.isfinite(np.asarray(scale)))
            or bool(np.asarray(scale) == 0)
        ):
            raise ValueError("operator_scale must be one finite nonzero scalar.")
        if shift.shape != () or not bool(np.isfinite(np.asarray(shift))):
            raise ValueError("diagonal_shift must be one finite scalar.")
        identifier = canonical_fingerprint(
            {
                "kind": "fd-laplacian-solve-plan",
                "diagonalization": diagonalization.diagonalization_id,
                "operator_scale": repr(np.asarray(scale).item()),
                "diagonal_shift": repr(np.asarray(shift).item()),
                "compatibility": compatibility,
                "gauge": gauge,
                "zero_tolerance": float(zero_tolerance),
            }
        )
        operator = diagonalization.operator * scale
        representation = TransformDiagonalRepresentation.from_transform(
            operator,
            scale * diagonalization.modal_values,
            diagonalization.transform,
            representation_id=f"{identifier}:representation",
        )
        prepared = TransformDiagonalSolvePlan(
            representation,
            diagonal_shift=shift,
            compatibility=compatibility,
            gauge=gauge,
            zero_tolerance=zero_tolerance,
            plan_id=f"{identifier}:transform-solve",
        ).prepare()
        self.diagonalization = diagonalization
        self.representation = representation
        self.prepared = prepared
        self.operator_scale = scale
        self.plan_id = identifier

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ) -> TransformDiagonalSolveResult:
        rhs = self.representation.operator.target.validate(jnp.asarray(right_hand_side))
        forcing = self.diagonalization.boundary_forcing(boundary_values)
        return self.prepared.solve(rhs - self.operator_scale * forcing)


def _broadcast_boundary(value: ArrayLike, shape: tuple[int, ...], dtype: Any, /) -> Array:
    data = jnp.asarray(value, dtype=dtype)
    if data.shape == ():
        return jnp.broadcast_to(data, shape)
    if data.shape != shape:
        raise ValueError(f"Boundary data must have shape {shape} or be scalar.")
    return data


def _boundary_scales(action: _FDSecondDifferenceAxis, /) -> tuple[float, float]:
    spacing = action.spacing
    if action.primary_entity == "interval":
        lower = (
            2.0 / spacing**2 if action.lower_boundary == "dirichlet" else -1.0 / spacing
        )
        upper = (
            2.0 / spacing**2 if action.upper_boundary == "dirichlet" else 1.0 / spacing
        )
        return lower, upper
    lower = 1.0 / spacing**2 if action.lower_boundary == "dirichlet" else -2.0 / spacing
    upper = 1.0 / spacing**2 if action.upper_boundary == "dirichlet" else 2.0 / spacing
    return lower, upper


def _normalize_boundaries(
    grid: PreparedTensorGrid,
    boundaries: Mapping[str, FDBoundaryPair] | Sequence[FDBoundaryPair],
    /,
) -> tuple[FDBoundaryPair, ...]:
    if isinstance(boundaries, Mapping):
        unknown = set(boundaries).difference(grid.axis_names)
        if unknown:
            raise ValueError(
                f"Boundary mapping contains unknown axes {sorted(unknown)!r}."
            )
        raw = tuple(boundaries.get(name) for name in grid.axis_names)
    else:
        raw = tuple(boundaries)
        if len(raw) != len(grid.axis_names):
            raise ValueError("Boundary pairs must align with every grid axis.")
    output = []
    for axis_name, axis, value in zip(
        grid.axis_names, grid.structured_axes, raw, strict=True
    ):
        if value is None:
            if axis.periodic:
                pair: FDBoundaryPair = ("periodic", "periodic")
            else:
                raise ValueError(f"Bounded axis {axis_name!r} requires two boundaries.")
        else:
            pair = tuple(value)
            if len(pair) != 2 or any(
                condition not in ("periodic", "dirichlet", "neumann")
                for condition in pair
            ):
                raise ValueError("FD boundary pairs require periodic/Dirichlet/Neumann.")
            pair = (pair[0], pair[1])
        if axis.periodic != (pair == ("periodic", "periodic")):
            raise ValueError("Periodic axis metadata and boundary pair must agree.")
        if (pair[0] == "periodic") != (pair[1] == "periodic"):
            raise ValueError("Periodicity must be declared on both sides.")
        output.append(pair)
    return tuple(output)


def _uniform_spacing(axis, /) -> tuple[np.ndarray, float]:
    coordinates = np.asarray(
        axis.interval_centers
        if axis.primary_entity == "interval"
        else axis.point_coordinates,
        dtype=float,
    )
    if axis.primary_entity == "interval" or axis.periodic:
        widths = np.asarray(axis.interval_widths, dtype=float)
        spacing = float(widths[0])
        uniform = np.allclose(widths, spacing, rtol=1e-10, atol=1e-12)
    else:
        differences = np.diff(coordinates)
        spacing = float(differences[0])
        uniform = np.allclose(differences, spacing, rtol=1e-10, atol=1e-12)
    if not uniform or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Certified FD diagonalization requires uniform axis spacing.")
    return coordinates, spacing


def _prepare_axis(
    axis_index: int,
    axis_name: str,
    axis,
    boundaries: FDBoundaryPair,
    dtype: np.dtype,
    /,
):
    coordinates, spacing = _uniform_spacing(axis)
    full_count = int(coordinates.size)
    lower, upper = boundaries
    lower_dirichlet = lower == "dirichlet"
    upper_dirichlet = upper == "dirichlet"
    unknown_count = (
        full_count
        if axis.primary_entity == "interval" or lower == "periodic"
        else full_count - int(lower_dirichlet) - int(upper_dirichlet)
    )
    if unknown_count < 1:
        raise ValueError("Boundary elimination leaves no finite-difference unknowns.")
    if lower == "periodic":
        family: Literal["fft", "dct", "dst"] = "fft"
        transform_type = None
        transform: AbstractLinearTransform = FFTLinearTransform(
            unknown_count,
            dtype=np.result_type(dtype, np.complex64),
        )
        angles = 2.0 * np.pi * np.arange(unknown_count) / unknown_count
    elif axis.primary_entity == "interval":
        if boundaries == ("dirichlet", "dirichlet"):
            family, transform_type = "dst", 2
            angles = (np.arange(unknown_count) + 1.0) * np.pi / unknown_count
        elif boundaries == ("neumann", "neumann"):
            family, transform_type = "dct", 2
            angles = np.arange(unknown_count) * np.pi / unknown_count
        elif boundaries == ("dirichlet", "neumann"):
            family, transform_type = "dst", 4
            angles = (np.arange(unknown_count) + 0.5) * np.pi / unknown_count
        else:
            family, transform_type = "dct", 4
            angles = (np.arange(unknown_count) + 0.5) * np.pi / unknown_count
        transform = RealTrigonometricTransform(
            family,
            transform_type,
            unknown_count,
            dtype=dtype,
        )
    elif boundaries == ("dirichlet", "dirichlet"):
        family, transform_type = "dst", 1
        transform = RealTrigonometricTransform(
            family,
            transform_type,
            unknown_count,
            dtype=dtype,
        )
        angles = (np.arange(unknown_count) + 1.0) * np.pi / (unknown_count + 1.0)
    else:
        scaling = np.ones((unknown_count,), dtype=dtype)
        if boundaries == ("neumann", "neumann"):
            family, transform_type = "dct", 1
            scaling[[0, -1]] = 1.0 / np.sqrt(2.0)
            angles = np.arange(unknown_count) * np.pi / (unknown_count - 1.0)
        elif boundaries == ("dirichlet", "neumann"):
            family, transform_type = "dst", 3
            scaling[-1] = 1.0 / np.sqrt(2.0)
            angles = (np.arange(unknown_count) + 0.5) * np.pi / unknown_count
        else:
            family, transform_type = "dct", 3
            scaling[0] = 1.0 / np.sqrt(2.0)
            angles = (np.arange(unknown_count) + 0.5) * np.pi / unknown_count
        base = RealTrigonometricTransform(
            family,
            transform_type,
            unknown_count,
            dtype=dtype,
        )
        transform = SimilarityScaledLinearTransform(base, scaling)
    spectrum = -4.0 * np.sin(0.5 * angles) ** 2 / spacing**2
    unknown_coordinates = (
        coordinates
        if axis.primary_entity == "interval" or lower == "periodic"
        else coordinates[int(lower_dirichlet) : full_count - int(upper_dirichlet)]
    )
    nullspace = int(boundaries in (("periodic", "periodic"), ("neumann", "neumann")))
    action = _FDSecondDifferenceAxis(
        axis_index,
        axis.primary_entity,
        boundaries,
        full_count,
        unknown_count,
        spacing,
    )
    report = FDTransformAxisReport(
        axis=axis_name,
        primary_entity=axis.primary_entity,
        lower_boundary=lower,
        upper_boundary=upper,
        transform_family=family,
        transform_type=transform_type,
        full_count=full_count,
        unknown_count=unknown_count,
        spacing=spacing,
        nullspace_dimension=nullspace,
    )
    return (
        action,
        report,
        transform,
        jnp.asarray(spectrum, dtype=dtype),
        jnp.asarray(unknown_coordinates, dtype=dtype),
    )


def diagonalize_fd_laplacian(
    grid: PreparedTensorGrid,
    boundaries: Mapping[str, FDBoundaryPair] | Sequence[FDBoundaryPair],
    /,
) -> FDLaplacianDiagonalization:
    """Build an exact FFT/DCT/DST diagonalization from entity and BC semantics."""
    return FDLaplacianDiagonalization(grid, boundaries)


def solve_fd_laplacian(
    diagonalization: FDLaplacianDiagonalization,
    right_hand_side: ArrayLike,
    /,
    *,
    boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    operator_scale: ArrayLike = 1.0,
    diagonal_shift: ArrayLike = 0.0,
    compatibility: CompatibilityPolicy = "error",
    gauge: GaugePolicy = "minimum_norm",
    zero_tolerance: float = 1e-10,
) -> TransformDiagonalSolveResult:
    """Solve one scaled/shifted FD Laplacian with explicit compatibility policy."""
    plan = FDLaplacianSolvePlan(
        diagonalization,
        operator_scale=operator_scale,
        diagonal_shift=diagonal_shift,
        compatibility=compatibility,
        gauge=gauge,
        zero_tolerance=zero_tolerance,
    )
    return plan.solve(right_hand_side, boundary_values=boundary_values)


__all__ = [
    "diagonalize_fd_laplacian",
    "solve_fd_laplacian",
    "FDBoundaryKind",
    "FDBoundaryPair",
    "FDLaplacianDiagonalization",
    "FDLaplacianSolvePlan",
    "FDTransformAxisReport",
]
