#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._rod_dynamics import PreparedRod


RodStrainBasisKind: TypeAlias = Literal[
    "explicit", "piecewise_constant", "shifted_legendre"
]

_STRAIN_COMPONENTS = ("nu_x", "nu_y", "nu_z", "kappa_x", "kappa_y", "kappa_z")
_COMPONENT_INDEX = {name: index for index, name in enumerate(_STRAIN_COMPONENTS)}
_PLANAR_COMPONENTS = (0, 1, 5)
_SPATIAL_COMPONENTS = tuple(range(6))


def _real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _dimension(value: int, /) -> int:
    dimension = int(value)
    if dimension not in (2, 3):
        raise ValueError("Rod strain bases require dimension 2 or 3.")
    return dimension


def _component_indices(
    dimension: int,
    components: Sequence[int | str] | None,
    /,
) -> tuple[int, ...]:
    allowed = _PLANAR_COMPONENTS if dimension == 2 else _SPATIAL_COMPONENTS
    if components is None:
        return allowed
    indices: list[int] = []
    for component in components:
        if isinstance(component, str):
            try:
                index = _COMPONENT_INDEX[component]
            except KeyError as error:
                raise ValueError(
                    f"Unknown material strain component {component!r}."
                ) from error
        else:
            index = int(component)
        if index not in allowed:
            raise ValueError(
                f"Material strain component {index} is unavailable in dimension {dimension}."
            )
        indices.append(index)
    result = tuple(indices)
    if not result or len(set(result)) != len(result):
        raise ValueError("components must be nonempty and contain no duplicates.")
    return result


def _component_scale_array(
    value: ArrayLike | None,
    components: tuple[int, ...],
    dtype: np.dtype,
    /,
) -> np.ndarray:
    if value is None:
        return np.ones((6,), dtype=dtype)
    scale = _real_array("component_scales", value, 1)
    if scale.shape == (6,):
        result = scale.astype(dtype, copy=False)
    elif scale.shape == (len(components),):
        result = np.ones((6,), dtype=dtype)
        result[np.asarray(components)] = scale.astype(dtype, copy=False)
    else:
        raise ValueError(
            "component_scales must contain six canonical values or one value per selected component."
        )
    if np.any(result <= 0.0):
        raise ValueError("component_scales must be strictly positive.")
    return result


def _breakpoint_array(value: ArrayLike, dtype: np.dtype, /) -> np.ndarray:
    points = _real_array("breakpoints", value, 1).astype(dtype, copy=False)
    if points.shape[0] < 2 or np.any(np.diff(points) <= 0.0):
        raise ValueError("breakpoints must be a strictly increasing rank-1 array.")
    tolerance = 32.0 * np.finfo(dtype).eps
    if not np.isclose(points[0], 0.0, rtol=0.0, atol=tolerance) or not np.isclose(
        points[-1], 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError("Normalized rod basis breakpoints must begin at 0 and end at 1.")
    result = points.copy()
    result[0] = 0.0
    result[-1] = 1.0
    return result


def _shifted_legendre_coefficients(degree: int, dtype: np.dtype, /) -> np.ndarray:
    coefficients = np.zeros((degree + 1, degree + 1), dtype=dtype)
    coefficients[0, 0] = 1.0
    if degree == 0:
        return coefficients
    coefficients[1, :2] = (-1.0, 2.0)
    shifted_x = np.asarray((-1.0, 2.0), dtype=dtype)
    for order in range(1, degree):
        product = np.polynomial.polynomial.polymul(
            coefficients[order, : order + 1], shifted_x
        )
        next_values = (
            (2 * order + 1) * product
            - order
            * np.pad(
                coefficients[order - 1, :order],
                (0, product.shape[0] - order),
            )
        ) / (order + 1)
        coefficients[order + 1, : order + 2] = next_values
    return coefficients


class RodStrainBasisPlan(StrictModule, NonTrainableState):
    """Finite material-strain basis over normalized rod arc length.

    Every basis value has the canonical component ordering
    ``[nu_x, nu_y, nu_z, kappa_x, kappa_y, kappa_z]``. Reduced coordinates
    are dimensionless; ``component_scales`` therefore carry the physical scale
    of every populated basis column. Polynomial coefficients use the local
    coordinate of each normalized breakpoint interval.
    """

    breakpoints: Array
    polynomial_coefficients: Array
    component_scales: Array
    component_indices: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    basis_kind: RodStrainBasisKind = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        breakpoints: ArrayLike,
        polynomial_coefficients: ArrayLike,
        /,
        *,
        component_scales: ArrayLike | None = None,
        components: Sequence[int | str] | None = None,
        quadrature_order: int = 4,
        rank_tolerance: float | None = None,
        maximum_condition_number: float = 1.0e8,
        basis_kind: RodStrainBasisKind = "explicit",
        label: str | None = None,
    ):
        dimension_ = _dimension(dimension)
        indices = _component_indices(dimension_, components)
        raw_coefficients = np.asarray(polynomial_coefficients)
        if raw_coefficients.ndim == 3:
            raw_coefficients = raw_coefficients[..., None]
        coefficients = _real_array("polynomial_coefficients", raw_coefficients, 4)
        if (
            coefficients.shape[0] < 1
            or coefficients.shape[1] != 6
            or coefficients.shape[2] < 1
            or coefficients.shape[3] < 1
        ):
            raise ValueError(
                "polynomial_coefficients must have shape "
                "(intervals, 6, coordinates, polynomial_coefficients)."
            )
        dtype = np.dtype(jnp.asarray(coefficients).dtype)
        coefficients = coefficients.astype(dtype, copy=False)
        points = _breakpoint_array(breakpoints, dtype)
        if coefficients.shape[0] != points.shape[0] - 1:
            raise ValueError(
                "polynomial_coefficients must contain one block per breakpoint interval."
            )
        inactive = tuple(index for index in range(6) if index not in indices)
        if inactive and np.any(coefficients[:, inactive, :, :] != 0.0):
            raise ValueError(
                "Polynomial coefficients populate material components not declared by components."
            )
        scales = _component_scale_array(component_scales, indices, dtype)
        order = int(quadrature_order)
        if order < 1:
            raise ValueError("quadrature_order must be positive.")
        rank = (
            float(max(coefficients.shape) * np.finfo(dtype).eps)
            if rank_tolerance is None
            else float(rank_tolerance)
        )
        condition = float(maximum_condition_number)
        if (
            not isfinite(rank)
            or rank <= 0.0
            or not isfinite(condition)
            or condition < 1.0
        ):
            raise ValueError(
                "Basis rank tolerance and maximum condition number must be finite and positive."
            )
        if basis_kind not in (
            "explicit",
            "piecewise_constant",
            "shifted_legendre",
        ):
            raise ValueError("Unknown rod strain basis kind.")
        arrays = {
            "breakpoints": points,
            "polynomial_coefficients": coefficients,
            "component_scales": scales,
        }
        identifier = canonical_fingerprint(
            {
                "kind": "rod-strain-basis-plan",
                "basis_kind": basis_kind,
                "dimension": dimension_,
                "components": list(indices),
                "coordinate_count": int(coefficients.shape[2]),
                "polynomial_degree": int(coefficients.shape[3] - 1),
                "quadrature_order": order,
                "rank_tolerance": rank,
                "maximum_condition_number": condition,
                "values": array_tree_fingerprint(arrays),
            }
        )
        self.breakpoints = jnp.asarray(points)
        self.polynomial_coefficients = jnp.asarray(coefficients)
        self.component_scales = jnp.asarray(scales)
        self.component_indices = indices
        self.dimension = dimension_
        self.coordinate_count = int(coefficients.shape[2])
        self.polynomial_degree = int(coefficients.shape[3] - 1)
        self.quadrature_order = order
        self.rank_tolerance = rank
        self.maximum_condition_number = condition
        self.basis_kind = basis_kind
        self.label = None if label is None else str(label)
        self.plan_id = identifier

    @classmethod
    def explicit(
        cls,
        breakpoints: ArrayLike,
        polynomial_coefficients: ArrayLike,
        *,
        dimension: int,
        component_scales: ArrayLike | None = None,
        components: Sequence[int | str] | None = None,
        quadrature_order: int = 4,
        rank_tolerance: float | None = None,
        maximum_condition_number: float = 1.0e8,
        label: str | None = None,
    ) -> "RodStrainBasisPlan":
        """Construct an explicit piecewise-polynomial finite strain basis."""
        return cls(
            dimension,
            breakpoints,
            polynomial_coefficients,
            component_scales=component_scales,
            components=components,
            quadrature_order=quadrature_order,
            rank_tolerance=rank_tolerance,
            maximum_condition_number=maximum_condition_number,
            basis_kind="explicit",
            label=label,
        )

    @classmethod
    def piecewise_constant(
        cls,
        breakpoints: ArrayLike,
        *,
        dimension: int,
        components: Sequence[int | str] | None = None,
        component_scales: ArrayLike | None = None,
        quadrature_order: int = 1,
        rank_tolerance: float | None = None,
        maximum_condition_number: float = 1.0e8,
        label: str | None = None,
    ) -> "RodStrainBasisPlan":
        """Construct a PCS basis with one coordinate per component and interval."""
        dimension_ = _dimension(dimension)
        indices = _component_indices(dimension_, components)
        raw_points = _real_array("breakpoints", breakpoints, 1)
        dtype = raw_points.dtype
        interval_count = int(raw_points.shape[0] - 1)
        if interval_count < 1:
            raise ValueError("piecewise_constant requires at least one interval.")
        coordinate_count = len(indices) * interval_count
        coefficients = np.zeros((interval_count, 6, coordinate_count, 1), dtype=dtype)
        coordinate = 0
        for component in indices:
            for interval in range(interval_count):
                coefficients[interval, component, coordinate, 0] = 1.0
                coordinate += 1
        return cls(
            dimension_,
            raw_points,
            coefficients,
            component_scales=component_scales,
            components=indices,
            quadrature_order=quadrature_order,
            rank_tolerance=rank_tolerance,
            maximum_condition_number=maximum_condition_number,
            basis_kind="piecewise_constant",
            label=label,
        )

    @classmethod
    def shifted_legendre(
        cls,
        degree: int | Sequence[int],
        *,
        dimension: int,
        components: Sequence[int | str] | None = None,
        component_scales: ArrayLike | None = None,
        quadrature_order: int | None = None,
        rank_tolerance: float | None = None,
        maximum_condition_number: float = 1.0e8,
        label: str | None = None,
    ) -> "RodStrainBasisPlan":
        """Construct a GVS basis from shifted Legendre modes on ``[0, 1]``."""
        dimension_ = _dimension(dimension)
        indices = _component_indices(dimension_, components)
        if isinstance(degree, Sequence):
            degrees = tuple(int(value) for value in degree)
            if len(degrees) != len(indices):
                raise ValueError("degree must contain one value per selected component.")
        else:
            degrees = (int(degree),) * len(indices)
        if any(value < 0 for value in degrees):
            raise ValueError("Shifted-Legendre degrees must be nonnegative.")
        maximum_degree = max(degrees)
        coordinate_count = sum(value + 1 for value in degrees)
        dtype = np.asarray(
            np.zeros((), dtype=np.float64)
            if component_scales is None
            else component_scales
        ).dtype
        if not np.issubdtype(dtype, np.inexact) or np.issubdtype(
            dtype, np.complexfloating
        ):
            dtype = np.dtype(np.float64)
        mode_coefficients = _shifted_legendre_coefficients(maximum_degree, dtype)
        coefficients = np.zeros((1, 6, coordinate_count, maximum_degree + 1), dtype=dtype)
        coordinate = 0
        for component, component_degree in zip(indices, degrees, strict=True):
            for mode in range(component_degree + 1):
                coefficients[0, component, coordinate, : mode + 1] = mode_coefficients[
                    mode, : mode + 1
                ]
                coordinate += 1
        order = (
            max(1, maximum_degree + 1)
            if quadrature_order is None
            else int(quadrature_order)
        )
        return cls(
            dimension_,
            np.asarray((0.0, 1.0), dtype=dtype),
            coefficients,
            component_scales=component_scales,
            components=indices,
            quadrature_order=order,
            rank_tolerance=rank_tolerance,
            maximum_condition_number=maximum_condition_number,
            basis_kind="shifted_legendre",
            label=label,
        )

    def evaluate_normalized(self, coordinates: ArrayLike, /) -> Array:
        """Evaluate all six canonical material components at normalized sites."""
        values = jnp.asarray(coordinates, dtype=self.breakpoints.dtype)
        return _evaluate_normalized(self, values)


class RodStrainBasisEvidence(StrictModule):
    """Weighted discrete observability evidence for one prepared basis."""

    singular_values: Array
    numerical_rank: Array
    rank_threshold: Array
    condition_number: Array
    full_column_rank: Array
    condition_valid: Array
    dtype_retained: Array
    finite: Array
    valid: Array
    weighted_row_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    source_dtype: str = eqx.field(static=True)
    prepared_dtype: str = eqx.field(static=True)


class PreparedRodStrainBasis(StrictModule, NonTrainableState):
    """Rod-bound native worksets and weighted certification for a strain basis."""

    plan: RodStrainBasisPlan
    rod: PreparedRod
    physical_breakpoints: Array
    stretch_arc_lengths: Array
    bend_arc_lengths: Array
    stretch_interval_ids: Array
    bend_interval_ids: Array
    quadrature_arc_lengths: Array
    quadrature_weights: Array
    quadrature_interval_ids: Array
    stretch_shear_basis: Array
    bend_twist_basis: Array
    quadrature_basis: Array
    evidence: RodStrainBasisEvidence
    total_length: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def breakpoints(self) -> Array:
        return self.physical_breakpoints

    @property
    def domain_start(self) -> Array:
        return self.physical_breakpoints[0]

    @property
    def domain_end(self) -> Array:
        return self.physical_breakpoints[-1]

    @property
    def method(self) -> RodStrainBasisKind:
        return self.plan.basis_kind

    @property
    def coordinate_count(self) -> int:
        return self.plan.coordinate_count

    @property
    def basis_id(self) -> str:
        return self.prepared_id

    def evaluate(self, arc_lengths: ArrayLike, /) -> Array:
        """Evaluate canonical basis values at arbitrary physical arc lengths."""
        arc = jnp.asarray(arc_lengths, dtype=self.total_length.dtype)
        normalized = arc / self.total_length
        return _evaluate_normalized(self.plan, normalized)

    def basis_matrix(self, arc_lengths: ArrayLike, /) -> Array:
        """Alias naming the matrix-valued basis action used by reconstruction."""
        return self.evaluate(arc_lengths)

    def strain(self, coefficients: ArrayLike, arc_lengths: ArrayLike, /) -> Array:
        """Evaluate the physical material strain increment for coefficients."""
        values = jnp.asarray(coefficients)
        if values.shape != (self.plan.coordinate_count,):
            raise ValueError("coefficients do not match the prepared strain basis.")
        if values.dtype != self.total_length.dtype:
            raise TypeError("coefficients must retain the prepared strain basis dtype.")
        return ein.contract("...ik,k->...i", self.evaluate(arc_lengths), values)


def _evaluate_normalized(plan: RodStrainBasisPlan, coordinates: Array, /) -> Array:
    clipped = jnp.clip(coordinates, plan.breakpoints[0], plan.breakpoints[-1])
    interval_ids = jnp.clip(
        jnp.searchsorted(plan.breakpoints, clipped, side="right") - 1,
        0,
        plan.breakpoints.shape[0] - 2,
    )
    lower = plan.breakpoints[interval_ids]
    upper = plan.breakpoints[interval_ids + 1]
    local = (clipped - lower) / (upper - lower)
    powers = local[..., None] ** jnp.arange(plan.polynomial_degree + 1, dtype=local.dtype)
    selected = plan.polynomial_coefficients[interval_ids]
    unscaled = ein.contract("...ikp,...p->...ik", selected, powers)
    return unscaled * plan.component_scales[..., None]


def _physical_worksets(
    plan: RodStrainBasisPlan,
    rod: PreparedRod,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(rod.stretch_shear_measures)
    total_length = float(np.sum(lengths))
    nodes = np.concatenate((np.zeros((1,), dtype=lengths.dtype), np.cumsum(lengths)))
    stretch = nodes[:-1] + 0.5 * lengths
    bend = nodes[1:-1]
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(
        plan.quadrature_order
    )
    normalized_breakpoints = np.asarray(plan.breakpoints)
    quadrature_nodes: list[np.ndarray] = []
    quadrature_weights: list[np.ndarray] = []
    for lower, upper in zip(
        normalized_breakpoints[:-1], normalized_breakpoints[1:], strict=True
    ):
        midpoint = 0.5 * (lower + upper) * total_length
        half_width = 0.5 * (upper - lower) * total_length
        quadrature_nodes.append(
            midpoint + half_width * reference_nodes.astype(lengths.dtype, copy=False)
        )
        quadrature_weights.append(
            half_width * reference_weights.astype(lengths.dtype, copy=False)
        )
    return (
        stretch,
        bend,
        np.concatenate(quadrature_nodes),
        np.concatenate(quadrature_weights),
    )


def _interval_ids(physical_breakpoints: Array, sites: Array, /) -> Array:
    return jnp.clip(
        jnp.searchsorted(physical_breakpoints, sites, side="right") - 1,
        0,
        physical_breakpoints.shape[0] - 2,
    ).astype(jnp.int32)


def prepare_rod_strain_basis(
    plan: RodStrainBasisPlan,
    rod: PreparedRod,
    /,
) -> PreparedRodStrainBasis:
    """Prepare native strain and fixed quadrature worksets for one rod."""
    if not isinstance(plan, RodStrainBasisPlan):
        raise TypeError("plan must be a RodStrainBasisPlan.")
    if not isinstance(rod, PreparedRod):
        raise TypeError("rod must be a PreparedRod.")
    if plan.dimension != rod.plan.dimension:
        raise ValueError(
            "Rod strain basis dimension is incompatible with the prepared rod."
        )
    plan_dtype = np.dtype(plan.polynomial_coefficients.dtype)
    rod_dtype = np.dtype(rod.plan.rest_positions.dtype)
    if plan_dtype != rod_dtype:
        raise TypeError(
            "Rod strain basis dtype must match the prepared rod dtype; implicit precision loss is forbidden."
        )
    if rod.plan.inextensible:
        raise ValueError(
            "Inextensible rods are unsupported because native projection would "
            "leave the declared reduced strain manifold."
        )
    if plan.dimension == 2 and np.any(
        np.asarray(plan.polynomial_coefficients)[:, (2, 3, 4), :, :] != 0.0
    ):
        raise ValueError("Planar strain bases may populate only nu_x, nu_y, and kappa_z.")

    stretch_host, bend_host, quadrature_host, quadrature_weights_host = (
        _physical_worksets(plan, rod)
    )
    total_length_host = np.asarray(
        np.sum(np.asarray(rod.stretch_shear_measures)), dtype=rod_dtype
    )
    total_length = jnp.asarray(total_length_host)
    physical_breakpoints = plan.breakpoints * total_length
    stretch_arc_lengths = jnp.asarray(stretch_host, dtype=rod_dtype)
    bend_arc_lengths = jnp.asarray(bend_host, dtype=rod_dtype)
    quadrature_arc_lengths = jnp.asarray(quadrature_host, dtype=rod_dtype)
    quadrature_weights = jnp.asarray(quadrature_weights_host, dtype=rod_dtype)
    stretch_full = _evaluate_normalized(plan, stretch_arc_lengths / total_length)
    bend_full = _evaluate_normalized(plan, bend_arc_lengths / total_length)
    quadrature_basis = _evaluate_normalized(plan, quadrature_arc_lengths / total_length)
    if plan.dimension == 2:
        stretch_shear_basis = stretch_full[:, :2, :]
        bend_twist_basis = bend_full[:, 5:6, :]
    else:
        stretch_shear_basis = stretch_full[:, :3, :]
        bend_twist_basis = bend_full[:, 3:, :]

    coordinate_count = plan.coordinate_count
    weighted_blocks = [
        np.asarray(stretch_shear_basis)
        * np.sqrt(np.asarray(rod.stretch_shear_measures))[:, None, None],
        np.asarray(bend_twist_basis)
        * np.sqrt(np.asarray(rod.bend_twist_measures))[:, None, None],
    ]
    weighted = np.concatenate(
        tuple(block.reshape((-1, coordinate_count)) for block in weighted_blocks),
        axis=0,
    )
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = plan.rank_tolerance * largest
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    condition_number = np.inf if smallest <= 0.0 else largest / smallest
    finite = bool(np.all(np.isfinite(singular_values)) and np.isfinite(condition_number))
    full_column_rank = numerical_rank == coordinate_count
    condition_valid = condition_number <= plan.maximum_condition_number
    dtype_retained = plan_dtype == np.dtype(stretch_shear_basis.dtype) == rod_dtype
    evidence = RodStrainBasisEvidence(
        jnp.asarray(singular_values, dtype=rod_dtype),
        jnp.asarray(numerical_rank, dtype=jnp.int32),
        jnp.asarray(threshold, dtype=rod_dtype),
        jnp.asarray(condition_number, dtype=rod_dtype),
        jnp.asarray(full_column_rank),
        jnp.asarray(condition_valid),
        jnp.asarray(dtype_retained),
        jnp.asarray(finite),
        jnp.asarray(finite and full_column_rank and condition_valid and dtype_retained),
        int(weighted.shape[0]),
        coordinate_count,
        plan_dtype.str,
        np.dtype(stretch_shear_basis.dtype).str,
    )
    if not finite or not full_column_rank:
        raise ValueError(
            "The weighted native rod strain basis must have full column rank in the prepared dtype."
        )
    if not condition_valid:
        raise ValueError(
            "The weighted native rod strain basis condition number exceeds maximum_condition_number."
        )

    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-native-discrete-rod-strain-basis",
            "plan": plan.plan_id,
            "rod": rod.prepared_id,
            "worksets": array_tree_fingerprint(
                {
                    "physical_breakpoints": np.asarray(physical_breakpoints),
                    "stretch_arc_lengths": stretch_host,
                    "bend_arc_lengths": bend_host,
                    "quadrature_arc_lengths": quadrature_host,
                    "quadrature_weights": quadrature_weights_host,
                    "stretch_shear_basis": np.asarray(stretch_shear_basis),
                    "bend_twist_basis": np.asarray(bend_twist_basis),
                    "quadrature_basis": np.asarray(quadrature_basis),
                }
            ),
            "weighted_rank": numerical_rank,
            "condition_number": condition_number,
            "dtype": rod_dtype.str,
        }
    )
    return PreparedRodStrainBasis(
        plan,
        rod,
        physical_breakpoints,
        stretch_arc_lengths,
        bend_arc_lengths,
        _interval_ids(physical_breakpoints, stretch_arc_lengths),
        _interval_ids(physical_breakpoints, bend_arc_lengths),
        quadrature_arc_lengths,
        quadrature_weights,
        _interval_ids(physical_breakpoints, quadrature_arc_lengths),
        stretch_shear_basis,
        bend_twist_basis,
        quadrature_basis,
        evidence,
        total_length,
        prepared_id,
    )


def piecewise_constant_rod_strain_basis(
    breakpoints: ArrayLike, **kwargs
) -> RodStrainBasisPlan:
    """Functional constructor for :meth:`RodStrainBasisPlan.piecewise_constant`."""
    return RodStrainBasisPlan.piecewise_constant(breakpoints, **kwargs)


def shifted_legendre_rod_strain_basis(
    degree: int | Sequence[int], **kwargs
) -> RodStrainBasisPlan:
    """Functional constructor for :meth:`RodStrainBasisPlan.shifted_legendre`."""
    return RodStrainBasisPlan.shifted_legendre(degree, **kwargs)


def explicit_rod_strain_basis(
    breakpoints: ArrayLike,
    polynomial_coefficients: ArrayLike,
    **kwargs,
) -> RodStrainBasisPlan:
    """Functional constructor for :meth:`RodStrainBasisPlan.explicit`."""
    return RodStrainBasisPlan.explicit(breakpoints, polynomial_coefficients, **kwargs)


__all__ = [
    "PreparedRodStrainBasis",
    "RodStrainBasisEvidence",
    "RodStrainBasisKind",
    "RodStrainBasisPlan",
    "explicit_rod_strain_basis",
    "piecewise_constant_rod_strain_basis",
    "prepare_rod_strain_basis",
    "shifted_legendre_rod_strain_basis",
]
