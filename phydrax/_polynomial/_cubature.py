#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from numbers import Integral
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import DTypeLike
from scipy.special import roots_jacobi

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse
from ._lebedev_cubature_data import LEBEDEV_RULES
from ._orthogonal import legendre_rule_data
from ._simplex_cubature_data import TETRAHEDRON_RULES, TRIANGLE_RULES


CubatureReference: TypeAlias = Literal[
    "triangle",
    "tetrahedron",
    "circle",
    "disk",
    "sphere",
    "ball",
    "standard-normal",
    "simplex",
    "tensor",
]
TensorProductRuleFamily: TypeAlias = Literal["gauss", "lobatto"]
CubatureFamily: TypeAlias = Literal[
    "xiao-gimbutas",
    "lebedev",
    "periodic-circle",
    "radial-product",
    "duffy",
    "stroud-secrest-3-1",
    "hadamard-3",
    "stroud-secrest-5-2",
    "stroud-secrest-5-3",
    "tensor-hermite",
    "tensor-product",
]

_REFERENCE_DIMENSION: dict[str, int | None] = {
    "triangle": 2,
    "tetrahedron": 3,
    "circle": 2,
    "disk": 2,
    "sphere": 3,
    "ball": 3,
    "standard-normal": None,
    "simplex": None,
    "tensor": None,
}
_REFERENCE_MASS: dict[str, float] = {
    "triangle": 0.5,
    "tetrahedron": 1.0 / 6.0,
    "circle": 2.0 * math.pi,
    "disk": math.pi,
    "sphere": 4.0 * math.pi,
    "ball": 4.0 * math.pi / 3.0,
    "standard-normal": 1.0,
}
_REFERENCE_MEASURE: dict[str, str] = {
    "triangle": "lebesgue",
    "tetrahedron": "lebesgue",
    "circle": "surface",
    "disk": "lebesgue",
    "sphere": "surface",
    "ball": "lebesgue",
    "standard-normal": "standard-normal",
}
_REFERENCE_MEASURE["simplex"] = "lebesgue"
_REFERENCE_MEASURE["tensor"] = "lebesgue"
_DEFAULT_RULE_BYTES = 64 * 1024**2


@final
class CubatureRuleData(StrictModule, NonTrainableState):
    """Prepared multidimensional cubature with explicit measure semantics."""

    points: Array
    weights: Array
    exact_degree: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
    reference_domain: str = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    integration_measure: str = eqx.field(static=True)
    measure_mass: float = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: npt.ArrayLike,
        weights: npt.ArrayLike,
        /,
        *,
        exact_degree: int,
        family: CubatureFamily,
        reference_domain: CubatureReference,
        backend: str,
        source_id: str,
        dtype: DTypeLike = jnp.float64,
        maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
        dimension: int | None = None,
    ) -> None:
        degree = _degree(exact_degree)
        reference_domain = parse(reference_domain, CubatureReference, "reference_domain")
        family = parse(family, CubatureFamily, "family")
        dtype_ = np.dtype(dtype)
        points_host = np.asarray(points, dtype=dtype_)
        weights_host = np.asarray(weights, dtype=dtype_).reshape((-1,))
        reference_dimension = _REFERENCE_DIMENSION[reference_domain]
        if dimension is not None:
            declared_dimension = _degree(dimension)
            if declared_dimension == 0:
                raise ValueError("Positive-dimensional cubature requires dimension >= 1.")
            if (
                reference_dimension is not None
                and declared_dimension != reference_dimension
            ):
                raise ValueError("Explicit dimension does not match cubature reference.")
            reference_dimension = declared_dimension
        elif reference_domain in ("simplex", "tensor"):
            raise ValueError("Generic reference cubature requires explicit dimension.")
        if (
            points_host.ndim != 2
            or points_host.shape[1] == 0
            or (
                reference_dimension is not None
                and points_host.shape[1] != reference_dimension
            )
            or points_host.shape[0] == 0
            or weights_host.shape != points_host.shape[:1]
        ):
            raise ValueError(
                "Cubature points must have shape (num_points, reference_dimension) with one aligned weight per point."
            )
        if (
            np.any(~np.isfinite(points_host))
            or np.any(~np.isfinite(weights_host))
            or np.any(weights_host <= 0.0)
        ):
            raise ValueError(
                "Cubature points must be finite and weights positive finite."
            )
        point_dimension = points_host.shape[1]
        order = np.lexsort(
            tuple(points_host[:, axis] for axis in range(point_dimension - 1, -1, -1))
        )
        points_host = np.asarray(points_host[order], dtype=dtype_)
        weights_host = np.asarray(weights_host[order], dtype=dtype_)
        if points_host.shape[0] > 1 and np.any(
            np.all(points_host[1:] == points_host[:-1], axis=1)
        ):
            raise ValueError("Cubature points must be unique.")
        tolerance = float(512.0 * np.finfo(dtype_).eps * max(1, points_host.shape[0]))
        _validate_reference_points(reference_domain, points_host, tolerance)
        mass = (
            math.exp(-math.lgamma(point_dimension + 1))
            if reference_domain == "simplex"
            else 1.0
            if reference_domain == "tensor"
            else _REFERENCE_MASS[reference_domain]
        )
        if not np.isclose(
            np.sum(weights_host),
            mass,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError("Cubature weights do not have their declared measure mass.")
        storage_bytes = points_host.nbytes + weights_host.nbytes
        if (
            isinstance(maximum_rule_bytes, bool)
            or not isinstance(maximum_rule_bytes, Integral)
            or int(maximum_rule_bytes) <= 0
        ):
            raise ValueError("maximum_rule_bytes must be a positive integer.")
        if storage_bytes > int(maximum_rule_bytes):
            raise ValueError("Cubature rule exceeds maximum_rule_bytes.")
        if not backend or not source_id:
            raise ValueError("Cubature backend and source_id must be nonempty.")
        self.points = jnp.asarray(points_host)
        self.weights = jnp.asarray(weights_host)
        self.exact_degree = degree
        self.family = family
        self.reference_domain = reference_domain
        self.reference_dimension = point_dimension
        self.integration_measure = _REFERENCE_MEASURE[reference_domain]
        self.measure_mass = mass
        self.backend = str(backend)
        self.source_id = str(source_id)
        self.storage_bytes = storage_bytes
        self.rule_id = canonical_fingerprint(
            {
                "kind": "cubature-rule",
                "family": family,
                "reference_domain": reference_domain,
                "reference_dimension": point_dimension,
                "integration_measure": self.integration_measure,
                "measure_mass": mass,
                "exact_degree": degree,
                "backend": self.backend,
                "source_id": self.source_id,
                "storage_bytes": storage_bytes,
                "data": array_tree_fingerprint((points_host, weights_host)),
            }
        )


def _degree(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Cubature degree must be an integer.")
    degree = int(value)
    if degree < 0:
        raise ValueError("Cubature degree must be nonnegative.")
    return degree


def _validate_reference_points(
    reference: CubatureReference,
    points: np.ndarray,
    tolerance: float,
    /,
) -> None:
    if reference == "standard-normal":
        return
    if reference in ("triangle", "tetrahedron", "simplex"):
        if np.any(points < -tolerance) or np.any(
            np.sum(points, axis=1) > 1.0 + tolerance
        ):
            raise ValueError("Simplex cubature points lie outside the unit simplex.")
        return
    if reference == "tensor":
        if np.any(points < -tolerance) or np.any(points > 1.0 + tolerance):
            raise ValueError("Tensor cubature points lie outside the unit cube.")
        return
    norms = np.linalg.norm(points, axis=1)
    if reference in ("circle", "sphere"):
        if np.any(np.abs(norms - 1.0) > tolerance):
            raise ValueError("Surface cubature points must lie on the unit sphere.")
        return
    if np.any(norms > 1.0 + tolerance):
        raise ValueError("Radial cubature points lie outside the unit ball.")


def _smallest_available(
    available: tuple[int, ...], requested: int, reference: str, /
) -> int:
    for degree in available:
        if degree >= requested:
            return degree
    raise ValueError(
        f"No positive {reference} cubature rule is available for degree {requested}; "
        f"the maximum certified degree is {available[-1]}."
    )


def xiao_gimbutas_rule_data(
    reference: Literal["triangle", "tetrahedron"],
    degree: int,
    /,
    *,
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    requested = _degree(degree)
    table = TRIANGLE_RULES if reference == "triangle" else TETRAHEDRON_RULES
    selected = _smallest_available(tuple(table), requested, reference)
    points, weights = table[selected]
    return CubatureRuleData(
        points,
        weights,
        exact_degree=selected,
        family="xiao-gimbutas",
        reference_domain=reference,
        backend="tabulated-modepy",
        source_id=f"modepy-2026.1:xiao-gimbutas:{reference}:{selected}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def _product_capacity(dimension: int, order: int, maximum_rule_bytes: int, /) -> int:
    if (
        isinstance(maximum_rule_bytes, bool)
        or not isinstance(maximum_rule_bytes, Integral)
        or maximum_rule_bytes <= 0
    ):
        raise ValueError("maximum_rule_bytes must be a positive integer.")
    # Points, tensor grids and transformed points coexist during preparation.
    capacity = maximum_rule_bytes // (8 * (3 * dimension + 1))
    if capacity < 1 or (order > 1 and dimension * math.log(order) > math.log(capacity)):
        raise ValueError("Cubature preparation exceeds maximum_rule_bytes.")
    count = order**dimension
    if count > capacity:
        raise ValueError("Cubature preparation exceeds maximum_rule_bytes.")
    return count


def _axis_product(
    dimension: int,
    order: int,
    family: TensorProductRuleFamily,
    maximum_rule_bytes: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    _product_capacity(dimension, order, maximum_rule_bytes)
    axis_rule = legendre_rule_data(order, family)
    axis = 0.5 * (np.asarray(axis_rule.nodes, dtype=np.float64) + 1.0)
    weights = 0.5 * np.asarray(axis_rule.weights, dtype=np.float64)
    axes = np.meshgrid(*(axis for _ in range(dimension)), indexing="ij")
    weight_axes = np.meshgrid(
        *(weights for _ in range(dimension)), indexing="ij", sparse=True
    )
    points = np.stack(axes, axis=-1).reshape((-1, dimension))
    combined = np.ones((order,) * dimension, dtype=np.float64)
    for values in weight_axes:
        combined *= values
    return points, combined.reshape((-1,))


def _duffy_product(
    dimension: int, order: int, maximum_rule_bytes: int, /
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = _axis_product(dimension, order, "gauss", maximum_rule_bytes)
    points = np.empty_like(nodes)
    remaining = np.ones((nodes.shape[0],), dtype=np.float64)
    for axis in range(dimension):
        points[:, axis] = remaining * nodes[:, axis]
        if axis < dimension - 1:
            weights *= (1.0 - nodes[:, axis]) ** (dimension - axis - 1)
        remaining *= 1.0 - nodes[:, axis]
    return points, weights


def simplex_rule_data(
    dimension: int, degree: int, /, *, maximum_rule_bytes: int = _DEFAULT_RULE_BYTES
) -> CubatureRuleData:
    """Dimension-explicit positive Duffy quadrature on the unit n-simplex."""
    dimension = _degree(dimension)
    if dimension == 0:
        raise ValueError("Simplex cubature requires dimension >= 1.")
    requested = _degree(degree)
    order = max(1, (requested + dimension + 1) // 2)
    points, weights = _duffy_product(dimension, order, maximum_rule_bytes)
    return CubatureRuleData(
        points,
        weights,
        exact_degree=2 * order - dimension,
        family="duffy",
        reference_domain="simplex",
        dimension=dimension,
        backend="analytic-product",
        source_id=f"duffy-gauss-legendre:simplex:{dimension}:{order}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def tensor_product_rule_data(
    dimension: int,
    points: int,
    /,
    *,
    family: TensorProductRuleFamily = "gauss",
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    """Positive tensor GL/GLL quadrature on the unit n-cube."""
    dimension, count = _degree(dimension), _degree(points)
    if dimension == 0 or count == 0:
        raise ValueError("Tensor cubature requires dimension and points >= 1.")
    family = parse(family, TensorProductRuleFamily, "family")
    match family:
        case "gauss":
            exact_degree = 2 * count - 1
        case "lobatto":
            if count < 2:
                raise ValueError("Lobatto tensor cubature requires points >= 2.")
            exact_degree = 2 * count - 3
        case _:
            assert_never(family)
    nodes, weights = _axis_product(dimension, count, family, maximum_rule_bytes)
    return CubatureRuleData(
        nodes,
        weights,
        exact_degree=exact_degree,
        family="tensor-product",
        reference_domain="tensor",
        dimension=dimension,
        backend="analytic-product",
        source_id=f"tensor-legendre:{family}:{dimension}:{count}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def duffy_simplex_rule_data(
    reference: Literal["triangle", "tetrahedron"],
    degree: int,
    /,
    *,
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    requested = _degree(degree)
    offset = 1 if reference == "triangle" else 2
    order = max(1, math.ceil((requested + offset + 1) / 2))
    dimension = 2 if reference == "triangle" else 3
    points, combined = _duffy_product(dimension, order, maximum_rule_bytes)
    exact_degree = 2 * order - dimension
    return CubatureRuleData(
        points.reshape((-1, points.shape[-1])),
        combined.reshape((-1,)),
        exact_degree=exact_degree,
        family="duffy",
        reference_domain=reference,
        backend="analytic-product",
        source_id=f"duffy-gauss-legendre:{reference}:{order}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def lebedev_rule_data(
    degree: int,
    /,
    *,
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    requested = _degree(degree)
    selected = _smallest_available(tuple(LEBEDEV_RULES), requested, "sphere")
    points, weights = LEBEDEV_RULES[selected]
    return CubatureRuleData(
        points,
        weights,
        exact_degree=selected,
        family="lebedev",
        reference_domain="sphere",
        backend="tabulated-scipy",
        source_id=f"scipy-1.18.0:lebedev:{selected}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def periodic_circle_rule_data(
    degree: int,
    /,
    *,
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    requested = _degree(degree)
    count = max(2, 2 * math.ceil((requested + 1) / 2))
    angles = 2.0 * math.pi * np.arange(count, dtype=np.float64) / float(count)
    points = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    weights = np.full((count,), 2.0 * math.pi / float(count))
    return CubatureRuleData(
        points,
        weights,
        exact_degree=count - 1,
        family="periodic-circle",
        reference_domain="circle",
        backend="analytic",
        source_id=f"periodic-circle:{count}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def radial_disk_rule_data(
    degree: int,
    /,
    *,
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    requested = _degree(degree)
    angular = periodic_circle_rule_data(requested)
    radial_order = max(1, math.ceil((requested // 2 + 1) / 2))
    radial = legendre_rule_data(radial_order, "gauss")
    squared_radius = 0.5 * (np.asarray(radial.nodes) + 1.0)
    radii = np.sqrt(squared_radius)
    radial_weights = 0.25 * np.asarray(radial.weights)
    directions = np.asarray(angular.points)
    points = (radii[:, None, None] * directions[None, :, :]).reshape((-1, 2))
    weights = (radial_weights[:, None] * np.asarray(angular.weights)[None, :]).reshape(
        (-1,)
    )
    return CubatureRuleData(
        points,
        weights,
        exact_degree=requested,
        family="radial-product",
        reference_domain="disk",
        backend="analytic-product",
        source_id=f"radial-disk:{requested}:{radial_order}:{directions.shape[0]}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def radial_ball_rule_data(
    degree: int,
    /,
    *,
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    requested = _degree(degree)
    angular = lebedev_rule_data(requested)
    radial_order = max(1, math.ceil((requested // 2 + 1) / 2))
    nodes, raw_weights = roots_jacobi(radial_order, 0.0, 0.5)
    squared_radius = 0.5 * (np.asarray(nodes) + 1.0)
    radii = np.sqrt(squared_radius)
    radial_weights = np.asarray(raw_weights) / (4.0 * math.sqrt(2.0))
    directions = np.asarray(angular.points)
    points = (radii[:, None, None] * directions[None, :, :]).reshape((-1, 3))
    weights = (radial_weights[:, None] * np.asarray(angular.weights)[None, :]).reshape(
        (-1,)
    )
    return CubatureRuleData(
        points,
        weights,
        exact_degree=requested,
        family="radial-product",
        reference_domain="ball",
        backend="scipy-product",
        source_id=(f"radial-ball:{requested}:{radial_order}:{angular.exact_degree}"),
        maximum_rule_bytes=maximum_rule_bytes,
    )


def cubature_rule_data(
    reference: CubatureReference,
    degree: int,
    /,
    *,
    allow_duffy_fallback: bool = True,
    maximum_rule_bytes: int = _DEFAULT_RULE_BYTES,
) -> CubatureRuleData:
    requested = _degree(degree)
    if reference == "triangle":
        if requested <= max(TRIANGLE_RULES):
            return xiao_gimbutas_rule_data(
                reference, requested, maximum_rule_bytes=maximum_rule_bytes
            )
        if allow_duffy_fallback:
            return duffy_simplex_rule_data(
                reference, requested, maximum_rule_bytes=maximum_rule_bytes
            )
        return xiao_gimbutas_rule_data(
            reference, requested, maximum_rule_bytes=maximum_rule_bytes
        )
    if reference == "tetrahedron":
        if requested <= max(TETRAHEDRON_RULES):
            return xiao_gimbutas_rule_data(
                reference, requested, maximum_rule_bytes=maximum_rule_bytes
            )
        if allow_duffy_fallback:
            return duffy_simplex_rule_data(
                reference, requested, maximum_rule_bytes=maximum_rule_bytes
            )
        return xiao_gimbutas_rule_data(
            reference, requested, maximum_rule_bytes=maximum_rule_bytes
        )
    if reference == "circle":
        return periodic_circle_rule_data(requested, maximum_rule_bytes=maximum_rule_bytes)
    if reference == "disk":
        return radial_disk_rule_data(requested, maximum_rule_bytes=maximum_rule_bytes)
    if reference == "sphere":
        return lebedev_rule_data(requested, maximum_rule_bytes=maximum_rule_bytes)
    if reference == "ball":
        return radial_ball_rule_data(requested, maximum_rule_bytes=maximum_rule_bytes)
    raise ValueError(f"Unsupported cubature reference: {reference!r}.")


__all__ = [
    "CubatureFamily",
    "CubatureReference",
    "CubatureRuleData",
    "cubature_rule_data",
    "duffy_simplex_rule_data",
    "lebedev_rule_data",
    "periodic_circle_rule_data",
    "radial_ball_rule_data",
    "radial_disk_rule_data",
    "xiao_gimbutas_rule_data",
    "simplex_rule_data",
    "tensor_product_rule_data",
    "TensorProductRuleFamily",
]
