#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, Key
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....linalg import DenseLinearOperator, OperatorProperties
from ....linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ....stochastic import (
    gaussian_field_diagnostics,
    GaussianCoefficientRealization,
    GaussianFieldDiagnostics,
    RandomFieldModel,
    RandomFieldSample,
    SpatialBasisSynthesis,
    StaticGaussianRandomField,
)
from ....uq import DenseCovariance
from .._quantities import CardiovascularQuantitySpec


def _identifier(value: str, name: str, /) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


@dataclass(frozen=True, slots=True)
class CanonicalCoordinateAxis:
    """One dimension of a mesh-independent canonical cardiac coordinate frame."""

    name: str
    lower: float
    upper: float
    unit: str = "1"
    periodic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "coordinate axis name"))
        object.__setattr__(self, "unit", _identifier(self.unit, "coordinate axis unit"))
        lower = float(self.lower)
        upper = float(self.upper)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("Canonical coordinate bounds must be finite and ordered.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def period(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class CanonicalCardiacCoordinates:
    """Canonical coordinates and physical quadrature on one fixed topology."""

    values: Array
    quadrature_weights: Array
    axes: tuple[CanonicalCoordinateAxis, ...]
    topology_id: str
    coordinate_id: str = field(init=False)

    def __post_init__(self) -> None:
        points = jnp.asarray(self.values, dtype=float)
        weights = jnp.asarray(self.quadrature_weights, dtype=float).reshape((-1,))
        axes = tuple(self.axes)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("Canonical coordinates must have shape (point, coordinate).")
        if len(axes) != int(points.shape[1]) or any(
            not isinstance(axis, CanonicalCoordinateAxis) for axis in axes
        ):
            raise TypeError("axes must define every canonical coordinate dimension.")
        if len({axis.name for axis in axes}) != len(axes):
            raise ValueError("Canonical coordinate axis names must be unique.")
        if weights.shape != (int(points.shape[0]),):
            raise ValueError("Quadrature weights must contain one value per point.")
        if (
            bool(jnp.any(~jnp.isfinite(points)))
            or bool(jnp.any(~jnp.isfinite(weights)))
            or bool(jnp.any(weights <= 0.0))
        ):
            raise ValueError(
                "Canonical coordinates and quadrature must be finite and positive."
            )
        for position, axis in enumerate(axes):
            coordinate = points[:, position]
            tolerance = (
                32.0
                * jnp.finfo(points.dtype).eps
                * max(1.0, abs(axis.lower), abs(axis.upper))
            )
            if bool(jnp.any(coordinate < axis.lower - tolerance)) or bool(
                jnp.any(coordinate > axis.upper + tolerance)
            ):
                raise ValueError(
                    f"Coordinate {axis.name!r} lies outside its canonical bounds."
                )
        topology = _identifier(self.topology_id, "topology_id")
        coordinate_id = canonical_fingerprint(
            {
                "kind": "canonical-cardiac-coordinate-system",
                "topology": topology,
                "axes": [
                    {
                        "name": axis.name,
                        "lower": axis.lower,
                        "upper": axis.upper,
                        "unit": axis.unit,
                        "periodic": axis.periodic,
                    }
                    for axis in axes
                ],
                "values": array_tree_fingerprint(points),
                "quadrature": array_tree_fingerprint(weights),
            }
        )
        object.__setattr__(self, "values", points)
        object.__setattr__(self, "quadrature_weights", weights)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "coordinate_id", coordinate_id)

    @property
    def point_count(self) -> int:
        return int(self.values.shape[0])


class CardiacFieldTransform(Protocol):
    """Shape-preserving physical transform for one latent Gaussian field."""

    @property
    def transform_id(self) -> str: ...

    def __call__(self, values: Array, /) -> Array: ...


@dataclass(frozen=True, slots=True)
class IdentityFieldTransform:
    transform_id: str = field(default="identity", init=False)

    def __call__(self, values: Array, /) -> Array:
        return values


@dataclass(frozen=True, slots=True)
class PositiveExponentialFieldTransform:
    """Map a latent Gaussian field to a strictly positive physical field."""

    floor: float = 0.0
    transform_id: str = field(init=False)

    def __post_init__(self) -> None:
        floor = float(self.floor)
        if not isfinite(floor) or floor < 0.0:
            raise ValueError("Positive-field floors must be finite and nonnegative.")
        object.__setattr__(self, "floor", floor)
        object.__setattr__(
            self,
            "transform_id",
            canonical_fingerprint(
                {"kind": "cardiac-positive-exponential-transform", "floor": floor}
            ),
        )

    def __call__(self, values: Array, /) -> Array:
        return self.floor + jnp.exp(values)


@dataclass(frozen=True, slots=True)
class BoundedLogisticFieldTransform:
    """Map a latent Gaussian field into explicit finite physical bounds."""

    lower: float
    upper: float
    transform_id: str = field(init=False)

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError("Bounded-field limits must be finite and ordered.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self,
            "transform_id",
            canonical_fingerprint(
                {"kind": "cardiac-bounded-logistic-transform", "bounds": [lower, upper]}
            ),
        )

    def __call__(self, values: Array, /) -> Array:
        return self.lower + (self.upper - self.lower) * jax_sigmoid(values)


def jax_sigmoid(value: Array, /) -> Array:
    """Stable local sigmoid without introducing a second transform framework."""

    exponential = jnp.exp(-jnp.abs(value))
    return jnp.where(
        value >= 0.0,
        1.0 / (1.0 + exponential),
        exponential / (1.0 + exponential),
    )


@dataclass(frozen=True, slots=True)
class CardiacRandomFieldRecipe:
    """Matérn-3/2 KL recipe expressed only in canonical cardiac coordinates."""

    field_name: str
    quantity: CardiovascularQuantitySpec
    latent_mean: float
    latent_standard_deviation: float
    correlation_lengths: tuple[float, ...]
    rank: int
    transform: CardiacFieldTransform = field(default_factory=IdentityFieldTransform)
    recipe_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _identifier(self.field_name, "field_name")
        if not isinstance(self.quantity, CardiovascularQuantitySpec):
            raise TypeError("quantity must be a CardiovascularQuantitySpec.")
        mean = float(self.latent_mean)
        standard_deviation = float(self.latent_standard_deviation)
        lengths = tuple(float(value) for value in self.correlation_lengths)
        rank = int(self.rank)
        if (
            not isfinite(mean)
            or not isfinite(standard_deviation)
            or standard_deviation <= 0.0
        ):
            raise ValueError(
                "Latent mean and standard deviation must be finite and positive."
            )
        if not lengths or any(not isfinite(value) or value <= 0.0 for value in lengths):
            raise ValueError("Correlation lengths must be finite and positive.")
        if rank <= 0:
            raise ValueError("Random-field rank must be positive.")
        if not isinstance(
            self.transform,
            (
                IdentityFieldTransform,
                PositiveExponentialFieldTransform,
                BoundedLogisticFieldTransform,
            ),
        ):
            raise TypeError("transform must be a supported cardiac field transform.")
        recipe_id = canonical_fingerprint(
            {
                "kind": "cardiac-canonical-matern32-random-field-recipe",
                "field_name": name,
                "quantity": self.quantity.quantity_id,
                "latent_mean": mean,
                "latent_standard_deviation": standard_deviation,
                "correlation_lengths": lengths,
                "rank": rank,
                "transform": self.transform.transform_id,
            }
        )
        object.__setattr__(self, "field_name", name)
        object.__setattr__(self, "latent_mean", mean)
        object.__setattr__(self, "latent_standard_deviation", standard_deviation)
        object.__setattr__(self, "correlation_lengths", lengths)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "recipe_id", recipe_id)

    def instantiate(
        self, coordinates: CanonicalCardiacCoordinates, /
    ) -> CanonicalRandomField:
        """Prepare a native stochastic random field through a dense native eigensolve."""

        if not isinstance(coordinates, CanonicalCardiacCoordinates):
            raise TypeError("coordinates must be CanonicalCardiacCoordinates.")
        if len(self.correlation_lengths) != len(coordinates.axes):
            raise ValueError("A correlation length is required for every canonical axis.")
        if self.rank > coordinates.point_count:
            raise ValueError("Random-field rank cannot exceed the canonical point count.")
        covariance_matrix = _matern32_covariance(
            coordinates,
            self.latent_standard_deviation,
            self.correlation_lengths,
        )
        weights = coordinates.quadrature_weights
        root_weights = jnp.sqrt(weights)
        symmetric = root_weights[:, None] * covariance_matrix * root_weights[None, :]
        properties = OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        )
        result = eigensolve(
            Eigenproblem(
                DenseLinearOperator(
                    symmetric,
                    properties=properties,
                    operator_id=f"cardiac-random-field:{self.recipe_id}",
                ),
                problem_id=f"cardiac-random-field-kl:{self.recipe_id}",
            ),
            policy=EigenSolvePolicy(
                DenseEigh(),
                count=coordinates.point_count,
                which="smallest-algebraic",
            ),
        )
        if not bool(result.successful):
            raise RuntimeError("Native KL covariance eigensolve did not converge.")
        eigenvalues = jnp.maximum(jnp.real(result.eigenvalues[::-1][: self.rank]), 0.0)
        vectors = jnp.real(jnp.asarray(result.eigenvectors)[:, ::-1][:, : self.rank])
        modes = vectors / root_weights[:, None]
        basis_id = canonical_fingerprint(
            {
                "kind": "cardiac-canonical-kl-basis",
                "recipe": self.recipe_id,
                "coordinates": coordinates.coordinate_id,
                "eigenvalues": array_tree_fingerprint(eigenvalues),
            }
        )
        synthesis = SpatialBasisSynthesis(
            modes,
            eigenvalues,
            weights,
            mode_ids=tuple(f"{basis_id}:{index}" for index in range(self.rank)),
            basis_id=basis_id,
            discretization_id=coordinates.topology_id,
            mean=jnp.full((coordinates.point_count,), self.latent_mean),
        )
        gaussian = StaticGaussianRandomField(
            synthesis,
            role="coefficient",
            source=f"canonical-coordinate:{coordinates.coordinate_id}",
        )
        model: RandomFieldModel = (
            gaussian
            if isinstance(self.transform, IdentityFieldTransform)
            else gaussian.transform(
                self.transform, transform_id=self.transform.transform_id
            )
        )
        return CanonicalRandomField(
            self,
            coordinates,
            DenseCovariance(covariance_matrix),
            gaussian,
            model,
            canonical_fingerprint(
                {
                    "kind": "prepared-cardiac-canonical-random-field",
                    "recipe": self.recipe_id,
                    "coordinates": coordinates.coordinate_id,
                    "basis": basis_id,
                }
            ),
        )


def _matern32_covariance(
    coordinates: CanonicalCardiacCoordinates,
    standard_deviation: float,
    correlation_lengths: tuple[float, ...],
    /,
) -> Array:
    delta = jnp.abs(coordinates.values[:, None, :] - coordinates.values[None, :, :])
    periodic_components = []
    for position, axis in enumerate(coordinates.axes):
        component = delta[..., position]
        if axis.periodic:
            component = jnp.minimum(component, axis.period - component)
        periodic_components.append(component / correlation_lengths[position])
    scaled = jnp.stack(periodic_components, axis=-1)
    squared_distance = contract("...d,...d->...", scaled, scaled)
    distance = jnp.sqrt(jnp.maximum(squared_distance, 0.0))
    radial = jnp.sqrt(jnp.asarray(3.0, dtype=distance.dtype)) * distance
    return standard_deviation**2 * (1.0 + radial) * jnp.exp(-radial)


@dataclass(frozen=True, slots=True)
class CanonicalRandomField:
    """Prepared native model with its full nodal covariance and coordinate binding."""

    recipe: CardiacRandomFieldRecipe
    coordinates: CanonicalCardiacCoordinates
    covariance: DenseCovariance
    gaussian_model: StaticGaussianRandomField
    model: RandomFieldModel
    field_id: str

    def realize(
        self,
        key: Key[Array, ""],
        /,
        *,
        sample_count: int,
        coupling_id: str | None = None,
    ) -> GaussianCoefficientRealization:
        count = int(sample_count)
        if count <= 0:
            raise ValueError("sample_count must be positive.")
        return self.gaussian_model.realize(
            key,
            sample_shape=(count,),
            coupling_id=coupling_id,
            label=self.field_id,
        )

    def sample(self, realization: GaussianCoefficientRealization, /) -> RandomFieldSample:
        return self.model.sample(realization)

    def diagnostics(
        self, realization: GaussianCoefficientRealization, /
    ) -> GaussianFieldDiagnostics:
        return gaussian_field_diagnostics(self.gaussian_model, realization)


__all__ = [
    "BoundedLogisticFieldTransform",
    "CanonicalCardiacCoordinates",
    "CanonicalCoordinateAxis",
    "CanonicalRandomField",
    "CardiacFieldTransform",
    "CardiacRandomFieldRecipe",
    "IdentityFieldTransform",
    "PositiveExponentialFieldTransform",
]
