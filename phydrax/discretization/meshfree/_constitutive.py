# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Frame-covariant edge features and construction-certified scalar flux laws."""

from __future__ import annotations

from abc import abstractmethod
from math import isfinite, prod
from typing import Any, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._differentiation import DerivativeRegularity, GradientLevel
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import AbstractArrayModel, ModelExecutionContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...nn._contracts import network_execution_contract
from ...nn._utils import _identity
from ...nn.layers import Linear
from ...nn.models import (
    InputConvexCertificate,
    InputConvexNetwork,
    PartiallyInputConvexNetwork,
)
from ...nn.parameters import LowRankUpdate
from ...typing import Dim, Float, Float64, Int32, parse, Scalar, Size
from ...units import DIMENSIONLESS, DimensionSignature


EdgeFeatureKind: TypeAlias = Literal["scalar", "vector", "symmetric-tensor"]


class _FeatureEdgeDim(Dim):
    """Edges in one immutable feature geometry."""


class _FeatureNodeDim(Dim):
    """Nodes of one immutable feature geometry."""


class _FeatureSpatialDim(Dim):
    """Cartesian tensor components of one feature frame."""


@final
class EdgeFeatureField(StrictModule, NonTrainableState):
    """One named nodal field, with a physical dimension and tensor character."""

    name: str = eqx.field(static=True)
    kind: EdgeFeatureKind = eqx.field(static=True)
    dimension: DimensionSignature = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: EdgeFeatureKind,
        /,
        *,
        dimension: DimensionSignature = DIMENSIONLESS,
    ) -> None:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("Feature names must be nonempty stripped strings.")
        if not isinstance(dimension, DimensionSignature):
            raise TypeError("Feature dimensions must be DimensionSignature values.")
        kind_ = parse(kind, EdgeFeatureKind, "kind")
        self.name = name
        self.kind = kind_
        self.dimension = dimension


@final
class EdgeFrameFeatures(StrictModule, NonTrainableState):
    """Endpoint averages and oriented differences in an explicit Cartesian frame.

    Raw averages are even and raw differences are odd under endpoint reversal.
    The tangent is also odd. Consequently ``t dot (v_j-v_i)`` is *even*, whereas
    ``t dot (v_i+v_j)/2`` is odd. ``even``/``odd`` pack rotation-invariant scalar
    contractions with these actual parities; they never relabel a vector's
    longitudinal projection as an even average. Tensor contractions use ``t T t``.
    Differences are endpoint differences, not derivatives divided by edge length.
    """

    __strict_contract__ = True
    schema: tuple[EdgeFeatureField, ...]
    points: Float64[_FeatureNodeDim, _FeatureSpatialDim]
    pairs: Int32[_FeatureEdgeDim, Literal[2]]
    averages: tuple[Array, ...]
    differences: tuple[Array, ...]
    tangents: Float64[_FeatureEdgeDim, _FeatureSpatialDim]
    lengths: Float64[_FeatureEdgeDim]
    spatial_dimension: Size[_FeatureSpatialDim] = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    even_names: tuple[str, ...] = eqx.field(static=True)
    odd_names: tuple[str, ...] = eqx.field(static=True)
    even_dimensions: tuple[DimensionSignature, ...] = eqx.field(static=True)
    odd_dimensions: tuple[DimensionSignature, ...] = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        pairs: ArrayLike,
        schema: tuple[EdgeFeatureField, ...] = (),
        values: tuple[ArrayLike, ...] = (),
        *,
        source_id: str | None = None,
    ) -> None:
        points_ = np.asarray(points)
        pairs_ = np.asarray(pairs)
        if (
            points_.ndim != 2
            or points_.shape[1] not in (1, 2, 3)
            or np.iscomplexobj(points_)
            or not np.all(np.isfinite(points_))
        ):
            raise ValueError(
                "Points must be finite Cartesian coordinates in dimension 1, 2, or 3."
            )
        if (
            pairs_.ndim != 2
            or pairs_.shape[1] != 2
            or not np.issubdtype(pairs_.dtype, np.integer)
        ):
            raise ValueError("Pairs must be an integer (edge, 2) array.")
        if np.any(pairs_ < 0) or np.any(pairs_ >= points_.shape[0]):
            raise ValueError("Edge endpoints lie outside the point cloud.")
        if (
            not isinstance(schema, tuple)
            or not isinstance(values, tuple)
            or len(schema) != len(values)
            or any(not isinstance(field, EdgeFeatureField) for field in schema)
        ):
            raise TypeError("Each typed feature field requires one nodal array.")
        if len({field.name for field in schema}) != len(schema):
            raise ValueError("Feature names must be unique.")
        if source_id is not None and (
            not isinstance(source_id, str)
            or not source_id
            or source_id.strip() != source_id
        ):
            raise ValueError(
                "A declared feature source_id must be a nonempty stripped scientific identity."
            )
        displacement = points_[pairs_[:, 1]] - points_[pairs_[:, 0]]
        lengths = np.sqrt(np.sum(displacement * displacement, axis=1))
        if np.any(lengths <= 0):
            raise ValueError("Edge features require distinct endpoints.")
        dimension = points_.shape[1]
        averages: list[Array] = []
        differences: list[Array] = []
        even_names: list[str] = []
        odd_names: list[str] = []
        even_dimensions: list[DimensionSignature] = []
        odd_dimensions: list[DimensionSignature] = []
        for field, value in zip(schema, values, strict=True):
            array = np.asarray(value)
            match field.kind:
                case "scalar":
                    shape = (points_.shape[0],)
                    even = ("average",)
                    odd = ("difference",)
                    even_dims = (field.dimension,)
                case "vector":
                    shape = (points_.shape[0], dimension)
                    even = ("average-norm-squared", "difference-longitudinal")
                    odd = ("average-longitudinal",)
                    even_dims = (field.dimension.power(2), field.dimension)
                case "symmetric-tensor":
                    shape = (points_.shape[0], dimension, dimension)
                    even = ("average-trace", "average-longitudinal")
                    odd = ("difference-trace", "difference-longitudinal")
                    even_dims = (field.dimension, field.dimension)
            if (
                array.shape != shape
                or not np.all(np.isfinite(array))
                or np.iscomplexobj(array)
            ):
                raise ValueError(
                    f"Feature {field.name!r} must be finite real data of shape {shape}."
                )
            if field.kind == "symmetric-tensor" and not np.allclose(
                array, array.swapaxes(-1, -2), rtol=0, atol=0
            ):
                raise ValueError(
                    "Symmetric tensor features must be symmetric; no repair is applied."
                )
            averages.append(
                jnp.asarray(
                    0.5 * (array[pairs_[:, 0]] + array[pairs_[:, 1]]), dtype=jnp.float64
                )
            )
            differences.append(
                jnp.asarray(array[pairs_[:, 1]] - array[pairs_[:, 0]], dtype=jnp.float64)
            )
            even_names.extend(f"{field.name}:{name}" for name in even)
            odd_names.extend(f"{field.name}:{name}" for name in odd)
            even_dimensions.extend(even_dims)
            odd_dimensions.extend(field.dimension for _ in odd)
        self.schema = schema
        self.points = jnp.asarray(points, dtype=jnp.float64)
        self.pairs = jnp.asarray(pairs, dtype=jnp.int32)
        self.averages = tuple(averages)
        self.differences = tuple(differences)
        self.tangents = jnp.asarray(displacement / lengths[:, None], dtype=jnp.float64)
        self.lengths = jnp.asarray(lengths, dtype=jnp.float64)
        self.spatial_dimension = dimension
        self.source_id = source_id
        self.even_names = tuple(even_names)
        self.odd_names = tuple(odd_names)
        self.even_dimensions = tuple(even_dimensions)
        self.odd_dimensions = tuple(odd_dimensions)

    @property
    def geometry_id(self) -> str:
        """Nominal source identity, used only at host geometry admission."""
        return canonical_fingerprint(array_tree_fingerprint((self.points, self.pairs)))

    @property
    def even(self) -> Array:
        values: list[Array] = []
        for field, average, difference in zip(
            self.schema, self.averages, self.differences, strict=True
        ):
            match field.kind:
                case "scalar":
                    values.append(average)
                case "vector":
                    values.extend(
                        (
                            jnp.sum(average * average, axis=-1),
                            jnp.sum(self.tangents * difference, axis=-1),
                        )
                    )
                case "symmetric-tensor":
                    values.extend(
                        (
                            jnp.trace(average, axis1=-2, axis2=-1),
                            contract(
                                "ei,eij,ej->e", self.tangents, average, self.tangents
                            ),
                        )
                    )
        return (
            jnp.stack(values, axis=-1)
            if values
            else jnp.zeros((self.lengths.size, 0), dtype=self.lengths.dtype)
        )

    @property
    def odd(self) -> Array:
        values: list[Array] = []
        for field, average, difference in zip(
            self.schema, self.averages, self.differences, strict=True
        ):
            match field.kind:
                case "scalar":
                    values.append(difference)
                case "vector":
                    values.append(jnp.sum(self.tangents * average, axis=-1))
                case "symmetric-tensor":
                    values.extend(
                        (
                            jnp.trace(difference, axis1=-2, axis2=-1),
                            contract(
                                "ei,eij,ej->e", self.tangents, difference, self.tangents
                            ),
                        )
                    )
        return (
            jnp.stack(values, axis=-1)
            if values
            else jnp.zeros((self.lengths.size, 0), dtype=self.lengths.dtype)
        )

    def reoriented(self) -> EdgeFrameFeatures:
        """Reverse every edge without changing any physical nodal feature."""
        return eqx.tree_at(
            lambda feature: (feature.differences, feature.tangents, feature.pairs),
            self,
            (
                tuple(-value for value in self.differences),
                -self.tangents,
                self.pairs[:, ::-1],
            ),
        )

    def transformed_frame(self, orthogonal: ArrayLike, /) -> EdgeFrameFeatures:
        """Apply a proper or improper orthogonal Cartesian change of coordinates."""
        rotation = np.asarray(orthogonal)
        dimension = self.spatial_dimension
        if (
            rotation.shape != (dimension, dimension)
            or np.iscomplexobj(rotation)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(
                rotation.T @ rotation, np.eye(dimension), rtol=1e-12, atol=1e-12
            )
        ):
            raise ValueError("Frame changes require a finite orthogonal matrix.")
        q = jnp.asarray(rotation, dtype=self.tangents.dtype)

        def transform(field: EdgeFeatureField, value: Array) -> Array:
            match field.kind:
                case "scalar":
                    return value
                case "vector":
                    return value @ q.T
                case "symmetric-tensor":
                    return q @ value @ q.T

        return eqx.tree_at(
            lambda feature: (
                feature.averages,
                feature.differences,
                feature.tangents,
                feature.points,
            ),
            self,
            (
                tuple(
                    transform(field, value)
                    for field, value in zip(self.schema, self.averages, strict=True)
                ),
                tuple(
                    transform(field, value)
                    for field, value in zip(self.schema, self.differences, strict=True)
                ),
                self.tangents @ q.T,
                self.points @ q.T,
            ),
        )


class AbstractEdgeConstitutiveLaw(AbstractArrayModel):
    """One scalar edge law; differences and odd context reverse together."""

    even_size: eqx.AbstractVar[int]
    odd_size: eqx.AbstractVar[int]
    background_conductance: eqx.AbstractVar[float]

    @abstractmethod
    def edge_flux(self, difference: Array, even: Array, odd: Array, /) -> Array:
        raise NotImplementedError

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = jnp.asarray(x)
        if values.shape != (self.in_size,):
            raise ValueError("A constitutive model call requires one packed edge.")
        return self.edge_flux(
            values[0], values[1 : 1 + self.even_size], values[1 + self.even_size :]
        )

    def flux(self, differences: ArrayLike, features: EdgeFrameFeatures, /) -> Array:
        values = jnp.asarray(differences)
        even, odd = features.even, features.odd
        if (
            values.shape != features.lengths.shape
            or even.shape[1] != self.even_size
            or odd.shape[1] != self.odd_size
        ):
            raise ValueError(
                "Edge state and typed feature widths must match the constitutive law."
            )
        return jax.vmap(self.edge_flux)(values, even, odd)

    def derivative(self, differences: ArrayLike, features: EdgeFrameFeatures, /) -> Array:
        values = jnp.asarray(differences)
        even, odd = features.even, features.odd
        if (
            values.shape != features.lengths.shape
            or even.shape[1] != self.even_size
            or odd.shape[1] != self.odd_size
        ):
            raise ValueError(
                "Edge state and typed feature widths must match the constitutive law."
            )
        return jax.vmap(jax.grad(self.edge_flux, argnums=0))(values, even, odd)


def _width(size: int | tuple[int, ...] | Literal["scalar"]) -> int:
    return 1 if size == "scalar" else prod(size) if isinstance(size, tuple) else size


def _positive_background(value: float) -> float:
    if not isfinite(value) or value <= 0:
        raise ValueError("Background conductance must be finite and strictly positive.")
    return float(value)


@final
class MonotoneEdgeConductance(AbstractEdgeConstitutiveLaw):
    r"""Odd strongly monotone slope of a conditioned certified convex potential.

    ``b D + (phi'(D;c)-phi'(-D;c))/2`` is odd and has derivative at least ``b``.
    Subtracting ``phi'(0;c)`` alone is not an odd law. Only *even* context enters
    this potential. Dissipativity additionally requires a positive exterior metric.
    """

    potential: InputConvexNetwork | PartiallyInputConvexNetwork
    certificate: InputConvexCertificate
    even_size: int = eqx.field(static=True)
    odd_size: int = eqx.field(static=True)
    background_conductance: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        potential: InputConvexNetwork | PartiallyInputConvexNetwork,
        /,
        *,
        background_conductance: float = 1.0,
        odd_size: int = 0,
    ) -> None:
        if not isinstance(potential, (InputConvexNetwork, PartiallyInputConvexNetwork)):
            raise TypeError(
                "Monotone laws require a canonical input-convex model, not a user assertion."
            )
        certificate = potential.input_convex_certificate()
        if _width(certificate.convex_input_size) != 1:
            raise ValueError("The convex input of an edge potential must be scalar.")
        if certificate.activation != "softplus":
            raise ValueError(
                "Implicit constitutive roots require a smooth potential; use softplus."
            )
        if isinstance(odd_size, bool) or not isinstance(odd_size, int) or odd_size < 0:
            raise ValueError("odd_size must be a nonnegative integer.")
        background = _positive_background(background_conductance)
        even_size = (
            0 if certificate.context_size is None else _width(certificate.context_size)
        )
        self.potential = potential
        self.certificate = certificate
        self.even_size = even_size
        self.odd_size = odd_size
        self.background_conductance = background
        self.in_size = 1 + even_size + odd_size
        self.out_size = "scalar"

    def conditioned_potential(self, difference: Array, even: Array, /) -> Array:
        convex = (
            difference.reshape(())
            if self.certificate.convex_input_size == "scalar"
            else difference.reshape((1,))
            if isinstance(self.certificate.convex_input_size, int)
            else difference.reshape(self.certificate.convex_input_size)
        )
        if isinstance(self.potential, InputConvexNetwork):
            return self.potential(convex).reshape(())
        size = self.certificate.context_size
        if size is None:
            raise ValueError(
                "A conditioned potential requires an explicit certified context size."
            )
        context = (
            even.reshape(())
            if size == "scalar"
            else even.reshape((size,))
            if isinstance(size, int)
            else even.reshape(size)
        )
        return self.potential((context, convex)).reshape(())

    def edge_flux(self, difference: Array, even: Array, odd: Array, /) -> Array:
        del odd
        slope = jax.grad(self.conditioned_potential, argnums=0)
        return self.background_conductance * difference + 0.5 * (
            slope(difference, even) - slope(-difference, even)
        )

    def energy(self, difference: Array, even: Array, /) -> Array:
        return (
            0.5 * self.background_conductance * difference**2
            + 0.5
            * (
                self.conditioned_potential(difference, even)
                + self.conditioned_potential(-difference, even)
            )
            - self.conditioned_potential(jnp.zeros_like(difference), even)
        )

    def model_execution_contract(self) -> ModelExecutionContract:
        return network_execution_contract(self, DerivativeRegularity.smooth())


@final
class EdgeModelLipschitzCertificate(StrictModule, NonTrainableState):
    """A global derivative bound derived from canonical affine layer weights.

    The Frobenius norm bounds the Euclidean operator norm without a Lanczos
    estimate. Certificates are derived again for the current model on admission;
    a caller-supplied number never certifies a model. Supported activations have
    analytical global derivative bounds, including identity, tanh and softplus.
    """

    __strict_contract__ = True
    bound: Float[Scalar]
    construction: str = eqx.field(static=True)

    @classmethod
    def from_model(cls, model: Linear, /) -> EdgeModelLipschitzCertificate:
        if not isinstance(model, Linear):
            raise TypeError(
                "Certified edge fluxes currently require a canonical dense Linear model."
            )
        raw_weight = model.weight
        if isinstance(raw_weight, LowRankUpdate):
            raise TypeError(
                "Low-rank edge flux models have no admitted dense-weight Lipschitz certificate."
            )
        if not eqx.is_array(raw_weight):
            raise TypeError("Certified edge fluxes require dense array weights.")
        if model.activation not in (
            _identity,
            jax.nn.tanh,
            jnp.tanh,
            jax.nn.softplus,
            jax.nn.sigmoid,
        ):
            raise ValueError(
                "The model activation has no admitted global derivative bound."
            )
        weight = (
            raw_weight
            if model.weight_transform is None
            else model.weight_transform(raw_weight)
        )
        if model.rwf_log_scales is not None:
            weight = jnp.exp(model.rwf_log_scales)[:, None] * weight
        multiplier = 0.25 if model.activation is jax.nn.sigmoid else 1.0
        bound = multiplier * jnp.sqrt(jnp.sum(jnp.abs(weight) ** 2))
        finite_bias = (
            jnp.asarray(True) if model.bias is None else jnp.all(jnp.isfinite(model.bias))
        )
        bound = eqx.error_if(
            bound,
            ~jnp.isfinite(bound) | ~finite_bias,
            "Model Lipschitz evidence requires finite effective weights and bias.",
        )
        return cls(bound, "affine-frobenius-times-analytical-activation-bound")


@final
class LipschitzEdgeFlux(AbstractEdgeConstitutiveLaw):
    """Background diffusion plus a certified, orientation-odd learned perturbation.

    A global model derivative bound is not itself a contraction claim. The
    conservation owner derives a bound in the anchored background-energy norm,
    or carries CONTRACT_UNCERTIFIED when its hypotheses are unavailable.
    """

    model: Linear
    even_size: int = eqx.field(static=True)
    odd_size: int = eqx.field(static=True)
    background_conductance: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        model: Linear,
        /,
        *,
        even_size: int = 0,
        odd_size: int = 0,
        background_conductance: float = 1.0,
    ) -> None:
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in (even_size, odd_size)
        ):
            raise ValueError("Feature widths must be nonnegative integers.")
        EdgeModelLipschitzCertificate.from_model(model)
        if model.in_size != 1 + even_size + odd_size or model.out_size != "scalar":
            raise ValueError(
                "A certified flux model must map one packed edge to a scalar."
            )
        regularity = model._value_regularity()
        if regularity is None:
            raise ValueError(
                "Implicit edge flux models require declared classical C1 regularity."
            )
        level, conditions = regularity.admits_order(1)
        if level is not GradientLevel.SMOOTH or conditions:
            raise ValueError(
                "Implicit edge flux models require unconditional classical C1 regularity."
            )
        background = _positive_background(background_conductance)
        self.model = model
        self.even_size = even_size
        self.odd_size = odd_size
        self.background_conductance = background
        self.in_size = 1 + even_size + odd_size
        self.out_size = "scalar"

    @property
    def certificate(self) -> EdgeModelLipschitzCertificate:
        """Evidence for the current weights, never a stale pre-training bound."""
        return EdgeModelLipschitzCertificate.from_model(self.model)

    def perturbation(self, difference: Array, even: Array, odd: Array, /) -> Array:
        forward = jnp.concatenate((difference.reshape((1,)), even, odd))
        reverse = jnp.concatenate(((-difference).reshape((1,)), even, -odd))
        return 0.5 * (self.model(forward) - self.model(reverse))

    def edge_flux(self, difference: Array, even: Array, odd: Array, /) -> Array:
        return self.background_conductance * difference + self.perturbation(
            difference, even, odd
        )

    def certified_lipschitz_bound(self) -> Array:
        return self.certificate.bound

    def model_execution_contract(self) -> ModelExecutionContract:
        return network_execution_contract(self, self.model._value_regularity())


__all__ = [
    "AbstractEdgeConstitutiveLaw",
    "EdgeFeatureField",
    "EdgeFeatureKind",
    "EdgeFrameFeatures",
    "EdgeModelLipschitzCertificate",
    "LipschitzEdgeFlux",
    "MonotoneEdgeConductance",
]
