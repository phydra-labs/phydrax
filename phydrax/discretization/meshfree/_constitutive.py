# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Frame-covariant edge features and construction-certified edge flux laws.

Scalar laws act on one scalar jump per edge. Coupled laws act on a packed O(3)
component state per edge (scalars, pseudoscalars, vectors, pseudovectors and
rank-two irreducible tensors) and return a flux in the same representation.
"""

from __future__ import annotations

from abc import abstractmethod
from math import isfinite, prod
from typing import Any, ClassVar, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._differentiation import (
    AbstractConstructionCertificate,
    DerivativeRegularity,
    GradientLevel,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import AbstractArrayModel, ModelExecutionContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import prepare_linearization, PreparedLinearization
from ...nn._contracts import network_execution_contract
from ...nn._utils import _identity
from ...nn.layers import Linear
from ...nn.models import (
    InputConvexCertificate,
    InputConvexNetwork,
    PartiallyInputConvexNetwork,
)
from ...nn.operator.layers import O3TensorProduct, O3TensorProductPlan
from ...nn.operator.representations import O3Features, O3Representation
from ...nn.parameters import LowRankUpdate
from ...typing import Dim, Float, Float64, Int32, parse, PRNGKey, Scalar, Size
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
    """A global derivative bound derived from canonical model weights.

    The Frobenius norm bounds the Euclidean operator norm without a Lanczos
    estimate. Certificates are derived again for the current model on admission;
    a caller-supplied number never certifies a model. Supported activations have
    analytical global derivative bounds, including identity, tanh and softplus.
    For a coupled O(3) edge network the bound is one Euclidean bound on the whole
    packed output, never a concatenation of per-component estimates.
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

    @classmethod
    def from_equivariant_network(
        cls, network: O3EdgeNetwork, /
    ) -> EdgeModelLipschitzCertificate:
        """Bound ``|G(a)-G(b)| <= L |a-b|`` in the packed state, for every unit tangent.

        Each tensor product is linear in the state for a fixed edge frame, and
        its matrix at tangent ``Q e_z`` is ``D_H(Q) M(e_z) D_R(Q)^T`` with
        orthogonal representation matrices. Rotations act transitively on unit
        tangents, so the Frobenius norm at ``e_z`` is exact for every edge. The
        channel squash ``y / sqrt(1 + |y|^2)`` is 1-Lipschitz per channel.
        """
        if not isinstance(network, O3EdgeNetwork):
            raise TypeError(
                "Coupled Lipschitz certificates require a canonical O3EdgeNetwork."
            )
        frame = _edge_frame(jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64))
        state_matrix = jax.jacfwd(lambda value: network.state_layer(value, frame))(
            jnp.zeros((network.representation.packed_size,), dtype=jnp.float64)
        )
        output_matrix = jax.jacfwd(lambda value: network.output_layer(value, frame))(
            jnp.zeros((network.hidden.packed_size,), dtype=jnp.float64)
        )
        bound = jnp.sqrt(jnp.sum(jnp.square(state_matrix))) * jnp.sqrt(
            jnp.sum(jnp.square(output_matrix))
        )
        bound = eqx.error_if(
            bound,
            ~jnp.isfinite(bound),
            "Coupled Lipschitz evidence requires finite tensor-product weights.",
        )
        return cls(bound, "o3-frame-frobenius-times-unit-channel-squash")


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


def _tangent_representation() -> O3Representation:
    return O3Representation(scalars=1, vectors=1, tensors=1)


def _multiplicities(representation: O3Representation, /) -> tuple[int, ...]:
    return (
        representation.scalars,
        representation.pseudoscalars,
        representation.vectors,
        representation.pseudovectors,
        representation.tensors,
        representation.pseudotensors,
    )


def _edge_frame(tangent: Array, /) -> Array:
    """Packed polar frame ``(1, t, t t^T - I/3)`` of one unit edge tangent."""
    dtype = tangent.dtype
    return _tangent_representation().join(
        O3Features(
            scalars=jnp.ones((1,), dtype=dtype),
            pseudoscalars=jnp.zeros((0,), dtype=dtype),
            vectors=tangent[None],
            pseudovectors=jnp.zeros((0, 3), dtype=dtype),
            tensors=(jnp.outer(tangent, tangent) - jnp.eye(3, dtype=dtype) / 3)[None],
            pseudotensors=jnp.zeros((0, 3, 3), dtype=dtype),
        )
    )


def _edge_product(
    left: O3Representation, output: O3Representation, key: PRNGKey, /
) -> O3TensorProduct:
    return O3TensorProduct(
        O3TensorProductPlan(left, _tangent_representation(), output), key=key
    )


def _require_edge_product(
    product: O3TensorProduct,
    left: O3Representation,
    output: O3Representation,
    name: str,
    /,
) -> None:
    if not isinstance(product, O3TensorProduct):
        raise TypeError(f"{name} must be a prepared O3TensorProduct.")
    if product.weight is None:
        raise ValueError(
            f"{name} must own its path weights; externally weighted products are not admitted."
        )
    plan = product.plan
    if (
        _multiplicities(plan.left_representation) != _multiplicities(left)
        or _multiplicities(plan.right_representation)
        != _multiplicities(_tangent_representation())
        or _multiplicities(plan.output_representation) != _multiplicities(output)
    ):
        raise ValueError(
            f"{name} must map the declared representation and the polar edge frame to its declared output."
        )


def _require_representation(value: object, name: str, /) -> O3Representation:
    if not isinstance(value, O3Representation):
        raise TypeError(f"{name} must be an O3Representation.")
    return value


def _feature_width(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


@final
class O3EdgeInvariants(StrictModule):
    """Convex O(3)-invariant features of a packed edge state and its tangent.

    ``linear`` maps (state, polar edge frame) to true scalars ``l`` that are
    linear in the state. ``quadratic`` maps them to an equivariant
    representation whose squared channel norms ``q`` are convex in the state.
    The feature vector is ``(l, -l, q)``; a potential that is convex and
    nondecreasing in every feature is therefore convex in the whole packed
    state. Pseudo channels have no linear invariant against the polar frame and
    enter only through ``q``. Every feature is invariant under simultaneous
    proper or improper frame changes of state and tangent.
    """

    representation: O3Representation
    quadratic_representation: O3Representation
    linear: O3TensorProduct | None
    quadratic: O3TensorProduct
    linear_count: int = eqx.field(static=True)

    def __init__(
        self,
        representation: O3Representation,
        /,
        *,
        quadratic_representation: O3Representation,
        linear_count: int,
        key: PRNGKey,
    ) -> None:
        state = _require_representation(representation, "representation")
        quadratic = _require_representation(
            quadratic_representation, "quadratic_representation"
        )
        count = _feature_width(linear_count, "linear_count")
        linear_key, quadratic_key = jr.split(key)
        self.representation = state
        self.quadratic_representation = quadratic
        self.linear = (
            None
            if count == 0
            else _edge_product(state, O3Representation(scalars=count), linear_key)
        )
        self.quadratic = _edge_product(state, quadratic, quadratic_key)
        self.linear_count = count

    @property
    def size(self) -> int:
        """Width of the invariant feature vector ``(l, -l, q)``."""
        return 2 * self.linear_count + self.quadratic_representation.channel_count

    def require_canonical(self) -> None:
        """Refuse replaced or externally weighted invariant tensor products."""
        if self.linear_count == 0:
            if self.linear is not None:
                raise ValueError("A zero linear_count admits no linear invariant map.")
        elif self.linear is None:
            raise ValueError("Declared linear invariants require their tensor product.")
        else:
            _require_edge_product(
                self.linear,
                self.representation,
                O3Representation(scalars=self.linear_count),
                "linear invariants",
            )
        _require_edge_product(
            self.quadratic,
            self.representation,
            self.quadratic_representation,
            "quadratic invariants",
        )

    def __call__(self, difference: Array, frame: Array, /) -> Array:
        squared = self.quadratic_representation.channel_squared_norms(
            self.quadratic(difference, frame)
        )
        if self.linear is None:
            return squared
        linear = self.linear(difference, frame)
        return jnp.concatenate((linear, -linear, squared))


@final
class InvariantConvexEdgeCertificate(AbstractConstructionCertificate):
    """Construction evidence of a strongly monotone, O(3)-covariant coupled flux.

    The flux is ``F(D) = b D + grad Phi_s(D) - grad Phi_s(0)``, where ``Phi_s``
    symmetrizes ``psi(l, -l, q; c)`` over edge reversal. ``psi`` is input-convex
    and nondecreasing in every invariant, ``l`` is linear and ``q`` convex in the
    whole packed state ``D``; hence ``Phi_s`` is convex in ``D`` and
    ``<F(D1)-F(D2), D1-D2> >= b |D1-D2|^2`` for every pair of states, with no
    componentwise argument. The certificate holds for every parameter value.
    """

    capability_id: ClassVar[str] = "invariant-convex-edge-potential"
    potential: InputConvexCertificate
    state_multiplicities: tuple[int, ...] = eqx.field(static=True)
    quadratic_multiplicities: tuple[int, ...] = eqx.field(static=True)
    linear_count: int = eqx.field(static=True)
    strong_monotonicity: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        invariants: O3EdgeInvariants,
        potential: InputConvexCertificate,
        /,
        *,
        strong_monotonicity: float,
    ) -> None:
        if not isinstance(invariants, O3EdgeInvariants) or not isinstance(
            potential, InputConvexCertificate
        ):
            raise TypeError(
                "Invariant convex certificates bind O3EdgeInvariants and an InputConvexCertificate."
            )
        invariants.require_canonical()
        if potential.input_monotonicity != "nondecreasing":
            raise ValueError(
                "Only a potential nondecreasing in every invariant certifies state convexity."
            )
        strong_monotonicity = _positive_background(strong_monotonicity)
        state = _multiplicities(invariants.representation)
        quadratic = _multiplicities(invariants.quadratic_representation)
        self.potential = potential
        self.state_multiplicities = state
        self.quadratic_multiplicities = quadratic
        self.linear_count = invariants.linear_count
        self.strong_monotonicity = strong_monotonicity
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "invariant-convex-edge-certificate",
                "potential": potential.certificate_id,
                "state_multiplicities": list(state),
                "quadratic_multiplicities": list(quadratic),
                "linear_count": invariants.linear_count,
                "linear_plan": None
                if invariants.linear is None
                else invariants.linear.plan.plan_id,
                "quadratic_plan": invariants.quadratic.plan.plan_id,
                "strong_monotonicity": strong_monotonicity,
                "orientation": "edge-reversal-odd",
                "equilibrium": "zero-jump",
            }
        )


class AbstractCoupledEdgeConstitutiveLaw(AbstractArrayModel):
    """Coupled multi-component edge law ``F(D; t, even, odd)`` in an O(3) representation.

    ``D`` is the packed endpoint difference of one nodal component field and
    ``F`` lies in the same representation. Laws satisfy, exactly by
    construction, ``F(-D; -t, even, -odd) = -F(D; t, even, odd)`` and
    ``F(R D; Q t, ...) = R F(D; t, ...)`` for every orthogonal ``Q`` with
    representation action ``R``. Context is immutable external data: no law
    admits state-dependent features without a whole-state monotonicity proof.
    Bounds are whole-output Euclidean bounds in the packed state.
    """

    even_size: eqx.AbstractVar[int]
    odd_size: eqx.AbstractVar[int]
    background_conductance: eqx.AbstractVar[float]

    @property
    @abstractmethod
    def representation(self) -> O3Representation:
        raise NotImplementedError

    @abstractmethod
    def edge_flux(
        self, difference: Array, tangent: Array, even: Array, odd: Array, /
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def monotonicity_lower_bound(self) -> Array:
        """``m`` with ``<F(D1)-F(D2), D1-D2> >= m |D1-D2|^2``; nonpositive is no claim."""
        raise NotImplementedError

    @abstractmethod
    def lipschitz_bound(self) -> Array:
        """``L`` with ``|F(D1)-F(D2)| <= L |D1-D2|``; infinite when none is constructed."""
        raise NotImplementedError

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = jnp.asarray(x)
        if values.shape != (self.in_size,):
            raise ValueError(
                "A coupled constitutive model call requires one packed (state, tangent, even, odd) edge."
            )
        size = self.representation.packed_size
        return self.edge_flux(
            values[:size],
            values[size : size + 3],
            values[size + 3 : size + 3 + self.even_size],
            values[size + 3 + self.even_size :],
        )

    def _edge_inputs(
        self, differences: ArrayLike, features: EdgeFrameFeatures, /
    ) -> tuple[Array, Array, Array, Array]:
        if not isinstance(features, EdgeFrameFeatures):
            raise TypeError("Coupled edge laws require typed EdgeFrameFeatures.")
        if features.spatial_dimension != 3:
            raise ValueError(
                "Coupled O(3) edge laws require a three-dimensional Cartesian edge frame."
            )
        values = jnp.asarray(differences, dtype=jnp.float64)
        even, odd = features.even, features.odd
        if (
            values.shape != (features.lengths.size, self.representation.packed_size)
            or even.shape[1] != self.even_size
            or odd.shape[1] != self.odd_size
        ):
            raise ValueError(
                "Packed edge states and typed feature widths must match the coupled law."
            )
        return values, features.tangents, even, odd

    def flux(self, differences: ArrayLike, features: EdgeFrameFeatures, /) -> Array:
        """Packed flux of every edge, shape ``(edge, component)``."""
        values, tangents, even, odd = self._edge_inputs(differences, features)
        return jax.vmap(self.edge_flux)(values, tangents, even, odd)

    def derivative(self, differences: ArrayLike, features: EdgeFrameFeatures, /) -> Array:
        """Per-edge Jacobian blocks ``dF/dD``, shape ``(edge, component, component)``."""
        values, tangents, even, odd = self._edge_inputs(differences, features)
        return jax.vmap(jax.jacfwd(self.edge_flux, argnums=0))(
            values, tangents, even, odd
        )

    def linearize(
        self, differences: ArrayLike, features: EdgeFrameFeatures, /
    ) -> PreparedLinearization:
        """One retained flux value with matrix-free Jacobian and transpose actions."""
        values, tangents, even, odd = self._edge_inputs(differences, features)

        def flux(state: Array) -> Array:
            return jax.vmap(self.edge_flux)(state, tangents, even, odd)

        return prepare_linearization(flux, values)


@final
class MonotoneCoupledEdgeFlux(AbstractCoupledEdgeConstitutiveLaw):
    """Strongly monotone coupled flux, the gradient of an invariant convex potential.

    ``F(D) = b D + g_s(D) - g_s(0)`` with ``g_s(D) = (g(D; t, c+, c-) -
    g(-D; -t, c+, -c-)) / 2`` and ``g`` the state gradient of
    ``psi(invariants(D, t); even, odd)``. Monotonicity in every invariant is
    required: an arbitrary input-convex network of nonlinear invariants is not
    convex in the state. The law is the gradient of ``energy``, which is at least
    ``b |D|^2 / 2``; no global Lipschitz bound is claimed. Interactions that are
    odd under reversal, such as ``s (t . v)`` without odd context, cancel in the
    symmetrization; reversal-even couplings such as ``v1 . v2`` remain.
    """

    invariants: O3EdgeInvariants
    potential: InputConvexNetwork | PartiallyInputConvexNetwork
    certificate: InvariantConvexEdgeCertificate
    even_size: int = eqx.field(static=True)
    odd_size: int = eqx.field(static=True)
    background_conductance: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        invariants: O3EdgeInvariants,
        potential: InputConvexNetwork | PartiallyInputConvexNetwork,
        /,
        *,
        background_conductance: float = 1.0,
        odd_size: int = 0,
    ) -> None:
        if not isinstance(invariants, O3EdgeInvariants):
            raise TypeError("Coupled monotone laws require canonical O3EdgeInvariants.")
        invariants.require_canonical()
        if not isinstance(potential, (InputConvexNetwork, PartiallyInputConvexNetwork)):
            raise TypeError(
                "Monotone laws require a canonical input-convex model, not a user assertion."
            )
        convex = potential.input_convex_certificate()
        if convex.input_monotonicity != "nondecreasing":
            raise ValueError(
                "A potential of nonlinear invariants is convex in the state only when it is "
                "nondecreasing in every invariant; use input_monotonicity='nondecreasing'."
            )
        if convex.activation != "softplus":
            raise ValueError(
                "Implicit constitutive roots require a smooth potential; use softplus."
            )
        if convex.convex_input_size != invariants.size:
            raise ValueError(
                f"The potential's convex input must be the flat invariant vector of size {invariants.size}."
            )
        odd = _feature_width(odd_size, "odd_size")
        match convex.context_size:
            case None:
                context = 0
            case int() as width:
                context = width
            case _:
                raise ValueError(
                    "Coupled potentials take one flat context of even then odd features."
                )
        if odd > context:
            raise ValueError("odd_size exceeds the potential's declared context width.")
        background = _positive_background(background_conductance)
        self.invariants = invariants
        self.potential = potential
        self.certificate = InvariantConvexEdgeCertificate(
            invariants, convex, strong_monotonicity=background
        )
        self.even_size = context - odd
        self.odd_size = odd
        self.background_conductance = background
        self.in_size = invariants.representation.packed_size + 3 + context
        self.out_size = invariants.representation.packed_size

    @property
    def representation(self) -> O3Representation:
        return self.invariants.representation

    def invariant_potential(
        self, difference: Array, tangent: Array, even: Array, odd: Array, /
    ) -> Array:
        """Unsymmetrized ``psi(invariants(D, t); even, odd)``, convex in ``D``."""
        features = self.invariants(difference, _edge_frame(tangent))
        if isinstance(self.potential, InputConvexNetwork):
            return self.potential(features).reshape(())
        return self.potential((jnp.concatenate((even, odd)), features)).reshape(())

    def _odd_gradient(
        self, difference: Array, tangent: Array, even: Array, odd: Array, /
    ) -> Array:
        # Explicit antisymmetric difference keeps reversal parity exact in floating point.
        gradient = jax.grad(self.invariant_potential, argnums=0)
        return 0.5 * (
            gradient(difference, tangent, even, odd)
            - gradient(-difference, -tangent, even, -odd)
        )

    def edge_flux(
        self, difference: Array, tangent: Array, even: Array, odd: Array, /
    ) -> Array:
        return self.background_conductance * difference + (
            self._odd_gradient(difference, tangent, even, odd)
            - self._odd_gradient(jnp.zeros_like(difference), tangent, even, odd)
        )

    def energy(
        self, difference: Array, tangent: Array, even: Array, odd: Array, /
    ) -> Array:
        """Dissipation potential with gradient ``edge_flux`` and minimum zero at ``D = 0``."""

        def symmetric(value: Array) -> Array:
            return 0.5 * (
                self.invariant_potential(value, tangent, even, odd)
                + self.invariant_potential(-value, -tangent, even, -odd)
            )

        zero = jnp.zeros_like(difference)
        return (
            0.5 * self.background_conductance * jnp.vdot(difference, difference)
            + symmetric(difference)
            - symmetric(zero)
            - jnp.vdot(self._odd_gradient(zero, tangent, even, odd), difference)
        )

    def monotonicity_lower_bound(self) -> Array:
        return jnp.asarray(self.certificate.strong_monotonicity, dtype=jnp.float64)

    def lipschitz_bound(self) -> Array:
        return jnp.asarray(jnp.inf, dtype=jnp.float64)

    def model_execution_contract(self) -> ModelExecutionContract:
        return network_execution_contract(self, DerivativeRegularity.smooth())


@final
class O3EdgeNetwork(StrictModule):
    """Smooth O(3)-equivariant edge network with a whole-output Lipschitz bound.

    ``G(D; t, c) = W_out(nu(W_D(D, f) + W_c((1, c), f)), f)`` with polar frame
    ``f = (1, t, t t^T - I/3)``, O(3) tensor products ``W`` and the per-channel
    squash ``nu(y) = y / sqrt(1 + |y|^2)``, which is equivariant and
    1-Lipschitz. Context and the constant enter additively, so the state bound
    holds for every context.
    """

    representation: O3Representation
    hidden: O3Representation
    state_layer: O3TensorProduct
    context_layer: O3TensorProduct
    output_layer: O3TensorProduct
    context_size: int = eqx.field(static=True)

    def __init__(
        self,
        representation: O3Representation,
        hidden: O3Representation,
        /,
        *,
        context_size: int = 0,
        key: PRNGKey,
    ) -> None:
        state = _require_representation(representation, "representation")
        hidden_ = _require_representation(hidden, "hidden")
        width = _feature_width(context_size, "context_size")
        if hidden_.scalars + hidden_.vectors + hidden_.tensors == 0:
            raise ValueError(
                "The hidden representation needs a polar scalar, vector, or tensor channel "
                "to receive context and bias."
            )
        state_key, context_key, output_key = jr.split(key, 3)
        self.representation = state
        self.hidden = hidden_
        self.state_layer = _edge_product(state, hidden_, state_key)
        self.context_layer = _edge_product(
            O3Representation(scalars=1 + width), hidden_, context_key
        )
        self.output_layer = _edge_product(hidden_, state, output_key)
        self.context_size = width

    def require_canonical(self) -> None:
        """Refuse replaced or externally weighted tensor products."""
        _require_edge_product(
            self.state_layer, self.representation, self.hidden, "state_layer"
        )
        _require_edge_product(
            self.context_layer,
            O3Representation(scalars=1 + self.context_size),
            self.hidden,
            "context_layer",
        )
        _require_edge_product(
            self.output_layer, self.hidden, self.representation, "output_layer"
        )

    def __call__(self, difference: Array, frame: Array, context: Array, /) -> Array:
        constant = jnp.concatenate((jnp.ones((1,), dtype=frame.dtype), context))
        hidden = self.state_layer(difference, frame) + self.context_layer(constant, frame)
        squash = jax.lax.rsqrt(1 + self.hidden.channel_squared_norms(hidden))
        return self.output_layer(self.hidden.scale_channels(hidden, squash), frame)


@final
class LipschitzCoupledEdgeFlux(AbstractCoupledEdgeConstitutiveLaw):
    """Background diffusion plus an orientation-odd equivariant perturbation.

    ``F(D) = b D + (G(D; t, e, o) - G(-D; -t, e, -o)) / 2``. The perturbation's
    whole-output Lipschitz constant ``L`` is derived from the current weights,
    so ``F`` is ``(b + L)``-Lipschitz and ``(b - L)``-strongly monotone. As
    for scalar laws, a contraction claim belongs to the consuming solve.
    """

    network: O3EdgeNetwork
    even_size: int = eqx.field(static=True)
    odd_size: int = eqx.field(static=True)
    background_conductance: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        network: O3EdgeNetwork,
        /,
        *,
        even_size: int = 0,
        odd_size: int = 0,
        background_conductance: float = 1.0,
    ) -> None:
        if not isinstance(network, O3EdgeNetwork):
            raise TypeError(
                "Coupled Lipschitz laws require a canonical O3EdgeNetwork with a derived bound."
            )
        network.require_canonical()
        even = _feature_width(even_size, "even_size")
        odd = _feature_width(odd_size, "odd_size")
        if network.context_size != even + odd:
            raise ValueError(
                "The network context must be exactly the even then odd edge features."
            )
        EdgeModelLipschitzCertificate.from_equivariant_network(network)
        background = _positive_background(background_conductance)
        self.network = network
        self.even_size = even
        self.odd_size = odd
        self.background_conductance = background
        self.in_size = network.representation.packed_size + 3 + even + odd
        self.out_size = network.representation.packed_size

    @property
    def representation(self) -> O3Representation:
        return self.network.representation

    @property
    def certificate(self) -> EdgeModelLipschitzCertificate:
        """Evidence for the current weights, never a stale pre-training bound."""
        return EdgeModelLipschitzCertificate.from_equivariant_network(self.network)

    def perturbation(
        self, difference: Array, tangent: Array, even: Array, odd: Array, /
    ) -> Array:
        return 0.5 * (
            self.network(difference, _edge_frame(tangent), jnp.concatenate((even, odd)))
            - self.network(
                -difference, _edge_frame(-tangent), jnp.concatenate((even, -odd))
            )
        )

    def edge_flux(
        self, difference: Array, tangent: Array, even: Array, odd: Array, /
    ) -> Array:
        return self.background_conductance * difference + self.perturbation(
            difference, tangent, even, odd
        )

    def certified_lipschitz_bound(self) -> Array:
        """Whole-output Lipschitz bound of the learned perturbation alone."""
        return self.certificate.bound

    def monotonicity_lower_bound(self) -> Array:
        return self.background_conductance - self.certificate.bound

    def lipschitz_bound(self) -> Array:
        return self.background_conductance + self.certificate.bound

    def model_execution_contract(self) -> ModelExecutionContract:
        return network_execution_contract(self, DerivativeRegularity.smooth())


__all__ = [
    "AbstractCoupledEdgeConstitutiveLaw",
    "AbstractEdgeConstitutiveLaw",
    "EdgeFeatureField",
    "EdgeFeatureKind",
    "EdgeFrameFeatures",
    "EdgeModelLipschitzCertificate",
    "InvariantConvexEdgeCertificate",
    "LipschitzCoupledEdgeFlux",
    "LipschitzEdgeFlux",
    "MonotoneCoupledEdgeFlux",
    "MonotoneEdgeConductance",
    "O3EdgeInvariants",
    "O3EdgeNetwork",
]
