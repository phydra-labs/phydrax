#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Joint categorical/continuous case and atom embeddings of the MACE family.

The source ``GenericJointEmbedding`` embeds every declared feature (a lookup
table for categorical codes, ``Linear -> SiLU -> Linear`` for continuous
values), concatenates the embeddings in declaration order, and adds
``SiLU(concat @ P)`` to the species embedding of every atom. Case-level
features are broadcast to their atoms. An optional linear embedding readout
reads that sum into a per-atom energy that the source adds to its reference
energy, unscaled.

Feature values are typed conditioning inputs. A categorical code ``c`` selects
row ``c + offset``; codes outside ``[-offset, classes - offset)`` are declared
category violations that make the case unsuccessful, never a clamped row.
``total_charge`` binds the physical net charge in elementary charges and
``total_spin`` the physical spin multiplicity ``2S + 1`` (the source's
conventions); every other feature binds a MACE source-feature port whose
physical meaning is not asserted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, ParameterOwner
from ..._validation import canonical_identifier, integer, positive_integer
from ...atomistic._conditioning import (
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionKind,
    AtomisticConditionLevel,
    SPIN_MULTIPLICITY_PORT,
    TOTAL_CHARGE_PORT,
)
from ...typing import parse, PRNGKey
from ...units import ELEMENTARY_CHARGE


MACEEmbeddingKind: TypeAlias = Literal["categorical", "continuous"]

_TOTAL_CHARGE = "total_charge"
_TOTAL_SPIN = "total_spin"


def mace_embedding_port(
    name: str, kind: MACEEmbeddingKind, input_width: int, /
) -> ValuePort:
    """Port of one source joint-embedding feature.

    ``total_charge`` and ``total_spin`` are the source's physical net charge and
    spin multiplicity; every other feature is an opaque source feature.
    """
    match name:
        case "total_charge":
            return TOTAL_CHARGE_PORT
        case "total_spin":
            return SPIN_MULTIPLICITY_PORT
        case _:
            shape: tuple[int, ...] = () if input_width == 1 else (input_width,)
            return ValuePort(
                f"mace.joint-embedding.{name}",
                event_shape=shape,
                component_ids=(name,)
                if not shape
                else tuple(f"{name}[{index}]" for index in range(input_width)),
                representation=f"mace-source-{kind}-feature",
                normalization_id="identity",
            )


@final
class MACEEmbeddingFeature(StrictModule, NonTrainableState):
    """Static declaration of one source joint-embedding feature.

    ``level`` is ``case`` (source ``per: graph``) or ``atom`` (``per: node``).
    Categorical features declare ``classes`` and ``offset``; continuous features
    declare ``input_width`` and whether their two affine maps carry biases.
    """

    name: str = eqx.field(static=True)
    kind: MACEEmbeddingKind = eqx.field(static=True)
    level: AtomisticConditionLevel = eqx.field(static=True)
    width: int = eqx.field(static=True)
    classes: int | None = eqx.field(static=True)
    offset: int = eqx.field(static=True)
    input_width: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: MACEEmbeddingKind,
        /,
        *,
        level: AtomisticConditionLevel | str,
        width: int,
        classes: int | None = None,
        offset: int = 0,
        input_width: int = 1,
        bias: bool = True,
    ) -> None:
        name_ = canonical_identifier(name, "name")
        kind_ = parse(kind, MACEEmbeddingKind, "kind")
        level_ = parse(
            AtomisticConditionLevel(level) if isinstance(level, str) else level,
            AtomisticConditionLevel,
            "level",
        )
        width_ = positive_integer(width, "width")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool.")
        match kind_:
            case "categorical":
                if classes is None:
                    raise ValueError("A categorical feature declares its classes.")
                classes_: int | None = positive_integer(classes, "classes")
                offset_ = integer(offset, "offset")
                if input_width != 1:
                    raise ValueError("A categorical feature reads one integer code.")
                inputs = 1
            case "continuous":
                if classes is not None or offset != 0:
                    raise ValueError("A continuous feature has no classes or offset.")
                classes_ = None
                offset_ = 0
                inputs = positive_integer(input_width, "input_width")
                if level_ is AtomisticConditionLevel.CASE and inputs != 1:
                    raise ValueError(
                        "The source broadcasts only scalar case-level continuous "
                        "features to atoms."
                    )
            case unreachable:
                assert_never(unreachable)
        if name_ in (_TOTAL_CHARGE, _TOTAL_SPIN) and (
            kind_ != "categorical" or level_ is not AtomisticConditionLevel.CASE
        ):
            raise ValueError(
                f"{name_!r} binds the physical case quantity as a categorical code."
            )
        self.name = name_
        self.kind = kind_
        self.level = level_
        self.width = width_
        self.classes = classes_
        self.offset = offset_
        self.input_width = inputs
        self.bias = bias
        self.feature_id = canonical_fingerprint(
            {
                "kind": "mace-joint-embedding-feature",
                "name": name_,
                "feature_kind": kind_,
                "level": level_.value,
                "width": width_,
                "classes": classes_,
                "offset": offset_,
                "input_width": inputs,
                "bias": bias,
            }
        )

    @property
    def port(self) -> ValuePort:
        return mace_embedding_port(self.name, self.kind, self.input_width)

    def condition_field(self) -> AtomisticConditionField:
        """The typed conditioning input this feature consumes."""
        match self.kind:
            case "categorical":
                classes = self.classes
                if classes is None:
                    raise RuntimeError("A categorical feature lost its classes.")
                return AtomisticConditionField(
                    self.name,
                    self.port,
                    level=self.level,
                    kind=AtomisticConditionKind.CATEGORICAL,
                    dtype=jnp.int64,
                    categories=tuple(range(-self.offset, classes - self.offset)),
                    unit=ELEMENTARY_CHARGE if self.name == _TOTAL_CHARGE else None,
                )
            case "continuous":
                return AtomisticConditionField(
                    self.name,
                    self.port,
                    level=self.level,
                    kind=AtomisticConditionKind.CONTINUOUS,
                    dtype=jnp.float64,
                )
            case unreachable:
                assert_never(unreachable)


def mace_embedding_conditioning(
    features: Sequence[MACEEmbeddingFeature], /
) -> AtomisticConditioningSpec:
    """Conditioning specification of a declared joint embedding."""
    return AtomisticConditioningSpec(
        tuple(feature.condition_field() for feature in features)
    )


def require_mace_conditioning(
    features: Sequence[MACEEmbeddingFeature],
    conditioning: AtomisticConditioning | None,
    /,
) -> None:
    """Refuse conditioning that lacks any declared feature binding exactly."""
    if not features:
        return
    if conditioning is None:
        raise ValueError(
            "This MACE model's joint embedding needs conditioning of "
            f"{tuple(feature.name for feature in features)!r}."
        )
    spec = conditioning.spec
    for feature in features:
        field = feature.condition_field()
        if field.name not in spec.names:
            raise ValueError(f"Conditioning lacks the embedding feature {field.name!r}.")
        declared = spec.fields[spec.index(field.name)]
        if declared.field_id != field.field_id:
            raise ValueError(
                f"Conditioning field {field.name!r} does not bind the declared "
                "embedding feature (port, kind, dtype or categories differ)."
            )
        if not conditioning.present(field.name):
            raise ValueError(f"Embedding feature {field.name!r} is absent.")


@final
class MACEJointEmbedding(StrictModule, ParameterOwner):
    """Source joint embedding added to the species embedding of every atom.

    ``parameters[k]`` holds feature ``k``'s leaves in source order: a
    categorical ``(table[classes, width],)`` or a continuous
    ``(W1[input, width], b1[width], W2[width, width], b2[width])`` (biases
    omitted when the feature declares none). ``projection`` is the bias-free
    ``[sum(width), channels]`` map. Every array is a PARAMETER leaf in the
    transpose of the source ``Linear.weight`` layout.
    """

    features: tuple[MACEEmbeddingFeature, ...] = eqx.field(static=True)
    parameters: tuple[tuple[Array, ...], ...]
    projection: Array

    def __init__(
        self,
        features: Sequence[MACEEmbeddingFeature],
        parameters: Sequence[Sequence[ArrayLike]],
        projection: ArrayLike,
        /,
    ) -> None:
        declared = tuple(features)
        if not declared or any(
            not isinstance(feature, MACEEmbeddingFeature) for feature in declared
        ):
            raise TypeError("features must be non-empty MACEEmbeddingFeature values.")
        if len({feature.name for feature in declared}) != len(declared):
            raise ValueError("Joint-embedding feature names must be distinct.")
        arrays = tuple(
            tuple(jnp.asarray(value) for value in group) for group in parameters
        )
        projection_ = jnp.asarray(projection)
        dtype = projection_.dtype
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("Joint-embedding parameters must be float32 or float64.")
        if len(arrays) != len(declared):
            raise ValueError("Every joint-embedding feature needs its parameters.")
        for feature, group in zip(declared, arrays, strict=True):
            expected = _parameter_shapes(feature)
            if tuple(value.shape for value in group) != expected or any(
                value.dtype != dtype for value in group
            ):
                raise ValueError(
                    f"Feature {feature.name!r} parameters must have shapes {expected} "
                    "in the projection dtype."
                )
        total = sum(feature.width for feature in declared)
        if projection_.ndim != 2 or projection_.shape[0] != total:
            raise ValueError("The projection must map the concatenated embeddings.")
        self.features = declared
        self.parameters = arrays
        self.projection = projection_

    @classmethod
    def initialize(
        cls,
        features: Sequence[MACEEmbeddingFeature],
        channel_count: int,
        /,
        *,
        key: PRNGKey,
        dtype: DTypeLike,
    ) -> MACEJointEmbedding:
        """Source initialization: unit-normal tables, uniform fan-in affine maps."""
        declared = tuple(features)
        keys = jr.split(key, len(declared) + 1)
        groups = []
        for feature, feature_key in zip(declared, keys[:-1]):
            shapes = _parameter_shapes(feature)
            match feature.kind:
                case "categorical":
                    groups.append((jr.normal(feature_key, shapes[0], dtype=dtype),))
                case "continuous":
                    parts = jr.split(feature_key, len(shapes))
                    fan_ins = (
                        (feature.input_width,) * 2 + (feature.width,) * 2
                        if feature.bias
                        else (feature.input_width, feature.width)
                    )
                    groups.append(
                        tuple(
                            jr.uniform(
                                part,
                                shape,
                                dtype,
                                -1.0 / math.sqrt(fan_in),
                                1.0 / math.sqrt(fan_in),
                            )
                            for part, shape, fan_in in zip(parts, shapes, fan_ins)
                        )
                    )
                case unreachable:
                    assert_never(unreachable)
        total = sum(feature.width for feature in declared)
        bound = 1.0 / math.sqrt(total)
        projection = jr.uniform(keys[-1], (total, channel_count), dtype, -bound, bound)
        return cls(declared, groups, projection)

    @property
    def channel_count(self) -> int:
        return self.projection.shape[1]

    @property
    def conditioning(self) -> AtomisticConditioningSpec:
        return mace_embedding_conditioning(self.features)

    def _feature(
        self,
        feature: MACEEmbeddingFeature,
        group: tuple[Array, ...],
        conditioning: AtomisticConditioning,
        atom_cases: Array,
        /,
    ) -> Array:
        value = jnp.asarray(conditioning.value(feature.name))
        if feature.level is AtomisticConditionLevel.CASE:
            value = value[atom_cases]
        dtype = self.projection.dtype
        match feature.kind:
            case "categorical":
                classes = feature.classes
                if classes is None:
                    raise RuntimeError("A categorical feature lost its classes.")
                code = value.astype(jnp.int64) + feature.offset
                # Out-of-range codes are category violations of the conditioning
                # status (the case is unsuccessful); the bounded row read only
                # keeps the gather in range.
                row = jnp.clip(code, 0, classes - 1).astype(jnp.int32)
                return group[0][row]
            case "continuous":
                inputs = value.astype(dtype).reshape(
                    (value.shape[0], feature.input_width)
                )
                if feature.bias:
                    first, first_bias, second, second_bias = group
                    hidden = jax.nn.silu(inputs @ first + first_bias)
                    return hidden @ second + second_bias
                first, second = group
                return jax.nn.silu(inputs @ first) @ second
            case unreachable:
                assert_never(unreachable)

    def __call__(
        self,
        conditioning: AtomisticConditioning,
        atom_cases: Array,
        atom_mask: Array,
        /,
    ) -> Array:
        """Per-atom addend ``SiLU(concat_k e_k @ P)`` (zero on inactive atoms)."""
        embeddings = [
            self._feature(feature, group, conditioning, atom_cases)
            for feature, group in zip(self.features, self.parameters, strict=True)
        ]
        joint = jax.nn.silu(jnp.concatenate(embeddings, axis=-1) @ self.projection)
        return jnp.where(atom_mask[:, None], joint, jnp.zeros((), dtype=joint.dtype))


def _parameter_shapes(feature: MACEEmbeddingFeature, /) -> tuple[tuple[int, ...], ...]:
    match feature.kind:
        case "categorical":
            classes = feature.classes
            if classes is None:
                raise RuntimeError("A categorical feature lost its classes.")
            return ((classes, feature.width),)
        case "continuous":
            width = feature.width
            if feature.bias:
                return ((feature.input_width, width), (width,), (width, width), (width,))
            return ((feature.input_width, width), (width, width))
        case unreachable:
            assert_never(unreachable)


def joint_embedding_identity(embedding: MACEJointEmbedding | None, /) -> str | None:
    return (
        None
        if embedding is None
        else canonical_fingerprint(
            {
                "kind": "mace-joint-embedding",
                "features": [feature.feature_id for feature in embedding.features],
                "channels": embedding.channel_count,
                "dtype": np.dtype(embedding.projection.dtype).name,
            }
        )
    )


__all__ = [
    "joint_embedding_identity",
    "mace_embedding_conditioning",
    "mace_embedding_port",
    "MACEEmbeddingFeature",
    "MACEEmbeddingKind",
    "MACEJointEmbedding",
    "require_mace_conditioning",
]

for _artifact in (MACEEmbeddingFeature, MACEJointEmbedding):
    register_artifact_value(f"phydrax.nn.atomistic:{_artifact.__name__}", _artifact)
del _artifact
