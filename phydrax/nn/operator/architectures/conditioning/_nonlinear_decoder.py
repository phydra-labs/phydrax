#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._frozendict import frozendict
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey, split_eval_key
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.architectures.conditioning._deeponet import (
    _AbstractBranchEncoder,
    BranchFusion,
    FixedBranchEncoder,
)
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.encoded import AbstractEncodedOperatorModel


class CoordinateDecoderState(StrictModule):
    """Function-level latent code retained for independent query decoding."""

    latent: Array
    case_shape: tuple[int, ...]
    source_names: tuple[str, ...]

    def __init__(
        self,
        latent: Array,
        /,
        *,
        case_shape: Sequence[int],
        source_names: Sequence[str],
    ):
        value = jnp.asarray(latent)
        cases = tuple(int(size) for size in case_shape)
        if value.ndim != len(cases) + 1 or value.shape[: len(cases)] != cases:
            raise ValueError(
                "Coordinate decoder latent must have shape case_shape + (channels,)."
            )
        self.latent = value
        self.case_shape = cases
        self.source_names = tuple(str(name) for name in source_names)


class FiLMCoordinateDecoder(StrictModule):
    """Nonlinear coordinate decoder modulated at every layer by a function code."""

    coordinate_lift: Linear
    hidden: tuple[Linear, ...]
    modulation: tuple[Linear, ...]
    projection: Linear
    latent_size: int
    coord_dim: int
    width: int
    depth: int
    in_size: int
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        *,
        latent_size: int,
        coord_dim: int,
        out_size: int | Literal["scalar"] = "scalar",
        width: int = 128,
        depth: int = 4,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.latent_size = int(latent_size)
        self.coord_dim = int(coord_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.in_size = self.latent_size + self.coord_dim
        self.out_size = out_size
        if min(self.latent_size, self.coord_dim, self.width, self.depth) <= 0:
            raise ValueError("latent_size, coord_dim, width, and depth must be positive.")
        keys = jr.split(key, 2 * self.depth + 2)
        self.coordinate_lift = Linear(
            in_size=self.coord_dim,
            out_size=self.width,
            activation=jax.nn.tanh,
            rwf=False,
            key=keys[0],
        )
        self.hidden = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                rwf=False,
                key=keys[1 + index],
            )
            for index in range(self.depth)
        )
        self.modulation = tuple(
            Linear(
                in_size=self.latent_size,
                out_size=2 * self.width,
                activation=None,
                rwf=False,
                key=keys[1 + self.depth + index],
            )
            for index in range(self.depth)
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=out_size,
            activation=None,
            rwf=False,
            key=keys[-1],
        )

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        value = jnp.asarray(x)
        if value.shape[-1:] != (self.in_size,):
            raise ValueError(
                f"FiLMCoordinateDecoder expects trailing size {self.in_size}."
            )
        latent = value[..., : self.latent_size]
        coordinates = value[..., self.latent_size :]
        hidden = self.coordinate_lift(coordinates, key=key)
        for layer, modulation in zip(
            self.hidden,
            self.modulation,
            strict=True,
        ):
            scale_shift = modulation(latent, key=key)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            hidden = jax.nn.tanh((1.0 + scale) * layer(hidden, key=key) + shift)
        return self.projection(hidden, key=key)


class CoordinateConditionedOperator(AbstractEncodedOperatorModel):
    """NOMAD-style operator with a nonlinear function-conditioned decoder."""

    operator_architecture = "CoordinateConditionedOperator"

    branches: frozendict[str, _AbstractBranchEncoder]
    decoder: Any
    branch_mixer: Any | None
    fusion: BranchFusion
    latent_size: int
    coord_dim: int
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        *,
        branch: Any | Mapping[str, Any],
        decoder: Any,
        coord_dim: int,
        latent_size: int,
        out_size: int | Literal["scalar"] = "scalar",
        in_size: int | Literal["scalar"] = "scalar",
        fusion: BranchFusion = "sum",
        branch_mixer: Any | None = None,
        source_key: str | None = None,
    ):
        self.latent_size = int(latent_size)
        self.coord_dim = int(coord_dim)
        self.in_size = in_size
        self.out_size = out_size
        self.fusion = fusion
        self.decoder = decoder
        self.branch_mixer = branch_mixer
        if self.latent_size <= 0 or self.coord_dim <= 0:
            raise ValueError("latent_size and coord_dim must be positive.")
        if fusion not in ("sum", "product", "concat"):
            raise ValueError("fusion must be 'sum', 'product', or 'concat'.")
        if isinstance(decoder, Linear):
            raise ValueError(
                "CoordinateConditionedOperator requires a genuinely nonlinear decoder."
            )
        if _get_size(decoder.in_size) != self.latent_size + self.coord_dim:
            raise ValueError("Decoder input size must equal latent_size + coord_dim.")
        if _get_size(decoder.out_size) != _get_size(out_size):
            raise ValueError("Decoder output size must match operator out_size.")

        if isinstance(branch, Mapping):
            branch_items = tuple((str(name), value) for name, value in branch.items())
        else:
            branch_items = ((source_key or "input", branch),)
        if not branch_items:
            raise ValueError("CoordinateConditionedOperator requires a branch encoder.")
        encoders: dict[str, _AbstractBranchEncoder] = {}
        for name, encoder in branch_items:
            if isinstance(encoder, _AbstractBranchEncoder):
                resolved = encoder
            else:
                resolved = FixedBranchEncoder(encoder, self.latent_size)
            if resolved.latent_size != self.latent_size:
                raise ValueError(f"Branch {name!r} has an incompatible latent size.")
            encoders[name] = resolved
        self.branches = frozendict(encoders)

        if fusion == "concat":
            if branch_mixer is None:
                raise ValueError("concat fusion requires branch_mixer.")
            if _get_size(branch_mixer.in_size) != len(encoders) * self.latent_size:
                raise ValueError("Concat branch mixer input size is incompatible.")
            if _get_size(branch_mixer.out_size) != self.latent_size:
                raise ValueError("Concat branch mixer output must match latent_size.")
        elif branch_mixer is not None:
            raise ValueError("branch_mixer is only valid for concat fusion.")

    def _source(self, batch: OperatorBatch, name: str, /) -> FunctionSamples:
        if name in batch.inputs:
            return batch.input(name)
        if len(self.branches) == 1 and len(batch.inputs) == 1:
            return next(iter(batch.inputs.values()))
        raise KeyError(f"No operator input matches branch {name!r}.")

    def encode_inputs(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> CoordinateDecoderState:
        keys = split_eval_key(key, len(self.branches) + 1)
        encoded = tuple(
            encoder(
                self._source(batch, name),
                case_ndim=len(batch.case_shape),
                key=keys[index],
            )
            for index, (name, encoder) in enumerate(self.branches.items())
        )
        if self.fusion == "sum":
            latent = encoded[0]
            for value in encoded[1:]:
                latent = latent + value
            latent = latent / jnp.sqrt(float(len(encoded)))
        elif self.fusion == "product":
            latent = encoded[0]
            for value in encoded[1:]:
                latent = latent * value
        else:
            branch_mixer = self.branch_mixer
            assert branch_mixer is not None
            concatenated = jnp.concatenate(encoded, axis=-1)
            flattened = concatenated.reshape((-1, concatenated.shape[-1]))
            latent = jax.vmap(lambda value: branch_mixer(value, key=keys[-1]))(
                flattened
            ).reshape(batch.case_shape + (self.latent_size,))
        return CoordinateDecoderState(
            latent,
            case_shape=batch.case_shape,
            source_names=tuple(self.branches),
        )

    def decode_query(
        self,
        state: CoordinateDecoderState,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        coordinates = query.coordinates_array(case_shape=state.case_shape)
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Expected query coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )
        sample_shape = query.sample_shape
        latent = state.latent.reshape(
            state.case_shape + (1,) * len(sample_shape) + (self.latent_size,)
        )
        latent = jnp.broadcast_to(
            latent,
            state.case_shape + sample_shape + (self.latent_size,),
        )
        features = jnp.concatenate((latent, coordinates), axis=-1)
        flattened = features.reshape((-1, self.latent_size + self.coord_dim))
        decoded = jax.vmap(lambda value: self.decoder(value, key=key))(flattened)
        channel_shape = () if self.out_size == "scalar" else (int(self.out_size),)
        output = jnp.asarray(decoded).reshape(
            state.case_shape + sample_shape + channel_shape
        )
        mask = query.mask_array(case_shape=state.case_shape)
        if channel_shape:
            mask = mask[..., None]
        return output * mask

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("CoordinateConditionedOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = [
    "CoordinateConditionedOperator",
    "CoordinateDecoderState",
    "FiLMCoordinateDecoder",
]
