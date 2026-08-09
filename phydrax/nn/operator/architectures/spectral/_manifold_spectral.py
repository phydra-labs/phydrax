#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import opt_einsum as oe
from jax import core as jax_core
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._spectral._modal import SpectralDiscretization
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey, fold_in_eval_key
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


class _ManifoldSpectralMixer(StrictModule):
    """Gauge-safe channel map with one learned matrix per aligned eigenspace."""

    weight: Array
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    num_modes: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralDiscretization,
        /,
        *,
        in_channels: int,
        out_channels: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size = int(in_channels)
        out_size = int(out_channels)
        if min(in_size, out_size) <= 0:
            raise ValueError("Spectral mixer channels must be positive.")
        self.in_channels = in_size
        self.out_channels = out_size
        self.num_modes = plan.num_modes
        self.num_groups = plan.num_groups
        self.basis_id = plan.basis_id
        scale = 1.0 / jnp.sqrt(float(in_size))
        self.weight = scale * jr.normal(
            key,
            (plan.num_groups, out_size, in_size),
        )

    def __call__(
        self,
        values: Array,
        source: SpectralDiscretization,
        target: SpectralDiscretization,
        /,
    ) -> Array:
        array = jnp.asarray(values)
        if source.num_modes != self.num_modes or target.num_modes != self.num_modes:
            raise ValueError("Source and target spectral mode counts must match the mixer.")
        if source.num_groups != self.num_groups or target.num_groups != self.num_groups:
            raise ValueError("Source and target eigenspace groups must match the mixer.")
        if (
            not isinstance(source.group_ids, jax_core.Tracer)
            and not isinstance(target.group_ids, jax_core.Tracer)
            and not np.array_equal(
                np.asarray(source.group_ids), np.asarray(target.group_ids)
            )
        ):
            raise ValueError("Source and target eigenspace groups must align.")
        if source.basis_id != self.basis_id or target.basis_id != self.basis_id:
            raise ValueError("Source and target plans require the mixer's aligned basis_id.")
        if array.shape[-2:] != (source.num_points, self.in_channels):
            raise ValueError(
                "Manifold spectral values must end in source points/channels "
                f"{(source.num_points, self.in_channels)}; got {array.shape}."
            )
        coefficients = oe.contract("mp,...pc->...mc", source.analysis, array)
        mode_weight = self.weight[source.group_ids]
        transformed = oe.contract("moc,...mc->...mo", mode_weight, coefficients)
        return oe.contract("pm,...mo->...po", target.synthesis, transformed)


class ManifoldSpectralOperator(AbstractOperatorModel):
    """Intrinsic Laplace-eigenbasis neural operator on a fixed/aligned manifold."""

    operator_architecture = "ManifoldSpectralOperator"

    lift: Linear
    spectral_mixers: tuple[_ManifoldSpectralMixer, ...]
    pointwise: tuple[Linear, ...]
    projection: Linear
    source_plan: SpectralDiscretization
    target_plan: SpectralDiscretization
    source_key: str | None
    activation: Callable[[Array], Array]
    residual: bool
    cross_discretization: bool
    in_channels: int
    out_channels: int
    width: int
    depth: int
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        source_plan: SpectralDiscretization,
        /,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 64,
        depth: int = 4,
        target_plan: SpectralDiscretization | None = None,
        source_key: str | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        residual: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(source_plan, SpectralDiscretization):
            raise TypeError("source_plan must be a SpectralDiscretization.")
        if target_plan is not None and not isinstance(
            target_plan, SpectralDiscretization
        ):
            raise TypeError("target_plan must be a SpectralDiscretization.")
        self.source_plan = source_plan
        self.target_plan = source_plan if target_plan is None else target_plan
        if source_plan.num_modes != self.target_plan.num_modes:
            raise ValueError("Source and target spectral mode counts must match.")
        if source_plan.basis_id != self.target_plan.basis_id:
            raise ValueError("Source and target plans require one aligned basis_id.")
        if not bool(jnp.array_equal(source_plan.group_ids, self.target_plan.group_ids)):
            raise ValueError("Source and target eigenspace groups must match.")
        self.cross_discretization = (
            target_plan is not None and target_plan is not source_plan
        )
        self.source_key = source_key
        self.activation = activation
        self.residual = bool(residual)
        self.in_channels = _get_size(in_channels)
        self.out_channels = _get_size(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.in_size = in_channels
        self.out_size = out_channels
        if min(self.in_channels, self.out_channels, self.width, self.depth) <= 0:
            raise ValueError("Channels, width, and depth must be positive.")
        keys = jr.split(key, 2 * self.depth + 2)
        self.lift = Linear(
            in_size=self.in_channels,
            out_size=self.width,
            activation=None,
            rwf=False,
            key=keys[0],
        )
        self.spectral_mixers = tuple(
            _ManifoldSpectralMixer(
                self.source_plan,
                in_channels=self.width,
                out_channels=self.width,
                key=keys[1 + index],
            )
            for index in range(self.depth)
        )
        self.pointwise = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                rwf=False,
                key=keys[1 + self.depth + index],
            )
            for index in range(self.depth)
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=self.out_channels,
            activation=None,
            rwf=False,
            key=keys[-1],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "ManifoldSpectralOperator requires source_key for multiple inputs."
            )
        return next(iter(batch.inputs.values()))

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        source = self._source(batch)
        query = batch.require_single_query()
        if source.values is None:
            raise ValueError("Manifold spectral source values cannot be None.")
        if prod(source.sample_shape) != self.source_plan.num_points:
            raise ValueError("Source sample count does not match the spectral plan.")
        if prod(query.sample_shape) != self.target_plan.num_points:
            raise ValueError("Query sample count does not match the target spectral plan.")
        values = jnp.asarray(source.values)
        sample_ndim = len(source.sample_shape)
        trailing = values.shape[len(batch.case_shape) + sample_ndim :]
        if not trailing:
            if self.in_channels != 1:
                raise ValueError("Scalar source values require one input channel.")
            values = values[..., None]
        elif tuple(int(size) for size in trailing) != (self.in_channels,):
            raise ValueError("Manifold source channel shape is incompatible.")
        values = values.reshape(
            batch.case_shape + (self.source_plan.num_points, self.in_channels)
        )
        source_mask = source.mask_array(case_shape=batch.case_shape).reshape(
            batch.case_shape + (self.source_plan.num_points, 1)
        )
        hidden = self.lift(values * source_mask, key=fold_in_eval_key(key, 0))
        for index, (mixer, pointwise) in enumerate(
            zip(self.spectral_mixers, self.pointwise, strict=True)
        ):
            current_plan = self.source_plan if index == 0 else self.target_plan
            spectral_update = mixer(hidden, current_plan, self.target_plan)
            if index == 0 and self.cross_discretization:
                hidden = self.activation(spectral_update)
                continue
            update = spectral_update + pointwise(
                hidden,
                key=fold_in_eval_key(key, 2 * index + 1),
            )
            hidden = self.activation(hidden + update if self.residual else update)
        output = self.projection(
            hidden,
            key=fold_in_eval_key(key, 2 * self.depth + 1),
        )
        query_mask = query.mask_array(case_shape=batch.case_shape).reshape(
            batch.case_shape + (self.target_plan.num_points, 1)
        )
        output = output * query_mask
        output = output.reshape(
            batch.case_shape + query.sample_shape + (self.out_channels,)
        )
        return output[..., 0] if self.out_size == "scalar" else output

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("ManifoldSpectralOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["ManifoldSpectralOperator", "SpectralDiscretization"]
