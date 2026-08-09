#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._spectral._spherical import SphericalHarmonicPlan
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey, fold_in_eval_key
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.data import FunctionSamples, OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


class SphericalSpectralConv(StrictModule):
    """Scalar SO(3)-equivariant channel mixing in an S2FFT coefficient basis."""

    weight: Array
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    bandlimit: int = eqx.field(static=True)
    transform_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SphericalHarmonicPlan,
        /,
        *,
        in_channels: int,
        out_channels: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(plan, SphericalHarmonicPlan):
            raise TypeError("plan must be a SphericalHarmonicPlan.")
        in_size = int(in_channels)
        out_size = int(out_channels)
        if min(in_size, out_size) <= 0:
            raise ValueError("Spherical spectral channels must be positive.")
        self.in_channels = in_size
        self.out_channels = out_size
        self.bandlimit = plan.bandlimit
        self.transform_fingerprint = plan.fingerprint
        self.weight = jr.normal(
            key,
            shape=(plan.bandlimit, in_size, out_size),
        ) / jnp.sqrt(float(in_size))

    def __call__(
        self,
        values: Array,
        plan: SphericalHarmonicPlan,
        /,
    ) -> Array:
        array = jnp.asarray(values)
        if plan.fingerprint != self.transform_fingerprint:
            raise ValueError("Spherical layer and transform plan do not match.")
        if (
            array.ndim < 3
            or tuple(int(size) for size in array.shape[-3:-1]) != plan.sample_shape
            or int(array.shape[-1]) != self.in_channels
        ):
            raise ValueError(
                "SphericalSpectralConv expects "
                f"(..., {plan.sample_shape[0]}, {plan.sample_shape[1]}, "
                f"{self.in_channels}) input."
            )
        coefficients = plan.analysis(array)
        transformed = oe.contract(
            "...lmi,lio->...lmo",
            coefficients,
            self.weight,
        )
        return plan.synthesis(transformed)


class _SFNOBlock(StrictModule):
    spectral: SphericalSpectralConv
    pointwise: Linear

    def __init__(
        self,
        plan: SphericalHarmonicPlan,
        /,
        *,
        channels: int,
        key: Key[Array, ""],
    ):
        spectral_key, pointwise_key = jr.split(key)
        self.spectral = SphericalSpectralConv(
            plan,
            in_channels=channels,
            out_channels=channels,
            key=spectral_key,
        )
        self.pointwise = Linear(
            in_size=channels,
            out_size=channels,
            activation=None,
            key=pointwise_key,
        )

    def __call__(
        self,
        values: Array,
        plan: SphericalHarmonicPlan,
        /,
        *,
        key: EvalKey,
    ) -> Array:
        hidden = self.spectral(values, plan) + self.pointwise(values, key=key)
        return (values + jax.nn.gelu(hidden)) / jnp.sqrt(2.0)


def _validate_axis(
    axis: OperatorAxis,
    expected_nodes: Array,
    expected_weights: Array,
    /,
    *,
    name: str,
    periodic: bool,
    token: Array,
) -> Array:
    if axis.size != int(expected_nodes.size):
        raise ValueError(f"SFNO {name} axis has the wrong sample count.")
    if axis.periodic != periodic:
        raise ValueError(f"SFNO {name} axis periodicity does not match its sampling.")
    token = eqx.error_if(
        token,
        ~jnp.allclose(axis.nodes, expected_nodes, rtol=1e-12, atol=1e-12),
        f"SFNO {name} nodes do not match the S2FFT sampling theorem.",
    )
    if axis.quadrature_weights is not None:
        token = eqx.error_if(
            token,
            ~jnp.allclose(
                axis.quadrature_weights,
                expected_weights,
                rtol=1e-10,
                atol=1e-12,
            ),
            f"SFNO {name} quadrature does not match the S2FFT plan.",
        )
    return token


class SFNO(AbstractOperatorModel):
    """Spin-zero real spherical Fourier neural operator on an exact S2FFT grid."""

    operator_architecture = "SFNO"

    plan: SphericalHarmonicPlan
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    width: int
    depth: int
    source_key: str | None
    lift: Linear
    blocks: tuple[_SFNOBlock, ...]
    projection: Linear

    def __init__(
        self,
        plan: SphericalHarmonicPlan,
        /,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(plan, SphericalHarmonicPlan):
            raise TypeError("plan must be a SphericalHarmonicPlan.")
        if plan.spin != 0 or not plan.reality:
            raise ValueError("SFNO currently requires a real spin-zero transform plan.")
        self.plan = plan
        self.in_size = in_channels
        self.out_size = out_channels
        self.width = int(width)
        self.depth = int(depth)
        self.source_key = source_key
        if min(self.width, self.depth) <= 0:
            raise ValueError("width and depth must be positive.")
        keys = jr.split(key, self.depth + 2)
        self.lift = Linear(
            in_size=_get_size(in_channels),
            out_size=self.width,
            activation=None,
            key=keys[0],
        )
        self.blocks = tuple(
            _SFNOBlock(
                plan,
                channels=self.width,
                key=block_key,
            )
            for block_key in keys[1:-1]
        )
        self.projection = Linear(
            in_size=self.width,
            out_size=_get_size(out_channels),
            activation=None,
            key=keys[-1],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("SFNO requires source_key for multiple inputs.")
        return next(iter(batch.inputs.values()))

    def _validate_grid(
        self,
        source: FunctionSamples,
        query: FunctionSamples,
        token: Array,
        /,
    ) -> Array:
        if not source.axes or not query.axes:
            raise ValueError("SFNO requires tensor-product colatitude/longitude axes.")
        if len(source.axes) != 2 or len(query.axes) != 2:
            raise ValueError("SFNO requires exactly two spherical axes.")
        if source.sample_shape != self.plan.sample_shape:
            raise ValueError("SFNO source shape does not match its transform plan.")
        if query.sample_shape != self.plan.sample_shape:
            raise ValueError("SFNO query shape does not match its transform plan.")
        if source.axis_names != query.axis_names:
            raise ValueError("SFNO source and query axis names must match.")
        for axes in (source.axes, query.axes):
            token = _validate_axis(
                axes[0],
                self.plan.theta,
                self.plan.theta_quadrature_weights,
                name="colatitude",
                periodic=False,
                token=token,
            )
            token = _validate_axis(
                axes[1],
                self.plan.phi,
                self.plan.phi_quadrature_weights,
                name="longitude",
                periodic=True,
                token=token,
            )
        return token

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
            raise ValueError("SFNO source values cannot be None.")
        values = jnp.asarray(source.values)
        values = self._validate_grid(source, query, values)
        values = eqx.error_if(
            values,
            ~jnp.all(source.mask_array(case_shape=batch.case_shape)),
            "SFNO does not support masked spherical samples.",
        )
        values = eqx.error_if(
            values,
            ~jnp.all(query.mask_array(case_shape=batch.case_shape)),
            "SFNO does not support masked spherical queries.",
        )
        scalar_shape = batch.case_shape + self.plan.sample_shape
        if tuple(int(size) for size in values.shape) == scalar_shape:
            if _get_size(self.in_size) != 1:
                raise ValueError("Multichannel SFNO input requires a channel axis.")
            values = values[..., None]
        else:
            expected = scalar_shape + (_get_size(self.in_size),)
            if tuple(int(size) for size in values.shape) != expected:
                raise ValueError(f"SFNO source values must have shape {expected}.")
        hidden = self.lift(values, key=fold_in_eval_key(key, 0))
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                self.plan,
                key=fold_in_eval_key(key, index + 1),
            )
        output = self.projection(
            hidden,
            key=fold_in_eval_key(key, self.depth + 1),
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
            raise TypeError("SFNO requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["SFNO", "SphericalHarmonicPlan", "SphericalSpectralConv"]
