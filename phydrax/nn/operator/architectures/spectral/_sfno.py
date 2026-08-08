#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jax.scipy.special import sph_harm_y
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.data import OperatorAxis, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


def _trapezoid(nodes: Array, /, *, periodic: bool = False) -> Array:
    if int(nodes.shape[0]) == 1:
        return jnp.ones_like(nodes)
    if periodic:
        return jnp.full_like(nodes, jnp.mean(jnp.diff(nodes)))
    return jnp.concatenate(
        (
            (0.5 * (nodes[1] - nodes[0]))[None],
            0.5 * (nodes[2:] - nodes[:-2]),
            (0.5 * (nodes[-1] - nodes[-2]))[None],
        )
    )


def _sphere_basis(
    theta: Array,
    phi: Array,
    degree: Array,
    order: Array,
    max_degree: int,
    /,
) -> Array:
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")
    theta_flat = theta_grid.reshape((-1,))
    phi_flat = phi_grid.reshape((-1,))
    return jax.vmap(
        lambda colatitude, longitude: sph_harm_y(
            degree,
            order,
            colatitude,
            longitude,
            n_max=max_degree - 1,
        )
    )(theta_flat, phi_flat)


class SphericalTransformPlan(StrictModule, NonTrainableState):
    """Reusable spherical-harmonic basis and quadrature for one grid."""

    basis: Array
    weights: Array
    sample_shape: tuple[int, int]
    max_degree: int

    def __init__(
        self,
        basis: Array,
        weights: Array,
        sample_shape: tuple[int, int],
        max_degree: int,
        /,
    ):
        self.basis = jnp.asarray(basis)
        self.weights = jnp.asarray(weights)
        self.sample_shape = (int(sample_shape[0]), int(sample_shape[1]))
        self.max_degree = int(max_degree)


class SphericalSpectralConv(StrictModule):
    """Rotation-equivariant spherical-harmonic convolution on the two-sphere."""

    in_channels: int
    out_channels: int
    max_degree: int
    degree: Array
    order: Array
    weight: Array
    theta_weights_include_sine: bool

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        max_degree: int,
        theta_weights_include_sine: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.max_degree = int(max_degree)
        self.theta_weights_include_sine = bool(theta_weights_include_sine)
        if self.in_channels <= 0 or self.out_channels <= 0 or self.max_degree <= 0:
            raise ValueError("channels and max_degree must be positive.")
        degrees = []
        orders = []
        for degree in range(self.max_degree):
            for order in range(-degree, degree + 1):
                degrees.append(degree)
                orders.append(order)
        self.degree = jnp.asarray(degrees, dtype=jnp.int32)
        self.order = jnp.asarray(orders, dtype=jnp.int32)
        # Sharing one real channel map across all orders m at fixed degree l is the
        # Schur form required for scalar SO(3)-equivariance and preserves real fields.
        self.weight = jr.normal(
            key,
            shape=(self.max_degree, self.in_channels, self.out_channels),
        ) / jnp.sqrt(float(self.in_channels))

    def _weights(
        self,
        theta_axis: OperatorAxis,
        phi_axis: OperatorAxis,
        sample_weights: Array | None,
        /,
    ) -> Array:
        if sample_weights is not None:
            weights = jnp.asarray(sample_weights, dtype=float)
            if weights.shape[-2:] == (theta_axis.size, phi_axis.size):
                return weights.reshape(weights.shape[:-2] + (-1,))
            if weights.shape[-1:] == (theta_axis.size * phi_axis.size,):
                return weights
            raise ValueError("Spherical sample weights do not match the spherical grid.")
        theta_weight = (
            _trapezoid(theta_axis.nodes)
            if theta_axis.quadrature_weights is None
            else theta_axis.quadrature_weights
        )
        if not self.theta_weights_include_sine:
            theta_weight = theta_weight * jnp.sin(theta_axis.nodes)
        phi_weight = (
            _trapezoid(phi_axis.nodes, periodic=phi_axis.periodic)
            if phi_axis.quadrature_weights is None
            else phi_axis.quadrature_weights
        )
        return jnp.multiply.outer(theta_weight, phi_weight).reshape((-1,))

    def plan(
        self,
        axes: tuple[OperatorAxis, OperatorAxis],
        /,
        *,
        sample_weights: Array | None = None,
    ) -> SphericalTransformPlan:
        """Precompute the basis and quadrature for repeated evaluations."""
        theta_axis, phi_axis = axes
        basis = _sphere_basis(
            theta_axis.nodes,
            phi_axis.nodes,
            self.degree,
            self.order,
            self.max_degree,
        )
        weights = self._weights(theta_axis, phi_axis, sample_weights)
        return SphericalTransformPlan(
            basis,
            weights,
            (theta_axis.size, phi_axis.size),
            self.max_degree,
        )

    def __call__(
        self,
        values: Array,
        axes: tuple[OperatorAxis, OperatorAxis],
        /,
        *,
        sample_weights: Array | None = None,
        plan: SphericalTransformPlan | None = None,
    ) -> Array:
        theta_axis, phi_axis = axes
        array = jnp.asarray(values)
        sample_shape = (theta_axis.size, phi_axis.size)
        if array.ndim < 3 or tuple(array.shape[-3:-1]) != sample_shape:
            raise ValueError(
                "SphericalSpectralConv expects (..., n_theta, n_phi, channels)."
            )
        if int(array.shape[-1]) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels.")
        transform = (
            self.plan(axes, sample_weights=sample_weights) if plan is None else plan
        )
        if (
            transform.sample_shape != sample_shape
            or transform.max_degree != self.max_degree
        ):
            raise ValueError(
                "Spherical transform plan does not match this layer or grid."
            )
        basis = transform.basis
        weights = transform.weights
        flattened = array.reshape(array.shape[:-3] + (-1, self.in_channels))
        coefficients = oe.contract(
            "nm,...ni,...n->...mi",
            jnp.conj(basis),
            flattened,
            weights,
        )
        mode_weight = self.weight[self.degree]
        output_coefficients = oe.contract("...mi,mio->...mo", coefficients, mode_weight)
        output = oe.contract("nm,...mo->...no", basis, output_coefficients).real
        return output.reshape(array.shape[:-3] + sample_shape + (self.out_channels,))


class _SFNOBlock(StrictModule):
    spectral: SphericalSpectralConv
    pointwise: Linear

    def __init__(
        self,
        *,
        channels: int,
        max_degree: int,
        theta_weights_include_sine: bool,
        key: Key[Array, ""],
    ):
        spectral_key, pointwise_key = jr.split(key)
        self.spectral = SphericalSpectralConv(
            in_channels=channels,
            out_channels=channels,
            max_degree=max_degree,
            theta_weights_include_sine=theta_weights_include_sine,
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
        axes: tuple[OperatorAxis, OperatorAxis],
        /,
        *,
        sample_weights: Array | None,
        plan: SphericalTransformPlan | None = None,
    ) -> Array:
        hidden = self.spectral(
            values,
            axes,
            sample_weights=sample_weights,
            plan=plan,
        ) + self.pointwise(values)
        return (values + jax.nn.gelu(hidden)) / jnp.sqrt(2.0)


class SFNO(AbstractOperatorModel):
    """Spherical Fourier Neural Operator using true spherical harmonics."""

    operator_architecture = "SFNO"

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    width: int
    max_degree: int
    source_key: str | None
    lift: Linear
    blocks: tuple[_SFNOBlock, ...]
    projection: Linear

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        max_degree: int = 12,
        theta_weights_include_sine: bool = False,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.width = int(width)
        self.max_degree = int(max_degree)
        self.source_key = source_key
        if self.width <= 0 or int(depth) <= 0 or self.max_degree <= 0:
            raise ValueError("width, depth, and max_degree must be positive.")
        keys = jr.split(key, int(depth) + 2)
        self.lift = Linear(
            in_size=_get_size(in_channels),
            out_size=self.width,
            activation=None,
            key=keys[0],
        )
        self.blocks = tuple(
            _SFNOBlock(
                channels=self.width,
                max_degree=self.max_degree,
                theta_weights_include_sine=theta_weights_include_sine,
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

    def _evaluate(
        self,
        values: Array,
        axes: tuple[OperatorAxis, OperatorAxis],
        /,
        *,
        sample_weights: Array | None,
    ) -> Array:
        array = jnp.asarray(values)
        sample_shape = (axes[0].size, axes[1].size)
        if array.ndim >= 2 and tuple(array.shape[-2:]) == sample_shape:
            if _get_size(self.in_size) != 1:
                raise ValueError("Multichannel SFNO input requires a channel axis.")
            array = array[..., None]
        elif array.ndim <= 2 or tuple(array.shape[-3:-1]) != sample_shape:
            raise ValueError("SFNO values do not match the spherical grid shape.")
        if int(array.shape[-1]) != _get_size(self.in_size):
            raise ValueError(f"Expected {_get_size(self.in_size)} input channels.")
        hidden = self.lift(array)
        plan = self.blocks[0].spectral.plan(
            axes,
            sample_weights=sample_weights,
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                axes,
                sample_weights=sample_weights,
                plan=plan,
            )
        output = self.projection(hidden)
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        if self.source_key is not None:
            source = batch.input(self.source_key)
        elif len(batch.inputs) == 1:
            source = next(iter(batch.inputs.values()))
        else:
            raise ValueError("source_key is required for multiple operator inputs.")
        axes = source.axes or batch.require_single_query().axes
        if len(axes) != 2 or source.values is None:
            raise ValueError("SFNO requires colatitude and longitude grid axes.")
        if (
            source.axes
            and source.sample_shape != batch.require_single_query().sample_shape
        ):
            raise ValueError("SFNO requires coincident source and query grids.")
        weights = source.weights(case_shape=batch.case_shape)
        return self._evaluate(
            jnp.asarray(source.values),
            (axes[0], axes[1]),
            sample_weights=weights,
        )

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        if isinstance(x, OperatorBatch):
            return self.__call_operator_batch__(x)
        if not isinstance(x, tuple) or len(x) != 3:
            raise ValueError("SFNO requires (values, colatitude_axis, longitude_axis).")
        axes = (
            OperatorAxis("theta", jnp.asarray(x[1])),
            OperatorAxis("phi", jnp.asarray(x[2]), periodic=True),
        )
        return self._evaluate(jnp.asarray(x[0]), axes, sample_weights=None)


__all__ = [
    "SFNO",
    "SphericalSpectralConv",
    "SphericalTransformPlan",
]
