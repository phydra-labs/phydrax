#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
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
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.models._mlp import MLP
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


KoopmanEvolution = Literal["discrete", "continuous"]


class KoopmanTemporalOperator(AbstractOperatorModel):
    """Stable latent Koopman evolution for temporal fields on tensor grids.

    A quadrature-aware encoder maps one source field at elapsed time zero to a
    global vector of nonlinear observables. Those observables evolve under a
    constrained linear system and a coordinate-conditioned decoder reconstructs
    the field at every requested space-time grid point.

    The continuous generator is ``S - diag(d)`` with ``S`` skew-symmetric and
    ``d > 0``. Its symmetric part is strictly negative definite, so the learned
    semigroup is contractive by construction. The discrete operator is
    ``Q diag(exp(-d)) Q.T``, where ``Q`` is the orthogonal Cayley transform of a
    learned skew-symmetric matrix. Its spectral radius is therefore strictly
    below one. Neither parameterization relies on post-update clipping.

    This architecture intentionally accepts tensor-product axes only. It can
    evaluate a different regular spatial query grid, but it does not claim a
    point-cloud or arbitrary-geometry discretization invariance.
    """

    operator_architecture = "KoopmanTemporalOperator"

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    spatial_ndim: int
    latent_size: int
    evolution: KoopmanEvolution
    time_axis: str
    source_key: str | None
    min_decay: float
    encoder: MLP
    decoder: MLP
    raw_decay: Array
    skew_parameter: Array

    def __init__(
        self,
        *,
        spatial_ndim: int,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        latent_size: int = 32,
        hidden_size: int = 64,
        depth: int = 2,
        evolution: KoopmanEvolution = "continuous",
        time_axis: str = "time",
        source_key: str | None = None,
        initial_decay: float = 0.1,
        min_decay: float = 1e-4,
        skew_scale: float = 0.05,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.spatial_ndim = int(spatial_ndim)
        self.latent_size = int(latent_size)
        self.evolution = evolution
        self.time_axis = str(time_axis)
        self.source_key = source_key
        self.min_decay = float(min_decay)

        if self.spatial_ndim <= 0:
            raise ValueError("spatial_ndim must be positive.")
        if self.latent_size <= 0 or int(hidden_size) <= 0 or int(depth) < 0:
            raise ValueError(
                "latent_size and hidden_size must be positive; depth cannot be negative."
            )
        if evolution not in ("discrete", "continuous"):
            raise ValueError("evolution must be 'discrete' or 'continuous'.")
        if not self.time_axis:
            raise ValueError("time_axis must be non-empty.")
        if self.min_decay <= 0.0 or float(initial_decay) <= self.min_decay:
            raise ValueError("initial_decay must be greater than positive min_decay.")
        if float(skew_scale) < 0.0:
            raise ValueError("skew_scale must be non-negative.")

        encoder_key, decoder_key, skew_key = jr.split(key, 3)
        in_count = _get_size(self.in_size)
        out_count = _get_size(self.out_size)
        self.encoder = MLP(
            in_size=in_count + self.spatial_ndim,
            out_size=self.latent_size,
            width_size=int(hidden_size),
            depth=int(depth),
            key=encoder_key,
        )
        self.decoder = MLP(
            in_size=self.latent_size + self.spatial_ndim,
            out_size=out_count,
            width_size=int(hidden_size),
            depth=int(depth),
            key=decoder_key,
        )
        unconstrained_decay = float(initial_decay) - self.min_decay
        initial_raw_decay = jnp.log(jnp.expm1(unconstrained_decay))
        self.raw_decay = jnp.full((self.latent_size,), initial_raw_decay)
        self.skew_parameter = (
            float(skew_scale)
            * jr.normal(
                skew_key,
                (self.latent_size, self.latent_size),
            )
            / jnp.sqrt(float(self.latent_size))
        )

    def decay_rates(self, /) -> Array:
        """Return the strictly positive learned decay rates."""
        return jax.nn.softplus(self.raw_decay) + self.min_decay

    def generator_matrix(self, /) -> Array:
        """Return the stable continuous-time generator."""
        if self.evolution != "continuous":
            raise ValueError("generator_matrix is defined only for continuous evolution.")
        skew = self.skew_parameter - self.skew_parameter.T
        return skew - jnp.diag(self.decay_rates())

    def discrete_matrix(self, /) -> Array:
        """Return the stable learned one-step discrete Koopman matrix."""
        if self.evolution != "discrete":
            raise ValueError("discrete_matrix is defined only for discrete evolution.")
        skew = self.skew_parameter - self.skew_parameter.T
        identity = jnp.eye(self.latent_size, dtype=skew.dtype)
        orthogonal = jnp.linalg.solve(identity + skew, identity - skew)
        retention = jnp.exp(-self.decay_rates())
        return (orthogonal * retention[None, :]) @ orthogonal.T

    def evolution_matrix(self, time: Array | float = 1.0, /) -> Array:
        """Return stable latent transitions at non-negative elapsed times.

        For discrete evolution, real-valued times are interpreted in units of
        learned Koopman steps using the unique stable spectral interpolation of
        the positive-definite one-step matrix.
        """
        times = jnp.asarray(time, dtype=self.raw_decay.dtype)
        times = eqx.error_if(
            times,
            jnp.any(times < 0.0),
            "Koopman evolution requires non-negative elapsed times.",
        )
        flat_times = times.reshape((-1,))
        if self.evolution == "continuous":
            generator = self.generator_matrix()
            matrices = jax.vmap(jax.scipy.linalg.expm)(
                flat_times[:, None, None] * generator[None, :, :]
            )
        else:
            skew = self.skew_parameter - self.skew_parameter.T
            identity = jnp.eye(self.latent_size, dtype=skew.dtype)
            orthogonal = jnp.linalg.solve(identity + skew, identity - skew)
            retention = jnp.exp(-self.decay_rates())
            powered = retention[None, :] ** flat_times[:, None]
            matrices = oe.contract(
                "li,ti,mi->tlm",
                orthogonal,
                powered,
                orthogonal,
            )
        return matrices.reshape(times.shape + (self.latent_size, self.latent_size))

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("source_key is required for multiple operator inputs.")
        return next(iter(batch.inputs.values()))

    def _validate_grids(
        self,
        source: FunctionSamples,
        query: FunctionSamples,
        /,
    ) -> int:
        if not source.axes or source.coordinates is not None:
            raise ValueError(
                "KoopmanTemporalOperator source requires tensor-product axes."
            )
        if not query.axes or query.coordinates is not None:
            raise ValueError(
                "KoopmanTemporalOperator query requires tensor-product axes."
            )
        if len(source.axes) != self.spatial_ndim:
            raise ValueError(
                f"Expected {self.spatial_ndim} source spatial axes, got {len(source.axes)}."
            )
        if len(query.axes) != self.spatial_ndim + 1:
            raise ValueError(
                "Query must contain the source spatial axes and one explicit time axis."
            )
        source_names = tuple(axis.name for axis in source.axes)
        if self.time_axis in source_names:
            raise ValueError(
                "The source field must represent the state at elapsed time zero."
            )
        query_names = tuple(axis.name for axis in query.axes)
        if query_names.count(self.time_axis) != 1:
            raise ValueError(f"Query requires exactly one {self.time_axis!r} time axis.")
        time_index = query_names.index(self.time_axis)
        spatial_names = query_names[:time_index] + query_names[time_index + 1 :]
        if spatial_names != source_names:
            raise ValueError(
                "Query spatial axis names and relative order must match the source grid."
            )
        return time_index

    def _source_values(
        self,
        source: FunctionSamples,
        case_shape: tuple[int, ...],
        /,
    ) -> Array:
        if source.values is None:
            raise ValueError("KoopmanTemporalOperator source values cannot be None.")
        values = source.values
        sample_shape = source.sample_shape
        case_ndim = len(case_shape)
        sample_ndim = len(sample_shape)
        if tuple(int(size) for size in values.shape[:case_ndim]) != case_shape:
            raise ValueError("Source values do not match OperatorBatch.case_shape.")
        if (
            tuple(int(size) for size in values.shape[case_ndim : case_ndim + sample_ndim])
            != sample_shape
        ):
            raise ValueError("Source values do not align with the source tensor grid.")
        trailing = tuple(int(size) for size in values.shape[case_ndim + sample_ndim :])
        in_count = _get_size(self.in_size)
        if not trailing and in_count == 1:
            values = values[..., None]
        elif trailing != (in_count,):
            raise ValueError(
                f"Expected {in_count} source channels; got trailing shape {trailing}."
            )
        return values

    def _encode(
        self,
        source: FunctionSamples,
        case_shape: tuple[int, ...],
        /,
    ) -> Array:
        values = self._source_values(source, case_shape)
        coordinates = source.coordinates_array(case_shape=case_shape)
        mask = source.mask_array(case_shape=case_shape)
        safe_values = jnp.where(mask[..., None], values, 0.0)
        features = jnp.concatenate((safe_values, coordinates), axis=-1)
        flat_features = features.reshape(
            (-1, _get_size(self.in_size) + self.spatial_ndim)
        )
        observables = jax.vmap(lambda feature: self.encoder(feature, key=None))(
            flat_features
        ).reshape(case_shape + source.sample_shape + (self.latent_size,))
        weights = source.weights(normalized=True, case_shape=case_shape)
        weighted = jnp.where(
            mask[..., None],
            observables * weights[..., None],
            0.0,
        )
        sample_axes = tuple(
            range(len(case_shape), len(case_shape) + len(source.sample_shape))
        )
        return jnp.sum(weighted, axis=sample_axes)

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        source = self._source(batch)
        query = batch.require_single_query()
        time_index = self._validate_grids(source, query)
        case_shape = batch.case_shape
        latent = self._encode(source, case_shape)

        time_nodes = query.axes[time_index].nodes
        transitions = self.evolution_matrix(time_nodes)
        evolved = oe.contract("...l,tlm->...tm", latent, transitions)
        latent_grid_shape = (
            case_shape
            + tuple(
                int(axis.size) if index == time_index else 1
                for index, axis in enumerate(query.axes)
            )
            + (self.latent_size,)
        )
        evolved = evolved.reshape(latent_grid_shape)
        evolved = jnp.broadcast_to(
            evolved,
            case_shape + query.sample_shape + (self.latent_size,),
        )

        coordinates = query.coordinates_array(case_shape=case_shape)
        spatial_coordinates = jnp.concatenate(
            (coordinates[..., :time_index], coordinates[..., time_index + 1 :]),
            axis=-1,
        )
        decoder_features = jnp.concatenate((evolved, spatial_coordinates), axis=-1)
        flattened = decoder_features.reshape((-1, self.latent_size + self.spatial_ndim))
        decoded = jax.vmap(lambda feature: self.decoder(feature, key=None))(flattened)
        out_count = _get_size(self.out_size)
        output = decoded.reshape(case_shape + query.sample_shape + (out_count,))
        output = output * query.mask_array(case_shape=case_shape)[..., None]
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("KoopmanTemporalOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["KoopmanTemporalOperator"]
