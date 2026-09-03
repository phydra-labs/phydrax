# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ....._doc import DOC_KEY0
from ....._strict import StrictModule
from ...._keys import EvalKey
from ...data import FunctionSamples, OperatorBatch
from ...engine import AbstractOperatorModel
from ...representations import TensorFieldLayout


class OrthogonalPointTopology(StrictModule):
    """Prepared fixed-capacity bipartite radius topology with a strict margin."""

    query_indices: Array
    source_indices: Array
    active: Array
    cutoff_margin: float = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        query_indices: ArrayLike,
        source_indices: ArrayLike,
        active: ArrayLike,
        /,
        *,
        cutoff_margin: float,
        support_id: str,
    ):
        query = jnp.asarray(query_indices, dtype=jnp.int32)
        source = jnp.asarray(source_indices, dtype=jnp.int32)
        mask = jnp.asarray(active, dtype=bool)
        if query.ndim != 1 or source.shape != query.shape or mask.shape != query.shape:
            raise ValueError("Prepared point topology arrays must share one edge shape.")
        margin = float(cutoff_margin)
        if not jnp.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                "Prepared radius topology requires a positive cutoff margin."
            )
        if not support_id:
            raise ValueError("support_id must be nonempty.")
        self.query_indices = query
        self.source_indices = source
        self.active = mask
        self.cutoff_margin = margin
        self.support_id = str(support_id)


class OrthogonalEquivariantPointCNO(AbstractOperatorModel):
    """Exact O(d)-covariant radial point convolution for declared tensor blocks."""

    operator_architecture = "OrthogonalEquivariantPointCNO"

    layout: TensorFieldLayout
    raw_amplitudes: Array
    raw_length_scales: Array
    source_key: str | None = eqx.field(static=True)
    topology: OrthogonalPointTopology | None
    dimension: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        layout: TensorFieldLayout,
        /,
        *,
        source_key: str | None = None,
        topology: OrthogonalPointTopology | None = None,
        initial_length_scale: float = 1.0,
    ):
        if not isinstance(layout, TensorFieldLayout):
            raise TypeError("layout must be TensorFieldLayout.")
        if layout.dimension not in (2, 3) or any(
            block.tensor_type.rank > 2 for block in layout.blocks
        ):
            raise ValueError(
                "Point CNO supports dimensions two/three and tensor rank at most two."
            )
        scale = float(initial_length_scale)
        if not jnp.isfinite(scale) or scale <= 0.0:
            raise ValueError("initial_length_scale must be finite and positive.")
        if topology is not None and not isinstance(topology, OrthogonalPointTopology):
            raise TypeError("topology must be OrthogonalPointTopology or None.")
        count = len(layout.blocks)
        self.layout = layout
        self.raw_amplitudes = jnp.zeros((count,), dtype=float)
        self.raw_length_scales = jnp.full((count,), jnp.log(jnp.expm1(scale)))
        self.source_key = source_key
        self.topology = topology
        self.dimension = layout.dimension
        self.in_size = layout.channel_count
        self.out_size = layout.channel_count

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("Point CNO requires source_key for multiple inputs.")
        return next(iter(batch.inputs.values()))

    def _dense(
        self,
        source_points: Array,
        query_points: Array,
        values: Array,
        weights: Array,
        source_mask: Array,
        query_mask: Array,
    ) -> Array:
        displacement = query_points[..., :, None, :] - source_points[..., None, :, :]
        squared_distance = contract("...qpd,...qpd->...qp", displacement, displacement)
        outputs = []
        for index, block_values in enumerate(self.layout.unpack(values)):
            flattened = block_values.reshape(
                block_values.shape[: -len(self.layout.blocks[index].value_shape)] + (-1,)
            )
            amplitude = self.raw_amplitudes[index]
            length = jax.nn.softplus(self.raw_length_scales[index])
            kernel = amplitude * jnp.exp(-0.5 * squared_distance / (length * length))
            kernel = kernel * weights[..., None, :] * source_mask[..., None, :]
            convolved = contract("...qp,...pc->...qc", kernel, flattened)
            outputs.append(
                convolved.reshape(
                    query_points.shape[:-1] + self.layout.blocks[index].value_shape
                )
            )
        packed = self.layout.pack(outputs)
        return jnp.where(query_mask[..., None], packed, jnp.zeros_like(packed))

    def _sparse(
        self,
        source_points: Array,
        query_points: Array,
        values: Array,
        weights: Array,
        source_mask: Array,
        query_mask: Array,
    ) -> Array:
        assert self.topology is not None
        topology = self.topology
        query_index = topology.query_indices
        source_index = topology.source_indices
        displacement = jnp.take(query_points, query_index, axis=-2) - jnp.take(
            source_points, source_index, axis=-2
        )
        squared_distance = contract("...ed,...ed->...e", displacement, displacement)
        edge_active = (
            topology.active
            & jnp.take(source_mask, source_index, axis=-1)
            & jnp.take(query_mask, query_index, axis=-1)
        )
        outputs = []
        query_count = int(query_points.shape[-2])
        for index, block_values in enumerate(self.layout.unpack(values)):
            block = self.layout.blocks[index]
            flattened = block_values.reshape(
                block_values.shape[: -len(block.value_shape)] + (-1,)
            )
            gathered = jnp.take(flattened, source_index, axis=-2)
            amplitude = self.raw_amplitudes[index]
            length = jax.nn.softplus(self.raw_length_scales[index])
            kernel = amplitude * jnp.exp(-0.5 * squared_distance / (length * length))
            kernel = kernel * jnp.take(weights, source_index, axis=-1) * edge_active
            contributions = kernel[..., None] * gathered
            output = jnp.zeros(
                flattened.shape[:-2] + (query_count, flattened.shape[-1]),
                dtype=flattened.dtype,
            )
            output = output.at[..., query_index, :].add(contributions)
            outputs.append(output.reshape(query_points.shape[:-1] + block.value_shape))
        packed = self.layout.pack(outputs)
        return jnp.where(query_mask[..., None], packed, jnp.zeros_like(packed))

    def __call_operator_batch__(
        self, batch: OperatorBatch, /, *, key: EvalKey = DOC_KEY0
    ) -> Array:
        del key
        source = self._source(batch)
        query = batch.require_single_query()
        if source.values is None:
            raise ValueError("Point CNO source requires values.")
        case_shape = batch.case_shape
        source_points = source.coordinates_array(case_shape=case_shape).reshape(
            case_shape + (-1, self.dimension)
        )
        query_points = query.coordinates_array(case_shape=case_shape).reshape(
            case_shape + (-1, self.dimension)
        )
        values = jnp.asarray(source.values).reshape(
            case_shape + (-1, self.layout.channel_count)
        )
        weights = source.weights(case_shape=case_shape, normalized=False).reshape(
            case_shape + (-1,)
        )
        source_mask = source.mask_array(case_shape=case_shape).reshape(case_shape + (-1,))
        query_mask = query.mask_array(case_shape=case_shape).reshape(case_shape + (-1,))
        if self.topology is None:
            output = self._dense(
                source_points, query_points, values, weights, source_mask, query_mask
            )
        else:
            if bool(
                jnp.any(self.topology.query_indices >= query_points.shape[-2])
            ) or bool(jnp.any(self.topology.source_indices >= source_points.shape[-2])):
                raise ValueError(
                    "Prepared point topology indices exceed the current supports."
                )
            output = self._sparse(
                source_points, query_points, values, weights, source_mask, query_mask
            )
        return output.reshape(
            case_shape + query.sample_shape + (self.layout.channel_count,)
        )

    def __call__(self, x: OperatorBatch, /, *, key: EvalKey = DOC_KEY0) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("OrthogonalEquivariantPointCNO requires OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["OrthogonalEquivariantPointCNO", "OrthogonalPointTopology"]
