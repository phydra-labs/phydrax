#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ....graph._query_batch import query_neighbors
from ..._utils import _get_size
from ..core._keys import EvalKey, split_eval_key
from ..core._operator import FunctionSamples, OperatorBatch
from ..core._operator_geometry import (
    RegionalPointLatentGeometry,
    TensorGridLatentGeometry,
)
from ..layers._graph_transfer import (
    GraphAttentionTransfer,
    GraphKernelTransfer,
    MultiscaleGraphTransfer,
)
from ..layers._linear import Linear


GeometryTransfer = GraphKernelTransfer | GraphAttentionTransfer | MultiscaleGraphTransfer
LatentGeometry = TensorGridLatentGeometry | RegionalPointLatentGeometry
TensorGridExecution = Literal["structured", "operator_batch"]
LatentSupportKind = Literal["occupancy", "sdf"]


def _sample_coordinates(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    if not samples.axes and samples.coordinates is None:
        raise ValueError(
            "Geometry operators require explicit source and query coordinates."
        )
    return samples.coordinates_array(case_shape=case_shape, flatten=True)


def _sample_mask(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    return samples.mask_array(case_shape=case_shape).reshape(
        case_shape + (prod(samples.sample_shape),)
    )


def _sample_measure(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
    *,
    assume_uniform_measure: bool,
    name: str = "source",
) -> Array:
    explicit = samples.quadrature_weights is not None or (
        bool(samples.axes)
        and all(axis.quadrature_weights is not None for axis in samples.axes)
    )
    if not explicit and not assume_uniform_measure:
        raise ValueError(
            f"Geometry operator integral transfers require explicit {name} quadrature; "
            "set assume_uniform_measure=True to opt into unit point weights."
        )
    return samples.weights(case_shape=case_shape).reshape(
        case_shape + (prod(samples.sample_shape),)
    )


def _sample_values(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    channels: int,
    /,
    *,
    name: str,
) -> Array:
    if samples.values is None:
        raise ValueError(f"{name} values cannot be None.")
    if isinstance(samples.values, Mapping):
        raise TypeError(f"{name} requires one array-valued FunctionSamples field.")
    values = jnp.asarray(samples.values)
    sample_shape = samples.sample_shape
    case_ndim = len(case_shape)
    sample_ndim = len(sample_shape)
    if tuple(int(size) for size in values.shape[:case_ndim]) != case_shape:
        raise ValueError(f"{name} case shape does not match OperatorBatch.case_shape.")
    if (
        tuple(int(size) for size in values.shape[case_ndim : case_ndim + sample_ndim])
        != sample_shape
    ):
        raise ValueError(f"{name} does not contain its sample shape after case axes.")
    trailing = tuple(int(size) for size in values.shape[case_ndim + sample_ndim :])
    if not trailing and int(channels) == 1:
        values = values[..., None]
    elif trailing != (int(channels),):
        raise ValueError(
            f"{name} expected {channels} channels; got trailing shape {trailing}."
        )
    return values.reshape(case_shape + (prod(sample_shape), int(channels)))


class GeometryOperatorDiagnostics(eqx.Module):
    """Latent geometry, processor routes, and conservation data for one evaluation."""

    processor: Any | None
    latent_coordinates: Array
    latent_measure: Array
    latent_mask: Array
    latent_support: Array | None
    source_mass: Array | None
    target_mass_before_projection: Array | None
    target_mass_after_projection: Array | None
    conservation_correction: Array | None

    def __init__(
        self,
        *,
        processor: Any | None,
        latent_coordinates: Array,
        latent_measure: Array,
        latent_mask: Array,
        latent_support: Array | None,
        source_mass: Array | None,
        target_mass_before_projection: Array | None,
        target_mass_after_projection: Array | None,
        conservation_correction: Array | None,
    ):
        self.processor = processor
        self.latent_coordinates = jnp.asarray(latent_coordinates)
        self.latent_measure = jnp.asarray(latent_measure)
        self.latent_mask = jnp.asarray(latent_mask, dtype=bool)
        self.latent_support = (
            None if latent_support is None else jnp.asarray(latent_support)
        )
        self.source_mass = None if source_mass is None else jnp.asarray(source_mass)
        self.target_mass_before_projection = (
            None
            if target_mass_before_projection is None
            else jnp.asarray(target_mass_before_projection)
        )
        self.target_mass_after_projection = (
            None
            if target_mass_after_projection is None
            else jnp.asarray(target_mass_after_projection)
        )
        self.conservation_correction = (
            None
            if conservation_correction is None
            else jnp.asarray(conservation_correction)
        )


class TensorGridProcessor(eqx.Module):
    """Adapt a structured channels-last processor to latent point arrays."""

    model: Any
    geometry: TensorGridLatentGeometry
    channels: int = eqx.field(static=True)
    execution: TensorGridExecution = eqx.field(static=True)
    source_key: str = eqx.field(static=True)
    conditioning_channels: tuple[tuple[str, int], ...] = eqx.field(static=True)
    supports_diagnostics: bool = eqx.field(static=True)

    def __init__(
        self,
        model: Any,
        geometry: TensorGridLatentGeometry,
        channels: int,
        *,
        execution: TensorGridExecution = "structured",
        source_key: str = "latent",
        conditioning_channels: Sequence[tuple[str, int]] = (),
        supports_diagnostics: bool = False,
    ):
        self.model = model
        self.geometry = geometry
        self.channels = int(channels)
        self.execution = execution
        self.source_key = str(source_key)
        self.conditioning_channels = tuple(
            (str(name), int(width)) for name, width in conditioning_channels
        )
        self.supports_diagnostics = bool(supports_diagnostics)
        if (
            _get_size(model.in_size) != self.channels
            or _get_size(model.out_size) != self.channels
        ):
            raise ValueError(
                "TensorGridProcessor model must preserve latent channel width."
            )
        if self.execution not in ("structured", "operator_batch"):
            raise ValueError("execution must be 'structured' or 'operator_batch'.")
        if len({name for name, _ in self.conditioning_channels}) != len(
            self.conditioning_channels
        ):
            raise ValueError("Tensor-grid processor condition names must be unique.")
        if any(width <= 0 for _, width in self.conditioning_channels):
            raise ValueError("Tensor-grid processor condition widths must be positive.")
        if self.execution == "structured" and self.conditioning_channels:
            raise ValueError(
                "Conditioning requires TensorGridProcessor execution='operator_batch'."
            )
        if self.execution == "structured" and self.supports_diagnostics:
            raise ValueError(
                "Structured TensorGridProcessor execution cannot return diagnostics."
            )

    def evaluate(
        self,
        values: Array,
        coordinates: Array,
        measure: Array,
        mask: Array,
        /,
        *,
        condition_values: Sequence[Array] = (),
        case_axes: Sequence[str] = (),
        key: EvalKey = None,
        return_diagnostics: bool = False,
    ) -> tuple[Array, Any | None]:
        del measure
        if int(values.shape[-2]) != self.geometry.point_count:
            raise ValueError(
                "Latent value count does not match TensorGridLatentGeometry."
            )
        if int(coordinates.shape[-2]) != self.geometry.point_count:
            raise ValueError(
                "Latent coordinate count does not match TensorGridLatentGeometry."
            )
        case_shape = tuple(int(size) for size in values.shape[:-2])
        axes = self.geometry.axes()
        grid = values.reshape(case_shape + self.geometry.shape + (self.channels,))
        grid_mask = jnp.asarray(mask, dtype=bool).reshape(
            case_shape + self.geometry.shape
        )
        if len(condition_values) != len(self.conditioning_channels):
            raise ValueError(
                "Tensor-grid processor condition values must match its declared "
                "conditioning channels."
            )

        if self.execution == "structured":
            if return_diagnostics:
                raise ValueError(
                    "Structured TensorGridProcessor execution has no diagnostics."
                )
            output = jnp.asarray(
                self.model((grid, *(axis.nodes for axis in axes)), key=key)
            )
            diagnostics = None
        else:
            inputs = {
                self.source_key: FunctionSamples(
                    values=grid,
                    axes=axes,
                    mask=grid_mask,
                )
            }
            for (name, width), condition in zip(
                self.conditioning_channels,
                condition_values,
                strict=True,
            ):
                condition_array = jnp.asarray(condition)
                expected = case_shape + (width,)
                if condition_array.shape != expected:
                    raise ValueError(
                        f"Condition {name!r} must have shape {expected}; "
                        f"got {condition_array.shape}."
                    )
                inputs[name] = FunctionSamples(values=condition_array)
            latent_batch = OperatorBatch(inputs=inputs, queries={"query": FunctionSamples(
                values=None,
                axes=axes,
                mask=grid_mask,
            )}, case_axes=tuple(case_axes),
            case_shape=case_shape if case_shape else None,)
            if return_diagnostics:
                if not self.supports_diagnostics:
                    raise ValueError(
                        "This tensor-grid processor does not expose diagnostics."
                    )
                output, diagnostics = self.model.evaluate_with_diagnostics(
                    latent_batch,
                    key=key,
                )
            else:
                output = self.model(latent_batch, key=key)
                diagnostics = None
            output = jnp.asarray(output)

        if output.ndim == len(case_shape) + len(self.geometry.shape):
            output = output[..., None]
        output = output.reshape(
            case_shape + (self.geometry.point_count, self.channels)
        )
        output = output * jnp.asarray(mask, dtype=bool)[..., None].astype(output.dtype)
        return output, diagnostics

    def __call__(
        self,
        values: Array,
        coordinates: Array,
        measure: Array,
        mask: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        output, _ = self.evaluate(
            values,
            coordinates,
            measure,
            mask,
            key=key,
        )
        return output


class _GeometryOperatorCore(eqx.Module):
    """Shared source-to-latent, process, latent-to-query execution core."""

    encoders: tuple[GeometryTransfer, ...]
    latent_mixer: Linear | None
    processor: Any
    decoder: GeometryTransfer
    latent_geometry: LatentGeometry
    source_keys: tuple[str, ...] = eqx.field(static=True)
    source_channels: tuple[int, ...] = eqx.field(static=True)
    latent_channels: int = eqx.field(static=True)
    query_channels: int = eqx.field(static=True)
    assume_uniform_measure: bool = eqx.field(static=True)
    conditioning_channels: tuple[tuple[str, int], ...] = eqx.field(static=True)
    latent_support_key: str | None = eqx.field(static=True)
    latent_support_kind: LatentSupportKind = eqx.field(static=True)
    latent_support_threshold: float = eqx.field(static=True)
    latent_support_neighbors: int = eqx.field(static=True)
    latent_support_radius: float | None = eqx.field(static=True)
    conserve_mass: bool = eqx.field(static=True)
    conservation_source_key: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        encoders: Sequence[GeometryTransfer],
        processor: Any,
        decoder: GeometryTransfer,
        latent_geometry: LatentGeometry,
        source_channels: Sequence[int],
        latent_channels: int,
        source_keys: Sequence[str] = (),
        query_channels: int = 0,
        latent_mixer: Linear | None = None,
        assume_uniform_measure: bool = False,
        conditioning_channels: Sequence[tuple[str, int]] = (),
        latent_support_key: str | None = None,
        latent_support_kind: LatentSupportKind = "occupancy",
        latent_support_threshold: float = 0.5,
        latent_support_neighbors: int = 4,
        latent_support_radius: float | None = None,
        conserve_mass: bool = False,
        conservation_source_key: str | None = None,
    ):
        encoders_ = tuple(encoders)
        channels_ = tuple(int(value) for value in source_channels)
        keys_ = tuple(str(value) for value in source_keys)
        conditions_ = tuple(
            (str(name), int(width)) for name, width in conditioning_channels
        )
        if not encoders_ or len(encoders_) != len(channels_):
            raise ValueError(
                "encoders and source_channels must have one non-empty entry per source."
            )
        if keys_ and len(keys_) != len(encoders_):
            raise ValueError("source_keys must be empty or match the encoder count.")
        if len(set(keys_)) != len(keys_):
            raise ValueError("source_keys must be unique.")
        if any(
            encoder.in_channels != channel
            for encoder, channel in zip(encoders_, channels_, strict=True)
        ):
            raise ValueError("Every encoder input width must match source_channels.")
        if any(encoder.out_channels != int(latent_channels) for encoder in encoders_):
            raise ValueError("Every encoder must produce latent_channels.")
        if decoder.in_channels != int(latent_channels):
            raise ValueError("Decoder input width must match latent_channels.")
        if len(encoders_) > 1:
            if latent_mixer is None:
                raise ValueError("Multiple source encoders require latent_mixer.")
            if _get_size(latent_mixer.in_size) != len(encoders_) * int(
                latent_channels
            ) or _get_size(latent_mixer.out_size) != int(latent_channels):
                raise ValueError(
                    "latent_mixer must map concatenated encoder channels to latent_channels."
                )
        elif latent_mixer is not None:
            raise ValueError("latent_mixer is only used with multiple source encoders.")
        if len({name for name, _ in conditions_}) != len(conditions_):
            raise ValueError("Conditioning channel names must be unique.")
        if any(width <= 0 for _, width in conditions_):
            raise ValueError("Conditioning channel widths must be positive.")
        condition_names = {name for name, _ in conditions_}
        if condition_names.intersection(keys_):
            raise ValueError("Conditioning inputs cannot also be encoded sources.")
        if latent_support_kind not in ("occupancy", "sdf"):
            raise ValueError("latent_support_kind must be 'occupancy' or 'sdf'.")
        if int(latent_support_neighbors) <= 0:
            raise ValueError("latent_support_neighbors must be positive.")
        if latent_support_radius is not None and float(latent_support_radius) <= 0.0:
            raise ValueError("latent_support_radius must be positive when supplied.")
        support_key = (
            None if latent_support_key is None else str(latent_support_key)
        )
        if support_key is not None and support_key in condition_names:
            raise ValueError("A latent support input cannot be a case condition.")
        if (conditions_ or support_key is not None) and (
            not isinstance(processor, TensorGridProcessor)
            or processor.execution != "operator_batch"
        ):
            raise ValueError(
                "Conditioning and hard latent support require an operator-batch "
                "TensorGridProcessor."
            )
        if support_key is not None and not isinstance(
            latent_geometry, TensorGridLatentGeometry
        ):
            raise ValueError("Hard latent support requires tensor-grid geometry.")
        if conditions_ and (
            not isinstance(processor, TensorGridProcessor)
            or processor.conditioning_channels != conditions_
        ):
            raise ValueError(
                "Processor conditioning channels must match the geometry core."
            )
        conservation_key = (
            None if conservation_source_key is None else str(conservation_source_key)
        )
        if conserve_mass:
            if conservation_key is None and len(channels_) != 1:
                raise ValueError(
                    "Multiple encoded sources require conservation_source_key."
                )
            if conservation_key is not None and keys_ and conservation_key not in keys_:
                raise ValueError(
                    "conservation_source_key must name an encoded source."
                )
            conservation_index = (
                0
                if conservation_key is None
                else keys_.index(conservation_key)
                if keys_
                else 0
            )
            if decoder.out_channels != channels_[conservation_index]:
                raise ValueError(
                    "Conservation requires matching source and output channel counts."
                )
        self.encoders = encoders_
        self.latent_mixer = latent_mixer
        self.processor = processor
        self.decoder = decoder
        self.latent_geometry = latent_geometry
        self.source_keys = keys_
        self.source_channels = channels_
        self.latent_channels = int(latent_channels)
        self.query_channels = int(query_channels)
        self.assume_uniform_measure = bool(assume_uniform_measure)
        self.conditioning_channels = conditions_
        self.latent_support_key = support_key
        self.latent_support_kind = latent_support_kind
        self.latent_support_threshold = float(latent_support_threshold)
        self.latent_support_neighbors = int(latent_support_neighbors)
        self.latent_support_radius = (
            None
            if latent_support_radius is None
            else float(latent_support_radius)
        )
        self.conserve_mass = bool(conserve_mass)
        self.conservation_source_key = conservation_key

    def _source_items(
        self,
        batch: OperatorBatch,
        /,
    ) -> tuple[tuple[str, FunctionSamples], ...]:
        if self.source_keys:
            return tuple((key, batch.input(key)) for key in self.source_keys)
        excluded = {name for name, _ in self.conditioning_channels}
        if self.latent_support_key is not None:
            excluded.add(self.latent_support_key)
        candidates = tuple(
            (name, samples)
            for name, samples in batch.inputs.items()
            if name not in excluded
        )
        if len(self.encoders) != 1 or len(candidates) != 1:
            raise ValueError(
                "Explicit source_keys are required unless the model and batch contain "
                "one non-auxiliary source."
            )
        return candidates

    def _condition_values(
        self,
        batch: OperatorBatch,
        /,
    ) -> tuple[Array, ...]:
        values = []
        case_ndim = len(batch.case_shape)
        for name, channels in self.conditioning_channels:
            samples = batch.input(name)
            if samples.values is None:
                raise ValueError(f"Condition {name!r} has no values.")
            array = samples.values
            if tuple(int(size) for size in array.shape[:case_ndim]) != batch.case_shape:
                raise ValueError(
                    f"Condition {name!r} must begin with case shape "
                    f"{batch.case_shape}; got {array.shape}."
                )
            trailing = tuple(int(size) for size in array.shape[case_ndim:])
            feature_count = prod(trailing) if trailing else 1
            if feature_count != channels:
                raise ValueError(
                    f"Condition {name!r} must contain {channels} features per case; "
                    f"got trailing shape {trailing}."
                )
            if samples.mask is not None:
                array = eqx.error_if(
                    array,
                    jnp.logical_not(
                        jnp.all(samples.mask_array(case_shape=batch.case_shape))
                    ),
                    f"Condition {name!r} cannot contain masked features.",
                )
            values.append(array.reshape(batch.case_shape + (channels,)))
        return tuple(values)

    def _latent_support(
        self,
        batch: OperatorBatch,
        latent_coordinates: Array,
        /,
    ) -> tuple[Array | None, Array]:
        if self.latent_support_key is None:
            return None, jnp.ones(latent_coordinates.shape[:-1], dtype=bool)
        support = batch.input(self.latent_support_key)
        case_shape = batch.case_shape
        source_coordinates = _sample_coordinates(support, case_shape)
        source_mask = _sample_mask(support, case_shape)
        source_values = _sample_values(
            support,
            case_shape,
            1,
            name=f"latent support {self.latent_support_key!r}",
        )[..., 0]
        source_values = eqx.error_if(
            source_values,
            jnp.any(jnp.logical_not(jnp.isfinite(source_values))),
            "Latent occupancy/SDF values must be finite.",
        )
        if self.latent_support_kind == "occupancy":
            source_values = eqx.error_if(
                source_values,
                jnp.any((source_values < 0.0) | (source_values > 1.0)),
                "Latent occupancy values must lie in [0, 1].",
            )

        source_count = int(source_coordinates.shape[-2])
        neighbors = min(self.latent_support_neighbors, source_count)
        neighborhood = query_neighbors(
            source_coordinates,
            latent_coordinates,
            source_mask=source_mask,
            max_neighbors=neighbors,
            radius=self.latent_support_radius,
            target_chunk_size=256,
        )
        case_count = prod(case_shape) if case_shape else 1
        flat_values = source_values.reshape((case_count, source_count))
        gathered = jax.vmap(lambda values, indices: values[indices])(
            flat_values,
            neighborhood.indices,
        )
        valid = neighborhood.mask
        tolerance = jnp.sqrt(jnp.finfo(source_values.dtype).eps)
        exact = valid & (neighborhood.distance <= tolerance)
        inverse_distance = jnp.where(
            valid,
            jnp.reciprocal(jnp.maximum(neighborhood.distance, tolerance)),
            0.0,
        )
        weights = jnp.where(
            jnp.any(exact, axis=-1, keepdims=True),
            exact.astype(source_values.dtype),
            inverse_distance,
        )
        denominator = jnp.sum(weights, axis=-1)
        interpolated = jnp.sum(gathered * weights, axis=-1) / jnp.maximum(
            denominator,
            jnp.asarray(1.0, dtype=source_values.dtype),
        )
        interpolated = interpolated.reshape(latent_coordinates.shape[:-1])
        interpolation_support = (denominator > 0.0).reshape(
            latent_coordinates.shape[:-1]
        )
        if self.latent_support_kind == "occupancy":
            mask = interpolated >= self.latent_support_threshold
        else:
            mask = interpolated < self.latent_support_threshold
        return interpolated, interpolation_support & mask

    def _project_conservation(
        self,
        *,
        source_values: Array,
        source_measure: Array,
        output: Array,
        query_measure: Array,
        query_mask: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        source_mass = jnp.sum(
            source_values * source_measure[..., None],
            axis=-2,
        )
        target_mass_before = jnp.sum(
            output * query_measure[..., None],
            axis=-2,
        )
        target_measure = jnp.sum(query_measure, axis=-1)
        output = eqx.error_if(
            output,
            jnp.any(target_measure <= 0.0),
            "Every conservative geometry-operator query case requires positive measure.",
        )
        correction = (source_mass - target_mass_before) / target_measure[..., None]
        corrected = output + correction[..., None, :]
        corrected = corrected * query_mask[..., None].astype(corrected.dtype)
        target_mass_after = jnp.sum(
            corrected * query_measure[..., None],
            axis=-2,
        )
        return (
            corrected,
            source_mass,
            target_mass_before,
            target_mass_after,
            correction,
        )

    def _evaluate(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey,
        return_diagnostics: bool,
    ) -> tuple[Array, GeometryOperatorDiagnostics | None]:
        source_items = self._source_items(batch)
        source_names = tuple(name for name, _ in source_items)
        sources = tuple(samples for _, samples in source_items)
        case_shape = batch.case_shape
        source_coordinates = tuple(
            _sample_coordinates(source, case_shape) for source in sources
        )
        source_masks = tuple(_sample_mask(source, case_shape) for source in sources)
        source_measures = tuple(
            _sample_measure(
                source,
                case_shape,
                assume_uniform_measure=self.assume_uniform_measure,
            )
            for source in sources
        )
        source_values = tuple(
            _sample_values(
                source,
                case_shape,
                channels,
                name=f"source {name!r}",
            )
            for (name, source), channels in zip(
                source_items,
                self.source_channels,
                strict=True,
            )
        )
        reference_coordinates = source_coordinates[0]
        reference_mask = source_masks[0]
        reference_measure = source_measures[0]
        if isinstance(self.latent_geometry, TensorGridLatentGeometry):
            latent_coordinates = self.latent_geometry.coordinates(
                case_shape,
                source_coordinates=reference_coordinates,
                source_mask=reference_mask,
            )
            latent_measure = self.latent_geometry.quadrature(
                case_shape,
                source_coordinates=reference_coordinates,
                source_mask=reference_mask,
            )
        elif isinstance(self.latent_geometry, RegionalPointLatentGeometry):
            latent_coordinates = self.latent_geometry.coordinates(
                reference_coordinates,
                reference_mask,
            )
            latent_measure = self.latent_geometry.quadrature(reference_measure)
        else:
            raise TypeError("Unsupported latent geometry type.")
        latent_support, latent_mask = self._latent_support(
            batch,
            latent_coordinates,
        )
        query_coordinates = _sample_coordinates(batch.require_single_query(), case_shape)
        query_mask = _sample_mask(batch.require_single_query(), case_shape)
        query_features = (
            None
            if self.query_channels == 0
            else _sample_values(
                batch.require_single_query(),
                case_shape,
                self.query_channels,
                name="query covariates",
            )
        )
        condition_values = self._condition_values(batch)

        keys = split_eval_key(key, len(self.encoders) + 3)
        encoded = tuple(
            encoder(
                values,
                coordinates,
                latent_coordinates,
                source_measure=measure,
                source_mask=mask,
                target_mask=latent_mask,
                key=keys[index],
            )
            for index, (encoder, values, coordinates, measure, mask) in enumerate(
                zip(
                    self.encoders,
                    source_values,
                    source_coordinates,
                    source_measures,
                    source_masks,
                    strict=True,
                )
            )
        )
        if self.latent_mixer is None:
            latent = encoded[0]
        else:
            latent = self.latent_mixer(jnp.concatenate(encoded, axis=-1))
        if isinstance(self.processor, TensorGridProcessor):
            processed, processor_diagnostics = self.processor.evaluate(
                latent,
                latent_coordinates,
                latent_measure,
                latent_mask,
                condition_values=condition_values,
                case_axes=batch.case_axes,
                key=keys[-2],
                return_diagnostics=return_diagnostics,
            )
        else:
            processed = self.processor(
                latent,
                latent_coordinates,
                latent_measure,
                latent_mask,
                key=keys[-2],
            )
            processor_diagnostics = None
        output = self.decoder(
            processed,
            latent_coordinates,
            query_coordinates,
            source_measure=latent_measure,
            source_mask=latent_mask,
            target_mask=query_mask,
            target_features=query_features,
            key=keys[-1],
        )
        output = output * query_mask[..., None].astype(output.dtype)

        source_mass = None
        target_mass_before = None
        target_mass_after = None
        correction = None
        if self.conserve_mass:
            conservation_index = (
                0
                if self.conservation_source_key is None
                else source_names.index(self.conservation_source_key)
            )
            query_measure = _sample_measure(
                batch.require_single_query(),
                case_shape,
                assume_uniform_measure=self.assume_uniform_measure,
                name="query",
            )
            (
                output,
                source_mass,
                target_mass_before,
                target_mass_after,
                correction,
            ) = self._project_conservation(
                source_values=source_values[conservation_index],
                source_measure=source_measures[conservation_index],
                output=output,
                query_measure=query_measure,
                query_mask=query_mask,
            )

        diagnostics = (
            GeometryOperatorDiagnostics(
                processor=processor_diagnostics,
                latent_coordinates=latent_coordinates,
                latent_measure=latent_measure,
                latent_mask=latent_mask,
                latent_support=latent_support,
                source_mass=source_mass,
                target_mass_before_projection=target_mass_before,
                target_mass_after_projection=target_mass_after,
                conservation_correction=correction,
            )
            if return_diagnostics
            else None
        )
        return output, diagnostics

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        output, _ = self._evaluate(
            batch,
            key=key,
            return_diagnostics=False,
        )
        return output

    def evaluate_with_diagnostics(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> tuple[Array, GeometryOperatorDiagnostics]:
        output, diagnostics = self._evaluate(
            batch,
            key=key,
            return_diagnostics=True,
        )
        if diagnostics is None:
            raise RuntimeError("Geometry-operator diagnostics were not produced.")
        return output, diagnostics


__all__ = [
    "GeometryOperatorDiagnostics",
    "GeometryTransfer",
    "LatentGeometry",
    "LatentSupportKind",
    "TensorGridExecution",
    "TensorGridProcessor",
]
