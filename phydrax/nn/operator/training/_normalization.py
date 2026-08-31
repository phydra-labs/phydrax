#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..data import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ..field import OperatorFieldSpec


def _encode_numeric_array(value: Array, /) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "real": np.real(array).tolist(),
        "imag": np.imag(array).tolist() if np.iscomplexobj(array) else None,
    }


def _decode_numeric_array(value: Mapping[str, Any], /) -> Array:
    real = jnp.asarray(value["real"])
    imag = value["imag"]
    return real if imag is None else real + 1j * jnp.asarray(imag)


@dataclass(frozen=True)
class AffineNormalizer:
    """Persistable per-field affine normalization statistics."""

    mean: Array
    scale: Array
    channel_axis: int | None
    epsilon: float

    def __post_init__(self):
        mean = jnp.asarray(self.mean)
        scale = jnp.asarray(self.scale)
        if mean.shape != scale.shape:
            raise ValueError("Normalizer mean and scale shapes must match.")
        if bool(jnp.any(~jnp.isfinite(mean))) or bool(jnp.any(~jnp.isfinite(scale))):
            raise ValueError("Normalizer statistics must be finite.")
        if bool(jnp.any(scale <= 0.0)):
            raise ValueError("Normalizer scales must be positive.")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "epsilon", float(self.epsilon))

    def _shape(self, array: Array) -> tuple[int, ...]:
        if self.channel_axis is None:
            return ()
        axis = int(self.channel_axis) % array.ndim
        if self.mean.ndim != 1 or int(self.mean.shape[0]) != int(array.shape[axis]):
            raise ValueError("Normalizer channel statistics do not match the array.")
        shape = [1] * array.ndim
        shape[axis] = int(self.mean.shape[0])
        return tuple(shape)

    def normalize(self, value: Any, /) -> Array:
        array = jnp.asarray(value)
        shape = self._shape(array)
        return (array - self.mean.reshape(shape)) / self.scale.reshape(shape)

    def denormalize(self, value: Any, /) -> Array:
        array = jnp.asarray(value)
        shape = self._shape(array)
        return array * self.scale.reshape(shape) + self.mean.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": _encode_numeric_array(self.mean),
            "scale": _encode_numeric_array(self.scale),
            "channel_axis": self.channel_axis,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "AffineNormalizer":
        return cls(
            mean=_decode_numeric_array(value["mean"]),
            scale=_decode_numeric_array(value["scale"]),
            channel_axis=value["channel_axis"],
            epsilon=float(value["epsilon"]),
        )


def _fit_arrays(
    arrays: Sequence[Array],
    masks: Sequence[Array],
    /,
    *,
    channel_axis: int | None,
    epsilon: float,
    weights: Sequence[Array] | None = None,
    center: bool = True,
) -> AffineNormalizer:
    if weights is not None and len(weights) != len(arrays):
        raise ValueError("Normalization weights and arrays must have the same length.")

    selected: list[np.ndarray] = []
    selected_weights: list[np.ndarray] = []
    for index, (value, mask) in enumerate(zip(arrays, masks, strict=True)):
        array = np.asarray(value)
        valid = np.asarray(mask, dtype=bool)
        weight: np.ndarray | None = None
        if weights is not None:
            weight = np.asarray(weights[index])
            if weight.shape != valid.shape:
                raise ValueError("Normalization weights must match case/sample geometry.")
            if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
                raise ValueError("Normalization weights must be finite and nonnegative.")
            selected_weights.append(weight[valid].reshape((-1,)))

        if channel_axis is None:
            if array.shape != valid.shape:
                raise ValueError(
                    "Scalar normalization requires values to match case/sample geometry."
                )
            selected.append(array[valid].reshape((-1, 1)))
            continue
        axis = int(channel_axis) % array.ndim
        moved = np.moveaxis(array, axis, -1)
        if moved.shape[:-1] != valid.shape:
            raise ValueError(
                "Channel normalization requires non-channel dimensions to match geometry."
            )
        selected.append(moved[valid].reshape((-1, moved.shape[-1])))
    if not selected or sum(item.shape[0] for item in selected) == 0:
        raise ValueError("Cannot fit normalization without valid samples.")
    stacked = np.concatenate(selected, axis=0)
    if center:
        if weights is None:
            mean = np.mean(stacked, axis=0)
            variance = np.mean(np.square(np.abs(stacked - mean)), axis=0)
        else:
            stacked_weights = np.concatenate(selected_weights, axis=0)
            total = np.sum(stacked_weights)
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("Cannot fit normalization from zero-measure samples.")
            mean = np.sum(stacked * stacked_weights[:, None], axis=0) / total
            variance = (
                np.sum(
                    np.square(np.abs(stacked - mean)) * stacked_weights[:, None],
                    axis=0,
                )
                / total
            )
    else:
        mean = np.zeros((stacked.shape[1],), dtype=stacked.dtype)
        if weights is None:
            variance = np.mean(np.square(np.abs(stacked)), axis=0)
        else:
            stacked_weights = np.concatenate(selected_weights, axis=0)
            total = np.sum(stacked_weights)
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("Cannot fit normalization from zero-measure samples.")
            variance = (
                np.sum(
                    np.square(np.abs(stacked)) * stacked_weights[:, None],
                    axis=0,
                )
                / total
            )
    scale = np.maximum(np.sqrt(variance), float(epsilon))
    if channel_axis is None:
        mean = mean.reshape(())
        scale = scale.reshape(())
    return AffineNormalizer(
        mean=jnp.asarray(mean),
        scale=jnp.asarray(scale),
        channel_axis=channel_axis,
        epsilon=epsilon,
    )


def _infer_channel_axis(array: Array, geometry_rank: int) -> int | None:
    if array.ndim == geometry_rank:
        return None
    if array.ndim == geometry_rank + 1:
        return -1
    raise ValueError(
        "Values must have case/sample dimensions and at most one channel dimension."
    )


def _coordinate_samples(samples: FunctionSamples, case_shape: tuple[int, ...]) -> Array:
    return samples.coordinates_array(case_shape=case_shape, flatten=True)


def _quadrature_fit_weights(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
) -> Array:
    quadrature = np.asarray(samples.quadrature(case_shape=case_shape))
    if np.any(~np.isfinite(quadrature)) or np.any(quadrature < 0.0):
        raise ValueError("Quadrature weights must be finite and nonnegative.")
    mask = np.asarray(samples.mask_array(case_shape=case_shape), dtype=bool)
    weights = np.where(mask, quadrature, 0.0)
    sample_count = int(np.prod(samples.sample_shape, dtype=int))
    per_case = weights.reshape(case_shape + (sample_count,))
    measure = np.sum(per_case, axis=-1, keepdims=True)
    if np.any(~np.isfinite(measure)) or np.any(measure <= 0.0):
        raise ValueError(
            "Every case must have positive quadrature measure after masking."
        )
    return jnp.asarray(per_case / measure).reshape(weights.shape)


@dataclass(frozen=True)
class OperatorNormalizationPolicy:
    """Per-field, per-query training statistics for operator data."""

    input_values: Mapping[str, AffineNormalizer]
    targets: Mapping[str, AffineNormalizer]
    input_coordinates: Mapping[str, AffineNormalizer]
    query_coordinates: Mapping[str, AffineNormalizer]

    def __post_init__(self):
        object.__setattr__(self, "input_values", dict(self.input_values))
        object.__setattr__(self, "targets", dict(self.targets))
        object.__setattr__(self, "input_coordinates", dict(self.input_coordinates))
        object.__setattr__(self, "query_coordinates", dict(self.query_coordinates))

    def normalize_targets(
        self,
        targets: OperatorTargetBatch,
        /,
        *,
        target_aliases: Mapping[str, str] | None = None,
    ) -> OperatorTargetBatch:
        aliases = {} if target_aliases is None else dict(target_aliases)
        missing = tuple(
            name
            for name, field in targets.fields.items()
            if field.spec.classification is None
            and aliases.get(name, name) not in self.targets
        )
        if missing:
            raise KeyError(f"Missing target normalizers for fields {missing}.")
        return OperatorTargetBatch(
            {
                name: OperatorFieldBatch(
                    (
                        field.values
                        if field.spec.classification is not None
                        else self.targets[aliases.get(name, name)].normalize(field.values)
                    ),
                    query_name=field.query_name,
                    spec=field.spec,
                )
                for name, field in targets.fields.items()
            },
            case_axes=targets.case_axes,
            case_shape=targets.case_shape,
        )

    def denormalize_targets(
        self,
        targets: OperatorTargetBatch,
        /,
        *,
        target_aliases: Mapping[str, str] | None = None,
    ) -> OperatorTargetBatch:
        aliases = {} if target_aliases is None else dict(target_aliases)
        missing = tuple(
            name
            for name, field in targets.fields.items()
            if field.spec.classification is None
            and aliases.get(name, name) not in self.targets
        )
        if missing:
            raise KeyError(f"Missing target normalizers for fields {missing}.")
        return OperatorTargetBatch(
            {
                name: OperatorFieldBatch(
                    (
                        field.values
                        if field.spec.classification is not None
                        else self.targets[aliases.get(name, name)].denormalize(
                            field.values
                        )
                    ),
                    query_name=field.query_name,
                    spec=field.spec,
                )
                for name, field in targets.fields.items()
            },
            case_axes=targets.case_axes,
            case_shape=targets.case_shape,
        )

    def normalize_prediction(
        self,
        prediction: OperatorPrediction,
        /,
    ) -> OperatorPrediction:
        missing = tuple(
            name
            for name, field in prediction.fields.items()
            if field.spec.classification is None and name not in self.targets
        )
        if missing:
            raise KeyError(f"Missing target normalizers for fields {missing}.")
        return OperatorPrediction(
            {
                name: OperatorFieldBatch(
                    (
                        field.values
                        if field.spec.classification is not None
                        else self.targets[name].normalize(field.values)
                    ),
                    query_name=field.query_name,
                    spec=field.spec,
                )
                for name, field in prediction.fields.items()
            },
            prediction.queries,
            case_axes=prediction.case_axes,
            case_shape=prediction.case_shape,
        )

    def denormalize_prediction(
        self,
        prediction: OperatorPrediction,
        /,
    ) -> OperatorPrediction:
        missing = tuple(
            name
            for name, field in prediction.fields.items()
            if field.spec.classification is None and name not in self.targets
        )
        if missing:
            raise KeyError(f"Missing target normalizers for fields {missing}.")
        return OperatorPrediction(
            {
                name: OperatorFieldBatch(
                    (
                        field.values
                        if field.spec.classification is not None
                        else self.targets[name].denormalize(field.values)
                    ),
                    query_name=field.query_name,
                    spec=field.spec,
                )
                for name, field in prediction.fields.items()
            },
            prediction.queries,
            case_axes=prediction.case_axes,
            case_shape=prediction.case_shape,
        )

    def _samples(
        self,
        samples: FunctionSamples,
        value_normalizer: AffineNormalizer | None,
        coordinate_normalizer: AffineNormalizer | None,
        case_shape: tuple[int, ...],
        /,
        *,
        inverse: bool,
    ) -> FunctionSamples:
        mask = samples.mask_array(case_shape=case_shape)
        values = samples.values
        if values is not None and value_normalizer is not None:
            operation = (
                value_normalizer.denormalize if inverse else value_normalizer.normalize
            )
            values = operation(values)
            trailing = (1,) * (values.ndim - mask.ndim)
            values = jnp.where(mask.reshape(mask.shape + trailing), values, 0.0)

        axes = samples.axes
        coordinates = samples.coordinates
        weights = samples.quadrature_weights
        if coordinate_normalizer is not None:
            operation = (
                coordinate_normalizer.denormalize
                if inverse
                else coordinate_normalizer.normalize
            )
            jacobian = jnp.prod(coordinate_normalizer.scale)
            if axes:
                transformed_axes = []
                for index, axis in enumerate(axes):
                    mean = coordinate_normalizer.mean[index]
                    scale = coordinate_normalizer.scale[index]
                    nodes = (
                        axis.nodes * scale + mean
                        if inverse
                        else (axis.nodes - mean) / scale
                    )
                    quadrature = axis.quadrature_weights
                    if quadrature is not None:
                        quadrature = quadrature * scale if inverse else quadrature / scale
                    transformed_axes.append(
                        OperatorAxis(
                            axis.name,
                            nodes,
                            quadrature_weights=quadrature,
                            basis=axis.basis,
                            periodic=axis.periodic,
                        )
                    )
                axes = tuple(transformed_axes)
            elif coordinates is not None:
                coordinates = operation(coordinates)
                if weights is not None:
                    weights = weights * jacobian if inverse else weights / jacobian

        return FunctionSamples(
            values=values,
            axes=axes,
            coordinates=coordinates,
            quadrature_weights=weights,
            mask=samples.mask,
            topology=samples.topology,
            support_id=(samples.support_id if coordinate_normalizer is None else None),
            measure_id=(samples.measure_id if coordinate_normalizer is None else None),
        )

    def normalize_batch(self, batch: OperatorBatch, /) -> OperatorBatch:
        return OperatorBatch(
            inputs={
                name: self._samples(
                    samples,
                    self.input_values.get(name),
                    self.input_coordinates.get(name),
                    batch.case_shape,
                    inverse=False,
                )
                for name, samples in batch.inputs.items()
            },
            queries={
                name: self._samples(
                    samples,
                    None,
                    self.query_coordinates.get(name),
                    batch.case_shape,
                    inverse=False,
                )
                for name, samples in batch.queries.items()
            },
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    def denormalize_batch(self, batch: OperatorBatch, /) -> OperatorBatch:
        return OperatorBatch(
            inputs={
                name: self._samples(
                    samples,
                    self.input_values.get(name),
                    self.input_coordinates.get(name),
                    batch.case_shape,
                    inverse=True,
                )
                for name, samples in batch.inputs.items()
            },
            queries={
                name: self._samples(
                    samples,
                    None,
                    self.query_coordinates.get(name),
                    batch.case_shape,
                    inverse=True,
                )
                for name, samples in batch.queries.items()
            },
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_values": {
                name: normalizer.to_dict()
                for name, normalizer in self.input_values.items()
            },
            "targets": {
                name: normalizer.to_dict() for name, normalizer in self.targets.items()
            },
            "input_coordinates": {
                name: normalizer.to_dict()
                for name, normalizer in self.input_coordinates.items()
            },
            "query_coordinates": {
                name: normalizer.to_dict()
                for name, normalizer in self.query_coordinates.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        /,
    ) -> "OperatorNormalizationPolicy":
        expected = {
            "input_values",
            "targets",
            "input_coordinates",
            "query_coordinates",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "Operator normalization must use the current canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        return cls(
            input_values={
                name: AffineNormalizer.from_dict(normalizer)
                for name, normalizer in value["input_values"].items()
            },
            targets={
                name: AffineNormalizer.from_dict(normalizer)
                for name, normalizer in value["targets"].items()
            },
            input_coordinates={
                name: AffineNormalizer.from_dict(normalizer)
                for name, normalizer in value["input_coordinates"].items()
            },
            query_coordinates={
                name: AffineNormalizer.from_dict(normalizer)
                for name, normalizer in value["query_coordinates"].items()
            },
        )


def fit_operator_normalization(
    batches: OperatorBatch | Sequence[OperatorBatch],
    targets: OperatorTargetBatch | Sequence[OperatorTargetBatch],
    /,
    *,
    normalize_coordinates: bool = False,
    input_channel_axes: Mapping[str, int | None] | None = None,
    target_channel_axes: Mapping[str, int | None] | None = None,
    weighting: Literal["uniform", "quadrature"] = "uniform",
    epsilon: float = 1e-6,
    fields: Sequence[OperatorFieldSpec] = (),
    target_aliases: Mapping[str, str] | None = None,
) -> OperatorNormalizationPolicy:
    """Fit named field and query statistics from training cases only."""
    batch_tuple = (batches,) if isinstance(batches, OperatorBatch) else tuple(batches)
    if not batch_tuple:
        raise ValueError("At least one training batch is required.")
    if weighting not in ("uniform", "quadrature"):
        raise ValueError("weighting must be 'uniform' or 'quadrature'.")
    target_tuple = (
        (targets,) if isinstance(targets, OperatorTargetBatch) else tuple(targets)
    )
    if len(target_tuple) != len(batch_tuple):
        raise ValueError("targets and batches must have the same length.")
    for batch, target in zip(batch_tuple, target_tuple, strict=True):
        target.validate(batch)

    input_names = tuple(batch_tuple[0].inputs)
    query_names = tuple(batch_tuple[0].queries)
    target_names = tuple(target_tuple[0].fields)
    if any(set(batch.inputs) != set(input_names) for batch in batch_tuple[1:]):
        raise ValueError("All normalization batches must have identical input names.")
    if any(set(batch.queries) != set(query_names) for batch in batch_tuple[1:]):
        raise ValueError("All normalization batches must have identical query names.")
    if any(set(target.fields) != set(target_names) for target in target_tuple[1:]):
        raise ValueError("All normalization targets must have identical field names.")
    aliases = {} if target_aliases is None else dict(target_aliases)
    unknown_aliases = set(aliases) - set(target_names)
    if unknown_aliases:
        raise ValueError(
            "Target aliases must exist in every normalization target batch; "
            f"unknown={tuple(sorted(unknown_aliases))!r}."
        )
    field_specs = tuple(fields)
    if any(not isinstance(field, OperatorFieldSpec) for field in field_specs):
        raise TypeError("fields must contain OperatorFieldSpec objects.")
    source_fields: dict[str, OperatorFieldSpec] = {}
    target_fields: dict[str, OperatorFieldSpec] = {}
    for field in field_specs:
        if field.is_source:
            assert field.source_name is not None
            if field.source_name in source_fields:
                raise ValueError(
                    "Operator source names must map to unique field semantics."
                )
            source_fields[field.source_name] = field
        if field.is_target:
            target_fields[field.name] = field

    configured_input_axes = {} if input_channel_axes is None else dict(input_channel_axes)
    input_values: dict[str, AffineNormalizer] = {}
    input_coordinates: dict[str, AffineNormalizer] = {}
    for name in input_names:
        arrays = []
        masks = []
        weights = []
        channel_axes: list[int | None] = []
        for batch in batch_tuple:
            samples = batch.input(name)
            if samples.values is None:
                raise ValueError(f"Input {name!r} has no values to normalize.")
            array = samples.values
            mask = samples.mask_array(case_shape=batch.case_shape)
            channel_axes.append(
                configured_input_axes.get(
                    name,
                    _infer_channel_axis(array, mask.ndim),
                )
            )
            arrays.append(array)
            masks.append(mask)
            if weighting == "quadrature":
                weights.append(_quadrature_fit_weights(samples, batch.case_shape))
        inferred_axis = channel_axes[0]
        if any(axis != inferred_axis for axis in channel_axes[1:]):
            raise ValueError(f"Input {name!r} channel layouts are inconsistent.")
        source_field = source_fields.get(name)
        source_cochain = None if source_field is None else source_field.cochain
        input_values[name] = _fit_arrays(
            arrays,
            masks,
            channel_axis=inferred_axis,
            epsilon=epsilon,
            weights=None if weighting == "uniform" else weights,
            center=not (
                source_cochain is not None and source_cochain.cell_orientation == "signed"
            ),
        )
        if normalize_coordinates:
            input_coordinates[name] = _fit_arrays(
                [
                    _coordinate_samples(batch.input(name), batch.case_shape)
                    for batch in batch_tuple
                ],
                [
                    batch.input(name)
                    .mask_array(case_shape=batch.case_shape)
                    .reshape(batch.case_shape + (-1,))
                    for batch in batch_tuple
                ],
                channel_axis=-1,
                epsilon=epsilon,
                weights=(
                    None
                    if weighting == "uniform"
                    else [
                        weight.reshape(batch.case_shape + (-1,))
                        for weight, batch in zip(
                            weights,
                            batch_tuple,
                            strict=True,
                        )
                    ]
                ),
            )

    configured_target_axes = (
        {} if target_channel_axes is None else dict(target_channel_axes)
    )
    canonical_target_names = tuple(
        dict.fromkeys(aliases.get(name, name) for name in target_names)
    )
    target_normalizers: dict[str, AffineNormalizer] = {}
    for canonical_name in canonical_target_names:
        grouped_names = tuple(
            name for name in target_names if aliases.get(name, name) == canonical_name
        )
        target_batches = tuple(
            target.field(name) for target in target_tuple for name in grouped_names
        )
        first_target = target_batches[0]
        if first_target.spec.classification is not None:
            if any(
                field.spec.to_dict() != first_target.spec.to_dict()
                for field in target_batches[1:]
            ):
                raise ValueError(
                    f"Classification target field {canonical_name!r} changed its "
                    "output spec."
                )
            continue
        query_name = first_target.query_name
        if any(
            field.query_name != query_name
            or field.spec.to_dict() != first_target.spec.to_dict()
            for field in target_batches[1:]
        ):
            raise ValueError(
                f"Target aliases for {canonical_name!r} must share one query and spec."
            )
        masks = [
            batch.query(query_name).mask_array(case_shape=batch.case_shape)
            for batch in batch_tuple
            for _ in grouped_names
        ]
        weights = (
            None
            if weighting == "uniform"
            else [
                _quadrature_fit_weights(
                    batch.query(query_name),
                    batch.case_shape,
                )
                for batch in batch_tuple
                for _ in grouped_names
            ]
        )
        target_field = target_fields.get(canonical_name)
        target_cochain = None if target_field is None else target_field.cochain
        target_normalizers[canonical_name] = _fit_arrays(
            [field.values for field in target_batches],
            masks,
            channel_axis=configured_target_axes.get(
                canonical_name,
                _infer_channel_axis(first_target.values, masks[0].ndim),
            ),
            epsilon=epsilon,
            weights=weights,
            center=not (
                target_cochain is not None and target_cochain.cell_orientation == "signed"
            ),
        )

    query_coordinates: dict[str, AffineNormalizer] = {}
    if normalize_coordinates:
        for name in query_names:
            query_weights = (
                None
                if weighting == "uniform"
                else [
                    _quadrature_fit_weights(batch.query(name), batch.case_shape)
                    for batch in batch_tuple
                ]
            )
            query_coordinates[name] = _fit_arrays(
                [
                    _coordinate_samples(batch.query(name), batch.case_shape)
                    for batch in batch_tuple
                ],
                [
                    batch.query(name)
                    .mask_array(case_shape=batch.case_shape)
                    .reshape(batch.case_shape + (-1,))
                    for batch in batch_tuple
                ],
                weights=(
                    None
                    if query_weights is None
                    else [
                        weight.reshape(batch.case_shape + (-1,))
                        for weight, batch in zip(
                            query_weights,
                            batch_tuple,
                            strict=True,
                        )
                    ]
                ),
                channel_axis=-1,
                epsilon=epsilon,
            )
    return OperatorNormalizationPolicy(
        input_values=input_values,
        targets=target_normalizers,
        input_coordinates=input_coordinates,
        query_coordinates=query_coordinates,
    )


def save_operator_normalization(
    path: str | Path,
    policy: OperatorNormalizationPolicy,
    /,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_operator_normalization(path: str | Path, /) -> OperatorNormalizationPolicy:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return OperatorNormalizationPolicy.from_dict(value)


__all__ = [
    "AffineNormalizer",
    "OperatorNormalizationPolicy",
    "fit_operator_normalization",
    "load_operator_normalization",
    "save_operator_normalization",
]
