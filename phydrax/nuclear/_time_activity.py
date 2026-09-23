#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Governed instantaneous activity series and finite-support integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..imaging._core import MedicalImageSupport
from ..measurement import (
    DataStage,
    DerivationRecord,
    IndexSampleSupport,
    MeasurementAsset,
    QualityFlag,
    QuantityField,
    RadiationQuantityKind,
    resolve_radiation_quantity,
    SampleTimeAxis,
    SamplingSemantics,
    TemporalSampling,
    TemporalSamplingKind,
    ValueKind,
)
from ..units import (
    BECQUEREL,
    BECQUEREL_PER_CUBIC_METER,
    BECQUEREL_SECOND,
    BECQUEREL_SECOND_PER_CUBIC_METER,
    conversion_factor,
    SECOND,
    TIME,
    UnitDefinition,
)
from ._activation import InventoryTransition


_ACTIVITY_KINDS = frozenset(
    {
        RadiationQuantityKind.ACTIVITY.value,
        RadiationQuantityKind.ACTIVITY_CONCENTRATION.value,
    }
)
_INTEGRATED_KIND = {
    RadiationQuantityKind.ACTIVITY.value: RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY,
    RadiationQuantityKind.ACTIVITY_CONCENTRATION.value: (
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION
    ),
}
_REFERENCE_UNIT = {
    RadiationQuantityKind.ACTIVITY.value: BECQUEREL,
    RadiationQuantityKind.ACTIVITY_CONCENTRATION.value: BECQUEREL_PER_CUBIC_METER,
}
_INTEGRATED_UNIT = {
    RadiationQuantityKind.ACTIVITY.value: BECQUEREL_SECOND,
    RadiationQuantityKind.ACTIVITY_CONCENTRATION.value: (
        BECQUEREL_SECOND_PER_CUBIC_METER
    ),
}


def _readonly_real(value: ArrayLike, name: str, /) -> np.ndarray:
    original = np.asarray(value)
    result = np.array(value, dtype=np.result_type(original.dtype, np.float64), copy=True)
    if not np.issubdtype(result.dtype, np.floating) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite and real-valued.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class _ScalarActivitySupport:
    """The remaining scalar support after integrating a time-only series."""

    source_support_id: str
    sample_shape: tuple[()] = field(default=(), init=False)
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = str(self.source_support_id)
        if not source or source != source.strip():
            raise ValueError(
                "source_support_id must be a non-empty canonical identifier."
            )
        object.__setattr__(self, "source_support_id", source)
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {"kind": "time-integrated-scalar-support", "source": source}
            ),
        )


def _time_axis_and_dimension(asset: MeasurementAsset, /) -> tuple[SampleTimeAxis, int]:
    support = asset.field.support
    if isinstance(support, IndexSampleSupport):
        if support.time_axis is None or support.time_axis_dimension is None:
            raise ValueError(
                "Time-activity index supports require an explicit time axis."
            )
        return support.time_axis, support.time_axis_dimension
    if isinstance(support, MedicalImageSupport):
        if support.time_axis is None:
            raise ValueError(
                "Time-activity image supports require an explicit time axis."
            )
        return support.time_axis, 3
    raise TypeError(
        "Time-activity series require IndexSampleSupport or MedicalImageSupport."
    )


def _reduced_support(asset: MeasurementAsset, time_dimension: int, /):
    support = asset.field.support
    if isinstance(support, MedicalImageSupport):
        return MedicalImageSupport(support.spatial_shape, support.spatial_affine)
    if not isinstance(support, IndexSampleSupport):
        raise TypeError("Unsupported time-activity support.")
    shape = (
        support.sample_shape[:time_dimension] + support.sample_shape[time_dimension + 1 :]
    )
    labels = (
        support.axis_labels[:time_dimension] + support.axis_labels[time_dimension + 1 :]
    )
    if not shape:
        return _ScalarActivitySupport(support.support_id)
    return IndexSampleSupport(shape, labels, frame_id=support.frame_id)


def _reduced_quantity_axes(
    asset: MeasurementAsset, time_dimension: int, /
) -> tuple[str, ...]:
    axes = asset.field.quantity.axes
    support = asset.field.support
    if isinstance(support, IndexSampleSupport):
        time_axis_label = support.axis_labels[time_dimension]
    elif len(axes) == asset.field.values.ndim:
        time_axis_label = axes[time_dimension]
    else:
        return axes
    return tuple(axis for axis in axes if axis != time_axis_label)


def _reference_ids(asset: MeasurementAsset, /) -> frozenset[str]:
    return frozenset(value.manifest_id for value in asset.references)


@dataclass(frozen=True, slots=True)
class TimeActivitySeries:
    """Instantaneous radionuclide activity samples on one explicit time axis.

    The decay identity is an existing :class:`InventoryTransition`; this contract
    deliberately does not define a second half-life or decay-record type.
    """

    asset: MeasurementAsset
    transition: InventoryTransition
    series_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MeasurementAsset):
            raise TypeError("asset must be MeasurementAsset.")
        if not isinstance(self.transition, InventoryTransition):
            raise TypeError("transition must be InventoryTransition.")
        if self.transition.flux_driven:
            raise ValueError("Time-activity series require a decay transition.")
        if self.asset.intended_use != "research":
            raise ValueError("Time-activity series are research-only assets.")
        field_ = self.asset.field
        if field_.quantity.quantity_kind not in _ACTIVITY_KINDS:
            raise ValueError(
                "Time-activity values must be activity or activity concentration."
            )
        if field_.layout.kind is not ValueKind.REAL_SCALAR:
            raise ValueError("Time-activity values must use a real-scalar layout.")
        time_axis, time_dimension = _time_axis_and_dimension(self.asset)
        if field_.sampling.temporal.kind is not TemporalSamplingKind.INSTANTANEOUS:
            raise ValueError(
                "Time-activity integration accepts instantaneous samples only."
            )
        if field_.values.shape[time_dimension] != time_axis.sample_count:
            raise ValueError("Time-activity values do not match the sample-time axis.")
        selected = np.asarray(field_.values)[np.asarray(field_.valid_mask)]
        if np.any(selected < 0.0):
            raise ValueError("Valid activity samples must be non-negative.")
        if self.transition.data.reference.manifest_id not in _reference_ids(self.asset):
            raise ValueError(
                "The activity asset references must include the transition data artifact."
            )
        if (
            self.asset.metadata.get("radionuclide_id")
            != self.transition.parent.nuclide_id
        ):
            raise ValueError(
                "Time-activity asset radionuclide identity must match the transition."
            )
        object.__setattr__(
            self,
            "series_id",
            canonical_fingerprint(
                {
                    "kind": "time-activity-series",
                    "asset": self.asset.content_id,
                    "time_axis": time_axis.time_axis_id,
                    "time_dimension": time_dimension,
                    "radionuclide": self.transition.parent.nuclide_id,
                    "transition": self.transition.transition_id,
                }
            ),
        )

    @property
    def time_axis(self) -> SampleTimeAxis:
        return _time_axis_and_dimension(self.asset)[0]

    @property
    def time_dimension(self) -> int:
        return _time_axis_and_dimension(self.asset)[1]

    @property
    def radionuclide(self):
        return self.transition.parent


class TimeActivityIntegralEvaluation(StrictModule):
    values: Array
    valid: Array


class PreparedTimeActivityIntegration(StrictModule, NonTrainableState):
    """Fixed weights for one finite-support piecewise-linear integral."""

    weights_s: Array
    used_samples: Array
    time_dimension: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def evaluate(
        self, values: ArrayLike, valid: ArrayLike, /
    ) -> TimeActivityIntegralEvaluation:
        activity = jnp.asarray(values)
        validity = jnp.asarray(valid, dtype=jnp.bool_)
        if activity.shape != validity.shape:
            raise ValueError("Activity values and validity must have the same shape.")
        if (
            activity.ndim <= self.time_dimension
            or activity.shape[self.time_dimension] != self.sample_count
        ):
            raise ValueError("Activity values do not match the integration time axis.")
        moved = jnp.moveaxis(activity, self.time_dimension, 0)
        moved_valid = jnp.moveaxis(validity, self.time_dimension, 0)
        canonical_valid = moved_valid & jnp.isfinite(moved) & (moved >= 0.0)
        safe = jnp.where(canonical_valid, moved, 0.0)
        values_out = jnp.tensordot(self.weights_s, safe, axes=((0,), (0,)))
        used = self.used_samples.reshape((self.sample_count,) + (1,) * (moved.ndim - 1))
        valid_out = jnp.all(jnp.where(used, canonical_valid, True), axis=0)
        return TimeActivityIntegralEvaluation(values_out, valid_out)


@dataclass(frozen=True, slots=True)
class TimeActivityIntegrationEvidence:
    source_id: str
    target_id: str
    radionuclide_id: str
    transition_id: str
    time_axis_id: str
    start_s: float
    end_s: float
    method: str
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = (
            self.source_id,
            self.target_id,
            self.radionuclide_id,
            self.transition_id,
            self.time_axis_id,
        )
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in identifiers
        ):
            raise ValueError("Integration evidence identifiers must be canonical text.")
        start, end = float(self.start_s), float(self.end_s)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Integration evidence requires a finite positive interval.")
        if self.method != "piecewise-linear-trapezoidal-no-extrapolation":
            raise ValueError("Unsupported time-activity integration method.")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        object.__setattr__(
            self,
            "evidence_id",
            canonical_fingerprint(
                {
                    "kind": "time-activity-integration-evidence",
                    "source": self.source_id,
                    "target": self.target_id,
                    "radionuclide": self.radionuclide_id,
                    "transition": self.transition_id,
                    "time_axis": self.time_axis_id,
                    "interval_s": [start, end],
                    "method": self.method,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TimeActivityIntegrationResult:
    asset: MeasurementAsset
    transition: InventoryTransition
    evidence: TimeActivityIntegrationEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MeasurementAsset):
            raise TypeError("asset must be MeasurementAsset.")
        if not isinstance(self.transition, InventoryTransition):
            raise TypeError("transition must be InventoryTransition.")
        if not isinstance(self.evidence, TimeActivityIntegrationEvidence):
            raise TypeError("evidence must be TimeActivityIntegrationEvidence.")
        if self.evidence.target_id != self.asset.content_id:
            raise ValueError(
                "Integration evidence target does not match the result asset."
            )
        if self.evidence.source_id not in self.asset.derivation.parent_ids:
            raise ValueError(
                "Integration evidence source is absent from result derivation."
            )
        if self.evidence.radionuclide_id != self.transition.parent.nuclide_id:
            raise ValueError("Integration evidence radionuclide and transition disagree.")
        if self.asset.intended_use != "research":
            raise ValueError("Time-integrated activity results are research-only.")
        if self.asset.field.quantity.quantity_kind not in {
            RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY.value,
            RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION.value,
        }:
            raise ValueError("Result asset does not contain time-integrated activity.")
        if self.transition.transition_id != self.evidence.transition_id:
            raise ValueError("Result transition and integration evidence disagree.")
        if self.asset.field.uncertainty is not None:
            raise ValueError(
                "Time integration cannot fabricate uncertainty without a covariance model."
            )

    @property
    def values(self) -> np.ndarray:
        return self.asset.field.values

    @property
    def valid_mask(self) -> np.ndarray:
        return self.asset.field.valid_mask

    @property
    def radionuclide(self):
        return self.transition.parent


@dataclass(frozen=True, slots=True)
class TimeActivityIntegrationPlan:
    """Trapezoidal integration over a closed subinterval of one sampled support."""

    time_axis: SampleTimeAxis
    start_time: float
    end_time: float
    time_unit: UnitDefinition
    weights_s: np.ndarray = field(init=False, repr=False)
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.time_axis, SampleTimeAxis):
            raise TypeError("time_axis must be SampleTimeAxis.")
        if (
            not isinstance(self.time_unit, UnitDefinition)
            or self.time_unit.dimension != TIME
        ):
            raise ValueError("time_unit must be a time UnitDefinition.")
        start = float(self.start_time)
        end = float(self.end_time)
        factor = float(conversion_factor(self.time_unit, SECOND))
        start_s, end_s = start * factor, end * factor
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            raise ValueError(
                "Integration endpoints must define a finite positive interval."
            )
        times = self.time_axis.values_in(SECOND)
        if times.size < 2:
            raise ValueError("Trapezoidal integration requires at least two samples.")
        if start_s < times[0] or end_s > times[-1]:
            raise ValueError(
                "Integration endpoints must lie inside the sampled time support; extrapolation is forbidden."
            )
        weights = np.zeros(times.shape, dtype=np.result_type(times.dtype, np.float64))
        for index in range(times.size - 1):
            left, right = float(times[index]), float(times[index + 1])
            lower, upper = max(start_s, left), min(end_s, right)
            if upper <= lower:
                continue
            width = right - left
            u_lower = (lower - left) / width
            u_upper = (upper - left) / width
            linear_moment = 0.5 * (u_upper * u_upper - u_lower * u_lower)
            span = u_upper - u_lower
            weights[index] += width * (span - linear_moment)
            weights[index + 1] += width * linear_moment
        if not np.any(weights > 0.0):
            raise ValueError("Integration interval has no sampled support.")
        weights.setflags(write=False)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "weights_s", weights)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "time-activity-integration-plan",
                    "time_axis": self.time_axis.time_axis_id,
                    "interval_s": [start_s, end_s],
                    "weights_s": array_tree_fingerprint(weights),
                    "method": "piecewise-linear-trapezoidal-no-extrapolation",
                }
            ),
        )

    @property
    def interval_seconds(self) -> tuple[float, float]:
        factor = float(conversion_factor(self.time_unit, SECOND))
        return self.start_time * factor, self.end_time * factor

    def prepare(self, time_dimension: int, /) -> PreparedTimeActivityIntegration:
        if isinstance(time_dimension, bool) or not isinstance(time_dimension, int):
            raise TypeError("time_dimension must be an integer.")
        if time_dimension < 0:
            raise ValueError("time_dimension must be non-negative.")
        return PreparedTimeActivityIntegration(
            jnp.asarray(self.weights_s),
            jnp.asarray(self.weights_s > 0.0),
            time_dimension,
            self.time_axis.sample_count,
            self.plan_id,
        )

    def integrate(self, series: TimeActivitySeries, /) -> TimeActivityIntegrationResult:
        if not isinstance(series, TimeActivitySeries):
            raise TypeError("series must be TimeActivitySeries.")
        if series.time_axis.time_axis_id != self.time_axis.time_axis_id:
            raise ValueError(
                "Time-activity series and integration plan use different axes."
            )
        source = series.asset
        field_ = source.field
        activity_kind = field_.quantity.quantity_kind
        source_reference_unit = _REFERENCE_UNIT[activity_kind]
        scale = float(conversion_factor(field_.quantity.unit, source_reference_unit))
        evaluation = self.prepare(series.time_dimension).evaluate(
            np.asarray(field_.values) * scale,
            field_.valid_mask,
        )
        values = np.asarray(evaluation.values)
        valid = np.asarray(evaluation.valid)
        support = _reduced_support(source, series.time_dimension)
        output_kind = _INTEGRATED_KIND[activity_kind]
        quantity = resolve_radiation_quantity(
            f"{field_.quantity.name}-time-integral",
            output_kind,
            _INTEGRATED_UNIT[activity_kind],
            axes=_reduced_quantity_axes(source, series.time_dimension),
            sign_convention=field_.quantity.sign_convention,
            support_association=field_.quantity.support_association,
            reference_configuration=field_.quantity.reference_configuration,
        )
        start_s, end_s = self.interval_seconds
        sampling = SamplingSemantics(
            field_.sampling.spatial_kind,
            TemporalSampling(
                TemporalSamplingKind.CUMULATIVE,
                origin=(
                    f"piecewise-linear-integral:{self.time_axis.time_axis_id}:{start_s:.17g}s:{end_s:.17g}s"
                ),
            ),
            field_.sampling.footprint_id,
            field_.sampling.normalization,
        )
        used = self.weights_s > 0.0
        flags = []
        for flag_ in field_.quality_flags:
            moved = np.moveaxis(flag_.mask, series.time_dimension, 0)
            flags.append(
                QualityFlag(
                    flag_.name,
                    np.any(moved[used], axis=0),
                    flag_.meaning,
                )
            )
        output_field = QuantityField(
            f"{field_.field_id}:time-integral:{self.plan_id[:12]}",
            quantity,
            field_.layout,
            support,
            sampling,
            values,
            valid,
            None,
            tuple(flags),
        )
        target = MeasurementAsset(
            f"{source.asset_id}:time-integral:{self.plan_id[:12]}",
            output_field,
            source.acquisition,
            source.references,
            DerivationRecord(
                source.derivation.origin,
                DataStage.DERIVED,
                (source.content_id,),
                self.plan_id,
            ),
            "research",
            {
                "time_activity_series_id": series.series_id,
                "time_axis_id": self.time_axis.time_axis_id,
                "transition_id": series.transition.transition_id,
                "radionuclide_id": series.radionuclide.nuclide_id,
                "uncertainty_propagation": (
                    "not-performed-covariance-unspecified"
                    if field_.uncertainty is not None
                    else "unquantified-no-input-uncertainty"
                ),
            },
        )
        evidence = TimeActivityIntegrationEvidence(
            source.content_id,
            target.content_id,
            series.radionuclide.nuclide_id,
            series.transition.transition_id,
            self.time_axis.time_axis_id,
            start_s,
            end_s,
            "piecewise-linear-trapezoidal-no-extrapolation",
        )
        return TimeActivityIntegrationResult(target, series.transition, evidence)


__all__ = [
    "PreparedTimeActivityIntegration",
    "TimeActivityIntegralEvaluation",
    "TimeActivityIntegrationEvidence",
    "TimeActivityIntegrationPlan",
    "TimeActivityIntegrationResult",
    "TimeActivitySeries",
]
