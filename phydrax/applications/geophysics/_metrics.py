"""Physical-unit verification on explicit lead/member/time/area/layer axes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...ml.metrics import (
    crps_ensemble,
    energy_score,
    mean_squared_error,
    METRIC_SUCCESS,
    METRIC_ZERO_DENOMINATOR,
    MetricResult,
    root_mean_squared_error,
)
from ._quantities import GeophysicalQuantity


@dataclass(frozen=True)
class GeophysicalClimatology:
    values: Any
    quantity_ids: tuple[str, ...]
    training_case_ids: tuple[str, ...]
    source_id: str

    def __post_init__(self):
        object.__setattr__(self, "values", jnp.asarray(self.values))
        object.__setattr__(self, "quantity_ids", tuple(self.quantity_ids))
        object.__setattr__(self, "training_case_ids", tuple(self.training_case_ids))
        if (
            not self.source_id
            or not self.training_case_ids
            or len(set(self.training_case_ids)) != len(self.training_case_ids)
        ):
            raise ValueError(
                "Anomaly climatology requires a source and unique training-case provenance."
            )
        if (
            self.values.ndim != 3
            or self.values.shape[-1] != len(self.quantity_ids)
            or not np.all(np.isfinite(np.asarray(self.values)))
        ):
            raise ValueError(
                "Climatology must be finite (area, layer, ordered variable)."
            )

    @property
    def climatology_id(self):
        return canonical_fingerprint(
            {
                "source": self.source_id,
                "cases": self.training_case_ids,
                "quantities": self.quantity_ids,
                "mean": array_tree_fingerprint(self.values),
            }
        )


def fit_geophysical_climatology(
    values: Any,
    /,
    *,
    quantities: Sequence[GeophysicalQuantity],
    time_weights: Any,
    mask: Any,
    training_case_ids: Sequence[str],
    source_id: str,
) -> GeophysicalClimatology:
    """Fit a fixed temporal climatology from identified training records only."""
    data = jnp.asarray(values)
    weights = np.asarray(time_weights)
    if (
        data.ndim != 4
        or weights.shape != (data.shape[0],)
        or len(training_case_ids) != data.shape[0]
    ):
        raise ValueError(
            "Climatology data require (time, area, layer, variable) and one case ID/time weight per time."
        )
    if not np.all(np.isfinite(weights) & (weights >= 0)):
        raise ValueError("Climatology time weights must be finite and nonnegative.")
    active = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), data.shape)
    if not np.all(np.asarray(~active | jnp.isfinite(data))):
        raise ValueError("Active climatology samples must be finite.")
    weight = jnp.asarray(weights)[:, None, None, None] * active
    mass = jnp.sum(weight, axis=0)
    if not np.all(np.asarray(mass) > 0):
        raise ValueError("Every climatology site requires positive training measure.")
    mean = jnp.sum(jnp.where(active, data, 0) * weight, axis=0) / mass
    return GeophysicalClimatology(
        mean,
        tuple(q.quantity_id for q in quantities),
        tuple(training_case_ids),
        source_id,
    )


@dataclass(frozen=True)
class GeophysicalFieldMetrics:
    quantity: GeophysicalQuantity
    rmse: MetricResult
    crps: MetricResult
    bias: MetricResult
    anomaly_correlation: MetricResult | None
    drift_per_second: MetricResult | None

    @property
    def error_unit(self):
        """RMSE and CRPS have the field unit, not the squared-error unit."""
        return self.quantity.unit


@dataclass(frozen=True)
class GeophysicalForecastMetrics:
    fields: tuple[GeophysicalFieldMetrics, ...]
    lead_seconds: tuple[float, ...]
    member_ids: tuple[str, ...]
    climatology_id: str | None
    scaled_energy_score: MetricResult | None
    multivariate_scales: tuple[float, ...] | None


def _diagnostic(value, reference: MetricResult, *, defined=None):
    defined = jnp.ones_like(reference.valid) if defined is None else defined
    valid = reference.valid & defined
    return MetricResult(
        jnp.where(valid, value, jnp.nan),
        valid=valid,
        status=jnp.where(
            reference.valid & ~defined, METRIC_ZERO_DENOMINATOR, reference.status
        ),
        effective_weight=reference.effective_weight,
    )


def geophysical_forecast_metrics(
    truth: Any,
    ensemble: Any,
    /,
    *,
    quantities: Sequence[GeophysicalQuantity],
    lead_seconds: Sequence[float],
    member_ids: Sequence[str],
    area_weights: Any,
    layer_weights: Any,
    time_weights: Any,
    mask: Any,
    member_weights: Any | None = None,
    climatology: GeophysicalClimatology | None = None,
    verification_case_ids: Sequence[str] = (),
    multivariate_scales: Sequence[float] | None = None,
) -> GeophysicalForecastMetrics:
    """Verify truth (lead,time,area,layer,variable), ensemble (lead,member,...).

    RMSE/bias/ACC retain (lead,member); CRPS and scaled energy retain lead.
    The common mask has truth's exact shape; masks never silently drop NaNs.
    Layer weights are either (layer,) or (lead,time,area,layer). No variable
    averaging occurs unless positive physical multivariate scales are supplied.
    """
    true, forecast = jnp.asarray(truth), jnp.asarray(ensemble)
    quantities = tuple(quantities)
    leads = np.asarray(lead_seconds, dtype=float)
    members = tuple(member_ids)
    if (
        true.ndim != 5
        or forecast.ndim != 6
        or forecast.shape[:1] + forecast.shape[2:] != true.shape
    ):
        raise ValueError(
            "Verification requires explicit lead/member/time/area/layer/variable axes."
        )
    nl, nt, na, nz, nv = true.shape
    nm = forecast.shape[1]
    if (
        leads.shape != (nl,)
        or not np.all(np.isfinite(leads))
        or np.any(leads < 0)
        or np.any(np.diff(leads) <= 0)
    ):
        raise ValueError(
            "Lead seconds must be finite, nonnegative and strictly increasing."
        )
    if (
        len(members) != nm
        or len(set(members)) != nm
        or not all(members)
        or len(quantities) != nv
    ):
        raise ValueError("Member and physical variable identities must match their axes.")
    if len({q.name for q in quantities}) != nv:
        raise ValueError(
            "Verification quantities must have distinct names in variable order."
        )
    area, duration, layer = map(np.asarray, (area_weights, time_weights, layer_weights))
    if (
        area.shape != (na,)
        or duration.shape != (nt,)
        or layer.shape not in ((nz,), (nl, nt, na, nz))
    ):
        raise ValueError("Area/time/layer weights must match their explicit axes.")
    if not all(np.all(np.isfinite(w) & (w >= 0)) for w in (area, duration, layer)):
        raise ValueError("Physical verification weights must be finite and nonnegative.")
    included = jnp.asarray(mask, dtype=bool)
    if included.shape != true.shape:
        raise ValueError("The verification mask must have truth's full shape.")
    measure = jnp.broadcast_to(
        jnp.asarray(duration)[None, :, None, None]
        * jnp.asarray(area)[None, None, :, None]
        * jnp.asarray(layer),
        true.shape[:-1],
    )
    weights = measure.reshape(nl, -1)
    mw = jnp.ones(nm) if member_weights is None else jnp.asarray(member_weights)
    if (
        mw.shape != (nm,)
        or not np.all(np.isfinite(np.asarray(mw)) & (np.asarray(mw) >= 0))
        or float(jnp.sum(mw)) <= 0
    ):
        raise ValueError(
            "Member weights must have positive finite nonnegative total mass."
        )
    if climatology is not None:
        if not verification_case_ids or set(verification_case_ids) & set(
            climatology.training_case_ids
        ):
            raise ValueError(
                "Climatology must be independent of identified verification cases."
            )
        if climatology.quantity_ids != tuple(
            q.quantity_id for q in quantities
        ) or climatology.values.shape != (na, nz, nv):
            raise ValueError(
                "Climatology geometry/physical variable order does not match verification."
            )
    results = []
    for variable, quantity in enumerate(quantities):
        observed = true[..., variable].reshape(nl, -1)
        predicted = forecast[..., variable].reshape(nl, nm, -1)
        active = included[..., variable].reshape(nl, -1)
        # Native CRPS uses a trailing member axis. Explicitly neutralize excluded
        # samples before member pair differences, including masked NaN storage.
        safe_true = jnp.where(active, observed, 0)
        safe_pred = jnp.where(active[:, None, :], predicted, 0)
        repeated_true = jnp.broadcast_to(safe_true[:, None, :], safe_pred.shape)
        rmse = root_mean_squared_error(
            repeated_true,
            safe_pred,
            sample_weight=weights[:, None, :],
            mask=active[:, None, :],
        )
        crps = crps_ensemble(
            safe_true,
            jnp.moveaxis(safe_pred, 1, -1),
            sample_weight=weights,
            mask=active,
            member_weight=mw,
        )
        w = jnp.where(active, weights, 0)
        mass = jnp.sum(w, axis=-1)
        safe_mass = jnp.where(mass > 0, mass, 1)
        bias_values = (
            jnp.sum((safe_pred - repeated_true) * w[:, None, :], axis=-1)
            / safe_mass[:, None]
        )
        bias = _diagnostic(bias_values, rmse)
        acc = None
        if climatology is not None:
            climate = jnp.broadcast_to(
                climatology.values[None, ..., variable], (nt, na, nz)
            ).reshape(-1)
            a = jnp.where(active, safe_true - climate, 0)
            b = jnp.where(active[:, None, :], safe_pred - climate, 0)
            numerator = jnp.sum(w[:, None, :] * a[:, None, :] * b, axis=-1)
            denominator = jnp.sqrt(
                jnp.sum(w * a * a, axis=-1)[:, None]
                * jnp.sum(w[:, None, :] * b * b, axis=-1)
            )
            acc = _diagnostic(
                numerator / jnp.where(denominator > 0, denominator, 1),
                rmse,
                defined=denominator > 0,
            )
        drift = None
        if nl > 1:
            # Endpoint physical bias trend, not a trend of squared errors.
            drift_valid = bias.valid[-1] & bias.valid[0]
            drift = MetricResult(
                (bias.value[-1] - bias.value[0]) / (leads[-1] - leads[0]),
                valid=drift_valid,
                status=jnp.where(
                    drift_valid,
                    METRIC_SUCCESS,
                    jnp.where(~bias.valid[0], bias.status[0], bias.status[-1]),
                ),
                effective_weight=jnp.minimum(
                    bias.effective_weight[0], bias.effective_weight[-1]
                ),
            )
        results.append(GeophysicalFieldMetrics(quantity, rmse, crps, bias, acc, drift))
    scales = None
    energy = None
    if multivariate_scales is not None:
        scales = tuple(float(value) for value in multivariate_scales)
        if (
            len(scales) != nv
            or not np.all(np.isfinite(scales))
            or not np.all(np.asarray(scales) > 0)
        ):
            raise ValueError(
                "Multivariate energy requires one positive physical scale per variable."
            )
        event_mask = jnp.all(included, axis=-1).reshape(nl, -1)
        scaled_true = true.reshape(nl, -1, nv) / jnp.asarray(scales)
        scaled_forecast = (
            jnp.moveaxis(forecast.reshape(nl, nm, -1, nv), 1, -1)
            / jnp.asarray(scales)[None, None, :, None]
        )
        energy = energy_score(
            jnp.where(event_mask[..., None], scaled_true, 0),
            jnp.where(event_mask[..., None, None], scaled_forecast, 0),
            sample_weight=weights,
            mask=event_mask,
            member_weight=mw,
        )
    return GeophysicalForecastMetrics(
        tuple(results),
        tuple(float(v) for v in leads),
        members,
        None if climatology is None else climatology.climatology_id,
        energy,
        scales,
    )


@dataclass(frozen=True)
class GeophysicalExtremeReliability:
    threshold: float
    quantity: GeophysicalQuantity
    brier: MetricResult
    bin_probability: Any
    bin_frequency: Any
    bin_weight: Any
    bin_edges: tuple[float, ...]


def geophysical_extreme_reliability(
    truth: Any,
    ensemble: Any,
    /,
    *,
    quantity: GeophysicalQuantity,
    threshold: float,
    sample_weights: Any,
    mask: Any,
    bin_edges: Sequence[float] = (0, 0.2, 0.4, 0.6, 0.8, 1),
) -> GeophysicalExtremeReliability:
    """Weighted exceedance reliability; arrays (lead,sample[,member]).

    Members are equally weighted. Empty reliability bins are NaN with zero mass,
    not perfect reliability. Brier is dimensionless; threshold has quantity units.
    """
    true, forecast = jnp.asarray(truth), jnp.asarray(ensemble)
    if true.ndim != 2 or forecast.shape[:-1] != true.shape or forecast.shape[-1] <= 0:
        raise ValueError("Extreme verification requires (lead,sample[,member]).")
    if not np.isfinite(threshold):
        raise ValueError(
            "Extreme threshold must be finite in the physical quantity unit."
        )
    edges = np.asarray(bin_edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or edges[0] != 0
        or edges[-1] != 1
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("Reliability edges must strictly partition [0,1].")
    active = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), true.shape)
    weights = jnp.broadcast_to(jnp.asarray(sample_weights), true.shape)
    if not np.all(np.isfinite(np.asarray(weights)) & (np.asarray(weights) >= 0)):
        raise ValueError("Reliability weights must be finite and nonnegative.")
    if not np.all(
        np.asarray(
            ~active | (jnp.isfinite(true) & jnp.all(jnp.isfinite(forecast), axis=-1))
        )
    ):
        raise ValueError(
            "Active extreme observations and ensemble members must be finite."
        )
    probability = jnp.mean(forecast > threshold, axis=-1)
    observed = (true > threshold).astype(probability.dtype)
    brier = mean_squared_error(observed, probability, sample_weight=weights, mask=active)
    bins_probability, bins_frequency, bins_weight = [], [], []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (
            active
            & (probability >= left)
            & (
                (probability <= right)
                if index == len(edges) - 2
                else (probability < right)
            )
        )
        w = jnp.where(selected, weights, 0)
        mass = jnp.sum(w, axis=-1)
        denominator = jnp.where(mass > 0, mass, 1)
        bins_probability.append(
            jnp.where(mass > 0, jnp.sum(w * probability, axis=-1) / denominator, jnp.nan)
        )
        bins_frequency.append(
            jnp.where(mass > 0, jnp.sum(w * observed, axis=-1) / denominator, jnp.nan)
        )
        bins_weight.append(mass)
    return GeophysicalExtremeReliability(
        float(threshold),
        quantity,
        brier,
        jnp.stack(bins_probability, axis=-1),
        jnp.stack(bins_frequency, axis=-1),
        jnp.stack(bins_weight, axis=-1),
        tuple(float(v) for v in edges),
    )


def geophysical_spectral_rmse(
    truth_coefficients: Any,
    forecast_coefficients: Any,
    /,
    *,
    mode_weights: Any,
    mask: Any,
) -> MetricResult:
    """Native weighted RMSE of compatible physical spectral coefficients.

    The last axis is mode; preceding lead/member axes are retained. Coefficients
    must come from the same native transform and normalization (e.g. spherical
    spectral space), not from an FFT applied across a flattened sphere. Physical
    field units are unchanged for dimensionless orthonormal basis functions.
    """
    return root_mean_squared_error(
        truth_coefficients, forecast_coefficients, sample_weight=mode_weights, mask=mask
    )
