#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike

from .._numerics import MetricName, pairwise_distances, segmented_weighted_mean
from ._base import (
    _broadcast_full,
    _normalize_axis,
    _prepare_pair,
    _prepare_values,
    _real_dtype,
    _reject_complex,
    _result,
    METRIC_SINGLE_CLASS,
    METRIC_ZERO_DENOMINATOR,
    MetricResult,
)


def _hard_cluster_inputs(
    features: ArrayLike,
    labels: ArrayLike,
    /,
    *,
    num_clusters: int,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
) -> tuple[Array, Array, Array, Array, Array, int]:
    x = jnp.asarray(features)
    labels_raw = jnp.asarray(labels)
    if x.ndim < 2 or labels_raw.shape != x.shape[:-1]:
        raise ValueError(
            f"{metric} requires features shaped case_shape + (sample, feature) and aligned labels."
        )
    axis = _normalize_axis(sample_axis, x.ndim)
    if axis != x.ndim - 2:
        raise ValueError(
            "Clustering features must have case_shape + (sample, feature) axes."
        )
    if (
        not jnp.issubdtype(labels_raw.dtype, jnp.integer)
        and labels_raw.dtype != jnp.bool_
    ):
        raise TypeError(f"{metric} requires integer cluster labels.")
    classes = int(num_clusters)
    if classes < 2:
        raise ValueError("num_clusters must be at least two.")
    labels_, weights, active, invalid, _ = _prepare_values(
        labels_raw,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=labels_raw.ndim - 1,
        metric=metric,
        allow_complex=False,
    )
    included = _broadcast_full(
        mask,
        tuple(int(size) for size in labels_raw.shape),
        dtype=bool,
        fill=True,
        name="mask",
    )
    features_finite = jnp.all(jnp.isfinite(x), axis=-1)
    labels_in_range = (labels_ >= 0) & (labels_ < classes)
    invalid = invalid | jnp.any(included & ~(features_finite & labels_in_range), axis=-1)
    active = active & features_finite & labels_in_range
    weights = jnp.where(active, weights, 0.0)
    x = jnp.where(active[..., None], x, 0)
    mass = jnp.sum(weights, axis=-1)
    return x, labels_.astype(jnp.int32), weights, invalid, mass, classes


def _soft_cluster_inputs(
    features: ArrayLike,
    memberships: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
    from_logits: bool,
) -> tuple[Array, Array, Array, Array, Array, int]:
    x = jnp.asarray(features)
    raw = jnp.asarray(memberships)
    if x.ndim < 2 or raw.ndim != x.ndim or raw.shape[:-1] != x.shape[:-1]:
        raise ValueError(
            f"{metric} requires aligned case_shape + (sample, feature/cluster) arrays."
        )
    axis = _normalize_axis(sample_axis, x.ndim)
    if axis != x.ndim - 2:
        raise ValueError("Clustering inputs must have case_shape + (sample, value) axes.")
    _reject_complex(raw, metric=metric)
    classes = int(raw.shape[-1])
    if classes < 2:
        raise ValueError(f"{metric} requires at least two clusters.")
    dummy = jnp.zeros(x.shape[:-1], dtype=x.real.dtype)
    _, weights, active, invalid, _ = _prepare_values(
        dummy,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=dummy.ndim - 1,
        metric=metric,
        allow_complex=False,
    )
    included = _broadcast_full(
        mask,
        tuple(int(size) for size in dummy.shape),
        dtype=bool,
        fill=True,
        name="mask",
    )
    features_finite = jnp.all(jnp.isfinite(x), axis=-1)
    raw_finite = jnp.all(jnp.isfinite(raw), axis=-1)
    if from_logits:
        probability = jax.nn.softmax(raw, axis=-1)
        membership_valid = raw_finite
    else:
        dtype = _real_dtype(raw)
        tolerance = jnp.asarray(32 * classes, dtype=dtype) * jnp.finfo(dtype).eps
        membership_valid = (
            raw_finite
            & jnp.all(raw >= 0.0, axis=-1)
            & jnp.all(raw <= 1.0, axis=-1)
            & (jnp.abs(jnp.sum(raw, axis=-1) - 1.0) <= tolerance)
        )
        probability = raw
    valid_sample = features_finite & membership_valid
    invalid = invalid | jnp.any(included & ~valid_sample, axis=-1)
    active = active & valid_sample
    weights = jnp.where(active, weights, 0.0)
    x = jnp.where(active[..., None], x, 0)
    probability = jnp.where(active[..., None], probability, 0.0)
    mass = jnp.sum(weights, axis=-1)
    return x, probability, weights, invalid, mass, classes


def silhouette_score(
    features: ArrayLike,
    labels: ArrayLike,
    /,
    *,
    num_clusters: int,
    distance: MetricName = "euclidean",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -2,
) -> MetricResult:
    """Exact weighted mean silhouette under hard labels and a hard nearest cluster."""
    x, labels_, weights, invalid, mass, classes = _hard_cluster_inputs(
        features,
        labels,
        num_clusters=num_clusters,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="silhouette_score",
    )
    distances = pairwise_distances(x, metric=distance)
    distances = jnp.where(jnp.eye(x.shape[-2], dtype=bool), 0.0, distances)
    membership = (
        jax.nn.one_hot(labels_, classes, dtype=weights.dtype) * weights[..., :, None]
    )
    cluster_mass = jnp.sum(membership, axis=-2)
    distance_total = jnp.einsum("...ij,...jc->...ic", distances, membership)
    own_hot = jax.nn.one_hot(labels_, classes, dtype=bool)
    own_denominator = cluster_mass[..., None, :] - membership
    cluster_mean = distance_total / jnp.where(own_denominator > 0.0, own_denominator, 1.0)
    intra = jnp.sum(jnp.where(own_hot, cluster_mean, 0.0), axis=-1)
    other_available = (~own_hot) & (cluster_mass[..., None, :] > 0.0)
    inter = jnp.min(jnp.where(other_available, cluster_mean, jnp.inf), axis=-1)
    own_mass = jnp.sum(jnp.where(own_hot, own_denominator, 0.0), axis=-1)
    denominator = jnp.maximum(intra, inter)
    sample_value = jnp.where(
        own_mass > 0.0,
        (inter - intra) / jnp.where(denominator > 0.0, denominator, 1.0),
        0.0,
    )
    value = jnp.sum(weights * sample_value, axis=-1) / jnp.where(mass > 0.0, mass, 1.0)
    represented = jnp.sum(cluster_mass > 0.0, axis=-1)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=represented < 2,
        undefined_status=METRIC_SINGLE_CLASS,
    )


def smooth_silhouette_score(
    features: ArrayLike,
    memberships: ArrayLike,
    /,
    *,
    temperature: float = 1.0,
    distance: MetricName = "euclidean",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -2,
    from_logits: bool = False,
) -> MetricResult:
    """Soft-membership, soft-min silhouette surrogate."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    x, probability, weights, invalid, mass, _ = _soft_cluster_inputs(
        features,
        memberships,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_silhouette_score",
        from_logits=from_logits,
    )
    distances = pairwise_distances(x, metric=distance)
    distances = jnp.where(jnp.eye(x.shape[-2], dtype=bool), 0.0, distances)
    weighted_membership = weights[..., :, None] * probability
    cluster_mass = jnp.sum(weighted_membership, axis=-2)
    distance_total = jnp.einsum("...ij,...jc->...ic", distances, weighted_membership)
    denominator = cluster_mass[..., None, :] - weighted_membership
    cluster_mean = distance_total / jnp.where(denominator > 0.0, denominator, 1.0)
    intra = jnp.sum(probability * cluster_mean, axis=-1)
    other_weight = (1.0 - probability) * (denominator > 0.0)
    logits = -cluster_mean / float(temperature)
    normalizer = jnp.sum(other_weight, axis=-1)
    inter = -float(temperature) * (
        logsumexp(logits, axis=-1, b=other_weight)
        - jnp.log(jnp.where(normalizer > 0.0, normalizer, 1.0))
    )
    smooth_max = float(temperature) * jnp.logaddexp(
        intra / float(temperature), inter / float(temperature)
    )
    sample_value = (inter - intra) / jnp.where(smooth_max > 0.0, smooth_max, 1.0)
    value = jnp.sum(weights * sample_value, axis=-1) / jnp.where(mass > 0.0, mass, 1.0)
    cluster_present = jnp.sum(cluster_mass > 0.0, axis=-1)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=cluster_present < 2,
        undefined_status=METRIC_SINGLE_CLASS,
    )


def davies_bouldin_score(
    features: ArrayLike,
    labels: ArrayLike,
    /,
    *,
    num_clusters: int,
    distance: MetricName = "euclidean",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -2,
) -> MetricResult:
    """Weighted Davies-Bouldin index over represented hard clusters."""
    x, labels_, weights, invalid, mass, classes = _hard_cluster_inputs(
        features,
        labels,
        num_clusters=num_clusters,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="davies_bouldin_score",
    )
    centroids, cluster_mass = segmented_weighted_mean(
        x, labels_, weights, num_segments=classes
    )
    point_to_centroid = pairwise_distances(x, centroids, metric=distance)
    selected_distance = jnp.take_along_axis(
        point_to_centroid, labels_[..., :, None], axis=-1
    )[..., 0]
    scatter_total = jnp.einsum(
        "...n,...nc->...c",
        weights * selected_distance,
        jax.nn.one_hot(labels_, classes, dtype=weights.dtype),
    )
    scatter = scatter_total / jnp.where(cluster_mass > 0.0, cluster_mass, 1.0)
    centroid_distance = pairwise_distances(centroids, metric=distance)
    ratio = (scatter[..., :, None] + scatter[..., None, :]) / jnp.where(
        centroid_distance > 0.0, centroid_distance, 1.0
    )
    represented = cluster_mass > 0.0
    candidates = (
        represented[..., :, None]
        & represented[..., None, :]
        & ~jnp.eye(classes, dtype=bool)
    )
    worst = jnp.max(jnp.where(candidates, ratio, -jnp.inf), axis=-1)
    represented_count = jnp.sum(represented, axis=-1)
    value = jnp.sum(jnp.where(represented, worst, 0.0), axis=-1) / jnp.maximum(
        represented_count, 1
    )
    coincident = jnp.any(candidates & (centroid_distance <= 0.0), axis=(-2, -1))
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=(represented_count < 2) | coincident,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def calinski_harabasz_score(
    features: ArrayLike,
    labels: ArrayLike,
    /,
    *,
    num_clusters: int,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -2,
) -> MetricResult:
    """Weighted variance-ratio criterion for hard clusters."""
    x, labels_, weights, invalid, mass, classes = _hard_cluster_inputs(
        features,
        labels,
        num_clusters=num_clusters,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="calinski_harabasz_score",
    )
    centroids, cluster_mass = segmented_weighted_mean(
        x, labels_, weights, num_segments=classes
    )
    global_mean = jnp.sum(weights[..., :, None] * x, axis=-2) / jnp.where(
        mass[..., None] > 0.0, mass[..., None], 1.0
    )
    selected_centroid = jnp.einsum(
        "...nc,...cf->...nf",
        jax.nn.one_hot(labels_, classes, dtype=centroids.real.dtype),
        centroids,
    )
    within = jnp.sum(
        weights
        * jnp.sum(
            jnp.real((x - selected_centroid) * jnp.conj(x - selected_centroid)), axis=-1
        ),
        axis=-1,
    )
    center_delta = centroids - global_mean[..., None, :]
    between = jnp.sum(
        cluster_mass * jnp.sum(jnp.real(center_delta * jnp.conj(center_delta)), axis=-1),
        axis=-1,
    )
    represented_count = jnp.sum(cluster_mass > 0.0, axis=-1)
    denominator_df = mass - represented_count
    value = (between / jnp.maximum(represented_count - 1, 1)) / jnp.where(
        within > 0.0,
        within / jnp.where(denominator_df > 0.0, denominator_df, 1.0),
        1.0,
    )
    undefined = (represented_count < 2) | (denominator_df <= 0.0) | (within <= 0.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=undefined,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def smooth_calinski_harabasz_score(
    features: ArrayLike,
    memberships: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -2,
    from_logits: bool = False,
) -> MetricResult:
    """Soft-membership variance-ratio surrogate."""
    x, probability, weights, invalid, mass, classes = _soft_cluster_inputs(
        features,
        memberships,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_calinski_harabasz_score",
        from_logits=from_logits,
    )
    weighted_membership = weights[..., :, None] * probability
    cluster_mass = jnp.sum(weighted_membership, axis=-2)
    centroids = jnp.einsum("...nc,...nf->...cf", weighted_membership, x) / jnp.where(
        cluster_mass[..., :, None] > 0.0, cluster_mass[..., :, None], 1.0
    )
    global_mean = jnp.sum(weights[..., :, None] * x, axis=-2) / jnp.where(
        mass[..., None] > 0.0, mass[..., None], 1.0
    )
    delta = x[..., :, None, :] - centroids[..., None, :, :]
    within = jnp.sum(
        weighted_membership * jnp.sum(jnp.real(delta * jnp.conj(delta)), axis=-1),
        axis=(-2, -1),
    )
    center_delta = centroids - global_mean[..., None, :]
    between = jnp.sum(
        cluster_mass * jnp.sum(jnp.real(center_delta * jnp.conj(center_delta)), axis=-1),
        axis=-1,
    )
    denominator_df = mass - classes
    value = (between / float(classes - 1)) / jnp.where(
        within > 0.0,
        within / jnp.where(denominator_df > 0.0, denominator_df, 1.0),
        1.0,
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=(denominator_df <= 0.0)
        | (within <= 0.0)
        | jnp.any(cluster_mass <= 0.0, axis=-1),
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def _label_pair_contingency(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    /,
    *,
    num_true_clusters: int,
    num_pred_clusters: int,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
) -> tuple[Array, Array, Array, Array]:
    true_raw = jnp.asarray(labels_true)
    pred_raw = jnp.asarray(labels_pred)
    if not jnp.issubdtype(true_raw.dtype, jnp.integer) or not jnp.issubdtype(
        pred_raw.dtype, jnp.integer
    ):
        raise TypeError(f"{metric} requires integer labels.")
    true, pred, weights, active, invalid, axis = _prepare_pair(
        true_raw,
        pred_raw,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric=metric,
        allow_complex=False,
    )
    if axis != true.ndim - 1:
        raise ValueError("Cluster label arrays must have case_shape + (sample,) axes.")
    rows = int(num_true_clusters)
    columns = int(num_pred_clusters)
    if rows < 1 or columns < 1:
        raise ValueError("Cluster counts must be positive.")
    in_range = (true >= 0) & (true < rows) & (pred >= 0) & (pred < columns)
    invalid = invalid | jnp.any(active & ~in_range, axis=-1)
    weights = jnp.where(active & in_range, weights, 0.0)
    true_hot = jax.nn.one_hot(true.astype(jnp.int32), rows, dtype=weights.dtype)
    pred_hot = jax.nn.one_hot(pred.astype(jnp.int32), columns, dtype=weights.dtype)
    contingency = jnp.einsum("...ni,...n,...nj->...ij", true_hot, weights, pred_hot)
    second_moment = jnp.einsum("...ni,...n,...nj->...ij", true_hot, weights**2, pred_hot)
    return contingency, second_moment, jnp.sum(weights, axis=-1), invalid


def adjusted_rand_score(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    /,
    *,
    num_true_clusters: int,
    num_pred_clusters: int,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Pair-weighted adjusted Rand index from a hard contingency table."""
    contingency, second_moment, mass, invalid = _label_pair_contingency(
        labels_true,
        labels_pred,
        num_true_clusters=num_true_clusters,
        num_pred_clusters=num_pred_clusters,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="adjusted_rand_score",
    )

    def pair_mass(first: Array, second: Array, /) -> Array:
        return 0.5 * (first**2 - second)

    cells = jnp.sum(pair_mass(contingency, second_moment), axis=(-2, -1))
    row_first = jnp.sum(contingency, axis=-1)
    row_second = jnp.sum(second_moment, axis=-1)
    column_first = jnp.sum(contingency, axis=-2)
    column_second = jnp.sum(second_moment, axis=-2)
    rows = jnp.sum(pair_mass(row_first, row_second), axis=-1)
    columns = jnp.sum(pair_mass(column_first, column_second), axis=-1)
    pairs = pair_mass(mass, jnp.sum(second_moment, axis=(-2, -1)))
    expected = rows * columns / jnp.where(pairs > 0.0, pairs, 1.0)
    maximum = 0.5 * (rows + columns)
    denominator = maximum - expected
    value = (cells - expected) / jnp.where(denominator != 0.0, denominator, 1.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=(pairs <= 0.0) | (denominator == 0.0),
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def normalized_mutual_info_score(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    /,
    *,
    num_true_clusters: int,
    num_pred_clusters: int,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Geometrically normalized mutual information of hard cluster labels."""
    contingency, _, mass, invalid = _label_pair_contingency(
        labels_true,
        labels_pred,
        num_true_clusters=num_true_clusters,
        num_pred_clusters=num_pred_clusters,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="normalized_mutual_info_score",
    )
    probability = contingency / jnp.where(
        mass[..., None, None] > 0.0, mass[..., None, None], 1.0
    )
    row = jnp.sum(probability, axis=-1)
    column = jnp.sum(probability, axis=-2)
    independent = row[..., :, None] * column[..., None, :]
    mutual_information = jnp.sum(
        jnp.where(
            probability > 0.0,
            probability
            * jnp.log(probability / jnp.where(independent > 0.0, independent, 1.0)),
            0.0,
        ),
        axis=(-2, -1),
    )
    row_entropy = -jnp.sum(jnp.where(row > 0.0, row * jnp.log(row), 0.0), axis=-1)
    column_entropy = -jnp.sum(
        jnp.where(column > 0.0, column * jnp.log(column), 0.0), axis=-1
    )
    denominator = jnp.sqrt(row_entropy * column_entropy)
    value = mutual_information / jnp.where(denominator > 0.0, denominator, 1.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def _membership_pair_inputs(
    memberships_true: ArrayLike,
    memberships_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    metric: str,
) -> tuple[Array, Array, Array, Array, Array]:
    true = jnp.asarray(memberships_true)
    pred = jnp.asarray(memberships_pred)
    if true.ndim < 2 or pred.ndim != true.ndim or pred.shape[:-1] != true.shape[:-1]:
        raise ValueError(f"{metric} requires aligned case/sample membership arrays.")
    _reject_complex(true, pred, metric=metric)
    dummy = jnp.zeros(true.shape[:-1], dtype=_real_dtype(true, pred))
    _, weights, active, invalid, _ = _prepare_values(
        dummy,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=-1,
        metric=metric,
        allow_complex=False,
    )
    tolerance = 32 * max(true.shape[-1], pred.shape[-1]) * jnp.finfo(dummy.dtype).eps
    true_valid = jnp.all(jnp.isfinite(true) & (true >= 0.0) & (true <= 1.0), axis=-1) & (
        jnp.abs(jnp.sum(true, axis=-1) - 1.0) <= tolerance
    )
    pred_valid = jnp.all(jnp.isfinite(pred) & (pred >= 0.0) & (pred <= 1.0), axis=-1) & (
        jnp.abs(jnp.sum(pred, axis=-1) - 1.0) <= tolerance
    )
    invalid = invalid | jnp.any(active & ~(true_valid & pred_valid), axis=-1)
    active = active & true_valid & pred_valid
    weights = jnp.where(active, weights, 0.0)
    true = jnp.where(active[..., None], true, 0.0)
    pred = jnp.where(active[..., None], pred, 0.0)
    return true, pred, weights, invalid, jnp.sum(weights, axis=-1)


def smooth_rand_score(
    memberships_true: ArrayLike,
    memberships_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
) -> MetricResult:
    """Expected pairwise agreement under two soft cluster memberships."""
    true, pred, weights, invalid, mass = _membership_pair_inputs(
        memberships_true,
        memberships_pred,
        sample_weight=sample_weight,
        mask=mask,
        metric="smooth_rand_score",
    )
    same_true = true @ jnp.swapaxes(true, -1, -2)
    same_pred = pred @ jnp.swapaxes(pred, -1, -2)
    agreement = same_true * same_pred + (1.0 - same_true) * (1.0 - same_pred)
    pair_weight = weights[..., :, None] * weights[..., None, :]
    pair_weight = jnp.where(~jnp.eye(weights.shape[-1], dtype=bool), pair_weight, 0.0)
    denominator = jnp.sum(pair_weight, axis=(-2, -1))
    value = jnp.sum(pair_weight * agreement, axis=(-2, -1)) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def smooth_normalized_mutual_info_score(
    memberships_true: ArrayLike,
    memberships_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
) -> MetricResult:
    """NMI of the expected contingency table from soft memberships."""
    true, pred, weights, invalid, mass = _membership_pair_inputs(
        memberships_true,
        memberships_pred,
        sample_weight=sample_weight,
        mask=mask,
        metric="smooth_normalized_mutual_info_score",
    )
    contingency = jnp.einsum("...ni,...n,...nj->...ij", true, weights, pred)
    probability = contingency / jnp.where(
        mass[..., None, None] > 0.0, mass[..., None, None], 1.0
    )
    row = jnp.sum(probability, axis=-1)
    column = jnp.sum(probability, axis=-2)
    independent = row[..., :, None] * column[..., None, :]
    mutual_information = jnp.sum(
        jnp.where(
            probability > 0.0,
            probability
            * jnp.log(probability / jnp.where(independent > 0.0, independent, 1.0)),
            0.0,
        ),
        axis=(-2, -1),
    )
    row_entropy = -jnp.sum(jnp.where(row > 0.0, row * jnp.log(row), 0.0), axis=-1)
    column_entropy = -jnp.sum(
        jnp.where(column > 0.0, column * jnp.log(column), 0.0), axis=-1
    )
    denominator = jnp.sqrt(row_entropy * column_entropy)
    value = mutual_information / jnp.where(denominator > 0.0, denominator, 1.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


__all__ = [
    "adjusted_rand_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "normalized_mutual_info_score",
    "silhouette_score",
    "smooth_calinski_harabasz_score",
    "smooth_normalized_mutual_info_score",
    "smooth_rand_score",
    "smooth_silhouette_score",
]
