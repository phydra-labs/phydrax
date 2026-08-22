#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._polynomial._cubature import CubatureRuleData
from .._strict import StrictModule
from .._trainable import NonTrainableState


_DEFAULT_PATH_BYTES = 64 * 1024**2


class WienerCubaturePathData(StrictModule, NonTrainableState):
    """Prepared weighted controls with independently certified signature degree."""

    increments: Array
    segment_widths: Array
    weights: Array
    noise_dimension: int = eqx.field(static=True)
    path_count: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    gaussian_degree: int = eqx.field(static=True)
    signature_degree: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
    source_rule_id: str = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    path_id: str = eqx.field(static=True)

    def __init__(
        self,
        increments: ArrayLike,
        segment_widths: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        gaussian_degree: int,
        signature_degree: int,
        family: str,
        source_rule_id: str,
        maximum_path_bytes: int = _DEFAULT_PATH_BYTES,
    ):
        increments_host = np.asarray(increments, dtype=float)
        widths_host = np.asarray(segment_widths, dtype=float).reshape((-1,))
        weights_host = np.asarray(weights, dtype=float).reshape((-1,))
        if increments_host.ndim != 3 or not all(increments_host.shape):
            raise ValueError(
                "Wiener cubature increments must have shape "
                "(path_count, segment_count, noise_dimension)."
            )
        path_count, segment_count, noise_dimension = increments_host.shape
        if widths_host.shape != (segment_count,):
            raise ValueError("segment_widths must contain one value per path segment.")
        if weights_host.shape != (path_count,):
            raise ValueError("weights must contain one value per cubature path.")
        order = np.lexsort(
            tuple(
                increments_host.reshape((path_count, -1))[:, axis]
                for axis in range(segment_count * noise_dimension - 1, -1, -1)
            )
        )
        increments_host = np.asarray(increments_host[order])
        weights_host = np.asarray(weights_host[order])
        flattened = increments_host.reshape((path_count, -1))
        if path_count > 1 and np.any(np.all(flattened[1:] == flattened[:-1], axis=1)):
            raise ValueError("Wiener cubature paths must be unique.")
        if (
            np.any(~np.isfinite(increments_host))
            or np.any(~np.isfinite(widths_host))
            or np.any(~np.isfinite(weights_host))
            or np.any(widths_host <= 0.0)
            or np.any(weights_host <= 0.0)
        ):
            raise ValueError("Wiener cubature path data must be finite and positive.")
        tolerance = np.finfo(float).eps * max(path_count, noise_dimension) * 256.0
        if not np.isclose(np.sum(widths_host), 1.0, rtol=tolerance, atol=tolerance):
            raise ValueError("Wiener cubature segment widths must sum to one.")
        if not np.isclose(np.sum(weights_host), 1.0, rtol=tolerance, atol=tolerance):
            raise ValueError("Wiener cubature path weights must sum to one.")
        gaussian_degree_ = _positive_integer(gaussian_degree, "gaussian_degree")
        signature_degree_ = _positive_integer(signature_degree, "signature_degree")
        if signature_degree_ > gaussian_degree_:
            raise ValueError("signature_degree cannot exceed gaussian_degree.")
        if signature_degree_ > 3:
            raise ValueError(
                "Built-in Wiener path certification supports degree at most three."
            )
        storage_bytes = int(
            increments_host.nbytes + widths_host.nbytes + weights_host.nbytes
        )
        maximum = _positive_integer(maximum_path_bytes, "maximum_path_bytes")
        if storage_bytes > maximum:
            raise ValueError("Wiener cubature path data exceeds maximum_path_bytes.")
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(source_rule_id, str)
            or not source_rule_id
        ):
            raise ValueError(
                "Wiener cubature family and source_rule_id must be nonempty."
            )
        _validate_wiener_signatures(
            increments_host,
            widths_host,
            weights_host,
            signature_degree_,
            maximum_workspace_bytes=maximum,
        )
        self.increments = jnp.asarray(increments_host)
        self.segment_widths = jnp.asarray(widths_host)
        self.weights = jnp.asarray(weights_host)
        self.noise_dimension = noise_dimension
        self.path_count = path_count
        self.segment_count = segment_count
        self.gaussian_degree = gaussian_degree_
        self.signature_degree = signature_degree_
        self.family = str(family)
        self.source_rule_id = str(source_rule_id)
        self.storage_bytes = storage_bytes
        self.path_id = canonical_fingerprint(
            {
                "kind": "wiener-cubature-path-v1",
                "noise_dimension": noise_dimension,
                "path_count": path_count,
                "segment_count": segment_count,
                "gaussian_degree": gaussian_degree_,
                "signature_degree": signature_degree_,
                "family": self.family,
                "source_rule_id": self.source_rule_id,
                "storage_bytes": storage_bytes,
                "data": array_tree_fingerprint(
                    (increments_host, widths_host, weights_host)
                ),
            }
        )


def _validation_tolerance(
    increments: np.ndarray,
    path_count: int,
    segment_count: int,
    noise_dimension: int,
    degree: int,
    /,
) -> float:
    magnitude = max(1.0, float(np.max(np.abs(increments)))) ** degree
    return (
        np.finfo(increments.dtype).eps
        * max(path_count, segment_count, noise_dimension)
        * magnitude
        * 512.0
    )


def _centrally_symmetric(
    increments: np.ndarray,
    weights: np.ndarray,
    tolerance: float,
    /,
) -> bool:
    return bool(
        np.allclose(
            increments,
            -increments[::-1],
            rtol=tolerance,
            atol=tolerance,
        )
        and np.allclose(
            weights,
            weights[::-1],
            rtol=tolerance,
            atol=tolerance,
        )
    )


def _third_signature_mean(
    increments: np.ndarray,
    weights: np.ndarray,
    /,
) -> np.ndarray:
    path_count, _, noise_dimension = increments.shape
    first = np.zeros((path_count, noise_dimension), dtype=increments.dtype)
    second = np.zeros(
        (path_count, noise_dimension, noise_dimension), dtype=increments.dtype
    )
    third_mean = np.zeros(
        (noise_dimension, noise_dimension, noise_dimension),
        dtype=increments.dtype,
    )
    for increment in np.moveaxis(increments, 1, 0):
        third_mean += np.einsum(
            "p,pij,pk->ijk", weights, second, increment, optimize=True
        )
        third_mean += 0.5 * np.einsum(
            "p,pi,pj,pk->ijk",
            weights,
            first,
            increment,
            increment,
            optimize=True,
        )
        third_mean += np.einsum(
            "p,pi,pj,pk->ijk",
            weights / 6.0,
            increment,
            increment,
            increment,
            optimize=True,
        )
        second += np.einsum("pi,pj->pij", first, increment)
        second += 0.5 * np.einsum("pi,pj->pij", increment, increment)
        first += increment
    return third_mean


def _validate_wiener_signatures(
    increments: np.ndarray,
    segment_widths: np.ndarray,
    weights: np.ndarray,
    signature_degree: int,
    /,
    *,
    maximum_workspace_bytes: int,
) -> None:
    path_count, segment_count, noise_dimension = increments.shape
    first = np.zeros((path_count, noise_dimension), dtype=increments.dtype)
    second_mean = np.zeros((noise_dimension, noise_dimension), dtype=increments.dtype)
    time_spatial_mean = np.zeros((noise_dimension,), dtype=increments.dtype)
    spatial_time_mean = np.zeros((noise_dimension,), dtype=increments.dtype)
    elapsed = 0.0
    for segment, segment_width in enumerate(segment_widths):
        increment = increments[:, segment, :]
        second_mean += np.einsum("p,pi,pj->ij", weights, first, increment, optimize=True)
        second_mean += 0.5 * np.einsum(
            "p,pi,pj->ij", weights, increment, increment, optimize=True
        )
        time_spatial_mean += weights @ ((elapsed + 0.5 * segment_width) * increment)
        spatial_time_mean += weights @ (
            segment_width * first + 0.5 * segment_width * increment
        )
        first += increment
        elapsed += float(segment_width)
    first_mean = weights @ first
    first_tolerance = _validation_tolerance(
        increments, path_count, segment_count, noise_dimension, 1
    )
    if not np.allclose(first_mean, 0.0, rtol=first_tolerance, atol=first_tolerance):
        raise ValueError("Wiener cubature paths do not match the first signature level.")
    if signature_degree < 2:
        return
    second_tolerance = _validation_tolerance(
        increments, path_count, segment_count, noise_dimension, 2
    )
    if not np.allclose(
        second_mean,
        0.5 * np.eye(noise_dimension),
        rtol=second_tolerance,
        atol=second_tolerance,
    ):
        raise ValueError("Wiener cubature paths do not match the second signature level.")
    if signature_degree < 3:
        return
    third_tolerance = _validation_tolerance(
        increments, path_count, segment_count, noise_dimension, 3
    )
    if not np.allclose(
        time_spatial_mean,
        0.0,
        rtol=third_tolerance,
        atol=third_tolerance,
    ) or not np.allclose(
        spatial_time_mean,
        0.0,
        rtol=third_tolerance,
        atol=third_tolerance,
    ):
        raise ValueError(
            "Wiener cubature paths do not match degree-three time-space signatures."
        )
    if _centrally_symmetric(increments, weights, third_tolerance):
        return
    workspace_bytes = (
        path_count * noise_dimension**2 + noise_dimension**3
    ) * increments.dtype.itemsize
    if workspace_bytes > maximum_workspace_bytes:
        raise ValueError(
            "Degree-three signature certification exceeds maximum_path_bytes."
        )
    third_mean = _third_signature_mean(increments, weights)
    if not np.allclose(third_mean, 0.0, rtol=third_tolerance, atol=third_tolerance):
        raise ValueError("Wiener cubature paths do not match the third signature level.")


def straight_wiener_cubature_path(
    rule: CubatureRuleData,
    /,
    *,
    maximum_path_bytes: int = _DEFAULT_PATH_BYTES,
) -> WienerCubaturePathData:
    """Lift a degree-three Gaussian formula to straight unit-time controls."""
    if not isinstance(rule, CubatureRuleData):
        raise TypeError("rule must be CubatureRuleData.")
    if rule.reference_domain != "standard-normal" or rule.exact_degree < 3:
        raise ValueError("Straight Wiener cubature requires degree-three Gaussian data.")
    points = np.asarray(rule.points, dtype=float)
    weights = np.asarray(rule.weights, dtype=float)
    return WienerCubaturePathData(
        points[:, None, :],
        np.asarray([1.0]),
        weights,
        gaussian_degree=rule.exact_degree,
        signature_degree=3,
        family="straight-gaussian",
        source_rule_id=rule.rule_id,
        maximum_path_bytes=maximum_path_bytes,
    )


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = ["WienerCubaturePathData", "straight_wiener_cubature_path"]
