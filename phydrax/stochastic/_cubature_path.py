#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.optimize import least_squares

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._polynomial._cubature import CubatureRuleData
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._signature import piecewise_linear_signature


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
    signature_residuals: Array
    certification_precision: str = eqx.field(static=True)

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
        signature_residuals = _validate_wiener_signatures(
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
        self.signature_residuals = jnp.asarray(signature_residuals)
        self.certification_precision = str(increments_host.dtype)
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


def _expected_wiener_signature_level(
    noise_dimension: int,
    degree: int,
    dtype: np.dtype,
    /,
) -> np.ndarray:
    driver_dimension = noise_dimension + 1
    shape = (driver_dimension,) * degree
    # E[S(t, B)] = exp(e_0 + 1/2 sum_i e_i e_i) at unit time.
    inverse_factorials = tuple(
        1.0 / float(factorial(count)) for count in range(degree + 1)
    )
    brownian_pair_weights = tuple(0.5**count for count in range(degree // 2 + 1))

    def coefficient(word):
        generator_factors = 0
        brownian_pairs = 0
        position = 0
        while position < degree:
            index = word[position]
            if index == 0:
                generator_factors += 1
                position += 1
            elif position + 1 < degree and word[position + 1] == index:
                generator_factors += 1
                brownian_pairs += 1
                position += 2
            else:
                return 0.0
        return (
            brownian_pair_weights[brownian_pairs] * inverse_factorials[generator_factors]
        )

    return np.fromiter(
        (coefficient(word) for word in np.ndindex(shape)),
        dtype=dtype,
        count=driver_dimension**degree,
    ).reshape(shape)


def _time_augmented_increments(
    increments: np.ndarray,
    segment_widths: np.ndarray,
    /,
) -> np.ndarray:
    time_increments = np.broadcast_to(
        segment_widths[None, :, None],
        increments.shape[:2] + (1,),
    )
    return np.concatenate((time_increments, increments), axis=-1)


def _signature_level_name(degree: int, /) -> str:
    names = ("first", "second", "third", "fourth", "fifth")
    return names[degree - 1] if degree <= len(names) else f"degree-{degree}"


def _signature_grades(
    driver_dimension: int,
    tensor_level: int,
    /,
) -> np.ndarray:
    shape = (driver_dimension,) * tensor_level
    return np.fromiter(
        (tensor_level + word.count(0) for word in np.ndindex(shape)),
        dtype=np.int32,
        count=driver_dimension**tensor_level,
    ).reshape(shape)


def _validate_wiener_signatures(
    increments: np.ndarray,
    segment_widths: np.ndarray,
    weights: np.ndarray,
    signature_degree: int,
    /,
    *,
    maximum_workspace_bytes: int,
) -> np.ndarray:
    path_count, segment_count, noise_dimension = increments.shape
    driver_dimension = noise_dimension + 1
    term_count = path_count * segment_count * driver_dimension + sum(
        path_count * driver_dimension**degree + driver_dimension**degree
        for degree in range(1, signature_degree + 1)
    )
    workspace_bytes = term_count * increments.dtype.itemsize
    if workspace_bytes > maximum_workspace_bytes:
        raise ValueError("Wiener signature certification exceeds maximum_path_bytes.")
    driver_increments = _time_augmented_increments(increments, segment_widths)
    signature = tuple(
        np.asarray(level)
        for level in piecewise_linear_signature(
            jnp.asarray(driver_increments),
            signature_degree,
        )
    )
    residuals = np.zeros((signature_degree,), dtype=increments.dtype)
    for tensor_level, level in enumerate(signature, start=1):
        represented = np.tensordot(weights, level, axes=(0, 0))
        target = _expected_wiener_signature_level(
            noise_dimension,
            tensor_level,
            increments.dtype,
        )
        tolerance = _validation_tolerance(
            driver_increments,
            path_count,
            segment_count,
            driver_dimension,
            tensor_level,
        )
        grades = _signature_grades(driver_dimension, tensor_level)
        for homogeneous_degree in range(
            tensor_level,
            min(2 * tensor_level, signature_degree) + 1,
        ):
            selected = grades == homogeneous_degree
            difference = np.abs(represented[selected] - target[selected])
            residuals[homogeneous_degree - 1] = max(
                residuals[homogeneous_degree - 1],
                float(np.max(difference)),
            )
            if np.allclose(
                represented[selected],
                target[selected],
                rtol=tolerance,
                atol=tolerance,
            ):
                continue
            level_name = _signature_level_name(homogeneous_degree)
            if homogeneous_degree == tensor_level:
                raise ValueError(
                    "Wiener cubature paths do not match the "
                    f"{level_name} signature level."
                )
            raise ValueError(
                "Wiener cubature paths do not match time-space signatures at the "
                f"{level_name} signature level."
            )
    return residuals


def straight_wiener_cubature_path(
    rule: CubatureRuleData,
    /,
    *,
    maximum_path_bytes: int = _DEFAULT_PATH_BYTES,
) -> WienerCubaturePathData:
    """Lift a finite-degree Gaussian formula to certified straight controls."""
    if not isinstance(rule, CubatureRuleData):
        raise TypeError("rule must be CubatureRuleData.")
    if rule.reference_domain != "standard-normal" or rule.exact_degree < 1:
        raise ValueError("Straight Wiener cubature requires standard-normal data.")
    points = np.asarray(rule.points, dtype=float)
    weights = np.asarray(rule.weights, dtype=float)
    return WienerCubaturePathData(
        points[:, None, :],
        np.asarray([1.0]),
        weights,
        gaussian_degree=rule.exact_degree,
        signature_degree=rule.exact_degree,
        family="straight-gaussian",
        source_rule_id=rule.rule_id,
        maximum_path_bytes=maximum_path_bytes,
    )


def fit_wiener_cubature_path(
    noise_dimension: int,
    signature_degree: int,
    /,
    *,
    path_count: int,
    segment_count: int,
    initial_data: WienerCubaturePathData | ArrayLike,
    optimizer=None,
    maximum_signature_terms: int = 1_000_000,
    maximum_workspace_bytes: int = _DEFAULT_PATH_BYTES,
) -> WienerCubaturePathData:
    """Fit one bounded positive finite path formula on the host.

    The solve is nonconvex and may fail.  A result is returned only after the normal
    constructor independently certifies every requested signature level; there is no
    degree downgrade or negative-weight fallback.  A custom optimizer receives
    ``(residual_function, initial_vector)`` and must return the final vector.
    """
    dimension = _positive_integer(noise_dimension, "noise_dimension")
    degree = _positive_integer(signature_degree, "signature_degree")
    paths = _positive_integer(path_count, "path_count")
    segments = _positive_integer(segment_count, "segment_count")
    term_cap = _positive_integer(maximum_signature_terms, "maximum_signature_terms")
    byte_cap = _positive_integer(maximum_workspace_bytes, "maximum_workspace_bytes")
    driver_dimension = dimension + 1
    signature_terms = sum(driver_dimension**level for level in range(1, degree + 1))
    if signature_terms > term_cap:
        raise ValueError("Requested signature degree exceeds maximum_signature_terms.")
    workspace = (
        paths * segments * driver_dimension + paths * signature_terms + signature_terms
    ) * np.dtype(float).itemsize
    if workspace > byte_cap:
        raise ValueError("Wiener cubature fitting exceeds maximum_workspace_bytes.")
    if isinstance(initial_data, WienerCubaturePathData):
        initial_increments = np.asarray(initial_data.increments, dtype=float)
        initial_weights = np.asarray(initial_data.weights, dtype=float)
        widths = np.asarray(initial_data.segment_widths, dtype=float)
    else:
        initial_increments = np.asarray(initial_data, dtype=float)
        initial_weights = np.full((paths,), 1.0 / paths)
        widths = np.full((segments,), 1.0 / segments)
    expected_shape = (paths, segments, dimension)
    if initial_increments.shape != expected_shape:
        raise ValueError(f"initial_data increments must have shape {expected_shape}.")
    if initial_weights.shape != (paths,) or np.any(initial_weights <= 0.0):
        raise ValueError("Initial path weights must be positive and path-aligned.")
    initial_logits = np.log(initial_weights)
    initial_vector = np.concatenate((initial_increments.reshape((-1,)), initial_logits))

    def unpack(vector):
        split = paths * segments * dimension
        increments = np.asarray(vector[:split]).reshape(expected_shape)
        logits = np.asarray(vector[split:])
        shifted = logits - np.max(logits)
        weights = np.exp(shifted)
        return increments, weights / np.sum(weights)

    def residual(vector):
        increments, weights = unpack(vector)
        driver_increments = _time_augmented_increments(increments, widths)
        levels = tuple(
            np.asarray(level)
            for level in piecewise_linear_signature(
                jnp.asarray(driver_increments),
                degree,
            )
        )
        parts = []
        for tensor_level, level in enumerate(levels, start=1):
            represented = np.tensordot(weights, level, axes=(0, 0))
            target = _expected_wiener_signature_level(
                dimension,
                tensor_level,
                increments.dtype,
            )
            grades = _signature_grades(driver_dimension, tensor_level)
            certified = grades <= degree
            parts.append((represented - target)[certified])
        return np.concatenate(parts)

    if optimizer is None:
        vector = least_squares(
            residual,
            initial_vector,
            max_nfev=max(200, 20 * initial_vector.size),
        ).x
    else:
        if not callable(optimizer):
            raise TypeError("optimizer must be callable or None.")
        vector = np.asarray(optimizer(residual, initial_vector), dtype=float)
        if vector.shape != initial_vector.shape:
            raise ValueError("Custom optimizer returned an incompatible vector.")
    increments, weights = unpack(vector)
    source_id = canonical_fingerprint(
        {
            "kind": "fitted-wiener-cubature-path-v1",
            "noise_dimension": dimension,
            "signature_degree": degree,
            "path_count": paths,
            "segment_count": segments,
        }
    )
    return WienerCubaturePathData(
        increments,
        widths,
        weights,
        gaussian_degree=degree,
        signature_degree=degree,
        family="fitted-positive-signature",
        source_rule_id=source_id,
        maximum_path_bytes=byte_cap,
    )


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = [
    "WienerCubaturePathData",
    "fit_wiener_cubature_path",
    "straight_wiener_cubature_path",
]
