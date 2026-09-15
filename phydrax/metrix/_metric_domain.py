#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chart import CoordinateChart


class MetricDomainStatus(IntEnum):
    """Generic per-lane chart-domain classification."""

    VALID = 0
    NEAR_BOUNDARY = 1
    OUTSIDE = 2
    NONFINITE = 3
    REJECTED = 4


class MetricDomainEvidence(StrictModule, NonTrainableState):
    """Fixed-shape JAX evidence for membership in one charted metric domain.

    ``margin`` is a signed provider-defined domain margin: positive values are
    inside, zero is on the boundary, and negative values are outside. Providers
    retain authority over ``valid`` and ``status`` so additional regularity or
    parameter predicates are not erased by a generic repair policy.
    """

    chart: CoordinateChart = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    valid: Array
    near_boundary: Array
    margin: Array
    status: Array

    def __init__(
        self,
        valid: ArrayLike,
        near_boundary: ArrayLike,
        margin: ArrayLike,
        status: ArrayLike,
        /,
        *,
        chart: CoordinateChart,
        domain_id: str,
    ):
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if (
            not isinstance(domain_id, str)
            or not domain_id
            or domain_id.strip() != domain_id
        ):
            raise ValueError("domain_id must be a non-empty stripped string.")
        valid_array = jnp.asarray(valid, dtype=bool)
        near_array = jnp.asarray(near_boundary, dtype=bool)
        margin_array = jnp.asarray(margin)
        status_value = jnp.asarray(status)
        if not jnp.issubdtype(status_value.dtype, jnp.integer):
            raise TypeError("Metric domain status must have an integer dtype.")
        status_array = status_value.astype(jnp.int32)
        if not jnp.issubdtype(margin_array.dtype, jnp.floating):
            raise TypeError("Metric domain margin must have a real floating-point dtype.")
        shape = margin_array.shape
        for name, value in (
            ("valid", valid_array),
            ("near_boundary", near_array),
            ("status", status_array),
        ):
            if value.shape != shape:
                raise ValueError(
                    f"Metric domain {name} must have shape {shape}; got {value.shape}."
                )
        self.chart = chart
        self.domain_id = domain_id
        self.valid = valid_array
        self.near_boundary = near_array
        self.margin = margin_array
        self.status = status_array

    @property
    def finite(self) -> Array:
        """Whether each lane's signed domain margin is finite."""
        return jnp.isfinite(self.margin)

    @property
    def inside(self) -> Array:
        """Whether each lane has non-negative signed membership margin."""
        return self.finite & (self.margin >= 0.0)

    @property
    def physically_valid(self) -> Array:
        """Whether each lane is finite, inside, and provider-valid."""
        return self.finite & self.inside & self.valid

    @property
    def qualified(self) -> Array:
        """Whether each lane has the generic success status as well as validity."""
        return self.physically_valid & (self.status == int(MetricDomainStatus.VALID))

    @property
    def derivative_valid(self) -> Array:
        """Conservative smooth-interior derivative admission evidence."""
        return self.qualified & ~self.near_boundary

    @classmethod
    def from_margin(
        cls,
        margin: ArrayLike,
        /,
        *,
        chart: CoordinateChart,
        domain_id: str,
        boundary_tolerance: ArrayLike = 0.0,
        extra_valid: ArrayLike = True,
    ) -> MetricDomainEvidence:
        """Classify a provider-defined signed margin without clipping it."""
        margin_array = jnp.asarray(margin)
        if not jnp.issubdtype(margin_array.dtype, jnp.floating):
            raise TypeError("Metric domain margin must have a real floating-point dtype.")
        tolerance = jnp.asarray(boundary_tolerance, dtype=margin_array.dtype)
        if tolerance.shape != ():
            raise ValueError("boundary_tolerance must be scalar.")
        finite = jnp.isfinite(margin_array) & jnp.isfinite(tolerance)
        tolerance_valid = tolerance >= 0.0
        inside = margin_array >= 0.0
        provider_valid = jnp.broadcast_to(
            jnp.asarray(extra_valid, dtype=bool),
            margin_array.shape,
        )
        valid = finite & tolerance_valid & inside & provider_valid
        near = valid & (margin_array <= tolerance)
        status = jnp.where(
            ~finite,
            int(MetricDomainStatus.NONFINITE),
            jnp.where(
                ~inside,
                int(MetricDomainStatus.OUTSIDE),
                jnp.where(
                    ~provider_valid | ~tolerance_valid,
                    int(MetricDomainStatus.REJECTED),
                    jnp.where(
                        near,
                        int(MetricDomainStatus.NEAR_BOUNDARY),
                        int(MetricDomainStatus.VALID),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return cls(
            valid,
            near,
            margin_array,
            status,
            chart=chart,
            domain_id=domain_id,
        )


__all__ = ["MetricDomainEvidence", "MetricDomainStatus"]
