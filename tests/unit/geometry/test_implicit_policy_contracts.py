# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Scalar-kind and count contracts of implicit preparation policies."""

import numpy as np
import pytest

from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    ImplicitProjectionPolicy,
    ImplicitSurfacePolicy,
)


def test_projection_step_capacity_refuses_a_boolean_count() -> None:
    with pytest.raises(TypeError, match="ImplicitProjectionPolicy.maximum_steps"):
        ImplicitProjectionPolicy(maximum_steps=True)


def test_dense_crossing_capacity_refuses_a_boolean_count() -> None:
    with pytest.raises(TypeError, match="ImplicitSurfacePolicy.maximum_crossings"):
        ImplicitSurfacePolicy(maximum_crossings=True)


def test_adaptive_level_refuses_a_nonintegral_scalar() -> None:
    with pytest.raises(TypeError, match="AdaptiveImplicitSurfacePolicy.initial_level"):
        AdaptiveImplicitSurfacePolicy(initial_level=2.5)  # ty: ignore[invalid-argument-type]


def test_projection_tolerance_refuses_a_boolean_real_value() -> None:
    with pytest.raises(TypeError, match="ImplicitProjectionPolicy.root_tolerance"):
        ImplicitProjectionPolicy(root_tolerance=True)


def test_integral_host_capacity_preserves_its_declared_metadata_value() -> None:
    capacity = np.int64(17)
    policy = ImplicitSurfacePolicy(maximum_crossings=capacity)  # ty: ignore[invalid-argument-type]
    assert policy.maximum_crossings is capacity
