#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.iga import BSplineGrid
from phydrax.discretization.iga._basis import TensorSplineBasisSpec
from phydrax.discretization.iga._identity import BaseSpanId
from phydrax.discretization.iga._overlay import IntegrationOverlay
from phydrax.discretization.iga._realization import (
    DirectTensorRealization,
    ExtractedBernsteinRealization,
)
from phydrax.discretization.iga._topology import PatchAtlas, SplineSpanTopology


def _basis():
    grid = BSplineGrid.open_uniform(1, 2)
    return TensorSplineBasisSpec((grid,), axis_names=("xi",))


def test_positive_span_topology_is_not_control_topology():
    basis = _basis()
    topology = SplineSpanTopology(basis, patch_id="p")

    assert topology.axis_sizes == basis.span_shape
    assert topology.axis_sizes != basis.control_shape
    assert topology.span_id(0) == BaseSpanId("p", (0,))


def test_direct_and_bernstein_realization_use_exact_transposes():
    basis = _basis()
    direct = DirectTensorRealization(basis, SplineSpanTopology(basis))
    coefficients = jnp.arange(
        basis.coefficient_count * 2,
        dtype=float,
    ).reshape(basis.coefficient_count, 2)
    local = direct.gather(coefficients)
    local_dual = jnp.arange(local.size, dtype=float).reshape(local.shape)
    np.testing.assert_allclose(
        jnp.vdot(local, local_dual),
        jnp.vdot(coefficients, direct.gather_transpose(local_dual)),
    )

    base_extraction = jnp.asarray([[2.0, 1.0], [0.5, 1.5]])
    extraction = jnp.broadcast_to(
        base_extraction,
        (direct.cell_count, direct.local_width, direct.local_width),
    )
    bernstein = ExtractedBernsteinRealization(direct, extraction)

    expanded_extraction = extraction[..., None]
    expected_realized = jnp.sum(
        expanded_extraction * local[:, None, :, :],
        axis=2,
    )
    realized = bernstein.realize(coefficients)
    np.testing.assert_allclose(realized, expected_realized)
    np.testing.assert_allclose(
        jax.jit(lambda x: bernstein.realize(x))(coefficients),
        expected_realized,
    )

    dual = jnp.arange(realized.size, dtype=float).reshape(realized.shape) / 3.0
    expected_local_dual = jnp.sum(
        expanded_extraction * dual[:, :, None, :],
        axis=1,
    )
    expected_transpose = direct.gather_transpose(expected_local_dual)
    transpose = bernstein.transpose(dual)
    np.testing.assert_allclose(transpose, expected_transpose)
    np.testing.assert_allclose(
        jax.jit(lambda x: bernstein.transpose(x))(dual),
        expected_transpose,
    )
    np.testing.assert_allclose(
        jnp.vdot(realized, dual),
        jnp.vdot(coefficients, transpose),
    )


def test_overlay_fails_closed_when_a_patch_is_not_covered():
    basis = _basis()
    topology = SplineSpanTopology(basis, patch_id="p")
    with pytest.raises(ValueError, match="does not cover"):
        IntegrationOverlay(PatchAtlas((topology,)), ((topology.span_id(0),),))
