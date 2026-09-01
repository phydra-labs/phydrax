#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
    coefficients = jnp.arange(basis.coefficient_count, dtype=float)
    local = direct.gather(coefficients)
    dual = jnp.arange(local.size, dtype=float).reshape(local.shape)
    np.testing.assert_allclose(
        jnp.vdot(local, dual), jnp.vdot(coefficients, direct.gather_transpose(dual))
    )

    extraction = jnp.broadcast_to(
        jnp.eye(direct.local_width),
        (direct.cell_count, direct.local_width, direct.local_width),
    )
    bernstein = ExtractedBernsteinRealization(direct, extraction)
    np.testing.assert_allclose(bernstein.realize(coefficients), local)
    np.testing.assert_allclose(bernstein.transpose(dual), direct.gather_transpose(dual))


def test_overlay_fails_closed_when_a_patch_is_not_covered():
    basis = _basis()
    topology = SplineSpanTopology(basis, patch_id="p")
    with pytest.raises(ValueError, match="does not cover"):
        IntegrationOverlay(PatchAtlas((topology,)), ((topology.span_id(0),),))
