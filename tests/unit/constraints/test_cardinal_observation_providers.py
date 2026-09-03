#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import coordax as cx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, PointBatch, SampleLayout
from phydrax.enforcement._cardinal import (
    CompactCardinalCorrectionProvider,
    IDWCardinalCorrectionProvider,
)
from phydrax.enforcement._observation import PointObservationAction


def _point_batch(domain: Interval1d, coordinates) -> PointBatch:
    structure = SampleLayout((("x",),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    points = jnp.asarray(coordinates, dtype=float).reshape((-1, 1))
    return PointBatch(
        points=frozendict({"x": cx.Field(points, dims=(axis_names[0], None))}),
        structure=structure,
    )


def test_point_observation_returns_exact_finite_values_and_selected_components():
    domain = Interval1d(-1.0, 2.0)
    batch = _point_batch(domain, [-1.0, 0.25, 2.0])

    @domain.Function("x")
    def field(x):
        return jnp.asarray((2.0 * x[0] - 1.0, x[0] ** 2 + 3.0, -x[0]))

    action = PointObservationAction("u", batch, components=(2, 0))
    observed = action.apply({"u": field})

    np.testing.assert_allclose(
        observed,
        np.asarray(((1.0, -3.0), (-0.25, -0.5), (-2.0, 3.0))),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert action.observation_count == 3
    assert action.evidence.components == (2, 0)
    assert action.evidence.exact_scope == "finite_restriction"


@pytest.mark.parametrize(
    "provider_type",
    (IDWCardinalCorrectionProvider, CompactCardinalCorrectionProvider),
)
def test_cardinal_provider_is_identity_on_every_anchor(provider_type):
    domain = Interval1d(0.0, 1.0)
    anchors = _point_batch(domain, [0.0, 0.3, 1.0])
    observation = PointObservationAction("u", anchors)
    provider = provider_type(
        observation,
        domain,
        support_radius=0.45,
        snap_tolerance_squared=0.0,
    )
    residual = jnp.asarray([1.5, -2.0, 4.25])

    correction = provider.candidate_action().lift(residual)[0]

    np.testing.assert_allclose(
        correction(anchors).data,
        residual,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert provider.evidence.restriction_scope == "exact_finite_observations"
    assert not provider.evidence.interpolation_exact_off_support


def test_cardinal_lift_scatter_preserves_unselected_output_components():
    domain = Interval1d(0.0, 1.0)
    anchors = _point_batch(domain, [0.0, 0.5, 1.0])
    observation = PointObservationAction("velocity", anchors, components=(2, 0))
    provider = IDWCardinalCorrectionProvider(observation, domain)
    residual = jnp.asarray(
        [
            [10.0, -1.0],
            [20.0, -2.0],
            [30.0, -3.0],
        ]
    )

    correction = provider.candidate_action(output_width=4).lift(residual)[0]

    np.testing.assert_allclose(
        correction(anchors).data,
        np.asarray(
            [
                [-1.0, 0.0, 10.0, 0.0],
                [-2.0, 0.0, 20.0, 0.0],
                [-3.0, 0.0, 30.0, 0.0],
            ]
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "provider_type",
    (IDWCardinalCorrectionProvider, CompactCardinalCorrectionProvider),
)
def test_cardinal_providers_reject_duplicate_anchors(provider_type):
    domain = Interval1d(0.0, 1.0)
    duplicate_anchors = _point_batch(domain, [0.0, 0.5, 0.5])
    observation = PointObservationAction("u", duplicate_anchors)

    with pytest.raises(ValueError, match="pairwise distinct"):
        provider_type(observation, domain)


def test_idw_source_envelope_attenuates_only_the_enabled_source():
    domain = Interval1d(0.0, 2.0)
    anchors = _point_batch(domain, [0.0, 2.0])
    query = _point_batch(domain, [1.0])
    observation = PointObservationAction("u", anchors)
    provider = IDWCardinalCorrectionProvider(
        observation,
        domain,
        source_index=jnp.asarray([0, 1]),
        envelope_enabled=(False, True),
        envelope_scale=jnp.asarray([1.0, 0.5]),
        snap_tolerance_squared=0.0,
    )

    correction = provider.candidate_action().lift(jnp.asarray([0.0, 2.0]))[0]

    np.testing.assert_allclose(
        correction(query).data,
        np.asarray([np.exp(-4.0)]),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert provider.evidence.uses_envelopes
    assert provider.evidence.extension_scope == "global_idw"
    assert not provider.evidence.local_support
