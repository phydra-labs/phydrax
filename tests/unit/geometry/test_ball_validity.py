#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ball validity follows finite coordinates and a strictly positive radius.

A radius of one with declared physical scale two has dimensionless margin
one half. This ratio is representable in binary and requires no tolerance.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._validity import GeometryValidityDisposition
from phydrax.units import METER


@pytest.mark.parametrize(
    "dimension", [2, 3, 4], ids=["circle", "sphere", "four-dimensional-ball"]
)
def test_zero_ball_radius_is_invalid(dimension: int) -> None:
    geometry = phx.geometry.Ball(
        np.zeros((dimension,), dtype=np.float64), 2.0, feature_id="validity-ball"
    ).compile(coordinate_unit=METER)
    radius = phx.geometry.ParameterId("validity-ball", "radius")
    state = geometry.state.replace_at(
        geometry.schema.index(radius), jnp.asarray(0.0, dtype=jnp.float64)
    )
    evidence = geometry.validity(state)
    assert bool(evidence.resolved)
    assert not bool(evidence.accepted)
    assert int(evidence.disposition) == int(GeometryValidityDisposition.INVALID)


def test_positive_ball_radius_margin_uses_declared_scale() -> None:
    geometry = phx.geometry.Ball(
        np.zeros((3,), dtype=np.float64), 2.0, feature_id="validity-ball"
    ).compile(coordinate_unit=METER)
    radius = phx.geometry.ParameterId("validity-ball", "radius")
    state = geometry.state.replace_at(
        geometry.schema.index(radius), jnp.asarray(1.0, dtype=jnp.float64)
    )
    evidence = geometry.validity(state)
    assert bool(evidence.accepted)
    margin = evidence.margin_names.index(f"{radius}:positive")
    assert float(evidence.margins[margin]) == 0.5


def test_ball_validity_rejects_nonfinite_center() -> None:
    geometry = phx.geometry.Ball(
        np.zeros((3,), dtype=np.float64), 2.0, feature_id="validity-ball"
    ).compile(coordinate_unit=METER)
    center = phx.geometry.ParameterId("validity-ball", "center")
    state = geometry.state.replace_at(
        geometry.schema.index(center), jnp.asarray([jnp.nan, 0.0, 0.0], dtype=jnp.float64)
    )
    evidence = geometry.validity(state)
    assert not bool(evidence.finite)
    assert not bool(evidence.accepted)


def test_restricted_ball_certificate_remains_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = phx.geometry.Ball(
        np.zeros((3,), dtype=np.float64), 2.0, feature_id="validity-ball"
    ).compile(coordinate_unit=METER)
    restricted = replace(geometry.field_certificate, validity_region="fixed_topology")
    monkeypatch.setattr(
        type(geometry.kernel), "field_certificate", property(lambda _kernel: restricted)
    )
    evidence = geometry.validity()
    assert bool(evidence.finite)
    assert bool(evidence.conditions_satisfied)
    assert not bool(evidence.resolved)
    assert int(evidence.disposition) == int(GeometryValidityDisposition.INCONCLUSIVE)
