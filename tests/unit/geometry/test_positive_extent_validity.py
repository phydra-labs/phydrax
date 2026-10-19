#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dynamic analytic extents must remain finite and strictly positive."""

from dataclasses import replace

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.geometry import CompiledGeometry, GeometrySource, ParameterId
from phydrax.geometry._validity import GeometryValidityDisposition
from phydrax.units import METER


def _extent_geometry(family: str) -> tuple[CompiledGeometry, ParameterId]:
    source: GeometrySource
    match family:
        case "orthotope":
            source = phx.geometry.Orthotope(
                (0.0, 0.0, 0.0), (2.0, 3.0, 4.0), feature_id="extent-owner"
            )
            parameter = "size"
        case "ellipse":
            source = phx.geometry.Ellipse(
                (0.0, 0.0), (2.0, 3.0), feature_id="extent-owner"
            )
            parameter = "radii"
        case "ellipsoid":
            source = phx.geometry.Ellipsoid(
                (0.0, 0.0, 0.0), (2.0, 3.0, 4.0), feature_id="extent-owner"
            )
            parameter = "radii"
        case "axis-aligned-ellipsoid":
            source = phx.geometry.AxisAlignedEllipsoid(
                (0.0, 0.0, 0.0, 0.0), (2.0, 3.0, 4.0, 5.0), feature_id="extent-owner"
            )
            parameter = "radii"
        case _:
            raise AssertionError(family)
    return source.compile(coordinate_unit=METER), ParameterId("extent-owner", parameter)


@pytest.mark.parametrize(
    "family", ["orthotope", "ellipse", "ellipsoid", "axis-aligned-ellipsoid"]
)
def test_zero_dynamic_extent_is_invalid(family: str) -> None:
    geometry, parameter = _extent_geometry(family)
    index = geometry.schema.index(parameter)
    zero_extent = (
        geometry.state.values[index].at[0].set(jnp.asarray(0.0, dtype=jnp.float64))
    )
    state = geometry.state.replace_at(index, zero_extent)
    evidence = geometry.validity(state)
    assert bool(evidence.resolved)
    assert not bool(evidence.accepted)
    assert int(evidence.disposition) == int(GeometryValidityDisposition.INVALID)


@pytest.mark.parametrize(
    "family", ["orthotope", "ellipse", "ellipsoid", "axis-aligned-ellipsoid"]
)
def test_positive_extent_margin_uses_declared_scale(family: str) -> None:
    geometry, parameter = _extent_geometry(family)
    index = geometry.schema.index(parameter)
    extents = geometry.state.values[index].at[0].set(jnp.asarray(1.0, dtype=jnp.float64))
    evidence = geometry.validity(geometry.state.replace_at(index, extents))
    assert bool(evidence.accepted)
    margin = evidence.margin_names.index(f"{parameter}:positive:0")
    # Every fixture declares physical scale two; binary arithmetic represents 1 / 2.
    assert float(evidence.margins[margin]) == 0.5


@pytest.mark.parametrize(
    "family", ["orthotope", "ellipse", "ellipsoid", "axis-aligned-ellipsoid"]
)
def test_nonfinite_extent_is_invalid(family: str) -> None:
    geometry, parameter = _extent_geometry(family)
    index = geometry.schema.index(parameter)
    extents = (
        geometry.state.values[index].at[0].set(jnp.asarray(jnp.inf, dtype=jnp.float64))
    )
    evidence = geometry.validity(geometry.state.replace_at(index, extents))
    assert not bool(evidence.finite)
    assert not bool(evidence.accepted)
    assert int(evidence.disposition) == int(GeometryValidityDisposition.INVALID)


@pytest.mark.parametrize(
    "family", ["orthotope", "ellipse", "ellipsoid", "axis-aligned-ellipsoid"]
)
def test_restricted_extent_certificate_remains_inconclusive(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry, _ = _extent_geometry(family)
    restricted = replace(geometry.field_certificate, validity_region="fixed_topology")
    monkeypatch.setattr(
        type(geometry.kernel), "field_certificate", property(lambda _kernel: restricted)
    )
    evidence = geometry.validity()
    assert bool(evidence.finite)
    assert bool(evidence.conditions_satisfied)
    assert not bool(evidence.resolved)
    assert int(evidence.disposition) == int(GeometryValidityDisposition.INCONCLUSIVE)
