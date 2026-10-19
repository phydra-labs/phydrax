"""Coverage assertions reject missing volume and missing parameter intervals."""

from dataclasses import replace

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.meshing._implicit_volume import (
    isolate_implicit_dual_roots,
    prepare_implicit_volume_domain,
)
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.units import METER
from tests._support.implicit_meshing import (
    assert_box_coverage,
    assert_terminal_cover,
    execution_scope as execution_scope,
    implicit_query,
    implicit_specification,
    NATIVE,
)


@NATIVE
def test_box_coverage_assertion_refuses_a_missing_tetrahedron(
    execution_scope: NativeExecutionScope,
) -> None:
    work = prepare_implicit_volume_domain(
        implicit_query(
            phx.geometry.Sphere((0.0, 0.0, 0.0), 3.0).compile(coordinate_unit=METER)
        ),
        implicit_specification(),
        NativeVolumeSchedule(refinement_rounds=1),
        scope=execution_scope,
    )
    assert_box_coverage(work)
    incomplete = replace(work, tetrahedra=work.tetrahedra[:-1])
    with pytest.raises(AssertionError):
        assert_box_coverage(incomplete)


def test_terminal_coverage_assertion_refuses_a_missing_initial_interval() -> None:
    work = isolate_implicit_dual_roots(
        implicit_query(
            phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
        ),
        np.asarray((((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float64),
        spatial_tolerance=0.1,
        maximum_depth=8,
        maximum_geometry_queries=1000,
    )
    assert_terminal_cover(work)
    intervals = work.parameter_intervals.copy()
    intervals[0, 0] += 0.5 * (intervals[0, 1] - intervals[0, 0])
    incomplete = replace(work, parameter_intervals=intervals)
    with pytest.raises(AssertionError):
        assert_terminal_cover(incomplete)
