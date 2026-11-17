#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np

from phydrax.geometry.implicit._source_charts import _root_displacement_upper
from tests._support.implicit_layer_curving_fixtures import (
    make_implicit_layer_curving_archive_case,
)


def test_displacement_upper_encloses_exact_binary64_distance() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    cases = (
        ((1.0e16, 1.0e16, 1.0e16), (-1.0, -1.0, -1.0), 0.0),
        ((0.0, -0.0, tiny), (tiny, 0.0, -tiny), tiny),
        ((0.5, -0.7, 0.3), (-0.8, 0.2, 0.9), 1.0e-10),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
    )
    roots = np.asarray([row[0] for row in cases])
    targets = np.asarray([row[1] for row in cases])
    errors = np.asarray([row[2] for row in cases])
    bounds = _root_displacement_upper(roots, targets, errors)
    for (root, target, error), bound in zip(cases, bounds, strict=True):
        remaining = Fraction.from_float(float(bound)) - Fraction.from_float(error)
        exact_squared = sum(
            (Fraction.from_float(a) - Fraction.from_float(b)) ** 2
            for a, b in zip(root, target, strict=True)
        )
        assert remaining >= 0
        assert remaining * remaining >= exact_squared


def test_legacy_norm_loses_exact_coordinate_differences() -> None:
    root = np.full((1, 3), 1.0e16)
    target = np.full((1, 3), -1.0)
    bound = float(_root_displacement_upper(root, target, np.zeros(1))[0])
    exact_squared = 3 * (10_000_000_000_000_001**2)
    assert Fraction.from_float(bound) ** 2 >= exact_squared


def test_curving_trace_displacements_bound_exact_target_distances() -> None:
    result, _, _, _ = make_implicit_layer_curving_archive_case()
    evidence = result.source_evidence
    assert evidence is not None and evidence.implicit_root_trace is not None
    trace = evidence.implicit_root_trace
    for root, target, error, displacement in zip(
        trace.source_points,
        trace.target_points,
        trace.root_error,
        trace.displacement_upper,
        strict=True,
    ):
        remaining = Fraction.from_float(float(displacement)) - Fraction.from_float(
            float(error)
        )
        exact_squared = sum(
            (Fraction.from_float(float(a)) - Fraction.from_float(float(b))) ** 2
            for a, b in zip(root, target, strict=True)
        )
        assert remaining >= 0
        assert remaining * remaining >= exact_squared
