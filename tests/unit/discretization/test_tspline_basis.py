#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace
from fractions import Fraction
from math import comb

import numpy as np
import pytest

from phydrax._interpolation._tspline import (
    ExtractedBernstein,
    LocalKnotVector2D,
    TAnchor2D,
    TMesh2D,
)


def test_topology_derived_knots_have_exact_partition_and_c2_traces() -> None:
    mesh = TMesh2D.tensor_product(
        (Fraction(0), Fraction(1, 3), Fraction(1)),
        (Fraction(0), Fraction(2, 5), Fraction(1)),
    )
    certificate = mesh.certify()

    assert certificate.passed
    assert certificate.dual_compatible
    assert certificate.partition_of_unity
    assert certificate.trace_continuous
    assert certificate.continuity_order == 2
    assert all(
        isinstance(knot, Fraction)
        for anchor in mesh.anchors
        for knot in anchor.local_knots.u
    )
    assert all(
        isinstance(knot, Fraction)
        for anchor in mesh.anchors
        for knot in anchor.local_knots.v
    )

    realization = mesh.realize()
    for extraction in realization.extractions:
        exact = extraction.exact_extraction_operator
        assert all(
            sum((row[column] for row in exact), Fraction(0)) == 1 for column in range(16)
        )
        assert all(value >= 0 for row in exact for value in row)


def test_persistent_anchor_ids_are_not_coefficient_rows() -> None:
    knots = LocalKnotVector2D(
        (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    anchor = TAnchor2D(91, (Fraction(0), Fraction(0)), knots)
    mesh = TMesh2D(((Fraction(0), Fraction(1), Fraction(0), Fraction(1)),), (anchor,))

    assert mesh.anchors[0].anchor_id == 91
    assert mesh.anchor_row(91) == 0
    with pytest.raises(KeyError, match="Unknown T-spline anchor ID"):
        mesh.anchor_row(0)
    with pytest.raises(ValueError, match="anchor IDs must be unique"):
        TMesh2D(
            ((Fraction(0), Fraction(1), Fraction(0), Fraction(1)),),
            (anchor, TAnchor2D(91, (Fraction(0), Fraction(0)), knots)),
        )
    foreign = TAnchor2D(
        92,
        (Fraction(0), Fraction(0)),
        LocalKnotVector2D(
            (Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(1)),
            knots.v,
        ),
    )
    with pytest.raises(ValueError, match="do not follow the T-mesh topology"):
        TMesh2D(
            ((Fraction(0), Fraction(1), Fraction(0), Fraction(1)),),
            (foreign,),
        )


def test_local_bernstein_piece_has_exact_anchor_rows() -> None:
    mesh = TMesh2D.tensor_product((0, Fraction(2, 3), 1), (0, 1))
    realization = mesh.realize()

    assert realization.global_coefficient_count == len(mesh.anchors)
    for extraction in realization.extractions:
        assert len(extraction.coefficient_row_ids) == len(extraction.anchor_ids)
        assert len(set(extraction.anchor_ids)) == extraction.local_width
        assert max(extraction.coefficient_row_ids) < mesh.coefficient_count
        np.testing.assert_array_equal(
            np.asarray(extraction.coefficient_rows),
            np.asarray(extraction.coefficient_row_ids),
        )


def test_realization_recomputes_retained_basis_certificate() -> None:
    mesh = TMesh2D.tensor_product((0, 1), (0, 1))
    certificate = mesh.certify()
    assert certificate.minimum_basis_value is not None
    forged = replace(
        certificate,
        minimum_basis_value=certificate.minimum_basis_value + 1.0,
    )

    with pytest.raises(ValueError, match="recomputed evidence"):
        mesh.realize(certificate=forged)


def test_extraction_rank_is_exact_where_binary64_rows_coincide() -> None:
    first = (Fraction(1), Fraction(1), *(Fraction(0),) * 14)
    nearby = (Fraction(1), 1 + Fraction(1, 2**60), *(Fraction(0),) * 14)
    bounds = (0, 1, 0, 1)
    independent = ExtractedBernstein(0, bounds, (3, 7), (0, 1), (first, nearby))
    dependent = ExtractedBernstein(
        0, bounds, (3, 7), (0, 1), (first, tuple(2 * value for value in first))
    )

    assert np.linalg.matrix_rank(np.asarray(independent.extraction_operator)) == 1
    assert independent.rank == 2
    assert dependent.rank == 1
    mesh = TMesh2D.tensor_product((0, Fraction(1, 3), 1), (0, Fraction(2, 5), 1))
    certificate = mesh.certify()
    realization = mesh.realize(certificate=certificate)
    np.testing.assert_array_equal(
        np.asarray(realization.piece_ranks),
        np.asarray(realization.piece_local_widths),
    )
    assert certificate.extraction_full_rank
    assert certificate.minimum_rank_margin == 0


def test_sampled_basis_minimum_is_the_exact_value_rounded_once() -> None:
    nodes = (
        0.06943184420297371,
        0.33000947820757187,
        0.6699905217924281,
        0.9305681557970262,
    )
    univariate_minimum = min(
        comb(3, index) * node**index * (1 - node) ** (3 - index)
        for node in map(Fraction.from_float, nodes)
        for index in range(4)
    )

    certificate = TMesh2D.tensor_product((0, 1), (0, 1)).certify()

    assert certificate.minimum_basis_value == float(univariate_minimum**2)
