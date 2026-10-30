#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from phydrax._interpolation._tspline import (
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
    assert all(isinstance(knot, Fraction) for anchor in mesh.anchors for knot in anchor.local_knots.u)
    assert all(isinstance(knot, Fraction) for anchor in mesh.anchors for knot in anchor.local_knots.v)

    realization = mesh.realize()
    for extraction in realization.extractions:
        exact = extraction.exact_extraction_operator
        assert all(
            sum((row[column] for row in exact), Fraction(0)) == 1
            for column in range(16)
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
