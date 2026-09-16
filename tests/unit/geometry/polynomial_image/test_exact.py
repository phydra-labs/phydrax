#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from phydrax.algebraic._exact import ExactSparsePolynomialSystem, QQ
from phydrax.algebraic._system import SparsePolynomialSupport
from phydrax.geometry.polynomial_image import (
    EvidenceDisposition,
    prove_exact_containment,
    SparsePolynomialMap,
)


def test_sparse_rational_composition_proves_twisted_cubic_containment_only():
    map_support = SparsePolynomialSupport(
        ("t",),
        ("x", "y", "z"),
        (0, 1, 2),
        ((1,), (2,), (3,)),
    )
    exact_map = ExactSparsePolynomialSystem(
        map_support,
        ("1", "1", "1"),
        domain=QQ,
    )
    polynomial_map = SparsePolynomialMap.from_exact(exact_map)

    relation_support = SparsePolynomialSupport(
        ("x", "y", "z"),
        ("x_squared_minus_y", "xy_minus_z"),
        (0, 0, 1, 1),
        (
            (0, 1, 0),
            (2, 0, 0),
            (0, 0, 1),
            (1, 1, 0),
        ),
    )
    relations = ExactSparsePolynomialSystem(
        relation_support,
        ("-1", "1", "-1", "1"),
        domain=QQ,
    )

    proof = prove_exact_containment(polynomial_map, relations)

    assert proof.contained
    assert all(remainder.is_zero for remainder in proof.remainders)
    assert proof.claims.exact_containment is EvidenceDisposition.SUPPORTED
    assert proof.claims.numerical_discovery is EvidenceDisposition.NOT_ASSESSED
    assert proof.claims.ideal_equality is EvidenceDisposition.NOT_ASSESSED
    assert proof.claims.real_geometry is EvidenceDisposition.NOT_ASSESSED
    assert proof.claims.topology is EvidenceDisposition.NOT_ASSESSED


def test_nonzero_exact_composition_is_retained_as_rejection_evidence():
    map_support = SparsePolynomialSupport(
        ("t",),
        ("x",),
        (0,),
        ((1,),),
    )
    polynomial_map = SparsePolynomialMap.from_exact(
        ExactSparsePolynomialSystem(map_support, ("1/2",), domain=QQ)
    )
    relation_support = SparsePolynomialSupport(
        ("x",),
        ("x_minus_one",),
        (0, 0),
        ((0,), (1,)),
    )
    relation = ExactSparsePolynomialSystem(
        relation_support,
        ("-1", "1"),
        domain=QQ,
    )

    proof = prove_exact_containment(polynomial_map, relation)

    assert not proof.contained
    assert proof.remainders[0].terms == (((0,), "-1"), ((1,), "1/2"))
    assert proof.claims.exact_containment is EvidenceDisposition.REJECTED
