#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.algebraic import (
    analyze_exponent_lattice_scaling,
    SparsePolynomialSupport,
    SparsePolynomialSystem,
)


def test_symmetry_scenario_1() -> None:
    support = SparsePolynomialSupport(
        ("x", "y"),
        ("f",),
        (0, 0),
        ((2, 0), (0, 2)),
    )

    evidence = analyze_exponent_lattice_scaling(support)
    assert evidence.lattice_rank == 1
    assert evidence.free_rank == 1
    assert len(evidence.free_generators) == 1
    assert evidence.torsion_orders == (2,)
    assert len(evidence.torsion_generators) == 1
    assert evidence.status == "exact_support_lattice"
    support = SparsePolynomialSupport(
        ("x",),
        ("f",),
        (0, 0),
        ((0,), (2,)),
    )
    system = SparsePolynomialSystem(support, jnp.asarray((1.0, 0.0)))

    support_evidence = analyze_exponent_lattice_scaling(support)
    system_evidence = analyze_exponent_lattice_scaling(system)
    assert support_evidence.evidence_id == system_evidence.evidence_id
    assert system_evidence.free_rank == 0
    assert system_evidence.torsion_orders == (2,)
    support = SparsePolynomialSupport(
        ("x", "y", "z"),
        ("f",),
        (0,),
        ((1, 2, 3),),
    )

    evidence = analyze_exponent_lattice_scaling(support)
    assert evidence.lattice_rank == 0
    assert evidence.free_rank == 3
    assert evidence.torsion_orders == ()
