#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.equations.fem import (
    derived_mortar_entropy_defect,
    entropy_stable_wall_evidence,
    WellBalancedSourceLedger,
)


def test_wall_evidence_and_well_balancing():
    evidence = entropy_stable_wall_evidence(
        jnp.asarray(((1.0, 0.0, 0.0, 2.0),)),
        jnp.zeros((1, 4)),
        jnp.ones((1, 4)),
        jnp.asarray(((1.0, 0.0),)),
    )
    assert bool(evidence.passed)
    ledger = WellBalancedSourceLedger(jnp.ones((3, 2)), jnp.ones((3, 2)))
    assert ledger.balance_error == 0.0


def test_derived_entropy_defect_uses_declared_thermodynamic_values():
    left = jnp.asarray(((1.0, 2.0), (0.5, -0.25)))
    right = jnp.asarray(((2.0, 3.0), (1.5, 0.75)))
    flux = 0.5 * (left + right)
    left_potential = 0.5 * jnp.sum(left**2, axis=-1)
    right_potential = 0.5 * jnp.sum(right**2, axis=-1)
    defect = derived_mortar_entropy_defect(
        left,
        right,
        left,
        right,
        flux,
        left_potential,
        right_potential,
    )
    np.testing.assert_allclose(np.asarray(defect), 0.0, atol=2.0e-14)
