#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization._kinetic_entropy import (
    KineticEntropyRootPlan,
    solve_kinetic_entropy_root,
)
from phydrax.discretization.lattice_boltzmann import (
    D2Q9,
    KBCCollisionPlan,
    LatticeBoltzmannPrecisionPolicy,
)
from phydrax.discretization.lattice_boltzmann._collision import (
    collide_detailed,
    quadratic_equilibrium,
)


def test_safeguarded_entropy_root_brackets_nontrivial_positive_mirror():
    populations = jnp.asarray((0.2, 0.6, 0.2))
    direction = jnp.asarray((0.1, -0.2, 0.1))
    result = solve_kinetic_entropy_root(
        KineticEntropyRootPlan(strategy="hybrid"),
        populations,
        direction,
    )

    assert bool(result.evidence.successful)
    assert bool(result.evidence.bracketed)
    assert float(result.evidence.alpha) > 1.0
    assert float(result.evidence.minimum_population) > 0.0
    assert float(result.evidence.residual) < 1e-9


def test_asymptotic_entropy_route_refuses_uncertified_approximation():
    result = solve_kinetic_entropy_root(
        KineticEntropyRootPlan(
            strategy="asymptotic",
            approximation_tolerance=1.0e-20,
        ),
        jnp.asarray((0.2, 0.6, 0.2)),
        jnp.asarray((0.1, -0.2, 0.1)),
    )

    assert not bool(result.evidence.successful)
    assert not bool(result.evidence.used_approximation)


def test_kbc_variants_return_positive_equilibrium_fixed_points():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    density = jnp.ones((2,))
    velocity = jnp.asarray(((0.02, 0.0), (0.0, 0.02)))
    equilibrium = quadratic_equilibrium(density, velocity, lattice, precision)
    force = jnp.zeros_like(equilibrium)

    for variant in ("a", "b", "c", "d"):
        result = collide_detailed(
            KBCCollisionPlan(variant=variant, stabilizer="hybrid"),
            equilibrium,
            equilibrium,
            force,
            jnp.asarray(1.0),
            velocity,
            lattice,
            precision,
        )
        assert jnp.all(result.successful)
        np.testing.assert_allclose(result.populations, equilibrium, atol=2e-14)
        np.testing.assert_allclose(result.diagnostics.root_residual, 0.0, atol=2e-14)
