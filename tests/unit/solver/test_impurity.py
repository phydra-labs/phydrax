#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.discretization.dlr import matsubara_frequencies
from phydrax.operators.quantum._impurity import (
    anderson_bath_to_matsubara,
    AndersonBath,
    ImpurityEnvironment,
)
from phydrax.solver._impurity import (
    AndersonBathFitPlan,
    fit_causal_anderson_bath,
    ImpuritySolveRequest,
    solve_all_sector_ed_impurity,
)


def test_environment_is_exclusive_and_causal_bath_fit_keeps_error_visible():
    bath = AndersonBath(jnp.asarray([-1.0, 1.0]), jnp.asarray([0.6, 0.4]))
    labels = jnp.arange(-24, 24)
    target = anderson_bath_to_matsubara(bath, 8.0, labels)

    with pytest.raises(ValueError, match="exactly one"):
        ImpurityEnvironment()
    with pytest.raises(ValueError, match="exactly one"):
        ImpurityEnvironment(hybridization=target, bath=bath)

    fit = fit_causal_anderson_bath(
        AndersonBathFitPlan(
            2,
            -1.0,
            1.0,
            maximum_iterations=4000,
            residual_tolerance=2e-5,
            moment_tolerance=2e-5,
        ),
        target,
    )
    assert bool(fit.evidence.valid)
    assert fit.evidence.causality_residual == pytest.approx(0.0, abs=1e-12)
    assert fit.evidence.relative_fit_residual < 2e-5
    assert fit.evidence.finite_bath_error >= 0.0


def test_all_sector_ed_has_noninteracting_and_hubbard_atomic_limits():
    beta = 8.0
    labels = jnp.arange(-64, 64)
    frequency = matsubara_frequencies(labels, beta=beta, statistics="fermionic")
    finite_bath = AndersonBath(jnp.asarray([0.7]), jnp.asarray([0.5]))
    free = solve_all_sector_ed_impurity(
        ImpuritySolveRequest(
            0.2,
            0.0,
            0.1,
            beta,
            labels,
            ImpurityEnvironment(bath=finite_bath),
        )
    )
    expected_free = 1.0 / (1j * frequency + 0.1 - 0.2 - 0.25 / (1j * frequency - 0.7))
    assert bool(free.evidence.valid)
    assert jnp.allclose(free.green.values, expected_free, rtol=2e-7, atol=2e-8)
    assert jnp.max(jnp.abs(free.self_energy.values)) < 2e-7

    interaction = 4.0
    atomic = solve_all_sector_ed_impurity(
        ImpuritySolveRequest(
            0.0,
            interaction,
            interaction / 2.0,
            beta,
            labels,
            ImpurityEnvironment(bath=AndersonBath(jnp.asarray([]), jnp.asarray([]))),
        )
    )
    expected_atomic = 0.5 / (1j * frequency - interaction / 2.0)
    expected_atomic += 0.5 / (1j * frequency + interaction / 2.0)
    assert bool(atomic.evidence.valid)
    assert atomic.density == pytest.approx(1.0, abs=1e-10)
    assert atomic.double_occupancy < 1e-6
    assert jnp.allclose(atomic.green.values, expected_atomic, rtol=2e-7, atol=2e-8)
    assert atomic.evidence.spectral_sum_residual < 1e-10
    assert atomic.evidence.dyson_residual < 1e-8
