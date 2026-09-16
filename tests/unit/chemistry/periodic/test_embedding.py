#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.chemistry.periodic._embedding import (
    admit_dmft_implicit_derivative,
    dmft_nonlinear_problem,
    dmft_physical_residual,
    DMFTResidualArguments,
    SingleSiteDMFTPlan,
    solve_single_site_dmft,
)
from phydrax.solver._impurity import (
    AndersonBathFitPlan,
    ExactDiagonalizationImpurityProvider,
)


def test_noninteracting_dmft_uses_shared_residual_and_exposes_finite_bath_error():
    labels = jnp.arange(-20, 20)
    plan = SingleSiteDMFTPlan(
        jnp.asarray([-1.0, 1.0]),
        jnp.asarray([0.5, 0.5]),
        labels,
        AndersonBathFitPlan(
            1,
            -1.0,
            1.0,
            maximum_iterations=2000,
            residual_tolerance=1e-7,
        ),
        onsite_energy=0.0,
        interaction=0.0,
        beta=6.0,
        target_density=1.0,
        maximum_iterations=4,
        fixed_point_tolerance=1e-7,
        density_tolerance=1e-7,
        bath_tolerance=1e-7,
    )
    provider = ExactDiagonalizationImpurityProvider(plan.impurity_policy)
    result = solve_single_site_dmft(plan, provider=provider)
    arguments = DMFTResidualArguments(plan, provider)
    direct = dmft_physical_residual(result.state, arguments)
    nonlinear = dmft_nonlinear_problem(arguments).residual(result.state, arguments)

    assert bool(result.evidence.valid)
    assert result.evidence.fixed_point_error < 1e-7
    assert result.evidence.density_error < 1e-7
    assert result.evidence.bath_fit_error < 1e-7
    assert result.evidence.finite_bath_error >= 0.0
    assert jnp.allclose(direct.self_energy, nonlinear.self_energy)
    assert direct.density == pytest.approx(nonlinear.density)

    with pytest.raises(ValueError, match="differentiable provider"):
        admit_dmft_implicit_derivative(
            result,
            arguments,
            branch_gap=1.0,
            jacobian_condition=1.0,
        )
