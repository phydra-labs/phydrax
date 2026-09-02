# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def test_bernoulli_state_space_laplace_is_explicitly_approximate():
    design = phx.uq.StateSpaceGaussianProcessDesign(
        jnp.asarray([0.0, 0.3, 0.9]),
        jnp.asarray([0.2, 0.7]),
    )
    plan = phx.uq.compile_state_space_kernel(
        phx.kernels.Matern32Kernel(length_scale=0.5), design
    )
    likelihood = phx.uq.ScalarNaturalExponentialFamilyLikelihood(phx.uq.BernoulliFamily())
    result = phx.uq.fit_state_space_approximate_gaussian_process(
        plan,
        jnp.asarray([0.0, 1.0, 1.0]),
        likelihood,
        phx.uq.StateSpaceGaussianProcessLaplace(
            max_iterations=12,
            damping=0.7,
            tolerance=1e-3,
        ),
    )
    assert result.exact is False
    assert result.approximation_kind == "fixed-iteration-log-concave-laplace"
    assert result.posterior_mean.shape == (2,)
    assert jnp.all(result.site_curvature > 0.0)
