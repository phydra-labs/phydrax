import jax.numpy as jnp

import phydrax as phx


def test_periodic_spectral_conservation_and_entropy_diagnostics():
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(24),),
        axis_names=("x",),
        field_name="u",
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="unit-advection",
    )
    entropy = phx.equations.ConvexEntropyPair(
        system,
        lambda state: 0.5 * state[..., 0] ** 2,
        lambda state: state,
        lambda state, axis, args: 0.5 * state[..., 0] ** 2,
        lambda state: jnp.ones(state.shape[:-1], dtype=bool),
        entropy_id="quadratic",
    )
    problem = phx.equations.ConservationProblemIR(
        "advection",
        "u",
        system,
        None,
    )
    method = phx.discretization.SpectralConservationMethodPlan(
        phx.discretization.PseudospectralMethodPlan(),
        flux_polynomial_degree=1,
        entropy_diagnostics=True,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        space,
        method,
        entropy_pair=entropy,
    )
    x = space.axes[0].nodes
    physical = jnp.sin(2.0 * jnp.pi * x)[..., None]
    state = space.project(physical)

    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    expected = -2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)

    assert jnp.allclose(
        space.reconstruct(residual)[..., 0],
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    assert jnp.max(jnp.abs(diagnostics.conservation_defect)) < 1e-11
    assert diagnostics.entropy is not None
    assert jnp.abs(diagnostics.entropy.semidiscrete_entropy_rate) < 1e-11
    assert diagnostics.entropy.admissible
