import jax.numpy as jnp

import phydrax as phx


def test_particle_inverse_realization_closes_density_and_spectral_workflow():
    cosmology = phx.applications.cosmology
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4),
        0.25 * jnp.ones((4,)),
        ambient_dimension=1,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    target_positions = jnp.asarray([[0.125], [0.375], [0.625], [0.875]])
    target = transfer.deposit_content(
        transfer.build(target_positions), particles.masses
    ).density
    layout = phx.observation.CoordinateLayout(
        tuple(f"density:{index}" for index in range(target.size))
    )
    observation = phx.solver.FieldObservationPlan(
        lambda field, args: field,
        target,
        phx.observation.CholeskyCovarianceAction(0.1 * jnp.eye(target.size), layout),
        observation_id="spectral-inverse-target",
    )
    inverse = cosmology.ParticleFieldRealizationPlan(
        transfer,
        observation,
        plan_id="spectral-inverse-workflow",
    )
    initial = jnp.asarray([[0.1], [0.35], [0.6], [0.85]])
    optimized = phx.optim.minimize(
        lambda positions, args: inverse.objective(positions),
        initial,
        method=phx.optim.NonlinearConjugateGradient(),
        termination=phx.optim.OptimizationTermination(maximum_steps=16),
    )
    final = inverse.evaluate(optimized.parameters)
    assert bool(final.successful)
    assert final.objective <= inverse.objective(initial)

    shells = phx.discretization.PeriodicFourierShellPlan(
        (8,), (1.0,), jnp.linspace(0.0, 8.0 * jnp.pi, 6)
    )
    discrepancy = cosmology.SpectralFieldDiscrepancyPlan(shells).evaluate(
        final.predicted_density,
        target,
        "optimized-density",
        "target-density",
    )
    assert bool(discrepancy.successful)
    assert discrepancy.parseval_residual < 1e-10
