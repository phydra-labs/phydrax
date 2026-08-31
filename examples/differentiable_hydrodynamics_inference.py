#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Whitened initial-condition inference through a fixed finite-volume rollout."""

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def build_problem(cell_count: int = 32, latent_rank: int = 6):
    bounds = jnp.asarray([[0.0], [1.0]])
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cell_count, periodic=True),),
        axis_names=("x",),
    ).prepare(bounds)
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "whitened-initial-density",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(phx.discretization.MCLimiter()),
        phx.discretization.HLLCFluxPlan(),
        positivity=phx.discretization.ConvexStateLimiterPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    temporal_mesh = phx.discretization.TemporalMesh.uniform(0.0, 0.01, 4, role="internal")
    rollout = phx.solver.ScheduledFiniteVolumeRolloutPlan(
        runtime,
        temporal_mesh,
        replay=phx.solver.FiniteVolumeReplayPolicy("block", block_size=2),
    )

    fd_axis = phx.discretization.UniformAxisSpec(
        cell_count, endpoint=False, periodic=True
    ).materialize(0.0, 1.0)
    finite_difference = phx.discretization.periodic_finite_difference(
        phx.discretization.PreparedTensorGrid((fd_axis,))
    )
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        finite_difference,
        lambda eigenvalues: jnp.exp(-0.04 * eigenvalues),
        rank=latent_rank,
    )

    def initial_state(latent):
        fluctuation = basis.diffusion_matrix @ latent
        density = jnp.exp(0.02 * fluctuation)
        primitive = jnp.stack(
            (density, jnp.zeros_like(density), jnp.ones_like(density)), axis=-1
        )
        return system.primitive_to_conserved(primitive)

    def predict(latent):
        conservative = initial_state(latent)
        state = runtime.initialize_state(
            conservative,
            temporal_mesh.t0,
            temporal_mesh.widths[0],
        )
        result = rollout.rollout(state)
        valid = jnp.all(result.accepted)
        final = result.final_state.cell_average().reshape(discretization.state_shape)
        density = final[..., 0][::4]
        return jnp.where(valid, density, jnp.full_like(density, jnp.nan))

    true_latent = jnp.linspace(-0.6, 0.6, latent_rank)
    observations = predict(true_latent)
    likelihood = phx.uq.GaussianLikelihood(0.01)
    parameter_space = phx.uq.ParameterSpace(
        jnp.zeros((latent_rank,)),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    posterior = phx.uq.PosteriorProblem(
        parameter_space,
        lambda latent: jnp.sum(likelihood.log_prob(predict(latent), observations)),
        predict=predict,
    )
    return posterior, observations, true_latent


def main() -> None:
    posterior, observations, _ = build_problem()
    value, gradient = posterior.validate()
    print(
        {
            "initial_log_density": float(value),
            "gradient_norm": float(jnp.linalg.norm(gradient)),
            "observation_count": int(observations.size),
        }
    )


if __name__ == "__main__":
    main()
