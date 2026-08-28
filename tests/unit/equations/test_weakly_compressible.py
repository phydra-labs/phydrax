#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _components(*, continuity=True):
    count = 6
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
        density=(
            phx.discretization.ContinuityDensityPlan()
            if continuity
            else phx.discretization.SummationDensityPlan()
        ),
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2,
        box=phx.discretization.ParticleBox([0.0], [1.0]),
    )
    problem = phx.equations.WeaklyCompressibleFluidProblemIR(
        "fluid", phx.equations.TaitBarotropicMaterial(1.0, 1.0)
    )
    return particles, method, neighborhood, problem


def test_weakly_compressible_problem_requires_stable_forcing_identity():
    material = phx.equations.TaitBarotropicMaterial(1.0, 1.0)

    def forcing(time, position, velocity, density, args):
        del time, velocity, density, args
        return jnp.zeros_like(position)

    with pytest.raises(ValueError, match="stable non-empty ID"):
        phx.equations.WeaklyCompressibleFluidProblemIR(
            "fluid", material, external_acceleration=forcing
        )
    with pytest.raises(ValueError, match="requires external acceleration"):
        phx.equations.WeaklyCompressibleFluidProblemIR(
            "fluid", material, external_acceleration_id="forcing:none"
        )


def test_wc_sph_compiler_initializes_continuity_density_once_by_summation():
    particles, method, neighborhood, problem = _components(continuity=True)
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        problem, particles, method, neighborhood=neighborhood
    )
    position = (jnp.arange(6, dtype=float) + 0.5)[:, None] / 6.0
    velocity = jnp.zeros_like(position)
    state = compiled.initialize_state(position, velocity)
    density = compiled.dynamics.state_layout.density(state)
    explicit = compiled.initialize_state(position, velocity, density)
    differential = compiled.as_differential_problem(position, velocity, t0=0.0, t1=0.1)

    assert jnp.allclose(state, explicit)
    assert differential.initial_state.shape == (6, 3)
    assert (
        differential.state_geometry_id == compiled.dynamics.state_layout.state_geometry_id
    )
    assert (
        differential.discretization_bundle_id == compiled.discretization_bundle.bundle_id
    )
    assert len(compiled.discretization_bundle.records) == 3
    assert (
        compiled.discretization_bundle.record(compiled.dynamics.key).artifact_kind
        == "weakly-compressible-sph-dynamics"
    )


def test_wc_sph_summation_state_rejects_explicit_density():
    particles, method, neighborhood, problem = _components(continuity=False)
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        problem, particles, method, neighborhood=neighborhood
    )
    position = (jnp.arange(6, dtype=float) + 0.5)[:, None] / 6.0
    velocity = jnp.zeros_like(position)

    assert compiled.initialize_state(position, velocity).shape == (6, 2)
    with pytest.raises(ValueError, match="does not accept density"):
        compiled.initialize_state(position, velocity, jnp.ones((6,)))


def test_wc_sph_compiler_validates_cell_search_and_execution_backend():
    particles, method, _, problem = _components(continuity=True)
    box = phx.discretization.ParticleBox([0.0], [1.0])
    too_short = phx.discretization.CellListParticleNeighborhoodPlan(
        0.99 * method.kernel.support_factor * method.smoothing_length,
        4,
        24,
        box,
    )
    with pytest.raises(ValueError, match="cover the SPH kernel support"):
        phx.equations.compile_weakly_compressible_sph_problem(
            problem, particles, method, neighborhood=too_short
        )
    with pytest.raises(ValueError, match="does not match"):
        phx.equations.compile_weakly_compressible_sph_problem(
            problem,
            particles,
            method,
            neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
                method.kernel.support_factor * method.smoothing_length,
                4,
                24,
                box,
            ),
            execution=phx.discretization.ParticleExecutionPolicy(
                realization="dense_pairs"
            ),
        )
