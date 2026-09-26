#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _phase(name, offset, density):
    count = 4
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), 1.0 / count), ambient_dimension=1, name=name
    ).prepare()
    box = phx.discretization.ParticleBox(jnp.asarray([0.0]), jnp.asarray([1.0]))
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1), 1.25 / count, density=density
    )
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            name, phx.equations.TaitBarotropicMaterial(1.0, 2.0)
        ),
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        ),
    )
    position = ((jnp.arange(count) + offset) / count)[:, None]
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    return phx.discretization.PhaseDefinition(name, compiled.dynamics), position, state


def test_interface_geometry_rejects_phases_without_continuity_density():
    target, target_position, target_state = _phase(
        "phase-a", 0.25, phx.discretization.ContinuityDensityPlan()
    )
    source, source_position, source_state = _phase(
        "phase-b", 0.75, phx.discretization.SummationDensityPlan()
    )
    relation = phx.discretization.DenseBipartiteParticleNeighborhoodPlan(16).prepare(
        target.dynamics.particles,
        source.dynamics.particles,
        target_population_id=target.phase_id,
        source_population_id=source.phase_id,
    )
    relation_state = relation.build(target_position, source_position)

    with pytest.raises(ValueError, match="continuity-density phases"):
        phx.discretization.corrected_phase_interface_geometry(
            target, source, relation_state, target_state, source_state
        )
