#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_particle_plan_prepares_material_support_without_current_geometry():
    plan = phx.discretization.ParticleSetPlan(
        [7, 11, 19],
        [0.2, 0.3, 0.5],
        ambient_dimension=2,
        name="fluid",
    )
    particles = plan.prepare(numeric_version="initial")

    assert particles.capacity == 3
    assert particles.active_count == 3
    assert particles.ambient_dimension == 2
    assert particles.support.topology.neighborhoods is None
    assert particles.position_space.representation == "particle_value"
    assert particles.velocity_space.representation == "particle_value"
    assert particles.position_space.vector_space.shape == (3, 2)
    assert particles.measures[0].total_mass == pytest.approx(1.0)
    assert particles.numeric_version == "initial"
    assert particles.plan_id == plan.plan_id
    assert particles.resource_evidence_id == particles.preparation.report_id


def test_particle_padding_is_inert_and_topological():
    subset = phx.discretization.EntitySubset("observed", [True, False, False])
    particles = phx.discretization.ParticleSetPlan(
        [5, 9, -1],
        [1.0, 2.0, np.nan],
        ambient_dimension=1,
        active_mask=[True, True, False],
        subsets=(subset,),
    ).prepare()

    integral = particles.measures[0].integrate(jnp.asarray([3.0, 4.0, jnp.nan]))
    assert integral == pytest.approx(11.0)
    assert int(jnp.sum(particles.entities.subset("observed").mask)) == 1
    assert jnp.all(jnp.isfinite(particles.safe_masses))
    assert particles.active_count == 2

    with pytest.raises(ValueError, match="cannot include inactive"):
        phx.discretization.ParticleSetPlan(
            [5, 9, -1],
            [1.0, 2.0, np.nan],
            ambient_dimension=1,
            active_mask=[True, True, False],
            subsets=(phx.discretization.EntitySubset("invalid", [False, False, True]),),
        )


def test_particle_plan_rejects_invalid_structural_data():
    with pytest.raises(ValueError, match="unique"):
        phx.discretization.ParticleSetPlan([2, 2], [1.0, 1.0], ambient_dimension=1)
    with pytest.raises(ValueError, match="finite and positive"):
        phx.discretization.ParticleSetPlan([2, 3], [1.0, 0.0], ambient_dimension=1)
    with pytest.raises(ValueError, match="at least one active"):
        phx.discretization.ParticleSetPlan(
            [-1, -1],
            [np.nan, np.nan],
            ambient_dimension=1,
            active_mask=[False, False],
        )
    with pytest.raises(ValueError, match="ambient_dimension"):
        phx.discretization.ParticleSetPlan([2], [1.0], ambient_dimension=0)


def test_particle_precision_and_execution_policies_are_explicit():
    precision = phx.discretization.ParticlePrecisionPolicy(
        geometry_dtype="float32",
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        certification_dtype="float64",
    )
    execution = phx.discretization.ParticleExecutionPolicy(
        realization="dense_pairs",
        accumulation="compensated",
    )

    assert precision.geometry(jnp.asarray(1.0)).dtype == jnp.float32
    assert (
        precision.accumulation(jnp.asarray(1.0, dtype=jnp.float32)).dtype == jnp.float64
    )
    assert precision.evidence().domain == "particle"
    assert execution.realization == "dense_pairs"
    assert execution.accumulation == "compensated"
    cell_execution = phx.discretization.ParticleExecutionPolicy(
        realization="cell_edge_list"
    )
    assert cell_execution.realization == "cell_edge_list"
    with pytest.raises(ValueError, match="cell_edge_list"):
        phx.discretization.ParticleExecutionPolicy(realization="cell_ranges")
