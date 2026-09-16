import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cosmology._wave_amr import (
    WaveAMRAdaptivityPlan,
    WaveAMRDiscretizationPlan,
    WaveAMRPhysicsPlan,
)


def test_wave_amr_fixed_step_adaptive_epoch_and_resource_workflow():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(12, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(
                0, (4,), 3, halo_width=1, refinement_ratio=2
            ),
            phx.discretization.BlockLevelPlan(1, (4,), 6, halo_width=1),
        ),
    )
    fd = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    topology = fd.initial_topology()
    discretization = WaveAMRDiscretizationPlan(
        fd,
        adaptivity=WaveAMRAdaptivityPlan(
            maximum_phase_change=0.01,
            current_relative_tolerance=0.5,
            phase_defect_tolerance=1.0,
        ),
        maximum_phase_radians=2.0,
    )
    physics = WaveAMRPhysicsPlan(
        1.0,
        gravitational_constant=0.01,
        reduced_planck_constant=0.03,
    )
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    prepared = discretization.prepare(physics, topology, background)
    coordinate = grid.points[:, 0].reshape((3, 4))
    base = jnp.exp(2j * jnp.pi * coordinate)
    state = prepared.initialize(
        (base, jnp.zeros((6, 4), dtype=jnp.complex128)),
        1.0,
    )

    first = prepared.step(state, 1.00002)
    assert bool(first.successful)
    proposal = prepared.propose_topology(first.state)
    assert proposal.compilation.status.changed
    transition = prepared.transition(first.state, proposal)
    assert bool(transition.successful)
    assert transition.state.psi.topology.epoch.index == 1
    assert transition.evidence.transition_id

    second = transition.prepared.step(transition.state, 1.00004)
    assert bool(second.successful)
    np.testing.assert_allclose(
        second.diagnostics.final_probability,
        first.diagnostics.final_probability,
        rtol=2.0e-8,
    )
    assert bool(second.diagnostics.poisson_closed)
    assert bool(second.diagnostics.kinetic_closed)

    resources = transition.prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(hierarchy, 2),
        maximum_bytes=10_000_000,
    )
    assert resources.admitted
    assert not resources.executable
    assert resources.hierarchy.resource_evidence_id
