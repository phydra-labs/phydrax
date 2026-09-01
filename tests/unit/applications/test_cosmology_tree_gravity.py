import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _direct(positions, masses, softening):
    displacement = positions[None, :, :] - positions[:, None, :]
    squared = jnp.sum(displacement**2, axis=-1) + softening**2
    mask = ~jnp.eye(positions.shape[0], dtype=bool)
    return jnp.sum(
        jnp.where(
            mask[..., None],
            masses[None, :, None] * displacement / squared[..., None] ** 1.5,
            0.0,
        ),
        axis=1,
    )


def test_octree_barnes_hut_fmm_and_treepm_are_finite():
    positions = jnp.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.8, 0.8, 0.8],
            [0.9, 0.8, 0.8],
        ]
    )
    masses = jnp.ones((4,))
    tree = cosmology.ParticleOctreePlan3D((1.0, 1.0, 1.0), 2).prepare(positions, masses)
    assert jnp.all(tree.morton_keys >= 0)
    reference = _direct(positions, masses, 0.01)

    bh = cosmology.BarnesHutGravityPlan(
        1.0, softening=0.01, opening_angle=0.5, use_quadrupole=True
    )
    bh_result = bh.evaluate(tree)
    np.testing.assert_allclose(bh_result.acceleration, reference, rtol=0.2, atol=1e-3)

    fmm = cosmology.UniformFMMPlan(
        1.0, cosmology.CartesianExpansionSpace(1), softening=0.01
    )
    fmm_result = fmm.evaluate(tree)
    assert bool(fmm_result.successful)
    np.testing.assert_allclose(fmm_result.acceleration, reference, rtol=0.3, atol=1e-3)

    split = cosmology.TreePMSplitPolicy(0.1, 0.5, "fixture-discrete-pm")
    treepm = cosmology.TreePMPlan(bh, split).evaluate(tree, jnp.zeros_like(positions))
    assert bool(treepm.successful)
    np.testing.assert_allclose(treepm.total_acceleration, treepm.short_range_acceleration)
    ewald = cosmology.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.01,
        alpha=5.0,
        real_shells=1,
        reciprocal_modes=3,
    )
    periodic = cosmology.PeriodicBarnesHutPlan(bh, ewald).evaluate(tree)
    assert bool(periodic.successful)
    calibration = cosmology.MeshComplementCalibrationPlan(1.0e-12).qualify(
        periodic.acceleration,
        jnp.zeros_like(periodic.acceleration),
        periodic.acceleration,
    )
    assert bool(calibration.successful)


def test_cartesian_fmm_operators_complete_all_six_passes():
    space = cosmology.CartesianExpansionSpace(1)
    operators = cosmology.CartesianFMMOperators(space, 1.0, 0.01)
    positions = jnp.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    masses = jnp.asarray([1.0, 2.0])
    multipole = operators.p2m(positions, masses, jnp.zeros((3,)))
    parent = operators.m2m(multipole, jnp.asarray([0.1, 0.0, 0.0]))
    local = operators.m2l(parent, jnp.zeros((3,)), jnp.ones((3,)))
    child_local = operators.l2l(local, jnp.asarray([0.1, 0.0, 0.0]))
    potential, acceleration = operators.l2p(child_local, jnp.asarray([0.01, 0.0, 0.0]))
    direct = operators.p2p(jnp.asarray([1.0, 1.0, 1.0]), positions, masses)
    assert jnp.isfinite(potential)
    assert jnp.all(jnp.isfinite(acceleration))
    assert jnp.all(jnp.isfinite(direct))


def test_distributed_particle_layout_assigns_key_ranges():
    layout = cosmology.DistributedParticleLayout(
        2, 16, jnp.asarray([0, 32, 64], dtype=jnp.uint32)
    )
    owners = layout.owners(jnp.asarray([0, 31, 32, 63], dtype=jnp.uint32))
    np.testing.assert_array_equal(owners, [0, 0, 1, 1])
