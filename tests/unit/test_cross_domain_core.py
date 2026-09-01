import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_scale_artifact_and_observation_contracts_are_shared():
    cosmology = phx.applications.cosmology
    astrodynamics = phx.applications.astrodynamics
    assert cosmology.CosmologyScaleContract is phx.DimensionalScaleContract
    assert astrodynamics.AstrodynamicsScaleContract is phx.DimensionalScaleContract
    assert (
        astrodynamics.AstrodynamicsScaleContract.si().length_coordinate_kind == "physical"
    )
    assert cosmology.CODE_COSMOLOGY_SCALE.length_coordinate_kind == "comoving"
    assert astrodynamics.ArtifactManifest is phx.artifacts.ArtifactManifest

    source = phx.observation.CoordinateLayout(("source:0", "source:1"))
    target = phx.observation.CoordinateLayout(("target:0",))
    response = phx.observation.LinearObservationPlan([[1.0, 2.0]], source, target)
    product = phx.observation.TheoryVector([2.0, 3.0], source, "fixture")
    np.testing.assert_allclose(response.apply(product).values, [8.0])
    covariance = phx.observation.CholeskyCovarianceAction([[2.0]], target)
    np.testing.assert_allclose(covariance.whiten([4.0]), [2.0])


def test_core_gravity_is_shared_by_cosmology_and_astrodynamics():
    cosmology = phx.applications.cosmology
    astrodynamics = phx.applications.astrodynamics
    assert cosmology.BarnesHutGravityPlan is phx.solver.BarnesHutGravityPlan
    positions = jnp.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = jnp.ones((2,))
    tree = astrodynamics.PreparedOctree3D(positions, masses, leaf_capacity=1)
    result = astrodynamics.BarnesHutGravityPlan3D(tree, masses).evaluate(positions)
    assert bool(result.valid)
    np.testing.assert_allclose(result.acceleration[:, 0], [0.25, -0.25], rtol=1e-10)

    kernel = phx.solver.NewtonianPairKernel(1.0, softening=1.0e-15)
    direct, evidence = phx.solver.DirectParticleGravityPlan(kernel).evaluate(
        positions, masses
    )
    np.testing.assert_allclose(direct, result.acceleration, rtol=1e-10)
    assert bool(evidence.successful)


def test_core_kdk_amr_and_event_replay_adapters():
    coefficients = phx.solver.KDKCoefficients(0.5, 1.0, 0.5)
    kdk = phx.solver.KDKTransactionPlan((10.0,))
    proposal = kdk.propose([[1.0]], [[0.0]], [1.0], [[1.0]], coefficients)
    completed = kdk.complete(proposal, [1.0], [[1.0]])
    np.testing.assert_allclose(completed.positions, [[1.5]])
    np.testing.assert_allclose(completed.momenta, [[1.0]])

    assert (
        phx.applications.cosmology.TwoLevelAMRPlan is phx.discretization.TwoLevelAMRPlan
    )
    amr = phx.discretization.TwoLevelAMRPlan((2,), 1)
    coarse = jnp.asarray([[1.0], [2.0]])
    np.testing.assert_allclose(amr.restrict(amr.prolong(coarse)), coarse)

    event = phx.events.FixedCapacityEventState(
        [1, 2],
        [3, 4],
        [0, 1],
        [phx.events.EVENT_COMMITTED, phx.events.EVENT_DEFERRED],
        [True, True],
        False,
    )
    np.testing.assert_array_equal(event.committed, [True, False])
    np.testing.assert_array_equal(event.deferred, [False, True])
