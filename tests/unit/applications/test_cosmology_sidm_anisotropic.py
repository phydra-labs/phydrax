import hashlib

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_kernels import TwoBodyDifferentialKernelPlan
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


def _source_kwargs(speeds, cosines, differential):
    payload = TwoBodyDifferentialKernelPlan.canonical_table_bytes(
        speeds, cosines, differential
    )
    digest = hashlib.sha256(payload).hexdigest()
    lineage = ("synthetic-anisotropic-sidm-fixture",)
    manifest = ReferenceArtifactManifest(
        "synthetic-anisotropic-sidm-table",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="unrestricted",
        nondimensionalization={"speed": 1.0, "cross_section": 1.0},
        uncertainty={"tabulation": 0.0},
        lineage_ids=lineage,
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="synthetic-differential-kernel",
        content_digest=digest,
        producer="unit-fixture",
        producer_version="native",
        build_id="hand-authored",
        license_id=manifest.license_id,
        parent_artifact_ids=lineage,
        resource_id="synthetic-anisotropic-sidm-table",
        status="complete",
    )
    return {
        "source_artifact": artifact,
        "reference_manifest": manifest,
        "commercial_use": True,
        "redistribution": False,
        "training_use": False,
        "export": False,
    }


cosmology = phx.applications.cosmology


def _gravity(particles):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    system = phx.equations.EulerSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "cosmology-sidm-anisotropic-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity = phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    return phx.solver.ParticleMeshGravityPlan(gravity, transfer)


def _case(cross_section, *, maximum_events=4):
    positions = jnp.asarray(
        [
            (0.25, 0.25, 0.25),
            (0.25, 0.25, 0.75),
            (0.25, 0.75, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.25),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
            (0.75, 0.75, 0.75),
        ]
    )
    count = positions.shape[0]
    masses = jnp.full((count,), 1.0 / count)
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((101, 7, 83, 19, 211, 43, 59, 131)),
        masses,
        ambient_dimension=3,
    ).prepare()
    gravity = _gravity(particles)
    kdk = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    pm = cosmology.CosmologicalParticleMeshPlan(kdk, gravity, (0.5, 0.55))
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2, box=box
    ).prepare(particles)
    smoothing = phx.discretization.CoupledSummationSmoothingLengthPlan(
        1.0,
        1.0e-3,
        2.0,
        maximum_iterations=80,
        tolerance=1.0e-6,
        relaxation=0.7,
    )
    spatial_kernel = phx.discretization.WendlandC2SPHKernel(3)
    policy = cosmology.SIDMCollisionPolicy(
        maximum_pair_probability=0.9,
        maximum_particle_probability=0.9,
        minimum_knudsen_number=1.0e-3,
        maximum_events_per_half_step=maximum_events,
    )
    plan = cosmology.CosmologicalSIDMPlan(
        pm,
        neighborhood,
        smoothing,
        spatial_kernel,
        cross_section,
        policy,
    )
    velocity = jnp.asarray(
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.5, -0.5, 0.0),
            (-0.5, 0.5, 0.0),
        ]
    )
    state = kdk.initialize(positions, masses[:, None] * 0.5 * velocity, 0.5)
    return plan, pm, state, neighborhood


def _anisotropic_kernel(*, maximum_speed=3.0):
    species = DarkSectorSpeciesPlan("chi", 1.0)
    speeds = jnp.asarray((0.0, 0.5 * maximum_speed, maximum_speed))
    cosines = jnp.linspace(-1.0, 1.0, 65)
    totals = jnp.asarray((0.01, 0.02, 0.04))
    differential = totals[:, None] * (1.0 + cosines[None, :]) / (4.0 * jnp.pi)
    return TwoBodyDifferentialKernelPlan(
        species,
        species,
        speeds,
        cosines,
        differential,
        **_source_kwargs(speeds, cosines, differential),
        identical_particle_convention="labelled-full-sphere",
    )


def test_constant_isotropic_kernel_specialization_regresses_existing_path():
    value = 0.01
    legacy, _, state, _ = _case(cosmology.SIDMCrossSectionPlan(value))
    species = DarkSectorSpeciesPlan("chi", 1.0)
    specialized, _, _, _ = _case(
        TwoBodyDifferentialKernelPlan.constant_isotropic(species, value)
    )

    expected = legacy.collide(state, jr.key(31), 9, 3.0)
    actual = specialized.collide(state, jr.key(31), 9, 3.0)
    assert legacy.cross_section.cross_section_per_mass_unit == (
        "physical-area/physical-mass"
    )
    assert legacy.cross_section.kernel.cross_section_unit == "physical-area"
    assert legacy.cross_section.kernel.first_species.mass_unit == "physical-mass"
    np.testing.assert_allclose(
        expected.diagnostics.kernel_total_cross_section_per_mass,
        value,
        rtol=0.0,
        atol=0.0,
    )
    assert bool(expected.successful)
    assert bool(actual.successful)
    np.testing.assert_array_equal(
        actual.diagnostics.selected_pairs, expected.diagnostics.selected_pairs
    )
    np.testing.assert_allclose(
        actual.diagnostics.pair_probability,
        expected.diagnostics.pair_probability,
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.accepted_state.canonical_momenta,
        expected.accepted_state.canonical_momenta,
        rtol=0.0,
        atol=2e-15,
    )


def test_velocity_dependent_anisotropic_events_are_stable_and_conservative():
    plan, pm, state, neighborhood = _case(_anisotropic_kernel())
    base = neighborhood.pair_relation
    permutation = jnp.arange(base.capacity - 1, -1, -1)
    relation = phx.sparse.EdgeRelation(
        base.left_indices[permutation],
        base.right_indices[permutation],
        source_size=base.relation.source_size,
        target_size=base.relation.target_size,
        valid=base.valid[permutation],
    )
    reordered_pairs = phx.discretization.ParticlePairRelation(
        relation,
        base.left_particle_ids[permutation],
        base.right_particle_ids[permutation],
        source_support_id=base.source_support_id,
        target_support_id=base.target_support_id,
        same_set=True,
        unordered=True,
        relation_schema_id=base.relation_schema_id,
    )
    reordered_neighborhood = eqx.tree_at(
        lambda prepared: prepared.pair_relation, neighborhood, reordered_pairs
    )
    reordered = cosmology.CosmologicalSIDMPlan(
        pm,
        reordered_neighborhood,
        plan.smoothing,
        plan.kernel,
        plan.cross_section,
        plan.policy,
    )

    first = plan.collide(state, jr.key(31), 9, 2.0)
    second = reordered.collide(state, jr.key(31), 9, 2.0)
    assert bool(first.successful)
    assert bool(second.successful)
    assert int(first.diagnostics.event_count) > 0
    assert bool(first.diagnostics.conservative)
    assert np.all(first.diagnostics.angular_sample_successful[first.pairs.valid])
    np.testing.assert_allclose(first.diagnostics.total_momentum_defect, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        first.diagnostics.total_kinetic_energy_defect, 0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        first.accepted_state.canonical_momenta,
        second.accepted_state.canonical_momenta,
        atol=2e-14,
    )


def test_anisotropic_kernel_domain_failure_rolls_back_atomically():
    plan, _, state, _ = _case(_anisotropic_kernel(maximum_speed=0.5))
    result = plan.collide(state, jr.key(3), 2, 0.1)

    assert not bool(result.successful)
    assert not np.all(result.diagnostics.kernel_supported[result.pairs.valid])
    np.testing.assert_array_equal(
        result.accepted_state.canonical_momenta, state.canonical_momenta
    )


def test_anisotropic_event_capacity_failure_rolls_back_atomically():
    plan, _, state, _ = _case(_anisotropic_kernel(), maximum_events=0)
    result = plan.collide(state, jr.key(31), 9, 2.0)

    assert int(result.diagnostics.event_count) > 0
    assert not bool(result.diagnostics.capacity_valid)
    assert not bool(result.successful)
    np.testing.assert_array_equal(
        result.accepted_state.canonical_momenta, state.canonical_momenta
    )
