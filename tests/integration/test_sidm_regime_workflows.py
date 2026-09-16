import hashlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm import SIDMCollisionPolicy
from phydrax.applications.cosmology._sidm_frequent import FrequentSmallAngleSIDMPlan
from phydrax.applications.cosmology._sidm_gravothermal import (
    gravothermal_calibration_payload,
    GravothermalSIDMPlan,
)
from phydrax.applications.cosmology._sidm_kernels import (
    SmallAngleSplitPlan,
    TwoBodyDifferentialKernelPlan,
)
from phydrax.applications.cosmology._sidm_weighted import WeightedSIDMPlan
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


cosmology = phx.applications.cosmology


def _particle_mesh(particles):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    system = phx.equations.EulerSystem(3)
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "sidm-regime-workflow",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        finite_volume,
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
    gravity = phx.solver.ParticleMeshGravityPlan(
        phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
            phx.solver.prepare_balance_law_transport(runtime)
        ),
        phx.discretization.ParticleGridSplatPlan(grid).prepare(particles),
    )
    kinematics = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    return cosmology.CosmologicalParticleMeshPlan(
        kinematics, gravity, jnp.asarray((0.5, 0.51))
    )


def _workflow():
    capacity = 4
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((101, 7, 83, 19)),
        jnp.ones((capacity,)),
        ambient_dimension=3,
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        capacity * (capacity - 1) // 2, box=box
    ).prepare(particles)
    species = DarkSectorSpeciesPlan("chi", 1.0)
    differential = TwoBodyDifferentialKernelPlan.constant_isotropic(species, 0.05)
    split = SmallAngleSplitPlan(differential, 0.8)
    spatial = phx.discretization.WendlandC2SPHKernel(3)
    rare = WeightedSIDMPlan(
        _particle_mesh(particles),
        neighborhood,
        spatial,
        differential,
        SIDMCollisionPolicy(
            maximum_pair_probability=0.2,
            maximum_particle_probability=0.2,
            minimum_knudsen_number=1.0e-12,
            maximum_events_per_half_step=2,
        ),
        smoothing_length_comoving=0.5,
        angular_split=split,
    )
    frequent = FrequentSmallAngleSIDMPlan(
        neighborhood,
        spatial,
        split,
        smoothing_length_comoving=0.5,
        maximum_drag_fraction_per_step=0.1,
        maximum_transverse_variance_per_step=0.2,
        moment_tolerance=0.2,
    )
    active = jnp.asarray((True, True, False, False))
    positions = jnp.asarray(
        (
            (0.45, 0.5, 0.5),
            (0.55, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    microscopic = jnp.where(active, 1.0, 0.0)
    weights = jnp.asarray((1.0, 1.0, 0.0, 0.0))
    velocity = jnp.asarray(
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    state = rare.initialize(
        positions,
        microscopic,
        weights,
        weights[:, None] * 0.5 * velocity,
        0.5,
        active_mask=active,
    )
    return rare, frequent, split, state


def test_rare_and_frequent_angular_profiles_overlap_only_through_explicit_split_evidence():
    rare, frequent, split, state = _workflow()
    physical_step = 1.0e-3
    rare_result = rare.collide(state, jr.key(1), 0, physical_step)
    frequent_result = frequent.apply(state, jr.key(2), 0, physical_step)
    speed = rare_result.diagnostics.relative_speed_physical
    split_evidence = split.moments(speed)
    pair = rare_result.pairs.valid & (
        rare_result.diagnostics.kernel_weight_physical > 0.0
    )

    assert bool(rare_result.successful)
    assert bool(frequent_result.successful)
    assert bool(jnp.all(split_evidence.successful[pair]))
    np.testing.assert_allclose(
        split_evidence.small.total[pair] + split_evidence.rare.total[pair],
        split_evidence.total.total[pair],
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        split_evidence.small.transfer[pair] + split_evidence.rare.transfer[pair],
        split_evidence.total.transfer[pair],
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        rare_result.diagnostics.total_cross_section[pair],
        split_evidence.rare.total[pair],
        rtol=2.0e-13,
    )
    assert bool(jnp.all(rare_result.diagnostics.angular_split_successful))
    assert rare_result.profile_id == rare.plan_id
    np.testing.assert_allclose(
        frequent_result.diagnostics.small_transfer_cross_section[pair],
        split_evidence.small.transfer[pair],
        rtol=2.0e-13,
    )
    assert rare.plan_id != frequent.plan_id


def test_weighted_frequent_and_spherical_closures_remain_separate_successful_workflows():
    rare, frequent, _, state = _workflow()
    collision = rare.collide(state, jr.key(11), 5, 1.0e-3)
    diffused = frequent.apply(collision.accepted_state, jr.key(12), 6, 1.0e-3)
    density, _ = rare.density(diffused.accepted_state)

    assert bool(collision.successful)
    assert bool(diffused.successful)
    assert bool(density.successful)
    np.testing.assert_allclose(
        density.balance.target_total,
        jnp.sum(diffused.accepted_state.gravitational_masses),
        rtol=2.0e-13,
    )

    radial_faces = jnp.linspace(0.0, 6.0, 49)
    payload = gravothermal_calibration_payload(
        radial_faces, 1.0, 0.2, 0.75, calibration_id="isolated-reference"
    )
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = ReferenceArtifactManifest(
        "isolated-gravothermal-reference",
        checksum_algorithm="sha256",
        checksum=checksum,
        size_bytes=len(payload),
        license_id="integration-test-license",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"radius": 1.0, "mass": 1.0, "time": 1.0},
        uncertainty={"conductivity_calibration": 0.05},
        lineage_ids=("integration-reference-source",),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="gravothermal-sidm-calibration",
        content_digest=checksum,
        producer="phydrax-integration-test",
        producer_version="1",
        build_id="isolated-reference-build",
        license_id=manifest.license_id,
        resource_id="isolated-reference",
        status="complete",
        parent_artifact_ids=(manifest.manifest_id,),
    )
    radial = GravothermalSIDMPlan(
        radial_faces,
        1.0,
        0.2,
        0.75,
        calibration_id="isolated-reference",
        calibration_manifest=manifest,
        calibration_artifact=artifact,
        commercial_use=False,
        redistribution=False,
        training_use=False,
        export=False,
    )
    radial_density = jnp.exp(-radial.radial_centers) + 0.1
    radial_dispersion = 1.0 + 0.2 * jnp.exp(-radial.radial_centers)
    gravothermal = radial.advance(
        radial.initialize(radial_density, radial_dispersion), 1.0e-4
    )

    assert bool(gravothermal.successful)
    assert bool(gravothermal.diagnostics.regime_supported)
    assert radial.geometry == "isolated-spherical"
