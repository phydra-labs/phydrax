#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.applications.relativistic_scattering import (
    bhabha_amplitude,
    breit_wheeler_amplitude,
    breit_wheeler_total_cross_section,
    compton_amplitude,
    electron_muon_annihilation_amplitude,
    electron_muon_differential_cross_section,
    electron_muon_total_cross_section,
    EventStatus,
    finite_scalar_bubble,
    klein_nishina_differential_cross_section,
    LorentzFrame,
    massive_vector_polarization_sum,
    MassShell,
    minkowski_dot,
    moller_amplitude,
    MultiChannelPhaseSpacePlan,
    Particle,
    photon_linear_polarization,
    pure_polarization_density,
    real_virtual_subtraction,
    RealVirtualSubtractionPlan,
    RecursivePhaseSpaceMap,
    rejection_unweight,
    RejectionUnweightingPlan,
    rotate_polarization_density,
    ScalarBubblePlan,
    slash,
    spinor_completeness,
    stokes_density,
    stokes_parameters,
    tensor_polarization_density,
    two_photon_annihilation_amplitude,
    TwoBodyPhaseSpaceMap,
    WeightedEventStream,
)
from phydrax.applications.relativistic_scattering._collision_environment import (
    assign_collision_pileup,
    CollisionEnvironmentPlan,
)


def _empty_event_batch(event_ids):
    identifiers = jnp.asarray(event_ids, dtype=jnp.int64)
    count = identifiers.size
    prepared = phx.particle_physics.ParticleEventPlan(
        catalog=phx.particle_physics.ParticleCatalogReference(
            source_id="collision-test-pdg",
            provider_release="test",
            checksum="test-checksum",
            citation_url="https://pdg.lbl.gov/",
        ),
        momentum_unit=phx.units.GIGAELECTRONVOLT,
        length_unit=phx.units.MILLIMETER,
        time_unit=phx.units.NANOSECOND,
        event_capacity=count,
        particle_capacity=1,
        vertex_capacity=1,
        provider_status_namespace="collision-test",
    ).prepare()
    weights = phx.particle_physics.EventWeightSet(
        jnp.ones((count, 1)),
        names=("nominal",),
        variation_kinds=(phx.particle_physics.WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
    )
    return prepared.admit(
        event_ids=identifiers,
        subevent_ids=jnp.zeros((count,), dtype=jnp.int64),
        event_active=jnp.ones((count,), dtype=jnp.bool_),
        pdg_ids=jnp.zeros((count, 1), dtype=jnp.int32),
        roles=jnp.zeros((count, 1), dtype=jnp.int32),
        provider_status=jnp.zeros((count, 1), dtype=jnp.int32),
        momenta=jnp.zeros((count, 1, 4)),
        rest_energies=jnp.zeros((count, 1)),
        particle_active=jnp.zeros((count, 1), dtype=jnp.bool_),
        mother_indices=jnp.full((count, 1, 2), -1),
        production_vertex_indices=jnp.full((count, 1), -1),
        end_vertex_indices=jnp.full((count, 1), -1),
        color_flow=jnp.zeros((count, 1, 2), dtype=jnp.int32),
        production_vertices=jnp.zeros((count, 1, 4)),
        vertex_active=jnp.zeros((count, 1), dtype=jnp.bool_),
        weights=weights,
        source_id="collision-test-events",
    )


def test_pileup_addressing_uses_all_event_id_bits():
    primary = _empty_event_batch((1, 1 + 2**32))
    pool = _empty_event_batch(tuple(range(8)))
    plan = CollisionEnvironmentPlan(
        20.0,
        maximum_pileup=8,
        instantaneous_luminosity=1.0,
        bunch_spacing=1.0,
        luminosity_unit="1/cm2/s",
        time_unit="ns",
        pileup_profile_id="full-event-id",
    )
    assigned = assign_collision_pileup(plan, primary, pool, jr.key(123))

    assert not (
        jnp.array_equal(assigned.requested_pileup[0], assigned.requested_pileup[1])
        and jnp.array_equal(assigned.pileup_indices[0], assigned.pileup_indices[1])
    )


def test_lorentz_and_mass_shell_identities():
    electron = Particle(
        "electron",
        mass=0.511,
        charge=-1.0,
        spin_twice=1,
        antiparticle="positron",
        statistics="fermion",
    )
    shell = MassShell(electron)
    momentum = shell.from_spatial(jnp.asarray([0.3, -0.2, 0.7]))
    boost = LorentzFrame.boost(jnp.asarray([0.2, -0.1, 0.05]))
    transformed = boost.apply(momentum)

    assert shell.contains(momentum)
    assert jnp.allclose(
        minkowski_dot(transformed.value, transformed.value),
        minkowski_dot(momentum.value, momentum.value),
        atol=1.0e-12,
    )
    assert jnp.allclose(
        boost.inverse().apply(transformed).value,
        momentum.value,
        atol=1.0e-12,
    )


def test_spinor_and_massive_vector_completeness():
    mass = 0.8
    momentum = jnp.asarray(
        [
            1.7,
            0.4,
            -0.3,
            jnp.sqrt(1.7**2 - mass**2 - 0.4**2 - 0.3**2),
        ]
    )
    positive = spinor_completeness(momentum, mass)
    negative = spinor_completeness(momentum, mass, antiparticle=True)
    vector_sum = massive_vector_polarization_sum(momentum, mass)
    metric = jnp.diag(jnp.asarray([1.0, -1.0, -1.0, -1.0]))

    assert jnp.allclose(positive, slash(momentum) + mass * jnp.eye(4), atol=1.0e-11)
    assert jnp.allclose(negative, slash(momentum) - mass * jnp.eye(4), atol=1.0e-11)
    assert jnp.allclose(
        vector_sum,
        -metric + momentum[:, None] * momentum[None, :] / mass**2,
        atol=1.0e-11,
    )


def test_compton_and_breit_wheeler_ward_identities():
    mass = 1.0
    energy = 0.7
    angle = 0.8
    outgoing_energy = energy / (1.0 + energy * (1.0 - jnp.cos(angle)) / mass)
    p = jnp.asarray([mass, 0.0, 0.0, 0.0])
    k = jnp.asarray([energy, 0.0, 0.0, energy])
    k_prime = jnp.asarray(
        [
            outgoing_energy,
            outgoing_energy * jnp.sin(angle),
            0.0,
            outgoing_energy * jnp.cos(angle),
        ]
    )
    p_prime = p + k - k_prime
    epsilon = photon_linear_polarization(k, 0)
    epsilon_prime = photon_linear_polarization(k_prime, 1)

    physical = compton_amplitude(
        p,
        k,
        p_prime,
        k_prime,
        (1, -1),
        epsilon,
        epsilon_prime,
        electron_mass=mass,
    )
    incoming_ward = compton_amplitude(
        p,
        k,
        p_prime,
        k_prime,
        (1, -1),
        k,
        epsilon_prime,
        electron_mass=mass,
    )
    outgoing_ward = compton_amplitude(
        p,
        k,
        p_prime,
        k_prime,
        (1, -1),
        epsilon,
        k_prime,
        electron_mass=mass,
    )

    assert jnp.isfinite(physical)
    assert jnp.abs(incoming_ward) < 1.0e-10
    assert jnp.abs(outgoing_ward) < 1.0e-10

    pair_energy = 1.4
    pair_momentum = jnp.sqrt(pair_energy**2 - mass**2)
    k1 = jnp.asarray([pair_energy, 0.0, 0.0, pair_energy])
    k2 = jnp.asarray([pair_energy, 0.0, 0.0, -pair_energy])
    electron = jnp.asarray([pair_energy, pair_momentum, 0.0, 0.0])
    positron = jnp.asarray([pair_energy, -pair_momentum, 0.0, 0.0])
    epsilon2 = photon_linear_polarization(k2, 1)
    pair_ward = breit_wheeler_amplitude(
        k1, k2, electron, positron, (1, -1), k1, epsilon2, electron_mass=mass
    )
    assert jnp.abs(pair_ward) < 1.0e-10


def test_all_tree_qed_processes_produce_finite_crossing_sums():
    mass = 1.0
    energy = 1.7
    magnitude = jnp.sqrt(energy**2 - mass**2)
    cosine = 0.31
    sine = jnp.sqrt(1.0 - cosine**2)
    p1 = jnp.asarray([energy, 0.0, 0.0, magnitude])
    p2 = jnp.asarray([energy, 0.0, 0.0, -magnitude])
    p3 = jnp.asarray([energy, magnitude * sine, 0.0, magnitude * cosine])
    p4 = p1 + p2 - p3
    k1 = jnp.asarray([energy, energy * sine, 0.0, energy * cosine])
    k2 = p1 + p2 - k1
    epsilon1 = photon_linear_polarization(k1, 0)
    epsilon2 = photon_linear_polarization(k2, 1)

    amplitudes = jnp.asarray(
        [
            electron_muon_annihilation_amplitude(
                p1,
                p2,
                p3,
                p4,
                (1, -1, 1, -1),
                electron_mass=mass,
                muon_mass=mass,
            ),
            moller_amplitude(
                p1,
                p2,
                p3,
                p4,
                (1, -1, 1, -1),
                electron_mass=mass,
            ),
            bhabha_amplitude(
                p1,
                p2,
                p3,
                p4,
                (1, -1, 1, -1),
                electron_mass=mass,
            ),
            two_photon_annihilation_amplitude(
                p1,
                p2,
                k1,
                k2,
                (1, -1),
                epsilon1,
                epsilon2,
                electron_mass=mass,
            ),
        ]
    )
    annihilation_ward = two_photon_annihilation_amplitude(
        p1,
        p2,
        k1,
        k2,
        (1, -1),
        k1,
        epsilon2,
        electron_mass=mass,
    )

    assert jnp.all(jnp.isfinite(amplitudes))
    assert jnp.all(jnp.abs(amplitudes) > 0.0)
    assert jnp.abs(annihilation_ward) < 1.0e-10


def test_analytic_annihilation_cross_section_integrates_exactly():
    s = 31.0
    nodes, weights = np.polynomial.legendre.leggauss(8)
    angular = (
        2.0
        * jnp.pi
        * jnp.sum(weights * electron_muon_differential_cross_section(s, nodes))
    )
    assert jnp.allclose(angular, electron_muon_total_cross_section(s), atol=1.0e-14)


def test_klein_nishina_has_thomson_limit_and_pair_threshold():
    mass = 0.73
    alpha = 1.0 / 137.035999084
    nodes, weights = np.polynomial.legendre.leggauss(16)
    integrated = (
        2.0
        * jnp.pi
        * jnp.sum(
            weights
            * klein_nishina_differential_cross_section(
                1.0e-9,
                nodes,
                electron_mass=mass,
                alpha=alpha,
            )
        )
    )
    thomson = 8.0 * jnp.pi * alpha**2 / (3.0 * mass**2)

    assert jnp.allclose(integrated, thomson, rtol=1.0e-8)
    assert (
        breit_wheeler_total_cross_section(3.9 * mass**2, electron_mass=mass, alpha=alpha)
        == 0.0
    )


def test_two_body_and_recursive_phase_space_normalization():
    total = jnp.asarray([3.0, 0.0, 0.0, 0.0])
    two_body = TwoBodyPhaseSpaceMap(0.0, 0.0)
    point = two_body.map(jnp.asarray([0.37, 0.61]), total)
    assert point.valid
    assert jnp.allclose(jnp.sum(point.momenta, axis=0), total, atol=1.0e-12)
    assert jnp.allclose(point.jacobian, 1.0 / (8.0 * jnp.pi), atol=1.0e-14)

    recursive = RecursivePhaseSpaceMap((0.0, 0.0, 0.0))
    mass_nodes = (jnp.arange(4096) + 0.5) / 4096.0
    coordinates = jnp.stack(
        (
            mass_nodes,
            jnp.full_like(mass_nodes, 0.3),
            jnp.full_like(mass_nodes, 0.7),
            jnp.full_like(mass_nodes, 0.4),
            jnp.full_like(mass_nodes, 0.2),
        ),
        axis=1,
    )
    jacobians = jax.vmap(recursive.map, in_axes=(0, None))(coordinates, total).jacobian
    assert jnp.allclose(
        jnp.mean(jacobians),
        9.0 / (256.0 * jnp.pi**3),
        rtol=2.0e-7,
    )


def test_multi_channel_weight_is_unbiased_for_overlapping_support():
    total = jnp.asarray([5.0, 0.0, 0.0, 0.0])
    first = TwoBodyPhaseSpaceMap(0.0, 0.0)
    second = TwoBodyPhaseSpaceMap(0.0, 0.0)
    plan = MultiChannelPhaseSpacePlan((first, second), (0.2, 0.8))
    mapped_first = plan.map_channel(0, jnp.asarray([0.2, 0.4]), total)
    mapped_second = plan.map_channel(1, jnp.asarray([0.8, 0.9]), total)

    assert mapped_first.valid & mapped_second.valid
    assert jnp.allclose(mapped_first.integration_weight, 1.0 / (8.0 * jnp.pi))
    assert jnp.allclose(mapped_second.integration_weight, 1.0 / (8.0 * jnp.pi))


def test_polarization_density_operations_preserve_probability():
    pure = pure_polarization_density(jnp.asarray([1.0, 1.0j]))
    rotation = jnp.asarray([[1.0, 1.0], [-1.0, 1.0]]) / jnp.sqrt(2.0)
    rotated = rotate_polarization_density(pure, rotation)
    product = tensor_polarization_density(pure, rotated)
    stokes = stokes_density(jnp.asarray([0.2, -0.3, 0.4]))

    assert pure.valid & rotated.valid & product.valid & stokes.valid
    assert jnp.allclose(jnp.trace(product.matrix), 1.0)
    assert jnp.allclose(stokes_parameters(stokes), jnp.asarray([0.2, -0.3, 0.4]))


def test_exact_signed_rejection_enforces_support_bound():
    momenta = jnp.zeros((4, 2, 4))
    stream = WeightedEventStream(
        momenta,
        jnp.asarray([0.25, -0.5, 0.75, -1.0]),
        provenance="signed-reference",
    )
    plan = RejectionUnweightingPlan(1.0, 4, signed=True)
    result = rejection_unweight(stream, plan, jr.key(7))
    violated = rejection_unweight(
        stream, RejectionUnweightingPlan(0.9, 4, signed=True), jr.key(7)
    )

    assert result.status == int(EventStatus.SUCCESS)
    assert jnp.all(result.acceptance_probability <= 1.0)
    assert jnp.all(jnp.isin(result.signs[result.active], jnp.asarray([-1.0, 1.0])))
    assert violated.status == int(EventStatus.SUPPORT_BOUND_VIOLATED)
    assert not jnp.any(violated.active)
    assert violated.support_excess > 0.0


def test_finite_scalar_loop_and_real_virtual_subtraction_cancel():
    bubble = finite_scalar_bubble(
        0.0,
        2.0,
        2.0,
        ScalarBubblePlan(renormalization_scale_squared=2.0),
    )
    first = real_virtual_subtraction(
        lambda z: 1.0 + z + z**2,
        RealVirtualSubtractionPlan(cutoff=1.0e-3),
        finite_virtual=0.7,
    )
    second = real_virtual_subtraction(
        lambda z: 1.0 + z + z**2,
        RealVirtualSubtractionPlan(cutoff=1.0e-6),
        finite_virtual=0.7,
    )

    assert jnp.abs(bubble.value) < 1.0e-12
    assert first.cancellation_residual < 1.0e-12
    assert second.cancellation_residual < 1.0e-12
    assert jnp.allclose(
        first.total,
        second.total,
        rtol=2.0e-3,
        atol=2.0e-5,
    )
