#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications import (
    diagrammatic_field as df,
    functional_rg as frg,
    nonequilibrium_field as nef,
    sign_problem as sp,
)
from phydrax.operators.quantum._thermal_green import MatsubaraGreenFunction
from phydrax.operators.quantum._two_particle_green import (
    FermionicTwoParticleChannelConvention,
    MatsubaraTwoParticleGreenFunction,
)


def _crossing_vertex(labels=jnp.arange(-2, 3, dtype=jnp.int32)):
    convention = FermionicTwoParticleChannelConvention("particle-hole-direct")
    route = convention.route(
        labels[:, None, None], labels[None, :, None], labels[None, None, :]
    )
    values = (route[..., 0] - route[..., 2]) * (route[..., 1] - route[..., 3])
    return MatsubaraTwoParticleGreenFunction(
        convention,
        4.0,
        labels,
        labels,
        labels,
        values[..., None, None, None, None].astype(complex),
        fermion_label_minimum=int(labels[0]),
        fermion_label_count=int(labels.size),
    )


def _fermion_diagram(order):
    fermion = df.FieldSpec(f"psi-{order}", statistics="fermion", mass_dimension=1.0)
    propagator = df.PropagatorSpec(fermion, mass=1.0)
    rule = df.VertexRule(
        f"fermion-bilinear-{order}",
        (fermion, fermion),
        1.0,
        perturbative_order=order,
    )
    return df.DiagramGraph(
        (df.VertexInsertion("v", rule),),
        (
            df.PropagatorLine(
                "loop",
                propagator,
                "v",
                "v",
                df.MomentumRoute([0.0]),
            ),
        ),
    )


def test_patch_frg_closes_reciprocal_routing_crossing_and_regulated_flow():
    pi = np.pi
    prepared = frg.FermiSurfacePatchRGPlan(
        jnp.asarray(((0.0, 0.0), (pi, 0.0), (0.0, pi), (pi, pi))),
        jnp.asarray(((2.0 * pi, 0.0), (0.0, 2.0 * pi))),
        jnp.asarray(((1.0, 0.2), (-1.0, 0.2), (0.2, 1.0), (0.2, -1.0))),
        jnp.ones(4),
        jnp.asarray((-1.0, -0.5, 0.5, 1.0)),
        jnp.ones(4) / 4.0,
        beta=6.0,
        matsubara_count=3,
    ).prepare()
    vertex = prepared.vertex(jnp.full((4, 4, 4), 0.15))
    result = prepared.evaluate(vertex, 0.75)

    assert bool(result.evidence.admissible)
    np.testing.assert_allclose(result.evidence.momentum_routing_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.evidence.incoming_crossing_residual, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.evidence.outgoing_crossing_residual, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(result.evidence.ward_residual, 0.0, atol=0.0)
    assert jnp.all(jnp.isfinite(result.particle_particle_loop))
    assert jnp.all(jnp.isfinite(result.particle_hole_loop))
    assert jnp.max(jnp.abs(result.beta_vertex)) > 0.0


def test_lattice_parquet_channels_and_schwinger_dyson_sign_close():
    bare = _crossing_vertex()
    parquet = (
        df.LatticeParquetPlan(tolerance=1e-12)
        .prepare(bare, jnp.zeros((3,) + bare.values.shape[:3]))
        .iterate()
    )

    assert bool(parquet.evidence.successful)
    np.testing.assert_allclose(parquet.full_vertex.values, bare.values)
    np.testing.assert_allclose(parquet.evidence.parquet_identity_residual, 0.0)
    np.testing.assert_allclose(parquet.evidence.routing_residual, 0.0)

    labels = jnp.asarray([0], dtype=jnp.int32)
    one_vertex = MatsubaraTwoParticleGreenFunction(
        FermionicTwoParticleChannelConvention("particle-hole-direct"),
        2.0,
        labels,
        labels,
        labels,
        jnp.ones((1, 1, 1, 1, 1, 1, 1), dtype=complex),
    )
    green = MatsubaraGreenFunction(2.0, labels, jnp.ones(1, dtype=complex))
    expected_self_energy = jnp.asarray([0.5 + 0.0j])
    self_energy = MatsubaraGreenFunction(2.0, labels, expected_self_energy)
    schwinger_dyson = df.lattice_schwinger_dyson_evidence(
        one_vertex,
        green,
        self_energy,
        interaction=2.0,
        density_per_spin=0.5,
    )
    assert bool(schwinger_dyson.satisfied)
    np.testing.assert_allclose(schwinger_dyson.computed_self_energy, expected_self_energy)


def test_sign_free_ctint_keeps_exact_birth_death_ratios_and_raw_chain():
    signs = jnp.asarray((1, -1), dtype=jnp.int32)

    def up_kernel(source, target, delta):
        diagonal = source == target
        return jnp.where(diagonal, jnp.where(delta >= 0.0, 0.75, -0.25), 0.1)

    def down_kernel(source, target, delta):
        return -signs[source] * signs[target] * jnp.conj(up_kernel(source, target, delta))

    plan = df.SignFreeCTINTPlan(2.0, 0.5, signs, steps=256, maximum_order=2)
    result = plan.prepare(
        up_kernel, down_kernel, kernel_id="two-site-half-filled-control"
    ).run(jr.key(11))

    assert bool(result.evidence.successful)
    assert result.chain.orders.shape == (256,)
    assert jnp.all(jnp.real(result.chain.phases) > 1.0 - 1e-12)
    np.testing.assert_allclose(jnp.imag(result.chain.phases), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.evidence.sign_free_symmetry_residual, 0.0, atol=1e-12
    )
    assert result.evidence.average_sign == 1.0

    accepted = result.chain.accepted
    insertion = result.chain.move_types == 1
    prior_order = jnp.where(
        accepted & insertion,
        result.chain.orders - 1,
        jnp.where(accepted & ~insertion, result.chain.orders + 1, result.chain.orders),
    )
    proposal_available = jnp.where(insertion, prior_order < 2, prior_order > 0)
    expected_ratio = jnp.where(
        proposal_available,
        jnp.where(insertion, 4.0 / (prior_order + 1), prior_order / 4.0),
        0.0,
    )
    np.testing.assert_allclose(result.chain.proposal_ratios, expected_ratio)


def test_low_order_diagram_mc_retains_order_sign_and_exact_mh_evidence():
    result = (
        df.LowOrderFermionDiagramMonteCarloPlan(
            steps=512, maximum_order=2, maximum_diagrams=2
        )
        .prepare(
            (_fermion_diagram(1), _fermion_diagram(2)),
            jnp.asarray((1.0 + 0.0j, -0.5 + 0.0j)),
            jnp.asarray(((0.5, 0.5), (0.5, 0.5))),
        )
        .run(jr.key(17))
    )

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.evidence.detailed_balance_residual, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        result.chain.acceptance_probabilities,
        jnp.minimum(1.0, result.chain.proposal_ratios * result.chain.weight_ratios),
    )
    assert result.order_histogram[1] > 0
    assert result.order_histogram[2] > 0
    assert 0.0 < result.evidence.average_sign < 1.0
    assert result.evidence.order_effective_sample_size > 0.0
    assert result.evidence.phase_covariance.shape == (2, 2)


def test_fermionic_keldysh_car_causality_and_second_born_conservation():
    grid = nef.ClosedTimePathPlan(jnp.linspace(0.0, 0.3, 4)).prepare()
    propagators = jnp.broadcast_to(jnp.eye(1, dtype=complex), (4, 1, 1))
    free = nef.fermionic_keldysh_from_propagators(
        grid,
        propagators,
        jnp.asarray([[0.5]]),
        source_id="stationary-half-filled-level",
    )
    identities = nef.fermionic_keldysh_identity_evidence(grid, free)

    assert bool(identities.satisfied)
    np.testing.assert_allclose(identities.equal_time_car_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(identities.retarded_causality_residual, 0.0, atol=0.0)

    result = (
        nef.FermionicSecondBornPlan(
            grid,
            jnp.zeros((1, 1)),
            jnp.asarray([0.02]),
            maximum_iterations=64,
            tolerance=1e-7,
            damping=0.75,
            conservation_tolerance=1e-3,
        )
        .prepare(free)
        .solve()
    )

    assert bool(result.evidence.converged)
    assert bool(result.evidence.conserved)
    assert result.evidence.car_residual < 1e-10
    assert result.evidence.causality_residual == 0.0
    assert result.evidence.schwinger_dyson_residual < 1e-3
    assert jnp.max(jnp.abs(result.self_energy.lesser)) > 0.0


def test_controlled_sign_study_reports_raw_ess_covariance_and_abstains():
    alternating = jnp.tile(jnp.asarray((1.0 + 0.0j, -1.0 + 0.0j)), (2, 32))
    observables = jnp.ones(alternating.shape + (1,))
    plan = sp.ControlledSignStudyPlan(
        minimum_chains=2,
        minimum_draws=32,
        minimum_average_phase=0.1,
        minimum_effective_sample_size=1_000.0,
    )
    result = sp.evaluate_controlled_sign_study(
        plan, alternating, observables, study_id="locked-phase-cancellation-control"
    )

    assert int(result.status) == int(sp.ControlledSignStudyStatus.PHASE_CANCELLATION)
    assert bool(result.abstained)
    np.testing.assert_array_equal(result.raw_weights, alternating)
    assert result.phase_covariance.shape == (2, 2)
    assert result.phase_observable_covariance.shape == (2, 1)
    assert result.effective_sample_size > 0.0
    assert jnp.all(jnp.isnan(result.reweighted_mean))
    assert "no production or generic sign-cure claim" in result.claim
