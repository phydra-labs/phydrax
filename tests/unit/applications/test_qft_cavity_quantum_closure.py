#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cavity_quantum import (
    AdaptiveHcurlCapabilityPlan,
    CavityDipoleCouplingPlan,
    CavityParticipationPlan,
    HcurlCapabilityStatus,
    MaxwellBlochPlan,
    MaxwellBlochState,
    MaxwellBlochVectorField,
    MaxwellEigenmodeNormalizationPlan,
    MaxwellLindbladPlan,
    MaxwellLindbladState,
    MaxwellLindbladVectorField,
    PurcellLoweringPlan,
)
from phydrax.discretization._cell_mesh import CellMesh


def _tetrahedral_mesh():
    return CellMesh.from_tetrahedra(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )


def test_qft_cavity_modes_have_declared_single_quantum_energy():
    plan = MaxwellEigenmodeNormalizationPlan(
        jnp.eye(2),
        jnp.eye(2),
        [3.0, 5.0],
        hbar=2.0,
        equipartition_tolerance=1e-12,
    )
    electric = jnp.asarray(((1.0, 0.0), (0.0, 2.0)), dtype=complex)
    magnetic = electric
    result = eqx.filter_jit(plan.prepare().normalize)(electric, magnetic)

    np.testing.assert_allclose(result.evidence.total_energies, [6.0, 10.0])
    np.testing.assert_allclose(result.evidence.normalization_residuals, 0.0, atol=1e-12)
    assert bool(jnp.all(result.evidence.successful))
    assert len(set(result.mode_ids)) == 2


def test_qft_participation_and_dipole_coupling_retain_si_units():
    modes = (
        MaxwellEigenmodeNormalizationPlan(jnp.eye(1), jnp.eye(1), [3.0], hbar=2.0)
        .prepare()
        .normalize(jnp.ones((1, 1)), jnp.ones((1, 1)))
    )
    participation = CavityParticipationPlan({"dielectric": jnp.eye(1)}).evaluate(modes)
    coupling = CavityDipoleCouplingPlan(
        jnp.ones((1, 1)), [4.0], emitter_id="emitter-a", hbar_joule_second=2.0
    ).evaluate(modes)

    np.testing.assert_allclose(participation.participation_ratios, [[0.5]])
    np.testing.assert_allclose(
        coupling.interaction_energies, 2.0 * coupling.angular_couplings
    )
    assert coupling.angular_coupling_unit == "rad s^-1"
    assert coupling.interaction_energy_unit == "J"
    assert bool(jnp.all(coupling.successful))


def test_qft_purcell_lowering_matches_resonant_bad_cavity_limit():
    plan = PurcellLoweringPlan(10.0, 10.0, 2.0, 0.25)
    result = plan.lower(0.5)

    np.testing.assert_allclose(result.induced_decay_rate, 0.5)
    np.testing.assert_allclose(result.total_decay_rate, 0.75)
    np.testing.assert_allclose(result.purcell_factor, 2.0)
    np.testing.assert_allclose(result.lamb_shift_angular_frequency, 0.0)
    assert result.rate_unit == "rad s^-1"
    assert bool(result.successful)


def test_qft_maxwell_bloch_and_lindblad_rhs_preserve_quantum_constraints():
    bloch_plan = MaxwellBlochPlan(
        cavity_detuning=0.1,
        emitter_detuning=-0.2,
        angular_coupling=0.3,
        cavity_energy_decay_rate=0.05,
        transverse_decay_rate=0.02,
        longitudinal_decay_rate=0.01,
    )
    bloch_state = MaxwellBlochState(0.2 + 0.1j, 0.1, -0.2, -0.8)
    bloch_rate = MaxwellBlochVectorField(bloch_plan)(jnp.asarray(0.0), bloch_state, None)
    assert bool(jnp.isfinite(bloch_rate.cavity_amplitude))
    assert bool(
        jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    (bloch_rate.bloch_u, bloch_rate.bloch_v, bloch_rate.inversion)
                )
            )
        )
    )
    assert bool(bloch_plan.evidence(bloch_state).successful)
    assert (
        bloch_plan.prepare(
            bloch_state, t0=0.0, t1=0.1
        ).state_coordinates.evidence.domain_kind
        == "full"
    )

    lowering = jnp.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=complex)
    lindblad_plan = MaxwellLindbladPlan(
        jnp.diag(jnp.asarray((0.0, 1.0))),
        lowering,
        (jnp.sqrt(0.2) * lowering)[None, ...],
        angular_coupling=0.4,
        cavity_detuning=0.1,
        cavity_energy_decay_rate=0.3,
    )
    state = MaxwellLindbladState(0.2 + 0.05j, jnp.diag(jnp.asarray((0.7, 0.3))))
    rate = MaxwellLindbladVectorField(lindblad_plan)(jnp.asarray(0.0), state, None)

    np.testing.assert_allclose(jnp.trace(rate.density_matrix), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        rate.density_matrix, jnp.conj(rate.density_matrix.T), atol=1e-12
    )
    assert bool(lindblad_plan.evidence(state).successful)
    assert (
        lindblad_plan.prepare(
            state, t0=0.0, t1=0.1
        ).state_coordinates.evidence.domain_kind
        == "full"
    )


def test_qft_adaptive_hcurl_fails_closed_for_unsupported_claims():
    mesh = _tetrahedral_mesh()
    high_order = AdaptiveHcurlCapabilityPlan(mesh, requested_polynomial_order=2)
    missing_adaptation = AdaptiveHcurlCapabilityPlan(
        mesh, requested_polynomial_order=1, require_adaptation=True
    )
    native = AdaptiveHcurlCapabilityPlan(mesh, requested_polynomial_order=1)
    resource_refused = AdaptiveHcurlCapabilityPlan(mesh, maximum_edges=5)

    assert high_order.evidence.status == int(
        HcurlCapabilityStatus.UNSUPPORTED_POLYNOMIAL_ORDER
    )
    assert not high_order.evidence.assembly_supported
    with pytest.raises(NotImplementedError, match="lowest-order"):
        high_order.prepare()
    assert missing_adaptation.evidence.status == int(
        HcurlCapabilityStatus.ADAPTATION_TRANSACTION_REQUIRED
    )
    with pytest.raises(NotImplementedError, match="transaction"):
        missing_adaptation.prepare()
    assert resource_refused.evidence.status == int(
        HcurlCapabilityStatus.RESOURCE_LIMIT_EXCEEDED
    )
    with pytest.raises(NotImplementedError, match="resources"):
        resource_refused.prepare()
    prepared = native.prepare()
    assert prepared.space.edge_count == 6
    assert prepared.evidence.assembly_supported
    assert not prepared.evidence.adaptation_supported
