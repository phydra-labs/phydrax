from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.nuclear_fuel import (
    CoupledChannelCoolant,
    IrradiationHistory,
    SingleChannelCoolant,
)
from phydrax.materials import iapws97_liquid_enthalpy
from phydrax.nonlinear import VectorLocalRootPlan
from phydrax.qualification import (
    CapabilityProfile,
    HMACSHA256ReleaseSigner,
    HMACSHA256TrustPolicy,
    ReleaseGateEvidence,
    ReleaseIndex,
)


pytestmark = pytest.mark.strict_jax


def _history(
    inlet: tuple[float, float] = (560.0, 570.0),
    mass_flux: tuple[float, float] = (1000.0, 2000.0),
) -> IrradiationHistory:
    return IrradiationHistory(
        times=np.asarray([0.0, 10.0]),
        zone_edges=np.asarray([0.0, 0.5, 1.0]),
        linear_power=np.zeros((2, 2), dtype=np.float64),
        fast_flux=np.zeros((2, 2), dtype=np.float64),
        coolant_inlet_temperature=np.asarray(inlet),
        coolant_pressure=np.asarray([15.5e6, 15.5e6]),
        coolant_mass_flux=np.asarray(mass_flux),
    )


def _binding() -> tuple[CoupledChannelCoolant, ReleaseIndex, HMACSHA256TrustPolicy]:
    history = _history()
    transfer = SingleChannelCoolant(hydraulic_diameter=0.0118, flow_area_per_rod=8.8e-5)
    # Isolated admission fixture, not experimental/channel release evidence.
    gate = ReleaseGateEvidence(
        "numerical",
        passed=True,
        evidence_ids=("test:channel-energy-contract",),
        reviewer_id="test:channel-reviewer",
        issued_at=10,
        expires_at=100,
    )
    profile = CapabilityProfile(
        "test.exact-channel",
        "test-fixture",
        (CoupledChannelCoolant.support_tuple(history, transfer),),
        required_gates=("numerical",),
        release_evidence=(gate,),
        released=True,
    )
    signer = HMACSHA256ReleaseSigner("test:channel-signer", b"isolated-channel-secret")
    trust = HMACSHA256TrustPolicy(
        {"test:channel-signer": b"isolated-channel-secret"},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    index = ReleaseIndex.sign((profile,), signer, issued_at=10)
    channel = CoupledChannelCoolant(
        history, transfer, index, trust, profile_id=profile.profile_id, at_time=20
    )
    return channel, index, trust


def test_coupled_channel_conserves_transient_energy_with_endpoint_flow() -> None:
    channel, _, _ = _binding()
    old = channel.initial_state
    time, dt = jnp.asarray(10.0), jnp.asarray(2.0)
    power = jnp.asarray([300.0, 900.0], dtype=jnp.float64)
    areas = jnp.asarray([0.02, 0.02], dtype=jnp.float64)
    wall_flux = power / areas
    wall_temperature = jnp.asarray([590.0, 590.0], dtype=jnp.float64)

    def residual(enthalpy: Array) -> Array:
        return (
            channel.residual(
                enthalpy,
                old,
                wall_temperature,
                wall_flux,
                time,
                dt,
                deposited_direct_power=jnp.zeros((2,), dtype=jnp.float64),
                interface_areas=areas,
            )
            / 1000.0
        )

    plan = VectorLocalRootPlan(
        2, plan_id="test:transient-channel-energy", tolerance=1e-10
    )
    solved, diagnostic = jax.jit(
        lambda initial: plan.solve_with_diagnostics(residual, initial)
    )(old)
    assert bool(diagnostic.converged)
    # Independent backward-Euler upwind cell balances, with final prescribed flow.
    masses = np.asarray(channel.balance.reference_cell_masses)
    inlet = float(iapws97_liquid_enthalpy(570.0, 15.5e6))
    mass_flow = 2000.0 * 8.8e-5
    previous = np.asarray(old)
    first = (masses[0] * previous[0] / 2.0 + mass_flow * inlet + 300.0) / (
        masses[0] / 2.0 + mass_flow
    )
    second = (masses[1] * previous[1] / 2.0 + mass_flow * first + 900.0) / (
        masses[1] / 2.0 + mass_flow
    )
    np.testing.assert_allclose(solved, [first, second], rtol=1e-12, atol=1e-7)
    storage = np.sum(masses * (np.asarray(solved) - previous)) / 2.0
    outflow = mass_flow * (float(solved[-1]) - inlet)
    np.testing.assert_allclose(storage + outflow, 1200.0, rtol=1e-10, atol=1e-7)
    assert bool(channel.admissible(solved, time))


def test_coupled_channel_rejects_same_shape_unqualified_history() -> None:
    channel, index, trust = _binding()
    with pytest.raises(ValueError, match="support"):
        CoupledChannelCoolant(
            _history((560.0, 571.0)),
            channel.wall_transfer,
            index,
            trust,
            profile_id=channel.qualification_profile.profile_id,
            at_time=20,
        )


def test_coupled_channel_refuses_a_history_without_forced_flow() -> None:
    channel, index, trust = _binding()
    with pytest.raises(ValueError, match="positive coolant_mass_flux"):
        CoupledChannelCoolant(
            _history(mass_flux=(1000.0, 0.0)),
            channel.wall_transfer,
            index,
            trust,
            profile_id=channel.qualification_profile.profile_id,
            at_time=20,
        )
