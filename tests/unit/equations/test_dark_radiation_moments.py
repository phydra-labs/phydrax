#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.equations._dark_radiation_moments import (
    CosmologicalMultigroupM1System,
    DarkRadiationBoltzmannHierarchyPlan,
    DarkRadiationVETPlan,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention


def _geometry(*, lanes=()):
    identity = jnp.broadcast_to(jnp.eye(3), lanes + (3, 3))
    return ADMGridGeometry(
        jnp.ones(lanes),
        jnp.zeros(lanes + (3,)),
        identity,
        identity,
        jnp.ones(lanes),
        jnp.zeros(lanes + (3, 3)),
        jnp.ones(lanes, dtype="bool"),
        jnp.ones(lanes, dtype="bool"),
        snapshot_token=jnp.asarray(11),
        chart_id="flat",
        convention_id="mostly-minus",
        scale_id="test-scale",
        topology_id="test-grid",
        geometry_lineage_id="flat-lineage",
    )


def _hierarchy_frame(*, time, scale_factor, snapshot):
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(snapshot),
        chart_id="hierarchy-flat",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="hierarchy-grid",
        geometry_lineage_id="hierarchy-flat-lineage",
    )
    return LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        jnp.zeros((4,)),
        jnp.asarray(time),
        jnp.asarray(scale_factor),
        observer_id="hierarchy-observer",
        orientation_id="right-handed-future",
    )


def test_multigroup_m1_physical_c_redshift_exchange_reflux_and_gravity():
    system = CosmologicalMultigroupM1System(
        jnp.asarray((1.0, 2.0, 4.0)),
        3,
        physical_light_speed=2.0,
        reduced_light_speed=0.5,
        beam_risk_limit=0.2,
    )
    diffusion = jnp.asarray((3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    streaming = jnp.asarray((3.0, 6.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0))

    assert bool(system.admissible(diffusion))
    assert bool(system.admissible(streaming))
    assert system.max_wave_speed(streaming, streaming, 0) == 0.5
    diffusion_tensor = system._eddington_tensor(system._groups(diffusion))
    np.testing.assert_allclose(
        diffusion_tensor,
        jnp.broadcast_to(jnp.eye(3) / 3.0, (2, 3, 3)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        system._eddington_tensor(system._groups(streaming))[:, 0, 0], 1.0, rtol=1e-6
    )
    assert not bool(system.qualify_beam_superposition(0.3).accepted)

    redshift = system.group_redshift_flux(diffusion, jnp.asarray(0.1))
    np.testing.assert_allclose(redshift.conservation_residual, 0.0, atol=1e-7)
    assert bool(redshift.accepted)

    exchange = eqx.filter_jit(system.imex_matter_exchange)(
        diffusion,
        jnp.asarray((0.5, 1.0)),
        jnp.asarray((2.0, 2.0)),
        jnp.asarray(0.25),
        jnp.asarray(1.25),
        frame_token=jnp.asarray(4),
        source_state_id="m1-stage-4",
        frame_id="local-frame-4",
        unit_contract_id="units-4",
        frame_realization_id="local-frame-realization-4",
    )
    assert bool(exchange.accepted)
    np.testing.assert_array_equal(
        exchange.exchange.radiation_four_force,
        -exchange.exchange.matter_four_force,
    )
    assert bool(exchange.exchange.exact_opposite)

    reflux = system.reflux(
        diffusion,
        jnp.asarray((0.1, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0)),
        jnp.asarray(2.0),
        topology_id="amr-epoch-2",
    )
    assert bool(reflux.accepted)
    np.testing.assert_allclose(reflux.conservation_residual, 0.0, atol=1e-7)

    projection = system.stress_energy_projection(
        diffusion, _geometry(), source_state_id="m1-stage-4"
    )
    assert bool(projection.all_active_valid)
    assert projection.topology_id == "test-grid"
    assert projection.snapshot_token == 11


def test_m1_realizability_repair_is_explicit_and_beam_risk_refuses():
    system = CosmologicalMultigroupM1System(
        jnp.asarray((1.0, 2.0)), physical_light_speed=1.0
    )
    invalid = jnp.asarray((-1.0, 4.0, 0.0, 0.0))
    repaired = system.enforce_realizability(invalid)
    assert bool(repaired.evidence.correction_applied)
    assert not bool(repaired.evidence.energy_positive_before)
    assert bool(repaired.evidence.energy_positive_after)
    assert bool(repaired.evidence.realizable_after)
    assert bool(system.admissible(repaired.accepted_state))
    qualification = system.qualify_beam_superposition(jnp.asarray(0.5))
    assert bool(qualification.beam_risk)
    assert not bool(qualification.accepted)
    assert "crossing-beam" in qualification.refusal_reason


def test_boltzmann_hierarchy_free_streaming_tight_coupling_and_line_of_sight():
    frame0 = _hierarchy_frame(time=0.0, scale_factor=1.0, snapshot=1)
    frame1 = _hierarchy_frame(time=0.01, scale_factor=1.001, snapshot=2)
    plan = DarkRadiationBoltzmannHierarchyPlan(
        jnp.asarray((0.2, 0.4)),
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((0.25, 0.75)),
        4,
        closure_tolerance=0.2,
        self_interaction_rate=0.5,
        frame=frame0,
    )
    intensity = jnp.zeros(plan.shape).at[..., 0].set(1.0)
    state = plan.initialize(intensity, state_id="hierarchy-0")
    rhs = plan.rhs(state, jnp.zeros((2, 2)), jnp.asarray(0.0))
    np.testing.assert_allclose(
        rhs.intensity[..., 1],
        jnp.broadcast_to(plan.wave_numbers[:, None] / 3.0, (2, 2)),
    )
    tight = plan.closure_evidence(state, jnp.asarray((10.0, 10.0)))
    assert bool(tight.tight_coupling)
    assert bool(tight.closure_qualified)

    advanced, evidence = eqx.filter_jit(plan.advance)(
        state,
        jnp.asarray(0.01),
        jnp.zeros((2, 2)),
        jnp.asarray((1.0, 1.0)),
        end_frame=frame1,
        end_frame_realization_id=frame1.realization_id(),
        state_id="hierarchy-1",
    )
    assert bool(evidence.accepted)
    assert advanced.conformal_time > state.conformal_time
    output = plan.line_of_sight(advanced)
    assert output.density_contrast.shape == (2,)
    assert output.polarization_e.shape == (2,)


def test_vet_formal_solve_preserves_directional_shadow_and_iteration_evidence():
    directions = jnp.asarray(
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    )
    plan = DarkRadiationVETPlan(
        directions,
        jnp.ones((4,)),
        maximum_iterations=24,
        residual_tolerance=1.0e-6,
    )
    result = eqx.filter_jit(plan.formal_solve)(
        jnp.asarray((4.0, 0.1, 0.1, 0.1)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(0.25),
        source_state_id="shadow-column",
        lagged=True,
    )
    assert bool(result.evidence.accepted)
    assert bool(result.evidence.lagged)
    assert result.evidence.shadow_contrast > 0.1
    np.testing.assert_allclose(jnp.trace(result.eddington_tensor), 1.0, rtol=1e-6)
