#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.numerical_relativity._coupled_runtime import (
    CoupledEvolutionStatus,
    Z4cMatterCoupledRuntime,
)
from phydrax.applications.numerical_relativity._matter_coupling import (
    ConservationLedger,
    ConstraintLedger,
    CoupledParticipantStatus,
    CoupledStageAddress,
    FloorLedger,
    HorizonFluxLedger,
    MatterCouplingPolicy,
    MatterStageProposal,
    SourceExchangeLedger,
    Z4cStageProposal,
)
from phydrax.applications.numerical_relativity._z4c import z4c_snapshot_token
from phydrax.metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection


_TOPOLOGY = "coupling-test-grid"
_CHART = "cartesian"
_CONVENTION = "mostly-plus"
_SCALE = "geometric"
_GEOMETRY_LINEAGE = "toy-adm-lineage"
_PROJECTION = "toy-eulerian-projection"
_SSPRK33_WEIGHTS = jnp.asarray((1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0))


def _controls(
    *,
    z4c_ok=True,
    matter_ok=True,
    stage_offset=0,
    time_offset=0.0,
    floor_mass=0.0,
    source_defect=0.0,
    conservation_defect=0.0,
    z4c_reject_stage=-1,
    matter_reject_stage=-1,
    reuse_snapshot_token=False,
):
    return (
        jnp.asarray(z4c_ok),
        jnp.asarray(matter_ok),
        jnp.asarray(stage_offset, dtype=jnp.int32),
        jnp.asarray(time_offset),
        jnp.asarray(floor_mass),
        jnp.asarray(source_defect),
        jnp.asarray(conservation_defect),
        jnp.asarray(z4c_reject_stage, dtype=jnp.int32),
        jnp.asarray(matter_reject_stage, dtype=jnp.int32),
        jnp.asarray(reuse_snapshot_token),
    )


def _geometry_at_stage(z4c, address, args):
    snapshot_token = jnp.where(
        args[9],
        jnp.int32(1),
        z4c_snapshot_token(address.step_id, address.stage_id + 1),
    )
    dtype = jnp.asarray(z4c).dtype
    identity = jnp.eye(3, dtype=dtype)
    lane_identity = jnp.broadcast_to(identity, (2, 3, 3))
    extrinsic = jnp.asarray(z4c, dtype=dtype) * lane_identity
    return ADMGridGeometry(
        jnp.ones((2,), dtype=dtype),
        jnp.zeros((2, 3), dtype=dtype),
        lane_identity,
        lane_identity,
        jnp.ones((2,), dtype=dtype),
        extrinsic,
        jnp.ones((2,), dtype=bool),
        jnp.ones((2,), dtype=bool),
        snapshot_token=snapshot_token,
        chart_id=_CHART,
        convention_id=_CONVENTION,
        scale_id=_SCALE,
        topology_id=_TOPOLOGY,
        geometry_lineage_id=_GEOMETRY_LINEAGE,
    )


def _projection(
    matter,
    geometry,
    *,
    topology_id=_TOPOLOGY,
    snapshot_offset=0,
):
    dtype = jnp.asarray(matter).dtype
    energy = jnp.full((2,), matter, dtype=dtype)
    return StressEnergyProjection(
        energy,
        jnp.zeros((2, 3), dtype=dtype),
        jnp.zeros((2, 3, 3), dtype=dtype),
        geometry.active,
        jnp.ones((2,), dtype=bool),
        jnp.zeros((2,), dtype=dtype),
        jnp.zeros((2,), dtype=dtype),
        snapshot_token=geometry.snapshot_token + snapshot_offset,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=topology_id,
        projection_id=_PROJECTION,
    )


def _stress_energy_at_stage(matter, geometry, address, args):
    del address, args
    return _projection(matter, geometry)


def _wrong_topology_stress_energy(matter, geometry, address, args):
    del address, args
    return _projection(matter, geometry, topology_id="other-grid")


def _wrong_snapshot_stress_energy(matter, geometry, address, args):
    del address, args
    return _projection(matter, geometry, snapshot_offset=1)


def _evidence(ok):
    return CoupledParticipantStatus(
        jnp.where(ok, jnp.int32(0), jnp.int32(1)),
        True,
        True,
        True,
        True,
        True,
    )


def _proposal_address(address, controls):
    return CoupledStageAddress(
        address.step_start_time,
        address.stage_time + controls[3],
        address.step_end_time,
        address.step_id,
        address.stage_id + controls[2],
        topology_id=address.topology_id,
    )


def _ssprk33_candidate(base, current, rate, address):
    euler = current + address.step_size * rate
    return jnp.where(
        address.stage_id == 0,
        euler,
        jnp.where(
            address.stage_id == 1,
            0.75 * base + 0.25 * euler,
            (1.0 / 3.0) * base + (2.0 / 3.0) * euler,
        ),
    )


def _stage_weight(address):
    return _SSPRK33_WEIGHTS[address.stage_id]


def _propose_z4c(base, current, geometry, stress_energy, address, controls):
    candidate = _ssprk33_candidate(
        base,
        current,
        2.0 + jnp.mean(stress_energy.energy_density),
        address,
    )
    weight = _stage_weight(address)
    source = SourceExchangeLedger(
        weight * address.step_size * jnp.sum(stress_energy.energy_density),
        weight * address.step_size * jnp.sum(stress_energy.momentum_covector, axis=0),
        controls[5],
        jnp.zeros((3,), dtype=jnp.asarray(current).dtype),
    )
    stage_ok = controls[0] & (address.stage_id != controls[7])
    return Z4cStageProposal(
        candidate,
        geometry,
        _proposal_address(address, controls),
        source,
        ConstraintLedger(0.0, jnp.zeros((3,)), 0.0),
        _evidence(stage_ok),
    )


def _propose_fixed_flat_z4c(base, current, geometry, stress_energy, address, controls):
    proposal = _propose_z4c(base, current, geometry, stress_energy, address, controls)
    return Z4cStageProposal(
        jnp.zeros_like(current),
        proposal.geometry,
        proposal.address,
        proposal.source,
        proposal.constraint,
        proposal.evidence,
    )


def _propose_matter(base, current, geometry, stress_energy, address, controls):
    curvature = jnp.mean(jnp.abs(geometry.extrinsic_curvature[..., 0, 0]))
    candidate = _ssprk33_candidate(
        base,
        current,
        (3.0 + curvature) * current,
        address,
    )
    weight = _stage_weight(address)
    conservation = ConservationLedger(
        weight * controls[6],
        0.0,
        jnp.zeros((3,), dtype=jnp.asarray(current).dtype),
    )
    floor = FloorLedger(
        weight * controls[4],
        weight * controls[4],
        jnp.zeros((3,), dtype=jnp.asarray(current).dtype),
        jnp.where(controls[4] > 0.0, jnp.int32(1), jnp.int32(0)),
    )
    horizon = HorizonFluxLedger(
        weight * address.step_size * jnp.sum(stress_energy.energy_density),
        weight * address.step_size * jnp.sum(stress_energy.energy_density),
        jnp.zeros((3,), dtype=jnp.asarray(current).dtype),
        jnp.zeros((3,), dtype=jnp.asarray(current).dtype),
    )
    stage_ok = controls[1] & (address.stage_id != controls[8])
    return MatterStageProposal(
        candidate,
        stress_energy,
        _proposal_address(address, controls),
        conservation,
        floor,
        horizon,
        _evidence(stage_ok),
        matter_kind="grhd",
    )


def _propose_grmhd(base, current, geometry, stress_energy, address, controls):
    proposal = _propose_matter(base, current, geometry, stress_energy, address, controls)
    return MatterStageProposal(
        proposal.candidate,
        proposal.stress_energy,
        proposal.address,
        ConservationLedger(
            proposal.conservation.rest_mass_defect,
            proposal.conservation.energy_defect,
            proposal.conservation.momentum_defect,
            0.0,
        ),
        proposal.floor,
        HorizonFluxLedger(
            proposal.horizon_flux.rest_mass,
            proposal.horizon_flux.energy,
            proposal.horizon_flux.momentum,
            proposal.horizon_flux.angular_momentum,
            0.0,
        ),
        proposal.evidence,
        matter_kind="grmhd",
    )


def _runtime(
    *,
    matter_kind="grhd",
    projection=_stress_energy_at_stage,
    z4c_proposal=_propose_z4c,
    maximum_failures=3,
    maximum_floor=1.0,
):
    matter_proposal = _propose_matter if matter_kind == "grhd" else _propose_grmhd
    return Z4cMatterCoupledRuntime(
        _geometry_at_stage,
        projection,
        z4c_proposal,
        matter_proposal,
        MatterCouplingPolicy(
            source_consistency_tolerance=1.0e-8,
            constraint_tolerance=1.0e-8,
            conservation_tolerance=1.0e-8,
            maximum_floor_rest_mass=maximum_floor,
            maximum_floor_energy=maximum_floor,
            maximum_consecutive_failures=maximum_failures,
        ),
        topology_id=_TOPOLOGY,
        matter_kind=matter_kind,
        z4c_runtime_id=(
            "toy-fixed-flat-z4c"
            if z4c_proposal is _propose_fixed_flat_z4c
            else "toy-z4c-ssprk33"
        ),
        matter_runtime_id=f"toy-{matter_kind}-ssprk33",
    )


def test_zero_matter_reduces_z4c_macro_step_to_vacuum():
    runtime = _runtime()
    start = runtime.initialize(jnp.asarray(0.4), jnp.asarray(0.0))
    result = runtime.advance(start, jnp.asarray(0.1), _controls())

    assert result.successful
    assert jnp.allclose(result.candidate.z4c, 0.4 + 0.1 * 2.0)
    assert jnp.array_equal(
        jnp.stack(tuple(value.stage_time for value in result.addresses)),
        jnp.asarray((0.0, 0.1, 0.05)),
    )
    assert jnp.array_equal(
        jnp.stack(tuple(value.step_id for value in result.addresses)),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    assert jnp.array_equal(
        jnp.stack(tuple(value.stage_id for value in result.addresses)),
        jnp.arange(3, dtype=jnp.int32),
    )


def test_fixed_flat_geometry_reduces_coupled_matter_step_to_standalone_ssprk33():
    runtime = _runtime(z4c_proposal=_propose_fixed_flat_z4c)
    start = runtime.initialize(jnp.asarray(0.0), jnp.asarray(1.0))
    result = runtime.advance(start, jnp.asarray(0.1), _controls())
    stability_polynomial = 1.0 + 0.3 + 0.3**2 / 2.0 + 0.3**3 / 6.0

    assert result.successful
    assert jnp.allclose(result.candidate.matter, stability_polynomial)
    assert jnp.allclose(result.candidate.z4c, 0.0)


@pytest.mark.parametrize(
    ("z4c_reject_stage", "matter_reject_stage"),
    ((1, -1), (-1, 1)),
)
def test_either_participant_rejects_the_entire_three_stage_step(
    z4c_reject_stage, matter_reject_stage
):
    runtime = _runtime()
    start = runtime.initialize(jnp.asarray(0.2), jnp.asarray(1.0))
    result = runtime.advance(
        start,
        jnp.asarray(0.1),
        _controls(
            z4c_reject_stage=z4c_reject_stage,
            matter_reject_stage=matter_reject_stage,
        ),
    )

    assert jnp.array_equal(result.stage_successful, (True, False, True))
    assert not result.successful
    assert jnp.array_equal(result.accepted.z4c, start.z4c)
    assert jnp.array_equal(result.accepted.matter, start.matter)
    assert jnp.array_equal(result.accepted.time, start.time)
    assert jnp.array_equal(
        result.accepted.budget.source_energy, start.budget.source_energy
    )
    assert result.accepted.rejected_steps == 1


@pytest.mark.parametrize(
    "controls",
    (
        _controls(stage_offset=1),
        _controls(time_offset=0.01),
    ),
)
def test_stage_and_time_identity_mismatch_rejects_without_partial_commit(controls):
    runtime = _runtime()
    start = runtime.initialize(jnp.asarray(0.2), jnp.asarray(1.0))
    result = runtime.advance(start, jnp.asarray(0.1), controls)

    assert not result.successful
    assert int(result.status) & int(CoupledEvolutionStatus.STAGE_IDENTITY_MISMATCH)
    assert jnp.array_equal(result.accepted.z4c, start.z4c)
    assert jnp.array_equal(result.accepted.matter, start.matter)


def test_geometry_projection_topology_identity_is_enforced():
    runtime = _runtime(projection=_wrong_topology_stress_energy)
    start = runtime.initialize(jnp.asarray(0.2), jnp.asarray(1.0))
    result = runtime.advance(start, jnp.asarray(0.1), _controls())

    assert not result.successful
    assert int(result.status) & int(CoupledEvolutionStatus.STAGE_IDENTITY_MISMATCH)
    assert jnp.array_equal(result.accepted.time, start.time)


def test_dynamic_snapshot_mismatch_is_folded_into_jit_rejection_status():
    runtime = _runtime(projection=_wrong_snapshot_stress_energy)
    start = runtime.initialize(jnp.asarray(0.2), jnp.asarray(1.0))
    result = jax.jit(lambda state: runtime.advance(state, jnp.asarray(0.1), _controls()))(
        start
    )

    assert not result.successful
    assert int(result.status) & int(CoupledEvolutionStatus.STAGE_IDENTITY_MISMATCH)
    assert jnp.array_equal(result.accepted.z4c, start.z4c)
    assert jnp.array_equal(result.accepted.matter, start.matter)


def test_stale_reused_stage_snapshot_token_rejects_the_macro_step():
    runtime = _runtime()
    start = runtime.initialize(jnp.asarray(0.2), jnp.asarray(1.0))
    result = jax.jit(
        lambda state: runtime.advance(
            state,
            jnp.asarray(0.1),
            _controls(reuse_snapshot_token=True),
        )
    )(start)

    assert jnp.all(result.stage_successful)
    assert not result.successful
    assert not result.qualified
    assert int(result.status) & int(CoupledEvolutionStatus.STAGE_IDENTITY_MISMATCH)
    assert jnp.array_equal(result.accepted.z4c, start.z4c)
    assert jnp.array_equal(result.accepted.matter, start.matter)


def test_only_accepted_stage_ledgers_enter_cumulative_coupled_budgets():
    runtime = _runtime(maximum_floor=1.0)
    start = runtime.initialize(jnp.asarray(0.0), jnp.asarray(2.0))
    first = runtime.advance(
        start,
        jnp.asarray(0.5),
        _controls(floor_mass=0.25),
    )

    assert first.successful
    assert jnp.allclose(first.accepted.budget.source_energy, first.ledgers.source.energy)
    assert jnp.allclose(first.accepted.budget.floor_rest_mass, 0.25)
    assert jnp.allclose(
        first.accepted.budget.horizon_rest_mass,
        first.ledgers.horizon_flux.rest_mass,
    )

    rejected = runtime.advance(
        first.accepted,
        jnp.asarray(0.5),
        _controls(matter_reject_stage=1, floor_mass=0.25),
    )
    assert not rejected.successful
    assert jnp.array_equal(
        rejected.accepted.budget.source_energy,
        first.accepted.budget.source_energy,
    )
    assert jnp.array_equal(
        rejected.accepted.budget.floor_rest_mass,
        first.accepted.budget.floor_rest_mass,
    )
    assert jnp.array_equal(
        rejected.accepted.budget.horizon_rest_mass,
        first.accepted.budget.horizon_rest_mass,
    )


def test_ledger_limit_rejects_candidate_and_preserves_all_accepted_budgets():
    runtime = _runtime()
    start = runtime.initialize(jnp.asarray(0.0), jnp.asarray(1.0))
    result = runtime.advance(
        start,
        jnp.asarray(0.1),
        _controls(source_defect=1.0e-4, conservation_defect=1.0e-4),
    )

    assert not result.successful
    assert int(result.status) & int(CoupledEvolutionStatus.LEDGER_LIMIT_EXCEEDED)
    assert jnp.array_equal(result.accepted.budget.source_energy, 0.0)
    assert jnp.array_equal(result.accepted.z4c, start.z4c)
    assert jnp.array_equal(result.accepted.matter, start.matter)


def test_invalid_step_is_a_bounded_rejection_not_a_partial_update():
    runtime = _runtime()
    start = runtime.initialize(jnp.asarray(0.0), jnp.asarray(1.0))
    result = runtime.advance(start, jnp.asarray(0.0), _controls())

    assert not result.successful
    assert int(result.status) & int(CoupledEvolutionStatus.INVALID_STEP)
    assert result.finite
    assert result.converged
    assert result.physically_valid
    assert jnp.array_equal(result.accepted.z4c, start.z4c)
    assert jnp.array_equal(result.accepted.matter, start.matter)
    assert jnp.array_equal(result.accepted.time, start.time)


def test_consecutive_failure_bound_makes_runtime_terminal_without_unbounded_retries():
    runtime = _runtime(maximum_failures=2)
    controls = _controls(matter_reject_stage=1)
    start = runtime.initialize(jnp.asarray(0.0), jnp.asarray(1.0))
    first = runtime.advance(start, jnp.asarray(0.1), controls)
    second = runtime.advance(first.accepted, jnp.asarray(0.1), controls)
    third = runtime.advance(second.accepted, jnp.asarray(0.1), controls)

    assert not first.accepted.terminal
    assert second.accepted.terminal
    assert second.accepted.consecutive_failures == 2
    assert int(second.status) & int(CoupledEvolutionStatus.FAILURE_LIMIT_REACHED)
    assert not third.attempted
    assert third.accepted.rejected_steps == second.accepted.rejected_steps
    assert int(third.status) & int(CoupledEvolutionStatus.TERMINAL)


@pytest.mark.parametrize("matter_kind", ("grhd", "grmhd"))
def test_coupled_grhd_and_grmhd_steps_retain_fixed_stage_shapes_under_jit(
    matter_kind,
):
    runtime = _runtime(matter_kind=matter_kind)
    state = runtime.initialize(jnp.asarray(0.0), jnp.asarray(1.0))
    controls = _controls(floor_mass=0.1)
    result = jax.jit(
        lambda current, step, values: runtime.advance(current, step, values)
    )(state, jnp.asarray(0.1), controls)

    assert len(result.addresses) == 3
    assert len(result.geometries) == 3
    assert len(result.stress_energy) == 3
    assert len(result.z4c_proposals) == 3
    assert len(result.matter_proposals) == 3
    assert len(result.stage_ledgers) == 3
    assert result.stage_status.shape == (3,)
    assert result.stage_successful.shape == (3,)
    assert result.successful.shape == ()
    assert result.status.shape == ()
    assert result.accepted.budget.source_energy.shape == ()
    assert result.accepted.budget.source_momentum.shape == (3,)
    assert result.accepted.budget.maximum_source_defect.shape == ()
    assert result.accepted.budget.conservation_momentum.shape == (3,)
    assert result.accepted.budget.maximum_conservation_defect.shape == ()
    assert result.accepted.budget.floor_momentum.shape == (3,)
    assert result.accepted.budget.floor_cell_count.shape == ()
    assert result.accepted.budget.horizon_momentum.shape == (3,)
    assert result.accepted.budget.horizon_angular_momentum.shape == (3,)
