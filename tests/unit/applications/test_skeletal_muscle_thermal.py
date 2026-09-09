#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.skeletal_muscle.energetics import (
    uchida_umberger_retained_heat,
    UchidaUmberger2010Parameters,
    UchidaUmberger2010Plan,
)
from phydrax.applications.skeletal_muscle.thermal import (
    Pennes1948Boundary,
    Pennes1948Parameters,
    RetainedHeatProjection,
)
from tools.skeletal_muscle_thermal_qualification import (
    manufactured_case,
    manufactured_source,
)


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64(True):
        yield


def test_heterogeneous_insulated_storage_and_rejected_stale_commit():
    prepared = manufactured_case(heterogeneous=True)
    initial = prepared.initial_state()
    source = manufactured_source(prepared, initial)
    candidate = eqx.filter_jit(prepared.propose)(initial, source)
    assert bool(candidate.evidence.successful)
    np.testing.assert_allclose(candidate.proposed_state.temperature_K, 300.025, atol=1e-9)
    np.testing.assert_allclose(candidate.ledger.storage_change_J, 0.2, atol=1e-9)
    np.testing.assert_allclose(candidate.ledger.balance_residual_J, 0, atol=1e-9)
    assert eqx.tree_equal(
        candidate.commit(
            initial,
            prepared=prepared,
            source_state_id=source.source_state_id,
            accept=False,
        ),
        initial,
    )
    committed = candidate.commit(
        initial, prepared=prepared, source_state_id=source.source_state_id
    )
    # Replaying the exact same accepted source interval must not increment either ledger.
    assert eqx.tree_equal(
        candidate.commit(
            committed, prepared=prepared, source_state_id=source.source_state_id
        ),
        committed,
    )
    with pytest.raises(ValueError, match="foreign"):
        candidate.commit(
            initial, prepared=prepared, source_state_id="another-source-state"
        )
    changed = eqx.tree_at(
        lambda x: x.plan.parameters.capacity_J_per_m3_K,
        prepared,
        2 * prepared.plan.parameters.capacity_J_per_m3_K,
    )
    assert eqx.tree_equal(
        candidate.commit(
            initial, prepared=changed, source_state_id=source.source_state_id
        ),
        initial,
    )
    mixed_precision = eqx.tree_at(
        lambda x: x.category_power_W, source, source.category_power_W.astype(jnp.float32)
    )
    with pytest.raises(ValueError, match="precision"):
        prepared.propose(initial, mixed_precision)
    rejected_source = eqx.tree_at(lambda x: x.successful, source, jnp.asarray(False))
    rejected = prepared.propose(initial, rejected_source)
    assert not bool(rejected.evidence.successful)
    assert eqx.tree_equal(
        rejected.commit(
            initial, prepared=prepared, source_state_id=source.source_state_id
        ),
        initial,
    )


def test_no_source_preserves_nonuniform_insulated_energy_and_perfusion_removes_heat():
    initial = np.asarray([301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0])
    insulated = manufactured_case(initial=initial)
    state = insulated.initial_state()
    candidate = insulated.propose(state, manufactured_source(insulated, state, power=0.0))
    assert bool(candidate.evidence.successful)
    np.testing.assert_allclose(candidate.ledger.storage_change_J, 0, atol=1e-9)
    assert float(jnp.max(candidate.proposed_state.temperature_K)) < float(
        jnp.max(state.temperature_K)
    )
    perfused = manufactured_case(perfusion=0.5, initial=np.full(8, 302.0))
    state = perfused.initial_state()
    candidate = perfused.propose(state, manufactured_source(perfused, state, power=0.0))
    assert bool(candidate.evidence.successful)
    np.testing.assert_allclose(
        candidate.proposed_state.temperature_K, 300 + 16 / 8.2, atol=1e-9
    )
    assert float(candidate.ledger.perfusion_out_J) > 0
    assert bool(jnp.all(candidate.volumetric_perfusion_exchange_W_per_m3 < 0))
    np.testing.assert_allclose(
        candidate.ledger.storage_change_J + candidate.ledger.perfusion_out_J, 0, atol=1e-9
    )


@pytest.mark.parametrize("boundary", ["linear-flux", "linear-convection"])
def test_mixed_boundary_linear_solution_and_equal_opposite_boundary_work(boundary):
    prepared = manufactured_case(boundary=boundary)
    initial = prepared.initial_state()
    candidate = prepared.propose(
        initial, manufactured_source(prepared, initial, power=0.0)
    )
    assert bool(candidate.evidence.successful)
    np.testing.assert_allclose(
        candidate.proposed_state.temperature_K, initial.temperature_K, atol=1e-9
    )
    np.testing.assert_allclose(candidate.ledger.dirichlet_out_J, 0.2, atol=1e-9)
    outgoing = candidate.ledger.prescribed_flux_out_J + candidate.ledger.convection_out_J
    np.testing.assert_allclose(outgoing, -0.2, atol=1e-9)


def test_sparse_source_projection_and_uchida_heat_corrections_are_not_double_counted():
    model = UchidaUmberger2010Plan(
        UchidaUmberger2010Parameters(
            [0.5, 1.0],
            [0.5, 0.7],
            [0.1, 0.12],
            [10.0, 10.0],
        ),
        ("shortening", "lengthening"),
    )
    result = model.evaluate(
        jnp.asarray([0.8, 0.8]),
        jnp.asarray([0.7, 0.7]),
        jnp.asarray([100.0, 20000.0]),
        jnp.ones(2),
        jnp.asarray([0.1, 0.12]),
        jnp.asarray([-0.01, 0.1]),
    )
    ledger = uchida_umberger_retained_heat(
        model,
        result,
        source_state_id="accepted-mechanics",
        evidence_id="sourced-retention",
        time_start_s=0.0,
        time_end_s=0.2,
    )
    assert bool(ledger.successful)
    assert float(ledger.category_power_W[2, 1]) > 0
    np.testing.assert_allclose(
        ledger.retained_power_W,
        result.muscle_mass_kg * result.heat_rate_W_per_kg,
        atol=1e-10,
    )
    # Positive shortening work is not deposited as local retained heat.
    assert float(result.muscle_metabolic_power_W[0]) > float(ledger.retained_power_W[0])
    projection = RetainedHeatProjection(
        model.muscle_ids,
        [0, 0, 1],
        [0, 1, 1],
        [0.25, 0.75, 1.0],
        cell_count=2,
        source_model_id=model.model_id,
        retention_evidence_id="sourced-retention",
        asset_id="registered-cell-volumes",
    )
    volumes = jnp.asarray([0.2, 0.8])
    q = projection.project(ledger, volumes)
    np.testing.assert_allclose(
        jnp.sum(q * volumes), jnp.sum(ledger.retained_power_W), atol=1e-10
    )
    with pytest.raises(ValueError, match="conservative"):
        RetainedHeatProjection(
            ("a",),
            [0],
            [0],
            [0.9],
            cell_count=1,
            source_model_id="a",
            retention_evidence_id="a",
            asset_id="a",
        )
    floor_result = model.evaluate(
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.ones(2),
        jnp.asarray([0.1, 0.12]),
        jnp.zeros(2),
    )
    floor = uchida_umberger_retained_heat(
        model,
        floor_result,
        source_state_id="rest",
        evidence_id="sourced-retention",
        time_start_s=0.0,
        time_end_s=0.2,
    )
    np.testing.assert_allclose(floor.category_power_W[3], [0.5, 1.0])
    np.testing.assert_allclose(floor.retained_power_W, [0.5, 1.0])


def test_scalar_field_rejects_invalid_parameters_and_boundary_partition():
    prepared = manufactured_case()
    invalid = Pennes1948Parameters([0.0], [8.0], [0.0], [4.0], [300.0])
    plan = eqx.tree_at(lambda x: x.parameters, prepared.plan, invalid)
    with pytest.raises(ValueError, match="admissible"):
        plan.prepare()
    boundary = prepared.plan.boundaries[0]
    partial = Pennes1948Boundary(
        boundary.facet_ids[:-1],
        "insulated",
        0.0,
        heat_transfer_W_per_m2_K=0.0,
        asset_id=boundary.asset_id,
    )
    plan = eqx.tree_at(lambda x: x.boundaries, prepared.plan, (partial,))
    with pytest.raises(ValueError, match="partition"):
        plan.prepare()
    with pytest.raises(ValueError, match="tensors"):
        Pennes1948Parameters(jnp.eye(3), [8.0], [0.0], [4.0], [300.0])


def test_source_response_derivative_and_vmap_match_conservative_slope():
    prepared = manufactured_case()
    initial = prepared.initial_state()

    def response(power):
        source = manufactured_source(prepared, initial, dt=0.1, power=power)
        return jnp.mean(prepared.propose(initial, source).proposed_state.temperature_K)

    values = eqx.filter_jit(jax.vmap(response))(jnp.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(
        values, 300 + jnp.asarray([1.0, 2.0, 3.0]) * 0.1 / 8, atol=1e-9
    )
    np.testing.assert_allclose(jax.grad(response)(2.0), 0.1 / 8, atol=1e-9)
