#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import fields
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._conservation_ledger import (
    AcceptedConservationIntegralLedger,
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from phydrax.discretization.finite_volume._amr import (
    flux_register_from_accepted_steps,
)


def _stage(
    flux_rate,
    source_rate,
    *,
    geometry_family_id="geometry-family:mesh",
    geometry_layout_id="geometry-layout:mesh",
    geometry_version=1,
    evidence_policy_id="evidence-policy:gcl",
    evidence_version=None,
    topology_epoch_id="topology:0",
    active_cell_mask=(True, True),
    owner=(0, 1),
    neighbour=(1, -1),
    active=(True, True),
    block_id="faces:physical",
    block_kind="physical",
):
    if evidence_version is None:
        evidence_version = geometry_version
    block = ConservationStageFluxRateBlock(
        jnp.asarray(flux_rate),
        jnp.asarray(owner, dtype=jnp.int32),
        jnp.asarray(neighbour, dtype=jnp.int32),
        jnp.asarray(active, dtype=bool),
        block_id,
        block_kind,
    )
    return ConservationStageLedger(
        (block,),
        jnp.asarray(source_rate),
        jnp.asarray(active_cell_mask),
        geometry_family_id=geometry_family_id,
        geometry_layout_id=geometry_layout_id,
        geometry_version=jnp.asarray(geometry_version),
        evidence_policy_id=evidence_policy_id,
        evidence_version=jnp.asarray(evidence_version),
        topology_epoch_id=topology_epoch_id,
    )


def _integrate(
    stage1,
    stage2,
    stage3,
    dt=0.25,
    *,
    start_version=None,
    end_version=4,
    start_evidence_version=None,
    end_evidence_version=None,
    start_time=1.25,
    end_time=None,
    accepted_step=7,
):
    if start_version is None:
        start_version = stage1.geometry_version
    if start_evidence_version is None:
        start_evidence_version = stage1.evidence_version
    if end_evidence_version is None:
        end_evidence_version = end_version
    if end_time is None:
        end_time = start_time + dt
    return AcceptedConservationIntegralLedger.integrate_ssprk33(
        stage1,
        stage2,
        stage3,
        jnp.asarray(dt),
        start_geometry_version=start_version,
        end_geometry_version=jnp.asarray(end_version),
        start_evidence_version=start_evidence_version,
        end_evidence_version=jnp.asarray(end_evidence_version),
        start_topology_epoch_id="topology:0",
        end_topology_epoch_id="topology:0",
        start_time=jnp.asarray(start_time),
        end_time=jnp.asarray(end_time),
        accepted_step=jnp.asarray(accepted_step),
    )


def test_stage_scatter_uses_owner_outward_signs_and_adds_source_rate():
    ledger = _stage(
        [[2.0, 3.0], [5.0, 7.0]],
        [[0.5, 1.0], [1.5, 2.0]],
    )

    np.testing.assert_allclose(
        ledger.scatter_content_rate(),
        np.asarray([[-1.5, -2.0], [-1.5, -2.0]]),
    )
    assert ledger.units == "content/time"
    assert ledger.blocks[0].units == "content/time"


def test_stage_rate_block_schema_contains_no_time_increment():
    field_names = {field.name for field in fields(ConservationStageFluxRateBlock)}

    assert "flux_rate" in field_names
    assert "dt" not in field_names
    assert "time_increment" not in field_names


def test_ssprk33_integrates_rates_exactly_and_multiplies_by_dt_once():
    rate1 = np.asarray([[6.0, 12.0], [18.0, 24.0]])
    rate2 = np.asarray([[12.0, 18.0], [24.0, 30.0]])
    rate3 = np.asarray([[30.0, 36.0], [42.0, 48.0]])
    source1 = np.asarray([[3.0, 6.0], [9.0, 12.0]])
    source2 = np.asarray([[6.0, 9.0], [12.0, 15.0]])
    source3 = np.asarray([[15.0, 18.0], [21.0, 24.0]])
    stage1 = _stage(rate1, source1, geometry_version=10)
    stage2 = _stage(rate2, source2, geometry_version=11)
    stage3 = _stage(rate3, source3, geometry_version=12)
    dt = 0.3

    accepted = _integrate(stage1, stage2, stage3, dt, start_version=10, end_version=13)

    expected_flux = dt * (rate1 / 6.0 + rate2 / 6.0 + 2.0 * rate3 / 3.0)
    expected_source = dt * (source1 / 6.0 + source2 / 6.0 + 2.0 * source3 / 3.0)
    np.testing.assert_allclose(accepted.blocks[0].flux_integral, expected_flux)
    np.testing.assert_allclose(accepted.source_integral, expected_source)
    np.testing.assert_array_equal(stage1.blocks[0].flux_rate, rate1)
    assert accepted.units == "content"
    assert accepted.blocks[0].units == "content"


def test_accepted_ledger_retains_dynamic_temporal_provenance_without_fingerprinting_it():
    stage = _stage([[1.0], [2.0]], [[0.0], [0.0]])
    field_names = {field.name for field in fields(AcceptedConservationIntegralLedger)}
    assert {"start_time", "end_time", "accepted_step"} <= field_names
    assert "dt" not in field_names

    first = _integrate(
        stage,
        stage,
        stage,
        0.25,
        start_time=1.5,
        end_time=1.75,
        accepted_step=8,
    )
    second = _integrate(
        stage,
        stage,
        stage,
        0.25,
        start_time=4.0,
        end_time=4.25,
        accepted_step=19,
    )

    assert float(first.start_time) == pytest.approx(1.5)
    assert float(first.end_time) == pytest.approx(1.75)
    assert int(first.accepted_step) == 8
    assert first.ledger_id == second.ledger_id


def test_ssprk33_accepts_dtype_roundoff_but_rejects_interval_mismatch():
    stage = _stage([[1.0], [2.0]], [[0.0], [0.0]])
    start = jnp.asarray(1.0, dtype=jnp.float32)
    dt = jnp.asarray(0.1, dtype=jnp.float32)
    rounded_end = start + dt

    accepted = _integrate(
        stage,
        stage,
        stage,
        dt,
        start_time=start,
        end_time=rounded_end,
    )
    jax.block_until_ready(accepted.source_integral)

    with pytest.raises(Exception, match="dt must match end_time - start_time"):
        invalid = _integrate(
            stage,
            stage,
            stage,
            dt,
            start_time=start,
            end_time=jnp.asarray(1.2, dtype=jnp.float32),
        )
        jax.block_until_ready(invalid.source_integral)


@pytest.mark.parametrize(
    ("start_time", "end_time", "message"),
    [
        pytest.param(jnp.nan, 1.0, "finite values", id="nonfinite-start"),
        pytest.param(0.0, jnp.inf, "finite values", id="nonfinite-end"),
        pytest.param(1.0, 1.0, "greater than start_time", id="empty"),
        pytest.param(2.0, 1.0, "greater than start_time", id="reversed"),
    ],
)
def test_accepted_ledger_rejects_invalid_temporal_interval(start_time, end_time, message):
    stage = _stage([[1.0], [2.0]], [[0.0], [0.0]])

    with pytest.raises(Exception, match=message):
        invalid = _integrate(
            stage,
            stage,
            stage,
            0.1,
            start_time=start_time,
            end_time=end_time,
        )
        jax.block_until_ready(invalid.source_integral)


@pytest.mark.parametrize(
    ("accepted_step", "message"),
    [
        pytest.param(-1, "nonnegative", id="negative"),
        pytest.param(1.5, "integer dtype", id="noninteger"),
        pytest.param([1], "scalar", id="nonscalar"),
    ],
)
def test_accepted_ledger_requires_nonnegative_scalar_integer_step(accepted_step, message):
    stage = _stage([[1.0], [2.0]], [[0.0], [0.0]])

    with pytest.raises(Exception, match=message):
        invalid = _integrate(
            stage,
            stage,
            stage,
            0.1,
            accepted_step=accepted_step,
        )
        jax.block_until_ready(invalid.source_integral)


def test_repeated_semantic_block_kinds_are_allowed_when_ids_and_routes_are_unique():
    left_patch = ConservationStageFluxRateBlock(
        jnp.asarray([[2.0]]),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([-1], dtype=jnp.int32),
        jnp.asarray([True]),
        "faces:wall-left",
        "physical",
    )
    right_patch = ConservationStageFluxRateBlock(
        jnp.asarray([[3.0]]),
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([-1], dtype=jnp.int32),
        jnp.asarray([True]),
        "faces:wall-right",
        "physical",
    )

    ledger = ConservationStageLedger(
        (left_patch, right_patch),
        jnp.zeros((2, 1)),
        jnp.asarray([True, True]),
        geometry_family_id="geometry-family:mesh",
        geometry_layout_id="geometry-layout:mesh",
        geometry_version=jnp.asarray(1, dtype=jnp.int32),
        evidence_policy_id="evidence-policy:gcl",
        evidence_version=jnp.asarray(1, dtype=jnp.int32),
        topology_epoch_id="topology:0",
    )

    assert tuple(block.block_kind for block in ledger.blocks) == (
        "physical",
        "physical",
    )
    assert ledger.blocks[0].route_id != ledger.blocks[1].route_id
    np.testing.assert_array_equal(ledger.scatter_content_rate(), [[-2.0], [-3.0]])
    accepted = _integrate(ledger, ledger, ledger, start_version=1, end_version=2)
    assert tuple(block.block_kind for block in accepted.blocks) == (
        "physical",
        "physical",
    )


def test_ledger_rejects_duplicate_block_ids_and_duplicate_routes():
    first = ConservationStageFluxRateBlock(
        jnp.ones((1, 1)),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([-1], dtype=jnp.int32),
        jnp.asarray([True]),
        "faces:a",
        "physical",
    )
    repeated_id = ConservationStageFluxRateBlock(
        jnp.ones((1, 1)),
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([-1], dtype=jnp.int32),
        jnp.asarray([True]),
        "faces:a",
        "physical",
    )
    repeated_route = ConservationStageFluxRateBlock(
        jnp.ones((1, 1)),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([-1], dtype=jnp.int32),
        jnp.asarray([True]),
        "faces:b",
        "physical",
    )
    kwargs = {
        "geometry_family_id": "geometry-family:mesh",
        "geometry_layout_id": "geometry-layout:mesh",
        "geometry_version": jnp.asarray(1, dtype=jnp.int32),
        "evidence_policy_id": "evidence-policy:gcl",
        "evidence_version": jnp.asarray(1, dtype=jnp.int32),
        "topology_epoch_id": "topology:0",
    }

    with pytest.raises(ValueError, match="IDs must be unique"):
        ConservationStageLedger(
            (first, repeated_id), jnp.zeros((2, 1)), jnp.ones(2, dtype=bool), **kwargs
        )
    with pytest.raises(ValueError, match="routes must be unique"):
        ConservationStageLedger(
            (first, repeated_route), jnp.zeros((2, 1)), jnp.ones(2, dtype=bool), **kwargs
        )


@pytest.mark.parametrize(
    "source_rate",
    [
        pytest.param([[0.0], [1.0]], id="positive"),
        pytest.param([[0.0], [-1.0]], id="negative"),
        pytest.param([[0.0], [np.finfo(np.float32).tiny]], id="tiny-nonzero"),
    ],
)
def test_stage_ledger_rejects_every_nonzero_source_rate_on_inactive_cells(source_rate):
    with pytest.raises(Exception, match="exactly zero on inactive cells"):
        _stage(
            [[2.0], [0.0]],
            source_rate,
            active_cell_mask=(True, False),
            owner=(0, 0),
            neighbour=(-1, -1),
            active=(True, False),
        )


def test_accepted_ledger_rejects_nonzero_source_integral_on_inactive_cells():
    with pytest.raises(Exception, match="exactly zero on inactive cells"):
        AcceptedConservationIntegralLedger(
            (),
            jnp.asarray([[0.0], [1.0]]),
            jnp.asarray([True, False]),
            geometry_family_id="geometry-family:mesh",
            geometry_layout_id="geometry-layout:mesh",
            stage_geometry_versions=(jnp.asarray(1), jnp.asarray(2), jnp.asarray(3)),
            start_geometry_version=jnp.asarray(1),
            end_geometry_version=jnp.asarray(4),
            evidence_policy_id="evidence-policy:gcl",
            stage_evidence_versions=(
                jnp.asarray(1),
                jnp.asarray(2),
                jnp.asarray(3),
            ),
            start_evidence_version=jnp.asarray(1),
            end_evidence_version=jnp.asarray(4),
            start_topology_epoch_id="topology:0",
            end_topology_epoch_id="topology:0",
            start_time=jnp.asarray(0.0),
            end_time=jnp.asarray(0.1),
            accepted_step=jnp.asarray(1),
        )


def test_active_cell_mask_is_exact_boolean_and_has_one_entry_per_cell():
    with pytest.raises(TypeError, match="boolean dtype"):
        _stage([[1.0], [2.0]], [[0.0], [0.0]], active_cell_mask=(1, 1))
    with pytest.raises(ValueError, match="one value per cell"):
        _stage([[1.0], [2.0]], [[0.0], [0.0]], active_cell_mask=(True,))


def test_active_face_routes_cannot_own_or_neighbour_an_inactive_cell():
    with pytest.raises(Exception, match="active route through an inactive cell"):
        _stage(
            [[1.0], [0.0]],
            [[0.0], [0.0]],
            active_cell_mask=(True, False),
            owner=(0, 0),
            neighbour=(1, -1),
            active=(True, False),
        )


def test_ssprk33_preserves_active_mask_and_rejects_stage_mask_mismatch():
    common = dict(
        flux_rate=[[2.0], [0.0]],
        source_rate=[[1.0], [0.0]],
        active_cell_mask=(True, False),
        owner=(0, 0),
        neighbour=(-1, -1),
        active=(True, False),
    )
    stage1 = _stage(**common, geometry_version=1)
    stage2 = _stage(**common, geometry_version=2)
    stage3 = _stage(**common, geometry_version=3)

    accepted = _integrate(stage1, stage2, stage3, start_version=1, end_version=4)

    np.testing.assert_array_equal(accepted.active_cell_mask, [True, False])
    np.testing.assert_array_equal(accepted.source_integral[1], [0.0])
    np.testing.assert_array_equal(accepted.scatter_content_integral()[1], [0.0])

    changed = _stage(
        [[2.0], [0.0]],
        [[1.0], [0.0]],
        active_cell_mask=(True, True),
        owner=(0, 0),
        neighbour=(-1, -1),
        active=(True, False),
        geometry_version=2,
    )
    with pytest.raises(Exception, match="identical active-cell masks"):
        mismatched = _integrate(stage1, changed, stage3, start_version=1, end_version=4)
        jax.block_until_ready(mismatched.source_integral)

    changed_family = _stage(
        **common, geometry_version=2, geometry_family_id="geometry-family:other"
    )
    with pytest.raises(ValueError, match="share one geometry family"):
        _integrate(stage1, changed_family, stage3, start_version=1, end_version=4)


def test_dynamic_ale_versions_and_rates_share_one_jit_geometry_layout():
    template = _stage(
        [[0.0], [0.0]],
        [[0.0], [0.0]],
        geometry_version=0,
    )
    trace_count = {"value": 0}

    def form_and_integrate(
        stage_rates,
        geometry_versions,
        evidence_versions,
        end_geometry_version,
        end_evidence_version,
        start_time,
        end_time,
        accepted_step,
    ):
        trace_count["value"] += 1
        stages = tuple(
            ConservationStageLedger(
                (template.blocks[0].with_flux_rate(rate),),
                jnp.zeros((2, 1)),
                template.active_cell_mask,
                geometry_family_id=template.geometry_family_id,
                geometry_layout_id=template.geometry_layout_id,
                geometry_version=geometry_version,
                evidence_policy_id=template.evidence_policy_id,
                evidence_version=evidence_version,
                topology_epoch_id=template.topology_epoch_id,
            )
            for rate, geometry_version, evidence_version in zip(
                stage_rates, geometry_versions, evidence_versions
            )
        )
        return AcceptedConservationIntegralLedger.integrate_ssprk33(
            stages[0],
            stages[1],
            stages[2],
            jnp.asarray(0.5),
            start_geometry_version=geometry_versions[0],
            end_geometry_version=end_geometry_version,
            start_evidence_version=evidence_versions[0],
            end_evidence_version=end_evidence_version,
            start_topology_epoch_id="topology:0",
            end_topology_epoch_id="topology:0",
            start_time=start_time,
            end_time=end_time,
            accepted_step=accepted_step,
        )

    integrate_jit = jax.jit(form_and_integrate)
    first_rates = (
        jnp.asarray([[2.0], [4.0]]),
        jnp.asarray([[4.0], [6.0]]),
        jnp.asarray([[8.0], [10.0]]),
    )
    second_rates = tuple(rate + 1.0 for rate in first_rates)

    first = integrate_jit(
        first_rates,
        jnp.asarray([10, 11, 12]),
        jnp.asarray([110, 117, 125]),
        jnp.asarray(13),
        jnp.asarray(131),
        jnp.asarray(2.0),
        jnp.asarray(2.5),
        jnp.asarray(4),
    )
    second = integrate_jit(
        second_rates,
        jnp.asarray([20, 21, 22]),
        jnp.asarray([210, 218, 229]),
        jnp.asarray(23),
        jnp.asarray(237),
        jnp.asarray(8.0),
        jnp.asarray(8.5),
        jnp.asarray(9),
    )
    jax.block_until_ready(first.source_integral)
    jax.block_until_ready(second.source_integral)

    assert trace_count["value"] == 1
    assert first.geometry_layout_id == second.geometry_layout_id
    assert first.evidence_policy_id == second.evidence_policy_id
    np.testing.assert_array_equal(
        [int(version) for version in first.stage_geometry_versions], [10, 11, 12]
    )
    np.testing.assert_array_equal(
        [int(version) for version in second.stage_geometry_versions], [20, 21, 22]
    )
    np.testing.assert_array_equal(
        [int(version) for version in first.stage_evidence_versions], [110, 117, 125]
    )
    np.testing.assert_array_equal(
        [int(version) for version in second.stage_evidence_versions],
        [210, 218, 229],
    )
    assert int(first.start_geometry_version) == 10
    assert int(first.end_geometry_version) == 13
    assert int(first.start_evidence_version) == 110
    assert int(first.end_evidence_version) == 131
    assert int(second.start_geometry_version) == 20
    assert int(second.end_geometry_version) == 23
    assert int(second.start_evidence_version) == 210
    assert int(second.end_evidence_version) == 237
    assert float(first.start_time) == pytest.approx(2.0)
    assert float(first.end_time) == pytest.approx(2.5)
    assert int(first.accepted_step) == 4
    assert float(second.start_time) == pytest.approx(8.0)
    assert float(second.end_time) == pytest.approx(8.5)
    assert int(second.accepted_step) == 9


def test_stage_evidence_identity_requires_canonical_policy_and_scalar_integer_version():
    ledger = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        evidence_policy_id="evidence-policy:ale-gcl",
        evidence_version=9,
    )

    assert ledger.evidence_policy_id == "evidence-policy:ale-gcl"
    assert int(ledger.evidence_version) == 9
    with pytest.raises(ValueError, match="nonempty canonical string"):
        _stage(
            [[1.0], [2.0]],
            [[0.0], [0.0]],
            evidence_policy_id=" evidence-policy:ale-gcl ",
        )
    with pytest.raises(ValueError, match="scalar"):
        _stage([[1.0], [2.0]], [[0.0], [0.0]], evidence_version=[1])
    with pytest.raises(TypeError, match="integer dtype"):
        _stage([[1.0], [2.0]], [[0.0], [0.0]], evidence_version=1.0)


def test_geometry_versions_are_dynamic_scalar_integers():
    with pytest.raises(ValueError, match="scalar"):
        _stage([[1.0], [2.0]], [[0.0], [0.0]], geometry_version=[1])
    with pytest.raises(TypeError, match="integer dtype"):
        _stage([[1.0], [2.0]], [[0.0], [0.0]], geometry_version=1.0)


def test_accepted_geometry_and_evidence_endpoints_are_retained_and_match_stage_one():
    stages = tuple(
        _stage(
            [[1.0], [2.0]],
            [[0.0], [0.0]],
            geometry_version=geometry_version,
            evidence_version=evidence_version,
        )
        for geometry_version, evidence_version in ((7, 70), (8, 83), (9, 91))
    )
    accepted = _integrate(
        *stages,
        start_version=7,
        end_version=10,
        start_evidence_version=70,
        end_evidence_version=104,
    )

    np.testing.assert_array_equal(
        [int(version) for version in accepted.stage_geometry_versions], [7, 8, 9]
    )
    np.testing.assert_array_equal(
        [int(version) for version in accepted.stage_evidence_versions], [70, 83, 91]
    )
    assert int(accepted.start_geometry_version) == 7
    assert int(accepted.end_geometry_version) == 10
    assert int(accepted.start_evidence_version) == 70
    assert int(accepted.end_evidence_version) == 104

    with pytest.raises(Exception, match="first stage geometry version"):
        invalid = _integrate(
            *stages,
            start_version=6,
            end_version=10,
            start_evidence_version=70,
        )
        jax.block_until_ready(invalid.source_integral)
    with pytest.raises(Exception, match="first stage evidence version"):
        invalid = _integrate(
            *stages,
            start_version=7,
            end_version=10,
            start_evidence_version=69,
        )
        jax.block_until_ready(invalid.source_integral)


def test_ssprk33_rejects_geometry_layout_evidence_policy_or_route_mismatch():
    stage1 = _stage([[1.0], [2.0]], [[0.0], [0.0]], geometry_version=1)
    changed_layout = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_layout_id="geometry-layout:changed",
        geometry_version=2,
    )
    changed_policy = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_version=2,
        evidence_policy_id="evidence-policy:changed",
    )
    changed_route = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_version=2,
        owner=(1, 1),
        neighbour=(0, -1),
    )

    with pytest.raises(ValueError, match="one geometry layout"):
        _integrate(stage1, changed_layout, stage1, start_version=1)
    with pytest.raises(ValueError, match="one evidence policy"):
        _integrate(stage1, changed_policy, stage1, start_version=1)
    with pytest.raises(ValueError, match="identical block IDs, block kinds, and routes"):
        _integrate(stage1, changed_route, stage1, start_version=1)


def test_ssprk33_cannot_span_a_topology_epoch_change():
    stage1 = _stage([[1.0], [2.0]], [[0.0], [0.0]], geometry_version=1)
    changed_stage = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_version=2,
        topology_epoch_id="topology:changed",
    )

    with pytest.raises(ValueError, match="one topology epoch"):
        _integrate(stage1, changed_stage, stage1, start_version=1)
    with pytest.raises(ValueError, match="end_topology_epoch_id"):
        AcceptedConservationIntegralLedger.integrate_ssprk33(
            stage1,
            stage1,
            stage1,
            0.1,
            start_geometry_version=jnp.asarray(1),
            end_geometry_version=jnp.asarray(2),
            start_evidence_version=jnp.asarray(1),
            end_evidence_version=jnp.asarray(2),
            start_topology_epoch_id="topology:0",
            end_topology_epoch_id="topology:changed",
            start_time=jnp.asarray(0.0),
            end_time=jnp.asarray(0.1),
            accepted_step=jnp.asarray(1),
        )
    with pytest.raises(ValueError, match="cannot span a topology epoch change"):
        AcceptedConservationIntegralLedger(
            (),
            jnp.zeros((2, 1)),
            jnp.asarray([True, True]),
            geometry_family_id="geometry-family:mesh",
            geometry_layout_id="geometry-layout:mesh",
            stage_geometry_versions=(jnp.asarray(1), jnp.asarray(2), jnp.asarray(3)),
            start_geometry_version=jnp.asarray(1),
            end_geometry_version=jnp.asarray(4),
            evidence_policy_id="evidence-policy:gcl",
            stage_evidence_versions=(
                jnp.asarray(1),
                jnp.asarray(2),
                jnp.asarray(3),
            ),
            start_evidence_version=jnp.asarray(1),
            end_evidence_version=jnp.asarray(4),
            start_topology_epoch_id="topology:0",
            end_topology_epoch_id="topology:changed",
            start_time=jnp.asarray(0.0),
            end_time=jnp.asarray(0.1),
            accepted_step=jnp.asarray(1),
        )


@pytest.mark.parametrize(
    ("dt", "message"),
    [
        pytest.param(jnp.asarray([0.1]), "scalar", id="nonscalar"),
        pytest.param(0.0, "positive", id="zero"),
        pytest.param(jnp.inf, "finite values", id="nonfinite"),
    ],
)
def test_ssprk33_rejects_invalid_time_increment(dt, message):
    stage = _stage([[1.0], [2.0]], [[0.0], [0.0]])

    with pytest.raises(Exception, match=message):
        invalid = _integrate(
            stage,
            stage,
            stage,
            dt,
            start_time=0.0,
            end_time=0.1,
        )
        jax.block_until_ready(invalid.source_integral)


def test_inactive_faces_are_zeroed_before_scatter_and_accepted_integration():
    stage1 = _stage(
        [[2.0, 4.0], [1000.0, 2000.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        active=(True, False),
        neighbour=(1, 1),
        geometry_version=1,
    )
    stage2 = _stage(
        [[4.0, 8.0], [3000.0, 4000.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        active=(True, False),
        neighbour=(1, 1),
        geometry_version=2,
    )
    stage3 = _stage(
        [[8.0, 16.0], [5000.0, 6000.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        active=(True, False),
        neighbour=(1, 1),
        geometry_version=3,
    )

    np.testing.assert_array_equal(stage1.blocks[0].flux_rate[1], np.zeros(2))
    accepted = _integrate(stage1, stage2, stage3, 0.5, start_version=1)
    np.testing.assert_array_equal(accepted.blocks[0].flux_integral[1], np.zeros(2))


def test_accepted_conservation_sums_use_content_without_measure_division():
    stage = _stage(
        [[2.0, 3.0], [5.0, 7.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    )
    accepted = _integrate(stage, stage, stage, 0.5)

    source_sum, boundary_outward_sum, net_cell_sum = accepted.conservation_sums()
    np.testing.assert_allclose(source_sum, np.asarray([2.0, 3.0]))
    np.testing.assert_allclose(boundary_outward_sum, np.asarray([2.5, 3.5]))
    np.testing.assert_allclose(net_cell_sum, source_sum - boundary_outward_sum)


def test_ledger_validates_routes_components_masks_and_finiteness():
    with pytest.raises(TypeError, match="boolean dtype"):
        ConservationStageFluxRateBlock(
            jnp.ones((1, 2)),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([-1], dtype=jnp.int32),
            jnp.asarray([1], dtype=jnp.int32),
            "faces:a",
            "physical",
        )
    with pytest.raises(ValueError, match="connect a cell to itself"):
        ConservationStageFluxRateBlock(
            jnp.ones((1, 2)),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([True]),
            "faces:a",
            "physical",
        )
    with pytest.raises(Exception, match="finite values"):
        _stage([[jnp.nan], [0.0]], [[0.0], [0.0]])
    with pytest.raises(Exception, match="finite values"):
        _stage([[0.0], [0.0]], [[jnp.inf], [0.0]])


def test_static_ledger_ids_include_evidence_policy_but_exclude_dynamic_versions():
    first = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_version=1,
        evidence_version=10,
    )
    changed_dynamic_values = _stage(
        [[3.0], [4.0]],
        [[5.0], [6.0]],
        geometry_version=99,
        evidence_version=999,
    )
    changed_layout = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_layout_id="geometry-layout:other",
        geometry_version=1,
        evidence_version=10,
    )
    changed_policy = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_version=1,
        evidence_policy_id="evidence-policy:other",
        evidence_version=10,
    )
    changed_topology = _stage(
        [[1.0], [2.0]],
        [[0.0], [0.0]],
        geometry_version=1,
        evidence_version=10,
        topology_epoch_id="topology:other",
    )
    accepted1 = _integrate(
        first,
        first,
        first,
        start_version=1,
        end_version=2,
        start_evidence_version=10,
        end_evidence_version=20,
    )
    accepted2 = _integrate(
        changed_dynamic_values,
        changed_dynamic_values,
        changed_dynamic_values,
        start_version=99,
        end_version=200,
        start_evidence_version=999,
        end_evidence_version=2000,
    )
    changed_policy_accepted = _integrate(
        changed_policy,
        changed_policy,
        changed_policy,
        start_version=1,
        end_version=2,
        start_evidence_version=10,
        end_evidence_version=20,
    )

    assert first.ledger_id == changed_dynamic_values.ledger_id
    assert first.ledger_id != changed_layout.ledger_id
    assert first.ledger_id != changed_policy.ledger_id
    assert first.ledger_id != changed_topology.ledger_id
    assert accepted1.ledger_id == accepted2.ledger_id
    assert accepted1.ledger_id != changed_policy_accepted.ledger_id
    assert first.blocks[0].rate_block_id != ""
    assert accepted1.blocks[0].integral_block_id != ""


def test_empty_block_ledger_derives_concrete_shape_and_is_immutable():
    ledger = ConservationStageLedger(
        (),
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        jnp.asarray([True, True]),
        geometry_family_id="geometry-family:source-only",
        geometry_layout_id="geometry-layout:source-only",
        geometry_version=jnp.asarray(1, dtype=jnp.int32),
        evidence_policy_id="evidence-policy:gcl",
        evidence_version=jnp.asarray(1, dtype=jnp.int32),
        topology_epoch_id="topology:0",
    )

    assert ledger.cell_count == 2
    assert type(ledger.cell_count) is int
    assert ledger.component_shape == (2,)
    assert type(ledger.component_shape) is tuple
    np.testing.assert_array_equal(ledger.scatter_content_rate(), ledger.source_rate)
    with pytest.raises(AttributeError, match="cannot assign"):
        ledger.geometry_layout_id = "geometry-layout:changed"


def _amr_accepted_ledger(flux_integral, start_time, end_time, accepted_step):
    interval = end_time - start_time
    integral = np.asarray(flux_integral, dtype=np.float32)
    stage = _stage(
        integral / interval,
        np.zeros((2, 1), dtype=np.float32),
    )
    return _integrate(
        stage,
        stage,
        stage,
        interval,
        start_time=start_time,
        end_time=end_time,
        accepted_step=accepted_step,
    )


def _accepted_result(ledger, *, accepted=True, step_size_label=99.0):
    return SimpleNamespace(
        accepted=jnp.asarray(accepted),
        accepted_flux_integrals=ledger,
        accepted_step_size=jnp.asarray(step_size_label),
    )


def test_amr_reflux_consumes_one_contiguous_fine_interval_union_without_extra_dt():
    coarse = _accepted_result(_amr_accepted_ledger([[0.4], [0.8]], 2.0, 2.2, 50))
    fine = (
        _accepted_result(
            _amr_accepted_ledger([[0.05], [0.2]], 2.0, 2.1, 100),
            step_size_label=500.0,
        ),
        _accepted_result(
            _amr_accepted_ledger([[0.15], [0.4]], 2.1, 2.2, 101),
            step_size_label=700.0,
        ),
    )

    register = flux_register_from_accepted_steps(
        coarse,
        fine,
        0,
        lambda value: value,
        jnp.asarray([True, True]),
    )

    np.testing.assert_allclose(register.coarse_flux, [[0.4], [0.8]])
    np.testing.assert_allclose(register.fine_flux, [[0.2], [0.6]])
    np.testing.assert_allclose(register.accumulated_time, 0.2)


@pytest.mark.parametrize(
    ("fine_intervals", "accepted_steps", "message"),
    [
        pytest.param(
            ((2.0, 2.09), (2.1, 2.2)),
            (100, 101),
            "gap",
            id="gap",
        ),
        pytest.param(
            ((2.0, 2.11), (2.1, 2.2)),
            (100, 101),
            "overlap",
            id="overlap",
        ),
        pytest.param(
            ((2.0, 2.1), (2.1, 2.2)),
            (101, 100),
            "strictly monotone",
            id="out-of-order-step-ids",
        ),
        pytest.param(
            ((2.01, 2.1), (2.1, 2.2)),
            (100, 101),
            "coarse interval start",
            id="wrong-union",
        ),
    ],
)
def test_amr_reflux_rejects_noncontiguous_or_out_of_order_fine_intervals(
    fine_intervals, accepted_steps, message
):
    coarse = _accepted_result(_amr_accepted_ledger([[0.4], [0.8]], 2.0, 2.2, 50))
    fine = tuple(
        _accepted_result(
            _amr_accepted_ledger(
                [[0.1], [0.3]],
                start_time,
                end_time,
                accepted_step,
            )
        )
        for (start_time, end_time), accepted_step in zip(
            fine_intervals, accepted_steps, strict=True
        )
    )

    with pytest.raises(Exception, match=message):
        register = flux_register_from_accepted_steps(
            coarse,
            fine,
            0,
            lambda value: value,
            jnp.asarray([True, True]),
        )
        jax.block_until_ready(register.coarse_flux)


@pytest.mark.parametrize("failed_level", ["coarse", "fine"])
def test_amr_reflux_requires_successful_accepted_results(failed_level):
    coarse = _accepted_result(
        _amr_accepted_ledger([[0.4], [0.8]], 2.0, 2.2, 50),
        accepted=failed_level != "coarse",
    )
    fine = (
        _accepted_result(
            _amr_accepted_ledger([[0.05], [0.2]], 2.0, 2.1, 100),
            accepted=failed_level != "fine",
        ),
        _accepted_result(_amr_accepted_ledger([[0.15], [0.4]], 2.1, 2.2, 101)),
    )

    with pytest.raises(Exception, match="successful accepted"):
        register = flux_register_from_accepted_steps(
            coarse,
            fine,
            0,
            lambda value: value,
            jnp.asarray([True, True]),
        )
        jax.block_until_ready(register.coarse_flux)
