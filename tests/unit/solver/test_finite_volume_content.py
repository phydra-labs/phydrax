#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._fv_precision import FiniteVolumePrecisionPolicy
from phydrax.discretization.finite_volume._flux_ledger import (
    FiniteVolumeStageFluxRateBlock,
    FiniteVolumeStageFluxRateLedger,
)
from phydrax.discretization.finite_volume._geometry_protocol import (
    lower_static_unstructured_stage_metrics,
)
from phydrax.solver._finite_volume_content import (
    apply_stage_rate_euler_update,
    FiniteVolumeConservativeContentState,
)


def _state(
    content=((2.0, -1.0), (6.0, 3.0)),
    volumes=(2.0, 3.0),
    active_cell_mask=(True, True),
    *,
    geometry_family_id="geometry-family:mesh",
    geometry_layout_id="geometry-layout:mesh",
    geometry_version=0,
    evidence_policy_id="evidence-policy:metrics",
    evidence_version=0,
    topology_epoch_id="topology:0",
    precision=None,
):
    return FiniteVolumeConservativeContentState(
        jnp.asarray(content),
        jnp.asarray(volumes),
        jnp.asarray(active_cell_mask),
        jnp.asarray(0.5),
        topology_epoch_id=topology_epoch_id,
        geometry_family_id=geometry_family_id,
        geometry_layout_id=geometry_layout_id,
        geometry_version=jnp.asarray(geometry_version),
        evidence_policy_id=evidence_policy_id,
        evidence_version=jnp.asarray(evidence_version),
        precision=FiniteVolumePrecisionPolicy() if precision is None else precision,
    )


def _source_ledger(
    rate,
    *,
    geometry_family_id="geometry-family:mesh",
    geometry_layout_id="geometry-layout:mesh",
    geometry_version=0,
    evidence_policy_id="evidence-policy:metrics",
    evidence_version=0,
    topology_epoch_id="topology:0",
    active_cell_mask=None,
):
    source_rate = jnp.asarray(rate)
    if active_cell_mask is None:
        active_cell_mask = jnp.ones((source_rate.shape[0],), dtype=bool)
    return FiniteVolumeStageFluxRateLedger(
        (),
        source_rate,
        jnp.asarray(active_cell_mask),
        geometry_family_id=geometry_family_id,
        geometry_layout_id=geometry_layout_id,
        geometry_version=jnp.asarray(geometry_version),
        evidence_policy_id=evidence_policy_id,
        evidence_version=jnp.asarray(evidence_version),
        topology_epoch_id=topology_epoch_id,
    )


def test_active_and_inactive_cell_average_content_round_trip_is_safe():
    precision = FiniteVolumePrecisionPolicy()
    average = jnp.asarray(((1.5, -2.0), (8.0, -9.0), (3.0, 0.5)))
    volumes = jnp.asarray((2.0, 0.0, 3.0))
    active_cell_mask = jnp.asarray((True, False, True))
    expected_average = jnp.asarray(((1.5, -2.0), (0.0, 0.0), (3.0, 0.5)))

    state = FiniteVolumeConservativeContentState.from_cell_average(
        average,
        volumes,
        active_cell_mask,
        jnp.asarray(1.25),
        topology_epoch_id="topology:round-trip",
        geometry_family_id="geometry-family:round-trip",
        geometry_layout_id="geometry-layout:round-trip",
        geometry_version=jnp.asarray(4, dtype=jnp.int32),
        evidence_policy_id="evidence-policy:round-trip",
        evidence_version=jnp.asarray(9, dtype=jnp.int32),
        precision=precision,
    )

    np.testing.assert_array_equal(state.active_cell_mask, active_cell_mask)
    np.testing.assert_allclose(state.cell_average(), expected_average)
    np.testing.assert_array_equal(state.cell_average()[1], jnp.zeros((2,)))
    assert bool(jnp.all(jnp.isfinite(state.cell_average())))
    np.testing.assert_allclose(
        state.conservative_content, expected_average * volumes[:, None]
    )
    np.testing.assert_array_equal(state.conservative_content[1], jnp.zeros((2,)))
    expected_integral = jnp.sum(expected_average * volumes[:, None], axis=0)
    np.testing.assert_allclose(state.volume_integral(), expected_integral)
    np.testing.assert_allclose(state.conservation_change(expected_integral), 0.0)
    np.testing.assert_allclose(state.conservation_change(state), 0.0)


def test_cell_average_changes_with_volume_while_content_stays_authoritative():
    original = _state(content=((4.0, 2.0), (3.0, 9.0)), volumes=(2.0, 3.0))
    changed_geometry = FiniteVolumeConservativeContentState(
        original.conservative_content,
        jnp.asarray((1.0, 6.0)),
        original.active_cell_mask,
        original.time,
        topology_epoch_id=original.topology_epoch_id,
        geometry_family_id=original.geometry_family_id,
        geometry_layout_id=original.geometry_layout_id,
        geometry_version=jnp.asarray(1),
        evidence_policy_id=original.evidence_policy_id,
        evidence_version=jnp.asarray(1),
        precision=original.precision,
    )

    np.testing.assert_array_equal(
        changed_geometry.conservative_content, original.conservative_content
    )
    np.testing.assert_allclose(original.cell_average(), ((2.0, 1.0), (1.0, 3.0)))
    np.testing.assert_allclose(changed_geometry.cell_average(), ((4.0, 2.0), (0.5, 1.5)))
    np.testing.assert_allclose(changed_geometry.conservation_change(original), 0.0)
    assert changed_geometry.geometry_layout_id == original.geometry_layout_id
    assert changed_geometry.geometry_family_id == original.geometry_family_id
    assert changed_geometry.evidence_policy_id == original.evidence_policy_id
    assert int(changed_geometry.geometry_version) == 1
    assert int(changed_geometry.evidence_version) == 1


def test_stage_rate_euler_update_applies_complete_rate_once_at_explicit_time():
    state = _state(content=((4.0, -2.0), (1.0, 3.0)), volumes=(2.0, 1.0))
    rate = jnp.asarray(((2.0, 4.0), (-6.0, 8.0)))
    ledger = _source_ledger(rate)

    updated = apply_stage_rate_euler_update(
        state,
        ledger,
        jnp.asarray(0.25),
        target_time=jnp.asarray(0.625),
        target_cell_volumes=jnp.asarray((2.5, 1.5)),
        target_geometry_version=jnp.asarray(1),
        target_evidence_version=jnp.asarray(3),
    )

    np.testing.assert_allclose(
        updated.conservative_content,
        state.conservative_content + 0.25 * rate,
    )
    np.testing.assert_allclose(updated.time, 0.625)
    assert float(updated.time) != float(state.time + 0.25)
    np.testing.assert_allclose(
        updated.cell_average(),
        updated.conservative_content / jnp.asarray((2.5, 1.5))[:, None],
    )
    np.testing.assert_allclose(updated.effective_cell_volumes, (2.5, 1.5))
    assert int(updated.geometry_version) == 1
    assert int(updated.evidence_version) == 3
    assert updated.geometry_family_id == state.geometry_family_id
    assert updated.geometry_layout_id == state.geometry_layout_id
    assert updated.evidence_policy_id == state.evidence_policy_id
    assert updated.topology_epoch_id == state.topology_epoch_id


def test_stage_rate_euler_update_preserves_mask_through_static_and_ale_updates():
    state = _state(
        content=((2.0, -1.0), (0.0, 0.0)),
        volumes=(2.0, 0.0),
        active_cell_mask=(True, False),
        geometry_version=5,
        evidence_version=7,
    )
    ledger = _source_ledger(
        jnp.zeros_like(state.conservative_content),
        geometry_version=5,
        evidence_version=7,
        active_cell_mask=state.active_cell_mask,
    )
    static_updated = apply_stage_rate_euler_update(
        state,
        ledger,
        jnp.asarray(0.125),
        target_time=jnp.asarray(4.0),
    )
    ale_updated = apply_stage_rate_euler_update(
        state,
        ledger,
        jnp.asarray(0.125),
        target_time=jnp.asarray(4.0),
        target_cell_volumes=jnp.asarray((2.5, 0.0)),
        target_geometry_version=jnp.asarray(6),
        target_evidence_version=jnp.asarray(9),
    )

    np.testing.assert_array_equal(
        static_updated.effective_cell_volumes, state.effective_cell_volumes
    )
    np.testing.assert_array_equal(static_updated.active_cell_mask, state.active_cell_mask)
    assert int(static_updated.geometry_version) == 5
    assert int(static_updated.evidence_version) == 7
    np.testing.assert_allclose(static_updated.time, 4.0)
    np.testing.assert_array_equal(ale_updated.active_cell_mask, state.active_cell_mask)
    np.testing.assert_array_equal(ale_updated.effective_cell_volumes, (2.5, 0.0))
    np.testing.assert_array_equal(ale_updated.cell_average()[1], jnp.zeros((2,)))
    assert int(ale_updated.geometry_version) == 6
    assert int(ale_updated.evidence_version) == 9
    assert ale_updated.geometry_family_id == state.geometry_family_id
    assert ale_updated.geometry_layout_id == state.geometry_layout_id
    assert ale_updated.evidence_policy_id == state.evidence_policy_id
    assert ale_updated.topology_epoch_id == state.topology_epoch_id


def test_stage_rate_update_requires_exact_active_cell_mask():
    state = _state(
        content=((2.0, -1.0), (0.0, 0.0)),
        volumes=(2.0, 0.0),
        active_cell_mask=(True, False),
    )
    mismatched = _source_ledger(
        jnp.zeros_like(state.conservative_content),
        active_cell_mask=(True, True),
    )

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="active-cell mask"):
        updated = apply_stage_rate_euler_update(
            state, mismatched, 0.25, target_time=jnp.asarray(0.75)
        )
        jax.block_until_ready(updated.conservative_content)


def test_stage_rate_update_conserves_internal_flux_and_includes_source_once():
    state = _state(content=((5.0, 2.0), (7.0, -1.0)))
    block = FiniteVolumeStageFluxRateBlock(
        jnp.asarray(((3.0, -2.0), (5.0, 1.0))),
        jnp.asarray((0, 1)),
        jnp.asarray((1, -1)),
        jnp.asarray((True, True)),
        "block:physical",
        "physical",
    )
    ledger = FiniteVolumeStageFluxRateLedger(
        (block,),
        jnp.asarray(((1.0, 4.0), (-2.0, 3.0))),
        state.active_cell_mask,
        geometry_family_id=state.geometry_family_id,
        geometry_layout_id=state.geometry_layout_id,
        geometry_version=state.geometry_version,
        evidence_policy_id=state.evidence_policy_id,
        evidence_version=state.evidence_version,
        topology_epoch_id=state.topology_epoch_id,
    )
    expected_rate = jnp.asarray(((-2.0, 6.0), (-4.0, 0.0)))

    updated = apply_stage_rate_euler_update(
        state,
        ledger,
        jnp.asarray(0.2),
        target_time=jnp.asarray(9.0),
    )

    np.testing.assert_allclose(ledger.scatter_content_rate(), expected_rate)
    np.testing.assert_allclose(
        updated.conservative_content,
        state.conservative_content + 0.2 * expected_rate,
    )
    np.testing.assert_allclose(
        updated.conservation_change(state),
        0.2 * jnp.sum(expected_rate, axis=0),
    )


def test_precision_policy_controls_storage_and_reduction_casts():
    precision = FiniteVolumePrecisionPolicy(
        "float32",
        reconstruction_dtype="float32",
        flux_dtype="float32",
        reduction_dtype="float64",
    )
    state = _state(precision=precision)
    ledger = _source_ledger(jnp.ones((2, 2), dtype=jnp.float64))
    updated = apply_stage_rate_euler_update(
        state, ledger, jnp.asarray(0.125), target_time=jnp.asarray(0.625)
    )

    assert state.conservative_content.dtype == jnp.float32
    assert state.cell_average().dtype == jnp.float32
    assert state.effective_cell_volumes.dtype == jnp.float64
    assert state.time.dtype == jnp.float64
    assert state.volume_integral().dtype == jnp.float64
    assert updated.conservative_content.dtype == jnp.float32
    assert updated.effective_cell_volumes.dtype == jnp.float64
    assert updated.time.dtype == jnp.float64


def test_stage_rate_update_requires_explicit_target_time():
    target_time = inspect.signature(apply_stage_rate_euler_update).parameters[
        "target_time"
    ]

    assert target_time.kind is inspect.Parameter.KEYWORD_ONLY
    assert target_time.default is inspect.Parameter.empty


def test_stage_rate_update_rejects_stale_starting_identities():
    state = _state(geometry_version=4, evidence_version=9)
    rate = jnp.zeros_like(state.conservative_content)
    geometry_family_mismatch = _source_ledger(
        rate,
        geometry_family_id="geometry-family:other",
        geometry_version=4,
        evidence_version=9,
    )
    layout_mismatch = _source_ledger(
        rate,
        geometry_layout_id="geometry-layout:other",
        geometry_version=4,
        evidence_version=9,
    )
    topology_mismatch = _source_ledger(
        rate,
        geometry_version=4,
        evidence_version=9,
        topology_epoch_id="topology:other",
    )
    geometry_version_mismatch = _source_ledger(
        rate,
        geometry_version=5,
        evidence_version=9,
    )
    evidence_policy_mismatch = _source_ledger(
        rate,
        geometry_version=4,
        evidence_policy_id="evidence-policy:stale",
        evidence_version=9,
    )
    evidence_version_mismatch = _source_ledger(
        rate,
        geometry_version=4,
        evidence_version=8,
    )

    with pytest.raises(ValueError, match="geometry family"):
        apply_stage_rate_euler_update(
            state, geometry_family_mismatch, 0.1, target_time=jnp.asarray(0.6)
        )
    with pytest.raises(ValueError, match="geometry layout"):
        apply_stage_rate_euler_update(
            state, layout_mismatch, 0.1, target_time=jnp.asarray(0.6)
        )
    with pytest.raises(ValueError, match="topology"):
        apply_stage_rate_euler_update(
            state, topology_mismatch, 0.1, target_time=jnp.asarray(0.6)
        )
    with pytest.raises(ValueError, match="evidence policy"):
        apply_stage_rate_euler_update(
            state, evidence_policy_mismatch, 0.1, target_time=jnp.asarray(0.6)
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="starting geometry version"
    ):
        updated = apply_stage_rate_euler_update(
            state,
            geometry_version_mismatch,
            0.1,
            target_time=jnp.asarray(0.6),
        )
        jax.block_until_ready(updated.conservative_content)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="starting evidence version"
    ):
        updated = apply_stage_rate_euler_update(
            state,
            evidence_version_mismatch,
            0.1,
            target_time=jnp.asarray(0.6),
        )
        jax.block_until_ready(updated.conservative_content)


def test_equal_volume_translated_geometry_rejects_stale_content_family():
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)))
    base = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        triangles=triangles,
    ).prepare()
    translated = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((4.0, -2.0), (5.0, -2.0), (5.0, -1.0), (4.0, -1.0))),
        triangles=triangles,
    ).prepare()
    base_metrics = lower_static_unstructured_stage_metrics(base)
    translated_metrics = lower_static_unstructured_stage_metrics(translated)
    np.testing.assert_allclose(
        base_metrics.effective_cell_volumes,
        translated_metrics.effective_cell_volumes,
    )
    assert base_metrics.geometry_layout_id == translated_metrics.geometry_layout_id
    assert base_metrics.geometry_family_id != translated_metrics.geometry_family_id

    state = _state(
        content=((1.0, 0.0), (1.0, 0.0)),
        volumes=base_metrics.effective_cell_volumes,
        geometry_family_id=base_metrics.geometry_family_id,
        geometry_layout_id=base_metrics.geometry_layout_id,
    )
    stale_ledger = _source_ledger(
        jnp.zeros_like(state.conservative_content),
        geometry_family_id=translated_metrics.geometry_family_id,
        geometry_layout_id=translated_metrics.geometry_layout_id,
    )
    with pytest.raises(ValueError, match="geometry family"):
        apply_stage_rate_euler_update(
            state,
            stale_ledger,
            0.1,
            target_time=jnp.asarray(0.6),
        )


def test_stage_rate_update_rejects_ledger_shape_mismatch():
    state = _state()
    wrong_shape = _source_ledger(jnp.zeros((3, 2)))

    with pytest.raises(ValueError, match="ledger.*exact state"):
        apply_stage_rate_euler_update(
            state, wrong_shape, 0.1, target_time=jnp.asarray(0.6)
        )


@pytest.mark.parametrize(
    (
        "include_volumes",
        "include_geometry_version",
        "include_evidence_version",
    ),
    (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ),
)
def test_stage_rate_update_requires_complete_ale_target_certificate(
    include_volumes,
    include_geometry_version,
    include_evidence_version,
):
    state = _state()
    ledger = _source_ledger(jnp.zeros_like(state.conservative_content))
    target = {}
    if include_volumes:
        target["target_cell_volumes"] = jnp.asarray((2.5, 3.5))
    if include_geometry_version:
        target["target_geometry_version"] = jnp.asarray(1)
    if include_evidence_version:
        target["target_evidence_version"] = jnp.asarray(1)

    with pytest.raises(ValueError, match="must be supplied together"):
        apply_stage_rate_euler_update(
            state,
            ledger,
            0.1,
            target_time=jnp.asarray(0.6),
            **target,
        )


def test_stage_rate_update_rejects_nonpositive_active_target_volumes():
    state = _state()
    ledger = _source_ledger(jnp.zeros_like(state.conservative_content))

    for volumes in ((2.0, 0.0), (2.0, -1.0)):
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError), match="strictly positive"
        ):
            apply_stage_rate_euler_update(
                state,
                ledger,
                0.1,
                target_time=jnp.asarray(0.6),
                target_cell_volumes=jnp.asarray(volumes),
                target_evidence_version=jnp.asarray(1),
                target_geometry_version=jnp.asarray(1),
            )


def test_state_validates_versions_active_mask_volumes_and_inactive_content():
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="strictly positive"):
        _state(volumes=(0.0, 3.0))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="strictly positive"):
        _state(volumes=(-1.0, 3.0))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="finite and strictly positive"
    ):
        _state(volumes=(jnp.inf, 3.0))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="finite and strictly positive"
    ):
        _state(volumes=(jnp.nan, 3.0))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="exactly zero"):
        _state(volumes=(2.0, 3.0), active_cell_mask=(True, False))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="zero conservative content"
    ):
        _state(
            content=((2.0, -1.0), (1.0, 0.0)),
            volumes=(2.0, 0.0),
            active_cell_mask=(True, False),
        )
    with pytest.raises(ValueError, match="boolean array with exact shape"):
        _state(active_cell_mask=(1, 0))
    with pytest.raises(ValueError, match="boolean array with exact shape"):
        _state(active_cell_mask=(True,))
    with pytest.raises(ValueError, match="exact shape"):
        _state(volumes=(2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="non-empty cell and component axes"):
        _state(content=(1.0, 2.0), volumes=(2.0, 3.0))
    with pytest.raises(ValueError, match="cell and component shapes"):
        _state().with_content(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="geometry_version must be scalar"):
        _state(geometry_version=(1, 2))
    with pytest.raises(TypeError, match="geometry_version must have an integer dtype"):
        _state(geometry_version=1.5)
    with pytest.raises(ValueError, match="evidence_version must be scalar"):
        _state(evidence_version=(1, 2))
    with pytest.raises(TypeError, match="evidence_version must have an integer dtype"):
        _state(evidence_version=1.5)
    with pytest.raises(ValueError, match="nonempty canonical string"):
        _state(geometry_family_id=" geometry-family")


def test_with_content_preserves_ownership_and_updates_dynamic_evidence_version():
    state = _state(
        content=((2.0, -1.0), (0.0, 0.0)),
        volumes=(2.0, 0.0),
        active_cell_mask=(True, False),
        geometry_version=3,
        evidence_version=4,
    )
    original_content = state.conservative_content
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="zero conservative content"
    ):
        state.with_content(state.conservative_content.at[1, 0].set(1.0))
    derived = state.cell_average()
    replacement_content = original_content.at[0].multiply(2.0)
    np.testing.assert_array_equal(replacement_content[1], jnp.zeros((2,)))
    replacement = state.with_content(replacement_content, evidence_version=jnp.asarray(5))

    assert "cell_average" not in vars(state)
    with pytest.raises(AttributeError):
        state.cell_average = jnp.zeros_like(derived)
    np.testing.assert_array_equal(state.conservative_content, original_content)
    np.testing.assert_allclose(replacement.cell_average(), 2.0 * derived)
    np.testing.assert_array_equal(replacement.active_cell_mask, state.active_cell_mask)
    assert replacement.geometry_family_id == state.geometry_family_id
    assert replacement.geometry_layout_id == state.geometry_layout_id
    assert replacement.evidence_policy_id == state.evidence_policy_id
    assert replacement.topology_epoch_id == state.topology_epoch_id
    assert int(replacement.geometry_version) == 3
    assert int(replacement.evidence_version) == 5


def test_cell_average_jit_and_gradient_are_finite_with_inactive_cells():
    precision = FiniteVolumePrecisionPolicy()
    volumes = jnp.asarray((2.0, 0.0, 4.0))
    active_cell_mask = jnp.asarray((True, False, True))

    def average_from_content(content):
        return FiniteVolumeConservativeContentState(
            content,
            volumes,
            active_cell_mask,
            jnp.asarray(0.5),
            topology_epoch_id="topology:jit",
            geometry_family_id="geometry-family:jit",
            geometry_layout_id="geometry-layout:jit",
            geometry_version=jnp.asarray(2),
            evidence_policy_id="evidence-policy:jit",
            evidence_version=jnp.asarray(6),
            precision=precision,
        ).cell_average()

    content = jnp.asarray(((4.0, -2.0), (0.0, 0.0), (8.0, 12.0)))
    compiled_average = jax.jit(average_from_content)(content)
    gradient = jax.jit(jax.grad(lambda value: jnp.sum(average_from_content(value) ** 2)))(
        content
    )

    assert bool(jnp.all(jnp.isfinite(compiled_average)))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_array_equal(compiled_average[1], jnp.zeros((2,)))
    np.testing.assert_array_equal(gradient[1], jnp.zeros((2,)))


def test_geometry_family_is_explicit_checkpoint_ready_static_state():
    state = _state(geometry_family_id="geometry-family:checkpoint")
    rebound = state.with_topology_epoch(
        "topology:1",
        geometry_family_id="geometry-family:successor",
    )

    assert "geometry_family_id" in vars(state)
    assert state.geometry_family_id == "geometry-family:checkpoint"
    assert rebound.geometry_family_id == "geometry-family:successor"
    assert rebound.geometry_layout_id == state.geometry_layout_id
    assert (
        inspect.signature(FiniteVolumeConservativeContentState)
        .parameters["geometry_family_id"]
        .default
        is inspect.Parameter.empty
    )


def test_dynamic_versions_reuse_one_jit_layout_and_static_fingerprints():
    trace_count = {"value": 0}

    def update_for_versions(
        geometry_version,
        evidence_version,
        target_geometry_version,
        target_evidence_version,
    ):
        trace_count["value"] += 1
        state = _state(
            geometry_version=geometry_version,
            evidence_version=evidence_version,
        )
        ledger = _source_ledger(
            jnp.ones_like(state.conservative_content),
            geometry_version=geometry_version,
            evidence_version=evidence_version,
            active_cell_mask=state.active_cell_mask,
        )
        return apply_stage_rate_euler_update(
            state,
            ledger,
            jnp.asarray(0.25),
            target_time=jnp.asarray(0.75),
            target_cell_volumes=jnp.asarray((2.5, 3.5)),
            target_geometry_version=target_geometry_version,
            target_evidence_version=target_evidence_version,
        )

    update_jit = jax.jit(update_for_versions)
    first = update_jit(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(10, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(11, dtype=jnp.int32),
    )
    second = update_jit(
        jnp.asarray(7, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
        jnp.asarray(8, dtype=jnp.int32),
        jnp.asarray(21, dtype=jnp.int32),
    )
    jax.block_until_ready(first.conservative_content)
    jax.block_until_ready(second.conservative_content)

    first_ledger = _source_ledger(jnp.zeros((2, 2)), geometry_version=1)
    second_ledger = _source_ledger(jnp.zeros((2, 2)), geometry_version=7)
    assert trace_count["value"] == 1
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)
    assert first_ledger.ledger_id == second_ledger.ledger_id
    assert first.geometry_layout_id == second.geometry_layout_id
    assert first.evidence_policy_id == second.evidence_policy_id
    assert int(first.geometry_version) == 2
    assert int(second.geometry_version) == 8
    assert int(first.evidence_version) == 11
    assert int(second.evidence_version) == 21
