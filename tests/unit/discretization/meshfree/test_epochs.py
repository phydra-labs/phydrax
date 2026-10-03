# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import TopologyEpoch, TopologyEpochTransition
from phydrax.discretization.meshfree._capacity import MeshfreeCapacityPolicy
from phydrax.discretization.meshfree._epochs import (
    commit_meshfree_epoch,
    MeshfreeEpochChange,
    remap_live_histories,
    stage_meshfree_epoch,
)
from phydrax.discretization.meshfree._resampling import (
    SurfaceResamplingPolicy,
    SurfaceResamplingResult,
)
from phydrax.discretization.meshfree._transfer import (
    PointTransferPlan,
    PointTransferRequest,
)
from phydrax.lifecycle import (
    Composition,
    CompositionEntry,
    CompositionFacet,
    CompositionRole,
)
from phydrax.sparse import EdgeRelation


SOURCE = TopologyEpoch(0, "cloud-4", "sheet", "serial")
TARGET = TopologyEpoch(1, "cloud-3", "sheet", "serial")
OWNER = "meshfree-surface"


def _route(old: np.ndarray, new: np.ndarray, /) -> TopologyEpochTransition:
    """Exactly normalized nonnegative allocation: conservative without a solve."""
    x = (np.arange(old.size) + 0.5) / old.size
    y = (np.arange(new.size) + 0.5) / new.size
    affinity = np.exp(-((y[:, None] - x[None, :]) ** 2) / 0.1)
    coefficients = affinity * old[None, :] / (new @ affinity)[None, :]
    rows, columns = np.divmod(np.arange(new.size * old.size), old.size)
    prepared = PointTransferPlan(
        EdgeRelation(columns, rows, source_size=old.size, target_size=new.size),
        coefficients.reshape(-1),
        old,
        new,
        source_id=f"support:{old.tolist()}",
        target_id=f"support:{new.tolist()}",
        request=PointTransferRequest("conservative-positive"),
    ).prepare()
    assert prepared.admitted and prepared.evidence.provider == "none"
    return prepared.epoch_transition(SOURCE, TARGET)


def _entry(
    value: object,
    entry_id: str,
    role: CompositionRole,
    structure: str,
    *dependencies: tuple[CompositionEntry, CompositionFacet],
) -> CompositionEntry:
    return CompositionEntry(
        value,
        entry_id=entry_id,
        role=role,
        owner_id=OWNER,
        structure_id=structure,
        revision_id=f"{entry_id}@{structure}",
        semantics_id=entry_id.split("/")[0],
        dependencies=tuple(item.binding(facet) for item, facet in dependencies),
    )


def _artifacts(
    epoch: CompositionEntry, tag: str, /, *, skip: str = ""
) -> dict[str, CompositionEntry]:
    """Geometry, stencils, metric, hierarchy, coupling query and derivative plan."""
    geometry = _entry(
        jnp.zeros(3), "geometry/points", "discretization", tag, (epoch, "structure")
    )
    stencils = _entry(
        jnp.zeros(3), "stencils/weights", "discretization", tag, (geometry, "revision")
    )
    metric = _entry(
        jnp.ones(3), "metric/hodge", "discretization", tag, (stencils, "revision")
    )
    hierarchy = _entry(
        jnp.ones(2), "hierarchy/levels", "preconditioner", tag, (metric, "revision")
    )
    coupling = _entry(
        jnp.ones(1), "coupling/query", "interface-route", tag, (geometry, "revision")
    )
    derivative = _entry(
        jnp.ones(1), "derivative/plan", "prepared-graph", tag, (stencils, "revision")
    )
    items = (geometry, stencils, metric, hierarchy, coupling, derivative)
    return {item.entry_id: item for item in items if item.entry_id != skip}


MEASURES = {
    "field/current": (np.full(4, 0.25), np.full(3, 1 / 3)),
    "history/0": (np.asarray([0.2, 0.3, 0.25, 0.25]), np.asarray([0.3, 0.4, 0.3])),
    "history/1": (np.asarray([0.1, 0.4, 0.4, 0.1]), np.asarray([0.35, 0.3, 0.35])),
    "predictor/rate": (np.full(4, 0.25), np.full(3, 1 / 3)),
}


def _composition(history: Array | None = None) -> Composition:
    epoch = _entry(SOURCE, "surface/epoch", "topology", SOURCE.epoch_id)
    values = {
        "field/current": jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        "history/0": jnp.asarray([0.5, 1.5, 2.5, 3.5]) if history is None else history,
        "history/1": jnp.asarray([0.2, 1.2, 2.2, 3.2]),
        "predictor/rate": jnp.asarray([0.1, -0.1, 0.2, 0.0]),
    }
    roles: dict[str, CompositionRole] = {
        "field/current": "physical-state",
        "history/0": "history",
        "history/1": "history",
        "predictor/rate": "model-state",
    }
    states = [
        _entry(values[name], name, roles[name], SOURCE.epoch_id, (epoch, "structure"))
        for name in values
    ]
    controller = _entry(jnp.asarray(0.01), "controller/step", "optimizer-state", "scalar")
    return Composition(
        (epoch, *states, controller, *_artifacts(epoch, "e0").values()),
        boundary_id="accepted-window-7",
    )


def _target_epoch(change: MeshfreeEpochChange) -> CompositionEntry:
    return CompositionEntry(
        TARGET,
        entry_id="surface/epoch",
        role="topology",
        owner_id=OWNER,
        structure_id=TARGET.epoch_id,
        revision_id=change.change_id,
        semantics_id="surface",
    )


def _proposal() -> SurfaceResamplingResult:
    return SurfaceResamplingPolicy(maximum_fill=0.3, minimum_separation=0.1).repair(
        jnp.array([[0.0, 0.0], [0.01, 0.0], [1.0, 0.0]]),
        jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        MeshfreeCapacityPolicy((3, 4)),
        lambda p: p.at[:, 1].set(0),
        lambda p: p[:, 1],
        lambda p: (jnp.ones(p.shape[0]), jnp.ones(p.shape[0])),
    )


def _change() -> MeshfreeEpochChange:
    return MeshfreeEpochChange(
        SOURCE, TARGET, cause="sample-repair", proposal=_proposal()
    )


def _routes(skip: str = "") -> dict[str, TopologyEpochTransition]:
    return {name: _route(*pair) for name, pair in MEASURES.items() if name != skip}


def test_epoch_rebuilds_every_dependency_and_remaps_every_live_history() -> None:
    source = _composition()
    change = _change()
    candidate = stage_meshfree_epoch(
        source,
        change,
        epoch_entry="surface/epoch",
        remap=_routes(),
        reprepare=tuple(_artifacts(_target_epoch(change), "e1").values()),
    )
    receipt = commit_meshfree_epoch(candidate, accepted_boundary=True)
    assert receipt.published and receipt.failed == ()
    assert receipt.remapped == tuple(sorted(MEASURES))
    composition = receipt.composition
    assert composition.value("surface/epoch").epoch_id == TARGET.epoch_id
    for name, (old, new) in MEASURES.items():
        moved = np.asarray(composition.value(name))
        assert moved.shape == (3,)
        # Each history keeps its own measures: content is conserved per route.
        np.testing.assert_allclose(
            new @ moved, old @ np.asarray(source.value(name)), rtol=1e-13
        )
        assert composition.entry(name).structure_id == TARGET.epoch_id
    assert composition.entry("stencils/weights").structure_id == "e1"
    assert composition.value("controller/step") is source.value("controller/step")
    assert receipt.value_derivative_available
    np.testing.assert_array_less(
        np.abs(receipt.conservation_residuals), receipt.content_tolerances + 1e-300
    )
    with pytest.raises(ValueError, match="nondifferentiable"):
        receipt.require_differentiable_selection()


def test_failure_in_any_history_returns_the_source_composition_unchanged() -> None:
    source = _composition(history=jnp.asarray([0.5, jnp.nan, 2.5, 3.5]))
    change = _change()
    candidate = stage_meshfree_epoch(
        source,
        change,
        epoch_entry="surface/epoch",
        remap=_routes(),
        reprepare=tuple(_artifacts(_target_epoch(change), "e1").values()),
    )
    receipt = commit_meshfree_epoch(candidate, accepted_boundary=True)
    assert not receipt.published and receipt.failed == ("history/0",)
    assert receipt.composition is source
    assert not receipt.value_derivative_available


def test_refused_boundary_publishes_nothing() -> None:
    source = _composition()
    change = _change()
    candidate = stage_meshfree_epoch(
        source,
        change,
        epoch_entry="surface/epoch",
        remap=_routes(),
        reprepare=tuple(_artifacts(_target_epoch(change), "e1").values()),
    )
    receipt = commit_meshfree_epoch(candidate, accepted_boundary=False)
    assert not receipt.published and receipt.failed == ()
    assert receipt.composition is source


def test_every_live_history_needs_its_own_route() -> None:
    change = _change()
    with pytest.raises(ValueError, match=r"own route \['history/1'\]"):
        stage_meshfree_epoch(
            _composition(),
            change,
            epoch_entry="surface/epoch",
            remap=_routes(skip="history/1"),
            reprepare=tuple(_artifacts(_target_epoch(change), "e1").values()),
        )


def test_every_epoch_bound_artifact_needs_a_target_rebuild() -> None:
    change = _change()
    with pytest.raises(ValueError, match=r"rebuild \['metric/hodge'\]"):
        stage_meshfree_epoch(
            _composition(),
            change,
            epoch_entry="surface/epoch",
            remap=_routes(),
            reprepare=tuple(
                _artifacts(_target_epoch(change), "e1", skip="metric/hodge").values()
            ),
        )


def test_live_histories_remap_all_or_report_every_failure() -> None:
    routes = _routes()
    names = tuple(MEASURES)
    fields = [
        jnp.ones(4),
        jnp.asarray([1.0, jnp.inf, 1.0, 1.0]),
        jnp.ones(4),
        jnp.ones(4),
    ]
    remap = remap_live_histories([routes[name] for name in names], fields)
    assert not remap.successful and remap.failed == (1,)
    accepted = remap_live_histories([routes[name] for name in names], [jnp.ones(4)] * 4)
    assert accepted.successful and len(accepted.values) == 4
    later = TopologyEpoch(2, "cloud-5", "sheet", "serial")
    stray = _route(np.full(4, 0.25), np.full(3, 1 / 3))
    # Same transfer, but crossing the next epoch change instead of this one.
    crossing = TopologyEpochTransition(
        TARGET, later, stray.transfer, stray.source_measures, stray.target_measures
    )
    with pytest.raises(ValueError, match="same epoch change"):
        remap_live_histories([routes[names[0]], crossing], [jnp.ones(4)] * 2)


def test_epoch_causes_are_exclusive_and_never_manufactured_from_samples() -> None:
    proposal = _proposal()
    with pytest.raises(ValueError, match="committed multiregion lineage"):
        MeshfreeEpochChange(SOURCE, TARGET, cause="surface-event", proposal=proposal)
    unconverged = SurfaceResamplingPolicy(
        maximum_fill=0.1, minimum_separation=0.01, maximum_iterations=2
    ).repair(
        jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        jnp.array([[0.5, 0.0]]),
        MeshfreeCapacityPolicy((2,)),
        lambda p: p,
        lambda p: p[:, 1],
        lambda p: (jnp.ones(p.shape[0]), jnp.ones(p.shape[0])),
    )
    with pytest.raises(ValueError, match="converged"):
        MeshfreeEpochChange(SOURCE, TARGET, cause="sample-repair", proposal=unconverged)
    change = MeshfreeEpochChange(SOURCE, TARGET, cause="sample-repair", proposal=proposal)
    assert change.lineage is None and change.proposal_id == proposal.proposal_id


def test_frozen_remap_values_differentiate_through_every_live_history() -> None:
    routes = _routes()
    names = tuple(MEASURES)
    transitions = [routes[name] for name in names]
    rng = np.random.default_rng(11)
    fields = tuple(jnp.asarray(rng.uniform(0.5, 2.0, 4)) for _ in names)
    tangents = tuple(jnp.asarray(rng.normal(size=4)) for _ in names)

    def remap(*histories: Array) -> tuple[Array, ...]:
        return remap_live_histories(transitions, histories).values

    _, pushed = jax.jvp(remap, fields, tangents)
    step = 1e-3
    ahead = remap(*(f + step * t for f, t in zip(fields, tangents, strict=True)))
    behind = remap(*(f - step * t for f, t in zip(fields, tangents, strict=True)))
    for derivative, plus, minus in zip(pushed, ahead, behind, strict=True):
        np.testing.assert_allclose(derivative, (plus - minus) / (2 * step), rtol=1e-9)
    # Each history crosses its own route: the second and third histories have
    # distinct measures, so the same input would map differently.
    assert not np.allclose(
        transitions[1].apply(fields[0]).values, transitions[2].apply(fields[0]).values
    )
    cotangents = tuple(jnp.asarray(rng.normal(size=3)) for _ in names)
    _, vjp = jax.vjp(remap, *fields)
    pulled = vjp(cotangents)
    remapped = remap_live_histories(transitions, fields)
    assert remapped.value_derivative_available
    for automatic, published in zip(pulled, remapped.pullback(cotangents), strict=True):
        np.testing.assert_allclose(automatic, published, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        sum(jnp.vdot(c, p) for c, p in zip(cotangents, pushed, strict=True)),
        sum(jnp.vdot(q, t) for q, t in zip(pulled, tangents, strict=True)),
        rtol=1e-12,
    )
    # The Hilbert adjoint is the adjoint in each history's own measure pairings.
    for (old, new), route, field, value, adjoint in zip(
        MEASURES.values(),
        transitions,
        fields,
        cotangents,
        remapped.adjoint(cotangents),
        strict=True,
    ):
        np.testing.assert_allclose(
            jnp.vdot(new * route.apply(field).values, value),
            jnp.vdot(old * field, adjoint),
            rtol=1e-12,
        )


def test_failed_remap_publishes_nan_and_refuses_reverse_maps() -> None:
    routes = _routes()
    transitions = [routes[name] for name in MEASURES]
    fields = [
        jnp.ones(4),
        jnp.asarray([1.0, jnp.inf, 1.0, 1.0]),
        jnp.ones(4),
        jnp.ones(4),
    ]
    remap = remap_live_histories(transitions, fields)
    assert not remap.successful and not remap.value_derivative_available
    assert all(np.all(np.isnan(value)) for value in remap.values)
    _, pushed = jax.jvp(
        lambda first: remap_live_histories(transitions, [first, *fields[1:]]).values[0],
        (fields[0],),
        (jnp.ones(4),),
    )
    assert np.all(np.isnan(pushed))
    with pytest.raises(ValueError, match="no value derivative"):
        remap.pullback([jnp.ones(3)] * 4)
