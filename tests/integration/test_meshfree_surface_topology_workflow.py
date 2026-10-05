# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Meshfree epochs across committed multiregion split and merge events.

The workflow lives in ``examples/meshfree_surface_topology_events.py``; the
multiregion transaction is the physical oracle (its own conservative sheet
transfer and committed state), never reimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import meshfree_surface_topology_events as ex
from phydrax.discretization.meshfree import MeshfreeEventLineage
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    MergeProposal,
    PreparedMultiRegionSurface,
    propose_merges,
    SurfaceMergePolicy,
)


def test_merge_epoch_consumes_lineage_and_matches_the_authority_state() -> None:
    topology, state = ex.two_bubbles()
    composition = ex.surfactant_composition(topology, state)
    prepared = PreparedMultiRegionSurface(topology, state)
    search = propose_merges(
        prepared, state, SurfaceMergePolicy(("ambient",), merge_distance=0.4)
    )
    proposal = next(item for item in search.proposals if isinstance(item, MergeProposal))
    far = SurfaceMergePolicy(("ambient",), merge_distance=0.01)
    refused = apply_surface_events(
        topology, state, [MergeProposal(proposal.face_ids, far)]
    )
    # A pass that commits nothing has no physical lineage to consume.
    with pytest.raises(ValueError, match="committed multiregion pass"):
        MeshfreeEventLineage(refused)
    merged = apply_surface_events(topology, state, [proposal])
    event, receipt = ex.advance_epoch(composition, topology, state, merged)
    lineage = event.change.lineage
    assert lineage is not None and event.change.cause == "surface-event"
    assert lineage.event_kinds == ("MERGE",) and lineage.topology_changed
    assert bool(jnp.all(lineage.ccd_certified))
    assert receipt.published and receipt.failed == ()
    published = receipt.composition
    # The meshfree concentration equals the authority's committed content per area.
    np.testing.assert_allclose(
        published.value(ex.CURRENT),
        event.target.concentration(merged.state.sheet_fields[..., 0]),
        rtol=1e-12,
    )
    # Zipping films changes sheet area: content is conserved, constants cannot be.
    evidence = event.transfer.evidence
    assert evidence.conservative and evidence.nonnegative
    assert not evidence.constant_preserving and abs(evidence.obstruction_defect) > 0.0
    assert ("left", "right") in event.target.labels
    assert ("left", "right") not in event.source.labels
    for entry_id in (ex.CURRENT, *ex.HISTORIES):
        np.testing.assert_allclose(
            ex._content(published, entry_id),
            ex._content(composition, entry_id),
            rtol=1e-12,
        )
    assert published.value(ex.CLOCK) is composition.value(ex.CLOCK)


def test_merge_state_continues_through_a_second_epoch_with_a_derivative_boundary() -> (
    None
):
    metrics = ex.run_merge_workflow()
    assert metrics.published and metrics.continuation_published
    assert metrics.kinds == ("MERGE",) and metrics.topology_changed
    assert metrics.ccd_certified and metrics.wall_slots > 0
    assert not metrics.constant_preserving and abs(metrics.area_defect) > 0.0
    assert metrics.continuation_kinds == ("SPLIT",)
    assert metrics.epoch_index == 2
    assert metrics.content_drift < 1e-12 and metrics.history_drift < 1e-12
    assert metrics.authority_mismatch < 1e-12
    assert metrics.value_jvp_finite


def test_frozen_event_transfer_differentiates_values_but_not_selection() -> None:
    topology, state = ex.two_bubbles()
    composition = ex.surfactant_composition(topology, state)
    prepared = PreparedMultiRegionSurface(topology, state)
    search = propose_merges(
        prepared, state, SurfaceMergePolicy(("ambient",), merge_distance=0.4)
    )
    proposal = next(item for item in search.proposals if isinstance(item, MergeProposal))
    merged = apply_surface_events(topology, state, [proposal])
    event, receipt = ex.advance_epoch(composition, topology, state, merged)
    transition = event.transition()
    values = jnp.asarray(composition.value(ex.CURRENT))
    cotangent = jnp.linspace(-1.0, 1.0, event.target.measures.shape[0])
    _, pullback = jax.vjp(lambda v: transition.apply(v).values, values)
    np.testing.assert_allclose(
        pullback(cotangent)[0], transition.pullback(cotangent), atol=1e-12
    )
    assert bool(transition.apply(values).value_derivative_available)
    assert not merged.evidence.derivative_available
    with pytest.raises(ValueError, match="nondifferentiable"):
        receipt.require_differentiable_selection()
    with pytest.raises(ValueError, match="nondifferentiable"):
        transition.require_differentiable_topology()


def test_pinch_split_continues_meshfree_state_with_region_lineage() -> None:
    metrics = ex.run_pinch_workflow()
    assert metrics.pinched and metrics.epochs >= 1
    assert metrics.region_ids == ("core/0", "core/1", "ambient")
    assert metrics.region_parents == ("core/0<-core", "core/1<-core")
    assert metrics.removed_regions == ("core",)
    assert metrics.child_labels == ("core/0", "core/1")
    assert metrics.content_drift < 1e-11 and metrics.history_drift < 1e-11
