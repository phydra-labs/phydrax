#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One-sided FE/FV adaptation published as one accepted cross-owner rebind.

The coupled problem is the nonmatching FE/FV conjugate-heat coupling of
`examples/mixed_method_time_coupling.py`. At the accepted boundary t = 0.2 the
solid (P1, 4×4 → nested 8×8) or the fluid (3×3 → 6×6 cells) is refined; the
changed side's physical state crosses through its native owner route, the
interface common refinement, preconditioner, observations, and prepared coupling
graph are reprepared, and the unchanged side and consumed budgets are retained.

Reference: exp(t A) of the assembled coupled semi-discrete system on each
topology (host SciPy, hand-assembled P1 matrices) joined by the exact nested
prolongation at t = 0.2 (P1 edge-midpoint means; cell injection), independent of
the coupling runtime, the FE/FV owners, and the lifecycle transaction.
"""

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from examples import adaptive_fe_fv_rebind as ex


lc = phx.lifecycle
_UNCHANGED = {"solid": "fluid-fv", "fluid": "solid-fe"}
_REPREPARED = {
    "solid": (
        "coupling/epoch",
        "interface/route",
        "solid-fe/discretization",
        "solid-fe/factorization",
        "solid-fe/probe",
    ),
    "fluid": (
        "coupling/epoch",
        "fluid-fv/discretization",
        "fluid-fv/probe",
        "interface/route",
    ),
}
_REMAPPED = {
    "solid": ("exchange/interface-heat", "solid-fe/native"),
    "fluid": ("exchange/interface-temperature", "fluid-fv/native"),
}


@pytest.fixture(scope="module", params=("solid", "fluid"))
def rebound(request: pytest.FixtureRequest) -> tuple[ex.Side, ex.RebindRun]:
    side: ex.Side = request.param
    return side, ex.run(0.05, side)


def test_rebind_reprepares_the_changed_side_and_remaps_its_state(
    rebound: tuple[ex.Side, ex.RebindRun],
) -> None:
    side, outcome = rebound
    receipt = outcome.receipt
    assert receipt is not None and receipt.published
    assert receipt.boundary_accepted and all(receipt.transport_accepted)
    assert receipt.reprepared == _REPREPARED[side]
    assert receipt.remapped == _REMAPPED[side]
    assert receipt.invalidated == () and receipt.consumed == ()
    assert "coupling/budget" in receipt.retained
    assert all(
        item in receipt.retained
        for item in receipt.composition.entry_ids
        if item.startswith(f"{_UNCHANGED[side]}/")
        and item not in ("solid-fe/native", "fluid-fv/native")
    )
    assert receipt.composition.structure_id != receipt.source_structure_id


def test_rebind_conserves_heat_and_retains_budgets(
    rebound: tuple[ex.Side, ex.RebindRun],
) -> None:
    _, outcome = rebound
    before = sum(ex.energies(outcome.before, outcome.boundary))
    after = sum(ex.energies(outcome.after, outcome.published))
    assert abs(after - before) <= 1e-13 * abs(before)
    assert outcome.receipt is not None
    transport = outcome.receipt.transports[0]
    assert transport.source_content is not None
    assert transport.target_content is not None
    np.testing.assert_allclose(
        np.asarray(transport.target_content),
        np.asarray(transport.source_content),
        rtol=1e-14,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        outcome.published.cumulative_exchange_budget,
        outcome.boundary.cumulative_exchange_budget,
    )
    assert outcome.published.window_index == outcome.boundary.window_index
    # The whole run, across the rebind, exchanges heat only through the ledger.
    initial_model, initial = ex.build(0.05)
    start = sum(ex.energies(initial_model, initial))
    end = sum(ex.energies(outcome.after, outcome.final))
    assert abs(end - start) <= 1e-12 * abs(start)
    heat = np.asarray(outcome.final.cumulative_exchange_budget)[
        outcome.final.budget_row_ids.index("interface-heat")
    ]
    assert abs(heat[0] + heat[1]) <= 1e-13 * abs(heat[1])


def test_unchanged_side_is_retained_bitwise_and_observations_rebuilt(
    rebound: tuple[ex.Side, ex.RebindRun],
) -> None:
    side, outcome = rebound
    unchanged = _UNCHANGED[side]
    kept = ex.participant(outcome.boundary, unchanged)
    retained = ex.participant(outcome.published, unchanged)
    assert retained.native is kept.native
    assert eqx.tree_equal(retained, kept)
    # Nested exact prolongation leaves both probes' physical values unchanged,
    # although the changed side's probe was rebuilt on its new support.
    np.testing.assert_allclose(
        ex.observe(outcome.after, outcome.published),
        ex.observe(outcome.before, outcome.boundary),
        rtol=0.0,
        atol=1e-13,
    )
    probe = f"{'solid-fe' if side == 'solid' else 'fluid-fv'}/probe"
    source = ex.compose(outcome.before, outcome.boundary)
    assert outcome.receipt is not None
    published = outcome.receipt.composition
    assert source.entry(probe).structure_id != published.entry(probe).structure_id


@pytest.mark.parametrize("side", ("solid", "fluid"))
def test_rebound_run_converges_to_the_rebound_reference(side: ex.Side) -> None:
    reference = ex.reference_temperatures(side)
    errors = []
    for window_size in ex.WINDOW_SIZES:
        outcome = ex.run(window_size, side)
        computed = ex.temperatures(outcome.after, outcome.final)
        errors.append(np.max(np.abs(computed - reference)))
    rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    # Backward Euler steps and uniform-rate spending of each window's heat.
    np.testing.assert_allclose(rates, 1.0, atol=0.2)
    assert errors[-1] < 5e-3


@pytest.fixture(scope="module")
def boundary() -> tuple[ex.CoupledModel, phx.solver.coupling.CouplingState]:
    model, state = ex.build(0.05)
    accepted, _ = ex.advance(model, state, 4)
    return model, accepted


def _continued_bitwise(
    model: ex.CoupledModel, state: phx.solver.coupling.CouplingState
) -> bool:
    """The old composition keeps advancing exactly as a never-rebound run."""
    continued, _ = ex.advance(model, state, 4)
    untouched_model, initial = ex.build(0.05)
    untouched, _ = ex.advance(untouched_model, initial, 8)
    return bool(eqx.tree_equal(continued, untouched))


def test_unknown_model_state_transport_is_refused(
    boundary: tuple[ex.CoupledModel, phx.solver.coupling.CouplingState],
) -> None:
    model, state = boundary
    dependent = model._replace(fluid_model_independent=False)
    composition = ex.compose(dependent, state)
    with pytest.raises(
        ValueError, match="'fluid-fv/model-state' is stale against the structure"
    ):
        ex.stage_fluid_refinement(dependent, composition)
    assert _continued_bitwise(model, state)


def test_failed_target_preparation_leaves_old_owners_usable(
    boundary: tuple[ex.CoupledModel, phx.solver.coupling.CouplingState],
) -> None:
    model, state = boundary
    composition = ex.compose(model, state)
    with pytest.raises(ValueError, match="conserv"):
        ex.stage_solid_refinement(model, composition, fine=ex.non_nested_solid())
    assert _continued_bitwise(model, state)


def test_retaining_a_stale_observation_is_refused(
    boundary: tuple[ex.CoupledModel, phx.solver.coupling.CouplingState],
) -> None:
    model, state = boundary
    composition = ex.compose(model, state)
    _, staged = ex.stage_solid_refinement(model, composition)
    with pytest.raises(ValueError, match="'solid-fe/probe' is stale"):
        lc.CompositionRebind(
            composition,
            retain=(*staged.retained, "solid-fe/probe"),
            reprepare=tuple(
                staged.candidate.entry(item)
                for item in staged.reprepared
                if item != "solid-fe/probe"
            ),
            transports=staged.transports,
        )


def _boundary_epoch(
    model: ex.CoupledModel, state: phx.solver.coupling.CouplingState, fluid_shift: float
) -> tuple[ex.CoupledModel, tuple[lc.CompositionEntry, ...]]:
    """A same-structure epoch lowered at the boundary from (shifted) checkpoints."""
    fluid = ex.participant(state, "fluid-fv")
    states = {
        "solid-fe": ex.participant(state, "solid-fe"),
        "fluid-fv": eqx.tree_at(
            lambda item: item.native, fluid, fluid.native + fluid_shift
        ),
    }
    epoch = ex.prepare_epoch(
        model.solid,
        model.fluid,
        model.interface,
        model.participants,
        states,
        float(np.asarray(state.time)),
        model.window_size,
    )
    target = model._replace(epoch=epoch)
    return target, tuple(ex.target_entries(target).values())


def test_exchange_rederivation_is_bound_to_the_accepted_boundary(
    boundary: tuple[ex.CoupledModel, phx.solver.coupling.CouplingState],
) -> None:
    model, state = boundary
    composition = ex.compose(model, state)
    target, entries = _boundary_epoch(model, state, 0.0)
    transport = phx.solver.coupling.coupling_exchange_transport(
        composition, target.epoch, entries, "interface-heat"
    )
    assert bool(transport.successful)
    # The t0 epoch re-derives exchange values of another time.
    with pytest.raises(ValueError, match="lowered at another time"):
        phx.solver.coupling.coupling_exchange_transport(
            composition,
            model.epoch,
            tuple(ex.target_entries(model).values()),
            "interface-heat",
        )
    # A fluid native other than the accepted one on its unchanged grid.
    shifted, shifted_entries = _boundary_epoch(model, state, 1.0)
    with pytest.raises(ValueError, match="other than the accepted source state"):
        phx.solver.coupling.coupling_exchange_transport(
            composition, shifted.epoch, shifted_entries, "interface-heat"
        )


def test_coupling_revisions_distinguish_states_at_one_boundary(
    boundary: tuple[ex.CoupledModel, phx.solver.coupling.CouplingState],
) -> None:
    model, state = boundary
    source = ex.compose(model, state)
    advanced = eqx.tree_at(
        lambda item: item.cumulative_exchange_budget,
        state,
        state.cumulative_exchange_budget + 1.0,
    )
    staged = ex.compose(model, advanced)
    budget = staged.entry("coupling/budget")
    assert budget.revision_id != source.entry("coupling/budget").revision_id
    assert (
        staged.entry("exchange/interface-heat").revision_id
        == source.entry("exchange/interface-heat").revision_id
    )
    # A same-boundary numeric refresh of the ledger now stages as a new revision.
    rebind = lc.CompositionRebind(
        source,
        retain=tuple(item for item in source.entry_ids if item != "coupling/budget"),
        refresh=(budget,),
    )
    assert rebind.refreshed == ("coupling/budget",)


def test_rejected_boundary_publishes_nothing(
    boundary: tuple[ex.CoupledModel, phx.solver.coupling.CouplingState],
) -> None:
    model, state = boundary
    after, published, receipt = ex.rebind(model, state, "solid", accepted=False)
    assert not receipt.published and not receipt.boundary_accepted
    assert after is model and published is state
