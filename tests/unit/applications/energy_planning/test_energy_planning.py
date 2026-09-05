# Copyright © 2026 PHYDRA, Inc. All rights reserved.
import equinox as eqx
import numpy as np
import pytest

from phydrax.applications import energy_planning as ep


def _single(
    *,
    durations=(1.0, 1.0),
    load=(0.0, 1.0),
    prices=(1.0, 10.0),
    store=None,
    chronology=None,
):
    return ep.EnergySystem(
        ep.Chronology((ep.Horizon("day", durations),))
        if chronology is None
        else chronology,
        (ep.Carrier("electricity"),),
        (ep.BalancePoint("bus", "electricity"),),
        sources=(ep.Source("grid", "bus", 20.0, marginal_cost=prices),),
        demands=(ep.Demand("load", "bus", load),),
        inventories=() if store is None else (store,),
    )


def test_unrelated_energy_basis_cannot_authorize_converter_energy_creation():
    from phydrax.units import KILOGRAM

    with pytest.raises(ValueError):
        ep.EnergySystem(
            ep.Chronology((ep.Horizon("interval", (1.0,)),)),
            (
                ep.Carrier("electricity"),
                ep.Carrier("heat"),
                ep.Carrier("unused-hydrogen", KILOGRAM, energy_content=120e6),
            ),
            (
                ep.BalancePoint("electric", "electricity"),
                ep.BalancePoint("thermal", "heat"),
            ),
            converters=(
                ep.Converter(
                    "energy-creating-device",
                    "electric",
                    "input",
                    (
                        ep.ConverterPort("electric", -1.0),
                        ep.ConverterPort("thermal", 1.1),
                    ),
                    1.0,
                ),
            ),
        )


def _solve(spec, **kwargs):
    compiled = ep.compile_energy_system(spec, **kwargs)
    solution = ep.solve_energy_system(compiled)
    assert solution.successful, (solution.native_result.status, solution.replay.failures)
    return compiled, solution


def test_energy_charge_and_discharge_capacities_are_independent():
    results = []
    # Each case activates a different physical limit, without imposing E = P * dt.
    for energy, charge, discharge in ((0.5, 3.0, 3.0), (3.0, 0.2, 3.0), (3.0, 3.0, 0.3)):
        store = ep.Inventory(
            "battery", "bus", energy, charge, discharge, (ep.InventoryBoundary("day"),)
        )
        _, result = _solve(_single(durations=(2.0, 1.0), load=(0.0, 2.0), store=store))
        charging = np.asarray(result.plan.values("inventory/battery/charge"))
        discharging = np.asarray(result.plan.values("inventory/battery/discharge"))
        inventory = np.asarray(result.plan.values("inventory/battery/state/day"))
        results.append(float(discharging[1] - charging[1]))
        assert np.max(inventory) <= energy + 2e-5
        assert np.max(charging) <= charge + 2e-5
        assert np.max(discharging) <= discharge + 2e-5
        np.testing.assert_allclose(
            np.diff(inventory), (charging - discharging) * (2.0, 1.0), atol=2e-5
        )
    np.testing.assert_allclose(results, (0.5, 0.4, 0.3), atol=2e-5)


def test_exact_storage_does_not_create_negative_price_loss_cycles():
    store = ep.Inventory(
        "battery",
        "bus",
        2.0,
        2.0,
        2.0,
        (ep.InventoryBoundary("day"),),
        charge_efficiency=0.8,
        discharge_efficiency=0.8,
        exclusive=True,
    )
    compiled, solution = _solve(
        _single(durations=(1.0,), load=(0.0,), prices=(-1.0,), store=store), exact=True
    )
    np.testing.assert_allclose(solution.plan.values("source/grid"), 0, atol=2e-5)
    assert solution.prices is None
    assert ep.fixed_integer_prices(compiled, solution).unique is False
    with pytest.raises(ValueError, match="exact"):
        ep.compile_energy_system(compiled.spec)


def test_physical_retention_and_irregular_duration_ignore_accounting_weights():
    chronology = ep.Chronology(
        (ep.Horizon("day", (0.5, 2.0), multiplicity=200, probability=0.25, year=3),),
        discount_rate=0.1,
    )
    store = ep.Inventory(
        "battery",
        "bus",
        5.0,
        0.0,
        0.0,
        (ep.InventoryBoundary("day", initial=4.0, terminal="free", target=None),),
        retention=0.81,
    )
    _, solution = _solve(_single(load=(0, 0), store=store, chronology=chronology))
    np.testing.assert_allclose(
        solution.plan.values("inventory/battery/state/day"),
        4 * np.power(0.81, (0, 0.5, 2.5)),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        chronology.objective_weight, np.asarray((0.5, 2.0)) * 50 / 1.1**3
    )


def test_fixed_free_periodic_and_linked_terminal_inventory():
    fixed = ep.Inventory(
        "battery", "bus", 2.0, 2.0, 2.0, (ep.InventoryBoundary("day", target=1.0),)
    )
    _, fixed_solution = _solve(
        _single(durations=(1,), load=(0,), prices=(2,), store=fixed)
    )
    np.testing.assert_allclose(fixed_solution.plan.values("source/grid"), (1,), atol=2e-5)
    free = eqx.tree_at(
        lambda s: s.boundaries,
        fixed,
        (ep.InventoryBoundary("day", terminal="free", target=None),),
    )
    _, free_solution = _solve(_single(durations=(1,), load=(0,), prices=(2,), store=free))
    np.testing.assert_allclose(free_solution.plan.values("source/grid"), (0,), atol=2e-5)
    periodic = eqx.tree_at(
        lambda s: s.boundaries,
        fixed,
        (ep.InventoryBoundary("day", initial=None, terminal="periodic", target=None),),
    )
    _, periodic_solution = _solve(_single(store=periodic))
    state = periodic_solution.plan.values("inventory/battery/state/day")
    np.testing.assert_allclose(state[0], state[-1], atol=2e-5)
    np.testing.assert_allclose(
        periodic_solution.plan.values("source/grid"), (1, 0), atol=2e-5
    )
    chronology = ep.Chronology(
        (
            ep.Horizon("charge", (1,), multiplicity=10),
            ep.Horizon("use", (1,), multiplicity=10),
        )
    )
    linked = ep.Inventory(
        "battery",
        "bus",
        1.0,
        1.0,
        1.0,
        (
            ep.InventoryBoundary("charge", terminal="linked", target=None, link="use"),
            ep.InventoryBoundary("use", initial=None),
        ),
    )
    _, linked_solution = _solve(_single(store=linked, chronology=chronology))
    np.testing.assert_allclose(
        linked_solution.plan.values("inventory/battery/state/charge"), (0, 1), atol=2e-5
    )
    np.testing.assert_allclose(
        linked_solution.plan.values("inventory/battery/state/use"), (1, 0), atol=2e-5
    )


def test_scenario_tree_blocks_anticipation_but_allows_recourse():
    tree = ep.ScenarioTree(
        (
            ep.ScenarioNode("root", None, 0),
            ep.ScenarioNode("low", "root", 1, 0.5),
            ep.ScenarioNode("high", "root", 1, 0.5),
        )
    )
    chronology = ep.Chronology(
        (
            ep.Horizon("low", (1, 1), probability=0.5, scenario="low"),
            ep.Horizon("high", (1, 1), probability=0.5, scenario="high"),
        ),
        scenario_tree=tree,
    )
    store = ep.Inventory(
        "battery",
        "bus",
        2.0,
        2.0,
        2.0,
        (
            ep.InventoryBoundary("low", terminal="free", target=None),
            ep.InventoryBoundary("high", terminal="free", target=None),
        ),
    )
    compiled, solution = _solve(
        _single(
            load=(0, 0, 0, 2), prices=(1, 10, 1, 10), store=store, chronology=chronology
        )
    )
    charge = np.asarray(solution.plan.values("inventory/battery/charge"))
    discharge = np.asarray(solution.plan.values("inventory/battery/discharge"))
    np.testing.assert_allclose((charge - discharge)[[0, 2]], (2, 2), atol=2e-5)
    np.testing.assert_allclose((discharge - charge)[[1, 3]], (0, 2), atol=2e-5)
    np.testing.assert_allclose(
        solution.plan.values("inventory/battery/state/low"), (0, 2, 2), atol=2e-5
    )
    np.testing.assert_allclose(
        solution.plan.values("inventory/battery/state/high"), (0, 2, 0), atol=2e-5
    )
    corrupted = eqx.tree_at(
        lambda p: p.dispatch,
        solution.plan,
        tuple(
            ep.EnergyDispatch(entry.name, entry.values.at[2].set(0))
            if entry.name == "inventory/battery/charge"
            else entry
            for entry in solution.plan.dispatch
        ),
    )
    assert any(
        "nonanticipativity" in failure
        for failure in ep.replay_energy_system(compiled.spec, corrupted).failures
    )


def test_vintage_retirement_is_distinct_from_financial_lifetime():
    chronology = ep.Chronology(
        (
            ep.Horizon("build-year", (1,), year=0),
            ep.Horizon("after-retirement", (1,), year=2),
        ),
        financial_years=(0, 1, 2),
    )
    spec = ep.EnergySystem(
        chronology,
        (ep.Carrier("electricity"),),
        (ep.BalancePoint("bus", "electricity"),),
        sources=(
            ep.Source("plant", "bus", 0),
            ep.Source("backup", "bus", 5, marginal_cost=100),
        ),
        demands=(ep.Demand("load", "bus", (1, 1)),),
        investments=(
            ep.Investment("new", "plant", "power", 0, 1, 3, maximum=1, capital_cost=90),
        ),
    )
    _, solution = _solve(spec)
    np.testing.assert_allclose(solution.plan.values("investment/new"), (1,), atol=2e-5)
    np.testing.assert_allclose(solution.plan.values("source/plant"), (1, 0), atol=2e-5)
    np.testing.assert_allclose(solution.replay.cost, 190, atol=2e-4)


def test_build_and_commitment_use_exact_native_integer_decisions():
    spec = ep.EnergySystem(
        ep.Chronology((ep.Horizon("day", (1, 1)),)),
        (ep.Carrier("electricity"),),
        (ep.BalancePoint("bus", "electricity"),),
        sources=(
            ep.Source(
                "plant",
                "bus",
                0,
                marginal_cost=1,
                minimum_fraction=0.5,
                commitment=True,
                startup_cost=2,
            ),
            ep.Source("backup", "bus", 5, marginal_cost=20),
        ),
        demands=(ep.Demand("load", "bus", (0, 1)),),
        investments=(
            ep.Investment(
                "build",
                "plant",
                "power",
                0,
                10,
                10,
                maximum=2,
                minimum_build=1,
                fixed_build_cost=3,
            ),
        ),
    )
    _, solution = _solve(spec, exact=True)
    np.testing.assert_allclose(solution.plan.values("on/plant"), (0, 1), atol=2e-5)
    np.testing.assert_allclose(solution.plan.values("startup/plant"), (0, 1), atol=2e-5)
    np.testing.assert_allclose(solution.plan.values("build/build"), (1,), atol=2e-5)
    np.testing.assert_allclose(solution.replay.cost, 6, atol=2e-4)


def test_replay_detects_corrupted_inventory_balance_and_reported_cost():
    compiled, solution = _solve(ep.electricity_heat_storage_example())
    damaged = tuple(
        ep.EnergyDispatch(entry.name, entry.values + 0.2)
        if entry.name == "inventory/heat-store/state/day"
        else entry
        for entry in solution.plan.dispatch
    )
    report = ep.replay_energy_system(
        compiled.spec, ep.EnergyPlan(damaged, solution.plan.objective + 10)
    )
    assert not report.successful
    assert "objective" in report.failures
    assert any("inventory-dynamics" in failure for failure in report.failures)
    damaged = tuple(
        ep.EnergyDispatch(entry.name, entry.values + 0.5)
        if entry.name == "source/grid-import"
        else entry
        for entry in solution.plan.dispatch
    )
    assert (
        "balance/grid"
        in ep.replay_energy_system(
            compiled.spec, ep.EnergyPlan(damaged, solution.plan.objective)
        ).failures
    )


def test_marginal_prices_remove_duration_weights_discount_and_solver_scaling():
    chronology = ep.Chronology(
        (ep.Horizon("day", (0.5, 2), multiplicity=7, probability=0.2, year=2),),
        discount_rate=0.1,
    )
    compiled, solution = _solve(
        _single(load=(1, 2), prices=(3, 5), chronology=chronology),
        scaling=ep.EnergyScaling(flow=2, balance=4, objective=11),
    )
    np.testing.assert_allclose(solution.prices.marginal_cost[0].values, (3, 5), atol=2e-5)
    changed = eqx.tree_at(lambda s: s.demands[0].rate, compiled.spec, (1.001, 2))
    refreshed = ep.refresh_energy_system(compiled, changed)
    perturbed = ep.solve_energy_system(refreshed)
    assert perturbed.successful
    expected = 0.001 * chronology.objective_weight[0] * 3
    np.testing.assert_allclose(
        perturbed.replay.cost - solution.replay.cost, expected, atol=2e-6
    )
    assert not solution.prices.unique


def test_multioutput_hydrogen_and_explicit_heat_pump_energy_closure():
    _, solution = _solve(ep.electricity_hydrogen_example())
    np.testing.assert_allclose(
        solution.plan.values("converter/fuel-cell")[1], 1, atol=2e-5
    )
    np.testing.assert_allclose(solution.plan.values("source/import")[1], 0, atol=2e-5)
    spec = ep.electricity_heat_storage_example()
    bad = ep.Converter(
        "unphysical",
        "grid",
        "input",
        (ep.ConverterPort("grid", -1), ep.ConverterPort("building", 3)),
        4,
    )
    with pytest.raises(ValueError, match="environmental"):
        ep.EnergySystem(spec.chronology, spec.carriers, spec.points, converters=(bad,))


def test_strict_balance_requires_explicit_unserved_or_spill():
    spec = ep.EnergySystem(
        ep.Chronology((ep.Horizon("day", (1,)),)),
        (ep.Carrier("electricity"),),
        (ep.BalancePoint("bus", "electricity"),),
        sources=(ep.Source("grid", "bus", 1, marginal_cost=2),),
        demands=(ep.Demand("load", "bus", (2,), unserved_cost=100, allow_unserved=True),),
    )
    _, solution = _solve(spec)
    np.testing.assert_allclose(solution.plan.values("unserved/load"), (1,), atol=2e-5)
    np.testing.assert_allclose(solution.replay.cost, 102, atol=2e-4)


def test_quadratic_dispatch_has_physical_cost_and_continuous_prices():
    spec = ep.EnergySystem(
        ep.Chronology((ep.Horizon("day", (1,)),)),
        (ep.Carrier("electricity"),),
        (ep.BalancePoint("bus", "electricity"),),
        sources=(
            ep.Source("quadratic", "bus", 5, quadratic_cost=2),
            ep.Source("linear", "bus", 5, marginal_cost=3),
        ),
        demands=(ep.Demand("load", "bus", (3,)),),
    )
    _, solution = _solve(spec)
    np.testing.assert_allclose(
        solution.plan.values("source/quadratic"), (1.5,), atol=2e-5
    )
    np.testing.assert_allclose(solution.replay.cost, 6.75, atol=2e-5)
    np.testing.assert_allclose(solution.prices.marginal_cost[0].values, (3,), atol=2e-5)


def test_shared_scenario_balance_price_uses_combined_probability():
    tree = ep.ScenarioTree(
        (
            ep.ScenarioNode("root", None, 0),
            ep.ScenarioNode("low", "root", 1, 0.25),
            ep.ScenarioNode("high", "root", 1, 0.75),
        )
    )
    chronology = ep.Chronology(
        (
            ep.Horizon("low", (2, 0.5), probability=0.25, scenario="low"),
            ep.Horizon("high", (2, 0.5), probability=0.75, scenario="high"),
        ),
        scenario_tree=tree,
    )
    _, solution = _solve(
        _single(load=(1, 1, 1, 1), prices=(3, 5, 3, 7), chronology=chronology)
    )
    np.testing.assert_allclose(
        solution.prices.marginal_cost[0].values, (3, 5, 3, 7), atol=2e-5
    )


def test_scenario_investments_cannot_operate_before_revelation():
    tree = ep.ScenarioTree(
        (
            ep.ScenarioNode("root", None, 0),
            ep.ScenarioNode("low", "root", 1, 0.5),
            ep.ScenarioNode("high", "root", 1, 0.5),
        )
    )
    chronology = ep.Chronology(
        (
            ep.Horizon("low", (1, 1), probability=0.5, scenario="low"),
            ep.Horizon("high", (1, 1), probability=0.5, scenario="high"),
        ),
        scenario_tree=tree,
    )
    spec = ep.EnergySystem(
        chronology,
        (ep.Carrier("electricity"),),
        (ep.BalancePoint("bus", "electricity"),),
        sources=(
            ep.Source("plant", "bus", 0, marginal_cost=1),
            ep.Source("backup", "bus", 5, marginal_cost=100),
        ),
        demands=(ep.Demand("load", "bus", (1, 1, 1, 1)),),
        investments=(
            ep.Investment(
                "low-build", "plant", "power", 0, 2, 2, maximum=1, scenario_node="low"
            ),
            ep.Investment(
                "high-build", "plant", "power", 0, 2, 2, maximum=1, scenario_node="high"
            ),
        ),
    )
    _, solution = _solve(spec)
    np.testing.assert_allclose(
        solution.plan.values("source/plant"), (0, 1, 0, 1), atol=2e-5
    )
    np.testing.assert_allclose(
        solution.plan.values("source/backup"), (1, 0, 1, 0), atol=2e-5
    )


def test_scenario_information_and_vintages_persist_across_years():
    tree = ep.ScenarioTree(
        (
            ep.ScenarioNode("root", None, 0),
            ep.ScenarioNode("low", "root", 1, 0.5),
            ep.ScenarioNode("high", "root", 1, 0.5),
        )
    )
    chronology = ep.Chronology(
        (
            ep.Horizon("early-low", (1,), probability=0.5, scenario="low", year=0),
            ep.Horizon("early-high", (1,), probability=0.5, scenario="high", year=0),
            ep.Horizon(
                "later-low", (1,), probability=0.5, scenario="low", year=1, stage_start=1
            ),
            ep.Horizon(
                "later-high",
                (1,),
                probability=0.5,
                scenario="high",
                year=1,
                stage_start=1,
            ),
        ),
        scenario_tree=tree,
    )
    spec = ep.EnergySystem(
        chronology,
        (ep.Carrier("electricity"),),
        (ep.BalancePoint("bus", "electricity"),),
        sources=(
            ep.Source("plant", "bus", 0, marginal_cost=1),
            ep.Source("backup", "bus", 5, marginal_cost=100),
        ),
        demands=(ep.Demand("load", "bus", (1, 1, 1, 1)),),
        investments=(
            ep.Investment(
                "high-build", "plant", "power", 1, 2, 2, maximum=1, scenario_node="high"
            ),
        ),
    )
    _, solution = _solve(spec)
    np.testing.assert_allclose(
        solution.plan.values("source/plant"), (0, 0, 0, 1), atol=2e-5
    )
    np.testing.assert_allclose(
        solution.plan.values("source/backup"), (1, 1, 1, 0), atol=2e-5
    )
