# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native execution, physical replay, and explicitly conditional dual pricing."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...optim import (
    Bounds,
    ConvexProgramResult,
    LinearProgram,
    MixedIntegerResult,
    MixedIntegerSolvePolicy,
    prepare_convex_program,
    QuadraticProgram,
    solve_mixed_integer_program,
    solve_prepared_convex_program,
)
from ._compile import CompiledEnergySystem, information_keys
from ._spec import EnergySystem, profile


class EnergyDispatch(StrictModule):
    name: str = eqx.field(static=True)
    values: Array


class EnergyPlan(StrictModule):
    dispatch: tuple[EnergyDispatch, ...]
    objective: Array

    def values(self, name: str) -> Array:
        for entry in self.dispatch:
            if entry.name == name:
                return entry.values
        raise KeyError(name)


class EnergyReplay(StrictModule):
    balance_residuals: tuple[EnergyDispatch, ...]
    inventory_residuals: tuple[EnergyDispatch, ...]
    cost: Array
    emissions: Array
    objective_error: Array
    maximum_physical_violation: Array
    failures: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return not self.failures


class EnergyPrices(StrictModule):
    marginal_cost: tuple[EnergyDispatch, ...]
    weakly_active_constraints: tuple[str, ...] = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    unique: bool = eqx.field(static=True, default=False)


class EnergySolution(StrictModule):
    native_result: ConvexProgramResult | MixedIntegerResult
    plan: EnergyPlan
    replay: EnergyReplay
    prices: EnergyPrices | None

    @property
    def successful(self) -> bool:
        return bool(np.asarray(self.native_result.successful)) and self.replay.successful


def decode_energy_plan(
    compiled: CompiledEnergySystem, primal: Array, objective: Array
) -> EnergyPlan:
    if np.shape(primal) != (compiled.program.num_variables,):
        raise ValueError(
            "Primal vector shape does not match the compiled energy program."
        )
    return EnergyPlan(
        tuple(
            EnergyDispatch(
                variable.name,
                jnp.asarray(primal)[jnp.asarray(variable.indices)] * variable.scale,
            )
            for variable in compiled.variables
        ),
        jnp.asarray(objective) * compiled.scaling.objective,
    )


def replay_energy_system(
    spec: EnergySystem, plan: EnergyPlan, *, atol: float = 1e-6, rtol: float = 1e-6
) -> EnergyReplay:
    """Recompute physical laws from decoded quantities, never the compiled matrix.

    Missing/malformed fields are rejected. Finite but corrupted dispatch, mode,
    inventory, investment, or reported objective is recorded as a failure. This
    evaluator intentionally knows nothing about solver rows or scaling maps.
    """
    if atol < 0 or rtol < 0 or not np.isfinite(atol + rtol):
        raise ValueError("Replay tolerances must be finite and nonnegative.")
    n, chronology = spec.chronology.size, spec.chronology
    dt, weight, accounting = (
        chronology.physical_duration,
        chronology.objective_weight,
        chronology.accounting_weight,
    )
    values = {
        entry.name: np.asarray(entry.values, dtype=float) for entry in plan.dispatch
    }
    if len(values) != len(plan.dispatch):
        raise ValueError("Decoded dispatch names must be unique.")
    failures, violations, consumed = [], [], set()
    balances = {p.name: np.zeros(n) for p in spec.points}
    dynamics = []
    cost, emissions = 0.0, 0.0

    def get(name, size=n):
        if name not in values or values[name].shape != (size,):
            raise ValueError(
                f"Missing or malformed decoded field {name!r}; expected ({size},)."
            )
        consumed.add(name)
        value = values[name]
        if not np.all(np.isfinite(value)):
            failures.append(f"nonfinite/{name}")
            violations.append(np.inf)
        return value

    def equal(name, left, right=0.0):
        left, right = np.asarray(left), np.asarray(right)
        residual = np.abs(left - right)
        maximum = float(np.max(residual, initial=0))
        violations.append(maximum)
        if not np.all(residual <= atol + rtol * np.maximum(np.abs(left), np.abs(right))):
            failures.append(name)

    def below(name, left, right):
        left, right = np.asarray(left), np.asarray(right)
        violation = np.maximum(left - right, 0)
        violations.append(float(np.max(violation, initial=0)))
        if not np.all(violation <= atol + rtol * np.maximum(np.abs(left), np.abs(right))):
            failures.append(name)

    investment_values, budget = {}, 0.0
    for investment in spec.investments:
        amount = (
            float(get(f"investment/{investment.name}", 1)[0])
            if investment.maximum
            else 0.0
        )
        investment_values[investment.name] = amount
        below(f"investment-lower/{investment.name}", 0, amount)
        below(f"investment-upper/{investment.name}", amount, investment.maximum)
        cost += amount * investment.present_cost(chronology)
        budget += amount * investment.capital_cost
        if investment.maximum and (
            investment.minimum_build or investment.fixed_build_cost
        ):
            build = float(get(f"build/{investment.name}", 1)[0])
            equal(f"build-integrality/{investment.name}", build, np.round(build))
            below(f"build-lower-bound/{investment.name}", 0, build)
            below(f"build-upper-bound/{investment.name}", build, 1)
            below(f"build-upper/{investment.name}", amount, build * investment.maximum)
            below(
                f"build-lower/{investment.name}", build * investment.minimum_build, amount
            )
            probability = (
                1
                if investment.scenario_node is None
                else chronology.scenario_tree.probability(investment.scenario_node)
            )
            cost += (
                build
                * investment.fixed_build_cost
                * chronology.discount(investment.year)
                * probability
            )
            budget += build * investment.fixed_build_cost
    if spec.policy.investment_budget is not None:
        below("policy/investment-budget", budget, spec.policy.investment_budget)

    def capacities(asset, dimension, base, *, state=False):
        return {
            h.name: np.asarray(
                [
                    base
                    + sum(
                        i.existing_capacity + investment_values[i.name]
                        for i in spec.investments
                        if i.asset == asset.name
                        and i.dimension == dimension
                        and i.active(
                            h, chronology, max(0, h.stage_start + t - int(state))
                        )
                    )
                    for t in range(len(h.durations) + int(state))
                ]
            )
            for h in chronology.horizons
        }

    keys = information_keys(spec)

    def nonanticipative(name, data, group_keys=keys):
        groups = {}
        for key, value in zip(group_keys, data, strict=True):
            if key in groups:
                equal(f"nonanticipativity/{name}/{key}", value, groups[key])
            else:
                groups[key] = value

    for point in spec.points:
        spill = get(f"spill/{point.name}")
        below(f"spill-lower/{point.name}", 0, spill)
        below(
            f"spill-upper/{point.name}", spill, profile(point.spill_capacity, n, "spill")
        )
        balances[point.name] -= spill
        cost += float(np.sum(weight * spill * profile(point.spill_cost, n, "spill cost")))
        nonanticipative(f"spill/{point.name}", spill)
    for demand in spec.demands:
        rate = profile(demand.rate, n, "demand")
        unserved = get(f"unserved/{demand.name}")
        below(f"unserved-lower/{demand.name}", 0, unserved)
        below(
            f"unserved-upper/{demand.name}",
            unserved,
            rate if demand.allow_unserved else 0,
        )
        balances[demand.point] += unserved - rate
        cost += float(
            np.sum(weight * unserved * profile(demand.unserved_cost, n, "unserved cost"))
        )
        nonanticipative(f"unserved/{demand.name}", unserved)
    slices = chronology.slices()
    for asset, kind in [
        *((a, "source") for a in spec.sources),
        *((a, "converter") for a in spec.converters),
    ]:
        flow = get(f"{kind}/{asset.name}")
        below(f"flow-lower/{asset.name}", 0, flow)
        nonanticipative(asset.name, flow)
        availability = (
            profile(asset.availability, n, "availability")
            if kind == "source"
            else np.ones(n)
        )
        capacity = capacities(asset, "power", asset.capacity)
        on = startup = None
        if asset.commitment:
            on, startup = get(f"on/{asset.name}"), get(f"startup/{asset.name}")
            for label, data in (("on", on), ("startup", startup)):
                equal(f"integrality/{label}/{asset.name}", data, np.round(data))
                below(f"lower/{label}/{asset.name}", 0, data)
                below(f"upper/{label}/{asset.name}", data, 1)
                nonanticipative(f"{label}/{asset.name}", data)
        for h in chronology.horizons:
            region = slices[h.name]
            available = capacity[h.name] * availability[region]
            below(f"capacity/{asset.name}/{h.name}", flow[region], available)
            if on is not None:
                below(
                    f"commit-upper/{asset.name}/{h.name}",
                    flow[region],
                    available * on[region],
                )
                below(
                    f"commit-lower/{asset.name}/{h.name}",
                    available * asset.minimum_fraction * on[region],
                    flow[region],
                )
                previous = np.concatenate(([float(asset.initially_on)], on[region][:-1]))
                equal(
                    f"startup-transition/{asset.name}/{h.name}",
                    startup[region],
                    np.maximum(on[region] - previous, 0),
                )
                cost += (
                    float(np.sum(startup[region]))
                    * asset.startup_cost
                    * h.multiplicity
                    * h.probability
                    * chronology.discount(h.year)
                )
        emission_rate = profile(asset.emissions, n, "emissions") * flow
        emissions += float(np.sum(accounting * emission_rate))
        cost += float(
            np.sum(
                weight
                * (
                    profile(asset.marginal_cost, n, "marginal cost") * flow
                    + 0.5 * profile(asset.quadratic_cost, n, "quadratic cost") * flow**2
                    + spec.policy.carbon_price * emission_rate
                )
            )
        )
        if kind == "source":
            balances[asset.point] += flow
        else:
            for port in asset.ports:
                balances[port.point] += (
                    profile(port.coefficient, n, "port coefficient") * flow
                )

    for store in spec.inventories:
        charge, discharge = (
            get(f"inventory/{store.name}/charge"),
            get(f"inventory/{store.name}/discharge"),
        )
        below(f"charge-lower/{store.name}", 0, charge)
        below(f"discharge-lower/{store.name}", 0, discharge)
        nonanticipative(f"charge/{store.name}", charge)
        nonanticipative(f"discharge/{store.name}", discharge)
        ecap = capacities(store, "energy", store.energy_capacity, state=True)
        ccap = capacities(store, "charge_power", store.charge_capacity)
        dcap = capacities(store, "discharge_power", store.discharge_capacity)
        mode = None
        if store.exclusive:
            mode = get(f"mode/{store.name}")
            equal(f"mode-integrality/{store.name}", mode, np.round(mode))
            below(f"mode-lower/{store.name}", 0, mode)
            below(f"mode-upper/{store.name}", mode, 1)
            # This product is an independent physical exclusivity check, not a solver row.
            equal(f"simultaneous-cycling/{store.name}", np.minimum(charge, discharge))
            nonanticipative(f"mode/{store.name}", mode)
        states, state_values, state_keys = {}, [], []
        for h in chronology.horizons:
            region = slices[h.name]
            state = get(f"inventory/{store.name}/state/{h.name}", len(h.durations) + 1)
            states[h.name] = state
            state_values.extend(state)
            state_keys.extend(information_keys(spec, state=True, horizon=h))
            below(f"inventory-lower/{store.name}/{h.name}", 0, state)
            below(f"inventory-upper/{store.name}/{h.name}", state, ecap[h.name])
            below(f"charge-capacity/{store.name}/{h.name}", charge[region], ccap[h.name])
            below(
                f"discharge-capacity/{store.name}/{h.name}",
                discharge[region],
                dcap[h.name],
            )
            if mode is not None:
                below(
                    f"charge-mode/{store.name}/{h.name}",
                    charge[region],
                    ccap[h.name] * mode[region],
                )
                below(
                    f"discharge-mode/{store.name}/{h.name}",
                    discharge[region],
                    dcap[h.name] * (1 - mode[region]),
                )
            expected = state[:-1] * store.retention ** dt[region] + dt[region] * (
                store.charge_efficiency * charge[region]
                - discharge[region] / store.discharge_efficiency
            )
            equal(f"inventory-dynamics/{store.name}/{h.name}", state[1:], expected)
            dynamics.append(
                EnergyDispatch(
                    f"{store.name}/{h.name}", jnp.asarray(state[1:] - expected)
                )
            )
        nonanticipative(f"state/{store.name}", state_values, state_keys)
        for boundary in store.boundaries:
            state = states[boundary.horizon]
            if boundary.initial is not None:
                equal(
                    f"initial/{store.name}/{boundary.horizon}", state[0], boundary.initial
                )
            if boundary.terminal == "fixed":
                equal(
                    f"terminal/{store.name}/{boundary.horizon}",
                    state[-1],
                    boundary.target,
                )
            elif boundary.terminal == "periodic":
                equal(f"terminal/{store.name}/{boundary.horizon}", state[-1], state[0])
            elif boundary.terminal == "linked":
                equal(
                    f"terminal/{store.name}/{boundary.horizon}",
                    state[-1],
                    states[boundary.link][0],
                )
        balances[store.point] += discharge - charge
        cost += float(
            np.sum(
                weight
                * (charge + discharge)
                * profile(store.throughput_cost, n, "throughput cost")
            )
        )
    for name, balance in balances.items():
        equal(f"balance/{name}", balance)
    if spec.policy.emissions_limit is not None:
        below("policy/emissions", emissions, spec.policy.emissions_limit)
    if set(values) != consumed:
        raise ValueError(f"Unexpected decoded fields: {sorted(set(values) - consumed)}.")
    physical_maximum = max(violations, default=0.0)
    equal("objective", cost, np.asarray(plan.objective))
    return EnergyReplay(
        tuple(
            EnergyDispatch(name, jnp.asarray(value)) for name, value in balances.items()
        ),
        tuple(dynamics),
        jnp.asarray(cost),
        jnp.asarray(emissions),
        jnp.asarray(cost) - plan.objective,
        jnp.asarray(physical_maximum),
        tuple(dict.fromkeys(failures)),
    )


def _prices(
    compiled: CompiledEnergySystem,
    result: ConvexProgramResult,
    *,
    fixed_integer: bool = False,
    tolerance: float = 1e-7,
) -> EnergyPrices:
    if not bool(np.asarray(result.successful)):
        raise ValueError("Prices require a successful audited continuous solve.")
    weights = compiled.spec.chronology.objective_weight
    balance_rows = [row for row in compiled.rows if row.name.startswith("balance/")]
    row_weights = {}
    for row in balance_rows:
        t = int(row.name.rsplit("/", 1)[1])
        row_weights[row.index] = row_weights.get(row.index, 0.0) + weights[t]
    prices = {
        point.name: np.zeros(compiled.spec.chronology.size)
        for point in compiled.spec.points
    }
    dual = np.asarray(result.equality_dual)
    for row in balance_rows:
        _, point, t = row.name.split("/")
        prices[point][int(t)] = (
            -dual[row.index]
            * compiled.scaling.objective
            / row.scale
            / row_weights[row.index]
        )
    weak = []
    slacks, multipliers = (
        np.asarray(result.inequality_slack),
        np.asarray(result.inequality_dual),
    )
    for row in compiled.rows:
        if (
            not row.equality
            and abs(slacks[row.index]) <= tolerance
            and abs(multipliers[row.index]) <= tolerance
        ):
            weak.append(row.name)
    x = np.asarray(result.primal)
    for name, bounds, bound_duals in (
        ("lower", compiled.program.lower_bounds, result.lower_bound_dual),
        ("upper", compiled.program.upper_bounds, result.upper_bound_dual),
    ):
        for index in np.flatnonzero(
            (np.abs(x - np.asarray(bounds)) <= tolerance)
            & (np.abs(np.asarray(bound_duals)) <= tolerance)
        ):
            weak.append(f"{name}-bound/{index}")
    interpretation = (
        "fixed-integer conditional continuous marginal values; not MIP duals"
        if fixed_integer
        else "selected continuous balance subgradients; uniqueness and differentiability are not certified"
    )
    return EnergyPrices(
        tuple(EnergyDispatch(name, jnp.asarray(value)) for name, value in prices.items()),
        tuple(weak),
        interpretation,
    )


def solve_energy_system(
    compiled: CompiledEnergySystem,
    *,
    mixed_integer_policy: MixedIntegerSolvePolicy | None = None,
    replay_atol: float = 1e-6,
    replay_rtol: float = 1e-6,
) -> EnergySolution:
    if compiled.binary_indices:
        policy = (
            MixedIntegerSolvePolicy(compiled.prepared.plan.policy)
            if mixed_integer_policy is None
            else mixed_integer_policy
        )
        result = solve_mixed_integer_program(compiled.mixed_integer_program, policy)
    else:
        if mixed_integer_policy is not None:
            raise ValueError("A mixed-integer policy requires discrete decisions.")
        result = solve_prepared_convex_program(compiled.prepared).result
    plan = decode_energy_plan(compiled, result.primal, result.objective)
    replay = replay_energy_system(compiled.spec, plan, atol=replay_atol, rtol=replay_rtol)
    # An incumbent relaxation's dual is not a mixed-integer price. Nor do
    # numerically optimal but physically invalid continuous plans get prices.
    prices = (
        _prices(compiled, result)
        if not compiled.binary_indices
        and replay.successful
        and bool(np.asarray(result.successful))
        else None
    )
    return EnergySolution(result, plan, replay, prices)


def fixed_integer_prices(
    compiled: CompiledEnergySystem, solution: EnergySolution
) -> EnergyPrices:
    """Solve a *separate* continuous problem with an accepted integer schedule fixed."""
    if (
        not compiled.binary_indices
        or not isinstance(solution.native_result, MixedIntegerResult)
        or not solution.successful
    ):
        raise ValueError(
            "Fixed-integer pricing requires a successful mixed-integer energy solution."
        )
    # The supplied incumbent must pass this specification's physical replay too.
    if not replay_energy_system(compiled.spec, solution.plan).successful:
        raise ValueError("The incumbent does not satisfy this energy specification.")
    program = compiled.program
    lower, upper = np.array(program.lower_bounds), np.array(program.upper_bounds)
    indices = np.asarray(compiled.binary_indices)
    integer_values = {}
    discrete = set(compiled.binary_indices)
    for variable in compiled.variables:
        if discrete.intersection(variable.indices):
            values = np.asarray(solution.plan.values(variable.name)) / variable.scale
            for index, value in zip(variable.indices, values, strict=True):
                if index in discrete:
                    integer_values[index] = value
    fixed = np.round([integer_values[index] for index in indices])
    lower[indices], upper[indices] = fixed, fixed
    neq = (
        program.num_equalities
        if isinstance(program, LinearProgram)
        else program.num_user_equalities
    )
    nineq = (
        program.num_inequalities
        if isinstance(program, LinearProgram)
        else program.num_user_inequalities
    )
    kwargs = dict(
        equality_matrix=program.equality_matrix[:neq],
        equality_rhs=program.equality_rhs[:neq],
        inequality_matrix=program.inequality_matrix[:nineq],
        inequality_rhs=program.inequality_rhs[:nineq],
        bounds=Bounds(jnp.asarray(lower), jnp.asarray(upper)),
        problem_id="energy-system-fixed-integer-pricing",
    )
    priced = (
        LinearProgram(program.linear, **kwargs)
        if isinstance(program, LinearProgram)
        else QuadraticProgram(
            program.quadratic, program.linear, convexity_evidence="construction", **kwargs
        )
    )
    result = solve_prepared_convex_program(
        prepare_convex_program(priced, compiled.prepared.plan.policy)
    ).result
    priced_plan = decode_energy_plan(compiled, result.primal, result.objective)
    if not replay_energy_system(compiled.spec, priced_plan).successful:
        raise ValueError(
            "The fixed-integer continuous pricing solve failed physical replay."
        )
    return _prices(compiled, result, fixed_integer=True)
