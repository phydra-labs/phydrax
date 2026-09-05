# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host lowering to native, prepared LP/QP and bounded mixed-integer programs."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._strict import StrictModule
from ...optim import (
    Bounds,
    ConvexSolvePolicy,
    LinearProgram,
    MixedIntegerProgram,
    prepare_convex_program,
    PreparedConvexProgram,
    QuadraticProgram,
    refresh_convex_program,
)
from ._spec import _positive, EnergySystem, Horizon, profile


class EnergyScaling(StrictModule):
    """Physical x = variable_scale * solver x; physical cost = objective * solver cost."""

    flow: float = 1.0
    inventory: float = 1.0
    capacity: float = 1.0
    balance: float = 1.0
    dynamics: float = 1.0
    objective: float = 1.0

    def __check_init__(self):
        for value in (
            self.flow,
            self.inventory,
            self.capacity,
            self.balance,
            self.dynamics,
            self.objective,
        ):
            _positive(value, "scaling")


class EnergyVariable(StrictModule):
    name: str = eqx.field(static=True)
    indices: tuple[int, ...] = eqx.field(static=True)
    scale: float = eqx.field(static=True)


class EnergyRow(StrictModule):
    name: str = eqx.field(static=True)
    index: int = eqx.field(static=True)
    equality: bool = eqx.field(static=True)
    scale: float = eqx.field(static=True)


class CompiledEnergySystem(StrictModule):
    spec: EnergySystem
    program: LinearProgram | QuadraticProgram
    prepared: PreparedConvexProgram
    variables: tuple[EnergyVariable, ...] = eqx.field(static=True)
    rows: tuple[EnergyRow, ...] = eqx.field(static=True)
    binary_indices: tuple[int, ...] = eqx.field(static=True)
    scaling: EnergyScaling
    exact: bool = eqx.field(static=True)

    @property
    def mixed_integer_program(self) -> MixedIntegerProgram:
        if not self.binary_indices:
            raise ValueError("This compilation contains no discrete decisions.")
        return MixedIntegerProgram(
            self.program, binary_indices=self.binary_indices, program_id="energy-system"
        )


def information_keys(
    spec: EnergySystem, *, state: bool = False, horizon: Horizon | None = None
) -> tuple[Any, ...]:
    """Keys for decisions made with the same observed history, not future leaves."""
    tree = spec.chronology.scenario_tree
    result = []
    for h in spec.chronology.horizons if horizon is None else (horizon,):
        path = None if tree is None else tree.ancestors(h.scenario)
        for t in range(len(h.durations) + int(state)):
            if path is None:
                result.append((h.name, t))
            else:
                stage = max(0, h.stage_start + t - int(state))
                known = path[min(stage, len(path) - 1)]
                result.append((h.year, h.representative, known, t))
    return tuple(result)


class _Builder:
    def __init__(self, scaling):
        self.scaling = scaling
        self.lower, self.upper, self.linear, self.quadratic = [], [], [], []
        self.variables, self.rows, self.binary = [], [], []
        self.equalities, self.inequalities = [], []
        self.equality_lookup = {}
        self.coordinate_scale = []
        self.shared = {}

    def variable(
        self,
        name,
        size,
        *,
        scale=1.0,
        lower=0.0,
        upper=np.inf,
        cost=0.0,
        quadratic=0.0,
        binary=False,
        keys=None,
        namespace=None,
    ):
        lo, hi, c, q = [
            np.broadcast_to(np.asarray(value), (size,))
            for value in (lower, upper, cost, quadratic)
        ]
        indices = []
        for i in range(size):
            key = None if keys is None else (namespace or name, keys[i])
            if key is not None and key in self.shared:
                index = self.shared[key]
                self.lower[index] = max(self.lower[index], float(lo[i]) / scale)
                self.upper[index] = min(self.upper[index], float(hi[i]) / scale)
                self.linear[index] += float(c[i]) * scale / self.scaling.objective
                self.quadratic[index] += float(q[i]) * scale**2 / self.scaling.objective
            else:
                index = len(self.lower)
                self.lower.append(float(lo[i]) / scale)
                self.upper.append(float(hi[i]) / scale)
                self.linear.append(float(c[i]) * scale / self.scaling.objective)
                self.quadratic.append(float(q[i]) * scale**2 / self.scaling.objective)
                self.coordinate_scale.append(scale)
                if key is not None:
                    self.shared[key] = index
                if binary:
                    self.binary.append(index)
            indices.append(index)
        self.variables.append(EnergyVariable(name, tuple(indices), scale))
        return np.asarray(indices, dtype=int)

    def row(self, name, terms, rhs=0.0, *, equality=False, scale=1.0):
        combined = {}
        for index, coefficient in terms:
            combined[int(index)] = (
                combined.get(int(index), 0.0)
                + float(coefficient) * self.coordinate_scale[index] / scale
            )
        combined = {i: v for i, v in combined.items() if v != 0}
        key = (tuple(sorted(combined.items())), float(rhs) / scale)
        if equality and key in self.equality_lookup:
            index = self.equality_lookup[key]
        else:
            rows = self.equalities if equality else self.inequalities
            index = len(rows)
            rows.append((combined, float(rhs) / scale))
            if equality:
                self.equality_lookup[key] = index
        self.rows.append(EnergyRow(name, index, equality, scale))

    def program(self):
        n = len(self.lower)

        def matrix(rows):
            a, b = np.zeros((len(rows), n)), np.zeros(len(rows))
            for i, (terms, rhs) in enumerate(rows):
                for j, value in terms.items():
                    a[i, j] = value
                b[i] = rhs
            return jnp.asarray(a), jnp.asarray(b)

        a, b = matrix(self.equalities)
        g, h = matrix(self.inequalities)
        kwargs = dict(
            equality_matrix=a,
            equality_rhs=b,
            inequality_matrix=g,
            inequality_rhs=h,
            bounds=Bounds(jnp.asarray(self.lower), jnp.asarray(self.upper)),
            problem_id="energy-system",
        )
        c = jnp.asarray(self.linear)
        if any(value != 0 for value in self.quadratic):
            return QuadraticProgram(
                jnp.diag(jnp.asarray(self.quadratic)),
                c,
                convexity_evidence="construction",
                **kwargs,
            )
        return LinearProgram(c, **kwargs)


def _lower(spec: EnergySystem, exact: bool, scaling: EnergyScaling):
    # Revalidate replaced Equinox specifications as well as ordinarily constructed ones.
    spec.__check_init__()
    chronology, n = spec.chronology, spec.chronology.size
    dt, weight = chronology.physical_duration, chronology.objective_weight
    accounting = chronology.accounting_weight
    keys, slices = information_keys(spec), chronology.slices()
    b = _Builder(scaling)
    investment_vars = {}
    budget_terms = []
    for investment in spec.investments:
        if investment.maximum <= 0:
            continue
        index = b.variable(
            f"investment/{investment.name}",
            1,
            scale=scaling.capacity,
            upper=investment.maximum,
            cost=investment.present_cost(chronology),
        )[0]
        investment_vars[investment.name] = index
        budget_terms.append((index, investment.capital_cost))
        if investment.minimum_build or investment.fixed_build_cost:
            if not exact:
                raise ValueError(
                    "Fixed-cost/minimum-build investments require exact=True, not a silent relaxation."
                )
            probability = (
                1
                if investment.scenario_node is None
                else chronology.scenario_tree.probability(investment.scenario_node)
            )
            build = b.variable(
                f"build/{investment.name}",
                1,
                upper=1,
                cost=investment.fixed_build_cost
                * chronology.discount(investment.year)
                * probability,
                binary=True,
            )[0]
            budget_terms.append((build, investment.fixed_build_cost))
            b.row(
                f"build-upper/{investment.name}",
                ((index, 1), (build, -investment.maximum)),
            )
            b.row(
                f"build-lower/{investment.name}",
                ((index, -1), (build, investment.minimum_build)),
            )
    if spec.policy.investment_budget is not None:
        b.row("policy/investment-budget", budget_terms, spec.policy.investment_budget)

    def capacity(asset, dimension, base, h, stage):
        fixed, terms, maximum = base, [], base
        for investment in spec.investments:
            if (
                investment.asset == asset.name
                and investment.dimension == dimension
                and investment.active(h, chronology, stage)
            ):
                fixed += investment.existing_capacity
                maximum += investment.existing_capacity + investment.maximum
                if investment.maximum:
                    terms.append((investment_vars[investment.name], -1))
        return fixed, terms, maximum

    balances = {point.name: [[] for _ in range(n)] for point in spec.points}
    demand_rhs = {point.name: np.zeros(n) for point in spec.points}
    emission_terms = []
    for point in spec.points:
        spill = b.variable(
            f"spill/{point.name}",
            n,
            scale=scaling.flow,
            upper=profile(point.spill_capacity, n, "spill"),
            cost=weight * profile(point.spill_cost, n, "spill cost"),
            keys=keys,
        )
        for t in range(n):
            balances[point.name][t].append((spill[t], -1))
    for demand in spec.demands:
        rate = profile(demand.rate, n, "demand")
        demand_rhs[demand.point] += rate
        unserved = b.variable(
            f"unserved/{demand.name}",
            n,
            scale=scaling.flow,
            upper=rate if demand.allow_unserved else 0,
            cost=weight * profile(demand.unserved_cost, n, "unserved cost"),
            keys=keys,
        )
        for t in range(n):
            balances[demand.point][t].append((unserved[t], 1))

    for asset, kind in [
        *((a, "source") for a in spec.sources),
        *((a, "converter") for a in spec.converters),
    ]:
        availability = (
            profile(asset.availability, n, "availability")
            if kind == "source"
            else np.ones(n)
        )
        emissions = profile(asset.emissions, n, "emissions")
        flow = b.variable(
            f"{kind}/{asset.name}",
            n,
            scale=scaling.flow,
            cost=weight
            * (
                profile(asset.marginal_cost, n, "cost")
                + spec.policy.carbon_price * emissions
            ),
            quadratic=weight * profile(asset.quadratic_cost, n, "quadratic cost"),
            keys=keys,
        )
        on = start = None
        if asset.commitment:
            if not exact:
                raise ValueError(
                    "Commitment requires exact=True, not a silent continuous relaxation."
                )
            on = b.variable(f"on/{asset.name}", n, upper=1, binary=True, keys=keys)
            startup_weight = np.concatenate(
                [
                    np.full(
                        len(h.durations),
                        h.multiplicity * h.probability * chronology.discount(h.year),
                    )
                    for h in chronology.horizons
                ]
            )
            start = b.variable(
                f"startup/{asset.name}",
                n,
                upper=1,
                binary=True,
                cost=asset.startup_cost * startup_weight,
                keys=keys,
            )
        for h in chronology.horizons:
            region = slices[h.name]
            for t in range(region.start, region.stop):
                fixed, terms, maximum = capacity(
                    asset, "power", asset.capacity, h, h.stage_start + t - region.start
                )
                av = availability[t]
                b.row(
                    f"capacity/{asset.name}/{t}",
                    [(flow[t], 1), *((i, v * av) for i, v in terms)],
                    fixed * av,
                )
                if on is not None:
                    b.row(
                        f"commit-upper/{asset.name}/{t}",
                        ((flow[t], 1), (on[t], -maximum * av)),
                    )
                    b.row(
                        f"commit-lower/{asset.name}/{t}",
                        [
                            (flow[t], -1),
                            (on[t], maximum * av),
                            *((i, -v * asset.minimum_fraction * av) for i, v in terms),
                        ],
                        (maximum - asset.minimum_fraction * fixed) * av,
                    )
                    previous = None if t == region.start else on[t - 1]
                    previous_on = float(asset.initially_on) if previous is None else 0.0
                    b.row(
                        f"startup-lower/{asset.name}/{t}",
                        [
                            (on[t], 1),
                            (start[t], -1),
                            *([] if previous is None else [(previous, -1)]),
                        ],
                        previous_on,
                    )
                    b.row(f"startup-on/{asset.name}/{t}", ((start[t], 1), (on[t], -1)))
                    b.row(
                        f"startup-off/{asset.name}/{t}",
                        [(start[t], 1), *([] if previous is None else [(previous, 1)])],
                        1 - previous_on,
                    )
        for t in range(n):
            emission_terms.append((flow[t], accounting[t] * emissions[t]))
            if kind == "source":
                balances[asset.point][t].append((flow[t], 1))
        if kind == "converter":
            for port in asset.ports:
                coefficients = profile(port.coefficient, n, "port coefficient")
                for t in range(n):
                    balances[port.point][t].append((flow[t], coefficients[t]))

    for store in spec.inventories:
        throughput = weight * profile(store.throughput_cost, n, "throughput cost")
        charge = b.variable(
            f"inventory/{store.name}/charge",
            n,
            scale=scaling.flow,
            cost=throughput,
            keys=keys,
        )
        discharge = b.variable(
            f"inventory/{store.name}/discharge",
            n,
            scale=scaling.flow,
            cost=throughput,
            keys=keys,
        )
        mode = None
        if store.exclusive:
            if not exact:
                raise ValueError("Exclusive storage modes require exact=True.")
            mode = b.variable(f"mode/{store.name}", n, upper=1, binary=True, keys=keys)
        states = {}
        for h in chronology.horizons:
            region = slices[h.name]
            boundary = next(
                boundary for boundary in store.boundaries if boundary.horizon == h.name
            )
            lower, upper = (
                np.zeros(len(h.durations) + 1),
                np.full(len(h.durations) + 1, np.inf),
            )
            if boundary.initial is not None:
                lower[0] = upper[0] = boundary.initial
            if boundary.terminal == "fixed":
                lower[-1] = upper[-1] = boundary.target
            state = b.variable(
                f"inventory/{store.name}/state/{h.name}",
                len(h.durations) + 1,
                scale=scaling.inventory,
                lower=lower,
                upper=upper,
                keys=information_keys(spec, state=True, horizon=h),
                namespace=f"state/{store.name}",
            )
            states[h.name] = state
            for t, index in enumerate(state):
                fixed_e, terms_e, _ = capacity(
                    store,
                    "energy",
                    store.energy_capacity,
                    h,
                    max(0, h.stage_start + t - 1),
                )
                b.row(
                    f"inventory-capacity/{store.name}/{h.name}/{t}",
                    [(index, 1), *terms_e],
                    fixed_e,
                    scale=scaling.dynamics,
                )
            for local, t in enumerate(range(region.start, region.stop)):
                fixed_c, terms_c, max_c = capacity(
                    store, "charge_power", store.charge_capacity, h, h.stage_start + local
                )
                fixed_d, terms_d, max_d = capacity(
                    store,
                    "discharge_power",
                    store.discharge_capacity,
                    h,
                    h.stage_start + local,
                )
                b.row(
                    f"charge-capacity/{store.name}/{t}",
                    [(charge[t], 1), *terms_c],
                    fixed_c,
                    scale=scaling.balance,
                )
                b.row(
                    f"discharge-capacity/{store.name}/{t}",
                    [(discharge[t], 1), *terms_d],
                    fixed_d,
                    scale=scaling.balance,
                )
                b.row(
                    f"inventory-dynamics/{store.name}/{h.name}/{local}",
                    (
                        (state[local + 1], 1),
                        (state[local], -(store.retention ** dt[t])),
                        (charge[t], -dt[t] * store.charge_efficiency),
                        (discharge[t], dt[t] / store.discharge_efficiency),
                    ),
                    equality=True,
                    scale=scaling.dynamics,
                )
                balances[store.point][t].extend(((charge[t], -1), (discharge[t], 1)))
                if mode is not None:
                    b.row(
                        f"charge-mode/{store.name}/{t}",
                        ((charge[t], 1), (mode[t], -max_c)),
                    )
                    b.row(
                        f"discharge-mode/{store.name}/{t}",
                        ((discharge[t], 1), (mode[t], max_d)),
                        max_d,
                    )
        for boundary in store.boundaries:
            if boundary.terminal in ("periodic", "linked"):
                other = (
                    boundary.horizon if boundary.terminal == "periodic" else boundary.link
                )
                b.row(
                    f"terminal/{store.name}/{boundary.horizon}",
                    ((states[boundary.horizon][-1], 1), (states[other][0], -1)),
                    equality=True,
                    scale=scaling.dynamics,
                )
    for point in spec.points:
        for t in range(n):
            b.row(
                f"balance/{point.name}/{t}",
                balances[point.name][t],
                demand_rhs[point.name][t],
                equality=True,
                scale=scaling.balance,
            )
    if spec.policy.emissions_limit is not None:
        b.row("policy/emissions", emission_terms, spec.policy.emissions_limit)
    return b.program(), tuple(b.variables), tuple(b.rows), tuple(b.binary)


def compile_energy_system(
    spec: EnergySystem,
    *,
    exact: bool = False,
    scaling: EnergyScaling | None = None,
    solver_policy: ConvexSolvePolicy | None = None,
) -> CompiledEnergySystem:
    scaling = EnergyScaling() if scaling is None else scaling
    program, variables, rows, binary = _lower(spec, exact, scaling)
    prepared = prepare_convex_program(program, solver_policy)
    return CompiledEnergySystem(
        spec, program, prepared, variables, rows, binary, scaling, exact
    )


def refresh_energy_system(
    compiled: CompiledEnergySystem, spec: EnergySystem
) -> CompiledEnergySystem:
    """Refresh numeric data only; reject changes to semantic topology or bound roles."""
    program, variables, rows, binary = _lower(spec, compiled.exact, compiled.scaling)
    var_signature = lambda vs: tuple((v.name, v.indices, v.scale) for v in vs)
    row_signature = lambda rs: tuple((r.name, r.index, r.equality, r.scale) for r in rs)
    if (
        var_signature(variables) != var_signature(compiled.variables)
        or row_signature(rows) != row_signature(compiled.rows)
        or binary != compiled.binary_indices
    ):
        raise ValueError(
            "Energy refresh must preserve variable, row, and discrete-decision semantics."
        )
    prepared = refresh_convex_program(compiled.prepared, program)
    return CompiledEnergySystem(
        spec, program, prepared, variables, rows, binary, compiled.scaling, compiled.exact
    )
