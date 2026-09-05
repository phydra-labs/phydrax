# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Physical and economic records for finite energy-system planning.

Rates are in carrier amount units per chronology time unit. Inventory is absolute
carrier amount, never a state-of-charge fraction. Compilation is a host operation.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ...units import conversion_factor, ENERGY, JOULE, SECOND, TIME, UnitDefinition


def profile(value: Any, size: int, name: str, *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape not in ((), (size,)):
        raise ValueError(f"{name} must be scalar or have shape ({size},).")
    result = np.broadcast_to(array, (size,))
    if not np.all(np.isfinite(result)) or (nonnegative and np.any(result < 0)):
        raise ValueError(
            f"{name} must be finite" + (" and nonnegative." if nonnegative else ".")
        )
    return result


def _positive(value: float, name: str, *, zero: bool = False) -> None:
    if not isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}."
        )


class Carrier(StrictModule):
    name: str = eqx.field(static=True)
    unit: UnitDefinition = JOULE
    energy_content: float | None = None
    environmental: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        if not self.name or not isinstance(self.unit, UnitDefinition):
            raise ValueError(
                "A carrier requires a name and a UnitDefinition amount unit."
            )
        if self.energy_content is not None:
            _positive(self.energy_content, "energy_content")

    @property
    def joules_per_unit(self) -> float:
        if self.unit.dimension == ENERGY:
            return float(conversion_factor(self.unit, JOULE))
        if self.energy_content is None:
            raise ValueError(
                f"Carrier {self.name!r} needs energy_content in joules per amount unit."
            )
        return float(self.energy_content)


class BalancePoint(StrictModule):
    name: str = eqx.field(static=True)
    carrier: str = eqx.field(static=True)
    spill_capacity: Any = 0.0
    spill_cost: Any = 0.0


class Source(StrictModule):
    name: str = eqx.field(static=True)
    point: str = eqx.field(static=True)
    capacity: float
    marginal_cost: Any = 0.0
    availability: Any = 1.0
    emissions: Any = 0.0
    quadratic_cost: Any = 0.0
    minimum_fraction: float = 0.0
    commitment: bool = eqx.field(static=True, default=False)
    startup_cost: float = 0.0
    initially_on: bool = eqx.field(static=True, default=False)


class Demand(StrictModule):
    name: str = eqx.field(static=True)
    point: str = eqx.field(static=True)
    rate: Any
    unserved_cost: Any = 0.0
    allow_unserved: bool = eqx.field(static=True, default=False)


class InventoryBoundary(StrictModule):
    """One horizon's inventory boundary; links equate end to another start.

    A None initial amount is an explicitly optimized initial inventory. It is
    required for an incoming link or a periodic horizon, rather than silently
    pinning a representative horizon to zero.
    """

    horizon: str = eqx.field(static=True)
    initial: float | None = 0.0
    terminal: str = eqx.field(static=True, default="fixed")
    target: float | None = 0.0
    link: str | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if self.terminal not in ("fixed", "free", "periodic", "linked"):
            raise ValueError("terminal must be fixed, free, periodic, or linked.")
        if self.initial is not None:
            _positive(self.initial, "initial inventory", zero=True)
        if self.terminal == "fixed":
            if self.target is None:
                raise ValueError("A fixed terminal boundary requires a target.")
            _positive(self.target, "terminal inventory", zero=True)
        elif self.target is not None:
            raise ValueError("Only fixed terminal boundaries accept a target.")
        if (self.terminal == "linked") != (self.link is not None):
            raise ValueError("Exactly linked boundaries require a destination horizon.")
        if self.terminal == "periodic" and self.initial is not None:
            raise ValueError(
                "Periodic inventory has an optimized initial amount; use initial=None."
            )


class Inventory(StrictModule):
    name: str = eqx.field(static=True)
    point: str = eqx.field(static=True)
    energy_capacity: float
    charge_capacity: float
    discharge_capacity: float
    boundaries: tuple[InventoryBoundary, ...]
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    retention: float = 1.0
    throughput_cost: Any = 0.0
    exclusive: bool = eqx.field(static=True, default=False)


class ConverterPort(StrictModule):
    point: str = eqx.field(static=True)
    coefficient: Any


class Converter(StrictModule):
    """Activity measured in reference-carrier amount/time, with signed port rates.

    Positive ports produce; negative ports consume. Reference basis is an
    explicit input or output port with coefficient -1 or +1 respectively.
    Energy outputs may not exceed all energy inputs, including ambient heat.
    """

    name: str = eqx.field(static=True)
    reference_point: str = eqx.field(static=True)
    reference_basis: str = eqx.field(static=True)
    ports: tuple[ConverterPort, ...]
    capacity: float
    marginal_cost: Any = 0.0
    emissions: Any = 0.0
    quadratic_cost: Any = 0.0
    minimum_fraction: float = 0.0
    commitment: bool = eqx.field(static=True, default=False)
    startup_cost: float = 0.0
    initially_on: bool = eqx.field(static=True, default=False)


class ScenarioNode(StrictModule):
    name: str = eqx.field(static=True)
    parent: str | None = eqx.field(static=True)
    stage: int = eqx.field(static=True)
    conditional_probability: float = 1.0


class ScenarioTree(StrictModule):
    nodes: tuple[ScenarioNode, ...]

    def __check_init__(self):
        lookup = {node.name: node for node in self.nodes}
        if not self.nodes or len(lookup) != len(self.nodes):
            raise ValueError("Scenario node names must be nonempty and unique.")
        roots = [node for node in self.nodes if node.parent is None]
        if (
            len(roots) != 1
            or roots[0].stage != 0
            or roots[0].conditional_probability != 1
        ):
            raise ValueError(
                "A scenario tree requires one probability-one stage-zero root."
            )
        for node in self.nodes:
            if not node.name or not 0 < node.conditional_probability <= 1:
                raise ValueError("Scenario probabilities must lie in (0, 1].")
            if node.parent is not None:
                if (
                    node.parent not in lookup
                    or lookup[node.parent].stage + 1 != node.stage
                ):
                    raise ValueError(
                        "Scenario parents must precede children by one stage."
                    )
            children = [child for child in self.nodes if child.parent == node.name]
            if children and not np.isclose(
                sum(child.conditional_probability for child in children), 1.0
            ):
                raise ValueError("Conditional probabilities of siblings must sum to one.")

    def ancestors(self, name: str) -> tuple[str, ...]:
        lookup = {node.name: node for node in self.nodes}
        if name not in lookup:
            raise ValueError(f"Unknown scenario node {name!r}.")
        result = []
        current = lookup[name]
        while True:
            result.append(current.name)
            if current.parent is None:
                return tuple(reversed(result))
            current = lookup[current.parent]

    def probability(self, name: str) -> float:
        lookup = {node.name: node for node in self.nodes}
        return float(
            np.prod([lookup[key].conditional_probability for key in self.ancestors(name)])
        )


class Horizon(StrictModule):
    name: str = eqx.field(static=True)
    durations: Any
    multiplicity: float = 1.0
    probability: float = 1.0
    year: int = eqx.field(static=True, default=0)
    scenario: str | None = eqx.field(static=True, default=None)
    representative: str = eqx.field(static=True, default="chronological")
    stage_start: int = eqx.field(static=True, default=0)

    def __check_init__(self):
        values = np.asarray(self.durations)
        if (
            not self.name
            or values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                "Horizon durations must be a nonempty positive finite vector."
            )
        _positive(self.multiplicity, "representative multiplicity")
        _positive(self.probability, "scenario probability")
        if self.probability > 1:
            raise ValueError("Scenario probability may not exceed one.")
        if not isinstance(self.stage_start, int) or self.stage_start < 0:
            raise ValueError(
                "stage_start must be a nonnegative integer information stage."
            )


class Chronology(StrictModule):
    horizons: tuple[Horizon, ...]
    time_unit: UnitDefinition = SECOND
    discount_rate: float = 0.0
    base_year: int = eqx.field(static=True, default=0)
    financial_years: tuple[int, ...] = eqx.field(static=True, default=())
    scenario_tree: ScenarioTree | None = None

    def __check_init__(self):
        if not self.horizons or len({h.name for h in self.horizons}) != len(
            self.horizons
        ):
            raise ValueError("Chronology requires uniquely named horizons.")
        if self.time_unit.dimension != TIME:
            raise ValueError("Chronology requires a physical time unit.")
        _positive(self.discount_rate, "discount rate", zero=True)
        if len(set(self.financial_years)) != len(self.financial_years):
            raise ValueError("Financial years must be unique.")
        tree = self.scenario_tree
        for h in self.horizons:
            if tree is None and h.scenario is not None:
                raise ValueError("A scenario horizon requires a scenario tree.")
            if tree is not None:
                if h.scenario is None:
                    raise ValueError(
                        "Every scenario-tree horizon requires a leaf scenario."
                    )
                tree.ancestors(h.scenario)
                if any(node.parent == h.scenario for node in tree.nodes):
                    raise ValueError(
                        "Scenario horizons must name terminal leaf scenarios."
                    )
                if not np.isclose(h.probability, tree.probability(h.scenario)):
                    raise ValueError(
                        "Horizon probability must equal its scenario-tree path probability."
                    )
        if tree is not None:
            leaves = {
                node.name
                for node in tree.nodes
                if not any(child.parent == node.name for child in tree.nodes)
            }
            groups = {(h.year, h.representative) for h in self.horizons}
            for group in groups:
                members = [
                    h for h in self.horizons if (h.year, h.representative) == group
                ]
                if {h.scenario for h in members} != leaves or len(members) != len(leaves):
                    raise ValueError(
                        "Every representative/year requires exactly one horizon per scenario leaf."
                    )
                if len({h.stage_start for h in members}) != 1:
                    raise ValueError(
                        "Scenario leaves in one representative/year require the same stage_start."
                    )
                known = {}
                for h in members:
                    path = tree.ancestors(h.scenario)
                    for t, duration in enumerate(h.durations):
                        key = (path[min(h.stage_start + t, len(path) - 1)], t)
                        value = (float(duration), h.multiplicity)
                        if key in known and known[key] != value:
                            raise ValueError(
                                "Shared scenario history requires identical physical duration and multiplicity."
                            )
                        known[key] = value
            for representative in {h.representative for h in self.horizons}:
                years = sorted(
                    {h.year for h in self.horizons if h.representative == representative}
                )
                previous_end = 0
                for year in years:
                    members = [
                        h
                        for h in self.horizons
                        if h.representative == representative and h.year == year
                    ]
                    if members[0].stage_start < previous_end:
                        raise ValueError(
                            "Information stages may not reset between years; advance stage_start."
                        )
                    previous_end = max(h.stage_start + len(h.durations) for h in members)

    @property
    def size(self) -> int:
        return sum(len(h.durations) for h in self.horizons)

    @property
    def physical_duration(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(h.durations, dtype=float) for h in self.horizons]
        )

    @property
    def accounting_weight(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(h.durations) * h.multiplicity * h.probability
                for h in self.horizons
            ]
        )

    @property
    def objective_weight(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(h.durations)
                * h.multiplicity
                * h.probability
                * self.discount(h.year)
                for h in self.horizons
            ]
        )

    def discount(self, year: int) -> float:
        return (1 + self.discount_rate) ** (-(year - self.base_year))

    def slices(self) -> dict[str, slice]:
        result = {}
        start = 0
        for h in self.horizons:
            result[h.name] = slice(start, start + len(h.durations))
            start += len(h.durations)
        return result


class Investment(StrictModule):
    """A shared decision vintage; technical retirement never cancels financing.

    Existing vintages have existing_capacity and maximum=0. A scenario-node
    decision is shared by all descendants and unavailable to other branches.
    Capital costs are annualized across financial_lifetime, charged in the
    explicitly selected financial_years (or represented years when omitted).
    """

    name: str = eqx.field(static=True)
    asset: str = eqx.field(static=True)
    dimension: str = eqx.field(static=True)
    year: int = eqx.field(static=True)
    technical_lifetime: int = eqx.field(static=True)
    financial_lifetime: int = eqx.field(static=True)
    maximum: float = 0.0
    existing_capacity: float = 0.0
    capital_cost: float = 0.0
    minimum_build: float = 0.0
    fixed_build_cost: float = 0.0
    scenario_node: str | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if self.technical_lifetime < 1 or self.financial_lifetime < 1:
            raise ValueError("Technical and financial lifetimes must be positive years.")
        for name, value in (
            ("maximum", self.maximum),
            ("existing capacity", self.existing_capacity),
            ("capital cost", self.capital_cost),
            ("minimum build", self.minimum_build),
            ("fixed build cost", self.fixed_build_cost),
        ):
            _positive(value, name, zero=True)
        if self.minimum_build > self.maximum:
            raise ValueError("Minimum build may not exceed maximum build.")
        if self.existing_capacity and (
            self.maximum or self.capital_cost or self.fixed_build_cost
        ):
            raise ValueError(
                "An existing vintage is sunk capacity; use a separate investment for new builds."
            )

    def active(
        self, horizon: Horizon, chronology: Chronology, stage: int | None = None
    ) -> bool:
        if not self.year <= horizon.year < self.year + self.technical_lifetime:
            return False
        if self.scenario_node is None:
            return True
        tree = chronology.scenario_tree
        if self.scenario_node not in tree.ancestors(horizon.scenario):
            return False
        decision_stage = next(
            node.stage for node in tree.nodes if node.name == self.scenario_node
        )
        return stage is None or decision_stage <= stage

    def present_cost(self, chronology: Chronology) -> float:
        rate = chronology.discount_rate
        annuity = (
            1 / self.financial_lifetime
            if rate == 0
            else rate / (1 - (1 + rate) ** -self.financial_lifetime)
        )
        years = chronology.financial_years or tuple(
            sorted({h.year for h in chronology.horizons})
        )
        probability = (
            1.0
            if self.scenario_node is None
            else chronology.scenario_tree.probability(self.scenario_node)
        )
        return (
            self.capital_cost
            * annuity
            * probability
            * sum(
                chronology.discount(y)
                for y in years
                if self.year <= y < self.year + self.financial_lifetime
            )
        )


class EnergyPolicy(StrictModule):
    emissions_limit: float | None = None
    carbon_price: float = 0.0
    investment_budget: float | None = None

    def __check_init__(self):
        for name, value in (
            ("emissions_limit", self.emissions_limit),
            ("investment_budget", self.investment_budget),
            ("carbon_price", self.carbon_price),
        ):
            if value is not None:
                _positive(value, name, zero=True)


class EnergySystem(StrictModule):
    chronology: Chronology
    carriers: tuple[Carrier, ...]
    points: tuple[BalancePoint, ...]
    sources: tuple[Source, ...] = ()
    demands: tuple[Demand, ...] = ()
    inventories: tuple[Inventory, ...] = ()
    converters: tuple[Converter, ...] = ()
    investments: tuple[Investment, ...] = ()
    policy: EnergyPolicy = eqx.field(default_factory=EnergyPolicy)

    def __check_init__(self):
        n = self.chronology.size
        for label, records in (
            ("carrier", self.carriers),
            ("point", self.points),
            (
                "asset",
                (*self.sources, *self.demands, *self.inventories, *self.converters),
            ),
            ("investment", self.investments),
        ):
            names = [record.name for record in records]
            if len(set(names)) != len(names) or any(
                not name or "/" in name for name in names
            ):
                raise ValueError(
                    f"{label} names must be unique, nonempty, and slash-free."
                )
        carriers = {carrier.name: carrier for carrier in self.carriers}
        points = {point.name: point for point in self.points}
        if not points or any(point.carrier not in carriers for point in self.points):
            raise ValueError("Balance points require known carriers.")
        for point in self.points:
            profile(point.spill_capacity, n, "spill capacity", nonnegative=True)
            profile(point.spill_cost, n, "spill cost")
        for asset in (*self.sources, *self.demands, *self.inventories):
            if asset.point not in points:
                raise ValueError(f"Unknown balance point {asset.point!r}.")
        for demand in self.demands:
            profile(demand.rate, n, "demand", nonnegative=True)
            profile(demand.unserved_cost, n, "unserved cost", nonnegative=True)
        for asset in (*self.sources, *self.converters):
            _positive(asset.capacity, "capacity", zero=True)
            _positive(asset.startup_cost, "startup cost", zero=True)
            if not 0 <= asset.minimum_fraction <= 1:
                raise ValueError("Minimum operating fraction must lie in [0, 1].")
            if not asset.commitment and (
                asset.minimum_fraction or asset.startup_cost or asset.initially_on
            ):
                raise ValueError(
                    "Minimum load and startup state/cost require commitment=True."
                )
            profile(asset.marginal_cost, n, "marginal cost")
            profile(asset.quadratic_cost, n, "quadratic cost", nonnegative=True)
            profile(asset.emissions, n, "emissions", nonnegative=True)
        for source in self.sources:
            available = profile(source.availability, n, "availability", nonnegative=True)
            if np.any(available > 1):
                raise ValueError("Availability must lie in [0, 1].")
        horizons = {h.name for h in self.chronology.horizons}
        for store in self.inventories:
            for value in (
                store.energy_capacity,
                store.charge_capacity,
                store.discharge_capacity,
            ):
                _positive(value, "storage capacity", zero=True)
            if (
                not 0 <= store.retention <= 1
                or not 0 < store.charge_efficiency <= 1
                or not 0 < store.discharge_efficiency <= 1
            ):
                raise ValueError(
                    "Retention must lie in [0, 1] and efficiencies in (0, 1]."
                )
            profile(store.throughput_cost, n, "throughput cost", nonnegative=True)
            boundaries = {b.horizon: b for b in store.boundaries}
            if set(boundaries) != horizons or len(store.boundaries) != len(horizons):
                raise ValueError(
                    "Every inventory requires exactly one boundary per horizon."
                )
            incoming = set()
            for boundary in store.boundaries:
                if boundary.link is not None:
                    if (
                        boundary.link not in horizons
                        or boundary.link == boundary.horizon
                        or boundary.link in incoming
                    ):
                        raise ValueError(
                            "Inventory links require a distinct, uniquely linked destination."
                        )
                    incoming.add(boundary.link)
                    if boundaries[boundary.link].initial is not None:
                        raise ValueError("A linked destination must have initial=None.")
                    hs = {h.name: h for h in self.chronology.horizons}
                    if hs[boundary.horizon].scenario != hs[boundary.link].scenario:
                        raise ValueError("Inventory links may not cross scenario paths.")
        for converter in self.converters:
            names = [port.point for port in converter.ports]
            if len(set(names)) != len(names) or not set(names) <= set(points):
                raise ValueError("Converter ports require distinct known balance points.")
            if (
                converter.reference_basis not in ("input", "output")
                or converter.reference_point not in names
            ):
                raise ValueError(
                    "Converter requires an explicit input/output reference port."
                )
            net = np.zeros(n)
            throughput = np.zeros(n)
            for port in converter.ports:
                coefficient = profile(port.coefficient, n, "converter coefficient")
                if np.any(coefficient == 0) or not (
                    np.all(coefficient > 0) or np.all(coefficient < 0)
                ):
                    raise ValueError(
                        "A converter port must retain a nonzero flow direction."
                    )
                if port.point == converter.reference_point and not np.all(
                    coefficient == (-1 if converter.reference_basis == "input" else 1)
                ):
                    raise ValueError(
                        "Reference-port coefficient must be exactly -1 (input) or +1 (output)."
                    )
                carrier = carriers[points[port.point].carrier]
                if carrier.environmental and np.any(coefficient > 0):
                    raise ValueError(
                        "Environmental ports are explicit inputs, not useful outputs."
                    )
                energy = coefficient * carrier.joules_per_unit
                net += energy
                throughput += np.abs(energy)
            if not np.all(np.isfinite(net)) or not np.all(np.isfinite(throughput)):
                raise ValueError(
                    "Converter energy coefficients exceed finite numerical support."
                )
            if np.any(net > 1e-9 * throughput):
                raise ValueError(
                    "Converter creates energy; COP>1 requires an explicit environmental heat input."
                )
        assets = {a.name: a for a in (*self.sources, *self.converters, *self.inventories)}
        for investment in self.investments:
            if investment.asset not in assets:
                raise ValueError("Investment requires a known capacity-bearing asset.")
            dimensions = (
                ("energy", "charge_power", "discharge_power")
                if isinstance(assets[investment.asset], Inventory)
                else ("power",)
            )
            if investment.dimension not in dimensions:
                raise ValueError(f"Investment dimension must be one of {dimensions}.")
            if investment.scenario_node is not None:
                if self.chronology.scenario_tree is None:
                    raise ValueError("Scenario investments require a scenario tree.")
                self.chronology.scenario_tree.ancestors(investment.scenario_node)
