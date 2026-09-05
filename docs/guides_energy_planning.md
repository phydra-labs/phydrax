# Energy-system planning

`phydrax.applications.energy_planning` composes the native convex-program lifecycle
and bounded mixed-integer solver. Specifications describe physical assets and
chronology; compilations contain actual `LinearProgram` or `QuadraticProgram`
objects and a `PreparedConvexProgram`. There is no external optimization engine
or alternate generic modeling framework.

## A complete electricity, heat, and storage calculation

```python
from phydrax.applications import energy_planning as ep

spec = ep.electricity_heat_storage_example()
compiled = ep.compile_energy_system(spec)
solution = ep.solve_energy_system(compiled)
if not solution.successful:
    raise RuntimeError((solution.native_result.status, solution.replay.failures))
print(solution.plan.values("source/grid-import"))
print(solution.plan.values("inventory/heat-store/state/day"))
print(solution.replay.cost, solution.replay.emissions)
```

`electricity_hydrogen_example()` supplies a second complete network: an
explicit-LHV electrolyzer with useful heat output, an absolute hydrogen inventory,
and a fuel cell. Its electricity/heat amount unit is MWh and its hydrogen amount
unit is kg with 120 MJ/kg energy content. Converter ratios include this unit
conversion explicitly. The heat-pump example includes its environmental heat
balance point and ambient source; electricity alone cannot generate COP > 1.

Use 64-bit JAX arithmetic for tight optimization tolerances, particularly when
mixing carrier units. Solver tolerances and resource limits remain explicit in
`optim.ConvexSolvePolicy`; unsuccessful or resource-limited solves are not
reported as successful energy plans.

## Standalone numerical composition

```python
spec = ep.EnergySystem(
    ep.Chronology((ep.Horizon("day", (0.5, 2.0)),)),
    (ep.Carrier("electricity"),),
    (ep.BalancePoint("bus", "electricity"),),
    sources=(ep.Source("grid", "bus", 10.0, marginal_cost=(1.0, 4.0)),),
    demands=(ep.Demand("building", "bus", (2.0, 3.0)),),
    inventories=(ep.Inventory(
        "battery", "bus", energy_capacity=3.0,
        charge_capacity=1.0, discharge_capacity=2.0,
        boundaries=(ep.InventoryBoundary("day"),),
        charge_efficiency=0.95, discharge_efficiency=0.95,
    ),),
)
```

All rate profiles are scalars or a vector over the concatenated horizons, in
chronology order. `Carrier.unit` is an **amount** unit (`JOULE` by default), not a
power unit. Rates are carrier amount per `Chronology.time_unit` (`SECOND` by
default). Thus J/s is power, kg/s is mass flow, and inventories are J or kg. Native
`UnitDefinition` objects can select kWh and hours, for example; the duration and
all rates must use that declared basis. When integrating an `EnergySeries`,
convert its units and interval semantics before extracting numerical profiles;
the planner deliberately does not impose a series envelope on arrays.

Every balance is an equality: sources + converter net outputs + discharge +
unserved = demand + charge + spill. A `Demand` allows unmet load only when
`allow_unserved=True`, bounded above by that demand, with explicit
`unserved_cost`. A balance point permits disposal only through its explicit
`spill_capacity` and `spill_cost`. Both default to disabled; balance inequality
slack is never an implicit source or sink.

A `Converter` has signed `ConverterPort` coefficients (negative input, positive
output), an explicit `reference_point`, and `reference_basis="input"` or
`"output"`. The reference coefficient must be -1 or +1. Capacity is the reference
carrier's amount/time. All port coefficients retain their direction. Energy
content of outputs cannot exceed energy content of inputs; mass carriers need an
explicit joules-per-amount `energy_content`. Heat-pump environmental energy is a
real input, not a free efficiency multiplier hidden in a balance equation.

## Physical duration is not a statistical or economic weight

The inventory law is

`end = retention**duration * start + duration * (eta_charge * charge - discharge / eta_discharge)`.

Retention is the fraction remaining after one declared physical time unit.
Energy, charge power, and discharge power are independent capacities. No
representative multiplicity, probability, year weight, or discount enters this
law. Each interval advances the actual amount once. The law models constant
interval flow using an end-of-interval flow contribution; retention acts on the
entering amount. It is not an exact continuous-loss convolution of within-step
flows.

Accounting totals use `duration * multiplicity * probability`; monetary operating
costs additionally use `(1 + discount_rate)**-(year - base_year)`. Emissions are
not financially discounted. Startup costs count events, so their weights omit
duration. Representative repetition does not silently propagate inventory
through repeated copies: explicitly link physical horizons when carry-over is
intended.

Every inventory has one `InventoryBoundary` per horizon:

- `fixed`: explicit terminal `target`, with an explicit `initial` amount.
- `free`: `target=None`; the terminal amount is optimized within its capacity.
- `periodic`: `initial=None, target=None`; end equals the optimized start.
- `linked`: `target=None, link="next-horizon"`; end equals that horizon's start,
  whose `initial` must be `None`. Distinct scenarios cannot be linked together.

`initial=None` outside a link or periodic cycle is an explicitly optimized initial
stock and may provide free entering energy. Choose it only when that is intended.
Fixed defaults are zero at both ends; periodicity is never inferred from
representative weights. Links connect adjacent boundaries without an unmodeled
time gap; represent gaps by physical intervals if retention should act on them.

## Investments, retirement, and scenarios

An `Investment` names an asset, dimension (`power`, or storage `energy`,
`charge_power`, `discharge_power`), decision `year`, `technical_lifetime`, and
`financial_lifetime`. A continuous amount lies between zero and `maximum`.
`existing_capacity` describes a separate sunk vintage (with `maximum=0`). Base
asset capacity is available in every horizon; put expiring capacity into vintages
instead. Vintages contribute only from their decision year up to, but excluding,
the technical retirement year.

New capacity's `capital_cost` is annualized over its financial lifetime using the
chronology discount rate, and charged in explicit `financial_years` (or represented
years if omitted). Financing can continue after technical retirement. Financial
years need not have operational horizons. There is no implicit salvage credit or
cancellation of debt at retirement. `fixed_build_cost` is paid at the decision
year; `minimum_build` and a fixed charge require an exact binary build decision.
`EnergyPolicy.investment_budget` caps the sum of undiscounted new capital and
fixed charges over all modeled decisions, not a separate budget for each scenario.
The policy can also set expected, multiplicity-weighted emissions and a carbon
price in the operating objective.

`ScenarioTree` contains one stage-zero probability-one root and conditional
probabilities summing to one at every branch. Each horizon names a terminal leaf.
`Horizon.stage_start` selects the information stage of its first physical interval
(default zero); each subsequent interval advances one stage. Once a leaf is
revealed it remains known. Advance `stage_start` across years rather than resetting
information; the chronology rejects regression between years of the same
representative. This permits one tree to span multiple physical horizons and
investment years. Every representative/year group must contain every leaf once
with the same starting stage. A horizon's `probability` must match its leaf path.
Physical duration and representative multiplicity agree over shared history.

Operational variables are shared by `(year, representative, observed node,
interval)`, so decisions cannot anticipate the terminal leaf. Entering inventory
shares the previous interval's information. Scenario-node investments are a single
shared decision among descendants; they cannot operate before both their decision
year and the node's revelation stage. A `scenario_node=None` investment is shared
by every scenario. Multiple physical representative blocks remain distinct until
explicit inventory links join their boundaries.

## Exact decisions and convex execution

`Inventory(exclusive=True)` introduces one binary charge/discharge mode per
information set. `Source` and `Converter` support binary commitment,
`minimum_fraction`, `initially_on`, and exact startup indicators with
`startup_cost`. Capacity expansion and commitment use finite maximum-capacity
bounds, not a bilinear capacity/status product. Exact build, commitment, and
storage-mode records require `compile_energy_system(..., exact=True)`; omitting
it raises rather than silently relaxing the contract.

An exact compilation still permits simultaneous flow for a store explicitly
configured with `exclusive=False`. Select the exclusivity contract when negative
prices could otherwise reward loss cycling. Mode flags do not forbid intentional
cross-asset conversions; those follow the declared network physics.

LPs are retained as LP records. Nonnegative `quadratic_cost` adds a physical
`0.5 * quadratic_cost * rate**2` operating cost and selects the native QP record.
The solver policy selects native dense primal-dual or native conic execution as
supported by the underlying program. The dense default has explicit bounded
materialization/KKT limits; this implementation is assembled, not a sparse
large-network planner. Native bounded branch-and-bound preserves gaps, node limits,
relaxation failures, and feasibility evidence in `native_result`.
Affine LP/QP branch nodes also undergo finite, dimension-capped linear-bound
propagation. Each inferred bound retains its combination of canonical rows;
presolve prunes only when the resulting Farkas ray passes the native independent
original-coordinate audit. Failure to find a proof is not a feasibility claim:
the ordinary selected numerical relaxation still runs, and an uncertified failure
still invalidates an optimal-tree claim. This presolve does not claim support for
nonlinear or general conic implications.

```python
spec = ep.electricity_heat_storage_example(exact=True)
compiled = ep.compile_energy_system(spec, exact=True)
solution = ep.solve_energy_system(compiled)
assert solution.prices is None  # No mixed-integer dual claim.
if solution.successful:
    conditional = ep.fixed_integer_prices(compiled, solution)
```

## Prepared refresh, semantic maps, replay, and prices

`refresh_energy_system(compiled, new_spec)` uses the native prepared numeric
refresh and preserves the symbolic template. It rejects changed variable/row
semantics, binary decisions, program type, or bound roles; recompile structural
changes. `compiled.variables` maps named decoded quantities to solver indices and
physical scales. Shared scenario quantities can map to the same indices.
`compiled.rows` maps balance, inventory, capacity, terminal, and policy meanings
to equality/inequality rows and scales. Identical physical scenario equations
share a row rather than introducing rank-deficient duplicates.

`EnergyScaling` makes both maps explicit: physical values equal solver values
times variable scale, physical rows are divided by their row scale, and physical
objective equals solver objective times objective scale. Scaling changes numerical
conditioning, not physical limits or economic weights.

`replay_energy_system(spec, plan)` consumes **decoded quantities**, independently
recomputes balances, inventory transitions/boundaries, capacity and investment
availability, mode/commitment transitions, nonanticipativity, total cost and
emissions, and compares the recomputed objective with the claimed one. It never
multiplies the solver's constraint matrix. Missing or malformed decoded fields
raise; finite violations and nonfinite values produce named failures. Solver
success alone does not imply `EnergySolution.successful`: replay must also pass.

Continuous prices use the negative equality multiplier, multiply by objective
scale, divide by physical row scale, and divide by duration, representative
multiplicity, probability and discount. When scenario histories share a physical
balance row, normalization uses the sum of their objective weights. Returned
values are local currency per carrier amount, not weighted solver multipliers.

Prices are selected continuous balance subgradients. `weakly_active_constraints`
identifies zero-slack/near-zero-multiplier boundaries; uniqueness and
differentiability are **not certified**, including when the list is empty. No
MIP result carries prices. `fixed_integer_prices` performs a separate continuous
solve with an accepted incumbent's integer coordinates fixed; it reports
conditional prices, not marginal values of integer build or commitment choices.

Run the standalone tool with `PYTHONPATH=. JAX_ENABLE_X64=1 python
tools/energy_planning_benchmarks.py --example heat --repeats 2`. It records actual
compile/solve time, native status, replay errors and costs as JSON; any failed
solve exits nonzero. Use `--exact` for explicit storage modes and `--example
hydrogen` for the multi-carrier case. No performance numbers are implied by the
existence of the tool.
