# Coupled slab and ocean climate exchanges

The geophysics participants adapt the existing partitioned-coupling solver and
native ocean methods. They do not introduce a second time integrator, global
coupler, ocean state representation, or mosaic seam owner.

## Physical semantics are not storage identity

`solver.coupling.CouplingQuantity` describes a physical quantity using a native
`UnitDefinition`, a physical kind, a reference configuration, and a sign
convention. `to_dict()` and `from_dict()` carry the canonical descriptor and its
content-derived identities; no process-local registry is needed. Its
`compatibility_id` excludes the spelling and scale of the unit, but retains
canonical dimensions, unit reference system, kind, reference configuration and
sign. Its `quantity_id` includes the complete unit descriptor.

`geophysical_coupling_quantity(GeophysicalQuantity(...))` adapts application
metadata into this solver-owned contract. The solver does not import any
geophysical application. Existing generic `CouplingPort` instances may keep
`quantity=None`; they remain genuine untyped mathematical ports, not disguised
climate fields. A connection cannot mix a typed and an untyped endpoint.

Physical ports declare `frame`. Measured ports additionally declare a native
`DiscreteMeasure` and `measure_unit`. Current measured port storage is scalar
cell averages, with one physical measure weight per degree of freedom. The
measure's support must match the field support. Instantaneous fields and
whole-window integrals have distinct `temporal_kind` values; the solver never
multiplies an endpoint sample by an interval and calls that an exact integral.

A direct physical exchange requires exact field, vector-space, frame and
measure identities. Two physically compatible quantities on different grids
require an explicit `FieldTransfer` and `CouplingTransferRequirement` describing
conservation, constant preservation, positivity and frame action. Native unit
conversion is applied after the spatial operator. The solver checks required
transfer properties, compatible physical measure dimensions, and the actual
weighted transposed-operator conservation identity. This requires one transpose
action rather than materializing a dense transfer matrix. A transfer falsely
claiming conservation fails preparation. A frame change must be explicitly
identified as a transform; no rotation or change of basis is guessed.

## One authoritative proposal and one atomic acceptance

An `interval_integral` output is a signed proposed physical amount per unit
measure over the **current complete coupling window**. The participant producing
it must apply the corresponding debit to its own candidate. The receiving
participant applies the transferred proposal; it does not independently
recalculate the interface flux. A conserved output may not fan out and spend
the same amount twice. Split the physical flux into separate source output
ports when multiple recipients are required.

Participants replay from their frozen window-start state. Their candidate
continuations and native candidate ledgers are not published as accepted state
during nonlinear iterations. After final participant evaluation, the coupling
runtime integrates the original source proposal and the actual received target
value against their respective physical measures. It checks both global
conservation and local agreement between the received and mapped proposal.
These checks use roundoff-level tolerances, independently of looser nonlinear
stopping criteria. Even an explicit Jacobi window fails certification if a
recipient used a stale integral.

`CouplingWindowResult` provides:

- `proposed_exchange_budget`: one source-debit/target-credit pair per exchange;
- `accepted_exchange_budget`: the same pairs on success, all zeros on rejection;
- `accepted_state.cumulative_exchange_budget`: cumulative accepted accounting;
- `candidate_state`: diagnostic-only proposed state, never the restart boundary
  after failure.

Budgets use the reference units of the quantity times the physical measure.
Instantaneous, non-conserved exchanges have zero budget rows. Exchange rows are
in the prepared graph's canonical exchange-ID order. Negative heat proposals
reverse the debit/credit signs naturally. No equal-and-opposite diagnostic is
fabricated by replacing one independently computed side with its negative.

Participant failure, nonfinite evaluation, nonlinear failure and physical
certification failure preserve **all** accepted participant states, native
ledgers, interface values, coupling time, window index and cumulative budgets.
Replaying the same window-start checkpoint with the same proposal is
idempotent. A new accepted window adds accounting once. Adaptive outer rejection
also preserves the cumulative budget. An epoch transition retains accepted
budgets in reference units, rejects physical relabeling of retained exchanges,
and refuses to discard an exchange carrying nonzero accepted accounting.

## Slab heat and water reservoir

`SlabReservoir` has one receiving temperature port (absolute kelvin), and two
outgoing whole-window integral ports:

- potential enthalpy per area, J/m², relative to 273.15 K;
- water mass per area, kg/m², positive from slab into ocean.

Its authoritative state is `SlabReservoirState(enthalpy, water_mass)`. Heat
capacity per area is `dry_heat_capacity + water_heat_capacity * water_mass`.
Temperature is recovered from that capacity and physical enthalpy. The sensible
heat law is conductance times the temperature difference. Outgoing water also
carries its own constant-specific-heat enthalpy at the slab temperature, using
the same reference. This advected enthalpy is included in the **same** heat
proposal, not counted again as a second heat source.

The law is frozen at the start of each window and is first-order in coupling
time. Conductance and outgoing water rate are nonnegative. If a window would
exhaust water or produce nonpositive absolute temperature, it rejects; it does
not clip a budget or silently borrow water. Reduce the interval and replay from
the last accepted boundary. This reservoir currently models outgoing water,
not arbitrary two-way phase change or evaporation thermodynamics.

## Hydrostatic ocean boundary adapter

`HydrostaticOceanCouplingSubsystem` retains `HydrostaticContinuationState`,
`HydrostaticIMEXMidpointMethod`, the native free-surface/external-mode schedule,
and the native tracer and volume ledgers. Its freshwater input is converted
from mass to a **real volume source** using explicit `freshwater_density`; it
changes free-surface elevation and metric volume. The background freshwater
rate must be zero and its incoming salinity must be zero, so there is only one
owner of that boundary flux. Negative freshwater removes pure water without
removing salt. The heat port must provide the complete signed heat exchange,
including any evaporative cooling prescribed by the source model: the adapter
does not infer latent heat from a water-mass proposal. Native admissibility
still rejects depletion or an invalid ocean state.

The heat port is explicitly **constant-cp potential enthalpy**, represented by
`rho0 * heat_capacity * conservative_temperature_volume_inventory`, relative to
273.15 K. It is not temperature inventory in J, not in-situ seawater enthalpy,
and not a claim of total mechanical-plus-thermodynamic energy conservation.
The native freshwater step first applies its declared incoming CT composition.
The adapter then adds only the remaining heat proposal to the top-layer CT
inventory and the same increment to the native candidate tracer ledgers. This
is a conservative first-order surface-source split; native interior dynamics
remain midpoint. Final EOS validity is checked after heat insertion.

Only one `PreparedHydrostaticOcean` is accepted by this wrapper. It does not
intercept mosaic boundary traces, reassign lateral interfaces, or turn a mosaic
into independent blocks. Native open boundaries retain their own ledgers;
closed-domain heat/water balance tests use closed or periodic lateral domains.

## Rigid-lid Boussinesq boundary adapter

`BoussinesqOceanCouplingSubsystem` retains
`OceanBoussinesqContinuationState` and `OceanBoussinesqSSPRK33Method`. Heat enters
the native conservative scalar flux boundary and its native accepted
quadrature. Positive inward heat is converted to the native outward-loss sign.
With `stress=True`, two additional signed impulse-per-area ports drive native
Cartesian surface stress and its work ledger. This stress interface is
explicitly spatially uniform per horizontal component; nonuniform proposals
fail rather than being silently averaged.

There is **no freshwater port** on a rigid-lid participant. Existing nonzero or
dynamic salinity surface fluxes are rejected because virtual salt forcing does
not implement the heat/mass/volume physics of freshwater exchange. Coupling also
refuses to overwrite a separately owned nonzero/dynamic heat boundary or, when
stress coupling is selected, a separately owned stress law. Uncoupled native
forcing otherwise retains its original owner.

## Running and restarting

From the worktree, using its configured Python environment:

```sh
python examples/coupled_slab_ocean.py
```

The scenario connects one four-square-metre slab cell to four one-square-metre
ocean surface cells with explicit conservative, positive, constant-preserving
forward and reverse `FieldTransfer` operators. A source-first Gauss–Seidel sweep
ensures that the ocean consumes exactly the proposed heat and water integrals.
The temperature return is instantaneous. The example checks equal/opposite
accepted budgets and independently checks slab enthalpy/water changes against
native ocean tracer/volume ledgers, then prints the physical totals.

`write_geophysical_coupling_checkpoint(path, prepared, accepted_state)` archives
the accepted coupling boundary with every native continuation leaf, cumulative
budget, graph identity and prepared plan identity. Read with
`read_geophysical_coupling_checkpoint(path, prepared)`. Reusing local port names
with different physical descriptors, model constants, geometry or native plans
does not make a restart compatible. Existing native ocean checkpoint APIs
remain unchanged and may still archive an individual accepted continuation.

Focused regression coverage is in
`tests/unit/applications/test_climate_coupling.py`: nonmatching-grid conservation,
unit conversion, false measure claims, stale proposal rejection, participant
failure atomicity, replay, constant-flux window refinement, slab depletion,
physical hydrostatic enthalpy/freshwater budgets, native restart, first-order
coupling-time convergence and native rigid-lid heat/stress quadrature. Existing
partitioned-coupling, waveform/differentiation and workflow tests cover the
unchanged generic solver surface. Validation commands are run once by the
integration owner after concurrent implementation has finished.
