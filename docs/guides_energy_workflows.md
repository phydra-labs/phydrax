# Executable energy workflows and qualification

These are consumer examples of the actual domain APIs, not an additional solver,
report schema, generic coupling framework, or performance claim. Run them from the
repository root in the worktree environment with 64-bit JAX enabled:

```sh
PYTHONPATH=. JAX_ENABLE_X64=1 python examples/energy/building_dispatch.py \
  --intervals 4 --output energy-results/building
PYTHONPATH=. JAX_ENABLE_X64=1 python examples/energy/power_fault.py \
  --output energy-results/power
PYTHONPATH=. JAX_ENABLE_X64=1 python tools/energy_qualification.py \
  --scenario building-dispatch --scenario power-fault \
  --output energy-results/qualification
```

The public `run_building_dispatch(output_dir, *, epw_path=None, intervals=4)` and
`run_power_fault(output_dir)` functions return dictionaries containing raw
`metrics`, reopened native lifecycle `archives`, exact execution identities, and
actual native control/optimization/power-flow/DAE results. The command-line routes
print physical metrics and artifact paths. Failed numerical work or physical
acceptance raises; it is not relabeled as success.

## Building, dispatch, and the meter boundary

The building workflow executes the following chain:

1. Parse an independently authored 24-hour synthetic EPW, or use `--epw /path/day.epw`.
   The synthetic dry bulb is a declared winter sinusoid; it is not measured weather
   and is not produced by EnergyPlus. Only the first 2–4 records are selected.
   Each EPW interval-ending Kelvin observation is explicitly held over its
   preceding interval. This is an offline forecast reconstruction, **not a causal
   estimator** that claims future measurements were available. Missing dry bulb
   fails. Actual record spacing is `3600 / records_per_hour` seconds; a caller's
   subhourly EPW therefore shortens the physical horizon without relabeling it.
2. Compile a one-zone native sensible RC model with explicit 2 MJ/K effective
   capacitance, 60 W/K envelope conductance, a named ambient boundary `air`, and
   150 W external internal gains. There is no moisture, latent heat, solar gain,
   inferred ground temperature, or undocumented air-change-rate conversion.
3. Start at 292.15 K and track 293.15 K using a COP-3 heat pump with a declared
   313.15 K supply. Solve the native two-interval linear-quadratic control problem,
   apply only its first electrical control, advance the actual RC model, observe
   the new temperature, and optimize again. Electrical bounds are 0–1500 W.
   The final solve has the one remaining interval. Comfort is a tracking objective,
   not a hard comfort-feasibility theorem; this particular run is separately
   required to track within 0.05 K.
4. Use the applied **delivered heat** as the planning demand, with a separate
   300 W baseline electricity demand. Planning declares electricity, useful heat,
   and environmental heat carriers; the converter consumes one electricity unit
   and two ambient units to deliver three heat units. A 2 kWh thermal store has
   independent 1.5 kW charge/discharge capacities, ideal retention/efficiencies,
   and explicitly zero entering and terminal inventory. Alternating positive
   0.10/0.40 currency/kWh prices and a positive throughput charge make load shifting
   meaningful. Storage is continuous and not assigned a mixed-integer exclusivity
   certificate. No disposal or unmet demand is enabled.
5. Independently replay decoded planning quantities through
   `replay_energy_system`, not the LP constraint matrix. Convert net storage-adjusted
   delivered heat back to W and require agreement with the requested RC heat to
   0.01 W. Replay the full building again. An independent closed-form one-zone
   exponential and its integrated envelope heat audit temperature and stored
   energy, rather than using a tautological model-derived instantaneous rate.
6. Test every unmodified meter demand against a two-bus balanced AC feeder with
   declared generator, voltage, and both-terminal branch limits. A converged root
   alone is insufficient. No load shedding, dispatch adjustment, extra slack,
   retry at a different load, or automatic AC OPF repair is attempted.

### Units and accounting

Planning `Carrier.unit` is **kWh of energy**, not kW. `Chronology.time_unit` is an
explicit hour unit. Rates are kWh/h, inventories are kWh, and durations are hours:

\[
r_{\mathrm{kWh/h}} = P_{\mathrm W}\frac{3600\ \mathrm{s/h}}{3.6\times10^6\ \mathrm{J/kWh}}
= P_{\mathrm W}/1000.
\]

The conversion helpers call native dimensional unit conversion; they do not infer
power from a label. AC power is per unit on a **total three-phase** 0.01 MVA base:
`P_pu = P_W / 10000`. There is no additional factor of three. The assumed meter
reactive demand is explicitly `Q_pu = 0.2 P_pu`.

The planner's grid boundary is delivered energy at the building meter. Upstream
AC generation supplies that unchanged demand **plus independently computed feeder
losses**. Reporting those upstream losses is not an AC repair and does not silently
add them to the planning objective. The workflow is AC feasibility, not joint
AC-optimal energy planning. Carbon accounting is explicitly 0.2 kg per metered
kWh; it does not claim upstream lifecycle emissions.

Archived interval ledgers and printed totals retain upstream electricity, ambient
heat, external internal gains, separate baseline electricity consumption, feeder
losses, integrated envelope heat out, building stored-energy change, and thermal
store change. Their global balance is audited within 1 J per interval. Internal
150 W gains are separately declared external heat; the 300 W electrical load is
not implicitly converted to room heat. The heat pump's ambient withdrawal and
the building envelope's outdoor exchange are distinct signed boundary flows.

Acceptance also requires native success, at most `1e-5 K` independent temperature
error, at most `1e-6 pu` AC bus residual, actual temperature excursion of at least
0.1 K, and actual storage use exceeding `1e-4 kWh`. These are declared specimen
criteria, not a guarantee for arbitrary EPWs. A warm-weather input incompatible
with this **heating-only** plant may correctly fail; the example does not add
cooling, clip the weather, or weaken its comfort criteria.

## Dispatch to finite fault and clearing

The power workflow starts with a native DC economic dispatch of two priced
sources. A cheap classical machine has an explicit 0.6 pu active-power limit and
a more expensive grid source supplies the balance. The two-bus network uses a
100 MVA three-phase base, 110 kV line-line buses, 60 Hz frequency, a lossless 0.2 pu
tie, and a 1 pu / 0.1 pu demand. The DC result is labeled as a lossless unit-voltage
approximation; it is **not** passed off as an AC operating point.

The dispatched P values are transferred to a separately compiled AC power-flow
study. Its explicit infinite-bus reference and machine PV control determine Q and
voltage. Only an operationally feasible AC result can initialize the dynamics.
The machine has `H=4 s`, `D=1`, `Xd_prime=0.3 pu`, and network MVA base. Its classical
internal EMF and fixed mechanical reference come from that operating point. The
load remains explicitly `constant_power`; it is not silently converted to an
impedance. The grid bus is explicitly passed as the infinite source.

Native index-one DAE initialization must satisfy both equilibrium and residual
norms of `1e-7`. The native segmented BDF execution samples 0–0.2 s at 0.005 s
spacing. It applies a finite balanced machine-bus shunt fault of `2-5j pu` at
0.05 s and clears it at exactly 0.10 s. There is no bolted-fault approximation,
EMT fidelity claim, custom power integrator, or interpolation across the event.

Evidence is based on actual states and rates:

- Reevaluate the native residual on every saved sample and on both sides of each
  event, with the correct topology for each side. Maximum residual must be `1e-5`.
- Require differential-state continuity to `1e-10`, actual event application,
  fresh first-order restart evidence, and no remaining fault after clearing.
- Require rotor angle excursion above `1e-4 rad`, speed excursion above `1e-6 pu`,
  and fault voltage jump above 0.01 pu. Frozen trajectories cannot pass simply
  because the native solver returned a successful status.
- Archive each segment separately, including duplicate event times for distinct
  pre-event and post-event states. Array column semantics and differential names
  are retained. Mixed state coordinates are not mislabeled as homogeneous volts.
- Reopen the physical final-state/rate/topology checkpoint, restore the classical
  EMF/mechanical reference arrays, and actually continue with fresh native BDF
  history to 0.21 s. Recompute the continuation residuals and archive them too.

The classical transient is a bounded deterministic specimen, not a large-system
stability certificate or an event-sensitivity claim.

## Lifecycle and qualification artifacts

The examples use the existing `ResultManifest`, `CheckpointManifest`,
`CheckpointShard`, `lifecycle.create`, and `lifecycle.open` APIs. Results retain
native numeric arrays; checkpoints retain physical restart coordinates. Every
result/checkpoint is reopened and checked for exact payload equality. They are
not pickled executable models or opaque optimizer/BDF cache snapshots. Building
checkpoints retain physical temperature, thermal inventory, time, and native
planning primal coordinates; power checkpoints additionally retain topology and
initialized machine coefficients. Resume requires the identified source model
and its declared boundary semantics.

`tools/energy_qualification.py` executes only selected scenarios. With no
`--scenario`, it selects the two native workflows. Additional selections are:

- `building-scientific`: the separately owned building benchmark's public native
  analytic RC, ground/air, HVAC, calibration, and held-out prediction helper.
- `energyplus`: real authored 100 W adiabatic ideal-load steady reference.
- `radiance`: real uniform sky irradiance compared with analytical π per RGB channel.
- `fmi`: the repository's **original accumulator specimen** compiled as a compatible
  FMI2 Co-Simulation FMU, including an internal time event, early return, and actual
  native state save/restore/replay. It is not an arbitrary-FMU physical validator.
- `helics`: two actual value federates delivering 125 W with exact double/W typing.
  No iterative convergence, rollback, or coupling stability is claimed.
- `opendss`: a real balanced 100 kW / 25 kvar feeder with independent aggregated
  source-minus-load-minus-loss checks. Native multiphase voltage results are
  retained; they are not silently coerced into a Phydrax positive-sequence model.

For example, using explicitly selected, qualified local installations:

```sh
PYTHONPATH=. JAX_ENABLE_X64=1 python tools/energy_qualification.py \
  --scenario building-scientific --output energy-results/scientific

PYTHONPATH=. JAX_ENABLE_X64=1 python tools/energy_qualification.py \
  --scenario energyplus --energyplus "$ENERGYPLUS_EXE" \
  --energyplus-version "$ENERGYPLUS_RELEASE" --energyplus-license "$ENERGYPLUS_LICENSE" \
  --scenario radiance --oconv "$OCONV_EXE" --rtrace "$RTRACE_EXE" --raypath "$RADIANCE_LIB" \
  --radiance-version "$RADIANCE_RELEASE" --radiance-license "$RADIANCE_LICENSE" \
  --output energy-results/external-buildings

PYTHONPATH=. JAX_ENABLE_X64=1 python tools/energy_qualification.py \
  --scenario fmi --fmu "$ACCUMULATOR_FMU" --fmu-sha256 "$ACCUMULATOR_SHA256" \
  --fmpy-version "$FMPY_RELEASE" --fmu-license "$FMU_LICENSE" \
  --scenario helics --helics-version "$HELICS_RELEASE" --helics-license "$HELICS_LICENSE" \
  --scenario opendss --opendss-version "$OPENDSSDIRECT_RELEASE" --opendss-license "$OPENDSS_LICENSE" \
  --output energy-results/external-interchange
```

No missing external capability is marked passed or substituted. Selected runtimes
need the explicit paths/version/license arguments shown; missing configuration is
an error. Runtime exceptions are recorded with the exact failure and traceback,
the selected matrix row fails, and the CLI exits nonzero. Other independently
selected scenarios may still execute; there is no retry or physical repair.

Outputs include raw metrics archives, a `qualification.json`, and a reopened
`qualification.zip` containing the existing `QualificationEvidence`,
`QualificationMatrix`, and coverage-report records. The matrix demands the exact
selected subject, evidence kind, source build identity, environment identity,
criterion, and raw artifact. It does not treat an unselected external capability
as covered. Evidence expires after one day by default; `--valid-for-seconds` and
`--reviewer` explicitly control the timestamp scope and reviewer identity.
The default reviewer identifies automated execution, not an independent human
scientific review.

Source identities hash the actual native Python source set, consumer examples,
qualification tool, consumed building helper, project/lock files, and original
FMI specimen source/XML. Environment records contain observed package releases,
Python/platform/device/backend identities, and JAX precision configuration.
External native adapters additionally retain actual package/native build digests
and artifact identities; command references pin executable bytes, and Radiance
also records its selected resource directory digest. A caller-declared executable
release is not independently authenticated by the script. These pins do not
certify all dynamically loaded system libraries or replace a qualified full
installation manifest. Source changes, runtime/environment changes, or model/input
changes require requalification.

The integration regression intentionally solves a real native carrier/storage
plan, then checks that a 1000-fold W versus kWh/h coupling error and corrupted
inventory history are rejected before building replay:

```sh
PYTHONPATH=. JAX_ENABLE_X64=1 python -m pytest -q tests/integration/test_energy_workflows.py
```

See [building energy](guides_building_energy.md), [energy planning](guides_energy_planning.md),
[balanced power](guides_power_systems.md), [energy series](guides_energy_series.md), and
[external execution](guides_energy_interchange.md) for the domain contracts.
