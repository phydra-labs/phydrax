# External energy execution and co-simulation

The energy interchange adapters execute **real external programs**. They do not
replace missing engines with numerical approximations, mocks, or cached success.
They are host-only, non-differentiable boundaries: call them outside `jit`, `vmap`,
`grad`, and other JAX transformations, then explicitly convert detached results
into native Phydrax arrays/series. Even argument-free calls inside a traced
function are rejected.

Host-only execution is one tier of the external-model capabilities
(`phydrax.ExecutionCapabilities`). Learned host runtimes use
`phydrax.export.HostInferenceAdapter` (eager only, detached values, declared
schemas and binding identity). Providers that supply adjoints use a staged
host-boundary VJP: `ExternalAdjointAction.stage_primal` evaluates the provider
once and returns an `ExternalPrimalStage`, and `apply_adjoint(stage, *ȳ)` returns
`Jᵀȳ` at exactly that staged realization, refusing any replay mismatch (see
`phydrax.interchange.dafoam.DAFoamAdjointAction`). A provider without an adjoint
reports its derivative-free alternatives through `ExternalDerivativeSupport` and
Phydrax never selects one for you. No external call runs inside a JAX
transformation.

## Ownership, identity, and limits

`phydrax.interchange.external_runtime` provides `PinnedExecutable`,
`pin_executable`, `run_pinned_command`, `run_energyplus`,
`run_radiance_command`, and `run_opendss`, together with the staged-adjoint
types `ExternalTensorSpec`, `ExternalPrimalStage`, `ExternalAdjointAction`, and
`ExternalDerivativeSupport`.

- Commands receive an argv sequence, never a shell command. Input files are exact
  bytes keyed by relative POSIX paths. Traversal, absolute paths, special output
  files, and symlink output reads are rejected.
- Every run has a private working directory. Returned file bytes, stdout, stderr,
  identities, exit status, elapsed time, and timeout evidence survive cleanup.
  `PinnedRunResult.output(path)` retrieves a requested file. Runtime failures raise
  `ExternalRuntimeError`; command failures carry `.result`, and optional native
  session failures carry `.evidence`, including bounded logs and process status.
- Large provider outputs stay on disk. `run_pinned_command(..., artifacts=
  PinnedFileOutputs(destination, requests, maximum_total_bytes))` declares each
  file as a `PinnedFileRequest(path, maximum_bytes)`. After a successful command
  every declared file must exist as a regular file within its own cap and the
  shared total cap; the set is then exclusively published under the
  caller-owned destination, and `PinnedRunResult.file_artifact(path)` returns its
  location, size, and SHA-256 digest. A missing or oversize file refuses the
  whole set (nothing is published) and raises `ExternalRuntimeError`; a failed
  command publishes nothing.
- `ScientificArtifactEnvelope` records producer/build/input identity and actual
  complete/failed status. Output identity does not certify physical model validity.
  Source URLs record provenance, not authorization or correctness.
- A pinned executable requires the expected SHA-256, declared release, and an
  explicit caller-supplied license identifier. Each run hashes an opened regular
  file and executes that exact held descriptor, so replacing its path cannot
  substitute code between verification and launch. `pin_executable`
  computes the pin for a selected file; independent release/digest qualification
  remains the caller's responsibility.
- Native sessions record installed distribution versions, manifests and native
  binary digests. `expected_version` (FMI: `expected_fmpy_version`) enforces a
  caller-selected package release. OpenDSS additionally identifies its DSS-Python
  binding and native backend build. No adapter invents redistribution permission
  for the engine, FMU, weather file, model, or reference outputs.
- Wall-clock timeout kills the owned process group and removes the work directory.
  Optional native libraries run in a separate trusted-local Python process using
  the current interpreter, not inside JAX or the parent's mutable working directory.
  These native sessions currently require POSIX (macOS/Linux).
- The default detached/log/message budget is 64 MiB. Input and output collections
  are bounded, worker responses are cardinality- and byte-bounded before sending,
  and logs are monitored and later read through the descriptors opened before
  launch.
- This is **trusted-local direct execution**, not a sandbox or a hard
  disk/CPU/memory quota. The child has ordinary host filesystem, process, and
  network access. `ExternalExecutionPolicy` rejects container/sandbox selectors
  and `network_access=False` because this runtime has no enforcing launcher.
  Execute untrusted native code only inside a separately provisioned
  OS/container sandbox and firewall boundary. Executable pinning does not pin
  every dynamically loaded system dependency.
- By default the command environment inherits the host and applies recorded
  explicit overrides; HOME/TMP point to the private directory. Inheritance can
  be disabled and overrides allowlisted, but that is not process isolation.
  Reproducible qualification must pin the engine's full installation and relevant
  environment/resources, not just its executable. Pass Radiance `RAYPATH`
  explicitly.

## EnergyPlus and Radiance command workflows

```python
from phydrax.interchange import (
    PinnedExecutable, run_energyplus, run_radiance_command,
)

# Executable paths, release labels, digests and licenses come from your qualified
# installation manifest, not values guessed by this adapter.
energyplus = PinnedExecutable(**energyplus_manifest)
run = run_energyplus(
    energyplus, idf_bytes, epw_bytes,
    model_format="idf", outputs=("eplusout.csv", "eplusout.err"),
    timeout=600,
)
csv_bytes = run.output("eplusout.csv")
```

`model_format` is `"idf"` or `"epjson"`; epJSON files are staged with `.epJSON`
extension. The EnergyPlus runner uses `--weather`, `--output-directory`, and
`--readvars`, always retains `eplusout.err`, and rejects severe/fatal diagnostics
even when a command exits zero. Therefore the selected installation must include
EnergyPlus's matching ReadVarsESO/resources. IDF/EPW semantic validation and result
alignment belong to the building application, not this process transport.

Radiance supports actual multi-command workflows without shell pipes. Each stage
gets the exact previous output bytes and its own pinned executable:

```python
oconv = PinnedExecutable(**oconv_manifest)
rtrace = PinnedExecutable(**rtrace_manifest)
context = {"RAYPATH": qualified_radiance_library_directory}
scene = run_radiance_command(
    oconv, ("scene.rad",), inputs={"scene.rad": scene_bytes},
    environment=context,
)
traced = run_radiance_command(
    rtrace, ("-h", "-I+", "scene.oct"),
    inputs={"scene.oct": scene.stdout}, stdin=sensor_rays_bytes,
    environment=context,
)
```

The same boundary accepts pinned `rfluxmtx`, `rcontrib`, `dctimestep`, or a
caller-installed producer executable; no unsupported claim about a particular
Frads release is made. Building daylight metadata, bases, units, calibration and
comparison semantics must be carried by the domain adapter. Radiance's binary
stdout is preserved as bytes rather than decoded as text.

## OpenDSS

`run_opendss` imports `opendssdirect` only in the isolated host worker. It creates
a new engine context, sets an isolated data path, executes explicit commands,
and returns the engine's actual solution and requested export files. The commands
must create one circuit and explicitly solve it. Engine errors and nonconvergence
are failures; an unsolved/missing circuit cannot become a successful result.

```python
from phydrax.interchange import run_opendss

result = run_opendss(
    (
        "Clear",
        "New Circuit.sample basekv=12.47 pu=1 phases=3 bus1=source",
        "New Line.feed bus1=source bus2=load phases=3 "
        "r1=0.1 x1=0.2 r0=0.3 x0=0.4 length=1 units=km",
        "New Load.demand bus1=load phases=3 conn=wye kv=12.47 kw=100 kvar=25",
        "Set voltagebases=[12.47]", "CalcVoltageBases", "Solve",
    ),
    license_id=qualified_engine_license,
    expected_version=qualified_opendssdirect_version,
)
```

The result deliberately preserves the raw **multiphase** engine convention:

- `node_names` orders `node_voltages`: `(real, imaginary)` RMS volts;
  `node_voltages_pu` are magnitudes with OpenDSS bus voltage bases.
- `total_power` is source-terminal inward-positive kW/kvar (normally negative
  for supplying sources); `losses` are positive-consumption W/var.
- `element_powers` entries are `(name, terminal_count, conductor_count, powers)`;
  powers are `(kW, kvar)` in terminal-major, conductor-minor order and positive
  into the element.

This is not silently coerced into balanced/positive-sequence, generation-positive
per-unit power-network data. Such a conversion requires explicit domain
qualification, base quantities, phase selection/aggregation, and sign conversion.

## FMI: closed FMI 2.0 Co-Simulation subset

`phydrax.interchange.fmi` exports `inspect_fmu`, `FMIVariable`, `FMIUnit`,
`FMIModelDescription`, `FMICoSimulationSession`, `FMIStepResult`, and `FMIState`,
plus the physical coupling binding described below.
The optional runtime is FMPy; a compatible FMU native binary is also required.
No source compilation, FMU conversion, Model Exchange solver, FMI3 clock support,
or missing-library fallback occurs implicitly.

Inspection checks the required archive digest and bounded file reads, then rejects
traversal, links/special files, encryption, duplicate members, unsupported
compression, excessive expansion, excessive member counts, and missing model XML.
UTF-8 XML has bounded size/depth/node/attribute counts; DTD/entity declarations
are rejected. Extraction is manual into a fresh directory after the same archive
policy checks. Default bounds are 64 MiB compressed, 256 MiB expanded, 4096 members,
32 levels, and 4 MiB for `modelDescription.xml`.

The declared runtime subset is:

- FMI **2.0 synchronous Co-Simulation**, not Model Exchange or asynchronous
  `fmi2Pending`. An asynchronously declared FMU is rejected before loading.
- Scalar Real, Integer, Boolean, String and Enumeration variable mappings by
  name/value reference. Units/causality/variability are retained, and every
  `UnitDefinitions/Unit` is kept as an `FMIUnit` with its `BaseUnit` exponents,
  factor, and offset (exact decimal text). Inputs can be
  set at communication points; parameters are set only before initialization.
  Boolean and integer writes are type checked, not silently coerced.
- Real instantiate, setup-experiment, enter/exit-initialization, scalar get/set,
  and `fmi2DoStep` calls. Fixed communication step restrictions are enforced when
  variable steps are not advertised.
- OK and warning results retain the reached communication point. Discard reads
  the actual `fmi2LastSuccessfulTime` and `fmi2Terminated` status. An early return
  never pretends the requested time was reached; the caller chooses the next
  action. Invalid returned time/status is a failure. FMI2 events remain internal
  to the Co-Simulation FMU; no external event iteration/convergence is claimed.
- `save_state`, `restore_state`, and `free_state` call the real native state API
  only if `canGetAndSetFMUstate` is advertised. State tokens belong to one live
  session, at most 64 are retained, and all native states are freed on close.
  They are not snapshots reconstructed from observable outputs. Although model
  inspection reports serialization capability, this API does not expose state
  serialization or derivatives.

```python
from phydrax.interchange.fmi import FMICoSimulationSession

with FMICoSimulationSession(
    "plant.fmu", trusted_root=fmu_directory, sha256=qualified_fmu_sha256,
    license_id=qualified_fmu_license,
    start_values={"u": 3.0, "gain": 2.0},
) as session:
    result = session.advance(0.25)
    values = session.get_values(("x",))
    if session.model.can_get_set_state:
        state = session.save_state()
        candidate = session.advance(0.4)
        session.restore_state(state)
        session.free_state(state)
```

The variable names above are those of Phydrax's original runnable qualification
FMU, not assumptions imposed on arbitrary FMUs. Its source is
`tests/interchange/data/energy_accumulator.c` with the adjacent XML description.
It integrates `dx/dt = gain*u` exactly and handles a time event by changing
`x` to `-x`; it exercises real early-return, Boolean/String/Integer mapping,
termination and native-state restoration. The matching test fixture compiles that
source using a local C compiler and packages a platform-compatible FMU. No
third-party source or binaries are vendored by that specimen.

### Coupling an FMU to native participants

`FMIVariableBinding(variable, port, realization)` declares how one FMI2 Real
communication variable realizes one physically typed, scalar `CouplingPort`
from `phydrax.solver.coupling`:

| Realization | Port | FMU variable |
|---|---|---|
| `"hold"` | instantaneous input | input set at the communication point and held over the step |
| `"uniform-rate"` | whole-window input amount | input rate `amount / window`, so exactly the amount enters under the FMI2 hold |
| `"sample"` | instantaneous output | output read at the reached communication point |
| `"increment"` | whole-window output amount | change of a cumulative output over the step |

`FMICouplingBinding(model, variables, time_unit=...)` checks every binding against
the model description: Real type, `input`/`output` causality, `continuous` or
`discrete` variability, and the variable's unit. The unit must name a
`UnitDefinitions` entry with a `BaseUnit`; its SI exponents and factor become an
exact `UnitDefinition` (`fmi_unit_definition`), whose dimension must equal the
port quantity's (divided by time for `"uniform-rate"`), and the exact factor
converts values at the host boundary. Undeclared units, units without `BaseUnit`,
affine units such as `degC`, and candela exponents are refused, never guessed.
`time_unit` is the unit of the coupling clock, which is the FMU's independent
variable: it must equal that variable's declared unit, must be seconds when the
FMU declares no independent variable (the FMI 2.0 default), and is refused when
the independent variable declares no unit. The binding records
`rollback="restore"` only when the FMU advertises `canGetAndSetFMUstate`.

`FMICouplingParticipant(session, binding, subsystem_id=...)` binds one live
session as a host participant of a `PartitionedCouplingDeclaration`, whose
`time_unit` must equal the binding's. Each window
sets every bound input at the accepted communication point and performs one
`fmi2DoStep` to the window end. An early return, a termination request, or a
non-finite input (refused before any process I/O) fails the window with an
`FMIWindowStatus`. The participant has no adjoint: its `derivative_support` is
route `"none"` with the derivative-free alternatives. Such declarations execute
only through `prepare_host_coupling` and `advance_host_coupling_window`; see
[partitioned coupling](guides_partitioned_coupling.md#host-participants).

```python
from phydrax.interchange.fmi import (
    FMICouplingBinding, FMICouplingParticipant, FMIVariableBinding,
)

binding = FMICouplingBinding(
    session.model,
    (
        FMIVariableBinding("boundary_temperature", boundary_port, "hold"),
        FMIVariableBinding("conducted_heat", conducted_heat_port, "increment"),
    ),
    time_unit=phx.units.SECOND,
)
zone = FMICouplingParticipant(session, binding, subsystem_id="zone")
```

`examples/fmi_host_coupling.py` compiles Phydrax's second original specimen,
`tests/interchange/data/thermal_zone.c` (one lumped capacity
`C dT/dt = G (T_b - T) + P` integrated exactly per held step, with units `K`, `W`,
`J`, `J/K`, `W/K`, an affine `degC` output, and a `maximum_temperature` cutout that
stops at the exact crossing with `fmi2Discard`), and couples it to a native
SSPRK(3,3) node. `tests/interchange/test_fmi_composition.py` checks both coupling
routes against their exact discrete recursions and the analytic two-capacity
solution, and the discard path against the FMU's native state restore.

## HELICS: typed value federation, not implicit coupling convergence

`phydrax.interchange.helics` exports `HelicsChannel`, `HelicsSample`,
`HelicsTimeGrant`, and `HelicsValueSession`. The optional native API is HELICS3;
zmq, tcp and ipc cores are the supported cross-process subset.

Publications are globally named. Each input has exactly one explicit target,
requires a connection, and enforces matching native type and **exact unit string**.
No implicit unit conversion, multi-input aggregation, or untyped subscription is
silently enabled. Types are double, int64 integer, Boolean, string, complex, and
bounded real vector. An input that has not received data reports `has_value=False`
and `value=None`, not an invented physical default.

```python
from phydrax.interchange.helics import HelicsChannel, HelicsValueSession

with HelicsValueSession(
    "plant", license_id=qualified_helics_license,
    publications=(HelicsChannel("plant/power", "double", "W"),),
    subscriptions=(HelicsChannel("observed", "double", "W", target="plant/power"),),
) as session:
    session.enter_execution()
    session.publish({"plant/power": 125.0})
    grant = session.advance(1.0)
    samples = session.read_values()
```

With no broker address the session owns a broker (one federate by default);
`federate_count` declares a larger federation. Its `broker_address` is available
before entry. A session using another broker's address never owns or destroys
that broker. For multiple sessions in one host thread, start all
`enter_execution_async()` operations before completing them with
`complete_execution()`. Likewise start `request_time_async(target)` on all
participants before `complete_time()`. These are real HELICS asynchronous API
pairs, not background guesses about federation progress.

Grants retain the actual requested/granted times, interruption and end-of-federation
status. They can arrive earlier than requested because another federate published
data. Reads preserve the last update time and updated flag. The adapter does not
promise rollback, fixed-point convergence, iterative coupling, derivatives, or
synchronization beyond the grants actually returned by HELICS. Unresponsive native
operations and missing participants are bounded by the session timeout; failure
kills the owned worker, while normal close finalizes/frees the federate and only
its owned broker.

## Qualification prerequisites and evidence

Run actual engine/FMU/federation scenarios in the deployment environment. Dependency
presence alone is not an execution qualification. The focused interchange tests
include a real native compiled FMI specimen, a real two-federate HELICS transfer,
a solved OpenDSS feeder, and executable timeout/resource-bound checks. Optional
runtime tests explicitly skip when their dependencies/compiler are unavailable.

EnergyPlus needs a matching executable, support resources, ReadVarsESO, validated
model and weather data. Radiance needs each selected command and its explicit
resource library path. FMI needs FMPy plus a compatible FMI2 Co-Simulation binary;
HELICS needs its Python/native installation and an accessible broker/core;
OpenDSS needs OpenDSSDirect.py 0.9 or newer with its DSS-Python native backend.
License/source/digest manifests must come from the caller's qualified artifacts.

The adapters follow the published interfaces rather than copying foreign code:
[FMI standard](https://fmi-standard.org/),
[FMPy FMI2 interface](https://github.com/CATIA-Systems/FMPy/blob/main/src/fmpy/fmi2.py),
[HELICS C API](https://docs.helics.org/en/latest/references/api-reference/C_API.html),
[PyHELICS](https://github.com/GMLC-TDC/pyhelics), and
[OpenDSSDirect.py](https://dss-extensions.org/OpenDSSDirect.py/).
