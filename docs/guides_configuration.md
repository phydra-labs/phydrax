# Python-first configuration

`import phydrax` is fast and device-inert. Its lazy root loads subpackages on
first access, not at startup. Startup enables x64 by default, but an explicit
`JAX_ENABLE_X64=0` is respected. Launcher `PHYDRAX_*` coordination settings are
consumed before backend discovery; see the [API reference](api/configuration.md).

## Resolve intent without applying it

Use frozen Python dataclasses with an explicit canonical `__config_kind__`.
Declare environment bindings with `__config_environment__`, mapping field paths
to variable names. No configuration base class or executable data loader is needed.
Precedence is **defaults < environment < values < overrides**.

```python executable
from dataclasses import dataclass, field
from typing import ClassVar

from phydrax import config


@dataclass(frozen=True)
class SolverInputs:
    tolerance: float = 1e-6


@dataclass(frozen=True)
class RunInputs:
    __config_kind__: ClassVar[str] = "example.run-inputs"
    __config_environment__: ClassVar[dict[str, str]] = {
        "solver.tolerance": "EXAMPLE_TOLERANCE",
    }

    output: str
    label: str = "baseline"
    solver: SolverInputs = field(default_factory=SolverInputs)


resolved = config.resolve(
    RunInputs,
    environment={"EXAMPLE_TOLERANCE": "1e-5"},
    values={"output": "results", "solver": {"tolerance": 1e-7}},
    overrides=["solver.tolerance=1e-8", "label=study with spaces"],
)
# Show-only resolution: no runtime settings are applied.
print(resolved.to_json())
restored = config.ResolvedConfig.from_record(RunInputs, resolved.to_record())
assert restored.configuration_id == resolved.configuration_id
assert restored.value.solver.tolerance == 1e-8
assert restored.value.label == "study with spaces"
print([(origin.path, origin.layer) for origin in resolved.origins])
```

`output` is required: resolution refuses missing constructor inputs. Dotted
paths address declared constructor fields, not methods, private attributes, or
derived fields. Duplicate paths and parent/child overlap in one override layer
are rejected. An optional nested record is supplied as a whole value rather
than mutated through a missing parent. Every record must be a frozen dataclass,
and a supplied record instance is validated and rebuilt field by field: mutable
mappings become immutable and later changes to the caller's objects cannot
change the resolved value or its ID.

Override text is **data, never code**: supported Python literals include finite
numbers, booleans, `None`, quoted text, and bounded collections. For text fields,
nonliteral text falls back to the verbatim input, preserving spaces and punctuation;
it is never evaluated or imported. Quote strings that could otherwise be parsed
as a non-text literal. Arrays, devices, callbacks, and live provider objects are
not host configuration values.

`resolved.value` is the typed immutable record. `origins` reports each field's
winning layer and, for environment inputs, every contributing variable in
`environment_keys`. `to_record()` and `to_json()` produce canonical persistence
data; `ResolvedConfig.from_record(Type, record)` checks the declared kind, the
complete exact fields and origins, and the recomputed content ID, decoding the
record without reading the environment or filling defaults. Origins are
provenance and do not change the ID of equal canonical intent. Restore with an
explicit trusted type, not a class name selected by serialized data.

Union values persist as `{"case": ..., "value": ...}`, so a restored value keeps
its exact alternative. A record or enum used as a union alternative, including
`Solver | None`, declares its own `__config_kind__`; that kind names its case.
Enum values persist as their canonical member name together with the member's
value, so changing either changes the configuration ID. Enum member values are
immutable builtin data: booleans, integers, finite floats, text, `None`, and
tuples of these.

## Compose runtime and numerical records

Compose records in Python, resolve them if layered input is needed, then apply
each record at its owning phase. Runtime intent is process-wide; numerical
intent is thread-local. `None` inherits the current setting. Package attributes
that load a module on first access inside a `context`, or inside a device scope,
load it under the process-wide settings and default device; write `import`
statements outside numerical scopes so a scope never fixes a module's
import-time constants.

Native records resolve their own environment inputs: `RuntimeConfig` reads
`JAX_ENABLE_X64`, `JAX_PLATFORMS`, `JAX_NUM_CPU_DEVICES` (or a forced-host
count in `XLA_FLAGS`), `PHYDRAX_CPU_COLLECTIVES`, the explicit `PHYDRAX_*`
coordinator settings, and its launch inputs; `NumericsConfig` reads the
matching `JAX_*` numerical options; `ProviderConfig` reads the declared library
and worker variables. This also holds for native records composed inside an
application record. Resolution is pure: the startup x64 default and foreign
launcher detection belong to process startup only.

```python executable
from dataclasses import dataclass, field
from typing import ClassVar

from phydrax import config


@dataclass(frozen=True)
class ApplicationInputs:
    __config_kind__: ClassVar[str] = "example.application-inputs"

    runtime: config.RuntimeConfig = field(
        default_factory=lambda: config.RuntimeConfig(enable_x64=True)
    )
    numerics: config.NumericsConfig = field(
        default_factory=lambda: config.NumericsConfig(
            dtype_promotion="strict", rank_promotion="raise"
        )
    )


resolved = config.resolve(ApplicationInputs, environment={})
config.configure(resolved.value.runtime)
with config.context(resolved.value.numerics):
    observed = config.snapshot()
    assert observed.dtype_promotion == "strict"
    assert observed.rank_promotion == "raise"
```

Configure platforms, CPU device count, CPU collectives, and distributed
`RuntimeBootstrap` before any backend exists. Late backend-bound requests are
refused, not silently ignored. A failed request is not rolled back: fields
applied before the failure stay applied and are recorded in
`snapshot().requested`, and `ConfigurationApplicationError.applied` lists the
effects that ran. `snapshot()` never discovers devices;
`phydrax.execution.runtime_info()` does. These settings do not replace scientific
precision policies, explicit array dtypes, or execution plans.

Logging emission is selected by
`config.configure(config.LoggingConfig(enabled=True))`; handlers, sinks, and
correlation context remain with `phydrax.logging`. Provider paths and worker
selection belong to `ProviderConfig`, while ABI checks and actual loaded-provider
identity remain with each provider owner.

## Project a child-process environment

Launch-only settings cannot be applied to the current process. Project them
before starting a child, without mutating `os.environ`:

```python executable
from phydrax import config

settings = config.RuntimeConfig(
    enable_x64=False,
    platforms=("cpu",),
    cpu_device_count=4,
    launch_environment=config.LaunchEnvironmentConfig(preallocate=False),
)
child_environment = config.environment(settings, base={})
assert child_environment["JAX_ENABLE_X64"] == "0"
assert child_environment["JAX_NUM_CPU_DEVICES"] == "4"
assert child_environment["JAX_PLATFORMS"] == "cpu"
# Pass child_environment as subprocess.run(..., env=child_environment).
```

Omit `base` to copy the current environment. `mode="override"` replaces selected
settings; `mode="defaults"` preserves settings already selected in the base.
An explicit bootstrap projects as one group: the coordinator, process count,
process ID, both timeouts (`PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS`,
`PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS`), and any local device IDs and bind
address. Override mode removes a base's rank settings the request omits;
defaults mode keeps a base's existing bootstrap group whole. Native CPU topology
uses `JAX_NUM_CPU_DEVICES=N`; a forced-host count already in the base's
`XLA_FLAGS` counts as a selection in defaults mode, and an override replaces it
while keeping the other flags. Forced host devices demonstrate functional
parity, not physical accelerator or multi-host performance.

Shell recipes may retain `JAX_ENABLE_X64=1` to make launch intent explicit,
although Phydrax enables x64 by default. All Python configuration reads, writes,
and scopes go through `phydrax.config`: there is one upstream adapter, not a
second configuration convention in applications.
