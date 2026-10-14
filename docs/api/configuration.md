# Configuration

`phydrax.config` owns typed host configuration, not scientific precision policy.
See the [configuration guide](../guides_configuration.md) for complete examples.

## Phases and startup

- `resolve()` layers immutable inputs and applies nothing. Native records
  (`RuntimeConfig`, `RuntimeBootstrap`, `LaunchEnvironmentConfig`,
  `NumericsConfig`, `ProviderConfig`) read their own `JAX_*`, `XLA_*`, and
  `PHYDRAX_*` inputs from the supplied environment, also when composed inside
  an application record. Each `FieldOrigin.environment_keys` lists every
  variable that set the field.
- `configure()` applies process-wide runtime, logging, or provider settings.
  Backend-bound platform, CPU topology, collectives, and bootstrap requests are
  refused with `ConfigurationApplicationError` after backend initialization.
  Launch-only settings are always refused in-process. A failed request keeps the
  fields applied before it, without rollback: `error.applied` lists the effects
  that ran, and `snapshot().requested` records every field that succeeded.
- `context()` applies thread-local numerical settings for a lexical scope and
  restores the previous settings on exit. Package attributes first resolved
  inside a scope load under the process settings and default device.
- `environment()` projects runtime intent for a child that has not started.
- `snapshot()` and the scalar observers never initialize a backend. Device and
  process facts belong to `phydrax.execution.runtime_info()`, which discovers devices.

The lazy `import phydrax` is fast and device-inert: it enables x64 unless
`JAX_ENABLE_X64` is set, including an explicit `JAX_ENABLE_X64=0`. It consumes
`PHYDRAX_CPU_COLLECTIVES` and launcher coordination settings before device use.
`PHYDRAX_COORDINATOR_ADDRESS`, `PHYDRAX_NUM_PROCESSES`, and `PHYDRAX_PROCESS_ID`
are required together; optional settings are `PHYDRAX_LOCAL_DEVICE_IDS`,
`PHYDRAX_COORDINATOR_BIND_ADDRESS`, `PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS`, and
`PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS` (positive integers). Foreign Slurm,
Open MPI, PMI, and PMIx launcher inputs retain their external contracts and are
consulted only at startup, never by `resolve()`.

There is one JAX configuration adapter, enforced by lint and architecture audit.
Applications use this public facade rather than direct upstream configuration
access. Scientific dtype choices, mixed-precision policies, tolerances, and
qualification evidence remain with their scientific owners.

Persistent records use one canonical wire form. Every union value is written as
`{"case": ..., "value": ...}`; nominal union alternatives (records and enums)
declare their own `__config_kind__`, which names their case. Enum values record
the canonical member name and its value. Restore decodes the record against the
explicit trusted type without reading the environment or filling defaults.

## Records and errors

::: phydrax.config.RuntimeConfig

::: phydrax.config.RuntimeBootstrap

::: phydrax.config.LaunchEnvironmentConfig

::: phydrax.config.NumericsConfig

::: phydrax.config.LoggingConfig

::: phydrax.config.ProviderConfig

::: phydrax.config.ConfigurationApplicationError

## Resolution and persistence

::: phydrax.config.ConfigurationDefinition

::: phydrax.config.ConfigurationLayer

::: phydrax.config.FieldOrigin

::: phydrax.config.ResolvedConfig

::: phydrax.config.resolve

## Application and observation

::: phydrax.config.configure

::: phydrax.config.context

::: phydrax.config.environment

::: phydrax.config.ConfigurationSnapshot

::: phydrax.config.snapshot

::: phydrax.config.x64_enabled

::: phydrax.config.logging_enabled

::: phydrax.config.default_prng_implementation

## Selectors

::: phydrax.config.CPUCollectives

::: phydrax.config.DTypePromotion

::: phydrax.config.EnvironmentMerge

::: phydrax.config.LibrarySetting

::: phydrax.config.RankPromotion

::: phydrax.config.TransferGuard

::: phydrax.config.WorkerSetting
