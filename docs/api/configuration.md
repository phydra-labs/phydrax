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
`JAX_ENABLE_X64` is set, including an explicit `JAX_ENABLE_X64=0`, and it sets
the default matmul precision to `"highest"` unless `JAX_DEFAULT_MATMUL_PRECISION`
is set or the process already selected a precision, so float32 contractions use
IEEE float32 rather than TF32 tensor cores on NVIDIA GPUs. It consumes
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

Apple GPU selection is `RuntimeConfig(platforms=("mps",), enable_x64=True)`.
Before applying it, admission checks the matching Metal companion's installation
and identity; it does not silently select another provider. Keep x64 enabled
for int64/uint64 addressing and choose float32/complex64 at numerical owners.
The configuration facade does not downcast models or change scientific dtype
policies. See [Apple GPU execution](../guides_metal_execution.md) and the
[Metal environment API](backends.md#apple-metal-environment).

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

`ProviderConfig.library_paths` accepts `LibrarySetting` keys `meshcore`, `metis`,
and the CUDA component settings below. CUDA overrides are intended for explicit
development or site builds and must be configured before first component use:

| Library setting | Environment variable |
| --- | --- |
| `cuda_cudss` | `PHYDRAX_CUDA_CUDSS_LIBRARY` |
| `cuda_amgx` | `PHYDRAX_CUDA_AMGX_LIBRARY` |
| `cuda_quda` | `PHYDRAX_CUDA_QUDA_LIBRARY` |
| `cuda_cutensor` | `PHYDRAX_CUDA_CUTENSOR_LIBRARY` |
| `cuda_cutensornet` | `PHYDRAX_CUDA_CUTENSORNET_LIBRARY` |
| `cuda_cuest` | `PHYDRAX_CUDA_CUEST_LIBRARY` |
| `cuda_collective` | `PHYDRAX_CUDA_COLLECTIVE_LIBRARY` |
| `cuda_cudss_mgmn` | `PHYDRAX_CUDA_CUDSS_MGMN_LIBRARY` |
| `cuda_cudecomp` | `PHYDRAX_CUDA_CUDECOMP_LIBRARY` |

The last three are site-bound MPI artifacts, not universal portable companion
components. An override does not bypass version, component, handler, dependency,
or site-MPI identity checks. Collective resources retain explicit lease-owned
lifetime rather than local memo eviction or implicit exit-time collectives.

Each value points to the built `libphydrax_cuda_<component>.so`. Overrides retain
release-version, CUDA-major, component, and handler checks; they do not override
the NVIDIA dependencies selected from installed wheels.


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
