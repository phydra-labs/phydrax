#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native configuration: typed resolution, process settings, and numerical scopes.

- `resolve` layers defaults, declared environment bindings, Python values, and
  dotted literal overrides into an immutable typed configuration with a
  canonical content ID; it applies nothing.
- `configure` applies process-wide runtime, logging, or provider settings.
- `context` enters thread-local numerical settings for a lexical scope.
- `environment` projects runtime settings into the environment of a child
  process that has not started.
- `snapshot`, `x64_enabled`, `compilation_cache_enabled`, `logging_enabled`,
  and `default_prng_implementation` observe effective settings without
  initializing any JAX backend.

Scientific precision, execution plans, and observed runtime facts remain with
their owners; these settings only select the process environment they run in.
"""

from ._config_resolution import (
    ConfigurationDefinition,
    ConfigurationLayer,
    FieldOrigin,
    resolve,
    ResolvedConfig,
)
from ._config_state import (
    ConfigurationSnapshot,
    configure,
    context,
    environment,
    logging_enabled,
    snapshot,
)
from ._config_types import (
    ConfigurationApplicationError,
    CPUCollectives,
    DTypePromotion,
    EnvironmentMerge,
    LaunchEnvironmentConfig,
    LibrarySetting,
    LoggingConfig,
    NumericsConfig,
    ProviderConfig,
    RankPromotion,
    RuntimeBootstrap,
    RuntimeConfig,
    TransferGuard,
    WorkerSetting,
)
from ._jax_config import (
    compilation_cache_enabled,
    default_prng_implementation,
    x64_enabled,
)


__all__ = [
    "CPUCollectives",
    "ConfigurationApplicationError",
    "ConfigurationDefinition",
    "ConfigurationLayer",
    "ConfigurationSnapshot",
    "DTypePromotion",
    "EnvironmentMerge",
    "FieldOrigin",
    "LaunchEnvironmentConfig",
    "LibrarySetting",
    "LoggingConfig",
    "NumericsConfig",
    "ProviderConfig",
    "RankPromotion",
    "ResolvedConfig",
    "RuntimeBootstrap",
    "RuntimeConfig",
    "TransferGuard",
    "WorkerSetting",
    "compilation_cache_enabled",
    "configure",
    "context",
    "default_prng_implementation",
    "environment",
    "logging_enabled",
    "resolve",
    "snapshot",
    "x64_enabled",
]
