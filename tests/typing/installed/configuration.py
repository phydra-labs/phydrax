from dataclasses import dataclass
from typing import assert_type, ClassVar

from jax import Device

from phydrax.backends import frontend_index_patch_identity
from phydrax.config import (
    ConfigurationSnapshot,
    configure,
    context,
    environment,
    FieldOrigin,
    NumericsConfig,
    resolve,
    ResolvedConfig,
    RuntimeConfig,
    snapshot,
    x64_enabled,
)
from phydrax.execution import default_device


@dataclass(frozen=True)
class MyConfig:
    __config_kind__: ClassVar[str] = "consumer.configuration"
    count: int = 2


def consumer_contract() -> None:
    assert_type(
        resolve(MyConfig, values={"count": 3}, overrides=["count=4"], environment={}),
        ResolvedConfig[MyConfig],
    )
    assert_type(x64_enabled(), bool)
    assert_type(
        FieldOrigin("count", "environment", ("COUNT",)).environment_keys, tuple[str, ...]
    )
    FieldOrigin("count", "environment", "COUNT")  # ty: ignore[invalid-argument-type]
    FieldOrigin("count", "value").environment_key  # ty: ignore[unresolved-attribute]
    assert_type(ResolvedConfig.from_record(MyConfig, {}), ResolvedConfig[MyConfig])
    assert_type(snapshot(), ConfigurationSnapshot)
    assert_type(environment(RuntimeConfig(enable_x64=True), base={}), dict[str, str])
    with context(NumericsConfig(enable_x64=True)):
        assert_type(x64_enabled(), bool)
    configure(NumericsConfig(enable_x64=True))  # ty: ignore[invalid-argument-type]
    configure(RuntimeConfig(), unknown=True)  # ty: ignore[unknown-argument]
    NumericsConfig(dtype_promotion="invalid")  # ty: ignore[invalid-argument-type]
    RuntimeConfig(cpu_collectives_implementation="invalid")  # ty: ignore[invalid-argument-type]


def execution_contract(device: Device) -> None:
    with default_device(device) as scope:
        assert_type(scope, None)
    assert_type(frontend_index_patch_identity(), dict[str, str] | None)
