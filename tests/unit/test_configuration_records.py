import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar

import pytest

from phydrax import config as pxconfig


def test_snapshot_record_is_json_compatible_and_contains_only_declared_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONFIGURATION_PRIVATE_SECRET", "must-not-be-captured")
    monkeypatch.setenv("PHYDRAX_METIS_LIBRARY", "/declared/metis")
    snapshot = pxconfig.snapshot()
    record = snapshot.to_record()
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == record
    assert "CONFIGURATION_PRIVATE_SECRET" not in encoded
    assert "must-not-be-captured" not in encoded
    assert record["library_paths"] == dict(snapshot.library_paths)
    assert record["workers"] == dict(snapshot.workers)
    assert snapshot.library_paths["metis"] == "/declared/metis"
    assert set(record) == {
        "requested",
        "enable_x64",
        "platforms",
        "cpu_device_count",
        "cpu_collectives_implementation",
        "matmul_precision",
        "dtype_promotion",
        "rank_promotion",
        "transfer_guard_device_to_host",
        "jit_disabled",
        "default_prng_implementation",
        "logging_enabled",
        "library_paths",
        "workers",
        "backends_initialized",
        "distributed_initialized",
    }


def test_snapshot_and_serialization_never_initialize_backends() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("JAX_", "PHYDRAX_"))
        and key
        not in {
            "XLA_FLAGS",
            "SLURM_NTASKS",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
        }
    }
    environment["PYTHONPATH"] = str(root)
    source = "from phydrax import config; import json; before = config.snapshot(); assert not before.backends_initialized; json.dumps(before.to_record(), allow_nan=False); assert not config.snapshot().backends_initialized"
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class Pair(Enum):
    __config_kind__: ClassVar[str] = "tests.configuration.pair"
    PAIR = (1, 2)
    ALIAS = (1, 2)


@dataclass(frozen=True)
class EnumRecipe:
    __config_kind__: ClassVar[str] = "tests.configuration.enum-recipe"
    selected: Pair | str = Pair.PAIR
    standalone: Pair = Pair.PAIR


@pytest.mark.parametrize("selected", [Pair.PAIR, "PAIR"], ids=["enum", "string"])
def test_enum_union_and_tuple_value_round_trip(selected: Pair | str) -> None:
    resolved = pxconfig.resolve(EnumRecipe, values={"selected": selected}, environment={})
    record = resolved.to_record()
    restored = pxconfig.ResolvedConfig.from_record(
        EnumRecipe, json.loads(json.dumps(record))
    )
    assert type(restored.value.selected) is type(selected)
    assert restored == resolved
    value = record["value"]
    assert isinstance(value, Mapping)
    assert value["standalone"] == {"name": "PAIR", "value": [1, 2]}


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "ALIAS", "value": [1, 2]},
        {"name": "PAIR", "value": [1, 3]},
        {"name": "UNKNOWN", "value": [1, 2]},
        {"name": "PAIR", "value": [1, 2], "extra": 1},
    ],
    ids=["alias", "changed-value", "unknown", "extra"],
)
def test_noncanonical_enum_payload_is_refused(payload: dict[str, object]) -> None:
    record = pxconfig.resolve(EnumRecipe, environment={}).to_record()
    value = record["value"]
    assert isinstance(value, dict)
    value["standalone"] = payload
    with pytest.raises(ValueError):
        pxconfig.ResolvedConfig.from_record(EnumRecipe, record)


@dataclass(frozen=True)
class OriginRecipe:
    __config_kind__: ClassVar[str] = "tests.configuration.origins"
    __config_environment__: ClassVar[Mapping[str, str]] = {"required": "TEST_REQUIRED"}
    required: int
    optional: int = 2


@pytest.mark.parametrize(
    "origins",
    [
        [],
        [{"path": "required", "layer": "value", "environment_keys": []}],
        [
            {"path": "required", "layer": "value", "environment_keys": []},
            {"path": "optional", "layer": "default", "environment_keys": []},
        ],
        [
            {"path": "optional", "layer": "default", "environment_keys": []},
            {"path": "required", "layer": "default", "environment_keys": []},
        ],
        [
            {"path": "optional", "layer": "default", "environment_keys": []},
            {"path": "required", "layer": "environment", "environment_keys": ["WRONG"]},
        ],
        [
            {"path": "optional", "layer": "default", "environment_keys": []},
            {"path": "required", "layer": "value", "environment_keys": ["TEST_REQUIRED"]},
        ],
        [
            {"path": "optional", "layer": "default", "environment_keys": []},
            {"path": "required", "layer": "environment", "environment_keys": []},
        ],
    ],
    ids=[
        "missing-all",
        "missing-one",
        "unsorted",
        "required-default",
        "wrong-binding",
        "value-with-keys",
        "environment-without-keys",
    ],
)
def test_invalid_restored_origins_are_refused(origins: list[dict[str, object]]) -> None:
    record = pxconfig.resolve(
        OriginRecipe, values={"required": 1}, environment={}
    ).to_record()
    record["origins"] = origins
    with pytest.raises(ValueError):
        pxconfig.ResolvedConfig.from_record(OriginRecipe, record)


def test_manually_constructed_result_is_validated_before_emission() -> None:
    resolved = pxconfig.resolve(OriginRecipe, values={"required": 1}, environment={})
    with pytest.raises(ValueError, match="ID"):
        replace(resolved, configuration_id="incorrect").to_record()
    with pytest.raises(ValueError, match="Origins"):
        replace(resolved, origins=()).to_record()


def test_restore_never_invokes_default_factories_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = pxconfig.resolve(OriginRecipe, values={"required": 1}, environment={})
    monkeypatch.setenv("TEST_REQUIRED", "not-an-integer")
    assert (
        pxconfig.ResolvedConfig.from_record(OriginRecipe, resolved.to_record())
        == resolved
    )


@dataclass(frozen=True)
class ContainerRecipe:
    __config_kind__: ClassVar[str] = "tests.configuration.container"
    selected: Mapping[str, int] | str
    sequence: tuple[int, ...] = ()


def test_union_envelope_cannot_collide_with_user_mapping_keys() -> None:
    resolved = pxconfig.resolve(
        ContainerRecipe, values={"selected": {"case": 1, "value": 2}}, environment={}
    )
    record = resolved.to_record()
    value = record["value"]
    assert isinstance(value, Mapping)
    envelope = value["selected"]
    assert isinstance(envelope, Mapping)
    assert envelope["value"] == {"case": 1, "value": 2}
    assert pxconfig.ResolvedConfig.from_record(ContainerRecipe, record) == resolved


@pytest.mark.parametrize(
    "envelope",
    [
        "PAIR",
        {"value": "PAIR"},
        {"case": "unknown", "value": "PAIR"},
        {"case": "str", "value": "PAIR", "extra": 1},
    ],
    ids=["untagged", "missing-case", "unknown-case", "extra"],
)
def test_malformed_union_envelopes_are_refused(envelope: object) -> None:
    record = pxconfig.resolve(EnumRecipe, environment={}).to_record()
    value = record["value"]
    assert isinstance(value, dict)
    value["selected"] = envelope
    with pytest.raises(ValueError, match="union"):
        pxconfig.ResolvedConfig.from_record(EnumRecipe, record)


def test_wire_collection_bound_is_enforced_before_id_validation() -> None:
    record = pxconfig.resolve(
        ContainerRecipe, values={"selected": "text"}, environment={}
    ).to_record()
    value = record["value"]
    assert isinstance(value, dict)
    value["sequence"] = [1] * 4097
    with pytest.raises(ValueError, match="tuple length"):
        pxconfig.ResolvedConfig.from_record(ContainerRecipe, record)


@pytest.mark.parametrize(
    "member_value",
    [[1, 2], {"a": 1}, (1, float("inf"))],
    ids=["mutable-list", "mutable-mapping", "nonfinite-tuple"],
)
def test_enum_member_values_are_validated_during_schema_compilation(
    member_value: object,
) -> None:
    members = Enum("members", {"ITEM": member_value})

    @dataclass(frozen=True)
    class Root:
        __config_kind__: ClassVar[str] = "tests.configuration.invalid-enum"
        selected: members

    with pytest.raises((TypeError, ValueError), match="Enum"):
        pxconfig.resolve(Root, environment={})


def test_codec_round_trip_in_fresh_interpreter() -> None:
    root = Path(__file__).resolve().parents[2]
    source = """
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar
from phydrax import config
@dataclass(frozen=True)
class Left:
    __config_kind__: ClassVar[str] = "smoke.left"
    count: int = 3
@dataclass(frozen=True)
class Right:
    __config_kind__: ClassVar[str] = "smoke.right"
    count: int = 3
class Pair(Enum):
    PAIR = (1, 2)
@dataclass(frozen=True)
class Root:
    __config_kind__: ClassVar[str] = "smoke.root"
    selected: Left | Right
    pair: Pair = Pair.PAIR
left = config.resolve(Root, values={"selected": Left()}, environment={})
right = config.resolve(Root, values={"selected": Right()}, environment={})
assert left.configuration_id != right.configuration_id
assert type(config.ResolvedConfig.from_record(Root, left.to_record()).value.selected) is Left
assert type(config.ResolvedConfig.from_record(Root, right.to_record()).value.selected) is Right
assert config.ResolvedConfig.from_record(Root, left.to_record()).value.pair is Pair.PAIR
print("canonical codec smoke passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "canonical codec smoke passed"
