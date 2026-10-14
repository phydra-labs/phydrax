from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal

import pytest

from phydrax import config as pxconfig


@dataclass(frozen=True)
class Solver:
    __config_kind__: ClassVar[str] = "tests.configuration.solver"
    tolerance: float = 0.01
    iterations: int = 10


class Method(Enum):
    DIRECT = "direct"
    ITERATIVE = "iterative"


@dataclass(frozen=True)
class Recipe:
    __config_kind__: ClassVar[str] = "tests.configuration.recipe"
    __config_environment__: ClassVar[Mapping[str, str]] = {
        "solver.iterations": "TEST_ITERATIONS"
    }
    solver: Solver = Solver()
    optional: Solver | None = None
    selection: Solver | str = "automatic"
    labels: Mapping[str, int] = field(default_factory=lambda: {"old": 1})
    name: str = "baseline"
    selector: Literal["fast", "accurate"] = "fast"
    method: Method = Method.DIRECT
    derived: int = field(init=False, default=42)


@pytest.mark.parametrize(
    ("environment", "values", "overrides", "expected", "layer"),
    [
        ({}, {}, [], 10, "default"),
        ({"TEST_ITERATIONS": "20"}, {}, [], 20, "environment"),
        ({"TEST_ITERATIONS": "20"}, {"solver": {"iterations": 30}}, [], 30, "value"),
        (
            {"TEST_ITERATIONS": "20"},
            {"solver": {"iterations": 30}},
            ["solver.iterations=40"],
            40,
            "override",
        ),
    ],
    ids=["default", "environment", "value", "override"],
)
def test_resolution_precedence(
    environment: dict[str, str],
    values: dict[str, object],
    overrides: list[str],
    expected: int,
    layer: str,
) -> None:
    resolved = pxconfig.resolve(
        Recipe, environment=environment, values=values, overrides=overrides
    )
    assert resolved.value.solver == Solver(iterations=expected)
    origin = next(
        origin for origin in resolved.origins if origin.path == "solver.iterations"
    )
    assert origin.layer == layer
    assert origin.environment_keys == (
        ("TEST_ITERATIONS",) if layer == "environment" else ()
    )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("optional={'iterations': 7}", {"optional": Solver(iterations=7)}),
        ("selection={'tolerance': 0.5}", {"selection": Solver(tolerance=0.5)}),
        ("labels={'new': 2}", {"labels": {"new": 2}}),
    ],
    ids=["optional-record", "union-record", "mapping"],
)
def test_whole_field_replacement(override: str, expected: dict[str, object]) -> None:
    resolved = pxconfig.resolve(Recipe, overrides=[override], environment={})
    assert {key: getattr(resolved.value, key) for key in expected} == expected


@pytest.mark.parametrize(
    "path",
    [
        "unknown",
        "derived",
        "_private",
        "optional.iterations",
        "selection.iterations",
        "labels.old",
    ],
    ids=[
        "unknown",
        "derived",
        "private",
        "optional-child",
        "union-child",
        "mapping-child",
    ],
)
def test_unaddressable_paths_are_refused(path: str) -> None:
    with pytest.raises(ValueError, match=path.split(".")[0]):
        pxconfig.resolve(Recipe, overrides=[f"{path}=1"], environment={})


@pytest.mark.parametrize(
    "overrides",
    [
        ["solver={'iterations': 1}", "solver.iterations=2"],
        ["solver.iterations=1", "solver={'iterations': 2}"],
        ["name=a", "name=b"],
    ],
    ids=["record-then-field", "field-then-record", "duplicate"],
)
def test_overlapping_overrides_are_refused(overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        pxconfig.resolve(Recipe, overrides=overrides, environment={})


@dataclass(frozen=True)
class RequiredSolver:
    tolerance: float


@dataclass(frozen=True)
class RequiredRecipe:
    __config_kind__: ClassVar[str] = "tests.configuration.required"
    name: str
    solver: RequiredSolver


def test_missing_required_inputs_name_full_paths() -> None:
    with pytest.raises(ValueError) as caught:
        pxconfig.resolve(RequiredRecipe, environment={})
    assert "name" in str(caught.value)
    assert "solver.tolerance" in str(caught.value)


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ("solver.iterations", True, TypeError),
        ("solver.iterations", 1.5, TypeError),
        ("solver.tolerance", float("nan"), ValueError),
        ("solver.tolerance", float("inf"), ValueError),
        ("solver.tolerance", float("-inf"), ValueError),
    ],
    ids=["bool-as-int", "lossy-int", "nan", "positive-infinity", "negative-infinity"],
)
def test_invalid_numeric_inputs(path: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match=path):
        pxconfig.resolve(
            Recipe, values={"solver": {path.split(".")[1]: value}}, environment={}
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("unquoted text", "unquoted text"),
        ("'quoted text'", "quoted text"),
        ('"line\\nend"', "line\nend"),
        ("123", "123"),
        ("__import__('os')", "__import__('os')"),
    ],
    ids=["verbatim", "single-quoted", "escaped-literal", "numeric-text", "expression"],
)
def test_string_literal_grammar_is_data_only(text: str, expected: str) -> None:
    assert (
        pxconfig.resolve(Recipe, overrides=[f"name={text}"], environment={}).value.name
        == expected
    )


def test_expressions_are_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    expression = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    assert (
        pxconfig.resolve(
            Recipe, overrides=[f"name={expression}"], environment={}
        ).value.name
        == expression
    )
    with pytest.raises(TypeError, match="iterations"):
        pxconfig.resolve(
            Recipe, overrides=[f"solver.iterations={expression}"], environment={}
        )
    assert not marker.exists()


@dataclass(frozen=True)
class Alternative:
    __config_kind__: ClassVar[str] = "tests.configuration.alternative"
    tolerance: float = 0.01
    iterations: int = 10


@dataclass(frozen=True)
class AmbiguousRecipe:
    __config_kind__: ClassVar[str] = "tests.configuration.ambiguous"
    solver: Solver | Alternative


def test_union_requires_unambiguous_record_selection() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        pxconfig.resolve(
            AmbiguousRecipe, values={"solver": {"iterations": 2}}, environment={}
        )


@pytest.mark.parametrize(
    ("overrides", "selector", "method"),
    [
        (["selector=accurate", "method=iterative"], "accurate", Method.ITERATIVE),
        (["selector='fast'", "method='direct'"], "fast", Method.DIRECT),
    ],
    ids=["verbatim-selectors", "quoted-selectors"],
)
def test_literal_and_enum_selectors(
    overrides: list[str], selector: str, method: Method
) -> None:
    value = pxconfig.resolve(Recipe, overrides=overrides, environment={}).value
    assert value.selector == selector
    assert value.method is method


@pytest.mark.parametrize(
    "override", ["selector=invalid", "method=invalid"], ids=["literal", "enum"]
)
def test_invalid_selectors_are_refused(override: str) -> None:
    with pytest.raises(ValueError):
        pxconfig.resolve(Recipe, overrides=[override], environment={})


def test_configuration_identity_ignores_input_provenance() -> None:
    default = pxconfig.resolve(Recipe, environment={})
    explicit = pxconfig.resolve(
        Recipe, environment={"TEST_ITERATIONS": "10"}, overrides=["name=baseline"]
    )
    assert default.configuration_id == explicit.configuration_id
    assert default.origins != explicit.origins


@dataclass(frozen=True)
class InheritedRecipe(Recipe):
    pass


def test_configuration_kind_must_be_declared_on_concrete_class() -> None:
    with pytest.raises(TypeError, match="__config_kind__"):
        pxconfig.resolve(InheritedRecipe, environment={})


def test_resolved_record_round_trip() -> None:
    resolved = pxconfig.resolve(
        Recipe,
        overrides=["optional={'iterations': 4}", "method=iterative"],
        environment={},
    )
    assert pxconfig.ResolvedConfig.from_record(Recipe, resolved.to_record()) == resolved


@pytest.mark.parametrize(
    "tamper",
    [
        "id",
        "kind",
        "missing-envelope",
        "unknown-envelope",
        "missing-value",
        "unknown-value",
    ],
    ids=[
        "id",
        "kind",
        "missing-envelope",
        "unknown-envelope",
        "missing-value",
        "unknown-value",
    ],
)
def test_tampered_records_are_refused(tamper: str) -> None:
    record = pxconfig.resolve(Recipe, environment={}).to_record()
    match tamper:
        case "id":
            record["configuration_id"] = "tampered"
        case "kind":
            record["kind"] = "other.kind"
        case "missing-envelope":
            del record["origins"]
        case "unknown-envelope":
            record["extra"] = 1
        case "missing-value" | "unknown-value":
            value = record["value"]
            assert isinstance(value, dict)
            if tamper == "missing-value":
                del value["name"]
            else:
                value["extra"] = 1
    with pytest.raises(ValueError):
        pxconfig.ResolvedConfig.from_record(Recipe, record)


def test_nested_exact_instances_are_revalidated() -> None:
    invalid = Solver(iterations=True)
    with pytest.raises(TypeError, match="optional.iterations"):
        pxconfig.resolve(Recipe, values={"optional": invalid}, environment={})


def test_nested_record_mapping_is_snapshotted() -> None:
    @dataclass(frozen=True)
    class Nested:
        __config_kind__: ClassVar[str] = "tests.configuration.nested"
        values: Mapping[str, int]

    @dataclass(frozen=True)
    class Root:
        __config_kind__: ClassVar[str] = "tests.configuration.root"
        nested: Nested | None = None

    caller = {"a": 1}
    supplied = Nested(caller)
    resolved = pxconfig.resolve(Root, values={"nested": supplied}, environment={})
    before = resolved.to_json()
    caller["a"] = 2
    assert resolved.to_json() == before
    assert resolved.value.nested is not supplied


def test_mutable_dataclass_with_attribute_hook_is_refused() -> None:
    @dataclass
    class Mutable:
        __config_kind__: ClassVar[str] = "tests.configuration.mutable"
        count: int = 1

        def __setattr__(self, name: str, value: object) -> None:
            object.__setattr__(self, name, value)

    with pytest.raises(TypeError, match="frozen"):
        pxconfig.resolve(Mutable, environment={})


def test_nominal_union_identity_and_exact_restore() -> None:
    left = pxconfig.resolve(AmbiguousRecipe, values={"solver": Solver()}, environment={})
    right = pxconfig.resolve(
        AmbiguousRecipe, values={"solver": Alternative()}, environment={}
    )
    assert left.configuration_id != right.configuration_id
    assert (
        type(
            pxconfig.ResolvedConfig.from_record(
                AmbiguousRecipe, left.to_record()
            ).value.solver
        )
        is Solver
    )
    assert (
        type(
            pxconfig.ResolvedConfig.from_record(
                AmbiguousRecipe, right.to_record()
            ).value.solver
        )
        is Alternative
    )


def test_unused_default_factory_is_not_called_for_complete_instance() -> None:
    def unused() -> int:
        raise AssertionError("Unused default factory was called.")

    @dataclass(frozen=True)
    class Complete:
        __config_kind__: ClassVar[str] = "tests.configuration.complete"
        count: int = field(default_factory=unused)

    assert pxconfig.resolve(Complete, values=Complete(3), environment={}).value.count == 3


def test_post_init_mutable_state_is_not_repaired() -> None:
    @dataclass(frozen=True)
    class MutableOutput:
        __config_kind__: ClassVar[str] = "tests.configuration.mutable-output"
        values: Mapping[str, int]

        def __post_init__(self) -> None:
            object.__setattr__(self, "values", dict(self.values))

    with pytest.raises(TypeError, match="frozendict"):
        pxconfig.resolve(MutableOutput, values={"values": {"a": 1}}, environment={})


@pytest.mark.parametrize(
    "alternative",
    ["missing", "inherited", "duplicate"],
    ids=["missing", "inherited", "duplicate"],
)
def test_union_requires_distinct_direct_nominal_kinds(alternative: str) -> None:
    @dataclass(frozen=True)
    class Missing:
        count: int = 1

    @dataclass(frozen=True)
    class Inherited(Solver):
        pass

    @dataclass(frozen=True)
    class Duplicate:
        __config_kind__: ClassVar[str] = Solver.__config_kind__
        count: int = 1

    selected = {"missing": Missing, "inherited": Inherited, "duplicate": Duplicate}[
        alternative
    ]

    @dataclass(frozen=True)
    class Root:
        __config_kind__: ClassVar[str] = "tests.configuration.invalid-union"
        value: Solver | selected

    with pytest.raises((TypeError, ValueError), match="kind"):
        pxconfig.resolve(Root, values={"value": Solver()}, environment={})


def test_composed_native_environment_is_layered_and_restorable() -> None:
    @dataclass(frozen=True)
    class Composed:
        __config_kind__: ClassVar[str] = "tests.configuration.composed"
        runtime: pxconfig.RuntimeConfig = pxconfig.RuntimeConfig()
        numerics: pxconfig.NumericsConfig = pxconfig.NumericsConfig()
        providers: pxconfig.ProviderConfig = field(
            default_factory=pxconfig.ProviderConfig
        )

    resolved = pxconfig.resolve(
        Composed,
        environment={"JAX_ENABLE_X64": "false", "PHYDRAX_METIS_LIBRARY": ""},
        values={"runtime": {"enable_x64": True}},
    )
    assert resolved.value.runtime.enable_x64 is True
    assert resolved.value.numerics.enable_x64 is False
    assert resolved.value.providers.library_paths["metis"] == ""
    origins = {origin.path: origin for origin in resolved.origins}
    assert origins["runtime.enable_x64"].environment_keys == ()
    assert origins["numerics.enable_x64"].environment_keys == ("JAX_ENABLE_X64",)
    assert origins["providers.library_paths"].environment_keys == (
        "PHYDRAX_METIS_LIBRARY",
    )
    assert pxconfig.ResolvedConfig.from_record(Composed, resolved.to_record()) == resolved


def test_native_and_declared_environment_overlap_is_refused() -> None:
    @dataclass(frozen=True)
    class Composed:
        __config_kind__: ClassVar[str] = "tests.configuration.overlapping"
        __config_environment__: ClassVar[Mapping[str, str]] = {
            "runtime.enable_x64": "EXPLICIT_X64"
        }
        runtime: pxconfig.RuntimeConfig = pxconfig.RuntimeConfig()

    with pytest.raises(ValueError, match="overlapping"):
        pxconfig.resolve(
            Composed, environment={"JAX_ENABLE_X64": "true", "EXPLICIT_X64": "false"}
        )


@pytest.mark.parametrize("container", ["tuple", "mapping"], ids=["tuple", "mapping"])
def test_records_inside_containers_are_validated_and_detached(container: str) -> None:
    @dataclass(frozen=True)
    class Root:
        __config_kind__: ClassVar[str] = "tests.configuration.record-containers"
        sequence: tuple[Solver, ...] = ()
        mapping: Mapping[str, Solver] = field(default_factory=dict)

    invalid = Solver(iterations=True)
    values: dict[str, object] = (
        {"sequence": (invalid,)} if container == "tuple" else {"mapping": {"a": invalid}}
    )
    with pytest.raises(TypeError, match="iterations"):
        pxconfig.resolve(Root, values=values, environment={})
    supplied = Solver()
    resolved = pxconfig.resolve(
        Root, values={"sequence": (supplied,), "mapping": {"a": supplied}}, environment={}
    )
    assert resolved.value.sequence[0] == supplied
    assert resolved.value.sequence[0] is not supplied
    assert resolved.value.mapping["a"] is not supplied
