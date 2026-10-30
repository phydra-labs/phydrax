from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._registry import NATIVE_CASE_REGISTRY, NATIVE_SCENARIOS, OPTIONAL_SCENARIOS
from ._runtime import runtime_identity


SCENARIOS = (*NATIVE_SCENARIOS, *OPTIONAL_SCENARIOS, "native-corpus")


def _invoke(name: str, arguments: argparse.Namespace, /) -> dict[str, object]:
    try:
        if name in NATIVE_CASE_REGISTRY:
            from importlib import import_module

            module_name, function_name = NATIVE_CASE_REGISTRY[name]
            function = getattr(import_module(module_name, __package__), function_name)
            record = function()
        elif name in OPTIONAL_SCENARIOS:
            from .scenarios.providers import run_optional

            record = run_optional(name, arguments)
        else:
            raise ValueError(f"Unknown meshing qualification scenario {name!r}.")
        status = record.get("status")
        if not isinstance(status, str) or not status:
            raise RuntimeError(
                f"Qualification scenario {name!r} omitted an explicit nonempty status."
            )
        reported_scenario = record.get("scenario")
        if reported_scenario is not None and reported_scenario != name:
            raise RuntimeError(
                f"Qualification scenario {name!r} reported noncanonical identity "
                f"{reported_scenario!r}."
            )
        record.setdefault("runtime", runtime_identity())
        return {**record, "scenario": name}
    except Exception as error:
        return {
            "scenario": name,
            "status": "failed",
            "error_type": type(error).__name__,
            "detail": str(error),
            "runtime": runtime_identity(),
        }


def _json_default(value: Any, /) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Qualification record contains unsupported {type(value).__name__}.")


def run(
    scenarios: tuple[str, ...], arguments: argparse.Namespace, /
) -> dict[str, object]:
    if not scenarios:
        raise ValueError("Qualification run requires at least one explicit scenario.")
    selected: list[str] = []
    for name in scenarios:
        expanded = NATIVE_SCENARIOS if name == "native-corpus" else (name,)
        for item in expanded:
            if item not in selected:
                selected.append(item)
    records = {name: _invoke(name, arguments) for name in selected}
    status = (
        "passed"
        if all(value["status"] == "passed" for value in records.values())
        else "failed"
    )
    return {
        "status": status,
        "scenarios": records,
        "runtime": runtime_identity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run scientific native meshing qualification or an explicit optional-provider comparison."
    )
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument("--executable", default="vc_mesh")
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.list:
        print("\n".join(SCENARIOS))
        return 0
    selected = tuple(arguments.scenario or ("native-corpus",))
    record = run(selected, arguments)
    rendered = json.dumps(
        record, default=_json_default, indent=2, sort_keys=True, allow_nan=False
    )
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    scenario_records = record.get("scenarios")
    if not isinstance(scenario_records, dict):
        raise RuntimeError("Qualification run omitted its scenario record mapping.")
    if any(
        isinstance(value, dict) and value.get("status") == "missing-dependency"
        for value in scenario_records.values()
    ):
        return 2
    return 0 if record["status"] == "passed" else 1
