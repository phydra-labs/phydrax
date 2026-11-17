from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

from ._registry import NATIVE_CASE_REGISTRY, NATIVE_SCENARIOS, OPTIONAL_SCENARIOS
from ._runtime import runtime_identity


SCENARIOS = (*NATIVE_SCENARIOS, *OPTIONAL_SCENARIOS, "native-corpus")
# Scenarios whose independent profiles can run as explicit subsets, with the
# owner function that combines the records of such runs.
_PROFILE_MERGERS = {
    "native-cad-profile-matrix": (
        ".scenarios.native_cad_profiles",
        "merge_native_cad_profile_selections",
    ),
    "native-hybrid-layer-matrix": (
        ".scenarios.native_hybrid_profiles",
        "merge_native_hybrid_layer_profile_selections",
    ),
    "native-implicit-adaptation": (
        ".scenarios.native_implicit",
        "merge_native_implicit_adaptation_selections",
    ),
    "native-reconstruction": (
        ".scenarios.reconstruction",
        "merge_native_reconstruction_selections",
    ),
}
_PROFILE_SELECTABLE = tuple(_PROFILE_MERGERS)


def _invoke(
    name: str,
    arguments: argparse.Namespace,
    /,
    *,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    profiles: frozenset[str] | None = None,
) -> dict[str, object]:
    def report_profile(event: dict[str, object]) -> None:
        if checkpoint is not None:
            checkpoint({**event, "scenario": name})

    try:
        if name in NATIVE_CASE_REGISTRY:
            from importlib import import_module

            module_name, function_name = NATIVE_CASE_REGISTRY[name]
            function = getattr(import_module(module_name, __package__), function_name)
            if profiles is not None and name not in _PROFILE_SELECTABLE:
                raise ValueError(
                    f"Qualification scenario {name!r} does not support profile selection."
                )
            if name in _PROFILE_SELECTABLE:
                record = function(
                    checkpoint=report_profile if checkpoint is not None else None,
                    profiles=profiles,
                )
            elif name in (
                "native-general-families",
                "native-planar-feature-matrix",
                "native-periodic-profile-matrix",
                "native-constrained-solids",
                "native-design-loop",
                "native-tpms",
                "native-polyhedral",
                "native-moving-overset",
                "native-image-material",
                "native-distributed-lifecycle",
                "native-cold-fe-fv-continuation",
            ):
                record = function(
                    checkpoint=report_profile if checkpoint is not None else None
                )
            else:
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
            "traceback": traceback.format_exc(),
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


def _write_event(stream: TextIO, event: dict[str, object], /, *, durable: bool) -> None:
    stream.write(json.dumps(event, default=_json_default, allow_nan=False) + "\n")
    stream.flush()
    if durable:
        os.fsync(stream.fileno())


def run(
    scenarios: tuple[str, ...],
    arguments: argparse.Namespace,
    /,
    *,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    profiles: frozenset[str] | None = None,
) -> dict[str, object]:
    if not scenarios:
        raise ValueError("Qualification run requires at least one explicit scenario.")
    selected: list[str] = []
    for name in scenarios:
        expanded = NATIVE_SCENARIOS if name == "native-corpus" else (name,)
        for item in expanded:
            if item not in selected:
                selected.append(item)
    distributed = "native-distributed-lifecycle"
    combined = "native-cold-fe-fv-continuation"
    shared_continuation = distributed in selected and combined in selected
    records: dict[str, dict[str, object]] = {}
    for name in selected:
        if shared_continuation and name == distributed:
            continue
        if checkpoint is not None:
            checkpoint({"event": "scenario-start", "scenario": name})
        record = _invoke(name, arguments, checkpoint=checkpoint, profiles=profiles)
        records[name] = record
        if checkpoint is not None:
            checkpoint({"event": "scenario-complete", "scenario": name, "record": record})
    if shared_continuation:
        component = records[combined].get("owner_local_fe_continuation")
        if (
            isinstance(component, dict)
            and isinstance(component.get("status"), str)
            and component["status"]
            and component.get("scenario", distributed) == distributed
        ):
            records[distributed] = {**component, "scenario": distributed}
        else:
            records[distributed] = {
                "scenario": distributed,
                "status": "failed",
                "detail": "Combined cold continuation omitted its canonical distributed FE result.",
                "runtime": records[combined]["runtime"],
            }
        if checkpoint is not None:
            checkpoint(
                {
                    "event": "scenario-complete",
                    "scenario": distributed,
                    "derived_from": combined,
                    "record": records[distributed],
                }
            )
        records = {name: records[name] for name in selected}
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


def merge_selections(name: str, paths: Sequence[Path], /) -> dict[str, object]:
    """Combine selected-profile result files of one scenario into one result.

    The files are final outputs of ``--profile`` runs under one runtime
    identity. The result has the shape ``run`` returns for the profiles they
    cover; the scenario owner decides coverage and completion.
    """
    if name not in _PROFILE_MERGERS:
        raise ValueError(
            f"Qualification scenario {name!r} does not support profile selection."
        )
    if not paths:
        raise ValueError("Merging requires at least one selected-profile result.")
    from importlib import import_module

    module_name, function_name = _PROFILE_MERGERS[name]
    merge = getattr(import_module(module_name, __package__), function_name)
    selections: list[dict[str, object]] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        scenarios = document.get("scenarios") if isinstance(document, dict) else None
        selection = scenarios.get(name) if isinstance(scenarios, dict) else None
        if not isinstance(selection, dict):
            raise ValueError(f"{path} holds no {name!r} result.")
        selections.append(selection)
    runtime = selections[0].get("runtime")
    if runtime is None or any(
        selection.get("runtime") != runtime for selection in selections
    ):
        raise ValueError("Selected-profile results must share one runtime identity.")
    record = {**merge(selections), "runtime": runtime, "scenario": name}
    return {
        "status": "passed" if record["status"] == "passed" else "failed",
        "scenarios": {name: record},
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
    parser.add_argument(
        "--events",
        type=Path,
        help="Durable scenario/profile JSONL receipts; requires a fresh file.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        help="Run only this profile of a scenario that supports profile selection; repeatable.",
    )
    parser.add_argument(
        "--merge",
        action="append",
        type=Path,
        help="Combine this selected-profile result of the one requested scenario instead of running it; repeatable.",
    )
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.list:
        print("\n".join(SCENARIOS))
        return 0
    selected = tuple(arguments.scenario or ("native-corpus",))
    if arguments.merge:
        if len(selected) != 1 or arguments.profile or arguments.events is not None:
            raise ValueError(
                "--merge combines selections of exactly one --scenario and takes no --profile or --events."
            )
        record = merge_selections(selected[0], tuple(arguments.merge))
    else:
        event_path = arguments.events
        if event_path is None and arguments.output is not None:
            event_path = arguments.output.with_name(
                arguments.output.name + ".profiles.jsonl"
            )
        if (
            event_path is not None
            and arguments.output is not None
            and event_path.resolve() == arguments.output.resolve()
        ):
            raise ValueError("Event receipts and final output require distinct paths.")
        with ExitStack() as stack:
            stream = (
                stack.enter_context(event_path.open("x", encoding="utf-8", newline="\n"))
                if event_path is not None
                else sys.stderr
            )

            def checkpoint(event: dict[str, object]) -> None:
                _write_event(stream, event, durable=event_path is not None)

            record = run(
                selected,
                arguments,
                checkpoint=checkpoint,
                profiles=frozenset(arguments.profile) if arguments.profile else None,
            )
    rendered = json.dumps(
        record, default=_json_default, indent=2, sort_keys=True, allow_nan=False
    )
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    scenario_records = record.get("scenarios")
    if not isinstance(scenario_records, dict):
        raise RuntimeError("Qualification run omitted its scenario record mapping.")
    if any(
        isinstance(value, dict) and value.get("status") == "missing-dependency"
        for value in scenario_records.values()
    ):
        return 2
    return 0 if record["status"] == "passed" else 1
