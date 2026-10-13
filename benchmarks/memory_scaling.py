#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Recompute committed memory evidence without running benchmark workloads.

Adapters identify scientific cases explicitly; source driver digests are provenance,
not equality requirements (drivers grow new evidence after baseline capture). Only
identical environment fingerprints and matching cases/evidence may be compared.
Timing and RSS are deliberately outside this ratchet.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    MemoryGatePolicy,
)


def _integers(values: Sequence[int], name: str) -> None:
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} must contain nonnegative integer counts.")


def compare_gate(
    capacities: Sequence[int],
    baseline: Sequence[int],
    candidate: Sequence[int],
    declared_decision: str,
    maximum_candidate_fraction: float = 0.5,
) -> dict[str, Any]:
    """Recompute the shared gate and refuse inconsistent declared decisions."""
    _integers(capacities, "Capacities")
    _integers(baseline, "Baseline")
    _integers(candidate, "Candidate")
    if not math.isfinite(maximum_candidate_fraction):
        raise ValueError("The candidate fraction must be finite.")
    result = evaluate_memory_gate(
        capacities, baseline, candidate, MemoryGatePolicy(maximum_candidate_fraction)
    ).to_dict()
    if result["decision"] != declared_decision:
        raise ValueError(
            f"Declared decision {declared_decision!r} disagrees with recomputed "
            f"memory gate {result['decision']!r}."
        )
    return result


def compare_growth(
    capacities: Sequence[int], values: Sequence[int], maximum_exponent: float
) -> dict[str, Any]:
    """Apply a dimensionless growth envelope using the shared log-log fit."""
    _integers(capacities, "Capacities")
    _integers(values, "Byte counts")
    if not math.isfinite(maximum_exponent):
        raise ValueError("The maximum exponent must be finite.")
    exponent = fitted_scaling_exponent(capacities, values)
    if exponent > maximum_exponent + 1.0e-12:
        raise ValueError(f"Memory growth exponent {exponent} exceeds {maximum_exponent}.")
    return {"exponent": exponent, "maximum_exponent": maximum_exponent}


def _value(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing evidence field {path!r}.")
        value = value[key]
    return value


def _rows(record: dict[str, Any], path: str) -> list[dict[str, Any]]:
    value = _value(record, path)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, dict) for row in value)
    ):
        raise ValueError(f"{path!r} must be a nonempty array of evidence objects.")
    return value


def _groups(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        identity = tuple(_value(row, key) for key in keys)
        groups.setdefault(identity, []).append(row)
    return groups


@dataclass
class Case:
    name: str
    rows: list[dict[str, Any]]
    capacity: str = "capacity"
    metric: str = "compiler.temporary_bytes"
    maximum_exponent: float | None = None
    maximum_ratio: float | None = None
    gate: dict[str, Any] | None = None
    baseline_name: str | None = None
    paired: list[dict[str, Any]] | None = None
    paired_metric: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    disposition: str = "unconditional"
    acceptance: dict[str, Any] | None = None


def _gate(record: dict[str, Any], path: str) -> dict[str, Any]:
    value = _value(record, path)
    if not isinstance(value, dict) or value.get("decision") not in (
        "adopt",
        "retain-baseline",
    ):
        raise ValueError(f"{path!r} must declare an adoption decision.")
    return value


def _nested_route_rows(rows: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    """Retain shared workload controls, overriding only recorded route evidence."""
    result: list[dict[str, Any]] = []
    for row in rows:
        route = _value(row, path)
        if not isinstance(route, dict):
            raise ValueError(f"{path!r} must contain route evidence.")
        result.append({**row, **route})
    return result


def _fitting_case(
    name: str,
    rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    capacity: str = "capacity",
    *,
    require_equal: bool = True,
) -> Case:
    """Check same-run fitting evidence separately from above-budget adoption."""
    capacities, values = _series(rows, capacity, "compiler.temporary_bytes")
    old_capacities, old = _series(paired, capacity, "compiler.temporary_bytes")
    if capacities != old_capacities:
        raise ValueError(f"Fitting capacities do not match {name}.")
    _match_controls(rows, paired)
    if require_equal and values != old:
        raise ValueError(f"Below-budget compiler equality disagrees for {name}.")
    return Case(
        f"fitting/{name}",
        rows,
        capacity,
        disposition="evidence-only",
        acceptance={
            "accepted": True,
            "comparison_policy": (
                "compiler-equality" if require_equal else "single-tile-route"
            ),
            "capacities": capacities,
            "default_temporary_bytes": values,
            "dense_temporary_bytes": old,
        },
    )


def _budget_simple(record: dict[str, Any], family: str) -> list[Case]:
    rows = _rows(record, "cases")
    if family in ("geometry_interface", "graph_queries"):
        gate_rows = _rows(record, "gate_cases")
        case = Case(
            "main",
            _nested_route_rows(gate_rows, "default"),
            gate=_gate(record, "gate"),
            paired=_nested_route_rows(gate_rows, "explicit_full_tile"),
            maximum_ratio=1.25 if family == "geometry_interface" else None,
        )
        if family == "geometry_interface":
            sources = _value(record, "configuration.sources")
            budget = _value(record, "configuration.interface_workspace_bytes")
            for row in rows:
                source_tile, target_tile = row["executed_tiles"]
                if row["logical_pair_tile_bytes"] != 8 * source_tile * target_tile:
                    raise ValueError(
                        "Interface logical tile bytes disagree with executed tiles."
                    )
                if source_tile != sources or target_tile != min(
                    row["capacity"], max(1, budget // (8 * sources))
                ):
                    raise ValueError(
                        "Interface executed tiles disagree with the default budget."
                    )
            fitting = [
                row for row in rows if row["executed_tiles"] == [sources, row["capacity"]]
            ]
            paired = _nested_route_rows(fitting, "paired_baseline")
            logical = "logical_pair_tile_bytes"
        else:
            for row in rows:
                source_tile, target_tile = (
                    row["source_chunk_size"],
                    row["target_chunk_size"],
                )
                logical_bytes = target_tile * (
                    source_tile * 29 + row["neighbor_count"] * 24
                )
                if row["logical_block_pair_bytes"] != logical_bytes:
                    raise ValueError(
                        "Graph logical tile bytes disagree with the query formula."
                    )
            fitting = [
                row
                for row in rows
                if row["source_chunk_size"] == row["capacity"]
                and row["target_chunk_size"] == row["capacity"]
            ]
            paired = _nested_route_rows(fitting, "whole_axis")
            logical = "logical_block_pair_bytes"
        return [
            case,
            Case("runtime/main", rows, metric=logical, disposition="evidence-only"),
            _fitting_case("main", fitting, paired),
        ]

    capacity = "time_count" if family == "ringdown" else "capacity"
    nested = "full_response" if family == "ringdown" else "parallel"
    logical = (
        "logical_response_bytes"
        if family == "ringdown"
        else "logical_frequency_transition_workspace_bytes"
    )
    fitting: list[dict[str, Any]] = []
    above: list[dict[str, Any]] = []
    for row in rows:
        count = row[capacity]
        if family == "ringdown":
            full_bytes = _value(row, f"{nested}.{logical}")
            per_item = 16 * row["mode_count"] ** 3
            if full_bytes != per_item * count:
                raise ValueError(
                    "Ringdown full response disagrees with the complex128 mode formula."
                )
            budget = _value(record, "configuration.response_workspace_bytes")
            requested = _value(record, "configuration.time_chunk_size")
            chunk_field, route_field = "time_chunk_size", "execution"
            route_names = ("vmap", "lax.map")
        else:
            per_item = (
                16
                * _value(record, "configuration.k_count")
                * _value(record, "configuration.bands") ** 2
                * _value(record, "configuration.dimension") ** 2
            )
            budget = _value(record, "configuration.frequency_workspace_bytes")
            requested = _value(record, "configuration.frequency_chunk_size")
            chunk_field, route_field = "frequency_chunk_size", "frequency_execution"
            route_names = ("parallel", "chunked")
        chunk = min(count, max(1, budget // per_item) if requested is None else requested)
        if (
            row[chunk_field] != chunk
            or row[logical] != chunk * per_item
            or row[route_field] != route_names[0 if chunk == count else 1]
        ):
            raise ValueError(
                f"Declared budget route or logical workspace disagrees for {family}."
            )
        (fitting if chunk == count else above).append(row)
    runtime = Case(
        "runtime/main",
        rows,
        capacity,
        logical,
        maximum_ratio=2.25 if family == "ringdown" else None,
        metrics={"compiler.temporary_bytes": None},
        disposition="evidence-only",
    )
    cases = [
        runtime,
        _fitting_case("main", fitting, _nested_route_rows(fitting, nested), capacity),
    ]
    if above:
        case = Case(
            "main",
            above,
            capacity,
            gate=_gate(record, "gate"),
            paired=_nested_route_rows(above, nested),
            maximum_ratio=2.25 if family == "ringdown" else None,
            metrics={logical: 0.0},
        )
        if family == "periodic_kubo":
            capacities, temporary = _series(above, capacity, case.metric)
            _, workspace = _series(above, capacity, logical)
            compiler_exponent = fitted_scaling_exponent(capacities, temporary)
            logical_exponent = fitted_scaling_exponent(capacities, workspace)
            declared = _value(record, "scaling.declared_logical_exponent")
            bound = _value(record, "scaling.declared_compiler_exponent_upper_bound")
            case.acceptance = {
                "accepted": abs(logical_exponent - declared) <= 0.1
                and compiler_exponent <= bound,
                "logical_exponent": logical_exponent,
                "declared_logical_exponent": declared,
                "compiler_exponent": compiler_exponent,
                "maximum_compiler_exponent": bound,
                "maximum_logical_exponent_defect": 0.1,
            }
        cases.append(case)
    return cases


def _simple(record: dict[str, Any], family: str) -> list[Case]:
    if family in ("ringdown", "geometry_interface", "graph_queries", "periodic_kubo"):
        return _budget_simple(record, family)
    # Capacity, owned workspace, and dimensionless envelopes are driver contracts,
    # not machine-specific portable byte ceilings.
    specifications = {
        "limber": ("records", "capacity", "logical_workspace_bytes", 1.0, "memory_gate"),
        "polymer_lag": ("rows", "capacity", "logical_workspace_bytes", 1.0, "gate"),
        "fel_wake": ("cases", "capacity", "compiler.temporary_bytes", None, "gate"),
        "phase_field_defects": ("cases", "count", "logical_workspace_bytes", 1.0, None),
        "csr": ("rows", "capacity", "row_workspace_bytes", 1.0, None),
        "csr_capacity": ("rows", "capacity", "row_workspace_bytes", 1.0, None),
        "linalg_solves": ("cases", "rhs_count", "compiler.temporary_bytes", None, "gate"),
        "linalg_spectral": ("cases", "probes", "compiler.temporary_bytes", None, "gate"),
        "kernel_correction": (
            "cases",
            "rows",
            "logical_probe_workspace_bytes",
            1.0,
            None,
        ),
        "structured_transfer": ("records", "n", "retained_bytes", None, None),
        "cut_cochain_transfer": ("cases", "capacity", "retained_bytes", None, None),
        "field_view": ("cases", "size", "retained_bytes", None, None),
        "polynomial_preparation": ("rows", "lanes", "live_bytes", 1.0, "gate"),
        "open_loop_games": (
            "cases",
            "capacity",
            "compiler.temporary_bytes",
            None,
            "gate",
        ),
        "open_loop_games_players": (
            "cases",
            "capacity",
            "compiler.temporary_bytes",
            None,
            None,
        ),
        "covariance_estimators": ("records", "dimension", "retained_bytes", None, None),
        "reactor_diffusion": ("records", "cells", "retained_bytes", None, None),
    }
    path, capacity, metric, maximum, gate_path = specifications[family]
    rows = _rows(record, path)
    case = Case("main", rows, capacity, metric, maximum)
    if gate_path is not None:
        case.gate = _gate(record, gate_path)
    if metric != "compiler.temporary_bytes" and all(
        "compiler" in row and row["compiler"] is not None for row in rows
    ):
        case.metrics["compiler.temporary_bytes"] = None
    if family in ("csr", "csr_capacity", "phase_field_defects"):
        case.metrics["retained_bytes"] = None
    if family == "structured_transfer":
        case.metrics["host_peak_bytes"] = None
    if family == "reactor_diffusion":
        case.metrics["host_peak_bytes"] = None
        for row in rows:
            cells, groups = row["cells"], row["groups"]
            if row["sparse_capacity"] != cells * groups**2 + 2 * (cells - 1) * groups:
                raise ValueError("Reactor sparse capacity disagrees with the stencil.")
    return [case]


def _named_cases(record: dict[str, Any], family: str) -> list[Case]:
    cases: list[Case] = []
    if family == "sensitivity_optics":
        names = (
            "sensitivity",
            "optics_batch1",
            "optics_batch16",
            "optics_batch256",
            "optics_batch1024",
        )
        if set(_value(record, "cases")) != set(names):
            raise ValueError("Unsupported sensitivity/optics case identity.")
        for name in names:
            cases.append(
                Case(
                    name,
                    _rows(record, f"cases.{name}"),
                    gate=_gate(record, f"gates.{name}"),
                )
            )
    elif family == "ocean_shell":
        if set(_value(record, "records")) != {"ocean", "shell"}:
            raise ValueError("Unsupported ocean/shell case identity.")
        for name in ("ocean", "shell"):
            cases.append(
                Case(
                    name,
                    _rows(record, f"records.{name}"),
                    metric="temporary_bytes",
                    gate=_gate(record, f"gates.{name}"),
                )
            )
    elif family in (
        "ml_neighbors",
        "contact_london_ptv",
        "optim_curvature",
        "optim_curvature_factor_batches",
    ):
        path, identity, gate_path = {
            "ml_neighbors": ("records", "name", "gates"),
            "contact_london_ptv": ("cases", "area", "gates"),
            "optim_curvature": ("cases", "name", "compiler_gates"),
            "optim_curvature_factor_batches": ("cases", "name", "compiler_gates"),
        }[family]
        allowed = {
            "ml_neighbors": {"knn", "kde", "dbscan", "spectral"},
            "contact_london_ptv": {"contact", "ptv", "london"},
            "optim_curvature": {"curvature", "sensitivity", "sensitivity-certification"},
            "optim_curvature_factor_batches": {
                "curvature",
                "sensitivity",
                "curvature-batch-1",
                "curvature-batch-4",
            },
        }[family]
        for (name,), rows in _groups(_rows(record, path), (identity,)).items():
            if name not in allowed:
                raise ValueError(f"Unsupported {family} case {name!r}.")
            gate = (
                None
                if name in ("spectral", "sensitivity-certification")
                else _gate(record, f"{gate_path}.{name}")
            )
            metric = (
                "temporary_bytes"
                if family == "ml_neighbors"
                else "compiler.temporary_bytes"
            )
            case = Case(name, rows, metric=metric, gate=gate)
            if family == "ml_neighbors" and name == "spectral":
                case.metrics["logical_workspace_bytes"] = None
            if family == "contact_london_ptv":
                case.metrics[
                    "logical_workspace_bytes"
                    if name != "london"
                    else "retained_plan_bytes"
                ] = None
            if family.startswith("optim_curvature"):
                case.metrics["logical_bytes"] = None
                if name.startswith("curvature-batch-"):
                    case.baseline_name = "curvature"
                    case.paired = [
                        row for row in _rows(record, path) if row["name"] == "curvature"
                    ]
            cases.append(case)
    elif family == "astrodynamics_pairs":
        above = _groups(_rows(record, "above_budget_rows"), ("route",))
        parallel = _groups(_rows(record, "parallel_comparison_rows"), ("route",))
        fitting = _groups(
            [row for row in _rows(record, "rows") if row["steps"] == 0], ("route",)
        )
        fitting_parallel = _groups(
            _rows(record, "fitting_parallel_comparison_rows"), ("route",)
        )
        expected = {("direct",), ("near",)}
        if any(
            set(groups) != expected
            for groups in (above, parallel, fitting, fitting_parallel)
        ):
            raise ValueError("Unsupported astrodynamics route identity.")
        for groups, full_tile in (
            (above, False),
            (parallel, True),
            (fitting, False),
            (fitting_parallel, True),
        ):
            for rows in groups.values():
                for row in rows:
                    count = row["capacity"]
                    tile = (
                        count
                        if full_tile
                        else max(
                            1, min(count, row["pair_workspace_bytes"] // (48 * count))
                        )
                    )
                    if (
                        row["steps"] != 0
                        or row["logical_pair_workspace_bytes"] != 48 * count * tile
                    ):
                        raise ValueError(
                            "Astrodynamics logical pair workspace disagrees with its budget."
                        )
        for (route,), rows in above.items():
            cases.append(
                Case(
                    route,
                    rows,
                    gate=_gate(record, f"gates.{route}"),
                    paired=parallel[route,],
                    metrics={"logical_pair_workspace_bytes": None},
                )
            )
            cases.append(_fitting_case(route, fitting[route,], fitting_parallel[route,]))
    elif family == "operator_workspace":
        if set(_value(record, "cases")) != {"D10", "D11", "D12", "D13"}:
            raise ValueError("Unsupported operator workspace identity.")
        for name in ("D10", "D11", "D12", "D13"):
            gate = _gate(record, f"gates.{name}")
            rows = _rows(record, f"cases.{name}")
            capacities, logical = _series(
                rows, "capacity", "logical_candidate_workspace_bytes"
            )
            _, temporary = _series(rows, "capacity", "compiler.temporary_bytes")
            logical_exponent = fitted_scaling_exponent(capacities, logical)
            compiler_exponent = (
                fitted_scaling_exponent(capacities, temporary) if all(temporary) else None
            )
            acceptance = (
                compiler_exponent is not None
                and abs(compiler_exponent - logical_exponent) <= 0.25
            )
            fraction_gate = dict(gate, decision=gate["fraction_decision"])
            decision = fraction_gate["decision"] if acceptance else "retain-baseline"
            if decision != gate["decision"]:
                raise ValueError(
                    f"Declared operator scaling decision disagrees for {name}."
                )
            case = Case(
                name,
                rows,
                gate=fraction_gate,
                metrics={"logical_candidate_workspace_bytes": None},
            )
            case.acceptance = {
                "compiler_exponent": compiler_exponent,
                "logical_exponent": logical_exponent,
                "maximum_exponent_defect": 0.25,
                "decision": decision,
            }
            cases.append(case)
    return cases


def _budget_gp(record: dict[str, Any]) -> list[Case]:
    rows = _rows(record, "records")
    cases: list[Case] = []
    default_budget = _value(record, "default_workspace_bytes")
    for family, name, width in (
        ("lanczos", "main", 1),
        ("variational", "variational", 4),
    ):
        routes = {
            route: _nested_route_rows(rows, f"{family}.{route}")
            for route in ("dense", "default_policy", "restricted_budget")
        }
        for route, evidence in routes.items():
            for row in evidence:
                per_row = (row["n"] + width) * 8
                budget = row["workspace_budget_bytes"]
                expected_budget = (
                    None
                    if route == "dense"
                    else default_budget
                    if route == "default_policy"
                    else per_row
                )
                batch = (
                    row["n"]
                    if budget is None
                    else min(row["n"], max(1, budget // per_row))
                )
                if budget != expected_budget or row["row_batch_size"] != batch:
                    raise ValueError(
                        f"GP {family}/{route} row_batch_size disagrees with the workspace formula."
                    )
            cases.append(
                Case(
                    f"routes/{family}/{route}",
                    evidence,
                    "n",
                    disposition="reference-only" if route == "dense" else "evidence-only",
                )
            )
        default, dense = routes["default_policy"], routes["dense"]
        fitting = [row for row in default if row["row_batch_size"] == row["n"]]
        fitting_dense = [
            row for row in dense if row["n"] in {entry["n"] for entry in fitting}
        ]
        for current, old in zip(fitting, fitting_dense):
            if _value(current, "compiler.output_bytes") != _value(
                old, "compiler.output_bytes"
            ) or current.get("retained_bytes") != old.get("retained_bytes"):
                raise ValueError(f"GP {family} single-tile output storage differs.")
        cases.append(
            _fitting_case(
                name, fitting, fitting_dense, "n", require_equal=family == "variational"
            )
        )
        above = [row for row in default if row["row_batch_size"] < row["n"]]
        if any(not row["compile_only"] for row in above):
            raise ValueError(
                "GP adoption evidence must use the declared compile-only cohort."
            )
        paired = [row for row in dense if row["n"] in {entry["n"] for entry in above}]
        cases.append(
            Case(
                name,
                above,
                "n",
                gate=_gate(record, "gate" if family == "lanczos" else "variational_gate"),
                paired=paired,
            )
        )
    return cases


def _budget_transport(record: dict[str, Any]) -> list[Case]:
    groups = _groups(_rows(record, "cases"), ("family", "requested_route"))
    expected = {
        (family, route)
        for family in ("attention", "balanced", "unbalanced")
        for route in ("dense", "default")
    }
    if set(groups) != expected:
        raise ValueError("Unsupported transport requested-route identity.")
    cases: list[Case] = []
    for family in ("attention", "balanced", "unbalanced"):
        default, dense = groups[family, "default"], groups[family, "dense"]
        budget_field = (
            "attention_workspace_bytes"
            if family == "attention"
            else "sinkhorn_workspace_bytes"
        )
        for route, rows in (("default", default), ("dense", dense)):
            for row in rows:
                full_bytes = (
                    (16 if family == "attention" else 32)
                    * row["query_count"]
                    * row["capacity"]
                )
                budget = row[budget_field]
                above = full_bytes > budget
                if (
                    row["full_pair_workspace_bytes"] != full_bytes
                    or row["above_budget"] != above
                ):
                    raise ValueError(
                        f"Transport {family} above-budget classification disagrees."
                    )
                tile = (
                    max(1, budget // (16 * row["query_count"]))
                    if family == "attention"
                    else max(1, budget // (32 * max(row["query_count"], row["capacity"])))
                )
                blocked = route == "default" and above
                logical_bytes = (
                    (
                        16 * row["query_count"] * tile
                        if family == "attention"
                        else 32 * tile**2
                    )
                    if blocked
                    else full_bytes
                )
                if (
                    row["route"] != ("blockwise" if blocked else "dense")
                    or row["block_size"] != (tile if blocked else None)
                    or row["logical_pair_workspace_bytes"] != logical_bytes
                ):
                    raise ValueError(
                        f"Transport {family} automatic tile disagrees with its budget."
                    )
            cases.append(
                Case(
                    f"routes/{family}/{route}",
                    rows,
                    metrics={"logical_pair_workspace_bytes": None},
                    disposition="reference-only" if route == "dense" else "evidence-only",
                )
            )
        fitting = [row for row in default if not row["above_budget"]]
        fitting_dense = [row for row in dense if not row["above_budget"]]
        for current, old in zip(fitting, fitting_dense):
            if current["output_bytes"] != old["output_bytes"] or current.get(
                "checksum"
            ) != old.get("checksum"):
                raise ValueError(f"Transport {family} full-fit output evidence differs.")
        cases.append(_fitting_case(family, fitting, fitting_dense))
        above = [row for row in default if row["above_budget"] and row["compile_only"]]
        paired = [row for row in dense if row["above_budget"] and row["compile_only"]]
        declared_capacities = _value(record, f"gates.{family}.capacities")
        if [row["capacity"] for row in above] != declared_capacities:
            raise ValueError(f"Transport {family} gate capacities disagree.")
        cases.append(
            Case(
                family,
                above,
                gate=_gate(record, f"gates.{family}.memory"),
                paired=paired,
                metrics={"logical_pair_workspace_bytes": None},
            )
        )
    return cases


def _budget_mac(record: dict[str, Any]) -> list[Case]:
    groups = _groups(_rows(record, "cases"), ("route",))
    if set(groups) != {("auto",), ("dense-svd",), ("block-banded",)}:
        raise ValueError("Unsupported MAC requested-route identity.")
    for (route,), rows in groups.items():
        for row in rows:
            above = row["dense_workspace_required_bytes"] > row["dense_workspace_bytes"]
            resolved = (
                ("block-banded" if above else "dense-svd") if route == "auto" else route
            )
            if row["resolved_route"] != resolved or row["gate_case"] != above:
                raise ValueError(
                    "MAC auto resolved_route or gate cohort disagrees with its budget."
                )
    auto, dense = groups["auto",], groups["dense-svd",]
    fitting = [row for row in auto if not row["gate_case"]]
    fitting_dense = [row for row in dense if not row["gate_case"]]
    above = [row for row in auto if row["gate_case"]]
    paired = [row for row in dense if row["gate_case"]]
    cases = [
        _fitting_case("auto", fitting, fitting_dense),
        Case(
            "auto-above-budget",
            above,
            gate=_gate(record, "memory_gate"),
            paired=paired,
            metrics={"logical_design_bytes": None},
        ),
        Case("reference/dense-svd-above-budget", paired, disposition="reference-only"),
    ]
    for (route,), rows in groups.items():
        small = [row for row in rows if not row["gate_case"]]
        cases.append(
            Case(
                f"routes/{route}",
                small,
                metrics={"logical_design_bytes": None},
                disposition="evidence-only" if route == "auto" else "reference-only",
            )
        )
    return cases


def _paired_cases(record: dict[str, Any], family: str) -> list[Case]:
    if family == "transport_attention":
        return _budget_transport(record)
    if family == "gp_actions":
        return _budget_gp(record)
    cases: list[Case] = []
    if family == "process_compression":
        identities = {
            "projection": ("selector", "production"),
            "contract": ("kronecker", "production"),
        }
        groups = _groups(_rows(record, "cases"), ("family", "route"))
        expected = {
            (name, route) for name, routes in identities.items() for route in routes
        }
        if set(groups) != expected:
            raise ValueError(f"Unsupported {family} route identity.")
        for name, (old, new) in identities.items():
            path = f"comparisons.{name}.memory"
            metrics = {
                "compiler.temporary_bytes": None,
                "logical_selector_bytes": 0.0,
                "logical_expanded_local_bytes": 0.0,
            }
            cases.append(
                Case(
                    name,
                    groups[name, new],
                    gate=_gate(record, path),
                    paired=groups[name, old],
                    metrics=metrics,
                )
            )
    elif family == "geometry_queries":
        rows = _rows(record, "cases")
        cases.append(
            Case(
                "main",
                rows,
                metric="candidate.compiler.temporary_bytes",
                gate=_gate(record, "gate"),
                paired=rows,
                paired_metric="dense.compiler.temporary_bytes",
            )
        )
        cases.append(
            Case("bvh", rows, metric="bvh.compiler.temporary_bytes", maximum_ratio=1.25)
        )
        cases.append(
            Case(
                "polygon_pairs",
                _rows(record, "polygon_pairs"),
                metric="retained_pair_bytes",
                maximum_exponent=0.0,
                metrics={"peak_bytes": None},
            )
        )
    elif family == "optim_interior_point":
        groups = _groups(_rows(record, "cases"), ("route", "batch_size"))
        reference = [
            row
            for row in _rows(record, "cases")
            if row["route"] == "all-derivative-trials"
        ]
        for gate in _rows(record, "gates"):
            batch = gate["batch_size"]
            cases.append(
                Case(
                    f"batch-{batch}",
                    groups["screened-bounded-derivatives", batch],
                    "size",
                    gate=gate,
                    paired=reference,
                    metrics={"logical_derivative_workspace_bytes": None},
                )
            )
    return cases


def _logarithmic_scan(
    capacities: list[int], values: list[int], points: int
) -> dict[str, Any]:
    """Recompute the BRep scan-only fit, excluding prepared topology and output."""
    _integers([points], "Query count")
    if points == 0:
        raise ValueError("BRep scan evidence requires query points.")
    x = [math.log2(capacity) for capacity in capacities]
    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(values) / len(values)
    slope = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, values)) / math.fsum(
        (a - x_mean) ** 2 for a in x
    )
    intercept = y_mean - slope * x_mean
    residuals = [
        (value - (intercept + slope * logarithm)) / max(value, 1)
        for logarithm, value in zip(x, values)
    ]
    normalized = [
        value / (points * (2.0 + logarithm)) for logarithm, value in zip(x, values)
    ]
    ratios = [later / earlier for earlier, later in zip(normalized, normalized[1:])]
    return {
        "formula": "O(Q*(k+log2(F))), k=2; prepared topology/output excluded",
        "accepted": len(capacities) >= 4
        and all(ratio <= 1.25 for ratio in ratios)
        and all(abs(residual) <= 0.25 for residual in residuals),
        "relative_residuals": residuals,
        "normalized_adjacent_ratios": ratios,
    }


def _geometry_cases(record: dict[str, Any], family: str) -> list[Case]:
    cases: list[Case] = []
    if family == "geometry_brep":
        rows = _rows(record, "cases")
        for name in ("nearest", "nearest_dynamic", "winding", "winding_dynamic"):
            gates = _value(record, f"gates.{name}")
            capacities, values = _series(
                rows, "faces", f"{name}.compiler.temporary_bytes"
            )
            if name.startswith("nearest"):
                _, scan = _series(rows, "faces", "nearest.compiler.temporary_bytes")
                acceptance: dict[str, Any] = _logarithmic_scan(
                    capacities, scan, _value(rows[0], "points")
                )
            else:
                ratios = [b / max(a, 1) for a, b in zip(values, values[1:])]
                acceptance = dict[str, Any](
                    accepted=all(ratio <= 1.25 for ratio in ratios),
                    maximum_adjacent_ratio=1.25,
                )
            gate = _gate(record, f"gates.{name}.memory_gate")
            decision = gate["decision"] if acceptance["accepted"] else "retain-baseline"
            if gates["decision"] != decision:
                raise ValueError(f"Declared geometry decision disagrees for {name}.")
            acceptance["decision"] = decision
            case = Case(
                name,
                rows,
                "faces",
                f"{name}.compiler.temporary_bytes",
                gate=gate,
                baseline_name=name.removesuffix("_dynamic"),
            )
            case.acceptance = acceptance
            cases.append(case)
    elif family == "periodic_ewald":
        rows = _rows(record, "cases")
        for gate in _rows(record, "gates"):
            shell, phase = gate["shell"], gate["phase"]
            selected = [row for row in rows if row["shell"] == shell]
            cases.append(
                Case(
                    f"shell-{shell}/{phase}",
                    selected,
                    "size",
                    f"phases.{phase}.compiler.temporary_bytes",
                    gate=gate,
                    metrics={
                        "logical_displacement_workspace_bytes": 0.0,
                        "logical_host_pair_workspace_bytes": 0.0,
                        "logical_retained_bytes": None,
                    },
                )
            )
    elif family == "host_exact":
        for name in ("quantum_hall", "scenarios"):
            cases.append(
                Case(
                    name,
                    _rows(record, name),
                    metric="host_allocation_peak_bytes",
                    gate=_gate(record, f"host_gates.{name}"),
                )
            )
    return cases


def _host_cases(record: dict[str, Any], family: str) -> list[Case]:
    cases: list[Case] = []
    if family in ("archive_streaming", "fingerprint_workset"):
        allowed = {
            "archive_streaming": {
                "archive-write",
                "archive-read",
                "checkpoint-read",
                "npz-export",
                "npz-read",
                "support-snapshot",
            },
            "fingerprint_workset": {
                "contiguous",
                "strided",
                "file-stream",
                "file-legacy-reference",
                "identity-stream",
                "identity-legacy-reference",
                "workset",
            },
        }[family]
        groups = _groups(_rows(record, "records"), ("case",))
        if {name for (name,) in groups} != allowed:
            raise ValueError(f"Unsupported {family} host operation.")
        for (name,), rows in groups.items():
            legacy = "legacy-reference" in name
            cases.append(
                Case(
                    name,
                    rows,
                    metric="scratch_peak_bytes",
                    maximum_ratio=1.5 if family == "archive_streaming" else None,
                    disposition="reference-only" if legacy else "unconditional",
                )
            )
    elif family == "host_caches":
        churn = _value(record, "churn")
        cap = churn["cap_bytes"]
        _integers([cap], "Cache budget")
        for row in _rows(record, "churn.records"):
            if _value(row, "retained_bytes") > 1.05 * cap:
                raise ValueError("Observed cache retention exceeds its declared budget.")
        for row in _rows(record, "hot_sets"):
            retained = _value(row, "retained_bytes")
            budget = _value(row, "configured_cap_bytes")
            entries = _value(row, "cache_entries")
            maximum = _value(row, "configured_max_entries")
            _integers([retained, budget, entries, maximum], "Cache evidence")
            if retained > budget or entries > maximum:
                raise ValueError(
                    "Observed hot-set retention exceeds its declared budget."
                )
        cases.append(
            Case("churn", _rows(record, "churn.records"), "keys", "retained_bytes")
        )
    return cases


def _extension_cases(record: dict[str, Any], family: str) -> list[Case]:
    cases: list[Case] = []
    if family == "banded_spectrum":
        for (route,), rows in _groups(_rows(record, "cases"), ("route",)).items():
            if route not in ("eigenvalue-brackets", "singular-threshold"):
                raise ValueError("Unsupported banded spectrum route.")
            cases.append(Case(route, rows, "rows", metrics={"logical_input_bytes": 1.0}))
    elif family == "linalg_operators":
        for name in ("selected", "certificate"):
            cases.append(
                Case(
                    name,
                    _rows(record, "cases"),
                    metric=f"{name}_logical_workspace_bytes",
                    metrics={f"{name}.compiler.temporary_bytes": None},
                )
            )
    elif family == "fem_assembly":
        for (route,), rows in _groups(_rows(record, "cases"), ("route",)).items():
            if route not in ("mass", "stiffness", "paired"):
                raise ValueError("Unsupported FEM assembly route.")
            cases.append(
                Case(
                    route,
                    rows,
                    "cells",
                    "retained_array_bytes",
                    metrics={"host_allocation_peak_bytes": None},
                )
            )
    elif family == "sbp_rows":
        for (order, periodic), rows in _groups(
            _rows(record, "cases"), ("order", "periodic")
        ).items():
            if order not in (2, 4, 6, 8) or type(periodic) is not bool:
                raise ValueError("Unsupported SBP row identity.")
            cases.append(
                Case(
                    f"order-{order}/periodic-{periodic}",
                    rows,
                    "count",
                    "retained_bytes",
                    metrics={"stencil_bytes": None, "host_peak_bytes": None},
                )
            )
    elif family == "spectral_split_form":
        for (chunk,), rows in _groups(_rows(record, "cases"), ("chunk",)).items():
            gates = [
                gate for gate in _rows(record, "compiler_gates") if gate["chunk"] == chunk
            ]
            if len(gates) != 1:
                raise ValueError("Split-form chunk must have one diagnostic gate.")
            cases.append(
                Case(
                    f"chunk-{chunk}",
                    rows,
                    "count",
                    "retained_split_bytes",
                    gate=gates[0],
                    metrics={"pair_workspace_bytes": None},
                )
            )
    elif family == "integration_lowering":
        cases.append(
            Case("main", _rows(record, "cases"), "terms", gate=_gate(record, "gate"))
        )
    elif family == "structured_laplace":
        for (mode,), rows in _groups(_rows(record, "records"), ("mode",)).items():
            if mode not in ("full", "diagonal", "lanczos", "lobpcg"):
                raise ValueError("Unsupported structured Laplace mode.")
            cases.append(Case(mode, rows, "dimension", "logical_output_bytes"))
    elif family in ("causal_adjoint", "control_horizon", "dae"):
        path = "records" if family == "control_horizon" else "rows"
        identity = (
            ("item", "route")
            if family == "control_horizon"
            else (
                ("dimension", "probe_batch_size", "reuse")
                if family == "dae"
                else ("route",)
            )
        )
        groups = _groups(_rows(record, path), identity)
        for key, rows in groups.items():
            case = Case("/".join(map(str, key)), rows, "horizon")
            if family == "control_horizon":
                item, route = key
                if (item, route) not in (
                    ("S10", "batched"),
                    ("S10", "reverse-scan"),
                    ("S10", "checkpointed-reverse-scan"),
                    ("S11", "dense"),
                    ("S11", "sparse"),
                ):
                    raise ValueError("Unsupported control horizon route.")
                if item == "S10":
                    case.metrics["logical_stage_model_bytes"] = (
                        1.0 if route == "batched" else 0.0
                    )
                    if route != "batched":
                        case.gate = _gate(record, f"gates.{route}")
                        case.paired = groups["S10", "batched"]
                        capacities, values = _series(rows, "horizon", case.metric)
                        if any(
                            b / a > 1.25 * d / c
                            for a, b, c, d in zip(
                                values, values[1:], capacities, capacities[1:]
                            )
                        ):
                            raise ValueError(
                                "Control horizon growth exceeds the declared linear ratio."
                            )
                else:
                    # Native QP preparation has host retention evidence, not an
                    # XLA compiler program. Sparse storage scales with horizon;
                    # the explicit dense alternative remains reference evidence.
                    case.metric = "retained_bytes"
                    case.maximum_exponent = 1.0 if route == "sparse" else None
                    case.metrics["host_peak_bytes"] = None
                    case.disposition = (
                        "unconditional" if route == "sparse" else "reference-only"
                    )
            elif family == "dae":
                dimension, batch, reuse = key
                if (
                    type(dimension) is not int
                    or dimension < 1
                    or batch not in (None, 1, 4)
                    or type(reuse) is not bool
                ):
                    raise ValueError("Unsupported DAE probe identity.")
                case.metrics["logical_evidence_output_bytes"] = None
                if batch == 1 and not reuse:
                    case.gate = _gate(record, "gate")
                    case.paired = groups[dimension, None, False]
            else:
                (route,) = key
                if route not in (
                    "dense-associative",
                    "action-scan",
                    "checkpointed-action-scan",
                ):
                    raise ValueError("Unsupported causal adjoint route.")
                case.metrics = {
                    "logical_action_carry_bytes": 0.0,
                    "logical_trajectory_bytes": 1.0,
                    "logical_transition_bytes": None,
                }
                case.disposition = (
                    "reference-only" if route == "dense-associative" else "unconditional"
                )
            cases.append(case)
    elif family == "mac_shadowing":
        return _budget_mac(record)
    elif family == "mac_shadowing_large":
        for (route,), rows in _groups(_rows(record, "cases"), ("route",)).items():
            if route not in ("block-banded", "dense-svd", "baseline"):
                raise ValueError("Unsupported MAC shadowing route.")
            cases.append(
                Case(
                    route,
                    rows,
                    disposition="reference-only"
                    if route != "block-banded"
                    else "unconditional",
                )
            )
    return cases


_SIMPLE = frozenset(
    (
        "limber",
        "polymer_lag",
        "fel_wake",
        "phase_field_defects",
        "csr",
        "csr_capacity",
        "ringdown",
        "geometry_interface",
        "graph_queries",
        "periodic_kubo",
        "linalg_solves",
        "linalg_spectral",
        "kernel_correction",
        "structured_transfer",
        "cut_cochain_transfer",
        "field_view",
        "polynomial_preparation",
        "open_loop_games",
        "open_loop_games_players",
        "covariance_estimators",
        "reactor_diffusion",
    )
)
_NAMED = frozenset(
    (
        "sensitivity_optics",
        "ocean_shell",
        "ml_neighbors",
        "contact_london_ptv",
        "optim_curvature",
        "optim_curvature_factor_batches",
        "astrodynamics_pairs",
        "operator_workspace",
    )
)
_PAIRED = frozenset(
    (
        "process_compression",
        "transport_attention",
        "geometry_queries",
        "gp_actions",
        "optim_interior_point",
    )
)
_GEOMETRY = frozenset(("geometry_brep", "periodic_ewald", "host_exact"))
_HOST = frozenset(("archive_streaming", "fingerprint_workset", "host_caches"))
_EXTENSIONS = frozenset(
    (
        "banded_spectrum",
        "linalg_operators",
        "fem_assembly",
        "sbp_rows",
        "spectral_split_form",
        "integration_lowering",
        "structured_laplace",
        "causal_adjoint",
        "control_horizon",
        "dae",
        "mac_shadowing",
        "mac_shadowing_large",
    )
)
_NAMED_BASELINES = frozenset(
    (
        "memory_mac_shadowing_baseline_large.json",
        "memory_mac_shadowing_baseline_realistic.json",
    )
)


def _adapt(record: dict[str, Any], family: str) -> list[Case]:
    if family in _SIMPLE:
        return _simple(record, family)
    if family in _NAMED:
        return _named_cases(record, family)
    if family in _PAIRED:
        return _paired_cases(record, family)
    if family in _GEOMETRY:
        return _geometry_cases(record, family)
    if family in _HOST:
        return _host_cases(record, family)
    if family in _EXTENSIONS:
        return _extension_cases(record, family)
    raise ValueError(f"Unsupported memory record family {family!r}.")


def _supplemental(record: dict[str, Any], family: str) -> list[Case]:
    """Give additional measured workloads their own explicit memory disposition."""
    cases: list[Case] = []
    if family == "astrodynamics_pairs":
        rows = [row for row in _rows(record, "rows") if row["steps"]]
        if rows:
            cases.append(
                Case(
                    "near/rollout",
                    rows,
                    "steps",
                    "logical_pair_workspace_bytes",
                    0.0,
                    metrics={"compiler.temporary_bytes": None},
                )
            )
    elif family == "contact_london_ptv":
        rows = _rows(record, "field_kernel_reference_comparison")
        cases.append(
            Case(
                "london_field",
                rows,
                metric="tiled.compiler.temporary_bytes",
                gate=_gate(record, "gates.london_field"),
                paired=rows,
                paired_metric="dense.compiler.temporary_bytes",
                metrics={"logical_tile_workspace_bytes": 1.0},
            )
        )
        for (route,), rows in _groups(
            _rows(record, "matrix_free_london"), ("route",)
        ).items():
            if route not in ("minres", "gmres"):
                raise ValueError("Unsupported matrix-free London route.")
            cases.append(
                Case(
                    f"matrix-free/{route}",
                    rows,
                    metric="logical_face_tile_bytes",
                    maximum_exponent=1.0,
                    metrics={
                        "retained_plan_bytes": None,
                        "compiler.temporary_bytes": None,
                    },
                )
            )
    elif family == "graph_queries":
        for path in ("source_scan_cases", "steady_source_scan_cases"):
            if path in record:
                cases.append(Case(path, _rows(record, path), "source_capacity"))
    elif family == "periodic_kubo":
        rows = _rows(record, "cases")
        if all("parallel" in row for row in rows):
            cases.append(
                Case("parallel", rows, metric="parallel.compiler.temporary_bytes")
            )
    elif family == "linalg_solves":
        cases.append(
            Case(
                "sparse-factor",
                _rows(record, "sparse"),
                "rhs_count",
                "host_peak_bytes",
                disposition="evidence-only",
            )
        )
        for name, capacity, metric, maximum in (
            ("logdet", "size", "diagonal_workspace_bytes", 1.0),
            ("nuisance", "size", "factor_bytes", None),
        ):
            cases.append(
                Case(name, _rows(record, f"auxiliary.{name}"), capacity, metric, maximum)
            )
        for orientation in ("tall", "wide"):
            rows = [
                row
                for row in _rows(record, "auxiliary.pseudoinverse")
                if (row["rows"] > row["columns"]) == (orientation == "tall")
            ]
            cases.append(
                Case(
                    f"pseudoinverse/{orientation}",
                    rows,
                    "rows" if orientation == "tall" else "columns",
                )
            )
    elif family == "linalg_spectral":
        cases.append(Case("inverse-diagonal", _rows(record, "inverse_cases"), "probes"))
        for (route,), rows in _groups(
            _rows(record, "resolvent_cases"), ("route",)
        ).items():
            if route not in ("inverse-reference", "rhs-action"):
                raise ValueError("Unsupported resolvent route.")
            cases.append(
                Case(
                    f"resolvent/{route}",
                    rows,
                    "size",
                    "action_rhs_workspace_bytes",
                    metrics={"compiler.temporary_bytes": None},
                    disposition="reference-only"
                    if route == "inverse-reference"
                    else "unconditional",
                )
            )
    elif family == "optim_interior_point":
        for (route, products), rows in _groups(
            _rows(record, "actions"), ("route", "products")
        ).items():
            if route not in ("prepared-actions", "reconstructed-actions"):
                raise ValueError("Unsupported interior-point action route.")
            cases.append(
                Case(
                    f"actions/{route}/products={products}",
                    rows,
                    "size",
                    disposition="evidence-only",
                )
            )
    elif family == "structured_laplace" and "diagonal_fit_latency" in record:
        cases.append(
            Case(
                "diagonal-fit",
                _rows(record, "diagonal_fit_latency"),
                "dimension",
                "routes.candidate.compiler.temporary_bytes",
                disposition="evidence-only",
            )
        )
    return cases


def _observed_metrics(case: Case) -> None:
    # These fields have the same meaning in all owning drivers. They are evidence,
    # never an inferred acceptance rule or a substitute for the adapter's metric.
    for name in (
        "host_peak_bytes",
        "host_allocation_peak_bytes",
        "host_preparation_peak_bytes",
        "preparation_host_peak_bytes",
        "retained_bytes",
        "retained_plan_bytes",
        "retained_input_bytes",
        "retained_output_bytes",
        "output_bytes",
        "logical_workspace_bytes",
        "logical_pair_workspace_bytes",
    ):
        if (
            name != case.metric
            and name not in case.metrics
            and all(name in row and row[name] is not None for row in case.rows)
        ):
            case.metrics[name] = None


def _unconditional_baseline(
    baseline: dict[str, Any], family: str, case: Case
) -> list[dict[str, Any]] | None:
    if case.disposition in ("evidence-only", "reference-only"):
        return None
    if family in _SIMPLE and case.name == "main":
        return _baseline_rows(baseline, family, case)[0]
    if family in ("archive_streaming", "fingerprint_workset"):
        rows = [row for row in _rows(baseline, "records") if row["case"] == case.name]
        return rows or None
    if family == "host_caches" and case.name == "churn":
        return _rows(baseline, "churn.records")
    if family in ("fem_assembly", "sbp_rows", "structured_laplace", "linalg_operators"):
        path = "records" if family == "structured_laplace" else "cases"
        rows = _rows(baseline, path)
        if family == "fem_assembly":
            return [row for row in rows if row["route"] == case.rows[0]["route"]]
        if family == "sbp_rows":
            return [
                row
                for row in rows
                if (row["order"], row["periodic"])
                == (case.rows[0]["order"], case.rows[0]["periodic"])
            ]
        if family == "structured_laplace":
            return [row for row in rows if row["mode"] == case.rows[0]["mode"]]
        return rows
    if family in ("mac_shadowing", "mac_shadowing_large"):
        rows = _rows(baseline, "cases")
        return [row for row in rows if row["route"] == "baseline"]
    return None


def _match_controls(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> None:
    # Capacity alone does not identify a scientific workload.
    controls = (
        "points",
        "mode_count",
        "neighbor_count",
        "wave_number_count",
        "representative_count",
        "flattened_factor_count",
        "batch",
        "shell",
        "dimension",
        "players",
        "samples",
        "rank",
        "bandwidth",
        "degree",
        "term_count",
        "node_count",
        "source_capacity",
        "target_capacity",
        "sample_count",
        "groups",
        "products",
        "query_count",
        "pair_workspace_bytes",
        "query_workspace_bytes",
        "full_pair_workspace_bytes",
        "sinkhorn_workspace_bytes",
        "attention_workspace_bytes",
        "frequency_workspace_bytes",
        "response_workspace_bytes",
        "dense_workspace_bytes",
        "dense_workspace_required_bytes",
    )
    for current, old in zip(candidate, baseline):
        for key in controls:
            if key in current and key in old and current[key] != old[key]:
                raise ValueError(f"Baseline workload control {key!r} does not match.")


def _match_configuration(
    candidate: dict[str, Any], baseline: dict[str, Any], family: str
) -> None:
    controls = {
        "periodic_kubo": ("k_count", "bands", "dimension"),
        "geometry_interface": ("sources", "chunk_size", "target_chunk_size"),
        "geometry_brep": ("points",),
    }.get(family, ())
    for key in controls:
        if _value(candidate, f"configuration.{key}") != _value(
            baseline, f"configuration.{key}"
        ):
            raise ValueError(f"Baseline configuration {key!r} does not match.")


def _summarize_case(
    case: Case, baseline: dict[str, Any] | None, family: str, comparable: bool
) -> dict[str, Any]:
    try:
        return _case_summary(case, baseline, family, comparable)
    except ValueError as error:
        raise ValueError(f"{family}/{case.name}: {error}") from error


def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Memory evidence must contain finite JSON numbers.")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def _series(
    rows: list[dict[str, Any]], capacity: str, metric: str
) -> tuple[list[int], list[int]]:
    capacities = [_value(row, capacity) for row in rows]
    values = [_value(row, metric) for row in rows]
    _integers(capacities, "Capacities")
    _integers(values, metric)
    # Shared validation also covers strict ordering, minimum sample count and zeros.
    evaluate_memory_gate(capacities, values, values, MemoryGatePolicy())
    return capacities, values


def _growth(
    capacities: list[int], values: list[int], maximum: float | None, ratio: float | None
) -> dict[str, Any]:
    result: dict[str, Any] = {"bytes": values, "disposition": "observed-only"}
    if all(values):
        result["exponent"] = fitted_scaling_exponent(capacities, values)
        if maximum is not None:
            result.update(compare_growth(capacities, values, maximum))
            result["disposition"] = "bounded-growth"
    elif any(values):
        if maximum is not None:
            raise ValueError(
                "A bounded growth series cannot mix zero and positive bytes."
            )
        result["exponent"] = None
    else:
        result["exponent"] = None
        result["disposition"] = "zero-workspace"
    ratios = [
        later / earlier if earlier else None for earlier, later in zip(values, values[1:])
    ]
    result["adjacent_ratios"] = ratios
    if ratio is not None:
        if any(value is not None and value > ratio + 1.0e-12 for value in ratios):
            raise ValueError(f"Adjacent memory ratio exceeds {ratio}.")
        result["maximum_adjacent_ratio"] = ratio
    return result


def _baseline_rows(
    baseline: dict[str, Any], family: str, case: Case
) -> tuple[list[dict[str, Any]], str]:
    # Baselines lack candidate decisions and often lack added supplemental rows.
    name = case.baseline_name or case.name
    if family in _SIMPLE:
        path = (
            "rows"
            if family in ("csr", "csr_capacity", "polymer_lag", "polynomial_preparation")
            else (
                "records"
                if family
                in (
                    "limber",
                    "structured_transfer",
                    "covariance_estimators",
                    "reactor_diffusion",
                )
                else "cases"
            )
        )
        return _rows(baseline, path), "compiler.temporary_bytes"
    if family in ("sensitivity_optics", "operator_workspace"):
        return _rows(baseline, f"cases.{name}"), case.metric
    if family == "ocean_shell":
        return _rows(baseline, f"records.{name}"), case.metric
    if family == "host_exact":
        return _rows(baseline, name), case.metric
    if family == "geometry_brep":
        return _rows(baseline, "cases"), f"{name}.compiler.temporary_bytes"
    if family == "periodic_ewald":
        return [
            row
            for row in _rows(baseline, "cases")
            if row["shell"] == case.rows[0]["shell"]
        ], case.metric
    if family == "astrodynamics_pairs":
        return [
            row
            for row in _rows(baseline, "rows")
            if row["route"] == name and row["steps"] == 0
        ], "compiler.temporary_bytes"
    if family in (
        "ml_neighbors",
        "contact_london_ptv",
        "optim_curvature",
        "optim_curvature_factor_batches",
    ):
        path, key = (
            ("records", "name")
            if family == "ml_neighbors"
            else ("cases", "area" if family == "contact_london_ptv" else "name")
        )
        return [
            row for row in _rows(baseline, path) if row[key] == name
        ], "temporary_bytes" if family == "ml_neighbors" else "compiler.temporary_bytes"
    if family == "gp_actions":
        return _rows(baseline, "records"), "compiler.temporary_bytes"
    if family == "spectral_split_form":
        return [
            row
            for row in _rows(baseline, "cases")
            if row["chunk"] == case.rows[0]["chunk"]
        ], "compiler.temporary_bytes"
    if family == "integration_lowering":
        return _rows(baseline, "cases"), case.metric
    raise ValueError(f"No baseline adapter for {family}/{name}.")


def _case_summary(
    case: Case, baseline: dict[str, Any] | None, family: str, comparable: bool
) -> dict[str, Any]:
    capacities, values = _series(case.rows, case.capacity, case.metric)
    metrics = {
        case.metric: _growth(
            capacities, values, case.maximum_exponent, case.maximum_ratio
        )
    }
    for metric, maximum in case.metrics.items():
        _, observed = _series(case.rows, case.capacity, metric)
        ratio = (
            {"ringdown": 2.25, "geometry_interface": 1.25}.get(family)
            if metric == "compiler.temporary_bytes"
            else None
        )
        metrics[metric] = _growth(capacities, observed, maximum, ratio)
    result: dict[str, Any] = {
        "case": case.name,
        "capacities": capacities,
        "metrics": metrics,
        "disposition": case.disposition,
    }
    if case.acceptance is not None:
        result["scaling_acceptance"] = case.acceptance
    if case.gate is None:
        old_rows = (
            _unconditional_baseline(baseline, family, case)
            if comparable and baseline is not None
            else None
        )
        if old_rows is not None:
            old_capacities, old = _series(old_rows, case.capacity, case.metric)
            if old_capacities != capacities:
                raise ValueError(f"Baseline capacities do not match {case.name}.")
            _match_controls(case.rows, old_rows)
            result["baseline_evidence"] = _growth(capacities, old, None, None)
            result["candidate_to_baseline_ratios"] = [
                new / previous if previous else None for previous, new in zip(old, values)
            ]
            result["comparison_scope"] = "matching-environment-baseline"
        return result
    gate = case.gate
    if case.paired is not None:
        old_rows, old_metric = (
            case.paired,
            case.paired_metric or "compiler.temporary_bytes",
        )
        result["comparison_scope"] = "same-record-paired"
    elif baseline is None:
        raise ValueError(
            f"A declared gate requires paired or baseline evidence for {case.name}."
        )
    elif comparable and baseline is not None:
        old_rows, old_metric = _baseline_rows(baseline, family, case)
        result["comparison_scope"] = "matching-environment-baseline"
    else:
        result["disposition"] = "not-compared"
        result["declared_decision"] = gate["decision"]
        return result
    old_capacities, old = _series(old_rows, case.capacity, old_metric)
    if old_capacities != capacities:
        raise ValueError(f"Baseline capacities do not match {case.name}.")
    _match_controls(case.rows, old_rows)
    metric = (
        case.metric
        if case.metric.endswith("temporary_bytes")
        or case.metric == "host_allocation_peak_bytes"
        else "compiler.temporary_bytes"
    )
    _, candidate = _series(case.rows, case.capacity, metric)
    if family == "periodic_kubo" and case.acceptance is not None:
        recomputed = evaluate_memory_gate(
            capacities,
            old,
            candidate,
            MemoryGatePolicy(gate.get("maximum_candidate_fraction", 0.5)),
        ).to_dict()
        decision = (
            recomputed["decision"] if case.acceptance["accepted"] else "retain-baseline"
        )
        if decision != gate["decision"]:
            raise ValueError(
                "Declared Kubo decision disagrees with observed fraction and scaling."
            )
        case.acceptance["decision"] = decision
    else:
        recomputed = compare_gate(
            capacities,
            old,
            candidate,
            gate["decision"],
            gate.get("maximum_candidate_fraction", 0.5),
        )
    for key in ("capacity", "baseline_bytes", "candidate_bytes"):
        if key in gate and gate[key] != recomputed[key]:
            raise ValueError(
                f"Declared {key} disagrees with observed series for {case.name}."
            )
    result["gate"] = recomputed
    result["disposition"] = (
        case.acceptance["decision"]
        if case.acceptance is not None
        else recomputed["decision"]
    )
    return result


def _load(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field {key!r} in {path.name}.")
            result[key] = value
        return result

    def refuse_constant(value: str) -> NoReturn:
        raise ValueError(f"Nonfinite JSON value {value} in {path.name}.")

    record = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=refuse_constant,
    )
    if not isinstance(record, dict):
        raise ValueError(f"{path.name} must contain an evidence object.")
    _finite_json(record)
    return record


def _environment(record: dict[str, Any]) -> str:
    fingerprint = _value(record, "environment.fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("An environment fingerprint is required.")
    return fingerprint


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    identity = record.get("identity", record.get("benchmark_identity"))
    if identity is None:
        return {}
    if not isinstance(identity, dict) or not isinstance(
        identity.get("benchmark_driver_fingerprint"), str
    ):
        raise ValueError("Malformed benchmark driver identity.")
    return identity


def _evidence_identity(identity: dict[str, Any]) -> Any:
    return identity.get("evidence_schema", identity.get("evidence_schema_signature"))


def build_summary(directory: Path, *, output: Path | None = None) -> dict[str, Any]:
    """Return deterministic dispositions for every candidate memory artifact."""
    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("memory_*.json")):
        if (
            path.name == "memory_scaling.json"
            or path.stem.endswith("_baseline")
            or path.name in _NAMED_BASELINES
            or (output is not None and path.resolve() == output.resolve())
        ):
            continue
        family = path.stem.removeprefix("memory_")
        record = _load(path)
        if family == "field_view_loop":
            # The isolated single-capacity record predates environment capture.
            # It cannot honestly supply either scaling or portable comparison.
            _integers(
                [_value(record, "size"), _value(record, "compiler.temporary_bytes")],
                "Isolated loop evidence",
            )
            summaries.append(
                {
                    "record": path.name,
                    "disposition": "insufficient-evidence",
                    "reason": "Single capacity and no environment fingerprint; not compared.",
                    "cases": [
                        {
                            "case": "isolated-loop",
                            "capacities": [record["size"]],
                            "compiler_temporary_bytes": [
                                record["compiler"]["temporary_bytes"]
                            ],
                            "disposition": "observed-only",
                        }
                    ],
                }
            )
            continue
        cases = _adapt(record, family) + _supplemental(record, family)
        for case in cases:
            _observed_metrics(case)
        if not cases:
            raise ValueError(f"No supported cases in {path.name}.")
        environment = _environment(record)
        identity = _identity(record)
        if not identity and family != "ocean_shell":
            raise ValueError(f"Missing benchmark driver identity in {path.name}.")
        baseline_name = {
            "mac_shadowing_large": "memory_mac_shadowing_baseline_realistic.json",
            "optim_curvature_factor_batches": "memory_optim_curvature_baseline.json",
        }.get(family, path.stem + "_baseline.json")
        baseline_path = path.with_name(baseline_name)
        baseline = _load(baseline_path) if baseline_path.exists() else None
        comparable = baseline is not None and _environment(baseline) == environment
        uses_historical_baseline = baseline is not None and any(
            case.gate is not None
            and case.paired is None
            or case.gate is None
            and _unconditional_baseline(baseline, family, case) is not None
            for case in cases
        )
        summary: dict[str, Any] = {
            "record": path.name,
            "environment_fingerprint": environment,
            "disposition": "compared"
            if comparable
            else ("environment-mismatch" if baseline is not None else "candidate-only"),
            "driver_family": family,
            "benchmark_driver_fingerprint": identity.get("benchmark_driver_fingerprint"),
        }
        if baseline is not None:
            baseline_identity = _identity(baseline)
            summary.update(
                {
                    "baseline": baseline_path.name,
                    "baseline_environment_fingerprint": _environment(baseline),
                    "baseline_driver_fingerprint": baseline_identity.get(
                        "benchmark_driver_fingerprint"
                    ),
                }
            )
            if not uses_historical_baseline:
                summary["historical_baseline_scope"] = "informational-only"
            if comparable and uses_historical_baseline and not baseline_identity:
                comparable = False
                summary["disposition"] = "baseline-identity-missing"
                summary["reason"] = (
                    "Baseline has no driver/evidence identity; not compared."
                )
            elif (
                comparable
                and uses_historical_baseline
                and _evidence_identity(identity) != _evidence_identity(baseline_identity)
            ):
                extensions = {
                    "fem_assembly": {"compiler_route", "requested_tensor_families"},
                    "polynomial_preparation": {
                        "compiler_scaling_exponent",
                        "declared_scaling_exponent",
                        "gate",
                    },
                    "linalg_solves": {"auxiliary"},
                }
                current_schema = _evidence_identity(identity)
                old_schema = _evidence_identity(baseline_identity)
                current_fields = (
                    current_schema.get("fields")
                    if isinstance(current_schema, dict)
                    else identity.get("evidence_fields")
                )
                old_fields = (
                    old_schema.get("fields")
                    if isinstance(old_schema, dict)
                    else baseline_identity.get("evidence_fields")
                )
                extra = extensions.get(family)
                if (
                    extra is None
                    or not isinstance(current_fields, list)
                    or not isinstance(old_fields, list)
                    or set(current_fields) != set(old_fields) | extra
                ):
                    raise ValueError(
                        f"Baseline evidence identity mismatch in {path.name}."
                    )
                summary["evidence_extension_fields"] = sorted(extra)
            if comparable and uses_historical_baseline:
                _match_configuration(record, baseline, family)
        summary["cases"] = [
            _summarize_case(case, baseline, family, comparable)
            for case in sorted(cases, key=lambda case: case.name)
        ]
        if family == "geometry_interface" and "decision" in record:
            main_case = next(case for case in summary["cases"] if case["case"] == "main")
            if (
                main_case["disposition"] != "not-compared"
                and record["decision"] != main_case["disposition"]
            ):
                raise ValueError(
                    "Declared directed-interface decision disagrees with observed gate and scaling."
                )
        summary["baseline_comparisons"] = sum(
            case.get("comparison_scope") == "matching-environment-baseline"
            for case in summary["cases"]
        )
        summary["paired_comparisons"] = sum(
            case.get("comparison_scope") == "same-record-paired"
            for case in summary["cases"]
        )
        if comparable and not summary["baseline_comparisons"]:
            summary["disposition"] = (
                "paired-evidence" if summary["paired_comparisons"] else "candidate-only"
            )
        if family == "host_caches":
            summary["hot_sets"] = [
                {
                    "module": row["module"],
                    "symbol": row["symbol"],
                    "retained_to_budget_ratio": row["retained_bytes"]
                    / row["configured_cap_bytes"],
                    "entries_to_budget_ratio": row["cache_entries"]
                    / row["configured_max_entries"],
                    "disposition": "within-declared-budget",
                }
                for row in sorted(
                    _rows(record, "hot_sets"),
                    key=lambda row: (row["module"], row["symbol"]),
                )
            ]
            summary["oversized_value"] = {
                "disposition": "observed-only",
                "reason": "Record has a not_retained boolean, not before/after byte and miss counters; it cannot be independently recomputed.",
            }
        summaries.append(summary)
    if not summaries:
        raise ValueError("No candidate memory records found.")
    return {
        "records": summaries,
        "disposition_counts": dict(
            sorted(Counter(row["disposition"] for row in summaries).items())
        ),
        "policy": {
            "comparison": "Equal environment fingerprint, explicit driver family and matching case/evidence identity",
            "gate": "benchmarks._memory.evaluate_memory_gate",
            "growth": "benchmarks._memory.fitted_scaling_exponent",
            "timing_and_rss": "observed evidence only; never hard gates",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the committed summary without rewriting it.",
    )
    options = parser.parse_args()
    destination = options.output or options.directory / "memory_scaling.json"
    summary = build_summary(options.directory, output=destination)
    if options.check:
        if not destination.exists() or _load(destination) != summary:
            raise ValueError(f"Memory summary is stale: {destination}.")
        print(f"Memory summary matches committed records: {destination}")
    else:
        write_json_atomic(destination, summary)
        print(destination)


__all__ = ["build_summary", "compare_gate", "compare_growth"]


if __name__ == "__main__":
    main()
