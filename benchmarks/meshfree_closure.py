# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Phase-separated meshfree closure workloads (P16).

Physical, distributed, adjoint, adaptive and restart workloads share the
evidence utilities of ``benchmarks.meshfree_scaling``: one phase vocabulary,
one recorder (wall, process CPU, XLA compile count, sampled memory, compiler
bytes), declared capacity refusal, and one record writer. Each workload calls
the public package API and reports its measured scientific evidence beside
its timing; a declared refusal is resource evidence, never an accepted row.
Workload groups live in ``benchmarks/meshfree_closure_<group>.py``.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from benchmarks._io import write_json_atomic
from benchmarks.meshfree_closure_adaptive_restart import ADAPTIVE_RESTART_WORKLOADS
from benchmarks.meshfree_closure_distributed import DISTRIBUTED_WORKLOADS
from benchmarks.meshfree_closure_flow_mechanics import FLOW_MECHANICS_WORKLOADS
from benchmarks.meshfree_closure_forms_sensitivity import FORMS_SENSITIVITY_WORKLOADS
from benchmarks.meshfree_closure_transport_transfer import (
    TRANSPORT_TRANSFER_WORKLOADS,
)
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    admitted_rows,
    apply_baseline,
    config_from_arguments,
    configure_precision,
    make_record,
    MeshfreeConfig,
)


type Workload = Callable[[int, int, MeshfreeConfig], dict[str, Any]]


def _registry(*groups: Mapping[str, Workload]) -> dict[str, Workload]:
    """Merge workload groups; one name never maps to two workloads."""
    registry: dict[str, Workload] = {}
    for group in groups:
        duplicated = sorted(set(registry) & set(group))
        if duplicated:
            raise ValueError("Duplicate closure workloads: " + ", ".join(duplicated))
        registry.update(group)
    return registry


WORKLOADS = _registry(
    TRANSPORT_TRANSFER_WORKLOADS,  # Q9 bulk transport, Q13 transfer/topology
    FLOW_MECHANICS_WORKLOADS,  # Q10 flow, Q11 mechanics/surface Stokes/FSI
    FORMS_SENSITIVITY_WORKLOADS,  # Q12 higher forms, Q14 sensitivities
    ADAPTIVE_RESTART_WORKLOADS,  # Q15 adaptive/learned/hybrid, Q17 restart
    DISTRIBUTED_WORKLOADS,  # Q16 distributed
)


def run(config: MeshfreeConfig, workloads: tuple[str, ...], /) -> dict[str, Any]:
    """Run the selected workloads at every requested capacity and seed."""
    unknown = sorted(set(workloads) - set(WORKLOADS))
    if not workloads or unknown or len(set(workloads)) != len(workloads):
        raise ValueError(
            "Select distinct registered workloads; unknown: " + ", ".join(unknown)
        )
    configure_precision(config, supported=("float64", "float32"))
    rows = [
        {"workload": name, **row}
        for name in workloads
        for row in admitted_rows(WORKLOADS[name], config)
    ]
    record = make_record(
        config,
        rows,
        Path(__file__),
        consumers=tuple(
            dict.fromkeys(Path(inspect.getfile(WORKLOADS[name])) for name in workloads)
        ),
    )
    record["workloads"] = list(workloads)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--workloads", nargs="+", required=True)
    args = parser.parse_args()
    record = run(config_from_arguments(args), tuple(args.workloads))
    apply_baseline(record, args.baseline)
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
