# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from time import perf_counter


_IMPORT_STARTED = perf_counter()
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import jax

from phydrax.meshing import BisectionCompatibility
from tools._meshing_cases import CASE_NAMES, POLYHEDRAL_CASE_NAMES
from tools.meshing_qualification._controls import (
    FAMILY_CASES,
    HYBRID_PROFILES,
    QualificationControls,
    SCENARIO_MODULES,
    SURFACE_CASES,
)
from tools.meshing_qualification._records import _finalize_mandatory_cli_record


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SCENARIO_MODULES), default="manifold")
    parser.add_argument("--executable", default="vc_mesh")
    parser.add_argument("--worker", default=None)
    parser.add_argument(
        "--corpus-case", choices=CASE_NAMES, default="planar-hole-feature"
    )
    parser.add_argument(
        "--corpus-profile",
        nargs="+",
        default=None,
        help="Execute only these original profiles; retain every unrun mandatory obligation.",
    )
    parser.add_argument(
        "--polyhedral-source",
        choices=POLYHEDRAL_CASE_NAMES,
        default="polyhedral-material-L",
    )
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=20000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Whole-workflow deadline in seconds; stages and native calls share it.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-error", type=float, default=0.05)
    parser.add_argument("--adaptation-rounds", type=int, default=1)
    parser.add_argument(
        "--cad-curving-fidelity",
        type=float,
        help="Required two-sided source-deviation request for the cad-curving scenario.",
    )
    parser.add_argument("--surface-case", choices=SURFACE_CASES, default="sphere")
    parser.add_argument("--cad-format", choices=("step", "iges", "brep"), default="step")
    parser.add_argument(
        "--periodic-case",
        choices=("skew-lattice", "skew-material-feature"),
        default="skew-lattice",
    )
    parser.add_argument("--maximum-memory-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--maximum-condition", type=float, default=1.0e12)
    parser.add_argument("--overset-motion-steps", type=int, default=2)
    parser.add_argument("--overset-wall-pairs", type=int, default=500000)
    parser.add_argument("--overset-donor-pairs", type=int, default=500000)
    parser.add_argument("--overset-overlap-pairs", type=int, default=500000)
    parser.add_argument("--family-case", choices=FAMILY_CASES, default="dual-hex")
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument("--hybrid-profile", choices=HYBRID_PROFILES, default="unit-box")
    parser.add_argument("--hybrid-archive")
    parser.add_argument("--hybrid-content-id")
    parser.add_argument(
        "--hybrid-region-densities", type=float, nargs="+", default=(2.0, 3.0)
    )
    parser.add_argument("--hybrid-material-site", default="two-material-density")
    parser.add_argument("--hybrid-core-size", type=float, default=0.1)
    parser.add_argument("--first-thickness", type=float, default=0.1)
    parser.add_argument("--layer-growth", type=float, default=1.0)
    parser.add_argument("--geometry-order", type=int, choices=(2, 3, 4, 6, 10), default=2)
    parser.add_argument("--implicit-radius", type=float, default=0.371)
    parser.add_argument("--implicit-fidelity", type=float, default=0.02)
    parser.add_argument("--implicit-maximum-level", type=int, default=9)
    parser.add_argument("--implicit-edge-isolation-depth", type=int, default=8)
    parser.add_argument("--implicit-root-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--implicit-refinement-rounds", type=int, default=8)
    parser.add_argument("--implicit-improvement-passes", type=int, default=8)
    parser.add_argument("--implicit-metric-passes", type=int, default=16)
    parser.add_argument("--distributed-parts", type=int, default=2)
    parser.add_argument("--restart-parts", type=int, default=1)
    parser.add_argument("--distributed-processes", type=int, default=2)
    parser.add_argument(
        "--distributed-source",
        choices=("native-unit-square",),
        default="native-unit-square",
    )
    parser.add_argument("--distributed-halo-width", type=int, default=2)
    parser.add_argument("--distributed-halo-capacity", type=int, default=20000)
    parser.add_argument("--distributed-cavity-capacity", type=int, default=20000)
    parser.add_argument("--distributed-slack", type=int, default=2)
    parser.add_argument(
        "--distributed-host-packet-bytes", type=int, default=64 * 1024 * 1024
    )
    parser.add_argument("--distributed-checkpoint-root", default=None)
    parser.add_argument(
        "--distributed-bisection-compatibility",
        type=BisectionCompatibility,
        choices=tuple(BisectionCompatibility),
        default=BisectionCompatibility.REJECT,
        help="Initial matching-label preparation, charged to the unchanged request limits.",
    )
    parser.add_argument("--image-size", type=int, default=2)
    parser.add_argument("--transport-steps", type=int, default=5)
    parser.add_argument("--transport-dt", type=float, default=0.01)
    parser.add_argument("--balance-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--oracle-capacity", type=int, default=2048)
    args = QualificationControls.from_namespace(parser.parse_args())
    scenario_module = importlib.import_module(
        "tools.meshing_qualification." + SCENARIO_MODULES[args.scenario]
    )
    _IMPORT_SECONDS = perf_counter() - _IMPORT_STARTED
    record = _finalize_mandatory_cli_record(args, scenario_module.run(args))
    record["import_seconds"] = _IMPORT_SECONDS
    record["peak_resident_unit"] = "bytes"
    record_directory = os.environ.get("PHYDRAX_MESHING_RECORD_DIRECTORY")
    if args.scenario == "native-distributed-lifecycle" and record_directory is not None:
        process_id = jax.process_index()
        jax.distributed.shutdown()
        Path(record_directory, f"process-{process_id}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    else:
        print(json.dumps(record, indent=2))
    if record.get("status") in ("failed", "timeout"):
        raise SystemExit(1)
    if args.scenario == "native-corpus":
        counts = record["counts"]
        if (
            counts.get("failed", 0)
            or counts.get("timeout", 0)
            or record.get("unqualified_selected_profiles")
            or record.get("mandatory_workflows_incomplete")
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
