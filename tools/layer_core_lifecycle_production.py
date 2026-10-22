#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Produce the original native layer/core accepted-state closure for a cold consumer.

Run ``python -m tools.layer_core_lifecycle_production DESTINATION`` in its own
process. It natively generates the original periodic two-material layer/core
source, authors and solves the initial accepted state, and for each of
``--rounds`` refines one physical column, continues the accepted state warm,
durably reopens that accepted history, applies the inverse coarsening and
continues the accepted state warm again, then writes ``DESTINATION/lifecycle``
with every accepted epoch. The single JSON line is the receipt whose
``archive``, ``content_id`` and ``accepted_epoch_ids`` the consumer
(``tools.layer_core_lifecycle_continuation`` or the ``archived-layer-core``
hybrid profile) re-authenticates against every manifest field and bank.

``--capacity`` bounds the native request without changing the original source.
The accepted material values and site are set by ``--hybrid-region-densities``
and ``--hybrid-material-site``. Resolution and core size are fixed-source
admission controls: unsupported variations refuse rather than producing a
different source or falsely claiming a resolution-scaled mesh.
"""

from __future__ import annotations

import argparse
import json
import os
from time import monotonic, perf_counter

from phydrax import config as pxconfig
from phydrax._validation import positive_real_scalar


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
from phydrax.meshing.qualification._layer_lineage import (
    produce_layer_core_lineage_archive,
)
from tools.meshing_qualification._controls import capacity_meshing_limits
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    Deadline,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--solver-repeats", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=20_000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument(
        "--hybrid-region-densities", type=float, nargs="+", default=(2.0, 3.0)
    )
    parser.add_argument("--hybrid-material-site", default="two-material-density")
    parser.add_argument("--hybrid-core-size", type=float, default=0.1)
    args = parser.parse_args()
    # This producer is the immutable original ridge source, not a grid family.
    # Generic campaign knobs must not silently relabel it as a different source.
    if args.resolution != 4 or args.hybrid_core_size != 0.1:
        raise ValueError(
            "layer_core_production.source_controls must retain the original "
            "resolution=4 and hybrid_core_size=0.1; got "
            f"{(args.resolution, args.hybrid_core_size)!r}"
        )
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        edges=16,
        faces=16,
        cells=8,
        connectivity_entries=128,
        data_bytes=4096,
        work_units=128,
        geometry_queries=128,
        scratch_bytes=8192,
    )
    started = perf_counter()
    deadline = Deadline(started + positive_real_scalar(args.timeout, "timeout"))
    deadline.remaining()
    receipt = produce_layer_core_lineage_archive(
        args.destination,
        rounds=args.rounds,
        solver_repeats=args.solver_repeats,
        limits=limits,
        region_densities=tuple(args.hybrid_region_densities),
        material_site=args.hybrid_material_site,
    )
    science_returned_at = monotonic()
    deadline.remaining()
    print(
        json.dumps(
            {
                **receipt.to_record(),
                "producer": {
                    "controls": {
                        "capacity": args.capacity,
                        "resolution": args.resolution,
                        "rounds": args.rounds,
                        "solver_repeats": args.solver_repeats,
                        "hybrid_region_densities": list(args.hybrid_region_densities),
                        "hybrid_material_site": args.hybrid_material_site,
                        "hybrid_core_size": args.hybrid_core_size,
                    },
                    "control_semantics": {
                        "capacity": "native-request-resource-limits",
                        "resolution": "fixed-original-source-admission-only",
                        "hybrid_core_size": "fixed-original-source-admission-only",
                        "hybrid_region_densities": "actual-initial-accepted-FV-density",
                        "hybrid_material_site": "actual-accepted-material-site",
                    },
                    "process_id": os.getpid(),
                    "runtime": _runtime_identity(),
                    "execution_seconds": perf_counter() - started,
                    "science_returned_before_deadline": True,
                    "science_returned_monotonic_seconds": science_returned_at,
                    "peak_process_resident_bytes": peak_resident_bytes(),
                    "process_peak_scope": PEAK_RESIDENT_SCOPE,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
