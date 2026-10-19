#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Present durable layer/core field, volume, history, and PDE continuation.

Run ``python -m tools.layer_core_lifecycle_continuation ARCHIVE CONTENT_ID``
on a durable meshing-source closure whose ``accepted_data`` holds the
``(refinement, coarsening)`` adaptation results. Scientific preparation,
continuation, qualification, and resource evidence belong to
``phydrax.meshing.qualification``.
"""

from __future__ import annotations

import argparse
import json

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
from phydrax.meshing.qualification import (
    continue_layer_core_lifecycle,
    LAYER_CORE_REGION_DENSITIES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("content_id")
    parser.add_argument(
        "--region-densities", type=float, nargs="+", default=LAYER_CORE_REGION_DENSITIES
    )
    parser.add_argument("--material-site", default="two-material-density")
    args = parser.parse_args()
    report = continue_layer_core_lifecycle(
        args.archive,
        args.content_id,
        region_densities=tuple(args.region_densities),
        material_site=args.material_site,
    )
    for record in report.to_records():
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
