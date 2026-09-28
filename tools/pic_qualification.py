#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from phydrax.solver._pic_qualification import run_pic_qualification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_pic_qualification(smoke=args.smoke)
    payload = json.dumps(asdict(report), indent=2, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
