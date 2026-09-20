#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail closed when checked public exports drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.generate_public_api_manifest import public_api_record


def public_api_manifest_errors(
    root: Path,
    /,
    *,
    manifest: Path,
) -> tuple[str, ...]:
    path = root / manifest
    if not path.is_file():
        return (f"missing-public-api-manifest:{manifest.as_posix()}",)
    checked = json.loads(path.read_text(encoding="utf-8"))
    current = public_api_record()
    if checked != current:
        return ("public-api-manifest-drift",)
    return ()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/data/public_api.json"),
    )
    arguments = parser.parse_args()
    errors = public_api_manifest_errors(
        arguments.root,
        manifest=arguments.manifest,
    )
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
