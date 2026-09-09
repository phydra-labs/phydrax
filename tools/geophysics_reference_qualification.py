#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compare predictions with a checksum-pinned external or field reference NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from phydrax.qualification import GeophysicalReferenceRecipe


_MAXIMUM_RECIPE_BYTES = 1_000_000


def _recipe(path: Path) -> GeophysicalReferenceRecipe:
    if path.stat().st_size > _MAXIMUM_RECIPE_BYTES:
        raise ValueError("Geophysical reference recipe exceeds the 1 MB metadata limit.")
    record = json.loads(path.read_text(encoding="utf-8"))
    return GeophysicalReferenceRecipe.from_record(record)


def _validate_npz_directory(path: Path, maximum_samples: int, /) -> None:
    allowed = {
        "prediction.npy",
        "reference.npy",
        "valid.npy",
        "standard_deviation.npy",
    }
    with ZipFile(path) as container:
        members = container.infolist()
    names = [member.filename for member in members]
    maximum_member_bytes = 4096 + 16 * maximum_samples
    if (
        not members
        or len(names) != len(set(names))
        or set(names) - allowed
        or any(member.file_size > maximum_member_bytes for member in members)
        or sum(member.file_size for member in members)
        > len(allowed) * maximum_member_bytes
    ):
        raise ValueError("Reference NPZ directory exceeds its declared array profile.")


def qualify(recipe_path: Path, artifact_path: Path) -> dict[str, object]:
    recipe = _recipe(recipe_path)
    if artifact_path.stat().st_size != recipe.artifact.size_bytes:
        raise ValueError("Reference artifact size does not match its governed manifest.")
    payload = artifact_path.read_bytes()
    recipe.artifact.verify_bytes(payload)
    _validate_npz_directory(artifact_path, recipe.maximum_samples)
    with np.load(artifact_path, allow_pickle=False) as archive:
        required = {"prediction", "reference"}
        missing = required - set(archive.files)
        unknown = set(archive.files) - {
            "prediction",
            "reference",
            "valid",
            "standard_deviation",
        }
        if missing or unknown:
            raise ValueError(
                "Reference NPZ needs prediction/reference and only optional "
                "valid/standard_deviation arrays."
            )
        comparison = recipe.compare(
            archive["prediction"],
            archive["reference"],
            valid=None if "valid" not in archive.files else archive["valid"],
            standard_deviation=(
                None
                if "standard_deviation" not in archive.files
                else archive["standard_deviation"]
            ),
        )
    return {
        "kind": "geophysical-reference-qualification",
        "recipe": recipe.to_record(),
        "comparison": comparison.to_record(),
        "passed": comparison.passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = qualify(arguments.recipe, arguments.artifact)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
