#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import warnings
from pathlib import Path


def _maybe_convert_cad_to_stl(mesh_path: Path) -> tuple[Path, Path | None]:
    name = mesh_path.name.lower()
    if not name.endswith((".stp", ".step", ".brep")):
        return mesh_path, None

    import build123d as bd

    if name.endswith((".step", ".brep")):
        name = name[:-4]
    else:
        name = name[:-3]
    assert name.endswith(".")

    stl_path = Path(f"/tmp/{name}stl").resolve()
    result = bd.import_step(mesh_path)
    if not result.is_leaf:
        assert result.children
        warnings.warn(
            "You are initializing a `Geometry3DFromCAD` object with a CAD file "
            "that contains a compound geometry. This will be represented as a "
            "single unified geometry, rather than separate components."
        )
    success = bd.export_stl(result, stl_path)
    assert success
    assert stl_path.is_file()
    return stl_path, stl_path


__all__ = ["_maybe_convert_cad_to_stl"]
