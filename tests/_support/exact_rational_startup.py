# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fresh-process admission of the exact source facade before other geometry owners."""


def source_facade_first() -> dict[str, bool]:
    from phydrax.discretization import ExactRationalCellGeometrySource

    return {
        "source_available": ExactRationalCellGeometrySource.__name__
        == "ExactRationalCellGeometrySource"
    }
