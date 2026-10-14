#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure-compression algorithms returning source indices and positive weights."""

from typing import TYPE_CHECKING

from .._deferred_exports import deferred_exports


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "kernel_herd": "._herding",
    "KernelHerding": "._herding",
    "weighted_mmd": "._herding",
    "randomized_pivoted_cholesky": "._pivoted_cholesky",
    "RandomizedPivotedCholesky": "._pivoted_cholesky",
    "moment_recombine": "._recombination",
    "MomentRecombination": "._recombination",
    "CoresetSelection": "._types",
    "KernelHerdingDiagnostics": "._types",
    "MomentRecombinationDiagnostics": "._types",
    "PivotedCholeskyDiagnostics": "._types",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._herding import kernel_herd, KernelHerding, weighted_mmd
    from ._pivoted_cholesky import (
        randomized_pivoted_cholesky,
        RandomizedPivotedCholesky,
    )
    from ._recombination import moment_recombine, MomentRecombination
    from ._types import (
        CoresetSelection,
        KernelHerdingDiagnostics,
        MomentRecombinationDiagnostics,
        PivotedCholeskyDiagnostics,
    )


__all__ = [
    "CoresetSelection",
    "KernelHerding",
    "KernelHerdingDiagnostics",
    "RandomizedPivotedCholesky",
    "MomentRecombination",
    "MomentRecombinationDiagnostics",
    "PivotedCholeskyDiagnostics",
    "randomized_pivoted_cholesky",
    "kernel_herd",
    "moment_recombine",
    "weighted_mmd",
]
