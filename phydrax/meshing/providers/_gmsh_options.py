#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh generation algorithms, thread policy, and optimization options."""

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class GmshSurfaceAlgorithm(StrEnum):
    MESH_ADAPT = "mesh_adapt"
    AUTOMATIC = "automatic"
    DELAUNAY = "delaunay"
    FRONTAL_DELAUNAY = "frontal_delaunay"
    BAMG = "bamg"
    FRONTAL_DELAUNAY_QUADS = "frontal_delaunay_quads"
    PACKING_PARALLELOGRAMS = "packing_parallelograms"
    QUASI_STRUCTURED_QUAD = "quasi_structured_quad"

    @property
    def gmsh_code(self) -> int:
        match self:
            case GmshSurfaceAlgorithm.MESH_ADAPT:
                return 1
            case GmshSurfaceAlgorithm.AUTOMATIC:
                return 2
            case GmshSurfaceAlgorithm.DELAUNAY:
                return 5
            case GmshSurfaceAlgorithm.FRONTAL_DELAUNAY:
                return 6
            case GmshSurfaceAlgorithm.BAMG:
                return 7
            case GmshSurfaceAlgorithm.FRONTAL_DELAUNAY_QUADS:
                return 8
            case GmshSurfaceAlgorithm.PACKING_PARALLELOGRAMS:
                return 9
            case GmshSurfaceAlgorithm.QUASI_STRUCTURED_QUAD:
                return 11
            case _:
                raise ValueError("Unsupported Gmsh surface algorithm.")


class GmshVolumeAlgorithm(StrEnum):
    DELAUNAY = "delaunay"
    FRONTAL = "frontal"
    HXT = "hxt"

    @property
    def gmsh_code(self) -> int:
        match self:
            case GmshVolumeAlgorithm.DELAUNAY:
                return 1
            case GmshVolumeAlgorithm.FRONTAL:
                return 4
            case GmshVolumeAlgorithm.HXT:
                return 10
            case _:
                raise ValueError("Unsupported Gmsh volume algorithm.")


class GmshHighOrderOptimization(StrEnum):
    """Gmsh curving strategy applied after high-order node placement."""

    NONE = "none"
    OPTIMIZATION = "optimization"
    ELASTIC_OPTIMIZATION = "elastic_optimization"
    ELASTIC = "elastic"
    FAST_CURVING = "fast_curving"

    @property
    def gmsh_code(self) -> int:
        match self:
            case GmshHighOrderOptimization.NONE:
                return 0
            case GmshHighOrderOptimization.OPTIMIZATION:
                return 1
            case GmshHighOrderOptimization.ELASTIC_OPTIMIZATION:
                return 2
            case GmshHighOrderOptimization.ELASTIC:
                return 3
            case GmshHighOrderOptimization.FAST_CURVING:
                return 4
            case _:
                raise ValueError("Unsupported Gmsh high-order optimization.")


class GmshOptions(StrictModule, NonTrainableState):
    """Provider execution options; scientific requests live in specifications.

    `num_threads` sets Gmsh's `General.NumThreads`; sessions refuse plans whose
    thread count exceeds their `MeshingExecutionPolicy.parallelism`, and
    multi-threaded generation is reported as nondeterministic.
    """

    algorithm_2d: GmshSurfaceAlgorithm = eqx.field(static=True)
    algorithm_3d: GmshVolumeAlgorithm = eqx.field(static=True)
    num_threads: int = eqx.field(static=True)
    optimize_netgen: bool = eqx.field(static=True)
    high_order_optimization: GmshHighOrderOptimization = eqx.field(static=True)
    terminal_output: bool = eqx.field(static=True)
    association_tolerance_factor: float = eqx.field(static=True)
    options_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        algorithm_2d: GmshSurfaceAlgorithm = GmshSurfaceAlgorithm.FRONTAL_DELAUNAY,
        algorithm_3d: GmshVolumeAlgorithm = GmshVolumeAlgorithm.DELAUNAY,
        num_threads: int = 1,
        optimize_netgen: bool = False,
        high_order_optimization: GmshHighOrderOptimization = (
            GmshHighOrderOptimization.NONE
        ),
        terminal_output: bool = False,
        association_tolerance_factor: float = 4.0,
    ) -> None:
        if not isinstance(algorithm_2d, GmshSurfaceAlgorithm):
            raise TypeError("algorithm_2d must be GmshSurfaceAlgorithm.")
        if not isinstance(algorithm_3d, GmshVolumeAlgorithm):
            raise TypeError("algorithm_3d must be GmshVolumeAlgorithm.")
        if not isinstance(high_order_optimization, GmshHighOrderOptimization):
            raise TypeError("high_order_optimization must be GmshHighOrderOptimization.")
        threads = int(num_threads)
        factor = float(association_tolerance_factor)
        if threads <= 0:
            raise ValueError("num_threads must be positive.")
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("association_tolerance_factor must be positive and finite.")
        self.algorithm_2d = algorithm_2d
        self.algorithm_3d = algorithm_3d
        self.num_threads = threads
        self.optimize_netgen = bool(optimize_netgen)
        self.high_order_optimization = high_order_optimization
        self.terminal_output = bool(terminal_output)
        self.association_tolerance_factor = factor
        self.options_id = canonical_fingerprint(
            {
                "kind": "gmsh-options",
                "algorithm_2d": algorithm_2d.value,
                "algorithm_3d": algorithm_3d.value,
                "num_threads": threads,
                "optimize_netgen": bool(optimize_netgen),
                "high_order_optimization": high_order_optimization.value,
                "terminal_output": bool(terminal_output),
                "association_tolerance_factor": factor,
            }
        )


__all__ = [
    "GmshHighOrderOptimization",
    "GmshOptions",
    "GmshSurfaceAlgorithm",
    "GmshVolumeAlgorithm",
]
