#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


ScalarTraceSide3D = Literal["interior", "exterior"]


class UnsupportedScalarBoundarySpaceError(ValueError):
    """Requested scalar boundary map is not admissible in the prepared space."""


class ScalarTraceConvention3D(StrictModule, NonTrainableState):
    """Outward-normal trace and jump convention for a closed surface in 3D.

    ``interior`` is the bounded side and ``exterior`` is the unbounded side.
    The normal points from the interior to the exterior, ``gamma0`` is the
    Dirichlet trace, and ``gamma1`` is differentiation along that same normal
    on both sides.  With ``D`` using the outward *source* normal,

    ``gamma0^- D = K - I/2``, ``gamma0^+ D = K + I/2``,
    ``gamma1^- S = K' + I/2``, and ``gamma1^+ S = K' - I/2``.

    The hypersingular convention is ``W = -gamma1 D``.  This class describes
    signs only; it makes no regularity or continuum-accuracy claim.
    """

    ambient_dimension: int = eqx.field(static=True)
    boundary_dimension: int = eqx.field(static=True)
    interior: str = eqx.field(static=True)
    exterior: str = eqx.field(static=True)
    normal_orientation: str = eqx.field(static=True)
    dirichlet_trace: str = eqx.field(static=True)
    neumann_trace: str = eqx.field(static=True)
    hypersingular_definition: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self):
        self.ambient_dimension = 3
        self.boundary_dimension = 2
        self.interior = "bounded-side"
        self.exterior = "unbounded-side"
        self.normal_orientation = "interior-to-exterior"
        self.dirichlet_trace = "gamma0:u"
        self.neumann_trace = "gamma1:outward-normal-derivative"
        self.hypersingular_definition = "W=-gamma1(D)"
        self.convention_id = canonical_fingerprint(
            {
                "kind": "closed-scalar-trace-convention-3d-v1",
                "normal": self.normal_orientation,
                "gamma0_D": {"interior": -0.5, "exterior": 0.5},
                "gamma1_S": {"interior": 0.5, "exterior": -0.5},
                "W": self.hypersingular_definition,
            }
        )

    @staticmethod
    def _side(side: ScalarTraceSide3D, /) -> ScalarTraceSide3D:
        if side not in ("interior", "exterior"):
            raise ValueError("Scalar trace side must be 'interior' or 'exterior'.")
        return side

    def double_layer_dirichlet_jump(self, side: ScalarTraceSide3D, /) -> float:
        """Return the identity coefficient in ``gamma0 D`` on ``side``."""
        return -0.5 if self._side(side) == "interior" else 0.5

    def single_layer_neumann_jump(self, side: ScalarTraceSide3D, /) -> float:
        """Return the identity coefficient in ``gamma1 S`` on ``side``."""
        return 0.5 if self._side(side) == "interior" else -0.5


SCALAR_TRACE_CONVENTION_3D = ScalarTraceConvention3D()


__all__ = [
    "SCALAR_TRACE_CONVENTION_3D",
    "ScalarTraceConvention3D",
    "ScalarTraceSide3D",
    "UnsupportedScalarBoundarySpaceError",
]
