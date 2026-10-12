#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, final, Literal

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse


type MPMGeometryKind = Literal["cartesian", "axisymmetric"]


@final
class MPMGeometryPlan(StrictModule, NonTrainableState):
    """Spatial and constitutive axes for material-point geometry.

    Axisymmetric positions, velocities and APIC tensors use (r, z); material
    tensors use (r, z, theta). Mass and reference volume represent complete
    physical rings, not meridional densities. There are no hidden 2*pi factors.

    Independently deriving cylindrical virtual power gives
    V0*(tau_ab*L_ab + tau_tt*v_r/r), since current volume V=J*V0 and
    Cauchy stress sigma=tau/J. Thus the radial hoop force is
    -V0*tau_tt*N/r, with tau=P*F.T. Forward Euler gives
    delta F_tt=1+dt*v_r/r. The method's origin is Sulsky & Schreyer
    (1996), CMAME 139:409--429, doi:10.1016/S0045-7825(96)01091-2;
    that citation is not a claim of equation-level manuscript verification.

    B-bar averages the full divergence L_rr+L_zz+L_tt over the background
    cell with r dr dz, and adds (average_divergence-tr(L))*I/3. Its force
    correction is the exact transpose, including the averaged hoop basis.
    Reflected ghost nodes have even scalar/axial and odd radial parity.
    Fixed uGIMP averages gradients and N/r with the physical ring measure;
    its APIC reconstruction uses covariance and a point-value intercept.
    """

    kind: MPMGeometryKind = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    constitutive_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MPMGeometryKind = "cartesian",
        /,
        *,
        spatial_dimension: int = 2,
    ) -> None:
        kind_ = parse(kind, MPMGeometryKind, "kind")
        if spatial_dimension not in (1, 2, 3):
            raise ValueError("MPM spatial_dimension must be one, two, or three.")
        match kind_:
            case "cartesian":
                constitutive_dimension = spatial_dimension
            case "axisymmetric":
                if spatial_dimension != 2:
                    raise ValueError("Axisymmetric MPM requires spatial_dimension=2.")
                constitutive_dimension = 3
            case _ as unreachable:
                assert_never(unreachable)
        self.kind = kind_
        self.spatial_dimension = spatial_dimension
        self.constitutive_dimension = constitutive_dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-geometry",
                "geometry": kind_,
                "spatial_dimension": spatial_dimension,
                "constitutive_dimension": constitutive_dimension,
            }
        )


__all__ = ["MPMGeometryKind", "MPMGeometryPlan"]
