import pytest

import phydrax as phx
from phydrax.domain import Boundary, Interval1d
from phydrax.enforcement import EnforcementProgram, EnforcementSpec


def test_cross_field_point_jet_requires_certified_lifting():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0]

    @geom.Function("x")
    def v(x):
        return 2.0 * x[0]

    boundary = geom.component({"x": Boundary()})
    spec = EnforcementSpec(
        phx.conditions.Residual(
            ("u", "v"), boundary, lambda first, second: first - second
        ),
        field="u",
        transform=phx.enforcement.AffineEnforcementTransform(
            phx.conditions.equal(
                phx.conditions.field_jet("u", "x")
                - phx.conditions.point_jet("v", "anchor"),
                0.0,
            ),
            phx.enforcement.TraceLifting("dirichlet", "x"),
            phx.enforcement.EnforcementProofObligations(
                pivot_identity="u:value",
                support_identity="interval-boundary",
                provider_certified=True,
            ),
        ),
    )
    program = EnforcementProgram.build(
        functions={"u": u, "v": v}, specs=[spec], num_reference=1024
    )

    with pytest.raises(ValueError, match="certified lifting provider"):
        program.apply({"u": u, "v": v})
