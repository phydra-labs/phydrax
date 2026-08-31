#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import phydrax as phx


def test_velocimetry_and_combinatorial_extensions_are_public():
    assert phx.velocimetry.__all__ == [
        "camera",
        "imaging",
        "io",
        "piv",
        "synthetic",
        "tracking",
    ]
    assert phx.velocimetry.piv.PIVPlan is not None
    assert phx.velocimetry.camera.CameraRig is not None
    assert phx.velocimetry.tracking.STBPlan is not None
    assert phx.velocimetry.synthetic.PIVScenarioPlan is not None
    assert phx.velocimetry.io.VelocimetryArchive is not None
    assert phx.combinatorial.SetPackingSpace is not None
    assert phx.combinatorial.CapacitatedFlowSpace is not None
