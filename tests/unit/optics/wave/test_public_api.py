#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import phydrax.optics.wave as wave


def test_wave_core_public_api_is_explicit():
    expected = {
        "AngularSpectrumEvidence",
        "AngularSpectrumPlan",
        "AngularSpectrumResult",
        "AngularSpectrumStatus",
        "IntensityPlane",
        "JonesThinTransmission",
        "PlaneFieldSpace",
        "PreparedAngularSpectrum",
        "ScalarPlaneField",
        "ScalarThinTransmission",
        "TangentialPlaneField",
        "coherent_mode_intensity",
        "ideal_square_law",
        "integrate_intensity",
        "propagate_angular_spectrum",
        "thin_lens",
    }
    assert expected <= set(wave.__all__)
    for name in expected:
        assert vars(wave)[name].__module__.startswith("phydrax.optics.wave")
