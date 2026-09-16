import pytest

import phydrax as phx


def test_medical_physics_capability_profiles_are_exact_and_unreleased():
    imaging = {
        profile.name: profile for profile in phx.imaging.imaging_candidate_profiles()
    }
    assert {
        "imaging.ct.hu-material-calibration",
        "imaging.ct.material-basis-polychromatic",
        "imaging.dicom.enhanced-ct",
        "imaging.dicom.legacy-ct",
        "imaging.dicom.nuclear-medicine-counts",
        "imaging.dicom.pet-activity-concentration",
        "imaging.dicom.rt-dose-linked-plan",
        "imaging.dicom.rt-dose-unlinked",
        "imaging.dicom.rt-plan-metadata",
        "imaging.dicom.rt-structure-closed-planar",
    }.issubset(imaging)
    assert all(not profile.released for profile in imaging.values())

    nuclear = {
        profile.name: profile for profile in phx.nuclear.nuclear_candidate_profiles()
    }
    assert {
        "nuclear.data.diagnostic-photon-coefficients",
        "nuclear.dosimetry.time-activity",
        "nuclear.dosimetry.regional-s-value",
        "nuclear.dosimetry.spatial-s-value",
    }.issubset(nuclear)
    assert all(
        not nuclear[name].released
        for name in nuclear
        if name.startswith("nuclear.dosimetry")
    )

    biophysical = {
        profile.name: profile
        for profile in phx.applications.biophysical_candidate_profiles()
    }
    assert {
        "radiation.blood-dose.ctmc-deterministic",
        "radiation.blood-dose.ctmc-stochastic",
        "radiation.external-score.mcgpu-raw",
        "radiation.external-score.moqui-array",
        "radiation.external-score.openxraymc-hdf5",
    }.issubset(biophysical)
    assert all(
        not biophysical[name].released
        for name in biophysical
        if name.startswith("radiation.")
    )


def test_unknown_medical_imaging_profile_fails_closed():
    with pytest.raises(ValueError, match="Unknown imaging candidate profile"):
        phx.imaging.imaging_candidate_profile("imaging.dicom.generic")
