#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source boundaries and scientific qualification for device-engine adapters."""

import csv
import hashlib
import io
import json
import math
import subprocess
from dataclasses import asdict

import jax
import pytest

from phydrax.interchange._device_design import DeviceSource
from phydrax.interchange.geant4_detector_design import (
    _correct_gym_sources,
    GYMDetectorProfile,
    read_gym_detector_config,
    read_gym_detector_metrics,
)
from phydrax.interchange.hfss_design import (
    HFSS_DEPENDENCY_PROFILE,
    HFSSDesignProfile,
    HFSSDesignTarget,
    QDESIGNOPTIMIZER_COMMIT,
    read_hfss_design_result,
)


def test_source_pin_rejects_changed_working_bytes_and_unsafe_selectors(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    source_file = tmp_path / "geometry.py"
    source_file.write_text("radius_cm = 4.0\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "geometry.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Adapter Test",
            "-c",
            "user.email=adapter@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "source fixture",
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = DeviceSource(str(tmp_path), commit, "LicenseRef-Test-Only")
    detached = source.snapshot(["geometry.py"], max_bytes=4096)
    source_file.write_text("radius_cm = 40.0\n")
    assert detached["geometry.py"] == b"radius_cm = 4.0\n"
    with pytest.raises(ValueError, match="differs from pinned commit"):
        source.snapshot(["geometry.py"], max_bytes=4096)
    with pytest.raises(ValueError, match="Unsafe"):
        source.snapshot(["../geometry.py"], max_bytes=4096)
    with pytest.raises(ValueError, match="Unsafe"):
        source.snapshot([":(glob)**/*.py"], max_bytes=4096)
    with pytest.raises(ValueError, match="absent"):
        source.snapshot(["missing.py"], max_bytes=4096)


def _hfss_fixture():
    targets = (
        HFSSDesignTarget("frequency", "frequency_hz", ("qubit_1",), 5e9, 1e8),
        HFSSDesignTarget("linewidth", "kappa_hz", ("resonator_1",), 2e6, 1e5),
        HFSSDesignTarget("cross-kerr", "chi_hz", ("qubit_1", "resonator_1"), 1.1e6, 1e5),
        HFSSDesignTarget(
            "participation", "participation", ("qubit_1", "jj_name_qubit_1_"), 0.75, 0.05
        ),
        HFSSDesignTarget(
            "coupling",
            "capacitance_ff",
            ("prime_cpw_name_tee_1_", "second_cpw_name_tee_1_"),
            -3.0,
            1.0,
        ),
    )
    profile = HFSSDesignProfile(
        1,
        (("qubit_1", 4e9, 6e9), ("resonator_1", 6.5e9, 8e9)),
        targets,
    )
    convergence = {
        "columns": ["Tetrahedra", "Delta Freq. [%]"],
        "index": ["1", "2"],
        "data": [[100, None], [300, 0.01]],
    }
    fixture_digest = hashlib.sha256(b"parser-fixture-only").hexdigest()
    source_files = {
        "LICENSE.txt": fixture_digest,
        "poetry.lock": fixture_digest,
        "pyproject.toml": fixture_digest,
        "src/qdesignoptimizer/__init__.py": fixture_digest,
        "tutorials/examples_coupled_transmon_chip/design.py": fixture_digest,
    }
    source_identity = hashlib.sha256(
        json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    exports = {
        name: fixture_digest
        for name in (
            "request.json",
            "eigenmode-mesh.txt",
            "eigenmode-convergence.txt",
            "capacitance-mesh.txt",
            "capacitance-convergence.txt",
        )
    }
    packages = {
        name: {
            "version": version,
            "metadata_sha256": fixture_digest,
            "record_sha256": fixture_digest,
            "direct_url_sha256": None,
            "content_sha256": fixture_digest,
            "file_count": 10,
            "total_bytes": 1000,
        }
        for name, version in HFSS_DEPENDENCY_PROFILE
    }
    result = {
        "profile": asdict(profile),
        "source_commit": QDESIGNOPTIMIZER_COMMIT,
        "source_license_id": "Apache-2.0",
        "source_repository_url": (
            "https://github.com/202Q-lab/QDesignOptimizer/tree/" + QDESIGNOPTIMIZER_COMMIT
        ),
        "source_files_sha256": source_files,
        "source_tree_identity": source_identity,
        "dependency_lock_sha256": fixture_digest,
        "worker_sha256": fixture_digest,
        "qdesignoptimizer_version": "0.2.0",
        "python_version": "3.12.0",
        "python_sha256": fixture_digest,
        "python_license_id": "PSF-2.0",
        "python_source_url": "https://python.org/",
        "engine_version": "2022.2.0",
        "engine_sha256": fixture_digest,
        "engine_license_id": "LicenseRef-Ansys",
        "engine_source_url": "https://www.ansys.com/",
        "packages": packages,
        "mode_identity_scope": "upstream-target-frequency-rank-with-disjoint-owner-windows",
        "mesh_identity_scope": "exported-statistics-and-adaptive-history-not-connectivity",
        "mode_labels": ["qubit_1", "resonator_1"],
        "junction_labels": ["jj_name_qubit_1_"],
        "eigenmode_frequency_hz": [5.1e9, 7.1e9],
        "eigenmode_kappa_hz": [1e5, 2e6],
        "epr_frequency_hz": [5e9, 7e9],
        "chi_hz": [[210e6, 1.1e6], [1.1e6, 0.1e6]],
        "participation": [[0.75], [0.02]],
        "eigenmode_convergence": convergence,
        "capacitance_convergence": convergence,
        "capacitance_ff": {
            "index": ["prime_cpw_name_tee_1_", "second_cpw_name_tee_1_"],
            "columns": ["prime_cpw_name_tee_1_", "second_cpw_name_tee_1_"],
            "data": [[40, -3.2], [-3.2, 50]],
        },
        "mesh_exports": exports,
        "mesh_identity": hashlib.sha256(
            json.dumps(exports, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return profile, result


def test_hfss_qualifies_eigenmode_epr_and_signed_capacitance_targets():
    profile, result = _hfss_fixture()
    values, _ = read_hfss_design_result(json.dumps(result).encode(), profile)
    assert values == (5e9, 2e6, 1.1e6, 0.75, -3.2)


def test_hfss_rejects_mode_crossing_in_eigenmode_or_epr_results():
    profile, result = _hfss_fixture()
    result["eigenmode_frequency_hz"] = [7.1e9, 5.1e9]
    with pytest.raises(ValueError, match="mode identity"):
        read_hfss_design_result(json.dumps(result).encode(), profile)
    profile, result = _hfss_fixture()
    result["epr_frequency_hz"] = [7.1e9, 5.1e9]
    with pytest.raises(ValueError, match="mode identity"):
        read_hfss_design_result(json.dumps(result).encode(), profile)


def test_hfss_rejects_unconverged_tampered_or_unpinned_evidence():
    profile, result = _hfss_fixture()
    result["eigenmode_convergence"]["data"][-1][1] = 0.1
    with pytest.raises(ValueError, match="convergence tolerance"):
        read_hfss_design_result(json.dumps(result).encode(), profile)
    profile, result = _hfss_fixture()
    result["mesh_exports"]["eigenmode-mesh.txt"] = "0" * 64
    with pytest.raises(ValueError, match="Mesh evidence identity"):
        read_hfss_design_result(json.dumps(result).encode(), profile)
    profile, result = _hfss_fixture()
    result["packages"]["pyaedt"]["version"] = "unqualified"
    with pytest.raises(ValueError, match="dependency version mismatch"):
        read_hfss_design_result(json.dumps(result).encode(), profile)


def test_hfss_target_schema_has_no_s_parameter_substitute():
    with pytest.raises(ValueError, match="quantity and label arity"):
        HFSSDesignTarget("s21", "s_parameter", ("port1", "port2"), 0.0, 1.0)


def _detector_fixture(*, bad_bin=None, drop_last=False):
    parameters, fits = io.StringIO(), io.StringIO()
    parameter_fields = (
        "etarange",
        "prange",
        "dp_p_p",
        "error_dp_p_p",
        "dth_th_p",
        "error_dth_th_p",
        "dph_ph_p",
        "error_dph_ph_p",
        "dca2d",
        "error_dca2d",
        "dca2d_v_pT",
        "error_dca2d_v_pT",
        "InEfficiency",
        "error_InEfficiency",
        "KF_InEfficiency",
        "error_KF_InEfficiency",
        "GlobalKFInEff",
        "error_GlobalKFInEff",
        "EventsInBin",
        "A1",
        "A2",
        "A1A2",
        "Integral",
        "PseudoGlobalKFInEff",
        "error_PseudoGlobalKFInEff",
        "OverFlowUnderFlowFlag",
    )
    fit_fields = (
        "etarange",
        "prange",
        "Chi2_dpp",
        "NDF_dpp",
        "Chi2_dth",
        "NDF_dth",
        "Chi2_dph",
        "NDF_dph",
        "Chi2_dca2d",
        "NDF_dca2d",
    )
    writer = csv.DictWriter(parameters, fieldnames=parameter_fields)
    writer.writeheader()
    fit_writer = csv.DictWriter(fits, fieldnames=fit_fields)
    fit_writer.writeheader()
    eta = (-3.4, -2, -1, 1, 2, 3.4)
    momentum = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    for i in range(5):
        for j in range(10):
            if drop_last and (i, j) == (4, 9):
                continue
            identity = {
                "etarange": f"{eta[i]} - {eta[i + 1]}",
                "prange": f"{momentum[j]} - {momentum[j + 1]}",
            }
            row = dict.fromkeys(parameter_fields, 0)
            row.update(
                {
                    **identity,
                    "dp_p_p": 1.5,
                    "error_dp_p_p": 0.2,
                    "KF_InEfficiency": 0.1,
                    "error_KF_InEfficiency": 0.03,
                    "EventsInBin": 100,
                    "OverFlowUnderFlowFlag": 0,
                }
            )
            if (i, j) == (0, 0) and bad_bin:
                row.update(bad_bin)
            writer.writerow(row)
            fit = dict.fromkeys(fit_fields, 0)
            fit.update({**identity, "Chi2_dpp": 18, "NDF_dpp": 20})
            fit_writer.writerow(fit)
    return parameters.getvalue().encode(), fits.getvalue().encode()


def test_detector_parser_preserves_fit_and_binomial_uncertainty():
    bins = read_gym_detector_metrics(*_detector_fixture())
    first = bins[0]
    assert first.eta_range == (-3.4, -2.0)
    assert first.momentum_range_gev == (1.0, 2.0)
    assert first.momentum_resolution_percent == 1.5
    assert first.momentum_fit_error_percent == 0.2
    assert first.generated_tracks == 100
    assert first.estimated_reconstructed_tracks == 90
    assert first.kalman_binomial_error == pytest.approx(
        math.sqrt(first.kalman_inefficiency * (1 - first.kalman_inefficiency) / 100)
    )
    with pytest.raises(ValueError, match="every declared"):
        read_gym_detector_metrics(*_detector_fixture(drop_last=True))


@pytest.mark.parametrize(
    "bad_bin, reason",
    [
        ({"dp_p_p": "nan"}, "Nonfinite"),
        ({"error_dp_p_p": 0}, "fit is not qualified"),
        ({"OverFlowUnderFlowFlag": 1}, "overflow/underflow"),
        (
            {
                "EventsInBin": 20,
                "KF_InEfficiency": 0.1,
                "error_KF_InEfficiency": math.sqrt(0.1 * 0.9 / 20),
            },
            "Insufficient reconstructed tracks",
        ),
        ({"error_KF_InEfficiency": 0.2}, "binomial error"),
    ],
)
def test_detector_failures_never_become_training_penalties(bad_bin, reason):
    with pytest.raises(ValueError, match=reason):
        read_gym_detector_metrics(*_detector_fixture(bad_bin=bad_bin))


def test_detector_rejects_excessive_reduced_chi_squared():
    parameters, fits = _detector_fixture()
    rows = list(csv.DictReader(io.StringIO(fits.decode())))
    rows[0]["Chi2_dpp"] = "220"
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError, match="reduced chi-square"):
        read_gym_detector_metrics(parameters, output.getvalue().encode())


def test_corrected_silicon_profile_records_every_exact_source_stage():
    inputs = {
        "G4_Barrel_EIC.C": (
            b"void BarrelSetup(PHG4Reco* g4Reco)\n"
            b"pitch / 10000. / sqrt(12.)\n"
            b"pitch / 10000. / sqrt(12.)\n"
        ),
        "G4_FST_EIC.C": (
            b'Form("SI_L%i_THICKNESS", j + 1)]*Units::um\n'
            b'Form("SI_L%i_THICKNESS", j)]*Units::um\n'
        ),
        "G4_TrackingSupport.C": (b'Form("SI_L%i_THICKNESS", ilyr)] * Units::um\n'),
    }
    corrected, records = _correct_gym_sources(inputs)
    assert len(records) == 5
    assert [record["stage"] for record in records] == [1, 2, 3, 4, 5]
    for record in records:
        assert record["before_sha256"] != record["after_sha256"]
        assert record["before"].encode() not in corrected[record["path"]]
    assert b"double BarrelSetup" in corrected["G4_Barrel_EIC.C"]
    assert corrected["G4_Barrel_EIC.C"].count(b"pitch / sqrt(12.)") == 2
    assert corrected["G4_FST_EIC.C"].count(b"9.37 / 100.") == 2


def _settings():
    return (
        ("NLAYERS_SI_BAR", 2.0),
        ("NLAYERS_SI_EDISK", 1.0),
        ("NLAYERS_SI_HDISK", 1.0),
        ("MIN_RADIUS", 3.3),
        ("MAX_RADIUS", 51.0),
        ("MIN_HZ", 10.0),
        ("MAX_HZ", 125.0),
        ("MIN_EZ", -10.0),
        ("MAX_EZ", -108.0),
        ("MAX_VTX_SPRT_RADIUS", 10.0),
        ("MAX_SAG_SPRT_RADIUS", 20.0),
        ("AVG_SUPRT_THICKNESS", 1.0),
        ("PDG", -211.0),
        ("N_PER_EVENT", 5.0),
        ("VERTEX_DISTRIBUTION_WIDTH", 5.0),
    )


def test_unpatched_known_bad_detector_profile_is_never_accepted():
    with pytest.raises(ValueError, match="corrected-silicon-units"):
        GYMDetectorProfile(
            _settings(), 200, 123, "container@sha256:test", "upstream-unpatched"
        )


def test_device_adapter_entrypoints_refuse_jax_tracing():
    @jax.jit
    def parse_detector_config():
        read_gym_detector_config(b"B_FIELD : 1.4\n")
        return 1

    @jax.jit
    def construct_hfss_target(value):
        HFSSDesignTarget("frequency", "frequency_hz", ("qubit_1",), value, 1.0)
        return value

    with pytest.raises(TypeError):
        parse_detector_config()
    with pytest.raises(TypeError):
        construct_hfss_target(5e9)
