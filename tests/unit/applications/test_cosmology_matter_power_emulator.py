from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._matter_power_emulator import (
    ExternalMatterPowerResult,
    MatterPowerEvaluationRequest,
    MatterPowerProviderError,
    SubprocessMatterPowerBackend,
)
from phydrax.interchange import AdapterStatus
from phydrax.qualification import ReferenceArtifactManifest


cosmology = phx.applications.cosmology


_WORKER = r"""from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = Path(sys.argv[2])
mode = sys.argv[3]
scale_factors = np.asarray(request["scale_factors"], dtype="float64")
wavenumbers = np.asarray(request["wavenumbers"], dtype="float64")
descriptor = dict(request["descriptor"])
cosmology = request["cosmology"]
provider = request["provider_contract"]
field_factor = {
    "cold_baryon": 1.0,
    "total_matter": 2.0,
    "massive_neutrino_total": 3.0,
}[descriptor["left_field"]]
stage_factor = (
    np.ones_like(wavenumbers)
    if descriptor["stage"] == "linear"
    else 1.0 + 0.25 * wavenumbers
)
power = (
    cosmology["primordial_amplitude"]
    * 1.0e9
    * field_factor
    * scale_factors[:, None] ** 2
    * wavenumbers[None, :] ** cosmology["primordial_tilt"]
    * stage_factor[None, :]
)
metadata = {
    "request_id": request["request_id"],
    "scale": cosmology["scale"],
    "descriptor": descriptor,
    "scale_factor_unit": request["scale_factor_unit"],
    "wavenumber_unit": request["wavenumber_unit"],
    "power_unit": request["power_unit"],
    "support": {
        "scale_factor_min": 0.25,
        "scale_factor_max": 1.25,
        "wavenumber_min": 0.01,
        "wavenumber_max": 1.0,
        "complete": True,
    },
    "evaluation": {
        "scale_factor_coordinates": "exact-request-grid",
        "wavenumber_coordinates": "exact-request-grid",
        "clamping_applied": False,
        "extrapolation_applied": False,
    },
    "neutrino_semantics": {
        "representation": "explicit-massive-neutrino-species",
        "effective_neutrino_number": cosmology["effective_neutrino_number"],
        "species": cosmology["neutrinos"],
    },
    "reference_manifest": provider["reference_manifest"],
    "producer": provider["producer"],
}
if mode == "outside":
    metadata["support"]["wavenumber_min"] = 0.15
elif mode == "incomplete":
    metadata["support"]["complete"] = False
elif mode == "bad-stage":
    metadata["descriptor"]["stage"] = (
        "linear" if descriptor["stage"] == "nonlinear" else "nonlinear"
    )
elif mode == "bad-field":
    metadata["descriptor"]["left_field"] = "cold_baryon"
elif mode == "bad-scale":
    metadata["scale"] = {**metadata["scale"], "scale_id": "wrong-scale"}
elif mode == "bad-neutrinos":
    metadata["neutrino_semantics"]["representation"] = "summed-mass"
elif mode == "clamped":
    metadata["evaluation"]["clamping_applied"] = True
elif mode == "no-manifest":
    del metadata["reference_manifest"]
elif mode == "malformed-power":
    power[0, 0] = np.nan
elif mode == "wrong-grid":
    wavenumbers = wavenumbers.copy()
    wavenumbers[0] *= 1.1
np.savez(
    output,
    metadata_json=np.asarray(json.dumps(metadata, allow_nan=False, sort_keys=True)),
    scale_factors=scale_factors,
    wavenumbers=wavenumbers,
    power_values=power,
)
print(f"computed analytic {descriptor['stage']} {descriptor['left_field']} power")
"""


def _write_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "analytic_matter_power.py"
    worker.write_text(_WORKER, encoding="utf-8")
    return worker


def _manifest(
    worker: Path, /, *, commercial_use_permitted: bool = True
) -> ReferenceArtifactManifest:
    payload = worker.read_bytes()
    return ReferenceArtifactManifest(
        "analytic-matter-power-test-provider",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-PHYDRA-Test",
        commercial_use_permitted=commercial_use_permitted,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="test-only",
        nondimensionalization={"analytic_power_amplitude": 1.0},
        uncertainty=None,
        lineage_ids=("analytic-power-law",),
    )


def _request(
    field: str = "total_matter", stage: str = "nonlinear"
) -> MatterPowerEvaluationRequest:
    scale = cosmology.CosmologyScaleContract(
        phx.units.MEGAPARSEC,
        phx.units.SOLAR_MASS,
        phx.units.GIGAYEAR,
        length_coordinate_kind="comoving",
    )
    model = cosmology.CosmologyModelRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        effective_neutrino_number=3.046,
        neutrinos=(
            cosmology.MassiveNeutrinoSpecies(
                0.06,
                temperature_ratio=0.71611,
                distribution_id="fermi-dirac-zero-chemical-potential",
            ),
        ),
        primordial_amplitude=2.1e-9,
        primordial_tilt=0.965,
        power_field=field,
    )
    descriptor = cosmology.MatterPowerDescriptor(
        field,
        field,
        gauge="synchronous",
        stage=stage,
        shot_noise="none",
        spatial_dimension=3,
    )
    return MatterPowerEvaluationRequest(
        model,
        np.asarray([0.5, 0.75, 1.0]),
        np.asarray([0.1, 0.2, 0.4]),
        descriptor,
    )


def _backend(
    worker: Path,
    manifest: ReferenceArtifactManifest,
    mode: str = "ok",
    **options,
) -> SubprocessMatterPowerBackend:
    return SubprocessMatterPowerBackend(
        sys.executable,
        worker,
        manifest,
        arguments=("{reference_artifact}", "{request}", "{output}", mode),
        timeout_seconds=30.0,
        backend_name="analytic-matter-power",
        backend_version="test-provider-1",
        build_id="analytic-worker-build",
        numerical_policy_id="analytic-exact-grid",
        **options,
    )


@pytest.mark.parametrize(
    ("field", "stage", "field_factor"),
    (("cold_baryon", "linear", 1.0), ("total_matter", "nonlinear", 2.0)),
)
def test_analytic_provider_preserves_power_semantics_and_external_evidence(
    tmp_path: Path, field: str, stage: str, field_factor: float
):
    worker = _write_worker(tmp_path)
    manifest = _manifest(worker)
    request = _request(field, stage)

    result = _backend(worker, manifest).run(request)

    assert isinstance(result, ExternalMatterPowerResult)
    expected_stage = (
        np.ones(3) if stage == "linear" else 1.0 + 0.25 * np.asarray(request.wavenumbers)
    )
    expected = (
        request.cosmology.primordial_amplitude
        * 1.0e9
        * field_factor
        * np.asarray(request.scale_factors)[:, None] ** 2
        * np.asarray(request.wavenumbers)[None, :] ** request.cosmology.primordial_tilt
        * expected_stage[None, :]
    )
    np.testing.assert_allclose(result.table.power_values, expected)
    assert result.table.descriptor.descriptor_id == request.descriptor.descriptor_id
    assert result.table.descriptor.left_field == field
    assert result.table.descriptor.stage == stage
    assert result.table.scale.length_coordinate_kind == "comoving"
    assert result.table.power_unit == request.cosmology.scale.power_spectrum_unit
    differentiation = result.table.provenance.differentiation
    assert (
        differentiation.contract_id
        == phx.artifacts.DifferentiationContract.constant().contract_id
    )
    assert not differentiation.upstream_physical_parameters
    assert not differentiation.stored_values
    assert not differentiation.query_coordinates
    gradient = jax.grad(lambda k: result.table.evaluate(k, 0.75))(jnp.asarray(0.2))
    np.testing.assert_allclose(gradient, 0.0)
    assert result.support.rectangular_range_covered
    assert result.support.provider_support_complete
    assert result.support.complete
    assert result.reference_manifest.manifest_id == manifest.manifest_id
    assert result.artifact.license_id == manifest.license_id
    assert manifest.manifest_id in result.artifact.parent_artifact_ids
    assert result.report.status == AdapterStatus.LOSSLESS
    assert result.report.valid
    assert result.process.return_code == 0
    assert result.process.request_bytes > 0
    assert result.process.result_bytes > 0
    assert "computed analytic" in result.process.standard_output
    mapping = request.to_mapping()
    assert mapping["scale_factor_unit"] == "dimensionless"
    assert mapping["wavenumber_unit"] == request.cosmology.scale.wavenumber_unit.to_dict()
    assert mapping["power_unit"] == request.cosmology.scale.power_spectrum_unit.to_dict()


def test_rectangular_range_and_incomplete_provider_support_are_distinct(tmp_path: Path):
    worker = _write_worker(tmp_path)
    manifest = _manifest(worker)
    request = _request()

    with pytest.raises(MatterPowerProviderError) as outside_error:
        _backend(worker, manifest, "outside").run(request)
    outside = outside_error.value
    assert outside.reason == "outside-rectangular-support"
    assert outside.support is not None
    assert not outside.support.rectangular_range_covered
    assert outside.support.provider_support_complete

    with pytest.raises(MatterPowerProviderError) as incomplete_error:
        _backend(worker, manifest, "incomplete").run(request)
    incomplete = incomplete_error.value
    assert incomplete.reason == "incomplete-provider-support"
    assert incomplete.support is not None
    assert incomplete.support.rectangular_range_covered
    assert not incomplete.support.provider_support_complete
    assert not incomplete.support.complete


@pytest.mark.parametrize(
    ("mode", "reason"),
    (
        ("bad-stage", "stage-field-mismatch"),
        ("bad-field", "stage-field-mismatch"),
        ("bad-scale", "scale-mismatch"),
        ("bad-neutrinos", "unsupported-neutrino-semantics"),
        ("clamped", "hidden-domain-transformation"),
        ("no-manifest", "malformed-output"),
        ("malformed-power", "malformed-power-values"),
        ("wrong-grid", "wavenumber-grid-mismatch"),
    ),
)
def test_provider_semantic_and_malformed_outputs_fail_closed(
    tmp_path: Path, mode: str, reason: str
):
    worker = _write_worker(tmp_path)
    manifest = _manifest(worker)

    with pytest.raises(MatterPowerProviderError) as error:
        _backend(worker, manifest, mode).run(_request())

    assert error.value.reason == reason


def test_subprocess_bounds_rights_and_checksum_are_enforced(tmp_path: Path):
    worker = _write_worker(tmp_path)
    manifest = _manifest(worker)
    request = _request()

    with pytest.raises(MatterPowerProviderError, match="byte limit") as size_error:
        _backend(worker, manifest, maximum_result_bytes=100).run(request)
    assert size_error.value.reason == "result-size-limit"

    denied = _manifest(worker, commercial_use_permitted=False)
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        _backend(worker, denied, commercial_use=True)

    backend = _backend(worker, manifest)
    worker.write_text(_WORKER + "\n# changed after admission\n", encoding="utf-8")
    with pytest.raises(MatterPowerProviderError) as checksum_error:
        backend.run(request)
    assert checksum_error.value.reason == "reference-artifact-mismatch"
