import sys
from pathlib import Path

import phydrax as phx


cosmology = phx.applications.cosmology


def test_concrete_precision_backends_cache_and_cross_compare(tmp_path):
    worker = Path(__file__).parents[2] / "_linear_theory_worker.py"
    scale = cosmology.CosmologyScaleContract("Mpc", "mass", "time")
    request = cosmology.CosmologyModelRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        neutrinos=(cosmology.MassiveNeutrinoSpecies(0.05),),
    )
    physics = cosmology.LinearTheoryPhysicsPolicy()
    outputs = cosmology.LinearTheoryOutputPolicy(
        request.transfer_fields,
        gauge=request.gauge,
        power_field=request.power_field,
        include_thermodynamics=True,
    )
    assert physics.policy_id and outputs.policy_id
    resources = cosmology.LinearTheoryResourcePolicy(timeout_seconds=60.0)

    class_build = cosmology.BackendBuildManifest(
        backend="class",
        release="fixture",
        application=sys.executable,
        arguments=(str(worker), "{request}", "{output}"),
        license_id="user-installed",
    )
    camb_build = cosmology.BackendBuildManifest(
        backend="camb",
        release="fixture",
        application=sys.executable,
        arguments=(str(worker), "{request}", "{output}"),
        license_id="user-installed",
    )
    class_backend = cosmology.ClassLinearTheoryBackend(
        class_build, resources, str(tmp_path / "class")
    )
    camb_backend = cosmology.CambLinearTheoryBackend(
        camb_build, resources, str(tmp_path / "camb")
    )
    first = class_backend.run(request)
    cached = class_backend.run(request)
    second = camb_backend.run(request)
    assert not first.cache_hit
    assert cached.cache_hit
    assert first.artifact.artifact_id == cached.artifact.artifact_id
    comparison = cosmology.compare_precision_backends(first, second)
    assert bool(comparison.successful)
    assert comparison.maximum_transfer_absolute_error == 0.0
    assert comparison.maximum_power_relative_error == 0.0
