from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

from phydrax import _rocm_environment as environment
from phydrax.backends import (
    assess_rocm_environment,
    RocmDeviceFacts,
    RocmEnvironmentFacts,
    RocmEnvironmentIssue,
    RocmEnvironmentReport,
)


def _facts() -> RocmEnvironmentFacts:
    return RocmEnvironmentFacts(
        host_platform="linux",
        host_machine="x86_64",
        python_implementation="CPython",
        python_version=(3, 13, 2),
        selected_major=10,
        selected_platforms=("rocm",),
        versions=(
            ("jax", "0.12.0"),
            ("jaxlib", "0.12.0"),
            ("jax-rocm10-plugin", "0.12.0"),
            ("jax-rocm10-pjrt", "0.12.0"),
            *((name, "10.1.0") for name in environment.DISTRIBUTIONS),
        ),
        bundle_manifest="/owned/phydrax_rocm_runtime/runtime.json",
        bundle_version="10.1.0",
        bundle_architectures=("gfx1250", "gfx942", "gfx950"),
        bundle_components=tuple(
            (name, "10.1.0", "a" * 64) for name in environment.DISTRIBUTIONS[1:]
        ),
    )


def _observed(version: str = "rocm 71626385") -> RocmEnvironmentFacts:
    return replace(
        _facts(),
        device_probed=True,
        runtime_platform="gpu",
        runtime_version=version,
        devices=(RocmDeviceFacts("gpu", "gpu", version, "AMD GPU", 0, 0),),
    )


def _provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        environment,
        "_runtime_provider",
        lambda: SimpleNamespace(activate_rocm_bundle=lambda: None),
    )


def test_host_assessment_does_not_claim_runtime_or_hardware() -> None:
    report = assess_rocm_environment(_facts())
    assert report.successful
    assert not report.hardware_qualified
    assert not report.facts.device_probed
    assert not report.facts.bundle_loaded
    assert report.facts.runtime_version is None
    assert report.facts.devices == ()
    assert tuple(issue.code for issue in report.issues) == (
        "runtime-requirements-unknown",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"host_platform": "darwin"},
        {"host_platform": "win32"},
        {"host_machine": "aarch64"},
        {"python_implementation": "PyPy"},
        {"python_version": (3, 12, 8)},
        {"python_version": (3, 14, 0)},
        {"python_free_threaded": True},
    ],
)
def test_unsupported_host_is_refused(changes: dict[str, object]) -> None:
    report = assess_rocm_environment(replace(_facts(), **changes))
    assert not report.successful
    assert any(issue.code == "unsupported-host" for issue in report.issues)


@pytest.mark.parametrize(
    "name", ["jax", "jaxlib", "jax-rocm10-plugin", "jax-rocm10-pjrt"]
)
@pytest.mark.parametrize("version", [None, "0.11.2", "0.12.1"])
def test_exact_distribution_pair_is_required(name: str, version: str | None) -> None:
    versions = dict(_facts().versions)
    del versions[name]
    if version is not None:
        versions[name] = version
    assert not assess_rocm_environment(
        replace(_facts(), versions=tuple(versions.items()))
    ).successful


def test_unsupported_plugin_family_is_refused() -> None:
    facts = replace(
        _facts(), versions=(*_facts().versions, ("jax-rocm99-plugin", "0.12.0"))
    )
    report = assess_rocm_environment(facts)
    assert not report.successful
    assert any(issue.code == "conflicting-plugin-majors" for issue in report.issues)


@pytest.mark.parametrize("name", environment.DISTRIBUTIONS)
@pytest.mark.parametrize("version", [None, "6.4.4"])
def test_owned_runtime_components_are_required(name: str, version: str | None) -> None:
    versions = dict(_facts().versions)
    del versions[name]
    if version is not None:
        versions[name] = version
    assert not assess_rocm_environment(
        replace(_facts(), versions=tuple(versions.items()))
    ).successful


@pytest.mark.parametrize(
    "changes",
    [
        {"bundle_manifest": None},
        {"bundle_version": "6.4.4"},
        {"bundle_architectures": ("gfx942",)},
        {"bundle_error": "Invalid payload digest"},
    ],
)
def test_manifest_and_all_architectures_are_required(changes: dict[str, object]) -> None:
    assert not assess_rocm_environment(replace(_facts(), **changes)).successful


def test_external_sdk_is_evidence_not_owned_userspace() -> None:
    facts = replace(_facts(), userspace_root="/opt/rocm", userspace_version="6.4.4")
    assert assess_rocm_environment(facts).successful
    assert not assess_rocm_environment(replace(facts, bundle_manifest=None)).successful


@pytest.mark.parametrize("version", ["rocm 71626385", "ROCm 10.1.0"])
def test_loaded_identity_is_separate_from_sdk_version(version: str) -> None:
    report = assess_rocm_environment(_observed(version))
    assert report.successful
    assert report.facts.runtime_version == version
    assert report.facts.selected_major == 10
    assert not report.hardware_qualified


@pytest.mark.parametrize(
    "platform,client,version",
    [
        ("cpu", "cpu", "rocm 71626385"),
        ("gpu", "gpu", "CUDA 13000"),
        ("gpu", "gpu", "unknown"),
        ("gpu", "cuda", "rocm 71626385"),
    ],
)
def test_display_name_and_raw_gpu_do_not_establish_rocm(
    platform: str, client: str, version: str
) -> None:
    facts = replace(
        _observed(),
        runtime_platform=client,
        runtime_version=version,
        devices=(RocmDeviceFacts(platform, client, version, "AMD MI300X", 0, 0),),
    )
    report = assess_rocm_environment(facts)
    assert not report.successful
    assert any(issue.code == "runtime-identity-mismatch" for issue in report.issues)


def test_runtime_summary_and_device_set_must_match() -> None:
    assert not assess_rocm_environment(
        replace(_observed(), runtime_version="different")
    ).successful
    assert not assess_rocm_environment(replace(_observed(), devices=())).successful


def test_host_collection_is_device_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = dict(_facts().versions)

    def version(name: str) -> str:
        if name not in versions:
            raise environment.metadata.PackageNotFoundError(name)
        return versions[name]

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(
            "Host collection must not discover devices or activate the bundle."
        )

    facts = _facts()
    bundle = SimpleNamespace(
        manifest_path=facts.bundle_manifest,
        sdk_version="10.1.0",
        architectures=facts.bundle_architectures,
        components=facts.bundle_components,
    )
    provider = SimpleNamespace(
        inspect_rocm_bundle=lambda: bundle, activate_rocm_bundle=forbidden
    )
    monkeypatch.setattr(environment.metadata, "version", version)
    monkeypatch.setattr(environment.metadata, "distributions", lambda: ())
    monkeypatch.setattr(environment, "_runtime_provider", lambda: provider)
    monkeypatch.setattr(
        environment, "_userspace_facts", lambda versions: ("/opt/rocm", "6.4.4", ())
    )
    monkeypatch.setattr(environment.jax, "devices", forbidden)
    monkeypatch.setattr(environment.jax, "local_devices", forbidden)
    observed = environment.collect_rocm_environment()
    assert observed.bundle_version == "10.1.0"
    assert observed.userspace_version == "6.4.4"
    assert not observed.device_probed


def test_missing_optional_provider_does_not_import_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> NoReturn:
        raise environment.metadata.PackageNotFoundError(name)

    def forbidden(name: str) -> NoReturn:
        raise AssertionError("Missing provider must not be imported.")

    monkeypatch.setattr(environment.metadata, "distribution", missing)
    monkeypatch.setattr(environment, "import_module", forbidden)
    assert environment._runtime_provider() is None


def test_explicit_system_sdk_manifest_preserves_external_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delitem(
        environment.sys.modules, "phydrax_rocm_runtime._loader", raising=False
    )
    (tmp_path / ".info").mkdir()
    (tmp_path / ".info" / "version").write_text("6.4.4-12345\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    root, version, missing = environment._userspace_facts({})
    assert root == str(tmp_path)
    assert version == "6.4.4"
    assert missing == ()


@pytest.mark.parametrize("selected,initialize", [(("cpu",), True), (("rocm",), False)])
def test_probe_is_opt_in_and_preserves_selection(
    monkeypatch: pytest.MonkeyPatch, selected: tuple[str, ...], initialize: bool
) -> None:
    facts = replace(_facts(), selected_platforms=selected)
    monkeypatch.setattr(environment, "collect_rocm_environment", lambda: facts)

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Unexpected activation or discovery.")

    monkeypatch.setattr(environment, "_runtime_provider", forbidden)
    monkeypatch.setattr(environment.jax, "devices", forbidden)
    assert environment.rocm_environment(initialize_backend=initialize).facts is facts


def test_opt_in_observes_only_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, "collect_rocm_environment", _facts)
    _provider(monkeypatch)
    calls = []

    def devices(backend: str) -> list[SimpleNamespace]:
        calls.append(backend)
        return [
            SimpleNamespace(
                platform="gpu",
                client=SimpleNamespace(platform="gpu", platform_version="rocm 71626385"),
                device_kind="AMD GPU",
                id=0,
                process_index=0,
            )
        ]

    monkeypatch.setattr(environment.jax, "devices", devices)
    report = environment.rocm_environment(initialize_backend=True)
    assert report.successful
    assert report.facts.device_probed
    assert calls == ["rocm"]


def test_initialization_error_preserves_cause_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, "collect_rocm_environment", _facts)
    _provider(monkeypatch)
    calls = []

    def devices(backend: str) -> NoReturn:
        calls.append(backend)
        try:
            raise OSError("libamdhip64.so: missing symbol")
        except OSError as cause:
            raise RuntimeError("ROCm initialization failed") from cause

    monkeypatch.setattr(environment.jax, "devices", devices)
    report = environment.rocm_environment(initialize_backend=True)
    assert not report.successful
    assert report.facts.backend_error is not None
    assert "missing symbol" in report.facts.backend_error
    assert "ROCm initialization failed" in report.facts.backend_error
    assert calls == ["rocm"]


def test_record_validation_and_canonical_versions() -> None:
    facts = _facts()
    assert replace(facts, versions=tuple(reversed(facts.versions))) == facts
    with pytest.raises(ValueError, match="unique"):
        replace(facts, versions=(*facts.versions, ("jax", "0.12.0")))
    with pytest.raises(ValueError, match="device_probed"):
        replace(facts, runtime_version="rocm 71626385")
    with pytest.raises(ValueError, match="must be 10"):
        replace(facts, selected_major=99)
    with pytest.raises(TypeError, match="booleans"):
        replace(facts, python_free_threaded=1)
    with pytest.raises(ValueError):
        RocmEnvironmentIssue("unknown", "error", "invalid")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="facts"):
        RocmEnvironmentReport((), ())  # ty: ignore[invalid-argument-type]


def test_missing_external_sdk_directory_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delitem(
        environment.sys.modules, "phydrax_rocm_runtime._loader", raising=False
    )
    missing = str(tmp_path / "not-provisioned")
    monkeypatch.setenv("ROCM_PATH", missing)
    assert environment._userspace_facts({}) == (missing, None, (missing,))


def test_owned_compiler_setup_does_not_overwrite_external_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".info").mkdir()
    (tmp_path / ".info" / "version").write_text("6.4.4\n")
    provider = SimpleNamespace(
        initial_rocm_environment=lambda: (("ROCM_PATH", str(tmp_path)),)
    )
    monkeypatch.setitem(environment.sys.modules, "phydrax_rocm_runtime._loader", provider)
    monkeypatch.setenv("ROCM_PATH", "/owned/_rocm_sdk_core")
    root, version, missing = environment._userspace_facts({})
    assert (root, version, missing) == (str(tmp_path), "6.4.4", ())
