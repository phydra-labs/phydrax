from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from phydrax.backends import (
    assess_metal_environment,
    collect_metal_environment,
    MetalEnvironmentFacts,
)


def _facts() -> MetalEnvironmentFacts:
    return MetalEnvironmentFacts(
        host_platform="darwin",
        host_machine="arm64",
        macos_version="15.6.0",
        versions=(
            ("phydrax", "0.2.2"),
            ("phydrax-metal", "0.2.2"),
            ("jax", "0.12.0"),
            ("jaxlib", "0.12.0"),
        ),
        selected_platforms=("mps",),
        build_id="a" * 64,
        source_id="b" * 64,
        companion_platform="mps",
        required_jax="0.12.0",
        required_jaxlib="0.12.0",
        minimum_macos="14.0",
    )


def test_compatible_installation_does_not_claim_hardware_execution() -> None:
    report = assess_metal_environment(_facts())
    assert report.successful
    assert not report.facts.device_probed
    assert not report.facts.backends_initialized
    assert report.facts.runtime_platform is None


@pytest.mark.parametrize(
    ("facts", "code"),
    [
        pytest.param(
            replace(_facts(), host_platform="linux", host_machine="x86_64"),
            "unsupported-host",
            id="host",
        ),
        pytest.param(
            replace(_facts(), macos_version="13.6"), "unsupported-host", id="old-macos"
        ),
        pytest.param(
            replace(_facts(), versions=(("phydrax", "0.2.2"),)),
            "companion-missing",
            id="missing",
        ),
        pytest.param(
            replace(
                _facts(), versions=(("phydrax", "0.2.3"), ("phydrax-metal", "0.2.2"))
            ),
            "companion-version-mismatch",
            id="release",
        ),
        pytest.param(
            replace(_facts(), required_jax="0.11.2"), "jax-version-mismatch", id="jax"
        ),
        pytest.param(
            replace(_facts(), required_jaxlib="0.11.2"),
            "jax-version-mismatch",
            id="jaxlib",
        ),
        pytest.param(
            replace(_facts(), competing_plugins=("jax-mps",)),
            "conflicting-plugins",
            id="collision",
        ),
        pytest.param(
            replace(_facts(), installation_error="Asset checksum mismatch."),
            "companion-invalid",
            id="asset",
        ),
        pytest.param(
            replace(_facts(), companion_platform="metal"),
            "companion-invalid",
            id="foreign-platform",
        ),
        pytest.param(
            replace(
                _facts(),
                device_probed=True,
                backend_error="Device initialization failed.",
            ),
            "backend-unavailable",
            id="initialization",
        ),
        pytest.param(
            replace(_facts(), device_probed=True, runtime_platform="cpu"),
            "runtime-identity-mismatch",
            id="cpu-fallback",
        ),
    ],
)
def test_incompatible_environment_refuses(
    facts: MetalEnvironmentFacts, code: str
) -> None:
    report = assess_metal_environment(facts)
    assert not report.successful
    assert any(
        issue.code == code and issue.severity == "error" for issue in report.issues
    )


def test_explicit_cpu_selection_is_not_a_broken_metal_installation() -> None:
    report = assess_metal_environment(replace(_facts(), selected_platforms=("cpu",)))
    assert report.successful
    assert tuple(issue.code for issue in report.issues) == ("platform-not-selected",)
    assert not report.facts.device_probed


def test_distribution_identity_is_canonical_and_unique() -> None:
    facts = _facts()
    reversed_facts = replace(facts, versions=tuple(reversed(facts.versions)))
    assert reversed_facts == facts
    with pytest.raises(ValueError, match="unique"):
        replace(facts, versions=(*facts.versions, ("jax", "0.11.2")))


@pytest.mark.parametrize("source", ("b" * 64, "c" * 64), ids=("owned", "foreign"))
def test_runtime_identity_binds_the_observed_native_source(source: str) -> None:
    facts = replace(
        _facts(),
        device_probed=True,
        runtime_platform="mps",
        runtime_version=f"phydrax-metal 0.2.2 {source}",
    )
    report = assess_metal_environment(facts)
    assert report.successful is (source == facts.source_id)


def test_missing_native_compatibility_evidence_is_not_accepted() -> None:
    report = assess_metal_environment(replace(_facts(), required_jaxlib=None))
    assert not report.successful
    assert any(issue.code == "companion-invalid" for issue in report.issues)


def test_equivalent_macos_release_spellings_are_admitted() -> None:
    report = assess_metal_environment(
        replace(_facts(), macos_version="14", minimum_macos="14.0")
    )
    assert report.successful


@pytest.mark.parametrize("defect", ("none", "unhashed", "missing", "changed"))
def test_host_collection_binds_the_private_indexing_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    """Synthetic installed metadata exercises host admission, not a device probe."""
    package = tmp_path / "phydrax_metal"
    (package / "lib").mkdir(parents=True)
    library = b"inert host-metadata fixture"
    helper = b'"""Inert host-metadata indexing fixture."""\n'
    (package / "lib/libpjrt_plugin_mps.dylib").write_bytes(library)
    if defect != "missing":
        (package / "_jax_indexing.py").write_bytes(
            b"changed helper" if defect == "changed" else helper
        )
    assets = {
        "lib/libpjrt_plugin_mps.dylib": hashlib.sha256(library).hexdigest(),
    }
    if defect != "unhashed":
        assets["_jax_indexing.py"] = hashlib.sha256(helper).hexdigest()
    record = {
        "version": "0.2.2",
        "platform": "mps",
        "jax_version": "0.12.0",
        "jaxlib_version": "0.12.0",
        "minimum_macos": "14.0",
        "source_id": "b" * 64,
        "library": "lib/libpjrt_plugin_mps.dylib",
        "assets": assets,
    }
    record["build_id"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (package / "build.json").write_text(json.dumps(record))
    distribution = tmp_path / "phydrax_metal-0.2.2.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: phydrax-metal\nVersion: 0.2.2\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    facts = collect_metal_environment()
    assert not facts.device_probed
    if defect == "none":
        assert facts.installation_error is None
        assert facts.build_id == record["build_id"]
    else:
        assert facts.installation_error is not None
        assert "_jax_indexing.py" in facts.installation_error or (
            "private indexing helper" in facts.installation_error
        )
        assert any(
            issue.code == "companion-invalid"
            for issue in assess_metal_environment(facts).issues
        )
