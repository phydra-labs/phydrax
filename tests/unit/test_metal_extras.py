"""The Metal extra resolves only the exact companion on supported Apple hosts."""

import hashlib
import json
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tools import check_metal_extras as extras
from tools.check_cuda_extras import wheel_metadata


LIBRARY = b"\xcf\xfa\xed\xfe pjrt plugin"
METALLIB = b"MTLB metal library"
INDEXING_HELPER = b'"""Private standalone indexing helper fixture."""\n'
CORE_REQUIREMENTS = (
    "jax>=0.12.0,<0.13",
    'phydrax-metal==0.2.2; extra == "metal"',
)


def _write_wheel(path: Path, members: Mapping[str, bytes]) -> Path:
    dist_info = next(
        name.split("/")[0]
        for name in members
        if name.split("/")[0].endswith(".dist-info")
    )
    record = "".join(
        f"{name},{extras.record_digest(payload)},{len(payload)}\n"
        for name, payload in members.items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr(f"{dist_info}/RECORD", record + f"{dist_info}/RECORD,,\n")
    return path


def _metadata(name: str, version: str, lines: Sequence[str]) -> bytes:
    return "\n".join(
        ("Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}", *lines, "")
    ).encode()


def _wheel_file(tag: str, purelib: str) -> bytes:
    return (
        f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: {purelib}\nTag: {tag}\n"
    ).encode()


def _core_wheel(
    directory: Path,
    *,
    requirements: Sequence[str] = CORE_REQUIREMENTS,
    provides: Sequence[str] = ("metal",),
) -> Path:
    info = "phydrax-0.2.2.dist-info"
    lines = [
        "Requires-Python: >=3.13,<3.14",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        *(f"Provides-Extra: {extra}" for extra in provides),
    ]
    return _write_wheel(
        directory / "phydrax-0.2.2-py3-none-any.whl",
        {
            "phydrax/__init__.py": b"",
            f"{info}/METADATA": _metadata("phydrax", "0.2.2", lines),
            f"{info}/WHEEL": _wheel_file("py3-none-any", "true"),
        },
    )


def _manifest(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "version": "0.2.2",
        "platform": "mps",
        "jax_version": "0.12.0",
        "jaxlib_version": "0.12.0",
        "pjrt_c_api": {"major": 0, "minor": 75},
        "stablehlo_version": "1.13.0",
        "minimum_macos": "14.0",
        "library": "lib/libpjrt_plugin_mps.dylib",
        "assets": {
            "lib/libpjrt_plugin_mps.dylib": hashlib.sha256(LIBRARY).hexdigest(),
            "lib/mlx.metallib": hashlib.sha256(METALLIB).hexdigest(),
            "_jax_indexing.py": hashlib.sha256(INDEXING_HELPER).hexdigest(),
        },
        "sources": {
            "lock_sha256": "a" * 64,
            "entries": [
                {"name": "jax-mps", "commit": "127ab63d0cab71320702e562703c4aaa0c6570d1"}
            ],
            "patches": [],
        },
        "toolchain": {"cmake": "4.1.0", "xcode": "16.4"},
        "source_id": "b" * 64,
    }
    record.update(overrides)
    if "build_id" not in overrides:
        content = {key: value for key, value in record.items() if key != "build_id"}
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        record["build_id"] = hashlib.sha256(encoded).hexdigest()
    return record


def _companion_wheel(
    directory: Path,
    *,
    version: str = "0.2.2",
    tag: str = "py3-none-macosx_14_0_arm64",
    manifest: Mapping[str, object] | None = None,
    requires: Sequence[str] = ("jax==0.12.0", "jaxlib==0.12.0"),
    requires_python: str = ">=3.13,<3.14",
    entry_points: str = "[jax_plugins]\nmps = phydrax_metal.plugin\n",
    purelib: str = "false",
    library: bytes = LIBRARY,
    indexing_helper: bytes = INDEXING_HELPER,
    tamper_record: bool = False,
) -> Path:
    info = f"phydrax_metal-{version}.dist-info"
    lines = [
        f"Requires-Python: {requires_python}",
        *(f"Requires-Dist: {requirement}" for requirement in requires),
    ]
    path = _write_wheel(
        directory / f"phydrax_metal-{version}-{tag}.whl",
        {
            "phydrax_metal/__init__.py": b"",
            "phydrax_metal/plugin.py": b"",
            "phydrax_metal/_jax_indexing.py": indexing_helper,
            "phydrax_metal/build.json": json.dumps(
                _manifest() if manifest is None else manifest
            ).encode(),
            "phydrax_metal/lib/libpjrt_plugin_mps.dylib": library,
            "phydrax_metal/lib/mlx.metallib": METALLIB,
            f"{info}/METADATA": _metadata("phydrax-metal", version, lines),
            f"{info}/WHEEL": _wheel_file(tag, purelib),
            f"{info}/entry_points.txt": entry_points.encode(),
        },
    )
    if tamper_record:
        tampered = path.with_suffix(".tmp")
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(tampered, "w") as target:
            for member in source.infolist():
                payload = source.read(member)
                if member.filename == "phydrax_metal/plugin.py":
                    payload = b"import os\n"
                target.writestr(member, payload)
        tampered.replace(path)
    return path


def _stub_wheel(directory: Path, name: str, version: str) -> None:
    info = f"{name}-{version}.dist-info"
    _write_wheel(
        directory / f"{name}-{version}-py3-none-any.whl",
        {
            f"{name}/__init__.py": b"",
            f"{info}/METADATA": _metadata(name, version, ()),
            f"{info}/WHEEL": _wheel_file("py3-none-any", "true"),
        },
    )


def _companion_errors(
    core: Path, companion: Path
) -> tuple[list[str], extras.CompanionFacts | None]:
    metadata = wheel_metadata(core)
    assert not isinstance(metadata, str)
    return extras.companion_errors(
        companion, core_version="0.2.2", core_metadata=metadata
    )


def test_built_metal_wheels_satisfy_every_metadata_obligation(tmp_path: Path) -> None:
    core = _core_wheel(tmp_path)
    metadata = wheel_metadata(core)
    assert not isinstance(metadata, str)
    assert extras.core_metadata_errors(metadata) == []
    errors, facts = _companion_errors(core, _companion_wheel(tmp_path))
    assert errors == []
    assert facts == {
        "version": "0.2.2",
        "jax_version": "0.12.0",
        "jaxlib_version": "0.12.0",
        "minimum_macos": "14.0",
        "build_id": _manifest()["build_id"],
        "source_id": "b" * 64,
        "tags": ["py3-none-macosx_14_0_arm64"],
    }


@pytest.mark.parametrize(
    ("requirements", "provides", "diagnostic"),
    [
        (
            (
                "jax>=0.12.0,<0.13",
                "phydrax-metal==0.2.2; sys_platform == 'darwin' and extra == 'metal'",
            ),
            ("metal",),
            "unmarked lockstep requirement",
        ),
        (
            ("jax>=0.12.0,<0.13", 'phydrax-metal>=0.2.2; extra == "metal"'),
            ("metal",),
            "lockstep",
        ),
        (
            ("jax>=0.12.0,<0.13", 'phydrax-metal==0.2.1; extra == "metal"'),
            ("metal",),
            "lockstep",
        ),
        (CORE_REQUIREMENTS, (), "Missing metal extra"),
        (
            (*CORE_REQUIREMENTS, 'jax-mps==0.11.0; extra == "metal"'),
            ("metal",),
            "Upstream Apple",
        ),
        (('phydrax-metal==0.2.2; extra == "metal"',), ("metal",), "base jax requirement"),
    ],
    ids=[
        "platform-marked",
        "range",
        "stale-version",
        "no-extra",
        "foreign-plugin",
        "no-jax",
    ],
)
def test_core_metadata_refuses_non_lockstep_or_silent_cpu_extras(
    tmp_path: Path, requirements: Sequence[str], provides: Sequence[str], diagnostic: str
) -> None:
    metadata = wheel_metadata(
        _core_wheel(tmp_path, requirements=requirements, provides=provides)
    )
    assert not isinstance(metadata, str)
    assert any(diagnostic in error for error in extras.core_metadata_errors(metadata))


_DEFECTS: dict[str, tuple[dict[str, Any], str]] = {
    "core-version-mismatch": (
        {"version": "0.2.3", "manifest": _manifest(version="0.2.3")},
        "does not match core",
    ),
    "jax-outside-core-range": (
        {
            "requires": ("jax==0.11.2", "jaxlib==0.11.2"),
            "manifest": _manifest(jax_version="0.11.2", jaxlib_version="0.11.2"),
        },
        "outside core requirement",
    ),
    "unmatched-jaxlib": (
        {"manifest": _manifest(jaxlib_version="0.12.1")},
        "matched pair",
    ),
    "pin-differs-from-manifest": (
        {"requires": ("jax==0.12.1", "jaxlib==0.12.1")},
        "pin exactly",
    ),
    "range-not-pin": ({"requires": ("jax>=0.12.0", "jaxlib==0.12.0")}, "pin exactly"),
    "upstream-plugin-dependency": (
        {"requires": ("jax==0.12.0", "jaxlib==0.12.0", "jax-mps==0.11.0")},
        "pin exactly",
    ),
    "python-excluded": ({"requires_python": ">=3.14"}, "Requires-Python"),
    "python-unbounded": ({"requires_python": ">=3.13"}, "Requires-Python"),
    "python-previous": ({"requires_python": ">=3.12,<3.13"}, "Requires-Python"),
    "metal-platform-token": ({"manifest": _manifest(platform="metal")}, "is not 'mps'"),
    "stale-build-id": (
        {"manifest": {**_manifest(), "minimum_macos": "15.0"}},
        "build_id",
    ),
    "schema-version-field": ({"manifest": _manifest(schemaVersion=1)}, "fields differ"),
    "missing-source-lock": (
        {"manifest": _manifest(sources={"entries": [], "patches": []})},
        "lock_sha256",
    ),
    "library-not-hashed": (
        {"manifest": _manifest(library="lib/other.dylib")},
        "library is not",
    ),
    "indexing-helper-not-hashed": (
        {
            "manifest": _manifest(
                assets={
                    "lib/libpjrt_plugin_mps.dylib": hashlib.sha256(LIBRARY).hexdigest(),
                    "lib/mlx.metallib": hashlib.sha256(METALLIB).hexdigest(),
                }
            )
        },
        "private indexing helper",
    ),
    "indexing-helper-bytes-differ": (
        {"indexing_helper": b"changed helper"},
        "differs from wheel bytes",
    ),
    "absolute-asset": (
        {
            "manifest": _manifest(
                assets={"/opt/homebrew/lib/libmlx.dylib": "c" * 64},
                library="/opt/homebrew/lib/libmlx.dylib",
            )
        },
        "package-relative",
    ),
    "asset-bytes-differ": ({"library": b"another build"}, "differs from wheel bytes"),
    "duplicate-entry-point": (
        {
            "entry_points": "[jax_plugins]\nmps = phydrax_metal.plugin\nmetal = phydrax_metal.plugin\n"
        },
        "jax_plugins entry",
    ),
    "upstream-entry-point": (
        {"entry_points": "[jax_plugins]\nmps = jax_plugins.mps\n"},
        "jax_plugins entry",
    ),
    "pure-python-wheel": ({"purelib": "true"}, "Root-Is-Purelib"),
    "intel-tag": ({"tag": "py3-none-macosx_14_0_x86_64"}, "macOS arm64 platform tag"),
    "linux-tag": ({"tag": "py3-none-manylinux_2_28_aarch64"}, "macOS arm64 platform tag"),
    "deployment-disagrees": (
        {"tag": "py3-none-macosx_13_0_arm64"},
        "disagrees with manifest minimum",
    ),
    "tampered-record": ({"tamper_record": True}, "RECORD hash or size mismatch"),
}


@pytest.mark.parametrize("defect", sorted(_DEFECTS))
def test_companion_wheel_defects_are_refused(tmp_path: Path, defect: str) -> None:
    arguments, diagnostic = _DEFECTS[defect]
    companion = _companion_wheel(tmp_path, **arguments)
    errors, _ = _companion_errors(_core_wheel(tmp_path), companion)
    assert any(diagnostic in error for error in errors), errors


def test_version_variant_is_a_consistent_wheel_that_only_breaks_lockstep(
    tmp_path: Path,
) -> None:
    variant = extras.variant_companion(
        _companion_wheel(tmp_path), tmp_path / "variant", version="0.2.3"
    )
    assert variant.name == "phydrax_metal-0.2.3-py3-none-macosx_14_0_arm64.whl"
    metadata = wheel_metadata(variant)
    assert not isinstance(metadata, str)
    assert str(metadata["Version"]) == "0.2.3"
    errors, _ = _companion_errors(_core_wheel(tmp_path), variant)
    assert not [error for error in errors if "RECORD" in error]
    assert any("does not match core 0.2.2" in error for error in errors)


@pytest.mark.parametrize(
    ("case", "resolved", "diagnostic"),
    [
        (
            "metal-supported",
            {"phydrax": "0.2.2", "jax": "0.12.0", "jaxlib": "0.12.0"},
            "expected",
        ),
        (
            "metal-supported",
            {
                "phydrax": "0.2.2",
                "phydrax-metal": "0.2.2",
                "jax": "0.12.1",
                "jaxlib": "0.12.1",
            },
            "expected",
        ),
        (
            "metal-supported",
            {
                "phydrax": "0.2.2",
                "phydrax-metal": "0.2.2",
                "jax": "0.12.0",
                "jaxlib": "0.12.0",
                "jax-mps": "0.11.0",
            },
            "foreign PJRT plugins",
        ),
        (
            "base-aarch64-apple-darwin",
            {"phydrax": "0.2.2", "phydrax-metal": "0.2.2"},
            "without the extra",
        ),
        (
            "base-x86_64-manylinux_2_28",
            {"phydrax": "0.2.2", "jax-cuda13-plugin": "0.12.0"},
            "foreign",
        ),
    ],
    ids=[
        "no-companion",
        "wrong-jax",
        "foreign-plugin",
        "base-selects-companion",
        "base-selects-cuda",
    ],
)
def test_resolution_refuses_wrong_selected_distributions(
    case: str, resolved: dict[str, str], diagnostic: str
) -> None:
    errors = extras.resolution_errors(
        case, resolved, version="0.2.2", jax_version="0.12.0"
    )
    assert any(diagnostic in error for error in errors), errors


def _resolution_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, core: Path
) -> list[extras.CaseEvidence]:
    companion = _companion_wheel(tmp_path / "built")
    errors, facts = _companion_errors(core, companion)
    assert errors == [] and facts is not None
    work = tmp_path / "work"
    links = {key: work / key for key in ("companion", "absent", "mismatch")}
    for directory in links.values():
        _stub_wheel(directory, "jax", "0.12.0")
        _stub_wheel(directory, "jaxlib", "0.12.0")
    shutil.copy2(companion, links["companion"] / companion.name)
    extras.variant_companion(companion, links["mismatch"], version="0.2.3")
    # The real solver runs offline against the local wheels only.
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    return [
        extras.compile_case(
            name,
            requirement=requirement,
            platform=platform,
            find_links=find_links,
            deployment_target=target,
            expected=expected,
            version="0.2.2",
            jax_version="0.12.0",
            work_dir=work,
        )
        for name, requirement, platform, find_links, target, expected in extras.resolution_matrix(
            core_wheel=core, companion=facts, links=links, linux_tag="manylinux_2_28"
        )
    ]


def test_uv_resolves_the_companion_only_on_supported_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = _resolution_cases(tmp_path, monkeypatch, _core_wheel(tmp_path / "core"))
    assert [case["name"] for case in cases if case["expected"] == "refused"] == [
        "metal-unsupported-x86_64-manylinux_2_28",
        "metal-unsupported-aarch64-manylinux_2_28",
        "metal-unsupported-x86_64-pc-windows-msvc",
        "metal-below-minimum-macos",
        "metal-absent-companion",
        "metal-companion-version-mismatch",
    ]
    failures = [
        (case["name"], case["errors"], case["command"]["stderr"])
        for case in cases
        if not case["successful"]
    ]
    assert failures == []


def test_uv_matrix_catches_a_platform_marked_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = _core_wheel(
        tmp_path / "core",
        requirements=(
            "jax>=0.12.0,<0.13",
            "phydrax-metal==0.2.2; sys_platform == 'darwin' and extra == 'metal'",
        ),
    )
    cases = {
        case["name"]: case for case in _resolution_cases(tmp_path, monkeypatch, core)
    }
    for platform in (
        "x86_64-manylinux_2_28",
        "aarch64-manylinux_2_28",
        "x86_64-pc-windows-msvc",
    ):
        case = cases[f"metal-unsupported-{platform}"]
        assert not case["successful"]
        assert "resolved although it must be refused" in case["errors"][0]
    assert cases["metal-supported"]["successful"]
