"""Archive namespace safety and durable source-build failure evidence."""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NoReturn

import pytest

from tools import build_cuda_pjrt_wheels as build


type Entry = tuple[str, Literal["file", "symlink", "hardlink", "directory"], bytes | str]


def _tar(path: Path, entries: tuple[Entry, ...]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, kind, payload in entries:
            info = tarfile.TarInfo(name)
            match kind:
                case "file":
                    if not isinstance(payload, bytes):
                        raise TypeError("File fixture requires bytes")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                case "symlink" | "hardlink":
                    if not isinstance(payload, str):
                        raise TypeError("Link fixture requires text")
                    info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                    info.linkname = payload
                    archive.addfile(info)
                case "directory":
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
    return path


@pytest.mark.parametrize(
    "compress",
    [gzip.compress, bz2.compress, lzma.compress],
    ids=["gzip", "bzip2", "xz"],
)
def test_compressed_out_of_order_members_preserve_contents_and_link_evidence(
    tmp_path: Path, compress: Callable[[bytes], bytes]
) -> None:
    archive = _tar(
        tmp_path / "source.tar",
        (
            ("package/z-last", "file", b"z" * 8192),
            ("package/a-first", "file", b"a" * 4096),
            ("package/alias", "symlink", "z-last"),
        ),
    )
    compressed = tmp_path / "compressed.tar"
    compressed.write_bytes(compress(archive.read_bytes()))
    reference = tmp_path / "reference"
    expected_links = build.extract(archive, reference)
    target = tmp_path / "target"
    links = build.extract(compressed, target)
    assert links == expected_links
    for name in ("a-first", "z-last", "alias"):
        assert (target / "package" / name).read_bytes() == (
            reference / "package" / name
        ).read_bytes()
    assert {path.name for path in tmp_path.iterdir()} == {
        "source.tar",
        "compressed.tar",
        "reference",
        "target",
    }


@pytest.mark.parametrize(
    "compress",
    [gzip.compress, bz2.compress, lzma.compress],
    ids=["gzip", "bzip2", "xz"],
)
def test_compressed_refusal_removes_temporary_input_and_partial_tree(
    tmp_path: Path, compress: Callable[[bytes], bytes]
) -> None:
    sentinel = tmp_path / "outside"
    sentinel.write_bytes(b"untouched")
    archive = _tar(
        tmp_path / "source.tar",
        (
            ("package/good", "file", b"owned"),
            ("package/alias", "symlink", "../../outside"),
        ),
    )
    compressed = tmp_path / "compressed.tar"
    compressed.write_bytes(compress(archive.read_bytes()))
    with pytest.raises(ValueError):
        build.extract(compressed, tmp_path / "target")
    assert sentinel.read_bytes() == b"untouched"
    assert {path.name for path in tmp_path.iterdir()} == {
        "source.tar",
        "compressed.tar",
        "outside",
    }


def test_parent_relative_symlink_and_archive_root_hardlink_materialize(
    tmp_path: Path,
) -> None:
    archive = _tar(
        tmp_path / "source.tar",
        (
            ("rapids-logger/tests/VERSION", "file", b"0.2.3\n"),
            (
                "rapids-logger/tests/template/generate_logger_macros/VERSION",
                "symlink",
                "../../VERSION",
            ),
            ("rapids-logger/hard-version", "hardlink", "rapids-logger/tests/VERSION"),
            ("rapids-logger/chained-version", "symlink", "hard-version"),
        ),
    )
    target = tmp_path / "extracted"
    links = build.extract(archive, target)
    expected = b"0.2.3\n"
    assert (
        target / "rapids-logger/tests/template/generate_logger_macros/VERSION"
    ).read_bytes() == expected
    assert (target / "rapids-logger/hard-version").read_bytes() == expected
    assert (target / "rapids-logger/chained-version").read_bytes() == expected
    assert not (target / "rapids-logger/chained-version").is_symlink()
    assert {item["representation"] for item in links} == {"materialized-file"}
    assert {item["resolved_target"] for item in links} == {"rapids-logger/tests/VERSION"}
    assert len(links) == 3


def test_directory_alias_and_link_through_alias_use_archive_contents(
    tmp_path: Path,
) -> None:
    archive = _tar(
        tmp_path / "source.tar",
        (
            ("package/data/value", "file", b"owned archive bytes"),
            ("package/alias", "symlink", "data"),
            ("package/through-alias", "symlink", "alias/value"),
        ),
    )
    target = tmp_path / "extracted"
    build.extract(archive, target)
    assert (target / "package/alias/value").read_bytes() == b"owned archive bytes"
    assert (target / "package/through-alias").read_bytes() == b"owned archive bytes"
    assert (target / "package/data/value").read_bytes() == b"owned archive bytes"


@pytest.mark.parametrize(
    "link", ["../../outside", "/outside", "C:/outside", "..\\outside"]
)
def test_escaping_link_is_not_resolved_on_host(tmp_path: Path, link: str) -> None:
    sentinel = tmp_path / "outside"
    sentinel.write_bytes(b"host sentinel")
    archive = _tar(
        tmp_path / "source.tar",
        (
            ("package/good", "file", b"good"),
            ("package/alias", "symlink", link),
        ),
    )
    target = tmp_path / "extracted"
    with pytest.raises(ValueError, match="archive|Archive"):
        build.extract(archive, target)
    assert sentinel.read_bytes() == b"host sentinel"
    assert not target.exists()
    assert not list(tmp_path.glob(".extracted.extract-*"))


@pytest.mark.parametrize(
    "entries",
    [
        (("package/a", "symlink", "b"), ("package/b", "symlink", "a")),
        (("package/a", "hardlink", "package/a"),),
        (("package/data/value", "file", b"value"), ("package/data/loop", "symlink", ".")),
    ],
)
def test_link_cycles_refuse_without_publishing_partial_tree(
    tmp_path: Path, entries: tuple[Entry, ...]
) -> None:
    archive = _tar(tmp_path / "source.tar", entries)
    target = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Cyclic"):
        build.extract(archive, target)
    assert not target.exists()


def test_missing_symlink_preserves_link_text_without_reading_host_file(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "missing"
    sentinel.write_bytes(b"not an archive member")
    archive = _tar(tmp_path / "source.tar", (("alias", "symlink", "./missing"),))
    target = tmp_path / "extracted"
    links = build.extract(archive, target)
    assert (target / "alias").read_bytes() == b"./missing"
    assert not (target / "missing").exists()
    assert sentinel.read_bytes() == b"not an archive member"
    assert links == [
        {
            "path": "alias",
            "archive_member": "alias",
            "kind": "symlink",
            "link_target": "./missing",
            "link_target_sha256": hashlib.sha256(b"./missing").hexdigest(),
            "resolved_target": "missing",
            "representation": "git-core.symlinks=false",
        }
    ]


def test_upstream_dangling_symlink_and_alias_preserve_exact_link_text(
    tmp_path: Path,
) -> None:
    member = "rapids-logger/tests/template/generate_logger_macros/VERSION"
    archive = _tar(
        tmp_path / "source.tar",
        (
            (member, "symlink", "../../VERSION"),
            ("rapids-logger/copied-template", "symlink", "tests/template"),
            (
                "rapids-logger/chain",
                "symlink",
                "tests/template/generate_logger_macros/VERSION",
            ),
        ),
    )
    target = tmp_path / "extracted"
    links = build.extract(archive, target)
    for relative in (
        member,
        "rapids-logger/copied-template/generate_logger_macros/VERSION",
    ):
        assert (target / relative).read_bytes() == b"../../VERSION"
    assert (target / "rapids-logger/chain").read_bytes() == (
        b"tests/template/generate_logger_macros/VERSION"
    )
    assert not (target / "rapids-logger/tests/VERSION").exists()
    assert len(links) == 4
    assert all(
        link["resolved_target"] == "rapids-logger/tests/VERSION"
        for link in links
        if link["representation"] == "git-core.symlinks=false"
    )


@pytest.mark.parametrize(
    "entries",
    [
        (("alias", "hardlink", "missing"),),
        (("alias", "hardlink", "dangling"), ("dangling", "symlink", "missing")),
        (("alias", "symlink", "hard"), ("hard", "hardlink", "missing")),
    ],
)
def test_dangling_hardlinks_refuse_atomically(
    tmp_path: Path, entries: tuple[Entry, ...]
) -> None:
    archive = _tar(tmp_path / "source.tar", entries)
    target = tmp_path / "extracted"
    with pytest.raises(ValueError, match="hardlink target is missing"):
        build.extract(archive, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".extracted.extract-*"))


def test_existing_partial_destination_is_not_adopted_or_overwritten(
    tmp_path: Path,
) -> None:
    target = tmp_path / "extracted"
    target.mkdir()
    (target / "sentinel").write_bytes(b"retain old attempt")
    archive = _tar(tmp_path / "source.tar", (("value", "file", b"new attempt"),))
    with pytest.raises(FileExistsError, match="already exists"):
        build.extract(archive, target)
    assert (target / "sentinel").read_bytes() == b"retain old attempt"
    assert not (target / "value").exists()


def test_case_ambiguous_implicit_directories_refuse(tmp_path: Path) -> None:
    archive = _tar(
        tmp_path / "source.tar",
        (
            ("package/Data/a", "file", b"a"),
            ("package/data/b", "file", b"b"),
        ),
    )
    with pytest.raises(ValueError, match="Case-ambiguous"):
        build.extract(archive, tmp_path / "extracted")


def _builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> build.Builder:
    monkeypatch.setattr(build, "owned", lambda value: Path(value).resolve())
    # Keep the real Windows system root: OpenSSL uses it even when urllib
    # constructs its default HTTPS handler for a local file:// download.
    if "SYSTEMROOT" not in os.environ:
        monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    git = tmp_path / "git.exe"
    git.write_bytes(b"not invoked by this fixture")
    args = argparse.Namespace(
        work_root=str(tmp_path / "work"), git=str(git), python=sys.executable, timeout=1
    )
    return build.Builder(args, {})


def test_windows_rc_preserves_recursive_configuration_and_setting_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _builder(tmp_path, monkeypatch)
    source = tmp_path / "jax"
    source.mkdir()
    builder.sources["jax"] = source
    global_setting = "build --define=selection=global"
    windows_setting = "build:windows --define=selection=windows"
    (source / ".bazelrc").write_text(
        "\n".join(
            [
                global_setting,
                "build:linux --define=selection=unreachable",
                "build:windows --config=common_gpu",
                "build:common_gpu --config nested",
                "build:nested --define=selection=nested",
                windows_setting,
                "startup --host_jvm_args=-Dscope=global",
                "try-import %workspace%/absent-local.rc",
            ]
        ),
        encoding="utf-8",
    )
    toolchain = tmp_path / "toolchain.rc"
    final_setting = "build:winjax_cuda --define=selection=toolchain"
    toolchain.write_text(final_setting + "\n", encoding="utf-8")
    selected = builder.windows_bazelrc(toolchain)
    assert selected.read_text().splitlines() == [
        global_setting,
        "build:windows --config=common_gpu",
        "build:common_gpu --config nested",
        "build:nested --define=selection=nested",
        windows_setting,
        "startup --host_jvm_args=-Dscope=global",
        final_setting,
    ]
    evidence = json.loads((builder.build / "bazelrc.json").read_text())
    assert evidence["selected_configurations"] == [
        "common_gpu",
        "nested",
        "windows",
        "winjax_cuda",
    ]
    assert evidence["upstream_sha256"] == build.digest(source / ".bazelrc")
    assert evidence["selected_sha256"] == build.digest(selected)


@pytest.mark.parametrize(
    "statement",
    [
        "try-import %workspace%/local.rc",
        "try-import %workspace%/../outside.rc",
        "import %workspace%/local.rc",
        "build:windows --config",
    ],
    ids=[
        "present-override",
        "escaped-override",
        "required-import",
        "missing-config-name",
    ],
)
def test_windows_rc_refuses_unadmitted_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str
) -> None:
    builder = _builder(tmp_path, monkeypatch)
    source = tmp_path / "jax"
    source.mkdir()
    builder.sources["jax"] = source
    (source / ".bazelrc").write_text(statement + "\n", encoding="utf-8")
    (source / "local.rc").write_text("build --define=foreign=true\n", encoding="utf-8")
    toolchain = tmp_path / "toolchain.rc"
    toolchain.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        builder.windows_bazelrc(toolchain)
    assert not (builder.build / "windows.bazelrc").exists()


def test_repeated_work_root_reuses_only_download_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _builder(tmp_path, monkeypatch)
    first.cache.mkdir()
    (first.cache / "authenticated-input").write_bytes(b"cached bytes")
    (first.build / "partial-source").write_bytes(b"old partial")
    second = _builder(tmp_path, monkeypatch)
    assert first.cache == second.cache
    assert first.build != second.build
    assert (first.build / "partial-source").read_bytes() == b"old partial"
    assert not (second.build / "partial-source").exists()
    assert (second.cache / "authenticated-input").read_bytes() == b"cached bytes"


@pytest.mark.parametrize("phase", ["acquisition", "patching"])
def test_failed_source_phase_retains_original_failure_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    builder = _builder(tmp_path, monkeypatch)
    original = ValueError("archive link refused")

    def fail() -> NoReturn:
        raise original

    monkeypatch.setattr(
        builder, "acquire_sources", fail if phase == "acquisition" else lambda: tmp_path
    )
    monkeypatch.setattr(builder, "patch_sources", fail)
    with pytest.raises(ValueError) as result:
        builder.prepare_sources()
    assert result.value is original
    record = json.loads((builder.build / "sources.json").read_text())
    assert record["status"] == "failed"
    assert record["phase"] == phase
    assert record["error_type"] == "ValueError" and record["error"] == str(original)
    assert record["sources_lock_sha256"] == build.digest(
        build.PACKAGE / "sources.lock.json"
    )


def test_extraction_identity_survives_source_evidence_and_build_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _builder(tmp_path, monkeypatch)
    module = b'module(name = "fixture")\n'
    entries = []
    for name in ("jax", "xla", "logger", "winjax"):
        prefix = "winjax-revision" if name == "winjax" else name
        archive = _tar(
            tmp_path / f"{name}.tar",
            (
                (f"{prefix}/MODULE.bazel", "file", module),
                (f"{prefix}/MODULE.bazel.lock", "file", b"{}\n"),
                (f"{prefix}/nested/link", "symlink", "../missing"),
            ),
        )
        entries.append(
            {
                "name": name,
                "url": archive.as_uri(),
                "sha256": build.digest(archive),
                "strip_prefix": prefix,
            }
        )
    builder.lock = {
        "sources": entries[:-1],
        "port_lineage": {**entries[-1], "revision": "revision"},
        "upstream_module_graph": {
            f"{name}_module_sha256": hashlib.sha256(module).hexdigest()
            for name in ("jax", "xla")
        },
        "patches": [],
        "toolchain": {},
        "capabilities": [],
        "unsupported_capabilities": [],
        "license_sources": [],
    }
    # This fixture exercises actual download authentication and extraction;
    # patch application has its own source-only integration gate.
    monkeypatch.setattr(builder, "patch_sources", lambda: None)
    builder.prepare_sources()
    # No patch/build subprocesses run in this fixture; retain an honest empty
    # command record for the manifest's native-build evidence digest.
    build.write_json_evidence(
        builder.build / "commands.json", {"commands": builder.commands}
    )
    evidence = json.loads((builder.build / "sources.json").read_text())
    assert evidence["status"] == "completed"
    assert set(evidence["source_extractions"]) == {"jax", "xla", "logger", "winjax"}
    for name, extraction in evidence["source_extractions"].items():
        expected = next(entry for entry in entries if entry["name"] == name)
        assert extraction["archive_sha256"] == expected["sha256"]
        assert extraction["links"][0]["representation"] == "git-core.symlinks=false"
        assert extraction["links"][0]["link_target"] == "../missing"
    staging = tmp_path / "staging"
    (staging / "phydrax_cuda13_pjrt").mkdir(parents=True)
    wheel = tmp_path / "kernels.whl"
    wheel.write_bytes(b"manifest identity fixture")
    config = {}
    for name in ("msvc", "windows_sdk", "llvm", "msys"):
        root = tmp_path / name
        root.mkdir()
        config[name] = str(root)
    builder.write_manifest(staging, wheel, config)
    manifest = json.loads((staging / "phydrax_cuda13_pjrt/manifest.json").read_bytes())
    assert manifest["source_extractions"] == evidence["source_extractions"]
    identity = manifest.pop("build_id")
    assert identity == hashlib.sha256(build.canonical(manifest)).hexdigest()
    manifest["source_extractions"]["logger"]["links"][0]["link_target"] = "different"
    assert identity != hashlib.sha256(build.canonical(manifest)).hexdigest()
