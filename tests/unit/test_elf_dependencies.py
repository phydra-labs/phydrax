"""Consumer contracts for the fresh-process ELF dependency model."""

import ctypes
import importlib.metadata
import shutil
import struct
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from phydrax._elf_dependencies import (
    DependencyClosure,
    ElfDynamic,
    read_dynamic_section,
    resolve_dependency_closure,
    ResolvedDependency,
    UnresolvedDependency,
)
from tests._support.elf import write_shared_object


def _resolve(
    root: Path,
    *,
    library_path: tuple[Path, ...] = (),
    executable: Path | None = None,
    preloaded: tuple[tuple[str, str], ...] = (),
    cache: Path | None = None,
) -> DependencyClosure:
    return resolve_dependency_closure(
        root,
        library_path=tuple(str(path) for path in library_path),
        executable=str(executable) if executable is not None else None,
        preloaded=preloaded,
        ld_so_cache=str(cache)
        if cache is not None
        else str(root.parent / "missing.cache"),
    )


def test_numpy_rpath_beats_library_path(tmp_path: Path) -> None:
    numpy_dir = Path(np.__file__).parent
    wheel_libs = numpy_dir.parent / "numpy.libs"
    if not wheel_libs.is_dir():
        pytest.skip("Installed NumPy has no numpy.libs wheel directory")
    root = next(numpy_dir.glob("**/_multiarray_umath*.so"))
    dynamic = read_dynamic_section(root)
    assert "$ORIGIN/../../numpy.libs" in dynamic.rpath
    openblas = next(name for name in dynamic.needed if "openblas" in name.lower())
    decoy = write_shared_object(tmp_path / openblas)
    closure = _resolve(root, library_path=(decoy.parent,))
    dependency = next(item for item in closure.resolved if item.soname == openblas)
    assert Path(dependency.path).resolve().parent == wheel_libs.resolve()


def test_library_path_beats_runpath(tmp_path: Path) -> None:
    preferred = write_shared_object(tmp_path / "environment" / "libchoice.so")
    write_shared_object(tmp_path / "runpath" / preferred.name)
    root = write_shared_object(
        tmp_path / "root.so",
        needed=(preferred.name,),
        runpath=(str(tmp_path / "runpath"),),
    )
    closure = _resolve(root, library_path=(preferred.parent,))
    assert closure.resolved[0].path == str(preferred)


def test_rpath_beats_library_path(tmp_path: Path) -> None:
    preferred = write_shared_object(tmp_path / "rpath" / "libchoice.so")
    decoy = write_shared_object(tmp_path / "environment" / preferred.name)
    root = write_shared_object(
        tmp_path / "root.so", needed=(preferred.name,), rpath=(str(preferred.parent),)
    )
    assert _resolve(root, library_path=(decoy.parent,)).resolved[0].path == str(preferred)


def test_runpath_suppresses_rpath(tmp_path: Path) -> None:
    ignored = write_shared_object(tmp_path / "rpath" / "libchoice.so")
    preferred = write_shared_object(tmp_path / "environment" / ignored.name)
    write_shared_object(tmp_path / "runpath" / ignored.name)
    root = write_shared_object(
        tmp_path / "root.so",
        needed=(preferred.name,),
        rpath=(str(ignored.parent),),
        runpath=(str(tmp_path / "runpath"),),
    )
    assert read_dynamic_section(root).rpath == ()
    assert _resolve(root, library_path=(preferred.parent,)).resolved[0].path == str(
        preferred
    )


def test_empty_runpath_suppresses_rpath_inheritance(tmp_path: Path) -> None:
    ignored = write_shared_object(tmp_path / "rpath" / "libempty-runpath.so")
    root = write_shared_object(
        tmp_path / "root.so",
        needed=(ignored.name,),
        rpath=(str(ignored.parent),),
        runpath=("",),
    )
    assert read_dynamic_section(root).rpath == ()
    assert _resolve(root).unresolved[0].soname == ignored.name


@pytest.mark.parametrize(
    "tag", ["rpath", "runpath"], ids=["rpath-inherited", "runpath-direct-only"]
)
def test_loader_path_inheritance(tmp_path: Path, tag: str) -> None:
    leaf = write_shared_object(tmp_path / "libraries" / "libinherit-leaf.so")
    middle = write_shared_object(
        leaf.parent / "libinherit-middle.so", needed=(leaf.name,)
    )
    root = write_shared_object(
        tmp_path / "root.so",
        needed=(middle.name,),
        rpath=(str(leaf.parent),) if tag == "rpath" else (),
        runpath=(str(leaf.parent),) if tag == "runpath" else (),
    )
    closure = _resolve(root)
    if tag == "rpath":
        assert [item.soname for item in closure.resolved] == [leaf.name, middle.name]
        assert closure.unresolved == ()
    else:
        assert [item.soname for item in closure.resolved] == [middle.name]
        assert closure.unresolved[0].soname == leaf.name
        assert closure.unresolved[0].requested_by == str(middle)


def test_executable_rpath_is_last_in_loader_chain(tmp_path: Path) -> None:
    preferred = write_shared_object(tmp_path / "root-libs" / "libexe-choice.so")
    write_shared_object(tmp_path / "executable-libs" / preferred.name)
    extra = write_shared_object(tmp_path / "executable-libs" / "libexe-only.so")
    executable = write_shared_object(tmp_path / "main", rpath=(str(extra.parent),))
    middle = write_shared_object(
        tmp_path / "middle.so", needed=(preferred.name, extra.name)
    )
    root = write_shared_object(
        tmp_path / "root.so",
        needed=(middle.name,),
        rpath=(str(preferred.parent), str(middle.parent)),
    )
    closure = _resolve(root, executable=executable)
    assert [(item.soname, item.path) for item in closure.resolved] == [
        (preferred.name, str(preferred)),
        (extra.name, str(extra)),
        (middle.name, str(middle)),
    ]


@pytest.mark.parametrize("origin", ["$ORIGIN", "${ORIGIN}"], ids=["bare", "braced"])
def test_origin_expands_relative_to_entry_owner(tmp_path: Path, origin: str) -> None:
    leaf = write_shared_object(tmp_path / "owner" / "libs" / "liborigin.so")
    root = write_shared_object(
        tmp_path / "owner" / "root.so", needed=(leaf.name,), runpath=(f"{origin}/libs",)
    )
    assert _resolve(root).resolved[0].path == str(leaf)


def test_origin_uses_opened_path_not_symlink_target(tmp_path: Path) -> None:
    leaf = write_shared_object(tmp_path / "alias" / "libs" / "liborigin-alias.so")
    target = write_shared_object(
        tmp_path / "target" / "root.so", needed=(leaf.name,), runpath=("$ORIGIN/libs",)
    )
    opened = tmp_path / "alias" / "root.so"
    opened.symlink_to(target)
    assert _resolve(opened).resolved[0].path == str(leaf)


def test_closure_is_dependency_first_but_loads_breadth_first(tmp_path: Path) -> None:
    preferred = write_shared_object(tmp_path / "direct" / "libshared.so")
    write_shared_object(tmp_path / "transitive" / preferred.name)
    middle = write_shared_object(
        tmp_path / "libmiddle.so",
        needed=(preferred.name,),
        rpath=(str(tmp_path / "transitive"),),
    )
    root = write_shared_object(
        tmp_path / "root.so",
        needed=(middle.name, preferred.name),
        runpath=(str(tmp_path), str(preferred.parent)),
    )
    closure = _resolve(root)
    assert [(item.soname, item.path) for item in closure.resolved] == [
        (preferred.name, str(preferred)),
        (middle.name, str(middle)),
    ]
    assert closure.resolved[0].requested_by == str(root)


def test_repeated_requested_soname_resolves_once(tmp_path: Path) -> None:
    leaf = write_shared_object(tmp_path / "libonce.so")
    root = write_shared_object(tmp_path / "root.so", needed=(leaf.name, leaf.name))
    closure = _resolve(root, library_path=(tmp_path,))
    assert len(closure.resolved) == 1
    assert closure.resolved[0].soname == leaf.name


def test_loaded_soname_satisfies_later_request(tmp_path: Path) -> None:
    alias = write_shared_object(tmp_path / "libalias.so", soname="libcanonical.so")
    root = write_shared_object(
        tmp_path / "root.so", needed=(alias.name, "libcanonical.so")
    )
    closure = _resolve(root, library_path=(tmp_path,))
    assert [item.path for item in closure.resolved] == [str(alias)]
    assert closure.unresolved == ()


def test_wrong_machine_candidate_is_skipped(tmp_path: Path) -> None:
    wrong = write_shared_object(tmp_path / "wrong" / "libmachine.so", machine="aarch64")
    right = write_shared_object(tmp_path / "right" / wrong.name)
    root = write_shared_object(tmp_path / "root.so", needed=(right.name,))
    closure = _resolve(root, library_path=(wrong.parent, right.parent))
    assert closure.resolved[0].path == str(right)


def test_unresolved_reports_requester(tmp_path: Path) -> None:
    root = write_shared_object(tmp_path / "root.so", needed=("libphydrax-missing.so",))
    assert _resolve(root).unresolved == (
        UnresolvedDependency(
            soname="libphydrax-missing.so", requested_by=str(root), reason="not found"
        ),
    )


@pytest.mark.parametrize(
    "token",
    ["$LIB", "${LIB}", "$PLATFORM", "${PLATFORM}"],
    ids=["lib", "braced-lib", "platform", "braced-platform"],
)
def test_unmodeled_tokens_report_reason(tmp_path: Path, token: str) -> None:
    root = write_shared_object(
        tmp_path / "root.so",
        needed=("libphydrax-token.so",),
        runpath=(f"$ORIGIN/{token}",),
    )
    assert _resolve(root).unresolved[0].reason == "requires $LIB/$PLATFORM expansion"


def test_unmodeled_entry_does_not_hide_other_search_paths(tmp_path: Path) -> None:
    leaf = write_shared_object(tmp_path / "libtoken-fallback.so")
    root = write_shared_object(
        tmp_path / "root.so", needed=(leaf.name,), rpath=("$ORIGIN/$LIB",)
    )
    assert _resolve(root, library_path=(tmp_path,)).resolved[0].path == str(leaf)


@pytest.mark.parametrize(
    "format_name",
    ["non-ELF", "ELF32", "big-endian", "other-machine", "no-dynamic", "truncated"],
    ids=str,
)
def test_unsupported_elf_inputs_raise_value_error(
    tmp_path: Path, format_name: str
) -> None:
    root = write_shared_object(tmp_path / "invalid.so")
    data = bytearray(root.read_bytes())
    if format_name == "non-ELF":
        data[:4] = b"NOPE"
    elif format_name == "ELF32":
        data[4] = 1
    elif format_name == "big-endian":
        data[5] = 2
    elif format_name == "other-machine":
        struct.pack_into("<H", data, 18, 3)
    elif format_name == "no-dynamic":
        struct.pack_into("<I", data, 64 + 56, 1)
    else:
        data = data[:17]
    root.write_bytes(data)
    with pytest.raises(ValueError, match=str(root)):
        read_dynamic_section(root)


def test_missing_elf_propagates_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_dynamic_section(tmp_path / "absent.so")


def test_preloaded_short_circuits_search_and_expansion(tmp_path: Path) -> None:
    seed = write_shared_object(
        tmp_path / "seed" / "libseed.so", needed=("libseed-unavailable-child.so",)
    )
    decoy = write_shared_object(tmp_path / "search" / seed.name)
    root = write_shared_object(tmp_path / "root.so", needed=(seed.name,))
    closure = _resolve(
        root, library_path=(decoy.parent,), preloaded=((seed.name, str(seed)),)
    )
    assert closure.resolved == (
        ResolvedDependency(
            soname=seed.name, path=str(seed), requested_by=str(root), preloaded=True
        ),
    )
    assert closure.unresolved == ()


def test_direct_needed_path_bypasses_search(tmp_path: Path) -> None:
    leaf = write_shared_object(tmp_path / "direct" / "libdirect.so")
    write_shared_object(tmp_path / "environment" / leaf.name)
    root = write_shared_object(tmp_path / "root.so", needed=(str(leaf),))
    assert _resolve(root, library_path=(tmp_path / "environment",)).resolved[
        0
    ].path == str(leaf)


def test_closure_ignores_live_process_decoy(tmp_path: Path) -> None:
    system_lib = Path("/lib/x86_64-linux-gnu/libm.so.6")
    if not system_lib.is_file():
        pytest.skip("Deterministic system libm decoy is unavailable on this host")
    preferred = tmp_path / "preferred" / "libprocess-decoy.so"
    decoy = tmp_path / "decoy" / preferred.name
    preferred.parent.mkdir()
    decoy.parent.mkdir()
    shutil.copyfile(system_lib, preferred)
    shutil.copyfile(system_lib, decoy)
    root = write_shared_object(tmp_path / "root.so", needed=(preferred.name,))
    before = _resolve(root, library_path=(preferred.parent,))
    loaded_decoy = ctypes.CDLL(str(decoy))
    after = _resolve(root, library_path=(preferred.parent,))
    assert loaded_decoy is not None
    assert before == after
    assert next(
        item for item in after.resolved if item.soname == preferred.name
    ).path == str(preferred)


def _write_cache(
    path: Path, rows: tuple[tuple[str, str, int, int], ...], *, old_prefix: bool
) -> Path:
    # glibc sysdeps/generic/dl-cache.h: cache_file_new (48 bytes), file_entry_new (24).
    strings = bytearray()
    entries = bytearray()
    string_offset = 48 + 24 * len(rows)
    for soname, filename, flags, hwcap in rows:
        key = string_offset + len(strings)
        strings.extend(soname.encode() + b"\0")
        value = string_offset + len(strings)
        strings.extend(filename.encode() + b"\0")
        entries.extend(struct.pack("<iIIIQ", flags, key, value, 0, hwcap))
    header = struct.pack(
        "<20sII5I", b"glibc-ld.so.cache1.1", len(rows), len(strings), 0, 0, 0, 0, 0
    )
    # The old empty cache header is padded to the new header's 8-byte alignment.
    prefix = struct.pack("<12sI", b"ld.so-1.7.0", 0) if old_prefix else b""
    path.write_bytes(prefix + header + entries + strings)
    return path


@pytest.mark.parametrize(
    "machine,arch,foreign",
    [("x86_64", 0x0303, 0x0A03), ("aarch64", 0x0A03, 0x0303)],
    ids=["x86-64", "aarch64"],
)
@pytest.mark.parametrize("old_prefix", [False, True], ids=["new-cache", "compat-cache"])
def test_ld_so_cache_filters_architecture_and_hwcaps(
    tmp_path: Path,
    machine: Literal["x86_64", "aarch64"],
    arch: int,
    foreign: int,
    old_prefix: bool,
) -> None:
    correct = write_shared_object(
        tmp_path / "correct" / "libcache-only.so", machine=machine
    )
    wrong_arch = write_shared_object(tmp_path / "foreign" / correct.name, machine=machine)
    hwcap = write_shared_object(tmp_path / "hwcap" / correct.name, machine=machine)
    root = write_shared_object(
        tmp_path / "root.so", needed=(correct.name,), machine=machine
    )
    cache = _write_cache(
        tmp_path / "ld.so.cache",
        (
            (correct.name, str(wrong_arch), foreign, 0),
            (correct.name, str(hwcap), arch, 1),
            (correct.name, str(correct), arch, 0),
        ),
        old_prefix=old_prefix,
    )
    assert _resolve(root, cache=cache).resolved[0].path == str(correct)


def test_cuda_wheel_dependencies_resolve_inside_wheels() -> None:
    installed = {
        distribution.metadata["Name"].lower(): distribution
        for distribution in importlib.metadata.distributions()
    }
    if "jax-cuda13-plugin" not in installed:
        pytest.skip("CUDA 13 JAX plugin wheels are not installed")
    if "nvidia-cublas" not in installed:
        pytest.skip("CUDA 13 cuBLAS wheel is not installed")
    cublas = installed["nvidia-cublas"]
    roots = [
        Path(str(cublas.locate_file(file)))
        for file in cublas.files or ()
        if file.as_posix().endswith("nvidia/cu13/lib/libcublas.so.13")
    ]
    assert len(roots) == 1, "the installed cuBLAS wheel lacks libcublas.so.13"
    (root,) = roots
    assert root.is_file(), f"the installed cuBLAS wheel lacks {root}"
    nvidia = root.parents[2]
    directories = tuple(sorted(nvidia.glob("*/lib")))
    closure = resolve_dependency_closure(
        root, library_path=tuple(str(path) for path in directories), executable=None
    )
    nvidia_needed = [
        name
        for name in read_dynamic_section(root).needed
        if name.startswith(("libcublas", "libcudart", "libnv"))
    ]
    assert nvidia_needed
    resolved = {item.soname: Path(item.path) for item in closure.resolved}
    assert all(resolved[name].is_relative_to(nvidia) for name in nvidia_needed)


def test_dynamic_constructor_validates_selector(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="machine"):
        ElfDynamic(
            path=str(tmp_path / "root.so"),
            machine="mips",  # ty: ignore[invalid-argument-type]
            soname=None,
            needed=(),
            rpath=(),
            runpath=(),
        )


def test_resolved_constructor_refuses_wrong_boolean_kind() -> None:
    with pytest.raises(TypeError, match="preloaded"):
        ResolvedDependency(
            soname="lib.so",
            path="/lib.so",
            requested_by="/root.so",
            preloaded=1,  # ty: ignore[invalid-argument-type]
        )


def test_unresolved_constructor_refuses_empty_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        UnresolvedDependency(soname="lib.so", requested_by="/root.so", reason="")


def test_closure_constructor_refuses_wrong_entry_kind() -> None:
    with pytest.raises(TypeError, match="resolved"):
        DependencyClosure(
            root="/root.so",
            machine="x86_64",
            resolved=("wrong",),  # ty: ignore[invalid-argument-type]
            unresolved=(),
        )
