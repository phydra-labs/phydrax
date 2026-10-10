#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import importlib.metadata
import shutil
from pathlib import Path
from typing import Any

# ty: ignore[unresolved-import]
import numpy._core._multiarray_umath as numpy_extension
import pytest

import phydrax._meshcore as meshcore
from phydrax._meshcore import load_meshcore, meshcore_available, MeshcoreUnavailableError


def test_shared_library_without_the_meshcore_abi_is_unavailable(monkeypatch: Any) -> None:
    # A loadable shared object that exports none of the meshcore C ABI.
    monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", numpy_extension.__file__)

    assert not meshcore_available()
    with pytest.raises(MeshcoreUnavailableError, match="exports no phx_mc_abi_contract"):
        load_meshcore()


class _Function:
    restype = None
    argtypes = None

    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        return self.value


class _Library:
    def __init__(self, contract: bytes, version: bytes | None) -> None:
        self.functions = {name: _Function(b"0" * 64) for name in meshcore._SIGNATURES}
        self.functions["phx_mc_abi_contract"] = _Function(contract)
        self.functions["phx_mc_version"] = _Function(version)

    def __getitem__(self, name: Any) -> Any:
        return self.functions[name]


def test_null_meshcore_identity_is_reported_as_unavailable() -> None:
    library = _Library(meshcore._ABI_CONTRACT.encode("ascii"), None)

    # ty: ignore[invalid-argument-type]
    unavailable = meshcore._bind(library, Path("malformed-meshcore"), "0" * 64)

    assert isinstance(unavailable, str)
    assert "returned null from phx_mc_version" in unavailable


def test_another_c_abi_contract_is_refused_before_binding() -> None:
    library = _Library(b"f" * 64, b"0.0.0")

    # ty: ignore[invalid-argument-type]
    unavailable = meshcore._bind(library, Path("other-abi-meshcore"), "0" * 64)

    assert isinstance(unavailable, str)
    assert f"implements C ABI contract {'f' * 64}" in unavailable
    assert all(
        function.calls == 0 and function.argtypes is None
        for name, function in library.functions.items()
        if name != "phx_mc_abi_contract"
    )


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
def test_meshcore_of_another_release_is_unavailable(
    monkeypatch: Any, tmp_path: Any
) -> None:
    installed = load_meshcore().path
    library = tmp_path / installed.name
    shutil.copyfile(installed, library)
    distribution_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "99.0.0" if name == "phydrax" else distribution_version(name),
    )
    monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", str(library))

    assert not meshcore_available()
    with pytest.raises(
        MeshcoreUnavailableError, match=r"requires phydrax-meshcore==99\.0\.0"
    ):
        load_meshcore()
