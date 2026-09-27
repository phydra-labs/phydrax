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
    with pytest.raises(
        MeshcoreUnavailableError, match="lacks C ABI symbols phx_mc_version"
    ):
        load_meshcore()


def test_null_meshcore_identity_is_reported_as_unavailable() -> None:
    class Function:
        restype = None
        argtypes = None

        def __init__(self, value: Any) -> None:
            self.value = value

        def __call__(self) -> Any:
            return self.value

    class Library:
        def __init__(self) -> None:
            self.functions = {
                name: Function(None if name == "phx_mc_version" else b"0" * 64)
                for name in meshcore._SIGNATURES
            }

        def __getitem__(self, name: Any) -> Any:
            return self.functions[name]

    # ty: ignore[invalid-argument-type]
    unavailable = meshcore._bind(Library(), Path("malformed-meshcore"))

    assert isinstance(unavailable, str)
    assert "returned null from phx_mc_version" in unavailable


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
