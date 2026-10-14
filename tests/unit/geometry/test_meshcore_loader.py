#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import importlib.metadata
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ty: ignore[unresolved-import]
import numpy._core._multiarray_umath as numpy_extension
import pytest

import phydrax._meshcore as meshcore
from phydrax import _config_state, config as pxconfig
from phydrax._meshcore import load_meshcore, meshcore_available, MeshcoreUnavailableError


@pytest.fixture(autouse=True)
def restore_meshcore_setting() -> Iterator[None]:
    """Own meshcore test overrides without changing other provider settings."""
    previous = _config_state._STATE.libraries.copy()
    _config_state._STATE.libraries.pop("meshcore", None)
    try:
        yield
    finally:
        _config_state._STATE.libraries.clear()
        _config_state._STATE.libraries.update(previous)


@pytest.mark.parametrize("blank", ("", " \t "))
def test_blank_meshcore_path_keeps_the_absent_path_fallback(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    monkeypatch.delenv("PHYDRAX_MESHCORE_LIBRARY", raising=False)
    if meshcore_available():
        fallback = load_meshcore()
        monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", blank)
        assert load_meshcore() is fallback
    else:
        with pytest.raises(MeshcoreUnavailableError) as absent:
            load_meshcore()
        monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", blank)
        with pytest.raises(MeshcoreUnavailableError) as empty:
            load_meshcore()
        assert str(empty.value) == str(absent.value)


@pytest.mark.parametrize("layer", ("environment", "configured"))
def test_missing_meshcore_path_refuses_packaged_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, layer: str
) -> None:
    path = str(tmp_path / "missing-meshcore")
    if layer == "environment":
        monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", path)
    else:
        pxconfig.configure(pxconfig.ProviderConfig(library_paths={"meshcore": path}))
    assert not meshcore_available()
    with pytest.raises(
        MeshcoreUnavailableError, match="PHYDRAX_MESHCORE_LIBRARY names a missing"
    ):
        load_meshcore()


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
