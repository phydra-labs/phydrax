#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import pytest

from phydrax._callable import _ensure_special_kwonly_args


def test_ensure_special_kwonly_args_ignores_key_when_not_supported() -> None:
    def f(x: Any) -> Any:
        return x + 1

    wrapped = _ensure_special_kwonly_args(f)
    assert wrapped(1, key="ignored") == 2


def test_ensure_special_kwonly_args_passes_key_when_supported() -> None:
    def f(x: Any, *, key: Any) -> Any:
        return (x, key)

    wrapped = _ensure_special_kwonly_args(f)
    assert wrapped(1, key="k") == (1, "k")


def test_ensure_special_kwonly_args_passes_key_with_var_kwargs() -> None:
    def f(x: Any, **kwargs: Any) -> Any:
        return (x, kwargs.get("key"))

    wrapped = _ensure_special_kwonly_args(f)
    assert wrapped(1, key="k") == (1, "k")


def test_ensure_special_kwonly_args_enforces_kwonly_key() -> None:
    def f(x: Any, key: Any) -> Any:
        return (x, key)

    with pytest.raises(TypeError, match="`key` must be a keyword-only argument"):
        _ensure_special_kwonly_args(f)
