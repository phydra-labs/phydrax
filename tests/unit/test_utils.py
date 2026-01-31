#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax._callable import _ensure_special_kwonly_args


def test_ensure_special_kwonly_args_ignores_key_when_not_supported():
    def f(x):
        return x + 1

    wrapped = _ensure_special_kwonly_args(f)
    assert wrapped(1, key="ignored") == 2


def test_ensure_special_kwonly_args_passes_key_when_supported():
    def f(x, *, key):
        return (x, key)

    wrapped = _ensure_special_kwonly_args(f)
    assert wrapped(1, key="k") == (1, "k")


def test_ensure_special_kwonly_args_passes_key_with_var_kwargs():
    def f(x, **kwargs):
        return (x, kwargs.get("key"))

    wrapped = _ensure_special_kwonly_args(f)
    assert wrapped(1, key="k") == (1, "k")


def test_ensure_special_kwonly_args_enforces_kwonly_key():
    def f(x, key):
        return (x, key)

    with pytest.raises(TypeError):
        _ensure_special_kwonly_args(f)
