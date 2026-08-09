import inspect

import phydrax as phx


def test_special_namespace_is_public():
    assert "special" in phx.__all__
    assert phx.special.__all__ == [
        "airy",
        "airye",
        "dawsn",
        "ellipam",
        "ellipe",
        "ellipeinc",
        "ellipj",
        "ellipk",
        "ellipkinc",
        "ellipkm1",
        "ellippi",
        "ellippiinc",
        "elliprc",
        "elliprd",
        "elliprf",
        "elliprg",
        "elliprj",
        "hankel1",
        "hankel2",
        "iv",
        "ive",
        "jv",
        "kv",
        "kve",
        "voigt_profile",
        "wofz",
        "yv",
    ]


def test_special_functions_are_importable_from_namespace():
    assert phx.special.dawsn.__name__ == "dawsn"
    assert phx.special.voigt_profile.__name__ == "voigt_profile"
    assert tuple(inspect.signature(phx.special.ellippi).parameters) == ("n", "m")
    assert tuple(inspect.signature(phx.special.ellippiinc).parameters) == (
        "n",
        "phi",
        "m",
    )
    assert phx.special.wofz.__name__ == "wofz"
    expected = {
        "airy",
        "airye",
        "ellipam",
        "ellipe",
        "ellipeinc",
        "ellipj",
        "ellipk",
        "ellipkinc",
        "ellipkm1",
        "ellippi",
        "ellippiinc",
        "elliprc",
        "elliprd",
        "elliprf",
        "elliprg",
        "elliprj",
        "hankel1",
        "hankel2",
        "iv",
        "ive",
        "jv",
        "kv",
        "kve",
        "yv",
    }
    assert {
        name for name in expected if getattr(phx.special, name).__name__ == name
    } == expected
