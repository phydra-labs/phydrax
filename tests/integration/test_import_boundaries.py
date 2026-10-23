from pathlib import Path

import pytest

from tools.check_import_boundaries import (
    configuration_bypass_errors,
    import_boundary_errors,
)


_ROOT = Path(__file__).resolve().parents[2]
# Assembled so this file itself never spells an upstream configuration access.
_UPSTREAM = "jax"


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_public_import_boundaries_are_canonical() -> None:
    assert import_boundary_errors(_ROOT) == ()


def test_configuration_bypass_audit_reports_embedded_scripts_and_docs(
    tmp_path: Path,
) -> None:
    upstream = _UPSTREAM
    _write(
        tmp_path,
        "phydrax/module.py",
        'script = """\n'
        + f'{upstream}.config.update("jax_enable_x64", True)\n'
        + '"""\n',
    )
    _write(
        tmp_path,
        "phydrax/_jax_config.py",
        f'{upstream}.config.update("jax_enable_x64", True)\n',
    )
    _write(
        tmp_path,
        "docs/guide.md",
        f"# Guide\n\n```python\nwith {upstream}.enable_x64(True):\n    pass\n```\n",
    )
    _write(
        tmp_path,
        "docs/plans/historical.md",
        f"```python\nwith {upstream}.enable_x64(True):\n    pass\n```\n",
    )

    assert configuration_bypass_errors(tmp_path) == (
        f"configuration-bypass:docs/guide.md:4:{upstream}.enable_x64",
        f"configuration-bypass:phydrax/module.py:2:{upstream}.config",
    )


@pytest.mark.parametrize(
    ("relative", "text", "line", "capability"),
    [
        (
            "tools/worker.py",
            'script = """\nimport {jax} as upstream\n'
            'upstream.config.update("jax_enable_x64", False)\n"""\n',
            3,
            "config",
        ),
        (
            "tests/test_child.py",
            "import subprocess\n"
            'code = "from {jax} import numpy as jnp, " + "config as c\\n"\n'
            "subprocess.run(['python', '-c', code], check=True)\n",
            2,
            "config",
        ),
        (
            "docs/guide.md",
            "# Guide\n\n- Step:\n\n  ```python executable\n  import {jax} as j\n"
            "  sink = j\n  with sink.default_device(None):\n      pass\n  ```\n",
            8,
            "default_device",
        ),
        (
            "phydrax/module.py",
            "from {jax} import enable_x64 as wide\n",
            1,
            "enable_x64",
        ),
        (
            "benchmarks/private.py",
            "from {jax}._src import config\n",
            1,
            "_src.config",
        ),
        (
            "examples/dynamic.py",
            "import {jax}\nselected = getattr({jax}, 'transfer_guard')\n",
            2,
            "transfer_guard",
        ),
    ],
    ids=[
        "aliased-embedded-script",
        "static-concatenation",
        "indented-executable-fence-alias",
        "direct-capability-import",
        "private-configuration-module",
        "constant-getattr",
    ],
)
def test_configuration_bypass_audit_resolves_aliases(
    tmp_path: Path, relative: str, text: str, line: int, capability: str
) -> None:
    _write(tmp_path, relative, text.format(jax=_UPSTREAM))

    assert configuration_bypass_errors(tmp_path) == (
        f"configuration-bypass:{relative}:{line}:{_UPSTREAM}.{capability}",
    )


@pytest.mark.parametrize(
    "text",
    [
        "class Store:\n    config = None\n\nupstream = Store()\nupstream.config = dict()\n",
        "import {jax}\n\n\ndef apply({jax}: object) -> object:\n    return {jax}.config\n",
        'message = "{jax} is a library; enable it explicitly"\n',
    ],
    ids=["unrelated-config-attribute", "shadowed-parameter", "prose-string"],
)
def test_configuration_bypass_audit_ignores_non_upstream_names(
    tmp_path: Path, text: str
) -> None:
    _write(tmp_path, "phydrax/module.py", text.format(jax=_UPSTREAM))

    assert configuration_bypass_errors(tmp_path) == ()


@pytest.mark.parametrize(
    ("relative", "function", "statement", "count"),
    [
        (
            "tests/unit/test_jax_indexing.py",
            "test_explicit_registration_is_device_inert_and_preserves_jax_surface",
            "options = dict({upstream}.config.values)",
            1,
        ),
        (
            "tests/unit/test_jax_indexing.py",
            "test_explicit_registration_is_device_inert_and_preserves_jax_surface",
            "assert {upstream}.config.values == options",
            2,
        ),
        (
            "tests/unit/test_jax_indexing_frontend.py",
            "test_installation_is_host_only_locked_idempotent_and_honestly_identified",
            "assert {upstream}.config.values == options",
            1,
        ),
        (
            "tests/unit/test_jax_indexing_frontend.py",
            "test_device_indices_do_not_require_host_synchronization",
            'with {upstream}.transfer_guard_device_to_host("disallow"):\n    pass',
            1,
        ),
    ],
)
def test_configuration_audit_allows_only_declared_isolated_probe_statements(
    tmp_path: Path, relative: str, function: str, statement: str, count: int
) -> None:
    source = statement.format(upstream=_UPSTREAM) + "\n"
    _write(
        tmp_path, relative, f'def {function}():\n    script = """\n{source * count}"""\n'
    )
    assert configuration_bypass_errors(tmp_path) == ()

    # An extra occurrence in the same probe is not covered by its declaration.
    _write(
        tmp_path,
        relative,
        f'def {function}():\n    script = """\n{source * (count + 1)}"""\n',
    )
    assert len(configuration_bypass_errors(tmp_path)) == count + 1

    # Neither another function in the same file nor another file inherits policy.
    _write(tmp_path, relative, f'def test_unreviewed():\n    script = """\n{source}"""\n')
    assert len(configuration_bypass_errors(tmp_path)) == 1
    _write(tmp_path, relative, "")
    _write(
        tmp_path,
        "tests/unit/test_unreviewed.py",
        f'def {function}():\n    script = """\n{source}"""\n',
    )
    assert len(configuration_bypass_errors(tmp_path)) == 1


def test_isolated_probe_policy_does_not_allow_upstream_configuration_mutation(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "tests/unit/test_jax_indexing.py",
        "def test_explicit_registration_is_device_inert_and_preserves_jax_surface():\n"
        '    script = """\n'
        f'{_UPSTREAM}.config.update("jax_enable_x64", False)\n'
        '"""\n',
    )
    assert configuration_bypass_errors(tmp_path) == (
        f"configuration-bypass:tests/unit/test_jax_indexing.py:3:{_UPSTREAM}.config",
    )
