"""Fixed accelerator compiler profile for fresh-process bitwise scenarios.

XLA GPU autotuning times candidate kernels in every process, so two processes
compiling the same program can select different tilings or emitters and round
differently. A scenario asserting bitwise agreement between the test process
and a fresh restart process therefore runs both under one fixed profile with
autotuning disabled. On accelerators the scenario re-executes itself in an
isolated pytest process before any numerical setup; the fresh processes it
spawns inherit the profile through the environment. CPU scenarios run in place.
The profile only fixes compiler choices: the executed programs must already be
free of nondeterministic operators (ordered accumulation).
"""

import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import jax
import pytest


FIXED_COMPILER_FLAG = "--xla_gpu_autotune_level=0"
_ISOLATED_NODE = "PHYDRAX_TEST_FIXED_COMPILER_PROFILE_NODE"
_REPOSITORY = Path(__file__).resolve().parents[2]


def _report(request: pytest.FixtureRequest, scope: str, flags: str) -> None:
    request.node.user_properties.append(("compiler_profile", flags))
    request.node.user_properties.append(("compiler_profile_scope", scope))
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"[compiler profile] {request.node.nodeid}: {scope}, XLA_FLAGS={flags!r}"
        )


def isolated_in_fixed_compiler_profile(request: pytest.FixtureRequest) -> bool:
    """Run the requesting scenario under the fixed compiler profile.

    Returns ``True`` after the scenario passed in an isolated process (the
    caller returns immediately) and ``False`` when its body must run here: on
    CPU, or inside the isolated process itself.
    """

    if jax.default_backend() == "cpu":
        return False
    node = request.node.nodeid
    flags = os.environ.get("XLA_FLAGS", "")
    isolated = os.environ.get(_ISOLATED_NODE)
    if isolated is not None:
        if isolated != node:
            raise RuntimeError(f"{node} cannot nest inside isolated scenario {isolated}.")
        if FIXED_COMPILER_FLAG not in flags.split():
            raise RuntimeError(f"Isolated scenario {node} lacks {FIXED_COMPILER_FLAG}.")
        _report(request, "isolated fixed-profile process", flags)
        return False
    fixed = f"{flags} {FIXED_COMPILER_FLAG}".strip()
    _report(request, "re-executing in an isolated fixed-profile process", fixed)
    report = request.getfixturevalue("tmp_path") / "fixed-compiler-profile.xml"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            f"--junitxml={report}",
            "-p",
            "no:cacheprovider",
            "-vv",
            "-rA",
            "--tb=short",
            node,
        ],
        cwd=_REPOSITORY,
        env={**os.environ, "XLA_FLAGS": fixed, _ISOLATED_NODE: node},
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    suites = tuple(ElementTree.parse(report).getroot().iter("testsuite"))
    outcomes = tuple(
        sum(int(suite.attrib[name]) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    )
    assert outcomes == (1, 0, 0, 0), outcomes
    return True
