#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""The coupled FE-VEM example programs run to completion.

Each example checks its own claims against independent host references and
raises ``RuntimeError`` when an accepted-step, agreement, or refusal check
fails, so running its ``__main__`` block is the consumer-visible contract.
"""

import contextlib
import io
import runpy

import pytest


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("examples/coupled_inverse_problem.py", id="inverse-problem"),
        pytest.param("examples/coupled_learned_interface.py", id="learned-interface"),
        pytest.param("examples/coupled_rom_swap.py", id="rom-swap"),
    ],
)
def test_coupled_example_passes_its_own_reference_checks(path: str) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(path, run_name="__main__")
