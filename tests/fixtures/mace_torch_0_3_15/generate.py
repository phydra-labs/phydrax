"""Generate tiny mace-torch 0.3.15 source-provider reference fixtures.

Run with an environment providing mace-torch 0.3.15 and e3nn 0.4.4 (and the
pinned LES 0.2.0 for the MACELES case; none is a Phydrax dependency), for
example::

    .tmp/mace-provider-0.3.15/.venv/bin/python tests/fixtures/mace_torch_0_3_15/generate.py

The cases exercise what differs in the 0.3.15 release: the residual-nonlinear
block gates through e3nn's ``nn.Gate`` (0.3.16 uses mace-torch's own
``GatedEquivariantBlock``), and the release has no ``keep_last_layer_irreps``.
Generation is shared with ``tests/fixtures/mace_torch_0_3_16/generate.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_DIRECTORY = Path(__file__).resolve().parent
_SHARED = _DIRECTORY.parent / "mace_torch_0_3_16" / "generate.py"
# Release cases: the 0.3.16 family cases that 0.3.15 can construct.
_RELEASE_CASES = (
    "two_attention",
    "two_nonlinear_bias",
    "two_gaussian_soft",
    "two_so3",
    "two_joint_embedding",
    "two_maceles",
)


def main() -> None:
    specification = importlib.util.spec_from_file_location("mace_fixtures", _SHARED)
    if specification is None or specification.loader is None:
        raise SystemExit(f"Cannot load the shared fixture generator {_SHARED}.")
    shared = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(shared)
    cases = {name: shared.FAMILY_CASES[name] for name in _RELEASE_CASES}
    shared.generate(_DIRECTORY, cases, "0.3.15")


if __name__ == "__main__":
    main()
