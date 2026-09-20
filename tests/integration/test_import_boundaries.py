from pathlib import Path

from tools.check_import_boundaries import import_boundary_errors


_ROOT = Path(__file__).resolve().parents[2]


def test_public_import_boundaries_are_canonical():
    assert import_boundary_errors(_ROOT) == ()
