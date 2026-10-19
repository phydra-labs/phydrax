# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Real import actions for the import-refusal harness contract."""

from importlib import import_module


def permitted_import() -> None:
    module = import_module("math")
    assert module.sqrt(4.0) == 2.0


def forbidden_root() -> None:
    import_module("xml")


def forbidden_descendant() -> None:
    import_module("xml.etree.ElementTree")
