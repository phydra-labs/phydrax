#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Navigation of canonical positional structure recipes for archive tests.

A registered dataclass recipe node stores its children as ``items`` in the
registered class's dataclass field order; there is no name-keyed form. Tests
that deliberately corrupt one named field locate it through that order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from phydrax._model._artifacts import artifact_value


def recipe_field_names(node: Mapping[str, Any], /) -> tuple[str, ...]:
    """Registered field names of one canonical dataclass recipe node, in order."""
    if node.get("kind") != "dataclass":
        raise ValueError("Expected a canonical dataclass recipe node.")
    return tuple(
        field.name for field in dataclasses.fields(artifact_value(node["type"]))
    )


def recipe_field_position(node: Mapping[str, Any], name: str, /) -> int:
    """Position of the registered field ``name`` in the node's ``items``."""
    return recipe_field_names(node).index(name)
