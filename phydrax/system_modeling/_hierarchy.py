#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Component:
    name: str
    children: tuple["Component", ...] = ()

    def __post_init__(self):
        if not self.name or len({x.name for x in self.children}) != len(self.children):
            raise ValueError("Component hierarchy invalid.")


def flatten_component(component: Component, prefix: str = ""):
    path = f"{prefix}.{component.name}" if prefix else component.name
    result = [path]
    for child in component.children:
        result.extend(flatten_component(child, path))
    return tuple(result)


__all__ = ["Component", "flatten_component"]
