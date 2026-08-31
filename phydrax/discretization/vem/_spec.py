#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VirtualElementSpec(StrictModule, NonTrainableState):
    """One enhanced conforming virtual-element space specification."""

    family: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    enhanced: bool = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: str,
        degree: int,
        /,
        *,
        conformity: str = "H1",
        enhanced: bool = True,
        value_shape: Sequence[int] = (),
    ):
        family_ = str(family)
        degree_ = int(degree)
        conformity_ = str(conformity)
        shape = tuple(int(value) for value in value_shape)
        if not family_:
            raise ValueError("Virtual-element family must be non-empty.")
        if degree_ not in (1, 2, 3):
            raise ValueError("Qualified virtual-element degree must be 1, 2, or 3.")
        if conformity_ != "H1":
            raise ValueError("Initial virtual-element support is H1-conforming only.")
        if not enhanced:
            raise ValueError("Initial virtual-element support requires enhanced spaces.")
        if any(value <= 0 for value in shape):
            raise ValueError("Virtual-element value dimensions must be positive.")
        self.family = family_
        self.degree = degree_
        self.conformity = conformity_
        self.enhanced = True
        self.value_shape = shape
        self.element_id = canonical_fingerprint(
            {
                "kind": "virtual-element-spec",
                "family": family_,
                "degree": degree_,
                "conformity": conformity_,
                "enhanced": True,
                "value_shape": list(shape),
            }
        )

    @property
    def cell_moment_count(self) -> int:
        return self.degree * (self.degree - 1) // 2

    @property
    def edge_interior_dof_count(self) -> int:
        return self.degree - 1

    def local_dof_count(self, arity: int, /) -> int:
        arity_ = int(arity)
        if arity_ < 3:
            raise ValueError("Virtual elements require polygon arity at least three.")
        return self.degree * arity_ + self.cell_moment_count


class VirtualElementFieldSpec(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    element: VirtualElementSpec
    component_shape: tuple[int, ...] = eqx.field(static=True)
    field_spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        element: VirtualElementSpec,
        /,
        *,
        component_shape: Sequence[int] = (),
    ):
        name_ = str(name)
        shape = tuple(int(value) for value in component_shape)
        if not name_:
            raise ValueError("Virtual-element field name must be non-empty.")
        if not isinstance(element, VirtualElementSpec):
            raise TypeError("element must be VirtualElementSpec.")
        if element.value_shape:
            raise ValueError("Initial virtual elements require scalar element values.")
        if shape:
            raise ValueError("Initial virtual-element qualification is scalar only.")
        self.name = name_
        self.element = element
        self.component_shape = shape
        self.field_spec_id = canonical_fingerprint(
            {
                "kind": "virtual-element-field",
                "name": name_,
                "element": element.element_id,
                "component_shape": list(shape),
            }
        )


def conforming_h1_virtual_element(degree: int, /) -> VirtualElementSpec:
    return VirtualElementSpec("ConformingH1", degree)


__all__ = [
    "VirtualElementFieldSpec",
    "VirtualElementSpec",
    "conforming_h1_virtual_element",
]
