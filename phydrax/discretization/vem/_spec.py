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
    """One bounded virtual-element family specification."""

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
        if family_ in ("ConformingHdiv", "ConformingHcurl"):
            if shape and shape != (2,):
                raise ValueError(f"{family_} requires value_shape=(2,).")
            shape = (2,)
        elif family_ == "DiscontinuousL2" and shape:
            raise ValueError("DiscontinuousL2 currently requires scalar values.")
        families = {
            "ConformingH1": "H1",
            "ConformingHdiv": "Hdiv",
            "ConformingHcurl": "Hcurl",
            "DiscontinuousL2": "L2",
        }
        if family_ not in families:
            raise ValueError(
                "Virtual-element family must be ConformingH1, ConformingHdiv, "
                "ConformingHcurl, or DiscontinuousL2."
            )
        if degree_ < 1:
            raise ValueError("Virtual-element degree must be positive.")
        expected_conformity = families[family_]
        if conformity_ == "H1" and family_ != "ConformingH1":
            conformity_ = expected_conformity
        if conformity_ != expected_conformity:
            raise ValueError(f"{family_} requires conformity={expected_conformity!r}.")
        if not enhanced:
            raise ValueError("Virtual-element spaces require enhanced projections.")
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
    def vertex_dofs_per_entity(self) -> int:
        return 1 if self.family == "ConformingH1" else 0

    @property
    def edge_dofs_per_entity(self) -> int:
        if self.family == "ConformingH1":
            return self.degree - 1
        if self.family in ("ConformingHdiv", "ConformingHcurl"):
            return self.degree + 1
        return 0

    @property
    def cell_dofs_per_entity(self) -> int:
        if self.family == "ConformingH1":
            return self.degree * (self.degree - 1) // 2
        if self.family in ("ConformingHdiv", "ConformingHcurl"):
            return self.degree * (self.degree + 1)
        return (self.degree + 1) * (self.degree + 2) // 2

    @property
    def cell_moment_count(self) -> int:
        return self.cell_dofs_per_entity

    @property
    def edge_interior_dof_count(self) -> int:
        return self.edge_dofs_per_entity

    @property
    def trace_kind(self) -> str:
        if self.family == "ConformingH1":
            return "value"
        if self.family == "ConformingHdiv":
            return "normal"
        if self.family == "ConformingHcurl":
            return "tangential"
        return "none"

    @property
    def differential_kind(self) -> str:
        if self.family == "ConformingH1":
            return "gradient"
        if self.family == "ConformingHdiv":
            return "divergence"
        if self.family == "ConformingHcurl":
            return "curl"
        return "none"

    def local_dof_count(self, arity: int, /) -> int:
        arity_ = int(arity)
        if arity_ < 3:
            raise ValueError("Virtual elements require polygon arity at least three.")
        return (
            self.vertex_dofs_per_entity * arity_
            + self.edge_dofs_per_entity * arity_
            + self.cell_dofs_per_entity
        )


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
        if element.value_shape and shape:
            raise ValueError("Declare vector values on the element or field, not both.")
        if any(value <= 0 for value in shape):
            raise ValueError("Virtual-element field dimensions must be positive.")
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


def conforming_hdiv_virtual_element(degree: int, /) -> VirtualElementSpec:
    return VirtualElementSpec(
        "ConformingHdiv", degree, conformity="Hdiv", value_shape=(2,)
    )


def conforming_hcurl_virtual_element(degree: int, /) -> VirtualElementSpec:
    return VirtualElementSpec(
        "ConformingHcurl", degree, conformity="Hcurl", value_shape=(2,)
    )


def discontinuous_l2_virtual_element(degree: int, /) -> VirtualElementSpec:
    return VirtualElementSpec("DiscontinuousL2", degree, conformity="L2")


__all__ = [
    "VirtualElementFieldSpec",
    "VirtualElementSpec",
    "conforming_h1_virtual_element",
    "conforming_hdiv_virtual_element",
    "conforming_hcurl_virtual_element",
    "discontinuous_l2_virtual_element",
]
