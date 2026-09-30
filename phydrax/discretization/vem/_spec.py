#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._polynomial._orthogonal import standard_vandermonde
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._form_type import FormType, FormValueSpec


def _family_value_spec(family: str, /) -> FormValueSpec:
    """Scientific values represented by the qualified planar VEM projectors."""
    match family:
        case "ConformingH1" | "DiscontinuousL2":
            # Discontinuous scalar polynomials are 0-forms, not densities.
            return FormValueSpec(FormType(2, 0), proxy="scalar")
        case "ConformingHdiv":
            return FormValueSpec(FormType(2, 1, twist="twisted"), proxy="flux")
        case "ConformingHcurl":
            return FormValueSpec(FormType(2, 1), proxy="circulation")
        case _:
            raise ValueError(
                "Virtual-element family must be ConformingH1, ConformingHdiv, "
                "ConformingHcurl, or DiscontinuousL2."
            )


class VirtualElementSpec(StrictModule, NonTrainableState):
    """One bounded virtual-element family specification."""

    family: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    value_spec: FormValueSpec = eqx.field(static=True)
    enhanced: bool = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: str,
        degree: int,
        /,
        *,
        value_spec: FormValueSpec,
        enhanced: bool = True,
    ) -> None:
        family_ = str(family)
        degree_ = int(degree)
        expected = _family_value_spec(family_)
        if not isinstance(value_spec, FormValueSpec):
            raise TypeError("value_spec must be FormValueSpec.")
        if value_spec.value_spec_id != expected.value_spec_id:
            raise ValueError(
                f"{family_} requires the qualified planar {expected.proxy} form "
                f"with twist={expected.form_type.twist!r}."
            )
        if degree_ < 1:
            raise ValueError("Virtual-element degree must be positive.")
        if not enhanced:
            raise ValueError("Virtual-element spaces require enhanced projections.")
        self.family = family_
        self.degree = degree_
        self.value_spec = value_spec
        self.enhanced = True
        self.element_id = canonical_fingerprint(
            {
                "kind": "virtual-element-spec",
                "family": family_,
                "degree": degree_,
                "value_spec": value_spec.value_spec_id,
                "enhanced": True,
            }
        )

    @property
    def form_type(self) -> FormType:
        return self.value_spec.form_type

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.value_spec.value_shape

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

    def edge_trace_basis(self, coordinates: ArrayLike, /) -> Array:
        """Tabulate the degree-`k` edge trace basis at canonical edge coordinates.

        `coordinates` lie in `[-1, 1]` along an edge from its lower to its higher
        vertex index; the result has shape `(*coordinates.shape, k + 1)`. H1
        value traces use the Lagrange basis on the `k + 1` Gauss--Lobatto
        nodes (start vertex, interior edge DOFs, end vertex). H(div) normal and
        H(curl) tangential traces use the dual-scaled Legendre basis
        `(2m + 1) P_m`, so the canonical trace is `sum_m (2m + 1) P_m d_m` for
        the Legendre moment DOFs `d_m`. Discontinuous L2 spaces have no trace.
        """
        from ...integration import GaussLobattoLegendreRule, interval_rule_data
        from ..fem import lagrange_1d_tabulation

        values = jnp.asarray(coordinates)
        flat = values.reshape((-1,))
        match self.trace_kind:
            case "value":
                nodes = interval_rule_data(
                    GaussLobattoLegendreRule(self.degree + 1)
                ).nodes
                basis, _ = lagrange_1d_tabulation(
                    jnp.asarray(nodes, dtype=flat.dtype), flat
                )
            case "normal" | "tangential":
                dual = 2 * jnp.arange(self.degree + 1, dtype=flat.dtype) + 1
                basis = standard_vandermonde("legendre", flat, self.degree) * dual
            case "none":
                raise ValueError(
                    "Discontinuous L2 virtual elements have no boundary trace."
                )
            case kind:
                raise ValueError(f"Unknown virtual-element trace kind {kind!r}.")
        return basis.reshape(values.shape + (self.degree + 1,))

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
    ) -> None:
        name_ = str(name)
        shape = tuple(component_shape)
        if not name_:
            raise ValueError("Virtual-element field name must be non-empty.")
        if not isinstance(element, VirtualElementSpec):
            raise TypeError("element must be VirtualElementSpec.")
        if element.value_shape and shape:
            raise ValueError("Declare vector values on the element or field, not both.")
        if any(value <= 0 for value in shape):
            raise ValueError("Virtual-element field dimensions must be positive.")
        if shape:
            raise NotImplementedError(
                "Component-replicated virtual-element fields are not supported."
            )
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
    return VirtualElementSpec(
        "ConformingH1", degree, value_spec=_family_value_spec("ConformingH1")
    )


def conforming_hdiv_virtual_element(degree: int, /) -> VirtualElementSpec:
    return VirtualElementSpec(
        "ConformingHdiv", degree, value_spec=_family_value_spec("ConformingHdiv")
    )


def conforming_hcurl_virtual_element(degree: int, /) -> VirtualElementSpec:
    return VirtualElementSpec(
        "ConformingHcurl", degree, value_spec=_family_value_spec("ConformingHcurl")
    )


def discontinuous_l2_virtual_element(degree: int, /) -> VirtualElementSpec:
    return VirtualElementSpec(
        "DiscontinuousL2", degree, value_spec=_family_value_spec("DiscontinuousL2")
    )


__all__ = [
    "VirtualElementFieldSpec",
    "VirtualElementSpec",
    "conforming_h1_virtual_element",
    "conforming_hdiv_virtual_element",
    "conforming_hcurl_virtual_element",
    "discontinuous_l2_virtual_element",
]
