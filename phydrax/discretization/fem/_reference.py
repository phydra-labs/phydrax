#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ...exterior._form_type import FormType, FormValueSpec
from .._reference_cell import reference_cell_topology
from . import (
    _high_order as high_order_owner,
    _hybrid_reference as hybrid_reference_owner,
)
from ._reference_types import FiniteElementSpec, FiniteElementTabulator


def lagrange_element(cell_kind: str, degree: int, /) -> FiniteElementSpec:
    """Construct one implemented scalar nodal Lagrange reference element."""

    cell = str(cell_kind)
    order = int(degree)
    scalar_spec = FormValueSpec(
        FormType(reference_cell_topology(cell).dimension, 0, twist="untwisted"),
        proxy="scalar",
    )
    if cell == "triangle" and order == 1:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            (((0,), (1,), (2,)), ((), (), ()), ((),)),
            value_spec=scalar_spec,
        )
    if cell == "triangle" and order == 2:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.0, 0.5),
            ),
            (((0,), (1,), (2,)), ((3,), (4,), (5,)), ((),)),
            value_spec=scalar_spec,
        )
    if cell == "quadrilateral" and order == 1:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            (((0,), (1,), (2,), (3,)), ((), (), (), ()), ((),)),
            value_spec=scalar_spec,
        )
    if cell == "tetrahedron" and order == 1:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (((0,), (1,), (2,), (3,)), ((),) * 6, ((),) * 4, ((),)),
            value_spec=scalar_spec,
        )
    if cell == "hexahedron" and order == 1:
        return FiniteElementSpec(
            "Lagrange",
            cell,
            order,
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            ),
            (
                tuple((index,) for index in range(8)),
                ((),) * 12,
                ((),) * 6,
                ((),),
            ),
            value_spec=scalar_spec,
        )
    if cell in ("triangle", "tetrahedron") and order >= 0:
        return high_order_owner.SimplexNodalFamily(cell, order).finite_element()
    if cell in ("interval", "quadrilateral", "hexahedron") and order >= 0:
        return high_order_owner.ReferenceNodalFamily(cell, order).finite_element()
    if cell in ("prism", "pyramid") and order >= 1:
        return hybrid_reference_owner.HybridReferenceFamily(cell, order).finite_element()
    raise ValueError(
        "lagrange_element.cell_kind_degree must select a supported cell and polynomial degree; got an unsupported combination"
    )


def discontinuous_element(cell_kind: str, degree: int = 0, /) -> FiniteElementSpec:
    cell = str(cell_kind)
    order = int(degree)
    if order >= 1:
        if cell in ("quadrilateral", "hexahedron"):
            base = high_order_owner.ReferenceNodalFamily(cell, order).finite_element()
        else:
            base = lagrange_element(cell, order)
        entities: list[tuple[tuple[int, ...], ...]] = [
            tuple(() for _ in dimension) for dimension in base.entity_dofs
        ]
        entities[-1] = (tuple(range(base.local_dof_count)),)
        return FiniteElementSpec(
            "DiscontinuousLagrange",
            cell,
            order,
            base.reference_nodes,
            tuple(entities),
            value_spec=base.value_spec,
            continuity="discontinuous",
            representation=base.representation,
            tabulator=FiniteElementTabulator(base),
            tabulator_id=f"discontinuous:{base.element_id}",
        )
    if order != 0:
        raise ValueError(
            "discontinuous_element.degree must be nonnegative; got a negative degree"
        )
    dimension = {
        "triangle": 2,
        "quadrilateral": 2,
        "interval": 1,
        "tetrahedron": 3,
        "hexahedron": 3,
        "prism": 3,
        "pyramid": 3,
    }.get(cell)
    if dimension is None:
        raise ValueError(
            "discontinuous_element.cell_kind must select a supported reference cell; got an unsupported cell"
        )
    center = {
        "interval": ((0.5,),),
        "triangle": ((1.0 / 3.0, 1.0 / 3.0),),
        "quadrilateral": ((0.5, 0.5),),
        "tetrahedron": ((0.25, 0.25, 0.25),),
        "hexahedron": ((0.5, 0.5, 0.5),),
        "prism": ((1.0 / 3.0, 1.0 / 3.0, 0.5),),
        "pyramid": ((0.5, 0.5, 0.25),),
    }[cell]
    entities = {
        "interval": (((), ()), ((0,),)),
        "triangle": (((), (), ()), ((), (), ()), ((0,),)),
        "quadrilateral": (((), (), (), ()), ((), (), (), ()), ((0,),)),
        "tetrahedron": (
            ((), (), (), ()),
            ((),) * 6,
            ((),) * 4,
            ((0,),),
        ),
        "hexahedron": (
            ((),) * 8,
            ((),) * 12,
            ((),) * 6,
            ((0,),),
        ),
        "prism": (((),) * 6, ((),) * 9, ((),) * 5, ((0,),)),
        "pyramid": (((),) * 5, ((),) * 8, ((),) * 5, ((0,),)),
    }[cell]
    return FiniteElementSpec(
        "DiscontinuousLagrange",
        cell,
        0,
        center,
        entities,
        value_spec=FormValueSpec(
            FormType(dimension, 0, twist="untwisted"), proxy="scalar"
        ),
        continuity="discontinuous",
    )


__all__ = [
    "discontinuous_element",
    "lagrange_element",
]
