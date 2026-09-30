#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Boundary trace-space capabilities published by boundary-integral owners.

A boundary trace-space capability binds one owner's boundary coefficient space
to its scientific trace quantity, representation, Sobolev conformity,
orientation, physical Gram pairing, and geometry revision. Scalar Cauchy data
(Dirichlet and Neumann traces), tangential RWG surface currents, and their
Buffa--Christiansen duals are distinct quantities and never interchangeable. The records carry no volume
support and no integration domain; a boundary coefficient space is paired
through the owner's actual arc-length or area mass, whose inverse is the
owner's prepared Riesz solve.
"""

from __future__ import annotations

from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..exterior._form_type import FormTwist, FormType
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalPairing,
    DualSpace,
    OperatorPairing,
)
from ..typing import parse


BoundaryTraceQuantity: TypeAlias = Literal[
    "dirichlet", "neumann", "surface-current", "surface-current-dual"
]
BoundaryTraceRepresentation: TypeAlias = Literal[
    "continuous-p1", "dp0", "rwg", "buffa-christiansen"
]
BoundaryTraceConformity: TypeAlias = Literal[
    "H^(1/2)(Gamma)", "H^(-1/2)(Gamma)", "H^(-1/2)(div_Gamma)"
]
BoundaryTraceOrientation: TypeAlias = Literal["unoriented", "outward", "surface-oriented"]
BoundaryEntityKind: TypeAlias = Literal["vertex", "panel", "face", "edge"]


def boundary_geometry_revision(vertices: ArrayLike, cells: ArrayLike, /) -> str:
    """Fingerprint one boundary realization by its vertex coordinates and cells."""
    return canonical_fingerprint(
        {
            "kind": "boundary-geometry-revision",
            "vertices": array_tree_fingerprint(np.asarray(vertices, dtype=np.float64)),
            "cells": array_tree_fingerprint(np.asarray(cells, dtype=np.int64)),
        }
    )


def _quantity_semantics(
    quantity: BoundaryTraceQuantity, /
) -> tuple[BoundaryTraceConformity, BoundaryTraceOrientation]:
    match quantity:
        case "dirichlet":
            return "H^(1/2)(Gamma)", "unoriented"
        case "neumann":
            return "H^(-1/2)(Gamma)", "outward"
        case "surface-current" | "surface-current-dual":
            return "H^(-1/2)(div_Gamma)", "surface-oriented"
        case _:
            assert_never(quantity)


def _representation_entity(
    representation: BoundaryTraceRepresentation,
    quantity: BoundaryTraceQuantity,
    boundary_dimension: int,
    /,
) -> BoundaryEntityKind:
    """Return the carrying entity and refuse representations of another quantity."""
    match representation:
        case "continuous-p1":
            admitted: BoundaryTraceQuantity = "dirichlet"
            entity: BoundaryEntityKind = "vertex"
        case "dp0":
            admitted = "neumann"
            entity = "panel" if boundary_dimension == 1 else "face"
        case "rwg":
            admitted = "surface-current"
            entity = "edge"
        case "buffa-christiansen":
            admitted = "surface-current-dual"
            entity = "edge"
        case _:
            assert_never(representation)
    if quantity != admitted:
        raise ValueError(
            f"A {representation} boundary space represents the {admitted} trace, "
            f"not the {quantity} trace."
        )
    if entity == "edge" and boundary_dimension != 2:
        raise ValueError("Tangential surface currents live on surfaces in 3-D.")
    return entity


def _require_rank_one(space: ArraySpace, name: str, /) -> None:
    if not isinstance(space, ArraySpace):
        raise TypeError(f"{name} must be an ArraySpace.")
    if len(space.shape) != 1:
        raise ValueError("Boundary coefficient spaces are rank-1 coordinate vectors.")


def _require_gram_space(
    coefficient_space: ArraySpace,
    gram_space: ArraySpace,
    mass: AbstractLinearOperator,
    /,
) -> None:
    _require_rank_one(coefficient_space, "coefficient_space")
    _require_rank_one(gram_space, "gram_space")
    if (
        gram_space.shape != coefficient_space.shape
        or gram_space.dtype != coefficient_space.dtype
    ):
        raise ValueError("gram_space must pair the coefficient space's own coordinates.")
    if not isinstance(gram_space.pairing, (OperatorPairing, DiagonalPairing)):
        raise ValueError("gram_space must carry the owner's physical Gram pairing.")
    if not isinstance(mass, AbstractLinearOperator):
        raise TypeError("mass must be an AbstractLinearOperator.")
    if (
        not gram_space.compatible(mass.source)
        or not isinstance(mass.target, DualSpace)
        or not gram_space.compatible(mass.target.primal)
    ):
        raise ValueError("mass must map the Gram space into its dual.")


@final
class BoundaryTraceSpaceCapability(StrictModule, NonTrainableState):
    """One boundary trace space of a boundary-integral owner.

    `coefficient_space` is the owner's native coordinate space, the source of
    its boundary operators. `gram_space` pairs the same coordinates through
    the owner's physical Gram map (arc length on curves, area on surfaces);
    `mass` maps them to that dual, `(M u)_i = ∫ u v_i ds`. The two spaces are
    the same object when the owner's native space already carries its Gram
    pairing. `conformity` and `orientation` follow from `quantity`: Dirichlet
    traces are unoriented, Neumann traces differentiate along the owner's
    interior-to-exterior normal, and surface currents follow the oriented
    surface complex, so reversing the triangle orientation negates every RWG
    and Buffa--Christiansen basis function. `entity_kind` names the boundary entities carrying one
    coefficient each. `revision_id` identifies the boundary geometry
    realization, so moved vertices give a new revision.
    """

    owner_id: str = eqx.field(static=True)
    quantity: BoundaryTraceQuantity = eqx.field(static=True)
    representation: BoundaryTraceRepresentation = eqx.field(static=True)
    conformity: BoundaryTraceConformity = eqx.field(static=True)
    orientation: BoundaryTraceOrientation = eqx.field(static=True)
    coefficient_space: ArraySpace
    gram_space: ArraySpace
    mass: AbstractLinearOperator
    entity_kind: BoundaryEntityKind = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    boundary_dimension: int = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        owner_id: str,
        quantity: BoundaryTraceQuantity,
        representation: BoundaryTraceRepresentation,
        coefficient_space: ArraySpace,
        gram_space: ArraySpace,
        mass: AbstractLinearOperator,
        ambient_dimension: int,
        revision_id: str,
    ) -> None:
        owner = canonical_identifier(owner_id, "owner_id")
        revision = canonical_identifier(revision_id, "revision_id")
        quantity = parse(quantity, BoundaryTraceQuantity, "quantity")
        representation = parse(
            representation, BoundaryTraceRepresentation, "representation"
        )
        match ambient_dimension:
            case 2 | 3:
                boundary_dimension = ambient_dimension - 1
            case _:
                raise ValueError("Boundary traces are published in 2-D or 3-D.")
        entity = _representation_entity(representation, quantity, boundary_dimension)
        conformity, orientation = _quantity_semantics(quantity)
        _require_gram_space(coefficient_space, gram_space, mass)
        self.owner_id = owner
        self.quantity = quantity
        self.representation = representation
        self.conformity = conformity
        self.orientation = orientation
        self.coefficient_space = coefficient_space
        self.gram_space = gram_space
        self.mass = mass
        self.entity_kind = entity
        self.entity_count = coefficient_space.shape[0]
        self.ambient_dimension = ambient_dimension
        self.boundary_dimension = boundary_dimension
        self.revision_id = revision
        self.capability_id = canonical_fingerprint(
            {
                "kind": "boundary-trace-space-capability",
                "owner": owner,
                "quantity": quantity,
                "representation": representation,
                "space": coefficient_space.space_id,
                "gram_space": gram_space.space_id,
                "pairing": gram_space.pairing.pairing_id,
                "mass": mass.operator_id,
                "ambient_dimension": ambient_dimension,
                "revision": revision,
            }
        )

    @property
    def form_type(self) -> FormType:
        """Intrinsic boundary form carried by the declared trace quantity."""
        twist: FormTwist
        match self.quantity:
            case "dirichlet":
                degree, twist = 0, "untwisted"
            case "neumann":
                degree, twist = self.boundary_dimension, "twisted"
            case "surface-current" | "surface-current-dual":
                degree, twist = 1, "twisted"
            case _:
                assert_never(self.quantity)
        return FormType(
            self.boundary_dimension,
            degree,
            twist=twist,
            ambient_dimension=self.ambient_dimension,
        )


def _require_cauchy_part(
    capability: BoundaryTraceSpaceCapability,
    quantity: BoundaryTraceQuantity,
    name: str,
    /,
) -> None:
    if not isinstance(capability, BoundaryTraceSpaceCapability):
        raise TypeError(f"{name} must be a BoundaryTraceSpaceCapability.")
    if capability.quantity != quantity:
        raise ValueError(
            f"Cauchy trace capabilities pair scalar Dirichlet and Neumann traces; "
            f"{name} is a {capability.quantity} trace."
        )


@final
class CauchyTraceCapability(StrictModule, NonTrainableState):
    """Scalar Cauchy data of one boundary-integral owner and their duality.

    `duality` maps Dirichlet coefficients into the Neumann dual,
    `(B φ)_i = ∫ φ q_i ds`, so `pair(q, φ)` is the physical duality
    `∫ q φ ds`. `interior` names the owner's declared interior; the Neumann
    trace differentiates along the normal pointing out of it, whatever the
    traversal or winding of the boundary cells. `convention_id` identifies the
    owner's trace, jump, and far-field convention. Both traces belong to one
    owner and one geometry revision; tangential surface currents are refused.
    """

    dirichlet: BoundaryTraceSpaceCapability
    neumann: BoundaryTraceSpaceCapability
    duality: AbstractLinearOperator
    interior: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        dirichlet: BoundaryTraceSpaceCapability,
        neumann: BoundaryTraceSpaceCapability,
        duality: AbstractLinearOperator,
        /,
        *,
        interior: str,
        convention_id: str,
    ) -> None:
        declared = canonical_identifier(interior, "interior")
        convention = canonical_identifier(convention_id, "convention_id")
        _require_cauchy_part(dirichlet, "dirichlet", "dirichlet")
        _require_cauchy_part(neumann, "neumann", "neumann")
        if dirichlet.owner_id != neumann.owner_id:
            raise ValueError("Cauchy traces must belong to one boundary owner.")
        if dirichlet.revision_id != neumann.revision_id:
            raise ValueError("Cauchy traces must share one boundary geometry revision.")
        if dirichlet.ambient_dimension != neumann.ambient_dimension:
            raise ValueError("Cauchy traces must share one ambient dimension.")
        if not isinstance(duality, AbstractLinearOperator):
            raise TypeError("duality must be an AbstractLinearOperator.")
        if (
            not dirichlet.coefficient_space.compatible(duality.source)
            or not isinstance(duality.target, DualSpace)
            or not neumann.coefficient_space.compatible(duality.target.primal)
        ):
            raise ValueError(
                "duality must map Dirichlet coefficients into the Neumann dual."
            )
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.duality = duality
        self.interior = declared
        self.convention_id = convention
        self.capability_id = canonical_fingerprint(
            {
                "kind": "cauchy-trace-capability",
                "dirichlet": dirichlet.capability_id,
                "neumann": neumann.capability_id,
                "duality": duality.operator_id,
                "interior": declared,
                "convention": convention,
            }
        )

    def pair(self, neumann: ArrayLike, dirichlet: ArrayLike, /) -> Array:
        """Physical duality `∫ q φ ds` of Neumann and Dirichlet coefficients."""
        flux = self.neumann.coefficient_space.validate(neumann)
        dual = self.duality.mv(self.dirichlet.coefficient_space.validate(dirichlet))
        return ein.contract("i,i->", flux, dual, backend="jax")


__all__ = [
    "BoundaryEntityKind",
    "BoundaryTraceConformity",
    "BoundaryTraceOrientation",
    "BoundaryTraceQuantity",
    "BoundaryTraceRepresentation",
    "BoundaryTraceSpaceCapability",
    "CauchyTraceCapability",
]
