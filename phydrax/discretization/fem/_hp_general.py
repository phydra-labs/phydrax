#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ReferenceRefinementTemplate(StrictModule, NonTrainableState):
    parent_cell_kind: str = eqx.field(static=True)
    child_cell_kinds: tuple[str, ...] = eqx.field(static=True)
    affine_matrices: Array
    affine_offsets: Array
    anisotropic_axes: tuple[int, ...] = eqx.field(static=True)
    template_id: str = eqx.field(static=True)

    def __init__(
        self,
        parent_cell_kind: str,
        child_cell_kinds: Sequence[str],
        affine_matrices: ArrayLike,
        affine_offsets: ArrayLike,
        /,
        *,
        anisotropic_axes: Sequence[int] = (),
    ):
        parent = str(parent_cell_kind)
        children = tuple(str(value) for value in child_cell_kinds)
        matrices = np.asarray(affine_matrices, dtype=float)
        offsets = np.asarray(affine_offsets, dtype=float)
        axes = tuple(int(value) for value in anisotropic_axes)
        if (
            not parent
            or not children
            or matrices.ndim != 3
            or matrices.shape[0] != len(children)
            or matrices.shape[1] != matrices.shape[2]
            or offsets.shape != (len(children), matrices.shape[1])
            or any(value < 0 or value >= matrices.shape[1] for value in axes)
            or not np.all(np.isfinite(matrices))
            or not np.all(np.isfinite(offsets))
        ):
            raise ValueError("Reference refinement template is inconsistent.")
        if any(abs(np.linalg.det(matrix)) <= 1.0e-14 for matrix in matrices):
            raise ValueError("Reference child maps must be nonsingular.")
        self.parent_cell_kind = parent
        self.child_cell_kinds = children
        self.affine_matrices = jnp.asarray(matrices)
        self.affine_offsets = jnp.asarray(offsets)
        self.anisotropic_axes = axes
        self.template_id = canonical_fingerprint(
            {
                "kind": "reference-refinement-template",
                "parent": parent,
                "children": children,
                "matrices": array_tree_fingerprint(matrices),
                "offsets": array_tree_fingerprint(offsets),
                "anisotropic_axes": axes,
            }
        )


def tensor_bisection_template(
    cell_kind: str, dimension: int, axis: int, /
) -> ReferenceRefinementTemplate:
    axis_ = int(axis)
    matrices = np.stack((np.eye(dimension), np.eye(dimension)))
    matrices[:, axis_, axis_] = 0.5
    offsets = np.zeros((2, dimension))
    offsets[1, axis_] = 0.5
    return ReferenceRefinementTemplate(
        cell_kind,
        (cell_kind, cell_kind),
        matrices,
        offsets,
        anisotropic_axes=(axis_,),
    )


def triangle_red_refinement_template() -> ReferenceRefinementTemplate:
    matrices = np.asarray(
        (
            ((0.5, 0.0), (0.0, 0.5)),
            ((0.5, 0.0), (0.0, 0.5)),
            ((0.5, 0.0), (0.0, 0.5)),
            ((0.0, -0.5), (0.5, 0.5)),
        )
    )
    offsets = np.asarray(((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.0)))
    return ReferenceRefinementTemplate("triangle", ("triangle",) * 4, matrices, offsets)


def prism_axial_refinement_template() -> ReferenceRefinementTemplate:
    return tensor_bisection_template("prism", 3, 2)


def pyramid_transition_refinement_template() -> ReferenceRefinementTemplate:
    matrices = np.asarray(tuple(np.diag((0.5, 0.5, 1.0)) for _ in range(4)))
    offsets = np.asarray(
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.5, 0.0))
    )
    return ReferenceRefinementTemplate(
        "pyramid", ("pyramid",) * 4, matrices, offsets, anisotropic_axes=(0, 1)
    )


class GeneralHPForest(StrictModule, NonTrainableState):
    cell_kinds: tuple[str, ...] = eqx.field(static=True)
    parent_slots: Array
    child_slots: Array
    child_valid: Array
    active: Array
    polynomial_orders: Array
    template_ids: tuple[str | None, ...] = eqx.field(static=True)
    forest_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_kinds: Sequence[str],
        parent_slots: ArrayLike,
        child_slots: ArrayLike,
        child_valid: ArrayLike,
        active: ArrayLike,
        polynomial_orders: ArrayLike,
        template_ids: Sequence[str | None],
        /,
    ):
        kinds = tuple(str(value) for value in cell_kinds)
        parents = np.asarray(parent_slots, dtype=np.int32)
        children = np.asarray(child_slots, dtype=np.int32)
        valid = np.asarray(child_valid, dtype=bool)
        active_ = np.asarray(active, dtype=bool)
        orders = np.asarray(polynomial_orders, dtype=np.int32)
        templates = tuple(None if value is None else str(value) for value in template_ids)
        capacity = len(kinds)
        if (
            not kinds
            or parents.shape != (capacity,)
            or children.ndim != 2
            or children.shape[0] != capacity
            or valid.shape != children.shape
            or active_.shape != (capacity,)
            or orders.shape != (capacity, 3)
            or len(templates) != capacity
            or np.any(orders < 0)
        ):
            raise ValueError("General hp forest arrays are inconsistent.")
        if np.any((children < 0) & valid) or np.any((children >= capacity) & valid):
            raise ValueError("General hp forest child slots are out of range.")
        self.cell_kinds = kinds
        self.parent_slots = jnp.asarray(parents)
        self.child_slots = jnp.asarray(children)
        self.child_valid = jnp.asarray(valid)
        self.active = jnp.asarray(active_)
        self.polynomial_orders = jnp.asarray(orders)
        self.template_ids = templates
        self.forest_id = canonical_fingerprint(
            {
                "kind": "general-hp-forest",
                "cell_kinds": kinds,
                "parents": array_tree_fingerprint(parents),
                "children": array_tree_fingerprint(children),
                "active": array_tree_fingerprint(active_),
                "orders": array_tree_fingerprint(orders),
                "templates": templates,
            }
        )

    @classmethod
    def roots(
        cls,
        cell_kinds: Sequence[str],
        polynomial_orders: ArrayLike,
        /,
        *,
        child_capacity: int = 8,
    ) -> "GeneralHPForest":
        kinds = tuple(cell_kinds)
        capacity = len(kinds)
        return cls(
            kinds,
            -np.ones((capacity,), dtype=np.int32),
            -np.ones((capacity, int(child_capacity)), dtype=np.int32),
            np.zeros((capacity, int(child_capacity)), dtype=bool),
            np.ones((capacity,), dtype=bool),
            polynomial_orders,
            (None,) * capacity,
        )


class NonconformingFacetOverlay(StrictModule, NonTrainableState):
    owner_cells: Array
    neighbour_cells: Array
    owner_subface_maps: Array
    neighbour_subface_maps: Array
    owner_levels: Array
    neighbour_levels: Array
    mortar_ids: tuple[str, ...] = eqx.field(static=True)
    overlay_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner_cells: ArrayLike,
        neighbour_cells: ArrayLike,
        owner_subface_maps: ArrayLike,
        neighbour_subface_maps: ArrayLike,
        owner_levels: ArrayLike,
        neighbour_levels: ArrayLike,
        mortar_ids: Sequence[str],
        /,
    ):
        owner = np.asarray(owner_cells, dtype=np.int32)
        neighbour = np.asarray(neighbour_cells, dtype=np.int32)
        owner_maps = np.asarray(owner_subface_maps, dtype=float)
        neighbour_maps = np.asarray(neighbour_subface_maps, dtype=float)
        owner_levels_ = np.asarray(owner_levels, dtype=np.int32)
        neighbour_levels_ = np.asarray(neighbour_levels, dtype=np.int32)
        mortars = tuple(str(value) for value in mortar_ids)
        count = owner.shape[0]
        if (
            owner.ndim != 1
            or neighbour.shape != owner.shape
            or owner_maps.shape != neighbour_maps.shape
            or owner_maps.shape[0] != count
            or owner_levels_.shape != owner.shape
            or neighbour_levels_.shape != owner.shape
            or len(mortars) != count
            or any(not value for value in mortars)
        ):
            raise ValueError("Nonconforming overlay arrays are inconsistent.")
        self.owner_cells = jnp.asarray(owner)
        self.neighbour_cells = jnp.asarray(neighbour)
        self.owner_subface_maps = jnp.asarray(owner_maps)
        self.neighbour_subface_maps = jnp.asarray(neighbour_maps)
        self.owner_levels = jnp.asarray(owner_levels_)
        self.neighbour_levels = jnp.asarray(neighbour_levels_)
        self.mortar_ids = mortars
        self.overlay_id = canonical_fingerprint(
            {
                "kind": "nonconforming-facet-overlay",
                "owner": array_tree_fingerprint(owner),
                "neighbour": array_tree_fingerprint(neighbour),
                "owner_maps": array_tree_fingerprint(owner_maps),
                "neighbour_maps": array_tree_fingerprint(neighbour_maps),
                "levels": (
                    array_tree_fingerprint(owner_levels_),
                    array_tree_fingerprint(neighbour_levels_),
                ),
                "mortars": mortars,
            }
        )


__all__ = [
    "GeneralHPForest",
    "NonconformingFacetOverlay",
    "ReferenceRefinementTemplate",
    "prism_axial_refinement_template",
    "pyramid_transition_refinement_template",
    "tensor_bisection_template",
    "triangle_red_refinement_template",
]
