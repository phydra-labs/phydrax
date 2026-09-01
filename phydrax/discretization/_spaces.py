#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractVectorSpace
from ..sparse import RowRelation
from ._core import nonempty_identifier, resolved_identifier


FieldRepresentation: TypeAlias = Literal[
    "point_value",
    "cell_average",
    "cell_integral",
    "basis_coefficient",
    "flux_moment",
    "circulation_moment",
    "polynomial_moment",
    "modal_coefficient",
    "particle_value",
    "cochain",
    "functional",
    "custom",
]
FieldConformity: TypeAlias = Literal[
    "H1",
    "Hdiv",
    "Hcurl",
    "L2",
    "discontinuous",
    "cochain",
    "unrestricted",
]


def _component_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("component_shape dimensions must be positive.")
    return shape


class AbstractDofLayout(StrictModule, NonTrainableState):
    """Abstract finite-coordinate layout for one field."""

    layout_id: str = eqx.field(static=True)

    @property
    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError


class TensorDofLayout(AbstractDofLayout):
    """Tensor-product degrees of freedom with explicit component axes."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    axis_shape: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    location_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        axis_shape: Sequence[int],
        /,
        *,
        component_shape: Sequence[int] = (),
        location_id: str | None = None,
        layout_id: str | None = None,
    ):
        names = tuple(str(name) for name in axis_names)
        shape = tuple(int(size) for size in axis_shape)
        components = _component_shape(component_shape)
        if not names or any(not name for name in names):
            raise ValueError("Tensor DOF layouts require non-empty axis names.")
        if len(set(names)) != len(names):
            raise ValueError("Tensor DOF axis names must be unique.")
        if len(shape) != len(names) or any(size <= 0 for size in shape):
            raise ValueError("Tensor DOF layouts require one positive size per axis.")
        location = None if location_id is None else str(location_id)
        if location is not None and not location:
            raise ValueError("location_id must be non-empty or None.")
        self.axis_names = names
        self.axis_shape = shape
        self.component_shape = components
        self.location_id = location
        self.layout_id = resolved_identifier(
            "layout_id",
            layout_id,
            {
                "kind": "tensor-dof-layout",
                "axis_names": list(names),
                "axis_shape": list(shape),
                "component_shape": list(components),
                "location": location,
            },
        )

    @property
    def size(self) -> int:
        return prod(self.axis_shape + self.component_shape)

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.axis_shape + self.component_shape


class EntityDofLayout(AbstractDofLayout):
    """Entity-associated global DOFs with an optional local gather relation."""

    entity_set_id: str = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    global_dof_count: int = eqx.field(static=True)
    dofs_per_entity: int = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    local_to_global: RowRelation | None
    orientation: Array | None

    def __init__(
        self,
        entity_set_id: str,
        entity_count: int,
        global_dof_count: int,
        /,
        *,
        dofs_per_entity: int = 1,
        component_shape: Sequence[int] = (),
        local_to_global: RowRelation | None = None,
        orientation: ArrayLike | None = None,
        layout_id: str | None = None,
    ):
        entity_set_id_ = nonempty_identifier("entity_set_id", entity_set_id)
        entity_count_ = int(entity_count)
        global_count = int(global_dof_count)
        per_entity = int(dofs_per_entity)
        components = _component_shape(component_shape)
        if entity_count_ < 0 or global_count < 0:
            raise ValueError("Entity and global DOF counts must be non-negative.")
        if per_entity <= 0:
            raise ValueError("dofs_per_entity must be positive.")
        orientation_ = None
        if local_to_global is not None:
            if not isinstance(local_to_global, RowRelation):
                raise TypeError("local_to_global must be a RowRelation or None.")
            if local_to_global.source_size != global_count:
                raise ValueError(
                    "local_to_global source size must equal global_dof_count."
                )
            if (
                not local_to_global.target_shape
                or local_to_global.target_shape[0] != entity_count_
            ):
                raise ValueError("local_to_global targets must begin with entity_count.")
            if orientation is not None:
                orientation_host = np.asarray(orientation)
                if orientation_host.shape != local_to_global.route_shape:
                    raise ValueError(
                        "orientation must have the local_to_global route shape."
                    )
                active = np.asarray(local_to_global.valid, dtype=bool)
                active_orientation = orientation_host[active]
                if np.any(~np.isfinite(active_orientation)) or np.any(
                    np.abs(active_orientation) != 1
                ):
                    raise ValueError("Active local DOF orientations must be ±1.")
                orientation_ = jnp.asarray(orientation_host, dtype=float)
        elif orientation is not None:
            raise ValueError("orientation requires local_to_global.")
        self.entity_set_id = entity_set_id_
        self.entity_count = entity_count_
        self.global_dof_count = global_count
        self.dofs_per_entity = per_entity
        self.component_shape = components
        self.local_to_global = local_to_global
        self.orientation = orientation_
        self.layout_id = resolved_identifier(
            "layout_id",
            layout_id,
            {
                "kind": "entity-dof-layout",
                "entity_set": entity_set_id_,
                "entity_count": entity_count_,
                "global_dof_count": global_count,
                "dofs_per_entity": per_entity,
                "component_shape": list(components),
                "local_to_global": None
                if local_to_global is None
                else {
                    "source": array_tree_fingerprint(local_to_global.source_indices),
                    "valid": array_tree_fingerprint(local_to_global.valid),
                    "target_shape": list(local_to_global.target_shape),
                },
                "orientation": None
                if orientation_ is None
                else array_tree_fingerprint(orientation_),
            },
        )

    @property
    def size(self) -> int:
        return self.global_dof_count * prod(self.component_shape)


class ModalDofLayout(AbstractDofLayout):
    """Stable modal coordinates with optional degeneracy groups."""

    mode_ids: tuple[str, ...] = eqx.field(static=True)
    group_ids: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        mode_ids: Sequence[str],
        /,
        *,
        group_ids: Sequence[int] | None = None,
        component_shape: Sequence[int] = (),
        layout_id: str | None = None,
    ):
        modes = tuple(str(value) for value in mode_ids)
        if not modes or any(not value for value in modes):
            raise ValueError("Modal DOF layouts require non-empty mode IDs.")
        if len(set(modes)) != len(modes):
            raise ValueError("Mode IDs must be unique.")
        groups = (
            tuple(range(len(modes)))
            if group_ids is None
            else tuple(int(value) for value in group_ids)
        )
        if len(groups) != len(modes) or any(value < 0 for value in groups):
            raise ValueError("group_ids must contain one non-negative value per mode.")
        components = _component_shape(component_shape)
        self.mode_ids = modes
        self.group_ids = groups
        self.component_shape = components
        self.layout_id = resolved_identifier(
            "layout_id",
            layout_id,
            {
                "kind": "modal-dof-layout",
                "mode_ids": list(modes),
                "group_ids": list(groups),
                "component_shape": list(components),
            },
        )

    @property
    def size(self) -> int:
        return len(self.mode_ids) * prod(self.component_shape)


class BlockDofLayout(AbstractDofLayout):
    """Ordered product of independently identified child layouts."""

    names: tuple[str, ...] = eqx.field(static=True)
    layouts: tuple[AbstractDofLayout, ...]

    def __init__(
        self,
        names: Sequence[str],
        layouts: Sequence[AbstractDofLayout],
        /,
        *,
        layout_id: str | None = None,
    ):
        names_ = tuple(str(name) for name in names)
        layouts_ = tuple(layouts)
        if not names_ or any(not name for name in names_):
            raise ValueError("Block layout names must be non-empty.")
        if len(set(names_)) != len(names_):
            raise ValueError("Block layout names must be unique.")
        if len(layouts_) != len(names_) or not all(
            isinstance(layout, AbstractDofLayout) for layout in layouts_
        ):
            raise TypeError("layouts must contain one AbstractDofLayout per name.")
        self.names = names_
        self.layouts = layouts_
        self.layout_id = resolved_identifier(
            "layout_id",
            layout_id,
            {
                "kind": "block-dof-layout",
                "blocks": [
                    {"name": name, "layout": layout.layout_id}
                    for name, layout in zip(names_, layouts_, strict=True)
                ],
            },
        )

    @property
    def size(self) -> int:
        return sum(layout.size for layout in self.layouts)


DofLayout: TypeAlias = TensorDofLayout | EntityDofLayout | ModalDofLayout | BlockDofLayout


class DiscreteFieldSpace(StrictModule, NonTrainableState):
    """One scientific field bound to exact finite coordinates and pairing."""

    name: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    layout: DofLayout
    vector_space: AbstractVectorSpace
    representation: FieldRepresentation = eqx.field(static=True)
    conformity: FieldConformity = eqx.field(static=True)
    projection_id: str | None = eqx.field(static=True)
    reconstruction_id: str | None = eqx.field(static=True)
    trace_space_id: str | None = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        support_id: str,
        layout: DofLayout,
        vector_space: AbstractVectorSpace,
        /,
        *,
        representation: FieldRepresentation,
        conformity: FieldConformity = "unrestricted",
        projection_id: str | None = None,
        reconstruction_id: str | None = None,
        trace_space_id: str | None = None,
        field_space_id: str | None = None,
    ):
        name_ = nonempty_identifier("name", name)
        support_id_ = nonempty_identifier("support_id", support_id)
        if not isinstance(
            layout,
            (TensorDofLayout, EntityDofLayout, ModalDofLayout, BlockDofLayout),
        ):
            raise TypeError("layout must be a supported DofLayout value.")
        if not isinstance(vector_space, AbstractVectorSpace):
            raise TypeError("vector_space must be an AbstractVectorSpace.")
        if vector_space.size != layout.size:
            raise ValueError(
                f"Vector-space size {vector_space.size} does not match "
                f"DOF-layout size {layout.size}."
            )
        if representation not in (
            "point_value",
            "basis_coefficient",
            "cell_average",
            "cell_integral",
            "flux_moment",
            "circulation_moment",
            "polynomial_moment",
            "modal_coefficient",
            "particle_value",
            "cochain",
            "functional",
            "custom",
        ):
            raise ValueError("Unknown field representation.")
        if conformity not in (
            "H1",
            "Hdiv",
            "Hcurl",
            "L2",
            "discontinuous",
            "cochain",
            "unrestricted",
        ):
            raise ValueError("Unknown field conformity.")
        projection = (
            None
            if projection_id is None
            else nonempty_identifier("projection_id", projection_id)
        )
        reconstruction = (
            None
            if reconstruction_id is None
            else nonempty_identifier("reconstruction_id", reconstruction_id)
        )
        trace = (
            None
            if trace_space_id is None
            else nonempty_identifier("trace_space_id", trace_space_id)
        )
        self.name = name_
        self.support_id = support_id_
        self.layout = layout
        self.vector_space = vector_space
        self.representation = representation
        self.conformity = conformity
        self.projection_id = projection
        self.reconstruction_id = reconstruction
        self.trace_space_id = trace
        self.field_space_id = resolved_identifier(
            "field_space_id",
            field_space_id,
            {
                "kind": "discrete-field-space",
                "name": name_,
                "support": support_id_,
                "layout": layout.layout_id,
                "vector_space": vector_space.space_id,
                "representation": representation,
                "conformity": conformity,
                "projection": projection,
                "reconstruction": reconstruction,
                "trace_space": trace,
            },
        )


__all__ = [
    "AbstractDofLayout",
    "BlockDofLayout",
    "DiscreteFieldSpace",
    "DofLayout",
    "EntityDofLayout",
    "FieldConformity",
    "FieldRepresentation",
    "ModalDofLayout",
    "TensorDofLayout",
]
