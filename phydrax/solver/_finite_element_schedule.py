#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import final, TYPE_CHECKING


if TYPE_CHECKING:
    from .._model._structure import FiniteElementAcceptedStateArchiveRecipe

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._array_archive import (
    array_collection_digest,
    array_payload_byte_count,
    array_payload_digest,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._spaces import FieldRepresentation
from ..discretization.fem import (
    discontinuous_element,
    ElementContinuity,
    FiniteElementDofOwnership,
    FiniteElementFieldSpec,
    FiniteElementSpec,
    form_element,
    FormElementFamily,
    lagrange_element,
)
from ..equations import (
    MaterialSiteId,
    MaterialState,
    MaterialTransaction,
)
from ..exterior._form_type import FormProxy, FormTwist, FormType, FormValueSpec
from ..lifecycle import (
    CheckpointManifest,
    CheckpointShard,
    create as create_lifecycle_archive,
    open as open_lifecycle_archive,
)
from ..typing import checked, parse
from ._finite_element_checkpoint import FiniteElementCheckpoint
from ._schedule import TimeLaw


class FiniteElementFieldTransferSemantics(StrEnum):
    """Declared action required for one accepted field across topology epochs."""

    MATERIAL_FOLLOWING_CONTENT = "material_following_content"
    EULERIAN_PHYSICAL_OVERLAP = "eulerian_physical_overlap"
    NESTED_FE = "nested_fe"
    NONCONSERVATIVE_INTERPOLATION = "nonconservative_interpolation"
    CONSERVATIVE_CELL_AVERAGE = "conservative_cell_average"


@final
class FiniteElementAcceptedElementRecipe(StrictModule, NonTrainableState):
    """Callable-free canonical element recipe retained by accepted field layouts."""

    recipe_kind: str = eqx.field(static=True)
    family: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    continuity: str = eqx.field(static=True)
    representation: str = eqx.field(static=True)
    value_spec: tuple[int, int, str, tuple[int, ...], int, str] = eqx.field(static=True)
    form_identity: tuple[int, int, int, str, str] | None = eqx.field(static=True)
    reference_nodes: Array
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    tabulator_id: str | None = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        recipe_kind: str,
        family: str,
        cell_kind: str,
        degree: int,
        continuity: str,
        representation: str,
        value_spec: tuple[int, int, str, tuple[int, ...], int, str],
        form_identity: tuple[int, int, int, str, str] | None,
        reference_nodes: ArrayLike,
        entity_dofs: tuple[tuple[tuple[int, ...], ...], ...],
        tabulator_id: str | None,
        element_id: str,
        /,
    ) -> None:
        if recipe_kind not in ("form", "lagrange", "discontinuous", "raw"):
            raise ValueError("Unsupported accepted FE element recipe kind.")
        self.recipe_kind = recipe_kind
        self.family = str(family)
        self.cell_kind = str(cell_kind)
        self.degree = int(degree)
        self.continuity = str(continuity)
        self.representation = str(representation)
        self.value_spec = value_spec
        self.form_identity = form_identity
        self.reference_nodes = jnp.asarray(reference_nodes, dtype=jnp.float64)
        self.entity_dofs = tuple(
            tuple(tuple(int(value) for value in entity) for entity in level)
            for level in entity_dofs
        )
        self.tabulator_id = None if tabulator_id is None else str(tabulator_id)
        self.element_id = str(element_id)
        if not self.family or not self.cell_kind or not self.element_id:
            raise ValueError("Accepted FE element recipe identities must be non-empty.")
        if (recipe_kind == "form") != (form_identity is not None):
            raise ValueError("Only canonical form element recipes carry a form identity.")

    def rebuild(self, /) -> FiniteElementSpec:
        dimension, form_degree, twist, fiber_shape, ambient_dimension, proxy = (
            self.value_spec
        )
        value_spec = FormValueSpec(
            FormType(
                dimension,
                form_degree,
                twist=parse(twist, FormTwist, "twist"),
                fiber_shape=fiber_shape,
                ambient_dimension=ambient_dimension,
            ),
            proxy=parse(proxy, FormProxy, "proxy"),
        )
        if self.recipe_kind == "form":
            if self.form_identity is None:
                raise ValueError("A form element recipe requires its form identity.")
            _, form_degree, order, family, twist = self.form_identity
            element = form_element(
                self.cell_kind,
                form_degree,
                order,
                family=parse(family, FormElementFamily, "family"),
                twist=parse(twist, FormTwist, "twist"),
                proxy=parse(proxy, FormProxy, "proxy"),
            )
        elif self.recipe_kind == "lagrange":
            element = lagrange_element(self.cell_kind, self.degree)
        elif self.recipe_kind == "discontinuous":
            element = discontinuous_element(self.cell_kind, self.degree)
        else:
            element = FiniteElementSpec(
                self.family,
                self.cell_kind,
                self.degree,
                np.asarray(self.reference_nodes, dtype=np.float64),
                self.entity_dofs,
                value_spec=value_spec,
                continuity=parse(self.continuity, ElementContinuity, "continuity"),
                representation=parse(
                    self.representation, FieldRepresentation, "representation"
                ),
                tabulator_id=self.tabulator_id,
            )
        if element.element_id != self.element_id:
            raise ValueError(
                "Accepted FE element recipe changes its canonical element ID."
            )
        return element


@final
class FiniteElementAcceptedFieldLayoutRecipe(StrictModule, NonTrainableState):
    """Callable-free typed recipe for one accepted finite-element field layout."""

    name: str = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    storage_kind: str = eqx.field(static=True)
    elements: tuple[FiniteElementAcceptedElementRecipe, ...]
    periodic_quotient: bool = eqx.field(static=True)
    transfer_semantics: FiniteElementFieldTransferSemantics = eqx.field(static=True)
    field_spec_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        block_names: tuple[str, ...],
        component_shape: tuple[int, ...],
        storage_kind: str,
        elements: tuple[FiniteElementAcceptedElementRecipe, ...],
        periodic_quotient: bool,
        transfer_semantics: FiniteElementFieldTransferSemantics,
        field_spec_id: str,
        layout_id: str,
        /,
    ) -> None:
        self.name = str(name)
        self.block_names = tuple(str(value) for value in block_names)
        self.component_shape = tuple(int(value) for value in component_shape)
        self.storage_kind = str(storage_kind)
        if not isinstance(periodic_quotient, (bool, np.bool_)):
            raise TypeError("periodic_quotient must be bool.")
        self.elements = tuple(elements)
        self.periodic_quotient = bool(periodic_quotient)
        self.transfer_semantics = parse(
            transfer_semantics,
            FiniteElementFieldTransferSemantics,
            "transfer_semantics",
        )
        self.field_spec_id = str(field_spec_id)
        self.layout_id = str(layout_id)
        if (
            not self.name
            or not self.field_spec_id
            or not self.layout_id
            or self.storage_kind not in ("canonical_dofs", "hp_cell_slots")
            or not self.elements
            or not all(
                isinstance(value, FiniteElementAcceptedElementRecipe)
                for value in self.elements
            )
        ):
            raise ValueError("Accepted FE layout recipe is incomplete.")

    def rebuild_field_spec(self, /) -> FiniteElementFieldSpec:
        elements = tuple(value.rebuild() for value in self.elements)
        if self.block_names:
            if len(self.block_names) != len(elements):
                raise ValueError(
                    "Accepted FE recipe block and element rows do not align."
                )
            assignments = dict(zip(self.block_names, elements, strict=True))
            field_spec = FiniteElementFieldSpec(
                self.name,
                assignments,
                component_shape=self.component_shape,
            )
        else:
            if len(elements) != 1:
                raise ValueError(
                    "Unscoped accepted FE recipes require one reference element."
                )
            field_spec = FiniteElementFieldSpec(
                self.name,
                elements[0],
                component_shape=self.component_shape,
            )
        if field_spec.field_spec_id != self.field_spec_id:
            raise ValueError("Accepted FE recipe changes its canonical field-spec ID.")
        return field_spec


def _accepted_element_recipe(
    element: FiniteElementSpec, /
) -> FiniteElementAcceptedElementRecipe:
    basis = element.form_basis
    if basis is not None:
        recipe_kind = "form"
        form_identity = (
            int(basis.dimension),
            int(basis.form_degree),
            int(basis.order),
            str(basis.family),
            str(basis.twist),
        )
    else:
        form_identity = None
        recipe_kind = "raw"
        for candidate_kind, factory in (
            ("lagrange", lagrange_element),
            ("discontinuous", discontinuous_element),
        ):
            try:
                candidate = factory(element.cell_kind, element.degree)
            except (TypeError, ValueError, NotImplementedError):
                continue
            if candidate.element_id == element.element_id:
                recipe_kind = candidate_kind
                break
        if recipe_kind == "raw" and (
            element.tabulator is not None or element.tabulator_id is not None
        ):
            raise ValueError(
                "Accepted FE layouts cannot persist an unregistered custom tabulator."
            )
    form = element.value_spec.form_type
    value_spec = (
        int(form.dimension),
        int(form.degree),
        str(form.twist),
        tuple(int(value) for value in form.fiber_shape),
        int(form.ambient_dimension),
        str(element.value_spec.proxy),
    )
    return FiniteElementAcceptedElementRecipe(
        recipe_kind,
        element.family,
        element.cell_kind,
        element.degree,
        element.continuity,
        element.representation,
        value_spec,
        form_identity,
        element.reference_nodes,
        element.entity_dofs,
        element.tabulator_id,
        element.element_id,
    )


@final
class FiniteElementAcceptedFieldLayout(StrictModule, NonTrainableState):
    """Accepted FE space, storage, and declared topology-transfer semantics."""

    field_spec: FiniteElementFieldSpec
    storage_kind: str = eqx.field(static=True)
    periodic_quotient: bool = eqx.field(static=True)
    transfer_semantics: FiniteElementFieldTransferSemantics = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_spec: FiniteElementFieldSpec,
        /,
        *,
        storage_kind: str = "canonical_dofs",
        periodic_quotient: bool | None = None,
        transfer_semantics: FiniteElementFieldTransferSemantics | None = None,
    ) -> None:
        if not isinstance(field_spec, FiniteElementFieldSpec):
            raise TypeError("field_spec must be FiniteElementFieldSpec.")
        if storage_kind not in ("canonical_dofs", "hp_cell_slots"):
            raise ValueError("storage_kind must be canonical_dofs or hp_cell_slots.")
        if periodic_quotient is not None and not isinstance(
            periodic_quotient, (bool, np.bool_)
        ):
            raise TypeError("periodic_quotient must be a bool when provided.")
        identities = tuple(
            None
            if element.form_basis is None
            else (
                int(element.form_basis.dimension),
                int(element.form_basis.form_degree),
                int(element.form_basis.order),
                str(element.form_basis.family),
                str(element.form_basis.twist),
            )
            for element in field_spec.elements
        )
        has_form_basis = any(value is not None for value in identities)
        if periodic_quotient is None and has_form_basis:
            raise ValueError(
                "Form layouts must explicitly declare lifted or quotient DOFs."
            )
        if transfer_semantics is None:
            if (
                has_form_basis
                and all(value is not None for value in identities)
                and len(set(identities)) == 1
            ):
                semantics = FiniteElementFieldTransferSemantics.NESTED_FE
            else:
                raise ValueError(
                    "Accepted fields must explicitly declare transfer_semantics "
                    "unless they have one canonical compatible form identity."
                )
        else:
            semantics = parse(
                transfer_semantics,
                FiniteElementFieldTransferSemantics,
                "transfer_semantics",
            )
        if (
            has_form_basis
            and semantics is not FiniteElementFieldTransferSemantics.NESTED_FE
        ):
            raise ValueError(
                "Finite-element form fields require nested_fe transfer semantics."
            )
        if semantics is FiniteElementFieldTransferSemantics.CONSERVATIVE_CELL_AVERAGE:
            if (
                storage_kind != "canonical_dofs"
                or has_form_basis
                or any(
                    element.continuity != "discontinuous"
                    or element.degree != 0
                    or element.local_dof_count != 1
                    for element in field_spec.elements
                )
            ):
                raise ValueError(
                    "conservative_cell_average requires one discontinuous degree-zero "
                    "DOF per cell."
                )
        quotient = bool(periodic_quotient) if periodic_quotient is not None else False
        if quotient and storage_kind != "canonical_dofs":
            raise ValueError("Periodic quotient layouts require canonical DOF storage.")
        if quotient and (
            field_spec.component_shape
            or any(value is None for value in identities)
            or len(set(identities)) != 1
        ):
            raise ValueError(
                "Periodic quotient layouts require one canonical compatible form identity."
            )
        self.field_spec = field_spec
        self.storage_kind = storage_kind
        self.transfer_semantics = semantics
        self.periodic_quotient = quotient
        self.layout_id = canonical_fingerprint(
            {
                "kind": "finite-element-accepted-field-layout",
                "field_spec": field_spec.field_spec_id,
                "storage_kind": storage_kind,
                "periodic_quotient": self.periodic_quotient,
                "form_identity": identities[0] if self.periodic_quotient else None,
                "transfer_semantics": semantics.value,
            }
        )

    @classmethod
    def hp_cell_slots(
        cls,
        field_spec: FiniteElementFieldSpec,
        /,
        *,
        transfer_semantics: FiniteElementFieldTransferSemantics,
    ) -> FiniteElementAcceptedFieldLayout:
        return cls(
            field_spec,
            storage_kind="hp_cell_slots",
            transfer_semantics=transfer_semantics,
        )

    @property
    def name(self) -> str:
        return self.field_spec.name

    @property
    def component_shape(self) -> tuple[int, ...]:
        return self.field_spec.component_shape

    @property
    def form_identity(self) -> tuple[int, int, int, FormElementFamily, FormTwist] | None:
        identities = tuple(
            None
            if element.form_basis is None
            else (
                element.form_basis.dimension,
                element.form_basis.form_degree,
                element.form_basis.order,
                element.form_basis.family,
                element.form_basis.twist,
            )
            for element in self.field_spec.elements
        )
        if any(value is None for value in identities) or len(set(identities)) != 1:
            return None
        return identities[0]

    def recipe(self, /) -> FiniteElementAcceptedFieldLayoutRecipe:
        return FiniteElementAcceptedFieldLayoutRecipe(
            self.field_spec.name,
            self.field_spec.block_names,
            self.field_spec.component_shape,
            self.storage_kind,
            tuple(
                _accepted_element_recipe(element) for element in self.field_spec.elements
            ),
            self.periodic_quotient,
            self.transfer_semantics,
            self.field_spec.field_spec_id,
            self.layout_id,
        )

    @classmethod
    def from_recipe(
        cls,
        recipe: FiniteElementAcceptedFieldLayoutRecipe,
        /,
    ) -> FiniteElementAcceptedFieldLayout:
        if not isinstance(recipe, FiniteElementAcceptedFieldLayoutRecipe):
            raise TypeError("recipe must be FiniteElementAcceptedFieldLayoutRecipe.")
        layout = cls(
            recipe.rebuild_field_spec(),
            storage_kind=recipe.storage_kind,
            periodic_quotient=recipe.periodic_quotient,
            transfer_semantics=recipe.transfer_semantics,
        )
        if layout.layout_id != recipe.layout_id:
            raise ValueError(
                "Accepted FE field layout recipe changes its canonical layout ID."
            )
        return layout


@final
class FiniteElementAcceptedState(StrictModule, NonTrainableState):
    """Immutable fields, declared FE layouts, and committed material data."""

    fields: tuple[Array, ...]
    field_layouts: tuple[FiniteElementAcceptedFieldLayout, ...]
    field_ownership: tuple[FiniteElementDofOwnership, ...] | None
    materials: MaterialTransaction | None
    time: Array
    step: int = eqx.field(static=True)
    schedule_cursor: int = eqx.field(static=True)
    state_version: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    accepted_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[ArrayLike],
        time: ArrayLike,
        step: int,
        topology_id: str,
        prepared_id: str,
        compilation_id: str,
        /,
        *,
        field_layouts: Sequence[FiniteElementAcceptedFieldLayout],
        materials: MaterialTransaction | None = None,
        schedule_cursor: int = 0,
        state_version: int = 0,
        field_ownership: Sequence[FiniteElementDofOwnership] | None = None,
    ) -> None:
        fields_ = tuple(jnp.asarray(value) for value in fields)
        layouts = tuple(field_layouts)
        ownership = None if field_ownership is None else tuple(field_ownership)
        time_ = jnp.asarray(time)
        step_ = int(step)
        cursor = int(schedule_cursor)
        version = int(state_version)
        topology = str(topology_id)
        prepared = str(prepared_id)
        compilation = str(compilation_id)
        if not fields_ or any(
            not jnp.issubdtype(value.dtype, jnp.inexact) for value in fields_
        ):
            raise ValueError("Accepted FE state requires one or more inexact fields.")
        if len(layouts) != len(fields_) or not all(
            isinstance(value, FiniteElementAcceptedFieldLayout) for value in layouts
        ):
            raise ValueError("Accepted FE fields require one explicit layout each.")
        if ownership is not None:
            if len(ownership) != len(fields_) or not all(
                isinstance(value, FiniteElementDofOwnership) for value in ownership
            ):
                raise ValueError(
                    "Owner-local FE fields require one explicit DOF ownership map each."
                )
            for value, layout, dof_ownership in zip(
                fields_, layouts, ownership, strict=True
            ):
                if (
                    layout.storage_kind != "canonical_dofs"
                    or dof_ownership.field_layout_id != layout.layout_id
                    or dof_ownership.topology_id != topology
                    or value.ndim == 0
                    or value.shape[0] != dof_ownership.dof_keys.shape[0]
                ):
                    raise ValueError(
                        "Accepted FE field rows disagree with their explicit DOF ownership."
                    )
            if (
                len(
                    {
                        (value.local_rank, value.part_count, value.distribution_id)
                        for value in ownership
                    }
                )
                != 1
            ):
                raise ValueError(
                    "Accepted FE fields must share one local rank and distribution."
                )
        if len({layout.name for layout in layouts}) != len(layouts):
            raise ValueError("Accepted FE field names must be unique.")
        for value, layout in zip(fields_, layouts, strict=True):
            if layout.storage_kind == "canonical_dofs":
                valid_shape = (
                    value.ndim >= 1 and tuple(value.shape[1:]) == layout.component_shape
                )
            else:
                valid_shape = (
                    value.ndim >= 2 and tuple(value.shape[2:]) == layout.component_shape
                )
            if not valid_shape:
                raise ValueError(
                    "Accepted FE field fibers disagree with their declared layouts."
                )
        if time_.shape != () or not bool(jnp.isfinite(time_)):
            raise ValueError("Accepted FE time must be one finite scalar.")
        if step_ < 0 or cursor < 0 or version < 0:
            raise ValueError("Accepted FE step, cursor, and version must be nonnegative.")
        if not topology or not prepared or not compilation:
            raise ValueError("Accepted FE identities must be non-empty.")
        if materials is not None and not isinstance(materials, MaterialTransaction):
            raise TypeError("materials must be FiniteElementMaterialTransaction or None.")
        self.fields = fields_
        self.field_layouts = layouts
        self.field_ownership = ownership
        self.materials = materials
        self.time = time_
        self.step = step_
        self.schedule_cursor = cursor
        self.state_version = version
        self.topology_id = topology
        self.prepared_id = prepared
        self.compilation_id = compilation
        accepted_identity = {
            "kind": "finite-element-accepted-state",
            "topology": topology,
            "prepared": prepared,
            "compilation": compilation,
            "step": step_,
            "cursor": cursor,
            "version": version,
            "field_layouts": [layout.layout_id for layout in layouts],
            "field_shapes": [list(value.shape) for value in fields_],
            "materials": (None if materials is None else materials.transaction_id),
        }
        if ownership is not None:
            accepted_identity["field_ownership"] = [
                value.ownership_id for value in ownership
            ]
        self.accepted_id = canonical_fingerprint(accepted_identity)

    @staticmethod
    def _renew_materials(
        value: MaterialTransaction | None,
    ) -> MaterialTransaction | None:
        if value is None:
            return None
        if type(value) is not MaterialTransaction:
            raise TypeError("Stored FE materials must be MaterialTransaction.")
        renewed_states = tuple(
            MaterialState(
                MaterialSiteId(state.site_id.key),
                state.model_id,
                state.committed,
                trial=state.trial,
                state_version=state.state_version,
            )
            for state in value.states
        )
        rebuilt = MaterialTransaction(renewed_states)
        if (
            rebuilt.layout_id != value.layout_id
            or rebuilt.transaction_id != value.transaction_id
        ):
            raise ValueError("Stored FE material transaction changed its identity.")
        return rebuilt

    @classmethod
    def from_stored_fields(
        cls,
        value: FiniteElementAcceptedState,
        /,
    ) -> FiniteElementAcceptedState:
        if type(value) is not cls:
            raise TypeError("Stored value must be FiniteElementAcceptedState.")
        layouts = tuple(
            FiniteElementAcceptedFieldLayout.from_recipe(layout.recipe())
            for layout in value.field_layouts
        )
        ownership = (
            None
            if value.field_ownership is None
            else tuple(
                FiniteElementDofOwnership.from_stored_fields(item)
                for item in value.field_ownership
            )
        )
        rebuilt = cls(
            value.fields,
            value.time,
            value.step,
            value.topology_id,
            value.prepared_id,
            value.compilation_id,
            field_layouts=layouts,
            materials=cls._renew_materials(value.materials),
            schedule_cursor=value.schedule_cursor,
            state_version=value.state_version,
            field_ownership=ownership,
        )
        if rebuilt.accepted_id != value.accepted_id:
            raise ValueError("Stored FE accepted state changed its canonical identity.")
        return rebuilt

    @classmethod
    def from_archive_recipe(
        cls,
        recipe: FiniteElementAcceptedStateArchiveRecipe,
        /,
    ) -> FiniteElementAcceptedState:
        from .._model._structure import FiniteElementAcceptedStateArchiveRecipe

        if not isinstance(recipe, FiniteElementAcceptedStateArchiveRecipe):
            raise TypeError("recipe must be FiniteElementAcceptedStateArchiveRecipe.")
        layout_recipes: list[FiniteElementAcceptedFieldLayoutRecipe] = []
        for value in recipe.field_layouts:
            if not isinstance(value, FiniteElementAcceptedFieldLayoutRecipe):
                raise TypeError("FE archive field layouts must be typed layout recipes.")
            layout_recipes.append(value)
        layouts = tuple(
            FiniteElementAcceptedFieldLayout.from_recipe(value)
            for value in layout_recipes
        )
        ownership: tuple[FiniteElementDofOwnership, ...] | None = None
        if recipe.field_ownership is not None:
            ownership_recipes: list[FiniteElementDofOwnership] = []
            for value in recipe.field_ownership:
                if not isinstance(value, FiniteElementDofOwnership):
                    raise TypeError(
                        "FE archive ownership must contain typed ownership recipes."
                    )
                ownership_recipes.append(value)
            ownership = tuple(
                FiniteElementDofOwnership.from_stored_fields(value)
                for value in ownership_recipes
            )
        materials = recipe.materials
        if materials is not None and not isinstance(materials, MaterialTransaction):
            raise TypeError("FE archive materials must be MaterialTransaction or None.")
        rebuilt = cls(
            tuple(jnp.asarray(value) for value in recipe.fields),
            jnp.asarray(recipe.time),
            recipe.step,
            recipe.topology_id,
            recipe.prepared_id,
            recipe.compilation_id,
            field_layouts=layouts,
            materials=cls._renew_materials(materials),
            schedule_cursor=recipe.schedule_cursor,
            state_version=recipe.state_version,
            field_ownership=ownership,
        )
        if rebuilt.accepted_id != recipe.expected_accepted_id:
            raise ValueError("FE accepted-state archive recipe changed its identity.")
        return rebuilt

    def checkpoint(self, /) -> FiniteElementCheckpoint:
        return FiniteElementCheckpoint(
            self.prepared_id,
            self.compilation_id,
            self.time,
            self.step,
            self.fields,
            materials=self.materials,
        )


class FiniteElementAttemptResult(StrictModule):
    """Candidate state and explicit accept/retry evidence for one attempt."""

    fields: tuple[Array, ...]
    materials: MaterialTransaction | None
    accepted: Array
    retry_requested: Array
    suggested_step: Array
    diagnostics: object

    def __init__(
        self,
        fields: Sequence[ArrayLike],
        accepted: ArrayLike,
        /,
        *,
        materials: MaterialTransaction | None = None,
        retry_requested: ArrayLike = False,
        suggested_step: ArrayLike = 0.0,
        diagnostics: object = None,
    ) -> None:
        fields_ = tuple(jnp.asarray(value) for value in fields)
        accepted_ = jnp.asarray(accepted, dtype=jnp.bool_)
        retry = jnp.asarray(retry_requested, dtype=jnp.bool_)
        suggested = jnp.asarray(suggested_step)
        if not fields_:
            raise ValueError("Finite-element attempts require candidate fields.")
        if accepted_.shape != () or retry.shape != () or suggested.shape != ():
            raise ValueError("Attempt decision and suggested step must be scalars.")
        if materials is not None and not isinstance(materials, MaterialTransaction):
            raise TypeError("materials must be FiniteElementMaterialTransaction or None.")
        self.fields = fields_
        self.materials = materials
        self.accepted = accepted_
        self.retry_requested = retry
        self.suggested_step = suggested
        self.diagnostics = diagnostics


class FiniteElementStepPolicy(StrictModule, NonTrainableState):
    minimum_step: float = eqx.field(static=True)
    reduction: float = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        minimum_step: float = 1.0e-8,
        reduction: float = 0.5,
        maximum_retries: int = 8,
    ) -> None:
        minimum = float(minimum_step)
        reduction_ = float(reduction)
        retries = int(maximum_retries)
        if minimum <= 0.0 or not 0.0 < reduction_ < 1.0 or retries < 0:
            raise ValueError("Accepted-step retry policy is invalid.")
        self.minimum_step = minimum
        self.reduction = reduction_
        self.maximum_retries = retries
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-step-policy",
                "minimum_step": minimum,
                "reduction": reduction_,
                "maximum_retries": retries,
            }
        )


class FiniteElementStepDiagnostics(StrictModule):
    accepted: Array
    attempts: Array
    rejected_attempts: Array
    attempted_step: Array
    accepted_step: Array
    terminal_reason: str = eqx.field(static=True)


class FiniteElementAcceptedStepSchedule(StrictModule, NonTrainableState):
    """Host transaction that promotes only an explicitly accepted FE candidate."""

    solve_attempt: Callable
    policy: FiniteElementStepPolicy
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        solve_attempt: Callable,
        /,
        *,
        policy: FiniteElementStepPolicy | None = None,
        schedule_id: str = "finite-element-accepted-step",
    ) -> None:
        if not callable(solve_attempt):
            raise TypeError("solve_attempt must be callable.")
        policy_ = FiniteElementStepPolicy() if policy is None else policy
        if not isinstance(policy_, FiniteElementStepPolicy):
            raise TypeError("policy must be FiniteElementStepPolicy or None.")
        identifier = str(schedule_id)
        if not identifier:
            raise ValueError("schedule_id must be non-empty.")
        self.solve_attempt = solve_attempt
        self.policy = policy_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "finite-element-accepted-step-schedule",
                "declared_id": identifier,
                "policy": policy_.policy_id,
            }
        )

    @checked
    def advance(
        self,
        accepted: FiniteElementAcceptedState,
        end_time: ArrayLike,
        time_law: TimeLaw,
        args: object = None,
        /,
    ) -> tuple[FiniteElementAcceptedState, FiniteElementStepDiagnostics]:
        target = float(jnp.asarray(end_time))
        start = float(accepted.time)
        if not jnp.isfinite(target) or target <= start:
            raise ValueError("Accepted-step target time must be finite and increasing.")
        attempted_step = target - start
        step_size = attempted_step
        rejected = 0
        for attempt in range(self.policy.maximum_retries + 1):
            candidate_end = start + step_size
            result = self.solve_attempt(
                accepted,
                start,
                candidate_end,
                time_law,
                args,
            )
            if not isinstance(result, FiniteElementAttemptResult):
                raise TypeError("solve_attempt must return FiniteElementAttemptResult.")
            if tuple(value.shape for value in result.fields) != tuple(
                value.shape for value in accepted.fields
            ):
                raise ValueError("Candidate fields must preserve accepted field shapes.")
            if bool(result.accepted):
                materials = (
                    None if result.materials is None else result.materials.commit()
                )
                promoted = FiniteElementAcceptedState(
                    result.fields,
                    candidate_end,
                    accepted.step + 1,
                    accepted.topology_id,
                    accepted.prepared_id,
                    accepted.compilation_id,
                    field_layouts=accepted.field_layouts,
                    materials=materials,
                    schedule_cursor=accepted.schedule_cursor + 1,
                    state_version=accepted.state_version + 1,
                    field_ownership=accepted.field_ownership,
                )
                return promoted, FiniteElementStepDiagnostics(
                    accepted=jnp.asarray(True),
                    attempts=jnp.asarray(attempt + 1, dtype=jnp.int32),
                    rejected_attempts=jnp.asarray(rejected, dtype=jnp.int32),
                    attempted_step=jnp.asarray(attempted_step),
                    accepted_step=jnp.asarray(step_size),
                    terminal_reason="accepted",
                )
            rejected += 1
            if not bool(result.retry_requested):
                break
            suggested = float(result.suggested_step)
            step_size = (
                suggested
                if jnp.isfinite(suggested) and 0.0 < suggested < step_size
                else self.policy.reduction * step_size
            )
            if step_size < self.policy.minimum_step:
                break
        return accepted, FiniteElementStepDiagnostics(
            accepted=jnp.asarray(False),
            attempts=jnp.asarray(rejected, dtype=jnp.int32),
            rejected_attempts=jnp.asarray(rejected, dtype=jnp.int32),
            attempted_step=jnp.asarray(attempted_step),
            accepted_step=jnp.asarray(0.0),
            terminal_reason="rejected",
        )


class FiniteElementRestartStateRole(StrEnum):
    """Semantic owner role of one named FE history payload."""

    HISTORY = "history"
    RNG = "rng"
    STATISTICS = "statistics"
    MODEL_STATE = "model-state"
    PHYSICAL_STATE = "physical-state"


class FiniteElementRestartStateAssociation(StrEnum):
    """Explicit placement semantics for a named FE history payload."""

    OWNED_ROWS = "owned_rows"
    REPLICATED = "replicated"


@final
class FiniteElementRestartStateLayout(StrictModule, NonTrainableState):
    """Declared role and row-ownership contract for one restart payload."""

    name: str = eqx.field(static=True)
    role: FiniteElementRestartStateRole = eqx.field(static=True)
    association: FiniteElementRestartStateAssociation = eqx.field(static=True)
    ownership: FiniteElementDofOwnership | None
    prng_impl: str | None = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: FiniteElementRestartStateRole,
        association: FiniteElementRestartStateAssociation,
        /,
        *,
        ownership: FiniteElementDofOwnership | None = None,
        prng_impl: str | None = None,
    ) -> None:
        name_ = str(name)
        role_ = parse(role, FiniteElementRestartStateRole, "role")
        association_ = parse(
            association, FiniteElementRestartStateAssociation, "association"
        )
        if not name_:
            raise ValueError("Restart state layout name must be non-empty.")
        if association_ is FiniteElementRestartStateAssociation.OWNED_ROWS:
            if not isinstance(ownership, FiniteElementDofOwnership):
                raise TypeError("Owned restart rows require explicit FE ownership.")
        elif ownership is not None:
            raise ValueError("Replicated restart state cannot carry row ownership.")
        implementation = None if prng_impl is None else str(prng_impl).strip()
        if implementation is not None and (
            not implementation or role_ is not FiniteElementRestartStateRole.RNG
        ):
            raise ValueError("Only RNG state layouts may declare a PRNG implementation.")
        self.name = name_
        self.role = role_
        self.association = association_
        self.ownership = ownership
        self.prng_impl = implementation
        self.layout_id = canonical_fingerprint(
            {
                "kind": "finite-element-restart-state-layout",
                "name": name_,
                "role": role_.value,
                "association": association_.value,
                "identity": None if ownership is None else ownership.identity_id,
                "prng_impl": implementation,
            }
        )

    def validate_value(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        typed = jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
        if typed and (
            self.role is not FiniteElementRestartStateRole.RNG
            or self.prng_impl is None
            or str(jax.random.key_impl(array)) != self.prng_impl
        ):
            raise ValueError(
                "Typed PRNG payloads require their declared RNG implementation."
            )
        if not typed and self.prng_impl is not None:
            raise ValueError("A PRNG implementation was declared for a non-typed key.")
        if self.ownership is not None and (
            array.ndim == 0 or array.shape[0] != self.ownership.dof_keys.shape[0]
        ):
            raise ValueError("Restart state rows disagree with their declared ownership.")
        return array

    @classmethod
    def from_stored_fields(
        cls,
        value: FiniteElementRestartStateLayout,
        /,
    ) -> FiniteElementRestartStateLayout:
        if type(value) is not cls:
            raise TypeError(
                "Stored restart layout must be FiniteElementRestartStateLayout."
            )
        ownership = value.ownership
        if ownership is not None:
            ownership = FiniteElementDofOwnership.from_stored_fields(ownership)
        rebuilt = cls(
            value.name,
            value.role,
            value.association,
            ownership=ownership,
            prng_impl=value.prng_impl,
        )
        if rebuilt.layout_id != value.layout_id:
            raise ValueError("Stored restart state layout changed its canonical ID.")
        return rebuilt


class FiniteElementRestartManifest(StrictModule, NonTrainableState):
    """Canonical fixed-topology accepted state plus named runtime histories."""

    state: FiniteElementAcceptedState
    auxiliary_state: tuple[tuple[str, Array], ...]
    integrator_state: tuple[tuple[str, Array], ...]
    auxiliary_layouts: tuple[FiniteElementRestartStateLayout, ...] | None
    integrator_layouts: tuple[FiniteElementRestartStateLayout, ...] | None
    material_layouts: tuple[FiniteElementRestartStateLayout, ...] | None
    manifest_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        state: FiniteElementAcceptedState,
        /,
        *,
        auxiliary_state: Sequence[tuple[str, ArrayLike]] = (),
        integrator_state: Sequence[tuple[str, ArrayLike]] = (),
        auxiliary_layouts: Sequence[FiniteElementRestartStateLayout] | None = None,
        integrator_layouts: Sequence[FiniteElementRestartStateLayout] | None = None,
        material_layouts: Sequence[FiniteElementRestartStateLayout] | None = None,
    ) -> None:

        def named(
            values: Sequence[tuple[str, ArrayLike]],
        ) -> tuple[tuple[str, Array], ...]:
            result = tuple((str(name), jnp.asarray(value)) for name, value in values)
            names = tuple(name for name, _ in result)
            if len(set(names)) != len(names) or any(not name for name in names):
                raise ValueError("Restart state names must be non-empty and unique.")
            return result

        auxiliary = named(auxiliary_state)
        integrator = named(integrator_state)

        def layouts(
            declared: Sequence[FiniteElementRestartStateLayout] | None,
            payload: tuple[tuple[str, Array], ...],
            name: str,
        ) -> tuple[FiniteElementRestartStateLayout, ...] | None:
            if declared is None:
                return None
            result = tuple(declared)
            if len(result) != len(payload) or not all(
                isinstance(value, FiniteElementRestartStateLayout) for value in result
            ):
                raise ValueError(
                    f"{name} restart payloads require one declared state layout each."
                )
            for (payload_name, payload_value), layout in zip(
                payload, result, strict=True
            ):
                if layout.name != payload_name:
                    raise ValueError(
                        f"{name} restart payload names disagree with their state layouts."
                    )
                layout.validate_value(payload_value)
                ownership = layout.ownership
                if ownership is not None and (
                    ownership.topology_id != state.topology_id
                    or ownership.distribution_id
                    != (
                        None
                        if state.field_ownership is None
                        else state.field_ownership[0].distribution_id
                    )
                ):
                    raise ValueError(
                        "Restart payload ownership differs from accepted FE placement."
                    )
            return result

        auxiliary_layouts_ = layouts(auxiliary_layouts, auxiliary, "Auxiliary")
        integrator_layouts_ = layouts(integrator_layouts, integrator, "Integrator")
        material_states = () if state.materials is None else state.materials.states
        material_payload = tuple(
            (
                material.site_id.key,
                jnp.stack((material.committed, material.trial))[None, ...],
            )
            for material in material_states
        )
        material_layouts_ = layouts(material_layouts, material_payload, "Material")
        if material_layouts_ is not None and any(
            layout.role is not FiniteElementRestartStateRole.MODEL_STATE
            for layout in material_layouts_
        ):
            raise ValueError("Material restart layouts require model-state role.")
        typed_auxiliary = any(
            jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            for _, value in auxiliary
        )
        typed_integrator = any(
            jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            for _, value in integrator
        )
        if (typed_auxiliary and auxiliary_layouts_ is None) or (
            typed_integrator and integrator_layouts_ is None
        ):
            raise ValueError("Typed PRNG restart payloads require explicit RNG layouts.")
        identity = {
            "kind": "finite-element-restart-manifest",
            "accepted": state.accepted_id,
            "auxiliary": [
                [name, list(value.shape), str(value.dtype)] for name, value in auxiliary
            ],
            "integrator": [
                [name, list(value.shape), str(value.dtype)] for name, value in integrator
            ],
        }
        if auxiliary_layouts_ is not None:
            identity["auxiliary_layouts"] = [
                value.layout_id for value in auxiliary_layouts_
            ]
        if integrator_layouts_ is not None:
            identity["integrator_layouts"] = [
                value.layout_id for value in integrator_layouts_
            ]
        if material_layouts_ is not None:
            identity["material_layouts"] = [
                value.layout_id for value in material_layouts_
            ]
        self.state = state
        self.auxiliary_state = auxiliary
        self.integrator_state = integrator
        self.auxiliary_layouts = auxiliary_layouts_
        self.integrator_layouts = integrator_layouts_
        self.material_layouts = material_layouts_
        self.manifest_id = canonical_fingerprint(identity)

    @classmethod
    def from_stored_fields(
        cls,
        value: FiniteElementRestartManifest,
        /,
    ) -> FiniteElementRestartManifest:
        if type(value) is not cls:
            raise TypeError("Stored value must be FiniteElementRestartManifest.")
        from .._model._structure import FiniteElementAcceptedStateArchiveRecipe

        state = value.state
        if type(state) is FiniteElementAcceptedState:
            rebuilt_state = FiniteElementAcceptedState.from_stored_fields(state)
        elif type(state) is FiniteElementAcceptedStateArchiveRecipe:
            rebuilt_state = FiniteElementAcceptedState.from_archive_recipe(state)
        else:
            raise TypeError(
                "Stored restart manifest has an unknown accepted-state recipe."
            )

        def renew_layouts(
            layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
        ) -> tuple[FiniteElementRestartStateLayout, ...] | None:
            return (
                None
                if layouts is None
                else tuple(
                    FiniteElementRestartStateLayout.from_stored_fields(item)
                    for item in layouts
                )
            )

        auxiliary_layouts = renew_layouts(value.auxiliary_layouts)
        integrator_layouts = renew_layouts(value.integrator_layouts)
        material_layouts = renew_layouts(value.material_layouts)

        def renew_values(
            values: tuple[tuple[str, Array], ...],
            layouts: tuple[FiniteElementRestartStateLayout, ...] | None,
        ) -> tuple[tuple[str, Array], ...]:
            if layouts is None:
                if any(
                    jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
                    for _, array in values
                ):
                    raise ValueError(
                        "Stored typed PRNG values require an explicit RNG layout."
                    )
                return values
            rebuilt = []
            for (name, value_), layout in zip(values, layouts, strict=True):
                if layout.prng_impl is not None and not jax.dtypes.issubdtype(
                    value_.dtype, jax.dtypes.prng_key
                ):
                    value_ = jax.random.wrap_key_data(value_, impl=layout.prng_impl)
                rebuilt.append((name, layout.validate_value(value_)))
            return tuple(rebuilt)

        auxiliary = renew_values(value.auxiliary_state, auxiliary_layouts)
        integrator = renew_values(value.integrator_state, integrator_layouts)
        rebuilt = cls(
            rebuilt_state,
            auxiliary_state=auxiliary,
            integrator_state=integrator,
            auxiliary_layouts=auxiliary_layouts,
            integrator_layouts=integrator_layouts,
            material_layouts=material_layouts,
        )
        if rebuilt.manifest_id != value.manifest_id:
            raise ValueError("Stored FE restart manifest changed its canonical identity.")
        return rebuilt


def _restart_ownership_record(
    ownership: FiniteElementDofOwnership | None,
    name: str,
    arrays: dict[str, np.ndarray],
    /,
) -> dict[str, object] | None:
    if ownership is None:
        return None
    key_name = f"{name}_keys"
    owner_name = f"{name}_owners"
    arrays[key_name] = np.asarray(ownership.dof_keys)
    arrays[owner_name] = np.asarray(ownership.owner_ranks)
    return {
        "keys": key_name,
        "owners": owner_name,
        "field_layout_id": ownership.field_layout_id,
        "identity_id": ownership.identity_id,
        "orientation_action_id": ownership.orientation_action_id,
        "distribution_id": ownership.distribution_id,
        "topology_id": ownership.topology_id,
        "geometry_id": ownership.geometry_id,
        "local_rank": ownership.local_rank,
        "part_count": ownership.part_count,
    }


def _restart_array_record(
    name: str,
    value: ArrayLike,
    arrays: dict[str, np.ndarray],
    typed_keys: dict[str, str],
    /,
) -> None:
    array = jnp.asarray(value)
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        arrays[name] = np.asarray(jax.random.key_data(array))
        typed_keys[name] = str(jax.random.key_impl(array))
    else:
        arrays[name] = np.asarray(array)


def _string_keyed_record(value: object, name: str, /) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a string-keyed record.")
    record: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
        record[key] = item
    return record


def _restore_restart_array(
    name: str,
    arrays: Mapping[str, ArrayLike],
    typed_keys: Mapping[str, str],
    /,
) -> Array:
    value = arrays[name]
    if name in typed_keys:
        return jax.random.wrap_key_data(jnp.asarray(value), impl=typed_keys[name])
    return jnp.asarray(value)


def _restore_ownership(
    record: dict[str, object] | None,
    arrays: Mapping[str, ArrayLike],
    /,
) -> FiniteElementDofOwnership | None:
    if record is None:
        return None
    local_rank = record["local_rank"]
    part_count = record["part_count"]
    if (
        isinstance(local_rank, bool)
        or not isinstance(local_rank, int)
        or isinstance(part_count, bool)
        or not isinstance(part_count, int)
    ):
        raise ValueError("FE ownership rank/count must be integer archive fields.")
    return FiniteElementDofOwnership(
        arrays[str(record["keys"])],
        arrays[str(record["owners"])],
        field_layout_id=str(record["field_layout_id"]),
        identity_id=str(record["identity_id"]),
        orientation_action_id=str(record["orientation_action_id"]),
        distribution_id=str(record["distribution_id"]),
        topology_id=str(record["topology_id"]),
        geometry_id=str(record["geometry_id"]),
        local_rank=local_rank,
        part_count=part_count,
    )


def _restart_state_layout_record(
    layout: FiniteElementRestartStateLayout,
    name: str,
    arrays: dict[str, np.ndarray],
    /,
) -> dict[str, object]:
    return {
        "name": layout.name,
        "role": layout.role.value,
        "association": layout.association.value,
        "layout_id": layout.layout_id,
        "prng_impl": layout.prng_impl,
        "ownership": _restart_ownership_record(layout.ownership, name, arrays),
    }


def _restore_state_layout(
    record: dict[str, object],
    arrays: Mapping[str, ArrayLike],
    /,
) -> FiniteElementRestartStateLayout:
    ownership_record = record["ownership"]
    if ownership_record is None:
        ownership = None
    elif isinstance(ownership_record, dict):
        ownership = _restore_ownership(
            _string_keyed_record(ownership_record, "restart ownership"),
            arrays,
        )
    else:
        raise ValueError("Restart ownership must be a record or None.")
    layout = FiniteElementRestartStateLayout(
        str(record["name"]),
        parse(
            str(record["role"]),
            FiniteElementRestartStateRole,
            "restart state role",
        ),
        parse(
            str(record["association"]),
            FiniteElementRestartStateAssociation,
            "restart state association",
        ),
        ownership=ownership,
        prng_impl=(None if record["prng_impl"] is None else str(record["prng_impl"])),
    )
    if layout.layout_id != record["layout_id"]:
        raise ValueError("Restart state layout recipe changes its canonical ID.")
    return layout


def _restore_state_layouts(
    records: object,
    arrays: Mapping[str, ArrayLike],
    name: str,
    /,
) -> tuple[FiniteElementRestartStateLayout, ...] | None:
    if records is None:
        return None
    if not isinstance(records, list):
        raise ValueError(f"{name} must be an array of restart state layouts.")
    return tuple(
        _restore_state_layout(_string_keyed_record(record, name), arrays)
        for record in records
    )


def write_finite_element_restart(
    path: str | Path,
    manifest: FiniteElementRestartManifest,
    /,
) -> None:
    if not isinstance(manifest, FiniteElementRestartManifest):
        raise TypeError("manifest must be FiniteElementRestartManifest.")
    material_states = (
        () if manifest.state.materials is None else manifest.state.materials.states
    )
    arrays: dict[str, np.ndarray] = {}
    typed_keys: dict[str, str] = {}
    _restart_array_record("time", manifest.state.time, arrays, typed_keys)
    for index, value in enumerate(manifest.state.fields):
        _restart_array_record(f"field_{index}", value, arrays, typed_keys)
    for index, value in enumerate(material_states):
        _restart_array_record(f"material_{index}", value.committed, arrays, typed_keys)
        _restart_array_record(f"material_trial_{index}", value.trial, arrays, typed_keys)
    for index, (_, value) in enumerate(manifest.auxiliary_state):
        _restart_array_record(f"auxiliary_{index}", value, arrays, typed_keys)
    for index, (_, value) in enumerate(manifest.integrator_state):
        _restart_array_record(f"integrator_{index}", value, arrays, typed_keys)
    metadata = {
        "kind": "finite-element-restart",
        "topology_id": manifest.state.topology_id,
        "step": manifest.state.step,
        "schedule_cursor": manifest.state.schedule_cursor,
        "state_version": manifest.state.state_version,
        "field_count": len(manifest.state.fields),
        "field_layout_ids": [layout.layout_id for layout in manifest.state.field_layouts],
        "materials": [
            [value.site_id.key, value.model_id, value.state_version]
            for value in material_states
        ],
        "auxiliary_names": [name for name, _ in manifest.auxiliary_state],
        "integrator_names": [name for name, _ in manifest.integrator_state],
    }
    if manifest.state.field_ownership is not None:
        metadata["field_ownership"] = [
            _restart_ownership_record(value, f"field_ownership_{index}", arrays)
            for index, value in enumerate(manifest.state.field_ownership)
        ]
    if manifest.auxiliary_layouts is not None:
        metadata["auxiliary_layouts"] = [
            _restart_state_layout_record(layout, f"auxiliary_layout_{index}", arrays)
            for index, layout in enumerate(manifest.auxiliary_layouts)
        ]
    if manifest.integrator_layouts is not None:
        metadata["integrator_layouts"] = [
            _restart_state_layout_record(layout, f"integrator_layout_{index}", arrays)
            for index, layout in enumerate(manifest.integrator_layouts)
        ]
    if manifest.material_layouts is not None:
        metadata["material_layouts"] = [
            _restart_state_layout_record(layout, f"material_layout_{index}", arrays)
            for index, layout in enumerate(manifest.material_layouts)
        ]
    if typed_keys:
        metadata["typed_keys"] = typed_keys
    metadata_text = json.dumps(metadata, allow_nan=False, sort_keys=True)
    shards = tuple(
        CheckpointShard(
            name,
            array_payload_digest(value),
            array_payload_byte_count(value),
            (
                manifest.state.topology_id,
                manifest.state.prepared_id,
                manifest.state.compilation_id,
            ),
            metadata={"finite_element_restart": metadata_text} if index == 0 else {},
        )
        for index, (name, value) in enumerate(sorted(arrays.items()))
    )
    lifecycle_manifest = CheckpointManifest(
        manifest.manifest_id,
        manifest.state.prepared_id,
        array_collection_digest(arrays),
        manifest.state.compilation_id,
        shards,
        complete=True,
    )
    create_lifecycle_archive(path, manifest=lifecycle_manifest, arrays=arrays)


def read_finite_element_restart(
    path: str | Path,
    /,
    *,
    field_layouts: Sequence[FiniteElementAcceptedFieldLayout],
) -> FiniteElementRestartManifest:
    archive = open_lifecycle_archive(path)
    lifecycle_manifest = archive.manifest
    if not isinstance(lifecycle_manifest, CheckpointManifest):
        raise ValueError("Finite-element restart is not a lifecycle checkpoint.")
    metadata_text = next(
        (
            dict(shard.metadata)["finite_element_restart"]
            for shard in lifecycle_manifest.shards
            if "finite_element_restart" in dict(shard.metadata)
        ),
        None,
    )
    if metadata_text is None:
        raise ValueError("Finite-element restart metadata is missing.")
    metadata = json.loads(metadata_text)
    arrays = dict(archive.arrays)
    typed_keys = {
        str(name): str(implementation)
        for name, implementation in metadata.get("typed_keys", {}).items()
    }
    fields = tuple(
        arrays[f"field_{index}"] for index in range(int(metadata["field_count"]))
    )
    layouts = tuple(field_layouts)
    layout_ids = tuple(
        layout.layout_id
        for layout in layouts
        if isinstance(layout, FiniteElementAcceptedFieldLayout)
    )
    if len(layout_ids) != len(layouts) or layout_ids != tuple(
        metadata["field_layout_ids"]
    ):
        raise ValueError(
            "Finite-element restart requires matching explicit field layouts."
        )
    field_ownership_records = metadata.get("field_ownership")
    field_ownership: tuple[FiniteElementDofOwnership, ...] | None = None
    if field_ownership_records is not None:
        if not isinstance(field_ownership_records, list):
            raise ValueError("Finite-element field ownership must be an archive array.")
        restored_ownership = tuple(
            _restore_ownership(
                _string_keyed_record(record, "finite-element field ownership"),
                arrays,
            )
            for record in field_ownership_records
        )
        if any(value is None for value in restored_ownership):
            raise ValueError("Finite-element ownership records cannot be null.")
        field_ownership = tuple(
            value for value in restored_ownership if value is not None
        )
    material_states_: list[MaterialState] = []
    for index, entry in enumerate(metadata["materials"]):
        if len(entry) == 2:
            material_id, version = entry
            model_id = material_id
        elif len(entry) == 3:
            material_id, model_id, version = entry
        else:
            raise ValueError("Finite-element material restart metadata is invalid.")
        committed = arrays[f"material_{index}"]
        trial = arrays.get(f"material_trial_{index}", committed)
        material_states_.append(
            MaterialState(
                MaterialSiteId(material_id),
                model_id,
                committed,
                trial=trial,
                state_version=int(version),
            )
        )
    material_states = tuple(material_states_)
    materials = None if not material_states else MaterialTransaction(material_states)
    state = FiniteElementAcceptedState(
        fields,
        _restore_restart_array("time", arrays, typed_keys),
        int(metadata["step"]),
        metadata["topology_id"],
        lifecycle_manifest.analysis_plan_id,
        lifecycle_manifest.execution_plan_id,
        field_layouts=layouts,
        materials=materials,
        schedule_cursor=int(metadata["schedule_cursor"]),
        state_version=int(metadata["state_version"]),
        field_ownership=field_ownership,
    )
    auxiliary = tuple(
        (
            name,
            _restore_restart_array(f"auxiliary_{index}", arrays, typed_keys),
        )
        for index, name in enumerate(metadata["auxiliary_names"])
    )
    integrator = tuple(
        (
            name,
            _restore_restart_array(f"integrator_{index}", arrays, typed_keys),
        )
        for index, name in enumerate(metadata["integrator_names"])
    )
    auxiliary_layouts = metadata.get("auxiliary_layouts")
    material_layouts = metadata.get("material_layouts")
    integrator_layouts = metadata.get("integrator_layouts")
    manifest = FiniteElementRestartManifest(
        state,
        auxiliary_state=auxiliary,
        integrator_state=integrator,
        auxiliary_layouts=_restore_state_layouts(
            auxiliary_layouts, arrays, "auxiliary layouts"
        ),
        integrator_layouts=_restore_state_layouts(
            integrator_layouts, arrays, "integrator layouts"
        ),
        material_layouts=_restore_state_layouts(
            material_layouts, arrays, "material layouts"
        ),
    )
    if manifest.manifest_id != lifecycle_manifest.checkpoint_id:
        raise ValueError("Finite-element restart manifest identity mismatch.")
    return manifest


__all__ = [
    "FiniteElementFieldTransferSemantics",
    "FiniteElementAcceptedFieldLayout",
    "FiniteElementAcceptedFieldLayoutRecipe",
    "FiniteElementAcceptedElementRecipe",
    "FiniteElementAcceptedState",
    "FiniteElementAcceptedStepSchedule",
    "FiniteElementAttemptResult",
    "FiniteElementRestartManifest",
    "FiniteElementRestartStateAssociation",
    "FiniteElementRestartStateLayout",
    "FiniteElementRestartStateRole",
    "FiniteElementStepPolicy",
    "read_finite_element_restart",
    "write_finite_element_restart",
]
