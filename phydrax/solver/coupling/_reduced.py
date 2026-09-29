#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Galerkin reduced-order models of coupled components.

A region of a spatial coupled problem is replaced by a prepared reduced model
without changing any binding, law, or observation: the reduced component keeps
the full component's name, field, discrete field space, boundary impositions,
side traces, conormal fluxes, and pointwise reconstruction, and publishes a
reduced state block whose field chart is the owner's own chart composed with the
fixed trial basis. Laws therefore read the reconstructed field ``P (V a + l) +
g`` and inject into the reduced rows through ``V^T P^T`` exactly once, as for any
other component. The reduced equations are the ROM owner's
``FullResidualGalerkin`` projection of the original component residual, so the
reduced model is full-order assisted: every residual and operator evaluation
calls the original owner.
"""

from __future__ import annotations

from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._trainable import NonTrainableState, resolve_array_roles
from ..._validation import canonical_identifier
from ...discretization import (
    BoundaryImposition,
    FacetTraceRule,
    IntegrationDomain,
    PreparedFieldReconstruction,
    PreparedFluxAction,
    PreparedTraceAction,
    SideTraceQuantity,
    TraceInverseEvidence,
)
from ...discretization._views import FieldTraceSide
from ...linalg import (
    AbstractVectorSpace,
    BlockLinearOperator,
    compose_constraint_maps,
    ConstraintMap,
    DenseLinearOperator,
    DualSpace,
)
from ...rom import AbstractResidualProvider, FullResidualGalerkin
from ._components import (
    AbstractReconstructionComponent,
    AbstractTraceComponent,
    ComponentBlock,
    ComponentField,
    ComponentSpace,
)


@final
class ComponentResidualProvider(AbstractResidualProvider):
    """The steady residual of one single-field coupled component as a ROM provider.

    ``residual(coordinate, state, state_rate, inputs)`` evaluates the component's
    original residual on its one row block at the owner state ``state`` (the
    component's solve coordinates) with the component's runtime ``inputs``; the
    rows are identified with the coordinate dual of the state block, which the
    component pairs with them. ``support_id`` is the field's discrete space and
    ``geometry_id`` the owner's identity, so a reduced basis built on the state
    space of another owner, mesh, or field refuses to reduce this residual.
    """

    component: AbstractTraceComponent
    field: str = eqx.field(static=True)
    state_space: AbstractVectorSpace
    residual_space: AbstractVectorSpace
    residual_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self, component: AbstractTraceComponent, /, *, field: str) -> None:
        if not isinstance(component, AbstractTraceComponent):
            raise TypeError(
                "A reduced coupled region reduces a component that publishes side "
                "traces (an AbstractTraceComponent)."
            )
        field_ = canonical_identifier(field, "field")
        record = component.field(field_)
        if len(component.state_blocks) != 1 or len(component.row_blocks) != 1:
            raise ValueError(
                "A component residual provider reduces a single-block component; "
                f"{component.name!r} has {len(component.state_blocks)} state and "
                f"{len(component.row_blocks)} row blocks."
            )
        state_space = component.state_blocks[0].space
        rows = component.row_blocks[0].space
        if (
            record.state_block != component.state_blocks[0].name
            or record.row_block != component.row_blocks[0].name
        ):
            raise ValueError(f"Field {field_!r} does not own the component's blocks.")
        if eqx.tree_equal(rows.structure(), state_space.structure()) is not True:
            raise ValueError(
                "The component's residual rows do not pair with its state coordinates."
            )
        self.component = component
        self.field = field_
        self.state_space = state_space
        self.residual_space = DualSpace(state_space)
        self.support_id = component.field_space_id(field_)
        self.geometry_id = component.owner_id
        self.residual_id = canonical_fingerprint(
            {
                "kind": "coupled-component-residual",
                "component": component.name,
                "owner": component.owner_id,
                "field": field_,
                "state": state_space.space_id,
            }
        )

    def residual(
        self,
        coordinate: Array,
        state: Any,
        state_rate: Any,
        inputs: Any,
        /,
    ) -> Array:
        del coordinate
        if state_rate is not None:
            raise ValueError(
                "A steady component residual has no state rate; reduce a transient "
                "owner through its stage residual instead."
            )
        return self.component.residual((state,), inputs)[0]


def _reduced_chart(record: ComponentField, trial: ConstraintMap, /) -> ConstraintMap:
    """The owner's field chart composed with the fixed trial chart ``z = V a``."""
    identifier = canonical_fingerprint(
        {
            "kind": "reduced-component-chart",
            "field": record.name,
            "owner-chart": None
            if record.constraint is None
            else record.constraint.constraint_id,
            "trial": trial.constraint_id,
        }
    )
    if record.constraint is not None:
        return compose_constraint_maps(record.constraint, trial, constraint_id=identifier)
    return ConstraintMap(
        record.full_space,
        trial.reduced_space,
        trial.prolongation,
        constraint_id=identifier,
    )


def _refuse_frozen_trainables(galerkin: FullResidualGalerkin, name: str, /) -> None:
    """Refuse trainable components that the fixed reduced provider would freeze."""
    violations = resolve_array_roles(galerkin).violations
    if violations:
        raise ValueError(
            f"ReducedComponent {name!r}: the reduced provider and basis are fixed "
            "prepared structure, but they hold trainable components that would be "
            "frozen silently; bind learned values through the owner's runtime "
            "arguments (a ParameterBinding) instead:\n"
            + "\n".join(
                f"  - {path or '<root>'} [{kind}]" for path, kind, _ in violations
            )
        )


@final
class ReducedComponent(
    AbstractTraceComponent, AbstractReconstructionComponent, NonTrainableState
):
    """A prepared Galerkin reduced-order model of one coupled component's region.

    ``galerkin`` is the ROM owner's ``FullResidualGalerkin`` over a
    ``ComponentResidualProvider`` of the full component, with a Galerkin
    reduction (one trial basis, used as the test basis) on the component's state
    coordinates. The published component keeps ``name`` (usually the full
    component's name, so interface bindings, laws, parameters, and observations
    stay unchanged) and the field's discrete space identity, boundary
    impositions, side traces, conormal (reaction) fluxes, pointwise flux, and
    reconstruction of the full owner: every one acts on the reconstructed field
    ``P (V a + l) + g``. The state block holds the reduced coordinates ``a`` and
    the row block the Galerkin rows ``V^T R``; laws inject through ``V^T P^T``.
    Interface certificates therefore measure the reconstruction against the
    original owner's traces and reaction fluxes, which is the interface residual
    of the reduced model. The basis, the provider, and the owner's topology are
    fixed prepared structure: a change is a new ``ReducedComponent`` and a new
    prepared problem, never an online derivative. Trainable values reach the
    owner only through its runtime arguments (parameter bindings); a trainable
    model hidden in the provider is refused. An owner that publishes a kernel
    is refused (the reduced kernel is not certified).
    """

    name: str = eqx.field(static=True)
    galerkin: FullResidualGalerkin
    field_name: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    def __init__(self, name: str, galerkin: FullResidualGalerkin, /) -> None:
        name_ = canonical_identifier(name, "name")
        if not isinstance(galerkin, FullResidualGalerkin):
            raise TypeError("galerkin must be a FullResidualGalerkin.")
        provider = galerkin.provider
        if not isinstance(provider, ComponentResidualProvider):
            raise TypeError(
                "A reduced component publishes the Galerkin reduction of a coupled "
                "component's residual; build the ROM over a ComponentResidualProvider."
            )
        reduction = galerkin.reduction
        if reduction.test_basis.artifact_id != reduction.trial_basis.artifact_id:
            raise ValueError(
                "A reduced component requires a Galerkin reduction: laws inject into "
                "its rows through the transpose of the same chart that reconstructs "
                "its field, so a Petrov-Galerkin test basis would not be applied."
            )
        _refuse_frozen_trainables(galerkin, name_)
        full = provider.component
        record = full.field(provider.field)
        reduced = reduction.trial.reduced_space
        self.name = name_
        self.galerkin = galerkin
        self.field_name = record.name
        self.owner_id = galerkin.model_id
        self.space = "reduced"
        self.state_blocks = (ComponentBlock(record.state_block, reduced),)
        self.row_blocks = (ComponentBlock(record.row_block, DualSpace(reduced)),)
        self.fields = (
            ComponentField(
                record.name,
                state_block=record.state_block,
                row_block=record.row_block,
                full_space=record.full_space,
                constraint=_reduced_chart(record, reduction.trial),
                free_rows=None,
            ),
        )

    @property
    def full(self) -> AbstractTraceComponent:
        """The full component whose residual the reduced model projects."""
        provider = self.galerkin.provider
        if not isinstance(provider, ComponentResidualProvider):
            raise TypeError("The reduced component lost its component provider.")
        return provider.component

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        zero = jnp.zeros((), dtype=state[0].dtype)
        return (self.galerkin.residual(zero, state[0], None, args),)

    def linear_operator(self, args: object, /) -> BlockLinearOperator | None:
        operator = self.full.linear_operator(args)
        if operator is None:
            return None
        block = operator.blocks[0][0]
        if block is None:
            raise ValueError("The full component publishes an empty operator block.")
        reduced = self.state_blocks[0].space
        # The reduced operator V^T A V is the bounded dense scientific result of
        # the ROM owner's projection (rank-sized), formed with ``rank`` owner actions.
        matrix = self.galerkin.reduction.project_operator(block)
        projected = DenseLinearOperator(matrix, source=reduced, target=DualSpace(reduced))
        return BlockLinearOperator(
            ((projected,),), source=self.state_space, target=self.row_space
        )

    def lift(self, field: str, args: object, /) -> Array:
        self.field(field)
        return self.full.expand(self.field_name, (self.galerkin.lift,), args)

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        if self.full.nullspace(args) is not None:
            raise ValueError(
                f"Component {self.full.name!r} publishes a kernel; the kernel of its "
                "reduced model is not certified, so the region is not reduced."
            )
        return None

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return self.full.boundary_impositions()

    def field_space_id(self, field: str, /) -> str:
        self.field(field)
        return self.full.field_space_id(self.field_name)

    def prepare_side_trace(
        self,
        field: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
    ) -> PreparedTraceAction:
        self.field(field)
        return self.full.prepare_side_trace(
            self.field_name, domain, rule=rule, quantity=quantity, side=side
        )

    def prepare_conormal_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        return self.full.prepare_conormal_flux(trace)

    def prepare_pointwise_flux(self, trace: PreparedTraceAction, /) -> PreparedFluxAction:
        return self.full.prepare_pointwise_flux(trace)

    def certify_flux_stability(self, flux: PreparedFluxAction, /) -> TraceInverseEvidence:
        return self.full.certify_flux_stability(flux)

    def prepare_field_reconstruction(self, field: str, /) -> PreparedFieldReconstruction:
        self.field(field)
        full = self.full
        if not isinstance(full, AbstractReconstructionComponent):
            raise TypeError(
                f"Component {full.name!r} publishes no pointwise field reconstruction."
            )
        return full.prepare_field_reconstruction(self.field_name)


__all__ = ["ComponentResidualProvider", "ReducedComponent"]
