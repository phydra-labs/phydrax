#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolved bubbly-flow coupling of the structured two-phase VOF step.

`BubblyFlowPlan` composes these components on top of one
`PreparedIncompressibleTwoPhaseVOF`:

- bubble identity (`BubbleComponentPlan`);
- optional multi-marker VOF (`MultiMarkerPlan`), with its near-contact
  potential;
- an optional compartment registry (`BubbleCompartmentPlan`) with the mixed
  compartment projection (`solver.MACCompartmentProjectionPlan`);
- optional film-drainage coalescence (`FilmDrainageCoalescencePlan`).

`IncompressibleTwoPhaseVOFMethod` calls it at three points of every step:

- after the geometric transport (`prepare`): replay markers, propose
  identities, map compartments, build the projection constraint and the
  near-contact force;
- instead of the variable-density projection (`project`), when compartments
  are declared;
- after the projection (`finish`): commit compartment work, advance the film
  contact ledger and propose the identity state.

A step is refused (`host_transaction_required`) when its proposed bubble
identities differ from the registered compartments. The host-side
`run_bubbly_flow` driver then journals the event, applies the compartment
transaction through the gas law, and repeats the step. Recoloring and
drainage-gated merges are host transactions on the accepted state. No
derivative is claimed across any of them.

The near-contact potential is the plan's model of the unresolved film. In
every gas cell of a bubble whose nearest bubble of a different color lies
closer than the marker proximity radius ``r`` (distance ``d``), it adds the
interfacial potential

```text
phi_nc = -Pi_0 max(0, 1 - d / r)
```

Through the balanced capillary construction this gives the face force
``face_average(phi_nc) (G alpha)``, which pushes the facing interfaces apart
(a disjoining-pressure analogue). The resolved film therefore does not drain
numerically below the grid scale. Half the potential pressure times the PLIC
facet measure of both interfaces is the pair load ``F``. The same ``F`` and
the potential's work enter the one film contact ledger that drives the
drainage law, so the load is not counted twice.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import ArrayRole, parameter_field, resolve_array_roles
from ...solver import (
    MACCompartmentConstraint,
    MACCompartmentProjectionPlan,
    MACCompartmentProjectionResult,
)
from ._bubble_components import (
    BubbleComponentLabels,
    BubbleComponentPlan,
    BubbleComponentState,
    BubbleIdentityEvent,
    BubbleIdentityProposal,
    BubbleTopologyEvidence,
)
from ._bubble_thermodynamics import (
    BubbleCompartmentEvaluation,
    BubbleCompartmentPlan,
    BubbleCompartmentState,
)
from ._coalescence import (
    film_equivalent_radius,
    FilmContactLedger,
    FilmContactObservation,
    FilmDrainageCoalescencePlan,
)
from ._flux_bundle import TwoPhaseFluxBundle
from ._multi_marker import (
    MarkerProximity,
    MarkerTransportResult,
    MultiMarkerPlan,
    MultiMarkerState,
)
from ._vof import FaceTuple, PreparedIncompressibleTwoPhaseVOF


def _face_cells(
    value: Array, axis: int, periodic: bool, fill: Array, /
) -> tuple[Array, Array]:
    """Lower and upper cell values of every MAC face normal to ``axis``."""

    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        lower = jnp.roll(moved, 1, axis=0)
        upper = moved
    else:
        ghost = jnp.broadcast_to(fill, (1,) + moved.shape[1:]).astype(moved.dtype)
        lower = jnp.concatenate((ghost, moved), axis=0)
        upper = jnp.concatenate((moved, ghost), axis=0)
    return jnp.moveaxis(lower, 0, axis), jnp.moveaxis(upper, 0, axis)


def _equivalent_sphere_radius(volume: Array, dimension: int, /) -> Array:
    """Radius of the circle (2D, per unit depth) or sphere of equal volume."""

    positive = jnp.maximum(volume, 0.0)
    if dimension == 2:
        return jnp.sqrt(positive / jnp.pi)
    return jnp.cbrt(3.0 * positive / (4.0 * jnp.pi))


class NearContactForce(StrictModule):
    """Near-contact potential of one step and its pair loads.

    ``face_force`` is the force density on MAC faces and ``potential`` the cell
    interfacial potential on ``support``. ``cell_pair`` is the close-pair slot
    of every supported cell. ``pair_load`` is the load ``F`` of every
    close-pair slot. ``net_force`` is the total momentum rate injected,
    ``sum_f f_f M_f`` per axis, and is the antisymmetry evidence.
    """

    face_force: FaceTuple
    potential: Array
    support: Array
    cell_pair: Array
    pair_load: Array
    net_force: Array


class BubblyFlowState(StrictModule):
    """Bubble state carried by the two-phase continuation.

    ``dilatation`` is the compartment divergence (per unit volume) that the
    last projection declared. The next transport's dilation correction uses
    it. ``event`` is the slot-sized record of the last committed identity
    proposal, and the host journal reads it.
    """

    identity: BubbleComponentState
    event: BubbleIdentityEvent
    markers: MultiMarkerState | None
    compartments: BubbleCompartmentState | None
    contacts: FilmContactLedger | None
    dilatation: Array


class BubblyStepContext(StrictModule):
    """Device quantities of one step computed after the geometric transport."""

    proposal: BubbleIdentityProposal
    markers: MarkerTransportResult | None
    proximity: MarkerProximity | None
    registry_slot: Array
    registry_mismatch: Array
    registry_volume: Array | None
    registry_centroid: Array | None
    evaluation: BubbleCompartmentEvaluation | None
    constraint: MACCompartmentConstraint | None
    near_contact: NearContactForce | None
    creation_pressure: Array


class BubblyFlowEvidence(StrictModule):
    """Per-step bubble evidence and the fail-closed acceptance decision.

    ``host_transaction_required`` asks the host driver to act:

    - journal an identity event;
    - apply a compartment transaction (``registry_mismatch``, which refuses
      the step);
    - recolor a same-color conflict;
    - execute a drainage-gated merge.
    """

    topology: BubbleTopologyEvidence
    marker_sum_residual: Array
    marker_minimum_content: Array
    recolor_conflicts: Array
    registry_mismatch: Array
    projection_status: Array
    compartment_pressure_residual: Array
    work_identity_residual: Array
    eos_residual: Array
    compartment_work: Array
    compartment_heat: Array
    pressure_work: Array
    near_contact_work: Array
    near_contact_net_force: Array
    merge_proposals: Array
    release_events: Array
    contact_overflow: Array
    creation_pressure: Array
    host_transaction_required: Array
    derivative_available: Array
    successful: Array


class BubblyFlowPlan(StrictModule):
    """Composition of bubble identity, markers, compartments and coalescence.

    ``near_contact_pressure`` (``Pi_0``, Pa) and ``atmosphere_pressure``
    (``p_atm``, Pa) are inferable leaves. Coalescence requires markers, a
    positive ``Pi_0`` and the film geometry of the grid dimension (``planar``
    in 2D, ``axisymmetric`` in 3D). Compartments compose the mixed MAC
    compartment projection with the prepared projection tolerances; the
    projection carries an atmosphere row exactly when the identity plan
    declares vent sides.
    """

    two_phase: PreparedIncompressibleTwoPhaseVOF
    identity: BubbleComponentPlan
    markers: MultiMarkerPlan | None
    compartments: BubbleCompartmentPlan | None
    projection: MACCompartmentProjectionPlan | None
    coalescence: FilmDrainageCoalescencePlan | None
    near_contact_pressure: Array = parameter_field()
    atmosphere_pressure: Array = parameter_field()
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        two_phase: PreparedIncompressibleTwoPhaseVOF,
        identity: BubbleComponentPlan,
        /,
        *,
        markers: MultiMarkerPlan | None = None,
        compartments: BubbleCompartmentPlan | None = None,
        coalescence: FilmDrainageCoalescencePlan | None = None,
        near_contact_pressure: ArrayLike = 0.0,
        atmosphere_pressure: ArrayLike = 0.0,
    ) -> None:
        if not isinstance(two_phase, PreparedIncompressibleTwoPhaseVOF):
            raise TypeError("two_phase must be PreparedIncompressibleTwoPhaseVOF.")
        if two_phase.geometry is not None:
            raise ValueError("Bubbly flow does not compose qualified sharp geometry.")
        if not isinstance(identity, BubbleComponentPlan):
            raise TypeError("identity must be a BubbleComponentPlan.")
        dimension = len(two_phase.plan.discretization.cell_shape)
        capacity = identity.component_capacity
        if markers is not None and markers.component_capacity != capacity:
            raise ValueError("markers must share the identity component capacity.")
        if compartments is not None and compartments.dimension != dimension:
            raise ValueError("compartments must match the grid dimension.")
        pressure = jnp.asarray(near_contact_pressure, dtype=jnp.float64)
        atmosphere = jnp.asarray(atmosphere_pressure, dtype=jnp.float64)
        if pressure.shape != () or not bool(jnp.isfinite(pressure) & (pressure >= 0.0)):
            raise ValueError("near_contact_pressure must be a finite nonnegative scalar.")
        if atmosphere.shape != () or not bool(jnp.isfinite(atmosphere)):
            raise ValueError("atmosphere_pressure must be a finite scalar.")
        if coalescence is not None:
            if markers is None:
                raise ValueError("Film-drainage coalescence requires multi-marker VOF.")
            if coalescence.pair_capacity != markers.pair_groups.group_capacity:
                raise ValueError(
                    "coalescence pair_capacity must equal the marker pair capacity."
                )
            expected = "planar" if dimension == 2 else "axisymmetric"
            if coalescence.geometry != expected:
                raise ValueError(
                    f"A {dimension}D grid resolves {expected} films; the coalescence "
                    f"plan declares {coalescence.geometry}."
                )
            if not bool(pressure > 0.0):
                raise ValueError(
                    "Coalescence needs a positive near_contact_pressure to carry "
                    "the film load."
                )
        projection = (
            None
            if compartments is None
            else MACCompartmentProjectionPlan(
                two_phase.operators,
                compartment_capacity=compartments.capacity,
                atmosphere=bool(identity.vent_sides),
                tolerance=two_phase.plan.tolerance,
                maximum_iterations=two_phase.plan.maximum_iterations,
            )
        )
        self.two_phase = two_phase
        self.identity = identity
        self.markers = markers
        self.compartments = compartments
        self.projection = projection
        self.coalescence = coalescence
        self.near_contact_pressure = pressure
        self.atmosphere_pressure = atmosphere
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bubbly-flow-plan",
                "two_phase": two_phase.prepared_id,
                "identity": identity.plan_id,
                "markers": None if markers is None else markers.plan_id,
                "compartments": None if compartments is None else compartments.plan_id,
                "projection": None if projection is None else projection.plan_id,
                "coalescence": None if coalescence is None else coalescence.plan_id,
            }
        )

    def parameter_realization_id(self) -> str:
        """Fingerprint the current dynamic physical parameters on the host.

        ``plan_id`` remains the immutable structural identity. This realization
        is recomputed from every declared parameter leaf, including nested gas,
        capillary and film-drainage parameters, so trained leaves do not leave a
        stale identity behind.
        """
        roles = resolve_array_roles(self)
        leaves = jax.tree_util.tree_leaves(self)
        parameters = {
            path: leaf
            for path, role, leaf in zip(
                roles.paths, roles.roles, leaves, strict=True
            )
            if role is ArrayRole.PARAMETER
        }
        return canonical_fingerprint(
            {
                "kind": "bubbly-flow-parameter-realization",
                "plan": self.plan_id,
                "parameters": parameters,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.two_phase.plan.discretization.cell_shape)

    def pressure_offset(self, alpha: ArrayLike, /) -> Array:
        """Offset ``h`` with absolute pressure ``phi + h`` in compartment runs.

        ``h = p_ref + rho(alpha) g . (x - Z)``. The compartment projection fixes
        the level of ``phi``, so there is no reference-cell anchoring.
        """

        two_phase = self.two_phase
        alpha_ = jnp.asarray(alpha)
        return two_phase.plan.reference_pressure + two_phase.mixture_density(
            alpha_
        ) * two_phase.hydrostatic_height(alpha_.dtype)

    def absolute_pressure(self, alpha: ArrayLike, pressure: ArrayLike, /) -> Array:
        """Absolute pressure of a dynamic pressure field under this plan."""

        if self.compartments is None:
            return self.two_phase.absolute_pressure(alpha, pressure)
        return jnp.asarray(pressure) + self.pressure_offset(alpha)

    def _component_mean(self, labels: BubbleComponentLabels, field: Array, /) -> Array:
        capacity = self.identity.component_capacity
        label = labels.labeling.label
        safe = jnp.where(label >= 0, label, capacity)
        weights = jnp.where(label >= 0, labels.gas_volume.reshape(-1), 0.0)
        total = jax.ops.segment_sum(weights * field.reshape(-1), safe, capacity + 1)
        volume = jnp.where(labels.volume > 0.0, labels.volume, 1.0)
        return jnp.where(labels.volume > 0.0, total[:capacity] / volume, 0.0)

    def initial_state(
        self,
        alpha: ArrayLike,
        /,
        *,
        color: ArrayLike | None = None,
        compartment_pressure: ArrayLike | None = None,
    ) -> BubblyFlowState:
        """Initial bubble state of a volume-fraction field (host preparation).

        ``color`` (one marker per cell) is required with markers.
        ``compartment_pressure`` (scalar or one absolute pressure per identity
        component slot) is required with compartments; for bubbles at rest it
        is the liquid pressure plus the Laplace jump.
        """

        alpha_ = jnp.asarray(alpha, dtype=self.two_phase.cell_fluid_measure.dtype)
        markers = None
        dominant = None
        if self.markers is not None:
            if color is None:
                raise ValueError("Multi-marker bubbly flow needs an initial color field.")
            markers = self.markers.initial_state(alpha_, color)
            dominant = self.markers.color(markers)
        elif color is not None:
            raise ValueError("color is only meaningful with multi-marker VOF.")
        identity = self.identity.initial_state(alpha_, color=dominant)
        compartments = None
        if self.compartments is not None:
            if compartment_pressure is None:
                raise ValueError("Compartments need an initial compartment_pressure.")
            compartments = self.compartments.initial_state(identity, compartment_pressure)
        elif compartment_pressure is not None:
            raise ValueError("compartment_pressure requires compartments.")
        contacts = (
            None
            if self.coalescence is None
            else FilmContactLedger.empty(self.coalescence.pair_capacity, jnp.float64)
        )
        event = self.identity.propose(identity, alpha_, color=dominant).event
        return BubblyFlowState(
            identity=identity,
            event=event,
            markers=markers,
            compartments=compartments,
            contacts=contacts,
            dilatation=jnp.zeros(alpha_.shape, dtype=alpha_.dtype),
        )

    def initial_evidence(
        self, state: BubblyFlowState, alpha: Array, /
    ) -> BubblyFlowEvidence:
        """Return accepted zero-work evidence for an initial bubble state."""

        color = (
            None
            if self.markers is None or state.markers is None
            else self.markers.color(state.markers)
        )
        proposal = self.identity.propose(state.identity, alpha, color=color)
        topology = proposal.evidence
        zero = jnp.zeros((), dtype=alpha.dtype)
        return BubblyFlowEvidence(
            topology=topology,
            marker_sum_residual=zero,
            marker_minimum_content=zero,
            recolor_conflicts=jnp.asarray(0, dtype=jnp.int32),
            registry_mismatch=jnp.asarray(False),
            projection_status=jnp.asarray(0, dtype=jnp.int32),
            compartment_pressure_residual=zero,
            work_identity_residual=zero,
            eos_residual=zero,
            compartment_work=zero,
            compartment_heat=zero,
            pressure_work=zero,
            near_contact_work=zero,
            near_contact_net_force=zero,
            merge_proposals=jnp.asarray(0, dtype=jnp.int32),
            release_events=jnp.asarray(0, dtype=jnp.int32),
            contact_overflow=jnp.asarray(False),
            creation_pressure=jnp.zeros(
                (self.identity.component_capacity,), dtype=alpha.dtype
            ),
            host_transaction_required=jnp.asarray(False),
            derivative_available=topology.derivative_available,
            successful=proposal.labels.labeling.successful
            & proposal.transition.successful,
        )

    def prepare(
        self,
        state: BubblyFlowState,
        bundle: TwoPhaseFluxBundle,
        alpha: Array,
        pressure: Array,
        facet_measure: Array,
        /,
    ) -> BubblyStepContext:
        """Markers, identity proposal, compartment constraint and contact force."""

        transport = None
        color = None
        if self.markers is not None:
            if state.markers is None:
                raise ValueError("The bubble state carries no markers.")
            transport = self.markers.transport(state.markers, bundle)
            color = self.markers.color(transport.state)
        proposal = self.identity.propose(state.identity, alpha, color=color)
        labels = proposal.labels
        proximity = None if self.markers is None else self.markers.proximity(labels)
        creation = self._component_mean(labels, self.absolute_pressure(alpha, pressure))
        near = (
            None
            if self.markers is None or proximity is None
            else self._near_contact(labels, proximity, alpha, facet_measure)
        )
        capacity = self.identity.component_capacity
        if self.compartments is None:
            return BubblyStepContext(
                proposal=proposal,
                markers=transport,
                proximity=proximity,
                registry_slot=jnp.full((capacity,), -1, dtype=jnp.int32),
                registry_mismatch=jnp.asarray(False),
                registry_volume=None,
                registry_centroid=None,
                evaluation=None,
                constraint=None,
                near_contact=near,
                creation_pressure=creation,
            )
        return self._compartment_context(
            state, proposal, transport, proximity, near, creation, alpha
        )

    def _compartment_context(
        self,
        state: BubblyFlowState,
        proposal: BubbleIdentityProposal,
        transport: MarkerTransportResult | None,
        proximity: MarkerProximity | None,
        near: NearContactForce | None,
        creation: Array,
        alpha: Array,
        /,
    ) -> BubblyStepContext:
        compartments = self.compartments
        registry = state.compartments
        if compartments is None or registry is None:
            raise ValueError("The bubble state carries no compartment registry.")
        labels = proposal.labels
        slots = compartments.capacity
        capacity = self.identity.component_capacity
        registry_slot, missing = compartments.slot_map(registry, proposal.slot_ids)
        target = jnp.where(registry_slot >= 0, registry_slot, slots)
        present = jnp.zeros((slots + 1,), dtype=jnp.int32).at[target].add(1)[:slots] > 0
        mismatch = jnp.any(missing) | jnp.any(registry.active & ~present)
        volume = (
            jnp.zeros((slots + 1,), dtype=labels.volume.dtype)
            .at[target]
            .add(labels.volume)[:slots]
        )
        centroid = (
            jnp.zeros((slots + 1, self.dimension), dtype=labels.centroid.dtype)
            .at[target]
            .add(labels.centroid)[:slots]
        )
        evaluation = compartments.evaluate(registry, volume)
        label = labels.labeling.label.reshape(labels.gas_volume.shape)
        safe = jnp.clip(label, 0, capacity - 1)
        cell_registry = jnp.where(label >= 0, registry_slot[safe], -1)
        atmosphere = (label >= 0) & labels.atmosphere[safe]
        constraint = MACCompartmentConstraint(
            labels=cell_registry,
            gas_volume=jnp.where(label >= 0, labels.gas_volume, 0.0),
            compliance=evaluation.compliance,
            pressure=evaluation.pressure,
            active=registry.active & present,
            atmosphere=atmosphere,
            atmosphere_pressure=self.atmosphere_pressure,
            pressure_offset=self.pressure_offset(alpha),
        )
        return BubblyStepContext(
            proposal=proposal,
            markers=transport,
            proximity=proximity,
            registry_slot=registry_slot,
            registry_mismatch=mismatch,
            registry_volume=volume,
            registry_centroid=centroid,
            evaluation=evaluation,
            constraint=constraint,
            near_contact=near,
            creation_pressure=creation,
        )

    def _near_contact(
        self,
        labels: BubbleComponentLabels,
        proximity: MarkerProximity,
        alpha: Array,
        facet_measure: Array,
        /,
    ) -> NearContactForce:
        markers = self.markers
        if markers is None:
            raise ValueError("The near-contact potential requires markers.")
        two_phase = self.two_phase
        label = labels.labeling.label.reshape(labels.gas_volume.shape)
        reach = markers.proximity_radius * markers.contact_distance
        distance = proximity.cell_foreign_distance
        support = (label >= 0) & (distance < reach)
        potential = jnp.where(
            support, -self.near_contact_pressure * (1.0 - distance / reach), 0.0
        ).astype(alpha.dtype)
        face_potential, _ = two_phase.capillarity.face_average(potential, support)
        gradient = two_phase.operators.gradient(alpha)
        face_force = tuple(
            value * derivative
            for value, derivative in zip(face_potential, gradient, strict=True)
        )
        pairs = markers.pair_groups.group_capacity
        cell_pair = jnp.where(support, proximity.cell_pair_slot, -1)
        safe = jnp.where(cell_pair >= 0, cell_pair, pairs).reshape(-1)
        load = (
            0.5
            * jax.ops.segment_sum(
                (-potential * facet_measure).reshape(-1), safe, pairs + 1
            )[:pairs]
        )
        net = jnp.stack(
            tuple(
                jnp.sum(force * measure)
                for force, measure in zip(
                    face_force, two_phase.face_open_dual_measure, strict=True
                )
            )
        )
        return NearContactForce(
            face_force=face_force,
            potential=potential,
            support=support,
            cell_pair=cell_pair,
            pair_load=load,
            net_force=net,
        )

    def project(
        self,
        context: BubblyStepContext,
        momentum: FaceTuple,
        face_inverse_density: FaceTuple,
        step_size: Array,
        pressure: Array,
        /,
    ) -> MACCompartmentProjectionResult:
        """Mixed compartment projection of the stage momentum density."""

        if self.projection is None or context.constraint is None:
            raise ValueError("This bubbly-flow plan declares no compartments.")
        return self.projection.project(
            momentum,
            face_inverse_density,
            step_size,
            context.constraint,
            pressure=pressure,
        )

    def _pair_work(
        self, near: NearContactForce, velocity: FaceTuple, step_size: Array, /
    ) -> Array:
        """Near-contact work of every pair slot, split over supporting cells."""

        markers = self.markers
        if markers is None:
            raise ValueError("Pair work requires markers.")
        pairs = markers.pair_groups.group_capacity
        axes = self.two_phase.plan.discretization.grid.structured_axes
        total = jnp.zeros((pairs + 1,), dtype=near.potential.dtype)
        fill = jnp.asarray(-1, dtype=jnp.int32)
        for axis, grid_axis in enumerate(axes):
            power = (
                step_size
                * near.face_force[axis]
                * velocity[axis]
                * self.two_phase.face_open_dual_measure[axis]
            )
            lower, upper = _face_cells(near.cell_pair, axis, grid_axis.periodic, fill)
            count = (lower >= 0).astype(power.dtype) + (upper >= 0).astype(power.dtype)
            share = power / jnp.where(count > 0.0, count, 1.0)
            for side in (lower, upper):
                slot = jnp.where(side >= 0, side, pairs).reshape(-1)
                total = total.at[slot].add(jnp.where(side >= 0, share, 0.0).reshape(-1))
        return total[:pairs]

    def _observation(
        self,
        context: BubblyStepContext,
        velocity: FaceTuple,
        step_size: Array,
        /,
    ) -> FilmContactObservation:
        proximity = context.proximity
        near = context.near_contact
        if proximity is None or near is None:
            raise ValueError("Film contact observations require markers.")
        labels = context.proposal.labels
        ids = context.proposal.slot_ids
        capacity = self.identity.component_capacity
        first = jnp.clip(proximity.pair_first, 0, capacity - 1)
        second = jnp.clip(proximity.pair_second, 0, capacity - 1)
        first_id = ids[first]
        second_id = ids[second]
        valid = (
            proximity.pair_active
            & ~proximity.same_color
            & (first_id >= 0)
            & (second_id >= 0)
        )
        radius = _equivalent_sphere_radius(labels.volume, self.dimension)
        radius = jnp.where(labels.atmosphere, jnp.inf, radius)
        equivalent = film_equivalent_radius(radius[first], radius[second])
        in_contact = valid & (near.pair_load > 0.0)
        return FilmContactObservation(
            first_id=jnp.where(valid, jnp.minimum(first_id, second_id), -1),
            second_id=jnp.where(valid, jnp.maximum(first_id, second_id), -1),
            valid=valid,
            in_contact=in_contact,
            load=jnp.where(in_contact, near.pair_load, 0.0),
            equivalent_radius=jnp.where(valid, equivalent, 1.0),
            near_contact_work=jnp.where(
                valid, self._pair_work(near, velocity, step_size), 0.0
            ),
        )

    def finish(
        self,
        state: BubblyFlowState,
        context: BubblyStepContext,
        projection: MACCompartmentProjectionResult | None,
        velocity: FaceTuple,
        step_size: Array,
        /,
    ) -> tuple[BubblyFlowState, BubblyFlowEvidence]:
        """Commit compartment work, drain films and propose the bubble state.

        ``velocity`` is the midpoint face velocity that pairs with the
        near-contact force work.
        """

        proposal = context.proposal
        dtype = proposal.labels.volume.dtype
        zero = jnp.zeros((), dtype=dtype)
        compartments = state.compartments
        dilatation = jnp.zeros_like(state.dilatation)
        projection_ok = jnp.asarray(True)
        projection_status = jnp.asarray(0, dtype=jnp.int32)
        pressure_residual = zero
        identity_residual = zero
        eos = zero
        work_total = zero
        heat_total = zero
        pressure_work = zero
        if self.compartments is not None and projection is not None:
            if (
                compartments is None
                or context.registry_volume is None
                or context.registry_centroid is None
            ):
                raise ValueError("The compartment context is incomplete.")
            compartments, work = self.compartments.commit_projection(
                compartments,
                context.registry_volume,
                context.registry_centroid,
                projection.compartment_pressure,
                projection.volume_rate,
                step_size,
            )
            dilatation = projection.divergence_target
            projection_ok = projection.successful & work.admissible
            projection_status = projection.status
            pressure_residual = projection.compartment_pressure_residual
            identity_residual = projection.work_identity_residual
            eos = jnp.max(work.eos_residual)
            work_total = jnp.sum(work.work)
            pressure_work = projection.dynamic_pressure_work
            heat_total = jnp.sum(work.heat)
        contacts = state.contacts
        merges = jnp.asarray(0, dtype=jnp.int32)
        releases = jnp.asarray(0, dtype=jnp.int32)
        overflow = jnp.asarray(False)
        contact_ok = jnp.asarray(True)
        if self.coalescence is not None:
            if contacts is None:
                raise ValueError("The bubble state carries no film contact ledger.")
            update = self.coalescence.advance(
                contacts, self._observation(context, velocity, step_size), step_size
            )
            contacts = update.ledger
            merges = jnp.sum(update.merge_proposals, dtype=jnp.int32)
            releases = jnp.sum(update.release_events, dtype=jnp.int32)
            overflow = update.capacity_overflow
            contact_ok = update.successful
        near = context.near_contact
        near_work = (
            zero
            if near is None
            else sum(
                (
                    step_size * jnp.sum(force * value * measure)
                    for force, value, measure in zip(
                        near.face_force,
                        velocity,
                        self.two_phase.face_open_dual_measure,
                        strict=True,
                    )
                ),
                start=zero,
            )
        )
        near_force = zero if near is None else jnp.sqrt(jnp.sum(near.net_force**2))
        proximity = context.proximity
        conflicts = (
            jnp.asarray(0, dtype=jnp.int32)
            if proximity is None
            else jnp.sum(proximity.pair_active & proximity.same_color, dtype=jnp.int32)
        )
        markers = context.markers
        markers_ok = jnp.asarray(True) if markers is None else markers.successful
        pair_overflow = (
            jnp.asarray(False) if proximity is None else proximity.pair_overflow
        )
        topology = proposal.evidence
        identity_ok = proposal.labels.labeling.successful & proposal.transition.successful
        successful = (
            identity_ok
            & markers_ok
            & ~pair_overflow
            & ~context.registry_mismatch
            & projection_ok
            & contact_ok
        )
        evidence = BubblyFlowEvidence(
            topology=topology,
            marker_sum_residual=zero if markers is None else markers.sum_residual,
            marker_minimum_content=zero if markers is None else markers.minimum_content,
            recolor_conflicts=conflicts,
            registry_mismatch=context.registry_mismatch,
            projection_status=projection_status,
            compartment_pressure_residual=pressure_residual,
            work_identity_residual=identity_residual,
            eos_residual=eos,
            compartment_work=work_total,
            compartment_heat=heat_total,
            pressure_work=pressure_work,
            near_contact_work=near_work,
            near_contact_net_force=near_force,
            merge_proposals=merges,
            release_events=releases,
            contact_overflow=overflow,
            creation_pressure=context.creation_pressure,
            host_transaction_required=context.registry_mismatch
            | (conflicts > 0)
            | (merges > 0)
            | topology.topology_changed,
            derivative_available=topology.derivative_available,
            successful=successful,
        )
        updated = BubblyFlowState(
            identity=proposal.commit(state.identity),
            event=proposal.event,
            markers=None if markers is None else markers.state,
            compartments=compartments,
            contacts=contacts,
            dilatation=dilatation,
        )
        return updated, evidence
