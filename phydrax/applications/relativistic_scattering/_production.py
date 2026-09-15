#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...integration import vegas_integrate, VegasPlan, VegasResult
from ...particle_physics import (
    CrossSectionLedger,
    EventWeightSet,
    ParticleEventBatch,
    ParticleEventPlan,
    ParticleRole,
    PreparedParticleEvents,
    summarize_event_weights,
    WeightVariationKind,
)
from ._amplitudes import ScatteringProcess
from ._events import WeightedEventStream
from ._phase_space import kallen, TwoBodyPhaseSpaceMap


MatrixElementSquared = Callable[[Array, Array], Array]


class BeamPlan(StrictModule, NonTrainableState):
    """Two-particle centre-of-momentum beam state with explicit identities."""

    pdg_ids: tuple[int, int] = eqx.field(static=True)
    center_of_mass_energy: Array
    spectrum_id: str = eqx.field(static=True)
    crossing_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pdg_ids: Sequence[int],
        center_of_mass_energy: float,
        /,
        *,
        spectrum_id: str = "monoenergetic",
        crossing_id: str = "head-on",
    ):
        identities = tuple(int(value) for value in pdg_ids)
        energy = float(center_of_mass_energy)
        spectrum = str(spectrum_id).strip()
        crossing = str(crossing_id).strip()
        if len(identities) != 2 or not spectrum or not crossing:
            raise ValueError(
                "BeamPlan requires two identities and named spectrum/crossing."
            )
        if not math.isfinite(energy) or energy <= 0.0:
            raise ValueError("center_of_mass_energy must be finite and positive.")
        self.pdg_ids = identities
        self.center_of_mass_energy = jnp.asarray(energy)
        self.spectrum_id = spectrum
        self.crossing_id = crossing
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-particle-com-beam",
                "pdg_ids": list(identities),
                "center_of_mass_energy": energy,
                "spectrum": spectrum,
                "crossing": crossing,
            }
        )


class ScalePlan(StrictModule, NonTrainableState):
    renormalization_scale: Array
    factorization_scale: Array
    scheme_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        renormalization_scale: float,
        factorization_scale: float,
        /,
        *,
        scheme_id: str,
    ):
        renormalization = float(renormalization_scale)
        factorization = float(factorization_scale)
        scheme = str(scheme_id).strip()
        if (
            not math.isfinite(renormalization)
            or not math.isfinite(factorization)
            or renormalization <= 0.0
            or factorization <= 0.0
            or not scheme
        ):
            raise ValueError("Scales must be finite positive values with a named scheme.")
        self.renormalization_scale = jnp.asarray(renormalization)
        self.factorization_scale = jnp.asarray(factorization)
        self.scheme_id = scheme
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hard-process-scales",
                "renormalization": renormalization,
                "factorization": factorization,
                "scheme": scheme,
            }
        )


class HardProcessPlan(StrictModule, NonTrainableState):
    """Enumerated two-to-two hard-process normalization and support contract."""

    process: ScatteringProcess
    beam: BeamPlan
    scales: ScalePlan
    incoming_pdg_ids: tuple[int, int] = eqx.field(static=True)
    outgoing_pdg_ids: tuple[int, int] = eqx.field(static=True)
    matrix_element_id: str = eqx.field(static=True)
    coupling_scheme_id: str = eqx.field(static=True)
    spin_average: float = eqx.field(static=True)
    color_average: float = eqx.field(static=True)
    minimum_cosine: float = eqx.field(static=True)
    maximum_cosine: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        process: ScatteringProcess,
        beam: BeamPlan,
        scales: ScalePlan,
        /,
        *,
        outgoing_pdg_ids: Sequence[int],
        matrix_element_id: str,
        coupling_scheme_id: str,
        spin_average: float = 1.0,
        color_average: float = 1.0,
        cosine_range: tuple[float, float] = (-1.0, 1.0),
    ):
        if not isinstance(process, ScatteringProcess):
            raise TypeError("process must be ScatteringProcess.")
        if not isinstance(beam, BeamPlan) or not isinstance(scales, ScalePlan):
            raise TypeError("beam and scales must use their explicit plan types.")
        if len(process.incoming) != 2 or len(process.outgoing) != 2:
            raise ValueError(
                "Native hard production currently supports two-to-two processes."
            )
        outgoing = tuple(int(value) for value in outgoing_pdg_ids)
        if len(outgoing) != 2:
            raise ValueError("outgoing_pdg_ids must contain two identities.")
        matrix_element = str(matrix_element_id).strip()
        coupling = str(coupling_scheme_id).strip()
        spin = float(spin_average)
        color = float(color_average)
        minimum, maximum = map(float, cosine_range)
        if not matrix_element or not coupling:
            raise ValueError(
                "Matrix-element and coupling scheme identities are required."
            )
        if not all(math.isfinite(value) and value > 0.0 for value in (spin, color)):
            raise ValueError("Spin and color averages must be finite and positive.")
        if not (-1.0 <= minimum < maximum <= 1.0):
            raise ValueError("cosine_range must be an ordered subset of [-1, 1].")
        if float(beam.center_of_mass_energy) < sum(
            float(p.mass) for p in process.outgoing
        ):
            raise ValueError("Beam energy lies below the final-state threshold.")
        self.process = process
        self.beam = beam
        self.scales = scales
        self.incoming_pdg_ids = beam.pdg_ids
        self.outgoing_pdg_ids = outgoing
        self.matrix_element_id = matrix_element
        self.coupling_scheme_id = coupling
        self.spin_average = spin
        self.color_average = color
        self.minimum_cosine = minimum
        self.maximum_cosine = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-two-body-hard-process",
                "process": process.process_id,
                "beam": beam.plan_id,
                "scales": scales.plan_id,
                "incoming_pdg_ids": list(beam.pdg_ids),
                "outgoing_pdg_ids": list(outgoing),
                "matrix_element": matrix_element,
                "coupling_scheme": coupling,
                "spin_average": spin,
                "color_average": color,
                "cosine_range": [minimum, maximum],
            }
        )

    def prepare(
        self, matrix_element_squared: MatrixElementSquared, /
    ) -> PreparedHardProcess:
        return PreparedHardProcess(self, matrix_element_squared)


class PreparedHardProcess(StrictModule):
    plan: HardProcessPlan
    phase_space: TwoBodyPhaseSpaceMap
    matrix_element_squared: MatrixElementSquared = eqx.field(static=True)
    incoming_momenta: Array
    flux: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: HardProcessPlan, matrix_element_squared: MatrixElementSquared, /
    ):
        if not isinstance(plan, HardProcessPlan):
            raise TypeError("plan must be HardProcessPlan.")
        if not callable(matrix_element_squared):
            raise TypeError("matrix_element_squared must be callable.")
        root_s = plan.beam.center_of_mass_energy
        s = root_s * root_s
        first_mass = plan.process.incoming[0].mass
        second_mass = plan.process.incoming[1].mass
        triangle = kallen(s, first_mass * first_mass, second_mass * second_mass)
        momentum = jnp.sqrt(jnp.maximum(triangle, 0.0)) / (2.0 * root_s)
        first_energy = (s + first_mass * first_mass - second_mass * second_mass) / (
            2.0 * root_s
        )
        second_energy = (s + second_mass * second_mass - first_mass * first_mass) / (
            2.0 * root_s
        )
        self.plan = plan
        self.phase_space = TwoBodyPhaseSpaceMap(
            float(plan.process.outgoing[0].mass), float(plan.process.outgoing[1].mass)
        )
        self.matrix_element_squared = matrix_element_squared
        self.incoming_momenta = jnp.asarray(
            [[first_energy, 0.0, 0.0, momentum], [second_energy, 0.0, 0.0, -momentum]]
        )
        self.flux = 2.0 * jnp.sqrt(jnp.maximum(triangle, 0.0))
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-two-body-hard-process",
                "plan": plan.plan_id,
                "phase_space": self.phase_space.map_id,
            }
        )


class HardEventProduction(StrictModule, NonTrainableState):
    stream: WeightedEventStream
    events: ParticleEventBatch
    ledger: CrossSectionLedger
    generated: Array
    selected: Array
    derivative_valid: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.events.successful & self.ledger.successful


def _one_hard_point(prepared: PreparedHardProcess, unit: Array, /):
    total = jnp.asarray([prepared.plan.beam.center_of_mass_energy, 0.0, 0.0, 0.0])
    point = prepared.phase_space.map(unit, total)
    matrix_element = jnp.asarray(
        prepared.matrix_element_squared(prepared.incoming_momenta, point.momenta)
    ).reshape(())
    normalization = prepared.plan.process.symmetry_factor / (
        prepared.plan.spin_average * prepared.plan.color_average
    )
    weight = matrix_element * point.jacobian * normalization / prepared.flux
    cosine = 2.0 * unit[0] - 1.0
    selected = (
        point.valid
        & (cosine >= prepared.plan.minimum_cosine)
        & (cosine <= prepared.plan.maximum_cosine)
    )
    finite = jnp.isfinite(weight) & jnp.all(jnp.isfinite(point.momenta))
    return (
        point.momenta,
        jnp.where(finite & point.valid, weight, jnp.nan),
        point.valid & finite,
        selected & finite,
    )


def hard_process_integrand(
    prepared: PreparedHardProcess, unit_points: ArrayLike, /
) -> Array:
    """Evaluate the selected differential cross-section contribution on unit points."""
    points = jnp.asarray(unit_points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("unit_points must have shape (sample_count, 2).")
    _, weights, _, selected = jax.vmap(lambda point: _one_hard_point(prepared, point))(
        points
    )
    return jnp.where(selected, weights, 0.0)


def integrate_hard_process(
    prepared: PreparedHardProcess,
    vegas_plan: VegasPlan,
    adaptation_key: Key[Array, ""],
    production_key: Key[Array, ""],
    /,
) -> VegasResult:
    if not isinstance(prepared, PreparedHardProcess):
        raise TypeError("prepared must be PreparedHardProcess.")
    if (
        vegas_plan.dimension != 2
        or not np.allclose(np.asarray(vegas_plan.lower), 0.0)
        or not np.allclose(np.asarray(vegas_plan.upper), 1.0)
    ):
        raise ValueError(
            "Hard-process VEGAS integration requires the two-dimensional unit square."
        )
    return vegas_integrate(
        lambda points: hard_process_integrand(prepared, points),
        vegas_plan,
        adaptation_key,
        production_key,
    )


def produce_hard_events(
    prepared: PreparedHardProcess,
    unit_points: ArrayLike,
    event_plan: ParticleEventPlan | PreparedParticleEvents,
    /,
    *,
    event_ids: ArrayLike | None = None,
    source_id: str | None = None,
) -> HardEventProduction:
    """Produce bounded weighted truth events and complete cross-section accounting."""
    if not isinstance(prepared, PreparedHardProcess):
        raise TypeError("prepared must be PreparedHardProcess.")
    points = jnp.asarray(unit_points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("unit_points must have shape (event_capacity, 2).")
    prepared_events = (
        event_plan.prepare() if isinstance(event_plan, ParticleEventPlan) else event_plan
    )
    if not isinstance(prepared_events, PreparedParticleEvents):
        raise TypeError("event_plan must be ParticleEventPlan or PreparedParticleEvents.")
    capacity = int(points.shape[0])
    if prepared_events.plan.event_capacity != capacity:
        raise ValueError("Event plan capacity must equal the unit-point count.")
    if (
        prepared_events.plan.particle_capacity < 4
        or prepared_events.plan.vertex_capacity < 1
    ):
        raise ValueError(
            "Hard-event lowering requires four particle and one vertex slots."
        )
    outgoing, weights, generated, selected = jax.vmap(
        lambda point: _one_hard_point(prepared, point)
    )(points)
    values = EventWeightSet(
        weights[:, None],
        names=("nominal",),
        variation_kinds=(WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
        event_active=generated,
    )
    event_ids_ = (
        jnp.arange(capacity, dtype=jnp.int64)
        if event_ids is None
        else jnp.asarray(event_ids)
    )
    particle_capacity = prepared_events.plan.particle_capacity
    vertex_capacity = prepared_events.plan.vertex_capacity
    momenta = jnp.zeros((capacity, particle_capacity, 4), dtype=outgoing.dtype)
    momenta = momenta.at[:, :2].set(prepared.incoming_momenta)
    momenta = momenta.at[:, 2:4].set(outgoing)
    rest_energies = jnp.zeros((capacity, particle_capacity), dtype=outgoing.dtype)
    rest_energies = rest_energies.at[:, :2].set(
        jnp.asarray([particle.mass for particle in prepared.plan.process.incoming])
    )
    rest_energies = rest_energies.at[:, 2:4].set(
        jnp.asarray([particle.mass for particle in prepared.plan.process.outgoing])
    )
    pdg_ids = jnp.zeros((capacity, particle_capacity), dtype=jnp.int32)
    pdg_ids = pdg_ids.at[:, :4].set(
        jnp.asarray(
            prepared.plan.incoming_pdg_ids + prepared.plan.outgoing_pdg_ids,
            dtype=jnp.int32,
        )
    )
    roles = jnp.zeros_like(pdg_ids)
    roles = roles.at[:, :2].set(int(ParticleRole.INCOMING))
    roles = roles.at[:, 2:4].set(int(ParticleRole.OUTGOING))
    status = jnp.zeros_like(pdg_ids)
    status = status.at[:, :2].set(-1)
    status = status.at[:, 2:4].set(1)
    particle_active = jnp.zeros((capacity, particle_capacity), dtype=bool)
    particle_active = particle_active.at[:, :4].set(generated[:, None])
    mothers = jnp.full((capacity, particle_capacity, 2), -1, dtype=jnp.int32)
    mothers = mothers.at[:, 2:4].set(jnp.asarray([0, 1], dtype=jnp.int32))
    vertices = jnp.zeros((capacity, vertex_capacity, 4), dtype=outgoing.dtype)
    vertex_active = jnp.zeros((capacity, vertex_capacity), dtype=bool)
    vertex_active = vertex_active.at[:, 0].set(generated)
    resolved_source = (
        prepared.prepared_id if source_id is None else str(source_id).strip()
    )
    events = prepared_events.admit(
        event_ids=event_ids_,
        subevent_ids=jnp.zeros((capacity,), dtype=jnp.int32),
        event_active=generated,
        pdg_ids=pdg_ids,
        roles=roles,
        provider_status=status,
        momenta=momenta,
        rest_energies=rest_energies,
        particle_active=particle_active,
        mother_indices=mothers,
        color_flow=jnp.zeros((capacity, particle_capacity, 2), dtype=jnp.int32),
        production_vertices=vertices,
        vertex_active=vertex_active,
        weights=values,
        source_id=resolved_source,
    )
    selected_weights = jnp.where(selected, weights, 0.0)
    cross_section = jnp.mean(selected_weights)
    standard_error = (
        jnp.std(selected_weights, ddof=1) / jnp.sqrt(max(capacity, 1))
        if capacity > 1
        else jnp.asarray(jnp.nan)
    )
    ledger = summarize_event_weights(
        values,
        selected=selected,
        attempted_count=capacity,
        cross_section=cross_section,
        cross_section_uncertainty=standard_error,
    )
    stream = WeightedEventStream(
        outgoing,
        weights,
        active=selected,
        provenance=prepared.prepared_id,
    )
    derivative_valid = generated & ~(
        jnp.isclose(2.0 * points[:, 0] - 1.0, prepared.plan.minimum_cosine)
        | jnp.isclose(2.0 * points[:, 0] - 1.0, prepared.plan.maximum_cosine)
    )
    return HardEventProduction(
        stream,
        events,
        ledger,
        generated,
        selected,
        derivative_valid,
        prepared.prepared_id,
    )


__all__ = [
    "BeamPlan",
    "HardEventProduction",
    "HardProcessPlan",
    "PreparedHardProcess",
    "ScalePlan",
    "hard_process_integrand",
    "integrate_hard_process",
    "produce_hard_events",
]
