# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit SI reduced-resistance/capacitance building models.

Topology is static; capacities, conductances and areas remain differentiable leaves.
There is deliberately no inference of RC order from geometry or material names.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics import (
    affine_exponential_step,
    ContinuousSystem,
    DAEStructure,
    DifferentialAlgebraicSystem,
    InputLayout,
    StateLayout,
)
from ...ein import contract
from ...linalg import DenseLinearOperator, LinearSystem, MatrixFunctionPolicy, solve


def _text(value: str, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be non-empty.")
    return value


def _scalar(value: ArrayLike, owner: str, *, positive: bool = False) -> Array:
    x = jnp.asarray(value, dtype=jnp.result_type(value, float))
    if x.shape != ():
        raise ValueError(f"{owner} must be scalar.")
    return eqx.error_if(
        x,
        ~jnp.isfinite(x) | (x <= 0 if positive else x < 0),
        f"{owner} must be finite and {'positive' if positive else 'nonnegative'}.",
    )


def _solve(matrix: Array, rhs: Array) -> Array:
    result = solve(LinearSystem(DenseLinearOperator(matrix)), rhs)
    return eqx.error_if(
        result.value,
        jnp.any(result.status != 0),
        "Building algebraic system is singular or its linear solve failed.",
    )


class Zone(StrictModule):
    """One well-mixed air node; capacity in J/K, volume in m³."""

    zone_id: str = eqx.field(static=True)
    capacity: Array
    volume: Array
    massless: bool = eqx.field(static=True)

    def __init__(
        self,
        zone_id: str,
        capacity: ArrayLike,
        *,
        volume: ArrayLike = 1.0,
        massless: bool = False,
    ):
        self.zone_id = _text(zone_id, "zone_id")
        self.capacity = _scalar(capacity, "capacity", positive=not massless)
        self.capacity = eqx.error_if(
            self.capacity,
            massless & (self.capacity != 0),
            "A declared massless node must have exactly zero capacity.",
        )
        self.volume = _scalar(volume, "volume", positive=True)
        self.massless = bool(massless)


class Construction(StrictModule):
    """Explicit single-centre wall reduction: resistance m² K/W, capacity J/(m² K).

    Zero areal capacity gives a direct conductance. Positive capacity creates one
    wall state with two half-resistances. Film resistances must be included by caller.
    """

    construction_id: str = eqx.field(static=True)
    resistance: Array
    areal_capacity: Array
    massive: bool = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        construction_id: str,
        resistance: ArrayLike,
        areal_capacity: ArrayLike = 0.0,
        *,
        massive: bool = False,
        provenance: Sequence[str] = (),
    ):
        self.construction_id = _text(construction_id, "construction_id")
        self.resistance = _scalar(resistance, "resistance", positive=True)
        self.areal_capacity = _scalar(areal_capacity, "areal_capacity", positive=massive)
        self.areal_capacity = eqx.error_if(
            self.areal_capacity,
            (not massive) & (self.areal_capacity != 0),
            "Nonzero wall capacity requires an explicit massive reduction.",
        )
        self.massive = bool(massive)
        self.provenance = tuple(provenance)


class Aperture(StrictModule):
    aperture_id: str = eqx.field(static=True)
    area: Array
    u_value: Array
    solar_transmittance: Array

    def __init__(
        self,
        aperture_id: str,
        area: ArrayLike,
        u_value: ArrayLike,
        solar_transmittance: ArrayLike = 0.0,
    ):
        self.aperture_id = _text(aperture_id, "aperture_id")
        self.area = _scalar(area, "aperture area", positive=True)
        self.u_value = _scalar(u_value, "aperture U-value", positive=True)
        transmittance = _scalar(solar_transmittance, "solar transmittance")
        self.solar_transmittance = eqx.error_if(
            transmittance, transmittance > 1, "Solar transmittance must be in [0,1]."
        )


class BuildingBoundary(StrictModule):
    """One ordered environmental temperature input, never an inferred zone."""

    boundary_id: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def __init__(self, boundary_id: str, *, kind: str = "ambient"):
        if kind not in ("ambient", "ground", "fixed"):
            raise ValueError("Boundary kind must be ambient, ground, or fixed.")
        self.boundary_id, self.kind = _text(boundary_id, "boundary_id"), kind


class Surface(StrictModule):
    """Gross area, with explicit adjacent-zone, named environment, or adiabatic semantics."""

    surface_id: str = eqx.field(static=True)
    zone_id: str = eqx.field(static=True)
    adjacent_zone: str | None = eqx.field(static=True)
    boundary_id: str | None = eqx.field(static=True)
    adiabatic: bool = eqx.field(static=True)
    area: Array
    construction: Construction
    apertures: tuple[Aperture, ...]
    geometry_binding: str = eqx.field(static=True)

    def __init__(
        self,
        surface_id: str,
        zone_id: str,
        area: ArrayLike,
        construction: Construction,
        *,
        adjacent_zone: str | None = None,
        apertures: Sequence[Aperture] = (),
        geometry_binding: str = "",
        boundary_id: str | None = None,
        adiabatic: bool = False,
    ):
        self.surface_id = _text(surface_id, "surface_id")
        self.zone_id = _text(zone_id, "zone_id")
        if adjacent_zone == zone_id:
            raise ValueError("Surface cannot be adjacent to its own zone.")
        if not isinstance(construction, Construction):
            raise TypeError("construction must be Construction.")
        self.adjacent_zone = adjacent_zone
        if (adjacent_zone is not None and boundary_id is not None) or (
            adiabatic and (adjacent_zone is not None or boundary_id is not None)
        ):
            raise ValueError(
                "Surface must select exactly one adjacent-zone, boundary, or adiabatic role."
            )
        self.boundary_id = (
            None
            if adjacent_zone is not None or adiabatic
            else ("outdoor" if boundary_id is None else _text(boundary_id, "boundary_id"))
        )
        self.adiabatic = bool(adiabatic)
        self.construction = construction
        self.apertures = tuple(apertures)
        if len({a.aperture_id for a in self.apertures}) != len(self.apertures):
            raise ValueError("Aperture identifiers must be unique on a surface.")
        area_ = _scalar(area, "surface area", positive=True)
        aperture_area = sum((a.area for a in self.apertures), jnp.asarray(0.0))
        self.area = eqx.error_if(
            area_,
            aperture_area > area_,
            "Apertures exceed their gross parent surface area.",
        )
        self.geometry_binding = str(geometry_binding)


class Adjacency(StrictModule):
    """Additional directed reporting edge; heat flows from left to right, W/K."""

    edge_id: str = eqx.field(static=True)
    left: str = eqx.field(static=True)
    right: str | None = eqx.field(static=True)
    boundary_id: str | None = eqx.field(static=True)
    conductance: Array

    def __init__(
        self,
        edge_id: str,
        left: str,
        right: str | None,
        conductance: ArrayLike,
        *,
        boundary_id: str | None = None,
    ):
        if left == right:
            raise ValueError("Adjacency needs distinct nodes.")
        self.edge_id = _text(edge_id, "edge_id")
        self.left = _text(left, "left")
        self.right = right
        if right is not None and boundary_id is not None:
            raise ValueError(
                "Internal adjacency cannot also reference an environmental boundary."
            )
        self.boundary_id = (
            None
            if right is not None
            else ("outdoor" if boundary_id is None else _text(boundary_id, "boundary_id"))
        )
        self.conductance = _scalar(conductance, "conductance", positive=True)


class VentilationExchange(StrictModule):
    """Sensible ventilation/infiltration conductance m_dot * c_p in W/K.

    Supply-air temperature is an explicitly named boundary input. This is not a
    moisture model. Exchange conductance may be differentiated or recompiled for
    another frozen operating point; no air-change-rate/unit inference occurs.
    """

    exchange_id: str = eqx.field(static=True)
    zone_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    conductance: Array

    def __init__(
        self,
        exchange_id: str,
        zone_id: str,
        conductance: ArrayLike,
        *,
        boundary_id: str = "outdoor",
        kind: str = "ventilation",
    ):
        if kind not in ("ventilation", "infiltration"):
            raise ValueError("Air exchange kind must be ventilation or infiltration.")
        self.exchange_id, self.zone_id = (
            _text(exchange_id, "exchange_id"),
            _text(zone_id, "zone_id"),
        )
        self.boundary_id, self.kind = _text(boundary_id, "boundary_id"), kind
        self.conductance = _scalar(conductance, "air exchange conductance")


class BuildingSource(StrictModule):
    zones: tuple[Zone, ...]
    surfaces: tuple[Surface, ...]
    adjacencies: tuple[Adjacency, ...]
    boundaries: tuple[BuildingBoundary, ...]
    ventilation: tuple[VentilationExchange, ...]
    source_id: str = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        zones: Sequence[Zone],
        *,
        surfaces: Sequence[Surface] = (),
        adjacencies: Sequence[Adjacency] = (),
        source_id: str,
        provenance: Sequence[str] = (),
        boundaries: Sequence[BuildingBoundary] | None = None,
        ventilation: Sequence[VentilationExchange] = (),
    ):
        self.zones, self.surfaces, self.adjacencies = (
            tuple(zones),
            tuple(surfaces),
            tuple(adjacencies),
        )
        self.boundaries = (
            (BuildingBoundary("outdoor"),) if boundaries is None else tuple(boundaries)
        )
        self.ventilation = tuple(ventilation)
        boundary_ids = tuple(boundary.boundary_id for boundary in self.boundaries)
        if len(set(boundary_ids)) != len(boundary_ids):
            raise ValueError("Building boundary identifiers must be unique and ordered.")
        if len({exchange.exchange_id for exchange in self.ventilation}) != len(
            self.ventilation
        ):
            raise ValueError(
                "Ventilation/infiltration exchange identifiers must be unique."
            )
        ids = tuple(z.zone_id for z in self.zones)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Building requires nonempty uniquely named zones.")
        for entries in (self.surfaces, self.adjacencies):
            names = [
                e.surface_id if isinstance(e, Surface) else e.edge_id for e in entries
            ]
            if len(set(names)) != len(names):
                raise ValueError(
                    "Source element identifiers must be unique within their kind."
                )
        for surface in self.surfaces:
            if surface.zone_id not in ids or (
                surface.adjacent_zone is not None and surface.adjacent_zone not in ids
            ):
                raise ValueError("Surface references an unknown zone.")
            if (
                surface.boundary_id is not None
                and surface.boundary_id not in boundary_ids
            ):
                raise ValueError("Surface references an unknown environmental boundary.")
        for edge in self.adjacencies:
            if edge.left not in ids or (edge.right is not None and edge.right not in ids):
                raise ValueError("Adjacency references an unknown zone.")
            if edge.boundary_id is not None and edge.boundary_id not in boundary_ids:
                raise ValueError(
                    "Adjacency references an unknown environmental boundary."
                )
        for exchange in self.ventilation:
            if exchange.zone_id not in ids or exchange.boundary_id not in boundary_ids:
                raise ValueError(
                    "Air exchange references an unknown zone or supply-air boundary."
                )
        self.source_id = _text(source_id, "source_id")
        self.provenance = tuple(provenance)


class _RCField(StrictModule):
    capacity: Array
    matrix: Array
    boundary: Array

    def __call__(self, time, temperature, inputs, args):
        del time, args
        count = self.boundary.shape[1]
        return (
            contract("ij,j->i", self.matrix, temperature)
            + contract("ib,b->i", self.boundary, inputs[:count])
            + inputs[count:]
        ) / self.capacity


class _RCResidual(StrictModule):
    capacity: Array
    matrix: Array
    boundary: Array

    def __call__(self, time, temperature, rate, inputs, args):
        del time, args
        count = self.boundary.shape[1]
        return (
            self.capacity * rate
            - contract("ij,j->i", self.matrix, temperature)
            - contract("ib,b->i", self.boundary, inputs[:count])
            - inputs[count:]
        )


class BuildingObservation(StrictModule):
    edge_heat_flow: Array
    net_heat: Array
    stored_energy: Array
    comfort_violation: Array
    balance_residual: Array


class BuildingStep(StrictModule):
    temperature: Array
    successful: Array
    residual_estimate: Array


class BuildingCompilation(StrictModule):
    source: BuildingSource
    capacity: Array
    matrix: Array
    boundary_conductance: Array
    incidence: Array
    edge_conductance: Array
    edge_boundary: Array
    system: ContinuousSystem | DifferentialAlgebraicSystem
    node_ids: tuple[str, ...] = eqx.field(static=True)
    edge_ids: tuple[str, ...] = eqx.field(static=True)
    boundary_ids: tuple[str, ...] = eqx.field(static=True)
    dynamic_indices: tuple[int, ...] = eqx.field(static=True)
    algebraic_indices: tuple[int, ...] = eqx.field(static=True)

    def boundary_values(self, boundary_temperature: ArrayLike) -> Array:
        values = jnp.asarray(boundary_temperature)
        if values.shape == () and len(self.boundary_ids) == 1:
            values = values[None]
        if values.shape != (len(self.boundary_ids),):
            raise ValueError(
                "Boundary temperatures must follow the compiled ordered boundary_ids; "
                "scalar is allowed only for one boundary."
            )
        return eqx.error_if(
            values,
            ~jnp.all(jnp.isfinite(values)),
            "Boundary temperatures must be finite K.",
        )

    def forcing(self, boundary_temperature: ArrayLike, heat: ArrayLike) -> Array:
        q = jnp.asarray(heat)
        if q.shape != self.capacity.shape:
            raise ValueError("Heat needs one W value per compiled node.")
        q = eqx.error_if(
            q, ~jnp.all(jnp.isfinite(q)), "Building heat forcing must be finite."
        )
        return (
            contract(
                "ib,b->i",
                self.boundary_conductance,
                self.boundary_values(boundary_temperature),
            )
            + q
        )

    def reduced_affine(self, forcing: Array):
        d, a = (
            jnp.asarray(self.dynamic_indices),
            jnp.asarray(self.algebraic_indices, dtype=int),
        )
        matrix = self.matrix[jnp.ix_(d, d)]
        force = forcing[d]
        if self.algebraic_indices:
            aa = self.matrix[jnp.ix_(a, a)]
            ad = self.matrix[jnp.ix_(a, d)]
            da = self.matrix[jnp.ix_(d, a)]
            lift = -_solve(aa, ad)
            offset = -_solve(aa, forcing[a])
            matrix = matrix + contract("ij,jk->ik", da, lift)
            force = force + contract("ij,j->i", da, offset)
        return matrix / self.capacity[d, None], force / self.capacity[d]

    def consistent_temperature(
        self, temperature: ArrayLike, boundary_temperature: ArrayLike, heat: ArrayLike
    ) -> Array:
        value = jnp.asarray(temperature)
        if value.shape != self.capacity.shape:
            raise ValueError("Temperature must contain all compiled nodes.")
        if self.algebraic_indices:
            a, d = jnp.asarray(self.algebraic_indices), jnp.asarray(self.dynamic_indices)
            f = self.forcing(boundary_temperature, heat)
            algebraic = -_solve(
                self.matrix[jnp.ix_(a, a)],
                contract("ij,j->i", self.matrix[jnp.ix_(a, d)], value[d]) + f[a],
            )
            value = value.at[a].set(algebraic)
        return value

    def step(
        self,
        temperature: ArrayLike,
        boundary_temperature: ArrayLike,
        heat: ArrayLike,
        duration: ArrayLike,
        *,
        policy: MatrixFunctionPolicy | None = None,
    ) -> BuildingStep:
        """Exact frozen affine flow, including index-one algebraic elimination.

        Massless temperatures are consistently projected at the start and end;
        their initial guesses do not represent stored energy.
        """
        value = self.consistent_temperature(temperature, boundary_temperature, heat)
        matrix, force = self.reduced_affine(self.forcing(boundary_temperature, heat))
        d = jnp.asarray(self.dynamic_indices)
        result = affine_exponential_step(
            DenseLinearOperator(matrix), value[d], force, duration, policy=policy
        )
        end = self.consistent_temperature(
            value.at[d].set(result.value), boundary_temperature, heat
        )
        return BuildingStep(
            end, result.successful & jnp.all(jnp.isfinite(end)), result.residual_estimate
        )

    def observe(
        self,
        temperature: ArrayLike,
        boundary_temperature: ArrayLike,
        heat: ArrayLike,
        *,
        temperature_rate: ArrayLike | None = None,
        reference_temperature: float = 273.15,
        comfort_lower: ArrayLike = 293.15,
        comfort_upper: ArrayLike = 299.15,
    ) -> BuildingObservation:
        value = jnp.asarray(temperature)
        f = self.forcing(boundary_temperature, heat)
        net = contract("ij,j->i", self.matrix, value) + f
        flow = self.edge_conductance * (
            contract("ei,i->e", self.incidence, value)
            - contract(
                "eb,b->e", self.edge_boundary, self.boundary_values(boundary_temperature)
            )
        )
        lower, upper = jnp.asarray(comfort_lower), jnp.asarray(comfort_upper)
        lower = eqx.error_if(
            lower, jnp.any(lower > upper), "Comfort bounds must be ordered."
        )
        zone_t = value[: len(self.source.zones)]
        comfort = jnp.maximum(lower - zone_t, 0) + jnp.maximum(zone_t - upper, 0)
        rate = (
            jnp.where(
                self.capacity > 0, net / jnp.where(self.capacity > 0, self.capacity, 1), 0
            )
            if temperature_rate is None
            else jnp.asarray(temperature_rate)
        )
        return BuildingObservation(
            flow,
            net,
            self.capacity * (value - reference_temperature),
            comfort,
            self.capacity * rate - net,
        )


def compile_building(source: BuildingSource) -> BuildingCompilation:
    """Compile explicitly chosen reductions into native ODE/DAE objects."""
    ids = [z.zone_id for z in source.zones]
    capacity = [z.capacity for z in source.zones]
    dynamic = [i for i, z in enumerate(source.zones) if not z.massless]
    algebraic = tuple(i for i, z in enumerate(source.zones) if z.massless)
    boundary_ids = tuple(boundary.boundary_id for boundary in source.boundaries)
    edges: list[tuple[str, str, str | None, Array, str | None]] = []
    for s in source.surfaces:
        opaque = s.area - sum((a.area for a in s.apertures), jnp.asarray(0.0))
        if s.construction.massive:
            opaque = eqx.error_if(
                opaque, opaque <= 0, "Massive surfaces need positive opaque area."
            )
            wall = f"wall:{s.surface_id}"
            if wall in ids:
                raise ValueError("Zone ID collides with a compiled wall ID.")
            dynamic.append(len(ids))
            ids.append(wall)
            capacity.append(opaque * s.construction.areal_capacity)
            conductance = 2 * opaque / s.construction.resistance
            edges.append((s.surface_id + ":inside", s.zone_id, wall, conductance, None))
            if not s.adiabatic:
                edges.append(
                    (
                        s.surface_id + ":outside",
                        wall,
                        s.adjacent_zone,
                        conductance,
                        s.boundary_id,
                    )
                )
        elif not s.adiabatic:
            edges.append(
                (
                    s.surface_id + ":opaque",
                    s.zone_id,
                    s.adjacent_zone,
                    opaque / s.construction.resistance,
                    s.boundary_id,
                )
            )
        if not s.adiabatic:
            edges.extend(
                (
                    s.surface_id + ":" + a.aperture_id,
                    s.zone_id,
                    s.adjacent_zone,
                    a.area * a.u_value,
                    s.boundary_id,
                )
                for a in s.apertures
            )
    edges.extend(
        (e.edge_id, e.left, e.right, e.conductance, e.boundary_id)
        for e in source.adjacencies
    )
    edges.extend(
        (e.kind + ":" + e.exchange_id, e.zone_id, None, e.conductance, e.boundary_id)
        for e in source.ventilation
    )
    if not dynamic:
        raise ValueError("Building evolution needs at least one positive-capacity node.")
    if len({e[0] for e in edges}) != len(edges):
        raise ValueError("Compiled heat-edge IDs collide.")
    # Every massless connected component must be anchored to a dynamic node or exterior.
    anchored = {ids[i] for i in dynamic} | {None}
    for _ in ids:
        for _, left, right, _, _ in edges:
            if left in anchored or right in anchored:
                anchored.update((left, right))
    if any(ids[i] not in anchored for i in algebraic):
        raise ValueError("Unanchored massless component has an undetermined temperature.")
    n = len(ids)
    incidence = jnp.zeros((len(edges), n))
    boundary_incidence = jnp.zeros((len(edges), len(boundary_ids)))
    for k, (_, left, right, _, boundary_id) in enumerate(edges):
        incidence = incidence.at[k, ids.index(left)].set(1)
        if right is None:
            boundary_incidence = boundary_incidence.at[
                k, boundary_ids.index(boundary_id)
            ].set(1)
        else:
            incidence = incidence.at[k, ids.index(right)].set(-1)
    g = jnp.stack([e[3] for e in edges]) if edges else jnp.zeros((0,))
    c = jnp.stack(capacity)
    matrix = -contract("ei,e,ej->ij", incidence, g, incidence)
    boundary = contract("ei,e,eb->ib", incidence, g, boundary_incidence)
    inputs = InputLayout(
        (n + len(boundary_ids),),
        roles="forcing",
        component_names=(
            *(f"temperature:{name}" for name in boundary_ids),
            *(f"heat:{name}" for name in ids),
        ),
    )
    if algebraic:
        system = DifferentialAlgebraicSystem(
            _RCResidual(c, matrix, boundary),
            state_shape=(n,),
            structure=DAEStructure(
                tuple("algebraic" if i in algebraic else "differential" for i in range(n))
            ),
            input_layout=inputs,
            system_id=source.source_id + ":rc-dae",
        )
    else:
        system = ContinuousSystem(
            _RCField(c, matrix, boundary),
            state_layout=StateLayout((n,), component_names=ids),
            input_layout=inputs,
            system_id=source.source_id + ":rc-ode",
        )
    return BuildingCompilation(
        source,
        c,
        matrix,
        boundary,
        incidence,
        g,
        boundary_incidence,
        system,
        tuple(ids),
        tuple(e[0] for e in edges),
        boundary_ids,
        tuple(dynamic),
        algebraic,
    )
