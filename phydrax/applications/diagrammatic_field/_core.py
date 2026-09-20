#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections import Counter, defaultdict
from enum import IntEnum
from itertools import permutations
from typing import Literal, Protocol

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FieldStatistics = Literal["boson", "fermion", "ghost"]


class DiagramStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    DISCONNECTED = 2
    CONSERVATION_FAILURE = 3


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _finite_scalar(value: complex | float, name: str, /) -> complex:
    scalar = complex(value)
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError(f"{name} must be finite.")
    real = 0.0 if scalar.real == 0.0 else scalar.real
    imaginary = 0.0 if scalar.imag == 0.0 else scalar.imag
    return complex(real, imaginary)


def _route_record(route: "MomentumRoute", /) -> tuple[tuple[float, ...], float]:
    momentum = tuple(
        0.0 if float(value) == 0.0 else float(value)
        for value in np.asarray(route.momentum)
    )
    frequency = float(np.asarray(route.frequency))
    return momentum, 0.0 if frequency == 0.0 else frequency


class FieldSpec(StrictModule, NonTrainableState):
    """Typed field identity used by every diagrammatic object."""

    name: str = eqx.field(static=True)
    statistics: FieldStatistics = eqx.field(static=True)
    components: int = eqx.field(static=True)
    mass_dimension: float = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        statistics: FieldStatistics = "boson",
        components: int = 1,
        mass_dimension: float = 1.0,
    ):
        name_ = _identifier(name, "name")
        if statistics not in ("boson", "fermion", "ghost"):
            raise ValueError("statistics must be 'boson', 'fermion', or 'ghost'.")
        components_ = int(components)
        dimension_ = float(mass_dimension)
        if components_ <= 0:
            raise ValueError("components must be positive.")
        if not np.isfinite(dimension_):
            raise ValueError("mass_dimension must be finite.")
        self.name = name_
        self.statistics = statistics
        self.components = components_
        self.mass_dimension = dimension_
        self.field_id = canonical_fingerprint(
            {
                "kind": "diagrammatic-field",
                "name": name_,
                "statistics": statistics,
                "components": components_,
                "mass_dimension": dimension_,
            }
        )


class MomentumRoute(StrictModule, NonTrainableState):
    """One oriented Euclidean momentum and Matsubara-frequency route."""

    momentum: Array
    frequency: Array
    dimension: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(self, momentum: ArrayLike, frequency: ArrayLike = 0.0, /):
        momentum_host = np.asarray(momentum, dtype=np.float64)
        frequency_host = np.asarray(frequency, dtype=np.float64)
        if momentum_host.ndim != 1 or momentum_host.size == 0:
            raise ValueError("momentum must be a non-empty rank-one vector.")
        if frequency_host.shape != ():
            raise ValueError("frequency must be scalar.")
        if not np.all(np.isfinite(momentum_host)) or not np.isfinite(frequency_host):
            raise ValueError("Momentum routes must be finite.")
        momentum_host = np.where(momentum_host == 0.0, 0.0, momentum_host)
        frequency_host = np.where(frequency_host == 0.0, 0.0, frequency_host)
        self.momentum = jnp.asarray(momentum_host)
        self.frequency = jnp.asarray(frequency_host)
        self.dimension = momentum_host.size
        self.route_id = canonical_fingerprint(
            {
                "kind": "momentum-frequency-route",
                "momentum": array_tree_fingerprint(momentum_host),
                "frequency": float(frequency_host),
            }
        )

    def reversed(self) -> "MomentumRoute":
        return MomentumRoute(-self.momentum, -self.frequency)


class RegulatorContract(Protocol):
    """Structural contract for additive inverse-propagator regulation."""

    regulator_id: str

    def inverse_propagator_shift(self, momentum_squared: Array, /) -> Array: ...


class SmoothEuclideanRegulator(StrictModule, NonTrainableState):
    """Smooth finite-scale regulator R_k(p²)=s k² exp(-p²/k²)."""

    scale: Array
    strength: Array
    regulator_id: str = eqx.field(static=True)

    def __init__(self, scale: float, /, *, strength: float = 1.0):
        scale_, strength_ = float(scale), float(strength)
        if (
            not np.isfinite(scale_)
            or not np.isfinite(strength_)
            or scale_ <= 0.0
            or strength_ < 0.0
        ):
            raise ValueError(
                "Regulator scale must be positive and strength non-negative."
            )
        self.scale = jnp.asarray(scale_)
        self.strength = jnp.asarray(strength_)
        self.regulator_id = canonical_fingerprint(
            {"kind": "smooth-euclidean-regulator", "scale": scale_, "strength": strength_}
        )

    def inverse_propagator_shift(self, momentum_squared: Array, /) -> Array:
        ratio = momentum_squared / (self.scale * self.scale)
        return self.strength * self.scale * self.scale * jnp.exp(-ratio)


class PropagatorSpec(StrictModule, NonTrainableState):
    """Euclidean scalar kernel attached to a typed field."""

    field: FieldSpec
    mass: Array
    residue: Array
    propagator_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: FieldSpec,
        /,
        *,
        mass: float = 0.0,
        residue: complex = 1.0,
    ):
        if not isinstance(field, FieldSpec):
            raise TypeError("field must be a FieldSpec.")
        mass_ = float(mass)
        residue_ = _finite_scalar(residue, "residue")
        if not np.isfinite(mass_) or mass_ < 0.0:
            raise ValueError("mass must be finite and non-negative.")
        self.field = field
        self.mass = jnp.asarray(mass_)
        self.residue = jnp.asarray(residue_)
        self.propagator_id = canonical_fingerprint(
            {
                "kind": "euclidean-propagator",
                "field": field.field_id,
                "mass": mass_,
                "residue": (residue_.real, residue_.imag),
            }
        )

    def evaluate(
        self,
        route: MomentumRoute,
        /,
        *,
        regulator: RegulatorContract | None = None,
    ) -> Array:
        if not isinstance(route, MomentumRoute):
            raise TypeError("route must be a MomentumRoute.")
        momentum_squared = (
            jnp.vdot(route.momentum, route.momentum).real + route.frequency**2
        )
        shift = (
            jnp.asarray(0.0, dtype=momentum_squared.dtype)
            if regulator is None
            else regulator.inverse_propagator_shift(momentum_squared)
        )
        return self.residue / (momentum_squared + self.mass * self.mass + shift)


class VertexRule(StrictModule, NonTrainableState):
    """Typed local interaction rule with explicit perturbative order."""

    name: str = eqx.field(static=True)
    fields: tuple[FieldSpec, ...]
    coupling: Array
    perturbative_order: int = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        fields: tuple[FieldSpec, ...],
        coupling: complex,
        /,
        *,
        perturbative_order: int = 1,
        derivative_order: int = 0,
    ):
        name_ = _identifier(name, "name")
        fields_ = tuple(fields)
        if len(fields_) < 2 or any(not isinstance(field, FieldSpec) for field in fields_):
            raise ValueError("Vertex fields must contain at least two FieldSpec values.")
        coupling_ = _finite_scalar(coupling, "coupling")
        order_, derivative_ = int(perturbative_order), int(derivative_order)
        if order_ <= 0 or derivative_ < 0:
            raise ValueError(
                "perturbative_order must be positive and derivative_order non-negative."
            )
        self.name = name_
        self.fields = fields_
        self.coupling = jnp.asarray(coupling_)
        self.perturbative_order = order_
        self.derivative_order = derivative_
        self.rule_id = canonical_fingerprint(
            {
                "kind": "diagrammatic-vertex-rule",
                "name": name_,
                "fields": [field.field_id for field in fields_],
                "coupling": (coupling_.real, coupling_.imag),
                "perturbative_order": order_,
                "derivative_order": derivative_,
            }
        )

    def evaluate(self, incident_routes: tuple[MomentumRoute, ...], /) -> Array:
        if len(incident_routes) != len(self.fields):
            raise ValueError("Vertex evaluation needs one route per field leg.")
        if self.derivative_order == 0:
            return self.coupling
        invariant = sum(
            jnp.vdot(route.momentum, route.momentum).real + route.frequency**2
            for route in incident_routes
        )
        return self.coupling * invariant ** (0.5 * self.derivative_order)


class VertexInsertion(StrictModule, NonTrainableState):
    """One labeled interaction insertion before graph canonicalization."""

    label: str = eqx.field(static=True)
    rule: VertexRule

    def __init__(self, label: str, rule: VertexRule, /):
        if not isinstance(rule, VertexRule):
            raise TypeError("rule must be a VertexRule.")
        self.label = _identifier(label, "label")
        self.rule = rule


class PropagatorLine(StrictModule, NonTrainableState):
    """One oriented internal contraction between vertex insertions."""

    label: str = eqx.field(static=True)
    propagator: PropagatorSpec
    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    route: MomentumRoute

    def __init__(
        self,
        label: str,
        propagator: PropagatorSpec,
        source: str,
        target: str,
        route: MomentumRoute,
        /,
    ):
        if not isinstance(propagator, PropagatorSpec):
            raise TypeError("propagator must be a PropagatorSpec.")
        if not isinstance(route, MomentumRoute):
            raise TypeError("route must be a MomentumRoute.")
        self.label = _identifier(label, "label")
        self.propagator = propagator
        self.source = _identifier(source, "source")
        self.target = _identifier(target, "target")
        self.route = route


class ExternalState(StrictModule, NonTrainableState):
    """Typed asymptotic state before attachment to a generated graph."""

    label: str = eqx.field(static=True)
    field: FieldSpec
    route: MomentumRoute
    incoming: bool = eqx.field(static=True)
    wavefunction: Array

    def __init__(
        self,
        label: str,
        field: FieldSpec,
        route: MomentumRoute,
        /,
        *,
        incoming: bool,
        wavefunction: complex = 1.0,
    ):
        if not isinstance(field, FieldSpec) or not isinstance(route, MomentumRoute):
            raise TypeError("External states require FieldSpec and MomentumRoute values.")
        wavefunction_ = _finite_scalar(wavefunction, "wavefunction")
        self.label = _identifier(label, "label")
        self.field = field
        self.route = route
        self.incoming = bool(incoming)
        self.wavefunction = jnp.asarray(wavefunction_)


class ExternalLeg(StrictModule, NonTrainableState):
    """Typed asymptotic state attached to one interaction vertex."""

    label: str = eqx.field(static=True)
    field: FieldSpec
    vertex: str = eqx.field(static=True)
    route: MomentumRoute
    incoming: bool = eqx.field(static=True)
    wavefunction: Array

    def __init__(
        self,
        label: str,
        field: FieldSpec,
        vertex: str,
        route: MomentumRoute,
        /,
        *,
        incoming: bool,
        wavefunction: complex = 1.0,
    ):
        if not isinstance(field, FieldSpec) or not isinstance(route, MomentumRoute):
            raise TypeError("External legs require FieldSpec and MomentumRoute values.")
        wavefunction_ = _finite_scalar(wavefunction, "wavefunction")
        self.label = _identifier(label, "label")
        self.field = field
        self.vertex = _identifier(vertex, "vertex")
        self.route = route
        self.incoming = bool(incoming)
        self.wavefunction = jnp.asarray(wavefunction_)

    @classmethod
    def from_state(cls, state: ExternalState, vertex: str, /) -> "ExternalLeg":
        return cls(
            state.label,
            state.field,
            vertex,
            state.route,
            incoming=state.incoming,
            wavefunction=complex(np.asarray(state.wavefunction)),
        )


class DiagramEvidence(StrictModule, NonTrainableState):
    """Conservation, topology, and fermionic-sign evidence for one diagram."""

    conservation_residual: Array
    maximum_conservation_residual: Array
    connected: Array
    finite: Array
    successful: Array
    fermion_loop_count: Array
    fermion_sign: Array
    status: Array


def _canonical_representation(
    vertices: tuple[VertexInsertion, ...],
    lines: tuple[PropagatorLine, ...],
    external_legs: tuple[ExternalLeg, ...],
    permutation: tuple[int, ...],
    /,
    *,
    include_internal_routes: bool,
) -> tuple[
    tuple[str, ...], tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]
]:
    position = {vertices[old].label: new for new, old in enumerate(permutation)}
    vertex_record = tuple(vertices[index].rule.rule_id for index in permutation)
    line_record: list[tuple[object, ...]] = []
    for line in lines:
        source, target = position[line.source], position[line.target]
        directed = line.propagator.field.statistics != "boson"
        canonical_route = line.route
        if not directed and target < source:
            source, target = target, source
            canonical_route = canonical_route.reversed()
        route: object = _route_record(canonical_route) if include_internal_routes else ()
        propagator_identity = (
            line.propagator.propagator_id
            if include_internal_routes
            else line.propagator.field.field_id
        )
        line_record.append((source, target, propagator_identity, directed, route))
    external_record = [
        (
            position[leg.vertex],
            leg.field.field_id,
            leg.incoming,
            _route_record(leg.route),
            complex(np.asarray(leg.wavefunction)).real,
            complex(np.asarray(leg.wavefunction)).imag,
        )
        for leg in external_legs
    ]
    return vertex_record, tuple(sorted(line_record)), tuple(sorted(external_record))


def _is_connected(labels: tuple[str, ...], lines: tuple[PropagatorLine, ...], /) -> bool:
    adjacency = {label: set() for label in labels}
    for line in lines:
        adjacency[line.source].add(line.target)
        adjacency[line.target].add(line.source)
    visited = {labels[0]}
    frontier = [labels[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == len(labels)


def _fermion_cycles(labels: tuple[str, ...], lines: tuple[PropagatorLine, ...], /) -> int:
    fermion_lines = tuple(
        line for line in lines if line.propagator.field.statistics in ("fermion", "ghost")
    )
    incident = {label for line in fermion_lines for label in (line.source, line.target)}
    visited: set[str] = set()
    cycles = 0
    for start in sorted(incident):
        if start in visited:
            continue
        component = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            visited.add(current)
            for line in fermion_lines:
                if line.source == current or line.target == current:
                    other = line.target if line.source == current else line.source
                    if other not in component:
                        component.add(other)
                        frontier.append(other)
        edge_count = sum(
            line.source in component and line.target in component
            for line in fermion_lines
        )
        cycles += max(0, edge_count - len(component) + 1)
    return cycles


def _line_symmetry_multiplier(lines: tuple[PropagatorLine, ...], /) -> int:
    groups: Counter[tuple[str, str, str, bool]] = Counter()
    multiplier = 1
    for line in lines:
        directed = line.propagator.field.statistics != "boson"
        source, target = line.source, line.target
        if not directed and target < source:
            source, target = target, source
        groups[(source, target, line.propagator.propagator_id, directed)] += 1
        if not directed and source == target:
            multiplier *= 2
    for count in groups.values():
        multiplier *= math.factorial(count)
    return multiplier


class DiagramGraph(StrictModule, NonTrainableState):
    """Canonical finite multigraph with routed internal and external lines."""

    vertices: tuple[VertexInsertion, ...]
    lines: tuple[PropagatorLine, ...]
    external_legs: tuple[ExternalLeg, ...]
    evidence: DiagramEvidence
    order: int = eqx.field(static=True)
    loop_order: int = eqx.field(static=True)
    automorphism_count: int = eqx.field(static=True)
    symmetry_factor: float = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: tuple[VertexInsertion, ...],
        lines: tuple[PropagatorLine, ...],
        external_legs: tuple[ExternalLeg, ...] = (),
        /,
        *,
        maximum_canonical_permutations: int = 40_320,
        conservation_tolerance: float = 1.0e-10,
    ):
        vertices_, lines_, external_ = tuple(vertices), tuple(lines), tuple(external_legs)
        if not vertices_ or any(
            not isinstance(item, VertexInsertion) for item in vertices_
        ):
            raise ValueError("vertices must contain at least one VertexInsertion.")
        if any(not isinstance(item, PropagatorLine) for item in lines_):
            raise TypeError("lines must contain PropagatorLine values.")
        if any(not isinstance(item, ExternalLeg) for item in external_):
            raise TypeError("external_legs must contain ExternalLeg values.")
        labels = tuple(vertex.label for vertex in vertices_)
        if len(set(labels)) != len(labels):
            raise ValueError("Vertex labels must be unique before canonicalization.")
        if len({line.label for line in lines_}) != len(lines_):
            raise ValueError("Propagator labels must be unique before canonicalization.")
        if len({leg.label for leg in external_}) != len(external_):
            raise ValueError(
                "External-leg labels must be unique before canonicalization."
            )
        known = set(labels)
        if any(line.source not in known or line.target not in known for line in lines_):
            raise ValueError("Every propagator endpoint must name a graph vertex.")
        if any(leg.vertex not in known for leg in external_):
            raise ValueError("Every external leg must attach to a graph vertex.")
        routes = tuple(line.route for line in lines_) + tuple(
            leg.route for leg in external_
        )
        if not routes:
            raise ValueError(
                "A diagram must contain at least one routed line or external leg."
            )
        dimension = routes[0].dimension
        if any(route.dimension != dimension for route in routes):
            raise ValueError("All diagram routes must have one momentum dimension.")
        tolerance = float(conservation_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conservation_tolerance must be finite and non-negative.")
        permutation_count = math.factorial(len(vertices_))
        maximum = int(maximum_canonical_permutations)
        if maximum <= 0 or permutation_count > maximum:
            raise ValueError(
                "Diagram canonicalization exceeds maximum_canonical_permutations."
            )

        incident: dict[str, list[str]] = defaultdict(list)
        for line in lines_:
            incident[line.source].append(line.propagator.field.field_id)
            incident[line.target].append(line.propagator.field.field_id)
        for leg in external_:
            incident[leg.vertex].append(leg.field.field_id)
        for vertex in vertices_:
            expected = Counter(field.field_id for field in vertex.rule.fields)
            if Counter(incident[vertex.label]) != expected:
                raise ValueError(
                    f"Incident fields do not match vertex rule at {vertex.label!r}."
                )

        candidates = (
            (
                _canonical_representation(
                    vertices_,
                    lines_,
                    external_,
                    permutation,
                    include_internal_routes=True,
                ),
                permutation,
            )
            for permutation in permutations(range(len(vertices_)))
        )
        representation, canonical_permutation = min(candidates, key=lambda item: item[0])
        canonical_position = {
            vertices_[old].label: new for new, old in enumerate(canonical_permutation)
        }
        canonical_vertices = tuple(
            VertexInsertion(f"v{new}", vertices_[old].rule)
            for new, old in enumerate(canonical_permutation)
        )
        canonical_lines_unsorted: list[PropagatorLine] = []
        for line in lines_:
            source_index = canonical_position[line.source]
            target_index = canonical_position[line.target]
            route = line.route
            if (
                line.propagator.field.statistics == "boson"
                and target_index < source_index
            ):
                source_index, target_index = target_index, source_index
                route = route.reversed()
            canonical_lines_unsorted.append(
                PropagatorLine(
                    line.label,
                    line.propagator,
                    f"v{source_index}",
                    f"v{target_index}",
                    route,
                )
            )
        canonical_lines_unsorted.sort(
            key=lambda line: (
                line.source,
                line.target,
                line.propagator.field.field_id,
                _route_record(line.route),
            )
        )
        canonical_lines = tuple(
            PropagatorLine(
                f"p{index}",
                line.propagator,
                line.source,
                line.target,
                line.route,
            )
            for index, line in enumerate(canonical_lines_unsorted)
        )
        canonical_external_unsorted = [
            ExternalLeg(
                leg.label,
                leg.field,
                f"v{canonical_position[leg.vertex]}",
                leg.route,
                incoming=leg.incoming,
                wavefunction=complex(np.asarray(leg.wavefunction)),
            )
            for leg in external_
        ]
        canonical_external_unsorted.sort(
            key=lambda leg: (
                leg.vertex,
                leg.field.field_id,
                leg.incoming,
                _route_record(leg.route),
                leg.label,
            )
        )
        canonical_external = tuple(
            ExternalLeg(
                f"x{index}",
                leg.field,
                leg.vertex,
                leg.route,
                incoming=leg.incoming,
                wavefunction=complex(np.asarray(leg.wavefunction)),
            )
            for index, leg in enumerate(canonical_external_unsorted)
        )

        residual = np.zeros((len(canonical_vertices), dimension + 1), dtype=np.float64)
        for line in canonical_lines:
            route = np.concatenate(
                (np.asarray(line.route.momentum), np.asarray(line.route.frequency)[None])
            )
            source = int(line.source[1:])
            target = int(line.target[1:])
            residual[source] += route
            residual[target] -= route
        for leg in canonical_external:
            route = np.concatenate(
                (np.asarray(leg.route.momentum), np.asarray(leg.route.frequency)[None])
            )
            residual[int(leg.vertex[1:])] += route if not leg.incoming else -route
        maximum_residual = float(np.max(np.abs(residual)))
        connected = _is_connected(
            tuple(vertex.label for vertex in canonical_vertices), canonical_lines
        )
        fermion_loops = _fermion_cycles(
            tuple(vertex.label for vertex in canonical_vertices), canonical_lines
        )
        finite = bool(np.all(np.isfinite(residual)))
        successful = finite and connected and maximum_residual <= tolerance

        topology_reference = _canonical_representation(
            vertices_,
            lines_,
            external_,
            tuple(range(len(vertices_))),
            include_internal_routes=False,
        )
        vertex_automorphisms = sum(
            _canonical_representation(
                vertices_,
                lines_,
                external_,
                permutation,
                include_internal_routes=False,
            )
            == topology_reference
            for permutation in permutations(range(len(vertices_)))
        )
        automorphisms = vertex_automorphisms * _line_symmetry_multiplier(canonical_lines)
        order = sum(vertex.rule.perturbative_order for vertex in canonical_vertices)
        loop_order = max(0, len(canonical_lines) - len(canonical_vertices) + 1)
        graph_id = canonical_fingerprint(
            {
                "kind": "canonical-diagram-graph",
                "vertices": representation[0],
                "lines": representation[1],
                "external_legs": representation[2],
                "order": order,
            }
        )
        self.vertices = canonical_vertices
        self.lines = canonical_lines
        self.external_legs = canonical_external
        self.evidence = DiagramEvidence(
            jnp.asarray(residual),
            jnp.asarray(maximum_residual),
            jnp.asarray(connected),
            jnp.asarray(finite),
            jnp.asarray(successful),
            jnp.asarray(fermion_loops, dtype=jnp.int32),
            jnp.asarray(-1 if fermion_loops % 2 else 1, dtype=jnp.int32),
            jnp.asarray(
                int(
                    DiagramStatus.SUCCESS
                    if successful
                    else (
                        DiagramStatus.NONFINITE
                        if not finite
                        else (
                            DiagramStatus.DISCONNECTED
                            if not connected
                            else DiagramStatus.CONSERVATION_FAILURE
                        )
                    )
                ),
                dtype=jnp.int32,
            ),
        )
        self.order = order
        self.loop_order = loop_order
        self.automorphism_count = automorphisms
        self.symmetry_factor = 1.0 / automorphisms
        self.graph_id = graph_id


__all__ = [
    "DiagramEvidence",
    "DiagramGraph",
    "DiagramStatus",
    "ExternalLeg",
    "ExternalState",
    "FieldSpec",
    "FieldStatistics",
    "MomentumRoute",
    "PropagatorLine",
    "PropagatorSpec",
    "RegulatorContract",
    "SmoothEuclideanRegulator",
    "VertexInsertion",
    "VertexRule",
]
