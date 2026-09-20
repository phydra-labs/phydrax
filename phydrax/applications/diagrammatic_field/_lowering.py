#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...graph import GraphIR
from ...integration import (
    IntegrationRealization,
    materialize,
    reduce,
    WeightedSampleTarget,
)
from ._core import DiagramGraph, MomentumRoute, RegulatorContract


def _incident_routes(
    diagram: DiagramGraph, vertex_index: int, /
) -> tuple[MomentumRoute, ...]:
    vertex = diagram.vertices[vertex_index]
    grouped: dict[str, list[MomentumRoute]] = {}
    for field in vertex.rule.fields:
        grouped.setdefault(field.field_id, [])
    for line in diagram.lines:
        field_id = line.propagator.field.field_id
        if line.source == vertex.label:
            grouped[field_id].append(line.route)
        if line.target == vertex.label:
            grouped[field_id].append(line.route)
    for leg in diagram.external_legs:
        if leg.vertex == vertex.label:
            grouped[leg.field.field_id].append(leg.route)
    consumed = {field_id: 0 for field_id in grouped}
    ordered: list[MomentumRoute] = []
    for field in vertex.rule.fields:
        offset = consumed[field.field_id]
        ordered.append(grouped[field.field_id][offset])
        consumed[field.field_id] = offset + 1
    return tuple(ordered)


class DiagramEvaluationEvidence(StrictModule, NonTrainableState):
    absolute_value: Array
    phase: Array
    real_sign: Array
    fermion_sign: Array
    conservation_residual: Array
    finite: Array
    successful: Array
    status: Array


class DiagramEvaluation(StrictModule, NonTrainableState):
    amplitude: Array
    factor_values: Array
    evidence: DiagramEvaluationEvidence
    graph_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class DiagramLoweringPlan(StrictModule, NonTrainableState):
    """Immutable resource policy for lowering factor products to GraphIR."""

    maximum_factors: int = eqx.field(static=True)
    maximum_nodes: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_factors: int = 4_096,
        maximum_nodes: int = 8_191,
        maximum_edges: int = 8_190,
    ):
        limits = (int(maximum_factors), int(maximum_nodes), int(maximum_edges))
        if any(value <= 0 for value in limits):
            raise ValueError("Diagram lowering limits must be positive.")
        self.maximum_factors, self.maximum_nodes, self.maximum_edges = limits
        self.plan_id = canonical_fingerprint(
            {"kind": "diagram-lowering-plan", "limits": limits}
        )

    def prepare(self, diagram: DiagramGraph, /) -> "PreparedDiagramEvaluation":
        if not isinstance(diagram, DiagramGraph):
            raise TypeError("diagram must be a DiagramGraph.")
        factor_count = (
            len(diagram.vertices) + len(diagram.lines) + len(diagram.external_legs)
        )
        node_count = 2 * factor_count - 1
        edge_count = 2 * (factor_count - 1)
        if factor_count > self.maximum_factors:
            raise ValueError("Diagram factor count exceeds maximum_factors.")
        if node_count > self.maximum_nodes or edge_count > self.maximum_edges:
            raise ValueError("Lowered computational DAG exceeds node or edge capacity.")

        operation = np.zeros((node_count,), dtype=np.int32)
        factor_index = np.full((node_count,), -1, dtype=np.int32)
        cursor = 0
        for code, count in (
            (1, len(diagram.vertices)),
            (2, len(diagram.lines)),
            (3, len(diagram.external_legs)),
        ):
            operation[cursor : cursor + count] = code
            factor_index[cursor : cursor + count] = np.arange(cursor, cursor + count)
            cursor += count
        operation[factor_count:] = 4
        senders: list[int] = []
        receivers: list[int] = []
        input_slot: list[int] = []
        for factor in range(1, factor_count):
            product_node = factor_count + factor - 1
            previous = 0 if factor == 1 else product_node - 1
            senders.extend((previous, factor))
            receivers.extend((product_node, product_node))
            input_slot.extend((0, 1))
        ir = GraphIR(
            nodes={
                "operation": jnp.asarray(operation),
                "factor_index": jnp.asarray(factor_index),
            },
            edges={"input_slot": jnp.asarray(input_slot, dtype=jnp.int32)},
            senders=jnp.asarray(senders, dtype=jnp.int32),
            receivers=jnp.asarray(receivers, dtype=jnp.int32),
            globals={
                "order": jnp.asarray([diagram.order], dtype=jnp.int32),
                "symmetry_factor": jnp.asarray([diagram.symmetry_factor]),
            },
            n_node=jnp.asarray([node_count], dtype=jnp.int32),
            n_edge=jnp.asarray([edge_count], dtype=jnp.int32),
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-diagram-evaluation",
                "plan": self.plan_id,
                "graph": diagram.graph_id,
                "factors": factor_count,
                "nodes": node_count,
                "edges": edge_count,
            }
        )
        return PreparedDiagramEvaluation(
            diagram,
            ir,
            factor_count,
            node_count,
            prepared_id,
        )


class PreparedDiagramEvaluation(StrictModule, NonTrainableState):
    """Prepared computational DAG; runtime only evaluates fixed-shape factors."""

    diagram: DiagramGraph
    ir: GraphIR
    factor_count: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        diagram: DiagramGraph,
        ir: GraphIR,
        factor_count: int,
        node_count: int,
        prepared_id: str,
        /,
    ):
        self.diagram = diagram
        self.ir = ir
        self.factor_count = int(factor_count)
        self.node_count = int(node_count)
        self.prepared_id = str(prepared_id)

    def evaluate(
        self,
        /,
        *,
        regulator: RegulatorContract | None = None,
        vertex_corrections: ArrayLike | None = None,
    ) -> DiagramEvaluation:
        corrections = (
            jnp.zeros((len(self.diagram.vertices),), dtype=jnp.complex128)
            if vertex_corrections is None
            else jnp.asarray(vertex_corrections)
        )
        if corrections.shape != (len(self.diagram.vertices),):
            raise ValueError("vertex_corrections must provide one value per vertex.")
        factors: list[Array] = []
        for index, vertex in enumerate(self.diagram.vertices):
            factors.append(
                vertex.rule.evaluate(_incident_routes(self.diagram, index))
                + corrections[index]
            )
        factors.extend(
            line.propagator.evaluate(line.route, regulator=regulator)
            for line in self.diagram.lines
        )
        factors.extend(leg.wavefunction for leg in self.diagram.external_legs)
        factor_values = jnp.stack(tuple(jnp.asarray(value) for value in factors)).astype(
            jnp.complex128
        )
        node_values = jnp.zeros((self.node_count,), dtype=factor_values.dtype)
        node_values = node_values.at[: self.factor_count].set(factor_values)
        for factor in range(1, self.factor_count):
            product_node = self.factor_count + factor - 1
            previous = 0 if factor == 1 else product_node - 1
            node_values = node_values.at[product_node].set(
                node_values[previous] * node_values[factor]
            )
        root = self.factor_count - 1 if self.factor_count == 1 else self.node_count - 1
        amplitude = (
            node_values[root]
            * self.diagram.symmetry_factor
            * self.diagram.evidence.fermion_sign
        )
        absolute = jnp.abs(amplitude)
        phase = jnp.where(absolute > 0.0, amplitude / absolute, 1.0 + 0.0j)
        finite = jnp.isfinite(amplitude.real) & jnp.isfinite(amplitude.imag)
        successful = finite & self.diagram.evidence.successful
        real_sign = jnp.where(
            jnp.abs(amplitude.imag) <= 64.0 * jnp.finfo(amplitude.real.dtype).eps,
            jnp.sign(amplitude.real),
            0.0,
        )
        evidence = DiagramEvaluationEvidence(
            absolute,
            phase,
            real_sign,
            self.diagram.evidence.fermion_sign,
            self.diagram.evidence.maximum_conservation_residual,
            finite,
            successful,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )
        return DiagramEvaluation(
            amplitude,
            factor_values,
            evidence,
            self.diagram.graph_id,
            self.prepared_id,
        )


class DiagramQuadratureResult(StrictModule, NonTrainableState):
    value: Array
    statistical_uncertainty: Array
    status: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DiagramQuadraturePlan(StrictModule, NonTrainableState):
    """Positive weighted loop-momentum samples for native Phydrax integration."""

    points: Array
    weights: Array
    maximum_samples: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        maximum_samples: int = 1_000_000,
    ):
        points_ = np.asarray(points, dtype=np.float64)
        weights_ = np.asarray(weights, dtype=np.float64)
        maximum = int(maximum_samples)
        if points_.ndim < 2 or points_.shape[0] < 2:
            raise ValueError("Quadrature points need at least two samples.")
        if weights_.shape != (points_.shape[0],):
            raise ValueError("Quadrature weights must align with the sample axis.")
        if (
            maximum <= 0
            or points_.shape[0] > maximum
            or not np.all(np.isfinite(points_))
            or not np.all(np.isfinite(weights_))
            or np.any(weights_ <= 0.0)
        ):
            raise ValueError(
                "Quadrature samples violate finiteness, positivity, or capacity."
            )
        self.points = jnp.asarray(points_)
        self.weights = jnp.asarray(weights_)
        self.maximum_samples = maximum
        self.sample_count = points_.shape[0]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "diagram-quadrature-plan",
                "points": array_tree_fingerprint(points_),
                "weights": array_tree_fingerprint(weights_),
                "maximum_samples": maximum,
            }
        )

    def prepare(self, /) -> "PreparedDiagramQuadrature":
        total_mass = jnp.sum(self.weights)
        target = WeightedSampleTarget(
            self.points,
            jnp.log(self.weights),
            normalized=False,
            target_mass=total_mass,
            independent=True,
            sample_axes=0,
            provenance=f"diagram-quadrature:{self.plan_id}",
        )
        realization = materialize(target)
        return PreparedDiagramQuadrature(
            realization,
            self.sample_count,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "prepared-diagram-quadrature", "plan": self.plan_id}
            ),
        )


class PreparedDiagramQuadrature(StrictModule, NonTrainableState):
    realization: IntegrationRealization
    sample_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        realization: IntegrationRealization,
        sample_count: int,
        plan_id: str,
        prepared_id: str,
        /,
    ):
        self.realization = realization
        self.sample_count = int(sample_count)
        self.plan_id = str(plan_id)
        self.prepared_id = str(prepared_id)

    def integrate(self, amplitudes: ArrayLike, /) -> DiagramQuadratureResult:
        values = jnp.asarray(amplitudes)
        if values.shape[:1] != (self.sample_count,):
            raise ValueError("amplitudes must begin with the prepared sample axis.")
        estimate: Any = reduce(values, self.realization)
        value = jnp.asarray(estimate.value.data)
        uncertainty = jnp.asarray(estimate.error_estimate)
        finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(uncertainty))
        successful = finite & jnp.all(estimate.status == 0)
        return DiagramQuadratureResult(
            value,
            uncertainty,
            estimate.status,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "DiagramEvaluation",
    "DiagramEvaluationEvidence",
    "DiagramLoweringPlan",
    "DiagramQuadraturePlan",
    "DiagramQuadratureResult",
    "PreparedDiagramEvaluation",
    "PreparedDiagramQuadrature",
]
