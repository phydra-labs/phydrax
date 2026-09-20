#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-bearing detector bias and weighting electrostatics.

The detector layer supplies physical electrode and extrusion semantics to the
existing cochain Poisson solver.  Bias and weighting solves intentionally have
different plan and result types: a physical space-charge solution is never a
weighting potential, and a unit weighting excitation is never a device bias.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import StructuredCochainBridge
from ...ein import contract
from ...solver._cochain_electrostatic import (
    CochainElectrostaticBoundaryPlan,
    CochainElectrostaticPlan,
    CochainElectrostaticResult,
    ElectrostaticBoundaryKind,
)
from ...units import (
    COULOMB,
    derived_unit,
    METER,
    ONE,
    UnitDefinition,
    VOLT,
)
from ._quantities import _si, PERMITTIVITY_UNIT, SQUARE_METER


VOLUME_CHARGE_DENSITY_UNIT = derived_unit("C/m3", ((COULOMB, 1), (METER, -3)))


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _positive_array(
    value: ArrayLike,
    unit: UnitDefinition,
    reference: UnitDefinition,
    shape: tuple[int, ...],
    name: str,
    /,
) -> Array:
    converted = _si(value, unit, reference)
    if converted.shape not in ((), shape):
        raise ValueError(f"{name} must be scalar or have shape {shape}.")
    host = np.asarray(converted)
    if np.any(~np.isfinite(host)) or np.any(host <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return jnp.broadcast_to(converted, shape)


class DetectorResourcePolicy(StrictModule, NonTrainableState):
    """Caller-declared hard bounds checked before detector allocations."""

    maximum_nodes: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_electrodes: int = eqx.field(static=True)
    maximum_linear_iterations: int = eqx.field(static=True)
    maximum_trajectory_cases: int = eqx.field(static=True)
    maximum_trajectory_samples: int = eqx.field(static=True)
    maximum_interpolation_routes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_nodes: int,
        maximum_edges: int,
        maximum_electrodes: int,
        maximum_linear_iterations: int,
        maximum_trajectory_cases: int,
        maximum_trajectory_samples: int,
        maximum_interpolation_routes: int,
    ):
        values = {
            "maximum_nodes": maximum_nodes,
            "maximum_edges": maximum_edges,
            "maximum_electrodes": maximum_electrodes,
            "maximum_linear_iterations": maximum_linear_iterations,
            "maximum_trajectory_cases": maximum_trajectory_cases,
            "maximum_trajectory_samples": maximum_trajectory_samples,
            "maximum_interpolation_routes": maximum_interpolation_routes,
        }
        resolved = {
            name: _positive_integer(value, name) for name, value in values.items()
        }
        for name, value in resolved.items():
            setattr(self, name, value)
        self.policy_id = canonical_fingerprint(
            {"kind": "semiconductor-detector-resource-policy", **resolved}
        )

    def admit_field_problem(
        self, node_count: int, edge_count: int, electrode_count: int, /
    ) -> None:
        if node_count > self.maximum_nodes:
            raise ValueError("Detector node count exceeds the declared resource policy.")
        if edge_count > self.maximum_edges:
            raise ValueError("Detector edge count exceeds the declared resource policy.")
        if electrode_count > self.maximum_electrodes:
            raise ValueError(
                "Detector electrode count exceeds the declared resource policy."
            )

    def admit_trajectory(
        self,
        case_count: int,
        sample_count: int,
        route_count: int,
        /,
    ) -> None:
        if case_count > self.maximum_trajectory_cases:
            raise ValueError(
                "Trajectory case count exceeds the declared detector resource policy."
            )
        if sample_count > self.maximum_trajectory_samples:
            raise ValueError(
                "Trajectory sample count exceeds the declared detector resource policy."
            )
        if route_count > self.maximum_interpolation_routes:
            raise ValueError(
                "Trajectory interpolation routes exceed the declared detector resource policy."
            )


class DetectorElectrode(StrictModule, NonTrainableState):
    """One named, nonempty set of physical boundary vertices."""

    name: str = eqx.field(static=True)
    node_mask: Array
    electrode_id: str = eqx.field(static=True)

    def __init__(self, name: str, node_mask: ArrayLike, /):
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("Electrode names must be nonempty canonical strings.")
        raw = np.asarray(node_mask)
        if raw.ndim != 1 or raw.dtype.kind != "b" or not np.any(raw):
            raise ValueError("An electrode mask must be a nonempty Boolean vector.")
        self.name = name
        self.node_mask = jnp.asarray(raw)
        self.electrode_id = canonical_fingerprint(
            {
                "kind": "semiconductor-detector-electrode",
                "name": name,
                "node_mask": array_tree_fingerprint(raw),
            }
        )


class SemiconductorDetectorPlan(StrictModule, NonTrainableState):
    """Fixed linear-dielectric detector geometry and complete electrode ledger.

    ``node_transverse_measure`` and ``edge_transverse_measure`` lift a one- or
    two-dimensional cochain support to physical three-dimensional charge and
    energy.  They have units m² for a one-dimensional support, m for a
    two-dimensional support, and must be exactly one for a three-dimensional
    support.  A radial coaxial reduction therefore supplies its cylindrical
    nodal and edge cross-sectional measures explicitly rather than disguising
    geometry as a dielectric constant.
    """

    bridge: StructuredCochainBridge
    electrodes: tuple[DetectorElectrode, ...]
    electrode_masks: Array
    boundary_mask: Array
    unassigned_boundary_mask: Array
    permittivity: Array
    node_transverse_measure: Array
    edge_transverse_measure: Array
    effective_permittivity: Array
    resources: DetectorResourcePolicy
    evidence_tolerance: float = eqx.field(static=True)
    complete_dirichlet_electrodes: bool = eqx.field(static=True)
    electrode_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        electrodes: Sequence[DetectorElectrode],
        resources: DetectorResourcePolicy,
        /,
        *,
        permittivity: ArrayLike,
        node_transverse_measure: ArrayLike,
        edge_transverse_measure: ArrayLike,
        permittivity_unit: UnitDefinition = PERMITTIVITY_UNIT,
        transverse_unit: UnitDefinition | None = None,
        evidence_tolerance: float = 1.0e-9,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        if not isinstance(resources, DetectorResourcePolicy):
            raise TypeError("resources must be DetectorResourcePolicy.")
        values = tuple(electrodes)
        if not values or not all(
            isinstance(value, DetectorElectrode) for value in values
        ):
            raise TypeError(
                "electrodes must contain one or more DetectorElectrode values."
            )
        names = tuple(value.name for value in values)
        if len(set(names)) != len(names):
            raise ValueError("Detector electrode names must be unique.")
        node_count = bridge.cochain.cell_counts[0]
        edge_count = bridge.cochain.cell_counts[1]
        resources.admit_field_problem(node_count, edge_count, len(values))
        masks = np.stack(tuple(np.asarray(value.node_mask) for value in values))
        if masks.shape != (len(values), node_count):
            raise ValueError("Every electrode mask must match the detector vertices.")
        boundary = np.asarray(bridge.cochain.boundary_masks[0], dtype=np.bool_)
        if not np.any(boundary):
            raise ValueError("Detector electrostatics requires a physical boundary.")
        if np.any(masks & ~boundary[None, :]):
            raise ValueError("Detector electrodes may select only boundary vertices.")
        ownership = np.sum(masks, axis=0)
        if np.any(ownership > 1):
            raise ValueError("Detector electrodes must be pairwise disjoint.")
        union = ownership == 1
        dimension = bridge.dimension
        reference = SQUARE_METER if dimension == 1 else METER if dimension == 2 else ONE
        unit = reference if transverse_unit is None else transverse_unit
        node_measure = _positive_array(
            node_transverse_measure,
            unit,
            reference,
            (node_count,),
            "node_transverse_measure",
        )
        edge_measure = _positive_array(
            edge_transverse_measure,
            unit,
            reference,
            (edge_count,),
            "edge_transverse_measure",
        )
        if dimension == 3 and (
            not np.array_equal(np.asarray(node_measure), np.ones(node_count))
            or not np.array_equal(np.asarray(edge_measure), np.ones(edge_count))
        ):
            raise ValueError(
                "Three-dimensional detector cochains require unit transverse measures."
            )
        epsilon = _positive_array(
            permittivity,
            permittivity_unit,
            PERMITTIVITY_UNIT,
            (edge_count,),
            "permittivity",
        )
        tolerance = float(evidence_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("evidence_tolerance must be finite and positive.")
        self.bridge = bridge
        self.electrodes = values
        self.electrode_masks = jnp.asarray(masks)
        self.boundary_mask = jnp.asarray(boundary)
        self.unassigned_boundary_mask = jnp.asarray(boundary & ~union)
        self.permittivity = epsilon
        self.node_transverse_measure = node_measure
        self.edge_transverse_measure = edge_measure
        self.effective_permittivity = epsilon * edge_measure
        self.resources = resources
        self.evidence_tolerance = tolerance
        self.complete_dirichlet_electrodes = bool(np.array_equal(union, boundary))
        self.electrode_names = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "semiconductor-detector-plan",
                "bridge": bridge.bridge_id,
                "electrodes": [value.electrode_id for value in values],
                "permittivity": array_tree_fingerprint(np.asarray(epsilon)),
                "node_transverse_measure": array_tree_fingerprint(
                    np.asarray(node_measure)
                ),
                "edge_transverse_measure": array_tree_fingerprint(
                    np.asarray(edge_measure)
                ),
                "resources": resources.policy_id,
                "evidence_tolerance": tolerance,
            }
        )

    @property
    def node_count(self) -> int:
        return self.bridge.cochain.cell_counts[0]

    @property
    def edge_count(self) -> int:
        return self.bridge.cochain.cell_counts[1]

    @property
    def electrode_count(self) -> int:
        return len(self.electrodes)

    def effective_space_charge(self, space_charge: Array, /) -> Array:
        return space_charge * self.node_transverse_measure.astype(space_charge.dtype)

    def electrode_reaction_charges(
        self,
        potential: Array,
        effective_space_charge: Array,
        /,
    ) -> Array:
        cochain = self.bridge.cochain
        gradient = cochain.exterior_derivative(0, potential)
        core = cochain.codifferential(
            1, self.effective_permittivity.astype(gradient.dtype) * gradient
        )
        reaction = cochain.hodge_stars[0].astype(core.dtype) * (
            core - effective_space_charge
        )
        return contract("en,n->e", self.electrode_masks.astype(reaction.dtype), reaction)


class DetectorBiasElectrostaticResult(StrictModule):
    """Physical-bias solution with electrode and total-charge evidence."""

    electrostatic: CochainElectrostaticResult
    space_charge_density: Array
    electrode_voltages: Array
    electrode_charges: Array
    charge_closure_defect: Array
    charge_closure_threshold: Array
    finite: Array
    charge_closed: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DetectorBiasElectrostaticPlan(StrictModule, NonTrainableState):
    """Physical detector bias with prescribed fixed space charge and no carriers."""

    detector: SemiconductorDetectorPlan
    electrode_voltages: Array
    electrostatic: CochainElectrostaticPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        detector: SemiconductorDetectorPlan,
        electrode_voltages: ArrayLike,
        /,
        *,
        voltage_unit: UnitDefinition = VOLT,
    ):
        if not isinstance(detector, SemiconductorDetectorPlan):
            raise TypeError("detector must be SemiconductorDetectorPlan.")
        voltages = _si(electrode_voltages, voltage_unit, VOLT)
        if voltages.shape != (detector.electrode_count,):
            raise ValueError("electrode_voltages must contain one value per electrode.")
        if np.any(~np.isfinite(np.asarray(voltages))):
            raise ValueError("electrode_voltages must be finite.")
        values = contract(
            "e,en->n", voltages, detector.electrode_masks.astype(voltages.dtype)
        )
        kind = (
            ElectrostaticBoundaryKind.DIRICHLET
            if detector.complete_dirichlet_electrodes
            else ElectrostaticBoundaryKind.MIXED
        )
        boundary = CochainElectrostaticBoundaryPlan(
            detector.bridge,
            kind,
            dirichlet_mask=jnp.any(detector.electrode_masks, axis=0),
            dirichlet_values=values,
        )
        electrostatic = CochainElectrostaticPlan(
            detector.bridge,
            boundary,
            permittivity=detector.effective_permittivity,
            tolerance=detector.evidence_tolerance,
            compatibility_tolerance=detector.evidence_tolerance,
            maximum_iterations=detector.resources.maximum_linear_iterations,
        )
        self.detector = detector
        self.electrode_voltages = voltages
        self.electrostatic = electrostatic
        self.plan_id = canonical_fingerprint(
            {
                "kind": "semiconductor-detector-bias-plan",
                "detector": detector.plan_id,
                "voltages": array_tree_fingerprint(np.asarray(voltages)),
                "electrostatic": electrostatic.plan_id,
            }
        )

    def solve(
        self,
        space_charge_density: ArrayLike,
        /,
        *,
        charge_density_unit: UnitDefinition = VOLUME_CHARGE_DENSITY_UNIT,
        initial_potential: ArrayLike | None = None,
    ) -> DetectorBiasElectrostaticResult:
        rho = _si(
            space_charge_density,
            charge_density_unit,
            VOLUME_CHARGE_DENSITY_UNIT,
        )
        if rho.shape != (self.detector.node_count,):
            raise ValueError("space_charge_density must contain one value per vertex.")
        rho = eqx.error_if(
            rho,
            jnp.any(~jnp.isfinite(rho)),
            "space_charge_density must be finite.",
        )
        effective = self.detector.effective_space_charge(rho)
        solved = self.electrostatic.solve(effective, initial_potential=initial_potential)
        electrode_charges = self.detector.electrode_reaction_charges(
            solved.potential, effective
        )
        cochain = self.detector.bridge.cochain
        volume_charge = jnp.sum(
            cochain.hodge_stars[0].astype(effective.dtype) * effective
        )
        closure = jnp.abs(jnp.sum(electrode_charges) + volume_charge)
        scale = jnp.maximum(
            jnp.sum(jnp.abs(electrode_charges)) + jnp.abs(volume_charge),
            jnp.finfo(electrode_charges.dtype).tiny,
        )
        threshold = self.detector.evidence_tolerance * scale
        finite = (
            solved.finite
            & jnp.all(jnp.isfinite(electrode_charges))
            & jnp.isfinite(closure)
        )
        closed = closure <= threshold
        return DetectorBiasElectrostaticResult(
            solved,
            rho,
            self.electrode_voltages,
            electrode_charges,
            closure,
            threshold,
            finite,
            closed,
            solved.successful & finite & closed,
            self.plan_id,
        )


class DetectorWeightingEvidence(StrictModule):
    """Conditional complete-electrode, reciprocity, and charge evidence."""

    complete_dirichlet_electrodes: Array
    unassigned_boundary_count: Array
    partition_of_unity_defect: Array
    partition_of_unity_threshold: Array
    partition_of_unity_certified: Array
    capacitance_reciprocity_defect: Array
    capacitance_reciprocity_threshold: Array
    capacitance_reciprocal: Array
    complete_electrode_charge_defect: Array
    complete_electrode_charge_threshold: Array
    complete_electrode_charge_closed: Array
    finite: Array
    converged: Array
    certified: Array


class DetectorWeightingFieldResult(StrictModule):
    """Zero-space-charge one-hot weighting basis and Maxwell capacitance."""

    potentials: Array
    electric: Array
    physical_electric: tuple[Array, ...]
    capacitance: Array
    solves: tuple[CochainElectrostaticResult, ...]
    evidence: DetectorWeightingEvidence
    plan_id: str = eqx.field(static=True)


class DetectorWeightingFieldPlan(StrictModule, NonTrainableState):
    """One zero-charge Poisson solve for every named unit electrode excitation."""

    detector: SemiconductorDetectorPlan
    electrostatics: tuple[CochainElectrostaticPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, detector: SemiconductorDetectorPlan, /):
        if not isinstance(detector, SemiconductorDetectorPlan):
            raise TypeError("detector must be SemiconductorDetectorPlan.")
        # All physical boundary vertices are Dirichlet for weighting solves.
        # Undeclared vertices are an explicit grounded remainder.  Consequently
        # sum(phi_w) == 1 is certified only when declared electrodes are complete.
        electrostatics = []
        for mask in np.asarray(detector.electrode_masks):
            boundary = CochainElectrostaticBoundaryPlan.dirichlet(
                detector.bridge,
                jnp.asarray(mask, dtype=detector.permittivity.dtype),
                mask=detector.boundary_mask,
            )
            electrostatics.append(
                CochainElectrostaticPlan(
                    detector.bridge,
                    boundary,
                    permittivity=detector.effective_permittivity,
                    tolerance=detector.evidence_tolerance,
                    compatibility_tolerance=detector.evidence_tolerance,
                    maximum_iterations=detector.resources.maximum_linear_iterations,
                )
            )
        self.detector = detector
        self.electrostatics = tuple(electrostatics)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "semiconductor-detector-weighting-plan",
                "detector": detector.plan_id,
                "electrostatics": [value.plan_id for value in electrostatics],
            }
        )

    def solve(self) -> DetectorWeightingFieldResult:
        zero = jnp.zeros(
            (self.detector.node_count,), dtype=self.detector.permittivity.dtype
        )
        solves = tuple(plan.solve(zero) for plan in self.electrostatics)
        potentials = jnp.stack(tuple(value.potential for value in solves))
        electric = jnp.stack(tuple(value.electric for value in solves))
        physical = tuple(
            jnp.stack(tuple(value.physical_electric[axis] for value in solves))
            for axis in range(self.detector.bridge.dimension)
        )
        columns = tuple(
            self.detector.electrode_reaction_charges(value.potential, zero)
            for value in solves
        )
        capacitance = jnp.stack(columns, axis=1)
        dtype = potentials.dtype
        partition_defect = jnp.max(jnp.abs(jnp.sum(potentials, axis=0) - 1.0))
        partition_threshold = jnp.asarray(self.detector.evidence_tolerance, dtype=dtype)
        reciprocity_defect = jnp.max(jnp.abs(capacitance - capacitance.T))
        capacitance_scale = jnp.maximum(
            jnp.max(jnp.abs(capacitance)), jnp.finfo(capacitance.dtype).tiny
        )
        reciprocity_threshold = self.detector.evidence_tolerance * capacitance_scale
        column_charge_defect = jnp.max(jnp.abs(jnp.sum(capacitance, axis=0)))
        column_charge_scale = jnp.maximum(
            jnp.max(jnp.sum(jnp.abs(capacitance), axis=0)),
            jnp.finfo(capacitance.dtype).tiny,
        )
        column_charge_threshold = self.detector.evidence_tolerance * column_charge_scale
        complete = jnp.asarray(self.detector.complete_dirichlet_electrodes)
        finite = (
            jnp.all(jnp.stack(tuple(value.finite for value in solves)))
            & jnp.all(jnp.isfinite(potentials))
            & jnp.all(jnp.isfinite(electric))
            & jnp.all(jnp.isfinite(capacitance))
        )
        converged = jnp.all(jnp.stack(tuple(value.converged for value in solves)))
        partition_certified = complete & (partition_defect <= partition_threshold)
        reciprocal = reciprocity_defect <= reciprocity_threshold
        charge_closed = complete & (column_charge_defect <= column_charge_threshold)
        certified = finite & converged & partition_certified & reciprocal & charge_closed
        evidence = DetectorWeightingEvidence(
            complete,
            jnp.sum(self.detector.unassigned_boundary_mask, dtype=jnp.int32),
            partition_defect,
            partition_threshold,
            partition_certified,
            reciprocity_defect,
            reciprocity_threshold,
            reciprocal,
            column_charge_defect,
            column_charge_threshold,
            charge_closed,
            finite,
            converged,
            certified,
        )
        return DetectorWeightingFieldResult(
            potentials,
            electric,
            physical,
            capacitance,
            solves,
            evidence,
            self.plan_id,
        )


__all__ = [
    "DetectorBiasElectrostaticPlan",
    "DetectorBiasElectrostaticResult",
    "DetectorElectrode",
    "DetectorResourcePolicy",
    "DetectorWeightingEvidence",
    "DetectorWeightingFieldPlan",
    "DetectorWeightingFieldResult",
    "SemiconductorDetectorPlan",
    "VOLUME_CHARGE_DENSITY_UNIT",
]
