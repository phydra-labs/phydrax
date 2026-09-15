#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prospective detector calibration/lock/refusal campaign runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.semiconductor import quantum as semiconductor_quantum
from phydrax.applications.semiconductor._detector import (
    DetectorElectrode,
    DetectorResourcePolicy,
    DetectorWeightingFieldPlan,
    SemiconductorDetectorPlan,
)
from phydrax.applications.semiconductor._detector_response import (
    DetectorTrajectoryRoute,
    PrescribedShockleyRamoPlan,
)
from phydrax.applications.semiconductor._production_qualification import (
    semiconductor_candidate_profile,
    semiconductor_detector_campaign,
    semiconductor_quantum_transport_campaign,
)
from phydrax.applications.semiconductor._quantities import SQUARE_METER
from phydrax.discretization import (
    NonuniformCellAxisSpec,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.dynamics import StateLayout, TrajectoryData
from phydrax.units import METER


jax.config.update("jax_enable_x64", True)
_EPSILON = 11.7 * 8.8541878128e-12
_ELEMENTARY_CHARGE = 1.602176634e-19
_PLANCK = 6.62607015e-34
_QUANTUM_REFERENCE = "semiconductor quantum qualification synthetic datum"


def _policy(**changes):
    values = dict(
        maximum_nodes=512,
        maximum_edges=1536,
        maximum_electrodes=8,
        maximum_linear_iterations=3000,
        maximum_trajectory_cases=8,
        maximum_trajectory_samples=512,
        maximum_interpolation_routes=4096,
    )
    values.update(changes)
    return DetectorResourcePolicy(**values)


def _endpoints(bridge):
    x = np.asarray(bridge.cochain.coordinates[0])[:, 0]
    return (
        DetectorElectrode("first", np.isclose(x, x[0])),
        DetectorElectrode("second", np.isclose(x, x[-1])),
    )


def _parallel_plate() -> dict[str, object]:
    length, area = 2.0e-3, 4.0e-4
    grid = TensorGridPlan(
        (UniformCellAxisSpec(32, periodic=False),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [length]]))
    bridge = StructuredCochainBridge(grid)
    detector = SemiconductorDetectorPlan(
        bridge,
        _endpoints(bridge),
        _policy(),
        permittivity=_EPSILON,
        node_transverse_measure=area,
        edge_transverse_measure=area,
        transverse_unit=SQUARE_METER,
    )
    result = DetectorWeightingFieldPlan(detector).solve()
    expected = _EPSILON * area / length
    return {
        "capacitance_relative_error": float(
            jnp.abs(result.capacitance[0, 0] - expected) / expected
        ),
        "partition_defect": float(result.evidence.partition_of_unity_defect),
        "reciprocity_defect": float(result.evidence.capacitance_reciprocity_defect),
        "certified": bool(result.evidence.certified),
    }


def _coax() -> dict[str, object]:
    inner, outer, length = 2.0e-4, 1.1e-3, 3.0e-3
    radii = np.geomspace(inner, outer, 33)
    normalized = (radii - inner) / (outer - inner)
    grid = TensorGridPlan(
        (NonuniformCellAxisSpec(normalized, periodic=False),),
        axis_names=("radius",),
    ).prepare(jnp.asarray([[inner], [outer]]))
    bridge = StructuredCochainBridge(grid)
    node_radius = np.asarray(bridge.cochain.coordinates[0])[:, 0]
    log_mean = np.diff(radii) / np.diff(np.log(radii))
    detector = SemiconductorDetectorPlan(
        bridge,
        _endpoints(bridge),
        _policy(),
        permittivity=_EPSILON,
        node_transverse_measure=2.0 * np.pi * node_radius * length,
        edge_transverse_measure=2.0 * np.pi * log_mean * length,
        transverse_unit=SQUARE_METER,
    )
    result = DetectorWeightingFieldPlan(detector).solve()
    expected = 2.0 * np.pi * _EPSILON * length / np.log(outer / inner)
    potential = np.log(outer / node_radius) / np.log(outer / inner)
    return {
        "capacitance_relative_error": float(
            jnp.abs(result.capacitance[0, 0] - expected) / expected
        ),
        "potential_maximum_error": float(
            jnp.max(jnp.abs(result.potentials[0] - potential))
        ),
        "certified": bool(result.evidence.certified),
    }


def _segmented(*, complete: bool):
    grid = TensorGridPlan(
        (UniformCellAxisSpec(6), UniformCellAxisSpec(6)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0e-3, 1.0e-3]]))
    bridge = StructuredCochainBridge(grid)
    points = np.asarray(bridge.cochain.coordinates[0])
    x, y = points[:, 0], points[:, 1]
    bottom_left = np.isclose(y, 0.0) & (x <= 0.5e-3)
    bottom_right = np.isclose(y, 0.0) & (x > 0.5e-3)
    top = np.isclose(y, 1.0e-3)
    guard = (
        (np.isclose(x, 0.0) | np.isclose(x, 1.0e-3)) & ~bottom_left & ~bottom_right & ~top
    )
    electrodes = [
        DetectorElectrode("bottom-left", bottom_left),
        DetectorElectrode("bottom-right", bottom_right),
        DetectorElectrode("top", top),
    ]
    if complete:
        electrodes.append(DetectorElectrode("guard", guard))
    detector = SemiconductorDetectorPlan(
        bridge,
        tuple(electrodes),
        _policy(),
        permittivity=_EPSILON,
        node_transverse_measure=2.0e-4,
        edge_transverse_measure=2.0e-4,
        transverse_unit=METER,
    )
    plan = DetectorWeightingFieldPlan(detector)
    return detector, plan, plan.solve()


def _segmented_locked() -> dict[str, object]:
    detector, weighting_plan, weighting = _segmented(complete=True)
    layout = StateLayout((2,), component_names=("x", "y"))
    states = jnp.asarray(
        [[1.0e-4, 1.0e-4], [2.5e-4, 3.0e-4], [5.0e-4, 5.5e-4], [8.0e-4, 9.0e-4]]
    )
    trajectory = TrajectoryData(
        jnp.linspace(0.0, 3.0e-9, states.shape[0]),
        states,
        state_layout=layout,
        source_id="detector-segmented-locked-path",
    )
    before = np.asarray(trajectory.states).copy()
    response = PrescribedShockleyRamoPlan(
        weighting_plan,
        weighting,
        DetectorTrajectoryRoute(layout, (0, 1)),
    ).evaluate(trajectory, -1.602176634e-19)
    return {
        "electrodes": detector.electrode_count,
        "partition_defect": float(weighting.evidence.partition_of_unity_defect),
        "reciprocity_defect": float(weighting.evidence.capacitance_reciprocity_defect),
        "current_endpoint_closure": float(
            jnp.max(response.current_integral_closure_defect)
        ),
        "carrier_state_change": float(
            np.max(np.abs(np.asarray(trajectory.states) - before))
        ),
        "certified": bool(weighting.evidence.certified and response.successful),
    }


def _expected_refusal(operation, message: str) -> dict[str, object]:
    try:
        operation()
    except ValueError as error:
        return {"refused": message in str(error), "message": str(error)}
    return {"refused": False, "message": "operation unexpectedly succeeded"}


def _refusal_cases() -> dict[str, object]:
    detector, weighting_plan, weighting = _segmented(complete=False)
    incomplete = _expected_refusal(
        lambda: PrescribedShockleyRamoPlan(
            weighting_plan,
            weighting,
            DetectorTrajectoryRoute(StateLayout((2,)), (0, 1)),
        ),
        "complete-electrode weighting basis",
    )

    grid = TensorGridPlan((UniformCellAxisSpec(8),), axis_names=("x",)).prepare(
        jnp.asarray([[0.0], [1.0]])
    )
    bridge = StructuredCochainBridge(grid)
    overflow = _expected_refusal(
        lambda: SemiconductorDetectorPlan(
            bridge,
            _endpoints(bridge),
            _policy(maximum_nodes=2),
            permittivity=_EPSILON,
            node_transverse_measure=1.0,
            edge_transverse_measure=1.0,
            transverse_unit=SQUARE_METER,
        ),
        "node count",
    )

    complete_detector = SemiconductorDetectorPlan(
        bridge,
        _endpoints(bridge),
        _policy(),
        permittivity=_EPSILON,
        node_transverse_measure=1.0,
        edge_transverse_measure=1.0,
        transverse_unit=SQUARE_METER,
    )
    complete_plan = DetectorWeightingFieldPlan(complete_detector)
    layout = StateLayout((1,))
    response_plan = PrescribedShockleyRamoPlan(
        complete_plan,
        complete_plan.solve(),
        DetectorTrajectoryRoute(layout, (0,)),
    )
    outside = TrajectoryData(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.25], [1.25]]),
        state_layout=layout,
        source_id="detector-locked-outside-route",
    )
    route = response_plan.evaluate(outside, 1.0)
    return {
        "incomplete_electrodes": incomplete,
        "resource_overflow": overflow,
        "invalid_route": {
            "refused": not bool(route.successful),
            "route_valid": bool(route.route_valid),
        },
        "unassigned_boundary_count": int(weighting.evidence.unassigned_boundary_count),
        "incomplete_partition_certified": bool(
            weighting.evidence.partition_of_unity_certified
        ),
        "detector_plan_id": detector.plan_id,
    }


def _quantum_resources():
    return semiconductor_quantum.QuantumResources(
        max_nodes=512,
        max_evaluations=20_000,
        max_intervals=256,
        workspace_bytes=512 * 1024 * 1024,
    )


def _quantum_lead(*, chemical_potential=2.0, onsite=2.0, coupling=-1.0):
    return semiconductor_quantum.SemiInfiniteLead(
        onsite * _ELEMENTARY_CHARGE,
        -_ELEMENTARY_CHARGE,
        coupling * _ELEMENTARY_CHARGE,
        chemical_potential * _ELEMENTARY_CHARGE,
        300.0,
        energy_reference=_QUANTUM_REFERENCE,
    )


def _quantum_device(*, left_mu=2.0, right_mu=2.0):
    hamiltonian = semiconductor_quantum.ChainHamiltonian(
        jnp.asarray([2.0 * _ELEMENTARY_CHARGE]),
        jnp.zeros((0,)),
        jnp.asarray([1.0e-27]),
        energy_reference=_QUANTUM_REFERENCE,
        resources=_quantum_resources(),
    )
    return semiconductor_quantum.CoherentDevice(
        hamiltonian,
        _quantum_lead(chemical_potential=left_mu),
        _quantum_lead(chemical_potential=right_mu),
        transverse=semiconductor_quantum.TransverseModes((0.0,), (1.0,)),
    )


def _quantum_transport_cases() -> tuple[dict[str, object], dict[str, bool]]:
    bias = 1.0e-3
    landauer = semiconductor_quantum.integrate_coherent(
        _quantum_device(left_mu=2.0 + bias / 2, right_mu=2.0 - bias / 2),
        tolerance=2.0e-7,
        spectral_tolerance=1.0e-4,
    )
    expected_current = _ELEMENTARY_CHARGE**2 / _PLANCK * bias
    landauer_case = {
        "successful": bool(landauer.successful),
        "analytic_current_relative_error": float(
            jnp.abs(jnp.abs(landauer.terminal_currents[0]) - expected_current)
            / expected_current
        ),
        "terminal_current_conservation": float(
            jnp.abs(jnp.sum(landauer.terminal_currents))
        ),
        "spectral_sum_error": float(landauer.evidence.spectral_sum_error),
        "refinement_error": float(landauer.evidence.refinement_error),
    }
    resonant_lead = _quantum_lead(chemical_potential=0.1, onsite=0.0, coupling=-0.2)
    resonant_device = semiconductor_quantum.CoherentDevice(
        semiconductor_quantum.ChainHamiltonian(
            [0.0],
            [],
            [1.0e-27],
            energy_reference=_QUANTUM_REFERENCE,
            resources=_quantum_resources(),
        ),
        resonant_lead,
        resonant_lead,
    )
    resonant_energy = 0.12 * _ELEMENTARY_CHARGE
    resonant_point = resonant_device.spectral(resonant_energy)
    resonant_sigma = (
        0.04
        * (
            resonant_energy
            - 1j * jnp.sqrt(4.0 * _ELEMENTARY_CHARGE**2 - resonant_energy**2)
        )
        / 2.0
    )
    resonant_gamma = -2.0 * jnp.imag(resonant_sigma)
    expected_transmission = (
        resonant_gamma**2 / jnp.abs(resonant_energy - 2.0 * resonant_sigma) ** 2
    )
    resonant_case = {
        "successful": bool(resonant_point.successful),
        "transmission_error": float(
            jnp.abs(resonant_point.transmission - expected_transmission)
        ),
    }

    equilibrium = _quantum_device()
    capacitance = semiconductor_quantum.QuantumCapacitance(
        jnp.asarray([[1.0e-18, 1.0e-18, 2.0e-18]])
    )
    ac = semiconductor_quantum.finite_frequency_quantum_response(
        equilibrium,
        capacitance,
        1.0e13,
        adiabatic_rate=7.0e13,
        lead_sites=96,
        tolerance=0.18,
    )
    ac_case = {
        "successful": bool(ac.successful),
        "gauge_error": float(ac.evidence.gauge_error),
        "kcl_error": float(ac.evidence.kcl_error),
        "ward_error": float(ac.evidence.ward_error),
        "lead_refinement_error": float(ac.evidence.lead_refinement_error),
        "adiabatic_refinement_error": float(ac.evidence.adiabatic_refinement_error),
        "recurrence_tail_bound": float(ac.evidence.recurrence_tail_bound),
    }

    times = jnp.asarray([0.0, 1.0e-17, 2.0e-17])
    pulse = semiconductor_quantum.QuantumPulse(
        times, jnp.zeros((2, 1)), jnp.zeros((2, 2))
    )
    transient = semiconductor_quantum.solve_quantum_transient(
        equilibrium,
        semiconductor_quantum.QuantumInitialState(preparation="equilibrium"),
        pulse,
        lead_sites=8,
        tolerance=0.1,
    )
    transient_case = {
        "successful": bool(transient.successful),
        "recurrence_valid": bool(transient.evidence.recurrence_valid),
        "kernel_error": float(transient.evidence.kernel_error),
        "number_error": float(transient.evidence.number_error),
        "unitary_error": float(transient.evidence.unitary_error),
        "energy_error": float(transient.evidence.energy_error),
    }

    bath = semiconductor_quantum.OpticalPhononBath(
        0.5 * _ELEMENTARY_CHARGE,
        0.03 * _ELEMENTARY_CHARGE,
        300.0,
        bath_id="semiconductor-qualification-optical-phonon",
    )
    energy_grid = semiconductor_quantum.PhononEnergyGrid(
        0.0,
        4.0 * _ELEMENTARY_CHARGE,
        points=16,
        phonon_energy=bath.energy,
    )
    scba = semiconductor_quantum.solve_phonon_transport(
        equilibrium,
        bath,
        energy_grid,
        tolerance=2.0e-6,
        observable_tolerance=0.2,
        maximum_steps=100,
        damping=0.4,
    )
    scba_case = {
        "successful": bool(scba.successful),
        "fixed_point_error": float(scba.evidence.fixed_point_error),
        "spectral_identity_error": float(scba.evidence.spectral_identity_error),
        "collision_particle_error": float(scba.evidence.collision_particle_error),
        "terminal_particle_error": float(scba.evidence.terminal_particle_error),
        "terminal_energy_error": float(scba.evidence.terminal_energy_error),
        "equilibrium_kms_error": float(scba.evidence.equilibrium_kms_error),
    }

    bound_lead = _quantum_lead(chemical_potential=4.0, onsite=0.0, coupling=-0.4)
    bound_device = semiconductor_quantum.CoherentDevice(
        semiconductor_quantum.ChainHamiltonian(
            [3.0 * _ELEMENTARY_CHARGE],
            [],
            [1.0e-27],
            energy_reference=_QUANTUM_REFERENCE,
            resources=_quantum_resources(),
        ),
        bound_lead,
        bound_lead,
    )
    bound_refusal = _expected_refusal(
        lambda: semiconductor_quantum.bound_states(bound_device),
        "occupation undetermined",
    )
    resource_refusal = _expected_refusal(
        lambda: semiconductor_quantum.ChainHamiltonian(
            [0.0, 0.0],
            [-_ELEMENTARY_CHARGE],
            [1.0e-27, 1.0e-27],
            energy_reference=_QUANTUM_REFERENCE,
            resources=semiconductor_quantum.QuantumResources(max_nodes=1),
        ),
        "max_nodes",
    )
    cases = {
        "landauer_transparent": landauer_case,
        "landauer_resonant": resonant_case,
        "coherent_ac_locked": ac_case,
        "finite_lead_transient_locked": transient_case,
        "optical_phonon_scba_locked": scba_case,
        "bound_state_refusal": bound_refusal,
        "resource_refusal": resource_refusal,
    }
    criteria = {
        "landauer": landauer_case["successful"]
        and resonant_case["successful"]
        and landauer_case["analytic_current_relative_error"] <= 5.0e-4
        and landauer_case["terminal_current_conservation"] <= 1.0e-15
        and resonant_case["transmission_error"] <= 1.0e-11,
        "coherent_ac": ac_case["successful"]
        and max(
            ac_case["gauge_error"],
            ac_case["kcl_error"],
            ac_case["ward_error"],
        )
        <= 1.0e-9
        and max(
            ac_case["lead_refinement_error"],
            ac_case["adiabatic_refinement_error"],
            ac_case["recurrence_tail_bound"],
        )
        <= 0.18,
        "finite_lead_transient": transient_case["successful"]
        and transient_case["recurrence_valid"]
        and max(
            transient_case["kernel_error"],
            transient_case["number_error"],
            transient_case["unitary_error"],
            transient_case["energy_error"],
        )
        <= 1.0e-9,
        "optical_phonon_scba": scba_case["successful"]
        and scba_case["fixed_point_error"] <= 2.0e-6
        and max(
            scba_case["collision_particle_error"],
            scba_case["terminal_particle_error"],
            scba_case["terminal_energy_error"],
            scba_case["equilibrium_kms_error"],
        )
        <= 1.0e-10,
        "refusals": bound_refusal["refused"] and resource_refusal["refused"],
    }
    return cases, criteria


def qualify() -> dict[str, object]:
    parallel = _parallel_plate()
    coax = _coax()
    segmented = _segmented_locked()
    refusals = _refusal_cases()
    detector_criteria = {
        "parallel_plate": parallel["certified"]
        and parallel["capacitance_relative_error"] <= 1.0e-8,
        "coax": coax["certified"]
        and coax["capacitance_relative_error"] <= 1.0e-8
        and coax["potential_maximum_error"] <= 1.0e-8,
        "segmented": segmented["certified"]
        and segmented["current_endpoint_closure"] <= 1.0e-28
        and segmented["carrier_state_change"] == 0.0,
        "refusals": all(
            (
                refusals["incomplete_electrodes"]["refused"],
                refusals["resource_overflow"]["refused"],
                refusals["invalid_route"]["refused"],
                not refusals["incomplete_partition_certified"],
            )
        ),
    }
    quantum_cases, quantum_criteria = _quantum_transport_cases()
    profile_names = (
        "semiconductor.detector.fixed-linear-electrostatics.v1",
        "semiconductor.detector.prescribed-shockley-ramo.v1",
        "semiconductor.quantum.chain-landauer.v1",
        "semiconductor.quantum.chain-coherent-ac.v1",
        "semiconductor.quantum.chain-finite-lead-transient.v1",
        "semiconductor.quantum.chain-optical-phonon-scba.v1",
    )
    return {
        "campaigns": {
            "detector": semiconductor_detector_campaign().to_record(),
            "quantum_transport": (semiconductor_quantum_transport_campaign().to_record()),
        },
        "candidate_profiles": [
            semiconductor_candidate_profile(name).to_record() for name in profile_names
        ],
        "cases": {
            "detector": {
                "parallel_plate": parallel,
                "coax": coax,
                "segmented_locked": segmented,
                "refusals": refusals,
            },
            "quantum_transport": quantum_cases,
        },
        "criteria": {
            "detector": detector_criteria,
            "quantum_transport": quantum_criteria,
        },
        "accepted": all(detector_criteria.values()) and all(quantum_criteria.values()),
        "release_granted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = qualify()
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    if not payload["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
