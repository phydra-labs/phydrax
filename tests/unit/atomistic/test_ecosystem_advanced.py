import os
import socket
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._identity import IdentityWords
from phydrax.atomistic._ensemble_advanced import EulerMaruyamaGeneralizedLangevinPlan
from phydrax.atomistic._reciprocal_mesh import ReciprocalMeshPlan
from phydrax.atomistic._thermal import (
    initial_nose_hoover_chain_state,
    NoseHooverChainParameters,
    propagate_nose_hoover_chain,
)
from phydrax.units import KILOJOULE_PER_MOLE
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_force_field,
    prepare_program,
    role_capacity,
    search_capacity,
)


def _periodic_resources(particles: int) -> dict[str, Any]:
    """Explicit storage of every pair of a few atoms: all directed routes."""
    directed = particles * (particles - 1)
    return {
        "search": phx.atomistic.AtomisticCellListImageSearch(
            maximum_candidate_slots=4096
        ),
        "skin": 0.5,
        "search_capacities": (
            search_capacity(16, particles_per_cell=particles, images=27),
        ),
        "role_capacities": (role_capacity(directed, particles - 1),),
    }


def _finite_resources(particles: int, *, extent: float = 20.0) -> dict[str, Any]:
    """Finite systems: nonperiodic frame enclosing the fixture and pair budget."""
    return {
        **_periodic_resources(particles),
        "maximum_pairs": particles * (particles - 1) // 2,
        "frame": phx.discretization.PeriodicCell(
            extent * jnp.eye(3),
            origin=jnp.full((3,), -0.5 * extent),
            periodic_axes=(False,) * 3,
        ),
    }


def _runtime(
    *, units: Any = None, topology: Any = None, charges: Any = None, cell: Any = None
) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced() if units is None else units
    system_plan = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0],
        # ty: ignore[invalid-argument-type]
        charges=[0.0, 0.0, 0.0] if charges is None else charges,
        topology=topology,
        cell=cell,
    )
    system = system_plan.prepare()
    resources = _finite_resources(3) if cell is None else _periodic_resources(3)
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
        ),
        system,
        **resources,
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(0),
    )
    return (
        system_plan,
        system,
        resources,
        potential,
        dynamics,
        thermodynamic,
        state,
    )


def test_force_field_contracts() -> None:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[10, 20]],
        bond_type_ids=[0],
        pair_exceptions=[[10, 20]],
        lennard_jones_scales=[0.5],
        electrostatic_scales=[0.25],
    )
    system_plan = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0],
        # ty: ignore[invalid-argument-type]
        charges=[0.4, -0.2, -0.2],
        topology=topology,
        name="mapping-roundtrip",
    )
    potential_plan = phx.atomistic.AtomisticPotentialProgram(
        [
            # ty: ignore[invalid-argument-type]
            phx.atomistic.HarmonicBondPotential([2.0], [1.0]),
            # ty: ignore[invalid-argument-type]
            phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5),
            phx.atomistic.DirectCoulombPotential(),
        ],
        # ty: ignore[invalid-argument-type]
        coefficients=[1.2, 0.8, 1.0],
    )
    force_field = phx.atomistic.AtomisticForceFieldPlan(
        system_plan,
        potential_plan,
        phx.atomistic.AtomisticNonbondedPolicy(2.5, electrostatics="direct"),
        phx.atomistic.AtomisticForceFieldProvenance(
            "native", ("qualification",), "custom", "roundtrip"
        ),
    )
    bundle = phx.atomistic.interchange.AtomisticInterchangeBundle(
        force_field,
        phx.atomistic.interchange.AtomisticInterchangeReport(
            "native", units, tuple(term.name for term in potential_plan.terms)
        ),
    )
    serialized = phx.atomistic.interchange.force_field_to_mapping(bundle)
    assert serialized["unit_system"]["unit_system_id"] == units.unit_system_id
    restored = phx.atomistic.interchange.force_field_from_mapping(serialized)
    assert restored.report.units.unit_system_id == units.unit_system_id
    first = prepare_force_field(force_field, **_finite_resources(3))
    second = prepare_force_field(restored.force_field, **_finite_resources(3))
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]])
    first_evaluation = first.potential.evaluate(
        positions, bundle_state(first.potential, positions)
    )
    second_evaluation = second.potential.evaluate(
        positions, bundle_state(second.potential, positions)
    )
    np.testing.assert_allclose(
        second_evaluation.energy, first_evaluation.energy, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        second_evaluation.forces, first_evaluation.forces, rtol=1.0e-12, atol=1.0e-12
    )
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 6.0)
    topology = phx.atomistic.MolecularTopologyPlan(
        torsions=[[0, 1, 2, 3]],
        impropers=[[0, 1, 2, 3]],
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1, 2, 3],
        # ty: ignore[invalid-argument-type]
        [8, 1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [16.0, 1.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0, 0],
        # ty: ignore[invalid-argument-type]
        charges=[-0.3, 0.1, 0.1, 0.1],
        topology=topology,
        cell=cell,
    ).prepare()
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.2, 1.0, 1.0],
        ]
    )
    route = [[0, 1, 2, 3]]
    terms = (
        # ty: ignore[invalid-argument-type]
        phx.atomistic.HarmonicImproperPotential([1.0], [0.2]),
        # ty: ignore[invalid-argument-type]
        phx.atomistic.UreyBradleyPotential([1.0], [1.4], [[0, 1, 2]]),
        phx.atomistic.PeriodicTorsionSeriesPotential(
            # ty: ignore[invalid-argument-type]
            [[0.2, 0.1]],
            # ty: ignore[invalid-argument-type]
            [[1, 2]],
            # ty: ignore[invalid-argument-type]
            [[0.0, 0.3]],
            # ty: ignore[invalid-argument-type]
            [[1.0, 1.0]],
            # ty: ignore[invalid-argument-type]
            route,
        ),
        # ty: ignore[invalid-argument-type]
        phx.atomistic.RyckaertBellemansPotential([[0.1, 0.2, 0.1, 0.0, 0.0, 0.0]], route),
        phx.atomistic.CMAPPotential(
            jnp.arange(64, dtype="float64").reshape((8, 8)) * 1.0e-3,
            # ty: ignore[invalid-argument-type]
            [[0, 1, 2, 3, 0, 1, 2, 3]],
        ),
        phx.atomistic.PairOverrideLennardJonesPotential(
            jnp.ones((4, 4)) * 0.1,
            jnp.ones((4, 4)),
            jnp.ones((4, 4)),
            2.5,
        ),
        # ty: ignore[invalid-argument-type]
        phx.atomistic.TabulatedPairPotential([0.5, 1.5, 2.5], [[[0.2, 0.1, 0.0]]], 2.5),
        phx.atomistic.ReactionFieldPotential(78.5, 2.5),
        phx.atomistic.LennardJonesDispersionCorrection(0.1),
    )
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(terms),
        system,
        **_periodic_resources(4),
    )
    evaluation = potential.evaluate(positions, bundle_state(potential, positions))
    assert bool(evaluation.successful)
    assert bool(jnp.isfinite(evaluation.energy))
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))

    settle_system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1, 2],
        # ty: ignore[invalid-argument-type]
        [8, 1, 1],
        # ty: ignore[invalid-argument-type]
        [16.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
    ).prepare()
    # ty: ignore[invalid-argument-type]
    settle = phx.atomistic.SETTLEPlan([[0, 1, 2]], 1.0, 1.6).prepare(settle_system)
    projection = settle.project(
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [-0.2, 0.9, 0.1]],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [-0.1, 0.0, 0.2]],
    )
    assert bool(projection.successful)
    assert float(projection.position_residual) <= 1.0e-10
    assert float(projection.velocity_residual) <= 1.0e-10


def test_ecosystem_advanced_scenario_1() -> None:
    topology = phx.atomistic.MolecularTopologyPlan(
        pair_exceptions=[[0, 1]],
        lennard_jones_scales=[0.0],
        electrostatic_scales=[0.0],
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        # ty: ignore[invalid-argument-type]
        charges=[1.0, -1.0],
        topology=topology,
    ).prepare()
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [
                # ty: ignore[invalid-argument-type]
                phx.atomistic.MorsePotential([[0.2]], [[2.0]], [[1.0]], 2.5),
                phx.atomistic.ReactionFieldPotential(78.5, 2.5),
            ]
        ),
        system,
        **_finite_resources(2),
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    result = potential.evaluate(positions, bundle_state(potential, positions))
    assert bool(result.successful)
    np.testing.assert_allclose(result.energy, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.forces, 0.0, atol=1.0e-12)
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    mapping = {
        "unit_system": units.to_dict(),
        "particle_ids": [10, 30, 70, 90],
        "atomic_numbers": [6, 6, 6, 6],
        "masses": [12.0, 12.0, 12.0, 12.0],
        "atom_type_ids": [0, 0, 0, 0],
        "charges": [0.0, 0.0, 0.0, 0.0],
        "topology": {
            "torsions": [[10, 30, 70, 90], [10, 70, 30, 90]],
            "torsion_type_ids": [0, 1],
        },
        "parameters": {
            "torsion_amplitude": [[0.8, 0.3], [0.2, 0.0]],
            "torsion_periodicity": [[1, 3], [2, 1]],
            "torsion_phase": [[0.37, -0.28], [-0.4, 0.0]],
            "torsion_mask": [[1.0, 1.0], [1.0, 0.0]],
        },
        "nonbonded": {"cutoff": 10.0},
    }
    reordered = {
        **mapping,
        "topology": {**mapping["topology"], "torsion_type_ids": [1, 0]},
        "parameters": {
            name: values[::-1] for name, values in mapping["parameters"].items()
        },
    }
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.7, 1.2, 0.9]]
    )
    evaluations = []
    for record in (mapping, reordered):
        prepared = prepare_force_field(
            phx.atomistic.interchange.force_field_from_mapping(record).force_field,
            **_finite_resources(4, extent=40.0),
        )
        result = prepared.potential.evaluate(
            positions, bundle_state(prepared.potential, positions)
        )
        assert bool(result.successful)
        evaluations.append(result)
    np.testing.assert_allclose(evaluations[0].energy, evaluations[1].energy, atol=1e-12)
    np.testing.assert_allclose(evaluations[0].forces, evaluations[1].forces, atol=1e-12)
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 8.0)
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1, 2, 3],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 2.0, 1.0, 2.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        cell=cell,
    ).prepare()
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    kinds = (
        (phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1], {}, None),
        (phx.atomistic.sampling.CollectiveVariableKind.ANGLE, [0, 1, 2], {}, None),
        (
            phx.atomistic.sampling.CollectiveVariableKind.TORSION,
            [0, 1, 2, 3],
            {},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.CENTER_OF_MASS_DISTANCE,
            [0, 1, 2, 3],
            {"parameters": [2]},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.RADIUS_OF_GYRATION,
            [0, 1, 2, 3],
            {},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.COORDINATION,
            [[0, 1], [2, 3]],
            {"parameters": [1.5, 6.0]},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.CONTACT_SIMILARITY,
            [[0, 1], [2, 3]],
            {"parameters": [0.5], "reference": [1.0, 1.0]},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.ALIGNED_RMSD,
            [0, 1, 2, 3],
            {"reference": np.asarray(positions)},
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.CELL_VOLUME,
            np.asarray([], dtype=np.int32),
            {},
            {"cell_vectors": cell.vectors},
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.DENSITY,
            np.asarray([], dtype=np.int32),
            {},
            {"cell_vectors": cell.vectors},
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.PATH_PROGRESS,
            [0, 1, 2, 3],
            {
                "parameters": [1.0],
                "reference": np.stack(
                    (np.asarray(positions), np.asarray(positions) + 1.0)
                ),
            },
            None,
        ),
        (
            phx.atomistic.sampling.CollectiveVariableKind.PATH_DISTANCE,
            [0, 1, 2, 3],
            {
                "parameters": [1.0],
                "reference": np.stack(
                    (np.asarray(positions), np.asarray(positions) + 1.0)
                ),
            },
            None,
        ),
    )
    for kind, indices, arguments, evaluation_arguments in kinds:
        prepared = phx.atomistic.sampling.CollectiveVariablePlan(
            kind,
            indices,
            # ty: ignore[invalid-argument-type]
            **arguments,
        ).prepare(system)
        # In-cell coordinates are the declared material (unwrapped) coordinates.
        geometry = phx.atomistic.sampling.CollectiveVariableGeometry(
            positions,
            unwrapped_positions=positions,
            cell=cell,
            cell_vectors=None
            if evaluation_arguments is None
            else evaluation_arguments["cell_vectors"],
        )
        evaluation = prepared.evaluate(geometry)
        assert bool(evaluation.successful), kind
        assert bool(jnp.isfinite(evaluation.value)), kind


def test_openmm_fourier_components_preserve_energy_and_force_through_roundtrip() -> None:
    openmm = pytest.importorskip("openmm")
    source = openmm.System()
    for _ in range(4):
        source.addParticle(12.0 * openmm.unit.dalton)
    torsions = openmm.PeriodicTorsionForce()
    for route, periodicity, phase, amplitude in (
        ((0, 1, 2, 3), 1, 0.37, 0.8),
        ((0, 1, 2, 3), 3, -0.28, 0.3),
        ((0, 2, 1, 3), 2, -0.4, 0.2),
    ):
        torsions.addTorsion(
            *route,
            periodicity,
            phase * openmm.unit.radian,
            amplitude * openmm.unit.kilojoule_per_mole,
        )
    source.addForce(torsions)
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    bundle = phx.atomistic.interchange.from_openmm_system(
        source, units, atomic_numbers=[6, 6, 6, 6], cutoff=10.0
    )
    bundle = phx.atomistic.interchange.force_field_from_mapping(
        phx.atomistic.interchange.force_field_to_mapping(bundle)
    )
    prepared = prepare_force_field(
        bundle.force_field, **_finite_resources(4, extent=40.0)
    )
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.7, 1.2, 0.9]]
    )
    neighbors = bundle_state(prepared.potential, positions)
    native = jax.jit(lambda value: prepared.potential.evaluate(value, neighbors))(
        positions
    )
    assert bool(native.successful)
    energy_factor = phx.atomistic.molar_energy_to_single_system_factor(
        KILOJOULE_PER_MOLE,
        units.scale.energy_unit,
        constant_set_id=units.constant_set_id,
    )
    exported, _, _ = phx.atomistic.interchange.to_openmm_system(bundle)
    for system in (source, exported):
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(np.asarray(positions) * openmm.unit.angstrom)
        state = context.getState(getEnergy=True, getForces=True)
        reference_energy = (
            state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            * energy_factor
        )
        reference_forces = (
            state.getForces(asNumpy=True).value_in_unit(
                openmm.unit.kilojoule_per_mole / openmm.unit.angstrom
            )
            * energy_factor
        )
        np.testing.assert_allclose(
            native.energy, reference_energy, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            native.forces, reference_forces, rtol=1e-10, atol=1e-12
        )
        del context, integrator


def _openmm_pair_source(openmm: Any, method: Any) -> Any:
    source = openmm.System()
    source.addParticle(12.0 * openmm.unit.dalton)
    source.addParticle(16.0 * openmm.unit.dalton)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(method)
    if method != openmm.NonbondedForce.NoCutoff:
        nonbonded.setCutoffDistance(10.0 * openmm.unit.angstrom)
    nonbonded.addParticle(
        0.1 * openmm.unit.elementary_charge,
        1.1 * openmm.unit.angstrom,
        0.3 * openmm.unit.kilojoule_per_mole,
    )
    nonbonded.addParticle(
        -0.1 * openmm.unit.elementary_charge,
        1.3 * openmm.unit.angstrom,
        0.2 * openmm.unit.kilojoule_per_mole,
    )
    bond = openmm.HarmonicBondForce()
    bond.addBond(
        0,
        1,
        1.0 * openmm.unit.angstrom,
        100.0 * openmm.unit.kilojoule_per_mole / openmm.unit.angstrom**2,
    )
    nonbonded.addException(
        0,
        1,
        -0.01 * openmm.unit.elementary_charge**2,
        1.2 * openmm.unit.angstrom,
        np.sqrt(0.3 * 0.2) * openmm.unit.kilojoule_per_mole,
    )
    source.addForce(nonbonded)
    source.addForce(bond)
    return source


def test_openmm_import_export_energy_force_parity() -> None:
    openmm = pytest.importorskip("openmm")
    source = _openmm_pair_source(openmm, openmm.NonbondedForce.NoCutoff)
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    bundle = phx.atomistic.interchange.from_openmm_system(
        source, units, atomic_numbers=[6, 8], cutoff=10.0
    )
    assert bundle.report.source_energy_unit == KILOJOULE_PER_MOLE
    assert bundle.report.avogadro_constant_set_id == units.constant_set_id
    energy_factor = phx.atomistic.molar_energy_to_single_system_factor(
        KILOJOULE_PER_MOLE,
        units.scale.energy_unit,
        constant_set_id=units.constant_set_id,
    )
    prepared = prepare_force_field(
        bundle.force_field, **_finite_resources(2, extent=40.0)
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    native = prepared.potential.evaluate(
        positions, bundle_state(prepared.potential, positions)
    )

    def openmm_evaluation(system: Any) -> Any:
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(np.asarray(positions) * openmm.unit.angstrom)
        state = context.getState(getEnergy=True, getForces=True)
        energy = (
            state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            * energy_factor
        )
        forces = (
            np.asarray(
                state.getForces(asNumpy=True).value_in_unit(
                    openmm.unit.kilojoule_per_mole / openmm.unit.angstrom
                )
            )
            * energy_factor
        )
        del context, integrator
        return energy, forces

    reference_energy, reference_forces = openmm_evaluation(source)
    np.testing.assert_allclose(
        native.energy, reference_energy, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        native.forces, reference_forces, rtol=1.0e-10, atol=1.0e-10
    )
    # The imported LJ is bounded by the adapter cutoff while direct Coulomb is
    # uncut; one OpenMM NonbondedForce cannot represent both, so export refuses.
    with pytest.raises(
        phx.atomistic.interchange.UnsupportedAtomisticContentError,
        match="finite-cutoff Lennard-Jones",
    ):
        phx.atomistic.interchange.to_openmm_system(bundle)

    source = _openmm_pair_source(openmm, openmm.NonbondedForce.CutoffNonPeriodic)
    bundle = phx.atomistic.interchange.from_openmm_system(
        source, units, atomic_numbers=[6, 8], cutoff=10.0
    )
    reference_energy, reference_forces = openmm_evaluation(source)
    exported, exported_topology, report = phx.atomistic.interchange.to_openmm_system(
        bundle
    )
    assert not report.unsupported_terms
    assert report.source_energy_unit == KILOJOULE_PER_MOLE
    assert report.avogadro_constant_set_id == units.constant_set_id
    exported_energy, exported_forces = openmm_evaluation(exported)
    np.testing.assert_allclose(
        exported_energy, reference_energy, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        exported_forces, reference_forces, rtol=1.0e-10, atol=1.0e-10
    )
    parmed = pytest.importorskip("parmed")
    with pytest.warns(parmed.exceptions.OpenMMWarning, match="incomplete exceptions"):
        structure = parmed.openmm.load_topology(exported_topology, exported)
    parmed_bundle = prepare_force_field(
        phx.atomistic.interchange.from_parmed_structure(
            structure, units, cutoff=10.0
        ).force_field,
        **_finite_resources(2, extent=40.0),
    )
    parmed_evaluation = parmed_bundle.potential.evaluate(
        positions, bundle_state(parmed_bundle.potential, positions)
    )
    np.testing.assert_allclose(
        parmed_evaluation.energy, reference_energy, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        parmed_evaluation.forces, reference_forces, rtol=1.0e-10, atol=1.0e-10
    )


def test_h5md_extended_xyz_and_rerun_reporting(tmp_path: Path) -> None:
    _, _, _, potential, dynamics, _, state = _runtime()
    frame = phx.atomistic.AtomisticFrame(
        state.time,
        state.step_index,
        state.kinematics.positions,
        dynamics.system.plan.particle_ids,
        velocities=dynamics.velocity(state),
        momenta=state.kinematics.momenta,
        forces=state.force.forces,
        # ty: ignore[invalid-argument-type]
        energy=[state.energy.total_energy],
        auxiliary={"temperature": jnp.asarray(1.0)},
        system_id=dynamics.system.plan.system_id,
        topology_id=dynamics.system.topology.topology_id,
        units=dynamics.system.plan.units,
        source_id="advanced-frame",
    )
    h5md = phx.atomistic.interchange.H5MDTrajectoryPlan(tmp_path / "input.h5")
    with h5md.open(append=False) as writer:
        writer.write(frame)
    with h5md.open(append=True) as writer:
        writer.write(
            phx.atomistic.AtomisticFrame(
                frame.time + 0.1,
                frame.step + 1,
                frame.positions,
                frame.stable_ids,
                velocities=frame.velocities,
                momenta=frame.momenta,
                forces=frame.forces,
                energy=frame.energy,
                # ty: ignore[invalid-argument-type]
                auxiliary=frame.auxiliary,
                system_id=frame.system_id,
                topology_id=frame.topology_id,
                units=frame.units,
                source_id="advanced-frame-1",
            )
        )
    with h5md.open() as reader:
        recovered = tuple(reader)
    assert len(recovered) == 2
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(recovered[1].velocities, frame.velocities)
    np.testing.assert_allclose(recovered[1].auxiliary["temperature"], 1.0)
    assert recovered[0].source_id == frame.source_id
    assert recovered[0].units.unit_system_id == frame.units.unit_system_id

    xyz = phx.atomistic.interchange.ExtendedXYZTrajectoryPlan(tmp_path / "output.xyz")
    with xyz.open(append=False) as writer:
        writer.write(frame)
    with xyz.open(append=True) as writer:
        writer.write(recovered[1])
    with xyz.open() as reader:
        xyz_frames = tuple(reader)
    assert len(xyz_frames) == 2
    xyz_frame = xyz_frames[0]
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(xyz_frame.forces, frame.forces)
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(xyz_frame.momenta, frame.momenta)
    assert xyz_frame.source_id == frame.source_id
    assert xyz_frame.units.unit_system_id == frame.units.unit_system_id

    rerun_output = phx.atomistic.interchange.H5MDTrajectoryPlan(tmp_path / "rerun.h5")
    fields = (
        phx.atomistic.AtomisticFrameFields.POSITIONS
        | phx.atomistic.AtomisticFrameFields.FORCES
        | phx.atomistic.AtomisticFrameFields.ENERGY
        | phx.atomistic.AtomisticFrameFields.AUXILIARY
    )
    reporter = phx.atomistic.AtomisticReporterPlan(rerun_output, fields=fields)
    result = phx.atomistic.AtomisticRerunPlan(
        h5md,
        potential,
        force_groups=(0,),
        reporter=reporter,
    ).run()
    assert bool(result.successful)
    assert result.reduction.frame_count == 2
    assert result.reduction.mean_energies.shape == (1,)
    assert result.reduction.mean_force_group_energies.shape == (1, 1)
    with rerun_output.open() as reader:
        reported = tuple(reader)
    assert len(reported) == 2
    assert reported[0].auxiliary["rerun_state_energies"].shape == (1,)
    assert reported[0].source_id.startswith(frame.source_id)
    assert reported[0].auxiliary["rerun_force_group_energies"].shape == (1, 1)
    h5py = pytest.importorskip("h5py")
    with h5py.File(h5md.path, "a") as handle:
        group = handle["particles/phydrax"]
        assert "unit_system_descriptor" in group.attrs
        del group.attrs["unit_system_descriptor"]
    with pytest.raises(ValueError, match="complete unit-system descriptor"):
        h5md.open()


def test_bias_term_and_abf_sample_contracts() -> None:
    _, system, _, _, _, _, state = _runtime()
    cv = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    variables = phx.atomistic.sampling.CollectiveVariableProgram((cv,))
    geometry = phx.atomistic.sampling.CollectiveVariableGeometry(
        state.kinematics.positions
    )
    bias = phx.atomistic.sampling.AtomisticBiasTerm(
        variables,
        phx.atomistic.sampling.HarmonicBiasForm(np.asarray([1.0]), 2.0),
        name="distance-umbrella",
        force_group=1,
    ).prepare(system)
    energy, _, successful = bias.bias(geometry, bias.initial_term_state(jnp.float64))
    assert bool(successful)
    np.testing.assert_allclose(energy, 0.04, atol=1e-12)

    abf = phx.atomistic.sampling.AtomisticABFPlan(
        variables, minimum=0.5, maximum=2.0, bins=16, temperature=1.0
    ).prepare(system)
    sample = abf.sample(geometry, jnp.zeros_like(state.force.forces))
    updated = abf.accumulate(abf.initialize(), sample)
    assert int(jnp.sum(updated.counts)) == 1
    np.testing.assert_allclose(sample.mean_force[0], -2.0 / 1.2, rtol=1e-12)
    with pytest.raises(ValueError):
        phx.atomistic.sampling.AtomisticBiasHistoryPlan(
            1, height=1.0, widths=0.1, capacity=0
        )


def test_mdanalysis_frame_contracts() -> None:
    atoms = SimpleNamespace(
        positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        velocities=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        forces=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        indices=np.asarray([0, 1]),
    )
    timestep = SimpleNamespace(
        dimensions=None,
        has_velocities=True,
        has_forces=True,
        time=2.0,
        frame=3,
    )
    universe = SimpleNamespace(
        atoms=atoms,
        trajectory=SimpleNamespace(ts=timestep),
    )
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()

    frame = phx.atomistic.interchange.atomistic_frame_from_mdanalysis(
        universe,
        system_id="mda-system",
        topology_id="mda-topology",
        units=units,
        source_id="mda-frame",
    )

    assert float(frame.time) == 2000.0
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(
        frame.velocities,
        [[0.001, 0.0, 0.0], [0.0, 0.002, 0.0]],
    )
    force_factor = phx.atomistic.molar_energy_to_single_system_factor(
        phx.units.KILOJOULE_PER_MOLE,
        phx.units.ELECTRONVOLT,
        constant_set_id=units.constant_set_id,
    )
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(frame.forces, atoms.forces * force_factor)
    with pytest.raises(ValueError, match="shared reference system"):
        phx.atomistic.interchange.atomistic_frame_from_mdanalysis(
            universe,
            system_id="mda-system",
            topology_id="mda-topology",
            units=phx.atomistic.AtomisticUnitSystem.reduced(),
            source_id="mda-frame",
        )
    mda = pytest.importorskip("MDAnalysis")
    universe = mda.Universe.empty(
        2,
        n_residues=2,
        n_segments=1,
        atom_resindex=[0, 1],
        residue_segindex=[0, 0],
        trajectory=True,
        velocities=True,
        forces=True,
    )
    universe.add_TopologyAttr("names", ["H1", "H2"])
    universe.add_TopologyAttr("elements", ["H", "H"])
    universe.add_TopologyAttr("resids", [1, 2])
    universe.add_TopologyAttr("resnames", ["MOL", "MOL"])
    universe.add_TopologyAttr("chainIDs", ["A", "A"])
    universe.add_TopologyAttr("segids", ["SYSTEM"])
    universe.atoms.positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    universe.atoms.velocities = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    universe.atoms.forces = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    universe.trajectory.ts.time = 2.0
    frame = phx.atomistic.interchange.atomistic_frame_from_mdanalysis(
        universe,
        system_id="mda-system",
        topology_id="mda-topology",
        units=phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        source_id="mda-frame",
    )
    metadata = phx.atomistic.interchange.atomistic_metadata_from_mdanalysis(universe)
    selection = phx.atomistic.interchange.mdanalysis_selection(
        universe, "name H1", stable_ids=[10, 20]
    )
    recovered = phx.atomistic.interchange.mdanalysis_universe_from_frames((frame,))
    assert metadata.atom_names == ("H1", "H2")
    np.testing.assert_array_equal(selection.mask, [True, False])
    np.testing.assert_allclose(recovered.atoms.positions, frame.positions)
    assert float(frame.time) == 2000.0
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(
        frame.velocities,
        [[0.001, 0.0, 0.0], [0.0, 0.002, 0.0]],
    )
    force_factor = phx.atomistic.molar_energy_to_single_system_factor(
        phx.units.KILOJOULE_PER_MOLE,
        phx.units.ELECTRONVOLT,
        constant_set_id=frame.units.constant_set_id,
    )
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(
        frame.forces,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]) * force_factor,
    )
    with pytest.raises(ValueError, match="shared reference system"):
        phx.atomistic.interchange.atomistic_frame_from_mdanalysis(
            universe,
            system_id="mda-system",
            topology_id="mda-topology",
            units=phx.atomistic.AtomisticUnitSystem.reduced(),
            source_id="mda-frame",
        )


def test_committee_acquisition_preserves_descriptor_diversity() -> None:
    _, system, resources, _, _, _, state = _runtime()
    frame = phx.atomistic.AtomisticFrame(
        0.0,
        0,
        state.kinematics.positions,
        system.plan.particle_ids,
        system_id=system.plan.system_id,
        topology_id=system.topology.topology_id,
        units=system.plan.units,
        source_id="acquisition-frame",
    )
    frames = tuple(
        phx.atomistic.AtomisticFrame(
            frame.time,
            index,
            frame.positions + index,
            frame.stable_ids,
            system_id=frame.system_id,
            topology_id=frame.topology_id,
            units=frame.units,
            source_id=f"frame-{index}",
        )
        for index in range(3)
    )
    evidence = tuple(
        phx.atomistic.AtomisticUncertaintyEvidence(
            jnp.asarray(score),
            jnp.zeros((3, 3)),
            jnp.zeros((3,)),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(False),
            jnp.asarray(True),
            "committee",
        )
        for score in (3.0, 2.0, 1.0)
    )
    records = phx.atomistic.AcquisitionPlan(
        2,
        phx.atomistic.CommitteeAcquisitionScorePolicy(1.0, 1.0, 1.0),
        # ty: ignore[invalid-argument-type]
    ).select(frames, evidence, descriptors=[[0.0], [0.1], [10.0]])
    assert tuple(record.source_index for record in records) == (0, 2)


def test_supported_committee_evidence_keeps_primary_segment() -> None:
    evidence = phx.atomistic.AtomisticUncertaintyEvidence(
        jnp.asarray(3.0),
        jnp.zeros((3, 3)),
        jnp.zeros((3,)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(False),
        jnp.asarray(True),
        "committee",
    )
    fallback = phx.atomistic.SegmentFallbackPolicy("reference-provider").decide(evidence)
    assert not fallback.use_fallback
    assert fallback.reason == "primary"


def test_nose_hoover_chain_heats_cold_momenta_across_substeps() -> None:
    _, system, _, _, _, _, _ = _runtime()

    masses = system.plan.masses
    mobile = system.active_mask
    momenta = jnp.ones((3, 3)) * 0.1
    # One Nose-Hoover chain coupled to all nine kinetic degrees of freedom at
    # kT = k_B * 1.0, applied over two consecutive substeps.
    chain = NoseHooverChainParameters(1.0, chain_length=3)
    thermal = system.plan.units.boltzmann_constant * 1.0

    def twice_kinetic(value: Any) -> Any:
        return system.plan.units.kinetic_to_energy * jnp.sum(
            jnp.where(mobile[:, None], value * value / masses[:, None], 0.0)
        )

    first = propagate_nose_hoover_chain(
        chain,
        initial_nose_hoover_chain_state(chain, momenta.dtype),
        twice_kinetic(momenta),
        9,
        thermal,
        0.001,
    )
    scaled = momenta * first.scale
    second = propagate_nose_hoover_chain(
        chain, first.state, twice_kinetic(scaled), 9, thermal, 0.001
    )
    assert first.state.velocities.shape == (3,)
    assert first.state.positions.shape == (3,)
    assert bool(first.successful)
    assert bool(second.successful)
    # Kinetic energy below N_f kT / 2 (2K = 0.09 < 9 kT) is heated by the chain.
    assert float(first.scale) > 1.0


def test_generalized_langevin_carries_auxiliary_memory() -> None:
    _, system, _, _, _, _, _ = _runtime()
    masses = system.plan.masses
    mobile = system.active_mask
    momenta = jnp.ones((3, 3)) * 0.1
    gle = EulerMaruyamaGeneralizedLangevinPlan(
        # ty: ignore[invalid-argument-type]
        [[1.0, 0.1], [0.1, 1.5]],
        # ty: ignore[invalid-argument-type]
        [[0.2, 0.0], [0.0, 0.2]],
        1.0,
    )
    gle_result = gle.apply(
        momenta,
        masses,
        mobile,
        jax.random.key(3),
        0,
        0.001,
        system.plan.units,
    )
    assert gle_result.auxiliary.shape == (3, 3, 1)


def test_ring_polymer_normal_modes_roundtrip() -> None:
    _, _, _, _, _, _, state = _runtime()
    normal_modes = phx.atomistic.RingPolymerNormalModePlan(4, 1.0)
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    np.testing.assert_allclose(
        normal_modes.inverse(normal_modes.forward(beads)), beads, atol=1.0e-6
    )


def test_ring_polymer_staging_coordinates_roundtrip() -> None:
    _, _, _, _, _, _, state = _runtime()
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    staging = phx.atomistic.StagingCoordinatePlan(4)
    np.testing.assert_allclose(
        staging.inverse(staging.forward(beads)), beads, atol=1.0e-12
    )


def test_ring_polymer_contraction_preserves_centroid() -> None:
    _, _, _, _, _, _, state = _runtime()
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    contracted = phx.atomistic.ring_polymer_contract(beads, 2)
    np.testing.assert_allclose(jnp.mean(contracted, axis=0), jnp.mean(beads, axis=0))


def test_ring_polymer_mode_thermostat_reports_success() -> None:
    _, system, _, _, _, _, state = _runtime()
    masses = system.plan.masses
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    normal_modes = phx.atomistic.RingPolymerNormalModePlan(4, 1.0)
    # ty: ignore[invalid-argument-type]
    trpmd = phx.atomistic.ThermostattedRPMDPlan(normal_modes, [0.0, 1.0, 1.0, 1.0])
    trpmd_result = trpmd.apply(
        jnp.zeros_like(beads),
        masses,
        1.0,
        0.001,
        jax.random.key(4),
        system.plan.units,
    )
    assert bool(trpmd_result.successful)


def test_piglet_applies_generalized_mode_controllers() -> None:
    _, system, _, _, _, _, state = _runtime()
    masses = system.plan.masses
    mobile = system.active_mask
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    normal_modes = phx.atomistic.RingPolymerNormalModePlan(4, 1.0)
    gle = EulerMaruyamaGeneralizedLangevinPlan(
        jnp.asarray([[1.0, 0.1], [0.1, 1.5]], dtype=jnp.float64),
        jnp.asarray([[0.2, 0.0], [0.0, 0.2]], dtype=jnp.float64),
        1.0,
    )
    piglet = phx.atomistic.PIGLETPlan((gle,) * 4, normal_modes)
    piglet_result = piglet.apply(
        jnp.zeros_like(beads),
        masses,
        mobile,
        jax.random.key(5),
        0,
        0.001,
        system.plan.units,
    )
    assert bool(piglet_result.successful)


def test_ring_polymer_estimators_admit_isotope_masses() -> None:
    _, system, _, _, _, _, state = _runtime()
    masses = system.plan.masses
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    polymer = phx.atomistic.RingPolymerState(
        beads,
        jnp.zeros_like(beads),
        jnp.asarray(0),
        jax.random.key(6),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        IdentityWords.of("barostat-test"),
    )
    estimators = phx.atomistic.quantum_estimators(
        polymer,
        masses,
        jnp.zeros_like(beads),
        1.0,
        system.plan.units.boltzmann_constant,
        1.0,
        isotope_masses=masses * 2.0,
    )
    assert bool(estimators.successful)


def test_open_path_momentum_distribution_has_one_value_per_bead() -> None:
    _, _, _, _, _, _, state = _runtime()
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    momentum_distribution = phx.atomistic.open_path_momentum_distribution(
        beads, jnp.zeros_like(beads)
    )
    assert momentum_distribution.shape == (4,)


def test_suzuki_chin_correction_retains_finite_directional_response() -> None:
    _, _, _, _, _, _, state = _runtime()
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    correction, directional = phx.atomistic.suzuki_chin_correction(
        lambda value: -value, beads, 0.1
    )
    assert bool(jnp.isfinite(correction) & jnp.all(jnp.isfinite(directional)))


def test_ring_polymer_pressure_step_retains_a_finite_cell() -> None:
    _, _, _, _, _, _, state = _runtime()
    beads = jnp.stack(tuple(state.kinematics.positions + 0.01 * i for i in range(4)))
    polymer = phx.atomistic.RingPolymerState(
        beads,
        jnp.zeros_like(beads),
        jnp.asarray(0),
        jax.random.key(6),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        IdentityWords.of("barostat-test"),
    )
    barostat = phx.atomistic.ConstantPressureRingPolymerPlan(1.0, 10.0)
    barostat_state = barostat.initialize(polymer, jnp.eye(3) * 6.0)
    barostat_state = barostat.step(barostat_state, 1.1, 0.001)
    assert bool(barostat_state.successful)
    assert bool(jnp.all(jnp.isfinite(barostat_state.cell_vectors)))


def test_cylindrical_manifold_projection_preserves_radius() -> None:
    manifold = phx.atomistic.ManifoldConstraintPlan(
        phx.atomistic.WallKind.CYLINDER,
        # ty: ignore[invalid-argument-type]
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
    )
    projection = manifold.project(
        # ty: ignore[invalid-argument-type]
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        # ty: ignore[invalid-argument-type]
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )
    assert bool(projection.successful)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(projection.positions[:, :2] ** 2, axis=-1)), 2.0
    )


def test_anisotropic_pressure_update_retains_a_finite_cell() -> None:
    pressure = phx.atomistic.AnisotropicPressurePlan(
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        # ty: ignore[invalid-argument-type]
        [0.01, 0.01, 0.01],
        semi_isotropic=True,
    )
    # ty: ignore[invalid-argument-type]
    updated_cell = pressure.update_cell(jnp.eye(3) * 6.0, [1.1, 0.9, 1.2], 0.01)
    assert bool(jnp.all(jnp.isfinite(updated_cell)))


def test_brownian_step_retains_finite_coordinates() -> None:
    _, system, _, _, _, _, state = _runtime()
    brownian = phx.atomistic.BrownianDynamicsPlan(0.1, 1.0).step(
        state.kinematics.positions,
        state.force.forces,
        0.001,
        jax.random.key(8),
        system.plan.units,
    )
    assert bool(jnp.all(jnp.isfinite(brownian)))


def test_active_force_preserves_unit_orientations() -> None:
    active = phx.atomistic.ActiveForcePlan(1.0, 0.1).evaluate(
        jnp.ones((3, 3)), jax.random.key(7), 0.01
    )
    assert bool(active.successful & jnp.all(jnp.isfinite(active.forces)))
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(active.orientations**2, axis=-1)), 1.0, atol=1.0e-12
    )


def test_dissipative_particle_pair_force_is_finite() -> None:
    dpd_force = phx.atomistic.DissipativeParticleDynamicsPlan(
        1.0, 0.5, 1.0, 2.5
    ).pair_force(
        jnp.asarray([[1.0, 0.0, 0.0]]),
        jnp.asarray([[0.1, 0.0, 0.0]]),
        jnp.asarray([0.2]),
        1.0,
        0.01,
    )
    assert bool(jnp.all(jnp.isfinite(dpd_force)))


def test_wall_term_evaluates_on_explicit_resources() -> None:
    _, system, resources, _, _, _, state = _runtime()
    wall = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [
                phx.atomistic.ScalarWallPotential(
                    phx.atomistic.WallKind.PLANE,
                    # ty: ignore[invalid-argument-type]
                    [1.0, 0.0, 0.0, 0.5],
                    2.0,
                )
            ]
        ),
        system,
        **resources,
    )
    wall_evaluation = wall.evaluate(
        state.kinematics.positions,
        bundle_state(wall, state.kinematics.positions),
    )
    assert bool(wall_evaluation.successful)


def test_eam_evaluates_on_explicit_graph_resources() -> None:
    _, system, resources, _, _, _, state = _runtime()
    graph_execution = phx.atomistic.AtomisticGraphExecutionPlan(
        system.capacity - 1,
        backend="particle",
    )

    eam = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.EAMPotential([2.0, 1.0, 1.0, 0.5, 2.0], 2.5)]
        ),
        system,
        graph_execution=graph_execution,
        **resources,
    )
    eam_result = eam.evaluate(
        state.kinematics.positions,
        bundle_state(eam, state.kinematics.positions),
    )
    assert bool(eam_result.successful)

    assert bool(jnp.isfinite(eam_result.energy))


@pytest.mark.parametrize(
    ("factory", "parameters", "cutoff"),
    (
        (
            phx.atomistic.StillingerWeberPotential,
            [1.0, 1.0, 7.049556277, 0.6022245584, 4.0, 0.0, 1.8, 21.0, 1.2, -1.0 / 3.0],
            1.8,
        ),
        (
            phx.atomistic.TersoffPotential,
            [1.0, 0.5, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -0.3, 2.0, 0.5, 3.0],
            2.5,
        ),
    ),
    ids=["stillinger-weber", "tersoff"],
)
def test_many_body_terms_evaluate_on_explicit_graph_resources(
    factory: Callable[[jax.Array, float], phx.atomistic.ManyBodyPotential],
    parameters: list[float],
    cutoff: float,
) -> None:
    _, system, resources, _, _, _, state = _runtime()
    graph_execution = phx.atomistic.AtomisticGraphExecutionPlan(
        system.capacity - 1, backend="particle"
    )
    many_body = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [factory(jnp.asarray(parameters, dtype=jnp.float64), cutoff)]
        ),
        system,
        graph_execution=graph_execution,
        **resources,
    )
    evaluation = many_body.evaluate(
        state.kinematics.positions,
        bundle_state(many_body, state.kinematics.positions),
    )
    assert bool(evaluation.successful)
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))


def test_ecosystem_advanced_scenario_2() -> None:
    particles = phx.discretization.ParticleSetPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        ambient_dimension=3,
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        # ty: ignore[invalid-argument-type]
        [0, 0],
        jnp.stack((jnp.eye(3), jnp.eye(3))),
    ).prepare(particles)
    orientation = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    kinematics = bodies.kinematics(
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        jnp.zeros((2, 3)),
        orientation,
        jnp.zeros((2, 3)),
    )
    coordinate_map = phx.atomistic.RigidAtomisticCoordinateMap(
        bodies,
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [10, 20],
    )
    sites = coordinate_map.realize(kinematics)
    np.testing.assert_allclose(sites.positions, [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    load = phx.discretization.RigidBodyLoad(
        jnp.zeros((2, 3)), jnp.asarray([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    )
    state = phx.atomistic.RotationalAtomisticState(
        kinematics, load.force, load.torque, jnp.asarray(True)
    )
    advanced = phx.atomistic.rotational_velocity_verlet(
        bodies,
        state,
        1.0e-3,
        lambda time, value, args: load,
    )
    assert bool(advanced.successful)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(advanced.kinematics.orientation**2, axis=-1)),
        1.0,
        atol=1.0e-12,
    )
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 6.0)
    _, system, resources, _, _, _, state = _runtime(cell=cell, charges=[0.4, -0.2, -0.2])
    multipoles = phx.atomistic.PermanentMultipoleSiteData(
        # ty: ignore[invalid-argument-type]
        [0.4, -0.2, -0.2],
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3, 3)),
        # ty: ignore[invalid-argument-type]
        [0.1, 0.1, 0.1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
    )
    plan = phx.atomistic.PolarizationPlan(
        coulomb_constant=system.plan.units.coulomb_constant,
        maximum_iterations=200,
        tolerance=1.0e-6,
    )
    response = phx.atomistic.evaluate_implicit_polarization_jvp(
        plan.prepare(multipoles),
        state.kinematics.positions,
        jnp.ones_like(state.kinematics.positions) * 0.01,
    )
    assert bool(response.successful)
    assert bool(jnp.all(jnp.isfinite(response.primal)))
    assert bool(jnp.all(jnp.isfinite(response.tangent)))
    multipole_pme = phx.atomistic.MultipoleElectrostaticsPlan(
        phx.atomistic.MultipoleElectrostaticsDefinition(
            "point", 2, coulomb_constant=system.plan.units.coulomb_constant
        ),
        phx.atomistic.ParticleMeshMultipoleRoute(0.4, 2.5, ReciprocalMeshPlan((8, 8, 8))),
    ).prepare(
        system.capacity, cell=system.cell, particles=system.particle_discretization()
    )
    energy = multipole_pme.energy(
        state.kinematics.positions,
        phx.atomistic.MultipoleSiteSet(multipoles.moments(), active=system.active_mask),
        cell_vectors=cell.vectors,
    )
    assert bool(energy.successful & jnp.isfinite(energy.energy))
    dispersion = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPMEPotential([[0.1]], 0.4, 2.5, (8, 8, 8))]
        ),
        system,
        **resources,
    )
    dispersion_result = dispersion.evaluate(
        state.kinematics.positions,
        bundle_state(dispersion, state.kinematics.positions),
    )
    assert bool(dispersion_result.successful)
    assert bool(jnp.all(jnp.isfinite(dispersion_result.forces)))
    np.testing.assert_allclose(
        jnp.sum(dispersion_result.forces, axis=0), 0.0, atol=1.0e-8
    )
    gb = phx.atomistic.ImplicitSolventPlan("gb").energy(
        state.kinematics.positions,
        # ty: ignore[invalid-argument-type]
        [0.4, -0.2, -0.2],
        # ty: ignore[invalid-argument-type]
        [1.2, 1.2, 1.2],
        1.0,
    )
    gk = phx.atomistic.ImplicitSolventPlan("gk").energy(
        state.kinematics.positions,
        # ty: ignore[invalid-argument-type]
        [0.4, -0.2, -0.2],
        # ty: ignore[invalid-argument-type]
        [1.2, 1.2, 1.2],
        1.0,
    )
    assert bool(jnp.isfinite(gb) & jnp.isfinite(gk))
    assert not bool(jnp.isclose(gb, gk))
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 6.0)
    _, system, resources, _, _, _, _ = _runtime(cell=cell)
    # ty: ignore[invalid-argument-type]
    reference = phx.atomistic.LennardJonesPMEPotential([[0.1]], 0.4, 2.5, (8, 8, 8))
    term = phx.atomistic.GeneralForceFieldTerm(
        phx.atomistic.ForceFieldTermKind.LENNARD_JONES_PME,
        reference.arrays,
        cutoff=None,
    )
    # A cutoff-free pair relation has unbounded periodic support: refused.
    with pytest.raises(ValueError, match="requires a cutoff|unbounded periodic support"):
        prepare_program(
            phx.atomistic.AtomisticPotentialProgram([term]),
            system,
            maximum_pairs=3,
            **resources,
        )


def _ipi_roundtrip(plan: Any, system: Any, positions: Any) -> None:
    listener = plan.listen()

    def evaluator(prepared: Any, coordinate: Any, cell_vectors: Any) -> Any:
        del prepared, cell_vectors
        return phx.atomistic.ExternalAtomisticEvaluation(
            jnp.sum(coordinate**2),
            -2.0 * coordinate,
            None,
            jnp.asarray(True),
            provider_label="local-provider",
        )

    provider = phx.atomistic.CallableBornOppenheimerProvider(evaluator, "local-provider")

    def serve() -> Any:
        with listener.accept() as session:
            return phx.atomistic.interchange.serve_ipi_once(session, provider, system)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve)
        with plan.connect() as session:
            remote = phx.atomistic.interchange.TransportedExternalAtomisticProvider(
                session, "remote-provider"
            )
            result = remote.evaluate(system, positions, None)
        status = future.result(timeout=10.0)
    listener.close()
    assert status is phx.atomistic.interchange.IPITransportStatus.READY
    np.testing.assert_allclose(result.energy, jnp.sum(positions**2), rtol=1e-12)
    np.testing.assert_allclose(result.forces, -2.0 * positions, rtol=1e-12)
    assert result.stress is None


@pytest.mark.parametrize("mode", ["unix", "tcp"])
def test_ipi_unix_and_tcp_roundtrip(tmp_path: Path, mode: str) -> None:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    _, system, _, _, _, _, state = _runtime(units=units)
    if mode == "unix":
        plan = phx.atomistic.interchange.IPITransportPlan.unix(
            f"/tmp/phydrax-ipi-{os.getpid()}.sock", timeout=5.0, virial="optional"
        )
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        plan = phx.atomistic.interchange.IPITransportPlan.tcp(
            "127.0.0.1", port, timeout=5.0, virial="optional"
        )
    _ipi_roundtrip(plan, system, state.kinematics.positions)


def test_packmol_subprocess_boundary_and_provenance(tmp_path: Path) -> None:
    executable = tmp_path / "fake_packmol.py"
    executable.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys
lines = sys.stdin.read().splitlines()
structure = next(line.split()[1] for line in lines if line.startswith('structure '))
number = int(next(line.split()[1] for line in lines if line.strip().startswith('number ')))
source = Path(structure).read_text().splitlines()
atoms = source[2:]
output = []
for molecule in range(number):
    for atom in atoms:
        fields = atom.split()
        output.append(f'{fields[0]} {float(fields[1]) + 4.0 * molecule} {fields[2]} {fields[3]}')
Path('output.xyz').write_text(f'{len(output)}\\nfake packmol\\n' + '\\n'.join(output) + '\\n')
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    template = phx.atomistic.AtomisticFrame(
        0.0,
        0,
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [10, 20],
        system_id="packmol-system",
        topology_id="packmol-topology",
        units=phx.atomistic.AtomisticUnitSystem.reduced(),
        source_id="packmol-template",
    )
    component = phx.atomistic.interchange.PackmolComponentPlan(
        template,
        2,
        constraints=(
            phx.atomistic.interchange.PackmolRegionConstraint(
                "inside-box", [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
            ),
        ),
    )
    result = phx.atomistic.interchange.PackmolAssemblyPlan(
        (component,), executable=str(executable)
    ).run()
    assert result.successful
    assert result.positions.shape == (4, 3)
    assert result.units.unit_system_id == template.units.unit_system_id
    assert result.component_slices == ((0, 4),)
    assert result.input_digest
