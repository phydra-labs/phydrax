import hashlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import atomistic, discretization
from phydrax.applications.protein_folding import (
    bind_protein,
    PreparedProteinQualification,
    ProteinAtomKey,
    ProteinConstruct,
    ProteinSourceAtom,
    ProteinStructureHypothesis,
    ResidueKey,
    ResolvedProteinChemistry,
)
from phydrax.applications.protein_folding.thermodynamics import native_enthalpy_series
from phydrax.applications.protein_folding.workflows import run_protein_dynamics
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM, CUBIC_METER, JOULE, PASCAL, UnitDefinition


NANOMETER = UnitDefinition("nm", ANGSTROM.dimension, ANGSTROM.reference_system_id, "1e-9")


def _fixture():
    # Explicit complete zwitterionic alanine inventory. The harmonic-only
    # model is an analytical numerical fixture, not a calibrated force field.
    construct = ProteinConstruct(("A",), ("A",))
    names = (
        "CA",
        "N",
        "C",
        "O",
        "OXT",
        "CB",
        "H1",
        "H2",
        "H3",
        "HA",
        "HB1",
        "HB2",
        "HB3",
    )
    numbers = (6, 7, 6, 8, 8, 6, 1, 1, 1, 1, 1, 1, 1)
    coordinates = np.asarray(
        [
            [0, 0, 0],
            [-1.45, 0, 0],
            [0.5, 1.42, 0],
            [-0.15, 2.45, 0],
            [1.78, 1.5, 0],
            [0.5, -0.75, -1.2],
            [-1.8, 0.8, 0.4],
            [-1.8, -0.8, 0.4],
            [-1.8, 0, -0.9],
            [0.35, -0.45, 0.9],
            [1.55, -0.6, -1.2],
            [0.2, -1.79, -1.15],
            [0.2, -0.4, -2.18],
        ]
    )
    keys = tuple(ProteinAtomKey(ResidueKey("A", 0), name) for name in names)
    digest = hashlib.sha256(coordinates.tobytes()).hexdigest()
    rights = ReferenceArtifactManifest(
        "synthetic-alanine-geometry",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=coordinates.nbytes,
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"angstrom": 1.0},
        uncertainty={"declared_coordinate_rounding_angstrom": 0.005},
        lineage_ids=("hand-authored-analytic-fixture",),
    )
    source = ScientificArtifactEnvelope(
        artifact_kind="synthetic-coordinate-fixture",
        content_digest=digest,
        producer="unit-fixture",
        producer_version="native",
        build_id="hand-authored",
        license_id="CC0-1.0",
        resource_id="alanine",
        status="complete",
    )
    chemistry = ResolvedProteinChemistry(
        construct, keys, numbers, ("standard",), (7,), "NH3+", "COO-", source.artifact_id
    )
    rows = tuple(
        ProteinSourceAtom(str(i), key, "1", "A", "17", "", "", 1.0, number)
        for i, (key, number) in enumerate(zip(keys, numbers, strict=True))
    )
    hypothesis = ProteinStructureHypothesis(
        construct, rows, coordinates, ANGSTROM, source, (rights,)
    )
    ids = np.arange(len(names), dtype=np.int64) * 13 + 97
    routes = np.asarray(
        [
            [0, 1],
            [0, 2],
            [2, 3],
            [2, 4],
            [0, 5],
            [1, 6],
            [1, 7],
            [1, 8],
            [0, 9],
            [5, 10],
            [5, 11],
            [5, 12],
        ]
    )
    # Topology canonicalizes endpoint order without changing row ordering.
    lengths = np.sqrt(
        np.sum((coordinates[routes[:, 0]] - coordinates[routes[:, 1]]) ** 2, axis=-1)
    )
    topology = atomistic.MolecularTopologyPlan(
        bonds=ids[routes], bond_type_ids=np.arange(len(routes))
    )
    units = atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = atomistic.AtomisticSystemPlan(
        ids,
        numbers,
        [12 if z == 6 else 14 if z == 7 else 16 if z == 8 else 1 for z in numbers],
        units,
        topology=topology,
        molecule_ids=np.zeros(len(names), dtype=int),
    )
    potential = atomistic.AtomisticPotentialProgram(
        [atomistic.HarmonicBondPotential(np.ones(len(routes)), 0.95 * lengths)]
    )
    force_field = atomistic.AtomisticForceFieldPlan(
        system,
        potential,
        atomistic.AtomisticNonbondedPolicy(8.0, electrostatics="direct"),
        atomistic.AtomisticForceFieldProvenance(
            "synthetic", (digest,), "analytic-harmonic", "explicit-unit-fixture"
        ),
    ).prepare()
    mapping = dict(zip(keys, map(int, ids), strict=True))
    binding = bind_protein(
        hypothesis,
        chemistry,
        force_field,
        mapping,
        parameter_energy_unit=units.scale.energy_unit,
        parameter_rights=(rights,),
    )
    neighborhood = discretization.DenseParticleNeighborhoodPlan(
        len(names) * (len(names) - 1) // 2
    ).prepare(force_field.system.particles)
    qualifier = PreparedProteinQualification(
        binding, bond_bounds=np.tile([0.7, 2.0], (len(routes), 1)), clash_distance=0.5
    )
    return binding, neighborhood, qualifier, mapping


def test_ordered_construct_and_source_permutation_do_not_redefine_atoms():
    assert ProteinConstruct(("A", "B"), ("AG", "V")).residue_keys == (
        ResidueKey("A", 0),
        ResidueKey("A", 1),
        ResidueKey("B", 0),
    )
    assert (
        ProteinConstruct(("A", "B"), ("AG", "V")).fingerprint()
        != ProteinConstruct(("B", "A"), ("V", "AG")).fingerprint()
    )
    binding, neighborhood, _, mapping = _fixture()
    h = binding.hypothesis
    order = np.arange(len(h.source_atoms))[::-1]
    reordered = ProteinStructureHypothesis(
        h.construct,
        tuple(h.source_atoms[i] for i in order),
        np.asarray(h.positions)[order] / 10,
        NANOMETER,
        h.source,
        h.rights,
    )
    rebound = bind_protein(
        reordered,
        binding.chemistry,
        binding.force_field,
        mapping,
        parameter_energy_unit=binding.force_field.system.plan.units.scale.energy_unit,
        parameter_rights=h.rights,
    )
    neighbors = neighborhood.build(binding.realized_positions)
    a, b = binding.evaluate(neighbors), rebound.evaluate(neighbors)
    np.testing.assert_allclose(a.energy, b.energy, rtol=1e-12)
    np.testing.assert_allclose(a.forces, b.forces, atol=1e-12)
    assert rebound.artifact.artifact_id != h.source.artifact_id
    assert rebound.hypothesis.hypothesis_id != h.hypothesis_id


def test_missing_atoms_and_mixed_parameter_scales_refuse_handoff():
    binding, _, _, mapping = _fixture()
    h = binding.hypothesis
    incomplete = ProteinStructureHypothesis(
        h.construct,
        h.source_atoms[:-1],
        h.positions[:-1],
        h.length_unit,
        h.source,
        h.rights,
    )
    with pytest.raises(ValueError, match="coverage"):
        bind_protein(
            incomplete,
            binding.chemistry,
            binding.force_field,
            mapping,
            parameter_energy_unit=binding.force_field.system.plan.units.scale.energy_unit,
            parameter_rights=h.rights,
        )
    with pytest.raises(ValueError, match="energy scale"):
        bind_protein(
            h,
            binding.chemistry,
            binding.force_field,
            mapping,
            parameter_energy_unit=JOULE,
            parameter_rights=h.rights,
        )
    with pytest.raises(ValueError, match="Hydrogen inventory"):
        replace(binding.chemistry, hydrogen_counts=(6,))


def test_geometry_detects_reflection_and_native_force_is_conservative():
    binding, neighborhood, qualifier, _ = _fixture()
    x = binding.realized_positions
    assert bool(qualifier.evaluate(x).successful)
    reflected = qualifier.evaluate(x * jnp.asarray([-1.0, 1.0, 1.0]))
    assert not bool(jnp.all(reflected.chirality_valid))
    rotated = jnp.stack((x[:, 1], -x[:, 0], x[:, 2]), axis=-1) + 8.0
    assert bool(
        jax.jit(lambda positions: qualifier.evaluate(positions))(rotated).successful
    )
    neighbors = neighborhood.build(x)
    evaluated = jax.jit(lambda positions: binding.evaluate(neighbors, positions))(x)
    gradient = jax.grad(
        lambda positions: binding.force_field.potential.energy(positions, neighbors)[0]
    )(x)
    np.testing.assert_allclose(evaluated.forces, -gradient, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(evaluated.forces, axis=0), 0.0, atol=1e-12)
    assert float(jnp.max(jnp.abs(evaluated.forces))) > 0.01


def test_short_native_nve_preserves_raw_hypothesis_and_energy():
    binding, neighborhood, qualifier, _ = _fixture()
    original = np.asarray(binding.hypothesis.positions).copy()
    result = run_protein_dynamics(
        binding,
        neighborhood,
        atomistic.VelocityVerletPlan(0.01),
        qualifier,
        velocity=np.zeros_like(original),
        velocity_unit=binding.force_field.system.plan.units.velocity_unit,
        key=jax.random.key(5),
        step_count=4,
    )
    assert bool(result.rollout.successful)
    np.testing.assert_array_equal(binding.hypothesis.positions, original)
    assert not np.array_equal(result.rollout.final_state.kinematics.positions, original)
    assert result.artifact.artifact_id != binding.artifact.artifact_id
    energy = np.asarray(result.rollout.trajectory.energies)[:, 2]
    np.testing.assert_allclose(energy, energy[0], atol=1e-8)
    assert int(jnp.sum(result.trajectory_data().transitions().valid)) == 4
    enthalpy = native_enthalpy_series(
        result.rollout.trajectory,
        pressure=1e5,
        pressure_unit=PASCAL,
        volumes=27e-27,
        volume_unit=CUBIC_METER,
        source_id=result.artifact.artifact_id,
    )
    # Physical pV is 2.7e-21 J, explicitly converted to single-system eV.
    np.testing.assert_allclose(
        enthalpy.values - energy, 2.7e-21 / 1.602176634e-19, atol=1e-12
    )


def test_short_native_nvt_uses_explicit_thermal_protocol_and_replays():
    binding, neighborhood, qualifier, _ = _fixture()
    integrator = atomistic.BAOABLangevinPlan(0.01, 300.0, 0.1)

    def run():
        return run_protein_dynamics(
            binding,
            neighborhood,
            integrator,
            qualifier,
            velocity=jnp.zeros_like(binding.realized_positions),
            velocity_unit=binding.force_field.system.plan.units.velocity_unit,
            key=jax.random.key(72),
            step_count=3,
        )

    first, replay = run(), run()
    assert bool(first.rollout.successful)
    assert float(first.rollout.trajectory.energies[-1, 1]) > 0.0
    np.testing.assert_array_equal(
        first.rollout.final_state.kinematics.positions,
        replay.rollout.final_state.kinematics.positions,
    )
