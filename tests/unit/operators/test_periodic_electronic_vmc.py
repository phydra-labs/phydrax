import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic import AtomicStructure, AtomisticScaleContract
from phydrax.discretization import PeriodicCell
from phydrax.nn.quantum._periodic_features import PeriodicCellFeatures
from phydrax.nn.quantum._periodic_ferminet import PeriodicFermiNet
from phydrax.operators.quantum._electronic import ElectronicKineticPolicy
from phydrax.operators.quantum._electronic_advanced import ElectronicVMCResourcePlan
from phydrax.operators.quantum._periodic_electronic import (
    PeriodicElectronicCoulombHamiltonian,
    PeriodicElectronicEwaldPolicy,
)
from phydrax.units import ANGSTROM, BOHR, conversion_factor, ELECTRONVOLT, HARTREE


def _structure(charges, positions, cell, scale, *, name):
    return AtomicStructure(
        jnp.asarray(charges, dtype=jnp.int32),
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.ones((len(charges),), dtype=jnp.float64),
        scale,
        cell=jnp.asarray(cell, dtype=jnp.float64),
        periodic_axes=jnp.ones((3,), dtype=bool),
        name=name,
    )


def _ewald(
    *, screening, background=False, maximum_real=10_000, maximum_reciprocal=10_000
):
    return PeriodicElectronicEwaldPolicy(
        real_image_radius=1,
        reciprocal_radius=1,
        screening=screening,
        uniform_background=background,
        maximum_real_pair_terms=maximum_real,
        maximum_reciprocal_structure_terms=maximum_reciprocal,
    )


def _one_electron_model(cell, twist):
    resource = ElectronicVMCResourcePlan(1, determinant_count=1, spatial_dimension=3)
    model = PeriodicFermiNet(
        cell,
        jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
        jnp.ones((1, 1, 1), dtype=jnp.float64),
        jnp.ones((1,), dtype=jnp.float64),
        twist=jnp.asarray(twist, dtype=jnp.float64),
        resource_plan=resource,
    )
    return model, resource


def test_cell_features_use_physical_metric_and_preserve_integer_translations():
    vectors = jnp.asarray([[1.0, 0.0], [0.9, 0.2]], dtype=jnp.float64)
    cell = PeriodicCell(vectors)
    feature_map = PeriodicCellFeatures(
        cell,
        jnp.asarray([[0, 0], [1, -1]], dtype=jnp.int32),
        twist=jnp.asarray([0.3, -0.2]),
    )
    fractional = jnp.asarray([[0.49, 0.49], [0.0, 0.0]], dtype=jnp.float64)
    coordinates = cell.cartesian(fractional)
    result = feature_map(coordinates)

    candidates = np.asarray(
        [
            np.asarray(coordinates[0] - coordinates[1])
            - np.asarray(shift) @ np.asarray(vectors)
            for shift in itertools.product(range(-2, 3), repeat=2)
        ]
    )
    expected = candidates[np.argmin(np.sum(candidates * candidates, axis=1))]
    assert jnp.allclose(result.pair_displacements[0, 1], expected)
    assert jnp.array_equal(result.pair_image_shifts[0, 1], jnp.asarray([0, 1]))

    translation = jnp.asarray([[1.0, -2.0], [0.0, 0.0]]) @ vectors
    translated = feature_map(coordinates + translation)
    assert jnp.allclose(translated.wrapped_coordinates, result.wrapped_coordinates)
    assert jnp.allclose(translated.reciprocal_features, result.reciprocal_features)
    assert jnp.allclose(translated.pair_distances, result.pair_distances)


def test_periodic_ferminet_is_antisymmetric_twist_covariant_and_batched():
    vectors = jnp.asarray(
        [[2.0, 0.1, 0.0], [0.3, 1.7, 0.1], [0.0, 0.2, 2.3]],
        dtype=jnp.float64,
    )
    cell = PeriodicCell(vectors)
    twist = jnp.asarray([0.2, -0.35, 0.1], dtype=jnp.float64)
    resource = ElectronicVMCResourcePlan(2, determinant_count=1, spatial_dimension=3)
    model = PeriodicFermiNet(
        cell,
        jnp.asarray([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32),
        jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
        jnp.ones((1,)),
        twist=twist,
        pair_jastrow_strength=0.15,
        resource_plan=resource,
    )
    coordinates = cell.cartesian(
        jnp.asarray([[0.11, 0.23, 0.31], [0.37, 0.19, 0.16]], dtype=jnp.float64)
    )
    baseline = model(coordinates)
    exchanged = model(coordinates[::-1])
    translated_coordinates = coordinates.at[0].add(cell.vectors[1])
    translated = model(translated_coordinates)
    batched = model(jnp.stack((coordinates, translated_coordinates)))

    assert bool(baseline.valid & exchanged.valid & translated.valid)
    assert jnp.allclose(exchanged.log_abs, baseline.log_abs)
    assert jnp.allclose(exchanged.phase, -baseline.phase)
    assert jnp.allclose(translated.log_abs, baseline.log_abs)
    assert jnp.allclose(translated.phase / baseline.phase, jnp.exp(1.0j * twist[1]))
    assert batched.log_abs.shape == (2,)
    assert batched.phase.shape == (2,)
    assert batched.valid.shape == (2,)


def test_periodic_local_energy_is_periodic_and_exactly_decomposed():
    vectors = 4.0 * jnp.eye(3, dtype=jnp.float64)
    cell = PeriodicCell(vectors)
    scale = AtomisticScaleContract(BOHR, HARTREE)
    nuclei = _structure([1], [[0.0, 0.0, 0.0]], vectors, scale, name="periodic-H")
    twist = jnp.asarray([0.24, -0.16, 0.08], dtype=jnp.float64)
    model, resource = _one_electron_model(cell, twist)
    operator = PeriodicElectronicCoulombHamiltonian(
        nuclei,
        1,
        cell,
        twist=twist,
        ewald=_ewald(screening=0.7),
        kinetic=ElectronicKineticPolicy(
            trace_method="chunked-exact", coordinate_chunk_size=2
        ),
        resource_plan=resource,
    )
    coordinate = jnp.asarray([[1.1, 0.6, 0.9]], dtype=jnp.float64)
    walkers = jnp.stack((coordinate, coordinate + cell.vectors[2]))
    local = operator.local_energy(model, walkers)
    estimate = operator.estimate(model, walkers)

    inverse = cell.inverse_vectors
    physical_wavevector = inverse @ twist
    expected_kinetic = 0.5 * jnp.sum(physical_wavevector * physical_wavevector)
    decomposed = (
        local.potential.electron_electron
        + local.potential.electron_nucleus
        + local.potential.nucleus_nucleus
    )
    assert jnp.all(local.valid)
    assert jnp.allclose(local.value[0], local.value[1])
    assert jnp.allclose(local.kinetic, expected_kinetic, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(local.potential.total, decomposed)
    assert jnp.allclose(estimate.value, local.value)
    assert jnp.all(
        estimate.work_count == operator.resource_evidence.primitive_work_per_configuration
    )
    assert operator.resource_evidence.pair_feature_elements == 3
    assert operator.resource_evidence.determinant_cubic_work == 1
    assert operator.resource_evidence.kinetic_hessian_vector_products == 3
    assert operator.resource_evidence.ewald_real_pair_terms == 108
    assert operator.resource_evidence.ewald_reciprocal_structure_terms == 52


def test_periodic_local_energy_converts_length_and_energy_units_consistently():
    bohr_per_angstrom = float(conversion_factor(ANGSTROM, BOHR))
    hartree_per_ev = float(conversion_factor(ELECTRONVOLT, HARTREE))
    vectors_bohr = 5.0 * jnp.eye(3, dtype=jnp.float64)
    coordinate_bohr = jnp.asarray([[1.2, 0.7, 0.4]], dtype=jnp.float64)
    zero_twist = jnp.zeros((3,), dtype=jnp.float64)

    cell_bohr = PeriodicCell(vectors_bohr)
    structure_bohr = _structure(
        [1],
        [[0.0, 0.0, 0.0]],
        vectors_bohr,
        AtomisticScaleContract(BOHR, HARTREE),
        name="H-bohr",
    )
    model_bohr, resource_bohr = _one_electron_model(cell_bohr, zero_twist)
    operator_bohr = PeriodicElectronicCoulombHamiltonian(
        structure_bohr,
        1,
        cell_bohr,
        twist=zero_twist,
        ewald=_ewald(screening=0.65),
        resource_plan=resource_bohr,
    )

    vectors_angstrom = vectors_bohr / bohr_per_angstrom
    coordinate_angstrom = coordinate_bohr / bohr_per_angstrom
    cell_angstrom = PeriodicCell(vectors_angstrom)
    structure_angstrom = _structure(
        [1],
        [[0.0, 0.0, 0.0]],
        vectors_angstrom,
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        name="H-angstrom",
    )
    model_angstrom, resource_angstrom = _one_electron_model(cell_angstrom, zero_twist)
    operator_angstrom = PeriodicElectronicCoulombHamiltonian(
        structure_angstrom,
        1,
        cell_angstrom,
        twist=zero_twist,
        ewald=_ewald(screening=0.65 * bohr_per_angstrom),
        resource_plan=resource_angstrom,
    )

    energy_bohr = operator_bohr.local_energy(model_bohr, coordinate_bohr).value
    energy_ev = operator_angstrom.local_energy(model_angstrom, coordinate_angstrom).value
    assert jnp.allclose(energy_bohr, energy_ev * hartree_per_ev, rtol=1e-10, atol=1e-10)


def test_neutrality_boundary_and_ewald_work_fail_closed():
    vectors = 3.5 * jnp.eye(3, dtype=jnp.float64)
    cell = PeriodicCell(vectors)
    nuclei = _structure(
        [2],
        [[0.0, 0.0, 0.0]],
        vectors,
        AtomisticScaleContract(BOHR, HARTREE),
        name="charged-He",
    )
    model, resource = _one_electron_model(cell, jnp.zeros((3,)))
    with pytest.raises(ValueError, match="neutrality"):
        PeriodicElectronicCoulombHamiltonian(
            nuclei,
            1,
            cell,
            twist=jnp.zeros((3,)),
            ewald=_ewald(screening=0.8),
            resource_plan=resource,
        )
    with pytest.raises(ValueError, match="real-space work"):
        PeriodicElectronicCoulombHamiltonian(
            nuclei,
            1,
            cell,
            twist=jnp.zeros((3,)),
            ewald=_ewald(screening=0.8, background=True, maximum_real=100),
            resource_plan=resource,
        )

    admitted = PeriodicElectronicCoulombHamiltonian(
        nuclei,
        1,
        cell,
        twist=jnp.zeros((3,)),
        ewald=_ewald(screening=0.8, background=True),
        resource_plan=resource,
    )
    mismatched_model, _ = _one_electron_model(cell, jnp.asarray([0.1, 0.0, 0.0]))
    with pytest.raises(ValueError, match="cell/twist"):
        admitted.local_energy(mismatched_model, jnp.asarray([[1.0, 0.4, 0.2]]))

    local = admitted.local_energy(model, jnp.asarray([[1.0, 0.4, 0.2]]))
    assert bool(local.valid)
    assert admitted.net_charge == 1
    assert not admitted.neutral
    assert admitted.ewald.uniform_background
