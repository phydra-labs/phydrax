import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic import AtomicStructure, AtomisticScaleContract
from phydrax.discretization import PeriodicCell
from phydrax.linalg import (
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LowRankDeterminantStatus,
    LowRankSolvePolicy,
    MixedPrecisionPolicy,
)
from phydrax.nn.quantum import (
    periodic_ferminet_incremental_target,
    PeriodicCellFeatures,
    PeriodicFermiNet,
    PeriodicFermiNetCache,
)
from phydrax.operators.quantum._electronic import ElectronicKineticPolicy
from phydrax.operators.quantum._electronic_advanced import ElectronicVMCResourcePlan
from phydrax.operators.quantum._periodic_electronic import (
    PeriodicElectronicCoulombHamiltonian,
    PeriodicElectronicEwaldPolicy,
)
from phydrax.sampling import SingleCoordinateProposalPayload
from phydrax.units import ANGSTROM, BOHR, conversion_factor, ELECTRONVOLT, HARTREE


def _structure(charges, positions, cell, scale, *, name):
    return AtomicStructure(
        jnp.asarray(charges, dtype=jnp.int32),
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.ones((len(charges),), dtype=jnp.float64),
        scale,
        cell=jnp.asarray(cell, dtype=jnp.float64),
        periodic_axes=jnp.ones((3,), dtype="bool"),
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


def _determinant_update_policy():
    return LowRankSolvePolicy(
        LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status")),
        base_nonsingularity="asserted",
        failure=FailurePolicy("status"),
    )


def test_periodic_incremental_target_rejects_mixed_precision_lu():
    model, _ = _one_electron_model(
        PeriodicCell(jnp.eye(3, dtype=jnp.float64)),
        jnp.zeros((3,), dtype=jnp.float64),
    )
    status = FailurePolicy("status")
    update_policy = LowRankSolvePolicy(
        LinearSolvePolicy(
            DenseLU(),
            failure=status,
            precision=MixedPrecisionPolicy(
                factorization_dtype=jnp.float32,
                maximum_refinement_steps=1,
            ),
        ),
        base_nonsingularity="asserted",
        failure=status,
    )

    with pytest.raises(ValueError, match="full-precision"):
        periodic_ferminet_incremental_target(
            model,
            capacity=1,
            update_policy=update_policy,
            maximum_chains=1,
        )


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


def test_periodic_ferminet_mixture_ignores_singular_components():
    cell = PeriodicCell(2.0 * jnp.eye(2, dtype=jnp.float64))
    modes = jnp.asarray([[0, 0], [1, 0]], dtype=jnp.int32)
    coefficients = jnp.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    resource = ElectronicVMCResourcePlan(
        2,
        determinant_count=2,
        spatial_dimension=2,
    )
    model = PeriodicFermiNet(
        cell,
        modes,
        coefficients,
        jnp.asarray([2.0, 1.0]),
        twist=jnp.zeros((2,), dtype=jnp.float64),
        resource_plan=resource,
    )
    reference = PeriodicFermiNet(
        cell,
        modes,
        coefficients[1:],
        jnp.ones((1,), dtype=jnp.float64),
        twist=jnp.zeros((2,), dtype=jnp.float64),
        resource_plan=ElectronicVMCResourcePlan(
            2,
            determinant_count=1,
            spatial_dimension=2,
        ),
    )
    coordinates = jnp.asarray([[0.1, 0.2], [0.7, 0.3]], dtype=jnp.float64)

    actual = model(coordinates)
    expected = reference(coordinates)

    assert bool(actual.valid & expected.valid)
    assert jnp.allclose(actual.log_abs, expected.log_abs)
    assert jnp.allclose(actual.phase, expected.phase)


def test_periodic_ferminet_singular_component_retains_parameter_derivative():
    cell = PeriodicCell(jnp.asarray([[1.0]], dtype=jnp.float64))
    modes = jnp.asarray([[0], [1]], dtype=jnp.int32)
    coefficients = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=jnp.complex128,
    )
    mixing = jnp.ones((2,), dtype=jnp.complex128)
    model = PeriodicFermiNet(
        cell,
        modes,
        coefficients,
        mixing,
        twist=jnp.zeros((1,), dtype=jnp.float64),
        resource_plan=ElectronicVMCResourcePlan(
            2,
            determinant_count=2,
            spatial_dimension=1,
        ),
    )
    coordinates = jnp.asarray([[0.1], [0.4]], dtype=jnp.float64)

    def candidate_coefficients(parameter):
        return coefficients.at[0, 1, 1].set(parameter)

    def amplitude_log_abs(parameter):
        candidate = eqx.tree_at(
            lambda value: value.orbital_coefficients,
            model,
            candidate_coefficients(parameter),
        )
        return candidate(coordinates).log_abs

    def explicit_log_abs(parameter):
        features = model.cell_features(coordinates).reciprocal_features
        matrices = jax.vmap(lambda value: features @ value.T)(
            candidate_coefficients(parameter)
        )
        determinants = (
            matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
        )
        return jnp.log(jnp.abs(jnp.sum(mixing * determinants)))

    actual = jax.grad(amplitude_log_abs)(jnp.asarray(0.0))
    expected = jax.grad(explicit_log_abs)(jnp.asarray(0.0))

    assert jnp.isfinite(actual)
    assert jnp.abs(actual) > 1.0e-6
    assert jnp.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-11)


def test_periodic_mixture_avoids_zero_times_infinite_lane_scaling():
    cell = PeriodicCell(jnp.asarray([[1.0]], dtype=jnp.float64))
    modes = jnp.asarray([[0], [1]], dtype=jnp.int32)
    coefficients = jnp.asarray(
        [
            [[1.0e300, 0.0], [1.0e300, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=jnp.complex128,
    )
    coordinates = jnp.asarray([[0.1], [0.4]], dtype=jnp.float64)
    model = PeriodicFermiNet(
        cell,
        modes,
        coefficients,
        jnp.asarray([0.0, 1.0], dtype=jnp.complex128),
        twist=jnp.zeros((1,), dtype=jnp.float64),
        resource_plan=ElectronicVMCResourcePlan(
            2,
            determinant_count=2,
            spatial_dimension=1,
        ),
    )
    reference = PeriodicFermiNet(
        cell,
        modes,
        coefficients[1:],
        jnp.ones((1,), dtype=jnp.complex128),
        twist=jnp.zeros((1,), dtype=jnp.float64),
        resource_plan=ElectronicVMCResourcePlan(
            2,
            determinant_count=1,
            spatial_dimension=1,
        ),
    )

    actual = eqx.filter_jit(model)(coordinates)
    expected = eqx.filter_jit(reference)(coordinates)

    assert bool(actual.valid & expected.valid)
    assert jnp.isfinite(actual.log_abs)
    assert jnp.allclose(actual.log_abs, expected.log_abs)
    assert jnp.allclose(actual.phase, expected.phase)


def test_periodic_ferminet_incremental_target_is_exact_jittable_and_rebases():
    cell = PeriodicCell(2.0 * jnp.eye(2, dtype=jnp.float64))
    modes = jnp.asarray([[0, 0], [1, 0], [0, 1]], dtype=jnp.int32)
    coefficients = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.2j], [0.0, 0.3, 1.0]],
        ],
        dtype=jnp.complex128,
    )
    mixing = jnp.asarray([1.0 + 0.2j, -0.35 + 0.6j])
    resource = ElectronicVMCResourcePlan(
        2,
        determinant_count=2,
        spatial_dimension=2,
    )
    model = PeriodicFermiNet(
        cell,
        modes,
        coefficients,
        mixing,
        twist=jnp.asarray([0.13, -0.27]),
        pair_jastrow_strength=0.12,
        resource_plan=resource,
    )
    policy = _determinant_update_policy()
    target = periodic_ferminet_incremental_target(
        model,
        capacity=1,
        update_policy=policy,
        maximum_chains=4,
        refresh_cadence=2,
    )
    current = jnp.asarray([[0.1, 0.2], [0.7, 0.4]], dtype=jnp.float64)
    proposed = current.at[1, 0].add(0.11)
    payload = SingleCoordinateProposalPayload(
        index=jnp.asarray(2, dtype=jnp.int32),
        displacement=jnp.asarray(0.11),
    )

    state = eqx.filter_jit(
        lambda bound_target, position: bound_target.initialize(position)
    )(target, current)
    proposal = eqx.filter_jit(
        lambda bound_target, bound_state, position, move_payload: bound_target.propose(
            bound_state, position, move_payload
        )
    )(target, state, proposed, payload)
    expected_current = model(current)
    expected_proposed = model(proposed)

    assert isinstance(state.cache, PeriodicFermiNetCache)
    assert bool(state.valid & proposal.valid)
    assert jnp.allclose(state.log_target, 2.0 * expected_current.log_abs)
    assert jnp.allclose(
        proposal.log_ratio,
        2.0 * (expected_proposed.log_abs - expected_current.log_abs),
    )
    assert jnp.allclose(proposal.proposed_cache.phase, expected_proposed.phase)

    accepted = target.commit(state, proposed, proposal, jnp.asarray(True))
    rejected = target.commit(state, proposed, proposal, jnp.asarray(False))
    assert jnp.all(accepted.cache.sequences.active_rank == 1)
    assert bool(accepted.cache.compact_update)
    assert jnp.all(
        accepted.cache.low_rank_status == int(LowRankDeterminantStatus.SUCCESS)
    )
    assert jnp.allclose(accepted.log_target, 2.0 * expected_proposed.log_abs)
    assert jnp.allclose(rejected.position, current)
    assert jnp.allclose(rejected.log_target, state.log_target)

    proposed_again = proposed.at[0, 1].add(-0.09)
    payload_again = SingleCoordinateProposalPayload(
        index=jnp.asarray(1, dtype=jnp.int32),
        displacement=jnp.asarray(-0.09),
    )
    proposal_again = target.propose(accepted, proposed_again, payload_again)
    accepted_again = target.commit(
        accepted,
        proposed_again,
        proposal_again,
        jnp.asarray(True),
    )
    expected_again = model(proposed_again)
    assert bool(proposal_again.valid)
    assert jnp.all(accepted_again.cache.sequences.active_rank == 0)
    assert bool(accepted_again.cache.rebased)
    assert jnp.all(
        accepted_again.cache.low_rank_status
        == int(LowRankDeterminantStatus.CAPACITY_EXCEEDED)
    )
    assert jnp.allclose(accepted_again.log_target, 2.0 * expected_again.log_abs)
    assert bool(target.refresh(accepted_again).valid)

    batched = jax.vmap(target.initialize)(jnp.stack((current, proposed)))
    assert batched.log_target.shape == (2,)
    assert jnp.all(batched.valid)

    reconstructed = PeriodicFermiNet(
        cell,
        modes,
        coefficients * (1.0 + 0.03j),
        mixing * (0.8 - 0.1j),
        twist=jnp.asarray([0.13, -0.27]),
        pair_jastrow_strength=0.2,
        resource_plan=resource,
    )
    reconstructed_target = periodic_ferminet_incremental_target(
        reconstructed,
        capacity=1,
        update_policy=_determinant_update_policy(),
        maximum_chains=4,
        refresh_cadence=2,
    )
    assert reconstructed_target.target_id == target.target_id


def test_periodic_ferminet_incremental_target_rebases_singular_components():
    cell = PeriodicCell(2.0 * jnp.eye(2, dtype=jnp.float64))
    modes = jnp.asarray([[0, 0], [1, 0]], dtype=jnp.int32)
    coefficients = jnp.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    model = PeriodicFermiNet(
        cell,
        modes,
        coefficients,
        jnp.asarray([2.0, 1.0]),
        twist=jnp.zeros((2,), dtype=jnp.float64),
        resource_plan=ElectronicVMCResourcePlan(
            2,
            determinant_count=2,
            spatial_dimension=2,
        ),
    )
    target = periodic_ferminet_incremental_target(
        model,
        capacity=2,
        update_policy=_determinant_update_policy(),
        maximum_chains=4,
    )
    current = jnp.asarray([[0.1, 0.2], [0.7, 0.3]], dtype=jnp.float64)
    proposed = current.at[0, 0].add(0.08)
    state = target.initialize(current)
    proposal = target.propose(
        state,
        proposed,
        SingleCoordinateProposalPayload(
            index=jnp.asarray(0, dtype=jnp.int32),
            displacement=jnp.asarray(0.08),
        ),
    )
    exact = model(proposed)

    assert bool(state.valid & proposal.valid & exact.valid)
    assert not bool(state.cache.compact_eligible[0])
    assert jnp.all(proposal.proposed_cache.sequences.active_rank == 0)
    assert bool(proposal.proposed_cache.rebased)
    assert proposal.proposed_cache.low_rank_status[0] == int(
        LowRankDeterminantStatus.BASE_SOLVE_FAILED
    )
    assert jnp.allclose(
        state.log_target + proposal.log_ratio,
        2.0 * exact.log_abs,
    )
    assert jnp.allclose(proposal.proposed_cache.phase, exact.phase)


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
