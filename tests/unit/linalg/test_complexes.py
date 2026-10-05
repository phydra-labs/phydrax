from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _triangle(*, filled: bool = True, complex_values: bool = False) -> la.HilbertComplex:
    dtype = jnp.complex128 if complex_values else jnp.float64
    weights = ((2.0, 3.0, 5.0), (7.0, 11.0, 13.0), (17.0,))
    sizes = (3, 3, 1) if filled else (3, 3)
    spaces = tuple(
        la.ArraySpace(
            (size,),
            dtype=dtype,
            pairing=la.DiagonalPairing(jnp.asarray(weights[k], dtype=jnp.float64)),
            space_id=f"triangle:{filled}:{complex_values}:{k}",
        )
        for k, size in enumerate(sizes)
    )
    d0 = jnp.asarray([[-1, 1, 0], [0, -1, 1], [-1, 0, 1]], dtype=dtype)
    if complex_values:
        d0 = d0 * (1 + 2j)
    matrices = (d0, jnp.asarray([[1, 1, -1]], dtype=dtype)) if filled else (d0,)
    operators = []
    for k, matrix in enumerate(matrices):
        rows, columns = np.nonzero(np.asarray(matrix))
        relation = EdgeRelation(
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=spaces[k].size,
            target_size=spaces[k + 1].size,
        )
        operators.append(
            SparseCoordinateOperator(
                relation,
                matrix[rows, columns],
                source=spaces[k],
                target=spaces[k + 1],
                operator_id=f"triangle:d:{filled}:{complex_values}:{k}",
            )
        )
    return la.HilbertComplex(
        spaces, operators, complex_id=f"triangle:{filled}:{complex_values}"
    )


def _matrix(operator: la.AbstractLinearOperator) -> Array:
    return la.materialize(operator, la.MaterializationPolicy(max_entries=10000))


@pytest.mark.parametrize("complex_values", [False, True])
def test_hilbert_duality_and_laplacian_energy(complex_values: bool) -> None:
    complex = _triangle(complex_values=complex_values)
    dtype = jnp.complex128 if complex_values else jnp.float64
    x = jnp.asarray([1, -2, 3], dtype=dtype)
    y = jnp.asarray([2, 3, -1], dtype=dtype)
    d0 = complex.differential(0)
    delta = la.codifferential(complex, 1)
    np.testing.assert_allclose(
        complex.space(1).inner(d0.mv(x), y),
        complex.space(0).inner(x, delta.mv(y)),
        atol=1e-12,
    )
    laplacian = la.hodge_laplacian(complex, 1)
    energy = complex.space(1).inner(y, laplacian.mv(y))
    lower, upper = delta.mv(y), complex.differential(1).mv(y)
    expected = complex.space(0).inner(lower, lower) + complex.space(2).inner(upper, upper)
    np.testing.assert_allclose(energy, expected, atol=1e-10)
    np.testing.assert_allclose(
        _matrix(laplacian).T @ y, laplacian.transpose_mv(y), atol=1e-10
    )
    assert laplacian.properties.certifies("self_adjoint")
    assert laplacian.properties.certifies("positive_semidefinite")
    evidence = jax.jit(lambda key: la.complex_nilpotency_evidence(complex, key=key))(
        jax.random.key(4)
    )
    assert bool(evidence.valid)
    with pytest.raises(ValueError):
        la.codifferential(complex, 0)
    with pytest.raises(ValueError):
        la.hodge_laplacian(complex, 0, part="lower")
    with pytest.raises(ValueError):
        la.hodge_laplacian(complex, 2, part="upper")


def test_weak_coordinate_grams_are_not_semantic_dual_endomorphisms() -> None:
    complex = _triangle()
    mass = la.mass_form(complex, 1)
    stiffness = la.stiffness_form(complex, 1)
    assert mass.source.compatible(mass.target)
    assert not mass.source.compatible(complex.space(1))
    assert mass.properties.certifies("positive_definite")
    d1 = np.asarray(_matrix(complex.differential(1)))
    np.testing.assert_allclose(
        _matrix(stiffness), d1.T @ np.diag([17.0]) @ d1, atol=1e-12
    )
    weak = la.hodge_laplacian_form(complex, 1)
    np.testing.assert_allclose(
        _matrix(weak),
        np.diag([7.0, 11.0, 13.0]) @ _matrix(la.hodge_laplacian(complex, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(_matrix(weak), _matrix(weak).T, atol=1e-12)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_prepared_decomposition_reconstructs_orthogonal_components(degree: int) -> None:
    complex = _triangle()
    harmonic = la.harmonic_subspace(
        complex, degree, expected_dimension=1 if degree == 0 else 0
    )
    lower = (
        None
        if degree == 0
        else la.harmonic_subspace(
            complex, degree - 1, expected_dimension=1 if degree == 1 else 0
        )
    )
    prepared = la.prepare_hodge_decomposition(
        complex, degree, harmonic=harmonic, lower_harmonic=lower
    )
    value = jnp.arange(1, complex.space(degree).size + 1, dtype=jnp.float64)
    decomposition = eqx.filter_jit(lambda prepared, values: prepared.apply(values))(
        prepared, value
    )
    assert bool(decomposition.valid)
    np.testing.assert_allclose(
        decomposition.exact + decomposition.coexact + decomposition.harmonic,
        value,
        atol=1e-10,
    )
    if degree > 0:
        np.testing.assert_allclose(
            la.codifferential(complex, degree).mv(decomposition.coexact), 0, atol=1e-8
        )
        assert decomposition.exact_potential is not None
        np.testing.assert_allclose(
            complex.differential(degree - 1).mv(decomposition.exact_potential),
            decomposition.exact,
            atol=1e-10,
        )
    if degree < complex.top_degree:
        np.testing.assert_allclose(
            complex.differential(degree).mv(decomposition.exact), 0, atol=1e-10
        )
    repeated = prepared.apply(2 * value)
    np.testing.assert_allclose(repeated.exact, 2 * decomposition.exact, atol=1e-9)
    np.testing.assert_allclose(repeated.coexact, 2 * decomposition.coexact, atol=1e-9)
    if degree > 0:
        with pytest.raises(ValueError):
            la.prepare_hodge_decomposition(complex, degree, harmonic=harmonic)


def test_nontrivial_ring_harmonic_projector_and_dimension_faults() -> None:
    complex = _triangle(filled=False)
    harmonic = la.harmonic_subspace(complex, 1, expected_dimension=1)
    assert bool(harmonic.valid)
    expected_vector = np.asarray([1 / 7, 1 / 11, -1 / 13], dtype=np.float64)
    expected_vector /= np.sqrt(
        expected_vector @ np.diag([7.0, 11.0, 13.0]) @ expected_vector
    )
    actual_projector = np.asarray(harmonic.basis @ harmonic.basis.T) @ np.diag(
        [7.0, 11.0, 13.0]
    )
    expected_projector = np.outer(expected_vector, expected_vector) @ np.diag(
        [7.0, 11.0, 13.0]
    )
    np.testing.assert_allclose(actual_projector, expected_projector, atol=1e-10)
    assert not bool(
        la.harmonic_subspace(complex, 1, expected_dimension=0).dimension_match
    )
    assert not bool(la.harmonic_subspace(complex, 1, expected_dimension=2).valid)
    inferred = la.harmonic_subspace(complex, 1)
    assert inferred.dimension == 1
    with pytest.raises(ValueError):
        la.harmonic_subspace(
            complex, 1, policy=la.HarmonicSubspacePolicy(dense_dimension=0)
        )


def test_mixed_sparse_saddle_solution_and_reusable_numeric_pattern() -> None:
    complex = _triangle()
    harmonic = la.harmonic_subspace(complex, 0, expected_dimension=1)
    operator = la.mixed_hodge_laplacian(complex, 0, harmonic=harmonic)
    assembly_policy = la.SparseAssemblyPolicy(
        materialization=la.MaterializationPolicy(max_entries=1000)
    )
    plan = la.plan_sparse_assembly(operator, assembly_policy)
    prepared = la.prepare_sparse_assembly(plan, operator)
    d = np.asarray(
        [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64
    )
    mass = np.diag(np.asarray([2.0, 3.0, 5.0], dtype=np.float64))
    edge_mass = np.diag(np.asarray([7.0, 11.0, 13.0], dtype=np.float64))
    expected = np.block(
        [
            [d.T @ edge_mass @ d, mass @ np.asarray(harmonic.basis)],
            [np.asarray(harmonic.basis).T @ mass, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    np.testing.assert_allclose(_matrix(prepared.operator), expected, atol=1e-12)
    rhs = operator.source.unflatten(jnp.asarray([1.0, 2.0, -1.0, 0.0], dtype=jnp.float64))
    result = la.solve(
        la.LinearSystem(prepared.operator),
        rhs,
        policy=la.LinearSolvePolicy(la.SparseLU(provider="jax-cpu")),
    )
    np.testing.assert_allclose(
        operator.source.flatten(result.value),
        np.linalg.solve(expected, np.asarray(operator.source.flatten(rhs))),
        atol=1e-9,
    )
    assert bool(result.status == 0)
    changed = eqx.tree_at(lambda h: h.basis, harmonic, -harmonic.basis)
    refreshed = la.refresh_sparse_assembly(
        prepared, la.mixed_hodge_laplacian(complex, 0, harmonic=changed)
    )
    updated = expected.copy()
    updated[:3, 3] *= -1
    updated[3, :3] *= -1
    np.testing.assert_allclose(_matrix(refreshed.operator), updated, atol=1e-12)
    with pytest.raises(la.LinearCapabilityError):
        la.plan_sparse_assembly(
            operator,
            la.SparseAssemblyPolicy(
                max_nnz=1, materialization=la.MaterializationPolicy()
            ),
        )


def test_general_rectangular_named_sparse_blocks_preserve_holes() -> None:
    a = la.ArraySpace((2,), dtype=jnp.float64, space_id="block:a")
    b = la.ArraySpace((1,), dtype=jnp.float64, space_id="block:b")
    source = la.BlockSpace((a, b), names=("a", "b"))
    target = la.BlockSpace((b, a), names=("b", "a"))
    cross = la.DenseLinearOperator(jnp.asarray([[3.0], [5.0]]), source=b, target=a)
    row = la.DenseLinearOperator(jnp.asarray([[2.0, -1.0]]), source=a, target=b)
    blocks = la.BlockLinearOperator(
        ((row, None), (None, cross)), source=source, target=target
    )
    sparse = la.assemble_sparse(
        blocks,
        la.SparseAssemblyPolicy(materialization=la.MaterializationPolicy(max_entries=10)),
    )
    np.testing.assert_array_equal(
        _matrix(sparse),
        np.asarray(
            [[2.0, -1.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 5.0]], dtype=np.float64
        ),
    )


def test_complex_map_commutation_and_hilbert_adjoint() -> None:
    complex = _triangle(complex_values=True)
    maps = tuple(la.IdentityLinearOperator(space) for space in complex.spaces)
    identity = la.ComplexMap(complex, complex, maps, map_id="triangle:identity")
    assert bool(la.complex_map_evidence(identity, key=jax.random.key(1)).valid)
    bad = la.ComplexMap(
        complex, complex, (2.0 * maps[0], *maps[1:]), map_id="triangle:bad"
    )
    assert not bool(la.complex_map_evidence(bad, key=jax.random.key(1)).valid)
    adjoint = bad.adjoint()
    x = jnp.asarray([1 + 1j, 2 - 1j, 3j], dtype=jnp.complex128)
    y = jnp.asarray([3 - 2j, 2j, -1j], dtype=jnp.complex128)
    np.testing.assert_allclose(
        complex.space(0).inner(bad.map(0).mv(x), y),
        complex.space(0).inner(x, adjoint.map(0).mv(y)),
        atol=1e-12,
    )


def test_shifted_map_checks_initial_missing_endpoint() -> None:
    source_space = la.ArraySpace((1,), dtype=jnp.float64, space_id="shift:s")
    target_space = la.ArraySpace((1,), dtype=jnp.float64, space_id="shift:t")
    source = la.HilbertComplex(
        (source_space, source_space),
        (la.IdentityLinearOperator(source_space),),
        complex_id="shift:source",
    )
    target = la.HilbertComplex((target_space,), (), complex_id="shift:target")
    degree_map = la.DenseLinearOperator(
        jnp.ones((1, 1), dtype=jnp.float64), source=source_space, target=target_space
    )
    map = la.ComplexMap(
        source, target, (degree_map,), degree_offset=-1, map_id="shift:bad"
    )
    evidence = la.complex_map_evidence(map, key=jax.random.key(2))
    assert not bool(evidence.valid)


def test_large_harmonic_path_uses_native_eigen_resources() -> None:
    size = 32
    space = la.ArraySpace((size,), dtype=jnp.float64, space_id="diagonal:large")
    diagonal = jnp.sqrt(jnp.arange(size, dtype=jnp.float64))
    complex = la.HilbertComplex(
        (space, space),
        (la.DiagonalLinearOperator(diagonal, space=space),),
        complex_id="diagonal:large:complex",
    )
    policy = la.HarmonicSubspacePolicy(
        dense_dimension=0,
        tolerance=1e-6,
        eigen_policy=la.eigen.EigenSolvePolicy(
            la.eigen.LOBPCG(block_dimension=3),
            count=3,
            max_steps=300,
            key=jax.random.key(8),
            tolerance=la.eigen.EigenTolerancePolicy(relative=1e-8, absolute=1e-8),
        ),
    )
    harmonic = la.harmonic_subspace(complex, 0, expected_dimension=1, policy=policy)
    assert bool(harmonic.valid)
    expected = np.zeros((size, size), dtype=np.float64)
    expected[0, 0] = 1
    np.testing.assert_allclose(harmonic.basis @ harmonic.basis.T, expected, atol=1e-6)
    refused = la.HarmonicSubspacePolicy(
        dense_dimension=0,
        eigen_policy=la.eigen.EigenSolvePolicy(
            la.eigen.LOBPCG(),
            count=2,
            resources=la.eigen.EigenResourcePolicy(krylov_basis_bytes=0),
        ),
    )
    with pytest.raises(ValueError):
        la.harmonic_subspace(complex, 0, expected_dimension=1, policy=refused)


def test_nondiagonal_pairings_use_true_adjoint_and_harmonic_kernel() -> None:
    matrices = (
        jnp.asarray(
            [[3.0, 0.5, 0.1], [0.5, 4.0, 0.2], [0.1, 0.2, 5.0]], dtype=jnp.float64
        ),
        jnp.asarray(
            [[5.0, 0.2, 0.4], [0.2, 6.0, 0.3], [0.4, 0.3, 7.0]], dtype=jnp.float64
        ),
    )
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={"self_adjoint": "construction", "positive_definite": "construction"},
    )
    spaces = []
    for k, matrix in enumerate(matrices):
        coordinates = la.ArraySpace(
            (3,), dtype=jnp.float64, space_id=f"dense-metric:coordinates:{k}"
        )
        gram = la.DenseLinearOperator(
            matrix,
            source=coordinates,
            target=coordinates,
            properties=properties,
            operator_id=f"dense-metric:gram:{k}",
        )
        inverse = la.prepare(
            la.LinearSystem(gram), la.LinearSolvePolicy(la.DenseCholesky())
        )
        pairing = la.OperatorPairing(
            gram, prepared_inverse=inverse, pairing_id=f"dense-metric:pairing:{k}"
        )
        spaces.append(
            la.ArraySpace(
                (3,),
                dtype=jnp.float64,
                pairing=pairing,
                space_id=f"dense-metric:primal:{k}",
            )
        )
    d = jnp.asarray(
        [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=jnp.float64
    )
    differential = la.DenseLinearOperator(
        d, source=spaces[0], target=spaces[1], operator_id="dense-metric:d"
    )
    complex = la.HilbertComplex(spaces, (differential,), complex_id="dense-metric:ring")
    expected_delta = np.linalg.solve(
        np.asarray(matrices[0]), np.asarray(d).T @ np.asarray(matrices[1])
    )
    np.testing.assert_allclose(
        _matrix(la.codifferential(complex, 1)), expected_delta, atol=1e-10
    )
    harmonic = la.harmonic_subspace(complex, 1, expected_dimension=1)
    lower = la.harmonic_subspace(complex, 0, expected_dimension=1)
    assert bool(harmonic.valid)
    vector = np.linalg.solve(np.asarray(matrices[1]), np.asarray([1.0, 1.0, -1.0]))
    vector /= np.sqrt(vector @ np.asarray(matrices[1]) @ vector)
    np.testing.assert_allclose(
        harmonic.basis @ harmonic.basis.T, np.outer(vector, vector), atol=1e-10
    )
    value = jnp.asarray([2.0, -1.0, 4.0], dtype=jnp.float64)
    result = la.prepare_hodge_decomposition(
        complex, 1, harmonic=harmonic, lower_harmonic=lower
    ).apply(value)
    assert bool(result.valid)
    np.testing.assert_allclose(result.coexact, 0, atol=1e-8)
    np.testing.assert_allclose(
        result.harmonic,
        np.outer(vector, vector) @ np.asarray(matrices[1]) @ value,
        atol=1e-9,
    )


def test_traced_metric_energy_gradient_matches_dirichlet_energy() -> None:
    matrix = jnp.asarray(
        [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=jnp.float64
    )
    value = jnp.asarray([2.0, -1.0, 4.0], dtype=jnp.float64)

    def energy(weights: Array) -> Array:
        source = la.ArraySpace(
            (3,),
            dtype=jnp.float64,
            pairing=la.DiagonalPairing(jnp.asarray([2.0, 3.0, 4.0])),
            space_id="dynamic:source",
        )
        target = la.ArraySpace(
            (3,),
            dtype=jnp.float64,
            pairing=la.DiagonalPairing(weights),
            space_id="dynamic:target",
        )
        differential = la.DenseLinearOperator(
            matrix, source=source, target=target, operator_id="dynamic:d"
        )
        complex = la.HilbertComplex(
            (source, target), (differential,), complex_id="dynamic:complex"
        )
        return jnp.real(source.inner(value, la.hodge_laplacian(complex, 0).mv(value)))

    gradient = jax.jit(jax.grad(energy))(jnp.asarray([5.0, 6.0, 7.0], dtype=jnp.float64))
    np.testing.assert_allclose(
        gradient, np.square(np.asarray(matrix @ value)), atol=1e-10
    )


def test_explicit_harmonic_dimension_is_device_traceable() -> None:
    complex = _triangle(filled=False)
    harmonic = eqx.filter_jit(
        lambda current: la.harmonic_subspace(current, 1, expected_dimension=1)
    )(complex)
    assert bool(harmonic.valid)
    np.testing.assert_allclose(
        complex.differential(0).adjoint_mv_block(harmonic.basis),
        0,
        atol=1e-9,
    )
    with pytest.raises(ValueError):
        eqx.filter_jit(lambda current: la.harmonic_subspace(current, 1))(complex)


def test_empty_lower_active_space_has_no_exact_potential_solve() -> None:
    lower_space = la.ArraySpace((0,), dtype=jnp.float64, space_id="relative:empty")
    space = la.ArraySpace((2,), dtype=jnp.float64, space_id="relative:active")
    differential = la.DenseLinearOperator(
        jnp.zeros((2, 0), dtype=jnp.float64),
        source=lower_space,
        target=space,
        operator_id="relative:d",
    )
    complex = la.HilbertComplex(
        (lower_space, space), (differential,), complex_id="relative:complex"
    )
    harmonic = la.harmonic_subspace(complex, 1, expected_dimension=2)
    lower = la.harmonic_subspace(complex, 0, expected_dimension=0)
    values = jnp.asarray([2.0, -3.0], dtype=jnp.float64)
    result = la.prepare_hodge_decomposition(
        complex, 1, harmonic=harmonic, lower_harmonic=lower
    ).apply(values)
    assert bool(result.valid)
    np.testing.assert_allclose(result.harmonic, values, atol=1e-12)
    np.testing.assert_array_equal(result.exact, np.zeros((2,), dtype=np.float64))
    np.testing.assert_array_equal(result.coexact, np.zeros((2,), dtype=np.float64))


@pytest.mark.parametrize("filled", [False, True])
def test_real_hilbert_complex_accepts_complex_decomposition_with_sesquilinear_evidence(
    filled: bool,
) -> None:
    complex = _triangle(filled=filled)
    harmonic = la.harmonic_subspace(complex, 1, expected_dimension=0 if filled else 1)
    lower = la.harmonic_subspace(complex, 0, expected_dimension=1)
    prepared = la.prepare_hodge_decomposition(
        complex, 1, harmonic=harmonic, lower_harmonic=lower
    )
    values = jnp.asarray([1 + 2j, -3 + 4j, 2 - 1j], dtype=jnp.complex128)
    result = eqx.filter_jit(lambda current, value: current.apply(value))(prepared, values)
    d = np.asarray(
        [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64
    )
    mass = np.diag(np.asarray([7.0, 11.0, 13.0], dtype=np.float64))
    constraint = np.asarray([2.0, 3.0, 5.0], dtype=np.float64)[:, None]
    saddle = np.block(
        [[d.T @ mass @ d, constraint], [constraint.T, np.zeros((1, 1), dtype=np.float64)]]
    )
    rhs = np.concatenate(
        (d.T @ mass @ np.asarray(values), np.zeros((1,), dtype=np.complex128))
    )
    potential = np.linalg.solve(saddle, rhs)[:3]
    expected_exact = d @ potential
    np.testing.assert_allclose(result.exact, expected_exact, atol=1e-9)
    np.testing.assert_allclose(result.exact_potential, potential, atol=1e-9)
    expected_harmonic = (
        np.zeros((3,), dtype=np.complex128)
        if filled
        else np.asarray(values) - expected_exact
    )
    np.testing.assert_allclose(result.harmonic, expected_harmonic, atol=1e-9)
    np.testing.assert_allclose(
        result.coexact, np.asarray(values) - expected_exact - expected_harmonic, atol=1e-9
    )
    components = tuple(
        np.asarray(component)
        for component in (result.exact, result.coexact, result.harmonic)
    )
    defect = max(
        abs(np.vdot(components[i], mass @ components[j]))
        for i, j in ((0, 1), (0, 2), (1, 2))
    )
    np.testing.assert_allclose(result.orthogonality_defect, defect, atol=1e-10)
    assert bool(result.valid)


def test_three_block_sparse_hodge_solve_satisfies_strong_and_gauge_equations() -> None:
    complex = _triangle(filled=False)
    harmonic = la.harmonic_subspace(complex, 1, expected_dimension=1)
    mixed = la.mixed_hodge_laplacian(complex, 1, harmonic=harmonic)
    sparse = la.assemble_sparse(
        mixed,
        la.SparseAssemblyPolicy(
            materialization=la.MaterializationPolicy(max_entries=1000)
        ),
    )
    forcing = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float64)
    rhs = (
        jnp.zeros((3,), dtype=jnp.float64),
        jnp.asarray([7.0, 11.0, 13.0]) * forcing,
        jnp.zeros((1,), dtype=jnp.float64),
    )
    result = la.solve(
        la.LinearSystem(sparse),
        rhs,
        policy=la.LinearSolvePolicy(la.SparseLU(provider="jax-cpu")),
    )
    sigma, values, multiplier = mixed.source.unflatten(mixed.source.flatten(result.value))
    differential = np.asarray(
        [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64
    )
    lower_mass = np.diag(np.asarray([2.0, 3.0, 5.0], dtype=np.float64))
    mass = np.diag(np.asarray([7.0, 11.0, 13.0], dtype=np.float64))
    np.testing.assert_allclose(
        sigma,
        np.linalg.solve(lower_mass, differential.T @ mass @ np.asarray(values)),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        differential @ np.asarray(sigma)
        + np.asarray(harmonic.basis) @ np.asarray(multiplier),
        forcing,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(harmonic.basis).T @ mass @ np.asarray(values), 0, atol=1e-9
    )
    assert bool(result.status == 0)


def _weighted_ring(count: int) -> la.HilbertComplex:
    """Cycle graph with irregular positive weights; its first cohomology is one-dimensional."""
    rng = np.random.default_rng(7)
    spaces = tuple(
        la.ArraySpace(
            (count,),
            dtype=jnp.float64,
            pairing=la.DiagonalPairing(jnp.asarray(rng.uniform(0.2, 5.0, count))),
            space_id=f"ring:{count}:{k}",
        )
        for k in range(2)
    )
    edges = np.arange(count)
    relation = EdgeRelation(
        jnp.asarray(np.concatenate((edges, (edges + 1) % count)), dtype=jnp.int32),
        jnp.asarray(np.concatenate((edges, edges)), dtype=jnp.int32),
        source_size=count,
        target_size=count,
    )
    differential = SparseCoordinateOperator(
        relation,
        jnp.concatenate((-jnp.ones(count), jnp.ones(count))),
        source=spaces[0],
        target=spaces[1],
        operator_id=f"ring:{count}:d0",
    )
    return la.HilbertComplex(spaces, (differential,), complex_id=f"ring:{count}")


def test_large_harmonic_kernel_converges_by_default_and_reports_unconverged_runs() -> (
    None
):
    # 400 coordinates exceed dense_dimension=256, so the iterative route runs.
    # Independent reference: a harmonic 1-form of the ring has M h constant
    # along the cycle (co-closed), i.e. h ∝ 1 / w_e, normalized in the M-norm.
    complex = _weighted_ring(400)
    harmonic = la.harmonic_subspace(
        complex,
        1,
        expected_dimension=1,
        policy=la.HarmonicSubspacePolicy(tolerance=1e-10),
    )
    assert bool(harmonic.valid)
    assert float(harmonic.residual_defect) <= 1e-10
    space = complex.space(1)
    assert isinstance(space, la.ArraySpace)
    assert isinstance(space.pairing, la.DiagonalPairing)
    weights = np.asarray(space.pairing.weights)
    reference = 1.0 / weights
    reference /= np.sqrt(reference @ (weights * reference))
    basis = np.asarray(harmonic.basis)[:, 0]
    np.testing.assert_allclose(np.abs(basis @ (weights * reference)), 1.0, atol=1e-9)
    # A deliberately truncated unpreconditioned run must not claim the kernel.
    truncated = la.harmonic_subspace(
        complex,
        1,
        expected_dimension=1,
        policy=la.HarmonicSubspacePolicy(
            tolerance=1e-10,
            eigen_policy=la.eigen.EigenSolvePolicy(
                la.eigen.LOBPCG(block_dimension=3),
                count=3,
                max_steps=2,
                key=jax.random.key(0),
                tolerance=la.eigen.EigenTolerancePolicy(
                    relative=1e-10, absolute=1e-10, orthogonality=1e-10
                ),
            ),
        ),
    )
    assert not bool(truncated.valid)
    assert float(truncated.residual_defect) > 1e-10
