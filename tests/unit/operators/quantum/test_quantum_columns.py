#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from phydrax.operators.quantum import AbelianGroup, FermionModeOrder
from phydrax.operators.quantum.lattice import (
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice_columns,
    QuantumColumnResourcePolicy,
    QuantumConfigurationDomain,
    QuantumLatticeColumnOperator,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    refresh_quantum_lattice_columns,
)


def _operator(
    specification: QuantumLatticeSpecification,
    resources: QuantumColumnResourcePolicy | None = None,
) -> QuantumLatticeColumnOperator:
    policy = QuantumColumnResourcePolicy() if resources is None else resources
    domain = QuantumConfigurationDomain(
        specification, species_ids=("declared-component",) * len(specification.spaces)
    )
    return QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(specification, policy), domain
    )


def _dense_column(
    operator: QuantumLatticeColumnOperator, source: int, dimension: int
) -> npt.NDArray[np.complex128]:
    result = operator.outgoing_column(operator.domain.address((source,)))
    assert bool(result.successful)
    dense = np.zeros(dimension, dtype=np.complex128)
    for key, value, active in zip(
        np.asarray(result.target_keys),
        np.asarray(result.matrix_elements),
        np.asarray(result.valid),
        strict=True,
    ):
        if active:
            target = np.asarray(
                operator.domain.decode(jnp.asarray(key, dtype=jnp.uint32))
            )[0]
            dense[target] = value
    return dense


def test_complex_action_is_outgoing_not_row_or_adjoint() -> None:
    space = LocalSpacePlan("site", ("a", "b", "c"), (), np.zeros((3, 0), dtype=np.int32))
    matrix = np.asarray(
        ((2, 1 + 2j, 0), (3 - 4j, -1, 5j), (-2j, 7, 4)), dtype=np.complex128
    )
    local = LocalOperatorPlan(space, "complex-matrix", matrix, ())
    specification = QuantumLatticeSpecification(
        (space,),
        (QuantumLatticeTerm((local,), coefficient=2 - 1j, label="complex-term"),),
    )
    operator = _operator(specification)
    np.testing.assert_allclose(_dense_column(operator, 0, 3), (2 - 1j) * matrix[:, 0])
    np.testing.assert_allclose(_dense_column(operator, 1, 3), (2 - 1j) * matrix[:, 1])


def test_repeated_site_ladders_return_to_source_in_exact_diagonal() -> None:
    space = LocalSpacePlan.boson("b", 3)
    creation = np.asarray(((0, 0, 0), (1, 0, 0), (0, np.sqrt(2), 0)), dtype=np.complex128)
    create = LocalOperatorPlan(space, "create", creation, (1,))
    destroy = LocalOperatorPlan(space, "destroy", creation.conj().T, (-1,))
    specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((destroy, create), label="ladder-return"),)
    )
    operator = _operator(specification)
    for source, diagonal in enumerate((1.0, 2.0, 0.0)):
        address = operator.domain.address((source,))
        np.testing.assert_allclose(operator.diagonal(address), diagonal)
        expected = np.zeros(3, dtype=np.complex128)
        expected[source] = diagonal
        np.testing.assert_allclose(_dense_column(operator, source, 3), expected)


def test_duplicate_routes_cancel_only_after_native_target_coalescing() -> None:
    space = LocalSpacePlan("q", ("0", "1"), (), np.zeros((2, 0), dtype=np.int32))
    flip = LocalOperatorPlan(space, "flip", ((0, 1), (1, 0)), ())
    specification = QuantumLatticeSpecification(
        (space,),
        (
            QuantumLatticeTerm((flip,), coefficient=1, label="positive"),
            QuantumLatticeTerm((flip,), coefficient=-1, label="negative"),
        ),
    )
    operator = _operator(specification)
    result = operator.outgoing_column(operator.domain.address((0,)))
    assert bool(result.successful)
    assert int(result.raw_route_count) == 2
    assert int(result.unique_target_count) == 1
    assert int(result.cancellation_count) == 1
    assert not bool(jnp.any(result.valid))
    np.testing.assert_allclose(result.matrix_elements, 0)


def test_hundred_mode_long_hop_counts_intermediate_fermion_spectator() -> None:
    labels = tuple(f"mode{i}" for i in range(100))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in labels)
    create = LocalOperatorPlan(spaces[0], "create", ((0, 0), (1, 0)), (1,))
    destroy = LocalOperatorPlan(spaces[-1], "destroy", ((0, 1), (0, 0)), (-1,))
    specification = QuantumLatticeSpecification(
        spaces,
        (
            QuantumLatticeTerm(
                (create, destroy), coefficient=1 + 2j, add_adjoint=True, label="long-hop"
            ),
        ),
        fermion_mode_order=FermionModeOrder(labels),
    )
    operator = _operator(specification)
    coordinate = np.zeros(100, dtype=np.int32)
    coordinate[[32, 99]] = 1
    source = operator.domain.address(coordinate)
    target_coordinate = coordinate.copy()
    target_coordinate[0] = 1
    target_coordinate[99] = 0
    target = operator.domain.address(target_coordinate)
    forward = operator.raw_route(operator.prepare_source(source), jnp.int32(0))
    reverse = operator.raw_route(target, jnp.int32(1))
    assert bool(forward.valid & forward.off_diagonal & forward.successful)
    np.testing.assert_array_equal(forward.target_key, target.key_words)
    np.testing.assert_allclose(forward.matrix_element, -1 - 2j)
    np.testing.assert_allclose(reverse.matrix_element, -1 + 2j)
    assert operator.raw_route_bound == 2


def _two_binding_specification(
    first: tuple[float, float], second: tuple[float, float]
) -> QuantumLatticeSpecification:
    space = LocalSpacePlan.spin("s", 1)
    a = LocalOperatorPlan(space, "same-structural-label", np.diag(first), (0,))
    b = LocalOperatorPlan(space, "same-structural-label", np.diag(second), (0,))
    return QuantumLatticeSpecification(
        (space,),
        (
            QuantumLatticeTerm((a,), label="first-occurrence"),
            QuantumLatticeTerm((b,), label="second-occurrence"),
        ),
    )


def test_equal_structural_operator_ids_do_not_alias_numeric_bindings() -> None:
    specification = _two_binding_specification((2, 3), (5, 11))
    assert (
        specification.terms[0].factors[0].operator_id
        == specification.terms[1].factors[0].operator_id
    )
    operator = _operator(specification)
    np.testing.assert_allclose(_dense_column(operator, 0, 2), (7, 0))
    np.testing.assert_allclose(_dense_column(operator, 1, 2), (0, 14))


def test_initially_equal_bindings_diverge_on_same_support_refresh() -> None:
    initial = _two_binding_specification((2, 3), (2, 3))
    prepared = prepare_quantum_lattice_columns(initial, QuantumColumnResourcePolicy())
    replacement = _two_binding_specification((2, 3), (5, 11))
    refreshed = refresh_quantum_lattice_columns(prepared, replacement)
    domain = QuantumConfigurationDomain(initial, species_ids=("spin",))
    before = QuantumLatticeColumnOperator(prepared, domain)
    after = QuantumLatticeColumnOperator(refreshed, domain)
    np.testing.assert_allclose(before.diagonal(domain.address((1,))), 6)
    np.testing.assert_allclose(after.diagonal(domain.address((1,))), 14)
    assert before.operator_id != after.operator_id
    assert prepared.structural_id == refreshed.structural_id


def test_refresh_recomputes_complex_adjoint_bindings() -> None:
    space = LocalSpacePlan("q", ("0", "1"), (), np.zeros((2, 0), dtype=np.int32))

    def specification(value: complex) -> QuantumLatticeSpecification:
        local = LocalOperatorPlan(space, "transition", ((0, value), (0, 0)), ())
        return QuantumLatticeSpecification(
            (space,),
            (
                QuantumLatticeTerm(
                    (local,), coefficient=2 - 1j, add_adjoint=True, label="paired"
                ),
            ),
        )

    initial = specification(1 + 1j)
    prepared = prepare_quantum_lattice_columns(initial, QuantumColumnResourcePolicy())
    refreshed = refresh_quantum_lattice_columns(prepared, specification(3 - 2j))
    domain = QuantumConfigurationDomain(initial, species_ids=("q",))
    operator = QuantumLatticeColumnOperator(refreshed, domain)
    np.testing.assert_allclose(_dense_column(operator, 0, 2), (0, (2 + 1j) * (3 + 2j)))
    np.testing.assert_allclose(_dense_column(operator, 1, 2), ((2 - 1j) * (3 - 2j), 0))


def test_numeric_refresh_refuses_changed_support() -> None:
    initial = _two_binding_specification((2, 3), (2, 3))
    prepared = prepare_quantum_lattice_columns(initial, QuantumColumnResourcePolicy())
    with pytest.raises(ValueError, match="support"):
        refresh_quantum_lattice_columns(
            prepared, _two_binding_specification((2, 0), (2, 3))
        )


def test_charge_changing_hamiltonian_is_refused_not_filtered() -> None:
    space = LocalSpacePlan.boson("b", 3)
    create = LocalOperatorPlan(space, "create", ((0, 0, 0), (1, 0, 0), (0, 1, 0)), (1,))
    specification = QuantumLatticeSpecification(
        (space,),
        (QuantumLatticeTerm((create,), add_adjoint=True, label="change-number"),),
    )
    domain = QuantumConfigurationDomain(
        specification,
        species_ids=("boson",),
        charge_group=AbelianGroup((None,)),
        charge_labels=("boson-number",),
        total_charge=(1,),
    )
    with pytest.raises(ValueError, match="preserve"):
        QuantumLatticeColumnOperator(
            prepare_quantum_lattice_columns(specification, QuantumColumnResourcePolicy()),
            domain,
        )


def test_raw_route_expectation_uses_paths_not_coalesced_probability() -> None:
    space = LocalSpacePlan("q", ("0", "1"), (), np.zeros((2, 0), dtype=np.int32))
    a = np.asarray(((1, 2j), (3, 4)), dtype=np.complex128)
    b = np.asarray(((2, 1j), (5, -3)), dtype=np.complex128)
    coefficient = np.complex128(-0.3 + 0.5j)
    factors = (LocalOperatorPlan(space, "a", a, ()), LocalOperatorPlan(space, "b", b, ()))
    specification = QuantumLatticeSpecification(
        (space,),
        (
            QuantumLatticeTerm(
                factors, coefficient=coefficient, add_adjoint=True, label="branched"
            ),
        ),
    )
    operator = _operator(specification)
    source = operator.domain.address((0,))
    expectation = np.zeros(2, dtype=np.complex128)
    for index in range(8):
        component, route = divmod(index, 4)
        intermediate, target = route % 2, route // 2
        matrices = (a, b) if component == 0 else (b.conj().T, a.conj().T)
        scalar = coefficient if component == 0 else coefficient.conjugate()
        expected_route = (
            scalar * matrices[1][intermediate, 0] * matrices[0][target, intermediate]
        )
        raw = operator.raw_route(source, jnp.int32(index))
        assert bool(raw.valid & raw.successful)
        np.testing.assert_allclose(raw.p_raw, 1 / 8)
        np.testing.assert_allclose(raw.matrix_element, expected_route)
        np.testing.assert_array_equal(
            raw.target_key, operator.domain.address((target,)).key_words
        )
        expectation[target] += float(raw.p_raw) * complex(raw.matrix_element / raw.p_raw)
    np.testing.assert_allclose(
        expectation,
        (coefficient * (a @ b) + coefficient.conjugate() * (b.conj().T @ a.conj().T))[
            :, 0
        ],
    )
    for seed in (0, 1, 17):
        sampled = eqx.filter_jit(operator.sample_raw_excitation)(
            jax.random.key(seed), source
        )
        reference = operator.raw_route(source, sampled.route_index)
        np.testing.assert_allclose(sampled.matrix_element, reference.matrix_element)
        np.testing.assert_allclose(sampled.p_raw, 1 / 8)
        np.testing.assert_array_equal(sampled.target_key, reference.target_key)


def test_dead_single_path_is_a_null_attempt_not_a_retry() -> None:
    space = LocalSpacePlan.boson("b", 2)
    create = LocalOperatorPlan(space, "create", ((0, 0), (1, 0)), (1,))
    operator = _operator(
        QuantumLatticeSpecification(
            (space,), (QuantumLatticeTerm((create,), label="create"),)
        )
    )
    sampled = operator.sample_raw_excitation(
        jax.random.key(3), operator.domain.address((1,))
    )
    assert bool(sampled.successful)
    assert not bool(sampled.valid | sampled.off_diagonal)
    np.testing.assert_allclose(sampled.matrix_element, 0)
    np.testing.assert_allclose(sampled.p_raw, 1)


def test_sparse_admission_uses_transition_width_not_local_dimension() -> None:
    space = LocalSpacePlan.boson("b", 128)
    create = LocalOperatorPlan(
        space, "create", np.diag(np.ones(127, dtype=np.float64), -1), (1,)
    )
    specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((create,), label="create"),)
    )
    prepared = prepare_quantum_lattice_columns(
        specification, QuantumColumnResourcePolicy(maximum_raw_routes=1)
    )
    assert prepared.raw_route_bound == 1
    np.testing.assert_allclose(
        _dense_column(_operator(specification, prepared.resources), 126, 128)[127], 1
    )


def test_genuinely_dense_transition_width_refuses_route_budget() -> None:
    space = LocalSpacePlan(
        "q", tuple(str(i) for i in range(16)), (), np.zeros((16, 0), dtype=np.int32)
    )
    local = LocalOperatorPlan(space, "dense", np.ones((16, 16), dtype=np.float64), ())
    specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((local,), label="dense"),)
    )
    with pytest.raises(ValueError, match="raw-route"):
        prepare_quantum_lattice_columns(
            specification, QuantumColumnResourcePolicy(maximum_raw_routes=8)
        )


def test_physical_route_bound_and_operator_id_exclude_resource_caps() -> None:
    specification = _two_binding_specification((2, 3), (5, 11))
    first = _operator(
        specification,
        QuantumColumnResourcePolicy(maximum_raw_routes=2, maximum_column_targets=2),
    )
    second = _operator(
        specification,
        QuantumColumnResourcePolicy(maximum_raw_routes=128, maximum_column_targets=128),
    )
    assert first.raw_route_bound == second.raw_route_bound == 2
    assert first.operator_id == second.operator_id
    assert first.prepared.prepared_id != second.prepared.prepared_id


def test_coalesced_target_capacity_has_explicit_failure_evidence() -> None:
    space = LocalSpacePlan("q", ("0", "1"), (), np.zeros((2, 0), dtype=np.int32))
    local = LocalOperatorPlan(space, "branch", ((1, 2), (3, 4)), ())
    specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((local,), label="branch"),)
    )
    operator = _operator(
        specification, QuantumColumnResourcePolicy(maximum_column_targets=1)
    )
    result = operator.outgoing_column(operator.domain.address((0,)))
    assert not bool(result.successful)


@pytest.mark.parametrize(
    ("resources", "message"),
    [
        (
            QuantumColumnResourcePolicy(maximum_transition_table_bytes=80),
            "transition-table",
        ),
        (
            QuantumColumnResourcePolicy(maximum_decoded_coordinate_bytes=3),
            "coordinate workset",
        ),
        (QuantumColumnResourcePolicy(maximum_workspace_bytes=200), "scratch"),
    ],
    ids=["transition-storage", "decoded-coordinate", "column-scratch"],
)
def test_sparse_resource_dimensions_refuse_independently(
    resources: QuantumColumnResourcePolicy, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_quantum_lattice_columns(
            _two_binding_specification((2, 3), (5, 11)), resources
        )


def test_duplicate_logical_term_identity_is_still_refused() -> None:
    specification = _two_binding_specification((2, 3), (5, 11))
    with pytest.raises(ValueError, match="Duplicate"):
        QuantumLatticeSpecification(
            specification.spaces, (specification.terms[0], specification.terms[0])
        )


def test_diagonal_does_not_require_unrelated_coalesced_target_capacity() -> None:
    space = LocalSpacePlan("q", ("0", "1"), (), np.zeros((2, 0), dtype=np.int32))
    local = LocalOperatorPlan(space, "branch", ((2, 3), (5, 7)), ())
    specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((local,), label="branch"),)
    )
    operator = _operator(
        specification, QuantumColumnResourcePolicy(maximum_column_targets=1)
    )
    np.testing.assert_allclose(operator.diagonal(operator.domain.address((0,))), 2)
    assert not bool(operator.outgoing_column(operator.domain.address((0,))).successful)


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    ("coefficient", "dtype", "complex_dtype"),
    [
        (0.25, "float16", "complex64"),
        (0.25, "float32", "complex64"),
        (0.25, "float64", "complex128"),
        (2, "int32", "complex128"),
        (True, "bool", "complex128"),
        (0.25, "complex64", "complex64"),
        (0.25, "complex128", "complex128"),
    ],
)
def test_real_coefficient_hamiltonian_action_under_strict_promotion(
    coefficient: float | int | bool, dtype: str, complex_dtype: str
) -> None:
    space = LocalSpacePlan("q", ("0", "1"), (), np.zeros((2, 0), dtype=np.int32))
    matrix = np.asarray(((2, 3 - 4j), (3 + 4j, -1)), dtype=np.complex128)
    local = LocalOperatorPlan(space, "hermitian", matrix, ())
    term = QuantumLatticeTerm(
        (local,), coefficient=jnp.asarray(coefficient, dtype=dtype), label="scaled"
    )
    operator = _operator(QuantumLatticeSpecification((space,), (term,)))
    assert term.coefficient.dtype == jax.dtypes.canonicalize_dtype(complex_dtype)
    for source in range(2):
        np.testing.assert_allclose(
            _dense_column(operator, source, 2), coefficient * matrix[:, source]
        )
        np.testing.assert_allclose(
            operator.diagonal(operator.domain.address((source,))),
            coefficient * matrix[source, source],
        )
