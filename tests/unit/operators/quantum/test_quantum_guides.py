#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from phydrax._strict import StrictModule
from phydrax.operators.quantum import LogAmplitude
from phydrax.operators.quantum.lattice import (
    GuidedQuantumColumnOperator,
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice_columns,
    QuantumAddress,
    QuantumColumnResourcePolicy,
    QuantumConfigurationDomain,
    QuantumGuide,
    QuantumLatticeColumnOperator,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from phydrax.solver import (
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloStatus,
    step_projector_monte_carlo,
)
from phydrax.typing import Complex128, Dim, Float64
from phydrax.units import ONE


class GuideStateDim(Dim):
    """Two states in the independently specified guide fixture."""


class FrozenTableProvider(StrictModule):
    __strict_contract__ = True

    logarithms: Float64[GuideStateDim]
    phases: Complex128[GuideStateDim]

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        state = address.key_words[0].astype(jnp.int32)
        return LogAmplitude(self.logarithms[state], self.phases[state])


class StaticMagnitudeProvider(StrictModule):
    logarithm: float = eqx.field(static=True)
    reciprocal: bool = eqx.field(static=True)

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        logarithm = -self.logarithm if self.reciprocal else self.logarithm
        return LogAmplitude(jnp.asarray(logarithm, dtype=jnp.float64))


class HiddenMutableProvider(StrictModule):
    state: npt.NDArray[np.float64] | list[float] | dict[str, float] = eqx.field(
        static=True
    )

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        value = self.state["logarithm"] if isinstance(self.state, dict) else self.state[0]
        return LogAmplitude(jnp.asarray(value, dtype=jnp.float64))


def _original(
    matrix: tuple[tuple[complex, complex], tuple[complex, complex]] = (
        (2 + 0j, 1 + 2j),
        (1 - 2j, 4 + 0j),
    ),
    *,
    coefficient: complex = 1 + 0j,
) -> QuantumLatticeColumnOperator:
    space = LocalSpacePlan("q", ("zero", "one"), (), np.zeros((2, 0), dtype=np.int32))
    local = LocalOperatorPlan(space, "matrix", matrix, ())
    specification = QuantumLatticeSpecification(
        (space,),
        (QuantumLatticeTerm((local,), coefficient=coefficient, label="hamiltonian"),),
    )
    domain = QuantumConfigurationDomain(specification, species_ids=("component",))
    return QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(specification, QuantumColumnResourcePolicy()),
        domain,
    )


def _provider(
    logarithms: tuple[float, float], phases: tuple[complex, complex] = (1 + 0j, 1 + 0j)
) -> FrozenTableProvider:
    return FrozenTableProvider(
        logarithms=jnp.asarray(logarithms, dtype=jnp.float64),
        phases=jnp.asarray(phases, dtype=jnp.complex128),
    )


def _guide(
    original: QuantumLatticeColumnOperator,
    logarithms: tuple[float, float],
    *,
    floor: float | None = None,
) -> QuantumGuide:
    return QuantumGuide(
        original.domain,
        _provider(logarithms),
        provider_id="independent-frozen-table",
        mapping_id="uint32-key-local-state-index",
        policy="reject" if floor is None else "positive-log-floor",
        log_floor=floor,
    )


def test_constant_three_guide_recovers_one_ninth_metric() -> None:
    original = _original()
    guide = _guide(original, (float(np.log(3)), float(np.log(3))))
    value, valid = eqx.filter_jit(guide.metric)(original.domain.address((0,)))
    assert bool(valid)
    np.testing.assert_allclose(value, 1 / 9, rtol=1e-14)
    represented = np.asarray((3 + 6j, -9 + 3j), dtype=np.complex128)
    physical = represented / 3
    np.testing.assert_allclose(
        np.vdot(represented, represented * float(value)), np.vdot(physical, physical)
    )


def test_nonconstant_positive_similarity_has_true_complex_column_orientation() -> None:
    original = _original()
    guide = _guide(original, (float(np.log(2)), float(np.log(3))))
    guided = GuidedQuantumColumnOperator(original, guide)
    source = original.domain.address((0,))
    result = guided.outgoing_column(source)
    assert bool(result.successful)
    expected = {0: 2 + 0j, 1: (3 / 2) * (1 - 2j)}
    for key, value, active in zip(
        np.asarray(result.target_keys),
        np.asarray(result.matrix_elements),
        np.asarray(result.valid),
        strict=True,
    ):
        if active:
            np.testing.assert_allclose(value, expected[int(key[0])])
    assert not guided.self_adjoint
    assert guided.original.self_adjoint
    route = guided.sample_raw_excitation(jax.random.key(3), guided.prepare_source(source))
    reference = original.raw_route(source, route.route_index)
    target = int(np.asarray(route.target_key)[0])
    np.testing.assert_allclose(
        route.matrix_element, reference.matrix_element * ((2, 3)[target] / 2)
    )
    np.testing.assert_allclose(route.p_raw, reference.p_raw)


def test_complex_physical_observable_and_bra_metric_recover_original_contraction() -> (
    None
):
    original = _original()
    guide = _guide(original, (float(np.log(2)), float(np.log(3))))
    physical_a = np.asarray((1 + 2j, -2 + 0.5j), dtype=np.complex128)
    physical_b = np.asarray((-0.3 + 1j, 4 - 2j), dtype=np.complex128)
    observable = np.asarray(((1, 2j), (3 - 4j, -2 + 1j)), dtype=np.complex128)
    magnitudes = np.asarray((2, 3), dtype=np.float64)
    represented_a = magnitudes * physical_a
    represented_b = magnitudes * physical_b
    physical_frame = np.zeros((2, 2), dtype=np.complex128)
    metric = np.zeros(2, dtype=np.float64)
    for bra in range(2):
        address_bra = original.domain.address((bra,))
        metric_value, metric_valid = guide.metric(address_bra)
        assert bool(metric_valid)
        metric[bra] = float(metric_value)
        for ket in range(2):
            value, valid = guide.physical_matrix_element(
                address_bra,
                original.domain.address((ket,)),
                jnp.asarray(observable[bra, ket], dtype=jnp.complex128),
            )
            assert bool(valid)
            physical_frame[bra, ket] = complex(value)
    np.testing.assert_allclose(
        np.vdot(represented_a, physical_frame @ represented_b),
        np.vdot(physical_a, observable @ physical_b),
    )
    np.testing.assert_allclose(
        np.vdot(represented_a, metric * represented_b), np.vdot(physical_a, physical_b)
    )
    # The general identity has a conjugated bra guide even when its phase is not
    # admitted by this positive-magnitude contract.
    complex_guide = magnitudes * np.exp(1j * np.asarray((0.4, -0.7), dtype=np.float64))
    general_frame = observable / (complex_guide.conj()[:, None] * complex_guide[None, :])
    np.testing.assert_allclose(
        np.vdot(complex_guide * physical_a, general_frame @ (complex_guide * physical_b)),
        np.vdot(physical_a, observable @ physical_b),
    )


def test_node_rejects_while_explicit_floor_is_the_same_in_action_and_metric() -> None:
    original = _original()
    rejected = _guide(original, (-np.inf, 0))
    floored = _guide(original, (-np.inf, 0), floor=-2)
    source = original.domain.address((0,))
    logarithm, valid = rejected.log_value(source)
    assert not bool(valid)
    assert bool(jnp.isneginf(logarithm))
    log_floor, floor_valid = floored.log_value(source)
    assert bool(floor_valid)
    np.testing.assert_allclose(log_floor, -2)
    metric, metric_valid = floored.metric(source)
    assert bool(metric_valid)
    np.testing.assert_allclose(metric, np.exp(4))
    guided = GuidedQuantumColumnOperator(original, floored)
    route = guided.raw_route(source, jnp.int32(1))
    assert bool(route.successful)
    np.testing.assert_allclose(route.matrix_element, (1 - 2j) * np.exp(2))


@pytest.mark.parametrize("invalid", [np.nan, np.inf], ids=["nan", "positive-infinity"])
def test_floor_never_repairs_invalid_log_amplitudes(invalid: float) -> None:
    original = _original()
    guide = _guide(original, (invalid, 0), floor=-5)
    _, valid = guide.log_value(original.domain.address((0,)))
    assert not bool(valid)
    route = GuidedQuantumColumnOperator(original, guide).raw_route(
        original.domain.address((0,)), jnp.int32(1)
    )
    assert not bool(route.successful)


def test_guide_snapshot_does_not_follow_replaced_provider_parameters() -> None:
    original = _original()
    provider = _provider((float(np.log(2)), float(np.log(3))))
    guide = QuantumGuide(
        original.domain, provider, provider_id="snapshot", mapping_id="local-index"
    )
    replacement = eqx.tree_at(
        lambda value: value.logarithms,
        provider,
        jnp.log(jnp.asarray((5, 7), dtype=jnp.float64)),
    )
    fresh = QuantumGuide(
        original.domain, replacement, provider_id="snapshot", mapping_id="local-index"
    )
    address = original.domain.address((1,))
    np.testing.assert_allclose(guide.log_value(address)[0], np.log(3))
    np.testing.assert_allclose(fresh.log_value(address)[0], np.log(7))
    assert guide.guide_id != fresh.guide_id


def test_positive_magnitude_discards_provider_phase_by_declared_contract() -> None:
    original = _original()
    provider = _provider((float(np.log(2)), float(np.log(3))), (1j, -1 + 0j))
    guide = QuantumGuide(
        original.domain,
        provider,
        provider_id="complex-amplitude",
        mapping_id="local-index",
    )
    value, valid = guide.metric(original.domain.address((1,)))
    assert bool(valid)
    np.testing.assert_allclose(value, 1 / 9)


def test_unrepresentable_similarity_ratio_and_metric_return_failure_not_jitter() -> None:
    original = _original()
    guide = _guide(original, (-1000, 1000))
    source = original.domain.address((0,))
    _, metric_valid = guide.metric(source)
    assert not bool(metric_valid)
    guided = GuidedQuantumColumnOperator(original, guide)
    result = eqx.filter_jit(guided.raw_route)(source, jnp.int32(1))
    assert not bool(result.successful)
    assert bool(result.valid & result.off_diagonal)


def test_generic_nonhermitian_original_is_not_admitted_by_guide() -> None:
    original = _original(((0 + 0j, 1 + 2j), (0 + 0j, 0 + 0j)))
    guide = _guide(original, (0, 0))
    with pytest.raises(ValueError, match="self-adjoint original"):
        GuidedQuantumColumnOperator(original, guide)


def test_same_shape_different_species_guide_domain_refuses() -> None:
    original = _original()
    space = LocalSpacePlan("q", ("zero", "one"), (), np.zeros((2, 0), dtype=np.int32))
    local = LocalOperatorPlan(space, "matrix", ((2, 1 + 2j), (1 - 2j, 4)), ())
    specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((local,), label="hamiltonian"),)
    )
    another_domain = QuantumConfigurationDomain(
        specification, species_ids=("different-component",)
    )
    guide = QuantumGuide(
        another_domain,
        _provider((0, 0)),
        provider_id="snapshot",
        mapping_id="local-index",
    )
    with pytest.raises(ValueError, match="different domains"):
        GuidedQuantumColumnOperator(original, guide)


def test_encountered_finite_value_does_not_hide_unseen_node() -> None:
    original = _original()
    guide = _guide(original, (0, -np.inf))
    assert bool(guide.log_value(original.domain.address((0,)))[1])
    result = GuidedQuantumColumnOperator(original, guide).outgoing_column(
        original.domain.address((0,))
    )
    assert not bool(result.successful)
    assert not guide.globally_positive


@pytest.mark.parametrize(
    ("logarithm", "reciprocal", "magnitude"),
    [(float(np.log(3)), False, 3.0), (float(np.log(2)), True, 0.5)],
    ids=["static-numeric-parameter", "static-control-parameter"],
)
def test_static_provider_parameters_change_frozen_identity_and_physical_metric(
    logarithm: float, reciprocal: bool, magnitude: float
) -> None:
    original = _original()
    first = QuantumGuide(
        original.domain,
        StaticMagnitudeProvider(logarithm=float(np.log(2)), reciprocal=False),
        provider_id="same-explicit-provider",
        mapping_id="same-explicit-mapping",
    )
    second = QuantumGuide(
        original.domain,
        StaticMagnitudeProvider(logarithm=logarithm, reciprocal=reciprocal),
        provider_id="same-explicit-provider",
        mapping_id="same-explicit-mapping",
    )
    source = original.domain.address((0,))
    assert first.numeric_id != second.numeric_id
    assert first.guide_id != second.guide_id
    np.testing.assert_allclose(first.metric(source)[0], 1 / 4)
    np.testing.assert_allclose(second.metric(source)[0], 1 / magnitude**2)
    cached = GuidedQuantumColumnOperator(original, first).prepare_source(source)
    with pytest.raises(ValueError, match="different frozen guide"):
        GuidedQuantumColumnOperator(original, second).raw_route(cached, jnp.int32(0))


@pytest.mark.filterwarnings("ignore:A JAX array is being set as static:UserWarning")
@pytest.mark.parametrize(
    "state",
    [
        np.asarray((np.log(2),), dtype=np.float64),
        [float(np.log(2))],
        {"logarithm": float(np.log(2))},
    ],
    ids=[
        "hidden-static-numpy-array",
        "hidden-static-numeric-list",
        "hidden-static-numeric-dict",
    ],
)
def test_hidden_static_mutable_provider_state_is_refused(
    state: npt.NDArray[np.float64] | list[float] | dict[str, float],
) -> None:
    original = _original()
    provider = HiddenMutableProvider(state=state)
    with pytest.raises(TypeError, match="mutable"):
        QuantumGuide(
            original.domain,
            provider,
            provider_id="mutable-provider",
            mapping_id="local-index",
        )


def test_similarity_product_underflow_refuses_raw_route_and_column() -> None:
    original = _original(((0j, 1e-300 + 0j), (1e-300 + 0j, 0j)))
    guided = GuidedQuantumColumnOperator(original, _guide(original, (0, -100)))
    source = original.domain.address((0,))
    assert float(jnp.exp(jnp.asarray(-100, dtype=jnp.float64))) > 0
    physical_route = original.raw_route(source, jnp.int32(0))
    assert bool(physical_route.valid & physical_route.off_diagonal)
    assert complex(physical_route.matrix_element) != 0
    route = eqx.filter_jit(guided.raw_route)(source, jnp.int32(0))
    assert bool(route.valid & route.off_diagonal)
    assert complex(route.matrix_element) == 0
    assert not bool(route.successful)
    column = eqx.filter_jit(guided.outgoing_column)(source)
    assert not bool(column.successful)
    np.testing.assert_array_equal(column.matrix_elements, 0)


@pytest.mark.parametrize("element", [0.0, 1e-300], ids=["legitimate-zero", "underflow"])
def test_physical_matrix_element_distinguishes_zero_from_product_underflow(
    element: float,
) -> None:
    original = _original()
    guide = _guide(original, (100, 0))
    bra = original.domain.address((0,))
    ket = original.domain.address((1,))
    value, valid = eqx.filter_jit(guide.physical_matrix_element)(
        bra, ket, jnp.asarray(element, dtype=jnp.complex128)
    )
    assert complex(value) == 0
    assert bool(valid) == (element == 0)


def test_zero_original_column_remains_successful_with_representable_guide_ratio() -> None:
    original = _original(((0j, 1 + 0j), (1 + 0j, 0j)), coefficient=0j)
    guided = GuidedQuantumColumnOperator(original, _guide(original, (0, -100)))
    source = original.domain.address((0,))
    route = eqx.filter_jit(guided.raw_route)(source, jnp.int32(0))
    assert bool(route.successful)
    assert complex(route.matrix_element) == 0
    column = eqx.filter_jit(guided.outgoing_column)(source)
    assert bool(column.successful)
    np.testing.assert_array_equal(column.matrix_elements, 0)


def test_compiled_projector_step_rolls_back_similarity_product_underflow() -> None:
    original = _original(((0j, 1e-300 + 0j), (1e-300 + 0j, 0j)))
    source = original.domain.address((0,))
    problem = ProjectorMonteCarloProblem(
        original,
        source.key_words[None, :],
        jnp.asarray((1 + 0j,), dtype=jnp.complex128),
        dt=1.0,
        energy_unit=ONE,
        inverse_energy_unit=ONE,
        provenance_id="similarity-product-underflow",
        guide=_guide(original, (0, -100)),
    )
    plan = ProjectorMonteCarloPlan(
        replicas=2,
        support_capacity=2,
        group_capacity=4,
        event_capacity=2,
        attempt_capacity=2,
        source_capacity=1,
        history_capacity=2,
        maximum_retained_bytes=10_000_000,
        maximum_workspace_bytes=10_000_000,
        spawn_policy="exact",
    )
    prepared = prepare_projector_monte_carlo(problem, plan)
    state = initialize_projector_monte_carlo(prepared, jax.random.key(17))
    result = step_projector_monte_carlo(prepared, state)
    assert int(result.status) == ProjectorMonteCarloStatus.GUIDE_FAILURE
    for old, new in zip(
        jax.tree_util.tree_leaves(state),
        jax.tree_util.tree_leaves(result.state),
        strict=True,
    ):
        if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jax.random.key_data(old), jax.random.key_data(new)
            )
        else:
            np.testing.assert_array_equal(old, new)
