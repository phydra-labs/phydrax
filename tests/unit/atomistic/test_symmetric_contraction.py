import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax.nn.atomistic._symmetric_contraction as symmetric_module
from phydrax._trainable import partition_parameters
from phydrax.ein import contract
from phydrax.nn.atomistic._symmetric_contraction import (
    StaleSymmetricContractionBinding,
    SymmetricContraction,
    SymmetricContractionBasis,
    SymmetricContractionPlan,
)
from phydrax.nn.operator.representations._irreps import (
    o3_irrep_action,
    O3IrrepBlock,
    O3IrrepLayout,
)


pytestmark = pytest.mark.strict_jax

SCALAR = O3IrrepBlock("scalar", 0, 1)
VECTOR = O3IrrepBlock("vector", 1, -1)
TENSOR = O3IrrepBlock("tensor", 2, 1)
SCALAR_VECTOR = O3IrrepLayout((SCALAR, VECTOR))
SCALAR_VECTOR_TENSOR = O3IrrepLayout((SCALAR, VECTOR, TENSOR))
SCALAR_OUTPUT = O3IrrepLayout((O3IrrepBlock("out-scalar", 0, 1),))
VECTOR_OUTPUT = O3IrrepLayout((O3IrrepBlock("out-vector", 1, -1),))
MIXED_OUTPUT = O3IrrepLayout(
    (O3IrrepBlock("out-scalar", 0, 1), O3IrrepBlock("out-vector", 1, -1))
)


def _rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0.0 else -q


def _features(seed: int, nodes: int, channels: int, size: int) -> jax.Array:
    return jnp.asarray(
        np.random.default_rng(seed).normal(size=(nodes, channels, size)),
        dtype=jnp.float64,
    )


def _contraction(
    input_layout: O3IrrepLayout,
    output_layout: O3IrrepLayout,
    correlation: int,
    *,
    species: int = 2,
    channels: int = 3,
    seed: int = 0,
) -> SymmetricContraction:
    plan = SymmetricContractionPlan.native(
        input_layout,
        output_layout,
        correlation,
        species_count=species,
        channel_count=channels,
    )
    return SymmetricContraction.initialize(plan, key=jr.key(seed))


@pytest.mark.parametrize(
    ("output", "correlation", "expected"),
    [
        # Invariants of (s, v): s and nothing else at degree one.
        pytest.param(SCALAR, 1, 1, id="scalar-degree1"),
        # s^2 and v.v.
        pytest.param(SCALAR, 2, 2, id="scalar-degree2"),
        # s^3 and s (v.v).
        pytest.param(SCALAR, 3, 2, id="scalar-degree3"),
        # v alone at degree one, s v at degree two, s^2 v and (v.v) v at degree three.
        pytest.param(VECTOR, 1, 1, id="vector-degree1"),
        pytest.param(VECTOR, 2, 1, id="vector-degree2"),
        pytest.param(VECTOR, 3, 2, id="vector-degree3"),
    ],
)
def test_native_basis_path_count_is_invariant_theory_dimension(
    output: O3IrrepBlock, correlation: int, expected: int
) -> None:
    basis = SymmetricContractionBasis.native(SCALAR_VECTOR, output, correlation)

    assert basis.path_count == expected
    assert basis.origin == "native-coupling"
    assert all(len(term) == correlation for term in basis.terms)


def test_pseudovector_output_has_no_degree_one_coupling_from_polar_vector() -> None:
    pseudovector = O3IrrepBlock("pseudovector", 1, 1)

    with pytest.raises(ValueError, match="No equivariant"):
        SymmetricContractionBasis.native(SCALAR_VECTOR, pseudovector, 1)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "improper"])
def test_native_contraction_is_o3_equivariant(improper: bool) -> None:
    module = _contraction(SCALAR_VECTOR_TENSOR, MIXED_OUTPUT, 3, channels=2)
    orthogonal = _rotation(3) * (-1.0 if improper else 1.0)
    action_in = np.asarray(SCALAR_VECTOR_TENSOR.representation_matrix(orthogonal))
    action_out = np.asarray(MIXED_OUTPUT.representation_matrix(orthogonal))
    features = _features(4, 5, 2, SCALAR_VECTOR_TENSOR.packed_size)
    species = jnp.asarray([0, 1, 1, 0, 1], dtype=jnp.int32)

    rotated = module(contract("ij,ncj->nci", jnp.asarray(action_in), features), species)
    expected = contract("ij,ncj->nci", jnp.asarray(action_out), module(features, species))

    np.testing.assert_allclose(rotated, expected, rtol=1e-10, atol=1e-10)


def test_scalar_degree_two_contraction_spans_independent_invariants() -> None:
    module = _contraction(SCALAR_VECTOR, SCALAR_OUTPUT, 2, species=1, channels=1)
    species = jnp.zeros((3,), dtype=jnp.int32)
    # Rows: (s, v) with known invariants s^2 and |v|^2.
    features = jnp.asarray(
        [[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.3, -0.4, 1.2]]],
        dtype=jnp.float64,
    )
    output = np.asarray(module(features, species))[:, 0, 0]
    # The output is a s^2 + b |v|^2; |v|^2 = 1.69 for the third row.
    np.testing.assert_allclose(output[2], 1.69 * output[1], rtol=1e-12)


def test_weight_directional_derivative_is_the_contraction_with_the_direction() -> None:
    module = _contraction(SCALAR_VECTOR, MIXED_OUTPUT, 3)
    features = _features(1, 4, 3, SCALAR_VECTOR.packed_size)
    species = jnp.asarray([1, 0, 1, 1], dtype=jnp.int32)
    direction = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(
            np.random.default_rng(leaf.size).normal(size=leaf.shape), dtype=leaf.dtype
        ),
        module.weights,
    )

    def evaluate(weights: tuple[tuple[jax.Array, ...], ...]) -> jax.Array:
        return eqx.tree_at(lambda m: m.weights, module, weights)(features, species)

    _, tangent = jax.jvp(evaluate, (module.weights,), (direction,))
    linear = SymmetricContraction(module.plan, direction)(features, species)

    np.testing.assert_allclose(tangent, linear, rtol=1e-12, atol=1e-12)


def test_original_weights_are_the_only_parameters_and_receive_gradients() -> None:
    module = _contraction(SCALAR_VECTOR, SCALAR_OUTPUT, 2)
    features = _features(2, 3, 3, SCALAR_VECTOR.packed_size)
    species = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    parameters = partition_parameters(module)[0]
    leaves = jax.tree_util.tree_leaves(parameters)

    assert len(leaves) == sum(len(block) for block in module.weights)
    gradient = eqx.filter_grad(lambda m: jnp.sum(m(features, species) ** 2))(module)
    for block in gradient.weights:
        for weight in block:
            assert weight.shape[0] == 2
            assert bool(jnp.any(weight != 0.0))


def test_species_select_distinct_weights_and_agnostic_product_uses_one_row() -> None:
    module = _contraction(SCALAR_VECTOR, SCALAR_OUTPUT, 2, species=2)
    features = jnp.broadcast_to(_features(5, 1, 3, 4), (2, 3, 4))
    by_species = module(features, jnp.asarray([0, 1], dtype=jnp.int32))
    agnostic = _contraction(SCALAR_VECTOR, SCALAR_OUTPUT, 2, species=1)

    assert not np.allclose(by_species[0], by_species[1])
    assert agnostic(features, jnp.zeros((2,), dtype=jnp.int32)).shape == (2, 3, 1)


def test_node_tiles_match_untiled_evaluation() -> None:
    module = _contraction(SCALAR_VECTOR_TENSOR, MIXED_OUTPUT, 2)
    features = _features(6, 7, 3, SCALAR_VECTOR_TENSOR.packed_size)
    species = jnp.asarray([0, 1, 0, 1, 1, 0, 0], dtype=jnp.int32)

    np.testing.assert_allclose(
        module(features, species, node_tile=3), module(features, species), rtol=1e-13
    )


def _source_vector_frame() -> np.ndarray:
    # A source component basis that differs from the native l=1 basis by an
    # orthogonal map: input and output vectors must be transformed together.
    return _rotation(11)


def _source_scalar_vector_u() -> np.ndarray:
    """Independent invariant source U for (s, v) -> scalar at correlation two.

    Paths: s*s and v.v / sqrt(3); the dot product is invariant in every
    orthonormal vector basis, so this U is equivariant in any source frame.
    """
    tensor = np.zeros((1, 4, 4, 2), dtype=np.float64)
    tensor[0, 0, 0, 0] = 1.0
    tensor[0, 1:, 1:, 1] = np.eye(3) / math.sqrt(3.0)
    return tensor


def _source_scalar_vector_to_vector_u() -> np.ndarray:
    """Source U for (s, v) -> vector at correlation two: path s v (both orders)."""
    tensor = np.zeros((3, 4, 4, 1), dtype=np.float64)
    for component in range(3):
        tensor[component, 0, 1 + component, 0] = 0.5
        tensor[component, 1 + component, 0, 0] = 0.5
    return tensor


def _block_transform(frame: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[1:, 1:] = frame
    return transform


@pytest.mark.parametrize(
    ("tensor", "output_block", "output_transform"),
    [
        pytest.param(_source_scalar_vector_u(), SCALAR, np.eye(1), id="scalar"),
        pytest.param(
            _source_scalar_vector_to_vector_u(),
            VECTOR,
            _source_vector_frame(),
            id="vector",
        ),
    ],
)
def test_imported_basis_reproduces_source_contraction_in_source_frame(
    tensor: np.ndarray, output_block: O3IrrepBlock, output_transform: np.ndarray
) -> None:
    input_transform = _block_transform(_source_vector_frame())
    basis = SymmetricContractionBasis.from_source_tensor(
        tensor,
        SCALAR_VECTOR,
        output_block,
        2,
        source_id="test-provider-u",
        input_transform=input_transform,
        output_transform=output_transform,
        maximum_entries=tensor.size,
    )
    output_layout = O3IrrepLayout(
        (O3IrrepBlock("out", output_block.degree, output_block.parity),)
    )
    first = SymmetricContractionBasis.native(SCALAR_VECTOR, output_layout.blocks[0], 1)
    plan = SymmetricContractionPlan(
        SCALAR_VECTOR, output_layout, ((first, basis),), species_count=1, channel_count=2
    )
    weights = (
        (
            jnp.zeros((1, first.path_count, 2), dtype=jnp.float64),
            jnp.asarray(
                np.random.default_rng(7).normal(size=(1, tensor.shape[-1], 2)),
                dtype=jnp.float64,
            ),
        ),
    )
    module = SymmetricContraction(plan, weights)
    source_features = np.random.default_rng(8).normal(size=(3, 2, 4))
    native_features = contract("ij,ncj->nci", input_transform, source_features)

    # Independent dense source evaluation: B[c, M] = sum W[k, c] U[M, i, j, k] x_i x_j.
    source_output = contract(
        "mijk,kc,nci,ncj->ncm",
        tensor,
        np.asarray(weights[0][1][0]),
        source_features,
        source_features,
    )
    expected = contract("om,ncm->nco", output_transform, source_output)
    actual = module(jnp.asarray(native_features), jnp.zeros((3,), dtype=jnp.int32))

    assert basis.origin == "imported-source"
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_imported_tensor_is_charged_before_conversion() -> None:
    huge = np.broadcast_to(np.zeros((), dtype=np.float64), (1, 4, 4, 4, 4, 4, 4, 2))

    with pytest.raises(ValueError, match="maximum_entries"):
        SymmetricContractionBasis.from_source_tensor(
            huge,
            SCALAR_VECTOR,
            SCALAR,
            6,
            source_id="huge",
            input_transform=np.eye(4),
            output_transform=np.eye(1),
            maximum_entries=1024,
        )


def test_imported_complex_tensor_is_refused_not_truncated() -> None:
    real = _source_scalar_vector_u()
    complex_tensor = real.astype(np.complex128)
    complex_tensor.flat[np.flatnonzero(real)[0]] += 9.0j

    with pytest.raises(TypeError, match="complex128"):
        SymmetricContractionBasis.from_source_tensor(
            complex_tensor,
            SCALAR_VECTOR,
            SCALAR,
            2,
            source_id="complex-provider-u",
            input_transform=np.eye(4),
            output_transform=np.eye(1),
            maximum_entries=real.size,
        )


def test_native_projection_budget_is_charged_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 4097 scalar blocks fit maximum_terms, but the order-one scalar projection
    # has 4097 x 1 x 4097 entries, one row past the dense-entry budget.
    layout = O3IrrepLayout(tuple(O3IrrepBlock(f"s{i}", 0, 1) for i in range(4097)))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("native coupling families generated before admission")

    monkeypatch.setattr(symmetric_module, "_native_coupling_span", forbidden)
    monkeypatch.setattr(symmetric_module, "_independent_families", forbidden)
    with pytest.raises(ValueError, match="16785409 projection entries"):
        SymmetricContractionBasis.native(layout, SCALAR, 1)


def test_mathematically_absent_irrep_is_not_a_rounding_path() -> None:
    # Sym^3(0e + 1o) = 0e + 1o + (0e + 2e) + (1o + 3o) has no pseudovector,
    # although the iterated coupling leaves rounding residue in that channel.
    with pytest.raises(ValueError, match="No equivariant"):
        SymmetricContractionBasis.native(
            SCALAR_VECTOR, O3IrrepBlock("pseudovector", 1, 1), 3
        )


def test_import_refuses_block_mixing_transform() -> None:
    mixing = np.eye(4)
    mixing[[0, 1]] = mixing[[1, 0]]

    with pytest.raises(ValueError, match="mix distinct irrep blocks"):
        SymmetricContractionBasis.from_source_tensor(
            _source_scalar_vector_u(),
            SCALAR_VECTOR,
            SCALAR,
            2,
            source_id="mixing",
            input_transform=mixing,
            output_transform=np.eye(1),
            maximum_entries=64,
        )


def test_import_refuses_basis_that_is_not_equivariant_in_declared_frame() -> None:
    # A wrong declared output frame turns the s v coupling non-equivariant.
    wrong = np.diag([1.0, 1.0, -1.0])

    with pytest.raises(ValueError, match="not equivariant"):
        SymmetricContractionBasis.from_source_tensor(
            _source_scalar_vector_to_vector_u(),
            SCALAR_VECTOR,
            VECTOR,
            2,
            source_id="wrong-frame",
            input_transform=np.eye(4),
            output_transform=wrong,
            maximum_entries=64,
        )


def test_merged_binding_matches_and_refuses_stale_weights() -> None:
    module = _contraction(SCALAR_VECTOR_TENSOR, MIXED_OUTPUT, 3)
    features = _features(9, 4, 3, SCALAR_VECTOR_TENSOR.packed_size)
    species = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    merged = module.merged()

    merged.require_current(module)
    np.testing.assert_allclose(
        merged(features, species), module(features, species), rtol=1e-11, atol=1e-12
    )
    assert jax.tree_util.tree_leaves(partition_parameters(merged)[0]) == []

    updated = eqx.tree_at(
        lambda m: m.weights[0][0], module, module.weights[0][0] + 1.0e-3
    )
    with pytest.raises(StaleSymmetricContractionBinding):
        merged.require_current(updated)


def test_construction_refusals() -> None:
    plan = SymmetricContractionPlan.native(
        SCALAR_VECTOR, SCALAR_OUTPUT, 2, species_count=2, channel_count=3
    )
    with pytest.raises(ValueError, match="weights must have shapes"):
        SymmetricContraction(plan, ((jnp.zeros((2, 1, 3)), jnp.zeros((2, 1, 3))),))
    with pytest.raises(ValueError, match="multiplicity one"):
        SymmetricContractionBasis.native(
            O3IrrepLayout((O3IrrepBlock("pair", 0, 1, multiplicity=2),)), SCALAR, 2
        )
    with pytest.raises(ValueError, match="maximum_terms"):
        SymmetricContractionBasis.native(
            SCALAR_VECTOR_TENSOR, SCALAR, 4, maximum_terms=10
        )


def _imported_contraction() -> SymmetricContraction:
    input_transform = _block_transform(_source_vector_frame())
    basis = SymmetricContractionBasis.from_source_tensor(
        _source_scalar_vector_u(),
        SCALAR_VECTOR,
        SCALAR_OUTPUT.blocks[0],
        2,
        source_id="test-provider-u",
        input_transform=input_transform,
        output_transform=np.eye(1),
        maximum_entries=32,
    )
    first = SymmetricContractionBasis.native(SCALAR_VECTOR, SCALAR_OUTPUT.blocks[0], 1)
    plan = SymmetricContractionPlan(
        SCALAR_VECTOR, SCALAR_OUTPUT, ((first, basis),), species_count=2, channel_count=3
    )
    return SymmetricContraction.initialize(plan, key=jr.key(4))


type _Where = Callable[[SymmetricContraction], jax.Array]
type _Replace = Callable[[jax.Array], jax.Array]


def _set(
    module: SymmetricContraction, where: _Where, replace: _Replace
) -> SymmetricContraction:
    return eqx.tree_at(where, module, replace(where(module)))


def _with_entry(values: jax.Array, value: float) -> jax.Array:
    return values.at[0].set(value)


_NATIVE_TAMPERS = {
    # Each keeps every recorded basis/plan identity and the array shapes.
    "basis-scaled": (lambda m: m.plan.bases[1][1].coefficients, lambda c: c * 2.0),
    "basis-entry": (
        lambda m: m.plan.bases[1][1].coefficients,
        lambda c: _with_entry(c, c[0] + 1.0e-3),
    ),
    "basis-nan": (
        lambda m: m.plan.bases[1][1].coefficients,
        lambda c: _with_entry(c, jnp.nan),
    ),
    "basis-rerouted": (
        lambda m: m.plan.bases[1][1].relation.target_indices,
        lambda t: t[::-1],
    ),
    "product-graph": (lambda m: m.plan.product_graph.term_node, lambda t: jnp.roll(t, 1)),
    "term-map": (lambda m: m.plan.term_maps[0][1], lambda t: t[::-1]),
}


@pytest.mark.parametrize("tamper", sorted(_NATIVE_TAMPERS))
def test_restored_native_plan_revalidates_fixed_data_not_recorded_ids(
    tamper: str,
) -> None:
    module = _contraction(SCALAR_VECTOR, MIXED_OUTPUT, 2)
    module.validate()
    where, replace = _NATIVE_TAMPERS[tamper]
    changed = _set(module, where, replace)
    assert changed.plan.plan_id == module.plan.plan_id

    with pytest.raises(ValueError):
        changed.validate()


@pytest.mark.parametrize(
    "replace",
    [lambda c: c * 2.0, lambda c: _with_entry(c, jnp.nan)],
    ids=["equivariant-rescale", "nan"],
)
def test_imported_basis_is_authoritative_and_refuses_changed_values(
    replace: _Replace,
) -> None:
    module = _imported_contraction()
    module.validate()
    changed = _set(module, lambda m: m.plan.bases[0][1].coefficients, replace)

    with pytest.raises(ValueError):
        changed.validate()


_MERGED_TAMPERS = {
    "coefficient-finite": (
        lambda m: m.coefficients[1],
        lambda c: c.at[0, 0, 0, 0].add(1.0e-6),
    ),
    "coefficient-nan": (
        lambda m: m.coefficients[1],
        lambda c: c.at[0, 0, 0, 0].set(jnp.nan),
    ),
    "term-map": (lambda m: m.block_term_maps[0], lambda t: t[::-1]),
    "duplicated-basis": (lambda m: m.plan.bases[0][0].coefficients, lambda c: c * 2.0),
}


@pytest.mark.parametrize("tamper", sorted(_MERGED_TAMPERS))
def test_merged_validation_recomputes_every_executed_field(tamper: str) -> None:
    module = _contraction(SCALAR_VECTOR, MIXED_OUTPUT, 2)
    merged = module.merged()
    merged.validate(module)
    where, replace = _MERGED_TAMPERS[tamper]
    changed = eqx.tree_at(where, merged, replace(where(merged)))
    changed.require_current(module)

    with pytest.raises(StaleSymmetricContractionBinding):
        changed.validate(module)


def test_vector_action_is_consistent_with_layout_representation() -> None:
    orthogonal = -_rotation(5)
    block = np.asarray(SCALAR_VECTOR.representation_matrix(orthogonal))[1:, 1:]

    np.testing.assert_allclose(
        block, np.asarray(o3_irrep_action(orthogonal, 1, -1)), rtol=1e-12, atol=1e-12
    )
