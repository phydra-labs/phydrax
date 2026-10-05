from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from opt_einsum import contract

from phydrax.nn.operator.layers import (
    O3TensorProduct,
    O3TensorProductPath,
    O3TensorProductPlan,
)
from phydrax.nn.operator.representations import (
    O3IrrepBlock,
    O3IrrepLayout,
    O3Parity,
    O3Representation,
)


def _rotation(key: Any) -> Any:
    matrix = jr.normal(key, (3, 3), dtype=jnp.float64)
    orthogonal, triangular = jnp.linalg.qr(matrix)
    signs = jnp.where(jnp.diag(triangular) < 0.0, -1.0, 1.0)
    orthogonal = orthogonal * signs[None, :]
    return orthogonal.at[:, 0].multiply(jnp.linalg.det(orthogonal))


def test_known_scalar_and_vector_products_have_component_normalization() -> None:
    scalar = O3Representation(scalars=1)
    vector = O3Representation(vectors=1)
    scalar_product = O3TensorProduct(
        O3TensorProductPlan(scalar, scalar, scalar), internal_weights=False
    )
    np.testing.assert_allclose(
        scalar_product(jnp.asarray([3.0]), jnp.asarray([4.0]), jnp.asarray([2.0])),
        24.0,
    )

    scalar_vector = O3TensorProduct(
        O3TensorProductPlan(scalar, vector, vector), internal_weights=False
    )
    observed = scalar_vector(
        jnp.asarray([2.0]),
        jnp.asarray([1.0, -2.0, 3.0]),
        jnp.ones((1,)),
    )
    np.testing.assert_allclose(observed, [2.0, -4.0, 6.0])


def test_vector_vector_decomposes_into_known_zero_one_two_maps() -> None:
    vector = O3Representation(vectors=1)
    output = O3Representation(scalars=1, pseudovectors=1, tensors=1)
    plan = O3TensorProductPlan(vector, vector, output)
    product = O3TensorProduct(plan, internal_weights=False)
    left = jnp.asarray([0.3, -0.5, 0.8], dtype=jnp.float64)
    right = jnp.asarray([-0.2, 0.7, 0.4], dtype=jnp.float64)
    features = output.split(product(left, right, jnp.ones((3,), dtype=left.dtype)))
    expected_scalar = contract("i,i->", left, right) / jnp.sqrt(3.0)
    expected_axial = jnp.cross(left, right) / jnp.sqrt(2.0)
    outer = 0.5 * (contract("i,j->ij", left, right) + contract("i,j->ij", right, left))
    expected_tensor = outer - jnp.trace(outer) * jnp.eye(3) / 3.0
    np.testing.assert_allclose(features.scalars[0], expected_scalar, rtol=1e-13)
    np.testing.assert_allclose(features.pseudovectors[0], expected_axial, rtol=1e-13)
    np.testing.assert_allclose(features.tensors[0], expected_tensor, rtol=1e-13)


def test_all_low_degree_paths_obey_inversion_parity_and_random_so3_equivariance() -> None:
    left_representation = O3Representation(vectors=1, pseudovectors=1)
    right_representation = O3Representation(vectors=1)
    output_representation = O3Representation(
        scalars=1,
        pseudoscalars=1,
        vectors=1,
        pseudovectors=1,
        tensors=1,
        pseudotensors=1,
    )
    plan = O3TensorProductPlan(
        left_representation, right_representation, output_representation
    )
    product = O3TensorProduct(plan, internal_weights=False, dtype=jnp.float64)
    left = jr.normal(jr.key(2), (left_representation.packed_size,), dtype=jnp.float64)
    right = jr.normal(jr.key(3), (right_representation.packed_size,), dtype=jnp.float64)
    weights = jr.normal(jr.key(4), (plan.parameter_count,), dtype=jnp.float64)
    reference = product(left, right, weights)

    inversion = -jnp.eye(3, dtype=jnp.float64)
    inverted = product(
        left_representation.transform(left, inversion),
        right_representation.transform(right, inversion),
        weights,
    )
    np.testing.assert_allclose(
        inverted,
        output_representation.transform(reference, inversion),
        rtol=2e-13,
        atol=2e-13,
    )

    rotation = _rotation(jr.key(5))
    rotated = product(
        left_representation.transform(left, rotation),
        right_representation.transform(right, rotation),
        weights,
    )
    np.testing.assert_allclose(
        rotated,
        output_representation.transform(reference, rotation),
        rtol=3e-13,
        atol=3e-13,
    )


def test_successive_rotations_batching_and_gradients_preserve_layout() -> None:
    representation = O3Representation(scalars=1, vectors=1, pseudovectors=1, tensors=1)
    edge = O3Representation(scalars=1, vectors=1, tensors=1)
    plan = O3TensorProductPlan(representation, edge, representation)
    product = O3TensorProduct(plan, internal_weights=False, dtype=jnp.float64)
    left = jr.normal(jr.key(10), (4, representation.packed_size), dtype=jnp.float64)
    right = jr.normal(jr.key(11), (4, edge.packed_size), dtype=jnp.float64)
    weights = jr.normal(jr.key(12), (4, plan.parameter_count), dtype=jnp.float64)
    observed = product(left, right, weights)
    assert observed.shape == (4, representation.packed_size)
    gradient = jax.grad(lambda value: jnp.sum(product(value, right, weights) ** 2))(left)
    assert gradient.shape == left.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))

    first = _rotation(jr.key(13))
    second = _rotation(jr.key(14))
    combined = second @ first
    direct = representation.transform(observed, combined)
    successive = representation.transform(
        representation.transform(observed, first), second
    )
    np.testing.assert_allclose(direct, successive, rtol=4e-13, atol=4e-13)


def test_plan_counts_identity_and_resource_rejection_are_resolved_before_layer() -> None:
    left = O3Representation(vectors=2)
    right = O3Representation(vectors=3)
    output = O3Representation(scalars=4, pseudovectors=5, tensors=6)
    plan = O3TensorProductPlan(left, right, output)
    assert plan.path_count == 3
    assert plan.parameter_count == 90
    # ty: ignore[unresolved-attribute]
    assert O3TensorProduct(plan, key=jr.key(20)).weight.size == plan.parameter_count
    assert plan.resource_evidence["parameter_count"] == 90
    assert plan.content_id == plan.plan_id
    assert plan.plan_id == O3TensorProductPlan(left, right, output).plan_id
    with pytest.raises(ValueError, match="parameters"):
        O3TensorProductPlan(left, right, output, maximum_parameters=89)
    with pytest.raises(ValueError, match="no legal"):
        O3TensorProductPlan(
            O3Representation(scalars=1),
            O3Representation(scalars=1),
            O3Representation(vectors=1),
        )
    with pytest.raises(ValueError, match="equal left and output multiplicities"):
        O3TensorProductPlan(left, right, output, connection_mode="uvu")


def test_product_rejects_mismatched_layouts_and_path_weight_axes() -> None:
    scalar = O3Representation(scalars=1)
    plan = O3TensorProductPlan(scalar, scalar, scalar)
    product = O3TensorProduct(plan, internal_weights=False)
    with pytest.raises(ValueError, match="Left values"):
        product(jnp.ones((2,)), jnp.ones((1,)), jnp.ones((1,)))
    with pytest.raises(ValueError, match="leading axes"):
        product(jnp.ones((2, 1)), jnp.ones((3, 1)), jnp.ones((1,)))
    with pytest.raises(ValueError, match="path weights"):
        product(jnp.ones((2, 1)), jnp.ones((2, 1)), jnp.ones((3, 1)))


def _harmonic_parity(degree: int) -> O3Parity:
    return 1 if degree % 2 == 0 else -1


def _irreps(*blocks: tuple[str, int, O3Parity, int]) -> O3IrrepLayout:
    return O3IrrepLayout(
        [
            O3IrrepBlock(name, degree, parity, multiplicity=multiplicity)
            for name, degree, parity, multiplicity in blocks
        ]
    )


def _improper(key: Any) -> Any:
    return -_rotation(key)


def _harmonic_message_plan(
    channels: int, node_degree: int, edge_degree: int
) -> O3TensorProductPlan:
    """MACE-style uvu messages: one output block per legal path, signed scales."""
    node = _irreps(
        *(
            (f"h{degree}", degree, _harmonic_parity(degree), channels)
            for degree in range(node_degree + 1)
        )
    )
    edge = _irreps(
        *(
            (f"y{degree}", degree, _harmonic_parity(degree), 1)
            for degree in range(edge_degree + 1)
        )
    )
    outputs = []
    paths = []
    for left in node.blocks:
        for right in edge.blocks:
            for degree in range(
                abs(left.degree - right.degree), left.degree + right.degree + 1
            ):
                name = f"m{left.degree}{right.degree}{degree}"
                outputs.append(
                    O3IrrepBlock(
                        name, degree, left.parity * right.parity, multiplicity=channels
                    )
                )
                paths.append(
                    O3TensorProductPath(
                        left.name,
                        right.name,
                        name,
                        connection_mode="uvu",
                        path_scale=-0.5 if degree % 2 else 2.0,
                    )
                )
    return O3TensorProductPlan(node, edge, O3IrrepLayout(outputs), paths=paths)


def test_general_degree_one_products_match_independent_dot_cross_and_traceless_maps() -> (
    None
):
    vector = _irreps(("a", 1, -1, 1))
    output = _irreps(("s", 0, 1, 1), ("p", 1, 1, 1), ("t", 2, 1, 1))
    plan = O3TensorProductPlan(vector, vector, output)
    product = O3TensorProduct(plan, internal_weights=False)
    a = jnp.asarray([0.3, -0.5, 0.8], dtype=jnp.float64)
    b = jnp.asarray([-0.2, 0.7, 0.4], dtype=jnp.float64)
    to_real = {
        degree: jnp.asarray(O3Representation.real_harmonic_basis(degree))
        for degree in (1, 2)
    }
    scalar, axial, tensor = output.split(
        product(to_real[1] @ a, to_real[1] @ b, jnp.ones((3,), dtype=jnp.float64))
    )
    outer = 0.5 * (contract("i,j->ij", a, b) + contract("i,j->ij", b, a))
    traceless = outer - jnp.trace(outer) * jnp.eye(3) / 3.0
    # Orthonormal symmetric-traceless coordinates in the Cartesian packing order.
    stf = jnp.asarray(
        [
            (traceless[0, 0] - traceless[1, 1]) / jnp.sqrt(2.0),
            (traceless[0, 0] + traceless[1, 1] - 2.0 * traceless[2, 2]) / jnp.sqrt(6.0),
            jnp.sqrt(2.0) * traceless[0, 1],
            jnp.sqrt(2.0) * traceless[0, 2],
            jnp.sqrt(2.0) * traceless[1, 2],
        ]
    )
    # Condon--Shortley coupling: (1, 1) -> 0 is minus the normalized dot product.
    np.testing.assert_allclose(scalar[0, 0], -jnp.dot(a, b) / jnp.sqrt(3.0), rtol=1e-13)
    np.testing.assert_allclose(
        to_real[1].T @ axial[0], jnp.cross(a, b) / jnp.sqrt(2.0), rtol=1e-13
    )
    np.testing.assert_allclose(to_real[2].T @ tensor[0], stf, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize(
    ("node_degree", "edge_degree"), [(3, 3), (2, 4)], ids=["l3xl3", "l2xl4"]
)
def test_high_degree_uvu_messages_are_rotation_and_reflection_equivariant(
    node_degree: int, edge_degree: int
) -> None:
    plan = _harmonic_message_plan(2, node_degree, edge_degree)
    left_layout = plan.left_representation
    right_layout = plan.right_representation
    output_layout = plan.output_representation
    assert isinstance(left_layout, O3IrrepLayout)
    assert isinstance(right_layout, O3IrrepLayout)
    assert isinstance(output_layout, O3IrrepLayout)
    product = O3TensorProduct(plan, internal_weights=False)
    left = jr.normal(jr.key(30), (3, left_layout.packed_size), dtype=jnp.float64)
    right = jr.normal(jr.key(31), (3, right_layout.packed_size), dtype=jnp.float64)
    weights = jr.normal(jr.key(32), (3, plan.parameter_count), dtype=jnp.float64)
    reference = product(left, right, weights)
    for frame in (_rotation(jr.key(33)), _improper(jr.key(34)), -jnp.eye(3)):
        np.testing.assert_allclose(
            product(
                left_layout.transform(left, frame),
                right_layout.transform(right, frame),
                weights,
            ),
            output_layout.transform(reference, frame),
            rtol=1e-12,
            atol=1e-12,
        )


def test_uvu_weights_scale_left_channels_and_unweighted_paths_sum_right_channels() -> (
    None
):
    left = _irreps(("h", 3, -1, 2))
    right = _irreps(("y", 2, 1, 3))
    output = _irreps(("o", 3, -1, 2))
    weighted = O3TensorProductPlan(left, right, output, connection_mode="uvu")
    assert weighted.parameter_count == 6
    assert weighted.path_weight_layout == ((0, (2, 3)),)
    unweighted = O3TensorProductPlan(
        left,
        right,
        output,
        paths=[O3TensorProductPath("h", "y", "o", connection_mode="uvu", weighted=False)],
    )
    assert unweighted.parameter_count == 0
    assert unweighted.path_weight_layout == ((0, ()),)
    x = jr.normal(jr.key(40), (left.packed_size,), dtype=jnp.float64)
    y = jr.normal(jr.key(41), (right.packed_size,), dtype=jnp.float64)
    summed = O3TensorProduct(unweighted)(x, y)
    ones = O3TensorProduct(weighted, internal_weights=False)(x, y, jnp.ones((6,)))
    np.testing.assert_allclose(summed, ones, rtol=1e-14)
    # Weight (u, v) only reaches output channel u.
    single = jnp.zeros((6,), dtype=jnp.float64).at[4].set(1.0)
    channels = output.split(
        O3TensorProduct(weighted, internal_weights=False)(x, y, single)
    )[0]
    np.testing.assert_array_equal(channels[0], 0.0)
    assert bool(jnp.any(channels[1] != 0.0))


def test_path_scales_multiply_their_own_path_only() -> None:
    left = _irreps(("a", 1, -1, 1))
    output = _irreps(("s", 0, 1, 1), ("t", 2, 1, 1))
    base = O3TensorProductPlan(left, left, output)
    scaled = O3TensorProductPlan(
        left,
        left,
        output,
        paths=[
            O3TensorProductPath("a", "a", "s", path_scale=-3.0),
            O3TensorProductPath("a", "a", "t"),
        ],
    )
    x = jr.normal(jr.key(50), (3,), dtype=jnp.float64)
    y = jr.normal(jr.key(51), (3,), dtype=jnp.float64)
    weights = jnp.asarray([0.7, -1.1], dtype=jnp.float64)
    expected = O3TensorProduct(base, internal_weights=False)(x, y, weights)
    observed = O3TensorProduct(scaled, internal_weights=False)(x, y, weights)
    np.testing.assert_allclose(observed[:1], -3.0 * expected[:1], rtol=1e-14)
    np.testing.assert_allclose(observed[1:], expected[1:], rtol=1e-14)
    assert scaled.plan_id != base.plan_id


def test_repeated_irrep_blocks_keep_distinct_path_identity() -> None:
    left = _irreps(("a", 1, -1, 2))
    right = _irreps(("b", 1, -1, 1))
    output = _irreps(("first", 1, 1, 2), ("second", 1, 1, 2))
    plan = O3TensorProductPlan(
        left,
        right,
        output,
        paths=[
            O3TensorProductPath("a", "b", "first", connection_mode="uvu"),
            O3TensorProductPath(
                "a", "b", "second", connection_mode="uvu", weighted=False
            ),
        ],
    )
    product = O3TensorProduct(plan, internal_weights=False)
    x = jr.normal(jr.key(60), (6,), dtype=jnp.float64)
    y = jr.normal(jr.key(61), (3,), dtype=jnp.float64)
    first, second = output.split(product(x, y, jnp.zeros((2,), dtype=jnp.float64)))
    np.testing.assert_array_equal(first, 0.0)
    assert bool(jnp.all(jnp.linalg.norm(second, axis=-1) > 0.0))


@pytest.mark.parametrize(
    ("path", "output", "message"),
    [
        (
            O3TensorProductPath("a", "a", "o"),
            ("o", 3, 1, 1),
            "degree triangle or parity",
        ),
        (
            O3TensorProductPath("a", "a", "o"),
            ("o", 1, -1, 1),
            "degree triangle or parity",
        ),
        (
            O3TensorProductPath("a", "a", "o", connection_mode="uvu"),
            ("o", 2, 1, 3),
            "equal left and output multiplicities",
        ),
        (
            O3TensorProductPath("a", "missing", "o"),
            ("o", 2, 1, 1),
            "no block named 'missing'",
        ),
    ],
    ids=["degree", "parity", "uvu-multiplicity", "unknown-block"],
)
def test_illegal_paths_are_refused_at_planning(
    path: O3TensorProductPath, output: tuple[str, int, O3Parity, int], message: str
) -> None:
    left = _irreps(("a", 1, -1, 2))
    with pytest.raises(ValueError, match=message):
        O3TensorProductPlan(left, left, _irreps(output), paths=[path])


def test_path_declarations_and_layout_kinds_are_validated() -> None:
    with pytest.raises(ValueError, match="'uvw' paths map multiplicities"):
        O3TensorProductPath("a", "b", "c", weighted=False)
    with pytest.raises(ValueError, match="nonzero"):
        O3TensorProductPath("a", "b", "c", path_scale=0.0)
    left = _irreps(("a", 1, -1, 1))
    path = O3TensorProductPath("a", "a", "o")
    output = _irreps(("o", 0, 1, 1))
    with pytest.raises(ValueError, match="redundant parameterization"):
        O3TensorProductPlan(left, left, output, paths=[path, path])
    with pytest.raises(ValueError, match="declare their own connection modes"):
        O3TensorProductPlan(left, left, output, paths=[path], connection_mode="uvw")
    with pytest.raises(ValueError, match="component bases are not interchangeable"):
        O3TensorProductPlan(left, O3Representation(vectors=1), output)


def _budgeted_plan(
    plan: O3TensorProductPlan, evidence_key: str, value: int
) -> O3TensorProductPlan:
    """Rebuild one plan with exactly one resource limit replaced."""
    limits = {
        "path_count": plan.maximum_paths,
        "parameter_count": plan.maximum_parameters,
        "multiply_add_count": plan.maximum_multiply_adds,
        "coefficient_count": plan.maximum_coefficients,
        "working_set": plan.maximum_working_set,
    }
    limits[evidence_key] = value
    return O3TensorProductPlan(
        plan.left_representation,
        plan.right_representation,
        plan.output_representation,
        paths=plan.paths,
        maximum_paths=limits["path_count"],
        maximum_parameters=limits["parameter_count"],
        maximum_multiply_adds=limits["multiply_add_count"],
        maximum_coefficients=limits["coefficient_count"],
        maximum_working_set=limits["working_set"],
    )


@pytest.mark.parametrize(
    ("evidence_key", "label"),
    [
        ("path_count", "paths"),
        ("parameter_count", "parameters"),
        ("multiply_add_count", "multiply-adds"),
        ("coefficient_count", "coefficients"),
        ("working_set", "working-set"),
    ],
)
def test_resource_limits_refuse_before_coefficient_allocation(
    evidence_key: str, label: str
) -> None:
    plan = _harmonic_message_plan(4, 3, 3)
    evidence = plan.resource_evidence
    exact = _budgeted_plan(plan, evidence_key, evidence[evidence_key])
    assert exact.resource_evidence == evidence
    with pytest.raises(ValueError, match=label):
        _budgeted_plan(plan, evidence_key, evidence[evidence_key] - 1)


def test_automatic_path_enumeration_stops_at_the_path_budget() -> None:
    layout = _harmonic_message_plan(1, 3, 3).left_representation
    with pytest.raises(ValueError, match="more than 2 paths"):
        O3TensorProductPlan(layout, layout, layout, maximum_paths=2)


def test_shared_internal_weights_equal_broadcast_external_weights() -> None:
    plan = _harmonic_message_plan(2, 2, 2)
    internal = O3TensorProduct(plan, key=jr.key(70))
    external = O3TensorProduct(plan, internal_weights=False)
    assert internal.weight is not None
    left = jr.normal(
        jr.key(71), (4, plan.left_representation.packed_size), dtype=jnp.float64
    )
    right = jr.normal(
        jr.key(72), (4, plan.right_representation.packed_size), dtype=jnp.float64
    )
    shared = internal(left, right)
    np.testing.assert_array_equal(shared, external(left, right, internal.weight))
    per_sample = jr.normal(jr.key(73), (4, plan.parameter_count), dtype=jnp.float64)
    batched = external(left, right, per_sample)
    for row in range(4):
        np.testing.assert_allclose(
            batched[row],
            external(left[row], right[row], per_sample[row]),
            rtol=1e-13,
            atol=1e-14,
        )
    with pytest.raises(ValueError, match="cannot override"):
        internal(left, right, per_sample)


def test_general_product_jvp_vjp_are_dual_and_match_finite_differences() -> None:
    plan = _harmonic_message_plan(2, 3, 3)
    product = O3TensorProduct(plan, internal_weights=False)
    left = jr.normal(
        jr.key(80), (plan.left_representation.packed_size,), dtype=jnp.float64
    )
    right = jr.normal(
        jr.key(81), (plan.right_representation.packed_size,), dtype=jnp.float64
    )
    weights = jr.normal(jr.key(82), (plan.parameter_count,), dtype=jnp.float64)
    primals = (left, right, weights)
    tangents = tuple(
        jr.normal(jr.key(83 + index), value.shape, dtype=jnp.float64)
        for index, value in enumerate(primals)
    )
    cotangent = jr.normal(
        jr.key(90), (plan.output_representation.packed_size,), dtype=jnp.float64
    )
    value, tangent_out = jax.jvp(product, primals, tangents)
    _, pullback = jax.vjp(product, *primals)
    cotangents = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.dot(cotangent, tangent_out),
        sum(jnp.dot(c, t) for c, t in zip(cotangents, tangents, strict=True)),
        rtol=1e-12,
    )
    step = 1e-6
    shifted = tuple(p + step * t for p, t in zip(primals, tangents, strict=True))
    backward = tuple(p - step * t for p, t in zip(primals, tangents, strict=True))
    np.testing.assert_allclose(
        (product(*shifted) - product(*backward)) / (2.0 * step),
        tangent_out,
        rtol=1e-7,
        atol=1e-7,
    )
    assert value.shape == (plan.output_representation.packed_size,)


def test_imported_coefficients_are_tolerance_checked_bound_and_identified() -> None:
    plan = _harmonic_message_plan(2, 2, 2)
    native = O3TensorProduct(plan, internal_weights=False)
    assert native.coefficient_origin == "native"
    # Source-style float32 rounding of the scaled native tables.
    rounded = [
        np.asarray(table, dtype=np.float32).astype(np.float64)
        for table in native.prepared.coefficients
    ]
    imported = O3TensorProduct(
        plan,
        internal_weights=False,
        coefficients=rounded,
        coefficient_tolerance=1e-6,
    )
    assert imported.coefficient_origin == "imported"
    assert imported.coefficient_tolerance == 1e-6
    for table, expected in zip(imported.prepared.coefficients, rounded, strict=True):
        np.testing.assert_array_equal(table, expected)
    assert imported.coefficient_id != native.coefficient_id
    assert (
        imported.coefficient_id
        == O3TensorProduct(
            plan,
            internal_weights=False,
            coefficients=rounded,
            coefficient_tolerance=1e-6,
        ).coefficient_id
    )
    left = jr.normal(
        jr.key(100), (plan.left_representation.packed_size,), dtype=jnp.float64
    )
    right = jr.normal(
        jr.key(101), (plan.right_representation.packed_size,), dtype=jnp.float64
    )
    weights = jr.normal(jr.key(102), (plan.parameter_count,), dtype=jnp.float64)
    np.testing.assert_allclose(
        imported(left, right, weights), native(left, right, weights), rtol=1e-5, atol=1e-6
    )


def test_imported_coefficients_outside_tolerance_or_shape_are_refused() -> None:
    plan = _harmonic_message_plan(1, 1, 1)
    native = [
        np.asarray(table)
        for table in O3TensorProduct(plan, internal_weights=False).prepared.coefficients
    ]
    flipped = [table.copy() for table in native]
    flipped[-1] = -flipped[-1]
    with pytest.raises(ValueError, match="deviate from the native coupling"):
        O3TensorProduct(
            plan, internal_weights=False, coefficients=flipped, coefficient_tolerance=1e-6
        )
    with pytest.raises(ValueError, match="output-major shape"):
        O3TensorProduct(
            plan,
            internal_weights=False,
            coefficients=[table[..., :0] for table in native],
            coefficient_tolerance=1e-6,
        )
    with pytest.raises(ValueError, match="one table per path"):
        O3TensorProduct(
            plan,
            internal_weights=False,
            coefficients=native[:-1],
            coefficient_tolerance=1.0,
        )
    with pytest.raises(ValueError, match="explicit coefficient_tolerance"):
        O3TensorProduct(plan, internal_weights=False, coefficients=native)
    with pytest.raises(ValueError, match="only to imported"):
        O3TensorProduct(plan, internal_weights=False, coefficient_tolerance=1e-6)


def test_imported_complex_coefficients_are_refused_not_truncated() -> None:
    plan = _harmonic_message_plan(1, 1, 1)
    native = [
        np.asarray(table)
        for table in O3TensorProduct(plan, internal_weights=False).prepared.coefficients
    ]
    with pytest.raises(TypeError, match="complex128"):
        O3TensorProduct(
            plan,
            internal_weights=False,
            coefficients=[table + 9.0j for table in native],
            coefficient_tolerance=1e-6,
        )


def test_coefficient_support_tracks_executed_tables_and_refuses_stale_mutation() -> None:
    plan = _harmonic_message_plan(1, 1, 1)
    native = O3TensorProduct(plan, internal_weights=False)
    tables = [np.asarray(table) for table in native.prepared.coefficients]
    path = next(index for index, table in enumerate(tables) if np.any(table == 0.0))
    zero_row = np.argwhere(tables[path] == 0.0)[0]
    nonzero_row = np.argwhere(tables[path] != 0.0)[0]
    zero = (int(zero_row[0]), int(zero_row[1]), int(zero_row[2]))
    nonzero = (int(nonzero_row[0]), int(nonzero_row[1]), int(nonzero_row[2]))

    # An admitted source entry where the native coupling is zero is executed
    # support, not silently dropped.
    perturbed = [table.copy() for table in tables]
    perturbed[path][zero] = 5e-7
    imported = O3TensorProduct(
        plan, internal_weights=False, coefficients=perturbed, coefficient_tolerance=1e-6
    )
    imported.validate()

    def mutated(entry: tuple[int, int, int], value: float) -> O3TensorProduct:
        return eqx.tree_at(
            lambda module: module.prepared.coefficients[path],
            native,
            native.prepared.coefficients[path].at[entry].set(value),
        )

    with pytest.raises(ValueError):
        mutated(zero, 1e-9).validate()
    with pytest.raises(ValueError):
        mutated(nonzero, 2.0 * float(tables[path][nonzero])).validate()
