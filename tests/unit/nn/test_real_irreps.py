from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.ein import contract
from phydrax.nn.operator.layers import O3IrrepLinear
from phydrax.nn.operator.representations import (
    o3_irrep_action,
    o3_real_coupling,
    O3IrrepBlock,
    O3IrrepLayout,
    O3Representation,
)
from phydrax.special import real_harmonic_basis, RealCartesianHarmonics


def _orthogonal(key: Any, *, improper: bool) -> Any:
    matrix = jr.normal(key, (3, 3), dtype=jnp.float64)
    orthogonal, triangular = jnp.linalg.qr(matrix)
    orthogonal = orthogonal * jnp.where(jnp.diag(triangular) < 0.0, -1.0, 1.0)[None, :]
    proper = orthogonal.at[:, 0].multiply(jnp.linalg.det(orthogonal))
    return -proper if improper else proper


def _layout() -> O3IrrepLayout:
    return O3IrrepLayout(
        [
            O3IrrepBlock("scalar", 0, 1, multiplicity=2),
            O3IrrepBlock("odd", 3, -1, multiplicity=3),
            O3IrrepBlock("axial", 1, 1),
            O3IrrepBlock("odd-again", 3, -1, multiplicity=1),
        ]
    )


def test_layout_packs_blocks_in_declaration_order_multiplicity_major() -> None:
    layout = _layout()
    assert layout.block_names == ("scalar", "odd", "axial", "odd-again")
    assert layout.packed_size == 2 + 21 + 3 + 7
    assert layout.channel_count == 7
    assert [layout.offset(name) for name in layout.block_names] == [0, 2, 23, 26]
    values = jnp.arange(layout.packed_size, dtype=jnp.float64)
    blocks = layout.split(values)
    assert [block.shape for block in blocks] == [(2, 1), (3, 7), (1, 3), (1, 7)]
    np.testing.assert_array_equal(blocks[1][1], jnp.arange(9, 16))
    np.testing.assert_array_equal(layout.join(blocks), values)
    batched = jnp.stack([values, -values])
    np.testing.assert_array_equal(layout.join(layout.split(batched)), batched)
    with pytest.raises(ValueError, match="packed size"):
        layout.split(values[:-1])
    with pytest.raises(ValueError, match="no block named"):
        layout.offset("missing")


def test_repeated_irreps_keep_distinct_identity_and_axes() -> None:
    layout = _layout()
    first = layout.multiplicity_axis("odd")
    second = layout.multiplicity_axis("odd-again")
    assert first.key != second.key
    assert layout.component_axis("odd").key == layout.component_axis("odd-again").key
    assert layout.component_axis("odd").labels[0] == "m=-3"
    with pytest.raises(ValueError, match="unique"):
        O3IrrepLayout([O3IrrepBlock("a", 1, -1), O3IrrepBlock("a", 2, 1)])


def test_equal_sized_layouts_are_not_interchangeable() -> None:
    general_vector = O3IrrepLayout([O3IrrepBlock("vectors", 1, -1)])
    axial = O3IrrepLayout([O3IrrepBlock("vectors", 1, 1)])
    cartesian = O3Representation(vectors=1)
    assert general_vector.packed_size == axial.packed_size == cartesian.packed_size
    identities = {general_vector.layout_id, axial.layout_id, cartesian.layout_id}
    assert len(identities) == 3
    assert cartesian.layout_id == O3Representation(vectors=1).layout_id
    # The Cartesian degree-one basis is (x, y, z); the general basis is (y, z, x).
    np.testing.assert_array_equal(
        O3Representation.real_harmonic_basis(1) @ np.asarray([1.0, 2.0, 3.0]),
        [2.0, 3.0, 1.0],
    )


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (("", 1, 1), ValueError),
        (("a", -1, 1), ValueError),
        (("a", 1, 0), ValueError),
        (("a", True, 1), TypeError),
    ],
    ids=["empty-name", "negative-degree", "zero-parity", "boolean-degree"],
)
def test_invalid_block_declarations_are_refused(
    arguments: tuple[Any, Any, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        O3IrrepBlock(*arguments)
    with pytest.raises(ValueError, match="positive"):
        O3IrrepBlock("a", 1, 1, multiplicity=0)


def test_cartesian_basis_maps_intertwine_existing_cartesian_actions() -> None:
    representation = O3Representation(vectors=1, tensors=1)
    values = jr.normal(jr.key(1), (representation.packed_size,), dtype=jnp.float64)
    for improper in (False, True):
        frame = _orthogonal(jr.key(2), improper=improper)
        transformed = representation.transform(values, frame)
        for degree, parity, sl in ((1, -1, slice(0, 3)), (2, 1, slice(3, 8))):
            basis = jnp.asarray(O3Representation.real_harmonic_basis(degree))
            np.testing.assert_allclose(basis @ basis.T, jnp.eye(2 * degree + 1))
            np.testing.assert_allclose(
                basis @ transformed[sl],
                o3_irrep_action(frame, degree, parity) @ (basis @ values[sl]),
                rtol=1e-13,
                atol=1e-13,
            )
    blocks = representation.irrep_blocks
    assert [(block.name, block.degree, block.parity) for block in blocks] == [
        ("vectors", 1, -1),
        ("tensors", 2, 1),
    ]


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 6])
def test_irrep_actions_are_orthogonal_homomorphisms_with_inversion_parity(
    degree: int,
) -> None:
    first = _orthogonal(jr.key(3), improper=False)
    second = _orthogonal(jr.key(4), improper=True)
    for parity in (-1, 1):
        combined = o3_irrep_action(second @ first, degree, parity)
        np.testing.assert_allclose(
            combined,
            o3_irrep_action(second, degree, parity)
            @ o3_irrep_action(first, degree, parity),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            combined @ combined.T, jnp.eye(2 * degree + 1), atol=1e-12
        )
        np.testing.assert_allclose(
            o3_irrep_action(-jnp.eye(3), degree, parity),
            parity * jnp.eye(2 * degree + 1),
            atol=1e-14,
        )


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_irrep_actions_match_independent_harmonic_rotation(improper: bool) -> None:
    harmonics = RealCartesianHarmonics(5)
    frame = _orthogonal(jr.key(5), improper=improper)
    directions = jr.normal(jr.key(6), (11, 3), dtype=jnp.float64)
    original = harmonics(directions)
    moved = harmonics(directions @ frame.T)
    for degree in range(6):
        block = slice(degree * degree, (degree + 1) ** 2)
        action = o3_irrep_action(frame, degree, (-1) ** degree)
        np.testing.assert_allclose(
            moved[:, block], original[:, block] @ action.T, rtol=1e-12, atol=1e-12
        )


def test_layout_transform_matches_its_packed_representation_matrix() -> None:
    layout = _layout()
    frame = _orthogonal(jr.key(7), improper=True)
    values = jr.normal(jr.key(8), (2, layout.packed_size), dtype=jnp.float64)
    matrix = layout.representation_matrix(frame)
    np.testing.assert_allclose(matrix @ matrix.T, jnp.eye(layout.packed_size), atol=1e-12)
    np.testing.assert_allclose(
        layout.transform(values, frame), values @ matrix.T, rtol=1e-12, atol=1e-12
    )
    with pytest.raises(ValueError, match="orthogonal"):
        layout.transform(values[0], 2.0 * jnp.eye(3))


def _sympy_real_coupling(left: int, right: int, output: int) -> np.ndarray:
    """Independent exact complex coefficients mapped by the real harmonic basis."""
    wigner = pytest.importorskip("sympy.physics.wigner")
    complex_table = np.zeros((2 * left + 1, 2 * right + 1, 2 * output + 1))
    for i, m1 in enumerate(range(-left, left + 1)):
        for j, m2 in enumerate(range(-right, right + 1)):
            m3 = m1 + m2
            if abs(m3) <= output:
                complex_table[i, j, m3 + output] = float(
                    wigner.clebsch_gordan(left, right, output, m1, m2, m3)
                )
    real = contract(
        "oz,ix,jy,xyz->oij",
        np.asarray(real_harmonic_basis(output)),
        np.asarray(real_harmonic_basis(left)).conj(),
        np.asarray(real_harmonic_basis(right)).conj(),
        complex_table,
    )
    gauge = -1j if (left + right + output) % 2 else 1.0
    return gauge * real


@pytest.mark.parametrize(
    "degrees", [(1, 1, 1), (2, 2, 2), (3, 2, 4), (3, 3, 3), (4, 4, 2), (4, 3, 6)]
)
def test_real_coupling_matches_independent_exact_coefficients_and_is_exactly_sparse(
    degrees: tuple[int, int, int],
) -> None:
    coupling = o3_real_coupling(*degrees)
    dense = coupling.dense()
    reference = _sympy_real_coupling(*degrees)
    assert np.max(np.abs(np.imag(reference))) < 1e-14
    np.testing.assert_allclose(dense, np.real(reference), atol=1e-14)
    # Omitted entries are mathematical zeros; stored entries are never residues.
    assert np.all(np.abs(coupling.values) > 1e-3)
    mask = np.zeros(dense.shape, dtype=bool)
    mask[tuple(coupling.indices.T)] = True
    assert np.all(np.abs(np.real(reference)[~mask]) < 1e-15)
    np.testing.assert_allclose(
        contract("oij,pij->op", dense, dense), np.eye(dense.shape[0]), atol=1e-14
    )


def test_real_coupling_is_equivariant_and_refuses_broken_triangles() -> None:
    frame = _orthogonal(jr.key(9), improper=False)
    dense = jnp.asarray(o3_real_coupling(3, 2, 4).dense())
    left = o3_irrep_action(frame, 3, 1)
    right = o3_irrep_action(frame, 2, 1)
    output = o3_irrep_action(frame, 4, 1)
    np.testing.assert_allclose(
        contract("oij,ia,jb->oab", dense, left, right),
        contract("po,oab->pab", output, dense),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="cannot couple"):
        o3_real_coupling(1, 1, 3)


def test_irrep_linear_is_equivariant_and_mixes_only_matching_irreps() -> None:
    source = _layout()
    target = O3IrrepLayout(
        [O3IrrepBlock("odd-out", 3, -1, multiplicity=2), O3IrrepBlock("s", 0, 1)]
    )
    linear = O3IrrepLinear(source, target, key=jr.key(10))
    assert [weight.shape for weight in linear.weights] == [(2, 4), (1, 2)]
    values = jr.normal(jr.key(11), (3, source.packed_size), dtype=jnp.float64)
    frame = _orthogonal(jr.key(12), improper=True)
    np.testing.assert_allclose(
        linear(source.transform(values, frame)),
        target.transform(linear(values), frame),
        rtol=1e-12,
        atol=1e-12,
    )
    ones = O3IrrepLinear(source, target, weights=[jnp.ones((2, 4)), jnp.ones((1, 2))])
    odd, scalar = target.split(ones(values))
    blocks = source.split(values)
    expected = (jnp.sum(blocks[1], axis=-2) + blocks[3][..., 0, :]) / 2.0
    np.testing.assert_allclose(odd[..., 0, :], expected, rtol=1e-13)
    np.testing.assert_allclose(
        scalar[..., 0, 0], jnp.sum(blocks[0][..., 0], -1) / jnp.sqrt(2.0)
    )
    with pytest.raises(ValueError, match="no input block"):
        O3IrrepLinear(source, O3IrrepLayout([O3IrrepBlock("t", 2, 1)]))
    with pytest.raises(ValueError, match="shapes"):
        O3IrrepLinear(source, target, weights=[jnp.ones((2, 3)), jnp.ones((1, 2))])


def test_irrep_action_is_differentiable_in_the_frame() -> None:
    frame = _orthogonal(jr.key(13), improper=False)
    generator = jnp.asarray([[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]])
    action = jax.jacfwd(
        lambda t: o3_irrep_action(frame @ jax.scipy.linalg.expm(t * generator), 3, -1)
    )(0.0)
    step = 1e-6
    finite = (
        o3_irrep_action(frame @ jax.scipy.linalg.expm(step * generator), 3, -1)
        - o3_irrep_action(frame @ jax.scipy.linalg.expm(-step * generator), 3, -1)
    ) / (2.0 * step)
    np.testing.assert_allclose(action, finite, rtol=1e-6, atol=1e-8)
