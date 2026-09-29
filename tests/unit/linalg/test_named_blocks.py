from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from tests._support.differentiation import (
    assert_coordinate_transpose_duality,
    assert_hilbert_adjoint_duality,
)


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]

_MATERIALIZATION = la.MaterializationPolicy(max_entries=10_000, max_bytes=1_000_000)

# Canonical coordinates of the nested space: velocity [0, 2), pressure [2, 5),
# multiplier [5, 6).
_VELOCITY_WEIGHTS = np.asarray([2.0, 3.0])
_PRESSURE_WEIGHTS = np.asarray([0.5, 4.0, 1.5])


def _nested_space() -> la.BlockSpace:
    velocity = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray(_VELOCITY_WEIGHTS)),
        space_id="named-velocity",
    )
    pressure = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray(_PRESSURE_WEIGHTS)),
        space_id="named-pressure",
    )
    multiplier = la.ArraySpace((1,), dtype=jnp.float64, space_id="named-multiplier")
    fluid = la.BlockSpace((velocity, pressure), names=("velocity", "pressure"))
    return la.BlockSpace((fluid, multiplier), names=("fluid", "multiplier"))


def _nested_vector(values: np.ndarray) -> tuple[tuple[Array, Array], Array]:
    return (
        (jnp.asarray(values[:2]), jnp.asarray(values[2:5])),
        jnp.asarray(values[5:]),
    )


def _flat(space: la.AbstractVectorSpace, vector: object) -> np.ndarray:
    return np.asarray(space.flatten(vector))


def _block(space: la.AbstractVectorSpace) -> la.BlockSpace:
    assert isinstance(space, la.BlockSpace)
    return space


def test_nested_path_selection_restricts_embeds_and_pairs_exactly() -> None:
    space = _nested_space()
    selection = la.BlockSelection(
        space,
        (
            ("pressure", ("fluid", "pressure")),
            ("coupled", (("multiplier",), ("fluid", "velocity"))),
        ),
    )
    restriction = la.BlockRestrictionLinearOperator(selection)
    prolongation = la.BlockProlongationLinearOperator(selection)
    values = np.asarray([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
    vector = _nested_vector(values)
    # Independent selection matrix: target coordinates are pressure, multiplier,
    # then velocity.
    matrix = np.eye(6)[[2, 3, 4, 5, 0, 1]]

    restricted = restriction.mv(vector)

    assert selection.names == ("pressure", "coupled")
    assert _block(selection.target.spaces[1]).names == ("multiplier", "fluid/velocity")
    assert selection.covers_source and selection.preserves_pairing
    np.testing.assert_allclose(_flat(selection.target, restricted), matrix @ values)
    np.testing.assert_allclose(_flat(space, prolongation.mv(restricted)), values)
    np.testing.assert_allclose(
        la.materialize(restriction, _MATERIALIZATION), matrix, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        la.materialize(prolongation, _MATERIALIZATION), matrix.T, rtol=1e-12, atol=1e-12
    )
    target_values = np.asarray([0.25, -1.5, 2.0, 3.5, 0.75, -0.5])
    target_vector = selection.target.unflatten(jnp.asarray(target_values))
    for operator, primal, covector in (
        (restriction, vector, target_vector),
        (prolongation, target_vector, vector),
    ):
        assert_coordinate_transpose_duality(
            operator, primal, covector, rtol=1e-12, atol=1e-12
        )
        assert_hilbert_adjoint_duality(operator, primal, covector, rtol=1e-12, atol=1e-12)
    # Path members keep the source pairings, so the embedding is the Hilbert adjoint.
    np.testing.assert_allclose(
        _flat(space, restriction.adjoint_mv(target_vector)),
        _flat(space, prolongation.mv(target_vector)),
        rtol=1e-12,
        atol=1e-12,
    )
    partial = la.BlockSelection(space, (("pressure", ("fluid", "pressure")),))
    embedded = la.BlockProlongationLinearOperator(partial).mv(
        (jnp.asarray([7.0, 8.0, 9.0]),)
    )
    np.testing.assert_allclose(_flat(space, embedded), [0.0, 0.0, 7.0, 8.0, 9.0, 0.0])


def test_coordinate_blocks_keep_transpose_and_hilbert_adjoint_distinct() -> None:
    weights = np.asarray([2.0, 0.5, 3.0, 8.0])
    source = la.ArraySpace(
        (4,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray(weights)),
        space_id="weighted-native",
    )
    first = la.ArraySpace((2,), dtype=jnp.float64, space_id="coordinate-first")
    second = la.ArraySpace((1,), dtype=jnp.float64, space_id="coordinate-second")
    selection = la.BlockSelection(
        source,
        (
            ("first", la.CoordinateBlock(first, ((2, 4),))),
            ("second", la.CoordinateBlock(second, ((0, 1),))),
        ),
    )
    restriction = la.BlockRestrictionLinearOperator(selection)
    matrix = np.eye(4)[[2, 3, 0]]
    covector = selection.target.unflatten(jnp.asarray([1.0, -2.0, 4.0]))

    transposed = np.asarray(restriction.transpose_mv(covector))
    adjoint = np.asarray(restriction.adjoint_mv(covector))

    assert not selection.covers_source and not selection.preserves_pairing
    np.testing.assert_allclose(transposed, matrix.T @ np.asarray([1.0, -2.0, 4.0]))
    # Euclidean member pairings against a weighted source: R* = W^{-1} R^T.
    np.testing.assert_allclose(adjoint, transposed / weights, rtol=1e-12, atol=1e-12)
    assert not np.allclose(adjoint, transposed)
    assert_hilbert_adjoint_duality(
        restriction, jnp.asarray([1.0, 2.0, 3.0, 4.0]), covector, rtol=1e-12, atol=1e-12
    )


def _refusals() -> dict[str, tuple[Callable[[], object], type[Exception]]]:
    space = _nested_space()
    flat = la.ArraySpace((6,), dtype=jnp.float64)
    single = la.ArraySpace((1,), dtype=jnp.float64)
    velocity = _block(space.spaces[0]).spaces[0]
    return {
        "overlapping-members": (
            lambda: la.BlockSelection(
                space, (("fluid", ("fluid",)), ("velocity", ("fluid", "velocity")))
            ),
            ValueError,
        ),
        "unknown-block": (
            lambda: la.BlockSelection(space, (("x", ("fluid", "density")),)),
            ValueError,
        ),
        "path-below-leaf": (
            lambda: la.BlockSelection(space, (("x", ("multiplier", "value")),)),
            ValueError,
        ),
        "path-on-array-source": (
            lambda: la.BlockSelection(flat, (("x", ("fluid",)),)),
            TypeError,
        ),
        "coordinate-size": (
            lambda: la.CoordinateBlock(single, ((0, 2),)),
            ValueError,
        ),
        "coordinate-dtype": (
            lambda: la.BlockSelection(
                flat,
                (
                    (
                        "x",
                        la.CoordinateBlock(
                            la.ArraySpace((1,), dtype=jnp.float32), ((0, 1),)
                        ),
                    ),
                ),
            ),
            TypeError,
        ),
        "partial-coordinate-map": (
            lambda: la.MappedBlockLinearOperator(
                la.IdentityLinearOperator(la.BlockSpace((single,), names=("x",))),
                row_map=la.BlockSelection(
                    flat, (("x", la.CoordinateBlock(single, ((0, 1),))),)
                ),
                column_map=la.BlockSelection(
                    flat, (("x", la.CoordinateBlock(single, ((0, 1),))),)
                ),
            ),
            ValueError,
        ),
        "repeated-entry": (
            lambda: la.assemble_block_operator(
                (
                    (
                        ("fluid", "velocity"),
                        ("fluid", "velocity"),
                        la.IdentityLinearOperator(velocity),
                    ),
                    (
                        ("fluid", "velocity"),
                        ("fluid", "velocity"),
                        la.IdentityLinearOperator(velocity),
                    ),
                ),
                source=space,
                target=space,
            ),
            ValueError,
        ),
        "entry-space": (
            lambda: la.assemble_block_operator(
                (
                    (
                        ("multiplier",),
                        ("fluid", "velocity"),
                        la.IdentityLinearOperator(velocity),
                    ),
                ),
                source=space,
                target=space,
            ),
            ValueError,
        ),
    }


@pytest.mark.parametrize("case", sorted(_refusals()))
def test_named_block_refusals(case: str) -> None:
    build, error = _refusals()[case]
    with pytest.raises(error):
        build()


def _saddle_matrix() -> np.ndarray:
    """Independent dense three-field matrix in canonical (u, T, lambda) order."""
    stiffness = np.asarray([[4.0, 1.0], [1.0, 3.0]])
    coupling = np.asarray([[0.5, -1.0, 0.0], [0.25, 0.0, 2.0]])
    mass = np.diag([5.0, 6.0, 7.0])
    constraint = np.asarray([[1.0, 2.0]])
    matrix = np.zeros((6, 6))
    matrix[:2, :2] = stiffness
    matrix[:2, 2:5] = coupling
    matrix[2:5, :2] = coupling.T
    matrix[2:5, 2:5] = mass
    matrix[:2, 5:] = constraint.T
    matrix[5:, :2] = constraint
    return matrix


def _assembled_saddle(space: la.BlockSpace, matrix: np.ndarray) -> la.BlockLinearOperator:
    velocity, pressure = _block(space.spaces[0]).spaces
    multiplier = space.spaces[1]
    spans = {
        ("fluid", "velocity"): (slice(0, 2), velocity),
        ("fluid", "pressure"): (slice(2, 5), pressure),
        ("multiplier",): (slice(5, 6), multiplier),
    }
    entries = []
    for row_path, (rows, target) in spans.items():
        for column_path, (columns, source) in spans.items():
            block = matrix[rows, columns]
            if np.any(block):
                entries.append(
                    (
                        row_path,
                        column_path,
                        la.DenseLinearOperator(
                            jnp.asarray(block), source=source, target=target
                        ),
                    )
                )
    return la.assemble_block_operator(tuple(entries), source=space, target=space)


def test_regrouped_three_field_system_feeds_the_two_by_two_block_factorization() -> None:
    space = _nested_space()
    matrix = _saddle_matrix()
    operator = _assembled_saddle(space, matrix)
    grouping = la.BlockSelection(
        space,
        (
            ("pivot", (("fluid", "velocity"), ("fluid", "pressure"))),
            ("constraint", (("multiplier",),)),
        ),
    )
    builder = la.BlockFactorizationPreconditionerBuilder(
        la.DenseInversePreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
        "ldu",
    )
    residual_values = np.asarray([1.0, -0.5, 2.0, 0.25, -1.0, 3.0])

    regrouped = la.select_block_operator(operator, grouping, grouping)
    local = builder.prepare(regrouped, materialization=_MATERIALIZATION)
    grouped_solution = local.apply(
        la.BlockRestrictionLinearOperator(grouping).mv(_nested_vector(residual_values))
    )
    term = la.SubspaceCorrectionTerm(
        la.BlockRestrictionLinearOperator(grouping),
        la.BlockProlongationLinearOperator(grouping),
        builder,
    )
    corrected = la.AdditiveSubspaceCorrectionBuilder((term,)).prepare(
        operator, materialization=_MATERIALIZATION
    )
    expected = np.linalg.solve(matrix, residual_values)

    np.testing.assert_allclose(
        la.materialize(operator, _MATERIALIZATION), matrix, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        la.materialize(regrouped, _MATERIALIZATION), matrix, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _flat(grouping.target, grouped_solution), expected, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        _flat(space, corrected.apply(_nested_vector(residual_values))),
        expected,
        rtol=1e-11,
        atol=1e-11,
    )


def test_named_block_jacobi_certifies_self_adjoint_only_for_paired_transfers() -> None:
    space = _nested_space()
    base = _saddle_matrix()
    matrix = base.T @ base + np.eye(6)
    operator = la.DenseLinearOperator(
        jnp.asarray(matrix),
        source=space,
        target=space,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"self_adjoint": "asserted", "positive_definite": "asserted"},
        ),
    )
    groups = (
        ("velocity", (("fluid", "velocity"),)),
        ("rest", (("fluid", "pressure"), ("multiplier",))),
    )
    terms = tuple(
        la.SubspaceCorrectionTerm(
            la.BlockRestrictionLinearOperator(selection),
            la.BlockProlongationLinearOperator(selection),
            la.DenseInversePreconditionerBuilder(),
        )
        for selection in (la.BlockSelection(space, (group,)) for group in groups)
    )
    residual = np.asarray([2.0, -1.0, 0.5, 3.0, -2.0, 1.0])
    expected = np.zeros(6)
    for block in (slice(0, 2), slice(2, 6)):
        expected[block] = np.linalg.solve(matrix[block, block], residual[block])
    flat = la.ArraySpace((6,), dtype=jnp.float64, space_id="flat-coordinates")
    identified = la.BlockSelection(space, (("all", la.CoordinateBlock(flat, ((0, 6),))),))
    unpaired = la.SubspaceCorrectionTerm(
        la.BlockRestrictionLinearOperator(identified),
        la.BlockProlongationLinearOperator(identified),
        la.DenseInversePreconditionerBuilder(),
    )

    additive = la.AdditiveSubspaceCorrectionBuilder(terms)
    prepared = additive.prepare(operator, materialization=_MATERIALIZATION)

    assert additive.properties_for(operator).certifies("self_adjoint")
    assert (
        la.MultiplicativeSubspaceCorrectionBuilder(terms, sweep="symmetric")
        .properties_for(operator)
        .certifies("self_adjoint")
    )
    assert not (
        la.AdditiveSubspaceCorrectionBuilder((unpaired,))
        .properties_for(operator)
        .certifies("self_adjoint")
    )
    np.testing.assert_allclose(
        _flat(space, prepared.apply(_nested_vector(residual))),
        expected,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mapped_block_operator_uses_explicit_row_and_column_maps() -> None:
    source_weights = np.asarray([2.0, 5.0, 0.5])
    target_weights = np.asarray([3.0, 0.25, 7.0])
    source = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray(source_weights)),
        space_id="mapped-native-source",
    )
    target = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray(target_weights)),
        space_id="mapped-native-target",
    )
    first = la.ArraySpace((2,), dtype=jnp.float64, space_id="mapped-first")
    second = la.ArraySpace((1,), dtype=jnp.float64, space_id="mapped-second")
    rows_first = la.ArraySpace((1,), dtype=jnp.float64, space_id="mapped-row-first")
    rows_second = la.ArraySpace((2,), dtype=jnp.float64, space_id="mapped-row-second")
    column_map = la.BlockSelection(
        source,
        (
            ("first", la.CoordinateBlock(first, ((0, 2),))),
            ("second", la.CoordinateBlock(second, ((2, 3),))),
        ),
    )
    permuted_rows = la.BlockSelection(
        target,
        (
            ("first", la.CoordinateBlock(rows_first, ((2, 3),))),
            ("second", la.CoordinateBlock(rows_second, ((0, 2),))),
        ),
    )
    ordered_rows = la.BlockSelection(
        target,
        (
            ("first", la.CoordinateBlock(rows_first, ((0, 1),))),
            ("second", la.CoordinateBlock(rows_second, ((1, 3),))),
        ),
    )
    block_matrix = np.asarray([[1.0, 2.0, -1.0], [0.5, 3.0, 4.0], [-2.0, 1.0, 6.0]])

    def mapped(rows: la.BlockSelection) -> la.MappedBlockLinearOperator:
        block = la.assemble_block_operator(
            tuple(
                (
                    (row_name,),
                    (column_name,),
                    la.DenseLinearOperator(
                        jnp.asarray(block_matrix[row_slice, column_slice]),
                        source=column_space,
                        target=row_space,
                    ),
                )
                for row_name, row_slice, row_space in (
                    ("first", slice(0, 1), rows_first),
                    ("second", slice(1, 3), rows_second),
                )
                for column_name, column_slice, column_space in (
                    ("first", slice(0, 2), first),
                    ("second", slice(2, 3), second),
                )
            ),
            source=column_map.target,
            target=rows.target,
        )
        return la.MappedBlockLinearOperator(block, row_map=rows, column_map=column_map)

    operator = mapped(permuted_rows)
    # Independent native matrix: block row "first" occupies native row 2.
    native = block_matrix[[1, 2, 0]]
    primal = np.asarray([1.0, -2.0, 0.5])
    covector = np.asarray([0.25, 3.0, -1.0])
    identity_ordered = mapped(ordered_rows)

    np.testing.assert_allclose(operator.mv(jnp.asarray(primal)), native @ primal)
    np.testing.assert_allclose(
        operator.transpose_mv(jnp.asarray(covector)), native.T @ covector
    )
    np.testing.assert_allclose(
        operator.adjoint_mv(jnp.asarray(covector)),
        (native.T @ (target_weights * covector)) / source_weights,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        la.materialize(operator, _MATERIALIZATION), native, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        la.materialize(operator.coordinate_transpose(), _MATERIALIZATION),
        native.T,
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="identity-ordered"):
        operator.with_inactive_identity(jnp.asarray(False), source=source, target=target)
    np.testing.assert_allclose(
        la.materialize(
            identity_ordered.with_inactive_identity(
                jnp.asarray(True), source=source, target=target
            ),
            _MATERIALIZATION,
        ),
        block_matrix,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        la.materialize(
            identity_ordered.with_inactive_identity(
                jnp.asarray(False), source=source, target=target
            ),
            _MATERIALIZATION,
        ),
        np.eye(3),
        rtol=1e-12,
        atol=1e-12,
    )
    # A canonical 2 + 1 split crosses the map's named blocks; the extracted blocks
    # are exact coordinate selections of the native matrix.
    canonical_rows = la.BlockSelection(
        target,
        (
            ("a", la.CoordinateBlock(la.ArraySpace((2,), dtype=jnp.float64), ((0, 2),))),
            ("b", la.CoordinateBlock(la.ArraySpace((1,), dtype=jnp.float64), ((2, 3),))),
        ),
    )
    canonical_columns = la.BlockSelection(
        source,
        (
            ("a", la.CoordinateBlock(la.ArraySpace((2,), dtype=jnp.float64), ((1, 3),))),
            ("b", la.CoordinateBlock(la.ArraySpace((1,), dtype=jnp.float64), ((0, 1),))),
        ),
    )
    grid = la.select_block_operator(identity_ordered, canonical_rows, canonical_columns)
    np.testing.assert_allclose(
        la.materialize(grid, _MATERIALIZATION),
        block_matrix[np.ix_([0, 1, 2], [1, 2, 0])],
        rtol=1e-12,
        atol=1e-12,
    )
