#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import gc
import pickle
import random
import weakref
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint, numeric_array_fingerprint
from phydrax.discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    coefficient_payload,
    RadicalElement,
)
from phydrax.discretization._exact_coordinate_bank import ExactCoordinateBank
from phydrax.geometry._rational import binary64_dyadic_entries, dyadic_entry


def _rows() -> tuple[tuple[Fraction | RadicalElement, ...], ...]:
    return (
        (Fraction(0), Fraction(3, 8), Fraction(-5)),
        (Fraction(1, 3), Fraction((1 << 70) + 1, 1 << 12), Fraction(6)),
        (RadicalElement.sqrt(Fraction(2)), Fraction(-1, 1 << 40), Fraction(7, 2)),
    )


def test_bank_packs_dyadics_densely_and_keeps_other_scalars_sparse() -> None:
    rows = _rows()
    bank = ExactCoordinateBank.from_rows(rows)
    assert bank.shape == (3, 3)
    assert tuple(bank) == rows
    assert bank[1] == rows[1] and bank[-1] == rows[2]
    assert bank[0:2] == rows[0:2]
    assert bank.rows(np.asarray((2, 0), dtype=np.int64)) == (rows[2], rows[0])
    # 1/3, the 71-bit significand and the radical are sparse; zero is (0, 0).
    assert len(bank._sparse_values) == 3
    assert not bank.rational
    assert bank._significands[0, 0] == 0 and bank._exponents[0, 0] == 0
    assert bank.coefficient_payloads() == tuple(
        tuple(coefficient_payload(value) for value in row) for row in rows
    )


@pytest.mark.parametrize(
    "rows",
    (_rows(), (), ((), ()), ((Fraction(1, 3),),)),
)
def test_canonical_parts_are_immutable_after_construction_and_restore(
    rows: tuple[tuple[Fraction | RadicalElement, ...], ...],
) -> None:
    bank = ExactCoordinateBank.from_rows(rows)
    restored = pickle.loads(pickle.dumps(bank))
    identity = bank.content_fingerprint()
    assert restored.content_fingerprint() == identity
    for current in (bank, restored):
        internal = (
            current._significands,
            current._exponents,
            current._sparse_index,
            current._sparse_row_starts,
        )
        significands, exponents, indices, offsets, _ = current.canonical_parts()
        for array in (*internal, significands, exponents, indices, offsets):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
            with pytest.raises(ValueError):
                array.view().setflags(write=True)
            np.testing.assert_array_equal(np.asarray(jnp.asarray(array)), array)
        assert current.content_fingerprint() == identity


def test_canonical_parts_preserve_packed_values_and_fingerprint() -> None:
    bank = ExactCoordinateBank.from_rows(_rows())
    significands, exponents, indices, offsets, sparse = bank.canonical_parts()
    assert significands.dtype == np.dtype(np.int64)
    assert exponents.dtype == np.dtype(np.int16)
    assert indices.dtype == offsets.dtype == np.dtype(np.int64)
    assert tuple(indices) == (3, 4, 6)
    assert tuple(offsets) == (0, 0, 2, 3)
    assert sparse == (_rows()[1][0], _rows()[1][1], _rows()[2][0])
    expected = canonical_fingerprint(
        {
            "kind": "exact-coordinate-bank",
            "significands": numeric_array_fingerprint(significands),
            "exponents": numeric_array_fingerprint(exponents),
            "sparse_indices": numeric_array_fingerprint(indices),
            "sparse_values": tuple(coefficient_payload(value) for value in sparse),
        }
    )
    assert bank.content_fingerprint() == expected
    assert bank.content_fingerprint() == expected


def test_fingerprint_tracks_shape_and_sparse_coefficients() -> None:
    rows = _rows()
    bank = ExactCoordinateBank.from_rows(rows)
    identity = bank.content_fingerprint()
    assert ExactCoordinateBank.from_rows(rows).content_fingerprint() == identity
    changed = list(map(list, rows))
    changed[1][0] = Fraction(1, 7)
    assert ExactCoordinateBank.from_rows(changed).content_fingerprint() != identity


@pytest.mark.parametrize("part", ("shape", "hash"))
def test_fingerprint_rejects_tampered_derived_bank_metadata(part: str) -> None:
    bank = ExactCoordinateBank.from_rows(((Fraction(1),), (Fraction(2),)))
    bank.content_fingerprint()
    if part == "shape":
        object.__setattr__(bank, "_shape", (1, 2))
    else:
        object.__setattr__(bank, "_hash", hash(bank) + 1)
    with pytest.raises(ValueError, match="derived"):
        bank.content_fingerprint()


def test_fingerprinting_does_not_retain_source_bank() -> None:
    bank = ExactCoordinateBank.from_rows(_rows())
    bank.content_fingerprint()
    bank.content_fingerprint()
    reference = weakref.ref(bank)
    del bank
    gc.collect()
    assert reference() is None


def test_interval_endpoints_match_directed_exact_coefficients() -> None:
    values = (
        (Fraction(0), Fraction(3, 8), Fraction(1, 3)),
        (Fraction(2**53 + 1, 2**53), Fraction(1, 2**1075), Fraction(-1, 2**1075)),
        (RadicalElement.sqrt(Fraction(2)), Fraction(-7, 8), Fraction(2**60 + 1)),
    )
    lower, upper = ExactCoordinateBank.from_rows(values).interval_endpoints()
    expected_lower = np.asarray(
        [
            [coefficient_endpoint(value, upper=False, saturate=True) for value in row]
            for row in values
        ],
        dtype=np.float64,
    )
    expected_upper = np.asarray(
        [
            [coefficient_endpoint(value, upper=True, saturate=True) for value in row]
            for row in values
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(lower.view(np.uint64), expected_lower.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected_upper.view(np.uint64))


def test_interval_endpoints_reject_an_inexact_binary64_proposal() -> None:
    values = (
        (Fraction(1), Fraction(-3, 8), Fraction(1, 2**1074)),
        (Fraction(2**53 + 1), Fraction(-(2**53 + 1)), Fraction(1, 2**1075)),
    )
    bank = ExactCoordinateBank.from_rows(values)
    lower, upper = bank.interval_endpoints()
    minimum = float.fromhex("0x0.0000000000001p-1022")
    expected_lower = np.asarray(
        ((1.0, -0.375, minimum), (float(2**53), -float(2**53 + 2), 0.0)),
        dtype=np.float64,
    )
    expected_upper = np.asarray(
        ((1.0, -0.375, minimum), (float(2**53 + 2), -float(2**53), minimum)),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(lower.view(np.uint64), expected_lower.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected_upper.view(np.uint64))
    assert np.all(lower[1] < upper[1])


def test_interval_endpoints_reject_wrapped_significand_proposals() -> None:
    exact = Fraction((2**62 + 2**50 + 1) * 4)
    bank = ExactCoordinateBank.from_rows(((exact, -exact),))
    proposal = float.fromhex("0x1.0000000000004p+52")
    proposals = np.asarray(((proposal, -proposal),), dtype=np.float64)
    lower, upper = bank._verified_interval_endpoints(proposals)
    below = float.fromhex("0x1.0010000000000p+64")
    above = float.fromhex("0x1.0010000000001p+64")
    expected_lower = np.asarray(((below, -above),), dtype=np.float64)
    expected_upper = np.asarray(((above, -below),), dtype=np.float64)
    np.testing.assert_array_equal(lower.view(np.uint64), expected_lower.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected_upper.view(np.uint64))


def test_interval_endpoints_reject_invalid_shift_proposals() -> None:
    values = (
        Fraction(1),
        Fraction(-1),
        Fraction(1, 2**1074),
        Fraction(-1, 2**1074),
    )
    bank = ExactCoordinateBank.from_rows((values,))
    proposals = np.asarray(
        (
            (
                float.fromhex("0x1p-100"),
                -float.fromhex("0x1p+100"),
                float.fromhex("0x1p-1000"),
                0.0,
            ),
        ),
        dtype=np.float64,
    )
    lower, upper = bank._verified_interval_endpoints(proposals)
    minimum = float.fromhex("0x0.0000000000001p-1022")
    expected = np.asarray(((1.0, -1.0, minimum, -minimum),), dtype=np.float64)
    np.testing.assert_array_equal(lower.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected.view(np.uint64))


def test_interval_endpoints_reject_discarded_bits_and_wrong_sign() -> None:
    bank = ExactCoordinateBank.from_rows(((Fraction(3), Fraction(-3)),))
    proposal = np.asarray(((np.nextafter(3.0, np.inf), 3.0),), dtype=np.float64)
    lower, upper = bank._verified_interval_endpoints(proposal)
    expected = np.asarray(((3.0, -3.0),), dtype=np.float64)
    np.testing.assert_array_equal(lower.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected.view(np.uint64))


def test_interval_endpoints_reject_wrong_binade_boundary_proposals() -> None:
    exact = Fraction((1 << 54) - 3, 1 << 54)
    bank = ExactCoordinateBank.from_rows(((exact, -exact),))
    proposals = np.asarray(((1.0, -1.0),), dtype=np.float64)
    lower, upper = bank._verified_interval_endpoints(proposals)
    previous = float.fromhex("0x1.fffffffffffffp-1")
    earlier = float.fromhex("0x1.ffffffffffffep-1")
    expected_lower = np.asarray(((earlier, -previous),), dtype=np.float64)
    expected_upper = np.asarray(((previous, -earlier),), dtype=np.float64)
    np.testing.assert_array_equal(lower.view(np.uint64), expected_lower.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected_upper.view(np.uint64))


def test_interval_endpoints_saturate_out_of_range_entries_in_mixed_bank() -> None:
    huge = Fraction(1 << 2000)
    values = (
        (Fraction(1), huge, -huge),
        (Fraction(1, 3), Fraction(-3, 8), Fraction((1 << 2000) + 1, 3)),
    )
    bank = ExactCoordinateBank.from_rows(values)
    with pytest.raises(OverflowError):
        bank.binary64()
    lower, upper = bank.interval_endpoints()
    expected_lower = np.asarray(
        [
            [coefficient_endpoint(value, upper=False, saturate=True) for value in row]
            for row in values
        ],
        dtype=np.float64,
    )
    expected_upper = np.asarray(
        [
            [coefficient_endpoint(value, upper=True, saturate=True) for value in row]
            for row in values
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(lower.view(np.uint64), expected_lower.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected_upper.view(np.uint64))


def test_interval_endpoints_match_exact_wide_dyadic_extremes() -> None:
    generator = random.Random(87461)
    values = []
    for _ in range(2000):
        width = generator.choice((1, 2, 51, 52, 53, 54, 61, 62, 63))
        magnitude = generator.randrange(1 << (width - 1), 1 << width) | 1
        exponent = generator.choice(
            (
                -32767,
                -2048,
                -1200,
                -1075,
                -1074,
                -1073,
                -1022,
                -100,
                -1,
                0,
                1,
                generator.randint(-1200, 1022 - (width - 1)),
                1022 - (width - 1),
            )
        )
        value = (
            Fraction(magnitude << exponent)
            if exponent >= 0
            else Fraction(magnitude, 1 << -exponent)
        )
        values.append(value if generator.getrandbits(1) else -value)
    values.extend((Fraction(float(np.finfo(np.float64).max)), Fraction(1, 2**1074)))
    bank = ExactCoordinateBank.from_rows((tuple(values),))
    lower, upper = bank.interval_endpoints()
    expected_lower = np.asarray(
        [[coefficient_endpoint(value, upper=False, saturate=True) for value in values]],
        dtype=np.float64,
    )
    expected_upper = np.asarray(
        [[coefficient_endpoint(value, upper=True, saturate=True) for value in values]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(lower.view(np.uint64), expected_lower.view(np.uint64))
    np.testing.assert_array_equal(upper.view(np.uint64), expected_upper.view(np.uint64))


def test_bank_equality_hash_and_immutability_are_canonical() -> None:
    rows = _rows()
    bank = ExactCoordinateBank.from_rows(rows)
    same = ExactCoordinateBank.from_rows(tuple(tuple(row) for row in rows))
    assert bank == same and hash(bank) == hash(same)
    assert bank != rows
    changed = list(map(list, rows))
    changed[0][1] = Fraction(5, 8)
    assert bank != ExactCoordinateBank.from_rows(changed)
    assert ExactCoordinateBank.from_rows(bank) is bank
    with pytest.raises(AttributeError):
        bank._shape = (0, 0)
    with pytest.raises(ValueError):
        bank._significands[0, 0] = 1
    restored = pickle.loads(pickle.dumps(bank))
    assert restored == bank and tuple(restored) == rows


@pytest.mark.parametrize(
    "rational",
    (Fraction(0), Fraction(1), Fraction(-3, 8), Fraction(1, 3), Fraction(1 << 80)),
)
def test_rational_radicals_and_fractions_share_canonical_bank(rational: Fraction) -> None:
    direct = ExactCoordinateBank.from_rows(((rational,),))
    radical = ExactCoordinateBank.from_rows(((RadicalElement(rational),),))
    assert direct == radical
    assert hash(direct) == hash(radical)
    assert direct.content_fingerprint() == radical.content_fingerprint()
    assert tuple(radical) == ((rational,),)
    assert pickle.loads(pickle.dumps(radical)) == direct


@pytest.mark.parametrize(
    ("significand", "exponent", "indices", "values"),
    (
        (2, 0, (), ()),
        (0, 1, (), ()),
        (0, -32768, (0,), (Fraction(1, 2),)),
        (0, -32768, (0,), (RadicalElement(0),)),
        (0, -32768, (0,), (RadicalElement(1),)),
        (0, -32768, (0,), (RadicalElement(Fraction(1, 3)),)),
    ),
)
def test_noncanonical_bank_parts_are_rejected_on_construction_and_restore(
    significand: int,
    exponent: int,
    indices: tuple[int, ...],
    values: tuple[Fraction | RadicalElement, ...],
) -> None:
    parts = (
        np.asarray(((significand,),), dtype=np.int64),
        np.asarray(((exponent,),), dtype=np.int16),
        np.asarray(indices, dtype=np.int64),
        values,
    )
    with pytest.raises(ValueError):
        ExactCoordinateBank(*parts)

    class InvalidStoredBank:
        def __reduce__(self) -> tuple[object, tuple[object, ...]]:
            return ExactCoordinateBank, parts

    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(InvalidStoredBank()))


def test_bank_take_and_replace_rows_preserve_sparse_entries() -> None:
    rows = _rows()
    bank = ExactCoordinateBank.from_rows(rows)
    taken = bank.take(np.asarray((2, 1), dtype=np.int64))
    assert tuple(taken) == (rows[2], rows[1])
    replacement = ExactCoordinateBank.from_rows(
        ((Fraction(1, 7), Fraction(2), Fraction(0)),)
    )
    replaced = bank.replace_rows(np.asarray((1,), dtype=np.int64), replacement)
    assert tuple(replaced) == (rows[0], replacement[0], rows[2])
    assert replaced == ExactCoordinateBank.from_rows((rows[0], replacement[0], rows[2]))
    with pytest.raises(IndexError):
        bank.take(np.asarray((3,), dtype=np.int64))


def test_bank_model_recipe_is_its_row_recipe_and_restores_the_bank(
    tmp_path: Path,
) -> None:
    from phydrax._model._structure import model_structure_recipe
    from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
    from phydrax.discretization.fem import lagrange_element
    from phydrax.meshing._curving import _exact_coordinate_bank_source_map
    from phydrax.meshing._model_codec import (
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    rows = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(0), Fraction(0)),
        (Fraction(0), RadicalElement.sqrt(Fraction(2)) / 4, Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(3, 8)),
    )
    bank = ExactCoordinateBank.from_rows(rows)
    # The portable form is the former tuple field's recipe, byte for byte.
    assert model_structure_recipe(bank) == model_structure_recipe(rows)
    execution = np.asarray(
        [[float(value) for value in row] for row in rows], dtype=np.float64
    )
    mesh = CellMesh(
        execution,
        (CellBlock("cells", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int64)),),
    )
    geometry = CellGeometrySpec(
        {"cells": lagrange_element("tetrahedron", 1)},
        {"cells": np.arange(4, dtype=np.int64)[None, :]},
        execution,
        source_map=_exact_coordinate_bank_source_map(
            mesh, "source", "revision", ("bank",), rows, execution
        ),
        source_mesh=mesh,
    )
    assert geometry.exact_coordinate_bank == bank
    path = write_model_artifact(
        tmp_path / "geometry.phx", geometry, limits=MODEL_ARCHIVE_LIMITS
    )
    restored = read_model_artifact(
        path,
        expected_type=CellGeometrySpec,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert type(restored.exact_coordinate_bank) is ExactCoordinateBank
    assert restored.exact_coordinate_bank == bank


def test_bank_binary64_matches_float_of_every_entry() -> None:
    rng = np.random.default_rng(7)
    values = rng.standard_normal((64, 3)) * np.exp2(rng.integers(-1070, 1000, (64, 3)))
    values[0] = (0.0, -0.0, np.finfo(np.float64).smallest_subnormal)
    bank = ExactCoordinateBank.from_binary64(values)
    assert np.array_equal(bank.binary64(), values)
    assert tuple(bank) == tuple(tuple(Fraction(float(v)) for v in row) for row in values)
    wide = (
        Fraction((1 << 60) + 1, 1 << 1130),  # rounds into the subnormal range
        Fraction((1 << 80) - 1, 1 << 20),
        Fraction(1, 3),
    )
    exact = ExactCoordinateBank.from_rows((wide,))
    assert exact.binary64().tolist() == [[float(value) for value in wide]]
    with pytest.raises(OverflowError):
        ExactCoordinateBank.from_rows(((Fraction(1 << 1100),),)).binary64()


def test_dyadic_helpers_return_canonical_odd_significands() -> None:
    values = np.asarray(
        [[0.0, -0.0, 0.375, -6.0, 2.0**-1074, 2.0**1000]], dtype=np.float64
    )
    significands, exponents = binary64_dyadic_entries(values)
    assert significands.tolist() == [[0, 0, 3, -3, 1, 1]]
    assert exponents.tolist() == [[0, 0, -3, 1, -1074, 1000]]
    for value, significand, exponent in zip(
        values.reshape(-1), significands.reshape(-1), exponents.reshape(-1), strict=True
    ):
        assert dyadic_entry(Fraction(float(value))) == (int(significand), int(exponent))
    assert dyadic_entry(Fraction(1, 3)) is None


def test_binary64_bank_exact_rows_match_random_finite_bits() -> None:
    rng = np.random.default_rng(702)
    bits = rng.integers(0, np.iinfo(np.uint64).max, size=384, dtype=np.uint64)
    bits[:6] = np.asarray(
        [0, 1 << 63, 1, (1 << 63) | 1, 0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF],
        dtype=np.uint64,
    )
    values = bits.view(np.float64)
    values[~np.isfinite(values)] = 0.0
    matrix = values.reshape(-1, 3)
    bank = ExactCoordinateBank.from_binary64(matrix)
    expected = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in matrix
    )
    assert tuple(bank) == expected
    assert bank.coefficient_payloads() == tuple(
        tuple(coefficient_payload(value) for value in row) for row in expected
    )
    device_values = jnp.asarray(matrix, dtype=jnp.float64)
    represented = np.asarray(device_values, dtype=np.float64)
    assert tuple(ExactCoordinateBank.from_binary64(device_values)) == tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in represented
    )


def test_exact_bank_fingerprint_frames_shape_and_sparse_payload() -> None:
    a = ExactCoordinateBank.from_rows(
        ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(4)))
    )
    reshaped = ExactCoordinateBank.from_rows(
        ((Fraction(1),), (Fraction(2),), (Fraction(3),), (Fraction(4),))
    )
    same = ExactCoordinateBank.from_rows(
        ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(4)))
    )
    changed = ExactCoordinateBank.from_rows(
        ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(5)))
    )
    sparse = ExactCoordinateBank.from_rows(
        ((Fraction(1, 3), Fraction(2)), (Fraction(3), Fraction(4)))
    )
    moved_sparse = ExactCoordinateBank.from_rows(
        ((Fraction(1), Fraction(1, 3)), (Fraction(3), Fraction(4)))
    )
    assert a.content_fingerprint() == same.content_fingerprint()
    assert (
        len(
            {
                a.content_fingerprint(),
                reshaped.content_fingerprint(),
                changed.content_fingerprint(),
                sparse.content_fingerprint(),
                moved_sparse.content_fingerprint(),
            }
        )
        == 5
    )
