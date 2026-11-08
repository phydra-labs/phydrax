#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from phydrax.discretization._exact_algebraic_geometry import (
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
