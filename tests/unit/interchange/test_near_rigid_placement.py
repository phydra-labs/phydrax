#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Placement acceptance with independent high-precision direction and circle oracles."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._boolean_core import BRepBooleanFailure
from phydrax.geometry.brep._constructors import brep_cylinder
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._sewing import BRepSewingFailure, sew_brep
from phydrax.interchange._cad import CadInterchangeError
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.interchange._cad_carriers import RigidPlacement
from phydrax.interchange._step import decode_step_bytes, write_step
from tests._support.cad_models import CONTRACT, native_solid
from tests.unit.interchange.test_step import POLICY


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=256 << 20) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000_000,
            maximum_memory_bytes=256 << 20,
        )
        with budget.activate():
            yield budget


def _assembly(
    model: BRepModel,
    path: Path,
    frames: tuple[np.ndarray, ...],
    *,
    scale: float = 1.0,
) -> bytes:
    write_step(model, path)
    text = path.read_text()
    representation = re.search(
        r"#(\d+)=ADVANCED_BREP_SHAPE_REPRESENTATION\((.*)\);", text
    )
    if representation is None:
        raise ValueError(
            "PlacementFixture.representation must contain one solid; got none."
        )
    identifier, body = representation.groups()
    context = re.search(r",(#\d+)$", body)
    if context is None:
        raise ValueError("PlacementFixture.context must be declared; got none.")
    next_id = max(int(value) for value in re.findall(r"#(\d+)=", text)) + 1
    records = []

    def add(record: str) -> str:
        nonlocal next_id
        reference = f"#{next_id}"
        next_id += 1
        records.append(f"{reference}={record};")
        return reference

    definition = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({body})")
    origin = add("CARTESIAN_POINT('',(0.,0.,0.))")
    frame = add(f"AXIS2_PLACEMENT_3D('',{origin},$,$)")
    mapping = add(f"REPRESENTATION_MAP({frame},{definition})")
    items = []
    for rotation in frames:
        directions = [
            add("DIRECTION('',(" + ",".join(repr(float(value)) for value in axis) + "))")
            for axis in rotation.T
        ]
        operator = add(
            f"CARTESIAN_TRANSFORMATION_OPERATOR_3D('',{directions[0]},{directions[1]},"
            f"{origin},{scale},{directions[2]})"
        )
        items.append(add(f"MAPPED_ITEM('',{mapping},{operator})"))
    replacement = (
        f"#{identifier}=SHAPE_REPRESENTATION('',({','.join(items)}),{context.group(1)});"
    )
    text = text[: representation.start()] + replacement + text[representation.end() :]
    return text.replace(
        "ENDSEC;\nEND-ISO-10303-21;", "\n".join(records) + "\nENDSEC;\nEND-ISO-10303-21;"
    ).encode()


def _import(data: bytes, maximum_deviation: float = 1.0e-12) -> BRepModel:
    with _budget() as budget:
        return decode_step_bytes(
            data,
            replace(POLICY, maximum_placement_deviation=maximum_deviation),
            coordinate_budget=budget,
        ).model


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _reference_frame(frame: np.ndarray) -> tuple[tuple[Decimal, ...], ...]:
    """Use decimal projection, independently of the rational triple-cross construction."""
    with localcontext() as context:
        context.prec = 100
        z = tuple(_decimal(value) for value in frame[:, 2])
        z_norm = sum((value * value for value in z), Decimal(0)).sqrt()
        z = tuple(value / z_norm for value in z)
        first = tuple(_decimal(value) for value in frame[:, 0])
        projection = sum(a * b for a, b in zip(first, z, strict=True))
        x = tuple(a - projection * b for a, b in zip(first, z, strict=True))
        x_norm = sum((value * value for value in x), Decimal(0)).sqrt()
        x = tuple(value / x_norm for value in x)
        y = (
            z[1] * x[2] - z[2] * x[1],
            z[2] * x[0] - z[0] * x[2],
            z[0] * x[1] - z[1] * x[0],
        )
        return tuple((x[row], y[row], z[row]) for row in range(3))


def test_thirty_degree_assembly_has_certified_polar_defect(tmp_path: Path) -> None:
    cosine = np.sqrt(3.0) / 2.0
    frame = np.asarray(((cosine, -0.5, 0.0), (0.5, cosine, 0.0), (0.0, 0.0, 1.0)))
    model = _import(_assembly(native_solid("box"), tmp_path / "thirty.step", (frame,)))
    occurrence = model.require_geometry().occurrences[0]
    evidence = occurrence.placement_evidence
    assert evidence is not None and evidence.point_deviation_bound > 0.0
    with localcontext() as context:
        context.prec = 100
        rotation = np.asarray(occurrence.rotation)
        norm = (_decimal(rotation[0, 0]) ** 2 + _decimal(rotation[1, 0]) ** 2).sqrt()
        defect = abs(norm - 1)
        assert defect <= _decimal(evidence.rotation_deviation_bound)
        assert _decimal(evidence.point_deviation_bound) >= (
            _decimal(evidence.rotation_deviation_bound) * _decimal(evidence.model_extent)
        )
    assert evidence.point_deviation_bound <= 1.0e-12


def test_oblique_assembly_encloses_independent_frame(tmp_path: Path) -> None:
    axis = np.asarray((1.0, 2.0, 3.0)) / np.sqrt(14.0)
    cosine, sine = np.sqrt(3.0) / 2.0, 0.5
    skew = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    oblique = cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis, axis) + sine * skew
    twist = np.asarray(((0.5, -cosine, 0.0), (cosine, 0.5, 0.0), (0.0, 0.0, 1.0)))
    frame = twist @ oblique
    model = _import(_assembly(native_solid("box"), tmp_path / "oblique.step", (frame,)))
    occurrence = model.require_geometry().occurrences[0]
    evidence = occurrence.placement_evidence
    assert evidence is not None
    reference = _reference_frame(frame)
    with localcontext() as context:
        context.prec = 100
        squared = sum(
            (
                (_decimal(occurrence.rotation[row][column]) - reference[row][column]) ** 2
                for row in range(3)
                for column in range(3)
            ),
            Decimal(0),
        )
        assert squared.sqrt() <= _decimal(evidence.rotation_deviation_bound)
        assert evidence.entrywise_deviation_bounds is not None
        for row in range(3):
            for column in range(3):
                error = abs(
                    _decimal(occurrence.rotation[row][column]) - reference[row][column]
                )
                assert error <= _decimal(evidence.entrywise_deviation_bounds[row][column])
        point = (Decimal(1), Decimal(2), Decimal(3))
        ideal = tuple(
            sum(reference[row][column] * point[column] for column in range(3))
            for row in range(3)
        )
        placed = occurrence.place(np.asarray((1.0, 2.0, 3.0)))
        distance = sum(
            ((_decimal(placed[row]) - ideal[row]) ** 2 for row in range(3)),
            Decimal(0),
        ).sqrt()
        assert distance <= _decimal(evidence.point_deviation_bound)


def test_six_hole_polar_pattern_contains_closed_form_positions(tmp_path: Path) -> None:
    radical = np.sqrt(3.0) / 2.0
    coefficients = (
        (1.0, 0.0),
        (0.5, radical),
        (-0.5, radical),
        (-1.0, 0.0),
        (-0.5, -radical),
        (0.5, -radical),
    )
    frames = tuple(
        np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))) for c, s in coefficients
    )
    hole = brep_cylinder(
        0.25, 1.0, base_center=(4.0, 0.0, 0.0), coordinate_contract=CONTRACT
    )
    model = _import(_assembly(hole, tmp_path / "polar.step", frames))
    occurrences = model.require_geometry().occurrences
    assert len(occurrences) == 6
    assert len({occurrence.solid for occurrence in occurrences}) == 1
    with localcontext() as context:
        context.prec = 100
        root = Decimal(3).sqrt() / 2
        exact = (
            (Decimal(1), Decimal(0)),
            (Decimal("0.5"), root),
            (Decimal("-0.5"), root),
            (Decimal(-1), Decimal(0)),
            (Decimal("-0.5"), -root),
            (Decimal("0.5"), -root),
        )
        for occurrence, (c, s) in zip(occurrences, exact, strict=True):
            placed = occurrence.place(np.asarray((4.0, 0.0, 0.5)))
            distance = (
                (_decimal(placed[0]) - 4 * c) ** 2 + (_decimal(placed[1]) - 4 * s) ** 2
            ).sqrt()
            bound = (
                0.0
                if occurrence.placement_evidence is None
                else occurrence.placement_evidence.point_deviation_bound
            )
            assert distance <= _decimal(bound)


def test_scale_one_plus_micro_refuses(tmp_path: Path) -> None:
    data = _assembly(
        native_solid("box"), tmp_path / "scaled.step", (np.eye(3),), scale=1.0 + 1.0e-6
    )
    with pytest.raises(CadInterchangeError) as caught:
        _import(data)
    assert caught.value.refusal.reason == "unsupported-entity"
    matrix = np.concatenate(((1.0 + 1.0e-6) * np.eye(3), np.zeros((3, 1))), axis=1)
    with _budget() as budget, pytest.raises(CadInterchangeError):
        RigidPlacement.from_certified_matrix(
            matrix, "scaled", maximum_deviation=1.0, coordinate_budget=budget
        )


def test_default_placement_limit_and_required_budget(tmp_path: Path) -> None:
    c = np.sqrt(3.0) / 2.0
    frame = np.asarray(((c, -0.5, 0.0), (0.5, c, 0.0), (0.0, 0.0, 1.0)))
    data = _assembly(native_solid("box"), tmp_path / "default.step", (frame,))
    with pytest.raises(CadInterchangeError) as caught:
        decode_step_bytes(data, POLICY)
    assert caught.value.refusal.reason == "unsupported-entity"
    with pytest.raises(CadInterchangeError, match="coordinate_budget"):
        decode_step_bytes(data, replace(POLICY, maximum_placement_deviation=1.0e-12))
    with pytest.raises(CadInterchangeError) as caught:
        _import(data, maximum_deviation=1.0e-20)
    assert caught.value.refusal.reason == "unsupported-entity"


def test_watertight_consumers_refuse_placement_deviation(tmp_path: Path) -> None:
    c = np.sqrt(3.0) / 2.0
    frame = np.asarray(((c, -0.5, 0.0), (0.5, c, 0.0), (0.0, 0.0, 1.0)))
    model = _import(_assembly(native_solid("box"), tmp_path / "consumer.step", (frame,)))
    with pytest.raises(BRepBooleanFailure, match="placement-deviation"):
        boolean_brep(model, native_solid("box"), "union")
    with pytest.raises(BRepSewingFailure, match="placement-deviation"):
        sew_brep(model.require_geometry(), model.orientation, model.topology.solid_faces)


def test_direct_shear_and_reflection_refuse() -> None:
    shear = np.asarray(((1.0, 0.01, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    for rotation in (shear, np.diag((1.0, 1.0, -1.0))):
        matrix = np.concatenate((rotation, np.zeros((3, 1))), axis=1)
        with _budget() as budget, pytest.raises(CadInterchangeError):
            RigidPlacement.from_certified_matrix(
                matrix, "nonrigid", maximum_deviation=1.0, coordinate_budget=budget
            )


def test_matrix_gram_bound_uses_rational_defect() -> None:
    c = np.sqrt(3.0) / 2.0
    matrix = np.asarray(((c, -0.5, 0.0, 0.0), (0.5, c, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)))
    with _budget() as budget:
        placement = RigidPlacement.from_certified_matrix(
            matrix, "rounded", maximum_deviation=1.0e-12, coordinate_budget=budget
        )
    defect = abs(Fraction(float(c)) ** 2 + Fraction(0.5) ** 2 - 1)
    assert Fraction(placement.rotation_deviation_bound) >= defect


def test_placement_archive_preserves_evidence_identity(tmp_path: Path) -> None:
    c = np.sqrt(3.0) / 2.0
    frame = np.asarray(((c, -0.5, 0.0), (0.5, c, 0.0), (0.0, 0.0, 1.0)))
    model = _import(_assembly(native_solid("box"), tmp_path / "archive.step", (frame,)))
    receipt = save_brep_archive(model, tmp_path / "placed.phx")
    restored = load_brep_archive(receipt.path)
    assert restored.model_id == model.model_id
    assert restored.require_geometry().geometry_id == model.require_geometry().geometry_id
    old = model.require_geometry().occurrences[0].placement_evidence
    new = restored.require_geometry().occurrences[0].placement_evidence
    assert old is not None and new is not None
    assert old.payload() == new.payload()


def test_prechange_no_gap_archive_retains_identity() -> None:
    baseline = Path(__file__).parents[2] / "data" / "brep_no_gap_prechange.phx"
    restored = load_brep_archive(baseline)
    assert (
        restored.model_id
        == "0d4aedd5ac4e2fe989ad244a6d0969eef42fe6393d263eb4242427ee7522181b"
    )
    assert all(
        occurrence.placement_evidence is None
        for occurrence in restored.require_geometry().occurrences
    )


def test_composed_source_poses_enclose_rotation_and_translation() -> None:
    first = np.stack(
        (np.asarray((1.0, 2.0, 3.0)), np.zeros(3), np.asarray((3.0, -1.0, 2.0))), axis=1
    )
    second = np.stack(
        (np.asarray((2.0, -1.0, 1.0)), np.zeros(3), np.asarray((1.0, 4.0, 2.0))), axis=1
    )
    with _budget() as budget:
        outer = RigidPlacement.from_source_directions(
            first[:, 0], first[:, 2], np.asarray((1.0, 2.0, 3.0)), "outer", budget
        )
        inner = RigidPlacement.from_source_directions(
            second[:, 0], second[:, 2], np.asarray((4.0, 5.0, 6.0)), "inner", budget
        )
        composed = outer.compose(inner, coordinate_budget=budget)
        inverse = composed.inverse(coordinate_budget=budget)
    r_first, r_second = _reference_frame(first), _reference_frame(second)
    with localcontext() as context:
        context.prec = 100
        rotation = tuple(
            tuple(
                sum(
                    (r_first[row][axis] * r_second[axis][column] for axis in range(3)),
                    Decimal(0),
                )
                for column in range(3)
            )
            for row in range(3)
        )
        translation = tuple(
            sum(
                (r_first[row][axis] * Decimal(axis + 4) for axis in range(3)),
                Decimal(row + 1),
            )
            for row in range(3)
        )
        rotation_error = sum(
            (
                (_decimal(composed.rotation[row, column]) - rotation[row][column]) ** 2
                for row in range(3)
                for column in range(3)
            ),
            Decimal(0),
        ).sqrt()
        translation_error = sum(
            (
                (_decimal(composed.translation[row]) - translation[row]) ** 2
                for row in range(3)
            ),
            Decimal(0),
        ).sqrt()
        assert rotation_error <= _decimal(composed.rotation_deviation_bound)
        assert translation_error <= _decimal(composed.translation_deviation_bound)
        ideal_inverse_translation = tuple(
            -sum(
                (rotation[axis][row] * translation[axis] for axis in range(3)), Decimal(0)
            )
            for row in range(3)
        )
        inverse_error = sum(
            (
                (_decimal(inverse.translation[row]) - ideal_inverse_translation[row]) ** 2
                for row in range(3)
            ),
            Decimal(0),
        ).sqrt()
        assert inverse_error <= _decimal(inverse.translation_deviation_bound)
