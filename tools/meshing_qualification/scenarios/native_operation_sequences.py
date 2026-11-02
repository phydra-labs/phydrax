"""Independently authored CAD sequences; no donor models or success labels.

All lengths are meters. Rectilinear references are signed, analytically clipped
boxes, not reconstructions from Boolean output. Curved references use analytic
integrals and an independently bounded Machin formula for pi.
"""

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import (
    boolean_brep,
    brep_box,
    brep_extrusion,
    brep_mass_properties,
    BRepModel,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._fillet import brep_fillet
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive

from ._rectilinear_oracle import Box, SignedBoxes, validate_rectilinear_result
from .native_cad_profiles import _roundtrip


def _box(low: tuple[float, float, float], high: tuple[float, float, float]) -> Box:
    return (
        (Fraction(low[0]), Fraction(low[1]), Fraction(low[2])),
        (Fraction(high[0]), Fraction(high[1]), Fraction(high[2])),
    )


def _solid(box: Box, name: str) -> BRepModel:
    points = np.asarray(box, dtype=np.float64)
    if any(
        Fraction(float(points[row, axis])) != box[row][axis]
        for row in range(2)
        for axis in range(3)
    ):
        raise ValueError(
            "Authored primitive coordinates must be exactly representable in binary64."
        )
    return brep_box(
        points[0],
        points[1],
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id=name,
    )


def _rectilinear_record(
    model: BRepModel, boxes: SignedBoxes, regions: int
) -> dict[str, object]:
    reference = validate_rectilinear_result(model, boxes)
    if model.topology.num_solids != regions:
        raise ValueError(
            "The operation changed the independently specified region count."
        )
    measure = brep_mass_properties(model, relative_tolerance=1e-10)
    if measure.status != ("certified",) * regions:
        raise ValueError("The source measure did not certify every material region.")
    bounds = np.asarray(measure.volume_enclosures)
    low = sum((Fraction(float(row[0])) for row in bounds), Fraction(0))
    high = sum((Fraction(float(row[1])) for row in bounds), Fraction(0))
    if not low <= reference.volume <= high:
        raise ValueError("The certified volume excludes the independent exact reference.")
    return {
        "source_revision": model.source_revision,
        "regions": regions,
        "volume_m3": str(reference.volume),
        "area_m2": str(reference.area),
        "first_moments_m4": tuple(map(str, reference.first_moments)),
        "second_moments_m5": tuple(map(str, reference.second_moments)),
        "whole_boundary_and_locality": "exact-oriented-plane-partition",
    }


def _enclosure_sequence() -> dict[str, object]:
    thickness = 1.0 / 32.0
    outer = _box((0.0, 0.0, 0.0), (4.0, 3.0, 2.0))
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))),
    )
    source = brep_extrusion(
        profile,
        (0.0, 0.0, 2.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="sequence-enclosure",
    )
    cavity = _box(
        (thickness, thickness, thickness),
        (4.0 - thickness, 3.0 - thickness, 2.0 + thickness),
    )
    clipped = _box(
        (thickness, thickness, thickness), (4.0 - thickness, 3.0 - thickness, 2.0)
    )
    first = boolean_brep(source, _solid(cavity, "sequence-cavity"), "difference").model
    first_reference: SignedBoxes = ((1, outer), (-1, clipped))
    first_record = _rectilinear_record(first, first_reference, 1)
    slot = _box((1.75, -thickness, 0.5), (2.25, 2.0 * thickness, 1.5))
    last = boolean_brep(first, _solid(slot, "sequence-wall-slot"), "difference").model
    reference = first_reference + ((-1, _box((1.75, 0.0, 0.5), (2.25, thickness, 1.5))),)
    record = _rectilinear_record(last, reference, 1)
    with TemporaryDirectory(prefix="phydrax-operation-sequence-") as temporary:
        directory = Path(temporary)
        archive = directory / "enclosure.phx"
        save_brep_archive(last, archive)
        restored = load_brep_archive(archive)
        cold_record = _rectilinear_record(restored, reference, 1)
        if restored.source_revision != last.source_revision:
            raise ValueError("The native source archive changed exact source identity.")
        step_model, step_record = _roundtrip(last, "enclosure", "step", directory)
        step_geometry = _rectilinear_record(step_model, reference, 1)
    return {
        "cavity": first_record,
        "wall_slot": record,
        "archive": cold_record,
        "step": step_record,
        "step_geometry": step_geometry,
    }


def _pattern_parameter_sequence() -> dict[str, object]:
    outer = _box((0.0, 0.0, 0.0), (4.0, 3.0, 1.0))
    first_hole = _box((0.75, 1.25, -1.0), (1.25, 1.75, 2.0))
    first = boolean_brep(
        _solid(outer, "pattern-plate"),
        _solid(first_hole, "pattern-hole-left"),
        "difference",
    ).model
    reference: SignedBoxes = (
        (1, outer),
        (-1, _box((0.75, 1.25, 0.0), (1.25, 1.75, 1.0))),
    )
    records: dict[str, object] = {"first_hole": _rectilinear_record(first, reference, 1)}
    for name, half_width in (("authored", 0.25), ("parameter_edit", 0.375)):
        hole = _box(
            (3.0 - half_width, 1.5 - half_width, -1.0),
            (3.0 + half_width, 1.5 + half_width, 2.0),
        )
        result = boolean_brep(
            first, _solid(hole, "pattern-hole-right"), "difference"
        ).model
        clipped = _box(
            (3.0 - half_width, 1.5 - half_width, 0.0),
            (3.0 + half_width, 1.5 + half_width, 1.0),
        )
        records[name] = _rectilinear_record(result, reference + ((-1, clipped),), 1)
    return records


def _split_sequence() -> dict[str, object]:
    outer = _box((0.0, 0.0, 0.0), (4.0, 1.0, 1.0))
    source = _solid(outer, "split-source")
    tool = _solid(_box((1.5, -1.0, -1.0), (2.5, 2.0, 2.0)), "split-tool")
    result = boolean_brep(source, tool, "difference").model
    return _rectilinear_record(
        result,
        (
            (1, _box((0.0, 0.0, 0.0), (1.5, 1.0, 1.0))),
            (1, _box((2.5, 0.0, 0.0), (4.0, 1.0, 1.0))),
        ),
        2,
    )


def _contact_algebra_sequence() -> dict[str, object]:
    first_box = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    touching_box = _box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    first = _solid(first_box, "contact-left")
    touching = _solid(touching_box, "contact-right")
    joined = boolean_brep(first, touching, "union")
    union_record = _rectilinear_record(
        joined.model, ((1, first_box), (1, touching_box)), 1
    )
    difference = boolean_brep(first, touching, "difference").model
    unchanged = _rectilinear_record(difference, ((1, first_box),), 1)
    coincident = boolean_brep(
        first, _solid(first_box, "contact-independent-copy"), "intersection"
    )
    idempotent = _rectilinear_record(coincident.model, ((1, first_box),), 1)
    recovered = boolean_brep(joined.model, touching, "difference").model
    algebra = _rectilinear_record(recovered, ((1, first_box),), 1)
    return {
        "face_contact_union": union_record,
        "tangent_difference": unchanged,
        "coincident_intersection": idempotent,
        "union_difference_identity": algebra,
    }


def _pi_bounds() -> tuple[Fraction, Fraction]:
    def arctangent(inverse: int) -> tuple[Fraction, Fraction]:
        x = Fraction(1, inverse)
        total, power, index = Fraction(0), x, 0
        while True:
            total += (-1 if index % 2 else 1) * power / (2 * index + 1)
            index += 1
            power *= x * x
            remainder = power / (2 * index + 1)
            if remainder < Fraction(1, 2**200):
                other = total + (-1 if index % 2 else 1) * remainder
                return min(total, other), max(total, other)

    first, second = arctangent(5), arctangent(239)
    return 16 * first[0] - 4 * second[1], 16 * first[1] - 4 * second[0]


def _curved_measure(
    model: BRepModel, constant: Fraction, pi_factor: Fraction, regions: int
) -> dict[str, object]:
    if model.topology.num_solids != regions:
        raise ValueError("Curved modification changed the analytic region count.")
    measure = brep_mass_properties(model, relative_tolerance=1e-10)
    if measure.status != ("certified",) * regions:
        raise ValueError("Curved source volume remained uncertified.")
    bounds = np.asarray(measure.volume_enclosures)
    low = sum((Fraction(float(row[0])) for row in bounds), Fraction(0))
    high = sum((Fraction(float(row[1])) for row in bounds), Fraction(0))
    reference = tuple(constant + pi_factor * value for value in _pi_bounds())
    if not low <= min(reference) <= max(reference) <= high:
        raise ValueError(
            "Certified curved volume excludes the independently bounded analytic integral."
        )
    if high - low > max(reference) / 10**10:
        raise ValueError(
            "Curved source measure exceeds its declared relative enclosure width."
        )
    return {
        "source_revision": model.source_revision,
        "regions": regions,
        "volume_enclosure_m3": [str(low), str(high)],
        "analytic_constant": str(constant),
        "analytic_pi_factor": str(pi_factor),
    }


def _blend_cut_sequence() -> dict[str, object]:
    source = _solid(_box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0)), "blend-source")
    geometry = source.geometry
    # The independently authored fixture chooses this explicit physical edge,
    # then the operation receives its revision-qualified native entity identity.
    candidates = tuple(
        index
        for index, (first, last) in enumerate(geometry.edge_vertices)
        if {geometry.vertex_exact_points[first], geometry.vertex_exact_points[last]}
        == {(0, 0, 0), (0, 0, 4)}
    )
    if len(candidates) != 1:
        raise ValueError("The authored corner must identify one source edge.")
    selected = source.edge_ids[candidates[0]]
    blended = brep_fillet(source, (selected,), 0.25)
    if blended.status != "certified" or blended.model is None:
        raise RuntimeError(f"Native blend did not certify: {blended.reason}")
    first = _curved_measure(blended.model, Fraction(95, 4), Fraction(1, 16), 1)
    tool = _solid(_box((-1.0, -1.0, 2.0), (3.0, 4.0, 3.0)), "blend-splitter")
    cut = boolean_brep(blended.model, tool, "difference").model
    second = _curved_measure(cut, Fraction(285, 16), Fraction(3, 64), 2)
    return {"fillet": first, "mixed_plane_cylinder_cut": second}


_CASES: tuple[tuple[str, Callable[[], dict[str, object]]], ...] = (
    ("thin-enclosure-and-wall-slot", _enclosure_sequence),
    ("patterned-holes-and-parameter-edit", _pattern_parameter_sequence),
    ("body-splitting-cut", _split_sequence),
    ("tangent-coincident-and-algebra", _contact_algebra_sequence),
    ("blend-then-mixed-curved-cut", _blend_cut_sequence),
)


def native_operation_sequences() -> dict[str, object]:
    """Run every authored sequence; report real failures without dropping cases."""
    records: dict[str, object] = {}
    failed = 0
    for name, run in _CASES:
        try:
            records[name] = {"status": "passed", "evidence": run()}
        except Exception as error:
            # Qualification must retain each independent case's failure evidence.
            failed += 1
            records[name] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "detail": str(error),
            }
    return {
        "scenario": "native-cad-operation-sequences",
        "workflow": "native-cad",
        "status": "failed" if failed else "passed",
        "mandatory_workflow_completion": "incomplete" if failed else "complete",
        "cases": records,
        "provenance": {
            "origin": "Independently authored PHYDRA meter-valued operation sequences",
            "reference": "Analytically clipped signed boxes, exact boundary chains, signed tetrahedron moments, and Machin-bounded analytic curved volumes",
            "idea_source": "https://github.com/tangency-labs/solidcraft/blob/147d1cfaddcc6fe824399a58fea36a022278961e/docs/oracle.md",
            "copied_code_or_assets": False,
        },
        "nonclaims": [
            "Scalar oracles do not replace source fidelity, embedding, coverage, or material ownership proofs.",
            "No optional CAD engine or donor success labels are used.",
        ],
    }
