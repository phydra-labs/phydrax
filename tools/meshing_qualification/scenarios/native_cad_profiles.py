from __future__ import annotations

import base64
import gzip
import hashlib
import math
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    reference_cell_topology,
)
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._measure import brep_mass_properties
from phydrax.meshing._association import GeometryAssociationArchiveRecipe
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider

from .._runtime import accepted_result_record, single_cell_kind


_SOURCE_BRANCH = "feature/native-meshing-20260929"
_SOURCE_COMMIT = "c56614a9fe52c4a8f56484874e91fde6498ec9ff"
_SOURCE_REVISION = "tools/meshing_qualification/scenarios/cad.py"
_RESOLUTION = 4
_VOLUME_RELATIVE_TOLERANCE = 1.0e-10
_SOURCE_DEVIATION_TOLERANCE_M = 0.02
_FE_RESIDUAL_TOLERANCE = 2.0e-8
_FE_BOUNDARY_TOLERANCE = 1.0e-10
# The donor CAD volume profiles ran the donor default NativeVolumeSchedule,
# whose radius-edge and dihedral targets were aims: unmet cells were published
# as evidence, never hidden (c56614 phydrax/meshing/_volume_generation.py
# 453-455, publication 5117-5139), and donor cad.py 2146-2148 recorded
# tetrahedral quality as evidence only. Soft shape strength keeps those
# identical targets and publishes every unmet row; validity, fidelity,
# coverage and the FE gates stay hard.
_CAD_VOLUME_SCHEDULE_CITATIONS = (
    f"{_SOURCE_COMMIT}:phydrax/meshing/_volume_generation.py:453-455",
    f"{_SOURCE_COMMIT}:phydrax/meshing/_volume_generation.py:5117-5139",
    f"{_SOURCE_COMMIT}:{_SOURCE_REVISION}:2146-2148",
)

# Byte-for-byte source data from the donor NIST FTC-11 STEP fixture. Keeping
# this licensed, SHA-checked source here preserves the external-input case
# without a second fixture file or an optional geometry-engine dependency.
_NIST_STEP_GZIP_B64 = """H4sIAAAAAAAAA61ZXW/iSBZ9X2n/A1JGMowAub7tGfHg2BViyWBkO9nuJ4tNSA+rBFoJ8zD/fm+5yh9gV+jp3XTjEHzr1r3nnlPlW8R5OkMucckMo9//+Y97GUQygzd3cSLLSOZhFm+KOF2Px85dEc4QGn3bHd92p/e/RsfD61+jl/fj2+j0x260jvNitLqVo80qHj1uX/fP29P+eBhtD8+j8Hh4Ob6/bQ9Pu1Gx+zjtD99Gm/fjf3ZPJ2cydfDvyJnUc66DlRw7h/3HqXw5PZUIlduPtx2af5y+O2DqIjZz2QyLAonfKPmNCWc6dsBLdXEc/b/xlof3chVA8GG6vouXJfwqsjRRmcXLtTNRhnId5TKEN1FQBPDrhi2CzSaJw0AlXg2RX4qx8wRZ7L/9+a7zgr9O78fX193ziESj593H/tvhY3R8Gb3tnv7YHvZP29fR9+376aNCYPvxsXv79+t+91HFdsPPpthkaZGGVVh38Tqu8Hbiw2n3fqgmA1f5Cdxs358hu+137JLyLJqyjaYkz6WOpjy+lG0wZRVMCV7KNpjybf9Wvr44U+y6dHrDqtjEAuKJHsKiTd2Be1OndaaT8GrDgXpBnAOf/TId34gK9Bu/mSWTSVDIqGxmhb+WafZ17KiQ9SBPD0JuM6qFqrxLs1WgQVPWYFzZoiHbJiXlG8r2sj/sFYI6Qw2czg7hgfFmBuTCC2kzbzFWnI3KBzAZ/zrZJMFalsF6CfSrPprksX7zy3SeBVEcrOcmG38RxSu5zsFxkJTyyyZdy3WRj925O7W8qoEYYOjMspJB/pDJ8l9xca9nGrirnCJBGcE+ZhMIX6OE6WIMmDzKTEVR3gZ5nYkTyWUmJQCD3UknQwh7IEft7BKLTuZ5Ievk8zSJo/5oghfjRK6XdRKTQVfzVZwk8Xw6X8kikwZJQhcP61BmRRCvi68DgBi/XSwAA4KnTgTrVgBjyyAMH7Ig/Fo+BsmDrFeRG8IX46VM1WxxCFzdZDKHIp2vDWSyTNJbqGE3iiBXawxkUFuNIc5JaxkXQyYYdIg9FdtkYpnNWR8POwiw+mUAEIvgS5zjEioTSuBUUZKo0i1RvvzpDdXUId4iDLIClr8A1p0UAlVWXcYZh4qbmQxr0jcWqLag7oUFuvRB0SK/DzbyArXKGCJWBeDacEhqZTV2rJd0UKO2JMZlx+7CO3iDbLU4qf/T4mTuT4qTof9dnNTXrth1cTJ0Jk7m2sTJLrH4W+Lk5KfFydnfFqf6UQLl5LpAufg/CpSzqwJlDED3VWx2gUJwJDLS5J8rbubOBTOWVzUnLjVXeUBd0QlkWwk4rAQcAheapwIvwq9JvI4ANkgXoL+DAZWlQFPEDZ0FsYY/U0Zqeuqf/ZhA6AKIC3i0o8CZvsU+h8TmkV/FR1zHx7PhI6CwgsNLaEN/EcZZmBhIvBYSz13IaClLoOOjuUvNy5/Oi7k2Qos0i8E9kEdZK7tf4R+Mns7vjA3WjpI03ehV0UM6So8sVDXK9AGUWd6mD2sdo4fbCeiVwjQpe6xfCY/qW/zHtgNvCNgz6D3vKvSeb4PeA9g9AS/9UOK7Xeg9v4XeR5fQe0y/fLdBxscW6H3U2pBL6H1sHkyphr4F3SftMOgNoke1HEVlLReoGgEjCuuVwE1tfTuyM0xA8wocI3zfhu6sgde/hHfW2259K7w+wOsDvL55MnbN86O+6ZsPkT1gAqFW8brdH1Q/lOM+v8CduWlfP0yGFqf0GuOQy65SDrncBgpEpi5UXZgxFl3ewdipStzc8y6ZB3k3F9EwBLm+hX3gomEHQu4l/2CgiRk6l0Hxw6B2GmQjudc16rEcxtWz9HgO5p2hg0yH2JSd4jqwqGP9+TrS2eQQurqUIHR9LUHIynYIRl2EuhjCQ8Nk2+vAUafK+JoI2gDwEOuxYT3+nPUdL9dpjn+A5thOc6xorroKcGSMz2mOuzTHfZpj3Fw6NMdWmuMOzUmf5rimObHRnHSIRWw0r9TUWPV5Tmqekz7PSYfnZJjnRPGcVDzH3XCGeQ7VwD5xsZAzF7EpnfuEEszUXxyqRHFNfnKd/OSS/L2FHhE794niPlHcJ4b7tLvYw1Dz6TDRZ9ibM44oF4IjH3lUqQOexoQQSOfWzYYOScC0Xohek0DHj00E7f6H6KUK+rBQuwioEgFVIqCG6PRcBJRP+6kby74kVItpLh1JUKskaEcSrC8JWkuC2STBOhxk2MJBD2jnQQ+Jied6iHkInjTmBAExB2rHyEDtGDY37c+Wl7VjQ6vTOaPZ5XN7v3TMen4BsagLUxdujL2z0jEx7aduLP1e6RhpLp0FhLu20kGr15SOo17peJ0Bx71lhqPOBGRwmWFqmeFYLTO0U2J+Df9m1uvo8+voczv6XKHPFfrcoA/9bJFmaRxdbqRcTIme2kiMf7pF1LmKvh54rQdh04PogCWsWwTtTtPfIkS9RYj+FiE6W4QY3iKEqp2otgjenegHWyo02Kye105c3wuEfS9QzSzMoi5mL4DOdbh20LgiAY83Te2s7WsliDrXfgOL6g4W2VpY1OlhYZX9pFWrZQf9a28WWs/Ce6WDbrD1LwZLpxo2GFodg3eYBD1smKTqkC2/l0li+kJFfqQupJKBGqqOCsB3HYO/WAXr+C5NYGB1fHabSYg0P77un1EVksFfNbZpkj5kZba8HTvR7mX75+tpjCYOVFRQ30U+x9iDBwlCBz9hHiWwvTMXc+Fis0FBU3wXJ0kZZDIo8+JrAgteNcv5DGp6MwBfDji3VGf8dRmhUTZMMa6bkcrK7BfQMzdGcSRtPkntk134fMgDqP38Ni3u5ypOauz44ux8TRvrMznF9LFyWm9E0ERX96MyLuTKFNrXNa6B8hYrGd5DrVQXoL8DLNuzw7O5hs6sYY720Bqpfrvmlip4OXzYHdzmmSad3w7G0IIPmesvw+BNfh8rBmVZfc6vYPHrg+3mK0t4M4vPv8L9LzNuH1XSHQAA"""
_NIST_STEP_SHA256 = "a5abfea460499dc31bc812aca68917948a117565fea4f5a080e36eb32f39650d"

_PROFILE_CASES = (
    ("native-cad-lifecycle:brep", "brep", "planar-box"),
    ("native-cad-lifecycle:brep:external-combined", "brep", "external-combined"),
    ("native-cad-lifecycle:brep:external-routine", "brep", "external-routine"),
    ("native-cad-lifecycle:brep:quadric-larger", "brep", "quadric-larger"),
    ("native-cad-lifecycle:brep:quadric-routine", "brep", "quadric-routine"),
    ("native-cad-lifecycle:iges", "iges", "planar-box"),
    ("native-cad-lifecycle:iges:external-combined", "iges", "external-combined"),
    ("native-cad-lifecycle:iges:external-routine", "iges", "external-routine"),
    ("native-cad-lifecycle:iges:quadric-larger", "iges", "quadric-larger"),
    ("native-cad-lifecycle:iges:quadric-routine", "iges", "quadric-routine"),
    ("native-cad-lifecycle:step", "step", "planar-box"),
    ("native-cad-lifecycle:step:external-combined", "step", "external-combined"),
    ("native-cad-lifecycle:step:external-routine", "step", "external-routine"),
    ("native-cad-lifecycle:step:quadric-larger", "step", "quadric-larger"),
    ("native-cad-lifecycle:step:quadric-routine", "step", "quadric-routine"),
)

_FIXTURE_PROVENANCE = {
    "planar-box": {
        "record_id": "fd939176a2a980d232b2317dd07cb734fdb8d015a1a7a8a5329a1b6f77f226a9",
        "fixture_path": "tests/data/cad/native_authored_boxes.json",
        "fixture_sha256": "04682ad36c95c1086ffd606d0a2a8904f67ab811c843460062b3218261ef65e0",
        "license": "LicenseRef-PHYDRA-Proprietary",
        "origin": "PHYDRA native meshing qualification corpus; independently authored meter-valued nested boxes",
        "input": "BRep box [0,0,0]-[1,1,1] intersected with [0.25,0,0]-[0.75,1,1]; selected top face z=1",
    },
    "quadric-routine": {
        "record_id": "c575556a201da438cbe2a17a3092bc93b7f6f45d3bec4ceb54eb89ea4da35d06",
        "fixture_path": "tests/data/cad/native_authored_quadrics.json",
        "fixture_sha256": "dc3c43488e9bd0702fa50ff1140ff7c1d1cc64ada29bc27002ad919b85becbae",
        "license": "LicenseRef-PHYDRA-Proprietary",
        "origin": "PHYDRA native meshing qualification corpus; independently authored analytic quadric coordinates in meters",
        "input": "Intersection of z-axis cylinder (R=1, H=2.4, base=(0,0,-1.2)) and x-axis cylinder (r=0.6, H=3, base=(-1.5,0,0)); one convex quartic solid",
    },
    "external-routine": {
        "record_id": "f6ca1aac3f6626ad1253906e43861e6096aa0775e93e824e31e243c9490f9c0f",
        "fixture_path": "tests/data/cad/nist_ftc_11_external.json",
        "fixture_sha256": "59edee958c5d8e53a2c56aaacf1a2cd24803498d03b35469c7b620bc4d67cfae",
        "license": "NIST statement: test cases, CAD models, and STEP files can be used without restrictions; acknowledgement requested",
        "origin": "NIST MBE PMI Validation and Conformance Testing Project, FTC-11 AP203 geometry-only source; retrieved 2026-10-20, unmodified archive member",
        "archive_url": "https://www.nist.gov/system/files/documents/noindex/2024/06/19/NIST-PMI-STEP-Files.zip",
        "archive_sha256": "8fa78429e6d8d9b0d7681d223b6aa9ec98c3772185c55b1a0e3679b21c181911",
        "archive_member": "NIST-PMI-STEP-Files/AP203 geometry only/nist_ftc_11_asme1_rb.stp",
        "source_path": "tests/data/cad/nist_ftc_11_asme1_rb.stp",
        "source_sha256": _NIST_STEP_SHA256,
        "source_length_unit_meters": 0.001,
        "input": "Unmodified NIST FTC-11 STEP washer, initially parsed as native STEP then re-encoded in the requested CAD format",
    },
    "external-combined": {
        "record_id": "f6ca1aac3f6626ad1253906e43861e6096aa0775e93e824e31e243c9490f9c0f",
        "fixture_path": "tests/data/cad/nist_ftc_11_external.json",
        "fixture_sha256": "59edee958c5d8e53a2c56aaacf1a2cd24803498d03b35469c7b620bc4d67cfae",
        "license": "NIST statement: test cases, CAD models, and STEP files can be used without restrictions; acknowledgement requested",
        "origin": "NIST MBE PMI Validation and Conformance Testing Project, FTC-11 AP203 geometry-only source; retrieved 2026-10-20, unmodified archive member, plus the original PHYDRA-authored radial bore",
        "archive_url": "https://www.nist.gov/system/files/documents/noindex/2024/06/19/NIST-PMI-STEP-Files.zip",
        "archive_sha256": "8fa78429e6d8d9b0d7681d223b6aa9ec98c3772185c55b1a0e3679b21c181911",
        "archive_member": "NIST-PMI-STEP-Files/AP203 geometry only/nist_ftc_11_asme1_rb.stp",
        "source_path": "tests/data/cad/nist_ftc_11_asme1_rb.stp",
        "source_sha256": _NIST_STEP_SHA256,
        "source_length_unit_meters": 0.001,
        "authored_operand": "difference with r=0.0004 m, axis=(1,0,0), base=(-0.04,0,-0.00075) m, height=0.08 m",
        "input": "Unmodified NIST FTC-11 STEP washer minus the original PHYDRA-authored radial cylinder; native Boolean difference",
    },
}


_FORMAT_WRITERS = {
    "brep": (".brep", "write_brep_text"),
    "step": (".step", "write_step"),
    "iges": (".iges", "write_iges"),
}

# STEP entity types without a geometric interpretation: measure records and
# presentation containers. The native reader ignores them by design but reports
# every unreached non-metadata record as a declared drop.
_NON_GEOMETRIC_STEP_ENTITIES = frozenset(
    {
        "DRAUGHTING_MODEL",
        "LENGTH_MEASURE_WITH_UNIT",
        "MEASURE_WITH_UNIT",
        "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
        "PLANE_ANGLE_MEASURE_WITH_UNIT",
        "SOLID_ANGLE_MEASURE_WITH_UNIT",
        "UNCERTAINTY_MEASURE_WITH_UNIT",
    }
)
_IGNORED_STEP_ENTITIES = "geometry.ignored-step-entities"
_SOURCE_TOLERANCE_LOSS = "source-tolerance:"


def _ignored_step_entities(rationale: str, /) -> tuple[tuple[str, frozenset[str]], ...]:
    """Parse ``#id NAME[+NAME]`` records of an ignored-STEP-entities declaration."""
    _, separator, listed = rationale.partition(": ")
    if not separator or not listed.strip():
        raise RuntimeError(
            f"Ignored STEP entity declaration lists no entities: {rationale!r}."
        )
    records = []
    for item in listed.split(", "):
        label, _, names = item.strip().partition(" ")
        if not label.startswith("#") or not names:
            raise RuntimeError(f"Ignored STEP entity record is malformed: {item!r}.")
        records.append((label, frozenset(names.split("+"))))
    return tuple(records)


def _require_geometric_losslessness(
    imported: phx.interchange.CadImportResult, /
) -> dict[str, object]:
    """Require exact geometry, an exact p-curve for every coedge, and certified gaps.

    A declared loss is admitted only for ignored non-geometric STEP entities and
    for source tolerance declarations the native model cannot carry as its own
    limit; the latter are bounded below by requiring every certified edge/p-curve
    gap to lie within the file's declared tolerance. Pole (degenerate) coedges a
    format cannot carry (STEP) are re-created by the reader from exact
    source-period evidence; every other coedge must honor a supplied p-curve.
    """
    report = imported.report
    status = report.status.name
    if status not in ("LOSSLESS", "DECLARED_LOSS"):
        raise RuntimeError(
            f"Native {imported.format} import declared {status} semantics: {report.losses!r}."
        )
    ignored: list[str] = []
    tolerance_records: list[str] = []
    for loss in report.losses:
        if loss.path == _IGNORED_STEP_ENTITIES and imported.format == "step":
            records = _ignored_step_entities(loss.rationale)
            geometric = [
                label
                for label, names in records
                if not names <= _NON_GEOMETRIC_STEP_ENTITIES
            ]
            if geometric:
                raise RuntimeError(
                    f"Native STEP import ignored geometry-bearing entities {geometric!r}."
                )
            ignored.extend(
                f"{label} {'+'.join(sorted(names))}" for label, names in records
            )
        elif loss.path.startswith(_SOURCE_TOLERANCE_LOSS):
            tolerance_records.append(loss.rationale)
        else:
            raise RuntimeError(
                f"Native {imported.format} import declared a geometry loss: {loss.path} ({loss.category}): {loss.rationale}"
            )
    model = imported.model
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError(
            f"Native {imported.format} import lost its exact B-Rep carriers."
        )
    coedges = len(geometry.coedge_edges)
    coverage = imported.coverage
    origins = {
        "from_file": coverage.pcurves_from_file,
        "recovered_exact": coverage.pcurves_recovered,
        "absent": coverage.pcurves_absent,
    }
    # Each model coedge's p-curve is the file's, recovered exactly from the
    # source maps, or absent from the source and constructed exactly by the
    # importer (natural boundaries, chart seams, pole edges); the counts are
    # per coedge and must account for every one.
    if sum(origins.values()) != coedges:
        raise RuntimeError(
            f"Native {imported.format} import accounts for the p-curves of {sum(origins.values())} of {coedges} coedges: {origins!r}."
        )
    evidence = imported.tolerance_evidence
    if evidence is None:
        raise RuntimeError(
            f"Native {imported.format} import carries no tolerance evidence."
        )
    edges_with_coedges = {int(edge) for edge in geometry.coedge_edges}
    gaps: list[tuple[int, float, float]] = []
    for entity, certified, declared in zip(
        evidence.entities, evidence.certified_bounds, evidence.source_bounds, strict=True
    ):
        if entity.kind != "edge" or entity.index not in edges_with_coedges:
            continue
        limit = (
            declared
            if declared is not None
            else evidence.source_tolerance
            if evidence.source_tolerance is not None
            else evidence.declared_limit
        )
        if certified is None or not certified <= limit:
            raise RuntimeError(
                f"Native {imported.format} edge {entity.index} p-curve gap {certified!r} is not certified within the declared tolerance {limit!r}."
            )
        gaps.append((entity.index, certified, limit))
    if {index for index, _, _ in gaps} != edges_with_coedges:
        raise RuntimeError(
            f"Native {imported.format} import certified p-curve gaps on {len({index for index, _, _ in gaps})} of {len(edges_with_coedges)} edges."
        )
    return {
        "status": status,
        "ignored_non_geometric_step_entities": ignored,
        "uncarried_source_tolerances": tolerance_records,
        "pcurve_origins": origins,
        "coedges": coedges,
        "maximum_certified_pcurve_gap": max(gap for _, gap, _ in gaps),
        "minimum_declared_tolerance": min(limit for _, _, limit in gaps),
    }


def _contract() -> SpatialCoordinateContract:
    return SpatialCoordinateContract.si()


def _topology_record(model: phx.geometry.brep.BRepModel) -> dict[str, object]:
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("Native CAD source lost its exact B-Rep carriers.")
    topology = model.topology
    payload = {
        "face_loops": tuple(
            tuple(tuple(int(coedge) for coedge in loop) for loop in face)
            for face in geometry.face_loops
        ),
        "coedge_edges": tuple(int(edge) for edge in geometry.coedge_edges),
        "edge_vertices": tuple(
            tuple(int(vertex) for vertex in edge)
            for edge in np.asarray(geometry.edge_vertices, dtype=np.int64)
        ),
        "solid_faces": tuple(
            tuple(int(face) for face in faces) for faces in topology.solid_faces
        ),
        "solid_face_orientations": tuple(
            tuple(int(sign) for sign in signs)
            for signs in topology.solid_face_orientations
        ),
    }
    occurrences = tuple(
        (int(occurrence.solid_index), tuple(occurrence.path))
        for occurrence in geometry.iter_occurrences()
    )
    return {
        "model_id": model.model_id,
        "source_id": model.source_id,
        "source_revision": model.source_revision,
        "geometry_id": geometry.geometry_id,
        "coordinate_contract_id": model.coordinate_contract.spatial_id,
        "coordinate_unit": "meter",
        "counts": {
            "solids": topology.num_solids,
            "faces": topology.num_faces,
            "edges": topology.num_edges,
            "vertices": topology.num_vertices,
        },
        # A trimmed face with k boundary loops contributes 2-k, not one,
        # to the boundary Euler characteristic.
        "boundary_euler_characteristic": topology.num_vertices
        - topology.num_edges
        + sum(2 - len(face_loops) for face_loops in geometry.face_loops),
        "occurrences": [
            {"solid_index": solid, "path": list(path)} for solid, path in occurrences
        ],
        "incidence_signature": canonical_fingerprint(payload),
    }


def _topology_oracle(model: phx.geometry.brep.BRepModel, case: str) -> dict[str, object]:
    record = _topology_record(model)
    counts = record["counts"]
    if not isinstance(counts, dict) or counts["solids"] != 1:
        raise RuntimeError(
            f"{case} source topology requires exactly one closed solid; got {counts!r}."
        )
    expected_euler = {
        "planar-box": 2,
        # The intersection of these two full-dimensional convex cylinders is
        # a convex solid, so its closed boundary is topologically a sphere.
        "quadric-routine": 2,
        "quadric-larger": 2,
        "external-routine": 0,
    }.get(case)
    actual_euler = record["boundary_euler_characteristic"]
    if expected_euler is not None and actual_euler != expected_euler:
        raise RuntimeError(
            f"{case} source topology has Euler characteristic {actual_euler}; expected {expected_euler}."
        )
    return {
        **record,
        "expected_solid_count": 1,
        "expected_boundary_euler_characteristic": expected_euler,
        "passed": True,
    }


def _import_cad(
    path: Path,
    directory: Path,
    contract: SpatialCoordinateContract,
    *,
    source_length_unit: Any = None,
) -> phx.interchange.CadImportResult:
    imported = phx.interchange.read_cad(
        path,
        phx.interchange.CadImportPolicy(contract, pcurve_fit=None),
        trusted_root=directory,
        source_length_unit=source_length_unit,
    )
    _require_geometric_losslessness(imported)
    return imported


def _roundtrip(
    model: phx.geometry.brep.BRepModel,
    name: str,
    format_name: str,
    directory: Path,
) -> tuple[phx.geometry.brep.BRepModel, dict[str, object]]:
    suffix, writer_name = _FORMAT_WRITERS[format_name]
    destination = directory / f"{name}{suffix}"
    exported = getattr(phx.interchange, writer_name)(
        model, destination, policy=phx.interchange.CadExportPolicy()
    )
    # External formats may declare identifier/carrier-encoding transforms and
    # physical-tag label drops; exact geometry may not be approximated, dropped,
    # or left unsupported. Geometry is checked independently after re-import.
    geometry_losses = tuple(
        loss
        for loss in exported.report.losses
        if loss.category not in ("transformed", "synthesized")
        and not (loss.category == "dropped" and loss.path.endswith(".physical-tag"))
    )
    if exported.report.status.name not in ("LOSSLESS", "DECLARED_LOSS") or (
        geometry_losses
    ):
        raise RuntimeError(
            f"Native {format_name} export did not preserve the exact CAD carriers: {exported.report.status.name}, {geometry_losses!r}."
        )
    imported = _import_cad(
        destination,
        directory,
        model.coordinate_contract,
        source_length_unit=(
            model.coordinate_contract.length_unit if format_name == "brep" else None
        ),
    )
    if imported.source_length_unit_meters != 1.0:
        raise RuntimeError(
            f"Native {format_name} round-trip changed the declared meter unit to {imported.source_length_unit_meters!r} m/unit."
        )
    before = model.topology
    after = imported.model.topology
    before_counts = (
        before.num_solids,
        before.num_faces,
        before.num_edges,
        before.num_vertices,
    )
    after_counts = (
        after.num_solids,
        after.num_faces,
        after.num_edges,
        after.num_vertices,
    )
    if after_counts != before_counts:
        raise RuntimeError(
            f"Native {format_name} round-trip changed source topology counts {before_counts!r} to {after_counts!r}."
        )
    return imported.model, {
        "operand": name,
        "format": imported.format,
        "schema": imported.schema,
        "source_digest": imported.source_digest,
        "source_length_unit_meters": imported.source_length_unit_meters,
        "export_report_id": exported.report.report_id,
        "export_status": exported.report.status.name,
        "export_declared_losses": [
            {"path": loss.path, "category": loss.category, "rationale": loss.rationale}
            for loss in exported.report.losses
        ],
        "export_entity_counts": dict(exported.entity_counts),
        "import_id": imported.result_id,
        "import_report_id": imported.report.report_id,
        "import_coverage": dict(imported.coverage.entity_counts),
        "pcurves_from_file": imported.coverage.pcurves_from_file,
        "pcurves_recovered": imported.coverage.pcurves_recovered,
        "pcurves_absent": imported.coverage.pcurves_absent,
        "source_id": imported.model.source_id,
        "source_revision": imported.model.source_revision,
        "topology": _topology_record(imported.model),
    }


def _boolean(
    first: phx.geometry.brep.BRepModel,
    second: phx.geometry.brep.BRepModel,
    operation: Literal["union", "intersection", "difference"],
    /,
) -> tuple[phx.geometry.brep.BRepModel, dict[str, object]]:
    result = phx.geometry.brep.boolean_brep(
        first, second, operation, policy=phx.geometry.brep.BRepBooleanPolicy()
    )
    coverage = result.association_graph.transaction.coverage
    if result.empty or not (coverage.source_exhaustive and coverage.target_exhaustive):
        raise RuntimeError(
            f"Native CAD {operation} did not produce a nonempty source-exhaustive Boolean result: {result!r}."
        )
    return result.model, {
        "operation": result.operation,
        "certificate_id": result.certificate_id,
        "association_graph_id": result.association_graph.graph_id,
        "source_exhaustive": coverage.source_exhaustive,
        "target_exhaustive": coverage.target_exhaustive,
        "deleted_source_ids": list(result.deleted_source_ids),
        "model_id": result.model.model_id,
        "solids": result.model.topology.num_solids,
    }


def _box_case(
    format_name: str, directory: Path
) -> tuple[phx.geometry.brep.BRepModel, float, dict[str, object], dict[str, object]]:
    contract = _contract()
    first = phx.geometry.brep.brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=contract,
        source_id="qualification:native-cad-box-first",
    )
    second = phx.geometry.brep.brep_box(
        (0.25, 0.0, 0.0),
        (0.75, 1.0, 1.0),
        coordinate_contract=contract,
        source_id="qualification:native-cad-box-second",
    )
    first, first_record = _roundtrip(first, "box-first", format_name, directory)
    second, second_record = _roundtrip(second, "box-second", format_name, directory)
    model, boolean_record = _boolean(first, second, "intersection")
    return (
        model,
        0.5,
        {
            "operands": [first_record, second_record],
            "reference_bounds_m": [[0.25, 0.0, 0.0], [0.75, 1.0, 1.0]],
            "reference_volume_m3": 0.5,
            "reference_top_area_m2": 0.5,
            "reference_method": "exact axis-aligned interval intersection",
        },
        boolean_record,
    )


def _quadric_case(
    format_name: str, directory: Path
) -> tuple[phx.geometry.brep.BRepModel, float, dict[str, object], dict[str, object]]:
    contract = _contract()
    authored = (
        ("vertical", 1.0, 2.4, (0.0, 0.0, -1.2), (0.0, 0.0, 1.0)),
        ("transverse", 0.6, 3.0, (-1.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    operands = []
    roundtrips = []
    for name, radius, height, base, axis in authored:
        source = phx.geometry.brep.brep_cylinder(
            radius,
            height,
            base_center=base,
            axis=axis,
            coordinate_contract=contract,
            source_id=f"qualification:native-cad-quadric-{name}",
        )
        restored, entry = _roundtrip(source, name, format_name, directory)
        operands.append(restored)
        roundtrips.append(entry)
    model, boolean_record = _boolean(operands[0], operands[1], "intersection")
    radius, major = 0.6, 1.0
    theta, weights = _gauss_interval(256, -0.5 * math.pi, 0.5 * math.pi)
    integrand = (
        4.0
        * radius**2
        * np.cos(theta) ** 2
        * np.sqrt(major**2 - (radius * np.sin(theta)) ** 2)
    )
    exact_volume = float(np.sum(weights * integrand))
    return (
        model,
        exact_volume,
        {
            "operands": roundtrips,
            "analytic_volume_m3": exact_volume,
            "manufactured_length_m": 1.0,
            "analytic_volume_method": "independent 256-point Gauss-Legendre evaluation of the closed-form one-dimensional cross-cylinder integral",
            "intersection_loops": [
                {
                    "axis_radius_m": 1.0,
                    "transverse_radius_m": 0.6,
                    "transverse_center_z_m": 0.0,
                    "side": 1,
                },
                {
                    "axis_radius_m": 1.0,
                    "transverse_radius_m": 0.6,
                    "transverse_center_z_m": 0.0,
                    "side": -1,
                },
            ],
            "expected_solids": 1,
            "expected_boundary_euler_characteristic": 2,
        },
        boolean_record,
    )


def _external_raw() -> bytes:
    raw = gzip.decompress(base64.b64decode(_NIST_STEP_GZIP_B64))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _NIST_STEP_SHA256:
        raise RuntimeError(
            f"Embedded donor NIST STEP source digest {digest} does not match {_NIST_STEP_SHA256}."
        )
    return raw


def _external_case(
    case: str, format_name: str, directory: Path
) -> tuple[
    phx.geometry.brep.BRepModel, float, dict[str, object], dict[str, object] | None
]:
    contract = _contract()
    source_path = directory / "nist_ftc_11_asme1_rb.stp"
    source_path.write_bytes(_external_raw())
    imported = phx.interchange.read_step(
        source_path,
        phx.interchange.CadImportPolicy(contract, pcurve_fit=None),
        trusted_root=directory,
    )
    if imported.source_digest != _NIST_STEP_SHA256:
        raise RuntimeError(
            f"Native STEP source digest {imported.source_digest} differs from the donor NIST fixture."
        )
    if imported.source_length_unit_meters != 0.001:
        raise RuntimeError(
            f"Native STEP import did not retain the NIST millimeter unit: {imported.source_length_unit_meters!r} m/unit."
        )
    source_losslessness = _require_geometric_losslessness(imported)
    model, roundtrip = _roundtrip(
        imported.model, "external-washer", format_name, directory
    )
    reference = {
        "inner_radius_mm": 16.0,
        "outer_radius_mm": 31.5,
        "bottom_z_mm": -1.5,
        "flat_z_mm": 0.425,
        "inner_bead": {"center_radius_mm": 17.5, "minor_radius_mm": 1.5},
        "outer_bead": {"center_radius_mm": 30.0, "minor_radius_mm": 1.5},
        "flat_inner_radius_mm": 18.93853223808158,
        "flat_outer_radius_mm": 28.56146776191842,
    }
    washer_volume_m3 = _washer_volume(reference) * 0.001**3
    boolean_record = None
    expected_volume = washer_volume_m3
    loops: tuple[tuple[float, float, float, float], ...] = ()
    if case in ("external-routine", "external-combined"):
        admitted_limit = source_losslessness["minimum_declared_tolerance"]
        if (
            not isinstance(admitted_limit, (int, float))
            or isinstance(admitted_limit, bool)
            or imported.source_tolerance is None
        ):
            raise RuntimeError("The original STEP source has no usable repair tolerance.")
        if not imported.source_tolerances:
            raise RuntimeError("The original STEP source declares no repair budget.")
        tolerance = min(
            float(admitted_limit),
            imported.source_tolerance,
            *(
                value * imported.source_length_unit_meters
                for _, value in imported.source_tolerances
            ),
        )
        original_revision = model.source_revision
        repair = phx.geometry.brep.repair_brep_source(model, tolerance=float(tolerance))
        if (
            model.source_revision != original_revision
            or repair.source_revision != original_revision
        ):
            raise RuntimeError(
                "Source repair altered or misidentified its original operand."
            )
        if (
            repair.target_revision == original_revision
            or repair.max_displacement > tolerance
        ):
            raise RuntimeError(
                "Source repair failed its revision or displacement contract."
            )
        repair_record = {
            "source_revision": repair.source_revision,
            "target_revision": repair.target_revision,
            "certificate_id": repair.certificate_id,
            "tolerance_m": float(tolerance),
            "maximum_displacement_m": repair.max_displacement,
            "changes_interpretation": repair.changes_interpretation,
            "completion_semantics": repair.completion_semantics,
            "completion_evidence": repair.completion_evidence,
            "completion_connector_bound_m": repair.completion_connector_bound,
            "completed_face_to_repaired_bound_m": repair.face_source_to_target_bound,
            "repaired_face_to_completed_bound_m": repair.face_target_to_source_bound,
            "correspondence": {
                kind: [
                    {"source": asdict(source), "target": asdict(target)}
                    for source, target in sorted(mapping.items())
                ]
                for kind, mapping in (
                    ("solids", repair.solid_map),
                    ("faces", repair.face_map),
                    ("edges", repair.edge_map),
                    ("vertices", repair.vertex_map),
                )
            },
        }
        model = repair.model
    if case == "external-combined":
        cutter = phx.geometry.brep.brep_cylinder(
            0.0004,
            0.08,
            base_center=(-0.04, 0.0, -0.00075),
            axis=(1.0, 0.0, 0.0),
            coordinate_contract=contract,
            source_id="qualification:native-cad-nist-radial-hole",
        )
        model, boolean_record = _boolean(model, cutter, "difference")
        hole = {
            "radius_mm": 0.4,
            "center_z_mm": -0.75,
            "half_length_mm": 40.0,
            "axis": [1.0, 0.0, 0.0],
        }
        hole_volume_m3 = _radial_hole_volume(reference, hole) * 0.001**3
        expected_volume -= hole_volume_m3
        loops = tuple(
            (radius * 0.001, 0.0004, -0.00075, side)
            for radius in (16.0, 31.5)
            for side in (1, -1)
        )
    provenance = {
        "initial_step_import": {
            "source_digest": imported.source_digest,
            "source_length_unit_meters": imported.source_length_unit_meters,
            "source_id": imported.model.source_id,
            "source_revision": imported.model.source_revision,
            "model_id": imported.model.model_id,
            "report_id": imported.report.report_id,
            "coverage": dict(imported.coverage.entity_counts),
            "pcurves_from_file": imported.coverage.pcurves_from_file,
            "pcurves_recovered": imported.coverage.pcurves_recovered,
            "pcurves_absent": imported.coverage.pcurves_absent,
            "pcurve_fit": "disabled",
            "geometric_losslessness": source_losslessness,
        },
        "requested_format_roundtrip": roundtrip,
        "explicit_source_repair": repair_record,
        "analytic_volume_m3": expected_volume,
        "manufactured_length_m": 0.0315,
        "analytic_volume_method": "independent Pappus revolution integral of the NIST profile; combined case subtracts the donor's independent radial-cylinder integral",
        "external_source_and_authored_hole": case == "external-combined",
        "intersection_loops": [
            {
                "axis_radius_m": radius,
                "transverse_radius_m": minor,
                "transverse_center_z_m": center,
                "side": side,
            }
            for radius, minor, center, side in loops
        ],
        "expected_solids": 1,
        "expected_boundary_euler_characteristic": 0
        if case == "external-routine"
        else None,
    }
    return model, expected_volume, provenance, boolean_record


def _gauss_interval(
    count: int, lower: float, upper: float
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    half = 0.5 * (upper - lower)
    return lower + half * (nodes + 1.0), half * weights


def _washer_volume(reference: dict[str, Any]) -> float:
    bottom = float(reference["bottom_z_mm"])
    inner = float(reference["inner_radius_mm"])
    outer = float(reference["outer_radius_mm"])
    flat_inner = float(reference["flat_inner_radius_mm"])
    flat_outer = float(reference["flat_outer_radius_mm"])
    flat = float(reference["flat_z_mm"])
    volume = math.pi * (outer**2 - inner**2) * (-bottom)
    volume += math.pi * (flat_outer**2 - flat_inner**2) * flat
    for name, start, end in (
        ("inner_bead", inner, flat_inner),
        ("outer_bead", flat_outer, outer),
    ):
        bead = reference[name]
        center = float(bead["center_radius_mm"])
        minor = float(bead["minor_radius_mm"])
        first = math.acos(max(-1.0, min(1.0, (start - center) / minor)))
        last = math.acos(max(-1.0, min(1.0, (end - center) / minor)))
        angle, weights = _gauss_interval(128, min(first, last), max(first, last))
        rho = center + minor * np.cos(angle)
        height = minor * np.sin(angle)
        volume += (
            2.0 * math.pi * float(np.sum(weights * rho * height * minor * np.sin(angle)))
        )
    return volume


def _radial_hole_volume(reference: dict[str, Any], hole: dict[str, Any]) -> float:
    inner = float(reference["inner_radius_mm"])
    outer = float(reference["outer_radius_mm"])
    radius = float(hole["radius_mm"])
    rho, rho_weights = _gauss_interval(64, 0.0, radius)
    phi, phi_weights = _gauss_interval(128, 0.0, 2.0 * math.pi)
    y = rho[:, None] * np.cos(phi)[None, :]
    chord = 2.0 * (np.sqrt(outer**2 - y**2) - np.sqrt(inner**2 - y**2))
    return float(
        np.sum(rho_weights[:, None] * phi_weights[None, :] * chord * rho[:, None])
    )


def _loop_length(axis_radius: float, transverse_radius: float) -> float:
    theta, weights = _gauss_interval(512, 0.0, 2.0 * math.pi)
    sine, cosine = np.sin(theta), np.cos(theta)
    speed = np.sqrt(
        transverse_radius**2
        + transverse_radius**4
        * sine**2
        * cosine**2
        / (axis_radius**2 - transverse_radius**2 * sine**2)
    )
    return float(np.sum(weights * speed))


def _intersection_evidence(
    model: phx.geometry.brep.BRepModel,
    loops: tuple[tuple[float, float, float, float], ...],
) -> dict[str, object]:
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("Intersection oracle requires the retained B-Rep carriers.")
    edges: dict[int, IntersectionCurve] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        if curve_index < 0:
            continue
        curve = geometry.curves[curve_index]
        if isinstance(curve, IntersectionCurve):
            edges[edge] = curve
    if not loops:
        if edges:
            raise RuntimeError(
                f"Unexpected exact intersection branches in a case without a quadric Boolean: {sorted(edges)}."
            )
        return {"branches": 0, "edges": 0, "loops": [], "passed": True}

    scale = max(loop[0] for loop in loops)
    tolerance = 1.0e-9 * scale
    edge_ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    lengths = [0.0 for _ in loops]
    endpoints: list[list[int]] = [[] for _ in loops]
    worst = {"three_dimensional": 0.0, "first_pcurve": 0.0, "second_pcurve": 0.0}
    branch_ids: set[str] = set()
    for edge, curve in edges.items():
        if not curve.fully_certified:
            raise RuntimeError(
                f"Boolean intersection edge {edge} has uncertified continuation charts."
            )
        branch_ids.add(curve.branch_id)
        first, last = edge_ranges[edge]
        parameters = np.linspace(first, last, 2049)
        evaluated = curve.evaluate_coupled(parameters)
        points = np.asarray(evaluated.point, dtype=np.float64)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        side = int(np.sign(np.median(x)))
        radial = np.hypot(x, y)
        matches = [
            index
            for index, loop in enumerate(loops)
            if loop[3] == side and np.max(np.abs(radial - loop[0])) <= 1.0e-6 * scale
        ]
        if len(matches) != 1 or np.any(np.sign(x) != side):
            raise RuntimeError(
                f"Intersection edge {edge} does not bind exactly one declared analytic loop: {matches!r}."
            )
        loop_index = matches[0]
        major, minor, center, _ = loops[loop_index]
        residual = np.maximum(
            np.abs(radial - major), np.abs(np.hypot(y, z - center) - minor)
        )
        worst["three_dimensional"] = max(
            worst["three_dimensional"], float(np.max(residual))
        )
        for name, parameters_uv, patch in (
            ("first_pcurve", evaluated.first_parameters, curve.first.patch),
            ("second_pcurve", evaluated.second_parameters, curve.second.patch),
        ):
            lifted = np.asarray(patch.evaluate(parameters_uv), dtype=np.float64)
            worst[name] = max(
                worst[name],
                float(np.max(np.linalg.norm(lifted - points, axis=1))),
            )
        lengths[loop_index] += float(
            np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        )
        endpoints[loop_index].extend(
            int(vertex) for vertex in geometry.edge_vertices[edge]
        )

    loop_records = []
    for index, (major, minor, center, side) in enumerate(loops):
        expected = _loop_length(major, minor)
        counts = np.unique(endpoints[index], return_counts=True)[1]
        closed = bool(endpoints[index]) and bool(np.all(counts % 2 == 0))
        relative = abs(lengths[index] - expected) / expected
        item = {
            "axis_radius_m": major,
            "transverse_radius_m": minor,
            "transverse_center_z_m": center,
            "side": side,
            "analytic_length_m": expected,
            "branch_length_m": lengths[index],
            "relative_length_defect": relative,
            "closed": closed,
        }
        loop_records.append(item)
        if not closed or relative > 1.0e-5:
            raise RuntimeError(
                f"Declared non-rational intersection loop is incomplete or not closed: {item!r}."
            )
    if max(worst.values()) > tolerance:
        raise RuntimeError(
            f"Intersection branch or either exact p-curve violates the analytic residual tolerance {tolerance:g} m: {worst!r}."
        )
    return {
        "branches": len(branch_ids),
        "edges": len(edges),
        "branch_ids": sorted(branch_ids),
        "maximum_residuals_m": worst,
        "residual_tolerance_m": tolerance,
        "loops": loop_records,
        "passed": True,
    }


def _source_volume_oracle(
    model: phx.geometry.brep.BRepModel, expected_volume: float, case: str
) -> dict[str, object]:
    if model.topology.num_solids != 1:
        raise RuntimeError(
            f"{case} source volume oracle requires one solid; got {model.topology.num_solids}."
        )
    measured = brep_mass_properties(model, relative_tolerance=_VOLUME_RELATIVE_TOLERANCE)
    if measured.status != ("certified",):
        raise RuntimeError(
            f"{case} exact B-Rep volume query failed: status={measured.status!r}, reasons={measured.reasons!r}."
        )
    lower, upper = (float(value) for value in np.asarray(measured.volume_enclosures)[0])
    relative_bound = (
        max(abs(lower - expected_volume), abs(upper - expected_volume)) / expected_volume
    )
    if (
        lower > expected_volume
        or upper < expected_volume
        or relative_bound > _VOLUME_RELATIVE_TOLERANCE
    ):
        raise RuntimeError(
            f"{case} exact source-volume enclosure [{lower!r}, {upper!r}] does not contain the independent {expected_volume!r} m^3 oracle within {_VOLUME_RELATIVE_TOLERANCE:g} relative tolerance."
        )
    return {
        "mass_properties_model_id": measured.model_id,
        "solid_ids": [str(value) for value in measured.solid_ids],
        "source_revision": model.source_revision,
        "status": list(measured.status),
        "reasons": list(measured.reasons),
        "independent_volume_m3": expected_volume,
        "source_volume_enclosure_m3": [lower, upper],
        "maximum_relative_enclosure_defect": relative_bound,
        "relative_tolerance": _VOLUME_RELATIVE_TOLERANCE,
        "passed": True,
    }


def _surface_case(model: phx.geometry.brep.BRepModel, size: float) -> dict[str, object]:
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("Planar CAD surface source lost its exact B-Rep geometry.")
    vertices = np.asarray(geometry.vertex_points, dtype=np.float64)
    selected = []
    for face, loops in enumerate(geometry.face_loops):
        members = np.unique(
            [
                vertex
                for loop in loops
                for coedge in loop
                for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
            ]
        )
        if np.all(np.abs(vertices[members, 2] - 1.0) <= 1.0e-12):
            selected.append(face)
    if len(selected) != 1:
        raise RuntimeError(
            f"Nested-box Boolean must retain exactly one source top face at z=1; selected {selected!r}."
        )

    domain = MeshingDomain.from_brep(model)
    if domain.authority_id != model.model_id:
        raise RuntimeError("Planar CAD meshing domain lost its B-Rep source authority.")
    patches = tuple(
        local
        for local, source_index in enumerate(domain.source_indices[2])
        if int(source_index) in selected
    )
    if len(patches) != 1:
        raise RuntimeError(
            f"Top source face did not map to exactly one native CAD patch: {patches!r}."
        )
    patch = patches[0]
    patch_key = (domain.entity_id(2, patch), domain.source_occurrences[2][patch])
    scope_ids = [
        int(identifier)
        for identifier in domain.scope_entity_ids(2)
        if domain.source_entity_key_for_scope(2, int(identifier)) == patch_key
    ]
    if len(scope_ids) != 1:
        raise RuntimeError("Top source face does not have one canonical scope identity.")
    scope = MeshingScope.from_geometry(
        domain, np.asarray(scope_ids, dtype=np.int64), dimension=2
    )
    source_tolerance = 1.0e-8
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope,
                phx.meshing.FeatureKind.SURFACE,
                maximum_deviation=source_tolerance,
            ),
        ),
    )
    plan = NativeMeshingProvider(NativeMeshingOptions("parametric_surface")).plan(
        domain, specification, coordinate_contract=model.coordinate_contract
    )
    result = plan.execute()
    acceptance_failures: list[dict[str, str]] = []
    try:
        fe_solve = _solve_planar_fe(result)
    except Exception as error:
        fe_solve = None
        acceptance_failures.append(
            {
                "stage": "finite_element_solve",
                "type": type(error).__name__,
                "detail": str(error),
            }
        )
    try:
        accepted = accepted_result_record(result)
    except Exception as error:
        accepted = None
        acceptance_failures.append(
            {
                "stage": "native_result_acceptance",
                "type": type(error).__name__,
                "detail": str(error),
            }
        )
    certification = result.certification
    fidelity = None if certification is None else certification.fidelity
    if fidelity is None:
        acceptance_failures.append(
            {
                "stage": "source_fidelity",
                "type": "RuntimeError",
                "detail": "Planar CAD surface result omitted its continuous source-fidelity certificate.",
            }
        )
    elif (
        fidelity.tolerance > source_tolerance
        or fidelity.mesh_to_source_upper > source_tolerance
        or fidelity.source_to_mesh_upper > source_tolerance
    ):
        acceptance_failures.append(
            {
                "stage": "source_fidelity",
                "type": "RuntimeError",
                "detail": f"Planar CAD surface source-fidelity bounds exceed the requested {source_tolerance:g} m: {fidelity!r}.",
            }
        )
    mesh = result.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    triangles = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    corners = points[triangles]
    area = float(
        np.sum(
            0.5
            * np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
        )
    )
    outside = bool(
        np.any(points[:, 0] < 0.25 - 1.0e-12)
        or np.any(points[:, 0] > 0.75 + 1.0e-12)
        or np.any(points[:, 1] < -1.0e-12)
        or np.any(points[:, 1] > 1.0 + 1.0e-12)
        or np.any(np.abs(points[:, 2] - 1.0) > 1.0e-10)
    )
    expected_faces = {domain.entity_id(2, patch) for patch in patches}
    expected_face_indices = {int(domain.source_indices[2][patch]) for patch in patches}
    face_associations = [
        association
        for association in result.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    ]
    face_association_matches = False
    if len(face_associations) == 1:
        face_association = face_associations[0]
        source_dimensions = np.asarray(face_association.source_dimensions, dtype=np.int64)
        source_indices = np.asarray(face_association.source_indices, dtype=np.int64)
        face_association_matches = (
            source_dimensions.shape == source_indices.shape
            and bool(np.all(source_dimensions == 2))
            and {int(index) for index in source_indices} == expected_face_indices
            and set(face_association.source_occurrence_paths)
            == {domain.source_occurrences[2][patch] for patch in patches}
            and set(map(int, np.asarray(face_association.target_global_ids)))
            == set(map(int, np.asarray(mesh.entity_set(2).entity_ids)))
        )
    if (
        not result.associations
        or any(
            association.association_kind
            not in (
                phx.meshing.GeometryAssociationKind.BREP,
                phx.meshing.GeometryAssociationKind.CURVE,
            )
            or (
                association.association_kind is phx.meshing.GeometryAssociationKind.CURVE
                and association.target_entity_set_id != mesh.entity_set(1).entity_set_id
            )
            or association.source_id != model.source_id
            or association.source_revision != model.source_revision
            for association in result.associations
        )
        or not face_association_matches
    ):
        acceptance_failures.append(
            {
                "stage": "source_association",
                "type": "RuntimeError",
                "detail": "Planar CAD surface result lost its exact B-Rep face association.",
            }
        )
    if abs(area - 0.5) > 1.0e-10 or outside:
        acceptance_failures.append(
            {
                "stage": "analytic_surface_measure",
                "type": "RuntimeError",
                "detail": f"Planar CAD surface differs from the exact 0.5 m^2 top rectangle: area={area!r}, outside={outside}.",
            }
        )
    if fidelity is None:
        fidelity_record = None
    else:
        fidelity_record = {
            "certificate_id": fidelity.certificate_id,
            "tolerance_m": fidelity.tolerance,
            "mesh_to_source_upper_m": fidelity.mesh_to_source_upper,
            "source_to_mesh_upper_m": fidelity.source_to_mesh_upper,
        }
    return {
        "route": plan.route,
        "plan_id": plan.plan_id,
        "native_result": accepted,
        "selected_source_faces": sorted(str(value) for value in expected_faces),
        "source_fidelity": fidelity_record,
        "analytic_area_m2": 0.5,
        "mesh_area_m2": area,
        "relative_area_defect": abs(area - 0.5) / 0.5,
        "outside_analytic_bounds": outside,
        "mesh_measure": "triangle area",
        "fe_solve": fe_solve,
        "acceptance_failures": acceptance_failures,
        "passed": not acceptance_failures,
    }


def _manufactured_quadratic(points: Any) -> Any:
    return jnp.sum(points * points, axis=-1)


def _planar_forcing(points: Any, context: Any) -> Any:
    del context
    return jnp.full(points.shape[:-1], -4.0, dtype=points.dtype)


def _solve_planar_fe(
    result: phx.meshing.CellMeshingResult,
) -> dict[str, object]:
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    space = FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    if constraint.free_dofs.size == 0:
        raise RuntimeError(
            "The original CAD surface PDE requires interior P1 degrees of freedom."
        )
    form = phx.equations.FiniteElementForm(
        "native-cad-planar-laplace-beltrami",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _planar_forcing,
                    coefficient_id="planar-surface-constant-minus-four",
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=_manufactured_quadratic,
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    coordinates = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    exact = np.asarray(_manufactured_quadratic(coordinates), dtype=np.float64)
    essential = np.ones(space.dof_maps[0].global_dof_count, dtype=np.bool_)
    essential[np.asarray(constraint.free_dofs, dtype=np.int64)] = False
    boundary_error = float(
        np.max(np.abs(values[essential] - exact[essential]), initial=0.0)
    )
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    rhs_norm = float(jnp.linalg.norm(rhs))
    normalized_residual = residual / max(1.0, rhs_norm)
    if (
        not bool(jnp.all(solved.successful))
        or boundary_error > _FE_BOUNDARY_TOLERANCE
        or normalized_residual > _FE_RESIDUAL_TOLERANCE
    ):
        raise RuntimeError(
            f"Original CAD planar Laplace-Beltrami FE solve failed: boundary_error={boundary_error:g}, normalized_residual={normalized_residual:g}."
        )
    return {
        "form_id": form.form_id,
        "field_spec_id": field.field_spec_id,
        "degree": 1,
        "dofs": int(values.size),
        "successful": bool(jnp.all(solved.successful)),
        "pde": "-Delta_S u=-4 on the exact planar CAD top face",
        "manufactured_solution": "u=x^2+y^2+z^2",
        "boundary_error": boundary_error,
        "normalized_residual": normalized_residual,
        "boundary_tolerance": _FE_BOUNDARY_TOLERANCE,
        "normalized_residual_tolerance": _FE_RESIDUAL_TOLERANCE,
        "passed": True,
    }


def _cad_manufactured_value(points: Any, length: float) -> Any:
    scaled = points / length
    return jnp.exp(scaled[..., 0]) * jnp.sin(scaled[..., 1]) * jnp.cos(scaled[..., 2])


def _solve_curved_cad_p2_fe(
    mesh: phx.discretization.CellMesh,
    geometry: phx.discretization.CellGeometrySpec,
    length: float,
) -> dict[str, object]:
    """Check the original physical solve; the caller owns geometric acceptance."""
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(
            f"CAD manufactured length must be finite and positive; got {length!r}."
        )
    field = FiniteElementFieldSpec(
        "u", lagrange_element(single_cell_kind(mesh, "tetrahedron"), 2)
    )
    space = FiniteElementPlan(mesh, field, coordinate_spec=geometry).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    if constraint.free_dofs.size == 0:
        raise RuntimeError(
            "The original curved CAD P2 diffusion solve requires interior degrees of freedom."
        )

    def exact(points: Any) -> Any:
        return _cad_manufactured_value(points, length)

    def forcing(points: Any, context: Any) -> Any:
        del context
        return exact(points) / (length * length)

    form = phx.equations.FiniteElementForm(
        "native-cad-curved-p2-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    forcing,
                    coefficient_id=canonical_fingerprint(
                        {"kind": "exp-sin-cos-negative-laplacian", "length": length}
                    ),
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=exact,
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    coordinates = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    exact_values = np.asarray(exact(coordinates), dtype=np.float64)
    essential = np.ones(space.dof_maps[0].global_dof_count, dtype=np.bool_)
    essential[np.asarray(constraint.free_dofs, dtype=np.int64)] = False
    boundary_error = float(
        np.max(np.abs(values[essential] - exact_values[essential]), initial=0.0)
    )
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    rhs_norm = float(jnp.linalg.norm(rhs))
    normalized_residual = residual / max(1.0, rhs_norm)
    if (
        not bool(jnp.all(solved.successful))
        or boundary_error > _FE_BOUNDARY_TOLERANCE
        or normalized_residual > _FE_RESIDUAL_TOLERANCE
    ):
        raise RuntimeError(
            f"Original curved CAD P2 FE solve failed: boundary_error={boundary_error:g}, normalized_residual={normalized_residual:g}."
        )
    return {
        "form_id": form.form_id,
        "field_spec_id": field.field_spec_id,
        "degree": 2,
        "dofs": int(values.size),
        "successful": bool(jnp.all(solved.successful)),
        "pde": "-Delta u=u/L^2 on certified P2 CAD geometry",
        "manufactured_solution": "u=exp(x/L) sin(y/L) cos(z/L)",
        "manufactured_length_m": length,
        "boundary_error": boundary_error,
        "normalized_residual": normalized_residual,
        "boundary_tolerance": _FE_BOUNDARY_TOLERANCE,
        "normalized_residual_tolerance": _FE_RESIDUAL_TOLERANCE,
        "passed": True,
    }


def _curve_cad_result(
    model: phx.geometry.brep.BRepModel,
    straight: phx.meshing.CellMeshingResult,
    maximum_deviation: float,
) -> tuple[
    phx.meshing.CellMeshingResult,
    dict[str, object],
    list[dict[str, str]],
]:
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2)
    projection = phx.geometry.prepare_brep_projection(model)
    association = phx.meshing.associate_mesh_vertices(
        straight.mesh,
        projection,
        policy=phx.meshing.AssociationPropagationPolicy(classification_tolerance=1.0e-8),
    )
    curved = phx.meshing.curve_cell_mesh(
        straight.mesh,
        association,
        projection,
        policy=policy,
        source_result=straight,
    )
    failures: list[dict[str, str]] = []
    result = curved.cell_result if curved.cell_result is not None else straight
    geometry_local_dof_counts = [
        element.local_dof_count for element in curved.geometry.elements
    ]
    geometry_cell_kinds = [element.cell_kind for element in curved.geometry.elements]
    geometry_is_p2 = bool(geometry_local_dof_counts) and all(
        reference_cell_topology(cell_kind).family == "tetrahedron"
        and count == lagrange_element(cell_kind, 2).local_dof_count
        for cell_kind, count in zip(
            geometry_cell_kinds, geometry_local_dof_counts, strict=True
        )
    )
    record: dict[str, object] = {
        "result_id": curved.result_id,
        "status": curved.status.value,
        "policy_id": policy.policy_id,
        "residual_tolerance_m": policy.residual_tolerance,
        "maximum_source_residual_m": (
            curved.evidence.maximum_residual
            if math.isfinite(curved.evidence.maximum_residual)
            else None
        ),
        "evidence_id": curved.evidence.evidence_id,
        "accepted_round": curved.accepted_round,
        "requested_geometry_order": 2,
        "geometry_order": 2 if geometry_is_p2 else None,
        "geometry_cell_kinds": geometry_cell_kinds,
        "geometry_local_dof_counts": geometry_local_dof_counts,
        "cell_result_id": None
        if curved.cell_result is None
        else curved.cell_result.result_id,
    }
    if curved.status is not phx.meshing.HighOrderCurvingStatus.CURVED:
        failures.append(
            {
                "stage": "native_p2_curving",
                "type": "RuntimeError",
                "detail": f"Native CAD P2 curving returned {curved.status.value}: source_failure={curved.evidence.source_failure!r}.",
            }
        )
    if curved.cell_result is None:
        failures.append(
            {
                "stage": "native_p2_publication",
                "type": "RuntimeError",
                "detail": "Native CAD P2 curving did not publish its target-bound CellMeshingResult.",
            }
        )
    else:
        try:
            record["accepted_curved_result"] = accepted_result_record(curved.cell_result)
        except Exception as error:
            failures.append(
                {
                    "stage": "curved_result_acceptance",
                    "type": type(error).__name__,
                    "detail": str(error),
                }
            )
    if not geometry_is_p2:
        failures.append(
            {
                "stage": "curved_geometry_order",
                "type": "RuntimeError",
                "detail": f"Native CAD geometry must use tetrahedral-family P2 reference elements; got cell_kinds={geometry_cell_kinds!r}, local_dof_counts={geometry_local_dof_counts!r}.",
            }
        )
    if curved.evidence.maximum_residual > policy.residual_tolerance:
        failures.append(
            {
                "stage": "curved_source_residual",
                "type": "RuntimeError",
                "detail": f"Native CAD P2 source residual {curved.evidence.maximum_residual:g} m exceeds {policy.residual_tolerance:g} m.",
            }
        )
    curved_certification = (
        None if curved.cell_result is None else curved.cell_result.certification
    )
    curved_fidelity = (
        None if curved_certification is None else curved_certification.fidelity
    )
    if curved_fidelity is None:
        failures.append(
            {
                "stage": "curved_source_fidelity",
                "type": "RuntimeError",
                "detail": "Native CAD P2 result omitted its continuous B-Rep source-fidelity certificate.",
            }
        )
    elif (
        curved_fidelity.tolerance > maximum_deviation
        or curved_fidelity.mesh_to_source_upper > maximum_deviation
        or curved_fidelity.source_to_mesh_upper > maximum_deviation
    ):
        failures.append(
            {
                "stage": "curved_source_fidelity",
                "type": "RuntimeError",
                "detail": f"Curved CAD source-fidelity bounds exceed {maximum_deviation:g} m: {curved_fidelity!r}.",
            }
        )
    record["source_fidelity"] = (
        None
        if curved_fidelity is None
        else {
            "certificate_id": curved_fidelity.certificate_id,
            "tolerance_m": curved_fidelity.tolerance,
            "mesh_to_source_upper_m": curved_fidelity.mesh_to_source_upper,
            "source_to_mesh_upper_m": curved_fidelity.source_to_mesh_upper,
        }
    )
    record["passed"] = not failures
    return result, record, failures


def _require_cad_source_associations(
    model: phx.geometry.brep.BRepModel,
    result: phx.meshing.CellMeshingResult,
) -> None:
    """Independently replay B-Rep strata and the CAD coedge lineage proof."""
    from phydrax.geometry._approximate_source_coverage import _cad_chart_partition
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness

    certification = result.certification
    if certification is None or certification.fidelity is None:
        raise ValueError(
            "CAD source association proof requires certified source fidelity."
        )
    certification.require_passed()
    inputs = certification.inputs
    domain, query = inputs.domain, inputs.source
    witness = certification.fidelity.witness
    if (
        not isinstance(domain, MeshingDomain)
        or query is None
        or not isinstance(witness, SourceBoundaryChartWitness)
        or inputs.cell_regions is None
        or (domain.source_id, domain.source_revision)
        != (model.source_id, model.source_revision)
    ):
        raise ValueError("CAD source association proof lost its source-bound premises.")
    inputs.require_current(result.mesh, result.geometry, domain=domain, source=query)
    banks = [
        set(zip(domain.source_indices[dim], domain.source_occurrences[dim], strict=True))
        for dim in range(3)
    ]
    banks.append(
        set(
            zip(
                domain.region_source_indices,
                domain.region_source_occurrences,
                strict=True,
            )
        )
    )
    covered: dict[int, set[int]] = {dim: set() for dim in range(4)}
    curve_ids: list[str] = []
    for stored in result.associations:
        association = GeometryAssociationArchiveRecipe.from_association(stored).restore()
        if association.association_id != stored.association_id or (
            association.source_id,
            association.source_revision,
        ) != (model.source_id, model.source_revision):
            raise ValueError("CAD association changed its source, revision, or identity.")
        targets = [
            dim
            for dim in range(4)
            if result.mesh.entity_set(dim).entity_set_id
            == association.target_entity_set_id
        ]
        if len(targets) != 1:
            raise ValueError("CAD association has no unique target stratum.")
        target_dim = targets[0]
        association.validate_target(result.mesh.entity_set(target_dim))
        if association.source_boundary_witness_id is not None:
            association.validate_source_boundary_support(
                result.mesh, result.geometry, query, witness
            )
        if association.association_kind is phx.meshing.GeometryAssociationKind.CURVE:
            if target_dim != 1:
                raise ValueError("CAD curve ancestry must target mesh edges.")
            curve_ids.append(association.association_id)
            continue
        if association.association_kind is not phx.meshing.GeometryAssociationKind.BREP:
            raise ValueError("CAD result contains an unsupported association kind.")
        if np.any(~np.asarray(association.resolved)) or np.any(
            np.asarray(association.ambiguous)
        ):
            raise ValueError(
                "CAD B-Rep stratum classifications are unresolved or ambiguous."
            )
        for dim, index, path in set(
            zip(
                map(int, np.asarray(association.source_dimensions)),
                map(int, np.asarray(association.source_indices)),
                association.source_occurrence_paths,
                strict=True,
            )
        ):
            if dim not in range(4) or (index, path) not in banks[dim]:
                raise ValueError("CAD association names an undeclared source occurrence.")
        covered[target_dim].update(map(int, np.asarray(association.target_global_ids)))
    for dim, identifiers in covered.items():
        if identifiers != set(
            map(int, np.asarray(result.mesh.entity_set(dim).entity_ids))
        ):
            raise ValueError(f"CAD B-Rep association does not cover mesh stratum {dim}.")
    lineage = witness.cad_curve_association
    if curve_ids != ([] if lineage is None else [lineage.association_id]):
        raise ValueError("CAD curve association differs from the exact coedge witness.")
    _cad_chart_partition(
        domain, query, witness, result.mesh, np.asarray(inputs.cell_regions)
    )


def _volume_run(
    model: phx.geometry.brep.BRepModel,
    size: float,
    maximum_deviation: float,
    expected_volume: float,
    manufactured_length: float,
    case: str,
) -> dict[str, object]:
    scope = MeshingScope.whole(model, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope,
                phx.meshing.FeatureKind.SURFACE,
                maximum_deviation=maximum_deviation,
            ),
        ),
        maximum_deviation=maximum_deviation,
    )
    schedule = phx.meshing.NativeVolumeSchedule(
        shape_strength=phx.meshing.SizeControlStrength.SOFT
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions("cad_tetrahedral", volume_schedule=schedule)
    ).plan(model, specification, coordinate_contract=model.coordinate_contract)
    straight = plan.execute()
    acceptance_failures: list[dict[str, str]] = []
    try:
        straight_accepted = accepted_result_record(straight)
    except Exception as error:
        straight_accepted = None
        acceptance_failures.append(
            {
                "stage": "native_result_acceptance",
                "type": type(error).__name__,
                "detail": str(error),
            }
        )
    certification = straight.certification
    fidelity = None if certification is None else certification.fidelity
    if fidelity is None:
        acceptance_failures.append(
            {
                "stage": "source_fidelity",
                "type": "RuntimeError",
                "detail": "Native CAD tetrahedral result omitted its continuous source-fidelity certificate.",
            }
        )
    elif (
        fidelity.tolerance > maximum_deviation
        or fidelity.mesh_to_source_upper > maximum_deviation
        or fidelity.source_to_mesh_upper > maximum_deviation
    ):
        acceptance_failures.append(
            {
                "stage": "source_fidelity",
                "type": "RuntimeError",
                "detail": f"{case} native CAD source-fidelity bounds exceed {maximum_deviation:g} m: {fidelity!r}.",
            }
        )
    try:
        _require_cad_source_associations(model, straight)
    except Exception as error:
        acceptance_failures.append(
            {
                "stage": "source_association",
                "type": type(error).__name__,
                "detail": str(error),
            }
        )
    if straight.mesh.topological_dimension != 3 or straight.mesh.ambient_dimension != 3:
        acceptance_failures.append(
            {
                "stage": "mesh_topology",
                "type": "RuntimeError",
                "detail": "Native CAD volume route did not publish a 3D tetrahedral mesh.",
            }
        )
    measured_mesh_volume = float(
        np.sum(np.asarray(straight.quality.evaluation.measures, dtype=np.float64))
    )
    fidelity_record = (
        None
        if fidelity is None
        else {
            "certificate_id": fidelity.certificate_id,
            "tolerance_m": fidelity.tolerance,
            "mesh_to_source_upper_m": fidelity.mesh_to_source_upper,
            "source_to_mesh_upper_m": fidelity.source_to_mesh_upper,
            "migration_maximum_deviation_m": _SOURCE_DEVIATION_TOLERANCE_M,
        }
    )
    route, plan_id = plan.route, plan.plan_id
    del certification, fidelity, plan

    curved_result = straight
    try:
        curved_result, curving, curving_failures = _curve_cad_result(
            model, straight, maximum_deviation
        )
        acceptance_failures.extend(curving_failures)
    except Exception as error:
        curving = {
            "status": "failed",
            "source_model_id": model.model_id,
            "source_revision": model.source_revision,
            "failure": {"type": type(error).__name__, "detail": str(error)},
        }
        acceptance_failures.append(
            {
                "stage": "native_p2_curving",
                "type": type(error).__name__,
                "detail": str(error),
            }
        )
    # The FE stage reads only the curved mesh and geometry: release both meshing
    # results (reports, certificates, exact witnesses) before it allocates.
    fe_mesh, fe_geometry = curved_result.mesh, curved_result.geometry
    fe_result_id = curved_result.result_id
    del straight, curved_result
    try:
        fe = _solve_curved_cad_p2_fe(fe_mesh, fe_geometry, manufactured_length)
    except Exception as error:
        fe = None
        acceptance_failures.append(
            {
                "stage": "finite_element_solve",
                "type": type(error).__name__,
                "detail": str(error),
            }
        )
    return {
        "route": route,
        "plan_id": plan_id,
        "volume_schedule": {
            "schedule_id": schedule.schedule_id,
            "radius_edge_bound": schedule.radius_edge_bound,
            "minimum_dihedral_degrees": schedule.minimum_dihedral_degrees,
            "shape_strength": schedule.shape_strength.value,
            "donor_citations": list(_CAD_VOLUME_SCHEDULE_CITATIONS),
        },
        "native_result": straight_accepted,
        "source_fidelity": fidelity_record,
        "source_volume_oracle_m3": expected_volume,
        "straight_mesh_volume_m3": measured_mesh_volume,
        "straight_mesh_volume_relative_defect": abs(
            measured_mesh_volume - expected_volume
        )
        / expected_volume,
        "mesh_measure": "straight tetrahedral cell volume",
        "p2_curving": curving,
        "p2_fe_result_id": fe_result_id,
        "fe_solve": fe,
        "acceptance_failures": acceptance_failures,
        "passed": not acceptance_failures,
    }


def _record_float(record: dict[str, object], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"CAD source oracle field {name!r} must be a finite number; got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(
            f"CAD source oracle field {name!r} must be finite; got {number!r}."
        )
    return number


def _intersection_loop_oracles(
    source_oracle: dict[str, object],
) -> tuple[tuple[float, float, float, int], ...]:
    raw_loops = source_oracle.get("intersection_loops", [])
    if not isinstance(raw_loops, list):
        raise TypeError(
            f"CAD intersection loop oracle must be a list; got {type(raw_loops).__name__}."
        )
    loops: list[tuple[float, float, float, int]] = []
    for index, raw_loop in enumerate(raw_loops):
        if not isinstance(raw_loop, dict):
            raise TypeError(
                f"CAD intersection loop {index} must be a record; got {type(raw_loop).__name__}."
            )
        side = raw_loop.get("side")
        if isinstance(side, bool) or not isinstance(side, int) or side not in (-1, 1):
            raise ValueError(
                f"CAD intersection loop {index} side must be -1 or 1; got {side!r}."
            )
        loops.append(
            (
                _record_float(raw_loop, "axis_radius_m"),
                _record_float(raw_loop, "transverse_radius_m"),
                _record_float(raw_loop, "transverse_center_z_m"),
                side,
            )
        )
    return tuple(loops)


def _acceptance_failures(
    result: dict[str, object],
) -> list[dict[str, str]]:
    raw_failures = result.get("acceptance_failures")
    if not isinstance(raw_failures, list):
        raise TypeError("CAD mesh result acceptance_failures must be a list.")
    failures: list[dict[str, str]] = []
    for index, raw_failure in enumerate(raw_failures):
        if not isinstance(raw_failure, dict):
            raise TypeError(
                f"CAD acceptance failure {index} must be a record; got {type(raw_failure).__name__}."
            )
        stage = raw_failure.get("stage")
        failure_type = raw_failure.get("type")
        detail = raw_failure.get("detail")
        if not (
            isinstance(stage, str)
            and isinstance(failure_type, str)
            and isinstance(detail, str)
        ):
            raise TypeError(
                f"CAD acceptance failure {index} must contain string stage, type, and detail fields."
            )
        failures.append({"stage": stage, "type": failure_type, "detail": detail})
    return failures


def _profile_record(profile_id: str, format_name: str, case: str) -> dict[str, object]:
    fixture_case = (
        "planar-box"
        if case == "planar-box"
        else ("quadric-routine" if case.startswith("quadric-") else case)
    )
    scales = (1, 2) if case == "quadric-larger" else (1,)
    record: dict[str, object] = {
        "profile_id": profile_id,
        "status": "failed",
        "mandatory": True,
        "source_request": f"tests/data/meshing/campaigns.json#/profile_requests/{profile_id}",
        "source_provenance": {
            "branch": _SOURCE_BRANCH,
            "commit": _SOURCE_COMMIT,
            "scenario_source": _SOURCE_REVISION,
            **_FIXTURE_PROVENANCE[fixture_case],
        },
        "profile_request_controls": {
            "scenario_controls": {"cad_case": case, "cad_format": format_name},
            "resolution_scale": 1,
            "minimum_adaptation_rounds": 0,
        },
        "producer_controls": {
            "native_format": format_name,
            "resolution_default": _RESOLUTION,
            "mesh_scales": list(scales),
            "optional_geometry_engines": "blocked; native constructors, codec, Boolean and meshing routes only",
        },
        "oracle_tolerances": {
            "source_volume_relative": _VOLUME_RELATIVE_TOLERANCE,
            "native_brep_maximum_deviation_m": _SOURCE_DEVIATION_TOLERANCE_M,
            "fe_normalized_residual": _FE_RESIDUAL_TOLERANCE,
            "fe_boundary_value": _FE_BOUNDARY_TOLERANCE,
            "source_revision_and_unit_identity": "exact",
        },
        "producer_route": (
            "native BRep box operand codec round-trip -> native Boolean intersection -> selected native parametric surface triangle mesh -> original P1 surface Laplace-Beltrami solve"
            if case == "planar-box"
            else "native BRep/STEP/IGES input -> exact native CAD carriers -> native parametric boundary -> cad_tetrahedral tetrahedra -> certified P2 curving -> original curved-CAD P2 diffusion solve; no external mesher"
        ),
        "attempted": True,
    }
    stage = "native_source_construction"
    validation_failures: list[dict[str, str]] = []
    try:
        with TemporaryDirectory(prefix="phydrax-native-cad-profile-") as temporary:
            directory = Path(temporary)
            if case == "planar-box":
                model, expected_volume, source_oracle, boolean_record = _box_case(
                    format_name, directory
                )
                maximum_deviation = 1.0e-8
                source_size = 0.5 / _RESOLUTION
            elif case.startswith("quadric-"):
                model, expected_volume, source_oracle, boolean_record = _quadric_case(
                    format_name, directory
                )
                maximum_deviation = (0.5 * 1.0 / _RESOLUTION) ** 2 / (4.0 * 0.6)
                source_size = 0.5 * 1.0 / _RESOLUTION
            else:
                model, expected_volume, source_oracle, boolean_record = _external_case(
                    case, format_name, directory
                )
                maximum_deviation = 4.0e-5
                source_size = 6.0e-3 / _RESOLUTION
            record["source_model"] = _topology_record(model)
            try:
                record["source_topology_oracle"] = _topology_oracle(model, case)
            except Exception as error:
                validation_failures.append(
                    {
                        "stage": "source_topology_oracle",
                        "type": type(error).__name__,
                        "detail": str(error),
                    }
                )
            record["boolean"] = boolean_record
            record["source_oracle"] = source_oracle
            stage = "exact_source_volume_oracle"
            try:
                record["source_volume_query"] = _source_volume_oracle(
                    model, expected_volume, case
                )
            except Exception as error:
                validation_failures.append(
                    {"stage": stage, "type": type(error).__name__, "detail": str(error)}
                )
            loops = _intersection_loop_oracles(source_oracle)
            stage = "intersection_branch_topology_and_pcurve_oracle"
            try:
                record["intersection_evidence"] = _intersection_evidence(model, loops)
            except Exception as error:
                validation_failures.append(
                    {"stage": stage, "type": type(error).__name__, "detail": str(error)}
                )
            if case == "planar-box":
                stage = "native_surface_mesh_and_fe_solve"
                surface = _surface_case(model, source_size)
                record["mesh_runs"] = [
                    {
                        "scale": 1,
                        "requested_size_m": source_size,
                        "requested_maximum_deviation_m": maximum_deviation,
                        "surface": surface,
                        "fe_solve": surface["fe_solve"],
                    }
                ]
                validation_failures.extend(
                    {
                        **failure,
                        "stage": f"surface_mesh_{failure['stage']}",
                    }
                    for failure in _acceptance_failures(surface)
                )
            else:
                runs = []
                for scale in scales:
                    run_size = source_size / scale
                    run_deviation = maximum_deviation / (
                        scale**2
                        if case in ("external-routine", "external-combined")
                        else 1.0
                    )
                    if case.startswith("quadric-"):
                        run_deviation = run_size**2 / (4.0 * 0.6)
                    stage = f"native_cad_volume_mesh_and_fe_solve_scale_{scale}"
                    run = _volume_run(
                        model,
                        run_size,
                        run_deviation,
                        expected_volume,
                        _record_float(source_oracle, "manufactured_length_m"),
                        case,
                    )
                    runs.append(
                        {
                            "scale": scale,
                            "requested_size_m": run_size,
                            "requested_maximum_deviation_m": run_deviation,
                            **run,
                        }
                    )
                    validation_failures.extend(
                        {
                            **failure,
                            "stage": f"volume_scale_{scale}_{failure['stage']}",
                        }
                        for failure in _acceptance_failures(run)
                    )
                record["mesh_runs"] = runs
        record["validation_failures"] = validation_failures
        record["status"] = "passed" if not validation_failures else "failed"
        record["case_contract_completion"] = (
            "complete" if not validation_failures else "incomplete"
        )
        record["independent_scientific_certification"] = (
            "passed" if not validation_failures else "failed"
        )
    except Exception as error:
        record["status"] = "failed"
        record["case_contract_completion"] = "incomplete"
        record["independent_scientific_certification"] = "failed"
        record["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "detail": str(error),
        }
        record["validation_failures"] = validation_failures
    return record


def native_cad_profile_matrix() -> dict[str, object]:
    """Execute all recovered BRep/STEP/IGES source profiles with native routes."""
    records = {
        profile_id: _profile_record(profile_id, format_name, case)
        for profile_id, format_name, case in _PROFILE_CASES
    }
    passed = sum(record["status"] == "passed" for record in records.values())
    failed = len(records) - passed
    complete = failed == 0
    return {
        "scenario": "native-cad-profile-matrix",
        "workflow": "native-cad",
        "status": "passed" if complete else "failed",
        "mandatory_workflow_completion": "complete" if complete else "incomplete",
        "profile_count": len(records),
        "passed_profiles": passed,
        "failed_profiles": failed,
        "profiles": records,
        "provenance": {
            "source_branch": _SOURCE_BRANCH,
            "source_commit": _SOURCE_COMMIT,
            "inventory": "tools/meshing_qualification/native_case_migration.json#/workflow_contracts/native-cad",
            "profile_controls_authority": "donor tests/data/meshing/campaigns.json#/profile_requests/<profile-id>",
        },
        "nonclaims": [
            "No optional geometry engine or external mesher is imported or substituted.",
            "A failed record is a real source/API/acceptance failure, never a skipped or substituted profile.",
        ],
    }
