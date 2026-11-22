#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-shell proof for native CAD solids: model part cached, chart part per mesh.

The native certificate proves embedded source boundaries with outward material
sides. Multi-solid sources additionally require exact disjoint material regions
and a manifold union exterior. The proof has two parts:

- the model part (:func:`prove_source_shell`, cached per domain by
  :func:`source_shell`): exact source topology and material orientation,
  embedded trim sheets and declared face contact only. Multiple solids use
  the exact planar convex-trim source proof and the union's exterior links;
- the chart part (:func:`certify_native_trim_references`), on the native chart
  export the mesh was built from: each face's chord loops become reference
  segments bound to their source use (p-curve, coedge endpoint roots and
  junction boxes); the trim-to-reference homotopy is proved embedded
  (``certify_trim_reference_homotopy``) and the face sheet, widened over the
  references, is proved an embedding (``certify_reference_trim_sheet``).

V4 (run first by the certificate) proves the chart triangles partition each
face's chord loops, which is the reference partition the Python proof checked
on its own charts. V3 binds every chart triangle and chord to its source
face and use within the tolerance.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from .._fingerprint import canonical_fingerprint
from ..geometry._approximate_source_coverage import _cad_trim_root_box
from ..geometry._domain_boundary import _rooted_patch_chart_junctions
from ..geometry._meshing_domain import SourcePreparation
from ..geometry.brep._curve_approximation_topology import (
    certify_trim_reference_homotopy,
    TrimReferenceSegment,
)
from ..geometry.brep._measure import brep_shell_volume_sign
from ..geometry.brep._source_embedding import (
    BRepTrimSheetProof,
    certify_brep_trim_sheet,
    certify_reference_trim_sheet,
)
from ..geometry.brep._source_shell_admissibility import (
    _source_region_exterior_is_manifold,
    certify_source_face_pairs,
    certify_source_regions,
    shell_source_model,
)


if TYPE_CHECKING:
    from .._meshcore import CadModel
    from ..geometry._meshing_domain import MeshingDomain
    from ._native_cad_certificate import NativeCadSurfaceWitness


# Source evaluators raise these on carriers, trims or enclosures they cannot
# decide; each is a refusal of the shell proof and its message is the reason.
_UNDECIDED = (
    ArithmeticError,
    IndexError,
    KeyError,
    NotImplementedError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class NativeSourceShell:
    """Model-only shell premises of one domain, or the step that refused them."""

    domain_id: str
    refusal: str | None
    sheets: tuple[BRepTrimSheetProof, ...]
    pair_certificates: tuple[str, ...]
    certificate_id: str


def _shell(
    domain: MeshingDomain,
    refusal: str | None,
    sheets: tuple[BRepTrimSheetProof, ...] = (),
    pairs: tuple[str, ...] = (),
    /,
    *,
    regional_material: str | None = None,
) -> NativeSourceShell:
    identity = canonical_fingerprint(
        {
            "kind": "native-cad-source-shell",
            "domain": domain.domain_id,
            "refusal": refusal,
            "sheets": tuple(sheet.embedding.source_id for sheet in sheets),
            "pairs": pairs,
            **(
                {}
                if regional_material is None
                else {"regional_material": regional_material}
            ),
            "volume_sign": None if refusal is not None else 1,
        }
    )
    return NativeSourceShell(domain.domain_id, refusal, sheets, pairs, identity)


def prove_source_shell(domain: MeshingDomain, /) -> NativeSourceShell:
    """Prove the mesh-independent shell premises of ``domain`` (uncached)."""
    try:
        regional_material = None
        model = None
        if len(domain.regions) > 1:
            material = certify_source_regions(domain)
            if material is None:
                return _shell(domain, "exact planar source material regions")
            if not _source_region_exterior_is_manifold(domain, material):
                return _shell(domain, "regional union exterior links")
            regional_material = material.certificate_id
        else:
            model = shell_source_model(domain)
            if model is None:
                return _shell(domain, "single-shell topology")
        sheets: list[BRepTrimSheetProof] = []
        for patch in range(len(domain.patches)):
            sheet = certify_brep_trim_sheet(domain, patch)
            if sheet is None:
                return _shell(domain, f"trim sheet of source patch {patch}")
            sheets.append(sheet)
        pairs = ()
        if model is not None:
            certified_pairs = certify_source_face_pairs(model)
            if certified_pairs is None:
                return _shell(domain, "source face pair separation")
            if brep_shell_volume_sign(model, model.geometry.solid_shells[0][0]) != 1:
                return _shell(domain, "shell volume sign")
            pairs = certified_pairs
    except _UNDECIDED as error:
        return _shell(domain, f"undecided source ({type(error).__name__}: {error})")
    return _shell(domain, None, tuple(sheets), pairs, regional_material=regional_material)


# The model part depends on the domain's BRep authority and its patch/loop
# occurrences, which ``domain.domain_id`` identifies by content. Like the
# geometry bundle cache, an entry serves only the domain object it was proved
# for (the proofs bind that object); any other object is proved anew.
_CACHE_CAPACITY = 32
_CACHE: OrderedDict[str, tuple[MeshingDomain, NativeSourceShell]] = OrderedDict()


def source_shell(domain: MeshingDomain, /) -> NativeSourceShell:
    """The model part of the shell proof of ``domain``, proved once per domain."""
    key = domain.domain_id
    cached = _CACHE.get(key)
    if cached is not None and cached[0] is domain:
        _CACHE.move_to_end(key)
        return cached[1]
    shell = prove_source_shell(domain)
    _CACHE[key] = (domain, shell)
    _CACHE.move_to_end(key)
    if len(_CACHE) > _CACHE_CAPACITY:
        _CACHE.popitem(last=False)
    return shell


def _fraction_point(values: np.ndarray, /) -> tuple[Fraction, Fraction]:
    return Fraction.from_float(float(values[0])), Fraction.from_float(float(values[1]))


def _reference_loops(
    domain: MeshingDomain,
    model: CadModel,
    witness: NativeCadSurfaceWitness,
    patch: int,
    /,
) -> tuple[tuple[TrimReferenceSegment, ...], ...] | None:
    """The patch's chord loops as reference segments bound to their source uses.

    Chords of a loop are chained bitwise from start to stop (V4 proved each
    loop one closed cycle). A chord end at its use's start or stop parameter
    carries that junction's root box and the coedge's endpoint root enclosure.
    """
    authority = domain.brep_authority
    assert authority is not None
    face = domain.source_indices[2][patch]
    offsets = np.asarray(model.loop_use_offsets, dtype=np.int64)
    loops = np.flatnonzero(np.asarray(model.loop_surfaces) == patch)
    chord_loop = np.asarray(witness.chord_loop)
    chord_use = np.asarray(witness.chord_use)
    chord_uv = np.asarray(witness.chord_uv)
    chord_range = np.asarray(witness.chord_range)
    junctions = _rooted_patch_chart_junctions(domain, patch)
    occurrence_loops = domain.patches[patch].loops
    if loops.size != len(occurrence_loops):
        return None
    result: list[tuple[TrimReferenceSegment, ...]] = []
    for local, loop in enumerate(loops.tolist()):
        rows = np.flatnonzero(chord_loop == loop)
        if rows.size == 0:
            return None
        by_start = {tuple(chord_uv[row, 0].tolist()): int(row) for row in rows}
        ordered = [int(rows[0])]
        while len(ordered) < rows.size:
            following = by_start.get(tuple(chord_uv[ordered[-1], 1].tolist()))
            if following is None or following in ordered:
                return None
            ordered.append(following)
        uses = occurrence_loops[local]
        coedges = authority.geometry.face_loops[face][local]
        segments: list[TrimReferenceSegment] = []
        for row in ordered:
            slot = int(chord_use[row]) - int(offsets[loop])
            if not 0 <= slot < len(uses):
                return None
            use = uses[slot]
            roots = authority.geometry.coedge_endpoint_roots[coedges[slot]]
            start_root, stop_root = roots if use.sense > 0 else roots[::-1]
            traversal = (use.first, use.last) if use.sense > 0 else (use.last, use.first)
            ends = {
                traversal[0]: (
                    _cad_trim_root_box(junctions[local][slot - 1]),
                    None if start_root is None else start_root.parameter_enclosure(),
                ),
                traversal[1]: (
                    _cad_trim_root_box(junctions[local][slot]),
                    None if stop_root is None else stop_root.parameter_enclosure(),
                ),
            }
            first, last = (float(value) for value in chord_range[row])
            start, stop = ends.get(first, (None, None)), ends.get(last, (None, None))
            segments.append(
                TrimReferenceSegment(
                    use.pcurve,
                    Fraction.from_float(first),
                    Fraction.from_float(last),
                    _fraction_point(chord_uv[row, 0]),
                    _fraction_point(chord_uv[row, 1]),
                    start[0],
                    stop[0],
                    start[1],
                    stop[1],
                )
            )
        result.append(tuple(segments))
    return tuple(result)


def certify_native_trim_references(
    domain: MeshingDomain,
    model: CadModel,
    shell: NativeSourceShell,
    witness: NativeCadSurfaceWitness,
    /,
) -> tuple[tuple[str, ...], str | None]:
    """Per-face reference sheet certificates, or the refusal reason.

    Requires a proved ``shell`` of ``domain`` and a witness whose chart
    partition V4 certified against ``model`` (the domain's native bundle).
    """
    if shell.refusal is not None or shell.domain_id != domain.domain_id:
        raise ValueError("Trim references need the domain's proved source shell.")
    preparation = SourcePreparation(domain)
    certificates: list[str] = []
    for patch, native in enumerate(shell.sheets):
        try:
            loops = _reference_loops(domain, model, witness, patch)
            if loops is None:
                return (), f"chord loops of source patch {patch}"
            # A ring torus proves its exact rectangular homotopy together
            # with both paired chord chains in certify_reference_trim_sheet.
            if native.embedding.deck_axes != (0, 1):
                certify_trim_reference_homotopy(loops, preparation)
        except _UNDECIDED as error:
            return (), (
                f"trim homotopy of source patch {patch} ({type(error).__name__}: {error})"
            )
        sheet = certify_reference_trim_sheet(domain, patch, native, loops, preparation)
        if sheet is None:
            return (), f"reference trim sheet of source patch {patch}"
        reference_id, embedding = sheet
        certificates.append(
            canonical_fingerprint(
                {
                    "face": domain.source_indices[2][patch],
                    "source": embedding.source_id,
                    "reference": reference_id,
                }
            )
        )
    return tuple(certificates), None


__all__ = [
    "NativeSourceShell",
    "certify_native_trim_references",
    "prove_source_shell",
    "source_shell",
]
