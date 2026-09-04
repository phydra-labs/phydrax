#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only validated SWC morphology parsing and stable-ID adaptation."""

from __future__ import annotations

from math import isfinite, sqrt
from pathlib import Path

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ._morphology import BranchSpec, CellMorphologyPlan, CompartmentSpec
from ._units import ELECTROPHYSIOLOGY_UNITS


class SWCAdapterEvidence(StrictModule, NonTrainableState):
    """SWC-specific counts and stable mapping beside the canonical report."""

    node_count: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    branch_count: int = eqx.field(static=True)
    root_swc_id: int = eqx.field(static=True)
    stable_mapping: tuple[tuple[int, str], ...] = eqx.field(static=True)
    total_segment_length_um: float = eqx.field(static=True)
    node_types: tuple[int, ...] = eqx.field(static=True)
    warnings: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    morphology_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_count: int,
        segment_count: int,
        branch_count: int,
        root_swc_id: int,
        stable_mapping: tuple[tuple[int, str], ...],
        total_segment_length_um: float,
        node_types: tuple[int, ...],
        warnings: tuple[str, ...],
        source_id: str,
        morphology_id: str,
        /,
    ):
        self.node_count = node_count
        self.segment_count = segment_count
        self.branch_count = branch_count
        self.root_swc_id = root_swc_id
        self.stable_mapping = stable_mapping
        self.total_segment_length_um = total_segment_length_um
        self.node_types = node_types
        self.warnings = warnings
        self.source_id = source_id
        self.morphology_id = morphology_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-swc-adapter-evidence-v1",
                "node_count": node_count,
                "segment_count": segment_count,
                "branch_count": branch_count,
                "root_swc_id": root_swc_id,
                "stable_mapping": [list(value) for value in stable_mapping],
                "total_segment_length_um": total_segment_length_um,
                "node_types": list(node_types),
                "warnings": list(warnings),
                "source_id": source_id,
                "morphology_id": morphology_id,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class SWCAdaptation(StrictModule, NonTrainableState):
    """Validated morphology, canonical interchange report, and SWC evidence."""

    morphology: CellMorphologyPlan
    report: AdapterReport
    evidence: SWCAdapterEvidence


def _parse_integer(token: str, line_number: int, field: str, /) -> int:
    value = int(token)
    if str(value) != token and token not in (f"+{value}", f"-{abs(value)}"):
        raise ValueError(f"SWC line {line_number} {field} must be an integer token.")
    return value


def parse_swc_text(
    text: str,
    cell_id: str,
    /,
    *,
    capacitance_density_uF_cm2: float = 1.0,
    axial_resistivity_ohm_cm: float = 100.0,
) -> SWCAdaptation:
    """Parse SWC text on the host and construct a stable-ID morphology plan."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string.")
    records: dict[int, tuple[int, float, float, float, float, int]] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        fields = content.split()
        if len(fields) != 7:
            raise ValueError(f"SWC line {line_number} must contain exactly seven fields.")
        node_id = _parse_integer(fields[0], line_number, "node id")
        node_type = _parse_integer(fields[1], line_number, "node type")
        x, y, z, radius = (float(value) for value in fields[2:6])
        parent = _parse_integer(fields[6], line_number, "parent id")
        if node_id <= 0:
            raise ValueError(f"SWC line {line_number} node id must be positive.")
        if node_type <= 0:
            raise ValueError(f"SWC line {line_number} node type must be positive.")
        if node_id in records:
            raise ValueError(f"SWC node id {node_id} is duplicated.")
        if not all(isfinite(value) for value in (x, y, z, radius)):
            raise ValueError(
                f"SWC line {line_number} coordinates and radius must be finite."
            )
        if radius <= 0.0:
            raise ValueError(f"SWC line {line_number} radius must be positive.")
        if parent == node_id:
            raise ValueError(f"SWC node {node_id} cannot parent itself.")
        records[node_id] = (node_type, x, y, z, radius, parent)
    if not records:
        raise ValueError("SWC text contains no morphology records.")
    roots = tuple(node_id for node_id, record in records.items() if record[5] == -1)
    if len(roots) != 1:
        raise ValueError("SWC morphology must contain exactly one -1 root parent.")
    for node_id, record in records.items():
        parent = record[5]
        if parent != -1 and parent not in records:
            raise ValueError(f"SWC node {node_id} references missing parent {parent}.")
    root = roots[0]
    for node_id in records:
        visited: set[int] = set()
        current = node_id
        while current != -1:
            if current in visited:
                raise ValueError("SWC parent relations must be acyclic.")
            visited.add(current)
            current = records[current][5]
        if root not in visited:
            raise ValueError("Every SWC node must be connected to the single root.")
    ordered_ids = tuple(sorted(records))
    mapping = tuple((node_id, f"swc-{node_id}") for node_id in ordered_ids)
    mapped = dict(mapping)
    lengths: dict[int, float] = {}
    compartments: list[CompartmentSpec] = []
    for node_id in ordered_ids:
        node_type, x, y, z, radius, parent = records[node_id]
        del node_type
        if parent == -1:
            length = 2.0 * radius
            parent_identifier = None
        else:
            parent_record = records[parent]
            length = sqrt(
                (x - parent_record[1]) ** 2
                + (y - parent_record[2]) ** 2
                + (z - parent_record[3]) ** 2
            )
            if length <= 0.0:
                raise ValueError(f"SWC segment {parent}->{node_id} has zero length.")
            parent_identifier = mapped[parent]
        lengths[node_id] = length
        compartments.append(
            CompartmentSpec(
                mapped[node_id],
                parent_identifier,
                length,
                2.0 * radius,
                capacitance_density_uF_cm2=capacitance_density_uF_cm2,
                axial_resistivity_ohm_cm=axial_resistivity_ohm_cm,
            )
        )
    children: dict[int, tuple[int, ...]] = {
        node_id: tuple(
            sorted(child for child, record in records.items() if record[5] == node_id)
        )
        for node_id in ordered_ids
    }
    branch_paths: list[tuple[int, ...]] = []
    if len(records) == 1:
        branch_paths.append((root,))
    else:
        starts = tuple(
            node_id
            for node_id in ordered_ids
            if node_id == root or len(children[node_id]) != 1
        )
        for start in starts:
            for child in children[start]:
                path = [start, child]
                current = child
                while len(children[current]) == 1:
                    current = children[current][0]
                    path.append(current)
                branch_paths.append(tuple(path))
    branches = tuple(
        BranchSpec(
            f"swc-branch-{path[0]}-{path[1] if len(path) > 1 else path[0]}",
            tuple(mapped[value] for value in path),
        )
        for path in branch_paths
    )
    morphology = CellMorphologyPlan(cell_id, compartments, branches=branches)
    warnings = (
        () if records[root][0] == 1 else ("Root node type is not SWC soma type 1.",)
    )
    source_id = canonical_fingerprint(
        {
            "kind": "swc-source-v1",
            "records": [[node_id, *records[node_id]] for node_id in ordered_ids],
            "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "SWC",
        "phydrax-electrophysiology-cell-morphology",
        source_id=source_id,
        target_id=morphology.plan_id,
        coordinate_mapping=(
            "SWC xyz micrometres -> Euclidean compartment length_um",
            "SWC radius micrometres -> compartment diameter_um = 2 * radius",
        ),
        preserved_fields=(
            "node identifiers",
            "parent topology",
            "segment lengths",
            "node radii as compartment diameters",
        ),
        assumptions=(
            "SWC coordinates and radii are expressed in micrometres",
            f"membrane capacitance density is {float(capacitance_density_uF_cm2)} uF/cm2",
            f"axial resistivity is {float(axial_resistivity_ohm_cm)} ohm*cm",
        ),
        losses=(
            AdapterLoss(
                "/nodes/*/absolute_xyz",
                "import",
                "dropped",
                "The cable morphology retains segment length but not absolute 3-D embedding.",
                changes_interpretation=True,
            ),
            AdapterLoss(
                "/nodes/*/type",
                "import",
                "dropped",
                "SWC node types are reported as evidence but are not assigned to compartments.",
                changes_interpretation=True,
            ),
            AdapterLoss(
                "/root/length_um",
                "import",
                "synthesized",
                "The root cylinder length is synthesized as twice the SWC root radius.",
                changes_interpretation=True,
            ),
            AdapterLoss(
                "/nodes/*/radius",
                "import",
                "transformed",
                "SWC radius is represented as cylindrical compartment diameter.",
                changes_interpretation=False,
            ),
        ),
    )
    evidence = SWCAdapterEvidence(
        len(records),
        len(records) - 1,
        len(branches),
        root,
        mapping,
        sum(lengths[node_id] for node_id in ordered_ids if node_id != root),
        tuple(sorted(set(record[0] for record in records.values()))),
        warnings,
        source_id,
        morphology.plan_id,
    )
    return SWCAdaptation(morphology, report, evidence)


def parse_swc_file(
    path: str | Path,
    cell_id: str,
    /,
    *,
    capacitance_density_uF_cm2: float = 1.0,
    axial_resistivity_ohm_cm: float = 100.0,
) -> SWCAdaptation:
    """Read and parse one UTF-8 SWC file on the host."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"SWC path is not a file: {source}.")
    return parse_swc_text(
        source.read_text(encoding="utf-8"),
        cell_id,
        capacitance_density_uF_cm2=capacitance_density_uF_cm2,
        axial_resistivity_ohm_cm=axial_resistivity_ohm_cm,
    )


__all__ = ["SWCAdaptation", "SWCAdapterEvidence", "parse_swc_file", "parse_swc_text"]
