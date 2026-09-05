#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Transitive contour clusters of initial lesions, not repair or survival states."""

from __future__ import annotations

from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint
from ._interactions import PrimaryHistoryKey
from ._lesions import InitialLesion, InitialLesionLedger
from ._targets import RadiationTargetGeometry, TargetMolecule, TargetSite


@dataclass(frozen=True, slots=True)
class LesionCluster:
    cluster_id: str
    history: PrimaryHistoryKey
    molecule_id: str
    lesion_ids: tuple[str, ...]
    classification: str
    causes: tuple[str, ...]
    backbone_breaks: int
    base_lesions: int


@dataclass(frozen=True, slots=True)
class RadiationClusters:
    clusters: tuple[LesionCluster, ...]
    geometry_id: str
    realization_id: str
    maximum_contour_gap: int


def contour_distance(
    left: TargetSite, right: TargetSite, molecule: TargetMolecule
) -> int:
    """Distance on explicitly aligned duplex contour, including circular closure."""
    if (
        left.molecule_id != molecule.molecule_id
        or right.molecule_id != molecule.molecule_id
    ):
        raise ValueError("Contour distance cannot cross molecules.")
    distance = abs(left.contour_position - right.contour_position)
    return (
        min(distance, molecule.contour_length - distance)
        if molecule.circular
        else distance
    )


def cluster_radiation_lesions(
    ledger: InitialLesionLedger,
    geometry: RadiationTargetGeometry,
    *,
    maximum_contour_gap: int,
) -> RadiationClusters:
    """Connected components under an inclusive contour-distance edge rule.

    A DSB cluster contains an actual opposite-strand backbone pair within the
    declared gap. Merely having both strands in a transitive/base-bridged cluster
    does not manufacture a DSB. Histories and fractions are never joined.
    """
    if type(maximum_contour_gap) is not int or maximum_contour_gap < 0:
        raise ValueError("Cluster contour gap must be a nonnegative integer.")
    geometry_id = geometry.fingerprint()
    if ledger.candidates.geometry_id != geometry_id:
        raise ValueError("Lesion topology differs from its original target mapping.")
    sites = {site.target_id: site for site in geometry.sites}
    molecules = {molecule.molecule_id: molecule for molecule in geometry.molecules}
    grouped: dict[tuple[PrimaryHistoryKey, str], list[InitialLesion]] = {}
    for lesion in ledger.lesions:
        if lesion.target_id not in sites:
            raise ValueError("Lesion target is absent from geometry.")
        grouped.setdefault(
            (lesion.history, sites[lesion.target_id].molecule_id), []
        ).append(lesion)
    clusters = []
    for (history, molecule_id), lesions in sorted(grouped.items()):
        lesions.sort(
            key=lambda item: (sites[item.target_id].contour_position, item.lesion_id)
        )
        molecule = molecules[molecule_id]
        parent = list(range(len(lesions)))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        # Sorted adjacent edges generate exactly the contour-distance connected
        # components. The closing edge is essential on a circle; no dense graph.
        adjacent = [(i, i + 1) for i in range(len(lesions) - 1)]
        if molecule.circular and len(lesions) > 1:
            adjacent.append((len(lesions) - 1, 0))
        for i, j in adjacent:
            if (
                contour_distance(
                    sites[lesions[i].target_id], sites[lesions[j].target_id], molecule
                )
                <= maximum_contour_gap
            ):
                parent[root(j)] = root(i)
        # The nearest opposite-strand backbone pair is also adjacent in the
        # backbone-only ordering. Base lesions cannot manufacture a DSB edge.
        backbone_indices = [
            i
            for i, lesion in enumerate(lesions)
            if sites[lesion.target_id].component == "backbone"
        ]
        backbone_pairs = list(zip(backbone_indices, backbone_indices[1:]))
        if molecule.circular and len(backbone_indices) > 1:
            backbone_pairs.append((backbone_indices[-1], backbone_indices[0]))
        dsb_edges = []
        for i, j in backbone_pairs:
            left_site, right_site = (
                sites[lesions[i].target_id],
                sites[lesions[j].target_id],
            )
            if (
                left_site.strand_id != right_site.strand_id
                and contour_distance(left_site, right_site, molecule)
                <= maximum_contour_gap
            ):
                dsb_edges.append((i, j))
        components: dict[int, list[InitialLesion]] = {}
        for index, lesion in enumerate(lesions):
            components.setdefault(root(index), []).append(lesion)
        dsb_roots = {root(i) for i, _ in dsb_edges}
        for representative, members in components.items():
            backbone = sum(
                sites[item.target_id].component == "backbone" for item in members
            )
            base = len(members) - backbone
            classification = (
                "DSB"
                if representative in dsb_roots
                else "base-damage"
                if not backbone
                else "SSB"
                if backbone == 1
                else "SSB-cluster"
            )
            ids = tuple(sorted(item.lesion_id for item in members))
            identity = canonical_fingerprint(
                {"lesions": ids, "geometry": geometry_id, "gap": maximum_contour_gap}
            )
            clusters.append(
                LesionCluster(
                    identity,
                    history,
                    molecule_id,
                    ids,
                    classification,
                    tuple(sorted({cause for item in members for cause in item.causes})),
                    backbone,
                    base,
                )
            )
    return RadiationClusters(
        tuple(sorted(clusters, key=lambda item: item.cluster_id)),
        geometry_id,
        ledger.realization_id,
        maximum_contour_gap,
    )
