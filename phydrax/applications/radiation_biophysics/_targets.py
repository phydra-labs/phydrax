#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source-linked target spheres compiled by the native geometry owner.

Targets are scoring geometry, not atomistic material. Directed contour coordinates
are explicitly aligned across strands by the caller; target IDs are never atom IDs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...geometry import CompiledGeometry
from ...geometry.analytic import Sphere
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ...units import conversion_factor, METER, UnitDefinition
from ._interactions import (
    _point,
    _text,
    InteractionLedger,
    PhysicalInteraction,
    RadiationEventKey,
    RadiationSource,
)
from ._reactions import ChemicalReaction, ReactionLedger


@dataclass(frozen=True, slots=True)
class TargetMolecule:
    molecule_id: str
    strand_ids: tuple[str, ...]
    contour_length: int
    circular: bool

    def __post_init__(self):
        _text(self.molecule_id, "molecule_id")
        if (
            not isinstance(self.strand_ids, tuple)
            or len(self.strand_ids) not in (1, 2)
            or len(set(self.strand_ids)) != len(self.strand_ids)
        ):
            raise ValueError(
                "Initial target profile supports one or two distinct strands."
            )
        for strand in self.strand_ids:
            _text(strand, "strand_id")
        if type(self.contour_length) is not int or self.contour_length <= 0:
            raise ValueError("Contour length must be a positive integer.")
        if type(self.circular) is not bool:
            raise TypeError("Circularity must be declared explicitly.")


@dataclass(frozen=True, slots=True)
class TargetSite:
    target_id: int
    molecule_id: str
    strand_id: str
    contour_position: int
    component: str
    center: tuple[float, float, float]
    radius: float
    material: str

    def __post_init__(self):
        if type(self.target_id) is not int or not 0 <= self.target_id < 2**63:
            raise ValueError("Derived target IDs must be nonnegative int64 identities.")
        if type(self.contour_position) is not int or self.contour_position < 0:
            raise ValueError("Contour position must be a nonnegative integer.")
        if self.component not in ("backbone", "base"):
            raise ValueError("Supported lesion components are backbone and base.")
        _point(self.center)
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Target radius must be finite and positive.")
        _text(self.material, "target material")


@dataclass(frozen=True, slots=True)
class SourceTargetRoute:
    """One many-to-many scoring route, with explicit deposited-energy allocation."""

    source_site_id: str
    target_id: int
    fraction: float

    def __post_init__(self):
        _text(self.source_site_id, "source site")
        if type(self.target_id) is not int or not 0 <= self.target_id < 2**63:
            raise ValueError("Source routes require a nonnegative int64 target identity.")
        if not math.isfinite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError("Route allocation must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class RadiationTargetGeometry:
    molecules: tuple[TargetMolecule, ...]
    sites: tuple[TargetSite, ...]
    routes: tuple[SourceTargetRoute, ...]
    source_geometry_id: str
    length_unit: UnitDefinition
    source_frame: str
    target_frame: str
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    material_policy: str
    approximation: str
    losses: tuple[AdapterLoss, ...] = ()

    def __post_init__(self):
        for values in (self.molecules, self.sites, self.routes, self.losses):
            if not isinstance(values, tuple):
                raise TypeError("Target support, routes and losses must be tuples.")
        conversion_factor(self.length_unit, METER)
        for value in (
            self.source_geometry_id,
            self.source_frame,
            self.target_frame,
            self.material_policy,
            self.approximation,
        ):
            _text(value, "geometry policy")
        if self.material_policy not in ("scoring-only", "require-match"):
            raise ValueError(
                "Target material policy must be scoring-only or require-match."
            )
        molecules = {item.molecule_id: item for item in self.molecules}
        sites = {item.target_id: item for item in self.sites}
        if (
            not molecules
            or not sites
            or len(molecules) != len(self.molecules)
            or len(sites) != len(self.sites)
        ):
            raise ValueError("Target molecule and site IDs must be nonempty and unique.")
        for site in self.sites:
            if site.molecule_id not in molecules:
                raise ValueError("Target molecule is absent from topology.")
            molecule = molecules[site.molecule_id]
            if (
                site.strand_id not in molecule.strand_ids
                or site.contour_position >= molecule.contour_length
            ):
                raise ValueError("Target strand/contour position is outside topology.")
        totals: dict[str, list[float]] = {}
        pairs = set()
        for route in self.routes:
            if (
                route.target_id not in sites
                or (route.source_site_id, route.target_id) in pairs
            ):
                raise ValueError("Routes require unique existing source-target pairs.")
            pairs.add((route.source_site_id, route.target_id))
            totals.setdefault(route.source_site_id, []).append(route.fraction)
        if any(
            not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-12)
            for values in totals.values()
        ):
            raise ValueError("Source-site deposited-energy allocations must sum to one.")
        if any(loss.changes_interpretation for loss in self.losses):
            raise ValueError(
                "Mapping losses material to target interpretation are refused."
            )
        if (
            not isinstance(self.rotation, tuple)
            or len(self.rotation) != 3
            or any(not isinstance(row, tuple) for row in self.rotation)
        ):
            raise ValueError("Frame rotation must be an immutable 3-by-3 matrix.")
        for row in self.rotation:
            _point(row)
        r = self.rotation
        if any(
            not math.isclose(
                math.fsum(r[i][k] * r[j][k] for k in range(3)),
                float(i == j),
                abs_tol=1e-10,
            )
            for i in range(3)
            for j in range(3)
        ):
            raise ValueError("Frame rotation must be orthogonal.")
        determinant = (
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        )
        if determinant <= 0:
            raise ValueError(
                "Frame transform must be a proper rotation, not a reflection."
            )
        _point(self.translation)

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "molecules": [
                    asdict(item)
                    for item in sorted(self.molecules, key=lambda x: x.molecule_id)
                ],
                "sites": [
                    asdict(item) for item in sorted(self.sites, key=lambda x: x.target_id)
                ],
                "routes": [
                    asdict(item)
                    for item in sorted(
                        self.routes, key=lambda x: (x.source_site_id, x.target_id)
                    )
                ],
                "source": self.source_geometry_id,
                "unit": self.length_unit.unit_id,
                "frames": [self.source_frame, self.target_frame],
                "rotation": self.rotation,
                "translation": self.translation,
                "material": self.material_policy,
                "approximation": self.approximation,
                "losses": [item.loss_id for item in self.losses],
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedRadiationTargets:
    geometry: RadiationTargetGeometry
    spheres: tuple[CompiledGeometry, ...]


def prepare_radiation_targets(
    geometry: RadiationTargetGeometry,
) -> PreparedRadiationTargets:
    """Compile scoring spheres through existing geometry; host-only preparation."""
    return PreparedRadiationTargets(
        geometry,
        tuple(
            Sphere(
                site.center, site.radius, feature_id=f"radiation-target:{site.target_id}"
            ).compile()
            for site in geometry.sites
        ),
    )


@dataclass(frozen=True, slots=True)
class TargetHit:
    event_key: RadiationEventKey
    target_id: int
    fraction: float
    method: str


@dataclass(frozen=True, slots=True)
class TargetMapping:
    source_id: str
    geometry_id: str
    ledger_ids: tuple[str, str]
    hits: tuple[TargetHit, ...]
    unmapped_events: tuple[RadiationEventKey, ...]
    report: AdapterReport


def map_radiation_targets(
    physical: InteractionLedger,
    chemical: ReactionLedger,
    prepared: PreparedRadiationTargets,
    *,
    overlap_policy: str = "error",
    unmapped_policy: str = "error",
    commercial_use: bool = False,
) -> TargetMapping:
    """Map explicit source routes or transformed coordinate hits in bounded batches.

    Coordinate overlaps require an explicit equal-share policy. Unmapped events
    are refused unless the caller declares them outside the scored support. Energy
    is allocated, never copied into every overlapping target. Reaction allocations
    become candidate probabilities, not fractional lesions.
    """
    source: RadiationSource = physical.source
    source.require_rights(commercial_use=commercial_use)
    if source.fingerprint() != chemical.source.fingerprint():
        raise ValueError("Transport and reaction sources/configurations must match.")
    geometry = prepared.geometry
    if source.coordinate_frame != geometry.source_frame:
        raise ValueError("Source coordinate frame has no declared target transform.")
    if overlap_policy not in ("error", "equal-share") or unmapped_policy not in (
        "error",
        "outside",
    ):
        raise ValueError("Unknown target mapping policy.")
    routes: dict[str, list[SourceTargetRoute]] = {}
    for route in geometry.routes:
        routes.setdefault(route.source_site_id, []).append(route)
    records: tuple[PhysicalInteraction | ChemicalReaction, ...] = (
        physical.records + chemical.records
    )
    scale = float(conversion_factor(source.length_unit, geometry.length_unit))
    hits = []
    unmapped = []
    coordinate_records = []
    for record in records:
        if record.source_site_id is not None:
            if record.source_site_id not in routes:
                if unmapped_policy == "outside":
                    unmapped.append(record.key)
                    continue
                raise ValueError("A reported source site has no explicit target route.")
            for route in routes[record.source_site_id]:
                hits.append(
                    TargetHit(record.key, route.target_id, route.fraction, "source-route")
                )
        else:
            coordinate_records.append(record)
    # Bounded temporary point-by-site mask, using the native sphere kernel.
    for start in range(0, len(coordinate_records), 1024):
        batch = coordinate_records[start : start + 1024]
        points = jnp.asarray(
            [
                [
                    math.fsum(
                        geometry.rotation[i][k] * record.position[k] * scale
                        for k in range(3)
                    )
                    + geometry.translation[i]
                    for i in range(3)
                ]
                for record in batch
            ]
        )
        membership = np.asarray(
            jnp.stack([sphere.contains(points) for sphere in prepared.spheres], axis=1)
        )
        for record, row in zip(batch, membership, strict=True):
            selected = [
                site.target_id
                for site, inside in zip(geometry.sites, row, strict=True)
                if inside
            ]
            if not selected:
                unmapped.append(record.key)
            elif len(selected) > 1 and overlap_policy == "error":
                raise ValueError(
                    "Coordinate hit overlaps targets without allocation policy."
                )
            else:
                hits.extend(
                    TargetHit(record.key, target, 1 / len(selected), "coordinate-sphere")
                    for target in selected
                )
    if unmapped and unmapped_policy == "error":
        raise ValueError(
            "Events outside target coverage require an explicit outside policy."
        )
    if geometry.material_policy == "require-match":
        records_by_key = {record.key: record for record in records}
        sites_by_id = {site.target_id: site for site in geometry.sites}
        if any(
            records_by_key[hit.event_key].material != sites_by_id[hit.target_id].material
            for hit in hits
        ):
            raise ValueError(
                "Mapped material is missing or differs from the required target material."
            )
    hits_ = tuple(sorted(hits, key=lambda hit: (hit.event_key, hit.target_id)))
    unmapped_ = tuple(sorted(unmapped))
    losses = geometry.losses
    if unmapped_:
        losses += (
            AdapterLoss(
                "events/outside",
                "import",
                "dropped",
                "Caller declares these events outside scored target support.",
                changes_interpretation=False,
            ),
        )
    geometry_id = geometry.fingerprint()
    target_id = canonical_fingerprint(
        {
            "geometry": geometry_id,
            "hits": [asdict(item) for item in hits_],
            "outside": [asdict(item) for item in unmapped_],
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "external-radiation-ledgers",
        "radiation-target-hits",
        source_id=canonical_fingerprint([physical.fingerprint(), chemical.fingerprint()]),
        target_id=target_id,
        losses=losses,
        coordinate_mapping=(geometry.source_frame, geometry.target_frame),
        assumptions=(
            geometry.material_policy,
            geometry.approximation,
            f"overlap:{overlap_policy}",
            f"unmapped:{unmapped_policy}",
        ),
        preserved_fields=("event_identity", "primary_history", "source_routes"),
    )
    return TargetMapping(
        source.artifact.artifact_id,
        geometry_id,
        (physical.fingerprint(), chemical.fingerprint()),
        hits_,
        unmapped_,
        report,
    )
