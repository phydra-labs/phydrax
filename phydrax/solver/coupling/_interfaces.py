#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Revision-bound physical interface bindings across native support owners.

An :class:`InterfaceBinding` names one physical interface explicitly, binds it
to the authoritative geometry source and revision it lives on, and records the
incidence of every endpoint's native support. Endpoints reference their owners
without converting them: exact mesh scopes through
:class:`~phydrax.meshing.MeshInterfaceAttachment`, analytic
:class:`~phydrax.domain.decomposition.PairedSupport` sides through
:class:`PairedSupportAttachment`, and multiregion sheet views through
:class:`SheetViewAttachment`. Each attachment carries the witness that relates
its support to the authoritative source (a certified geometry association, a
paired-support audit, or the owner's sheet face selection). Matching names,
shapes, or dimensions are never a witness. A law side is tied to its endpoint
by :meth:`InterfaceEndpoint.require_side`: the side's trace sites must lie on
the attached support with the side's outward normal on the attached side.

Two-sided interfaces order their endpoints ``(minus, plus)``; the interface
normal points out of the minus side into the plus side. Junctions keep their
declared incidence order and never expand into pairwise laws. Overlaps carry no
normal. Embedded (mixed-dimensional) incidence states whether the host field is
traced, averaged, or integrated over the embedded support.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import phydrax.axes as cx

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...domain import (
    DomainFunction,
    GraphBatch,
    GridBatch,
    PointBatch,
    SampleLayout,
)
from ...domain.decomposition import (
    PairedSupport,
    PairedSupportEvidence,
    PairingTopology,
    SubdomainCover,
    SubdomainPatch,
)
from ...geometry.multiregion_surface import MultiRegionSheetView, MultiRegionSheetViews
from ...geometry.surface import InterfaceSide
from ...meshing import MeshAssembly, MeshInterfaceAttachment, MeshPart
from ...typing import parse


InterfaceIncidence: TypeAlias = Literal["two-sided", "junction", "overlap", "embedded"]
EmbeddedMeaning: TypeAlias = Literal["trace", "average", "integral"]

type InterfaceAttachment = (
    MeshInterfaceAttachment | PairedSupportAttachment | SheetViewAttachment
)
type InterfaceOwner = MeshPart | MeshAssembly | SubdomainCover | MultiRegionSheetViews


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _identifiers(values: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of identifiers.")
    identifiers = tuple(_identifier(value, name) for value in values)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(identifiers))


def _sheet_revision(views: MultiRegionSheetViews, /) -> str:
    """Revision of every sheet's current geometry under one prepared view set."""
    return canonical_fingerprint(
        {
            "kind": "multiregion-sheet-revision",
            "views": views.views_id,
            "geometry": array_tree_fingerprint(tuple(view.mesh for view in views.views)),
        }
    )


def _sheet_view(
    views: MultiRegionSheetViews, region_ids: tuple[str, str], /
) -> MultiRegionSheetView:
    if not isinstance(views, MultiRegionSheetViews):
        raise TypeError("views must be MultiRegionSheetViews.")
    for view in views.views:
        if view.region_ids == region_ids:
            return view
    raise ValueError(
        f"No manifold sheet view joins the canonical region pair {region_ids!r}."
    )


# Largest number of (site, simplex) pairs measured at once by the nearest-simplex search.
_PAIR_BUDGET = 1 << 22
# Roundoff allowance of a site-to-simplex distance, in units of eps * coordinate scale.
_ROUNDOFF = 64.0


def _support_label(cover: SubdomainCover, patches: tuple[SubdomainPatch, ...], /) -> str:
    """The one ambient coordinate on which the probed patch supports are evaluated."""
    labels = cover.ambient.labels
    if len(labels) != 1 or any(patch.support.deps != labels for patch in patches):
        raise ValueError(
            "Paired-support audits probe the patch supports on one ambient "
            f"coordinate; cover {cover.cover_id!r} has ambient labels {labels!r}."
        )
    return labels[0]


def _inside(patch: SubdomainPatch, label: str, points: np.ndarray, /) -> np.ndarray:
    """Pointwise support membership: one sampled point per row of ``points``."""
    structure = SampleLayout(((label,),)).canonicalize((label,))
    dims = (structure.axis_for(label),) + (None,) * (points.ndim - 1)
    batch = PointBatch(
        frozendict({label: cx.AxisArray(jnp.asarray(points), dims=dims)}), structure
    )
    values = np.asarray(patch.support(batch).data)
    if values.shape != (points.shape[0],):
        raise ValueError(
            f"Patch {patch.patch_id!r} support must be one scalar per point; it "
            f"returned shape {values.shape} for {points.shape[0]} points."
        )
    return values > 0.0


def _separated(
    cover: SubdomainCover,
    own: SubdomainPatch,
    other: SubdomainPatch,
    points: np.ndarray,
    directions: np.ndarray,
    distance: float,
    /,
) -> np.ndarray:
    """Whether ``own`` alone holds each point stepped back along its direction and
    ``other`` alone holds it stepped forward by ``distance``.

    Only a point within ``distance`` of the common boundary of the two patches,
    with ``own`` behind and ``other`` ahead of its direction, passes.
    """
    label = _support_label(cover, (own, other))
    behind = points - distance * directions
    ahead = points + distance * directions
    return (
        _inside(own, label, behind)
        & ~_inside(other, label, behind)
        & _inside(other, label, ahead)
        & ~_inside(own, label, ahead)
    )


def _require_normal_sign(
    cover: SubdomainCover,
    pairing: PairedSupport,
    normal: DomainFunction,
    points: PointBatch | GridBatch | GraphBatch,
    distance: float,
    /,
) -> None:
    """Refuse a pairing normal that does not point out of the left patch into the right."""
    left = cover.patch(pairing.left_patch_id)
    right = cover.patch(pairing.right_patch_id)
    label = _support_label(cover, (left, right))
    directions = np.asarray(normal(points).data, dtype=np.float64)
    separated = np.concatenate(
        [
            _separated(
                cover,
                left,
                right,
                np.asarray(
                    pairing.trace(dict(patch.to_ambient)[label], side=side)(points).data,
                    dtype=np.float64,
                ),
                directions,
                distance,
            )
            for patch, side in ((left, "left"), (right, "right"))
        ]
    )
    failed = int(np.count_nonzero(~separated))
    if failed:
        raise ValueError(
            f"The normal of paired support {pairing.pairing_id!r} does not point out of "
            f"patch {pairing.left_patch_id!r} into patch {pairing.right_patch_id!r} at "
            f"{failed} of {separated.size} probes (probe distance {distance:.3e})."
        )


def _simplex_distance(points: np.ndarray, corners: np.ndarray, /) -> np.ndarray:
    """Distance of points ``(..., d)`` to broadcast segments or triangles ``(..., k, d)``."""
    count = corners.shape[-2]
    edges = ((0, 1),) if count == 2 else ((0, 1), (1, 2), (2, 0))
    distance = np.full(np.broadcast_shapes(points.shape[:-1], corners.shape[:-2]), np.inf)
    for first, second in edges:
        start = corners[..., first, :]
        direction = corners[..., second, :] - start
        offset = points - start
        along = np.clip(
            np.sum(offset * direction, axis=-1) / np.sum(direction * direction, axis=-1),
            0.0,
            1.0,
        )
        distance = np.minimum(
            distance, np.linalg.norm(offset - along[..., None] * direction, axis=-1)
        )
    if count == 3:
        # A point whose plane projection falls inside the triangle is at its height.
        first = corners[..., 1, :] - corners[..., 0, :]
        second = corners[..., 2, :] - corners[..., 0, :]
        normal = np.cross(first, second)
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        offset = points - corners[..., 0, :]
        height = np.sum(offset * normal, axis=-1)
        planar = offset - height[..., None] * normal
        g00 = np.sum(first * first, axis=-1)
        g01 = np.sum(first * second, axis=-1)
        g11 = np.sum(second * second, axis=-1)
        r0 = np.sum(planar * first, axis=-1)
        r1 = np.sum(planar * second, axis=-1)
        determinant = g00 * g11 - g01 * g01
        a = (g11 * r0 - g01 * r1) / determinant
        b = (g00 * r1 - g01 * r0) / determinant
        inside = (a >= 0.0) & (b >= 0.0) & (a + b <= 1.0)
        distance = np.where(inside, np.minimum(distance, np.abs(height)), distance)
    return distance


def _simplex_side(
    role: str,
    corners: np.ndarray,
    authoritative: np.ndarray,
    orientation: int,
    sites: np.ndarray,
    normals: np.ndarray,
    tolerance: float,
    epsilon: float,
    assigned: np.ndarray | None,
    /,
) -> None:
    """Refuse sites off the attached simplices or outward normals off the attached side.

    ``assigned`` names each site's simplex when the owner identifies it exactly;
    otherwise a site passes on any simplex within tolerance on which its outward
    normal has sign ``orientation`` against the authoritative normal.
    """
    if sites.shape[1] != corners.shape[-1]:
        raise ValueError(
            f"Endpoint {role!r} is attached to a {corners.shape[-1]}-D support, but the "
            f"law side's sites are {sites.shape[1]}-D."
        )
    scale = max(float(np.max(np.abs(corners))), float(np.max(np.abs(sites))))
    bound = tolerance + _ROUNDOFF * epsilon * scale
    if assigned is None:
        on = np.zeros((sites.shape[0],), dtype=np.bool_)
        facing = np.zeros((sites.shape[0],), dtype=np.bool_)
        step = max(1, _PAIR_BUDGET // corners.shape[0])
        for start in range(0, sites.shape[0], step):
            block = slice(start, start + step)
            near = _simplex_distance(sites[block, None, :], corners[None]) <= bound
            aligned = orientation * (normals[block] @ authoritative.T) > 0.0
            on[block] = np.any(near, axis=1)
            facing[block] = np.any(near & aligned, axis=1)
    else:
        on = _simplex_distance(sites, corners[assigned]) <= bound
        facing = orientation * np.sum(normals * authoritative[assigned], axis=-1) > 0.0
    if not np.all(on):
        raise ValueError(
            f"Endpoint {role!r}: {int(np.count_nonzero(~on))} of {on.size} law-side "
            f"sites lie off the attached support (tolerance {bound:.3e}); the law "
            "side is not this interface."
        )
    if not np.all(facing):
        raise ValueError(
            f"Endpoint {role!r}: the law side's outward normal is not on the attached "
            f"side (orientation {orientation:+d}) at {int(np.count_nonzero(~facing))} "
            f"of {facing.size} sites; the law side lies across the interface."
        )


@final
class InterfaceSource(StrictModule, NonTrainableState):
    """Authoritative geometry identity and revision of one physical interface.

    ``source_id`` names the owning geometry source (B-Rep model, analytic
    cover, or prepared multiregion view set), ``source_revision`` its exact
    revision, and ``entity_ids`` the owner's entities that carry the interface.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    entity_ids: tuple[str, ...] = eqx.field(static=True)
    source_key: str = eqx.field(static=True)

    def __init__(
        self, source_id: str, source_revision: str, entity_ids: tuple[str, ...], /
    ) -> None:
        source = _identifier(source_id, "source_id")
        revision = _identifier(source_revision, "source_revision")
        entities = _identifiers(entity_ids, "entity_ids")
        self.source_id = source
        self.source_revision = revision
        self.entity_ids = entities
        self.source_key = canonical_fingerprint(
            {
                "kind": "interface-source",
                "source_id": source,
                "source_revision": revision,
                "entity_ids": entities,
            }
        )

    @classmethod
    def paired_support(cls, cover: SubdomainCover, pairing_id: str, /) -> InterfaceSource:
        """The analytic cover as the authority of one of its paired supports."""
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be SubdomainCover.")
        pairing = cover.pairing(_identifier(pairing_id, "pairing_id"))
        return cls(cover.cover_id, cover.revision, (pairing.pairing_id,))

    @classmethod
    def sheet_views(
        cls, views: MultiRegionSheetViews, region_pairs: tuple[tuple[str, str], ...], /
    ) -> InterfaceSource:
        """The prepared multiregion view set as the authority of selected sheets."""
        if not isinstance(region_pairs, tuple) or not region_pairs:
            raise TypeError("region_pairs must be a non-empty tuple of region pairs.")
        entity_ids = tuple(_sheet_view(views, pair).view_id for pair in region_pairs)
        return cls(views.views_id, _sheet_revision(views), entity_ids)


@final
class PairedSupportAttachment(StrictModule, NonTrainableState):
    """One patch side of an analytic paired support, audited against its cover.

    The pairing owner's orientation is authoritative: its optional normal points
    out of the left patch into the right patch, so the left side is ``minus``
    (``orientation = +1``) and the right side ``plus`` (``-1``). Overlap-volume
    pairings and pairings without a normal are unsided (``0``). ``evidence`` is
    the verified map/normal audit on caller-supplied support points.

    The normal's sign is audited against the cover on the same points: at each
    point's left and right ambient images, the point ``tolerance`` behind the
    normal must lie in the left patch support only and the point ``tolerance``
    ahead in the right patch support only. ``tolerance`` is therefore both the
    map-agreement bound and this probe distance, and must be positive; sided
    pairings whose patch supports overlap at the cut cannot witness their side
    and are refused. The audit is sampled, not a proof. Supports are probed on
    the cover's single ambient coordinate.
    """

    cover_id: str = eqx.field(static=True)
    cover_revision: str = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)
    patch_id: str = eqx.field(static=True)
    side: str = eqx.field(static=True)
    topology: PairingTopology = eqx.field(static=True)
    codimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    evidence: PairedSupportEvidence
    orientation: int = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)
    attachment_id: str = eqx.field(static=True)

    def __init__(
        self,
        cover: SubdomainCover,
        pairing_id: str,
        patch_id: str,
        points: PointBatch | GridBatch | GraphBatch,
        /,
        *,
        tolerance: float = 1.0e-8,
    ) -> None:
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be SubdomainCover.")
        pairing = cover.pairing(_identifier(pairing_id, "pairing_id"))
        patch = _identifier(patch_id, "patch_id")
        if patch == pairing.left_patch_id:
            side, sided = "left", 1
        elif patch == pairing.right_patch_id:
            side, sided = "right", -1
        else:
            raise ValueError(
                f"Patch {patch!r} is not an endpoint of paired support {pairing.pairing_id!r}."
            )
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError(
                "tolerance must be finite and positive; it is also the normal probe "
                "distance."
            )
        evidence = pairing.audit(
            points,
            cover.patch(pairing.left_patch_id),
            cover.patch(pairing.right_patch_id),
            tolerance=tolerance_,
        )
        if not evidence.verified:
            raise ValueError(
                f"Paired support {pairing.pairing_id!r} failed its audit: map mismatch "
                f"{evidence.maximum_map_mismatch:.3e}, normal error "
                f"{evidence.maximum_normal_error:.3e}, tolerance {tolerance_:.3e}."
            )
        normal = None if pairing.topology == "overlap-volume" else pairing.normal
        if normal is not None:
            _require_normal_sign(cover, pairing, normal, points, tolerance_)
        oriented = normal is not None
        revision = cover.revision
        witness = canonical_fingerprint(
            {
                "kind": "paired-support-audit",
                "cover_revision": revision,
                "pairing": pairing.pairing_id,
                "scope": evidence.scope,
                "maximum_map_mismatch": evidence.maximum_map_mismatch,
                "maximum_normal_error": evidence.maximum_normal_error,
                "tolerance": tolerance_,
            }
        )
        orientation = sided if oriented else 0
        self.cover_id = cover.cover_id
        self.cover_revision = revision
        self.pairing_id = pairing.pairing_id
        self.patch_id = patch
        self.side = side
        self.topology = pairing.topology
        self.codimension = pairing.codimension
        self.tolerance = tolerance_
        self.evidence = evidence
        self.orientation = orientation
        self.witness_id = witness
        self.attachment_id = canonical_fingerprint(
            {
                "kind": "paired-support-attachment",
                "cover": cover.cover_id,
                "cover_revision": revision,
                "pairing": pairing.pairing_id,
                "patch": patch,
                "side": side,
                "topology": pairing.topology,
                "codimension": pairing.codimension,
                "orientation": orientation,
                "witness": witness,
            }
        )

    def require_current(self, cover: SubdomainCover, /) -> None:
        """Refuse a cover that is not the exact revision this attachment binds."""
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be SubdomainCover.")
        if cover.cover_id != self.cover_id:
            raise ValueError(f"Cover {cover.cover_id!r} does not own this attachment.")
        if cover.revision != self.cover_revision:
            raise ValueError(
                f"Paired-support attachment of cover {self.cover_id!r} is stale: the "
                "cover revision changed."
            )


@final
class SheetViewAttachment(StrictModule, NonTrainableState):
    """One multiregion sheet view, or one declared side of it.

    The sheet owner's orientation is authoritative: view normals point out of
    ``region_ids[0]`` into ``region_ids[1]``, so the first region is the
    ``minus`` side (``orientation = +1``) and the second the ``plus`` side
    (``-1``). Without a declared side the attachment is the whole, unsided sheet.
    The witness is the owner's face-selection identity ``view_id``. ``tolerance``
    bounds the distance of a law side's sites to the sheet triangles.
    """

    views_id: str = eqx.field(static=True)
    surface_revision: str = eqx.field(static=True)
    view_id: str = eqx.field(static=True)
    region_ids: tuple[str, str] = eqx.field(static=True)
    side: InterfaceSide | None = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    attachment_id: str = eqx.field(static=True)

    def __init__(
        self,
        views: MultiRegionSheetViews,
        region_ids: tuple[str, str],
        /,
        *,
        side: InterfaceSide | None = None,
        tolerance: float = 1.0e-8,
    ) -> None:
        if not isinstance(region_ids, tuple) or len(region_ids) != 2:
            raise TypeError("region_ids must be one canonical pair of region IDs.")
        view = _sheet_view(views, region_ids)
        match side:
            case None:
                orientation = 0
            case InterfaceSide.MINUS:
                orientation = 1
            case InterfaceSide.PLUS:
                orientation = -1
            case _:
                raise TypeError("side must be InterfaceSide or None.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        revision = _sheet_revision(views)
        self.views_id = views.views_id
        self.surface_revision = revision
        self.view_id = view.view_id
        self.region_ids = view.region_ids
        self.side = side
        self.orientation = orientation
        self.tolerance = tolerance_
        self.attachment_id = canonical_fingerprint(
            {
                "kind": "sheet-view-attachment",
                "views": views.views_id,
                "surface_revision": revision,
                "view": view.view_id,
                "region_ids": view.region_ids,
                "side": None if side is None else side.value,
                "orientation": orientation,
                "tolerance": tolerance_,
            }
        )

    @property
    def region_id(self) -> str | None:
        """The region on the attached side, or ``None`` for the whole sheet."""
        match self.orientation:
            case 1:
                return self.region_ids[0]
            case -1:
                return self.region_ids[1]
            case _:
                return None

    def require_current(self, views: MultiRegionSheetViews, /) -> None:
        """Refuse a view set that is not the exact revision this attachment binds."""
        if not isinstance(views, MultiRegionSheetViews):
            raise TypeError("views must be MultiRegionSheetViews.")
        if views.views_id != self.views_id:
            raise ValueError("The sheet view set does not own this attachment.")
        if _sheet_revision(views) != self.surface_revision:
            raise ValueError(
                "Sheet attachment is stale: the multiregion surface geometry changed."
            )


class _Facts(NamedTuple):
    """Owner-independent facts of one attachment used by binding validation."""

    owner_id: str
    source_id: str
    source_revision: str
    entity_ids: tuple[str, ...]
    codimension: int
    orientation: int
    witness_id: str
    attachment_id: str


def _facts(attachment: InterfaceAttachment, /) -> _Facts:
    match attachment:
        case MeshInterfaceAttachment():
            return _Facts(
                attachment.part_name,
                attachment.geometry_source_id,
                attachment.geometry_source_revision,
                attachment.geometry_entity_ids,
                attachment.ambient_dimension - attachment.scope.entity_dimension,
                attachment.orientation,
                attachment.association_id,
                attachment.attachment_id,
            )
        case PairedSupportAttachment():
            return _Facts(
                attachment.cover_id,
                attachment.cover_id,
                attachment.cover_revision,
                (attachment.pairing_id,),
                attachment.codimension,
                attachment.orientation,
                attachment.witness_id,
                attachment.attachment_id,
            )
        case SheetViewAttachment():
            return _Facts(
                attachment.views_id,
                attachment.views_id,
                attachment.surface_revision,
                (attachment.view_id,),
                1,
                attachment.orientation,
                attachment.view_id,
                attachment.attachment_id,
            )
        case _:
            raise TypeError(
                "attachment must be MeshInterfaceAttachment, PairedSupportAttachment, "
                "or SheetViewAttachment."
            )


@final
class InterfaceEndpoint(StrictModule, NonTrainableState):
    """One named incidence of an interface and the fields attached to it.

    ``role`` is the endpoint's explicit identity within its binding. ``fields``
    maps field attachment roles (for example ``"value"`` or ``"flux"``) to the
    explicit identities of the numerical fields bound at this endpoint.
    """

    role: str = eqx.field(static=True)
    attachment: InterfaceAttachment
    fields: tuple[tuple[str, str], ...] = eqx.field(static=True)
    endpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        role: str,
        attachment: InterfaceAttachment,
        /,
        *,
        fields: Mapping[str, str],
    ) -> None:
        role_ = _identifier(role, "role")
        facts = _facts(attachment)
        if not isinstance(fields, Mapping):
            raise TypeError("fields must map attachment roles to field identities.")
        pairs = tuple(
            sorted(
                (_identifier(name, "field role"), _identifier(value, "field identity"))
                for name, value in fields.items()
            )
        )
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError("Field attachment roles must be unique after trimming.")
        self.role = role_
        self.attachment = attachment
        self.fields = pairs
        self.endpoint_id = canonical_fingerprint(
            {
                "kind": "interface-endpoint",
                "role": role_,
                "attachment": facts.attachment_id,
                "fields": [list(pair) for pair in pairs],
            }
        )

    @property
    def orientation(self) -> int:
        """Outward-normal relation of this side to the authoritative normal."""
        return _facts(self.attachment).orientation

    @property
    def witness_id(self) -> str:
        return _facts(self.attachment).witness_id

    def field_id(self, role: str, /) -> str:
        """Identity of the field attached under ``role``."""
        for name, value in self.fields:
            if name == role:
                return value
        raise KeyError(f"Endpoint {self.role!r} attaches no field role {role!r}.")

    def require_side(
        self,
        owners: tuple[InterfaceOwner, ...],
        sites: ArrayLike,
        normals: ArrayLike,
        /,
        *,
        facets: tuple[str, ArrayLike] | None = None,
    ) -> None:
        """Refuse a law side whose numerical support is not this endpoint's attachment.

        ``sites`` ``(facets, points, d)`` are the side's ambient quadrature sites
        and ``normals`` its outward normals there; ``facets`` optionally names the
        side's facet entity set and its ``(facets,)`` entity rows. Every site must
        lie on the attached support and the side's outward normal must have sign
        ``orientation`` against the authoritative normal:

        - mesh attachments: when the side acts on the attachment's own entity set
          its facets must equal the attached entities exactly; the sites must lie
          within ``tolerance`` (plus roundoff) of the attached simplices;
        - paired supports (sampled, probe distance ``tolerance``): each site,
          moved ``tolerance`` toward its facet's site centroid, stepped back along
          its outward normal lies in this side's patch support only and stepped
          forward in the opposite patch support only;
        - sheet views: the sites lie within ``tolerance`` of the sheet triangles.

        Unsided endpoints are refused: a side's outward normal has no attached
        side to agree with.
        """
        sites_ = np.asarray(sites)
        normals_ = np.asarray(normals, dtype=np.float64)
        if sites_.ndim != 3 or normals_.shape != sites_.shape:
            raise ValueError(
                "sites and normals must have shape (facets, points, dimension)."
            )
        if not np.issubdtype(sites_.dtype, np.floating):
            raise TypeError("sites must be real floating-point coordinates.")
        if self.orientation == 0:
            raise ValueError(
                f"Endpoint {self.role!r} is unsided; a law side's outward normal has "
                "no attached side to agree with."
            )
        epsilon = float(np.finfo(sites_.dtype).eps)
        points = sites_.reshape(-1, sites_.shape[-1]).astype(np.float64)
        directions = normals_.reshape(points.shape)
        missing = f"No owner of interface endpoint {self.role!r} was supplied."
        match self.attachment:
            case MeshInterfaceAttachment():
                part = _mesh_owner(self.attachment, owners)
                if part is None:
                    raise ValueError(missing)
                rows, corners, authoritative = self.attachment.oriented_simplices(part)
                assigned = None
                if (
                    facets is not None
                    and facets[0] == self.attachment.scope.entity_set_id
                ):
                    assigned = _exact_facets(self.role, rows, facets[1], sites_.shape)
                _simplex_side(
                    self.role,
                    corners,
                    authoritative,
                    self.orientation,
                    points,
                    directions,
                    self.attachment.tolerance,
                    epsilon,
                    assigned,
                )
            case PairedSupportAttachment():
                cover = _cover_owner(self.attachment, owners)
                if cover is None:
                    raise ValueError(missing)
                inner = _toward_facet_center(
                    sites_.astype(np.float64), self.attachment.tolerance
                )
                _paired_side(
                    self.role,
                    self.attachment,
                    cover,
                    inner.reshape(points.shape),
                    directions,
                )
            case SheetViewAttachment():
                views = _views_owner(self.attachment, owners)
                if views is None:
                    raise ValueError(missing)
                view = _sheet_view(views, self.attachment.region_ids)
                corners = np.asarray(view.mesh.vertices, dtype=np.float64)[
                    np.asarray(view.mesh.faces)
                ]
                normal = np.cross(
                    corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
                )
                _simplex_side(
                    self.role,
                    corners,
                    normal / np.linalg.norm(normal, axis=1, keepdims=True),
                    self.orientation,
                    points,
                    directions,
                    self.attachment.tolerance,
                    epsilon,
                    None,
                )
            case _:
                assert_never(self.attachment)


def _toward_facet_center(sites: np.ndarray, distance: float, /) -> np.ndarray:
    """Sites ``(facets, points, d)`` moved ``distance`` toward their facet's centroid.

    On a facet the move is tangential, so a site at a facet end (a crosspoint on
    the edge of the paired support, where roundoff may put it just outside the
    closed patch supports) probes the facet's own interior.
    """
    offset = np.mean(sites, axis=1, keepdims=True) - sites
    length = np.linalg.norm(offset, axis=-1, keepdims=True)
    step = np.minimum(distance, length) / np.where(length > 0.0, length, 1.0)
    return sites + step * offset


def _exact_facets(
    role: str, rows: np.ndarray, facets: ArrayLike, shape: tuple[int, ...], /
) -> np.ndarray:
    """Attached-entity position of every site of a side on the attachment's entity set."""
    side = np.asarray(facets, dtype=np.int64).reshape(-1)
    if side.shape[0] != shape[0]:
        raise ValueError("facets must name one entity row per facet of the sites.")
    extra = np.setdiff1d(side, rows).size
    absent = np.setdiff1d(rows, side).size
    if extra or absent or np.unique(side).size != side.size:
        raise ValueError(
            f"Endpoint {role!r}: the law side acts on {extra} facets outside the "
            f"attached entities and misses {absent} of them; a side on the "
            "attachment's entity set must act on exactly the attached facets."
        )
    order = np.argsort(rows)
    position = order[np.searchsorted(rows[order], side)]
    return np.repeat(position, shape[1])


def _paired_side(
    role: str,
    attachment: PairedSupportAttachment,
    cover: SubdomainCover,
    sites: np.ndarray,
    normals: np.ndarray,
    /,
) -> None:
    pairing = cover.pairing(attachment.pairing_id)
    other = pairing.right_patch_id if attachment.side == "left" else pairing.left_patch_id
    separated = _separated(
        cover,
        cover.patch(attachment.patch_id),
        cover.patch(other),
        sites,
        normals,
        attachment.tolerance,
    )
    failed = int(np.count_nonzero(~separated))
    if failed:
        raise ValueError(
            f"Endpoint {role!r}: {failed} of {separated.size} law-side sites are not "
            f"on paired support {pairing.pairing_id!r} with patch "
            f"{attachment.patch_id!r} behind their outward normal and {other!r} "
            f"ahead (probe distance {attachment.tolerance:.3e}); the law side is not "
            "this interface or lies across it."
        )


def _check_sources(
    source: InterfaceSource,
    endpoints: tuple[InterfaceEndpoint, ...],
    facts: tuple[_Facts, ...],
    /,
) -> None:
    for endpoint, fact in zip(endpoints, facts, strict=True):
        if fact.source_id != source.source_id:
            raise ValueError(
                f"Endpoint {endpoint.role!r} is attached to source {fact.source_id!r}, "
                f"not the interface source {source.source_id!r}."
            )
        if fact.source_revision != source.source_revision:
            raise ValueError(
                f"Endpoint {endpoint.role!r} is attached to source revision "
                f"{fact.source_revision!r}, not the interface revision "
                f"{source.source_revision!r}."
            )
        outside = sorted(set(fact.entity_ids) - set(source.entity_ids))
        if outside:
            raise ValueError(
                f"Endpoint {endpoint.role!r} attaches entities outside the interface "
                f"source: {outside!r}."
            )


def _check_two_sided(
    endpoints: tuple[InterfaceEndpoint, ...], facts: tuple[_Facts, ...], /
) -> None:
    if len(endpoints) != 2:
        raise ValueError(
            "A two-sided interface has exactly one minus and one plus endpoint."
        )
    minus, plus = endpoints
    unsided = tuple(
        endpoint.role
        for endpoint, fact in zip(endpoints, facts, strict=True)
        if fact.orientation == 0
    )
    if unsided:
        raise ValueError(
            f"Two-sided interfaces require sided endpoints; {unsided!r} carry no "
            "oriented normal."
        )
    match (facts[0].orientation, facts[1].orientation):
        case (1, -1):
            pass
        case (-1, 1):
            raise ValueError(
                f"Interface normals are reversed: declared minus endpoint {minus.role!r} "
                f"lies on the plus side and {plus.role!r} on the minus side."
            )
        case _:
            raise ValueError(
                f"Endpoints {minus.role!r} and {plus.role!r} lie on the same side of the "
                "interface normal."
            )
    if facts[0].entity_ids != facts[1].entity_ids:
        raise ValueError("Two-sided endpoints must attach the same interface entities.")


def _check_overlap(
    endpoints: tuple[InterfaceEndpoint, ...], facts: tuple[_Facts, ...], /
) -> None:
    if len(endpoints) < 2:
        raise ValueError("An overlap joins at least two endpoints.")
    for endpoint, fact in zip(endpoints, facts, strict=True):
        if fact.orientation != 0:
            raise ValueError(
                f"An overlap does not acquire a normal; endpoint {endpoint.role!r} is sided."
            )
        if fact.codimension != 0:
            raise ValueError(
                f"Overlap endpoint {endpoint.role!r} is not a codimension-zero support."
            )


def _check_embedded(
    endpoints: tuple[InterfaceEndpoint, ...],
    facts: tuple[_Facts, ...],
    meaning: EmbeddedMeaning | None,
    measure_id: str | None,
    /,
) -> tuple[EmbeddedMeaning, str | None]:
    if len(endpoints) != 2:
        raise ValueError(
            "An embedded incidence joins one host and one embedded endpoint."
        )
    if meaning is None:
        raise ValueError(
            "Embedded incidence must declare whether the host field is traced, "
            "averaged, or integrated over the embedded support."
        )
    meaning_ = parse(meaning, EmbeddedMeaning, "meaning")
    match meaning_:
        case "trace":
            if measure_id is not None:
                raise ValueError("A pointwise trace has no averaging measure.")
            measure = None
        case "average" | "integral":
            if measure_id is None:
                raise ValueError(
                    f"An embedded {meaning_} requires the identity of its measure."
                )
            measure = _identifier(measure_id, "measure_id")
        case _:
            assert_never(meaning_)
    if facts[0].codimension >= facts[1].codimension:
        raise ValueError(
            f"Embedded endpoint {endpoints[1].role!r} must have higher codimension than "
            f"host {endpoints[0].role!r}."
        )
    return meaning_, measure


def _endpoint_facts(endpoints: tuple[InterfaceEndpoint, ...], /) -> tuple[_Facts, ...]:
    """Facts of uniquely named endpoints that attach distinct supports."""
    if not isinstance(endpoints, tuple) or not all(
        isinstance(endpoint, InterfaceEndpoint) for endpoint in endpoints
    ):
        raise TypeError("endpoints must be a tuple of InterfaceEndpoint values.")
    roles = tuple(endpoint.role for endpoint in endpoints)
    if len(set(roles)) != len(roles):
        raise ValueError("Interface endpoint roles must be unique.")
    facts = tuple(_facts(endpoint.attachment) for endpoint in endpoints)
    if len({fact.attachment_id for fact in facts}) != len(facts):
        raise ValueError("Interface endpoints must attach distinct supports.")
    return facts


def _incidence_structure(
    incidence: InterfaceIncidence,
    endpoints: tuple[InterfaceEndpoint, ...],
    facts: tuple[_Facts, ...],
    meaning: EmbeddedMeaning | None,
    measure_id: str | None,
    /,
) -> tuple[tuple[InterfaceEndpoint, ...], EmbeddedMeaning | None, str | None]:
    """Validate one incidence; return canonical endpoints, meaning, and measure."""
    if incidence != "embedded" and (meaning is not None or measure_id is not None):
        raise ValueError(
            "Only embedded incidence declares an averaging or integration meaning."
        )
    match incidence:
        case "two-sided":
            _check_two_sided(endpoints, facts)
            return endpoints, None, None
        case "junction":
            if len(endpoints) < 3:
                raise ValueError("A junction joins at least three incidences.")
            return endpoints, None, None
        case "overlap":
            _check_overlap(endpoints, facts)
            return tuple(sorted(endpoints, key=lambda value: value.role)), None, None
        case "embedded":
            meaning_, measure = _check_embedded(endpoints, facts, meaning, measure_id)
            return endpoints, meaning_, measure
        case _:
            assert_never(incidence)


@final
class InterfaceBinding(StrictModule, NonTrainableState):
    """One explicitly named physical interface bound to its native supports.

    ``incidence`` fixes the endpoint structure: ``"two-sided"`` endpoints are
    ``(minus, plus)`` with the normal pointing out of minus; ``"junction"``
    keeps its declared incidence order (three or more) and implies no pairwise
    law; ``"overlap"`` endpoints are codimension-zero, unsided, and sorted by
    role; ``"embedded"`` endpoints are ``(host, embedded)`` with a declared
    ``meaning`` (and ``measure_id`` for averages and integrals). ``binding_id``
    identifies the interface, its source revision, every endpoint attachment,
    and the attached fields; computational ownership (distribution) is not part
    of it, so redistribution never renames an interface.
    """

    interface_id: str = eqx.field(static=True)
    source: InterfaceSource
    incidence: InterfaceIncidence = eqx.field(static=True)
    endpoints: tuple[InterfaceEndpoint, ...]
    meaning: EmbeddedMeaning | None = eqx.field(static=True)
    measure_id: str | None = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_id: str,
        source: InterfaceSource,
        incidence: InterfaceIncidence,
        endpoints: tuple[InterfaceEndpoint, ...],
        /,
        *,
        meaning: EmbeddedMeaning | None = None,
        measure_id: str | None = None,
    ) -> None:
        identifier = _identifier(interface_id, "interface_id")
        if not isinstance(source, InterfaceSource):
            raise TypeError("source must be InterfaceSource.")
        incidence_ = parse(incidence, InterfaceIncidence, "incidence")
        facts = _endpoint_facts(endpoints)
        _check_sources(source, endpoints, facts)
        ordered, meaning_, measure = _incidence_structure(
            incidence_, endpoints, facts, meaning, measure_id
        )
        self.interface_id = identifier
        self.source = source
        self.incidence = incidence_
        self.endpoints = ordered
        self.meaning = meaning_
        self.measure_id = measure
        self.binding_id = canonical_fingerprint(
            {
                "kind": "interface-binding",
                "interface_id": identifier,
                "source": source.source_key,
                "incidence": incidence_,
                "endpoints": [endpoint.endpoint_id for endpoint in ordered],
                "meaning": meaning_,
                "measure_id": measure,
            }
        )

    @property
    def roles(self) -> tuple[str, ...]:
        """Endpoint roles in canonical incidence order."""
        return tuple(endpoint.role for endpoint in self.endpoints)

    @property
    def witness_ids(self) -> tuple[str, ...]:
        """Witness identity of every endpoint, in canonical incidence order."""
        return tuple(endpoint.witness_id for endpoint in self.endpoints)

    def endpoint(self, role: str, /) -> InterfaceEndpoint:
        for endpoint in self.endpoints:
            if endpoint.role == role:
                return endpoint
        raise KeyError(f"Interface {self.interface_id!r} has no endpoint {role!r}.")

    def _require_sides(self) -> None:
        if self.incidence != "two-sided":
            raise ValueError(
                f"Interface {self.interface_id!r} has {self.incidence} incidence, not "
                "minus and plus sides."
            )

    def side(self, side: InterfaceSide, /) -> InterfaceEndpoint:
        """The minus or plus endpoint of a two-sided interface."""
        self._require_sides()
        match side:
            case InterfaceSide.MINUS:
                return self.endpoints[0]
            case InterfaceSide.PLUS:
                return self.endpoints[1]
            case _:
                raise TypeError("side must be InterfaceSide.")

    def side_of(self, role: str, /) -> InterfaceSide:
        """The side of a two-sided interface on which endpoint ``role`` lies."""
        self._require_sides()
        self.endpoint(role)
        return (
            InterfaceSide.MINUS if role == self.endpoints[0].role else InterfaceSide.PLUS
        )

    def require_current(self, *owners: InterfaceOwner) -> None:
        """Refuse the binding unless every endpoint's owner is supplied at its revision.

        Mesh endpoints accept their :class:`MeshPart` or an owning
        :class:`MeshAssembly`; analytic endpoints their :class:`SubdomainCover`;
        sheet endpoints their :class:`MultiRegionSheetViews`.
        """
        for owner in owners:
            if not isinstance(
                owner, (MeshPart, MeshAssembly, SubdomainCover, MultiRegionSheetViews)
            ):
                raise TypeError(
                    "Interface owners must be MeshPart, MeshAssembly, SubdomainCover, "
                    "or MultiRegionSheetViews values."
                )
        for endpoint in self.endpoints:
            if not _require_owner(endpoint.attachment, owners):
                raise ValueError(
                    f"No owner of interface endpoint {endpoint.role!r} was supplied."
                )


def _mesh_owner(
    attachment: MeshInterfaceAttachment, owners: tuple[InterfaceOwner, ...], /
) -> MeshPart | None:
    """The current part owning a mesh attachment, or ``None`` if not supplied."""
    for owner in owners:
        match owner:
            case MeshPart() if owner.name == attachment.part_name:
                attachment.require_current(owner)
                return owner
            case MeshAssembly() if any(
                part.name == attachment.part_name for part in owner.parts
            ):
                return owner.require_attachment(attachment)
            case _:
                continue
    return None


def _cover_owner(
    attachment: PairedSupportAttachment, owners: tuple[InterfaceOwner, ...], /
) -> SubdomainCover | None:
    """The current cover owning a paired-support attachment, or ``None``."""
    for owner in owners:
        if isinstance(owner, SubdomainCover) and owner.cover_id == attachment.cover_id:
            attachment.require_current(owner)
            return owner
    return None


def _views_owner(
    attachment: SheetViewAttachment, owners: tuple[InterfaceOwner, ...], /
) -> MultiRegionSheetViews | None:
    """The current view set owning a sheet attachment, or ``None``."""
    for owner in owners:
        if (
            isinstance(owner, MultiRegionSheetViews)
            and owner.views_id == attachment.views_id
        ):
            attachment.require_current(owner)
            return owner
    return None


def _require_owner(
    attachment: InterfaceAttachment, owners: tuple[InterfaceOwner, ...], /
) -> bool:
    """Validate the attachment against its supplied owner; ``False`` if absent."""
    match attachment:
        case MeshInterfaceAttachment():
            return _mesh_owner(attachment, owners) is not None
        case PairedSupportAttachment():
            return _cover_owner(attachment, owners) is not None
        case SheetViewAttachment():
            return _views_owner(attachment, owners) is not None
        case _:
            assert_never(attachment)


__all__ = [
    "EmbeddedMeaning",
    "InterfaceBinding",
    "InterfaceEndpoint",
    "InterfaceIncidence",
    "InterfaceSource",
    "PairedSupportAttachment",
    "SheetViewAttachment",
]
