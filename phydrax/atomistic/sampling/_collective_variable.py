#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Metric- and image-correct collective variables on certified geometry.

Every CV evaluates one :class:`CollectiveVariableGeometry` (current wrapped and
material coordinates, current cell vectors and bound neighborhood views) and
reports ``successful`` (the value is defined), ``regular`` (its first
derivative exists) and an explicit :class:`CollectiveVariableStatus` word.
Point-pair vectors use the certified minimum image; groups are placed on one
declared branch (accepted unwrapped coordinates, bond-tree reconstruction or
an extent-certified anchor) before center/gyration/alignment.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable
from enum import StrEnum
from typing import assert_never

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import IdentityWords
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
)
from .._sites import AtomisticSiteDomain
from .._system import PreparedAtomisticSystem
from ._alignment import aligned_deviation
from ._cv_geometry import (
    CollectiveVariableGeometry,
    CollectiveVariableImageBranch,
    CollectiveVariableStatus,
    PreparedGroupBranch,
    reduce_status,
    safe_norm,
    status_flag,
    VALUE_FAILURES,
)


class CollectiveVariableKind(StrEnum):
    DISTANCE = "distance"
    ANGLE = "angle"
    TORSION = "torsion"
    CENTER_OF_MASS_DISTANCE = "center-of-mass-distance"
    RADIUS_OF_GYRATION = "radius-of-gyration"
    COORDINATION = "coordination"
    GROUP_COORDINATION = "group-coordination"
    CONTACT_SIMILARITY = "contact-similarity"
    ALIGNED_RMSD = "aligned-rmsd"
    ALIGNED_MSD = "aligned-msd"
    CONFIGURATION = "configuration"
    CELL_VOLUME = "cell-volume"
    DENSITY = "density"
    PATH_PROGRESS = "path-progress"
    PATH_DISTANCE = "path-distance"


_POINT_KINDS = (
    CollectiveVariableKind.DISTANCE,
    CollectiveVariableKind.ANGLE,
    CollectiveVariableKind.TORSION,
    CollectiveVariableKind.COORDINATION,
    CollectiveVariableKind.CONTACT_SIMILARITY,
)
_GROUP_KINDS = (
    CollectiveVariableKind.CENTER_OF_MASS_DISTANCE,
    CollectiveVariableKind.RADIUS_OF_GYRATION,
    CollectiveVariableKind.ALIGNED_RMSD,
    CollectiveVariableKind.ALIGNED_MSD,
    CollectiveVariableKind.CONFIGURATION,
    CollectiveVariableKind.PATH_PROGRESS,
    CollectiveVariableKind.PATH_DISTANCE,
)
_CELL_KINDS = (CollectiveVariableKind.CELL_VOLUME, CollectiveVariableKind.DENSITY)
_EXACT_POINT_COUNTS = {
    CollectiveVariableKind.DISTANCE: 2,
    CollectiveVariableKind.ANGLE: 3,
    CollectiveVariableKind.TORSION: 4,
}
# Two-group CVs: declared parameter count; the last parameter is the split.
_SPLIT_PARAMETER_COUNTS = {
    CollectiveVariableKind.CENTER_OF_MASS_DISTANCE: 1,
    CollectiveVariableKind.GROUP_COORDINATION: 4,
}


def _admit_group_split(
    kind: CollectiveVariableKind, index: np.ndarray, params: np.ndarray, /
) -> int:
    """Integer split of a two-group CV into two non-empty groups."""
    if params.size != _SPLIT_PARAMETER_COUNTS[kind] or params[-1] != int(params[-1]):
        raise ValueError(f"{kind.value} requires its integer group split.")
    split = int(params[-1])
    if index.ndim != 1 or not 0 < split < index.size:
        raise ValueError("Both groups must be non-empty.")
    if kind is CollectiveVariableKind.GROUP_COORDINATION and (
        params[0] <= 0.0 or params[1] <= 0.0 or params[2] <= params[0]
    ):
        raise ValueError("Group coordination requires 0 < r0 < cutoff and power > 0.")
    return split


def _admit_pair_layout(
    kind: CollectiveVariableKind,
    index: np.ndarray,
    params: np.ndarray,
    ref: np.ndarray,
    /,
) -> None:
    """Pair-list CVs: whole index pairs and positive switching parameters."""
    match kind:
        case CollectiveVariableKind.COORDINATION:
            if index.size == 0 or index.size % 2 or params.size != 2:
                raise ValueError("Coordination requires pairs, distance, and power.")
            if params[0] <= 0.0 or params[1] <= 0.0:
                raise ValueError("Coordination distance and power must be positive.")
        case CollectiveVariableKind.CONTACT_SIMILARITY:
            if (
                index.size == 0
                or index.size % 2
                or ref.size != index.size // 2
                or params.size != 1
                or params[0] <= 0.0
            ):
                raise ValueError("Contact similarity requires pairs, targets, and scale.")
        case _:
            raise RuntimeError(f"{kind.value} is not a pair-list CV.")


def _admit_frame_layout(
    kind: CollectiveVariableKind, index: np.ndarray, ref: np.ndarray, /
) -> None:
    """Single-group CVs: gyration, aligned-reference and configuration frames."""
    match kind:
        case CollectiveVariableKind.RADIUS_OF_GYRATION:
            if index.size == 0:
                raise ValueError("Radius of gyration requires at least one index.")
        case CollectiveVariableKind.ALIGNED_RMSD | CollectiveVariableKind.ALIGNED_MSD:
            if index.ndim != 1 or index.size < 3 or ref.shape != (index.size, 3):
                raise ValueError("Alignment requires three or more reference points.")
        case CollectiveVariableKind.CONFIGURATION:
            if index.ndim != 1 or index.size < 3:
                raise ValueError("A configuration CV requires three or more points.")
        case _:
            raise RuntimeError(f"{kind.value} is not a single-group frame CV.")


def _admit_path_layout(
    index: np.ndarray, params: np.ndarray, ref: np.ndarray, aligned: bool, /
) -> None:
    """Path CVs: at least two whole reference frames and a positive scale."""
    frame_size = index.size * 3
    if (
        index.ndim != 1
        or index.size == 0
        or (aligned and index.size < 3)
        or ref.size < 2 * frame_size
        or ref.size % frame_size
        or params.size != 1
        or params[0] <= 0.0
    ):
        raise ValueError("Path CV requires at least two images and positive scale.")


def _admit_layout(
    kind: CollectiveVariableKind,
    index: np.ndarray,
    params: np.ndarray,
    ref: np.ndarray,
    aligned: bool,
    /,
) -> int | None:
    """Kind-specific index/parameter/reference layout; returns the group split.

    Each kind owns exactly one layout contract; only path CVs may declare
    aligned frame comparison.
    """
    split = None
    path = False
    match kind:
        case (
            CollectiveVariableKind.DISTANCE
            | CollectiveVariableKind.ANGLE
            | CollectiveVariableKind.TORSION
        ):
            count = _EXACT_POINT_COUNTS[kind]
            if index.ndim != 1 or index.size != count:
                raise ValueError(f"{kind.value} requires {count} indices.")
        case (
            CollectiveVariableKind.CENTER_OF_MASS_DISTANCE
            | CollectiveVariableKind.GROUP_COORDINATION
        ):
            split = _admit_group_split(kind, index, params)
        case (
            CollectiveVariableKind.COORDINATION
            | CollectiveVariableKind.CONTACT_SIMILARITY
        ):
            _admit_pair_layout(kind, index, params, ref)
        case CollectiveVariableKind.PATH_PROGRESS | CollectiveVariableKind.PATH_DISTANCE:
            path = True
            _admit_path_layout(index, params, ref, aligned)
        case (
            CollectiveVariableKind.RADIUS_OF_GYRATION
            | CollectiveVariableKind.ALIGNED_RMSD
            | CollectiveVariableKind.ALIGNED_MSD
            | CollectiveVariableKind.CONFIGURATION
        ):
            _admit_frame_layout(kind, index, ref)
        case CollectiveVariableKind.CELL_VOLUME | CollectiveVariableKind.DENSITY:
            pass
        case _:
            assert_never(kind)
    if aligned and not path:
        raise ValueError("Only path CVs declare aligned frame comparison.")
    return split


def _admit_branch(
    kind: CollectiveVariableKind,
    branch: CollectiveVariableImageBranch | None,
    maximum_extent: float | None,
    /,
) -> CollectiveVariableImageBranch:
    """Declared (or the kind's default) image branch and its extent contract."""
    if branch is None:
        branch_ = (
            CollectiveVariableImageBranch.UNWRAPPED
            if kind in _GROUP_KINDS
            else CollectiveVariableImageBranch.MINIMUM_IMAGE
        )
    else:
        branch_ = branch
    if not isinstance(branch_, CollectiveVariableImageBranch):
        raise TypeError("branch must be a CollectiveVariableImageBranch.")
    if kind in _GROUP_KINDS and branch_ is CollectiveVariableImageBranch.MINIMUM_IMAGE:
        raise ValueError("Group CVs require an unwrapped, topology or anchored branch.")
    if kind not in _GROUP_KINDS and branch_ not in (
        CollectiveVariableImageBranch.MINIMUM_IMAGE,
        CollectiveVariableImageBranch.UNWRAPPED,
    ):
        raise ValueError("Point CVs use minimum-image or unwrapped vectors.")
    if (branch_ is CollectiveVariableImageBranch.ANCHORED) != (
        maximum_extent is not None
    ):
        raise ValueError("maximum_extent is required by, and only by, anchored groups.")
    return branch_


def _admit_metric(
    kind: CollectiveVariableKind, metric: CollectiveVariableMetric | None, /
) -> CollectiveVariableMetric:
    """Declared (or the kind's default) metric; a torsion has period 2 pi."""
    metric_ = (
        CollectiveVariableMetric(
            period=2.0 * np.pi if kind is CollectiveVariableKind.TORSION else None
        )
        if metric is None
        else metric
    )
    if kind is CollectiveVariableKind.TORSION and metric_.period != 2.0 * np.pi:
        raise ValueError("A torsion is an angle with period 2 pi.")
    return metric_


class CollectiveVariableMetric(StrictModule, NonTrainableState):
    """Displacement metric of one CV component.

    A periodic metric maps differences to the nearest image in
    ``[-period/2, period/2)``; consumers needing a smooth branch (walls,
    profiles) declare their branch interval explicitly.
    """

    periodic: bool = eqx.field(static=True)
    period: float | None = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(self, /, *, period: float | None = None) -> None:
        period_ = None if period is None else float(period)
        if period_ is not None and (not np.isfinite(period_) or period_ <= 0.0):
            raise ValueError("CV period must be finite and positive.")
        self.periodic = period_ is not None
        self.period = period_
        self.metric_id = canonical_fingerprint({"kind": "cv-metric", "period": period_})

    def difference(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        delta = jnp.asarray(left) - jnp.asarray(right)
        if self.period is None:
            return delta
        period = jnp.asarray(self.period, dtype=delta.dtype)
        return jnp.mod(delta + 0.5 * period, period) - 0.5 * period

    def seam_margin(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Distance of a nearest-image difference from the periodic seam."""

        delta = self.difference(left, right)
        if self.period is None:
            return jnp.full(delta.shape, jnp.inf, dtype=delta.dtype)
        return 0.5 * self.period - jnp.abs(delta)


class CollectiveVariableEvaluation(StrictModule):
    """One CV value (scalar or a flat configuration vector) with evidence.

    ``cv_identity`` carries the evaluated prepared CV's exact digest as
    dynamic words (``cv_id`` decodes it on the host).
    """

    value: Array
    successful: Array
    regular: Array
    status: Array
    branch_margin: Array
    cv_identity: IdentityWords

    @property
    def cv_id(self) -> str:
        """Host decoding of the evaluated prepared-CV digest."""
        return self.cv_identity.fingerprint()


class CollectiveVariableProgramEvaluation(StrictModule):
    """Stacked program values ``(output_size,)`` and per-CV evidence.

    ``program_identity`` is the evaluated program's exact digest as dynamic
    words (``program_id`` decodes it on the host).
    """

    values: Array
    successful: Array
    regular: Array
    status: Array
    branch_margins: Array
    program_identity: IdentityWords

    @property
    def program_id(self) -> str:
        """Host decoding of the evaluated CV-program digest."""
        return self.program_identity.fingerprint()


class AbstractCollectiveVariablePlan(StrictModule, NonTrainableState):
    """Declared CV; ``cv_identity`` is its exact canonical digest as words."""

    cv_identity: eqx.AbstractVar[IdentityWords]
    metric: eqx.AbstractVar[CollectiveVariableMetric]

    @property
    def cv_id(self) -> str:
        """Host decoding of the canonical CV digest."""
        return self.cv_identity.fingerprint()

    @abc.abstractmethod
    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedCollectiveVariable":
        raise NotImplementedError


class AbstractCollectiveVariableProgram(StrictModule):
    """Prepared CV program of one system.

    ``support`` (host) lists the physical particle slots its value depends on
    (empty for interaction-site or cell-only programs); ``requirements`` the
    bundle neighborhoods a consumer term must bind.  The PyTree is
    layout-only: ``program_identity`` (exact program digest) and
    ``system_identity`` (the bound prepared system) are dynamic words decoded
    on the host by ``program_id``/``system_id``; the support is derived on the
    host from the dynamic index leaves instead of a static route table.
    """

    output_size: eqx.AbstractVar[int]
    names: eqx.AbstractVar[tuple[str, ...]]
    metrics: eqx.AbstractVar[tuple[CollectiveVariableMetric, ...]]
    program_identity: eqx.AbstractVar[IdentityWords]
    system_identity: eqx.AbstractVar[IdentityWords]
    requirements: eqx.AbstractVar[tuple[AtomisticNeighborhoodRequirement, ...]]
    cell_dependent: eqx.AbstractVar[bool]

    @property
    def program_id(self) -> str:
        """Host decoding of the canonical CV-program digest."""
        return self.program_identity.fingerprint()

    @property
    def system_id(self) -> str:
        """Host decoding of the bound prepared-system digest."""
        return self.system_identity.fingerprint()

    @property
    @abc.abstractmethod
    def support(self) -> tuple[int, ...]:
        """Host: sorted physical particle slots the program value depends on."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self, geometry: CollectiveVariableGeometry, /
    ) -> CollectiveVariableProgramEvaluation:
        raise NotImplementedError


class CollectiveVariablePlan(AbstractCollectiveVariablePlan):
    """Declared CV.

    ``parameters`` per kind: center-of-mass ``[split]``; coordination
    ``[r0, power]``; group coordination ``[r0, power, cutoff, split]``;
    contact similarity ``[scale]``; path CVs ``[lambda]``.  ``reference`` holds
    contact targets, alignment reference ``(n, 3)`` or path frames.  ``branch``
    selects the image branch (default: minimum image for point-pair CVs,
    accepted unwrapped coordinates for groups); ``maximum_extent`` belongs to
    anchored reconstruction; ``aligned`` aligns path frames before comparing.

    PyTree: ``kind``, ``domain``, ``branch`` and ``aligned`` are static method
    selectors and ``group_split`` (the integer group split of center-of-mass
    and group-coordination CVs) is declared layout because it fixes the group
    shapes; indices, parameters, reference and the anchored
    ``maximum_extent`` are dynamic leaves, and the exact canonical CV digest
    is the dynamic ``cv_identity`` (``cv_id`` decodes it on the host).
    """

    kind: CollectiveVariableKind = eqx.field(static=True)
    indices: Array
    parameters: Array
    reference: Array
    domain: AtomisticSiteDomain = eqx.field(static=True)
    metric: CollectiveVariableMetric
    branch: CollectiveVariableImageBranch = eqx.field(static=True)
    maximum_extent: Array | None
    aligned: bool = eqx.field(static=True)
    group_split: int | None = eqx.field(static=True)
    cv_identity: IdentityWords

    def __init__(
        self,
        kind: CollectiveVariableKind,
        indices: npt.ArrayLike,
        /,
        *,
        parameters: npt.ArrayLike = (),
        reference: npt.ArrayLike = (),
        domain: AtomisticSiteDomain = AtomisticSiteDomain.DOF_ATOMS,
        metric: CollectiveVariableMetric | None = None,
        branch: CollectiveVariableImageBranch | None = None,
        maximum_extent: float | None = None,
        aligned: bool = False,
    ) -> None:
        if not isinstance(kind, CollectiveVariableKind):
            raise TypeError("kind must be CollectiveVariableKind.")
        index = np.asarray(indices)
        if index.ndim not in (1, 2) or not np.issubdtype(index.dtype, np.integer):
            raise TypeError("CV indices must be rank-one or rank-two integers.")
        if kind in _CELL_KINDS and index.size:
            raise ValueError("Cell CVs take no particle indices.")
        params = np.asarray(parameters, dtype=np.float64).reshape((-1,))
        ref = np.asarray(reference, dtype=np.float64)
        if (
            not isinstance(domain, AtomisticSiteDomain)
            or np.any(~np.isfinite(params))
            or np.any(~np.isfinite(ref))
        ):
            raise ValueError("CV domain, parameters, and reference must be valid.")
        split = _admit_layout(kind, index, params, ref, aligned)
        if domain is AtomisticSiteDomain.INTERACTION_SITES and kind not in _POINT_KINDS:
            raise ValueError(
                "Interaction-site CVs are point-pair CVs; mass-weighted, group and "
                "neighborhood CVs require the physical DOF atom domain."
            )
        branch_ = _admit_branch(kind, branch, maximum_extent)
        metric_ = _admit_metric(kind, metric)
        self.kind = kind
        self.indices = jnp.asarray(index, dtype=jnp.int32)
        self.parameters = jnp.asarray(params)
        self.reference = jnp.asarray(ref)
        self.domain = domain
        self.metric = metric_
        self.branch = branch_
        extent = None if maximum_extent is None else float(maximum_extent)
        self.maximum_extent = None if extent is None else jnp.asarray(extent)
        self.aligned = bool(aligned)
        self.group_split = split
        self.cv_identity = IdentityWords.of(
            {
                "kind": "atomistic-cv",
                "cv_kind": kind.value,
                "indices": array_tree_fingerprint(index),
                "parameters": array_tree_fingerprint(params),
                "reference": array_tree_fingerprint(ref),
                "domain": domain.value,
                "metric": metric_.metric_id,
                "branch": branch_.value,
                "maximum_extent": extent,
                "aligned": self.aligned,
            }
        )

    @property
    def size(self) -> int:
        if self.kind is CollectiveVariableKind.CONFIGURATION:
            return 3 * int(self.indices.size)
        return 1

    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedCollectiveVariable":
        capacity = (
            system.capacity
            if self.domain is not AtomisticSiteDomain.INTERACTION_SITES
            else system.coordinate_map.plan.sites.capacity
        )
        index = np.asarray(self.indices, dtype=np.int64).reshape((-1,))
        if index.size and (index.min() < 0 or index.max() >= capacity):
            raise ValueError("CV index exceeds selected coordinate domain.")
        if (
            self.domain is not AtomisticSiteDomain.INTERACTION_SITES
            and index.size
            and np.any(~np.asarray(system.active_mask)[index])
        ):
            raise ValueError("CV indices may reference only active particle slots.")
        cell = system.cell
        if self.kind in _CELL_KINDS and (cell is None or cell.rank != 3):
            raise ValueError("Cell CVs require a three-dimensional periodic cell.")
        return PreparedCollectiveVariable(self, system)


class PreparedCollectiveVariable(StrictModule, NonTrainableState):
    """One CV bound to a prepared system.

    PyTree: the exact prepared-CV digest is the dynamic ``prepared_identity``
    (``prepared_id`` decodes it on the host) and ``support`` is a host
    property derived from the dynamic indices.  A group-coordination
    neighborhood requirement is owned by the layout-only label
    ``"cv:group-coordination"``: its identity is fixed by role and the
    resource-defining cutoff, so equal-cutoff CVs share one neighborhood view
    and no content digest enters the static requirement.
    """

    plan: CollectiveVariablePlan
    system: PreparedAtomisticSystem
    groups: tuple[PreparedGroupBranch, ...]
    requirement: AtomisticNeighborhoodRequirement | None
    prepared_identity: IdentityWords

    def __init__(
        self, plan: CollectiveVariablePlan, system: PreparedAtomisticSystem, /
    ) -> None:
        index = np.asarray(plan.indices, dtype=np.int64).reshape((-1,))
        groups: tuple[PreparedGroupBranch, ...] = ()
        kind = plan.kind
        if kind in _GROUP_KINDS:
            if kind is CollectiveVariableKind.CENTER_OF_MASS_DISTANCE:
                split = plan.group_split
                parts = (index[:split], index[split:])
            else:
                parts = (index,)
            extent = (
                None
                if plan.maximum_extent is None
                else float(np.asarray(plan.maximum_extent))
            )
            groups = tuple(
                PreparedGroupBranch(plan.branch, part, system, maximum_extent=extent)
                for part in parts
            )
        requirement = None
        if kind is CollectiveVariableKind.GROUP_COORDINATION:
            requirement = AtomisticNeighborhoodRequirement(
                "cv:group-coordination",
                AtomisticNeighborhoodRole.COORDINATION,
                float(np.asarray(plan.parameters)[2]),
            )
        self.plan = plan
        self.system = system
        self.groups = groups
        self.requirement = requirement
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-cv",
                "plan": plan.cv_id,
                "system": system.prepared_id,
                "groups": [group.branch_id for group in groups],
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-CV digest."""
        return self.prepared_identity.fingerprint()

    @property
    def support(self) -> tuple[int, ...]:
        """Host: sorted physical particle slots of this CV (none for sites)."""
        if self.plan.domain is AtomisticSiteDomain.INTERACTION_SITES:
            return ()
        index = np.asarray(self.plan.indices, dtype=np.int64).reshape((-1,))
        return tuple(sorted(set(int(value) for value in index)))

    def _coordinates(
        self, geometry: CollectiveVariableGeometry, /
    ) -> tuple[Array, Array, Array]:
        """Point coordinates of the CV domain plus their realization status."""

        if self.plan.domain is not AtomisticSiteDomain.INTERACTION_SITES:
            source = (
                geometry.unwrapped_positions
                if self.plan.branch is CollectiveVariableImageBranch.UNWRAPPED
                else geometry.positions
            )
            return source, jnp.asarray(True), jnp.int32(0)
        sites = geometry.site_positions
        successful = geometry.site_successful
        if sites is None:
            cell = geometry.cell
            vectors = geometry.cell_vectors
            fractional = (
                None
                if cell is None or vectors is None
                else cell.fractional_with_vectors(geometry.positions, vectors)
            )
            state = self.system.coordinate_map.realize(
                geometry.positions,
                cell=cell,
                fractional_positions=fractional,
                cell_vectors=vectors,
            )
            sites, successful = state.positions, state.successful
        return (
            sites,
            successful,
            status_flag(~successful, CollectiveVariableStatus.SITE_REALIZATION),
        )

    def _vector(
        self, geometry: CollectiveVariableGeometry, left: Array, right: Array, /
    ) -> tuple[Array, Array, Array]:
        vector = left - right
        if self.plan.branch is CollectiveVariableImageBranch.UNWRAPPED:
            return (
                vector,
                jnp.full(vector.shape[:-1], jnp.inf, dtype=vector.dtype),
                jnp.int32(0),
            )
        image, margin = geometry.minimum_image(vector)
        return (
            image,
            margin,
            status_flag(jnp.any(margin <= 0.0), CollectiveVariableStatus.IMAGE_AMBIGUOUS),
        )

    def evaluate(
        self, geometry: CollectiveVariableGeometry, /
    ) -> CollectiveVariableEvaluation:
        if not isinstance(geometry, CollectiveVariableGeometry):
            raise TypeError("CVs evaluate a CollectiveVariableGeometry.")
        kind = self.plan.kind
        index = self.plan.indices
        dtype = geometry.positions.dtype
        infinite = jnp.asarray(jnp.inf, dtype=dtype)
        status = jnp.int32(0)
        margin = infinite
        regular = jnp.asarray(True)
        if kind in _POINT_KINDS:
            coordinates, _, status = self._coordinates(geometry)
        if kind is CollectiveVariableKind.DISTANCE:
            vector, image_margin, flag = self._vector(
                geometry, coordinates[index[0]], coordinates[index[1]]
            )
            value, positive = safe_norm(vector)
            status = (
                status
                | flag
                | status_flag(~positive, CollectiveVariableStatus.ZERO_LENGTH)
            )
            margin = jnp.minimum(image_margin, value)
            regular = positive
        elif kind is CollectiveVariableKind.ANGLE:
            left, m0, f0 = self._vector(
                geometry, coordinates[index[0]], coordinates[index[1]]
            )
            right, m1, f1 = self._vector(
                geometry, coordinates[index[2]], coordinates[index[1]]
            )
            left_norm, left_ok = safe_norm(left)
            right_norm, right_ok = safe_norm(right)
            cross_norm, not_collinear = safe_norm(jnp.cross(left, right))
            dot = jnp.sum(left * right)
            value = jnp.arctan2(cross_norm, jnp.where(left_ok & right_ok, dot, 1.0))
            status = (
                status
                | f0
                | f1
                | status_flag(~(left_ok & right_ok), CollectiveVariableStatus.ZERO_LENGTH)
                | status_flag(~not_collinear, CollectiveVariableStatus.COLLINEAR)
            )
            margin = jnp.minimum(jnp.minimum(m0, m1), jnp.minimum(left_norm, right_norm))
            regular = left_ok & right_ok & not_collinear
        elif kind is CollectiveVariableKind.TORSION:
            b0, m0, f0 = self._vector(
                geometry, coordinates[index[0]], coordinates[index[1]]
            )
            b1, m1, f1 = self._vector(
                geometry, coordinates[index[2]], coordinates[index[1]]
            )
            b2, m2, f2 = self._vector(
                geometry, coordinates[index[3]], coordinates[index[2]]
            )
            axis_norm, axis_ok = safe_norm(b1)
            axis = b1 / jnp.where(axis_ok, axis_norm, 1.0)
            v = b0 - jnp.sum(b0 * axis) * axis
            w = b2 - jnp.sum(b2 * axis) * axis
            v_norm, v_ok = safe_norm(v)
            w_norm, w_ok = safe_norm(w)
            ok = axis_ok & v_ok & w_ok
            y = jnp.sum(jnp.cross(axis, v) * w)
            x = jnp.sum(v * w)
            value = jnp.arctan2(jnp.where(ok, y, 0.0), jnp.where(ok, x, 1.0))
            status = (
                status
                | f0
                | f1
                | f2
                | status_flag(~axis_ok, CollectiveVariableStatus.ZERO_LENGTH)
                | status_flag(~(v_ok & w_ok), CollectiveVariableStatus.COLLINEAR)
            )
            margin = jnp.minimum(
                jnp.minimum(jnp.minimum(m0, m1), m2),
                jnp.minimum(axis_norm, jnp.minimum(v_norm, w_norm)),
            )
            regular = ok
        elif kind is CollectiveVariableKind.COORDINATION:
            pairs = index.reshape((-1, 2))
            vector, image_margin, flag = self._vector(
                geometry, coordinates[pairs[:, 0]], coordinates[pairs[:, 1]]
            )
            distance, _ = safe_norm(vector)
            r0, power = self.plan.parameters[0], self.plan.parameters[1]
            value = jnp.sum(1.0 / (1.0 + (distance / r0) ** power))
            status = status | flag
            margin = jnp.min(image_margin)
        elif kind is CollectiveVariableKind.CONTACT_SIMILARITY:
            pairs = index.reshape((-1, 2))
            vector, image_margin, flag = self._vector(
                geometry, coordinates[pairs[:, 0]], coordinates[pairs[:, 1]]
            )
            distance, positive = safe_norm(vector)
            target = self.plan.reference.reshape((-1,)).astype(dtype)
            scale = self.plan.parameters[0]
            value = jnp.mean(jnp.exp(-(((distance - target) / scale) ** 2)))
            status = status | flag
            margin = jnp.min(image_margin)
            regular = jnp.all(positive)
        elif kind is CollectiveVariableKind.GROUP_COORDINATION:
            value, margin, status = self._group_coordination(geometry)
        elif kind in _CELL_KINDS:
            volume = geometry.volume()
            if kind is CollectiveVariableKind.CELL_VOLUME:
                value = volume
            else:
                value = (
                    jnp.sum(
                        jnp.where(self.system.active_mask, self.system.plan.masses, 0.0)
                    )
                    / volume
                )
        else:
            value, margin, status, regular = self._group_value(geometry)
        finite = jnp.all(jnp.isfinite(value))
        status = status | status_flag(~finite, CollectiveVariableStatus.NONFINITE)
        successful = (status & VALUE_FAILURES) == 0
        return CollectiveVariableEvaluation(
            value=jnp.asarray(value),
            successful=successful,
            regular=successful & regular,
            status=jnp.asarray(status, dtype=jnp.int32),
            branch_margin=jnp.asarray(margin, dtype=dtype),
            cv_identity=self.prepared_identity,
        )

    def _group_value(
        self, geometry: CollectiveVariableGeometry, /
    ) -> tuple[Array, Array, Array, Array]:
        kind = self.plan.kind
        placed = tuple(group.coordinates(geometry) for group in self.groups)
        status = jnp.int32(0)
        margin = jnp.asarray(jnp.inf, dtype=geometry.positions.dtype)
        for _, group_margin, group_status in placed:
            status = status | group_status
            margin = jnp.minimum(margin, group_margin)
        regular = jnp.asarray(True)
        masses = self.system.plan.masses
        if kind is CollectiveVariableKind.CENTER_OF_MASS_DISTANCE:
            centers = []
            for group, (coordinates, _, _) in zip(self.groups, placed, strict=True):
                mass = masses[group.indices].astype(coordinates.dtype)
                centers.append(
                    jnp.sum(mass[:, None] * coordinates, axis=0) / jnp.sum(mass)
                )
            vector, image_margin = geometry.minimum_image(centers[0] - centers[1])
            value, positive = safe_norm(vector)
            status = (
                status
                | status_flag(
                    image_margin <= 0.0, CollectiveVariableStatus.IMAGE_AMBIGUOUS
                )
                | status_flag(~positive, CollectiveVariableStatus.ZERO_LENGTH)
            )
            return value, jnp.minimum(margin, image_margin), status, positive
        coordinates = placed[0][0]
        if kind is CollectiveVariableKind.RADIUS_OF_GYRATION:
            mass = masses[self.groups[0].indices].astype(coordinates.dtype)
            center = jnp.sum(mass[:, None] * coordinates, axis=0) / jnp.sum(mass)
            squared = jnp.sum(
                mass * jnp.sum((coordinates - center) ** 2, axis=-1)
            ) / jnp.sum(mass)
            positive = squared > 0.0
            value = jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)
            status = status | status_flag(~positive, CollectiveVariableStatus.ZERO_LENGTH)
            return value, margin, status, positive
        if kind in (
            CollectiveVariableKind.ALIGNED_RMSD,
            CollectiveVariableKind.ALIGNED_MSD,
        ):
            reference = self.plan.reference.astype(coordinates.dtype)
            alignment = aligned_deviation(coordinates, reference)
            status = status | alignment.status
            if kind is CollectiveVariableKind.ALIGNED_RMSD:
                return alignment.rmsd, margin, status, alignment.rmsd_regular
            gap_only = status & ~int(CollectiveVariableStatus.ZERO_DISTANCE)
            return alignment.msd, margin, gap_only, alignment.msd_regular
        if kind is CollectiveVariableKind.CONFIGURATION:
            return coordinates.reshape((-1,)), margin, status, regular
        frames = self.plan.reference.reshape((-1,) + coordinates.shape).astype(
            coordinates.dtype
        )
        if self.plan.aligned:
            deviations = jax.vmap(lambda frame: aligned_deviation(coordinates, frame))(
                frames
            )
            msd = deviations.msd
            status = status | (
                reduce_status(deviations.status)
                & int(CollectiveVariableStatus.ALIGNMENT_GAP)
            )
            regular = jnp.all(deviations.msd_regular)
        else:
            msd = jnp.mean(jnp.sum((frames - coordinates[None]) ** 2, axis=-1), axis=-1)
        scale = self.plan.parameters[0]
        logits = -scale * msd
        if self.plan.kind is CollectiveVariableKind.PATH_PROGRESS:
            weights = jax.nn.softmax(logits)
            value = jnp.sum(jnp.arange(msd.size, dtype=msd.dtype) * weights)
        else:
            value = -jax.nn.logsumexp(logits) / scale
        return value, margin, status, regular

    def _group_coordination(
        self, geometry: CollectiveVariableGeometry, /
    ) -> tuple[Array, Array, Array]:
        requirement = self.requirement
        if requirement is None:
            raise RuntimeError("Group coordination lost its neighborhood requirement.")
        view = geometry.neighborhood(requirement.requirement_id)
        vectors = geometry.vectors
        displacement = view.displacement(
            geometry.positions,
            jnp.zeros((0, 3), dtype=geometry.positions.dtype)
            if vectors is None
            else vectors,
        )
        distance, positive = safe_norm(displacement)
        r0, power, cutoff = (self.plan.parameters[k] for k in range(3))
        split = self.plan.group_split
        first = (
            jnp.zeros((self.system.capacity,), dtype=jnp.bool_)
            .at[self.plan.indices[:split]]
            .set(True)
        )
        second = (
            jnp.zeros((self.system.capacity,), dtype=jnp.bool_)
            .at[self.plan.indices[split:]]
            .set(True)
        )
        relation = view.relation.relation
        member = first[relation.source_indices] & second[relation.target_indices]
        inside = view.relation.relation.valid & member & positive & (distance < cutoff)
        taper = (1.0 - (distance / cutoff) ** 2) ** 2
        switching = taper / (1.0 + (distance / r0) ** power)
        value = jnp.sum(jnp.where(inside, switching, 0.0))
        status = status_flag(
            ~jnp.all(view.evidence.successful), CollectiveVariableStatus.NEIGHBORHOOD
        )
        return value, jnp.asarray(jnp.inf, dtype=value.dtype), status


class CollectiveVariableProgram(AbstractCollectiveVariableProgram, NonTrainableState):
    """Named stack of prepared CVs of one system.

    Requirements of equal identity (equal role and cutoff) are deduplicated
    and ordered by ``requirement_id``.
    """

    variables: tuple[PreparedCollectiveVariable, ...]
    output_size: int = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    metrics: tuple[CollectiveVariableMetric, ...]
    program_identity: IdentityWords
    system_identity: IdentityWords
    requirements: tuple[AtomisticNeighborhoodRequirement, ...]
    cell_dependent: bool = eqx.field(static=True)

    def __init__(
        self,
        variables: Iterable[PreparedCollectiveVariable],
        /,
        *,
        names: Iterable[str] | None = None,
    ) -> None:
        values = tuple(variables)
        if not values or any(
            not isinstance(value, PreparedCollectiveVariable) for value in values
        ):
            raise TypeError("variables must contain prepared collective variables.")
        systems = {value.system.prepared_id for value in values}
        if len(systems) != 1:
            raise ValueError("A CV program binds the CVs of one prepared system.")
        base = (
            tuple(f"cv:{index}" for index in range(len(values)))
            if names is None
            else tuple(str(name) for name in names)
        )
        if (
            len(base) != len(values)
            or any(not name for name in base)
            or len(set(base)) != len(base)
        ):
            raise ValueError("CV names must be non-empty, unique, and match variables.")
        expanded: list[str] = []
        metrics: list[CollectiveVariableMetric] = []
        for name, value in zip(base, values, strict=True):
            size = value.plan.size
            expanded.extend(
                (name,) if size == 1 else (f"{name}[{k}]" for k in range(size))
            )
            metrics.extend((value.plan.metric,) * size)
        self.variables = values
        self.output_size = len(expanded)
        self.names = tuple(expanded)
        self.metrics = tuple(metrics)
        self.system_identity = values[0].system.prepared_identity
        requirements = {
            value.requirement.requirement_id: value.requirement
            for value in values
            if value.requirement is not None
        }
        self.requirements = tuple(requirements[key] for key in sorted(requirements))
        self.cell_dependent = any(value.plan.kind in _CELL_KINDS for value in values)
        self.program_identity = IdentityWords.of(
            {
                "kind": "cv-program",
                "variables": [value.prepared_id for value in values],
                "names": list(base),
            }
        )

    @property
    def support(self) -> tuple[int, ...]:
        """Host: sorted physical particle slots of every CV of the program."""
        return tuple(sorted({slot for value in self.variables for slot in value.support}))

    def evaluate(
        self, geometry: CollectiveVariableGeometry, /
    ) -> CollectiveVariableProgramEvaluation:
        evaluations = tuple(value.evaluate(geometry) for value in self.variables)
        return CollectiveVariableProgramEvaluation(
            values=jnp.concatenate(
                tuple(jnp.reshape(value.value, (-1,)) for value in evaluations)
            ),
            successful=jnp.all(
                jnp.stack(tuple(value.successful for value in evaluations))
            ),
            regular=jnp.all(jnp.stack(tuple(value.regular for value in evaluations))),
            status=jnp.stack(tuple(value.status for value in evaluations)),
            branch_margins=jnp.stack(tuple(value.branch_margin for value in evaluations)),
            program_identity=self.program_identity,
        )


__all__ = [
    "AbstractCollectiveVariablePlan",
    "AbstractCollectiveVariableProgram",
    "CollectiveVariableEvaluation",
    "CollectiveVariableKind",
    "CollectiveVariableMetric",
    "CollectiveVariablePlan",
    "CollectiveVariableProgram",
    "CollectiveVariableProgramEvaluation",
    "PreparedCollectiveVariable",
]
