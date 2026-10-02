#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked, parse
from .._components import DomainComponent
from .._domain import Domain
from .._evaluation import PointwiseEvaluator
from .._function import DomainFunction
from ._periodic import PeriodicIdentification


CoordinateMap = tuple[tuple[str, DomainFunction], ...]
PairingTopology: TypeAlias = Literal[
    "shared-interface",
    "overlap-volume",
    "directed-transmission",
    "periodic-interface",
]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _coordinate_map(
    values: Mapping[str, DomainFunction],
    *,
    labels: tuple[str, ...],
    domain: Domain,
    name: str,
) -> CoordinateMap:
    mapping = dict(values)
    if set(mapping) != set(labels):
        raise ValueError(f"{name} must define exactly coordinates {labels!r}.")
    records: list[tuple[str, DomainFunction]] = []
    for label in labels:
        field = mapping[label]
        if not isinstance(field, DomainFunction):
            raise TypeError(f"{name}[{label!r}] must be a DomainFunction.")
        if not field.domain.same_support(domain):
            raise ValueError(f"{name}[{label!r}] must live on domain {domain.labels!r}.")
        records.append((label, field))
    return tuple(records)


def _coordinate_dict(values: CoordinateMap, /) -> dict[str, DomainFunction]:
    return dict(values)


def _native_field_metadata(field: DomainFunction, /) -> dict[str, Any] | None:
    from ._cartesian import (
        _AffineCoordinate,
        _BoxSupport,
        _BoxWindow,
        _FaceEmbedding,
        _PeriodicCoordinate,
    )
    from ._periodic import _FaceProjection, _IdentityCoordinate

    function = (
        field.func.function if isinstance(field.func, PointwiseEvaluator) else field.func
    )
    if not isinstance(
        function,
        (
            _AffineCoordinate,
            _BoxSupport,
            _BoxWindow,
            _FaceEmbedding,
            _PeriodicCoordinate,
            _FaceProjection,
            _IdentityCoordinate,
        ),
    ):
        return None
    return {
        "type": type(function).__qualname__,
        "parameters": asdict(function),
    }


def _native_coordinate_metadata(
    coordinates: CoordinateMap, /
) -> list[tuple[str, dict[str, Any] | None]]:
    return [(label, _native_field_metadata(field)) for label, field in coordinates]


class SubdomainPatch(StrictModule, NonTrainableState):
    """One fixed local domain embedded in an ambient domain."""

    domain: Domain
    interior: DomainComponent
    support: DomainFunction
    window: DomainFunction | None
    to_local: CoordinateMap
    to_ambient: CoordinateMap
    patch_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: Domain,
        interior: DomainComponent,
        support: DomainFunction,
        to_local: Mapping[str, DomainFunction],
        to_ambient: Mapping[str, DomainFunction],
        /,
        *,
        patch_id: str,
        window: DomainFunction | None = None,
    ) -> None:
        if not interior.domain.same_support(domain):
            raise ValueError("interior must live on the patch domain.")
        if window is not None:
            if not isinstance(window, DomainFunction):
                raise TypeError("window must be a DomainFunction or None.")
            if not window.domain.same_support(support.domain):
                raise ValueError("window must live on the ambient domain.")

        self.patch_id = _identifier(patch_id, "patch_id")
        self.domain = domain
        self.interior = interior
        self.support = support
        self.window = window
        self.to_local = _coordinate_map(
            to_local,
            labels=domain.labels,
            domain=support.domain,
            name="to_local",
        )
        self.to_ambient = _coordinate_map(
            to_ambient,
            labels=support.domain.labels,
            domain=domain,
            name="to_ambient",
        )

    @property
    def ambient_domain(self) -> Domain:
        return self.support.domain

    @checked
    def lift(self, field: DomainFunction, /) -> DomainFunction:
        """Pull a local field onto the ambient domain."""
        if not field.domain.same_support(self.domain):
            raise ValueError("field must live on the patch domain.")
        from ...operators import pullback

        coordinates = _coordinate_dict(self.to_local)
        substitutions = {label: coordinates[label] for label in field.deps}
        return pullback(field, substitutions, domain=self.ambient_domain)

    @checked
    def restrict(self, field: DomainFunction, /) -> DomainFunction:
        """Pull an ambient field onto the local domain."""
        if not field.domain.same_support(self.ambient_domain):
            raise ValueError("field must live on the ambient domain.")
        from ...operators import pullback

        coordinates = _coordinate_dict(self.to_ambient)
        substitutions = {label: coordinates[label] for label in field.deps}
        return pullback(field, substitutions, domain=self.domain)


class PairedSupport(StrictModule, NonTrainableState):
    """One physical support carrying paired traces from two local domains.

    A ``"periodic-interface"`` pairing carries the `PeriodicIdentification` that
    glues its two faces. Its right side is the identification's lower (source)
    face and its left side the upper (target) face. Only an identified periodic
    seam may join a patch to itself.
    """

    component: DomainComponent
    left_coordinates: CoordinateMap
    right_coordinates: CoordinateMap
    normal: DomainFunction | None
    pairing_id: str = eqx.field(static=True)
    left_patch_id: str = eqx.field(static=True)
    right_patch_id: str = eqx.field(static=True)
    codimension: int = eqx.field(static=True)
    topology: PairingTopology = eqx.field(static=True)
    identification: PeriodicIdentification | None

    @checked
    def __init__(
        self,
        component: DomainComponent,
        left_coordinates: Mapping[str, DomainFunction],
        right_coordinates: Mapping[str, DomainFunction],
        /,
        *,
        pairing_id: str,
        left_patch_id: str,
        right_patch_id: str,
        normal: DomainFunction | None = None,
        codimension: int = 1,
        topology: PairingTopology = "shared-interface",
        identification: PeriodicIdentification | None = None,
    ) -> None:
        left_id = _identifier(left_patch_id, "left_patch_id")
        right_id = _identifier(right_patch_id, "right_patch_id")
        topology = parse(topology, PairingTopology, "topology")
        if topology == "periodic-interface":
            if not isinstance(identification, PeriodicIdentification):
                raise TypeError(
                    "A periodic-interface pairing requires a PeriodicIdentification."
                )
        elif identification is not None:
            raise ValueError(
                "Only periodic-interface pairings carry a periodic identification."
            )
        if left_id == right_id and topology != "periodic-interface":
            raise ValueError("A paired support must join two distinct patches.")
        if isinstance(codimension, bool) or not isinstance(codimension, int):
            raise TypeError("codimension must be an integer.")
        codimension_ = codimension
        if codimension_ < 0:
            raise ValueError("codimension must be non-negative.")

        if topology == "overlap-volume" and codimension_ != 0:
            raise ValueError("Overlap-volume pairings require codimension=0.")
        if topology != "overlap-volume" and codimension_ == 0:
            raise ValueError("Codimension-zero pairings require overlap-volume topology.")
        if normal is not None:
            if not isinstance(normal, DomainFunction):
                raise TypeError("normal must be a DomainFunction or None.")
            if not normal.domain.same_support(component.domain):
                raise ValueError("normal must live on the paired-support domain.")

        self.component = component
        self.left_coordinates = _coordinate_map(
            left_coordinates,
            labels=tuple(sorted(left_coordinates)),
            domain=component.domain,
            name="left_coordinates",
        )
        self.right_coordinates = _coordinate_map(
            right_coordinates,
            labels=tuple(sorted(right_coordinates)),
            domain=component.domain,
            name="right_coordinates",
        )
        self.normal = normal
        self.pairing_id = _identifier(pairing_id, "pairing_id")
        self.left_patch_id = left_id
        self.right_patch_id = right_id
        self.codimension = codimension_
        self.topology = topology
        self.identification = identification

    @property
    def source_side(self) -> Literal["right"]:
        """Side carrying the periodic source (lower) face."""
        self._require_periodic()
        return "right"

    @property
    def target_side(self) -> Literal["left"]:
        """Side carrying the periodic target (upper) face."""
        self._require_periodic()
        return "left"

    @property
    def self_seam(self) -> bool:
        """Whether this periodic seam joins one patch to itself."""
        return self.left_patch_id == self.right_patch_id

    def _require_periodic(self) -> None:
        if self.identification is None:
            raise ValueError(
                f"Paired support {self.pairing_id!r} is not a periodic identification."
            )

    def coordinate_labels(self, side: Literal["left", "right"], /) -> tuple[str, ...]:
        """Labels pulled back by one side's coordinate map."""
        if side == "left":
            return tuple(label for label, _ in self.left_coordinates)
        if side == "right":
            return tuple(label for label, _ in self.right_coordinates)
        raise ValueError("side must be 'left' or 'right'.")

    def bind(self, left: SubdomainPatch, right: SubdomainPatch, /) -> None:
        """Validate this pairing against its endpoint patches."""
        if not isinstance(left, SubdomainPatch) or not isinstance(right, SubdomainPatch):
            raise TypeError("Paired-support endpoints must be SubdomainPatch objects.")
        if not left.ambient_domain.same_support(right.ambient_domain):
            raise ValueError("Paired-support endpoints must share an ambient domain.")
        if (
            self.identification is not None
            and not self.identification.domain.same_support(left.ambient_domain)
        ):
            raise ValueError(
                "The periodic identification must live on the ambient domain."
            )
        if left.patch_id != self.left_patch_id or right.patch_id != self.right_patch_id:
            raise ValueError(
                "Paired-support endpoints do not match the supplied patches."
            )
        _coordinate_map(
            dict(self.left_coordinates),
            labels=left.domain.labels,
            domain=self.component.domain,
            name="left_coordinates",
        )
        _coordinate_map(
            dict(self.right_coordinates),
            labels=right.domain.labels,
            domain=self.component.domain,
            name="right_coordinates",
        )

    @checked
    def trace(
        self,
        field: DomainFunction,
        /,
        *,
        side: Literal["left", "right"],
    ) -> DomainFunction:
        """Pull a local field onto this support using one declared side."""
        if side == "left":
            coordinates = _coordinate_dict(self.left_coordinates)
        elif side == "right":
            coordinates = _coordinate_dict(self.right_coordinates)
        else:
            raise ValueError("side must be 'left' or 'right'.")
        missing = tuple(label for label in field.deps if label not in coordinates)
        if missing:
            raise ValueError(f"Pairing has no coordinates for dependencies {missing!r}.")
        from ...operators import pullback

        substitutions = {label: coordinates[label] for label in field.deps}
        return pullback(field, substitutions, domain=self.component.domain)

    def audit(
        self,
        points: Any,
        left: SubdomainPatch,
        right: SubdomainPatch,
        /,
        *,
        tolerance: float = 1.0e-8,
    ) -> PairedSupportEvidence:
        """Audit physical map agreement and normal magnitude on fixed points.

        For a periodic seam the two ambient images must differ by the
        identification shift, modulo the period along the identified direction
        only; transverse coordinates must agree exactly.
        """
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        self.bind(left, right)
        mismatch = jnp.asarray(0.0)
        left_ambient = _coordinate_dict(left.to_ambient)
        right_ambient = _coordinate_dict(right.to_ambient)
        identification = self.identification
        if identification is not None:
            from ._cartesian import _PeriodicCoordinate

            endpoint_records: tuple[
                tuple[
                    CoordinateMap,
                    dict[str, DomainFunction],
                    Literal["left", "right"],
                    float,
                ],
                ...,
            ] = (
                (self.left_coordinates, left_ambient, "left", identification.upper),
                (self.right_coordinates, right_ambient, "right", identification.lower),
            )
            for coordinates, ambient, side, expected in endpoint_records:
                ambient_coordinate = ambient[identification.label]
                function = (
                    ambient_coordinate.func.function
                    if isinstance(ambient_coordinate.func, PointwiseEvaluator)
                    else ambient_coordinate.func
                )
                if (
                    isinstance(function, _PeriodicCoordinate)
                    and function.direction == "to-ambient"
                ):
                    # The native cover map canonicalizes upper to lower. Its input
                    # retains endpoint orientation before that quotient operation.
                    if len(ambient_coordinate.deps) != 1:
                        raise ValueError(
                            "A native periodic coordinate map requires one dependency."
                        )
                    endpoint = dict(coordinates)[ambient_coordinate.deps[0]](points).data
                else:
                    endpoint = self.trace(ambient_coordinate, side=side)(points).data
                if identification.vector_coordinate:
                    component = identification.component
                    if component is None:
                        raise RuntimeError(
                            "A vector identification requires a component."
                        )
                    endpoint = endpoint[..., component]
                mismatch = jnp.maximum(
                    mismatch,
                    jnp.max(jnp.abs(endpoint - expected)),
                )
        for label in left.ambient_domain.labels:
            left_values = self.trace(left_ambient[label], side="left")(points).data
            right_values = self.trace(right_ambient[label], side="right")(points).data
            defect = left_values - right_values
            if identification is not None and label == identification.label:
                period = identification.period
                shift = identification.shift()
                defect = defect - shift
                # Only the identified direction wraps: a transverse offset equal
                # to the period is a twisted seam, not this identification.
                defect = defect - shift * jnp.round(defect / period)
            mismatch = jnp.maximum(mismatch, jnp.max(jnp.abs(defect)))
        normal_error = jnp.asarray(0.0)
        if self.normal is not None:
            normal_field = self.normal(points)
            normal_values = jnp.asarray(normal_field.data)
            if normal_field.dims and normal_field.dims[-1] is None:
                magnitude = jnp.linalg.norm(normal_values, axis=-1)
            else:
                magnitude = jnp.abs(normal_values)
            normal_error = jnp.max(jnp.abs(magnitude - 1.0))
        mismatch_value = float(mismatch)
        normal_value = float(normal_error)
        return PairedSupportEvidence(
            pairing_id=self.pairing_id,
            scope="sampled",
            maximum_map_mismatch=mismatch_value,
            maximum_normal_error=normal_value,
            verified=(
                mismatch_value <= float(tolerance) and normal_value <= float(tolerance)
            ),
        )


class PairedSupportEvidence(StrictModule, NonTrainableState):
    """Sampled or exact geometric evidence for one paired support."""

    pairing_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    maximum_map_mismatch: float = eqx.field(static=True)
    maximum_normal_error: float = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        pairing_id: str,
        scope: str,
        maximum_map_mismatch: float,
        maximum_normal_error: float,
        verified: bool,
    ) -> None:
        self.pairing_id = _identifier(pairing_id, "pairing_id")
        self.scope = _identifier(scope, "scope")
        self.maximum_map_mismatch = float(maximum_map_mismatch)
        self.maximum_normal_error = float(maximum_normal_error)
        self.verified = bool(verified)


class SubdomainCoverEvidence(StrictModule, NonTrainableState):
    """Coverage evidence for one immutable subdomain cover."""

    cover_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    num_points: int = eqx.field(static=True)
    min_coverage: int = eqx.field(static=True)
    max_coverage: int = eqx.field(static=True)
    uncovered_points: int = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        cover_id: str,
        scope: str,
        num_points: int,
        min_coverage: int,
        max_coverage: int,
        uncovered_points: int,
        verified: bool,
    ) -> None:
        self.cover_id = _identifier(cover_id, "cover_id")
        self.scope = _identifier(scope, "scope")
        self.num_points = int(num_points)
        self.min_coverage = int(min_coverage)
        self.max_coverage = int(max_coverage)
        self.uncovered_points = int(uncovered_points)
        self.verified = bool(verified)


class SubdomainCover(StrictModule, NonTrainableState):
    """Fixed subdomain topology over one ambient domain."""

    ambient: Domain
    patches: tuple[SubdomainPatch, ...]
    pairings: tuple[PairedSupport, ...]
    cover_id: str = eqx.field(static=True)
    exact_coverage: bool = eqx.field(static=True)
    maximum_overlap: int | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        ambient: Domain,
        patches: Sequence[SubdomainPatch],
        pairings: Sequence[PairedSupport] = (),
        /,
        *,
        cover_id: str,
        exact_coverage: bool = False,
        maximum_overlap: int | None = None,
    ) -> None:
        patches_ = tuple(patches)
        if not patches_:
            raise ValueError("A subdomain cover requires at least one patch.")
        if any(not isinstance(patch, SubdomainPatch) for patch in patches_):
            raise TypeError("patches must contain SubdomainPatch objects.")
        patches_ = tuple(sorted(patches_, key=lambda patch: patch.patch_id))
        patch_ids = tuple(patch.patch_id for patch in patches_)
        if len(set(patch_ids)) != len(patch_ids):
            raise ValueError("Subdomain patch IDs must be unique.")
        for patch in patches_:
            if not patch.ambient_domain.same_support(ambient):
                raise ValueError("Every patch support must live on the ambient domain.")

        by_id = {patch.patch_id: patch for patch in patches_}
        pairings_ = tuple(pairings)
        pairing_ids: set[str] = set()
        for pairing in pairings_:
            if not isinstance(pairing, PairedSupport):
                raise TypeError("pairings must contain PairedSupport objects.")
            if pairing.pairing_id in pairing_ids:
                raise ValueError("Paired-support IDs must be unique.")
            pairing_ids.add(pairing.pairing_id)
            if pairing.left_patch_id not in by_id or pairing.right_patch_id not in by_id:
                raise ValueError("Paired-support endpoint references an unknown patch.")
            pairing.bind(
                by_id[pairing.left_patch_id],
                by_id[pairing.right_patch_id],
            )

        pairings_ = tuple(sorted(pairings_, key=lambda pairing: pairing.pairing_id))
        if not isinstance(exact_coverage, bool):
            raise TypeError("exact_coverage must be a boolean.")
        if maximum_overlap is None:
            maximum_overlap_ = None
        else:
            if isinstance(maximum_overlap, bool) or not isinstance(maximum_overlap, int):
                raise TypeError("maximum_overlap must be an integer or None.")
            maximum_overlap_ = maximum_overlap
            if maximum_overlap_ <= 0:
                raise ValueError("maximum_overlap must be positive when supplied.")

        self.ambient = ambient
        self.patches = patches_
        self.pairings = pairings_
        self.cover_id = _identifier(cover_id, "cover_id")
        self.exact_coverage = bool(exact_coverage)
        self.maximum_overlap = maximum_overlap_

    @property
    def patch_ids(self) -> tuple[str, ...]:
        return tuple(patch.patch_id for patch in self.patches)

    @property
    def pairing_ids(self) -> tuple[str, ...]:
        return tuple(pairing.pairing_id for pairing in self.pairings)

    @property
    def revision(self) -> str:
        """Content revision of this cover for revision-bound interface bindings.

        Declared cover, patch, and pairing IDs, pairing topology, codimension,
        normal presence, coordinate labels, native coordinate/support/window
        parameters (including static fields), periodic identification revisions,
        and every numeric array leaf are content-addressed. Maps authored as opaque
        Python callables are identified only through the declared IDs, so
        replacing such a map requires a new ``cover_id``.
        """
        return canonical_fingerprint(
            {
                "kind": "subdomain-cover-revision",
                "cover_id": self.cover_id,
                "ambient": list(self.ambient.labels),
                "native_maps": {
                    "patches": [
                        [
                            patch.patch_id,
                            _native_field_metadata(patch.support),
                            None
                            if patch.window is None
                            else _native_field_metadata(patch.window),
                            _native_coordinate_metadata(patch.to_local),
                            _native_coordinate_metadata(patch.to_ambient),
                        ]
                        for patch in self.patches
                    ],
                    "pairings": [
                        [
                            pairing.pairing_id,
                            _native_coordinate_metadata(pairing.left_coordinates),
                            _native_coordinate_metadata(pairing.right_coordinates),
                        ]
                        for pairing in self.pairings
                    ],
                },
                "patches": [
                    [patch.patch_id, list(patch.domain.labels)] for patch in self.patches
                ],
                "pairings": [
                    [
                        pairing.pairing_id,
                        pairing.left_patch_id,
                        pairing.right_patch_id,
                        pairing.topology,
                        pairing.codimension,
                        pairing.normal is not None,
                    ]
                    + (
                        []
                        if pairing.identification is None
                        else [pairing.identification.revision]
                    )
                    for pairing in self.pairings
                ],
                "exact_coverage": self.exact_coverage,
                "maximum_overlap": self.maximum_overlap,
                "arrays": array_tree_fingerprint(
                    (self.ambient, self.patches, self.pairings)
                ),
            }
        )

    @property
    def adjacency(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Distinct neighboring patches; a periodic self-seam is not a neighbor."""
        neighbors = {patch_id: set() for patch_id in self.patch_ids}
        for pairing in self.pairings:
            if pairing.left_patch_id == pairing.right_patch_id:
                continue
            neighbors[pairing.left_patch_id].add(pairing.right_patch_id)
            neighbors[pairing.right_patch_id].add(pairing.left_patch_id)
        return tuple(
            (patch_id, tuple(sorted(neighbors[patch_id]))) for patch_id in self.patch_ids
        )

    def patch(self, patch_id: str, /) -> SubdomainPatch:
        identifier = str(patch_id)
        for patch in self.patches:
            if patch.patch_id == identifier:
                return patch
        raise KeyError(f"Unknown subdomain patch {identifier!r}.")

    def pairing(self, pairing_id: str, /) -> PairedSupport:
        identifier = str(pairing_id)
        for pairing in self.pairings:
            if pairing.pairing_id == identifier:
                return pairing
        raise KeyError(f"Unknown paired support {identifier!r}.")

    def structural_evidence(self) -> SubdomainCoverEvidence:
        if not self.exact_coverage or self.maximum_overlap is None:
            raise ValueError("This cover has no exact structural coverage evidence.")
        return SubdomainCoverEvidence(
            cover_id=self.cover_id,
            scope="exact",
            num_points=0,
            min_coverage=1,
            max_coverage=self.maximum_overlap,
            uncovered_points=0,
            verified=True,
        )

    def audit(self, points: Any, /) -> SubdomainCoverEvidence:
        """Audit pointwise coverage without upgrading it to an exact proof."""
        memberships = []
        for patch in self.patches:
            values = jnp.asarray(patch.support(points).data)
            memberships.append(values > 0)
        coverage = jnp.sum(jnp.stack(memberships, axis=0), axis=0)
        uncovered = int(jnp.sum(coverage == 0))
        minimum = int(jnp.min(coverage))
        maximum = int(jnp.max(coverage))
        within_capacity = self.maximum_overlap is None or maximum <= self.maximum_overlap
        return SubdomainCoverEvidence(
            cover_id=self.cover_id,
            scope="sampled",
            num_points=coverage.size,
            min_coverage=minimum,
            max_coverage=maximum,
            uncovered_points=uncovered,
            verified=uncovered == 0 and within_capacity,
        )


__all__ = [
    "PairedSupport",
    "PairedSupportEvidence",
    "PairingTopology",
    "SubdomainCover",
    "SubdomainCoverEvidence",
    "SubdomainPatch",
]
