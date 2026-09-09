#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._components import DomainComponent
from .._domain import Domain
from .._function import DomainFunction


CoordinateMap = tuple[tuple[str, DomainFunction], ...]
PairingTopology = Literal[
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


class SubdomainPatch(StrictModule, NonTrainableState):
    """One fixed local domain embedded in an ambient domain."""

    domain: Domain
    interior: DomainComponent
    support: DomainFunction
    window: DomainFunction | None
    to_local: CoordinateMap
    to_ambient: CoordinateMap
    patch_id: str = eqx.field(static=True)

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
    ):
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain.")
        if not isinstance(interior, DomainComponent):
            raise TypeError("interior must be a DomainComponent.")
        if not interior.domain.same_support(domain):
            raise ValueError("interior must live on the patch domain.")
        if not isinstance(support, DomainFunction):
            raise TypeError("support must be a DomainFunction.")
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

    def lift(self, field: DomainFunction, /) -> DomainFunction:
        """Pull a local field onto the ambient domain."""
        if not isinstance(field, DomainFunction):
            raise TypeError("field must be a DomainFunction.")
        if not field.domain.same_support(self.domain):
            raise ValueError("field must live on the patch domain.")
        from ...operators import pullback

        coordinates = _coordinate_dict(self.to_local)
        substitutions = {label: coordinates[label] for label in field.deps}
        return pullback(field, substitutions, domain=self.ambient_domain)

    def restrict(self, field: DomainFunction, /) -> DomainFunction:
        """Pull an ambient field onto the local domain."""
        if not isinstance(field, DomainFunction):
            raise TypeError("field must be a DomainFunction.")
        if not field.domain.same_support(self.ambient_domain):
            raise ValueError("field must live on the ambient domain.")
        from ...operators import pullback

        coordinates = _coordinate_dict(self.to_ambient)
        substitutions = {label: coordinates[label] for label in field.deps}
        return pullback(field, substitutions, domain=self.domain)


class PairedSupport(StrictModule, NonTrainableState):
    """One physical support carrying paired traces from two local domains."""

    component: DomainComponent
    left_coordinates: CoordinateMap
    right_coordinates: CoordinateMap
    normal: DomainFunction | None
    pairing_id: str = eqx.field(static=True)
    left_patch_id: str = eqx.field(static=True)
    right_patch_id: str = eqx.field(static=True)
    codimension: int = eqx.field(static=True)
    topology: PairingTopology = eqx.field(static=True)

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
    ):
        if not isinstance(component, DomainComponent):
            raise TypeError("component must be a DomainComponent.")
        left_id = _identifier(left_patch_id, "left_patch_id")
        right_id = _identifier(right_patch_id, "right_patch_id")
        if left_id == right_id:
            raise ValueError("A paired support must join two distinct patches.")
        codimension_ = int(codimension)
        if codimension_ < 0:
            raise ValueError("codimension must be non-negative.")
        if topology not in (
            "shared-interface",
            "overlap-volume",
            "directed-transmission",
            "periodic-interface",
        ):
            raise ValueError("Unknown paired-support topology.")
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
        self.left_coordinates = tuple(left_coordinates.items())
        self.right_coordinates = tuple(right_coordinates.items())
        self.normal = normal
        self.pairing_id = _identifier(pairing_id, "pairing_id")
        self.left_patch_id = left_id
        self.right_patch_id = right_id
        self.codimension = codimension_
        self.topology = topology

    def bind(self, left: SubdomainPatch, right: SubdomainPatch, /) -> None:
        """Validate this pairing against its endpoint patches."""
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

    def trace(
        self,
        field: DomainFunction,
        /,
        *,
        side: Literal["left", "right"],
    ) -> DomainFunction:
        """Pull a local field onto this support using one declared side."""
        if not isinstance(field, DomainFunction):
            raise TypeError("field must be a DomainFunction.")
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
        """Audit physical map agreement and normal magnitude on fixed points."""
        self.bind(left, right)
        mismatch = jnp.asarray(0.0)
        if self.topology != "periodic-interface":
            left_ambient = _coordinate_dict(left.to_ambient)
            right_ambient = _coordinate_dict(right.to_ambient)
            for label in left.ambient_domain.labels:
                left_values = self.trace(left_ambient[label], side="left")(points).data
                right_values = self.trace(right_ambient[label], side="right")(points).data
                mismatch = jnp.maximum(
                    mismatch,
                    jnp.max(jnp.abs(left_values - right_values)),
                )
        normal_error = jnp.asarray(0.0)
        if self.normal is not None:
            normal_values = jnp.asarray(self.normal(points).data)
            if normal_values.ndim == 1:
                magnitude = jnp.abs(normal_values)
            else:
                magnitude = jnp.linalg.norm(normal_values, axis=-1)
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
    ):
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
    ):
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
    ):
        if not isinstance(ambient, Domain):
            raise TypeError("ambient must be a Domain.")
        patches_ = tuple(patches)
        if not patches_:
            raise ValueError("A subdomain cover requires at least one patch.")
        if any(not isinstance(patch, SubdomainPatch) for patch in patches_):
            raise TypeError("patches must contain SubdomainPatch objects.")
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

        if maximum_overlap is None:
            maximum_overlap_ = None
        else:
            maximum_overlap_ = int(maximum_overlap)
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
    def adjacency(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        neighbors = {patch_id: set() for patch_id in self.patch_ids}
        for pairing in self.pairings:
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
            num_points=int(coverage.size),
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
