#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DecayOwnership(StrEnum):
    GENERATOR = "generator"
    TRANSPORT = "transport"
    NATIVE_BOUNDED = "native-bounded"


class PDFProviderPlan(StrictModule, NonTrainableState):
    set_name: str = eqx.field(static=True)
    member: int = eqx.field(static=True)
    release: str = eqx.field(static=True)
    interpolation_id: str = eqx.field(static=True)
    alpha_s_id: str = eqx.field(static=True)
    flavor_scheme: str = eqx.field(static=True)
    x_support: tuple[float, float] = eqx.field(static=True)
    scale_support: tuple[float, float] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        set_name: str,
        member: int,
        release: str,
        interpolation_id: str,
        alpha_s_id: str,
        flavor_scheme: str,
        x_support: tuple[float, float],
        scale_support: tuple[float, float],
    ):
        labels = tuple(
            str(value).strip()
            for value in (set_name, release, interpolation_id, alpha_s_id, flavor_scheme)
        )
        x_minimum, x_maximum = map(float, x_support)
        q_minimum, q_maximum = map(float, scale_support)
        if any(not value for value in labels) or int(member) < 0:
            raise ValueError("PDF provider identities and member are invalid.")
        if not (0.0 < x_minimum < x_maximum <= 1.0):
            raise ValueError("x_support must be an ordered subset of (0, 1].")
        if (
            not all(math.isfinite(value) for value in (q_minimum, q_maximum))
            or not 0.0 < q_minimum < q_maximum
        ):
            raise ValueError("scale_support must be finite, positive, and ordered.")
        (
            self.set_name,
            self.release,
            self.interpolation_id,
            self.alpha_s_id,
            self.flavor_scheme,
        ) = labels
        self.member = int(member)
        self.x_support = (x_minimum, x_maximum)
        self.scale_support = (q_minimum, q_maximum)
        self.provider_id = canonical_fingerprint(
            {
                "kind": "pdf-provider-plan",
                "set": self.set_name,
                "member": self.member,
                "release": self.release,
                "interpolation": self.interpolation_id,
                "alpha_s": self.alpha_s_id,
                "flavor_scheme": self.flavor_scheme,
                "x_support": list(self.x_support),
                "scale_support": list(self.scale_support),
            }
        )


class MatchingMergingPlan(StrictModule, NonTrainableState):
    method_id: str = eqx.field(static=True)
    multiplicities: tuple[int, ...] = eqx.field(static=True)
    merging_scale: float = eqx.field(static=True)
    shower_provider_id: str = eqx.field(static=True)
    weight_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        method_id: str,
        multiplicities: Sequence[int],
        merging_scale: float,
        shower_provider_id: str,
        weight_names: Sequence[str],
    ):
        method = str(method_id).strip()
        shower = str(shower_provider_id).strip()
        multiplicities_ = tuple(int(value) for value in multiplicities)
        names = tuple(str(value).strip() for value in weight_names)
        scale = float(merging_scale)
        if (
            not method
            or not shower
            or not multiplicities_
            or any(value < 0 for value in multiplicities_)
        ):
            raise ValueError("Matching/merging provider metadata is incomplete.")
        if tuple(sorted(set(multiplicities_))) != multiplicities_:
            raise ValueError("multiplicities must be unique and increasing.")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("merging_scale must be finite and positive.")
        if any(not value for value in names) or len(set(names)) != len(names):
            raise ValueError("weight_names must be distinct non-empty strings.")
        self.method_id = method
        self.multiplicities = multiplicities_
        self.merging_scale = scale
        self.shower_provider_id = shower
        self.weight_names = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "matching-merging-plan",
                "method": method,
                "multiplicities": list(multiplicities_),
                "merging_scale": scale,
                "shower_provider": shower,
                "weights": list(names),
            }
        )


__all__ = ["DecayOwnership", "MatchingMergingPlan", "PDFProviderPlan"]
