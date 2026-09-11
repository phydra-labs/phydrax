#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-free synthesis plans for regular and irregular solid harmonics."""

from __future__ import annotations

from numbers import Integral

import equinox as eqx
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...special._solid_harmonic import (
    _solid_harmonic_synthesis,
    SolidHarmonicKind,
)
from .._core import DiscretizationCapability, PreparationReport
from ._spherical_layout import SphericalModeLayout


class SolidHarmonicPlan(StrictModule, NonTrainableState):
    """Symbolic matrix-free solid-harmonic synthesis plan."""

    layout: SphericalModeLayout
    kind: SolidHarmonicKind = eqx.field(static=True)
    reality: bool = eqx.field(static=True)
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bandlimit: int,
        /,
        *,
        kind: SolidHarmonicKind = "regular",
        reality: bool = True,
    ):
        if isinstance(bandlimit, bool) or not isinstance(bandlimit, Integral):
            raise TypeError("bandlimit must be a static integer.")
        limit = int(bandlimit)
        kind_ = str(kind).lower()
        if kind_ not in ("regular", "irregular"):
            raise ValueError("kind must be 'regular' or 'irregular'.")
        reality_ = bool(reality)
        layout = SphericalModeLayout(limit, spin=0, reality=reality_)
        capabilities = (
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.RECONSTRUCTION,
        )
        self.layout = layout
        self.kind = kind_
        self.reality = reality_
        self.capabilities = capabilities
        self.plan_id = canonical_fingerprint(
            {
                "kind": "solid-harmonic-plan",
                "layout": layout.layout_id,
                "radial_kind": kind_,
                "reality": reality_,
            }
        )

    @property
    def bandlimit(self) -> int:
        return self.layout.bandlimit

    def prepare(self, /) -> "PreparedSolidHarmonicSynthesis":
        """Prepare a table-free synthesis operator."""
        return PreparedSolidHarmonicSynthesis(self)


class PreparedSolidHarmonicSynthesis(StrictModule, NonTrainableState):
    """Prepared fused solid-harmonic recurrence and coefficient contraction."""

    plan: SolidHarmonicPlan
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SolidHarmonicPlan, /):
        if not isinstance(plan, SolidHarmonicPlan):
            raise TypeError("plan must be a SolidHarmonicPlan.")
        layout = plan.layout
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                f"kind:{plan.kind}",
                f"reality:{int(plan.reality)}",
                "execution:fused-recurrence",
            ),
            resource_counts={
                "logical_modes": layout.logical_mode_count,
                "padded_coefficients": layout.coefficient_shape[0]
                * layout.coefficient_shape[1],
                "persistent_array_bytes": 0,
                "point_mode_table_entries": 0,
            },
        )
        self.plan = plan
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-solid-harmonic-synthesis",
                "plan": plan.plan_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def layout(self) -> SphericalModeLayout:
        return self.plan.layout

    @property
    def bandlimit(self) -> int:
        return self.plan.bandlimit

    @property
    def kind(self) -> SolidHarmonicKind:
        return self.plan.kind

    @property
    def reality(self) -> bool:
        return self.plan.reality

    def evaluate(
        self,
        coefficients: ArrayLike,
        displacements: ArrayLike,
        /,
    ) -> Array:
        """Evaluate leading-layout coefficients at Cartesian displacements."""
        return _solid_harmonic_synthesis(
            coefficients,
            displacements,
            bandlimit=self.plan.bandlimit,
            kind=self.plan.kind,
            real_output=self.plan.reality,
        )


__all__ = ["PreparedSolidHarmonicSynthesis", "SolidHarmonicPlan"]
