#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._boundary import FiniteVolumeBoundarySet
from ...discretization.finite_volume._dynamics import (
    FiniteVolumeMethodPlan,
    PreparedFiniteVolumeDynamics,
)
from ...discretization.finite_volume._mapped import MappedFiniteVolumeDiscretization
from ...discretization.finite_volume._positivity import FluxPositivityPlan
from ...discretization.finite_volume._precision import FiniteVolumePrecisionPolicy
from ...discretization.finite_volume._riemann import RusanovFluxPlan
from ...discretization.finite_volume._structured import FiniteVolumeDiscretization
from ...solver._finite_volume_runtime import (
    FiniteVolumeStepPolicy,
    PreparedFiniteVolumeRuntime,
)
from ._state import ReactiveConservedLayout, ReactiveEulerSystem


class ReactiveStructuredFiniteVolumePlan(StrictModule, NonTrainableState):
    """Bind reacting Euler physics to the existing structured FV runtime."""

    layout: ReactiveConservedLayout
    system: ReactiveEulerSystem
    method: FiniteVolumeMethodPlan
    boundaries: FiniteVolumeBoundarySet
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: ReactiveConservedLayout,
        method: FiniteVolumeMethodPlan,
        boundaries: FiniteVolumeBoundarySet,
        /,
    ):
        if not isinstance(layout, ReactiveConservedLayout):
            raise TypeError("layout must be ReactiveConservedLayout.")
        if not isinstance(method, FiniteVolumeMethodPlan):
            raise TypeError("method must be FiniteVolumeMethodPlan.")
        if not isinstance(boundaries, FiniteVolumeBoundarySet):
            raise TypeError("boundaries must be FiniteVolumeBoundarySet.")
        system = ReactiveEulerSystem(layout)
        self.layout = layout
        self.system = system
        self.method = method
        self.boundaries = boundaries
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-structured-finite-volume-plan",
                "layout": layout.layout_id,
                "system": system.system_id,
                "method": method.method_id,
                "boundaries": boundaries.boundary_set_id,
            }
        )

    def prepare(
        self,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        /,
        *,
        capacity: ArrayLike | None = None,
        source: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
        source_id: str | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ) -> PreparedFiniteVolumeDynamics:
        if not isinstance(
            discretization,
            (FiniteVolumeDiscretization, MappedFiniteVolumeDiscretization),
        ):
            raise TypeError("discretization must be prepared structured FV geometry.")
        if discretization.component_count != self.layout.component_count:
            raise ValueError(
                "Reactive layout and FV discretization component counts must match."
            )
        if len(discretization.cell_shape) != self.layout.dimension:
            raise ValueError("Reactive layout and FV grid dimensions must match.")
        return PreparedFiniteVolumeDynamics(
            self.system,
            discretization,
            self.method,
            self.boundaries,
            capacity=capacity,
            source=source,
            source_id=source_id,
            precision=precision,
        )

    def prepare_runtime(
        self,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        /,
        *,
        capacity: ArrayLike | None = None,
        source: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
        source_id: str | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
        step_policy: FiniteVolumeStepPolicy | None = None,
        positivity_iterations: int = 32,
    ) -> PreparedFiniteVolumeRuntime:
        """Prepare SSPRK transport with a layout-generic Rusanov fallback."""
        dynamics = self.prepare(
            discretization,
            capacity=capacity,
            source=source,
            source_id=source_id,
            precision=precision,
        )
        positivity = FluxPositivityPlan(
            positivity_iterations,
            fallback_flux=RusanovFluxPlan(),
        )
        return PreparedFiniteVolumeRuntime(
            dynamics,
            positivity,
            step_policy,
        )


__all__ = ["ReactiveStructuredFiniteVolumePlan"]
