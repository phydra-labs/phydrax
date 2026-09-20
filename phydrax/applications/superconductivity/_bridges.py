#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...chemistry.periodic._superconductivity import SuperconductingMeanFieldResult
from ._cable import SuperconductingCablePlan
from ._ginzburg_landau import GaugeCovariantGLPlan, GinzburgLandauState
from ._london import ThinFilmLondonPlan, ThinFilmLondonResult
from ._quasiclassical import QuasiclassicalEquilibriumResult


class SuperconductingFidelityBridgeEvidence(StrictModule):
    source_observable: Array
    target_observable: Array
    absolute_residual: Array
    relative_residual: Array
    scale_separation: Array
    finite: Array
    supported: Array
    successful: Array
    bridge_id: str = eqx.field(static=True)


class _AbstractBridgePolicy(StrictModule, NonTrainableState, abc.ABC):
    relative_tolerance: float = eqx.field(static=True)
    minimum_scale_separation: float = eqx.field(static=True)
    bridge_id: str = eqx.field(static=True)

    @property
    @abc.abstractmethod
    def bridge_kind(self) -> str:
        raise NotImplementedError

    def __init__(
        self,
        kind: str,
        relative_tolerance: float,
        minimum_scale_separation: float,
        payload: dict[str, object],
        /,
    ):
        tolerance = float(relative_tolerance)
        separation = float(minimum_scale_separation)
        if (
            not isfinite(tolerance)
            or tolerance < 0.0
            or not isfinite(separation)
            or separation <= 0.0
        ):
            raise ValueError("Bridge tolerance or scale separation is invalid.")
        self.relative_tolerance = tolerance
        self.minimum_scale_separation = separation
        self.bridge_id = canonical_fingerprint(
            {
                "kind": kind,
                "relative_tolerance": tolerance,
                "minimum_scale_separation": separation,
                **payload,
            }
        )

    def evidence(self, source, target, scale_separation, supported=True):
        source_ = jnp.asarray(source)
        target_ = jnp.asarray(target, dtype=source_.dtype)
        separation = jnp.asarray(scale_separation, dtype=jnp.real(source_).dtype)
        if source_.shape != target_.shape:
            raise ValueError("Bridge observables must have the same shape.")
        absolute = jnp.linalg.norm(source_ - target_)
        scale = jnp.maximum(
            jnp.maximum(jnp.linalg.norm(source_), jnp.linalg.norm(target_)), 1.0e-30
        )
        relative = absolute / scale
        finite = (
            jnp.all(jnp.isfinite(source_))
            & jnp.all(jnp.isfinite(target_))
            & jnp.isfinite(separation)
            & jnp.isfinite(relative)
        )
        support = jnp.asarray(supported, dtype=jnp.bool_) & (
            separation >= self.minimum_scale_separation
        )
        successful = finite & support & (relative <= self.relative_tolerance)
        return SuperconductingFidelityBridgeEvidence(
            source_,
            target_,
            absolute,
            relative,
            separation,
            finite,
            support,
            successful,
            self.bridge_id,
        )


class BdGQuasiclassicalBridgePlan(_AbstractBridgePolicy):
    energy_scale: float = eqx.field(static=True)

    @property
    def bridge_kind(self) -> str:
        return "bdg-to-quasiclassical"

    def __init__(
        self,
        /,
        *,
        energy_scale: float,
        relative_tolerance: float,
        minimum_scale_separation: float,
    ):
        energy = float(energy_scale)
        if not isfinite(energy) or energy <= 0.0:
            raise ValueError("BdG/quasiclassical bridge energy_scale is invalid.")
        super().__init__(
            "bdg-to-quasiclassical-bridge",
            relative_tolerance,
            minimum_scale_separation,
            {"energy_scale": energy},
        )
        self.energy_scale = energy

    def evaluate(
        self,
        bdg: SuperconductingMeanFieldResult,
        quasiclassical: QuasiclassicalEquilibriumResult,
        /,
        *,
        scale_separation: ArrayLike,
    ) -> SuperconductingFidelityBridgeEvidence:
        if not isinstance(bdg, SuperconductingMeanFieldResult) or not isinstance(
            quasiclassical, QuasiclassicalEquilibriumResult
        ):
            raise TypeError("BdG/quasiclassical bridge requires typed results.")
        source = bdg.spectrum.minimum_direct_gap / self.energy_scale
        target = jnp.mean(jnp.abs(quasiclassical.trajectory_gap)) / self.energy_scale
        return self.evidence(
            source,
            target,
            scale_separation,
            bdg.successful & quasiclassical.evidence.successful,
        )


class QuasiclassicalGLBridgePlan(_AbstractBridgePolicy):
    gap_to_order_parameter_scale: float = eqx.field(static=True)

    @property
    def bridge_kind(self) -> str:
        return "quasiclassical-to-gl"

    def __init__(
        self,
        /,
        *,
        gap_to_order_parameter_scale: float,
        relative_tolerance: float,
        minimum_scale_separation: float,
    ):
        scale = float(gap_to_order_parameter_scale)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("Quasiclassical/GL bridge scale is invalid.")
        super().__init__(
            "quasiclassical-to-gl-bridge",
            relative_tolerance,
            minimum_scale_separation,
            {"gap_to_order_parameter_scale": scale},
        )
        self.gap_to_order_parameter_scale = scale

    def evaluate(
        self,
        quasiclassical: QuasiclassicalEquilibriumResult,
        gl_state: GinzburgLandauState,
        /,
        *,
        scale_separation: ArrayLike,
    ) -> SuperconductingFidelityBridgeEvidence:
        source = jnp.mean(jnp.abs(quasiclassical.trajectory_gap))
        target = self.gap_to_order_parameter_scale * jnp.mean(
            jnp.abs(gl_state.gauge.scalar)
        )
        return self.evidence(
            source,
            target,
            scale_separation,
            quasiclassical.evidence.successful,
        )


class GLLondonBridgePlan(_AbstractBridgePolicy):
    current_projection: Array

    @property
    def bridge_kind(self) -> str:
        return "gl-to-london"

    def __init__(
        self,
        current_projection: ArrayLike,
        /,
        *,
        relative_tolerance: float,
        minimum_scale_separation: float,
    ):
        projection = np.asarray(current_projection, dtype=np.float64)
        if projection.ndim != 2 or np.any(~np.isfinite(projection)):
            raise ValueError("GL/London current projection must be a finite matrix.")
        super().__init__(
            "gl-to-london-bridge",
            relative_tolerance,
            minimum_scale_separation,
            {"current_projection": array_tree_fingerprint(projection)},
        )
        self.current_projection = jnp.asarray(projection)

    def evaluate(
        self,
        gl_plan: GaugeCovariantGLPlan,
        gl_state: GinzburgLandauState,
        london: ThinFilmLondonResult,
        /,
        *,
        scale_separation: ArrayLike,
    ) -> SuperconductingFidelityBridgeEvidence:
        gl_current = (
            2.0
            * gl_plan.kinetic_coefficient
            * gl_plan.operators.edge_weights
            * jnp.imag(
                jnp.conj(gl_state.gauge.scalar[gl_plan.gauge.edges[:, 0]])
                * gl_plan.gauge.links(gl_state.gauge)
                * gl_state.gauge.scalar[gl_plan.gauge.edges[:, 1]]
            )
        )
        flattened_london = london.face_sheet_current.reshape((-1,))
        if self.current_projection.shape != (gl_current.size, flattened_london.size):
            raise ValueError("GL/London current projection has incompatible dimensions.")
        projected = self.current_projection @ flattened_london
        return self.evidence(
            gl_current, projected, scale_separation, london.evidence.successful
        )


class LondonCableBridgePlan(_AbstractBridgePolicy):
    @property
    def bridge_kind(self) -> str:
        return "london-to-cable"

    def __init__(
        self,
        /,
        *,
        relative_tolerance: float,
        minimum_scale_separation: float,
    ):
        super().__init__(
            "london-to-cable-bridge",
            relative_tolerance,
            minimum_scale_separation,
            {},
        )

    def evaluate(
        self,
        london: ThinFilmLondonPlan,
        cable: SuperconductingCablePlan,
        /,
        *,
        scale_separation: ArrayLike,
    ) -> SuperconductingFidelityBridgeEvidence:
        inductance, reciprocity = london.inductance_matrix()
        if inductance.shape != (1, 1):
            raise ValueError(
                "London/cable bridge currently requires one physical winding."
            )
        return self.evidence(
            inductance[0, 0],
            jnp.asarray(cable.inductance),
            scale_separation,
            reciprocity <= london.tolerance,
        )


__all__ = [
    "BdGQuasiclassicalBridgePlan",
    "GLLondonBridgePlan",
    "LondonCableBridgePlan",
    "QuasiclassicalGLBridgePlan",
    "SuperconductingFidelityBridgeEvidence",
]
