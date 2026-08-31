#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LearnedClosureDiagnostics(StrictModule):
    face_correction_norm: Array
    edge_correction_norm: Array
    consistency_defect: Array
    out_of_distribution_score: Array
    fallback_activated: Array


class StructurePreservingFaceClosurePlan(StrictModule, NonTrainableState):
    """Antisymmetric dissipative correction with explicit OOD fallback."""

    dissipation: Callable = eqx.field(static=True)
    out_of_distribution_score: Callable = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        dissipation: Callable,
        out_of_distribution_score: Callable,
        /,
        *,
        closure_id: str,
        threshold: float = 1.0,
    ):
        if not callable(dissipation) or not callable(out_of_distribution_score):
            raise TypeError("Closure dissipation and OOD score must be callable.")
        identifier = str(closure_id)
        threshold_ = float(threshold)
        if not identifier or threshold_ < 0.0:
            raise ValueError("Structure-preserving closure metadata is invalid.")
        self.dissipation = dissipation
        self.out_of_distribution_score = out_of_distribution_score
        self.threshold = threshold_
        self.closure_id = canonical_fingerprint(
            {
                "kind": "structure-preserving-face-closure",
                "declared_id": identifier,
                "threshold": threshold_,
            }
        )

    def correction(
        self,
        left: Array,
        right: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        jump = jnp.asarray(right) - jnp.asarray(left)
        coefficient = jnp.asarray(self.dissipation(left, right, args))
        coefficient = jnp.maximum(coefficient, 0.0)
        while coefficient.ndim < jump.ndim:
            coefficient = coefficient[..., None]
        learned = -coefficient * jump
        score = jnp.asarray(self.out_of_distribution_score(left, right, args))
        fallback = score > self.threshold
        correction = jnp.where(fallback[..., None], jnp.zeros_like(learned), learned)
        return correction, score, fallback


class ConstrainedMHDClosurePlan(StrictModule, NonTrainableState):
    """Coupled material face-flux and edge-EMF learned correction."""

    face_closure: StructurePreservingFaceClosurePlan
    edge_correction: Callable = eqx.field(static=True)
    consistency_tolerance: float = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        face_closure: StructurePreservingFaceClosurePlan,
        edge_correction: Callable,
        /,
        *,
        closure_id: str,
        consistency_tolerance: float = 1e-10,
    ):
        if not isinstance(
            face_closure, StructurePreservingFaceClosurePlan
        ) or not callable(edge_correction):
            raise TypeError("Constrained-MHD closure components are invalid.")
        tolerance = float(consistency_tolerance)
        identifier = str(closure_id)
        if not identifier or tolerance < 0.0:
            raise ValueError("Constrained-MHD closure metadata is invalid.")
        self.face_closure = face_closure
        self.edge_correction = edge_correction
        self.consistency_tolerance = tolerance
        self.closure_id = canonical_fingerprint(
            {
                "kind": "constrained-mhd-closure",
                "face_closure": face_closure.closure_id,
                "declared_id": identifier,
                "consistency_tolerance": tolerance,
            }
        )

    def apply(
        self,
        left: Array,
        right: Array,
        baseline_face_flux: Array,
        baseline_edge_electromotive: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, LearnedClosureDiagnostics]:
        face_correction, score, fallback = self.face_closure.correction(left, right, args)
        edge_correction = jnp.asarray(
            self.edge_correction(
                left,
                right,
                baseline_face_flux,
                baseline_edge_electromotive,
                args,
            )
        )
        if edge_correction.shape != baseline_edge_electromotive.shape:
            raise ValueError("Learned edge correction shape is invalid.")
        equal = jnp.max(jnp.abs(left - right), axis=-1) <= self.consistency_tolerance
        face_defect = jnp.max(
            jnp.where(equal[..., None], jnp.abs(face_correction), 0.0), initial=0.0
        )
        edge_defect = jnp.max(jnp.abs(edge_correction), initial=0.0) * jnp.all(equal)
        defect = jnp.maximum(face_defect, edge_defect)
        edge_correction = eqx.error_if(
            edge_correction,
            defect > self.consistency_tolerance,
            "MHD closure violates equal-state consistency.",
        )
        edge_correction = jnp.where(
            jnp.any(fallback), jnp.zeros_like(edge_correction), edge_correction
        )
        face = baseline_face_flux + face_correction
        edge = baseline_edge_electromotive + edge_correction
        diagnostics = LearnedClosureDiagnostics(
            face_correction_norm=jnp.sqrt(jnp.sum(face_correction**2)),
            edge_correction_norm=jnp.sqrt(jnp.sum(edge_correction**2)),
            consistency_defect=defect,
            out_of_distribution_score=jnp.max(score),
            fallback_activated=jnp.any(fallback),
        )
        return face, edge, diagnostics


class MultiresolutionMHDClosurePlan(StrictModule, NonTrainableState):
    closures: tuple[ConstrainedMHDClosurePlan, ...]
    physical_scales: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        closures: tuple[ConstrainedMHDClosurePlan, ...],
        physical_scales: tuple[float, ...],
        /,
    ):
        closures_ = tuple(closures)
        scales = tuple(float(value) for value in physical_scales)
        if (
            not closures_
            or len(closures_) != len(scales)
            or any(value <= 0.0 for value in scales)
            or tuple(sorted(scales)) != scales
        ):
            raise ValueError("Multiresolution MHD closure levels are invalid.")
        self.closures = closures_
        self.physical_scales = scales
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiresolution-mhd-closure",
                "closures": [closure.closure_id for closure in closures_],
                "physical_scales": list(scales),
            }
        )

    def closure(self, physical_scale: float, /) -> ConstrainedMHDClosurePlan:
        scale = float(physical_scale)
        index = min(
            range(len(self.physical_scales)),
            key=lambda position: abs(self.physical_scales[position] - scale),
        )
        return self.closures[index]


__all__ = [
    "ConstrainedMHDClosurePlan",
    "LearnedClosureDiagnostics",
    "StructurePreservingFaceClosurePlan",
    "MultiresolutionMHDClosurePlan",
]
