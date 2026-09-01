#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


DeformedMeasureKind = Literal["volume", "surface"]


def _real_inexact_array(name: str, value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact) or jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    return array


class DeformedMeasureState(StrictModule):
    """Dynamic reference/current measure and orientation evidence.

    Measures include the caller's quadrature weights. Surface normals are unit
    vectors; their orientation is inherited from the reference normal without a
    sign choice. ``admissible`` is pointwise, while ``valid`` certifies the whole
    evaluated measure.
    """

    reference_measure: Array
    current_measure: Array
    measure_ratio: Array
    jacobian: Array
    reference_normal: Array | None
    current_normal: Array | None
    admissible: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    kind: DeformedMeasureKind = eqx.field(static=True)

    def measure(self, frame: Literal["reference", "current"], /) -> Array:
        if frame == "reference":
            return self.reference_measure
        if frame == "current":
            return self.current_measure
        raise ValueError("Measure frame must be 'reference' or 'current'.")

    def normal(self, frame: Literal["reference", "current"], /) -> Array:
        if self.kind != "surface":
            raise ValueError("Volume measures do not carry oriented normals.")
        normal = self.reference_normal if frame == "reference" else self.current_normal
        if frame not in ("reference", "current"):
            raise ValueError("Normal frame must be 'reference' or 'current'.")
        assert normal is not None
        return normal


class DeformedMeasurePlan(StrictModule, NonTrainableState):
    """Reference measure prepared for dynamic volume or Nanson evaluation.

    ``reference_measure`` is the complete positive reference quadrature measure.
    For a surface, ``reference_normal`` supplies only orientation: it is
    normalized during preparation, so its magnitude can never silently rescale
    the measure. Evaluation uses the canonical finite-strain/Nanson operators and
    leaves all state dependence inside JAX differentiation.
    """

    reference_measure: Array
    reference_normal: Array | None
    minimum_jacobian: float = eqx.field(static=True)
    kind: DeformedMeasureKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: DeformedMeasureKind,
        reference_measure: ArrayLike,
        /,
        *,
        reference_normal: ArrayLike | None = None,
        minimum_jacobian: float = 0.0,
        plan_id: str | None = None,
    ):
        if kind not in ("volume", "surface"):
            raise ValueError("Deformed measure kind must be 'volume' or 'surface'.")
        measure = _real_inexact_array("reference_measure", reference_measure)
        measure_host = np.asarray(measure)
        if not np.all(np.isfinite(measure_host)) or np.any(measure_host <= 0.0):
            raise ValueError("Reference measures must be finite and strictly positive.")
        minimum = float(minimum_jacobian)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_jacobian must be finite and nonnegative.")

        if kind == "volume":
            if reference_normal is not None:
                raise ValueError("A volume measure cannot carry a reference normal.")
            normal = None
        else:
            if reference_normal is None:
                raise ValueError(
                    "A surface measure requires an oriented reference normal."
                )
            supplied = _real_inexact_array("reference_normal", reference_normal)
            if supplied.shape[-1:] not in ((2,), (3,)):
                raise ValueError("Surface normals must have two or three components.")
            leading = np.broadcast_shapes(measure.shape, supplied.shape[:-1])
            measure = jnp.broadcast_to(measure, leading)
            supplied = jnp.broadcast_to(supplied, leading + supplied.shape[-1:])
            norm = jnp.sqrt(jnp.sum(supplied * supplied, axis=-1))
            norm_host = np.asarray(norm)
            if not np.all(np.isfinite(norm_host)) or np.any(norm_host <= 0.0):
                raise ValueError("Reference surface normals must be finite and nonzero.")
            normal = supplied / norm[..., None]

        generated = canonical_fingerprint(
            {
                "kind": "deformed-measure-plan",
                "measure_kind": kind,
                "reference_measure": array_tree_fingerprint(measure),
                "reference_normal": (
                    None if normal is None else array_tree_fingerprint(normal)
                ),
                "minimum_jacobian": minimum.hex(),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.reference_measure = measure
        self.reference_normal = normal
        self.minimum_jacobian = minimum
        self.kind = kind
        self.plan_id = identifier

    def evaluate(self, deformation_gradient: ArrayLike, /) -> DeformedMeasureState:
        """Evaluate the current measure without freezing deformation dependence."""
        from ..operators.mechanics import finite_strain_kinematics, nanson_transform

        kinematics = finite_strain_kinematics(deformation_gradient)
        deformation = jnp.asarray(deformation_gradient)
        leading = deformation.shape[:-2]
        reference_measure = jnp.broadcast_to(self.reference_measure, leading)
        jacobian = kinematics.jacobian
        orientation_preserved = kinematics.admissible & (jacobian > self.minimum_jacobian)

        if self.kind == "volume":
            ratio = jacobian
            current_measure = reference_measure * ratio
            reference_normal = None
            current_normal = None
        else:
            assert self.reference_normal is not None
            reference_normal = jnp.broadcast_to(
                self.reference_normal, leading + (deformation.shape[-1],)
            )
            current_area = nanson_transform(deformation_gradient, reference_normal)
            ratio = jnp.sqrt(jnp.sum(current_area * current_area, axis=-1))
            safe_ratio = jnp.maximum(ratio, jnp.finfo(ratio.dtype).tiny)
            current_normal = current_area / safe_ratio[..., None]
            current_measure = reference_measure * ratio

        finite = (
            jnp.isfinite(reference_measure)
            & jnp.isfinite(current_measure)
            & jnp.isfinite(ratio)
            & jnp.isfinite(jacobian)
        )
        positive = (reference_measure > 0.0) & (current_measure > 0.0) & (ratio > 0.0)
        admissible = orientation_preserved & finite & positive
        current_measure = jnp.where(admissible, current_measure, jnp.nan)
        ratio = jnp.where(admissible, ratio, jnp.nan)
        if current_normal is not None:
            current_normal = jnp.where(
                admissible[..., None],
                current_normal,
                jnp.full_like(current_normal, jnp.nan),
            )
        return DeformedMeasureState(
            reference_measure=reference_measure,
            current_measure=current_measure,
            measure_ratio=ratio,
            jacobian=jacobian,
            reference_normal=reference_normal,
            current_normal=current_normal,
            admissible=admissible,
            valid=jnp.all(admissible),
            plan_id=self.plan_id,
            kind=self.kind,
        )


__all__ = [
    "DeformedMeasureKind",
    "DeformedMeasurePlan",
    "DeformedMeasureState",
]
