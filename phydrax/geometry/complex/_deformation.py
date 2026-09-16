#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-support projective complex-structure deformation families."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._moduli import TrainableHomogeneousHypersurface


class ComplexStructureFamilyPlan(StrictModule):
    """Transverse coefficient directions in one normalized hypersurface family."""

    base: TrainableHomogeneousHypersurface
    deformation_directions: Array
    modulus_labels: tuple[str, ...] = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    family_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: TrainableHomogeneousHypersurface,
        modulus_labels: Sequence[str],
        deformation_directions: ArrayLike,
        /,
        *,
        rank_tolerance: float = 1e-10,
    ):
        if not isinstance(base, TrainableHomogeneousHypersurface):
            raise TypeError("base must be TrainableHomogeneousHypersurface.")
        labels = tuple(str(value) for value in modulus_labels)
        directions = np.asarray(deformation_directions, dtype=np.complex128)
        tolerance = float(rank_tolerance)
        if (
            not labels
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("modulus_labels must be unique and non-empty.")
        if directions.shape != (len(labels), base.coefficients.shape[0]):
            raise ValueError(
                "deformation_directions must have shape (moduli, coefficients)."
            )
        if not np.all(np.isfinite(directions)):
            raise ValueError("deformation_directions must be finite.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("rank_tolerance must be positive and finite.")
        if np.max(np.abs(directions[:, base.pivot])) > tolerance:
            raise ValueError(
                "Deformation directions must preserve the coefficient pivot."
            )
        singular_values = np.linalg.svd(directions, compute_uv=False)
        if singular_values.size < len(labels) or singular_values[-1] <= tolerance:
            raise ValueError("Deformation directions are linearly dependent.")
        if base.pgl_slice is not None:
            slice_residual = np.max(np.abs(np.asarray(base.pgl_slice) @ directions.T))
            if slice_residual > tolerance:
                raise ValueError("A deformation direction leaves the declared PGL slice.")
        self.base = base
        self.deformation_directions = jnp.asarray(directions)
        self.modulus_labels = labels
        self.rank_tolerance = tolerance
        self.family_plan_id = canonical_fingerprint(
            {
                "kind": "complex-structure-family-plan",
                "family": base.family_id,
                "base_coefficients": array_tree_fingerprint(base.coefficients),
                "modulus_labels": labels,
                "deformation_directions": array_tree_fingerprint(directions),
                "rank_tolerance": tolerance,
            }
        )

    @property
    def modulus_count(self) -> int:
        return len(self.modulus_labels)

    def deformed(self, parameters: ArrayLike, /) -> TrainableHomogeneousHypersurface:
        values = jnp.asarray(parameters, dtype=self.base.coefficients.dtype)
        if values.shape != (self.modulus_count,):
            raise ValueError("parameters must provide one value per modulus.")
        coefficients = self.base.coefficients + ein.contract(
            "m,mc->c", values, self.deformation_directions
        )
        return self.base.with_coefficients(coefficients)

    def evaluate_deformations(self, points: ArrayLike, /) -> Array:
        """Evaluate every coefficient deformation polynomial at homogeneous points."""
        coordinates = jnp.asarray(points, dtype=self.base.coefficients.dtype)
        if coordinates.shape[-1:] != (self.base.projective_dimension + 1,):
            raise ValueError("Homogeneous points have the wrong trailing dimension.")
        monomials = jnp.prod(coordinates[..., None, :] ** self.base.exponents, axis=-1)
        return ein.contract("...c,mc->...m", monomials, self.deformation_directions)


class ComplexStructureFamilyEvidence(StrictModule):
    direction_gram: Array
    minimum_singular_value: Array
    pivot_residual: Array
    pgl_slice_residual: Array
    finite: Array
    accepted: Array
    family_plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def assess_complex_structure_family(
    plan: ComplexStructureFamilyPlan, /
) -> ComplexStructureFamilyEvidence:
    if not isinstance(plan, ComplexStructureFamilyPlan):
        raise TypeError("plan must be ComplexStructureFamilyPlan.")
    directions = plan.deformation_directions
    gram = jnp.conj(directions) @ jnp.swapaxes(directions, 0, 1)
    singular_values = jnp.linalg.svd(directions, compute_uv=False)
    minimum = jnp.min(singular_values)
    pivot = jnp.max(jnp.abs(directions[:, plan.base.pivot]))
    slice_residual = (
        jnp.asarray(0.0, dtype=minimum.dtype)
        if plan.base.pgl_slice is None
        else jnp.max(jnp.abs(plan.base.pgl_slice @ jnp.swapaxes(directions, 0, 1)))
    )
    finite = (
        jnp.all(jnp.isfinite(gram))
        & jnp.isfinite(minimum)
        & jnp.isfinite(pivot)
        & jnp.isfinite(slice_residual)
    )
    accepted = (
        finite
        & (minimum > plan.rank_tolerance)
        & (pivot <= plan.rank_tolerance)
        & (slice_residual <= plan.rank_tolerance)
    )
    return ComplexStructureFamilyEvidence(
        direction_gram=gram,
        minimum_singular_value=minimum,
        pivot_residual=pivot,
        pgl_slice_residual=slice_residual,
        finite=finite,
        accepted=accepted,
        family_plan_id=plan.family_plan_id,
        claim="algebraic-transverse-deformation-family-not-harmonic-representatives",
    )


__all__ = [
    "ComplexStructureFamilyEvidence",
    "ComplexStructureFamilyPlan",
    "assess_complex_structure_family",
]
