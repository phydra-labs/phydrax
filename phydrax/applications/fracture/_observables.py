#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._enrichment import CrackTipMaterial


class StressIntensityFactors(StrictModule, NonTrainableState):
    """Path-resolved interaction-integral evidence for one accepted sharp state."""

    mode_i: Array
    mode_ii: Array
    j_integral: Array
    mode_i_by_contour: Array
    mode_ii_by_contour: Array
    j_by_contour: Array
    contour_radii: Array
    path_independence_defect: Array
    energy_consistency_defect: Array
    qualified: Array
    method: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    state_version: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_i_by_contour: ArrayLike,
        mode_ii_by_contour: ArrayLike,
        j_by_contour: ArrayLike,
        contour_radii: ArrayLike,
        path_independence_defect: ArrayLike,
        energy_consistency_defect: ArrayLike,
        qualified: ArrayLike,
        /,
        *,
        topology_id: str,
        quadrature_id: str,
        state_version: int,
    ):
        mode_i = np.asarray(mode_i_by_contour)
        mode_ii = np.asarray(mode_ii_by_contour)
        j_values = np.asarray(j_by_contour)
        radii = np.asarray(contour_radii)
        path_defect = np.asarray(path_independence_defect)
        energy_defect = np.asarray(energy_consistency_defect)
        qualified_ = np.asarray(qualified, dtype=bool)
        topology_identifier = str(topology_id)
        quadrature_identifier = str(quadrature_id)
        version = int(state_version)
        if (
            mode_i.ndim != 1
            or mode_i.size < 3
            or mode_ii.shape != mode_i.shape
            or j_values.shape != mode_i.shape
            or radii.shape != mode_i.shape
            or np.any(~np.isfinite(mode_i))
            or np.any(~np.isfinite(mode_ii))
            or np.any(~np.isfinite(j_values))
            or np.any(~np.isfinite(radii))
            or np.any(radii <= 0.0)
            or np.any(np.diff(radii) <= 0.0)
            or path_defect.shape != ()
            or energy_defect.shape != ()
            or qualified_.shape != ()
            or not np.isfinite(path_defect)
            or not np.isfinite(energy_defect)
            or path_defect < 0.0
            or energy_defect < 0.0
            or not topology_identifier
            or not quadrature_identifier
            or version < 0
        ):
            raise ValueError("Stress-intensity evidence and provenance are inconsistent.")
        self.mode_i = jnp.asarray(np.mean(mode_i))
        self.mode_ii = jnp.asarray(np.mean(mode_ii))
        self.j_integral = jnp.asarray(np.mean(j_values))
        self.mode_i_by_contour = jnp.asarray(mode_i)
        self.mode_ii_by_contour = jnp.asarray(mode_ii)
        self.j_by_contour = jnp.asarray(j_values)
        self.contour_radii = jnp.asarray(radii)
        self.path_independence_defect = jnp.asarray(path_defect)
        self.energy_consistency_defect = jnp.asarray(energy_defect)
        self.qualified = jnp.asarray(qualified_)
        self.method = "domain-interaction-integral"
        self.topology_id = topology_identifier
        self.quadrature_id = quadrature_identifier
        self.state_version = version
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "interaction-integral-sif-evidence",
                "mode_i": mode_i.tolist(),
                "mode_ii": mode_ii.tolist(),
                "j": j_values.tolist(),
                "radii": radii.tolist(),
                "path_defect": float(path_defect),
                "energy_defect": float(energy_defect),
                "qualified": bool(qualified_),
                "topology": topology_identifier,
                "quadrature": quadrature_identifier,
                "state_version": version,
            }
        )


def _validate_tensor_fields(
    stress: np.ndarray,
    displacement_gradient: np.ndarray,
    auxiliary_mode_i_stress: np.ndarray,
    auxiliary_mode_i_gradient: np.ndarray,
    auxiliary_mode_ii_stress: np.ndarray,
    auxiliary_mode_ii_gradient: np.ndarray,
    q_gradient: np.ndarray,
    weights: np.ndarray,
) -> tuple[int, int]:
    if stress.ndim != 4 or stress.shape[-2:] != (2, 2):
        raise ValueError(
            "Interaction-integral stress requires shape (contour, point, 2, 2)."
        )
    contour_count, point_count = stress.shape[:2]
    tensor_shape = (contour_count, point_count, 2, 2)
    vector_shape = (contour_count, point_count, 2)
    scalar_shape = (contour_count, point_count)
    if (
        displacement_gradient.shape != tensor_shape
        or auxiliary_mode_i_stress.shape != tensor_shape
        or auxiliary_mode_i_gradient.shape != tensor_shape
        or auxiliary_mode_ii_stress.shape != tensor_shape
        or auxiliary_mode_ii_gradient.shape != tensor_shape
        or q_gradient.shape != vector_shape
        or weights.shape != scalar_shape
    ):
        raise ValueError(
            "Interaction-integral physical, auxiliary, and q-field layouts differ."
        )
    arrays = (
        stress,
        displacement_gradient,
        auxiliary_mode_i_stress,
        auxiliary_mode_i_gradient,
        auxiliary_mode_ii_stress,
        auxiliary_mode_ii_gradient,
        q_gradient,
        weights,
    )
    if any(np.any(~np.isfinite(value)) for value in arrays) or np.any(weights <= 0.0):
        raise ValueError(
            "Interaction-integral fields must be finite with positive weights."
        )
    return contour_count, point_count


def _interaction_integral(
    stress: Array,
    displacement_gradient: Array,
    auxiliary_stress: Array,
    auxiliary_gradient: Array,
    q_gradient: Array,
    weights: Array,
    tangent: Array,
) -> Array:
    auxiliary_strain = 0.5 * (
        auxiliary_gradient + jnp.swapaxes(auxiliary_gradient, -1, -2)
    )
    displacement_tangent = ein.contract("...ij,j->...i", displacement_gradient, tangent)
    auxiliary_tangent = ein.contract("...ij,j->...i", auxiliary_gradient, tangent)
    mutual_energy = ein.contract("...ij,...ij->...", stress, auxiliary_strain)
    interaction_flux = (
        ein.contract("...ij,...i->...j", stress, auxiliary_tangent)
        + ein.contract("...ij,...i->...j", auxiliary_stress, displacement_tangent)
        - mutual_energy[..., None] * tangent
    )
    return -ein.contract("cq,cqj,cqj->c", weights, interaction_flux, q_gradient)


def _path_defect(values: Array) -> Array:
    mean = jnp.mean(values)
    scale = jnp.maximum(jnp.max(jnp.abs(values)), jnp.finfo(values.dtype).eps)
    return jnp.max(jnp.abs(values - mean)) / scale


def evaluate_interaction_integral(
    stress: ArrayLike,
    displacement_gradient: ArrayLike,
    auxiliary_mode_i_stress: ArrayLike,
    auxiliary_mode_i_gradient: ArrayLike,
    auxiliary_mode_ii_stress: ArrayLike,
    auxiliary_mode_ii_gradient: ArrayLike,
    q_gradient: ArrayLike,
    weights: ArrayLike,
    contour_radii: ArrayLike,
    tangent: ArrayLike,
    material: CrackTipMaterial,
    /,
    *,
    topology_id: str,
    quadrature_id: str,
    state_version: int,
    qualification_tolerance: float = 5.0e-2,
) -> StressIntensityFactors:
    """Evaluate mixed-mode SIFs from unit-K auxiliary domain interaction integrals."""

    if not isinstance(material, CrackTipMaterial):
        raise TypeError("material must be CrackTipMaterial.")
    arrays = tuple(
        np.asarray(value)
        for value in (
            stress,
            displacement_gradient,
            auxiliary_mode_i_stress,
            auxiliary_mode_i_gradient,
            auxiliary_mode_ii_stress,
            auxiliary_mode_ii_gradient,
            q_gradient,
            weights,
        )
    )
    contour_count, _ = _validate_tensor_fields(*arrays)
    radii = np.asarray(contour_radii)
    tangent_ = np.asarray(tangent)
    tolerance = float(qualification_tolerance)
    if (
        radii.shape != (contour_count,)
        or contour_count < 3
        or np.any(~np.isfinite(radii))
        or np.any(radii <= 0.0)
        or np.any(np.diff(radii) <= 0.0)
        or tangent_.shape != (2,)
        or np.any(~np.isfinite(tangent_))
        or not np.isclose(np.linalg.norm(tangent_), 1.0, rtol=0.0, atol=1.0e-10)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("Interaction contours, crack tangent, or tolerance are invalid.")

    (
        stress_,
        gradient_,
        mode_i_stress,
        mode_i_gradient,
        mode_ii_stress,
        mode_ii_gradient,
        q_gradient_,
        weights_,
    ) = tuple(jnp.asarray(value) for value in arrays)
    tangent_value = jnp.asarray(tangent_)
    interaction_i = _interaction_integral(
        stress_,
        gradient_,
        mode_i_stress,
        mode_i_gradient,
        q_gradient_,
        weights_,
        tangent_value,
    )
    interaction_ii = _interaction_integral(
        stress_,
        gradient_,
        mode_ii_stress,
        mode_ii_gradient,
        q_gradient_,
        weights_,
        tangent_value,
    )
    mode_i = 0.5 * material.effective_modulus * interaction_i
    mode_ii = 0.5 * material.effective_modulus * interaction_ii

    strain = 0.5 * (gradient_ + jnp.swapaxes(gradient_, -1, -2))
    energy_density = 0.5 * ein.contract("...ij,...ij->...", stress_, strain)
    displacement_tangent = ein.contract("...ij,j->...i", gradient_, tangent_value)
    j_flux = (
        ein.contract("...ij,...i->...j", stress_, displacement_tangent)
        - energy_density[..., None] * tangent_value
    )
    j_values = -ein.contract("cq,cqj,cqj->c", weights_, j_flux, q_gradient_)

    mode_i_defect = _path_defect(mode_i)
    mode_ii_defect = _path_defect(mode_ii)
    j_defect = _path_defect(j_values)
    path_defect = jnp.maximum(jnp.maximum(mode_i_defect, mode_ii_defect), j_defect)
    j_from_sif = (mode_i * mode_i + mode_ii * mode_ii) / material.effective_modulus
    energy_scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(j_values)), jnp.max(jnp.abs(j_from_sif))),
        jnp.finfo(j_values.dtype).eps,
    )
    energy_defect = jnp.max(jnp.abs(j_values - j_from_sif)) / energy_scale
    qualified = (path_defect <= tolerance) & (energy_defect <= tolerance)
    return StressIntensityFactors(
        mode_i,
        mode_ii,
        j_values,
        radii,
        path_defect,
        energy_defect,
        qualified,
        topology_id=topology_id,
        quadrature_id=quadrature_id,
        state_version=state_version,
    )


__all__ = ["StressIntensityFactors", "evaluate_interaction_integral"]
