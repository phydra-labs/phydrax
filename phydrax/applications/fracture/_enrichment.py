#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._geometry import CrackFrontGeometry


class CrackTipMaterial(StrictModule, NonTrainableState):
    """Small-strain isotropic material convention used by tip fields and SIFs."""

    young_modulus: Array
    poisson_ratio: Array
    kappa: Array
    effective_modulus: Array
    kinematics: Literal["plane_strain", "plane_stress"] = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        young_modulus: ArrayLike,
        poisson_ratio: ArrayLike,
        /,
        *,
        kinematics: Literal["plane_strain", "plane_stress"] = "plane_strain",
    ):
        young = np.asarray(young_modulus)
        poisson = np.asarray(poisson_ratio)
        convention = str(kinematics)
        if (
            young.shape != ()
            or poisson.shape != ()
            or not np.isfinite(young)
            or not np.isfinite(poisson)
            or young <= 0.0
            or not -1.0 < poisson < 0.5
            or convention not in ("plane_strain", "plane_stress")
        ):
            raise ValueError("Crack-tip material data or kinematics are inadmissible.")
        if convention == "plane_strain":
            kappa = 3.0 - 4.0 * poisson
            effective = young / (1.0 - poisson * poisson)
        else:
            kappa = (3.0 - poisson) / (1.0 + poisson)
            effective = young
        self.young_modulus = jnp.asarray(young)
        self.poisson_ratio = jnp.asarray(poisson)
        self.kappa = jnp.asarray(kappa)
        self.effective_modulus = jnp.asarray(effective)
        self.kinematics = convention
        self.material_id = canonical_fingerprint(
            {
                "kind": "isotropic-crack-tip-material",
                "young_modulus": float(young),
                "poisson_ratio": float(poisson),
                "kinematics": convention,
            }
        )


class IsotropicWilliamsCrackTipBasis(StrictModule, NonTrainableState):
    """Four leading scalar Williams branch functions for isotropic 2-D cracks."""

    material: CrackTipMaterial
    basis_id: str = eqx.field(static=True)

    def __init__(self, material: CrackTipMaterial, /):
        if not isinstance(material, CrackTipMaterial):
            raise TypeError("material must be CrackTipMaterial.")
        self.material = material
        self.basis_id = canonical_fingerprint(
            {
                "kind": "isotropic-williams-crack-tip-basis",
                "material": material.material_id,
                "functions": "sqrt-r-four-branch",
            }
        )

    def values(self, local_coordinates: ArrayLike, /) -> Array:
        local = jnp.asarray(local_coordinates)
        if local.ndim < 1 or local.shape[-1] != 2:
            raise ValueError("Williams coordinates must end in (radius, angle).")
        radius = local[..., 0]
        angle = local[..., 1]
        radius = eqx.error_if(
            radius,
            jnp.any(~jnp.isfinite(radius) | (radius < 0.0)),
            "Williams radii must be finite and nonnegative.",
        )
        root = jnp.sqrt(radius)
        half_sine = jnp.sin(0.5 * angle)
        half_cosine = jnp.cos(0.5 * angle)
        sine = jnp.sin(angle)
        return root[..., None] * jnp.stack(
            (
                half_sine,
                half_cosine,
                half_sine * sine,
                half_cosine * sine,
            ),
            axis=-1,
        )

    def local_derivatives(self, local_coordinates: ArrayLike, /) -> tuple[Array, Array]:
        local = jnp.asarray(local_coordinates)
        if local.ndim < 1 or local.shape[-1] != 2:
            raise ValueError("Williams coordinates must end in (radius, angle).")
        radius = local[..., 0]
        angle = local[..., 1]
        radius = eqx.error_if(
            radius,
            jnp.any(~jnp.isfinite(radius) | (radius <= 0.0)),
            "Williams derivatives require strictly positive finite radius.",
        )
        values = self.values(local)
        radial = 0.5 * values / radius[..., None]
        root = jnp.sqrt(radius)
        half_sine = jnp.sin(0.5 * angle)
        half_cosine = jnp.cos(0.5 * angle)
        sine = jnp.sin(angle)
        cosine = jnp.cos(angle)
        angular = root[..., None] * jnp.stack(
            (
                0.5 * half_cosine,
                -0.5 * half_sine,
                0.5 * half_cosine * sine + half_sine * cosine,
                -0.5 * half_sine * sine + half_cosine * cosine,
            ),
            axis=-1,
        )
        return radial, angular

    def local_cartesian_gradients(self, local_coordinates: ArrayLike, /) -> Array:
        local = jnp.asarray(local_coordinates)
        radial, angular = self.local_derivatives(local)
        radius = local[..., 0]
        angle = local[..., 1]
        cosine = jnp.cos(angle)[..., None]
        sine = jnp.sin(angle)[..., None]
        first = cosine * radial - sine * angular / radius[..., None]
        second = sine * radial + cosine * angular / radius[..., None]
        return jnp.stack((first, second), axis=-1)


class CrackEnrichmentValues(StrictModule, NonTrainableState):
    """Shifted partition-of-unity values at one collection of query points."""

    heaviside: Array
    williams: Array


class ShiftedCrackEnrichment(StrictModule, NonTrainableState):
    """Shifted Heaviside and Williams bases that vanish at their owning nodes."""

    geometry: CrackFrontGeometry
    heaviside_node_points: Array
    branch_node_points: Array
    basis: IsotropicWilliamsCrackTipBasis
    tip_id: int = eqx.field(static=True)
    enrichment_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CrackFrontGeometry,
        heaviside_node_points: ArrayLike,
        branch_node_points: ArrayLike,
        basis: IsotropicWilliamsCrackTipBasis,
        /,
        *,
        tip_id: int,
    ):
        if not isinstance(geometry, CrackFrontGeometry):
            raise TypeError("geometry must be CrackFrontGeometry.")
        if not isinstance(basis, IsotropicWilliamsCrackTipBasis):
            raise TypeError("basis must be IsotropicWilliamsCrackTipBasis.")
        heaviside_nodes = np.asarray(heaviside_node_points)
        branch_nodes = np.asarray(branch_node_points)
        identifier = int(tip_id)
        if (
            heaviside_nodes.ndim != 2
            or heaviside_nodes.shape[1:] != (2,)
            or branch_nodes.ndim != 2
            or branch_nodes.shape[1:] != (2,)
            or np.any(~np.isfinite(heaviside_nodes))
            or np.any(~np.isfinite(branch_nodes))
            or identifier not in set(np.asarray(geometry.tip_ids).tolist())
        ):
            raise ValueError("Shifted crack-enrichment nodes or tip ID are invalid.")
        self.geometry = geometry
        self.heaviside_node_points = jnp.asarray(heaviside_nodes)
        self.branch_node_points = jnp.asarray(branch_nodes)
        self.basis = basis
        self.tip_id = identifier
        self.enrichment_id = canonical_fingerprint(
            {
                "kind": "shifted-sharp-crack-enrichment",
                "geometry": geometry.geometry_id,
                "tip_id": identifier,
                "heaviside_nodes": heaviside_nodes.tolist(),
                "branch_nodes": branch_nodes.tolist(),
                "basis": basis.basis_id,
            }
        )

    def evaluate(self, points: ArrayLike, /) -> CrackEnrichmentValues:
        query = jnp.asarray(points)
        if query.ndim < 1 or query.shape[-1] != 2:
            raise ValueError("Crack-enrichment points must end in two coordinates.")
        query_heaviside = self.geometry.heaviside(query)
        nodal_heaviside = self.geometry.heaviside(self.heaviside_node_points)
        shifted_heaviside = query_heaviside[..., None] - nodal_heaviside
        query_williams = self.basis.values(
            self.geometry.tip_local_coordinates(query, self.tip_id)
        )
        nodal_williams = self.basis.values(
            self.geometry.tip_local_coordinates(self.branch_node_points, self.tip_id)
        )
        shifted_williams = query_williams[..., None, :] - nodal_williams
        return CrackEnrichmentValues(shifted_heaviside, shifted_williams)


def shifted_heaviside_enrichment(
    points: ArrayLike,
    node_points: ArrayLike,
    geometry: CrackFrontGeometry,
    /,
) -> Array:
    """Evaluate H(x) - H(x_i) for every query/node pair."""

    if not isinstance(geometry, CrackFrontGeometry):
        raise TypeError("geometry must be CrackFrontGeometry.")
    query = jnp.asarray(points)
    nodes = jnp.asarray(node_points)
    if (
        query.ndim < 1
        or query.shape[-1] != 2
        or nodes.ndim != 2
        or nodes.shape[1:] != (2,)
    ):
        raise ValueError("Shifted Heaviside points must be two-dimensional.")
    return geometry.heaviside(query)[..., None] - geometry.heaviside(nodes)


def shifted_williams_enrichment(
    points: ArrayLike,
    node_points: ArrayLike,
    geometry: CrackFrontGeometry,
    basis: IsotropicWilliamsCrackTipBasis,
    /,
    *,
    tip_id: int,
) -> Array:
    """Evaluate F_alpha(x) - F_alpha(x_i) for every query/node pair."""

    if not isinstance(geometry, CrackFrontGeometry) or not isinstance(
        basis, IsotropicWilliamsCrackTipBasis
    ):
        raise TypeError("Shifted Williams enrichment requires geometry and basis.")
    query = jnp.asarray(points)
    nodes = jnp.asarray(node_points)
    if (
        query.ndim < 1
        or query.shape[-1] != 2
        or nodes.ndim != 2
        or nodes.shape[1:] != (2,)
    ):
        raise ValueError("Shifted Williams points must be two-dimensional.")
    query_values = basis.values(geometry.tip_local_coordinates(query, int(tip_id)))
    node_values = basis.values(geometry.tip_local_coordinates(nodes, int(tip_id)))
    return query_values[..., None, :] - node_values


__all__ = [
    "CrackEnrichmentValues",
    "CrackTipMaterial",
    "IsotropicWilliamsCrackTipBasis",
    "ShiftedCrackEnrichment",
    "shifted_heaviside_enrichment",
    "shifted_williams_enrichment",
]
