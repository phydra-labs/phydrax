#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


class DEMMulticontactCorrection(StrictModule):
    gap_correction: Array
    endpoint_deformation: Array
    residual: Array
    regularity_margin: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class AbstractDEMContactGraphCorrectionPlan(StrictModule, NonTrainableState):
    plan_id: AbstractAttribute[str]
    iterations: AbstractAttribute[int]
    convergence_tolerance: AbstractAttribute[float]

    @abc.abstractmethod
    def evaluate(
        self,
        left_indices,
        right_indices,
        contact_points,
        contact_normals,
        compressive_force,
        material_ids,
        materials,
        valid,
        /,
    ) -> DEMMulticontactCorrection:
        raise NotImplementedError


class ElasticHalfSpaceMulticontactPlan(AbstractDEMContactGraphCorrectionPlan):
    geometric_prefactor: float = eqx.field(static=True)
    distance_regularization: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometric_prefactor: float = 1.0,
        distance_regularization: float = 1.0e-6,
        iterations: int = 2,
        convergence_tolerance: float = 1.0e-6,
        plan_id: str | None = None,
    ):
        prefactor = float(geometric_prefactor)
        regularization = float(distance_regularization)
        count = int(iterations)
        tolerance = float(convergence_tolerance)
        if (
            not np.isfinite(prefactor)
            or prefactor <= 0.0
            or not np.isfinite(regularization)
            or regularization <= 0.0
            or count <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Multicontact controls must be finite and positive.")
        generated = canonical_fingerprint(
            {
                "kind": "elastic-half-space-multicontact",
                "geometric_prefactor": prefactor,
                "distance_regularization": regularization,
                "iterations": count,
                "convergence_tolerance": tolerance,
            }
        )
        self.geometric_prefactor = prefactor
        self.distance_regularization = regularization
        self.iterations = count
        self.convergence_tolerance = tolerance
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def evaluate(
        self,
        left_indices,
        right_indices,
        contact_points,
        contact_normals,
        compressive_force,
        material_ids,
        materials,
        valid,
        /,
    ):
        left = jnp.asarray(left_indices, dtype=jnp.int32)
        right = jnp.asarray(right_indices, dtype=jnp.int32)
        points = jnp.asarray(contact_points)
        normal = jnp.asarray(contact_normals)
        force = jnp.asarray(compressive_force, dtype=points.dtype)
        mask = jnp.asarray(valid, dtype=bool)
        if (
            left.shape != right.shape
            or left.ndim != 1
            or points.ndim != 2
            or points.shape[0] != left.shape[0]
            or points.shape[1] not in (2, 3)
            or normal.shape != points.shape
            or force.shape != left.shape
            or mask.shape != left.shape
        ):
            raise ValueError("Multicontact pair data has inconsistent shapes.")
        edge_count = left.shape[0]
        endpoint_owner = jnp.concatenate((left, right))
        endpoint_edge = jnp.concatenate(
            (
                jnp.arange(edge_count, dtype=jnp.int32),
                jnp.arange(edge_count, dtype=jnp.int32),
            )
        )
        endpoint_point = jnp.concatenate((points, points), axis=0)
        endpoint_normal = jnp.concatenate((-normal, normal), axis=0)
        endpoint_force = jnp.concatenate((force, force))
        endpoint_valid = jnp.concatenate((mask, mask))
        owner_material = jnp.asarray(material_ids, dtype=jnp.int32)[endpoint_owner]
        young = materials.young_modulus[owner_material].astype(points.dtype)
        poisson = materials.poisson_ratio[owner_material].astype(points.dtype)
        displacement = endpoint_point[:, None, :] - endpoint_point[None, :, :]
        distance = jnp.linalg.norm(displacement, axis=-1)
        same_owner = endpoint_owner[:, None] == endpoint_owner[None, :]
        distinct_contact = endpoint_edge[:, None] != endpoint_edge[None, :]
        active = (
            same_owner
            & distinct_contact
            & endpoint_valid[:, None]
            & endpoint_valid[None, :]
        )
        alignment = 0.5 * (
            1.0
            + jnp.abs(
                jnp.sum(
                    endpoint_normal[:, None, :] * endpoint_normal[None, :, :],
                    axis=-1,
                )
            )
        )
        compliance = (
            self.geometric_prefactor
            * (1.0 - poisson[:, None] ** 2)
            / (
                jnp.pi
                * young[:, None]
                * jnp.maximum(distance, self.distance_regularization)
            )
        )
        influence = jnp.where(
            active,
            compliance * alignment * endpoint_force[None, :],
            0.0,
        )
        endpoint_deformation = jnp.sum(influence, axis=1)
        gap_correction = (
            endpoint_deformation[:edge_count] + endpoint_deformation[edge_count:]
        )
        regularity = jnp.min(
            jnp.where(active, distance, jnp.inf),
            axis=1,
        )
        regularity_margin = jnp.min(
            jnp.concatenate(
                (
                    regularity,
                    jnp.asarray([jnp.inf], dtype=points.dtype),
                )
            )
        )
        finite = (
            jnp.all(jnp.isfinite(gap_correction))
            & jnp.all(gap_correction >= 0.0)
            & jnp.all(jnp.isfinite(endpoint_deformation))
        )
        return DEMMulticontactCorrection(
            gap_correction,
            endpoint_deformation,
            jnp.zeros((), dtype=points.dtype),
            regularity_margin,
            finite,
            self.plan_id,
        )


__all__ = [
    "AbstractDEMContactGraphCorrectionPlan",
    "DEMMulticontactCorrection",
    "ElasticHalfSpaceMulticontactPlan",
]
