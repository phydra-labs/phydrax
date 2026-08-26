#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Balanced, geometry-bound surface-tension actions for unstructured FV.

The capillary action in this module is deliberately a rate block rather than a
source hidden in an equation/system object.  PLIC provides the interface
orientation and centres; the cell least-squares reconstruction supplies the
same owner-oriented face gradient used by the collocated pressure operators.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class CurvatureStatus(IntEnum):
    """Per-cell status of the PLIC-normal least-squares curvature fit."""

    VALID = 0
    MISSING_INTERFACE = 1
    UNCERTAIN = 2
    INVALID_GEOMETRY = 3
    MISMATCHED_GEOMETRY = 4


class CurvatureUncertaintyError(ValueError):
    """Raised when a capillary action would use uncertified curvature."""


class CurvatureGeometryError(ValueError):
    """Raised when PLIC and finite-volume geometry identities disagree."""


class SurfaceTensionPolicy(StrictModule, NonTrainableState):
    """Immutable surface-tension and capillary-CFL policy.

    ``surface_tension=0`` is a useful exact disabling mode.  Positive surface
    tension requires a positive density floor and CFL safety factor.  All
    policy values are static so changing any one produces a new policy ID.
    """

    surface_tension: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    capillary_cfl: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_tension: float,
        density_floor: float,
        capillary_cfl: float,
        policy_id: str = "surface-tension-policy",
    ):
        sigma = float(surface_tension)
        floor = float(density_floor)
        cfl = float(capillary_cfl)
        identifier = str(policy_id)
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("surface_tension must be finite and nonnegative.")
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("density_floor must be finite and positive.")
        if not np.isfinite(cfl) or cfl <= 0.0:
            raise ValueError("capillary_cfl must be finite and positive.")
        if not identifier or identifier != identifier.strip():
            raise ValueError("policy_id must be a non-empty canonical identifier.")
        self.surface_tension = sigma
        self.density_floor = floor
        self.capillary_cfl = cfl
        self.policy_id = canonical_fingerprint(
            {
                "kind": "surface-tension-policy",
                "surface_tension": sigma,
                "density_floor": floor,
                "capillary_cfl": cfl,
                "policy_id": identifier,
            }
        )

    @property
    def sigma(self) -> float:
        """Short physical name for the configured surface tension."""

        return self.surface_tension


class CurvatureEvidence(StrictModule, NonTrainableState):
    """Least-squares curvature and its explicit validity evidence.

    ``status`` is one value per cell.  Inactive cells are marked
    ``MISSING_INTERFACE`` and have zero curvature; an action must use only the
    ``valid_mask`` cells and must reject any active non-valid status.
    """

    curvature: Array
    residual: Array
    status: Array
    interface_active: Array
    geometry_id: str = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        curvature: ArrayLike,
        residual: ArrayLike,
        status: ArrayLike,
        *,
        interface_active: ArrayLike | None = None,
        geometry_id: str = "unknown-geometry",
        reconstruction_id: str = "unknown-reconstruction",
        evidence_id: str | None = None,
        tolerance: float = 1.0e-6,
    ):
        kappa = jnp.asarray(curvature)
        fit_residual = jnp.asarray(residual)
        if (
            jnp.iscomplexobj(kappa)
            or jnp.iscomplexobj(fit_residual)
            or not jnp.issubdtype(kappa.dtype, jnp.floating)
            or not jnp.issubdtype(fit_residual.dtype, jnp.floating)
        ):
            raise TypeError(
                "Curvature and residual evidence must be real floating arrays."
            )
        state = jnp.asarray(status)
        if not jnp.issubdtype(state.dtype, jnp.signedinteger):
            raise TypeError("Curvature status must have a signed integer dtype.")
        state = state.astype(jnp.int8)
        state = eqx.error_if(
            state,
            jnp.any(
                (state < int(CurvatureStatus.VALID))
                | (state > int(CurvatureStatus.MISMATCHED_GEOMETRY))
            ),
            "Curvature status contains an unknown code.",
        )
        fit_residual = fit_residual.astype(kappa.dtype)
        if (
            kappa.ndim != 1
            or fit_residual.shape != kappa.shape
            or state.shape != kappa.shape
        ):
            raise ValueError("Curvature evidence arrays must have one value per cell.")
        if interface_active is None:
            active = state != int(CurvatureStatus.MISSING_INTERFACE)
        else:
            active = jnp.asarray(interface_active)
            if active.dtype != jnp.bool_:
                raise TypeError("interface_active must have boolean dtype.")
        if active.shape != kappa.shape:
            raise ValueError("interface_active must have one value per cell.")
        kappa = eqx.error_if(
            kappa,
            jnp.any((~active) & ((kappa != 0.0) | (fit_residual != 0.0))),
            "Inactive curvature evidence must be exactly zero.",
        )
        tol = float(tolerance)
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("Curvature tolerance must be finite and positive.")
        geometry = str(geometry_id)
        reconstruction = str(reconstruction_id)
        if not geometry or not reconstruction:
            raise ValueError("Curvature geometry and reconstruction IDs are required.")
        evidence = (
            canonical_fingerprint(
                {
                    "kind": "curvature-evidence",
                    "geometry": geometry,
                    "reconstruction": reconstruction,
                    "tolerance": tol,
                    "curvature": array_tree_fingerprint(kappa),
                    "residual": array_tree_fingerprint(fit_residual),
                    "status": array_tree_fingerprint(state),
                    "active": array_tree_fingerprint(active),
                }
            )
            if evidence_id is None
            else str(evidence_id)
        )
        if not evidence:
            raise ValueError("evidence_id must be non-empty.")
        self.curvature = kappa
        self.residual = fit_residual
        self.status = state
        self.interface_active = active
        self.geometry_id = geometry
        self.reconstruction_id = reconstruction
        self.evidence_id = evidence
        self.tolerance = tol

    @property
    def valid_mask(self) -> Array:
        return (
            self.interface_active
            & jnp.isfinite(self.curvature)
            & jnp.isfinite(self.residual)
            & (self.status == int(CurvatureStatus.VALID))
            & (self.residual <= self.tolerance)
        )

    @property
    def uncertain(self) -> Array:
        return self.interface_active & (self.status != int(CurvatureStatus.VALID))

    @property
    def curvature_status(self) -> Array:
        """Descriptive alias retained on the evidence object, not the operator."""

        return self.status

    @property
    def is_valid(self) -> Array:
        return self.valid_mask


class CapillaryFaceRateBlock(StrictModule, NonTrainableState):
    """Owner-oriented equal/opposite capillary momentum and work rates.

    ``momentum_rate`` and ``energy_work_rate`` are face rates.  The owner gets
    their negative and an interior neighbour gets their positive, exactly as
    for an ordinary owner-oriented finite-volume flux block.
    """

    momentum_rate: Array
    energy_work_rate: Array
    owner_cells: Array
    neighbour_cells: Array
    active_mask: Array
    curvature: Array
    orientation: Array
    surface_tension: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    block_id: str = eqx.field(static=True)
    rate_block_id: str = eqx.field(static=True)

    def __init__(
        self,
        momentum_rate: ArrayLike,
        energy_work_rate: ArrayLike,
        owner_cells: ArrayLike,
        neighbour_cells: ArrayLike,
        active_mask: ArrayLike,
        curvature: ArrayLike,
        orientation: ArrayLike,
        *,
        surface_tension: float,
        geometry_id: str,
        evidence_id: str,
        block_id: str = "capillary",
    ):
        momentum = jnp.asarray(momentum_rate)
        work = jnp.asarray(energy_work_rate, dtype=momentum.dtype)
        owners = jnp.asarray(owner_cells, dtype=jnp.int32)
        neighbours = jnp.asarray(neighbour_cells, dtype=jnp.int32)
        active = jnp.asarray(active_mask, dtype=bool)
        kappa = jnp.asarray(curvature, dtype=momentum.dtype)
        direction = jnp.asarray(orientation, dtype=momentum.dtype)
        if momentum.ndim != 2 or work.shape != (momentum.shape[0],):
            raise ValueError(
                "Capillary momentum/work rates must be (face, dimension)/(face,)."
            )
        face_count, dimension = momentum.shape
        if (
            owners.shape != (face_count,)
            or neighbours.shape != (face_count,)
            or active.shape != (face_count,)
            or kappa.shape != (face_count,)
            or direction.shape != (face_count, dimension)
        ):
            raise ValueError("Capillary face arrays must share one routed face axis.")
        if not np.isfinite(float(surface_tension)) or float(surface_tension) < 0.0:
            raise ValueError("surface_tension must be finite and nonnegative.")
        momentum = eqx.error_if(
            momentum,
            jnp.any(owners < 0) | jnp.any(neighbours < -1),
            "Capillary owner/neighbour routes are invalid.",
        )
        momentum = eqx.error_if(
            momentum,
            jnp.any(~jnp.isfinite(momentum)) | jnp.any(~jnp.isfinite(work)),
            "Capillary rates must be finite.",
        )
        geometry = str(geometry_id)
        evidence = str(evidence_id)
        block = str(block_id)
        if not geometry or not evidence or not block:
            raise ValueError("Capillary rate metadata IDs must be non-empty.")
        self.momentum_rate = jnp.where(active[:, None], momentum, 0.0)
        self.energy_work_rate = jnp.where(active, work, 0.0)
        self.owner_cells = owners
        self.neighbour_cells = neighbours
        self.active_mask = active
        self.curvature = jnp.where(active, kappa, 0.0)
        self.orientation = jnp.where(active[:, None], direction, 0.0)
        self.surface_tension = float(surface_tension)
        self.geometry_id = geometry
        self.evidence_id = evidence
        self.block_id = block
        self.rate_block_id = canonical_fingerprint(
            {
                "kind": "capillary-face-rate-block",
                "block": block,
                "geometry": geometry,
                "evidence": evidence,
                "surface_tension": float(surface_tension),
                "dimension": dimension,
            }
        )

    @property
    def force_rate(self) -> Array:
        return self.momentum_rate

    @property
    def work_rate(self) -> Array:
        return self.energy_work_rate

    @property
    def owner_momentum_rate(self) -> Array:
        return -self.momentum_rate

    @property
    def neighbour_momentum_rate(self) -> Array:
        return self.momentum_rate

    @property
    def owner_energy_rate(self) -> Array:
        return -self.energy_work_rate

    @property
    def neighbour_energy_rate(self) -> Array:
        return self.energy_work_rate

    @property
    def momentum_force_rate(self) -> Array:
        return self.momentum_rate

    @property
    def energy_rate(self) -> Array:
        return self.energy_work_rate

    @property
    def owner_force_rate(self) -> Array:
        return self.owner_momentum_rate

    @property
    def neighbour_force_rate(self) -> Array:
        return self.neighbour_momentum_rate

    @property
    def owner_work_rate(self) -> Array:
        return self.owner_energy_rate

    @property
    def neighbour_work_rate(self) -> Array:
        return self.neighbour_energy_rate

    def cell_momentum_rate(self, cell_count: int, /) -> Array:
        """Scatter this block into cell momentum force rates."""

        count = int(cell_count)
        result = jnp.zeros(
            (count, self.momentum_rate.shape[-1]), dtype=self.momentum_rate.dtype
        )
        safe = jnp.maximum(self.neighbour_cells, 0)
        result = result.at[self.owner_cells].add(-self.momentum_rate)
        return result.at[safe].add(
            jnp.where(self.neighbour_cells[:, None] >= 0, self.momentum_rate, 0.0)
        )

    def cell_energy_rate(self, cell_count: int, /) -> Array:
        """Scatter this block into cell energy work rates."""

        count = int(cell_count)
        result = jnp.zeros((count,), dtype=self.energy_work_rate.dtype)
        safe = jnp.maximum(self.neighbour_cells, 0)
        result = result.at[self.owner_cells].add(-self.energy_work_rate)
        return result.at[safe].add(
            jnp.where(self.neighbour_cells >= 0, self.energy_work_rate, 0.0)
        )

    def momentum_budget(self, cell_count: int, /) -> Array:
        return jnp.sum(self.cell_momentum_rate(cell_count), axis=0)

    def energy_budget(self, cell_count: int, /) -> Array:
        return jnp.sum(self.cell_energy_rate(cell_count))


class BalancedCapillaryOperator(StrictModule, NonTrainableState):
    """Geometry-bound balanced capillary pressure-gradient operator."""

    discretization: UnstructuredFiniteVolumeDiscretization
    gradient: PreparedCellPolynomialReconstruction
    policy: SurfaceTensionPolicy
    curvature_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        gradient: PreparedCellPolynomialReconstruction,
        policy: SurfaceTensionPolicy,
        *,
        curvature_tolerance: float = 1.0e-6,
        condition_limit: float = 1.0e8,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Balanced capillarity requires unstructured FV geometry.")
        if discretization.cell_dimension != 2:
            raise ValueError("Balanced capillarity currently requires 2-D PLIC geometry.")
        if not isinstance(gradient, PreparedCellPolynomialReconstruction):
            raise TypeError("gradient must be PreparedCellPolynomialReconstruction.")
        if gradient.basis.degree != 1:
            raise ValueError("Capillary curvature requires a degree-one gradient.")
        if gradient.discretization.prepared_id != discretization.prepared_id:
            raise CurvatureGeometryError(
                "Capillary gradient belongs to different geometry."
            )
        if not isinstance(policy, SurfaceTensionPolicy):
            raise TypeError("policy must be SurfaceTensionPolicy.")
        tolerance = float(curvature_tolerance)
        condition = float(condition_limit)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("curvature_tolerance must be finite and positive.")
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        self.discretization = discretization
        self.gradient = gradient
        self.policy = policy
        self.curvature_tolerance = tolerance
        self.condition_limit = condition
        self.operator_id = canonical_fingerprint(
            {
                "kind": "balanced-capillary-operator",
                "geometry": discretization.prepared_id,
                "gradient": gradient.prepared_id,
                "policy": policy.policy_id,
                "curvature_tolerance": tolerance,
                "condition_limit": condition,
            }
        )

    def _plic_values(self, plic: Any, /):
        try:
            normals = jnp.asarray(plic.normals)
            centres = getattr(plic, "interface_centers", None)
            if centres is None:
                centres = getattr(plic, "interface_centres")
            centres = jnp.asarray(centres)
            measures = jnp.asarray(plic.interface_measures)
            active = jnp.asarray(plic.interface_active, dtype=bool)
        except AttributeError as error:
            raise TypeError(
                "plic must provide normals, centres, measures, and active mask."
            ) from error
        cell_count = self.discretization.cell_count
        dimension = self.discretization.cell_dimension
        if (
            normals.shape != (cell_count, dimension)
            or centres.shape != (cell_count, dimension)
            or measures.shape != (cell_count,)
            or active.shape != (cell_count,)
        ):
            raise ValueError("PLIC arrays do not match capillary geometry.")
        geometry = getattr(plic, "geometry_id", None)
        if geometry is None:
            geometry = getattr(plic, "prepared_id", None)
        expected = {
            self.discretization.geometry_id,
            self.discretization.prepared_id,
            self.discretization.plan_id,
        }
        mismatch = geometry is not None and str(geometry) not in expected
        reconstruction = str(getattr(plic, "reconstruction_id", "unknown-reconstruction"))
        volume_fraction_id = str(
            getattr(plic, "volume_fraction_id", "unknown-volume-fraction")
        )
        return (
            normals,
            centres,
            measures,
            active,
            mismatch,
            reconstruction,
            volume_fraction_id,
        )

    def _validate_volume_fraction(self, volume_fraction: ArrayLike, /) -> Array:
        alpha = jnp.asarray(volume_fraction)
        shape = (self.discretization.cell_count,)
        if alpha.shape != shape:
            raise ValueError(f"Volume fraction must have shape {shape}.")
        return eqx.error_if(
            alpha,
            jnp.any(~jnp.isfinite(alpha) | (alpha < 0.0) | (alpha > 1.0)),
            "Capillary volume fraction must be finite and lie in [0, 1].",
        )

    def _validate_density(self, density: ArrayLike, /) -> Array:
        value = jnp.asarray(density)
        shape = (self.discretization.cell_count,)
        if value.ndim == 0:
            value = jnp.broadcast_to(value, shape)
        elif value.shape != shape:
            raise ValueError(f"Density must have shape {shape} or be scalar.")
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value < self.policy.density_floor)),
            "Capillary density is below the positive policy floor.",
        )

    def curvature(
        self,
        plic: Any,
        volume_fraction: ArrayLike,
    ) -> CurvatureEvidence:
        """Fit ``div(n)`` from PLIC normals at neighbouring PLIC centres."""

        alpha = self._validate_volume_fraction(volume_fraction)
        (
            normals_raw,
            centres,
            measures,
            active_raw,
            mismatch,
            reconstruction,
            volume_fraction_id,
        ) = self._plic_values(plic)
        dtype = jnp.result_type(alpha, normals_raw, centres)
        normals = normals_raw.astype(dtype)
        centres = centres.astype(dtype)
        measure = measures.astype(dtype)
        magnitude = jnp.linalg.norm(normals, axis=-1)
        fit_active = active_raw & (magnitude > 64.0 * jnp.finfo(dtype).eps)
        normals = normals / jnp.maximum(magnitude[:, None], jnp.finfo(dtype).tiny)
        routes = self.gradient.stencil_cells
        stencil_valid = self.gradient.stencil_valid
        same = (
            routes
            == jnp.arange(self.discretization.cell_count, dtype=routes.dtype)[:, None]
        )
        neighbours_active = fit_active[routes]
        usable = stencil_valid & neighbours_active & ~same
        usable = (
            usable
            & fit_active[:, None]
            & jnp.isfinite(measure[:, None])
            & (measure[:, None] > 0.0)
        )
        offsets = centres[routes] - centres[:, None, :]
        differences = normals[routes] - normals[:, None, :]
        distance = jnp.sqrt(jnp.sum(offsets * offsets, axis=-1))
        weights = jnp.where(
            usable, 1.0 / jnp.maximum(distance, jnp.finfo(dtype).tiny) ** 2, 0.0
        )
        gram = oe.contract("csi,csj,cs->cij", offsets, offsets, weights)
        rhs = oe.contract("csi,csj,cs->cij", differences, offsets, weights)
        scale = jnp.maximum(jnp.trace(gram, axis1=-2, axis2=-1), jnp.finfo(dtype).tiny)
        regularizer = jnp.finfo(dtype).eps * scale[:, None, None]
        fit_matrix = gram + regularizer * jnp.eye(
            self.discretization.cell_dimension,
            dtype=dtype,
        )
        solved = solve(
            LinearSystem(
                DenseLinearOperator(fit_matrix),
                problem_id=f"{self.operator_id}:curvature-fit",
            ),
            jnp.swapaxes(rhs, -1, -2),
            policy=LinearSolvePolicy(
                DenseLU(),
                failure=FailurePolicy("status"),
            ),
        )
        jacobian = jnp.swapaxes(solved.value, -1, -2)
        predicted = oe.contract("cij,csj->csi", jacobian, offsets)
        error = differences - predicted
        residual = jnp.sqrt(
            oe.contract("csi,csi,cs->c", error, error, weights)
            / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        )
        determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]
        trace = jnp.trace(gram, axis1=-2, axis2=-1)
        discriminant = jnp.sqrt(jnp.maximum(trace * trace - 4.0 * determinant, 0.0))
        minimum = 0.5 * (trace - discriminant)
        maximum = 0.5 * (trace + discriminant)
        condition = maximum / jnp.maximum(minimum, jnp.finfo(dtype).tiny)
        neighbour_count = jnp.sum(usable, axis=1)
        constant_normal = (neighbour_count >= 1) & (
            jnp.max(
                jnp.where(
                    usable,
                    jnp.sqrt(jnp.sum(differences * differences, axis=-1)),
                    0.0,
                ),
                axis=1,
            )
            <= self.curvature_tolerance
        )

        rank_valid = (
            (neighbour_count >= self.discretization.cell_dimension)
            & (determinant > self.curvature_tolerance**2 * jnp.maximum(scale, 1.0) ** 2)
            & (condition <= self.condition_limit)
        )
        finite = jnp.isfinite(jacobian).all(axis=(-2, -1)) & jnp.isfinite(residual)
        valid = fit_active & (rank_valid | constant_normal) & finite & (not mismatch)
        kappa = jnp.trace(jacobian, axis1=-2, axis2=-1)
        kappa = jnp.where(constant_normal, 0.0, kappa)
        kappa = jnp.where(valid, kappa, 0.0)
        residual = jnp.where(active_raw, residual, 0.0)
        status = jnp.where(
            mismatch,
            int(CurvatureStatus.MISMATCHED_GEOMETRY),
            jnp.where(
                active_raw,
                jnp.where(
                    valid, int(CurvatureStatus.VALID), int(CurvatureStatus.UNCERTAIN)
                ),
                int(CurvatureStatus.MISSING_INTERFACE),
            ),
        ).astype(jnp.int8)
        plic_geometry = getattr(plic, "geometry_id", None)
        if plic_geometry is None:
            plic_geometry = getattr(plic, "prepared_id", self.discretization.prepared_id)
        return CurvatureEvidence(
            kappa,
            residual,
            status,
            interface_active=active_raw,
            geometry_id=str(plic_geometry),
            reconstruction_id=reconstruction,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "curvature-evidence",
                    "operator": self.operator_id,
                    "reconstruction": reconstruction,
                    "volume_fraction": volume_fraction_id,
                    "geometry": str(plic_geometry),
                }
            ),
            tolerance=self.curvature_tolerance,
        )

    def _face_arrays(
        self,
        plic: Any,
        density: ArrayLike,
        volume_fraction: ArrayLike | None,
        velocity: ArrayLike | None,
        /,
    ):
        density_ = self._validate_density(density)
        del density_
        if volume_fraction is None:
            # This fallback is intentionally conservative: only PLIC-cut cells
            # carry phase-one volume in the absence of a caller-owned alpha.
            active = jnp.asarray(plic.interface_active, dtype=bool)
            alpha = active.astype(
                jnp.result_type(self.discretization.cell_volumes, jnp.float32)
            )
        else:
            alpha = self._validate_volume_fraction(volume_fraction)
        evidence = self.curvature(plic, alpha)
        sigma = self.policy.surface_tension
        dtype = jnp.result_type(alpha, self.discretization.area_vectors)
        zero_force = jnp.zeros(
            (self.discretization.face_measures.size, self.discretization.cell_dimension),
            dtype=dtype,
        )
        zero_work = jnp.zeros((self.discretization.face_measures.size,), dtype=dtype)
        normals = self.discretization.area_vectors.astype(dtype)
        orientation = normals / jnp.maximum(
            self.discretization.face_measures.astype(dtype)[:, None],
            jnp.finfo(dtype).tiny,
        )
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        interior = neighbour >= 0
        if sigma == 0.0:
            return (
                zero_force,
                zero_work,
                jnp.zeros_like(zero_work),
                orientation,
                interior,
                evidence,
            )
        valid = evidence.valid_mask
        measure = jnp.asarray(plic.interface_measures, dtype=dtype)
        weights = jnp.where(valid, measure, 0.0)
        mean_curvature = jnp.sum(weights * evidence.curvature) / jnp.maximum(
            jnp.sum(weights), jnp.finfo(dtype).tiny
        )
        # PLIC curvature is defined on cut cells.  Extending its local value
        # through alpha gives a pressure jump in pure cells while retaining
        # local curvature at the interface itself.
        local = jnp.where(valid, evidence.curvature, mean_curvature)
        pressure = sigma * local * alpha
        coefficients = self.gradient.coefficients(pressure)
        lengths = self.gradient.characteristic_lengths.astype(dtype)
        cell_gradient = coefficients / lengths[:, None]
        average = 0.5 * (cell_gradient[owner] + cell_gradient[safe_neighbour])
        normal_gradient = oe.contract("fd,fd->f", average, orientation)
        normal_gradient = jnp.where(interior, normal_gradient, 0.0)
        force = normal_gradient[:, None] * normals
        face_curvature = 0.5 * (local[owner] + local[safe_neighbour])
        face_curvature = jnp.where(interior, face_curvature, 0.0)
        if velocity is None:
            work = zero_work
        else:
            speed = jnp.asarray(velocity, dtype=dtype)
            expected = (
                self.discretization.cell_count,
                self.discretization.cell_dimension,
            )
            if speed.shape != expected:
                raise ValueError(f"Velocity must have shape {expected}.")
            face_speed = 0.5 * (speed[owner] + speed[safe_neighbour])
            face_speed = jnp.where(interior[:, None], face_speed, 0.0)
            work = oe.contract("fd,fd->f", force, face_speed)
        has_interface = jnp.any(evidence.interface_active)
        force = jnp.where(has_interface, force, zero_force)
        work = jnp.where(has_interface, work, zero_work)
        face_curvature = jnp.where(has_interface, face_curvature, 0.0)
        return force, work, face_curvature, orientation, interior, evidence

    def _check_evidence(
        self,
        value: Array,
        evidence: CurvatureEvidence,
        /,
    ) -> Array:
        return eqx.error_if(
            value,
            jnp.any(
                evidence.interface_active
                & (evidence.status != int(CurvatureStatus.VALID))
            )
            & (self.policy.surface_tension != 0.0),
            "Capillary curvature evidence is not valid.",
        )

    def face_rate_block(
        self,
        plic: Any,
        density: ArrayLike,
        volume_fraction: ArrayLike | None = None,
        velocity: ArrayLike | None = None,
        *,
        block_id: str = "capillary",
    ) -> CapillaryFaceRateBlock:
        """Return one balanced owner-oriented capillary face-rate block."""

        force, work, curvature, orientation, active, evidence = self._face_arrays(
            plic, density, volume_fraction, velocity
        )
        force = self._check_evidence(force, evidence)
        return CapillaryFaceRateBlock(
            force,
            work,
            self.discretization.owner_cells,
            self.discretization.neighbour_cells,
            active,
            curvature,
            orientation,
            surface_tension=self.policy.surface_tension,
            geometry_id=self.discretization.prepared_id,
            evidence_id=evidence.evidence_id,
            block_id=block_id,
        )

    def momentum_force_rate(
        self,
        plic: Any,
        density: ArrayLike,
        volume_fraction: ArrayLike | None = None,
    ) -> Array:
        """JAX-traceable face momentum rate without metadata construction."""

        force, _, _, _, _, evidence = self._face_arrays(
            plic, density, volume_fraction, None
        )
        checked = self._check_evidence(force, evidence)
        return checked

    def energy_work_rate(
        self,
        plic: Any,
        density: ArrayLike,
        volume_fraction: ArrayLike | None = None,
        velocity: ArrayLike | None = None,
    ) -> Array:
        """JAX-traceable owner-oriented face work rate."""

        _, work, _, _, _, evidence = self._face_arrays(
            plic, density, volume_fraction, velocity
        )
        return self._check_evidence(work, evidence)

    def force_rate(
        self,
        plic: Any,
        density: ArrayLike,
        volume_fraction: ArrayLike | None = None,
    ) -> Array:
        return self.momentum_force_rate(plic, density, volume_fraction)

    def work_rate(
        self,
        plic: Any,
        density: ArrayLike,
        volume_fraction: ArrayLike | None = None,
        velocity: ArrayLike | None = None,
    ) -> Array:
        return self.energy_work_rate(plic, density, volume_fraction, velocity)

    def laplace_pressure_jump(
        self,
        plic: Any,
        volume_fraction: ArrayLike,
    ) -> Array:
        """Return the signed ``sigma * kappa`` jump (phase one minus zero)."""

        evidence = self.curvature(plic, volume_fraction)
        jump = self.policy.surface_tension * evidence.curvature
        return self._check_evidence(jump, evidence)

    def capillary_step(
        self,
        cell_size: ArrayLike,
        density: ArrayLike,
        /,
        *,
        interface_active: ArrayLike | None = None,
    ) -> Array:
        """Return the capillary restriction, or infinity without an interface."""

        rho = self._validate_density(density)
        if interface_active is None:
            has_interface = jnp.asarray(True)
        else:
            active = jnp.asarray(interface_active, dtype=bool)
            if active.shape != rho.shape:
                raise ValueError(
                    "interface_active must have one value per capillary cell."
                )
            has_interface = jnp.any(active)
        h = jnp.asarray(cell_size, dtype=rho.dtype)
        if h.ndim == 0:
            h = jnp.broadcast_to(h, rho.shape)
        elif h.shape == (
            self.discretization.cell_count,
            self.discretization.cell_dimension,
        ):
            h = jnp.min(h, axis=-1)
        elif h.shape != rho.shape:
            raise ValueError(
                "cell_size must be scalar, per-cell, or per-cell/per-dimension."
            )
        h = eqx.error_if(
            h,
            jnp.any(~jnp.isfinite(h) | (h <= 0.0)),
            "Capillary cell sizes must be finite and positive.",
        )
        if self.policy.surface_tension == 0.0:
            return jnp.asarray(jnp.inf, dtype=rho.dtype)
        restricted = self.policy.capillary_cfl * jnp.min(
            jnp.sqrt(rho * h**3 / self.policy.surface_tension)
        )
        return jnp.where(
            has_interface,
            restricted,
            jnp.asarray(jnp.inf, dtype=rho.dtype),
        )


__all__ = [
    "BalancedCapillaryOperator",
    "CapillaryFaceRateBlock",
    "CurvatureEvidence",
    "CurvatureGeometryError",
    "CurvatureStatus",
    "CurvatureUncertaintyError",
    "SurfaceTensionPolicy",
]
