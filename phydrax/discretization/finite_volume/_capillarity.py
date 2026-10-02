#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Balanced, geometry-bound surface-tension actions for finite-volume grids.

Sign convention (shared by every action in this module).  ``alpha`` is the
volume fraction of the alpha phase (phase zero of ``TwoMaterialVOFSystem``,
the liquid of structured two-phase VOF).  The interface normal
``n = -grad(alpha) / |grad(alpha)|`` points out of the alpha phase, the
curvature is ``kappa = div(n)`` (positive for a convex alpha region, e.g.
``1/R`` for a circular drop and ``-1/R`` for a circular bubble), and the
Laplace jump is ``p_alpha - p_complement = sigma * kappa``.  The interfacial
potential ``phi = sigma * kappa`` is the variational derivative of the surface
energy ``sigma * integral |grad(alpha)|`` with respect to alpha, and the
continuum capillary force density is ``phi * grad(alpha)``.

The unstructured action is a rate block rather than a source hidden in an
equation/system object: PLIC provides the interface orientation and centers
for the curvature, and the balanced force ``phi_f`` times the face alpha jump
uses the same face operator as the arithmetic-mean Riemann face pressure of
the collocated update.  The MAC action applies
the balanced interfacial-potential force ``phi_f (G alpha)_f`` with ``G`` the
exact face gradient of the MAC pressure projection, so a spatially constant
potential is removed to rounding by the projection and appears as the
pressure jump instead of a parasitic current.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, parameter_field
from ...typing import checked
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._incompressible import FaceVelocity, PreparedMACOperators
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class CurvatureStatus(IntEnum):
    """Per-cell status of a curvature estimate.

    ``FALLBACK`` is a bounded secondary estimate that remains usable by
    status-driven MAC actions; ``UNDERRESOLVED`` refuses an interface whose
    local stencil cannot support a curvature estimate.
    """

    VALID = 0
    MISSING_INTERFACE = 1
    UNCERTAIN = 2
    INVALID_GEOMETRY = 3
    MISMATCHED_GEOMETRY = 4
    FALLBACK = 5
    UNDERRESOLVED = 6


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
    ) -> None:
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


@final
class LinearSurfaceTensionLaw(StrictModule):
    """Linear material-scalar law ``sigma(q) = sigma_0 + sigma_q (q - q_0)``.

    All three physical coefficients are dynamic parameter leaves. The law
    refuses negative surface tension at evaluation rather than clipping it.
    """

    reference_surface_tension: Array = parameter_field()
    scalar_slope: Array = parameter_field()
    reference_scalar: Array = parameter_field()
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_surface_tension: ArrayLike,
        scalar_slope: ArrayLike,
        reference_scalar: ArrayLike,
        /,
    ) -> None:
        sigma = jnp.asarray(reference_surface_tension)
        slope = jnp.asarray(scalar_slope, dtype=sigma.dtype)
        reference = jnp.asarray(reference_scalar, dtype=sigma.dtype)
        if (
            sigma.shape != ()
            or slope.shape != ()
            or reference.shape != ()
            or bool(
                ~jnp.isfinite(sigma)
                | ~jnp.isfinite(slope)
                | ~jnp.isfinite(reference)
                | (sigma < 0.0)
            )
        ):
            raise ValueError(
                "Linear surface-tension coefficients must be finite scalars and "
                "reference_surface_tension must be nonnegative."
            )
        self.reference_surface_tension = sigma
        self.scalar_slope = slope
        self.reference_scalar = reference
        self.law_id = canonical_fingerprint({"kind": "linear-surface-tension-law"})

    def __call__(
        self, coordinates: Array, state: Array, args: Any = None, /
    ) -> tuple[Array, Array]:
        """Return surface tension and its derivative with respect to state."""

        del coordinates, args
        if state.ndim == 0 or state.shape[-1] != 1:
            raise ValueError("LinearSurfaceTensionLaw expects one scalar state channel.")
        scalar = state[..., 0]
        sigma = self.reference_surface_tension + self.scalar_slope * (
            scalar - self.reference_scalar
        )
        derivative = jnp.broadcast_to(self.scalar_slope, state.shape)
        return sigma, derivative


class SurfaceTensionEvaluation(StrictModule):
    surface_tension: Array
    marangoni_gradient: Array
    active: Array


class VariableSurfaceTensionPolicy(StrictModule):
    """Typed variable-sigma law with a projected Marangoni gradient."""

    evaluator: Any
    density_floor: float = eqx.field(static=True)
    capillary_cfl: float = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Any,
        /,
        *,
        density_floor: float,
        capillary_cfl: float,
        law_id: str,
    ) -> None:
        if not callable(evaluator):
            raise TypeError("Variable surface-tension evaluator must be callable.")
        floor = float(density_floor)
        cfl = float(capillary_cfl)
        identifier = str(law_id)
        if (
            not np.isfinite(floor)
            or floor <= 0.0
            or not np.isfinite(cfl)
            or cfl <= 0.0
            or not identifier
        ):
            raise ValueError("Variable surface-tension policy parameters are invalid.")
        self.evaluator = evaluator
        self.density_floor = floor
        self.capillary_cfl = cfl
        self.law_id = identifier
        self.policy_id = canonical_fingerprint(
            {
                "kind": "variable-surface-tension-policy",
                "law": identifier,
                "density_floor": floor,
                "capillary_cfl": cfl,
            }
        )

    def evaluate(
        self,
        coordinates: ArrayLike,
        state: ArrayLike,
        normal: ArrayLike,
        state_gradient: ArrayLike,
        args: Any = None,
        /,
    ) -> SurfaceTensionEvaluation:
        points = jnp.asarray(coordinates)
        values = jnp.asarray(state)
        normal_ = jnp.asarray(normal)
        gradient = jnp.asarray(state_gradient)
        sigma, sigma_state_gradient = self.evaluator(points, values, args)
        sigma_ = jnp.asarray(sigma)
        derivative = jnp.asarray(sigma_state_gradient)
        if sigma_.shape != values.shape[:-1]:
            raise ValueError("Surface-tension law must match the interface batch.")
        if derivative.shape != values.shape:
            raise ValueError("Surface-tension state derivative must match state.")
        physical_gradient = ein.contract(
            "...i,...id->...d", derivative, gradient, backend="jax"
        )
        tangential = (
            physical_gradient
            - ein.contract("...d,...d->...", physical_gradient, normal_, backend="jax")[
                ..., None
            ]
            * normal_
        )
        sigma_ = eqx.error_if(
            sigma_,
            jnp.any(~jnp.isfinite(sigma_) | (sigma_ < 0.0))
            | jnp.any(~jnp.isfinite(tangential)),
            "Variable surface tension and Marangoni gradient must be finite.",
        )
        return SurfaceTensionEvaluation(sigma_, tangential, sigma_ > 0.0)


class CurvatureEvidence(StrictModule, NonTrainableState):
    """Curvature estimate and its explicit validity evidence.

    ``status`` holds one value per cell (a flat cell axis for unstructured
    geometry, the structured cell shape for MAC grids).  ``interface_delta``
    is a cell-centered surface-delta estimate with an explicit support mask;
    unsupported evidence stores exact zeros. Inactive cells are marked
    ``MISSING_INTERFACE`` and have zero curvature.  The unstructured action
    uses only ``valid_mask`` cells; status-driven MAC actions use
    ``usable_mask`` (valid or bounded fallback) and refuse every other active
    status.
    """

    curvature: Array
    residual: Array
    status: Array
    interface_active: Array
    interface_delta: Array
    interface_delta_supported: Array
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
        interface_delta: ArrayLike | None = None,
        interface_delta_supported: ArrayLike | None = None,
        geometry_id: str = "unknown-geometry",
        reconstruction_id: str = "unknown-reconstruction",
        evidence_id: str | None = None,
        tolerance: float = 1.0e-6,
    ) -> None:
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
            jnp.any((state < min(CurvatureStatus)) | (state > max(CurvatureStatus))),
            "Curvature status contains an unknown code.",
        )
        fit_residual = fit_residual.astype(kappa.dtype)
        if (
            kappa.ndim < 1
            or fit_residual.shape != kappa.shape
            or state.shape != kappa.shape
        ):
            raise ValueError("Curvature evidence arrays must share one cell shape.")
        if interface_active is None:
            active = state != int(CurvatureStatus.MISSING_INTERFACE)
        else:
            active = jnp.asarray(interface_active)
            if active.dtype != jnp.bool_:
                raise TypeError("interface_active must have boolean dtype.")
        if active.shape != kappa.shape:
            raise ValueError("interface_active must have one value per cell.")
        if (interface_delta is None) != (interface_delta_supported is None):
            raise ValueError(
                "interface_delta and interface_delta_supported must be given together."
            )
        if interface_delta is None:
            delta = jnp.zeros_like(kappa)
            delta_supported = jnp.zeros_like(active)
        else:
            delta = jnp.asarray(interface_delta, dtype=kappa.dtype)
            delta_supported = jnp.asarray(interface_delta_supported)
            if delta_supported.dtype != jnp.bool_:
                raise TypeError("interface_delta_supported must have boolean dtype.")
            if delta.shape != kappa.shape or delta_supported.shape != kappa.shape:
                raise ValueError(
                    "Interface-delta evidence must match the curvature cell shape."
                )
            delta = eqx.error_if(
                delta,
                jnp.any(~jnp.isfinite(delta) | (delta < 0.0))
                | jnp.any((~delta_supported) & (delta != 0.0))
                | jnp.any(delta_supported & ((~active) | (delta <= 0.0))),
                "Interface delta must be finite, positive on active support, and "
                "zero off support.",
            )
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
                    "interface_delta": array_tree_fingerprint(delta),
                    "interface_delta_supported": array_tree_fingerprint(delta_supported),
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
        self.interface_delta = delta
        self.interface_delta_supported = delta_supported
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
    def usable_mask(self) -> Array:
        """Active cells carrying a finite valid or bounded-fallback curvature."""

        return (
            self.interface_active
            & jnp.isfinite(self.curvature)
            & (
                (self.status == int(CurvatureStatus.VALID))
                | (self.status == int(CurvatureStatus.FALLBACK))
            )
        )

    @property
    def curvature_status(self) -> Array:
        """Descriptive alias retained on the evidence object, not the operator."""

        return self.status

    @property
    def is_valid(self) -> Array:
        return self.valid_mask


class CapillaryFaceRateBlock(StrictModule, NonTrainableState):
    """Balanced capillary face forces and their work on the adjacent cells.

    ``face_force[f] = phi_f (alpha_neighbor - alpha_owner) A_f n_f`` is the
    force carried by interior face ``f``, with ``A_f n_f`` its
    owner-to-neighbor area vector and ``phi_f = sigma * kappa_f`` the face
    interfacial potential in ``face_potential``.  Each adjacent cell receives
    half of it, so a cell's momentum rate is
    ``sum_f phi_f (alpha_f - alpha_P) A_f n_out`` with the arithmetic-mean
    face fraction ``alpha_f``: the Green-Gauss volume integral of the force
    density ``phi grad(alpha)``.  Both halves point the same way; this is a
    body force, not a flux.  :attr:`net_force` is the discrete
    ``oint sigma kappa n dS``: exactly zero for a uniform potential, a
    truncation residual otherwise.  ``owner_work_rate`` and
    ``neighbor_work_rate`` are the work of each half at that cell's
    velocity, so the capillary energy rate equals the kinetic-energy rate of
    the momentum source and does no work on internal energy.  Physical
    boundary faces carry no force.
    """

    face_force: Array
    owner_work_rate: Array
    neighbor_work_rate: Array
    face_potential: Array
    owner_cells: Array
    neighbor_cells: Array
    surface_tension: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    block_id: str = eqx.field(static=True)
    rate_block_id: str = eqx.field(static=True)

    def __init__(
        self,
        face_force: ArrayLike,
        owner_work_rate: ArrayLike,
        neighbor_work_rate: ArrayLike,
        face_potential: ArrayLike,
        owner_cells: ArrayLike,
        neighbor_cells: ArrayLike,
        *,
        surface_tension: float,
        geometry_id: str,
        evidence_id: str,
        block_id: str = "capillary",
    ) -> None:
        force = jnp.asarray(face_force)
        owner_work = jnp.asarray(owner_work_rate, dtype=force.dtype)
        neighbor_work = jnp.asarray(neighbor_work_rate, dtype=force.dtype)
        potential = jnp.asarray(face_potential, dtype=force.dtype)
        owners = jnp.asarray(owner_cells, dtype=jnp.int32)
        neighbors = jnp.asarray(neighbor_cells, dtype=jnp.int32)
        if force.ndim != 2:
            raise ValueError("Capillary face force must have shape (face, dimension).")
        face_count, dimension = force.shape
        if any(
            value.shape != (face_count,)
            for value in (owner_work, neighbor_work, potential, owners, neighbors)
        ):
            raise ValueError("Capillary face arrays must share one routed face axis.")
        sigma = float(surface_tension)
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("surface_tension must be finite and nonnegative.")
        geometry = str(geometry_id)
        evidence = str(evidence_id)
        block = str(block_id)
        if not geometry or not evidence or not block:
            raise ValueError("Capillary rate metadata IDs must be non-empty.")
        boundary = neighbors < 0
        force = eqx.error_if(
            force,
            jnp.any(owners < 0) | jnp.any(neighbors < -1),
            "Capillary owner/neighbor routes are invalid.",
        )
        force = eqx.error_if(
            force,
            jnp.any(~jnp.isfinite(force))
            | jnp.any(~jnp.isfinite(owner_work))
            | jnp.any(~jnp.isfinite(neighbor_work))
            | jnp.any(~jnp.isfinite(potential)),
            "Capillary rates must be finite.",
        )
        force = eqx.error_if(
            force,
            jnp.any(boundary[:, None] & (force != 0.0))
            | jnp.any(boundary & (neighbor_work != 0.0)),
            "Capillary boundary faces must carry no force.",
        )
        self.face_force = force
        self.owner_work_rate = owner_work
        self.neighbor_work_rate = neighbor_work
        self.face_potential = potential
        self.owner_cells = owners
        self.neighbor_cells = neighbors
        self.surface_tension = sigma
        self.geometry_id = geometry
        self.evidence_id = evidence
        self.block_id = block
        self.rate_block_id = canonical_fingerprint(
            {
                "kind": "capillary-face-force-block",
                "block": block,
                "geometry": geometry,
                "evidence": evidence,
                "surface_tension": sigma,
                "dimension": dimension,
            }
        )

    def cell_momentum_rate(self, cell_count: int, /) -> Array:
        """Return the cell momentum rates: half of each face force per cell."""

        half = 0.5 * self.face_force
        result = jnp.zeros((cell_count, half.shape[-1]), dtype=half.dtype)
        result = result.at[self.owner_cells].add(half)
        return result.at[jnp.maximum(self.neighbor_cells, 0)].add(
            jnp.where(self.neighbor_cells[:, None] >= 0, half, 0.0)
        )

    def cell_energy_rate(self, cell_count: int, /) -> Array:
        """Return the cell capillary work rates."""

        result = jnp.zeros((cell_count,), dtype=self.owner_work_rate.dtype)
        result = result.at[self.owner_cells].add(self.owner_work_rate)
        return result.at[jnp.maximum(self.neighbor_cells, 0)].add(
            jnp.where(self.neighbor_cells >= 0, self.neighbor_work_rate, 0.0)
        )

    @property
    def net_force(self) -> Array:
        """Total capillary force over all cells (discrete closed-surface integral)."""

        return jnp.sum(self.face_force, axis=0)

    @property
    def net_power(self) -> Array:
        """Total capillary work rate over all cells."""

        return jnp.sum(self.owner_work_rate) + jnp.sum(self.neighbor_work_rate)


class BalancedCapillaryOperator(StrictModule, NonTrainableState):
    """Balanced capillary force for collocated unstructured finite volumes.

    The continuum force density is ``phi grad(alpha)`` with the interfacial
    potential ``phi = sigma * kappa`` (module sign convention); at rest it is
    balanced by ``p = p0 + phi * alpha`` when ``phi`` is uniform.

    The collocated Riemann update gives a cell ``P`` at rest the pressure
    force ``-sum_f p_f A_f n_out`` with the arithmetic-mean face pressure
    ``p_f = (p_P + p_Q) / 2``.  The area vectors of a closed cell sum to zero,
    so this is ``-1/2 sum_f (p_Q - p_P) A_f n_out``.  The capillary force uses
    the same face operator on alpha::

        F_P = 1/2 sum_f phi_f (alpha_Q - alpha_P) A_f n_out

    so ``F_P`` cancels the pressure force of ``p0 + phi * alpha`` to rounding
    whenever the face potentials are uniform.  That includes wall and
    extrapolated boundary faces, where ``p_f = p_P`` and alpha has no jump.
    Per unit volume ``F_P`` approximates ``phi grad(alpha)``, independent of
    the mesh size.  A nonuniform potential leaves the non-gradient part
    ``grad(phi) x grad(alpha) != 0`` that drives capillary motion.  Each face
    force ``phi_f (alpha_Q - alpha_P) A_f n_f`` is shared equally by its two
    cells; see :class:`CapillaryFaceRateBlock` for momentum and work.

    A pressure-like flux ``-phi_f alpha_f A_f n_f`` conserves momentum face
    by face, but to leading order it is the gradient of ``phi * alpha``.
    Pressure absorbs that curl-free force for any interface shape, which
    removes the capillary restoring force.

    ``phi_f`` is ``sigma`` times the mean curvature over the adjacent cells
    with valid curvature (:attr:`CurvatureEvidence.valid_mask`).  This is
    the Francois et al. (2006) balanced-force construction, as in
    :class:`MACBalancedCapillaryOperator`.  ``condition_limit`` bounds the
    tangential curvature fit (see :meth:`curvature`).
    """

    discretization: UnstructuredFiniteVolumeDiscretization
    gradient: PreparedCellPolynomialReconstruction
    policy: SurfaceTensionPolicy
    curvature_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        gradient: PreparedCellPolynomialReconstruction,
        policy: SurfaceTensionPolicy,
        *,
        curvature_tolerance: float = 1.0e-6,
        condition_limit: float = 1.0e8,
    ) -> None:
        if discretization.cell_dimension != 2:
            raise ValueError("Balanced capillarity currently requires 2-D PLIC geometry.")
        if gradient.basis.degree != 1:
            raise ValueError("Capillary curvature requires a degree-one gradient.")
        if gradient.discretization.prepared_id != discretization.prepared_id:
            raise CurvatureGeometryError(
                "Capillary gradient belongs to different geometry."
            )
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
                "curvature_fit": "tangential-normal-turn",
                "face_force": "valid-neighbor-potential-alpha-jump",
            }
        )

    def _plic_values(
        self, plic: Any, /
    ) -> tuple[Array, Array, Array, Array, bool, str, str]:
        try:
            normals = jnp.asarray(plic.normals)
            centers = getattr(plic, "interface_centers", None)
            if centers is None:
                centers = getattr(plic, "interface_centers")
            centers = jnp.asarray(centers)
            measures = jnp.asarray(plic.interface_measures)
            active = jnp.asarray(plic.interface_active, dtype=jnp.bool_)
        except AttributeError as error:
            raise TypeError(
                "plic must provide normals, centers, measures, and active mask."
            ) from error
        cell_count = self.discretization.cell_count
        dimension = self.discretization.cell_dimension
        if (
            normals.shape != (cell_count, dimension)
            or centers.shape != (cell_count, dimension)
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
            centers,
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
        """Fit ``kappa = t . dn/ds`` from PLIC normals at neighboring centers.

        A cell is ``VALID`` with at least two usable neighbors whose offsets
        are not too close to normal to the tangent: the inverse mean squared
        cosine between offset and tangent must not exceed ``condition_limit``.
        A constant normal counts as zero curvature.  ``residual`` is the
        weighted RMS misfit of the tangential normal turn.
        """

        alpha = self._validate_volume_fraction(volume_fraction)
        (
            normals_raw,
            centers,
            measures,
            active_raw,
            mismatch,
            reconstruction,
            volume_fraction_id,
        ) = self._plic_values(plic)
        dtype = jnp.result_type(alpha, normals_raw, centers)
        normals = normals_raw.astype(dtype)
        centers = centers.astype(dtype)
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
        neighbors_active = fit_active[routes]
        usable = stencil_valid & neighbors_active & ~same
        usable = (
            usable
            & fit_active[:, None]
            & jnp.isfinite(measure[:, None])
            & (measure[:, None] > 0.0)
        )
        offsets = centers[routes] - centers[:, None, :]
        differences = normals[routes] - normals[:, None, :]
        distance = jnp.sqrt(jnp.sum(offsets * offsets, axis=-1))
        weights = jnp.where(
            usable, 1.0 / jnp.maximum(distance, jnp.finfo(dtype).tiny) ** 2, 0.0
        )
        # In 2-D, kappa = div_s(n) = t . dn/ds along the unit tangent t.  A
        # weighted one-dimensional regression of the tangential normal turn on
        # the tangential offset of neighboring PLIC centers measures exactly
        # that.  The trace of a full 2-D Jacobian fit would add the normal
        # derivative n . (dn/dx) n, which interface points cannot observe
        # (an exact circle gives 2/R) and which is ill-conditioned because
        # neighboring interface centers are nearly collinear.
        tangent = jnp.stack((-normals[:, 1], normals[:, 0]), axis=-1)
        arc = ein.contract("csd,cd->cs", offsets, tangent)
        turn = ein.contract("csd,cd->cs", differences, tangent)
        spread = jnp.sum(weights * arc * arc, axis=1)
        safe_spread = jnp.maximum(spread, jnp.finfo(dtype).tiny)
        fitted = jnp.sum(weights * arc * turn, axis=1) / safe_spread
        error = turn - fitted[:, None] * arc
        residual = jnp.sqrt(
            jnp.sum(weights * error * error, axis=1)
            / jnp.maximum(jnp.sum(weights, axis=1), jnp.finfo(dtype).tiny)
        )
        neighbor_count = jnp.sum(usable, axis=1)
        # weights * distance**2 is one per usable neighbor, so the condition
        # is the inverse mean squared cosine between offsets and the tangent.
        condition = neighbor_count.astype(dtype) / safe_spread
        constant_normal = (neighbor_count >= 1) & (
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

        rank_valid = (neighbor_count >= self.discretization.cell_dimension) & (
            condition <= self.condition_limit
        )
        finite = jnp.isfinite(fitted) & jnp.isfinite(residual)
        valid = fit_active & (rank_valid | constant_normal) & finite & (not mismatch)
        kappa = jnp.where(constant_normal, 0.0, fitted)
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
            interface_delta=jnp.where(
                valid,
                measure / self.discretization.cell_volumes.astype(dtype),
                0.0,
            ),
            interface_delta_supported=valid,
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
        volume_fraction: ArrayLike,
        velocity: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, CurvatureEvidence]:
        alpha = self._validate_volume_fraction(volume_fraction)
        evidence = self.curvature(plic, alpha)
        dtype = jnp.result_type(alpha, self.discretization.area_vectors)
        speed = jnp.asarray(velocity, dtype=dtype)
        expected = (self.discretization.cell_count, self.discretization.cell_dimension)
        if speed.shape != expected:
            raise ValueError(f"Velocity must have shape {expected}.")
        owner = self.discretization.owner_cells
        neighbor = self.discretization.neighbor_cells
        safe_neighbor = jnp.maximum(neighbor, 0)
        interior = neighbor >= 0
        # A physical boundary face sees alpha_b = alpha_owner: no jump, no
        # force, balanced by a wall/extrapolated pressure p_b = p_owner.
        jump = jnp.where(interior, alpha[safe_neighbor] - alpha[owner], 0.0).astype(dtype)
        sigma = self.policy.surface_tension
        if sigma == 0.0:
            zero = jnp.zeros_like(jump)
            return (
                jnp.zeros((jump.size, expected[1]), dtype=dtype),
                zero,
                zero,
                zero,
                jnp.zeros((), dtype=jnp.int32),
                evidence,
            )
        usable = evidence.valid_mask
        weight = usable.astype(dtype)
        kappa = jnp.where(usable, evidence.curvature, 0.0).astype(dtype)
        count = weight[owner] + jnp.where(interior, weight[safe_neighbor], 0.0)
        total = kappa[owner] + jnp.where(interior, kappa[safe_neighbor], 0.0)
        supported = count > 0.0
        potential = sigma * total / jnp.where(supported, count, 1.0)
        # Jumps no larger than sqrt(eps) are rounding residue: they take the
        # supported potential when one exists and otherwise none.
        unsupported = jnp.sum(
            ~supported & (jnp.abs(jump) > jnp.sqrt(jnp.finfo(dtype).eps)),
            dtype=jnp.int32,
        )
        force = (potential * jump)[:, None] * self.discretization.area_vectors.astype(
            dtype
        )
        owner_work = 0.5 * ein.contract("fd,fd->f", force, speed[owner])
        neighbor_work = jnp.where(
            interior, 0.5 * ein.contract("fd,fd->f", force, speed[safe_neighbor]), 0.0
        )
        return force, owner_work, neighbor_work, potential, unsupported, evidence

    def _check_evidence(
        self,
        value: Array,
        evidence: CurvatureEvidence,
        /,
    ) -> Array:
        if self.policy.surface_tension == 0.0:
            return value
        return eqx.error_if(
            value,
            jnp.any(
                evidence.interface_active
                & (evidence.status != int(CurvatureStatus.VALID))
            ),
            "Capillary curvature evidence is not valid.",
        )

    def face_rate_block(
        self,
        plic: Any,
        volume_fraction: ArrayLike,
        velocity: ArrayLike,
        *,
        block_id: str = "capillary",
    ) -> CapillaryFaceRateBlock:
        """Return the balanced capillary face forces and their cell work.

        Refuses active interface cells without valid curvature and interior
        faces whose alpha jump exceeds ``sqrt(eps)`` without a valid adjacent
        curvature, unless ``surface_tension == 0`` (the exact disabling path).
        """

        force, owner_work, neighbor_work, potential, unsupported, evidence = (
            self._face_arrays(plic, volume_fraction, velocity)
        )
        force = self._check_evidence(force, evidence)
        force = eqx.error_if(
            force,
            unsupported > 0,
            "Capillary alpha jump lacks valid adjacent curvature.",
        )
        return CapillaryFaceRateBlock(
            force,
            owner_work,
            neighbor_work,
            potential,
            self.discretization.owner_cells,
            self.discretization.neighbor_cells,
            surface_tension=self.policy.surface_tension,
            geometry_id=self.discretization.prepared_id,
            evidence_id=evidence.evidence_id,
            block_id=block_id,
        )

    def laplace_pressure_jump(
        self,
        plic: Any,
        volume_fraction: ArrayLike,
    ) -> Array:
        """Return the signed ``sigma * kappa`` jump (alpha phase minus complement)."""

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
            active = jnp.asarray(interface_active, dtype=jnp.bool_)
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


def _mac_face_neighbors(
    value: Array, axis: int, periodic: bool, fill: Array, /
) -> tuple[Array, Array]:
    """Return lower and upper cell values of every MAC face normal to ``axis``.

    Periodic face ``i`` joins cells ``i - 1`` and ``i``; nonperiodic axes add
    both physical boundary faces, whose missing exterior cell takes ``fill``.
    """

    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        lower = jnp.roll(moved, 1, axis=0)
        upper = moved
    else:
        ghost = jnp.broadcast_to(fill, (1,) + moved.shape[1:]).astype(moved.dtype)
        lower = jnp.concatenate((ghost, moved), axis=0)
        upper = jnp.concatenate((moved, ghost), axis=0)
    return jnp.moveaxis(lower, 0, axis), jnp.moveaxis(upper, 0, axis)


def _unsupported_faces(jumps: FaceVelocity, supports: FaceVelocity, /) -> Array:
    """Count faces carrying an alpha jump without a supported potential."""

    return sum(
        (
            jnp.sum(jump & ~support, dtype=jnp.int32)
            for jump, support in zip(jumps, supports, strict=True)
        ),
        start=jnp.asarray(0, dtype=jnp.int32),
    )


class MACCapillaryForceResult(StrictModule):
    """Balanced interfacial-potential force on structured MAC faces.

    ``face_force[axis]`` is the force density ``phi_f (G alpha)_f`` on the
    faces normal to ``axis``; the momentum rate is its product with the face
    dual measure.  ``face_potential`` is the total face potential (capillary
    plus any supplied body potential), ``capillary_face_potential`` its
    ``sigma * kappa_f`` part.  ``pressure_jump`` is the mean usable
    ``sigma * kappa`` (alpha phase minus complement).  A face carrying an
    alpha jump without usable adjacent curvature, or an interface cell whose
    curvature status is refused, makes the result invalid.
    """

    face_force: FaceVelocity
    face_potential: FaceVelocity
    capillary_face_potential: FaceVelocity
    marangoni_face_force: FaceVelocity
    pressure_jump: Array
    surface_tension_minimum: Array
    surface_tension_maximum: Array
    unsupported_face_count: Array
    refused_cell_count: Array
    finite: Array
    valid: Array
    operator_id: str = eqx.field(static=True)
    curvature_evidence_id: str = eqx.field(static=True)


class MACBalancedCapillaryOperator(StrictModule):
    """Balanced normal capillary and tangential Marangoni action on MAC faces.

    The normal force on each face is ``phi_f (G alpha)_f`` with ``G`` the
    projection gradient and ``phi_f`` the usable-neighbor average of
    ``sigma kappa``. A variable policy additionally contributes
    ``(grad_s sigma)_d delta_f`` on each face family, where the scalar
    ``delta_f`` is the supported-neighbor average of the canonical
    cell-centered interface delta. Its tangential projection uses the caller's
    PLIC normal, so normal and tangential actions share one interface geometry.
    A spatially constant normal potential is exactly the projection gradient
    of ``phi alpha`` and preserves the balanced static jump. Constant
    ``surface_tension == 0`` remains an exact disabling path.
    Alpha jumps no larger than ``jump_tolerance`` are rounding-level residue:
    they receive a supported potential when one exists and otherwise none,
    and are not counted as unsupported.
    """

    operators: PreparedMACOperators
    policy: SurfaceTensionPolicy | VariableSurfaceTensionPolicy
    jump_tolerance: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        operators: PreparedMACOperators,
        policy: SurfaceTensionPolicy | VariableSurfaceTensionPolicy,
        /,
        *,
        jump_tolerance: float | None = None,
    ) -> None:
        if not isinstance(policy, SurfaceTensionPolicy | VariableSurfaceTensionPolicy):
            raise TypeError(
                "policy must be SurfaceTensionPolicy or VariableSurfaceTensionPolicy."
            )
        epsilon = float(np.finfo(np.dtype(operators.pressure_space.dtype)).eps)
        tolerance = (
            float(np.sqrt(epsilon)) if jump_tolerance is None else float(jump_tolerance)
        )
        if not np.isfinite(tolerance) or not 0.0 <= tolerance < 0.5:
            raise ValueError("jump_tolerance must lie in [0, 1/2).")
        self.operators = operators
        self.policy = policy
        self.jump_tolerance = tolerance
        self.operator_id = canonical_fingerprint(
            {
                "kind": "mac-balanced-capillary-operator",
                "operators": operators.prepared_id,
                "policy": policy.policy_id,
                "jump_tolerance": tolerance,
                "face_gradient": "mac-projection-gradient",
                "face_potential": "usable-neighbor-mean",
                "marangoni_delta": "conservative-supported-cell-delta-face-average",
            }
        )

    def _validate_curvature(self, curvature: CurvatureEvidence | None, /) -> None:
        if curvature is None:
            if (
                isinstance(self.policy, VariableSurfaceTensionPolicy)
                or self.policy.surface_tension != 0.0
            ):
                raise ValueError("Active surface tension requires curvature evidence.")
            return
        if not isinstance(curvature, CurvatureEvidence):
            raise TypeError("curvature must be CurvatureEvidence or None.")
        if curvature.curvature.shape != self.operators.discretization.cell_shape:
            raise ValueError("MAC curvature evidence must match the cell shape.")

    def face_average(
        self, cell_value: ArrayLike, supported: ArrayLike, /
    ) -> tuple[FaceVelocity, FaceVelocity]:
        """Average a cell potential over the supported cells adjacent to faces.

        Returns the face values and the per-face support mask (at least one
        supported adjacent cell).  Unsupported cells never contribute, so a
        potential defined only near the interface is not diluted by zeros.
        """

        discretization = self.operators.discretization
        mask = jnp.asarray(supported, dtype=jnp.bool_)
        value = jnp.where(mask, self.operators.validate_pressure(cell_value), 0.0)
        if mask.shape != discretization.cell_shape:
            raise ValueError("Potential support must match the MAC cell shape.")
        averages = []
        supports = []
        for axis, grid_axis in enumerate(discretization.grid.structured_axes):
            lower, upper = _mac_face_neighbors(
                value, axis, grid_axis.periodic, jnp.zeros((), dtype=value.dtype)
            )
            lower_mask, upper_mask = _mac_face_neighbors(
                mask, axis, grid_axis.periodic, jnp.zeros((), dtype=jnp.bool_)
            )
            count = lower_mask.astype(value.dtype) + upper_mask.astype(value.dtype)
            averages.append((lower + upper) / jnp.where(count > 0.0, count, 1.0))
            supports.append(count > 0.0)
        return tuple(averages), tuple(supports)

    def _face_interface_delta(
        self,
        cell_delta: Array,
        supported: Array,
        required: Array,
        /,
    ) -> tuple[FaceVelocity, FaceVelocity]:
        """Interpolate a zero-extended cell delta conservatively to MAC faces."""

        discretization = self.operators.discretization
        mask = jnp.asarray(supported, dtype=jnp.bool_)
        required_mask = jnp.asarray(required, dtype=jnp.bool_)
        if (
            cell_delta.shape != discretization.cell_shape
            or mask.shape != cell_delta.shape
            or required_mask.shape != cell_delta.shape
        ):
            raise ValueError("Interface-delta evidence must match the MAC cell shape.")
        value = jnp.where(mask, cell_delta, 0.0)
        faces = []
        supports = []
        cell_present = jnp.ones(discretization.cell_shape, dtype=jnp.bool_)
        for axis, grid_axis in enumerate(discretization.grid.structured_axes):
            lower, upper = _mac_face_neighbors(
                value, axis, grid_axis.periodic, jnp.zeros((), dtype=value.dtype)
            )
            lower_present, upper_present = _mac_face_neighbors(
                cell_present,
                axis,
                grid_axis.periodic,
                jnp.zeros((), dtype=jnp.bool_),
            )
            lower_supported, upper_supported = _mac_face_neighbors(
                mask, axis, grid_axis.periodic, jnp.zeros((), dtype=jnp.bool_)
            )
            lower_missing, upper_missing = _mac_face_neighbors(
                required_mask & ~mask,
                axis,
                grid_axis.periodic,
                jnp.zeros((), dtype=jnp.bool_),
            )
            count = lower_present.astype(value.dtype) + upper_present.astype(value.dtype)
            faces.append((lower + upper) / count)
            supports.append(
                (lower_supported | upper_supported) & ~(lower_missing | upper_missing)
            )
        return tuple(faces), tuple(supports)

    def evaluate(
        self,
        alpha: ArrayLike,
        curvature: CurvatureEvidence | None,
        /,
        *,
        variable_surface_tension: SurfaceTensionEvaluation | None = None,
        body_potential: ArrayLike | None = None,
        body_support: ArrayLike | None = None,
    ) -> MACCapillaryForceResult:
        """Return balanced normal capillary, tangential Marangoni and body forces."""

        alpha_ = self.operators.validate_pressure(alpha)
        self._validate_curvature(curvature)
        variable = isinstance(self.policy, VariableSurfaceTensionPolicy)
        if variable != (variable_surface_tension is not None):
            raise ValueError(
                "variable_surface_tension must be supplied exactly for a variable policy."
            )
        if (body_potential is None) != (body_support is None):
            raise ValueError("body_potential and body_support must be given together.")
        alpha_gradient = self.operators.gradient(alpha_)
        jumps = []
        for axis, grid_axis in enumerate(
            self.operators.discretization.grid.structured_axes
        ):
            lower, upper = _mac_face_neighbors(
                alpha_, axis, grid_axis.periodic, jnp.zeros((), dtype=alpha_.dtype)
            )
            jumps.append(
                (jnp.abs(upper - lower) > self.jump_tolerance)
                & (alpha_gradient[axis] != 0.0)
            )
        zero = tuple(jnp.zeros_like(value) for value in alpha_gradient)
        unsupported = jnp.asarray(0, dtype=jnp.int32)
        sigma_minimum = jnp.zeros((), dtype=alpha_.dtype)
        sigma_maximum = jnp.zeros((), dtype=alpha_.dtype)
        if curvature is None:
            capillary = zero
            marangoni = zero
            refused = jnp.asarray(0, dtype=jnp.int32)
            pressure_jump = jnp.zeros((), dtype=alpha_.dtype)
        else:
            usable = curvature.usable_mask
            if variable_surface_tension is None:
                if not isinstance(self.policy, SurfaceTensionPolicy):
                    raise ValueError("Variable surface-tension evaluation is missing.")
                sigma = jnp.full_like(alpha_, self.policy.surface_tension)
                kappa, supports = self.face_average(curvature.curvature, usable)
                capillary = tuple(self.policy.surface_tension * value for value in kappa)
                marangoni = zero
            else:
                sigma = jnp.asarray(
                    variable_surface_tension.surface_tension, dtype=alpha_.dtype
                )
                marangoni_gradient = jnp.asarray(
                    variable_surface_tension.marangoni_gradient, dtype=alpha_.dtype
                )
                if (
                    sigma.shape != alpha_.shape
                    or marangoni_gradient.shape != alpha_.shape + (alpha_.ndim,)
                ):
                    raise ValueError(
                        "Variable surface tension must match the MAC cell geometry."
                    )
                capillary, supports = self.face_average(
                    sigma * curvature.curvature, usable
                )
                interface_cells = (
                    usable
                    & (alpha_ > self.jump_tolerance)
                    & (alpha_ < 1.0 - self.jump_tolerance)
                )
                delta, delta_support = self._face_interface_delta(
                    curvature.interface_delta,
                    curvature.interface_delta_supported & interface_cells,
                    interface_cells,
                )
                marangoni_values = []
                marangoni_required = []
                for axis in range(alpha_.ndim):
                    tangential, tangential_support = self.face_average(
                        marangoni_gradient[..., axis], interface_cells
                    )
                    component = tangential[axis]
                    marangoni_values.append(component * delta[axis])
                    marangoni_required.append(
                        tangential_support[axis] & (component != 0.0)
                    )
                marangoni = tuple(marangoni_values)
                unsupported = unsupported + _unsupported_faces(
                    tuple(marangoni_required), delta_support
                )
            unsupported = unsupported + _unsupported_faces(tuple(jumps), supports)
            refused = jnp.sum(curvature.interface_active & ~usable, dtype=jnp.int32)
            weight = usable.astype(alpha_.dtype)
            potential = sigma * curvature.curvature
            if variable_surface_tension is None:
                if not isinstance(self.policy, SurfaceTensionPolicy):
                    raise ValueError("Constant surface-tension policy is unavailable.")
                pressure_jump = (
                    self.policy.surface_tension
                    * jnp.sum(weight * jnp.where(usable, curvature.curvature, 0.0))
                    / jnp.maximum(jnp.sum(weight), 1.0)
                )
            else:
                pressure_jump = jnp.sum(
                    weight * jnp.where(usable, potential, 0.0)
                ) / jnp.maximum(jnp.sum(weight), 1.0)
            sigma_minimum = jnp.min(jnp.where(usable, sigma, jnp.inf))
            sigma_minimum = jnp.where(jnp.any(usable), sigma_minimum, 0.0)
            sigma_maximum = jnp.max(jnp.where(usable, sigma, 0.0))
        if body_potential is None or body_support is None:
            body = zero
        else:
            body, supports = self.face_average(body_potential, body_support)
            unsupported = unsupported + _unsupported_faces(tuple(jumps), supports)
        potential = tuple(
            left + right for left, right in zip(capillary, body, strict=True)
        )
        force = (
            tuple(
                value * gradient
                for value, gradient in zip(potential, alpha_gradient, strict=True)
            )
            if variable_surface_tension is None
            else tuple(
                value * gradient + tangent
                for value, gradient, tangent in zip(
                    potential, alpha_gradient, marangoni, strict=True
                )
            )
        )
        finite = (
            jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in force)))
            & jnp.isfinite(pressure_jump)
            & jnp.isfinite(sigma_minimum)
            & jnp.isfinite(sigma_maximum)
        )
        return MACCapillaryForceResult(
            face_force=force,
            face_potential=potential,
            capillary_face_potential=capillary,
            marangoni_face_force=marangoni,
            pressure_jump=pressure_jump,
            surface_tension_minimum=sigma_minimum,
            surface_tension_maximum=sigma_maximum,
            unsupported_face_count=unsupported,
            refused_cell_count=refused,
            finite=finite,
            valid=finite & (unsupported == 0) & (refused == 0),
            operator_id=self.operator_id,
            curvature_evidence_id="none" if curvature is None else curvature.evidence_id,
        )

    def capillary_step(
        self,
        density: ArrayLike,
        /,
        *,
        interface_active: ArrayLike | None = None,
        surface_tension: ArrayLike | None = None,
    ) -> Array:
        """Return the capillary CFL step using local or constant surface tension."""

        discretization = self.operators.discretization
        rho = jnp.asarray(density)
        if rho.shape != discretization.cell_shape:
            raise ValueError("MAC capillary density must match the cell shape.")
        rho = eqx.error_if(
            rho,
            jnp.any(~jnp.isfinite(rho) | (rho < self.policy.density_floor)),
            "Capillary density is below the positive policy floor.",
        )
        active = (
            jnp.ones(discretization.cell_shape, dtype=jnp.bool_)
            if interface_active is None
            else jnp.asarray(interface_active, dtype=jnp.bool_)
        )
        if active.shape != discretization.cell_shape:
            raise ValueError("interface_active must match the MAC cell shape.")
        if isinstance(self.policy, SurfaceTensionPolicy):
            if surface_tension is not None:
                raise ValueError("surface_tension is only valid for a variable policy.")
            sigma = jnp.full_like(rho, self.policy.surface_tension)
        else:
            if surface_tension is None:
                raise ValueError("A variable policy requires local surface_tension.")
            sigma = jnp.asarray(surface_tension, dtype=rho.dtype)
            if sigma.shape != rho.shape:
                raise ValueError("surface_tension must match the MAC cell shape.")
        sigma = eqx.error_if(
            sigma,
            jnp.any(~jnp.isfinite(sigma) | (sigma < 0.0)),
            "Capillary surface tension must be finite and nonnegative.",
        )
        infinite = jnp.asarray(jnp.inf, dtype=rho.dtype)
        width = jnp.full(discretization.cell_shape, jnp.inf, dtype=rho.dtype)
        for axis, grid_axis in enumerate(discretization.grid.structured_axes):
            shape = [1] * len(discretization.cell_shape)
            shape[axis] = discretization.cell_shape[axis]
            width = jnp.minimum(
                width, grid_axis.interval_widths.astype(rho.dtype).reshape(shape)
            )
        positive = active & (sigma > 0.0)
        step = jnp.sqrt(rho * width**3 / jnp.where(sigma > 0.0, sigma, 1.0))
        return self.policy.capillary_cfl * jnp.min(jnp.where(positive, step, infinite))


__all__ = [
    "BalancedCapillaryOperator",
    "CapillaryFaceRateBlock",
    "CurvatureEvidence",
    "CurvatureGeometryError",
    "CurvatureStatus",
    "CurvatureUncertaintyError",
    "LinearSurfaceTensionLaw",
    "MACBalancedCapillaryOperator",
    "MACCapillaryForceResult",
    "SurfaceTensionPolicy",
    "SurfaceTensionEvaluation",
    "VariableSurfaceTensionPolicy",
]
