#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Farina et al. 2004 concentric muscle/fat/skin surface conductor.

Independent implementation of Eqs. 7, 11, 13--17 and 32. Infinite air is
insulating. A fixed axial Fourier period is a numerical identity, not a
finite-limb end boundary. Finite rectangular contacts perform passive aperture
averaging; they are not complete-electrode impedance or implanted contacts.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ....linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    RHSLayout,
    solve,
)
from ....special import ive, kve
from ._fiber_current import FiberCurrentState, PreparedFiberCurrent


FARINA_2004_CYLINDRICAL_DOI = "10.1109/TBME.2003.820998"


def _i_basis(order: Array, q: Array, radius: Array, reference: Array):
    x, xref = q * radius, q * reference
    scale = jnp.exp(x - xref) / ive(order, xref)
    value = ive(order, x) * scale
    derivative = 0.5 * q * (ive(jnp.abs(order - 1), x) + ive(order + 1, x)) * scale
    return value, derivative


def _k_basis(order: Array, q: Array, radius: Array, reference: Array):
    x, xref = q * radius, q * reference
    scale = jnp.exp(xref - x) / kve(order, xref)
    value = kve(order, x) * scale
    derivative = -0.5 * q * (kve(jnp.abs(order - 1), x) + kve(order + 1, x)) * scale
    return value, derivative


class CylindricalConductorState(StrictModule, NonTrainableState):
    time_ms: Array
    accepted_observations: Array
    contact_potential_V: Array
    lead_voltage_V: Array
    prepared_id: str = eqx.field(static=True)


class CylindricalConductorEvidence(StrictModule, NonTrainableState):
    identity_matches: Array
    source_neutral: Array
    finite: Array
    time_admissible: Array
    interface_relative_residual: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class CylindricalConductorCandidate(StrictModule, NonTrainableState):
    source_state: CylindricalConductorState
    candidate_state: CylindricalConductorState
    current_source: FiberCurrentState
    evidence: CylindricalConductorEvidence

    def commit(
        self,
        current: CylindricalConductorState,
        source: FiberCurrentState,
        /,
    ) -> CylindricalConductorState:
        """Commit observation only, requiring both accepted snapshots unchanged."""
        prior, proposed, expected = (
            self.source_state,
            self.candidate_state,
            self.current_source,
        )
        if (
            current.prepared_id != prior.prepared_id
            or source.prepared_id != expected.prepared_id
            or source.geometry_id != expected.geometry_id
        ):
            return current
        same = (
            (current.time_ms == prior.time_ms)
            & (current.accepted_observations == prior.accepted_observations)
            & jnp.all(current.contact_potential_V == prior.contact_potential_V)
            & jnp.all(current.lead_voltage_V == prior.lead_voltage_V)
            & (source.time_ms == expected.time_ms)
            & (source.accepted_observations == expected.accepted_observations)
            & jnp.all(source.transmembrane_current_A == expected.transmembrane_current_A)
        )
        accepted = self.evidence.successful & same
        return CylindricalConductorState(
            jnp.where(accepted, proposed.time_ms, current.time_ms),
            jnp.where(
                accepted, proposed.accepted_observations, current.accepted_observations
            ),
            jnp.where(
                accepted, proposed.contact_potential_V, current.contact_potential_V
            ),
            jnp.where(accepted, proposed.lead_voltage_V, current.lead_voltage_V),
            current.prepared_id,
        )


class Farina2004CylindricalConductorPlan(StrictModule):
    """Fixed straight-fiber concentric cylinder, aligned to the supplied +z frame.

    Required radii are the muscle/fat/skin outer radii. Conductivities are
    (muscle transverse, muscle longitudinal, fat isotropic, skin isotropic).
    Electrode centers are (azimuth radians, axial metres); contact sizes are
    (circumferential arc length metres, axial length metres). All are explicit.
    """

    layer_radii_m: Array
    conductivity_S_per_m: Array
    electrode_centers: Array
    contact_sizes_m: Array
    lead_weights: Array
    electrode_ids: tuple[str, ...] = eqx.field(static=True)
    lead_ids: tuple[str, ...] = eqx.field(static=True)
    axial_period_m: float = eqx.field(static=True)
    longitudinal_modes: int = eqx.field(static=True)
    angular_modes: int = eqx.field(static=True)
    coordinate_frame_id: str = eqx.field(static=True)
    material_source_id: str = eqx.field(static=True)
    electrode_source_id: str = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layer_radii_m: ArrayLike,
        conductivity_S_per_m: ArrayLike,
        electrode_centers: ArrayLike,
        contact_sizes_m: ArrayLike,
        lead_weights: ArrayLike,
        electrode_ids: tuple[str, ...],
        lead_ids: tuple[str, ...],
        /,
        *,
        axial_period_m: float,
        longitudinal_modes: int,
        angular_modes: int,
        coordinate_frame_id: str,
        material_source_id: str,
        electrode_source_id: str,
        residual_tolerance: float = 1.0e-7,
    ):
        radii = jnp.asarray(layer_radii_m, dtype=float)
        sigma = jnp.asarray(conductivity_S_per_m, dtype=radii.dtype)
        centers = jnp.asarray(electrode_centers, dtype=radii.dtype)
        sizes = jnp.asarray(contact_sizes_m, dtype=radii.dtype)
        weights = jnp.asarray(lead_weights, dtype=radii.dtype)
        for ids in (electrode_ids, lead_ids):
            if (
                not ids
                or any(not value.strip() for value in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("Electrode and lead IDs must be nonempty and unique.")
        if radii.shape != (3,) or sigma.shape != (4,):
            raise ValueError("Require three tissue radii and four source conductivities.")
        if centers.shape != (len(electrode_ids), 2) or sizes.shape != centers.shape:
            raise ValueError(
                "Electrode centers and finite contact sizes require shape (electrode, 2)."
            )
        if weights.shape != (len(lead_ids), len(electrode_ids)):
            raise ValueError("lead_weights must have shape (lead, electrode).")
        arrays = (radii, sigma, centers, sizes, weights)
        if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
            raise ValueError("All conductor inputs must be finite.")
        if not (
            np.all(np.asarray(radii) > 0)
            and np.all(np.diff(np.asarray(radii)) > 0)
            and np.all(np.asarray(sigma) > 0)
            and np.all(np.asarray(sizes) > 0)
        ):
            raise ValueError(
                "Radii must be strictly increasing; materials and apertures positive."
            )
        if not np.isfinite(axial_period_m) or axial_period_m <= 0:
            raise ValueError("The numerical axial period must be positive and finite.")
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (longitudinal_modes, angular_modes)
        ):
            raise TypeError("Spectral capacities must be integers.")
        if longitudinal_modes < 1 or angular_modes < 0:
            raise ValueError(
                "Require positive longitudinal and nonnegative angular truncation."
            )
        if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
            raise ValueError("Residual tolerance must be positive and finite.")
        if np.any(
            np.asarray(sizes)[:, 0] >= 2 * np.pi * float(np.asarray(radii[-1]))
        ) or np.any(np.asarray(sizes)[:, 1] >= axial_period_m):
            raise ValueError(
                "Contacts must not wrap around the circumference or axial period."
            )
        neutrality = np.sum(np.asarray(weights), axis=-1)
        if np.any(np.abs(neutrality) > 1.0e-12) or np.any(
            np.sum(np.abs(np.asarray(weights)), axis=-1) == 0
        ):
            raise ValueError(
                "Each passive voltage lead must be nonzero and exactly gauge-neutral."
            )
        if not all(
            v.strip()
            for v in (coordinate_frame_id, material_source_id, electrode_source_id)
        ):
            raise ValueError(
                "Explicit coordinate, conductivity and electrode provenance are required."
            )
        self.layer_radii_m, self.conductivity_S_per_m = radii, sigma
        self.electrode_centers, self.contact_sizes_m, self.lead_weights = (
            centers,
            sizes,
            weights,
        )
        self.electrode_ids, self.lead_ids = tuple(electrode_ids), tuple(lead_ids)
        self.axial_period_m = float(axial_period_m)
        self.longitudinal_modes, self.angular_modes = longitudinal_modes, angular_modes
        self.coordinate_frame_id = coordinate_frame_id.strip()
        self.material_source_id, self.electrode_source_id = (
            material_source_id.strip(),
            electrode_source_id.strip(),
        )
        self.residual_tolerance = float(residual_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "farina-2004-muscle-fat-skin-cylinder-rectangular-aperture",
                "doi": FARINA_2004_CYLINDRICAL_DOI,
                "source_pdf_sha256": "53ea990abcb221e33459558cd7b8ed431f6a013caf715a08ae5f8a00289af0be",
                "arrays": array_tree_fingerprint(arrays),
                "electrodes": self.electrode_ids,
                "leads": self.lead_ids,
                "period_m": self.axial_period_m.hex(),
                "longitudinal_modes": longitudinal_modes,
                "angular_modes": angular_modes,
                "frame": self.coordinate_frame_id,
                "materials": self.material_source_id,
                "electrode_source": self.electrode_source_id,
                "gauge": "zero axial-average potential; source neutral separately on every straight fiber",
                "boundary": "outer skin insulating; regular cylinder axis; periodic axial continuation",
                "numerics": "scaled Bessel layer basis; row equilibrated native LU",
                "residual_tolerance": self.residual_tolerance.hex(),
            }
        )

    def prepare(
        self,
        source: PreparedFiberCurrent,
        /,
        *,
        coordinate_frame_id: str,
    ) -> PreparedFarina2004CylindricalConductor:
        return PreparedFarina2004CylindricalConductor(self, source, coordinate_frame_id)


class PreparedFarina2004CylindricalConductor(StrictModule):
    plan: Farina2004CylindricalConductorPlan
    contact_lead_field_ohm: Array
    radial_transfer_ohm_m: Array
    interface_relative_residual: Array
    source_current_shape: tuple[int, int] = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    absolute_neutrality_tolerance_A: float = eqx.field(static=True)
    relative_neutrality_tolerance: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: Farina2004CylindricalConductorPlan,
        source: PreparedFiberCurrent,
        coordinate_frame_id: str,
        /,
    ):
        if coordinate_frame_id != plan.coordinate_frame_id:
            raise ValueError(
                "The committed source and cylinder must use the same registered frame."
            )
        positions = source.plan.positions_m
        host = np.asarray(positions)
        if not np.allclose(host[..., :2], host[:, :1, :2], rtol=0, atol=1.0e-12):
            raise ValueError(
                "This source fidelity requires straight fibers parallel to the cylinder axis."
            )
        axial_edges = np.diff(host[..., 2], axis=1)
        monotone = np.all(axial_edges > 0, axis=1) | np.all(axial_edges < 0, axis=1)
        if not np.all(monotone):
            raise ValueError(
                "Cylinder fibers must have monotone axial coordinates without foldback."
            )
        radius = jnp.sqrt(jnp.sum(positions[:, 0, :2] ** 2, axis=-1))
        if np.any(
            np.asarray(radius) + np.asarray(source.plan.radius_m)
            >= np.asarray(plan.layer_radii_m[0])
        ):
            raise ValueError("Every entire source fiber must lie strictly inside muscle.")
        if np.any(np.ptp(host[..., 2], axis=1) >= plan.axial_period_m):
            raise ValueError(
                "Fiber length must be shorter than the declared axial Fourier period."
            )
        b, c, d = plan.layer_radii_m
        sm, sl, sf, ss = plan.conductivity_S_per_m
        # Eq.21 scaling and Fourier symmetry: factor only one spectral quadrant.
        # kz=0 vanishes separately for every neutral straight fiber.
        n = jnp.arange(plan.angular_modes + 1)[:, None]
        k = (
            2 * jnp.pi / plan.axial_period_m * jnp.arange(1, plan.longitudinal_modes + 1)
        )[None, :]
        order = n
        q = k
        qm = q * jnp.sqrt(sl / sm)
        im_b, dim_b = _i_basis(order, qm, b, b)
        if_b, dif_b = _i_basis(order, q, b, c)
        kf_b, dkf_b = _k_basis(order, q, b, b)
        if_c, dif_c = _i_basis(order, q, c, c)
        kf_c, dkf_c = _k_basis(order, q, c, b)
        is_c, dis_c = _i_basis(order, q, c, d)
        ks_c, dks_c = _k_basis(order, q, c, c)
        is_d, dis_d = _i_basis(order, q, d, d)
        ks_d, dks_d = _k_basis(order, q, d, c)
        zero = jnp.zeros_like(im_b)
        matrix = jnp.stack(
            (
                jnp.stack((im_b, -if_b, -kf_b, zero, zero), axis=-1),
                jnp.stack((sm * dim_b, -sf * dif_b, -sf * dkf_b, zero, zero), axis=-1),
                jnp.stack((zero, if_c, kf_c, -is_c, -ks_c), axis=-1),
                jnp.stack(
                    (zero, sf * dif_c, sf * dkf_c, -ss * dis_c, -ss * dks_c), axis=-1
                ),
                jnp.stack((zero, zero, zero, ss * dis_d, ss * dks_d), axis=-1),
            ),
            axis=-2,
        )
        source_x = qm[None, ...] * radius[:, None, None]
        boundary_x = qm * b
        source_i = ive(order[None, ...], source_x)
        exponential = jnp.exp(source_x - boundary_x)
        particular = source_i * kve(order, boundary_x) * exponential / sm
        derivative = (
            -0.5
            * qm
            * source_i
            * (kve(jnp.abs(order - 1), boundary_x) + kve(order + 1, boundary_x))
            * exponential
            / sm
        )
        right = jnp.stack(
            (
                -particular,
                -sm * derivative,
                jnp.zeros_like(particular),
                jnp.zeros_like(particular),
                jnp.zeros_like(particular),
            ),
            axis=-1,
        )
        row_scale = jnp.max(jnp.abs(matrix), axis=-1)
        scaled = matrix / row_scale[..., None]
        rhs = right / row_scale
        result = solve(
            LinearSystem(DenseLinearOperator(scaled)),
            jnp.moveaxis(rhs, 0, -1),
            policy=LinearSolvePolicy(DenseLU()),
            rhs_layout=RHSLayout((radius.size,)),
        )
        # All fibers share one factorization per mode, with one RHS per fiber.
        coefficients = jnp.moveaxis(result.value, -1, 0)
        residual = contract("nkij,fnkj->fnki", scaled, coefficients) - rhs
        relative = jnp.max(jnp.abs(residual), axis=-1) / jnp.maximum(
            jnp.max(jnp.abs(rhs), axis=-1), jnp.finfo(rhs.dtype).tiny
        )
        if not bool(np.all(np.asarray(result.successful))) or not bool(
            np.all(np.asarray(relative) <= plan.residual_tolerance)
        ):
            raise ValueError(
                "Cylindrical interface/insulation solve failed its declared residual."
            )
        radial = coefficients[..., 3] * is_d + coefficients[..., 4] * ks_d
        # Eq.32: finite rectangular aperture, no hidden point electrode.
        aperture = jnp.sinc(
            n[None, ...] * plan.contact_sizes_m[:, 0, None, None] / (2 * jnp.pi * d)
        ) * jnp.sinc(k[None, ...] * plan.contact_sizes_m[:, 1, None, None] / (2 * jnp.pi))
        theta = jnp.arctan2(positions[:, 0, 1], positions[:, 0, 0])
        # Sum angular modes first; never allocate electrode*fiber*node*n*k.
        angular_phase = n[None, None, ...] * (
            plan.electrode_centers[:, 0, None, None, None] - theta[None, :, None, None]
        )
        angular = jnp.sum(
            jnp.cos(angular_phase)
            * radial[None, ...]
            * aperture[:, None, ...]
            * jnp.where(n == 0, 1.0, 2.0)[None, None, ...],
            axis=-2,
        )
        electrode_phase = plan.electrode_centers[:, 1, None] * k
        node_phase = positions[..., 2, None] * k
        contact_field = (
            contract(
                "efk,ek,fjk->efj", angular, jnp.cos(electrode_phase), jnp.cos(node_phase)
            )
            + contract(
                "efk,ek,fjk->efj", angular, jnp.sin(electrode_phase), jnp.sin(node_phase)
            )
        ) / (jnp.pi * plan.axial_period_m)
        if not np.all(np.isfinite(np.asarray(contact_field))):
            raise ValueError(
                "Cylindrical lead field is nonfinite; refine the declared spectral policy."
            )
        self.plan = plan
        self.contact_lead_field_ohm = contact_field
        self.radial_transfer_ohm_m = radial
        self.interface_relative_residual = jnp.max(relative)
        self.source_current_shape = source.control_length_m.shape
        self.source_prepared_id, self.geometry_id = (
            source.prepared_id,
            source.plan.geometry_id,
        )
        self.absolute_neutrality_tolerance_A = source.plan.absolute_neutrality_tolerance_A
        self.relative_neutrality_tolerance = source.plan.relative_neutrality_tolerance
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-farina-2004-concentric-cylinder",
                "plan": plan.plan_id,
                "source": source.prepared_id,
                "geometry": self.geometry_id,
                "lead_field": array_tree_fingerprint(contact_field),
            }
        )

    def initialize(self, /) -> CylindricalConductorState:
        dtype = self.contact_lead_field_ohm.dtype
        return CylindricalConductorState(
            jnp.asarray(-jnp.inf, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros((len(self.plan.electrode_ids),), dtype=dtype),
            jnp.zeros((len(self.plan.lead_ids),), dtype=dtype),
            self.prepared_id,
        )

    def propose(
        self,
        current: CylindricalConductorState,
        source: FiberCurrentState,
        /,
    ) -> CylindricalConductorCandidate:
        if source.transmembrane_current_A.shape != self.source_current_shape:
            raise ValueError("Current source has foreign fiber/node support.")
        if current.contact_potential_V.shape != (
            len(self.plan.electrode_ids),
        ) or current.lead_voltage_V.shape != (len(self.plan.lead_ids),):
            raise ValueError("Conductor state has foreign contact/lead topology.")
        amperes = source.transmembrane_current_A
        contact = contract("efj,fj->e", self.contact_lead_field_ohm, amperes)
        lead = contract("le,e->l", self.plan.lead_weights, contact)
        neutral = jnp.all(
            jnp.abs(jnp.sum(amperes, axis=-1))
            <= (
                self.absolute_neutrality_tolerance_A
                + self.relative_neutrality_tolerance * jnp.sum(jnp.abs(amperes), axis=-1)
            )
        )
        finite = (
            jnp.all(jnp.isfinite(contact))
            & jnp.all(jnp.isfinite(lead))
            & jnp.isfinite(source.time_ms)
        )
        identities = jnp.asarray(
            source.prepared_id == self.source_prepared_id
            and source.geometry_id == self.geometry_id
            and current.prepared_id == self.prepared_id
        )
        time_valid = (source.time_ms >= current.time_ms) & (
            source.accepted_observations > 0
        )
        successful = identities & neutral & finite & time_valid
        return CylindricalConductorCandidate(
            current,
            CylindricalConductorState(
                source.time_ms,
                current.accepted_observations + 1,
                contact,
                lead,
                self.prepared_id,
            ),
            source,
            CylindricalConductorEvidence(
                identities,
                neutral,
                finite,
                time_valid,
                self.interface_relative_residual,
                successful,
                self.prepared_id,
            ),
        )


__all__ = [
    "FARINA_2004_CYLINDRICAL_DOI",
    "Farina2004CylindricalConductorPlan",
    "PreparedFarina2004CylindricalConductor",
    "CylindricalConductorCandidate",
    "CylindricalConductorEvidence",
    "CylindricalConductorState",
]
