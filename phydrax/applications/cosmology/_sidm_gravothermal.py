#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...linalg import DenseLinearOperator, LinearSystem, solve
from ...qualification import ReferenceArtifactManifest


def gravothermal_calibration_payload(
    radial_faces: ArrayLike,
    gravitational_constant: float,
    cross_section_per_mass: float,
    conductivity_calibration: float,
    /,
    *,
    calibration_id: str,
) -> bytes:
    """Return the exact governed bytes for one gravothermal calibration table."""

    faces = np.asarray(radial_faces, dtype=np.float64)
    coupling = float(gravitational_constant)
    cross_section = float(cross_section_per_mass)
    calibration = float(conductivity_calibration)
    identity = str(calibration_id).strip()
    if (
        faces.ndim != 1
        or faces.size < 4
        or np.any(~np.isfinite(faces))
        or faces[0] != 0.0
        or np.any(np.diff(faces) <= 0.0)
        or not np.isfinite(coupling)
        or coupling <= 0.0
        or not np.isfinite(cross_section)
        or cross_section <= 0.0
        or not np.isfinite(calibration)
        or calibration <= 0.0
        or not identity
    ):
        raise ValueError("Gravothermal calibration payload inputs are invalid.")
    return canonical_json(
        {
            "kind": "isolated-spherical-gravothermal-sidm-calibration",
            "radial_faces": array_tree_fingerprint(faces),
            "gravitational_constant": coupling,
            "cross_section_per_mass": cross_section,
            "conductivity_calibration": calibration,
            "geometry": "isolated-spherical",
            "velocity_model": "isotropic",
            "collision_model": "elastic-single-species",
            "boundary_condition": "reflecting-center-zero-flux-outer",
            "calibration_id": identity,
        }
    ).encode("ascii")


class GravothermalSIDMState(StrictModule):
    """Isolated spherical shell state in physical radius and physical time."""

    mass_density: Array
    radial_faces: Array
    velocity_dispersion_squared: Array
    time: Array

    def __init__(
        self,
        mass_density: ArrayLike,
        velocity_dispersion_squared: ArrayLike,
        radial_faces: ArrayLike,
        time: ArrayLike,
        /,
    ):
        density = jnp.asarray(mass_density)
        faces = jnp.asarray(radial_faces, dtype=density.dtype)
        self.mass_density = density
        self.radial_faces = faces
        self.velocity_dispersion_squared = jnp.asarray(
            velocity_dispersion_squared, dtype=density.dtype
        )
        self.time = jnp.asarray(time, dtype=density.dtype).reshape(())


class GravothermalSIDMDiagnostics(StrictModule):
    radial_centers: Array
    shell_volumes: Array
    enclosed_mass: Array
    pressure: Array
    hydrostatic_residual: Array
    hydrostatic_relative_residual: Array
    initial_hydrostatic_valid: Array
    structural_converged: Array
    structural_iterations: Array
    structural_shell_mass_defect: Array
    structural_entropy_defect: Array
    structural_conservative: Array
    relaxation_time: Array
    mean_free_path: Array
    scale_height: Array
    conductivity: Array
    heat_flux_faces: Array
    luminosity_faces: Array
    thermal_energy_before: Array
    post_conduction_thermal_energy: Array
    thermal_energy_after: Array
    gravitational_energy_before: Array
    gravitational_energy_after: Array
    total_energy_before: Array
    total_energy_after: Array
    boundary_energy_transfer: Array
    energy_balance_defect: Array
    total_energy_defect: Array
    total_energy_valid: Array
    maximum_fractional_energy_change: Array
    regime_supported: Array
    finite: Array
    timestep_valid: Array
    positive: Array
    successful: Array


class GravothermalSIDMResult(StrictModule):
    candidate_state: GravothermalSIDMState
    accepted_state: GravothermalSIDMState
    diagnostics: GravothermalSIDMDiagnostics
    successful: Array


def _state_where(
    accepted: Array, candidate: GravothermalSIDMState, original: GravothermalSIDMState
) -> GravothermalSIDMState:
    return jax.tree.map(
        lambda new, old: jnp.where(accepted, new, old), candidate, original
    )


class GravothermalSIDMPlan(StrictModule, NonTrainableState):
    """Quasi-static isolated spherical gravothermal conduction closure.

    The support is an isolated spherical Lagrangian shell mass grid. Each
    conductive update is followed by an entropy-preserving quasi-static
    structural solve for shell radii, density, pressure, and dispersion under
    hydrostatic balance with a zero-pressure outer surface. The closure assumes
    one isotropic elastic species, a reflecting center, and zero conductive
    heat flux at the outer boundary. It makes no generic three-dimensional,
    anisotropic, cosmological, tidal, or open-boundary claim.
    """

    radial_faces: Array
    radial_centers: Array
    shell_volumes: Array
    gravitational_constant: float = eqx.field(static=True)
    cross_section_per_mass: float = eqx.field(static=True)
    conductivity_calibration: float = eqx.field(static=True)
    maximum_fractional_energy_change: float = eqx.field(static=True)
    total_energy_relative_tolerance: float = eqx.field(static=True)
    structural_residual_tolerance: float = eqx.field(static=True)
    maximum_structural_iterations: int = eqx.field(static=True)
    structural_relaxation: float = eqx.field(static=True)
    geometry: Literal["isolated-spherical"] = eqx.field(static=True)
    velocity_model: Literal["isotropic"] = eqx.field(static=True)
    collision_model: Literal["elastic-single-species"] = eqx.field(static=True)
    boundary_condition: Literal["reflecting-center-zero-flux-outer"] = eqx.field(
        static=True
    )
    calibration_id: str = eqx.field(static=True)
    calibration_manifest: ReferenceArtifactManifest
    calibration_artifact: ScientificArtifactEnvelope
    commercial_use: bool = eqx.field(static=True)
    redistribution: bool = eqx.field(static=True)
    training_use: bool = eqx.field(static=True)
    export: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radial_faces: ArrayLike,
        gravitational_constant: float,
        cross_section_per_mass: float,
        conductivity_calibration: float,
        /,
        *,
        maximum_fractional_energy_change: float = 0.1,
        structural_residual_tolerance: float = 1.0e-8,
        total_energy_relative_tolerance: float = 1.0e-4,
        maximum_structural_iterations: int = 32,
        structural_relaxation: float = 1.0,
        geometry: Literal["isolated-spherical"] = "isolated-spherical",
        velocity_model: Literal["isotropic"] = "isotropic",
        collision_model: Literal["elastic-single-species"] = "elastic-single-species",
        boundary_condition: Literal[
            "reflecting-center-zero-flux-outer"
        ] = "reflecting-center-zero-flux-outer",
        calibration_id: str,
        calibration_manifest: ReferenceArtifactManifest,
        calibration_artifact: ScientificArtifactEnvelope,
        commercial_use: bool,
        redistribution: bool,
        training_use: bool,
        export: bool,
    ):
        faces = np.asarray(radial_faces, dtype=np.float64)
        if (
            faces.ndim != 1
            or faces.size < 4
            or np.any(~np.isfinite(faces))
            or faces[0] != 0.0
            or np.any(np.diff(faces) <= 0.0)
        ):
            raise ValueError(
                "radial_faces must be increasing finite physical shells beginning at zero."
            )
        coupling = float(gravitational_constant)
        cross_section = float(cross_section_per_mass)
        calibration = float(conductivity_calibration)
        maximum_change = float(maximum_fractional_energy_change)
        structural_tolerance = float(structural_residual_tolerance)
        energy_tolerance = float(total_energy_relative_tolerance)
        if not np.isfinite(energy_tolerance) or energy_tolerance <= 0.0:
            raise ValueError(
                "total_energy_relative_tolerance must be finite and positive."
            )
        structural_iterations = int(maximum_structural_iterations)
        structural_relaxation_ = float(structural_relaxation)
        if not np.isfinite(coupling) or coupling <= 0.0:
            raise ValueError("gravitational_constant must be finite and positive.")
        if not np.isfinite(structural_tolerance) or structural_tolerance <= 0.0:
            raise ValueError("structural_residual_tolerance must be finite and positive.")
        if structural_iterations <= 0:
            raise ValueError("maximum_structural_iterations must be positive.")
        if (
            not np.isfinite(structural_relaxation_)
            or not 0.0 < structural_relaxation_ <= 1.0
        ):
            raise ValueError("structural_relaxation must lie in (0,1].")
        if not np.isfinite(cross_section) or cross_section <= 0.0:
            raise ValueError("cross_section_per_mass must be finite and positive.")
        if not np.isfinite(calibration) or calibration <= 0.0:
            raise ValueError("conductivity_calibration must be finite and positive.")
        if not np.isfinite(maximum_change) or not 0.0 < maximum_change < 1.0:
            raise ValueError("maximum_fractional_energy_change must lie in (0,1).")
        if geometry != "isolated-spherical":
            raise ValueError(
                "Gravothermal SIDM supports isolated-spherical geometry only."
            )
        if velocity_model != "isotropic":
            raise ValueError("Gravothermal SIDM supports isotropic dispersion only.")
        if collision_model != "elastic-single-species":
            raise ValueError(
                "Gravothermal SIDM supports elastic-single-species collisions only."
            )
        if boundary_condition != "reflecting-center-zero-flux-outer":
            raise ValueError(
                "Gravothermal SIDM requires reflecting-center-zero-flux-outer BCs."
            )
        calibration_identity = str(calibration_id).strip()
        if not calibration_identity:
            raise ValueError("calibration_id must be non-empty.")
        if not isinstance(calibration_manifest, ReferenceArtifactManifest):
            raise TypeError("calibration_manifest must be ReferenceArtifactManifest.")
        if not isinstance(calibration_artifact, ScientificArtifactEnvelope):
            raise TypeError("calibration_artifact must be ScientificArtifactEnvelope.")
        calibration_manifest.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        calibration_payload = gravothermal_calibration_payload(
            faces,
            coupling,
            cross_section,
            calibration,
            calibration_id=calibration_identity,
        )
        calibration_manifest.verify_bytes(calibration_payload)
        if (
            calibration_artifact.status != "complete"
            or calibration_artifact.artifact_kind != "gravothermal-sidm-calibration"
            or calibration_artifact.content_digest != calibration_manifest.checksum
            or calibration_artifact.license_id != calibration_manifest.license_id
            or calibration_manifest.manifest_id
            not in calibration_artifact.parent_artifact_ids
        ):
            raise ValueError(
                "Gravothermal calibration envelope and manifest identity, digest, license, or lineage disagree."
            )
        inner = faces[:-1]
        outer = faces[1:]
        volumes = (4.0 * np.pi / 3.0) * (outer**3 - inner**3)
        centers = 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)
        self.radial_faces = jax.lax.stop_gradient(jnp.asarray(faces))
        self.radial_centers = jax.lax.stop_gradient(jnp.asarray(centers))
        self.shell_volumes = jax.lax.stop_gradient(jnp.asarray(volumes))
        self.gravitational_constant = coupling
        self.cross_section_per_mass = cross_section
        self.conductivity_calibration = calibration
        self.maximum_fractional_energy_change = maximum_change
        self.total_energy_relative_tolerance = energy_tolerance
        self.structural_residual_tolerance = structural_tolerance
        self.maximum_structural_iterations = structural_iterations
        self.structural_relaxation = structural_relaxation_
        self.geometry = geometry
        self.velocity_model = velocity_model
        self.collision_model = collision_model
        self.boundary_condition = boundary_condition
        self.calibration_id = calibration_identity
        self.calibration_manifest = calibration_manifest
        self.calibration_artifact = calibration_artifact
        self.commercial_use = commercial_use
        self.redistribution = redistribution
        self.training_use = training_use
        self.export = export
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isolated-spherical-gravothermal-sidm",
                "radial_faces": faces,
                "gravitational_constant": coupling,
                "cross_section_per_mass": cross_section,
                "conductivity_calibration": calibration,
                "maximum_fractional_energy_change": maximum_change,
                "total_energy_relative_tolerance": energy_tolerance,
                "structural_residual_tolerance": structural_tolerance,
                "maximum_structural_iterations": structural_iterations,
                "structural_relaxation": structural_relaxation_,
                "geometry": geometry,
                "velocity_model": velocity_model,
                "collision_model": collision_model,
                "boundary_condition": boundary_condition,
                "calibration_id": calibration_identity,
                "calibration_manifest": calibration_manifest.manifest_id,
                "calibration_artifact": calibration_artifact.artifact_id,
                "requested_use": {
                    "commercial": commercial_use,
                    "redistribution": redistribution,
                    "training": training_use,
                    "export": export,
                },
            }
        )

    @property
    def shell_count(self) -> int:
        return self.radial_centers.size

    def _gravitational_binding_energy(self, faces: Array, shell_mass: Array, /) -> Array:
        inner = faces[:-1]
        outer = faces[1:]
        cubic_width = outer**3 - inner**3
        inner_mass = jnp.cumsum(shell_mass) - shell_mass
        reciprocal_radius_average = 1.5 * (outer**2 - inner**2) / cubic_width
        shell_integral = (outer**5 - inner**5) / 5.0 - 0.5 * inner**3 * (
            outer**2 - inner**2
        )
        self_energy_factor = 3.0 * shell_mass**2 * shell_integral / cubic_width**2
        coupling = jnp.asarray(self.gravitational_constant, dtype=faces.dtype)
        return -coupling * jnp.sum(
            inner_mass * shell_mass * reciprocal_radius_average + self_energy_factor
        )

    def _hydrostatic_from_faces(
        self, faces: Array, shell_mass: Array, dispersion_squared: Array, /
    ):
        inner = faces[:-1]
        outer = faces[1:]
        volumes = (4.0 * jnp.pi / 3.0) * (outer**3 - inner**3)
        centers = 0.75 * (outer**4 - inner**4) / (outer**3 - inner**3)
        density = shell_mass / volumes
        enclosed_fraction = (centers**3 - inner**3) / (outer**3 - inner**3)
        enclosed = jnp.cumsum(shell_mass) - shell_mass + enclosed_fraction * shell_mass
        pressure = density * dispersion_squared
        left_center = centers[:-2]
        center = centers[1:-1]
        right_center = centers[2:]
        left_weight = (center - right_center) / (
            (left_center - center) * (left_center - right_center)
        )
        center_weight = (2.0 * center - left_center - right_center) / (
            (center - left_center) * (center - right_center)
        )
        right_weight = (center - left_center) / (
            (right_center - left_center) * (right_center - center)
        )
        pressure_gradient = jnp.empty_like(pressure)
        pressure_gradient = pressure_gradient.at[1:-1].set(
            left_weight * pressure[:-2]
            + center_weight * pressure[1:-1]
            + right_weight * pressure[2:]
        )
        x0, x1, x2 = centers[0], centers[1], centers[2]
        pressure_gradient = pressure_gradient.at[0].set(
            (2.0 * x0 - x1 - x2) * pressure[0] / ((x0 - x1) * (x0 - x2))
            + (x0 - x2) * pressure[1] / ((x1 - x0) * (x1 - x2))
            + (x0 - x1) * pressure[2] / ((x2 - x0) * (x2 - x1))
        )
        x0, x1, x2 = centers[-3], centers[-2], centers[-1]
        pressure_gradient = pressure_gradient.at[-1].set(
            (x2 - x1) * pressure[-3] / ((x0 - x1) * (x0 - x2))
            + (x2 - x0) * pressure[-2] / ((x1 - x0) * (x1 - x2))
            + (2.0 * x2 - x0 - x1) * pressure[-1] / ((x2 - x0) * (x2 - x1))
        )
        gravity_force_density = (
            density
            * jnp.asarray(self.gravitational_constant, dtype=density.dtype)
            * enclosed
            / centers**2
        )
        residual = pressure_gradient + gravity_force_density
        residual_scale = jnp.maximum(
            jnp.abs(pressure_gradient) + jnp.abs(gravity_force_density),
            jnp.finfo(density.dtype).tiny,
        )
        relative = residual / residual_scale
        return (
            centers,
            volumes,
            density,
            enclosed,
            pressure,
            residual,
            relative,
        )

    def _structural_state(
        self, log_width: Array, shell_mass: Array, entropy_proxy: Array, /
    ):
        dtype = shell_mass.dtype
        faces = jnp.concatenate(
            (jnp.zeros((1,), dtype=dtype), jnp.cumsum(jnp.exp(log_width)))
        )
        inner = faces[:-1]
        outer = faces[1:]
        volumes = (4.0 * jnp.pi / 3.0) * (outer**3 - inner**3)
        density = shell_mass / volumes
        dispersion_squared = entropy_proxy * density ** (2.0 / 3.0)
        pressure = density * dispersion_squared
        mass_center = jnp.cumsum(shell_mass) - 0.5 * shell_mass
        mass_face = jnp.cumsum(shell_mass)[:-1]
        pressure_gradient = (pressure[1:] - pressure[:-1]) / (
            mass_center[1:] - mass_center[:-1]
        )
        gravity_gradient = (
            jnp.asarray(self.gravitational_constant, dtype=dtype)
            * mass_face
            / (4.0 * jnp.pi * faces[1:-1] ** 4)
        )
        interior_scale = jnp.maximum(
            jnp.abs(pressure_gradient) + jnp.abs(gravity_gradient),
            jnp.finfo(dtype).tiny,
        )
        interior_residual = (pressure_gradient + gravity_gradient) / interior_scale
        outer_pressure_gradient = -pressure[-1] / (0.5 * shell_mass[-1])
        outer_gravity_gradient = (
            jnp.asarray(self.gravitational_constant, dtype=dtype)
            * jnp.sum(shell_mass)
            / (4.0 * jnp.pi * faces[-1] ** 4)
        )
        outer_scale = jnp.maximum(
            jnp.abs(outer_pressure_gradient) + jnp.abs(outer_gravity_gradient),
            jnp.finfo(dtype).tiny,
        )
        outer_residual = (outer_pressure_gradient + outer_gravity_gradient) / outer_scale
        residual = jnp.concatenate((interior_residual, outer_residual.reshape((1,))))
        return faces, density, dispersion_squared, residual

    def _readjust_structure(
        self,
        shell_mass: Array,
        entropy_proxy: Array,
        initial_faces: Array,
        /,
    ):
        initial_log_width = jnp.log(jnp.diff(initial_faces))

        def residual(log_width):
            return self._structural_state(log_width, shell_mass, entropy_proxy)[-1]

        initial_residual = residual(initial_log_width)
        initial_converged = (
            jnp.max(jnp.abs(initial_residual)) <= self.structural_residual_tolerance
        )
        line_search_factors = 0.5 ** jnp.arange(14, dtype=shell_mass.dtype)

        def iteration(_, carry):
            log_width, converged, solver_valid, count = carry

            def attempt(current):
                current_residual = residual(current)
                jacobian = jax.jacfwd(residual)(current)
                solved = solve(
                    LinearSystem(DenseLinearOperator(jacobian)),
                    -current_residual,
                )
                proposals = (
                    current[None, :]
                    + self.structural_relaxation
                    * line_search_factors[:, None]
                    * solved.value[None, :]
                )
                proposal_residuals = jax.vmap(residual)(proposals)
                proposal_finite = jnp.all(jnp.isfinite(proposals), axis=1) & jnp.all(
                    jnp.isfinite(proposal_residuals), axis=1
                )
                proposal_norms = jnp.where(
                    proposal_finite,
                    jnp.max(jnp.abs(proposal_residuals), axis=1),
                    jnp.inf,
                )
                best = jnp.argmin(proposal_norms)
                best_log_width = proposals[best]
                best_norm = proposal_norms[best]
                current_norm = jnp.max(jnp.abs(current_residual))
                accepted_step = (
                    solved.successful
                    & jnp.isfinite(best_norm)
                    & (best_norm < current_norm)
                )
                accepted = jnp.where(accepted_step, best_log_width, current)
                reached = accepted_step & (
                    best_norm <= self.structural_residual_tolerance
                )
                return accepted, reached, accepted_step

            proposed, reached, valid = jax.lax.cond(
                converged | ~solver_valid,
                lambda current: (current, converged, solver_valid),
                attempt,
                log_width,
            )
            return (
                proposed,
                converged | reached,
                solver_valid & valid,
                count + (~converged & solver_valid).astype(jnp.int32),
            )

        log_width, converged, solver_valid, iterations = jax.lax.fori_loop(
            0,
            self.maximum_structural_iterations,
            iteration,
            (
                initial_log_width,
                initial_converged,
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        faces, density, dispersion_squared, structural_residual = self._structural_state(
            log_width, shell_mass, entropy_proxy
        )
        hydrostatic_fields = self._hydrostatic_from_faces(
            faces, shell_mass, dispersion_squared
        )
        converged = (
            converged
            & solver_valid
            & (
                jnp.max(jnp.abs(structural_residual))
                <= self.structural_residual_tolerance
            )
            & jnp.all(jnp.isfinite(faces))
            & jnp.all(jnp.diff(faces) > 0.0)
            & jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(dispersion_squared))
        )
        return (
            faces,
            density,
            dispersion_squared,
            hydrostatic_fields,
            structural_residual,
            converged,
            iterations,
        )

    def initialize(
        self,
        mass_density: ArrayLike,
        velocity_dispersion_squared: ArrayLike,
        time: ArrayLike = 0.0,
        /,
    ) -> GravothermalSIDMState:
        density = jnp.asarray(mass_density)
        dispersion = jnp.asarray(velocity_dispersion_squared, dtype=density.dtype)
        state = GravothermalSIDMState(
            density,
            dispersion,
            self.radial_faces.astype(density.dtype),
            time,
        )
        self._require_shape(state)
        valid = self._finite_positive(state)
        shell_mass = density * self.shell_volumes.astype(density.dtype)
        entropy_proxy = dispersion / density ** (2.0 / 3.0)
        (
            faces,
            adjusted_density,
            adjusted_dispersion,
            _,
            _,
            converged,
            _,
        ) = self._readjust_structure(shell_mass, entropy_proxy, state.radial_faces)
        candidate = GravothermalSIDMState(
            adjusted_density, adjusted_dispersion, faces, state.time
        )
        accepted_density = eqx.error_if(
            candidate.mass_density,
            ~valid | ~converged | ~self._finite_positive(candidate),
            "Gravothermal initialization failed quasi-static hydrostatic readjustment.",
        )
        return eqx.tree_at(lambda value: value.mass_density, candidate, accepted_density)

    def _require_shape(self, state: GravothermalSIDMState) -> None:
        if not isinstance(state, GravothermalSIDMState):
            raise TypeError("state must be GravothermalSIDMState.")
        expected = (self.shell_count,)
        if (
            state.mass_density.shape != expected
            or state.velocity_dispersion_squared.shape != expected
            or state.radial_faces.shape != (self.shell_count + 1,)
        ):
            raise ValueError(
                f"Gravothermal shell fields must have shape {expected} and radial_faces one additional entry."
            )

    def _finite_positive(self, state: GravothermalSIDMState) -> Array:
        return (
            jnp.isfinite(state.time)
            & jnp.all(jnp.isfinite(state.mass_density))
            & jnp.all(jnp.isfinite(state.velocity_dispersion_squared))
            & jnp.all(jnp.isfinite(state.radial_faces))
            & jnp.all(state.mass_density > 0.0)
            & jnp.all(state.velocity_dispersion_squared > 0.0)
            & (state.radial_faces[0] == 0.0)
            & jnp.all(jnp.diff(state.radial_faces) > 0.0)
        )

    def _closure(self, state: GravothermalSIDMState):
        dtype = state.mass_density.dtype
        faces = state.radial_faces.astype(dtype)
        density = state.mass_density
        dispersion_squared = state.velocity_dispersion_squared
        dispersion = jnp.sqrt(dispersion_squared)
        inner = faces[:-1]
        outer = faces[1:]
        volumes = (4.0 * jnp.pi / 3.0) * (outer**3 - inner**3)
        shell_mass = density * volumes
        (
            centers,
            _,
            reconstructed_density,
            enclosed,
            pressure,
            hydrostatic_residual,
            hydrostatic_relative_residual,
        ) = self._hydrostatic_from_faces(faces, shell_mass, dispersion_squared)
        density = reconstructed_density
        cross_section = jnp.asarray(self.cross_section_per_mass, dtype=dtype)
        collision_rate = density * cross_section * jnp.sqrt(3.0) * dispersion
        relaxation_time = 1.0 / collision_rate
        mean_free_path = 1.0 / (density * cross_section)
        scale_height = jnp.sqrt(
            dispersion_squared
            / (
                4.0
                * jnp.pi
                * jnp.asarray(self.gravitational_constant, dtype=dtype)
                * density
            )
        )
        transport_length_squared = 1.0 / (1.0 / scale_height**2 + 1.0 / mean_free_path**2)
        conductivity = (
            1.5
            * jnp.asarray(self.conductivity_calibration, dtype=dtype)
            * density
            * transport_length_squared
            / relaxation_time
        )
        left_distance = faces[1:-1] - centers[:-1]
        right_distance = centers[1:] - faces[1:-1]
        face_conductivity = (left_distance + right_distance) / (
            left_distance / conductivity[:-1] + right_distance / conductivity[1:]
        )
        temperature_gradient = (dispersion_squared[1:] - dispersion_squared[:-1]) / (
            centers[1:] - centers[:-1]
        )
        interior_heat_flux = -face_conductivity * temperature_gradient
        heat_flux = jnp.zeros((self.shell_count + 1,), dtype=dtype)
        heat_flux = heat_flux.at[1:-1].set(interior_heat_flux)
        luminosity = 4.0 * jnp.pi * faces**2 * heat_flux
        return (
            centers,
            volumes,
            shell_mass,
            enclosed,
            pressure,
            hydrostatic_residual,
            hydrostatic_relative_residual,
            relaxation_time,
            mean_free_path,
            scale_height,
            conductivity,
            heat_flux,
            luminosity,
        )

    def advance(
        self,
        state: GravothermalSIDMState,
        physical_time_step: ArrayLike,
        /,
    ) -> GravothermalSIDMResult:
        self._require_shape(state)
        dtype = state.mass_density.dtype
        dt = jnp.asarray(physical_time_step, dtype=dtype).reshape(())
        timestep_positive = jnp.isfinite(dt) & (dt > 0.0)
        safe_dt = jnp.where(timestep_positive, dt, 0.0)
        (
            centers,
            volumes,
            shell_mass,
            enclosed,
            pressure,
            hydrostatic_residual,
            hydrostatic_relative_residual,
            relaxation_time,
            mean_free_path,
            scale_height,
            conductivity,
            heat_flux,
            luminosity,
        ) = self._closure(state)
        energy_before = 1.5 * shell_mass * state.velocity_dispersion_squared
        energy_rate = luminosity[:-1] - luminosity[1:]
        conductive_increment = safe_dt * energy_rate
        unchanged_energy = conductive_increment == 0.0
        energy_after = jnp.where(
            unchanged_energy,
            energy_before,
            energy_before + conductive_increment,
        )
        dispersion_after = jnp.where(
            unchanged_energy,
            state.velocity_dispersion_squared,
            (2.0 / 3.0) * energy_after / shell_mass,
        )
        entropy_after = dispersion_after / state.mass_density ** (2.0 / 3.0)
        (
            structural_faces,
            final_density,
            final_dispersion,
            structural_fields,
            structural_residual,
            structural_converged,
            structural_iterations,
        ) = self._readjust_structure(shell_mass, entropy_after, state.radial_faces)
        (
            final_centers,
            final_volumes,
            _,
            final_enclosed,
            final_pressure,
            final_hydrostatic_residual,
            final_hydrostatic_relative_residual,
        ) = structural_fields
        no_conduction = jnp.all(unchanged_energy)
        structural_faces = jnp.where(no_conduction, state.radial_faces, structural_faces)
        final_density = jnp.where(no_conduction, state.mass_density, final_density)
        final_dispersion = jnp.where(
            no_conduction,
            state.velocity_dispersion_squared,
            final_dispersion,
        )
        candidate = GravothermalSIDMState(
            final_density,
            final_dispersion,
            structural_faces,
            state.time + safe_dt,
        )
        final_shell_mass = final_density * final_volumes
        shell_mass_defect = final_shell_mass - shell_mass
        final_entropy = final_dispersion / final_density ** (2.0 / 3.0)
        entropy_defect = final_entropy - entropy_after
        mass_scale = jnp.maximum(jnp.abs(shell_mass), jnp.finfo(dtype).tiny)
        entropy_scale = jnp.maximum(jnp.abs(entropy_after), jnp.finfo(dtype).tiny)
        structural_conservative = jnp.all(
            jnp.abs(shell_mass_defect) <= 2048.0 * jnp.finfo(dtype).eps * mass_scale
        ) & jnp.all(
            jnp.abs(entropy_defect) <= 2048.0 * jnp.finfo(dtype).eps * entropy_scale
        )
        boundary_transfer = safe_dt * (luminosity[0] - luminosity[-1])
        energy_balance_defect = (
            jnp.sum(energy_after) - jnp.sum(energy_before) - boundary_transfer
        )
        final_thermal_energy = 1.5 * shell_mass * final_dispersion
        gravitational_energy_before = self._gravitational_binding_energy(
            state.radial_faces, shell_mass
        )
        gravitational_energy_after = self._gravitational_binding_energy(
            structural_faces, shell_mass
        )
        total_energy_before = jnp.sum(energy_before) + gravitational_energy_before
        total_energy_after = jnp.sum(final_thermal_energy) + gravitational_energy_after
        total_energy_defect = total_energy_after - total_energy_before - boundary_transfer
        total_energy_scale = jnp.maximum(
            jnp.maximum(jnp.abs(total_energy_before), jnp.abs(total_energy_after)),
            jnp.maximum(
                jnp.sum(jnp.abs(energy_before)) + jnp.abs(gravitational_energy_before),
                jnp.finfo(dtype).tiny,
            ),
        )
        total_energy_valid = jnp.isfinite(total_energy_defect) & (
            jnp.abs(total_energy_defect)
            <= self.total_energy_relative_tolerance * total_energy_scale
        )
        fractional_change = jnp.abs(energy_after - energy_before) / energy_before
        maximum_change = jnp.max(fractional_change)
        timestep_valid = timestep_positive & (
            maximum_change <= self.maximum_fractional_energy_change
        )
        positive = (
            jnp.all(energy_after > 0.0)
            & jnp.all(final_dispersion > 0.0)
            & jnp.all(final_density > 0.0)
        )
        initial_entropy = state.velocity_dispersion_squared / state.mass_density ** (
            2.0 / 3.0
        )
        initial_structural_residual = self._structural_state(
            jnp.log(jnp.diff(state.radial_faces)),
            shell_mass,
            initial_entropy,
        )[-1]
        initial_hydrostatic_valid = (
            jnp.max(jnp.abs(initial_structural_residual))
            <= self.structural_residual_tolerance
        )
        regime_supported = (
            initial_hydrostatic_valid
            & structural_converged
            & structural_conservative
            & (
                jnp.max(jnp.abs(structural_residual))
                <= self.structural_residual_tolerance
            )
        )
        energy_scale = jnp.maximum(jnp.sum(jnp.abs(energy_before)), jnp.finfo(dtype).tiny)
        balance_valid = jnp.abs(energy_balance_defect) <= (
            1024.0 * jnp.finfo(dtype).eps * energy_scale
        )
        finite = (
            self._finite_positive(state)
            & jnp.all(jnp.isfinite(centers))
            & jnp.all(jnp.isfinite(volumes))
            & jnp.all(jnp.isfinite(enclosed))
            & jnp.all(jnp.isfinite(pressure))
            & jnp.all(jnp.isfinite(hydrostatic_residual))
            & jnp.all(jnp.isfinite(hydrostatic_relative_residual))
            & jnp.all(jnp.isfinite(final_hydrostatic_residual))
            & jnp.all(jnp.isfinite(final_hydrostatic_relative_residual))
            & jnp.all(jnp.isfinite(structural_residual))
            & jnp.isfinite(gravitational_energy_before)
            & jnp.isfinite(gravitational_energy_after)
            & jnp.isfinite(total_energy_before)
            & jnp.isfinite(total_energy_after)
            & jnp.isfinite(total_energy_defect)
            & jnp.all(jnp.isfinite(shell_mass_defect))
            & jnp.all(jnp.isfinite(entropy_defect))
            & jnp.all(jnp.isfinite(relaxation_time))
            & jnp.all(jnp.isfinite(mean_free_path))
            & jnp.all(jnp.isfinite(scale_height))
            & jnp.all(jnp.isfinite(conductivity))
            & jnp.all(jnp.isfinite(heat_flux))
            & jnp.all(jnp.isfinite(luminosity))
            & jnp.all(jnp.isfinite(energy_after))
            & jnp.isfinite(energy_balance_defect)
            & self._finite_positive(candidate)
        )
        successful = (
            regime_supported
            & finite
            & timestep_valid
            & positive
            & balance_valid
            & total_energy_valid
        )
        accepted = _state_where(successful, candidate, state)
        diagnostics = GravothermalSIDMDiagnostics(
            final_centers,
            final_volumes,
            final_enclosed,
            final_pressure,
            final_hydrostatic_residual,
            structural_residual,
            initial_hydrostatic_valid,
            structural_converged,
            structural_iterations,
            shell_mass_defect,
            entropy_defect,
            structural_conservative,
            relaxation_time,
            mean_free_path,
            scale_height,
            conductivity,
            heat_flux,
            luminosity,
            energy_before,
            energy_after,
            final_thermal_energy,
            gravitational_energy_before,
            gravitational_energy_after,
            total_energy_before,
            total_energy_after,
            boundary_transfer,
            energy_balance_defect,
            total_energy_defect,
            total_energy_valid,
            maximum_change,
            regime_supported,
            finite,
            timestep_valid,
            positive,
            successful,
        )
        return GravothermalSIDMResult(candidate, accepted, diagnostics, successful)


__all__ = [
    "gravothermal_calibration_payload",
    "GravothermalSIDMDiagnostics",
    "GravothermalSIDMPlan",
    "GravothermalSIDMResult",
    "GravothermalSIDMState",
]
