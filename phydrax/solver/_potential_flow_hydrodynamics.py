#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..linalg import (
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    MaterializationPolicy,
    prepare as prepare_linear,
    solve,
)
from ..operators.integral.layer_potential._free_surface_hydrodynamics3d import (
    FreeSurfaceHydrodynamicsAssemblyReport3D,
    PreparedFreeSurfaceHydrodynamics3D,
)


_NON_GOALS = (
    "continuum-discretization certification",
    "forward speed",
    "viscous or nonlinear loads",
    "surface-piercing-body hydrodynamics",
    "irregular-frequency removal",
)


class PotentialFlowHydrodynamicsResult3D(StrictModule):
    """Radiation coefficients and regular-wave loads for one prepared frequency.

    Radiation columns correspond to unit generalized velocity, not unit
    displacement. Source densities are unweighted DP0 single-layer strengths.
    With exp(-iωt), raw A=-ρ Re(I) and raw B=-ρω Im(I), where
    I_ij=∫_body phi_j m_i dS. Reported A/B are the transparent symmetric
    reciprocity projections; ``radiation_integrals`` and pre-projection defects
    retain the unmodified discretization evidence. Incident amplitudes are
    free-surface elevation amplitudes and excitation loads are fluid-on-body
    generalized forces.
    """

    radiation_density: Array
    radiation_potential_trace: Array
    radiation_integrals: Array
    added_mass: Array
    radiation_damping: Array
    diffraction_density: Array
    diffraction_potential_trace: Array
    excitation_loads: Array
    radiation_linear_results: tuple[LinearSolveResult, ...]
    diffraction_linear_results: tuple[LinearSolveResult, ...]
    assembly_report: FreeSurfaceHydrodynamicsAssemblyReport3D
    fluid_density: Array
    added_mass_reciprocity_defect: Array
    damping_reciprocity_defect: Array
    minimum_radiated_power_eigenvalue: Array
    radiated_power_nonnegative: Array
    valid: Array
    incident_headings: tuple[float, ...] = eqx.field(static=True)
    mode_names: tuple[str, ...] = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    pde_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    time_convention: str = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    density_semantics: str = eqx.field(static=True)
    resource_evidence: tuple[int, int, int] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)

    def radiated_power(self, generalized_velocity: ArrayLike, /) -> Array:
        """Return the unmodified time-average radiated-power quadratic form."""
        velocity = jnp.asarray(generalized_velocity, dtype=jnp.complex128)
        size = len(self.mode_names)
        if velocity.shape != (size,):
            raise ValueError(f"generalized_velocity must have shape ({size},).")
        return 0.5 * jnp.real(jnp.vdot(velocity, self.radiation_damping @ velocity))


def _default_linear(prepared: PreparedFreeSurfaceHydrodynamics3D) -> LinearSolvePolicy:
    entries = prepared.face_count * prepared.face_count
    return LinearSolvePolicy(
        DenseLU(),
        materialization=MaterializationPolicy(
            max_entries=max(entries, 1),
            max_bytes=max(entries * np.dtype(np.complex128).itemsize, 1),
        ),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )


def _reciprocity_defect(matrix: Array, /) -> Array:
    transpose = jnp.swapaxes(matrix, -1, -2)
    scale = jnp.maximum(
        jnp.max(jnp.maximum(jnp.abs(matrix), jnp.abs(transpose))),
        jnp.finfo(matrix.dtype).tiny,
    )
    return jnp.max(jnp.abs(matrix - transpose)) / scale


def solve_potential_flow_hydrodynamics_3d(
    prepared: PreparedFreeSurfaceHydrodynamics3D,
    /,
    *,
    fluid_density: float = 1025.0,
    incident_headings: Sequence[float] = (0.0,),
    incident_amplitude: ArrayLike = 1.0,
    linear: LinearSolvePolicy | None = None,
    reciprocity_tolerance: float = 2.5e-1,
    radiated_power_tolerance: float = 1.0e-8,
) -> PotentialFlowHydrodynamicsResult3D:
    """Solve all rigid radiation modes and requested incident-wave headings."""
    if not isinstance(prepared, PreparedFreeSurfaceHydrodynamics3D):
        raise TypeError("prepared must be PreparedFreeSurfaceHydrodynamics3D.")
    if not bool(prepared.assembly_report.supported):
        raise ValueError("Prepared hydrodynamics evidence does not support this solve.")
    density_value = float(fluid_density)
    reciprocity_limit = float(reciprocity_tolerance)
    power_limit = float(radiated_power_tolerance)
    if not math.isfinite(density_value) or density_value <= 0.0:
        raise ValueError("fluid_density must be finite and positive.")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (reciprocity_limit, power_limit)
    ):
        raise ValueError("Diagnostic tolerances must be finite and nonnegative.")
    headings = tuple(float(value) for value in incident_headings)
    if any(not math.isfinite(value) for value in headings):
        raise ValueError("Every incident heading must be finite.")
    amplitude = jnp.asarray(incident_amplitude, dtype=jnp.complex128)
    if amplitude.shape not in ((), (len(headings),)):
        raise ValueError(
            "incident_amplitude must be scalar or contain one value per heading."
        )
    amplitudes = (
        jnp.broadcast_to(amplitude, (len(headings),))
        if amplitude.shape == ()
        else amplitude
    )
    if not bool(jnp.all(jnp.isfinite(amplitudes))):
        raise ValueError("incident_amplitude must be finite.")
    policy = _default_linear(prepared) if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError(
            "Potential-flow hydrodynamics currently requires differentiation mode 'none'."
        )

    problem = LinearSystem(
        prepared.boundary_operator,
        problem_id="zero-speed-free-surface-source-neumann-3d",
    )
    prepared_linear = prepare_linear(problem, policy)
    radiation_rhs = jnp.asarray(prepared.rigid_mode_normal_velocity, dtype=jnp.complex128)
    radiation_results = tuple(
        solve(prepared_linear, radiation_rhs[:, index])
        for index in range(prepared.degree_of_freedom_count)
    )
    radiation_density = jnp.stack(
        tuple(jnp.asarray(result.value) for result in radiation_results), axis=1
    )
    radiation_trace = jnp.stack(
        tuple(
            prepared.trace_operator.mv(radiation_density[:, index])
            for index in range(prepared.degree_of_freedom_count)
        ),
        axis=1,
    )
    radiation_integrals = ein.contract(
        "fi,f,fj->ij",
        prepared.rigid_mode_normal_velocity,
        prepared.face_areas,
        radiation_trace,
        backend="jax",
    )
    density = jnp.asarray(density_value, dtype=jnp.float64)
    raw_added_mass = -density * jnp.real(radiation_integrals)
    raw_damping = -density * prepared.angular_frequency * jnp.imag(radiation_integrals)
    added_mass_defect = _reciprocity_defect(raw_added_mass)
    damping_defect = _reciprocity_defect(raw_damping)
    added_mass = 0.5 * (raw_added_mass + raw_added_mass.T)
    damping = 0.5 * (raw_damping + raw_damping.T)
    damping_eigenvalues = jnp.linalg.eigvalsh(damping)
    minimum_power_eigenvalue = 0.5 * jnp.min(damping_eigenvalues)
    damping_scale = jnp.maximum(jnp.max(jnp.abs(damping)), jnp.finfo(damping.dtype).tiny)
    power_nonnegative = minimum_power_eigenvalue >= -power_limit * damping_scale

    diffraction_results_list: list[LinearSolveResult] = []
    diffraction_density_columns: list[Array] = []
    diffraction_trace_columns: list[Array] = []
    excitation_columns: list[Array] = []
    for index, heading in enumerate(headings):
        incident_trace, incident_normal = prepared.incident_wave(
            amplitudes[index], heading
        )
        linear_result = solve(prepared_linear, -incident_normal)
        scattering_density = jnp.asarray(linear_result.value)
        scattering_trace = prepared.trace_operator.mv(scattering_density)
        total_trace = incident_trace + scattering_trace
        excitation = (
            -1j
            * density
            * prepared.angular_frequency
            * ein.contract(
                "fi,f,f->i",
                prepared.rigid_mode_normal_velocity,
                prepared.face_areas,
                total_trace,
                backend="jax",
            )
        )
        diffraction_results_list.append(linear_result)
        diffraction_density_columns.append(scattering_density)
        diffraction_trace_columns.append(scattering_trace)
        excitation_columns.append(excitation)
    if headings:
        diffraction_density = jnp.stack(tuple(diffraction_density_columns), axis=1)
        diffraction_trace = jnp.stack(tuple(diffraction_trace_columns), axis=1)
        excitation_loads = jnp.stack(tuple(excitation_columns), axis=1)
    else:
        diffraction_density = jnp.zeros((prepared.face_count, 0), dtype=jnp.complex128)
        diffraction_trace = jnp.zeros((prepared.face_count, 0), dtype=jnp.complex128)
        excitation_loads = jnp.zeros(
            (prepared.degree_of_freedom_count, 0), dtype=jnp.complex128
        )
    diffraction_results = tuple(diffraction_results_list)

    all_results = radiation_results + diffraction_results
    linear_valid = jnp.all(
        jnp.stack(
            tuple(result.successful & result.diagnostics.finite for result in all_results)
        )
    )
    finite = (
        jnp.all(jnp.isfinite(radiation_density))
        & jnp.all(jnp.isfinite(radiation_trace))
        & jnp.all(jnp.isfinite(added_mass))
        & jnp.all(jnp.isfinite(damping))
        & jnp.all(jnp.isfinite(diffraction_density))
        & jnp.all(jnp.isfinite(excitation_loads))
    )
    reciprocal = (added_mass_defect <= reciprocity_limit) & (
        damping_defect <= reciprocity_limit
    )
    valid = (
        prepared.assembly_report.supported
        & linear_valid
        & finite
        & reciprocal
        & power_nonnegative
    )
    return PotentialFlowHydrodynamicsResult3D(
        radiation_density=radiation_density,
        radiation_potential_trace=radiation_trace,
        radiation_integrals=radiation_integrals,
        added_mass=added_mass,
        radiation_damping=damping,
        diffraction_density=diffraction_density,
        diffraction_potential_trace=diffraction_trace,
        excitation_loads=excitation_loads,
        radiation_linear_results=radiation_results,
        diffraction_linear_results=diffraction_results,
        assembly_report=prepared.assembly_report,
        fluid_density=density,
        added_mass_reciprocity_defect=added_mass_defect,
        damping_reciprocity_defect=damping_defect,
        minimum_radiated_power_eigenvalue=minimum_power_eigenvalue,
        radiated_power_nonnegative=power_nonnegative,
        valid=valid,
        incident_headings=headings,
        mode_names=prepared.mode_names,
        ambient_dimension=3,
        coordinate_convention=prepared.coordinate_convention,
        pde_id=prepared.pde_id,
        geometry_id=prepared.geometry_id,
        formulation_id="unit-generalized-velocity-radiation-and-fixed-body-diffraction",
        provider_id="phydrax-dense-lu-dp0-free-surface-green",
        precision_id=prepared.precision_id,
        frame_id=prepared.frame_id,
        unit_system_id=prepared.unit_system_id,
        time_convention=prepared.time_convention,
        normal_convention=prepared.normal_convention,
        density_semantics=prepared.density_semantics,
        resource_evidence=(
            prepared.assembly_report.resident_bytes,
            int(radiation_density.nbytes),
            int(diffraction_density.nbytes),
        ),
        error_evidence=(
            "linear residual diagnostics are retained per right-hand side",
            "A/B are symmetric reciprocity projections; raw defects are retained",
            "no continuum collocation or geometry-discretization error estimate",
        ),
        non_goals=_NON_GOALS,
    )


__all__ = [
    "PotentialFlowHydrodynamicsResult3D",
    "solve_potential_flow_hydrodynamics_3d",
]
