#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSystem,
    MaterializationPolicy,
    prepare as prepare_linear,
    solve,
    TolerancePolicy,
)
from ._potential_flow_hydrodynamics import PotentialFlowHydrodynamicsResult3D


_TIME_CONVENTION = "exp(-i*omega*t)"
_FORMULATION_ID = "rao:[K_hs+K_moor-omega^2*(M_dry+A)-i*omega*(C_dry+B)]q=F_inc"
_NON_GOALS = (
    "continuum certification",
    "periodic fluid-kernel synthesis",
    "unprepared repeated-cell or Bloch hydrodynamic coefficients",
    "forward-speed, viscous, or nonlinear response",
    "structural modes outside the supplied work-conjugate wet-force map",
)


class HydrodynamicResponseStatus(IntEnum):
    """Aggregate status of one dense frequency-domain response."""

    SUCCESS = 0
    SINGULAR_DYNAMICS = 1
    LINEAR_SOLVE_FAILED = 2
    NONFINITE_OUTPUT = 3
    RESIDUAL_FAILED = 4


def _nonempty(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _relative_defect(left: np.ndarray, right: np.ndarray, /) -> float:
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _checked_matrix(
    value: ArrayLike,
    size: int,
    name: str,
    tolerance: float,
    /,
) -> tuple[Array, float, float, float]:
    host = np.asarray(value, dtype=np.complex128)
    if host.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}).")
    if np.any(~np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    transpose_defect = _relative_defect(host, host.T)
    hermitian_defect = _relative_defect(host, host.conj().T)
    if transpose_defect > tolerance:
        raise ValueError(f"{name} must be symmetric within symmetry_tolerance.")
    if hermitian_defect > tolerance:
        raise ValueError(f"{name} must be Hermitian within symmetry_tolerance.")
    symmetric = 0.5 * (host + host.T)
    hermitian = 0.5 * (symmetric + symmetric.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian)))
    return (
        jnp.asarray(hermitian, dtype=jnp.complex128),
        transpose_defect,
        hermitian_defect,
        minimum_eigenvalue,
    )


def _require_positive_definite(
    minimum: float,
    matrix: Array,
    tolerance: float,
    name: str,
    /,
) -> None:
    scale = max(float(np.max(np.abs(np.asarray(matrix)))), np.finfo(float).tiny)
    if minimum <= tolerance * scale:
        raise ValueError(f"{name} must be positive definite.")


def _require_positive_semidefinite(
    minimum: float,
    matrix: Array,
    tolerance: float,
    name: str,
    /,
) -> None:
    scale = max(float(np.max(np.abs(np.asarray(matrix)))), np.finfo(float).tiny)
    if minimum < -tolerance * scale:
        raise ValueError(f"{name} violates the declared passive envelope.")


def _checked_excitation(value: ArrayLike, size: int, /) -> Array:
    host = np.asarray(value, dtype=np.complex128)
    if host.ndim == 1:
        host = host[:, None]
    if host.ndim != 2 or host.shape[0] != size or host.shape[1] < 1:
        raise ValueError(
            "incident_excitation must have shape (rigid_dof_count,) or "
            "(rigid_dof_count, load_case_count)."
        )
    if np.any(~np.isfinite(host)):
        raise ValueError("incident_excitation must be finite.")
    return jnp.asarray(host, dtype=jnp.complex128)


def _block_diagonal(left: Array, right: Array, /) -> Array:
    left_size = int(left.shape[0])
    right_size = int(right.shape[0])
    result = jnp.zeros(
        (left_size + right_size, left_size + right_size),
        dtype=jnp.result_type(left, right),
    )
    result = result.at[:left_size, :left_size].set(left)
    return result.at[left_size:, left_size:].set(right)


def _column_quadratic(matrix: Array, vectors: Array, /) -> Array:
    return jnp.real(jnp.sum(jnp.conj(vectors) * (matrix @ vectors), axis=0))


class WetSurfaceModalGeneralizedForceMap3D(StrictModule):
    """Checked work-conjugate reduction from wet forces to structural modes.

    ``matrix`` maps generalized forces in the already-prepared hydrodynamic mode
    coordinates to structural modal forces. Its conjugate transpose is, by
    definition, the work-conjugate map from modal displacement to those wet
    coordinates. This object does not construct fluid modes or a fluid kernel.
    """

    matrix: Array
    rank: int = eqx.field(static=True)
    source_mode_names: tuple[str, ...] = eqx.field(static=True)
    modal_names: tuple[str, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    reference_point_id: str = eqx.field(static=True)
    time_convention: str = eqx.field(static=True)
    resource_evidence: tuple[int, int] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        source_mode_names: tuple[str, ...],
        modal_names: tuple[str, ...],
        geometry_id: str,
        mapping_id: str,
        frame_id: str,
        unit_system_id: str,
        reference_point_id: str,
        provider_id: str = "caller-supplied-checked-wet-force-map",
        precision_id: str = "complex128",
    ):
        source_names = tuple(str(value) for value in source_mode_names)
        target_names = tuple(str(value) for value in modal_names)
        if not source_names or any(not value for value in source_names):
            raise ValueError(
                "source_mode_names must be non-empty and contain no empty ID."
            )
        if not target_names or any(not value for value in target_names):
            raise ValueError("modal_names must be non-empty and contain no empty ID.")
        if len(set(source_names)) != len(source_names):
            raise ValueError("source_mode_names must be unique.")
        if len(set(target_names)) != len(target_names):
            raise ValueError("modal_names must be unique.")
        host = np.asarray(matrix, dtype=np.complex128)
        expected = (len(target_names), len(source_names))
        if host.shape != expected:
            raise ValueError(f"matrix must have shape {expected}.")
        if np.any(~np.isfinite(host)):
            raise ValueError("Wet-surface modal map coefficients must be finite.")
        identifiers = tuple(
            _nonempty(value, name)
            for value, name in (
                (geometry_id, "geometry_id"),
                (mapping_id, "mapping_id"),
                (provider_id, "provider_id"),
                (precision_id, "precision_id"),
                (frame_id, "frame_id"),
                (unit_system_id, "unit_system_id"),
                (reference_point_id, "reference_point_id"),
            )
        )
        self.matrix = jnp.asarray(host, dtype=jnp.complex128)
        self.rank = int(np.linalg.matrix_rank(host))
        self.source_mode_names = source_names
        self.modal_names = target_names
        (
            self.geometry_id,
            self.mapping_id,
            self.provider_id,
            self.precision_id,
            self.frame_id,
            self.unit_system_id,
            self.reference_point_id,
        ) = identifiers
        self.physics_id = "linear-potential-flow-wet-force-to-structural-mode-map"
        self.formulation_id = "Q_modal=G*F_wet; q_wet=G^H*eta_modal"
        self.time_convention = _TIME_CONVENTION
        self.resource_evidence = (int(host.nbytes), int(host.size))
        self.error_evidence = (
            (
                "shape, finite coefficients, unique coordinate IDs, and "
                "numerical rank checked"
            ),
            "work conjugacy is exact algebraically through the conjugate-transpose map",
            "no continuum modal-projection error estimate",
        )
        self.non_goals = (
            "fluid-kernel construction",
            "continuum certification",
            "modal truncation certification",
        )

    def mv(self, wet_generalized_force: ArrayLike, /) -> Array:
        """Apply the wet-force to modal-force map."""
        value = jnp.asarray(wet_generalized_force, dtype=jnp.complex128)
        if value.ndim not in (1, 2) or value.shape[0] != self.matrix.shape[1]:
            raise ValueError("wet_generalized_force has incompatible leading dimension.")
        return self.matrix @ value

    def transpose_mv(self, modal_covector: ArrayLike, /) -> Array:
        """Apply the exact algebraic transpose of the force map."""
        value = jnp.asarray(modal_covector, dtype=jnp.complex128)
        if value.ndim not in (1, 2) or value.shape[0] != self.matrix.shape[0]:
            raise ValueError("modal_covector has incompatible leading dimension.")
        return self.matrix.T @ value

    def adjoint_mv(self, modal_displacement: ArrayLike, /) -> Array:
        """Apply the exact Hilbert adjoint/work-conjugate displacement map."""
        value = jnp.asarray(modal_displacement, dtype=jnp.complex128)
        if value.ndim not in (1, 2) or value.shape[0] != self.matrix.shape[0]:
            raise ValueError("modal_displacement has incompatible leading dimension.")
        return jnp.conj(self.matrix.T) @ value


class HydrodynamicResponseResult3D(StrictModule):
    """Checked rigid or reduced hydroelastic response at one frequency.

    Incident columns are generalized-force amplitudes per unit incident-wave
    amplitude, so ``rigid_body_rao`` and ``modal_response`` are displacement
    response-amplitude operators. The retained dense operator supplies exact
    forward, algebraic-transpose, and Hilbert-adjoint actions.
    """

    displacement_response: Array
    rigid_body_rao: Array
    modal_response: Array
    incident_excitation: Array
    generalized_excitation: Array
    wet_coordinate_transform: Array
    physical_mass_inertia: Array
    structural_modal_mass: Array
    dry_mass: Array
    hydrodynamic_added_mass: Array
    external_damping: Array
    structural_modal_damping: Array
    dry_damping: Array
    hydrodynamic_radiation_damping: Array
    hydrostatic_restoring: Array
    mooring_restoring: Array
    structural_modal_stiffness: Array
    total_restoring: Array
    dynamic_operator: DenseLinearOperator
    residual: Array
    residual_norm: Array
    relative_residual: Array
    residual_threshold: Array
    linear_statuses: Array
    linear_results: tuple[LinearSolveResult, ...]
    status: Array
    finite: Array
    residual_accepted: Array
    transpose_symmetry_defects: Array
    hermitian_defects: Array
    minimum_mass_eigenvalue: Array
    minimum_total_damping_eigenvalue: Array
    minimum_restoring_eigenvalue: Array
    passive: Array
    average_incident_power: Array
    average_radiated_power: Array
    average_external_dissipation: Array
    average_total_dissipation: Array
    average_power_balance_residual: Array
    angular_frequency: Array
    hydrodynamics: PotentialFlowHydrodynamicsResult3D
    modal_force_map: WetSurfaceModalGeneralizedForceMap3D | None
    rigid_mode_names: tuple[str, ...] = eqx.field(static=True)
    modal_names: tuple[str, ...] = eqx.field(static=True)
    matrix_evidence_names: tuple[str, ...] = eqx.field(static=True)
    coefficient_frequency_id: str = eqx.field(static=True)
    excitation_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    reference_point_id: str = eqx.field(static=True)
    time_convention: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    pde_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    hydrodynamic_provider_id: str = eqx.field(static=True)
    solver_provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    excitation_semantics: str = eqx.field(static=True)
    resource_evidence: tuple[int, int, int, int] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    response_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(HydrodynamicResponseStatus.SUCCESS)

    def apply_dynamic(self, displacement: ArrayLike, /) -> Array:
        return self.dynamic_operator.mv(displacement)

    def apply_dynamic_transpose(self, covector: ArrayLike, /) -> Array:
        return self.dynamic_operator.transpose_mv(covector)

    def apply_dynamic_adjoint(self, covector: ArrayLike, /) -> Array:
        return self.dynamic_operator.adjoint_mv(covector)


def solve_hydrodynamic_response_3d(
    hydrodynamics: PotentialFlowHydrodynamicsResult3D,
    /,
    *,
    angular_frequency: float,
    coefficient_frequency_id: str,
    excitation_id: str,
    physical_mass_inertia: ArrayLike,
    external_damping: ArrayLike,
    hydrostatic_restoring: ArrayLike,
    mooring_restoring: ArrayLike,
    incident_excitation: ArrayLike,
    frame_id: str,
    unit_system_id: str,
    reference_point_id: str,
    hydrodynamic_reference_point_id: str,
    structural_modal_mass: ArrayLike | None = None,
    structural_modal_stiffness: ArrayLike | None = None,
    structural_modal_damping: ArrayLike | None = None,
    modal_force_map: WetSurfaceModalGeneralizedForceMap3D | None = None,
    symmetry_tolerance: float = 1.0e-10,
    passivity_tolerance: float = 1.0e-10,
    relative_residual_tolerance: float = 1.0e-10,
    absolute_residual_tolerance: float = 1.0e-12,
) -> HydrodynamicResponseResult3D:
    """Solve the exp(-iωt) rigid/modal frequency-domain response.

    The hydrodynamic coefficients are consumed exactly as supplied by
    ``PotentialFlowHydrodynamicsResult3D``. ``coefficient_frequency_id`` and
    ``hydrodynamic_reference_point_id`` are explicit caller assertions because
    that result does not expose its preparation frequency or reference-point ID.
    Every unsupported metadata or evidence mismatch fails before linear-system
    preparation.
    """
    if not isinstance(hydrodynamics, PotentialFlowHydrodynamicsResult3D):
        raise TypeError("hydrodynamics must be PotentialFlowHydrodynamicsResult3D.")
    if not bool(hydrodynamics.valid):
        raise ValueError("Hydrodynamic coefficient evidence is not valid.")
    if not bool(hydrodynamics.radiated_power_nonnegative):
        raise ValueError("Hydrodynamic radiation damping lacks passivity evidence.")
    if hydrodynamics.ambient_dimension != 3:
        raise ValueError("Only three-dimensional hydrodynamic results are supported.")
    if hydrodynamics.time_convention != _TIME_CONVENTION:
        raise ValueError(f"Hydrodynamic coefficients must use {_TIME_CONVENTION!r}.")

    omega = float(angular_frequency)
    symmetry_limit = float(symmetry_tolerance)
    passivity_limit = float(passivity_tolerance)
    relative_limit = float(relative_residual_tolerance)
    absolute_limit = float(absolute_residual_tolerance)
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("angular_frequency must be finite and positive.")
    limits = (symmetry_limit, passivity_limit, relative_limit, absolute_limit)
    if any(not math.isfinite(value) or value < 0.0 for value in limits):
        raise ValueError("All response tolerances must be finite and nonnegative.")

    coefficient_id = _nonempty(coefficient_frequency_id, "coefficient_frequency_id")
    load_id = _nonempty(excitation_id, "excitation_id")
    frame = _nonempty(frame_id, "frame_id")
    units = _nonempty(unit_system_id, "unit_system_id")
    reference = _nonempty(reference_point_id, "reference_point_id")
    hydro_reference = _nonempty(
        hydrodynamic_reference_point_id, "hydrodynamic_reference_point_id"
    )
    if frame != hydrodynamics.frame_id:
        raise ValueError("Response and hydrodynamic frame IDs do not match.")
    if units != hydrodynamics.unit_system_id:
        raise ValueError("Response and hydrodynamic unit-system IDs do not match.")
    if reference != hydro_reference:
        raise ValueError("Response and hydrodynamic reference-point IDs do not match.")

    rigid_names = tuple(hydrodynamics.mode_names)
    rigid_size = len(rigid_names)
    if rigid_size < 1:
        raise ValueError("Hydrodynamic result contains no rigid modes.")
    excitation = _checked_excitation(incident_excitation, rigid_size)

    matrix_names: list[str] = []
    transpose_defects: list[float] = []
    hermitian_defects: list[float] = []

    def checked(value: ArrayLike, name: str) -> tuple[Array, float]:
        matrix, transpose_defect, hermitian_defect, minimum = _checked_matrix(
            value, rigid_size, name, symmetry_limit
        )
        matrix_names.append(name)
        transpose_defects.append(transpose_defect)
        hermitian_defects.append(hermitian_defect)
        return matrix, minimum

    physical_mass, minimum_physical_mass = checked(
        physical_mass_inertia, "physical_mass_inertia"
    )
    external, minimum_external_damping = checked(external_damping, "external_damping")
    hydrostatic, _ = checked(hydrostatic_restoring, "hydrostatic_restoring")
    mooring, _ = checked(mooring_restoring, "mooring_restoring")
    added_mass, _ = checked(hydrodynamics.added_mass, "hydrodynamic_added_mass")
    radiation_damping, minimum_radiation_damping = checked(
        hydrodynamics.radiation_damping, "hydrodynamic_radiation_damping"
    )
    _require_positive_definite(
        minimum_physical_mass,
        physical_mass,
        passivity_limit,
        "physical_mass_inertia",
    )
    _require_positive_semidefinite(
        minimum_external_damping,
        external,
        passivity_limit,
        "external_damping",
    )
    _require_positive_semidefinite(
        minimum_radiation_damping,
        radiation_damping,
        passivity_limit,
        "hydrodynamic_radiation_damping",
    )

    modal_values = (
        structural_modal_mass,
        structural_modal_stiffness,
        structural_modal_damping,
        modal_force_map,
    )
    modal_enabled = all(value is not None for value in modal_values)
    if any(value is not None for value in modal_values) and not modal_enabled:
        raise ValueError(
            (
                "Modal mass, stiffness, damping, and modal_force_map must be "
                "supplied together."
            )
        )

    if modal_enabled:
        if not isinstance(modal_force_map, WetSurfaceModalGeneralizedForceMap3D):
            raise TypeError(
                "modal_force_map must be WetSurfaceModalGeneralizedForceMap3D."
            )
        if modal_force_map.source_mode_names != rigid_names:
            raise ValueError("Modal-map hydrodynamic mode IDs do not match the result.")
        if modal_force_map.geometry_id != hydrodynamics.geometry_id:
            raise ValueError("Modal-map and hydrodynamic geometry IDs do not match.")
        if modal_force_map.frame_id != frame:
            raise ValueError("Modal-map and response frame IDs do not match.")
        if modal_force_map.unit_system_id != units:
            raise ValueError("Modal-map and response unit-system IDs do not match.")
        if modal_force_map.reference_point_id != reference:
            raise ValueError("Modal-map and response reference-point IDs do not match.")
        if modal_force_map.time_convention != _TIME_CONVENTION:
            raise ValueError("Modal-map time convention is unsupported.")
        modal_names = modal_force_map.modal_names
        modal_size = len(modal_names)

        def checked_modal(value: ArrayLike, name: str) -> tuple[Array, float]:
            matrix, transpose_defect, hermitian_defect, minimum = _checked_matrix(
                value, modal_size, name, symmetry_limit
            )
            matrix_names.append(name)
            transpose_defects.append(transpose_defect)
            hermitian_defects.append(hermitian_defect)
            return matrix, minimum

        modal_mass, minimum_modal_mass = checked_modal(
            structural_modal_mass, "structural_modal_mass"
        )
        modal_stiffness, minimum_modal_stiffness = checked_modal(
            structural_modal_stiffness, "structural_modal_stiffness"
        )
        modal_damping, minimum_modal_damping = checked_modal(
            structural_modal_damping, "structural_modal_damping"
        )
        _require_positive_definite(
            minimum_modal_mass,
            modal_mass,
            passivity_limit,
            "structural_modal_mass",
        )
        _require_positive_semidefinite(
            minimum_modal_stiffness,
            modal_stiffness,
            passivity_limit,
            "structural_modal_stiffness",
        )
        _require_positive_semidefinite(
            minimum_modal_damping,
            modal_damping,
            passivity_limit,
            "structural_modal_damping",
        )
        transform = jnp.concatenate(
            (
                jnp.eye(rigid_size, dtype=jnp.complex128),
                jnp.conj(modal_force_map.matrix.T),
            ),
            axis=1,
        )
        dry_mass = _block_diagonal(physical_mass, modal_mass)
        dry_damping = _block_diagonal(external, modal_damping)
        total_restoring = _block_diagonal(hydrostatic + mooring, modal_stiffness)
    else:
        modal_force_map = None
        modal_names = ()
        modal_size = 0
        modal_mass = jnp.zeros((0, 0), dtype=jnp.complex128)
        modal_stiffness = jnp.zeros((0, 0), dtype=jnp.complex128)
        modal_damping = jnp.zeros((0, 0), dtype=jnp.complex128)
        transform = jnp.eye(rigid_size, dtype=jnp.complex128)
        dry_mass = physical_mass
        dry_damping = external
        total_restoring = hydrostatic + mooring

    transform_adjoint = jnp.conj(transform.T)
    generalized_added_mass = transform_adjoint @ added_mass @ transform
    generalized_radiation_damping = transform_adjoint @ radiation_damping @ transform
    generalized_excitation = transform_adjoint @ excitation
    total_mass = dry_mass + generalized_added_mass
    total_damping = dry_damping + generalized_radiation_damping

    total_mass_host = np.asarray(total_mass)
    total_damping_host = np.asarray(total_damping)
    total_restoring_host = np.asarray(total_restoring)
    total_mass_hermitian_defect = _relative_defect(
        total_mass_host, total_mass_host.conj().T
    )
    total_damping_hermitian_defect = _relative_defect(
        total_damping_host, total_damping_host.conj().T
    )
    total_restoring_hermitian_defect = _relative_defect(
        total_restoring_host, total_restoring_host.conj().T
    )
    if (
        max(
            total_mass_hermitian_defect,
            total_damping_hermitian_defect,
            total_restoring_hermitian_defect,
        )
        > symmetry_limit
    ):
        raise ValueError(
            "Projected response matrices are not Hermitian within tolerance."
        )
    minimum_total_mass = float(np.min(np.linalg.eigvalsh(total_mass_host)))
    minimum_total_damping = float(np.min(np.linalg.eigvalsh(total_damping_host)))
    minimum_total_restoring = float(np.min(np.linalg.eigvalsh(total_restoring_host)))
    total_damping_scale = max(
        float(np.max(np.abs(total_damping_host))), np.finfo(float).tiny
    )
    passive = minimum_total_damping >= -passivity_limit * total_damping_scale
    _require_positive_definite(
        minimum_total_mass, total_mass, passivity_limit, "total mass"
    )
    _require_positive_semidefinite(
        minimum_total_damping, total_damping, passivity_limit, "total damping"
    )

    omega_array = jnp.asarray(omega, dtype=jnp.float64)
    dynamic_matrix = (
        total_restoring - omega_array**2 * total_mass - 1j * omega_array * total_damping
    )
    operator = DenseLinearOperator(dynamic_matrix)
    dof_count = int(dynamic_matrix.shape[0])
    entries = dof_count * dof_count
    policy = LinearSolvePolicy(
        DenseLU(),
        tolerance=TolerancePolicy(
            relative=relative_limit,
            absolute=absolute_limit,
        ),
        materialization=MaterializationPolicy(
            max_entries=max(entries, 1),
            max_bytes=max(entries * np.dtype(np.complex128).itemsize, 1),
        ),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    prepared = prepare_linear(
        LinearSystem(operator, problem_id="rigid-modal-hydrodynamic-response-3d"),
        policy,
    )
    linear_results = tuple(
        solve(prepared, generalized_excitation[:, index])
        for index in range(generalized_excitation.shape[1])
    )
    response = jnp.stack(
        tuple(jnp.asarray(result.value) for result in linear_results), axis=1
    )
    residual = dynamic_matrix @ response - generalized_excitation
    residual_norm = jnp.linalg.norm(residual, axis=0)
    excitation_norm = jnp.linalg.norm(generalized_excitation, axis=0)
    residual_threshold = absolute_limit + relative_limit * excitation_norm
    relative_residual = residual_norm / jnp.maximum(
        excitation_norm, jnp.finfo(residual_norm.dtype).tiny
    )
    linear_statuses = jnp.stack(tuple(result.status for result in linear_results))
    linear_successful = jnp.all(
        jnp.stack(
            tuple(
                result.successful & result.diagnostics.finite for result in linear_results
            )
        )
    )
    finite = (
        jnp.all(jnp.isfinite(response))
        & jnp.all(jnp.isfinite(residual))
        & jnp.all(jnp.isfinite(residual_norm))
    )
    residual_accepted = jnp.all(residual_norm <= residual_threshold)
    singular = jnp.any(
        (linear_statuses == int(LinearSolveStatus.SINGULAR))
        | (linear_statuses == int(LinearSolveStatus.RANK_DEFICIENT))
    )
    status = jnp.where(
        singular,
        int(HydrodynamicResponseStatus.SINGULAR_DYNAMICS),
        jnp.where(
            ~finite,
            int(HydrodynamicResponseStatus.NONFINITE_OUTPUT),
            jnp.where(
                ~linear_successful,
                int(HydrodynamicResponseStatus.LINEAR_SOLVE_FAILED),
                jnp.where(
                    ~residual_accepted,
                    int(HydrodynamicResponseStatus.RESIDUAL_FAILED),
                    int(HydrodynamicResponseStatus.SUCCESS),
                ),
            ),
        ),
    )

    velocity = -1j * omega_array * response
    incident_power = 0.5 * jnp.real(
        jnp.sum(jnp.conj(generalized_excitation) * velocity, axis=0)
    )
    radiated_power = (
        0.5 * omega_array**2 * _column_quadratic(generalized_radiation_damping, response)
    )
    external_power = 0.5 * omega_array**2 * _column_quadratic(dry_damping, response)
    total_power = radiated_power + external_power
    power_balance = incident_power - total_power

    matrix_names.extend(
        ("projected_total_mass", "projected_total_damping", "total_restoring")
    )
    transpose_defects.extend(
        (
            _relative_defect(total_mass_host, total_mass_host.T),
            _relative_defect(total_damping_host, total_damping_host.T),
            _relative_defect(total_restoring_host, total_restoring_host.T),
        )
    )
    hermitian_defects.extend(
        (
            total_mass_hermitian_defect,
            total_damping_hermitian_defect,
            total_restoring_hermitian_defect,
        )
    )
    solver_providers = tuple(result.provenance.backend for result in linear_results)
    response_id = canonical_fingerprint(
        {
            "kind": "hydrodynamic-response-3d",
            "operator": operator.operator_id,
            "hydrodynamics": hydrodynamics.geometry_id,
            "frequency": coefficient_id,
            "excitation": load_id,
            "reference": reference,
            "modal_map": (
                None if modal_force_map is None else modal_force_map.mapping_id
            ),
        }
    )
    return HydrodynamicResponseResult3D(
        displacement_response=response,
        rigid_body_rao=response[:rigid_size],
        modal_response=response[rigid_size:],
        incident_excitation=excitation,
        generalized_excitation=generalized_excitation,
        wet_coordinate_transform=transform,
        physical_mass_inertia=physical_mass,
        structural_modal_mass=modal_mass,
        dry_mass=dry_mass,
        hydrodynamic_added_mass=generalized_added_mass,
        external_damping=external,
        structural_modal_damping=modal_damping,
        dry_damping=dry_damping,
        hydrodynamic_radiation_damping=generalized_radiation_damping,
        hydrostatic_restoring=hydrostatic,
        mooring_restoring=mooring,
        structural_modal_stiffness=modal_stiffness,
        total_restoring=total_restoring,
        dynamic_operator=operator,
        residual=residual,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        residual_threshold=residual_threshold,
        linear_statuses=linear_statuses,
        linear_results=linear_results,
        status=status,
        finite=finite,
        residual_accepted=residual_accepted,
        transpose_symmetry_defects=jnp.asarray(transpose_defects),
        hermitian_defects=jnp.asarray(hermitian_defects),
        minimum_mass_eigenvalue=jnp.asarray(minimum_total_mass),
        minimum_total_damping_eigenvalue=jnp.asarray(minimum_total_damping),
        minimum_restoring_eigenvalue=jnp.asarray(minimum_total_restoring),
        passive=jnp.asarray(passive),
        average_incident_power=incident_power,
        average_radiated_power=radiated_power,
        average_external_dissipation=external_power,
        average_total_dissipation=total_power,
        average_power_balance_residual=power_balance,
        angular_frequency=omega_array,
        hydrodynamics=hydrodynamics,
        modal_force_map=modal_force_map,
        rigid_mode_names=rigid_names,
        modal_names=modal_names,
        matrix_evidence_names=tuple(matrix_names),
        coefficient_frequency_id=coefficient_id,
        excitation_id=load_id,
        frame_id=frame,
        unit_system_id=units,
        reference_point_id=reference,
        time_convention=_TIME_CONVENTION,
        coordinate_convention=hydrodynamics.coordinate_convention,
        pde_id=hydrodynamics.pde_id,
        geometry_id=hydrodynamics.geometry_id,
        formulation_id=_FORMULATION_ID,
        hydrodynamic_provider_id=hydrodynamics.provider_id,
        solver_provider_id="phydrax.linalg.DenseLU:" + ",".join(solver_providers),
        precision_id=hydrodynamics.precision_id + ":complex128-response",
        excitation_semantics=(
            "complex fluid-on-body generalized-force amplitude per unit "
            "incident-wave amplitude"
        ),
        resource_evidence=(
            int(dynamic_matrix.nbytes),
            int(transform.nbytes),
            int(generalized_excitation.nbytes),
            len(linear_results),
        ),
        error_evidence=(
            "exact discrete residual D*q-F is retained for every load column",
            (
                "phydrax.linalg finite/status/residual diagnostics are retained "
                "per DenseLU solve"
            ),
            "coefficient frequency and reference IDs are explicit caller assertions",
            (
                "no continuum hydrodynamic, structural-discretization, or "
                "modal-truncation bound"
            ),
        ),
        non_goals=_NON_GOALS,
        continuum_certified=False,
        response_id=response_id,
    )


__all__ = [
    "HydrodynamicResponseResult3D",
    "HydrodynamicResponseStatus",
    "WetSurfaceModalGeneralizedForceMap3D",
    "solve_hydrodynamic_response_3d",
]
