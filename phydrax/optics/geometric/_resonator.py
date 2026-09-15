#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    HermitianSpectrum,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenResourcePolicy,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
    GeneralEigenSolveResult,
)
from ._paraxial import (
    _COORDINATE_CONVENTION,
    DifferentialRayMap,
    ParaxialOpticsStatus,
)


class ParaxialResonatorStatus(IntEnum):
    """Terminal status for a coupled paraxial round-trip analysis."""

    SUCCESS_STABLE = 0
    SUCCESS_UNSTABLE = 1
    MARGINAL = 2
    SINGULAR_CLOSED_ORBIT = 3
    INCOMPATIBLE_MAPS = 4
    NONSYMPLECTIC = 5
    INVALID_MODE = 6
    NONFINITE = 7


class ParaxialResonatorMode(StrictModule):
    """Positive invariant complex Lagrangian plane of a stable round trip."""

    lagrangian_state: Array
    curvature: Array
    round_trip_action: Array
    positivity_eigenvalues: Array
    lagrangian_residual: Array
    invariance_residual: Array
    unitarity_residual: Array
    curvature_symmetry_residual: Array
    valid: Array
    frame_id: str = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    resonator_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)

    @property
    def h(self) -> Array:
        return self.lagrangian_state[:2, :]

    @property
    def u(self) -> Array:
        return self.lagrangian_state[2:, :]


class ParaxialResonatorEvidence(StrictModule, NonTrainableState):
    """Compatibility, closure, symplectic, spectral, and mode evidence."""

    map_valid: Array
    frame_consistent: Array
    coordinate_consistent: Array
    provenance_consistent: Array
    finite: Array
    symplectic_error: Array
    closed_orbit_residual: Array
    closed_orbit_condition: Array
    eigen_residual: Array
    eigen_condition_estimate: Array
    reciprocal_pairing_error: Array
    unit_circle_error: Array
    real_axis_distance: Array
    minimum_branch_margin: Array
    mode_positivity_minimum: Array
    spectrum_certified: Array
    valid: Array
    status: Array
    map_count: int = eqx.field(static=True)
    retained_scalar_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)


class ParaxialResonatorResult(StrictModule):
    """Closed orbit, Floquet spectrum, stability, and certified Gaussian mode."""

    round_trip_jacobian: Array
    round_trip_offset: Array
    closed_orbit: Array
    eigenvalues: Array
    phase_advances: Array
    stable: Array
    mode: ParaxialResonatorMode
    evidence: ParaxialResonatorEvidence
    status: Array

    @property
    def successful(self) -> Array:
        return self.evidence.valid


class ParaxialResonatorPlan(StrictModule, NonTrainableState):
    """Fixed route and numerical contract for one physical paraxial round trip."""

    maps: tuple[DifferentialRayMap, ...]
    closed_orbit_solve_plan: SmallLinearSolvePlan
    symplectic_tolerance: float = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)
    marginal_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    map_count: int = eqx.field(static=True)
    retained_scalar_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    frame_consistent: bool = eqx.field(static=True)
    coordinate_consistent: bool = eqx.field(static=True)
    provenance_consistent: bool = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maps: tuple[DifferentialRayMap, ...],
        /,
        *,
        symplectic_tolerance: float = 1e-7,
        closure_tolerance: float = 1e-9,
        marginal_tolerance: float = 1e-7,
        condition_limit: float = 1e12,
    ):
        if not isinstance(maps, tuple) or not maps:
            raise TypeError("maps must be a nonempty tuple of DifferentialRayMap values.")
        if any(not isinstance(ray_map, DifferentialRayMap) for ray_map in maps):
            raise TypeError("maps must contain only DifferentialRayMap values.")
        tolerances = (
            float(symplectic_tolerance),
            float(closure_tolerance),
            float(marginal_tolerance),
            float(condition_limit),
        )
        if any(not isfinite(value) or value <= 0.0 for value in tolerances[:3]):
            raise ValueError("Resonator tolerances must be finite and positive.")
        if not isfinite(tolerances[3]) or tolerances[3] <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        for ray_map in maps:
            input_reference = jnp.asarray(ray_map.input_reference)
            output_reference = jnp.asarray(ray_map.output_reference)
            jacobian = jnp.asarray(ray_map.jacobian)
            reference_shapes = (input_reference.shape, output_reference.shape)
            if reference_shapes != ((4,), (4,)) or jacobian.shape != (4, 4):
                raise ValueError(
                    "Each differential map must be an unbatched affine 4D ray map."
                )
            numeric_values = (
                input_reference,
                output_reference,
                jacobian,
                jnp.asarray(ray_map.branch_margin),
            )
            if any(
                not jnp.issubdtype(value.dtype, jnp.floating) for value in numeric_values
            ):
                raise TypeError(
                    "Paraxial resonator map coordinates and Jacobians must be real "
                    "floating-point values."
                )
            if any(
                jnp.asarray(value).shape != ()
                for value in (
                    ray_map.branch_margin,
                    ray_map.finite,
                    ray_map.valid,
                    ray_map.status,
                )
            ):
                raise ValueError("Differential-map evidence must be scalar.")
        frame_consistent = all(
            bool(ray_map.input_frame_id)
            and bool(ray_map.output_frame_id)
            and (ray_map.output_frame_id == maps[(index + 1) % len(maps)].input_frame_id)
            for index, ray_map in enumerate(maps)
        )
        coordinate_consistent = all(
            ray_map.coordinate_convention == _COORDINATE_CONVENTION for ray_map in maps
        )
        source_ids = tuple(ray_map.source_prepared_id for ray_map in maps)
        provenance_consistent = bool(source_ids[0]) and all(
            source_id == source_ids[0] for source_id in source_ids
        )
        scalar_count = len(maps) * (16 + 4 + 4 + 4) + 16 + 4 + 32
        dtype_itemsize = max(
            np.dtype(jnp.asarray(ray_map.jacobian).dtype).itemsize for ray_map in maps
        )
        complex_itemsize = 8 if dtype_itemsize <= 4 else 16
        runtime_workspace = (12 * 16 + 8 * 4) * complex_itemsize
        eigen_workspace = 16 * 16 * complex_itemsize
        workspace_bytes = max(runtime_workspace, eigen_workspace)
        self.maps = maps
        self.closed_orbit_solve_plan = SmallLinearSolvePlan(
            4,
            singular_tolerance=min(tolerances[1], 1e-12),
            maximum_condition=tolerances[3],
            refinement_iterations=2,
        )
        self.symplectic_tolerance = tolerances[0]
        self.closure_tolerance = tolerances[1]
        self.marginal_tolerance = tolerances[2]
        self.condition_limit = tolerances[3]
        self.map_count = len(maps)
        self.retained_scalar_count = scalar_count
        self.workspace_bytes = workspace_bytes
        self.frame_consistent = frame_consistent
        self.coordinate_consistent = coordinate_consistent
        self.provenance_consistent = provenance_consistent
        self.frame_id = maps[0].input_frame_id
        self.source_prepared_id = source_ids[0]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-paraxial-resonator-plan",
                "maps": tuple(
                    {
                        "input_frame": ray_map.input_frame_id,
                        "output_frame": ray_map.output_frame_id,
                        "source": ray_map.source_prepared_id,
                        "coordinate_convention": ray_map.coordinate_convention,
                        "arrays": array_tree_fingerprint(
                            (
                                ray_map.input_reference,
                                ray_map.output_reference,
                                ray_map.jacobian,
                            )
                        ),
                    }
                    for ray_map in maps
                ),
                "symplectic_tolerance": tolerances[0],
                "closure_tolerance": tolerances[1],
                "marginal_tolerance": tolerances[2],
                "condition_limit": tolerances[3],
            }
        )

    def prepare(self, /) -> PreparedParaxialResonator:
        return prepare_paraxial_resonator(self)


class PreparedParaxialResonator(StrictModule, NonTrainableState):
    """Composed affine round trip and prepared dense Floquet eigenspectrum."""

    plan: ParaxialResonatorPlan
    round_trip_jacobian: Array
    round_trip_offset: Array
    eigen_result: GeneralEigenSolveResult
    mode_indices: Array
    prepared_id: str = eqx.field(static=True)

    def execute(self, /) -> ParaxialResonatorResult:
        return analyze_paraxial_resonator(self)


def _compose_affine_maps(
    maps: tuple[DifferentialRayMap, ...],
    /,
) -> tuple[Array, Array]:
    dtype = jnp.result_type(*(jnp.asarray(ray_map.jacobian) for ray_map in maps), 0.0)
    jacobian = jnp.eye(4, dtype=dtype)
    offset = jnp.zeros((4,), dtype=dtype)
    for ray_map in maps:
        local_jacobian = jnp.asarray(ray_map.jacobian, dtype=dtype)
        local_offset = jnp.asarray(ray_map.output_reference, dtype=dtype) - contract(
            "ij,j->i",
            local_jacobian,
            jnp.asarray(ray_map.input_reference, dtype=dtype),
        )
        offset = contract("ij,j->i", local_jacobian, offset) + local_offset
        jacobian = contract("ij,jk->ik", local_jacobian, jacobian)
    return jacobian, offset


def _ordered_positive_mode_indices(
    eigenvalues: Array,
    eigenvectors: Array,
    /,
) -> Array:
    values = np.asarray(eigenvalues)
    vectors = np.asarray(eigenvectors)
    symplectic_form = np.asarray(
        (
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
        ),
        dtype=vectors.dtype,
    )
    signatures = np.real(-1j * np.diag(np.conj(vectors.T) @ symplectic_form @ vectors))
    scores = np.nan_to_num(signatures, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    selected = np.argsort(-scores, kind="stable")[:2]
    phase = np.abs(np.angle(values[selected]))
    selected = selected[np.argsort(phase, kind="stable")]
    return jnp.asarray(selected, dtype=jnp.int32)


def prepare_paraxial_resonator(
    plan: ParaxialResonatorPlan,
    /,
) -> PreparedParaxialResonator:
    """Compose a physical fixed route and prepare its complete Floquet spectrum."""
    if not isinstance(plan, ParaxialResonatorPlan):
        raise TypeError("plan must be a ParaxialResonatorPlan.")
    jacobian, offset = _compose_affine_maps(plan.maps)
    if bool(np.all(np.isfinite(np.asarray(jacobian)))):
        eigen_jacobian = jacobian
    else:
        phases = np.asarray((0.37, 0.71))
        cosine = np.diag(np.cos(phases))
        sine = np.diag(np.sin(phases))
        eigen_jacobian = jnp.asarray(
            np.block([[cosine, sine], [-sine, cosine]]),
            dtype=jacobian.dtype,
        )
    eigen_policy = GeneralEigenSolvePolicy(
        DenseSchurQZ(),
        selection=GeneralEigenSelection.all(),
        resources=GeneralEigenResourcePolicy(
            max_dimension=4,
            preparation_bytes=max(plan.workspace_bytes, 4096),
            workspace_bytes=max(plan.workspace_bytes, 4096),
            krylov_basis_bytes=0,
            operator_matvecs=0,
        ),
    )
    eigen_result = general_eigensolve(
        GeneralEigenproblem(
            DenseLinearOperator(
                eigen_jacobian,
                operator_id=f"{plan.plan_id}:round-trip-jacobian",
            ),
            problem_id=f"{plan.plan_id}:floquet",
        ),
        policy=eigen_policy,
    )
    indices = _ordered_positive_mode_indices(
        eigen_result.eigenvalues,
        eigen_result.right_eigenvector_coordinates,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-coupled-paraxial-resonator",
            "plan": plan.plan_id,
            "eigen_prepared": eigen_result.provenance.prepared_id,
        }
    )
    return PreparedParaxialResonator(
        plan,
        jacobian,
        offset,
        eigen_result,
        indices,
        prepared_id=prepared_id,
    )


def _symplectic_form(dtype, /) -> Array:
    return jnp.asarray(
        (
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
        ),
        dtype=dtype,
    )


def _frobenius_norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(jnp.abs(value) ** 2))


def analyze_paraxial_resonator(
    prepared: PreparedParaxialResonator,
    /,
) -> ParaxialResonatorResult:
    """Analyze one prepared coupled round trip with fixed-shape device execution."""
    if not isinstance(prepared, PreparedParaxialResonator):
        raise TypeError("prepared must be a PreparedParaxialResonator.")
    plan = prepared.plan
    jacobian = jnp.asarray(prepared.round_trip_jacobian)
    offset = jnp.asarray(prepared.round_trip_offset)
    identity4 = jnp.eye(4, dtype=jacobian.dtype)
    closed_solve = solve_small_linear(
        plan.closed_orbit_solve_plan,
        identity4 - jacobian,
        offset,
    )
    closed_orbit = closed_solve.value
    closed_residual = jnp.sqrt(
        jnp.sum(
            jnp.abs(contract("ij,j->i", jacobian, closed_orbit) + offset - closed_orbit)
            ** 2
        )
    )
    symplectic_form = _symplectic_form(jacobian.dtype)
    symplectic_defect = (
        contract("ji,jk,kl->il", jacobian, symplectic_form, jacobian) - symplectic_form
    )
    symplectic_error = _frobenius_norm(symplectic_defect)

    eigenvalues = jnp.asarray(prepared.eigen_result.eigenvalues)
    eigenvectors = jnp.asarray(prepared.eigen_result.right_eigenvector_coordinates)
    eigen_defect = contract("ij,jk->ik", jacobian, eigenvectors) - (
        eigenvectors * eigenvalues[None, :]
    )
    eigen_scale = jnp.maximum(
        _frobenius_norm(jacobian) * _frobenius_norm(eigenvectors)
        + _frobenius_norm(eigenvectors * eigenvalues[None, :]),
        jnp.asarray(1.0, dtype=jacobian.dtype),
    )
    eigen_residual = _frobenius_norm(eigen_defect) / eigen_scale
    eigen_condition_estimate = jnp.max(
        prepared.eigen_result.diagnostics.eigenvalue_condition_estimates
    )
    reciprocal_products = jnp.abs(eigenvalues[:, None] * eigenvalues[None, :] - 1.0)
    reciprocal_products = jnp.where(
        jnp.eye(4, dtype=bool),
        jnp.inf,
        reciprocal_products,
    )
    reciprocal_pairing_error = jnp.max(jnp.min(reciprocal_products, axis=-1))
    moduli = jnp.abs(eigenvalues)
    unit_circle_error = jnp.max(jnp.abs(moduli - 1.0))
    real_axis_distance = jnp.min(jnp.abs(jnp.imag(eigenvalues)))
    roundoff = 64.0 * jnp.finfo(jacobian.dtype).eps
    symplectic_limit = jnp.maximum(plan.symplectic_tolerance, roundoff)
    closure_limit = jnp.maximum(plan.closure_tolerance, roundoff)
    marginal_limit = jnp.maximum(plan.marginal_tolerance, roundoff)
    unit_circle = unit_circle_error <= marginal_limit
    marginal = unit_circle & (real_axis_distance <= marginal_limit)
    unstable = ~unit_circle
    spectrally_stable = unit_circle & ~marginal

    selected_values = eigenvalues[prepared.mode_indices]
    selected = eigenvectors[:, prepared.mode_indices]
    complex_symplectic_form = symplectic_form.astype(selected.dtype)
    raw_positivity = -1j * contract(
        "ji,jk,kl->il",
        jnp.conj(selected),
        complex_symplectic_form,
        selected,
    )
    raw_positivity = 0.5 * (
        raw_positivity + jnp.conj(jnp.swapaxes(raw_positivity, -1, -2))
    )
    positivity_spectrum = HermitianSpectrum(
        raw_positivity,
        tolerance=plan.marginal_tolerance,
    )
    safe_positivity = jnp.where(
        positivity_spectrum.eigenvalues > marginal_limit,
        positivity_spectrum.eigenvalues,
        1.0,
    )
    whitening = (
        positivity_spectrum.eigenvectors * jax.lax.rsqrt(safe_positivity)[None, :]
    ) @ jnp.conj(positivity_spectrum.eigenvectors.T)
    lagrangian_candidate = contract("ij,jk->ik", selected, whitening)
    transported_candidate = contract("ij,jk->ik", jacobian, lagrangian_candidate)
    normalized_positivity = -1j * contract(
        "ji,jk,kl->il",
        jnp.conj(lagrangian_candidate),
        complex_symplectic_form,
        lagrangian_candidate,
    )
    round_trip_action_candidate = -1j * contract(
        "ji,jk,kl->il",
        jnp.conj(lagrangian_candidate),
        complex_symplectic_form,
        transported_candidate,
    )
    invariance_residual = _frobenius_norm(
        transported_candidate
        - contract(
            "ij,jk->ik",
            lagrangian_candidate,
            round_trip_action_candidate,
        )
    ) / jnp.maximum(_frobenius_norm(transported_candidate), 1.0)
    lagrangian_defect = contract(
        "ji,jk,kl->il",
        lagrangian_candidate,
        complex_symplectic_form,
        lagrangian_candidate,
    )
    lagrangian_residual = _frobenius_norm(lagrangian_defect) / jnp.maximum(
        _frobenius_norm(lagrangian_candidate) ** 2,
        1.0,
    )
    unitarity_residual = _frobenius_norm(
        jnp.conj(round_trip_action_candidate.T) @ round_trip_action_candidate
        - jnp.eye(2, dtype=selected.dtype)
    )
    h = lagrangian_candidate[:2, :]
    u = lagrangian_candidate[2:, :]
    curvature_solve = solve_small_linear(
        SmallLinearSolvePlan(
            2,
            singular_tolerance=min(plan.closure_tolerance, 1e-12),
            maximum_condition=plan.condition_limit,
            refinement_iterations=1,
        ),
        jnp.swapaxes(h, -1, -2),
        jnp.swapaxes(u, -1, -2),
    )
    curvature_candidate = jnp.swapaxes(curvature_solve.value, -1, -2)
    curvature_symmetry_residual = _frobenius_norm(
        curvature_candidate - curvature_candidate.T
    ) / jnp.maximum(_frobenius_norm(curvature_candidate), 1.0)
    positivity_minimum = positivity_spectrum.minimum_eigenvalue

    map_valid = jnp.all(
        jnp.stack(
            tuple(
                jnp.asarray(ray_map.valid, dtype=bool)
                & (
                    jnp.asarray(ray_map.status, dtype=jnp.int32)
                    == int(ParaxialOpticsStatus.SUCCESS)
                )
                & (jnp.asarray(ray_map.branch_margin) > 0.0)
                for ray_map in plan.maps
            )
        )
    )
    map_finite = jnp.all(
        jnp.stack(
            tuple(
                jnp.asarray(ray_map.finite, dtype=bool)
                & jnp.all(jnp.isfinite(ray_map.input_reference))
                & jnp.all(jnp.isfinite(ray_map.output_reference))
                & jnp.all(jnp.isfinite(ray_map.jacobian))
                & jnp.isfinite(ray_map.branch_margin)
                for ray_map in plan.maps
            )
        )
    )
    finite = (
        map_finite
        & jnp.all(jnp.isfinite(jacobian))
        & jnp.all(jnp.isfinite(offset))
        & jnp.all(jnp.isfinite(eigenvalues))
        & jnp.all(jnp.isfinite(eigenvectors))
        & jnp.isfinite(closed_residual)
        & jnp.isfinite(symplectic_error)
        & jnp.isfinite(eigen_residual)
        & jnp.isfinite(eigen_condition_estimate)
        & jnp.isfinite(reciprocal_pairing_error)
    )
    frame_consistent = jnp.asarray(plan.frame_consistent)
    coordinate_consistent = jnp.asarray(plan.coordinate_consistent)
    provenance_consistent = jnp.asarray(plan.provenance_consistent)
    compatible = (
        map_valid & frame_consistent & coordinate_consistent & provenance_consistent
    )
    symplectic = symplectic_error <= symplectic_limit
    closed = closed_solve.successful & (closed_residual <= closure_limit)
    eigen_successful = jnp.asarray(prepared.eigen_result.successful)
    spectrum_certified = (
        eigen_successful
        & (eigen_residual <= jnp.maximum(symplectic_limit, marginal_limit))
        & (eigen_condition_estimate <= plan.condition_limit)
        & (reciprocal_pairing_error <= marginal_limit)
    )
    mode_valid = (
        finite
        & compatible
        & symplectic
        & closed
        & spectrum_certified
        & spectrally_stable
        & positivity_spectrum.valid
        & (positivity_minimum > marginal_limit)
        & curvature_solve.successful
        & (lagrangian_residual <= symplectic_limit)
        & (invariance_residual <= closure_limit)
        & (unitarity_residual <= symplectic_limit)
        & (curvature_symmetry_residual <= symplectic_limit)
        & jnp.all(jnp.isfinite(normalized_positivity))
    )
    zero_lagrangian = jnp.zeros((4, 2), dtype=selected.dtype)
    zero_square = jnp.zeros((2, 2), dtype=selected.dtype)
    mode = ParaxialResonatorMode(
        jnp.where(mode_valid, lagrangian_candidate, zero_lagrangian),
        jnp.where(mode_valid, curvature_candidate, zero_square),
        jnp.where(mode_valid, round_trip_action_candidate, zero_square),
        jnp.where(
            mode_valid,
            jnp.real(jnp.diag(normalized_positivity)),
            jnp.zeros((2,), dtype=jacobian.dtype),
        ),
        lagrangian_residual,
        invariance_residual,
        unitarity_residual,
        curvature_symmetry_residual,
        mode_valid,
        frame_id=plan.frame_id,
        source_prepared_id=plan.source_prepared_id,
        resonator_id=plan.plan_id,
        coordinate_convention=_COORDINATE_CONVENTION,
    )
    phase_advances = jnp.arccos(
        jnp.clip(
            jnp.real(selected_values) / jnp.maximum(jnp.abs(selected_values), 1e-30),
            -1.0,
            1.0,
        )
    )

    status = jnp.asarray(int(ParaxialResonatorStatus.INVALID_MODE), dtype=jnp.int32)
    status = jnp.where(
        spectrally_stable & mode_valid,
        int(ParaxialResonatorStatus.SUCCESS_STABLE),
        status,
    )
    status = jnp.where(
        unstable & spectrum_certified,
        int(ParaxialResonatorStatus.SUCCESS_UNSTABLE),
        status,
    )
    status = jnp.where(
        marginal & spectrum_certified,
        int(ParaxialResonatorStatus.MARGINAL),
        status,
    )
    status = jnp.where(
        ~eigen_successful,
        int(ParaxialResonatorStatus.INVALID_MODE),
        status,
    )
    status = jnp.where(
        ~closed,
        int(ParaxialResonatorStatus.SINGULAR_CLOSED_ORBIT),
        status,
    )
    status = jnp.where(
        ~symplectic,
        int(ParaxialResonatorStatus.NONSYMPLECTIC),
        status,
    )
    status = jnp.where(
        ~compatible,
        int(ParaxialResonatorStatus.INCOMPATIBLE_MAPS),
        status,
    )
    status = jnp.where(
        ~finite,
        int(ParaxialResonatorStatus.NONFINITE),
        status,
    )
    valid = (status == int(ParaxialResonatorStatus.SUCCESS_STABLE)) | (
        status == int(ParaxialResonatorStatus.SUCCESS_UNSTABLE)
    )
    minimum_branch_margin = jnp.min(
        jnp.stack(tuple(jnp.asarray(ray_map.branch_margin) for ray_map in plan.maps))
    )
    evidence = ParaxialResonatorEvidence(
        map_valid,
        frame_consistent,
        coordinate_consistent,
        provenance_consistent,
        finite,
        symplectic_error,
        closed_residual,
        closed_solve.condition_estimate,
        eigen_residual,
        eigen_condition_estimate,
        reciprocal_pairing_error,
        unit_circle_error,
        real_axis_distance,
        minimum_branch_margin,
        positivity_minimum,
        spectrum_certified,
        valid,
        status,
        map_count=plan.map_count,
        retained_scalar_count=plan.retained_scalar_count,
        workspace_bytes=plan.workspace_bytes,
        plan_id=plan.plan_id,
        prepared_id=prepared.prepared_id,
        source_prepared_id=plan.source_prepared_id,
        frame_id=plan.frame_id,
        coordinate_convention=_COORDINATE_CONVENTION,
    )
    return ParaxialResonatorResult(
        jacobian,
        offset,
        closed_orbit,
        eigenvalues,
        phase_advances,
        status == int(ParaxialResonatorStatus.SUCCESS_STABLE),
        mode,
        evidence,
        status,
    )


__all__ = [
    "ParaxialResonatorEvidence",
    "ParaxialResonatorMode",
    "ParaxialResonatorPlan",
    "ParaxialResonatorResult",
    "ParaxialResonatorStatus",
    "PreparedParaxialResonator",
    "analyze_paraxial_resonator",
    "prepare_paraxial_resonator",
]
