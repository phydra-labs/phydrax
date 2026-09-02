#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy import special

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import PeriodicCell
from ....discretization.bem._rwg import RWGSurfaceCurrentSpace3D
from ....linalg import DenseLinearOperator
from ._maxwell3d import _point_triangle_distance
from ._periodic_core3d import (
    _ewald_green_host,
    _integer_cube,
    _reduced_bloch_wavevector,
    _require_periodic_cell_3d,
    periodic_reciprocal_vectors_3d,
    PeriodicEwaldPolicy3D,
    PeriodicScalarResourceError,
)
from ._periodic_helmholtz3d import _guard_nonwood_modes, _validated_wavenumber


class PeriodicVectorResourceError(PeriodicScalarResourceError):
    """A periodic vector preparation exceeded a declared hard resource limit."""


class PeriodicVectorCompatibilityError(ValueError):
    """A vector source space violates a required periodic compatibility law."""


class PeriodicMaxwellElectricFieldSupport3D(StrictModule, NonTrainableState):
    """Exact scope and finite Ewald evidence for one periodic vector action.

    The shell indicators measure the final retained real and reciprocal shells;
    they are convergence evidence, not certified truncation-error bounds.  The
    action is an off-surface field evaluator and deliberately carries no
    boundary trace, EFIE self-action, or boundary-solve capability.
    """

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    family: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    real_image_count: int = eqx.field(static=True)
    reciprocal_mode_count: int = eqx.field(static=True)
    dense_entries: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    certified_fractional_clearance: float = eqx.field(static=True)
    minimum_fractional_clearance: Array
    minimum_target_distance: Array
    maximum_charge_neutrality_defect: Array
    bloch_wavevector: Array
    scalar_real_shell_indicator: Array
    scalar_reciprocal_shell_indicator: Array
    dyadic_real_shell_indicator: Array
    dyadic_reciprocal_shell_indicator: Array
    finite: Array
    charge_neutrality_enforced: bool = eqx.field(static=True)
    exact_transpose: bool = eqx.field(static=True)
    exact_adjoint: bool = eqx.field(static=True)
    off_surface_field_action_supported: bool = eqx.field(static=True)
    boundary_self_action_supported: bool = eqx.field(static=True)
    boundary_solve_supported: bool = eqx.field(static=True)
    truncation_error_certified: bool = eqx.field(static=True)
    continuum_discretization_error_certified: bool = eqx.field(static=True)
    support_id: str = eqx.field(static=True)


class PeriodicMaxwellElectricFieldResult3D(StrictModule):
    """One periodic Maxwell electric field together with its support evidence."""

    electric_field: Array
    support: PeriodicMaxwellElectricFieldSupport3D
    successful: Array


class PeriodicMaxwellElectricFieldAction3D(StrictModule, NonTrainableState):
    """Dense off-surface periodic Maxwell dyadic action from RWG coefficients."""

    current_space: RWGSurfaceCurrentSpace3D
    cell: PeriodicCell
    targets: Array
    operator: DenseLinearOperator
    wavenumber: Array
    wave_impedance: Array
    support: PeriodicMaxwellElectricFieldSupport3D
    action_id: str = eqx.field(static=True)

    def electric_field(self, coefficients: ArrayLike, /) -> Array:
        return self.operator.mv(coefficients).reshape((self.targets.shape[0], 3))

    def evaluate(
        self, coefficients: ArrayLike, /
    ) -> PeriodicMaxwellElectricFieldResult3D:
        field = self.electric_field(coefficients)
        successful = self.support.finite & jnp.all(jnp.isfinite(field))
        return PeriodicMaxwellElectricFieldResult3D(
            electric_field=field,
            support=self.support,
            successful=successful,
        )

    def transpose_mv(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field, dtype=self.operator.matrix.dtype)
        expected = (self.targets.shape[0], 3)
        if values.shape != expected:
            raise ValueError(f"field must have shape {expected}.")
        return self.operator.transpose_mv(values.reshape(-1))

    def adjoint_mv(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field, dtype=self.operator.matrix.dtype)
        expected = (self.targets.shape[0], 3)
        if values.shape != expected:
            raise ValueError(f"field must have shape {expected}.")
        return self.operator.adjoint_mv(values.reshape(-1))


class _DyadicEwaldBlock:
    def __init__(
        self,
        value: np.ndarray,
        scalar_real_shell: float,
        scalar_reciprocal_shell: float,
        dyadic_real_shell: float,
        dyadic_reciprocal_shell: float,
    ):
        self.value = value
        self.scalar_real_shell = float(scalar_real_shell)
        self.scalar_reciprocal_shell = float(scalar_reciprocal_shell)
        self.dyadic_real_shell = float(dyadic_real_shell)
        self.dyadic_reciprocal_shell = float(dyadic_reciprocal_shell)


def _canonical_displacements(
    displacements: np.ndarray,
    cell: PeriodicCell,
    bloch_wavevector: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lattice = np.asarray(cell.vectors, dtype=float)
    inverse = np.asarray(cell.inverse_vectors, dtype=float)
    fractional = displacements @ inverse
    image_indices = np.floor(fractional + 0.5).astype(np.int64)
    translations = image_indices @ lattice
    reduced = displacements - translations
    phase = np.exp(1j * (translations @ bloch_wavevector))
    return reduced, phase


def _helmholtz_dyadic_ewald_block(
    displacements: np.ndarray,
    cell: PeriodicCell,
    bloch_wavevector: np.ndarray,
    policy: PeriodicEwaldPolicy3D,
    wavenumber: float,
    /,
) -> _DyadicEwaldBlock:
    """Differentiate the same finite Ewald scalar family analytically."""

    vectors = np.asarray(displacements, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or np.any(~np.isfinite(vectors)):
        raise ValueError(
            "Dyadic Ewald displacements must be finite with shape (count, 3)."
        )
    screening = complex(0.0, -wavenumber)
    scalar = _ewald_green_host(
        vectors,
        cell,
        bloch_wavevector,
        policy,
        screening,
        subtract_central_laplace=False,
        remove_zero_mode=False,
    )
    count = vectors.shape[0]
    identity = np.eye(3, dtype=np.complex128)
    hessian = np.zeros((count, 3, 3), dtype=np.complex128)
    real_shell_accumulator = np.zeros((count,), dtype=float)
    reciprocal_shell_accumulator = np.zeros((count,), dtype=float)

    lattice = np.asarray(cell.vectors, dtype=float)
    real_indices = _integer_cube(policy.real_cutoff)
    real_shifts = real_indices @ lattice
    real_phases = np.exp(1j * (real_shifts @ bloch_wavevector))
    real_shell = np.max(np.abs(real_indices), axis=1) == policy.real_cutoff
    eta = policy.splitting_parameter
    b = screening / (2.0 * eta)
    normalization = 8.0 * math.pi
    singular_scale = max(float(np.linalg.norm(lattice, ord=2)), 1.0)
    singular_tolerance = 64.0 * np.finfo(float).eps * singular_scale

    for mode, shift in enumerate(real_shifts):
        difference = vectors - shift[None, :]
        radius = np.linalg.norm(difference, axis=1)
        if np.any(radius <= singular_tolerance):
            raise ValueError(
                "Periodic Maxwell targets must not meet the RWG support or any image."
            )
        plus = np.exp(screening * radius) * special.erfc(eta * radius + b)
        minus = np.exp(-screening * radius) * special.erfc(eta * radius - b)
        numerator = plus + minus
        gaussian = 2.0 * eta / math.sqrt(math.pi) * np.exp(-((eta * radius) ** 2) - b * b)
        numerator_first = screening * (plus - minus) - 2.0 * gaussian
        numerator_second = (
            screening * screening * numerator + 4.0 * eta * eta * radius * gaussian
        )
        radial = numerator / (normalization * radius)
        radial_first = (
            numerator_first / radius - numerator / (radius * radius)
        ) / normalization
        radial_second = (
            numerator_second / radius
            - 2.0 * numerator_first / (radius * radius)
            + 2.0 * numerator / (radius * radius * radius)
        ) / normalization
        isotropic = radial_first / radius
        directional = radial_second - isotropic
        unit = difference / radius[:, None]
        outer = unit[:, :, None] * unit[:, None, :]
        hessian_term = (
            isotropic[:, None, None] * identity + directional[:, None, None] * outer
        )
        phased_hessian = real_phases[mode] * hessian_term
        hessian += phased_hessian
        if real_shell[mode]:
            dyadic_term = real_phases[mode] * (
                radial[:, None, None] * identity
                + hessian_term / (wavenumber * wavenumber)
            )
            real_shell_accumulator += np.linalg.norm(
                dyadic_term.reshape((count, 9)), axis=1
            )

    reciprocal = np.asarray(periodic_reciprocal_vectors_3d(cell), dtype=float)
    reciprocal_indices = _integer_cube(policy.reciprocal_cutoff)
    wavevectors = reciprocal_indices @ reciprocal + bloch_wavevector[None, :]
    wavevector_norm_squared = np.sum(wavevectors * wavevectors, axis=1)
    denominators = wavevector_norm_squared - wavenumber * wavenumber
    coefficients = np.exp(-denominators / (4.0 * eta * eta)) / (
        cell.volume * denominators
    )
    reciprocal_shell = (
        np.max(np.abs(reciprocal_indices), axis=1) == policy.reciprocal_cutoff
    )
    for mode, wavevector in enumerate(wavevectors):
        phase = np.exp(1j * (vectors @ wavevector))
        outer = wavevector[:, None] * wavevector[None, :]
        weighted = phase * coefficients[mode]
        hessian -= weighted[:, None, None] * outer[None, :, :]
        if reciprocal_shell[mode]:
            dyadic_coefficient = coefficients[mode] * (
                identity - outer / (wavenumber * wavenumber)
            )
            reciprocal_shell_accumulator += np.abs(phase) * float(
                np.linalg.norm(dyadic_coefficient.reshape(-1))
            )

    dyadic = scalar.value[:, None, None] * identity + hessian / (wavenumber * wavenumber)
    if np.any(~np.isfinite(dyadic)):
        raise ValueError("Periodic Maxwell dyadic Ewald preparation is non-finite.")
    return _DyadicEwaldBlock(
        dyadic,
        scalar.real_shell,
        scalar.reciprocal_shell,
        float(np.max(real_shell_accumulator, initial=0.0)),
        float(np.max(reciprocal_shell_accumulator, initial=0.0)),
    )


def _surface_fractional_clearance(
    current_space: RWGSurfaceCurrentSpace3D, cell: PeriodicCell, /
) -> float:
    vertices = np.asarray(current_space.surface.vertices, dtype=float)
    origin = np.asarray(cell.origin, dtype=float)
    inverse = np.asarray(cell.inverse_vectors, dtype=float)
    fractional = (vertices - origin) @ inverse
    return float(np.min(np.minimum(fractional, 1.0 - fractional)))


def _periodic_target_clearance(
    targets: np.ndarray,
    current_space: RWGSurfaceCurrentSpace3D,
    cell: PeriodicCell,
    /,
) -> float:
    lattice = np.asarray(cell.vectors, dtype=float)
    origin = np.asarray(cell.origin, dtype=float)
    inverse = np.asarray(cell.inverse_vectors, dtype=float)
    target_fractional = (targets - origin) @ inverse
    wrapped_targets = origin + (target_fractional - np.floor(target_fractional)) @ lattice
    triangles = np.asarray(current_space.surface.vertices, dtype=float)[
        np.asarray(current_space.surface.triangles)
    ]
    translations = np.asarray(cell.image_shifts, dtype=int) @ lattice
    return min(
        _point_triangle_distance(target, triangle + translation)
        for target in wrapped_targets
        for triangle in triangles
        for translation in translations
    )


def _charge_neutrality_defect(
    current_space: RWGSurfaceCurrentSpace3D,
    policy: PeriodicEwaldPolicy3D,
    /,
) -> float:
    areas = np.asarray(current_space.surface.face_areas, dtype=float)
    divergence = np.asarray(current_space.divergence_matrix, dtype=float)
    charges = areas @ divergence
    scales = areas @ np.abs(divergence)
    tolerances = (
        policy.neutrality_absolute_tolerance
        + policy.neutrality_relative_tolerance * scales
    )
    if np.any(np.abs(charges) > tolerances):
        raise PeriodicVectorCompatibilityError(
            "Periodic Maxwell RWG sources must be cell-charge neutral."
        )
    return float(np.max(np.abs(charges), initial=0.0))


def prepare_periodic_maxwell_electric_field_action_3d(
    current_space: RWGSurfaceCurrentSpace3D,
    targets: ArrayLike,
    cell: PeriodicCell,
    /,
    *,
    wavenumber: float,
    wave_impedance: float = 1.0,
    bloch_wavevector: ArrayLike | None = None,
    certified_fractional_clearance: float,
    minimum_clearance_h: float = 0.25,
    policy: PeriodicEwaldPolicy3D | None = None,
    numeric_version: str = "0",
) -> PeriodicMaxwellElectricFieldAction3D:
    r"""Prepare a non-Wood quasi-periodic Maxwell electric field action.

    This is the periodic counterpart of the bounded RWG centroid field action:
    for the exp(-i omega t) convention it applies
    ``i*k*Z*(I + grad grad/k**2) G_k^alpha`` to a closed-surface RWG current.
    ``G_k^alpha`` is the landed, guarded scalar Helmholtz Ewald family.  Its
    real-space radial terms and reciprocal modes are differentiated
    analytically, then canonical-cell reduction enforces the declared Bloch
    character exactly for translated fixed targets.  Preparation rejects Wood
    modes, non-neutral RWG source spaces, cell-touching surfaces, near/on-image
    targets, and every hard resource overrun before matrix construction.

    The returned object is only an off-surface field action.  It is not a
    periodic EFIE self operator and does not imply a periodic boundary solve.
    """

    if not isinstance(current_space, RWGSurfaceCurrentSpace3D):
        raise TypeError("current_space must be RWGSurfaceCurrentSpace3D.")
    _require_periodic_cell_3d(cell)
    selected = PeriodicEwaldPolicy3D() if policy is None else policy
    if not isinstance(selected, PeriodicEwaldPolicy3D):
        raise TypeError("policy must be PeriodicEwaldPolicy3D or None.")
    k = _validated_wavenumber(wavenumber)
    impedance = float(wave_impedance)
    if not math.isfinite(impedance) or impedance <= 0.0:
        raise ValueError("wave_impedance must be finite and positive.")
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    wood_guard_workspace = (2 * selected.reciprocal_cutoff + 1) ** 3 * 128
    if wood_guard_workspace > selected.max_preparation_workspace_bytes:
        raise PeriodicVectorResourceError(
            "Periodic Maxwell Wood-mode guard exceeds max_preparation_workspace_bytes."
        )
    _guard_nonwood_modes(cell, k, wavevector, selected)

    surface = current_space.surface
    points = np.asarray(targets, dtype=np.asarray(surface.vertices).dtype)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] == 0
        or np.any(~np.isfinite(points))
    ):
        raise ValueError("targets must be one nonempty finite array of shape (count, 3).")
    clearance_certificate = float(certified_fractional_clearance)
    clearance_ratio = float(minimum_clearance_h)
    if (
        not math.isfinite(clearance_certificate)
        or clearance_certificate <= 0.0
        or not math.isfinite(clearance_ratio)
        or clearance_ratio <= 0.0
    ):
        raise ValueError(
            "Periodic Maxwell clearance declarations must be finite and positive."
        )
    minimum_fractional_clearance = _surface_fractional_clearance(current_space, cell)
    if minimum_fractional_clearance < clearance_certificate:
        raise ValueError(
            "The RWG surface does not satisfy its certified fractional cell clearance."
        )

    target_count = int(points.shape[0])
    face_count = surface.face_count
    edge_count = current_space.size
    pair_count = target_count * face_count
    dense_entries = target_count * 3 * edge_count
    clearance_pair_count = pair_count * int(cell.image_shifts.shape[0])
    if dense_entries > selected.max_matrix_entries:
        raise PeriodicVectorResourceError(
            "Periodic Maxwell dense action exceeds max_matrix_entries."
        )
    if max(pair_count, clearance_pair_count) > selected.max_exception_pairs:
        raise PeriodicVectorResourceError(
            "Periodic Maxwell target/source work exceeds max_exception_pairs."
        )
    conservative_resident_bytes = (
        dense_entries * np.dtype(np.complex128).itemsize
        + (points.size + surface.vertices.size) * np.dtype(np.float64).itemsize
    )
    if conservative_resident_bytes > selected.max_resident_bytes:
        raise PeriodicVectorResourceError(
            "Periodic Maxwell action exceeds max_resident_bytes."
        )
    real_count = (2 * selected.real_cutoff + 1) ** 3
    reciprocal_count = (2 * selected.reciprocal_cutoff + 1) ** 3
    pair_block_size = min(selected.action_block_size, pair_count)
    preparation_workspace = (
        conservative_resident_bytes
        + pair_block_size * (real_count + reciprocal_count) * 128
        + pair_block_size * 1024
    )
    if preparation_workspace > selected.max_preparation_workspace_bytes:
        raise PeriodicVectorResourceError(
            "Periodic Maxwell Ewald action exceeds max_preparation_workspace_bytes."
        )

    neutrality_defect = _charge_neutrality_defect(current_space, selected)
    minimum_target_distance = _periodic_target_clearance(points, current_space, cell)
    maximum_edge_length = float(np.max(np.asarray(surface.edge_lengths)))
    if minimum_target_distance < clearance_ratio * maximum_edge_length:
        raise ValueError(
            "Periodic Maxwell targets are too close to the RWG surface or an image."
        )

    face_edges = np.asarray(surface.face_edges)
    local_basis = np.asarray(current_space.centroid_basis, dtype=np.complex128)
    face_basis = np.zeros((face_count, 3, edge_count), dtype=np.complex128)
    for face in range(face_count):
        for local_edge in range(3):
            face_basis[face, :, face_edges[face, local_edge]] = local_basis[
                face, local_edge
            ]
    areas = np.asarray(surface.face_areas, dtype=float)
    centroids = np.asarray(surface.face_centroids, dtype=float)
    displacements = (points[:, None, :] - centroids[None, :, :]).reshape((-1, 3))
    reduced_displacements, bloch_phases = _canonical_displacements(
        displacements, cell, wavevector
    )
    matrix = np.zeros((target_count, 3, edge_count), dtype=np.complex128)
    scalar_real_shell = 0.0
    scalar_reciprocal_shell = 0.0
    dyadic_real_shell = 0.0
    dyadic_reciprocal_shell = 0.0
    for start in range(0, pair_count, pair_block_size):
        stop = min(start + pair_block_size, pair_count)
        block = _helmholtz_dyadic_ewald_block(
            reduced_displacements[start:stop],
            cell,
            wavevector,
            selected,
            k,
        )
        dyadics = block.value * bloch_phases[start:stop, None, None]
        for local, pair in enumerate(range(start, stop)):
            target = pair // face_count
            face = pair % face_count
            matrix[target] += areas[face] * (dyadics[local] @ face_basis[face])
        scalar_real_shell = max(scalar_real_shell, block.scalar_real_shell)
        scalar_reciprocal_shell = max(
            scalar_reciprocal_shell, block.scalar_reciprocal_shell
        )
        dyadic_real_shell = max(dyadic_real_shell, block.dyadic_real_shell)
        dyadic_reciprocal_shell = max(
            dyadic_reciprocal_shell, block.dyadic_reciprocal_shell
        )
    flat_host = (1j * k * impedance * matrix).reshape((target_count * 3, edge_count))
    flat_matrix = selected.precision.accumulation(jnp.asarray(flat_host))
    finite = jnp.all(jnp.isfinite(flat_matrix))
    if not bool(finite):
        raise ValueError("Periodic Maxwell field matrix is non-finite.")
    resident_bytes = int(
        flat_matrix.size * flat_matrix.dtype.itemsize
        + points.size * points.dtype.itemsize
        + cell.vectors.size * cell.vectors.dtype.itemsize
        + cell.inverse_vectors.size * cell.inverse_vectors.dtype.itemsize
        + cell.origin.size * cell.origin.dtype.itemsize
    )
    if resident_bytes > selected.max_resident_bytes:
        raise PeriodicVectorResourceError(
            "Prepared periodic Maxwell action exceeds max_resident_bytes."
        )
    action_workspace = int((target_count * 3 + edge_count) * flat_matrix.dtype.itemsize)
    action_id = canonical_fingerprint(
        {
            "kind": "periodic-maxwell-electric-field-action-3d-v1",
            "source_space": current_space.space_id,
            "cell": cell.cell_id,
            "policy": selected.policy_id,
            "targets": array_tree_fingerprint(jnp.asarray(points)),
            "bloch": array_tree_fingerprint(jnp.asarray(wavevector)),
            "wavenumber": k,
            "wave_impedance": impedance,
            "clearance": (clearance_certificate, clearance_ratio),
            "numeric_version": str(numeric_version),
        }
    )
    operator = DenseLinearOperator(
        flat_matrix,
        source=current_space.vector_space,
        operator_id=action_id,
    )
    precision_description = (
        "SciPy/NumPy complex128 host analytic Ewald differentiation; JAX "
        f"{flat_matrix.dtype} dense action; integration policy "
        f"{selected.precision.policy_id}"
    )
    support_id = canonical_fingerprint(
        {
            "kind": "periodic-maxwell-electric-field-support-3d-v1",
            "action": action_id,
            "matrix": array_tree_fingerprint(flat_matrix),
        }
    )
    support = PeriodicMaxwellElectricFieldSupport3D(
        ambient_dimension=3,
        pde=(
            "source-free time-harmonic Maxwell electric field away from the "
            "periodic RWG current support, exp(-i omega t) convention"
        ),
        geometry=(
            "fixed targets separated from every image of one closed piecewise-"
            "planar RWG surface strictly inside an affine fully periodic 3D cell"
        ),
        formulation=(
            "i*k*Z*(I + grad grad/k^2) G_k^alpha applied by RWG centroid "
            "surface quadrature; analytic derivatives of the guarded scalar "
            "Helmholtz real/reciprocal Ewald family"
        ),
        provider=(
            "SciPy/NumPy bounded host Ewald differentiation and PHYDRA RWG/"
            "DenseLinearOperator fixed-shape JAX actions"
        ),
        precision=precision_description,
        resource_evidence=(
            f"dense {(target_count * 3)}x{edge_count} complex action; "
            f"{pair_count} target-face pairs in blocks of {pair_block_size}; "
            f"{real_count} real images and {reciprocal_count} reciprocal modes"
        ),
        error_evidence=(
            "exact periodic target-to-triangle clearance and final-shell "
            "indicators; RWG centroid quadrature and finite Ewald truncation "
            "have no continuum-certified error bound"
        ),
        non_goals=(
            "no on-surface trace or jump relation",
            "no periodic EFIE, MFIE, CFIE, or boundary self action",
            "no inference of boundary-solve support",
            "no Wood-mode or resonant resolvent",
            "no near-singular surface quadrature",
            "no adaptive or unbounded image allocation",
            "no truncation or continuum error certification",
        ),
        family="non-Wood quasi-periodic Maxwell electric Green dyadic",
        source_space_id=current_space.space_id,
        cell_id=cell.cell_id,
        policy_id=selected.policy_id,
        target_count=target_count,
        face_count=face_count,
        edge_count=edge_count,
        real_image_count=real_count,
        reciprocal_mode_count=reciprocal_count,
        dense_entries=dense_entries,
        resident_bytes=resident_bytes,
        preparation_workspace_bytes=preparation_workspace,
        action_workspace_bytes_per_rhs=action_workspace,
        certified_fractional_clearance=clearance_certificate,
        minimum_fractional_clearance=selected.precision.decision(
            minimum_fractional_clearance
        ),
        minimum_target_distance=selected.precision.decision(minimum_target_distance),
        maximum_charge_neutrality_defect=selected.precision.decision(neutrality_defect),
        bloch_wavevector=selected.precision.evaluation(jnp.asarray(wavevector)),
        scalar_real_shell_indicator=selected.precision.decision(scalar_real_shell),
        scalar_reciprocal_shell_indicator=selected.precision.decision(
            scalar_reciprocal_shell
        ),
        dyadic_real_shell_indicator=selected.precision.decision(dyadic_real_shell),
        dyadic_reciprocal_shell_indicator=selected.precision.decision(
            dyadic_reciprocal_shell
        ),
        finite=finite,
        charge_neutrality_enforced=True,
        exact_transpose=True,
        exact_adjoint=True,
        off_surface_field_action_supported=True,
        boundary_self_action_supported=False,
        boundary_solve_supported=False,
        truncation_error_certified=False,
        continuum_discretization_error_certified=False,
        support_id=support_id,
    )
    return PeriodicMaxwellElectricFieldAction3D(
        current_space=current_space,
        cell=cell,
        targets=jnp.asarray(points),
        operator=operator,
        wavenumber=jnp.asarray(k, dtype=surface.vertices.dtype),
        wave_impedance=jnp.asarray(impedance, dtype=surface.vertices.dtype),
        support=support,
        action_id=action_id,
    )


__all__ = [
    "PeriodicMaxwellElectricFieldAction3D",
    "PeriodicMaxwellElectricFieldResult3D",
    "PeriodicMaxwellElectricFieldSupport3D",
    "PeriodicVectorCompatibilityError",
    "PeriodicVectorResourceError",
    "prepare_periodic_maxwell_electric_field_action_3d",
]
