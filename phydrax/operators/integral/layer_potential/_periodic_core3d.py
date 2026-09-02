#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy import special

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import PeriodicCell
from ....geometry import MeshRegion
from ....integration import IntegrationPrecisionPolicy
from ._galerkin3d import (
    LaplaceSingleLayerDP0Galerkin3D,
    LaplaceSingleLayerDP0GalerkinPolicy3D,
    prepare_laplace_single_layer_dp0_3d,
)


_Family = Literal["modified-helmholtz", "laplace", "helmholtz"]
_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi


class PeriodicScalarResourceError(RuntimeError):
    """A periodic scalar preparation exceeded a declared hard resource limit."""


class PeriodicScalarCompatibilityError(ValueError):
    """A density is outside the compatibility subspace of a periodic PDE."""


def _require_periodic_cell_3d(cell: PeriodicCell, /) -> PeriodicCell:
    if not isinstance(cell, PeriodicCell):
        raise TypeError("cell must be a PeriodicCell.")
    if cell.rank != 3 or cell.ambient_dimension != 3 or not cell.fully_periodic:
        raise ValueError(
            "Periodic scalar 3D requires a fully periodic rank-3 PeriodicCell."
        )
    return cell


def periodic_reciprocal_vectors_3d(cell: PeriodicCell, /) -> Array:
    """Return row reciprocal vectors for one fully periodic rank-3 cell."""

    return _require_periodic_cell_3d(cell).reciprocal_vectors


def periodic_lattice_translation_3d(cell: PeriodicCell, index: ArrayLike, /) -> Array:
    """Map one integer lattice row index to a 3D physical translation."""

    cell_ = _require_periodic_cell_3d(cell)
    index_ = jnp.asarray(index)
    if index_.shape != (3,):
        raise ValueError("A lattice index must have shape (3,).")
    if not jnp.issubdtype(index_.dtype, jnp.integer):
        raise TypeError("A lattice index must have integer dtype.")
    return index_ @ cell_.vectors


def periodic_bloch_phase_3d(
    cell: PeriodicCell,
    index: ArrayLike,
    bloch_wavevector: ArrayLike,
    /,
) -> Array:
    """Return exp(i alpha·A n) for the scalar quasi-periodic convention."""

    wavevector = jnp.asarray(bloch_wavevector)
    if wavevector.shape != (3,) or not bool(jnp.all(jnp.isfinite(wavevector))):
        raise ValueError("bloch_wavevector must be a finite vector of shape (3,).")
    return jnp.exp(1j * (wavevector @ periodic_lattice_translation_3d(cell, index)))


class PeriodicEwaldPolicy3D(StrictModule, NonTrainableState):
    """Finite deterministic Ewald, quadrature, and allocation envelope.

    ``real_cutoff`` and ``reciprocal_cutoff`` retain integer cubes in their
    respective lattices. ``exact_image_cutoff`` identifies the central/near
    real-space images evaluated with the unsplit fundamental solution before a
    smooth Ewald complement is added. Reported shell indicators are convergence
    evidence, not certified truncation or continuum error bounds.
    """

    splitting_parameter: float = eqx.field(static=True)
    real_cutoff: int = eqx.field(static=True)
    reciprocal_cutoff: int = eqx.field(static=True)
    exact_image_cutoff: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    neutrality_absolute_tolerance: float = eqx.field(static=True)
    neutrality_relative_tolerance: float = eqx.field(static=True)
    wood_tolerance: float = eqx.field(static=True)
    action_block_size: int = eqx.field(static=True)
    max_exception_pairs: int = eqx.field(static=True)
    max_matrix_entries: int = eqx.field(static=True)
    max_preparation_workspace_bytes: int = eqx.field(static=True)
    max_resident_bytes: int = eqx.field(static=True)
    precision: IntegrationPrecisionPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        splitting_parameter: float = 2.0,
        real_cutoff: int = 3,
        reciprocal_cutoff: int = 4,
        exact_image_cutoff: int = 1,
        quadrature_order: int = 3,
        absolute_tolerance: float = 1.0e-7,
        relative_tolerance: float = 1.0e-6,
        neutrality_absolute_tolerance: float = 1.0e-10,
        neutrality_relative_tolerance: float = 1.0e-10,
        wood_tolerance: float = 1.0e-10,
        action_block_size: int = 16,
        max_exception_pairs: int = 100_000,
        max_matrix_entries: int = 1_000_000,
        max_preparation_workspace_bytes: int = 256 * 1024 * 1024,
        max_resident_bytes: int = 256 * 1024 * 1024,
        precision: IntegrationPrecisionPolicy | None = None,
    ):
        eta = float(splitting_parameter)
        real = int(real_cutoff)
        reciprocal = int(reciprocal_cutoff)
        exact = int(exact_image_cutoff)
        order = int(quadrature_order)
        tolerances = (
            float(absolute_tolerance),
            float(relative_tolerance),
            float(neutrality_absolute_tolerance),
            float(neutrality_relative_tolerance),
            float(wood_tolerance),
        )
        limits = (
            int(action_block_size),
            int(max_exception_pairs),
            int(max_matrix_entries),
            int(max_preparation_workspace_bytes),
            int(max_resident_bytes),
        )
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("splitting_parameter must be finite and positive.")
        if real < 0 or reciprocal < 0 or exact < 0 or exact > real:
            raise ValueError(
                "Ewald cutoffs must be nonnegative and exact_image_cutoff "
                "must not exceed real_cutoff."
            )
        if order < 2:
            raise ValueError("quadrature_order must be at least two.")
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Periodic scalar tolerances must be finite and nonnegative.")
        if any(value <= 0 for value in limits):
            raise ValueError("Periodic scalar resource limits must be positive.")
        precision_ = IntegrationPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, IntegrationPrecisionPolicy):
            raise TypeError("precision must be IntegrationPrecisionPolicy or None.")
        self.splitting_parameter = eta
        self.real_cutoff = real
        self.reciprocal_cutoff = reciprocal
        self.exact_image_cutoff = exact
        self.quadrature_order = order
        (
            self.absolute_tolerance,
            self.relative_tolerance,
            self.neutrality_absolute_tolerance,
            self.neutrality_relative_tolerance,
            self.wood_tolerance,
        ) = tolerances
        (
            self.action_block_size,
            self.max_exception_pairs,
            self.max_matrix_entries,
            self.max_preparation_workspace_bytes,
            self.max_resident_bytes,
        ) = limits
        self.precision = precision_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-policy-3d-v1",
                "eta": eta,
                "cutoffs": (real, reciprocal, exact),
                "quadrature_order": order,
                "tolerances": tolerances,
                "limits": limits,
                "precision": precision_.policy_id,
            }
        )


class PeriodicScalarReport3D(StrictModule, NonTrainableState):
    """Scientific scope plus finite-resource/error evidence for one preparation.

    This report is deliberately not a continuum certificate. It describes one
    3D scalar PDE, one watertight polyhedral inclusion geometry, one DP0
    Galerkin/Ewald formulation, SciPy/NumPy host preparation, fixed-shape JAX
    actions, requested/realized precision, allocation counts, and Ewald shell
    indicators. Its ``non_goals`` field excludes vector PDEs, open or
    cell-touching surfaces, Wood modes, and continuum error certification.
    """

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    gauge: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    family: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    real_image_count: int = eqx.field(static=True)
    reciprocal_mode_count: int = eqx.field(static=True)
    exact_image_count: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    action_workspace_bytes_per_rhs: int = eqx.field(static=True)
    certified_fractional_clearance: float = eqx.field(static=True)
    minimum_fractional_clearance: Array
    bloch_wavevector: Array
    real_shell_indicator: Array
    reciprocal_shell_indicator: Array
    central_quadrature_errors: Array
    finite: Array
    truncation_error_certified: bool = eqx.field(static=True)
    continuum_discretization_error_certified: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class PeriodicScalarDP0Operator3D(StrictModule, NonTrainableState):
    """Prepared strong DP0 scalar single-layer map on one closed 3D inclusion.

    The central singular Laplace part uses the existing surface DP0 Galerkin
    operator. ``smooth_weak_matrix`` contains the family-specific central
    regular remainder, exact declared near images, and smooth Ewald complement.
    Forward, algebraic transpose, and conjugate-adjoint actions are fixed-shape
    JAX numeric actions. Laplace at zero Bloch vector is defined only on the
    reported neutral DP0 subspace and uses the reported zero-mode gauge.
    """

    central_galerkin: LaplaceSingleLayerDP0Galerkin3D
    cell: PeriodicCell
    policy: PeriodicEwaldPolicy3D
    smooth_weak_matrix: Array
    inverse_face_areas: Array
    face_areas: Array
    report: PeriodicScalarReport3D
    require_neutrality: bool = eqx.field(static=True)
    neutrality_absolute_tolerance: float = eqx.field(static=True)
    neutrality_relative_tolerance: float = eqx.field(static=True)

    @property
    def face_count(self) -> int:
        return self.report.face_count

    def _source(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        if value.shape != (self.face_count,):
            raise ValueError(
                f"A periodic DP0 source must have shape ({self.face_count},)."
            )
        if self.require_neutrality:
            charge = self.face_areas @ value
            scale = jnp.sum(self.face_areas * jnp.abs(value))
            tolerance = (
                self.neutrality_absolute_tolerance
                + self.neutrality_relative_tolerance * scale
            )
            if not bool(jnp.abs(charge) <= tolerance):
                raise PeriodicScalarCompatibilityError(
                    "The zero-Bloch periodic Laplace single layer requires "
                    "zero total DP0 charge."
                )
        return value

    def _target(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        if value.shape != (self.face_count,):
            raise ValueError(
                f"A periodic DP0 target must have shape ({self.face_count},)."
            )
        return value

    def _central_action(self, value: Array, /, *, transpose: bool) -> Array:
        operator = self.central_galerkin.weak_operator
        dtype = operator.source.structure().dtype
        action = operator.transpose_mv if transpose else operator.mv
        real_output = action(jnp.asarray(jnp.real(value), dtype=dtype))
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            return real_output
        imaginary_output = action(jnp.asarray(jnp.imag(value), dtype=dtype))
        return real_output + 1j * imaginary_output

    def weak_mv(self, vector: ArrayLike, /) -> Array:
        value = self._source(vector)
        return (
            self._central_action(value, transpose=False) + self.smooth_weak_matrix @ value
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.weak_mv(vector) * self.inverse_face_areas

    def weak_transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self._target(vector)
        return (
            self._central_action(value, transpose=True)
            + self.smooth_weak_matrix.T @ value
        )

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self._target(vector) * self.inverse_face_areas
        return (
            self._central_action(value, transpose=True)
            + self.smooth_weak_matrix.T @ value
        )

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self._target(vector) * self.inverse_face_areas
        return (
            self._central_action(value, transpose=True)
            + self.smooth_weak_matrix.conj().T @ value
        )


class _EwaldEvaluation:
    def __init__(self, value: np.ndarray, real_shell: float, reciprocal_shell: float):
        self.value = value
        self.real_shell = float(real_shell)
        self.reciprocal_shell = float(reciprocal_shell)


def _integer_cube(cutoff: int) -> np.ndarray:
    axis = np.arange(-cutoff, cutoff + 1, dtype=np.int64)
    first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((first.ravel(), second.ravel(), third.ravel()), axis=1)


def _validated_bloch_wavevector(bloch_wavevector: ArrayLike | None) -> np.ndarray:
    value = (
        np.zeros((3,), dtype=float)
        if bloch_wavevector is None
        else np.asarray(bloch_wavevector, dtype=float)
    )
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError("bloch_wavevector must be a finite vector of shape (3,).")
    return value


def _reduced_bloch_wavevector(
    cell: PeriodicCell, bloch_wavevector: ArrayLike | None, /
) -> np.ndarray:
    cell_ = _require_periodic_cell_3d(cell)
    value = _validated_bloch_wavevector(bloch_wavevector)
    reciprocal = np.asarray(periodic_reciprocal_vectors_3d(cell_), dtype=float)
    coordinates = np.linalg.solve(reciprocal.T, value)
    reduced_coordinates = coordinates - np.floor(coordinates + 0.5)
    return reduced_coordinates @ reciprocal


def _screened_real_split(
    radius: np.ndarray, screening: complex, eta: float
) -> np.ndarray:
    b = screening / (2.0 * eta)
    return (
        np.exp(screening * radius) * special.erfc(eta * radius + b)
        + np.exp(-screening * radius) * special.erfc(eta * radius - b)
    ) / (8.0 * math.pi * radius)


def _screened_smooth_at_zero(screening: complex, eta: float) -> complex:
    b = screening / (2.0 * eta)
    return -screening * special.erfc(b) / (4.0 * math.pi) + eta * np.exp(-(b * b)) / (
        2.0 * math.pi**1.5
    )


def _ewald_green_host(
    displacements: np.ndarray,
    cell: PeriodicCell,
    bloch_wavevector: np.ndarray,
    policy: PeriodicEwaldPolicy3D,
    screening: complex,
    /,
    *,
    subtract_central_laplace: bool,
    remove_zero_mode: bool,
) -> _EwaldEvaluation:
    vectors = np.asarray(displacements, dtype=float)
    if vectors.shape[-1:] != (3,):
        raise ValueError("Ewald displacements must have trailing shape (3,).")
    original_shape = vectors.shape[:-1]
    flat = vectors.reshape((-1, 3))
    cell_ = _require_periodic_cell_3d(cell)
    lattice = np.asarray(cell_.vectors, dtype=float)
    inverse_lattice = np.asarray(cell_.inverse_vectors, dtype=float)
    reciprocal = np.asarray(periodic_reciprocal_vectors_3d(cell_), dtype=float)
    eta = policy.splitting_parameter
    real_count = (2 * policy.real_cutoff + 1) ** 3
    reciprocal_count = (2 * policy.reciprocal_cutoff + 1) ** 3
    workspace_bytes = (
        flat.shape[0] * (real_count + reciprocal_count) * 128
        + (real_count + reciprocal_count) * 64
    )
    if workspace_bytes > policy.max_preparation_workspace_bytes:
        raise PeriodicScalarResourceError(
            "Periodic Ewald evaluation exceeds max_preparation_workspace_bytes."
        )
    if not subtract_central_laplace:
        fractional = flat @ inverse_lattice
        nearest_images = np.rint(fractional) @ lattice
        singular_scale = max(float(np.linalg.norm(lattice, ord=2)), 1.0)
        singular_tolerance = 64.0 * np.finfo(float).eps * singular_scale
        if np.any(np.linalg.norm(flat - nearest_images, axis=1) <= singular_tolerance):
            raise ValueError(
                "The periodic scalar Green function is singular at a source image."
            )

    real_indices = _integer_cube(policy.real_cutoff)
    real_shifts = real_indices @ lattice
    real_phase = np.exp(1j * (real_shifts @ bloch_wavevector))
    near = np.max(np.abs(real_indices), axis=1) <= policy.exact_image_cutoff
    central = np.all(real_indices == 0, axis=1)
    real_shell_mask = np.max(np.abs(real_indices), axis=1) == policy.real_cutoff

    differences = flat[:, None, :] - real_shifts[None, :, :]
    radii = np.linalg.norm(differences, axis=-1)
    if not subtract_central_laplace and np.any(radii == 0.0):
        raise ValueError(
            "The periodic scalar Green function is singular at a source image."
        )

    values = np.zeros((flat.shape[0],), dtype=np.complex128)
    real_terms = np.zeros_like(radii, dtype=np.complex128)
    positive = radii > 0.0
    if np.any(positive):
        real_terms[positive] = _screened_real_split(radii[positive], screening, eta)

    far = ~near
    if np.any(far):
        values += np.sum(real_terms[:, far] * real_phase[None, far], axis=1)

    exact_terms = np.zeros_like(radii, dtype=np.complex128)
    exact_terms[positive] = np.exp(-screening * radii[positive]) / (
        _FOUR_PI * radii[positive]
    )
    h_terms = exact_terms - real_terms
    zero = ~positive
    if np.any(zero):
        h_terms[zero] = _screened_smooth_at_zero(screening, eta)

    noncentral_near = near & ~central
    if np.any(noncentral_near):
        values += np.sum(
            exact_terms[:, noncentral_near] * real_phase[None, noncentral_near],
            axis=1,
        )
    central_radius = radii[:, central][:, 0]
    if subtract_central_laplace:
        regular = np.empty_like(central_radius, dtype=np.complex128)
        central_positive = central_radius > 0.0
        regular[central_positive] = np.expm1(
            -screening * central_radius[central_positive]
        ) / (_FOUR_PI * central_radius[central_positive])
        regular[~central_positive] = -screening / _FOUR_PI
        values += regular
    else:
        values += exact_terms[:, central][:, 0]
    values -= np.sum(h_terms[:, near] * real_phase[None, near], axis=1)

    reciprocal_indices = _integer_cube(policy.reciprocal_cutoff)
    modes = reciprocal_indices @ reciprocal + bloch_wavevector[None, :]
    mode_norm_squared = np.sum(modes * modes, axis=1)
    denominators = mode_norm_squared + screening * screening
    zero_mode = np.abs(denominators) <= 64.0 * np.finfo(float).eps
    if not remove_zero_mode and np.any(zero_mode):
        raise ValueError(
            "A retained reciprocal mode has a zero scalar resolvent denominator."
        )
    active = ~zero_mode if remove_zero_mode else np.ones_like(zero_mode, dtype=bool)
    reciprocal_coefficients = np.zeros(denominators.shape, dtype=np.complex128)
    reciprocal_coefficients[active] = np.exp(
        -denominators[active] / (4.0 * eta * eta)
    ) / (cell.volume * denominators[active])
    reciprocal_phase = np.exp(1j * (flat @ modes.T))
    values += reciprocal_phase @ reciprocal_coefficients
    if remove_zero_mode:
        values -= 1.0 / (4.0 * eta * eta * cell.volume)

    real_shell = float(
        np.max(
            np.sum(
                np.abs(real_terms[:, real_shell_mask]),
                axis=1,
            ),
            initial=0.0,
        )
    )
    reciprocal_shell_mask = (
        np.max(np.abs(reciprocal_indices), axis=1) == policy.reciprocal_cutoff
    )
    reciprocal_shell = float(
        np.sum(np.abs(reciprocal_coefficients[reciprocal_shell_mask]))
    )
    if (
        np.any(~np.isfinite(values))
        or not math.isfinite(real_shell)
        or not math.isfinite(reciprocal_shell)
    ):
        raise ValueError(
            "The bounded periodic Ewald evaluation produced non-finite evidence."
        )
    return _EwaldEvaluation(values.reshape(original_shape), real_shell, reciprocal_shell)


def _direct_screened_image_sum_host(
    displacement: np.ndarray,
    cell: PeriodicCell,
    screening: float,
    bloch_wavevector: np.ndarray,
    image_cutoff: int,
    max_image_count: int,
) -> complex:
    cell_ = _require_periodic_cell_3d(cell)
    vector = np.asarray(displacement, dtype=float)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise ValueError("displacement must be a finite vector of shape (3,).")
    cutoff = int(image_cutoff)
    if cutoff < 0:
        raise ValueError("image_cutoff must be nonnegative.")
    maximum = int(max_image_count)
    if maximum <= 0:
        raise ValueError("max_image_count must be positive.")
    image_count = (2 * cutoff + 1) ** 3
    if image_count > maximum:
        raise PeriodicScalarResourceError(
            "The direct Yukawa image cube exceeds max_image_count."
        )
    indices = _integer_cube(cutoff)
    shifts = indices @ np.asarray(cell_.vectors, dtype=float)
    radii = np.linalg.norm(vector[None, :] - shifts, axis=1)
    if np.any(radii == 0.0):
        raise ValueError("The Yukawa image sum is singular at a source image.")
    phases = np.exp(1j * (shifts @ bloch_wavevector))
    return complex(np.sum(phases * np.exp(-screening * radii) / (_FOUR_PI * radii)))


def _laplace_zero_bloch(cell: PeriodicCell, bloch_wavevector: np.ndarray) -> bool:
    reciprocal = np.asarray(periodic_reciprocal_vectors_3d(cell), dtype=float)
    coordinates = np.linalg.solve(reciprocal.T, bloch_wavevector)
    return bool(
        np.linalg.norm(coordinates - np.rint(coordinates), ord=np.inf)
        <= 64.0 * np.finfo(float).eps
    )


def _strict_fractional_clearance(region: MeshRegion, cell: PeriodicCell) -> float:
    vertices = np.asarray(region.triangle_mesh.vertices, dtype=float)
    origin = np.asarray(cell.origin, dtype=float)
    inverse = np.asarray(cell.inverse_vectors, dtype=float)
    fractional = (vertices - origin) @ inverse
    return float(np.min(np.minimum(fractional, 1.0 - fractional)))


def _build_smooth_weak_matrix(
    panelization,
    cell: PeriodicCell,
    bloch_wavevector: np.ndarray,
    policy: PeriodicEwaldPolicy3D,
    screening: complex,
    /,
    *,
    remove_zero_mode: bool,
) -> tuple[np.ndarray, float, float, int]:
    face_count = panelization.panel_count
    nodes_per_panel = panelization.nodes_per_panel
    points = np.asarray(panelization.points, dtype=float).reshape(
        (face_count, nodes_per_panel, 3)
    )
    weights = np.asarray(panelization.weights, dtype=float).reshape(
        (face_count, nodes_per_panel)
    )
    real_count = (2 * policy.real_cutoff + 1) ** 3
    reciprocal_count = (2 * policy.reciprocal_cutoff + 1) ** 3
    pair_points = nodes_per_panel * nodes_per_panel
    workspace_bytes = (
        face_count * face_count * np.dtype(np.complex128).itemsize
        + pair_points * (real_count + reciprocal_count) * 128
    )
    if workspace_bytes > policy.max_preparation_workspace_bytes:
        raise PeriodicScalarResourceError(
            "Periodic Ewald pair workspace exceeds max_preparation_workspace_bytes."
        )
    matrix = np.empty((face_count, face_count), dtype=np.complex128)
    maximum_real_shell = 0.0
    maximum_reciprocal_shell = 0.0
    for target in range(face_count):
        for source in range(face_count):
            differences = (
                points[target, :, None, :] - points[source, None, :, :]
            ).reshape((-1, 3))
            evaluated = _ewald_green_host(
                differences,
                cell,
                bloch_wavevector,
                policy,
                screening,
                subtract_central_laplace=True,
                remove_zero_mode=remove_zero_mode,
            )
            kernel = evaluated.value.reshape((nodes_per_panel, nodes_per_panel))
            matrix[target, source] = weights[target] @ kernel @ weights[source]
            maximum_real_shell = max(maximum_real_shell, evaluated.real_shell)
            maximum_reciprocal_shell = max(
                maximum_reciprocal_shell, evaluated.reciprocal_shell
            )
    return matrix, maximum_real_shell, maximum_reciprocal_shell, workspace_bytes


def _prepare_periodic_scalar_dp0_3d(
    region: MeshRegion,
    cell: PeriodicCell,
    /,
    *,
    family: _Family,
    screening: complex,
    bloch_wavevector: ArrayLike | None,
    policy: PeriodicEwaldPolicy3D | None,
    certified_fractional_clearance: float,
    pde: str,
    formulation: str,
    gauge: str,
    non_goals: tuple[str, ...],
    numeric_version: str,
) -> PeriodicScalarDP0Operator3D:
    if not isinstance(region, MeshRegion):
        raise TypeError("Periodic scalar DP0 preparation requires a MeshRegion.")
    _require_periodic_cell_3d(cell)
    selected = PeriodicEwaldPolicy3D() if policy is None else policy
    if not isinstance(selected, PeriodicEwaldPolicy3D):
        raise TypeError("policy must be PeriodicEwaldPolicy3D or None.")
    clearance_certificate = float(certified_fractional_clearance)
    if not math.isfinite(clearance_certificate) or clearance_certificate <= 0.0:
        raise ValueError("certified_fractional_clearance must be finite and positive.")
    minimum_clearance = _strict_fractional_clearance(region, cell)
    if minimum_clearance < clearance_certificate:
        raise ValueError(
            "The watertight inclusion does not satisfy its certified "
            "fractional cell clearance."
        )
    wavevector = _reduced_bloch_wavevector(cell, bloch_wavevector)
    remove_zero_mode = family == "laplace" and _laplace_zero_bloch(cell, wavevector)

    face_count = int(region.triangle_mesh.faces.shape[0])
    entries = face_count * face_count
    if entries > selected.max_matrix_entries:
        raise PeriodicScalarResourceError(
            "Periodic smooth-complement matrix exceeds max_matrix_entries."
        )
    smooth_host_bytes = entries * np.dtype(np.complex128).itemsize
    if smooth_host_bytes > selected.max_resident_bytes:
        raise PeriodicScalarResourceError(
            "Periodic smooth-complement matrix exceeds max_resident_bytes."
        )
    laplace_policy = LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=selected.quadrature_order,
        singular_order=selected.quadrature_order,
        near_order=selected.quadrature_order,
        near_ratio=1.0,
        near_max_depth=1,
        absolute_tolerance=selected.absolute_tolerance,
        relative_tolerance=selected.relative_tolerance,
        target_block_size=selected.action_block_size,
        source_block_size=selected.action_block_size,
        max_exception_pairs=selected.max_exception_pairs,
        max_preparation_workspace_bytes=selected.max_preparation_workspace_bytes,
        max_resident_bytes=selected.max_resident_bytes,
        precision=selected.precision,
    )
    central = prepare_laplace_single_layer_dp0_3d(
        region,
        policy=laplace_policy,
        numeric_version=numeric_version,
    )
    (
        smooth_host,
        real_shell,
        reciprocal_shell,
        ewald_workspace,
    ) = _build_smooth_weak_matrix(
        central.panelization,
        cell,
        wavevector,
        selected,
        screening,
        remove_zero_mode=remove_zero_mode,
    )
    smooth = selected.precision.accumulation(jnp.asarray(smooth_host))
    face_areas = selected.precision.accumulation(central.face_areas)
    inverse_areas = jnp.reciprocal(face_areas)
    smooth_bytes = int(smooth.size * smooth.dtype.itemsize)
    owned_state_bytes = int(
        (face_areas.size + inverse_areas.size) * face_areas.dtype.itemsize
        + cell.vectors.size * cell.vectors.dtype.itemsize
        + cell.inverse_vectors.size * cell.inverse_vectors.dtype.itemsize
        + cell.origin.size * cell.origin.dtype.itemsize
        + cell.periodic_mask.size * cell.periodic_mask.dtype.itemsize
        + cell.image_shifts.size * cell.image_shifts.dtype.itemsize
    )
    resident_bytes = (
        central.assembly_report.resident_bytes + smooth_bytes + owned_state_bytes
    )
    if resident_bytes > selected.max_resident_bytes:
        raise PeriodicScalarResourceError(
            "Periodic prepared operator exceeds max_resident_bytes."
        )
    action_workspace = central.assembly_report.action_workspace_bytes_per_rhs + int(
        3 * face_count * max(smooth.dtype.itemsize, face_areas.dtype.itemsize)
    )
    preparation_workspace = max(
        central.assembly_report.preparation_workspace_bytes,
        ewald_workspace,
    )
    finite = jnp.all(jnp.isfinite(smooth)) & central.assembly_report.finite
    if not bool(finite):
        raise ValueError("Periodic scalar preparation produced non-finite numeric state.")
    precision_description = (
        "SciPy/NumPy complex128 host Ewald; JAX "
        f"{smooth.dtype} accumulation/action; integration policy "
        f"{selected.precision.policy_id}"
    )
    report_id = canonical_fingerprint(
        {
            "kind": "periodic-scalar-dp0-report-3d-v1",
            "family": family,
            "cell": cell.cell_id,
            "policy": selected.policy_id,
            "binding": central.assembly_report.binding_id,
            "bloch": array_tree_fingerprint(jnp.asarray(wavevector)),
            "screening": (float(np.real(screening)), float(np.imag(screening))),
            "smooth": array_tree_fingerprint(smooth),
        }
    )
    report = PeriodicScalarReport3D(
        ambient_dimension=3,
        pde=pde,
        geometry=(
            "outward-oriented watertight polyhedral MeshRegion components, strictly "
            "inside one affine fully periodic rank-3 PeriodicCell; scalar facewise "
            "DP0 trial/test space"
        ),
        formulation=formulation,
        provider=(
            "SciPy/NumPy host Ewald preparation and existing PHYDRA DP0 "
            "Galerkin; fixed-shape JAX actions"
        ),
        precision=precision_description,
        gauge=gauge,
        non_goals=non_goals,
        family=family,
        cell_id=cell.cell_id,
        policy_id=selected.policy_id,
        binding_id=central.assembly_report.binding_id,
        face_count=central.face_count,
        component_count=central.component_count,
        real_image_count=(2 * selected.real_cutoff + 1) ** 3,
        reciprocal_mode_count=(2 * selected.reciprocal_cutoff + 1) ** 3,
        exact_image_count=(2 * selected.exact_image_cutoff + 1) ** 3,
        resident_bytes=resident_bytes,
        preparation_workspace_bytes=preparation_workspace,
        action_workspace_bytes_per_rhs=action_workspace,
        certified_fractional_clearance=clearance_certificate,
        minimum_fractional_clearance=selected.precision.decision(minimum_clearance),
        bloch_wavevector=selected.precision.evaluation(jnp.asarray(wavevector)),
        real_shell_indicator=selected.precision.decision(real_shell),
        reciprocal_shell_indicator=selected.precision.decision(reciprocal_shell),
        central_quadrature_errors=central.assembly_report.maximum_errors,
        finite=finite,
        truncation_error_certified=False,
        continuum_discretization_error_certified=False,
        report_id=report_id,
    )
    return PeriodicScalarDP0Operator3D(
        central_galerkin=central,
        cell=cell,
        policy=selected,
        smooth_weak_matrix=smooth,
        inverse_face_areas=inverse_areas,
        face_areas=face_areas,
        report=report,
        require_neutrality=remove_zero_mode,
        neutrality_absolute_tolerance=selected.neutrality_absolute_tolerance,
        neutrality_relative_tolerance=selected.neutrality_relative_tolerance,
    )


__all__ = [
    "PeriodicEwaldPolicy3D",
    "PeriodicScalarCompatibilityError",
    "PeriodicScalarDP0Operator3D",
    "PeriodicScalarReport3D",
    "periodic_bloch_phase_3d",
    "periodic_lattice_translation_3d",
    "periodic_reciprocal_vectors_3d",
    "PeriodicScalarResourceError",
]
