#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Discrete-dispersion audit of the executed compatible Maxwell update.

The audit linearizes the actual leapfrog step of a prepared runtime, extracts its
translation-invariant stencil inside a declared homogeneous region, and forms
the one-step Bloch map ``M(k)``. Mode multipliers ``λ = exp(-iωΔt)`` give the
numerical dispersion ``ω = i log(λ)/Δt`` under the ``exp(-iωt)`` convention.

`CherenkovRegimePlan` compares the resonance ``k·v = ω`` of a uniformly moving
source with the continuum index and with the numerical Bloch branches. It
separates *numerical Cherenkov radiation* (a physical source radiating into grid
modes slowed by discrete dispersion) from the *numerical Cherenkov instability*
of drifting PIC plasmas, which is an aliasing instability of the coupled
particle-field system owned by the spectral-PIC NCI analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from math import pi
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ..linalg import (
    DenseLinearOperator,
    determinant_small_linear,
    prepare_linearization,
    PreparedLinearization,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ..linalg.eigen import general_eigensolve, GeneralEigenproblem
from ..nonlinear import NonlinearTermination, scalar_root, ScalarRootProblem, TOMS748
from ._maxwell import (
    CompatibleMaxwellState,
    MaxwellAuxiliaryState,
    MaxwellPrimaryState,
    PreparedCompatibleMaxwell,
)
from ._maxwell_materials import (
    MagnetizedColdPlasmaState,
    PreparedMagnetizedColdPlasmaMaxwellConstitutive,
)
from ._maxwell_sources import MaxwellSourceForcing


# Dependency-radius bound, in cells, of one executed leapfrog step (curl
# applications plus the vertex/edge averaging of vertex-collocated currents);
# the global Bloch-wave check of every eroded cell verifies it.
_STENCIL_RADIUS = 3
_STENCIL_DEFECT_TOLERANCE = 1e-9
# Unit-modulus multipliers of defective blocks (e.g. static Drude currents)
# split by O(sqrt(machine epsilon)) in floating point.
_STABILITY_TOLERANCE = 1e-6


class MaxwellMaterialRegion(StrictModule, NonTrainableState):
    """Half-open cell-index box ``[lower, upper)`` declared materially homogeneous."""

    lower: tuple[int, ...] = eqx.field(static=True)
    upper: tuple[int, ...] = eqx.field(static=True)
    region_id: str = eqx.field(static=True)

    def __init__(self, lower: tuple[int, ...], upper: tuple[int, ...], /) -> None:
        lower_ = tuple(int(value) for value in lower)
        upper_ = tuple(int(value) for value in upper)
        if not lower_ or len(lower_) != len(upper_):
            raise ValueError("Material region bounds must be nonempty and aligned.")
        if any(low < 0 or high <= low for low, high in zip(lower_, upper_, strict=True)):
            raise ValueError("Material region requires 0 <= lower < upper per axis.")
        self.lower = lower_
        self.upper = upper_
        self.region_id = canonical_fingerprint(
            {"kind": "maxwell-material-region", "lower": lower_, "upper": upper_}
        )


class MaxwellDispersionResult(StrictModule):
    """Bloch multipliers and numerical frequencies at sampled wavevectors.

    Columns are ordered by ascending ``Re ω`` then ``Im ω``. ``modes[k, :, n]`` is
    the local amplitude vector of mode ``n`` in `CompatibleMaxwellDispersionAudit`
    local-type order. ``stable`` states ``max |λ| ≤ 1`` within tolerance.
    """

    wavevectors: Array
    multipliers: Array
    angular_frequencies: Array
    modes: Array
    spectral_radius: Array
    stable: Array
    status: Array
    step_size: Array


class _LocalType(StrictModule, NonTrainableState):
    leaf: int = eqx.field(static=True)
    component: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    orientation: int = eqx.field(static=True)


def _uniform_spacing(prepared: PreparedCompatibleMaxwell, /) -> np.ndarray:
    spacing = []
    for axis in prepared.plan.bridge.grid.structured_axes:
        widths = np.asarray(axis.interval_widths)
        if not np.allclose(widths, widths[0], rtol=1e-12, atol=0.0):
            raise ValueError("Dispersion audit requires uniform structured axes.")
        spacing.append(float(widths[0]))
    return np.asarray(spacing, dtype=np.float64)


class CompatibleMaxwellDispersionAudit(StrictModule):
    """Exact one-step Bloch map of the executed update in a homogeneous region.

    Local types enumerate electric orientations, magnetic orientations, and every
    auxiliary material component (``auxiliary_degrees`` order). The stencil is
    extracted from the linearized step at the region centre and verified against
    a global Bloch wave on every cell of the eroded region; ``stencil_defect``
    reports that relative mismatch. ``cyclotron_resonance_shift`` is the executed
    numerical gyration frequency minus ``|ω_c|`` per plasma species and
    ``cyclotron_step_phase`` is ``|ω_c|Δt`` (both empty for other materials).
    """

    prepared: PreparedCompatibleMaxwell
    region: MaxwellMaterialRegion
    step_size: Array
    spacing: Array
    local_types: tuple[_LocalType, ...]
    stencil_rows: Array
    stencil_columns: Array
    stencil_displacements: Array
    stencil_coefficients: Array
    reference_cell: tuple[int, ...] = eqx.field(static=True)
    stencil_defect: Array
    cyclotron_resonance_shift: Array
    cyclotron_step_phase: Array
    audit_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedCompatibleMaxwell,
        region: MaxwellMaterialRegion,
        step_size: ArrayLike,
        /,
    ) -> None:
        if not isinstance(prepared, PreparedCompatibleMaxwell):
            raise TypeError("Dispersion audit requires a PreparedCompatibleMaxwell.")
        if not isinstance(region, MaxwellMaterialRegion):
            raise TypeError("region must be a MaxwellMaterialRegion.")
        capabilities = prepared.capabilities
        if capabilities.nonlinear or not capabilities.linear_time_invariant:
            raise ValueError("Dispersion audit requires linear time-invariant dynamics.")
        if not prepared.magnetic_projection_elided or (
            prepared.harmonic_constraint is not None
        ):
            raise ValueError(
                "Dispersion audit requires a local update; magnetic projection is global."
            )
        dt = float(np.asarray(step_size))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Dispersion audit step_size must be finite and positive.")
        bridge = prepared.plan.bridge
        spacing = _uniform_spacing(prepared)
        cells = tuple(axis.interval_centers.size for axis in bridge.grid.structured_axes)
        if len(region.lower) != bridge.dimension:
            raise ValueError("Material region dimension does not match the grid.")
        if any(high > count for high, count in zip(region.upper, cells, strict=True)):
            raise ValueError("Material region exceeds the grid.")
        if any(
            high - low <= 2 * _STENCIL_RADIUS
            for low, high in zip(region.lower, region.upper, strict=True)
        ):
            raise ValueError(
                f"Material region must span more than {2 * _STENCIL_RADIUS} cells per axis."
            )
        # Axes that are periodic and fully covered wrap: every cell is verified
        # with a commensurate wavevector; other axes are eroded by the radius.
        wrapped = tuple(
            axis.periodic and low == 0 and high == count
            for axis, low, high, count in zip(
                bridge.grid.structured_axes,
                region.lower,
                region.upper,
                cells,
                strict=True,
            )
        )
        eroded_lower = tuple(
            low if wrap else low + _STENCIL_RADIUS
            for low, wrap in zip(region.lower, wrapped, strict=True)
        )
        eroded_upper = tuple(
            high if wrap else high - _STENCIL_RADIUS
            for high, wrap in zip(region.upper, wrapped, strict=True)
        )
        layout = prepared.layout
        material = prepared.constitutive
        material_state = material.initialize_state()
        material_leaves = jax.tree_util.tree_leaves(material_state)
        degrees = material.auxiliary_degrees
        if len(degrees) != len(material_leaves):
            raise ValueError("auxiliary_degrees must name one degree per state leaf.")
        leaf_degrees = (layout.electric_degree, layout.magnetic_degree, *degrees)
        zero_point = (
            jnp.zeros((layout.electric_count,), dtype=jnp.float64),
            jnp.zeros((layout.magnetic_count,), dtype=jnp.float64),
            jax.tree_util.tree_map(
                lambda leaf: jnp.zeros(leaf.shape, dtype=jnp.float64), material_state
            ),
        )
        flat_point, unravel = ravel_pytree(zero_point)
        leaf_shapes = tuple(leaf.shape for leaf in jax.tree_util.tree_leaves(zero_point))
        for shape, degree in zip(leaf_shapes, leaf_degrees, strict=True):
            if shape[-1] != bridge.cochain.cell_counts[degree]:
                raise ValueError(
                    "Auxiliary state leaves must end with their declared cochain axis."
                )
        leaf_offsets = np.cumsum(
            (0,) + tuple(int(np.prod(shape)) for shape in leaf_shapes)
        )
        local_types = tuple(
            _LocalType(leaf, component, degree, orientation)
            for leaf, (shape, degree) in enumerate(
                zip(leaf_shapes, leaf_degrees, strict=True)
            )
            for component in range(int(np.prod(shape[:-1])))
            for orientation in range(len(bridge.orientations[degree]))
        )
        step = _linear_step(prepared, jnp.asarray(dt), unravel)
        linearization = prepare_linearization(step, flat_point)
        reference = tuple(
            (low + high) // 2
            for low, high in zip(region.lower, region.upper, strict=True)
        )
        indexer = _TypeIndexer(bridge, leaf_shapes, leaf_offsets)
        coordinates = _entity_coordinates(bridge.cochain.coordinates)
        rows, columns, displacements, coefficients = _extract_stencil(
            linearization,
            local_types,
            indexer,
            coordinates,
            reference,
            flat_point.size,
        )
        self.prepared = prepared
        self.region = region
        self.step_size = jnp.asarray(dt, dtype=jnp.float64)
        self.spacing = jnp.asarray(spacing)
        self.local_types = local_types
        self.stencil_rows = jnp.asarray(rows)
        self.stencil_columns = jnp.asarray(columns)
        self.stencil_displacements = jnp.asarray(displacements)
        self.stencil_coefficients = jnp.asarray(coefficients)
        self.reference_cell = reference
        defect = _verify_stencil(
            (rows, columns, displacements, coefficients),
            local_types,
            linearization,
            indexer,
            coordinates,
            eroded_lower,
            eroded_upper,
            _verification_wavevector(spacing, cells, wrapped),
        )
        if not defect <= _STENCIL_DEFECT_TOLERANCE:
            raise ValueError(
                "The update is not translation invariant within the material region "
                f"(relative stencil defect {defect:.3e}); the region is heterogeneous "
                "or intersects a boundary, source support, or absorbing layer."
            )
        self.stencil_defect = jnp.asarray(defect)
        shift, phase = _cyclotron_resonance(prepared, dt)
        self.cyclotron_resonance_shift = shift
        self.cyclotron_step_phase = phase
        self.audit_id = canonical_fingerprint(
            {
                "kind": "compatible-maxwell-dispersion-audit",
                "prepared": prepared.prepared_id,
                "region": region.region_id,
                "step_size": dt,
                "stencil": array_tree_fingerprint(np.asarray(coefficients)),
            }
        )

    @property
    def local_dimension(self) -> int:
        return len(self.local_types)

    def bloch_matrix(self, wavevector: ArrayLike, /) -> Array:
        """One-step Bloch map ``M(k)`` of the local amplitudes (traceable)."""
        k = jnp.asarray(wavevector, dtype=jnp.float64)
        if k.shape != (self.prepared.plan.bridge.dimension,):
            raise ValueError("wavevector must have one component per grid axis.")
        return _bloch_matrix(
            self.stencil_rows,
            self.stencil_columns,
            self.stencil_displacements,
            self.stencil_coefficients,
            self.local_dimension,
            k,
        )

    def dispersion(self, wavevectors: ArrayLike, /) -> MaxwellDispersionResult:
        """Native dense eigen-analysis of ``M(k)`` at host-sampled wavevectors."""
        k = np.asarray(wavevectors, dtype=np.float64)
        dimension = self.prepared.plan.bridge.dimension
        if k.ndim != 2 or k.shape[1] != dimension or k.shape[0] == 0:
            raise ValueError(f"wavevectors must have shape (count, {dimension}).")
        dt = float(np.asarray(self.step_size))
        multipliers, frequencies, modes, statuses = [], [], [], []
        for row in k:
            solved = general_eigensolve(
                GeneralEigenproblem(
                    DenseLinearOperator(self.bloch_matrix(row)),
                    problem_id="maxwell-bloch-step-map",
                )
            )
            values = np.asarray(solved.eigenvalues)
            vectors = np.asarray(solved.right_eigenvector_coordinates)
            omega = 1j * np.log(values) / dt
            order = np.lexsort((omega.imag, omega.real))
            multipliers.append(values[order])
            frequencies.append(omega[order])
            modes.append(vectors[:, order])
            statuses.append(int(np.asarray(solved.status)))
        multiplier_array = np.stack(multipliers)
        radius = np.max(np.abs(multiplier_array), axis=1)
        return MaxwellDispersionResult(
            wavevectors=jnp.asarray(k),
            multipliers=jnp.asarray(multiplier_array),
            angular_frequencies=jnp.asarray(np.stack(frequencies)),
            modes=jnp.asarray(np.stack(modes)),
            spectral_radius=jnp.asarray(radius),
            stable=jnp.asarray(radius <= 1.0 + _STABILITY_TOLERANCE),
            status=jnp.asarray(np.asarray(statuses, dtype=np.int32)),
            step_size=self.step_size,
        )

    def branch_frequencies(self, wavevector: ArrayLike, /) -> Array:
        """Upper half of the numerical frequencies sorted by ``Re ω`` (traceable).

        Leapfrog maps pair forward and backward waves, so the upper half holds
        every positive-frequency propagating branch; static and purely damped
        branches sit at ``Re ω = 0`` and never satisfy a positive resonance.
        """
        # The native general eigensolver runs on host; the Cherenkov bracketed
        # root evaluates branches inside traced iterations, so this small dense
        # eigenvalue problem uses the traceable LAPACK route directly.
        values = jnp.linalg.eigvals(self.bloch_matrix(wavevector))
        omega = 1j * jnp.log(values) / self.step_size
        order = jnp.argsort(jnp.real(omega))
        return omega[order][self.local_dimension - self.local_dimension // 2 :]

    def bloch_state(
        self, wavevector: ArrayLike, amplitudes: ArrayLike, /
    ) -> CompatibleMaxwellState:
        """Bloch-periodic runtime state ``a_t exp(ik·x)`` on every entity of each type."""
        k = np.asarray(wavevector, dtype=np.float64)
        amplitude = np.asarray(amplitudes, dtype=np.complex128)
        if amplitude.shape != (self.local_dimension,):
            raise ValueError("amplitudes must have one entry per local type.")
        bridge = self.prepared.plan.bridge
        layout = self.prepared.layout
        material_state = self.prepared.constitutive.initialize_state()
        leaves = [
            np.zeros((layout.electric_count,), dtype=np.complex128),
            np.zeros((layout.magnetic_count,), dtype=np.complex128),
            *(
                np.zeros(leaf.shape, dtype=np.complex128)
                for leaf in jax.tree_util.tree_leaves(material_state)
            ),
        ]
        coordinates = _entity_coordinates(bridge.cochain.coordinates)
        for local, value in zip(self.local_types, amplitude, strict=True):
            points = coordinates[local.degree]
            shape = bridge.orientation_shapes[local.degree][local.orientation]
            offset = bridge.orientation_offsets[local.degree][local.orientation]
            entities = offset + np.arange(int(np.prod(shape)))
            target = leaves[local.leaf].reshape((-1, leaves[local.leaf].shape[-1]))
            target[local.component, entities] = value * np.exp(
                1j * (points[entities] @ k)
            )
        material = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(material_state),
            [jnp.asarray(leaf) for leaf in leaves[2:]],
        )
        return self.prepared.pack(
            jnp.asarray(leaves[0]), jnp.asarray(leaves[1]), material_state=material
        )


def _entity_coordinates(values: tuple[Array | None, ...], /) -> tuple[np.ndarray, ...]:
    points = []
    for value in values:
        if value is None:
            raise ValueError("Dispersion audit requires cochain entity coordinates.")
        points.append(np.asarray(value))
    return tuple(points)


def _bloch_matrix(
    rows: Array,
    columns: Array,
    displacements: Array,
    coefficients: Array,
    size: int,
    wavevector: Array,
    /,
) -> Array:
    phase = jnp.exp(1j * (displacements @ wavevector))
    return (
        jnp.zeros((size, size), dtype=jnp.complex128)
        .at[rows, columns]
        .add(coefficients * phase)
    )


class _TypeIndexer(StrictModule, NonTrainableState):
    shapes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    leaf_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    leaf_offsets: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        leaf_shapes: tuple[tuple[int, ...], ...],
        leaf_offsets: np.ndarray,
        /,
    ) -> None:
        self.shapes = bridge.orientation_shapes
        self.offsets = bridge.orientation_offsets
        self.leaf_shapes = leaf_shapes
        self.leaf_offsets = tuple(int(value) for value in leaf_offsets)

    def entities(self, local: _LocalType, cells: np.ndarray, /) -> np.ndarray:
        shape = self.shapes[local.degree][local.orientation]
        return self.offsets[local.degree][local.orientation] + np.ravel_multi_index(
            tuple(cells.T), shape
        )

    def flat(self, local: _LocalType, entities: np.ndarray, /) -> np.ndarray:
        count = self.leaf_shapes[local.leaf][-1]
        return self.leaf_offsets[local.leaf] + local.component * count + entities


def _linear_step(
    prepared: PreparedCompatibleMaxwell,
    step_size: Array,
    unravel: Callable[[Array], tuple[Array, Array, Any]],
    /,
) -> Callable[[Array], Array]:
    layout = prepared.layout
    charge = jnp.zeros((layout.charge_count,), dtype=jnp.float64)
    magnetic_charge = jnp.zeros((prepared.magnetic_charge_count,), dtype=jnp.float64)
    boundary = (
        None if prepared.pml is None else prepared.pml.initialize(dtype=jnp.float64)
    )
    observations = tuple(observer.initialize() for observer in prepared.observers)
    zero_forcing = MaxwellSourceForcing(
        jnp.zeros((layout.electric_count,), dtype=jnp.float64),
        jnp.zeros((layout.magnetic_count,), dtype=jnp.float64),
    )

    def step(flat: Array) -> Array:
        displacement, flux, material = unravel(flat)
        state = CompatibleMaxwellState(
            MaxwellPrimaryState(displacement, flux, charge),
            MaxwellAuxiliaryState(material, boundary, magnetic_charge),
            observations,
        )
        stepped = prepared._step_core(
            jnp.asarray(0.0),
            state,
            step_size,
            None,
            source_samples=(zero_forcing, zero_forcing, zero_forcing),
        )
        output, _ = ravel_pytree(
            (
                stepped.primary.electric_displacement,
                stepped.primary.magnetic_flux,
                stepped.auxiliary.material,
            )
        )
        return output

    return step


def _extract_stencil(
    linearization: PreparedLinearization,
    local_types: tuple[_LocalType, ...],
    indexer: _TypeIndexer,
    coordinates: tuple[np.ndarray, ...],
    reference: tuple[int, ...],
    size: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dimension = len(reference)
    centre = np.asarray(reference, dtype=np.int64)[None, :]
    offsets = np.stack(
        np.meshgrid(
            *(np.arange(-_STENCIL_RADIUS, _STENCIL_RADIUS + 1),) * dimension,
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, dimension))
    neighbours = centre + offsets
    basis = np.zeros((len(local_types), size), dtype=np.float64)
    for column, local in enumerate(local_types):
        basis[column, indexer.flat(local, indexer.entities(local, centre))] = 1.0
    responses = np.asarray(jax.vmap(linearization.jvp)(jnp.asarray(basis)))
    rows, columns, displacements, coefficients = [], [], [], []
    for column, source in enumerate(local_types):
        source_point = coordinates[source.degree][indexer.entities(source, centre)[0]]
        for row, target in enumerate(local_types):
            shape = indexer.shapes[target.degree][target.orientation]
            valid = np.all((neighbours >= 0) & (neighbours < np.asarray(shape)), axis=1)
            entities = indexer.entities(target, neighbours[valid])
            values = responses[column, indexer.flat(target, entities)]
            keep = values != 0.0
            points = coordinates[target.degree][entities[keep]]
            rows.append(np.full((int(np.sum(keep)),), row, dtype=np.int32))
            columns.append(np.full((int(np.sum(keep)),), column, dtype=np.int32))
            displacements.append(source_point[None, :] - points)
            coefficients.append(values[keep])
    return (
        np.concatenate(rows),
        np.concatenate(columns),
        np.concatenate(displacements, axis=0),
        np.concatenate(coefficients),
    )


def _verification_wavevector(
    spacing: np.ndarray, cells: tuple[int, ...], wrapped: tuple[bool, ...], /
) -> np.ndarray:
    """Generic oblique wavevector, commensurate with every wrapped periodic axis."""
    fractions = 0.31 + 0.17 * np.arange(spacing.size)
    values = []
    for fraction, step, count, wrap in zip(
        fractions, spacing, cells, wrapped, strict=True
    ):
        if wrap:
            harmonic = max(1, round(0.5 * fraction * count))
            values.append(2.0 * pi * harmonic / (count * step))
        else:
            values.append(fraction * pi / step)
    return np.asarray(values, dtype=np.float64)


def _verify_stencil(
    stencil: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    local_types: tuple[_LocalType, ...],
    linearization: PreparedLinearization,
    indexer: _TypeIndexer,
    coordinates: tuple[np.ndarray, ...],
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    k: np.ndarray,
    /,
) -> float:
    dimension = len(lower)
    amplitudes = np.exp(1j * (0.7 + 1.3 * np.arange(len(local_types))))
    size = indexer.leaf_offsets[-1]
    field = np.zeros((size,), dtype=np.complex128)
    for local, amplitude in zip(local_types, amplitudes, strict=True):
        shape = indexer.shapes[local.degree][local.orientation]
        offset = indexer.offsets[local.degree][local.orientation]
        entities = offset + np.arange(int(np.prod(shape)))
        points = coordinates[local.degree][entities]
        field[indexer.flat(local, entities)] = amplitude * np.exp(1j * (points @ k))
    real = np.asarray(linearization.jvp(jnp.asarray(field.real)))
    imaginary = np.asarray(linearization.jvp(jnp.asarray(field.imag)))
    response = real + 1j * imaginary
    rows, columns, displacements, coefficients = stencil
    matrix = _bloch_matrix(
        jnp.asarray(rows),
        jnp.asarray(columns),
        jnp.asarray(displacements),
        jnp.asarray(coefficients),
        len(local_types),
        jnp.asarray(k),
    )
    predicted = np.asarray(matrix) @ amplitudes
    cells = np.stack(
        np.meshgrid(
            *(np.arange(low, high) for low, high in zip(lower, upper, strict=True)),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, dimension))
    worst, scale = 0.0, float(np.max(np.abs(predicted)))
    for row, local in enumerate(local_types):
        entities = indexer.entities(local, cells)
        expected = predicted[row] * np.exp(1j * (coordinates[local.degree][entities] @ k))
        observed = response[indexer.flat(local, entities)]
        worst = max(worst, float(np.max(np.abs(observed - expected))))
    return worst / max(scale, np.finfo(np.float64).tiny)


def _cyclotron_resonance(
    prepared: PreparedCompatibleMaxwell, step_size: float, /
) -> tuple[Array, Array]:
    material = prepared.constitutive
    if not isinstance(material, PreparedMagnetizedColdPlasmaMaxwellConstitutive):
        empty = jnp.zeros((0,), dtype=jnp.float64)
        return empty, empty
    species, vertices = material.plasma_weight.shape
    displacement = jnp.zeros((prepared.layout.electric_count,), dtype=jnp.float64)
    flux = jnp.zeros((prepared.layout.magnetic_count,), dtype=jnp.float64)
    half = jnp.asarray(0.5 * step_size)

    def executed(current: Array) -> Array:
        state = MagnetizedColdPlasmaState(current)
        state = material.advance_state(half, state, displacement, flux, half, None)
        return material.advance_state(half, state, displacement, flux, half, None).current

    columns = []
    for axis in range(3):
        unit = (
            jnp.zeros((species, 3, vertices), dtype=jnp.float64).at[:, axis, 0].set(1.0)
        )
        columns.append(np.asarray(executed(unit))[:, :, 0])
    maps = np.stack(columns, axis=-1)
    magnitude = np.sqrt(np.sum(np.asarray(material.cyclotron_frequency) ** 2, axis=1))
    shifts = []
    for index in range(species):
        solved = general_eigensolve(
            GeneralEigenproblem(
                DenseLinearOperator(jnp.asarray(maps[index])),
                problem_id="maxwell-cyclotron-step-map",
            )
        )
        omega = 1j * np.log(np.asarray(solved.eigenvalues)) / step_size
        shifts.append(float(np.max(omega.real)) - float(magnitude[index]))
    return (
        jnp.asarray(np.asarray(shifts, dtype=np.float64)),
        jnp.asarray(magnitude * step_size),
    )


class CherenkovRegimeEvidence(StrictModule):
    """Physical versus numerical Cherenkov resonance ``k·v = ω`` per frequency.

    Angles are wavevector cone angles from the velocity; physical cones use the
    continuum index from ``frequency_response`` (``|Re n|``, so negative-index
    bands give the backward-energy cone), numerical cones use the audited Bloch
    branches inside the inscribed Brillouin sphere ``|k| < π/max(h)``
    (``resolved``). ``attenuation`` is the temporal damping ``-Im ω`` of the
    resonant numerical branch; ``anisotropy`` is the spread of numerical cone
    angles over azimuths. Missing cones are NaN.
    """

    angular_frequencies: Array
    azimuths: Array
    beta: Array
    continuum_index: Array
    resolved: Array
    physical_emission: Array
    physical_cone_angle: Array
    numerical_emission: Array
    numerical_cone_angle: Array
    numerical_only: Array
    anisotropy: Array
    attenuation: Array
    root_converged: Array


class CherenkovRegimePlan(StrictModule):
    """Numerical-versus-physical Cherenkov regime of a moving source in a medium."""

    audit: CompatibleMaxwellDispersionAudit
    velocity: Array
    angular_frequencies: Array
    azimuth_count: int = eqx.field(static=True)
    vacuum_speed_of_light: float = eqx.field(static=True)
    termination: NonlinearTermination
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        audit: CompatibleMaxwellDispersionAudit,
        velocity: ArrayLike,
        angular_frequencies: ArrayLike,
        azimuth_count: int,
        /,
        *,
        vacuum_speed_of_light: float = 1.0,
        termination: NonlinearTermination | None = None,
    ) -> None:
        if not isinstance(audit, CompatibleMaxwellDispersionAudit):
            raise TypeError("audit must be a CompatibleMaxwellDispersionAudit.")
        if audit.prepared.layout.polarization != "full_3d":
            raise ValueError("Cherenkov regime analysis requires a full_3d audit.")
        speed = np.asarray(velocity, dtype=np.float64)
        if speed.shape != (3,) or not np.all(np.isfinite(speed)):
            raise ValueError("velocity must be a finite 3-vector.")
        if not np.linalg.norm(speed) > 0.0:
            raise ValueError("velocity must be nonzero.")
        omega = np.asarray(angular_frequencies, dtype=np.float64)
        if omega.ndim != 1 or omega.size == 0 or not np.all(np.isfinite(omega)):
            raise ValueError("angular_frequencies must be a finite nonempty vector.")
        if np.any(omega <= 0.0):
            raise ValueError("angular_frequencies must be positive.")
        count = int(azimuth_count)
        if count < 1:
            raise ValueError("azimuth_count must be positive.")
        light = float(vacuum_speed_of_light)
        if not np.isfinite(light) or light <= 0.0:
            raise ValueError("vacuum_speed_of_light must be finite and positive.")
        selected = NonlinearTermination() if termination is None else termination
        if not isinstance(selected, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        self.audit = audit
        self.velocity = jnp.asarray(speed)
        self.angular_frequencies = jnp.asarray(omega)
        self.azimuth_count = count
        self.vacuum_speed_of_light = light
        self.termination = selected
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cherenkov-regime-plan",
                "audit": audit.audit_id,
                "velocity": array_tree_fingerprint(speed),
                "angular_frequencies": array_tree_fingerprint(omega),
                "azimuth_count": count,
                "vacuum_speed_of_light": light,
            }
        )

    def _frame(self, /) -> tuple[Array, Array, Array]:
        axis = self.velocity / jnp.linalg.norm(self.velocity)
        trial = jnp.where(
            jnp.abs(axis[0]) < 0.9,
            jnp.asarray([1.0, 0.0, 0.0]),
            jnp.asarray([0.0, 1.0, 0.0]),
        )
        first = trial - jnp.dot(trial, axis) * axis
        first = first / jnp.linalg.norm(first)
        second = jnp.cross(axis, first)
        return axis, first, second

    def _direction(self, cosine: Array, azimuth: Array, /) -> Array:
        axis, first, second = self._frame()
        sine = jnp.sqrt(jnp.maximum(1.0 - cosine**2, 0.0))
        return cosine * axis + sine * (
            jnp.cos(azimuth) * first + jnp.sin(azimuth) * second
        )

    def _continuum_tensors(self, omega: Array, /) -> tuple[Array, Array]:
        prepared = self.audit.prepared
        bridge = prepared.plan.bridge
        response = prepared.constitutive.frequency_response(omega)
        reference = self.audit.reference_cell
        columns_e, columns_h = [], []
        for axis in range(3):
            edges = bridge.orientation_shapes[1]
            faces = bridge.orientation_shapes[2]
            # Uniform Cartesian unit field along ``axis`` (x, y, z components).
            unit = tuple(float(index == axis) for index in range(3))
            displacement = bridge.unpack_edge_circulation(
                response.electric_displacement(
                    bridge.pack_edge_circulation(
                        (
                            jnp.full(edges[0], unit[0]),
                            jnp.full(edges[1], unit[1]),
                            jnp.full(edges[2], unit[2]),
                        )
                    )
                )
            )
            columns_e.append(jnp.stack(tuple(value[reference] for value in displacement)))
            magnetic = bridge.unpack_face_flux(
                response.magnetic_field(
                    bridge.pack_face_flux(
                        (
                            jnp.full(faces[2], unit[0]),
                            jnp.full(faces[1], unit[1]),
                            jnp.full(faces[0], unit[2]),
                        )
                    )
                )
            )
            columns_h.append(jnp.stack(tuple(value[reference] for value in magnetic)))
        permittivity = jnp.stack(columns_e, axis=-1)
        inverse_permeability = jnp.stack(columns_h, axis=-1)
        permeability = solve_small_linear(
            SmallLinearSolvePlan(3),
            inverse_permeability,
            jnp.eye(3, dtype=inverse_permeability.dtype),
        ).value
        return permittivity, permeability

    def _continuum_indices(
        self, permittivity: Array, permeability: Array, direction: Array, /
    ) -> Array:
        # Booker quadratic A m² − B m + C = 0 for m = n²/μ with scalar μ.
        kappa = direction.astype(jnp.complex128)
        trace = jnp.trace(permittivity)
        a = kappa @ permittivity @ kappa
        b = kappa @ (trace * permittivity - permittivity @ permittivity) @ kappa
        c = determinant_small_linear(SmallLinearSolvePlan(3), permittivity)
        root = jnp.sqrt(b * b - 4.0 * a * c)
        roots = jnp.stack(((b - root) / (2.0 * a), (b + root) / (2.0 * a)))
        index = self.vacuum_speed_of_light * jnp.sqrt(permeability) * jnp.sqrt(roots)
        return index[jnp.argsort(jnp.real(index))]

    def evaluate(self, /) -> CherenkovRegimeEvidence:
        audit = self.audit
        speed = jnp.linalg.norm(self.velocity)
        beta = speed / self.vacuum_speed_of_light
        azimuths = 2.0 * pi * jnp.arange(self.azimuth_count) / self.azimuth_count
        radius = pi / jnp.max(audit.spacing)
        branch_count = audit.local_dimension // 2
        method = TOMS748()
        tensors = [self._continuum_tensors(omega) for omega in self.angular_frequencies]
        mu = []
        for permittivity, permeability in tensors:
            scalar = permeability[0, 0]
            if not bool(
                jnp.allclose(permeability, scalar * jnp.eye(3), rtol=1e-10, atol=1e-14)
            ):
                raise ValueError(
                    "Cherenkov continuum index requires an isotropic permeability."
                )
            mu.append(scalar)
        permittivities = jnp.stack(tuple(value[0] for value in tensors))
        permeabilities = jnp.stack(mu)
        axis = self._frame()[0]

        def physical(omega_index: Array, azimuth: Array, branch: Array) -> Array:
            permittivity = permittivities[omega_index]
            permeability = permeabilities[omega_index]

            def residual(angle: Array, args: object) -> Array:
                del args
                direction = self._direction(jnp.cos(angle), azimuth)
                index = self._continuum_indices(permittivity, permeability, direction)
                return beta * jnp.abs(jnp.real(index[branch])) * jnp.cos(angle) - 1.0

            upper = jnp.asarray(0.5 * pi - 1e-9)
            emits = residual(jnp.asarray(0.0), None) > 0.0
            result = scalar_root(
                ScalarRootProblem(
                    residual,
                    bracket=(jnp.asarray(0.0), upper),
                    problem_id="cherenkov-physical-cone",
                ),
                method=method,
                termination=self.termination,
            )
            return jnp.where(emits, result.root, jnp.nan)

        def numerical(omega: Array, azimuth: Array, branch: Array) -> tuple[Array, ...]:
            lower = omega / speed

            def direction(k: Array) -> Array:
                return self._direction(jnp.clip(omega / (speed * k), -1.0, 1.0), azimuth)

            def branch_value(k: Array) -> Array:
                return audit.branch_frequencies(k * direction(k))[branch]

            def residual(k: Array, args: object) -> Array:
                del args
                return jnp.real(branch_value(k)) - omega

            upper = jnp.asarray(radius * (1.0 - 1e-9))
            resolved = lower < upper
            safe_lower = jnp.where(resolved, lower, 0.5 * upper)
            emits = (
                resolved
                & (residual(safe_lower, None) < 0.0)
                & (residual(upper, None) > 0.0)
            )
            result = scalar_root(
                ScalarRootProblem(
                    residual,
                    bracket=(safe_lower, upper),
                    problem_id="cherenkov-numerical-cone",
                ),
                method=method,
                termination=self.termination,
            )
            k = result.root
            angle = jnp.arccos(jnp.clip(omega / (speed * k), -1.0, 1.0))
            damping = -jnp.imag(branch_value(k))
            return (
                jnp.where(emits, angle, jnp.nan),
                jnp.where(emits, damping, jnp.nan),
                emits & result.successful,
                emits,
            )

        frequency_index = jnp.arange(self.angular_frequencies.size)
        physical_branches = jnp.arange(2)
        numerical_branches = jnp.arange(branch_count)
        physical_angles = jax.vmap(
            lambda index: jax.vmap(
                lambda azimuth: jax.vmap(lambda branch: physical(index, azimuth, branch))(
                    physical_branches
                )
            )(azimuths)
        )(frequency_index)
        numerical_angles, attenuation, converged, emitting = jax.vmap(
            lambda omega: jax.vmap(
                lambda azimuth: jax.vmap(
                    lambda branch: numerical(omega, azimuth, branch)
                )(numerical_branches)
            )(azimuths)
        )(self.angular_frequencies)
        continuum_index = jax.vmap(
            lambda permittivity, permeability: self._continuum_indices(
                permittivity, permeability, axis
            )
        )(permittivities, permeabilities)
        physical_emission = jnp.any(jnp.isfinite(physical_angles), axis=-1)
        numerical_emission = jnp.any(emitting, axis=-1)
        spread = jnp.nanmax(numerical_angles, axis=1) - jnp.nanmin(
            numerical_angles, axis=1
        )
        complete = jnp.sum(emitting, axis=1) >= 2
        return CherenkovRegimeEvidence(
            angular_frequencies=self.angular_frequencies,
            azimuths=azimuths,
            beta=beta,
            continuum_index=continuum_index,
            resolved=self.angular_frequencies / speed < radius,
            physical_emission=physical_emission,
            physical_cone_angle=physical_angles,
            numerical_emission=numerical_emission,
            numerical_cone_angle=numerical_angles,
            numerical_only=numerical_emission & ~physical_emission,
            anisotropy=jnp.where(complete, spread, jnp.nan),
            attenuation=attenuation,
            root_converged=converged | ~emitting,
        )


__all__ = [
    "CherenkovRegimeEvidence",
    "CherenkovRegimePlan",
    "CompatibleMaxwellDispersionAudit",
    "MaxwellDispersionResult",
    "MaxwellMaterialRegion",
]
