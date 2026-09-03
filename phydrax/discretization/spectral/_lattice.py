#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace
from ._precision import SpectralPrecisionPolicy


HarmonicTruncationKind: TypeAlias = Literal[
    "circular",
    "parallelogramic",
    "custom",
]


def _canonical_coefficients(coefficients: ArrayLike) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] not in (1, 2) or values.shape[0] == 0:
        raise ValueError(
            "Harmonic coefficients must have shape (count, periodic_dimension), "
            "with periodic_dimension equal to one or two."
        )
    rows = [tuple(int(value) for value in row) for row in values]
    if len(set(rows)) != len(rows):
        raise ValueError("Harmonic coefficients must be unique.")
    zero = (0,) * values.shape[1]
    if zero not in rows:
        raise ValueError("Harmonic coefficients must contain the zero harmonic.")
    row_set = set(rows)
    missing = [row for row in rows if tuple(-value for value in row) not in row_set]
    if missing:
        raise ValueError("Harmonic coefficients must be closed under conjugation.")
    order = [rows.index(zero)] + [index for index, row in enumerate(rows) if row != zero]
    return values[np.asarray(order, dtype=np.int64)]


def _conjugate_indices(coefficients: np.ndarray) -> np.ndarray:
    lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(coefficients)
    }
    return np.asarray(
        [lookup[tuple(-int(value) for value in row)] for row in coefficients],
        dtype=np.int64,
    )


def _reciprocal_vectors(primitive_vectors: Array) -> Array:
    periodic_dimension = primitive_vectors.shape[0]
    if periodic_dimension == 1:
        vector = primitive_vectors[0]
        return (2.0 * jnp.pi * vector / jnp.vdot(vector, vector).real)[None, :]
    a0 = primitive_vectors[0]
    a1 = primitive_vectors[1]
    determinant = a0[0] * a1[1] - a0[1] * a1[0]
    return (
        2.0
        * jnp.pi
        * jnp.stack(
            (
                jnp.stack((a1[1], -a1[0])) / determinant,
                jnp.stack((-a0[1], a0[0])) / determinant,
            ),
            axis=0,
        )
    )


def _cell_measure(primitive_vectors: Array) -> Array:
    if primitive_vectors.shape[0] == 1:
        return jnp.sqrt(jnp.vdot(primitive_vectors[0], primitive_vectors[0]).real)
    a0 = primitive_vectors[0]
    a1 = primitive_vectors[1]
    return jnp.abs(a0[0] * a1[1] - a0[1] * a1[0])


def _pixel_centers(sample_shape: tuple[int, ...], dtype: np.dtype) -> Array:
    axes = tuple(
        (jnp.arange(size, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)) / size
        for size in sample_shape
    )
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)


class LatticeHarmonicLayout(StrictModule, NonTrainableState):
    """Static reciprocal-harmonic topology."""

    coefficients: Array
    conjugate_indices: Array
    difference_coefficients: Array
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    periodic_dimension: int = eqx.field(static=True)
    harmonic_count: int = eqx.field(static=True)
    zero_index: int = eqx.field(static=True)
    truncation: HarmonicTruncationKind = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        /,
        *,
        truncation: HarmonicTruncationKind = "custom",
    ):
        coefficients_host = _canonical_coefficients(coefficients)
        conjugates_host = _conjugate_indices(coefficients_host)
        differences_host = coefficients_host[:, None, :] - coefficients_host[None, :, :]
        mode_ids = tuple(
            "g:" + ",".join(str(int(value)) for value in row) for row in coefficients_host
        )
        self.coefficients = jnp.asarray(coefficients_host, dtype=jnp.int32)
        self.conjugate_indices = jnp.asarray(conjugates_host, dtype=jnp.int32)
        self.difference_coefficients = jnp.asarray(differences_host, dtype=jnp.int32)
        self.mode_ids = mode_ids
        self.periodic_dimension = int(coefficients_host.shape[1])
        self.harmonic_count = int(coefficients_host.shape[0])
        self.zero_index = 0
        self.truncation = truncation
        self.layout_id = canonical_fingerprint(
            {
                "kind": "lattice-harmonic-layout",
                "coefficients": array_tree_fingerprint(coefficients_host),
                "conjugates": array_tree_fingerprint(conjugates_host),
                "truncation": truncation,
            }
        )

    @property
    def minimum_sample_shape(self) -> tuple[int, ...]:
        differences = np.asarray(self.difference_coefficients)
        maximum = np.max(np.abs(differences), axis=(0, 1))
        return tuple(int(2 * value + 1) for value in maximum)


class LatticeHarmonicPlan(StrictModule, NonTrainableState):
    """Symbolic selected-harmonic Fourier plan on a periodic lattice."""

    layout: LatticeHarmonicLayout
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    precision: SpectralPrecisionPolicy
    max_harmonics: int = eqx.field(static=True)
    max_convolution_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        sample_shape: tuple[int, ...],
        /,
        *,
        truncation: HarmonicTruncationKind = "custom",
        precision: SpectralPrecisionPolicy | None = None,
        max_harmonics: int = 4096,
        max_convolution_bytes: int = 2**31,
    ):
        layout = LatticeHarmonicLayout(coefficients, truncation=truncation)
        shape = tuple(int(value) for value in sample_shape)
        if len(shape) != layout.periodic_dimension or any(value < 1 for value in shape):
            raise ValueError(
                "sample_shape must contain one positive size per periodic dimension."
            )
        if any(
            actual < required
            for actual, required in zip(shape, layout.minimum_sample_shape, strict=True)
        ):
            raise ValueError(
                f"sample_shape {shape} cannot resolve all pairwise harmonic differences; "
                f"minimum is {layout.minimum_sample_shape}."
            )
        if layout.harmonic_count > int(max_harmonics):
            raise ValueError("The harmonic count exceeds max_harmonics.")
        precision_ = SpectralPrecisionPolicy() if precision is None else precision
        itemsize = np.dtype(precision_.coefficient_dtype).itemsize
        convolution_bytes = layout.harmonic_count**2 * itemsize
        if convolution_bytes > int(max_convolution_bytes):
            raise ValueError(
                "One Fourier-convolution matrix exceeds max_convolution_bytes."
            )
        self.layout = layout
        self.sample_shape = shape
        self.precision = precision_
        self.max_harmonics = int(max_harmonics)
        self.max_convolution_bytes = int(max_convolution_bytes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-harmonic-plan",
                "layout": layout.layout_id,
                "sample_shape": list(shape),
                "precision": precision_.policy_id,
                "max_harmonics": self.max_harmonics,
                "max_convolution_bytes": self.max_convolution_bytes,
            }
        )

    @classmethod
    def parallelogramic(
        cls,
        mode_counts: tuple[int, ...],
        sample_shape: tuple[int, ...],
        /,
        **kwargs,
    ) -> "LatticeHarmonicPlan":
        counts = tuple(int(value) for value in mode_counts)
        if len(counts) not in (1, 2) or any(
            value < 1 or value % 2 == 0 for value in counts
        ):
            raise ValueError("mode_counts must contain one or two positive odd values.")
        axes = tuple(np.arange(-(count // 2), count // 2 + 1) for count in counts)
        coefficients = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            (-1, len(counts))
        )
        return cls(
            coefficients,
            sample_shape,
            truncation="parallelogramic",
            **kwargs,
        )

    @classmethod
    def circular(
        cls,
        reference_primitive_vectors: ArrayLike,
        harmonic_count: int,
        sample_shape: tuple[int, int],
        /,
        **kwargs,
    ) -> "LatticeHarmonicPlan":
        primitive = np.asarray(reference_primitive_vectors, dtype=np.float64)
        count = int(harmonic_count)
        if primitive.shape != (2, 2):
            raise ValueError("Circular truncation requires two 2D primitive vectors.")
        if count < 1 or count % 2 == 0:
            raise ValueError("harmonic_count must be positive and odd.")
        determinant = (
            primitive[0, 0] * primitive[1, 1] - primitive[0, 1] * primitive[1, 0]
        )
        if abs(determinant) <= np.finfo(np.float64).eps:
            raise ValueError("reference_primitive_vectors must be nonsingular.")
        reciprocal = (
            2.0
            * np.pi
            * np.asarray(
                (
                    (primitive[1, 1], -primitive[1, 0]),
                    (-primitive[0, 1], primitive[0, 0]),
                )
            )
            / determinant
        )
        radius = max(2, int(np.ceil(np.sqrt(count))) + 2)
        candidates = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i == 0 and j == 0:
                    continue
                if i > 0 or (i == 0 and j > 0):
                    coefficient = np.asarray((i, j), dtype=np.int64)
                    wavevector = coefficient @ reciprocal
                    candidates.append((float(wavevector @ wavevector), i, j))
        candidates.sort(key=lambda value: (value[0], value[1], value[2]))
        pair_count = (count - 1) // 2
        if len(candidates) < pair_count:
            raise ValueError("Unable to construct the requested circular truncation.")
        coefficients = [np.asarray((0, 0), dtype=np.int64)]
        for _, i, j in candidates[:pair_count]:
            coefficients.append(np.asarray((i, j), dtype=np.int64))
            coefficients.append(np.asarray((-i, -j), dtype=np.int64))
        return cls(
            np.stack(coefficients, axis=0),
            sample_shape,
            truncation="circular",
            **kwargs,
        )

    def prepare(
        self,
        primitive_vectors: ArrayLike,
        /,
        *,
        numeric_version: str = "0",
    ) -> "LatticeHarmonicDiscretization":
        return LatticeHarmonicDiscretization(
            self,
            primitive_vectors,
            numeric_version=numeric_version,
        )


class LatticeHarmonicDiscretization(StrictModule, NonTrainableState):
    """Prepared reciprocal geometry, transforms, and convolution indexing."""

    plan: LatticeHarmonicPlan
    primitive_vectors: Array
    reciprocal_vectors: Array
    harmonic_wavevectors: Array
    fractional_coordinates: Array
    physical_coordinates: Array
    cell_measure: Array
    physical_space: ArraySpace
    modal_space: ArraySpace
    preparation_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LatticeHarmonicPlan,
        primitive_vectors: ArrayLike,
        /,
        *,
        numeric_version: str = "0",
    ):
        vectors = jnp.asarray(
            primitive_vectors,
            dtype=jnp.dtype(plan.precision.physical_dtype),
        )
        expected_shape = (plan.layout.periodic_dimension, 2)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"primitive_vectors must have shape {expected_shape}; got {vectors.shape}."
            )
        if not isinstance(vectors, jax.core.Tracer):
            vectors_host = np.asarray(vectors)
            measure_host = (
                np.linalg.norm(vectors_host[0])
                if expected_shape[0] == 1
                else abs(np.linalg.det(vectors_host))
            )
            if (
                not np.isfinite(measure_host)
                or measure_host <= np.finfo(vectors_host.dtype).eps
            ):
                raise ValueError(
                    "primitive_vectors must define a finite nondegenerate cell."
                )
        reciprocal = _reciprocal_vectors(vectors)
        coefficients = plan.layout.coefficients.astype(vectors.dtype)
        harmonic_wavevectors = contract("hp,pd->hd", coefficients, reciprocal)
        fractional = _pixel_centers(plan.sample_shape, vectors.dtype)
        physical = contract("...p,pd->...d", fractional, vectors)
        coefficient_dtype = jnp.dtype(plan.precision.coefficient_dtype)
        self.plan = plan
        self.primitive_vectors = vectors
        self.reciprocal_vectors = reciprocal
        self.harmonic_wavevectors = harmonic_wavevectors
        self.fractional_coordinates = fractional
        self.physical_coordinates = physical
        self.cell_measure = _cell_measure(vectors)
        self.physical_space = ArraySpace(plan.sample_shape, dtype=coefficient_dtype)
        self.modal_space = ArraySpace(
            (plan.layout.harmonic_count,), dtype=coefficient_dtype
        )
        self.numeric_version = str(numeric_version)
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "lattice-harmonic-discretization",
                "plan": plan.plan_id,
                "numeric_version": self.numeric_version,
            }
        )

    @property
    def harmonic_count(self) -> int:
        return self.plan.layout.harmonic_count

    @property
    def periodic_dimension(self) -> int:
        return self.plan.layout.periodic_dimension

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.plan.sample_shape

    def real_coordinates(
        self,
        /,
        *,
        component_shape: tuple[int, ...] = (),
        reality_tolerance: float = 1e-10,
        maximum_coordinate_size: int = 10_000_000,
    ):
        """Return independent Hermitian coordinates for a real lattice field."""
        from ._signed_coordinates import SignedHermitianSpectralCoordinates

        layout = self.plan.layout
        return SignedHermitianSpectralCoordinates(
            (layout.harmonic_count,),
            layout.conjugate_indices,
            np.ones((layout.harmonic_count,), dtype=np.int8),
            component_shape=component_shape,
            coefficient_dtype=self.modal_space.dtype,
            layout_id=layout.layout_id,
            reality_tolerance=reality_tolerance,
            maximum_coordinate_size=maximum_coordinate_size,
        )

    def in_plane_wavevectors(self, bloch_wavevector: ArrayLike, /) -> Array:
        bloch = jnp.asarray(
            bloch_wavevector,
            dtype=jnp.dtype(self.plan.precision.transform_dtype),
        )
        if bloch.shape[-1:] != (2,):
            raise ValueError("bloch_wavevector must have trailing shape (2,).")
        return bloch[..., None, :] + self.harmonic_wavevectors

    def analysis(self, values: ArrayLike, /) -> Array:
        value = self.plan.precision.transform(values)
        if value.shape[: self.periodic_dimension] != self.sample_shape:
            raise ValueError(
                "values must begin with the prepared physical sample shape "
                f"{self.sample_shape}; got {value.shape}."
            )
        axes = tuple(range(self.periodic_dimension))
        spectrum = jnp.fft.fftn(value, axes=axes) / np.prod(self.sample_shape)
        coefficients = self.plan.layout.coefficients
        indices = tuple(
            coefficients[:, axis] % self.sample_shape[axis]
            for axis in range(self.periodic_dimension)
        )
        gathered = spectrum[indices]
        phase_argument = jnp.zeros((self.harmonic_count,), dtype=spectrum.real.dtype)
        for axis, size in enumerate(self.sample_shape):
            phase_argument = phase_argument + coefficients[:, axis] / size
        phase = jnp.exp(-1j * jnp.pi * phase_argument)
        phase = phase.reshape((self.harmonic_count,) + (1,) * (gathered.ndim - 1))
        return self.plan.precision.coefficients(gathered * phase)

    def synthesis(self, coefficients: ArrayLike, /) -> Array:
        values = self.plan.precision.transform(coefficients)
        if values.shape[:1] != (self.harmonic_count,):
            raise ValueError(
                f"coefficients must begin with harmonic shape {(self.harmonic_count,)}."
            )
        harmonic_coefficients = self.plan.layout.coefficients
        phase_argument = jnp.zeros((self.harmonic_count,), dtype=values.real.dtype)
        for axis, size in enumerate(self.sample_shape):
            phase_argument = phase_argument + harmonic_coefficients[:, axis] / size
        phase = jnp.exp(1j * jnp.pi * phase_argument)
        phase = phase.reshape((self.harmonic_count,) + (1,) * (values.ndim - 1))
        grid = jnp.zeros(
            self.sample_shape + values.shape[1:],
            dtype=jnp.dtype(self.plan.precision.transform_dtype),
        )
        indices = tuple(
            harmonic_coefficients[:, axis] % self.sample_shape[axis]
            for axis in range(self.periodic_dimension)
        )
        grid = grid.at[indices].add(values * phase)
        axes = tuple(range(self.periodic_dimension))
        reconstructed = jnp.fft.ifftn(grid, axes=axes) * np.prod(self.sample_shape)
        return self.plan.precision.transform(reconstructed)

    def evaluate(self, coefficients: ArrayLike, coordinates: ArrayLike, /) -> Array:
        values = self.plan.precision.transform(coefficients)
        if values.shape[:1] != (self.harmonic_count,):
            raise ValueError("coefficients have an incompatible harmonic dimension.")
        points = jnp.asarray(coordinates, dtype=self.primitive_vectors.dtype)
        if points.shape[-1:] != (2,):
            raise ValueError("coordinates must have trailing shape (2,).")
        phase = jnp.exp(1j * contract("...d,hd->...h", points, self.harmonic_wavevectors))
        trailing_shape = values.shape[1:]
        flattened = values.reshape((self.harmonic_count, -1))
        evaluated = contract("...h,hc->...c", phase, flattened)
        return evaluated.reshape(points.shape[:-1] + trailing_shape)

    def convolution_matrix(self, values: ArrayLike, /) -> Array:
        value = self.plan.precision.transform(values)
        if value.shape[: self.periodic_dimension] != self.sample_shape:
            raise ValueError(
                "values must begin with the prepared physical sample shape "
                f"{self.sample_shape}; got {value.shape}."
            )
        axes = tuple(range(self.periodic_dimension))
        spectrum = jnp.fft.fftn(value, axes=axes) / np.prod(self.sample_shape)
        differences = self.plan.layout.difference_coefficients
        indices = tuple(
            differences[..., axis] % self.sample_shape[axis]
            for axis in range(self.periodic_dimension)
        )
        gathered = spectrum[indices]
        phase_argument = jnp.zeros(
            (self.harmonic_count, self.harmonic_count),
            dtype=spectrum.real.dtype,
        )
        for axis, size in enumerate(self.sample_shape):
            phase_argument = phase_argument + differences[..., axis] / size
        phase = jnp.exp(-1j * jnp.pi * phase_argument)
        phase = phase.reshape(
            (self.harmonic_count, self.harmonic_count) + (1,) * (gathered.ndim - 2)
        )
        return self.plan.precision.coefficients(gathered * phase)

    def translation_phase(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement, dtype=self.primitive_vectors.dtype)
        if value.shape != (2,):
            raise ValueError("displacement must have shape (2,).")
        return jnp.exp(-1j * contract("hd,d->h", self.harmonic_wavevectors, value))

    def translate_coefficients(
        self,
        coefficients: ArrayLike,
        displacement: ArrayLike,
        /,
    ) -> Array:
        values = self.plan.precision.coefficients(coefficients)
        if values.shape[:1] != (self.harmonic_count,):
            raise ValueError("coefficients have an incompatible harmonic dimension.")
        phase = self.translation_phase(displacement).reshape(
            (self.harmonic_count,) + (1,) * (values.ndim - 1)
        )
        return values * phase

    def translate_convolution(
        self,
        matrix: ArrayLike,
        displacement: ArrayLike,
        /,
    ) -> Array:
        value = self.plan.precision.coefficients(matrix)
        if value.shape[:2] != (self.harmonic_count, self.harmonic_count):
            raise ValueError("matrix has incompatible harmonic dimensions.")
        phase = self.translation_phase(displacement)
        trailing = (1,) * (value.ndim - 2)
        return (
            phase.reshape((self.harmonic_count, 1) + trailing)
            * value
            * jnp.conj(phase.reshape((1, self.harmonic_count) + trailing))
        )


class BrillouinZonePlan(StrictModule, NonTrainableState):
    """Gamma-containing periodic trapezoid rule on a reciprocal unit cell."""

    grid_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, grid_shape: tuple[int, ...], /):
        shape = tuple(int(value) for value in grid_shape)
        if len(shape) not in (1, 2) or any(value < 1 for value in shape):
            raise ValueError("grid_shape must contain one or two positive sizes.")
        self.grid_shape = shape
        self.plan_id = canonical_fingerprint(
            {"kind": "brillouin-zone-plan", "grid_shape": list(shape)}
        )

    def prepare(
        self,
        lattice: LatticeHarmonicDiscretization,
        /,
    ) -> "PreparedBrillouinZone":
        return PreparedBrillouinZone(self, lattice)


class PreparedBrillouinZone(StrictModule, NonTrainableState):
    """Prepared reciprocal-cell wavevectors and normalized quadrature weights."""

    plan: BrillouinZonePlan
    wavevectors: Array
    weights: Array
    lattice_preparation_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BrillouinZonePlan,
        lattice: LatticeHarmonicDiscretization,
        /,
    ):
        if len(plan.grid_shape) != lattice.periodic_dimension:
            raise ValueError("The Brillouin rule and lattice dimensions must match.")
        axes = tuple(
            (jnp.arange(size, dtype=lattice.primitive_vectors.dtype) - size // 2) / size
            for size in plan.grid_shape
        )
        fractions = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        wavevectors = contract("...p,pd->...d", fractions, lattice.reciprocal_vectors)
        weights = jnp.full(
            plan.grid_shape,
            1.0 / np.prod(plan.grid_shape),
            dtype=jnp.dtype(lattice.plan.precision.reduction_dtype),
        )
        self.plan = plan
        self.wavevectors = wavevectors
        self.weights = weights
        self.lattice_preparation_id = lattice.preparation_id
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-brillouin-zone",
                "plan": plan.plan_id,
                "lattice": lattice.preparation_id,
            }
        )
