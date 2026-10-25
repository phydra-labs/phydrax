#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomistic coordinate/species diffusion and periodic generation samplers.

Nonperiodic fixed-support diffusion (`AtomisticCoordinateDiffusion`,
`AtomisticHybridDiffusion`) is mass centered in Cartesian coordinates and
refuses periodic batches.  Periodic generation is admitted deliberately and
separately: fractional coordinates live on the unit `FlatTorusManifold` with
its fractional-Euclidean metric declared explicitly, cells use the
`AtomisticLogCholeskyCellChart` positive-volume chart, and variable compositions corrupt
species with a categorical schedule over the support vocabulary.

Generation pseudotime is never molecular-dynamics time.  No density or
likelihood statement is made: gauge (rotation/translation), centering, cell
chart and fractional-to-Cartesian Jacobians are not accounted for here.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords, NumericRevision
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..domain import DomainFunction, HyperRectangle
from ..dynamics import StateLayout
from ..ein import contract
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..metrix._flat_torus import FlatTorusManifold
from ..solver._dynamics_evolution import DiffraxEvolution
from ..solver._temporal_method import configuration_id
from ..stochastic._categorical_diffusion import (
    CategoricalDiffusionSchedule,
    CategoricalReverseDiffusion,
    CategoricalTerminalRelationship,
)
from ..stochastic._gaussian_diffusion import (
    AbstractGaussianDiffusion,
    DiffusionTerminalReference,
    VarianceExplodingDiffusion,
)
from ..transport.continuous import ContinuousTransport
from ..transport.diffusion import probability_flow_system, ReverseDiffusion, ScoreContext
from ..transport.diffusion._reverse import PreparedReverseDiffusion
from ..typing import checked, PRNGKey
from ._conditioning import AtomisticConditioning, AtomisticConditionLevel
from ._generation import (
    _callable_identity,
    _callable_numeric_content,
    AbstractPreparedAtomisticSampler,
    AtomisticCellAdmission,
    AtomisticCellBounds,
    AtomisticChemistryAdmission,
    AtomisticGenerationSupport,
    AtomisticPayloadSpec,
    AtomisticProposalBlock,
    AtomisticProposalStatus,
    map_admitted_lanes,
    role_keys,
)
from ._types import AtomisticBatch


def _expanded_mask(batch: AtomisticBatch) -> Array:
    return batch.atom_mask[..., None]


def _center_positions(
    batch: AtomisticBatch, positions: ArrayLike, /
) -> tuple[Array, Array]:
    value = jnp.asarray(positions, dtype=batch.positions.dtype)
    mask = _expanded_mask(batch)
    mass = jnp.where(batch.atom_mask, batch.masses, 0.0)
    total = jnp.sum(mass, axis=1, keepdims=True)
    center = jnp.sum(mass[..., None] * jnp.where(mask, value, 0.0), axis=1) / total
    centered = jnp.where(mask, value - center[:, None, :], 0.0)
    return centered, center


class AtomisticCoordinateDiffusion(StrictModule):
    """Mass-centered continuous diffusion on one fixed-capacity atomistic batch."""

    template: AtomisticBatch
    process: AbstractGaussianDiffusion
    center_weights: Array
    center_weight_squared_norm: Array
    active_count: Array
    process_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        template: AtomisticBatch,
        process: AbstractGaussianDiffusion,
        /,
        *,
        process_id: str | None = None,
    ) -> None:
        if template.has_periodic_metadata:
            raise ValueError(
                "Atomistic coordinate diffusion is nonperiodic and mass centered; "
                "periodic supports use AtomisticPeriodicCoordinateDiffusion."
            )
        if process.state_shape != (template.positions.size,):
            raise ValueError(
                "Coordinate diffusion dimension must equal padded position size."
            )
        mass = jnp.where(template.atom_mask, template.masses, 0.0)
        center_weights = mass / jnp.sum(mass, axis=1, keepdims=True)
        self.template = template
        self.process = process
        self.center_weights = center_weights
        self.center_weight_squared_norm = jnp.sum(
            center_weights**2, axis=1, keepdims=True
        )
        self.active_count = jnp.sum(
            template.atom_mask, axis=1, keepdims=True, dtype=template.positions.dtype
        )
        self.process_id = process_id or canonical_fingerprint(
            {
                "kind": "atomistic-coordinate-diffusion",
                "batch_topology_id": template.atom_topology_id,
                "process_id": process.process_id,
            }
        )

    @checked
    def _require_batch(self, batch: AtomisticBatch, /) -> AtomisticBatch:
        if batch.atom_topology_id != self.template.atom_topology_id:
            raise ValueError(
                "Atomistic diffusion requires the template candidate topology."
            )
        if batch.scale.scale_id != self.template.scale.scale_id:
            raise ValueError("Coordinate diffusion requires the template scale.")
        invalid = ~(
            jnp.array_equal(batch.atomic_numbers, self.template.atomic_numbers)
            & jnp.array_equal(batch.atom_mask, self.template.atom_mask)
            & jnp.array_equal(batch.particle_ids, self.template.particle_ids)
            & jnp.array_equal(batch.masses, self.template.masses)
        )
        checked_positions = eqx.error_if(
            batch.positions,
            invalid,
            "Coordinate diffusion requires fixed species, masks, IDs, and masses.",
        )
        return eqx.tree_at(lambda value: value.positions, batch, checked_positions)

    def perturb(
        self, batch: AtomisticBatch, key: PRNGKey, /, *, time: ArrayLike
    ) -> AtomisticBatch:
        batch = self._require_batch(batch)
        centered, _ = _center_positions(batch, batch.positions)
        perturbed = self.process.perturb(key, centered.reshape((-1,)), t1=time).reshape(
            centered.shape
        )
        perturbed, _ = _center_positions(batch, perturbed)
        perturbed = jnp.where(_expanded_mask(batch), perturbed, batch.positions)
        return batch.with_positions(perturbed)

    def conditional_score(
        self, perturbed: AtomisticBatch, clean: AtomisticBatch, /, *, time: ArrayLike
    ) -> Array:
        perturbed = self._require_batch(perturbed)
        clean = self._require_batch(clean)
        noisy, _ = _center_positions(perturbed, perturbed.positions)
        source, _ = _center_positions(clean, clean.positions)
        unconstrained = self.process.conditional_score(
            noisy.reshape((-1,)), source.reshape((-1,)), t1=time
        ).reshape(noisy.shape)
        unconstrained = jnp.where(_expanded_mask(clean), unconstrained, 0.0)
        summed = jnp.sum(unconstrained, axis=1, keepdims=True)
        denominator = self.active_count * self.center_weight_squared_norm
        direction = (
            self.center_weight_squared_norm * clean.atom_mask - self.center_weights
        )
        return unconstrained - summed / denominator[..., None] * direction[..., None]


class AtomisticHybridDiffusion(StrictModule):
    """Coupled continuous coordinates and categorical species corruption."""

    coordinate: AtomisticCoordinateDiffusion
    species_schedule: CategoricalDiffusionSchedule
    species: tuple[int, ...] = eqx.field(static=True)
    species_to_index: Array
    process_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        coordinate: AtomisticCoordinateDiffusion,
        species_schedule: CategoricalDiffusionSchedule,
        species: Sequence[int],
        /,
    ) -> None:
        values = tuple(species)
        if len(values) != species_schedule.num_classes or len(set(values)) != len(values):
            raise ValueError("species must uniquely map every categorical class.")
        if any(number <= 0 for number in values):
            raise ValueError(
                "Atomistic species vocabulary must contain positive numbers."
            )
        maximum = max(values)
        mapping = jnp.full((maximum + 1,), -1, dtype=jnp.int32)
        mapping = mapping.at[jnp.asarray(values)].set(
            jnp.arange(len(values), dtype=jnp.int32)
        )
        active_numbers = coordinate.template.atomic_numbers[coordinate.template.atom_mask]
        if bool(jnp.any(active_numbers > maximum)) or bool(
            jnp.any(mapping[active_numbers] < 0)
        ):
            raise ValueError(
                "Template contains atomic numbers outside species vocabulary."
            )
        self.coordinate = coordinate
        self.species_schedule = species_schedule
        self.species = values
        self.species_to_index = mapping
        self.process_id = canonical_fingerprint(
            {
                "kind": "atomistic-hybrid-diffusion",
                "coordinate_process_id": coordinate.process_id,
                "species_schedule_id": species_schedule.schedule_id,
                "species": values,
            }
        )

    def perturb(
        self,
        batch: AtomisticBatch,
        coordinate_key: PRNGKey,
        species_key: PRNGKey,
        /,
        *,
        continuous_time: ArrayLike,
        discrete_timestep: ArrayLike,
    ) -> tuple[AtomisticBatch, Array]:
        coordinates = self.coordinate.perturb(batch, coordinate_key, time=continuous_time)
        indices = jnp.where(
            batch.atom_mask,
            self.species_to_index[batch.atomic_numbers],
            0,
        )
        noised = self.species_schedule.corrupt(indices, discrete_timestep, species_key)
        noised = jnp.where(batch.atom_mask, noised, -1)
        return coordinates, noised

    def decode_species(self, indices: ArrayLike, mask: ArrayLike, /) -> Array:
        raw = jnp.asarray(indices)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Species indices must use an integer dtype.")
        value = raw.astype(jnp.int32)
        active = jnp.asarray(mask, dtype=jnp.bool_)
        vocabulary = jnp.asarray(self.species, dtype=jnp.int32)
        if value.shape != active.shape:
            raise ValueError("Species indices and mask must have identical shapes.")
        value = eqx.error_if(
            value,
            jnp.any(active & ((value < 0) | (value >= len(self.species)))),
            "Active species index lies outside the declared vocabulary.",
        )
        safe = jnp.where(active, value, 0)
        return jnp.where(active, vocabulary[safe], 0)


class AtomisticEquivarianceReport(StrictModule):
    rotation_defect: Array
    translation_residual: Array
    permutation_defect: Array
    valid: Array


def atomistic_score_equivariance(
    score_model: Any,
    batch: AtomisticBatch,
    time: ArrayLike,
    rotation: ArrayLike,
    permutation: ArrayLike,
    /,
) -> AtomisticEquivarianceReport:
    raw_matrix = jnp.asarray(rotation)
    raw_order = jnp.asarray(permutation)
    if jnp.iscomplexobj(raw_matrix):
        raise TypeError("rotation must be real-valued.")
    if not jnp.issubdtype(raw_order.dtype, jnp.integer):
        raise TypeError("permutation must use an integer dtype.")
    matrix = raw_matrix.astype(batch.positions.dtype)
    order = raw_order.astype(jnp.int32)
    if matrix.shape != (3, 3) or order.shape != (batch.atom_capacity,):
        raise ValueError("rotation/permutation shapes are invalid.")
    orthogonality = matrix.T @ matrix
    if not bool(
        jnp.allclose(
            orthogonality,
            jnp.eye(3, dtype=matrix.dtype),
            atol=1e-7,
            rtol=1e-7,
        )
        & jnp.isclose(jnp.linalg.det(matrix), 1.0, atol=1e-7, rtol=1e-7)
    ):
        raise ValueError("rotation must be a proper orthogonal matrix.")
    if not bool(jnp.array_equal(jnp.sort(order), jnp.arange(batch.atom_capacity))):
        raise ValueError("permutation must contain every atom-axis index exactly once.")
    base = jnp.asarray(score_model(batch, time))
    if base.shape != batch.positions.shape:
        raise ValueError("Atomistic score model must return one vector per padded atom.")
    rotated_batch = batch.with_positions(batch.positions @ matrix.T)
    rotated = jnp.asarray(score_model(rotated_batch, time))
    if rotated.shape != base.shape:
        raise ValueError("Rotated atomistic score changed shape.")
    rotation_defect = jnp.max(jnp.abs(rotated - base @ matrix.T))
    permuted = AtomisticBatch(
        batch.atomic_numbers[:, order],
        batch.positions[:, order],
        batch.masses[:, order],
        batch.scale,
        particle_ids=batch.particle_ids[:, order],
        atom_mask=batch.atom_mask[:, order],
        cells=batch.cells,
        periodic_axes=batch.periodic_axes,
        structure_ids=batch.structure_ids,
        coordinate_dtype=batch.positions.dtype,
        mass_accumulation_dtype=tuple(
            particles.mass_measure.total_mass.dtype for particles in batch.particles
        ),
    )
    permuted_score = jnp.asarray(score_model(permuted, time))
    if permuted_score.shape != base.shape:
        raise ValueError("Permuted atomistic score changed shape.")
    permutation_defect = jnp.max(jnp.abs(permuted_score - base[:, order]))
    mass = jnp.where(batch.atom_mask, batch.masses, 0.0)
    translation = jnp.sum(mass[..., None] * base, axis=1)
    translation_residual = jnp.max(jnp.abs(translation))
    finite = jnp.all(
        jnp.isfinite(
            jnp.asarray([rotation_defect, permutation_defect, translation_residual])
        )
    )
    return AtomisticEquivarianceReport(
        rotation_defect,
        translation_residual,
        permutation_defect,
        finite,
    )


# -- periodic coordinates, cells and species ------------------------------------------------

_CHART_SIZE = 6
_SMALL3 = SmallLinearSolvePlan(3)
_SPECIES_CONTEXT = "species"
_MASK_CONTEXT = "atom_mask"
_COORDINATE_ROLE = "periodic-coordinates"
_SPECIES_ROLE = "periodic-species"
_SCORE_ROLE = "periodic-score"
_SPECIES_PREDICTOR_CONTEXT = "periodic species predictor"


def _positive_count(value: int, name: str, /) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _floating(value: ArrayLike, trailing: tuple[int, ...], name: str, /) -> Array:
    array = jnp.asarray(value)
    rank = len(trailing)
    if array.ndim < rank or array.shape[array.ndim - rank :] != trailing:
        raise ValueError(f"{name} must end in shape {trailing}; got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be a real floating array.")
    return array


@final
class AtomisticLogCholeskyCellChartEvidence(StrictModule):
    """Decoded row cells and their declared bound evidence (never repaired)."""

    cell: Array
    admitted: Array
    heights: Array
    condition: Array
    volume: Array


@final
class AtomisticLogCholeskyCellChart(StrictModule, NonTrainableState):
    """Positive-volume lower-triangular log-diagonal chart of 3-D row cells.

    Chart coordinates
    ``c = (log(L00/l), log(L11/l), log(L22/l), L10/l, L20/l, L21/l)`` decode
    to the lower-triangular row lattice ``L`` (rows ``a, b, c``) with the
    declared reference length ``l``; every chart point has volume
    ``L00 L11 L22 > 0``.  The chart is Euclidean, so its retraction is
    addition.  It fixes the rotation gauge: `encode` maps a right-handed row
    cell ``H = L Q`` (``Q`` proper orthogonal) to the Cholesky factor of its
    metric ``H H^T`` and discards ``Q``; left-handed or singular cells are
    refused through the returned status.

    Admission uses the declared `AtomisticCellBounds` (perpendicular height,
    native small-solve condition and volume).  `evaluate` reports them; an
    inadmissible cell is never repaired, projected or clipped.  No cell
    density is claimed: the chart Jacobian (``L_ii`` on the log-diagonal) and
    the rotation gauge are not accounted for.
    """

    bounds: AtomisticCellBounds
    reference_length: float = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)

    def __init__(
        self, bounds: AtomisticCellBounds, /, *, reference_length: float
    ) -> None:
        if not isinstance(bounds, AtomisticCellBounds):
            raise TypeError("bounds must be AtomisticCellBounds.")
        length = float(reference_length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("reference_length must be finite and positive.")
        self.bounds = bounds
        self.reference_length = length
        self.chart_id = canonical_fingerprint(
            {
                "kind": "atomistic-lower-triangular-log-cell-chart",
                "bounds": bounds.bounds_id,
                "reference_length": length,
            }
        )

    def decode(self, chart: ArrayLike, /) -> Array:
        """Lower-triangular row cells ``(..., 3, 3)`` of chart points ``(..., 6)``."""
        value = _floating(chart, (_CHART_SIZE,), "Cell chart coordinates")
        diagonal = jnp.exp(value[..., :3])
        zero = jnp.zeros_like(value[..., 0])
        rows = jnp.stack(
            (
                jnp.stack((diagonal[..., 0], zero, zero), axis=-1),
                jnp.stack((value[..., 3], diagonal[..., 1], zero), axis=-1),
                jnp.stack((value[..., 4], value[..., 5], diagonal[..., 2]), axis=-1),
            ),
            axis=-2,
        )
        return self.reference_length * rows

    def encode(self, cells: ArrayLike, /) -> tuple[Array, Array]:
        """Gauge-fixed chart of right-handed row cells and the encode status.

        Refused rows (nonfinite, singular or left-handed cells) carry ``nan``
        chart coordinates and ``False`` status.
        """
        vectors = _floating(cells, (3, 3), "Row cells")
        metric = contract("...ik,...jk->...ij", vectors, vectors, backend="jax")
        tiny = jnp.finfo(vectors.dtype).tiny
        pivot0 = metric[..., 0, 0]
        l00 = jnp.sqrt(jnp.maximum(pivot0, tiny))
        l10 = metric[..., 1, 0] / l00
        l20 = metric[..., 2, 0] / l00
        pivot1 = metric[..., 1, 1] - l10 * l10
        l11 = jnp.sqrt(jnp.maximum(pivot1, tiny))
        l21 = (metric[..., 2, 1] - l20 * l10) / l11
        pivot2 = metric[..., 2, 2] - l20 * l20 - l21 * l21
        l22 = jnp.sqrt(jnp.maximum(pivot2, tiny))
        determinant = determinant_small_linear(_SMALL3, vectors)
        successful = (
            jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
            & (pivot0 > 0.0)
            & (pivot1 > 0.0)
            & (pivot2 > 0.0)
            & (determinant > 0.0)
        )
        length = self.reference_length
        chart = jnp.stack(
            (
                jnp.log(l00 / length),
                jnp.log(l11 / length),
                jnp.log(l22 / length),
                l10 / length,
                l20 / length,
                l21 / length,
            ),
            axis=-1,
        )
        return jnp.where(successful[..., None], chart, jnp.nan), successful

    def evaluate(self, chart: ArrayLike, /) -> AtomisticLogCholeskyCellChartEvidence:
        """Decode chart points and report the declared bound admission."""
        cell = self.decode(chart)
        admitted, heights, condition, volume = self.bounds.evaluate(cell)
        return AtomisticLogCholeskyCellChartEvidence(
            cell, admitted, heights, condition, volume
        )


def _wrapped_normal_score(
    displacement: Array, variance: Array, extent: int, /
) -> tuple[Array, Array]:
    """Truncated image-sum score of a unit-period wrapped normal and its error bound.

    For a reduced displacement ``delta in [-1/2, 1/2]`` the density is
    ``sum_n exp(-(delta + n)^2 / (2 s^2))``; the score keeps ``|n| <= K``.
    With ``a = K + 1/2``, ``q = exp(-K (K + 1) / (2 s^2))`` and
    ``r = exp(-a / s^2)``, the omitted mass relative to the retained sum is
    at most ``tau = 2 q / (1 - r)`` and the omitted score numerator at most
    ``alpha = 2 q / s^2 ((a + 1)/(1 - r) + r/(1 - r)^2)``, so
    ``|score_K - score| <= |score_K| tau + alpha`` componentwise (to floating
    precision of the exponentials).
    """
    images = jnp.arange(-extent, extent + 1, dtype=displacement.dtype)
    shifted = displacement[..., None] + images
    weights = jax.nn.softmax(-0.5 * shifted * shifted / variance, axis=-1)
    score = -jnp.sum(weights * shifted, axis=-1) / variance
    reach = extent + 0.5
    retained = jnp.exp(-0.5 * extent * (extent + 1) / variance)
    ratio = jnp.exp(-reach / variance)
    mass = 2.0 * retained / (1.0 - ratio)
    numerator = (
        2.0
        * retained
        / variance
        * ((reach + 1.0) / (1.0 - ratio) + ratio / (1.0 - ratio) ** 2)
    )
    return score, jnp.abs(score) * mass + numerator


@final
class AtomisticPeriodicPerturbation(StrictModule):
    """Wrapped fractional coordinates and (variable-cell) chart after corruption."""

    fractional: Array
    chart: Array | None


@final
class AtomisticWrappedNormalScore(StrictModule):
    """Conditional score of the wrapped normal and its truncation evidence.

    ``truncation_bound`` bounds the componentwise absolute error of the
    truncated image sum against the infinite sum (zero on inactive slots);
    ``chart`` is the exact Gaussian chart score of variable-cell supports.
    """

    fractional: Array
    chart: Array | None
    truncation_bound: Array
    maximum_truncation_bound: Array
    image_extent: int = eqx.field(static=True)


@final
class AtomisticPeriodicCoordinateDiffusion(StrictModule):
    """Variance-exploding diffusion of fractional coordinates on the unit flat torus.

    Fractional coordinates of ``atom_capacity`` slots live on
    ``FlatTorusManifold((atom_capacity, 3))`` with the explicitly declared
    fractional-Euclidean metric.  The stochastic VE process acts on the lift
    ``R^{3N}`` (plus the six `AtomisticLogCholeskyCellChart` coordinates for variable
    cells); perturbations are wrapped onto the torus, so the transition law
    is a wrapped normal.  Its conditional score is the declared truncated
    image sum ``|n| <= image_extent`` with an explicit truncation-error bound;
    the chart block is an ordinary (Euclidean) Gaussian.  The common VE
    schedule acts on fractional and dimensionless chart coordinates alike.

    Inactive slots are neither perturbed nor scored.  There is no centering:
    the torus translation gauge is left free, and no likelihood statement is
    made.  Pseudotime is not molecular-dynamics time.
    """

    process: VarianceExplodingDiffusion
    manifold: FlatTorusManifold
    cell_chart: AtomisticLogCholeskyCellChart | None
    atom_capacity: int = eqx.field(static=True)
    image_extent: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        atom_capacity: int,
        process: VarianceExplodingDiffusion,
        /,
        *,
        image_extent: int,
        cell_chart: AtomisticLogCholeskyCellChart | None = None,
        process_id: str | None = None,
    ) -> None:
        capacity = _positive_count(atom_capacity, "atom_capacity")
        if not isinstance(process, VarianceExplodingDiffusion):
            raise TypeError("process must be a VarianceExplodingDiffusion.")
        extent = _positive_count(image_extent, "image_extent")
        if cell_chart is not None and not isinstance(
            cell_chart, AtomisticLogCholeskyCellChart
        ):
            raise TypeError(
                "cell_chart must be an AtomisticLogCholeskyCellChart or None."
            )
        dimension = 3 * capacity + (0 if cell_chart is None else _CHART_SIZE)
        if process.state_shape != (dimension,):
            raise ValueError(
                f"The VE process must act on the lifted state of dimension {dimension}."
            )
        manifold = FlatTorusManifold((capacity, 3))
        identifier = process_id or canonical_fingerprint(
            {
                "kind": "atomistic-periodic-coordinate-diffusion",
                "atom_capacity": capacity,
                "process_id": process.process_id,
                "manifold_id": manifold.manifold_id,
                "cell_chart": None if cell_chart is None else cell_chart.chart_id,
                "image_extent": extent,
            }
        )
        if not identifier:
            raise ValueError("process_id must be non-empty or None.")
        self.process = process
        self.manifold = manifold
        self.cell_chart = cell_chart
        self.atom_capacity = capacity
        self.image_extent = extent
        self.process_id = identifier

    @property
    def fractional_size(self) -> int:
        return 3 * self.atom_capacity

    @property
    def state_dimension(self) -> int:
        return self.process.dimension

    def _chart(self, chart: ArrayLike | None, name: str, /) -> Array | None:
        if self.cell_chart is None:
            if chart is not None:
                raise ValueError("A fixed-cell diffusion takes no cell chart.")
            return None
        if chart is None:
            raise ValueError(f"A variable-cell diffusion requires {name}.")
        return _floating(chart, (_CHART_SIZE,), name)

    def join(self, fractional: ArrayLike, chart: ArrayLike | None = None, /) -> Array:
        """Lifted state ``(..., D)`` of fractional ``(..., N, 3)`` and chart ``(..., 6)``."""
        value = _floating(fractional, (self.atom_capacity, 3), "Fractional coordinates")
        flat = value.reshape(value.shape[:-2] + (self.fractional_size,))
        block = self._chart(chart, "a cell chart")
        if block is None:
            return flat
        return jnp.concatenate((flat, block.astype(flat.dtype)), axis=-1)

    def split(self, state: ArrayLike, /) -> tuple[Array, Array | None]:
        """Fractional ``(..., N, 3)`` and chart ``(..., 6)`` blocks of a lifted state."""
        value = _floating(state, (self.state_dimension,), "Lifted state")
        size = self.fractional_size
        fractional = value[..., :size].reshape(value.shape[:-1] + (self.atom_capacity, 3))
        return fractional, None if self.cell_chart is None else value[..., size:]

    def wrapped_normal_score(
        self, displacement: ArrayLike, variance: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Truncated wrapped-normal score of reduced displacements and its error bound.

        ``displacement`` must be the componentwise minimum image in
        ``[-1/2, 1/2]``; the bound is componentwise.
        """
        value = jnp.asarray(displacement)
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("displacement must be a real floating array.")
        scale = jnp.asarray(variance, dtype=value.dtype)
        if scale.shape != ():
            raise ValueError("variance must be scalar.")
        scale = eqx.error_if(
            scale, ~(scale > 0.0), "Wrapped-normal variance must be positive."
        )
        return _wrapped_normal_score(value, scale, self.image_extent)

    def perturb(
        self,
        key: PRNGKey,
        fractional: ArrayLike,
        atom_mask: ArrayLike,
        /,
        *,
        time: ArrayLike,
        chart: ArrayLike | None = None,
    ) -> AtomisticPeriodicPerturbation:
        """Wrapped-normal corruption of active slots (inactive slots are unchanged)."""
        clean = _floating(fractional, (self.atom_capacity, 3), "Fractional coordinates")
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        if clean.ndim != 2 or mask.shape != (self.atom_capacity,):
            raise ValueError("Periodic perturbation takes one structure and its mask.")
        state = self.join(clean, chart)
        noisy = self.process.perturb(key, state, t1=time)
        noisy_fractional, noisy_chart = self.split(noisy)
        wrapped = self.manifold.wrap(noisy_fractional)
        return AtomisticPeriodicPerturbation(
            jnp.where(mask[:, None], wrapped, clean), noisy_chart
        )

    def conditional_score(
        self,
        perturbed: ArrayLike,
        clean: ArrayLike,
        atom_mask: ArrayLike,
        /,
        *,
        time: ArrayLike,
        perturbed_chart: ArrayLike | None = None,
        clean_chart: ArrayLike | None = None,
    ) -> AtomisticWrappedNormalScore:
        """Exact chart score and truncated wrapped-normal fractional score."""
        noisy = _floating(perturbed, (self.atom_capacity, 3), "Perturbed coordinates")
        source = _floating(clean, (self.atom_capacity, 3), "Clean coordinates")
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        if noisy.ndim != 2 or mask.shape != (self.atom_capacity,):
            raise ValueError("Periodic scores take one structure and its mask.")
        variance = self.process.transition_variance(0.0, time).astype(noisy.dtype)
        delta = self.manifold.log_with_evidence(source, noisy).tangent
        score, bound = _wrapped_normal_score(delta, variance, self.image_extent)
        active = mask[:, None]
        score = jnp.where(active, score, 0.0)
        bound = jnp.where(active, bound, 0.0)
        chart_noisy = self._chart(perturbed_chart, "perturbed_chart")
        chart_clean = self._chart(clean_chart, "clean_chart")
        chart_score = (
            None
            if chart_noisy is None or chart_clean is None
            else -(chart_noisy - chart_clean) / variance
        )
        return AtomisticWrappedNormalScore(
            score, chart_score, bound, jnp.max(bound), self.image_extent
        )


@final
class AtomisticSpeciesDiffusion(StrictModule):
    """Categorical species corruption over an explicit element vocabulary.

    Classes ``0 .. V-1`` are the vocabulary elements in declared order; with
    ``vacancy=True`` class ``V`` is an empty slot, so the active-atom count
    varies.  Decoding a class gives an element and an occupancy only: it
    infers no bonds, charge or oxidation state, stereochemistry or molecule
    identity.  Chemistry qualification is an explicit filter (for example
    `AtomisticChargeBalanceQualifier`).
    """

    schedule: CategoricalDiffusionSchedule
    species_to_class: Array
    vocabulary: tuple[int, ...] = eqx.field(static=True)
    vacancy: bool = eqx.field(static=True)
    diffusion_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: CategoricalDiffusionSchedule,
        vocabulary: Sequence[int],
        /,
        *,
        vacancy: bool,
    ) -> None:
        if not isinstance(schedule, CategoricalDiffusionSchedule):
            raise TypeError("schedule must be a CategoricalDiffusionSchedule.")
        values = tuple(int(number) for number in vocabulary)
        if not values or len(set(values)) != len(values) or min(values) <= 0:
            raise ValueError("vocabulary must list unique positive atomic numbers.")
        if not isinstance(vacancy, bool):
            raise TypeError("vacancy must be a bool.")
        if schedule.num_classes != len(values) + int(vacancy):
            raise ValueError(
                "The schedule must have one class per element (plus the vacancy class)."
            )
        mapping = np.full((max(values) + 1,), -1, dtype=np.int32)
        mapping[np.asarray(values)] = np.arange(len(values), dtype=np.int32)
        self.schedule = schedule
        self.species_to_class = jnp.asarray(mapping)
        self.vocabulary = values
        self.vacancy = vacancy
        self.diffusion_id = canonical_fingerprint(
            {
                "kind": "atomistic-species-diffusion",
                "schedule_id": schedule.schedule_id,
                "vocabulary": values,
                "vacancy": vacancy,
            }
        )

    @property
    def num_classes(self) -> int:
        return self.schedule.num_classes

    def encode(self, species: ArrayLike, atom_mask: ArrayLike, /) -> Array:
        """Class of every slot; refuses unknown elements and unrepresentable vacancies."""
        raw = jnp.asarray(species)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("species must use an integer dtype.")
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        if raw.shape != mask.shape:
            raise ValueError("species and atom_mask must have identical shapes.")
        value = raw.astype(jnp.int32)
        inside = (value >= 0) & (value < self.species_to_class.shape[0])
        classes = self.species_to_class[jnp.where(inside, value, 0)]
        unknown = jnp.any(mask & (~inside | (classes < 0)))
        unrepresentable = jnp.any(~mask) & (not self.vacancy)
        classes = eqx.error_if(
            classes,
            unknown | unrepresentable,
            "Species lie outside the vocabulary or vacancies lack a vacancy class.",
        )
        empty = len(self.vocabulary) if self.vacancy else 0
        return jnp.where(mask, classes, empty).astype(jnp.int32)

    def decode(self, classes: ArrayLike, /) -> tuple[Array, Array, Array]:
        """``(species, atom_mask, valid)`` of categorical classes (zero species if empty)."""
        raw = jnp.asarray(classes)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("classes must use an integer dtype.")
        value = raw.astype(jnp.int32)
        valid = (value >= 0) & (value < self.num_classes)
        occupied = valid & (value < len(self.vocabulary))
        vocabulary = jnp.asarray(self.vocabulary, dtype=jnp.int32)
        species = jnp.where(occupied, vocabulary[jnp.where(occupied, value, 0)], 0)
        return species, occupied, valid

    def corrupt(
        self,
        species: ArrayLike,
        atom_mask: ArrayLike,
        timestep: ArrayLike,
        key: PRNGKey,
        /,
    ) -> Array:
        """Categorical corruption of the encoded classes at ``timestep``."""
        return self.schedule.corrupt(self.encode(species, atom_mask), timestep, key)


# -- prepared periodic samplers --------------------------------------------------------------


@final
class AtomisticPeriodicSamplerMethod(StrictModule):
    """Terminal one-shot state of periodic generation lanes (leading lane axis)."""

    fractional: Array
    cell: Array
    chart: Array | None
    species: Array
    atom_mask: Array
    classes: Array | None
    solver_valid: Array
    solver_detail: Array
    status: Array


@final
class _PeriodicLiftedScore(StrictModule):
    """Caller score on the torus-wrapped lifted state; zero on inactive fractional slots.

    Arguments arrive in ``labels`` order: the lifted state, time, ``'atom_mask'``
    and the caller's context labels.  The fractional block is wrapped onto
    ``[0, 1)`` before the caller model sees it; chart coordinates pass through.
    Scores of inactive fractional slots are zero, so the lifted reverse dynamics
    is the torus dynamics of the active atoms.  Every numerical value is an
    explicit argument: lane data arrive through the score context.
    """

    function: DomainFunction
    manifold: FlatTorusManifold
    chart_size: int = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    state_label: str = eqx.field(static=True)

    def __call__(self, *arguments: Array, key: PRNGKey | None = None) -> Array:
        values = dict(zip(self.labels, arguments, strict=True))
        state = jnp.asarray(values[self.state_label])
        shape = self.manifold.point_shape
        size = shape[0] * shape[1]
        fractional = self.manifold.wrap(state[:size].reshape(shape)).reshape((size,))
        values[self.state_label] = jnp.concatenate((fractional, state[size:]))
        score = jnp.asarray(
            self.function.func(*(values[label] for label in self.function.deps), key=key)
        )
        if score.shape != state.shape:
            raise ValueError("The periodic score model must return one lifted state.")
        mask = jnp.asarray(values[_MASK_CONTEXT], dtype=jnp.bool_)
        active = jnp.concatenate(
            (jnp.repeat(mask, 3), jnp.ones((self.chart_size,), dtype=jnp.bool_))
        )
        return jnp.where(active, score, jnp.zeros_like(score))


@final
class _ConditionedSpeciesPredictor(StrictModule):
    """Caller species denoiser bound to the condition values of one lane."""

    predictor: Callable[..., Any]
    values: tuple[Array, ...]
    labels: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, state: Array, timestep: Array, *, key: PRNGKey) -> Array:
        return self.predictor(
            state, timestep, key=key, **dict(zip(self.labels, self.values, strict=True))
        )


def _condition_rows(
    conditioning: AtomisticConditioning | None, lanes: int, capacity: int, /
) -> dict[str, Array]:
    """Present condition values with a leading lane axis (atom rows per lane)."""
    if conditioning is None:
        return {}
    rows: dict[str, Array] = {}
    for field, value in zip(conditioning.spec.fields, conditioning.values, strict=True):
        if value is None:
            continue
        if field.level is AtomisticConditionLevel.CASE:
            rows[field.name] = value
        else:
            rows[field.name] = value.reshape((lanes, capacity) + value.shape[1:])
    return rows


def _periodic_payload(
    support: AtomisticGenerationSupport, detail: tuple[str, str], /
) -> AtomisticPayloadSpec:
    capacity = support.atom_capacity
    entries: list[tuple[str, tuple[int, ...], str]] = [
        ("solver_valid", (), "bool"),
        (detail[0], (), detail[1]),
        ("fractional_positions", (capacity, 3), "float64"),
    ]
    if support.cell_admission is AtomisticCellAdmission.VARIABLE_CELL:
        entries.append(("cell_chart", (_CHART_SIZE,), "float64"))
    if support.chemistry is AtomisticChemistryAdmission.VARIABLE_COMPOSITION:
        entries.append(("species_classes", (capacity,), "int32"))
    return AtomisticPayloadSpec(entries)


@final
class _SpeciesStage(StrictModule):
    """Admitted species stage of a periodic sampler.

    ``predictor_id`` is the `callable_payload` semantic identity of the caller
    denoiser; ``predictor_numeric`` holds the declared numeric identity of an
    opaque denoiser as dynamic words (None when it is content-addressed).
    """

    chain: CategoricalReverseDiffusion | None
    predictor_numeric: IdentityWords | None
    predictor_id: str | None = eqx.field(static=True)

    @property
    def chain_id(self) -> str | None:
        return None if self.chain is None else self.chain.sampler_id


def _validated_periodic(
    support: AtomisticGenerationSupport,
    diffusion: AtomisticPeriodicCoordinateDiffusion,
    score: DomainFunction,
    species: AtomisticSpeciesDiffusion | None,
    species_predictor: Callable[..., Any] | None,
    species_terminal_probabilities: ArrayLike | None,
    species_terminal_relationship: CategoricalTerminalRelationship,
    /,
    *,
    species_predictor_semantic_id: str | None,
    species_predictor_numeric_id: str | None,
    score_id: str,
    state_label: str,
    time_label: str,
) -> tuple[_SpeciesStage, tuple[str, ...]]:
    """Admission of a periodic sampler; returns its species stage and score context."""
    if not isinstance(support, AtomisticGenerationSupport):
        raise TypeError("support must be an AtomisticGenerationSupport.")
    if not isinstance(diffusion, AtomisticPeriodicCoordinateDiffusion):
        raise TypeError("diffusion must be an AtomisticPeriodicCoordinateDiffusion.")
    if not isinstance(score, DomainFunction):
        raise TypeError("score must be a DomainFunction.")
    if not isinstance(score_id, str) or not score_id:
        raise ValueError("score_id must be a non-empty string.")
    if not support.periodic:
        raise ValueError(
            "Periodic samplers require a periodic support; nonperiodic supports use "
            "Cartesian coordinate generation."
        )
    if diffusion.atom_capacity != support.atom_capacity:
        raise ValueError("The diffusion atom capacity must equal the support capacity.")
    chart = diffusion.cell_chart
    if support.cell_admission is AtomisticCellAdmission.VARIABLE_CELL:
        if chart is None:
            raise ValueError("A variable-cell support requires a cell-chart diffusion.")
        if (
            support.cell_bounds is None
            or chart.bounds.bounds_id != support.cell_bounds.bounds_id
        ):
            raise ValueError("The cell chart must declare the support cell bounds.")
    elif chart is not None:
        raise ValueError("A fixed-cell support takes a diffusion without cell chart.")
    stage = _SpeciesStage(None, None, None)
    if support.chemistry is AtomisticChemistryAdmission.VARIABLE_COMPOSITION:
        if not isinstance(species, AtomisticSpeciesDiffusion):
            raise ValueError(
                "Variable composition requires an AtomisticSpeciesDiffusion."
            )
        if species.vocabulary != support.species_vocabulary:
            raise ValueError(
                "The species diffusion must use the support vocabulary order."
            )
        if not callable(species_predictor):
            raise ValueError("Variable composition requires a species denoiser.")
        predictor_id, declared = _callable_identity(
            species_predictor,
            species_predictor_semantic_id,
            species_predictor_numeric_id,
            context=_SPECIES_PREDICTOR_CONTEXT,
        )
        chain = CategoricalReverseDiffusion(
            species.schedule,
            species_predictor,
            (support.atom_capacity,),
            terminal_probabilities=species_terminal_probabilities,
            terminal_relationship=species_terminal_relationship,
        )
        stage = _SpeciesStage(chain, declared, predictor_id)
    elif (
        species is not None
        or species_predictor is not None
        or species_terminal_probabilities is not None
        or species_predictor_semantic_id is not None
        or species_predictor_numeric_id is not None
    ):
        raise ValueError("Fixed chemistry takes no species diffusion or denoiser.")
    names = support.conditioning.names
    reserved = {_SPECIES_CONTEXT, _MASK_CONTEXT, *names}
    if (
        not state_label
        or not time_label
        or state_label == time_label
        or state_label in reserved
        or time_label in reserved
        or len(reserved) != 2 + len(names)
    ):
        raise ValueError(
            "State/time labels, 'species', 'atom_mask' and condition names must be distinct."
        )
    allowed = reserved | {state_label, time_label}
    if state_label not in score.deps or any(label not in allowed for label in score.deps):
        raise ValueError(
            "The score must depend on the state and only on time, 'species', "
            "'atom_mask' and declared condition names."
        )
    if time_label not in score.domain.labels:
        raise ValueError("The score domain must declare the time label.")
    context = tuple(
        label for label in score.deps if label not in (state_label, time_label)
    )
    return stage, context


def _lifted_score(
    support: AtomisticGenerationSupport,
    diffusion: AtomisticPeriodicCoordinateDiffusion,
    score: DomainFunction,
    context: tuple[str, ...],
    /,
    *,
    score_id: str,
    state_label: str,
    time_label: str,
) -> tuple[DomainFunction, tuple[str, ...], str]:
    """Lifted periodic score, its score-context labels and its scientific id.

    The lifted score always reads ``'atom_mask'``; when the caller domain does
    not declare it, the lifted domain adds it as the indicator cube ``[0, 1]^N``.
    """
    labels_context = (_MASK_CONTEXT,) + tuple(
        label for label in context if label != _MASK_CONTEXT
    )
    labels = (state_label, time_label) + labels_context
    domain = score.domain
    indicator = _MASK_CONTEXT not in domain.labels
    if indicator:
        capacity = support.atom_capacity
        domain = domain @ HyperRectangle(
            jnp.zeros((capacity,)), jnp.ones((capacity,)), label=_MASK_CONTEXT
        )
    chart_size = 0 if diffusion.cell_chart is None else _CHART_SIZE
    lifted = DomainFunction(
        domain=domain,
        deps=labels,
        func=_PeriodicLiftedScore(
            score, diffusion.manifold, chart_size, labels, state_label
        ),
    )
    lifted_id = canonical_fingerprint(
        {
            "kind": "atomistic-periodic-lifted-score",
            "score_id": score_id,
            "manifold_id": diffusion.manifold.manifold_id,
            "chart_size": chart_size,
            "labels": list(labels),
            "atom_mask_domain": "indicator-cube" if indicator else "caller",
        }
    )
    return lifted, labels_context, lifted_id


def _role_key(key: PRNGKey, role: str, /) -> PRNGKey:
    return role_keys(key[None], role)[0]


class _AbstractPeriodicGenerationSampler(AbstractPreparedAtomisticSampler):
    """One-shot periodic generation shared by reverse-SDE and probability-flow samplers.

    Admission runs one stable compiled entrypoint per sampler and resident lane
    count.  `map_admitted_lanes` compacts the admitted lanes and evaluates them
    in bounded chunks of ``lane_chunk`` vmapped lanes; vacant lanes are never
    read and spend no species, score or random work.  Each lane draws only from
    its own caller-supplied key through `role_keys`, never from a batch key, its
    lane or its chunk, so a proposal is a pure function of ``(key,
    conditioning)`` and the sampler is idempotent.  A lane draws species first
    (variable composition), then solves its coordinates through the prepared
    canonical transport with its mask, species and condition values as a
    dynamic `ScoreContext`.  The emitted Cartesian positions are the wrapped
    fractional coordinates times the row cell.

    The species denoiser holds its numeric parameters as visible dynamic PyTree
    leaves (callables hiding arrays of any dtype are refused, with or without
    declared IDs) and is identified by `callable_payload`: a StrictModule or a
    plain stateless module function is content-addressed; any other callable
    declares ``species_predictor_semantic_id`` and
    ``species_predictor_numeric_id``.  Its semantic identity
    (``species_predictor_id``) enters ``sampler_id`` and therefore every
    proposal identity; its numeric content (including a declared numeric ID,
    held as dynamic `IdentityWords`) enters `numeric_revision` beside the
    categorical schedule and terminal law, so a different composition law
    refuses an exact restore while numeric-only changes reuse the executable.
    """

    support: eqx.AbstractVar[AtomisticGenerationSupport]
    diffusion: eqx.AbstractVar[AtomisticPeriodicCoordinateDiffusion]
    species: eqx.AbstractVar[AtomisticSpeciesDiffusion | None]
    species_chain: eqx.AbstractVar[CategoricalReverseDiffusion | None]
    species_predictor_id: eqx.AbstractVar[str | None]
    species_predictor_numeric: eqx.AbstractVar[IdentityWords | None]
    score_context_labels: eqx.AbstractVar[tuple[str, ...]]
    detail_dtype: eqx.AbstractVar[str]
    lane_chunk: eqx.AbstractVar[int]

    @abc.abstractmethod
    def _lane_coordinates(
        self, key: PRNGKey, context: ScoreContext, /
    ) -> tuple[Array, Array, Array]:
        """Terminal lifted state ``(D,)``, solver validity and solver detail of one lane."""

    def _species_numeric_content(self) -> dict[str, Any] | None:
        """Numeric content of the species stage (None for fixed chemistry).

        The categorical schedule and terminal probabilities contribute their
        arrays; the denoiser contributes its `callable_payload` numeric content.
        """
        chain = self.species_chain
        if chain is None:
            return None
        return {
            "chain": tuple(
                jax.tree.leaves(
                    eqx.filter(
                        (chain.schedule, chain.terminal_probabilities), eqx.is_array
                    )
                )
            ),
            "predictor": _callable_numeric_content(
                chain.predictor,
                self.species_predictor_numeric,
                context=_SPECIES_PREDICTOR_CONTEXT,
            ),
        }

    def _species_facts(self) -> dict[str, str | int]:
        """Semantic executable fact of the species denoiser (numeric-free)."""
        if self.species_predictor_id is None:
            return {}
        return {"species_predictor": self.species_predictor_id}

    def _fixed_cell(self, dtype: Any, /) -> Array:
        if self.support.cell is None:
            raise ValueError("The support binds no fixed cell.")
        return jnp.asarray(self.support.cell, dtype=dtype)

    def vacant_method(self, lanes: int, /) -> AtomisticPeriodicSamplerMethod:
        count = _positive_count(lanes, "lanes")
        capacity = self.support.atom_capacity
        dtype = jnp.result_type(float)
        variable_cell = self.diffusion.cell_chart is not None
        return AtomisticPeriodicSamplerMethod(
            fractional=jnp.zeros((count, capacity, 3), dtype=dtype),
            cell=(
                jnp.zeros((count, 3, 3), dtype=dtype)
                if variable_cell
                else jnp.broadcast_to(self._fixed_cell(dtype), (count, 3, 3))
            ),
            chart=jnp.zeros((count, _CHART_SIZE), dtype=dtype) if variable_cell else None,
            species=jnp.zeros((count, capacity), dtype=jnp.int32),
            atom_mask=jnp.zeros((count, capacity), dtype=jnp.bool_),
            classes=(
                None
                if self.species_chain is None
                else jnp.zeros((count, capacity), dtype=jnp.int32)
            ),
            solver_valid=jnp.zeros((count,), dtype=jnp.bool_),
            solver_detail=jnp.zeros((count,), dtype=self.detail_dtype),
            status=jnp.zeros((count,), dtype=jnp.int32),
        )

    def _lane_species(
        self, key: PRNGKey, conditions: Mapping[str, Array], /
    ) -> tuple[Array, Array, Array | None, Array]:
        """``(species, atom_mask, classes, decoder_valid)`` of one lane."""
        chain = self.species_chain
        if chain is None or self.species is None:
            return (
                jnp.asarray(self.support.fixed_species, dtype=jnp.int32),
                jnp.asarray(self.support.fixed_mask, dtype=jnp.bool_),
                None,
                jnp.asarray(True),
            )
        if conditions:
            chain = eqx.tree_at(
                lambda value: value.predictor,
                chain,
                _ConditionedSpeciesPredictor(
                    chain.predictor, tuple(conditions.values()), tuple(conditions)
                ),
            )
        sample = chain.sample(_role_key(key, _SPECIES_ROLE), (1,))
        classes = sample.final_state[0].astype(jnp.int32)
        species, mask, decodable = self.species.decode(classes)
        return species, mask, classes, sample.valid[0] & jnp.all(decodable)

    def _score_context(
        self, species: Array, mask: Array, conditions: Mapping[str, Array], /
    ) -> ScoreContext:
        values: dict[str, Array] = {}
        for label in self.score_context_labels:
            if label == _MASK_CONTEXT:
                values[label] = mask
            elif label == _SPECIES_CONTEXT:
                values[label] = species
            elif label in conditions:
                values[label] = conditions[label]
            else:
                raise ValueError(f"The score depends on absent condition {label!r}.")
        return ScoreContext(values)

    def _lane(
        self, key: PRNGKey, conditions: Mapping[str, Array], /
    ) -> AtomisticPeriodicSamplerMethod:
        """Complete proposal state of one admitted lane (no lane axis)."""
        species, mask, classes, decoded = self._lane_species(key, conditions)
        final, valid, detail = self._lane_coordinates(
            _role_key(key, _COORDINATE_ROLE),
            self._score_context(species, mask, conditions),
        )
        fractional, chart = self.diffusion.split(final)
        fractional = jnp.where(
            mask[:, None], self.diffusion.manifold.wrap(fractional), 0.0
        )
        cell_chart = self.diffusion.cell_chart
        cell = (
            self._fixed_cell(final.dtype)
            if cell_chart is None or chart is None
            else cell_chart.decode(chart)
        )
        solver_valid = jnp.asarray(valid, dtype=jnp.bool_)
        status = jnp.where(
            solver_valid, 0, int(AtomisticProposalStatus.SAMPLER_FAILURE)
        ) | jnp.where(decoded, 0, int(AtomisticProposalStatus.DECODER_INVALID))
        return AtomisticPeriodicSamplerMethod(
            fractional=fractional,
            cell=cell,
            chart=chart,
            species=species.astype(jnp.int32),
            atom_mask=mask,
            classes=classes,
            solver_valid=solver_valid,
            solver_detail=jnp.asarray(detail).astype(self.detail_dtype),
            status=status.astype(jnp.int32),
        )

    def _condition_bits(
        self, conditioning: AtomisticConditioning | None, mask: Array, /
    ) -> Array:
        """Condition validity of every lane on its own (decoded) active atoms."""
        if conditioning is None:
            return jnp.zeros(mask.shape[:1], dtype=jnp.int32)
        status = conditioning.case_status(mask.reshape((-1,)))
        return jnp.where(
            status == 0, 0, int(AtomisticProposalStatus.CONDITION_INVALID)
        ).astype(jnp.int32)

    def _admitted_lanes(
        self,
        keys: Array,
        conditioning: AtomisticConditioning | None,
        selected: Array,
        /,
    ) -> AtomisticPeriodicSamplerMethod:
        lanes = int(keys.shape[0])
        rows = _condition_rows(conditioning, lanes, self.support.atom_capacity)
        names = tuple(rows)

        def lane(key: PRNGKey, *values: Array) -> AtomisticPeriodicSamplerMethod:
            return self._lane(key, dict(zip(names, values, strict=True)))

        method = map_admitted_lanes(
            lane,
            selected,
            (keys, *rows.values()),
            self.vacant_method(lanes),
            lane_chunk=self.lane_chunk,
        )
        bits = jnp.where(
            selected, self._condition_bits(conditioning, method.atom_mask), 0
        )
        return eqx.tree_at(
            lambda value: value.status, method, (method.status | bits).astype(jnp.int32)
        )

    def initialize_method(
        self,
        keys: Array,
        ordinals: Array,
        conditioning: AtomisticConditioning | None,
        selected: Array,
        /,
    ) -> AtomisticPeriodicSamplerMethod:
        # The ordinal enters through its semantic lane key only.
        del ordinals
        return _admit_periodic_lanes(
            self, keys, conditioning, jnp.asarray(selected, dtype=jnp.bool_)
        )

    def emit_method(
        self, method: AtomisticPeriodicSamplerMethod, ordinals: Array, /
    ) -> AtomisticProposalBlock:
        positions = contract(
            "lni,lij->lnj", method.fractional, method.cell, backend="jax"
        )
        positions = jnp.where(method.atom_mask[..., None], positions, 0.0)
        payload = [method.solver_valid, method.solver_detail, method.fractional]
        if method.chart is not None:
            payload.append(method.chart)
        if method.classes is not None:
            payload.append(method.classes)
        return AtomisticProposalBlock(
            ordinals,
            positions,
            method.species,
            method.atom_mask,
            method.cell,
            method.status,
            tuple(payload),
            sampler_id=self.sampler_id,
            payload_names=self.payload_spec.names,
        )


@eqx.filter_jit
def _admit_periodic_lanes(
    sampler: _AbstractPeriodicGenerationSampler,
    keys: Array,
    conditioning: AtomisticConditioning | None,
    selected: Array,
) -> AtomisticPeriodicSamplerMethod:
    """Stable compiled admission: one trace per sampler structure and lane count."""
    return sampler._admitted_lanes(keys, conditioning, selected)


@final
class PreparedPeriodicDiffusionSampler(_AbstractPeriodicGenerationSampler):
    """Periodic proposals by the native reverse-time VE SDE on the lifted torus.

    One `PreparedReverseDiffusion` (sample shape ``(1,)``) is prepared from the
    process's asymptotic terminal reference over the lifted periodic score; each
    admitted lane realizes it from its own coordinate key and solves it with its
    own score context.  The terminal fractional block is wrapped and mapped to
    Cartesian positions through the support's fixed cell or the decoded
    variable-cell chart.  The caller score ``DomainFunction`` depends on the
    state label (lifted state of length ``3N`` or ``3N + 6``), optionally the
    time label, ``'species'`` (int32 atomic numbers), ``'atom_mask'`` and
    condition names (case rows or the lane's atom rows).  Variable compositions
    first draw species through the native `CategoricalReverseDiffusion` with
    the caller denoiser ``predictor(classes, timestep, *, key, **conditions)``
    (identified as described in the shared periodic sampler contract);
    species and coordinates are sampled in that declared order.  Cell admission
    is the support's (`AtomisticCellBounds`), never repaired.  ``lane_chunk`` is
    the number of lanes solved together; it is an executable choice that never
    changes a proposal and is excluded from ``sampler_id``.

    Payload: ``solver_valid``, ``backend_successful``,
    ``fractional_positions`` and, when declared, ``cell_chart`` and
    ``species_classes``.  Proposals carry no likelihood; pseudotime is not MD
    time.
    """

    support: AtomisticGenerationSupport
    diffusion: AtomisticPeriodicCoordinateDiffusion
    species: AtomisticSpeciesDiffusion | None
    species_chain: CategoricalReverseDiffusion | None
    species_predictor_id: str | None = eqx.field(static=True)
    species_predictor_numeric: IdentityWords | None
    prepared: PreparedReverseDiffusion
    payload_spec: AtomisticPayloadSpec
    score_id: str = eqx.field(static=True)
    lifted_score_id: str = eqx.field(static=True)
    context_labels: tuple[str, ...] = eqx.field(static=True)
    score_context_labels: tuple[str, ...] = eqx.field(static=True)
    detail_dtype: str = eqx.field(static=True)
    lane_chunk: int = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    idempotent: bool = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: AtomisticGenerationSupport,
        diffusion: AtomisticPeriodicCoordinateDiffusion,
        score: DomainFunction,
        /,
        *,
        score_id: str,
        dt0: float,
        wiener_tolerance: float,
        species: AtomisticSpeciesDiffusion | None = None,
        species_predictor: Callable[..., Any] | None = None,
        species_predictor_semantic_id: str | None = None,
        species_predictor_numeric_id: str | None = None,
        species_terminal_probabilities: ArrayLike | None = None,
        species_terminal_relationship: CategoricalTerminalRelationship = "assumed",
        state_label: str = "x",
        time_label: str = "t",
        solver: Any = None,
        stepsize_controller: Any = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 4096,
        lane_chunk: int = 1,
    ) -> None:
        stage, context = _validated_periodic(
            support,
            diffusion,
            score,
            species,
            species_predictor,
            species_terminal_probabilities,
            species_terminal_relationship,
            species_predictor_semantic_id=species_predictor_semantic_id,
            species_predictor_numeric_id=species_predictor_numeric_id,
            score_id=score_id,
            state_label=state_label,
            time_label=time_label,
        )
        chunk = _positive_count(lane_chunk, "lane_chunk")
        limit = _positive_count(max_steps, "max_steps")
        lifted, score_context, lifted_id = _lifted_score(
            support,
            diffusion,
            score,
            context,
            score_id=score_id,
            state_label=state_label,
            time_label=time_label,
        )
        terminal = diffusion.process.asymptotic_terminal_reference()
        prepared = ReverseDiffusion(
            diffusion.process,
            lifted,
            terminal,
            score_id=lifted_id,
            dt0=dt0,
            wiener_tolerance=wiener_tolerance,
            state_label=state_label,
            time_label=time_label,
            context_labels=score_context,
            solver=solver,
            stepsize_controller=stepsize_controller,
            rtol=rtol,
            atol=atol,
            max_steps=limit,
        ).prepare((1,))
        payload = _periodic_payload(support, ("backend_successful", "bool"))
        self.support = support
        self.diffusion = diffusion
        self.species = species
        self.species_chain = stage.chain
        self.species_predictor_id = stage.predictor_id
        self.species_predictor_numeric = stage.predictor_numeric
        self.prepared = prepared
        self.payload_spec = payload
        self.score_id = score_id
        self.lifted_score_id = lifted_id
        self.context_labels = context
        self.score_context_labels = score_context
        self.detail_dtype = "bool"
        self.lane_chunk = chunk
        self.stateful = False
        self.idempotent = True
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-reverse-diffusion-sampler",
                "support": support.support_id,
                "diffusion": diffusion.process_id,
                "score_id": score_id,
                "lifted_score": lifted_id,
                "species": None if species is None else species.diffusion_id,
                "species_chain": stage.chain_id,
                "species_predictor": stage.predictor_id,
                "context": list(context),
                "state_label": state_label,
                "time_label": time_label,
                "terminal_reference": terminal.reference_id,
                "transport": prepared.prepared_id,
                "solver": configuration_id(
                    (solver, stepsize_controller), prefix="periodic-reverse-sde"
                ),
                "payload": payload.spec_id,
            }
        )

    def _lane_coordinates(
        self, key: PRNGKey, context: ScoreContext, /
    ) -> tuple[Array, Array, Array]:
        result = self.prepared.sample_with_diagnostics(key, context=context)
        return (
            result.final_states[0],
            result.successful,
            jnp.all(result.solution.backend_successful),
        )

    def signature_facts(self) -> dict[str, str | int]:
        transport = self.prepared.transport
        return {
            "kind": "periodic-reverse-sde",
            "atom_capacity": self.support.atom_capacity,
            "state_dimension": self.diffusion.state_dimension,
            "cell_admission": self.support.cell_admission.value,
            "chemistry": self.support.chemistry.value,
            "process_id": self.diffusion.process_id,
            "score_id": self.score_id,
            "lifted_score_id": self.lifted_score_id,
            "dt0": repr(transport.dt0),
            "max_steps": transport.max_steps,
            "lane_chunk": self.lane_chunk,
            "payload": self.payload_spec.spec_id,
            **self._species_facts(),
        }

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(
            self.sampler_id,
            {
                "coordinates": tuple(
                    jax.tree.leaves(
                        eqx.filter((self.prepared, self.diffusion), eqx.is_array)
                    )
                ),
                "species": self._species_numeric_content(),
            },
        )


@final
class PreparedPeriodicFlowSampler(_AbstractPeriodicGenerationSampler):
    """Periodic proposals by the probability-flow ODE of the VE process on the torus.

    One `ContinuousTransport` is prepared: the process's asymptotic terminal
    reference pushed through `probability_flow_system` (argument mode) evolved
    by the native `DiffraxEvolution`.  Each admitted lane binds its own score key
    and score context as the transport arguments and pushes one draw from its
    own coordinate key; the vector field uses the lifted periodic score, so the
    lifted flow is a flow on `FlatTorusManifold`.  Species stage, score context,
    cell handling, lane mapping and admission are those of
    `PreparedPeriodicDiffusionSampler`.

    Payload: ``solver_valid``, ``evolution_status`` (`ContinuousTransport`
    status), ``fractional_positions`` and, when declared, ``cell_chart`` and
    ``species_classes``.  The flow is deterministic given its source draw;
    no transported density or likelihood is claimed (gauge, centering and
    cell Jacobians are not included), and pseudotime is not MD time.
    """

    support: AtomisticGenerationSupport
    diffusion: AtomisticPeriodicCoordinateDiffusion
    species: AtomisticSpeciesDiffusion | None
    species_chain: CategoricalReverseDiffusion | None
    species_predictor_id: str | None = eqx.field(static=True)
    species_predictor_numeric: IdentityWords | None
    terminal_reference: DiffusionTerminalReference
    transport: ContinuousTransport
    payload_spec: AtomisticPayloadSpec
    score_id: str = eqx.field(static=True)
    lifted_score_id: str = eqx.field(static=True)
    context_labels: tuple[str, ...] = eqx.field(static=True)
    score_context_labels: tuple[str, ...] = eqx.field(static=True)
    detail_dtype: str = eqx.field(static=True)
    lane_chunk: int = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    idempotent: bool = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: AtomisticGenerationSupport,
        diffusion: AtomisticPeriodicCoordinateDiffusion,
        score: DomainFunction,
        /,
        *,
        score_id: str,
        species: AtomisticSpeciesDiffusion | None = None,
        species_predictor: Callable[..., Any] | None = None,
        species_predictor_semantic_id: str | None = None,
        species_predictor_numeric_id: str | None = None,
        species_terminal_probabilities: ArrayLike | None = None,
        species_terminal_relationship: CategoricalTerminalRelationship = "assumed",
        state_label: str = "x",
        time_label: str = "t",
        solver: Any = None,
        stepsize_controller: Any = None,
        dt0: float | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 4096,
        lane_chunk: int = 1,
    ) -> None:
        stage, context = _validated_periodic(
            support,
            diffusion,
            score,
            species,
            species_predictor,
            species_terminal_probabilities,
            species_terminal_relationship,
            species_predictor_semantic_id=species_predictor_semantic_id,
            species_predictor_numeric_id=species_predictor_numeric_id,
            score_id=score_id,
            state_label=state_label,
            time_label=time_label,
        )
        chunk = _positive_count(lane_chunk, "lane_chunk")
        limit = _positive_count(max_steps, "max_steps")
        step = None if dt0 is None else float(dt0)
        if step is not None and (not np.isfinite(step) or step <= 0.0):
            raise ValueError("dt0 must be finite and positive or None.")
        lifted, score_context, lifted_id = _lifted_score(
            support,
            diffusion,
            score,
            context,
            score_id=score_id,
            state_label=state_label,
            time_label=time_label,
        )
        process = diffusion.process
        terminal = process.asymptotic_terminal_reference()
        system = probability_flow_system(
            process,
            lifted,
            state_layout=StateLayout(process.state_shape),
            score_id=lifted_id,
            context_labels=score_context,
            score_inputs="args",
            state_label=state_label,
            time_label=time_label,
        )
        evolution = DiffraxEvolution(
            system,
            solver=solver,
            stepsize_controller=stepsize_controller,
            dt0=step,
            rtol=rtol,
            atol=atol,
            max_steps=limit,
        )
        transport = ContinuousTransport(
            terminal.law,
            evolution,
            source_coordinate=0.0,
            target_coordinate=process.terminal_time,
        )
        payload = _periodic_payload(support, ("evolution_status", "int32"))
        self.support = support
        self.diffusion = diffusion
        self.species = species
        self.species_chain = stage.chain
        self.species_predictor_id = stage.predictor_id
        self.species_predictor_numeric = stage.predictor_numeric
        self.terminal_reference = terminal
        self.transport = transport
        self.payload_spec = payload
        self.score_id = score_id
        self.lifted_score_id = lifted_id
        self.context_labels = context
        self.score_context_labels = score_context
        self.detail_dtype = "int32"
        self.lane_chunk = chunk
        self.stateful = False
        self.idempotent = True
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-probability-flow-sampler",
                "support": support.support_id,
                "diffusion": diffusion.process_id,
                "score_id": score_id,
                "lifted_score": lifted_id,
                "species": None if species is None else species.diffusion_id,
                "species_chain": stage.chain_id,
                "species_predictor": stage.predictor_id,
                "context": list(context),
                "state_label": state_label,
                "time_label": time_label,
                "terminal_reference": terminal.reference_id,
                "transport": transport.transport_id,
                "evolution": evolution.method_id,
                "payload": payload.spec_id,
            }
        )

    def _lane_coordinates(
        self, key: PRNGKey, context: ScoreContext, /
    ) -> tuple[Array, Array, Array]:
        transport = eqx.tree_at(
            lambda value: value.args,
            self.transport,
            (_role_key(key, _SCORE_ROLE), context),
            is_leaf=lambda value: value is None,
        )
        sample = transport.sample_with_diagnostics(key, (1,))
        return sample.final_states[0], sample.successful, sample.status[0]

    def signature_facts(self) -> dict[str, str | int]:
        evolution = self.transport.evolution
        if not isinstance(evolution, DiffraxEvolution):
            raise TypeError("Periodic flow transport requires a DiffraxEvolution.")
        if evolution.max_steps is None:
            raise ValueError("Periodic flow transport requires a finite step budget.")
        return {
            "kind": "periodic-probability-flow",
            "atom_capacity": self.support.atom_capacity,
            "state_dimension": self.diffusion.state_dimension,
            "cell_admission": self.support.cell_admission.value,
            "chemistry": self.support.chemistry.value,
            "process_id": self.diffusion.process_id,
            "score_id": self.score_id,
            "lifted_score_id": self.lifted_score_id,
            "evolution": evolution.method_id,
            "max_steps": evolution.max_steps,
            "lane_chunk": self.lane_chunk,
            "payload": self.payload_spec.spec_id,
            **self._species_facts(),
        }

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(
            self.sampler_id,
            {
                "coordinates": tuple(
                    jax.tree.leaves(
                        eqx.filter((self.transport, self.diffusion), eqx.is_array)
                    )
                ),
                "species": self._species_numeric_content(),
            },
        )


__all__ = [
    "AtomisticLogCholeskyCellChart",
    "AtomisticLogCholeskyCellChartEvidence",
    "AtomisticCoordinateDiffusion",
    "AtomisticEquivarianceReport",
    "AtomisticHybridDiffusion",
    "AtomisticPeriodicCoordinateDiffusion",
    "AtomisticPeriodicPerturbation",
    "AtomisticPeriodicSamplerMethod",
    "AtomisticSpeciesDiffusion",
    "AtomisticWrappedNormalScore",
    "PreparedPeriodicDiffusionSampler",
    "PreparedPeriodicFlowSampler",
    "atomistic_score_equivariance",
]
