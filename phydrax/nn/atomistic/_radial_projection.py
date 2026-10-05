#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen cubic-spline tabulation of an exact radial realization.

A table is a separate numerical realization of the same learned radial network:
it is bound to the network's numeric revision, to the full model species domain
and to the active ordered species pairs it was prepared for, and it never
replaces exact execution silently. Node slopes follow the pinned source
construction (left not-a-knot, right clamped zero slope at the cutoff), solved
once for every pair and channel through the native prepared spline owner.

Regularity: the interpolant is C2 in the interior and joins the exact zero
extension beyond the cutoff with matching value and first derivative only. The
admitted coordinate derivative order of a table is therefore one (energies and
forces); Hessian-vector products and force-training derivatives remain on the
exact network, and the measured cutoff second-derivative jump is reported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import NumericRevision, SemanticProvenance
from ..._interpolation import (
    cubic_hermite_knot_jets,
    cubic_hermite_segment,
    CubicSplineSlopePlan,
    UniformNodeGrid,
)
from ..._model import register_artifact_value
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
)
from ...typing import Dim, Float, Int32, parse
from ._fixed_binding import require_identical, require_realized
from ._radial import RadialEmbedding, RadialMLP


type _RadialFunction = Callable[[Array, Array, Array], Array]


RadialTableLayout: TypeAlias = Literal["projected-width", "embedding-width"]
RadialTableObjective: TypeAlias = Literal["minimum-table-bytes", "minimum-edge-work"]

# Coordinate derivative order a source-style cubic table can admit: the cutoff
# join is C1 (clamped zero slope), not C2.
_TABLE_COORDINATE_DERIVATIVE_ORDER = 1
# Hermite segment evaluation per output channel: four basis products and adds
# for the value, counted as multiply-add pairs.
_HERMITE_FLOPS_PER_CHANNEL = 8


class RadialTableRowDim(Dim):
    """Prepared (ordered species pair) table rows."""


class RadialTableNodeDim(Dim):
    """Uniform spline nodes on ``[grid_min, cutoff]``."""


class RadialTableWidthDim(Dim):
    """Tabulated channels (projected or embedding width)."""


class RadialEmbeddingWidthDim(Dim):
    """Hidden width entering the final radial linear layer."""


class RadialOutputWidthDim(Dim):
    """Radial network output channels."""


class RadialDomainDim(Dim):
    """Full model species domain."""


class StaleRadialTableBinding(ValueError):
    """A radial table is bound to different radial weights or semantics."""


class RadialTableQualificationError(ValueError):
    """A radial table failed its declared qualification gate."""


def radial_source_revision(
    embedding: RadialEmbedding, mlp: RadialMLP, /
) -> NumericRevision:
    """Host numeric revision of the exact radial realization a table tabulates."""
    if not isinstance(embedding, RadialEmbedding) or not isinstance(mlp, RadialMLP):
        raise TypeError("A radial revision requires a RadialEmbedding and RadialMLP.")
    return NumericRevision(
        SemanticProvenance(
            {
                "kind": "exact-radial-realization",
                "embedding": embedding.embedding_id,
                "widths": list(mlp.widths),
                "activation_scale": mlp.activation_scale,
                "postprocess": mlp.postprocess,
            }
        ),
        {
            f"weights[{index}]": np.asarray(jax.device_get(weight))
            for index, weight in enumerate(mlp.weights)
        },
    )


@final
class RadialTableDeclaration(StrictModule, NonTrainableState):
    """Domain-owned table declaration: uniform support and tabulated layout."""

    grid_min: float = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    layout: RadialTableLayout = eqx.field(static=True)
    declaration_id: str = eqx.field(static=True)

    def __init__(
        self, grid_min: float, node_count: int, /, *, layout: RadialTableLayout
    ) -> None:
        minimum = positive_finite_float(grid_min, "grid_min")
        count = positive_integer(node_count, "node_count")
        if count < 4:
            raise ValueError("Source-style radial tables require at least four nodes.")
        layout_ = parse(layout, RadialTableLayout, "layout")
        self.grid_min = minimum
        self.node_count = count
        self.layout = layout_
        self.declaration_id = canonical_fingerprint(
            {
                "kind": "radial-table-declaration",
                "grid_min": minimum,
                "node_count": count,
                "layout": layout_,
                "left_end": "not-a-knot",
                "right_end": "clamped-zero-slope",
            }
        )


@final
class RadialSpeciesBinding(StrictModule, NonTrainableState):
    """Active ordered species-pair rows inside the full model species domain.

    ``species_domain`` keeps every scientific species of the model (nonnegative
    identifiers: atom-type ID ``0`` is valid; atomic-number positivity belongs
    to the architecture); only the ``active_species`` rows are tabulated.
    Pair-independent realizations share one row; pair-dependent ones tabulate
    every ordered active pair, without assuming symmetry.
    """

    __strict_contract__ = True

    domain_to_active: Int32[RadialDomainDim]
    species_domain: tuple[int, ...] = eqx.field(static=True)
    active_species: tuple[int, ...] = eqx.field(static=True)
    pair_dependent: bool = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_domain: Sequence[int],
        active_species: Sequence[int],
        /,
        *,
        pair_dependent: bool,
    ) -> None:
        domain = tuple(
            nonnegative_integer(value, "species_domain") for value in species_domain
        )
        if not domain or len(set(domain)) != len(domain):
            raise ValueError("species_domain must be nonempty and distinct.")
        active = tuple(
            nonnegative_integer(value, "active_species") for value in active_species
        )
        if (
            not active
            or len(set(active)) != len(active)
            or list(active) != sorted(active)
        ):
            raise ValueError("active_species must be nonempty, distinct and ascending.")
        if active[-1] >= len(domain):
            raise ValueError("active_species must index species_domain.")
        if not isinstance(pair_dependent, bool):
            raise TypeError("pair_dependent must be a bool.")
        lookup = np.full((len(domain),), -1, dtype=np.int32)
        lookup[np.asarray(active, dtype=np.int64)] = np.arange(
            len(active), dtype=np.int32
        )
        self.domain_to_active = jnp.asarray(lookup)
        self.species_domain = domain
        self.active_species = active
        self.pair_dependent = pair_dependent
        self.binding_id = canonical_fingerprint(
            {
                "kind": "radial-species-binding",
                "species_domain": list(domain),
                "active_species": list(active),
                "pair_dependent": pair_dependent,
            }
        )

    @property
    def row_count(self) -> int:
        return len(self.active_species) ** 2 if self.pair_dependent else 1

    def row_pairs(self) -> tuple[tuple[int, int], ...]:
        """Domain species indices ``(sender, receiver)`` of every row, row-major."""
        if not self.pair_dependent:
            return ((self.active_species[0], self.active_species[0]),)
        active = self.active_species
        return tuple((sender, receiver) for sender in active for receiver in active)

    def rows(self, sender: Array, receiver: Array, /) -> tuple[Array, Array]:
        """Return table rows and whether both domain indices are actively bound.

        Indices outside ``[0, len(species_domain))`` are unbound (never wrapped
        or clipped onto another species); their row is zero.
        """
        sender_active = self._active(sender)
        receiver_active = self._active(receiver)
        bound = (sender_active >= 0) & (receiver_active >= 0)
        if not self.pair_dependent:
            return jnp.zeros_like(sender_active), bound
        count = len(self.active_species)
        row = jnp.maximum(sender_active, 0) * count + jnp.maximum(receiver_active, 0)
        return row, bound

    def _active(self, index: Array, /) -> Array:
        inside = (index >= 0) & (index < self.domain_to_active.shape[0])
        safe = jnp.where(inside, index, jnp.zeros_like(index))
        return jnp.where(inside, self.domain_to_active[safe], -1)


@final
class RadialTableResources(StrictModule, NonTrainableState):
    """Measured storage and per-edge work of one table layout."""

    table_bytes: int = eqx.field(static=True)
    projection_bytes: int = eqx.field(static=True)
    edge_flops: int = eqx.field(static=True)


def radial_table_resources(
    mlp: RadialMLP,
    row_count: int,
    node_count: int,
    itemsize: int,
    layout: RadialTableLayout,
    /,
) -> RadialTableResources:
    """Exact table bytes (values and slopes) and per-edge evaluation flops."""
    rows = positive_integer(row_count, "row_count")
    nodes = positive_integer(node_count, "node_count")
    item = positive_integer(itemsize, "itemsize")
    match parse(layout, RadialTableLayout, "layout"):
        case "projected-width":
            width = mlp.output_width
            return RadialTableResources(
                table_bytes=2 * rows * nodes * width * item,
                projection_bytes=0,
                edge_flops=_HERMITE_FLOPS_PER_CHANNEL * width,
            )
        case "embedding-width":
            width = mlp.embedding_width
            return RadialTableResources(
                table_bytes=2 * rows * nodes * width * item,
                projection_bytes=width * mlp.output_width * item,
                edge_flops=_HERMITE_FLOPS_PER_CHANNEL * width
                + 2 * width * mlp.output_width,
            )
        case unreachable:
            assert_never(unreachable)


@final
class RadialTableLayoutChoice(StrictModule, NonTrainableState):
    """Objective-driven layout selection with the evidence for both layouts."""

    objective: RadialTableObjective = eqx.field(static=True)
    selected: RadialTableLayout = eqx.field(static=True)
    projected: RadialTableResources
    embedding: RadialTableResources | None
    reason: str = eqx.field(static=True)


def select_radial_table_layout(
    mlp: RadialMLP,
    row_count: int,
    node_count: int,
    itemsize: int,
    /,
    *,
    objective: RadialTableObjective,
) -> RadialTableLayoutChoice:
    """Choose projected- or embedding-width tables by an explicit objective.

    Embedding-width tables are admissible only when the final layer is linear
    with no postprocessing, because interpolation then commutes with it.
    """
    objective_ = parse(objective, RadialTableObjective, "objective")
    projected = radial_table_resources(
        mlp, row_count, node_count, itemsize, "projected-width"
    )
    if mlp.postprocess != "none":
        return RadialTableLayoutChoice(
            objective=objective_,
            selected="projected-width",
            projected=projected,
            embedding=None,
            reason="embedding-width-inadmissible-nonlinear-postprocess",
        )
    embedding = radial_table_resources(
        mlp, row_count, node_count, itemsize, "embedding-width"
    )
    match objective_:
        case "minimum-table-bytes":
            embedding_wins = (
                embedding.table_bytes + embedding.projection_bytes < projected.table_bytes
            )
        case "minimum-edge-work":
            embedding_wins = embedding.edge_flops < projected.edge_flops
        case unreachable:
            assert_never(unreachable)
    return RadialTableLayoutChoice(
        objective=objective_,
        selected="embedding-width" if embedding_wins else "projected-width",
        projected=projected,
        embedding=embedding,
        reason="objective-comparison",
    )


def _table_identity(
    declaration: RadialTableDeclaration,
    binding: RadialSpeciesBinding,
    grid: UniformNodeGrid,
    embedding_id: str,
    revision_id: str,
    plan_id: str,
    values: ArrayLike,
    /,
) -> str:
    """Content identity of one table from its policy, source and stored values."""
    return canonical_fingerprint(
        {
            "kind": "prepared-radial-tables",
            "declaration": declaration.declaration_id,
            "binding": binding.binding_id,
            "grid": grid.grid_id,
            "embedding": embedding_id,
            "source_revision": revision_id,
            "spline_plan": plan_id,
            "values": array_tree_fingerprint(np.asarray(values)),
        }
    )


@final
class PreparedRadialTables(StrictModule, NonTrainableState):
    """Frozen tables of one exact radial realization for active species pairs."""

    __strict_contract__ = True

    declaration: RadialTableDeclaration
    binding: RadialSpeciesBinding
    grid: UniformNodeGrid
    values: Float[RadialTableRowDim, RadialTableNodeDim, RadialTableWidthDim]
    slopes: Float[RadialTableRowDim, RadialTableNodeDim, RadialTableWidthDim]
    projection: Float[RadialEmbeddingWidthDim, RadialOutputWidthDim] | None
    slope_status: Int32[RadialTableRowDim, RadialTableWidthDim]
    cutoff: float = eqx.field(static=True)
    output_width: int = eqx.field(static=True)
    embedding_id: str = eqx.field(static=True)
    source_revision_id: str = eqx.field(static=True)
    resources: RadialTableResources
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        embedding: RadialEmbedding,
        mlp: RadialMLP,
        declaration: RadialTableDeclaration,
        binding: RadialSpeciesBinding,
        /,
    ) -> None:
        if not isinstance(embedding, RadialEmbedding) or not isinstance(mlp, RadialMLP):
            raise TypeError("Radial tables require a RadialEmbedding and RadialMLP.")
        if not isinstance(declaration, RadialTableDeclaration):
            raise TypeError("declaration must be a RadialTableDeclaration.")
        if not isinstance(binding, RadialSpeciesBinding):
            raise TypeError("binding must be a RadialSpeciesBinding.")
        if binding.pair_dependent != embedding.pair_dependent:
            raise ValueError("Species binding pair dependence must match the embedding.")
        if embedding.transform is not None and (
            embedding.transform.atomic_numbers != binding.species_domain
        ):
            raise ValueError("Species domain must match the Agnesi species order.")
        if mlp.input_width != embedding.basis_count:
            raise ValueError("Radial MLP input width must match the radial basis.")
        if mlp.weights[0].dtype != embedding.dtype:
            raise TypeError("Radial MLP and embedding must share one dtype.")
        if declaration.grid_min >= embedding.radius:
            raise ValueError("grid_min must lie below the cutoff.")
        layout = declaration.layout
        if layout == "embedding-width" and mlp.postprocess != "none":
            raise ValueError(
                "Embedding-width tables require a linear final layer without "
                "postprocessing."
            )
        revision = radial_source_revision(embedding, mlp)
        grid = UniformNodeGrid(
            declaration.grid_min,
            embedding.radius,
            declaration.node_count,
            dtype=embedding.dtype,
        )
        pairs = np.asarray(binding.row_pairs(), dtype=np.int32).reshape((-1, 2))
        nodes = jnp.broadcast_to(grid.nodes, (pairs.shape[0], grid.node_count))
        sender = jnp.broadcast_to(jnp.asarray(pairs[:, :1]), nodes.shape)
        receiver = jnp.broadcast_to(jnp.asarray(pairs[:, 1:]), nodes.shape)
        features = embedding.unmasked(nodes, sender, receiver)
        match layout:
            case "projected-width":
                tabulated = mlp(features)
                projection = None
            case "embedding-width":
                tabulated = mlp.penultimate(features)
                projection = mlp.final_projection
            case unreachable:
                assert_never(unreachable)
        # One factorization, one multi-right-hand-side solve for every row and
        # channel: node axis first, (row, channel) columns.
        plan = CubicSplineSlopePlan(grid, left="not-a-knot", right="clamped")
        solved = plan.slopes(jnp.moveaxis(tabulated, 1, 0), right_slope=0.0)
        if not bool(np.asarray(solved.successful)):
            raise ValueError(
                "Radial table slope solve failed (native status retained in the "
                "solve result); the table is not admitted."
            )
        resources = radial_table_resources(
            mlp,
            binding.row_count,
            grid.node_count,
            np.dtype(embedding.dtype).itemsize,
            layout,
        )
        self.declaration = declaration
        self.binding = binding
        self.grid = grid
        self.values = tabulated
        self.slopes = jnp.moveaxis(solved.slopes, 0, 1)
        self.projection = projection
        self.slope_status = solved.status
        self.cutoff = embedding.radius
        self.output_width = mlp.output_width
        self.embedding_id = embedding.embedding_id
        self.source_revision_id = revision.revision_id
        self.resources = resources
        self.table_id = _table_identity(
            declaration,
            binding,
            grid,
            embedding.embedding_id,
            revision.revision_id,
            plan.plan_id,
            tabulated,
        )

    @property
    def admitted_derivative_order(self) -> int:
        """Coordinate derivative order admitted by the C1 cutoff join."""
        return _TABLE_COORDINATE_DERIVATIVE_ORDER

    def require_current(self, embedding: RadialEmbedding, mlp: RadialMLP, /) -> None:
        """Refuse execution when the exact realization differs from the binding.

        A host boundary: the embedding is re-admitted from its stored arrays
        first, so its identity is recomputed rather than trusted.
        """
        try:
            embedding.validate()
        except ValueError as error:
            raise StaleRadialTableBinding(
                f"Radial table embedding is not admitted: {error}"
            ) from error
        if embedding.embedding_id != self.embedding_id:
            raise StaleRadialTableBinding("Radial table embedding semantics changed.")
        if radial_source_revision(embedding, mlp).revision_id != self.source_revision_id:
            raise StaleRadialTableBinding("Radial table weights are stale.")

    def validate(self, embedding: RadialEmbedding, mlp: RadialMLP, /) -> None:
        """Refuse unless every executed table field re-tabulates ``embedding``/``mlp``.

        This is a host boundary over concrete arrays; ``mlp`` must already be
        validated by its owner and ``embedding`` is re-admitted from its stored
        fixed arrays. The declaration and species
        binding are re-admitted through their constructors, the grid, slope
        status and resources must match exactly, and values, slopes and the
        embedding-width projection must be finite and reproduce a fresh
        tabulation to rounding. ``table_id`` is recomputed from the stored values.
        """
        self.require_current(embedding, mlp)
        declaration = RadialTableDeclaration(
            self.declaration.grid_min,
            self.declaration.node_count,
            layout=self.declaration.layout,
        )
        binding = RadialSpeciesBinding(
            self.binding.species_domain,
            self.binding.active_species,
            pair_dependent=self.binding.pair_dependent,
        )
        require_identical(
            (self.declaration, self.binding),
            (declaration, binding),
            "Radial table declaration and species binding",
            StaleRadialTableBinding,
        )
        self._require_stored_extents(declaration, binding, mlp)
        expected = PreparedRadialTables(embedding, mlp, declaration, binding)
        require_identical(
            (self.grid, self.slope_status, self.resources),
            (expected.grid, expected.slope_status, expected.resources),
            "Radial table grid, slope status and resources",
            StaleRadialTableBinding,
        )
        require_realized(
            (self.values, self.slopes, self.projection),
            (expected.values, expected.slopes, expected.projection),
            "Radial table values, slopes and projection",
            StaleRadialTableBinding,
        )
        plan = CubicSplineSlopePlan(expected.grid, left="not-a-knot", right="clamped")
        recorded = _table_identity(
            declaration,
            binding,
            expected.grid,
            expected.embedding_id,
            expected.source_revision_id,
            plan.plan_id,
            self.values,
        )
        if (
            self.cutoff,
            self.output_width,
            self.embedding_id,
            self.source_revision_id,
            self.table_id,
        ) != (
            expected.cutoff,
            expected.output_width,
            expected.embedding_id,
            expected.source_revision_id,
            recorded,
        ):
            raise StaleRadialTableBinding(
                "Radial table identity does not match its source."
            )

    def _require_stored_extents(
        self,
        declaration: RadialTableDeclaration,
        binding: RadialSpeciesBinding,
        mlp: RadialMLP,
        /,
    ) -> None:
        """Bind declared node/row/width counts to the stored table arrays.

        Re-tabulation allocates from the declaration and binding, which are
        static metadata. Requiring them to equal the extents of the stored,
        already size-admitted arrays bounds that work by the payload itself, so
        corrupted metadata is refused before any derived allocation.
        """
        match declaration.layout:
            case "projected-width":
                width, projection_shape = mlp.output_width, None
            case "embedding-width":
                width = mlp.embedding_width
                projection_shape = (mlp.embedding_width, mlp.output_width)
            case unreachable:
                assert_never(unreachable)
        table_shape = (binding.row_count, declaration.node_count, width)
        stored_projection = None if self.projection is None else self.projection.shape
        if (
            self.values.shape != table_shape
            or self.slopes.shape != table_shape
            or self.slope_status.shape != (binding.row_count, width)
            or self.grid.node_count != declaration.node_count
            or stored_projection != projection_shape
        ):
            raise StaleRadialTableBinding(
                "Radial table arrays do not have the extents of their declaration "
                f"and species binding (expected values {table_shape})."
            )

    def support(
        self,
        distance: ArrayLike,
        sender_species: ArrayLike,
        receiver_species: ArrayLike,
        /,
    ) -> Array:
        """Whether each query lies inside the declared table support.

        Support is ``r >= grid_min`` for an actively bound species pair; radii at
        or beyond the cutoff are supported by the exact zero extension.
        """
        radius = jnp.asarray(distance, dtype=self.values.dtype)
        _, bound = self.binding.rows(
            jnp.asarray(sender_species, dtype=jnp.int32),
            jnp.asarray(receiver_species, dtype=jnp.int32),
        )
        return bound & (radius >= self.declaration.grid_min)

    def evaluate(
        self,
        distance: ArrayLike,
        sender_species: ArrayLike,
        receiver_species: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
    ) -> Array:
        """Evaluate tabulated radial outputs ``(..., output_width)``.

        Valid queries outside the declared support raise; there is no exact
        fallback under a tabulated realization. Invalid (padded) lanes evaluate
        at an interior node and return exact zeros.
        """
        radius = jnp.asarray(distance, dtype=self.values.dtype)
        sender = jnp.asarray(sender_species, dtype=jnp.int32)
        receiver = jnp.asarray(receiver_species, dtype=jnp.int32)
        if sender.shape != radius.shape or receiver.shape != radius.shape:
            raise ValueError("Species indices must match the distance shape.")
        mask = (
            jnp.ones(radius.shape, dtype=jnp.bool_)
            if valid is None
            else jnp.asarray(valid, dtype=jnp.bool_)
        )
        if mask.shape != radius.shape:
            raise ValueError("valid must match the distance shape.")
        rows, bound = self.binding.rows(sender, receiver)
        supported = bound & (radius >= self.declaration.grid_min)
        radius = eqx.error_if(
            radius,
            jnp.any(mask & ~supported),
            "Radial table query lies outside the declared support (below grid_min "
            "or an unbound species pair).",
        )
        inside = mask & supported & (radius < self.cutoff)
        sentinel = jnp.asarray(
            0.5 * (self.declaration.grid_min + self.cutoff), radius.dtype
        )
        # Queries are already admitted to [grid_min, cutoff) or replaced by the
        # sentinel, so no clamping is needed. Clamping would be wrong at the
        # support boundary: clip's tie gradient halves dr at r == grid_min.
        location = self.grid.locate(
            jnp.where(inside, radius, sentinel), bounds="extrapolate"
        )
        upper = location.lower + 1
        rows = jnp.where(inside, rows, jnp.zeros_like(rows))
        output = cubic_hermite_segment(
            self.values[rows, location.lower],
            self.values[rows, upper],
            self.slopes[rows, location.lower],
            self.slopes[rows, upper],
            location.fraction,
            self.grid.spacing,
        )
        if self.projection is not None:
            output = output @ self.projection
        return jnp.where(inside[..., None], output, jnp.zeros_like(output))

    def cutoff_jets(self) -> tuple[Array, Array, Array]:
        """Return (left slope at cutoff, left curvature at cutoff, max interior
        curvature jump) over every row and tabulated channel."""
        jets = jax.vmap(
            lambda values, slopes: cubic_hermite_knot_jets(self.grid, values, slopes)
        )(self.values, self.slopes)
        slope = jets.segment_end[:, 0, -1]
        curvature = jets.segment_end[:, 1, -1]
        interior = jets.segment_end[:, 1, :-1] - jets.segment_start[:, 1, 1:]
        if self.projection is not None:
            slope = slope @ self.projection
            curvature = curvature @ self.projection
            interior = interior @ self.projection
        return slope, curvature, jnp.max(jnp.abs(interior))


def prepare_radial_tables(
    embedding: RadialEmbedding,
    mlp: RadialMLP,
    declaration: RadialTableDeclaration,
    /,
    *,
    species_domain: Sequence[int],
    active_species: Sequence[int],
) -> PreparedRadialTables:
    """Tabulate an exact radial realization for the active species pairs."""
    binding = RadialSpeciesBinding(
        species_domain, active_species, pair_dependent=embedding.pair_dependent
    )
    return PreparedRadialTables(embedding, mlp, declaration, binding)


@final
class RadialTableQualificationPolicy(StrictModule, NonTrainableState):
    """Gate tolerances and probe counts, fixed before any result is collected."""

    value_tolerance: float = eqx.field(static=True)
    first_derivative_tolerance: float = eqx.field(static=True)
    second_derivative_tolerance: float | None = eqx.field(static=True)
    probe_count: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        value_tolerance: float,
        first_derivative_tolerance: float,
        second_derivative_tolerance: float | None = None,
        probe_count: int = 257,
    ) -> None:
        values = positive_finite_float(value_tolerance, "value_tolerance")
        first = positive_finite_float(
            first_derivative_tolerance, "first_derivative_tolerance"
        )
        second = (
            None
            if second_derivative_tolerance is None
            else positive_finite_float(
                second_derivative_tolerance, "second_derivative_tolerance"
            )
        )
        probes = positive_integer(probe_count, "probe_count")
        if probes < 8:
            raise ValueError("probe_count must be at least eight.")
        self.value_tolerance = values
        self.first_derivative_tolerance = first
        self.second_derivative_tolerance = second
        self.probe_count = probes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "radial-table-qualification-policy",
                "value_tolerance": values,
                "first_derivative_tolerance": first,
                "second_derivative_tolerance": second,
                "probe_count": probes,
            }
        )


@final
class RadialTableQualification(StrictModule, NonTrainableState):
    """Retained exact-versus-table evidence in internal radial feature units.

    ``maximum_errors`` rows are ``(probe category, derivative order, max abs
    error)``. Failures are retained, never discarded; internal feature error is
    not an energy, force or stress bound.
    """

    table_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    source_revision_id: str = eqx.field(static=True)
    maximum_errors: tuple[tuple[str, int, float], ...] = eqx.field(static=True)
    cutoff_slope: float = eqx.field(static=True)
    cutoff_second_derivative_jump: float = eqx.field(static=True)
    interior_second_derivative_jump: float = eqx.field(static=True)
    below_support_refused: bool = eqx.field(static=True)
    beyond_cutoff_exact_zero: bool = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)
    admitted_derivative_order: int = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return not self.failures

    def require_passed(self) -> None:
        if self.failures:
            raise RadialTableQualificationError(
                "Radial table qualification failed: " + "; ".join(self.failures)
            )


# Probe categories whose derivatives are compared; at the lower bound only
# values are gated, because exact network derivatives at ``r -> 0`` lose
# precision to cancellation in ``sin(w r) / r`` and are not a table property.
_VALUE_ONLY_CATEGORIES = frozenset({"lower-bound"})


def _probe_radii(
    grid_min: float, cutoff: float, node_count: int, probes: int, /
) -> dict[str, np.ndarray]:
    """Held-out radius sets; only the ``nodes`` category coincides with knots."""
    spacing = (cutoff - grid_min) / (node_count - 1)
    nodes = grid_min + spacing * np.arange(1, node_count - 1, dtype=np.float64)
    offsets = np.asarray([1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4], dtype=np.float64)
    return {
        "nodes": nodes,
        "mid-spans": nodes - 0.5 * spacing,
        "linear": np.linspace(
            grid_min + 0.37 * spacing, cutoff - 0.29 * spacing, probes, dtype=np.float64
        ),
        "logarithmic": np.geomspace(
            max(grid_min, 1.0e-3 * cutoff), cutoff * (1.0 - 1.0e-7), probes
        ),
        "near-cutoff": cutoff - offsets * spacing,
        "lower-bound": np.asarray([grid_min, grid_min + 1.0e-3 * spacing]),
    }


def _radial_jets(
    function: _RadialFunction, radii: Array, pairs: np.ndarray, /
) -> tuple[Array, Array, Array]:
    """Value, first and second radial derivatives ``(rows, radii, width)``."""

    def jet(radius: Array, sender: Array, receiver: Array) -> tuple[Array, Array, Array]:
        def value(r: Array) -> Array:
            return function(r, sender, receiver)

        first = jax.jacfwd(value)
        return value(radius), first(radius), jax.jacfwd(first)(radius)

    per_radius = jax.vmap(jet, in_axes=(0, None, None))
    return jax.vmap(per_radius, in_axes=(None, 0, 0))(
        radii, jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])
    )


def qualify_radial_tables(
    tables: PreparedRadialTables,
    embedding: RadialEmbedding,
    mlp: RadialMLP,
    policy: RadialTableQualificationPolicy,
    /,
) -> RadialTableQualification:
    """Compare exact network and table on held-out radii for every active row."""
    if not isinstance(tables, PreparedRadialTables):
        raise TypeError("tables must be PreparedRadialTables.")
    if not isinstance(policy, RadialTableQualificationPolicy):
        raise TypeError("policy must be a RadialTableQualificationPolicy.")
    tables.require_current(embedding, mlp)
    dtype = tables.values.dtype
    pairs = np.asarray(tables.binding.row_pairs(), dtype=np.int32).reshape((-1, 2))

    def exact(radius: Array, sender: Array, receiver: Array) -> Array:
        return mlp(embedding.unmasked(radius, sender, receiver))

    def tabulated(radius: Array, sender: Array, receiver: Array) -> Array:
        return tables.evaluate(radius, sender, receiver)

    tolerances = (policy.value_tolerance, policy.first_derivative_tolerance)
    errors: list[tuple[str, int, float]] = []
    failures: list[str] = []
    failed_orders: set[int] = set()
    probes = _probe_radii(
        tables.declaration.grid_min,
        tables.cutoff,
        tables.declaration.node_count,
        policy.probe_count,
    )
    for category, radii in probes.items():
        points = jnp.asarray(radii, dtype=dtype)
        exact_jets = _radial_jets(exact, points, pairs)
        table_jets = _radial_jets(tabulated, points, pairs)
        orders = 1 if category in _VALUE_ONLY_CATEGORIES else 3
        for order in range(orders):
            difference = np.asarray(exact_jets[order] - table_jets[order])
            error = float(np.max(np.abs(difference)))
            errors.append((category, order, error))
            if order == 2:
                second = policy.second_derivative_tolerance
                if second is not None and not error <= second:
                    failures.append(
                        f"{category} derivative-2 error {error:.3e} exceeds {second:.3e}"
                    )
                continue
            if not error <= tolerances[order]:
                failed_orders.add(order)
                failures.append(
                    f"{category} derivative-{order} error {error:.3e} exceeds "
                    f"{tolerances[order]:.3e}"
                )
    slope, curvature, interior_jump = tables.cutoff_jets()
    cutoff_slope = float(np.max(np.abs(np.asarray(slope))))
    cutoff_jump = float(np.max(np.abs(np.asarray(curvature))))
    interior = float(np.asarray(interior_jump))
    if not cutoff_slope <= policy.first_derivative_tolerance:
        failed_orders.add(1)
        failures.append(f"cutoff slope {cutoff_slope:.3e} is not zero within tolerance")
    if policy.second_derivative_tolerance is not None:
        # Interior second-derivative errors are recorded above; the exact cutoff
        # jet vanishes through second order while the table's left curvature
        # does not, so the join is C1 and Hessian use is refused.
        failures.append(
            "second derivatives are not admitted: the cutoff join is C1 "
            f"(left curvature {cutoff_jump:.3e} versus exact zero extension)"
        )
    edge = jnp.asarray(pairs[:1, 0])
    below = tables.support(
        jnp.asarray([0.5 * tables.declaration.grid_min], dtype=dtype), edge, edge
    )
    below_refused = not bool(np.asarray(below)[0])
    if not below_refused:
        failures.append("radii below grid_min are not refused")
    beyond_radii = jnp.asarray(
        [tables.cutoff, tables.cutoff * (1.0 + 1.0e-9), 2.0 * tables.cutoff], dtype=dtype
    )
    beyond_zero = bool(
        np.all(np.asarray(_radial_jets(tabulated, beyond_radii, pairs)[0]) == 0.0)
    ) and bool(np.all(np.asarray(_radial_jets(exact, beyond_radii, pairs)[0]) == 0.0))
    if not beyond_zero:
        failed_orders.add(0)
        failures.append("table or exact network is nonzero at or beyond the cutoff")
    if 0 in failed_orders:
        admitted = -1
    elif 1 in failed_orders:
        admitted = 0
    else:
        admitted = _TABLE_COORDINATE_DERIVATIVE_ORDER
    return RadialTableQualification(
        table_id=tables.table_id,
        policy_id=policy.policy_id,
        source_revision_id=tables.source_revision_id,
        maximum_errors=tuple(errors),
        cutoff_slope=cutoff_slope,
        cutoff_second_derivative_jump=cutoff_jump,
        interior_second_derivative_jump=interior,
        below_support_refused=below_refused,
        beyond_cutoff_exact_zero=beyond_zero,
        failures=tuple(failures),
        admitted_derivative_order=admitted,
    )


for _artifact in (
    PreparedRadialTables,
    RadialSpeciesBinding,
    RadialTableDeclaration,
    RadialTableResources,
):
    register_artifact_value(f"phydrax.nn.atomistic:{_artifact.__name__}", _artifact)
del _artifact
# The uniform grid is registered by its first artifact consumer: the leaf
# interpolation package is imported before the model artifact registry.
register_artifact_value("phydrax.interpolation.internal:UniformNodeGrid", UniformNodeGrid)


__all__ = [
    "PreparedRadialTables",
    "prepare_radial_tables",
    "qualify_radial_tables",
    "radial_source_revision",
    "radial_table_resources",
    "RadialSpeciesBinding",
    "RadialTableDeclaration",
    "RadialTableLayout",
    "RadialTableLayoutChoice",
    "RadialTableObjective",
    "RadialTableQualification",
    "RadialTableQualificationError",
    "RadialTableQualificationPolicy",
    "RadialTableResources",
    "select_radial_table_layout",
    "StaleRadialTableBinding",
]
