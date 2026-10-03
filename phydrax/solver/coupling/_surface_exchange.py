#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative point-query exchange, native Langmuir transport, host relocation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation._stencil import GatherStencil
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_finite_float, positive_integer
from ...discretization import (
    PreparedFieldQuery,
    PreparedFieldReconstruction,
    TopologyEpochTransition,
)
from ...discretization.meshfree._epochs import (
    commit_meshfree_epoch,
    MeshfreeEpochChange,
    MeshfreeEpochReceipt,
    stage_meshfree_epoch,
)
from ...discretization.meshfree._exterior import PreparedMeshfreeExteriorCalculus
from ...interfacial_transport._core import AdsorptionKinetics
from ...interfacial_transport._coupled import CoupledBulkSurfaceTransport
from ...lifecycle import Composition, CompositionEntry
from ...linalg import ArraySpace
from ...typing import Bool, checked, Dim, Float, Int32, parse, Scalar
from .._fixed_step import AbstractFixedStepMethod, FixedStepResult
from .._partitioned_coupling_types import CouplingPort
from ._components import AbstractReconstructionComponent, AbstractSpatialComponent
from ._contributions import (
    AbstractContributionResidual,
    ContributionEndpoint,
    ResidualContribution,
)
from ._interfaces import InterfaceBinding, InterfaceOwner
from ._laws import (
    AbstractCouplingLaw,
    AbstractInterfaceFlux,
    AbstractLawCertificate,
    FieldStates,
    InterfaceDefectReport,
    PreparedLaw,
)
from ._meshfree_components import MeshfreeComponent
from ._method_participants import FixedStepCouplingParticipant, MethodWindowBinding
from ._parameters import RuntimeInput


SurfaceDeposition: TypeAlias = Literal["signed", "positive"]


class _BulkPointDim(Dim):
    """Bulk concentration and amount coordinates."""


class _SurfacePointDim(Dim):
    """Admitted surface quadrature coordinates."""


class _SurfaceCoordinateDim(Dim):
    """Ambient point and normal components."""


class _SurfaceEdgeDim(Dim):
    """Actual prepared surface graph edges."""


@final
class LangmuirAdsorptionFlux(AbstractInterfaceFlux):
    """Native interface conormal convention: negative adsorption leaves bulk."""

    kinetics: AdsorptionKinetics

    @checked
    def __init__(self, kinetics: AdsorptionKinetics, /) -> None:
        self.kinetics = kinetics

    @property
    def affine(self) -> bool:
        return False

    @property
    def trace_degree(self) -> int:
        return 2

    def evaluate(
        self, minus: Array, plus: Array, points: Array, normals: Array, args: object, /
    ) -> Array:
        del points, normals, args
        return -self.kinetics.flux(minus, plus)


@final
class SurfaceExchangeEvidence(StrictModule, NonTrainableState):
    """Host preparation evidence of one exchange-law revision.

    ``surface_id`` is the exact native surface source (owner revision, point
    enumeration, and capacity) the law was prepared on. ``deposition`` is the
    declared amount partition and ``minimum_query_weight`` the smallest valid
    nonzero weight of a bounded gather route, or ``None`` for an owner route
    (for example a finite-element cell route) that publishes no gather
    partition. ``displacement`` and ``lag`` are the host relocation motion and
    geometry lag of a fixed-topology window refresh; an epoch relocation has no
    point correspondence and reports its transfer ledger instead.
    """

    query_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    deposition: SurfaceDeposition = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    constant_reproduction_error: float = eqx.field(static=True)
    minimum_query_weight: float | None = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    refresh_time: float = eqx.field(static=True)
    displacement: float = eqx.field(static=True)
    lag: float = eqx.field(static=True)
    geometry_differentiated_within_window: bool = eqx.field(static=True, default=False)


@final
class SurfaceEpochRelocation(StrictModule):
    """Atomic epoch relocation of one surface exchange.

    ``published`` holds only when every surface state route conserved its
    content and the host boundary was accepted. Otherwise ``law``, ``surface``,
    and ``surface_states`` are the unchanged source objects. Residuals and
    tolerances follow the input state order; ``epoch`` is the complete native
    epoch account.
    """

    law: SurfaceExchangeLaw
    surface: MeshfreeComponent
    surface_states: tuple[Array, ...]
    conservation_residuals: Array
    content_tolerances: Array
    epoch: MeshfreeEpochReceipt
    published: bool = eqx.field(static=True)


def _admit_query(query: PreparedFieldQuery, tolerance: float) -> float:
    if not isinstance(query, PreparedFieldQuery):
        raise TypeError("query must be PreparedFieldQuery.")
    if not query.complete or not query.coefficient_linear or query.value_shape:
        raise ValueError("Exchange requires a complete scalar coefficient-linear query.")
    if any(query.derivative):
        raise ValueError("Exchange requires a value query, not a derivative query.")
    if len(query.coefficient_shape) != 1:
        raise ValueError("Exchange bulk coordinates must be scalar point coefficients.")
    constant = np.asarray(query.apply(jnp.ones(query.coefficient_shape)))
    error = float(np.max(np.abs(constant - 1.0)))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(
            "Bulk query does not reproduce constants; transpose exchange is not conservative."
        )
    return error


def _carrying(route: GatherStencil, /) -> np.ndarray:
    """Route entries that carry amount: valid and nonzero."""
    return np.asarray(route.valid) & (np.asarray(route.weights) != 0.0)


def _positive_partition(query: PreparedFieldQuery, /) -> tuple[GatherStencil, np.ndarray]:
    """The owning gather route of a nonnegative amount partition and its carriers."""
    route = query.route
    if not isinstance(route, GatherStencil):
        raise ValueError(
            "A positive amount partition requires an owning bounded GatherStencil query."
        )
    carrying = _carrying(route)
    if np.any(np.asarray(route.weights)[carrying] < 0):
        raise ValueError(
            "Signed query weights cannot define a positive amount partition."
        )
    return route, carrying


def _minimum_query_weight(query: PreparedFieldQuery, /) -> float | None:
    route = query.route
    if not isinstance(route, GatherStencil):
        return None
    return float(np.min(np.asarray(route.weights)[_carrying(route)]))


def _surface_source_id(surface: MeshfreeComponent, field: str, /) -> str:
    """Exact identity of a native surface source; never inferred from shapes."""
    surface.field(field)
    owner = surface.owner
    # The exterior owner carries no prepared revision: its complete numeric
    # content (points, graph, metric, and measures) is its revision.
    revision = (
        array_tree_fingerprint(owner)
        if isinstance(owner, PreparedMeshfreeExteriorCalculus)
        else owner.prepared_id
    )
    return canonical_fingerprint(
        {
            "kind": "surface-exchange-source",
            "owner_id": surface.owner_id,
            "owner_revision": revision,
            "reconstruction": surface.reconstruction.reconstruction_id,
            "field": field,
            "points": np.asarray(owner.points),
            "capacity": np.asarray(surface.mass_diagonal),
        }
    )


@final
class _ExchangeResidual(AbstractContributionResidual):
    __strict_contract__ = True
    query: PreparedFieldQuery
    measures: Float[_SurfacePointDim]
    normals: Float[_SurfacePointDim, _SurfaceCoordinateDim]
    flux: AbstractInterfaceFlux

    @property
    def runtime_inputs(self) -> tuple[RuntimeInput, ...]:
        return self.flux.runtime_inputs

    def evaluate(self, inputs: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        bulk, surface = inputs
        density = self.flux.evaluate(
            self.query.apply(bulk), surface, self.query.points, self.normals, args
        )
        if density.shape != self.measures.shape:
            raise ValueError(
                "Interface flux must return one scalar density per surface query."
            )
        amount = self.measures * density
        # Residual convention M dc/dt + Kc - outward-conormal-load = 0.
        return (-self.query.transpose(amount), amount)


@final
class _ExchangeCertificate(AbstractLawCertificate):
    law_id: str = eqx.field(static=True)
    bulk: ContributionEndpoint
    surface: ContributionEndpoint
    residual: _ExchangeResidual
    deposition: SurfaceDeposition = eqx.field(static=True)

    def defects(
        self,
        fields: FieldStates,
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> InterfaceDefectReport:
        del law_state
        bulk = fields[(self.bulk.owner, self.bulk.block)]
        surface = fields[(self.surface.owner, self.surface.block)]
        bulk_rows, surface_rows = self.residual.evaluate((bulk, surface), args)
        balance = jnp.abs(jnp.sum(bulk_rows) + jnp.sum(surface_rows))
        scale = jnp.maximum(1.0, jnp.sum(jnp.abs(surface_rows)))
        physical = jnp.all(jnp.isfinite(bulk)) & jnp.all(jnp.isfinite(surface))
        flux = self.residual.flux
        if isinstance(flux, LangmuirAdsorptionFlux):
            physical = physical & jnp.all(bulk >= 0.0) & jnp.all(surface >= 0.0)
            physical = physical & jnp.all(self.residual.query.apply(bulk) >= 0.0)
            physical = physical & jnp.all(
                surface <= flux.kinetics.maximum_surface_concentration_mol_m2
            )
        names: tuple[str, ...] = ("amount_balance", "admissibility")
        values = [balance, jnp.where(physical, 0.0, 1.0)]
        scales = [scale, jnp.asarray(1.0)]
        match self.deposition:
            case "signed":
                pass
            case "positive":
                names = (*names, "deposition_sign")
                values.append(self._deposition_sign(surface_rows))
                scales.append(scale)
            case unknown:
                assert_never(unknown)
        return InterfaceDefectReport(
            self.law_id,
            names,
            (True,) * len(names),
            jnp.stack(values),
            jnp.stack(scales),
        )

    def _deposition_sign(self, surface_rows: Array, /) -> Array:
        """Amount gained by any bulk node through the one-signed deposit parts.

        The surface gains ``d = -surface_rows`` and bulk node ``b`` loses
        ``(Q^T d)_b``. A nonnegative partition keeps the withdrawal driven by
        adsorbing points, ``Q^T max(d, 0)``, nonnegative at every node and the
        release driven by desorbing points, ``Q^T min(d, 0)``, nonpositive: no
        bulk node gains amount while the surface adsorbs.
        """
        deposit = -surface_rows
        query = self.residual.query
        adsorbed = query.transpose(jnp.maximum(deposit, 0.0))
        released = query.transpose(jnp.minimum(deposit, 0.0))
        return jnp.sum(jnp.maximum(-adsorbed, 0.0) + jnp.maximum(released, 0.0))


_EPOCH_ENTRY = "surface/epoch"
_SOURCE_ENTRY = "surface/source"
_LAW_ENTRY = "surface/exchange"


def _state_entry(slot: int, /) -> str:
    return f"surface/state/{slot}"


def _exchange_entries(
    law: SurfaceExchangeLaw, surface: MeshfreeComponent, epoch: CompositionEntry, /
) -> tuple[CompositionEntry, CompositionEntry]:
    """The surface source and exchange law prepared against one surface epoch."""
    identity = law.evidence.surface_id
    source = CompositionEntry(
        surface,
        entry_id=_SOURCE_ENTRY,
        role="discretization",
        owner_id=law.law_id,
        structure_id=identity,
        revision_id=identity,
        semantics_id="surface-exchange-source",
        dependencies=(epoch.binding("structure"),),
    )
    exchange = CompositionEntry(
        law,
        entry_id=_LAW_ENTRY,
        role="interface-route",
        owner_id=law.law_id,
        structure_id=law.evidence.query_id,
        revision_id=law.evidence.query_id,
        semantics_id="surface-exchange-law",
        dependencies=(source.binding("revision"),),
    )
    return source, exchange


@final
class SurfaceExchangeLaw(AbstractCouplingLaw, NonTrainableState):
    """Non-facet exchange between a reconstructing bulk and a meshfree surface.

    The bulk endpoint is any reconstruction-capable component (meshfree,
    finite-element, ...) whose complete scalar value query reproduces
    constants, so the exact coordinate transpose pairs bulk loss with surface
    gain. The surface endpoint is a native ``MeshfreeComponent`` whose owner
    revision, point enumeration, and capacity diagonal are recorded at
    construction and verified by ``prepare``; a same-shaped source of another
    revision is refused.

    ``deposition="signed"`` admits any constant-reproducing route.
    ``"positive"`` additionally requires a nonnegative bounded gather
    partition and certifies at every solution that no bulk node gains amount
    while the surface adsorbs.

    Queries are frozen inside numerical windows. ``refresh_at_window`` is a
    host preparation boundary for fixed-topology motion with explicit
    displacement and geometry-lag evidence; ``relocate_at_epoch`` rebinds the
    law, its surface, and the surface state history atomically across a
    meshfree topology epoch. Neither differentiates relocation.
    """

    __strict_contract__ = True
    law_id: str = eqx.field(static=True)
    bulk: ContributionEndpoint
    surface: ContributionEndpoint
    query: PreparedFieldQuery
    measures: Float[_SurfacePointDim]
    normals: Float[_SurfacePointDim, _SurfaceCoordinateDim]
    flux: AbstractInterfaceFlux
    deposition: SurfaceDeposition = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    evidence: SurfaceExchangeEvidence

    @checked
    def __init__(
        self,
        bulk: ContributionEndpoint,
        surface: ContributionEndpoint,
        query: PreparedFieldQuery,
        surface_component: MeshfreeComponent,
        normals: ArrayLike,
        flux: AbstractInterfaceFlux,
        /,
        *,
        deposition: SurfaceDeposition = "signed",
        law_id: str = "surface_exchange",
        constant_tolerance: float = 1e-10,
        geometry_epoch: int = 0,
        refresh_time: float = 0.0,
        displacement: float = 0.0,
        lag: float = 0.0,
    ) -> None:
        if bulk.space != "full" or surface.space != "full" or bulk.owner == surface.owner:
            raise ValueError(
                "Exchange endpoints must name full fields of distinct components."
            )
        if surface_component.name != surface.owner:
            raise ValueError("surface_component must own the declared surface endpoint.")
        deposition_ = parse(deposition, SurfaceDeposition, "deposition")
        tolerance = positive_finite_float(constant_tolerance, "constant_tolerance")
        error = _admit_query(query, tolerance)
        match deposition_:
            case "signed":
                pass
            case "positive":
                _positive_partition(query)
            case unknown:
                assert_never(unknown)
        if surface_component.field(surface.block).full_space.shape != query.output_shape:
            raise ValueError(
                "Surface field coordinates do not match the exchange quadrature."
            )
        if not np.array_equal(
            np.asarray(surface_component.owner.points), np.asarray(query.points)
        ):
            raise ValueError(
                "Surface endpoint must own the actual query point enumeration."
            )
        normal = np.asarray(normals, dtype=np.float64)
        if normal.shape != query.points.shape or not np.all(np.isfinite(normal)):
            raise ValueError("Normals must match the finite surface query points.")
        if not np.allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("Surface exchange normals must be unit vectors.")
        if (
            isinstance(geometry_epoch, bool)
            or not isinstance(geometry_epoch, int)
            or geometry_epoch < 0
        ):
            raise ValueError("geometry_epoch must be a nonnegative integer.")
        if (
            not np.isfinite(refresh_time)
            or not np.isfinite(displacement)
            or displacement < 0
            or not np.isfinite(lag)
            or lag < 0
        ):
            raise ValueError(
                "Refresh time and nonnegative displacement/lag must be finite."
            )
        self.law_id = canonical_identifier(law_id, "law_id")
        self.bulk, self.surface, self.query = bulk, surface, query
        # Exchange measures are the surface owner's native capacity diagonal.
        self.measures = jnp.asarray(surface_component.mass_diagonal, dtype=jnp.float64)
        self.normals, self.flux = jnp.asarray(normal), flux
        self.deposition = deposition_
        self.tolerance = tolerance
        self.evidence = SurfaceExchangeEvidence(
            query.query_id,
            _surface_source_id(surface_component, surface.block),
            deposition_,
            query.complete,
            error,
            _minimum_query_weight(query),
            geometry_epoch,
            float(refresh_time),
            float(displacement),
            float(lag),
        )

    @property
    def bindings(self) -> tuple[InterfaceBinding, ...]:
        return ()

    @property
    def runtime_inputs(self) -> tuple[RuntimeInput, ...]:
        return self.flux.runtime_inputs

    def _bulk_reconstruction(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> PreparedFieldReconstruction:
        bulk = components[self.bulk.owner]
        if not isinstance(bulk, AbstractReconstructionComponent):
            raise TypeError("Bulk exchange requires the owning field reconstruction.")
        if bulk.field(self.bulk.block).full_space.shape != self.query.coefficient_shape:
            raise ValueError("Query coefficients do not match the declared bulk field.")
        return bulk.prepare_field_reconstruction(self.bulk.block)

    def _current_surface(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> MeshfreeComponent:
        surface = components[self.surface.owner]
        if not isinstance(surface, MeshfreeComponent):
            raise TypeError(
                "Surface exchange requires a native nodal MeshfreeComponent endpoint."
            )
        if _surface_source_id(surface, self.surface.block) != self.evidence.surface_id:
            raise ValueError(
                "Surface endpoint is not the exact native source (owner revision, "
                "point enumeration, capacity) this exchange was prepared on; "
                "refresh or relocate the exchange instead of reusing it."
            )
        return surface

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        del interface_owners
        self.query.require_reconstruction(self._bulk_reconstruction(components))
        self._current_surface(components)
        residual = _ExchangeResidual(self.query, self.measures, self.normals, self.flux)
        contribution = ResidualContribution(
            (self.bulk, self.surface),
            (self.bulk, self.surface),
            residual,
            affine=self.flux.affine,
            law_id=self.law_id,
            imposition_id=self.law_id,
        )
        # Point queries are not facets; field ownership is declared by endpoints.
        return PreparedLaw(
            self.law_id,
            binding_id=None,
            state_blocks=(),
            row_blocks=(),
            contributions=(contribution,),
            impositions=(),
            certificate=_ExchangeCertificate(
                self.law_id, self.bulk, self.surface, residual, self.deposition
            ),
            evidence=self.evidence,
        )

    def refresh_at_window(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        normals: ArrayLike,
        /,
        *,
        window_start: float,
        geometry_time: float,
        geometry_epoch: int,
        maximum_displacement: float,
        maximum_lag: float,
    ) -> SurfaceExchangeLaw:
        """Relocate fixed-topology sites at a host window boundary.

        The moved surface is the actual surface endpoint in ``components``; its
        points, capacity, and owner revision become the refreshed identity.
        Stale epochs, excess motion or lag, and changed point counts (an epoch
        relocation) are refused.
        """
        if geometry_epoch <= self.evidence.geometry_epoch:
            raise ValueError("Moving query refresh requires a newer geometry epoch.")
        if (
            not np.isfinite(window_start)
            or not np.isfinite(geometry_time)
            or window_start < self.evidence.refresh_time
        ):
            raise ValueError(
                "Moving exchange refresh requires a finite nondecreasing host window boundary."
            )
        surface = components[self.surface.owner]
        if not isinstance(surface, MeshfreeComponent):
            raise TypeError(
                "Surface exchange requires a native nodal MeshfreeComponent endpoint."
            )
        values = np.asarray(surface.owner.points, dtype=np.float64)
        if values.shape != self.query.points.shape:
            raise ValueError(
                "Moving exchange refresh preserves surface topology; a changed point "
                "set is an epoch relocation."
            )
        displacement = float(
            np.max(np.linalg.norm(values - np.asarray(self.query.points), axis=1))
        )
        lag = float(window_start - geometry_time)
        if (
            not np.isfinite(maximum_displacement)
            or maximum_displacement < 0
            or not np.isfinite(maximum_lag)
            or maximum_lag < 0
        ):
            raise ValueError("Motion and lag bounds must be finite and nonnegative.")
        if displacement > maximum_displacement or not 0 <= lag <= maximum_lag:
            raise ValueError(
                "Moving exchange exceeds its displacement or geometry-lag admission."
            )
        query = self._bulk_reconstruction(components).prepare_query(
            values, coverage="complete"
        )
        return SurfaceExchangeLaw(
            self.bulk,
            self.surface,
            query,
            surface,
            normals,
            self.flux,
            deposition=self.deposition,
            law_id=self.law_id,
            constant_tolerance=self.tolerance,
            geometry_epoch=geometry_epoch,
            refresh_time=window_start,
            displacement=displacement,
            lag=lag,
        )

    def _epoch_routes(
        self,
        surface: MeshfreeComponent,
        surface_states: Sequence[ArrayLike],
        transitions: Sequence[TopologyEpochTransition],
        /,
    ) -> tuple[tuple[Array, ...], tuple[TopologyEpochTransition, ...]]:
        """One conservative capacity-to-capacity route per surface state."""
        states = tuple(jnp.asarray(value, dtype=jnp.float64) for value in surface_states)
        routes = tuple(transitions)
        if not states or len(states) != len(routes):
            raise ValueError(
                "Every surface state (current first, then live histories) needs "
                "exactly one epoch route."
            )
        if any(not isinstance(route, TopologyEpochTransition) for route in routes):
            raise TypeError(
                "Surface state routes must be TopologyEpochTransition values."
            )
        if any(state.shape != self.measures.shape for state in states):
            raise ValueError("Surface states must use the exchange surface coordinates.")
        source, target = np.asarray(self.measures), np.asarray(surface.mass_diagonal)
        if any(
            not np.array_equal(np.asarray(route.source_measures), source)
            or not np.array_equal(np.asarray(route.target_measures), target)
            for route in routes
        ):
            raise ValueError(
                "Surface state routes must conserve content between the exchange "
                "capacity and the target surface capacity."
            )
        return states, routes

    def relocate_at_epoch(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        surface: MeshfreeComponent,
        normals: ArrayLike,
        surface_states: Sequence[ArrayLike],
        transitions: Sequence[TopologyEpochTransition],
        change: MeshfreeEpochChange,
        /,
        *,
        window_start: float,
        accepted_boundary: bool,
    ) -> SurfaceEpochRelocation:
        """Rebind exchange, surface, and surface states across one topology epoch.

        ``components`` hold the bulk and the current surface (verified against
        this law's identity); ``surface`` is the target-epoch surface endpoint,
        whose point count may differ. The re-prepared query is a derived
        artifact of the target surface; every surface state (current field
        first, then live histories) crosses through its own conservative
        ``transitions`` route of ``change``. Inadmissible target preparation
        raises before anything is staged; a failed state route or an
        unaccepted boundary publishes nothing and returns the source law,
        surface, and states.
        """
        if not isinstance(change, MeshfreeEpochChange):
            raise TypeError("change must be a MeshfreeEpochChange.")
        if not isinstance(accepted_boundary, bool):
            raise TypeError("accepted_boundary must be a bool.")
        if change.target.index <= self.evidence.geometry_epoch:
            raise ValueError("Epoch relocation requires a newer geometry epoch.")
        if not np.isfinite(window_start) or window_start < self.evidence.refresh_time:
            raise ValueError(
                "Epoch relocation requires a finite nondecreasing host window boundary."
            )
        current = self._current_surface(components)
        if not isinstance(surface, MeshfreeComponent):
            raise TypeError("The target surface must be a native MeshfreeComponent.")
        states, routes = self._epoch_routes(surface, surface_states, transitions)
        query = self._bulk_reconstruction(components).prepare_query(
            surface.owner.points, coverage="complete"
        )
        target_law = SurfaceExchangeLaw(
            self.bulk,
            self.surface,
            query,
            surface,
            normals,
            self.flux,
            deposition=self.deposition,
            law_id=self.law_id,
            constant_tolerance=self.tolerance,
            geometry_epoch=change.target.index,
            refresh_time=window_start,
        )
        epoch = CompositionEntry(
            change.source,
            entry_id=_EPOCH_ENTRY,
            role="topology",
            owner_id=self.law_id,
            structure_id=change.source.epoch_id,
            revision_id=change.source.epoch_id,
            semantics_id="surface-topology",
        )
        semantics = f"{self.surface.owner}/{self.surface.block}"
        state_entries = tuple(
            CompositionEntry(
                value,
                entry_id=_state_entry(slot),
                role="physical-state" if slot == 0 else "history",
                owner_id=self.law_id,
                structure_id=change.source.epoch_id,
                revision_id=f"{self.evidence.query_id}/state/{slot}",
                semantics_id=semantics,
                dependencies=(epoch.binding("structure"),),
            )
            for slot, value in enumerate(states)
        )
        # The staged epoch entry binds the target structure; derived artifacts
        # of the target surface are prepared against exactly that identity.
        target_epoch = CompositionEntry(
            change.target,
            entry_id=_EPOCH_ENTRY,
            role="topology",
            owner_id=self.law_id,
            structure_id=change.target.epoch_id,
            revision_id=change.change_id,
            semantics_id="surface-topology",
        )
        candidate = stage_meshfree_epoch(
            Composition(
                (epoch, *_exchange_entries(self, current, epoch), *state_entries),
                boundary_id=canonical_fingerprint(
                    {
                        "kind": "surface-exchange-boundary",
                        "query": self.evidence.query_id,
                        "window": float(window_start),
                    }
                ),
            ),
            change,
            epoch_entry=_EPOCH_ENTRY,
            remap={_state_entry(slot): route for slot, route in enumerate(routes)},
            reprepare=_exchange_entries(target_law, surface, target_epoch),
        )
        receipt = commit_meshfree_epoch(candidate, accepted_boundary=accepted_boundary)
        published = receipt.composition
        law, component = published.value(_LAW_ENTRY), published.value(_SOURCE_ENTRY)
        if not isinstance(law, SurfaceExchangeLaw) or not isinstance(
            component, MeshfreeComponent
        ):
            raise RuntimeError("The epoch transaction lost its exchange artifacts.")
        slots = range(len(states))
        order = jnp.asarray(
            [receipt.remapped.index(_state_entry(slot)) for slot in slots]
        )
        return SurfaceEpochRelocation(
            law,
            component,
            tuple(jnp.asarray(published.value(_state_entry(slot))) for slot in slots),
            receipt.conservation_residuals[order],
            receipt.content_tolerances[order],
            receipt,
            receipt.published,
        )


@final
class MeshfreeBulkSurfaceEvidence(StrictModule):
    """Native evidence of Langmuir bulk-surface transport over one step or window.

    `status` and `nonlinear_status` are the native film and nonlinear statuses,
    with `nonlinear_iterations` and `nonlinear_residual_norm` the solve work and
    final residual. `maximum_coverage` is the largest fractional Langmuir surface
    coverage and `minimum_amount_mol` the smallest amount of the candidate.
    `transferred_to_surface_mol` is the amount actually adsorbed per surface cell,
    spent only by accepted steps, while `candidate_transferred_to_surface_mol`
    keeps the raw candidate transfer, refused or not.
    `total_amount_residual_mol` is the candidate's closed-system amount defect.
    `query_complete` reports complete bulk-query coverage of every surface point
    and `metric_exact` whether the surface metric was admitted by its exact rather
    than relaxed moment equations.

    Over a window `MeshfreeBulkSurfaceMethod.reduce_evidence` reports the status of
    the refusing substep (else the accepted status), summed work and transfers,
    extremal coverage, amount, and residual norm, and the signed amount defect of
    largest magnitude.
    """

    __strict_contract__ = True
    status: Int32[Scalar]
    nonlinear_status: Int32[Scalar]
    nonlinear_iterations: Int32[Scalar]
    nonlinear_residual_norm: Float[Scalar]
    maximum_coverage: Float[Scalar]
    minimum_amount_mol: Float[Scalar]
    transferred_to_surface_mol: Float[_SurfacePointDim]
    candidate_transferred_to_surface_mol: Float[_SurfacePointDim]
    total_amount_residual_mol: Float[Scalar]
    query_complete: Bool[Scalar]
    metric_exact: Bool[Scalar]


@final
class MeshfreeBulkSurfaceMethod(AbstractFixedStepMethod, NonTrainableState):
    """Fixed-topology composition of the native conservative Langmuir solver.

    Only nonnegative sparse query routes can be amounts partitions. Signed MLS
    routes remain valid for SurfaceExchangeLaw but cannot claim positivity in
    this native amount-transport composition and are explicitly refused.
    """

    __strict_contract__ = True
    transport: CoupledBulkSurfaceTransport
    query: PreparedFieldQuery
    bulk_volumes: Float[_BulkPointDim]
    surface_measures: Float[_SurfacePointDim]
    surface_conductances: Float[_SurfaceEdgeDim]
    metric_exact: Bool[Scalar]
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        query: PreparedFieldQuery,
        surface_graph: PreparedMeshfreeExteriorCalculus,
        bulk_volumes: ArrayLike,
        kinetics: AdsorptionKinetics,
        /,
        *,
        surface_diffusivity: float = 0.0,
        tolerance: float = 1e-12,
        maximum_iterations: int = 30,
    ) -> None:
        _admit_query(query, tolerance)
        if not isinstance(surface_graph, PreparedMeshfreeExteriorCalculus):
            raise TypeError(
                "surface_graph must be the native prepared conservative exterior owner."
            )
        if not np.array_equal(np.asarray(query.points), np.asarray(surface_graph.points)):
            raise ValueError(
                "Exchange query must use the actual prepared surface graph points."
            )
        metric = surface_graph.metric_result
        if not bool(np.asarray(metric.accepted & metric.nonnegative)):
            raise ValueError(
                "Transport requires an accepted nonnegative native surface metric."
            )
        volume = np.asarray(bulk_volumes, dtype=np.float64)
        if volume.shape != query.coefficient_shape or not np.all(
            np.isfinite(volume) & (volume > 0)
        ):
            raise ValueError("Bulk volumes must be finite and strictly positive.")
        diffusivity = float(surface_diffusivity)
        if not np.isfinite(diffusivity) or diffusivity < 0:
            raise ValueError("surface_diffusivity must be finite and nonnegative.")
        route, carrying = _positive_partition(query)
        weights = np.asarray(route.weights)
        surface_indices = np.broadcast_to(
            np.arange(query.admitted_count)[:, None], weights.shape
        )[carrying]
        bulk_indices = np.asarray(route.indices)[carrying]
        values = weights[carrying]
        area = np.asarray(surface_graph.node_volumes)
        conductances = (
            jnp.zeros_like(surface_graph.metric_result.weights)
            if diffusivity == 0.0
            else surface_graph.diffusion(diffusivity).conductances
        )
        self.transport = CoupledBulkSurfaceTransport(
            surface_graph.pairs,
            bulk_indices,
            surface_indices,
            values,
            kinetics,
            bulk_size=volume.size,
            surface_size=area.size,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        self.query = query
        self.bulk_volumes = jnp.asarray(volume)
        self.surface_measures = jnp.asarray(area)
        self.surface_conductances = conductances
        self.metric_exact = jnp.asarray(metric.exact, dtype=jnp.bool_)
        self.method_id = canonical_fingerprint(
            {
                "kind": "meshfree-bulk-surface-method",
                "transport": self.transport.transport_id,
                "query": query.query_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: tuple[Array, Array],
        step_size: Array,
        args: object,
        /,
    ) -> FixedStepResult:
        del step_index, time, args
        bulk, surface = state
        result = self.transport.advance(
            bulk,
            surface,
            self.bulk_volumes,
            self.surface_measures,
            self.surface_conductances,
            step_size,
        )
        return FixedStepResult(
            candidate_state=(
                result.candidate_bulk_amount_mol,
                result.candidate_surface_amount_mol,
            ),
            accepted_state=(result.bulk_amount_mol, result.surface_amount_mol),
            successful=result.successful,
            residual=result.nonlinear_residual_norm,
            iterations=result.nonlinear_iterations,
            work=result.nonlinear_iterations,
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.asarray(0.0),
            evidence=MeshfreeBulkSurfaceEvidence(
                status=result.status,
                nonlinear_status=result.nonlinear_status,
                nonlinear_iterations=result.nonlinear_iterations,
                nonlinear_residual_norm=result.nonlinear_residual_norm,
                maximum_coverage=result.maximum_coverage,
                minimum_amount_mol=result.minimum_amount_mol,
                # A refused candidate spends no amount; its transfer stays a
                # candidate diagnostic.
                transferred_to_surface_mol=jnp.where(
                    result.successful, result.transferred_to_surface_mol, 0.0
                ),
                candidate_transferred_to_surface_mol=result.transferred_to_surface_mol,
                total_amount_residual_mol=result.total_amount_residual_mol,
                query_complete=jnp.asarray(self.query.complete, dtype=jnp.bool_),
                metric_exact=self.metric_exact,
            ),
        )

    def reduce_evidence(
        self,
        evidence: object,
        executed: Array,
        successful: Array,
        /,
    ) -> MeshfreeBulkSurfaceEvidence:
        """Window evidence of consecutive substeps with physical reduction semantics.

        Spent transfers and nonlinear work are additive; coverage, minimum amount,
        and residual norm are extremal over executed substeps; the status is that of
        the refusing substep, or the accepted status when every executed substep
        was accepted; the amount defect keeps the sign of its largest magnitude.
        """
        if not isinstance(evidence, MeshfreeBulkSurfaceEvidence):
            raise TypeError("Bulk-surface evidence must be MeshfreeBulkSurfaceEvidence.")
        refused = executed & ~successful
        # The first substep always runs, so index 0 is the accepted status when no
        # executed substep was refused.
        terminal = jnp.where(jnp.any(refused), jnp.argmax(refused), 0)
        committed = (executed & successful)[:, None]
        worst = jnp.argmax(
            jnp.where(executed, jnp.abs(evidence.total_amount_residual_mol), -1.0)
        )
        return MeshfreeBulkSurfaceEvidence(
            status=evidence.status[terminal],
            nonlinear_status=evidence.nonlinear_status[terminal],
            nonlinear_iterations=jnp.sum(
                jnp.where(executed, evidence.nonlinear_iterations, 0)
            ).astype(jnp.int32),
            nonlinear_residual_norm=jnp.max(
                jnp.where(executed, evidence.nonlinear_residual_norm, 0.0)
            ),
            maximum_coverage=jnp.max(
                jnp.where(executed, evidence.maximum_coverage, -jnp.inf)
            ),
            minimum_amount_mol=jnp.min(
                jnp.where(executed, evidence.minimum_amount_mol, jnp.inf)
            ),
            transferred_to_surface_mol=jnp.sum(
                jnp.where(committed, evidence.transferred_to_surface_mol, 0.0), axis=0
            ),
            candidate_transferred_to_surface_mol=jnp.sum(
                jnp.where(
                    executed[:, None],
                    evidence.candidate_transferred_to_surface_mol,
                    0.0,
                ),
                axis=0,
            ),
            total_amount_residual_mol=evidence.total_amount_residual_mol[worst],
            query_complete=jnp.all(evidence.query_complete),
            metric_exact=jnp.all(evidence.metric_exact),
        )

    def participant(
        self, /, *, subsystem_id: str = "bulk_surface", substeps: int = 1
    ) -> FixedStepCouplingParticipant:
        """Bind the real native method and concentration outputs to host windows."""
        ports = (
            CouplingPort(
                "bulk_concentration",
                "output",
                ArraySpace(self.bulk_volumes.shape),
                reference_scale=1.0,
            ),
            CouplingPort(
                "surface_concentration",
                "output",
                ArraySpace(self.surface_measures.shape),
                reference_scale=1.0,
            ),
        )

        def bind(
            window: object,
            inputs: tuple[object, ...],
            model_state: object,
            key: object,
            args: object,
        ) -> MethodWindowBinding:
            del window, inputs, key
            return MethodWindowBinding(args, model_state)

        def observe(state: tuple[Array, Array], args: object) -> tuple[Array, Array]:
            del args
            return state[0] / self.bulk_volumes, state[1] / self.surface_measures

        return FixedStepCouplingParticipant(
            self,
            bind,
            observe,
            subsystem_id=subsystem_id,
            substeps=positive_integer(substeps, "substeps"),
            output_ports=ports,
        )


__all__ = [
    "LangmuirAdsorptionFlux",
    "SurfaceDeposition",
    "SurfaceEpochRelocation",
    "SurfaceExchangeEvidence",
    "SurfaceExchangeLaw",
    "MeshfreeBulkSurfaceMethod",
]
