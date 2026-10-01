#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative point-query exchange, native Langmuir transport, host refresh."""

from __future__ import annotations

from collections.abc import Mapping
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation._stencil import GatherStencil
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_finite_float, positive_integer
from ...discretization import PreparedFieldQuery
from ...interfacial_transport._core import AdsorptionKinetics
from ...interfacial_transport._coupled import CoupledBulkSurfaceTransport
from ...linalg import ArraySpace
from ...typing import Dim, Float
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


if TYPE_CHECKING:
    from ...discretization.meshfree._exterior import PreparedMeshfreeExteriorCalculus


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

    def __init__(self, kinetics: AdsorptionKinetics, /) -> None:
        if not isinstance(kinetics, AdsorptionKinetics):
            raise TypeError("kinetics must be AdsorptionKinetics.")
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
    query_id: str = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    constant_reproduction_error: float = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    refresh_time: float = eqx.field(static=True)
    displacement: float = eqx.field(static=True)
    lag: float = eqx.field(static=True)
    geometry_differentiated_within_window: bool = eqx.field(static=True, default=False)


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
        return InterfaceDefectReport(
            self.law_id,
            ("amount_balance", "admissibility"),
            (True, True),
            jnp.stack((balance, jnp.where(physical, 0.0, 1.0))),
            jnp.stack((scale, jnp.asarray(1.0))),
        )


@final
class SurfaceExchangeLaw(AbstractCouplingLaw, NonTrainableState):
    """Non-facet exchange between declared full bulk and nodal surface fields.

    Queries are frozen inside numerical windows. ``refresh_at_window`` is a
    host preparation boundary and returns a new law with explicit displacement
    and geometry-lag evidence; it does not pretend to differentiate relocation.
    Positive pairings and row-constant reproduction are mandatory.
    """

    __strict_contract__ = True
    law_id: str = eqx.field(static=True)
    bulk: ContributionEndpoint
    surface: ContributionEndpoint
    query: PreparedFieldQuery
    measures: Float[_SurfacePointDim]
    normals: Float[_SurfacePointDim, _SurfaceCoordinateDim]
    flux: AbstractInterfaceFlux
    tolerance: float = eqx.field(static=True)
    evidence: SurfaceExchangeEvidence

    def __init__(
        self,
        bulk: ContributionEndpoint,
        surface: ContributionEndpoint,
        query: PreparedFieldQuery,
        surface_measures: ArrayLike,
        normals: ArrayLike,
        flux: AbstractInterfaceFlux,
        /,
        *,
        law_id: str = "surface_exchange",
        constant_tolerance: float = 1e-10,
        geometry_epoch: int = 0,
        refresh_time: float = 0.0,
        displacement: float = 0.0,
        lag: float = 0.0,
    ) -> None:
        if not isinstance(bulk, ContributionEndpoint) or not isinstance(
            surface, ContributionEndpoint
        ):
            raise TypeError("Exchange endpoints must be ContributionEndpoint values.")
        if bulk.space != "full" or surface.space != "full" or bulk.owner == surface.owner:
            raise ValueError(
                "Exchange endpoints must name full fields of distinct components."
            )
        if not isinstance(flux, AbstractInterfaceFlux):
            raise TypeError("flux must be a native AbstractInterfaceFlux.")
        tolerance = positive_finite_float(constant_tolerance, "constant_tolerance")
        error = _admit_query(query, tolerance)
        measures = np.asarray(surface_measures, dtype=np.float64)
        normal = np.asarray(normals, dtype=np.float64)
        if measures.shape != query.output_shape or not np.all(
            np.isfinite(measures) & (measures > 0)
        ):
            raise ValueError("Exchange needs positive measures at every surface point.")
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
        self.measures, self.normals, self.flux = (
            jnp.asarray(measures),
            jnp.asarray(normal),
            flux,
        )
        self.tolerance = tolerance
        self.evidence = SurfaceExchangeEvidence(
            query.query_id,
            query.complete,
            error,
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

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> PreparedLaw:
        del interface_owners
        bulk, surface = components[self.bulk.owner], components[self.surface.owner]
        if not isinstance(bulk, AbstractReconstructionComponent):
            raise TypeError("Bulk exchange requires the owning field reconstruction.")
        self.query.require_reconstruction(
            bulk.prepare_field_reconstruction(self.bulk.block)
        )
        if bulk.field(self.bulk.block).full_space.shape != self.query.coefficient_shape:
            raise ValueError("Query coefficients do not match the declared bulk field.")
        if not isinstance(surface, MeshfreeComponent):
            raise TypeError(
                "Surface exchange requires a native nodal MeshfreeComponent endpoint."
            )
        if not np.array_equal(
            np.asarray(surface.owner.points), np.asarray(self.query.points)
        ):
            raise ValueError(
                "Surface endpoint must own the actual query point enumeration."
            )
        if not np.array_equal(
            np.asarray(surface.mass_diagonal), np.asarray(self.measures)
        ):
            raise ValueError(
                "Exchange measures must be the surface endpoint's native capacity diagonal."
            )
        if surface.field(self.surface.block).full_space.shape != self.measures.shape:
            raise ValueError(
                "Surface field coordinates do not match the exchange quadrature."
            )
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
                self.law_id, self.bulk, self.surface, residual
            ),
            evidence=self.evidence,
        )

    def refresh_at_window(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        points: ArrayLike,
        measures: ArrayLike,
        normals: ArrayLike,
        /,
        *,
        window_start: float,
        geometry_time: float,
        geometry_epoch: int,
        maximum_displacement: float,
        maximum_lag: float,
    ) -> SurfaceExchangeLaw:
        """Relocate at a host window boundary, refusing stale or excessive motion."""
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
        values = np.asarray(points, dtype=np.float64)
        if values.shape != self.query.points.shape:
            raise ValueError("Moving exchange refresh preserves surface topology.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Moving exchange points must be finite.")
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
        bulk = components[self.bulk.owner]
        if not isinstance(bulk, AbstractReconstructionComponent):
            raise TypeError("Bulk exchange requires reconstruction capability.")
        query = bulk.prepare_field_reconstruction(self.bulk.block).prepare_query(
            values, coverage="complete"
        )
        return SurfaceExchangeLaw(
            self.bulk,
            self.surface,
            query,
            measures,
            normals,
            self.flux,
            law_id=self.law_id,
            constant_tolerance=self.tolerance,
            geometry_epoch=geometry_epoch,
            refresh_time=window_start,
            displacement=displacement,
            lag=lag,
        )


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
        from ...discretization.meshfree._exterior import PreparedMeshfreeExteriorCalculus

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
        route = query.route
        if not isinstance(route, GatherStencil):
            raise ValueError(
                "Fixed-topology amount transport requires an owning bounded GatherStencil query."
            )
        weights = np.asarray(route.weights)
        valid = np.asarray(route.valid) & (weights != 0.0)
        if np.any(weights[valid] < 0):
            raise ValueError(
                "Signed query weights cannot define a positive amount partition."
            )
        surface_indices = np.broadcast_to(
            np.arange(query.admitted_count)[:, None], weights.shape
        )[valid]
        bulk_indices = np.asarray(route.indices)[valid]
        values = weights[valid]
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
        self.method_id = canonical_fingerprint(
            {
                "kind": "meshfree-bulk-surface-method",
                "transport": self.transport.transport_id,
                "query": query.query_id,
            }
        )

    @property
    def evidence_template(self) -> tuple[Array, Array, Array, Array]:
        return (
            jnp.zeros((), dtype=self.surface_measures.dtype),
            jnp.zeros((), dtype=self.surface_measures.dtype),
            jnp.zeros_like(self.surface_measures),
            jnp.zeros((), dtype=jnp.int32),
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
            evidence=(
                result.total_amount_residual_mol,
                result.maximum_coverage,
                result.transferred_to_surface_mol,
                result.status,
            ),
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
    "SurfaceExchangeEvidence",
    "SurfaceExchangeLaw",
    "MeshfreeBulkSurfaceMethod",
]
