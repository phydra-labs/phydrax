#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum, compensated_sum_chunks
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._conservation_boundary import (
    AbstractConservationBoundary,
    ConstantStateBoundary,
    ExtrapolationBoundary,
    PrescribedStateBoundary,
)
from ._physical_boundaries import (
    NoSlipAdiabaticWallBoundary,
    NoSlipIsothermalWallBoundary,
    PrescribedHeatFluxWallBoundary,
    SlipWallBoundary,
    SupersonicInflowBoundary,
    SupersonicOutflowBoundary,
)
from ._positivity import EinfeldtHLLFluxPlan
from ._precision import FiniteVolumePrecisionPolicy
from ._reconstruction import PiecewiseConstantReconstruction
from ._riemann import (
    AbstractNumericalFluxPlan,
    HLLCFluxPlan,
    HLLFluxPlan,
    RusanovFluxPlan,
)
from ._triangle_fv import TriangleFiniteVolumeDiscretization
from ._triangle_polynomial import TriangleKExactReconstructionPlan
from ._triangle_reconstruction import TriangleMUSCLReconstructionPlan
from ._triangle_viscous import TriangleViscousFluxPlan


SourceFunction = Callable[[Array, Array, Array, Any], ArrayLike]


class TriangleFiniteVolumeBoundarySet(StrictModule, NonTrainableState):
    patch_names: tuple[str, ...] = eqx.field(static=True)
    boundaries: tuple[AbstractConservationBoundary, ...]
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_names: tuple[str, ...],
        boundaries: Mapping[str, AbstractConservationBoundary],
        /,
    ):
        names = tuple(patch_names)
        if set(boundaries) != set(names):
            raise ValueError(
                "Triangle FV boundaries must cover every mesh patch exactly."
            )
        values = tuple(boundaries[name] for name in names)
        allowed = (
            ConstantStateBoundary,
            ExtrapolationBoundary,
            PrescribedStateBoundary,
            SlipWallBoundary,
            SupersonicInflowBoundary,
            SupersonicOutflowBoundary,
            NoSlipAdiabaticWallBoundary,
            NoSlipIsothermalWallBoundary,
            PrescribedHeatFluxWallBoundary,
        )
        if any(not isinstance(value, allowed) for value in values):
            raise TypeError(
                "Triangle FV boundaries must be normal-oriented exterior-state policies."
            )
        self.patch_names = names
        self.boundaries = values
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "triangle-fv-boundary-set",
                "patches": [
                    {"name": name, "boundary": value.boundary_id}
                    for name, value in zip(names, values, strict=True)
                ],
            }
        )


class TriangleFiniteVolumeMethodPlan(StrictModule, NonTrainableState):
    reconstruction: (
        PiecewiseConstantReconstruction
        | TriangleMUSCLReconstructionPlan
        | TriangleKExactReconstructionPlan
    )
    interface_solver: RusanovFluxPlan | HLLFluxPlan | HLLCFluxPlan | EinfeldtHLLFluxPlan
    viscous: TriangleViscousFluxPlan | None
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: (
            PiecewiseConstantReconstruction
            | TriangleMUSCLReconstructionPlan
            | TriangleKExactReconstructionPlan
        ),
        interface_solver: (
            RusanovFluxPlan | HLLFluxPlan | HLLCFluxPlan | EinfeldtHLLFluxPlan
        ),
        /,
        *,
        viscous: TriangleViscousFluxPlan | None = None,
    ):
        if not isinstance(
            reconstruction,
            (
                PiecewiseConstantReconstruction,
                TriangleMUSCLReconstructionPlan,
                TriangleKExactReconstructionPlan,
            ),
        ):
            raise TypeError(
                "Triangle FV reconstruction must be piecewise constant, MUSCL, or k-exact."
            )
        if not isinstance(
            interface_solver,
            (
                RusanovFluxPlan,
                HLLFluxPlan,
                HLLCFluxPlan,
                EinfeldtHLLFluxPlan,
            ),
        ):
            raise TypeError("Triangle FV supports Rusanov, HLL, or HLLC flux.")
        if viscous is not None and not isinstance(viscous, TriangleViscousFluxPlan):
            raise TypeError("viscous must be TriangleViscousFluxPlan or None.")
        self.reconstruction = reconstruction
        self.interface_solver = interface_solver
        self.viscous = viscous
        self.method_id = canonical_fingerprint(
            {
                "kind": "triangle-fv-method",
                "reconstruction": reconstruction.plan_id,
                "flux": interface_solver.flux_id,
                "viscous": None if viscous is None else viscous.plan_id,
            }
        )


class TriangleFiniteVolumeDiagnostics(StrictModule):
    normal_flux: Array
    signal_speed: Array
    boundary_outward_flux: Array
    source_integral: Array
    conservation_defect: Array
    maximum_rate: Array
    precision_evidence: PrecisionEvidenceEnvelope


class PreparedTriangleFiniteVolumeDynamics(StrictModule):
    system: Any
    discretization: TriangleFiniteVolumeDiscretization
    method: TriangleFiniteVolumeMethodPlan
    boundaries: TriangleFiniteVolumeBoundarySet
    precision: FiniteVolumePrecisionPolicy
    source: SourceFunction | None = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
        discretization: TriangleFiniteVolumeDiscretization,
        method: TriangleFiniteVolumeMethodPlan,
        boundaries: TriangleFiniteVolumeBoundarySet,
        /,
        *,
        source: SourceFunction | None = None,
        source_id: str | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        if not isinstance(discretization, TriangleFiniteVolumeDiscretization):
            raise TypeError("discretization must be triangular finite-volume geometry.")
        if not isinstance(method, TriangleFiniteVolumeMethodPlan):
            raise TypeError("method must be TriangleFiniteVolumeMethodPlan.")
        if not isinstance(boundaries, TriangleFiniteVolumeBoundarySet):
            raise TypeError("boundaries must be TriangleFiniteVolumeBoundarySet.")
        if boundaries.patch_names != discretization.boundary_patch_names:
            raise ValueError("Boundary set patch names must match prepared mesh patches.")
        if (
            system.dimension != 2
            or system.component_count != discretization.component_count
        ):
            raise ValueError(
                "Triangle FV system dimension/components do not match geometry."
            )
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        source_identifier = None if source_id is None else str(source_id)
        if (source is None) != (source_identifier is None) or source_identifier == "":
            raise ValueError(
                "A source callable requires exactly one non-empty source_id."
            )
        precision_ = (
            FiniteVolumePrecisionPolicy(jnp.dtype(discretization.cell_volumes.dtype).name)
            if precision is None
            else precision
        )
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy.")
        self.system = system
        self.discretization = discretization
        self.method = method
        self.boundaries = boundaries
        self.precision = precision_
        self.source = source
        self.source_id = source_identifier
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-triangle-fv-dynamics",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "boundaries": boundaries.boundary_set_id,
                "precision": precision_.policy_id,
                "source": source_identifier,
            }
        )

    def _face_states(
        self, time: Array, state: Array, args: Any, /
    ) -> tuple[Array, Array, Array]:
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        if isinstance(
            self.method.reconstruction,
            (
                TriangleMUSCLReconstructionPlan,
                TriangleKExactReconstructionPlan,
            ),
        ):
            left, right = self.method.reconstruction.reconstruct(
                self.precision.reconstruction(state)
            )
        else:
            left = self.precision.reconstruction(state[owner])
            right = self.precision.reconstruction(state[safe_neighbour])
        unit_normal = (
            self.discretization.area_vectors / self.discretization.face_measures[:, None]
        )
        boundary = neighbour < 0
        for patch_id, policy in enumerate(self.boundaries.boundaries):
            patch_mask = boundary & (self.discretization.boundary_patch_ids == patch_id)
            exterior = policy.exterior_state(
                self.system,
                self.precision.decision(time),
                left,
                self.precision.reconstruction(self.discretization.face_centers),
                self.precision.reconstruction(unit_normal),
                0,
                args,
            )
            right = jnp.where(patch_mask[:, None], exterior, right)
        return left, right, unit_normal

    def make_fallback_dynamics(
        self, fallback_flux: AbstractNumericalFluxPlan, /
    ) -> "PreparedTriangleFiniteVolumeDynamics":
        method = TriangleFiniteVolumeMethodPlan(
            PiecewiseConstantReconstruction(), fallback_flux
        )
        return PreparedTriangleFiniteVolumeDynamics(
            self.system,
            self.discretization,
            method,
            self.boundaries,
            source=self.source,
            source_id=self.source_id,
            precision=self.precision,
        )

    def _quadrature_face_states(
        self, time: Array, state: Array, args: Any, /
    ) -> tuple[Array, Array, Array]:
        reconstruction = self.method.reconstruction
        if not isinstance(reconstruction, TriangleKExactReconstructionPlan):
            raise TypeError("Quadrature traces require k-exact reconstruction.")
        left, right = reconstruction.reconstruct_at(
            self.precision.reconstruction(state),
            self.precision.reconstruction(self.discretization.face_quadrature_points),
        )
        neighbour = self.discretization.neighbour_cells
        normal = (
            self.discretization.area_vectors / self.discretization.face_measures[:, None]
        )
        normal = jnp.broadcast_to(
            normal[:, None, :],
            self.discretization.face_quadrature_points.shape,
        )
        boundary = neighbour < 0
        for patch_id, policy in enumerate(self.boundaries.boundaries):
            patch_mask = boundary & (self.discretization.boundary_patch_ids == patch_id)
            exterior = policy.exterior_state(
                self.system,
                self.precision.decision(time),
                left,
                self.precision.reconstruction(self.discretization.face_quadrature_points),
                self.precision.reconstruction(normal),
                0,
                args,
            )
            right = jnp.where(patch_mask[:, None, None], exterior, right)
        return left, right, normal

    def face_fluxes(
        self, time: Array, state: Array, args: Any = None, /
    ) -> tuple[Array, Array]:
        value = jnp.asarray(state)
        if value.shape != self.discretization.state_shape:
            raise ValueError(
                f"Triangle FV state must have shape {self.discretization.state_shape}."
            )
        self.precision.validate_state(value)
        if isinstance(self.method.reconstruction, TriangleKExactReconstructionPlan):
            left, right, normal = self._quadrature_face_states(time, value, args)
            result = self.method.interface_solver.normal_face_flux(
                self.system,
                self.precision.flux(left),
                self.precision.flux(right),
                self.precision.flux(normal),
                args,
            )
            weights = self.precision.reduction(
                self.discretization.face_quadrature_weights
            )
            integrated = jnp.sum(
                weights[..., None] * self.precision.reduction(result.normal_flux),
                axis=1,
            )
            average_flux = integrated / self.precision.reduction(
                self.discretization.face_measures[:, None]
            )
            return self.precision.flux(average_flux), self.precision.decision(
                jnp.max(result.max_speed, axis=1)
            )
        left, right, normal = self._face_states(time, value, args)
        result = self.method.interface_solver.normal_face_flux(
            self.system,
            self.precision.flux(left),
            self.precision.flux(right),
            self.precision.flux(normal),
            args,
        )
        return self.precision.flux(result.normal_flux), self.precision.decision(
            result.max_speed
        )

    def residual_from_fluxes(self, normal_flux: Array, /) -> Array:
        integrated = self.precision.reduction(normal_flux) * self.precision.reduction(
            self.discretization.face_measures[:, None]
        )
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        residual = jnp.zeros(
            self.discretization.state_shape,
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        residual = residual.at[owner].add(-integrated)
        safe_neighbour = jnp.maximum(neighbour, 0)
        residual = residual.at[safe_neighbour].add(
            jnp.where((neighbour >= 0)[:, None], integrated, 0.0)
        )
        return self.precision.storage(
            residual / self.precision.reduction(self.discretization.cell_volumes[:, None])
        )

    def source_value(self, time: Array, state: Array, args: Any, /) -> Array:
        if self.source is None:
            return jnp.zeros_like(state)
        value = self.precision.flux(
            self.source(
                self.precision.decision(time),
                self.precision.flux(state),
                self.precision.flux(self.discretization.cell_centers),
                args,
            )
        )
        if value.shape != state.shape:
            raise ValueError("Triangle FV source must match the state shape.")
        return self.precision.storage(value)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        flux, _ = self.face_fluxes(time, state, args)
        residual = self.precision.reduction(self.residual_from_fluxes(flux))
        if self.method.viscous is not None:
            residual = residual + self.precision.reduction(
                self.method.viscous.residual(
                    self.system,
                    self.precision.decision(time),
                    self.precision.flux(state),
                    self.discretization,
                    self.boundaries,
                    args,
                )
            )
        return self.precision.storage(
            residual + self.precision.reduction(self.source_value(time, state, args))
        )

    def _cell_rate(self, speed: Array, /) -> Array:
        weighted = self.precision.reduction(speed) * self.precision.reduction(
            self.discretization.face_measures
        )
        rate = jnp.zeros(
            (self.discretization.cell_count,),
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        rate = rate.at[self.discretization.owner_cells].add(weighted)
        neighbour = self.discretization.neighbour_cells
        rate = rate.at[jnp.maximum(neighbour, 0)].add(
            jnp.where(neighbour >= 0, weighted, 0.0)
        )
        return self.precision.decision(
            rate / self.precision.reduction(self.discretization.cell_volumes)
        )

    def stable_step(
        self,
        state: Array,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> Array:
        self.precision.validate_state(state)
        _, speed = self.face_fluxes(jnp.asarray(0.0), state, args)
        maximum = jnp.max(self._cell_rate(speed))
        hyperbolic = jnp.where(maximum > 0.0, float(cfl) / maximum, jnp.inf)
        if self.method.viscous is None:
            return self.precision.decision(hyperbolic)
        viscous = self.method.viscous.stability_report(
            self.system,
            self.precision.flux(state),
            self.discretization,
            args,
            safety=cfl,
        ).selected_step
        return self.precision.decision(jnp.minimum(hyperbolic, viscous))

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        residual, jvp = jax.linearize(lambda value: self(time, value, args), state)
        _, vjp = jax.vjp(lambda value: self(time, value, args), state)
        return residual, jvp, vjp

    def residual_with_diagnostics(
        self, time: Array, state: Array, args: Any = None, /
    ) -> tuple[Array, TriangleFiniteVolumeDiagnostics]:
        flux, speed = self.face_fluxes(time, state, args)
        source = self.source_value(time, state, args)
        residual = self(time, state, args)
        boundary = self.discretization.neighbour_cells < 0
        integrated = self.precision.reduction(flux) * self.precision.reduction(
            self.discretization.face_measures[:, None]
        )
        boundary_terms = jnp.where(boundary[:, None], integrated, 0.0)
        source_terms = self.precision.reduction(
            self.discretization.cell_volumes[:, None] * source
        )
        change_terms = self.precision.reduction(
            self.discretization.cell_volumes[:, None] * residual
        )
        boundary_flux = compensated_sum(boundary_terms, axis=0)
        source_integral = compensated_sum(source_terms, axis=0)
        defect = compensated_sum_chunks(
            (change_terms, -source_terms, boundary_terms),
            output_ndim=1,
        )
        return self.precision.storage(residual), TriangleFiniteVolumeDiagnostics(
            normal_flux=flux,
            signal_speed=speed,
            boundary_outward_flux=boundary_flux,
            source_integral=source_integral,
            conservation_defect=defect,
            maximum_rate=jnp.max(self._cell_rate(speed)),
            precision_evidence=self.precision.evidence(),
        )


__all__ = [
    "PreparedTriangleFiniteVolumeDynamics",
    "TriangleFiniteVolumeBoundarySet",
    "TriangleFiniteVolumeDiagnostics",
    "TriangleFiniteVolumeMethodPlan",
]
