#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary import (
    FiniteVolumeBoundarySet,
    PrescribedNormalFluxBoundary,
)
from ._halo import FiniteVolumeHaloPlan, PreparedFiniteVolumeHaloPlan
from ._high_resolution import (
    CharacteristicReconstructionPlan,
    HighResolutionReconstructionPlan,
    NonuniformWENOReconstructionPlan,
)
from ._mapped import MappedFiniteVolumeDiscretization
from ._positivity import EinfeldtHLLFluxPlan
from ._precision import FiniteVolumePrecisionPolicy
from ._reconstruction import (
    AbstractFaceReconstructionPlan,
    PiecewiseConstantReconstruction,
    reconstruct_ghosted_axis,
)
from ._riemann import AbstractNumericalFluxPlan, HLLFluxPlan, RusanovFluxPlan
from ._structured import FiniteVolumeDiscretization
from ._viscous import ViscousFluxPlan
from ._wave import AbstractWavePropagationPlan, WaveFamilyLimiterPlan
from ._weno import WENOReconstructionPlan


DifferentiabilityPolicy: TypeAlias = Literal[
    "smooth_discrete",
    "branchwise",
    "smooth_surrogate",
    "unsupported",
]
SourceFunction = Callable[[Array, Array, Array, Any], ArrayLike]


class ConvexStateLimiterPlan(StrictModule, NonTrainableState):
    """Scale face states toward admissible cell averages by fixed bisection."""

    iterations: int = eqx.field(static=True)
    limiter_id: str = eqx.field(static=True)

    def __init__(self, iterations: int = 32, /):
        iterations_ = int(iterations)
        if iterations_ <= 0:
            raise ValueError("State-limiter iterations must be positive.")
        self.iterations = iterations_
        self.limiter_id = canonical_fingerprint(
            {"kind": "convex-state-limiter", "iterations": iterations_}
        )

    def limit(self, system: Any, average: Array, face: Array, /) -> Array:
        average_ = eqx.error_if(
            jnp.asarray(average),
            jnp.any(~system.admissible(average)),
            "Finite-volume cell average is not admissible.",
        )
        direction = jnp.asarray(face) - average_

        def body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            candidate = average_ + midpoint[..., None] * direction
            valid = system.admissible(candidate)
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

        lower, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            body,
            (jnp.zeros(average_.shape[:-1]), jnp.ones(average_.shape[:-1])),
        )
        return average_ + lower[..., None] * direction


class FiniteVolumeMethodPlan(StrictModule, NonTrainableState):
    """Validated composition of reconstruction and one interface method."""

    reconstruction: Any
    interface_solver: Any
    positivity: ConvexStateLimiterPlan | None
    wave_limiter: WaveFamilyLimiterPlan | None
    viscous: ViscousFluxPlan | None
    entropy_diagnostics: bool = eqx.field(static=True)
    differentiability: DifferentiabilityPolicy = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: AbstractFaceReconstructionPlan
        | HighResolutionReconstructionPlan
        | NonuniformWENOReconstructionPlan
        | CharacteristicReconstructionPlan
        | WENOReconstructionPlan,
        interface_solver: AbstractNumericalFluxPlan | AbstractWavePropagationPlan,
        /,
        *,
        positivity: ConvexStateLimiterPlan | None = None,
        wave_limiter: WaveFamilyLimiterPlan | None = None,
        viscous: ViscousFluxPlan | None = None,
        entropy_diagnostics: bool = False,
        differentiability: DifferentiabilityPolicy = "branchwise",
    ):
        if not isinstance(
            reconstruction,
            (
                AbstractFaceReconstructionPlan,
                HighResolutionReconstructionPlan,
                NonuniformWENOReconstructionPlan,
                CharacteristicReconstructionPlan,
                WENOReconstructionPlan,
            ),
        ):
            raise TypeError("Unsupported finite-volume reconstruction plan.")
        if not isinstance(
            interface_solver, (AbstractNumericalFluxPlan, AbstractWavePropagationPlan)
        ):
            raise TypeError("Unsupported finite-volume interface solver.")
        if positivity is not None and not isinstance(positivity, ConvexStateLimiterPlan):
            raise TypeError("positivity must be ConvexStateLimiterPlan or None.")
        if wave_limiter is not None and not isinstance(
            wave_limiter, WaveFamilyLimiterPlan
        ):
            raise TypeError("wave_limiter must be WaveFamilyLimiterPlan or None.")
        if wave_limiter is not None and not isinstance(
            interface_solver, AbstractWavePropagationPlan
        ):
            raise ValueError(
                "Wave limiting requires a wave-propagation interface solver."
            )
        if viscous is not None and not isinstance(viscous, ViscousFluxPlan):
            raise TypeError("viscous must be ViscousFluxPlan or None.")
        if differentiability not in (
            "smooth_discrete",
            "branchwise",
            "smooth_surrogate",
            "unsupported",
        ):
            raise ValueError("Unknown differentiability policy.")
        interface_id = (
            interface_solver.flux_id
            if isinstance(interface_solver, AbstractNumericalFluxPlan)
            else interface_solver.wave_plan_id
        )
        reconstruction_id = reconstruction.plan_id
        self.reconstruction = reconstruction
        self.interface_solver = interface_solver
        self.positivity = positivity
        self.wave_limiter = wave_limiter
        self.viscous = viscous
        self.entropy_diagnostics = bool(entropy_diagnostics)
        self.differentiability = differentiability
        self.method_id = canonical_fingerprint(
            {
                "kind": "finite-volume-method",
                "reconstruction": reconstruction_id,
                "interface": interface_id,
                "positivity": None if positivity is None else positivity.limiter_id,
                "wave_limiter": None if wave_limiter is None else wave_limiter.limiter_id,
                "viscous": None if viscous is None else viscous.plan_id,
                "entropy_diagnostics": bool(entropy_diagnostics),
                "differentiability": differentiability,
            }
        )


class FiniteVolumeResidualDiagnostics(StrictModule):
    """Observable flux, signal-speed, and global-balance evidence."""

    normal_fluxes: tuple[Array, ...]
    signal_speeds: tuple[Array, ...]
    boundary_outward_flux: Array
    source_integral: Array
    conservation_defect: Array
    maximum_rate: Array
    precision_evidence: PrecisionEvidenceEnvelope
    entropy_dissipation: Array


class PreparedFiniteVolumeDynamics(StrictModule):
    """Pure structured finite-volume semidiscretization."""

    system: Any
    discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization
    method: FiniteVolumeMethodPlan
    boundaries: FiniteVolumeBoundarySet
    halo: PreparedFiniteVolumeHaloPlan
    capacity: Array
    bathymetry: Array | None
    axis_reconstructions: tuple[Any, ...]
    precision: FiniteVolumePrecisionPolicy
    source: SourceFunction | None = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        method: FiniteVolumeMethodPlan,
        boundaries: FiniteVolumeBoundarySet,
        /,
        *,
        capacity: ArrayLike | None = None,
        bathymetry: ArrayLike | None = None,
        source: SourceFunction | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        if not isinstance(
            discretization,
            (FiniteVolumeDiscretization, MappedFiniteVolumeDiscretization),
        ):
            raise TypeError("discretization must be prepared finite-volume geometry.")
        if not isinstance(method, FiniteVolumeMethodPlan):
            raise TypeError("method must be a FiniteVolumeMethodPlan.")
        if not isinstance(boundaries, FiniteVolumeBoundarySet):
            raise TypeError("boundaries must be a FiniteVolumeBoundarySet.")
        if boundaries.axis_names != discretization.grid.axis_names:
            raise ValueError("Boundary axes must match the finite-volume grid.")
        if system.dimension != len(discretization.cell_shape):
            raise ValueError("Conservation-system dimension must match the grid.")
        if system.component_count != discretization.component_count:
            raise ValueError("System and finite-volume component counts must match.")
        for axis, structured_axis in enumerate(discretization.grid.structured_axes):
            pair = boundaries.pairs[axis]
            if structured_axis.periodic and pair is not None:
                raise ValueError("Periodic axes cannot declare physical boundary pairs.")
            if not structured_axis.periodic and pair is None:
                raise ValueError("Every bounded axis requires a boundary pair.")
        precision_ = (
            FiniteVolumePrecisionPolicy(jnp.dtype(discretization.cell_volumes.dtype).name)
            if precision is None
            else precision
        )
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be a FiniteVolumePrecisionPolicy.")
        capacity_ = precision_.reduction(
            jnp.ones(discretization.cell_shape) if capacity is None else capacity
        )
        if capacity_.shape != discretization.cell_shape:
            raise ValueError("capacity must match the finite-volume cell shape.")
        capacity_ = eqx.error_if(
            capacity_,
            jnp.any(~jnp.isfinite(capacity_) | (capacity_ <= 0.0)),
            "Finite-volume capacity must be finite and positive.",
        )
        bathymetry_ = (
            None if bathymetry is None else precision_.reconstruction(bathymetry)
        )
        if bathymetry_ is not None and bathymetry_.shape != discretization.cell_shape:
            raise ValueError("bathymetry must match the finite-volume cell shape.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        halo = FiniteVolumeHaloPlan(
            discretization, method.reconstruction, boundaries
        ).prepare()
        if isinstance(method.reconstruction, NonuniformWENOReconstructionPlan):
            prepared_reconstructions = []
            for axis, structured_axis in enumerate(discretization.grid.structured_axes):
                edges = np.asarray(method.reconstruction.cell_edges)
                if edges.size != discretization.cell_shape[axis] + 1:
                    raise ValueError(
                        "Nonuniform WENO edges must match each prepared axis."
                    )
                depth = halo.depth_by_axis[axis]
                widths = np.diff(edges)
                if structured_axis.periodic:
                    period = edges[-1] - edges[0]
                    lower_edges = edges[-(depth + 1) : -1] - period
                    upper_edges = edges[1 : depth + 1] + period
                else:
                    lower_edges = edges[0] - np.cumsum(widths[:depth])[::-1]
                    upper_edges = edges[-1] + np.cumsum(widths[-depth:][::-1])
                ghost_edges = np.concatenate((lower_edges, edges, upper_edges))
                prepared_reconstructions.append(
                    NonuniformWENOReconstructionPlan(
                        ghost_edges,
                        method=method.reconstruction.method,
                        epsilon=method.reconstruction.epsilon,
                        power=method.reconstruction.power,
                        cutoff=method.reconstruction.cutoff,
                    )
                )
            axis_reconstructions = tuple(prepared_reconstructions)
        else:
            axis_reconstructions = (method.reconstruction,) * len(
                discretization.cell_shape
            )
        self.system = system
        self.discretization = discretization
        self.method = method
        self.boundaries = boundaries
        self.halo = halo
        self.capacity = capacity_
        self.bathymetry = bathymetry_
        self.axis_reconstructions = axis_reconstructions
        self.precision = precision_
        self.source = source
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-volume-dynamics",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "boundaries": boundaries.boundary_set_id,
                "halo": halo.prepared_id,
                "axis_reconstructions": [
                    reconstruction.plan_id for reconstruction in axis_reconstructions
                ],
                "capacity": array_tree_fingerprint(np.asarray(capacity_)),
                "bathymetry": None
                if bathymetry_ is None
                else array_tree_fingerprint(np.asarray(bathymetry_)),
                "precision": precision_.policy_id,
            }
        )

    @property
    def effective_volumes(self) -> Array:
        return self.precision.reduction(
            self.discretization.cell_volumes
        ) * self.precision.reduction(self.capacity)

    def _boundary_states(
        self,
        time: Array,
        state: Array,
        axis: int,
        args: Any,
        /,
    ) -> tuple[Array | None, Array | None]:
        return self.halo.boundary_states(
            self.system,
            self.precision.decision(time),
            self.precision.reconstruction(state),
            axis,
            args,
        )

    def _reconstruct(
        self,
        time: Array,
        state: Array,
        axis: int,
        args: Any,
        /,
    ) -> tuple[Array, Array]:
        time = self.precision.decision(time)
        state = self.precision.reconstruction(state)
        periodic = self.discretization.grid.structured_axes[axis].periodic
        ghosted = self.halo.materialize_axis(self.system, time, state, axis, args)
        left, right = reconstruct_ghosted_axis(
            self.axis_reconstructions[axis],
            ghosted.values,
            axis,
            interior_cell_count=self.discretization.cell_shape[axis],
            ghost_depth=ghosted.depth,
            periodic=periodic,
            axis_coordinates=self.precision.reconstruction(ghosted.axis_coordinates),
        )
        if self.method.positivity is not None:
            averages = reconstruct_ghosted_axis(
                PiecewiseConstantReconstruction(),
                ghosted.values,
                axis,
                interior_cell_count=self.discretization.cell_shape[axis],
                ghost_depth=ghosted.depth,
                periodic=periodic,
                axis_coordinates=self.precision.reconstruction(ghosted.axis_coordinates),
            )
            left = self.method.positivity.limit(self.system, averages[0], left)
            right = self.method.positivity.limit(self.system, averages[1], right)
        return self.precision.reconstruction(left), self.precision.reconstruction(right)

    def _override_boundary_flux(
        self,
        time: Array,
        state: Array,
        axis: int,
        flux: Array,
        args: Any,
        /,
    ) -> Array:
        flux = self.precision.flux(flux)
        pair = self.boundaries.pairs[axis]
        if pair is None:
            return flux
        output = flux
        if isinstance(pair.lower, PrescribedNormalFluxBoundary):
            interior = jnp.take(state, 0, axis=axis)
            coordinates = jnp.take(self.discretization.face_centers[axis], 0, axis=axis)
            outward = pair.lower.normal_flux(
                time,
                interior,
                coordinates,
                self.discretization.outward_normal(axis, "lower"),
                args,
            )
            index: list[slice | int] = [slice(None)] * output.ndim
            index[axis] = 0
            output = output.at[tuple(index)].set(-outward)
        if isinstance(pair.upper, PrescribedNormalFluxBoundary):
            interior = jnp.take(state, state.shape[axis] - 1, axis=axis)
            coordinates = jnp.take(
                self.discretization.face_centers[axis],
                self.discretization.face_layouts[axis].shape[axis] - 1,
                axis=axis,
            )
            outward = pair.upper.normal_flux(
                time,
                interior,
                coordinates,
                self.discretization.outward_normal(axis, "upper"),
                args,
            )
            index: list[slice | int] = [slice(None)] * output.ndim
            index[axis] = output.shape[axis] - 1
            output = output.at[tuple(index)].set(outward)
        return output

    def face_fluxes(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        if not isinstance(self.method.interface_solver, AbstractNumericalFluxPlan):
            raise TypeError(
                "Wave-propagation methods do not expose a unique normal flux."
            )
        value = jnp.asarray(state)
        if value.shape != self.discretization.state_shape:
            raise ValueError(
                f"Finite-volume state must have shape {self.discretization.state_shape}."
            )
        self.precision.validate_state(value)
        fluxes = []
        speeds = []
        for axis in range(len(self.discretization.cell_shape)):
            left, right = self._reconstruct(time, value, axis, args)
            if isinstance(self.discretization, MappedFiniteVolumeDiscretization):
                solver = self.method.interface_solver
                if not isinstance(
                    solver, (RusanovFluxPlan, HLLFluxPlan, EinfeldtHLLFluxPlan)
                ):
                    raise ValueError(
                        "Mapped finite volumes currently require Rusanov, HLL, "
                        "or Einfeldt HLL flux."
                    )
                normal = (
                    self.discretization.face_area_vectors[axis]
                    / self.discretization.face_measures[axis][..., None]
                )
                result = solver.normal_face_flux(
                    self.system,
                    self.precision.flux(left),
                    self.precision.flux(right),
                    self.precision.flux(normal),
                    args,
                )
            else:
                result = self.method.interface_solver.face_flux(
                    self.system,
                    self.precision.flux(left),
                    self.precision.flux(right),
                    axis,
                    args,
                )
            fluxes.append(
                self.precision.flux(
                    self._override_boundary_flux(
                        time,
                        value,
                        axis,
                        result.normal_flux,
                        args,
                    )
                )
            )
            speeds.append(self.precision.decision(result.max_speed))
        return tuple(fluxes), tuple(speeds)

    def _flux_residual(
        self,
        fluxes: tuple[Array, ...],
        /,
    ) -> Array:
        residual = jnp.zeros(
            self.discretization.state_shape,
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        for axis, normal_flux in enumerate(fluxes):
            measure = self.precision.reduction(
                self.discretization.face_measures[axis][..., None]
            )
            integrated = self.precision.reduction(normal_flux) * measure
            if self.discretization.grid.structured_axes[axis].periodic:
                difference = jnp.roll(integrated, -1, axis=axis) - integrated
            else:
                lower_index = [slice(None)] * integrated.ndim
                upper_index = [slice(None)] * integrated.ndim
                lower_index[axis] = slice(0, integrated.shape[axis] - 1)
                upper_index[axis] = slice(1, integrated.shape[axis])
                difference = (
                    integrated[tuple(upper_index)] - integrated[tuple(lower_index)]
                )
            residual = self.precision.reduction(
                residual
                - self.precision.reduction(difference) / self.effective_volumes[..., None]
            )
        return self.precision.storage(residual)

    def _wave_residual(
        self,
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> tuple[Array, tuple[Array, ...]]:
        solver = self.method.interface_solver
        if not isinstance(solver, AbstractWavePropagationPlan):
            raise TypeError("Wave residual requires a wave-propagation plan.")
        residual = jnp.zeros(
            self.discretization.state_shape,
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        speeds = []
        for axis in range(len(self.discretization.cell_shape)):
            left, right = self._reconstruct(time, state, axis, args)
            auxiliary_left = None
            auxiliary_right = None
            if self.bathymetry is not None:
                lower, upper = self._boundary_states(time, state, axis, args)
                del lower, upper
                auxiliary_left, auxiliary_right = (
                    PiecewiseConstantReconstruction().reconstruct_axis(
                        self.bathymetry[..., None],
                        axis,
                        periodic=self.discretization.grid.structured_axes[axis].periodic,
                        lower_exterior=jnp.take(self.bathymetry, 0, axis=axis)[..., None],
                        upper_exterior=jnp.take(
                            self.bathymetry, self.bathymetry.shape[axis] - 1, axis=axis
                        )[..., None],
                    )
                )
                auxiliary_left = auxiliary_left[..., 0]
                auxiliary_right = auxiliary_right[..., 0]
            decomposition = solver.decompose(
                self.system,
                self.precision.flux(left),
                self.precision.flux(right),
                axis,
                args,
                auxiliary_left=auxiliary_left,
                auxiliary_right=auxiliary_right,
            )
            if self.method.wave_limiter is not None:
                decomposition = self.method.wave_limiter.limit(decomposition, axis)
            speeds.append(
                self.precision.decision(jnp.max(jnp.abs(decomposition.speeds), axis=-1))
            )
            left_fluctuation = self.precision.reduction(decomposition.left_fluctuation)
            right_fluctuation = self.precision.reduction(decomposition.right_fluctuation)
            if self.discretization.grid.structured_axes[axis].periodic:
                cell_fluctuation = right_fluctuation + jnp.roll(
                    left_fluctuation, -1, axis=axis
                )
            else:
                lower_index = [slice(None)] * left_fluctuation.ndim
                upper_index = [slice(None)] * left_fluctuation.ndim
                lower_index[axis] = slice(0, left_fluctuation.shape[axis] - 1)
                upper_index[axis] = slice(1, left_fluctuation.shape[axis])
                cell_fluctuation = (
                    right_fluctuation[tuple(lower_index)]
                    + left_fluctuation[tuple(upper_index)]
                )
            tangential_measure = self.precision.reduction(
                self.discretization.face_measures[axis]
            )
            if self.discretization.grid.structured_axes[axis].periodic:
                cell_measure = tangential_measure
            else:
                take = [slice(None)] * tangential_measure.ndim
                take[axis] = slice(0, tangential_measure.shape[axis] - 1)
                cell_measure = tangential_measure[tuple(take)]
            residual = self.precision.reduction(
                residual
                - cell_fluctuation
                * cell_measure[..., None]
                / self.effective_volumes[..., None]
            )
        return self.precision.storage(residual), tuple(speeds)

    def _source_value(
        self,
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> Array:
        if self.source is None:
            return jnp.zeros_like(state)
        source = self.precision.flux(
            self.source(
                self.precision.decision(time),
                self.precision.flux(state),
                self.precision.flux(self.discretization.cell_centers),
                args,
            )
        )
        if source.shape != state.shape:
            raise ValueError("Finite-volume source must match the state shape.")
        return self.precision.storage(source)

    def axis_residual(
        self,
        time: Array,
        state: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        """Return one directional conservative contribution without source terms."""
        axis_ = int(axis)
        if not 0 <= axis_ < len(self.discretization.cell_shape):
            raise ValueError("Finite-volume residual axis is out of range.")
        if not isinstance(self.method.interface_solver, AbstractNumericalFluxPlan):
            raise TypeError("Directional splitting currently requires numerical fluxes.")
        fluxes, _ = self.face_fluxes(time, state, args)
        integrated = self.precision.reduction(fluxes[axis_]) * (
            self.precision.reduction(self.discretization.face_measures[axis_][..., None])
        )
        if self.discretization.grid.structured_axes[axis_].periodic:
            difference = jnp.roll(integrated, -1, axis=axis_) - integrated
        else:
            lower = [slice(None)] * integrated.ndim
            upper = [slice(None)] * integrated.ndim
            lower[axis_] = slice(0, integrated.shape[axis_] - 1)
            upper[axis_] = slice(1, integrated.shape[axis_])
            difference = integrated[tuple(upper)] - integrated[tuple(lower)]
        return self.precision.storage(
            -self.precision.reduction(difference) / self.effective_volumes[..., None]
        )

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.discretization.state_shape:
            raise ValueError(
                f"Finite-volume state must have shape {self.discretization.state_shape}."
            )
        self.precision.validate_state(value)
        if isinstance(self.method.interface_solver, AbstractNumericalFluxPlan):
            fluxes, _ = self.face_fluxes(time, value, args)
            residual = self._flux_residual(fluxes)
        else:
            residual, _ = self._wave_residual(time, value, args)
        residual = self.precision.reduction(residual) + self.precision.reduction(
            self._source_value(time, value, args)
        )
        if self.method.viscous is not None:
            residual = residual + self.precision.reduction(
                self.method.viscous.residual(
                    self.system,
                    time,
                    self.precision.flux(value),
                    self.discretization,
                    self.halo,
                    args,
                )
            )
        return self.precision.storage(residual)

    def _rate_from_speeds(self, speeds: tuple[Array, ...], /) -> Array:
        rate = jnp.zeros(
            self.discretization.cell_shape,
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        for axis, face_speed in enumerate(speeds):
            weighted = self.precision.reduction(face_speed) * self.precision.reduction(
                self.discretization.face_measures[axis]
            )
            if self.discretization.grid.structured_axes[axis].periodic:
                contribution = weighted + jnp.roll(weighted, -1, axis=axis)
            else:
                lower_index = [slice(None)] * weighted.ndim
                upper_index = [slice(None)] * weighted.ndim
                lower_index[axis] = slice(0, weighted.shape[axis] - 1)
                upper_index[axis] = slice(1, weighted.shape[axis])
                contribution = weighted[tuple(lower_index)] + weighted[tuple(upper_index)]
            rate = self.precision.reduction(rate + contribution / self.effective_volumes)
        return self.precision.decision(rate)

    def stable_step(
        self,
        state: Array,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> Array:
        cfl_ = float(cfl)
        if not np.isfinite(cfl_) or cfl_ <= 0.0:
            raise ValueError("cfl must be finite and positive.")
        value = jnp.asarray(state)
        self.precision.validate_state(value)
        if isinstance(self.method.interface_solver, AbstractNumericalFluxPlan):
            _, speeds = self.face_fluxes(jnp.asarray(0.0), value, args)
        else:
            _, speeds = self._wave_residual(jnp.asarray(0.0), value, args)
        maximum = self.precision.decision(jnp.max(self._rate_from_speeds(speeds)))
        hyperbolic_step = jnp.where(
            maximum > 0.0,
            self.precision.decision(cfl_) / maximum,
            jnp.inf,
        )
        if self.method.viscous is None:
            return self.precision.decision(hyperbolic_step)
        viscous_step = self.method.viscous.stable_step(
            self.system,
            self.precision.flux(value),
            self.discretization,
            args,
            safety=cfl_,
        )
        return self.precision.decision(jnp.minimum(hyperbolic_step, viscous_step))

    def residual_with_diagnostics(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, FiniteVolumeResidualDiagnostics]:
        value = jnp.asarray(state)
        self.precision.validate_state(value)
        source = self._source_value(time, value, args)
        if isinstance(self.method.interface_solver, AbstractNumericalFluxPlan):
            fluxes, speeds = self.face_fluxes(time, value, args)
            residual = self.precision.reduction(
                self._flux_residual(fluxes)
            ) + self.precision.reduction(source)
            if self.method.viscous is not None:
                residual = residual + self.precision.reduction(
                    self.method.viscous.residual(
                        self.system,
                        time,
                        self.precision.flux(value),
                        self.discretization,
                        self.halo,
                        args,
                    )
                )
            boundary_flux = jnp.zeros(
                (self.discretization.component_count,),
                dtype=jnp.dtype(self.precision.reduction_dtype),
            )
            for axis, flux in enumerate(fluxes):
                if self.discretization.grid.structured_axes[axis].periodic:
                    continue
                lower = self.precision.reduction(jnp.take(flux, 0, axis=axis))
                upper = self.precision.reduction(
                    jnp.take(flux, flux.shape[axis] - 1, axis=axis)
                )
                lower_measure = self.precision.reduction(
                    jnp.take(self.discretization.face_measures[axis], 0, axis=axis)
                )
                upper_measure = self.precision.reduction(
                    jnp.take(
                        self.discretization.face_measures[axis],
                        self.discretization.face_measures[axis].shape[axis] - 1,
                        axis=axis,
                    )
                )
                reduction_axes = tuple(range(lower.ndim - 1))
                boundary_flux = self.precision.reduction(
                    boundary_flux
                    + jnp.sum(
                        upper * upper_measure[..., None]
                        - lower * lower_measure[..., None],
                        axis=reduction_axes,
                    )
                )
        else:
            residual, speeds = self._wave_residual(time, value, args)
            residual = self.precision.reduction(residual) + self.precision.reduction(
                source
            )
            fluxes = ()
            boundary_flux = jnp.full(
                (self.discretization.component_count,),
                jnp.nan,
                dtype=jnp.dtype(self.precision.reduction_dtype),
            )
        spatial_axes = tuple(range(len(self.discretization.cell_shape)))
        source_integral = jnp.sum(
            self.precision.reduction(self.effective_volumes[..., None] * source),
            axis=spatial_axes,
        )
        change_integral = jnp.sum(
            self.precision.reduction(self.effective_volumes[..., None] * residual),
            axis=spatial_axes,
        )
        defect = change_integral - source_integral + boundary_flux
        rate = self._rate_from_speeds(speeds)
        entropy = jnp.asarray(
            0.0,
            dtype=jnp.dtype(self.precision.reduction_dtype),
        )
        if self.method.entropy_diagnostics and fluxes:
            entropy = jnp.sum(
                self.precision.reduction(
                    self.system.entropy_variables(self.precision.flux(value))
                    * self.precision.reduction(residual)
                )
            )
        diagnostics = FiniteVolumeResidualDiagnostics(
            normal_fluxes=fluxes,
            signal_speeds=speeds,
            boundary_outward_flux=boundary_flux,
            source_integral=source_integral,
            conservation_defect=defect,
            maximum_rate=self.precision.decision(jnp.max(rate)),
            entropy_dissipation=self.precision.decision(entropy),
            precision_evidence=self.precision.evidence(),
        )
        return self.precision.storage(residual), diagnostics

    def linearize(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ):
        residual, jvp = jax.linearize(lambda value: self(time, value, args), state)
        _, vjp = jax.vjp(lambda value: self(time, value, args), state)
        return residual, jvp, vjp


__all__ = [
    "ConvexStateLimiterPlan",
    "DifferentiabilityPolicy",
    "FiniteVolumeMethodPlan",
    "FiniteVolumeResidualDiagnostics",
    "PreparedFiniteVolumeDynamics",
]
