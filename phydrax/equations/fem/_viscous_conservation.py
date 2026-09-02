#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._conservation_boundary import PrescribedNormalFluxBoundary
from ...discretization.fem._boundary import tensor_local_face
from ...discretization.finite_volume._physical_boundaries import (
    PrescribedHeatFluxWallBoundary,
)
from .._hyperbolic_systems import (
    AbstractEntropyDiffusionSystem,
)


class EntropyDiffusionEvidence(StrictModule, NonTrainableState):
    production: Array
    minimum_production: Array
    nonnegative: Array
    evidence_id: str = eqx.field(static=True)


def entropy_diffusion_evidence(
    system: AbstractEntropyDiffusionSystem,
    state: ArrayLike,
    conserved_gradient: ArrayLike,
    args: Any = None,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> EntropyDiffusionEvidence:
    if not isinstance(system, AbstractEntropyDiffusionSystem):
        raise TypeError("system must implement AbstractEntropyDiffusionSystem.")
    production = system.entropy_viscous_production(
        jnp.asarray(state), jnp.asarray(conserved_gradient), args
    )
    minimum = jnp.min(production)
    evidence_id = canonical_fingerprint(
        {
            "kind": "entropy-diffusion-evidence",
            "system": system.system_id,
            "tolerance": float(tolerance),
        }
    )
    return EntropyDiffusionEvidence(
        production,
        minimum,
        minimum >= -float(tolerance),
        evidence_id,
    )


class ViscousBoundaryClosure(StrictModule, NonTrainableState):
    boundary_id: str = eqx.field(static=True)
    gradient_provider: Any = eqx.field(static=True)
    normal_flux_provider: Any = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary_id: str,
        /,
        *,
        gradient_provider=None,
        normal_flux_provider=None,
    ):
        identifier = str(boundary_id)
        gradient = lambda time, state, gradient, points, normal, args: (
            gradient if gradient_provider is None else gradient_provider
        )
        normal_flux = (
            lambda time, plus, minus, plus_gradient, minus_gradient, normal, default, args: (
                default if normal_flux_provider is None else normal_flux_provider
            )
        )
        if not identifier or not callable(gradient) or not callable(normal_flux):
            raise ValueError("Viscous boundary closure definition is invalid.")
        self.boundary_id = identifier
        self.gradient_provider = gradient
        self.normal_flux_provider = normal_flux
        self.closure_id = canonical_fingerprint(
            {
                "kind": "viscous-boundary-closure",
                "boundary": identifier,
            }
        )

    def gradient_trace(
        self,
        time: Array,
        state: Array,
        gradient: Array,
        points: Array,
        normal: Array,
        args: Any,
        /,
    ) -> Array:
        value = jnp.asarray(
            self.gradient_provider(time, state, gradient, points, normal, args)
        )
        if value.shape != gradient.shape:
            raise ValueError("Viscous boundary gradient trace changed shape.")
        return value

    def normal_flux(
        self,
        time: Array,
        plus: Array,
        minus: Array,
        plus_gradient: Array,
        minus_gradient: Array,
        normal: Array,
        default_flux: Array,
        args: Any,
        /,
    ) -> Array:
        value = jnp.asarray(
            self.normal_flux_provider(
                time,
                plus,
                minus,
                plus_gradient,
                minus_gradient,
                normal,
                default_flux,
                args,
            )
        )
        if value.shape != default_flux.shape:
            raise ValueError("Viscous boundary normal flux changed shape.")
        return value


class ViscousDGPlan(StrictModule, NonTrainableState):
    formulation: str = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    penalty: float = eqx.field(static=True)
    boundary_closures: tuple[ViscousBoundaryClosure, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        formulation: str = "entropy_br1",
        beta: float = 0.0,
        penalty: float = 1.0,
        boundary_closures: Sequence[ViscousBoundaryClosure] = (),
    ):
        formulation_ = str(formulation)
        beta_ = float(beta)
        penalty_ = float(penalty)
        if formulation_ not in ("entropy_br1", "ldg"):
            raise ValueError("Viscous DG formulation must be entropy_br1 or ldg.")
        if not math.isfinite(beta_) or abs(beta_) > 0.5:
            raise ValueError("Viscous DG beta must be finite and lie in [-0.5, 0.5].")
        if formulation_ == "entropy_br1" and beta_ != 0.0:
            raise ValueError("Entropy BR1 requires beta=0.")
        if not math.isfinite(penalty_) or penalty_ <= 0.0:
            raise ValueError("Viscous DG penalty must be finite and positive.")
        closures = tuple(boundary_closures)
        if any(
            not isinstance(value, ViscousBoundaryClosure) for value in closures
        ) or len({value.boundary_id for value in closures}) != len(closures):
            raise ValueError("Viscous boundary closures must be typed and unique.")
        self.formulation = formulation_
        self.beta = beta_
        self.penalty = penalty_
        self.boundary_closures = closures
        self.plan_id = canonical_fingerprint(
            {
                "kind": "viscous-dg-plan",
                "formulation": formulation_,
                "beta": beta_,
                "penalty": penalty_,
                "entropy_evidence": (
                    "entropy-diffusion-capability"
                    if formulation_ == "entropy_br1"
                    else "uncertified"
                ),
                "boundary_closures": tuple(value.closure_id for value in closures),
            }
        )

    def boundary_closure(self, boundary_id: str, /) -> ViscousBoundaryClosure:
        identifier = str(boundary_id)
        for closure in self.boundary_closures:
            if closure.boundary_id == identifier:
                return closure
        raise ValueError(f"Missing viscous closure for physical boundary {identifier!r}.")


class ViscousDGStabilityEvidence(StrictModule, NonTrainableState):
    step: Array
    maximum_diffusive_rate: Array
    positive: Array
    plan_id: str = eqx.field(static=True)


class PreparedViscousDGOperator(StrictModule):
    plan: ViscousDGPlan
    dynamics: Any
    operator_id: str = eqx.field(static=True)

    def __init__(self, plan: ViscousDGPlan, dynamics: Any, /):
        if not isinstance(plan, ViscousDGPlan):
            raise TypeError("plan must be ViscousDGPlan.")
        if not isinstance(dynamics.system, AbstractEntropyDiffusionSystem):
            raise TypeError("Viscous DG requires AbstractEntropyDiffusionSystem.")
        if (
            dynamics.discretization.mesh.topological_dimension
            != dynamics.system.dimension
        ):
            raise ValueError("Viscous DG mesh and system dimensions must match.")
        if dynamics.sbp.order < 1:
            raise ValueError("Viscous DG requires polynomial degree >= 1.")
        if dynamics.boundaries is not None and any(
            isinstance(patch.boundary, PrescribedNormalFluxBoundary)
            for patch in dynamics.boundaries.patches
        ):
            raise ValueError(
                "Prescribed normal-flux boundaries lack a viscous state closure."
            )
        if dynamics.boundaries is not None:
            for patch in dynamics.boundaries.patches:
                plan.boundary_closure(patch.boundary.boundary_id)
        self.plan = plan
        self.dynamics = dynamics
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-viscous-dg-operator",
                "plan": plan.plan_id,
                "dynamics": dynamics.dynamics_id,
            }
        )

    @property
    def _dimension(self) -> int:
        return self.dynamics.metrics.dimension

    @property
    def _cell_kind(self) -> str:
        return self.dynamics.discretization.mesh.blocks[0].cell_kind

    def _differentiate(self, value: Array, axis: int, /) -> Array:
        moved = jnp.moveaxis(value, axis + 1, 1)
        differentiated = ein.contract(
            "ij,cj...->ci...",
            self.dynamics.sbp.derivative_matrix,
            moved,
            backend="jax",
        )
        return jnp.moveaxis(differentiated, 1, axis + 1)

    def _physical_gradient(self, local: Array, /) -> Array:
        reference = jnp.stack(
            tuple(self._differentiate(local, axis) for axis in range(self._dimension)),
            axis=-1,
        )
        return (
            ein.contract(
                "c...ia,c...ak->c...ik",
                reference,
                self.dynamics.metrics.contravariant_cofactors,
                backend="jax",
            )
            / self.dynamics.metrics.determinant[..., None, None]
        )

    def _face_weights(self) -> Array:
        result = self.dynamics.sbp.norm_weights
        for _axis in range(1, self._dimension - 1):
            result = ein.contract("...i,j->...ij", result, self.dynamics.sbp.norm_weights)
        return result.reshape((-1,))

    def _local_mass(self) -> Array:
        routes = self.dynamics.discretization.dof_maps[0].cell_dofs[0]
        node_count = self.dynamics.sbp.node_count
        return self.dynamics.scalar_mass_weights[routes].reshape(
            (routes.shape[0],) + (node_count,) * self._dimension
        )

    def _face_index(self, cell: int, axis: int, side: int, /) -> tuple:
        index = [int(cell)] + [slice(None)] * self._dimension
        index[axis + 1] = 0 if side == 0 else -1
        return tuple(index)

    def _face_tensor(self, local: Array, cell: int, axis: int, side: int, /) -> Array:
        value = jnp.take(local[int(cell)], 0 if side == 0 else -1, axis=axis)
        return value.reshape((-1,) + local.shape[-2:])

    def _add_gradient_face(
        self,
        gradient: Array,
        local_mass: Array,
        cell: int,
        axis: int,
        side: int,
        state_correction: Array,
        normal: Array,
        surface_jacobian: Array,
        /,
    ) -> Array:
        index = self._face_index(cell, axis, side)
        mass = local_mass[index].reshape((-1,))
        scale = self._face_weights() * surface_jacobian / mass
        correction = (
            scale[:, None, None] * state_correction[..., :, None] * normal[..., None, :]
        )
        face_shape = gradient[index].shape
        return gradient.at[index].add(correction.reshape(face_shape))

    def _add_flux_face(
        self,
        rate: Array,
        local_mass: Array,
        cell: int,
        axis: int,
        side: int,
        flux_correction: Array,
        surface_jacobian: Array,
        /,
    ) -> Array:
        index = self._face_index(cell, axis, side)
        mass = local_mass[index].reshape((-1,))
        scale = self._face_weights() * surface_jacobian / mass
        correction = scale[:, None] * flux_correction
        return rate.at[index].add(correction.reshape(rate[index].shape))

    def corrected_gradient(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> Array:
        value = self.dynamics._state(state)
        local = self.dynamics._local_state(value)
        gradient = self._physical_gradient(local)
        local_mass = self._local_mass()
        for pair, permutation in zip(
            self.dynamics.face_pairs,
            self.dynamics.face_permutations,
            strict=True,
        ):
            plus = self.dynamics._face_value(
                local, pair.owner_cell, pair.owner_axis, pair.owner_side
            )
            minus = self.dynamics._face_value(
                local, pair.neighbour_cell, pair.neighbour_axis, pair.neighbour_side
            )[permutation]
            scaled_normal = self.dynamics.metrics.face_scaled_normals[pair.owner_axis][
                pair.owner_cell, pair.owner_side
            ].reshape((-1, self._dimension))
            measure = jnp.sqrt(
                ein.contract("qd,qd->q", scaled_normal, scaled_normal, backend="jax")
            )
            normal = scaled_normal / measure[:, None]
            common = 0.5 * (plus + minus) + self.plan.beta * (plus - minus)
            gradient = self._add_gradient_face(
                gradient,
                local_mass,
                pair.owner_cell,
                pair.owner_axis,
                pair.owner_side,
                common - plus,
                normal,
                measure,
            )
            inverse = jnp.argsort(permutation)
            gradient = self._add_gradient_face(
                gradient,
                local_mass,
                pair.neighbour_cell,
                pair.neighbour_axis,
                pair.neighbour_side,
                (common - minus)[inverse],
                (-normal)[inverse],
                measure[inverse],
            )
        if self.dynamics.boundaries is not None:
            context = self.dynamics._context(jnp.asarray(time), args)
            for patch in self.dynamics.boundaries.patches:
                owners = np.asarray(patch.domain.owner_cells, dtype=np.int32)
                local_facets = np.asarray(
                    patch.domain.owner_local_entities, dtype=np.int32
                )
                for owner_cell, local_facet in zip(owners, local_facets, strict=True):
                    axis, side = tensor_local_face(self._cell_kind, int(local_facet))
                    plus = self.dynamics._face_value(local, int(owner_cell), axis, side)
                    points = self.dynamics.metrics.face_coordinates[axis][
                        int(owner_cell), side
                    ].reshape((-1, self._dimension))
                    scaled_normal = self.dynamics.metrics.face_scaled_normals[axis][
                        int(owner_cell), side
                    ].reshape((-1, self._dimension))
                    measure = jnp.sqrt(
                        ein.contract(
                            "qd,qd->q", scaled_normal, scaled_normal, backend="jax"
                        )
                    )
                    normal = scaled_normal / measure[:, None]
                    exterior = patch.boundary.exterior_state(
                        self.dynamics.system,
                        jnp.asarray(time),
                        plus,
                        points,
                        normal,
                        axis,
                        context.user_args,
                    )
                    gradient = self._add_gradient_face(
                        gradient,
                        local_mass,
                        int(owner_cell),
                        axis,
                        side,
                        exterior - plus,
                        normal,
                        measure,
                    )
        return gradient

    def rate(self, time: Array, state: ArrayLike, args: Any = None, /) -> Array:
        value = self.dynamics._state(state)
        local = self.dynamics._local_state(value)
        gradient = self.corrected_gradient(time, value, args)
        context = self.dynamics._context(jnp.asarray(time), args)
        flux = self.dynamics.system.viscous_flux(local, gradient, context.user_args)
        contravariant = ein.contract(
            "c...ik,c...ak->c...ia",
            flux,
            self.dynamics.metrics.contravariant_cofactors,
            backend="jax",
        )
        divergence = jnp.zeros_like(local)
        for axis in range(self._dimension):
            divergence = divergence + self._differentiate(contravariant[..., axis], axis)
        rate = divergence / self.dynamics.metrics.determinant[..., None]
        local_mass = self._local_mass()
        for pair, permutation in zip(
            self.dynamics.face_pairs,
            self.dynamics.face_permutations,
            strict=True,
        ):
            plus = self.dynamics._face_value(
                local, pair.owner_cell, pair.owner_axis, pair.owner_side
            )
            minus = self.dynamics._face_value(
                local, pair.neighbour_cell, pair.neighbour_axis, pair.neighbour_side
            )[permutation]
            plus_flux = self._face_tensor(
                flux, pair.owner_cell, pair.owner_axis, pair.owner_side
            )
            minus_flux = self._face_tensor(
                flux, pair.neighbour_cell, pair.neighbour_axis, pair.neighbour_side
            )[permutation]
            scaled_normal = self.dynamics.metrics.face_scaled_normals[pair.owner_axis][
                pair.owner_cell, pair.owner_side
            ].reshape((-1, self._dimension))
            measure = jnp.sqrt(
                ein.contract("qd,qd->q", scaled_normal, scaled_normal, backend="jax")
            )
            normal = scaled_normal / measure[:, None]
            plus_normal = ein.contract("qik,qk->qi", plus_flux, normal, backend="jax")
            minus_normal = ein.contract("qik,qk->qi", minus_flux, normal, backend="jax")
            common = (
                0.5 * (plus_normal + minus_normal)
                + self.plan.beta * (plus_normal - minus_normal)
                + self.plan.penalty * (minus - plus)
            )
            rate = self._add_flux_face(
                rate,
                local_mass,
                pair.owner_cell,
                pair.owner_axis,
                pair.owner_side,
                common - plus_normal,
                measure,
            )
            inverse = jnp.argsort(permutation)
            rate = self._add_flux_face(
                rate,
                local_mass,
                pair.neighbour_cell,
                pair.neighbour_axis,
                pair.neighbour_side,
                (-common + minus_normal)[inverse],
                measure[inverse],
            )
        if self.dynamics.boundaries is not None:
            local_context = self.dynamics._context(jnp.asarray(time), args)
            for patch in self.dynamics.boundaries.patches:
                owners = np.asarray(patch.domain.owner_cells, dtype=np.int32)
                local_facets = np.asarray(
                    patch.domain.owner_local_entities, dtype=np.int32
                )
                for owner_cell, local_facet in zip(owners, local_facets, strict=True):
                    axis, side = tensor_local_face(self._cell_kind, int(local_facet))
                    plus = self.dynamics._face_value(local, int(owner_cell), axis, side)
                    plus_gradient = self._face_tensor(
                        gradient, int(owner_cell), axis, side
                    )
                    plus_flux = self._face_tensor(flux, int(owner_cell), axis, side)
                    points = self.dynamics.metrics.face_coordinates[axis][
                        int(owner_cell), side
                    ].reshape((-1, self._dimension))
                    scaled_normal = self.dynamics.metrics.face_scaled_normals[axis][
                        int(owner_cell), side
                    ].reshape((-1, self._dimension))
                    measure = jnp.sqrt(
                        ein.contract(
                            "qd,qd->q", scaled_normal, scaled_normal, backend="jax"
                        )
                    )
                    normal = scaled_normal / measure[:, None]
                    exterior = patch.boundary.exterior_state(
                        self.dynamics.system,
                        jnp.asarray(time),
                        plus,
                        points,
                        normal,
                        axis,
                        local_context.user_args,
                    )
                    exterior_flux = self.dynamics.system.viscous_flux(
                        exterior, plus_gradient, local_context.user_args
                    )
                    plus_normal = ein.contract(
                        "qik,qk->qi", plus_flux, normal, backend="jax"
                    )
                    exterior_normal = ein.contract(
                        "qik,qk->qi", exterior_flux, normal, backend="jax"
                    )
                    common = (
                        0.5 * (plus_normal + exterior_normal)
                        + self.plan.beta * (plus_normal - exterior_normal)
                        + self.plan.penalty * (exterior - plus)
                    )
                    if isinstance(patch.boundary, PrescribedHeatFluxWallBoundary):
                        prescribed = patch.boundary.normal_heat_flux(
                            jnp.asarray(time),
                            plus,
                            points,
                            normal,
                            local_context.user_args,
                        )
                        traction = common[..., 1 : 1 + self._dimension]
                        mechanical = ein.contract(
                            "i,qi->q",
                            patch.boundary.wall_velocity.astype(common.dtype),
                            traction,
                            backend="jax",
                        )
                        common = common.at[..., -1].set(mechanical + prescribed)
                    rate = self._add_flux_face(
                        rate,
                        local_mass,
                        int(owner_cell),
                        axis,
                        side,
                        common - plus_normal,
                        measure,
                    )
        routes = self.dynamics.discretization.dof_maps[0].cell_dofs[0]
        result = jnp.zeros_like(value)
        return result.at[routes].set(rate.reshape(value[routes].shape))

    def weak_residual(self, time: Array, state: ArrayLike, args: Any = None, /) -> Array:
        return -self.dynamics.mass_operator.mv(self.rate(time, state, args))

    def linearize(self, time: Array, state: ArrayLike, args: Any = None, /):
        value = self.dynamics._state(state)
        residual, pushforward = jax.linearize(
            lambda candidate: self.rate(time, candidate, args), value
        )
        _, pullback = jax.vjp(lambda candidate: self.rate(time, candidate, args), value)
        return residual, pushforward, pullback

    def stability_evidence(
        self, state: ArrayLike, args: Any = None, /, *, cfl: float = 0.2
    ) -> ViscousDGStabilityEvidence:
        value = self.dynamics._state(state)
        local = self.dynamics._local_state(value)
        context = self.dynamics._context(jnp.asarray(0.0, dtype=value.dtype), args)
        diffusivity = self.dynamics.system.maximum_diffusivity(local, context.user_args)
        inverse_metric_scale = (
            jnp.sum(self.dynamics.metrics.contravariant_cofactors**2, axis=(-2, -1))
            / self.dynamics.metrics.determinant**2
        )
        maximum = jnp.max(
            (self.dynamics.sbp.order + 1) ** 2 * diffusivity * inverse_metric_scale
        )
        cfl_ = float(cfl)
        if not math.isfinite(cfl_) or cfl_ <= 0.0:
            raise ValueError("Viscous CFL must be finite and positive.")
        step = jnp.asarray(cfl_, dtype=maximum.dtype) / jnp.where(
            maximum > 0.0, maximum, jnp.inf
        )
        return ViscousDGStabilityEvidence(
            step,
            maximum,
            jnp.isfinite(step) & (step > 0.0),
            self.plan.plan_id,
        )


__all__ = [
    "EntropyDiffusionEvidence",
    "entropy_diffusion_evidence",
    "PreparedViscousDGOperator",
    "ViscousBoundaryClosure",
    "ViscousDGPlan",
    "ViscousDGStabilityEvidence",
]
