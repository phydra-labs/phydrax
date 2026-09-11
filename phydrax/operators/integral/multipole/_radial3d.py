#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Outgoing Helmholtz and screened modified-Helmholtz multipole families."""

from __future__ import annotations

import math
from operator import index
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._trainable import NonTrainableState
from ....special._spherical_bessel import (
    _spherical_hankel1_sequence,
    _spherical_i_sequence,
    _spherical_j_sequence,
    _spherical_k_sequence,
)
from ._laplace3d import (
    _flatten_payload,
    _mode_basis,
    _translation_quadrature,
    LaplaceMultipoleEvaluation3D,
    LaplaceMultipolePlan3D,
    PreparedLaplaceMultipole3D,
)


RadialKernel3D = Literal["helmholtz", "modified-helmholtz"]


class HelmholtzMultipoleEvaluation3D(LaplaceMultipoleEvaluation3D):
    """Complete outgoing-Helmholtz FMM evaluation."""

    wavenumber: float = eqx.field(static=True)


class ModifiedHelmholtzMultipoleEvaluation3D(LaplaceMultipoleEvaluation3D):
    """Complete screened modified-Helmholtz FMM evaluation."""

    decay: float = eqx.field(static=True)


class _RadialMultipolePlan3D(eqx.Module, NonTrainableState):
    substrate: LaplaceMultipolePlan3D
    parameter: float = eqx.field(static=True)
    kernel: RadialKernel3D = eqx.field(static=True)
    maximum_dimensionless_argument: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_sources: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        parameter: float,
        kernel: RadialKernel3D,
        reference_targets: ArrayLike | None = None,
        depth: int = 3,
        expansion_order: int = 4,
        maximum_reference_displacement: float = 0.0,
        far_interaction_capacity: int | None = None,
        near_interaction_capacity: int | None = None,
        maximum_coefficient_bytes: int = 512 * 1024**2,
        maximum_dimensionless_argument: float = 24.0,
    ):
        parameter_ = float(parameter)
        maximum_argument = float(maximum_dimensionless_argument)
        order = index(expansion_order)
        if not math.isfinite(parameter_) or parameter_ <= 0.0:
            raise ValueError("The radial kernel parameter must be finite and positive.")
        if not math.isfinite(maximum_argument) or maximum_argument <= 0.0:
            raise ValueError(
                "maximum_dimensionless_argument must be finite and positive."
            )
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        diameter = float(np.linalg.norm(upper_ - lower_))
        if parameter_ * diameter > maximum_argument:
            raise ValueError(
                "The requested radial translation domain exceeds "
                "maximum_dimensionless_argument."
            )
        if order > 12:
            raise ValueError(
                "Dense radial translations currently support order at most 12."
            )
        substrate = LaplaceMultipolePlan3D(
            reference_sources,
            lower,
            upper,
            reference_targets=reference_targets,
            depth=depth,
            expansion_order=order,
            maximum_reference_displacement=maximum_reference_displacement,
            far_interaction_capacity=far_interaction_capacity,
            near_interaction_capacity=near_interaction_capacity,
            maximum_coefficient_bytes=maximum_coefficient_bytes,
            translation_route="dense",
        )
        self.substrate = substrate
        self.parameter = parameter_
        self.kernel = kernel
        self.maximum_dimensionless_argument = maximum_argument
        self.plan_id = canonical_fingerprint(
            {
                "kind": f"{kernel}-multipole-plan-3d",
                "substrate": substrate.plan_id,
                "parameter": parameter_,
                "maximum_dimensionless_argument": maximum_argument,
                "radiation": "outgoing" if kernel == "helmholtz" else "decaying",
            }
        )


class HelmholtzMultipolePlan3D(_RadialMultipolePlan3D):
    """Bounded dense outgoing-Helmholtz radial translation plan."""

    def __init__(
        self,
        reference_sources: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        wavenumber: float,
        reference_targets: ArrayLike | None = None,
        depth: int = 3,
        expansion_order: int = 4,
        maximum_reference_displacement: float = 0.0,
        far_interaction_capacity: int | None = None,
        near_interaction_capacity: int | None = None,
        maximum_coefficient_bytes: int = 512 * 1024**2,
        maximum_dimensionless_argument: float = 24.0,
    ):
        super().__init__(
            reference_sources,
            lower,
            upper,
            parameter=wavenumber,
            kernel="helmholtz",
            reference_targets=reference_targets,
            depth=depth,
            expansion_order=expansion_order,
            maximum_reference_displacement=maximum_reference_displacement,
            far_interaction_capacity=far_interaction_capacity,
            near_interaction_capacity=near_interaction_capacity,
            maximum_coefficient_bytes=maximum_coefficient_bytes,
            maximum_dimensionless_argument=maximum_dimensionless_argument,
        )

    @property
    def wavenumber(self) -> float:
        return self.parameter

    def prepare(self, /) -> PreparedHelmholtzMultipole3D:
        return PreparedHelmholtzMultipole3D(self)


class ModifiedHelmholtzMultipolePlan3D(_RadialMultipolePlan3D):
    """Bounded dense Yukawa/modified-Helmholtz radial translation plan."""

    def __init__(
        self,
        reference_sources: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        decay: float,
        reference_targets: ArrayLike | None = None,
        depth: int = 3,
        expansion_order: int = 4,
        maximum_reference_displacement: float = 0.0,
        far_interaction_capacity: int | None = None,
        near_interaction_capacity: int | None = None,
        maximum_coefficient_bytes: int = 512 * 1024**2,
        maximum_dimensionless_argument: float = 24.0,
    ):
        super().__init__(
            reference_sources,
            lower,
            upper,
            parameter=decay,
            kernel="modified-helmholtz",
            reference_targets=reference_targets,
            depth=depth,
            expansion_order=expansion_order,
            maximum_reference_displacement=maximum_reference_displacement,
            far_interaction_capacity=far_interaction_capacity,
            near_interaction_capacity=near_interaction_capacity,
            maximum_coefficient_bytes=maximum_coefficient_bytes,
            maximum_dimensionless_argument=maximum_dimensionless_argument,
        )

    @property
    def decay(self) -> float:
        return self.parameter

    def prepare(self, /) -> PreparedModifiedHelmholtzMultipole3D:
        return PreparedModifiedHelmholtzMultipole3D(self)


class _PreparedRadialMultipole3D(PreparedLaplaceMultipole3D):
    kernel_plan: _RadialMultipolePlan3D
    projection_directions: Array
    projection_weights: Array
    projection_harmonics: Array
    equivalent_radius: float = eqx.field(static=True)

    def __init__(self, plan: _RadialMultipolePlan3D, /):
        if not isinstance(plan, _RadialMultipolePlan3D):
            raise TypeError("plan must be a radial multipole plan.")
        super().__init__(plan.substrate)
        quadrature_limit = 3 * self.layout.bandlimit + 2
        directions_, weights_ = _translation_quadrature(quadrature_limit)
        directions = jnp.asarray(directions_)
        weights = jnp.asarray(weights_)
        harmonics = _mode_basis(self.layout, directions, radial="regular")
        extent = np.asarray(plan.substrate.upper) - np.asarray(plan.substrate.lower)
        radius = min(0.125 * float(np.min(extent)), 0.5 / plan.parameter)
        self.kernel_plan = plan
        self.projection_directions = directions
        self.projection_weights = weights
        self.projection_harmonics = harmonics
        self.equivalent_radius = radius
        self.prepared_id = canonical_fingerprint(
            {
                "kind": f"prepared-{plan.kernel}-multipole-3d",
                "plan": plan.plan_id,
                "layout": self.layout.layout_id,
                "topology": self.topology.tree_id,
                "projection_nodes": int(directions.shape[0]),
                "equivalent_radius": radius,
            }
        )
        if plan.kernel == "helmholtz":
            self.source_convention = (
                "M[l,m]=i*k*sum(q*j_l(k*r)*conj(Y[l,m])); "
                "field=sum(M[l,m]*h_l^(1)(k*r)*Y[l,m])"
            )
            self.local_convention = (
                "L[l,m]=i*k*sum(q*h_l^(1)(k*r)*conj(Y[l,m])); "
                "field=sum(L[l,m]*j_l(k*r)*Y[l,m])"
            )
        else:
            self.source_convention = (
                "M[l,m]=(2*kappa/pi)*sum(q*i_l(kappa*r)*conj(Y[l,m])); "
                "field=sum(M[l,m]*k_l(kappa*r)*Y[l,m])"
            )
            self.local_convention = (
                "L[l,m]=(2*kappa/pi)*sum(q*k_l(kappa*r)*conj(Y[l,m])); "
                "field=sum(L[l,m]*i_l(kappa*r)*Y[l,m])"
            )

    def _radial_sequence(
        self, radius: Array, /, *, radial: Literal["regular", "irregular"]
    ) -> Array:
        argument = self.kernel_plan.parameter * radius
        maximum = self.layout.bandlimit - 1
        if self.kernel_plan.kernel == "helmholtz":
            return (
                _spherical_j_sequence(maximum, argument)
                if radial == "regular"
                else _spherical_hankel1_sequence(maximum, argument)
            )
        return (
            _spherical_i_sequence(maximum, argument)
            if radial == "regular"
            else _spherical_k_sequence(maximum, argument)
        )

    def _wave_basis(
        self,
        vectors: Array,
        /,
        *,
        radial: Literal["regular", "irregular"],
    ) -> Array:
        values = jnp.asarray(vectors)
        radius = jnp.linalg.norm(values, axis=-1)
        valid_direction = radius > 0.0
        fallback = jnp.asarray((0.0, 0.0, 1.0), dtype=values.dtype)
        directions = jnp.where(
            valid_direction[..., None],
            values / jnp.where(valid_direction, radius, 1.0)[..., None],
            fallback,
        )
        angular = _mode_basis(self.layout, directions, radial="regular")
        radial_values = self._radial_sequence(radius, radial=radial)
        return angular * radial_values[:, None]

    def _kernel_prefactor(self, dtype) -> Array:
        parameter = jnp.asarray(self.kernel_plan.parameter, dtype=dtype)
        if self.kernel_plan.kernel == "helmholtz":
            return 1j * parameter
        return 2.0 * parameter / jnp.pi

    def _p2m_basis(self, relative: Array, /) -> Array:
        basis = self._wave_basis(relative, radial="regular")
        return self._kernel_prefactor(relative.dtype) * jnp.conj(basis)

    def _p2l_basis(self, relative: Array, /) -> Array:
        basis = self._wave_basis(relative, radial="irregular")
        return self._kernel_prefactor(relative.dtype) * jnp.conj(basis)

    def _evaluate_basis(self, coefficients: Array, relative: Array, radial: str) -> Array:
        modal = self._validate_coefficients(coefficients, "coefficients")
        basis = self._wave_basis(relative, radial=radial)
        modal_flat, payload_shape = _flatten_payload(modal, 2)
        point_shape = tuple(relative.shape[:-1])
        point_count = math.prod(point_shape) if point_shape else 1
        result = basis.reshape((-1, point_count)).T @ modal_flat.reshape(
            (-1, modal_flat.shape[-1])
        )
        return result.reshape(point_shape + payload_shape)

    def _project_on_sphere(
        self,
        function,
        center: Array,
        radius: Array,
        /,
        *,
        radial: Literal["regular", "irregular"],
    ) -> Array:
        directions = self.projection_directions.astype(center.dtype)
        weights = self.projection_weights.astype(center.dtype)
        points = center[None, :] + radius * directions
        values = function(points)
        payload_shape = tuple(values.shape[1:])
        output = jnp.zeros(
            self.layout.coefficient_shape + payload_shape,
            dtype=jnp.result_type(values.dtype, 1j),
        )
        radial_values = self._radial_sequence(radius, radial=radial)
        offset = self.layout.bandlimit - 1
        for degree in range(self.layout.bandlimit):
            for order in range(-degree, degree + 1):
                harmonic = self.projection_harmonics[degree, offset + order]
                projected = jnp.tensordot(
                    weights * jnp.conj(harmonic), values, axes=((0,), (0,))
                )
                output = output.at[degree, offset + order].set(
                    projected / radial_values[degree]
                )
        return self._mask_coefficients(output)

    def m2m(
        self,
        multipole: ArrayLike,
        child_center: ArrayLike,
        parent_center: ArrayLike,
        /,
    ) -> Array:
        modal = self._validate_coefficients(multipole, "multipole")
        child = jnp.asarray(child_center)
        parent = jnp.asarray(parent_center, dtype=child.dtype)
        if child.shape != (3,) or parent.shape != (3,):
            raise ValueError("M2M centers must have shape (3,).")
        displacement = jnp.linalg.norm(child - parent)
        radius = 4.0 * displacement + self.equivalent_radius
        return self._project_on_sphere(
            lambda points: self.multipole_to_point(modal, child, points),
            parent,
            radius,
            radial="irregular",
        )

    def m2l(
        self,
        multipole: ArrayLike,
        source_center: ArrayLike,
        target_center: ArrayLike,
        /,
    ) -> Array:
        modal = self._validate_coefficients(multipole, "multipole")
        source = jnp.asarray(source_center)
        target = jnp.asarray(target_center, dtype=source.dtype)
        if source.shape != (3,) or target.shape != (3,):
            raise ValueError("M2L centers must have shape (3,).")
        distance = jnp.linalg.norm(target - source)
        target = eqx.error_if(target, distance <= 0.0, "M2L centers must be distinct.")
        radius = jnp.minimum(0.2 * distance, self.equivalent_radius)
        return self._project_on_sphere(
            lambda points: self.multipole_to_point(modal, source, points),
            target,
            radius,
            radial="regular",
        )

    def l2l(
        self,
        local: ArrayLike,
        parent_center: ArrayLike,
        child_center: ArrayLike,
        /,
    ) -> Array:
        modal = self._validate_coefficients(local, "local")
        parent = jnp.asarray(parent_center)
        child = jnp.asarray(child_center, dtype=parent.dtype)
        if parent.shape != (3,) or child.shape != (3,):
            raise ValueError("L2L centers must have shape (3,).")
        return self._project_on_sphere(
            lambda points: self.l2p(modal, parent, points),
            child,
            jnp.asarray(self.equivalent_radius, dtype=child.dtype),
            radial="regular",
        )

    def _p2p_masked(
        self,
        sources: Array,
        strengths: Array,
        targets: Array,
        pair_mask: Array,
        /,
        *,
        source_normals: ArrayLike | None,
    ) -> tuple[Array, Array]:
        differences = targets[:, None, :] - sources[None, :, :]
        squared = jnp.sum(differences * differences, axis=-1)
        strengths = eqx.error_if(
            strengths,
            jnp.any(pair_mask & (squared == 0.0)),
            "A non-excluded radial-kernel point pair is singular.",
        )
        safe_squared = jnp.where(pair_mask, squared, 1.0)
        radii = jnp.sqrt(safe_squared)
        parameter = jnp.asarray(self.kernel_plan.parameter, dtype=radii.dtype)
        if self.kernel_plan.kernel == "helmholtz":
            exponential = jnp.exp(1j * parameter * radii)
            if source_normals is None:
                kernels = exponential / (4.0 * jnp.pi * radii)
            else:
                normals = jnp.asarray(source_normals, dtype=sources.dtype)
                if normals.shape != sources.shape:
                    raise ValueError("source_normals must match source_positions.")
                kernels = (
                    exponential
                    * (1.0 - 1j * parameter * radii)
                    * jnp.sum(differences * normals[None, :, :], axis=-1)
                    / (4.0 * jnp.pi * safe_squared * radii)
                )
        else:
            exponential = jnp.exp(-parameter * radii)
            if source_normals is None:
                kernels = exponential / (4.0 * jnp.pi * radii)
            else:
                normals = jnp.asarray(source_normals, dtype=sources.dtype)
                if normals.shape != sources.shape:
                    raise ValueError("source_normals must match source_positions.")
                kernels = (
                    exponential
                    * (1.0 + parameter * radii)
                    * jnp.sum(differences * normals[None, :, :], axis=-1)
                    / (4.0 * jnp.pi * safe_squared * radii)
                )
        kernels = jnp.where(pair_mask, kernels, 0.0)
        strength_flat, payload_shape = _flatten_payload(strengths, 1)
        values = kernels @ strength_flat
        return values.reshape((targets.shape[0],) + payload_shape), jnp.sum(
            pair_mask, dtype=jnp.int32
        )

    def _wrap_evaluation(self, evaluation: LaplaceMultipoleEvaluation3D):
        common = dict(
            values=evaluation.values,
            far_values=evaluation.far_values,
            near_values=evaluation.near_values,
            truncation=evaluation.truncation,
            capacity=evaluation.capacity,
            maximum_reference_displacement=evaluation.maximum_reference_displacement,
            stale_topology=evaluation.stale_topology,
            finite=evaluation.finite,
            successful=evaluation.successful,
            p2m_count=evaluation.p2m_count,
            m2m_count=evaluation.m2m_count,
            m2l_count=evaluation.m2l_count,
            l2l_count=evaluation.l2l_count,
            l2p_count=evaluation.l2p_count,
            p2p_count=evaluation.p2p_count,
            expansion_order=evaluation.expansion_order,
            source_convention=self.source_convention,
            local_convention=self.local_convention,
            evaluation_id=canonical_fingerprint(
                {
                    "kind": f"{self.kernel_plan.kernel}-multipole-evaluation-3d",
                    "prepared": self.prepared_id,
                }
            ),
        )
        if self.kernel_plan.kernel == "helmholtz":
            return HelmholtzMultipoleEvaluation3D(
                **common, wavenumber=self.kernel_plan.parameter
            )
        return ModifiedHelmholtzMultipoleEvaluation3D(
            **common, decay=self.kernel_plan.parameter
        )

    def evaluate(
        self,
        source_positions: ArrayLike,
        source_strengths: ArrayLike,
        target_positions: ArrayLike | None = None,
        /,
        *,
        source_normals: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
        target_source_indices: ArrayLike | None = None,
    ):
        evaluation = super().evaluate(
            source_positions,
            source_strengths,
            target_positions,
            source_normals=source_normals,
            active_mask=active_mask,
            target_source_indices=target_source_indices,
        )
        return self._wrap_evaluation(evaluation)


class PreparedHelmholtzMultipole3D(_PreparedRadialMultipole3D):
    """Prepared complete outgoing-Helmholtz multipole pipeline."""

    def __init__(self, plan: HelmholtzMultipolePlan3D, /):
        if not isinstance(plan, HelmholtzMultipolePlan3D):
            raise TypeError("plan must be HelmholtzMultipolePlan3D.")
        super().__init__(plan)


class PreparedModifiedHelmholtzMultipole3D(_PreparedRadialMultipole3D):
    """Prepared complete screened modified-Helmholtz multipole pipeline."""

    def __init__(self, plan: ModifiedHelmholtzMultipolePlan3D, /):
        if not isinstance(plan, ModifiedHelmholtzMultipolePlan3D):
            raise TypeError("plan must be ModifiedHelmholtzMultipolePlan3D.")
        super().__init__(plan)


__all__ = [
    "HelmholtzMultipoleEvaluation3D",
    "HelmholtzMultipolePlan3D",
    "ModifiedHelmholtzMultipoleEvaluation3D",
    "ModifiedHelmholtzMultipolePlan3D",
    "PreparedHelmholtzMultipole3D",
    "PreparedModifiedHelmholtzMultipole3D",
]
