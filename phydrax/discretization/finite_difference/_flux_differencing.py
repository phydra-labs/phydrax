#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import AbstractPreparedDiscretization, validate_prepared_metadata
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace
from .._tensor_support import PreparedTensorGrid
from ..finite_volume._riemann import AbstractSymmetricTwoPointFluxPlan
from ._sbp import PreparedSBPOperator, SBPDerivativePlan, SBPInteriorOrder


class TensorSBPPlan(StrictModule, NonTrainableState):
    """Periodic tensor-product diagonal-norm SBP state-space preparation."""

    grid: PreparedTensorGrid
    field_name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    interior_order: SBPInteriorOrder = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        field_name: str = "state",
        component_names: Sequence[str],
        interior_order: SBPInteriorOrder = 4,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        if any(
            not axis.periodic or axis.primary_entity != "point"
            for axis in grid.structured_axes
        ):
            raise ValueError(
                "Initial tensor SBP conservation requires periodic point-primary axes."
            )
        field = str(field_name)
        components = tuple(str(name) for name in component_names)
        if (
            not field
            or not components
            or any(not name for name in components)
            or len(set(components)) != len(components)
        ):
            raise ValueError("Field and component names must be unique and non-empty.")
        order = int(interior_order)
        if order not in (2, 4, 6, 8):
            raise ValueError("Tensor SBP order must be 2, 4, 6, or 8.")
        self.grid = grid
        self.field_name = field
        self.component_names = components
        self.interior_order = order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tensor-sbp-plan-v1",
                "grid": grid.prepared_id,
                "field": field,
                "components": list(components),
                "interior_order": order,
            }
        )

    def prepare(self, /) -> TensorSBPDiscretization:
        return TensorSBPDiscretization(self)


class TensorSBPDiscretization(AbstractPreparedDiscretization):
    """Prepared periodic tensor SBP calculus and conserved nodal state space."""

    grid: PreparedTensorGrid
    derivatives: tuple[PreparedSBPOperator, ...]
    state_space: DiscreteFieldSpace
    support: Any
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    preparation: PreparationReport
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(self, plan: TensorSBPPlan, /):
        if not isinstance(plan, TensorSBPPlan):
            raise TypeError("plan must be a TensorSBPPlan.")
        derivatives = tuple(
            SBPDerivativePlan(
                plan.grid,
                axis,
                interior_order=plan.interior_order,
            ).prepare()
            for axis in plan.grid.axis_names
        )
        reference_norm = derivatives[0].norm_weights
        if any(
            not bool(jnp.allclose(reference_norm, derivative.norm_weights))
            for derivative in derivatives[1:]
        ):
            raise RuntimeError("Tensor periodic SBP norms disagree across axes.")
        field = plan.grid.field_space(
            plan.field_name,
            component_shape=(len(plan.component_names),),
            dtype=plan.grid.points.dtype,
            representation="point_value",
        )
        measure = DiscreteMeasure(
            "tensor_sbp_norm",
            plan.grid.support.support_id,
            plan.grid.primary_entity_layout.entity_set_id,
            reference_norm.reshape((-1,)),
            normalization="physical",
        )
        capabilities = (
            DiscretizationCapability.STRONG_DERIVATIVE,
            DiscretizationCapability.CONSERVATIVE_FLUX,
            DiscretizationCapability.MATRIX_FREE,
        )
        preparation = PreparationReport(
            capabilities=capabilities,
            diagnostics=tuple(
                derivative.stability_report.report_id for derivative in derivatives
            ),
            resource_counts={
                "state_dofs": field.vector_space.size,
                "axes": len(derivatives),
                "stencil_entries": sum(
                    int(derivative.operator.valid.size) for derivative in derivatives
                ),
            },
        )
        key = DiscretizationKey(
            "tensor_sbp",
            DiscretizationRole.PHYSICAL,
            domain_labels=plan.grid.axis_names,
        )
        spaces, measures, capabilities_ = validate_prepared_metadata(
            key=key,
            support=plan.grid.support,
            field_spaces=(field,),
            measures=(measure,),
            capabilities=capabilities,
            preparation=preparation,
        )
        identifier = canonical_fingerprint(
            {
                "kind": "tensor-sbp-discretization-v1",
                "plan": plan.plan_id,
                "derivatives": [value.prepared_id for value in derivatives],
                "field": field.field_space_id,
                "measure": measure.measure_id,
            }
        )
        self.grid = plan.grid
        self.derivatives = derivatives
        self.state_space = field
        self.support = plan.grid.support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities_
        self.preparation = preparation
        self.key = key
        self.plan_id = plan.plan_id
        self.prepared_id = identifier
        self.numeric_version = "tensor-sbp-v1"

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.state_space.vector_space.shape

    @property
    def quadrature_weights(self) -> Array:
        return self.derivatives[0].norm_weights


class SBPFluxDifferencingMethodPlan(StrictModule, NonTrainableState):
    """Symmetric two-point volume flux with optional entropy diagnostics."""

    volume_flux: AbstractSymmetricTwoPointFluxPlan
    entropy_diagnostics: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_flux: AbstractSymmetricTwoPointFluxPlan,
        /,
        *,
        entropy_diagnostics: bool = False,
    ):
        if not isinstance(volume_flux, AbstractSymmetricTwoPointFluxPlan):
            raise TypeError("volume_flux must be an AbstractSymmetricTwoPointFluxPlan.")
        if not volume_flux.symmetric or not volume_flux.consistent:
            raise ValueError("SBP volume flux must declare symmetry and consistency.")
        self.volume_flux = volume_flux
        self.entropy_diagnostics = bool(entropy_diagnostics)
        self.method_id = canonical_fingerprint(
            {
                "kind": "sbp-flux-differencing-method-v1",
                "volume_flux": volume_flux.flux_id,
                "entropy_diagnostics": bool(entropy_diagnostics),
            }
        )


class SBPFluxDifferencingReport(StrictModule, NonTrainableState):
    """Pair count, sparsity, and SBP identity evidence."""

    pair_counts: tuple[int, ...] = eqx.field(static=True)
    state_dofs: int = eqx.field(static=True)
    dense_pair_count: int = eqx.field(static=True)
    sparse: bool = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair_counts: Sequence[int],
        state_dofs: int,
        /,
        *,
        dynamics_id: str,
    ):
        counts = tuple(int(value) for value in pair_counts)
        dofs = int(state_dofs)
        dense = dofs * max(dofs - 1, 0) // 2
        sparse = all(count < dense for count in counts) if dofs > 2 else True
        self.pair_counts = counts
        self.state_dofs = dofs
        self.dense_pair_count = dense
        self.sparse = sparse
        self.passed = bool(counts) and all(value > 0 for value in counts) and sparse
        self.report_id = canonical_fingerprint(
            {
                "kind": "sbp-flux-differencing-report-v1",
                "dynamics": dynamics_id,
                "pair_counts": list(counts),
                "state_dofs": dofs,
                "dense_pair_count": dense,
            }
        )


class SBPFluxDifferencingDiagnostics(StrictModule):
    total_entropy: Array
    semidiscrete_entropy_rate: Array
    convective_entropy_rate: Array
    source_entropy_rate: Array
    conservation_rate: Array
    admissible: Array
    pair_id: str = eqx.field(static=True)


class PreparedSBPConservationDynamics(StrictModule):
    """Periodic conservative nodal dynamics using sparse SBP flux differencing."""

    system: Any
    discretization: TensorSBPDiscretization
    method: SBPFluxDifferencingMethodPlan
    source: Callable[[Array, Array, Array, Any], ArrayLike] | None = eqx.field(
        static=True
    )
    entropy_pair: Any
    pair_left: tuple[Array, ...]
    pair_right: tuple[Array, ...]
    pair_coefficients: tuple[Array, ...]
    row_sum_bounds: tuple[float, ...] = eqx.field(static=True)
    report: SBPFluxDifferencingReport
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Any,
        discretization: TensorSBPDiscretization,
        method: SBPFluxDifferencingMethodPlan,
        /,
        *,
        source: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
        entropy_pair: Any = None,
    ):
        if not isinstance(discretization, TensorSBPDiscretization):
            raise TypeError("discretization must be a TensorSBPDiscretization.")
        if not isinstance(method, SBPFluxDifferencingMethodPlan):
            raise TypeError("method must be an SBPFluxDifferencingMethodPlan.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        if method.entropy_diagnostics != (entropy_pair is not None):
            raise ValueError(
                "SBP entropy diagnostics and entropy_pair must be enabled together."
            )
        left_indices = []
        right_indices = []
        pair_weights = []
        row_bounds = []
        for derivative in discretization.derivatives:
            indices = np.asarray(derivative.operator.indices)
            weights = np.asarray(derivative.operator.weights)
            valid = np.asarray(derivative.operator.valid)
            axis_norm = np.asarray(derivative.axis_norm_weights)
            left = []
            right = []
            coefficients = []
            for row in range(indices.shape[0]):
                for slot in range(indices.shape[1]):
                    if not valid[row, slot]:
                        continue
                    column = int(indices[row, slot])
                    coefficient = float(axis_norm[row] * weights[row, slot])
                    if row < column and abs(coefficient) > 5e-14:
                        left.append(row)
                        right.append(column)
                        coefficients.append(2.0 * coefficient)
            left_indices.append(jnp.asarray(left, dtype=jnp.int32))
            right_indices.append(jnp.asarray(right, dtype=jnp.int32))
            pair_weights.append(
                jnp.asarray(coefficients, dtype=discretization.grid.points.dtype)
            )
            row_bounds.append(float(np.max(np.sum(np.abs(weights), axis=1))))
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-sbp-conservation-dynamics-v1",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "source": "none" if source is None else repr(source),
                "entropy_pair": None if entropy_pair is None else entropy_pair.pair_id,
            }
        )
        report = SBPFluxDifferencingReport(
            tuple(value.size for value in left_indices),
            max(discretization.grid.shape),
            dynamics_id=identifier,
        )
        if not report.passed:
            raise RuntimeError("Prepared SBP flux differencing failed sparsity evidence.")
        self.system = system
        self.discretization = discretization
        self.method = method
        self.source = source
        self.entropy_pair = entropy_pair
        self.pair_left = tuple(left_indices)
        self.pair_right = tuple(right_indices)
        self.pair_coefficients = tuple(pair_weights)
        self.row_sum_bounds = tuple(row_bounds)
        self.report = report
        self.dynamics_id = identifier

    def _validate_state(self, state: ArrayLike, /) -> Array:
        return self.discretization.state_space.vector_space.validate(jnp.asarray(state))

    def _source_value(self, time: Array, state: Array, args: Any, /) -> Array:
        if self.source is None:
            return jnp.zeros_like(state)
        value = jnp.asarray(
            self.source(time, state, self.discretization.grid.points, args)
        )
        if value.shape != state.shape:
            raise ValueError("SBP conservation source must match the state shape.")
        return value

    def _axis_residual(self, state: Array, axis: int, args: Any, /) -> Array:
        left = self.pair_left[axis]
        right = self.pair_right[axis]
        coefficients = self.pair_coefficients[axis]
        moved = jnp.moveaxis(state, axis, 0)
        flux = self.method.volume_flux.two_point_flux(
            self.system,
            moved[left],
            moved[right],
            axis,
            args,
        )
        shape = coefficients.shape + (1,) * (flux.ndim - 1)
        content = coefficients.reshape(shape) * flux
        accumulated = jnp.zeros_like(moved)
        accumulated = accumulated.at[left].add(content)
        accumulated = accumulated.at[right].add(-content)
        norm = self.discretization.derivatives[axis].axis_norm_weights
        norm_shape = norm.shape + (1,) * (moved.ndim - 1)
        residual = -accumulated / norm.reshape(norm_shape)
        return jnp.moveaxis(residual, 0, axis)

    def residual_parts(
        self,
        time: Array,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        value = self._validate_state(state)
        convective = jnp.zeros_like(value)
        for axis in range(len(self.discretization.grid.shape)):
            convective = convective + self._axis_residual(value, axis, args)
        return convective, self._source_value(jnp.asarray(time), value, args)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        convective, source = self.residual_parts(time, state, args)
        return convective + source

    def residual_with_diagnostics(
        self,
        time: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, SBPFluxDifferencingDiagnostics | None]:
        value = self._validate_state(state)
        convective, source = self.residual_parts(time, value, args)
        residual = convective + source
        if self.entropy_pair is None:
            return residual, None
        weights = self.discretization.quadrature_weights
        entropy_variables = self.entropy_pair.entropy_variables(value)
        convective_density = ein.contract(
            "...i,...i->...", entropy_variables, convective, backend="jax"
        )
        source_density = ein.contract(
            "...i,...i->...", entropy_variables, source, backend="jax"
        )
        convective_rate = jnp.sum(weights * convective_density)
        source_rate = jnp.sum(weights * source_density)
        total_entropy = jnp.sum(weights * self.entropy_pair.entropy(value))
        conservation_rate = compensated_sum(
            weights[..., None] * residual,
            axis=tuple(range(len(self.discretization.grid.shape))),
        )
        diagnostics = SBPFluxDifferencingDiagnostics(
            total_entropy=total_entropy,
            semidiscrete_entropy_rate=convective_rate + source_rate,
            convective_entropy_rate=convective_rate,
            source_entropy_rate=source_rate,
            conservation_rate=conservation_rate,
            admissible=jnp.all(self.entropy_pair.admissible(value)),
            pair_id=self.entropy_pair.pair_id,
        )
        return residual, diagnostics

    def stable_step(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> Array:
        value = self._validate_state(state)
        rate = jnp.asarray(0.0, dtype=value.real.dtype)
        for axis, row_bound in enumerate(self.row_sum_bounds):
            speed = self.system.max_wave_speed(value, value, axis, args)
            rate = rate + row_bound * jnp.max(speed)
        return jnp.asarray(cfl, dtype=rate.dtype) / jnp.where(rate > 0.0, rate, jnp.inf)

    def face_fluxes(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "SBP flux differencing has volume pairs, not finite-volume face fluxes."
        )

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        value = self._validate_state(state)
        residual, pushforward = jax.linearize(lambda item: self(time, item, args), value)
        _, pullback = jax.vjp(lambda item: self(time, item, args), value)
        return residual, pushforward, pullback


__all__ = [
    "PreparedSBPConservationDynamics",
    "SBPFluxDifferencingDiagnostics",
    "SBPFluxDifferencingMethodPlan",
    "SBPFluxDifferencingReport",
    "TensorSBPDiscretization",
    "TensorSBPPlan",
]
