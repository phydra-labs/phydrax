#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    PreparedMACMomentumOperators,
)
from ..discretization.spectral import (
    PeriodicLerayProjector,
    TensorSpectralDiscretization,
)
from ._mac_les import (
    _MAC_LES_REGIME,
    _mac_velocity_gradient,
    _periodic_uniform_mac_stress_rate,
)
from ._periodic_les import (
    _periodic_fourier_stress_rate,
    PeriodicFourierGridFilterPlan,
    PreparedPeriodicFourierGridFilter,
)


if TYPE_CHECKING:
    from ..closure_data._binding import (
        LearnedStressResult,
        PreparedLearnedStressBinding,
    )
    from ..solver._structured_incompressible import (
        MACPressureProjectionPlan,
        MACRateProjectionResult,
    )


LEARNED_STRESS_FEATURE_NAME = "resolved-velocity-gradient"
LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS = (
    "du_dx",
    "du_dy",
    "du_dz",
    "dv_dx",
    "dv_dy",
    "dv_dz",
    "dw_dx",
    "dw_dy",
    "dw_dz",
)
LEARNED_STRESS_VELOCITY_GRADIENT_UNITS = ("1/s",) * 9
_PERIODIC_LES_REGIME = "three-dimensional-periodic-unit-density"
_SPECIFIC_STRESS_UNITS = "(m/s)^2"


def _validate_binding_abi(binding: PreparedLearnedStressBinding, /) -> None:
    from ..closure_data._binding import PreparedLearnedStressBinding

    if not isinstance(binding, PreparedLearnedStressBinding):
        raise TypeError("binding must be a PreparedLearnedStressBinding.")
    schema = binding.plan.feature_schema
    output = binding.plan.output_contract
    if (
        schema.name != LEARNED_STRESS_FEATURE_NAME
        or schema.component_names != LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS
        or schema.component_units != LEARNED_STRESS_VELOCITY_GRADIENT_UNITS
        or schema.shape[-1:] != (9,)
    ):
        raise ValueError(
            "Learned stress backends require the exact row-major nine-component "
            "resolved-velocity-gradient feature ABI."
        )
    if output.units != _SPECIFIC_STRESS_UNITS:
        raise ValueError(
            "Learned stress backends require constant-density specific stress "
            "units '(m/s)^2'."
        )


def _deviatoric_strain(gradient: Array, /) -> Array:
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    trace = jnp.trace(strain, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=strain.dtype)
    return strain - (trace / 3.0)[..., None, None] * identity


def _policy_satisfied(
    result: LearnedStressResult,
    tolerance: Array,
    /,
) -> Array:
    policy = result.evidence.energy_policy
    if policy == "dissipative":
        return jnp.all(result.local_transfer >= -tolerance)
    if policy == "bounded_backscatter":
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=tolerance.dtype),
            jnp.abs(result.evidence.backscatter_limit),
        )
        return (
            result.evidence.selected_backscatter_transfer
            <= result.evidence.backscatter_limit + tolerance * scale
        )
    return jnp.asarray(True)


def _validate_bound_layout(
    binding: PreparedLearnedStressBinding,
    sample_shape: tuple[int, ...],
    dtype: Any,
    /,
) -> None:
    schema = binding.plan.feature_schema
    output = binding.plan.output_contract
    dtype_name = np.dtype(dtype).name
    if schema.shape != sample_shape + (9,):
        raise ValueError(
            "Learned stress feature layout does not match the backend sample grid."
        )
    if output.shape != sample_shape + (3, 3):
        raise ValueError(
            "Learned stress output layout does not match the backend sample grid."
        )
    if schema.dtype != dtype_name or output.dtype != dtype_name:
        raise TypeError("Learned stress dtype does not match backend physical precision.")


class PeriodicLearnedStressStage(StrictModule):
    """One bound learned stress and its exact retained Fourier momentum action."""

    velocity_gradient: Array
    strain: Array
    features: Array
    learned_result: LearnedStressResult
    modal_deviatoric_specific_stress: Array
    unprojected_rate: Array
    projected_rate: Array
    integrated_transfer: Array
    unprojected_work: Array
    projected_work: Array
    energy_identity_defect: Array
    projection_work_defect: Array
    momentum_conservation_defect: Array
    divergence_norm: Array
    imaginary_leakage: Array
    energy_policy_active: Array
    energy_policy_satisfied: Array
    energy_consistent: Array
    stable: Array
    finite: Array
    successful: Array
    feature_schema_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PeriodicLearnedStressPlan(StrictModule, NonTrainableState):
    """Bind learned specific stress to one exact three-dimensional Fourier backend."""

    binding: PreparedLearnedStressBinding
    grid_filter: PeriodicFourierGridFilterPlan
    energy_tolerance: float = eqx.field(static=True)
    feature_schema_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: PreparedLearnedStressBinding,
        /,
        *,
        energy_tolerance: float = 1.0e-9,
    ):
        _validate_binding_abi(binding)
        tolerance = float(energy_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_tolerance must be finite and nonnegative.")
        provenance = binding.plan.parameter_provenance
        if provenance.regime != _PERIODIC_LES_REGIME:
            raise ValueError(
                "Periodic learned stress requires the "
                "'three-dimensional-periodic-unit-density' regime."
            )
        grid_filter = PeriodicFourierGridFilterPlan(binding.plan.resolved_filter)
        self.binding = binding
        self.grid_filter = grid_filter
        self.energy_tolerance = tolerance
        self.feature_schema_id = binding.plan.feature_schema.feature_schema_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-learned-stress-plan",
                "binding": binding.prepared_id,
                "feature_schema": self.feature_schema_id,
                "grid_filter": grid_filter.plan_id,
                "energy_tolerance": tolerance,
                "feature_order": list(LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS),
                "stress_sign": "rhs-minus-divergence-of-specific-sgs-stress",
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        projector: PeriodicLerayProjector,
        /,
    ) -> PreparedPeriodicLearnedStress:
        return PreparedPeriodicLearnedStress(self, discretization, projector)


class PreparedPeriodicLearnedStress(StrictModule, NonTrainableState):
    """Prepared learned-stress adapter for an exact periodic Fourier grid."""

    plan: PeriodicLearnedStressPlan
    binding: PreparedLearnedStressBinding
    grid_filter: PreparedPeriodicFourierGridFilter
    projector: PeriodicLerayProjector
    feature_schema_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicLearnedStressPlan,
        discretization: TensorSpectralDiscretization,
        projector: PeriodicLerayProjector,
        /,
    ):
        if not isinstance(plan, PeriodicLearnedStressPlan):
            raise TypeError("plan must be a PeriodicLearnedStressPlan.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be a PeriodicLerayProjector.")
        if projector.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("Learned stress projector and discretization disagree.")
        if projector.spatial_dimension != 3:
            raise ValueError("Periodic learned stress is implemented only in 3-D.")
        provenance = plan.binding.plan.parameter_provenance
        if provenance.discretization_id != discretization.prepared_id:
            raise ValueError(
                "Learned stress provenance does not match the Fourier discretization."
            )
        grid_filter = plan.grid_filter.prepare(discretization)
        _validate_bound_layout(
            plan.binding,
            discretization.physical_shape,
            discretization.plan.precision.physical_dtype,
        )
        self.plan = plan
        self.binding = plan.binding
        self.grid_filter = grid_filter
        self.projector = projector
        self.feature_schema_id = plan.feature_schema_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-learned-stress",
                "plan": plan.plan_id,
                "binding": plan.binding.prepared_id,
                "feature_schema": plan.feature_schema_id,
                "filter": grid_filter.prepared_id,
                "projector": projector.projector_id,
                "discretization": discretization.prepared_id,
                "runtime_scope": "single-device-3d-unit-density-full-complex-fourier",
            }
        )

    def evaluate(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> PeriodicLearnedStressStage:
        """Build features, evaluate stress, and own divergence and projection."""
        retained = self.projector.validate_state(state)
        live = self.grid_filter.apply(retained)
        discretization = self.grid_filter.discretization
        gradient_modal = jnp.stack(
            tuple(discretization.modal_derivative(live, axis=axis) for axis in range(3)),
            axis=-1,
        )
        gradient = discretization.reconstruct(gradient_modal)
        features = gradient.reshape(discretization.physical_shape + (9,))
        strain = _deviatoric_strain(gradient)
        learned = self.binding(features, strain, args)
        modal_stress = self.grid_filter.apply(discretization.project(learned.stress))
        unprojected_rate, projected_rate = _periodic_fourier_stress_rate(
            modal_stress, self.grid_filter, self.projector
        )
        weights = discretization.quadrature_weights
        integrated_transfer = jnp.sum(weights * learned.local_transfer)
        unprojected_work = jnp.real(jnp.vdot(live, unprojected_rate))
        projected_work = jnp.real(jnp.vdot(live, projected_rate))
        energy_identity_defect = projected_work + integrated_transfer
        projection_work_defect = projected_work - unprojected_work
        zero_mode = self.projector.wavenumber_squared == 0.0
        momentum_defect = jnp.linalg.norm(
            jnp.where(zero_mode[..., None], unprojected_rate, 0.0)
        )
        divergence_norm = self.projector.divergence_norm(projected_rate)
        imaginary_leakage = discretization.imaginary_leakage(projected_rate)
        tolerance = jnp.asarray(
            self.plan.energy_tolerance, dtype=integrated_transfer.dtype
        )
        energy_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=integrated_transfer.dtype),
            jnp.maximum(jnp.abs(projected_work), jnp.abs(integrated_transfer)),
        )
        energy_consistent = jnp.abs(energy_identity_defect) <= tolerance * energy_scale
        policy_satisfied = _policy_satisfied(learned, tolerance)
        stable = policy_satisfied & (integrated_transfer >= -tolerance * energy_scale)
        finite = (
            learned.evidence.valid
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(learned.stress))
            & jnp.all(jnp.isfinite(projected_rate))
            & jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            integrated_transfer,
                            unprojected_work,
                            projected_work,
                            energy_identity_defect,
                            projection_work_defect,
                            momentum_defect,
                            divergence_norm,
                            imaginary_leakage,
                        )
                    )
                )
            )
        )
        successful = finite & policy_satisfied & energy_consistent
        return PeriodicLearnedStressStage(
            velocity_gradient=gradient,
            strain=strain,
            features=features,
            learned_result=learned,
            modal_deviatoric_specific_stress=modal_stress,
            unprojected_rate=unprojected_rate,
            projected_rate=projected_rate,
            integrated_transfer=integrated_transfer,
            unprojected_work=unprojected_work,
            projected_work=projected_work,
            energy_identity_defect=energy_identity_defect,
            projection_work_defect=projection_work_defect,
            momentum_conservation_defect=momentum_defect,
            divergence_norm=divergence_norm,
            imaginary_leakage=imaginary_leakage,
            energy_policy_active=learned.evidence.correction_applied,
            energy_policy_satisfied=policy_satisfied,
            energy_consistent=energy_consistent,
            stable=stable,
            finite=finite,
            successful=successful,
            feature_schema_id=self.feature_schema_id,
            prepared_id=self.prepared_id,
        )

    def __call__(
        self,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> PeriodicLearnedStressStage:
        return self.evaluate(state, args)


class MACLearnedStressStage(StrictModule):
    """One bound learned stress and its conservative projected MAC rate."""

    velocity_gradient: Array
    strain: Array
    features: Array
    learned_result: LearnedStressResult
    unprojected_rate: FaceVelocity
    projected_rate: FaceVelocity
    projection: MACRateProjectionResult
    integrated_transfer: Array
    unprojected_work: Array
    projected_work: Array
    energy_identity_defect: Array
    projection_work_defect: Array
    momentum_conservation_defect: Array
    energy_policy_active: Array
    energy_policy_satisfied: Array
    energy_consistent: Array
    conservative: Array
    stable: Array
    finite: Array
    successful: Array
    feature_schema_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    boundary_stage_id: str = eqx.field(static=True)

    @property
    def physical_rate(self) -> FaceVelocity:
        """Return the projected face rate for closure-alternative consumers."""
        return self.projected_rate

    @property
    def integrated_work(self) -> Array:
        """Return work of the projected closure rate on the supplied velocity."""
        return self.projected_work


class MACLearnedStressPlan(StrictModule, NonTrainableState):
    """Bind learned specific stress to periodic-uniform MAC stress realization."""

    binding: PreparedLearnedStressBinding
    energy_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    feature_schema_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: PreparedLearnedStressBinding,
        /,
        *,
        energy_tolerance: float = 1.0e-9,
        conservation_tolerance: float = 1.0e-10,
    ):
        _validate_binding_abi(binding)
        energy = float(energy_tolerance)
        conservation = float(conservation_tolerance)
        if (
            not isfinite(energy)
            or energy < 0.0
            or not isfinite(conservation)
            or conservation < 0.0
        ):
            raise ValueError("Learned stress tolerances must be finite and nonnegative.")
        provenance = binding.plan.parameter_provenance
        if provenance.regime != _MAC_LES_REGIME:
            raise ValueError(
                "MAC learned stress requires the 'incompressible-unit-density' regime."
            )
        resolved_filter = binding.plan.resolved_filter
        if (
            resolved_filter.family != "implicit-grid-volume"
            or resolved_filter.topology != "tensor-product"
            or resolved_filter.boundary_class != "periodic"
            or resolved_filter.scale_rule != "volume-equivalent"
            or resolved_filter.commutation_status != "unmodeled"
            or resolved_filter.repeated_filter_semantics != "unmodeled"
        ):
            raise ValueError(
                "MAC learned stress requires the periodic tensor-product implicit "
                "grid-volume filter with unmodeled commutation and repetition."
            )
        self.binding = binding
        self.energy_tolerance = energy
        self.conservation_tolerance = conservation
        self.feature_schema_id = binding.plan.feature_schema.feature_schema_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-learned-stress-plan",
                "binding": binding.prepared_id,
                "feature_schema": self.feature_schema_id,
                "energy_tolerance": energy,
                "conservation_tolerance": conservation,
                "feature_order": list(LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS),
                "stress_sign": "rhs-minus-divergence-of-specific-sgs-stress",
            }
        )

    def prepare(
        self,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        /,
    ) -> PreparedMACLearnedStress:
        return PreparedMACLearnedStress(self, momentum, projection)


class PreparedMACLearnedStress(StrictModule, NonTrainableState):
    """Prepared learned-stress adapter for a uniform three-dimensional periodic MAC grid."""

    plan: MACLearnedStressPlan
    binding: PreparedLearnedStressBinding
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    feature_schema_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MACLearnedStressPlan,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        /,
    ):
        from ..solver._structured_incompressible import MACPressureProjectionPlan

        if not isinstance(plan, MACLearnedStressPlan):
            raise TypeError("plan must be a MACLearnedStressPlan.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        if not isinstance(projection, MACPressureProjectionPlan):
            raise TypeError("projection must be a MACPressureProjectionPlan.")
        if momentum.dimension != 3:
            raise ValueError("MAC learned stress requires a three-dimensional grid.")
        discretization = momentum.operators.discretization
        axes = discretization.grid.structured_axes
        periodic_uniform = all(
            axis.periodic
            and np.allclose(
                np.asarray(axis.interval_widths),
                float(np.asarray(axis.interval_widths)[0]),
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            for axis in axes
        )
        if not periodic_uniform:
            raise ValueError(
                "MAC learned stress supports only periodic-uniform tensor grids."
            )
        resolved_filter = plan.binding.plan.resolved_filter
        if resolved_filter.axis_names != discretization.grid.axis_names:
            raise ValueError("Learned stress filter axes do not match the MAC grid.")
        provenance = plan.binding.plan.parameter_provenance
        if provenance.discretization_id != discretization.prepared_id:
            raise ValueError(
                "Learned stress provenance does not match the MAC discretization."
            )
        if projection.operators.prepared_id != momentum.operators.prepared_id:
            raise ValueError("Learned stress projection and MAC momentum disagree.")
        if projection.boundaries.prepared_id != momentum.boundaries.prepared_id:
            raise ValueError("Learned stress projection and MAC boundaries disagree.")
        if projection.density != 1.0:
            raise ValueError("MAC learned stress projection requires unit density.")
        if projection.transform_plan is None:
            raise ValueError(
                "Periodic-uniform MAC learned stress requires a certified transform "
                "pressure-projection route."
            )
        _validate_bound_layout(
            plan.binding,
            discretization.cell_shape,
            momentum.operators.pressure_space.dtype,
        )
        self.plan = plan
        self.binding = plan.binding
        self.momentum = momentum
        self.projection = projection
        self.feature_schema_id = plan.feature_schema_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-learned-stress",
                "plan": plan.plan_id,
                "binding": plan.binding.prepared_id,
                "feature_schema": plan.feature_schema_id,
                "momentum": momentum.prepared_id,
                "projection": projection.plan_id,
                "discretization": discretization.prepared_id,
                "runtime_scope": "single-device-3d-unit-density-periodic-uniform-mac",
            }
        )

    def _momentum_conservation_defect(self, rate: FaceVelocity, /) -> Array:
        space = self.momentum.operators.velocity_space
        defects = []
        for component in range(3):
            constant = tuple(
                jnp.ones_like(value) if axis == component else jnp.zeros_like(value)
                for axis, value in enumerate(rate)
            )
            defects.append(jnp.abs(jnp.real(space.inner(constant, rate))))
        return jnp.stack(tuple(defects))

    def evaluate(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        args: Any = None,
        /,
    ) -> MACLearnedStressStage:
        """Build features, evaluate stress, and own conservative rate projection."""
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        values = self.momentum.boundaries.enforce(
            self.momentum.operators.validate_velocity(velocity), stage
        )
        gradient = _mac_velocity_gradient(self.momentum, values)
        cell_shape = self.momentum.operators.discretization.cell_shape
        features = gradient.reshape(cell_shape + (9,))
        strain = _deviatoric_strain(gradient)
        learned = self.binding(features, strain, args)
        unprojected_rate = _periodic_uniform_mac_stress_rate(
            self.momentum, learned.stress
        )
        projected = self.projection.project_rate(unprojected_rate, boundary_stage=stage)
        space = self.momentum.operators.velocity_space
        volumes = self.momentum.operators.discretization.cell_volumes.astype(
            learned.local_transfer.dtype
        )
        integrated_transfer = jnp.sum(volumes * learned.local_transfer)
        unprojected_work = jnp.real(space.inner(values, unprojected_rate))
        projected_work = jnp.real(space.inner(values, projected.rate))
        energy_identity_defect = unprojected_work + integrated_transfer
        projection_work_defect = projected_work - unprojected_work
        momentum_defect = self._momentum_conservation_defect(unprojected_rate)
        tolerance = jnp.asarray(
            self.plan.energy_tolerance, dtype=integrated_transfer.dtype
        )
        energy_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=integrated_transfer.dtype),
            jnp.maximum(jnp.abs(unprojected_work), jnp.abs(integrated_transfer)),
        )
        energy_consistent = jnp.abs(energy_identity_defect) <= tolerance * energy_scale
        policy_satisfied = _policy_satisfied(learned, tolerance)
        stable = policy_satisfied & (integrated_transfer >= -tolerance * energy_scale)
        conservative = jnp.all(
            momentum_defect
            <= jnp.asarray(self.plan.conservation_tolerance, dtype=momentum_defect.dtype)
        )
        rate_finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(component))
                    for block in (unprojected_rate, projected.rate)
                    for component in block
                )
            )
        )
        scalar_finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        jnp.stack(
                            (
                                integrated_transfer,
                                unprojected_work,
                                projected_work,
                                energy_identity_defect,
                                projection_work_defect,
                            )
                        ),
                        momentum_defect,
                    )
                )
            )
        )
        finite = (
            stage.finite
            & learned.evidence.valid
            & projected.finite
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(learned.stress))
            & rate_finite
            & scalar_finite
        )
        successful = (
            stage.successful
            & projected.converged
            & finite
            & policy_satisfied
            & energy_consistent
            & conservative
        )
        return MACLearnedStressStage(
            velocity_gradient=gradient,
            strain=strain,
            features=features,
            learned_result=learned,
            unprojected_rate=unprojected_rate,
            projected_rate=projected.rate,
            projection=projected,
            integrated_transfer=integrated_transfer,
            unprojected_work=unprojected_work,
            projected_work=projected_work,
            energy_identity_defect=energy_identity_defect,
            projection_work_defect=projection_work_defect,
            momentum_conservation_defect=momentum_defect,
            energy_policy_active=learned.evidence.correction_applied,
            energy_policy_satisfied=policy_satisfied,
            energy_consistent=energy_consistent,
            conservative=conservative,
            stable=stable,
            finite=finite,
            successful=successful,
            feature_schema_id=self.feature_schema_id,
            prepared_id=self.prepared_id,
            boundary_stage_id=stage.stage_id,
        )

    def __call__(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        args: Any = None,
        /,
    ) -> MACLearnedStressStage:
        return self.evaluate(velocity, boundary_stage, args)


__all__ = [
    "LEARNED_STRESS_FEATURE_NAME",
    "LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS",
    "LEARNED_STRESS_VELOCITY_GRADIENT_UNITS",
    "MACLearnedStressPlan",
    "MACLearnedStressStage",
    "PeriodicLearnedStressPlan",
    "PeriodicLearnedStressStage",
    "PreparedMACLearnedStress",
    "PreparedPeriodicLearnedStress",
]
