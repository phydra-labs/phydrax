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
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._dynamics import (
    ConvexStateLimiterPlan,
    FiniteVolumeMethodPlan,
    PreparedFiniteVolumeDynamics,
)
from ...discretization.finite_volume._high_resolution import (
    HighResolutionReconstructionPlan,
)
from ...discretization.finite_volume._mapped import MappedFiniteVolumeDiscretization
from ...discretization.finite_volume._positivity import FluxPositivityPlan
from ...discretization.finite_volume._riemann import (
    AbstractArbitraryNormalNumericalFluxPlan,
    AbstractSymmetricTwoPointFluxPlan,
)
from ...discretization.finite_volume._viscous import ViscousFluxPlan
from ...equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
    HomogeneousMixtureEulerSystem,
)
from ...equations.fem._conservation import (
    DGSEMConservationMethodPlan,
    DGSEMSampledFluxCompatibilityEvidence,
    PreparedDGSEMConservationDynamics,
)
from ...equations.fem._nodal_conservation import (
    NodalDGConservationMethodPlan,
    PreparedNodalDGConservationDynamics,
)
from ...equations.fem._viscous_conservation import ViscousDGPlan
from ...solver._conservation_temporal import ConservationIMEXMethod
from ...solver._finite_volume_runtime import (
    FiniteVolumeRuntimeState,
    FiniteVolumeStepPolicy,
    PreparedFiniteVolumeRuntime,
)
from ...solver._fixed_step import (
    AbstractFixedStepMethod,
    FixedStepResult,
    RobustRetryPolicy,
    SSPRK33FixedStepMethod,
    SSPRK54FixedStepMethod,
)
from ...solver._production_runtime import (
    ProductionCaseManifest,
    ProductionRunPlan,
)
from ._all_speed import ShockAwareAllSpeedFluxPlan
from ._contracts import (
    CompressibleFlowCaseSpec,
    CompressibleQualificationEvidence,
    ShockResolvingPolicy,
)


CompressibleTemporalMode: TypeAlias = Literal["explicit", "additive-imex"]


class CompressibleResourcePreflight(StrictModule, NonTrainableState):
    state_bytes: int = eqx.field(static=True)
    estimated_device_bytes: int = eqx.field(static=True)
    maximum_device_bytes: int = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    preflight_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: PyTree[Array],
        /,
        *,
        work_array_count: int,
        maximum_device_bytes: int,
    ):
        count = int(work_array_count)
        maximum = int(maximum_device_bytes)
        leaves = jax.tree.leaves(state)
        if (
            not leaves
            or any(not eqx.is_array(leaf) for leaf in leaves)
            or count <= 0
            or maximum <= 0
        ):
            raise ValueError("Compressible production resource preflight is invalid.")
        state_bytes = sum(int(leaf.size * leaf.dtype.itemsize) for leaf in leaves)
        estimated = state_bytes * count
        passed = estimated <= maximum
        self.state_bytes = state_bytes
        self.estimated_device_bytes = estimated
        self.maximum_device_bytes = maximum
        self.passed = passed
        self.preflight_id = canonical_fingerprint(
            {
                "kind": "compressible-production-resource-preflight",
                "state_bytes": state_bytes,
                "work_array_count": count,
                "estimated_device_bytes": estimated,
                "maximum_device_bytes": maximum,
                "passed": passed,
            }
        )

    def require_supported(self, /) -> None:
        if not self.passed:
            raise MemoryError(
                "Compressible production device-memory forecast exceeds its budget."
            )


class ExplicitCompressibleFixedStepAdapter(AbstractFixedStepMethod):
    """SSPRK fixed-step adapter for a prepared compressible spatial operator."""

    method: SSPRK33FixedStepMethod | SSPRK54FixedStepMethod
    spatial_operator_id: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        spatial_operator_id: str,
        /,
        *,
        order: int = 3,
    ):
        identifier = str(spatial_operator_id)
        order_ = int(order)
        if not callable(vector_field) or not identifier or order_ not in (3, 4):
            raise ValueError("Explicit compressible adapter inputs are invalid.")
        method = (
            SSPRK33FixedStepMethod(vector_field)
            if order_ == 3
            else SSPRK54FixedStepMethod(vector_field)
        )
        self.method = method
        self.spatial_operator_id = identifier
        self.order = order_
        self.method_id = canonical_fingerprint(
            {
                "kind": "explicit-compressible-fixed-step-adapter",
                "spatial_operator": identifier,
                "order": order_,
                "base_method": method.method_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        return self.method.step(step_index, time, state, step_size, args)


class AdditiveIMEXCompressibleFixedStepAdapter(AbstractFixedStepMethod):
    """Production fixed-step surface for a partitioned conservation IMEX method."""

    method: ConservationIMEXMethod
    explicit_operator_id: str = eqx.field(static=True)
    implicit_operator_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: ConservationIMEXMethod,
        /,
        *,
        explicit_operator_id: str,
        implicit_operator_id: str,
    ):
        explicit = str(explicit_operator_id)
        implicit = str(implicit_operator_id)
        if not isinstance(method, ConservationIMEXMethod) or not explicit or not implicit:
            raise ValueError("Additive IMEX compressible adapter inputs are invalid.")
        partition = canonical_fingerprint(
            {
                "kind": "compressible-additive-partition",
                "explicit": explicit,
                "implicit": implicit,
            }
        )
        self.method = method
        self.explicit_operator_id = explicit
        self.implicit_operator_id = implicit
        self.partition_id = partition
        self.method_id = canonical_fingerprint(
            {
                "kind": "additive-imex-compressible-fixed-step-adapter",
                "method": method.method_id,
                "partition": partition,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        if not eqx.is_array(state):
            raise TypeError("Conservation IMEX compressible state must be one array.")
        result = self.method.step(time, state, step_size, args)
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            result.maximum_implicit_residual,
            result.implicit_iterations,
            jnp.asarray(self.method.tableau.stage_count, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
        )


class FiniteVolumeRuntimeFixedStepAdapter(AbstractFixedStepMethod):
    """Fixed-step facade preserving FV positivity and fallback ledgers."""

    runtime: PreparedFiniteVolumeRuntime
    method_id: str = eqx.field(static=True)

    def __init__(self, runtime: PreparedFiniteVolumeRuntime, /):
        if not isinstance(runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("runtime must be PreparedFiniteVolumeRuntime.")
        self.runtime = runtime
        self.method_id = canonical_fingerprint(
            {
                "kind": "finite-volume-runtime-fixed-step-adapter",
                "runtime": runtime.runtime_id,
                "positivity": runtime.positivity.plan_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        if not isinstance(state, FiniteVolumeRuntimeState):
            raise TypeError("FV production state must be FiniteVolumeRuntimeState.")
        step_ = eqx.error_if(
            jnp.asarray(step_size),
            (jnp.asarray(step_index, dtype=jnp.int32) != state.accepted_step)
            | (jnp.asarray(time) != state.time),
            "FV production schedule and runtime state disagree.",
        )
        scheduled = self.runtime.advance_prescribed(state, step_, args)
        report = scheduled.attempted.positivity
        return FixedStepResult(
            scheduled.attempted.runtime_state,
            scheduled.runtime_state,
            scheduled.accepted,
            jnp.maximum(-scheduled.stability_margin, 0.0),
            scheduled.attempted.retries,
            jnp.asarray(3, dtype=jnp.int32),
            jnp.any(report.activated),
            jnp.max(1.0 - report.blend_factor),
        )


class CompressibleProductionRestart(StrictModule):
    accepted_state: PyTree[Array]
    step_index: Array
    time: Array
    method_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    restart_id: str = eqx.field(static=True)


class PreparedCompressibleProduction(StrictModule, NonTrainableState):
    """Route-bound fixed-step method with exact restart and runtime composition."""

    method: AbstractFixedStepMethod
    route_label: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    spatial_operator_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        route_label: str,
        spatial_operator_id: str,
        /,
    ):
        route = str(route_label)
        spatial = str(spatial_operator_id)
        if not isinstance(method, AbstractFixedStepMethod) or not route or not spatial:
            raise ValueError("Prepared compressible production inputs are invalid.")
        route_id = canonical_fingerprint(
            {"kind": "compressible-production-route", "label": route, "spatial": spatial}
        )
        self.method = method
        self.route_label = route
        self.route_id = route_id
        self.spatial_operator_id = spatial
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-compressible-production",
                "route": route_id,
                "method": method.method_id,
            }
        )

    def step(
        self,
        step_index: ArrayLike,
        time: ArrayLike,
        state: PyTree[Array],
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FixedStepResult:
        return self.method.step(
            jnp.asarray(step_index, dtype=jnp.int32),
            jnp.asarray(time),
            state,
            jnp.asarray(step_size),
            args,
        )

    def checkpoint(
        self,
        state: PyTree[Array],
        step_index: ArrayLike,
        time: ArrayLike,
        topology_id: str,
        /,
    ) -> CompressibleProductionRestart:
        topology = str(topology_id)
        leaves = jax.tree.leaves(state)
        if not topology or not leaves or any(not eqx.is_array(leaf) for leaf in leaves):
            raise ValueError("Compressible restart state or topology is invalid.")
        step = jnp.asarray(step_index, dtype=jnp.int32).reshape(())
        time_ = jnp.asarray(time).reshape(())
        restart_id = canonical_fingerprint(
            {
                "kind": "compressible-production-restart",
                "method": self.method.method_id,
                "route": self.route_id,
                "topology": topology,
                "step": int(np.asarray(step)),
                "time": float(np.asarray(time_)),
                "tree": tuple(
                    (tuple(leaf.shape), jnp.dtype(leaf.dtype).name) for leaf in leaves
                ),
                "state": array_tree_fingerprint(state),
            }
        )
        return CompressibleProductionRestart(
            state,
            step,
            time_,
            self.method.method_id,
            self.route_id,
            topology,
            restart_id,
        )

    def restore(
        self,
        restart: CompressibleProductionRestart,
        topology_id: str,
        /,
    ) -> tuple[PyTree[Array], Array, Array]:
        if not isinstance(restart, CompressibleProductionRestart):
            raise TypeError("restart must be CompressibleProductionRestart.")
        if (
            restart.method_id != self.method.method_id
            or restart.route_id != self.route_id
            or restart.topology_id != str(topology_id)
        ):
            raise ValueError(
                "Compressible restart identity does not match this production route."
            )
        return restart.accepted_state, restart.step_index, restart.time

    def production_run_plan(
        self,
        /,
        *,
        step_size: float,
        end_time: float,
        maximum_steps: int,
        checkpoint_interval: int,
        segment_steps: int = 32,
        retry_policy: RobustRetryPolicy | None = None,
    ) -> ProductionRunPlan:
        retry = RobustRetryPolicy() if retry_policy is None else retry_policy
        return ProductionRunPlan(
            self.method,
            retry,
            step_size=step_size,
            end_time=end_time,
            maximum_steps=maximum_steps,
            checkpoint_interval=checkpoint_interval,
            segment_steps=segment_steps,
        )

    def manifest(
        self,
        case: CompressibleFlowCaseSpec,
        /,
        *,
        precision_id: str,
        topology_id: str,
        geometry_layout_id: str,
        dtype: str,
    ) -> ProductionCaseManifest:
        if not isinstance(case, CompressibleFlowCaseSpec):
            raise TypeError("case must be CompressibleFlowCaseSpec.")
        return ProductionCaseManifest(
            problem_id=case.case_id,
            method_id=self.method.method_id,
            precision_id=precision_id,
            topology_id=topology_id,
            geometry_layout_id=geometry_layout_id,
            dtype=dtype,
        )


class SmoothCompressibleProductionPlan(StrictModule, NonTrainableState):
    """Tensor DGSEM enabled only by canonical sampled entropy evidence."""

    method: DGSEMConservationMethodPlan
    route_label: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_flux: AbstractSymmetricTwoPointFluxPlan,
        interface_flux: AbstractArbitraryNormalNumericalFluxPlan,
        /,
        *,
        compatibility: DGSEMSampledFluxCompatibilityEvidence | None = None,
        viscous: ViscousDGPlan | None = None,
        accumulation: str = "deterministic",
    ):
        if (
            not isinstance(compatibility, DGSEMSampledFluxCompatibilityEvidence)
            or compatibility.volume_flux_id != volume_flux.flux_id
            or compatibility.interface_flux_id != interface_flux.flux_id
            or not compatibility.volume_entropy_conservative
            or not compatibility.interface_entropy_stable
        ):
            raise TypeError(
                "Smooth tensor DGSEM requires passing system-specific sampled entropy evidence."
            )
        viscous_ = (
            ViscousDGPlan(formulation="entropy_br1") if viscous is None else viscous
        )
        if (
            not isinstance(viscous_, ViscousDGPlan)
            or viscous_.formulation != "entropy_br1"
        ):
            raise ValueError("Smooth tensor DGSEM requires entropy-BR1 viscosity.")
        method = DGSEMConservationMethodPlan(
            volume_flux,
            interface_flux,
            compatibility=compatibility,
            viscous=viscous_,
            accumulation=accumulation,
        )
        route = "smooth:tensor-dgsem:canonical-entropy-evidence:entropy-br1"
        self.method = method
        self.route_label = route
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-production-plan",
                "route": route,
                "method": method.method_id,
            }
        )

    def prepare_explicit(
        self,
        dynamics: PreparedDGSEMConservationDynamics,
        /,
        *,
        order: int = 3,
    ) -> PreparedCompressibleProduction:
        if not isinstance(dynamics, PreparedDGSEMConservationDynamics):
            raise TypeError("Smooth production requires prepared tensor DGSEM dynamics.")
        if dynamics.method.method_id != self.method.method_id:
            raise ValueError("Prepared DGSEM dynamics do not belong to this plan.")
        if (
            not isinstance(
                dynamics.system,
                (
                    HomogeneousMixtureEulerSystem,
                    HomogeneousMixtureCompressibleNavierStokesSystem,
                ),
            )
            or dynamics.system.system_id != self.method.compatibility.system_id
        ):
            raise ValueError(
                "Prepared DGSEM dynamics lack exact canonical entropy-system evidence."
            )
        temporal = ExplicitCompressibleFixedStepAdapter(
            dynamics, dynamics.dynamics_id, order=order
        )
        return PreparedCompressibleProduction(
            temporal, self.route_label, dynamics.dynamics_id
        )

    def prepare_imex(
        self,
        dynamics: PreparedDGSEMConservationDynamics,
        method: ConservationIMEXMethod,
        /,
        *,
        explicit_operator_id: str,
        implicit_operator_id: str,
    ) -> PreparedCompressibleProduction:
        if not isinstance(dynamics, PreparedDGSEMConservationDynamics):
            raise TypeError("Smooth production requires prepared tensor DGSEM dynamics.")
        if dynamics.method.method_id != self.method.method_id:
            raise ValueError("Prepared DGSEM dynamics do not belong to this plan.")
        if (
            not isinstance(
                dynamics.system,
                (
                    HomogeneousMixtureEulerSystem,
                    HomogeneousMixtureCompressibleNavierStokesSystem,
                ),
            )
            or dynamics.system.system_id != self.method.compatibility.system_id
        ):
            raise ValueError(
                "Prepared DGSEM dynamics lack exact canonical entropy-system evidence."
            )
        temporal = AdditiveIMEXCompressibleFixedStepAdapter(
            method,
            explicit_operator_id=explicit_operator_id,
            implicit_operator_id=implicit_operator_id,
        )
        return PreparedCompressibleProduction(
            temporal, self.route_label, dynamics.dynamics_id
        )

    def qualification_evidence(
        self,
        case: CompressibleFlowCaseSpec,
        /,
    ) -> CompressibleQualificationEvidence:
        if not isinstance(case, CompressibleFlowCaseSpec) or case.route != "tensor-dgsem":
            raise ValueError("Smooth qualification requires a tensor-DGSEM case.")
        compatibility = self.method.compatibility
        canonical_system = case.prepare_inviscid_system()
        if compatibility is None or compatibility.system_id != canonical_system.system_id:
            raise ValueError(
                "Smooth qualification requires entropy evidence for the exact canonical gas system."
            )
        return CompressibleQualificationEvidence(
            case.case_id,
            self.route_label,
            self.method.method_id,
            (
                ("tensor-dgsem", True),
                ("compressible-navier-stokes", case.equation == "navier_stokes"),
                (
                    "symmetric-consistent-volume-flux",
                    self.method.volume_flux.symmetric
                    and self.method.volume_flux.consistent,
                ),
                ("sampled-flux-compatibility", compatibility is not None),
                (
                    "volume-entropy-conservative",
                    compatibility is not None
                    and compatibility.volume_entropy_conservative,
                ),
                (
                    "interface-entropy-stable",
                    compatibility is not None and compatibility.interface_entropy_stable,
                ),
                (
                    "entropy-br1",
                    self.method.viscous is not None
                    and self.method.viscous.formulation == "entropy_br1",
                ),
            ),
        )


class NodalDGCompressibleProductionPlan(StrictModule, NonTrainableState):
    """Separate non-tensor nodal-DG route with LDG viscous traces."""

    method: NodalDGConservationMethodPlan
    route_label: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_flux: AbstractArbitraryNormalNumericalFluxPlan,
        /,
        *,
        viscous: ViscousDGPlan | None = None,
        accumulation: str = "deterministic",
    ):
        viscous_ = ViscousDGPlan(formulation="ldg") if viscous is None else viscous
        if not isinstance(viscous_, ViscousDGPlan) or viscous_.formulation != "ldg":
            raise ValueError("Non-tensor nodal DG requires its separate LDG route.")
        method = NodalDGConservationMethodPlan(
            interface_flux,
            viscous=viscous_,
            accumulation=accumulation,
        )
        route = "nodal-dg:overintegrated:ldg"
        self.method = method
        self.route_label = route
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nodal-dg-compressible-production-plan",
                "route": route,
                "method": method.method_id,
            }
        )

    def prepare_explicit(
        self,
        dynamics: PreparedNodalDGConservationDynamics,
        /,
        *,
        order: int = 3,
    ) -> PreparedCompressibleProduction:
        if not isinstance(dynamics, PreparedNodalDGConservationDynamics):
            raise TypeError("Nodal production requires prepared nodal-DG dynamics.")
        if not isinstance(
            dynamics.system,
            (
                HomogeneousMixtureEulerSystem,
                HomogeneousMixtureCompressibleNavierStokesSystem,
            ),
        ):
            raise TypeError(
                "Nodal production requires a canonical homogeneous-mixture system."
            )
        if dynamics.method.method_id != self.method.method_id:
            raise ValueError("Prepared nodal dynamics do not belong to this plan.")
        temporal = ExplicitCompressibleFixedStepAdapter(
            dynamics, dynamics.dynamics_id, order=order
        )
        return PreparedCompressibleProduction(
            temporal, self.route_label, dynamics.dynamics_id
        )


class StructuredFVCompressibleProductionPlan(StrictModule, NonTrainableState):
    """Structured/mapped high-resolution FV with generic canonical HLL fallback."""

    shock: ShockResolvingPolicy
    method: FiniteVolumeMethodPlan
    positivity: FluxPositivityPlan
    geometry_route: str = eqx.field(static=True)
    route_label: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry_route: Literal["structured", "mapped"] = "structured",
        /,
        *,
        shock: ShockResolvingPolicy | None = None,
        interface_solver: AbstractArbitraryNormalNumericalFluxPlan | None = None,
        viscous: ViscousFluxPlan | None = None,
        reconstruction_order: int = 5,
        positivity_iterations: int = 32,
    ):
        shock_ = ShockResolvingPolicy() if shock is None else shock
        if geometry_route not in ("structured", "mapped") or not isinstance(
            shock_, ShockResolvingPolicy
        ):
            raise ValueError("Structured FV compressible production route is invalid.")
        interface = (
            ShockAwareAllSpeedFluxPlan(shock_)
            if interface_solver is None
            else interface_solver
        )
        if (
            not isinstance(interface, ShockAwareAllSpeedFluxPlan)
            or interface.policy.policy_id != shock_.policy_id
        ):
            raise TypeError(
                "FV all-speed shock production requires a "
                "ShockAwareAllSpeedFluxPlan bound to the shock policy."
            )
        reconstruction = HighResolutionReconstructionPlan(
            shock_.reconstruction, order=reconstruction_order
        )
        method = FiniteVolumeMethodPlan(
            reconstruction,
            interface,
            positivity=ConvexStateLimiterPlan(positivity_iterations),
            viscous=viscous,
        )
        stage_positivity = FluxPositivityPlan(
            positivity_iterations, fallback_flux=shock_.fallback_flux
        )
        route = f"shock:{geometry_route}-fv:{shock_.reconstruction}:{type(interface).__name__}->generic-hll"
        self.geometry_route = geometry_route
        self.shock = shock_
        self.method = method
        self.positivity = stage_positivity
        self.route_label = route
        self.plan_id = canonical_fingerprint(
            {
                "kind": "structured-fv-compressible-production-plan",
                "geometry_route": geometry_route,
                "shock": shock_.policy_id,
                "method": method.method_id,
                "positivity": stage_positivity.plan_id,
                "route": route,
            }
        )

    def prepare_runtime(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        /,
        *,
        step_policy: FiniteVolumeStepPolicy | None = None,
    ) -> PreparedFiniteVolumeRuntime:
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("Structured FV production requires prepared FV dynamics.")
        if not isinstance(
            dynamics.system,
            (
                HomogeneousMixtureEulerSystem,
                HomogeneousMixtureCompressibleNavierStokesSystem,
            ),
        ):
            raise TypeError(
                "Structured FV production requires a canonical homogeneous-mixture system."
            )
        if dynamics.method.method_id != self.method.method_id:
            raise ValueError(
                "Prepared FV dynamics do not belong to this production plan."
            )
        mapped = isinstance(dynamics.discretization, MappedFiniteVolumeDiscretization)
        if mapped != (self.geometry_route == "mapped"):
            raise ValueError("Prepared FV geometry does not match the declared route.")
        return PreparedFiniteVolumeRuntime(
            dynamics,
            self.positivity,
            step_policy,
        )

    def prepare_explicit(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        /,
        *,
        step_policy: FiniteVolumeStepPolicy | None = None,
        order: int = 3,
    ) -> PreparedCompressibleProduction:
        if int(order) != 3:
            raise ValueError("Prepared FV runtime owns its SSPRK3 temporal scheme.")
        runtime = self.prepare_runtime(dynamics, step_policy=step_policy)
        temporal = FiniteVolumeRuntimeFixedStepAdapter(runtime)
        return PreparedCompressibleProduction(
            temporal, self.route_label, runtime.runtime_id
        )

    def qualification_evidence(
        self,
        case: CompressibleFlowCaseSpec,
        /,
    ) -> CompressibleQualificationEvidence:
        expected = "mapped-fv" if self.geometry_route == "mapped" else "structured-fv"
        if not isinstance(case, CompressibleFlowCaseSpec) or case.route != expected:
            raise ValueError("FV qualification case does not match the geometry route.")
        return CompressibleQualificationEvidence(
            case.case_id,
            self.route_label,
            self.method.method_id,
            (
                ("geometry-route-exact", True),
                (
                    "high-resolution-reconstruction",
                    self.method.reconstruction.method in ("weno_z", "teno", "mp5"),
                ),
                ("face-state-positivity", self.method.positivity is not None),
                (
                    "stage-flux-positivity",
                    isinstance(self.positivity, FluxPositivityPlan),
                ),
                (
                    "canonical-generic-hll-fallback",
                    self.positivity.fallback_flux.flux_id
                    == self.shock.fallback_flux.flux_id,
                ),
                ("shock-route-labeled", self.route_label.startswith("shock:")),
            ),
        )


__all__ = [
    "AdditiveIMEXCompressibleFixedStepAdapter",
    "CompressibleProductionRestart",
    "CompressibleResourcePreflight",
    "ExplicitCompressibleFixedStepAdapter",
    "FiniteVolumeRuntimeFixedStepAdapter",
    "NodalDGCompressibleProductionPlan",
    "PreparedCompressibleProduction",
    "SmoothCompressibleProductionPlan",
    "StructuredFVCompressibleProductionPlan",
]
