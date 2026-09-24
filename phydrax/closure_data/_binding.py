#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._admissibility import AdmissibilityHeader, guard_derivative_validity
from .._differentiation import (
    branch_policy_contract,
    BranchDifferentiationPolicy,
    DerivativeContract,
    DerivativeSurface,
)
from .._fingerprint import canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._model import ValuePort
from .._model._ports import (
    bind_model_ports,
    intrinsic_model_ports,
    ModelPorts,
    PortBindingEvidence,
    PortMapping,
    require_mapped_order,
)
from .._strict import StrictModule
from .._trainable import ExplicitFreeze, NonTrainableState
from ..discretization.finite_volume._closure import (
    ArbitraryNormalFaceClosurePlan,
    FaceFluxContext,
)
from ..discretization.spectral._coordinates import HermitianSpectralCoordinates
from ..discretization.spectral._dealias import (
    OversamplingDealiasingPlan,
    PreparedDealiasingPlan,
)
from ..discretization.spectral._incompressible import PeriodicLerayProjector
from ..equations._les_closures import LESParameterProvenance, ResolvedLESFilter
from ._dataset import TrainOnlyNormalizer
from ._les import LESStressConvention
from ._state import _declared_dimension, FlowStateSchema


ClosureDeploymentKind = Literal["conservative_face", "spectral_drift"]
SpectralEnergyPolicy = Literal["nonincreasing", "diagnostic"]
StressEnergyPolicy = Literal["signed", "dissipative", "bounded_backscatter"]


def conservative_face_numeric_revision(
    binding: LearnedClosureBindingPlan, /
) -> NumericRevision:
    """Numeric revision of a conservative-face predictor under its binding identity.

    The revision binds the predictor's inexact array leaves to the binding
    semantics (`binding_id`: deployment kind, schema, components, model artifact,
    normalizer provenance, differentiability).
    """
    return NumericRevision(
        binding.binding_id,
        {
            "predictor": tuple(
                jax.tree_util.tree_leaves(
                    eqx.filter(binding.predictor, eqx.is_inexact_array)
                )
            )
        },
    )


def _conservative_face_correction(
    binding: LearnedClosureBindingPlan,
    system: Any,
    left: Array,
    right: Array,
    baseline_normal_flux: Array,
    context: FaceFluxContext,
    args: Any,
    /,
) -> Array:
    if binding.deployment_kind != "conservative_face":
        raise ValueError("Only conservative_face bindings evaluate face corrections.")
    return binding.predictor(system, left, right, baseline_normal_flux, context, args)


def _bind_conservative_faces(
    binding: LearnedClosureBindingPlan,
    schema: FlowStateSchema,
    numeric_revision: NumericRevision,
    consistency_tolerance: float,
    /,
) -> ArbitraryNormalFaceClosurePlan:
    binding._validate_schema(schema)
    if binding.deployment_kind != "conservative_face":
        raise ValueError("Only conservative_face bindings can enter a face flux plan.")
    if binding.output_component_names != schema.component_names:
        raise ValueError(
            "A conservative face closure must output every conservative component in schema order."
        )
    if not isinstance(numeric_revision, NumericRevision):
        raise TypeError("numeric_revision must be a NumericRevision.")
    expected = conservative_face_numeric_revision(binding)
    if (
        numeric_revision.semantic_id != expected.semantic_id
        or numeric_revision.revision_id != expected.revision_id
    ):
        raise ValueError(
            "Numeric revision does not match the bound conservative-face predictor."
        )
    # The binding itself is the correction child: its role marker governs the
    # predictor arrays inside the finite-volume tree, and its static identity is
    # the closure identity, so weights never change the method identity.
    return ArbitraryNormalFaceClosurePlan(
        binding,
        closure_id=binding.binding_id,
        consistency_tolerance=consistency_tolerance,
        differentiability=binding.differentiability,
    )


class LearnedClosureBindingPlan(StrictModule, NonTrainableState):
    """Artifact- and schema-bound learned predictor with an explicit deployment ABI."""

    predictor: Callable = eqx.field(static=True)
    deployment_kind: ClosureDeploymentKind = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    input_component_names: tuple[str, ...] = eqx.field(static=True)
    output_component_names: tuple[str, ...] = eqx.field(static=True)
    model_artifact_id: str = eqx.field(static=True)
    normalizer_provenance_id: str = eqx.field(static=True)
    differentiability: BranchDifferentiationPolicy = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        predictor: Callable,
        /,
        *,
        deployment_kind: ClosureDeploymentKind,
        schema_id: str,
        input_component_names: tuple[str, ...],
        output_component_names: tuple[str, ...],
        model_artifact_id: str,
        normalizer_provenance_id: str,
        differentiability: BranchDifferentiationPolicy = (
            BranchDifferentiationPolicy.SMOOTH
        ),
    ):
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        # Face corrections and modal drifts are structured callables over native
        # solver values; they expose no owner value ports a model could bind to.
        if intrinsic_model_ports(predictor) is not None:
            raise ValueError(
                "LearnedClosureBindingPlan deployment ABIs declare no owner value "
                f"ports; {type(predictor).__name__} declares model ports and cannot "
                "be bound as a face-correction or spectral-drift predictor."
            )
        if not isinstance(differentiability, BranchDifferentiationPolicy):
            raise TypeError("differentiability must be a BranchDifferentiationPolicy.")
        match differentiability:
            case (
                BranchDifferentiationPolicy.SMOOTH
                | BranchDifferentiationPolicy.BRANCHWISE
                | BranchDifferentiationPolicy.SMOOTH_SURROGATE
            ):
                pass
            case _:
                raise ValueError(
                    "LearnedClosureBindingPlan supports SMOOTH, BRANCHWISE, or "
                    f"SMOOTH_SURROGATE; got {differentiability.name}."
                )
        deployment = str(deployment_kind).strip()
        schema = str(schema_id).strip()
        inputs = tuple(str(value).strip() for value in input_component_names)
        outputs = tuple(str(value).strip() for value in output_component_names)
        artifact = str(model_artifact_id).strip()
        normalizer = str(normalizer_provenance_id).strip()
        if (
            deployment not in ("conservative_face", "spectral_drift")
            or not schema
            or not inputs
            or not outputs
            or any(not value for value in (*inputs, *outputs))
            or len(set(inputs)) != len(inputs)
            or len(set(outputs)) != len(outputs)
            or not artifact
            or not normalizer
        ):
            raise ValueError("Learned closure binding metadata is invalid.")
        self.predictor = predictor
        self.deployment_kind = deployment
        self.schema_id = schema
        self.input_component_names = inputs
        self.output_component_names = outputs
        self.model_artifact_id = artifact
        self.normalizer_provenance_id = normalizer
        self.differentiability = differentiability
        self.binding_id = canonical_fingerprint(
            {
                "kind": "learned-closure-binding-plan",
                "deployment_kind": deployment,
                "schema": schema,
                "input_components": list(inputs),
                "output_components": list(outputs),
                "model_artifact": artifact,
                "normalizer_provenance": normalizer,
                "differentiability": differentiability.value,
            }
        )

    def __call__(
        self,
        system: Any,
        left: Array,
        right: Array,
        baseline_normal_flux: Array,
        context: FaceFluxContext,
        args: Any = None,
        /,
    ) -> Array:
        """Evaluate the conservative-face correction ABI of the bound predictor."""
        return _conservative_face_correction(
            self, system, left, right, baseline_normal_flux, context, args
        )

    def bind_conservative_faces(
        self,
        schema: FlowStateSchema,
        numeric_revision: NumericRevision,
        /,
        *,
        consistency_tolerance: float = 1e-10,
    ) -> ArbitraryNormalFaceClosurePlan:
        """Bind the verified predictor revision as an arbitrary-normal face closure.

        `numeric_revision` must equal `conservative_face_numeric_revision(self)`,
        the revision recorded with the model artifact.
        """
        return _bind_conservative_faces(
            self, schema, numeric_revision, consistency_tolerance
        )

    def bind_spectral_drift(
        self,
        schema: FlowStateSchema,
        projector: PeriodicLerayProjector,
        hermitian_coordinates: HermitianSpectralCoordinates,
        dealiasing: PreparedDealiasingPlan,
        /,
        *,
        energy_policy: SpectralEnergyPolicy = "nonincreasing",
        evidence_tolerance: float = 1e-10,
    ) -> PreparedSpectralDriftHook:
        self._validate_schema(schema)
        if self.deployment_kind != "spectral_drift":
            raise ValueError("Only spectral_drift bindings can enter a spectral hook.")
        if self.output_component_names != schema.velocity_names:
            raise ValueError(
                "A spectral drift binding must output the declared velocity components in order."
            )
        return PreparedSpectralDriftHook(
            self.predictor,
            projector,
            hermitian_coordinates,
            dealiasing,
            binding_id=self.binding_id,
            energy_policy=energy_policy,
            evidence_tolerance=evidence_tolerance,
        )

    def _validate_schema(self, schema: FlowStateSchema) -> None:
        if not isinstance(schema, FlowStateSchema):
            raise TypeError("schema must be a FlowStateSchema.")
        if schema.schema_id != self.schema_id:
            raise ValueError(
                "Learned closure binding schema identity does not match deployment."
            )
        if any(
            value not in schema.component_names for value in self.input_component_names
        ):
            raise ValueError(
                "Learned closure input components are absent from the schema."
            )
        if any(
            value not in schema.component_names for value in self.output_component_names
        ):
            raise ValueError(
                "Learned closure output components are absent from the schema."
            )


class LearnedStressFeatureSchema(StrictModule, NonTrainableState):
    """Exact feature layout and physical metadata consumed by one stress model."""

    name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_units: tuple[str, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    flow_schema_id: str = eqx.field(static=True)
    feature_schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        name: str,
        component_names: tuple[str, ...],
        component_units: tuple[str, ...],
        shape: tuple[int, ...],
        dtype: Any,
        flow_schema_id: str,
    ):
        name_ = str(name).strip()
        names = tuple(str(value).strip() for value in component_names)
        units = tuple(str(value).strip() for value in component_units)
        shape_ = tuple(shape)
        dtype_ = np.dtype(dtype)
        schema = str(flow_schema_id).strip()
        if (
            not name_
            or not names
            or len(names) != len(units)
            or any(not value for value in (*names, *units))
            or len(set(names)) != len(names)
            or not shape_
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape_
            )
            or shape_[-1] != len(names)
            or not jnp.issubdtype(dtype_, jnp.floating)
            or not schema
        ):
            raise ValueError("Learned stress feature schema is invalid.")
        self.name = name_
        self.component_names = names
        self.component_units = units
        self.shape = shape_
        self.dtype = dtype_.name
        self.flow_schema_id = schema
        self.feature_schema_id = canonical_fingerprint(
            {
                "kind": "learned-stress-feature-schema",
                "name": name_,
                "component_names": list(names),
                "component_units": list(units),
                "shape": list(shape_),
                "dtype": dtype_.name,
                "flow_schema": schema,
            }
        )

    def value_port(self) -> ValuePort:
        """Return the per-sample feature port in declared component order.

        `shape` is the declared sample layout followed by the component axis, so
        the event is the trailing axis: `event_shape` is `(shape[-1],)` and the
        leading `shape[:-1]` axes are sample axes, not event axes.
        `semantic_id` is the declared feature `name`, `component_ids` are
        `component_names`, and each dimension is
        `phydrax.units.parse_unit(unit).dimension` of the declared component
        unit. The representation is the fixed literal
        `"learned-stress-features"`; space, frame, normalization, and event axes
        are undeclared; variance is neutral. `dtype` and `flow_schema_id` stay
        enforced by the binding plan through `feature_schema_id`.

        Raises `ValueError` for a component unit that `parse_unit` cannot
        resolve.
        """
        return ValuePort(
            self.name,
            event_shape=(self.shape[-1],),
            component_ids=self.component_names,
            representation="learned-stress-features",
            dimensions=tuple(
                _declared_dimension(
                    unit,
                    f"Learned stress feature schema {self.name!r} component {name!r}",
                )
                for name, unit in zip(
                    self.component_names, self.component_units, strict=True
                )
            ),
        )


class LearnedStressOutputContract(StrictModule, NonTrainableState):
    """Validated ABI for constant-density specific deviatoric SGS stress."""

    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    units: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    regime: str = eqx.field(static=True)
    stress_convention: LESStressConvention = eqx.field(static=True)
    density_semantics: str = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    trace_tolerance: float = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        units: str,
        target_id: str,
        filter_id: str,
        discretization_id: str,
        regime: str,
        stress_convention: LESStressConvention = "deviatoric",
        symmetry_tolerance: float = 1e-6,
        trace_tolerance: float = 1e-6,
    ):
        shape_ = tuple(shape)
        dtype_ = np.dtype(dtype)
        values = tuple(
            str(value).strip()
            for value in (units, target_id, filter_id, discretization_id, regime)
        )
        convention = str(stress_convention).strip()
        symmetry = float(symmetry_tolerance)
        trace = float(trace_tolerance)
        if (
            len(shape_) < 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape_
            )
            or shape_[-2:] != (3, 3)
            or not jnp.issubdtype(dtype_, jnp.floating)
            or any(not value for value in values)
            or convention != "deviatoric"
            or not np.isfinite(symmetry)
            or symmetry < 0.0
            or not np.isfinite(trace)
            or trace < 0.0
        ):
            raise ValueError(
                "Learned stress output must be a real, specific, deviatoric 3x3 tensor."
            )
        units_, target, filter_, discretization, regime_ = values
        self.shape = shape_
        self.dtype = dtype_.name
        self.units = units_
        self.target_id = target
        self.filter_id = filter_
        self.discretization_id = discretization
        self.regime = regime_
        self.stress_convention = "deviatoric"
        self.density_semantics = "constant-density-specific"
        self.symmetry_tolerance = symmetry
        self.trace_tolerance = trace
        self.contract_id = canonical_fingerprint(
            {
                "kind": "learned-stress-output-contract",
                "shape": list(shape_),
                "dtype": dtype_.name,
                "units": units_,
                "target": target,
                "filter": filter_,
                "discretization": discretization,
                "regime": regime_,
                "stress_convention": "deviatoric",
                "density_semantics": self.density_semantics,
                "symmetry_tolerance": symmetry,
                "trace_tolerance": trace,
            }
        )

    def value_port(self) -> ValuePort:
        """Return the per-sample 3x3 stress-tensor port.

        The declared `shape` ends with the `(3, 3)` tensor, which is the event
        shape; the leading `shape[:-2]` axes are sample axes. `semantic_id` is
        `target_id`. The contract names no tensor components, so
        `component_ids` are `f"{target_id}[{i}]"` for the row-major flat index
        `i`. Every component has `phydrax.units.parse_unit(units).dimension`.
        The representation is `f"{stress_convention}-{density_semantics}"`
        (`"deviatoric-constant-density-specific"`) and `space_id` is the
        declared `discretization_id`; frame, normalization, and event axes are
        undeclared, and variance is neutral. `dtype`, `filter_id`, `regime`,
        and the tolerances stay enforced by the binding plan through
        `contract_id`.

        Raises `ValueError` when `units` cannot be resolved by `parse_unit`.
        """
        dimension = _declared_dimension(
            self.units, f"Learned stress output contract {self.target_id!r}"
        )
        return ValuePort(
            self.target_id,
            event_shape=(3, 3),
            component_ids=tuple(f"{self.target_id}[{index}]" for index in range(9)),
            representation=f"{self.stress_convention}-{self.density_semantics}",
            space_id=self.discretization_id,
            dimensions=(dimension,) * 9,
        )


class LearnedStressBindingPlan(StrictModule, NonTrainableState):
    """Declarative artifact and LES-identity contract for a stress predictor."""

    feature_schema: LearnedStressFeatureSchema
    output_contract: LearnedStressOutputContract
    resolved_filter: ResolvedLESFilter
    parameter_provenance: LESParameterProvenance
    model_artifact_id: str = eqx.field(static=True)
    normalizer_id: str = eqx.field(static=True)
    energy_policy: StressEnergyPolicy = eqx.field(static=True)
    maximum_backscatter_fraction: float | None = eqx.field(static=True)
    differentiation_semantics: BranchDifferentiationPolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        feature_schema: LearnedStressFeatureSchema,
        output_contract: LearnedStressOutputContract,
        resolved_filter: ResolvedLESFilter,
        parameter_provenance: LESParameterProvenance,
        /,
        *,
        model_artifact_id: str,
        normalizer_id: str,
        energy_policy: StressEnergyPolicy = "signed",
        maximum_backscatter_fraction: float | None = None,
    ):
        if not isinstance(feature_schema, LearnedStressFeatureSchema):
            raise TypeError("feature_schema must be a LearnedStressFeatureSchema.")
        if not isinstance(output_contract, LearnedStressOutputContract):
            raise TypeError("output_contract must be a LearnedStressOutputContract.")
        if not isinstance(resolved_filter, ResolvedLESFilter):
            raise TypeError("resolved_filter must be a ResolvedLESFilter.")
        if not isinstance(parameter_provenance, LESParameterProvenance):
            raise TypeError("parameter_provenance must be LESParameterProvenance.")
        artifact = str(model_artifact_id).strip()
        normalizer = str(normalizer_id).strip()
        policy = str(energy_policy).strip()
        fraction = (
            None
            if maximum_backscatter_fraction is None
            else float(maximum_backscatter_fraction)
        )
        if not artifact or not normalizer:
            raise ValueError("Learned stress artifact identities must be non-empty.")
        if policy not in ("signed", "dissipative", "bounded_backscatter"):
            raise ValueError("Unsupported learned stress energy policy.")
        if policy == "bounded_backscatter":
            if (
                fraction is None
                or not np.isfinite(fraction)
                or not 0.0 <= fraction <= 1.0
            ):
                raise ValueError(
                    "Bounded backscatter requires a finite fraction in [0, 1]."
                )
        elif fraction is not None:
            raise ValueError(
                "maximum_backscatter_fraction is only valid for bounded_backscatter."
            )
        if output_contract.shape[:-2] != feature_schema.shape[:-1]:
            raise ValueError("Feature and stress layouts do not share one sample grid.")
        if output_contract.filter_id != resolved_filter.filter_id:
            raise ValueError("Stress target and deployment filter identities differ.")
        if (
            parameter_provenance.resolved_filter.filter_id != resolved_filter.filter_id
            or output_contract.discretization_id != parameter_provenance.discretization_id
            or output_contract.regime != parameter_provenance.regime
        ):
            raise ValueError(
                "Stress target and LES parameter provenance identities differ."
            )
        self.feature_schema = feature_schema
        self.output_contract = output_contract
        self.resolved_filter = resolved_filter
        self.parameter_provenance = parameter_provenance
        self.model_artifact_id = artifact
        self.normalizer_id = normalizer
        self.energy_policy = policy
        self.maximum_backscatter_fraction = fraction
        self.differentiation_semantics = (
            BranchDifferentiationPolicy.SMOOTH
            if policy == "signed"
            else BranchDifferentiationPolicy.BRANCHWISE
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "learned-stress-binding-plan",
                "feature_schema": feature_schema.feature_schema_id,
                "output_contract": output_contract.contract_id,
                "resolved_filter": resolved_filter.filter_id,
                "parameter_provenance": parameter_provenance.provenance_id,
                "model_artifact": artifact,
                "normalizer": normalizer,
                "energy_policy": policy,
                "maximum_backscatter_fraction": fraction,
                "differentiation_semantics": self.differentiation_semantics.value,
            }
        )

    def prepare(
        self,
        predictor: Callable,
        normalizer: TrainOnlyNormalizer,
        /,
        *,
        model_artifact_id: str,
        target_id: str,
        output_units: str,
        port_mapping: PortMapping | None = None,
    ) -> PreparedLearnedStressBinding:
        """Bind loaded runtime objects only when their artifact metadata is exact.

        `port_mapping` binds a port-declaring predictor to the plan's owner ports;
        see `PreparedLearnedStressBinding`.
        """

        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        if not isinstance(normalizer, TrainOnlyNormalizer):
            raise TypeError("normalizer must be a TrainOnlyNormalizer.")
        artifact = str(model_artifact_id).strip()
        target = str(target_id).strip()
        units = str(output_units).strip()
        if artifact != self.model_artifact_id:
            raise ValueError("Loaded predictor artifact does not match the binding plan.")
        if normalizer.normalizer_id != self.normalizer_id:
            raise ValueError("Loaded normalizer does not match the binding plan.")
        if (
            normalizer.provenance.schema_id != self.feature_schema.flow_schema_id
            or normalizer.provenance.feature_name != self.feature_schema.name
            or normalizer.mean.shape != (len(self.feature_schema.component_names),)
            or np.dtype(normalizer.mean.dtype).name != self.feature_schema.dtype
            or np.dtype(normalizer.scale.dtype).name != self.feature_schema.dtype
        ):
            raise ValueError(
                "Normalizer provenance or layout does not match the feature schema."
            )
        if target != self.output_contract.target_id:
            raise ValueError("Loaded predictor target does not match the binding plan.")
        if units != self.output_contract.units:
            raise ValueError("Loaded predictor output units do not match the contract.")
        return PreparedLearnedStressBinding(
            predictor, normalizer, self, port_mapping=port_mapping
        )


class LearnedStressEvidence(StrictModule, NonTrainableState):
    """Invariant and positive-forward transfer evidence for one evaluation.

    Bounded-backscatter totals are aggregated over the bound output layout; the
    policy deliberately does not claim pointwise dissipation.
    """

    raw_local_transfer: Array
    selected_local_transfer: Array
    raw_forward_transfer: Array
    raw_backscatter_transfer: Array
    selected_forward_transfer: Array
    selected_backscatter_transfer: Array
    backscatter_limit: Array
    correction_active: Array
    correction_applied: Array
    correction_norm: Array
    symmetry_defect: Array
    trace_defect: Array
    nonfinite_count: Array
    valid: Array
    binding_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    regime: str = eqx.field(static=True)
    parameter_provenance_id: str = eqx.field(static=True)
    energy_policy: StressEnergyPolicy = eqx.field(static=True)
    maximum_backscatter_fraction: float | None = eqx.field(static=True)
    differentiation_semantics: BranchDifferentiationPolicy = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        raw_local_transfer: ArrayLike,
        selected_local_transfer: ArrayLike,
        raw_forward_transfer: ArrayLike,
        raw_backscatter_transfer: ArrayLike,
        selected_forward_transfer: ArrayLike,
        selected_backscatter_transfer: ArrayLike,
        backscatter_limit: ArrayLike,
        correction_active: ArrayLike,
        correction_applied: ArrayLike,
        correction_norm: ArrayLike,
        symmetry_defect: ArrayLike,
        trace_defect: ArrayLike,
        nonfinite_count: ArrayLike,
        valid: ArrayLike,
        plan: LearnedStressBindingPlan,
    ):
        if not isinstance(plan, LearnedStressBindingPlan):
            raise TypeError("plan must be a LearnedStressBindingPlan.")
        self.raw_local_transfer = jnp.asarray(raw_local_transfer)
        self.selected_local_transfer = jnp.asarray(selected_local_transfer)
        self.raw_forward_transfer = jnp.asarray(raw_forward_transfer)
        self.raw_backscatter_transfer = jnp.asarray(raw_backscatter_transfer)
        self.selected_forward_transfer = jnp.asarray(selected_forward_transfer)
        self.selected_backscatter_transfer = jnp.asarray(selected_backscatter_transfer)
        self.backscatter_limit = jnp.asarray(backscatter_limit)
        self.correction_active = jnp.asarray(correction_active, dtype=jnp.bool_)
        self.correction_applied = jnp.asarray(correction_applied, dtype=jnp.bool_)
        self.correction_norm = jnp.asarray(correction_norm)
        self.symmetry_defect = jnp.asarray(symmetry_defect)
        self.trace_defect = jnp.asarray(trace_defect)
        self.nonfinite_count = jnp.asarray(nonfinite_count)
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.binding_id = plan.plan_id
        self.target_id = plan.output_contract.target_id
        self.filter_id = plan.resolved_filter.filter_id
        self.discretization_id = plan.parameter_provenance.discretization_id
        self.regime = plan.parameter_provenance.regime
        self.parameter_provenance_id = plan.parameter_provenance.provenance_id
        self.energy_policy = plan.energy_policy
        self.maximum_backscatter_fraction = plan.maximum_backscatter_fraction
        self.differentiation_semantics = plan.differentiation_semantics
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "learned-stress-evidence",
                "binding": self.binding_id,
                "target": self.target_id,
                "filter": self.filter_id,
                "discretization": self.discretization_id,
                "regime": self.regime,
                "parameter_provenance": self.parameter_provenance_id,
                "energy_policy": self.energy_policy,
                "maximum_backscatter_fraction": self.maximum_backscatter_fraction,
                "differentiation_semantics": self.differentiation_semantics.value,
            }
        )


class LearnedStressResult(StrictModule, NonTrainableState):
    """Specific deviatoric stress and Π = -τᵢⱼSᵢⱼ, positive for forward transfer.

    Beside the domain `evidence`, every sample lane carries the common
    `header` (margin: distance of the selected stress to the output contract's
    symmetry and trace tolerances; `model_id`: the bound predictor's component
    identity; `evidence_id`: the prepared binding identity), the canonical
    `derivative_contract` of the energy policy, and `derivative_valid`. Lanes
    whose selected energy branch sits on a branch boundary keep their primal
    stress and transfer but carry NaN derivatives.
    """

    stress: Array
    local_transfer: Array
    evidence: LearnedStressEvidence
    header: AdmissibilityHeader
    derivative_contract: DerivativeContract
    derivative_valid: Array

    def __init__(
        self,
        stress: ArrayLike,
        local_transfer: ArrayLike,
        evidence: LearnedStressEvidence,
        header: AdmissibilityHeader,
        derivative_contract: DerivativeContract,
        derivative_valid: ArrayLike,
        /,
    ):
        if not isinstance(evidence, LearnedStressEvidence):
            raise TypeError("evidence must be LearnedStressEvidence.")
        if not isinstance(header, AdmissibilityHeader):
            raise TypeError("header must be an AdmissibilityHeader.")
        if not isinstance(derivative_contract, DerivativeContract):
            raise TypeError("derivative_contract must be a DerivativeContract.")
        self.stress = jnp.asarray(stress)
        self.local_transfer = jnp.asarray(local_transfer)
        self.evidence = evidence
        self.header = header
        self.derivative_contract = derivative_contract
        self.derivative_valid = jnp.asarray(derivative_valid, dtype=jnp.bool_)


class PreparedLearnedStressBinding(StrictModule, ExplicitFreeze):
    """JIT-compatible stress evaluation without a backend divergence operator.

    The bound predictor is a loaded, artifact-identified deployment and is
    intentionally frozen (`ExplicitFreeze`); retraining binds a new predictor.

    The owner ports are the plan's feature port
    (`feature_schema.value_port()`), consumed as the predictor's single
    positional feature input, and its stress port (`output_contract.value_port()`),
    the single predictor output. A predictor declaring model ports requires an
    explicit `port_mapping` binding its ordered inputs and outputs to exactly
    those owner ports; `port_binding` holds the resulting `PortBindingEvidence`
    and is `None` for predictors without intrinsic ports, which take no mapping.
    """

    predictor: Callable
    normalizer: TrainOnlyNormalizer
    plan: LearnedStressBindingPlan
    port_binding: PortBindingEvidence | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        predictor: Callable,
        normalizer: TrainOnlyNormalizer,
        plan: LearnedStressBindingPlan,
        /,
        *,
        port_mapping: PortMapping | None = None,
    ):
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        if not isinstance(normalizer, TrainOnlyNormalizer):
            raise TypeError("normalizer must be a TrainOnlyNormalizer.")
        if not isinstance(plan, LearnedStressBindingPlan):
            raise TypeError("plan must be a LearnedStressBindingPlan.")
        site = "PreparedLearnedStressBinding"
        feature_port = plan.feature_schema.value_port()
        stress_port = plan.output_contract.value_port()
        port_binding = bind_model_ports(
            predictor,
            ModelPorts(inputs=(feature_port,), outputs=(stress_port,)),
            port_mapping,
            site=site,
        )
        if port_binding is not None:
            require_mapped_order(
                port_binding, "input", (feature_port.port_id,), site=site
            )
            require_mapped_order(
                port_binding, "output", (stress_port.port_id,), site=site
            )
        self.predictor = predictor
        self.normalizer = normalizer
        self.plan = plan
        self.port_binding = port_binding
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-learned-stress-binding",
                "plan": plan.plan_id,
                "normalizer": normalizer.normalizer_id,
            }
        )

    def apply(
        self,
        features: ArrayLike,
        strain: ArrayLike,
        args: Any = None,
        /,
    ) -> LearnedStressResult:
        """Predict stress and apply only the selected stress-energy policy."""

        feature_array = jnp.asarray(features)
        feature_schema = self.plan.feature_schema
        output_contract = self.plan.output_contract
        if feature_array.shape != feature_schema.shape:
            raise ValueError("Learned stress features do not match the bound shape.")
        if np.dtype(feature_array.dtype).name != feature_schema.dtype:
            raise TypeError("Learned stress features do not match the bound dtype.")
        feature_array = eqx.error_if(
            feature_array,
            jnp.any(~jnp.isfinite(feature_array)),
            "Learned stress features must be finite.",
        )
        strain_array = jnp.asarray(strain)
        if strain_array.shape != output_contract.shape:
            raise ValueError("Supplied strain does not match the bound stress shape.")
        if np.dtype(strain_array.dtype).name != output_contract.dtype:
            raise TypeError("Supplied strain does not match the bound stress dtype.")
        strain_array = eqx.error_if(
            strain_array,
            jnp.any(~jnp.isfinite(strain_array)),
            "Supplied strain must be finite.",
        )
        normalized = self.normalizer.normalize(feature_array)
        raw_stress = jnp.asarray(self.predictor(normalized, args))
        if raw_stress.shape != output_contract.shape:
            raise ValueError(
                "Learned stress predictor output does not match the bound shape."
            )
        if np.dtype(raw_stress.dtype).name != output_contract.dtype:
            raise TypeError(
                "Learned stress predictor output does not match the bound dtype."
            )
        nonfinite_count = jnp.sum(~jnp.isfinite(raw_stress))
        branch_tolerance = 32.0 * jnp.finfo(raw_stress.dtype).eps
        symmetry_defect = jnp.max(jnp.abs(raw_stress - jnp.swapaxes(raw_stress, -1, -2)))
        trace_defect = jnp.max(jnp.abs(jnp.trace(raw_stress, axis1=-2, axis2=-1)))
        raw_stress = eqx.error_if(
            raw_stress,
            nonfinite_count > 0,
            "Learned stress predictor output contains nonfinite values.",
        )
        raw_stress = eqx.error_if(
            raw_stress,
            symmetry_defect > output_contract.symmetry_tolerance,
            "Learned stress predictor output is not symmetric.",
        )
        raw_stress = eqx.error_if(
            raw_stress,
            trace_defect > output_contract.trace_tolerance,
            "Learned stress predictor output is not trace-free.",
        )
        strain_deviatoric = _symmetric_deviatoric(strain_array)
        raw_transfer = _stress_transfer(raw_stress, strain_deviatoric)
        selected_transfer = raw_transfer
        backscatter_limit = jnp.asarray(jnp.inf, dtype=raw_transfer.dtype)
        strain_norm_squared = ein.contract(
            "...ij,...ij->...", strain_deviatoric, strain_deviatoric, backend="jax"
        )
        # A lane's derivative is branch-local: invalid where the executed energy
        # branch sits on its boundary (zero local transfer, or the aggregate
        # backscatter cap switching on) relative to the lane's transfer scale.
        derivative_valid = jnp.ones(raw_transfer.shape, dtype=jnp.bool_)
        if self.plan.energy_policy != "signed":
            stress_norm_squared = ein.contract(
                "...ij,...ij->...", raw_stress, raw_stress, backend="jax"
            )
            derivative_valid = jnp.abs(raw_transfer) > branch_tolerance * jnp.sqrt(
                stress_norm_squared * strain_norm_squared
            )
        if self.plan.energy_policy == "dissipative":
            selected_transfer = jnp.maximum(raw_transfer, 0.0)
            backscatter_limit = jnp.asarray(0.0, dtype=raw_transfer.dtype)
        elif self.plan.energy_policy == "bounded_backscatter":
            raw_forward = jnp.sum(jnp.maximum(raw_transfer, 0.0))
            raw_backscatter = jnp.sum(jnp.maximum(-raw_transfer, 0.0))
            backscatter_limit = self.plan.maximum_backscatter_fraction * raw_forward
            safe_backscatter = jnp.where(raw_backscatter > 0.0, raw_backscatter, 1.0)
            backscatter_scale = jnp.minimum(1.0, backscatter_limit / safe_backscatter)
            selected_transfer = jnp.where(
                raw_transfer < 0.0,
                backscatter_scale * raw_transfer,
                raw_transfer,
            )
            cap_resolved = jnp.abs(
                backscatter_limit - raw_backscatter
            ) > branch_tolerance * jnp.maximum(backscatter_limit, raw_backscatter)
            derivative_valid = derivative_valid & ((raw_transfer > 0.0) | cap_resolved)
        transfer_delta = raw_transfer - selected_transfer
        safe_strain_norm = jnp.where(strain_norm_squared > 0.0, strain_norm_squared, 1.0)
        correction_coefficient = jnp.where(
            strain_norm_squared > 0.0,
            transfer_delta / safe_strain_norm,
            0.0,
        )
        correction = correction_coefficient[..., None, None] * strain_deviatoric
        selected_stress = raw_stress + correction
        selected_stress = eqx.error_if(
            selected_stress,
            jnp.any(~jnp.isfinite(selected_stress)),
            "Learned stress energy projection produced nonfinite values.",
        )
        local_transfer = _stress_transfer(selected_stress, strain_deviatoric)
        lane_symmetry_defect = jnp.max(
            jnp.abs(selected_stress - jnp.swapaxes(selected_stress, -1, -2)),
            axis=(-2, -1),
        )
        lane_trace_defect = jnp.abs(jnp.trace(selected_stress, axis1=-2, axis2=-1))
        selected_symmetry_defect = jnp.max(lane_symmetry_defect)
        selected_trace_defect = jnp.max(lane_trace_defect)
        selected_stress = eqx.error_if(
            selected_stress,
            (selected_symmetry_defect > output_contract.symmetry_tolerance)
            | (selected_trace_defect > output_contract.trace_tolerance),
            "Learned stress energy projection violated tensor invariants.",
        )
        correction_active = selected_transfer != raw_transfer
        evidence = LearnedStressEvidence(
            raw_local_transfer=raw_transfer,
            selected_local_transfer=local_transfer,
            raw_forward_transfer=jnp.sum(jnp.maximum(raw_transfer, 0.0)),
            raw_backscatter_transfer=jnp.sum(jnp.maximum(-raw_transfer, 0.0)),
            selected_forward_transfer=jnp.sum(jnp.maximum(local_transfer, 0.0)),
            selected_backscatter_transfer=jnp.sum(jnp.maximum(-local_transfer, 0.0)),
            backscatter_limit=backscatter_limit,
            correction_active=correction_active,
            correction_applied=jnp.any(correction_active),
            correction_norm=jnp.sqrt(jnp.sum(correction * correction)),
            symmetry_defect=selected_symmetry_defect,
            trace_defect=selected_trace_defect,
            nonfinite_count=nonfinite_count,
            valid=jnp.asarray(True),
            plan=self.plan,
        )
        header = AdmissibilityHeader(
            jnp.minimum(
                output_contract.symmetry_tolerance - lane_symmetry_defect,
                output_contract.trace_tolerance - lane_trace_defect,
            ),
            jnp.zeros(raw_transfer.shape, dtype=jnp.uint32),
            _stress_component_id(self.plan),
            self.prepared_id,
        )
        dependencies = (feature_array, strain_array)
        selected_stress = guard_derivative_validity(
            selected_stress,
            derivative_valid[..., None, None],
            dependencies=dependencies,
        )
        local_transfer = guard_derivative_validity(
            local_transfer, derivative_valid, dependencies=dependencies
        )
        return LearnedStressResult(
            selected_stress,
            local_transfer,
            evidence,
            header,
            branch_policy_contract(
                self.plan.differentiation_semantics,
                surfaces=(DerivativeSurface.INPUT, DerivativeSurface.PRIMAL_STATE),
            ),
            derivative_valid,
        )

    def __call__(
        self,
        features: ArrayLike,
        strain: ArrayLike,
        args: Any = None,
        /,
    ) -> LearnedStressResult:
        return self.apply(features, strain, args)


def _stress_component_id(plan: LearnedStressBindingPlan, /) -> str:
    """Declared artifact identity of the predictor bound by `plan`."""
    return SemanticProvenance(
        {
            "kind": "learned-stress-component",
            "features": plan.feature_schema.feature_schema_id,
            "target": plan.output_contract.target_id,
        },
        resource_ids={
            "model_artifact": plan.model_artifact_id,
            "normalizer": plan.normalizer_id,
        },
    ).semantic_id


def _symmetric_deviatoric(tensor: Array, /) -> Array:
    symmetric = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
    trace = jnp.trace(symmetric, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=tensor.dtype)
    return symmetric - (trace / 3.0)[..., None, None] * identity


def _stress_transfer(stress: Array, strain: Array, /) -> Array:
    return -ein.contract("...ij,...ij->...", stress, strain, backend="jax")


class SpectralDriftEvidence(StrictModule, NonTrainableState):
    raw_energy_rate: Array
    constrained_energy_rate: Array
    divergence_norm: Array
    hermitian_defect: Array
    nonfinite_count: Array
    valid: Array
    binding_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    hermitian_coordinate_id: str = eqx.field(static=True)
    dealiasing_id: str = eqx.field(static=True)
    dealiasing_kind: str = eqx.field(static=True)
    dealiasing_exact: bool = eqx.field(static=True)
    energy_policy: SpectralEnergyPolicy = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        raw_energy_rate: ArrayLike,
        constrained_energy_rate: ArrayLike,
        divergence_norm: ArrayLike,
        hermitian_defect: ArrayLike,
        nonfinite_count: ArrayLike,
        valid: ArrayLike,
        binding_id: str,
        projector_id: str,
        hermitian_coordinate_id: str,
        dealiasing_id: str,
        dealiasing_kind: str,
        dealiasing_exact: bool,
        energy_policy: SpectralEnergyPolicy,
    ):
        self.raw_energy_rate = jnp.asarray(raw_energy_rate)
        self.constrained_energy_rate = jnp.asarray(constrained_energy_rate)
        self.divergence_norm = jnp.asarray(divergence_norm)
        self.hermitian_defect = jnp.asarray(hermitian_defect)
        self.nonfinite_count = jnp.asarray(nonfinite_count)
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.binding_id = str(binding_id)
        self.projector_id = str(projector_id)
        self.hermitian_coordinate_id = str(hermitian_coordinate_id)
        self.dealiasing_id = str(dealiasing_id)
        self.dealiasing_kind = str(dealiasing_kind)
        self.dealiasing_exact = bool(dealiasing_exact)
        self.energy_policy = energy_policy
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "spectral-drift-evidence",
                "binding": self.binding_id,
                "projector": self.projector_id,
                "hermitian_coordinates": self.hermitian_coordinate_id,
                "dealiasing": self.dealiasing_id,
                "dealiasing_kind": self.dealiasing_kind,
                "dealiasing_exact": self.dealiasing_exact,
                "energy_policy": energy_policy,
            }
        )


class SpectralFallbackArtifact(StrictModule, NonTrainableState):
    """Typed record of the explicit zero-drift fallback selected by evidence."""

    used: Array
    reason_code: Array
    binding_id: str = eqx.field(static=True)
    fallback_kind: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(self, *, used: ArrayLike, reason_code: ArrayLike, binding_id: str):
        binding = str(binding_id).strip()
        if not binding:
            raise ValueError("binding_id must be non-empty.")
        self.used = jnp.asarray(used, dtype=jnp.bool_)
        self.reason_code = jnp.asarray(reason_code, dtype=jnp.int32)
        self.binding_id = binding
        self.fallback_kind = "zero_spectral_drift"
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "spectral-drift-fallback-artifact",
                "binding": binding,
                "fallback_kind": self.fallback_kind,
                "reason_codes": {
                    "0": "not_used",
                    "1": "nonfinite_prediction",
                    "2": "constraint_evidence_failed",
                    "3": "energy_contract_failed",
                },
            }
        )


class SpectralDriftResult(StrictModule, NonTrainableState):
    drift: Array
    evidence: SpectralDriftEvidence
    fallback: SpectralFallbackArtifact

    def __init__(
        self,
        drift: ArrayLike,
        evidence: SpectralDriftEvidence,
        fallback: SpectralFallbackArtifact,
        /,
    ):
        if not isinstance(evidence, SpectralDriftEvidence):
            raise TypeError("evidence must be SpectralDriftEvidence.")
        if not isinstance(fallback, SpectralFallbackArtifact):
            raise TypeError("fallback must be SpectralFallbackArtifact.")
        self.drift = jnp.asarray(drift)
        self.evidence = evidence
        self.fallback = fallback


class PreparedSpectralDriftHook(StrictModule, NonTrainableState):
    """Dealiased, solenoidal, Hermitian learned drift with energy evidence."""

    predictor: Callable = eqx.field(static=True)
    projector: PeriodicLerayProjector
    hermitian_coordinates: HermitianSpectralCoordinates
    dealiasing: PreparedDealiasingPlan
    binding_id: str = eqx.field(static=True)
    energy_policy: SpectralEnergyPolicy = eqx.field(static=True)
    evidence_tolerance: float = eqx.field(static=True)
    hook_id: str = eqx.field(static=True)

    def __init__(
        self,
        predictor: Callable,
        projector: PeriodicLerayProjector,
        hermitian_coordinates: HermitianSpectralCoordinates,
        dealiasing: PreparedDealiasingPlan,
        /,
        *,
        binding_id: str,
        energy_policy: SpectralEnergyPolicy,
        evidence_tolerance: float,
    ):
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be a PeriodicLerayProjector.")
        if not isinstance(hermitian_coordinates, HermitianSpectralCoordinates):
            raise TypeError("hermitian_coordinates must be HermitianSpectralCoordinates.")
        if not isinstance(dealiasing, PreparedDealiasingPlan):
            raise TypeError("dealiasing must be a PreparedDealiasingPlan.")
        if isinstance(dealiasing.plan, OversamplingDealiasingPlan):
            raise ValueError(
                "Oversampling dealiasing cannot serve as a learned spectral drift "
                "output filter because its filter action is identity."
            )
        binding = str(binding_id).strip()
        policy = str(energy_policy).strip()
        tolerance = float(evidence_tolerance)
        if (
            not binding
            or policy not in ("nonincreasing", "diagnostic")
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or projector.state_shape != hermitian_coordinates.state_shape
            or dealiasing.retained.prepared_id != projector.discretization.prepared_id
            or hermitian_coordinates.discretization.prepared_id
            != projector.discretization.prepared_id
        ):
            raise ValueError("Spectral drift hook contracts are incompatible.")
        self.predictor = predictor
        self.projector = projector
        self.hermitian_coordinates = hermitian_coordinates
        self.dealiasing = dealiasing
        self.binding_id = binding
        self.energy_policy = policy
        self.evidence_tolerance = tolerance
        self.hook_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-drift-hook",
                "binding": binding,
                "projector": projector.projector_id,
                "hermitian_coordinates": hermitian_coordinates.coordinate_id,
                "dealiasing": dealiasing.prepared_id,
                "energy_policy": policy,
                "evidence_tolerance": tolerance,
            }
        )

    def apply(self, state: ArrayLike, args: Any = None, /) -> SpectralDriftResult:
        value = self.projector.validate_state(state, owner="Spectral closure state")
        value = self.hermitian_coordinates.validate_state(value)
        input_divergence = self.projector.divergence_norm(value)
        input_hermitian = self.hermitian_coordinates.reality_defect(value)
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value))
            | (input_divergence > self.evidence_tolerance)
            | (input_hermitian > self.evidence_tolerance),
            "Spectral closure input violates finiteness, projection, or Hermitian contracts.",
        )
        raw = jnp.asarray(self.predictor(value, args))
        if raw.shape != value.shape:
            raise ValueError("Spectral closure predictor output must match state shape.")
        if not jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError(
                "Spectral closure predictor output must be complex modal data."
            )
        raw_nonfinite = jnp.sum(~jnp.isfinite(raw))
        raw_energy_rate = _energy_rate(value, raw)
        constrained = self.dealiasing.filter(raw)
        constrained = self.projector.project(constrained)
        constrained = self.hermitian_coordinates.project(constrained)
        constrained = self.projector.project(constrained)
        if self.energy_policy == "nonincreasing":
            rate = _energy_rate(value, constrained)
            state_norm_squared = jnp.real(jnp.vdot(value, value))
            safe_norm = jnp.where(state_norm_squared > 0.0, state_norm_squared, 1.0)
            multiplier = jnp.where(rate > 0.0, rate / safe_norm, 0.0)
            constrained = constrained - multiplier.astype(constrained.dtype) * value
            constrained = self.projector.project(
                self.hermitian_coordinates.project(constrained)
            )
        constrained_energy_rate = _energy_rate(value, constrained)
        divergence = self.projector.divergence_norm(constrained)
        hermitian = self.hermitian_coordinates.reality_defect(constrained)
        constraint_failed = (
            (divergence > self.evidence_tolerance)
            | (hermitian > self.evidence_tolerance)
            | jnp.any(~jnp.isfinite(constrained))
        )
        energy_failed = (self.energy_policy == "nonincreasing") & (
            constrained_energy_rate > self.evidence_tolerance
        )
        nonfinite_failed = raw_nonfinite > 0
        fallback_used = nonfinite_failed | constraint_failed | energy_failed
        reason_code = jnp.where(
            nonfinite_failed,
            1,
            jnp.where(constraint_failed, 2, jnp.where(energy_failed, 3, 0)),
        )
        drift = jnp.where(fallback_used, jnp.zeros_like(constrained), constrained)
        evidence = SpectralDriftEvidence(
            raw_energy_rate=raw_energy_rate,
            constrained_energy_rate=constrained_energy_rate,
            divergence_norm=divergence,
            hermitian_defect=hermitian,
            nonfinite_count=raw_nonfinite,
            valid=~fallback_used,
            binding_id=self.binding_id,
            projector_id=self.projector.projector_id,
            hermitian_coordinate_id=self.hermitian_coordinates.coordinate_id,
            dealiasing_id=self.dealiasing.prepared_id,
            dealiasing_kind=self.dealiasing.report.kind,
            dealiasing_exact=self.dealiasing.report.exact,
            energy_policy=self.energy_policy,
        )
        fallback = SpectralFallbackArtifact(
            used=fallback_used,
            reason_code=reason_code,
            binding_id=self.binding_id,
        )
        return SpectralDriftResult(drift, evidence, fallback)

    def __call__(self, state: ArrayLike, args: Any = None, /) -> Array:
        return self.apply(state, args).drift


def _energy_rate(state: Array, drift: Array) -> Array:
    return jnp.real(jnp.vdot(state, drift))


__all__ = [
    "ClosureDeploymentKind",
    "conservative_face_numeric_revision",
    "LearnedClosureBindingPlan",
    "LearnedStressBindingPlan",
    "LearnedStressEvidence",
    "LearnedStressFeatureSchema",
    "LearnedStressOutputContract",
    "LearnedStressResult",
    "PreparedLearnedStressBinding",
    "PreparedSpectralDriftHook",
    "SpectralDriftEvidence",
    "SpectralDriftResult",
    "SpectralEnergyPolicy",
    "SpectralFallbackArtifact",
    "StressEnergyPolicy",
]
