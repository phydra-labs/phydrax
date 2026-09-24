#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Intrinsic model execution contracts and explicit component binding.

A `ModelExecutionContract` describes what a model is: its derivative claims,
execution capabilities, precision, randomness, ports, construction certificates,
and semantic provenance. It never declares authority. Authority is conferred by
binding: an owner slot (`AbstractComponentSlot`) for inline components, or an
explicit `ComponentBinding` for a model held separately from the owner. The bound
`ComponentContract` records the slot identity, authority, port binding evidence,
admissibility requirements, and derivative admission of one binding.
"""

from collections.abc import Iterable
from math import isfinite
from numbers import Real
from typing import Any, ClassVar, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractClassVar

from .._differentiation import (
    _identifier,
    _identifier_set,
    AbstractConstructionCertificate,
    CapabilityEvidenceKind,
    CapabilityRequirement,
    ComponentAuthority,
    DerivativeAdmission,
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DifferentiationRequest,
    RegularityPolicy,
)
from .._fingerprint import canonical_fingerprint
from .._identity import SemanticProvenance
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._ports import ModelPorts, PortBindingEvidence, PortMapping, resolve_port_mapping


if TYPE_CHECKING:
    from ..domain._derivative import DerivativeMode
    from ._array import AbstractArrayModel


ExecutionTier: TypeAlias = Literal[
    "native-jax",
    "functional-jax",
    "converted-native",
    "compiled-inference",
    "host-inference",
    "external-adjoint",
]
RandomnessMode: TypeAlias = Literal["deterministic", "fixed-realization", "resampled"]
CertificateRecord: TypeAlias = tuple[str, str, CapabilityEvidenceKind]

_PYTHON_SCALAR_TYPES = (bool, int, float, complex)


def _flag(value: Any, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool.")
    return value


def _optional_type(value: Any, expected: type, name: str, /) -> Any:
    if value is not None and not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__} or None.")
    return value


@final
class ExecutionCapabilities(StrictModule, NonTrainableState):
    """Static execution capabilities of one model.

    `tier` names how the model executes: `"native-jax"` (a Phydrax JAX model),
    `"functional-jax"` (an external functional JAX adapter), `"converted-native"`
    (an external model converted into native JAX), `"compiled-inference"` (a
    precompiled executable), `"host-inference"` (a host runtime outside JAX), or
    `"external-adjoint"` (an external provider supplying adjoint actions).

    A `host_only` model runs outside JAX tracing, so it supports neither `jit`
    nor `vmap`; both default to `not host_only`, and requesting either for a
    host-only model is a `ValueError`. JAX tiers (`"native-jax"`,
    `"functional-jax"`, `"converted-native"`) are never host-only, and
    `"host-inference"` always is. `stateful` declares a `MODEL_STATE` lane.
    Derivative support is not an execution capability: use `supports_derivative`
    on the model's `DerivativeContract`.
    """

    tier: ExecutionTier = eqx.field(static=True)
    jit: bool = eqx.field(static=True)
    vmap: bool = eqx.field(static=True)
    host_only: bool = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)

    def __init__(
        self,
        tier: ExecutionTier,
        /,
        *,
        host_only: bool = False,
        jit: bool | None = None,
        vmap: bool | None = None,
        stateful: bool = False,
    ):
        host_only_ = _flag(host_only, "host_only")
        jit_ = not host_only_ if jit is None else _flag(jit, "jit")
        vmap_ = not host_only_ if vmap is None else _flag(vmap, "vmap")
        stateful_ = _flag(stateful, "stateful")
        if host_only_ and (jit_ or vmap_):
            raise ValueError("A host-only model supports neither jit nor vmap.")
        if not isinstance(tier, str):
            raise TypeError("tier must be a string.")
        match tier:
            case "native-jax" | "functional-jax" | "converted-native":
                if host_only_:
                    raise ValueError(f"Execution tier {tier!r} is never host-only.")
            case "host-inference":
                if not host_only_:
                    raise ValueError("Execution tier 'host-inference' is host-only.")
            case "compiled-inference" | "external-adjoint":
                pass
            case _:
                raise ValueError(f"Unknown execution tier {tier!r}.")
        self.tier = tier
        self.jit = jit_
        self.vmap = vmap_
        self.host_only = host_only_
        self.stateful = stateful_


def supports_derivative(
    contract: DerivativeContract,
    request: DifferentiationRequest,
    /,
    *,
    mode: "DerivativeMode",
    policy: RegularityPolicy | None = None,
) -> bool:
    """Return whether `request` may be formed in `mode` under `contract`.

    `mode` is `"forward"` (JVP) or `"reverse"` (VJP). Support is derived from the
    derivative contract alone: the admission of `request` (under `policy`) must be
    supported, and the contract route must provide the mode. JAX-transform
    routes (`DIRECT`, `IMPLICIT`, `UNROLLED`, `SPECTRAL`, `RELAXED`) provide both
    modes; `EXTERNAL_ADJOINT` provides only reverse mode, because an external
    provider supplies adjoint actions and no tangents; `STOPPED` provides none.
    """
    if not isinstance(contract, DerivativeContract):
        raise TypeError("contract must be a DerivativeContract.")
    if not isinstance(mode, str):
        raise TypeError("mode must be 'forward' or 'reverse'.")
    if mode not in ("forward", "reverse"):
        raise ValueError(f"Unknown derivative mode {mode!r}.")
    if not contract.admit(request, policy=policy).supported:
        return False
    match contract.route:
        case (
            DerivativeRoute.DIRECT
            | DerivativeRoute.IMPLICIT
            | DerivativeRoute.UNROLLED
            | DerivativeRoute.SPECTRAL
            | DerivativeRoute.RELAXED
        ):
            return True
        case DerivativeRoute.EXTERNAL_ADJOINT:
            return mode == "reverse"
        case DerivativeRoute.STOPPED:
            return False
        case _:
            raise ValueError(f"Unknown derivative route {contract.route!r}.")


@final
class RandomnessContract(StrictModule, NonTrainableState):
    """Declared randomness of one model's evaluation.

    `mode` is `"deterministic"` (no randomness), `"fixed-realization"` (random
    structure frozen to one realization, identified by `realization_id` once
    bound), or `"resampled"` (fresh randomness on every evaluation).
    `requires_inference_state` declares that the mode holds only in the model's
    inference state (dropout disabled, running statistics frozen), which the
    owner must bind. `realization_id` is declared only for fixed realizations.
    """

    mode: RandomnessMode = eqx.field(static=True)
    requires_inference_state: bool = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        mode: RandomnessMode,
        /,
        *,
        requires_inference_state: bool = False,
        realization_id: str | None = None,
    ):
        if not isinstance(mode, str):
            raise TypeError("mode must be a string.")
        match mode:
            case "deterministic" | "resampled":
                if realization_id is not None:
                    raise ValueError(
                        "realization_id is declared only for fixed-realization "
                        "randomness."
                    )
            case "fixed-realization":
                pass
            case _:
                raise ValueError(f"Unknown randomness mode {mode!r}.")
        requires_state = _flag(requires_inference_state, "requires_inference_state")
        realization = (
            None
            if realization_id is None
            else _identifier(realization_id, "realization_id")
        )
        self.mode = mode
        self.requires_inference_state = requires_state
        self.realization_id = realization


def admit_randomness(
    contract: RandomnessContract | None,
    /,
    *,
    implicit: bool,
    authoritative: bool,
    realization_bound: bool,
    inference_state_bound: bool,
) -> tuple[bool, str]:
    """Admit a component's declared randomness for one owner use.

    Returns `(admitted, reason)`; `reason` names the deciding rule and is
    recorded on admission as well as rejection:

    - undeclared randomness (`contract is None`) is rejected for implicit or
      authoritative use and admitted, recorded as `"randomness-undeclared"`,
      otherwise;
    - a required inference state must be bound (`"inference-state-unbound"`);
    - deterministic randomness is admitted;
    - a fixed realization needs the owner's realization binding
      (`"realization-unbound"`) and a declared `realization_id`
      (`"realization-unidentified"`);
    - resampled randomness is rejected for implicit or authoritative use.
    """
    implicit_ = _flag(implicit, "implicit")
    authoritative_ = _flag(authoritative, "authoritative")
    realization_bound_ = _flag(realization_bound, "realization_bound")
    inference_state_bound_ = _flag(inference_state_bound, "inference_state_bound")
    strict = implicit_ or authoritative_
    if contract is None:
        return not strict, "randomness-undeclared"
    if not isinstance(contract, RandomnessContract):
        raise TypeError("contract must be a RandomnessContract or None.")
    if contract.requires_inference_state and not inference_state_bound_:
        return False, "inference-state-unbound"
    match contract.mode:
        case "deterministic":
            return True, "deterministic"
        case "fixed-realization":
            if not realization_bound_:
                return False, "realization-unbound"
            if contract.realization_id is None:
                return False, "realization-unidentified"
            return True, "fixed-realization"
        case "resampled":
            if strict:
                return False, "resampled-randomness-not-admitted"
            return True, "resampled"
        case _:
            raise ValueError(f"Unknown randomness mode {contract.mode!r}.")


def _dtype_name(value: Any, name: str, /) -> str:
    if value is None or any(value is scalar for scalar in _PYTHON_SCALAR_TYPES):
        raise TypeError(f"{name} must be an explicit NumPy or JAX dtype.")
    dtype = np.dtype(value)
    if isinstance(value, str) and dtype.name != value:
        raise ValueError(f"{name} must name a canonical dtype, not {value!r}.")
    if not (jnp.issubdtype(dtype, jnp.number) or dtype == np.bool_):
        raise ValueError(f"{name} must be a numeric or boolean dtype, not {dtype}.")
    return dtype.name


def _error_floor(value: Any, name: str, /) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None.")
    floor = float(value)
    if not isfinite(floor) or floor < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return floor


@final
class ComponentPrecisionContract(StrictModule, NonTrainableState):
    """Declared numeric precision of one component's evaluation.

    Dtypes are stored as canonical dtype names. `absolute_error_floor` and
    `relative_error_floor` bound the evaluation error the component can resolve;
    `None` means undeclared, and machine epsilon of `compute_dtype` alone is never
    substituted for an undeclared floor. `amplification_evidence` and
    `cast_boundary_evidence` identify the evidence supporting the floors (error
    amplification analysis and dtype-cast boundaries).
    """

    input_dtype: str = eqx.field(static=True)
    parameter_dtype: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    output_dtype: str = eqx.field(static=True)
    absolute_error_floor: float | None = eqx.field(static=True)
    relative_error_floor: float | None = eqx.field(static=True)
    amplification_evidence: tuple[str, ...] = eqx.field(static=True)
    cast_boundary_evidence: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_dtype: Any,
        parameter_dtype: Any,
        compute_dtype: Any,
        accumulation_dtype: Any,
        output_dtype: Any,
        absolute_error_floor: float | None = None,
        relative_error_floor: float | None = None,
        amplification_evidence: Iterable[str] = (),
        cast_boundary_evidence: Iterable[str] = (),
    ):
        dtypes = tuple(
            _dtype_name(value, label)
            for value, label in (
                (input_dtype, "input_dtype"),
                (parameter_dtype, "parameter_dtype"),
                (compute_dtype, "compute_dtype"),
                (accumulation_dtype, "accumulation_dtype"),
                (output_dtype, "output_dtype"),
            )
        )
        absolute = _error_floor(absolute_error_floor, "absolute_error_floor")
        relative = _error_floor(relative_error_floor, "relative_error_floor")
        amplification = _identifier_set(amplification_evidence, "amplification_evidence")
        casts = _identifier_set(cast_boundary_evidence, "cast_boundary_evidence")
        (
            self.input_dtype,
            self.parameter_dtype,
            self.compute_dtype,
            self.accumulation_dtype,
            self.output_dtype,
        ) = dtypes
        self.absolute_error_floor = absolute
        self.relative_error_floor = relative
        self.amplification_evidence = amplification
        self.cast_boundary_evidence = casts

    @classmethod
    def native(cls, dtype: Any, /) -> "ComponentPrecisionContract":
        """Contract of a component evaluated entirely in `dtype`; floors undeclared."""
        return cls(
            input_dtype=dtype,
            parameter_dtype=dtype,
            compute_dtype=dtype,
            accumulation_dtype=dtype,
            output_dtype=dtype,
        )

    def residual_floor(self, scale: float, /) -> float | None:
        """Return the evaluation-error floor at magnitude `scale`.

        The floor is `max(absolute_error_floor, relative_error_floor * scale)`
        over the declared floors, or `None` when neither is declared. `scale` is
        a host-side finite non-negative real.
        """
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise TypeError("scale must be a real number.")
        scale_ = float(scale)
        if not isfinite(scale_) or scale_ < 0.0:
            raise ValueError("scale must be finite and non-negative.")
        floors = tuple(
            floor
            for floor in (
                self.absolute_error_floor,
                None
                if self.relative_error_floor is None
                else self.relative_error_floor * scale_,
            )
            if floor is not None
        )
        return max(floors) if floors else None


def _certificate_records(
    values: Iterable[AbstractConstructionCertificate | CertificateRecord], /
) -> tuple[CertificateRecord, ...]:
    if isinstance(values, (str, AbstractConstructionCertificate)):
        raise TypeError("certificates must be a collection, not one value.")
    records = set()
    for value in values:
        if isinstance(value, AbstractConstructionCertificate):
            records.add(
                (
                    _identifier(value.capability_id, "capability_id"),
                    _identifier(value.certificate_id, "certificate_id"),
                    CapabilityEvidenceKind.CONSTRUCTED,
                )
            )
            continue
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError(
                "certificates must contain AbstractConstructionCertificate values or "
                "(capability_id, certificate_id, evidence_kind) records."
            )
        capability_id, certificate_id, kind = value
        if not isinstance(kind, CapabilityEvidenceKind):
            raise TypeError("Certificate evidence kinds must be CapabilityEvidenceKind.")
        if kind is CapabilityEvidenceKind.DECLARED:
            raise ValueError("A certificate records constructed or checked evidence.")
        records.add(
            (
                _identifier(capability_id, "capability_id"),
                _identifier(certificate_id, "certificate_id"),
                kind,
            )
        )
    return tuple(sorted(records))


def _precision_payload(precision: ComponentPrecisionContract | None, /) -> Any:
    if precision is None:
        return None
    return {
        "dtypes": [
            precision.input_dtype,
            precision.parameter_dtype,
            precision.compute_dtype,
            precision.accumulation_dtype,
            precision.output_dtype,
        ],
        "absolute_error_floor": precision.absolute_error_floor,
        "relative_error_floor": precision.relative_error_floor,
        "amplification_evidence": list(precision.amplification_evidence),
        "cast_boundary_evidence": list(precision.cast_boundary_evidence),
    }


def _randomness_payload(randomness: RandomnessContract | None, /) -> Any:
    if randomness is None:
        return None
    return {
        "mode": randomness.mode,
        "requires_inference_state": randomness.requires_inference_state,
        "realization_id": randomness.realization_id,
    }


@final
class ModelExecutionContract(StrictModule, NonTrainableState):
    """Intrinsic execution contract of one model.

    It describes the model independent of any owner: its `derivative` contract
    (whose `regularity` is the model's value regularity), `execution`
    capabilities, `precision`, `randomness`, intrinsic `ports`, construction
    `certificates`, and `semantic_provenance`. `None` means undeclared. It never
    declares authority; binding to an owner slot does.

    Certificates are stored as sorted `(capability_id, certificate_id,
    evidence_kind)` records; certificate instances record `CONSTRUCTED`
    evidence. A host-only model offers no JAX derivative route: its contract
    route is `EXTERNAL_ADJOINT` or `STOPPED`, or it declares no supported
    surface. The `EXTERNAL_ADJOINT` route requires the `"external-adjoint"` tier.
    `contract_id` content-addresses the contract.
    """

    derivative: DerivativeContract
    execution: ExecutionCapabilities
    precision: ComponentPrecisionContract | None
    randomness: RandomnessContract | None
    ports: ModelPorts | None
    semantic_provenance: SemanticProvenance | None
    certificates: tuple[CertificateRecord, ...] = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        derivative: DerivativeContract,
        execution: ExecutionCapabilities,
        precision: ComponentPrecisionContract | None = None,
        randomness: RandomnessContract | None = None,
        ports: ModelPorts | None = None,
        certificates: Iterable[AbstractConstructionCertificate | CertificateRecord] = (),
        semantic_provenance: SemanticProvenance | None = None,
    ):
        if not isinstance(derivative, DerivativeContract):
            raise TypeError("derivative must be a DerivativeContract.")
        if not isinstance(execution, ExecutionCapabilities):
            raise TypeError("execution must be ExecutionCapabilities.")
        _optional_type(precision, ComponentPrecisionContract, "precision")
        _optional_type(randomness, RandomnessContract, "randomness")
        _optional_type(ports, ModelPorts, "ports")
        _optional_type(semantic_provenance, SemanticProvenance, "semantic_provenance")
        records = _certificate_records(certificates)
        route = derivative.route
        if (
            execution.host_only
            and derivative.supported_surfaces
            and route not in (DerivativeRoute.EXTERNAL_ADJOINT, DerivativeRoute.STOPPED)
        ):
            raise ValueError(
                "A host-only model cannot be differentiated by JAX transforms; declare "
                "an EXTERNAL_ADJOINT or STOPPED route, or no supported surfaces."
            )
        if route is DerivativeRoute.EXTERNAL_ADJOINT and execution.tier != (
            "external-adjoint"
        ):
            raise ValueError(
                "The EXTERNAL_ADJOINT route requires the 'external-adjoint' tier."
            )
        self.derivative = derivative
        self.execution = execution
        self.precision = precision
        self.randomness = randomness
        self.ports = ports
        self.semantic_provenance = semantic_provenance
        self.certificates = records
        self.contract_id = canonical_fingerprint(
            {
                "kind": "model-execution-contract",
                "derivative": derivative.contract_id,
                "execution": {
                    "tier": execution.tier,
                    "jit": execution.jit,
                    "vmap": execution.vmap,
                    "host_only": execution.host_only,
                    "stateful": execution.stateful,
                },
                "precision": _precision_payload(precision),
                "randomness": _randomness_payload(randomness),
                "ports": None if ports is None else ports.ports_id,
                "certificates": [list(record) for record in records],
                "semantic_provenance": (
                    None
                    if semantic_provenance is None
                    else semantic_provenance.semantic_id
                ),
            }
        )

    @property
    def regularity(self) -> DerivativeRegularity | None:
        """Declared value regularity (`None` when undeclared)."""
        return self.derivative.regularity

    @property
    def evidence(self) -> tuple[tuple[str, CapabilityEvidenceKind], ...]:
        """Sorted unique `(capability_id, evidence_kind)` pairs of the certificates."""
        return tuple(
            sorted({(capability, kind) for capability, _, kind in self.certificates})
        )

    @property
    def evidence_model_id(self) -> str:
        """Identity naming this model in admissibility evidence.

        The declared semantic provenance ID, else a marked fingerprint of this
        contract so undeclared provenance is never mistaken for a declared one.
        """
        if self.semantic_provenance is not None:
            return self.semantic_provenance.semantic_id
        return canonical_fingerprint(
            {"kind": "undeclared-model-provenance", "model_contract": self.contract_id}
        )


def _requirement_key(requirement: CapabilityRequirement, /) -> tuple[Any, ...]:
    return (
        requirement.capability_id,
        tuple(
            tuple(kind.value for kind in option) for option in requirement.alternatives
        ),
        requirement.safety_critical,
    )


def _requirements(
    values: Iterable[CapabilityRequirement], /
) -> tuple[CapabilityRequirement, ...]:
    unique = {}
    for requirement in values:
        if not isinstance(requirement, CapabilityRequirement):
            raise TypeError("requirements must contain CapabilityRequirement values.")
        unique.setdefault(_requirement_key(requirement), requirement)
    return tuple(unique[key] for key in sorted(unique))


def _require_authority(authority: Any, /) -> ComponentAuthority:
    if not isinstance(authority, ComponentAuthority):
        raise TypeError("authority must be a ComponentAuthority.")
    return authority


def _slot_semantic_id(value: Any, /) -> str | None:
    return None if value is None else _identifier(value, "slot_semantic_id")


@final
class ComponentSlotContract(StrictModule, NonTrainableState):
    """Authority, identity, and admissibility requirements of one owner slot.

    `slot_semantic_id` is `None` for a binding that names an authority without
    a slot. Requirements are deduplicated and stored in canonical order.
    """

    authority: ComponentAuthority = eqx.field(static=True)
    slot_semantic_id: str | None = eqx.field(static=True)
    requirements: tuple[CapabilityRequirement, ...]

    def __init__(
        self,
        authority: ComponentAuthority,
        /,
        *,
        slot_semantic_id: str | None = None,
        requirements: Iterable[CapabilityRequirement] = (),
    ):
        authority_ = _require_authority(authority)
        semantic_id = _slot_semantic_id(slot_semantic_id)
        requirements_ = _requirements(requirements)
        self.authority = authority_
        self.slot_semantic_id = semantic_id
        self.requirements = requirements_


class AbstractComponentSlot(StrictModule):
    """Marker base of an owner slot holding one component.

    A slot class declares the `component_authority` it confers, its
    `slot_semantic_id`, and the `slot_requirements` every bound component must
    satisfy. It fixes no call signature: each slot family defines its own. An
    inline component is an instance of a slot class and takes its authority from
    that class.
    """

    component_authority: AbstractClassVar[ComponentAuthority]
    slot_semantic_id: AbstractClassVar[str]
    slot_requirements: ClassVar[tuple[CapabilityRequirement, ...]] = ()

    @classmethod
    def slot_contract(cls) -> ComponentSlotContract:
        """Return the validated authority, identity, and requirements of the slot."""
        return ComponentSlotContract(
            cls.component_authority,
            slot_semantic_id=_identifier(cls.slot_semantic_id, "slot_semantic_id"),
            requirements=cls.slot_requirements,
        )


def _requirement_payload(requirement: CapabilityRequirement, /) -> dict[str, Any]:
    capability_id, alternatives, safety_critical = _requirement_key(requirement)
    return {
        "capability_id": capability_id,
        "alternatives": [list(option) for option in alternatives],
        "safety_critical": safety_critical,
    }


def _admission_payload(admission: DerivativeAdmission | None, /) -> Any:
    if admission is None:
        return None
    request = admission.request
    return {
        "surfaces": [surface.value for surface in request.surfaces],
        "order": request.order,
        "authority": None if request.authority is None else request.authority.value,
        "levels": [level.value for level in admission.levels],
        "route": admission.route.value,
        "conditions": list(admission.conditions),
        "reasons": list(admission.reasons),
        "nondifferentiable_outputs": list(admission.nondifferentiable_outputs),
    }


@final
class ComponentContract(StrictModule, NonTrainableState):
    """Contract of one model bound to an owner slot.

    It records the conferred `authority`, the `slot_semantic_id` (`None` for an
    authority-only binding), the intrinsic `model_contract`, the
    `port_binding` evidence (`None` when ports were not bound), the
    admissibility `requirements`, and an optional `derivative_admission` made
    for this authority. `evidence` holds the `(capability_id, evidence_kind)`
    pairs the model provides. Construction fails closed: every requirement must
    be satisfied by the evidence for its capability, and the admission must be
    requested by this authority on the model's derivative route.
    `bound_semantic_id` content-addresses the binding.
    """

    authority: ComponentAuthority = eqx.field(static=True)
    slot_semantic_id: str | None = eqx.field(static=True)
    model_contract: ModelExecutionContract
    port_binding: PortBindingEvidence | None
    requirements: tuple[CapabilityRequirement, ...]
    derivative_admission: DerivativeAdmission | None
    evidence: tuple[tuple[str, CapabilityEvidenceKind], ...] = eqx.field(static=True)
    bound_semantic_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        authority: ComponentAuthority,
        model_contract: ModelExecutionContract,
        slot_semantic_id: str | None = None,
        port_binding: PortBindingEvidence | None = None,
        requirements: Iterable[CapabilityRequirement] = (),
        derivative_admission: DerivativeAdmission | None = None,
    ):
        authority_ = _require_authority(authority)
        semantic_id = _slot_semantic_id(slot_semantic_id)
        if not isinstance(model_contract, ModelExecutionContract):
            raise TypeError("model_contract must be a ModelExecutionContract.")
        _optional_type(port_binding, PortBindingEvidence, "port_binding")
        _optional_type(derivative_admission, DerivativeAdmission, "derivative_admission")
        requirements_ = _requirements(requirements)
        evidence = model_contract.evidence
        unsatisfied = tuple(
            requirement.capability_id
            for requirement in requirements_
            if not requirement.is_satisfied_by(
                kind
                for capability, kind in evidence
                if capability == requirement.capability_id
            )
        )
        if unsatisfied:
            raise ValueError(
                f"Component bound with {authority_.value} authority lacks the evidence "
                f"required for capabilities {unsatisfied!r}."
            )
        if derivative_admission is not None:
            if derivative_admission.request.authority is not authority_:
                raise ValueError(
                    "derivative_admission must be requested with the bound authority."
                )
            if derivative_admission.route is not model_contract.derivative.route:
                raise ValueError(
                    "derivative_admission must use the model's derivative route."
                )
        self.authority = authority_
        self.slot_semantic_id = semantic_id
        self.model_contract = model_contract
        self.port_binding = port_binding
        self.requirements = requirements_
        self.derivative_admission = derivative_admission
        self.evidence = evidence
        self.bound_semantic_id = canonical_fingerprint(
            {
                "kind": "component-contract",
                "authority": authority_.value,
                "slot_semantic_id": semantic_id,
                "model_contract": model_contract.contract_id,
                "port_binding": (
                    None if port_binding is None else port_binding.binding_fingerprint
                ),
                "requirements": [
                    _requirement_payload(requirement) for requirement in requirements_
                ],
                "derivative_admission": _admission_payload(derivative_admission),
                "evidence": [[capability, kind.value] for capability, kind in evidence],
            }
        )


def _port_binding(
    model_ports: ModelPorts | None,
    owner_ports: ModelPorts | None,
    mapping: PortMapping | None,
    /,
) -> PortBindingEvidence | None:
    if model_ports is None:
        if mapping is not None:
            raise ValueError("A port mapping was supplied for a model without ports.")
        return None
    if owner_ports is None:
        return None
    if mapping is None:
        raise ValueError(
            "Binding a model with declared ports to owner ports requires an "
            "explicit PortMapping."
        )
    return resolve_port_mapping(model_ports, owner_ports, mapping)


def _bound_contract(
    model: "AbstractArrayModel",
    authority: ComponentAuthority,
    slot_semantic_id: str | None,
    port_mapping: PortMapping | None,
    owner_ports: ModelPorts | None,
    requirements: tuple[CapabilityRequirement, ...],
    request: DifferentiationRequest | None,
    policy: RegularityPolicy | None,
    /,
) -> ComponentContract:
    model_contract = model.model_execution_contract()
    return ComponentContract(
        authority=authority,
        model_contract=model_contract,
        slot_semantic_id=slot_semantic_id,
        port_binding=_port_binding(model_contract.ports, owner_ports, port_mapping),
        requirements=requirements,
        derivative_admission=(
            None
            if request is None
            else model_contract.derivative.admit(request, policy=policy)
        ),
    )


@final
class ComponentBinding(StrictModule):
    """Explicit binding of a model held separately from its owner.

    The `model` is a dynamic child whose arrays keep their own roles (a
    `ParameterOwner` model's arrays stay PARAMETER); the binding's authority,
    slot identity, port mapping, owner ports, and requirements are static
    metadata. Construction validates the binding by forming its contract, so
    port and requirement violations fail at binding time. A mapping is required
    exactly when the model declares ports and `owner_ports` are supplied.
    """

    model: "AbstractArrayModel"
    authority: ComponentAuthority = eqx.field(static=True)
    slot_semantic_id: str | None = eqx.field(static=True)
    port_mapping: PortMapping | None
    owner_ports: ModelPorts | None
    requirements: tuple[CapabilityRequirement, ...]

    def __init__(
        self,
        model: "AbstractArrayModel",
        /,
        *,
        authority: ComponentAuthority,
        slot_semantic_id: str | None = None,
        port_mapping: PortMapping | None = None,
        owner_ports: ModelPorts | None = None,
        requirements: Iterable[CapabilityRequirement] = (),
    ):
        # Imported here: `_array` imports this module for the model contract types.
        from ._array import AbstractArrayModel

        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        authority_ = _require_authority(authority)
        semantic_id = _slot_semantic_id(slot_semantic_id)
        _optional_type(port_mapping, PortMapping, "port_mapping")
        _optional_type(owner_ports, ModelPorts, "owner_ports")
        if port_mapping is not None and owner_ports is None:
            raise ValueError("A port mapping binds to owner ports; supply owner_ports.")
        requirements_ = _requirements(requirements)
        _bound_contract(
            model,
            authority_,
            semantic_id,
            port_mapping,
            owner_ports,
            requirements_,
            None,
            None,
        )
        self.model = model
        self.authority = authority_
        self.slot_semantic_id = semantic_id
        self.port_mapping = port_mapping
        self.owner_ports = owner_ports
        self.requirements = requirements_

    def contract(
        self,
        *,
        request: DifferentiationRequest | None = None,
        policy: RegularityPolicy | None = None,
    ) -> ComponentContract:
        """Return the bound contract, admitting `request` when supplied.

        Ports are resolved with `resolve_port_mapping`; requirements are checked
        against the model's evidence. `request` must carry the bound authority
        and is admitted against the model's derivative contract under `policy`.
        """
        return _bound_contract(
            self.model,
            self.authority,
            self.slot_semantic_id,
            self.port_mapping,
            self.owner_ports,
            self.requirements,
            request,
            policy,
        )


def bind_component(
    model: "AbstractArrayModel",
    slot: type[AbstractComponentSlot] | ComponentAuthority,
    /,
    *,
    port_mapping: PortMapping | None = None,
    owner_ports: ModelPorts | None = None,
    requirements: Iterable[CapabilityRequirement] = (),
) -> ComponentBinding:
    """Bind `model` to an owner slot class or to a bare authority.

    A slot class confers its authority, semantic ID, and requirements, to which
    `requirements` are added; a bare `ComponentAuthority` binds without a slot
    identity.
    """
    if isinstance(slot, ComponentAuthority):
        slot_contract = ComponentSlotContract(slot)
    elif isinstance(slot, type) and issubclass(slot, AbstractComponentSlot):
        slot_contract = slot.slot_contract()
    else:
        raise TypeError(
            "slot must be an AbstractComponentSlot subclass or a ComponentAuthority."
        )
    return ComponentBinding(
        model,
        authority=slot_contract.authority,
        slot_semantic_id=slot_contract.slot_semantic_id,
        port_mapping=port_mapping,
        owner_ports=owner_ports,
        requirements=(*slot_contract.requirements, *requirements),
    )


def slot_component_contracts(
    slot: type[AbstractComponentSlot],
    tree: Any,
    /,
    *,
    scope: str,
) -> tuple[tuple[str, ComponentContract], ...]:
    """Bind every model below `tree` to `slot`; return `(location, contract)` pairs.

    A bare `AbstractArrayModel` is bound to `slot`; a `ComponentBinding` must
    already carry the slot's authority and its slot identity (or none).
    Locations are `scope` plus the PyTree path, in flattening order.
    """
    from ._array import AbstractArrayModel

    if not (isinstance(slot, type) and issubclass(slot, AbstractComponentSlot)):
        raise TypeError("slot must be an AbstractComponentSlot subclass.")
    entries, _ = jax.tree_util.tree_flatten_with_path(
        tree,
        is_leaf=lambda node: isinstance(node, (AbstractArrayModel, ComponentBinding)),
    )
    contracts = []
    for path, node in entries:
        location = scope + jax.tree_util.keystr(path)
        if isinstance(node, ComponentBinding):
            if node.authority is not slot.component_authority or (
                node.slot_semantic_id not in (None, slot.slot_semantic_id)
            ):
                raise ValueError(
                    f"Component {location} is bound with {node.authority.value} "
                    f"authority to slot {node.slot_semantic_id!r}; a "
                    f"{slot.__name__} slot confers "
                    f"{slot.component_authority.value} authority."
                )
            contracts.append((location, node.contract()))
        elif isinstance(node, AbstractArrayModel):
            contracts.append((location, bind_component(node, slot).contract()))
    return tuple(contracts)


__all__ = [
    "AbstractComponentSlot",
    "CertificateRecord",
    "ComponentBinding",
    "ComponentContract",
    "ComponentPrecisionContract",
    "ComponentSlotContract",
    "ExecutionCapabilities",
    "ExecutionTier",
    "ModelExecutionContract",
    "RandomnessContract",
    "RandomnessMode",
    "admit_randomness",
    "bind_component",
    "slot_component_contracts",
    "supports_derivative",
]
