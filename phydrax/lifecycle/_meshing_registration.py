"""Live producer-issued registration reuse, never portable archive authority."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from os import getpid
from typing import Any

import jax
import numpy as np

from .._meshcore import (
    current_native_execution_scope,
    NativeBatchTicket,
    NativeExecutionScope,
    NativeScopeTicket,
)
from ..config import snapshot as configuration_snapshot
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._compartments import CompartmentMeshingSource
from ..meshing._certification import MeshCertificationReport
from ..meshing._result import CellMeshingResult
from ..meshing.providers._native import NativeMeshingPlan, NativeMeshingProvider
from ..meshing.providers._native_options import NativeMeshingOptions
from ._meshing_source_families import NativeGenerationSource
from ._meshing_source_records import stored_record_content_id


@dataclass(frozen=True, slots=True)
class _IssuedRegistration:
    source: weakref.ReferenceType[Any]
    report: weakref.ReferenceType[MeshCertificationReport]
    constructors: tuple[weakref.ReferenceType[Any], ...]
    process_id: int


@dataclass(frozen=True, slots=True)
class _RegistrationPayload:
    configuration: Mapping[str, object]
    source: NativeGenerationSource
    specification: object
    options: NativeMeshingOptions
    result_fields: tuple[tuple[str, object], ...]


# Neither keys nor values retain a numerical owner, an inspection plan, or a
# request budget. A fresh process, dead owner, changed payload or changed
# constructor must follow the ordinary full source-inspection path.
_issued: dict[tuple[int, str], _IssuedRegistration] = {}


@dataclass(frozen=True, slots=True)
class _IssuedMaterialRegistration:
    source: weakref.ReferenceType[Any]
    report: weakref.ReferenceType[MeshCertificationReport]
    evidence: weakref.ReferenceType[Any]
    authority: weakref.ReferenceType[Any]
    root: weakref.ReferenceType[NativeExecutionScope]
    constructors: tuple[weakref.ReferenceType[Any], ...]
    content: str
    process_id: int


# Identity keys are not authority. Every owner must still be live and identical;
# callbacks capture only these integer keys, never any of their numerical owners.
_issued_material: dict[tuple[int, ...], _IssuedMaterialRegistration] = {}


def _constructors(
    source: Any, specification: Any, options: NativeMeshingOptions,
    result: CellMeshingResult, /,
) -> tuple[Callable[..., Any], ...]:
    # Resolve current module attributes, not a saved imported function alias.
    from ..meshing import _certification as certification_owner
    from ..meshing.providers import _native as native_owner
    from ..meshing.providers import _native_publication as publication_owner
    from ..meshing.qualification import _hybrid as hybrid_owner
    from . import _meshing_sources as source_owner

    functions = (
        NativeMeshingProvider.plan, NativeMeshingProvider.validate,
        NativeMeshingProvider._plan_bound, NativeMeshingPlan.__init__,
        NativeMeshingPlan._execute_in_session,
        native_owner._prepare_bound_request, native_owner._bind_request,
        publication_owner.publish_native_result,
        publication_owner._assemble_publication,
        publication_owner._certify_publication,
        hybrid_owner.qualify_hybrid_curved_publication,
        certification_owner.certify_meshing_acceptance,
        source_owner._audit.audit_cell_mesh,
        source_owner._cell_geometry_validity.certify_cell_geometry_validity,
    )
    found: dict[int, Callable[..., Any]] = {id(function): function for function in functions}
    pending = [source, specification, options, result]
    seen: set[int] = set()
    admissions = source_owner._source_admissions()
    while pending:
        value = pending.pop()
        if id(value) in seen or isinstance(value, (jax.Array, np.ndarray)):
            continue
        seen.add(id(value))
        if is_dataclass(value):
            owner = type(value)
            candidates = [owner.__init__]
            candidates.extend(
                function for name in ("from_stored_fields", "validate_restored")
                if (function := getattr(owner, name, None)) is not None
            )
            admission = admissions.get(owner)
            if admission is not None:
                candidates.extend(function for function in (
                    admission.reconstruct, admission.validate_restored
                ) if function is not None)
            for function in candidates:
                function = getattr(function, "__func__", function)
                found[id(function)] = function
            pending.extend(
                getattr(value, member.name) for member in fields(value)
                if value is not result or member.name != "execution_evidence"
            )
        elif isinstance(value, Mapping):
            pending.extend(child for pair in value.items() for child in pair)
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
    return tuple(found[key] for key in sorted(found))


def _content(
    source: Any, specification: Any, options: NativeMeshingOptions,
    result: CellMeshingResult, budget: CoordinateEnclosureBudget, /,
    *, memoize: bool = True,
) -> str:
    # Full current NumPy bytes are reauthenticated by the canonical record
    # codec; read-only flags are not an immutability premise. Only the root
    # execution receipt is outside scientific publication identity.
    return stored_record_content_id(
        _RegistrationPayload(
            configuration_snapshot().to_record(), source, specification, options,
            tuple(
                (member.name, getattr(result, member.name))
                for member in fields(result) if member.name != "execution_evidence"
            ),
        ),
        excluded="", budget=budget, memoize=memoize,
    )


def _remember_issued_native_generation_registration(
    source: NativeGenerationSource,
    specification: Any,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Called only by a genuine publisher after its actual issued-plan proof.

    The producer authenticates its complete current prepared inspection payload
    and actual constructor identities before calling. This private issuer is
    neither a public proof constructor nor a source/archive admission shortcut.
    """
    report = result.certification
    if report is None:
        raise ValueError("native_registration.report must retain the genuine final certificate; got None")
    MeshCertificationReport.require_passed(report)
    constructors = tuple(weakref.ref(function) for function in _constructors(
        source, specification, options, result
    ))
    key = (id(source), _content(source, specification, options, result, budget))

    def forget(reference: weakref.ReferenceType[Any]) -> None:
        entry = _issued.get(key)
        if entry is not None and (entry.source is reference or entry.report is reference):
            del _issued[key]

    _issued[key] = _IssuedRegistration(
        weakref.ref(source, forget), weakref.ref(report, forget), constructors, getpid()
    )


def has_issued_native_generation_registration(
    source: NativeGenerationSource,
    specification: Any,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> bool:
    """Reuse only exact live source/report owners on a current paid request."""
    if budget is None:
        return False
    candidates = tuple(
        (key, entry) for key, entry in tuple(_issued.items())
        if key[0] == id(source) and entry.process_id == getpid()
        and entry.source() is source and entry.report() is result.certification
    )
    if not candidates:
        return False
    constructors = _constructors(source, specification, options, result)
    candidates = tuple(
        (key, entry) for key, entry in candidates
        if len(entry.constructors) == len(constructors)
        and all(reference() is function for reference, function in zip(
            entry.constructors, constructors, strict=True
        ))
    )
    if not candidates:
        return False
    content = _content(source, specification, options, result, budget)
    return any(key == (id(source), content) for key, _ in candidates)


def _material_root(
    budget: CoordinateEnclosureBudget | None, /,
) -> NativeExecutionScope | None:
    """Require an actual unsettled native ticket in its current owner tree."""
    if budget is None:
        return None
    ticket = budget.ticket
    if not isinstance(ticket, (NativeScopeTicket, NativeBatchTicket)) or ticket.settled:
        return None
    try:
        active = current_native_execution_scope()
    except RuntimeError:
        # A copied context cannot transfer the native thread-local allowance.
        return None
    owner = active
    while owner is not None and owner is not ticket.scope:
        owner = owner.parent
    if owner is None or active is None:
        return None
    while active.parent is not None:
        active = active.parent
    return active


def _material_constructors(
    source: CompartmentMeshingSource, specification: Any,
    options: NativeMeshingOptions, result: CellMeshingResult, /,
) -> tuple[Callable[..., Any], ...]:
    from ..geometry import _domain_coverage as coverage_owner
    from ..meshing import _compartments as material_owner
    from ..meshing import _organization_records as organization_owner
    from ..meshing.providers import _native_compartment as producer_owner

    functions = (
        *_constructors(source, specification, options, result),
        type(source).validate_source_integrity,
        material_owner.generate_compartment_volume,
        material_owner._generate_compartment_volume,
        material_owner.prepare_compartment_complex,
        material_owner._prepare_compartment_outer_domain,
        material_owner._compartment_outer_domain,
        material_owner._exact_compartment_domains,
        material_owner._material_patches,
        material_owner.revalidate_region_evidence,
        material_owner._renew_region_evidence,
        material_owner._prepare_renewed_region_source,
        material_owner._prepare_renewed_region_organization,
        material_owner._certify_renewed_region,
        material_owner._certify_coverage_stage,
        coverage_owner.certify_domain_coverage,
        material_owner.certify_domain_coverage,
        organization_owner.RegionMeshingEvidence.require_source,
        organization_owner.RegionMeshingEvidence.require_current,
        producer_owner.execute_compartment_route,
        producer_owner.generate_compartment_volume,
        producer_owner.publish_native_result,
        producer_owner.coordinate_budget,
        producer_owner.NativePublicationExecution.__init__,
        producer_owner._CompartmentOuterProofPayload.__init__,
        producer_owner._issue_compartment_material_authority,
        producer_owner._require_issued_compartment_material_authority,
    )
    found = {id(function): function for function in functions}
    return tuple(found[key] for key in sorted(found))


def _remember_issued_compartment_material_registration(
    source: CompartmentMeshingSource,
    specification: Any,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Seal a genuine producer authority, not a passed certificate or plan.

    Only the actual compartment producer can issue the prerequisite authority
    after completing material, interface, independent outer-source and final
    publication proofs under this very request. This registration cannot earn
    that authority, renew material evidence, or survive its paid native root.

    The producer-owned ``_native_compartment`` verifier contract is
    ``_require_issued_compartment_material_authority(source, specification,
    options, result, /, *, budget) -> object``. It returns its actual live,
    weak-referenceable proof owner, or raises ``ValueError`` on refusal.
    It must authenticate issuance at the real proof boundary, including the
    source snapshot, constructors and original paid root. Distinct producer
    and caller ledgers are admissible only when the verifier authenticates
    both actual ledger attachments and their limits under that same live
    root. Python budget identity is neither necessary nor sufficient.
    A later passed-status check, matching IDs or plan inspection cannot issue it.
    The verifier must not silently earn fresh proof or open a replacement
    request. Missing producer support is a refusal, never a cold fallback
    authority. Current bytes and constructors are independently checked here.
    """
    from ..meshing.providers import _native_compartment as producer_owner

    root = _material_root(budget)
    if root is None:
        raise ValueError(
            "material_registration.budget must retain its actual live native root and ledger"
        )
    report, evidence = result.certification, result.region_evidence
    if not isinstance(source, CompartmentMeshingSource) or report is None or evidence is None:
        raise ValueError(
            "material_registration.result must retain its actual image source, report and material evidence"
        )
    verifier = getattr(producer_owner, "_require_issued_compartment_material_authority", None)
    if verifier is None:
        raise ValueError(
            "material_registration.authority requires the actual compartment producer issuer"
        )
    authority = verifier(source, specification, options, result, budget=budget)
    if authority is None:
        raise ValueError("material_registration.authority must retain its live producer owner")
    constructors = tuple(weakref.ref(function) for function in _material_constructors(
        source, specification, options, result
    ))
    # One-shot canonical hashing must not retain this payload (and therefore its
    # source/report graph) in the budget's ordinary strong content memo.
    content = _content(source, specification, options, result, budget, memoize=False)
    key = (id(source), id(report), id(evidence), id(root))

    def forget(reference: weakref.ReferenceType[Any]) -> None:
        entry = _issued_material.get(key)
        if entry is not None and any(owner is reference for owner in (
            entry.source, entry.report, entry.evidence, entry.authority,
            entry.root,
        )):
            del _issued_material[key]

    _issued_material[key] = _IssuedMaterialRegistration(
        weakref.ref(source, forget), weakref.ref(report, forget),
        weakref.ref(evidence, forget), weakref.ref(authority, forget),
        weakref.ref(root, forget),
        constructors, content, getpid(),
    )


def has_issued_compartment_material_registration(
    source: CompartmentMeshingSource,
    specification: Any,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> bool:
    """Reuse only the complete unchanged proof within its original paid root.

    Cold, reconstructed, changed, expired or foreign-context owners return
    false. Callers must retain all ordinary source and request guards and take
    the full paid material/source renewal path on refusal.
    The producer verifier must authenticate this caller ledger's actual root
    attachment and limits even when a different ledger earned the proof.
    """
    from ..meshing.providers import _native_compartment as producer_owner

    root = _material_root(budget)
    if root is None or budget is None or not isinstance(source, CompartmentMeshingSource):
        return False
    report, evidence = result.certification, result.region_evidence
    key = (id(source), id(report), id(evidence), id(root))
    entry = _issued_material.get(key)
    if (
        entry is None or entry.process_id != getpid()
        or entry.source() is not source or entry.report() is not report
        or entry.evidence() is not evidence or entry.root() is not root
        or entry.authority() is None
    ):
        return False
    verifier = getattr(producer_owner, "_require_issued_compartment_material_authority", None)
    if verifier is None:
        return False
    constructors = _material_constructors(source, specification, options, result)
    if len(constructors) != len(entry.constructors) or any(
        reference() is not function for reference, function in zip(
            entry.constructors, constructors, strict=True
        )
    ):
        return False
    try:
        authority = verifier(source, specification, options, result, budget=budget)
    except ValueError:
        return False
    if authority is not entry.authority():
        return False
    return entry.content == _content(
        source, specification, options, result, budget, memoize=False
    )
