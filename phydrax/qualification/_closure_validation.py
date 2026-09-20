#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Human-readable closure, source, benchmark, and public-surface diagnostics."""

from __future__ import annotations

from ._catalog import CapabilityCatalog
from ._closure_matrix import CapabilityClosureMatrix
from ._closure_taxonomy import ClosureDisposition
from ._source_reference import SourceAbsorptionLedger, SourceReview


def validate_closure_catalog(
    catalog: CapabilityCatalog, matrices: tuple[CapabilityClosureMatrix, ...], /
) -> tuple[str, ...]:
    declarations = {value.capability: value for value in catalog.declarations}
    capabilities = set(declarations)
    errors: list[str] = []
    families = tuple(value.family for value in matrices)
    if len(set(families)) != len(families):
        errors.append("duplicate-closure-family")
    for matrix in matrices:
        requirements = {
            requirement.requirement_id: requirement for requirement in matrix.requirements
        }
        for requirement in matrix.requirements:
            if not requirement.source_ids:
                errors.append(f"missing-source:{matrix.family}:{requirement.requirement}")
        for resolution in matrix.resolutions:
            missing = set(resolution.capability_ids).difference(capabilities)
            for capability in sorted(missing):
                errors.append(f"unknown-capability:{matrix.family}:{capability}")
            if resolution.disposition is not ClosureDisposition.IMPLEMENTED:
                continue
            requirement = requirements[resolution.requirement_id]
            if not resolution.capability_ids:
                errors.append(
                    f"implemented-without-capability:{matrix.family}:{resolution.requirement_id}"
                )
            if not resolution.evidence_ids:
                errors.append(
                    f"implemented-without-evidence:{matrix.family}:{resolution.requirement_id}"
                )
            if resolution.actual_depth < requirement.minimum_depth:
                errors.append(
                    f"insufficient-depth:{matrix.family}:{resolution.requirement_id}"
                )
            missing_providers = set(requirement.required_providers).difference(
                resolution.provider_ids
            )
            for provider in sorted(missing_providers):
                errors.append(
                    f"missing-provider:{matrix.family}:{resolution.requirement_id}:{provider}"
                )
            resolved_declarations = tuple(
                declarations[capability]
                for capability in resolution.capability_ids
                if capability in declarations
            )
            public_symbols = {
                symbol
                for declaration in resolved_declarations
                for symbol in declaration.public_symbols
            }
            documents = {
                document
                for declaration in resolved_declarations
                for document in declaration.documentation
            }
            for symbol in sorted(
                set(requirement.required_public_symbols).difference(public_symbols)
            ):
                errors.append(
                    f"missing-public-symbol:{matrix.family}:{resolution.requirement_id}:{symbol}"
                )
            for document in sorted(
                set(requirement.required_documents).difference(documents)
            ):
                errors.append(
                    f"missing-document:{matrix.family}:{resolution.requirement_id}:{document}"
                )
            if resolution.release_authorized and any(
                declaration.disposition.value != "released"
                for declaration in resolved_declarations
            ):
                errors.append(
                    f"unauthorized-release:{matrix.family}:{resolution.requirement_id}"
                )
    return tuple(sorted(errors))


def validate_source_coverage(ledger: SourceAbsorptionLedger, /) -> tuple[str, ...]:
    errors = []
    unresolved = ("unpinned", "unresolved")
    for source in ledger.sources:
        if source.revision in unresolved:
            errors.append(f"unpinned-source:{source.source_id}")
        if source.archive_digest in unresolved:
            errors.append(f"unresolved-source-tree:{source.source_id}")
        if source.licence_digest in unresolved:
            errors.append(f"unresolved-source-licence:{source.source_id}")
        if len(source.relevant_documents) < 2:
            errors.append(f"unreviewed-source-documents:{source.source_id}")
        if source.review_status is SourceReview.UNREVIEWED:
            errors.append(f"unreviewed-source:{source.source_id}")
    return tuple(sorted(errors))


__all__ = ["validate_closure_catalog", "validate_source_coverage"]
