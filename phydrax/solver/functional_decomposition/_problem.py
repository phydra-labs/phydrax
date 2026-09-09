#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..._term import AbstractScalarTerm
from ...conditions import localize_residual
from ...domain import (
    DomainComponent,
    DomainFunction,
    LocalFieldFamily,
    LocalFieldRef,
    partition_of_unity_field,
    SubdomainCover,
    SubdomainPatch,
)
from ...enforcement import EnforcementProgram
from ...terms import ResidualPenalty


ScopeKind = Literal["patch", "pair", "global"]
AssemblyKind = Literal["partition-of-unity", "broken"]


class PatchScope(StrictModule):
    patch_id: str = eqx.field(static=True)

    def __init__(self, patch_id: str, /):
        value = str(patch_id)
        if not value:
            raise ValueError("patch_id must be non-empty.")
        self.patch_id = value


class PairScope(StrictModule):
    pairing_id: str = eqx.field(static=True)

    def __init__(self, pairing_id: str, /):
        value = str(pairing_id)
        if not value:
            raise ValueError("pairing_id must be non-empty.")
        self.pairing_id = value


class GlobalScope(StrictModule):
    def __init__(self):
        pass


TermScope = PatchScope | PairScope | GlobalScope


class ScopedFunctionalTerm(StrictModule):
    """One ordinary functional term with explicit decomposition ownership."""

    term: AbstractScalarTerm
    scope: TermScope

    def __init__(self, term: AbstractScalarTerm, scope: TermScope, /):
        if not isinstance(term, AbstractScalarTerm):
            raise TypeError("term must be an AbstractScalarTerm.")
        if not isinstance(scope, (PatchScope, PairScope, GlobalScope)):
            raise TypeError("scope must be PatchScope, PairScope, or GlobalScope.")
        self.term = term
        self.scope = scope


def _scoped_terms(
    values: AbstractScalarTerm
    | ScopedFunctionalTerm
    | Sequence[AbstractScalarTerm | ScopedFunctionalTerm],
    /,
) -> tuple[ScopedFunctionalTerm, ...]:
    if isinstance(values, (AbstractScalarTerm, ScopedFunctionalTerm)):
        entries = (values,)
    else:
        entries = tuple(values)
    result: list[ScopedFunctionalTerm] = []
    for entry in entries:
        if isinstance(entry, ScopedFunctionalTerm):
            result.append(entry)
        elif isinstance(entry, AbstractScalarTerm):
            result.append(ScopedFunctionalTerm(entry, GlobalScope()))
        else:
            raise TypeError(
                "Decomposition terms must be scalar terms or ScopedFunctionalTerm values."
            )
    return tuple(result)


class FunctionalDecompositionProblem(StrictModule):
    """Static functional problem over one local-field family and cover."""

    cover: SubdomainCover
    family: LocalFieldFamily
    functions: frozendict[str, DomainFunction]
    terms: tuple[ScopedFunctionalTerm, ...]
    evaluation_terms: tuple[ScopedFunctionalTerm, ...]
    enforcement: EnforcementProgram | None
    collocation_key: Key[Array, ""]
    assembly: AssemblyKind = eqx.field(static=True)
    field_name: str = eqx.field(static=True)

    def __init__(
        self,
        cover: SubdomainCover,
        family: LocalFieldFamily,
        functions: Mapping[str, DomainFunction],
        terms: AbstractScalarTerm
        | ScopedFunctionalTerm
        | Sequence[AbstractScalarTerm | ScopedFunctionalTerm],
        /,
        *,
        assembly: AssemblyKind,
        field_name: str,
        evaluation_terms: AbstractScalarTerm
        | ScopedFunctionalTerm
        | Sequence[AbstractScalarTerm | ScopedFunctionalTerm] = (),
        enforcement: EnforcementProgram | None = None,
        collocation_key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be a SubdomainCover.")
        if not isinstance(family, LocalFieldFamily):
            raise TypeError("family must be a LocalFieldFamily.")
        if family.cover.cover_id != cover.cover_id:
            raise ValueError("family and problem cover identities must match.")
        if assembly not in ("partition-of-unity", "broken"):
            raise ValueError("Unknown decomposition field assembly.")
        name = str(field_name)
        if not name:
            raise ValueError("field_name must be non-empty.")
        functions_ = frozendict(functions)
        if not functions_:
            raise ValueError("A decomposition problem requires bound functions.")
        if any(not isinstance(value, DomainFunction) for value in functions_.values()):
            raise TypeError("functions must map names to DomainFunction objects.")
        if enforcement is not None and not isinstance(enforcement, EnforcementProgram):
            raise TypeError("enforcement must be an EnforcementProgram or None.")
        if assembly == "broken" and enforcement is not None:
            raise ValueError(
                "Broken decomposition currently uses residual boundary conditions; "
                "global hard enforcement is unsupported."
            )

        self.cover = cover
        self.family = family
        self.functions = functions_
        self.terms = _scoped_terms(terms)
        self.evaluation_terms = _scoped_terms(evaluation_terms)
        self.enforcement = enforcement
        self.collocation_key = collocation_key
        self.assembly = assembly
        self.field_name = name

    @classmethod
    def partition_of_unity(
        cls,
        family: LocalFieldFamily,
        terms: AbstractScalarTerm
        | ScopedFunctionalTerm
        | Sequence[AbstractScalarTerm | ScopedFunctionalTerm],
        /,
        *,
        field_name: str | None = None,
        additional_functions: Mapping[str, DomainFunction] | None = None,
        evaluation_terms: AbstractScalarTerm
        | ScopedFunctionalTerm
        | Sequence[AbstractScalarTerm | ScopedFunctionalTerm] = (),
        enforcement: EnforcementProgram | None = None,
        collocation_key: Key[Array, ""] = DOC_KEY0,
    ) -> FunctionalDecompositionProblem:
        if not isinstance(family, LocalFieldFamily):
            raise TypeError("family must be a LocalFieldFamily.")
        name = family.field_id if field_name is None else str(field_name)
        functions = {} if additional_functions is None else dict(additional_functions)
        if name in functions:
            raise ValueError(f"Additional functions already contain {name!r}.")
        functions[name] = partition_of_unity_field(family)
        return cls(
            family.cover,
            family,
            functions,
            terms,
            assembly="partition-of-unity",
            field_name=name,
            evaluation_terms=evaluation_terms,
            enforcement=enforcement,
            collocation_key=collocation_key,
        )

    @classmethod
    def broken(
        cls,
        family: LocalFieldFamily,
        terms: AbstractScalarTerm
        | ScopedFunctionalTerm
        | Sequence[AbstractScalarTerm | ScopedFunctionalTerm],
        /,
        *,
        additional_functions: Mapping[str, DomainFunction] | None = None,
        evaluation_terms: AbstractScalarTerm
        | ScopedFunctionalTerm
        | Sequence[AbstractScalarTerm | ScopedFunctionalTerm] = (),
        collocation_key: Key[Array, ""] = DOC_KEY0,
    ) -> FunctionalDecompositionProblem:
        if not isinstance(family, LocalFieldFamily):
            raise TypeError("family must be a LocalFieldFamily.")
        functions = family.solver_functions()
        for name, function in (
            () if additional_functions is None else additional_functions.items()
        ):
            if name in functions:
                raise ValueError(f"Duplicate decomposition function {name!r}.")
            functions[name] = function
        return cls(
            family.cover,
            family,
            functions,
            terms,
            assembly="broken",
            field_name=family.field_id,
            evaluation_terms=evaluation_terms,
            collocation_key=collocation_key,
        )

    @property
    def training_terms(self) -> tuple[AbstractScalarTerm, ...]:
        return tuple(value.term for value in self.terms)

    @property
    def diagnostic_terms(self) -> tuple[AbstractScalarTerm, ...]:
        return tuple(value.term for value in self.evaluation_terms)

    def local_ref(self, patch_id: str, /) -> LocalFieldRef:
        return self.family.ref(patch_id)


def localize_residual_penalty(
    term: ResidualPenalty,
    patch: SubdomainPatch,
    source: Any,
    /,
    *,
    on: DomainComponent | None = None,
) -> ScopedFunctionalTerm:
    """Bind one ambient residual penalty to a local ownership component."""
    if not isinstance(term, ResidualPenalty):
        raise TypeError("term must be a ResidualPenalty.")
    condition = localize_residual(term.condition, patch, on=on)
    density = None if term.density is None else patch.restrict(term.density)
    localized = ResidualPenalty(
        condition,
        source,
        scale=term.scale,
        density=density,
        blocks=term.blocks,
        label=term.label,
        data_accuracy_eps=term.data_accuracy_eps,
    )
    return ScopedFunctionalTerm(localized, PatchScope(patch.patch_id))


__all__ = [
    "FunctionalDecompositionProblem",
    "LocalFieldRef",
    "GlobalScope",
    "PairScope",
    "PatchScope",
    "ScopedFunctionalTerm",
    "localize_residual_penalty",
]
