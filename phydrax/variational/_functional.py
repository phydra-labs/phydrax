#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._local import LocalIntegralTerm


class Functional(StrictModule):
    """Representation-independent ordered scalar functional."""

    identifier: str = eqx.field(static=True)
    terms: tuple[LocalIntegralTerm, ...]
    variable_fields: tuple[str, ...] = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        identifier: str,
        terms: Sequence[LocalIntegralTerm],
        /,
        *,
        variable_fields: Sequence[str],
    ):
        identifier_ = str(identifier)
        if not identifier_:
            raise ValueError("Functional identifier must be non-empty.")
        terms_ = tuple(terms)
        if not terms_ or any(not isinstance(term, LocalIntegralTerm) for term in terms_):
            raise TypeError("Functional terms must contain LocalIntegralTerm values.")
        term_identifiers = tuple(term.identifier for term in terms_)
        if len(set(term_identifiers)) != len(term_identifiers):
            raise ValueError("Functional term identifiers must be unique.")
        variables = tuple(str(name) for name in variable_fields)
        if not variables or any(not name for name in variables):
            raise ValueError("variable_fields must contain non-empty names.")
        if len(set(variables)) != len(variables):
            raise ValueError("variable_fields must not contain duplicates.")
        used_fields = {field.field_name for term in terms_ for field in term.fields}
        missing = tuple(name for name in variables if name not in used_fields)
        if missing:
            raise ValueError(
                "Functional variable fields must occur in at least one term; "
                f"missing={missing}."
            )
        self.identifier = identifier_
        self.terms = terms_
        self.variable_fields = variables
        self.functional_id = canonical_fingerprint(
            {
                "identifier": identifier_,
                "term_ids": [term.term_id for term in terms_],
                "variable_fields": list(variables),
            }
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return all semantic fields in first-occurrence order."""
        return tuple(
            dict.fromkeys(
                field.field_name for term in self.terms for field in term.fields
            )
        )

    @property
    def region_names(self) -> tuple[str, ...]:
        """Return semantic regions in first-occurrence order."""
        return tuple(dict.fromkeys(term.region for term in self.terms))


class FunctionalEvaluation(StrictModule):
    """Ordered scalar functional value with backend diagnostics."""

    value: Array
    term_values: tuple[Array, ...]
    diagnostics: tuple[Any, ...]
    functional_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: Array,
        term_values: Sequence[Array],
        /,
        *,
        diagnostics: Sequence[Any] = (),
        functional_id: str,
        binding_id: str,
    ):
        value_ = jnp.asarray(value)
        terms_ = tuple(jnp.asarray(term) for term in term_values)
        if value_.shape != () or any(term.shape != () for term in terms_):
            raise ValueError("Functional values must be scalar arrays.")
        if jnp.iscomplexobj(value_) or any(jnp.iscomplexobj(term) for term in terms_):
            raise TypeError("Functional values must be real.")
        functional_id_ = str(functional_id)
        binding_id_ = str(binding_id)
        if not functional_id_ or not binding_id_:
            raise ValueError("Functional and binding identifiers must be non-empty.")
        diagnostics_ = tuple(diagnostics)
        if diagnostics_ and len(diagnostics_) != len(terms_):
            raise ValueError("Functional diagnostics must align with term values.")
        self.value = value_.reshape(())
        self.term_values = tuple(term.reshape(()) for term in terms_)
        self.diagnostics = diagnostics_
        self.functional_id = functional_id_
        self.binding_id = binding_id_


__all__ = ["Functional", "FunctionalEvaluation"]
