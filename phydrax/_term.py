#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax.domain import DomainFunction

from ._doc import DOC_KEY0
from ._frozendict import frozendict
from ._strict import AbstractAttribute, StrictModule


class TermEvaluation(StrictModule):
    """Validated scalar term value with fixed-structure diagnostics."""

    value: Array
    diagnostics: Any

    def __init__(self, value: Any, /, *, diagnostics: Any = None):
        scalar = jnp.asarray(value)
        if scalar.shape != ():
            raise ValueError(f"Scalar terms must return shape (), got {scalar.shape}.")
        if jnp.iscomplexobj(scalar):
            raise TypeError("Scalar terms must return a real value.")
        self.value = scalar.reshape(())
        self.diagnostics = frozendict() if diagnostics is None else diagnostics


class AbstractScalarTerm(StrictModule):
    """A real scalar term evaluated from the solver's current domain functions."""

    label: AbstractAttribute[str | None]

    @abstractmethod
    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        raise NotImplementedError

    def evaluate(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        step: int | Array | None = None,
        **kwargs: Any,
    ) -> TermEvaluation:
        """Evaluate this term through the shared scalar validation path."""
        return evaluate(self, functions, key=key, step=step, **kwargs)


class AbstractSamplingTerm(AbstractScalarTerm):
    """A scalar term with an explicitly materializable evaluation batch."""

    @abstractmethod
    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> Any:
        raise NotImplementedError


class AbstractEvaluatedScalarTerm(AbstractScalarTerm):
    """A scalar term that supplies structured diagnostics with its value."""

    @abstractmethod
    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> TermEvaluation:
        raise NotImplementedError


def evaluate(
    term: AbstractScalarTerm,
    functions: Mapping[str, DomainFunction],
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    step: int | Array | None = None,
    **kwargs: Any,
) -> TermEvaluation:
    """Evaluate any scalar term through one shape and dtype validation path."""
    if not isinstance(term, AbstractScalarTerm):
        raise TypeError("evaluate expects an AbstractScalarTerm.")
    if isinstance(term, AbstractEvaluatedScalarTerm):
        result = term.term_evaluation(
            functions,
            key=key,
            iter_=step,
            **kwargs,
        )
        if not isinstance(result, TermEvaluation):
            raise TypeError("term_evaluation must return TermEvaluation.")
        return result
    return TermEvaluation(term.loss(functions, key=key, iter_=step, **kwargs))


__all__ = [
    "AbstractEvaluatedScalarTerm",
    "AbstractSamplingTerm",
    "AbstractScalarTerm",
    "TermEvaluation",
    "evaluate",
]
