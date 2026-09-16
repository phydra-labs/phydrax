#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Minimal configuration-dependent Dirac contract for pseudofermion actions."""

from __future__ import annotations

import abc

from jaxtyping import ArrayLike

from ...linalg import AbstractLinearOperator


class AbstractPseudofermionDiracOperator(AbstractLinearOperator):
    """Linear Dirac-like operator whose numeric link field can be replaced.

    The contract intentionally does not require gamma matrices, gamma5
    Hermiticity, checkerboarding, canonical spinors, or a Wilson diagonal. Those
    belong to concrete lattice-fermion formulations. Pseudofermion execution
    requires only a differentiable endomorphism source, an adjoint action, and a
    structural ``with_links`` replacement.
    """

    @abc.abstractmethod
    def with_links(self, links: ArrayLike, /) -> AbstractPseudofermionDiracOperator:
        raise NotImplementedError


__all__ = ["AbstractPseudofermionDiracOperator"]
