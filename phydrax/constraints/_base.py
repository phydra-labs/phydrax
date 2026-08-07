#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

from jaxtyping import Array, Key

from phydrax.domain import (
    ComponentSum,
    DomainComponent,
    DomainFunction,
    GridBatch,
    PointBatch,
)
from phydrax.domain.graph import GraphBatch

from .._doc import DOC_KEY0
from .._objective import AbstractObjectiveTerm
from .._strict import AbstractAttribute


class AbstractConstraint(AbstractObjectiveTerm):
    r"""Common interface for all soft/penalty constraints.

    A constraint is an objective term $\ell(\theta)$ evaluated from a set of
    `DomainFunction` fields (often parameterized by neural-network parameters
    $\theta$). Solvers typically minimize a weighted sum of constraints:

    $$
    L(\theta) = \sum_i \ell_i(\theta).
    $$
    """

    constraint_vars: AbstractAttribute[tuple[str, ...]]
    weight: AbstractAttribute[Array]
    label: AbstractAttribute[str | None]

    @abstractmethod
    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        **kwargs: Any,
    ) -> Array:
        raise NotImplementedError


class AbstractSamplingConstraint(AbstractConstraint):
    r"""Base class for constraints that sample from a domain component.

    Sampling constraints are defined over a `DomainComponent` (or union) and draw
    point batches in order to estimate integrals/means of a residual over the domain.
    """

    component: AbstractAttribute[DomainComponent | ComponentSum]
    over: AbstractAttribute[str | tuple[str, ...] | None]
    reduction: AbstractAttribute[Literal["mean", "integral"]]

    @abstractmethod
    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointBatch | GridBatch | GraphBatch | tuple[PointBatch, ...]:
        raise NotImplementedError
