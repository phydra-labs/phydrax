#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._strict import AbstractAttribute, StrictModule
from ..domain._components import DomainComponent, DomainComponentUnion
from ..domain._function import DomainFunction
from ..domain._structure import (
    CoordSeparableBatch,
    NumPoints,
    PointsBatch,
    ProductStructure,
)


class AbstractConstraint(StrictModule):
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

    component: AbstractAttribute[DomainComponent | DomainComponentUnion]
    structure: AbstractAttribute[ProductStructure]
    coord_separable: AbstractAttribute[Mapping[str, int | Sequence[int]] | None]
    dense_structure: AbstractAttribute[ProductStructure | None]
    num_points: AbstractAttribute[NumPoints | tuple[Any, ...]]
    sampler: AbstractAttribute[str]
    over: AbstractAttribute[str | tuple[str, ...] | None]
    reduction: AbstractAttribute[Literal["mean", "integral"]]

    @abstractmethod
    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...]:
        raise NotImplementedError
