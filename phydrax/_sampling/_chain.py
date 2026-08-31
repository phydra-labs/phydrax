#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule


class AbstractChainSampleResult(StrictModule):
    """Correlated chain-by-draw samples with stable provenance semantics."""

    samples: AbstractAttribute[PyTree[Array]]

    @property
    @abstractmethod
    def num_chains(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_draws(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def chain_provenance(self) -> str:
        raise NotImplementedError


__all__ = ["AbstractChainSampleResult"]
