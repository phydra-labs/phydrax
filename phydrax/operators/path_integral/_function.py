#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ...domain._domain import _AbstractDomain
from ...domain._function import DomainFunction
from ._discretization import PathDiscretization
from ._euclidean import euclidean_kernel
from ._potential import PotentialLike


class _EuclideanKernelCallable(StrictModule):
    potential: PotentialLike | None
    slicing: PathDiscretization
    mass: ArrayLike
    hbar: ArrayLike
    num_paths: int
    chunk_size: int | None
    position_var: str
    time_var: str

    def __init__(
        self,
        potential: PotentialLike | None,
        slicing: PathDiscretization,
        mass: ArrayLike,
        hbar: ArrayLike,
        num_paths: int,
        chunk_size: int | None,
        position_var: str,
        time_var: str,
        /,
    ):
        self.potential = potential
        self.slicing = slicing
        self.mass = mass
        self.hbar = hbar
        self.num_paths = int(num_paths)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.position_var = position_var
        self.time_var = time_var

    def __call__(
        self,
        x0: ArrayLike,
        x1: ArrayLike,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        return euclidean_kernel(
            self.potential,
            x0,
            x1,
            slicing=self.slicing,
            mass=self.mass,
            hbar=self.hbar,
            num_paths=self.num_paths,
            chunk_size=self.chunk_size,
            position_var=self.position_var,
            time_var=self.time_var,
            key=key,
        ).value


def euclidean_kernel_function(
    domain: _AbstractDomain,
    potential: PotentialLike | None,
    /,
    *,
    start_var: str,
    end_var: str,
    slicing: PathDiscretization,
    mass: ArrayLike = 1.0,
    hbar: ArrayLike = 1.0,
    num_paths: int,
    chunk_size: int | None = None,
    position_var: str = "q",
    time_var: str = "t",
    metadata: dict[str, Any] | None = None,
) -> DomainFunction:
    """Wrap a Euclidean kernel estimate as an endpoint ``DomainFunction``.

    ``slicing`` fixes the Euclidean time interval. ``start_var`` and ``end_var``
    select the two endpoint factors from ``domain``. The returned scalar function
    composes with Phydrax quantum operators, constraints, and sampled integrals.
    """
    if start_var == end_var:
        raise ValueError("start_var and end_var must be distinct labels.")
    for label in (start_var, end_var):
        if label not in domain.labels:
            raise ValueError(
                f"Endpoint label {label!r} is absent from domain {domain.labels!r}."
            )
    func = _EuclideanKernelCallable(
        potential,
        slicing,
        mass,
        hbar,
        num_paths,
        chunk_size,
        position_var,
        time_var,
    )
    return DomainFunction(
        domain=domain,
        deps=(start_var, end_var),
        func=func,
        metadata=metadata,
    )


__all__ = ["euclidean_kernel_function"]
