#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._objective import AbstractSamplingObjectiveTerm
from ..constraints._sampling_spec import (
    CoordSamplingMap,
    parse_sampling_num_points,
    SamplingNumPoints,
)
from ..domain._components import DomainComponent, DomainComponentUnion
from ..domain._function import DomainFunction
from ..domain._structure import CoordSeparableBatch, PointsBatch, ProductStructure
from ..operators.integral._batch_ops import integral


class IntegralFunctional(AbstractSamplingObjectiveTerm):
    r"""A raw signed integral objective.

    Given an integrand $f$ on a component $\Omega_{\mathrm{comp}}$, this term
    contributes

    $$
    \mathcal J = w\int_{\Omega_{\mathrm{comp}}} f(z)\,d\mu(z)
    $$

    directly to the solver objective. Unlike an integral equality constraint, the
    integral is not compared with a target and is not squared.
    """

    objective_vars: tuple[str, ...]
    component: DomainComponent | DomainComponentUnion
    structure: ProductStructure
    coord_sampling: CoordSamplingMap | None
    dense_structure: ProductStructure | None
    num_points: Any
    sampler: str
    weight: Array
    label: str | None
    over: str | tuple[str, ...] | None
    sampling_mode: Literal["resample", "fixed"]
    fixed_batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
    integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction] | DomainFunction

    def __init__(
        self,
        *,
        component: DomainComponent | DomainComponentUnion,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
        | DomainFunction,
        num_points: SamplingNumPoints,
        structure: ProductStructure,
        dense_structure: ProductStructure | None = None,
        objective_vars: Sequence[str] | None = None,
        sampler: str = "latin_hypercube",
        weight: ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
        sampling_mode: Literal["resample", "fixed"] = "resample",
        fixed_batch: (
            PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
        ) = None,
        fixed_batch_key: Key[Array, ""] = DOC_KEY0,
    ):
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.component = component
        if not isinstance(integrand, DomainFunction) and not callable(integrand):
            raise TypeError("integrand must be a DomainFunction or callable.")
        self.integrand = integrand
        self.structure = structure
        dense_num_points, coord_sampling, dense_structure_out = parse_sampling_num_points(
            component,
            num_points=num_points,
            structure=structure,
            dense_structure=dense_structure,
        )
        self.num_points = dense_num_points
        self.coord_sampling = coord_sampling
        self.dense_structure = dense_structure_out
        self.sampler = str(sampler)
        self.weight = jnp.asarray(weight, dtype=float).reshape(())
        self.label = None if label is None else str(label)
        self.over = over

        sampling_mode_ = str(sampling_mode).lower()
        if sampling_mode_ not in ("resample", "fixed"):
            raise ValueError("sampling_mode must be either 'resample' or 'fixed'.")
        if sampling_mode_ == "fixed":
            self.sampling_mode = "fixed"
            self.fixed_batch = (
                self._sample_once(key=fixed_batch_key)
                if fixed_batch is None
                else fixed_batch
            )
        else:
            self.sampling_mode = "resample"
            if fixed_batch is not None:
                raise ValueError("fixed_batch is only valid when sampling_mode='fixed'.")
            self.fixed_batch = None

    @classmethod
    def from_operator(
        cls,
        *,
        component: DomainComponent | DomainComponentUnion,
        operator: Callable[..., DomainFunction],
        objective_vars: str | Sequence[str],
        num_points: SamplingNumPoints,
        structure: ProductStructure,
        dense_structure: ProductStructure | None = None,
        sampler: str = "latin_hypercube",
        weight: ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
        sampling_mode: Literal["resample", "fixed"] = "resample",
        fixed_batch: (
            PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
        ) = None,
        fixed_batch_key: Key[Array, ""] = DOC_KEY0,
    ) -> "IntegralFunctional":
        """Build an integral functional from an operator on named solver fields."""
        vars_tuple = (
            (objective_vars,)
            if isinstance(objective_vars, str)
            else tuple(objective_vars)
        )

        def integrand(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return operator(*(functions[name] for name in vars_tuple))

        return cls(
            component=component,
            integrand=integrand,
            num_points=num_points,
            structure=structure,
            dense_structure=dense_structure,
            objective_vars=vars_tuple,
            sampler=sampler,
            weight=weight,
            label=label,
            over=over,
            sampling_mode=sampling_mode,
            fixed_batch=fixed_batch,
            fixed_batch_key=fixed_batch_key,
        )

    def _sample_once(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...]:
        if self.coord_sampling is not None:
            if isinstance(self.component, DomainComponentUnion):
                raise ValueError(
                    "coord-separable sampling is not supported for DomainComponentUnion."
                )
            return self.component.sample_coord_separable(
                self.coord_sampling,
                num_points=self.num_points,
                dense_structure=self.dense_structure,
                sampler=self.sampler,
                key=key,
            )
        return self.component.sample(
            self.num_points,
            structure=self.structure,
            sampler=self.sampler,
            key=key,
        )

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...]:
        """Sample the configured integration component."""
        if self.sampling_mode == "fixed":
            if self.fixed_batch is None:
                raise ValueError("sampling_mode='fixed' requires fixed_batch to be set.")
            return self.fixed_batch
        return self._sample_once(key=key)

    def _integrand_function(
        self, functions: Mapping[str, DomainFunction], /
    ) -> DomainFunction:
        value = self.integrand
        integrand = value if isinstance(value, DomainFunction) else value(functions)
        if not isinstance(integrand, DomainFunction):
            raise TypeError(
                "IntegralFunctional integrand must produce a DomainFunction; "
                f"got {type(integrand).__name__}."
            )
        return integrand

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None = None,
        **kwargs: Any,
    ) -> Array:
        """Estimate and return the raw signed integral."""
        integrand = self._integrand_function(functions)
        batch_ = self.sample(key=key) if batch is None else batch
        out = integral(
            integrand,
            batch_,
            component=self.component,
            over=self.over,
            key=key,
            **kwargs,
        )
        if not isinstance(out, cx.Field):
            raise TypeError("Expected integral to return a coordax.Field.")
        if out.dims != ():
            raise ValueError(
                f"IntegralFunctional must reduce to a scalar Field, got dims={out.dims}."
            )
        value = jnp.asarray(out.data).reshape(())
        if jnp.iscomplexobj(value):
            raise TypeError(
                "IntegralFunctional requires a real scalar integrand; "
                "use real_part(...) to select an explicitly real objective."
            )
        return self.weight * value


__all__ = ["IntegralFunctional"]
