#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, FunctionLinearOperator, OperatorProperties
from ...sparse import RelationAccumulation, scatter_local
from ...typing import checked, parse


class FactorizedVirtualElementOperator(StrictModule, NonTrainableState):
    """Bucketed C-transpose G C actions plus projector-kernel stabilization."""

    coefficient_maps: tuple[Array, ...]
    polynomial_matrices: tuple[Array, ...]
    stabilization_matrices: tuple[Array, ...]
    gathers: tuple[Array, ...]
    global_size: int = eqx.field(static=True)
    accumulation: RelationAccumulation = eqx.field(static=True)
    properties: OperatorProperties
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        coefficient_maps: tuple[ArrayLike, ...],
        polynomial_matrices: tuple[ArrayLike, ...],
        stabilization_matrices: tuple[ArrayLike, ...],
        gathers: tuple[ArrayLike, ...],
        global_size: int,
        /,
        *,
        accumulation: RelationAccumulation = "fast",
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
    ) -> None:
        coefficients = tuple(jnp.asarray(value) for value in coefficient_maps)
        polynomials = tuple(jnp.asarray(value) for value in polynomial_matrices)
        stabilizations = tuple(jnp.asarray(value) for value in stabilization_matrices)
        routes = tuple(jnp.asarray(value, dtype=jnp.int32) for value in gathers)
        if not coefficients or not (
            len(coefficients) == len(polynomials) == len(stabilizations) == len(routes)
        ):
            raise ValueError(
                f"FactorizedVirtualElementOperator.buckets must be nonempty and aligned; got coefficients={len(coefficients)}, polynomials={len(polynomials)}, stabilizations={len(stabilizations)}, routes={len(routes)}"
            )
        for coefficient, polynomial, stabilization, route in zip(
            coefficients, polynomials, stabilizations, routes, strict=True
        ):
            if coefficient.ndim != 3 or polynomial.ndim != 3 or stabilization.ndim != 3:
                raise ValueError(
                    f"FactorizedVirtualElementOperator.factors must be rank-three arrays; got coefficient={coefficient.shape}, polynomial={polynomial.shape}, stabilization={stabilization.shape}"
                )
            if coefficient.shape[0] != route.shape[0]:
                raise ValueError(
                    f"FactorizedVirtualElementOperator.gathers must share coefficient cells; got coefficients={coefficient.shape[0]}, gathers={route.shape[0]}"
                )
            if (
                coefficient.shape[1] != polynomial.shape[1]
                or polynomial.shape[1] != polynomial.shape[2]
            ):
                raise ValueError(
                    f"FactorizedVirtualElementOperator.polynomial_matrices must be square with matching coefficient dimension; got coefficient={coefficient.shape}, polynomial={polynomial.shape}"
                )
            if coefficient.shape[2] != route.shape[1] or stabilization.shape[1:] != (
                route.shape[1],
                route.shape[1],
            ):
                raise ValueError(
                    f"FactorizedVirtualElementOperator.local_factors must match gather width; got coefficient={coefficient.shape}, stabilization={stabilization.shape}, gather={route.shape}"
                )
        accumulation_ = parse(
            accumulation,
            RelationAccumulation,
            "FactorizedVirtualElementOperator.accumulation",
        )
        size = int(global_size)
        if size <= 0:
            raise ValueError(
                f"FactorizedVirtualElementOperator.global_size must be positive; got {size}"
            )
        properties_ = OperatorProperties() if properties is None else properties
        self.coefficient_maps = coefficients
        self.polynomial_matrices = polynomials
        self.stabilization_matrices = stabilizations
        self.gathers = routes
        self.global_size = size
        self.accumulation = accumulation_
        self.properties = properties_
        self.operator_id = (
            canonical_fingerprint(
                {
                    "kind": "factorized-virtual-element-operator",
                    "bucket_shapes": [list(value.shape) for value in coefficients],
                    "global_size": size,
                    "accumulation": accumulation_,
                }
            )
            if operator_id is None
            else str(operator_id)
        )

    def mv(self, value: ArrayLike, /) -> Array:
        state = jnp.asarray(value)
        if state.shape != (self.global_size,):
            raise ValueError(
                f"FactorizedVirtualElementOperator.value must match global size; got shape={state.shape}, global_size={self.global_size}"
            )
        result = jnp.zeros_like(state)
        for coefficient, polynomial, stabilization, gather in zip(
            self.coefficient_maps,
            self.polynomial_matrices,
            self.stabilization_matrices,
            self.gathers,
            strict=True,
        ):
            local = state[gather]
            projected = ein.contract("cai,ci->ca", coefficient, local)
            polynomial_action = ein.contract("cab,cb->ca", polynomial, projected)
            consistent = ein.contract("cai,ca->ci", coefficient, polynomial_action)
            stabilized = ein.contract("cij,cj->ci", stabilization, local)
            result = scatter_local(
                result,
                gather,
                consistent + stabilized,
                self.accumulation,
            )
        return result

    def transpose_mv(self, value: ArrayLike, /) -> Array:
        state = jnp.asarray(value)
        if state.shape != (self.global_size,):
            raise ValueError(
                f"FactorizedVirtualElementOperator.value must match global size; got shape={state.shape}, global_size={self.global_size}"
            )
        result = jnp.zeros_like(state)
        for coefficient, polynomial, stabilization, gather in zip(
            self.coefficient_maps,
            self.polynomial_matrices,
            self.stabilization_matrices,
            self.gathers,
            strict=True,
        ):
            local = state[gather]
            projected = ein.contract("cai,ci->ca", coefficient, local)
            polynomial_action = ein.contract("cba,cb->ca", polynomial, projected)
            consistent = ein.contract("cai,ca->ci", coefficient, polynomial_action)
            stabilized = ein.contract("cji,cj->ci", stabilization, local)
            result = scatter_local(
                result, gather, consistent + stabilized, self.accumulation
            )
        return result

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        dtype = self.coefficient_maps[0].dtype
        space = ArraySpace((self.global_size,), dtype=dtype)
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            transpose_action=self.transpose_mv,
            properties=self.properties,
            operator_id=self.operator_id,
        )

    def local_tensors(self, /) -> tuple[Array, ...]:
        """Assembled `C^T G C + S` cell tensors, one per bucket, on `gathers`.

        Only the tensors are runtime data; the gathers stay the prepared routes,
        so coordinate storage can be rebuilt inside traced code without host
        route validation or fingerprinting.
        """
        return tuple(
            ein.contract("cai,cab,cbj->cij", coefficient, polynomial, coefficient)
            + stabilization
            for coefficient, polynomial, stabilization in zip(
                self.coefficient_maps,
                self.polynomial_matrices,
                self.stabilization_matrices,
                strict=True,
            )
        )


__all__ = ["FactorizedVirtualElementOperator"]
