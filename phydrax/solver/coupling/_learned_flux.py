#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Learned monotone, dissipative interface conductance for ``ConservativeFluxLaw``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._model import AbstractArrayModel
from ..._trainable import NonTrainableState
from ..._validation import finite_real_scalar, positive_integer
from ._laws import AbstractInterfaceFlux
from ._parameters import RuntimeInput


_CONVEX_CAPABILITY = "input-convex"


@final
class MonotoneInterfaceConductance(AbstractInterfaceFlux, NonTrainableState):
    """Learned contact conductance that is monotone and dissipative by construction.

    With the temperature jump ``d = u_minus - u_plus``, the heat leaving the minus
    side is ``q(d) = h d + phi'(d) - phi'(0)``, where ``phi`` is a learned scalar
    potential read at every solve from the runtime input ``response``, which
    preparation requires to be the target of a refresh ``ParameterBinding`` (so
    the model passes its port and model/discretization authority admission and a
    raw argument at that key is refused), and
    ``h >= 0`` a declared baseline conductance. The law requires ``phi`` to carry
    the model's ``input-convex`` construction certificate: convexity makes ``q``
    nondecreasing with ``q(0) = 0`` for every parameter value, so the exchange is
    monotone and dissipative (``d q(d) >= 0``, heat flows from hot to cold). A
    response without that certificate is refused when the flux is evaluated. The
    minus conormal flux is ``-q``. The density is not polynomial in the traces;
    ``quadrature_degree`` is the declared exactness degree of the common
    quadrature, an approximation of the interface integral that is part of the
    law's discretization.
    """

    response: RuntimeInput
    baseline: Array
    quadrature_degree: int = eqx.field(static=True)

    def __init__(
        self,
        response: RuntimeInput,
        /,
        *,
        baseline: float = 0.0,
        quadrature_degree: int = 4,
    ) -> None:
        if not isinstance(response, RuntimeInput):
            raise TypeError("response must be a RuntimeInput.")
        value = finite_real_scalar(baseline, "baseline")
        if value < 0.0:
            raise ValueError("baseline must be a nonnegative conductance.")
        self.response = response
        self.baseline = jnp.asarray(value, dtype=jnp.float64)
        self.quadrature_degree = positive_integer(quadrature_degree, "quadrature_degree")

    @property
    def affine(self) -> bool:
        return False

    @property
    def trace_degree(self) -> int:
        return self.quadrature_degree

    @property
    def runtime_inputs(self) -> tuple[RuntimeInput, ...]:
        return (self.response,)

    def potential(self, args: object, /) -> AbstractArrayModel:
        """The certified convex potential bound at ``response`` in ``args``."""
        if not isinstance(args, Mapping):
            raise TypeError(
                "A learned interface conductance reads its response from the coupled "
                "runtime arguments."
            )
        owner = args[self.response.component]
        if not isinstance(owner, Mapping) or self.response.name not in owner:
            raise ValueError(
                f"Runtime input {self.response.name!r} of component "
                f"{self.response.component!r} binds no learned interface response."
            )
        model = owner[self.response.name]
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("The learned interface response must be an array model.")
        certificates = model.model_execution_contract().certificates
        if _CONVEX_CAPABILITY not in {record[0] for record in certificates}:
            raise ValueError(
                "The interface law requires a monotone response: the learned potential "
                "carries no input-convex construction certificate, so its derivative "
                "need not be nondecreasing and the exchange need not be dissipative."
            )
        return model

    def heat_flow(self, jump: Array, args: object, /) -> Array:
        """Heat ``q(d)`` leaving the minus side at temperature jumps ``d``."""
        model = self.potential(args)

        def potential(value: Array) -> Array:
            output = jnp.asarray(model(value))
            if output.shape != ():
                raise ValueError("The learned interface potential must be scalar.")
            return output

        slope = jax.grad(potential)
        zero = slope(jnp.zeros((), dtype=jump.dtype))
        return self.baseline * jump + jax.vmap(slope)(jump) - zero

    def evaluate(
        self,
        minus: Array,
        plus: Array,
        points: Array,
        normals: Array,
        args: object,
        /,
    ) -> Array:
        del points, normals
        return -self.heat_flow(minus - plus, args)


__all__ = ["MonotoneInterfaceConductance"]
