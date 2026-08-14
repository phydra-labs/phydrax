#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
from jaxtyping import ArrayLike

from phydrax.domain import DomainFunction

from ..._strict import StrictModule
from ._dynamics import von_neumann_residual
from ._validation import join_function_arguments, validate_matrix_value


_ADEngine = Literal["auto", "reverse", "forward", "jvp"]


def _collapse_operator_tuple(
    collapse_operators: DomainFunction | Sequence[DomainFunction],
    /,
) -> tuple[DomainFunction, ...]:
    if isinstance(collapse_operators, DomainFunction):
        return (collapse_operators,)
    if not isinstance(collapse_operators, Sequence):
        raise TypeError(
            "collapse_operators must be a DomainFunction or a sequence of "
            "DomainFunctions."
        )
    operators = tuple(collapse_operators)
    for index, operator in enumerate(operators):
        if not isinstance(operator, DomainFunction):
            raise TypeError(
                "collapse_operators must contain only DomainFunctions; "
                f"item {index} is {type(operator).__name__}."
            )
    return operators


class _LindbladDissipatorCallable(StrictModule):
    density: DomainFunction
    collapse_operators: tuple[DomainFunction, ...]
    density_positions: tuple[int, ...]
    collapse_positions: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        density: DomainFunction,
        collapse_operators: tuple[DomainFunction, ...],
        density_positions: tuple[int, ...],
        collapse_positions: tuple[tuple[int, ...], ...],
    ):
        self.density = density
        self.collapse_operators = collapse_operators
        self.density_positions = density_positions
        self.collapse_positions = collapse_positions

    def __call__(self, *args, key=None, **kwargs):
        density_args = tuple(args[index] for index in self.density_positions)
        density = validate_matrix_value(
            self.density.func(*density_args, key=key, **kwargs),
            role="Lindblad density operator",
        )
        result = jnp.zeros_like(density)
        for index, (operator, positions) in enumerate(
            zip(self.collapse_operators, self.collapse_positions, strict=True)
        ):
            operator_args = tuple(args[position] for position in positions)
            value = validate_matrix_value(
                operator.func(*operator_args, key=key, **kwargs),
                role=f"collapse operator {index}",
            )
            if value.shape != density.shape:
                raise ValueError(
                    "Collapse-operator and density dimensions must match; "
                    f"operator {index} has shape {value.shape}, density has "
                    f"shape {density.shape}."
                )
            adjoint = jnp.conj(value.T)
            rate_operator = adjoint @ value
            result = (
                result
                + value @ density @ adjoint
                - 0.5 * (rate_operator @ density + density @ rate_operator)
            )
        return result


def lindblad_dissipator(
    density: DomainFunction,
    collapse_operators: DomainFunction | Sequence[DomainFunction],
    /,
) -> DomainFunction:
    r"""Construct the Lindblad dissipator for a density operator.

    The pointwise operation is

    $$
    \mathcal D(\rho)=\sum_k\left(
      L_k\rho L_k^\dagger
      -\frac12\{L_k^\dagger L_k,\rho\}_+
    \right).
    $$

    A single collapse-operator ``DomainFunction`` or a sequence may be supplied.
    Rates belong in the operators themselves, conventionally as $L_k=\sqrt{\gamma_k}C_k$.
    An empty sequence is valid and produces a zero matrix with the density shape.
    """
    if not isinstance(density, DomainFunction):
        raise TypeError("lindblad_dissipator density must be a DomainFunction.")
    operators = _collapse_operator_tuple(collapse_operators)
    domain, deps, promoted, positions = join_function_arguments(density, *operators)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_LindbladDissipatorCallable(
            promoted[0],
            promoted[1:],
            positions[0],
            positions[1:],
        ),
        metadata={},
    )


def lindblad_residual(
    density: DomainFunction,
    hamiltonian: DomainFunction,
    collapse_operators: DomainFunction | Sequence[DomainFunction],
    /,
    *,
    time_var: str = "t",
    hbar: ArrayLike = 1.0,
    mode: Literal["reverse", "forward"] = "reverse",
    ad_engine: _ADEngine = "auto",
) -> DomainFunction:
    r"""Construct the closed-plus-dissipative Lindblad evolution residual.

    The returned field is

    $$
    r_\rho=\partial_t\rho
      -\frac{[H,\rho]}{i\hbar}
      -\mathcal D(\rho).
    $$

    ``density`` and ``hamiltonian`` must be square matrix-valued
    ``DomainFunction`` objects. Collapse operators may be constant or depend on any
    compatible labeled coordinates.
    """
    operators = _collapse_operator_tuple(collapse_operators)
    coherent_residual = von_neumann_residual(
        density,
        hamiltonian,
        time_var=time_var,
        hbar=hbar,
        mode=mode,
        ad_engine=ad_engine,
    )
    return coherent_residual - lindblad_dissipator(density, operators)


__all__ = ["lindblad_dissipator", "lindblad_residual"]
