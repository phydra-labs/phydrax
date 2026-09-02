#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import combine_trainable
from ..enforcement import EnforcementProgram
from ..integration import IntegrationRealization
from ..operators.differential._runtime import derivative_runtime_context
from ..terms import ResidualPenalty
from ._functional_objective import _PreparedObjective


class PreparedResidualTerm(StrictModule):
    """One residual penalty bound to an immutable same-update realization."""

    term: ResidualPenalty
    realization: IntegrationRealization
    index: int = eqx.field(static=True)
    selection_scale: Array

    def __init__(
        self,
        term: ResidualPenalty,
        realization: IntegrationRealization,
        index: int,
        selection_scale: Any = 1.0,
        /,
    ):
        self.term = term
        self.realization = realization
        self.index = int(index)
        self.selection_scale = jnp.asarray(selection_scale)

    @property
    def label(self) -> str:
        return self.term.label or type(self.term).__name__

class ResidualRootBlock(StrictModule):
    """One real, flat, square-root-weighted residual-root block."""

    values: Array
    term_index: int = eqx.field(static=True)
    term_label: str = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    coordinate_kind: Literal["real", "imag"] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        values: Any,
        /,
        *,
        term_index: int,
        term_label: str,
        block_name: str,
        source_index: int,
        coordinate_kind: Literal["real", "imag"] = "real",
        event_shape: Sequence[int] = (),
    ):
        array = jnp.asarray(values)
        if array.ndim != 1 or jnp.iscomplexobj(array):
            raise TypeError("Residual root blocks must be one-dimensional real arrays.")
        if coordinate_kind not in ("real", "imag"):
            raise ValueError("Unknown residual-root coordinate kind.")
        self.values = array
        self.term_index = int(term_index)
        self.term_label = str(term_label)
        self.block_name = str(block_name)
        self.source_index = int(source_index)
        self.coordinate_kind = coordinate_kind
        self.event_shape = tuple(int(size) for size in event_shape)


@dataclass(frozen=True, slots=True)
class ResidualRootEntry:
    term_index: int
    term_label: str
    block_name: str
    source_index: int
    coordinate_kind: Literal["real", "imag"]
    event_shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


class FunctionalResidualLayout(StrictModule):
    """Stable canonical-coordinate layout for prepared residual roots."""

    entries: tuple[ResidualRootEntry, ...] = eqx.field(static=True)
    total_size: int = eqx.field(static=True)
    logical_blocks: tuple[tuple[int, str], ...] = eqx.field(static=True)

    def __init__(self, blocks: Sequence[ResidualRootBlock], /):
        values = tuple(blocks)
        entries: list[ResidualRootEntry] = []
        logical: list[tuple[int, str]] = []
        start = 0
        for block in values:
            if not isinstance(block, ResidualRootBlock):
                raise TypeError("Functional residual layouts require ResidualRootBlock values.")
            stop = start + int(block.values.size)
            entries.append(
                ResidualRootEntry(
                    block.term_index,
                    block.term_label,
                    block.block_name,
                    block.source_index,
                    block.coordinate_kind,
                    block.event_shape,
                    start,
                    stop,
                )
            )
            key = (block.term_index, block.block_name)
            if key not in logical:
                logical.append(key)
            start = stop
        self.entries = tuple(entries)
        self.total_size = start
        self.logical_blocks = tuple(logical)

    def flatten(self, blocks: Sequence[ResidualRootBlock], /) -> Array:
        values = tuple(blocks)
        if len(values) != len(self.entries):
            raise ValueError("Residual-root block count changed after preparation.")
        pieces: list[Array] = []
        for block, entry in zip(values, self.entries, strict=True):
            if (
                block.term_index != entry.term_index
                or block.block_name != entry.block_name
                or block.source_index != entry.source_index
                or block.coordinate_kind != entry.coordinate_kind
                or tuple(block.event_shape) != entry.event_shape
                or int(block.values.size) != entry.size
            ):
                raise ValueError("Residual-root structure changed after preparation.")
            pieces.append(block.values)
        if not pieces:
            return jnp.zeros((0,), dtype=float)
        return jnp.concatenate(tuple(pieces), axis=0)

    def split(self, coordinates: Any, /) -> tuple[Array, ...]:
        vector = jnp.asarray(coordinates)
        if vector.shape != (self.total_size,):
            raise ValueError(
                f"Residual coordinates must have shape ({self.total_size},); "
                f"got {vector.shape}."
            )
        return tuple(vector[entry.start : entry.stop] for entry in self.entries)

    def logical_indices(self, term_index: int, block_name: str, /) -> Array:
        selected: list[Array] = []
        for entry in self.entries:
            if entry.term_index == int(term_index) and entry.block_name == str(block_name):
                selected.append(jnp.arange(entry.start, entry.stop, dtype=jnp.int32))
        if not selected:
            raise KeyError(
                f"Unknown residual block ({int(term_index)}, {str(block_name)!r})."
            )
        return selected[0] if len(selected) == 1 else jnp.concatenate(tuple(selected))

    def metadata(self) -> frozendict[str, Any]:
        return frozendict(
            {
                "total_size": self.total_size,
                "entries": tuple(
                    {
                        "term_index": entry.term_index,
                        "term_label": entry.term_label,
                        "block_name": entry.block_name,
                        "source_index": entry.source_index,
                        "coordinate_kind": entry.coordinate_kind,
                        "event_shape": entry.event_shape,
                        "start": entry.start,
                        "stop": entry.stop,
                    }
                    for entry in self.entries
                ),
            }
        )


def materialize_prepared_residual_terms(
    prepared: _PreparedObjective,
    /,
    *,
    require_all: bool = False,
) -> tuple[PreparedResidualTerm, ...]:
    """Bind selected residual penalties to their same-update realizations."""
    terms: list[PreparedResidualTerm] = []
    for prepared_term in prepared.terms:
        term = prepared_term.term
        if not isinstance(term, ResidualPenalty):
            if require_all:
                raise TypeError(
                    "ResidualPenalty training terms only; got "
                    f"{type(term).__name__}."
                )
            continue
        if prepared_term.payload_kind != "realization" or not isinstance(
            prepared_term.payload, IntegrationRealization
        ):
            raise TypeError("Residual terms require a prepared IntegrationRealization.")
        terms.append(
            PreparedResidualTerm(
                term,
                prepared_term.payload,
                prepared_term.index,
                jnp.asarray(prepared.selection.scale),
            )
        )
    return tuple(terms)


def _event_shape(field: cx.Field, /) -> tuple[int, ...]:
    return tuple(
        int(field.data.shape[index])
        for index, dimension in enumerate(field.dims)
        if dimension is None
    )


def _root_blocks_from_data(
    prepared: PreparedResidualTerm,
    data: Any,
    /,
) -> tuple[ResidualRootBlock, ...]:
    blocks: list[ResidualRootBlock] = []
    for source_index, (residual, coefficient) in enumerate(
        zip(data.residuals, data.coefficients, strict=True)
    ):
        residual_blocks = (
            (residual,)
            if prepared.term.blocks is None
            else prepared.term.blocks.split(residual)
        )
        names = (
            (prepared.label,)
            if prepared.term.blocks is None
            else prepared.term.blocks.names
        )
        root = cx.Field(
            jnp.sqrt(
                jnp.asarray(prepared.selection_scale)
                * jnp.asarray(coefficient.data)
            ),
            dims=coefficient.dims,
        )
        for block_name, residual_block in zip(names, residual_blocks, strict=True):
            weighted = root * residual_block
            values = jnp.asarray(weighted.data)
            shape = _event_shape(residual_block)
            blocks.append(
                ResidualRootBlock(
                    jnp.real(values).reshape((-1,)),
                    term_index=prepared.index,
                    term_label=prepared.label,
                    block_name=block_name,
                    source_index=source_index,
                    coordinate_kind="real",
                    event_shape=shape,
                )
            )
            if jnp.iscomplexobj(values):
                blocks.append(
                    ResidualRootBlock(
                        jnp.imag(values).reshape((-1,)),
                        term_index=prepared.index,
                        term_label=prepared.label,
                        block_name=block_name,
                        source_index=source_index,
                        coordinate_kind="imag",
                        event_shape=shape,
                    )
                )
    return tuple(blocks)


def evaluate_prepared_residual_term(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    enforcement: EnforcementProgram | None,
    prepared: PreparedResidualTerm,
    /,
    *,
    iteration: Array | int | None,
    residual_override: Any = None,
) -> tuple[ResidualRootBlock, ...]:
    """Evaluate one prepared term as canonical real residual-root blocks."""
    functions = combine_trainable(params, non_trainable)
    enforced = functions if enforcement is None else enforcement.apply(functions)
    with derivative_runtime_context():
        data = prepared.term._quadratic_residual_data(
            enforced,
            residual_override,
            realization=prepared.realization,
            iter_=iteration,
        )
    return _root_blocks_from_data(prepared, data)

def prepared_term_residual_vector(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    enforcement: EnforcementProgram | None,
    prepared: PreparedResidualTerm,
    /,
    *,
    iteration: Array | int | None,
) -> Array:
    """Evaluate one prepared term as one canonical real root vector."""
    blocks = evaluate_prepared_residual_term(
        params,
        non_trainable,
        enforcement,
        prepared,
        iteration=iteration,
    )
    pieces = tuple(block.values for block in blocks)
    if not pieces:
        return jnp.zeros((0,), dtype=float)
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)


def prepared_residual_terms_loss(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    enforcement: EnforcementProgram | None,
    terms: Sequence[PreparedResidualTerm],
    /,
    *,
    iteration: Array | int | None,
) -> Array:
    """Evaluate one pure prepared residual objective from its exact roots."""
    total = jnp.asarray(0.0, dtype=float)
    for term in terms:
        roots = prepared_term_residual_vector(
            params,
            non_trainable,
            enforcement,
            term,
            iteration=iteration,
        )
        total = total + jnp.real(jnp.vdot(roots, roots))
    return total


def prepared_residual_loss_and_flat_gradient(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    enforcement: EnforcementProgram | None,
    terms: Sequence[PreparedResidualTerm],
    /,
    *,
    iteration: Array | int | None,
) -> tuple[Array, Array, Any]:
    """Return a pure residual loss and gradient in shared flat coordinates."""
    from jax.flatten_util import ravel_pytree

    flat_params, unravel = ravel_pytree(params)

    def loss_from_flat(flat):
        return prepared_residual_terms_loss(
            unravel(flat),
            non_trainable,
            enforcement,
            terms,
            iteration=iteration,
        )

    loss, gradient = jax.value_and_grad(loss_from_flat)(flat_params)
    return loss, gradient, unravel


class PreparedFunctionalResidual(StrictModule):
    """Prepared parameter-to-residual-root map with a stable output layout."""

    terms: tuple[PreparedResidualTerm, ...]
    non_trainable: PyTree[Any]
    enforcement: EnforcementProgram | None
    transform: Any
    layout: FunctionalResidualLayout
    iteration: Any

    def __init__(
        self,
        terms: Sequence[PreparedResidualTerm],
        non_trainable: PyTree[Any],
        enforcement: EnforcementProgram | None,
        layout: FunctionalResidualLayout,
        iteration: Any,
        transform: Any = None,
        /,
    ):
        values = tuple(terms)
        if not values:
            raise ValueError("PreparedFunctionalResidual requires residual terms.")
        if any(not isinstance(term, PreparedResidualTerm) for term in values):
            raise TypeError("Prepared functional residual terms have invalid types.")
        if not isinstance(layout, FunctionalResidualLayout):
            raise TypeError("layout must be a FunctionalResidualLayout.")
        self.terms = values
        self.non_trainable = non_trainable
        self.enforcement = enforcement
        self.layout = layout
        self.iteration = iteration
        self.transform = transform

    def blocks_for(
        self,
        params: PyTree[Any],
        term: PreparedResidualTerm,
        /,
    ) -> tuple[ResidualRootBlock, ...]:
        if all(value is not term for value in self.terms):
            raise ValueError("Prepared residual term is not part of this objective.")
        return (
            evaluate_prepared_residual_term(
                params,
                self.non_trainable,
                self.enforcement,
                term,
                iteration=self.iteration,
            )
            if self.transform is None
            else self.transform.term_blocks(params, self, term)
        )

    def term_blocks(
        self,
        params: PyTree[Any],
        /,
    ) -> tuple[tuple[ResidualRootBlock, ...], ...]:
        return tuple(self.blocks_for(params, term) for term in self.terms)

    def blocks(self, params: PyTree[Any], /) -> tuple[ResidualRootBlock, ...]:
        return tuple(block for term in self.term_blocks(params) for block in term)

    def roots(self, params: PyTree[Any], /) -> Array:
        return self.layout.flatten(self.blocks(params))

    def loss(self, params: PyTree[Any], /) -> Array:
        roots = self.roots(params)
        return jnp.real(jnp.vdot(roots, roots))


def prepare_functional_residual(
    prepared: _PreparedObjective,
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    enforcement: EnforcementProgram | None,
    /,
    *,
    require_all: bool = False,
) -> PreparedFunctionalResidual:
    """Prepare a stable residual-root map from one prepared objective."""
    terms = materialize_prepared_residual_terms(prepared, require_all=require_all)
    blocks = tuple(
        block
        for term in terms
        for block in evaluate_prepared_residual_term(
            params,
            non_trainable,
            enforcement,
            term,
            iteration=prepared.iteration,
        )
    )
    if not blocks:
        raise ValueError("Prepared objective produced no residual roots.")
    return PreparedFunctionalResidual(
        terms,
        non_trainable,
        enforcement,
        FunctionalResidualLayout(blocks),
        prepared.iteration,
    )


def prepared_residual_jacobians(
    residual: PreparedFunctionalResidual,
    params: PyTree[Any],
    /,
) -> tuple[Array, tuple[Array, ...], Any]:
    """Differentiate each residual term in one shared flat parameter order."""
    from jax.flatten_util import ravel_pytree

    flat_params, unravel = ravel_pytree(params)
    jacobians: list[Array] = []
    for term in residual.terms:

        def term_roots(flat, _term=term):
            blocks = residual.blocks_for(unravel(flat), _term)
            term_entries = tuple(
                entry
                for entry in residual.layout.entries
                if entry.term_index == _term.index
            )
            pieces = tuple(block.values for block in blocks)
            if len(pieces) != len(term_entries):
                raise ValueError("Prepared term residual layout changed.")
            return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)

        jacobian = jax.jacrev(term_roots)(flat_params)
        jacobians.append(jnp.asarray(jacobian).reshape((-1, int(flat_params.size))))
    return flat_params, tuple(jacobians), unravel


__all__ = [
    "FunctionalResidualLayout",
    "PreparedFunctionalResidual",
    "PreparedResidualTerm",
    "ResidualRootBlock",
    "ResidualRootEntry",
    "evaluate_prepared_residual_term",
    "materialize_prepared_residual_terms",
    "prepare_functional_residual",
    "prepared_residual_jacobians",
    "prepared_residual_loss_and_flat_gradient",
    "prepared_residual_terms_loss",
    "prepared_term_residual_vector",
]
