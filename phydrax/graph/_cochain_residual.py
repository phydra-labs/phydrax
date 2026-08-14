#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reusable, degree-aware residual programs over canonical cochain graphs."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Key

from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ._cochain import CochainFieldSpec
from ._ir import GraphIR


def _callable_identity(fn: Callable, explicit: str | None, /) -> str:
    if explicit is not None:
        identity = str(explicit)
        if not identity:
            raise ValueError("Cochain residual program identity must be non-empty.")
        return identity
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        return f"{fn.__module__}.{fn.__qualname__}"
    fn_type = type(fn)
    return f"{fn_type.__module__}.{fn_type.__qualname__}"


def _validate_specs(
    name: str,
    specs: Mapping[str, CochainFieldSpec],
    /,
) -> frozendict[str, CochainFieldSpec]:
    if not isinstance(specs, Mapping) or not specs:
        raise ValueError(f"Cochain residual {name} specs must be a non-empty mapping.")
    out: dict[str, CochainFieldSpec] = {}
    for field_name, spec in specs.items():
        key = str(field_name)
        if not key:
            raise ValueError(f"Cochain residual {name} names must be non-empty.")
        if not isinstance(spec, CochainFieldSpec):
            raise TypeError(
                f"Cochain residual {name} spec {key!r} is not a CochainFieldSpec."
            )
        if spec.complex_side != "primal":
            raise ValueError(
                "CochainResidualProgram currently supports primal fields only."
            )
        expected = "invariant" if spec.degree == 0 else "signed"
        if spec.cell_orientation != expected:
            raise ValueError(
                f"Degree-{spec.degree} residual fields require {expected!r} orientation."
            )
        out[key] = spec
    return frozendict(out)


def _node_degrees(graph: GraphIR, /) -> Array:
    if not isinstance(graph, GraphIR):
        raise TypeError("CochainResidualProgram requires a GraphIR.")
    if not isinstance(graph.nodes, Mapping) or "cell_dim" not in graph.nodes:
        raise ValueError(
            "CochainResidualProgram requires graph.nodes['cell_dim'] metadata."
        )
    degree = jnp.asarray(graph.nodes["cell_dim"], dtype=jnp.int32)
    if degree.ndim != 1:
        raise ValueError("graph.nodes['cell_dim'] must be rank-1.")
    return degree


def _degree_mask(graph: GraphIR, degree: Array, target: int, /) -> Array:
    mask = degree == int(target)
    if graph.node_mask is not None:
        node_mask = jnp.asarray(graph.node_mask, dtype=bool)
        if node_mask.shape != mask.shape:
            raise ValueError("Graph node_mask shape must match cell_dim shape.")
        mask = mask & node_mask
    return mask


def _mask_field(
    name: str,
    value: Any,
    mask: Array,
    /,
) -> Array:
    array = jnp.asarray(value)
    if array.ndim == 0 or int(array.shape[0]) != int(mask.shape[0]):
        raise ValueError(
            f"Cochain field {name!r} must have leading full-complex cell axis "
            f"of size {mask.shape[0]}, got shape {array.shape!r}."
        )
    expanded = mask.reshape(mask.shape + (1,) * (array.ndim - 1))
    return jnp.where(expanded, array, jnp.zeros((), dtype=array.dtype))


class CochainResidualProgram(StrictModule):
    """A canonical full-complex residual with static field semantics.

    ``residual_fn(graph, fields, *, key=...)`` receives a canonical ``GraphIR`` and
    degree-masked full-cell arrays. It must return a mapping containing exactly the
    declared outputs. Returned arrays are shape-checked and degree-masked again.
    """

    input_specs: frozendict[str, CochainFieldSpec]
    output_specs: frozendict[str, CochainFieldSpec]
    residual_fn: Callable
    identity: str

    def __init__(
        self,
        *,
        inputs: Mapping[str, CochainFieldSpec],
        outputs: Mapping[str, CochainFieldSpec],
        residual_fn: Callable[..., Mapping[str, Any]],
        identity: str | None = None,
    ):
        if not callable(residual_fn):
            raise TypeError("CochainResidualProgram residual_fn must be callable.")
        self.input_specs = _validate_specs("input", inputs)
        self.output_specs = _validate_specs("output", outputs)
        self.identity = _callable_identity(residual_fn, identity)
        self.residual_fn = _ensure_special_kwonly_args(residual_fn)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "cochain_residual_program",
                "identity": self.identity,
                "inputs": {
                    name: spec.to_dict() for name, spec in self.input_specs.items()
                },
                "outputs": {
                    name: spec.to_dict() for name, spec in self.output_specs.items()
                },
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __call__(
        self,
        graph: GraphIR,
        fields: Mapping[str, Any],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> frozendict[str, Array]:
        if not isinstance(fields, Mapping):
            raise TypeError("CochainResidualProgram fields must be a mapping.")
        supplied = frozenset(fields)
        expected = frozenset(self.input_specs)
        if supplied != expected:
            missing = tuple(sorted(expected - supplied))
            extra = tuple(sorted(supplied - expected))
            raise ValueError(
                f"Cochain residual input schema mismatch; missing={missing}, extra={extra}."
            )

        cell_degree = _node_degrees(graph)
        masked_inputs = frozendict(
            {
                name: _mask_field(
                    name,
                    fields[name],
                    _degree_mask(graph, cell_degree, spec.degree),
                )
                for name, spec in self.input_specs.items()
            }
        )
        raw = self.residual_fn(graph, masked_inputs, key=key, **kwargs)
        if not isinstance(raw, Mapping):
            raise TypeError("Cochain residual functions must return a mapping.")
        supplied_outputs = frozenset(raw)
        expected_outputs = frozenset(self.output_specs)
        if supplied_outputs != expected_outputs:
            missing = tuple(sorted(expected_outputs - supplied_outputs))
            extra = tuple(sorted(supplied_outputs - expected_outputs))
            raise ValueError(
                "Cochain residual output schema mismatch; "
                f"missing={missing}, extra={extra}."
            )
        return frozendict(
            {
                name: _mask_field(
                    name,
                    raw[name],
                    _degree_mask(graph, cell_degree, spec.degree),
                )
                for name, spec in self.output_specs.items()
            }
        )


__all__ = ["CochainResidualProgram"]
