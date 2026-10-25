#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Executable layout identity of prepared atomistic runtimes.

Every object reachable from a compiled atomistic argument keeps only
content-free static configuration: capacities, dtypes, structural kinds,
program slots and their labels, force groups, method selectors and declared
resource envelopes (including resource-defining cutoffs and the reference
cell, which fix search/stencil shapes).  Scientific, material and
numeric-revision identities are the owners' exact canonical digests carried as
dynamic `phydrax._identity.IdentityWords`; host properties decode the same
strings for provenance, checkpoints and transactions, and compiled code
compares the words and folds a mismatch into the owner's status.

`atomistic_term_layout_id` and `atomistic_executable_signature` describe the
executable a prepared object compiles to.  Neither ever reads numerical
values or identity words, so all material and parameter revisions of one
layout share them.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._identity import ExecutableSignature, execution_metadata_payload


def _leaf_layout(value: object, /) -> list[list[Any]]:
    leaves = jax.tree_util.tree_flatten_with_path(eqx.filter(value, eqx.is_array))[0]
    return [
        [jax.tree_util.keystr(path), list(leaf.shape), str(leaf.dtype)]
        for path, leaf in leaves
    ]


def atomistic_case_address_words(identifiers: tuple[str, ...], /) -> jax.Array:
    """Encode declared case identities as ``(case, 8)`` uint32 random-address words.

    The address is the digest of ``{"kind": "atomistic-case-address",
    "identity": identifier}``, never the identity digest itself, so stochastic
    model addressing is unchanged by how identities are carried.  A host
    boundary: owners compute it at preparation and carry the words dynamically.
    """
    rows = []
    for identifier in identifiers:
        digest = canonical_fingerprint(
            {"kind": "atomistic-case-address", "identity": identifier}
        )
        rows.append(
            tuple(int(digest[offset : offset + 8], 16) for offset in range(0, 64, 8))
        )
    return jax.numpy.asarray(np.asarray(rows, dtype=np.uint32))


def _static_tree_id(value: object, /) -> str:
    """Fingerprint declared PyTree statics without reading numerical leaf values."""
    return canonical_fingerprint(
        execution_metadata_payload(jax.tree_util.tree_structure(value))
    )


def atomistic_term_layout_id(term: Any, /) -> str:
    """Content-free layout of one prepared energy term.

    Its class, program slot, force group, resource requirements, capabilities
    and the paths/shapes/dtypes of its dynamic leaves; never values or identity
    words.
    """
    return canonical_fingerprint(
        {
            "kind": "atomistic-term-layout",
            "type": f"{type(term).__module__}.{type(term).__qualname__}",
            "name": term.name,
            "force_group": term.force_group,
            "requirements": term.requirements.requirements_id,
            "capabilities": term.capabilities.capabilities_id,
            "static_tree": _static_tree_id(term),
            "leaves": _leaf_layout(term),
        }
    )


def atomistic_hamiltonian_layout_id(program: Any, /) -> str:
    """Content-free executable layout of one prepared Hamiltonian.

    The single canonical grouping identity of prepared Hamiltonians: its type,
    control layout, system layout, resource-bundle layout, every term layout
    and the paths/shapes/dtypes of all dynamic leaves.  Two revisions share it
    exactly when they share one executable; their scientific identities and
    numeric revisions remain distinct.
    """
    return canonical_fingerprint(
        {
            "kind": "atomistic-hamiltonian-layout",
            "type": f"{type(program).__module__}.{type(program).__qualname__}",
            "controls": program.control_layout_id,
            "system": program.system.layout_id,
            "resources": program.neighborhood_bundle.layout_id,
            "terms": [term.layout_id for term in program.terms],
            "static_tree": _static_tree_id(program),
            "leaves": _leaf_layout(program),
        }
    )


def atomistic_executable_signature(
    value: object, /, *, layout_id: str
) -> ExecutableSignature:
    """Static compilation identity of one prepared atomistic executable.

    ``layout_id`` is the owner's content-free configuration digest; the other
    facts are the paths, shapes and dtypes of its dynamic array leaves.
    """
    leaves = jax.tree_util.tree_flatten_with_path(eqx.filter(value, eqx.is_array))[0]
    return ExecutableSignature(
        shapes=tuple(
            (jax.tree_util.keystr(path), tuple(leaf.shape)) for path, leaf in leaves
        ),
        dtypes=tuple(
            (jax.tree_util.keystr(path), np.dtype(leaf.dtype)) for path, leaf in leaves
        ),
        algorithm_facts={"layout_id": layout_id, "static_tree": _static_tree_id(value)},
        backend_facts={"backend": jax.default_backend()},
    )


__all__ = [
    "atomistic_case_address_words",
    "atomistic_executable_signature",
    "atomistic_hamiltonian_layout_id",
    "atomistic_term_layout_id",
]
