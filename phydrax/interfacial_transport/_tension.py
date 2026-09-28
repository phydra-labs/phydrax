#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pairwise interface tension and mobility contracts for labeled multiphase systems.

A labeled system with stable label identifiers ``l_0, ..., l_{L-1}`` carries one
effective interfacial tension ``sigma_ij`` and one normal mobility ``mu_ij`` per
unordered pair of distinct labels. The interface ``Gamma_ij`` moves with normal
velocity ``mu_ij * sigma_ij * kappa`` under curvature-driven flow and stores energy
``sigma_ij * |Gamma_ij|``. The label index order is the declared ``label_ids``
order; consumers map their own region/grain identities onto it explicitly.

Two static structures are supported:

- ``"pairwise"``: a dense symmetric ``L x L`` matrix with zero diagonal;
- ``"uniform"``: one scalar shared by every pair of distinct labels, for systems
  whose label count makes an ``L x L`` matrix a resource defect (grain/foam cells).

Construction validates the representation contract (finite, exactly symmetric,
zero diagonal, nonnegative tension, positive mobility) on the host before
assignment. Scientific admissibility — the triangle inequality required for
lower semicontinuity of the multiphase energy and the conditional negative
semidefiniteness under which threshold-dynamics energy dissipation is proved
(Esedoglu and Otto, CPAM 68, 2015) — is reported as device evidence by
`InterfaceTensionMatrix.admissibility`, because values are dynamic parameter
leaves that may change under inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState, parameter_field, ParameterOwner
from .._validation import unique_identifiers
from ..linalg import DensePropertyVerificationPolicy, verify_dense_properties
from ..typing import (
    as_host_array,
    Dim,
    Float64,
    HostFloat64,
    Identifier,
    Identifiers,
    parse,
    Scalar,
    Scope,
)


InterfacePairStructure: TypeAlias = Literal["pairwise", "uniform"]

_CNSD_POLICY = DensePropertyVerificationPolicy(require_positive_semidefinite=True)
# Roundoff admission for evidence decisions: dtype epsilon scaled by the largest
# pair value, matching the native dense property verification default.
_RELATIVE_TOLERANCE = 64.0


class _InterfaceLabelDim(Dim, minimum=2):
    """Stable labels of one multiphase system."""


def _label_ids(label_ids: Sequence[str], /) -> tuple[str, ...]:
    labels = unique_identifiers(label_ids, "label_ids")
    if len(labels) < 2:
        raise ValueError("label_ids must declare at least two labels.")
    return labels


def _pair_values(
    values: ArrayLike,
    labels: tuple[str, ...],
    structure: InterfacePairStructure,
    name: str,
    /,
    *,
    strictly_positive: bool,
) -> np.ndarray:
    scope = Scope()
    parse(labels, Identifiers[_InterfaceLabelDim], "label_ids", scope=scope)
    match structure:
        case "pairwise":
            host = as_host_array(
                values,
                HostFloat64[_InterfaceLabelDim, _InterfaceLabelDim],
                name,
                scope=scope,
            )
            off_diagonal = host[~np.eye(len(labels), dtype=np.bool_)]
            if not np.all(np.isfinite(host)):
                raise ValueError(f"{name} must be finite.")
            if not np.array_equal(host, host.T):
                raise ValueError(f"{name} must be exactly symmetric.")
            if np.any(np.diagonal(host) != 0.0):
                raise ValueError(f"{name} must have a zero diagonal.")
        case "uniform":
            host = as_host_array(values, HostFloat64[Scalar], name)
            off_diagonal = host.reshape((1,))
            if not np.isfinite(host):
                raise ValueError(f"{name} must be finite.")
        case _:
            assert_never(structure)
    if strictly_positive and np.any(off_diagonal <= 0.0):
        raise ValueError(f"{name} must be positive between distinct labels.")
    if np.any(off_diagonal < 0.0):
        raise ValueError(f"{name} must be nonnegative between distinct labels.")
    return host


def _structure_id(
    kind: str, labels: tuple[str, ...], structure: InterfacePairStructure, /
) -> str:
    return canonical_fingerprint(
        {"kind": kind, "label_ids": list(labels), "structure": structure}
    )


def _gather_pairs(
    values: Array,
    structure: InterfacePairStructure,
    first: ArrayLike,
    second: ArrayLike,
    /,
) -> Array:
    first_ = jnp.asarray(first)
    second_ = jnp.asarray(second)
    if not jnp.issubdtype(first_.dtype, jnp.integer) or not jnp.issubdtype(
        second_.dtype, jnp.integer
    ):
        raise TypeError("Pair label indices must be integer arrays.")
    match structure:
        case "pairwise":
            return values[first_, second_]
        case "uniform":
            return jnp.where(first_ == second_, jnp.zeros((), values.dtype), values)
        case _:
            assert_never(structure)


def _dense(values: Array, structure: InterfacePairStructure, count: int, /) -> Array:
    match structure:
        case "pairwise":
            return values
        case "uniform":
            return values * (
                jnp.ones((count, count), values.dtype)
                - jnp.eye(count, dtype=values.dtype)
            )
        case _:
            assert_never(structure)


def _helmert_basis(count: int, /) -> np.ndarray:
    """Orthonormal ``(L-1) x L`` basis of the sum-zero subspace (host constant)."""
    rows = np.arange(1, count, dtype=np.float64)[:, None]
    columns = np.arange(count, dtype=np.float64)[None, :]
    scale = 1.0 / np.sqrt(rows * (rows + 1.0))
    return np.where(columns < rows, scale, np.where(columns == rows, -rows * scale, 0.0))


def _conditional_spectrum(values: Array, /) -> tuple[Array, Array]:
    """Smallest eigenvalue of ``-Q values Q^T`` on the sum-zero subspace and CNSD flag.

    Shared with threshold-dynamics kernel matrices, which carry the same
    conditional-negative-semidefinite dissipation criterion.
    """
    basis = jnp.asarray(_helmert_basis(values.shape[0]), dtype=values.dtype)
    spectrum = verify_dense_properties(-(basis @ values @ basis.T), policy=_CNSD_POLICY)
    return jnp.min(spectrum.eigenvalues), spectrum.positive_semidefinite


def _triangle_excess(values: Array, /) -> Array:
    """Largest ``sigma_ik - sigma_ij - sigma_jk`` over distinct label triples."""
    count = values.shape[0]
    if count < 3:
        return jnp.asarray(-jnp.inf, dtype=values.dtype)
    indices = jnp.arange(count)
    distinct = indices[:, None] != indices[None, :]

    def through(middle: Array, /) -> Array:
        excess = values - values[:, middle][:, None] - values[middle, :][None, :]
        valid = distinct & (indices[:, None] != middle) & (indices[None, :] != middle)
        return jnp.max(jnp.where(valid, excess, -jnp.inf))

    # One L x L slab per middle label bounds the O(L^3) check to O(L^2) memory.
    return jnp.max(jax.lax.map(through, indices))


class InterfaceTensionEvidence(StrictModule, NonTrainableState):
    """Device admissibility evidence for one interface tension matrix.

    ``triangle_excess`` is the largest ``sigma_ik - sigma_ij - sigma_jk`` over
    distinct label triples (``-inf`` for two labels); the triangle inequality holds
    when it is at most ``tolerance``. ``conditional_minimum_eigenvalue`` is the
    smallest eigenvalue of ``-Q sigma Q^T`` for an orthonormal basis ``Q`` of the
    sum-zero subspace; ``sigma`` is conditionally negative semidefinite when it is
    at least ``-tolerance``. ``energy_dissipation_admitted`` is the sufficient
    condition under which Esedoglu–Otto threshold dynamics dissipates its
    discrete energy for every time step.
    """

    finite: Array
    minimum_pair_value: Array
    nonnegative: Array
    triangle_excess: Array
    triangle_inequality: Array
    strict_triangle_inequality: Array
    conditional_minimum_eigenvalue: Array
    conditionally_negative_semidefinite: Array
    tolerance: Array
    admissible: Array
    energy_dissipation_admitted: Array


class InterfaceMobilityEvidence(StrictModule, NonTrainableState):
    """Device admissibility evidence for one interface mobility matrix."""

    finite: Array
    minimum_pair_value: Array
    positive: Array
    admissible: Array


class InterfaceTensionMatrix(StrictModule, ParameterOwner):
    """Effective interfacial tensions ``sigma_ij`` between stable labels.

    ``values`` is the dense symmetric zero-diagonal matrix (``structure="pairwise"``)
    or the one scalar shared by every distinct pair (``structure="uniform"``). A
    soap film between two gas cells carries the effective tension of both
    surfaces (``2 * gamma``); generic material interfaces carry their single
    tension. Values are inferable parameter leaves; the label identity and the
    structure are static.
    """

    __strict_contract__ = True

    values: Float64[_InterfaceLabelDim, _InterfaceLabelDim] | Float64[Scalar] = (
        parameter_field()
    )
    label_ids: Identifiers[_InterfaceLabelDim] = eqx.field(static=True)
    structure: InterfacePairStructure = eqx.field(static=True)
    structure_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        label_ids: Sequence[str],
        values: ArrayLike,
        /,
        *,
        structure: InterfacePairStructure = "pairwise",
    ) -> None:
        structure_ = parse(structure, InterfacePairStructure, "structure")
        labels = _label_ids(label_ids)
        host = _pair_values(
            values, labels, structure_, "tension values", strictly_positive=False
        )
        self.values = jnp.asarray(host)
        self.label_ids = labels
        self.structure = structure_
        self.structure_id = _structure_id("interface-tension", labels, structure_)

    @property
    def label_count(self) -> int:
        return len(self.label_ids)

    def label_index(self, label_id: str, /) -> int:
        """Host index of one declared label."""
        if label_id not in self.label_ids:
            raise ValueError(f"Unknown interface label {label_id!r}.")
        return self.label_ids.index(label_id)

    def pair_values(self, first: ArrayLike, second: ArrayLike, /) -> Array:
        """Gather ``sigma`` for broadcast integer label-index arrays (zero on ties)."""
        return _gather_pairs(self.values, self.structure, first, second)

    def dense_values(self) -> Array:
        """Materialize the ``L x L`` matrix; callers own the ``L**2`` storage."""
        return _dense(self.values, self.structure, self.label_count)

    def admissibility(self) -> InterfaceTensionEvidence:
        """Triangle-inequality and conditional-negative-semidefinite evidence."""
        match self.structure:
            case "pairwise":
                return self._pairwise_admissibility()
            case "uniform":
                return self._uniform_admissibility()
            case _:
                assert_never(self.structure)

    def _pairwise_admissibility(self) -> InterfaceTensionEvidence:
        values = self.values
        count = self.label_count
        off_diagonal = ~jnp.eye(count, dtype=jnp.bool_)
        minimum = jnp.min(jnp.where(off_diagonal, values, jnp.inf))
        scale = jnp.max(jnp.abs(values))
        tolerance = _RELATIVE_TOLERANCE * jnp.finfo(values.dtype).eps * scale
        excess = _triangle_excess(values)
        conditional_minimum, conditionally_negative_semidefinite = _conditional_spectrum(
            values
        )
        return _tension_evidence(
            finite=jnp.all(jnp.isfinite(values)),
            minimum=minimum,
            excess=excess,
            conditional_minimum=conditional_minimum,
            conditionally_negative_semidefinite=conditionally_negative_semidefinite,
            tolerance=tolerance,
        )

    def _uniform_admissibility(self) -> InterfaceTensionEvidence:
        value = self.values
        tolerance = _RELATIVE_TOLERANCE * jnp.finfo(value.dtype).eps * jnp.abs(value)
        # sigma * (1 1^T - I) restricted to the sum-zero subspace is -sigma * I, and
        # every distinct triple has excess sigma - 2 sigma = -sigma.
        excess = -value if self.label_count >= 3 else jnp.asarray(-jnp.inf, value.dtype)
        return _tension_evidence(
            finite=jnp.isfinite(value),
            minimum=value,
            excess=excess,
            conditional_minimum=value,
            conditionally_negative_semidefinite=value >= -tolerance,
            tolerance=tolerance,
        )


def _tension_evidence(
    *,
    finite: Array,
    minimum: Array,
    excess: Array,
    conditional_minimum: Array,
    conditionally_negative_semidefinite: Array,
    tolerance: Array,
) -> InterfaceTensionEvidence:
    nonnegative = minimum >= 0.0
    triangle = excess <= tolerance
    admissible = finite & nonnegative & triangle
    return InterfaceTensionEvidence(
        finite=finite,
        minimum_pair_value=minimum,
        nonnegative=nonnegative,
        triangle_excess=excess,
        triangle_inequality=triangle,
        strict_triangle_inequality=excess < -tolerance,
        conditional_minimum_eigenvalue=conditional_minimum,
        conditionally_negative_semidefinite=conditionally_negative_semidefinite,
        tolerance=tolerance,
        admissible=admissible,
        energy_dissipation_admitted=admissible & conditionally_negative_semidefinite,
    )


class InterfaceMobilityMatrix(StrictModule, ParameterOwner):
    """Normal interface mobilities ``mu_ij`` between stable labels.

    Interface ``Gamma_ij`` moves with normal velocity ``mu_ij * sigma_ij * kappa``.
    Mobilities are strictly positive between distinct labels; the diagonal is zero
    for the ``"pairwise"`` structure. Values are inferable parameter leaves.
    """

    __strict_contract__ = True

    values: Float64[_InterfaceLabelDim, _InterfaceLabelDim] | Float64[Scalar] = (
        parameter_field()
    )
    label_ids: Identifiers[_InterfaceLabelDim] = eqx.field(static=True)
    structure: InterfacePairStructure = eqx.field(static=True)
    structure_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        label_ids: Sequence[str],
        values: ArrayLike,
        /,
        *,
        structure: InterfacePairStructure = "pairwise",
    ) -> None:
        structure_ = parse(structure, InterfacePairStructure, "structure")
        labels = _label_ids(label_ids)
        host = _pair_values(
            values, labels, structure_, "mobility values", strictly_positive=True
        )
        self.values = jnp.asarray(host)
        self.label_ids = labels
        self.structure = structure_
        self.structure_id = _structure_id("interface-mobility", labels, structure_)

    @property
    def label_count(self) -> int:
        return len(self.label_ids)

    def label_index(self, label_id: str, /) -> int:
        """Host index of one declared label."""
        if label_id not in self.label_ids:
            raise ValueError(f"Unknown interface label {label_id!r}.")
        return self.label_ids.index(label_id)

    def pair_values(self, first: ArrayLike, second: ArrayLike, /) -> Array:
        """Gather ``mu`` for broadcast integer label-index arrays (zero on ties)."""
        return _gather_pairs(self.values, self.structure, first, second)

    def dense_values(self) -> Array:
        """Materialize the ``L x L`` matrix; callers own the ``L**2`` storage."""
        return _dense(self.values, self.structure, self.label_count)

    def admissibility(self) -> InterfaceMobilityEvidence:
        """Finite strictly positive mobility evidence for the current values."""
        match self.structure:
            case "pairwise":
                off_diagonal = ~jnp.eye(self.label_count, dtype=jnp.bool_)
                minimum = jnp.min(jnp.where(off_diagonal, self.values, jnp.inf))
            case "uniform":
                minimum = self.values
            case _:
                assert_never(self.structure)
        finite = jnp.all(jnp.isfinite(self.values))
        positive = minimum > 0.0
        return InterfaceMobilityEvidence(
            finite=finite,
            minimum_pair_value=minimum,
            positive=positive,
            admissible=finite & positive,
        )


__all__ = [
    "InterfaceMobilityEvidence",
    "InterfaceMobilityMatrix",
    "InterfacePairStructure",
    "InterfaceTensionEvidence",
    "InterfaceTensionMatrix",
]
