#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source AIMNet2 ``AEVSV`` expansion and ``ConvSV`` environment contractions.

Formulas follow AIMNetcentral 0.2.0 ``aimnet/modules/aev.py`` and
``aimnet/ops.py``. For a center ``i`` and neighbor ``j`` with displacement
``r_ij = r_j - r_i`` (``d = |r_ij|``, ``u = r_ij / d``):

* scalar AEV ``g_s(d) = exp(-eta (d - s_k)^2) f_c(d)`` with the cosine cutoff
  ``f_c(d) = (cos(pi clamp(d, 1e-6, rc) / rc) + 1) / 2`` (C1 at ``rc``: zero
  value and slope beyond it, discontinuous second derivative there);
* vector AEV ``g_v(d) = g_s(d) u`` (``g_sv = [g_s, g_s u]``, shape ``(G, 4)``).

The source registers a second ``(rc_v, eta_v, shifts_v)`` basis but its forward
never reads it: the vector channels use the scalar basis and scalar cutoff.
Reconstruction therefore requires ``nshifts_v == nshifts_s`` (else the source
fails) and keeps the inert vector declaration as recorded source evidence only.

``ConvSV`` contracts neighbor features with ``g_sv``: for 1-D features
``A[c, k, :] = sum_j a_j[c] g_sv(r_ij)[k, :]``; with ``d2features``
``A[c, k, :] = sum_j a_j[c, k] g_sv(r_ij)[k, :]`` (elementwise in the shift
index, not a contraction). The output is
``[A[:, :, 0].flatten(), sum_xyz((sum_k agh[c, k, h] A[c, k, xyz])^2).flatten()]``,
rotation invariant and of size ``C (G + H)``. Every receiver sum runs on the
graph's prepared streamed schedule; nothing ``(E, G, 4)``-sized is stored.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import register_artifact_value
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedPayloadSpec,
    StreamedRelationResult,
)


SOURCE_MINIMUM_SHIFT = 0.8
SOURCE_CUTOFF_DISTANCE_FLOOR = 1.0e-6


def source_default_shifts(cutoff: float, count: int, minimum: float, /) -> np.ndarray:
    """``torch.linspace(rmin, rc, n + 1)[:n]`` as float32 (default construction only)."""
    return np.linspace(minimum, cutoff, count + 1, dtype=np.float64)[:count].astype(
        np.float32
    )


def source_default_eta(cutoff: float, count: int, minimum: float, /) -> float:
    """``(1 / ((rc - rmin) / n))**2`` (default construction only)."""
    return (1.0 / ((cutoff - minimum) / count)) ** 2


def source_cosine_cutoff(distance: Array, cutoff: Array, /) -> Array:
    """``ops.cosine_cutoff`` with torch ``clamp`` derivative semantics."""
    floor = jnp.asarray(SOURCE_CUTOFF_DISTANCE_FLOOR, dtype=distance.dtype)
    radius = cutoff.astype(distance.dtype)
    inside = (distance >= floor) & (distance <= radius)
    clamped = jnp.where(inside, distance, jnp.where(distance < floor, floor, radius))
    return 0.5 * (jnp.cos(clamped * (jnp.pi / radius)) + 1.0)


class AIMNet2RadialBasis(StrictModule, NonTrainableState):
    """Fixed source Gaussian shifts, width and cosine cutoff of ``AEVSV``.

    ``radius`` and ``basis_id`` derive from the executed (stored) arrays, so a
    restored basis rebuilds to the identical value from its own leaves.
    """

    shifts: Array
    eta: Array
    cutoff: Array
    radius: float = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        shifts: ArrayLike,
        eta: ArrayLike,
        cutoff: ArrayLike,
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        source_shifts = np.asarray(shifts)
        source_eta = np.asarray(eta)
        source_cutoff = np.asarray(cutoff)
        for name, value in (
            ("shifts", source_shifts),
            ("eta", source_eta),
            ("cutoff", source_cutoff),
        ):
            if value.dtype.kind != "f" or not np.all(np.isfinite(value)):
                raise ValueError(f"AEV {name} must be finite floating values.")
        if source_shifts.ndim != 1 or source_shifts.size == 0:
            raise ValueError("AEV shifts must be one nonempty axis.")
        if source_eta.shape != () or source_cutoff.shape != ():
            raise ValueError("AEV eta and cutoff must be scalars.")
        if float(source_cutoff) <= 0.0 or float(source_eta) <= 0.0:
            raise ValueError("AEV eta and cutoff must be positive.")
        self.shifts = jnp.asarray(source_shifts, dtype=dtype)
        self.eta = jnp.asarray(source_eta, dtype=dtype)
        self.cutoff = jnp.asarray(source_cutoff, dtype=dtype)
        executed = tuple(
            np.asarray(jax.device_get(value))
            for value in (self.shifts, self.eta, self.cutoff)
        )
        self.radius = float(executed[2])
        self.basis_id = canonical_fingerprint(
            {
                "kind": "aimnet2-aev-basis",
                "values": array_tree_fingerprint(executed),
                "dtype": executed[0].dtype.name,
                "cutoff_function": "cosine-clamped-1e-6",
            }
        )

    def rebuilt(self) -> AIMNet2RadialBasis:
        """The constructor's value from this basis's own stored leaves."""
        return AIMNet2RadialBasis(
            np.asarray(jax.device_get(self.shifts)),
            np.asarray(jax.device_get(self.eta)),
            np.asarray(jax.device_get(self.cutoff)),
            dtype=self.shifts.dtype,
        )

    @property
    def shift_count(self) -> int:
        return int(self.shifts.shape[0])

    def scalar(self, distance: Array, /) -> Array:
        """``g_s`` of ``(...)`` distances, shape ``(..., G)``."""
        envelope = source_cosine_cutoff(distance, self.cutoff)
        shifted = distance[..., None] - self.shifts.astype(distance.dtype)
        return (
            jnp.exp(-self.eta.astype(distance.dtype) * shifted**2) * envelope[..., None]
        )

    def scalar_vector(self, distance: Array, unit: Array, /) -> Array:
        """``g_sv`` of one route set: ``(..., G, 4)`` scalar then ``x, y, z``."""
        scalar = self.scalar(distance)
        vector = scalar[..., :, None] * unit[..., None, :]
        return jnp.concatenate((scalar[..., None], vector), axis=-1)


class AIMNet2ConvSV(StrictModule):
    """Source ``ConvSV`` combination tensor ``agh`` (a trainable parameter)."""

    agh: Array
    channel_count: int = eqx.field(static=True)
    shift_count: int = eqx.field(static=True)
    combination_count: int = eqx.field(static=True)
    d2_features: bool = eqx.field(static=True)

    def __init__(
        self, agh: ArrayLike, /, *, d2_features: bool, dtype: DTypeLike = jnp.float64
    ) -> None:
        weights = np.asarray(agh)
        if weights.dtype.kind != "f" or weights.ndim != 3 or 0 in weights.shape:
            raise ValueError("ConvSV agh must be a nonempty (C, G, H) float tensor.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("ConvSV agh must be finite.")
        if not isinstance(d2_features, bool):
            raise TypeError("d2_features must be a boolean.")
        self.agh = jnp.asarray(weights, dtype=dtype)
        self.channel_count = int(weights.shape[0])
        self.shift_count = int(weights.shape[1])
        self.combination_count = int(weights.shape[2])
        self.d2_features = d2_features

    def rebuilt(self) -> AIMNet2ConvSV:
        """The constructor's value from this convolution's own stored leaves."""
        return AIMNet2ConvSV(
            np.asarray(jax.device_get(self.agh)),
            d2_features=self.d2_features,
            dtype=self.agh.dtype,
        )

    @property
    def output_size(self) -> int:
        return self.channel_count * (self.shift_count + self.combination_count)

    def message(self, features: Array, basis: Array, /) -> Array:
        """One route's ``(C, G, 4)`` contribution from sender features."""
        if self.d2_features:
            return features[:, :, None] * basis[None, :, :]
        return features[:, None, None] * basis[None, :, :]

    def finish(self, aggregate: Array, /) -> Array:
        """Receiver output ``[scalar (C G), squared combinations (C H)]``."""
        scalar = aggregate[..., 0].reshape((-1,))
        combined = contract(
            "cgh,cgd->chd", self.agh.astype(aggregate.dtype), aggregate[..., 1:]
        )
        return jnp.concatenate((scalar, jnp.sum(combined**2, axis=-1).reshape((-1,))))


def _environment_edge(
    parameters: tuple[AIMNet2RadialBasis, AIMNet2ConvSV, AIMNet2ConvSV | None],
    sender: tuple[Array, Array],
    receiver: Array,
    edge: tuple[Array, Array],
    /,
) -> dict[str, Array]:
    basis, atom_conv, charge_conv = parameters
    del receiver
    features, charges = sender
    distance, unit = edge
    geometry = basis.scalar_vector(distance, unit.astype(features.dtype))
    messages = {"atom": atom_conv.message(features, geometry)}
    if charge_conv is not None:
        messages["charge"] = charge_conv.message(charges, geometry)
    return messages


def _environment_epilogue(
    parameters: tuple[AIMNet2RadialBasis, AIMNet2ConvSV, AIMNet2ConvSV | None],
    receiver: Array,
    aggregate: dict[str, Array],
    /,
) -> tuple[Array, ...]:
    _, atom_conv, charge_conv = parameters
    del receiver
    atom = atom_conv.finish(aggregate["atom"])
    if charge_conv is None:
        return (atom,)
    return atom, charge_conv.finish(aggregate["charge"])


def aimnet2_environment(
    relation: PreparedStreamedRelation,
    basis: AIMNet2RadialBasis,
    atom_conv: AIMNet2ConvSV,
    charge_conv: AIMNet2ConvSV | None,
    features: Array,
    charges: Array,
    receiver: Array,
    edge: tuple[Array, Array],
    edge_active: Array,
    /,
) -> StreamedRelationResult:
    """Fused ``conv_a`` (and ``conv_q``) of one charge pass, once per receiver.

    ``features`` are ``(S, C)`` or ``(S, C, G)`` source atom features and
    ``charges`` ``(S, Q)`` source charges (``S`` relation sources, which may be
    owner columns rather than receivers); ``receiver`` is the receiver weight
    row; ``edge = (distance, u)`` with ``u`` the center-to-neighbor unit vector.
    The result's receiver outputs are ``(conv_a,)`` or ``(conv_a, conv_q)`` per
    receiver; its evidence is the streamed schedule's.
    """
    dtype = features.dtype
    if atom_conv.shift_count != basis.shift_count or (
        charge_conv is not None and charge_conv.shift_count != basis.shift_count
    ):
        raise ValueError("ConvSV shift count differs from its AEV basis.")
    shift = basis.shift_count
    message = {"atom": jax.ShapeDtypeStruct((atom_conv.channel_count, shift, 4), dtype)}
    output = [jax.ShapeDtypeStruct((atom_conv.output_size,), dtype)]
    if charge_conv is not None:
        message["charge"] = jax.ShapeDtypeStruct(
            (charge_conv.channel_count, shift, 4), dtype
        )
        output.append(jax.ShapeDtypeStruct((charge_conv.output_size,), dtype))
    return relation.evaluate(
        StreamedPayloadSpec(message=message, output=tuple(output)),
        _environment_edge,
        _environment_epilogue,
        (basis, atom_conv, charge_conv),
        (features, charges),
        receiver,
        edge,
        edge_active=edge_active,
    )


def source_ahg_initialization(
    channels: int, shifts: int, combinations: int, *, seed: int
) -> np.ndarray:
    """The source ``_init_ahg`` maxmin construction from a NumPy generator.

    Default random construction only: the source draws from torch's generator,
    so values are a different draw of the same distribution and algorithm.
    """
    generator = np.random.default_rng(seed)
    result = np.zeros((channels, shifts, combinations), dtype=np.float32)
    x = np.arange(shifts, dtype=np.float64)[None]
    for channel in range(channels):
        draws = generator.standard_normal((8 * combinations, 4))
        a1, a2, a3, a4 = (draws[:, index : index + 1] for index in range(4))
        y = a1 * np.sin(a2 * 2.0 * x * math.pi / shifts) + a3 * np.cos(
            a4 * 2.0 * x * math.pi / shifts
        )
        y = y - y.mean(axis=-1, keepdims=True)
        y = y / y.std(axis=-1, ddof=1, keepdims=True)
        distances = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
        selected = np.zeros((combinations, shifts))
        available = np.ones(y.shape[0], dtype=bool)
        first = int(np.argmax(distances.sum(axis=-1)))
        selected[0] = y[first]
        available[first] = False
        for column in range(1, combinations):
            minimum = np.min(
                np.linalg.norm(selected[:column, None, :] - y[None, :, :], axis=-1),
                axis=0,
            )
            # The source masks the *sorted* index positions with the element
            # mask (``argsort(mindist)[mask][-1]``); mirrored as written.
            order = np.argsort(minimum, kind="stable")
            chosen = int(order[available][-1])
            selected[column] = y[chosen]
            available[chosen] = False
        result[channel] = selected.T
    return result


register_artifact_value("phydrax.nn.atomistic:AIMNet2RadialBasis", AIMNet2RadialBasis)
register_artifact_value("phydrax.nn.atomistic:AIMNet2ConvSV", AIMNet2ConvSV)


__all__ = [
    "AIMNet2ConvSV",
    "AIMNet2RadialBasis",
    "aimnet2_environment",
    "SOURCE_CUTOFF_DISTANCE_FLOOR",
    "SOURCE_MINIMUM_SHIFT",
    "source_ahg_initialization",
    "source_cosine_cutoff",
    "source_default_eta",
    "source_default_shifts",
]
