#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

import phydrax.ein as ein
from phydrax._doc import DOC_KEY0
from phydrax._trainable import NonTrainableState
from phydrax.nn.operator.representations import O3Features, O3Representation


class RadialBasis(eqx.Module, NonTrainableState):
    """Smooth compactly supported Gaussian radial basis."""

    centers: Array
    radius: float = eqx.field(static=True)
    width: float = eqx.field(static=True)
    count: int = eqx.field(static=True)

    def __init__(self, count: int, radius: float, /):
        self.count = int(count)
        self.radius = float(radius)
        if self.count <= 0 or self.radius <= 0.0:
            raise ValueError("Radial basis count and radius must be positive.")
        self.centers = jnp.linspace(0.0, self.radius, self.count)
        spacing = self.radius / float(max(1, self.count - 1))
        self.width = max(spacing, self.radius / float(self.count))

    def __call__(self, distance: Array, /) -> Array:
        normalized = (distance[..., None] - self.centers) / self.width
        gaussian = jnp.exp(-(normalized**2))
        envelope = jnp.where(
            distance < self.radius,
            0.5 * (jnp.cos(jnp.pi * distance / self.radius) + 1.0),
            0.0,
        )
        return gaussian * envelope[..., None]


class RadialMap(eqx.Module):
    """Distance-conditioned linear map between irrep multiplicities."""

    weight: Array
    in_multiplicity: int = eqx.field(static=True)
    out_multiplicity: int = eqx.field(static=True)

    def __init__(
        self,
        basis_count: int,
        in_multiplicity: int,
        out_multiplicity: int,
        /,
        *,
        key: Key[Array, ""],
    ):
        self.in_multiplicity = int(in_multiplicity)
        self.out_multiplicity = int(out_multiplicity)
        scale = 1.0 / sqrt(float(int(basis_count) * self.in_multiplicity))
        self.weight = scale * jr.normal(
            key,
            (int(basis_count), self.out_multiplicity, self.in_multiplicity),
        )

    def __call__(self, radial: Array, /) -> Array:
        return ein.contract("...k,koi->...oi", radial, self.weight)


def _radial_map(
    basis_count: int,
    in_multiplicity: int,
    out_multiplicity: int,
    key: Key[Array, ""],
    /,
) -> RadialMap | None:
    if int(in_multiplicity) == 0 or int(out_multiplicity) == 0:
        return None
    return RadialMap(
        basis_count,
        in_multiplicity,
        out_multiplicity,
        key=key,
    )


def _zero_features(
    batch: int,
    queries: int,
    representation: O3Representation,
    dtype: jnp.dtype,
    /,
) -> O3Features:
    return O3Features(
        scalars=jnp.zeros((batch, queries, representation.scalars), dtype=dtype),
        pseudoscalars=jnp.zeros(
            (batch, queries, representation.pseudoscalars), dtype=dtype
        ),
        vectors=jnp.zeros((batch, queries, representation.vectors, 3), dtype=dtype),
        pseudovectors=jnp.zeros(
            (batch, queries, representation.pseudovectors, 3), dtype=dtype
        ),
        tensors=jnp.zeros((batch, queries, representation.tensors, 3, 3), dtype=dtype),
        pseudotensors=jnp.zeros(
            (batch, queries, representation.pseudotensors, 3, 3), dtype=dtype
        ),
    )


def _symmetric_traceless(left: Array, right: Array, /) -> Array:
    outer = 0.5 * (
        ein.contract("...i,...j->...ij", left, right)
        + ein.contract("...i,...j->...ij", right, left)
    )
    trace = jnp.sum(left * right, axis=-1)
    identity = jnp.eye(3, dtype=outer.dtype)
    return outer - trace[..., None, None] * identity / 3.0


class EquivariantIntegralLayer(eqx.Module):
    """Quadrature-aware O(3)-equivariant source-to-target kernel integral."""

    in_representation: O3Representation
    out_representation: O3Representation
    radial_basis: RadialBasis
    ss: RadialMap | None
    sv: RadialMap | None
    st: RadialMap | None
    pp: RadialMap | None
    pa: RadialMap | None
    pt: RadialMap | None
    vs: RadialMap | None
    vv: RadialMap | None
    vv_longitudinal: RadialMap | None
    va_cross: RadialMap | None
    vt: RadialMap | None
    ap: RadialMap | None
    aa: RadialMap | None
    aa_longitudinal: RadialMap | None
    av_cross: RadialMap | None
    apt: RadialMap | None
    ts: RadialMap | None
    tv: RadialMap | None
    tt: RadialMap | None
    tt_longitudinal: RadialMap | None
    tp: RadialMap | None
    ta: RadialMap | None
    tpt: RadialMap | None
    tpt_longitudinal: RadialMap | None

    def __init__(
        self,
        in_representation: O3Representation,
        out_representation: O3Representation,
        /,
        *,
        radius: float,
        radial_basis_size: int = 16,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_representation = in_representation
        self.out_representation = out_representation
        self.radial_basis = RadialBasis(radial_basis_size, radius)
        keys = iter(jr.split(key, 24))
        source = in_representation
        target = out_representation
        basis = int(radial_basis_size)
        self.ss = _radial_map(basis, source.scalars, target.scalars, next(keys))
        self.sv = _radial_map(basis, source.vectors, target.scalars, next(keys))
        self.st = _radial_map(basis, source.tensors, target.scalars, next(keys))
        self.pp = _radial_map(
            basis, source.pseudoscalars, target.pseudoscalars, next(keys)
        )
        self.pa = _radial_map(
            basis, source.pseudovectors, target.pseudoscalars, next(keys)
        )
        self.pt = _radial_map(
            basis, source.pseudotensors, target.pseudoscalars, next(keys)
        )
        self.vs = _radial_map(basis, source.scalars, target.vectors, next(keys))
        self.vv = _radial_map(basis, source.vectors, target.vectors, next(keys))
        self.vv_longitudinal = _radial_map(
            basis, source.vectors, target.vectors, next(keys)
        )
        self.va_cross = _radial_map(
            basis, source.pseudovectors, target.vectors, next(keys)
        )
        self.vt = _radial_map(basis, source.tensors, target.vectors, next(keys))
        self.ap = _radial_map(
            basis, source.pseudoscalars, target.pseudovectors, next(keys)
        )
        self.aa = _radial_map(
            basis, source.pseudovectors, target.pseudovectors, next(keys)
        )
        self.aa_longitudinal = _radial_map(
            basis, source.pseudovectors, target.pseudovectors, next(keys)
        )
        self.av_cross = _radial_map(
            basis, source.vectors, target.pseudovectors, next(keys)
        )
        self.apt = _radial_map(
            basis, source.pseudotensors, target.pseudovectors, next(keys)
        )
        self.ts = _radial_map(basis, source.scalars, target.tensors, next(keys))
        self.tv = _radial_map(basis, source.vectors, target.tensors, next(keys))
        self.tt = _radial_map(basis, source.tensors, target.tensors, next(keys))
        self.tt_longitudinal = _radial_map(
            basis, source.tensors, target.tensors, next(keys)
        )
        self.tp = _radial_map(
            basis, source.pseudoscalars, target.pseudotensors, next(keys)
        )
        self.ta = _radial_map(
            basis, source.pseudovectors, target.pseudotensors, next(keys)
        )
        self.tpt = _radial_map(
            basis, source.pseudotensors, target.pseudotensors, next(keys)
        )
        self.tpt_longitudinal = _radial_map(
            basis, source.pseudotensors, target.pseudotensors, next(keys)
        )

    @staticmethod
    def _reduce(message: Array, weights: Array, /) -> Array:
        return ein.contract("bqs...,bs->bq...", message, weights)

    def __call__(
        self,
        source_values: Array,
        source_coordinates: Array,
        target_coordinates: Array,
        source_weights: Array,
        /,
        *,
        source_mask: Array | None = None,
        target_mask: Array | None = None,
    ) -> Array:
        source_array = jnp.asarray(source_values)
        batch, source_count = source_array.shape[:2]
        query_count = int(target_coordinates.shape[1])
        features = self.in_representation.split(source_array)
        displacement = (
            jnp.asarray(target_coordinates)[:, :, None, :]
            - jnp.asarray(source_coordinates)[:, None, :, :]
        )
        distance = jnp.linalg.norm(displacement, axis=-1)
        unit = jnp.where(
            distance[..., None] > 0.0,
            displacement / jnp.maximum(distance[..., None], 1e-12),
            0.0,
        )
        radial = self.radial_basis(distance)
        dyad = _symmetric_traceless(unit, unit)
        weights = jnp.asarray(source_weights, dtype=source_array.dtype)
        if source_mask is not None:
            weights = weights * jnp.asarray(source_mask, dtype=weights.dtype)
        output = _zero_features(
            int(batch), query_count, self.out_representation, source_array.dtype
        )
        scalars = output.scalars
        pseudoscalars = output.pseudoscalars
        vectors = output.vectors
        pseudovectors = output.pseudovectors
        tensors = output.tensors
        pseudotensors = output.pseudotensors

        if self.ss is not None:
            scalars = scalars + self._reduce(
                ein.contract("bqsoi,bsi->bqso", self.ss(radial), features.scalars),
                weights,
            )
        vector_projection = ein.contract("bsic,bqsc->bqsi", features.vectors, unit)
        if self.sv is not None:
            scalars = scalars + self._reduce(
                ein.contract("bqsoi,bqsi->bqso", self.sv(radial), vector_projection),
                weights,
            )
        tensor_projection = ein.contract("bsicd,bqscd->bqsi", features.tensors, dyad)
        if self.st is not None:
            scalars = scalars + self._reduce(
                ein.contract("bqsoi,bqsi->bqso", self.st(radial), tensor_projection),
                weights,
            )
        if self.pp is not None:
            pseudoscalars = pseudoscalars + self._reduce(
                ein.contract("bqsoi,bsi->bqso", self.pp(radial), features.pseudoscalars),
                weights,
            )
        axial_projection = ein.contract("bsic,bqsc->bqsi", features.pseudovectors, unit)
        if self.pa is not None:
            pseudoscalars = pseudoscalars + self._reduce(
                ein.contract("bqsoi,bqsi->bqso", self.pa(radial), axial_projection),
                weights,
            )
        pseudotensor_projection = ein.contract(
            "bsicd,bqscd->bqsi", features.pseudotensors, dyad
        )
        if self.pt is not None:
            pseudoscalars = pseudoscalars + self._reduce(
                ein.contract(
                    "bqsoi,bqsi->bqso", self.pt(radial), pseudotensor_projection
                ),
                weights,
            )

        if self.vs is not None:
            coefficient = ein.contract(
                "bqsoi,bsi->bqso", self.vs(radial), features.scalars
            )
            vectors = vectors + self._reduce(
                coefficient[..., None] * unit[..., None, :], weights
            )
        if self.vv is not None:
            vectors = vectors + self._reduce(
                ein.contract("bqsoi,bsic->bqsoc", self.vv(radial), features.vectors),
                weights,
            )
        if self.vv_longitudinal is not None:
            coefficient = ein.contract(
                "bqsoi,bqsi->bqso",
                self.vv_longitudinal(radial),
                vector_projection,
            )
            vectors = vectors + self._reduce(
                coefficient[..., None] * unit[..., None, :], weights
            )
        if self.va_cross is not None:
            crossed = jnp.cross(
                unit[..., None, :], features.pseudovectors[:, None, ...], axis=-1
            )
            vectors = vectors + self._reduce(
                ein.contract("bqsoi,bqsic->bqsoc", self.va_cross(radial), crossed),
                weights,
            )
        if self.vt is not None:
            applied = ein.contract("bsicd,bqsd->bqsic", features.tensors, unit)
            vectors = vectors + self._reduce(
                ein.contract("bqsoi,bqsic->bqsoc", self.vt(radial), applied),
                weights,
            )

        if self.ap is not None:
            coefficient = ein.contract(
                "bqsoi,bsi->bqso", self.ap(radial), features.pseudoscalars
            )
            pseudovectors = pseudovectors + self._reduce(
                coefficient[..., None] * unit[..., None, :], weights
            )
        if self.aa is not None:
            pseudovectors = pseudovectors + self._reduce(
                ein.contract(
                    "bqsoi,bsic->bqsoc", self.aa(radial), features.pseudovectors
                ),
                weights,
            )
        if self.aa_longitudinal is not None:
            coefficient = ein.contract(
                "bqsoi,bqsi->bqso",
                self.aa_longitudinal(radial),
                axial_projection,
            )
            pseudovectors = pseudovectors + self._reduce(
                coefficient[..., None] * unit[..., None, :], weights
            )
        if self.av_cross is not None:
            crossed = jnp.cross(
                unit[..., None, :], features.vectors[:, None, ...], axis=-1
            )
            pseudovectors = pseudovectors + self._reduce(
                ein.contract("bqsoi,bqsic->bqsoc", self.av_cross(radial), crossed),
                weights,
            )
        if self.apt is not None:
            applied = ein.contract("bsicd,bqsd->bqsic", features.pseudotensors, unit)
            pseudovectors = pseudovectors + self._reduce(
                ein.contract("bqsoi,bqsic->bqsoc", self.apt(radial), applied),
                weights,
            )

        if self.ts is not None:
            coefficient = ein.contract(
                "bqsoi,bsi->bqso", self.ts(radial), features.scalars
            )
            tensors = tensors + self._reduce(
                coefficient[..., None, None] * dyad[..., None, :, :], weights
            )
        vector_tensor = _symmetric_traceless(
            features.vectors[:, None, ...], unit[..., None, :]
        )
        if self.tv is not None:
            tensors = tensors + self._reduce(
                ein.contract("bqsoi,bqsicd->bqsocd", self.tv(radial), vector_tensor),
                weights,
            )
        if self.tt is not None:
            tensors = tensors + self._reduce(
                ein.contract("bqsoi,bsicd->bqsocd", self.tt(radial), features.tensors),
                weights,
            )
        if self.tt_longitudinal is not None:
            coefficient = ein.contract(
                "bqsoi,bqsi->bqso",
                self.tt_longitudinal(radial),
                tensor_projection,
            )
            tensors = tensors + self._reduce(
                coefficient[..., None, None] * dyad[..., None, :, :], weights
            )

        if self.tp is not None:
            coefficient = ein.contract(
                "bqsoi,bsi->bqso", self.tp(radial), features.pseudoscalars
            )
            pseudotensors = pseudotensors + self._reduce(
                coefficient[..., None, None] * dyad[..., None, :, :], weights
            )
        axial_tensor = _symmetric_traceless(
            features.pseudovectors[:, None, ...], unit[..., None, :]
        )
        if self.ta is not None:
            pseudotensors = pseudotensors + self._reduce(
                ein.contract("bqsoi,bqsicd->bqsocd", self.ta(radial), axial_tensor),
                weights,
            )
        if self.tpt is not None:
            pseudotensors = pseudotensors + self._reduce(
                ein.contract(
                    "bqsoi,bsicd->bqsocd",
                    self.tpt(radial),
                    features.pseudotensors,
                ),
                weights,
            )
        if self.tpt_longitudinal is not None:
            coefficient = ein.contract(
                "bqsoi,bqsi->bqso",
                self.tpt_longitudinal(radial),
                pseudotensor_projection,
            )
            pseudotensors = pseudotensors + self._reduce(
                coefficient[..., None, None] * dyad[..., None, :, :], weights
            )

        packed = self.out_representation.join(
            O3Features(
                scalars=scalars,
                pseudoscalars=pseudoscalars,
                vectors=vectors,
                pseudovectors=pseudovectors,
                tensors=tensors,
                pseudotensors=pseudotensors,
            )
        )
        if target_mask is not None:
            packed = packed * jnp.asarray(target_mask, dtype=packed.dtype)[..., None]
        return packed


class O3PointwiseLinear(eqx.Module):
    """Equivariant multiplicity mixing that never mixes irrep types."""

    in_representation: O3Representation
    out_representation: O3Representation
    scalar_weight: Array | None
    pseudoscalar_weight: Array | None
    vector_weight: Array | None
    pseudovector_weight: Array | None
    tensor_weight: Array | None
    pseudotensor_weight: Array | None
    scalar_bias: Array | None

    def __init__(
        self,
        in_representation: O3Representation,
        out_representation: O3Representation,
        /,
        *,
        use_scalar_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_representation = in_representation
        self.out_representation = out_representation
        keys = iter(jr.split(key, 7))

        def weight(in_count: int, out_count: int) -> Array | None:
            weight_key = next(keys)
            if in_count == 0 or out_count == 0:
                return None
            return jr.normal(weight_key, (out_count, in_count)) / sqrt(float(in_count))

        self.scalar_weight = weight(in_representation.scalars, out_representation.scalars)
        self.pseudoscalar_weight = weight(
            in_representation.pseudoscalars, out_representation.pseudoscalars
        )
        self.vector_weight = weight(in_representation.vectors, out_representation.vectors)
        self.pseudovector_weight = weight(
            in_representation.pseudovectors, out_representation.pseudovectors
        )
        self.tensor_weight = weight(in_representation.tensors, out_representation.tensors)
        self.pseudotensor_weight = weight(
            in_representation.pseudotensors, out_representation.pseudotensors
        )
        bias_key = next(keys)
        self.scalar_bias = (
            jnp.zeros((out_representation.scalars,))
            + 0.01 * jr.normal(bias_key, (out_representation.scalars,))
            if use_scalar_bias and out_representation.scalars > 0
            else None
        )

    @staticmethod
    def _mix(weight: Array | None, values: Array, out_count: int, /) -> Array:
        if weight is None:
            return jnp.zeros(values.shape[:-1] + (out_count,), dtype=values.dtype)
        return ein.contract("oi,...i->...o", weight, values)

    @staticmethod
    def _mix_geometric(
        weight: Array | None,
        values: Array,
        out_count: int,
        /,
    ) -> Array:
        if weight is None:
            return jnp.zeros(
                values.shape[:-2] + (out_count, values.shape[-1]), dtype=values.dtype
            )
        return ein.contract("oi,...ic->...oc", weight, values)

    @staticmethod
    def _mix_tensor(
        weight: Array | None,
        values: Array,
        out_count: int,
        /,
    ) -> Array:
        if weight is None:
            return jnp.zeros(values.shape[:-3] + (out_count, 3, 3), dtype=values.dtype)
        return ein.contract("oi,...icd->...ocd", weight, values)

    def __call__(self, values: Array, /) -> Array:
        features = self.in_representation.split(values)
        scalars = self._mix(
            self.scalar_weight, features.scalars, self.out_representation.scalars
        )
        if self.scalar_bias is not None:
            scalars = scalars + self.scalar_bias
        return self.out_representation.join(
            O3Features(
                scalars=scalars,
                pseudoscalars=self._mix(
                    self.pseudoscalar_weight,
                    features.pseudoscalars,
                    self.out_representation.pseudoscalars,
                ),
                vectors=self._mix_geometric(
                    self.vector_weight,
                    features.vectors,
                    self.out_representation.vectors,
                ),
                pseudovectors=self._mix_geometric(
                    self.pseudovector_weight,
                    features.pseudovectors,
                    self.out_representation.pseudovectors,
                ),
                tensors=self._mix_tensor(
                    self.tensor_weight,
                    features.tensors,
                    self.out_representation.tensors,
                ),
                pseudotensors=self._mix_tensor(
                    self.pseudotensor_weight,
                    features.pseudotensors,
                    self.out_representation.pseudotensors,
                ),
            )
        )


def o3_gated_activation(values: Array, representation: O3Representation, /) -> Array:
    """Apply parity-safe scalar and invariant norm-gated nonlinearities."""
    features = representation.split(values)

    def gate_geometric(array: Array, geometric_axes: tuple[int, ...]) -> Array:
        norm = jnp.sqrt(jnp.sum(array**2, axis=geometric_axes, keepdims=True) + 1e-12)
        return array * jnn.sigmoid(norm)

    return representation.join(
        O3Features(
            scalars=jnn.gelu(features.scalars),
            pseudoscalars=jnp.tanh(features.pseudoscalars),
            vectors=gate_geometric(features.vectors, (-1,)),
            pseudovectors=gate_geometric(features.pseudovectors, (-1,)),
            tensors=gate_geometric(features.tensors, (-2, -1)),
            pseudotensors=gate_geometric(features.pseudotensors, (-2, -1)),
        )
    )
