#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..data import OperatorAxis


SpectralBasis = Literal["fourier", "sine", "cosine", "legendre"]


def _trapezoid_weights(nodes: Array, /) -> Array:
    nodes_ = jnp.asarray(nodes, dtype=float)
    if int(nodes_.shape[0]) == 1:
        return jnp.ones_like(nodes_)
    interior = 0.5 * (nodes_[2:] - nodes_[:-2])
    return jnp.concatenate(
        (
            (0.5 * (nodes_[1] - nodes_[0]))[None],
            interior,
            (0.5 * (nodes_[-1] - nodes_[-2]))[None],
        )
    )


def _normalized_nodes(axis: OperatorAxis, /) -> Array:
    nodes = jnp.asarray(axis.nodes, dtype=float)
    span = nodes[-1] - nodes[0]
    if axis.periodic and int(nodes.shape[0]) > 1:
        if axis.quadrature_weights is None:
            span = span + jnp.mean(jnp.diff(nodes))
        else:
            span = jnp.sum(jnp.asarray(axis.quadrature_weights, dtype=float))
    nodes = eqx.error_if(
        nodes,
        jnp.isclose(span, 0.0),
        "Spectral basis nodes must span a nonzero interval.",
    )
    return (nodes - nodes[0]) / span


def _basis_matrix(axis: OperatorAxis, basis: SpectralBasis, modes: int, /) -> Array:
    coordinate = _normalized_nodes(axis)
    columns: list[Array] = []
    if basis == "fourier":
        columns.append(jnp.ones_like(coordinate))
        frequency = 1
        while len(columns) < modes:
            columns.append(jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * frequency * coordinate))
            if len(columns) < modes:
                columns.append(
                    jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * frequency * coordinate)
                )
            frequency += 1
    elif basis == "sine":
        columns.extend(
            jnp.sqrt(2.0) * jnp.sin(jnp.pi * (index + 1) * coordinate)
            for index in range(modes)
        )
    elif basis == "cosine":
        columns.append(jnp.ones_like(coordinate))
        columns.extend(
            jnp.sqrt(2.0) * jnp.cos(jnp.pi * index * coordinate)
            for index in range(1, modes)
        )
    elif basis == "legendre":
        z = 2.0 * coordinate - 1.0
        columns.append(jnp.ones_like(z))
        if modes > 1:
            columns.append(z)
        for degree in range(2, modes):
            columns.append(
                ((2.0 * degree - 1.0) * z * columns[-1] - (degree - 1.0) * columns[-2])
                / float(degree)
            )
    else:
        raise ValueError("basis must be 'fourier', 'sine', 'cosine', or 'legendre'.")
    return jnp.stack(columns[:modes], axis=-1)


def _analysis_matrix(axis: OperatorAxis, basis: SpectralBasis, modes: int, /) -> Array:
    basis_matrix = _basis_matrix(axis, basis, modes)
    weights = (
        _trapezoid_weights(axis.nodes)
        if axis.quadrature_weights is None
        else jnp.asarray(axis.quadrature_weights, dtype=float)
    )
    weighted_basis = weights[:, None] * basis_matrix
    gram = basis_matrix.T @ weighted_basis
    regularizer = jnp.finfo(basis_matrix.dtype).eps * jnp.trace(gram)
    return jnp.linalg.solve(
        gram + regularizer * jnp.eye(modes, dtype=gram.dtype),
        weighted_basis.T,
    )


def _apply_axis_matrix(values: Array, matrix: Array, axis: int, /) -> Array:
    transformed = jnp.tensordot(matrix, values, axes=((1,), (axis,)))
    return jnp.moveaxis(transformed, 0, axis)


class BasisTransformPlan(StrictModule):
    """Reusable analysis/synthesis matrices for one separable discretization."""

    analysis_matrices: tuple[Array, ...]
    synthesis_matrices: tuple[Array, ...]
    sample_shape: tuple[int, ...]
    bases: tuple[SpectralBasis, ...]
    n_modes: tuple[int, ...]

    def __init__(
        self,
        axes: Sequence[OperatorAxis],
        bases: Sequence[SpectralBasis],
        n_modes: Sequence[int],
        /,
    ):
        axes_ = tuple(axes)
        bases_ = tuple(bases)
        modes_ = tuple(int(mode) for mode in n_modes)
        if len(axes_) != len(bases_) or len(axes_) != len(modes_):
            raise ValueError("Transform plan axes, bases, and modes must align.")
        if any(mode > axis.size for mode, axis in zip(modes_, axes_, strict=True)):
            raise ValueError("Basis mode counts cannot exceed the available nodes.")
        self.analysis_matrices = tuple(
            _analysis_matrix(axis, basis, mode)
            for axis, basis, mode in zip(axes_, bases_, modes_, strict=True)
        )
        self.synthesis_matrices = tuple(
            _basis_matrix(axis, basis, mode)
            for axis, basis, mode in zip(axes_, bases_, modes_, strict=True)
        )
        self.sample_shape = tuple(axis.size for axis in axes_)
        self.bases = bases_
        self.n_modes = modes_


class BasisSpectralConvND(StrictModule):
    """Quadrature-projected spectral convolution on separable nonperiodic bases.

    Fourier uses an explicit real constant/cosine/sine basis, while sine, cosine,
    and Legendre policies encode common boundary and polynomial structures. Analysis
    is a weighted least-squares projection, so nonuniform nodes remain meaningful
    when their quadrature weights are supplied.
    """

    in_channels: int
    out_channels: int
    n_modes: tuple[int, ...]
    bases: tuple[SpectralBasis, ...]
    weight: Array

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_modes: int | Sequence[int],
        bases: SpectralBasis | Sequence[SpectralBasis],
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if isinstance(n_modes, int):
            modes = (int(n_modes),)
        else:
            modes = tuple(int(mode) for mode in n_modes)
        if isinstance(bases, str):
            bases_ = (cast(SpectralBasis, bases),) * len(modes)
        else:
            bases_ = tuple(bases)
        if not modes or any(mode <= 0 for mode in modes):
            raise ValueError("Every basis mode count must be positive.")
        if len(bases_) != len(modes):
            raise ValueError("bases and n_modes must have the same length.")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        for basis in bases_:
            if basis not in ("fourier", "sine", "cosine", "legendre"):
                raise ValueError(f"Unsupported spectral basis {basis!r}.")
        self.n_modes = modes
        self.bases = bases_
        scale = 1.0 / float(self.in_channels * self.out_channels)
        self.weight = scale * jr.normal(
            key,
            shape=(self.in_channels, self.out_channels, *self.n_modes),
        )

    def plan(self, axes: Sequence[OperatorAxis], /) -> BasisTransformPlan:
        """Precompute reusable transforms for a fixed discretization."""
        return BasisTransformPlan(axes, self.bases, self.n_modes)

    def __call__(
        self,
        values: Array,
        axes: Sequence[OperatorAxis],
        /,
        *,
        plan: BasisTransformPlan | None = None,
    ) -> Array:
        array = jnp.asarray(values)
        axes_ = tuple(axes)
        ndim = len(self.n_modes)
        if len(axes_) != ndim:
            raise ValueError(f"Expected {ndim} OperatorAxis values, got {len(axes_)}.")
        if array.ndim < ndim + 1 or int(array.shape[-1]) != self.in_channels:
            raise ValueError(
                "BasisSpectralConvND expects (..., spatial..., in_channels) input."
            )
        sample_shape = tuple(int(size) for size in array.shape[-ndim - 1 : -1])
        if sample_shape != tuple(axis.size for axis in axes_):
            raise ValueError(
                f"Input spatial shape {sample_shape} does not match axis sizes "
                f"{tuple(axis.size for axis in axes_)}."
            )
        spatial_start = array.ndim - ndim - 1
        transform = self.plan(axes_) if plan is None else plan
        if (
            transform.sample_shape != sample_shape
            or transform.bases != self.bases
            or transform.n_modes != self.n_modes
        ):
            raise ValueError("Basis transform plan does not match this layer or grid.")
        coefficients = array
        for index, analysis in enumerate(transform.analysis_matrices):
            coefficients = _apply_axis_matrix(
                coefficients,
                analysis,
                spatial_start + index,
            )

        letters = "abcdefghijklmnopqrstuvwxyz"[:ndim]
        coefficients = oe.contract(
            f"...{letters}i,io{letters}->...{letters}o",
            coefficients,
            self.weight,
        )
        output = coefficients
        for index in range(ndim - 1, -1, -1):
            output = _apply_axis_matrix(
                output,
                transform.synthesis_matrices[index],
                spatial_start + index,
            )
        return output


__all__ = ["BasisSpectralConvND", "BasisTransformPlan", "SpectralBasis"]
