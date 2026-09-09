# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Numeric coordinate-representation ABI; molecular chemistry stays domain-owned."""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._support import PreparedCoordinateSupport


class CoordinateEncoding(StrictModule):
    """Representation coordinates and one validity bit per leading case."""

    coordinates: Array
    valid: Array
    representation_id: str = eqx.field(static=True)


class CoordinateDecoding(StrictModule):
    """Decoded Cartesian coordinates and explicit numeric decoder evidence."""

    positions: Array
    valid: Array
    residuals: Array
    representation_id: str = eqx.field(static=True)


class AbstractCoordinateDecoder(StrictModule, NonTrainableState):
    """Fixed-support differentiable encoding/decoding contract.

    Implementations own only a numerical representation. They must retain every
    input case, return Cartesian positions ending in ``(atom_capacity, 3)``, and
    report invalid representations rather than repairing or resampling them.
    """

    @property
    @abc.abstractmethod
    def coordinate_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def support_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def representation_id(self) -> str: ...

    @abc.abstractmethod
    def encode(self, positions) -> CoordinateEncoding: ...

    @abc.abstractmethod
    def decode(self, coordinates) -> CoordinateDecoding: ...

    @abc.abstractmethod
    def project(self, coordinates) -> Array:
        """Apply representation identities used by both source and velocity."""


class CartesianCoordinateDecoder(AbstractCoordinateDecoder):
    """Original mass-centred Cartesian representation, preserved as baseline."""

    support: PreparedCoordinateSupport
    _support_id: str = eqx.field(static=True)
    _representation_id: str = eqx.field(static=True)
    _coordinate_size: int = eqx.field(static=True)

    def __init__(self, support: PreparedCoordinateSupport):
        if not isinstance(support, PreparedCoordinateSupport):
            raise TypeError("Cartesian decoding requires PreparedCoordinateSupport.")
        self.support = support
        self._support_id = support.support_id
        self._representation_id = "cartesian-mass-centered:" + support.support_id
        self._coordinate_size = support.dimension

    @property
    def coordinate_size(self):
        return self._coordinate_size

    @property
    def support_id(self):
        return self._support_id

    @property
    def representation_id(self):
        return self._representation_id

    def encode(self, positions):
        values = jnp.asarray(positions)
        if values.shape[-2:] != (self.support.template.atom_capacity, 3):
            raise ValueError("Cartesian encoder input must end in (atom_capacity, 3).")
        canonical, gauge_valid = self.support.canonicalize(values)
        finite = jnp.all(
            jnp.isfinite(
                jnp.where(self.support.template.atom_mask[0, :, None], values, 0.0)
            ),
            axis=(-2, -1),
        )
        return CoordinateEncoding(
            canonical.reshape(values.shape[:-2] + (self.coordinate_size,)),
            finite & gauge_valid,
            self.representation_id,
        )

    def decode(self, coordinates):
        values = jnp.asarray(coordinates)
        if values.shape[-1:] != (self.coordinate_size,):
            raise ValueError(
                "Cartesian decoder input must end in its fixed coordinate size."
            )
        positions = values.reshape(
            values.shape[:-1] + (self.support.template.atom_capacity, 3)
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.where(self.support.template.atom_mask[0, :, None], positions, 0.0)
            ),
            axis=(-2, -1),
        )
        _, gauge_valid = self.support.canonicalize(positions)
        return CoordinateDecoding(
            positions,
            finite & gauge_valid,
            jnp.zeros(values.shape[:-1] + (0,), dtype=values.dtype),
            self.representation_id,
        )

    def project(self, coordinates):
        values = jnp.asarray(coordinates)
        if values.shape != (self.coordinate_size,):
            raise ValueError(
                "Cartesian model state must have its fixed one-case coordinate shape."
            )
        return self.support.center(
            values.reshape((self.support.template.atom_capacity, 3))
        ).reshape((self.coordinate_size,))


__all__ = [
    "AbstractCoordinateDecoder",
    "CartesianCoordinateDecoder",
    "CoordinateDecoding",
    "CoordinateEncoding",
]
