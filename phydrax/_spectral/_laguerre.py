#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_genlaguerre

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract


_DEFAULT_PRECOMPUTE_BYTES = 512 * 1024**2
_RADIAL_LAGUERRE_NORMALIZATION = "generalized-laguerre-alpha-2-r2dr"


def _array_bytes(tree: object, /) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, (np.ndarray, jax.Array))
    )


class RadialLaguerrePlan(StrictModule, NonTrainableState):
    """Exact order-two generalized Laguerre transform on the radial half-line.

    The normalized basis is orthonormal under the physical ``r**2 dr`` pairing.
    ``tau`` is a radial length scale, not a finite outer boundary.
    """

    dimensionless_nodes: Array
    nodes: Array
    quadrature_weights: Array
    sqrt_quadrature_weights: Array
    inverse_sqrt_quadrature_weights: Array
    balanced_basis: Array
    radial_bandlimit: int
    tau: float
    sample_shape: tuple[int]
    coefficient_shape: tuple[int]
    normalization: str
    fingerprint: str
    layout_id: str
    max_precompute_bytes: int
    orthogonality_defect: float

    def __init__(
        self,
        radial_bandlimit: int,
        /,
        *,
        tau: float = 1.0,
        max_precompute_bytes: int = _DEFAULT_PRECOMPUTE_BYTES,
    ):
        selected_bandlimit = int(radial_bandlimit)
        selected_tau = float(tau)
        selected_limit = int(max_precompute_bytes)
        if selected_bandlimit < 1:
            raise ValueError("radial_bandlimit must be positive.")
        if not np.isfinite(selected_tau) or selected_tau <= 0.0:
            raise ValueError("tau must be finite and positive.")
        if selected_limit <= 0:
            raise ValueError("max_precompute_bytes must be positive.")

        # eigh_tridiagonal materializes eigenvectors and workspace in addition to
        # the six persistent arrays. Reject before entering the host eigensolver.
        preparation_estimate = 8 * (4 * selected_bandlimit**2 + 12 * selected_bandlimit)
        if preparation_estimate > selected_limit:
            raise ValueError(
                "Radial Laguerre preparation exceeds max_precompute_bytes; "
                f"estimated {preparation_estimate} bytes."
            )

        degree = np.arange(selected_bandlimit, dtype=float)
        diagonal = 2.0 * degree + 3.0
        recurrence_degree = np.arange(1, selected_bandlimit, dtype=float)
        off_diagonal = -np.sqrt(recurrence_degree * (recurrence_degree + 2.0))
        dimensionless_nodes, eigenvectors = eigh_tridiagonal(
            diagonal,
            off_diagonal,
            check_finite=True,
        )
        order = np.argsort(dimensionless_nodes)
        dimensionless_nodes = np.asarray(dimensionless_nodes[order], dtype=float)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
        for node_index, node in enumerate(dimensionless_nodes):
            anchor = int(np.argmax(np.abs(eigenvectors[:, node_index])))
            expected_anchor = float(eval_genlaguerre(anchor, 2, node))
            if not np.isfinite(expected_anchor) or expected_anchor == 0.0:
                raise ValueError(
                    "radial_bandlimit exceeds the representable float64 Laguerre range."
                )
            if np.signbit(eigenvectors[anchor, node_index]) != np.signbit(
                expected_anchor
            ):
                eigenvectors[:, node_index] *= -1.0
        balanced_basis = eigenvectors.T

        terminal_polynomial = np.asarray(
            eval_genlaguerre(
                selected_bandlimit + 1,
                2,
                dimensionless_nodes,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(terminal_polynomial)) or np.any(
            terminal_polynomial == 0.0
        ):
            raise ValueError(
                "radial_bandlimit exceeds the representable float64 Laguerre range."
            )
        log_gauss_weights = (
            np.log(selected_bandlimit + 2.0)
            + np.log(dimensionless_nodes)
            - np.log(selected_bandlimit + 1.0)
            - 2.0 * np.log(np.abs(terminal_polynomial))
        )
        log_weights = 3.0 * np.log(selected_tau) + dimensionless_nodes + log_gauss_weights
        sqrt_weights = np.exp(0.5 * log_weights)
        inverse_sqrt_weights = np.exp(-0.5 * log_weights)
        quadrature_weights = np.exp(log_weights)
        nodes = selected_tau * dimensionless_nodes
        arrays = (
            dimensionless_nodes,
            nodes,
            quadrature_weights,
            sqrt_weights,
            inverse_sqrt_weights,
            balanced_basis,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError(
                "tau and radial_bandlimit produce non-finite float64 Laguerre data."
            )
        if np.any(np.diff(dimensionless_nodes) <= 0.0):
            raise RuntimeError("Golub-Welsch returned unordered Laguerre nodes.")
        identity = np.eye(selected_bandlimit, dtype=float)
        defect = float(np.max(np.abs(balanced_basis.T @ balanced_basis - identity)))
        tolerance = 256.0 * selected_bandlimit * np.finfo(float).eps
        if defect > tolerance:
            raise RuntimeError(
                "Golub-Welsch Laguerre basis failed its orthogonality certificate."
            )

        self.dimensionless_nodes = jnp.asarray(dimensionless_nodes)
        self.nodes = jnp.asarray(nodes)
        self.quadrature_weights = jnp.asarray(quadrature_weights)
        self.sqrt_quadrature_weights = jnp.asarray(sqrt_weights)
        self.inverse_sqrt_quadrature_weights = jnp.asarray(inverse_sqrt_weights)
        self.balanced_basis = jnp.asarray(balanced_basis)
        self.radial_bandlimit = selected_bandlimit
        self.tau = selected_tau
        self.sample_shape = (selected_bandlimit,)
        self.coefficient_shape = (selected_bandlimit,)
        self.normalization = _RADIAL_LAGUERRE_NORMALIZATION
        self.max_precompute_bytes = selected_limit
        self.orthogonality_defect = defect
        self.layout_id = canonical_fingerprint(
            {
                "kind": "radial-laguerre-layout-v1",
                "radial_bandlimit": selected_bandlimit,
                "normalization": self.normalization,
            }
        )
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "radial-laguerre-plan-v1",
                "radial_bandlimit": selected_bandlimit,
                "tau": selected_tau,
                "normalization": self.normalization,
                "nodes": array_tree_fingerprint(dimensionless_nodes),
            }
        )
        if self.precompute_bytes > selected_limit:
            raise ValueError(
                "Radial Laguerre materialization exceeds max_precompute_bytes; "
                f"materialized {self.precompute_bytes} bytes."
            )

    @property
    def transform_id(self) -> str:
        """Mathematical radial-transform identity."""
        return self.fingerprint

    @property
    def precompute_bytes(self) -> int:
        """Bytes retained by the prepared radial transform."""
        return _array_bytes(
            (
                self.dimensionless_nodes,
                self.nodes,
                self.quadrature_weights,
                self.sqrt_quadrature_weights,
                self.inverse_sqrt_quadrature_weights,
                self.balanced_basis,
            )
        )

    @property
    def execution_id(self) -> str:
        """Identity of the Golub-Welsch execution realization."""
        return canonical_fingerprint(
            {
                "kind": "radial-laguerre-execution-v1",
                "transform": self.transform_id,
                "construction": "symmetric-golub-welsch",
                "precompute_bytes": self.precompute_bytes,
                "basis": array_tree_fingerprint(np.asarray(self.balanced_basis)),
            }
        )

    def _analysis_field(self, values: Array, /) -> Array:
        weighted = values * self.sqrt_quadrature_weights
        return contract("rp,r->p", self.balanced_basis, weighted)

    def _synthesis_field(self, coefficients: Array, /) -> Array:
        values = contract("rp,p->r", self.balanced_basis, coefficients)
        return values * self.inverse_sqrt_quadrature_weights

    def analysis(self, values: ArrayLike, /) -> Array:
        """Transform scalar or channel-last radial samples to Laguerre modes."""
        array = jnp.asarray(values)
        scalar = array.ndim >= 1 and int(array.shape[-1]) == self.radial_bandlimit
        if scalar:
            leading_shape = tuple(int(size) for size in array.shape[:-1])
            fields = array.reshape((prod(leading_shape), self.radial_bandlimit))
        elif array.ndim >= 2 and int(array.shape[-2]) == self.radial_bandlimit:
            leading_shape = tuple(int(size) for size in array.shape[:-2])
            channels = int(array.shape[-1])
            fields = jnp.moveaxis(array, -1, -2).reshape(
                (prod(leading_shape) * channels, self.radial_bandlimit)
            )
        else:
            raise ValueError(
                "Radial analysis expects (..., radial_bandlimit) or "
                "(..., radial_bandlimit, channels)."
            )
        coefficients = jax.vmap(self._analysis_field)(fields)
        if scalar:
            return coefficients.reshape(leading_shape + self.coefficient_shape)
        result = coefficients.reshape(leading_shape + (channels, self.radial_bandlimit))
        return jnp.moveaxis(result, -2, -1)

    def synthesis(self, coefficients: ArrayLike, /) -> Array:
        """Transform scalar or channel-last Laguerre modes to radial samples."""
        array = jnp.asarray(coefficients)
        scalar = array.ndim >= 1 and int(array.shape[-1]) == self.radial_bandlimit
        if scalar:
            leading_shape = tuple(int(size) for size in array.shape[:-1])
            fields = array.reshape((prod(leading_shape), self.radial_bandlimit))
        elif array.ndim >= 2 and int(array.shape[-2]) == self.radial_bandlimit:
            leading_shape = tuple(int(size) for size in array.shape[:-2])
            channels = int(array.shape[-1])
            fields = jnp.moveaxis(array, -1, -2).reshape(
                (prod(leading_shape) * channels, self.radial_bandlimit)
            )
        else:
            raise ValueError(
                "Radial synthesis expects (..., radial_bandlimit) or "
                "(..., radial_bandlimit, channels)."
            )
        values = jax.vmap(self._synthesis_field)(fields)
        if scalar:
            return values.reshape(leading_shape + self.sample_shape)
        result = values.reshape(leading_shape + (channels, self.radial_bandlimit))
        return jnp.moveaxis(result, -2, -1)


__all__ = ["RadialLaguerrePlan"]
