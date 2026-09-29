#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified trace-inverse constants from local flux/energy pencils.

The reference pencil is a 1-D linear element on ``[0, h]`` with diffusivity
``kappa`` and its flux at ``x = h``: ``q(v) = kappa (v_1 - v_0) / h`` and
``a(v, v) = kappa (v_1 - v_0)^2 / h``, so the sharp constant is ``kappa / h``
independently of the deflation scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization import certify_trace_inverse, TraceInverseEvidence


def _interval(kappa: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = kappa * np.asarray([[[-1.0 / h, 1.0 / h]]])
    energy = kappa / h * np.asarray([[[1.0, -1.0], [-1.0, 1.0]]])
    moments = np.asarray([[0.5 * h, 0.5 * h]])
    return flux, energy, moments


def _certify(
    flux: np.ndarray, energy: np.ndarray, moments: np.ndarray, valid: np.ndarray
) -> TraceInverseEvidence:
    return certify_trace_inverse(
        flux,
        energy,
        moments,
        valid,
        owner_id="interval-owner",
        flux_action_id="interval-flux",
        facets=np.arange(flux.shape[0], dtype=np.int32),
        side_cells=np.zeros((flux.shape[0],), dtype=np.int32),
        facet_rule_id="point",
        facet_exact_degree=2,
    )


def test_interval_constant_is_kappa_over_h_and_padding_is_inert() -> None:
    """Two facets of one cell: the sharp constant, a padded slot, and the multiplicity."""
    flux, energy, moments = _interval(3.0, 0.25)
    padded_flux = np.concatenate((flux, np.zeros((1, 1, 1))), axis=-1)
    padded_energy = np.pad(energy, ((0, 0), (0, 1), (0, 1)))
    padded_moments = np.pad(moments, ((0, 0), (0, 1)))
    evidence = _certify(
        np.concatenate((padded_flux, padded_flux)),
        np.concatenate((padded_energy, padded_energy)),
        np.concatenate((padded_moments, padded_moments)),
        np.asarray([[True, True, False], [True, True, False]]),
    )

    np.testing.assert_allclose(np.asarray(evidence.constants), 12.0, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(evidence.cell_multiplicity), (2, 2))
    assert np.all(np.asarray(evidence.relative_residuals) < 1e-12)


def test_energy_kernel_larger_than_the_constants_is_refused() -> None:
    """A block energy with two decoupled constants leaves two zero-energy modes."""
    link = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    energy = np.zeros((1, 4, 4))
    energy[0, :2, :2] = link
    energy[0, 2:, 2:] = link
    flux = np.asarray([[[-1.0, 1.0, 0.0, 0.0]]])
    with pytest.raises(ValueError, match="larger than the constants"):
        _certify(flux, energy, np.full((1, 4), 0.25), np.ones((1, 4), dtype=bool))


def test_vanishing_energy_is_refused() -> None:
    flux, _, moments = _interval(1.0, 0.5)
    with pytest.raises(ValueError, match="nonzero energy"):
        _certify(flux, np.zeros((1, 2, 2)), moments, np.ones((1, 2), dtype=bool))


def test_flux_carried_by_the_energy_kernel_has_no_finite_constant() -> None:
    """``q(v) = v_0 + v_1`` is nonzero on the zero-energy constant: sup q^2/a = inf.

    The constant-deflated pencil alone would report the finite value 1.
    """
    _, energy, moments = _interval(1.0, 1.0)
    with pytest.raises(ValueError, match="does not vanish on the cell energy kernel"):
        _certify(np.asarray([[[1.0, 1.0]]]), energy, moments, np.ones((1, 2), dtype=bool))


def test_positive_definite_energy_is_refused() -> None:
    """Deflating a kernel-free energy would underestimate the sharp constant."""
    flux, energy, moments = _interval(1.0, 1.0)
    with pytest.raises(ValueError, match="has no kernel"):
        _certify(flux, energy + np.eye(2), moments, np.ones((1, 2), dtype=bool))
