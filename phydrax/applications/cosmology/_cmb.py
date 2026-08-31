#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._products import CosmologyProductProvenance
from ._scales import CosmologyScaleContract


CMBConvention: TypeAlias = Literal["Cl", "Dl"]


class CMBAngularPowerTable(StrictModule, NonTrainableState):
    multipoles: Array
    spectra: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    spectrum_names: tuple[str, ...] = eqx.field(static=True)
    convention: CMBConvention = eqx.field(static=True)
    units: str = eqx.field(static=True)
    lensed: bool = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        multipoles: ArrayLike,
        spectra: ArrayLike,
        spectrum_names: tuple[str, ...],
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        /,
        *,
        convention: CMBConvention,
        units: str,
        lensed: bool,
    ):
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be a CosmologyScaleContract.")
        if not isinstance(provenance, CosmologyProductProvenance):
            raise TypeError("provenance must be CosmologyProductProvenance.")
        if provenance.scale_id != scale.scale_id:
            raise ValueError("CMB scale and provenance disagree.")
        names = tuple(str(name).strip() for name in spectrum_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("spectrum_names must be unique non-empty identifiers.")
        ell_host = np.asarray(multipoles, dtype=float)
        spectra_host = np.asarray(spectra, dtype=float)
        if (
            ell_host.ndim != 1
            or ell_host.size < 2
            or spectra_host.shape != (len(names), ell_host.size)
        ):
            raise ValueError("CMB spectra must have shape (num_spectra, num_multipoles).")
        if (
            np.any(~np.isfinite(ell_host))
            or np.any(np.diff(ell_host) <= 0.0)
            or np.any(ell_host < 0.0)
            or np.any(~np.isfinite(spectra_host))
        ):
            raise ValueError("CMB multipoles and spectra must be finite and ordered.")
        if convention not in ("Cl", "Dl"):
            raise ValueError("CMB convention must be 'Cl' or 'Dl'.")
        unit_name = str(units).strip()
        if not unit_name:
            raise ValueError("CMB units must be non-empty.")
        if not isinstance(lensed, bool):
            raise TypeError("lensed must be a bool.")
        self.multipoles = jnp.asarray(ell_host)
        self.spectra = jnp.asarray(spectra_host)
        self.scale = scale
        self.provenance = provenance
        self.spectrum_names = names
        self.convention = convention
        self.units = unit_name
        self.lensed = lensed
        self.product_id = canonical_fingerprint(
            {
                "kind": "cmb-angular-power-table",
                "spectra": list(names),
                "convention": convention,
                "units": unit_name,
                "lensed": lensed,
                "provenance": provenance.provenance_id,
                "num_multipoles": int(ell_host.size),
            }
        )

    def converted(self, convention: CMBConvention, /) -> CMBAngularPowerTable:
        if convention == self.convention:
            return self
        factor = self.multipoles * (self.multipoles + 1.0) / (2.0 * jnp.pi)
        values = (
            self.spectra * factor[None, :]
            if convention == "Dl"
            else self.spectra / jnp.where(factor[None, :] != 0.0, factor[None, :], 1.0)
        )
        return CMBAngularPowerTable(
            self.multipoles,
            values,
            self.spectrum_names,
            self.scale,
            self.provenance,
            convention=convention,
            units=self.units,
            lensed=self.lensed,
        )


class CMBResponseResult(StrictModule):
    predicted_bandpowers: Array
    residual: Array
    whitened_residual: Array
    log_likelihood: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class CMBResponsePlan(StrictModule, NonTrainableState):
    windows: Array
    observed_bandpowers: Array
    covariance_cholesky: Array
    expected_spectrum_names: tuple[str, ...] = eqx.field(static=True)
    expected_convention: CMBConvention = eqx.field(static=True)
    expected_units: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        windows: ArrayLike,
        observed_bandpowers: ArrayLike,
        covariance_cholesky: ArrayLike,
        expected_spectrum_names: tuple[str, ...],
        /,
        *,
        expected_convention: CMBConvention,
        expected_units: str,
        response_id: str,
    ):
        windows_host = np.asarray(windows, dtype=float)
        observed_host = np.asarray(observed_bandpowers, dtype=float)
        cholesky_host = np.asarray(covariance_cholesky, dtype=float)
        names = tuple(str(name).strip() for name in expected_spectrum_names)
        if windows_host.ndim != 3 or windows_host.shape[1] != len(names):
            raise ValueError("CMB windows must have shape (bands, spectra, multipoles).")
        bands = windows_host.shape[0]
        if observed_host.shape != (bands,) or cholesky_host.shape != (bands, bands):
            raise ValueError("CMB observations and covariance must match band count.")
        if (
            np.any(~np.isfinite(windows_host))
            or np.any(~np.isfinite(observed_host))
            or np.any(~np.isfinite(cholesky_host))
        ):
            raise ValueError("CMB response arrays must be finite.")
        if np.any(np.diag(cholesky_host) <= 0.0):
            raise ValueError("CMB covariance Cholesky diagonal must be positive.")
        if expected_convention not in ("Cl", "Dl"):
            raise ValueError("Unknown expected CMB convention.")
        units = str(expected_units).strip()
        identifier = str(response_id).strip()
        if not units or not identifier:
            raise ValueError("CMB response units and ID must be non-empty.")
        self.windows = jnp.asarray(windows_host)
        self.observed_bandpowers = jnp.asarray(observed_host)
        self.covariance_cholesky = jnp.asarray(cholesky_host)
        self.expected_spectrum_names = names
        self.expected_convention = expected_convention
        self.expected_units = units
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cmb-response-plan",
                "response_id": identifier,
                "spectra": list(names),
                "convention": expected_convention,
                "units": units,
                "shape": list(windows_host.shape),
            }
        )

    def evaluate(self, table: CMBAngularPowerTable, /) -> CMBResponseResult:
        if not isinstance(table, CMBAngularPowerTable):
            raise TypeError("table must be a CMBAngularPowerTable.")
        if table.spectrum_names != self.expected_spectrum_names:
            raise ValueError("CMB spectrum names do not match the response.")
        if table.units != self.expected_units:
            raise ValueError("CMB spectrum units do not match the response.")
        converted = table.converted(self.expected_convention)
        if converted.spectra.shape != self.windows.shape[1:]:
            raise ValueError("CMB spectrum grid does not match response windows.")
        predicted = contract("bsl,sl->b", self.windows, converted.spectra)
        residual = self.observed_bandpowers - predicted
        whitened = jsp.linalg.solve_triangular(
            self.covariance_cholesky, residual, lower=True
        )
        finite = jnp.all(jnp.isfinite(predicted)) & jnp.all(jnp.isfinite(whitened))
        log_likelihood = -0.5 * jnp.sum(whitened * whitened)
        return CMBResponseResult(
            predicted,
            residual,
            whitened,
            jnp.where(finite, log_likelihood, -jnp.inf),
            finite,
            self.plan_id,
            table.product_id,
        )


__all__ = [
    "CMBAngularPowerTable",
    "CMBConvention",
    "CMBResponsePlan",
    "CMBResponseResult",
]
