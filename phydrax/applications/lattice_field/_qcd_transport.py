#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class QCDTransportSourceKind(StrEnum):
    LATTICE_INFERRED = "lattice-inferred"
    PHENOMENOLOGICAL = "phenomenological"
    EFFECTIVE_THEORY = "effective-theory"


class QCDTransportTable(StrictModule, NonTrainableState):
    temperatures: Array
    baryon_chemical_potentials: Array
    shear_viscosity_over_entropy: Array
    bulk_viscosity_over_entropy: Array
    baryon_diffusion: Array
    valid: Array
    source_kind: QCDTransportSourceKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures: ArrayLike,
        baryon_chemical_potentials: ArrayLike,
        shear_viscosity_over_entropy: ArrayLike,
        bulk_viscosity_over_entropy: ArrayLike,
        baryon_diffusion: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        source_kind: QCDTransportSourceKind,
        source_id: str,
    ):
        temperatures_ = np.asarray(temperatures, dtype=np.float64)
        chemical = np.asarray(baryon_chemical_potentials, dtype=np.float64)
        shear = np.asarray(shear_viscosity_over_entropy, dtype=np.float64)
        bulk = np.asarray(bulk_viscosity_over_entropy, dtype=np.float64)
        diffusion = np.asarray(baryon_diffusion, dtype=np.float64)
        if (
            temperatures_.ndim != 1
            or chemical.ndim != 1
            or temperatures_.size < 2
            or chemical.size < 2
            or np.any(np.diff(temperatures_) <= 0.0)
            or np.any(np.diff(chemical) <= 0.0)
        ):
            raise ValueError("QCD transport table axes must be increasing vectors.")
        expected = (temperatures_.size, chemical.size)
        if (
            shear.shape != expected
            or bulk.shape != expected
            or diffusion.shape != expected
        ):
            raise ValueError("QCD transport fields must align with table axes.")
        valid_ = (
            np.ones(expected, dtype=np.bool_)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
        )
        if (
            valid_.shape != expected
            or np.any(~np.isfinite(shear[valid_]))
            or np.any(~np.isfinite(bulk[valid_]))
            or np.any(~np.isfinite(diffusion[valid_]))
            or np.any(shear[valid_] < 0.0)
            or np.any(bulk[valid_] < 0.0)
            or np.any(diffusion[valid_] < 0.0)
        ):
            raise ValueError("Valid QCD transport fields must be finite and nonnegative.")
        if (
            not isinstance(source_kind, QCDTransportSourceKind)
            or not str(source_id).strip()
        ):
            raise ValueError("QCD transport source kind and identity are required.")
        self.temperatures = jnp.asarray(temperatures_)
        self.baryon_chemical_potentials = jnp.asarray(chemical)
        self.shear_viscosity_over_entropy = jnp.asarray(shear)
        self.bulk_viscosity_over_entropy = jnp.asarray(bulk)
        self.baryon_diffusion = jnp.asarray(diffusion)
        self.valid = jnp.asarray(valid_)
        self.source_kind = source_kind
        self.source_id = str(source_id).strip()
        self.table_id = canonical_fingerprint(
            {
                "kind": "qcd-transport-table",
                "axes": array_tree_fingerprint((temperatures_, chemical)),
                "fields": array_tree_fingerprint((shear, bulk, diffusion, valid_)),
                "source_kind": source_kind.value,
                "source": self.source_id,
            }
        )


class QCDTransportEvaluation(StrictModule, NonTrainableState):
    shear_viscosity_over_entropy: Array
    bulk_viscosity_over_entropy: Array
    baryon_diffusion: Array
    valid: Array
    table_id: str = eqx.field(static=True)


def evaluate_qcd_transport(
    table: QCDTransportTable,
    temperature: ArrayLike,
    baryon_chemical_potential: ArrayLike,
    /,
) -> QCDTransportEvaluation:
    if not isinstance(table, QCDTransportTable):
        raise TypeError("table must be QCDTransportTable.")
    temperature_ = jnp.asarray(temperature, dtype=table.temperatures.dtype).reshape(())
    chemical = jnp.asarray(baryon_chemical_potential, dtype=temperature_.dtype).reshape(
        ()
    )
    ti = jnp.clip(
        jnp.searchsorted(table.temperatures, temperature_, side="right") - 1,
        0,
        table.temperatures.size - 2,
    )
    mi = jnp.clip(
        jnp.searchsorted(table.baryon_chemical_potentials, chemical, side="right") - 1,
        0,
        table.baryon_chemical_potentials.size - 2,
    )
    ft = (temperature_ - table.temperatures[ti]) / (
        table.temperatures[ti + 1] - table.temperatures[ti]
    )
    fm = (chemical - table.baryon_chemical_potentials[mi]) / (
        table.baryon_chemical_potentials[mi + 1] - table.baryon_chemical_potentials[mi]
    )

    def interpolate(values):
        return (
            (1.0 - ft) * (1.0 - fm) * values[ti, mi]
            + ft * (1.0 - fm) * values[ti + 1, mi]
            + (1.0 - ft) * fm * values[ti, mi + 1]
            + ft * fm * values[ti + 1, mi + 1]
        )

    corners_valid = (
        table.valid[ti, mi]
        & table.valid[ti + 1, mi]
        & table.valid[ti, mi + 1]
        & table.valid[ti + 1, mi + 1]
    )
    in_domain = (
        (temperature_ >= table.temperatures[0])
        & (temperature_ <= table.temperatures[-1])
        & (chemical >= table.baryon_chemical_potentials[0])
        & (chemical <= table.baryon_chemical_potentials[-1])
    )
    valid = corners_valid & in_domain
    shear = interpolate(table.shear_viscosity_over_entropy)
    bulk = interpolate(table.bulk_viscosity_over_entropy)
    diffusion = interpolate(table.baryon_diffusion)
    return QCDTransportEvaluation(
        jnp.where(valid, shear, jnp.nan),
        jnp.where(valid, bulk, jnp.nan),
        jnp.where(valid, diffusion, jnp.nan),
        valid,
        table.table_id,
    )


__all__ = [
    "QCDTransportEvaluation",
    "QCDTransportSourceKind",
    "QCDTransportTable",
    "evaluate_qcd_transport",
]
