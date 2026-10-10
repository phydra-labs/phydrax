from __future__ import annotations

import hashlib
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


_PACKAGED = (
    Path(phx.nuclear.__file__).resolve().parent / "data" / "fission_gas_yields.json"
)

# JEFF-3.3 U-235 thermal (0.0253 eV) cumulative yields (MF8/MT459, R. Mills
# evaluation, IAEA NDS distribution nfpy_9228_92-U-235.zip, sha256
# 98ce6116af6ad4b49b627c9355fd2f2a30f7e76a527ae019277721297d0f3de7) summed over
# Kr-78..86 and Xe-124..136 stable/long-lived ground states (Xe-129 excluded):
# 0.25424328 atoms/fission; 1-sigma 0.00179671 from the listed uncertainties
# combined in quadrature. The admitted band is +/- 3 sigma.
_JEFF33_U235_THERMAL_GAS_YIELD = 0.25424328
_JEFF33_U235_THERMAL_GAS_SIGMA = 0.00179671


def _mixed_energy(table: phx.nuclear.FissionGasYieldTable, fractions: Array) -> Array:
    return table.mixed_energy_per_fission(fractions)


@pytest.fixture(scope="module")
def table() -> phx.nuclear.FissionGasYieldTable:
    return phx.nuclear.FissionGasYieldTable.endf_b_viii_0()


def test_packaged_table_carries_verified_endf_provenance(
    table: phx.nuclear.FissionGasYieldTable,
) -> None:
    provenance = table.provenance
    packaged = _PACKAGED.read_bytes()

    assert provenance.reference.checksum == hashlib.sha256(packaged).hexdigest()
    assert provenance.reference.size_bytes == len(packaged)
    assert provenance.library_name == "ENDF/B-VIII.0"
    assert provenance.source_uri == (
        "https://www.nndc.bnl.gov/endf-b8.0/zips/ENDF-B-VIII.0_nfy.zip"
    )
    assert provenance.is_processed
    # NNDC NFY sublibrary archive (its published MD5/SHA1 match this download).
    assert (
        "92c5371fdb21eecf4989f48828671b904186abc6386b3d7510c8fcee2ee5ffcf"
        in provenance.parent_data_ids
    )
    assert table.nuclides == phx.nuclear.FISSIONING_NUCLIDES


def test_u235_thermal_gas_yield_agrees_with_jeff33(
    table: phx.nuclear.FissionGasYieldTable,
) -> None:
    index = phx.nuclear.FISSIONING_NUCLIDES.index("U235")
    total = float(table.xenon_yields[index] + table.krypton_yields[index])

    assert abs(total - _JEFF33_U235_THERMAL_GAS_YIELD) <= (
        3.0 * _JEFF33_U235_THERMAL_GAS_SIGMA
    )


def test_u235_energy_per_fission_is_endf_recoverable_energy(
    table: phx.nuclear.FissionGasYieldTable,
) -> None:
    # ENDF/B-VIII.0 U-235 (MAT 9228) MF1/MT458 thermal point: ER = 1.940770E+8 eV
    # (ET = 2.028270E+8 eV less ENU = 8.750000E+6 eV).
    index = phx.nuclear.FISSIONING_NUCLIDES.index("U235")

    np.testing.assert_allclose(
        float(table.energy_per_fission[index]), 1.940770e8 * 1.602176634e-19, rtol=1e-15
    )


@pytest.mark.strict_jax
def test_mixing_is_linear_in_fission_fractions(
    table: phx.nuclear.FissionGasYieldTable,
) -> None:
    pure = jnp.eye(len(phx.nuclear.FISSIONING_NUCLIDES), dtype=jnp.float64)
    xenon, krypton = table.mixed_yields(pure)
    np.testing.assert_allclose(xenon, table.xenon_yields, rtol=1e-15)
    np.testing.assert_allclose(krypton, table.krypton_yields, rtol=1e-15)

    first = jnp.asarray([0.7, 0.2, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)
    second = jnp.asarray([0.3, 0.1, 0.4, 0.1, 0.05, 0.05], dtype=jnp.float64)
    weight = 0.25
    blend = weight * first + (1.0 - weight) * second
    mixed = eqx.filter_jit(_mixed_energy)(table, jnp.stack((first, second, blend)))
    np.testing.assert_allclose(
        mixed[2], weight * mixed[0] + (1.0 - weight) * mixed[1], rtol=1e-14
    )
    blend_xenon, _ = table.mixed_yields(blend)
    np.testing.assert_allclose(
        blend_xenon,
        weight * table.mixed_yields(first)[0]
        + (1.0 - weight) * table.mixed_yields(second)[0],
        rtol=1e-14,
    )


@pytest.mark.parametrize(
    "fractions",
    [
        pytest.param([0.7, 0.2, 0.0, 0.0, 0.0, 0.0], id="sum-below-one"),
        pytest.param([0.7, 0.4, 0.0, 0.0, 0.0, 0.0], id="sum-above-one"),
        pytest.param([1.2, -0.2, 0.0, 0.0, 0.0, 0.0], id="negative-fraction"),
    ],
)
def test_mixing_refuses_fractions_that_are_not_a_partition(
    table: phx.nuclear.FissionGasYieldTable, fractions: list[float]
) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="sum to one"):
        table.mixed_yields(jnp.asarray(fractions))
    with pytest.raises(eqx.EquinoxRuntimeError, match="sum to one"):
        table.mixed_energy_per_fission(jnp.asarray(fractions))


def test_mixing_refuses_a_foreign_nuclide_axis(
    table: phx.nuclear.FissionGasYieldTable,
) -> None:
    with pytest.raises(ValueError, match="trailing axis"):
        table.mixed_yields(jnp.asarray([0.5, 0.2, 0.2, 0.1]))
