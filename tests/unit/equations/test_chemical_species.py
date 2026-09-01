#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _schema():
    return phx.equations.ChemicalSpeciesSchema(
        ("cation", "anion", "solid"),
        (
            phx.equations.ChemicalPhaseKind.LIQUID,
            phx.equations.ChemicalPhaseKind.LIQUID,
            phx.equations.ChemicalPhaseKind.SOLID,
        ),
        jnp.asarray((0.023, 0.035, 0.058)),
        ("M", "X"),
        jnp.asarray(((1, 0, 1), (0, 1, 1)), dtype=jnp.int32),
        jnp.asarray((1, -1, 0), dtype=jnp.int32),
    )


def test_chemical_species_schema_tracks_elements_charge_and_phases():
    schema = _schema()
    amount = jnp.asarray((2.0, 2.0, 1.0))

    np.testing.assert_allclose(schema.element_amount(amount), (3.0, 3.0))
    np.testing.assert_allclose(schema.charge_amount(amount), 0.0)
    np.testing.assert_array_equal(
        schema.phase_mask(phx.equations.ChemicalPhaseKind.LIQUID),
        (True, True, False),
    )
    assert schema.phase_species_indices(phx.equations.ChemicalPhaseKind.SOLID) == (2,)
    assert schema.phase_count == 2


def test_chemical_species_schema_rejects_invalid_charge_and_surface_phase():
    with pytest.raises(ValueError, match="integer"):
        phx.equations.ChemicalSpeciesSchema(
            ("A",),
            (phx.equations.ChemicalPhaseKind.GAS,),
            jnp.asarray((1.0,)),
            ("X",),
            jnp.asarray(((1,),), dtype=jnp.int32),
            jnp.asarray((0.5,)),
        )
    with pytest.raises(ValueError, match="explicit ChemicalPhaseSpec"):
        phx.equations.ChemicalSpeciesSchema(
            ("A*",),
            (phx.equations.ChemicalPhaseKind.SURFACE,),
            jnp.asarray((1.0,)),
            ("X",),
            jnp.asarray(((1,),), dtype=jnp.int32),
            jnp.asarray((0,), dtype=jnp.int32),
        )


def test_surface_phase_requires_positive_site_density():
    with pytest.raises(ValueError, match="site_density"):
        phx.equations.ChemicalPhaseSpec(
            "electrode",
            phx.equations.ChemicalPhaseKind.SURFACE,
            2,
        )
    phase = phx.equations.ChemicalPhaseSpec(
        "electrode",
        phx.equations.ChemicalPhaseKind.SURFACE,
        2,
        site_density=2.5,
    )
    schema = phx.equations.ChemicalSpeciesSchema(
        ("A*",),
        (phx.equations.ChemicalPhaseKind.SURFACE,),
        jnp.asarray((1.0,)),
        ("X",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        phase_specs=(phase,),
    )
    assert schema.phase_specs == (phase,)
