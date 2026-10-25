#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small explicit molecules for crystallographic geometry tests.

Geometries are idealized textbook bond lengths (angstrom) chosen so that the
declared stereo matches the coordinates: CHFClBr with one tetrahedral centre,
centrosymmetric trans-1,2-dichloroethene and a single argon atom.
"""

from __future__ import annotations

import numpy as np

from phydrax.atomistic.crystal import (
    ConformerPool,
    MolecularComponent,
    MolecularConformer,
)


_SQRT3 = np.sqrt(3.0)


def chlorofluorobromomethane(sign: int = 1) -> ConformerPool:
    """Return stereo-chiral CHFClBr with ``sign(det[H-C, F-C, Cl-C]) = sign``."""
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1.")
    component = MolecularComponent(
        (6, 1, 9, 17, 35),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        bond_orders=(1, 1, 1, 1),
        masses=(12.011, 1.008, 18.998, 35.45, 79.904),
        spin_multiplicity=1,
        tetrahedral_stereo=((0, 1, 2, 3, sign),),
        name="chlorofluorobromomethane",
    )
    positions = sign * np.asarray(
        [
            (0.0, 0.0, 0.0),
            np.asarray((1.0, 1.0, 1.0)) * 1.09 / _SQRT3,
            np.asarray((1.0, -1.0, -1.0)) * 1.35 / _SQRT3,
            np.asarray((-1.0, 1.0, -1.0)) * 1.77 / _SQRT3,
            np.asarray((-1.0, -1.0, 1.0)) * 1.94 / _SQRT3,
        ]
    )
    conformer = MolecularConformer(
        component, positions, provenance="idealized test geometry"
    )
    return ConformerPool(component, (conformer,))


def trans_dichloroethene() -> ConformerPool:
    """Return planar centrosymmetric trans-ClHC=CHCl (atoms C, C, Cl, Cl, H, H)."""
    component = MolecularComponent(
        (6, 6, 17, 17, 1, 1),
        ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5)),
        bond_orders=(2, 1, 1, 1, 1),
        masses=(12.011, 12.011, 35.45, 35.45, 1.008, 1.008),
        spin_multiplicity=1,
        double_bond_stereo=((2, 0, 1, 3, -1),),
        name="trans-1,2-dichloroethene",
    )
    carbon = np.asarray((0.665, 0.0, 0.0))
    chlorine = carbon + 1.73 * np.asarray((0.5, np.sqrt(0.75), 0.0))
    hydrogen = carbon + 1.08 * np.asarray((0.5, -np.sqrt(0.75), 0.0))
    positions = np.stack((carbon, -carbon, chlorine, -chlorine, hydrogen, -hydrogen))
    conformer = MolecularConformer(
        component, positions, provenance="idealized test geometry"
    )
    return ConformerPool(component, (conformer,))


def argon() -> ConformerPool:
    """Return a one-atom component."""
    component = MolecularComponent(
        (18,),
        (),
        bond_orders=(),
        masses=(39.948,),
        spin_multiplicity=1,
        valence="unchecked",
    )
    conformer = MolecularConformer(component, np.zeros((1, 3)), provenance="single atom")
    return ConformerPool(component, (conformer,))


def nontrivial_automorphism(pool: ConformerPool) -> tuple[int, ...]:
    """Return the unique non-identity automorphism of a two-fold symmetric graph."""
    identity = tuple(range(pool.component.atom_count))
    others = [p for p in pool.component.automorphisms if p != identity]
    if len(others) != 1:
        raise AssertionError(f"expected one non-identity automorphism, got {others}")
    return others[0]
