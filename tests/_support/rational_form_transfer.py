from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    RationalComposedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import RationalPolynomial
from phydrax.discretization._nested_reference_types import ReferenceCellMapWitnesses
from phydrax.discretization.fem._de_rham import (
    FiniteElementDeRhamComplex,
    FiniteElementTransferResult,
)
from phydrax.discretization.fem._form_elements import form_element
from phydrax.geometry._common_root_volume_overlap import CurvedOverlapIntegrationPolicy


def make_rational_form_transfer_case(
    integration_policy: CurvedOverlapIntegrationPolicy | None = None,
    /,
) -> FiniteElementTransferResult:
    """Prepare an accepted interval transfer under an exact rational self-chart."""
    coordinates = np.asarray(((0.0,), (1.0,)), dtype=np.float64)
    route = np.asarray(((0, 1),), dtype=np.int32)
    source_mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "edge",
                "interval",
                np.asarray(((0, 1),), dtype=np.int32),
                global_ids=np.asarray((10,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((0, 1), dtype=np.int64),
    )
    target_mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "edge",
                "interval",
                np.asarray(((0, 1),), dtype=np.int32),
                global_ids=np.asarray((20,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((0, 1), dtype=np.int64),
    )
    coordinate_element = form_element(
        "interval", 0, 1, family="trimmed", proxy="scalar"
    )
    source_geometry = CellGeometrySpec(
        {"edge": coordinate_element}, {"edge": route}, coordinates
    )
    argument = RationalPolynomial(
        {(1,): Fraction(2)},
        {(0,): Fraction(1), (1,): Fraction(1)},
    )
    target_element = RationalComposedCellGeometryElement(
        coordinate_element, "interval", (argument,)
    )
    target_geometry = CellGeometrySpec(
        {"edge": target_element}, {"edge": route}, coordinates
    )
    witness = ReferenceCellMapWitnesses(
        np.asarray((20,), dtype=np.int64),
        np.asarray((10,), dtype=np.int64),
        "interval",
        "interval",
        np.asarray((((0.0,), (1.0,)),), dtype=np.float64),
        (((Fraction(0),), (Fraction(1),)),),
        ((argument,),),
    )
    transition = CellGeometryTransition(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", reference_map_count=1
        ),
        reference_map=witness,
    )
    policy = (
        CurvedOverlapIntegrationPolicy(
            relative_tolerance=1.0e-9, absolute_tolerance=1.0e-12
        )
        if integration_policy is None
        else integration_policy
    )
    source = FiniteElementDeRhamComplex(
        source_mesh,
        family="trimmed",
        order=1,
        coordinate_spec=source_geometry,
    )
    target = FiniteElementDeRhamComplex(
        target_mesh,
        family="trimmed",
        order=1,
        coordinate_spec=target_geometry,
    )
    return source.transfer(
        target,
        reference_maps=witness,
        geometry_transition=transition,
        integration_policy=policy,
    )
