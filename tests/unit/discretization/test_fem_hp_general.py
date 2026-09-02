#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

from phydrax.discretization.fem._hp_general import (
    GeneralHPForest,
    NonconformingFacetOverlay,
    prism_axial_refinement_template,
    pyramid_transition_refinement_template,
    tensor_bisection_template,
    triangle_red_refinement_template,
)


def test_reference_refinement_templates_are_nonsingular_and_shape_complete():
    templates = (
        tensor_bisection_template("quadrilateral", 2, 0),
        triangle_red_refinement_template(),
        prism_axial_refinement_template(),
        pyramid_transition_refinement_template(),
    )
    assert tuple(len(value.child_cell_kinds) for value in templates) == (2, 4, 2, 4)
    for template in templates:
        determinants = np.linalg.det(np.asarray(template.affine_matrices))
        assert np.all(np.abs(determinants) > 0.0)


def test_general_hp_forest_and_nonconforming_overlay_have_stable_identity():
    forest = GeneralHPForest.roots(
        ("triangle", "quadrilateral", "prism", "pyramid"),
        np.asarray(((2, 2, 0), (3, 2, 0), (2, 2, 3), (2, 2, 2))),
    )
    assert forest.active.shape == (4,)
    assert len(set(forest.cell_kinds)) == 4
    overlay = NonconformingFacetOverlay(
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((2, 3), dtype=np.int32),
        np.asarray((((0.5, 0.0), (0.0, 1.0)),) * 2),
        np.asarray((((1.0, 0.0), (0.0, 1.0)),) * 2),
        np.asarray((1, 2), dtype=np.int32),
        np.asarray((2, 1), dtype=np.int32),
        ("mortar-a", "mortar-b"),
    )
    assert overlay.owner_subface_maps.shape == (2, 2, 2)
    assert overlay.overlay_id
