#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.geometry import (
    CompartmentComplex,
    CompartmentDefinition,
    CompartmentInterfaceDefinition,
)


def test_compartment_complex_normalization_preserves_identity() -> None:
    compartments = (
        CompartmentDefinition("left", ("left-label",), "material"),
        CompartmentDefinition("right", ("right-label",), "material"),
    )
    interfaces = (CompartmentInterfaceDefinition("contact", "left", "right", "exchange"),)
    first = CompartmentComplex(
        "revision",
        compartments,
        interfaces,
        (("right", 3.0), ("left", 2.0)),
        (("right", "left"),),
    )
    second = CompartmentComplex(
        "revision",
        compartments,
        interfaces,
        (("left", 2.0), ("right", 3.0)),
        (("left", "right"),),
    )
    assert first.compartment_measures == (("left", 2.0), ("right", 3.0))
    assert first.observed_adjacencies == (("left", "right"),)
    assert first.adjacency.successful
    assert first.adjacency.report_id == second.adjacency.report_id
    assert first.complex_id == second.complex_id


def test_compartment_declarations_fail_before_measure_preparation() -> None:
    compartments = (
        CompartmentDefinition(
            "inside", ("inside-label",), "material", containment_parent_id="undeclared"
        ),
    )
    with pytest.raises(ValueError, match="CompartmentComplex.containment_parent_id"):
        CompartmentComplex("revision", compartments, (), (("inside", -1.0),), ())


def test_compartment_adjacency_retains_missing_and_forbidden_pairs() -> None:
    compartments = (
        CompartmentDefinition("left", ("left-label",), "material"),
        CompartmentDefinition("right", ("right-label",), "material"),
        CompartmentDefinition("other", ("other-label",), "material"),
    )
    interfaces = (
        CompartmentInterfaceDefinition("required-contact", "left", "right", "exchange"),
    )
    complex_ = CompartmentComplex(
        "revision",
        compartments,
        interfaces,
        (("left", 1.0), ("right", 1.0), ("other", 1.0)),
        (("left", "other"),),
    )
    assert complex_.adjacency.missing_pairs == (("left", "right"),)
    assert complex_.adjacency.forbidden_pairs == (("left", "other"),)
    assert not complex_.adjacency.successful
    with pytest.raises(ValueError, match="CompartmentComplex.adjacency"):
        complex_.require_valid_adjacency()
