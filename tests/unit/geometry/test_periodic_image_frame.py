from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    PeriodicIsometryGroup,
    PeriodicIsometryIdentityError,
)
from phydrax.geometry._periodic_image_frame import prepare_periodic_image_frame
from phydrax.geometry._periodic_power_preparation import PeriodicPowerPreparation


def _screw_translation_group() -> PeriodicIsometryGroup:
    screw = np.eye(4, dtype=np.float64)
    screw[0, 0] = -1.0
    screw[1, 1] = -1.0
    screw[2, 3] = 1.0
    return PeriodicIsometryGroup(screw[None, :, :])


def _exact_point_action(
    matrix: tuple[tuple[Fraction, ...], ...], point: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (matrix[row][axis] * point[axis] for axis in range(len(point))),
            matrix[row][len(point)],
        )
        for row in range(len(point))
    )


def _power_coordinates(
    preparation: PeriodicPowerPreparation,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    offsets = np.asarray(preparation.image_coordinate_offsets, dtype=np.int64)
    components = np.asarray(preparation.image_coordinate_components, dtype=np.float64)
    values = tuple(
        sum(
            (
                Fraction(float(component))
                for component in components[offsets[index] : offsets[index + 1]]
            ),
            Fraction(0),
        )
        for index in range(offsets.size - 1)
    )
    return tuple(
        (values[row], values[row + 1], values[row + 2])
        for row in range(0, len(values), 3)
    )


def test_skew_three_dimensional_lattice_frame_preserves_lifted_sources() -> None:
    lattice = np.asarray(
        ((2.0, 0.0, 0.0), (0.5, 2.0, 0.0), (0.25, 0.5, 2.0)),
        dtype=np.float64,
    )
    base = np.asarray((0.125, 0.25, 0.5), dtype=np.float64)
    source_shifts = np.asarray(((0, 0, 0), (1, 0, 0), (0, -1, 1)), dtype=np.int64)
    coordinates = np.asarray(
        (base, base + lattice[0], base - lattice[1] + lattice[2]),
        dtype=np.float64,
    )
    roots = np.zeros(3, dtype=np.int64)
    frame = prepare_periodic_image_frame(
        coordinates, roots, source_shifts, PeriodicCell(lattice)
    )

    reconstructed = frame.image_points((0, 0, 0))
    exact_scale = Fraction(2) ** frame.exponent
    exact_coordinates = tuple(
        tuple(Fraction(int(value)) * exact_scale for value in row)
        for row in reconstructed
    )
    expected = tuple(
        tuple(Fraction(float(value)) for value in row) for row in coordinates
    )

    assert frame.image_count > 0
    np.testing.assert_array_equal(frame.source_roots, roots)
    np.testing.assert_array_equal(frame.source_exponents, source_shifts)
    assert exact_coordinates == expected


def test_isometry_image_frame_retains_lift_actions_and_complete_support_images() -> None:
    group = _screw_translation_group()
    roots = np.asarray((0, 0, 2), dtype=np.int64)
    source_exponents = np.asarray(((0,), (1,), (0,)), dtype=np.int64)
    coordinates = np.asarray(
        ((0.25, 0.25, 0.25), (-0.25, -0.25, 1.25), (0.75, 0.5, 0.75)),
        dtype=np.float64,
    )
    support = ((np.zeros(3, dtype=np.float64), np.asarray((1.0, 1.0, 5.0))),)
    frame = prepare_periodic_image_frame(
        coordinates,
        roots,
        source_exponents,
        group,
        source_support_boxes=support,
    )

    assert frame.lattice is None
    assert frame.image_exponents is not None
    assert frame.image_count == len(frame.image_exponents) > 2
    np.testing.assert_array_equal(frame.source_roots, roots)
    np.testing.assert_array_equal(frame.source_exponents, source_exponents)
    exact_scale = Fraction(2) ** (3 * frame.exponent)
    for image_index, image_exponent in enumerate(frame.image_exponents):
        image_matrix = exact_periodic_element(group, image_exponent)
        image_coordinates = frame.image_points(image_index)
        for row, (root, lift_exponent) in enumerate(
            zip(roots, source_exponents, strict=True)
        ):
            lift_matrix = exact_periodic_element(group, lift_exponent)
            root_point = tuple(Fraction(float(value)) for value in coordinates[int(root)])
            lifted = _exact_point_action(lift_matrix, root_point)
            expected = _exact_point_action(image_matrix, lifted)
            actual = tuple(
                Fraction(int(value)) * exact_scale for value in image_coordinates[row]
            )
            assert actual == expected


def test_glide_reflection_group_is_refused_as_orientation_reversing() -> None:
    glide = np.eye(4, dtype=np.float64)
    glide[0, 0] = -1.0
    glide[2, 3] = 1.0

    with pytest.raises(
        PeriodicIsometryIdentityError,
        match="Periodic generators must preserve orientation exactly.",
    ):
        PeriodicIsometryGroup(glide[None, :, :])


def test_periodic_power_preparation_keeps_authored_sites_and_group_actions() -> None:
    lattice = np.asarray(
        ((2.0, 0.0, 0.0), (0.5, 2.0, 0.0), (0.25, 0.5, 2.0)),
        dtype=np.float64,
    )
    sites = np.asarray(((0.25, 0.25, 0.25), (0.75, 0.5, 0.75)))
    weights = np.asarray((0.0, 0.125), dtype=np.float64)
    carrier = np.asarray(tuple(product((0.0, 1.0), repeat=3)), dtype=np.float64)
    group = PeriodicCell(lattice)
    preparation = PeriodicPowerPreparation(sites, weights, carrier, group)
    exact_coordinates = _power_coordinates(preparation)

    assert preparation.image_count == len(preparation.image_sites)
    assert preparation.image_count == len(preparation.image_exponents)
    assert set(np.asarray(preparation.image_sites, dtype=np.int64)) == {0, 1}
    lattice_exact = tuple(
        tuple(Fraction(float(value)) for value in row) for row in lattice
    )
    for image, (site, exponent) in enumerate(
        zip(preparation.image_sites, preparation.image_exponents, strict=True)
    ):
        source = tuple(Fraction(float(value)) for value in sites[int(site)])
        expected = tuple(
            source[axis]
            + sum(
                (
                    int(exponent[generator]) * lattice_exact[generator][axis]
                    for generator in range(3)
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        assert exact_coordinates[image] == expected

    isometry_group = _screw_translation_group()
    isometry_preparation = PeriodicPowerPreparation(
        sites[:1],
        weights[:1],
        carrier,
        isometry_group,
    )
    isometry_coordinates = _power_coordinates(isometry_preparation)
    for image, (site, exponent) in enumerate(
        zip(
            isometry_preparation.image_sites,
            isometry_preparation.image_exponents,
            strict=True,
        )
    ):
        matrix = exact_periodic_element(isometry_group, exponent)
        source = tuple(Fraction(float(value)) for value in sites[int(site)])
        assert isometry_coordinates[image] == _exact_point_action(matrix, source)
