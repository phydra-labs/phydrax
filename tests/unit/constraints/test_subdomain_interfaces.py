#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _family():
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 2)
    left, right = cover.patches
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            left.patch_id: left.domain.Function("x")(lambda x: x[0] ** 2),
            right.patch_id: right.domain.Function("x")(lambda x: x[0] ** 2 + 1.0),
        },
    )
    return cover, family


def _interface_batch(pairing):
    return pairing.component.sample(
        phx.domain.PointSampling(6),
        key=jr.key(0),
    )


def test_value_and_flux_jumps_use_one_paired_realization_and_orientation():
    cover, family = _family()
    pairing = cover.pairings[0]
    left_name = family.field_name(pairing.left_patch_id)
    right_name = family.field_name(pairing.right_patch_id)
    functions = family.solver_functions()
    batch = _interface_batch(pairing)

    value = phx.conditions.SubdomainValueJump(
        left_name,
        right_name,
        pairing,
        target=1.0,
    ).residual(functions)
    flux = phx.conditions.SubdomainFluxJump(
        left_name,
        right_name,
        pairing,
        lambda field: phx.operators.grad(field, var="x"),
        lambda field: phx.operators.grad(field, var="x"),
    ).residual(functions)

    np.testing.assert_allclose(value(batch).data, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(flux(batch).data, 0.0, atol=1.0e-12)
    assert pairing.normal is not None
    np.testing.assert_allclose(pairing.normal(batch).data, 1.0)


def test_general_transmission_receives_local_fields_and_returns_pair_support():
    cover, family = _family()
    pairing = cover.pairings[0]
    left_name = family.field_name(pairing.left_patch_id)
    right_name = family.field_name(pairing.right_patch_id)

    condition = phx.conditions.SubdomainTransmission(
        left_name,
        right_name,
        pairing,
        lambda left, right, support: (
            support.trace(right, side="right") - support.trace(left, side="left") - 1.0
        ),
    )
    residual = condition.residual(family.solver_functions())

    assert residual.domain.same_support(pairing.component.domain)
    np.testing.assert_allclose(residual(_interface_batch(pairing)).data, 0.0)


def test_overlap_consistency_requires_volume_pairing():
    cover, family = _family()
    pairing = cover.pairings[0]

    with pytest.raises(ValueError, match="overlap-volume"):
        phx.conditions.subdomain_overlap_consistency(
            family.ref(pairing.left_patch_id),
            family.ref(pairing.right_patch_id),
            pairing,
        )


def test_flux_jump_rejects_pairing_without_normal():
    cover, family = _family()
    original = cover.pairings[0]
    pairing = phx.domain.PairedSupport(
        original.component,
        dict(original.left_coordinates),
        dict(original.right_coordinates),
        pairing_id="normal-free",
        left_patch_id=original.left_patch_id,
        right_patch_id=original.right_patch_id,
        codimension=1,
    )

    with pytest.raises(ValueError, match="normal"):
        phx.conditions.SubdomainFluxJump(
            family.field_name(pairing.left_patch_id),
            family.field_name(pairing.right_patch_id),
            pairing,
            lambda field: field,
            lambda field: field,
        )
