from __future__ import annotations

import hashlib
import math

import numpy as np

import phydrax as phx


def _eqdsk_bytes():
    nw, nh = 3, 3
    header = "synthetic equilibrium".ljust(48) + f" 0 {nw} {nh}\n"
    scalars = [
        1.0,
        2.0,
        1.5,
        1.0,
        0.0,
        1.5,
        0.0,
        0.0,
        1.0,
        5.0,
        1.0e6,
        0.0,
        0.0,
        1.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    fpol = [7.5, 7.4, 7.3]
    pressure = [1.0e5, 5.0e4, 0.0]
    ffprime = [-0.2, -0.1, 0.0]
    pprime = [-1.0e5, -5.0e4, 0.0]
    psi = [1.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 1.0]
    q = [1.0, 1.5, 2.0]
    boundary = [1.0, -1.0, 2.0, -1.0, 2.0, 1.0, 1.0, 1.0]
    limiter = [1.0, -1.0, 2.0, -1.0, 2.0, 1.0, 1.0, 1.0]
    body = scalars + fpol + pressure + ffprime + pprime + psi + q
    body += [4.0, 4.0] + boundary + limiter
    return (header + " ".join(f"{value:.12E}" for value in body) + "\n").encode()


def _reference(payload):
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-eqdsk",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )


def _resource(payload):
    return phx.interchange.bounded_resource_from_bytes(
        payload,
        limits=phx.interchange.ResourceLimits(
            max_bytes=100_000,
            max_depth=4,
            max_nodes=10_000,
            max_attributes=10_000,
            max_losses=10,
        ),
    )


def test_tokamak_flux_normalization_round_trip_is_complete():
    canonical = phx.applications.tokamak.TokamakMagneticConvention.canonical()
    total_flux = phx.applications.tokamak.TokamakMagneticConvention(
        "total-weber",
        phx.applications.tokamak.PoloidalFluxNormalization.WEBER,
        poloidal_field_sign=-1,
        coordinate_handedness=1,
        radial_toroidal_flux_sign=1,
        plasma_current_sign=1,
        toroidal_field_sign=1,
        safety_factor_sign=1,
    )
    forward = total_flux.transform_to(canonical)
    reverse = canonical.transform_to(total_flux)
    assert math.isclose(forward.poloidal_flux_factor, 1.0 / (2.0 * math.pi))
    assert math.isclose(forward.poloidal_flux_factor * reverse.poloidal_flux_factor, 1.0)


def test_eqdsk_import_preserves_fields_and_canonicalizes_flux():
    payload = _eqdsk_bytes()
    canonical = phx.applications.tokamak.TokamakMagneticConvention.canonical()
    result = phx.applications.tokamak.interchange.import_eqdsk(
        _resource(payload),
        _reference(payload),
        phx.applications.tokamak.AxisymmetricMachineFrame("synthetic-machine"),
        canonical,
    )

    equilibrium = result.equilibrium
    assert result.report.valid
    assert equilibrium.poloidal_flux_wb_per_rad.shape == (3, 3)
    assert equilibrium.reference_major_radius_m == 1.5
    assert equilibrium.reference_toroidal_field_t == 5.0
    assert equilibrium.plasma_current_a == 1.0e6
    np.testing.assert_allclose(equilibrium.normalized_flux[1, 1], 0.0)
    np.testing.assert_allclose(equilibrium.normalized_flux[0, 0], 1.0)
    assert equilibrium.prepare().equilibrium_id == equilibrium.equilibrium_id


def test_eqdsk_import_rejects_unsupported_trailing_fields():
    payload = _eqdsk_bytes() + b"1.0\n"
    with np.testing.assert_raises_regex(ValueError, "trailing numeric"):
        phx.applications.tokamak.interchange.import_eqdsk(
            _resource(payload),
            _reference(payload),
            phx.applications.tokamak.AxisymmetricMachineFrame("synthetic-machine"),
            phx.applications.tokamak.TokamakMagneticConvention.canonical(),
        )
