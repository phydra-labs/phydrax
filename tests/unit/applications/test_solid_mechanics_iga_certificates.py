#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.applications.solid_mechanics._iga_shells import _surface
from phydrax.applications.solid_mechanics._iga_solids import _local_certificate
from phydrax.discretization.iga._certificate import (
    CertificateDisposition,
    LocalGeometryCertificate,
    SurfaceEmbeddingCertificate,
)


def _local(disposition: CertificateDisposition) -> LocalGeometryCertificate:
    return LocalGeometryCertificate(
        disposition,
        "block",
        2,
        (),
        (),
        None,
        "policy",
        f"local-{disposition.value}",
    )


def _embedding(disposition: CertificateDisposition) -> SurfaceEmbeddingCertificate:
    return SurfaceEmbeddingCertificate(
        disposition,
        "surface",
        (),
        (),
        None,
        "policy",
        f"surface-{disposition.value}",
    )


def test_iga_contracts() -> None:
    accepted = _local(CertificateDisposition.PASS)
    assert _local_certificate(accepted) is accepted
    for disposition in (
        CertificateDisposition.COUNTEREXAMPLE,
        CertificateDisposition.INCONCLUSIVE,
    ):
        with pytest.raises(ValueError, match="did not pass"):
            _local_certificate(_local(disposition))
    accepted = _embedding(CertificateDisposition.PASS)
    assert _surface(accepted) is accepted
    for disposition in (
        CertificateDisposition.COUNTEREXAMPLE,
        CertificateDisposition.INCONCLUSIVE,
    ):
        with pytest.raises(ValueError, match="did not pass"):
            _surface(_embedding(disposition))
