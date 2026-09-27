from typing import Any

import numpy as np
import pytest

from phydrax.chemistry.interchange import (
    lower_wannier90_mmn,
    read_wannier90_hr,
    read_wannier90_mmn,
)
from phydrax.chemistry.periodic._source import (
    PeriodicProvenanceManifest,
    PeriodicSourceContext,
)
from phydrax.discretization import (
    PeriodicCell,
    ReciprocalConnectivityPlan,
    ReciprocalMeshPlan,
)
from phydrax.operators.periodic import (
    periodic_translation_family_from_dense_blocks,
    PeriodicBandManifold,
    PeriodicBlochGauge,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
    PeriodicResourceError,
    PeriodicSpectrumPlan,
)
from phydrax.units import ANGSTROM, ELECTRONVOLT


def _context(payload: Any, cell: Any, labels: Any, centers: Any, source: Any) -> Any:
    basis = PeriodicOrbitalBasisPlan(
        cell,
        labels,
        centers,
        ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    provenance = PeriodicProvenanceManifest.for_bytes(
        payload, source, "CC0-1.0-test-fixture"
    )
    return PeriodicSourceContext(basis, ELECTRONVOLT, provenance)


def test_wannier90_scenario_1() -> None:
    payload = (
        "independent one-orbital chain\n1\n3\n2 1 2\n-1 0 0 1 1 -2.0 0.0\n0 0 0 1 1 -1.0 0.0\n1 0 0 1 1 -2.0 0.0\n"
    ).encode()
    # ty: ignore[invalid-argument-type]
    context = _context(payload, PeriodicCell([[1.0]]), ("s",), [[0.0]], "hr-chain")
    imported = read_wannier90_hr(payload, context)

    # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(imported.prepared_family.evaluate([[0.0]])[0, 0, 0], -3.0)
    np.testing.assert_array_equal(imported.degeneracies, [2, 1, 2])
    np.testing.assert_allclose(
        imported.raw_hamiltonian_blocks[:, 0, 0], [-2.0, -1.0, -2.0]
    )
    payload = (
        "bad reverse\n1\n2\n1 1\n-1 0 0 1 1 -1.0 0.0\n1 0 0 1 1 -2.0 0.0\n"
    ).encode()
    # ty: ignore[invalid-argument-type]
    context = _context(payload, PeriodicCell([[1.0]]), ("s",), [[0.0]], "bad-hr")
    with pytest.raises(ValueError, match="Hermitian adjoints"):
        read_wannier90_hr(payload, context)
    with pytest.raises(TypeError, match="PeriodicSourceContext"):
        # ty: ignore[invalid-argument-type]
        read_wannier90_hr(payload, None)
    with pytest.raises(PeriodicResourceError, match="maximum_bytes"):
        read_wannier90_hr(payload, context, maximum_bytes=4)
    altered = payload + b" "
    with pytest.raises(ValueError, match="digest"):
        read_wannier90_hr(altered, context)
    payload = (
        "independent one-band links\n"
        "1 2 2\n"
        "1 2 0 0 0\n1.0 0.0\n"
        "1 2 -1 0 0\n1.0 0.0\n"
        "2 1 1 0 0\n1.0 0.0\n"
        "2 1 0 0 0\n1.0 0.0\n"
    ).encode()
    # ty: ignore[invalid-argument-type]
    cell = PeriodicCell([[1.0]])
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (2,))
    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    context = _context(payload, cell, ("band-1",), [[0.0]], "mmn-chain")
    imported = read_wannier90_mmn(payload, context, connectivity)

    np.testing.assert_allclose(imported.raw_overlaps, 1.0)
    np.testing.assert_allclose(imported.singular_values, 1.0)
    assert imported.connectivity.prepared_id == connectivity.prepared_id
    hamiltonian = periodic_translation_family_from_dense_blocks(
        # ty: ignore[invalid-argument-type]
        [[0]],
        np.asarray([[[[[-1.0]]]]]),
    )
    pencil = PeriodicOrbitalPencilPlan.orthonormal(
        context.basis,
        hamiltonian.plan,
        hamiltonian.state,
        ELECTRONVOLT,
    ).prepare()
    manifold = PeriodicBandManifold(PeriodicSpectrumPlan(pencil, mesh).evaluate(), [0])
    bundle = lower_wannier90_mmn(imported, manifold)
    np.testing.assert_allclose(bundle.raw_overlaps, imported.raw_overlaps)
    assert bundle.source_id.startswith("wannier90-mmn:")
    payload = (
        "missing link\n1 2 2\n1 2 0 0 0\n1.0 0.0\n1 2 -1 0 0\n1.0 0.0\n2 1 1 0 0\n1.0 0.0\n"
    ).encode()
    # ty: ignore[invalid-argument-type]
    cell = PeriodicCell([[1.0]])
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (2,))
    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    context = _context(payload, cell, ("band-1",), [[0.0]], "truncated-mmn")
    with pytest.raises(ValueError, match="truncated"):
        read_wannier90_mmn(payload, context, connectivity)

    other_payload = payload + b"\n"
    # ty: ignore[invalid-argument-type]
    other = _context(other_payload, PeriodicCell([[2.0]]), ("band-1",), [[0.0]], "other")
    with pytest.raises(ValueError, match="different periodic cells"):
        read_wannier90_mmn(other_payload, other, connectivity)
