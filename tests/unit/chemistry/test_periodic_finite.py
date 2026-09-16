import numpy as np

from phydrax.chemistry.periodic._finite import (
    finite_layer_populations,
    PeriodicFiniteBoundaryPlan,
    PeriodicFiniteOrbitalPlan,
    PrescribedPeriodicDisorder,
)
from phydrax.chemistry.periodic._orbital_model import (
    PeriodicBlochGauge,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
)
from phydrax.discretization import PeriodicCell
from phydrax.operators.periodic import periodic_translation_family_from_dense_blocks
from phydrax.units import ANGSTROM, ELECTRONVOLT


def _chain():
    cell = PeriodicCell([[1.0]])
    basis = PeriodicOrbitalBasisPlan(
        cell, ("s",), [[0.0]], ANGSTROM, PeriodicBlochGauge("lattice")
    )
    h = periodic_translation_family_from_dense_blocks(
        [[-1], [0], [1]],
        np.asarray([[[[[-1.0]]]], [[[[0.0]]]], [[[[-1.0]]]]]),
    )
    return basis, PeriodicOrbitalPencilPlan.orthonormal(
        basis, h.plan, h.state, ELECTRONVOLT
    ).prepare()


def test_open_periodic_and_twisted_realizations_have_exact_boundary_entries():
    _, pencil = _chain()
    open_result = PeriodicFiniteOrbitalPlan(
        pencil, PeriodicFiniteBoundaryPlan.open((3,))
    ).realize()
    periodic_result = PeriodicFiniteOrbitalPlan(
        pencil, PeriodicFiniteBoundaryPlan.periodic((3,))
    ).realize()
    twisted_result = PeriodicFiniteOrbitalPlan(
        pencil, PeriodicFiniteBoundaryPlan.twisted((3,), (np.pi / 2.0,))
    ).realize()

    np.testing.assert_allclose(
        open_result.hamiltonian.to_dense(),
        [[0.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    )
    np.testing.assert_allclose(
        periodic_result.hamiltonian.to_dense(),
        [[0.0, -1.0, -1.0], [-1.0, 0.0, -1.0], [-1.0, -1.0, 0.0]],
    )
    twisted = np.asarray(twisted_result.hamiltonian.to_dense())
    np.testing.assert_allclose(twisted, twisted.conj().T, atol=1.0e-14)
    np.testing.assert_allclose(abs(twisted[0, 2]), 1.0, atol=1.0e-14)


def test_prescribed_disorder_is_applied_in_documented_cell_orbital_order():
    basis, pencil = _chain()
    disorder = PrescribedPeriodicDisorder(
        [[0.1], [0.2], [0.3]], basis.basis_id, "fixed-disorder-case"
    )
    realization = PeriodicFiniteOrbitalPlan(
        pencil,
        PeriodicFiniteBoundaryPlan.open((3,)),
        disorder=disorder,
    ).realize()

    np.testing.assert_allclose(
        np.diag(np.asarray(realization.hamiltonian.to_dense())), [0.1, 0.2, 0.3]
    )
    assert realization.order == "cell-c-order_then-orbital-order"


def test_layer_populations_partition_generalized_metric_population():
    _, pencil = _chain()
    realization = PeriodicFiniteOrbitalPlan(
        pencil, PeriodicFiniteBoundaryPlan.open((3,))
    ).realize()
    result = finite_layer_populations(realization, np.eye(3), [1.0, 0.5, 0.0], 0)

    np.testing.assert_allclose(result.layer_populations, [1.0, 0.5, 0.0])
    np.testing.assert_allclose(result.total_population, 1.5)
    assert bool(result.successful)


def test_slab_route_opens_only_the_selected_axis():
    boundary = PeriodicFiniteBoundaryPlan.slab((2, 3, 4), 2)
    assert boundary.periodic_axes == (True, True, False)
    assert boundary.kind == "slab"
