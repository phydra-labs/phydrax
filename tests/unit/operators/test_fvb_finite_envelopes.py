import jax.numpy as jnp
import numpy as np

from phydrax.discretization import CellMesh
from phydrax.discretization.fem import (
    prepare_maxwell_mortar_interface_trace_3d,
    prepare_scalar_mortar_interface_trace_3d,
)
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
from phydrax.discretization.vem import (
    adapt_virtual_element_p,
    CurvedVirtualElementEdge,
    VirtualElementAdaptivityPolicy,
    VirtualElementEpoch,
)
from phydrax.linalg import ArraySpace
from phydrax.operators.integral.layer_potential import (
    prepare_displacement_discontinuity_3d,
)
from phydrax.solver._nonmatching_fem_bem3d import prepare_scalar_nonmatching_fem_bem_3d
from phydrax.solver._scalar_screen_junction3d import (
    prepare_scalar_screen_junction_solve_3d,
    ScalarScreenJunctionCondition3D,
)


def test_fvb_finite_envelopes_scenario_1() -> None:
    mass = np.asarray([[2.0, 0.2], [0.1, 1.5]])
    scalar = prepare_scalar_mortar_interface_trace_3d(
        mass, mass, coverage_fraction=1.0, orientation_margin=1.0, geometric_residual=0.0
    )
    complex = FiniteElementDeRhamComplex(
        CellMesh.from_tetrahedra(
            np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        ),
        family="trimmed",
        order=1,
    )
    maxwell_mass = np.concatenate((mass, np.zeros((2, 4), dtype=np.float64)), axis=1)
    maxwell = prepare_maxwell_mortar_interface_trace_3d(
        maxwell_mass,
        maxwell_mass,
        coverage_fraction=1.0,
        orientation_margin=1.0,
        geometric_residual=0.0,
        commuting_defect=0.0,
        volume_complex=complex,
        boundary_space=ArraySpace(
            (2,),
            dtype=complex.hilbert_complex().space(1).structure().dtype,
            space_id="mortar-contract-boundary",
        ),
    )
    x = jnp.asarray([0.3, -0.2])
    y = jnp.asarray([0.4, 0.7])
    assert jnp.allclose(jnp.vdot(y, scalar.trace.mv(x)), jnp.vdot(scalar.load.mv(y), x))
    maxwell_x = jnp.concatenate((x, jnp.zeros(4, dtype=x.dtype)))
    assert jnp.allclose(
        jnp.vdot(y, maxwell.tangential_trace.mv(maxwell_x)),
        jnp.vdot(maxwell.boundary_load.mv(y), maxwell_x),
    )
    coupled = prepare_scalar_nonmatching_fem_bem_3d(jnp.eye(2), jnp.eye(2), scalar)
    coupled_result = coupled.solve(jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0]))
    assert bool(coupled_result.successful)
    edge = CurvedVirtualElementEdge(
        "circle",
        # ty: ignore[invalid-argument-type]
        [[1.0, 0.0], [0.0, 1.0]],
        # ty: ignore[invalid-argument-type]
        [[0.0, 1.0], [-1.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [0.5, 0.5],
    )
    assert edge.minimum_jacobian > 0.0
    # ty: ignore[invalid-argument-type]
    epoch = VirtualElementEpoch([3, 7], [1, 2])
    result = adapt_virtual_element_p(
        epoch,
        # ty: ignore[invalid-argument-type]
        [1.0, 0.1],
        jnp.eye(2),
        VirtualElementAdaptivityPolicy(maximum_degree=4),
    )
    assert result.conservation_defect == 0.0
    assert result.target.generation == 1
    # ty: ignore[invalid-argument-type]
    condition = ScalarScreenJunctionCondition3D("tip", "continuity", [[1.0, -1.0]])
    prepared = prepare_scalar_screen_junction_solve_3d(jnp.eye(2), (condition,))
    result = prepared.solve(jnp.asarray([1.0, 1.0]))
    assert bool(result.successful)
    assert result.constraint_defect < 1e-10

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]])
    displacement = prepare_displacement_discontinuity_3d(
        vertices, faces, shear_modulus=1.0, poisson_ratio=0.25
    )
    rigid = jnp.ones((4, 3))
    assert jnp.linalg.norm(displacement.traction(rigid)) < 1e-8
    assert not displacement.evidence.dp0_hypersingular_supported
