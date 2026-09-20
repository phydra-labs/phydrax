#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_bounded_atlas_distinguishes_sampled_and_cell_certified_cover():
    chart = phx.metrix.CoordinateChart("line", ("x",))
    candidate = phx.metrix.AtlasCandidate(
        chart,
        lambda coordinate: jnp.concatenate(
            (coordinate, jnp.zeros_like(coordinate)), axis=-1
        ),
        lambda point: point[..., :1],
        lambda coordinate: jnp.abs(coordinate[..., 0]) < 2.0,
        candidate_id="line-chart",
    )
    points = jnp.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    sampled = phx.metrix.prepare_atlas(
        (candidate,),
        phx.metrix.CompactAtlasDomain(points, domain_id="sampled-line"),
    )
    assert bool(sampled.certificate.valid)
    assert bool(sampled.certificate.sampled)
    assert not bool(sampled.certificate.certified)

    certified = phx.metrix.prepare_atlas(
        (candidate,),
        phx.metrix.CompactAtlasDomain(
            points,
            certified_cells=jnp.ones((3,), dtype="bool"),
            domain_id="certified-line",
        ),
        phx.metrix.AtlasConstructionPolicy(require_certified_cover=True),
    )
    assert bool(certified.certificate.certified)
    assert certified.path_table == ((0, 0),)


def test_regular_level_set_projector_retraction_and_immersion_measure():
    sphere = phx.metrix.RegularLevelSetManifold(
        lambda point: jnp.asarray([jnp.dot(point, point) - 1.0]),
        ambient_dimension=2,
        codimension=1,
        tolerance=1e-6,
        manifold_id="unit-circle-level-set",
    )
    point = jnp.asarray([1.0, 0.0])
    evidence = sphere.local_geometry(point)
    assert bool(evidence.valid)
    assert jnp.allclose(
        evidence.constraint_jacobian @ evidence.tangent_projector, 0.0, atol=1e-6
    )
    assert jnp.allclose(
        evidence.tangent_projector @ evidence.tangent_projector,
        evidence.tangent_projector,
        atol=1e-6,
    )
    retracted = sphere.retract(point, jnp.asarray([0.0, 0.1]))
    assert bool(sphere.contains(retracted))

    immersion = phx.metrix.ImmersedRiemannianManifoldAdapter(
        lambda coordinate: jnp.stack((jnp.cos(coordinate[0]), jnp.sin(coordinate[0]))),
        coordinate_dimension=1,
        ambient_dimension=2,
        manifold_id="circle-immersion",
    )
    map_evidence = immersion.map_measure_evidence(jnp.asarray([0.3]))
    assert bool(map_evidence.valid)
    assert jnp.allclose(map_evidence.metric, jnp.ones((1, 1)), atol=1e-6)
    assert jnp.allclose(map_evidence.hausdorff_jacobian, 1.0, atol=1e-6)


def test_complex_leaf_consumes_jax_cotangent_once_and_adam_moments_are_real():
    point = jnp.asarray([1.0 + 2.0j, -0.5 + 0.25j])
    manifold = phx.metrix.ComplexEuclideanManifold((2,))
    cotangent = jax.grad(lambda value: jnp.real(jnp.vdot(value, value)))(point)
    gradient = manifold.egrad_to_rgrad(point, cotangent)
    assert jnp.allclose(gradient, 2.0 * point)
    geometry = phx.optim.ParameterGeometry(point, {"<root>": manifold})
    optimizer = phx.optim.riemannian_adam(geometry, learning_rate=1e-2)
    state = optimizer.init(point)
    assert jnp.issubdtype(state.second_moment.dtype, jnp.floating)
    destination, state = optimizer.update(cotangent, state, point)
    assert destination.dtype == point.dtype
    assert jnp.issubdtype(state.second_moment.dtype, jnp.floating)


def test_fixed_rank_strata_are_smooth_only_inside_one_rank_epoch():
    manifold = phx.metrix.FixedRankDensityManifold(3, 2, tolerance=1e-7)
    factor = jnp.asarray(
        [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j], [0.0j, 0.0j]]
    ) / jnp.sqrt(2.0)
    assert bool(manifold.contains(factor))
    gauge = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype="complex128")
    assert jnp.allclose(manifold.density(factor), manifold.density(factor @ gauge))

    stratification = phx.metrix.DensityRankStratification(3)
    evidence = stratification.classify(manifold.density(factor))
    assert bool(evidence.valid)
    assert evidence.rank == 2
    proposal = stratification.propose_transition(
        factor, 1, trigger="discard-small-support"
    )
    assert proposal.target_rank == 1
    assert bool(proposal.valid)
    ambiguous = stratification.classify(jnp.diag(jnp.asarray([1.0, 1e-8, 0.0])))
    assert bool(ambiguous.ambiguous)


def test_calabi_yau_certificate_never_promotes_residual_to_exact_ricci_flatness():
    certificate = phx.geometry.complex.CalabiYauCertificate(
        projective_dimension=4,
        degree=5,
        nonzero_polynomial=True,
        cellular_cover_certified=True,
        gradient_lower_bound=0.5,
        transition_residual=1e-10,
        residue_residual=1e-10,
        metric_minimum_eigenvalue=0.2,
        volume_error_bound=1e-4,
        monge_ampere_sup_bound=1e-3,
        topology_certified=False,
        tolerance=1e-6,
    )
    assert bool(certificate.adjunction_conclusion)
    assert bool(certificate.completeness_conclusion)
    assert bool(certificate.epsilon_candidate)
    assert not certificate.ricci_flat_claim
    assert not bool(certificate.topology_certified)


def test_trainable_hypersurface_continues_only_fixed_simple_root_ancestry():
    family = phx.geometry.complex.TrainableHomogeneousHypersurface(
        jnp.asarray([[2, 0], [0, 2]]),
        jnp.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
        pivot=0,
        family_id="quadratic-cp1-family",
    )
    epoch = phx.geometry.complex.PreparedHypersurfaceEpoch(
        jnp.asarray([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]),
        jnp.asarray([[1.0 + 0.0j, 0.0j], [1.0 + 0.0j, 0.0j]]),
        jnp.asarray([1.0j, -1.0j]),
        jnp.asarray([1, 1]),
        epoch_id="quadratic-simple-roots",
    )
    continued = family.with_coefficients(jnp.asarray([1.0 + 0.0j, 1.21 + 0.0j]))
    roots, evidence = epoch.continue_roots(continued)
    assert bool(evidence.valid)
    assert jnp.allclose(
        jnp.sort(jnp.imag(roots)),
        jnp.asarray([-1.1, 1.1]),
        atol=1e-5,
    )
    singular = family.with_coefficients(jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j]))
    _, singular_evidence = epoch.continue_roots(singular)
    assert not bool(singular_evidence.valid)
