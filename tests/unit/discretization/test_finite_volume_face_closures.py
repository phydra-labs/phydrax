#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _euler_states(count, *, seed=0):
    system = phx.equations.EulerSystem(2)
    rng = np.random.default_rng(seed)
    primitive = np.stack(
        (
            rng.uniform(0.8, 1.2, count),
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(0.8, 1.2, count),
        ),
        axis=-1,
    )
    return system, system.primitive_to_conserved(jnp.asarray(primitive))


def _unit_normals(count, *, seed=1):
    angles = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, count)
    return jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)


def _context(normal, *, measure=None, velocity=None):
    batch = normal.shape[:-1]
    return phx.discretization.FaceFluxContext(
        normal,
        jnp.ones(batch) if measure is None else measure,
        velocity,
        geometry_id="probe-geometry",
    )


def _rotate(values, rotation):
    momentum = values[..., 1:3] @ rotation.T
    return values.at[..., 1:3].set(momentum)


def _rotation(angle):
    return jnp.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))


class _NetworkGenerator(phx.StrictModule):
    """Unconstrained learned generator g(a, b, n) of a face correction."""

    network: phx.nn.models.MLP

    def __call__(self, system, left, right, baseline, context, args=None):
        del system, baseline, args
        features = jnp.concatenate((left, right, context.unit_normal), axis=-1)
        flat = features.reshape((-1, features.shape[-1]))
        output = jax.vmap(self.network)(flat).reshape(left.shape)
        return 0.05 * output


def _network_generator(key=0, components=4, dimension=2):
    return _NetworkGenerator(
        phx.nn.models.MLP(
            in_size=2 * components + dimension,
            out_size=components,
            width_size=8,
            depth=1,
            key=jr.key(key),
        )
    )


def _learned_closure(key=0, **options):
    return phx.discretization.ArbitraryNormalFaceClosurePlan(
        phx.discretization.SymmetrizedFaceClosure(_network_generator(key)),
        closure_id="learned-symmetrized-face-closure",
        **options,
    )


def test_symmetrized_learned_closure_is_consistent_and_orientation_antisymmetric():
    system, left = _euler_states(6)
    _, right = _euler_states(6, seed=3)
    normal = _unit_normals(6)
    baseline = phx.discretization.RusanovFluxPlan().normal_face_flux(
        system, left, right, normal
    )
    closure = _learned_closure()

    forward = closure.apply(system, left, right, baseline.normal_flux, _context(normal))
    reversed_ = closure.apply(
        system, right, left, -baseline.normal_flux, _context(-normal)
    )
    equal = closure.apply(system, left, left, baseline.normal_flux, _context(normal))

    correction = forward - baseline.normal_flux
    assert float(jnp.max(jnp.abs(correction))) > 1e-4
    np.testing.assert_allclose(reversed_ + baseline.normal_flux, -correction, atol=1e-14)
    np.testing.assert_array_equal(equal, baseline.normal_flux)
    assert tuple(certificate.capability_id for certificate in closure.certificates) == (
        "face-closure-consistency-antisymmetry",
    )


def test_declared_closure_contract_rejects_inconsistent_nonfinite_and_mistyped():
    system, left = _euler_states(3)
    normal = _unit_normals(3)
    baseline = jnp.zeros_like(left)

    offset = phx.discretization.ArbitraryNormalFaceClosurePlan(
        lambda system, left, right, baseline, context, args: jnp.full_like(
            baseline, 1e-3
        ),
        closure_id="constant-offset",
    )
    with pytest.raises(Exception, match="equal-state consistency"):
        offset.apply(system, left, left, baseline, _context(normal))

    nonfinite = phx.discretization.ArbitraryNormalFaceClosurePlan(
        lambda system, left, right, baseline, context, args: jnp.full_like(
            baseline, jnp.nan
        ),
        closure_id="nonfinite",
    )
    with pytest.raises(Exception, match="nonfinite correction"):
        nonfinite.apply(system, left, left + 1.0, baseline, _context(normal))

    single = phx.discretization.ArbitraryNormalFaceClosurePlan(
        lambda system, left, right, baseline, context, args: (right - left).astype(
            jnp.float32
        ),
        closure_id="single-precision",
    )
    with pytest.raises(TypeError, match="dtype"):
        single.apply(system, left, left + 1.0, baseline, _context(normal))


def test_face_flux_context_requires_unit_normal_positive_measure_and_axis_identity():
    normal = _unit_normals(4)
    with pytest.raises(Exception, match="finite unit normals"):
        _context(2.0 * normal)
    with pytest.raises(Exception, match="positive finite"):
        _context(normal, measure=jnp.asarray([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="broadcasting is not permitted"):
        _context(normal, measure=jnp.ones(()))
    with pytest.raises(Exception, match="Cartesian"):
        phx.discretization.FaceFluxContext(
            normal, jnp.ones(4), geometry_id="probe", axis=0
        )
    inactive = phx.discretization.FaceFluxContext(
        jnp.zeros((2, 2)),
        jnp.zeros(2),
        geometry_id="probe",
        active=jnp.zeros(2, dtype=bool),
    )
    assert not bool(jnp.any(inactive.active))


def test_normal_frame_closure_is_rotation_covariant_and_requires_capability():
    system, left = _euler_states(5)
    _, right = _euler_states(5, seed=7)
    normal = _unit_normals(5)
    rusanov = phx.discretization.RusanovFluxPlan()

    def anisotropic(system, left, right, baseline, context, args):
        # Deliberately frame-dependent: it weights only the first momentum slot.
        return (right - left) * jnp.asarray((0.01, 0.05, 0.0, 0.02))

    rotation = _rotation(0.83)

    def corrected(closure, rotate):
        a = _rotate(left, rotation) if rotate else left
        b = _rotate(right, rotation) if rotate else right
        n = normal @ rotation.T if rotate else normal
        baseline = rusanov.normal_face_flux(system, a, b, n).normal_flux
        return closure.apply(system, a, b, baseline, _context(n)) - baseline

    normal_frame = phx.discretization.ArbitraryNormalFaceClosurePlan(
        anisotropic, closure_id="normal-frame", frame="face-normal"
    )
    global_frame = phx.discretization.ArbitraryNormalFaceClosurePlan(
        anisotropic, closure_id="global-frame"
    )
    np.testing.assert_allclose(
        corrected(normal_frame, True),
        _rotate(corrected(normal_frame, False), rotation),
        atol=1e-14,
    )
    assert not np.allclose(
        corrected(global_frame, True),
        _rotate(corrected(global_frame, False), rotation),
        atol=1e-6,
    )
    scalar = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="normal-frame-free-scalar",
    )
    with pytest.raises(ValueError, match="AbstractNormalFrameSystem"):
        normal_frame.admit_system(scalar)


def test_normal_frame_transforms_are_inverse_and_scalar_preserving():
    system, state = _euler_states(4)
    normal = _unit_normals(4)
    local = system.rotate_state_to_normal_frame(state, normal)
    np.testing.assert_allclose(local[..., (0, 3)], state[..., (0, 3)])
    np.testing.assert_allclose(
        local[..., 1], jnp.sum(state[..., 1:3] * normal, axis=-1), atol=1e-15
    )
    np.testing.assert_allclose(
        system.rotate_flux_from_normal_frame(
            system.rotate_flux_to_normal_frame(state, normal), normal
        ),
        state,
        atol=1e-15,
    )


def test_face_closure_refuses_magnetic_systems():
    closure = _learned_closure()
    with pytest.raises(ValueError, match="constrained MHD"):
        closure.admit_system(phx.equations.IdealMHDSystem(2))


def _grid(shape):
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xy"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def _extrapolation_pair():
    return phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )


def _structured(system, mapped, closure, interface_solver=None):
    discretization = phx.discretization.FiniteVolumePlan(
        _grid((4, 3)), component_names=system.component_names
    ).prepare()
    if mapped:
        discretization = phx.discretization.MappedFiniteVolumePlan(
            discretization, lambda point: point, mapping_id="identity"
        ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "closure-structured",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(
            ("x", "y"), (_extrapolation_pair(), _extrapolation_pair())
        ),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan()
        if interface_solver is None
        else interface_solver,
        closure=closure,
    )
    return phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics


def _state(system, shape):
    x = jnp.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
    primitive = jnp.stack(
        (1.0 + 0.2 * jnp.sin(3.0 * x), 0.2 * x, -0.1 * x**2, 1.0 + 0.1 * x), axis=-1
    )
    return system.primitive_to_conserved(primitive)


def test_mapped_geometry_admits_custom_arbitrary_normal_flux_and_refuses_axis_only():
    system = phx.equations.EulerSystem(2)

    class _CentralFlux(phx.discretization.AbstractArbitraryNormalNumericalFluxPlan):
        def __init__(self):
            self.flux_id = "custom-central-dissipative"
            self.differentiability = phx.BranchDifferentiationPolicy.SMOOTH

        def face_flux(self, system, left, right, axis, args=None, /):
            normal = jnp.zeros((system.dimension,)).at[axis].set(1.0)
            return self.normal_face_flux(
                system, left, right, jnp.broadcast_to(normal, left.shape[:-1] + (2,))
            )

        def normal_face_flux(self, system, left, right, normal, args=None, /):
            central = 0.5 * (
                system.physical_normal_flux(left, normal)
                + system.physical_normal_flux(right, normal)
            )
            speed = jnp.full(left.shape[:-1], 2.0)
            return phx.discretization.NumericalFluxResult(
                central - 0.5 * speed[..., None] * (right - left), speed
            )

    state = _state(system, (4, 3))
    cartesian = _structured(system, False, None, _CentralFlux())
    mapped = _structured(system, True, None, _CentralFlux())
    np.testing.assert_allclose(mapped(0.0, state), cartesian(0.0, state), atol=1e-12)
    with pytest.raises(ValueError, match="arbitrary-normal numerical flux"):
        _structured(system, True, None, phx.discretization.RoeFluxPlan())


def test_triangle_closure_is_applied_at_every_face_and_conserves():
    system = phx.equations.EulerSystem(2)
    x = np.linspace(0.0, 1.0, 4)
    vertices = np.asarray([(xi, yi) for yi in x for xi in x])
    triangles = []
    for j in range(3):
        for i in range(3):
            lower_left = j * 4 + i
            triangles.append((lower_left, lower_left + 1, lower_left + 5))
            triangles.append((lower_left, lower_left + 5, lower_left + 4))
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices,
        np.asarray(triangles, dtype=np.int32),
        component_names=system.component_names,
    ).prepare()
    boundaries = phx.discretization.TriangleFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "closure-triangle", "state", system, boundaries
    )

    def compiled(closure):
        return phx.equations.compile_conservation_problem(
            problem,
            discretization,
            phx.discretization.TriangleFiniteVolumeMethodPlan(
                phx.discretization.PiecewiseConstantReconstruction(),
                phx.discretization.HLLCFluxPlan(),
                closure=closure,
            ),
        ).dynamics

    state = _state(system, (discretization.cell_count,))
    with_closure = compiled(_learned_closure())
    contribution = with_closure(0.0, state) - compiled(None)(0.0, state)
    _, diagnostics = with_closure.residual_with_diagnostics(0.0, state)

    assert float(jnp.max(jnp.abs(contribution))) > 1e-5
    np.testing.assert_allclose(
        jnp.sum(contribution * discretization.cell_volumes[:, None], axis=0),
        0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(diagnostics.conservation_defect, 0.0, atol=1e-13)
