#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_spatial_noise_separates_basis_runtime_and_certification_precision():
    precision = phx.stochastic.SpatialNoisePrecisionPolicy(
        construction_dtype="float64",
        basis_storage_dtype="float32",
        runtime_dtype="float32",
        certification_dtype="float64",
    )
    modes = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=jnp.float64,
    )
    basis = phx.stochastic.SpatialNoiseBasis(
        modes,
        jnp.asarray([2.0, 0.5]),
        quadrature_weights=jnp.ones((4,)),
        field_space_id="field",
        precision=precision,
    )

    assert basis.modes.dtype == jnp.float32
    assert basis.eigenvalues.dtype == jnp.float32
    assert basis.quadrature_weights.dtype == jnp.float64
    assert basis.diffusion.dtype == jnp.float32
    assert basis.precision_evidence.evidence_id


def test_spatial_noise_factory_uses_construction_then_storage_precision():
    precision = phx.stochastic.SpatialNoisePrecisionPolicy(
        construction_dtype="float64",
        basis_storage_dtype="float32",
        runtime_dtype="float32",
        certification_dtype="float64",
    )
    basis = phx.stochastic.SpatialNoiseBasis.from_discrete_covariance(
        jnp.asarray([[2.0, 0.0], [0.0, 0.5]], dtype=jnp.float64),
        state_shape=(2,),
        quadrature_weights=jnp.ones((2,), dtype=jnp.float64),
        rank=2,
        precision=precision,
    )

    assert basis.modes.dtype == jnp.float32
    assert basis.eigenvalues.dtype == jnp.float32
    assert basis.diffusion.dtype == jnp.float32
    assert basis.approximation is not None
    assert basis.approximation.method == "dense_eigh"


def _level(index, *, parent=None, witness=None):
    return phx.stochastic.StochasticLevelSpec(
        f"level-{index}",
        index,
        refinement_axes=("space",),
        resolutions=(1.0 / (index + 1),),
        state_shape=(4,),
        problem_id="problem",
        observable_id="observable",
        solver_id="solver",
        approximation_id=f"approx-{index}",
        parent_level_id=parent,
        discretization_id=f"space-{index}",
        basis_id=f"basis-{index}",
        state_transfer_id=None,
        noise_coupling="shared" if index == 0 else "nested",
        noise_witness=witness,
        metadata={"noise_family_id": "family"},
    )


def test_nested_noise_requires_a_passing_projection_witness():
    base = _level(0)
    with pytest.raises(ValueError, match="passing projection witness"):
        phx.stochastic.StochasticCouplingPlan(
            (base, _level(1, parent="level-0")),
            hierarchy_id="hierarchy",
        )

    witness = phx.stochastic.NoiseCouplingWitness(
        "basis-map",
        covariance_residual=1e-8,
        increment_residual=2e-8,
        tolerance=1e-6,
    )
    hierarchy = phx.stochastic.StochasticCouplingPlan(
        (base, _level(1, parent="level-0", witness=witness)),
        hierarchy_id="hierarchy",
    )
    assert hierarchy.coupled
    assert hierarchy.level(1).noise_witness is witness


def test_semidiscrete_spde_composes_spatial_and_noise_precision_evidence():
    fd_precision = phx.discretization.FDExecutionPrecisionPolicy(
        coefficient_dtype="float32",
        field_dtype="float32",
        accumulation_dtype="float64",
        certification_dtype="float64",
    )
    axis = phx.discretization.UniformAxisSpec(
        8,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    discretization = phx.discretization.periodic_finite_difference(
        phx.discretization.PreparedTensorGrid((axis,)),
        precision=fd_precision,
    )
    noise_precision = phx.stochastic.SpatialNoisePrecisionPolicy(
        construction_dtype="float64",
        basis_storage_dtype="float32",
        runtime_dtype="float32",
        certification_dtype="float64",
    )
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        discretization,
        0.03,
        rank=2,
        precision=noise_precision,
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros(discretization.state_shape, dtype=jnp.float32),
        discretization,
        t0=0.0,
        t1=0.1,
        kappa=0.05,
        noise_basis=basis,
    )

    children = dict(spde.precision_evidence.children)
    assert children["spatial"].evidence_id == fd_precision.evidence().evidence_id
    assert children["noise"].evidence_id == basis.precision_evidence.evidence_id
    assert spde.precision_evidence_id == spde.precision_evidence.evidence_id
    assert spde.discretization_bundle.records[-1].precision_evidence_id == (
        spde.precision_evidence_id
    )
