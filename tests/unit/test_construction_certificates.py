import pytest

import phydrax as phx
from phydrax import AbstractConstructionCertificate
from phydrax.nn.operator.architectures import ConditionalHolomorphicMapCertificate
from phydrax.nn.operator.layers import CliffordEquivarianceCertificate


def _frame():
    return phx.equations.HolomorphicLinearFrameCertificate(
        complex_input_size=1,
        complex_output_size=1,
        real_coefficient_count=4,
        maximum_derivative_order=2,
        normalization_id="n",
        basis_construction="b",
        construction_dependencies=("d",),
    )


def _meromorphic_frame():
    return phx.equations.MeromorphicLinearFrameCertificate(
        complex_output_size=1,
        real_coefficient_count=6,
        maximum_derivative_order=2,
        normalization_id="n",
        pole_set_id="p",
    )


# Certificate identities are persisted in interchange records, constraint plans,
# and trial-space audits, so migrating the certificates must not change them.
_CASES = {
    "holomorphic-map": (
        lambda: phx.equations.HolomorphicMapCertificate(
            complex_input_size=1,
            complex_output_size=2,
            construction="c",
            normalization_id="n",
            maximum_derivative_order=3,
            operations=("complex-affine",),
            parameter_coverage="finite-parametric-family",
            linear_in_parameters=False,
            construction_dependencies=("d",),
        ),
        "fb5e5482f1a5e0603cec1e1f297b2c87aa6903cf8ac6194f56f657d0b6129bd9",
    ),
    "holomorphic-linear-frame": (
        _frame,
        "c2d2026828cdd28c74dfee279bb3e656451f678f5ca86a754b66fba9ceab55af",
    ),
    "exact-pde-trial-space": (
        lambda: phx.equations.TrialSpaceCertificate(
            equation_family="laplace",
            ambient_dimension=2,
            construction="c",
            normalization_id="n",
            basis_id="b",
            rank=3,
            assumptions=("a",),
            construction_residual=0.0,
            construction_tolerance=0.0,
        ),
        "2bec94d57b9fb7e7053a9c36e8e995aec54539c3fbcb7aafdfbb29927b9b09fa",
    ),
    "meromorphic-linear-frame": (
        _meromorphic_frame,
        "cff1a57fe941328349a13260880cc9048040c25bc4b0744c8fcf123f3a202c14",
    ),
    "meromorphic-map": (
        lambda: phx.equations.MeromorphicMapCertificate(
            _meromorphic_frame(), parameter_mode="m", construction_dependency="d"
        ),
        "b74f0f0e2972a3bc0cfbd7bd09eb804cfae6549ad1d53bf81969e5c094455579",
    ),
    "pluriharmonic-map": (
        lambda: phx.equations.PluriharmonicCertificate(
            complex_dimension=2, branch=0, holomorphic_certificate_id="h"
        ),
        "c534535b706154e79bf03a713c57d896aaa1bd0176c33635467b4e07b8da82e1",
    ),
    "clifford-equivariance": (
        lambda: CliffordEquivarianceCertificate(
            algebra_id="a",
            input_representation_id="i",
            output_representation_id="o",
            construction="c",
        ),
        "ca2c3b5ce362a255c439a167a3fa207c12e0d83db93dcba53882d9f5f06d9bb8",
    ),
    "conditional-holomorphic-map": (
        lambda: ConditionalHolomorphicMapCertificate(
            query_complex_input_size=1,
            complex_output_size=1,
            latent_size=2,
            maximum_derivative_order=2,
            trunk_mode="unconstrained",
            frame_id="f",
            constraint_operator_id=None,
            coefficient_layout="l",
            bias_mode="b",
            branch_names=("u",),
            branch_fusion="sum",
        ),
        "c79aaf6440b28f7051026db6eb9b910eaef568ff58548fcac3061333022b2326",
    ),
}


@pytest.mark.parametrize("capability", sorted(_CASES))
def test_construction_certificates_keep_their_identities(capability):
    build, identity = _CASES[capability]
    certificate = build()

    assert isinstance(certificate, AbstractConstructionCertificate)
    assert certificate.capability_id == capability
    assert certificate.certificate_id == identity


def test_meromorphic_map_references_its_frame_certificate():
    frame = _meromorphic_frame()
    certificate = phx.equations.MeromorphicMapCertificate(
        frame, parameter_mode="m", construction_dependency="d"
    )

    assert certificate.frame_id == frame.certificate_id
