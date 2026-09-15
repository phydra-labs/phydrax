import pytest

from phydrax.metrix._spacetime_conventions import RelativityConvention


def test_canonical_relativity_convention_round_trips_with_stable_identity():
    convention = RelativityConvention.canonical()
    restored = RelativityConvention.from_dict(convention.to_dict())

    assert convention.metric_signature == "mostly_plus"
    assert convention.riemann_sign == 1
    assert convention.extrinsic_curvature_sign == -1
    assert convention.spacetime_orientation == 1
    assert convention.future_time_orientation == 1
    assert convention.azimuthal_orientation == 1
    assert convention.fourier_sign == -1
    assert restored.convention_id == convention.convention_id
    assert restored.to_dict() == convention.to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric_signature", "mostly_minus"),
        ("riemann_sign", -1),
        ("extrinsic_curvature_sign", 1),
        ("spacetime_orientation", -1),
        ("future_time_orientation", -1),
        ("azimuthal_orientation", -1),
        ("fourier_sign", 1),
    ],
)
def test_every_relativity_convention_choice_changes_content_identity(field, value):
    canonical = RelativityConvention.canonical()
    values = {
        "metric_signature": canonical.metric_signature,
        "riemann_sign": canonical.riemann_sign,
        "extrinsic_curvature_sign": canonical.extrinsic_curvature_sign,
        "spacetime_orientation": canonical.spacetime_orientation,
        "future_time_orientation": canonical.future_time_orientation,
        "azimuthal_orientation": canonical.azimuthal_orientation,
        "fourier_sign": canonical.fourier_sign,
    }
    values[field] = value

    changed = RelativityConvention(**values)

    assert changed.convention_id != canonical.convention_id


def test_relativity_convention_rejects_ambiguous_signs_and_tampered_payloads():
    with pytest.raises(ValueError, match="either -1 or \\+1"):
        RelativityConvention(riemann_sign=0)
    with pytest.raises(TypeError, match="integer sign"):
        RelativityConvention(fourier_sign=True)
    with pytest.raises(ValueError, match="metric_signature"):
        RelativityConvention(metric_signature="euclidean")

    payload = RelativityConvention.canonical().to_dict()
    payload["fourier_sign"] = 1
    with pytest.raises(ValueError, match="fingerprint"):
        RelativityConvention.from_dict(payload)
