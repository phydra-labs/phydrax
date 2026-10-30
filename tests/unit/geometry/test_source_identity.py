import pytest

from phydrax.geometry._source_identity import source_entity_id


type SourceAddress = tuple[str, str, str, str | int, tuple[str, ...]]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(
            ("first-source", "revision", "face", "entity", ()),
            ("second-source", "revision", "face", "entity", ()),
            id="source-namespace",
        ),
        pytest.param(
            ("source", "before", "face", "entity", ()),
            ("source", "after", "face", "entity", ()),
            id="source-revision",
        ),
        pytest.param(
            ("source", "revision", "face", "entity", ()),
            ("source", "revision", "curve", "entity", ()),
            id="scientific-role",
        ),
        pytest.param(
            ("source", "revision", "face", 1, ()),
            ("source", "revision", "face", "1", ()),
            id="integer-versus-authored-string",
        ),
        pytest.param(
            ("source", "revision", "face", "entity", ("first-instance",)),
            ("source", "revision", "face", "entity", ("second-instance",)),
            id="shared-definition-distinct-occurrences",
        ),
        pytest.param(
            ("source", "revision", "face", 'entity:occurrence:["instance"]', ()),
            ("source", "revision", "face", "entity", ("instance",)),
            id="delimiter-shaped-authored-identifier",
        ),
        pytest.param(
            ("source", "revision", "face", "entity", ("parent", "child")),
            ("source", "revision", "face", "entity", ("child", "parent")),
            id="ordered-occurrence-path",
        ),
    ],
)
def test_source_identity_preserves_declared_entity_address(
    left: SourceAddress, right: SourceAddress
) -> None:
    left_source, left_revision, left_kind, left_entity, left_path = left
    right_source, right_revision, right_kind, right_entity, right_path = right
    left_identity = source_entity_id(
        left_source,
        left_revision,
        left_kind,
        left_entity,
        occurrence_path=left_path,
    )
    right_identity = source_entity_id(
        right_source,
        right_revision,
        right_kind,
        right_entity,
        occurrence_path=right_path,
    )
    assert left_identity != right_identity
