import pytest

from phydrax.atomistic.interchange._structure_records import (
    read_pdb_atom_records,
    select_pdb_model,
)
from phydrax.units import ANGSTROM


def _atom(serial, *, name="CA", alt="", occupancy=1.0, insertion="A", x=1.0):
    return (
        f"ATOM  {serial:5d} {name:>4s}{alt:1s}ALA B{17:4d}{insertion:1s}   "
        f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}{occupancy:6.2f}{15.0:6.2f}          {'C':>2s}  "
    )


def test_models_author_insertions_and_alternate_coordinates_are_retained():
    lines = [
        "MODEL        1",
        _atom(1, alt="A", occupancy=0.6),
        _atom(2, alt="B", occupancy=0.4, x=9.0),
        "ENDMDL",
        "MODEL        2",
        _atom(1, x=4.0),
        "ENDMDL",
    ]
    rows = read_pdb_atom_records("\n".join(lines), source_id="raw-pdb-digest")
    assert len(rows) == 3
    assert rows[0].record_id != rows[2].record_id
    assert rows[0].raw_line == lines[1]
    assert rows[0].atom_identity == ("B", "17", "A", "CA")
    assert rows[0].length_unit == ANGSTROM
    with pytest.raises(ValueError, match="explicit conformer"):
        select_pdb_model(rows, "1", alternate_locations={})
    selected = select_pdb_model(
        rows, "1", alternate_locations={("B", "17", "A", "CA"): "B"}
    )
    assert selected[0].position == (9.0, 2.0, 3.0)
    assert selected[0].occupancy == 0.4
    second = select_pdb_model(rows, "2", alternate_locations={})
    assert second[0].position == (4.0, 2.0, 3.0)


def test_incomplete_elements_zero_occupancy_and_duplicate_serials_refuse():
    with pytest.raises(ValueError, match="element"):
        read_pdb_atom_records(_atom(1)[:76], source_id="bad-source")
    with pytest.raises(ValueError, match="repeated atom serial"):
        read_pdb_atom_records(_atom(1) + "\n" + _atom(1, name="C"), source_id="duplicate")
    zero = read_pdb_atom_records(_atom(1, occupancy=0.0), source_id="zero-occupancy")
    with pytest.raises(ValueError, match="positive-occupancy"):
        select_pdb_model(zero, "1", alternate_locations={})
