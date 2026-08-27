"""JunctionValidityRepair must clamp descriptor genes into the encoding's
declared domains (generate_bounds), matching the decoder's interpretation."""
import pytest

from src.encoding import (
    PieceIndex,
    compute_dimensions,
    create_chromosome_from_pieces,
    get_junction,
    set_junction,
)
from src.repair import JunctionValidityRepair
from src.templates import TEMPLATES


@pytest.fixture
def repair_and_dims(catalog, switches_config):
    dims = compute_dimensions(switches_config, catalog)
    inv = catalog.inventory_by_index(switches_config.inventory)
    return JunctionValidityRepair(dims, inv), dims


def _repaired_handedness(repair: JunctionValidityRepair, dims, hand: int) -> int:
    x = create_chromosome_from_pieces(dims, [PieceIndex.R40_CURVE] * 16)
    set_junction(x, dims, 0, active=1, position=3, handedness=hand, n_straights=1)
    repair.repair_chromosome(x)
    _active, _pos, hand_after, _n_str = get_junction(x, dims, 0)
    return int(hand_after)


class TestHandednessClamp:
    """The handedness gene's declared domain is [0, 1]; repair maps stray
    values with the decoder's own rule (mod n_templates), so a repaired gene
    always decodes to the same template as the raw value would."""

    @pytest.mark.parametrize("hand", [-2, -1, 2, 3])
    def test_out_of_domain_lands_in_declared_bounds(self, repair_and_dims, hand):
        repair, dims = repair_and_dims
        assert 0 <= _repaired_handedness(repair, dims, hand) <= 1

    @pytest.mark.parametrize("hand", [-2, -1, 0, 1, 2, 3])
    def test_repair_preserves_decoded_template(self, repair_and_dims, hand):
        repair, dims = repair_and_dims
        assert _repaired_handedness(repair, dims, hand) == hand % len(TEMPLATES)
