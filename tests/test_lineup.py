import pytest

from fantasy_ml.decisions.lineup import optimize_lineup


def player(player_id, position, points, eligible_positions=None):
    return {
        "player_id": player_id,
        "name": player_id,
        "position": position,
        "projected_points": points,
        **(
            {"eligible_positions": eligible_positions}
            if eligible_positions is not None
            else {}
        ),
    }


def test_optimizer_selects_highest_projected_legal_lineup():
    roster = [
        player("qb1", "QB", 20),
        player("rb1", "RB", 18),
        player("rb2", "RB", 15),
        player("rb3", "RB", 12),
        player("wr1", "WR", 17),
        player("wr2", "WR", 14),
        player("wr3", "WR", 11),
        player("te1", "TE", 10),
        player("te2", "TE", 7),
        player("k1", "K", 8),
        player("def1", "DEF", 6),
    ]

    result = optimize_lineup(roster)

    assert len(result["starters"]) == 9
    assert len(result["bench"]) == 2
    assert result["projected_points"] == pytest.approx(120.0)
    assert len({p["player_id"] for p in result["starters"]}) == 9


def test_flex_uses_best_eligible_player_without_duplicate_assignment():
    roster = [
        player("qb1", "QB", 20),
        player("rb1", "RB", 18),
        player("rb2", "RB", 15),
        player("wr1", "WR", 17),
        player("wr2", "WR", 14),
        player("wr3", "WR", 13),
        player("te1", "TE", 10),
        player("k1", "K", 8),
        player("def1", "DEF", 6),
    ]

    result = optimize_lineup(roster)
    flex = next(p for p in result["starters"] if p["slot"] == "FLEX")

    assert flex["player_id"] == "wr3"
    assert len({p["player_id"] for p in result["starters"]}) == 9


def test_optimizer_respects_custom_slot_eligibility():
    roster = [
        player("rb1", "RB", 12),
        player("wr1", "WR", 15),
    ]
    slots = {"FLEX": ("RB", "WR")}

    result = optimize_lineup(roster, slots=slots)

    assert result["starters"][0]["player_id"] == "wr1"
    assert result["bench"][0]["player_id"] == "rb1"


def test_optimizer_rejects_duplicate_player_ids():
    roster = [
        player("p1", "QB", 20),
        player("p1", "RB", 10),
    ]

    with pytest.raises(ValueError, match="Duplicate player_id"):
        optimize_lineup(roster, slots={"QB": ("QB",)})


def test_optimizer_rejects_roster_that_cannot_fill_slots():
    roster = [player("rb1", "RB", 12)]

    with pytest.raises(ValueError, match="cannot fill every slot"):
        optimize_lineup(
            roster,
            slots={"QB": ("QB",), "RB": ("RB",)},
        )


def test_optimizer_rejects_nonfinite_projection():
    roster = [player("qb1", "QB", float("nan"))]

    with pytest.raises(ValueError, match="finite numeric"):
        optimize_lineup(roster, slots={"QB": ("QB",)})

def test_optimizer_rejects_empty_slot_eligibility():
    roster = [player("rb1", "RB", 12)]

    with pytest.raises(ValueError, match="at least one position"):
        optimize_lineup(roster, slots={"FLEX": ()})


def test_optimizer_rejects_empty_player_eligibility():
    roster = [
        player("rb1", "RB", 12, eligible_positions=[])
    ]

    with pytest.raises(ValueError, match="eligible_positions cannot be empty"):
        optimize_lineup(roster, slots={"FLEX": ("RB", "WR", "TE")})

from fantasy_ml.decisions.sleeper_roster import (
    sleeper_roster_positions_to_slots,
)


def test_sleeper_standard_roster_slots():
    roster_positions = [
        "QB", "RB", "RB", "WR", "WR", "TE",
        "FLEX", "K", "DEF", "BN", "BN",
    ]

    slots = sleeper_roster_positions_to_slots(roster_positions)

    assert len(slots) == 9
    assert slots["RB1"] == ("RB",)
    assert slots["RB2"] == ("RB",)
    assert slots["FLEX"] == ("RB", "WR", "TE")
    assert "BN" not in slots


def test_sleeper_superflex_roster_slots():
    slots = sleeper_roster_positions_to_slots(
        ["QB", "RB", "WR", "TE", "SUPER_FLEX", "BN"]
    )

    assert slots["SUPER_FLEX"] == ("QB", "RB", "WR", "TE")
    assert len(slots) == 5


def test_sleeper_special_flex_slots():
    slots = sleeper_roster_positions_to_slots(
        ["WRRB_FLEX", "REC_FLEX"]
    )

    assert slots["WRRB_FLEX"] == ("RB", "WR")
    assert slots["REC_FLEX"] == ("WR", "TE")


def test_sleeper_rejects_unknown_roster_slot():
    with pytest.raises(ValueError, match="Unsupported Sleeper roster slot"):
        sleeper_roster_positions_to_slots(["QB", "UNKNOWN_SLOT"])


def test_sleeper_rejects_roster_without_starters():
    with pytest.raises(ValueError, match="no starter slots"):
        sleeper_roster_positions_to_slots(["BN", "IR"])