"""Optimize fantasy football lineups from player projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
from scipy.optimize import linear_sum_assignment

# Default starter slots for a standard fantasy lineup.
DEFAULT_SLOTS: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB1": ("RB",),
    "RB2": ("RB",),
    "WR1": ("WR",),
    "WR2": ("WR",),
    "TE": ("TE",),
    "FLEX": ("RB", "WR", "TE"),
    "K": ("K",),
    "DEF": ("DEF",),
}


def optimize_lineup(
    players: Sequence[Mapping],
    slots: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Choose the highest-projected legal lineup.

    Each player must provide:
      - player_id: unique identifier
      - name: display name
      - position: primary position
      - projected_points: finite numeric projection

    Optional:
      - eligible_positions: positions the player can fill

    `slots` maps slot names to eligible positions. Players not selected
    as starters are returned as bench players.

    Raises ValueError for malformed inputs or if the roster cannot fill
    every requested slot.
    """
    if slots is None:
        slots = DEFAULT_SLOTS

    if not slots:
        raise ValueError("slots must not be empty")
    if not players:
        raise ValueError("players must not be empty")

    normalized = []
    seen_ids = set()

    for player in players:
        required = ("player_id", "name", "position", "projected_points")
        missing = [key for key in required if key not in player]
        if missing:
            raise ValueError(f"Player is missing required fields: {missing}")

        player_id = player["player_id"]
        if player_id in seen_ids:
            raise ValueError(f"Duplicate player_id: {player_id!r}")
        seen_ids.add(player_id)

        points = player["projected_points"]
        if (
            isinstance(points, bool)
            or not isinstance(points, Real)
            or not np.isfinite(points)
        ):
            raise ValueError(
                f"projected_points must be finite numeric value for {player_id!r}"
            )

        position = str(player["position"]).upper()
        eligible = player.get("eligible_positions", (position,))
        if isinstance(eligible, str):
            eligible = (eligible,)
        eligible = {str(pos).upper() for pos in eligible}
        if not eligible:
            raise ValueError(f"eligible_positions cannot be empty for {player_id!r}")

        normalized.append(
            {
                **dict(player),
                "position": position,
                "eligible_positions": eligible,
                "projected_points": float(points),
            }
        )

    slot_names = list(slots)
    slot_eligibility = {}
    for slot_name, positions in slots.items():
        if isinstance(positions, str):
            positions = (positions,)
        allowed = {str(pos).upper() for pos in positions}
        if not allowed:
            raise ValueError(f"Slot {slot_name!r} must allow at least one position")
        slot_eligibility[slot_name] = allowed

    if len(normalized) < len(slot_names):
        raise ValueError(
            f"Roster cannot fill every slot: "
            f"{len(normalized)} players for {len(slot_names)} slots"
        )

    # Rows are slots, columns are players. Invalid assignments receive
    # a large negative score; linear_sum_assignment maximizes total points.
    impossible = -1e12
    scores = np.full((len(slot_names), len(normalized)), impossible)

    for slot_index, slot_name in enumerate(slot_names):
        allowed = slot_eligibility[slot_name]
        for player_index, player in enumerate(normalized):
            if player["eligible_positions"] & allowed:
                scores[slot_index, player_index] = player["projected_points"]

    row_indices, col_indices = linear_sum_assignment(scores, maximize=True)

    if len(row_indices) != len(slot_names) or any(
        scores[row, col] == impossible
        for row, col in zip(row_indices, col_indices)
    ):
        raise ValueError(
            "Roster cannot fill every slot with eligible players. "
            "Check roster size and position eligibility."
        )

    starter_by_id = {}
    starters = []
    for row, col in zip(row_indices, col_indices):
        player = normalized[col]
        slot_name = slot_names[row]
        starter = {
            "slot": slot_name,
            "player_id": player["player_id"],
            "name": player["name"],
            "position": player["position"],
            "projected_points": player["projected_points"],
        }
        starters.append(starter)
        starter_by_id[player["player_id"]] = slot_name

    # Preserve the configured slot order in the result.
    starters.sort(key=lambda item: slot_names.index(item["slot"]))
    bench = [
        {
            "player_id": player["player_id"],
            "name": player["name"],
            "position": player["position"],
            "projected_points": player["projected_points"],
        }
        for player in normalized
        if player["player_id"] not in starter_by_id
    ]

    return {
        "starters": starters,
        "bench": bench,
        "projected_points": float(
            sum(player["projected_points"] for player in starters)
        ),
        "bench_projected_points": float(
            sum(player["projected_points"] for player in bench)
        ),
    }
