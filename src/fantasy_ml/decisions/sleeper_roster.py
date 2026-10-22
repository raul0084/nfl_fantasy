"""Convert Sleeper league roster positions into optimizer slots."""

from collections import Counter
from collections.abc import Sequence

SlotMap = dict[str, tuple[str, ...]]

SLOT_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "K": ("K",),
    "DEF": ("DEF",),
    "FLEX": ("RB", "WR", "TE"),
    "REC_FLEX": ("WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"),
    "SUPER_FLEX": ("QB", "RB", "WR", "TE"),
}

NON_STARTER_SLOTS = {"BN", "IR", "TAXI"}


def sleeper_roster_positions_to_slots(
    roster_positions: Sequence[str],
) -> SlotMap:
    """Convert Sleeper roster_positions into named optimizer slots.

    Bench, injured-reserve, and taxi slots are excluded.
    Unknown slot types raise ValueError rather than being silently ignored.
    """
    if isinstance(roster_positions, (str, bytes)) or not roster_positions:
        raise ValueError("roster_positions must be a non-empty sequence of slot names")

    positions = []
    for position in roster_positions:
        if not isinstance(position, str) or not position.strip():
            raise ValueError("Each roster position must be a non-empty string")
        positions.append(position.strip().upper())

    unknown = sorted(
        set(positions) - set(SLOT_ELIGIBILITY) - NON_STARTER_SLOTS
    )
    if unknown:
        raise ValueError(f"Unsupported Sleeper roster slot(s): {unknown}")

    starter_positions = [
        position for position in positions if position not in NON_STARTER_SLOTS
    ]
    if not starter_positions:
        raise ValueError("Roster configuration contains no starter slots")

    counts = Counter(starter_positions)
    seen = Counter()
    slots: SlotMap = {}

    for position in starter_positions:
        seen[position] += 1
        if counts[position] == 1:
            slot_name = position
        else:
            slot_name = f"{position}{seen[position]}"

        slots[slot_name] = SLOT_ELIGIBILITY[position]

    return slots
