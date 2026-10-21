"""ffa.py — Free-For-All targeting helpers (4-player FFA, 1 active per side).

Relative target locations as the sim resolves them: a side's own loc is
negative, foes are positive, and the far pair of sides sits at ±2. These
locs are the choice suffix of `move <i> <loc>` for targeted moves.
"""

SELF_LOC: dict[int, int] = {0: -1, 1: -1, 2: -2, 3: -2}

FOE_LOCS: dict[int, dict[int, int]] = {
    0: {1: 1, 2: -2, 3: 2},
    1: {0: 1, 2: 2, 3: -2},
    2: {0: -1, 1: 1, 3: 2},
    3: {0: 1, 1: -1, 2: 2},
}


def target_loc(my_n: int, foe_n: int) -> int:
    if my_n not in FOE_LOCS or foe_n not in FOE_LOCS[my_n]:
        raise ValueError(f"no FFA target loc for side {my_n} -> side {foe_n}")
    return FOE_LOCS[my_n][foe_n]


TARGETED_MOVE_TARGETS: frozenset[str] = frozenset({"normal", "any", "adjacentAlly"})


def needs_target(move_target: str) -> bool:
    return move_target in TARGETED_MOVE_TARGETS


SPREAD_MODIFIER = 0.5


if __name__ == "__main__":
    expected_foes = {
        0: {1: 1, 2: -2, 3: 2},
        1: {0: 1, 2: 2, 3: -2},
        2: {0: -1, 1: 1, 3: 2},
        3: {0: 1, 1: -1, 2: 2},
    }
    assert FOE_LOCS == expected_foes
    assert SELF_LOC == {0: -1, 1: -1, 2: -2, 3: -2}
    for my_n, foes in expected_foes.items():
        locs = list(foes.values())
        assert all(abs(loc) in (1, 2) for loc in locs)
        assert SELF_LOC[my_n] not in locs
        assert len(set(locs)) == 3
        for foe_n, loc in foes.items():
            assert target_loc(my_n, foe_n) == loc
    for bad in ((-1, 0), (4, 0), (0, -1), (0, 4), (0, 0), (3, 3)):
        try:
            target_loc(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target_loc{bad} should raise ValueError")
    for t in ("normal", "any", "adjacentAlly"):
        assert needs_target(t)
    for t in ("self", "all", "allySide", "foeSide", "allAdjacent",
              "allAdjacentFoes", "randomNormal", "adjacentAllyOrSelf", "scripted"):
        assert not needs_target(t)
    assert SPREAD_MODIFIER == 0.5
    print("sb.ffa OK: 12 foe locs, 4 self locs, targeted={normal,any,adjacentAlly}, spread=0.5")
