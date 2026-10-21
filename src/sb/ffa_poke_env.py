"""ffa_poke_env.py — FFABattle: poke-env Battle subclass for 4-player free-for-alls.

poke-env 0.16 hardcodes two players: |title| unpacks "A vs. B", |player| flips
the opponent's role onto ours, and every non-us mon lands in one merged
_opponent_team. FFABattle intercepts title/player, routes each opponent mon
into a per-side OpponentView by ident prefix (p3a -> side 2), and tracks
elimination; install() monkeypatches Player._create_battle so formats whose id
contains "freeforall" construct FFABattle instances (mirrors the auth-patch
precedent in scripts/selfplay.py).
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from poke_env.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import STACKABLE_CONDITIONS, SideCondition
from poke_env.data import GenData
from poke_env.player import Player
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

_IDENT = re.compile(r"^(?:\[of\] )?p([1-4])([a-c])?: (.+)$")
_SPREAD_TOKEN = re.compile(r"^p([1-4])[a-c]?$")
_ROLE = re.compile(r"^p[1-4]$")
_ACTIVE_EVENTS = {"switch", "drag", "replace", "faint"}
_SIDE_EVENTS = {"-sidestart", "-sideend"}


class OpponentView:
    """One opponent side of a free-for-all: per-side team bucket + side state.

    team maps species name to the shared poke-env Pokemon object (the same
    instance the merged battle state holds). preview_species records |poke|
    teampreview reveals WITHOUT creating team entries — mirrors
    showdown_state.py, where unrevealed mons stay out of the team dict."""

    def __init__(self, n: int, username: str = ""):
        self.n = n
        self.username = username
        self.team: Dict[str, Pokemon] = {}
        self.active: Optional[Pokemon] = None
        self.side_conditions: Dict[SideCondition, int] = {}
        self.tera_used = False
        self.eliminated = False
        self.preview_species: List[str] = []

    @property
    def pokemon_left(self) -> int:
        return sum(1 for m in self.team.values() if not m.fainted)


class FFABattle(Battle):
    """Battle with four sides (FFA). poke-env's singles machinery is reused for
    our own side and the merged opponent bucket; per-opponent truth lives in
    OpponentViews."""

    def __init__(self, battle_tag: str, username: str, logger, gen: int,
                 save_replays=False):
        super().__init__(battle_tag=battle_tag, username=username, logger=logger,
                         gen=gen, save_replays=save_replays)
        self.my_side_n: Optional[int] = None
        self.username_to_n: Dict[str, int] = {}
        self._views: Dict[int, OpponentView] = {}
        self.title_usernames: List[str] = []
        self.winner_username: Optional[str] = None

    @property
    def opponents(self) -> List[OpponentView]:
        """All opponent sides with any known state, sorted by side index."""
        return [v for n, v in sorted(self._views.items()) if n != self.my_side_n]

    @property
    def live_opponents(self) -> List[OpponentView]:
        return [v for v in self.opponents if not v.eliminated]

    def parse_message(self, split_message: List[str]):
        if len(split_message) > 1 and split_message[1] == "title":
            self._replay_data.append(split_message[:])
            self.title_usernames = split_message[2].split(" vs. ")
            # the inherited players setter unpacks exactly two names
            return
        if len(split_message) > 1 and split_message[1] == "player":
            self._replay_data.append(split_message[:])
            self._parse_player(split_message)
            return
        try:
            super().parse_message(split_message)
        except Exception as e:
            # FFA battles emit events poke-env 0.16 never saw (room-level
            # noise, newer protocol); a parse must never take the battle down.
            if self.logger is not None:
                self.logger.warning("FFABattle dropped event %s: %r",
                                    split_message, e)
            return
        try:
            self._bucket(split_message)
        except Exception as e:
            if self.logger is not None:
                self.logger.warning("FFABattle bucketing failed for %s: %r",
                                    split_message, e)

    def parse_request(self, request: Dict[str, Any],
                      strict_battle_tracking: bool = False):
        self._last_request = request
        if request.get("wait") and "side" not in request:
            # eliminated FFA sides keep receiving wait-only requests, which the
            # inherited singles parser would KeyError on (no "side" key)
            self._wait = True
            return
        super().parse_request(request, strict_battle_tracking)

    def get_pokemon(self, identifier: str, force_self_team: bool = False,
                    details: str = "", request: Optional[Dict[str, Any]] = None
                    ) -> Pokemon:
        # two FFA hazards in the inherited method:
        # - the teamsize fullness guard compares the MERGED opponent bucket's
        #   length against ONE side's teamsize (always false-positive)
        # - the nickname-rename scans the whole merged bucket, so an ident whose
        #   key is absent steals another side's same-species mon
        # pre-creating absent opponent mons under their normalized key skips both
        team_size = self._team_size
        self._team_size = {}
        try:
            if (not force_self_team and identifier[:2] != self.player_role
                    and len(identifier) > 3):
                key = identifier
                if key[3] != " ":
                    key = key[:2] + key[3:]
                if key not in self._opponent_team and key not in self._team:
                    name = key[3:].strip()
                    if request:
                        mon = Pokemon(request_pokemon=request, name=name, gen=self.gen)
                    elif details:
                        mon = Pokemon(details=details, name=name, gen=self.gen)
                    else:
                        mon = Pokemon(species=key[4:], name=name, gen=self.gen)
                    self._opponent_team[key] = mon
            return super().get_pokemon(identifier, force_self_team, details, request)
        finally:
            self._team_size = team_size

    def won_by(self, player_name: str):
        self.winner_username = player_name
        super().won_by(player_name)

    def move_request_target(self, move_id: str) -> Optional[str]:
        """The 'target' field of move_id in the last move request
        (active[0].moves[]), None when unknown (e.g. replay parsing)."""
        active = (self._last_request or {}).get("active") or []
        if not active:
            return None
        for move in active[0].get("moves", []):
            if move.get("id") == move_id:
                return move.get("target")
        return None

    def _parse_player(self, split_message: List[str]):
        event = split_message
        if len(event) == 6:
            player, username, avatar, rating = event[2:6]
        elif len(event) == 5:
            player, username, avatar = event[2:5]
            rating = None
        elif len(event) == 4 and not event[-1]:
            return  # player left
        else:
            if self.logger is not None:
                self.logger.warning("FFABattle: odd player event %s", event)
            return
        if not _ROLE.match(player):
            if self.logger is not None:
                self.logger.warning("FFABattle: odd player role %s", event)
            return
        n = int(player[1]) - 1
        self.username_to_n[username] = n
        if username == self._player_username:
            self.my_side_n = n
            self._player_role = player
        else:
            view = self._views.get(n)
            if view is None:
                self._views[n] = OpponentView(n, username)
            else:
                view.username = username
        entry = {"username": username, "player": player, "avatar": avatar}
        if rating is not None:
            entry["rating"] = rating
        self._players.append(entry)

    def _bucket(self, split_message: List[str]):
        event = split_message[1]
        if event == "poke":
            self._bucket_poke(split_message)
            return
        if event in _SIDE_EVENTS:
            self._bucket_side(split_message)
        actor_side = None
        if event in _ACTIVE_EVENTS and len(split_message) > 2:
            m = _IDENT.match(split_message[2])
            if m:
                actor_side = int(m.group(1)) - 1
        for side_n, ident in self._idents(split_message):
            mon = self._find_mon(ident)
            if mon is None or side_n == self.my_side_n:
                continue
            view = self._view_for(side_n)
            if view.team.get(mon.species) is not mon:
                view.team[mon.species] = mon
            if side_n == actor_side:
                view.active = mon
            if event == "-terastallize":
                view.tera_used = True
            self._refresh(view)

    def _bucket_poke(self, split_message: List[str]):
        if len(split_message) < 4:
            return
        role = split_message[2]
        if not _ROLE.match(role) or role == self._player_role:
            return
        view = self._view_for(int(role[1]) - 1)
        species = split_message[3].split(", ")[0]
        if species not in view.preview_species:
            view.preview_species.append(species)

    def _bucket_side(self, split_message: List[str]):
        if len(split_message) < 4:
            return
        role = split_message[2].split(":")[0]
        if not _ROLE.match(role):
            return
        n = int(role[1]) - 1
        if n == self.my_side_n:
            return
        condition = SideCondition.from_showdown_message(split_message[3])
        if condition is SideCondition.UNKNOWN:
            return
        view = self._view_for(n)
        if split_message[1] == "-sidestart":
            if condition in STACKABLE_CONDITIONS:
                view.side_conditions[condition] = (
                    view.side_conditions.get(condition, 0) + 1)
            elif condition not in view.side_conditions:
                view.side_conditions[condition] = self.turn
        else:
            view.side_conditions.pop(condition, None)

    def _idents(self, split_message: List[str]) -> List[Tuple[int, Optional[str]]]:
        """(side_n, ident) pairs for every pokemon identifier in the message;
        ident is None for bare positional tokens ([spread] p2a)."""
        found = []
        for field in split_message[2:]:
            m = _IDENT.match(field)
            if m:
                found.append((int(m.group(1)) - 1, field))
            elif field.startswith("[spread] "):
                for token in field[len("[spread] "):].split(","):
                    mt = _SPREAD_TOKEN.match(token.strip())
                    if mt:
                        found.append((int(mt.group(1)) - 1, None))
        return found

    def _find_mon(self, ident: Optional[str]) -> Optional[Pokemon]:
        if ident is None:
            return None
        if len(ident) > 3 and ident[3] != " ":
            # strip the position letter exactly like poke-env's get_pokemon
            ident = ident[:2] + ident[3:]
        mon = self._team.get(ident)
        if mon is None:
            mon = self._opponent_team.get(ident)
        return mon

    def _view_for(self, n: int) -> OpponentView:
        view = self._views.get(n)
        if view is None:
            view = self._views[n] = OpponentView(n)
        if not view.username:
            view.username = next((u for u, k in self.username_to_n.items()
                                  if k == n and u != self._player_username), "")
        return view

    def _refresh(self, view: OpponentView):
        if not (view.team and view.pokemon_left == 0
                and (view.active is None or view.active.fainted)):
            return
        # zero known unfainted mons only means elimination once the side's
        # full roster is known — otherwise unrevealed mons may remain
        size = self._team_size.get(f"p{view.n + 1}")
        if size is None or len(view.team) >= size:
            view.eliminated = True


_ORIGINAL_CREATE_BATTLE = None


def install() -> None:
    """Monkeypatch Player._create_battle: freeforall formats build an FFABattle
    (same constructor args the original passes to Battle). Idempotent."""
    global _ORIGINAL_CREATE_BATTLE
    if _ORIGINAL_CREATE_BATTLE is not None:
        return
    _ORIGINAL_CREATE_BATTLE = Player._create_battle

    async def _create_battle(self, split_message: List[str]) -> Battle:
        battle_format = split_message[1]
        if "freeforall" not in battle_format:
            return await _ORIGINAL_CREATE_BATTLE(self, split_message)
        # Route on the battle's format tag, never self._format: the standing
        # queue can race battle-init, and poke-env's own gate raises
        # ShowdownException on any format the player didn't ask for.
        # body mirrors player.py (_create_battle) minus the doubles/vgc/OTS
        # branches, which random-FFA never hits
        battle_tag = "-".join(split_message)[1:]
        if battle_tag in self._battles:
            return self._battles[battle_tag]
        gen = GenData.from_format(battle_format).gen
        battle = FFABattle(
            battle_tag=battle_tag,
            username=self.username,
            logger=self.logger,
            gen=gen,
            save_replays=self._save_replays,
        )
        if isinstance(self._team, ConstantTeambuilder):
            battle.teampreview_team = [
                Pokemon(gen=gen, teambuilder=tb_mon) for tb_mon in self._team.team
            ]
        if battle_format.endswith("bo3"):
            self._battles[battle_tag] = battle
        else:
            await self._battle_count_queue.put(None)
            if battle_tag in self._battles:
                await self._battle_count_queue.get()
                return self._battles[battle_tag]
            async with self._battle_start_condition:
                self._battle_semaphore.release()
                self._battle_start_condition.notify_all()
                self._battles[battle_tag] = battle
        if self._start_timer_on_battle_start:
            await self.ps_client.send_message("/timer on", battle.battle_tag)
        return battle

    Player._create_battle = _create_battle
