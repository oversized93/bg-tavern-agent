"""Battlegrounds boards from real games: HSReplay XML (Firestone's replays) to library entries.

A replay is the game's whole entity log as seen by the player who recorded it (the "main"
player). Battlegrounds plays each turn as two game TURNs: an odd one for the Tavern, an even
one for the combat, so player turn k fights in TURN 2k. What we read at each combat:
- the main player's board as it enters the combat (its minions in play when TURN turns even),
  its Tavern Tier and hero;
- the opponent's board: the minions the game creates for the opponent's side (`opp`
  controller, zone PLAY) in the first block of that TURN that creates any, each with its
  combat-start Attack, Health and keywords (later blocks are summons during the fight);
- the opponent's hero, and its Tavern Tier from the hidden copy of that hero the game gives
  the main player for the leaderboard (zone SETASIDE, PLAYER_TECH_LEVEL).

Account names and ids in the replay are never read. Boards become library entries
(`bgagent.library`) by building them in a scratch TavernGym game with its debug commands, so
every field the engine needs is filled the engine's own way.
"""

import io
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hearthstone.enums import CardType, GameTag as T, Zone
from lxml import etree

# Hearthstone keyword tags -> TavernGym keyword names (taverngym::types::Kw::NAMES).
KEYWORD_TAGS = (
    (T.TAUNT, "TAUNT"),
    (T.DIVINE_SHIELD, "DIVINE_SHIELD"),
    (T.REBORN, "REBORN"),
    (T.VENOMOUS, "VENOMOUS"),
    (T.POISONOUS, "POISONOUS"),
    (T.STEALTH, "STEALTH"),
)
MEGA_WINDFURY = 3  # WINDFURY tag value of "attacks four times"


@dataclass
class Minion:
    card: str
    atk: int
    hp: int
    golden: bool
    keywords: list


@dataclass
class Board:
    turn: int  # player turn (1 = the first Tavern turn)
    side: str  # "main" or "opponent"
    hero: str
    tier: int  # 0 = unknown
    minions: list = field(default_factory=list)

    def strength(self):
        return sum(m.atk + m.hp for m in self.minions)


@dataclass
class Game:
    build: int
    main_hero: str
    turns: int
    boards: list  # [Board], in order

    def to_json(self):
        return {"build": self.build, "main_hero": self.main_hero, "turns": self.turns,
                "boards": [asdict(b) for b in self.boards]}


def _minion(card, tags):
    kw = [name for tag, name in KEYWORD_TAGS if tags.get(tag)]
    wf = tags.get(T.WINDFURY, 0)
    if wf:
        kw.append("MEGA_WINDFURY" if wf >= MEGA_WINDFURY else "WINDFURY")
    return Minion(card=card, atk=int(tags.get(T.ATK, 0)),
                  hp=int(tags.get(T.HEALTH, 0)) - int(tags.get(T.DAMAGE, 0)),
                  golden=bool(tags.get(T.PREMIUM, 0)) or card.endswith("_G"), keywords=kw)


class _Reader:
    """Replays the XML's entity log and collects one main and one opponent board per combat."""

    def __init__(self):
        self.ents = {}  # entity id -> [card id, {tag: value}]
        self.game_id = None
        self.main_ctrl = self.opp_ctrl = None
        self.main_player = None  # entity id of the main Player
        self.build = 0
        self.turn = 0
        self.depth = 0  # Block nesting
        self.boards = []
        self.opp = None  # the combat's opponent Board while it is being created
        self.opp_tiers = {}  # hero card -> Tier, from this combat's leaderboard hero copies
        self.opp_eids = []  # entity ids of the opponent's board as the game creates it
        self.opp_block = None  # top-level block number the opponent's board is created in
        self.opp_closed = True
        self.block_no = 0  # top-level blocks seen

    # ---- entity log
    def _tags(self, el):
        return {int(t.get("tag")): int(t.get("value")) for t in el if t.tag == "Tag"}

    def full_entity(self, el):
        eid = int(el.get("id"))
        card = el.get("cardID", "")
        tags = self._tags(el)
        self.ents[eid] = [card, tags]
        if self.turn % 2 == 0 and self.turn > 0:
            self._combat_created(eid, card, tags)

    def show_entity(self, el):
        eid = int(el.get("entity"))
        e = self.ents.setdefault(eid, ["", {}])
        e[0] = el.get("cardID", e[0])
        e[1].update(self._tags(el))

    def tag_change(self, eid, tag, value):
        self.ents.setdefault(eid, ["", {}])[1][tag] = value
        if eid == self.game_id and tag == T.TURN and value != self.turn:
            self.turn = value
            if value % 2 == 0:
                self._combat_starts()

    # ---- combats
    def _main_board(self):
        rows = []
        for card, tags in self.ents.values():
            if (tags.get(T.CONTROLLER) == self.main_ctrl and tags.get(T.ZONE) == Zone.PLAY
                    and tags.get(T.CARDTYPE) == CardType.MINION):
                rows.append((tags.get(T.ZONE_POSITION, 0), _minion(card, tags)))
        return [m for _, m in sorted(rows, key=lambda r: r[0])]

    def _combat_starts(self):
        k = self.turn // 2
        p = self.ents.get(self.main_player, ["", {}])[1]
        hero = self.ents.get(p.get(T.HERO_ENTITY), [""])[0]
        self.boards.append(Board(turn=k, side="main", hero=hero,
                                 tier=int(p.get(T.PLAYER_TECH_LEVEL, 0)),
                                 minions=self._main_board()))
        self._resolve_tier()
        self.opp = Board(turn=k, side="opponent", hero="", tier=0)
        self.boards.append(self.opp)
        self.opp_tiers = {}
        self.opp_eids = []
        self.opp_block = None
        self.opp_closed = False

    def _resolve_tier(self):
        if self.opp is not None and not self.opp.tier:
            self.opp.tier = self.opp_tiers.get(self.opp.hero, 0)

    def _combat_created(self, eid, card, tags):
        if self.opp is None:
            return
        kind, zone, ctrl = tags.get(T.CARDTYPE), tags.get(T.ZONE), tags.get(T.CONTROLLER)
        if kind == CardType.HERO:
            if ctrl == self.opp_ctrl and zone == Zone.PLAY and not self.opp.hero:
                self.opp.hero = card
            elif ctrl == self.main_ctrl and zone == Zone.SETASIDE and T.PLAYER_TECH_LEVEL in tags:
                # the leaderboard copy of this combat's opponent carries its Tier; matched to
                # the opponent's hero once both are known (either can come first)
                self.opp_tiers[card] = int(tags[T.PLAYER_TECH_LEVEL])
        elif kind == CardType.MINION and ctrl == self.opp_ctrl and zone == Zone.PLAY:
            if self.opp_closed or self.depth == 0:
                return
            if self.opp_block is None:
                self.opp_block = self.block_no
            if self.block_no == self.opp_block and len(self.opp_eids) < 7:
                self.opp_eids.append(eid)

    def block_start(self):
        if self.depth == 0:
            self.block_no += 1
        self.depth += 1

    def block_end(self):
        self.depth -= 1
        if self.depth == 0 and self.opp_block is not None and self.block_no == self.opp_block \
                and not self.opp_closed:
            # A minion is created with its card's base stats; the same block then sets its real
            # Attack, Health and keywords. Read them once the block is done.
            self.opp_closed = True
            rows = [(self.ents[e][1].get(T.ZONE_POSITION, 0), _minion(*self.ents[e]))
                    for e in self.opp_eids]
            self.opp.minions = [m for _, m in sorted(rows, key=lambda r: r[0])]

    # ---- driver
    def read(self, source):
        for ev, el in etree.iterparse(source, events=("start", "end")):
            tag = el.tag
            if tag == "Block":
                if ev == "start":
                    self.block_start()
                else:
                    self.block_end()
                    el.clear()
                continue
            if ev == "start":
                if tag == "Game":
                    self.build = int(el.get("buildNumber") or 0)
                continue
            if tag == "FullEntity":
                self.full_entity(el)
                el.clear()
            elif tag == "TagChange":
                self.tag_change(int(el.get("entity")), int(el.get("tag")), int(el.get("value")))
                el.clear()
            elif tag in ("ShowEntity", "ChangeEntity"):
                self.show_entity(el)
                el.clear()
            elif tag == "GameEntity":
                self.game_id = int(el.get("id"))
                self.ents[self.game_id] = ["", self._tags(el)]
                el.clear()
            elif tag == "Player":
                eid = int(el.get("id"))
                tags = self._tags(el)
                self.ents[eid] = ["", tags]
                if el.get("isMainPlayer") == "true":
                    self.main_player, self.main_ctrl = eid, int(el.get("playerID"))
                else:
                    self.opp_ctrl = int(el.get("playerID"))
                el.clear()
        self._resolve_tier()
        # an opponent board that never appeared (e.g. the game ended first) is dropped
        boards = [b for b in self.boards if b.side == "main" or b.minions or b.hero]
        main = next((b.hero for b in reversed(boards) if b.side == "main" and b.hero), "")
        return Game(build=self.build, main_hero=main, turns=self.turn // 2, boards=boards)


def read_replay(path_or_bytes):
    """A Game from an HSReplay XML file, a Firestone .xml.zip, or their bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        data = Path(path_or_bytes).read_bytes()
    if data[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(data))
        data = z.read(z.namelist()[0])
    # Player names are never used, and some aren't escaped (name=""Lich King" Deryl"), which
    # breaks the XML: blank them before parsing.
    data = _PLAYER_NAME.sub(rb'\1 name="" \2', data)
    return _Reader().read(io.BytesIO(data))


_PLAYER_NAME = re.compile(rb'(<Player\b[^>]*?) name="[^>]*?" (isMainPlayer=)')


# ---------------------------------------------------------------- to TavernGym library entries

class EntryBuilder:
    """Builds library entries (`library.make_entry` format) from boards, in a scratch
    TavernGym game: each minion is created by the engine (`add_minion`), then given the
    replay's Attack, Health and keywords. Cards TavernGym doesn't know are skipped and
    counted in `unknown`."""

    def __init__(self, seed=0):
        import json

        import taverngym

        self._json = json
        self.game = taverngym.Game(seed)
        names = ("TAUNT", "DIVINE_SHIELD", "WINDFURY", "REBORN", "VENOMOUS", "POISONOUS",
                 "STEALTH", "MAGNETIC", "MEGA_WINDFURY", "CLEAVE", "IMMUNE", "IMMUNE_ATTACKING")
        self.kw_bit = {n: 1 << i for i, n in enumerate(names)}
        # keywords a replay's tags can carry; the rest (MAGNETIC, CLEAVE, IMMUNE...) stay as
        # the card has them
        self.tag_mask = sum(self.kw_bit[n] for _, n in KEYWORD_TAGS) | self.kw_bit["WINDFURY"] \
            | self.kw_bit["MEGA_WINDFURY"]
        self.unknown = {}

    def _cmd(self, **kw):
        return self.game.debug(self._json.dumps(kw))

    def _add(self, m):
        tries = [(m.card, False)] if not m.golden else []
        if m.golden:
            base = m.card[:-2] if m.card.endswith("_G") else m.card
            tries = [(base, True), (m.card, False)]
        for card, golden in tries:
            try:
                self._cmd(cmd="add_minion", p=0, zone="board", card=card, golden=golden,
                          atk=m.atk, hp=max(1, m.hp))
                return True
            except Exception:
                continue
        self.unknown[m.card] = self.unknown.get(m.card, 0) + 1
        return False

    def entry(self, board):
        """The library entry for a Board, or None if none of its minions could be built."""
        from bgagent.library import board_strength

        self._cmd(cmd="clear_zone", p=0, zone="board")
        kept = [m for m in board.minions[:7] if self._add(m)]
        if board.minions and not kept:
            return None
        st = self._json.loads(self.game.state_json())
        tm = st["players"][0]["board"]
        for m, j in zip(kept, tm):
            kw = sum(self.kw_bit[n] for n in m.keywords if n in self.kw_bit)
            j["kw"] = (j["kw"] & ~self.tag_mask) | kw
            j["shield_hits"] = max(1, j.get("shield_max", 1)) if kw & self.kw_bit["DIVINE_SHIELD"] else 0
        return {"turn": int(board.turn), "strength": board_strength(tm),
                "tier": int(board.tier) if board.tier else _tier_guess(tm),
                "board": tm, "hand": [], "stats": {}, "counters": []}


def _tier_guess(board):
    """Tier when the replay doesn't say: at least the highest tier on the board."""
    return max([int(m.get("tier", 1)) for m in board] + [1])
