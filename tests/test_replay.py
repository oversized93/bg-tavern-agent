"""bgagent/replay.py on a hand-built replay with the shape of Firestone's: one Tavern turn,
one combat."""

from hearthstone.enums import CardType, GameTag as T, Zone

from bgagent.replay import Board, EntryBuilder, Minion, read_replay


def _tags(**kw):
    return "".join(f'<Tag tag="{int(getattr(T, k))}" value="{int(v)}" />' for k, v in kw.items())


def _full(eid, card, **kw):
    return f'<FullEntity id="{eid}" cardID="{card}">{_tags(**kw)}</FullEntity>'


def _change(eid, tag, value):
    return f'<TagChange entity="{eid}" tag="{int(getattr(T, tag))}" value="{value}" />'


MINION, HERO, PLAY, ASIDE = int(CardType.MINION), int(CardType.HERO), int(Zone.PLAY), int(Zone.SETASIDE)


def _replay(name='"Quoted" Name'):
    return f"""<?xml version="1.0" encoding="utf-8"?><HSReplay><Game ts="00:00:00.0" buildNumber="255790">
<GameEntity id="1">{_tags(TURN=0)}</GameEntity>
<Player id="2" accountHi="1" accountLo="2" playerID="1" name="{name}" isMainPlayer="true">{_tags(PLAYER_TECH_LEVEL=1, HERO_ENTITY=10)}</Player>
<Player id="3" accountHi="0" accountLo="0" playerID="9" name="Bartender Bob" isMainPlayer="false"></Player>
{_full(10, "BG21_HERO_010", CARDTYPE=HERO, ZONE=PLAY, CONTROLLER=1)}
{_change(1, "TURN", 1)}
<Block entity="3" type="5">
{_full(20, "BG29_816", CARDTYPE=MINION, ZONE=PLAY, CONTROLLER=1, ATK=2, HEALTH=8, ZONE_POSITION=2)}
{_full(21, "BG29_816", CARDTYPE=MINION, ZONE=PLAY, CONTROLLER=1, ATK=5, HEALTH=9, ZONE_POSITION=1, TAUNT=1)}
{_full(22, "BG29_816", CARDTYPE=MINION, ZONE=PLAY, CONTROLLER=9, ATK=2, HEALTH=8, ZONE_POSITION=1)}
</Block>
{_change(2, "PLAYER_TECH_LEVEL", 2)}
{_change(1, "TURN", 2)}
<Block entity="1" type="5">
{_full(40, "BG20_HERO_201", CARDTYPE=HERO, ZONE=ASIDE, CONTROLLER=1, PLAYER_TECH_LEVEL=3)}
{_full(41, "BG20_HERO_201", CARDTYPE=HERO, ZONE=PLAY, CONTROLLER=9)}
{_full(30, "BG29_816", CARDTYPE=MINION, ZONE=PLAY, CONTROLLER=9, ATK=2, HEALTH=8, ZONE_POSITION=2)}
{_change(30, "ATK", 7)}
{_full(31, "BG29_816_G", CARDTYPE=MINION, ZONE=PLAY, CONTROLLER=9, ATK=4, HEALTH=16, ZONE_POSITION=1, PREMIUM=1, DIVINE_SHIELD=1, WINDFURY=3)}
</Block>
<Block entity="30" type="1">
{_full(32, "BG29_816", CARDTYPE=MINION, ZONE=PLAY, CONTROLLER=9, ATK=1, HEALTH=1, ZONE_POSITION=3)}
{_change(21, "DAMAGE", 3)}
</Block>
{_change(1, "TURN", 3)}
</Game></HSReplay>""".encode()


def test_reads_both_boards_of_a_combat():
    g = read_replay(_replay())
    assert g.build == 255790 and g.turns == 1 and g.main_hero == "BG21_HERO_010"
    main, opp = g.boards
    # the winner's board as it entered the combat: in position order, Tier from that moment
    assert (main.side, main.turn, main.tier, main.hero) == ("main", 1, 2, "BG21_HERO_010")
    assert [(m.atk, m.hp, m.keywords) for m in main.minions] == [(5, 9, ["TAUNT"]), (2, 8, [])]
    # the opponent's: created in the combat's first block, with the stats that block set
    # (30: 2 -> 7 Attack), not the summon in the attack block; Tier from the hidden hero copy
    assert (opp.side, opp.turn, opp.tier, opp.hero) == ("opponent", 1, 3, "BG20_HERO_201")
    assert [(m.card, m.atk, m.hp, m.golden) for m in opp.minions] == [
        ("BG29_816_G", 4, 16, True), ("BG29_816", 7, 8, False)]
    assert opp.minions[0].keywords == ["DIVINE_SHIELD", "MEGA_WINDFURY"]
    assert opp.strength() == 4 + 16 + 7 + 8


def test_unescaped_player_names_are_blanked():
    # a nickname with quotes in it isn't valid XML; names are never needed
    assert read_replay(_replay('"Lich King" Deryl')).boards


def test_entry_builder_makes_engine_minions():
    eb = EntryBuilder()
    board = Board(turn=6, side="opponent", hero="x", tier=4, minions=[
        Minion("BG29_816_G", 30, 40, True, ["DIVINE_SHIELD", "TAUNT"]),
        Minion("BG29_816", 7, 8, False, []),
        Minion("NOT_A_CARD", 1, 1, False, [])])
    e = eb.entry(board)
    assert eb.unknown == {"NOT_A_CARD": 1}
    assert (e["turn"], e["tier"], e["strength"]) == (6, 4, 30 + 40 + 7 + 8)
    gold, plain = e["board"]
    assert gold["golden"] and (gold["atk"], gold["hp"], gold["max_hp"]) == (30, 40, 40)
    assert gold["kw"] & 0b11 == 0b11 and gold["shield_hits"] >= 1  # TAUNT | DIVINE_SHIELD
    assert not plain["golden"] and plain["kw"] & 0b11 == 0 and plain["shield_hits"] == 0
