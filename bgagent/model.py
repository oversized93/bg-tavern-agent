"""Entity-transformer policy for TavernGym, modeled on the paper's TavernAgent.

Each entity becomes one token: a global token (hero, gold, tier, quests, pending choice and the
pooled context cards), 7 warband minions, 10 hand cards, 8 Tavern offers, 16 pending-choice
options and 7 opponents (public stats + hero + the warband last seen against them, pooled).
A pointer-style head scores every one of the 245 actions from the token it is about, so the
action for "buy shop slot 3" is read off the shop-3 token, "magnetize hand 2 onto minion 5" off
a bilinear score between those two tokens, and so on. Heads: policy logits, value (expected
placement reward in [-1, 1]) and an auxiliary 8-way placement classifier.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import layout as L

# TavernGym encodes stats linearly (attack / 10 and so on) with no ceiling, and rollouts are
# stored as float16, whose largest value is 65504. Inputs are clamped to that, so a minion past
# 655,040 attack is seen the same way when acting and when training (and never becomes inf).
FEATURE_MAX = 65504.0


# Action types for the factored head: contiguous ranges of the 245 actions.
ACTION_GROUPS = [
    (L.A_END, L.A_END + 1), (L.A_REFRESH, L.A_REFRESH + 1), (L.A_FREEZE, L.A_FREEZE + 1),
    (L.A_UPGRADE, L.A_UPGRADE + 1), (L.A_HERO_POWER, L.A_DARK_GIFT),
    (L.A_DARK_GIFT, L.A_CANCEL), (L.A_CANCEL, L.A_BUY), (L.A_BUY, L.A_SELL),
    (L.A_SELL, L.A_ACTIVATE), (L.A_ACTIVATE, L.A_CHOOSE), (L.A_CHOOSE, L.A_PLAY),
    (L.A_PLAY, L.A_MAGNETIZE), (L.A_MAGNETIZE, L.A_MOVE), (L.A_MOVE, L.N_ACTIONS),
]
assert ACTION_GROUPS[0][0] == 0 and ACTION_GROUPS[-1][1] == L.N_ACTIONS
assert all(a[1] == b[0] for a, b in zip(ACTION_GROUPS, ACTION_GROUPS[1:]))


@dataclass
class ModelConfig:
    d: int = 384
    layers: int = 8
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    # Factored policy: P(action type) x P(target | type). Without it, the 42 legal minion
    # moves on a full board swamp every other choice and the policy learns to shuffle.
    # Checkpoints saved before this option existed load with factored=False.
    factored: bool = True
    # Action filter rules the policy is trained and used with (see filters.py).
    move_cap: int = 6
    no_refreeze: bool = True
    cancel_cap: int = 2
    # Auxiliary head predicting the combat right after this turn: (lost it, damage taken).
    # A label every turn instead of one per game; training only (see ppo.combat_labels).
    combat_head: bool = False

    @staticmethod
    def preset(name: str) -> "ModelConfig":
        return {
            "tiny": ModelConfig(d=64, layers=2, heads=4),  # CPU smoke tests
            "small": ModelConfig(d=192, layers=4, heads=8),  # ~3.5M, the paper's small model
            "medium": ModelConfig(d=256, layers=6, heads=8),  # ~8M
            "full": ModelConfig(d=384, layers=8, heads=8),  # ~18M, the paper's final size
        }[name]


class TavernNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.card = nn.Embedding(L.N_CARDS + 1, d, padding_idx=0)
        self.ctx_pos = nn.Parameter(torch.zeros(L.N_GLOBAL_CARDS, d))
        self.global_in = nn.Linear(L.GLOBAL_TOKEN_F, d)
        self.minion_in = nn.Linear(L.MINION_F, d)
        self.item_in = nn.Linear(L.ITEM_F, d)
        self.opp_in = nn.Linear(L.OPP_F, d)
        self.opp_board_proj = nn.Linear(d, d)
        self.type_emb = nn.Parameter(torch.zeros(L.N_TOKENS, d))  # type + slot, one per token
        layer = nn.TransformerEncoderLayer(
            d, cfg.heads, d * cfg.ff_mult, cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.trunk = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        # pointer heads
        self.h_global = nn.Linear(d, L.N_GLOBAL_ACTIONS)
        self.h_buy = nn.Linear(d, 1)
        self.h_sell = nn.Linear(d, 1)
        self.h_activate = nn.Linear(d, 1)
        self.h_choose = nn.Linear(d, 1)
        self.h_play = nn.Linear(d, L.N_POS)
        self.h_move = nn.Linear(d, L.N_BOARD)
        self.mag_q = nn.Linear(d, d)
        self.mag_k = nn.Linear(d, d)
        if cfg.factored:
            self.h_type = nn.Linear(2 * d, len(ACTION_GROUPS))
            nn.init.normal_(self.h_type.weight, std=0.01)
            nn.init.zeros_(self.h_type.bias)
        # value + auxiliary placement from the global token and a mean over all tokens
        self.v = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.place = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 8))
        if cfg.combat_head:
            self.combat = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)
        for h in (self.h_global, self.h_buy, self.h_sell, self.h_activate, self.h_choose,
                  self.h_play, self.h_move):
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)

    def tokens(self, feats: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        feats = feats.clamp(-FEATURE_MAX, FEATURE_MAX)  # same bound as rollout storage
        cards = cards.long()
        emb = self.card(cards)  # [B, 124, d]

        def fblock(name, n, w):
            a, b = L.F[name]
            return feats[:, a:b].reshape(B, n, w)

        def cblock(name):
            a, b = L.C[name]
            return emb[:, a:b]

        # global token: scalar state + mean of the present context cards
        ctx = torch.cat([emb[:, a:b] for a, b in L.GLOBAL_CARD_SPANS], 1)
        ctx_ids = torch.cat([cards[:, a:b] for a, b in L.GLOBAL_CARD_SPANS], 1)
        present = (ctx_ids > 0).unsqueeze(-1).to(ctx.dtype)
        ctx = ((ctx + self.ctx_pos) * present).sum(1) / present.sum(1).clamp(min=1.0)
        g = self.global_in(feats[:, : L.GLOBAL_TOKEN_F]) + ctx

        board = self.minion_in(fblock("board", L.N_BOARD, L.MINION_F)) + cblock("board")
        hand = self.item_in(fblock("hand", L.N_HAND, L.ITEM_F)) + cblock("hand")
        shop = self.item_in(fblock("shop", L.N_SHOP, L.ITEM_F)) + cblock("shop")
        choice = self.item_in(fblock("choice", L.N_CHOICE, L.ITEM_F)) + cblock("choice")

        # opponents: public stats + hero + pooled last-seen warband
        ob_f = fblock("opp_board", L.N_OPP * L.N_BOARD, L.MINION_F)
        ob = self.minion_in(ob_f) + cblock("opp_board")
        ob = ob.reshape(B, L.N_OPP, L.N_BOARD, -1)
        ob_present = ob_f[..., 0].reshape(B, L.N_OPP, L.N_BOARD, 1)
        ob = (ob * ob_present).sum(2) / ob_present.sum(2).clamp(min=1.0)
        opp = self.opp_in(fblock("opp", L.N_OPP, L.OPP_F)) + cblock("opp_hero")
        opp = opp + self.opp_board_proj(ob)

        x = torch.cat([g.unsqueeze(1), board, hand, shop, choice, opp], 1)
        return x + self.type_emb

    def forward(self, feats, cards, mask=None, combat=False):
        """Returns (logits [B, 245] with illegal actions at -inf if mask given, value [B],
        placement logits [B, 8]); with `combat`, also the combat head's [B, 2] (loss logit,
        damage / 10)."""
        h = self.norm(self.trunk(self.tokens(feats, cards)))
        s = lambda name: h[:, L.T[name][0] : L.T[name][1]]  # noqa: E731
        g, board, hand, shop, choice = h[:, 0], s("board"), s("hand"), s("shop"), s("choice")
        B = h.shape[0]
        mag = torch.einsum("bhd,bmd->bhm", self.mag_q(hand), self.mag_k(board))
        mag = mag / (self.cfg.d ** 0.5)
        logits = torch.cat(
            [
                self.h_global(g),  # 0..7
                self.h_buy(shop).squeeze(-1),  # 8..15
                self.h_sell(board).squeeze(-1),  # 16..22
                self.h_activate(board).squeeze(-1),  # 23..29
                self.h_choose(choice).squeeze(-1),  # 30..45
                self.h_play(hand).reshape(B, -1),  # 46..125  (hand * 8 + pos)
                mag.reshape(B, -1),  # 126..195 (hand * 7 + board)
                self.h_move(board).reshape(B, -1),  # 196..244 (from * 7 + to)
            ],
            1,
        ).float()
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        pooled = torch.cat([g, h.mean(1)], -1)
        if self.cfg.factored:
            logits = self.factor(logits, self.h_type(pooled).float())
        value = torch.tanh(self.v(pooled).squeeze(-1).float())
        place = self.place(pooled).float()
        if combat:
            return logits, value, place, self.combat(pooled).float()
        return logits, value, place

    @staticmethod
    def factor(target_logits, type_logits):
        """logit(a) = type(g) + log softmax of the targets within g, over legal targets only.
        Softmax over all 245 of these is exactly P(type) x P(target | type), with the type
        distribution renormalised over the types that have a legal action."""
        parts = []
        for k, (s, e) in enumerate(ACTION_GROUPS):
            x = target_logits[:, s:e]
            lse = torch.logsumexp(x, -1, keepdim=True)
            lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))  # no legal target
            parts.append(x - lse + type_logits[:, k : k + 1])
        return torch.cat(parts, 1)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
