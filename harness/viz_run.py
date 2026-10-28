"""Decision and network-internals panels for every step of a recorded run (for a video).

    viz_run.py REC.jsonl STATES.jsonl NETS.json OUTDIR [--loc ../sts2-cli]

One 1920x1080 PNG per recorded action (OUTDIR/step_0000.png ...) plus intro.png and
outro.png. The top-left 1280x720 is left for the game video. Right: the run state, the
decision (the unified policy's option probabilities outside combat; the planner's
first-action scores in combat) and the value head's estimate along the run. Bottom: the
transformer's attention (each layer, option tokens -> the tokens they attend to most) and
the token activation norms per layer.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bot"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for f in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAMES = None
BG, FG, DIM, ACC, ACC2 = "#14161a", "#e8e6e1", "#8a8f98", "#f2b134", "#5aa9e6"
W, H = 1920, 1080
ROOMS = {"Monster": "普通战斗", "Elite": "精英", "RestSite": "休息处", "Shop": "商店", "Unknown": "未知",
         "Treasure": "宝箱", "Boss": "Boss", "Event": "事件", "Ancient": "远古", "Map": "地图"}
DECISIONS = {"card_reward": "选卡奖励", "map_select": "选择路线", "rest_site": "休息处", "event_choice": "事件",
             "shop": "商店", "card_select": "选牌", "combat_play": "战斗", "bundle_select": "卡包"}


# ---------------------------------------------------------------- names

class Names:
    def __init__(self, loc_root):
        self.zh, self.en2id = {}, {}
        for f in glob.glob(os.path.join(loc_root, "localization_zhs", "*.json")):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, str):
                        self.zh.setdefault(k, v)
        for f in glob.glob(os.path.join(loc_root, "localization_eng", "*.json")):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, str) and (k.endswith(".title") or k.endswith(".name")):
                        self.en2id.setdefault(v.lower(), k.rsplit(".", 1)[0])

    def card(self, tok):
        tok = (tok or "").replace("CARD.", "")
        up = tok.endswith("+")
        base = tok.rstrip("+")
        return self.zh.get(base + ".title", base.title().replace("_", " ")) + ("+" if up else "")

    def thing(self, ident):
        ident = (ident or "").replace("RELIC.", "").replace("POTION.", "")
        if ident + ".title" in self.zh:
            return self.zh[ident + ".title"]
        k = self.en2id.get(ident.lower())
        return self.zh.get(k + ".title", ident) if k else ident

    def monster(self, name):
        k = self.en2id.get((name or "").lower())
        return self.zh.get(k + ".name", name) if k else name

    def event(self, desc):
        key = desc[0] or ""
        if key + ".title" in self.zh:
            return self.zh[key + ".title"]
        last = key.split(".")[-1]
        if last + ".title" in self.zh:
            return self.zh[last + ".title"]
        text = desc[2] if len(desc) > 2 else last
        return " ".join(str(text).split()[:3])


# ---------------------------------------------------------------- network with attention

def net_internals(net, T):
    """The unified network's forward pass, layer by layer: (logits, value, attention per layer
    [heads, L, L], token norms per stage [L])."""
    x, pad = net.embed(T)
    attns, norms = [], [x.norm(dim=-1)[0].numpy()]
    for layer in net.tf.layers:
        xa = layer.norm1(x)
        a, w = layer.self_attn(xa, xa, xa, key_padding_mask=pad, need_weights=True, average_attn_weights=False)
        x = x + a
        x = x + layer._ff_block(layer.norm2(x))
        attns.append(w[0].numpy())
        norms.append(x.norm(dim=-1)[0].numpy())
    if net.tf.norm is not None:
        x = net.tf.norm(x)
    z = net.option_scores(x, T)[0].numpy()
    v = float(net.value(x[:, 0]).squeeze(-1)[0])
    return z, v, attns, norms, pad[0].numpy()


# ---------------------------------------------------------------- drawing helpers

def fig_canvas():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    return fig


def ax_px(fig, x, y, w, h):
    """An axes at pixel box (x, y from the top-left)."""
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#3a3f47")
    ax.tick_params(colors=DIM, labelsize=9)
    return ax


def text_px(fig, x, y, s, size=14, color=FG, weight="normal", ha="left", va="top"):
    fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, weight=weight, ha=ha, va=va)


def wrap(label, width=13):
    if "（" in label and len(label) > width:
        a, b = label.split("（", 1)
        return a + "\n（" + b
    return label if len(label) <= width else label[:width] + "\n" + label[width:2 * width]


def bars(ax, labels, values, chosen, title, fmt="{:.0%}", color=ACC2, xmax=None):
    labels = [wrap(l) for l in labels]
    n = len(labels)
    y = np.arange(n)[::-1]
    cols = [ACC if i == chosen else color for i in range(n)]
    ax.barh(y, values, color=cols, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=FG)
    for yi, v, i in zip(y, values, range(n)):
        ax.text(v, yi, " " + fmt.format(v) + ("  ← 选择" if i == chosen else ""), va="center", fontsize=11,
                color=ACC if i == chosen else FG)
    ax.set_xlim(0, xmax if xmax is not None else max(1e-6, max(values)) * 1.45)
    ax.set_xticks([])
    if title:
        ax.figure.text(1300 / W, 1 - 140 / H, title, fontsize=14, color=FG, va="top")


# ---------------------------------------------------------------- per-step content

def option_labels(names, dec, opts):
    out = []
    for kind, o in opts:
        if kind == "card":
            out.append(names.card(o))
        elif kind == "skip":
            out.append("跳过")
        elif kind == "rest":
            a, _, c = o.partition(":")
            out.append({"HEAL": "休息（回血）", "SMITH": "锻造 " + names.card(c), "LIFT": "举重", "DIG": "挖掘",
                        "TOKE": "冥想"}.get(a, a))
        elif kind == "event":
            out.append(names.event(o))
        elif kind == "map":
            t, row, route = (list(o) + [None, None, None])[:3]
            counts = {}
            for r in route or []:
                counts[r] = counts.get(r, 0) + 1
            tail = "，".join(f"{ROOMS.get(r, r)}{c}" for r, c in sorted(counts.items(), key=lambda kv: -kv[1])[:3])
            out.append(f"{ROOMS.get(t, t)}（后续：{tail}）" if tail else ROOMS.get(t, t))
        else:   # shop line
            sk, name, cost = (list(o) + ["", "", 0])[:3]
            lab = {"remove": "删牌服务", "leave": "离开商店"}.get(sk)
            if lab is None:
                lab = names.card(name) if sk == "card" else names.thing(name)
            out.append(f"{lab}  {cost}金" if cost else lab)
    return out


def token_labels(names, s, pad):
    deck = [names.card(c) for c in (s.get("deck") or [])[:60]]
    rel = [names.thing(r) for r in (s.get("relics") or [])[:40]]
    pots = [names.thing(p) for p in (s.get("pots") or [])[:5]]
    labs = ["〈局面〉"] + deck + rel + pots
    return labs


def action_label(names, st, a, kw):
    if a == "end_turn":
        return "结束回合"
    if a == "play_card":
        card = next((c for c in st.get("hand") or [] if c.get("index") == kw.get("card_index")), None)
        lab = names.card((card or {}).get("id", "?"))
        if "target_index" in kw:
            en = next((e for e in st.get("enemies") or [] if e.get("index") == kw["target_index"]), None)
            if en:
                lab += f" → {NAMES.monster(en.get('name'))}"
        return lab
    if a == "use_potion":
        pots = (st.get("player") or {}).get("potions") or []
        p = pots[kw.get("potion_index", 0)] if kw.get("potion_index", 0) < len(pots) else {}
        return "药水：" + names.thing((p or {}).get("name", "?"))
    return a


def draw_header(fig, st, i, n, dec):
    ctx = st.get("context") or {}
    p = st.get("player") or {}
    text_px(fig, 1300, 18, f"杀戮尖塔 2 · 进阶 10 · 铁甲战士 · 种子 full_65", 13, DIM)
    text_px(fig, 1300, 44, f"第 {ctx.get('act')} 幕  第 {ctx.get('floor')} 层   {ROOMS.get(ctx.get('room_type'), ctx.get('room_type') or '')}",
            20, FG, "bold")
    deck = p.get("deck_size") or len(p.get("deck") or [])
    text_px(fig, 1300, 80, f"生命 {p.get('hp')}/{p.get('max_hp')}    金币 {p.get('gold')}    牌组 {deck} 张", 15, FG)
    text_px(fig, 1300, 108, f"第 {i + 1} / {n} 步 · {DECISIONS.get(dec, dec)}", 13, ACC)


def draw_value(fig, values, i):
    ax = ax_px(fig, 1330, 790, 560, 225)
    xs = sorted(values)
    ax.plot(xs, [values[k] for k in xs], color=ACC2, lw=1.6)
    cur = max([k for k in xs if k <= i], default=None)
    if cur is not None:
        ax.scatter([cur], [values[cur]], color=ACC, s=40, zorder=3)
    ax.set_xlim(0, max(xs[-1], 1) if xs else 1)
    ax.set_title("价值网络：对这局最终得分的估计（0–100）", color=FG, fontsize=13, loc="left")
    ax.set_xlabel("步", color=DIM, fontsize=10)


def draw_attention(fig, attns, norms, tok_labs, opt_labs, pad, chosen):
    L = len(pad)
    no = len(opt_labs)
    valid = [j for j in range(L - no) if not pad[j]]
    # columns: the tokens the options attend to most (summed over layers and options)
    score = np.zeros(L)
    for w in attns:
        score += w.mean(0)[L - no:, :].sum(0)
    cols = sorted(valid, key=lambda j: -score[j])[:12]
    cols.sort()
    labs = [tok_labs[j] if j < len(tok_labs) else "?" for j in cols]
    for li, w in enumerate(attns):
        ax = ax_px(fig, 140 + li * 300, 760, 270, 245)
        m = w.mean(0)[L - no:, :][:, cols]
        ax.imshow(m, aspect="auto", cmap="magma", vmin=0, vmax=max(1e-6, m.max()))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8, color=FG)
        ax.set_yticks(range(no))
        if li == 0:
            ax.set_yticklabels([("▶" if r == chosen else "") + s.split("（")[0][:7] for r, s in enumerate(opt_labs)],
                               fontsize=8, color=FG)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"第 {li + 1} 层注意力：选项 → 牌/遗物", color=FG, fontsize=10, loc="left")
    ax = ax_px(fig, 1100, 760, 160, 245)
    stages = ["嵌入"] + [f"第{k}层" for k in range(1, len(norms))]
    kinds = {"局面": [0], "牌组遗物": [j for j in valid if j > 0], "选项": list(range(L - no, L))}
    mat = np.array([[float(np.mean(nr[idx])) if idx else 0.0 for nr in norms] for idx in kinds.values()])
    mat = mat / np.maximum(mat.max(1, keepdims=True), 1e-6)
    ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9, color=FG)
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(list(kinds), fontsize=9, color=FG)
    ax.set_title("各层激活强度（按行归一）", color=FG, fontsize=10, loc="left")


def draw_combat(fig, st, row, plan_actions, pos, names):
    ax_txt = 175
    if row.get("x") and (row["x"].get("q") or row["x"].get("h")):
        q, h = row["x"].get("q") or {}, row["x"].get("h") or {}
        keys = sorted(set(q) | set(h), key=lambda k: (k not in q, -(q.get(k, -1e9)), -h.get(k, -1e9)))[:8]
        played = json.dumps([row["played"][0], row["played"][1] or {}], sort_keys=True)
        labs, hv, notes, chosen = [], [], [], -1
        for n_, k in enumerate(keys):
            a = json.loads(k)
            labs.append(action_label(names, st, a[0], a[1] or {}))
            hv.append(h.get(k, 0.0))
            notes.append(f"推演 {q[k]:.0f}" if k in q else "")
            if json.dumps([a[0], a[1] or {}], sort_keys=True) == played:
                chosen = n_
        lo = min(hv)
        shown = [v - lo + 0.08 * (max(hv) - lo + 1) for v in hv]
        ax = ax_px(fig, 1560, 160, 300, 300)
        bars(ax, labs, shown, chosen, "规划器：本回合第一步的候选动作", fmt="", color=ACC2, xmax=max(shown) * 1.9)
        for t in list(ax.texts):
            t.remove()
        y = np.arange(len(labs))[::-1]
        for yi, v, note, k in zip(y, shown, notes, range(len(labs))):
            ax.text(v, yi, f"  启发式 {hv[k]:.0f}" + (f" · {note}" if note else "") + ("  ← 选择" if k == chosen else ""),
                    va="center", fontsize=10, color=ACC if k == chosen else FG)
        text_px(fig, 1300, 470, "先用启发式评分给本回合所有出牌序列排序，\n再对前几名做多次随机推演（模拟未来几回合），\n选推演平均结果最好的序列。", 11, DIM)
    else:
        text_px(fig, 1300, 175, "执行规划好的本回合出牌序列：", 15, FG)
    y = 545
    text_px(fig, 1300, y, "本回合计划：", 13, DIM)
    for k, (a, kw, s2) in enumerate(plan_actions[:7]):
        mark = "▶ " if k == pos else "   "
        text_px(fig, 1300, y + 26 * (k + 1), mark + action_label(names, s2, a, kw), 13, ACC if k == pos else FG)
    ens = [e for e in st.get("enemies") or [] if (e.get("hp") or 0) > 0]
    text_px(fig, 30, 760, "战斗中由规划器决策：枚举本回合所有出牌顺序（最多 1000 个节点），对最好的几个结果做多次随机推演，选平均最好的。",
            13, DIM)
    text_px(fig, 30, 800, "敌人：" + "   ".join(f"{names.monster(e.get('name'))} {e.get('hp')}/{e.get('max_hp')}" for e in ens), 15, FG)
    hand = [names.card(c.get("id")) for c in st.get("hand") or []]
    text_px(fig, 30, 840, f"能量 {st.get('energy')}   手牌：" + "、".join(hand), 14, FG)
    p = st.get("player") or {}
    text_px(fig, 30, 880, f"玩家 生命 {p.get('hp')}/{p.get('max_hp')}  格挡 {p.get('block')}", 14, FG)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("states")
    ap.add_argument("nets")
    ap.add_argument("out")
    ap.add_argument("--loc", default=os.path.join(HERE, "..", "sts2-cli"))
    ap.add_argument("--seed", default="full_65")
    args = ap.parse_args()
    import torch
    from upolicy import UPolicy, collate, options_of
    from record import run_state
    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    global NAMES
    names = NAMES = Names(args.loc)
    rows = [r for r in (json.loads(l) for l in open(args.rec)) if r.get("seed") == args.seed and r.get("k") in ("d", "c")]
    steps = [json.loads(l) for l in open(args.states)]
    assert len(rows) == len(steps), (len(rows), len(steps))
    cfg = json.load(open(os.path.join(HERE, "..", "bot", args.nets)))
    up = UPolicy(cfg["unified"])
    up.net.eval()

    # pass 1: the network's view of every decision it made (and its value estimates)
    info, values = {}, {}
    for i, (r, s) in enumerate(zip(rows, steps)):
        if r["k"] != "d":
            continue
        opts = options_of(r)
        rs = r.get("s") or run_state(s["state"])
        route = (r.get("x") or {}).get("route")
        if opts and len(opts) >= 2:
            item = up.enc.encode(r["dec"], rs, opts, route)
            with torch.no_grad():
                T = collate([item], "cpu")
                z, v, attns, norms, pad = net_internals(up.net, T)
                z_ref, v_ref = up.net(T, with_value=True)
            assert np.allclose(z[:len(opts)], z_ref[0].numpy()[:len(opts)], atol=1e-4), "manual forward differs"
            p = np.exp(z[:len(opts)] - z[:len(opts)].max())
            p /= p.sum()
            info[i] = {"opts": opts, "p": p, "attns": attns, "norms": norms, "pad": pad, "rs": rs}
            values[i] = 100 * v
    # combat: group the actions of each plan (a row with x starts a plan)
    plan_of = {}
    cur = []
    for i, r in enumerate(rows):
        if r["k"] == "c" and (r.get("x") or i == 0 or rows[i - 1]["k"] != "c"):
            cur = []
        if r["k"] == "c":
            cur.append(i)
            for j in cur:
                plan_of[j] = cur
    n = len(steps)
    last_net = None
    for i, (r, s) in enumerate(zip(rows, steps)):
        st, (a, kw) = s["state"], s["action"]
        d = st.get("decision")
        fig = fig_canvas()
        text_px(fig, 640, 360, "（游戏画面）", 20, "#2a2e35", ha="center", va="center")
        draw_header(fig, st, i, n, d)
        if i in info:
            last_net = info[i]
            labs = option_labels(names, r["dec"], info[i]["opts"])
            chosen = r.get("a", -1) if r["dec"] == "card_reward" else (r.get("x") or {}).get(
                {"rest_site": "rk", "event_choice": "ek", "map_select": "mk", "shop": "sk"}.get(r["dec"]), -1)
            if r["dec"] == "rest_site":
                ks = (r.get("x") or {}).get("rkeys") or []
                chosen = ks.index((r.get("x") or {}).get("rk")) if (r.get("x") or {}).get("rk") in ks else -1
            ax = ax_px(fig, 1600, 160, 300, 560)
            bars(ax, labs, list(info[i]["p"]), chosen, f"统一策略网络：各选项的概率（{DECISIONS.get(r['dec'])}）", xmax=1.75)
            text_px(fig, 1300, 730, "一个 Transformer 同时负责选卡、路线、休息、事件、商店：\n输入局面、牌组、遗物、药水和每个选项，输出每个选项的分数。", 11, DIM)
            tok = token_labels(names, info[i]["rs"], info[i]["pad"])
            draw_attention(fig, info[i]["attns"], info[i]["norms"], tok, labs, info[i]["pad"], chosen)
        elif d == "combat_play" or (d == "card_select" and r["k"] == "c"):
            plan = plan_of.get(i, [i])
            acts = [(steps[j]["action"][0], steps[j]["action"][1], steps[j]["state"]) for j in plan]
            draw_combat(fig, st, r, acts, plan.index(i), names)
        else:
            text_px(fig, 1300, 175, f"{DECISIONS.get(d, d)}：按规则决定", 16, FG)
            cards = [names.card(c) for c in r.get("cards") or []]
            if a == "select_cards":
                idx = [int(x) for x in str(kw.get("indices", "")).split(",") if x != ""]
                text_px(fig, 1300, 215, "选中：" + "、".join(cards[k] for k in idx if k < len(cards)), 15, ACC)
            else:
                text_px(fig, 1300, 215, f"动作：{a}", 15, ACC)
            if last_net is not None:
                text_px(fig, 30, 760, "（这一步不由网络决定；下方保留上一次网络决策的内部状态）", 12, DIM)
        if values:
            draw_value(fig, values, i)
        fig.savefig(os.path.join(args.out, f"step_{i:04d}.png"), facecolor=BG)
        plt.close(fig)
        if i % 50 == 0:
            print("rendered", i, flush=True)
    for name, title, sub in (("intro", "AI 打通《杀戮尖塔 2》进阶 10", "铁甲战士 · 种子 full_65 · 统一 Transformer 策略（第 18 轮 RL）+ 搜索规划器\n右侧：每一步决策；下方：网络内部"),
                             ("outro", "胜利：进阶 10 双 Boss 通关", "最终生命 29/86 · 游戏记录 win = True")):
        fig = fig_canvas()
        text_px(fig, W / 2, 420, title, 46, FG, "bold", ha="center")
        text_px(fig, W / 2, 520, sub, 22, DIM, ha="center")
        fig.savefig(os.path.join(args.out, f"{name}.png"), facecolor=BG)
        plt.close(fig)
    json.dump({"steps": n, "values": values}, open(os.path.join(args.out, "meta.json"), "w"))
    print("done", n)


if __name__ == "__main__":
    main()
