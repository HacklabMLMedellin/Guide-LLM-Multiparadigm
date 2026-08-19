"""
GPT-Style Hybrid Architecture Animation  v2  (no text overlaps)
=================================================================
Fixes applied:
  • Stack shifted LEFT, annotations placed RIGHT with clipped width
  • Table redesigned: descriptions go BELOW icons in smaller font,
    columns wider, icon centred, text under it — no bleed across columns
  • Takeaway rows constrained to screen width with smaller font
  • Brace label kept separate from header
  • Header never overlaps top block (stack pushed down)

Render:
    manim -qh gpt_hybrid_v2.py GPTHybridAnimation
"""

from manim import *

# ─── Palette ──────────────────────────────────────────────────────────────────
BG = "#0D0D1A"
C_CLK = "#2A78D6"
C_QUA = "#8B5CF6"
C_THR = "#F59E0B"
C_TEXT = "#E8E8E0"
C_MUTED = "#5A5A65"
C_PANEL = "#13131F"

# ─── Module data ──────────────────────────────────────────────────────────────
MODULES = [
    # (name,                   q_label,                  t_label)
    ("API", None, None),
    ("Tokenizer", None, None),
    ("Embeddings", "Quantum embed.", "Analog embed."),
    ("Vector Search", "Quantum k-NN", "Assoc. mem."),
    ("Attention", "Exper. QPU", "Stochastic opt."),
    ("Matrix Multiply", "HHL-like algo.", "Analog crossbars"),
    ("Feed-Forward", "Quantum FF", "Analog MAC"),
    ("Sampling", "Quantum sampling", "Boltzmann samp."),
    ("Routing (MoE)", "QAOA routing", "Gibbs sampling"),
    ("Training", "Quantum optim.", "Energy-based opt."),
]

CLASSICAL_ONLY = {0, 1}  # indices with no quantum/thermic alternative

# ─── Layout ───────────────────────────────────────────────────────────────────
BLK_W = 3.6  # narrower blocks → room for annotations on right
BLK_H = 0.50
BLK_GAP = 0.165
STACK_X = -1.8  # shifted left
N = len(MODULES)
STACK_H = N * BLK_H + (N - 1) * BLK_GAP  # ≈ 6.49
STACK_Y = -STACK_H / 2 + BLK_H / 2  # centre vertically → bottom block at -3.0


# ─── Helpers ──────────────────────────────────────────────────────────────────
def make_block(label, color, w=BLK_W, h=BLK_H, fs=16):
    r = RoundedRectangle(
        corner_radius=0.10,
        width=w,
        height=h,
        fill_color=color,
        fill_opacity=0.20,
        stroke_color=color,
        stroke_width=2.0,
    )
    t = Text(label, font_size=fs, color=color, weight=BOLD).move_to(r)
    return VGroup(r, t)


def build_stack(colors):
    blocks = VGroup()
    for i, (name, _, _) in enumerate(MODULES):
        b = make_block(name, colors[i])
        b.move_to([STACK_X, STACK_Y + i * (BLK_H + BLK_GAP), 0])
        blocks.add(b)
    arrows = VGroup()
    for i in range(N - 1):
        a = Arrow(
            blocks[i].get_top(),
            blocks[i + 1].get_bottom(),
            buff=0.05,
            stroke_width=1.5,
            color=C_MUTED,
            max_tip_length_to_length_ratio=0.32,
        )
        arrows.add(a)
    return blocks, arrows


def paradigm_colors(mode):
    out = []
    for i, (_, q, t) in enumerate(MODULES):
        if mode == "classical":
            out.append(C_CLK)
        elif mode == "quantum":
            out.append(C_QUA if q else C_CLK)
        elif mode == "thermic":
            out.append(C_THR if t else C_CLK)
        elif mode == "hybrid":
            if i in CLASSICAL_ONLY:
                out.append(C_CLK)
            elif q and t:
                out.append(C_QUA if i % 2 == 0 else C_THR)
            elif q:
                out.append(C_QUA)
            else:
                out.append(C_CLK)
    return out


def legend_row(color, label, fs=17):
    d = Dot(radius=0.11, color=color)
    t = Text(label, font_size=fs, color=C_TEXT).next_to(d, RIGHT, buff=0.14)
    return VGroup(d, t)


def header(text, color=C_TEXT, fs=28):
    h = Text(text, font_size=fs, color=color, weight=BOLD)
    h.to_edge(UP, buff=0.30)
    return h


# ═══════════════════════════════════════════════════════════════════════════════
class GPTHybridAnimation(Scene):
    def construct(self):
        self.camera.background_color = BG
        self._title_card()
        blocks, arrows = self._section_classical()
        blocks, arrows = self._section_quantum(blocks, arrows)
        blocks, arrows = self._section_thermic(blocks, arrows)
        blocks, arrows = self._section_hybrid(blocks, arrows)
        self._section_table(blocks, arrows)
        self._section_takeaway()

    def _clear(self, *mobs, t=0.38):
        self.play(*[FadeOut(m) for m in mobs], run_time=t)

    # ── 0. Title ───────────────────────────────────────────────────────────────
    def _title_card(self):
        chips = (
            VGroup(
                Text("⬡", font_size=68, color=C_CLK),
                Text("⬡", font_size=68, color=C_QUA),
                Text("⬡", font_size=68, color=C_THR),
            )
            .arrange(RIGHT, buff=0.45)
            .shift(UP * 1.1)
        )

        t1 = Text(
            "ChatGPT-Style Hybrid Architecture", font_size=36, color=C_TEXT, weight=BOLD
        )
        t2 = Text(
            "Classical · Quantum · Thermodynamic",
            font_size=21,
            color=C_MUTED,
            slant=ITALIC,
        )
        VGroup(t1, t2).arrange(DOWN, buff=0.28).shift(DOWN * 0.65)

        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.25) for c in chips], lag_ratio=0.2),
            run_time=0.7,
        )
        self.play(Write(t1), run_time=0.6)
        self.play(FadeIn(t2, shift=UP * 0.1), run_time=0.4)
        self.wait(0.8)
        self._clear(chips, t1, t2)

    # ── 1. Classical ───────────────────────────────────────────────────────────
    def _section_classical(self):
        hdr = header("Classical GPT Architecture", color=C_CLK)
        sub = Text(
            "All modules run on classical GPU / CPU",
            font_size=18,
            color=C_MUTED,
            slant=ITALIC,
        )
        sub.to_edge(DOWN, buff=0.28)

        blocks, arrows = build_stack(paradigm_colors("classical"))

        self.play(Write(hdr), run_time=0.45)
        self.play(FadeIn(sub))
        for i, blk in enumerate(blocks):
            if i > 0:
                self.play(GrowArrow(arrows[i - 1]), run_time=0.07)
            self.play(FadeIn(blk, shift=UP * 0.10), run_time=0.13)

        self.wait(1.0)
        self._clear(sub, hdr, t=0.35)
        return blocks, arrows

    # ── 2. Quantum ─────────────────────────────────────────────────────────────
    def _section_quantum(self, old_blks, old_arrs):
        hdr_new = header("Quantum-Enhanced Modules", color=C_QUA)
        self.play(Write(hdr_new), run_time=0.45)

        new_blks, new_arrs = build_stack(paradigm_colors("quantum"))
        anims = [Transform(ob, nb) for ob, nb in zip(old_blks, new_blks)] + [
            Transform(oa, na) for oa, na in zip(old_arrs, new_arrs)
        ]
        self.play(LaggedStart(*anims, lag_ratio=0.07), run_time=1.4)

        # Annotations — RIGHT side, capped width, no overlap
        ann_x = STACK_X + BLK_W / 2 + 0.28  # left edge of annotation column
        ann_max_w = 6.8 - ann_x - 0.15  # stay inside screen
        anns = VGroup()
        for i, (_, q_lbl, _) in enumerate(MODULES):
            if q_lbl:
                blk_y = old_blks[i].get_center()[1]
                txt = Text(f"← {q_lbl}", font_size=13, color=C_QUA)
                txt.move_to([ann_x + txt.width / 2, blk_y, 0])
                # Clip if too wide
                if txt.get_right()[0] > 6.8:
                    txt.scale(ann_max_w / txt.width)
                    txt.move_to([ann_x + txt.width / 2, blk_y, 0])
                anns.add(txt)

        self.play(
            LaggedStart(*[FadeIn(a, shift=LEFT * 0.07) for a in anns], lag_ratio=0.09),
            run_time=0.8,
        )
        self.wait(1.2)
        self._clear(anns, hdr_new, t=0.3)
        return old_blks, old_arrs

    # ── 3. Thermic ─────────────────────────────────────────────────────────────
    def _section_thermic(self, old_blks, old_arrs):
        hdr = header("Thermodynamic-Enhanced Modules", color=C_THR)
        self.play(Write(hdr), run_time=0.4)

        new_blks, new_arrs = build_stack(paradigm_colors("thermic"))
        anims = [Transform(ob, nb) for ob, nb in zip(old_blks, new_blks)] + [
            Transform(oa, na) for oa, na in zip(old_arrs, new_arrs)
        ]
        self.play(LaggedStart(*anims, lag_ratio=0.07), run_time=1.4)

        ann_x = STACK_X + BLK_W / 2 + 0.28
        ann_max_w = 6.8 - ann_x - 0.15
        anns = VGroup()
        for i, (_, _, t_lbl) in enumerate(MODULES):
            if t_lbl:
                blk_y = old_blks[i].get_center()[1]
                txt = Text(f"← {t_lbl}", font_size=13, color=C_THR)
                txt.move_to([ann_x + txt.width / 2, blk_y, 0])
                if txt.get_right()[0] > 6.8:
                    txt.scale(ann_max_w / txt.width)
                    txt.move_to([ann_x + txt.width / 2, blk_y, 0])
                anns.add(txt)

        self.play(
            LaggedStart(*[FadeIn(a, shift=LEFT * 0.07) for a in anns], lag_ratio=0.09),
            run_time=0.8,
        )
        self.wait(1.2)
        self._clear(anns, hdr, t=0.3)
        return old_blks, old_arrs

    # ── 4. Hybrid ──────────────────────────────────────────────────────────────
    def _section_hybrid(self, old_blks, old_arrs):
        hdr = header("Full Hybrid Architecture", color=C_TEXT)
        self.play(Write(hdr), run_time=0.4)

        new_blks, new_arrs = build_stack(paradigm_colors("hybrid"))
        anims = [Transform(ob, nb) for ob, nb in zip(old_blks, new_blks)] + [
            Transform(oa, na) for oa, na in zip(old_arrs, new_arrs)
        ]
        self.play(LaggedStart(*anims, lag_ratio=0.06), run_time=1.3)

        # Legend RIGHT column, clear of stack
        leg = VGroup(
            legend_row(C_CLK, "Classical  (GPU/CPU)"),
            legend_row(C_QUA, "Quantum  (QPU)"),
            legend_row(C_THR, "Thermodynamic"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        leg.move_to([3.8, 0, 0])  # centred right side, never touches stack

        self.play(
            LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in leg], lag_ratio=0.2),
            run_time=0.6,
        )
        self.wait(1.4)
        self._clear(hdr, leg, t=0.35)
        return old_blks, old_arrs

    # ── 5. Table ───────────────────────────────────────────────────────────────
    def _section_table(self, old_blks, old_arrs):
        self._clear(old_blks, old_arrs, t=0.4)

        hdr = header("Module Capability Matrix", color=C_TEXT)
        self.play(Write(hdr), run_time=0.4)

        # Column layout — 4 equal columns filling full width
        # x centres: module name | classical | quantum | thermic
        COL = [-3.8, -0.5, 1.9, 4.3]  # x centres
        COL_W = [3.2, 1.2, 2.0, 2.0]  # max text width per column

        # Header row
        hdrs = VGroup(
            Text("Module", font_size=15, color=C_MUTED, weight=BOLD),
            Text("Classical", font_size=15, color=C_CLK, weight=BOLD),
            Text("Quantum", font_size=15, color=C_QUA, weight=BOLD),
            Text("Thermic", font_size=15, color=C_THR, weight=BOLD),
        )
        for lbl, cx in zip(hdrs, COL):
            lbl.move_to([cx, 2.90, 0])

        sep = Line([-6.4, 2.60, 0], [6.4, 2.60, 0], stroke_width=0.8, color=C_MUTED)

        self.play(
            LaggedStart(*[FadeIn(l) for l in hdrs], lag_ratio=0.12), run_time=0.45
        )
        self.play(Create(sep), run_time=0.25)

        ROW_H = 0.48
        Y_TOP = 2.30
        table = VGroup()

        for ri, (name, q_lbl, t_lbl) in enumerate(MODULES):
            y = Y_TOP - ri * ROW_H

            # Alternating row bg
            if ri % 2 == 1:
                bg = Rectangle(
                    width=13.0,
                    height=ROW_H,
                    fill_color=C_PANEL,
                    fill_opacity=0.65,
                    stroke_width=0,
                ).move_to([0, y, 0])
                table.add(bg)

            # Module name (col 0)
            nm = Text(name, font_size=13, color=C_TEXT)
            nm.move_to([COL[0], y, 0])
            table.add(nm)

            # Classical: always ✅ (col 1)
            table.add(Text("✅", font_size=14).move_to([COL[1], y, 0]))

            # Quantum (col 2) — icon + short description stacked vertically
            if q_lbl:
                ico = Text("⚠️", font_size=12)
                lbl_t = Text(q_lbl, font_size=9, color=C_QUA)
                cell = VGroup(ico, lbl_t).arrange(DOWN, buff=0.04)
                cell.move_to([COL[2], y, 0])
                # Scale down if wider than column
                if cell.width > COL_W[2]:
                    cell.scale(COL_W[2] / cell.width)
                    cell.move_to([COL[2], y, 0])
                table.add(cell)
            else:
                table.add(Text("❌", font_size=14).move_to([COL[2], y, 0]))

            # Thermic (col 3)
            if t_lbl:
                ico = Text("⚠️", font_size=12)
                lbl_t = Text(t_lbl, font_size=9, color=C_THR)
                cell = VGroup(ico, lbl_t).arrange(DOWN, buff=0.04)
                cell.move_to([COL[3], y, 0])
                if cell.width > COL_W[3]:
                    cell.scale(COL_W[3] / cell.width)
                    cell.move_to([COL[3], y, 0])
                table.add(cell)
            else:
                table.add(Text("❌", font_size=14).move_to([COL[3], y, 0]))

        self.play(
            LaggedStart(
                *[FadeIn(m, shift=RIGHT * 0.05) for m in table], lag_ratio=0.03
            ),
            run_time=1.6,
        )

        # Legend footer — constrained to bottom strip
        foot = VGroup(
            VGroup(
                Text("✅", font_size=13),
                Text("= production-ready", font_size=13, color=C_TEXT),
            ).arrange(RIGHT, buff=0.08),
            VGroup(
                Text("⚠️", font_size=13),
                Text("= experimental", font_size=13, color=C_TEXT),
            ).arrange(RIGHT, buff=0.08),
            VGroup(
                Text("❌", font_size=13),
                Text("= not available", font_size=13, color=C_TEXT),
            ).arrange(RIGHT, buff=0.08),
        ).arrange(RIGHT, buff=0.55)
        foot.to_edge(DOWN, buff=0.30)

        self.play(FadeIn(foot), run_time=0.35)
        self.wait(2.0)
        self._clear(hdr, table, sep, foot, hdrs, t=0.4)

    # ── 6. Takeaway ────────────────────────────────────────────────────────────
    def _section_takeaway(self):
        title = Text("Key Takeaway", font_size=38, color=C_TEXT, weight=BOLD)
        title.to_edge(UP, buff=0.42)

        # Keep rows within safe width: screen is ±7.1, use ±6.0
        pts = [
            (C_CLK, "Classical", "API & Tokenizer — production-ready today"),
            (C_CLK, "Classical", "GPU/TPU handle core training & inference"),
            (C_QUA, "Quantum", "Embeddings, search, sampling — near-term wins"),
            (C_QUA, "Quantum", "QAOA/HHL offer exponential advantages (research)"),
            (
                C_THR,
                "Thermodynamic",
                "Analog crossbars → matrix ops at ~10 000× less energy",
            ),
            (
                C_THR,
                "Thermodynamic",
                "Boltzmann/Gibbs sampling → native diffusion training",
            ),
        ]

        rows = VGroup()
        for color, paradigm, desc in pts:
            dot = Dot(radius=0.09, color=color)
            par = Text(paradigm, font_size=18, color=color, weight=BOLD)
            dash = Text("–", font_size=18, color=C_MUTED)
            dsc = Text(desc, font_size=17, color=C_TEXT)
            row = VGroup(dot, par, dash, dsc).arrange(RIGHT, buff=0.15)
            # Constrain row width
            max_w = 12.5
            if row.width > max_w:
                row.scale(max_w / row.width)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.30)
        rows.next_to(title, DOWN, buff=0.50)
        # Centre horizontally
        rows.move_to([0, rows.get_center()[1], 0])

        footer = Text(
            "Hybrid = best of three worlds — practical today, quantum-ready tomorrow",
            font_size=16,
            color=C_MUTED,
            slant=ITALIC,
        ).to_edge(DOWN, buff=0.35)
        # Constrain footer
        if footer.width > 13.0:
            footer.scale(13.0 / footer.width)

        self.play(Write(title), run_time=0.45)
        self.play(
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.18) for r in rows], lag_ratio=0.15),
            run_time=1.4,
        )
        self.play(FadeIn(footer), run_time=0.35)
        self.wait(3.0)
