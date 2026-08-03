import textwrap, math
import numpy as np
from manim import *
from manim_slides import Slide


class ComputingArchitecturesPresentation(Slide):
    # ------------------------------------------------------------------
    # Global visual style
    # ------------------------------------------------------------------
    BG_COLOR = "#0E1116"
    TITLE_COLOR = "#F5F6F7"
    BODY_COLOR = "#D6D9DE"
    MUTED_COLOR = "#6E7480"

    ACCENT_TEAL = "#4FD1C5"
    ACCENT_ORANGE = "#F0A860"
    ACCENT_BLUE = "#7AA2F7"
    ACCENT_PURPLE = "#B39DFF"
    ACCENT_RED = "#E06C75"
    ACCENT_GREEN = "#73D997"

    SECTION_COLORS = {
        "OVERVIEW": "#B39DFF",
        "CLASSICAL": "#F0A860",
        "THERMODYNAMIC": "#4FD1C5",
        "QUANTUM": "#7AA2F7",
        "HARDWARE": "#E06C75",
        "FUTURE": "#B39DFF",
        "ANIMATION": "#73D997",
        "DRIVERS": "#F0A860",
        "CMOS": "#B39DFF",
        "COMPARISON": "#B39DFF",
    }

    TOTAL_SLIDES = 37
    CONTENT_WIDTH = 11.6
    CONTENT_MAX_HEIGHT = 4.9

    # ==================================================================
    # ENTRY POINT
    # ==================================================================
    def construct(self):
        self.camera.background_color = self.BG_COLOR

        # ── SECTION 1: OVERVIEW ──────────────────────────────────────
        self.s01_title()
        self.s02_why_new_architectures()
        self.s03_three_ways_to_compute()
        self.s04_types_of_computing()

        # ── SECTION 2: CLASSICAL ─────────────────────────────────────
        self.s05_classical_computing()
        self.s06_cmos()
        self.s07_what_is_noise()
        self.s08_avoiding_noise()
        self.s09_drivers_overview()
        self.s10_block_vs_character_devices()

        # ── SECTION 3: THERMODYNAMIC ─────────────────────────────────
        self.s11_thermodynamic_computing()
        self.s12_probabilistic_bit()
        self.s13_noise_becomes_useful()
        self.s14_temperature_and_noise()
        self.s15_gibbs_sampling()
        self.s16_gibbs_animation()
        self.s17_gradient_vs_gibbs()
        self.s18_gradient_vs_gibbs_animation()
        self.s19_energy_based_models()
        self.s20_ising_models()
        self.s21_ising_animation()
        self.s22_boltzmann_machines()
        self.s23_boltzmann_animation()

        # ── SECTION 4: QUANTUM ───────────────────────────────────────
        self.s24_quantum_computing()
        self.s25_bits_pbits_qubits()
        self.s26_quantum_classical_connection()
        self.s27_quantum_hardware()

        # ── SECTION 5: COMPARISONS ───────────────────────────────────
        self.s28_comparing_paradigms()
        self.s29_quantum_vs_thermo_economics()

        # ── SECTION 6: HARDWARE & IMPLEMENTATION ─────────────────────
        self.s30_extropic_tsu()
        self.s31_hopfield_networks()
        self.s32_denoising_models()
        self.s33_fpga_prototype()

        # ── SECTION 7: AI DRIVERS ────────────────────────────────────
        self.s34_deepseek_drivers()
        self.s35_character_drivers()

        # ── SECTION 8: CONCLUSION ────────────────────────────────────
        self.s36_fpga_dev_flow()
        self.s37_conclusion()

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================
    def _next_slide(self):
        """manim-slides: pause here and wait for presenter to advance."""
        self.next_slide()

    def _new_slide(self):
        """Wipe the canvas, then signal manim-slides to advance."""
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.45)
        self.clear()
        self.wait(0.05)
        self._next_slide()

    def _header(self, title_text, section, num):
        color = self.SECTION_COLORS.get(section, self.ACCENT_TEAL)
        tag_label = Text(section, font_size=18, color=self.BG_COLOR, weight=BOLD)
        tag_bg = RoundedRectangle(
            corner_radius=0.08,
            width=tag_label.width + 0.5,
            height=tag_label.height + 0.32,
            stroke_width=0,
            fill_color=color,
            fill_opacity=1,
        )
        tag_label.move_to(tag_bg.get_center())
        tag = VGroup(tag_bg, tag_label).to_corner(UL, buff=0.5)

        title = Text(title_text, font_size=38, color=self.TITLE_COLOR, weight=BOLD)
        if title.width > self.CONTENT_WIDTH:
            title.set_width(self.CONTENT_WIDTH)
        title.next_to(tag, DOWN, buff=0.3, aligned_edge=LEFT)

        underline = Line(LEFT, RIGHT, color=color, stroke_width=4)
        underline.set_width(min(3.2, title.width))
        underline.next_to(title, DOWN, buff=0.22, aligned_edge=LEFT)

        page = Text(
            f"{num:02d} / {self.TOTAL_SLIDES}", font_size=20, color=self.MUTED_COLOR
        )
        page.to_corner(DR, buff=0.4)

        self.play(FadeIn(tag, shift=DOWN * 0.1), run_time=0.35)
        self.play(Write(title), run_time=0.6)
        self.play(Create(underline), FadeIn(page), run_time=0.35)
        return underline

    def _place(self, mobj, anchor, max_width=None, max_height=None, buff=0.55):
        max_width = self.CONTENT_WIDTH if max_width is None else max_width
        max_height = self.CONTENT_MAX_HEIGHT if max_height is None else max_height
        scale = 1.0
        if mobj.width > max_width:
            scale = min(scale, max_width / mobj.width)
        if mobj.height > max_height:
            scale = min(scale, max_height / mobj.height)
        if scale < 1.0:
            mobj.scale(scale)
        mobj.next_to(anchor, DOWN, buff=buff, aligned_edge=LEFT)
        return mobj

    def _bullets(self, items, font_size=28, color=None, wrap=54, marker="›"):
        color = color or self.BODY_COLOR
        rows = VGroup()
        for it in items:
            wrapped = textwrap.fill(it, width=wrap)
            dot = Text(marker, font_size=font_size, color=self.ACCENT_TEAL, weight=BOLD)
            body = Text(wrapped, font_size=font_size, color=color, line_spacing=1.15)
            row = VGroup(dot, body).arrange(RIGHT, buff=0.28, aligned_edge=UP)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        return rows

    def _paragraph(self, text, font_size=30, color=None, wrap=58, weight=NORMAL):
        color = color or self.BODY_COLOR
        return Text(
            textwrap.fill(text, width=wrap),
            font_size=font_size,
            color=color,
            line_spacing=1.25,
            weight=weight,
        )

    def _two_col(
        self,
        left_title,
        left_items,
        right_title,
        right_items,
        left_color=None,
        right_color=None,
        font_size=26,
    ):
        left_color = left_color or self.ACCENT_ORANGE
        right_color = right_color or self.ACCENT_TEAL
        left = VGroup(
            Text(left_title, font_size=30, color=left_color, weight=BOLD),
            self._bullets(left_items, font_size=font_size, wrap=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        right = VGroup(
            Text(right_title, font_size=30, color=right_color, weight=BOLD),
            self._bullets(right_items, font_size=font_size, wrap=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        cols = VGroup(left, right).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        divider = Line(
            UP * (cols.height / 2),
            DOWN * (cols.height / 2),
            color=self.MUTED_COLOR,
            stroke_width=1.5,
        )
        divider.move_to(cols.get_center())
        return VGroup(cols, divider)

    def _flow(
        self,
        labels,
        direction=DOWN,
        box_width=4.6,
        box_height=0.72,
        font_size=24,
        color=None,
    ):
        color = color or self.ACCENT_TEAL
        boxes = VGroup()
        for label in labels:
            box = RoundedRectangle(
                corner_radius=0.12,
                width=box_width,
                height=box_height,
                stroke_color=color,
                stroke_width=2.5,
                fill_color=color,
                fill_opacity=0.12,
            )
            txt = Text(label, font_size=font_size, color=self.TITLE_COLOR)
            if txt.width > box_width - 0.4:
                txt.set_width(box_width - 0.4)
            txt.move_to(box.get_center())
            boxes.add(VGroup(box, txt))
        boxes.arrange(direction, buff=0.45)
        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrow = Arrow(
                boxes[i].get_edge_center(direction),
                boxes[i + 1].get_edge_center(-direction),
                buff=0.06,
                color=self.MUTED_COLOR,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.35,
            )
            arrows.add(arrow)
        return VGroup(arrows, boxes)

    def _grid_table(self, headers, rows, font_size=24):
        col_labels = [
            Text(h, font_size=font_size, color=self.ACCENT_TEAL, weight=BOLD)
            for h in headers
        ]
        table = Table(
            rows,
            col_labels=col_labels,
            element_to_mobject=Text,
            element_to_mobject_config={
                "font_size": font_size - 4,
                "color": self.BODY_COLOR,
            },
            include_outer_lines=True,
            line_config={"color": self.MUTED_COLOR, "stroke_width": 1.5},
            v_buff=0.45,
            h_buff=0.7,
        )
        table.get_horizontal_lines().set_color(self.MUTED_COLOR)
        table.get_vertical_lines().set_color(self.MUTED_COLOR)
        return table

    def _quote(self, text, font_size=32, color=None):
        color = color or self.TITLE_COLOR
        mark = Text(
            "\u201C", font_size=font_size + 20, color=self.ACCENT_TEAL, weight=BOLD
        )
        body = self._paragraph(text, font_size=font_size, color=color, wrap=44)
        return VGroup(mark, body).arrange(RIGHT, buff=0.2, aligned_edge=UP)

    def _pill(self, text, color):
        label = Text(text, font_size=20, color=self.BG_COLOR, weight=BOLD)
        bg = RoundedRectangle(
            corner_radius=0.18,
            width=label.width + 0.55,
            height=label.height + 0.38,
            stroke_width=0,
            fill_color=color,
            fill_opacity=1,
        )
        label.move_to(bg.get_center())
        return VGroup(bg, label)

    def _neuron(self, radius=0.22, color=None, fill_opacity=0.25):
        color = color or self.ACCENT_TEAL
        return Circle(
            radius=radius,
            color=color,
            stroke_width=2.5,
            fill_color=color,
            fill_opacity=fill_opacity,
        )

    # ==================================================================
    # ═══════════════════════  SECTION 1: OVERVIEW  ═══════════════════
    # ==================================================================

    # ── SLIDE 01 — TITLE ─────────────────────────────────────────────
    def s01_title(self):
        self._new_slide()
        kicker = Text(
            "A TOUR OF NEXT-GENERATION COMPUTE",
            font_size=22,
            color=self.MUTED_COLOR,
            weight=BOLD,
        )
        kicker.to_edge(UP, buff=1.1)
        title = Text(
            "New Computing\nArchitectures",
            font_size=56,
            color=self.TITLE_COLOR,
            weight=BOLD,
            line_spacing=1.1,
        )
        title.next_to(kicker, DOWN, buff=0.5)
        subtitle = Text(
            "Classical · Quantum · Thermodynamic Computing",
            font_size=26,
            color=self.ACCENT_TEAL,
        )
        subtitle.next_to(title, DOWN, buff=0.45)
        chips = VGroup(
            *[
                self._pill(label, color)
                for label, color in [
                    ("CLASSICAL", self.ACCENT_ORANGE),
                    ("QUANTUM", self.ACCENT_BLUE),
                    ("THERMODYNAMIC", self.ACCENT_TEAL),
                ]
            ]
        ).arrange(RIGHT, buff=0.4)
        chips.next_to(subtitle, DOWN, buff=0.7)
        self.play(FadeIn(kicker, shift=DOWN * 0.2))
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.1))
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in chips], lag_ratio=0.2))
        self.wait(1.2)

    # ── SLIDE 02 — WHY NEW ARCHITECTURES ─────────────────────────────
    def s02_why_new_architectures(self):
        self._new_slide()
        anchor = self._header("Why New Computing Architectures?", "OVERVIEW", 2)
        intro = self._paragraph(
            "AI workloads are growing faster than traditional computer "
            "architectures. Modern AI demands enormous:",
            font_size=26,
        )
        self._place(intro, anchor, buff=0.5)
        needs = self._bullets(
            [
                "Matrix multiplication",
                "Memory bandwidth",
                "Random sampling",
                "Optimization",
                "Energy",
            ],
            font_size=26,
            wrap=40,
        )
        needs.next_to(intro, DOWN, buff=0.35, aligned_edge=LEFT)
        question = self._quote(
            "Can we build hardware that computes differently instead of "
            "simply making CPUs and GPUs faster?",
            font_size=24,
        )
        self._place(question, needs, buff=0.4)
        self.play(FadeIn(intro, shift=UP * 0.15))
        self.play(LaggedStartMap(FadeIn, needs, shift=RIGHT * 0.1, lag_ratio=0.15))
        self.play(FadeIn(question, shift=UP * 0.15))
        self.wait(1.2)

    # ── SLIDE 03 — THREE WAYS TO COMPUTE ─────────────────────────────
    def s03_three_ways_to_compute(self):
        self._new_slide()
        anchor = self._header("Three Ways to Compute", "OVERVIEW", 3)
        headers = ["Classical", "Quantum", "Thermodynamic"]
        rows = [
            ["Deterministic", "Quantum mechanics", "Statistical physics"],
            ["Bits", "Qubits", "Probabilistic bits (p-bits)"],
            [
                "Exact operations",
                "Superposition & interference",
                "Controlled randomness",
            ],
            ["Logic circuits", "Quantum gates", "Stochastic circuits"],
        ]
        table = self._grid_table(headers, rows, font_size=24)
        self._place(table, anchor, max_height=4.2, buff=0.55)
        note = self._paragraph(
            "All three manipulate information — just through different physical principles.",
            font_size=22,
            color=self.MUTED_COLOR,
            wrap=64,
        )
        note.next_to(table, DOWN, buff=0.4)
        self.play(FadeIn(table, shift=UP * 0.1))
        self.play(FadeIn(note))
        self.wait(1.2)

    # ── SLIDE 04 — DIFFERENT TYPES OF COMPUTING ───────────────────────
    def s04_types_of_computing(self):
        self._new_slide()
        anchor = self._header("Different Types of Computing", "OVERVIEW", 4)
        headers = ["Paradigm", "Mechanism", "Strength", "Weakness", "Example HW"]
        rows = [
            [
                "Classical",
                "Boolean logic / CMOS",
                "General purpose",
                "Energy wall",
                "CPU, GPU",
            ],
            [
                "Thermodynamic",
                "Thermal fluctuations / p-bits",
                "Sampling, ML opt.",
                "Niche algorithms",
                "TSU (Extropic)",
            ],
            [
                "Quantum",
                "Superposition / entanglement",
                "Quantum algorithms",
                "Error, cold, costly",
                "IBM QPU, IonQ",
            ],
            [
                "Neuromorphic",
                "Spike-based neural dynamics",
                "Ultra-low power AI",
                "Hard to program",
                "Intel Loihi",
            ],
            [
                "Analog",
                "Continuous voltage/current",
                "Fast, energy-eff.",
                "Noise, precision",
                "Mythic AI",
            ],
            [
                "In-Memory",
                "Compute inside DRAM/SRAM",
                "Memory bandwidth",
                "Limited precision",
                "Samsung HBM-PIM",
            ],
            [
                "Optical",
                "Light-speed matrix ops",
                "Ultra-fast linear",
                "WIP integration",
                "Lightmatter",
            ],
        ]
        table = self._grid_table(headers, rows, font_size=16)
        self._place(table, anchor, max_height=4.9, buff=0.45)
        self.play(FadeIn(table, shift=UP * 0.1))
        closing = self._paragraph(
            "No single paradigm dominates all workloads — heterogeneous accelerators will coexist.",
            font_size=21,
            color=self.ACCENT_TEAL,
            wrap=64,
        )
        closing.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(closing))
        self.wait(1.5)

    # ==================================================================
    # ═══════════════════  SECTION 2: CLASSICAL  ══════════════════════
    # ==================================================================

    # ── SLIDE 05 — CLASSICAL COMPUTING ───────────────────────────────
    def s05_classical_computing(self):
        self._new_slide()
        anchor = self._header("Classical Computing", "CLASSICAL", 5)
        text = self._paragraph(
            "Everything inside a CPU or GPU is built from deterministic logic. "
            "The fundamental unit is the bit.",
            font_size=26,
        )
        self._place(text, anchor, buff=0.5)
        bit_box = VGroup(
            *[self._pill(b, self.ACCENT_ORANGE) for b in ["0", "1"]]
        ).arrange(RIGHT, buff=0.6)
        bit_box.next_to(text, DOWN, buff=0.5)
        eq = Text("2 + 2 = 4", font_size=34, color=self.TITLE_COLOR, weight=BOLD)
        always = Text("— always.", font_size=24, color=self.MUTED_COLOR)
        eq_group = VGroup(eq, always).arrange(RIGHT, buff=0.3)
        eq_group.next_to(bit_box, DOWN, buff=0.55)
        footer_txt = self._paragraph(
            "This predictability is the foundation of operating systems, "
            "databases, networking, and programming languages.",
            font_size=22,
            color=self.MUTED_COLOR,
            wrap=62,
        )
        footer_txt.next_to(eq_group, DOWN, buff=0.45)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(LaggedStart(*[FadeIn(b, scale=0.8) for b in bit_box], lag_ratio=0.3))
        self.play(Write(eq_group))
        self.play(FadeIn(footer_txt))
        self.wait(1.2)

    # ── SLIDE 06 — CMOS ──────────────────────────────────────────────
    def s06_cmos(self):
        self._new_slide()
        anchor = self._header("CMOS Technology", "CMOS", 6)
        intro = self._paragraph(
            "Complementary Metal-Oxide-Semiconductor (CMOS) is the foundation of virtually "
            "all modern digital chips — from microcontrollers to the latest AI accelerators.",
            font_size=25,
        )
        self._place(intro, anchor, buff=0.45)
        self.play(FadeIn(intro, shift=UP * 0.1))

        # CMOS inverter diagram (symbolic)
        line_v = Line(UP * 1.1, DOWN * 1.1, color=self.BODY_COLOR, stroke_width=3)
        pmos_box = Rectangle(
            width=0.9,
            height=0.5,
            color=self.ACCENT_PURPLE,
            stroke_width=2.5,
            fill_color=self.ACCENT_PURPLE,
            fill_opacity=0.15,
        )
        pmos_lbl = Text("pMOS", font_size=17, color=self.ACCENT_PURPLE)
        pmos_lbl.move_to(pmos_box.get_center())
        pmos = VGroup(pmos_box, pmos_lbl)
        pmos.move_to(line_v.get_top() + DOWN * 0.28)
        nmos_box = Rectangle(
            width=0.9,
            height=0.5,
            color=self.ACCENT_ORANGE,
            stroke_width=2.5,
            fill_color=self.ACCENT_ORANGE,
            fill_opacity=0.15,
        )
        nmos_lbl = Text("nMOS", font_size=17, color=self.ACCENT_ORANGE)
        nmos_lbl.move_to(nmos_box.get_center())
        nmos = VGroup(nmos_box, nmos_lbl)
        nmos.move_to(line_v.get_bottom() + UP * 0.28)
        vdd = Text("VDD", font_size=20, color=self.ACCENT_RED)
        gnd = Text("GND", font_size=20, color=self.ACCENT_BLUE)
        vdd.next_to(line_v, UP, buff=0.15)
        gnd.next_to(line_v, DOWN, buff=0.15)
        vin = Text("Vin", font_size=20, color=self.ACCENT_ORANGE)
        vout = Text("Vout", font_size=20, color=self.ACCENT_TEAL)
        vin.next_to(pmos, LEFT, buff=0.4)
        arrow_in = Arrow(
            vin.get_right(),
            pmos.get_left(),
            buff=0.05,
            stroke_width=2,
            color=self.MUTED_COLOR,
            max_tip_length_to_length_ratio=0.4,
        )
        vout.next_to(pmos, RIGHT, buff=0.5)
        arrow_out = Arrow(
            pmos.get_right(),
            vout.get_left(),
            buff=0.05,
            stroke_width=2,
            color=self.MUTED_COLOR,
            max_tip_length_to_length_ratio=0.4,
        )
        cmos_diag = VGroup(line_v, pmos, nmos, vdd, gnd, vin, vout, arrow_in, arrow_out)
        cmos_diag.next_to(intro, DOWN, buff=0.5)
        cmos_diag.align_to(intro, LEFT)
        cmos_diag.shift(RIGHT * 1.5)
        self._place(cmos_diag, intro, buff=0.5, max_height=2.2, max_width=4.0)

        facts = self._bullets(
            [
                "pMOS pulls output HIGH when input is LOW",
                "nMOS pulls output LOW when input is HIGH",
                "Never both ON simultaneously → near-zero static power",
                "Logic families: CMOS, BiCMOS, FinFET, Gate-All-Around (2nm+)",
                "Thermodynamic computing exploits transistor noise in CMOS fabric",
            ],
            font_size=21,
            wrap=38,
        )
        facts.next_to(intro, DOWN, buff=0.5)
        facts.to_edge(RIGHT, buff=0.8)
        self.play(FadeIn(cmos_diag))
        self.play(LaggedStartMap(FadeIn, facts, shift=RIGHT * 0.1, lag_ratio=0.12))
        self.wait(1.5)

    # ── SLIDE 07 — WHAT IS NOISE? ─────────────────────────────────────
    def s07_what_is_noise(self):
        self._new_slide()
        anchor = self._header("What is Noise?", "CLASSICAL", 7)
        intro = self._paragraph(
            "Every electronic circuit contains physical noise. Sources include:",
            font_size=26,
        )
        self._place(intro, anchor, buff=0.5)
        sources = self._bullets(
            [
                "Thermal motion",
                "Electron movement",
                "Electromagnetic interference",
                "Shot noise",
                "Flicker noise",
                "Quantum effects",
            ],
            font_size=25,
            wrap=40,
        )
        sources.next_to(intro, DOWN, buff=0.4, aligned_edge=LEFT)
        tagline = self._paragraph(
            "Noise exists whether we want it or not.",
            font_size=24,
            color=self.ACCENT_TEAL,
            wrap=50,
        )
        tagline.next_to(sources, DOWN, buff=0.45)
        self.play(FadeIn(intro, shift=UP * 0.15))
        self.play(LaggedStartMap(FadeIn, sources, shift=RIGHT * 0.1, lag_ratio=0.12))
        self.play(FadeIn(tagline))
        self.wait(1.2)

    # ── SLIDE 08 — WHY CLASSICAL COMPUTERS AVOID NOISE ───────────────
    def s08_avoiding_noise(self):
        self._new_slide()
        anchor = self._header("Why Classical Computers Avoid Noise", "CLASSICAL", 8)
        ok_row = VGroup(
            Text("5.00V → 4.98V", font_size=26, color=self.BODY_COLOR),
            Text("still reads", font_size=22, color=self.MUTED_COLOR),
            self._pill("1", self.ACCENT_TEAL),
        ).arrange(RIGHT, buff=0.3)
        bad_row = VGroup(
            Text("5V → 2.3V", font_size=26, color=self.BODY_COLOR),
            Text("misread as", font_size=22, color=self.MUTED_COLOR),
            self._pill("0", self.ACCENT_RED),
            Text("(hardware error)", font_size=20, color=self.MUTED_COLOR),
        ).arrange(RIGHT, buff=0.3)
        rows = VGroup(ok_row, bad_row).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self._place(rows, anchor, buff=0.55)
        mitig_title = Text(
            "Fighting noise costs enormous effort:",
            font_size=24,
            color=self.TITLE_COLOR,
            weight=BOLD,
        )
        mitig_title.next_to(rows, DOWN, buff=0.5).align_to(rows, LEFT)
        mitig = self._bullets(
            [
                "Voltage margins",
                "Error correction",
                "Shielding",
                "Clock synchronization",
                "Stable CMOS circuits",
            ],
            font_size=22,
            wrap=42,
        )
        mitig.next_to(mitig_title, DOWN, buff=0.3, aligned_edge=LEFT)
        whole = VGroup(rows, mitig_title, mitig)
        self._place(whole, anchor, buff=0.5, max_height=4.9)
        self.play(FadeIn(ok_row, shift=UP * 0.1))
        self.play(FadeIn(bad_row, shift=UP * 0.1))
        self.play(FadeIn(mitig_title))
        self.play(LaggedStartMap(FadeIn, mitig, shift=RIGHT * 0.1, lag_ratio=0.12))
        self.wait(1.2)

    # ── SLIDE 09 — OS DRIVERS OVERVIEW ───────────────────────────────
    def s09_drivers_overview(self):
        self._new_slide()
        anchor = self._header("Operating System Drivers — Overview", "DRIVERS", 9)
        intro = self._paragraph(
            "A device driver is software that allows the OS kernel to communicate "
            "with hardware components. Drivers abstract hardware details behind a "
            "standard interface.",
            font_size=25,
        )
        self._place(intro, anchor, buff=0.5)
        self.play(FadeIn(intro, shift=UP * 0.1))
        flow = self._flow(
            [
                "User Application",
                "System Call (read/write/ioctl)",
                "Kernel VFS / Subsystem",
                "Device Driver",
                "Hardware Register / Bus (PCIe, USB, I²C …)",
            ],
            direction=DOWN,
            box_width=7.5,
            box_height=0.78,
            font_size=21,
            color=self.ACCENT_ORANGE,
        )
        self._place(flow, intro, buff=0.45, max_height=3.8)
        self.play(
            LaggedStart(*[FadeIn(b) for b in flow[1]], lag_ratio=0.2),
            LaggedStart(*[Create(a) for a in flow[0]], lag_ratio=0.2),
        )
        self.wait(1.4)

    # ── SLIDE 10 — BLOCK vs CHARACTER DEVICES ────────────────────────
    def s10_block_vs_character_devices(self):
        self._new_slide()
        anchor = self._header("Block Devices vs Character Devices", "DRIVERS", 10)
        compare = self._two_col(
            "Block Devices",
            [
                "Data transferred in fixed-size blocks",
                "Random access supported",
                "Buffered I/O via page cache",
                "Examples: HDDs, SSDs, NVMe, RAM disks",
            ],
            "Character Devices",
            [
                "Data transferred byte-by-byte (stream)",
                "Sequential access only",
                "No kernel buffer (direct)",
                "Examples: serial ports, keyboard, /dev/random",
            ],
            left_color=self.ACCENT_BLUE,
            right_color=self.ACCENT_ORANGE,
            font_size=22,
        )
        self._place(compare, anchor, buff=0.6)
        self.play(FadeIn(compare, shift=UP * 0.1))
        examples = self._paragraph(
            "In Linux both appear as files under /dev. "
            "Block: /dev/sda  ·  Character: /dev/ttyS0, /dev/null",
            font_size=21,
            color=self.MUTED_COLOR,
            wrap=62,
        )
        examples.next_to(compare, DOWN, buff=0.5)
        self.play(FadeIn(examples))
        self.wait(1.5)

    # ==================================================================
    # ══════════════════  SECTION 3: THERMODYNAMIC  ═══════════════════
    # ==================================================================

    # ── SLIDE 11 — THERMODYNAMIC COMPUTING ───────────────────────────
    def s11_thermodynamic_computing(self):
        self._new_slide()
        anchor = self._header("Thermodynamic Computing", "THERMODYNAMIC", 11)
        q = self._quote(
            "Instead of removing noise — can we compute with it?", font_size=30
        )
        self._place(q, anchor, buff=0.7)
        compare = self._two_col(
            "Classical bits",
            ["Deterministic", "Certainty stored", "Noise = enemy"],
            "Thermodynamic p-bits",
            ["Probabilistic", "Probability stored", "Noise = resource"],
            left_color=self.ACCENT_ORANGE,
            right_color=self.ACCENT_TEAL,
        )
        self._place(compare, q, buff=0.7)
        self.play(FadeIn(q, shift=UP * 0.15))
        self.play(FadeIn(compare, shift=UP * 0.1))
        self.wait(1.2)

    # ── SLIDE 12 — THE P-BIT ─────────────────────────────────────────
    def s12_probabilistic_bit(self):
        self._new_slide()
        anchor = self._header("The Probabilistic Bit (p-bit)", "THERMODYNAMIC", 12)
        text = self._paragraph(
            "A p-bit is neither permanently 0 nor permanently 1 — it continuously "
            "fluctuates. The probability of observing 0 or 1 depends on an input signal.",
            font_size=26,
        )
        self._place(text, anchor, buff=0.55)
        fluctuation = Text(
            "1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1",
            font_size=28,
            color=self.ACCENT_TEAL,
            weight=BOLD,
        )
        fluctuation.next_to(text, DOWN, buff=0.5)
        caption = self._paragraph(
            "Instead of storing certainty, the circuit stores probability.",
            font_size=24,
            color=self.MUTED_COLOR,
            wrap=54,
        )
        caption.next_to(fluctuation, DOWN, buff=0.45)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(Write(fluctuation))
        self.play(FadeIn(caption))
        self.wait(1.2)

    # ── SLIDE 13 — WHY NOISE BECOMES USEFUL ──────────────────────────
    def s13_noise_becomes_useful(self):
        self._new_slide()
        anchor = self._header("Why Noise Becomes Useful", "THERMODYNAMIC", 13)
        intro = self._paragraph(
            "Many AI algorithms already require randomness:", font_size=26
        )
        self._place(intro, anchor, buff=0.5)
        algos = self._bullets(
            [
                "Monte Carlo",
                "Gibbs Sampling",
                "Markov Chain Monte Carlo",
                "Diffusion Models",
                "Bayesian Inference",
                "Boltzmann Machines",
            ],
            font_size=25,
            wrap=40,
        )
        algos.next_to(intro, DOWN, buff=0.4, aligned_edge=LEFT)
        note = self._paragraph(
            "GPUs generate randomness digitally. Thermodynamic computers "
            "measure it directly from physical fluctuations.",
            font_size=22,
            color=self.MUTED_COLOR,
            wrap=58,
        )
        note.next_to(algos, DOWN, buff=0.4)
        self.play(FadeIn(intro, shift=UP * 0.15))
        self.play(LaggedStartMap(FadeIn, algos, shift=RIGHT * 0.1, lag_ratio=0.12))
        self.play(FadeIn(note))
        self.wait(1.2)

    # ── SLIDE 14 — TEMPERATURE AND NOISE ─────────────────────────────
    def s14_temperature_and_noise(self):
        self._new_slide()
        anchor = self._header("Temperature and Noise", "THERMODYNAMIC", 14)
        low = VGroup(
            Text("LOW TEMPERATURE", font_size=22, color=self.ACCENT_BLUE, weight=BOLD),
            self._paragraph(
                "Very little randomness → trapped in local minima",
                font_size=22,
                wrap=28,
            ),
        ).arrange(DOWN, buff=0.3)
        high = VGroup(
            Text("HIGH TEMPERATURE", font_size=22, color=self.ACCENT_RED, weight=BOLD),
            self._paragraph(
                "Too much randomness → never converges", font_size=22, wrap=28
            ),
        ).arrange(DOWN, buff=0.3)
        pair = VGroup(low, high).arrange(RIGHT, buff=1.3, aligned_edge=UP)
        self._place(pair, anchor, buff=0.6)
        sweet_spot = self._paragraph(
            "The optimal operating point balances exploration and convergence — "
            "seen in Simulated Annealing, Gibbs Sampling, Boltzmann Machines, "
            "and Thermodynamic Hardware.",
            font_size=23,
            color=self.ACCENT_TEAL,
            wrap=58,
        )
        sweet_spot.next_to(pair, DOWN, buff=0.55)
        self.play(FadeIn(low, shift=UP * 0.1))
        self.play(FadeIn(high, shift=UP * 0.1))
        self.play(FadeIn(sweet_spot))
        self.wait(1.2)

    # ── SLIDE 15 — GIBBS SAMPLING (concept) ──────────────────────────
    def s15_gibbs_sampling(self):
        self._new_slide()
        anchor = self._header("Gibbs Sampling", "THERMODYNAMIC", 15)
        intro = self._paragraph(
            "Gibbs Sampling updates one variable at a time according to probability.",
            font_size=25,
        )
        self._place(intro, anchor, buff=0.5)
        flow = self._flow(
            [
                "Current state",
                "Compute local\nprobabilities",
                "Sample",
                "Update",
                "Repeat",
            ],
            direction=RIGHT,
            box_width=2.5,
            box_height=1.1,
            font_size=20,
        )
        self._place(flow, intro, buff=0.6, max_width=self.CONTENT_WIDTH)
        outro = self._paragraph(
            "Eventually, the system approaches the desired probability distribution.",
            font_size=22,
            color=self.MUTED_COLOR,
            wrap=58,
        )
        outro.next_to(flow, DOWN, buff=0.5)
        self.play(FadeIn(intro, shift=UP * 0.15))
        self.play(
            LaggedStart(*[FadeIn(b) for b in flow[1]], lag_ratio=0.2),
            LaggedStart(*[Create(a) for a in flow[0]], lag_ratio=0.2),
        )
        self.play(FadeIn(outro))
        self.wait(1.2)

    # ── SLIDE 16 — GIBBS SAMPLING ANIMATION ──────────────────────────
    def s16_gibbs_animation(self):
        self._new_slide()
        anchor = self._header("Gibbs Sampling — Step by Step", "ANIMATION", 16)
        intro = self._paragraph(
            "Watch one Gibbs step: fix X₂ and X₃, then resample X₁ from its "
            "conditional distribution.",
            font_size=24,
        )
        self._place(intro, anchor, buff=0.45)
        self.play(FadeIn(intro, shift=UP * 0.1))

        labels = ["X₁", "X₂", "X₃"]
        colors = [self.ACCENT_ORANGE, self.ACCENT_TEAL, self.ACCENT_BLUE]
        nodes = VGroup()
        for lbl, col in zip(labels, colors):
            c = Circle(
                radius=0.42,
                color=col,
                stroke_width=3,
                fill_color=col,
                fill_opacity=0.18,
            )
            t = Text(lbl, font_size=26, color=self.TITLE_COLOR, weight=BOLD)
            t.move_to(c.get_center())
            nodes.add(VGroup(c, t))
        nodes.arrange(RIGHT, buff=1.1)
        nodes.next_to(intro, DOWN, buff=0.6).align_to(intro, LEFT)

        edges = VGroup()
        for i in range(len(nodes) - 1):
            edges.add(
                Line(
                    nodes[i].get_right(),
                    nodes[i + 1].get_left(),
                    color=self.MUTED_COLOR,
                    stroke_width=2,
                )
            )
        self.play(
            Create(edges), LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.2)
        )
        self.wait(0.3)

        value_labels = VGroup(
            *[Text(v, font_size=22, color=self.BODY_COLOR) for v in ["?", "1", "0"]]
        )
        for vl, nd in zip(value_labels, nodes):
            vl.next_to(nd, DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(vl) for vl in value_labels], lag_ratio=0.2))
        self.wait(0.3)

        highlight = SurroundingRectangle(
            nodes[0],
            color=self.ACCENT_TEAL,
            stroke_width=3,
            corner_radius=0.15,
            buff=0.1,
        )
        label_fix = Text("Resampling…", font_size=21, color=self.ACCENT_TEAL)
        label_fix.next_to(highlight, UP, buff=0.15)
        self.play(Create(highlight), FadeIn(label_fix))
        self.wait(0.5)

        TARGET_W = 2.4
        FILL_W = TARGET_W * 0.75
        bar_bg = Rectangle(
            width=TARGET_W,
            height=0.35,
            color=self.MUTED_COLOR,
            stroke_width=1.5,
            fill_color=self.MUTED_COLOR,
            fill_opacity=0.15,
        )
        bar_bg.next_to(nodes[0], DOWN, buff=1.0)
        bar_fg = Rectangle(
            width=0.001,
            height=0.35,
            stroke_width=0,
            fill_color=self.ACCENT_TEAL,
            fill_opacity=0.9,
        )
        bar_fg.align_to(bar_bg, LEFT).align_to(bar_bg, UP)
        prob_lbl = Text("P(X₁=1 | X₂,X₃)", font_size=19, color=self.MUTED_COLOR)
        prob_lbl.next_to(bar_bg, UP, buff=0.1)
        self.play(FadeIn(bar_bg), FadeIn(prob_lbl), FadeIn(bar_fg))
        bar_fg_full = Rectangle(
            width=FILL_W,
            height=0.35,
            stroke_width=0,
            fill_color=self.ACCENT_TEAL,
            fill_opacity=0.9,
        )
        bar_fg_full.align_to(bar_bg, LEFT).align_to(bar_bg, UP)
        self.play(bar_fg.animate.become(bar_fg_full), run_time=1.0)
        self.wait(0.3)

        new_val = Text("1", font_size=22, color=self.ACCENT_TEAL, weight=BOLD)
        new_val.move_to(value_labels[0].get_center())
        self.play(Transform(value_labels[0], new_val))
        self.wait(0.4)
        self.play(
            FadeOut(highlight),
            FadeOut(label_fix),
            FadeOut(bar_bg),
            FadeOut(bar_fg),
            FadeOut(prob_lbl),
        )
        outro = self._paragraph(
            "After many sweeps the joint distribution P(X₁,X₂,X₃) is sampled correctly — "
            "even from complex multimodal distributions.",
            font_size=21,
            color=self.MUTED_COLOR,
            wrap=62,
        )
        outro.next_to(nodes, DOWN, buff=1.0)
        self.play(FadeIn(outro))
        self.wait(1.5)

    # ── SLIDE 17 — GRADIENT DESCENT vs GIBBS (concept) ───────────────
    def s17_gradient_vs_gibbs(self):
        self._new_slide()
        anchor = self._header("Gradient Descent vs Gibbs Sampling", "THERMODYNAMIC", 17)
        compare = self._two_col(
            "Gradient Descent",
            ["Deterministic", "Follows steepest path", "Can become trapped"],
            "Gibbs Sampling",
            ["Stochastic", "Explores possibilities", "Escapes local minima"],
            left_color=self.ACCENT_ORANGE,
            right_color=self.ACCENT_TEAL,
        )
        self._place(compare, anchor, buff=0.7)
        note = self._paragraph(
            "Thermodynamic hardware naturally accelerates sampling-based optimization.",
            font_size=24,
            color=self.ACCENT_TEAL,
            wrap=54,
        )
        note.next_to(compare, DOWN, buff=0.55)
        self.play(FadeIn(compare, shift=UP * 0.1))
        self.play(FadeIn(note))
        self.wait(1.2)

    # ── SLIDE 18 — GRADIENT vs GIBBS ANIMATION ───────────────────────
    def s18_gradient_vs_gibbs_animation(self):
        self._new_slide()
        anchor = self._header("Gradient Descent vs Gibbs — Animated", "ANIMATION", 18)

        def make_landscape(label, color, ax_x_range, ax_y_range):
            axes = Axes(
                x_range=ax_x_range,
                y_range=ax_y_range,
                x_length=4.8,
                y_length=2.8,
                axis_config={"color": self.MUTED_COLOR, "stroke_width": 2},
            )
            for ax in (axes.x_axis, axes.y_axis):
                if hasattr(ax, "get_tick_marks"):
                    ax.get_tick_marks().set_opacity(0)
                if hasattr(ax, "tip"):
                    ax.tip.set_opacity(0)
            curve = axes.plot(
                lambda x: 0.55 * x**2 - 1.1 * x + 0.6 * np.sin(3 * x) + 0.5,
                x_range=ax_x_range[:2],
                color=color,
                stroke_width=3,
            )
            lbl = Text(label, font_size=22, color=color, weight=BOLD)
            lbl.next_to(axes, UP, buff=0.2)
            return VGroup(axes, curve, lbl), axes

        gd_group, gd_axes = make_landscape(
            "Gradient Descent", self.ACCENT_ORANGE, [-2, 2, 1], [-1, 3, 1]
        )
        gi_group, gi_axes = make_landscape(
            "Gibbs Sampling", self.ACCENT_TEAL, [-2, 2, 1], [-1, 3, 1]
        )
        all_plots = VGroup(gd_group, gi_group).arrange(RIGHT, buff=1.2)
        all_plots.next_to(anchor, DOWN, buff=0.6).align_to(anchor, LEFT)
        self._place(all_plots, anchor, buff=0.55, max_height=3.5)
        self.play(FadeIn(gd_group), FadeIn(gi_group))
        self.wait(0.2)

        gd_ball = Dot(gd_axes.c2p(-1.6, 1.8), color=self.ACCENT_ORANGE, radius=0.12)
        self.play(FadeIn(gd_ball))
        path_gd = [
            gd_axes.c2p(-1.6, 1.8),
            gd_axes.c2p(-1.0, 0.9),
            gd_axes.c2p(-0.4, 0.45),
            gd_axes.c2p(-0.15, 0.40),
        ]
        self.play(
            MoveAlongPath(gd_ball, VMobject().set_points_as_corners(path_gd)),
            run_time=1.4,
            rate_func=rush_into,
        )
        stuck = Text("Stuck!", font_size=19, color=self.ACCENT_RED)
        stuck.next_to(gd_ball, UP, buff=0.1)
        self.play(FadeIn(stuck, scale=0.8))
        self.wait(0.4)

        gi_ball = Dot(gi_axes.c2p(-1.6, 1.8), color=self.ACCENT_TEAL, radius=0.12)
        self.play(FadeIn(gi_ball))
        jump_points = [
            gi_axes.c2p(-1.6, 1.8),
            gi_axes.c2p(-0.5, 1.2),
            gi_axes.c2p(0.3, 0.55),
            gi_axes.c2p(0.9, 0.35),
            gi_axes.c2p(1.1, 0.3),
        ]
        for pt in jump_points[1:]:
            self.play(gi_ball.animate.move_to(pt), run_time=0.35)
        found = Text("Global min!", font_size=19, color=self.ACCENT_GREEN)
        found.next_to(gi_ball, UP, buff=0.1)
        self.play(FadeIn(found, scale=0.8))
        self.wait(0.5)

        caption = self._paragraph(
            "Gibbs sampling's stochastic jumps allow escaping local minima.",
            font_size=21,
            color=self.MUTED_COLOR,
            wrap=64,
        )
        caption.next_to(all_plots, DOWN, buff=0.4)
        self.play(FadeIn(caption))
        self.wait(1.5)

    # ── SLIDE 19 — ENERGY-BASED MODELS ───────────────────────────────
    def s19_energy_based_models(self):
        self._new_slide()
        anchor = self._header("Energy-Based Models", "THERMODYNAMIC", 19)
        text = self._paragraph(
            "Instead of predicting outputs directly, Energy-Based Models define an "
            "energy function. The hardware repeatedly searches for lower-energy states.",
            font_size=27,
        )
        self._place(text, anchor, buff=0.6)
        scale = VGroup(
            self._pill("Lower energy", self.ACCENT_TEAL),
            Arrow(LEFT, RIGHT, color=self.MUTED_COLOR, stroke_width=3),
            self._pill("Better solution", self.ACCENT_BLUE),
        ).arrange(RIGHT, buff=0.4)
        scale.next_to(text, DOWN, buff=0.65)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(FadeIn(scale, shift=UP * 0.1))
        self.wait(1.3)

    # ── SLIDE 20 — ISING MODELS (concept) ────────────────────────────
    def s20_ising_models(self):
        self._new_slide()
        anchor = self._header("Ising Models", "THERMODYNAMIC", 20)
        text = self._paragraph(
            "Many optimization problems can be represented as interacting spins. "
            "The objective is to minimize the total energy.",
            font_size=26,
        )
        self._place(text, anchor, buff=0.55)
        spins = ["+1", "-1", "+1", "+1", "-1"]
        spin_row = VGroup(
            *[
                self._pill(s, self.ACCENT_TEAL if s == "+1" else self.ACCENT_RED)
                for s in spins
            ]
        ).arrange(RIGHT, buff=0.35)
        spin_row.next_to(text, DOWN, buff=0.55)
        apps_title = Text(
            "Applications:", font_size=22, color=self.TITLE_COLOR, weight=BOLD
        )
        apps = self._bullets(
            [
                "Routing",
                "Scheduling",
                "Graph Coloring",
                "AI Optimization",
                "Protein Folding",
            ],
            font_size=22,
            wrap=42,
        )
        apps_group = VGroup(apps_title, apps).arrange(
            DOWN, aligned_edge=LEFT, buff=0.25
        )
        apps_group.next_to(spin_row, DOWN, buff=0.5)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(
            LaggedStart(*[FadeIn(s, scale=0.8) for s in spin_row], lag_ratio=0.15)
        )
        self.play(FadeIn(apps_title))
        self.play(LaggedStartMap(FadeIn, apps, shift=RIGHT * 0.1, lag_ratio=0.12))
        self.wait(1.2)

    # ── SLIDE 21 — ISING MODEL ANIMATION ─────────────────────────────
    def s21_ising_animation(self):
        self._new_slide()
        anchor = self._header("Ising Model — Spin Dynamics", "ANIMATION", 21)
        intro = self._paragraph(
            "A 5×5 Ising lattice. Each cell is a spin (+1 or −1). "
            "Watch spins align as the system relaxes to lower energy.",
            font_size=24,
        )
        self._place(intro, anchor, buff=0.4)
        self.play(FadeIn(intro, shift=UP * 0.1))

        N = 5
        cell_size = 0.7
        spins = np.random.choice([-1, 1], size=(N, N))
        cells = []
        grid = VGroup()
        origin = np.array([-cell_size * (N - 1) / 2, -0.2, 0])
        for r in range(N):
            row = []
            for c in range(N):
                pos = origin + np.array([c * cell_size, -r * cell_size - 1.8, 0])
                col = self.ACCENT_TEAL if spins[r, c] == 1 else self.ACCENT_RED
                sq = Square(
                    side_length=cell_size - 0.07,
                    color=col,
                    fill_color=col,
                    fill_opacity=0.75,
                    stroke_width=1.5,
                )
                sq.move_to(pos)
                lbl = Text(
                    "+" if spins[r, c] == 1 else "−",
                    font_size=22,
                    color=self.BG_COLOR,
                    weight=BOLD,
                )
                lbl.move_to(pos)
                row.append(VGroup(sq, lbl))
                grid.add(VGroup(sq, lbl))
            cells.append(row)

        grid.center().shift(DOWN * 0.6)
        self.play(FadeIn(grid))
        self.wait(0.3)

        for _ in range(8):
            r, c = np.random.randint(0, N, size=2)
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    neighbors += spins[nr, nc]
            dE = 2 * spins[r, c] * neighbors
            if dE < 0 or np.random.rand() < np.exp(-dE):
                spins[r, c] *= -1
                new_col = self.ACCENT_TEAL if spins[r, c] == 1 else self.ACCENT_RED
                new_lbl_str = "+" if spins[r, c] == 1 else "−"
                cell = cells[r][c]
                new_sq = Square(
                    side_length=cell_size - 0.07,
                    color=new_col,
                    fill_color=new_col,
                    fill_opacity=0.75,
                    stroke_width=1.5,
                )
                new_sq.move_to(cell[0].get_center())
                new_lbl = Text(
                    new_lbl_str, font_size=22, color=self.BG_COLOR, weight=BOLD
                )
                new_lbl.move_to(cell[0].get_center())
                self.play(
                    Transform(cell[0], new_sq),
                    Transform(cell[1], new_lbl),
                    run_time=0.25,
                )
        self.wait(0.4)

        caption = self._paragraph(
            "The system evolves toward ferromagnetic alignment — the lowest-energy configuration.",
            font_size=21,
            color=self.MUTED_COLOR,
            wrap=60,
        )
        caption.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))
        self.wait(1.5)

    # ── SLIDE 22 — BOLTZMANN MACHINES (concept) ───────────────────────
    def s22_boltzmann_machines(self):
        self._new_slide()
        anchor = self._header("Boltzmann Machines", "THERMODYNAMIC", 22)
        text = self._paragraph(
            "Boltzmann Machines are neural networks built around stochastic "
            "neurons — every neuron updates according to probability.",
            font_size=27,
        )
        self._place(text, anchor, buff=0.6)
        layer1 = VGroup(
            *[Dot(radius=0.14, color=self.ACCENT_ORANGE) for _ in range(3)]
        ).arrange(DOWN, buff=0.5)
        layer2 = VGroup(
            *[Dot(radius=0.14, color=self.ACCENT_TEAL) for _ in range(4)]
        ).arrange(DOWN, buff=0.4)
        layers = VGroup(layer1, layer2).arrange(RIGHT, buff=2.2)
        edges = VGroup()
        for a in layer1:
            for b in layer2:
                edges.add(
                    Line(
                        a.get_center(),
                        b.get_center(),
                        color=self.MUTED_COLOR,
                        stroke_width=1,
                        stroke_opacity=0.5,
                    )
                )
        net = VGroup(edges, layers)
        self._place(net, text, buff=0.6, max_height=2.6)
        caption = self._paragraph(
            "Thermodynamic hardware is naturally suited to this computation.",
            font_size=22,
            color=self.MUTED_COLOR,
            wrap=54,
        )
        caption.next_to(net, DOWN, buff=0.4)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(Create(edges), FadeIn(layers))
        self.play(FadeIn(caption))
        self.wait(1.2)

    # ── SLIDE 23 — BOLTZMANN MACHINE ANIMATION ───────────────────────
    def s23_boltzmann_animation(self):
        self._new_slide()
        anchor = self._header("Boltzmann Machine — Live Network", "ANIMATION", 23)
        intro = self._paragraph(
            "An RBM has visible units (v) and hidden units (h). "
            "At each step, units fire stochastically based on weighted input sums.",
            font_size=24,
        )
        self._place(intro, anchor, buff=0.4)
        self.play(FadeIn(intro, shift=UP * 0.1))

        n_vis, n_hid = 4, 3
        vis_nodes, hid_nodes = VGroup(), VGroup()
        for _ in range(n_vis):
            c = self._neuron(radius=0.26, color=self.ACCENT_ORANGE, fill_opacity=0.2)
            t = Text("v", font_size=17, color=self.ACCENT_ORANGE)
            t.move_to(c.get_center())
            vis_nodes.add(VGroup(c, t))
        for _ in range(n_hid):
            c = self._neuron(radius=0.26, color=self.ACCENT_TEAL, fill_opacity=0.2)
            t = Text("h", font_size=17, color=self.ACCENT_TEAL)
            t.move_to(c.get_center())
            hid_nodes.add(VGroup(c, t))

        vis_nodes.arrange(DOWN, buff=0.45)
        hid_nodes.arrange(DOWN, buff=0.6)
        net_group = VGroup(vis_nodes, hid_nodes).arrange(RIGHT, buff=2.8)
        net_group.next_to(intro, DOWN, buff=0.55).align_to(intro, LEFT)

        edges = VGroup()
        for v in vis_nodes:
            for h in hid_nodes:
                edges.add(
                    Line(
                        v.get_right(),
                        h.get_left(),
                        stroke_width=1.2,
                        color=self.MUTED_COLOR,
                        stroke_opacity=0.4,
                    )
                )

        labels_v = Text("Visible", font_size=20, color=self.ACCENT_ORANGE)
        labels_h = Text("Hidden", font_size=20, color=self.ACCENT_TEAL)
        labels_v.next_to(vis_nodes, UP, buff=0.2)
        labels_h.next_to(hid_nodes, UP, buff=0.2)

        self.play(
            Create(edges),
            FadeIn(vis_nodes),
            FadeIn(hid_nodes),
            FadeIn(labels_v),
            FadeIn(labels_h),
        )
        self.wait(0.3)

        for _ in range(3):
            anims = []
            for h in hid_nodes:
                fire = np.random.rand() > 0.45
                col = self.ACCENT_TEAL if fire else self.MUTED_COLOR
                anims.append(h[0].animate.set_fill(col, opacity=0.75 if fire else 0.1))
            self.play(*anims, run_time=0.6)
            anims = []
            for v in vis_nodes:
                fire = np.random.rand() > 0.45
                col = self.ACCENT_ORANGE if fire else self.MUTED_COLOR
                anims.append(v[0].animate.set_fill(col, opacity=0.75 if fire else 0.1))
            self.play(*anims, run_time=0.6)
        self.wait(0.4)

        caption = self._paragraph(
            "After many stochastic updates the visible layer represents samples from the "
            "learned distribution — directly implementable in thermodynamic hardware.",
            font_size=20,
            color=self.MUTED_COLOR,
            wrap=62,
        )
        caption.next_to(net_group, DOWN, buff=0.45)
        self.play(FadeIn(caption))
        self.wait(1.5)

    # ==================================================================
    # ═══════════════════════  SECTION 4: QUANTUM  ════════════════════
    # ==================================================================

    # ── SLIDE 24 — WHY QUANTUM COMPUTING EXISTS ───────────────────────
    def s24_quantum_computing(self):
        self._new_slide()
        anchor = self._header("Why Quantum Computing Exists", "QUANTUM", 24)
        text = self._paragraph(
            "Quantum computers also perform probabilistic computation, but their "
            "randomness comes from quantum mechanics instead of thermal fluctuations.",
            font_size=27,
        )
        self._place(text, anchor, buff=0.6)
        props = VGroup(
            *[
                self._pill(p, self.ACCENT_BLUE)
                for p in ["Superposition", "Entanglement", "Interference"]
            ]
        ).arrange(RIGHT, buff=0.4)
        props.next_to(text, DOWN, buff=0.65)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(LaggedStart(*[FadeIn(p, scale=0.85) for p in props], lag_ratio=0.2))
        self.wait(1.3)

    # ── SLIDE 25 — BITS, P-BITS, QUBITS ──────────────────────────────
    def s25_bits_pbits_qubits(self):
        self._new_slide()
        anchor = self._header("Bits, p-bits, and Qubits", "QUANTUM", 25)
        cols = VGroup()
        specs = [
            ("Bit", "0 or 1", self.ACCENT_ORANGE),
            ("p-bit", "Mostly 0,\nsometimes 1", self.ACCENT_TEAL),
            ("Qubit", "\u03B1|0\u27E9 + \u03B2|1\u27E9", self.ACCENT_BLUE),
        ]
        for name, val, color in specs:
            title = Text(name, font_size=28, color=color, weight=BOLD)
            box = RoundedRectangle(
                corner_radius=0.12,
                width=3.4,
                height=1.3,
                stroke_color=color,
                stroke_width=2.5,
                fill_color=color,
                fill_opacity=0.1,
            )
            val_txt = Text(val, font_size=24, color=self.TITLE_COLOR)
            if val_txt.width > box.width - 0.4:
                val_txt.set_width(box.width - 0.4)
            val_txt.move_to(box.get_center())
            col_grp = VGroup(title, box, val_txt).arrange(DOWN, buff=0.3)
            val_txt.move_to(box.get_center())
            cols.add(col_grp)
        cols.arrange(RIGHT, buff=0.8, aligned_edge=UP)
        self._place(cols, anchor, buff=0.7)
        caption = self._paragraph(
            "Each represents information differently.",
            font_size=22,
            color=self.MUTED_COLOR,
            wrap=50,
        )
        caption.next_to(cols, DOWN, buff=0.5)
        self.play(
            LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cols], lag_ratio=0.25)
        )
        self.play(FadeIn(caption))
        self.wait(1.3)

    # ── SLIDE 26 — QUANTUM ↔ CLASSICAL CONNECTION ─────────────────────
    def s26_quantum_classical_connection(self):
        self._new_slide()
        anchor = self._header("Quantum ↔ Classical Interface", "QUANTUM", 26)
        intro = self._paragraph(
            "A quantum processor cannot operate in isolation. It is always "
            "orchestrated by a classical host system through multiple layers.",
            font_size=25,
        )
        self._place(intro, anchor, buff=0.45)
        self.play(FadeIn(intro, shift=UP * 0.1))

        import os

        photo_path = "photo.png"
        has_photo = os.path.isfile(photo_path)
        if has_photo:
            photo = ImageMobject(photo_path)
            photo.set_height(2.8)
            photo.next_to(intro, DOWN, buff=0.45).align_to(intro, LEFT)
            self.play(FadeIn(photo))
            next_anchor = photo
        else:
            placeholder = Rectangle(
                width=4.5,
                height=2.6,
                color=self.MUTED_COLOR,
                stroke_width=2,
                fill_color=self.MUTED_COLOR,
                fill_opacity=0.08,
            )
            ph_label = Text("[ photo.png ]", font_size=22, color=self.MUTED_COLOR)
            ph_label.move_to(placeholder.get_center())
            ph_group = VGroup(placeholder, ph_label)
            ph_group.next_to(intro, DOWN, buff=0.45).align_to(intro, LEFT)
            self.play(FadeIn(ph_group))
            next_anchor = ph_group

        layers = self._bullets(
            [
                "Classical host (Linux, scheduling, compilation)",
                "Control electronics (FPGAs, AWGs, signal gen.)",
                "Cryogenic infrastructure (dilution refrigerator at 15 mK)",
                "QPU — quantum processor (superconducting, trapped-ion, photonic …)",
                "Readout electronics → digitised qubit measurement back to host",
            ],
            font_size=21,
            wrap=44,
        )
        layers.next_to(next_anchor, RIGHT, buff=0.6).align_to(next_anchor, UP)
        self.play(LaggedStartMap(FadeIn, layers, shift=RIGHT * 0.1, lag_ratio=0.13))
        self.wait(1.5)

    # ── SLIDE 27 — HARDWARE BEHIND QUANTUM COMPUTING ─────────────────
    def s27_quantum_hardware(self):
        self._new_slide()
        anchor = self._header("Hardware Behind Quantum Computing", "HARDWARE", 27)
        intro = self._paragraph(
            "A quantum computer is much more than a QPU:", font_size=25
        )
        self._place(intro, anchor, buff=0.45)
        flow = self._flow(
            [
                "Classical control server\n(runs OS, schedules jobs)",
                "Control electronics\n(FPGAs, DACs, ADCs)",
                "Transmission lines\n(into the cryostat)",
                "Quantum Processing Unit\n(measures qubits)",
            ],
            direction=DOWN,
            box_width=8.6,
            box_height=0.95,
            font_size=20,
            color=self.ACCENT_BLUE,
        )
        self._place(flow, intro, buff=0.4, max_height=4.3)
        self.play(FadeIn(intro, shift=UP * 0.15))
        self.play(
            LaggedStart(*[FadeIn(b) for b in flow[1]], lag_ratio=0.25),
            LaggedStart(*[Create(a) for a in flow[0]], lag_ratio=0.25),
        )
        self.wait(1.3)

    # ==================================================================
    # ══════════════════  SECTION 5: COMPARISONS  ═════════════════════
    # ==================================================================

    # ── SLIDE 28 — COMPARING PARADIGMS ───────────────────────────────
    def s28_comparing_paradigms(self):
        self._new_slide()
        anchor = self._header("Comparing Computing Paradigms", "COMPARISON", 28)
        headers = ["Property", "Classical", "Thermodynamic", "Quantum"]
        rows = [
            ["Information", "Bit", "p-bit", "Qubit"],
            ["Behavior", "Deterministic", "Probabilistic", "Quantum"],
            ["Noise", "Avoided", "Used", "Coherence"],
            [
                "Best for",
                "Exact computation",
                "Sampling / optimization",
                "Specialized algorithms",
            ],
        ]
        table = self._grid_table(headers, rows, font_size=22)
        self._place(table, anchor, max_height=4.5, buff=0.55)
        self.play(FadeIn(table, shift=UP * 0.1))
        self.wait(1.4)

    # ── SLIDE 29 — QUANTUM vs THERMODYNAMIC ECONOMICS ────────────────
    def s29_quantum_vs_thermo_economics(self):
        self._new_slide()
        anchor = self._header("Quantum vs Thermodynamic — Economics", "COMPARISON", 29)
        headers = ["Factor", "Quantum", "Thermodynamic"]
        rows = [
            ["Operating temperature", "~15 millikelvin", "Room temperature"],
            ["Infrastructure", "Cryostat + dilution fridge", "Standard CMOS fab"],
            ["Cost (prototype)", "$10M – $50M+", "$10K – $1M"],
            ["Qubit/p-bit count (2024)", "~1000 qubits", "Millions of p-bits"],
            ["Error correction", "Active (expensive)", "Statistical (cheap)"],
            ["Target workloads", "Cryptography, QFT", "Sampling, ML, Opt."],
            ["Maturity", "Research / early NISQ", "Early commercial"],
        ]
        table = self._grid_table(headers, rows, font_size=19)
        self._place(table, anchor, max_height=4.7, buff=0.45)
        self.play(FadeIn(table, shift=UP * 0.1))
        self.wait(1.5)

    # ==================================================================
    # ═══════════════  SECTION 6: HARDWARE & IMPLEMENTATION  ══════════
    # ==================================================================

    # ── SLIDE 30 — EXTROPIC'S TSU ─────────────────────────────────────
    def s30_extropic_tsu(self):
        self._new_slide()
        anchor = self._header("Extropic's Thermodynamic Computing", "HARDWARE", 30)
        text = self._paragraph(
            "Extropic proposes a Thermodynamic Processing Unit (TSU). Instead of "
            "eliminating transistor noise, the hardware exploits it.",
            font_size=28,
        )
        self._place(text, anchor, buff=0.6)
        highlight = self._quote(
            "The transistor itself becomes a probabilistic computing element.",
            font_size=27,
        )
        self._place(highlight, text, buff=0.6)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(FadeIn(highlight, shift=UP * 0.1))
        self.wait(1.3)

    # ── SLIDE 31 — HOPFIELD NETWORKS ─────────────────────────────────
    def s31_hopfield_networks(self):
        self._new_slide()
        anchor = self._header("Hopfield Networks", "THERMODYNAMIC", 31)
        intro = self._paragraph(
            "A Hopfield Network is a fully connected recurrent network used as "
            "content-addressable memory. Stored patterns are energy minima.",
            font_size=25,
        )
        self._place(intro, anchor, buff=0.5)
        self.play(FadeIn(intro, shift=UP * 0.1))

        n = 5
        radius = 1.05
        centers = [
            np.array(
                [
                    radius * math.cos(2 * math.pi * i / n - math.pi / 2),
                    radius * math.sin(2 * math.pi * i / n - math.pi / 2),
                    0,
                ]
            )
            for i in range(n)
        ]
        net = VGroup()
        for i, c in enumerate(centers):
            node = self._neuron(radius=0.26, color=self.ACCENT_PURPLE)
            node.move_to(c)
            lbl = Text(f"n{i + 1}", font_size=17, color=self.TITLE_COLOR)
            lbl.move_to(c)
            net.add(VGroup(node, lbl))
        edges = VGroup()
        for i in range(n):
            for j in range(i + 1, n):
                edges.add(
                    Line(
                        centers[i],
                        centers[j],
                        stroke_width=1.2,
                        color=self.MUTED_COLOR,
                        stroke_opacity=0.5,
                    )
                )
        network = VGroup(edges, net)
        network.next_to(intro, DOWN, buff=0.5).align_to(intro, LEFT)
        self._place(network, intro, buff=0.5, max_height=2.8, max_width=4.5)

        props = self._bullets(
            [
                "Stores patterns as attractors",
                "Corrupted input converges to nearest stored pattern",
                "Energy decreases at each update step",
                "Modern variant (Dense Associative Memory) scales to 2^n patterns",
            ],
            font_size=22,
            wrap=36,
        )
        props.next_to(intro, DOWN, buff=0.5).to_edge(RIGHT, buff=1.0)

        self.play(Create(edges), FadeIn(net))
        self.play(LaggedStartMap(FadeIn, props, shift=RIGHT * 0.1, lag_ratio=0.15))

        noisy = SurroundingRectangle(
            net[2], color=self.ACCENT_RED, stroke_width=2, buff=0.06
        )
        conv = SurroundingRectangle(
            net[2], color=self.ACCENT_GREEN, stroke_width=2, buff=0.06
        )
        self.play(Create(noisy))
        self.wait(0.4)
        self.play(Transform(noisy, conv))
        self.wait(0.4)
        self.play(FadeOut(noisy))

        caption = self._paragraph(
            "Used in AI for pattern completion and error correction.",
            font_size=21,
            color=self.MUTED_COLOR,
            wrap=56,
        )
        caption.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))
        self.wait(1.5)

    # ── SLIDE 32 — DENOISING THERMODYNAMIC MODELS ────────────────────
    def s32_denoising_models(self):
        self._new_slide()
        anchor = self._header("Denoising Thermodynamic Models", "THERMODYNAMIC", 32)
        text = self._paragraph(
            "Instead of removing noise digitally, the hardware evolves noisy "
            "states until a stable solution emerges.",
            font_size=27,
        )
        self._place(text, anchor, buff=0.6)
        flow = self._flow(
            ["Noisy state", "Evolve", "Denoise", "Stable solution"],
            direction=RIGHT,
            box_width=2.8,
            box_height=1.0,
            font_size=21,
            color=self.ACCENT_TEAL,
        )
        self._place(flow, text, buff=0.6)
        caption = self._paragraph(
            "Conceptually similar to diffusion models, but implemented through "
            "stochastic physical dynamics.",
            font_size=22,
            color=self.MUTED_COLOR,
            wrap=58,
        )
        caption.next_to(flow, DOWN, buff=0.5)
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(
            LaggedStart(*[FadeIn(b) for b in flow[1]], lag_ratio=0.2),
            LaggedStart(*[Create(a) for a in flow[0]], lag_ratio=0.2),
        )
        self.play(FadeIn(caption))
        self.wait(1.2)

    # ── SLIDE 33 — FPGA PROTOTYPE ─────────────────────────────────────
    def s33_fpga_prototype(self):
        self._new_slide()
        anchor = self._header(
            "Can an FPGA Become a Thermodynamic Computer?", "HARDWARE", 33
        )
        verdict = VGroup(self._pill("Short answer: No", self.ACCENT_RED))
        self._place(verdict, anchor, buff=0.5)
        text = self._paragraph(
            "An FPGA cannot reproduce the analog transistor physics of a true TSU. "
            "However, it CAN accurately emulate thermodynamic algorithms and architectures:",
            font_size=25,
        )
        text.next_to(verdict, DOWN, buff=0.45)
        items = self._bullets(
            [
                "Digital p-bits",
                "Gibbs Sampling",
                "Energy-Based Models",
                "Ising Models",
                "Hopfield Networks",
                "Boltzmann Machines",
            ],
            font_size=22,
            wrap=40,
        )
        items.next_to(text, DOWN, buff=0.35, aligned_edge=LEFT)
        whole = VGroup(verdict, text, items)
        self._place(whole, anchor, buff=0.5, max_height=4.9)
        self.play(FadeIn(verdict, scale=0.9))
        self.play(FadeIn(text, shift=UP * 0.15))
        self.play(LaggedStartMap(FadeIn, items, shift=RIGHT * 0.1, lag_ratio=0.1))
        self.wait(1.3)

    # ==================================================================
    # ═══════════════════  SECTION 7: AI DRIVERS  ═════════════════════
    # ==================================================================

    # ── SLIDE 34 — DEEPSEEK & AI DRIVERS ─────────────────────────────
    def s34_deepseek_drivers(self):
        self._new_slide()
        anchor = self._header("DeepSeek & the AI Hardware Race", "DRIVERS", 34)
        intro = self._paragraph(
            "DeepSeek's R1 and V3 models demonstrated that aggressive "
            "algorithmic efficiency can match frontier models at a fraction "
            "of the compute cost — reshaping hardware roadmaps.",
            font_size=25,
        )
        self._place(intro, anchor, buff=0.5)
        self.play(FadeIn(intro, shift=UP * 0.1))
        bullets = self._bullets(
            [
                "DeepSeek V3 trained on ~$6M of H800 compute vs GPT-4's estimated $100M+",
                "Mixture-of-Experts (MoE) activates only relevant parameters per token",
                "Multi-head Latent Attention compresses the KV-cache dramatically",
                "Demonstrated that algorithmic efficiency can outrun raw hardware scaling",
                "Revived interest in alt substrates: thermodynamic, analog, in-memory",
            ],
            font_size=22,
            wrap=56,
        )
        bullets.next_to(intro, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(LaggedStartMap(FadeIn, bullets, shift=RIGHT * 0.1, lag_ratio=0.12))
        note = self._paragraph(
            "Key lesson: the right algorithm + architecture may matter more than sheer FLOPS.",
            font_size=22,
            color=self.ACCENT_TEAL,
            wrap=60,
        )
        note.next_to(bullets, DOWN, buff=0.4)
        self.play(FadeIn(note))
        self.wait(1.5)

    # ── SLIDE 35 — CHARACTER DRIVERS ─────────────────────────────────
    def s35_character_drivers(self):
        self._new_slide()
        anchor = self._header("Character Drivers — Deep Dive", "DRIVERS", 35)
        intro = self._paragraph(
            "A Linux character driver registers itself with the kernel through a "
            "set of file_operations callbacks.",
            font_size=25,
        )
        self._place(intro, anchor, buff=0.5)
        self.play(FadeIn(intro, shift=UP * 0.1))
        headers = ["Callback", "Triggered by", "Typical action"]
        rows = [
            [".open", "open() syscall", "Allocate state, check permissions"],
            [".release", "close() syscall", "Free resources, flush buffers"],
            [".read", "read() syscall", "Copy data from device to user-space"],
            [".write", "write() syscall", "Copy data from user-space to device"],
            [".ioctl", "ioctl() syscall", "Device-specific control commands"],
            [".poll", "select/poll/epoll", "Signal readability / writability"],
            [".mmap", "mmap() syscall", "Map device memory into user VA space"],
        ]
        table = self._grid_table(headers, rows, font_size=19)
        self._place(table, intro, buff=0.45, max_height=3.8)
        self.play(FadeIn(table, shift=UP * 0.1))
        footer = self._paragraph(
            "The driver also registers a major/minor number pair via alloc_chrdev_region() "
            "and exposes the device via cdev_add().",
            font_size=20,
            color=self.MUTED_COLOR,
            wrap=62,
        )
        footer.next_to(table, DOWN, buff=0.35)
        self.play(FadeIn(footer))
        self.wait(1.5)

    # ==================================================================
    # ═══════════════════  SECTION 8: CONCLUSION  ═════════════════════
    # ==================================================================

    # ── SLIDE 36 — FPGA DEVELOPMENT FLOW ─────────────────────────────
    def s36_fpga_dev_flow(self):
        self._new_slide()
        anchor = self._header("FPGA Development Flow", "HARDWARE", 36)
        flow = self._flow(
            [
                "Algorithm",
                "Python / JAX",
                "RTL / Verilog",
                "FPGA prototype",
                "Verification",
                "ASIC",
                "Thermodynamic ASIC",
            ],
            direction=RIGHT,
            box_width=1.75,
            box_height=1.4,
            font_size=16,
            color=self.ACCENT_TEAL,
        )
        self._place(
            flow, anchor, buff=0.9, max_width=self.CONTENT_WIDTH + 0.4, max_height=2.0
        )
        caption = self._paragraph(
            "The FPGA validates the architecture before fabricating custom thermodynamic silicon.",
            font_size=24,
            color=self.MUTED_COLOR,
            wrap=56,
        )
        caption.next_to(flow, DOWN, buff=0.7)
        self.play(
            LaggedStart(*[FadeIn(b) for b in flow[1]], lag_ratio=0.15),
            LaggedStart(*[Create(a) for a in flow[0]], lag_ratio=0.15),
        )
        self.play(FadeIn(caption))
        self.wait(1.3)

    # ── SLIDE 37 — CONCLUSION ─────────────────────────────────────────
    def s37_conclusion(self):
        self._new_slide()
        anchor = self._header("Conclusion", "FUTURE", 37)
        paradigms = VGroup(
            self._pill("Classical — exact arithmetic", self.ACCENT_ORANGE),
            self._pill("Quantum — quantum-native algorithms", self.ACCENT_BLUE),
            self._pill("Thermodynamic — sampling & optimization", self.ACCENT_TEAL),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        self._place(paradigms, anchor, buff=0.6)
        closing = self._quote(
            "Noise, once considered the enemy of computation, may become "
            "a computational resource.",
            font_size=28,
        )
        self._place(closing, paradigms, buff=0.65)
        self.play(
            LaggedStart(
                *[FadeIn(p, shift=RIGHT * 0.15) for p in paradigms], lag_ratio=0.25
            )
        )
        self.play(FadeIn(closing, shift=UP * 0.15))
        self.wait(2)
        self._next_slide()  # final pause so presenter can linger on last slide
