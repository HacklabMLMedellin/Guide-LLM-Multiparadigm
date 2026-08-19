from manim import *
from manim_slides import Slide
import numpy as np

config.background_color = "#08080c"

WHITE_T = "#f2f2f5"
GRAY_T = "#7d8290"
DIVIDER_C = "#33353f"
GASOLINE = "#e0483e"
ELECTRIC = "#2dd4bf"
DIESEL = "#9aa0ab"
NATGAS = "#4db8ff"
CLASSICAL = "#3b82f6"
QUANTUM = "#5bad1e"
PHOTONIC = "#f0ab00"
THERMO = "#e8690a"
ACCENT = "#a855f7"

LEFT_X = -3.4
RIGHT_X = 3.4


# ── text helpers ──────────────────────────────────────────────────────────────
def title_text(text, color=WHITE_T, size=34):
    return Text(text, font_size=size, color=color, weight="BOLD")


def caption_text(text, color=WHITE_T, size=22):
    return Text(text, font_size=size, color=color, weight="BOLD")


def note_text(text, color=GRAY_T, size=17):
    return Text(text, font_size=size, color=color)


# ── image helpers ─────────────────────────────────────────────────────────────
IMG = "img"  # folder containing all PNGs


def img_mob(filename, height=1.0):
    """Load img/<filename>.png, scale to *height* scene-units."""
    m = ImageMobject(f"{IMG}/{filename}.png")
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.height = height
    return m


# Convenience wrappers that match the old symbol API (return a single Mobject).
def car_gasoline(height=1.0):
    return img_mob("gasolin_car", height)


def car_electric(height=1.0):
    return img_mob("electric_car", height)


def car_diesel(height=1.0):
    return img_mob("disel_car", height)


def car_gas(height=1.0):
    return img_mob("gas_car", height)


def car_hybrid(height=1.0):
    return img_mob("hybrid_car", height)


def img_city(height=0.9):
    return img_mob("city", height)


def img_highway(height=0.9):
    return img_mob("highway", height)


def img_mountain(height=0.9):
    return img_mob("mountain", height)


def img_heavycargo(height=0.9):
    return img_mob("heavycargo", height)


def qpt_logo(height=1.2):
    return img_mob("qpt_transparent_logo", height)


# ── badge (image version) ─────────────────────────────────────────────────────
def fit(symbol, radius):
    if symbol.width > radius * 1.2:
        symbol.scale_to_fit_width(radius * 1.2)
    if symbol.height > radius * 1.2:
        symbol.scale_to_fit_height(radius * 1.2)
    return symbol


def badge_img(image_mob, color, text, radius=0.6, size=18):
    ring = Circle(
        radius=radius, color=color, stroke_width=3, fill_color=color, fill_opacity=0.08
    )
    fit(image_mob, radius)
    image_mob.move_to(ring.get_center())
    tag = note_text(text, color=WHITE_T, size=size).next_to(ring, DOWN, buff=0.24)
    return Group(ring, image_mob, tag)


# Keep vector-only badges for the compute paradigms (unchanged).
def bolt_symbol(color=WHITE_T):
    pts = [
        [0.1, 0.5, 0.0],
        [-0.16, 0.02, 0.0],
        [0.02, 0.02, 0.0],
        [-0.1, -0.5, 0.0],
        [0.18, -0.02, 0.0],
        [0.0, -0.02, 0.0],
        [0.1, 0.5, 0.0],
    ]
    bolt = VMobject(stroke_width=0, fill_color=color, fill_opacity=0.95)
    bolt.set_points_as_corners([np.array(p) for p in pts])
    return bolt


def cpu_symbol(color=WHITE_T):
    chip = Square(
        side_length=0.5,
        stroke_width=2.5,
        color=color,
        fill_color=color,
        fill_opacity=0.12,
    )
    pins = Group()
    for i in (-1, 0, 1):
        pins.add(
            Line(
                chip.get_top() + RIGHT * i * 0.15,
                chip.get_top() + RIGHT * i * 0.15 + UP * 0.12,
                stroke_width=2.5,
                color=color,
            )
        )
        pins.add(
            Line(
                chip.get_bottom() + RIGHT * i * 0.15,
                chip.get_bottom() + RIGHT * i * 0.15 + DOWN * 0.12,
                stroke_width=2.5,
                color=color,
            )
        )
        pins.add(
            Line(
                chip.get_left() + UP * i * 0.15,
                chip.get_left() + UP * i * 0.15 + LEFT * 0.12,
                stroke_width=2.5,
                color=color,
            )
        )
        pins.add(
            Line(
                chip.get_right() + UP * i * 0.15,
                chip.get_right() + UP * i * 0.15 + RIGHT * 0.12,
                stroke_width=2.5,
                color=color,
            )
        )
    return Group(chip, pins)


def quantum_symbol(color=QUANTUM):
    a = Dot(radius=0.075, color=color).shift(LEFT * 0.26)
    b = Dot(radius=0.075, color=color).shift(RIGHT * 0.26)
    rod = Line(a.get_center(), b.get_center(), color=color, stroke_width=2)
    ring_a = Circle(
        radius=0.17, color=color, stroke_width=1.5, stroke_opacity=0.6
    ).move_to(a)
    ring_b = Circle(
        radius=0.17, color=color, stroke_width=1.5, stroke_opacity=0.6
    ).move_to(b)
    group = Group(rod, ring_a, ring_b, a, b)
    group.a_dot = a
    group.b_dot = b
    return group


def photonic_symbol(color=PHOTONIC):
    pts = []
    for i in range(6):
        x = -0.4 + i * 0.16
        y = 0.2 if i % 2 == 0 else -0.2
        pts.append(np.array([x, y, 0.0]))
    wave = VMobject(color=color, stroke_width=2.5)
    wave.set_points_as_corners(pts)
    dot = Dot(radius=0.05, color="#ffffff").move_to(pts[0])
    group = Group(wave, dot)
    group.path = wave
    group.pulse = dot
    return group


def thermo_symbol(color=THERMO):
    f = lambda x: 0.34 * x**2 - 0.18
    pts = [np.array([x, f(x), 0.0]) for x in np.linspace(-0.5, 0.5, 24)]
    curve = VMobject(color=color, stroke_width=2.5)
    curve.set_points_smoothly(pts)
    dot = Dot(radius=0.06, color="#ffd27a").move_to(pts[0])
    group = Group(curve, dot)
    group.curve = curve
    group.walker = dot
    return group


def badge(symbol, color, text, radius=0.6, size=18):
    ring = Circle(
        radius=radius, color=color, stroke_width=3, fill_color=color, fill_opacity=0.08
    )
    symbol.set_color(color)
    fit(symbol, radius)
    symbol.move_to(ring.get_center())
    tag = note_text(text, color=WHITE_T, size=size).next_to(ring, DOWN, buff=0.24)
    return Group(ring, symbol, tag)


# ── layout helpers ─────────────────────────────────────────────────────────────
def router_symbol(color=WHITE_T):
    return Square(
        side_length=0.46,
        stroke_width=2.5,
        color=color,
        fill_color=color,
        fill_opacity=0.1,
    ).rotate(PI / 4)


def person_symbol(color=WHITE_T):
    head = Circle(
        radius=0.12, stroke_width=0, fill_color=color, fill_opacity=0.95
    ).shift(UP * 0.26)
    body = (
        Triangle(stroke_width=0, fill_color=color, fill_opacity=0.95)
        .scale(0.26)
        .shift(DOWN * 0.08)
    )
    return VGroup(head, body)


def edge_point(m, direction):
    if direction[0] > 0.5:
        return m.get_right()
    if direction[0] < -0.5:
        return m.get_left()
    if direction[1] < -0.5:
        return m.get_bottom()
    if direction[1] > 0.5:
        return m.get_top()
    return m.get_center()


def flow_chain(labels, color=WHITE_T, size=16, direction=RIGHT, buff=0.55):
    cells = VGroup()
    for lab in labels:
        t = note_text(lab, color=color, size=size)
        box = SurroundingRectangle(t, color=color, stroke_width=1.5, buff=0.14)
        cells.add(VGroup(box, t))
    cells.arrange(direction, buff=buff)
    arrows = VGroup()
    for i in range(len(cells) - 1):
        arrows.add(
            Arrow(
                edge_point(cells[i], direction),
                edge_point(cells[i + 1], -direction),
                buff=0.05,
                color=GRAY_T,
                stroke_width=2.5,
                max_tip_length_to_length_ratio=0.35,
            )
        )
    return VGroup(cells, arrows), cells, arrows


def attr_row(text, color=WHITE_T, size=17):
    dot = Dot(radius=0.035, color=color)
    txt = note_text(text, color=WHITE_T, size=size)
    return Group(dot, txt).arrange(RIGHT, buff=0.14)


def attr_list(items, color=WHITE_T):
    return Group(*[attr_row(i, color=color) for i in items]).arrange(
        DOWN, aligned_edge=LEFT, buff=0.16
    )


# ─────────────────────────────────────────────────────────────────────────────
class QPTStory(Slide):
    def construct(self):
        self.divider = None
        self.left_title = None
        self.right_title = None
        self.persistent = []
        self.intro_question()
        self.split_screen_intro()
        self.automotive_gasoline()
        self.automotive_electric()
        self.automotive_diesel()
        self.automotive_natgas()
        self.automotive_insight()
        self.transition_to_ai()
        self.classical_computing()
        self.quantum_computing()
        self.photonic_computing()
        self.thermodynamic_computing()
        self.qpt_architecture()
        self.parallel_analogy()
        self.final_scene()

    def clear_stage(self, run_time=0.5):
        to_fade = [m for m in self.mobjects if m not in self.persistent]
        if to_fade:
            self.play(*[FadeOut(m) for m in to_fade], run_time=run_time)

    def intro_question(self):
        q = title_text("Can one engine be optimal for everything?", size=32).to_edge(
            UP, buff=1.0
        )
        self.play(Write(q, run_time=1.6))
        self.next_slide()
        # Use Group (not VGroup) because ImageMobject is not a VMobject
        situations = Group(
            Group(img_city(0.65), note_text("CITY", size=15)).arrange(DOWN, buff=0.15),
            Group(img_highway(0.65), note_text("HIGHWAY", size=15)).arrange(
                DOWN, buff=0.15
            ),
            Group(img_heavycargo(0.65), note_text("HEAVY CARGO", size=15)).arrange(
                DOWN, buff=0.15
            ),
            Group(img_mountain(0.65), note_text("MOUNTAIN", size=15)).arrange(
                DOWN, buff=0.15
            ),
        )
        situations.arrange(RIGHT, buff=0.9).next_to(q, DOWN, buff=1.1)
        self.play(
            LaggedStart(
                *[FadeIn(s, shift=UP * 0.15) for s in situations], lag_ratio=0.2
            )
        )
        self.next_slide()
        self.play(FadeOut(q), FadeOut(situations))

    def split_screen_intro(self):
        self.divider = DashedLine(
            UP * 3.4, DOWN * 3.4, color=DIVIDER_C, stroke_width=1.5, dash_length=0.12
        )
        self.left_title = title_text("TRANSPORTATION", size=24).move_to(
            [LEFT_X, 3.0, 0]
        )
        self.right_title = title_text("COMPUTATION", size=24).move_to([RIGHT_X, 3.0, 0])
        self.play(Create(self.divider), run_time=0.8)
        self.play(
            FadeIn(self.left_title, shift=DOWN * 0.15),
            FadeIn(self.right_title, shift=DOWN * 0.15),
        )
        self.persistent = [self.divider, self.left_title, self.right_title]
        self.next_slide()

    def stage(self, left_group, right_group, run_time=1.0):
        self.clear_stage(run_time=run_time * 0.5)
        self.play(
            FadeIn(left_group, shift=UP * 0.2),
            FadeIn(right_group, shift=UP * 0.2),
            run_time=run_time,
        )

    # ── automotive slides ─────────────────────────────────────────────────────

    def automotive_gasoline(self):
        car_img = car_gasoline(height=1.1)
        b = badge_img(car_img, GASOLINE, "GASOLINE", radius=0.65)
        b.move_to([LEFT_X, 1.3, 0])
        attrs = attr_list(
            ["Mature technology", "Powerful", "Widely available", "Can be expensive"],
            GASOLINE,
        )
        attrs.next_to(b, DOWN, buff=0.4)
        left_group = Group(b, attrs)  # ← Group, not VGroup
        self.stage(left_group, Group())  # ← Group, not VGroup (if stage accepts it)

        # animated car driving across the bottom
        anim_car = car_gasoline(height=0.75)
        anim_car.next_to(attrs, DOWN, buff=0.4).set_x(LEFT_X - 1.2)
        self.play(FadeIn(anim_car))
        self.play(anim_car.animate.shift(RIGHT * 2.4), run_time=1.0, rate_func=smooth)
        self.next_slide()

    def automotive_electric(self):
        car_img = car_electric(height=1.1)
        b = badge_img(car_img, ELECTRIC, "ELECTRIC", radius=0.65)
        b.move_to([LEFT_X, 1.3, 0])
        attrs = attr_list(
            [
                "Simple drive",
                "Efficient",
                "Easy energy conversion",
                "Lower operating cost",
            ],
            ELECTRIC,
        )
        attrs.next_to(b, DOWN, buff=0.4)
        left_group = Group(b, attrs)
        self.stage(left_group, Group())

        slope = img_mountain(height=1.1).set_x(LEFT_X).next_to(attrs, DOWN, buff=0.45)
        anim_car = car_electric(height=0.75).move_to(slope.get_left() + DOWN * 0.08)
        self.play(FadeIn(slope), FadeIn(anim_car))
        self.play(
            anim_car.animate.move_to(slope.get_top() + DOWN * 0.12),
            run_time=1.6,
            rate_func=lambda t: t**1.6,
        )
        note = note_text(
            "Different power systems have different strengths and constraints.", size=14
        )
        note.next_to(slope, DOWN, buff=0.3)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.next_slide()

    def automotive_diesel(self):
        car_img = car_diesel(height=1.1)
        b = badge_img(car_img, DIESEL, "DIESEL", radius=0.65)
        b.move_to([LEFT_X, 1.3, 0])
        arrow_label = note_text("HIGH TORQUE  →  HEAVY LOADS", color=DIESEL, size=16)
        arrow_label.next_to(b, DOWN, buff=0.4)
        left_group = Group(b, arrow_label)
        self.stage(left_group, Group())
        cargo_img = img_heavycargo(height=0.95).move_to([LEFT_X, -1.4, 0])
        self.play(FadeIn(cargo_img, shift=DOWN * 0.1))
        caption = note_text("Powerful, but higher emissions.", size=15)
        caption.next_to(cargo_img, DOWN, buff=0.45)
        self.play(FadeIn(caption))
        self.next_slide()

    def automotive_natgas(self):
        car_img = car_gas(height=1.1)
        b = badge_img(car_img, NATGAS, "NATURAL GAS", radius=0.65)
        b.move_to([LEFT_X, 1.0, 0])
        attrs = attr_list(
            [
                "Lower operating cost",
                "Different combustion profile",
                "Potentially lower wear",
            ],
            NATGAS,
        )
        attrs.next_to(b, DOWN, buff=0.5)
        left_group = Group(b, attrs)
        self.stage(left_group, Group())
        self.next_slide()

    def automotive_insight(self):
        self.clear_stage()
        # central hybrid car image
        center_car = car_hybrid(height=1.5).move_to([LEFT_X, 0.4, 0])
        # four fuel-type badges with real car images
        badges = Group(
            badge_img(car_gasoline(0.55), GASOLINE, "GASOLINE", radius=0.4, size=13),
            badge_img(car_electric(0.55), ELECTRIC, "ELECTRIC", radius=0.4, size=13),
            badge_img(car_diesel(0.55), DIESEL, "DIESEL", radius=0.4, size=13),
            badge_img(car_gas(0.55), NATGAS, "GAS", radius=0.4, size=13),
        )
        positions = [UP * 1.4, RIGHT * 1.6, DOWN * 1.4, LEFT * 1.6]
        for bge, pos in zip(badges, positions):
            bge.move_to(center_car.get_center() + pos)
        self.play(FadeIn(center_car))
        self.play(LaggedStart(*[FadeIn(b, scale=0.6) for b in badges], lag_ratio=0.2))
        self.next_slide()

        rules = (
            Group(
                note_text("CITY → ELECTRIC", color=ELECTRIC, size=15),
                note_text("HIGHWAY → GASOLINE", color=GASOLINE, size=15),
                note_text("HEAVY LOAD → DIESEL", color=DIESEL, size=15),
                note_text("OTHER → NATURAL GAS", color=NATGAS, size=15),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            .move_to([LEFT_X, -2.1, 0])
        )
        self.play(FadeIn(rules, shift=UP * 0.15))
        self.next_slide()

        # --- NEW SLIDE: clear everything before the flow chain ---
        self.play(FadeOut(rules), FadeOut(center_car), FadeOut(badges))
        self.clear_stage()
        self.next_slide()

        chain, cells, arrows = flow_chain(
            [
                "ONE DRIVER",
                "MULTIPLE POWER SOURCES",
                "AUTOMATIC SELECTION",
                "MORE EFFICIENT SYSTEM",
            ],
            color=WHITE_T,
            size=13,
            direction=DOWN,
            buff=0.35,
        )

        chain.scale(1.25).move_to([LEFT_X, 0.9, 0])
        self.play(Create(chain), run_time=1.4)
        self.next_slide()

    # ── compute slides ────────────────────────────────────────────────────────

    def transition_to_ai(self):
        self.clear_stage()
        engine_word = title_text("ENGINE", size=30).move_to([LEFT_X, 0.4, 0])
        self.play(FadeIn(engine_word))
        comp_word = title_text("COMPUTATION", size=30).move_to([LEFT_X, 0.4, 0])
        self.play(FadeTransform(engine_word, comp_word))
        self.next_slide()
        arrow_text = note_text("ONE ENGINE  →  ONE COMPUTATIONAL PARADIGM", size=16)
        arrow_text.next_to(comp_word, DOWN, buff=0.5)
        self.play(FadeIn(arrow_text, shift=UP * 0.1))
        self.next_slide()
        question = title_text("What if AI worked the same way?", size=26).move_to(
            ORIGIN
        )
        self.play(
            FadeOut(comp_word), FadeOut(arrow_text), Write(question, run_time=1.4)
        )
        self.next_slide()
        self.play(FadeOut(question))

    def classical_computing(self):
        b = badge(cpu_symbol(), CLASSICAL, "CLASSICAL", radius=0.65).move_to(
            [RIGHT_X, 1.2, 0]
        )
        chain, cells, arrows = flow_chain(
            ["CPU/GPU", "Neural Net", "LLM", "User"],
            color=CLASSICAL,
            size=13,
            direction=DOWN,
            buff=0.32,
        )
        chain.scale(0.85).next_to(b, DOWN, buff=0.4)
        attrs = attr_list(["Mature", "General purpose", "Available today"], CLASSICAL)
        attrs.scale(0.85).next_to(chain, DOWN, buff=0.3)
        self.stage(Group(), Group(b, chain, attrs))
        self.next_slide()

    def quantum_computing(self):
        b = badge(quantum_symbol(), QUANTUM, "QUANTUM", radius=0.65).move_to(
            [RIGHT_X, 1.4, 0]
        )
        labels = (
            Group(
                note_text("HIGH-DIMENSIONAL SPACES", color=QUANTUM, size=14),
                note_text("QUANTUM STATES", color=QUANTUM, size=14),
                note_text("SPECIALIZED PROBLEMS", color=QUANTUM, size=14),
            )
            .arrange(DOWN, buff=0.14)
            .next_to(b, DOWN, buff=0.45)
        )
        right_group = Group(b, labels)
        self.stage(Group(), right_group)
        q = right_group[0][1]
        self.play(
            Rotate(
                q, angle=PI, about_point=q.get_center(), run_time=1.6, rate_func=smooth
            )
        )
        self.next_slide()
        warn = note_text("NOT A UNIVERSAL REPLACEMENT", color=GRAY_T, size=14)
        warn.next_to(labels, DOWN, buff=0.35)
        self.play(FadeIn(warn, shift=UP * 0.1))
        self.next_slide()

    def photonic_computing(self):
        b = badge(photonic_symbol(), PHOTONIC, "PHOTONIC", radius=0.65).move_to(
            [RIGHT_X, 1.4, 0]
        )
        header = note_text("LIGHT → COMPUTATION", color=PHOTONIC, size=15).next_to(
            b, DOWN, buff=0.35
        )
        right_group = Group(b, header)
        self.stage(Group(), right_group)
        wave_symbol = right_group[0][1]
        self.play(
            MoveAlongPath(
                wave_symbol.pulse, wave_symbol.path, run_time=1.4, rate_func=linear
            )
        )
        self.next_slide()
        labels = (
            Group(
                note_text("FAST", color=PHOTONIC, size=14),
                note_text("PARALLEL", color=PHOTONIC, size=14),
                note_text("EMERGING", color=PHOTONIC, size=14),
            )
            .arrange(RIGHT, buff=0.5)
            .next_to(header, DOWN, buff=0.35)
        )
        self.play(
            LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in labels], lag_ratio=0.2)
        )
        self.next_slide()

    def thermodynamic_computing(self):
        b = badge(thermo_symbol(), THERMO, "THERMODYNAMIC", radius=0.65).move_to(
            [RIGHT_X, 1.5, 0]
        )
        right_group = Group(b)
        self.stage(Group(), right_group)
        t_symbol = right_group[0][1]
        descent_pts = [
            t_symbol.curve.point_from_proportion(p)
            for p in [0.0, 0.32, 0.16, 0.6, 0.85, 1.0]
        ]
        path = VMobject().set_points_smoothly(descent_pts)
        self.play(MoveAlongPath(t_symbol.walker, path, run_time=1.6, rate_func=smooth))
        self.next_slide()
        chain, cells, arrows = flow_chain(
            [
                "ENERGY LANDSCAPE",
                "PROBABILITY DISTRIBUTION",
                "OPTIMIZATION",
                "LEARNING",
            ],
            color=THERMO,
            size=13,
            direction=DOWN,
            buff=0.25,
        )
        chain.scale(0.8).next_to(b, DOWN, buff=0.4)
        self.play(Create(chain), run_time=1.2)
        note = note_text("Computation inspired by physical dynamics", size=13)
        note.next_to(chain, DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.next_slide()

    def qpt_architecture(self):
        self.clear_stage()
        center = qpt_logo().move_to([RIGHT_X, 0, 0])
        center_label = note_text("QPT-LLM", color=WHITE_T, size=15).next_to(
            center, DOWN, buff=0.55
        )
        engines = Group(
            badge(cpu_symbol(), CLASSICAL, "CLASSICAL", radius=0.4, size=12),
            badge(quantum_symbol(), QUANTUM, "QUANTUM", radius=0.4, size=12),
            badge(photonic_symbol(), PHOTONIC, "PHOTONIC", radius=0.4, size=12),
            badge(thermo_symbol(), THERMO, "THERMO", radius=0.4, size=12),
        )
        offsets = [
            UP * 1.5 + LEFT * 1.3,
            UP * 1.5 + RIGHT * 1.3,
            DOWN * 1.5 + LEFT * 1.3,
            DOWN * 1.5 + RIGHT * 1.3,
        ]
        for e, off in zip(engines, offsets):
            e.move_to(center.get_center() + off)
        self.play(FadeIn(center), FadeIn(center_label))
        self.play(LaggedStart(*[FadeIn(e, scale=0.7) for e in engines], lag_ratio=0.15))
        # Use VGroup because Line is a VMobject and Create requires VMobject
        links = VGroup(
            *[
                Line(
                    center.get_center(),
                    e[0].get_center(),
                    color=GRAY_T,
                    stroke_width=1.5,
                    stroke_opacity=0.6,
                )
                for e in engines
            ]
        )
        self.play(Create(links), run_time=1.0)
        self.next_slide()
        criteria = Group(
            *[
                note_text(c, color=GRAY_T, size=12)
                for c in [
                    "SPEED",
                    "MEMORY",
                    "ENERGY",
                    "PRECISION",
                    "SCALABILITY",
                    "HARDWARE",
                ]
            ]
        )
        criteria.arrange(DOWN, buff=0.1).next_to(center, LEFT, buff=1.1)
        self.play(
            LaggedStart(
                *[FadeIn(c, shift=RIGHT * 0.1) for c in criteria], lag_ratio=0.15
            )
        )
        self.next_slide()
        self.play(
            LaggedStart(
                *[Indicate(link, color=WHITE_T, scale_factor=1.0) for link in links],
                lag_ratio=0.25,
                run_time=1.6,
            )
        )
        self.next_slide()

    def parallel_analogy(self):
        self.clear_stage()
        left_chain, _, _ = flow_chain(
            ["CAR", "Gasoline / Electric / Diesel / Gas", "Intelligent Power Mgmt"],
            color=WHITE_T,
            size=14,
            direction=DOWN,
            buff=0.4,
        )
        left_chain.scale(0.85).move_to([LEFT_X, 0.3, 0])
        right_chain, _, _ = flow_chain(
            [
                "LLM",
                "Classical / Quantum / Photonic / Thermo",
                "Intelligent Compute Mgmt",
            ],
            color=WHITE_T,
            size=14,
            direction=DOWN,
            buff=0.4,
        )
        right_chain.scale(0.85).move_to([RIGHT_X, 0.3, 0])
        self.play(
            FadeIn(left_chain, shift=UP * 0.2), FadeIn(right_chain, shift=UP * 0.2)
        )
        self.next_slide()
        equiv = (
            Group(
                note_text("MULTI-POWER VEHICLE", color=WHITE_T, size=18),
                note_text("≈", color=ACCENT, size=26),
                note_text("MULTI-PARADIGM AI", color=WHITE_T, size=18),
            )
            .arrange(DOWN, buff=0.25)
            .move_to(DOWN * 2.6)
        )
        self.play(FadeIn(equiv, shift=UP * 0.2))
        self.next_slide()
        self.play(FadeOut(equiv))

    def final_scene(self):
        self.clear_stage()
        self.play(
            FadeOut(self.left_title), FadeOut(self.right_title), FadeOut(self.divider)
        )
        self.persistent = []

        final1 = caption_text("ONE INTERFACE.", size=26).move_to(UP * 1.4)
        final2 = caption_text("MULTIPLE COMPUTATIONAL PARADIGMS.", size=22).next_to(
            final1, DOWN, buff=0.3
        )
        self.play(Write(final1, run_time=1.0))
        self.play(Write(final2, run_time=1.2))
        self.next_slide()

        # QPT logo image on the final slide
        logo = qpt_logo(height=1.5).move_to(UP * 0.5)
        qpt_sub = note_text("Quantum · Photonic · Thermodynamic", size=18).next_to(
            logo, DOWN, buff=0.3
        )
        foundation_label = note_text(
            "Classical Computing — foundational layer", color=CLASSICAL, size=15
        )
        foundation = SurroundingRectangle(
            foundation_label, color=CLASSICAL, stroke_width=1.5, buff=0.18
        )
        foundation_label.move_to(foundation.get_center())
        foundation_group = Group(foundation, foundation_label).next_to(
            qpt_sub, DOWN, buff=0.6
        )

        self.play(FadeOut(final1), FadeOut(final2))
        self.play(FadeIn(logo, scale=0.8), FadeIn(qpt_sub, shift=UP * 0.1))
        self.play(FadeIn(foundation_group, shift=UP * 0.1))
        self.next_slide()

        statement1 = note_text(
            "The future of AI may not belong to one type of computation.", size=18
        )
        statement1.move_to(DOWN * 2.2)
        self.play(FadeIn(statement1, shift=UP * 0.15))
        self.next_slide()
        statement2 = note_text(
            "It may belong to choosing the right one for the right problem.", size=18
        )
        statement2.move_to(DOWN * 2.7)
        self.play(FadeIn(statement2, shift=UP * 0.15))
        self.next_slide()


# transportation overlaping
