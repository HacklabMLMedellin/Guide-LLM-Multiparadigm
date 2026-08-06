from manim import *
from manim_slides import Slide


DARK_BG = "#0D1117"
CARD_BG = "#161B22"
ACCENT = "#58A6FF"
GREEN = "#3FB950"
ORANGE = "#F0883E"
PURPLE = "#BC8CFF"
RED = "#FF7B72"
MUTED = "#8B949E"
WHITE = "#E6EDF3"

TOKEN_COLORS = [ACCENT, GREEN, ORANGE, PURPLE, RED, YELLOW, TEAL, MAROON]


def styled_text(text, size=36, color=WHITE, weight=NORMAL):
    return Text(text, font="JetBrains Mono", font_size=size, color=color, weight=weight)


def token_box(char, idx=0, width=0.65, height=0.65):
    color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
    box = RoundedRectangle(
        corner_radius=0.08,
        width=width,
        height=height,
        fill_color=color,
        fill_opacity=0.18,
        stroke_color=color,
        stroke_width=2,
    )
    label = Text(char, font="JetBrains Mono", font_size=22, color=color, weight=BOLD)
    label.move_to(box.get_center())
    return VGroup(box, label)


def id_box(num, idx=0, width=0.65, height=0.65):
    color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
    box = RoundedRectangle(
        corner_radius=0.08,
        width=width,
        height=height,
        fill_color=CARD_BG,
        fill_opacity=1,
        stroke_color=color,
        stroke_width=1.5,
    )
    label = Text(str(num), font="JetBrains Mono", font_size=19, color=color)
    label.move_to(box.get_center())
    return VGroup(box, label)


class NanoGPTPresentation(Slide):
    def construct(self):
        self.camera.background_color = ManimColor(DARK_BG)
        self.slide_title_screen()
        self.slide_dataset()
        self.slide_tokenization()
        self.slide_encoding()
        self.slide_batching()
        self.slide_bigram_model()
        self.slide_forward_pass()
        self.slide_generate()

    def _section_header(self, title, subtitle=None):
        t = styled_text(title, size=40, color=ACCENT, weight=BOLD)
        t.to_edge(UP, buff=0.4)
        line = Line(
            LEFT * 6.5,
            RIGHT * 6.5,
            stroke_color=ACCENT,
            stroke_width=1.2,
            stroke_opacity=0.35,
        )
        line.next_to(t, DOWN, buff=0.18)
        group = VGroup(t, line)
        if subtitle:
            s = styled_text(subtitle, size=20, color=MUTED)
            s.next_to(line, DOWN, buff=0.22)
            group.add(s)
        return group

    def _clear(self, *mobjects, run_time=0.35):
        if mobjects:
            self.play(*[FadeOut(m) for m in mobjects], run_time=run_time)
        else:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)

    def slide_title_screen(self):
        bg_rect = Rectangle(
            width=16, height=9, fill_color=DARK_BG, fill_opacity=1, stroke_width=0
        )

        title = styled_text("nanoGPT", size=88, color=WHITE, weight=BOLD)
        sub = styled_text(
            "How a character-level language model predicts text", size=26, color=MUTED
        )
        sub.next_to(title, DOWN, buff=0.45)

        accent_line = Line(LEFT * 2.2, RIGHT * 2.2, stroke_color=ACCENT, stroke_width=3)
        accent_line.next_to(sub, DOWN, buff=0.55)

        pills = VGroup()
        for label, color in [
            ("Bigram Model", ACCENT),
            ("PyTorch", GREEN),
            ("Shakespeare", ORANGE),
        ]:
            pill_bg = RoundedRectangle(
                corner_radius=0.22,
                width=2.4,
                height=0.5,
                fill_color=color,
                fill_opacity=0.15,
                stroke_color=color,
                stroke_width=1.5,
            )
            pill_text = styled_text(label, size=17, color=color)
            pill_text.move_to(pill_bg)
            pills.add(VGroup(pill_bg, pill_text))
        pills.arrange(RIGHT, buff=0.45)
        pills.next_to(accent_line, DOWN, buff=0.55)

        group = VGroup(title, sub, accent_line, pills)
        group.move_to(ORIGIN)

        self.play(FadeIn(title, shift=UP * 0.3), run_time=0.8)
        self.play(FadeIn(sub), FadeIn(accent_line), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(p, shift=UP * 0.15) for p in pills], lag_ratio=0.18),
            run_time=0.6,
        )
        self.next_slide()
        self._clear(title, sub, accent_line, pills)

    def slide_dataset(self):
        header = self._section_header(
            "The Dataset", "1,115,394 characters of Shakespeare"
        )
        self.play(FadeIn(header), run_time=0.5)

        excerpt_lines = [
            "First Citizen:",
            "Before we proceed any further, hear me speak.",
            "",
            "All:",
            "Speak, speak.",
            "",
            "First Citizen:",
            "You are all resolved rather to die than to famish?",
        ]

        card = RoundedRectangle(
            corner_radius=0.2,
            width=11,
            height=4.5,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=ACCENT,
            stroke_width=1.2,
            stroke_opacity=0.4,
        )
        card.next_to(header, DOWN, buff=0.55)

        text_group = VGroup()
        for i, line in enumerate(excerpt_lines):
            if not line:
                continue
            color = ORANGE if line.endswith(":") else WHITE
            size = 18 if line.endswith(":") else 16
            t = styled_text(line, size=size, color=color)
            text_group.add(t)
        text_group.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        text_group.move_to(card.get_center()).shift(LEFT * 0.3)

        self.play(FadeIn(card), run_time=0.4)
        self.play(
            LaggedStart(
                *[FadeIn(t, shift=RIGHT * 0.1) for t in text_group], lag_ratio=0.06
            ),
            run_time=1.0,
        )

        stat_row = VGroup()
        for val, label, color in [
            ("1.1M", "characters", ACCENT),
            ("65", "unique chars", GREEN),
            ("90/10", "train/val split", ORANGE),
        ]:
            stat_card = RoundedRectangle(
                corner_radius=0.15,
                width=2.8,
                height=1.1,
                fill_color=CARD_BG,
                fill_opacity=1,
                stroke_color=color,
                stroke_width=1.5,
            )
            v_text = styled_text(val, size=30, color=color, weight=BOLD)
            l_text = styled_text(label, size=14, color=MUTED)
            v_text.move_to(stat_card).shift(UP * 0.17)
            l_text.move_to(stat_card).shift(DOWN * 0.27)
            stat_row.add(VGroup(stat_card, v_text, l_text))
        stat_row.arrange(RIGHT, buff=0.4)
        stat_row.next_to(card, DOWN, buff=0.38)

        self.play(
            LaggedStart(*[FadeIn(s, shift=UP * 0.15) for s in stat_row], lag_ratio=0.2),
            run_time=0.7,
        )
        self.next_slide()
        self._clear(header, card, text_group, stat_row)

    def slide_tokenization(self):
        header = self._section_header(
            "Tokenization", "Characters → Integer IDs via a lookup table"
        )
        self.play(FadeIn(header), run_time=0.5)

        vocab_label = styled_text(
            "Vocabulary  (65 unique characters)", size=18, color=MUTED
        )
        vocab_label.next_to(header, DOWN, buff=0.5)

        sample_chars = list(
            " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )[:30]
        vocab_row = VGroup()
        for i, ch in enumerate(sample_chars):
            display = "·" if ch == " " else ch
            b = token_box(display, idx=i % len(TOKEN_COLORS), width=0.52, height=0.52)
            vocab_row.add(b)
        vocab_row.arrange(RIGHT, buff=0.06)
        vocab_row.next_to(vocab_label, DOWN, buff=0.3)
        if vocab_row.width > 13:
            vocab_row.scale(13 / vocab_row.width)

        self.play(FadeIn(vocab_label), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.7) for b in vocab_row], lag_ratio=0.04),
            run_time=1.0,
        )
        self.next_slide()

        example_word = "hello"
        example_ids = [46, 43, 50, 50, 53]

        arrow_label = styled_text('encode("hello")', size=22, color=GREEN)
        arrow_label.next_to(vocab_row, DOWN, buff=0.55)

        char_row = VGroup(*[token_box(c, idx=i) for i, c in enumerate(example_word)])
        char_row.arrange(RIGHT, buff=0.12)
        char_row.next_to(arrow_label, DOWN, buff=0.4)

        arrow = Arrow(
            char_row.get_bottom(),
            char_row.get_bottom() + DOWN * 0.9,
            stroke_color=ACCENT,
            stroke_width=2.5,
            tip_length=0.2,
        )

        id_row = VGroup(*[id_box(n, idx=i) for i, n in enumerate(example_ids)])
        id_row.arrange(RIGHT, buff=0.12)
        id_row.next_to(arrow, DOWN, buff=0.12)

        self.play(FadeIn(arrow_label, shift=DOWN * 0.1), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(b) for b in char_row], lag_ratio=0.12), run_time=0.6
        )
        self.play(GrowArrow(arrow), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in id_row], lag_ratio=0.12),
            run_time=0.6,
        )

        decode_label = styled_text(
            '[46, 43, 50, 50, 53]  →  "hello"', size=18, color=ORANGE
        )
        decode_label.next_to(id_row, DOWN, buff=0.45)
        self.play(FadeIn(decode_label), run_time=0.4)

        self.next_slide()
        self._clear(
            header,
            vocab_label,
            vocab_row,
            arrow_label,
            char_row,
            arrow,
            id_row,
            decode_label,
        )

    def slide_encoding(self):
        header = self._section_header(
            "Encoding the Full Text", "text  →  torch.Tensor  (dtype=int64)"
        )
        self.play(FadeIn(header), run_time=0.5)

        pipeline_steps = [
            ("input.txt", ACCENT, "📄"),
            ("encode(text)", GREEN, "⚙"),
            ("torch.Tensor", ORANGE, "📦"),
        ]

        step_boxes = VGroup()
        for label, color, icon in pipeline_steps:
            rect = RoundedRectangle(
                corner_radius=0.18,
                width=3.1,
                height=1.1,
                fill_color=CARD_BG,
                fill_opacity=1,
                stroke_color=color,
                stroke_width=1.8,
            )
            t1 = styled_text(icon + "  " + label, size=20, color=color)
            t1.move_to(rect)
            step_boxes.add(VGroup(rect, t1))

        arrows = VGroup()
        step_boxes.arrange(RIGHT, buff=1.3)
        step_boxes.next_to(header, DOWN, buff=0.7)

        for i in range(len(step_boxes) - 1):
            a = Arrow(
                step_boxes[i].get_right(),
                step_boxes[i + 1].get_left(),
                stroke_color=MUTED,
                stroke_width=2,
                tip_length=0.2,
                buff=0.08,
            )
            arrows.add(a)

        self.play(
            LaggedStart(
                *[FadeIn(s, shift=RIGHT * 0.2) for s in step_boxes], lag_ratio=0.25
            ),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3), run_time=0.5
        )
        self.next_slide()

        tensor_card = RoundedRectangle(
            corner_radius=0.18,
            width=12,
            height=1.8,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=ORANGE,
            stroke_width=1.2,
            stroke_opacity=0.5,
        )
        tensor_card.next_to(step_boxes, DOWN, buff=0.7)

        shape_text = styled_text(
            "torch.Size([1,115,394])   dtype=torch.int64", size=19, color=ORANGE
        )
        shape_text.move_to(tensor_card).shift(UP * 0.3)

        sample_ids = [18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52]
        id_display = styled_text(
            "tensor([" + ", ".join(str(x) for x in sample_ids) + ", ...])",
            size=15,
            color=MUTED,
        )
        id_display.move_to(tensor_card).shift(DOWN * 0.3)

        self.play(
            FadeIn(tensor_card), FadeIn(shape_text), FadeIn(id_display), run_time=0.6
        )

        split_label = styled_text(
            "train  =  data[:1,003,854]       val  =  data[1,003,854:]",
            size=18,
            color=GREEN,
        )
        split_label.next_to(tensor_card, DOWN, buff=0.5)

        split_bar = Rectangle(
            width=11,
            height=0.28,
            fill_color=GREEN,
            fill_opacity=0.25,
            stroke_color=GREEN,
            stroke_width=1.2,
        )
        split_bar.next_to(split_label, DOWN, buff=0.22)
        val_bar = Rectangle(
            width=11 * 0.1,
            height=0.28,
            fill_color=RED,
            fill_opacity=0.35,
            stroke_color=RED,
            stroke_width=1.2,
        )
        val_bar.align_to(split_bar, RIGHT)
        val_bar.align_to(split_bar, UP)

        self.play(FadeIn(split_label), FadeIn(split_bar), FadeIn(val_bar), run_time=0.6)
        self.next_slide()
        self._clear(
            header,
            step_boxes,
            arrows,
            tensor_card,
            shape_text,
            id_display,
            split_label,
            split_bar,
            val_bar,
        )

    def slide_batching(self):
        header = self._section_header(
            "Batching & Context Windows", "block_size=8   batch_size=4"
        )
        self.play(FadeIn(header), run_time=0.5)

        context_label = styled_text(
            "Each sample trains on ALL sub-sequences up to block_size",
            size=18,
            color=MUTED,
        )
        context_label.next_to(header, DOWN, buff=0.5)
        self.play(FadeIn(context_label), run_time=0.35)

        example_tokens = [18, 47, 56, 57, 58, 1, 15, 47]
        example_chars = list("First Ci")

        token_row = VGroup()
        for i, (ch, tid) in enumerate(zip(example_chars, example_tokens)):
            disp = "·" if ch == " " else ch
            b = token_box(disp, idx=i)
            n = styled_text(str(tid), size=12, color=MUTED)
            n.next_to(b, DOWN, buff=0.08)
            token_row.add(VGroup(b, n))
        token_row.arrange(RIGHT, buff=0.1)
        token_row.next_to(context_label, DOWN, buff=0.55)

        self.play(
            LaggedStart(*[FadeIn(t) for t in token_row], lag_ratio=0.09), run_time=0.8
        )
        self.next_slide()

        sub_rows = VGroup()
        for length in range(1, 5):
            highlight = VGroup()
            for i in range(length):
                hl = token_row[i][0][0].copy()
                hl.set_fill(opacity=0.55)
                highlight.add(hl)

            arrow = Arrow(
                RIGHT * 0.05,
                RIGHT * 0.6,
                stroke_color=ACCENT,
                stroke_width=1.8,
                tip_length=0.16,
                buff=0,
            )

            target_box = token_box(
                example_chars[length] if example_chars[length] != " " else "·",
                idx=length,
            )
            row = VGroup(highlight.copy(), arrow, target_box)
            row.arrange(RIGHT, buff=0.15)
            sub_rows.add(row)

        sub_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        sub_rows.next_to(token_row, DOWN, buff=0.55)
        sub_rows.shift(LEFT * 1.5)

        input_lbl = styled_text("input", size=15, color=MUTED)
        target_lbl = styled_text("target", size=15, color=MUTED)
        input_lbl.next_to(sub_rows, UP, buff=0.18).align_to(sub_rows, LEFT)

        self.play(FadeIn(input_lbl), run_time=0.3)
        self.play(
            LaggedStart(
                *[FadeIn(r, shift=RIGHT * 0.1) for r in sub_rows], lag_ratio=0.15
            ),
            run_time=1.0,
        )

        batch_label = styled_text(
            "→  Stack 4 random windows  →  xb shape: (4, 8)", size=18, color=GREEN
        )
        batch_label.next_to(sub_rows, DOWN, buff=0.55)
        self.play(FadeIn(batch_label), run_time=0.4)

        self.next_slide()
        self._clear(header, context_label, token_row, sub_rows, input_lbl, batch_label)

    def slide_bigram_model(self):
        header = self._section_header(
            "BigramLanguageModel", "nn.Embedding(vocab_size, vocab_size)"
        )
        self.play(FadeIn(header), run_time=0.5)

        class_box = RoundedRectangle(
            corner_radius=0.2,
            width=11,
            height=5.2,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=PURPLE,
            stroke_width=1.8,
        )
        class_box.next_to(header, DOWN, buff=0.5)

        class_title = styled_text(
            "class  BigramLanguageModel(nn.Module):", size=20, color=PURPLE, weight=BOLD
        )
        class_title.move_to(class_box.get_top()).shift(DOWN * 0.4)

        methods = [
            ("__init__", "token_embedding_table  =  nn.Embedding(65, 65)", ACCENT),
            (
                "forward",
                "logits  =  embedding(idx)    →    loss  =  cross_entropy(logits, targets)",
                GREEN,
            ),
            ("generate", "logits[-1]  →  softmax  →  multinomial  →  append", ORANGE),
        ]

        method_group = VGroup()
        for name, desc, color in methods:
            name_t = styled_text(
                "def  " + name + "(self, ...):", size=17, color=color, weight=BOLD
            )
            desc_t = styled_text("    " + desc, size=14, color=MUTED)
            block = VGroup(name_t, desc_t)
            block.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            sep = Line(
                LEFT * 4.5,
                RIGHT * 4.5,
                stroke_color=color,
                stroke_width=0.6,
                stroke_opacity=0.25,
            )
            method_group.add(VGroup(block, sep))

        method_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        method_group.move_to(class_box).shift(DOWN * 0.25 + LEFT * 0.2)

        self.play(FadeIn(class_box), FadeIn(class_title), run_time=0.5)
        self.play(
            LaggedStart(
                *[FadeIn(m, shift=DOWN * 0.08) for m in method_group], lag_ratio=0.22
            ),
            run_time=1.0,
        )
        self.next_slide()
        self._clear(header, class_box, class_title, method_group)

    def slide_forward_pass(self):
        header = self._section_header("Forward Pass", "idx  →  logits  →  loss")
        self.play(FadeIn(header), run_time=0.5)

        shapes = [
            ("idx", "(B, T)", "(4, 8)", ACCENT, "input token IDs"),
            ("embedding", "(B, T, C)", "(4, 8, 65)", GREEN, "looked-up vectors"),
            ("logits\n(reshaped)", "(B·T, C)", "(32, 65)", ORANGE, "scores per token"),
            ("loss", "scalar", "4.87", RED, "cross-entropy"),
        ]

        boxes = VGroup()
        for name, shape_label, example, color, desc in shapes:
            rect = RoundedRectangle(
                corner_radius=0.15,
                width=2.5,
                height=2.0,
                fill_color=CARD_BG,
                fill_opacity=1,
                stroke_color=color,
                stroke_width=1.8,
            )
            name_t = styled_text(name, size=18, color=color, weight=BOLD)
            shape_t = styled_text(shape_label, size=15, color=WHITE)
            ex_t = styled_text(example, size=13, color=MUTED)
            desc_t = styled_text(desc, size=11, color=MUTED)
            name_t.move_to(rect).shift(UP * 0.6)
            shape_t.move_to(rect).shift(UP * 0.15)
            ex_t.move_to(rect).shift(DOWN * 0.25)
            desc_t.move_to(rect).shift(DOWN * 0.6)
            boxes.add(VGroup(rect, name_t, shape_t, ex_t, desc_t))

        boxes.arrange(RIGHT, buff=0.55)
        boxes.next_to(header, DOWN, buff=0.65)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_right(),
                boxes[i + 1].get_left(),
                stroke_color=MUTED,
                stroke_width=2,
                tip_length=0.18,
                buff=0.06,
            )
            arrows.add(a)

        self.play(
            LaggedStart(*[FadeIn(b, shift=DOWN * 0.12) for b in boxes], lag_ratio=0.2),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.25), run_time=0.6
        )
        self.next_slide()

        loss_note = styled_text(
            "Untrained model: loss ≈ 4.87  ≈  -ln(1/65)  ✓", size=20, color=GREEN
        )
        loss_note.next_to(boxes, DOWN, buff=0.65)

        ln_eq = styled_text(
            "-ln(1/65)  =  4.174   (expected random baseline)", size=16, color=MUTED
        )
        ln_eq.next_to(loss_note, DOWN, buff=0.25)

        self.play(FadeIn(loss_note, shift=UP * 0.1), FadeIn(ln_eq), run_time=0.5)
        self.next_slide()
        self._clear(header, boxes, arrows, loss_note, ln_eq)

    def slide_generate(self):
        header = self._section_header(
            "Generation Loop", "Autoregressive sampling from the model"
        )
        self.play(FadeIn(header), run_time=0.5)

        steps = [
            ("idx  =  [[0]]", ACCENT, "start token (B=1, T=1)"),
            ("logits  =  model(idx)", GREEN, "shape  →  (1, T, 65)"),
            ("logits  =  logits[:, -1, :]", ORANGE, "last timestep  →  (1, 65)"),
            ("probs  =  softmax(logits)", PURPLE, "probability distribution"),
            ("next  =  multinomial(probs)", RED, "sample 1 token"),
            ("idx  =  cat([idx, next], dim=1)", ACCENT, "append & repeat"),
        ]

        step_group = VGroup()
        for i, (code, color, desc) in enumerate(steps):
            num_circle = Circle(
                radius=0.22,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color,
                stroke_width=1.5,
            )
            num_t = styled_text(str(i + 1), size=15, color=color, weight=BOLD)
            num_t.move_to(num_circle)

            code_t = styled_text(code, size=16, color=WHITE)
            desc_t = styled_text(desc, size=13, color=MUTED)
            text_group = VGroup(code_t, desc_t)
            text_group.arrange(RIGHT, buff=0.3)

            row = VGroup(VGroup(num_circle, num_t), text_group)
            row.arrange(RIGHT, buff=0.35)
            step_group.add(row)

        step_group.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        step_group.next_to(header, DOWN, buff=0.5)
        step_group.shift(LEFT * 1.2)

        self.play(
            LaggedStart(
                *[FadeIn(s, shift=RIGHT * 0.12) for s in step_group], lag_ratio=0.14
            ),
            run_time=1.2,
        )
        self.next_slide()

        output_card = RoundedRectangle(
            corner_radius=0.18,
            width=11,
            height=1.4,
            fill_color=CARD_BG,
            fill_opacity=1,
            stroke_color=MUTED,
            stroke_width=1,
            stroke_opacity=0.4,
        )
        output_card.next_to(step_group, DOWN, buff=0.5)

        before_label = styled_text("Before training:", size=14, color=MUTED)
        before_label.move_to(output_card).shift(UP * 0.28 + LEFT * 2.8)
        sample_output = styled_text(
            "SKIcLT;AcELMoTbvZv C?nq-QE33:CJqk...", size=15, color=RED
        )
        sample_output.move_to(output_card).shift(DOWN * 0.2)

        after_label = styled_text("→ After training:", size=14, color=GREEN)

        self.play(
            FadeIn(output_card),
            FadeIn(before_label),
            FadeIn(sample_output),
            run_time=0.5,
        )
        self.next_slide()

        final_card = RoundedRectangle(
            corner_radius=0.2,
            width=11.5,
            height=1.8,
            fill_color="#0D2818",
            fill_opacity=1,
            stroke_color=GREEN,
            stroke_width=1.8,
        )
        final_card.move_to(ORIGIN).shift(DOWN * 2.8)

        done_text = styled_text(
            "nanoGPT  ·  character prediction  ·  step by step",
            size=20,
            color=GREEN,
            weight=BOLD,
        )
        done_text.move_to(final_card)

        self.play(
            FadeOut(output_card),
            FadeOut(before_label),
            FadeOut(sample_output),
            FadeIn(final_card),
            FadeIn(done_text),
            run_time=0.6,
        )
        self.next_slide()
