from __future__ import annotations

from pathlib import Path
from textwrap import fill
from math import cos, pi, sin

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from app.services.market import SignalCandidate


class Renderer:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.chart_dir = storage_dir / "charts"
        self.news_dir = storage_dir / "news"
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        self.news_dir.mkdir(parents=True, exist_ok=True)

    def render_signal_chart(self, candidate: SignalCandidate) -> Path:
        df = candidate.candles.copy()
        output = self.chart_dir / f"{candidate.symbol.lower()}_{candidate.side}.png"

        fig = plt.figure(figsize=(16, 9), facecolor="#f2eee7")
        gs = GridSpec(5, 1, figure=fig, hspace=0.0)
        ax = fig.add_subplot(gs[:4, 0], facecolor="#f2eee7")
        axv = fig.add_subplot(gs[4, 0], sharex=ax, facecolor="#f2eee7")

        self._draw_candles(ax, axv, df)
        self._draw_signal_levels(ax, candidate, df)

        ax.set_xlim(-1, len(df) + 4)
        ax.grid(True, color="#d8d0c7", alpha=0.6, linewidth=0.8)
        ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", colors="#5c5148")
        ax.spines[:].set_visible(False)

        axv.grid(False)
        axv.tick_params(axis="x", colors="#5c5148")
        axv.tick_params(axis="y", colors="#5c5148")
        axv.spines[:].set_visible(False)

        fig.savefig(output, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return output

    def _draw_candles(self, ax, axv, df: pd.DataFrame) -> None:
        candle_width = 0.55
        volume_colors = []
        for idx, row in df.iterrows():
            up = row["close"] >= row["open"]
            color = "#13a690" if up else "#ef476f"
            wick = ax.vlines(idx, row["low"], row["high"], color=color, linewidth=1.3, alpha=0.9)
            wick.set_capstyle("round")
            body_bottom = min(row["open"], row["close"])
            body_height = max(abs(row["close"] - row["open"]), 0.0001)
            ax.add_patch(
                Rectangle(
                    (idx - candle_width / 2, body_bottom),
                    candle_width,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.0,
                )
            )
            volume_colors.append(color)

        axv.bar(range(len(df)), df["volume"], color=volume_colors, width=0.65, alpha=0.35)

    def _draw_signal_levels(self, ax, candidate: SignalCandidate, df: pd.DataFrame) -> None:
        levels = [
            ("Вход", candidate.entry, "#f2c94c"),
            ("Стоп", candidate.stop, "#eb5757"),
            ("Тейк 1", candidate.take1, "#27ae60"),
            ("Тейк 2", candidate.take2, "#219653"),
        ]
        x_label = len(df) + 2.2
        for name, price, color in levels:
            ax.axhline(price, color=color, linestyle=(0, (2, 2)), linewidth=1.5, alpha=0.85)
            ax.text(
                x_label,
                price,
                f"{name}  {price:.4f}",
                va="center",
                ha="left",
                color="white",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none"),
            )

    def render_news_card(self, title: str, items: list[dict[str, str]]) -> Path:
        output = self.news_dir / "daily_news.png"
        width, height = 900, 980
        image = Image.new("RGB", (width, height), "#102033")
        draw = ImageDraw.Draw(image)

        font_tabs = self._load_font(28, bold=True)
        font_time = self._load_font(21, bold=False)
        font_ccy = self._load_font(28, bold=True)
        font_body = self._load_font(26, bold=True)
        font_meta = self._load_font(21, bold=False)
        font_footer = self._load_font(26, bold=True)

        outer_margin = 26
        white_bottom = height - 152
        draw.rounded_rectangle(
            [outer_margin, outer_margin, width - outer_margin, height - outer_margin],
            radius=28,
            fill="#132235",
        )
        draw.rounded_rectangle(
            [outer_margin, outer_margin, width - outer_margin, white_bottom],
            radius=28,
            fill="#ffffff",
        )
        draw.rectangle([outer_margin, white_bottom - 24, width - outer_margin, white_bottom], fill="#ffffff")

        tabs = [("Вчера", "#8b8b8b"), ("Сегодня", "#111111"), ("Завтра", "#8b8b8b")]
        tab_x = [78, 298, 520]
        for index, (text, color) in enumerate(tabs):
            x = tab_x[index]
            draw.text((x, 58), text, fill=color, font=font_tabs)
            if text == "Сегодня":
                draw.line((x + 2, 112, x + 148, 112), fill="#111111", width=4)

        draw.line((44, 138, width - 44, 138), fill="#ececec", width=2)

        section_top = 150
        section_height = 160
        left_x = 54
        divider_x = 190
        content_x = 226

        for index, item in enumerate(items[:4]):
            top = section_top + index * section_height
            bottom = top + section_height
            if index:
                draw.line((44, top, width - 44, top), fill="#ececec", width=2)

            draw.text((left_x, top + 26), item.get("time", "15:30"), fill="#7c7c7c", font=font_time)
            draw.text((left_x + 58, top + 18), item.get("tag", "USD"), fill="#111111", font=font_ccy)
            self._draw_stars(draw, left_x + 4, top + 72)
            self._draw_us_flag(draw, left_x + 66, top + 78)
            draw.line((divider_x, top + 18, divider_x, bottom - 20), fill="#e8e8e8", width=2)

            headline = fill(item["headline"], width=30)
            draw.multiline_text((content_x, top + 18), headline, fill="#111111", font=font_body, spacing=4)
            meta = item.get("meta", "")
            if meta:
                draw.text((content_x, top + 94), meta, fill="#8b8b8b", font=font_meta)

        footer_top = height - 152
        draw.rounded_rectangle(
            [outer_margin, footer_top, width - outer_margin, height - outer_margin],
            radius=28,
            fill="#132235",
        )
        draw.text((52, footer_top + 30), title, fill="#ffffff", font=font_footer)
        draw.text((52, footer_top + 74), "Подборка для канала на сегодня", fill="#8fb3ff", font=font_meta)

        image.save(output)
        return output

    def _draw_stars(self, draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
        for index in range(3):
            self._draw_star_shape(draw, x + index * 16 + 6, y + 7, 5, "#9ea6b1")

    def _draw_us_flag(self, draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
        width, height = 22, 15
        stripe_h = max(height // 7, 1)
        for index in range(7):
            color = "#c93a3a" if index % 2 == 0 else "#ffffff"
            draw.rectangle([x, y + index * stripe_h, x + width, y + (index + 1) * stripe_h], fill=color)
        draw.rectangle([x, y, x + 9, y + 7], fill="#3157a5")
        draw.rounded_rectangle([x, y, x + width, y + height], radius=2, outline="#d9d9d9", width=1)

    def _draw_star_shape(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int, color: str) -> None:
        points: list[tuple[float, float]] = []
        inner = radius * 0.45
        for index in range(10):
            angle = -pi / 2 + index * pi / 5
            r = radius if index % 2 == 0 else inner
            points.append((cx + cos(angle) * r, cy + sin(angle) * r))
        draw.polygon(points, fill=color)

    def _load_font(self, size: int, *, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        return ImageFont.load_default()
